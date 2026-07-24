"""ai-toolkit LoRA training for all five validator-supported image types.

We wrap ai-toolkit's ``run.py <config.yaml>`` (the validator's proven trainer)
rather than reinventing a diffusion loop — our edge lives in the config (see
forge/config.py + forge/recipe.py) and the orchestration here.

Pacing / kill-safety (INV-2): ai-toolkit trains to a fixed step count and can't be
stopped by a Python callback mid-run, so we launch it as a subprocess and
TERMINATE it when the wall-clock reserve is hit. The fixed-candidate
``save_every`` policy leaves periodic saves on disk. Finalization considers only
files created or replaced by this attempt, applies the conservative selection
policy in ``forge.tasks.checkpoints``, and atomically promotes the result to
``{save_root}/last.safetensors`` — the filename the evaluator matches first.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time

from forge import recipe, telemetry
from forge.clock import Deadline
from forge.config import build_config, resolve_base_model, write_config
from forge.tasks import checkpoints, holdout
from forge.data import dataset
from forge.data.schema import ImageSpec

# ai-toolkit checkout inside diagonalge/ai-toolkit:latest. Overridable for local
# tests (like SDXL used SD_SCRIPTS_DIR); run.py lives here and is the cwd.
_AI_TOOLKIT_DIR = os.environ.get("AI_TOOLKIT_DIR", "/app/ai-toolkit")
_POLL_SECONDS = 5
# Extra cushion ON TOP OF the export reserve. We gate termination on
# deadline.remaining() (the SOFT stop = hard_stop - export_reserve), so training
# is stopped ~(reserve + _STOP_MARGIN_S) before the hard kill. That preserves the
# full 180s export reserve for _terminate + _finalize (promote-to-last), instead
# of squeezing finalize into a hand-picked 45s window off the hard stop.
_STOP_MARGIN_S = holdout.boundary_margin_s()


def run(spec: ImageSpec, deadline: Deadline) -> None:
    os.makedirs(spec.save_root, exist_ok=True)
    os.makedirs(spec.training_folder, exist_ok=True)
    # Snapshot before ANY fallible preparation.  If dataset/model preparation
    # fails, the CLI fallback can distinguish an intact previous-run artifact
    # from output produced by this attempt.
    # The CLI scopes before dispatch so import/preparation failures are covered;
    # direct handler calls create the same durable scope here.  Reusing it is
    # essential: beginning twice would lose the first quarantine journal.
    scope = checkpoints.ensure_run(spec.save_root, spec.expected_repo_name)

    base_model = resolve_base_model(spec.cached_model_dir)
    images_dir, total_pairs = dataset.prepare_aitoolkit_dataset(
        spec.cached_zip_path,
        images_dir=spec.dataset_images_dir,
        trigger_word=spec.trigger_word,
    )
    holdout_feature_ready = holdout.budget_allows(
        spec.model_type,
        deadline.remaining(),
    )
    if holdout.enabled_for(spec.model_type) and not holdout_feature_ready:
        telemetry.event(
            "holdout_scoring_skipped",
            reason="insufficient_total_budget_before_split",
            remaining_s=round(deadline.remaining(), 1),
        )
    holdout_pairs = (
        dataset.reserve_holdout(
            images_dir,
            holdout_dir=spec.dataset_holdout_dir,
        )
        if holdout_feature_ready
        else 0
    )
    pairs = total_pairs - holdout_pairs
    telemetry.collect_env()
    telemetry.set_meta(
        model_type=spec.model_type,
        pairs=pairs,
        total_pairs=total_pairs,
        holdout_pairs=holdout_pairs,
        base_model=os.path.basename(base_model),
        trigger_word=spec.trigger_word,
    )
    telemetry.event(
        "dataset_ready",
        pairs=pairs,
        total_pairs=total_pairs,
        holdout_pairs=holdout_pairs,
    )

    scoring_reserve_s = (
        holdout.scoring_reserve_s(spec.model_type) if holdout_pairs > 0 else 0.0
    )
    # The recipe's wall-clock cap must not plan optimizer steps inside time
    # explicitly reserved for checkpoint scoring. It already accounts for the
    # ordinary export reserve itself.
    hours = _recipe_hours(deadline, scoring_reserve_s)
    cfg = build_config(spec, num_images=pairs, hours_to_complete=hours)
    p = cfg["config"]["process"][0]
    steps = p["train"]["steps"]
    scope = checkpoints.set_planned_steps(
        spec.save_root,
        scope,
        steps,
        model_type=spec.model_type,
    )
    telemetry.set_meta(
        steps=steps,
        num_images=pairs,
        save_every=p["save"]["save_every"],
        scoring_reserve_s=scoring_reserve_s,
    )
    write_config(cfg, spec.config_path)

    scoring_budget_ready = _run_toolkit(
        spec.config_path,
        deadline,
        spec,
        scope,
        scoring_reserve_s=scoring_reserve_s,
    )
    if scoring_budget_ready:
        holdout.produce(
            spec,
            cfg,
            scope,
            deadline,
            holdout_pairs=holdout_pairs,
        )
    elif holdout_pairs > 0:
        telemetry.event(
            "holdout_scoring_skipped",
            reason="scoring_budget_not_reserved",
        )
    _finalize(spec, scope)


def _recipe_hours(deadline: Deadline, scoring_reserve_s: float) -> float:
    """Return the recipe budget, preserving exact pre-Gate-B dormancy.

    The boundary margin belongs to the optional scorer reserve.  Charging that
    margin when no true holdout was reserved silently shortens ordinary runs,
    even though the feature is disabled.  Keep both deductions conditional on
    a positive scorer reserve so the unset/declined feature is byte-for-byte
    equivalent at config-planning time.
    """
    reserve_before_recipe_margin = 0.0
    if scoring_reserve_s > 0.0:
        reserve_before_recipe_margin = (
            (scoring_reserve_s + _STOP_MARGIN_S)
            / max(0.01, float(recipe.MARGIN))
        )
    return max(
        0.0,
        deadline.remaining_hard() - reserve_before_recipe_margin,
    ) / 3600.0


def _run_toolkit(
    cfg_path: str,
    deadline: Deadline,
    spec: ImageSpec,
    scope: dict,
    *,
    scoring_reserve_s: float = 0.0,
) -> bool:
    # Run ai-toolkit's run.py with the CURRENT interpreter, not a bare `python3`:
    # in the validator's Docker image sys.executable IS the env python with torch,
    # and in a local venv test it's the venv python — either way it has ai-toolkit's
    # deps, whereas a bare `python3` may resolve to a torch-less system python.
    cmd = [sys.executable, "run.py", cfg_path]
    # Logs are diagnostics, not evaluator artifacts. Keep them outside the
    # uploaded repo and make the name run-specific so concurrent/retried jobs
    # cannot truncate one another or leak another run's tail into telemetry.
    log_path = _toolkit_log_path(spec)
    telemetry.event("toolkit_start")
    started = time.monotonic()

    # GPU peak sampler: torch.max_memory_allocated is 0 in the parent (ai-toolkit
    # is a subprocess), so poll nvidia-smi in a daemon thread. Fully wrapped.
    gpu_stop = threading.Event()
    gpu_peak = {"mb": 0}

    def _sample_gpu():
        while not gpu_stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                vals = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
                if vals:
                    gpu_peak["mb"] = max(gpu_peak["mb"], max(vals))
            except Exception:
                pass
            gpu_stop.wait(5)

    try:
        gpu_thread = threading.Thread(target=_sample_gpu, daemon=True)
        gpu_thread.start()
    except Exception:
        gpu_thread = None

    stopped_by_deadline = False
    scoring_decision: bool | None = None
    with open(log_path, "w", encoding="utf-8") as log:
        # New session → we can signal the whole process GROUP, so ai-toolkit's
        # DataLoader workers can't outlive the kill holding GPU memory while
        # _finalize runs.
        proc = subprocess.Popen(
            cmd, cwd=_AI_TOOLKIT_DIR, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while proc.poll() is None:
            # remaining() already subtracts the 180s export reserve, so we begin
            # terminating ~(reserve + margin) before the hard kill — leaving the
            # whole reserve for _terminate + finalize rather than a 45s sliver.
            remaining = deadline.remaining()
            reserve = 0.0
            if (
                scoring_reserve_s > 0
                and scoring_decision is None
                and remaining <= _STOP_MARGIN_S + scoring_reserve_s
            ):
                # Decide once at the reserve boundary. Enabling the scorer later
                # would give it less time than the measured reserve and turn the
                # reserve into a predictable timeout.
                scoring_decision = _latch_scoring_decision(
                    scoring_decision,
                    remaining=remaining,
                    reserve_s=scoring_reserve_s,
                    candidates_ready=holdout.has_scoring_candidates(
                        spec.save_root, scope
                    ),
                )
                telemetry.event(
                    "holdout_scoring_budget_decided",
                    reserved=scoring_decision,
                    remaining_s=round(remaining, 1),
                )
            if scoring_decision is True:
                reserve = scoring_reserve_s
            if remaining <= _STOP_MARGIN_S + reserve:
                stopped_by_deadline = True
                _terminate(proc)
                break
            time.sleep(_POLL_SECONDS)
    rc = proc.returncode

    try:
        gpu_stop.set()
        if gpu_thread is not None:
            gpu_thread.join(timeout=6)
        if gpu_peak["mb"] > 0:
            telemetry.sample("gpu_peak_mb", gpu_peak["mb"])
    except Exception:
        pass

    telemetry.event(
        "toolkit_end", returncode=rc, stopped_by_deadline=stopped_by_deadline,
        elapsed_s=round(time.monotonic() - started, 1),
    )
    try:
        loss, step = _parse_toolkit_log(log_path)
        telemetry.event("toolkit_metrics", loss=loss, last_step=step)
        if loss is not None:
            telemetry.sample("final_loss", loss)
            telemetry.train_point(step or 0, loss, None)
    except Exception:
        pass

    # A clean exit (0) or a deadline stop are both success: a checkpoint should be
    # on disk. A nonzero exit we did NOT trigger means ai-toolkit failed — but if
    # it still wrote a LoRA we keep it; the CLI fallback covers the empty case.
    if (
        rc not in (0, None)
        and not stopped_by_deadline
        and not _has_current_lora(spec, scope)
    ):
        _tail_log(log_path)
        raise RuntimeError(f"ai-toolkit failed (rc={rc}) with no checkpoint")

    if scoring_reserve_s <= 0:
        return False
    if scoring_decision is not None:
        return scoring_decision
    # Natural completion before the reserve boundary may create the exact final
    # without ever entering the loop's boundary branch. Score only when the
    # full measured reserve still remains and at least two candidates exist.
    return (
        deadline.remaining() >= scoring_reserve_s + _STOP_MARGIN_S
        and holdout.has_scoring_candidates(spec.save_root, scope)
    )


def _latch_scoring_decision(
    decision: bool | None,
    *,
    remaining: float,
    reserve_s: float,
    candidates_ready: bool,
) -> bool | None:
    """Make the scorer-budget decision once; never enable it late."""
    if decision is not None:
        return decision
    if reserve_s > 0 and remaining <= _STOP_MARGIN_S + reserve_s:
        return bool(candidates_ready)
    return None


def _toolkit_log_path(spec: ImageSpec) -> str:
    log_dir = os.path.join(os.path.dirname(spec.config_path), "forge-logs")
    os.makedirs(log_dir, exist_ok=True)
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.task_id)
    safe_repo = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.expected_repo_name)
    return os.path.join(
        log_dir, f"{safe_task}-{safe_repo}-{os.getpid()}-{time.time_ns()}.log"
    )


def _parse_toolkit_log(log_path: str):
    """Read the last progress/loss pair from ai-toolkit's console log.

    Current Krea2 prints losses in scientific notation (for example
    ``2.531e-02``).  Keep exponent handling explicit: silently recording that as
    ``2.531`` makes the flight recorder wrong by two orders of magnitude.
    """
    loss = step = None
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        losses = re.findall(rf"loss[=:]\s*({number})", text, re.IGNORECASE)
        if losses:
            loss = float(losses[-1])
        progress = re.findall(r"(\d+)\s*/\s*(\d+)", text)
        if progress:
            current, total = (int(value) for value in progress[-1])
            step = current
            # ai-toolkit's tqdm counter is zero-based in the final visible loss
            # line (35/36 for the 36th update).  Its subsequent unnumbered
            # terminal save is the durable proof that all planned steps landed.
            saves = re.findall(
                r"Saved checkpoint to\s+([^\r\n]+\.safetensors)", text
            )
            if saves and not re.search(
                r"_\d{9}\.safetensors$", os.path.basename(saves[-1].strip())
            ):
                step = total
    except Exception:
        pass
    return loss, step


def _finalize(spec: ImageSpec, scope: dict | None = None) -> None:
    """Promote a selected current-run LoRA or explicitly retain a prior one."""
    record = checkpoints.finalize(
        spec.save_root,
        spec.expected_repo_name,
        scope,
        context="training",
    )
    if record is None:
        raise RuntimeError("ai-toolkit produced no valid current or prior LoRA")
    telemetry.event(
        "checkpoint_finalized",
        status=record["status"],
        source=record["source"],
        selected_step=record["selected_step"],
    )
    telemetry.note_peak_memory()


def _has_current_lora(spec: ImageSpec, scope: dict) -> bool:
    return bool(checkpoints.current_loras(spec.save_root, scope))


def _terminate(proc: subprocess.Popen) -> None:
    # ai-toolkit won't flush on SIGTERM (its save cadence is our kill-safety, not
    # a signal handler), so a long grace is wasted budget. Give it a few seconds
    # to unwind, then SIGKILL — keeping the export reserve for _finalize.
    # Signals go to the whole process group (start_new_session at spawn) so
    # DataLoader workers die with the parent instead of holding GPU memory.
    def _signal_group(sig: int) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            proc.send_signal(sig)

    _signal_group(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _signal_group(signal.SIGKILL)
        proc.wait(timeout=10)


def _tail_log(log_path: str, n: int = 15) -> None:
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as fh:
            tail = fh.readlines()[-n:]
        telemetry.event("toolkit_log_tail", tail="".join(tail)[-1500:])
    except Exception:
        pass
