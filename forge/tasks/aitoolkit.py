"""ai-toolkit LoRA training (flux / krea2 / ideogram4).

We wrap ai-toolkit's ``run.py <config.yaml>`` (the validator's proven trainer)
rather than reinventing a diffusion loop — our edge lives in the config (see
forge/config.py + forge/recipe.py) and the orchestration here.

Pacing / kill-safety (INV-2): ai-toolkit trains to a fixed step count and can't be
stopped by a Python callback mid-run, so we launch it as a subprocess and
TERMINATE it when the wall-clock reserve is hit. The kill-safe ``save_every``
guarantees a periodic ``{repo}_{step:09d}.safetensors`` on disk, and
``_finalize`` promotes the final/highest-step LoRA to ``{save_root}/last.safetensors``
— the filename the evaluator matches FIRST (immune to the repo-name-ends-in-digit
step-misparse trap). This is the #1 zero-score guard.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

from forge import telemetry
from forge.clock import Deadline
from forge.config import build_config, resolve_base_model, write_config
from forge.tasks.integrity import valid_safetensors
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
_STOP_MARGIN_S = 45


def run(spec: ImageSpec, deadline: Deadline) -> None:
    os.makedirs(spec.save_root, exist_ok=True)
    os.makedirs(spec.training_folder, exist_ok=True)

    base_model = resolve_base_model(spec.cached_model_dir)
    images_dir, pairs = dataset.prepare_aitoolkit_dataset(
        spec.cached_zip_path,
        images_dir=spec.dataset_images_dir,
        trigger_word=spec.trigger_word,
    )
    telemetry.collect_env()
    telemetry.set_meta(
        model_type=spec.model_type,
        pairs=pairs,
        base_model=os.path.basename(base_model),
        trigger_word=spec.trigger_word,
    )
    telemetry.event("dataset_ready", pairs=pairs)

    hours = deadline.remaining_hard() / 3600.0
    cfg = build_config(spec, num_images=pairs, hours_to_complete=hours)
    p = cfg["config"]["process"][0]
    steps = p["train"]["steps"]
    telemetry.set_meta(
        steps=steps, num_images=pairs, save_every=p["save"]["save_every"]
    )
    write_config(cfg, spec.config_path)

    _run_toolkit(spec.config_path, deadline, spec)
    _finalize(spec)


def _run_toolkit(cfg_path: str, deadline: Deadline, spec: ImageSpec) -> None:
    # Run ai-toolkit's run.py with the CURRENT interpreter, not a bare `python3`:
    # in the validator's Docker image sys.executable IS the env python with torch,
    # and in a local venv test it's the venv python — either way it has ai-toolkit's
    # deps, whereas a bare `python3` may resolve to a torch-less system python.
    cmd = [sys.executable, "run.py", cfg_path]
    log_path = os.path.join(spec.training_folder, "aitoolkit.log")
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
            if deadline.remaining() <= _STOP_MARGIN_S:
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
    if rc not in (0, None) and not stopped_by_deadline and not _has_lora(spec):
        _tail_log(log_path)
        raise RuntimeError(f"ai-toolkit failed (rc={rc}) with no checkpoint")


def _parse_toolkit_log(log_path: str):
    """ai-toolkit prints per-step 'loss: 0.0xxx' style lines. Best-effort — the
    exact token format isn't pinned from source; telemetry gaps are acceptable.
    """
    loss = step = None
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        losses = re.findall(r"loss[=:]\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
        if losses:
            loss = float(losses[-1])
        steps = re.findall(r"(\d+)\s*/\s*\d+", text)  # progress-bar cur/total
        if steps:
            step = int(steps[-1])
    except Exception:
        pass
    return loss, step


def _finalize(spec: ImageSpec) -> None:
    """Guarantee {save_root}/last.safetensors from the final/highest-step LoRA.

    Raises ONLY when zero LoRA exists → the CLI fallback then accepts -1.
    """
    loras = _loras(spec)
    if not loras:
        raise RuntimeError("ai-toolkit produced no LoRA safetensors")
    last = os.path.join(spec.save_root, "last.safetensors")
    if not os.path.isfile(last):
        _atomic_copy(_pick_final(loras, spec.expected_repo_name), last)
    telemetry.event("checkpoint_finalized", files=len(loras))
    telemetry.note_peak_memory()
    telemetry.write_into(spec.output_dir)


def _loras(spec: ImageSpec) -> list[str]:
    if not os.path.isdir(spec.save_root):
        return []
    return [
        p
        for p in glob.glob(os.path.join(spec.save_root, "*.safetensors"))
        if os.path.basename(p) != "last.safetensors"
    ]


def _pick_final(loras: list[str], repo: str) -> str:
    """Prefer the exact {repo}.safetensors (final unconditioned save). Else the
    highest {repo}_{step:09d}. Matching the KNOWN repo name disambiguates the
    digit-trap safely (repo name ending in a digit can't misparse as a step).

    Candidates are integrity-checked: a deadline kill mid-save leaves the
    NEWEST file truncated, and promoting it would zero-score a task with an
    intact older checkpoint next to it — step down to the newest valid one.
    """
    final = os.path.join(os.path.dirname(loras[0]), f"{repo}.safetensors")
    for p in loras:
        if p == final and valid_safetensors(p):
            return p

    def step_of(p: str) -> int:
        m = re.search(r"_(\d+)\.safetensors$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    for p in sorted(loras, key=step_of, reverse=True):
        if valid_safetensors(p):
            return p
    telemetry.event("no_valid_checkpoint", candidates=len(loras))
    return max(loras, key=step_of)


def _has_lora(spec: ImageSpec) -> bool:
    return bool(_loras(spec))


def _atomic_copy(src: str, dst: str) -> None:
    """Copy to a temp file on the SAME dir/fs, then os.replace onto dst.

    last.safetensors is the evaluator's FIRST-matched submission, so a mid-copy
    kill onto the final path would leave a TRUNCATED file that is loaded in
    preference to the intact periodic checkpoints → zero score. os.replace is
    atomic within save_root, so a kill leaves either the old file or the complete
    one, never a partial. Mirrors telemetry.write_into's tmp+replace.
    """
    tmp = dst + ".tmp"
    try:
        shutil.copy(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise


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
