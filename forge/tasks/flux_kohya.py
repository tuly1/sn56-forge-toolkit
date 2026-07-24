"""Deadline-safe Kohya backend for standalone FLUX checkpoints.

The validator routes FLUX through the legacy-named Dockerfile and may stage the
base as one monolithic ``.safetensors`` file.  Kohya understands that shape;
ai-toolkit expects a Diffusers directory.  Forge still owns the surrounding
run scope, telemetry, kill-safe checkpoint promotion, publication scrub, and
never-forfeit fallback.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time

from forge import flux_kohya_config, telemetry
from forge.clock import Deadline
from forge.data import dataset
from forge.data.schema import ImageSpec
from forge.tasks import checkpoints, holdout


_SD_SCRIPTS_DIR = os.environ.get("SD_SCRIPTS_DIR", "/app/sd-scripts")
_POLL_SECONDS = 5.0
_STOP_MARGIN_S = holdout.boundary_margin_s()


def run(spec: ImageSpec, deadline: Deadline) -> None:
    os.makedirs(spec.save_root, exist_ok=True)
    os.makedirs(spec.training_folder, exist_ok=True)
    scope = checkpoints.ensure_run(spec.save_root, spec.expected_repo_name)

    base_model = resolve_standalone_model_file(spec.cached_model_dir)
    train_data_dir, pairs = dataset.prepare_kohya_flux_dataset(
        spec.cached_zip_path,
        images_root=spec.dataset_images_dir,
        trigger_word=spec.trigger_word,
    )
    steps = flux_kohya_config.MAX_TRAIN_STEPS
    scope = checkpoints.set_planned_steps(
        spec.save_root,
        scope,
        steps,
        model_type=spec.model_type,
    )
    config_path = _config_path(spec)
    config = flux_kohya_config.build_config(
        base_model=base_model,
        train_data_dir=train_data_dir,
        output_dir=spec.save_root,
        output_name=spec.expected_repo_name,
        config_file=config_path,
        steps=steps,
    )
    flux_kohya_config.write_config(config, config_path)

    telemetry.collect_env()
    telemetry.set_meta(
        model_type=spec.model_type,
        backend="kohya",
        pairs=pairs,
        base_model=os.path.basename(base_model),
        steps=steps,
        save_every=config["save_every_n_steps"],
        trigger_word=spec.trigger_word,
    )
    telemetry.event("dataset_ready", pairs=pairs)

    _run_kohya(config_path, deadline, spec, scope)
    record = checkpoints.finalize(
        spec.save_root,
        spec.expected_repo_name,
        scope,
        context="flux_kohya_training",
    )
    if record is None:
        raise RuntimeError("Kohya produced no valid current or prior FLUX LoRA")
    telemetry.event(
        "checkpoint_finalized",
        status=record["status"],
        source=record["source"],
        selected_step=record["selected_step"],
    )
    telemetry.note_peak_memory()


def resolve_standalone_model_file(cached_model_dir: str) -> str:
    """Resolve exactly one direct-child FLUX checkpoint, failing closed."""
    if not os.path.isdir(cached_model_dir):
        raise FileNotFoundError(
            f"standalone FLUX cache directory not found: {cached_model_dir!r}"
        )
    candidates = sorted(
        os.path.join(cached_model_dir, name)
        for name in os.listdir(cached_model_dir)
        if name.lower().endswith(".safetensors")
        and os.path.isfile(os.path.join(cached_model_dir, name))
        and not os.path.islink(os.path.join(cached_model_dir, name))
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "standalone FLUX cache must contain exactly one direct "
            f".safetensors file; found {len(candidates)}"
        )
    return candidates[0]


def _config_path(spec: ImageSpec) -> str:
    # Execute the exact TOML that Gate C later detaches into the private record.
    # Keeping a single copy avoids an unverifiable "generated here, archived
    # there" gap and ensures the recipe bytes attested by the run are the bytes
    # Kohya actually parsed.
    return os.path.join(spec.save_root, "config.toml")


def _run_kohya(
    config_path: str,
    deadline: Deadline,
    spec: ImageSpec,
    scope: dict,
) -> None:
    if deadline.remaining() <= _STOP_MARGIN_S:
        telemetry.event("kohya_skipped", reason="insufficient_soft_deadline")
        return
    script = os.path.join(_SD_SCRIPTS_DIR, "flux_train_network.py")
    if not os.path.isfile(script):
        raise FileNotFoundError(f"Kohya FLUX trainer not found: {script!r}")
    cmd = _command(config_path, script=script)
    log_path = _log_path(spec)
    telemetry.event("kohya_start")
    started = time.monotonic()
    gpu_stop = threading.Event()
    gpu_peak = {"mb": 0}
    gpu_thread = _start_gpu_sampler(gpu_stop, gpu_peak)
    stopped_by_deadline = False
    proc: subprocess.Popen | None = None

    try:
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=_SD_SCRIPTS_DIR,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while proc.poll() is None:
                # Deadline.remaining() already excludes the CLI's 180-second
                # export reserve. Stop one additional boundary margin early so
                # process-group termination and atomic promotion cannot race it.
                if deadline.remaining() <= _STOP_MARGIN_S:
                    stopped_by_deadline = True
                    _terminate(proc)
                    break
                time.sleep(_POLL_SECONDS)
    finally:
        gpu_stop.set()
        if gpu_thread is not None:
            gpu_thread.join(timeout=6)
        if gpu_peak["mb"] > 0:
            telemetry.sample("gpu_peak_mb", gpu_peak["mb"])

    rc = None if proc is None else proc.returncode
    telemetry.event(
        "kohya_end",
        returncode=rc,
        stopped_by_deadline=stopped_by_deadline,
        elapsed_s=round(time.monotonic() - started, 1),
    )
    loss, step = _parse_log(log_path)
    if loss is not None or step is not None:
        telemetry.event("kohya_metrics", loss=loss, last_step=step)
    if loss is not None:
        telemetry.sample("final_loss", loss)
        telemetry.train_point(step or 0, loss, None)

    if (
        rc not in (0, None)
        and not stopped_by_deadline
        and not checkpoints.current_loras(spec.save_root, scope)
    ):
        _tail_log(log_path)
        raise RuntimeError(f"Kohya failed (rc={rc}) with no current checkpoint")


def _command(config_path: str, *, script: str | None = None) -> list[str]:
    script = script or os.path.join(_SD_SCRIPTS_DIR, "flux_train_network.py")
    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--dynamo_backend",
        "no",
        "--dynamo_mode",
        "default",
        "--mixed_precision",
        "bf16",
        "--num_processes",
        "1",
        "--num_machines",
        "1",
        "--num_cpu_threads_per_process",
        "2",
        script,
        "--config_file",
        config_path,
    ]


def _start_gpu_sampler(
    stop: threading.Event,
    peak: dict[str, int],
) -> threading.Thread | None:
    def sample() -> None:
        while not stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                values = [
                    int(value)
                    for value in result.stdout.split()
                    if value.strip().isdigit()
                ]
                if values:
                    peak["mb"] = max(peak["mb"], max(values))
            except Exception:
                pass
            stop.wait(5)

    try:
        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        return thread
    except Exception:
        return None


def _log_path(spec: ImageSpec) -> str:
    log_dir = os.path.join(os.path.dirname(spec.config_path), "forge-logs")
    os.makedirs(log_dir, exist_ok=True)
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.task_id)
    safe_repo = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.expected_repo_name)
    return os.path.join(
        log_dir,
        f"{safe_task}-{safe_repo}-kohya-{os.getpid()}-{time.time_ns()}.log",
    )


def _parse_log(path: str) -> tuple[float | None, int | None]:
    loss = None
    step = None
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        losses = re.findall(rf"(?:loss|avr_loss)[=:]\s*({number})", text, re.I)
        if losses:
            loss = float(losses[-1])
        progress = re.findall(r"(\d+)\s*/\s*(\d+)", text)
        if progress:
            step = int(progress[-1][0])
    except Exception:
        pass
    return loss, step


def _terminate(proc: subprocess.Popen) -> None:
    def signal_group(sig: int) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                pass

    signal_group(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        signal_group(signal.SIGKILL)
        proc.wait(timeout=10)


def _tail_log(path: str, lines: int = 15) -> None:
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            tail = fh.readlines()[-lines:]
        telemetry.event("kohya_log_tail", tail="".join(tail)[-1500:])
    except Exception:
        pass
