"""Shape-aware legacy FLUX backend with deadline-safe Kohya support.

The validator routes FLUX through the legacy-named Dockerfile.  Its downloader
normalizes a standalone FLUX repository to exactly one root ``.safetensors``
file, but preserves every other repository as a full snapshot directory. Kohya
understands the former shape; ai-toolkit understands the latter. Forge chooses
between them from the trusted, read-only cache shape while retaining the same
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
_KOHYA_PYTHONPATH_ENV = "FORGE_KOHYA_PYTHONPATH"
_KOHYA_LD_LIBRARY_PATH_ENV = "FORGE_KOHYA_LD_LIBRARY_PATH"
_KOHYA_LD_PRELOAD_ENV = "FORGE_KOHYA_LD_PRELOAD"
_KOHYA_PROTOBUF_ENV = "FORGE_KOHYA_PROTOBUF_IMPLEMENTATION"
_KOHYA_PATH_ENV = "FORGE_KOHYA_PATH"
_STANDALONE_LAYOUT = "standalone_checkpoint"
_SNAPSHOT_LAYOUT = "snapshot_directory"
_MODEL_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".ckpt", ".pt")
_DIFFUSERS_COMPONENT_DIRS = {
    "scheduler",
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
    "transformer",
    "unet",
    "vae",
}
_WEIGHT_INDEX_SUFFIXES = (".bin.index.json", ".safetensors.index.json")
_SHARDED_CHECKPOINT_PATTERN = re.compile(r"-\d{5}-of-\d{5}\.safetensors$")


def run(spec: ImageSpec, deadline: Deadline) -> None:
    layout, standalone_model = resolve_flux_cache_layout(spec.cached_model_dir)
    if layout == _SNAPSHOT_LAYOUT:
        telemetry.set_meta(backend="aitoolkit", base_model_layout=layout)
        telemetry.event(
            "flux_backend_selected",
            backend="aitoolkit",
            cache_layout=layout,
        )
        # Import lazily: the Kohya image contains both runtimes, while this
        # module must remain importable in unit tests without ai-toolkit deps.
        from forge.tasks import aitoolkit

        aitoolkit.run(spec, deadline)
        return

    telemetry.event(
        "flux_backend_selected",
        backend="kohya",
        cache_layout=layout,
    )
    _run_standalone_kohya(spec, deadline, standalone_model)


def _run_standalone_kohya(
    spec: ImageSpec,
    deadline: Deadline,
    base_model: str | None,
) -> None:
    if base_model is None:  # defensive: the resolver binds this invariant
        raise RuntimeError("standalone FLUX layout resolved without a checkpoint")
    os.makedirs(spec.save_root, exist_ok=True)
    os.makedirs(spec.training_folder, exist_ok=True)
    scope = checkpoints.ensure_run(spec.save_root, spec.expected_repo_name)

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

    # The parent process uses the image's ai-toolkit dependency graph; Kohya
    # runs in a child-only graph selected by _kohya_subprocess_env(). Calling
    # collect_env here would therefore attest Torch/CUDA/library versions that
    # did not train this checkpoint. Accurate absence is better than false
    # provenance; the child process and image build gates own runtime identity.
    telemetry.set_meta(
        model_type=spec.model_type,
        backend="kohya",
        base_model_layout=_STANDALONE_LAYOUT,
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


def resolve_flux_cache_layout(cached_model_dir: str) -> tuple[str, str | None]:
    """Resolve the validator's two FLUX cache shapes, failing closed.

    G.O.D's downloader (introduced in #1309) normalizes a standalone FLUX repo
    to exactly one direct, regular ``.safetensors`` file. Its legacy path
    resolver makes the same exact-one-root-file distinction. Any other complete
    model repository remains a directory for ai-toolkit. We mirror that contract
    and add local integrity checks because guessing a backend for an empty,
    root-symlinked, or weight-free cache would consume the task budget first.
    """
    if os.path.islink(cached_model_dir):
        raise RuntimeError(
            f"FLUX cache root must not be a symlink: {cached_model_dir!r}"
        )
    if not os.path.isdir(cached_model_dir):
        raise FileNotFoundError(f"FLUX cache directory not found: {cached_model_dir!r}")

    direct_files: list[str] = []
    direct_directories: set[str] = set()
    model_weights: list[str] = []
    try:
        for entry in os.scandir(cached_model_dir):
            if entry.is_symlink():
                raise RuntimeError(
                    f"FLUX cache contains a symlink: {entry.path!r}"
                )
            if entry.is_file(follow_symlinks=False):
                direct_files.append(entry.path)
            elif entry.is_dir(follow_symlinks=False):
                direct_directories.add(entry.name)
    except OSError as exc:
        raise RuntimeError(
            f"unable to inspect FLUX cache directory {cached_model_dir!r}"
        ) from exc

    # Mirror G.O.D #1309's repository classifier while tolerating directories its
    # normalizer deliberately leaves in reused caches. Arbitrary assets/examples
    # (and Hugging Face's .cache metadata) do not change a standalone checkpoint;
    # known Diffusers components, a weight index anywhere, or a shard-form name
    # are semantic proof that the directory belongs to ai-toolkit instead.
    has_weight_index = False
    try:
        for _root, _dirs, files in os.walk(
            cached_model_dir,
            followlinks=False,
            onerror=_raise_walk_error,
        ):
            if any(name.endswith(_WEIGHT_INDEX_SUFFIXES) for name in files):
                has_weight_index = True
                break
    except OSError as exc:
        raise RuntimeError(
            f"unable to inspect FLUX cache metadata {cached_model_dir!r}"
        ) from exc

    direct_names = {os.path.basename(path) for path in direct_files}
    standalone_candidate = (
        len(direct_files) == 1
        and direct_files[0].endswith(".safetensors")
        and "model_index.json" not in direct_names
        and not (direct_directories & _DIFFUSERS_COMPONENT_DIRS)
        and not has_weight_index
        and not _SHARDED_CHECKPOINT_PATTERN.search(
            os.path.basename(direct_files[0])
        )
    )
    if standalone_candidate:
        try:
            if os.path.getsize(direct_files[0]) <= 0:
                raise RuntimeError(
                    f"standalone FLUX checkpoint is empty: {direct_files[0]!r}"
                )
        except OSError as exc:
            raise RuntimeError(
                f"unable to inspect standalone FLUX checkpoint {direct_files[0]!r}"
            ) from exc
        return _STANDALONE_LAYOUT, direct_files[0]

    # Every other cache shape is a directory candidate for ai-toolkit. Require
    # at least one non-empty model weight and reject symlinks rather than guessing
    # at a malformed/incomplete cache. Full snapshots may be sharded or keep all
    # weights under component directories.
    try:
        for root, dirs, files in os.walk(
            cached_model_dir,
            followlinks=False,
            onerror=_raise_walk_error,
        ):
            for name in dirs:
                path = os.path.join(root, name)
                if os.path.islink(path):
                    raise RuntimeError(f"FLUX cache contains a symlink: {path!r}")
            for name in files:
                path = os.path.join(root, name)
                if os.path.islink(path):
                    raise RuntimeError(f"FLUX cache contains a symlink: {path!r}")
                if name.lower().endswith(_MODEL_WEIGHT_SUFFIXES):
                    if os.path.getsize(path) <= 0:
                        raise RuntimeError(
                            f"FLUX snapshot contains an empty model weight: {path!r}"
                        )
                    model_weights.append(path)
    except OSError as exc:
        raise RuntimeError(
            f"unable to inspect FLUX snapshot {cached_model_dir!r}"
        ) from exc

    if not model_weights:
        raise RuntimeError(
            "FLUX cache is neither an exact-one-file standalone checkpoint "
            "nor a snapshot containing model weights"
        )
    return _SNAPSHOT_LAYOUT, None


def resolve_standalone_model_file(cached_model_dir: str) -> str:
    """Resolve only the downloader's normalized one-file FLUX shape."""
    layout, checkpoint = resolve_flux_cache_layout(cached_model_dir)
    if layout != _STANDALONE_LAYOUT or checkpoint is None:
        raise RuntimeError(
            "standalone FLUX cache must match G.O.D's exact-one direct "
            ".safetensors contract without Diffusers/index/shard markers"
        )
    return checkpoint


def _raise_walk_error(error: OSError) -> None:
    raise error


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
                env=_kohya_subprocess_env(),
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
    loss = _parse_loss(log_path)
    if loss is not None:
        # Checkpoint selection is the only authoritative source of training
        # progress. Kohya logs contain many unrelated ``current/total`` counters
        # (cache passes, data loaders, saves), so publishing a guessed last_step
        # can contradict the checkpoint record. Retain the observed loss only.
        telemetry.event("kohya_metrics", loss=loss)
        telemetry.sample("final_loss", loss)

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


def _parse_loss(path: str) -> float | None:
    loss = None
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        losses = re.findall(rf"(?:loss|avr_loss)[=:]\s*({number})", text, re.I)
        if losses:
            loss = float(losses[-1])
    except Exception:
        pass
    return loss


def _kohya_subprocess_env() -> dict[str, str]:
    """Return a child-only environment for the pinned Kohya dependency graph.

    The legacy image carries ai-toolkit as its default Python graph so directory
    snapshots work. Its Kohya graph remains at the pinned base-image location
    and is exposed only to the standalone subprocess; mixing both graphs in one
    interpreter would make dispatch shape-aware but runtime behavior ambiguous.
    """
    env = os.environ.copy()
    kohya_pythonpath = env.get(_KOHYA_PYTHONPATH_ENV)
    if kohya_pythonpath:
        env["PYTHONPATH"] = kohya_pythonpath
    kohya_library_path = env.get(_KOHYA_LD_LIBRARY_PATH_ENV)
    if kohya_library_path:
        env["LD_LIBRARY_PATH"] = kohya_library_path
    kohya_ld_preload = env.get(_KOHYA_LD_PRELOAD_ENV)
    if kohya_ld_preload:
        env["LD_PRELOAD"] = kohya_ld_preload
    kohya_protobuf = env.get(_KOHYA_PROTOBUF_ENV)
    if kohya_protobuf:
        env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = kohya_protobuf
    kohya_path = env.get(_KOHYA_PATH_ENV)
    if kohya_path:
        env["PATH"] = kohya_path
    return env


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
