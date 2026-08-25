"""Shape-aware FLUX backend with deadline-safe Kohya support.

The validator routes FLUX through the legacy-named Dockerfile.  Its downloader
normalizes a standalone FLUX repository to exactly one root ``.safetensors``
file, but preserves every other repository as a full snapshot directory. Kohya
understands the former shape; ai-toolkit understands the latter. Forge chooses
between them from the trusted, read-only cache shape while retaining the same
run scope, telemetry, kill-safe checkpoint promotion, publication scrub, and
never-forfeit fallback.

Week 11 validated the same Kohya family on two closed snapshot-shaped tasks at
exactly 94 optimizer steps and training seed 1.  A snapshot takes that route
only when one direct, regular, non-sharded file over 10 GiB has a safetensors
header carrying BFL ``double_blocks.*`` tensors.  Every ineligible or safely
contained failed attempt falls back to the unchanged ai-toolkit path.  The
legacy standalone-checkpoint route below remains unchanged.
"""

from __future__ import annotations

import json
import os
import re
import signal
import stat
import struct
import subprocess
import sys
import threading
import time

from forge import flux_kohya_config, telemetry
from forge.clock import Deadline
from forge.data import dataset
from forge.data.schema import ImageSpec
from forge.tasks import checkpoints, holdout
from forge.tasks.integrity import valid_safetensors


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
_SNAPSHOT_MIN_CHECKPOINT_BYTES = 10 * 1024**3
_MAX_SAFETENSORS_HEADER_BYTES = 256 * 1024 * 1024


class _SnapshotFinalizationError(RuntimeError):
    """A validated final may have been published; a second trainer is unsafe."""


def run(spec: ImageSpec, deadline: Deadline) -> None:
    try:
        layout, standalone_model = resolve_flux_cache_layout(spec.cached_model_dir)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        telemetry.event(
            "flux_snapshot_kohya_skipped",
            reason="cache_inspection_failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        _run_aitoolkit(spec, deadline, cache_layout="unresolved")
        return
    if layout == _SNAPSHOT_LAYOUT:
        if _attempt_snapshot_kohya(spec, deadline):
            return
        _run_aitoolkit(spec, deadline, cache_layout=layout)
        return

    telemetry.event(
        "flux_backend_selected",
        backend="kohya",
        cache_layout=layout,
    )
    _run_standalone_kohya(spec, deadline, standalone_model)


def _run_aitoolkit(
    spec: ImageSpec,
    deadline: Deadline,
    *,
    cache_layout: str,
) -> None:
    telemetry.set_meta(backend="aitoolkit", base_model_layout=cache_layout)
    telemetry.event(
        "flux_backend_selected",
        backend="aitoolkit",
        cache_layout=cache_layout,
    )
    # Import lazily: the Kohya image contains both runtimes, while this module
    # must remain importable in unit tests without ai-toolkit dependencies.
    from forge.tasks import aitoolkit

    aitoolkit.run(spec, deadline)


def _attempt_snapshot_kohya(spec: ImageSpec, deadline: Deadline) -> bool:
    """Run the exact Week-11 snapshot recipe or safely decline to ai-toolkit."""
    checkpoint = resolve_snapshot_kohya_checkpoint(spec.cached_model_dir)
    if checkpoint is None:
        telemetry.event(
            "flux_snapshot_kohya_skipped",
            reason="no_single_eligible_bfl_checkpoint",
        )
        return False

    ready, reason = _kohya_runtime_ready()
    if not ready:
        telemetry.event("flux_snapshot_kohya_skipped", reason=reason)
        return False

    remaining_soft_s = deadline.remaining()
    planned = flux_kohya_config.budgeted_train_steps(
        remaining_soft_s,
        boundary_margin_s=_STOP_MARGIN_S,
        max_steps=flux_kohya_config.WEEK11_SNAPSHOT_TRAIN_STEPS,
    )
    if planned != flux_kohya_config.WEEK11_SNAPSHOT_TRAIN_STEPS:
        telemetry.event(
            "flux_snapshot_kohya_skipped",
            reason="insufficient_budget_for_exact_week11_recipe",
            planned_steps=planned,
            required_steps=flux_kohya_config.WEEK11_SNAPSHOT_TRAIN_STEPS,
            remaining_soft_s=round(remaining_soft_s, 1),
        )
        return False

    try:
        telemetry.event(
            "flux_backend_selected",
            backend="kohya",
            cache_layout=_SNAPSHOT_LAYOUT,
            checkpoint=os.path.basename(checkpoint),
        )
        _run_snapshot_kohya(
            spec,
            deadline,
            checkpoint,
            remaining_soft_s=remaining_soft_s,
        )
        return True
    except _SnapshotFinalizationError:
        # finalize may already have atomically replaced the public candidate.
        # Starting another trainer in that ambiguity is less safe than letting
        # the outer no-artifact fallback handle the task.
        raise
    except Exception as exc:
        # Move this attempt's partials before begin_run: begin_run deliberately
        # promotes the highest visible repo-prefixed checkpoint to a kill-safe
        # last.safetensors, which would otherwise make an unvalidated Kohya
        # periodic eligible if the ai-toolkit fallback also failed.
        try:
            _quarantine_failed_snapshot_attempt(spec)
            checkpoints.begin_run(spec.save_root, spec.expected_repo_name)
        except Exception as reset_exc:
            raise RuntimeError(
                "failed snapshot Kohya attempt could not be isolated for "
                "ai-toolkit fallback"
            ) from reset_exc
        telemetry.event(
            "flux_snapshot_kohya_fallback",
            reason="kohya_attempt_failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        return False


def _quarantine_failed_snapshot_attempt(spec: ImageSpec) -> None:
    """Recoverably isolate current Kohya outputs before ai-toolkit replans."""
    scope = checkpoints.load_run(spec.save_root)
    if scope is None:
        raise RuntimeError("snapshot Kohya fallback lost its checkpoint scope")
    paths = checkpoints.current_loras(spec.save_root, scope)
    for name in ("optimizer.pt", "learnable_snr.json"):
        path = os.path.join(spec.save_root, name)
        if os.path.lexists(path):
            paths.append(path)
    if not paths:
        return

    quarantine = os.path.join(
        os.path.dirname(os.path.abspath(spec.save_root)),
        f".forge-failed-snapshot-kohya-{os.getpid()}-{time.time_ns()}",
    )
    os.makedirs(quarantine, mode=0o700)
    moved: list[str] = []
    try:
        for source in sorted(set(paths)):
            if os.path.islink(source) or not os.path.isfile(source):
                raise RuntimeError(
                    f"unsafe failed snapshot Kohya output: {source!r}"
                )
            destination = os.path.join(quarantine, os.path.basename(source))
            os.replace(source, destination)
            moved.append(destination)
        _fsync_dir(quarantine)
        _fsync_dir(spec.save_root)
    except BaseException:
        # Best-effort rollback keeps the failed attempt visible under its
        # original names; the caller will refuse to start ai-toolkit.
        for destination in reversed(moved):
            source = os.path.join(spec.save_root, os.path.basename(destination))
            try:
                os.replace(destination, source)
            except OSError:
                pass
        raise


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def resolve_snapshot_kohya_checkpoint(cached_model_dir: str) -> str | None:
    """Return one direct >10-GiB BFL safetensors file, otherwise ``None``."""
    try:
        safetensors: list[str] = []
        for entry in os.scandir(cached_model_dir):
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                continue
            if not entry.name.endswith(".safetensors"):
                continue
            safetensors.append(entry.path)
            if _SHARDED_CHECKPOINT_PATTERN.search(entry.name):
                return None
        if len(safetensors) != 1:
            return None
        checkpoint = safetensors[0]
        return checkpoint if _is_bfl_flux_checkpoint(checkpoint) else None
    except (OSError, ValueError):
        return None


def _is_bfl_flux_checkpoint(path: str) -> bool:
    """Read only the bounded safetensors header and prove BFL tensor names."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    try:
        fd = os.open(path, flags)
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size <= _SNAPSHOT_MIN_CHECKPOINT_BYTES
        ):
            return False
        raw_length = os.read(fd, 8)
        if len(raw_length) != 8:
            return False
        (header_length,) = struct.unpack("<Q", raw_length)
        if not 0 < header_length <= _MAX_SAFETENSORS_HEADER_BYTES:
            return False
        raw_header = bytearray()
        while len(raw_header) < header_length:
            chunk = os.read(fd, header_length - len(raw_header))
            if not chunk:
                return False
            raw_header.extend(chunk)
        header = json.loads(raw_header.decode("utf-8"))
        return isinstance(header, dict) and any(
            isinstance(name, str) and name.startswith("double_blocks.")
            for name in header
        )
    except (OSError, ValueError, UnicodeDecodeError, struct.error):
        return False
    finally:
        if fd is not None:
            os.close(fd)


def _kohya_runtime_ready() -> tuple[bool, str]:
    """Preflight the baked Kohya runtime before spending the training clock."""
    if not os.path.isfile(os.path.join(_SD_SCRIPTS_DIR, "flux_train_network.py")):
        return False, "kohya_script_missing"
    for asset in (
        flux_kohya_config.AE_PATH,
        flux_kohya_config.CLIP_L_PATH,
        flux_kohya_config.T5XXL_PATH,
    ):
        if not os.path.isfile(asset):
            return False, "kohya_flux_asset_missing"
    if not os.path.isdir(flux_kohya_config.TOKENIZER_CACHE_DIR):
        return False, "kohya_tokenizer_cache_missing"
    return True, "ready"


def _run_snapshot_kohya(
    spec: ImageSpec,
    deadline: Deadline,
    base_model: str,
    *,
    remaining_soft_s: float,
) -> None:
    """Train and accept only the natural exact 94-step Week-11 final."""
    os.makedirs(spec.save_root, exist_ok=True)
    os.makedirs(spec.training_folder, exist_ok=True)
    scope = checkpoints.ensure_run(spec.save_root, spec.expected_repo_name)
    train_data_dir, pairs = dataset.prepare_kohya_flux_dataset(
        spec.cached_zip_path,
        images_root=spec.dataset_images_dir,
        trigger_word=spec.trigger_word,
    )
    steps = flux_kohya_config.WEEK11_SNAPSHOT_TRAIN_STEPS
    scope = checkpoints.set_planned_steps(
        spec.save_root,
        scope,
        steps,
        model_type=spec.model_type,
    )
    config_path = _config_path(spec)
    config = flux_kohya_config.build_week11_snapshot_config(
        base_model=base_model,
        train_data_dir=train_data_dir,
        output_dir=spec.save_root,
        output_name=spec.expected_repo_name,
        config_file=config_path,
    )
    flux_kohya_config.write_config(config, config_path)
    telemetry.event(
        "kohya_step_budgeted",
        max_steps=steps,
        planned_steps=steps,
        remaining_soft_s=round(remaining_soft_s, 1),
        boundary_margin_s=_STOP_MARGIN_S,
        recipe="week11_matched_seed1",
    )
    telemetry.set_meta(
        model_type=spec.model_type,
        backend="kohya",
        base_model_layout=_SNAPSHOT_LAYOUT,
        pairs=pairs,
        base_model=os.path.basename(base_model),
        steps=steps,
        save_every=config["save_every_n_steps"],
        trigger_word=spec.trigger_word,
        training_seed=flux_kohya_config.WEEK11_SNAPSHOT_SEED,
    )
    telemetry.event("dataset_ready", pairs=pairs)

    rc, stopped_by_deadline = _run_kohya(config_path, deadline, spec, scope)
    if rc != 0 or stopped_by_deadline:
        raise RuntimeError(
            "snapshot Kohya did not exit naturally after the exact 94 steps"
        )
    exact_final = os.path.join(spec.save_root, f"{spec.expected_repo_name}.safetensors")
    if (
        exact_final not in checkpoints.current_loras(spec.save_root, scope)
        or not valid_safetensors(exact_final)
    ):
        raise RuntimeError("snapshot Kohya produced no valid natural 94-step final")
    try:
        record = checkpoints.finalize(
            spec.save_root,
            spec.expected_repo_name,
            scope,
            context="flux_kohya_week11_snapshot_training",
        )
    except Exception as exc:
        raise _SnapshotFinalizationError(
            "snapshot Kohya exact final could not be published safely"
        ) from exc
    if (
        record is None
        or record.get("source") != "exact_final"
        or record.get("selected_step") != steps
    ):
        raise _SnapshotFinalizationError(
            "snapshot Kohya finalization did not bind the exact 94-step final"
        )
    telemetry.event(
        "checkpoint_finalized",
        status=record["status"],
        source=record["source"],
        selected_step=record["selected_step"],
    )


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
    remaining_soft_s = deadline.remaining()
    steps = flux_kohya_config.budgeted_train_steps(
        remaining_soft_s,
        boundary_margin_s=_STOP_MARGIN_S,
    )
    telemetry.event(
        "kohya_step_budgeted",
        max_steps=flux_kohya_config.MAX_TRAIN_STEPS,
        planned_steps=steps,
        remaining_soft_s=round(remaining_soft_s, 1),
        boundary_margin_s=_STOP_MARGIN_S,
        last_durable_steps=flux_kohya_config.R11_LAST_DURABLE_STEPS,
        observed_child_runtime_s=(
            flux_kohya_config.R11_OBSERVED_CHILD_RUNTIME_S
        ),
        throughput_headroom=flux_kohya_config.DEADLINE_THROUGHPUT_HEADROOM,
    )
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
) -> tuple[int | None, bool]:
    if deadline.remaining() <= _STOP_MARGIN_S:
        telemetry.event("kohya_skipped", reason="insufficient_soft_deadline")
        return None, True
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
    return rc, stopped_by_deadline


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
