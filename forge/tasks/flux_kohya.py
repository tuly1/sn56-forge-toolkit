"""Shape-aware legacy FLUX backend with deadline-safe Kohya support.

The validator routes FLUX through the legacy-named Dockerfile.  Its downloader
normalizes a standalone FLUX repository to exactly one root ``.safetensors``
file, but preserves every other repository as a full snapshot directory. Kohya
understands the former shape; ai-toolkit understands the latter. Forge chooses
between them from the trusted, read-only cache shape while retaining the same
run scope, telemetry, kill-safe checkpoint promotion, publication scrub, and
never-forfeit fallback.

WEEK-9 FIELD-FAMILY PORT (evidence/week9-flux-family-port-20260819/CHANGES.md):
the beta Phase D A/B measured the field's kohya dim-128 TE-inclusive family
19.55% ahead of our ai-toolkit flux family on the archived Aug-3 flux task
(beta REPORT §5).  Snapshot-directory FLUX caches therefore now ATTEMPT the
same Kohya recipe first — the full BFL checkpoint that Kohya consumes ships at
the snapshot root (the evaluator's own >10 GiB rule finds it there) — and fall
back to the unchanged ai-toolkit path after ordinary failures such as an
ineligible cache, missing runtime, insufficient budget, config rejection, or a
crash without a checkpoint, but only after Kohya shutdown is verified.  An
unverified trainer fails closed to the CLI's non-trainer fallback instead of
starting ai-toolkit on the same GPU.  A safe fallback re-plans on the remaining
clock, so degrade-not-forfeit is preserved whenever containment is proven.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import signal
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

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
# WEEK-9 snapshot->kohya route.  Opt-out env for a no-rebuild A/B and a
# one-line operational rollback: "aitoolkit" restores the pre-port behavior.
_SNAPSHOT_BACKEND_ENV = "FORGE_FLUX_SNAPSHOT_BACKEND"
# The evaluator's own flux base-file rule (is_safetensors_available @ G.O.D
# f7caab6c: the largest .safetensors over 10 GiB).  A full BFL FLUX checkpoint
# is the only snapshot member that big; adapters/VAE/TE shards are far smaller.
_SNAPSHOT_MIN_CHECKPOINT_BYTES = 10 * 1024**3
# safetensors headers are single-digit MB in practice; anything past this is
# not a header we should trust enough to parse.
_MAX_SAFETENSORS_HEADER_BYTES = 256 * 1024 * 1024
_KOHYA_RUN_TOKEN_ENV = "FORGE_KOHYA_RUN_TOKEN"


# WEEK-9 HAZARD 2 (evidence/week9-hazards-20260819/CHANGES.md).  The snapshot
# route may fall back to ai-toolkit only after verified Kohya shutdown.  A
# child that outlives that transition would share the GPU with ai-toolkit: VRAM
# exhaustion, or a truncated/interleaved artifact.  Every child this process
# spawns is registered here so the fallback can prove none is left alive,
# including on the paths that never reach _run_kohya's own cleanup.
@dataclass(frozen=True)
class _ProcessIdentity:
    """A PID bound to one kernel process lifetime, not just a reusable number."""

    pid: int
    started: str


@dataclass(frozen=True)
class _ProcessInfo:
    identity: _ProcessIdentity
    ppid: int
    pgid: int
    state: str


@dataclass
class _KohyaChild:
    """Lifecycle ownership retained until the whole trainer tree is gone."""

    proc: subprocess.Popen
    pid: int
    token: str | None
    leader_identity: _ProcessIdentity | None = None
    pgid: int | None = None
    descendants: dict[int, _ProcessIdentity] = field(default_factory=dict)
    last_error: str | None = None


class KohyaContainmentError(RuntimeError):
    """Kohya may still own the GPU, so another trainer must not be launched."""


class _ProcessInspectionError(RuntimeError):
    pass


_ACTIVE_CHILDREN_LOCK = threading.Lock()
_ACTIVE_KOHYA_CHILDREN: list[_KohyaChild] = []


def _register_kohya_child(
    proc: subprocess.Popen,
    *,
    token: str | None = None,
) -> _KohyaChild | None:
    child = _ensure_kohya_child_registered(proc, token=token)
    if child is None:
        return None
    try:
        _refresh_kohya_identities(child)
    except BaseException as exc:  # noqa: BLE001 — retain ownership on uncertainty
        child.last_error = f"identity registration failed: {type(exc).__name__}: {exc}"
    return child


def _ensure_kohya_child_registered(
    proc: subprocess.Popen,
    *,
    token: str | None,
) -> _KohyaChild | None:
    """Append-only ownership primitive used in the post-Popen recovery gap."""
    # Test doubles for an already-complete Popen may omit pid.  A real Popen
    # always has one; a live pid-less object cannot be owned safely.
    pid = getattr(proc, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        try:
            if proc.poll() is not None:
                return None
        except Exception:
            pass
        raise KohyaContainmentError("live Kohya child has no usable pid")
    # Append BEFORE process inspection.  /proc, ps, or even a test injection
    # may fail after Popen; the descriptor must already be recoverable then.
    with _ACTIVE_CHILDREN_LOCK:
        for child in _ACTIVE_KOHYA_CHILDREN:
            if child.proc is proc:
                if child.token is None:
                    child.token = token
                return child
        child = _KohyaChild(proc=proc, pid=pid, token=token)
        _ACTIVE_KOHYA_CHILDREN.append(child)
    return child


def _forget_kohya_child(child: _KohyaChild) -> None:
    with _ACTIVE_CHILDREN_LOCK:
        try:
            _ACTIVE_KOHYA_CHILDREN.remove(child)
        except ValueError:
            pass


def reap_kohya_children(reason: str) -> int:
    """Terminate AND reap every Kohya child this process still owns.

    Returns the number of lifecycle records completely released.  A failed
    record remains registered, with its Popen descriptor and identity-bound
    descendants intact, so a later barrier can retry.  Helper failures are
    reported but never converted into permission to start a second trainer.
    """
    with _ACTIVE_CHILDREN_LOCK:
        pending = list(_ACTIVE_KOHYA_CHILDREN)
    reaped = 0
    survived = 0
    for child in pending:
        try:
            confirmed = _terminate(child, sweep_descendants=True)
        except BaseException as exc:  # noqa: BLE001 — fail closed, retain handle
            child.last_error = (
                f"termination helper failed: {type(exc).__name__}: {exc}"
            )
            confirmed = False
        if confirmed:
            _forget_kohya_child(child)
            reaped += 1
        else:
            survived += 1
    if reaped or survived:
        try:
            telemetry.event(
                "kohya_child_reaped",
                reason=reason,
                reaped=reaped,
                survived=survived,
            )
        except Exception:  # noqa: BLE001 — telemetry must never forfeit
            pass
    return reaped


def _require_kohya_quiescent(reason: str) -> None:
    """Positive barrier required before handing the GPU to another trainer."""
    reap_kohya_children(reason)
    with _ACTIVE_CHILDREN_LOCK:
        pending = list(_ACTIVE_KOHYA_CHILDREN)
    if not pending:
        return
    leaders = ",".join(str(child.pid) for child in pending)
    failures = "; ".join(
        f"pid={child.pid}: {child.last_error or 'shutdown unverified'}"
        for child in pending
    )
    raise KohyaContainmentError(
        "refusing ai-toolkit handoff because Kohya shutdown is unverified "
        f"(reason={reason}; leader_pids=[{leaders}]; {failures}); "
        "outer no-artifact fallback required"
    )


def run(spec: ImageSpec, deadline: Deadline) -> None:
    # HAZARD 2, outer guard: the barrier must run on EVERY exit, not only on
    # the fallback.  On the SUCCESS path a child whose death `_terminate`
    # could not confirm (the CUDA uninterruptible-sleep case this fix exists
    # for) would otherwise stay alive through export, still writing into
    # save_root and still holding VRAM.  A no-op when the registry is empty.
    try:
        layout, standalone_model = resolve_flux_cache_layout(spec.cached_model_dir)
        if layout == _SNAPSHOT_LAYOUT:
            if _attempt_snapshot_kohya(spec, deadline):
                return
            # HAZARD 2 barrier.  This is the single choke point every fallback
            # path funnels through — env opt-out, ineligible cache, missing
            # runtime, budget floor, config rejection, crash, timeout — so it
            # is the only place that can prove the GPU is free BEFORE
            # ai-toolkit starts.  It emits nothing when no child was left
            # behind, so the byte-identical fallback contract is untouched.
            _require_kohya_quiescent("flux_snapshot_fallback")
            telemetry.set_meta(backend="aitoolkit", base_model_layout=layout)
            telemetry.event(
                "flux_backend_selected",
                backend="aitoolkit",
                cache_layout=layout,
            )
            # Import lazily: the Kohya image contains both runtimes, while
            # this module must remain importable in unit tests without
            # ai-toolkit deps.
            from forge.tasks import aitoolkit

            aitoolkit.run(spec, deadline)
            return

        telemetry.event(
            "flux_backend_selected",
            backend="kohya",
            cache_layout=layout,
        )
        _run_standalone_kohya(spec, deadline, standalone_model)
    finally:
        _require_kohya_quiescent("flux_run_exit")


def _attempt_snapshot_kohya(spec: ImageSpec, deadline: Deadline) -> bool:
    """Try the week-9 field-family Kohya route on a snapshot cache.

    Returns True only when Kohya finalized an artifact.  Ordinary failures —
    opt-out, ineligible checkpoint, missing runtime, budget below the depth
    floor, or training errors — return False so the caller can run ai-toolkit
    after its positive shutdown barrier.  A ``KohyaContainmentError`` is
    deliberately re-raised so the CLI uses only its non-trainer fallback.
    """
    try:
        if os.environ.get(_SNAPSHOT_BACKEND_ENV, "kohya").strip().lower() != "kohya":
            telemetry.event(
                "flux_snapshot_kohya_skipped", reason="env_opt_out"
            )
            return False
        checkpoint = resolve_snapshot_kohya_checkpoint(spec.cached_model_dir)
        if checkpoint is None:
            telemetry.event(
                "flux_snapshot_kohya_skipped", reason="no_eligible_checkpoint"
            )
            return False
        ready, not_ready_reason = _kohya_runtime_ready()
        if not ready:
            telemetry.event(
                "flux_snapshot_kohya_skipped", reason=not_ready_reason
            )
            return False
        telemetry.event(
            "flux_backend_selected",
            backend="kohya",
            cache_layout=_SNAPSHOT_LAYOUT,
            checkpoint=os.path.basename(checkpoint),
        )
        if _train_with_kohya(spec, deadline, checkpoint, layout=_SNAPSHOT_LAYOUT):
            return True
        telemetry.event(
            "flux_snapshot_kohya_fallback", reason="budget_below_field_floor"
        )
        return False
    except KohyaContainmentError:
        # This is not an ordinary Kohya failure: another trainer is forbidden.
        # Let forge.cli choose its non-trainer, no-artifact fallback.
        raise
    except BaseException as exc:  # noqa: BLE001 — ordinary failure => aitoolkit
        try:
            telemetry.event(
                "flux_snapshot_kohya_fallback",
                reason="kohya_route_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        return False


def resolve_snapshot_kohya_checkpoint(cached_model_dir: str) -> str | None:
    """Resolve the single Kohya-consumable full FLUX checkpoint in a snapshot.

    Mirrors the evaluator's flux base-file selection (largest ``.safetensors``
    over 10 GiB, is_safetensors_available @ f7caab6c) restricted to DIRECT
    root regular files with non-shard names, then proves the file is a
    BFL-format FLUX checkpoint (``double_blocks.*`` tensors) by reading the
    safetensors header offline — the pinned Kohya flux loader consumes exactly
    that format (tonight's beta field arm trained from the same root file of
    the same snapshot shape, beta REPORT §5).  Returns None instead of raising:
    an ineligible snapshot falls back to ai-toolkit.
    """
    try:
        candidates: list[tuple[int, str]] = []
        for entry in os.scandir(cached_model_dir):
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                continue
            if not entry.name.endswith(".safetensors"):
                continue
            if _SHARDED_CHECKPOINT_PATTERN.search(entry.name):
                continue
            size = entry.stat(follow_symlinks=False).st_size
            if size > _SNAPSHOT_MIN_CHECKPOINT_BYTES:
                candidates.append((size, entry.path))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        checkpoint = candidates[0][1]
        if not _is_bfl_flux_checkpoint(checkpoint):
            return None
        return checkpoint
    except OSError:
        return None


def _is_bfl_flux_checkpoint(path: str) -> bool:
    """Whether the safetensors header carries BFL FLUX transformer tensors."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                return False
            (header_len,) = struct.unpack("<Q", raw)
            if not 0 < header_len <= _MAX_SAFETENSORS_HEADER_BYTES:
                return False
            header = json.loads(fh.read(header_len).decode("utf-8"))
        if not isinstance(header, dict):
            return False
        return any(key.startswith("double_blocks.") for key in header)
    except (OSError, ValueError, UnicodeDecodeError, struct.error):
        return False


def _kohya_runtime_ready() -> tuple[bool, str]:
    """Preflight the baked Kohya runtime surface before committing the clock."""
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


def _run_standalone_kohya(
    spec: ImageSpec,
    deadline: Deadline,
    base_model: str | None,
) -> None:
    if base_model is None:  # defensive: the resolver binds this invariant
        raise RuntimeError("standalone FLUX layout resolved without a checkpoint")
    _train_with_kohya(spec, deadline, base_model, layout=_STANDALONE_LAYOUT)


def _train_with_kohya(
    spec: ImageSpec,
    deadline: Deadline,
    base_model: str,
    *,
    layout: str,
) -> bool:
    os.makedirs(spec.save_root, exist_ok=True)
    os.makedirs(spec.training_folder, exist_ok=True)
    scope = checkpoints.ensure_run(spec.save_root, spec.expected_repo_name)

    train_data_dir, pairs = dataset.prepare_kohya_flux_dataset(
        spec.cached_zip_path,
        images_root=spec.dataset_images_dir,
        trigger_word=spec.trigger_word,
    )
    remaining_soft_s = deadline.remaining()
    budget_steps = flux_kohya_config.budgeted_train_steps(
        remaining_soft_s,
        boundary_margin_s=_STOP_MARGIN_S,
    )
    # WEEK-9 field depth law: plan the beta-validated field depth, never more
    # than the R11 completion budget allows.  The law also CAPS the standalone
    # path: every archived flux winner shipped 540-754 presentations, and the
    # only deeper observed arm (~960) lost — the previous budget-fill plans
    # (128/196 steps at 1.0/1.5 h = 1024/1568 presentations) overshot the whole
    # winner band (CHANGES.md §5).
    law_steps = flux_kohya_config.field_epoch_steps(pairs)
    steps = min(budget_steps, law_steps)
    if layout == _SNAPSHOT_LAYOUT:
        floor_steps = flux_kohya_config.field_floor_steps(pairs)
        if budget_steps < floor_steps:
            # Not enough clock to reach the shallow edge of the measured flat
            # band (40 epochs): the kohya family has no evidence of beating
            # the aitoolkit route there, and the fallback can still re-plan
            # its own clock-capped run on the remaining budget.
            telemetry.event(
                "kohya_snapshot_budget_below_floor",
                budget_steps=budget_steps,
                floor_steps=floor_steps,
                pairs=pairs,
                remaining_soft_s=round(remaining_soft_s, 1),
            )
            return False
    telemetry.event(
        "kohya_step_budgeted",
        max_steps=flux_kohya_config.MAX_TRAIN_STEPS,
        planned_steps=steps,
        budget_steps=budget_steps,
        field_law_steps=law_steps,
        pairs=pairs,
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
        base_model_layout=layout,
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
    return True


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
    child: _KohyaChild | None = None
    run_token: str | None = None
    registration_error: BaseException | None = None

    try:
        with open(log_path, "w", encoding="utf-8") as log:
            child_env = _kohya_subprocess_env()
            run_token = (
                f"{os.getpid()}-{time.time_ns()}-{secrets.token_hex(12)}"
            )
            child_env[_KOHYA_RUN_TOKEN_ENV] = run_token
            proc = subprocess.Popen(
                cmd,
                cwd=_SD_SCRIPTS_DIR,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=child_env,
            )
            child = _register_kohya_child(proc, token=run_token)
            while proc.poll() is None:
                if child is not None:
                    try:
                        _refresh_kohya_identities(child)
                    except BaseException as exc:  # noqa: BLE001 — barrier retries
                        child.last_error = (
                            "identity refresh failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                # Deadline.remaining() already excludes the CLI's 180-second
                # export reserve. Stop one additional boundary margin early so
                # process-group termination and atomic promotion cannot race it.
                if deadline.remaining() <= _STOP_MARGIN_S:
                    stopped_by_deadline = True
                    break
                time.sleep(_POLL_SECONDS)
    finally:
        # Containment comes first.  Sampler joins and telemetry are advisory
        # and must never be able to bypass ownership/reaping after Popen.
        if proc is not None:
            # Covers an injected/asynchronous BaseException after Popen
            # returned but before _register_kohya_child returned to this frame.
            # The ensure primitive also recovers a record already appended by
            # a partially completed registration, without duplicating it.
            try:
                ensured = _ensure_kohya_child_registered(proc, token=run_token)
                if ensured is not None:
                    child = ensured
            except BaseException as exc:  # noqa: BLE001
                registration_error = exc
        if child is not None:
            try:
                confirmed = _terminate(child, sweep_descendants=True)
            except BaseException as exc:  # noqa: BLE001 — retain for outer barrier
                child.last_error = (
                    f"termination helper failed: {type(exc).__name__}: {exc}"
                )
                confirmed = False
            if confirmed:
                _forget_kohya_child(child)
            try:
                telemetry.event(
                    "kohya_child_reaped",
                    reason="run_kohya_exit",
                    reaped=1 if confirmed else 0,
                    survived=0 if confirmed else 1,
                )
            except Exception:  # noqa: BLE001 — telemetry must not forfeit
                pass
        try:
            gpu_stop.set()
            if gpu_thread is not None:
                gpu_thread.join(timeout=6)
            if gpu_peak["mb"] > 0:
                telemetry.sample("gpu_peak_mb", gpu_peak["mb"])
        finally:
            if registration_error is not None:
                raise KohyaContainmentError(
                    "post-Popen Kohya registration could not be retained; "
                    "refusing ai-toolkit handoff; outer no-artifact fallback "
                    "required"
                ) from registration_error

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


def _process_table() -> dict[int, _ProcessInfo]:
    """Return a complete visible process snapshot or raise uncertainty.

    Linux's start-time clock tick is an immutable production process identity.
    The second-resolution ps fallback is best-effort support for CPU tests on
    macOS, never the release proof.  An empty/failed snapshot is not interpreted
    as proof that Kohya is gone.
    """
    table: dict[int, _ProcessInfo] = {}
    if os.path.isdir("/proc"):
        try:
            entries = os.listdir("/proc")
        except OSError as exc:
            raise _ProcessInspectionError(f"cannot list /proc: {exc}") from exc
        for entry in entries:
            if not entry.isdigit():
                continue
            try:
                info = _process_info(int(entry))
            except ValueError:
                continue
            if info is not None:
                table[info.identity.pid] = info
    else:
        try:
            result = subprocess.run(
                ["ps", "-Ao", "pid=,ppid=,pgid=,stat=,lstart="],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as exc:  # noqa: BLE001
            raise _ProcessInspectionError(f"ps process snapshot failed: {exc}") from exc
        if result.returncode != 0:
            raise _ProcessInspectionError(
                f"ps process snapshot returned {result.returncode}: "
                f"{result.stderr.strip()[-200:]}"
            )
        for line in result.stdout.splitlines():
            parts = line.split(None, 4)
            if len(parts) != 5:
                continue
            try:
                pid, ppid, pgid = (int(value) for value in parts[:3])
            except ValueError:
                continue
            table[pid] = _ProcessInfo(
                identity=_ProcessIdentity(pid, f"ps:{parts[4].strip()}"),
                ppid=ppid,
                pgid=pgid,
                state=parts[3],
            )
    if not table:
        raise _ProcessInspectionError("process snapshot was empty")
    return table


def _process_info(pid: int) -> _ProcessInfo | None:
    """One identity snapshot; Linux is production, ps is test-only fallback."""
    if os.path.isdir("/proc"):
        try:
            with open(f"/proc/{pid}/stat", "rb") as handle:
                raw = handle.read()
        except (FileNotFoundError, ProcessLookupError):
            return None
        except OSError as exc:
            raise _ProcessInspectionError(
                f"cannot read /proc/{pid}/stat: {exc}"
            ) from exc
        # stat field 2 (comm) may contain spaces and parentheses.  Fields after
        # its final ')' are state, ppid, pgrp, session, ..., starttime at 19.
        close = raw.rfind(b")")
        tail = raw[close + 1 :].split() if close >= 0 else []
        if len(tail) < 20:
            raise _ProcessInspectionError(f"malformed /proc/{pid}/stat")
        try:
            return _ProcessInfo(
                identity=_ProcessIdentity(
                    pid, f"proc:{tail[19].decode('ascii')}"
                ),
                ppid=int(tail[1]),
                pgid=int(tail[2]),
                state=tail[0].decode("ascii", errors="replace"),
            )
        except (ValueError, IndexError) as exc:
            raise _ProcessInspectionError(
                f"malformed numeric fields in /proc/{pid}/stat"
            ) from exc

    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid=,ppid=,pgid=,stat=,lstart="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:  # noqa: BLE001
        raise _ProcessInspectionError(f"ps process lookup failed: {exc}") from exc
    if result.returncode not in (0, 1):
        raise _ProcessInspectionError(
            f"ps process lookup returned {result.returncode}: "
            f"{result.stderr.strip()[-200:]}"
        )
    line = result.stdout.strip()
    if not line:
        return None
    parts = line.split(None, 4)
    if len(parts) != 5:
        raise _ProcessInspectionError(f"malformed ps identity for pid {pid}")
    try:
        found_pid, ppid, pgid = (int(value) for value in parts[:3])
    except ValueError as exc:
        raise _ProcessInspectionError(f"malformed ps identity for pid {pid}") from exc
    if found_pid != pid:
        raise _ProcessInspectionError(
            f"ps identity lookup returned pid {found_pid} for requested {pid}"
        )
    return _ProcessInfo(
        identity=_ProcessIdentity(pid, f"ps:{parts[4].strip()}"),
        ppid=ppid,
        pgid=pgid,
        state=parts[3],
    )


def _token_process_identities(
    token: str,
    table: dict[int, _ProcessInfo],
) -> dict[int, _ProcessIdentity]:
    """Find every process inheriting one Kohya launch token.

    The environment marker survives fork, exec, setsid, and reparenting.  It
    closes the race where a descendant leaves both the leader's ancestry and
    process group before teardown takes its snapshot.
    """
    marker = f"{_KOHYA_RUN_TOKEN_ENV}={token}"
    found: dict[int, _ProcessIdentity] = {}
    if os.path.isdir("/proc"):
        encoded = marker.encode()
        for pid, info in table.items():
            path = f"/proc/{pid}/environ"
            try:
                with open(path, "rb") as handle:
                    environment = handle.read().split(b"\0")
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError as exc:
                # A descendant can change uid without losing its inherited
                # token.  Any opaque environment makes the scan incomplete.
                raise _ProcessInspectionError(
                    f"cannot inspect process {pid} environment"
                ) from exc
            except OSError as exc:
                raise _ProcessInspectionError(
                    f"cannot inspect process {pid} environment: {exc}"
                ) from exc
            if encoded in environment:
                found[pid] = info.identity
        return found

    try:
        result = subprocess.run(
            ["ps", "eww", "-Ao", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:  # noqa: BLE001
        raise _ProcessInspectionError(f"ps environment scan failed: {exc}") from exc
    if result.returncode != 0:
        raise _ProcessInspectionError(
            f"ps environment scan returned {result.returncode}: "
            f"{result.stderr.strip()[-200:]}"
        )
    for line in result.stdout.splitlines():
        stripped = line.lstrip()
        pid_text, separator, command = stripped.partition(" ")
        if not separator or not pid_text.isdigit() or marker not in command:
            continue
        pid = int(pid_text)
        info = table.get(pid)
        if info is not None:
            found[pid] = info.identity
    return found


def _descendant_identities(
    root_pid: int,
    table: dict[int, _ProcessInfo],
) -> dict[int, _ProcessIdentity]:
    children: dict[int, list[int]] = {}
    for pid, info in table.items():
        children.setdefault(info.ppid, []).append(pid)
    found: dict[int, _ProcessIdentity] = {}
    frontier = list(children.get(root_pid, ()))
    while frontier:
        pid = frontier.pop()
        if pid == root_pid or pid in found:
            continue
        info = table.get(pid)
        if info is None:
            continue
        found[pid] = info.identity
        frontier.extend(children.get(pid, ()))
    return found


def _refresh_kohya_identities(
    child: _KohyaChild,
) -> dict[int, _ProcessInfo]:
    """Refresh every identity owned by a launch; uncertainty raises."""
    table = _process_table()
    leader = table.get(child.pid)
    if child.leader_identity is None and leader is not None:
        if leader.pgid != child.pid:
            raise _ProcessInspectionError(
                f"leader {child.pid} is not its private process-group leader"
            )
        child.leader_identity = leader.identity
        child.pgid = leader.pgid

    leader_matches = (
        leader is not None
        and child.leader_identity is not None
        and leader.identity == child.leader_identity
    )
    if leader_matches:
        child.descendants.update(_descendant_identities(child.pid, table))
        # A descendant may already be reparented but remain in the private
        # session's process group.  Capture it while the leader identity still
        # proves that this group cannot have been recycled.
        if child.pgid is not None:
            child.descendants.update(
                {
                    pid: info.identity
                    for pid, info in table.items()
                    if pid != child.pid and info.pgid == child.pgid
                }
            )
    elif leader is not None and child.leader_identity is not None:
        try:
            leader_reported_live = child.proc.poll() is None
        except Exception as exc:
            raise _ProcessInspectionError(
                f"leader {child.pid} state unavailable after identity mismatch"
            ) from exc
        if leader_reported_live:
            raise _ProcessInspectionError(
                f"leader pid {child.pid} identity changed while Popen reports it live"
            )

    if child.token is not None:
        token_identities = _token_process_identities(child.token, table)
        child.descendants.update(
            {
                pid: identity
                for pid, identity in token_identities.items()
                if pid != child.pid
            }
        )
    elif child.leader_identity is None:
        try:
            leader_reported_live = child.proc.poll() is None
        except Exception as exc:
            raise _ProcessInspectionError(
                f"leader {child.pid} state unavailable"
            ) from exc
        if leader_reported_live:
            raise _ProcessInspectionError(
                f"leader {child.pid} identity unavailable"
            )
    return table


def _signal_process_identity(identity: _ProcessIdentity, sig: int) -> bool:
    """Signal only the exact process lifetime captured earlier.

    False means inspection/signalling uncertainty.  Absence or a different
    start identity is success for the old process and MUST NOT signal the
    replacement PID.
    """
    if sys.platform.startswith("linux"):
        pidfd_open = getattr(os, "pidfd_open", None)
        pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
        if pidfd_open is None or pidfd_send_signal is None:
            raise _ProcessInspectionError(
                "Linux pidfd signalling is unavailable; refusing numeric-pid signal"
            )
        try:
            pidfd = pidfd_open(identity.pid)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        try:
            # Open first, then validate.  If the PID is reused on either side
            # of this check, the pidfd still names only the process opened by
            # the kernel and can never signal the numeric-PID replacement.
            current = _process_info(identity.pid)
            if current is None or current.identity != identity:
                return True
            if current.state.startswith("Z"):
                return True
            try:
                pidfd_send_signal(pidfd, sig, None, 0)
            except ProcessLookupError:
                return True
            except OSError:
                return False
            return True
        finally:
            os.close(pidfd)

    # Portable CPU-test fallback.  Production training is Linux, where the
    # pidfd branch above makes identity validation and signalling atomic.
    current = _process_info(identity.pid)
    if current is None or current.identity != identity:
        return True
    if current.state.startswith("Z"):
        return True
    try:
        os.kill(identity.pid, sig)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return True


def _signal_leader_group(child: _KohyaChild, sig: int) -> bool:
    current = _process_info(child.pid)
    if (
        child.leader_identity is None
        or current is None
        or current.identity != child.leader_identity
    ):
        child.last_error = f"leader {child.pid} identity unavailable or changed"
        return False
    if child.pgid is None or current.pgid != child.pgid:
        child.last_error = f"leader {child.pid} process group identity changed"
        return False
    try:
        os.killpg(child.pgid, sig)
    except ProcessLookupError:
        return True
    except OSError as exc:
        child.last_error = f"cannot signal Kohya process group {child.pgid}: {exc}"
        return False
    return True


def _signal_owned_descendants(child: _KohyaChild, sig: int) -> bool:
    for identity in list(child.descendants.values()):
        try:
            if not _signal_process_identity(identity, sig):
                child.last_error = (
                    f"cannot signal descendant pid={identity.pid} safely"
                )
                return False
        except _ProcessInspectionError as exc:
            child.last_error = f"descendant identity inspection failed: {exc}"
            return False
    return True


def _wait_for_descendants(child: _KohyaChild, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    quiet_scans = 0
    while True:
        try:
            table = _refresh_kohya_identities(child)
        except _ProcessInspectionError as exc:
            child.last_error = f"descendant discovery failed: {exc}"
            return False

        alive: list[_ProcessIdentity] = []
        for pid, identity in list(child.descendants.items()):
            current = table.get(pid)
            if current is None or current.identity != identity:
                # PID reuse proves the recorded lifetime is gone.  Never touch
                # the replacement, and release only this old identity.
                child.descendants.pop(pid, None)
                continue
            alive.append(identity)

        if not alive:
            quiet_scans += 1
            # One /proc list can race a final fork.  Require a second complete
            # token/identity scan after a settle interval before releasing the
            # registry and handing the GPU to another trainer.
            if quiet_scans >= 2:
                return True
            time.sleep(0.05)
            continue
        quiet_scans = 0
        if time.monotonic() >= deadline:
            child.last_error = (
                "descendants did not disappear/reap: "
                + ",".join(str(identity.pid) for identity in alive)
            )
            return False
        for identity in alive:
            try:
                if not _signal_process_identity(identity, signal.SIGKILL):
                    child.last_error = (
                        f"cannot SIGKILL descendant pid={identity.pid} safely"
                    )
                    return False
            except _ProcessInspectionError as exc:
                child.last_error = f"descendant identity inspection failed: {exc}"
                return False
        time.sleep(0.05)


def _as_kohya_child(proc_or_child: subprocess.Popen | _KohyaChild) -> _KohyaChild:
    if isinstance(proc_or_child, _KohyaChild):
        return proc_or_child
    with _ACTIVE_CHILDREN_LOCK:
        for child in _ACTIVE_KOHYA_CHILDREN:
            if child.proc is proc_or_child:
                return child
    pid = getattr(proc_or_child, "pid", 0)
    child = _KohyaChild(proc=proc_or_child, pid=pid, token=None)
    try:
        _refresh_kohya_identities(child)
    except BaseException as exc:  # noqa: BLE001
        child.last_error = f"identity registration failed: {type(exc).__name__}: {exc}"
    return child


def _terminate(
    proc_or_child: subprocess.Popen | _KohyaChild,
    *,
    sweep_descendants: bool = False,
) -> bool:
    """Terminate the identity-bound trainer tree and reap its Popen leader.

    True is a positive proof: the direct child was waited/reaped and every
    recorded or token-discovered descendant is absent (or its PID now has a
    different start identity).  Any helper/identity uncertainty returns False
    and leaves a registered record owned by the outer fail-closed barrier.
    """
    child = _as_kohya_child(proc_or_child)
    child.last_error = None
    try:
        _refresh_kohya_identities(child)
    except _ProcessInspectionError as exc:
        child.last_error = f"initial process inspection failed: {exc}"

    try:
        running = child.proc.poll() is None
    except Exception as exc:  # noqa: BLE001
        child.last_error = f"leader poll failed: {type(exc).__name__}: {exc}"
        return False

    if running:
        try:
            if not _signal_leader_group(child, signal.SIGTERM):
                return False
            if sweep_descendants and not _signal_owned_descendants(
                child, signal.SIGTERM
            ):
                return False
        except _ProcessInspectionError as exc:
            child.last_error = f"leader identity inspection failed: {exc}"
            return False
        try:
            child.proc.wait(timeout=5)
            running = False
        except subprocess.TimeoutExpired:
            pass
        except Exception as exc:  # noqa: BLE001
            child.last_error = f"leader wait failed: {type(exc).__name__}: {exc}"
            return False

    if running:
        try:
            _refresh_kohya_identities(child)
            if not _signal_leader_group(child, signal.SIGKILL):
                return False
            if sweep_descendants and not _signal_owned_descendants(
                child, signal.SIGKILL
            ):
                return False
        except _ProcessInspectionError as exc:
            child.last_error = f"kill identity inspection failed: {exc}"
            return False
        try:
            child.proc.wait(timeout=10)
            running = False
        except subprocess.TimeoutExpired:
            child.last_error = f"leader pid={child.pid} survived SIGKILL deadline"
            return False
        except Exception as exc:  # noqa: BLE001
            child.last_error = f"leader reap failed: {type(exc).__name__}: {exc}"
            return False

    if running:
        child.last_error = f"leader pid={child.pid} shutdown unverified"
        return False
    if not sweep_descendants:
        return True
    if not _signal_owned_descendants(child, signal.SIGKILL):
        return False
    return _wait_for_descendants(child)


def _tail_log(path: str, lines: int = 15) -> None:
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            tail = fh.readlines()[-lines:]
        telemetry.event("kohya_log_tail", tail="".join(tail)[-1500:])
    except Exception:
        pass
