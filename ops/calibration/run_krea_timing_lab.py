#!/usr/bin/env python3
"""LAB ONLY: produce one Krea timing raw-record/profile evidence package.

This program is an explicit calibration supervisor.  It is not imported by
Forge's tournament entrypoint and its host-bound output must never be placed
under the uploaded checkpoint tree.  Production consumes reviewed constants in
``forge.recipe``; this tool creates evidence that may justify a later reviewed
constant change.

The supervised lifecycle is deliberately one process so Forge's current-run
checkpoint scope cannot be reconstructed or forged between stages:

``bootstrap -> first_checkpoint -> terminal -> lab-only profile``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

# The sealed-session checkout must remain fully clean through the identity gate.
# Disable bytecode before importing Forge so this supervisor cannot create its
# own ignored ``forge/__pycache__`` execution surface and then reject itself.
sys.dont_write_bytecode = True

import yaml

# Direct ``python ops/calibration/run_krea_timing_lab.py`` execution otherwise
# places only this leaf directory on sys.path.  Resolve the repository root from
# this immutable script location; no caller-controlled import path is accepted.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from forge import adaptive_timing, krea_runtime
from forge.file_evidence import read_regular_bytes
from forge.tasks import checkpoints
from forge.tasks.integrity import inspect_training_artifact


_PROGRAM = "sn56-krea-timing-lab"
_RECEIPT_KIND = "sn56-krea-timing-lab-receipt"
_RECEIPT_SCHEMA = 2
_GATE_EVENT_KIND = "sn56.week6.friday-h100-timing-evidence-sealed.v2"
_CERTIFICATE_SCOPE = "toolkit-krea-only"
_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_RECORD_BYTES = 1024 * 1024
_MAX_PROFILE_BYTES = 64 * 1024
_MAX_LOG_BYTES = 1024 * 1024 * 1024
_CHECKPOINT_PATTERN = re.compile(r"(?:_|-step)(\d+)\.safetensors$")
_GATE_SESSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_GIT_OBJECT_PATTERN = re.compile(r"[0-9a-f]{40}")


class LabTimingError(RuntimeError):
    """The lab-only timing gate could not produce trustworthy evidence."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes, *, mode: int = 0o440) -> None:
    """Commit one complete output on the destination filesystem."""

    path = path.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _commit_file(temporary: Path, destination: Path, *, mode: int = 0o440) -> None:
    """Commit a completely written same-filesystem temporary file."""

    os.chmod(temporary, mode)
    os.replace(temporary, destination)
    directory_descriptor = os.open(
        destination.parent,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _sha256(path: Path, *, maximum_size: int) -> str:
    payload = read_regular_bytes(
        str(path),
        label=f"lab evidence {path.name}",
        maximum_size=maximum_size,
    )
    return hashlib.sha256(payload).hexdigest()


def _open_sealed_config(path: Path) -> tuple[int, str]:
    """Open and hash the staged config once for descriptor handoff."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LabTimingError("sealed executed config could not be opened") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise LabTimingError("sealed executed config is not a regular file")
        if before.st_size <= 0 or before.st_size > _MAX_CONFIG_BYTES:
            raise LabTimingError("sealed executed config size is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise LabTimingError("sealed executed config was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise LabTimingError("sealed executed config grew while being read")
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        path_state = os.stat(path, follow_symlinks=False)
        if (
            identity_before != identity_after
            or not stat.S_ISREG(path_state.st_mode)
            or (path_state.st_dev, path_state.st_ino)
            != (after.st_dev, after.st_ino)
        ):
            raise LabTimingError("sealed executed config changed while opening")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor, hashlib.sha256(b"".join(chunks)).hexdigest()
    except BaseException:
        os.close(descriptor)
        raise


def _descriptor_path(descriptor: int) -> str:
    for root in ("/proc/self/fd", "/dev/fd"):
        candidate = f"{root}/{descriptor}"
        if os.path.exists(candidate):
            return candidate
    raise LabTimingError("this host cannot expose an inherited config descriptor")


def _positive_number(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LabTimingError(f"{label} must be a positive finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise LabTimingError(f"{label} must be a positive finite number")
    return parsed


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise LabTimingError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LabTimingError(f"{label} must be a positive integer") from exc
    if str(parsed) != str(value).strip() or parsed <= 0:
        raise LabTimingError(f"{label} must be a positive integer")
    return parsed


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise LabTimingError(f"{label} is absent")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabTimingError(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise LabTimingError(f"{label} must be UTC")
    return parsed.astimezone(timezone.utc)


def _declared_gate(args: argparse.Namespace) -> tuple[datetime, datetime]:
    if (
        not isinstance(args.gate_session_id, str)
        or _GATE_SESSION_PATTERN.fullmatch(args.gate_session_id) is None
    ):
        raise LabTimingError("gate session id is invalid")
    started = _parse_utc(args.rental_started_at_utc, "rental start")
    ended = _parse_utc(args.rental_ended_at_utc, "rental end")
    if started >= ended:
        raise LabTimingError("rental window is empty or reversed")
    if not started <= datetime.now(timezone.utc) <= ended:
        raise LabTimingError("current execution is outside the declared rental window")
    return started, ended


def _git_release_identity() -> tuple[str, str]:
    def git(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                [
                    "git",
                    "-C",
                    str(_REPOSITORY_ROOT),
                    *arguments,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception as exc:
            raise LabTimingError("Forge release identity lookup failed") from exc
        if completed.returncode != 0:
            raise LabTimingError("Forge release identity lookup failed")
        return completed.stdout.strip()

    commit = git("rev-parse", "--verify", "HEAD^{commit}")
    tree = git("rev-parse", "--verify", "HEAD^{tree}")
    if (
        _GIT_OBJECT_PATTERN.fullmatch(commit) is None
        or _GIT_OBJECT_PATTERN.fullmatch(tree) is None
    ):
        raise LabTimingError("Forge release identity lookup failed")
    status = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored=matching",
    )
    if status:
        raise LabTimingError(
            "Forge checkout is not fully clean for the sealed gate session"
        )
    return commit, tree


def _absent_save_root(path: Path) -> Path:
    """Resolve a new output root whose parent already exists, without writing."""

    if not path.is_absolute():
        raise LabTimingError("save root must be absolute")
    try:
        path.lstat()
    except FileNotFoundError:
        try:
            parent = path.parent.resolve(strict=True)
        except OSError as exc:
            raise LabTimingError("save root parent must already exist") from exc
        if not parent.is_dir():
            raise LabTimingError("save root parent must be a directory")
        return parent / path.name
    raise LabTimingError("save root must be absent")


def _validate_repo_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or "\x00" in value
    ):
        raise LabTimingError("repository name must be one safe path component")
    return value


def _validate_config_output_binding(
    document: dict[str, Any],
    *,
    save_root: Path,
    repo_name: str,
    terminal_artifact: Path,
) -> Path:
    config_name = document["config"].get("name")
    if config_name != repo_name:
        raise LabTimingError("config repository name differs from --repo-name")
    process = document["config"]["process"][0]
    training_folder = process.get("training_folder")
    if (
        not isinstance(training_folder, str)
        or not training_folder
        or "\x00" in training_folder
        or not os.path.isabs(training_folder)
    ):
        raise LabTimingError("config training_folder must be an absolute path")
    expected_output_root = (
        Path(training_folder).resolve(strict=False) / config_name
    ).resolve(strict=False)
    if expected_output_root != save_root:
        raise LabTimingError(
            "config training_folder/name output differs from --save-root"
        )
    expected_terminal = (expected_output_root / f"{repo_name}.safetensors").resolve(
        strict=False
    )
    if terminal_artifact != expected_terminal:
        raise LabTimingError(
            "terminal artifact differs from the config-derived exact-final path"
        )
    return expected_output_root


def _outside_upload_tree(path: Path, save_root: Path, label: str) -> Path:
    resolved = path.resolve(strict=False)
    root = save_root.resolve(strict=False)
    try:
        inside = os.path.commonpath((resolved, root)) == str(root)
    except ValueError:
        inside = False
    if inside:
        raise LabTimingError(f"{label} must remain outside the uploaded save root")
    return resolved


def _load_config(path: Path) -> tuple[bytes, dict[str, Any], int]:
    raw = read_regular_bytes(
        str(path),
        label="lab Krea config",
        maximum_size=_MAX_CONFIG_BYTES,
    )
    try:
        document = yaml.safe_load(raw)
        process = document["config"]["process"]
        if not isinstance(process, list) or len(process) != 1:
            raise ValueError
        item = process[0]
        if item["model"]["arch"] != "krea2":
            raise ValueError
        planned_steps = item["train"]["steps"]
    except Exception as exc:
        raise LabTimingError("config is not one single-process Krea run") from exc
    if isinstance(planned_steps, bool) or not isinstance(planned_steps, int):
        raise LabTimingError("config planned steps must be a positive integer")
    if planned_steps <= 0:
        raise LabTimingError("config planned steps must be a positive integer")
    return raw, document, planned_steps


def _first_current_checkpoint(
    save_root: Path,
    scope: dict[str, Any],
) -> tuple[int, Path] | None:
    candidates: list[tuple[int, Path]] = []
    for candidate in checkpoints.current_loras(str(save_root), scope):
        try:
            evidence = inspect_training_artifact(candidate)
        except Exception:
            continue
        name = os.path.basename(candidate)
        match = _CHECKPOINT_PATTERN.search(name)
        if match is not None:
            step = int(match.group(1))
        elif name == f"{scope.get('repo')}.safetensors":
            # A short natural run may emit only its exact final. It is still a
            # durable current-run checkpoint; its step comes from the artifact.
            step = evidence.checkpoint_step
        else:
            continue
        if step > 0 and evidence.checkpoint_step == step:
            candidates.append((step, Path(evidence.path)))
    return min(candidates) if candidates else None


def _terminate(process: subprocess.Popen[Any]) -> None:
    """Bounded process-group termination for a lab timeout or failed gate."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=10)


def _receipt_base(args: argparse.Namespace) -> dict[str, Any]:
    try:
        dataset_size: Any = int(args.dataset_size)
    except (TypeError, ValueError):
        dataset_size = args.dataset_size
    return {
        "schema": _RECEIPT_SCHEMA,
        "kind": _RECEIPT_KIND,
        "evidence_scope": "lab-only",
        "bundle": args.bundle,
        "task_id": args.task_id,
        "dataset_size": dataset_size,
        "gate_session_id": args.gate_session_id,
    }


def run_lab(args: argparse.Namespace) -> dict[str, Any]:
    """Supervise one real lab run and return its PASS receipt."""

    # Phase one is strictly read-only.  No checkpoint scope, directory, raw
    # record, quarantine, or log may be created until every path authority and
    # runtime authority check below has succeeded. The save root must not exist;
    # atomic creation after preflight closes the populate-then-quarantine race.
    save_root_argument = Path(args.save_root)
    save_root = _absent_save_root(save_root_argument)
    repo_name = _validate_repo_name(args.repo_name)
    rental_started, rental_ended = _declared_gate(args)
    config_path = _outside_upload_tree(
        Path(args.config), save_root, "config and raw timing record"
    )
    profile_path = _outside_upload_tree(
        Path(args.output_profile), save_root, "timing profile"
    )
    receipt_path = _outside_upload_tree(
        Path(args.output_receipt), save_root, "timing receipt"
    )
    log_path = _outside_upload_tree(
        Path(args.output_log), save_root, "training log"
    )
    gate_log_path = _outside_upload_tree(
        Path(args.output_gate_log), save_root, "Friday H100 gate log"
    )
    sealed_config_path = _outside_upload_tree(
        receipt_path.with_name(f".{receipt_path.name}.executed-config.yaml"),
        save_root,
        "sealed executed config",
    )
    for path, label in (
        (profile_path, "timing profile"),
        (receipt_path, "timing receipt"),
        (log_path, "training log"),
        (gate_log_path, "Friday H100 gate log"),
        (sealed_config_path, "sealed executed config"),
    ):
        if path.exists() or path.is_symlink():
            raise LabTimingError(f"{label} output already exists: {path}")
    outputs = {
        profile_path,
        receipt_path,
        log_path,
        gate_log_path,
        sealed_config_path,
    }
    if len(outputs) != 5:
        raise LabTimingError(
            "profile, receipt, log, gate-log, and sealed-config outputs "
            "must be distinct"
        )

    raw_record_path = Path(str(sealed_config_path) + ".effective-runtime.json")
    if any(
        output in {config_path, raw_record_path}
        for output in outputs
    ):
        raise LabTimingError(
            "profile, receipt, log, gate-log, and sealed-config outputs cannot "
            "alias original config/raw evidence"
        )
    if raw_record_path.exists() or raw_record_path.is_symlink():
        raise LabTimingError(
            f"raw timing record output already exists: {raw_record_path}"
        )

    config_payload, config_document, planned_steps = _load_config(config_path)
    original_config_sha256 = hashlib.sha256(config_payload).hexdigest()
    terminal_artifact = Path(args.terminal_artifact).resolve(strict=False)
    _validate_config_output_binding(
        config_document,
        save_root=save_root,
        repo_name=repo_name,
        terminal_artifact=terminal_artifact,
    )
    dataset_size = _positive_integer(args.dataset_size, "dataset size")
    timeout_seconds = _positive_number(args.timeout_seconds, "timeout seconds")
    poll_seconds = _positive_number(args.poll_seconds, "poll seconds")

    try:
        requested_runtime = Path(args.runtime_dir).resolve(strict=True)
    except OSError as exc:
        raise LabTimingError("selected runtime checkout is unavailable") from exc
    explicit_environment = {
        krea_runtime.BUNDLE_ENV: args.bundle,
        krea_runtime.OWNED_KREA_RUNTIME_DIR_ENV: str(requested_runtime),
    }
    verified_runtime = Path(
        krea_runtime.verify_selected_runtime(
            "krea2",
            args.bundle,
            environ=explicit_environment,
        )
    ).resolve(strict=True)
    if verified_runtime != requested_runtime:
        raise LabTimingError("attested runtime differs from the checkout to execute")
    manifest = krea_runtime.load_capability_manifest(
        model_type="krea2",
        bundle=args.bundle,
        environ=explicit_environment,
    )
    forge_commit, release_tree = _git_release_identity()
    bundle_sha256 = krea_runtime.bundle_contract_sha256(args.bundle)
    dataset_regime = adaptive_timing.dataset_regime(dataset_size)
    # The public API performs a live nvidia-smi query by default. Its value is
    # deliberately labeled operator-attested in the raw/profile schemas.
    accelerator_identity = adaptive_timing.current_accelerator_identity()

    # Close the preflight/use gap for the checkpoint destination.  A competing
    # writer or stale artifact appearing during the read-only checks aborts;
    # begin_run must never quarantine it under this one-shot lab contract.
    if _absent_save_root(save_root_argument) != save_root:
        raise LabTimingError("save root identity changed during preflight")
    for path, label in (
        (profile_path, "timing profile"),
        (receipt_path, "timing receipt"),
        (log_path, "training log"),
        (gate_log_path, "Friday H100 gate log"),
        (sealed_config_path, "sealed executed config"),
        (raw_record_path, "raw timing record"),
    ):
        if path.exists() or path.is_symlink():
            raise LabTimingError(f"{label} output appeared during preflight: {path}")

    # Phase two begins here. Stage exactly the descriptor-captured source bytes
    # under a private derived path. The caller's original config path is never
    # reopened as execution authority after this point.
    _atomic_write(sealed_config_path, config_payload, mode=0o400)
    executed_config_sha256 = _sha256(
        sealed_config_path,
        maximum_size=_MAX_CONFIG_BYTES,
    )
    if executed_config_sha256 != original_config_sha256:
        raise LabTimingError("sealed executed config differs from captured bytes")

    # Atomic save-root creation follows every read-only gate. Every derived path
    # and attested runtime is fixed before checkpoint scope state can exist.
    try:
        os.mkdir(save_root, 0o750)
    except FileExistsError as exc:
        raise LabTimingError("save root appeared during preflight") from exc
    except OSError as exc:
        raise LabTimingError("save root could not be created atomically") from exc
    scope = checkpoints.begin_run(str(save_root), repo_name)
    scope = checkpoints.set_planned_steps(
        str(save_root),
        scope,
        planned_steps,
        model_type="krea2",
    )
    source_run_id = f"{args.task_id}:{scope['attempt_nonce']}"
    krea_runtime.emit_effective_runtime_record(
        config_document,
        "krea2",
        str(sealed_config_path),
        manifest,
        timing_probe=True,
        source_run_id=source_run_id,
        current_dataset_size=dataset_size,
        current_accelerator_identity=accelerator_identity,
        environ=explicit_environment,
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_descriptor, temporary_log_name = tempfile.mkstemp(
        prefix=f".{log_path.name}.",
        suffix=".tmp",
        dir=log_path.parent,
    )
    temporary_log = Path(temporary_log_name)
    pending_log_descriptor: int | None = log_descriptor
    sealed_config_descriptor: int | None = None
    try:
        launch_runtime = Path(
            krea_runtime.verify_selected_runtime(
                "krea2",
                args.bundle,
                environ=explicit_environment,
            )
        ).resolve(strict=True)
        if launch_runtime != verified_runtime:
            raise LabTimingError("selected runtime changed before process launch")
        sealed_config_descriptor, handed_config_sha256 = _open_sealed_config(
            sealed_config_path
        )
        if handed_config_sha256 != executed_config_sha256:
            raise LabTimingError(
                "sealed executed config changed before process launch"
            )
        handed_config_path = _descriptor_path(sealed_config_descriptor)
        training_started_at_utc = _utc_now()
        started = time.monotonic()
        stopped_by_deadline = False
        observed = False
        launch_environment = os.environ.copy()
        launch_environment["PYTHONDONTWRITEBYTECODE"] = "1"
        log_handle = os.fdopen(log_descriptor, "wb")
        pending_log_descriptor = None
        with log_handle:
            process = subprocess.Popen(
                [sys.executable, "run.py", handed_config_path],
                cwd=launch_runtime,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=launch_environment,
                start_new_session=True,
                pass_fds=(sealed_config_descriptor,),
            )
            try:
                while process.poll() is None:
                    if not observed:
                        checkpoint = _first_current_checkpoint(save_root, scope)
                        if checkpoint is not None:
                            checkpoint_step, _checkpoint_path = checkpoint
                            observation = adaptive_timing.emit_bootstrap_first_checkpoint_observation(
                                bundle_id=args.bundle,
                                checkpoint_step=checkpoint_step,
                                elapsed_since_launch_s=time.monotonic() - started,
                                active_planned_steps=planned_steps,
                                event_sink=lambda *_args, **_kwargs: None,
                            )
                            krea_runtime.persist_first_checkpoint_observation(
                                str(sealed_config_path), observation
                            )
                            observed = True
                    if time.monotonic() - started >= timeout_seconds:
                        stopped_by_deadline = True
                        _terminate(process)
                        break
                    time.sleep(poll_seconds)
                returncode = process.wait(timeout=10)
            except BaseException:
                _terminate(process)
                raise
            log_handle.flush()
            os.fsync(log_handle.fileno())
    finally:
        if sealed_config_descriptor is not None:
            os.close(sealed_config_descriptor)
        if pending_log_descriptor is not None:
            os.close(pending_log_descriptor)
        if temporary_log.exists():
            _commit_file(temporary_log, log_path)
    elapsed_seconds = time.monotonic() - started

    if not observed:
        checkpoint = _first_current_checkpoint(save_root, scope)
        if checkpoint is None:
            raise LabTimingError("run produced no durable current-run checkpoint")
        checkpoint_step, _checkpoint_path = checkpoint
        observation = adaptive_timing.emit_bootstrap_first_checkpoint_observation(
            bundle_id=args.bundle,
            checkpoint_step=checkpoint_step,
            elapsed_since_launch_s=elapsed_seconds,
            active_planned_steps=planned_steps,
            event_sink=lambda *_args, **_kwargs: None,
        )
        krea_runtime.persist_first_checkpoint_observation(
            str(sealed_config_path), observation
        )

    krea_runtime.persist_training_completion_observation(
        str(sealed_config_path),
        artifact_path=str(terminal_artifact),
        save_root=str(save_root),
        scope=scope,
        training_elapsed_seconds=elapsed_seconds,
        returncode=returncode,
        stopped_by_deadline=stopped_by_deadline,
    )
    raw_record_produced_at_utc = _utc_now()
    profile = adaptive_timing.produce_profile_document(
        str(raw_record_path),
        source_run_id=source_run_id,
        bundle_id=args.bundle,
        model_type="krea2",
        measured_dataset_size=dataset_size,
    )
    profile_produced_at_utc = profile["provenance"]["measured_at_utc"]
    _atomic_write(profile_path, _canonical_bytes(profile))
    raw_record = json.loads(
        read_regular_bytes(
            str(raw_record_path),
            label="lab raw timing record",
            maximum_size=_MAX_RECORD_BYTES,
        ).decode("utf-8")
    )
    completion = raw_record["training_completion_observation"]

    sealed_at_utc = _utc_now()
    training_started = _parse_utc(training_started_at_utc, "training start")
    raw_produced = _parse_utc(
        raw_record_produced_at_utc, "raw-record production time"
    )
    profile_produced = _parse_utc(
        profile_produced_at_utc, "profile production time"
    )
    sealed = _parse_utc(sealed_at_utc, "gate seal time")
    if not (
        rental_started
        <= training_started
        <= raw_produced
        <= profile_produced
        <= sealed
        <= rental_ended
    ):
        raise LabTimingError("timing evidence timestamps fall outside rental window")

    profile_file_sha256 = _sha256(
        profile_path,
        maximum_size=_MAX_PROFILE_BYTES,
    )
    raw_record_file_sha256 = _sha256(
        raw_record_path,
        maximum_size=_MAX_RECORD_BYTES,
    )
    event = {
        "event": _GATE_EVENT_KIND,
        "gate_session_id": args.gate_session_id,
        "source_run_id": source_run_id,
        "rental_started_at_utc": args.rental_started_at_utc,
        "rental_ended_at_utc": args.rental_ended_at_utc,
        "training_started_at_utc": training_started_at_utc,
        "raw_record_produced_at_utc": raw_record_produced_at_utc,
        "profile_produced_at_utc": profile_produced_at_utc,
        "sealed_at_utc": sealed_at_utc,
        "profile_file_sha256": profile_file_sha256,
        "raw_record_file_sha256": raw_record_file_sha256,
        "terminal_artifact_file_sha256": completion["artifact_sha256"],
        "profile_semantic_sha256": profile["profile_sha256"],
        "raw_record_semantic_sha256": raw_record["record_sha256"],
        "forge_commit": forge_commit,
        "release_tree": release_tree,
        "certificate_scope": _CERTIFICATE_SCOPE,
        "bundle_id": args.bundle,
        "bundle_sha256": bundle_sha256,
        "model_type": "krea2",
        "current_dataset_size": dataset_size,
        "dataset_regime": dataset_regime,
        "accelerator_identity": profile["provenance"]["accelerator_identity"],
    }
    event_payload = _canonical_bytes(event)
    _atomic_write(gate_log_path, event_payload)
    gate_log_file_sha256 = _sha256(
        gate_log_path,
        maximum_size=_MAX_PROFILE_BYTES,
    )

    receipt = {
        **_receipt_base(args),
        "state": "PASS",
        "source_run_id": source_run_id,
        "gate_session_id": args.gate_session_id,
        "planned_steps": planned_steps,
        "completed_steps": profile["measurement"]["completed_steps"],
        "accelerator_identity": profile["provenance"]["accelerator_identity"],
        "accelerator_identity_evidence": profile["provenance"][
            "accelerator_identity_evidence"
        ],
        "config": {
            "original": {
                "path": str(config_path),
                "captured_sha256": original_config_sha256,
            },
            "executed": {
                "path": str(sealed_config_path),
                "sha256": _sha256(
                    sealed_config_path,
                    maximum_size=_MAX_CONFIG_BYTES,
                ),
            },
        },
        "raw_record": {
            "path": str(raw_record_path),
            "sha256": raw_record_file_sha256,
            "semantic_sha256": raw_record["record_sha256"],
        },
        "profile": {
            "path": str(profile_path),
            "sha256": profile_file_sha256,
            "semantic_sha256": profile["profile_sha256"],
        },
        "terminal_artifact": {
            "path": str(terminal_artifact),
            "sha256": completion["artifact_sha256"],
            "size_bytes": completion["artifact_size_bytes"],
            "checkpoint_step": completion["artifact_checkpoint_step"],
        },
        "training_log": {
            "path": str(log_path),
            "sha256": _sha256(log_path, maximum_size=_MAX_LOG_BYTES),
        },
        "friday_h100_gate_log": {
            "path": str(gate_log_path),
            "sha256": gate_log_file_sha256,
            "event": _GATE_EVENT_KIND,
            "event_sha256": hashlib.sha256(event_payload).hexdigest(),
        },
        "rental_window": {
            "started_at_utc": args.rental_started_at_utc,
            "ended_at_utc": args.rental_ended_at_utc,
        },
        "evidence_timestamps": {
            "training_started_at_utc": training_started_at_utc,
            "raw_record_produced_at_utc": raw_record_produced_at_utc,
            "profile_produced_at_utc": profile_produced_at_utc,
            "sealed_at_utc": sealed_at_utc,
        },
        "forge": {
            "commit": forge_commit,
            "tree": release_tree,
        },
        "certificate_scope": _CERTIFICATE_SCOPE,
        "scope": {
            "bundle_id": args.bundle,
            "bundle_sha256": bundle_sha256,
            "model_type": "krea2",
            "current_dataset_size": dataset_size,
            "dataset_regime": dataset_regime,
            "accelerator_identity": profile["provenance"][
                "accelerator_identity"
            ],
        },
    }
    _atomic_write(receipt_path, _canonical_bytes(receipt))
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_PROGRAM,
        description=(
            "LAB ONLY: supervise one real Krea timing run and atomically emit "
            "its host-bound raw-record/profile evidence package."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="pre-generated Krea YAML")
    parser.add_argument(
        "--runtime-dir",
        required=True,
        help="attested owned Krea ai-toolkit checkout",
    )
    parser.add_argument("--save-root", required=True, help="isolated checkpoint root")
    parser.add_argument("--repo-name", required=True, help="config/output repository name")
    parser.add_argument("--task-id", required=True, help="stable lab run identifier")
    parser.add_argument("--dataset-size", required=True, help="training pair count")
    parser.add_argument(
        "--bundle",
        required=True,
        choices=tuple(
            sorted(krea_runtime.KNOWN_BUNDLES - {krea_runtime.INCUMBENT_BUNDLE})
        ),
        help="explicit experimental Krea bundle",
    )
    parser.add_argument(
        "--terminal-artifact",
        required=True,
        help="expected exact-final safetensors path",
    )
    parser.add_argument("--output-profile", required=True, help="atomic profile JSON")
    parser.add_argument("--output-receipt", required=True, help="atomic receipt JSON")
    parser.add_argument("--output-log", required=True, help="exclusive training log")
    parser.add_argument(
        "--output-gate-log",
        required=True,
        help="atomic Friday H100 timing-seal JSONL outside the upload tree",
    )
    parser.add_argument(
        "--gate-session-id",
        required=True,
        help="declared sealed Friday H100 gate-session identifier",
    )
    parser.add_argument(
        "--rental-started-at-utc",
        required=True,
        help="declared UTC start bound for the H100 rental session",
    )
    parser.add_argument(
        "--rental-ended-at-utc",
        required=True,
        help="declared UTC end bound for the H100 rental session",
    )
    parser.add_argument(
        "--timeout-seconds",
        default="10800",
        help="hard lab timeout; a timeout cannot produce a PASS profile",
    )
    parser.add_argument(
        "--poll-seconds",
        default="2",
        help="durable-checkpoint polling interval",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        receipt = run_lab(args)
    except Exception as exc:
        try:
            receipt_path = Path(args.output_receipt).resolve(strict=False)
            save_root = Path(args.save_root).resolve(strict=False)
            receipt_path = _outside_upload_tree(
                receipt_path,
                save_root,
                "timing receipt",
            )
            if not receipt_path.exists() and not receipt_path.is_symlink():
                failure = {
                    **_receipt_base(args),
                    "state": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                _atomic_write(receipt_path, _canonical_bytes(failure))
        except Exception:
            pass
        print(f"SN56_KREA_TIMING_LAB=FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "SN56_KREA_TIMING_LAB=PASS "
        f"profile_sha256={receipt['profile']['semantic_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
