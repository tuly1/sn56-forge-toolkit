#!/usr/bin/env python3
"""Validate the procedural provenance envelope for a Lane-A timing profile.

This validator deliberately calls the evidence ``operator-attested``.  It
checks internal consistency, exact-file hashes, terminal-artifact binding,
dataset/accelerator-scoped profile structure, and a sealed Friday H100 gate-log
cross-reference.  It does not claim to authenticate elapsed time independently
of the operator and gate harness that produced the record.

The gate log is JSON Lines.  The matching event must contain these fields:

``event``
    ``sn56.week6.friday-h100-timing-evidence-sealed.v1``
``gate_session_id`` / ``source_run_id``
    Exact values supplied to this validator.
``rental_started_at_utc`` / ``rental_ended_at_utc``
    Exact declared session bounds.
``raw_record_produced_at_utc`` / ``profile_produced_at_utc`` /
``sealed_at_utc``
    Ordered UTC timestamps inside the rental window.
``profile_file_sha256`` / ``raw_record_file_sha256`` /
``terminal_artifact_file_sha256``
    Exact hashes of the three consumed files.
``profile_semantic_sha256`` / ``raw_record_semantic_sha256``
    Canonical semantic hashes embedded in the profile and raw record.
``forge_commit`` / ``bundle_id`` / ``bundle_sha256`` / ``model_type``
``current_dataset_size`` / ``dataset_regime`` / ``accelerator_identity``
    The exact Forge contract and scope that the event supervised.

Extra event fields are permitted so the Friday harness can carry its own host,
image, command, and evidence identities.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


EVENT_KIND = "sn56.week6.friday-h100-timing-evidence-sealed.v2"
RECEIPT_KIND = "sn56.week6.operator-attested-timing-provenance.v2"
PROFILE_KIND = "forge-operator-attested-throughput-profile"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SOURCE_RUN_RE = re.compile(r".+:[0-9a-f]{32}")
SESSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_AT_FDCWD = -100
_OPENAT2_SYSCALL = 437
_RESOLVE_NO_SYMLINKS = 0x04
_OPENAT2_MACHINES = frozenset(
    {"aarch64", "arm64", "riscv64", "x86_64", "amd64"}
)


class ProvenanceError(RuntimeError):
    """The timing provenance envelope is incomplete or inconsistent."""


class _OpenHow(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint64),
        ("mode", ctypes.c_uint64),
        ("resolve", ctypes.c_uint64),
    ]


def openat2_no_symlinks(path: str, flags: int) -> int:
    encoded = os.fsencode(path)
    if b"\x00" in encoded:
        raise ValueError("evidence path contains an embedded NUL")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    how = _OpenHow(flags=flags, mode=0, resolve=_RESOLVE_NO_SYMLINKS)
    result = libc.syscall(
        ctypes.c_long(_OPENAT2_SYSCALL),
        ctypes.c_int(_AT_FDCWD),
        ctypes.c_char_p(encoded),
        ctypes.byref(how),
        ctypes.c_size_t(ctypes.sizeof(how)),
    )
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), path)
    return int(result)


def open_evidence_path(path: str, flags: int) -> int:
    """Use full-path Linux protection or an honest final-component fallback."""

    machine = os.uname().machine.lower() if hasattr(os, "uname") else ""
    if sys.platform.startswith("linux") and machine in _OPENAT2_MACHINES:
        try:
            return openat2_no_symlinks(path, flags)
        except OSError as exc:
            if exc.errno != errno.ENOSYS:
                raise
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise OSError(
            errno.ENOTSUP,
            "final-component symlink protection is unavailable",
            path,
        )
    return os.open(path, flags | nofollow)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_sha256_without(value: dict[str, Any], field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def forge_runtime_record_sha256_without(
    value: dict[str, Any], field: str
) -> str:
    """Match Forge runtime records: ASCII JSON plus one trailing newline."""

    body = dict(value)
    body.pop(field, None)
    payload = (
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_sha(value: str, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ProvenanceError(f"{label} must be a lowercase SHA-256")
    return value


def parse_utc(value: str, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ProvenanceError(f"{label} is absent")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProvenanceError(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ProvenanceError(f"{label} must be UTC")
    return parsed.astimezone(timezone.utc)


def checked_path(path: str, label: str) -> str:
    if not isinstance(path, str) or not os.path.isabs(path):
        raise ProvenanceError(f"{label} path must be absolute")
    absolute = os.path.abspath(path)
    if os.path.normpath(path) != path:
        raise ProvenanceError(f"{label} path contains lexical indirection")
    return absolute


def load_forge_contract(
    repository: str,
    expected_commit: str,
    *,
    require_clean: bool,
):
    """Load the exact Forge contract implementation named by the certificate."""

    if GIT_COMMIT_RE.fullmatch(expected_commit or "") is None:
        raise ProvenanceError("Forge commit is invalid")
    if not isinstance(repository, str) or not os.path.isabs(repository):
        raise ProvenanceError("Forge repository path must be absolute")
    root = os.path.abspath(repository)
    try:
        resolved = os.path.realpath(root, strict=True)
    except OSError as exc:
        raise ProvenanceError("Forge repository is unavailable") from exc
    if resolved != root or not os.path.isdir(root):
        raise ProvenanceError("Forge repository is symlinked or not a directory")

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", root, *arguments],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            raise ProvenanceError(
                f"Forge repository check failed: {' '.join(arguments)}"
            )
        return completed.stdout.strip()

    if git("rev-parse", "HEAD") != expected_commit:
        raise ProvenanceError("Forge repository HEAD differs from certificate pin")
    if os.path.realpath(git("rev-parse", "--show-toplevel")) != root:
        raise ProvenanceError("Forge path is not the exact repository root")
    if require_clean and git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored=matching",
    ):
        raise ProvenanceError(
            "Forge repository has changed, untracked, or ignored surfaces"
        )

    sys.dont_write_bytecode = True
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from forge import adaptive_timing
    except Exception as exc:
        raise ProvenanceError("exact Forge timing contract could not be imported") from exc
    loaded_path = os.path.realpath(adaptive_timing.__file__)
    expected_prefix = os.path.join(root, "forge") + os.sep
    if not loaded_path.startswith(expected_prefix):
        raise ProvenanceError("a different Forge package was imported")
    return adaptive_timing


def hash_regular_file(
    path: str,
    *,
    label: str,
    expected_sha256: str,
    maximum_bytes: int | None = None,
    capture_bytes: bool = False,
) -> tuple[str, int, os.stat_result, bytes | None]:
    checked = checked_path(path, label)
    expected = require_sha(expected_sha256, f"expected {label} hash")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = open_evidence_path(checked, flags)
    except OSError as exc:
        raise ProvenanceError(f"{label} could not be opened") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise ProvenanceError(f"{label} must be a nonempty regular file")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise ProvenanceError(f"{label} exceeds its size limit")
        digest = hashlib.sha256()
        consumed = 0
        captured: list[bytes] | None = [] if capture_bytes else None
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            consumed += len(block)
            if captured is not None:
                captured.append(block)
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
        if consumed != before.st_size or identity_before != identity_after:
            raise ProvenanceError(f"{label} changed while it was hashed")
        actual = digest.hexdigest()
        if actual != expected:
            raise ProvenanceError(
                f"{label} hash mismatch: expected={expected} actual={actual}"
            )
        payload = b"".join(captured) if captured is not None else None
        return checked, consumed, before, payload
    finally:
        os.close(descriptor)


def load_json_file(
    path: str,
    *,
    label: str,
    expected_sha256: str,
    maximum_bytes: int,
) -> tuple[str, dict[str, Any], int, bytes]:
    checked, size, _, payload = hash_regular_file(
        path,
        label=label,
        expected_sha256=expected_sha256,
        maximum_bytes=maximum_bytes,
        capture_bytes=True,
    )
    try:
        value = json.loads(payload)
    except Exception as exc:
        raise ProvenanceError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ProvenanceError(f"{label} must contain a JSON object")
    return checked, value, size, payload


def inspect_hashed_training_artifact(
    path: str,
    *,
    expected_sha256: str,
):
    """Inspect and hash one terminal artifact through one opened descriptor."""

    from forge.tasks.integrity import inspect_training_artifact_fd

    checked = checked_path(path, "terminal artifact")
    expected = require_sha(expected_sha256, "expected terminal artifact hash")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = open_evidence_path(checked, flags)
    except OSError as exc:
        raise ProvenanceError("terminal artifact could not be opened") from exc
    try:
        evidence = inspect_training_artifact_fd(descriptor, path_label=checked)
    except Exception as exc:
        raise ProvenanceError(
            "terminal artifact is not a descriptor-bound loadable checkpoint"
        ) from exc
    finally:
        os.close(descriptor)
    if evidence.sha256 != expected:
        raise ProvenanceError(
            "terminal artifact hash mismatch: "
            f"expected={expected} actual={evidence.sha256}"
        )
    return checked, evidence


def file_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def validate(
    args: argparse.Namespace,
    *,
    _after_capture: Callable[[], None] | None = None,
    _before_final_binding: Callable[[], None] | None = None,
) -> dict[str, Any]:
    source_run_id = args.source_run_id
    if SOURCE_RUN_RE.fullmatch(source_run_id or "") is None:
        raise ProvenanceError("source run id is invalid")
    if SESSION_RE.fullmatch(args.gate_session_id or "") is None:
        raise ProvenanceError("gate session id is invalid")
    if GIT_COMMIT_RE.fullmatch(args.release_tree or "") is None:
        raise ProvenanceError("release tree is invalid")
    if args.certificate_scope != "toolkit-krea-only":
        raise ProvenanceError("certificate scope is invalid")

    rental_start = parse_utc(args.rental_started_at_utc, "rental start")
    rental_end = parse_utc(args.rental_ended_at_utc, "rental end")
    if rental_start >= rental_end:
        raise ProvenanceError("rental window is empty or reversed")
    try:
        expected_dataset_size = int(args.current_dataset_size)
    except (TypeError, ValueError) as exc:
        raise ProvenanceError("current dataset size is invalid") from exc
    if expected_dataset_size <= 0:
        raise ProvenanceError("current dataset size is invalid")
    expected_bundle_sha = require_sha(
        args.bundle_sha256, "expected bundle hash"
    )
    adaptive_timing = load_forge_contract(
        args.forge_repository,
        args.forge_commit,
        require_clean=not getattr(args, "allow_dirty_forge", False),
    )

    profile_path, profile, profile_size, profile_payload = load_json_file(
        args.profile,
        label="timing profile",
        expected_sha256=args.profile_file_sha256,
        maximum_bytes=64 * 1024,
    )
    raw_path, raw_record, raw_size, raw_payload = load_json_file(
        args.raw_record,
        label="raw runtime record",
        expected_sha256=args.raw_record_file_sha256,
        maximum_bytes=1024 * 1024,
    )
    artifact_path, artifact_evidence = inspect_hashed_training_artifact(
        args.terminal_artifact,
        expected_sha256=args.terminal_artifact_file_sha256,
    )
    artifact_size = artifact_evidence.size_bytes
    archived_artifact_path, archived_artifact_size, _, _ = hash_regular_file(
        args.archived_terminal_artifact,
        label="archived terminal artifact",
        expected_sha256=args.terminal_artifact_file_sha256,
    )
    if archived_artifact_size != artifact_size:
        raise ProvenanceError("archived terminal artifact size mismatch")
    gate_path, gate_size, _, gate_payload = hash_regular_file(
        args.gate_log,
        label="Friday gate log",
        expected_sha256=args.gate_log_file_sha256,
        maximum_bytes=16 * 1024 * 1024,
        capture_bytes=True,
    )
    if _after_capture is not None:
        _after_capture()

    if (
        profile.get("schema") != 4
        or profile.get("kind") != PROFILE_KIND
        or profile.get("evidence_scope") != "lab-only"
    ):
        raise ProvenanceError("unsupported operator-attested timing profile contract")
    if raw_record.get("schema") != 5:
        raise ProvenanceError("unsupported raw runtime record schema")

    try:
        profile_provenance = profile["provenance"]
        profile_semantic = require_sha(
            profile["profile_sha256"], "profile semantic hash"
        )
        raw_semantic = require_sha(
            raw_record["record_sha256"], "raw-record semantic hash"
        )
    except (KeyError, TypeError) as exc:
        raise ProvenanceError("profile or raw-record provenance is incomplete") from exc
    if not isinstance(profile_provenance, dict):
        raise ProvenanceError("profile provenance must be an object")
    if canonical_sha256_without(profile, "profile_sha256") != profile_semantic:
        raise ProvenanceError("profile canonical semantic hash mismatch")
    if forge_runtime_record_sha256_without(
        raw_record, "record_sha256"
    ) != raw_semantic:
        raise ProvenanceError("raw-record canonical semantic hash mismatch")
    if profile_provenance.get("source_run_id") != source_run_id:
        raise ProvenanceError("profile source run id mismatch")
    if raw_record.get("source_run_id") != source_run_id:
        raise ProvenanceError("raw-record source run id mismatch")
    if profile_provenance.get("source_record_sha256") != args.raw_record_file_sha256:
        raise ProvenanceError("profile does not bind the exact raw-record bytes")
    profile_time = parse_utc(
        profile_provenance.get("measured_at_utc"), "profile production time"
    )

    if raw_record.get("lifecycle") != "terminal":
        raise ProvenanceError("raw record is not terminal")
    completion = raw_record.get("training_completion_observation")
    if not isinstance(completion, dict):
        raise ProvenanceError("raw record lacks terminal artifact evidence")
    if completion.get("artifact_sha256") != args.terminal_artifact_file_sha256:
        raise ProvenanceError("raw record terminal artifact hash mismatch")
    if completion.get("artifact_size_bytes") != artifact_size:
        raise ProvenanceError("raw record terminal artifact size mismatch")
    if completion.get("artifact_loadable") is not True:
        raise ProvenanceError("raw record does not mark the artifact loadable")
    recorded_artifact = completion.get("artifact_path")
    if not isinstance(recorded_artifact, str):
        raise ProvenanceError("raw record terminal artifact path is absent")
    checked_path(recorded_artifact, "recorded terminal artifact")

    try:
        loaded_profile = adaptive_timing.load_profile_bytes(
            profile_payload,
            source_record_bytes=raw_payload,
            expected_bundle_id=args.bundle_id,
            expected_bundle_sha256=expected_bundle_sha,
            expected_model_type=args.model_type,
            current_dataset_size=expected_dataset_size,
            expected_dataset_regime=args.dataset_regime,
            expected_accelerator_identity=args.accelerator_identity,
            terminal_artifact_evidence=artifact_evidence,
            require_artifact_file_identity=False,
        )
    except Exception as exc:
        raise ProvenanceError(
            f"exact Forge timing contract rejected the package: {exc}"
        ) from exc
    if loaded_profile.source_run_id != source_run_id:
        raise ProvenanceError("Forge-loaded profile source run id mismatch")

    matches: list[tuple[int, dict[str, Any]]] = []
    try:
        gate_text = gate_payload.decode("utf-8")
        for line_number, line in enumerate(gate_text.splitlines(), 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if isinstance(event, dict) and event.get("event") == EVENT_KIND:
                if (
                    event.get("gate_session_id") == args.gate_session_id
                    and event.get("source_run_id") == source_run_id
                    and event.get("profile_file_sha256")
                    == args.profile_file_sha256
                    and event.get("raw_record_file_sha256")
                    == args.raw_record_file_sha256
                    and event.get("terminal_artifact_file_sha256")
                    == args.terminal_artifact_file_sha256
                    and event.get("profile_semantic_sha256") == profile_semantic
                    and event.get("raw_record_semantic_sha256") == raw_semantic
                    and event.get("forge_commit") == args.forge_commit
                    and event.get("release_tree") == args.release_tree
                    and event.get("certificate_scope")
                    == args.certificate_scope
                    and event.get("bundle_id") == args.bundle_id
                    and event.get("bundle_sha256") == expected_bundle_sha
                    and event.get("model_type") == args.model_type
                    and event.get("current_dataset_size")
                    == expected_dataset_size
                    and event.get("dataset_regime") == args.dataset_regime
                    and event.get("accelerator_identity")
                    == args.accelerator_identity
                ):
                    matches.append((line_number, event))
    except Exception as exc:
        raise ProvenanceError("Friday gate log is not valid JSON Lines") from exc
    if len(matches) != 1:
        raise ProvenanceError(
            f"Friday gate log has {len(matches)} matching timing seal events"
        )
    event_line, event = matches[0]
    if event.get("rental_started_at_utc") != args.rental_started_at_utc:
        raise ProvenanceError("gate event rental start differs from declaration")
    if event.get("rental_ended_at_utc") != args.rental_ended_at_utc:
        raise ProvenanceError("gate event rental end differs from declaration")
    raw_time = parse_utc(
        event.get("raw_record_produced_at_utc"), "raw-record production time"
    )
    event_profile_time = parse_utc(
        event.get("profile_produced_at_utc"), "gate profile production time"
    )
    sealed_time = parse_utc(event.get("sealed_at_utc"), "gate seal time")
    training_started = parse_utc(
        event.get("training_started_at_utc"), "training start time"
    )
    if event_profile_time != profile_time:
        raise ProvenanceError("profile timestamp differs from gate-log event")
    if not (
        rental_start
        <= training_started
        <= raw_time
        <= event_profile_time
        <= sealed_time
        <= rental_end
    ):
        raise ProvenanceError("timing evidence timestamps fall outside the rental window")
    elapsed = completion.get("training_elapsed_seconds")
    observed_window = (raw_time - training_started).total_seconds()
    duration_tolerance = max(5.0, observed_window * 0.10)
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or float(elapsed) <= 0
        or abs(float(elapsed) - observed_window) > duration_tolerance
    ):
        raise ProvenanceError(
            "training duration is inconsistent with the gate-recorded execution window"
        )

    if _before_final_binding is not None:
        _before_final_binding()
    for current_path, label, expected_hash in (
        (profile_path, "timing profile", args.profile_file_sha256),
        (raw_path, "raw runtime record", args.raw_record_file_sha256),
        (
            artifact_path,
            "terminal artifact",
            args.terminal_artifact_file_sha256,
        ),
        (
            archived_artifact_path,
            "archived terminal artifact",
            args.terminal_artifact_file_sha256,
        ),
        (gate_path, "Friday gate log", args.gate_log_file_sha256),
    ):
        assert_regular_file_hash(
            current_path,
            label=f"final {label} binding",
            expected_sha256=expected_hash,
        )

    receipt: dict[str, Any] = {
        "schema": 2,
        "kind": RECEIPT_KIND,
        "state": "PASS",
        "evidence_class": "operator-attested",
        "claim_limit": "not-independent-proof-of-elapsed-time-or-hardware-measurement",
        "gate_session_id": args.gate_session_id,
        "source_run_id": source_run_id,
        "forge": {
            "repository": args.forge_repository,
            "commit": args.forge_commit,
            "tree": args.release_tree,
        },
        "certificate_scope": args.certificate_scope,
        "scope": {
            "bundle_id": args.bundle_id,
            "bundle_sha256": expected_bundle_sha,
            "model_type": args.model_type,
            "current_dataset_size": expected_dataset_size,
            "dataset_regime": args.dataset_regime,
            "accelerator_identity": args.accelerator_identity,
        },
        "rental_window": {
            "started_at_utc": args.rental_started_at_utc,
            "ended_at_utc": args.rental_ended_at_utc,
        },
        "gate_event": {
            "line": event_line,
            "training_started_at_utc": event["training_started_at_utc"],
            "raw_record_produced_at_utc": event["raw_record_produced_at_utc"],
            "profile_produced_at_utc": event["profile_produced_at_utc"],
            "sealed_at_utc": event["sealed_at_utc"],
        },
        "files": {
            "profile": {
                "path": profile_path,
                "bytes": profile_size,
                "file_sha256": args.profile_file_sha256,
                "semantic_sha256": profile_semantic,
            },
            "raw_record": {
                "path": raw_path,
                "bytes": raw_size,
                "file_sha256": args.raw_record_file_sha256,
                "semantic_sha256": raw_semantic,
            },
            "terminal_artifact": {
                "path": artifact_path,
                "bytes": artifact_size,
                "file_sha256": args.terminal_artifact_file_sha256,
            },
            "archived_terminal_artifact": {
                "path": archived_artifact_path,
                "bytes": archived_artifact_size,
                "file_sha256": args.terminal_artifact_file_sha256,
            },
            "gate_log": {
                "path": gate_path,
                "bytes": gate_size,
                "file_sha256": args.gate_log_file_sha256,
            },
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
    return receipt


def write_receipt(path: str, receipt: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(receipt) + b"\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o440)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def stage_validated_file(
    source: str,
    destination: str,
    *,
    label: str,
    expected_sha256: str,
    maximum_bytes: int | None = None,
    _after_open: Callable[[], None] | None = None,
) -> str:
    """Forward exactly the descriptor-opened bytes into a private stage.

    Replacing ``source`` after the open cannot change what is forwarded. Every
    later consumer must still call :func:`assert_regular_file_hash`, so a swap
    of the staged path is an explicit failure rather than a second path trust.
    """

    checked = checked_path(source, label)
    expected = require_sha(expected_sha256, f"expected {label} hash")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        source_fd = open_evidence_path(checked, flags)
    except OSError as exc:
        raise ProvenanceError(f"{label} could not be opened") from exc
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    output_fd: int | None = None
    temporary: str | None = None
    try:
        opened = os.fstat(source_fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size <= 0:
            raise ProvenanceError(f"{label} must be a nonempty regular file")
        if maximum_bytes is not None and opened.st_size > maximum_bytes:
            raise ProvenanceError(f"{label} exceeds its size limit")
        if _after_open is not None:
            _after_open()
        output_fd, temporary = tempfile.mkstemp(
            prefix=f".{destination_path.name}.", dir=destination_path.parent
        )
        digest = hashlib.sha256()
        consumed = 0
        while True:
            block = os.read(source_fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            consumed += len(block)
            view = memoryview(block)
            while view:
                written = os.write(output_fd, view)
                if written <= 0:
                    raise ProvenanceError(f"{label} staging write failed")
                view = view[written:]
        closed_over = os.fstat(source_fd)
        if consumed != opened.st_size or file_identity(opened) != file_identity(
            closed_over
        ):
            raise ProvenanceError(f"{label} changed while it was staged")
        actual = digest.hexdigest()
        if actual != expected:
            raise ProvenanceError(
                f"{label} hash mismatch: expected={expected} actual={actual}"
            )
        os.fsync(output_fd)
        os.close(output_fd)
        output_fd = None
        os.chmod(temporary, 0o440)
        os.replace(temporary, destination_path)
        temporary = None
    finally:
        os.close(source_fd)
        if output_fd is not None:
            os.close(output_fd)
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    assert_regular_file_hash(
        str(destination_path), label=f"staged {label}", expected_sha256=expected
    )
    return expected


def assert_regular_file_hash(
    path: str, *, label: str, expected_sha256: str
) -> None:
    hash_regular_file(
        path,
        label=label,
        expected_sha256=expected_sha256,
    )


def parse_result_env_bytes(payload: bytes) -> dict[str, str]:
    """Parse a data-only result.env without shell evaluation."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProvenanceError("delegated result.env is not UTF-8") from exc
    result: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line or "=" not in line:
            raise ProvenanceError(
                f"delegated result.env line {line_number} is malformed"
            )
        key, value = line.split("=", 1)
        if re.fullmatch(r"[a-z][a-z0-9_]*", key) is None or not value:
            raise ProvenanceError(
                f"delegated result.env line {line_number} is malformed"
            )
        if key in result:
            raise ProvenanceError(f"delegated result.env duplicates {key}")
        result[key] = value
    return result


def assert_delegated_result(
    path: str,
    *,
    release_commit: str,
    release_tree: str,
    certificate_scope: str,
) -> dict[str, str]:
    payload = read_small_regular_file(path, "delegated result.env", 64 * 1024)
    result = parse_result_env_bytes(payload)
    expected = {
        "schema": "sn56.week5.final-release-cert.v2",
        "state": "PASS",
        "source_commit": release_commit,
        "source_tree": release_tree,
        "certificate_scope": certificate_scope,
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise ProvenanceError(
                f"delegated result.env {field} differs from release authority"
            )
    return result


def read_small_regular_file(path: str, label: str, maximum_bytes: int) -> bytes:
    checked = checked_path(path, label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = open_evidence_path(checked, flags)
    except OSError as exc:
        raise ProvenanceError(f"{label} could not be opened") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_size <= 0
            or opened.st_size > maximum_bytes
        ):
            raise ProvenanceError(f"{label} must be a bounded nonempty regular file")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise ProvenanceError(f"{label} was truncated")
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise ProvenanceError(f"{label} grew while it was read")
        if file_identity(opened) != file_identity(os.fstat(descriptor)):
            raise ProvenanceError(f"{label} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def assert_pass_receipt(
    path: str,
    *,
    release_commit: str,
    release_tree: str,
    certificate_scope: str,
    expected_file_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    payload = read_small_regular_file(path, "timing receipt", 256 * 1024)
    try:
        receipt = json.loads(payload)
    except Exception as exc:
        raise ProvenanceError("timing receipt is not valid JSON") from exc
    if not isinstance(receipt, dict):
        raise ProvenanceError("timing receipt is not an object")
    declared = receipt.get("receipt_sha256")
    forge_identity = receipt.get("forge")
    expected_forge = None
    if isinstance(forge_identity, dict):
        repository = forge_identity.get("repository")
        if not isinstance(repository, str) or not os.path.isabs(repository):
            repository = None
        expected_forge = {
            "repository": repository,
            "commit": release_commit,
            "tree": release_tree,
        }
    if (
        receipt.get("schema") != 2
        or receipt.get("kind") != RECEIPT_KIND
        or receipt.get("state") != "PASS"
        or receipt.get("evidence_class") != "operator-attested"
        or receipt.get("claim_limit")
        != "not-independent-proof-of-elapsed-time-or-hardware-measurement"
        or receipt.get("certificate_scope") != certificate_scope
        or forge_identity != expected_forge
        or canonical_sha256_without(receipt, "receipt_sha256") != declared
    ):
        raise ProvenanceError("timing receipt is not an authoritative PASS")
    if expected_file_hashes is not None:
        files = receipt.get("files")
        if not isinstance(files, dict):
            raise ProvenanceError("timing receipt file bindings are absent")
        for name, expected_hash in expected_file_hashes.items():
            item = files.get(name)
            if not isinstance(item, dict) or item.get("file_sha256") != expected_hash:
                raise ProvenanceError(f"timing receipt {name} hash differs")
    return receipt


def run_pinned_validator(
    validator_path: str,
    validator_sha256: str,
    arguments: Sequence[str],
    *,
    receipt_path: str,
    release_commit: str,
    release_tree: str,
    certificate_scope: str,
) -> subprocess.CompletedProcess[str]:
    """Testable authority runner: zero scripts and missing receipts are fatal."""

    with tempfile.TemporaryDirectory() as directory:
        staged = os.path.join(directory, "validator.py")
        stage_validated_file(
            validator_path,
            staged,
            label="release validator",
            expected_sha256=validator_sha256,
            maximum_bytes=4 * 1024 * 1024,
        )
        completed = subprocess.run(
            [sys.executable, staged, *arguments],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if completed.returncode != 0:
            raise ProvenanceError(
                f"release validator failed with exit {completed.returncode}"
            )
        assert_pass_receipt(
            receipt_path,
            release_commit=release_commit,
            release_tree=release_tree,
            certificate_scope=certificate_scope,
        )
        return completed


def assert_reviewed_release_policy(
    repository: str,
    release_commit: str,
    release_tree: str,
) -> dict[str, Any]:
    """Bind production to the readable conservative constant in the release."""

    load_forge_contract(repository, release_commit, require_clean=True)
    completed = subprocess.run(
        ["git", "-C", repository, "rev-parse", "HEAD^{tree}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0 or completed.stdout.strip() != release_tree:
        raise ProvenanceError("release tree differs from the release commit")
    from forge import recipe

    expected = {
        "schema": 1,
        "kind": "forge-reviewed-conservative-timing-constant",
        "model_type": "krea2",
        "seconds_per_step": 2.2,
        "basis": "week5-validator-field-depth-owner-reviewed-2026-08-03",
        "evidence_boundary": (
            "host-bound-lab-profile-never-consumed-by-production"
        ),
    }
    if recipe.KREA_RELEASE_TIMING_POLICY != expected:
        raise ProvenanceError("reviewed Krea release timing policy drifted")
    if recipe.SEC_PER_IT.get("krea2") != expected["seconds_per_step"]:
        raise ProvenanceError("production Krea timing constant differs from policy")
    result = {
        "schema": 1,
        "kind": "sn56.week6.reviewed-release-timing-policy",
        "state": "PASS",
        "release_commit": release_commit,
        "release_tree": release_tree,
        "policy": expected,
        "policy_sha256": hashlib.sha256(canonical_bytes(expected)).hexdigest(),
    }
    result["receipt_sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    return result


def self_test(
    forge_repository: str,
    forge_commit: str,
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        try:
            load_forge_contract(
                os.path.join(forge_repository, "forge"),
                forge_commit,
                require_clean=False,
            )
        except ProvenanceError as exc:
            assert "exact repository root" in str(exc)
        else:
            raise AssertionError("Forge repository subdirectory was accepted")

        ignored_repo = base / "ignored-surface-repository"
        ignored_repo.mkdir()
        subprocess.run(
            ["git", "init", "-q", str(ignored_repo)],
            check=True,
            timeout=30,
        )
        subprocess.run(
            ["git", "-C", str(ignored_repo), "config", "user.name", "self-test"],
            check=True,
            timeout=30,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(ignored_repo),
                "config",
                "user.email",
                "self-test@example.invalid",
            ],
            check=True,
            timeout=30,
        )
        (ignored_repo / ".gitignore").write_text("ignored/\n", encoding="utf-8")
        (ignored_repo / "README").write_text("fixture\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(ignored_repo), "add", ".gitignore", "README"],
            check=True,
            timeout=30,
        )
        subprocess.run(
            ["git", "-C", str(ignored_repo), "commit", "-q", "-m", "fixture"],
            check=True,
            timeout=30,
        )
        ignored_commit = subprocess.run(
            ["git", "-C", str(ignored_repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        (ignored_repo / "ignored" / "forge").mkdir(parents=True)
        (ignored_repo / "ignored" / "forge" / "__init__.py").write_text(
            "raise RuntimeError('shadow import')\n",
            encoding="utf-8",
        )
        try:
            load_forge_contract(
                os.path.realpath(ignored_repo),
                ignored_commit,
                require_clean=True,
            )
        except ProvenanceError as exc:
            assert "ignored surfaces" in str(exc)
        else:
            raise AssertionError("ignored Forge execution surface was accepted")

        adaptive_timing = load_forge_contract(
            forge_repository,
            forge_commit,
            require_clean=False,
        )
        from forge import krea_runtime
        from forge.tasks.integrity import inspect_training_artifact

        artifact = base / "last.safetensors"
        metadata = {"training_info": json.dumps({"step": 1000, "epoch": 1})}
        header = json.dumps(
            {
                "__metadata__": metadata,
                "weight": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                },
            }
        ).encode("utf-8")
        artifact.write_bytes(
            struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.0)
        )
        archived_artifact = base / "archived-last.safetensors"
        archived_artifact.write_bytes(artifact.read_bytes())
        artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        artifact_evidence = inspect_training_artifact(str(artifact))
        source_run_id = "fixture-task:" + "a" * 32
        session_id = "friday-h100-self-test"
        bundle_id = krea_runtime.LEADER_BUNDLE
        bundle_sha = krea_runtime.bundle_contract_sha256(bundle_id)
        model_type = "krea2"
        dataset_size = 24
        regime = adaptive_timing.dataset_regime(dataset_size)
        accelerator = "NVIDIA H100 PCIe|81559-MiB"
        started = "2026-08-07T12:00:00Z"
        training_started_at = "2026-08-07T12:02:00Z"
        raw_at = "2026-08-07T12:25:00Z"
        profile_at = "2026-08-07T12:26:00Z"
        sealed_at = "2026-08-07T12:27:00Z"
        ended = "2026-08-07T18:00:00Z"
        release_tree = subprocess.run(
            ["git", "-C", forge_repository, "rev-parse", "HEAD^{tree}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        certificate_scope = "toolkit-krea-only"

        raw: dict[str, Any] = {
            "schema": krea_runtime.EFFECTIVE_RUNTIME_SCHEMA,
            "runtime_contract_id": krea_runtime.RUNTIME_CONTRACT_ID,
            "source_run_id": source_run_id,
            "model_type": model_type,
            "runtime_repository": krea_runtime.OWNED_RUNTIME_REPOSITORY,
            "runtime_commit": krea_runtime.OWNED_RUNTIME_COMMIT,
            "bundle": bundle_id,
            "bundle_claim": krea_runtime.bundle_claim_document(bundle_id),
            "bundle_contract_sha256": bundle_sha,
            "generated_config_sha256": "d" * 64,
            "capability_manifest_file_sha256": "e" * 64,
            "capability_manifest_semantic_sha256": "f" * 64,
            "capabilities": sorted(krea_runtime.REQUIRED_CAPABILITIES),
            "runtime_manifest_capability_aliases": (
                krea_runtime.bundle_contract_document(bundle_id)[
                    "runtime_manifest_capability_aliases"
                ]
            ),
            "timing": {
                "mode": "bootstrap_probe_unmeasured",
                "profile_sha256": None,
                "runtime_commit": krea_runtime.OWNED_RUNTIME_COMMIT,
                "measured_dataset_size": None,
                "current_dataset_size": dataset_size,
                "dataset_regime": regime,
                "accelerator_identity": accelerator,
                "accelerator_identity_evidence": "operator-attested",
            },
            "effective": {
                "planned_steps": 1000,
                "normalized_config_projection": (
                    krea_runtime.bundle_contract_document(bundle_id)[
                        "normalized_config_projection"
                    ]
                ),
            },
            "lifecycle": "terminal",
            "first_checkpoint_observation": {
                "bundle_id": bundle_id,
                "timing_profile_sha256": None,
                "observation_mode": "bootstrap_raw_first_checkpoint",
                "checkpoint_step": 200,
                "elapsed_since_launch_s": 260.0,
                "active_planned_steps": 1000,
                "active_plan_mutable": False,
                "active_plan_action": "observe_only_fixed_subprocess",
            },
            "training_completion_observation": {
                "training_elapsed_seconds": 1300.0,
                "returncode": 0,
                "stopped_by_deadline": False,
                "natural_completion": True,
                "artifact_path": str(artifact),
                "artifact_name": artifact.name,
                "artifact_size_bytes": artifact_evidence.size_bytes,
                "artifact_sha256": artifact_sha,
                "artifact_loadable": True,
                "artifact_checkpoint_step": 1000,
                "completed_steps": 1000,
                "scope_attempt_nonce": "a" * 32,
                "artifact_file_identity": artifact_evidence.file_identity,
            },
        }
        raw["record_sha256"] = krea_runtime._canonical_sha256(raw)
        raw_path = base / "effective-runtime.json"
        raw_path.write_bytes(krea_runtime._canonical_bytes(raw))
        raw_file_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()

        profile = adaptive_timing.produce_profile_document(
            str(raw_path),
            source_run_id=source_run_id,
            bundle_id=bundle_id,
            model_type=model_type,
            measured_dataset_size=dataset_size,
            measured_at_utc=profile_at,
            expected_accelerator_identity=accelerator,
        )
        profile_path = base / "profile.json"
        profile_path.write_bytes(canonical_bytes(profile) + b"\n")
        profile_file_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()

        event = {
            "event": EVENT_KIND,
            "gate_session_id": session_id,
            "source_run_id": source_run_id,
            "rental_started_at_utc": started,
            "rental_ended_at_utc": ended,
            "training_started_at_utc": training_started_at,
            "raw_record_produced_at_utc": raw_at,
            "profile_produced_at_utc": profile_at,
            "sealed_at_utc": sealed_at,
            "profile_file_sha256": profile_file_sha,
            "raw_record_file_sha256": raw_file_sha,
            "terminal_artifact_file_sha256": artifact_sha,
            "profile_semantic_sha256": profile["profile_sha256"],
            "raw_record_semantic_sha256": raw["record_sha256"],
            "forge_commit": forge_commit,
            "release_tree": release_tree,
            "certificate_scope": certificate_scope,
            "bundle_id": bundle_id,
            "bundle_sha256": bundle_sha,
            "model_type": model_type,
            "current_dataset_size": dataset_size,
            "dataset_regime": regime,
            "accelerator_identity": accelerator,
        }
        gate_path = base / "gate.jsonl"
        gate_path.write_bytes(canonical_bytes(event) + b"\n")
        gate_sha = hashlib.sha256(gate_path.read_bytes()).hexdigest()

        args = argparse.Namespace(
            profile=str(profile_path),
            profile_file_sha256=profile_file_sha,
            raw_record=str(raw_path),
            raw_record_file_sha256=raw_file_sha,
            terminal_artifact=str(artifact),
            archived_terminal_artifact=str(archived_artifact),
            terminal_artifact_file_sha256=artifact_sha,
            gate_log=str(gate_path),
            gate_log_file_sha256=gate_sha,
            source_run_id=source_run_id,
            gate_session_id=session_id,
            rental_started_at_utc=started,
            rental_ended_at_utc=ended,
            forge_repository=forge_repository,
            forge_commit=forge_commit,
            release_tree=release_tree,
            certificate_scope=certificate_scope,
            bundle_id=bundle_id,
            bundle_sha256=bundle_sha,
            model_type=model_type,
            current_dataset_size=str(dataset_size),
            dataset_regime=regime,
            accelerator_identity=accelerator,
            allow_dirty_forge=True,
        )
        receipt = validate(args)
        assert receipt["state"] == "PASS"
        assert receipt["evidence_class"] == "operator-attested"

        # Exercise the actual capture-to-Forge handoff, not merely a helper.
        # A path swap after capture cannot alter the bytes/artifact descriptor
        # evidence Forge consumes. Restoring A before the final path binding is
        # therefore safe and must preserve the A-bound receipt.
        profile_a = profile_path.read_bytes()
        raw_a = raw_path.read_bytes()
        artifact_a = artifact.read_bytes()
        profile_b = b'{"schema":999,"kind":"path-swap-B"}\n'
        raw_b = b'{"schema":999,"source_run_id":"path-swap-B"}\n'
        artifact_b = b"path-swap-B-is-not-safetensors"

        def replace_bytes(path: Path, payload: bytes) -> None:
            replacement = path.with_name(f".{path.name}.swap")
            replacement.write_bytes(payload)
            os.replace(replacement, path)

        def swap_actual_handoff_to_b() -> None:
            replace_bytes(profile_path, profile_b)
            replace_bytes(raw_path, raw_b)
            replace_bytes(artifact, artifact_b)

        def restore_actual_handoff_to_a() -> None:
            replace_bytes(profile_path, profile_a)
            replace_bytes(raw_path, raw_a)
            replace_bytes(artifact, artifact_a)

        swapped_receipt = validate(
            args,
            _after_capture=swap_actual_handoff_to_b,
            _before_final_binding=restore_actual_handoff_to_a,
        )
        assert swapped_receipt["state"] == "PASS"
        assert (
            swapped_receipt["files"]["profile"]["file_sha256"]
            == profile_file_sha
        )

        # If B remains at a bound path, the final binding must abort instead
        # of issuing an A receipt for the current B path.
        try:
            validate(args, _after_capture=swap_actual_handoff_to_b)
        except ProvenanceError as exc:
            assert "final timing profile binding hash mismatch" in str(exc)
        else:
            raise AssertionError("persistent A/B evidence swap was accepted")
        restore_actual_handoff_to_a()

        def swap_terminal_artifact_before_final_binding() -> None:
            replace_bytes(artifact, artifact_b)

        try:
            validate(
                args,
                _before_final_binding=swap_terminal_artifact_before_final_binding,
            )
        except ProvenanceError as exc:
            assert "final terminal artifact binding hash mismatch" in str(exc)
        else:
            raise AssertionError("terminal-artifact final-binding swap was accepted")
        restore_actual_handoff_to_a()

        def swap_archived_artifact_before_final_binding() -> None:
            replace_bytes(archived_artifact, artifact_b)

        try:
            validate(
                args,
                _before_final_binding=swap_archived_artifact_before_final_binding,
            )
        except ProvenanceError as exc:
            assert "final archived terminal artifact binding hash mismatch" in str(exc)
        else:
            raise AssertionError("archived-artifact final-binding swap was accepted")
        replace_bytes(archived_artifact, artifact_a)

        old_labeled_profile = dict(profile)
        old_labeled_profile["kind"] = "forge-measured-throughput-profile"
        old_labeled_profile["profile_sha256"] = canonical_sha256_without(
            old_labeled_profile, "profile_sha256"
        )
        profile_path.write_bytes(canonical_bytes(old_labeled_profile) + b"\n")
        args.profile_file_sha256 = hashlib.sha256(
            profile_path.read_bytes()
        ).hexdigest()
        try:
            validate(args)
        except ProvenanceError as exc:
            assert "unsupported operator-attested timing profile contract" in str(exc)
        else:
            raise AssertionError("old measured-labeled profile was accepted")
        profile_path.write_bytes(canonical_bytes(profile) + b"\n")
        args.profile_file_sha256 = profile_file_sha

        old_schema_profile = dict(profile)
        old_schema_profile["schema"] = 3
        old_schema_profile["profile_sha256"] = canonical_sha256_without(
            old_schema_profile, "profile_sha256"
        )
        profile_path.write_bytes(canonical_bytes(old_schema_profile) + b"\n")
        args.profile_file_sha256 = hashlib.sha256(
            profile_path.read_bytes()
        ).hexdigest()
        try:
            validate(args)
        except ProvenanceError as exc:
            assert "unsupported operator-attested timing profile contract" in str(exc)
        else:
            raise AssertionError("schema-v3 timing profile was accepted")
        profile_path.write_bytes(canonical_bytes(profile) + b"\n")
        args.profile_file_sha256 = profile_file_sha

        old_raw = dict(raw)
        old_raw["schema"] = 4
        old_raw.pop("record_sha256")
        old_raw["record_sha256"] = krea_runtime._canonical_sha256(old_raw)
        raw_path.write_bytes(krea_runtime._canonical_bytes(old_raw))
        args.raw_record_file_sha256 = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
        try:
            validate(args)
        except ProvenanceError as exc:
            assert "unsupported raw runtime record schema" in str(exc)
        else:
            raise AssertionError("schema-v4 raw runtime record was accepted")
        raw_path.write_bytes(krea_runtime._canonical_bytes(raw))
        args.raw_record_file_sha256 = raw_file_sha

        def stage_gate(value: dict[str, Any]) -> None:
            gate_path.write_bytes(canonical_bytes(value) + b"\n")
            args.gate_log_file_sha256 = hashlib.sha256(
                gate_path.read_bytes()
            ).hexdigest()

        bad = json.loads(gate_path.read_text(encoding="utf-8"))
        bad["raw_record_produced_at_utc"] = "2026-08-07T11:59:59Z"
        stage_gate(bad)
        try:
            validate(args)
        except ProvenanceError as exc:
            assert "outside the rental window" in str(exc)
        else:
            raise AssertionError("out-of-window gate record was accepted")

        impossible_duration = dict(event)
        impossible_duration["raw_record_produced_at_utc"] = (
            "2026-08-07T12:10:00Z"
        )
        stage_gate(impossible_duration)
        try:
            validate(args)
        except ProvenanceError as exc:
            assert "duration is inconsistent" in str(exc)
        else:
            raise AssertionError("impossible training duration was accepted")

        materially_underreported = dict(event)
        materially_underreported["training_started_at_utc"] = started
        stage_gate(materially_underreported)
        try:
            validate(args)
        except ProvenanceError as exc:
            assert "duration is inconsistent" in str(exc)
        else:
            raise AssertionError("materially underreported duration was accepted")

        stage_gate(event)
        args.gate_session_id = "different-session"
        try:
            validate(args)
        except ProvenanceError as exc:
            assert "0 matching timing seal events" in str(exc)
        else:
            raise AssertionError("mismatched gate session was accepted")
        args.gate_session_id = session_id

        mismatched_hash = dict(event)
        mismatched_hash["raw_record_file_sha256"] = "b" * 64
        stage_gate(mismatched_hash)
        try:
            validate(args)
        except ProvenanceError as exc:
            assert "0 matching timing seal events" in str(exc)
        else:
            raise AssertionError("mismatched gate-event hash was accepted")

        stage_gate(event)
        profile_symlink = base / "profile-link.json"
        profile_symlink.symlink_to(profile_path)
        args.profile = str(profile_symlink)
        try:
            validate(args)
        except ProvenanceError as exc:
            assert "timing profile could not be opened" in str(exc)
        else:
            raise AssertionError("symlinked final profile component was accepted")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--stage-source")
    result.add_argument("--stage-destination")
    result.add_argument("--stage-sha256")
    result.add_argument("--stage-label", default="release evidence")
    result.add_argument("--stage-maximum-bytes", type=int)
    result.add_argument("--assert-receipt")
    result.add_argument("--assert-result-env")
    result.add_argument("--assert-file")
    result.add_argument("--assert-file-sha256")
    result.add_argument("--assert-release-policy", action="store_true")
    result.add_argument("--profile")
    result.add_argument("--profile-file-sha256")
    result.add_argument("--raw-record")
    result.add_argument("--raw-record-file-sha256")
    result.add_argument("--terminal-artifact")
    result.add_argument("--archived-terminal-artifact")
    result.add_argument("--terminal-artifact-file-sha256")
    result.add_argument("--gate-log")
    result.add_argument("--gate-log-file-sha256")
    result.add_argument("--source-run-id")
    result.add_argument("--gate-session-id")
    result.add_argument("--rental-started-at-utc")
    result.add_argument("--rental-ended-at-utc")
    result.add_argument("--forge-repository")
    result.add_argument("--forge-commit")
    result.add_argument("--release-tree")
    result.add_argument("--certificate-scope")
    result.add_argument("--bundle-id")
    result.add_argument("--bundle-sha256")
    result.add_argument("--model-type")
    result.add_argument("--current-dataset-size")
    result.add_argument("--dataset-regime")
    result.add_argument("--accelerator-identity")
    result.add_argument("--receipt")
    result.add_argument("--self-test", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.stage_source:
        if not args.stage_destination or not args.stage_sha256:
            raise ProvenanceError(
                "staging requires destination and expected hash"
            )
        stage_validated_file(
            args.stage_source,
            args.stage_destination,
            label=args.stage_label,
            expected_sha256=args.stage_sha256,
            maximum_bytes=args.stage_maximum_bytes,
        )
        print("SN56_RELEASE_EVIDENCE_STAGE=PASS")
        return 0
    if args.assert_file:
        if not args.assert_file_sha256:
            raise ProvenanceError("file assertion requires expected hash")
        assert_regular_file_hash(
            args.assert_file,
            label=args.stage_label,
            expected_sha256=args.assert_file_sha256,
        )
        print("SN56_RELEASE_FILE_BINDING=PASS")
        return 0
    if args.assert_receipt:
        if not args.forge_commit or not args.release_tree or not args.certificate_scope:
            raise ProvenanceError("receipt assertion lacks release identity")
        file_hash_arguments = {
            "profile": args.profile_file_sha256,
            "raw_record": args.raw_record_file_sha256,
            "terminal_artifact": args.terminal_artifact_file_sha256,
            "archived_terminal_artifact": args.terminal_artifact_file_sha256,
            "gate_log": args.gate_log_file_sha256,
        }
        expected_file_hashes = None
        if any(file_hash_arguments.values()):
            if not all(file_hash_arguments.values()):
                raise ProvenanceError(
                    "receipt assertion has an incomplete file-hash set"
                )
            expected_file_hashes = file_hash_arguments
        assert_pass_receipt(
            args.assert_receipt,
            release_commit=args.forge_commit,
            release_tree=args.release_tree,
            certificate_scope=args.certificate_scope,
            expected_file_hashes=expected_file_hashes,
        )
        print("SN56_RELEASE_RECEIPT=PASS")
        return 0
    if args.assert_result_env:
        if not args.forge_commit or not args.release_tree or not args.certificate_scope:
            raise ProvenanceError("delegated assertion lacks release identity")
        assert_delegated_result(
            args.assert_result_env,
            release_commit=args.forge_commit,
            release_tree=args.release_tree,
            certificate_scope=args.certificate_scope,
        )
        print("SN56_DELEGATED_RESULT=PASS")
        return 0
    if args.assert_release_policy:
        if not args.forge_repository or not args.forge_commit or not args.release_tree:
            raise ProvenanceError("release-policy assertion lacks release identity")
        policy_receipt = assert_reviewed_release_policy(
            args.forge_repository,
            args.forge_commit,
            args.release_tree,
        )
        if not args.receipt:
            raise ProvenanceError("release-policy assertion requires receipt")
        write_receipt(args.receipt, policy_receipt)
        print("SN56_RELEASE_TIMING_POLICY=PASS")
        return 0
    if args.self_test:
        if not args.forge_repository or not args.forge_commit:
            raise ProvenanceError(
                "self-test requires --forge-repository and --forge-commit"
            )
        self_test(
            args.forge_repository,
            args.forge_commit,
        )
        print("SN56_WEEK6_TIMING_PROVENANCE_SELF_TEST=PASS")
        return 0
    required = (
        "profile",
        "profile_file_sha256",
        "raw_record",
        "raw_record_file_sha256",
        "terminal_artifact",
        "archived_terminal_artifact",
        "terminal_artifact_file_sha256",
        "gate_log",
        "gate_log_file_sha256",
        "source_run_id",
        "gate_session_id",
        "rental_started_at_utc",
        "rental_ended_at_utc",
        "forge_repository",
        "forge_commit",
        "release_tree",
        "certificate_scope",
        "bundle_id",
        "bundle_sha256",
        "model_type",
        "current_dataset_size",
        "dataset_regime",
        "accelerator_identity",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise ProvenanceError("missing arguments: " + ", ".join(missing))
    receipt = validate(args)
    if args.receipt:
        write_receipt(args.receipt, receipt)
    print(canonical_bytes(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProvenanceError as exc:
        print(f"SN56_WEEK6_TIMING_PROVENANCE=FAIL reason={exc}", file=sys.stderr)
        raise SystemExit(1)
