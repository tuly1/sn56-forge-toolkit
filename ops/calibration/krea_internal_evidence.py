#!/usr/bin/env python3
"""Canonical K5 internal-evidence record and portable project-root binding."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any

try:
    from . import krea_provenance
except ImportError:  # pragma: no cover - direct script execution.
    import krea_provenance  # type: ignore[no-redef]


_SHA256 = re.compile(r"[0-9a-f]{64}")
_RECORD_PATH = "K5-INTERNAL-EVIDENCE-RECORD.json"
_EVIDENCE_PATHS = (
    "SN56-GATE-B-H100-RESULTS.md",
    "SN56-WEEK4-GPU-CAMPAIGN-RESULTS-2026-07-23.md",
    "week4-gpu-evidence-2026-07-22/EVIDENCE-ERRATA.md",
)
_PROJECT_ROOT_CONTRACT = "explicit_bound_input_not_process_cwd"
_CLAIM_LIMIT = (
    "Same-fixture, same-training-seed shallow evidence only. This record does "
    "not establish an independent seed, a second fixture, a deep-arm effect, "
    "universal LR superiority, field parity, or production authorization."
)
_OBSERVATIONS = {
    "gate_b_367_step_guidance_on_final": {
        "learning_rate_1e_4_loss": 0.028422827,
        "learning_rate_2e_4_loss": 0.027107584,
        "seed": 42565431,
    },
    "repeat_367_step_guidance_on_final": {
        "learning_rate_1e_4_loss": 0.0283811,
        "learning_rate_2e_4_loss": 0.0274579,
        "seed": 42565431,
    },
    "scope_correction": (
        "The repeat used the same fixture bytes, same external rows, and same "
        "training seed as Gate B; it is not second-seed evidence."
    ),
}


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(value: dict[str, Any], keys: set[str], label: str) -> None:
    missing = keys - set(value)
    extra = set(value) - keys
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _project_root(value: str | Path) -> Path:
    supplied = Path(value)
    if not supplied.is_absolute():
        raise ValueError("project_root must be an explicit absolute path")
    path = Path(os.path.abspath(os.path.expanduser(supplied)))
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"project_root has a symlink ancestor: {current}")
        current = current.parent
    if not path.is_dir():
        raise ValueError(f"project_root is not a real directory: {path}")
    return path


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a portable project-root-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a portable project-root-relative path")
    if path.as_posix() != value:
        raise ValueError(f"{label} is not canonical POSIX relative syntax")
    return value


def _read_project_file(
    project_root: Path, relative: str, label: str
) -> tuple[bytes, os.stat_result]:
    relative = _relative_path(relative, label)
    path = project_root.joinpath(*PurePosixPath(relative).parts)
    current = path.parent
    while current != project_root.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        if current == project_root:
            break
        current = current.parent
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} is missing or unsafe: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file")
        chunks = []
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
    )
    if identity(before) != identity(after):
        raise RuntimeError(f"{label} changed while read")
    return b"".join(chunks), after


def build_record(*, project_root: str | Path) -> dict[str, Any]:
    """Rebuild K5's record from the exact canonical project evidence bytes."""

    root = _project_root(project_root)
    evidence = []
    for relative in _EVIDENCE_PATHS:
        raw, metadata = _read_project_file(root, relative, f"K5 evidence {relative}")
        evidence.append(
            {
                "path": relative,
                "bytes": metadata.st_size,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    body = {
        "schema": 2,
        "kind": "forge-krea-k5-internal-evidence-record",
        "project_root_contract": _PROJECT_ROOT_CONTRACT,
        "evidence_files": evidence,
        "observations": _OBSERVATIONS,
        "claim_limit": _CLAIM_LIMIT,
    }
    return {**body, "record_sha256": krea_provenance.canonical_sha256(body)}


def validate_record(
    value: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    value = _object(value, "K5 internal evidence record")
    _exact(
        value,
        {
            "schema",
            "kind",
            "project_root_contract",
            "evidence_files",
            "observations",
            "claim_limit",
            "record_sha256",
        },
        "K5 internal evidence record",
    )
    body = {key: item for key, item in value.items() if key != "record_sha256"}
    if (
        value["schema"] != 2
        or value["kind"] != "forge-krea-k5-internal-evidence-record"
        or value["project_root_contract"] != _PROJECT_ROOT_CONTRACT
        or value["record_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("K5 internal evidence record identity is invalid")
    rows = value["evidence_files"]
    if not isinstance(rows, list) or len(rows) != len(_EVIDENCE_PATHS):
        raise ValueError("K5 evidence file coverage is incomplete")
    observed_paths = []
    for index, raw in enumerate(rows):
        label = f"K5 evidence_files[{index}]"
        row = _object(raw, label)
        _exact(row, {"path", "bytes", "sha256"}, label)
        observed_paths.append(_relative_path(row["path"], f"{label}.path"))
        if (
            isinstance(row["bytes"], bool)
            or not isinstance(row["bytes"], int)
            or row["bytes"] <= 0
        ):
            raise ValueError(f"{label}.bytes must be a positive integer")
        _digest(row["sha256"], f"{label}.sha256")
    if tuple(observed_paths) != _EVIDENCE_PATHS:
        raise ValueError("K5 evidence paths differ from the frozen canonical set")
    expected = build_record(project_root=project_root)
    if value != expected:
        raise ValueError("K5 evidence record differs from its bound project bytes")
    return value


def load_record(
    path: str | Path, *, project_root: str | Path
) -> tuple[dict[str, Any], str, int]:
    root = _project_root(project_root)
    expected_path = root / _RECORD_PATH
    supplied = Path(os.path.abspath(os.path.expanduser(path)))
    if supplied != expected_path:
        raise ValueError("K5 record must be the canonical project-root copy")
    raw, metadata = _read_project_file(root, _RECORD_PATH, "K5 evidence record")
    try:
        value = _object(json.loads(raw), "K5 evidence record")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("K5 evidence record is not JSON") from exc
    if raw != krea_provenance.canonical_bytes(value) + b"\n":
        raise ValueError("K5 evidence record must be canonical JSON plus one newline")
    validate_record(value, project_root=root)
    return value, hashlib.sha256(raw).hexdigest(), metadata.st_size


def build_anchor(
    *, record_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    record, file_sha, size = load_record(record_path, project_root=project_root)
    return {
        "project_root_contract": _PROJECT_ROOT_CONTRACT,
        "record_path": _RECORD_PATH,
        "record_bytes": size,
        "record_file_sha256": file_sha,
        "record_sha256": record["record_sha256"],
        "evidence_paths": list(_EVIDENCE_PATHS),
    }


def validate_anchor(value: Any) -> dict[str, Any]:
    anchor = _object(value, "K5 internal evidence anchor")
    _exact(
        anchor,
        {
            "project_root_contract",
            "record_path",
            "record_bytes",
            "record_file_sha256",
            "record_sha256",
            "evidence_paths",
        },
        "K5 internal evidence anchor",
    )
    if (
        anchor["project_root_contract"] != _PROJECT_ROOT_CONTRACT
        or anchor["record_path"] != _RECORD_PATH
        or anchor["evidence_paths"] != list(_EVIDENCE_PATHS)
        or isinstance(anchor["record_bytes"], bool)
        or not isinstance(anchor["record_bytes"], int)
        or anchor["record_bytes"] <= 0
    ):
        raise ValueError("K5 internal evidence anchor is invalid")
    _digest(anchor["record_file_sha256"], "K5 anchor record_file_sha256")
    _digest(anchor["record_sha256"], "K5 anchor record_sha256")
    return anchor
