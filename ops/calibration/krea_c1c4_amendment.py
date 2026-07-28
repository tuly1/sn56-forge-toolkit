#!/usr/bin/env python3
"""Fail-closed binding for the public C1-C4 shape-contract amendment.

The amendment is deliberately a repository-local public artifact.  Consumers
must load its bytes from this checkout rather than accepting the discovery
plan's three digest strings as proof that the artifact still exists.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

try:
    from . import krea_provenance
except ImportError:  # pragma: no cover - direct script execution.
    import krea_provenance  # type: ignore[no-redef]


_SHA256 = re.compile(r"[0-9a-f]{64}")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_RECORD = "SN56-project/SN56-WEEK5-C1C4-SEALED-COMMITMENT-2026-07-27.md"
PUBLIC_RECORD_SHA256 = (
    "f907c40e362378c1b82e7455d96ffd8bd876696f25cef21705e14bbba2d4ffc0"
)
COMMITMENT_SHA256 = "0a12c416bcef48805132e80f9de65d0d248ef4415d617715d5736c189a379dbc"
PRE_AMENDMENT_PLAN_FILE_SHA256 = (
    "6365f150352de1497fbf32edc8ea07bc2859c3096c95796cff708c89382aee6a"
)
PRE_AMENDMENT_PLAN_COMMIT = "1bd7477717ab8d96d208d9fe265f071f08e47e73"
AMENDMENT_PATH = "ops/calibration/week5/krea-c1c4-shape-contract-amendment.json"
AMENDMENT_FILE_SHA256 = (
    "5f1b02ab78d6f82da6587c533af19e61ead5aa2e821ce268fa94c9bd0ad9587e"
)
AMENDMENT_SHA256 = "367fbcd46827e49efa4d14bf50d1533d85f56d5354a3233d4ea41a81779ef61c"
SHAPE_CONTRACT = {
    "C1": {
        "concept_class": "architectural object",
        "training_pairs": 20,
        "evaluation_rows": 6,
    },
    "C2": {
        "concept_class": "art/print-style series",
        "training_pairs": 45,
        "evaluation_rows": 6,
    },
    "C3": {
        "concept_class": "natural subject",
        "training_pairs": 30,
        "evaluation_rows": 8,
    },
    "C4": {
        "concept_class": "product/design object set",
        "training_pairs": 12,
        "evaluation_rows": 5,
    },
}
PRE_AMENDMENT_SHAPE_CONTRACT = {
    "C1": {
        "dataset_shape": "small",
        "training_pair_range": [18, 24],
        "evaluation_rows": 24,
    },
    "C2": {
        "dataset_shape": "small",
        "training_pair_range": [18, 24],
        "evaluation_rows": 24,
    },
    "C3": {
        "dataset_shape": "large",
        "training_pair_range": [36, 48],
        "evaluation_rows": 40,
    },
    "C4": {
        "dataset_shape": "large",
        "training_pair_range": [36, 48],
        "evaluation_rows": 40,
    },
}
MANIFEST_FILE_SHA256S = {
    "C1": "ed287150fd4d189b3a0964d87c5fc50de11851ab372dabe30da9d9f87fdc450e",
    "C2": "902a4a6716a9210694f3f441d54b4def19e9bc64d0a49be4cb832ccff8605083",
    "C3": "74ebbfaf91b156741d34b10ba2d37600076844c010ea6ea83d4af36a386eda09",
    "C4": "7a3fb670bed78d851cf8c066696b61ccc79d78dffd1ecb633520493772210872",
}
AUTHORSHIP_ORDER = (
    "authored after the public commitment and the independent reviewer finding; "
    "this amendment was not part of the original fixture seal"
)
CLAIM_LIMIT = (
    "Corrects only the public per-fixture concept classes and train/evaluation "
    "counts; fixture bytes, manifest digests, aggregate commitment, identities, "
    "and custody are unchanged."
)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(value: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = required - set(value)
    extra = set(value) - required
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _portable_relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a portable relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a portable relative path")
    return path


def _safe_repository_file(
    repository_root: Path | str, relative_path: Any, label: str
) -> Path:
    root = Path(os.path.abspath(os.path.expanduser(repository_root)))
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"repository root must be a real directory: {root}")
    relative = _portable_relative_path(relative_path, f"{label}.path")
    path = root / relative
    current = path
    while current != root.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink component: {current}")
        current = current.parent
    if not path.is_file():
        raise ValueError(f"{label} must be a repository-local regular file: {path}")
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository root") from exc
    return path


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
    )
    if identity(before) != identity(after):
        raise RuntimeError(f"{label} changed while read")
    try:
        value = _object(json.loads(raw), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    return value, hashlib.sha256(raw).hexdigest()


def validate_amendment(value: Any) -> dict[str, Any]:
    """Validate the post-publication correction without resealing C1-C4."""

    value = _object(value, "confirmation shape amendment")
    _exact(
        value,
        {
            "schema",
            "kind",
            "operation",
            "amends_discovery_plan_file_sha256",
            "amends_discovery_plan_commit",
            "authorship_order",
            "source_public_record",
            "source_public_record_sha256",
            "commitment_sha256_before",
            "commitment_sha256_after",
            "fixture_commitment_resealed",
            "implementation_read_sealed_contents",
            "prior_fixture_shape_contract",
            "amended_fixture_shape_contract",
            "published_manifest_file_sha256s",
            "claim_limit",
            "amendment_sha256",
        },
        "confirmation shape amendment",
    )
    body = {key: item for key, item in value.items() if key != "amendment_sha256"}
    expected = {
        "schema": 1,
        "kind": "forge-krea-confirmation-fixture-shape-contract-amendment",
        "operation": "amend_contract_metadata_without_resealing_fixtures",
        "amends_discovery_plan_file_sha256": PRE_AMENDMENT_PLAN_FILE_SHA256,
        "amends_discovery_plan_commit": PRE_AMENDMENT_PLAN_COMMIT,
        "authorship_order": AUTHORSHIP_ORDER,
        "source_public_record": PUBLIC_RECORD,
        "source_public_record_sha256": PUBLIC_RECORD_SHA256,
        "commitment_sha256_before": COMMITMENT_SHA256,
        "commitment_sha256_after": COMMITMENT_SHA256,
        "fixture_commitment_resealed": False,
        "implementation_read_sealed_contents": False,
        "prior_fixture_shape_contract": PRE_AMENDMENT_SHAPE_CONTRACT,
        "amended_fixture_shape_contract": SHAPE_CONTRACT,
        "published_manifest_file_sha256s": MANIFEST_FILE_SHA256S,
        "claim_limit": CLAIM_LIMIT,
    }
    if body != expected:
        raise ValueError("confirmation shape amendment differs from its frozen basis")
    if value["amendment_sha256"] != krea_provenance.canonical_sha256(body):
        raise ValueError("confirmation shape amendment self-digest mismatch")
    return dict(value)


def _expected_commitment() -> dict[str, Any]:
    return {
        "state": "published_external_unaccepted_by_named_human",
        "public_record": PUBLIC_RECORD,
        "public_record_sha256": PUBLIC_RECORD_SHA256,
        "commitment_sha256": COMMITMENT_SHA256,
        "shape_contract_amendment": {
            "path": AMENDMENT_PATH,
            "file_sha256": AMENDMENT_FILE_SHA256,
            "amendment_sha256": AMENDMENT_SHA256,
        },
        "implementation_read_sealed_contents": False,
    }


def validate_bound_plan_amendment(
    value: Any, *, repository_root: Path | str | None = None
) -> dict[str, Any]:
    """Load and verify the repo-local amendment bound by a discovery plan."""

    plan = _object(value, "discovery plan")
    commitment = _object(
        plan.get("confirmation_fixture_commitment"),
        "confirmation_fixture_commitment",
    )
    _exact(
        commitment,
        {
            "state",
            "public_record",
            "public_record_sha256",
            "commitment_sha256",
            "shape_contract_amendment",
            "implementation_read_sealed_contents",
        },
        "confirmation_fixture_commitment",
    )
    expected_commitment = _expected_commitment()
    if commitment != expected_commitment:
        raise ValueError("confirmation fixture commitment differs from publication")

    confirmation = _object(plan.get("confirmation_contract"), "confirmation_contract")
    if confirmation.get("fixture_shape_contract") != SHAPE_CONTRACT:
        raise ValueError("confirmation fixture shape contract differs from amendment")

    binding = _object(
        commitment["shape_contract_amendment"], "shape_contract_amendment"
    )
    _exact(
        binding,
        {"path", "file_sha256", "amendment_sha256"},
        "shape_contract_amendment",
    )
    root = REPOSITORY_ROOT if repository_root is None else repository_root
    path = _safe_repository_file(root, binding["path"], "shape contract amendment")
    amendment, file_sha256 = _load_json(path, "shape contract amendment")
    if file_sha256 != _digest(binding["file_sha256"], "amendment file_sha256"):
        raise ValueError("confirmation shape amendment file SHA-256 mismatch")
    amendment = validate_amendment(amendment)
    if amendment["amendment_sha256"] != _digest(
        binding["amendment_sha256"], "amendment_sha256"
    ):
        raise ValueError("confirmation shape amendment digest binding mismatch")
    return amendment
