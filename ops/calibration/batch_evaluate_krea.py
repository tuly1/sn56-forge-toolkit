#!/usr/bin/env python3
"""Exact-score a reviewed Krea candidate set in isolated processes.

Every candidate is evaluated by ``evaluate_krea_local.py`` in a fresh process.
The legacy ``batch`` mode remains available.  Long campaigns can instead run
one create-only ``candidate-shard`` per candidate and later use
``assemble-shards``; assembly independently revalidates every staged input,
result, log, binding, and complete-set hash before publishing.  No aggregate is
published without full candidate coverage under one common dataset, evaluator,
source, asset, and runtime envelope.  This is an offline calibration artifact;
it can never write Forge's production selection file.
"""

from __future__ import annotations

import argparse
import fcntl
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

try:  # Direct script execution places this directory on sys.path.
    from . import krea_dataset_identity
    from . import krea_delegated_review_contract
    from . import krea_discovery_authorization
    from . import krea_execution_plan
    from . import krea_execution_surface_policy
    from . import krea_fixture
    from . import krea_historical_training_evidence
    from . import krea_provenance
    from . import krea_scorer_extension_policy
except ImportError:  # pragma: no cover - exercised by CLI, not module tests.
    import krea_dataset_identity  # type: ignore[no-redef]
    import krea_delegated_review_contract  # type: ignore[no-redef]
    import krea_discovery_authorization  # type: ignore[no-redef]
    import krea_execution_plan  # type: ignore[no-redef]
    import krea_execution_surface_policy  # type: ignore[no-redef]
    import krea_fixture  # type: ignore[no-redef]
    import krea_historical_training_evidence  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_scorer_extension_policy  # type: ignore[no-redef]


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SCHEMA = 1
_KIND = "forge-krea-exact-score-batch"
_FORBIDDEN_OUTPUT = "forge_holdout_scores.json"
_COMFY_LORA_PLACEHOLDER = "put_loras_here"
_SCORER_SUPPORT_MODULE_SHA256 = {
    "krea_execution_surface_policy.py": (
        "597e5047e419a5007e5dd7e9c80c3d771ac21995028899edaac38ba47bf02722"
    ),
    "krea_historical_training_evidence.py": (
        "6734200163e856a14a2d41a370e98e1b4b801091e5f828c37817ff9d4435f3d0"
    ),
    "krea_scorer_extension_policy.py": (
        "b0c033025dc35f0cdc3e348234507c24f17ac636f522eaff3a392cd3e74a062b"
    ),
}
_FIXTURE_KIND = "forge-krea-fixture-split"
_CONDITION_KIND = "forge-krea-training-condition"
_COMPLETION_KIND = "forge-krea-training-completion"
_SOURCE_APPROVAL_KIND = "forge-krea-source-normalization-approval"
_PLAN_APPROVAL_KIND = "forge-krea-sealed-plan-approval"
_DISCOVERY_DECISION_BINDING = {
    "paired_rows_required": True,
    "discovery_tie_band": 0.01,
    "cluster_unit": "task/concept",
    "bootstrap": "cluster-bootstrap by task/concept",
    "bootstrap_confidence": 0.95,
    "bootstrap_resamples": 10_000,
    "bootstrap_seed": 42_565_431,
    "material_rank_reversal_definition": (
        "any non-control pair switches order across D1/D2 with >0.01 "
        "relative-improvement separation in both directions"
    ),
    "checkpoint_tie_breaker": (
        "earliest actual step among candidates within 0.01 of best"
    ),
}
_CONFIRMATION_DECISION_BINDING = {
    "field_parity_noninferiority_cap": 0.01,
    "concept_regression_cap": 0.03,
    "minimum_point_estimate_wins_or_ties": 3,
    "point_win_or_tie_cap": 0.01,
    "strongest_public_reference_rule": (
        "minimum loss among exhaustive approved K2-K4 local public-family "
        "reproductions for the same "
        "concept and seed"
    ),
    "boundary_gate": "mechanics_only_natural_completion_upload_ready_clean",
}
_UNIT_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_ROLE_LABELS = frozenset(
    {
        "reviewer",
        "human reviewer",
        "human owner",
        "owner",
        "engineer",
        "response engineer",
        "review engineer",
        "user",
        "operator",
        "dri",
    }
)
_SCORER_LEASE_GUARD = threading.Lock()
_ACTIVE_SCORER_LEASES: dict[str, tuple[int, str]] = {}


class _EvaluatorCancellation(BaseException):
    def __init__(self, signum: int):
        super().__init__(f"batch received signal {signum}")
        self.signum = signum


def _minimal_evaluator_environment(
    *, driver_python: str, isolated_root: Path
) -> dict[str, str]:
    """Construct the exact outer evaluator environment from empty storage."""

    root = isolated_root.resolve(strict=True)
    python = _safe_file(driver_python, "evaluator driver Python", executable=True)
    directories = {
        "HOME": root / "home",
        "XDG_CACHE_HOME": root / "xdg-cache",
        "TORCH_HOME": root / "torch",
        "HF_HOME": root / "huggingface",
        "HF_HUB_CACHE": root / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": root / "transformers",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "PATH": f"{python.parent}:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        **{name: str(path) for name, path in directories.items()},
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("batch", "candidate-shard", "assemble-shards"),
        default="batch",
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--shard", action="append", type=Path, default=[])
    return parser.parse_args()


def _absolute_lexical(value: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(value)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact_keys(
    value: dict[str, Any],
    required: set[str],
    label: str,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _safe_file(value: str | Path, label: str, *, executable: bool = False) -> Path:
    path = _absolute_lexical(value)
    # Virtual-environment launchers are commonly symlinks.  Resolve only to
    # validate the target; executing the resolved base interpreter would lose
    # pyvenv activation and silently change the package environment.
    if executable:
        try:
            target = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"{label} does not resolve to a file: {path}") from exc
        if not target.is_file():
            raise ValueError(f"{label} does not resolve to a regular file: {path}")
    if (not executable and path.is_symlink()) or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise ValueError(f"{label} is not executable: {path}")
    return path


def _safe_directory(value: str | Path, label: str) -> Path:
    path = _absolute_lexical(value)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real directory: {path}")
    return path


def _load_json_file(path: Path, label: str) -> tuple[dict[str, Any], str, bytes]:
    path = _safe_file(path, label)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"{label} changed while it was read")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    return _object(value, label), hashlib.sha256(raw).hexdigest(), raw


def _canonical_control_file(value: dict[str, Any], raw: bytes, label: str) -> None:
    if raw != krea_provenance.canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} must use canonical JSON plus one newline")


def _named_human(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must name a human reviewer")
    identity = " ".join(value.split())
    if identity.casefold() in _ROLE_LABELS:
        raise ValueError(f"{label} is a role label, not a named human")
    words = identity.split(" ")
    if len(words) < 2 or any(
        not any(character.isalpha() for character in word) for word in words
    ):
        raise ValueError(f"{label} must contain a named human identity")
    return identity


def _bound_file(
    value: Any, label: str, *, require_nonempty: bool = False
) -> tuple[Path, str, dict[str, Any], bytes]:
    binding = _object(value, label)
    _exact_keys(binding, {"path", "sha256"}, label)
    path = _safe_file(binding["path"], label)
    expected_sha = binding["sha256"]
    if not isinstance(expected_sha, str) or not _SHA256.fullmatch(expected_sha):
        raise ValueError(f"{label} has invalid SHA-256")
    if require_nonempty and path.stat().st_size <= 0:
        raise ValueError(f"{label} must not be empty")
    value_object, actual_sha, raw = _load_json_file(path, label)
    if actual_sha != expected_sha:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path, expected_sha, value_object, raw


def _bound_bytes_file(
    value: Any, label: str, *, require_nonempty: bool = True
) -> tuple[Path, str]:
    binding = _object(value, label)
    _exact_keys(binding, {"path", "sha256"}, label)
    path = _safe_file(binding["path"], label)
    expected_sha = binding["sha256"]
    if not isinstance(expected_sha, str) or not _SHA256.fullmatch(expected_sha):
        raise ValueError(f"{label} has invalid SHA-256")
    before = path.stat()
    actual_sha = _sha256(path)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"{label} changed while it was hashed")
    if actual_sha != expected_sha:
        raise ValueError(f"{label} SHA-256 mismatch")
    if require_nonempty and after.st_size <= 0:
        raise ValueError(f"{label} must not be empty")
    return path, expected_sha


def _row_set(
    value: Any, label: str
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty row array")
    rows: list[dict[str, Any]] = []
    identities = {
        "row_id": set(),
        "content_sha256": set(),
        "image_sha256": set(),
        "decoded_pixels_sha256": set(),
    }
    for index, raw in enumerate(value):
        row = _object(raw, f"{label}[{index}]")
        _exact_keys(
            row,
            {
                "row_id",
                "content_sha256",
                "image_sha256",
                "decoded_pixels_sha256",
                "caption_sha256",
                "normalized_caption_sha256",
                "width",
                "height",
                "mode",
                "perceptual_hash64",
            },
            f"{label}[{index}]",
        )
        row_id = row["row_id"]
        if not isinstance(row_id, str) or not _SAFE_ID.fullmatch(row_id):
            raise ValueError(f"{label}[{index}].row_id is invalid")
        for key in (
            "content_sha256",
            "image_sha256",
            "decoded_pixels_sha256",
            "caption_sha256",
            "normalized_caption_sha256",
        ):
            if not isinstance(row[key], str) or not _SHA256.fullmatch(row[key]):
                raise ValueError(f"{label}[{index}].{key} is invalid")
        for key in ("width", "height"):
            if (
                not isinstance(row[key], int)
                or isinstance(row[key], bool)
                or row[key] <= 0
            ):
                raise ValueError(f"{label}[{index}].{key} is invalid")
        if not isinstance(row["mode"], str) or not row["mode"].strip():
            raise ValueError(f"{label}[{index}].mode is invalid")
        if not isinstance(row["perceptual_hash64"], str) or not re.fullmatch(
            r"[0-9a-f]{16}", row["perceptual_hash64"]
        ):
            raise ValueError(f"{label}[{index}].perceptual_hash64 is invalid")
        content = {
            key: row[key]
            for key in (
                "image_sha256",
                "decoded_pixels_sha256",
                "caption_sha256",
                "normalized_caption_sha256",
                "width",
                "height",
                "mode",
            )
        }
        if row["content_sha256"] != krea_provenance.canonical_sha256(content):
            raise ValueError(f"{label}[{index}].content_sha256 does not recompute")
        if any(row[key] in identities[key] for key in identities):
            raise ValueError(f"{label} contains duplicate rows/images")
        for key in identities:
            identities[key].add(row[key])
        rows.append(dict(row))
    if rows != sorted(rows, key=lambda row: row["row_id"]):
        raise ValueError(f"{label} must be sorted by row_id")
    return rows, identities


def _validate_fixture_split(value: dict[str, Any], raw: bytes) -> dict[str, Any]:
    label = "fixture split manifest"
    _canonical_control_file(value, raw, label)
    _exact_keys(
        value,
        {
            "schema",
            "kind",
            "concept_id",
            "concept_evidence_sha256",
            "training_dataset_sha256",
            "evaluation_dataset_sha256",
            "training_rows",
            "evaluation_rows",
            "near_duplicate_policy",
        },
        label,
    )
    if value["schema"] != 1 or value["kind"] != _FIXTURE_KIND:
        raise ValueError("unsupported fixture split schema or kind")
    concept_id = value["concept_id"]
    if not isinstance(concept_id, str) or not concept_id.strip():
        raise ValueError("fixture split must name one concept")
    if not isinstance(value["concept_evidence_sha256"], str) or not _SHA256.fullmatch(
        value["concept_evidence_sha256"]
    ):
        raise ValueError("fixture split concept evidence SHA-256 is invalid")
    for key in ("training_dataset_sha256", "evaluation_dataset_sha256"):
        if not isinstance(value[key], str) or not _SHA256.fullmatch(value[key]):
            raise ValueError(f"fixture split {key} is invalid")
    if value["training_dataset_sha256"] == value["evaluation_dataset_sha256"]:
        raise ValueError("training and evaluation dataset identities must differ")
    training_rows, training_identities = _row_set(
        value["training_rows"], "fixture training_rows"
    )
    evaluation_rows, evaluation_identities = _row_set(
        value["evaluation_rows"], "fixture evaluation_rows"
    )
    for key in ("row_id", "content_sha256", "image_sha256", "decoded_pixels_sha256"):
        if training_identities[key] & evaluation_identities[key]:
            raise ValueError(
                f"fixture training and evaluation rows are not disjoint by {key}"
            )
    near = _object(value["near_duplicate_policy"], "near_duplicate_policy")
    _exact_keys(
        near,
        {
            "detector",
            "implementation_sha256",
            "maximum_hamming_distance",
            "report",
            "report_sha256",
            "passed",
        },
        "near_duplicate_policy",
    )
    if (
        near["detector"] != "pillow-rgb-average-hash-8x8"
        or not isinstance(near["implementation_sha256"], str)
        or not _SHA256.fullmatch(near["implementation_sha256"])
        or not isinstance(near["report"], dict)
        or not isinstance(near["report_sha256"], str)
        or not _SHA256.fullmatch(near["report_sha256"])
        or isinstance(near["maximum_hamming_distance"], bool)
        or not isinstance(near["maximum_hamming_distance"], int)
        or not 0 <= near["maximum_hamming_distance"] <= 64
        or near["passed"] is not True
    ):
        raise ValueError("fixture near-duplicate policy is invalid or did not pass")
    report = near["report"]
    _exact_keys(
        report,
        {"comparisons", "minimum_hamming_distance", "matches"},
        "near_duplicate_policy.report",
    )
    if (
        near["report_sha256"] != krea_provenance.canonical_sha256(report)
        or not isinstance(report["comparisons"], int)
        or isinstance(report["comparisons"], bool)
        or report["comparisons"] != len(training_rows) * len(evaluation_rows)
        or not isinstance(report["minimum_hamming_distance"], int)
        or not 0 <= report["minimum_hamming_distance"] <= 64
        or report["matches"] != []
    ):
        raise ValueError("fixture near-duplicate report is invalid")
    return {
        **value,
        "training_rows_sha256": krea_provenance.canonical_sha256(training_rows),
        "evaluation_rows_sha256": krea_provenance.canonical_sha256(evaluation_rows),
    }


def _training_geometry(value: Any) -> dict[str, Any]:
    label = "training condition geometry"
    geometry = _object(value, label)
    _exact_keys(
        geometry,
        {
            "resolution_policy_sha256",
            "precision_policy_sha256",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "data_parallel_replicas",
            "effective_batch_size",
        },
        label,
    )
    for key in ("resolution_policy_sha256", "precision_policy_sha256"):
        if not isinstance(geometry[key], str) or not _SHA256.fullmatch(geometry[key]):
            raise ValueError(f"{label} {key} is invalid")
    for key in (
        "micro_batch_size",
        "gradient_accumulation_steps",
        "data_parallel_replicas",
        "effective_batch_size",
    ):
        if (
            not isinstance(geometry[key], int)
            or isinstance(geometry[key], bool)
            or geometry[key] <= 0
        ):
            raise ValueError(f"{label} {key} is invalid")
    expected_effective = (
        geometry["micro_batch_size"]
        * geometry["gradient_accumulation_steps"]
        * geometry["data_parallel_replicas"]
    )
    if geometry["effective_batch_size"] != expected_effective:
        raise ValueError("training condition effective batch is inconsistent")
    return dict(geometry)


def _validate_condition(
    value: dict[str, Any],
    raw: bytes,
    *,
    source_arm_id: str,
    provenance: dict[str, Any],
    fixture_sha256: str,
    fixture: dict[str, Any],
) -> dict[str, Any]:
    label = "training condition"
    _canonical_control_file(value, raw, label)
    _exact_keys(
        value,
        {
            "schema",
            "kind",
            "source_arm_id",
            "provenance_manifest_sha256",
            "normalized_recipe",
            "base_model",
            "seed",
            "training_dataset_sha256",
            "fixture_split_manifest_sha256",
            "train_rows_sha256",
            "runtime_identity_sha256",
            "training_geometry",
            "predeclared_recipe_axes",
        },
        label,
    )
    if value["schema"] != 1 or value["kind"] != _CONDITION_KIND:
        raise ValueError("unsupported training-condition schema or kind")
    base_model = _object(value["base_model"], "training condition base_model")
    _exact_keys(base_model, {"model_id", "revision"}, "training condition base_model")
    if (
        not isinstance(base_model["model_id"], str)
        or not base_model["model_id"].strip()
    ):
        raise ValueError("training condition base model id is empty")
    revision = base_model["revision"]
    if not isinstance(revision, str) or not _GIT_SHA.fullmatch(revision):
        raise ValueError("training condition base model revision is not immutable")
    seed = value["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("training condition seed is invalid")
    runtime_identity_sha = value["runtime_identity_sha256"]
    if not isinstance(runtime_identity_sha, str) or not _SHA256.fullmatch(
        runtime_identity_sha
    ):
        raise ValueError("training condition runtime identity is invalid")
    geometry = _training_geometry(value["training_geometry"])
    raw_axes = value["predeclared_recipe_axes"]
    execution_recipe = krea_provenance.normalize_execution_recipe(
        value["normalized_recipe"], source_recipe=provenance["normalized_recipe"]
    )
    recipe_fields = execution_recipe["fields"]
    if (
        not isinstance(raw_axes, list)
        or not raw_axes
        or any(
            not isinstance(axis, str) or axis not in recipe_fields for axis in raw_axes
        )
        or raw_axes != sorted(set(raw_axes))
    ):
        raise ValueError("training condition recipe axes are invalid")
    if (
        value["source_arm_id"] != source_arm_id
        or value["provenance_manifest_sha256"] != provenance["manifest_sha256"]
        or value["normalized_recipe"] != execution_recipe
        or value["training_dataset_sha256"] != fixture["training_dataset_sha256"]
        or value["fixture_split_manifest_sha256"] != fixture_sha256
        or value["train_rows_sha256"] != fixture["training_rows_sha256"]
    ):
        raise ValueError(
            "training condition does not bind provenance/recipe/train split"
        )
    return {
        "schema": 1,
        "kind": "forge-krea-common-training-envelope",
        "concept_id": fixture["concept_id"],
        "concept_evidence_sha256": fixture["concept_evidence_sha256"],
        "fixture_split_manifest_sha256": fixture_sha256,
        "training_dataset_sha256": fixture["training_dataset_sha256"],
        "evaluation_dataset_sha256": fixture["evaluation_dataset_sha256"],
        "training_rows_sha256": fixture["training_rows_sha256"],
        "evaluation_rows_sha256": fixture["evaluation_rows_sha256"],
        "base_model": base_model,
        "seed": seed,
        "runtime_identity_sha256": runtime_identity_sha,
        "training_geometry": geometry,
        "predeclared_recipe_axes": raw_axes,
        "fixed_recipe_fields": {
            name: recipe_fields[name]
            for name in sorted(recipe_fields)
            if name not in raw_axes
        },
    }


def _validate_completion(
    value: dict[str, Any],
    raw: bytes,
    *,
    source_arm_id: str,
    provenance_sha256: str,
    condition_sha256: str,
    fixture_sha256: str,
    training_dataset_sha256: str,
    candidate_sha256: str,
    run_record_sha256: str,
    training_log_sha256: str,
) -> None:
    label = "trainer completion manifest"
    _canonical_control_file(value, raw, label)
    _exact_keys(
        value,
        {
            "schema",
            "kind",
            "source_arm_id",
            "provenance_manifest_sha256",
            "training_condition_sha256",
            "fixture_split_manifest_sha256",
            "training_dataset_sha256",
            "candidate_sha256",
            "run_record_sha256",
            "training_log_sha256",
            "natural_completion",
        },
        label,
    )
    expected = {
        "schema": 1,
        "kind": _COMPLETION_KIND,
        "source_arm_id": source_arm_id,
        "provenance_manifest_sha256": provenance_sha256,
        "training_condition_sha256": condition_sha256,
        "fixture_split_manifest_sha256": fixture_sha256,
        "training_dataset_sha256": training_dataset_sha256,
        "candidate_sha256": candidate_sha256,
        "run_record_sha256": run_record_sha256,
        "training_log_sha256": training_log_sha256,
        "natural_completion": True,
    }
    if value != expected:
        raise ValueError("trainer completion manifest is incomplete or unbound")


def _validate_source_approval(
    value: dict[str, Any],
    raw: bytes,
    *,
    source_arm_id: str,
    provenance: dict[str, Any],
) -> dict[str, str]:
    label = "source-normalization approval record"
    _canonical_control_file(value, raw, label)
    _exact_keys(
        value,
        {
            "schema",
            "kind",
            "decision",
            "reviewer_identity",
            "source_arm_id",
            "provenance_manifest_sha256",
        },
        label,
    )
    reviewer = _named_human(value["reviewer_identity"], "approval reviewer_identity")
    if (
        value["schema"] != 1
        or value["kind"] != _SOURCE_APPROVAL_KIND
        or value["decision"] != "approved"
        or value["source_arm_id"] != source_arm_id
        or value["provenance_manifest_sha256"] != provenance["manifest_sha256"]
        or reviewer != provenance["review_assertion"]["reviewer_identity"]
    ):
        raise ValueError("external approval record does not approve this provenance")
    return {"reviewer_identity": reviewer, "decision": "approved"}


def _validate_evaluator(value: Any) -> dict[str, Any]:
    evaluator = _object(value, "plan.evaluator")
    _exact_keys(
        evaluator,
        {
            "comfy_root",
            "comfy_python",
            "god_root",
            "expected_god_commit",
            "expected_comfy_commit",
            "expected_tooling_commit",
            "expected_evaluator_script_sha256",
            "expected_dataset_identity_module_sha256",
            "expected_eval_defaults",
            "expected_runtime_identity",
            "expected_assets",
            "cache_provenance_sha256",
            "containment",
        },
        "plan.evaluator",
        {
            "driver_python",
            "base_name",
            "port",
            "startup_timeout_s",
            "evaluation_timeout_s",
            "shutdown_timeout_s",
            "scorer_extension_policy",
            "scorer_timeout_profile",
        },
    )
    normalized = dict(evaluator)
    normalized["comfy_root"] = str(
        _safe_directory(evaluator["comfy_root"], "comfy_root")
    )
    normalized["god_root"] = str(_safe_directory(evaluator["god_root"], "god_root"))
    normalized["comfy_python"] = str(
        _safe_file(evaluator["comfy_python"], "comfy_python", executable=True)
    )
    normalized["driver_python"] = str(
        _safe_file(
            evaluator.get("driver_python", sys.executable),
            "driver_python",
            executable=True,
        )
    )
    for key in (
        "expected_god_commit",
        "expected_comfy_commit",
        "expected_tooling_commit",
    ):
        revision = normalized[key]
        if not isinstance(revision, str) or not _GIT_SHA.fullmatch(revision.lower()):
            raise ValueError(f"plan.evaluator.{key} is not an immutable revision")
        normalized[key] = revision.lower()
    script_sha = normalized["expected_evaluator_script_sha256"]
    if not isinstance(script_sha, str) or not _SHA256.fullmatch(script_sha):
        raise ValueError("expected evaluator-script SHA-256 is invalid")
    evaluator_script = (
        Path(__file__).with_name("evaluate_krea_local.py").resolve(strict=True)
    )
    if _sha256(evaluator_script) != script_sha:
        raise ValueError("predeclared evaluator-script identity mismatch")
    identity_module = (
        Path(__file__).with_name("krea_dataset_identity.py").resolve(strict=True)
    )
    identity_sha = normalized["expected_dataset_identity_module_sha256"]
    if (
        not isinstance(identity_sha, str)
        or not _SHA256.fullmatch(identity_sha)
        or _sha256(identity_module) != identity_sha
    ):
        raise ValueError("predeclared dataset-identity module mismatch")
    defaults = _object(normalized["expected_eval_defaults"], "expected_eval_defaults")
    _exact_keys(
        defaults,
        {"steps", "cfg", "denoise", "generations", "text_weight", "master_seed"},
        "expected_eval_defaults",
    )
    for key in ("steps", "generations", "master_seed"):
        if (
            isinstance(defaults[key], bool)
            or not isinstance(defaults[key], int)
            or defaults[key] <= 0
        ):
            raise ValueError(f"expected_eval_defaults.{key} is invalid")
    for key in ("cfg", "denoise", "text_weight"):
        value = defaults[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"expected_eval_defaults.{key} is invalid")
    if (
        not 0 <= float(defaults["denoise"]) <= 1
        or not 0 <= float(defaults["text_weight"]) <= 1
        or float(defaults["cfg"]) < 0
    ):
        raise ValueError("expected evaluator defaults are outside valid bounds")
    normalized["expected_eval_defaults"] = dict(defaults)
    runtime_identity = _object(
        normalized["expected_runtime_identity"], "expected_runtime_identity"
    )
    _exact_keys(
        runtime_identity,
        {"comfy_python_identity_sha256", "driver_python_identity_sha256"},
        "expected_runtime_identity",
    )
    for key, digest in runtime_identity.items():
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"expected runtime identity {key} is invalid")
    assets = _object(normalized["expected_assets"], "expected_assets")
    _exact_keys(
        assets,
        {"diffusion_model", "text_encoder", "vae"},
        "expected_assets",
    )
    normalized_assets: dict[str, Any] = {}
    for name, raw_asset in assets.items():
        asset = _object(raw_asset, f"expected_assets.{name}")
        _exact_keys(
            asset,
            {"canonical_path", "sha256", "bytes"},
            f"expected_assets.{name}",
        )
        if (
            not isinstance(asset["canonical_path"], str)
            or not asset["canonical_path"].startswith("/")
            or not isinstance(asset["sha256"], str)
            or not _SHA256.fullmatch(asset["sha256"])
            or isinstance(asset["bytes"], bool)
            or not isinstance(asset["bytes"], int)
            or asset["bytes"] <= 0
        ):
            raise ValueError(f"expected asset {name} identity is invalid")
        normalized_assets[name] = dict(asset)
    normalized["expected_assets"] = normalized_assets
    sealed_base_name = Path(normalized_assets["diffusion_model"]["canonical_path"]).name
    if "base_name" in normalized and normalized["base_name"] != sealed_base_name:
        raise ValueError("base_name differs from sealed diffusion-model identity")
    normalized["base_name"] = sealed_base_name
    if not isinstance(
        normalized["cache_provenance_sha256"], str
    ) or not _SHA256.fullmatch(normalized["cache_provenance_sha256"]):
        raise ValueError("cache_provenance_sha256 is invalid")
    containment = _object(normalized["containment"], "containment")
    _exact_keys(
        containment,
        {
            "mode",
            "term_grace_s",
            "systemd_run_path",
            "systemd_run_sha256",
            "systemctl_path",
            "systemctl_sha256",
            "unit_type",
            "network_policy",
        },
        "containment",
    )
    if containment["mode"] != "systemd_transient_service":
        raise ValueError("candidate evaluator requires systemd containment")
    if containment["unit_type"] != "transient_service":
        raise ValueError("candidate evaluator requires a transient systemd service")
    network_policy = _object(containment["network_policy"], "network_policy")
    expected_network_policy = {
        "private_network": True,
        "restrict_address_families": ["AF_UNIX", "AF_INET", "AF_INET6"],
        "loopback_allowed": True,
        "outbound_network_blocked": True,
    }
    if network_policy != expected_network_policy:
        raise ValueError(
            "candidate evaluator network policy is not offline/loopback-only"
        )
    normalized_containment = dict(containment)
    for prefix in ("systemd_run", "systemctl"):
        binary = _safe_file(
            containment[f"{prefix}_path"],
            f"containment.{prefix}_path",
            executable=True,
        )
        digest = containment[f"{prefix}_sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"containment.{prefix}_sha256 is invalid")
        if _sha256(binary.resolve(strict=True)) != digest:
            raise ValueError(f"containment.{prefix} binary identity mismatch")
        normalized_containment[f"{prefix}_path"] = str(binary)
    normalized["containment"] = normalized_containment
    grace = containment["term_grace_s"]
    if (
        isinstance(grace, bool)
        or not isinstance(grace, (int, float))
        or not math.isfinite(grace)
        or not 0.1 <= grace <= 60.0
    ):
        raise ValueError("containment.term_grace_s is invalid")
    if "port" in normalized:
        port = normalized["port"]
        if (
            not isinstance(port, int)
            or isinstance(port, bool)
            or not 1 <= port <= 65535
        ):
            raise ValueError("plan.evaluator.port is invalid")
    for key in ("startup_timeout_s", "evaluation_timeout_s", "shutdown_timeout_s"):
        if key in normalized:
            number = normalized[key]
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                or number <= 0
            ):
                raise ValueError(f"plan.evaluator.{key} must be finite and positive")
    if "scorer_extension_policy" in normalized:
        normalized["scorer_extension_policy"] = (
            krea_scorer_extension_policy.validate(
                normalized["scorer_extension_policy"]
            )
        )
    if "scorer_timeout_profile" in normalized:
        profile = normalized["scorer_timeout_profile"]
        krea_scorer_extension_policy.timeout_profile(profile)
    return normalized


def _validate_scorer_fixture_timeout(
    evaluator: dict[str, Any], fixture: dict[str, Any]
) -> dict[str, Any]:
    """Bind the effective scorer ceiling to the admitted fixture shape."""

    extension_present = "scorer_extension_policy" in evaluator
    profile_present = "scorer_timeout_profile" in evaluator
    if not extension_present and not profile_present:
        return {}
    if extension_present != profile_present:
        raise ValueError("scorer extension and timeout profile must be paired")
    role = evaluator.get("scorer_timeout_profile")
    if role != fixture.get("experimental_role"):
        raise ValueError("scorer timeout profile differs from fixture role")
    rows = fixture.get("evaluation_rows")
    if not isinstance(rows, list):
        raise ValueError("fixture evaluation rows are missing")
    defaults = evaluator.get("expected_eval_defaults")
    generations = defaults.get("generations") if isinstance(defaults, dict) else None
    return krea_scorer_extension_policy.validate_fixture_profile(
        role,
        evaluation_rows=len(rows),
        generations=generations,
    )


def _timeout_policy(evaluator: dict[str, Any]) -> dict[str, Any]:
    startup = float(evaluator.get("startup_timeout_s", 300.0))
    evaluation = float(evaluator.get("evaluation_timeout_s", 3600.0))
    shutdown = float(evaluator.get("shutdown_timeout_s", 20.0))
    total = startup + evaluation + shutdown + 60.0
    return {
        "startup_timeout_s": startup,
        "evaluation_timeout_s": evaluation,
        "shutdown_timeout_s": shutdown,
        "batch_overhead_s": 60.0,
        "total_candidate_timeout_s": total,
        "manager_runtime_max_s": total
        + float(evaluator["containment"]["term_grace_s"]),
        "containment": evaluator["containment"],
    }


def _plan_payload_sha256(plan: dict[str, Any]) -> str:
    payload = {
        key: value for key, value in plan.items() if key != "sealed_plan_approval"
    }
    return krea_provenance.canonical_sha256(payload)


def _plan_approval_expected(
    plan: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
    evaluator: dict[str, Any],
) -> dict[str, Any]:
    fixtures = sorted(
        {
            candidate["candidate_binding"]["fixture_split_manifest_sha256"]
            for candidate in candidates
            if candidate["candidate_binding"]["mode"] == "local_reproduction"
        }
    )
    conditions = sorted(
        (
            {
                "candidate_id": candidate["id"],
                "sha256": candidate["candidate_binding"]["training_condition_sha256"],
            }
            for candidate in candidates
            if candidate["candidate_binding"]["mode"] == "local_reproduction"
        ),
        key=lambda row: row["candidate_id"],
    )
    return {
        "plan_payload_sha256": _plan_payload_sha256(plan),
        "fixture_split_manifest_sha256s": fixtures,
        "candidate_training_conditions": conditions,
        "evaluator_identity": {
            "god_commit": evaluator["expected_god_commit"],
            "comfy_commit": evaluator["expected_comfy_commit"],
            "tooling_commit": evaluator["expected_tooling_commit"],
            "evaluator_script_sha256": evaluator["expected_evaluator_script_sha256"],
            "dataset_identity_module_sha256": evaluator[
                "expected_dataset_identity_module_sha256"
            ],
            "eval_defaults": evaluator["expected_eval_defaults"],
            "runtime_identity": evaluator["expected_runtime_identity"],
            "assets": evaluator["expected_assets"],
            "cache_provenance_sha256": evaluator["cache_provenance_sha256"],
            "scorer_extension_policy": evaluator.get("scorer_extension_policy"),
            "scorer_timeout_profile": evaluator.get("scorer_timeout_profile"),
        },
        "timeout_policy": _timeout_policy(evaluator),
        "batch_runner_sha256": _sha256(Path(__file__).resolve(strict=True)),
    }


def build_sealed_plan_approval(
    plan: dict[str, Any],
    *,
    reviewer_identity: str,
) -> dict[str, Any]:
    """Build the exact human-review record for an already complete plan.

    The returned record is data only.  A named human must review it and publish
    it independently; this helper does not authenticate or approve anything.
    """

    reviewer = _named_human(reviewer_identity, "sealed-plan reviewer_identity")
    if plan.get("schema") == 2:
        evaluator = _validate_evaluator(plan["evaluator"])
        candidates = []
        for raw in plan.get("candidates", []):
            row = _object(raw, "plan candidate")
            _, binding, binding_sha = _load_candidate_binding(
                row.get("candidate_binding"), "candidate binding"
            )
            candidates.append(
                {
                    "id": row.get("id"),
                    "candidate_binding": {
                        "mode": binding.get("mode"),
                        "binding_manifest_sha256": binding_sha,
                    },
                }
            )
        candidates.sort(key=lambda item: item["id"])
        expected = _v2_plan_approval_expected(
            plan, candidates=candidates, evaluator=evaluator
        )
        return {
            "schema": 2,
            "kind": "forge-krea-exact-score-plan-approval",
            "decision": "approved",
            "reviewer_identity": reviewer,
            **expected,
        }
    provisional = dict(plan)
    provisional.setdefault(
        "sealed_plan_approval", {"path": "<external>", "sha256": "0" * 64}
    )
    # Plan validation would need the approval record itself, so normalize the
    # evaluator directly and consume normalized candidate bindings supplied by
    # the trusted producer/caller.
    evaluator = _validate_evaluator(provisional["evaluator"])
    candidates: list[dict[str, Any]] = []
    for raw in provisional.get("candidates", []):
        row = _object(raw, "plan candidate")
        binding = _object(row.get("candidate_binding"), "candidate binding")
        mode = binding.get("mode")
        normalized_binding: dict[str, Any] = {"mode": mode}
        if mode == "local_reproduction":
            for source_key, target_key in (
                ("fixture_split_manifest", "fixture_split_manifest_sha256"),
                ("training_condition", "training_condition_sha256"),
            ):
                bound = _object(binding.get(source_key), source_key)
                normalized_binding[target_key] = bound.get("sha256")
        candidates.append(
            {"id": row.get("id"), "candidate_binding": normalized_binding}
        )
    expected = _plan_approval_expected(
        provisional, candidates=candidates, evaluator=evaluator
    )
    return {
        "schema": 1,
        "kind": _PLAN_APPROVAL_KIND,
        "decision": "approved",
        "reviewer_identity": reviewer,
        **expected,
    }


def _validate_plan_approval(
    value: dict[str, Any],
    raw: bytes,
    *,
    plan: dict[str, Any],
    candidates: list[dict[str, Any]],
    evaluator: dict[str, Any],
) -> dict[str, Any]:
    label = "sealed plan approval"
    _canonical_control_file(value, raw, label)
    expected_fields = _plan_approval_expected(
        plan, candidates=candidates, evaluator=evaluator
    )
    _exact_keys(
        value,
        {
            "schema",
            "kind",
            "decision",
            "reviewer_identity",
            *expected_fields.keys(),
        },
        label,
    )
    reviewer = _named_human(value["reviewer_identity"], "plan reviewer_identity")
    expected = {
        "schema": 1,
        "kind": _PLAN_APPROVAL_KIND,
        "decision": "approved",
        "reviewer_identity": reviewer,
        **expected_fields,
    }
    if value != expected:
        raise ValueError("sealed plan approval is incomplete or does not bind the plan")
    return {"reviewer_identity": reviewer, "decision": "approved"}


def _validate_plan_v1(
    plan: dict[str, Any]
) -> tuple[Path, str, list[dict[str, Any]], dict[str, Any]]:
    _exact_keys(
        plan,
        {"schema", "dataset", "candidates", "evaluator", "sealed_plan_approval"},
        "plan",
    )
    if plan["schema"] != _SCHEMA:
        raise ValueError("unsupported batch plan schema")
    dataset_spec = _object(plan["dataset"], "plan.dataset")
    _exact_keys(dataset_spec, {"path", "sha256"}, "plan.dataset")
    dataset = _safe_directory(dataset_spec["path"], "dataset")
    evaluation_dataset_sha = dataset_spec["sha256"]
    if not isinstance(evaluation_dataset_sha, str) or not _SHA256.fullmatch(
        evaluation_dataset_sha
    ):
        raise ValueError("plan.dataset.sha256 must be a full SHA-256")
    evaluator = _validate_evaluator(plan["evaluator"])
    raw_candidates = plan["candidates"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("plan.candidates must be a non-empty array")

    candidates: list[dict[str, Any]] = []
    common_training_envelope: dict[str, Any] | None = None
    common_training_envelope_sha: str | None = None
    ids: set[str] = set()
    hashes: set[str] = set()
    paths: set[Path] = set()
    for index, raw in enumerate(raw_candidates):
        row = _object(raw, f"plan.candidates[{index}]")
        _exact_keys(
            row,
            {
                "id",
                "source_arm_id",
                "path",
                "sha256",
                "provenance",
                "candidate_binding",
            },
            f"plan.candidates[{index}]",
        )
        candidate_id = row["id"]
        source_arm_id = row["source_arm_id"]
        for identifier, label in (
            (candidate_id, "candidate id"),
            (source_arm_id, "source arm id"),
        ):
            if (
                not isinstance(identifier, str)
                or not _SAFE_ID.fullmatch(identifier)
                or identifier in {".", ".."}
            ):
                raise ValueError(f"invalid {label} at index {index}")
        candidate_path = _safe_file(row["path"], f"candidate {candidate_id}")
        candidate_sha = row["sha256"]
        if not isinstance(candidate_sha, str) or not _SHA256.fullmatch(candidate_sha):
            raise ValueError(f"candidate {candidate_id} has invalid SHA-256")
        if _sha256(candidate_path) != candidate_sha:
            raise ValueError(f"candidate {candidate_id} SHA-256 mismatch")
        provenance_path = _safe_file(row["provenance"], f"provenance {candidate_id}")
        provenance, provenance_file_sha, provenance_raw = _load_json_file(
            provenance_path, f"provenance {candidate_id}"
        )
        krea_provenance.validate_manifest(provenance)
        _canonical_control_file(
            provenance, provenance_raw, f"candidate {candidate_id} provenance"
        )
        if provenance["source_arm_id"] != source_arm_id:
            raise ValueError(f"candidate {candidate_id} provenance source-arm mismatch")
        if provenance["evaluator_sha"] != evaluator["expected_god_commit"]:
            raise ValueError(f"candidate {candidate_id} provenance evaluator mismatch")
        if provenance["review_assertion"]["status"] != "approved":
            raise ValueError(
                f"candidate {candidate_id} review assertion is not approved"
            )
        if candidate_id in ids or candidate_sha in hashes or candidate_path in paths:
            raise ValueError(f"duplicate candidate id, bytes, or path: {candidate_id}")

        binding = _object(row["candidate_binding"], f"candidate {candidate_id} binding")
        mode = binding.get("mode")
        source_artifact_sha = provenance["files"]["source_artifact"]["sha256"]
        private: dict[str, str] = {}
        if mode == "local_reproduction":
            _exact_keys(
                binding,
                {
                    "mode",
                    "training_dataset_sha256",
                    "evaluation_dataset_sha256",
                    "fixture_split_manifest",
                    "training_condition",
                    "completion_manifest",
                    "run_record",
                    "training_log",
                    "source_normalization_approval",
                },
                f"candidate {candidate_id} binding",
            )
            if provenance["adaptation_target"]["mode"] != mode:
                raise ValueError(
                    f"candidate {candidate_id} binding contradicts adaptation target"
                )
            if candidate_sha == source_artifact_sha:
                raise ValueError(
                    f"candidate {candidate_id} local reproduction equals source artifact"
                )
            training_dataset_sha = binding["training_dataset_sha256"]
            bound_evaluation_sha = binding["evaluation_dataset_sha256"]
            for digest, label in (
                (training_dataset_sha, "training dataset"),
                (bound_evaluation_sha, "evaluation dataset"),
            ):
                if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                    raise ValueError(f"candidate {candidate_id} {label} SHA is invalid")
            if bound_evaluation_sha != evaluation_dataset_sha:
                raise ValueError(
                    f"candidate {candidate_id} evaluation dataset identity mismatch"
                )
            if training_dataset_sha == bound_evaluation_sha:
                raise ValueError(
                    f"candidate {candidate_id} training/evaluation identities are not split"
                )

            fixture_path, fixture_sha, fixture_value, fixture_raw = _bound_file(
                binding["fixture_split_manifest"],
                f"candidate {candidate_id} fixture split manifest",
            )
            fixture = _validate_fixture_split(fixture_value, fixture_raw)
            if (
                fixture["training_dataset_sha256"] != training_dataset_sha
                or fixture["evaluation_dataset_sha256"] != bound_evaluation_sha
            ):
                raise ValueError(
                    f"candidate {candidate_id} fixture split dataset mismatch"
                )
            condition_path, condition_sha, condition, condition_raw = _bound_file(
                binding["training_condition"],
                f"candidate {candidate_id} training condition",
            )
            condition_envelope = _validate_condition(
                condition,
                condition_raw,
                source_arm_id=source_arm_id,
                provenance=provenance,
                fixture_sha256=fixture_sha,
                fixture=fixture,
            )
            condition_envelope_sha = krea_provenance.canonical_sha256(
                condition_envelope
            )
            if common_training_envelope is None:
                common_training_envelope = condition_envelope
                common_training_envelope_sha = condition_envelope_sha
            elif (
                condition_envelope_sha != common_training_envelope_sha
                or condition_envelope != common_training_envelope
            ):
                raise ValueError(
                    f"candidate {candidate_id} escaped the common training envelope"
                )
            run_path, run_sha = _bound_bytes_file(
                binding["run_record"], f"candidate {candidate_id} run record"
            )
            log_path, log_sha = _bound_bytes_file(
                binding["training_log"], f"candidate {candidate_id} training log"
            )
            completion_path, completion_sha, completion, completion_raw = _bound_file(
                binding["completion_manifest"],
                f"candidate {candidate_id} completion manifest",
            )
            _validate_completion(
                completion,
                completion_raw,
                source_arm_id=source_arm_id,
                provenance_sha256=provenance["manifest_sha256"],
                condition_sha256=condition_sha,
                fixture_sha256=fixture_sha,
                training_dataset_sha256=training_dataset_sha,
                candidate_sha256=candidate_sha,
                run_record_sha256=run_sha,
                training_log_sha256=log_sha,
            )
            approval_path, approval_sha, approval, approval_raw = _bound_file(
                binding["source_normalization_approval"],
                f"candidate {candidate_id} source-normalization approval record",
            )
            approval_summary = _validate_source_approval(
                approval,
                approval_raw,
                source_arm_id=source_arm_id,
                provenance=provenance,
            )
            normalized_binding = {
                "mode": mode,
                "normalized_recipe": condition["normalized_recipe"],
                "training_dataset_sha256": training_dataset_sha,
                "evaluation_dataset_sha256": bound_evaluation_sha,
                "fixture_split_manifest_sha256": fixture_sha,
                "training_condition_sha256": condition_sha,
                "completion_manifest_sha256": completion_sha,
                "run_record_sha256": run_sha,
                "training_log_sha256": log_sha,
                "source_normalization_approval_sha256": approval_sha,
                "source_normalization_approval": approval_summary,
            }
            bound_paths = {
                "fixture_split_manifest": (fixture_path, fixture_sha),
                "training_condition": (condition_path, condition_sha),
                "completion_manifest": (completion_path, completion_sha),
                "run_record": (run_path, run_sha),
                "training_log": (log_path, log_sha),
                "source_normalization_approval": (approval_path, approval_sha),
            }
            for label, (bound_path, bound_sha) in bound_paths.items():
                private[f"_{label}_path"] = str(bound_path)
                private[f"_{label}_sha256"] = bound_sha
        elif mode == "direct_public_artifact":
            _exact_keys(
                binding,
                {"mode", "source_normalization_approval"},
                f"candidate {candidate_id} binding",
            )
            if provenance["adaptation_target"]["mode"] != mode:
                raise ValueError(
                    f"candidate {candidate_id} binding contradicts adaptation target"
                )
            matched = provenance["matched_concept"]
            if (
                not matched["available"]
                or matched["dataset_sha256"] != evaluation_dataset_sha
            ):
                raise ValueError(
                    f"candidate {candidate_id} direct public artifact lacks matched concept"
                )
            if candidate_sha != source_artifact_sha:
                raise ValueError(
                    f"candidate {candidate_id} direct artifact SHA differs from source"
                )
            krea_provenance.validate_manifest(
                provenance, source_artifact_path=candidate_path
            )
            approval_path, approval_sha, approval, approval_raw = _bound_file(
                binding["source_normalization_approval"],
                f"candidate {candidate_id} source-normalization approval record",
            )
            approval_summary = _validate_source_approval(
                approval,
                approval_raw,
                source_arm_id=source_arm_id,
                provenance=provenance,
            )
            normalized_binding = {
                "mode": mode,
                "evaluation_dataset_sha256": evaluation_dataset_sha,
                "source_normalization_approval_sha256": approval_sha,
                "source_normalization_approval": approval_summary,
            }
            private = {
                "_source_normalization_approval_path": str(approval_path),
                "_source_normalization_approval_sha256": approval_sha,
            }
        else:
            raise ValueError(f"candidate {candidate_id} has unsupported binding mode")

        ids.add(candidate_id)
        hashes.add(candidate_sha)
        paths.add(candidate_path)
        candidates.append(
            {
                "id": candidate_id,
                "source_arm_id": source_arm_id,
                "path": candidate_path,
                "sha256": candidate_sha,
                "provenance": provenance,
                "provenance_path": provenance_path,
                "provenance_file_sha256": provenance_file_sha,
                "candidate_binding": normalized_binding,
                **private,
            }
        )
    candidates.sort(key=lambda item: item["id"])
    approval_path, approval_sha, approval, approval_raw = _bound_file(
        plan["sealed_plan_approval"], "sealed plan approval"
    )
    approval_summary = _validate_plan_approval(
        approval,
        approval_raw,
        plan=plan,
        candidates=candidates,
        evaluator=evaluator,
    )
    evaluator["_common_training_envelope"] = common_training_envelope
    evaluator["_common_training_envelope_sha256"] = common_training_envelope_sha
    evaluator["_sealed_plan_approval_path"] = str(approval_path)
    evaluator["_sealed_plan_approval_sha256"] = approval_sha
    evaluator["_sealed_plan_approval"] = approval_summary
    evaluator["_plan_payload_sha256"] = _plan_payload_sha256(plan)
    evaluator["_batch_runner_sha256"] = _sha256(Path(__file__).resolve(strict=True))
    return dataset, evaluation_dataset_sha, candidates, evaluator


def _load_candidate_binding(value: Any, label: str) -> tuple[Path, dict[str, Any], str]:
    binding = _object(value, label)
    _exact_keys(binding, {"path", "sha256"}, label)
    path, digest, document, raw = _bound_file(binding, label)
    _canonical_control_file(document, raw, label)
    return path, document, digest


def _validate_cross_fixture_review_surface(
    value: dict[str, Any],
    *,
    fixture: dict[str, Any],
    source_path: Path | None = None,
    fixture_admission_validator: Any | None = None,
) -> dict[str, Any]:
    """Validate the admitted six-fixture review surface available to scoring.

    Current schema-2 discovery binds the complete admission envelope whose
    blinded acceptance validates the owner-ratified agent review. The legacy
    named-human record remains readable only for older plans.
    """

    if value.get("kind") == "forge-krea-fixture-admission-envelope":
        if source_path is None:
            raise ValueError("agent cross-fixture admission requires its bound path")
        if fixture_admission_validator is None:
            try:
                from . import krea_fixture_admission
            except ImportError:  # pragma: no cover - direct script execution.
                import krea_fixture_admission  # type: ignore[no-redef]

            fixture_admission_validator = krea_fixture_admission
        resolved = fixture_admission_validator.validate_envelope(source_path)
        role = fixture.get("experimental_role")
        admitted = _object(resolved.get("fixtures"), "admitted fixtures").get(role)
        acceptance = _object(resolved.get("blinded_acceptance"), "blinded acceptance")
        assertions = _object(
            acceptance.get("assertions"), "blinded acceptance assertions"
        )
        manifests = _object(
            acceptance.get("fixture_manifest_sha256s"),
            "accepted six-fixture manifest map",
        )
        if (
            resolved.get("envelope") != value
            or role not in {"D1", "D2"}
            or admitted != fixture
            or manifests.get(role) != fixture.get("manifest_sha256")
            or assertions.get(
                "all_six_cross_fixture_review_preexists_discovery_execution"
            )
            is not True
            or assertions.get("agent_review_is_not_human_review") is not True
            or acceptance.get("decision") != "accepted_for_d1_d2_discovery_admission"
            or acceptance.get("c1c4_revealed") is not False
        ):
            raise ValueError(
                "agent cross-fixture admission does not bind this discovery fixture"
            )
        return value

    _exact_keys(
        value,
        {
            "schema",
            "kind",
            "fixture_manifest_sha256s",
            "reviewer_identity",
            "reviewed_at_utc",
            "decision",
            "reviewed_pairs",
            "flagged_pairs",
            "claim_limit",
        },
        "cross-fixture human review",
    )
    manifests = _object(
        value["fixture_manifest_sha256s"], "cross-fixture manifest identities"
    )
    if set(manifests) != {"D1", "D2", "C1", "C2", "C3", "C4"} or any(
        not isinstance(digest, str) or not _SHA256.fullmatch(digest)
        for digest in manifests.values()
    ):
        raise ValueError("cross-fixture review does not bind all six fixtures")
    role = fixture["experimental_role"]
    pairs = value["reviewed_pairs"]
    if (
        value["schema"] != 1
        or value["kind"] != "forge-krea-cross-fixture-human-similarity-review"
        or manifests[role] != fixture["manifest_sha256"]
        or value["decision"] != "passed"
        or value["flagged_pairs"] != []
        or value["claim_limit"] != "cross-fixture-nonoverlap-only"
        or not isinstance(pairs, list)
        or not pairs
    ):
        raise ValueError("cross-fixture review is incomplete or does not pass")
    normalized_pairs: list[tuple[str, str]] = []
    for pair in pairs:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(item, str) or ":" not in item for item in pair)
            or pair[0] >= pair[1]
            or pair[0].split(":", 1)[0] == pair[1].split(":", 1)[0]
        ):
            raise ValueError("cross-fixture reviewed pair is malformed")
        normalized_pairs.append((pair[0], pair[1]))
    if normalized_pairs != sorted(set(normalized_pairs)):
        raise ValueError("cross-fixture reviewed pairs are duplicate or unsorted")
    _named_human(value["reviewer_identity"], "cross-fixture reviewer_identity")
    krea_fixture.canonical_utc(
        value["reviewed_at_utc"], "cross-fixture reviewed_at_utc"
    )
    return value


def seal_campaign_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Seal the complete set of runs/candidates before exact scores exist."""

    if "manifest_sha256" in payload:
        raise ValueError("unsealed campaign payload contains manifest_sha256")
    manifest = {
        **payload,
        "manifest_sha256": krea_provenance.canonical_sha256(payload),
    }
    _validate_campaign_manifest(manifest)
    return manifest


def _validate_campaign_manifest(value: Any) -> dict[str, Any]:
    manifest = _object(value, "exact-score campaign manifest")
    _exact_keys(
        manifest,
        {
            "schema",
            "kind",
            "fixture_manifest_sha256",
            "discovery_plan_sha256",
            "runs",
            "zero_control_manifest_sha256",
            "decision_contract",
            "confirmation_contract",
            "manifest_sha256",
        },
        "exact-score campaign manifest",
        {"historical_training_evidence_validator"},
    )
    body = {key: item for key, item in manifest.items() if key != "manifest_sha256"}
    if (
        manifest["schema"] != 2
        or manifest["kind"] != "forge-krea-exact-score-campaign"
        or manifest["manifest_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("exact-score campaign manifest identity is invalid")
    for key in (
        "fixture_manifest_sha256",
        "discovery_plan_sha256",
        "zero_control_manifest_sha256",
    ):
        digest = manifest[key]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"campaign {key} is invalid")
    if manifest["decision_contract"] != _DISCOVERY_DECISION_BINDING:
        raise ValueError("campaign decision contract differs from the frozen policy")
    if manifest["confirmation_contract"] != _CONFIRMATION_DECISION_BINDING:
        raise ValueError(
            "campaign confirmation contract differs from the frozen policy"
        )
    if "historical_training_evidence_validator" in manifest:
        krea_historical_training_evidence.validate_identity(
            manifest["historical_training_evidence_validator"]
        )
    runs = manifest["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("exact-score campaign has no runs")
    normalized_runs = []
    seen_arms: set[str] = set()
    seen_completions: set[str] = set()
    for index, raw in enumerate(runs):
        run = _object(raw, f"campaign.runs[{index}]")
        _exact_keys(
            run,
            {
                "arm_id",
                "execution_plan_sha256",
                "run_completion_sha256",
                "candidates",
            },
            f"campaign.runs[{index}]",
        )
        arm_id = run["arm_id"]
        if not isinstance(arm_id, str) or not _SAFE_ID.fullmatch(arm_id):
            raise ValueError("campaign arm_id is invalid")
        for key in ("execution_plan_sha256", "run_completion_sha256"):
            if not isinstance(run[key], str) or not _SHA256.fullmatch(run[key]):
                raise ValueError(f"campaign run {key} is invalid")
        if arm_id in seen_arms or run["run_completion_sha256"] in seen_completions:
            raise ValueError("campaign contains duplicate arm or run completion")
        seen_arms.add(arm_id)
        seen_completions.add(run["run_completion_sha256"])
        candidates = run["candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("campaign run has no candidates")
        normalized_candidates = []
        seen_ids: set[str] = set()
        seen_hashes: set[str] = set()
        for candidate in candidates:
            candidate = _object(candidate, "campaign run candidate")
            _exact_keys(
                candidate,
                {"candidate_id", "sha256", "bytes", "step", "fraction"},
                "campaign run candidate",
            )
            candidate_id = candidate["candidate_id"]
            digest = candidate["sha256"]
            if (
                not isinstance(candidate_id, str)
                or not _SAFE_ID.fullmatch(candidate_id)
                or not isinstance(digest, str)
                or not _SHA256.fullmatch(digest)
                or candidate_id in seen_ids
                or digest in seen_hashes
                or isinstance(candidate["bytes"], bool)
                or not isinstance(candidate["bytes"], int)
                or candidate["bytes"] <= 0
                or isinstance(candidate["step"], bool)
                or not isinstance(candidate["step"], int)
                or candidate["step"] <= 0
            ):
                raise ValueError("campaign candidate identity is invalid")
            fraction = _object(candidate["fraction"], "campaign candidate fraction")
            _exact_keys(fraction, {"numerator", "denominator"}, "candidate fraction")
            if (
                fraction["numerator"] != candidate["step"]
                or isinstance(fraction["denominator"], bool)
                or not isinstance(fraction["denominator"], int)
                or fraction["denominator"] < candidate["step"]
            ):
                raise ValueError("campaign candidate fraction is invalid")
            seen_ids.add(candidate_id)
            seen_hashes.add(digest)
            normalized_candidates.append(dict(candidate))
        if normalized_candidates != sorted(
            normalized_candidates, key=lambda row: (row["step"], row["sha256"])
        ):
            raise ValueError("campaign candidates must be sorted by step/hash")
        normalized_runs.append({**run, "candidates": normalized_candidates})
    if normalized_runs != sorted(normalized_runs, key=lambda row: row["arm_id"]):
        raise ValueError("campaign runs must be sorted by arm_id")
    return manifest


def _schema2_training_run_envelopes(
    *,
    campaign: dict[str, Any],
    candidates: list[dict[str, Any]],
    envelopes: list[dict[str, Any]],
    decision_context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return the decision-consumer view of the sealed training envelopes.

    The plan's decision context is sealed and independently approved before the
    run.  Boundary context additionally binds a validated frozen discovery
    record, its exact checkpoint rule, and the exact planned artifact.  The
    selected digest is cross-checked again against the sealed campaign, score
    row, and execution envelope immediately before publication.
    """

    published = [dict(envelope) for envelope in envelopes]
    if decision_context["phase"] != "boundary":
        return published

    runs = campaign["runs"]
    run = runs[0]
    local = [row for row in candidates if row["mode"] == "local_run_candidate"]
    if len(local) != 1 or len(published) != 1:
        raise RuntimeError(
            "single-candidate campaign does not have one local score/envelope"
        )
    candidate = local[0]
    sealed_candidate = run["candidates"][0]
    envelope = published[0]
    if (
        candidate["candidate_id"] != sealed_candidate["candidate_id"]
        or candidate["candidate_sha256"] != sealed_candidate["sha256"]
        or candidate["candidate_bytes"] != sealed_candidate["bytes"]
        or candidate["step"] != sealed_candidate["step"]
        or candidate["fraction_numerator"] != sealed_candidate["fraction"]["numerator"]
        or candidate["fraction_denominator"]
        != sealed_candidate["fraction"]["denominator"]
        or candidate["arm_id"] != run["arm_id"]
        or candidate["execution_plan_sha256"] != run["execution_plan_sha256"]
        or candidate["run_completion_sha256"] != run["run_completion_sha256"]
        or envelope["arm_id"] != run["arm_id"]
        or envelope["execution_plan_sha256"] != run["execution_plan_sha256"]
        or decision_context["candidate_family_id"] != run["arm_id"]
        or decision_context["selected_candidate"] != sealed_candidate
    ):
        raise RuntimeError(
            "boundary candidate decision is not bound to the sealed campaign"
        )
    envelope["candidate_decision"] = {
        "mode": "frozen_checkpoint_rule",
        "selected_candidate_sha256": decision_context["selected_candidate"]["sha256"],
        "decision_completed_before_export_reserve": True,
        "fallback_used": False,
    }
    return published


def _validate_score_decision_context(
    value: Any, *, campaign: dict[str, Any]
) -> dict[str, Any]:
    """Validate an explicit, pre-score phase/frozen-checkpoint binding.

    A one-run/one-candidate campaign is otherwise indistinguishable from a
    truncated discovery or confirmation batch.  Such a campaign is accepted
    only as an explicitly bound boundary cell.  Conversely, a boundary context
    cannot be attached to a multi-run or multi-candidate campaign.
    """

    context = _object(value, "score decision context")
    common = {"schema", "kind", "phase"}
    phase = context.get("phase")
    if phase not in {"discovery", "confirmation", "boundary"}:
        raise ValueError("score decision context phase is invalid")
    if phase != "boundary":
        _exact_keys(context, common, "score decision context")
        if len(campaign["runs"]) == 1 and len(campaign["runs"][0]["candidates"]) == 1:
            raise ValueError(
                "single-run/single-candidate score plan is boundary-ambiguous"
            )
        if (
            context["schema"] != 1
            or context["kind"] != "forge-krea-exact-score-decision-context"
        ):
            raise ValueError("score decision context identity is invalid")
        return dict(context)

    _exact_keys(
        context,
        common
        | {
            "frozen_discovery_decision",
            "candidate_family_id",
            "checkpoint_rule_sha256",
            "selected_candidate",
            "decision_completed_before_export_reserve",
            "fallback_used",
        },
        "boundary score decision context",
    )
    if (
        context["schema"] != 1
        or context["kind"] != "forge-krea-exact-score-decision-context"
        or context["decision_completed_before_export_reserve"] is not True
        or context["fallback_used"] is not False
        or len(campaign["runs"]) != 1
        or len(campaign["runs"][0]["candidates"]) != 1
    ):
        raise ValueError("boundary decision context/campaign shape is invalid")

    decision_path, decision_file_sha, decision, decision_raw = _bound_file(
        context["frozen_discovery_decision"], "frozen discovery decision"
    )
    _canonical_control_file(decision, decision_raw, "frozen discovery decision")
    # Keep the producer and consumer on one definition of a valid, recomputable
    # frozen discovery decision.  This import is lazy to keep direct-script use
    # and the training-evidence import graph acyclic.
    try:
        from . import krea_decision
    except ImportError:  # pragma: no cover - direct CLI execution.
        import krea_decision  # type: ignore[no-redef]
    krea_decision._validate_discovery_record(decision)
    family = context["candidate_family_id"]
    if (
        not isinstance(family, str)
        or not _SAFE_ID.fullmatch(family)
        or family == "K0"
        or decision["outcome"] != "finalists_frozen"
        or family not in decision["finalist_family_ids"]
        or family not in decision["checkpoint_rules"]
    ):
        raise ValueError("boundary family is not a frozen non-control finalist")
    rule = decision["checkpoint_rules"][family]
    rule_sha = krea_provenance.canonical_sha256(rule)
    if context["checkpoint_rule_sha256"] != rule_sha:
        raise ValueError("boundary checkpoint rule SHA-256 mismatch")

    selected = _object(context["selected_candidate"], "boundary selected candidate")
    _exact_keys(
        selected,
        {"candidate_id", "sha256", "bytes", "step", "fraction"},
        "boundary selected candidate",
    )
    run = campaign["runs"][0]
    if family != run["arm_id"] or selected != run["candidates"][0]:
        raise ValueError("boundary selection differs from the sealed campaign")

    # Map the frozen target to the nearest integer step of this boundary run;
    # an exact half-step chooses the earlier step.  This prevents a sole
    # candidate from making the nearest-candidate check vacuously true.
    fraction = selected["fraction"]
    denominator = fraction["denominator"]
    target = Fraction(str(rule["target_fraction"])) * denominator
    lower = target.numerator // target.denominator
    upper = lower + (target.denominator != 1)
    expected_step = lower if target - lower <= upper - target else upper
    expected_step = min(denominator, max(1, expected_step))
    if selected["step"] != expected_step or fraction != {
        "numerator": expected_step,
        "denominator": denominator,
    }:
        raise ValueError(
            "boundary candidate step/fraction does not map the frozen checkpoint rule"
        )
    return {
        **context,
        "_frozen_discovery_decision_path": str(decision_path),
        "_frozen_discovery_decision_file_sha256": decision_file_sha,
    }


def _validate_safetensors_identity(value: Any, label: str) -> dict[str, Any]:
    identity = _object(value, label)
    _exact_keys(
        identity,
        {
            "bytes",
            "header_sha256",
            "metadata",
            "metadata_sha256",
            "tensor_layout_sha256",
            "tensor_count",
            "tensor_data_bytes",
        },
        label,
    )
    for key in ("header_sha256", "metadata_sha256", "tensor_layout_sha256"):
        if not isinstance(identity[key], str) or not _SHA256.fullmatch(identity[key]):
            raise ValueError(f"{label}.{key} is invalid")
    for key in ("bytes", "tensor_count", "tensor_data_bytes"):
        if (
            isinstance(identity[key], bool)
            or not isinstance(identity[key], int)
            or identity[key] <= 0
        ):
            raise ValueError(f"{label}.{key} is invalid")
    metadata = _object(identity["metadata"], f"{label}.metadata")
    if any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in metadata.items()
    ) or identity["metadata_sha256"] != krea_provenance.canonical_sha256(metadata):
        raise ValueError(f"{label}.metadata is invalid")
    return identity


def _validate_run_completion(
    completion: dict[str, Any],
    *,
    execution_plan: dict[str, Any],
    execution_plan_file_sha: str,
    execution_approval: dict[str, Any],
    execution_approval_file_sha: str,
    fixture: dict[str, Any],
    execution_plan_validator: Any = krea_execution_plan,
    expected_execution_surface_policy_sha256: str = (
        krea_execution_surface_policy.POLICY["policy_sha256"]
    ),
) -> None:
    _exact_keys(
        completion,
        {
            "schema",
            "kind",
            "arm_id",
            "task_id",
            "execution_plan_sha256",
            "execution_plan_file_sha256",
            "execution_approval_sha256",
            "execution_approval_file_sha256",
            "fixture_manifest_sha256",
            "training_dataset_sha256",
            "training_rows_sha256",
            "training_archive",
            "execution_envelope_sha256",
            "host_execution_identity_sha256",
            "throughput_profile_sha256",
            "discovery_profile_index_sha256",
            "discovery_profile_index_file_sha256",
            "discovery_execution_authorization_sha256",
            "discovery_execution_authorization_file_sha256",
            "host_bootstrap_receipt_sha256",
            "host_bootstrap_receipt_file_sha256",
            "execution_surface_policy_sha256",
            "execution_surface",
            "execution_scope",
            "budget_plan_sha256",
            "schedule",
            "run_record_sha256",
            "training_log_sha256",
            "natural_completion",
            "in_task_proxy_selection",
            "candidates",
        },
        "run completion",
    )
    if (
        completion["schema"] != 3
        or completion["kind"] != "forge-krea-training-completion"
        or completion["arm_id"] != execution_plan["arm_id"]
        or completion["task_id"] != execution_plan["task_id"]
        or completion["execution_plan_sha256"] != execution_plan["plan_sha256"]
        or completion["execution_plan_file_sha256"] != execution_plan_file_sha
        or completion["execution_approval_sha256"]
        != execution_approval["approval_sha256"]
        or completion["execution_approval_file_sha256"] != execution_approval_file_sha
        or completion["fixture_manifest_sha256"] != fixture["manifest_sha256"]
        or completion["training_dataset_sha256"]
        != fixture["training_dataset_identity"]["sha256"]
        or completion["training_rows_sha256"]
        != krea_provenance.canonical_sha256(fixture["training_rows"])
        or completion["training_archive"]
        != {
            "sha256": fixture["training_archive"]["sha256"],
            "bytes": fixture["training_archive"]["bytes"],
        }
        or completion["execution_envelope_sha256"]
        != execution_plan["execution_envelope_sha256"]
        or completion["natural_completion"] is not True
        or completion["in_task_proxy_selection"] != {"enabled": False, "reserve_s": 0}
        or completion["execution_surface_policy_sha256"]
        != expected_execution_surface_policy_sha256
        or completion["execution_surface"] != "staged_host_venv"
        or completion["execution_scope"] != "discovery_only"
    ):
        raise ValueError("run completion does not bind the approved natural run")
    resolved = execution_plan_validator.validate_plan(execution_plan)
    if (
        completion["host_execution_identity_sha256"]
        != resolved["host_execution_manifest"]["host_execution_identity_sha256"]
        or completion["throughput_profile_sha256"]
        != resolved["throughput_profile"]["profile_sha256"]
        or completion["budget_plan_sha256"] != execution_plan["budget_plan_sha256"]
        or completion["schedule"] != execution_plan["schedule"]
        or completion["discovery_profile_index_sha256"]
        != resolved["discovery_profile_index"]["index_sha256"]
        or completion["discovery_profile_index_file_sha256"]
        != resolved["discovery_profile_index"]["file_sha256"]
        or completion["discovery_execution_authorization_sha256"]
        != resolved["discovery_execution_authorization"]["authorization_sha256"]
        or completion["discovery_execution_authorization_file_sha256"]
        != resolved["discovery_execution_authorization"]["file_sha256"]
        or completion["host_bootstrap_receipt_sha256"]
        != resolved["host_bootstrap_receipt"]["receipt_sha256"]
        or completion["host_bootstrap_receipt_file_sha256"]
        != resolved["host_bootstrap_receipt"]["file_sha256"]
    ):
        raise ValueError("run completion execution envelope is incomplete")
    rows = completion["candidates"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("run completion has no candidates")
    seen: set[tuple[str, int]] = set()
    for row in rows:
        row = _object(row, "run completion candidate")
        _exact_keys(
            row,
            {
                "candidate_id",
                "sha256",
                "step",
                "fraction_numerator",
                "fraction_denominator",
                "aliases",
                "bytes",
                "safetensors",
            },
            "run completion candidate",
        )
        key = (row["sha256"], row["step"])
        if key in seen:
            raise ValueError("run completion contains duplicate candidates")
        seen.add(key)
        if (
            not isinstance(row["sha256"], str)
            or not _SHA256.fullmatch(row["sha256"])
            or isinstance(row["step"], bool)
            or not isinstance(row["step"], int)
            or row["step"] <= 0
            or row["fraction_numerator"] != row["step"]
            or row["fraction_denominator"]
            != execution_plan["schedule"]["planned_steps"]
            or isinstance(row["bytes"], bool)
            or not isinstance(row["bytes"], int)
            or row["bytes"] <= 0
            or not isinstance(row["aliases"], list)
            or not row["aliases"]
            or not isinstance(row["safetensors"], dict)
        ):
            raise ValueError("run completion candidate identity is invalid")
    if sorted(row[1] for row in seen) != execution_plan["schedule"]["candidate_steps"]:
        raise ValueError("run completion does not contain the sealed candidate grid")


def _v2_plan_approval_expected(
    plan: dict[str, Any], *, candidates: list[dict[str, Any]], evaluator: dict[str, Any]
) -> dict[str, Any]:
    return {
        "plan_payload_sha256": _plan_payload_sha256(plan),
        "fixture_manifest_sha256": plan["fixture_manifest"]["sha256"],
        "fixture_approval_sha256": plan["fixture_approval"]["sha256"],
        "campaign_manifest_sha256": plan["campaign_manifest"]["sha256"],
        "candidate_bindings": [
            {
                "candidate_id": candidate["id"],
                "mode": candidate["candidate_binding"]["mode"],
                "binding_manifest_sha256": candidate["candidate_binding"][
                    "binding_manifest_sha256"
                ],
            }
            for candidate in candidates
        ],
        "evaluator_identity": {
            "god_commit": evaluator["expected_god_commit"],
            "comfy_commit": evaluator["expected_comfy_commit"],
            "tooling_commit": evaluator["expected_tooling_commit"],
            "evaluator_script_sha256": evaluator["expected_evaluator_script_sha256"],
            "dataset_identity_module_sha256": evaluator[
                "expected_dataset_identity_module_sha256"
            ],
            "eval_defaults": evaluator["expected_eval_defaults"],
            "runtime_identity": evaluator["expected_runtime_identity"],
            "assets": evaluator["expected_assets"],
            "cache_provenance_sha256": evaluator["cache_provenance_sha256"],
        },
        "timeout_policy": _timeout_policy(evaluator),
        "batch_runner_sha256": _sha256(Path(__file__).resolve(strict=True)),
    }


def _validate_scorer_support_modules() -> dict[str, str]:
    calibration_root = Path(__file__).resolve(strict=True).parent
    modules = {
        "krea_execution_surface_policy.py": krea_execution_surface_policy,
        "krea_historical_training_evidence.py": krea_historical_training_evidence,
        "krea_scorer_extension_policy.py": krea_scorer_extension_policy,
    }
    observed = {}
    for name, module in modules.items():
        path = Path(module.__file__).resolve(strict=True)
        if path != calibration_root / name:
            raise ValueError(f"scorer support module resolved outside Forge: {name}")
        observed[name] = _sha256(_safe_file(path, f"scorer support module {name}"))
    if observed != _SCORER_SUPPORT_MODULE_SHA256:
        raise ValueError("scorer support module bytes drifted")
    return observed


def _validate_stage1_exact_scorer(evaluator: dict[str, Any]) -> dict[str, Any]:
    """Reject any schema-3 scorer surface not frozen by owner ratification."""

    _validate_scorer_support_modules()
    contract = krea_execution_surface_policy.POLICY["stage1_exact_scorer_contract"]
    if (
        krea_execution_surface_policy.POLICY["policy_sha256"]
        != krea_scorer_extension_policy.POLICY[
            "base_execution_surface_policy_sha256"
        ]
    ):
        raise ValueError("scorer extension is not bound to this training policy")
    extension = krea_scorer_extension_policy.validate(
        evaluator.get("scorer_extension_policy")
    )
    effective_timeouts = krea_scorer_extension_policy.effective_timeouts(
        contract["timeouts_s"], evaluator.get("scorer_timeout_profile")
    )
    timeout_policy = _timeout_policy(evaluator)
    if {
        "startup": timeout_policy["startup_timeout_s"],
        "evaluation": timeout_policy["evaluation_timeout_s"],
        "shutdown": timeout_policy["shutdown_timeout_s"],
        "containment_term_grace": evaluator["containment"]["term_grace_s"],
    } != effective_timeouts:
        raise ValueError("exact scorer effective timeouts differ from its extension")
    marker = extension["changes"]["comfy_lora_placeholder"]
    if (
        marker["relative_path"]
        != f"models/loras/{_COMFY_LORA_PLACEHOLDER}"
        or marker["required_type"] != "regular_file"
        or marker["required_bytes"] != 0
        or marker["required_link_count"] != 1
    ):
        raise ValueError("exact scorer placeholder exception widened")
    observed_assets = {
        name: {
            "basename": Path(row["canonical_path"]).name,
            "relative_path": str(
                Path(row["canonical_path"]).relative_to(Path(evaluator["comfy_root"]))
            ),
            "sha256": row["sha256"],
            "bytes": row["bytes"],
        }
        for name, row in evaluator["expected_assets"].items()
    }
    observed = {
        "god_commit": evaluator["expected_god_commit"],
        "comfy_commit": evaluator["expected_comfy_commit"],
        "tooling_commit": evaluator["expected_tooling_commit"],
        "evaluator_script_sha256": evaluator["expected_evaluator_script_sha256"],
        "dataset_identity_module_sha256": evaluator[
            "expected_dataset_identity_module_sha256"
        ],
        "eval_defaults": evaluator["expected_eval_defaults"],
        "assets": observed_assets,
        "source_trees": contract["source_trees"],
        "runtime_materialization": contract["runtime_materialization"],
        # The base policy remains byte/semantic compatible with the training
        # evidence.  Effective scorer timeouts are validated separately above.
        "timeouts_s": contract["timeouts_s"],
        "estimated_seconds_per_candidate": 720,
        "execution_scope": "offline_stage1_discovery_only",
    }
    if observed != contract:
        raise ValueError("exact scorer differs from owner-ratified Stage-1 contract")
    return observed


def _stage1_exact_scorer_readiness(evaluator: dict[str, Any]) -> dict[str, Any]:
    """Recompute the complete Stage-1 scorer surface before approval/execution."""

    try:
        from . import evaluate_krea_local as local_evaluator
    except ImportError:  # pragma: no cover - direct script execution.
        import evaluate_krea_local as local_evaluator  # type: ignore[no-redef]

    contract = krea_execution_surface_policy.POLICY["stage1_exact_scorer_contract"]
    _validate_stage1_exact_scorer(evaluator)
    comfy_root = Path(evaluator["comfy_root"])
    god_root = Path(evaluator["god_root"])
    tooling_root = comfy_root / "custom_nodes" / "comfyui-tooling-nodes"
    snapshots = {
        "god": local_evaluator._git_snapshot(
            god_root, expected_commit=evaluator["expected_god_commit"]
        ),
        "comfyui": local_evaluator._git_snapshot(
            comfy_root, expected_commit=evaluator["expected_comfy_commit"]
        ),
        "tooling_nodes": local_evaluator._git_snapshot(
            tooling_root, expected_commit=evaluator["expected_tooling_commit"]
        ),
    }
    observed_trees = {name: row["tree"] for name, row in snapshots.items()}
    if observed_trees != contract["source_trees"]:
        raise ValueError("exact scorer source trees differ from owner contract")

    requirements = {
        "comfyui": comfy_root / "requirements.txt",
        "tooling_nodes": tooling_root / "requirements.txt",
        "god_validator": god_root / "ops/docker/requirements/validator.txt",
    }
    requirement_hashes = {
        name: _sha256(_safe_file(path, f"{name} requirements"))
        for name, path in requirements.items()
    }
    runtime_contract = contract["runtime_materialization"]
    lock_contract = runtime_contract["exact_lock"]
    lock_path = Path(__file__).parent / lock_contract["relative_path"]
    lock_path = _safe_file(lock_path, "exact scorer dependency lock")
    if (
        _sha256(lock_path) != lock_contract["sha256"]
        or len(lock_path.read_text(encoding="utf-8").splitlines())
        != lock_contract["line_count"]
    ):
        raise ValueError("exact scorer dependency lock differs from owner contract")
    locked_rows: list[tuple[str, str]] = []
    vcs_versions = lock_contract["vcs_distribution_versions"]
    for number, raw_line in enumerate(
        lock_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            raise ValueError(f"exact scorer lock line {number} is not canonical")
        if " @ " in line:
            name, _url = line.split(" @ ", 1)
            version = vcs_versions.get(re.sub(r"[-_.]+", "-", name).lower())
            if version is None:
                raise ValueError(
                    f"exact scorer lock line {number} has an unbound VCS version"
                )
        else:
            name, separator, version = line.partition("==")
            if separator != "==" or not name or not version or "==" in version:
                raise ValueError(f"exact scorer lock line {number} is not exact")
        locked_rows.append((re.sub(r"[-_.]+", "-", name).lower(), version))
    locked_rows.sort()
    if (
        len(locked_rows) != lock_contract["resolved_distribution_count"]
        or krea_provenance.canonical_sha256(locked_rows)
        != lock_contract["normalized_name_version_sha256"]
    ):
        raise ValueError(
            "exact scorer lock does not resolve to its owner-bound distribution set"
        )
    if requirement_hashes != runtime_contract["requirements_sha256"]:
        raise ValueError("exact scorer requirement inputs differ from owner contract")

    expected_asset_paths = {
        name: comfy_root / row["relative_path"]
        for name, row in contract["assets"].items()
    }
    observed_assets = {}
    for name, expected_path in expected_asset_paths.items():
        configured = Path(evaluator["expected_assets"][name]["canonical_path"])
        if configured != expected_path:
            raise ValueError(f"exact scorer {name} is not at its canonical destination")
        path = _safe_file(configured, f"exact scorer {name}")
        observed_assets[name] = {
            "canonical_path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        if observed_assets[name] != evaluator["expected_assets"][name]:
            raise ValueError(f"exact scorer {name} bytes differ from owner contract")

    if (comfy_root / "extra_model_paths.yaml").exists():
        raise ValueError("exact scorer refuses Comfy extra_model_paths.yaml")
    lora_root = comfy_root / "models" / "loras"
    if os.path.lexists(lora_root):
        _empty_real_directory(
            lora_root,
            "exact scorer LoRA directory before scoring",
            allowed_zero_byte_placeholder=_COMFY_LORA_PLACEHOLDER,
        )

    comfy_python = Path(evaluator["comfy_python"])
    driver_python = Path(evaluator["driver_python"])
    if not os.path.samefile(comfy_python, driver_python):
        raise ValueError("Stage-1 scorer must use one Python for Comfy and driver")
    python_environment = local_evaluator._python_environment(comfy_python)
    driver_environment = {
        key: python_environment[key]
        for key in (
            "executable",
            "prefix",
            "base_prefix",
            "python",
            "distribution_count",
            "distributions_sha256",
            "normalized_distributions_sha256",
        )
    }
    if (
        python_environment["python"] != runtime_contract["python_version"]
        or python_environment["distribution_count"]
        != runtime_contract["distribution_count"]
        or python_environment["distributions_sha256"]
        != runtime_contract["distributions_sha256"]
        or python_environment["normalized_distributions_sha256"]
        != lock_contract["normalized_name_version_sha256"]
    ):
        raise ValueError(
            "exact scorer Python distribution set differs from owner contract"
        )
    critical_probe = (
        "import importlib.metadata as m,json;"
        "print(json.dumps({n:m.version(n) for n in "
        "('torch','torchvision','torchaudio')},sort_keys=True))"
    )
    try:
        critical = json.loads(
            subprocess.check_output(
                [str(comfy_python), "-I", "-c", critical_probe],
                env=local_evaluator._inspection_environment(),
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
            )
        )
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise RuntimeError("could not inspect exact scorer critical packages") from exc
    if critical != runtime_contract["critical_distributions"]:
        raise ValueError(
            "exact scorer critical package versions differ from owner contract"
        )
    cuda_probe = """
import base64
import csv
import hashlib
import importlib.metadata
import io
import json

import torch


def active_owner(cu12_name, cu13_name, relative_path):
    distributions = [
        importlib.metadata.distribution(cu12_name),
        importlib.metadata.distribution(cu13_name),
    ]
    path = distributions[0].locate_file(relative_path)
    actual = hashlib.sha256(path.read_bytes()).digest()
    matches = []
    for distribution in distributions:
        records = {
            row[0]: row[1]
            for row in csv.reader(io.StringIO(distribution.read_text("RECORD")))
        }
        encoded = records.get(relative_path, "")
        if not encoded.startswith("sha256="):
            raise RuntimeError(f"missing wheel RECORD hash: {relative_path}")
        digest = encoded.split("=", 1)[1]
        expected = base64.urlsafe_b64decode(digest + "=" * (-len(digest) % 4))
        if actual == expected:
            matches.append(
                f"{distribution.metadata['Name']}=={distribution.version}"
            )
    if len(matches) != 1:
        raise RuntimeError(f"ambiguous CUDA namespace owner: {relative_path}")
    return matches[0]


owners = {
    "cudnn": active_owner(
        "nvidia-cudnn-cu12",
        "nvidia-cudnn-cu13",
        "nvidia/cudnn/lib/libcudnn.so.9",
    ),
    "cusparselt": active_owner(
        "nvidia-cusparselt-cu12",
        "nvidia-cusparselt-cu13",
        "nvidia/cusparselt/lib/libcusparseLt.so.0",
    ),
    "nccl": active_owner(
        "nvidia-nccl-cu12",
        "nvidia-nccl-cu13",
        "nvidia/nccl/lib/libnccl.so.2",
    ),
    "nvshmem": active_owner(
        "nvidia-nvshmem-cu12",
        "nvidia-nvshmem-cu13",
        "nvidia/nvshmem/lib/libnvshmem_host.so.3",
    ),
}
x = torch.randn(1, 4, 4, 32, 32, device="cuda", dtype=torch.bfloat16)
layer = torch.nn.Conv3d(4, 8, 3, padding=1, device="cuda", dtype=torch.bfloat16)
result = layer(x)
torch.cuda.synchronize()
print(
    json.dumps(
        {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "bf16_conv3d": tuple(result.shape) == (1, 8, 4, 32, 32),
            "overlapping_namespace_owners": owners,
        },
        sort_keys=True,
    )
)
""".strip()
    try:
        cuda_runtime = json.loads(
            subprocess.check_output(
                [str(comfy_python), "-I", "-c", cuda_probe],
                env=local_evaluator._inspection_environment(),
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60,
            )
        )
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RuntimeError("exact scorer CUDA/cuDNN probe failed") from exc
    if cuda_runtime != runtime_contract["cuda_runtime_probe"]:
        raise ValueError("exact scorer CUDA/cuDNN runtime differs from owner contract")
    observed_runtime_identity = {
        "comfy_python_identity_sha256": krea_provenance.canonical_sha256(
            python_environment
        ),
        "driver_python_identity_sha256": krea_provenance.canonical_sha256(
            driver_environment
        ),
    }
    if observed_runtime_identity != evaluator["expected_runtime_identity"]:
        raise ValueError("exact scorer runtime identity differs from live environment")
    body = {
        "schema": 1,
        "kind": "forge-krea-stage1-exact-scorer-readiness",
        "source_snapshots": snapshots,
        "requirements_sha256": requirement_hashes,
        "dependency_lock_sha256": lock_contract["sha256"],
        "assets": observed_assets,
        "runtime_identity": observed_runtime_identity,
        "cuda_runtime_probe": cuda_runtime,
        "runtime_contract_sha256": krea_provenance.canonical_sha256(runtime_contract),
        "base_timeouts_s": contract["timeouts_s"],
        "effective_timeouts_s": krea_scorer_extension_policy.effective_timeouts(
            contract["timeouts_s"], evaluator["scorer_timeout_profile"]
        ),
        "scorer_extension_policy": krea_scorer_extension_policy.POLICY,
        "scorer_timeout_profile": evaluator["scorer_timeout_profile"],
        "scorer_support_module_sha256": _validate_scorer_support_modules(),
        "ready": True,
    }
    return {**body, "readiness_sha256": krea_provenance.canonical_sha256(body)}


def _plan_authority_modules(plan: dict[str, Any]) -> dict[str, Any]:
    """Resolve the authority graph that emitted a schema-2 campaign.

    The scorer may evolve without forcing already-admitted training evidence to
    be rewritten.  When the campaign explicitly binds the one admitted c9f30b1
    validator, every authority-sensitive read must use that isolated graph.
    """

    campaign_binding = plan.get("campaign_manifest")
    historical = None
    # Approval builders are also used to construct a document before the full
    # plan has been materialized in tests/tools.  Only an exact bound campaign
    # may select a historical graph; anything else remains on current authority
    # and is still rejected later by full plan validation.
    if isinstance(campaign_binding, dict) and set(campaign_binding) == {
        "path",
        "sha256",
    }:
        _, _, campaign, campaign_raw = _bound_file(
            campaign_binding, "exact-score campaign manifest"
        )
        _canonical_control_file(
            campaign, campaign_raw, "exact-score campaign manifest"
        )
        _validate_campaign_manifest(campaign)
        historical = campaign.get("historical_training_evidence_validator")
    if historical is not None:
        _validate_scorer_support_modules()
        return krea_historical_training_evidence.load_modules(historical)
    try:
        from . import krea_fixture_admission
    except ImportError:  # pragma: no cover - direct script execution.
        import krea_fixture_admission  # type: ignore[no-redef]

    return {
        "discovery_authorization": krea_discovery_authorization,
        "delegated_review_contract": krea_delegated_review_contract,
        "execution_plan": krea_execution_plan,
        "fixture": krea_fixture,
        "fixture_admission": krea_fixture_admission,
    }


def build_agent_sealed_plan_approval(
    plan: dict[str, Any],
    *,
    technical_reviewer_actor: dict[str, Any],
    discovery_execution_authorization: dict[str, Any],
) -> dict[str, Any]:
    """Build the fixed Stage-1 agent approval; full plan validation follows."""

    if (
        plan.get("schema") != 2
        or plan.get("decision_context", {}).get("phase") != "discovery"
    ):
        raise ValueError("agent exact-score approval is discovery-only")
    evaluator = _validate_evaluator(plan["evaluator"])
    _validate_stage1_exact_scorer(evaluator)
    readiness = _stage1_exact_scorer_readiness(evaluator)
    authority = _plan_authority_modules(plan)
    discovery_authorization = authority["discovery_authorization"]
    delegated_review_contract = authority["delegated_review_contract"]
    _, authorization, _ = discovery_authorization.load_binding(
        discovery_execution_authorization
    )
    if "offline_exact_scoring" not in authorization["authorized_actions"]:
        raise ValueError("discovery authorization does not permit exact scoring")
    actor = delegated_review_contract.validate_actor(
        "exact_score_plan_reviewer", technical_reviewer_actor
    )
    candidates = []
    for raw in plan.get("candidates", []):
        row = _object(raw, "plan candidate")
        _, binding, binding_sha = _load_candidate_binding(
            row.get("candidate_binding"), "candidate binding"
        )
        candidates.append(
            {
                "id": row.get("id"),
                "candidate_binding": {
                    "mode": binding.get("mode"),
                    "binding_manifest_sha256": binding_sha,
                },
            }
        )
    candidates.sort(key=lambda item: item["id"])
    expected = _v2_plan_approval_expected(
        plan, candidates=candidates, evaluator=evaluator
    )
    return {
        "schema": 3,
        "kind": "forge-krea-agent-exact-score-plan-approval",
        "decision": "approved",
        "technical_reviewer_actor": actor,
        "accountable_owner_identity": authorization["accountable_owner_identity"],
        "owner_ratification_sha256": authorization["fixture_admission_envelope"][
            "owner_ratification_sha256"
        ],
        "discovery_execution_authorization": dict(discovery_execution_authorization),
        "delegated_review_contract": delegated_review_contract.binding(),
        "agent_review_is_not_human_review": True,
        "scorer_readiness": readiness,
        **expected,
    }


def _validate_v2_approval(
    value: dict[str, Any],
    raw: bytes,
    *,
    plan: dict[str, Any],
    candidates: list[dict[str, Any]],
    evaluator: dict[str, Any],
    common_authorization_sha256: str | None,
) -> dict[str, Any]:
    _canonical_control_file(value, raw, "sealed exact-score approval")
    expected_fields = _v2_plan_approval_expected(
        plan, candidates=candidates, evaluator=evaluator
    )
    if value.get("schema") == 3:
        _exact_keys(
            value,
            {
                "schema",
                "kind",
                "decision",
                "technical_reviewer_actor",
                "accountable_owner_identity",
                "owner_ratification_sha256",
                "discovery_execution_authorization",
                "delegated_review_contract",
                "agent_review_is_not_human_review",
                "scorer_readiness",
                *expected_fields,
            },
            "agent exact-score approval",
        )
        _validate_stage1_exact_scorer(evaluator)
        readiness = _stage1_exact_scorer_readiness(evaluator)
        authority = _plan_authority_modules(plan)
        discovery_authorization = authority["discovery_authorization"]
        delegated_review_contract = authority["delegated_review_contract"]
        _, authorization, _ = discovery_authorization.load_binding(
            value["discovery_execution_authorization"]
        )
        actor = delegated_review_contract.validate_actor(
            "exact_score_plan_reviewer", value["technical_reviewer_actor"]
        )
        expected = {
            "schema": 3,
            "kind": "forge-krea-agent-exact-score-plan-approval",
            "decision": "approved",
            "technical_reviewer_actor": actor,
            "accountable_owner_identity": authorization["accountable_owner_identity"],
            "owner_ratification_sha256": authorization["fixture_admission_envelope"][
                "owner_ratification_sha256"
            ],
            "discovery_execution_authorization": dict(
                value["discovery_execution_authorization"]
            ),
            "delegated_review_contract": delegated_review_contract.binding(),
            "agent_review_is_not_human_review": True,
            "scorer_readiness": readiness,
            **expected_fields,
        }
        if (
            value != expected
            or plan["decision_context"].get("phase") != "discovery"
            or common_authorization_sha256 != authorization["authorization_sha256"]
            or "offline_exact_scoring" not in authorization["authorized_actions"]
        ):
            raise ValueError(
                "agent exact-score approval escaped owner-ratified discovery"
            )
        return {
            "technical_reviewer_actor": actor,
            "accountable_owner_identity": authorization["accountable_owner_identity"],
            "decision": "approved",
            "agent_review_is_not_human_review": True,
        }
    if common_authorization_sha256 is not None:
        raise ValueError(
            "authorization-bound Stage-1 scoring requires the delegated schema-3 approval"
        )
    _exact_keys(
        value,
        {
            "schema",
            "kind",
            "decision",
            "reviewer_identity",
            *expected_fields,
        },
        "sealed exact-score approval",
    )
    reviewer = _named_human(value["reviewer_identity"], "exact-score reviewer")
    expected = {
        "schema": 2,
        "kind": "forge-krea-exact-score-plan-approval",
        "decision": "approved",
        "reviewer_identity": reviewer,
        **expected_fields,
    }
    if value != expected:
        raise ValueError("exact-score approval does not bind the complete batch plan")
    return {"reviewer_identity": reviewer, "decision": "approved"}


def _validate_plan_v2(
    plan: dict[str, Any]
) -> tuple[Path, str, list[dict[str, Any]], dict[str, Any]]:
    _exact_keys(
        plan,
        {
            "schema",
            "kind",
            "dataset",
            "fixture_manifest",
            "fixture_approval",
            "cross_fixture_review",
            "campaign_manifest",
            "decision_context",
            "candidates",
            "evaluator",
            "sealed_plan_approval",
        },
        "plan",
    )
    if plan["schema"] != 2 or plan["kind"] != "forge-krea-exact-score-plan":
        raise ValueError("unsupported exact-score plan schema or kind")
    dataset_spec = _object(plan["dataset"], "plan.dataset")
    _exact_keys(dataset_spec, {"path", "sha256"}, "plan.dataset")
    dataset = _safe_directory(dataset_spec["path"], "exact-evaluation dataset")
    campaign_path, campaign_file_sha, campaign, campaign_raw = _bound_file(
        plan["campaign_manifest"], "exact-score campaign manifest"
    )
    _canonical_control_file(campaign, campaign_raw, "exact-score campaign manifest")
    _validate_campaign_manifest(campaign)
    if campaign_file_sha != plan["campaign_manifest"]["sha256"]:
        raise ValueError("campaign manifest file binding mismatch")
    authority = _plan_authority_modules(plan)
    fixture_validator = authority["fixture"]
    fixture_admission_validator = authority["fixture_admission"]
    execution_plan_validator = authority["execution_plan"]
    training_evidence_validator = authority.get("training_evidence")
    if training_evidence_validator is None:
        try:
            from . import krea_training_evidence as training_evidence_validator
        except ImportError:  # pragma: no cover
            import krea_training_evidence as training_evidence_validator  # type: ignore[no-redef]
    historical_identity = campaign.get("historical_training_evidence_validator")
    expected_training_policy_sha256 = (
        krea_execution_surface_policy.POLICY["policy_sha256"]
        if historical_identity is None
        else historical_identity["execution_surface_policy_sha256"]
    )
    fixture_path, fixture_file_sha, fixture, fixture_raw = _bound_file(
        plan["fixture_manifest"], "fixture manifest"
    )
    _canonical_control_file(fixture, fixture_raw, "fixture manifest")
    fixture_validator.validate_manifest(fixture)
    if fixture_file_sha != plan["fixture_manifest"]["sha256"]:
        raise ValueError("fixture manifest file binding mismatch")
    (
        fixture_approval_path,
        fixture_approval_file_sha,
        fixture_approval,
        fixture_approval_raw,
    ) = _bound_file(plan["fixture_approval"], "fixture approval")
    _canonical_control_file(fixture_approval, fixture_approval_raw, "fixture approval")
    fixture_validator.validate_approval(fixture_approval, fixture_manifest=fixture)
    cross_review_path, cross_review_file_sha, cross_review, cross_review_raw = (
        _bound_file(plan["cross_fixture_review"], "cross-fixture admission surface")
    )
    _canonical_control_file(
        cross_review, cross_review_raw, "cross-fixture admission surface"
    )
    _validate_cross_fixture_review_surface(
        cross_review,
        fixture=fixture,
        source_path=cross_review_path,
        fixture_admission_validator=fixture_admission_validator,
    )
    decision_context = _validate_score_decision_context(
        plan["decision_context"], campaign=campaign
    )
    expected_identity = fixture["evaluation_dataset_identity"]
    observed_identity = krea_dataset_identity.capture_dataset(
        dataset,
        list_supported_images=lambda _root, _extensions: list(
            expected_identity["evaluator_order"]
        ),
        extensions=tuple(fixture["tool_identity"]["extensions"]),
    )
    dataset_sha = dataset_spec["sha256"]
    if (
        observed_identity != expected_identity
        or dataset_sha != expected_identity["sha256"]
    ):
        raise ValueError("batch dataset differs from the approved exact-eval fixture")

    evaluator = _validate_evaluator(plan["evaluator"])
    _validate_scorer_fixture_timeout(evaluator, fixture)
    raw_candidates = plan["candidates"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("plan.candidates must be non-empty")
    candidates: list[dict[str, Any]] = []
    ids: set[str] = set()
    hashes: set[str] = set()
    common_envelope: dict[str, Any] | None = None
    run_groups: dict[str, dict[str, Any]] = {}
    discovery_plan_shas: set[str] = set()
    zero_manifest_shas: list[str] = []
    local_base_model: dict[str, Any] | None = None
    zero_base_model: dict[str, Any] | None = None
    for index, raw in enumerate(raw_candidates):
        row = _object(raw, f"plan.candidates[{index}]")
        _exact_keys(
            row,
            {"id", "arm_id", "path", "sha256", "candidate_binding"},
            f"plan.candidates[{index}]",
        )
        candidate_id = row["id"]
        arm_id = row["arm_id"]
        if (
            not isinstance(candidate_id, str)
            or not _SAFE_ID.fullmatch(candidate_id)
            or not isinstance(arm_id, str)
            or not _SAFE_ID.fullmatch(arm_id)
        ):
            raise ValueError("candidate/arm id is invalid")
        candidate_path = _safe_file(row["path"], f"candidate {candidate_id}")
        candidate_sha = row["sha256"]
        if not isinstance(candidate_sha, str) or not _SHA256.fullmatch(candidate_sha):
            raise ValueError(f"candidate {candidate_id} SHA-256 is invalid")
        if _sha256(candidate_path) != candidate_sha:
            raise ValueError(f"candidate {candidate_id} SHA-256 mismatch")
        if candidate_id in ids or candidate_sha in hashes:
            raise ValueError("duplicate candidate id or bytes")
        binding_path, binding, binding_sha = _load_candidate_binding(
            row["candidate_binding"], f"candidate {candidate_id} binding"
        )
        mode = binding.get("mode")
        private: dict[str, Any] = {
            "_binding_path": str(binding_path),
            "_binding_sha256": binding_sha,
        }
        if mode == "local_run_candidate":
            _exact_keys(
                binding,
                {
                    "schema",
                    "kind",
                    "mode",
                    "arm_id",
                    "candidate_id",
                    "candidate",
                    "execution_plan",
                    "execution_approval",
                    "run_completion",
                    "run_record",
                    "training_log",
                    "evaluation_dataset_sha256",
                },
                f"candidate {candidate_id} binding",
            )
            if (
                binding["schema"] != 2
                or binding["kind"] != "forge-krea-local-candidate-binding"
                or binding["arm_id"] != arm_id
                or binding["candidate_id"] != candidate_id
                or binding["evaluation_dataset_sha256"] != dataset_sha
            ):
                raise ValueError("local candidate binding identity mismatch")
            candidate_identity = _object(binding["candidate"], "bound candidate")
            _exact_keys(
                candidate_identity,
                {
                    "path",
                    "sha256",
                    "bytes",
                    "step",
                    "fraction_numerator",
                    "fraction_denominator",
                    "aliases",
                    "safetensors",
                },
                "bound candidate",
            )
            if (
                candidate_identity.get("sha256") != candidate_sha
                or Path(candidate_identity.get("path", "")) != candidate_path
                or candidate_identity.get("bytes") != candidate_path.stat().st_size
            ):
                raise ValueError("local candidate bytes/path differ from binding")
            _validate_safetensors_identity(
                candidate_identity["safetensors"], "bound candidate safetensors"
            )
            if (
                candidate_identity["safetensors"]["bytes"]
                != candidate_identity["bytes"]
            ):
                raise ValueError("bound candidate safetensors size is inconsistent")
            execution_plan_path, execution_plan_value, execution_plan_file_sha = (
                _load_candidate_binding(binding["execution_plan"], "execution plan")
            )
            execution_resolved = execution_plan_validator.validate_plan(
                execution_plan_value
            )
            if (
                execution_plan_value["arm_id"] != arm_id
                or execution_resolved["fixture"]["manifest_sha256"]
                != fixture["manifest_sha256"]
            ):
                raise ValueError("candidate execution plan differs from batch fixture")
            discovery_plan_shas.add(execution_plan_value["discovery_plan"]["sha256"])
            if local_base_model is None:
                local_base_model = execution_plan_value["base_model"]
            elif local_base_model != execution_plan_value["base_model"]:
                raise ValueError("local exact-score runs use different base identities")
            approval_path, approval_value, approval_file_sha = _load_candidate_binding(
                binding["execution_approval"], "execution approval"
            )
            execution_plan_validator.validate_approval(
                approval_value,
                plan=execution_plan_value,
                approval_path=approval_path,
            )
            completion_path, completion, completion_sha = _load_candidate_binding(
                binding["run_completion"], "run completion"
            )
            _validate_run_completion(
                completion,
                execution_plan=execution_plan_value,
                execution_plan_file_sha=execution_plan_file_sha,
                execution_approval=approval_value,
                execution_approval_file_sha=approval_file_sha,
                fixture=fixture,
                execution_plan_validator=execution_plan_validator,
                expected_execution_surface_policy_sha256=(
                    expected_training_policy_sha256
                ),
            )
            matching = [
                item
                for item in completion["candidates"]
                if item["candidate_id"] == candidate_id
                and item["sha256"] == candidate_sha
                and item["step"] == candidate_identity.get("step")
            ]
            if len(matching) != 1:
                raise ValueError("candidate is absent/ambiguous in run completion")
            completion_candidate = matching[0]
            if (
                completion_candidate["bytes"] != candidate_identity["bytes"]
                or completion_candidate["safetensors"]
                != candidate_identity["safetensors"]
                or completion_candidate["fraction_numerator"]
                != candidate_identity["fraction_numerator"]
                or completion_candidate["fraction_denominator"]
                != candidate_identity["fraction_denominator"]
            ):
                raise ValueError("candidate binding differs from run completion")
            run_path, run_sha = _bound_bytes_file(binding["run_record"], "run record")
            log_path, log_sha = _bound_bytes_file(
                binding["training_log"], "training log"
            )
            if (
                run_sha != completion["run_record_sha256"]
                or log_sha != completion["training_log_sha256"]
            ):
                raise ValueError("run completion does not bind run/log bytes")
            recipe_fields = execution_resolved["execution_recipe"]["fields"]
            axes = execution_plan_value["predeclared_recipe_axes"]
            measured_envelope = execution_resolved["throughput_profile"][
                "execution_envelope"
            ]
            envelope = {
                "schema": 2,
                "kind": "forge-krea-common-local-execution-envelope",
                "discovery_execution_authorization_sha256": execution_resolved[
                    "discovery_execution_authorization"
                ]["authorization_sha256"],
                "fixture_manifest_sha256": fixture["manifest_sha256"],
                "training_dataset_sha256": fixture["training_dataset_identity"][
                    "sha256"
                ],
                "evaluation_dataset_sha256": dataset_sha,
                "base_model": execution_plan_value["base_model"],
                "seed": execution_plan_value["seed"],
                "runtime_identity_sha256": execution_plan_value[
                    "runtime_identity_sha256"
                ],
                "host_execution_identity_sha256": measured_envelope[
                    "host_execution_identity_sha256"
                ],
                "execution_surface": measured_envelope["execution_surface"],
                "execution_scope": measured_envelope["execution_scope"],
                "venv_tree_manifest_sha256": measured_envelope[
                    "venv_tree_manifest_sha256"
                ],
                "reference_container_image_sha256": measured_envelope[
                    "reference_container_image_sha256"
                ],
                "gpu_identity_sha256": measured_envelope["gpu_identity_sha256"],
                "trainer_identity_sha256": measured_envelope["trainer_identity_sha256"],
                "measurement_tool_sha256": measured_envelope["measurement_tool_sha256"],
                "training_pair_count": measured_envelope["training_pair_count"],
                "training_dataset_shape_sha256": measured_envelope[
                    "training_dataset_shape_sha256"
                ],
                "resolution_policy_sha256": measured_envelope[
                    "resolution_policy_sha256"
                ],
                "precision_policy_sha256": measured_envelope["precision_policy_sha256"],
                "cache_latents_to_disk": measured_envelope["cache_latents_to_disk"],
                "cache_text_embeddings": measured_envelope["cache_text_embeddings"],
                "compile_enabled": measured_envelope["compile_enabled"],
                "jit_enabled": measured_envelope["jit_enabled"],
                "dataloader_workers": measured_envelope["dataloader_workers"],
                "hard_budget_s": execution_plan_value["budget_plan"]["hard_budget_s"],
                "predeclared_recipe_axes": axes,
                "fixed_execution_fields": {
                    name: recipe_fields[name]["effective_value"]
                    for name in sorted(recipe_fields)
                    if name not in axes and name not in {"submitted_step", "selector"}
                },
                "in_task_proxy_selection": {"enabled": False, "reserve_s": 0},
            }
            if common_envelope is None:
                common_envelope = envelope
            elif envelope != common_envelope:
                raise ValueError(
                    f"candidate {candidate_id} escaped the common local envelope"
                )
            normalized_binding = {
                "mode": mode,
                "binding_manifest_sha256": binding_sha,
                "execution_plan_sha256": execution_plan_value["plan_sha256"],
                "run_completion_sha256": completion_sha,
                "candidate_sha256": candidate_sha,
                "candidate_bytes": candidate_identity["bytes"],
                "candidate_step": candidate_identity["step"],
                "candidate_fraction": {
                    "numerator": candidate_identity["fraction_numerator"],
                    "denominator": candidate_identity["fraction_denominator"],
                },
                "candidate_image_exposures": (
                    candidate_identity["step"]
                    * execution_resolved["throughput_profile"]["execution_envelope"][
                        "micro_batch_size"
                    ]
                    * execution_resolved["throughput_profile"]["execution_envelope"][
                        "gradient_accumulation_steps"
                    ]
                    * execution_resolved["throughput_profile"]["execution_envelope"][
                        "data_parallel_replicas"
                    ]
                ),
                "normalized_recipe": execution_resolved["execution_recipe"],
            }
            group = run_groups.setdefault(
                completion_sha,
                {
                    "arm_id": arm_id,
                    "execution_plan_sha256": execution_plan_value["plan_sha256"],
                    "run_completion_sha256": completion_sha,
                    "completion_candidates": completion["candidates"],
                    "listed_candidates": [],
                    "run_envelope": {
                        "execution_envelope": measured_envelope,
                        "throughput_profile_sha256": execution_resolved[
                            "throughput_profile"
                        ]["profile_sha256"],
                        "budget_plan": execution_plan_value["budget_plan"],
                        "budget_plan_sha256": execution_plan_value[
                            "budget_plan_sha256"
                        ],
                        "schedule": execution_plan_value["schedule"],
                    },
                },
            )
            if (
                group["arm_id"] != arm_id
                or group["execution_plan_sha256"] != execution_plan_value["plan_sha256"]
                or group["completion_candidates"] != completion["candidates"]
            ):
                raise ValueError("one run completion is bound inconsistently")
            group["listed_candidates"].append(
                {
                    "candidate_id": candidate_id,
                    "sha256": candidate_sha,
                    "bytes": candidate_identity["bytes"],
                    "step": candidate_identity["step"],
                    "fraction": {
                        "numerator": candidate_identity["fraction_numerator"],
                        "denominator": candidate_identity["fraction_denominator"],
                    },
                }
            )
            private.update(
                {
                    "_execution_plan_path": str(execution_plan_path),
                    "_execution_plan_sha256": execution_plan_file_sha,
                    "_execution_approval_path": str(approval_path),
                    "_execution_approval_sha256": approval_file_sha,
                    "_run_completion_path": str(completion_path),
                    "_run_completion_sha256": completion_sha,
                    "_run_record_path": str(run_path),
                    "_run_record_sha256": run_sha,
                    "_training_log_path": str(log_path),
                    "_training_log_sha256": log_sha,
                }
            )
        elif mode == "zero_lora_control":
            training_evidence_validator.validate_zero_control(
                binding, artifact_path=candidate_path
            )
            if binding["evaluation_dataset_sha256"] != dataset_sha or binding[
                "base_model"
            ] != {
                "model_id": "krea/Krea-2-Raw",
                "revision": binding["base_model"].get("revision"),
                "training_identity_sha256": binding["base_model"].get(
                    "training_identity_sha256"
                ),
                "evaluation_assets": evaluator["expected_assets"],
            }:
                raise ValueError("zero control is not bound to this base/eval fixture")
            zero_manifest_shas.append(binding["manifest_sha256"])
            zero_base_model = binding["base_model"]
            normalized_binding = {
                "mode": mode,
                "binding_manifest_sha256": binding_sha,
                "zero_control_manifest_sha256": binding["manifest_sha256"],
                "candidate_sha256": candidate_sha,
                "candidate_bytes": candidate_path.stat().st_size,
                "candidate_step": None,
                "candidate_fraction": None,
                "candidate_image_exposures": 0,
            }
        else:
            raise ValueError(f"candidate {candidate_id} binding mode is unsupported")
        ids.add(candidate_id)
        hashes.add(candidate_sha)
        candidates.append(
            {
                "id": candidate_id,
                "source_arm_id": arm_id,
                "path": candidate_path,
                "sha256": candidate_sha,
                "provenance": {"manifest_sha256": None},
                "provenance_path": binding_path,
                "provenance_file_sha256": binding_sha,
                "candidate_binding": normalized_binding,
                **private,
            }
        )
    candidates.sort(key=lambda item: item["id"])
    if len(zero_manifest_shas) != 1:
        raise ValueError("exact-score plan requires exactly one zero-LoRA control")
    if len(discovery_plan_shas) != 1:
        raise ValueError("local runs do not share one frozen discovery plan")
    if local_base_model is None or zero_base_model != local_base_model:
        raise ValueError("zero control does not share the local-run base identity")
    actual_runs = []
    for group in run_groups.values():
        expected_candidates = [
            {
                "candidate_id": row["candidate_id"],
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "step": row["step"],
                "fraction": {
                    "numerator": row["fraction_numerator"],
                    "denominator": row["fraction_denominator"],
                },
            }
            for row in group["completion_candidates"]
        ]
        expected_candidates.sort(key=lambda row: (row["step"], row["sha256"]))
        listed = sorted(
            group["listed_candidates"], key=lambda row: (row["step"], row["sha256"])
        )
        if listed != expected_candidates:
            raise ValueError(
                f"exact-score plan cherry-picks candidates from arm {group['arm_id']}"
            )
        actual_runs.append(
            {
                "arm_id": group["arm_id"],
                "execution_plan_sha256": group["execution_plan_sha256"],
                "run_completion_sha256": group["run_completion_sha256"],
                "candidates": expected_candidates,
            }
        )
    actual_runs.sort(key=lambda row: row["arm_id"])
    if (
        campaign["fixture_manifest_sha256"] != fixture["manifest_sha256"]
        or campaign["discovery_plan_sha256"] != next(iter(discovery_plan_shas))
        or campaign["zero_control_manifest_sha256"] != zero_manifest_shas[0]
        or campaign["runs"] != actual_runs
    ):
        raise ValueError("exact-score plan does not exhaust the sealed campaign")
    approval_path, approval_sha, approval, approval_raw = _bound_file(
        plan["sealed_plan_approval"], "sealed exact-score approval"
    )
    approval_summary = _validate_v2_approval(
        approval,
        approval_raw,
        plan=plan,
        candidates=candidates,
        evaluator=evaluator,
        common_authorization_sha256=(
            common_envelope.get("discovery_execution_authorization_sha256")
            if common_envelope is not None
            else None
        ),
    )
    evaluator["_expected_dataset_identity"] = expected_identity
    evaluator["_common_training_envelope"] = common_envelope
    evaluator["_common_training_envelope_sha256"] = (
        krea_provenance.canonical_sha256(common_envelope)
        if common_envelope is not None
        else None
    )
    evaluator["_sealed_plan_approval_path"] = str(approval_path)
    evaluator["_sealed_plan_approval_sha256"] = approval_sha
    evaluator["_sealed_plan_approval"] = approval_summary
    evaluator["_plan_payload_sha256"] = _plan_payload_sha256(plan)
    evaluator["_batch_runner_sha256"] = _sha256(Path(__file__).resolve(strict=True))
    evaluator["_fixture_manifest_path"] = str(fixture_path)
    evaluator["_fixture_manifest_file_sha256"] = fixture_file_sha
    evaluator["_fixture_approval_path"] = str(fixture_approval_path)
    evaluator["_fixture_approval_file_sha256"] = fixture_approval_file_sha
    evaluator["_cross_fixture_review_path"] = str(cross_review_path)
    evaluator["_cross_fixture_review_file_sha256"] = cross_review_file_sha
    evaluator["_cross_fixture_review"] = cross_review
    evaluator["_campaign_manifest_path"] = str(campaign_path)
    evaluator["_campaign_manifest_file_sha256"] = campaign_file_sha
    evaluator["_campaign_manifest_sha256"] = campaign["manifest_sha256"]
    evaluator["_fixture_validator"] = fixture_validator
    evaluator["_decision_context"] = decision_context
    evaluator["_training_run_envelopes"] = [
        {
            "arm_id": group["arm_id"],
            "execution_plan_sha256": group["execution_plan_sha256"],
            **group["run_envelope"],
        }
        for group in sorted(run_groups.values(), key=lambda row: row["arm_id"])
    ]
    return dataset, dataset_sha, candidates, evaluator


def _validate_plan(
    plan: dict[str, Any]
) -> tuple[Path, str, list[dict[str, Any]], dict[str, Any]]:
    if plan.get("schema") == 2:
        return _validate_plan_v2(plan)
    return _validate_plan_v1(plan)


def _result_envelope(result: dict[str, Any]) -> dict[str, Any]:
    runtime = _object(result["runtime"], "result.runtime")
    return {
        "evaluator": result["evaluator"],
        "model_type": result["model_type"],
        "dataset": result["dataset"],
        "dataset_sha256": result["dataset_sha256"],
        "image_count": result["image_count"],
        "base_name": result["base_name"],
        "asset_sha256": result["asset_sha256"],
        "asset_bytes": result["asset_bytes"],
        "steps": result["steps"],
        "cfg": result["cfg"],
        "denoise": result["denoise"],
        "generations": result["generations"],
        "master_seed": result["master_seed"],
        "seeds": result["seeds"],
        "text_weight": result["text_weight"],
        "direction": result["direction"],
        "source": result["source"],
        "runtime_source": {
            key: runtime[key]
            for key in (
                "fresh_comfy_process",
                "loopback",
                "cache",
                "database",
                "api_nodes_disabled",
                "isolated_input_output_temp_user",
                "offline_environment",
                "custom_node_allowlist",
                "python",
                "driver_python",
            )
        },
    }


def _validate_result(
    result: dict[str, Any],
    *,
    candidate: dict[str, Any],
    dataset: Path,
    expected_dataset_sha256: str,
    expected_dataset_identity: dict[str, Any] | None,
    evaluator: dict[str, Any],
    evaluator_script_sha: str,
    log_path: Path,
    comfy_staging: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required = {
        "schema",
        "evaluator",
        "candidate",
        "candidate_sha256",
        "candidate_bytes",
        "staged_candidate_sha256",
        "comfy_lora_name",
        "model_type",
        "dataset",
        "dataset_sha256",
        "image_count",
        "scored_rows",
        "base_name",
        "asset_sha256",
        "asset_bytes",
        "steps",
        "cfg",
        "denoise",
        "generations",
        "master_seed",
        "seeds",
        "text_guided_losses",
        "blank_prompt_losses",
        "text_mean",
        "blank_mean",
        "text_weight",
        "weighted_loss",
        "direction",
        "elapsed_s",
        "source",
        "runtime",
    }
    _exact_keys(result, required, f"candidate {candidate['id']} result")
    if result["schema"] != 2 or result["evaluator"] != "god_krea2_img2img_exact":
        raise ValueError(
            f"candidate {candidate['id']} returned an unsupported evaluator result"
        )
    if result["candidate"] != candidate["path"].name:
        raise ValueError(f"candidate {candidate['id']} result filename mismatch")
    if (
        result["candidate_sha256"] != candidate["sha256"]
        or result["staged_candidate_sha256"] != candidate["sha256"]
        or result["candidate_bytes"] != candidate["path"].stat().st_size
    ):
        raise ValueError(f"candidate {candidate['id']} result SHA-256 mismatch")
    expected_lora_name = f"candidate-{candidate['sha256']}.safetensors"
    if result["comfy_lora_name"] != expected_lora_name:
        raise ValueError(f"candidate {candidate['id']} Comfy LoRA name mismatch")
    if comfy_staging is not None and comfy_staging != {
        "comfy_lora_name": expected_lora_name,
        "comfy_lora_path": str(
            _absolute_lexical(
                Path(evaluator["comfy_root"]) / "models" / "loras" / expected_lora_name
            )
        ),
        "sha256": candidate["sha256"],
        "bytes": candidate["path"].stat().st_size,
    }:
        raise ValueError(f"candidate {candidate['id']} Comfy staging mismatch")
    if Path(result["dataset"]).resolve() != dataset.resolve():
        raise ValueError(f"candidate {candidate['id']} result dataset path mismatch")
    if result["model_type"] != "krea2" or result["direction"] != "min":
        raise ValueError(f"candidate {candidate['id']} result semantics mismatch")
    elapsed = result["elapsed_s"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) <= 0
    ):
        raise ValueError(f"candidate {candidate['id']} elapsed_s is invalid")
    defaults = evaluator["expected_eval_defaults"]
    if (
        result["steps"] != defaults["steps"]
        or result["cfg"] != defaults["cfg"]
        or result["denoise"] != defaults["denoise"]
        or result["generations"] != defaults["generations"]
        or result["master_seed"] != defaults["master_seed"]
        or result["text_weight"] != defaults["text_weight"]
    ):
        raise ValueError(f"candidate {candidate['id']} evaluator defaults mismatch")
    expected_assets = evaluator["expected_assets"]
    expected_asset_sha = {
        name: identity["sha256"] for name, identity in expected_assets.items()
    }
    expected_asset_bytes = {
        name: identity["bytes"] for name, identity in expected_assets.items()
    }
    if (
        result["asset_sha256"] != expected_asset_sha
        or result["asset_bytes"] != expected_asset_bytes
    ):
        raise ValueError(
            f"candidate {candidate['id']} result base assets differ from the sealed plan"
        )
    expected_base_name = Path(expected_assets["diffusion_model"]["canonical_path"]).name
    if result["base_name"] != expected_base_name:
        raise ValueError(f"candidate {candidate['id']} result base filename mismatch")
    if not isinstance(result["dataset_sha256"], str) or not _SHA256.fullmatch(
        result["dataset_sha256"]
    ):
        raise ValueError(f"candidate {candidate['id']} result dataset hash is invalid")
    if result["dataset_sha256"] != expected_dataset_sha256:
        raise ValueError(
            f"candidate {candidate['id']} result dataset identity mismatch"
        )
    count = result["image_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError(f"candidate {candidate['id']} result image count is invalid")
    for key in ("scored_rows", "text_guided_losses", "blank_prompt_losses"):
        if not isinstance(result[key], list) or len(result[key]) != count:
            raise ValueError(f"candidate {candidate['id']} result {key} is incomplete")
    identity_rows = []
    for row_index, row in enumerate(result["scored_rows"]):
        if not isinstance(row, dict):
            raise ValueError(f"candidate {candidate['id']} scored row is not an object")
        expected_loss_keys = {"text_guided_loss", "blank_prompt_loss"}
        if not expected_loss_keys.issubset(row):
            raise ValueError(f"candidate {candidate['id']} scored row lacks losses")
        if (
            row["text_guided_loss"] != result["text_guided_losses"][row_index]
            or row["blank_prompt_loss"] != result["blank_prompt_losses"][row_index]
        ):
            raise ValueError(
                f"candidate {candidate['id']} scored-row losses differ from arrays"
            )
        identity_rows.append(
            {key: value for key, value in row.items() if key not in expected_loss_keys}
        )
    if expected_dataset_identity is not None and (
        count != len(expected_dataset_identity["rows"])
        or identity_rows != expected_dataset_identity["rows"]
    ):
        raise ValueError(
            f"candidate {candidate['id']} scored rows differ from the sealed fixture"
        )
    for key in ("text_guided_losses", "blank_prompt_losses"):
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in result[key]
        ):
            raise ValueError(f"candidate {candidate['id']} result {key} is invalid")
    for key in ("text_mean", "blank_mean", "text_weight", "weighted_loss"):
        value = result[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"candidate {candidate['id']} result {key} is invalid")
    text_mean = sum(result["text_guided_losses"]) / count
    blank_mean = sum(result["blank_prompt_losses"]) / count
    weighted = (
        result["text_weight"] * text_mean + (1.0 - result["text_weight"]) * blank_mean
    )
    if (
        not math.isclose(result["text_mean"], text_mean, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(
            result["blank_mean"], blank_mean, rel_tol=0.0, abs_tol=1e-12
        )
        or not math.isclose(
            result["weighted_loss"], weighted, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError(f"candidate {candidate['id']} aggregate loss is inconsistent")
    source = _object(result["source"], "result.source")
    god = _object(source.get("god"), "result.source.god")
    if god.get("commit") != evaluator["expected_god_commit"]:
        raise ValueError(f"candidate {candidate['id']} result G.O.D commit mismatch")
    for key, source_key in (
        ("expected_comfy_commit", "comfyui"),
        ("expected_tooling_commit", "tooling_nodes"),
    ):
        if key in evaluator:
            source_row = _object(source.get(source_key), f"result.source.{source_key}")
            if source_row.get("commit") != evaluator[key]:
                raise ValueError(
                    f"candidate {candidate['id']} result {source_key} commit mismatch"
                )
    expected_commits = _object(
        source.get("expected_commits"), "result.source.expected_commits"
    )
    if expected_commits != {
        "god": evaluator["expected_god_commit"],
        "comfyui": evaluator["expected_comfy_commit"],
        "tooling_nodes": evaluator["expected_tooling_commit"],
    }:
        raise ValueError(
            f"candidate {candidate['id']} expected-commit envelope mismatch"
        )
    if source.get("calibration_shim_sha256") != evaluator_script_sha:
        raise ValueError(f"candidate {candidate['id']} evaluator script hash mismatch")
    if evaluator_script_sha != evaluator["expected_evaluator_script_sha256"]:
        raise ValueError(
            f"candidate {candidate['id']} evaluator identity was not predeclared"
        )
    if (
        _sha256(
            Path(__file__).with_name("krea_dataset_identity.py").resolve(strict=True)
        )
        != evaluator["expected_dataset_identity_module_sha256"]
    ):
        raise ValueError("dataset identity module changed during exact scoring")
    if not log_path.is_file() or log_path.is_symlink():
        raise ValueError(
            f"candidate {candidate['id']} evaluator log is missing or unsafe"
        )
    runtime = _object(result["runtime"], "result.runtime")
    required_runtime = {
        "fresh_comfy_process": True,
        "loopback": "127.0.0.1",
        # evaluate_krea_local's evidence schema uses ``memory`` for its
        # explicit sqlite:///:memory: launch argument.
        "database": "memory",
        "api_nodes_disabled": True,
        "isolated_input_output_temp_user": True,
        "offline_environment": True,
    }
    mismatched_runtime = {
        key: {"expected": expected, "actual": runtime.get(key)}
        for key, expected in required_runtime.items()
        if runtime.get(key) != expected
    }
    if mismatched_runtime:
        raise ValueError(
            f"candidate {candidate['id']} unsafe evaluator runtime: {mismatched_runtime}"
        )
    runtime_identity = evaluator["expected_runtime_identity"]
    if (
        krea_provenance.canonical_sha256(runtime.get("python"))
        != runtime_identity["comfy_python_identity_sha256"]
        or krea_provenance.canonical_sha256(runtime.get("driver_python"))
        != runtime_identity["driver_python_identity_sha256"]
    ):
        raise ValueError(f"candidate {candidate['id']} runtime identity mismatch")
    if runtime.get("comfy_log_sha256") != _sha256(log_path):
        raise ValueError(f"candidate {candidate['id']} evaluator log hash mismatch")
    if runtime.get("comfy_log_bytes") != log_path.stat().st_size:
        raise ValueError(f"candidate {candidate['id']} evaluator log size mismatch")
    # Constructing the envelope also asserts all common source/runtime keys exist.
    return _result_envelope(result)


def _evaluator_command(
    *,
    evaluator_script: Path,
    dataset: Path,
    candidate: dict[str, Any],
    result_path: Path,
    evaluator: dict[str, Any],
) -> list[str]:
    command = [
        evaluator["driver_python"],
        str(evaluator_script),
        "--dataset",
        str(dataset),
        "--candidate-path",
        str(candidate["path"]),
        "--comfy-root",
        evaluator["comfy_root"],
        "--comfy-python",
        evaluator["comfy_python"],
        "--god-root",
        evaluator["god_root"],
        "--output",
        str(result_path),
        "--expected-god-commit",
        evaluator["expected_god_commit"],
    ]
    flags = {
        "base_name": "--base-name",
        "port": "--port",
        "startup_timeout_s": "--startup-timeout-s",
        "evaluation_timeout_s": "--evaluation-timeout-s",
        "shutdown_timeout_s": "--shutdown-timeout-s",
        "expected_comfy_commit": "--expected-comfy-commit",
        "expected_tooling_commit": "--expected-tooling-commit",
    }
    for key, flag in flags.items():
        if key in evaluator:
            command.extend((flag, str(evaluator[key])))
    expected_identity = evaluator.get("_expected_dataset_identity")
    if expected_identity is not None:
        order = expected_identity.get("evaluator_order")
        if not isinstance(order, list) or not order:
            raise ValueError("sealed evaluator image order is missing")
        for image_name in order:
            if not isinstance(image_name, str) or not image_name:
                raise ValueError("sealed evaluator image order is invalid")
            command.extend(("--expected-image", image_name))
    return command


def _validated_process_group(process: subprocess.Popen[str]) -> int:
    pid = process.pid
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        raise RuntimeError(f"unsafe evaluator PID: {pid!r}")
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError as exc:
        raise RuntimeError("evaluator exited before containment was validated") from exc
    if pgid != pid or pgid == os.getpgrp() or pgid <= 1:
        raise RuntimeError(
            f"unsafe evaluator process group: pid={pid}, pgid={pgid}, "
            f"batch_pgid={os.getpgrp()}"
        )
    return pgid


def _signal_process_group(pgid: int, stop_signal: signal.Signals) -> None:
    if (
        not isinstance(pgid, int)
        or isinstance(pgid, bool)
        or pgid <= 1
        or pgid == os.getpgrp()
    ):
        raise RuntimeError(f"refusing unsafe process-group target: {pgid!r}")
    try:
        os.killpg(pgid, stop_signal)
    except ProcessLookupError:
        pass


def _signal_live_process_group(
    process: subprocess.Popen[str], pgid: int, stop_signal: signal.Signals
) -> None:
    """Signal a cached group only while its original leader is still that PID."""

    if process.poll() is not None:
        return
    try:
        current = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    if current != pgid or current != process.pid:
        raise RuntimeError(
            f"evaluator process-group identity changed: pid={process.pid}, "
            f"expected={pgid}, actual={current}"
        )
    _signal_process_group(pgid, stop_signal)


def _terminate_process_group(
    process: subprocess.Popen[str], *, term_grace_s: float
) -> tuple[str, str]:
    """Terminate and reap one validated process group, escalating once."""

    pgid = _validated_process_group(process)
    _signal_process_group(pgid, signal.SIGTERM)
    try:
        return process.communicate(timeout=term_grace_s)
    except subprocess.TimeoutExpired:
        _signal_process_group(pgid, signal.SIGKILL)
        return process.communicate(timeout=max(1.0, term_grace_s))


def _scope_signal(
    unit: str,
    stop_signal: signal.Signals,
    *,
    systemctl_path: str = "/usr/bin/systemctl",
) -> None:
    if not _UNIT_COMPONENT.fullmatch(unit):
        raise RuntimeError(f"unsafe systemd scope unit: {unit!r}")
    result = subprocess.run(
        [
            systemctl_path,
            "kill",
            "--kill-who=all",
            f"--signal={stop_signal.name}",
            f"{unit}.service",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    if result.returncode != 0:
        # A collected/not-found unit is already empty.  Any manager/DBus or
        # permission failure is indeterminate and must fail closed.
        state = _scope_state(unit, systemctl_path=systemctl_path)
        if state != "inactive":
            raise RuntimeError(
                f"could not signal systemd scope {unit}: "
                f"rc={result.returncode}, stderr={result.stderr[-1000:]}"
            )


def _scope_state(unit: str, *, systemctl_path: str = "/usr/bin/systemctl") -> str:
    if not _UNIT_COMPONENT.fullmatch(unit):
        raise RuntimeError(f"unsafe systemd scope unit: {unit!r}")
    result = subprocess.run(
        [
            systemctl_path,
            "show",
            "--no-pager",
            "--all",
            "--property=LoadState",
            "--property=ActiveState",
            "--property=ControlGroup",
            f"{unit}.service",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"systemd scope state is indeterminate for {unit}: "
            f"rc={result.returncode}, stderr={result.stderr[-1000:]}"
        )
    properties = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            properties[key] = value
    if set(properties) != {"LoadState", "ActiveState", "ControlGroup"}:
        raise RuntimeError(f"systemd returned incomplete state for {unit}")
    active = properties["ActiveState"]
    control_group = properties["ControlGroup"]
    if properties["LoadState"] == "not-found":
        if control_group:
            raise RuntimeError(f"not-found systemd unit still reports a cgroup: {unit}")
        return "inactive"
    if active in {"active", "reloading", "activating", "deactivating"}:
        return "active"
    if active not in {"inactive", "failed"}:
        raise RuntimeError(f"unrecognized systemd ActiveState for {unit}: {active}")
    if control_group:
        events = Path("/sys/fs/cgroup") / control_group.lstrip("/") / "cgroup.events"
        if events.is_symlink() or not events.is_file():
            raise RuntimeError(f"cannot prove recursive cgroup emptiness for {unit}")
        values = {}
        for line in events.read_text(encoding="ascii").splitlines():
            parts = line.split()
            if len(parts) == 2:
                values[parts[0]] = parts[1]
        if values.get("populated") not in {"0", "1"}:
            raise RuntimeError(f"invalid cgroup.events for {unit}")
        if values["populated"] == "1":
            return "active"
    return "inactive"


def _scope_is_active(unit: str, *, systemctl_path: str = "/usr/bin/systemctl") -> bool:
    return _scope_state(unit, systemctl_path=systemctl_path) == "active"


def _process_group_is_empty(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError as exc:
        raise RuntimeError(f"cannot prove process group {pgid} is empty") from exc
    return False


def _terminate_scope_and_group(
    process: subprocess.Popen[str],
    *,
    pgid: int,
    unit: str,
    term_grace_s: float,
    systemctl_path: str = "/usr/bin/systemctl",
) -> tuple[str, str]:
    """Kill the cgroup (including setsid descendants), then the client PGID."""

    if (
        not isinstance(pgid, int)
        or isinstance(pgid, bool)
        or pgid <= 1
        or pgid == os.getpgrp()
    ):
        raise RuntimeError(f"refusing unsafe prevalidated process group: {pgid!r}")
    scope_error: BaseException | None = None
    try:
        _scope_signal(unit, signal.SIGTERM, systemctl_path=systemctl_path)
    except BaseException as exc:
        scope_error = exc
    finally:
        _signal_live_process_group(process, pgid, signal.SIGTERM)
    try:
        streams = process.communicate(timeout=term_grace_s)
    except subprocess.TimeoutExpired:
        try:
            _scope_signal(unit, signal.SIGKILL, systemctl_path=systemctl_path)
        except BaseException as exc:
            scope_error = scope_error or exc
        finally:
            _signal_live_process_group(process, pgid, signal.SIGKILL)
        streams = process.communicate(timeout=max(1.0, term_grace_s))
    deadline = time.monotonic() + term_grace_s
    try:
        while (
            _scope_is_active(unit, systemctl_path=systemctl_path)
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        if _scope_is_active(unit, systemctl_path=systemctl_path):
            _scope_signal(unit, signal.SIGKILL, systemctl_path=systemctl_path)
    except BaseException as exc:
        scope_error = scope_error or exc
    deadline = time.monotonic() + max(1.0, term_grace_s)
    while not _process_group_is_empty(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if not _process_group_is_empty(pgid):
        raise RuntimeError(f"evaluator process group still has survivors: {pgid}")
    try:
        # The client PGID can empty before systemd observes recursive cgroup
        # cleanup.  Give a delivered SIGKILL its own bounded propagation poll.
        deadline = time.monotonic() + max(1.0, term_grace_s)
        while (
            _scope_is_active(unit, systemctl_path=systemctl_path)
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        if _scope_is_active(unit, systemctl_path=systemctl_path):
            raise RuntimeError(
                f"systemd scope still has survivors after SIGKILL: {unit}"
            )
    except BaseException as exc:
        scope_error = scope_error or exc
    if scope_error is not None:
        raise RuntimeError(
            f"could not prove complete evaluator containment cleanup: {unit}"
        ) from scope_error
    return streams


def _terminate_scope_only(
    *, unit: str, term_grace_s: float, systemctl_path: str = "/usr/bin/systemctl"
) -> None:
    """Terminate a scope after its systemd-run client has already exited."""

    _scope_signal(unit, signal.SIGTERM, systemctl_path=systemctl_path)
    deadline = time.monotonic() + term_grace_s
    while (
        _scope_is_active(unit, systemctl_path=systemctl_path)
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    if _scope_is_active(unit, systemctl_path=systemctl_path):
        _scope_signal(unit, signal.SIGKILL, systemctl_path=systemctl_path)
        deadline = time.monotonic() + max(1.0, term_grace_s)
        while (
            _scope_is_active(unit, systemctl_path=systemctl_path)
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
    if _scope_is_active(unit, systemctl_path=systemctl_path):
        raise RuntimeError(f"systemd scope still has survivors after SIGKILL: {unit}")


def _run_contained(
    command: list[str],
    *,
    timeout_s: float,
    candidate_id: str,
    containment: dict[str, Any],
) -> subprocess.CompletedProcess[str]:
    if containment["mode"] != "systemd_transient_service":
        raise RuntimeError("unsupported evaluator containment mode")
    suffix = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:12]
    unit = f"forge-krea-eval-{os.getpid()}-{suffix}"
    if not _UNIT_COMPONENT.fullmatch(unit):
        raise RuntimeError("generated an unsafe systemd unit name")
    grace = float(containment["term_grace_s"])
    systemctl_path = containment["systemctl_path"]
    wrapped = [
        containment["systemd_run_path"],
        "--quiet",
        "--wait",
        "--collect",
        "--pipe",
        f"--unit={unit}",
        "--property=KillMode=control-group",
        f"--property=TimeoutStopSec={grace}s",
        f"--property=RuntimeMaxSec={timeout_s + grace}s",
        # Offline mode is enforced by a transient service network namespace,
        # not merely asserted through Hugging Face environment variables.
        # PrivateNetwork retains loopback inside the unit and removes external
        # interfaces; the AF allowlist remains explicit and reviewable.
        "--property=PrivateNetwork=yes",
        "--property=RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "--",
        *command,
    ]
    previous_handlers: dict[signal.Signals, Any] = {}

    def cancel(signum: int, _frame: Any) -> None:
        raise _EvaluatorCancellation(signum)

    if threading.current_thread() is threading.main_thread():
        for stop_signal in (signal.SIGTERM, signal.SIGHUP):
            previous_handlers[stop_signal] = signal.getsignal(stop_signal)
            signal.signal(stop_signal, cancel)
    process: subprocess.Popen[str] | None = None
    environment_root = tempfile.TemporaryDirectory(prefix="forge-krea-eval-env-")
    try:
        evaluator_environment = _minimal_evaluator_environment(
            driver_python=command[0], isolated_root=Path(environment_root.name)
        )
        process = subprocess.Popen(
            wrapped,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=evaluator_environment,
        )
        try:
            pgid = _validated_process_group(process)
        except BaseException:
            try:
                _terminate_scope_only(
                    unit=unit,
                    term_grace_s=grace,
                    systemctl_path=systemctl_path,
                )
            except BaseException:
                pass
            process.terminate()
            try:
                process.communicate(timeout=grace)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=max(1.0, grace))
            raise
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = _terminate_scope_and_group(
                process,
                pgid=pgid,
                unit=unit,
                term_grace_s=grace,
                systemctl_path=systemctl_path,
            )
            raise TimeoutError(
                f"candidate {candidate_id} evaluator exceeded {timeout_s}s; "
                "the complete systemd scope was terminated"
            ) from exc
        except BaseException:
            _terminate_scope_and_group(
                process,
                pgid=pgid,
                unit=unit,
                term_grace_s=grace,
                systemctl_path=systemctl_path,
            )
            raise
        if _scope_is_active(unit, systemctl_path=systemctl_path):
            _terminate_scope_only(
                unit=unit,
                term_grace_s=grace,
                systemctl_path=systemctl_path,
            )
            raise RuntimeError(f"candidate {candidate_id} scope survived normal exit")
    finally:
        for stop_signal, previous in previous_handlers.items():
            signal.signal(stop_signal, previous)
        environment_root.cleanup()
    if process is None:  # pragma: no cover - defensive type narrowing.
        raise RuntimeError("evaluator process was never started")
    return subprocess.CompletedProcess(wrapped, process.returncode, stdout, stderr)


def _publish_exclusive(output: Path, value: dict[str, Any]) -> None:
    _reject_symlink_ancestors(output.parent, "aggregate output parent")
    temporary = Path(f"{output}.tmp")
    if os.path.lexists(output) or os.path.lexists(temporary):
        raise FileExistsError(f"refusing stale aggregate path: {output} or {temporary}")
    payload = krea_provenance.canonical_bytes(value) + b"\n"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, output)
    temporary.unlink()
    directory_fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved.is_relative_to(right_resolved)
        or right_resolved.is_relative_to(left_resolved)
    )


def _reject_symlink_ancestors(path: Path, label: str) -> None:
    current = _absolute_lexical(path)
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent


def _copy_verified(
    source: Path, target: Path, *, expected_sha256: str
) -> dict[str, Any]:
    """Copy one input through no-follow descriptors into an exclusive file."""

    _reject_symlink_ancestors(source, "staged source")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source, flags)
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"staged source is not regular: {source}")
        target_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        digest = hashlib.sha256()
        written = 0
        try:
            while True:
                block = os.read(source_fd, 8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                view = memoryview(block)
                while view:
                    count = os.write(target_fd, view)
                    view = view[count:]
                written += len(block)
            os.fsync(target_fd)
        finally:
            os.close(target_fd)
        after = os.fstat(source_fd)
        identity = lambda item: (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
        )
        actual = digest.hexdigest()
        if identity(before) != identity(after):
            raise RuntimeError(f"staged source changed while copied: {source}")
        if actual != expected_sha256:
            raise ValueError(f"staged source hash mismatch: {source}")
        if _sha256(target) != expected_sha256 or target.stat().st_size != written:
            raise RuntimeError(f"staged copy did not rehash: {target}")
        return {
            "source_path": str(source),
            "staged_path": str(target),
            "sha256": actual,
            "bytes": written,
        }
    finally:
        os.close(source_fd)


def _empty_real_directory(
    path: Path,
    label: str,
    *,
    allowed_zero_byte_placeholder: str | None = None,
) -> Path:
    """Require an empty real directory, optionally allowing one inert marker.

    ComfyUI tracks ``models/loras/put_loras_here`` as a zero-byte regular file.
    It is not a model and cannot be removed from a clean pinned checkout.  The
    exception is deliberately exact: a different name, a symlink, a hardlink,
    a directory, or any non-zero content still fails closed.
    """

    path = _absolute_lexical(path)
    _reject_symlink_ancestors(path, label)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real directory: {path}")
    with os.scandir(path) as scan:
        entries = sorted(list(scan), key=lambda entry: entry.name)
    if not entries:
        return path
    if (
        allowed_zero_byte_placeholder is not None
        and len(entries) == 1
        and entries[0].name == allowed_zero_byte_placeholder
    ):
        placeholder = entries[0]
        details = placeholder.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_size != 0
            or details.st_nlink != 1
        ):
            raise ValueError(
                f"{label} placeholder must be one zero-byte regular file "
                "with no hardlinks"
            )
        return path
    names = [entry.name for entry in entries]
    raise ValueError(f"{label} must be empty; found {names}")


def _publish_decision_evidence_bundle(
    *,
    output: Path,
    plan: dict[str, Any],
    plan_raw: bytes,
    approval_path: Path,
    completed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Publish the raw decision inputs beside the aggregate, then bind them.

    The aggregate is published only after this directory is atomically renamed
    into place.  Thus a crash may leave an unreferenced evidence directory, but
    can never leave a visible aggregate whose raw decision evidence is absent.
    Every path in the manifest is relative to the bundle root so the aggregate
    and its ``.evidence`` directory can be archived off-host as one unit.
    """

    final = output.parent / f"{output.name}.evidence"
    temporary = output.parent / f".{output.name}.evidence.tmp"
    for path in (final, temporary):
        if os.path.lexists(path):
            raise FileExistsError(f"refusing stale decision-evidence path: {path}")
    temporary.mkdir(mode=0o700)
    try:
        results = temporary / "results"
        results.mkdir(mode=0o700)
        plan_target = temporary / "score-plan.json"
        with plan_target.open("xb") as handle:
            handle.write(plan_raw)
            handle.flush()
            os.fsync(handle.fileno())
        if _sha256(plan_target) != hashlib.sha256(plan_raw).hexdigest():
            raise RuntimeError("decision-evidence score-plan copy did not rehash")

        approval_path = _safe_file(approval_path, "sealed plan approval")
        approval_target = temporary / "score-plan-approval.json"
        _copy_verified(
            approval_path,
            approval_target,
            expected_sha256=_sha256(approval_path),
        )
        approval, approval_sha, approval_raw = _load_json_file(
            approval_target, "decision-evidence score-plan approval"
        )
        _canonical_control_file(
            approval,
            approval_raw,
            "decision-evidence score-plan approval",
        )
        result_rows = []
        seen_names: set[str] = set()
        for row in sorted(completed, key=lambda item: item["candidate_id"]):
            result_name = row["result_file"]
            if (
                not isinstance(result_name, str)
                or Path(result_name).name != result_name
                or result_name in {"", ".", ".."}
                or result_name in seen_names
            ):
                raise ValueError("decision-evidence result filename is unsafe")
            seen_names.add(result_name)
            source = _safe_file(row["_result_path"], "raw evaluator result")
            target = results / result_name
            copied = _copy_verified(
                source,
                target,
                expected_sha256=row["result_file_sha256"],
            )
            result, result_sha, _ = _load_json_file(
                target, f"decision-evidence result {row['candidate_id']}"
            )
            canonical_sha = krea_provenance.canonical_sha256(result)
            if (
                copied["sha256"] != result_sha
                or canonical_sha != row["result_canonical_sha256"]
            ):
                raise RuntimeError("decision-evidence result copy changed semantics")
            result_rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "path": f"results/{result_name}",
                    "file_sha256": result_sha,
                    "canonical_sha256": canonical_sha,
                }
            )
        body = {
            "schema": 1,
            "kind": "forge-krea-decision-evidence-bundle",
            "path_rule": "relative_to_this_manifest_parent",
            "score_plan": {
                "path": "score-plan.json",
                "file_sha256": _sha256(plan_target),
                "canonical_sha256": krea_provenance.canonical_sha256(plan),
            },
            "score_plan_approval": {
                "path": "score-plan-approval.json",
                "file_sha256": approval_sha,
                "canonical_sha256": krea_provenance.canonical_sha256(approval),
            },
            "evaluator_results": result_rows,
        }
        manifest = {
            **body,
            "manifest_sha256": krea_provenance.canonical_sha256(body),
        }
        manifest_target = temporary / "manifest.json"
        _publish_exclusive(manifest_target, manifest)
        # Each copied file is already fsynced.  Persist both levels of
        # directory entries as well so a host power loss cannot expose the
        # renamed bundle without one of its nested result names.
        for durable_directory in (results, temporary):
            directory_fd = os.open(
                durable_directory,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        for directory, subdirectories, files in os.walk(temporary, topdown=False):
            for name in files:
                path = Path(directory) / name
                if path.stat().st_nlink != 1:
                    raise RuntimeError(
                        "decision-evidence file has an unexpected hardlink"
                    )
                os.chmod(path, 0o400)
            for name in subdirectories:
                os.chmod(Path(directory) / name, 0o500)
        os.chmod(temporary, 0o500)
        os.rename(temporary, final)
        parent_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return {
            "archive_path": final.name,
            "manifest_path": "manifest.json",
            "manifest_file_sha256": _sha256(final / "manifest.json"),
            "manifest_sha256": manifest["manifest_sha256"],
        }
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise


def _scorer_lease_path(comfy_root: Path) -> Path:
    root = _safe_directory(comfy_root, "scorer lease Comfy root")
    identity = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:20]
    parent = _safe_directory(root.parent, "scorer lease directory")
    return parent / f".forge-krea-scorer-{identity}.lock"


def _acquire_scorer_lease(*, comfy_root: Path, target: Path) -> str:
    """Fail closed if another batch/shard owns the shared scorer surface."""

    lock_path = _scorer_lease_path(comfy_root)
    key = str(lock_path)
    target_text = str(_absolute_lexical(target))
    with _SCORER_LEASE_GUARD:
        if key in _ACTIVE_SCORER_LEASES:
            raise RuntimeError("exact scorer surface is already leased")
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise RuntimeError("could not open exact scorer lease") from exc
        try:
            details = os.fstat(descriptor)
            if (
                not stat.S_ISREG(details.st_mode)
                or details.st_nlink != 1
                or details.st_uid != os.geteuid()
                or details.st_mode & 0o077
            ):
                raise RuntimeError("exact scorer lease file is unsafe")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("exact scorer surface is already leased") from exc
            _ACTIVE_SCORER_LEASES[key] = (descriptor, target_text)
            return key
        except BaseException:
            os.close(descriptor)
            raise


def _release_scorer_lease(key: str, *, target: Path) -> None:
    target_text = str(_absolute_lexical(target))
    with _SCORER_LEASE_GUARD:
        active = _ACTIVE_SCORER_LEASES.pop(key, None)
    if active is None or active[1] != target_text:
        if active is not None:
            with _SCORER_LEASE_GUARD:
                _ACTIVE_SCORER_LEASES[key] = active
        raise RuntimeError("exact scorer lease ownership is inconsistent")
    descriptor = active[0]
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _stage_comfy_lora(
    *, comfy_root: Path, candidate: Path, candidate_sha256: str
) -> dict[str, Any]:
    """Exclusively stage the exact bytes ComfyUI will resolve by LoRA name."""

    lora_name = f"candidate-{candidate_sha256}.safetensors"
    target = _absolute_lexical(comfy_root / "models" / "loras" / lora_name)
    lease_key = _acquire_scorer_lease(comfy_root=comfy_root, target=target)
    try:
        lora_dir = _empty_real_directory(
            comfy_root / "models" / "loras",
            "ComfyUI LoRA staging directory",
            allowed_zero_byte_placeholder=_COMFY_LORA_PLACEHOLDER,
        )
        row = _copy_verified(candidate, target, expected_sha256=candidate_sha256)
        os.chmod(target, 0o400)
        directory_fd = os.open(
            lora_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {
            "comfy_lora_name": lora_name,
            "comfy_lora_path": str(target),
            "sha256": row["sha256"],
            "bytes": row["bytes"],
        }
    except BaseException:
        if target.exists() and not target.is_symlink() and target.is_file():
            target.unlink()
        _release_scorer_lease(lease_key, target=target)
        raise


def _remove_comfy_lora(binding: dict[str, Any], *, comfy_root: Path) -> None:
    lora_dir = _absolute_lexical(comfy_root / "models" / "loras")
    target = _absolute_lexical(binding["comfy_lora_path"])
    lease_key = str(_scorer_lease_path(comfy_root))
    if (
        target.parent != lora_dir
        or target.name != binding["comfy_lora_name"]
    ):
        raise RuntimeError("refusing unsafe ComfyUI LoRA cleanup target")
    try:
        safe = _safe_file(target, "staged ComfyUI LoRA")
        if (
            _sha256(safe) != binding["sha256"]
            or safe.stat().st_size != binding["bytes"]
        ):
            raise RuntimeError("staged ComfyUI LoRA changed before cleanup")
        safe.unlink()
        directory_fd = os.open(
            lora_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _empty_real_directory(
            lora_dir,
            "ComfyUI LoRA staging directory after cleanup",
            allowed_zero_byte_placeholder=_COMFY_LORA_PLACEHOLDER,
        )
    finally:
        _release_scorer_lease(lease_key, target=target)


def _stage_inputs(
    *,
    results_dir: Path,
    dataset: Path,
    candidates: list[dict[str, Any]],
    evaluator_script: Path,
) -> tuple[Path, dict[str, Path], Path, dict[str, Any]]:
    root = results_dir / "_inputs"
    root.mkdir(mode=0o700)
    dataset_target = root / "dataset"
    candidates_target = root / "candidates"
    evaluator_target = root / "evaluator"
    for directory in (dataset_target, candidates_target, evaluator_target):
        directory.mkdir(mode=0o700)
    dataset_rows = []
    _reject_symlink_ancestors(dataset, "dataset")
    with os.scandir(dataset) as scan:
        entries = sorted(scan, key=lambda item: item.name)
    for entry in entries:
        source = Path(entry.path)
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise ValueError(f"dataset contains unsafe entry during staging: {source}")
        dataset_rows.append(
            _copy_verified(
                source,
                dataset_target / entry.name,
                expected_sha256=_sha256(source),
            )
        )
    staged_candidates: dict[str, Path] = {}
    candidate_rows = []
    for candidate in candidates:
        directory = candidates_target / candidate["id"]
        directory.mkdir(mode=0o700)
        target = directory / candidate["path"].name
        candidate_rows.append(
            {
                "candidate_id": candidate["id"],
                **_copy_verified(
                    candidate["path"], target, expected_sha256=candidate["sha256"]
                ),
            }
        )
        staged_candidates[candidate["id"]] = target
    script_target = evaluator_target / evaluator_script.name
    script_row = _copy_verified(
        evaluator_script,
        script_target,
        expected_sha256=_sha256(evaluator_script),
    )
    identity_module = evaluator_script.with_name("krea_dataset_identity.py")
    module_target = evaluator_target / identity_module.name
    module_row = _copy_verified(
        identity_module,
        module_target,
        expected_sha256=_sha256(identity_module),
    )
    for directory, subdirs, files in os.walk(root, topdown=False):
        for name in files:
            os.chmod(Path(directory) / name, 0o400)
        for name in subdirs:
            os.chmod(Path(directory) / name, 0o500)
    os.chmod(root, 0o500)
    body = {
        "schema": 1,
        "kind": "forge-krea-exclusive-staged-inputs",
        "dataset": dataset_rows,
        "candidates": candidate_rows,
        "evaluator_script": script_row,
        "dataset_identity_module": module_row,
    }
    manifest = {**body, "manifest_sha256": krea_provenance.canonical_sha256(body)}
    return dataset_target, staged_candidates, script_target, manifest


def _candidate_completion_row(
    *,
    candidate: dict[str, Any],
    result_path: Path,
    result_file_sha256: str,
    result: dict[str, Any],
    log_path: Path,
    comfy_staging: dict[str, Any],
) -> dict[str, Any]:
    """Normalize one independently validated result for later publication."""

    return {
        "candidate_id": candidate["id"],
        "arm_id": candidate["source_arm_id"],
        "family_id": candidate["source_arm_id"],
        "mode": candidate["candidate_binding"]["mode"],
        "source_arm_id": candidate["source_arm_id"],
        "candidate_sha256": candidate["sha256"],
        "candidate_bytes": candidate["path"].stat().st_size,
        "provenance_manifest_sha256": candidate["provenance"]["manifest_sha256"],
        "provenance_file_sha256": candidate["provenance_file_sha256"],
        "normalized_recipe": candidate["candidate_binding"].get(
            "normalized_recipe", candidate["provenance"].get("normalized_recipe")
        ),
        "candidate_binding": candidate["candidate_binding"],
        "result_file": result_path.name,
        "result_file_sha256": result_file_sha256,
        "result_canonical_sha256": krea_provenance.canonical_sha256(result),
        "weighted_loss": result["weighted_loss"],
        "text_mean": result["text_mean"],
        "blank_mean": result["blank_mean"],
        "paired_rows": result["scored_rows"],
        "mechanics": (
            {
                "natural_completion": True,
                "upload_ready": True,
                "clean_telemetry": True,
            }
            if candidate["candidate_binding"]["mode"] == "local_run_candidate"
            else None
        ),
        "comfy_staging": comfy_staging,
        "_candidate_path": str(candidate["path"]),
        "_provenance_path": str(candidate["provenance_path"]),
        "_result_path": str(result_path),
        "_log_path": str(log_path),
        "_log_sha256": result["runtime"]["comfy_log_sha256"],
        **{key: value for key, value in candidate.items() if key.startswith("_")},
    }


def run_batch(
    plan: dict[str, Any],
    *,
    results_dir: Path,
    output: Path,
    plan_raw: bytes | None = None,
) -> dict[str, Any]:
    if plan_raw is None:
        plan_raw = krea_provenance.canonical_bytes(plan) + b"\n"
    try:
        decoded_plan = json.loads(plan_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("raw plan bytes are not valid JSON") from exc
    if decoded_plan != plan:
        raise ValueError("raw plan bytes do not decode to the supplied plan")
    plan_raw_sha256 = hashlib.sha256(plan_raw).hexdigest()
    plan_canonical_sha256 = krea_provenance.canonical_sha256(plan)
    dataset, dataset_sha256, candidates, evaluator = _validate_plan(plan)
    results_dir = _absolute_lexical(results_dir)
    output = _absolute_lexical(output)
    if _paths_overlap(results_dir, output) or _paths_overlap(
        results_dir, Path(f"{output}.tmp")
    ):
        raise ValueError("aggregate output and results-dir must not overlap")
    if _paths_overlap(results_dir, dataset) or _paths_overlap(output, dataset):
        raise ValueError("evidence outputs must not overlap evaluator inputs")
    for candidate in candidates:
        if _paths_overlap(results_dir, candidate["path"]) or _paths_overlap(
            output, candidate["path"]
        ):
            raise ValueError("evidence outputs must not overlap candidate inputs")
    if output.name == _FORBIDDEN_OUTPUT:
        raise ValueError(f"refusing production selection filename: {_FORBIDDEN_OUTPUT}")
    evidence_path = output.parent / f"{output.name}.evidence"
    evidence_temporary = output.parent / f".{output.name}.evidence.tmp"
    if any(
        os.path.lexists(path)
        for path in (output, Path(f"{output}.tmp"), evidence_path, evidence_temporary)
    ):
        raise FileExistsError(f"refusing stale aggregate path: {output}")
    _reject_symlink_ancestors(output.parent, "aggregate output parent")
    _reject_symlink_ancestors(results_dir.parent, "results parent")
    output.parent.mkdir(parents=True, exist_ok=True)
    results_dir.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(output.parent, "aggregate output parent")
    _reject_symlink_ancestors(results_dir.parent, "results parent")
    try:
        results_dir.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise FileExistsError("results-dir must be exclusively new") from exc

    candidate_result_paths = {
        results_dir / f"{index:03d}-{candidate['id']}.json"
        for index, candidate in enumerate(candidates)
    }
    reserved_result_paths = set(candidate_result_paths)
    reserved_result_paths |= {Path(f"{path}.tmp") for path in candidate_result_paths}
    reserved_result_paths |= {
        Path(f"{path}.comfy.log") for path in candidate_result_paths
    }
    if output in reserved_result_paths:
        raise ValueError("aggregate output collides with per-candidate evidence")

    evaluator_script = (
        Path(__file__).with_name("evaluate_krea_local.py").resolve(strict=True)
    )
    (
        staged_dataset,
        staged_candidates,
        staged_evaluator_script,
        staged_input_manifest,
    ) = _stage_inputs(
        results_dir=results_dir,
        dataset=dataset,
        candidates=candidates,
        evaluator_script=evaluator_script,
    )
    evaluator_script_sha = _sha256(staged_evaluator_script)
    comfy_root = _safe_directory(evaluator["comfy_root"], "comfy_root")
    _empty_real_directory(
        comfy_root / "models" / "loras",
        "ComfyUI LoRA staging directory",
        allowed_zero_byte_placeholder=_COMFY_LORA_PLACEHOLDER,
    )
    completed: list[dict[str, Any]] = []
    common_envelope: dict[str, Any] | None = None
    common_envelope_sha: str | None = None
    for index, candidate in enumerate(candidates):
        result_path = results_dir / f"{index:03d}-{candidate['id']}.json"
        temp_path = Path(f"{result_path}.tmp")
        log_path = Path(f"{result_path}.comfy.log")
        if any(os.path.lexists(path) for path in (result_path, temp_path, log_path)):
            raise FileExistsError(f"stale candidate evidence for {candidate['id']}")
        staged_candidate = staged_candidates[candidate["id"]]
        comfy_staging: dict[str, Any] | None = None
        try:
            comfy_staging = _stage_comfy_lora(
                comfy_root=comfy_root,
                candidate=staged_candidate,
                candidate_sha256=candidate["sha256"],
            )
            command = _evaluator_command(
                evaluator_script=staged_evaluator_script,
                dataset=staged_dataset,
                candidate={**candidate, "path": staged_candidate},
                result_path=result_path,
                evaluator=evaluator,
            )
            timeout = _timeout_policy(evaluator)["total_candidate_timeout_s"]
            process = _run_contained(
                command,
                timeout_s=timeout,
                candidate_id=candidate["id"],
                containment=evaluator["containment"],
            )
            if process.returncode != 0:
                raise RuntimeError(
                    f"candidate {candidate['id']} evaluator failed "
                    f"({process.returncode}): {process.stderr[-4000:]}"
                )
            if os.path.lexists(temp_path):
                raise RuntimeError(f"candidate {candidate['id']} left a partial result")
            result, result_file_sha, _ = _load_json_file(
                result_path, f"result {candidate['id']}"
            )
            envelope = _validate_result(
                result,
                candidate={**candidate, "path": staged_candidate},
                dataset=staged_dataset,
                expected_dataset_sha256=dataset_sha256,
                expected_dataset_identity=evaluator.get("_expected_dataset_identity"),
                evaluator=evaluator,
                evaluator_script_sha=evaluator_script_sha,
                log_path=log_path,
                comfy_staging=comfy_staging,
            )
        finally:
            if comfy_staging is not None:
                _remove_comfy_lora(comfy_staging, comfy_root=comfy_root)
        if comfy_staging is None:  # pragma: no cover - fail-closed type guard
            raise RuntimeError("candidate LoRA was not staged")
        envelope_sha = krea_provenance.canonical_sha256(envelope)
        if common_envelope is None:
            common_envelope = envelope
            common_envelope_sha = envelope_sha
        elif envelope_sha != common_envelope_sha or envelope != common_envelope:
            raise RuntimeError(
                f"candidate {candidate['id']} escaped the common evaluation envelope"
            )
        completed.append(
            _candidate_completion_row(
                candidate=candidate,
                result_path=result_path,
                result_file_sha256=result_file_sha,
                result=result,
                log_path=log_path,
                comfy_staging=comfy_staging,
            )
        )

    if (
        len(completed) != len(candidates)
        or common_envelope is None
        or common_envelope_sha is None
    ):
        raise RuntimeError("refusing incomplete batch coverage")
    # Rebind every file immediately before publication.  A long batch cannot
    # silently certify an input or earlier result that changed between arms.
    public_completed: list[dict[str, Any]] = []
    for row in completed:
        if _sha256(Path(row["_candidate_path"])) != row["candidate_sha256"]:
            raise RuntimeError(
                f"candidate {row['candidate_id']} changed during the batch"
            )
        if _sha256(Path(row["_provenance_path"])) != row["provenance_file_sha256"]:
            raise RuntimeError(
                f"provenance {row['candidate_id']} changed during the batch"
            )
        if _sha256(Path(row["_result_path"])) != row["result_file_sha256"]:
            raise RuntimeError(f"result {row['candidate_id']} changed during the batch")
        if _sha256(Path(row["_log_path"])) != row["_log_sha256"]:
            raise RuntimeError(f"log {row['candidate_id']} changed during the batch")
        for prefix in (
            "fixture_split_manifest",
            "training_condition",
            "completion_manifest",
            "run_record",
            "training_log",
            "source_normalization_approval",
        ):
            path_key = f"_{prefix}_path"
            sha_key = f"_{prefix}_sha256"
            if path_key in row and _sha256(Path(row[path_key])) != row[sha_key]:
                raise RuntimeError(
                    f"{prefix.replace('_', ' ')} {row['candidate_id']} "
                    "changed during the batch"
                )
        for path_key, bound_path in row.items():
            if not path_key.startswith("_") or not path_key.endswith("_path"):
                continue
            digest_key = f"{path_key[:-5]}_sha256"
            if digest_key in row and _sha256(Path(bound_path)) != row[digest_key]:
                raise RuntimeError(
                    f"bound evidence {path_key} for {row['candidate_id']} "
                    "changed during the batch"
                )
        public_completed.append(
            {key: value for key, value in row.items() if not key.startswith("_")}
        )
    sealed_path = Path(evaluator["_sealed_plan_approval_path"])
    if _sha256(sealed_path) != evaluator["_sealed_plan_approval_sha256"]:
        raise RuntimeError("sealed plan approval changed during the batch")
    for path_key, sha_key, label in (
        (
            "_fixture_manifest_path",
            "_fixture_manifest_file_sha256",
            "fixture manifest",
        ),
        (
            "_campaign_manifest_path",
            "_campaign_manifest_file_sha256",
            "campaign manifest",
        ),
        (
            "_fixture_approval_path",
            "_fixture_approval_file_sha256",
            "fixture approval",
        ),
        (
            "_cross_fixture_review_path",
            "_cross_fixture_review_file_sha256",
            "cross-fixture human review",
        ),
    ):
        if (
            path_key in evaluator
            and _sha256(Path(evaluator[path_key])) != evaluator[sha_key]
        ):
            raise RuntimeError(f"{label} changed during the batch")
    decision_context = evaluator.get("_decision_context")
    if decision_context is not None and decision_context["phase"] == "boundary":
        frozen_path = Path(decision_context["_frozen_discovery_decision_path"])
        if (
            _sha256(frozen_path)
            != decision_context["_frozen_discovery_decision_file_sha256"]
        ):
            raise RuntimeError("frozen discovery decision changed during the batch")
    for group in ("dataset", "candidates"):
        for staged in staged_input_manifest[group]:
            if _sha256(Path(staged["staged_path"])) != staged["sha256"]:
                raise RuntimeError("immutable staged input changed during evaluation")
    for group in ("evaluator_script", "dataset_identity_module"):
        staged = staged_input_manifest[group]
        if _sha256(Path(staged["staged_path"])) != staged["sha256"]:
            raise RuntimeError("immutable staged evaluator code changed")
    is_v2 = "_campaign_manifest_sha256" in evaluator
    if is_v2:
        campaign, campaign_file_sha, _campaign_raw = _load_json_file(
            Path(evaluator["_campaign_manifest_path"]), "campaign manifest"
        )
        if campaign_file_sha != evaluator["_campaign_manifest_file_sha256"]:
            raise RuntimeError("campaign manifest changed before publication")
        _validate_campaign_manifest(campaign)
        fixture, fixture_file_sha, _fixture_raw = _load_json_file(
            Path(evaluator["_fixture_manifest_path"]), "fixture manifest"
        )
        if fixture_file_sha != evaluator["_fixture_manifest_file_sha256"]:
            raise RuntimeError("fixture manifest changed before publication")
        evaluator.get("_fixture_validator", krea_fixture).validate_manifest(fixture)
    body: dict[str, Any] = {
        "schema": 2 if is_v2 else _SCHEMA,
        "kind": _KIND,
        "coverage": {
            "planned": len(candidates),
            "completed": len(completed),
            "complete": True,
        },
        "direction": "min",
        "plan": {
            "raw_sha256": plan_raw_sha256,
            "canonical_sha256": plan_canonical_sha256,
            "approved_payload_sha256": evaluator["_plan_payload_sha256"],
        },
        "sealed_plan_approval_sha256": evaluator["_sealed_plan_approval_sha256"],
        "sealed_plan_approval": evaluator["_sealed_plan_approval"],
        "batch_runner_sha256": evaluator["_batch_runner_sha256"],
        "staged_input_manifest": staged_input_manifest,
        "common_training_envelope": evaluator["_common_training_envelope"],
        "common_training_envelope_sha256": evaluator[
            "_common_training_envelope_sha256"
        ],
        "evaluator_script_sha256": evaluator_script_sha,
        "evaluation_envelope": common_envelope,
        "evaluation_envelope_sha256": common_envelope_sha,
        "candidates": public_completed,
    }
    if is_v2:
        schema2_candidates = []
        for row in public_completed:
            binding = row["candidate_binding"]
            fraction = binding["candidate_fraction"]
            schema2_candidates.append(
                {
                    "candidate_id": row["candidate_id"],
                    "arm_id": (
                        row["arm_id"] if row["mode"] == "local_run_candidate" else None
                    ),
                    "mode": row["mode"],
                    "family_id": (
                        row["family_id"]
                        if row["mode"] == "local_run_candidate"
                        else None
                    ),
                    "candidate_sha256": row["candidate_sha256"],
                    "candidate_bytes": row["candidate_bytes"],
                    "execution_plan_sha256": binding.get("execution_plan_sha256"),
                    "run_completion_sha256": binding.get("run_completion_sha256"),
                    "step": binding["candidate_step"],
                    "fraction_numerator": (
                        fraction["numerator"] if fraction is not None else None
                    ),
                    "fraction_denominator": (
                        fraction["denominator"] if fraction is not None else None
                    ),
                    "image_exposures": (
                        binding["candidate_image_exposures"]
                        if row["mode"] == "local_run_candidate"
                        else None
                    ),
                    "binding_manifest_sha256": binding["binding_manifest_sha256"],
                    "zero_control_manifest_sha256": (
                        binding["zero_control_manifest_sha256"]
                        if row["mode"] == "zero_lora_control"
                        else None
                    ),
                    "result_file": row["result_file"],
                    "result_file_sha256": row["result_file_sha256"],
                    "result_canonical_sha256": row["result_canonical_sha256"],
                    "weighted_loss": row["weighted_loss"],
                    "text_mean": row["text_mean"],
                    "blank_mean": row["blank_mean"],
                    "paired_rows": row["paired_rows"],
                    "mechanics": row["mechanics"],
                }
            )
        body["candidates"] = schema2_candidates
        body.update(
            {
                "campaign_manifest_sha256": evaluator["_campaign_manifest_file_sha256"],
                "fixture_manifest_sha256": evaluator["_fixture_manifest_file_sha256"],
                "fixture_approval_sha256": evaluator["_fixture_approval_file_sha256"],
                "fixture_contract": {
                    "fixture_manifest_identity_sha256": fixture["manifest_sha256"],
                    "training_pair_count": len(fixture["training_rows"]),
                    "evaluation_row_count": len(fixture["evaluation_rows"]),
                    "training_dataset_sha256": fixture["training_dataset_identity"][
                        "sha256"
                    ],
                    "evaluation_dataset_sha256": fixture["evaluation_dataset_identity"][
                        "sha256"
                    ],
                    "cross_fixture_review_sha256": evaluator[
                        "_cross_fixture_review_file_sha256"
                    ],
                },
                "campaign": {
                    "manifest_sha256": evaluator["_campaign_manifest_sha256"],
                    "file_sha256": evaluator["_campaign_manifest_file_sha256"],
                    "fixture_manifest_sha256": campaign["fixture_manifest_sha256"],
                    "discovery_plan_sha256": campaign["discovery_plan_sha256"],
                    "zero_control_manifest_sha256": campaign[
                        "zero_control_manifest_sha256"
                    ],
                    "decision_contract": campaign["decision_contract"],
                    "confirmation_contract": campaign["confirmation_contract"],
                    "runs": campaign["runs"],
                },
                "fixture": {
                    "manifest_sha256": fixture["manifest_sha256"],
                    "file_sha256": evaluator["_fixture_manifest_file_sha256"],
                    "concept_id": fixture["concept_id"],
                    "experimental_role": fixture["experimental_role"],
                    "evaluation_dataset_sha256": dataset_sha256,
                },
                "training_run_envelopes": _schema2_training_run_envelopes(
                    campaign=campaign,
                    candidates=schema2_candidates,
                    envelopes=evaluator["_training_run_envelopes"],
                    decision_context=evaluator["_decision_context"],
                ),
            }
        )
        body["decision_evidence"] = _publish_decision_evidence_bundle(
            output=output,
            plan=plan,
            plan_raw=plan_raw,
            approval_path=sealed_path,
            completed=completed,
        )
    aggregate = {**body, "aggregate_sha256": krea_provenance.canonical_sha256(body)}
    _publish_exclusive(output, aggregate)
    return aggregate


def _validated_plan_bytes(
    plan: dict[str, Any], plan_raw: bytes | None
) -> tuple[bytes, dict[str, str]]:
    if plan_raw is None:
        plan_raw = krea_provenance.canonical_bytes(plan) + b"\n"
    try:
        decoded = json.loads(plan_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("raw plan bytes are not valid JSON") from exc
    if decoded != plan:
        raise ValueError("raw plan bytes do not decode to the supplied plan")
    return plan_raw, {
        "raw_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "canonical_sha256": krea_provenance.canonical_sha256(plan),
    }


def _portable_staged_inputs(
    manifest: dict[str, Any], *, results_dir: Path
) -> dict[str, Any]:
    def portable(row: dict[str, Any], *, candidate_id: str | None = None) -> dict[str, Any]:
        path = Path(row["staged_path"])
        try:
            relative = path.relative_to(results_dir)
        except ValueError as exc:
            raise RuntimeError("staged input escaped its candidate shard") from exc
        value = {
            "path": relative.as_posix(),
            "sha256": row["sha256"],
            "bytes": row["bytes"],
        }
        if candidate_id is not None:
            value["candidate_id"] = candidate_id
        return value

    body = {
        "schema": 1,
        "kind": "forge-krea-portable-candidate-staged-inputs",
        "path_rule": "relative_to_shard_manifest_parent",
        "dataset": [portable(row) for row in manifest["dataset"]],
        "candidates": [
            portable(row, candidate_id=row["candidate_id"])
            for row in manifest["candidates"]
        ],
        "evaluator_script": portable(manifest["evaluator_script"]),
        "dataset_identity_module": portable(manifest["dataset_identity_module"]),
    }
    return {**body, "manifest_sha256": krea_provenance.canonical_sha256(body)}


def _relative_shard_file(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{label} path must be a strict relative path")
    path = root.joinpath(*relative.parts)
    _reject_symlink_ancestors(path, label)
    return _safe_file(path, label)


def _validate_portable_file_row(
    row: Any,
    *,
    root: Path,
    label: str,
    expected_path: str,
    expected_sha256: str,
    expected_bytes: int,
    candidate_id: str | None = None,
) -> Path:
    value = _object(row, label)
    required = {"path", "sha256", "bytes"}
    if candidate_id is not None:
        required.add("candidate_id")
    _exact_keys(value, required, label)
    if candidate_id is not None and value["candidate_id"] != candidate_id:
        raise ValueError(f"{label} candidate id mismatch")
    if (
        value["path"] != expected_path
        or value["sha256"] != expected_sha256
        or value["bytes"] != expected_bytes
    ):
        raise ValueError(f"{label} differs from the approved input")
    path = _relative_shard_file(root, value["path"], label)
    if _sha256(path) != expected_sha256 or path.stat().st_size != expected_bytes:
        raise ValueError(f"{label} bytes changed")
    return path


def _staged_dataset_rows(dataset: Path) -> list[dict[str, Any]]:
    _reject_symlink_ancestors(dataset, "exact-evaluation dataset")
    with os.scandir(dataset) as scan:
        entries = sorted(scan, key=lambda item: item.name)
    rows = []
    for entry in entries:
        source = Path(entry.path)
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise ValueError(f"dataset contains unsafe entry: {source}")
        rows.append(
            {
                "path": f"_inputs/dataset/{entry.name}",
                "sha256": _sha256(source),
                "bytes": source.stat().st_size,
            }
        )
    return rows


def run_candidate_shard(
    plan: dict[str, Any],
    *,
    candidate_id: str,
    results_dir: Path,
    output: Path,
    plan_raw: bytes | None = None,
) -> dict[str, Any]:
    """Evaluate exactly one approved candidate and publish a create-only shard."""

    if plan.get("schema") != 2:
        raise ValueError("candidate shards require a schema-2 approved plan")
    plan_raw, plan_hashes = _validated_plan_bytes(plan, plan_raw)
    dataset, dataset_sha256, candidates, evaluator = _validate_plan(plan)
    selected = [candidate for candidate in candidates if candidate["id"] == candidate_id]
    if len(selected) != 1:
        raise ValueError("candidate shard id must select exactly one planned candidate")
    candidate = selected[0]
    results_dir = _absolute_lexical(results_dir)
    output = _absolute_lexical(output)
    if output != results_dir / "shard.json":
        raise ValueError("candidate shard output must be RESULTS_DIR/shard.json")
    if output.name == _FORBIDDEN_OUTPUT:
        raise ValueError(f"refusing production selection filename: {_FORBIDDEN_OUTPUT}")
    if _paths_overlap(results_dir, dataset) or _paths_overlap(
        results_dir, candidate["path"]
    ):
        raise ValueError("candidate shard output overlaps an approved input")
    _reject_symlink_ancestors(results_dir.parent, "candidate shard parent")
    results_dir.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(results_dir.parent, "candidate shard parent")
    try:
        results_dir.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise FileExistsError("candidate shard results-dir must be exclusively new") from exc

    result_path = results_dir / f"{candidate['id']}.json"
    temp_path = Path(f"{result_path}.tmp")
    log_path = Path(f"{result_path}.comfy.log")
    if any(os.path.lexists(path) for path in (output, result_path, temp_path, log_path)):
        raise FileExistsError("candidate shard has stale evidence paths")
    evaluator_script = Path(__file__).with_name("evaluate_krea_local.py").resolve(
        strict=True
    )
    (
        staged_dataset,
        staged_candidates,
        staged_evaluator_script,
        staged_input_manifest,
    ) = _stage_inputs(
        results_dir=results_dir,
        dataset=dataset,
        candidates=[candidate],
        evaluator_script=evaluator_script,
    )
    evaluator_script_sha = _sha256(staged_evaluator_script)
    comfy_root = _safe_directory(evaluator["comfy_root"], "comfy_root")
    _empty_real_directory(
        comfy_root / "models" / "loras",
        "ComfyUI LoRA staging directory",
        allowed_zero_byte_placeholder=_COMFY_LORA_PLACEHOLDER,
    )
    staged_candidate = staged_candidates[candidate["id"]]
    comfy_staging: dict[str, Any] | None = None
    try:
        comfy_staging = _stage_comfy_lora(
            comfy_root=comfy_root,
            candidate=staged_candidate,
            candidate_sha256=candidate["sha256"],
        )
        command = _evaluator_command(
            evaluator_script=staged_evaluator_script,
            dataset=staged_dataset,
            candidate={**candidate, "path": staged_candidate},
            result_path=result_path,
            evaluator=evaluator,
        )
        process = _run_contained(
            command,
            timeout_s=_timeout_policy(evaluator)["total_candidate_timeout_s"],
            candidate_id=candidate["id"],
            containment=evaluator["containment"],
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"candidate {candidate['id']} evaluator failed "
                f"({process.returncode}): {process.stderr[-4000:]}"
            )
        if os.path.lexists(temp_path):
            raise RuntimeError(f"candidate {candidate['id']} left a partial result")
        result, result_file_sha, _ = _load_json_file(
            result_path, f"result {candidate['id']}"
        )
        envelope = _validate_result(
            result,
            candidate={**candidate, "path": staged_candidate},
            dataset=staged_dataset,
            expected_dataset_sha256=dataset_sha256,
            expected_dataset_identity=evaluator.get("_expected_dataset_identity"),
            evaluator=evaluator,
            evaluator_script_sha=evaluator_script_sha,
            log_path=log_path,
            comfy_staging=comfy_staging,
        )
    finally:
        if comfy_staging is not None:
            _remove_comfy_lora(comfy_staging, comfy_root=comfy_root)
    if comfy_staging is None:  # pragma: no cover - fail-closed type guard
        raise RuntimeError("candidate LoRA was not staged")

    completed = _candidate_completion_row(
        candidate=candidate,
        result_path=result_path,
        result_file_sha256=result_file_sha,
        result=result,
        log_path=log_path,
        comfy_staging=comfy_staging,
    )
    for path_key, digest_key in (
        ("_candidate_path", "candidate_sha256"),
        ("_provenance_path", "provenance_file_sha256"),
        ("_result_path", "result_file_sha256"),
        ("_log_path", "_log_sha256"),
    ):
        if _sha256(Path(completed[path_key])) != completed[digest_key]:
            raise RuntimeError(f"candidate shard input changed: {path_key}")
    portable_inputs = _portable_staged_inputs(
        staged_input_manifest, results_dir=results_dir
    )
    for group in ("dataset", "candidates"):
        for row in staged_input_manifest[group]:
            if _sha256(Path(row["staged_path"])) != row["sha256"]:
                raise RuntimeError("candidate shard staged input changed")
    for group in ("evaluator_script", "dataset_identity_module"):
        row = staged_input_manifest[group]
        if _sha256(Path(row["staged_path"])) != row["sha256"]:
            raise RuntimeError("candidate shard staged evaluator changed")
    public_completed = {
        key: value for key, value in completed.items() if not key.startswith("_")
    }
    result_binding = {
        "path": result_path.name,
        "file_sha256": result_file_sha,
        "canonical_sha256": krea_provenance.canonical_sha256(result),
        "log_path": log_path.name,
        "log_sha256": result["runtime"]["comfy_log_sha256"],
        "log_bytes": result["runtime"]["comfy_log_bytes"],
        "evaluation_envelope": envelope,
        "evaluation_envelope_sha256": krea_provenance.canonical_sha256(envelope),
    }
    body = {
        "schema": 1,
        "kind": "forge-krea-exact-score-candidate-shard",
        "status": "complete",
        "path_rule": "relative_to_shard_manifest_parent",
        "plan": {
            **plan_hashes,
            "approved_payload_sha256": evaluator["_plan_payload_sha256"],
        },
        "sealed_plan_approval_sha256": evaluator["_sealed_plan_approval_sha256"],
        "batch_runner_sha256": evaluator["_batch_runner_sha256"],
        "candidate": {
            "id": candidate["id"],
            "source_arm_id": candidate["source_arm_id"],
            "sha256": candidate["sha256"],
            "bytes": candidate["path"].stat().st_size,
            "binding_manifest_sha256": candidate["provenance_file_sha256"],
            "candidate_binding": candidate["candidate_binding"],
            "candidate_binding_sha256": krea_provenance.canonical_sha256(
                candidate["candidate_binding"]
            ),
        },
        "staged_inputs": portable_inputs,
        "result": result_binding,
        "completed_candidate": public_completed,
    }
    shard = {**body, "shard_sha256": krea_provenance.canonical_sha256(body)}
    _publish_exclusive(output, shard)
    return shard


def _validate_candidate_shard(
    shard_path: Path,
    *,
    plan_hashes: dict[str, str],
    dataset: Path,
    dataset_sha256: str,
    candidate: dict[str, Any],
    evaluator: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    shard_path = _safe_file(shard_path, "candidate shard manifest")
    _reject_symlink_ancestors(shard_path, "candidate shard manifest")
    shard, shard_file_sha, shard_raw = _load_json_file(
        shard_path, "candidate shard manifest"
    )
    _canonical_control_file(shard, shard_raw, "candidate shard manifest")
    _exact_keys(
        shard,
        {
            "schema",
            "kind",
            "status",
            "path_rule",
            "plan",
            "sealed_plan_approval_sha256",
            "batch_runner_sha256",
            "candidate",
            "staged_inputs",
            "result",
            "completed_candidate",
            "shard_sha256",
        },
        "candidate shard manifest",
    )
    body = {key: value for key, value in shard.items() if key != "shard_sha256"}
    if (
        shard["schema"] != 1
        or shard["kind"] != "forge-krea-exact-score-candidate-shard"
        or shard["status"] != "complete"
        or shard["path_rule"] != "relative_to_shard_manifest_parent"
        or shard["shard_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("candidate shard manifest is incomplete or does not reseal")
    expected_plan = {
        **plan_hashes,
        "approved_payload_sha256": evaluator["_plan_payload_sha256"],
    }
    if (
        shard["plan"] != expected_plan
        or shard["sealed_plan_approval_sha256"]
        != evaluator["_sealed_plan_approval_sha256"]
        or shard["batch_runner_sha256"] != evaluator["_batch_runner_sha256"]
    ):
        raise ValueError("candidate shard differs from the approved score plan")
    candidate_binding = candidate["candidate_binding"]
    expected_candidate = {
        "id": candidate["id"],
        "source_arm_id": candidate["source_arm_id"],
        "sha256": candidate["sha256"],
        "bytes": candidate["path"].stat().st_size,
        "binding_manifest_sha256": candidate["provenance_file_sha256"],
        "candidate_binding": candidate_binding,
        "candidate_binding_sha256": krea_provenance.canonical_sha256(
            candidate_binding
        ),
    }
    if shard["candidate"] != expected_candidate:
        raise ValueError("candidate shard binding differs from the planned candidate")

    root = shard_path.parent
    staged = _object(shard["staged_inputs"], "candidate shard staged inputs")
    _exact_keys(
        staged,
        {
            "schema",
            "kind",
            "path_rule",
            "dataset",
            "candidates",
            "evaluator_script",
            "dataset_identity_module",
            "manifest_sha256",
        },
        "candidate shard staged inputs",
    )
    staged_body = {
        key: value for key, value in staged.items() if key != "manifest_sha256"
    }
    if (
        staged["schema"] != 1
        or staged["kind"] != "forge-krea-portable-candidate-staged-inputs"
        or staged["path_rule"] != "relative_to_shard_manifest_parent"
        or staged["manifest_sha256"]
        != krea_provenance.canonical_sha256(staged_body)
    ):
        raise ValueError("candidate shard staged-input manifest does not reseal")
    expected_dataset_rows = _staged_dataset_rows(dataset)
    dataset_rows = staged["dataset"]
    if not isinstance(dataset_rows, list) or len(dataset_rows) != len(
        expected_dataset_rows
    ):
        raise ValueError("candidate shard dataset coverage is incomplete")
    rebound: list[dict[str, Any]] = []
    for index, expected in enumerate(expected_dataset_rows):
        path = _validate_portable_file_row(
            dataset_rows[index],
            root=root,
            label=f"candidate shard dataset[{index}]",
            expected_path=expected["path"],
            expected_sha256=expected["sha256"],
            expected_bytes=expected["bytes"],
        )
        rebound.append({"path": str(path), "sha256": expected["sha256"]})
    candidate_rows = staged["candidates"]
    if not isinstance(candidate_rows, list) or len(candidate_rows) != 1:
        raise ValueError("candidate shard must stage exactly one candidate")
    expected_candidate_path = (
        f"_inputs/candidates/{candidate['id']}/{candidate['path'].name}"
    )
    staged_candidate = _validate_portable_file_row(
        candidate_rows[0],
        root=root,
        label="candidate shard candidate",
        expected_path=expected_candidate_path,
        expected_sha256=candidate["sha256"],
        expected_bytes=candidate["path"].stat().st_size,
        candidate_id=candidate["id"],
    )
    rebound.append({"path": str(staged_candidate), "sha256": candidate["sha256"]})
    evaluator_source = Path(__file__).with_name("evaluate_krea_local.py").resolve(
        strict=True
    )
    staged_evaluator = _validate_portable_file_row(
        staged["evaluator_script"],
        root=root,
        label="candidate shard evaluator",
        expected_path=f"_inputs/evaluator/{evaluator_source.name}",
        expected_sha256=_sha256(evaluator_source),
        expected_bytes=evaluator_source.stat().st_size,
    )
    rebound.append({"path": str(staged_evaluator), "sha256": _sha256(evaluator_source)})
    identity_source = evaluator_source.with_name("krea_dataset_identity.py")
    staged_identity = _validate_portable_file_row(
        staged["dataset_identity_module"],
        root=root,
        label="candidate shard dataset identity module",
        expected_path=f"_inputs/evaluator/{identity_source.name}",
        expected_sha256=_sha256(identity_source),
        expected_bytes=identity_source.stat().st_size,
    )
    rebound.append({"path": str(staged_identity), "sha256": _sha256(identity_source)})

    result_binding = _object(shard["result"], "candidate shard result binding")
    _exact_keys(
        result_binding,
        {
            "path",
            "file_sha256",
            "canonical_sha256",
            "log_path",
            "log_sha256",
            "log_bytes",
            "evaluation_envelope",
            "evaluation_envelope_sha256",
        },
        "candidate shard result binding",
    )
    result_path = _relative_shard_file(
        root, result_binding["path"], "candidate shard raw result"
    )
    log_path = _relative_shard_file(
        root, result_binding["log_path"], "candidate shard evaluator log"
    )
    if (
        result_binding["path"] != f"{candidate['id']}.json"
        or result_binding["log_path"] != f"{candidate['id']}.json.comfy.log"
    ):
        raise ValueError("candidate shard result filenames are not candidate-bound")
    result, result_file_sha, _ = _load_json_file(
        result_path, f"candidate shard result {candidate['id']}"
    )
    if (
        result_file_sha != result_binding["file_sha256"]
        or krea_provenance.canonical_sha256(result)
        != result_binding["canonical_sha256"]
        or _sha256(log_path) != result_binding["log_sha256"]
        or log_path.stat().st_size != result_binding["log_bytes"]
    ):
        raise ValueError("candidate shard raw result or log changed")
    completed_public = _object(
        shard["completed_candidate"], "candidate shard completed candidate"
    )
    comfy_staging = _object(
        completed_public.get("comfy_staging"), "candidate shard Comfy staging"
    )
    envelope = _validate_result(
        result,
        candidate={**candidate, "path": staged_candidate},
        # The evaluator records its original absolute shard path.  Portable
        # relocation is proven separately by the content-bound staged manifest.
        dataset=Path(result["dataset"]),
        expected_dataset_sha256=dataset_sha256,
        expected_dataset_identity=evaluator.get("_expected_dataset_identity"),
        evaluator=evaluator,
        evaluator_script_sha=_sha256(staged_evaluator),
        log_path=log_path,
        comfy_staging=comfy_staging,
    )
    if (
        envelope != result_binding["evaluation_envelope"]
        or krea_provenance.canonical_sha256(envelope)
        != result_binding["evaluation_envelope_sha256"]
    ):
        raise ValueError("candidate shard evaluation envelope changed")
    completed = _candidate_completion_row(
        candidate=candidate,
        result_path=result_path,
        result_file_sha256=result_file_sha,
        result=result,
        log_path=log_path,
        comfy_staging=comfy_staging,
    )
    expected_public = {
        key: value for key, value in completed.items() if not key.startswith("_")
    }
    if completed_public != expected_public:
        raise ValueError("candidate shard summary differs from its raw result")
    normalized_envelope = dict(envelope)
    normalized_envelope["dataset"] = str(dataset)
    rebound.extend(
        [
            {"path": str(result_path), "sha256": result_file_sha},
            {"path": str(log_path), "sha256": result_binding["log_sha256"]},
        ]
    )
    summary = {
        "candidate_id": candidate["id"],
        "shard_file_sha256": shard_file_sha,
        "shard_sha256": shard["shard_sha256"],
        "staged_input_manifest_sha256": staged["manifest_sha256"],
        "result_file_sha256": result_file_sha,
        "result_canonical_sha256": result_binding["canonical_sha256"],
        "log_sha256": result_binding["log_sha256"],
    }
    return completed, normalized_envelope, summary, rebound


def _schema2_candidate_rows(completed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for private in completed:
        row = {key: value for key, value in private.items() if not key.startswith("_")}
        binding = row["candidate_binding"]
        fraction = binding["candidate_fraction"]
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "arm_id": row["arm_id"] if row["mode"] == "local_run_candidate" else None,
                "mode": row["mode"],
                "family_id": (
                    row["family_id"] if row["mode"] == "local_run_candidate" else None
                ),
                "candidate_sha256": row["candidate_sha256"],
                "candidate_bytes": row["candidate_bytes"],
                "execution_plan_sha256": binding.get("execution_plan_sha256"),
                "run_completion_sha256": binding.get("run_completion_sha256"),
                "step": binding["candidate_step"],
                "fraction_numerator": fraction["numerator"] if fraction is not None else None,
                "fraction_denominator": (
                    fraction["denominator"] if fraction is not None else None
                ),
                "image_exposures": (
                    binding["candidate_image_exposures"]
                    if row["mode"] == "local_run_candidate"
                    else None
                ),
                "binding_manifest_sha256": binding["binding_manifest_sha256"],
                "zero_control_manifest_sha256": (
                    binding["zero_control_manifest_sha256"]
                    if row["mode"] == "zero_lora_control"
                    else None
                ),
                "result_file": row["result_file"],
                "result_file_sha256": row["result_file_sha256"],
                "result_canonical_sha256": row["result_canonical_sha256"],
                "weighted_loss": row["weighted_loss"],
                "text_mean": row["text_mean"],
                "blank_mean": row["blank_mean"],
                "paired_rows": row["paired_rows"],
                "mechanics": row["mechanics"],
            }
        )
    return rows


def assemble_candidate_shards(
    plan: dict[str, Any],
    *,
    shard_paths: list[Path],
    output: Path,
    plan_raw: bytes | None = None,
) -> dict[str, Any]:
    """Validate one immutable shard per candidate and publish one full aggregate."""

    if plan.get("schema") != 2:
        raise ValueError("candidate-shard assembly requires a schema-2 approved plan")
    plan_raw, plan_hashes = _validated_plan_bytes(plan, plan_raw)
    dataset, dataset_sha256, candidates, evaluator = _validate_plan(plan)
    output = _absolute_lexical(output)
    if output.name == _FORBIDDEN_OUTPUT:
        raise ValueError(f"refusing production selection filename: {_FORBIDDEN_OUTPUT}")
    evidence_path = output.parent / f"{output.name}.evidence"
    evidence_temporary = output.parent / f".{output.name}.evidence.tmp"
    if any(
        os.path.lexists(path)
        for path in (output, Path(f"{output}.tmp"), evidence_path, evidence_temporary)
    ):
        raise FileExistsError(f"refusing stale aggregate path: {output}")
    if _paths_overlap(output, dataset):
        raise ValueError("aggregate output overlaps the exact-evaluation dataset")
    for candidate in candidates:
        if _paths_overlap(output, candidate["path"]):
            raise ValueError("aggregate output overlaps a candidate input")
    _reject_symlink_ancestors(output.parent, "aggregate output parent")
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(output.parent, "aggregate output parent")

    candidates_by_id = {candidate["id"]: candidate for candidate in candidates}
    observed: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    staged_bindings: list[dict[str, Any]] = []
    for raw_path in shard_paths:
        shard_path = _absolute_lexical(raw_path)
        preview, _, _ = _load_json_file(shard_path, "candidate shard manifest")
        candidate_id = _object(
            preview.get("candidate"), "candidate shard candidate"
        ).get("id")
        if candidate_id not in candidates_by_id:
            raise ValueError("candidate shard is extra or not present in the score plan")
        if candidate_id in observed:
            raise ValueError(f"duplicate candidate shard: {candidate_id}")
        completed, envelope, summary, rebound = _validate_candidate_shard(
            shard_path,
            plan_hashes=plan_hashes,
            dataset=dataset,
            dataset_sha256=dataset_sha256,
            candidate=candidates_by_id[candidate_id],
            evaluator=evaluator,
        )
        observed[candidate_id] = (completed, envelope, summary)
        staged_bindings.extend(rebound)
    if set(observed) != set(candidates_by_id):
        missing = sorted(set(candidates_by_id) - set(observed))
        extra = sorted(set(observed) - set(candidates_by_id))
        raise ValueError(
            f"candidate shard coverage is incomplete: missing={missing}, extra={extra}"
        )

    completed = [observed[candidate["id"]][0] for candidate in candidates]
    envelopes = [observed[candidate["id"]][1] for candidate in candidates]
    common_envelope = envelopes[0]
    common_envelope_sha = krea_provenance.canonical_sha256(common_envelope)
    if any(
        envelope != common_envelope
        or krea_provenance.canonical_sha256(envelope) != common_envelope_sha
        for envelope in envelopes[1:]
    ):
        raise RuntimeError("candidate shards escaped the common evaluation envelope")
    shard_rows = [observed[candidate["id"]][2] for candidate in candidates]
    staged_body = {
        "schema": 1,
        "kind": "forge-krea-complete-sharded-staged-inputs",
        "candidate_order": [candidate["id"] for candidate in candidates],
        "shards": shard_rows,
    }
    staged_input_manifest = {
        **staged_body,
        "manifest_sha256": krea_provenance.canonical_sha256(staged_body),
    }

    # Rebind every approved source and every shard byte immediately before the
    # complete-set publication.  A long campaign cannot silently absorb drift.
    for row in completed:
        if _sha256(Path(row["_candidate_path"])) != row["candidate_sha256"]:
            raise RuntimeError(f"candidate {row['candidate_id']} changed before assembly")
        if _sha256(Path(row["_provenance_path"])) != row["provenance_file_sha256"]:
            raise RuntimeError(
                f"provenance {row['candidate_id']} changed before assembly"
            )
        if _sha256(Path(row["_result_path"])) != row["result_file_sha256"]:
            raise RuntimeError(f"result {row['candidate_id']} changed before assembly")
        if _sha256(Path(row["_log_path"])) != row["_log_sha256"]:
            raise RuntimeError(f"log {row['candidate_id']} changed before assembly")
        for path_key, bound_path in row.items():
            if not path_key.startswith("_") or not path_key.endswith("_path"):
                continue
            digest_key = f"{path_key[:-5]}_sha256"
            if digest_key in row and _sha256(Path(bound_path)) != row[digest_key]:
                raise RuntimeError(
                    f"bound evidence {path_key} for {row['candidate_id']} changed"
                )
    for binding in staged_bindings:
        if _sha256(Path(binding["path"])) != binding["sha256"]:
            raise RuntimeError("candidate shard bytes changed before assembly")
    sealed_path = Path(evaluator["_sealed_plan_approval_path"])
    if _sha256(sealed_path) != evaluator["_sealed_plan_approval_sha256"]:
        raise RuntimeError("sealed plan approval changed before assembly")
    for path_key, sha_key, label in (
        ("_fixture_manifest_path", "_fixture_manifest_file_sha256", "fixture manifest"),
        ("_campaign_manifest_path", "_campaign_manifest_file_sha256", "campaign manifest"),
        ("_fixture_approval_path", "_fixture_approval_file_sha256", "fixture approval"),
        (
            "_cross_fixture_review_path",
            "_cross_fixture_review_file_sha256",
            "cross-fixture review",
        ),
    ):
        if _sha256(Path(evaluator[path_key])) != evaluator[sha_key]:
            raise RuntimeError(f"{label} changed before assembly")
    decision_context = evaluator.get("_decision_context")
    if decision_context is not None and decision_context["phase"] == "boundary":
        frozen = Path(decision_context["_frozen_discovery_decision_path"])
        if _sha256(frozen) != decision_context[
            "_frozen_discovery_decision_file_sha256"
        ]:
            raise RuntimeError("frozen discovery decision changed before assembly")

    campaign, campaign_file_sha, _ = _load_json_file(
        Path(evaluator["_campaign_manifest_path"]), "campaign manifest"
    )
    if campaign_file_sha != evaluator["_campaign_manifest_file_sha256"]:
        raise RuntimeError("campaign manifest changed before assembly")
    _validate_campaign_manifest(campaign)
    fixture, fixture_file_sha, _ = _load_json_file(
        Path(evaluator["_fixture_manifest_path"]), "fixture manifest"
    )
    if fixture_file_sha != evaluator["_fixture_manifest_file_sha256"]:
        raise RuntimeError("fixture manifest changed before assembly")
    evaluator.get("_fixture_validator", krea_fixture).validate_manifest(fixture)

    schema2_candidates = _schema2_candidate_rows(completed)
    body: dict[str, Any] = {
        "schema": 2,
        "kind": _KIND,
        "coverage": {
            "planned": len(candidates),
            "completed": len(completed),
            "complete": True,
        },
        "direction": "min",
        "plan": {
            **plan_hashes,
            "approved_payload_sha256": evaluator["_plan_payload_sha256"],
        },
        "sealed_plan_approval_sha256": evaluator["_sealed_plan_approval_sha256"],
        "sealed_plan_approval": evaluator["_sealed_plan_approval"],
        "batch_runner_sha256": evaluator["_batch_runner_sha256"],
        "staged_input_manifest": staged_input_manifest,
        "common_training_envelope": evaluator["_common_training_envelope"],
        "common_training_envelope_sha256": evaluator[
            "_common_training_envelope_sha256"
        ],
        "evaluator_script_sha256": _sha256(
            Path(__file__).with_name("evaluate_krea_local.py").resolve(strict=True)
        ),
        "evaluation_envelope": common_envelope,
        "evaluation_envelope_sha256": common_envelope_sha,
        "candidates": schema2_candidates,
        "campaign_manifest_sha256": evaluator["_campaign_manifest_file_sha256"],
        "fixture_manifest_sha256": evaluator["_fixture_manifest_file_sha256"],
        "fixture_approval_sha256": evaluator["_fixture_approval_file_sha256"],
        "fixture_contract": {
            "fixture_manifest_identity_sha256": fixture["manifest_sha256"],
            "training_pair_count": len(fixture["training_rows"]),
            "evaluation_row_count": len(fixture["evaluation_rows"]),
            "training_dataset_sha256": fixture["training_dataset_identity"]["sha256"],
            "evaluation_dataset_sha256": fixture["evaluation_dataset_identity"]["sha256"],
            "cross_fixture_review_sha256": evaluator[
                "_cross_fixture_review_file_sha256"
            ],
        },
        "campaign": {
            "manifest_sha256": evaluator["_campaign_manifest_sha256"],
            "file_sha256": evaluator["_campaign_manifest_file_sha256"],
            "fixture_manifest_sha256": campaign["fixture_manifest_sha256"],
            "discovery_plan_sha256": campaign["discovery_plan_sha256"],
            "zero_control_manifest_sha256": campaign["zero_control_manifest_sha256"],
            "decision_contract": campaign["decision_contract"],
            "confirmation_contract": campaign["confirmation_contract"],
            "runs": campaign["runs"],
        },
        "fixture": {
            "manifest_sha256": fixture["manifest_sha256"],
            "file_sha256": evaluator["_fixture_manifest_file_sha256"],
            "concept_id": fixture["concept_id"],
            "experimental_role": fixture["experimental_role"],
            "evaluation_dataset_sha256": dataset_sha256,
        },
        "training_run_envelopes": _schema2_training_run_envelopes(
            campaign=campaign,
            candidates=schema2_candidates,
            envelopes=evaluator["_training_run_envelopes"],
            decision_context=evaluator["_decision_context"],
        ),
    }
    body["decision_evidence"] = _publish_decision_evidence_bundle(
        output=output,
        plan=plan,
        plan_raw=plan_raw,
        approval_path=sealed_path,
        completed=completed,
    )
    aggregate = {**body, "aggregate_sha256": krea_provenance.canonical_sha256(body)}
    _publish_exclusive(output, aggregate)
    return aggregate


def main() -> int:
    args = _parse()
    plan, _, plan_raw = _load_json_file(args.plan, "plan")
    if args.mode == "batch":
        if args.results_dir is None or args.candidate_id is not None or args.shard:
            raise ValueError(
                "batch mode requires --results-dir and forbids shard-only arguments"
            )
        result = run_batch(
            plan,
            results_dir=args.results_dir,
            output=args.output,
            plan_raw=plan_raw,
        )
    elif args.mode == "candidate-shard":
        if args.results_dir is None or args.candidate_id is None or args.shard:
            raise ValueError(
                "candidate-shard mode requires --results-dir and --candidate-id"
            )
        result = run_candidate_shard(
            plan,
            candidate_id=args.candidate_id,
            results_dir=args.results_dir,
            output=args.output,
            plan_raw=plan_raw,
        )
    else:
        if args.results_dir is not None or args.candidate_id is not None or not args.shard:
            raise ValueError(
                "assemble-shards mode requires one or more --shard arguments only"
            )
        result = assemble_candidate_shards(
            plan,
            shard_paths=args.shard,
            output=args.output,
            plan_raw=plan_raw,
        )
    print(krea_provenance.canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
