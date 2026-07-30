#!/usr/bin/env python3
"""Fail-closed Week-5 Krea discovery and confirmation decisions.

This module is deliberately incapable of changing the production trainer or
publishing Forge's selector file.  Discovery consumes exhaustive, sealed exact
score curves and may only freeze finalists/checkpoint rules or request Seed B.
Confirmation consumes those frozen rules plus independently sealed C1-C4 and
boundary evidence and may only report PASS, FAIL, or no-go for human review.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence

try:
    from . import batch_evaluate_krea as krea_batch
    from . import krea_c1c4_amendment
    from . import krea_delegated_review_contract
    from . import krea_fixture
    from . import krea_provenance
    from . import krea_discovery_authorization
except ImportError:  # pragma: no cover - direct script execution.
    import batch_evaluate_krea as krea_batch  # type: ignore[no-redef]
    import krea_c1c4_amendment  # type: ignore[no-redef]
    import krea_delegated_review_contract  # type: ignore[no-redef]
    import krea_fixture  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_discovery_authorization  # type: ignore[no-redef]


_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_FORBIDDEN_OUTPUTS = frozenset(
    {
        "forge_holdout_scores.json",
        "last.safetensors",
        "config.yaml",
        "training_repo.py",
    }
)
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
_DISCOVERY_FIXTURES = ("D1", "D2")
_CONFIRMATION_FIXTURES = ("C1", "C2", "C3", "C4")
_PUBLIC_FAMILIES = ("K2", "K3", "K4")
_CONTROL_FAMILY = "K0"
_DISCOVERY_STATUS = "draft_blocked_pre_gpu"
_DISCOVERY_GPU_BLOCKERS = [
    (
        "create-only Stage-1 discovery execution authorization bound to the "
        "admitted D1/D2 envelope is external to this immutable freeze"
    ),
    (
        "six fixture-scoped D1/D2 by A/B/C throughput profiles and their "
        "post-timing profile index do not exist before authorized timing probes"
    ),
]
_PROFILE_INDEX_SENTINEL = "deferred_to_fixture_scoped_profile_index"
_DISCOVERY_TIE = Decimal("0.01")
_FIELD_CAP = Decimal("0.01")
_CONCEPT_CAP = Decimal("0.03")
_CONFIDENCE = Decimal("0.95")
_BOOTSTRAP_RESAMPLES = 10_000
_BOOTSTRAP_SEED = 42_565_431
_C1C4_PUBLIC_RECORD = "SN56-project/SN56-WEEK5-C1C4-SEALED-COMMITMENT-2026-07-27.md"
_C1C4_PUBLIC_RECORD_SHA256 = (
    "f907c40e362378c1b82e7455d96ffd8bd876696f25cef21705e14bbba2d4ffc0"
)
_C1C4_COMMITMENT_SHA256 = (
    "0a12c416bcef48805132e80f9de65d0d248ef4415d617715d5736c189a379dbc"
)
_C1C4_PRE_AMENDMENT_PLAN_FILE_SHA256 = (
    "6365f150352de1497fbf32edc8ea07bc2859c3096c95796cff708c89382aee6a"
)
_C1C4_PRE_AMENDMENT_PLAN_COMMIT = "1bd7477717ab8d96d208d9fe265f071f08e47e73"
_C1C4_SHAPE_AMENDMENT_PATH = (
    "ops/calibration/week5/krea-c1c4-shape-contract-amendment.json"
)
_C1C4_SHAPE_AMENDMENT_FILE_SHA256 = (
    "5f1b02ab78d6f82da6587c533af19e61ead5aa2e821ce268fa94c9bd0ad9587e"
)
_C1C4_SHAPE_AMENDMENT_SHA256 = (
    "367fbcd46827e49efa4d14bf50d1533d85f56d5354a3233d4ea41a81779ef61c"
)
_CONFIRMATION_SHAPE_CONTRACT = {
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
_PRE_AMENDMENT_CONFIRMATION_SHAPE_CONTRACT = {
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
_C1C4_MANIFEST_FILE_SHA256S = {
    "C1": "ed287150fd4d189b3a0964d87c5fc50de11851ab372dabe30da9d9f87fdc450e",
    "C2": "902a4a6716a9210694f3f441d54b4def19e9bc64d0a49be4cb832ccff8605083",
    "C3": "74ebbfaf91b156741d34b10ba2d37600076844c010ea6ea83d4af36a386eda09",
    "C4": "7a3fb670bed78d851cf8c066696b61ccc79d78dffd1ecb633520493772210872",
}
_C1C4_AMENDMENT_AUTHORSHIP_ORDER = (
    "authored after the public commitment and the independent reviewer finding; "
    "this amendment was not part of the original fixture seal"
)
_C1C4_AMENDMENT_CLAIM_LIMIT = (
    "Corrects only the public per-fixture concept classes and train/evaluation "
    "counts; fixture bytes, manifest digests, aggregate commitment, identities, "
    "and custody are unchanged."
)
_DISCOVERY_FIXTURE_COUNTS = {
    "D1": (18, 24, 24),
    "D2": (36, 48, 40),
}
_BOUNDARY_FIXTURE_COUNTS = {
    "small": (18, 24, 24),
    "large": (36, 48, 40),
}
_LOSS_KEYS = frozenset({"text_guided_loss", "blank_prompt_loss"})
_OUTPUT_NAME = re.compile(
    r"krea-(?:discovery|confirmation)-decision(?:-[A-Za-z0-9_.-]+)?\.json"
)
_DISCOVERY_CAMPAIGN_CONTRACT = {
    "paired_rows_required": True,
    "discovery_tie_band": 0.01,
    "cluster_unit": "task/concept",
    "bootstrap": "cluster-bootstrap by task/concept",
    "bootstrap_confidence": 0.95,
    "bootstrap_resamples": _BOOTSTRAP_RESAMPLES,
    "bootstrap_seed": _BOOTSTRAP_SEED,
    "material_rank_reversal_definition": (
        "any non-control pair switches order across D1/D2 with >0.01 "
        "relative-improvement separation in both directions"
    ),
    "checkpoint_tie_breaker": (
        "earliest actual step among candidates within 0.01 of best"
    ),
}
_CONFIRMATION_CAMPAIGN_CONTRACT = {
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


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = keys - set(value)
    extra = set(value) - keys
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def _require_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = keys - set(value)
    if missing:
        raise ValueError(f"{label} missing required keys: {sorted(missing)}")


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} is invalid")
    return value


def _named_human(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must name a human")
    identity = " ".join(value.split())
    if identity.casefold() in _ROLE_LABELS:
        raise ValueError(f"{label} is a role label, not a named human")
    words = identity.split()
    if len(words) < 2 or any(
        not any(character.isalpha() for character in word) for word in words
    ):
        raise ValueError(f"{label} must contain a named human identity")
    return identity


def _timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be an RFC3339 timestamp")
    normalized = value.strip()
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a UTC offset")
    return normalized


def _timestamp_value(value: Any, label: str) -> datetime:
    normalized = _timestamp(value, label)
    return datetime.fromisoformat(normalized.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _decimal(value: Any, label: str, *, minimum: Decimal, maximum: Decimal) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError(f"{label} must be numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not result.is_finite() or not minimum <= result <= maximum:
        raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _seed(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise ValueError(f"{label} must be an integer in [0, 2**32)")
    return value


def _safe_file(path: Path | str, label: str) -> Path:
    value = Path(os.path.abspath(os.path.expanduser(path)))
    current = value
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    if value.is_symlink() or not value.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {value}")
    return value


def _load_canonical(path: Path | str, label: str) -> tuple[dict[str, Any], str]:
    path = _safe_file(path, label)
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
    if raw != krea_provenance.canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline")
    return value, hashlib.sha256(raw).hexdigest()


def _load_json_evidence(
    path: Path | str, label: str
) -> tuple[dict[str, Any], str, bytes]:
    """Load a stable JSON evidence file without rewriting its original bytes."""

    path = _safe_file(path, label)
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
    return value, hashlib.sha256(raw).hexdigest(), raw


def _portable_relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a portable relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a portable relative path")
    return path


def _archive_member(root: Path, relative: Any, label: str) -> Path:
    member = _portable_relative_path(relative, label)
    root = root.resolve(strict=True)
    path = _safe_file(root / member, label)
    try:
        path.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its evidence archive") from exc
    if path.stat().st_nlink != 1:
        raise ValueError(f"{label} has an unexpected hardlink")
    return path


def _manifest_file_entry(value: Any, label: str) -> dict[str, str]:
    row = _object(value, label)
    _exact(row, {"path", "file_sha256", "canonical_sha256"}, label)
    _portable_relative_path(row["path"], f"{label}.path")
    return {
        "path": row["path"],
        "file_sha256": _digest(row["file_sha256"], f"{label}.file_sha256"),
        "canonical_sha256": _digest(
            row["canonical_sha256"], f"{label}.canonical_sha256"
        ),
    }


def _validate_decision_evidence_binding(value: Any) -> dict[str, Any]:
    binding = _object(value, "decision evidence binding")
    _exact(
        binding,
        {
            "archive_path",
            "manifest_path",
            "manifest_file_sha256",
            "manifest_sha256",
            "score_plan",
            "score_plan_approval",
            "evaluator_results",
        },
        "decision evidence binding",
    )
    archive_path = _portable_relative_path(
        binding["archive_path"], "decision evidence archive_path"
    )
    if len(archive_path.parts) != 1:
        raise ValueError("decision evidence archive_path must be beside the aggregate")
    manifest_path = _portable_relative_path(
        binding["manifest_path"], "decision evidence manifest_path"
    )
    plan = _manifest_file_entry(binding["score_plan"], "decision evidence plan")
    approval = _manifest_file_entry(
        binding["score_plan_approval"], "decision evidence approval"
    )
    raw_results = binding["evaluator_results"]
    if not isinstance(raw_results, list) or not raw_results:
        raise ValueError("decision evidence evaluator_results must be non-empty")
    results = []
    previous = None
    for index, raw in enumerate(raw_results):
        label = f"decision evidence evaluator_results[{index}]"
        row = _object(raw, label)
        _exact(
            row,
            {"candidate_id", "path", "file_sha256", "canonical_sha256"},
            label,
        )
        candidate_id = _identifier(row["candidate_id"], f"{label}.candidate_id")
        if previous is not None and candidate_id <= previous:
            raise ValueError("decision evidence results must be unique and sorted")
        previous = candidate_id
        entry = _manifest_file_entry(
            {key: row[key] for key in ("path", "file_sha256", "canonical_sha256")},
            label,
        )
        results.append({"candidate_id": candidate_id, **entry})
    return {
        "archive_path": archive_path.as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "manifest_file_sha256": _digest(
            binding["manifest_file_sha256"],
            "decision evidence manifest_file_sha256",
        ),
        "manifest_sha256": _digest(
            binding["manifest_sha256"], "decision evidence manifest_sha256"
        ),
        "score_plan": plan,
        "score_plan_approval": approval,
        "evaluator_results": results,
    }


def _binding(value: Any, label: str) -> tuple[Path, dict[str, Any], str]:
    binding = _object(value, label)
    _exact(binding, {"path", "sha256"}, label)
    expected = _digest(binding["sha256"], f"{label}.sha256")
    path = _safe_file(binding["path"], label)
    document, actual = _load_canonical(path, label)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path, document, actual


def _actor(value: Any, label: str) -> dict[str, Any]:
    return krea_fixture._agent_actor(value, label)


def _require_distinct_actors(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if (
        left["actor_id"] == right["actor_id"]
        or left["review_instance_id"] == right["review_instance_id"]
    ):
        raise ValueError(f"{label} must use distinct actor and review-instance ids")


def _decision_authorization_context(value: Any) -> dict[str, Any]:
    """Reopen the Stage-1 authority and its owner-ratified admission bundle."""

    path, authorization, file_sha = _binding(value, "discovery execution authorization")
    krea_discovery_authorization.validate(authorization)
    if "discovery_decision_evaluation" not in authorization.get(
        "authorized_actions", []
    ):
        raise ValueError(
            "discovery authorization does not permit discovery_decision_evaluation"
        )
    (
        _,
        admission,
        _,
        ratification,
        _,
    ) = krea_discovery_authorization._load_admission_binding(
        authorization["fixture_admission_envelope"]
    )
    governance = _object(admission.get("governance"), "admission governance")
    custodian_binding = _object(
        governance.get("sealed_custodian_actor"), "sealed custodian binding"
    )
    custodian = _actor(custodian_binding.get("actor"), "sealed confirmation custodian")
    if custodian["role"] != "sealed_confirmation_custodian":
        raise ValueError("admission custodian has the wrong role")
    owner = krea_fixture.named_human(
        authorization.get("accountable_owner_identity"),
        "authorization accountable owner",
    )
    ratification_sha = _digest(
        authorization["fixture_admission_envelope"].get("owner_ratification_sha256"),
        "authorization owner ratification",
    )
    if (
        admission.get("accountable_owner_identity") != owner
        or ratification.get("ratification_sha256") != ratification_sha
    ):
        raise ValueError("decision authority differs from its admitted owner")
    return {
        "binding": {"path": str(path), "sha256": file_sha},
        "authorization": authorization,
        "owner": owner,
        "owner_ratification_sha256": ratification_sha,
        "custodian_actor": custodian,
    }


def _validate_agent_governance(
    value: Mapping[str, Any],
    *,
    actor_field: str,
    delegated_actor_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = _decision_authorization_context(
        value["discovery_execution_authorization"]
    )
    actor = krea_delegated_review_contract.validate_actor(
        delegated_actor_name, value[actor_field]
    )
    krea_delegated_review_contract.validate_binding(value["delegated_review_contract"])
    if (
        value["accountable_owner_identity"] != context["owner"]
        or value["owner_ratification_sha256"] != context["owner_ratification_sha256"]
        or value["fixture_admission_envelope"]
        != context["authorization"]["fixture_admission_envelope"]
        or value["agent_review_is_not_human_review"] is not True
    ):
        raise ValueError("agent governance differs from owner-ratified authority")
    _require_distinct_actors(
        actor,
        context["custodian_actor"],
        label=f"{actor_field} and sealed confirmation custodian",
    )
    return actor, context


def _validate_bootstrap(value: Any) -> dict[str, Any]:
    value = _object(value, "bootstrap policy")
    _exact(
        value,
        {"method", "cluster_unit", "confidence", "resamples", "seed"},
        "bootstrap policy",
    )
    expected = {
        "method": "paired_cluster_bootstrap",
        "cluster_unit": "task/concept",
        "confidence": float(_CONFIDENCE),
        "resamples": _BOOTSTRAP_RESAMPLES,
        "seed": _BOOTSTRAP_SEED,
    }
    if value != expected:
        raise ValueError(f"bootstrap policy must equal the frozen policy: {expected}")
    return dict(value)


def validate_confirmation_shape_amendment(value: Any) -> dict[str, Any]:
    """Validate the post-publication shape correction without resealing C1-C4."""

    return krea_c1c4_amendment.validate_amendment(value)


def _validate_discovery_plan(value: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "kind",
        "status",
        "model",
        "model_type",
        "training_seed_a",
        "training_seed_b_contingency",
        "gpu_execution_authorized",
        "gpu_blockers",
        "discovery_tasks",
        "arms",
        "budget_contract",
        "candidate_contract",
        "decision_contract",
        "confirmation_fixture_commitment",
        "confirmation_contract",
        "prohibited",
    }
    _require_keys(value, required, "discovery plan")
    if (
        value["schema"] != 2
        or value["kind"] != "sn56-week5-krea-discovery-freeze"
        or value["model"] != "krea/Krea-2-Raw"
        or value["model_type"] != "krea2"
    ):
        raise ValueError("discovery plan is not the Week-5 Krea freeze")
    if value["gpu_execution_authorized"] is not False:
        raise ValueError(
            "immutable discovery freeze must never self-authorize GPU work"
        )
    if value["status"] != _DISCOVERY_STATUS:
        raise ValueError(
            "discovery plan status is not the frozen non-authorizing state"
        )
    if value["gpu_blockers"] != _DISCOVERY_GPU_BLOCKERS:
        raise ValueError("discovery plan blockers differ from the frozen sentinels")
    seed_a = _seed(value["training_seed_a"], "training_seed_a")
    seed_b = _seed(value["training_seed_b_contingency"], "training_seed_b")
    if seed_a == seed_b:
        raise ValueError("Seed A and Seed B must differ")

    tasks = _object(value["discovery_tasks"], "discovery_tasks")
    if set(tasks) != set(_DISCOVERY_FIXTURES):
        raise ValueError("discovery tasks must be exactly D1 and D2")
    for fixture_id in _DISCOVERY_FIXTURES:
        task = _object(tasks[fixture_id], f"discovery task {fixture_id}")
        identity = _digest(
            task.get("identity"), f"{fixture_id} stable candidate identity"
        )
        candidate_sha = _digest(
            task.get("fixture_split_manifest_sha256"),
            f"{fixture_id} candidate manifest",
        )
        expected_shape = {"D1": ([18, 18], 24), "D2": ([36, 36], 40)}[fixture_id]
        if (
            identity != candidate_sha
            or task.get("required_training_pair_range") != expected_shape[0]
            or task.get("required_evaluation_rows") != expected_shape[1]
            or task.get("identity_semantics")
            != (
                "pre-governance candidate_manifest_sha256; final governance-bearing "
                "manifest and approval are bound only through the admitted envelope "
                "and external execution authorization"
            )
        ):
            raise ValueError(
                f"discovery task {fixture_id} shape/identity is not frozen"
            )

    arms = value["arms"]
    if not isinstance(arms, list) or not arms:
        raise ValueError("discovery plan arms are absent")
    arm_ids = []
    for raw in arms:
        arm = _object(raw, "discovery arm")
        arm_ids.append(_identifier(arm.get("id"), "discovery arm id"))
    if arm_ids != sorted(set(arm_ids)):
        raise ValueError("discovery arms must be unique and sorted")
    if _CONTROL_FAMILY not in arm_ids or not set(_PUBLIC_FAMILIES).issubset(arm_ids):
        raise ValueError("discovery plan lacks K0 or the K2-K4 public families")

    profiles = _object(
        _object(value["budget_contract"], "budget_contract").get(
            "throughput_profiles_by_equivalence_class"
        ),
        "throughput profiles",
    )
    for name, digest in profiles.items():
        _identifier(name, "throughput equivalence class")
        if digest != _PROFILE_INDEX_SENTINEL:
            raise ValueError(
                f"throughput profile {name} must remain the deferred index sentinel"
            )
    profile_index_contract = _object(
        _object(value["budget_contract"], "budget_contract").get(
            "profile_index_contract"
        ),
        "profile index contract",
    )
    if profile_index_contract != {
        "fixtures": ["D1", "D2"],
        "equivalence_classes": sorted(profiles),
        "required_profile_count": 6,
        "cross_fixture_profile_reuse_forbidden": True,
        "post_timing_only": True,
        "gpu_execution_authorized": False,
    }:
        raise ValueError("six-cell deferred profile-index contract is invalid")

    candidate = _object(value["candidate_contract"], "candidate_contract")
    if candidate.get("exact_score_during_discovery") != (
        "zero-LoRA and every valid current-attempt candidate, offline after training"
    ):
        raise ValueError("discovery plan does not require exhaustive offline curves")
    report_targets = candidate.get("report_targets")
    if report_targets != [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        raise ValueError("discovery report targets are not frozen")

    decision = _object(value["decision_contract"], "decision_contract")
    required_decision = {
        "relative_loss_formula",
        "paired_rows_required",
        "discovery_tie_band",
        "cluster_unit",
        "bootstrap",
        "bootstrap_confidence",
        "bootstrap_resamples",
        "bootstrap_seed",
        "advance_if_D1_D2_agree",
        "advance_if_D1_D2_disagree",
        "seed_b_trigger",
        "material_rank_reversal_definition",
        "checkpoint_tie_breaker",
        "maximum_finalists",
    }
    _require_keys(decision, required_decision, "decision_contract")
    expected_decision = {
        "relative_loss_formula": "(L_control-L_candidate)/L_control",
        "paired_rows_required": True,
        "discovery_tie_band": 0.01,
        "cluster_unit": "task/concept",
        "bootstrap": "cluster-bootstrap by task/concept",
        "bootstrap_confidence": 0.95,
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 42_565_431,
        "advance_if_D1_D2_agree": [
            "shared winner",
            "next-lowest minimax-regret non-control",
            "K0",
        ],
        "advance_if_D1_D2_disagree": [
            "D1 winner",
            "D2 winner",
            "lowest-minimax-regret remaining non-control",
            "K0",
        ],
        "seed_b_trigger": (
            "three or more non-controls inside tie band or material rank reversal"
        ),
        "material_rank_reversal_definition": (
            "any non-control pair switches order across D1/D2 with >0.01 "
            "relative-improvement separation in both directions"
        ),
        "checkpoint_tie_breaker": (
            "earliest actual step among candidates within 0.01 of best"
        ),
        "maximum_finalists": 4,
    }
    mismatches = {
        key: {"expected": expected, "actual": decision.get(key)}
        for key, expected in expected_decision.items()
        if decision.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"decision contract differs from frozen protocol: {mismatches}"
        )

    commitment = _object(
        value["confirmation_fixture_commitment"],
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
    amendment = _object(
        commitment["shape_contract_amendment"], "shape_contract_amendment"
    )
    expected_commitment = {
        "state": "published_external_agent_custody_pending_owner_ratification",
        "public_record": _C1C4_PUBLIC_RECORD,
        "public_record_sha256": _C1C4_PUBLIC_RECORD_SHA256,
        "commitment_sha256": _C1C4_COMMITMENT_SHA256,
        "shape_contract_amendment": {
            "path": _C1C4_SHAPE_AMENDMENT_PATH,
            "file_sha256": _C1C4_SHAPE_AMENDMENT_FILE_SHA256,
            "amendment_sha256": _C1C4_SHAPE_AMENDMENT_SHA256,
        },
        "implementation_read_sealed_contents": False,
    }
    if amendment != expected_commitment["shape_contract_amendment"]:
        raise ValueError("confirmation shape amendment binding is not frozen")
    if commitment != expected_commitment:
        raise ValueError("confirmation fixture commitment differs from publication")
    confirmation = _object(value["confirmation_contract"], "confirmation_contract")
    required_confirmation = {
        "fixtures",
        "identities",
        "sealed_by_independent_reviewer_before_discovery_unblinding",
        "paired_predeclared_seed_per_concept",
        "second_seed_repeats",
        "fixture_shape_contract",
        "field_parity_noninferiority_cap",
        "concept_regression_cap",
        "minimum_point_estimate_wins_or_ties",
        "point_win_or_tie_cap",
        "strongest_public_reference_rule",
        "boundary_hours",
        "dataset_boundaries",
        "boundary_gate",
    }
    _require_keys(confirmation, required_confirmation, "confirmation_contract")
    expected_confirmation = {
        "fixtures": list(_CONFIRMATION_FIXTURES),
        "sealed_by_independent_reviewer_before_discovery_unblinding": True,
        "paired_predeclared_seed_per_concept": True,
        "second_seed_repeats": 2,
        "fixture_shape_contract": _CONFIRMATION_SHAPE_CONTRACT,
        "field_parity_noninferiority_cap": 0.01,
        "concept_regression_cap": 0.03,
        "minimum_point_estimate_wins_or_ties": 3,
        "point_win_or_tie_cap": 0.01,
        "strongest_public_reference_rule": (
            "minimum loss among exhaustive approved K2-K4 local public-family "
            "reproductions for the same "
            "concept and seed"
        ),
        "boundary_hours": [0.5, 0.75, 1.0],
        "dataset_boundaries": ["small", "large"],
        "boundary_gate": "mechanics_only_natural_completion_upload_ready_clean",
    }
    mismatches = {
        key: {"expected": expected, "actual": confirmation.get(key)}
        for key, expected in expected_confirmation.items()
        if confirmation.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"confirmation contract differs from frozen protocol: {mismatches}"
        )
    identities = _object(confirmation["identities"], "confirmation identities")
    if set(identities) != set(_CONFIRMATION_FIXTURES):
        raise ValueError("confirmation identities must commit exactly C1-C4")
    for fixture_id, digest in identities.items():
        _digest(digest, f"confirmation identity {fixture_id}")
    # The three literal binding strings above are not evidence that the public
    # artifact still exists.  Every policy load reopens the repository-local
    # amendment, verifies its exact bytes and self-digest, and fails closed on
    # absence, corruption, symlink substitution, or drift.
    krea_c1c4_amendment.validate_bound_plan_amendment(value)
    return {
        "document": value,
        "arm_ids": arm_ids,
        "seed_a": seed_a,
        "seed_b": seed_b,
        "report_targets": [Decimal(str(item)) for item in report_targets],
        "confirmation_identities": identities,
        "confirmation_shape_contract": _CONFIRMATION_SHAPE_CONTRACT,
        "protocol_sha256": _discovery_protocol_sha(value),
    }


def seal_discovery_execution_authorization(payload: dict[str, Any]) -> dict[str, Any]:
    """Seal readiness separately from the immutable non-authorizing freeze."""

    return krea_discovery_authorization.seal(payload)


def validate_discovery_execution_authorization(
    value: dict[str, Any],
) -> dict[str, Any]:
    return krea_discovery_authorization.validate(value)


def _discovery_protocol_sha(value: Mapping[str, Any]) -> str:
    """Bind the protocol without creating a plan/fixture-seal hash cycle."""

    confirmation = dict(_object(value["confirmation_contract"], "confirmation"))
    confirmation["identities"] = "<independently-sealed-commitments>"
    payload = {
        "schema": value["schema"],
        "kind": value["kind"],
        "model": value["model"],
        "model_type": value["model_type"],
        "training_seed_a": value["training_seed_a"],
        "training_seed_b_contingency": value["training_seed_b_contingency"],
        "candidate_contract": value["candidate_contract"],
        "decision_contract": value["decision_contract"],
        "confirmation_fixture_commitment": value["confirmation_fixture_commitment"],
        "confirmation_contract": confirmation,
        "prohibited": value["prohibited"],
    }
    return krea_provenance.canonical_sha256(payload)


def seal_confirmation_fixture_commitments(payload: dict[str, Any]) -> dict[str, Any]:
    """Seal C1-C4 commitments before D1/D2 results are revealed."""

    if "seal_sha256" in payload:
        raise ValueError("unsealed fixture commitments contain seal_sha256")
    record = {
        **payload,
        "seal_sha256": krea_provenance.canonical_sha256(payload),
    }
    validate_confirmation_fixture_commitments(record)
    return record


def validate_confirmation_fixture_commitments(value: dict[str, Any]) -> dict[str, Any]:
    value = _object(value, "confirmation fixture commitments")
    if value.get("schema") == 2:
        return _validate_agent_confirmation_fixture_commitments(value)
    _exact(
        value,
        {
            "schema",
            "kind",
            "discovery_protocol_sha256",
            "sealed_at_utc",
            "reviewer_identity",
            "sealed_before_discovery_unblinding",
            "cross_fixture_review_sha256",
            "fixtures",
            "seal_sha256",
        },
        "confirmation fixture commitments",
    )
    body = {key: item for key, item in value.items() if key != "seal_sha256"}
    if (
        value["schema"] != 1
        or value["kind"] != "forge-krea-confirmation-fixture-commitments"
        or value["sealed_before_discovery_unblinding"] is not True
        or value["seal_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("confirmation fixture commitment identity is invalid")
    _digest(value["discovery_protocol_sha256"], "fixture seal discovery protocol")
    _digest(value["cross_fixture_review_sha256"], "cross-fixture review")
    _timestamp(value["sealed_at_utc"], "fixture sealed_at_utc")
    _named_human(value["reviewer_identity"], "fixture seal reviewer")
    fixtures = value["fixtures"]
    if not isinstance(fixtures, list) or len(fixtures) != 4:
        raise ValueError("confirmation fixture seal must contain C1-C4")
    normalized = []
    for raw in fixtures:
        row = _object(raw, "confirmation fixture commitment")
        _exact(
            row,
            {
                "fixture_id",
                "identity_commitment_sha256",
                "fixture_manifest_sha256",
                "fixture_approval_sha256",
            },
            "confirmation fixture commitment",
        )
        fixture_id = _identifier(row["fixture_id"], "confirmation fixture id")
        for key in (
            "identity_commitment_sha256",
            "fixture_manifest_sha256",
            "fixture_approval_sha256",
        ):
            _digest(row[key], f"{fixture_id}.{key}")
        normalized.append(dict(row))
    if [row["fixture_id"] for row in normalized] != list(_CONFIRMATION_FIXTURES):
        raise ValueError("confirmation fixtures must be ordered C1, C2, C3, C4")
    return value


def _validate_agent_confirmation_fixture_commitments(
    value: dict[str, Any],
) -> dict[str, Any]:
    _exact(
        value,
        {
            "schema",
            "kind",
            "discovery_protocol_sha256",
            "sealed_at_utc",
            "technical_custodian_actor",
            "accountable_owner_identity",
            "owner_ratification_sha256",
            "discovery_execution_authorization",
            "agent_review_is_not_human_review",
            "sealed_before_discovery_unblinding",
            "cross_fixture_review_sha256",
            "fixtures",
            "seal_sha256",
        },
        "agent confirmation fixture commitments",
    )
    body = {key: item for key, item in value.items() if key != "seal_sha256"}
    if (
        value["schema"] != 2
        or value["kind"] != "forge-krea-agent-confirmation-fixture-commitments"
        or value["sealed_before_discovery_unblinding"] is not True
        or value["agent_review_is_not_human_review"] is not True
        or value["seal_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("agent confirmation fixture commitment identity is invalid")
    _digest(value["discovery_protocol_sha256"], "fixture seal discovery protocol")
    _digest(value["cross_fixture_review_sha256"], "cross-fixture review")
    sealed_at = _timestamp_value(value["sealed_at_utc"], "fixture sealed_at_utc")
    context = _decision_authorization_context(
        value["discovery_execution_authorization"]
    )
    custodian = _actor(
        value["technical_custodian_actor"], "fixture seal technical custodian"
    )
    if (
        custodian != context["custodian_actor"]
        or value["accountable_owner_identity"] != context["owner"]
        or value["owner_ratification_sha256"] != context["owner_ratification_sha256"]
    ):
        raise ValueError("fixture seal differs from owner-ratified custodian authority")
    authorized_at = _timestamp_value(
        context["authorization"]["authorized_at_utc"], "authorized_at_utc"
    )
    if sealed_at <= authorized_at:
        raise ValueError("fixture commitments must be sealed after authorization")
    fixtures = value["fixtures"]
    if not isinstance(fixtures, list) or len(fixtures) != 4:
        raise ValueError("confirmation fixture seal must contain C1-C4")
    normalized = []
    for raw in fixtures:
        row = _object(raw, "confirmation fixture commitment")
        _exact(
            row,
            {
                "fixture_id",
                "identity_commitment_sha256",
                "fixture_manifest_sha256",
                "fixture_approval_sha256",
            },
            "confirmation fixture commitment",
        )
        fixture_id = _identifier(row["fixture_id"], "confirmation fixture id")
        for key in (
            "identity_commitment_sha256",
            "fixture_manifest_sha256",
            "fixture_approval_sha256",
        ):
            _digest(row[key], f"{fixture_id}.{key}")
        normalized.append(dict(row))
    if [row["fixture_id"] for row in normalized] != list(_CONFIRMATION_FIXTURES):
        raise ValueError("confirmation fixtures must be ordered C1, C2, C3, C4")
    return value


def _bound_plan_and_seal(
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _, plan, _ = _binding(policy["discovery_plan"], "discovery plan")
    plan_state = _validate_discovery_plan(plan)
    _, authorization, _ = _binding(
        policy["discovery_execution_authorization"],
        "discovery execution authorization",
    )
    validate_discovery_execution_authorization(authorization)
    if (
        authorization["discovery_plan"]["file_sha256"]
        != policy["discovery_plan"]["sha256"]
    ):
        raise ValueError("discovery authorization binds another immutable freeze")
    _, seal, _ = _binding(
        policy["confirmation_fixture_seal"], "confirmation fixture seal"
    )
    validate_confirmation_fixture_commitments(seal)
    if seal["discovery_protocol_sha256"] != plan_state["protocol_sha256"]:
        raise ValueError("confirmation fixture seal is bound to another protocol")
    committed = {
        row["fixture_id"]: row["identity_commitment_sha256"] for row in seal["fixtures"]
    }
    if committed != plan_state["confirmation_identities"]:
        raise ValueError("discovery plan C1-C4 commitments differ from reviewer seal")
    return plan_state, plan, seal


def _validate_score_batch(value: Any, *, confirmation: bool) -> dict[str, Any]:
    row = _object(value, "score batch")
    base_keys = {
        "batch_id",
        "phase",
        "fixture_id",
        "seed_role",
        "seed",
        "hours",
        "dataset_boundary",
        "plan_canonical_sha256",
        "sealed_plan_approval_sha256",
        "campaign_manifest_sha256",
        "fixture_manifest_sha256",
        "fixture_approval_sha256",
    }
    _exact(row, base_keys, "score batch")
    normalized = {
        "batch_id": _identifier(row["batch_id"], "batch_id"),
        "phase": _identifier(row["phase"], "batch phase"),
        "fixture_id": _identifier(row["fixture_id"], "fixture_id"),
        "seed_role": _identifier(row["seed_role"], "seed_role"),
        "seed": _seed(row["seed"], "batch seed"),
        "hours": (
            None
            if row["hours"] is None
            else float(
                _decimal(
                    row["hours"],
                    "batch hours",
                    minimum=Decimal("0.01"),
                    maximum=Decimal("24"),
                )
            )
        ),
        "dataset_boundary": row["dataset_boundary"],
        "plan_canonical_sha256": _digest(row["plan_canonical_sha256"], "score plan"),
        "sealed_plan_approval_sha256": _digest(
            row["sealed_plan_approval_sha256"], "score-plan approval"
        ),
        "campaign_manifest_sha256": _digest(
            row["campaign_manifest_sha256"], "campaign manifest"
        ),
        "fixture_manifest_sha256": _digest(
            row["fixture_manifest_sha256"], "fixture manifest"
        ),
        "fixture_approval_sha256": _digest(
            row["fixture_approval_sha256"], "fixture approval"
        ),
    }
    if row["dataset_boundary"] is not None:
        normalized["dataset_boundary"] = _identifier(
            row["dataset_boundary"], "dataset boundary"
        )
    allowed_phases = {"confirmation", "boundary"} if confirmation else {"discovery"}
    if normalized["phase"] not in allowed_phases:
        raise ValueError("score batch phase is not valid for this policy")
    if normalized["seed_role"] not in {"A", "B"}:
        raise ValueError("score batch seed role must be A or B")
    return normalized


def _validate_policy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the discovery policy payload (compatibility public API)."""

    if payload.get("schema") == 3:
        return _validate_agent_discovery_policy_payload(payload)

    _exact(
        payload,
        {
            "schema",
            "kind",
            "phase",
            "prepared_by",
            "discovery_plan",
            "discovery_execution_authorization",
            "confirmation_fixture_seal",
            "score_batches",
            "bootstrap",
        },
        "discovery decision policy payload",
    )
    if (
        payload["schema"] != 2
        or payload["kind"] != "forge-krea-discovery-decision-policy"
        or payload["phase"] != "discovery"
    ):
        raise ValueError("unsupported discovery decision policy")
    prepared_by = _named_human(payload["prepared_by"], "policy preparer")
    plan_state, _, seal = _bound_plan_and_seal(payload)
    _, authorization, _ = _binding(
        payload["discovery_execution_authorization"],
        "discovery execution authorization",
    )
    if "discovery_decision_evaluation" in authorization.get("authorized_actions", []):
        raise ValueError(
            "owner-ratified Stage-1 decision authority requires the agent policy"
        )
    if seal.get("schema") != 1:
        raise ValueError("legacy discovery policy requires the legacy human seal")
    if prepared_by == seal["reviewer_identity"]:
        raise ValueError("fixture sealer must be independent from policy preparer")
    bootstrap = _validate_bootstrap(payload["bootstrap"])
    raw_batches = payload["score_batches"]
    if not isinstance(raw_batches, list) or not raw_batches:
        raise ValueError("discovery policy requires score batches")
    batches = [_validate_score_batch(row, confirmation=False) for row in raw_batches]
    if batches != sorted(batches, key=lambda row: row["batch_id"]):
        raise ValueError("score batches must be sorted by batch_id")
    if len({row["batch_id"] for row in batches}) != len(batches) or len(
        {row["plan_canonical_sha256"] for row in batches}
    ) != len(batches):
        raise ValueError("score batches contain duplicate ids or plans")
    roles = [(row["fixture_id"], row["seed_role"]) for row in batches]
    if roles.count(("D1", "A")) != 1 or roles.count(("D2", "A")) != 1:
        raise ValueError("discovery policy requires exactly D1/A and D2/A")
    b_roles = [role for role in roles if role[1] == "B"]
    if b_roles and (set(b_roles) != {("D1", "B"), ("D2", "B")} or len(b_roles) != 2):
        raise ValueError("Seed B must be absent or predeclared for both D1 and D2")
    if len(set(roles)) != len(roles):
        raise ValueError("duplicate discovery fixture/seed role")
    for row in batches:
        expected_seed = (
            plan_state["seed_a"] if row["seed_role"] == "A" else plan_state["seed_b"]
        )
        if row["seed"] != expected_seed:
            raise ValueError("discovery score batch seed differs from frozen plan")
        if row["fixture_id"] not in _DISCOVERY_FIXTURES:
            raise ValueError("discovery batch fixture must be D1 or D2")
        if row["hours"] is not None or row["dataset_boundary"] is not None:
            raise ValueError("discovery batch must not masquerade as a boundary cell")
    return {
        "schema": 2,
        "kind": "forge-krea-discovery-decision-policy",
        "phase": "discovery",
        "prepared_by": prepared_by,
        "discovery_plan": dict(payload["discovery_plan"]),
        "discovery_execution_authorization": dict(
            payload["discovery_execution_authorization"]
        ),
        "confirmation_fixture_seal": dict(payload["confirmation_fixture_seal"]),
        "score_batches": batches,
        "bootstrap": bootstrap,
    }


def _validate_agent_discovery_policy_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    _exact(
        payload,
        {
            "schema",
            "kind",
            "phase",
            "technical_preparer_actor",
            "accountable_owner_identity",
            "owner_ratification_sha256",
            "fixture_admission_envelope",
            "discovery_plan",
            "discovery_execution_authorization",
            "confirmation_fixture_seal",
            "score_batches",
            "bootstrap",
            "delegated_review_contract",
            "agent_review_is_not_human_review",
        },
        "agent discovery decision policy payload",
    )
    if (
        payload["schema"] != 3
        or payload["kind"] != "forge-krea-agent-discovery-decision-policy"
        or payload["phase"] != "discovery"
    ):
        raise ValueError("unsupported agent discovery decision policy")
    preparer, context = _validate_agent_governance(
        payload,
        actor_field="technical_preparer_actor",
        delegated_actor_name="discovery_decision_policy_preparer",
    )
    plan_state, plan, seal = _bound_plan_and_seal(payload)
    if (
        context["authorization"]["discovery_plan"]["file_sha256"]
        != payload["discovery_plan"]["sha256"]
    ):
        raise ValueError("decision authority binds another discovery freeze")
    if seal.get("schema") != 2:
        raise ValueError("agent discovery policy requires the agent C1-C4 seal")
    if (
        seal["discovery_execution_authorization"]
        != payload["discovery_execution_authorization"]
        or seal["accountable_owner_identity"] != context["owner"]
        or seal["owner_ratification_sha256"] != context["owner_ratification_sha256"]
    ):
        raise ValueError("agent discovery policy and fixture seal authority differ")
    _require_distinct_actors(
        preparer,
        _actor(seal["technical_custodian_actor"], "fixture seal custodian"),
        label="decision policy preparer and fixture custodian",
    )
    if (
        seal["discovery_protocol_sha256"] != plan_state["protocol_sha256"]
        or {
            row["fixture_id"]: row["identity_commitment_sha256"]
            for row in seal["fixtures"]
        }
        != plan_state["confirmation_identities"]
    ):
        raise ValueError("agent fixture seal differs from the discovery freeze")
    bootstrap = _validate_bootstrap(payload["bootstrap"])
    raw_batches = payload["score_batches"]
    if not isinstance(raw_batches, list) or not raw_batches:
        raise ValueError("discovery policy requires score batches")
    batches = [_validate_score_batch(row, confirmation=False) for row in raw_batches]
    if batches != sorted(batches, key=lambda row: row["batch_id"]):
        raise ValueError("score batches must be sorted by batch_id")
    if len({row["batch_id"] for row in batches}) != len(batches) or len(
        {row["plan_canonical_sha256"] for row in batches}
    ) != len(batches):
        raise ValueError("score batches contain duplicate ids or plans")
    roles = [(row["fixture_id"], row["seed_role"]) for row in batches]
    if roles.count(("D1", "A")) != 1 or roles.count(("D2", "A")) != 1:
        raise ValueError("discovery policy requires exactly D1/A and D2/A")
    b_roles = [role for role in roles if role[1] == "B"]
    if b_roles and (set(b_roles) != {("D1", "B"), ("D2", "B")} or len(b_roles) != 2):
        raise ValueError("Seed B must be absent or predeclared for both D1 and D2")
    if len(set(roles)) != len(roles):
        raise ValueError("duplicate discovery fixture/seed role")
    for row in batches:
        expected_seed = (
            plan_state["seed_a"] if row["seed_role"] == "A" else plan_state["seed_b"]
        )
        if row["seed"] != expected_seed:
            raise ValueError("discovery score batch seed differs from frozen plan")
        if row["fixture_id"] not in _DISCOVERY_FIXTURES:
            raise ValueError("discovery batch fixture must be D1 or D2")
        if row["hours"] is not None or row["dataset_boundary"] is not None:
            raise ValueError("discovery batch must not masquerade as a boundary cell")
    return {
        "schema": 3,
        "kind": "forge-krea-agent-discovery-decision-policy",
        "phase": "discovery",
        "technical_preparer_actor": preparer,
        "accountable_owner_identity": context["owner"],
        "owner_ratification_sha256": context["owner_ratification_sha256"],
        "fixture_admission_envelope": dict(
            context["authorization"]["fixture_admission_envelope"]
        ),
        "discovery_plan": dict(payload["discovery_plan"]),
        "discovery_execution_authorization": dict(
            payload["discovery_execution_authorization"]
        ),
        "confirmation_fixture_seal": dict(payload["confirmation_fixture_seal"]),
        "score_batches": batches,
        "bootstrap": bootstrap,
        "delegated_review_contract": krea_delegated_review_contract.binding(),
        "agent_review_is_not_human_review": True,
    }


def seal_policy(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _validate_policy_payload(_object(payload, "discovery policy"))
    if normalized != payload:
        raise ValueError("discovery policy is not canonically normalized")
    return {
        **normalized,
        "policy_sha256": krea_provenance.canonical_sha256(normalized),
    }


seal_discovery_policy = seal_policy


def validate_policy(value: dict[str, Any]) -> dict[str, Any]:
    value = _object(value, "discovery policy")
    if value.get("schema") == 3:
        required = {
            "schema",
            "kind",
            "phase",
            "technical_preparer_actor",
            "accountable_owner_identity",
            "owner_ratification_sha256",
            "fixture_admission_envelope",
            "discovery_plan",
            "discovery_execution_authorization",
            "confirmation_fixture_seal",
            "score_batches",
            "bootstrap",
            "delegated_review_contract",
            "agent_review_is_not_human_review",
            "policy_sha256",
        }
    else:
        required = {
            "schema",
            "kind",
            "phase",
            "prepared_by",
            "discovery_plan",
            "discovery_execution_authorization",
            "confirmation_fixture_seal",
            "score_batches",
            "bootstrap",
            "policy_sha256",
        }
    _exact(value, required, "discovery policy")
    body = {key: item for key, item in value.items() if key != "policy_sha256"}
    normalized = _validate_policy_payload(body)
    if normalized != body or value["policy_sha256"] != krea_provenance.canonical_sha256(
        body
    ):
        raise ValueError("discovery policy digest/normalization mismatch")
    return value


validate_discovery_policy = validate_policy


def _validate_confirmation_policy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    _exact(
        payload,
        {
            "schema",
            "kind",
            "phase",
            "prepared_by",
            "discovery_plan",
            "discovery_execution_authorization",
            "confirmation_fixture_seal",
            "discovery_decision",
            "candidate_family_id",
            "score_batches",
            "public_reference_family_ids",
            "deployed_control_family_id",
            "bootstrap",
        },
        "confirmation decision policy payload",
    )
    if (
        payload["schema"] != 2
        or payload["kind"] != "forge-krea-confirmation-decision-policy"
        or payload["phase"] != "confirmation"
    ):
        raise ValueError("unsupported confirmation decision policy")
    prepared_by = _named_human(payload["prepared_by"], "policy preparer")
    plan_state, _, seal = _bound_plan_and_seal(payload)
    if seal.get("schema") == 1 and prepared_by == seal["reviewer_identity"]:
        raise ValueError("fixture sealer must be independent from policy preparer")
    _, discovery, _ = _binding(payload["discovery_decision"], "discovery decision")
    _validate_discovery_record(discovery)
    if (
        discovery["outcome"] != "finalists_frozen"
        or discovery["discovery_plan_file_sha256"]
        != payload["discovery_plan"]["sha256"]
        or discovery["confirmation_fixture_seal_sha256"] != seal["seal_sha256"]
    ):
        raise ValueError("confirmation policy lacks a matching frozen discovery")
    if set(discovery["all_family_checkpoint_rules"]) != set(plan_state["arm_ids"]):
        raise ValueError("discovery did not freeze checkpoint rules for every arm")
    if _timestamp_value(
        seal["sealed_at_utc"], "fixture sealed_at_utc"
    ) >= _timestamp_value(discovery["decided_at_utc"], "discovery decided_at_utc"):
        raise ValueError("C1-C4 were not sealed before discovery unblinding")
    candidate_family_id = _identifier(
        payload["candidate_family_id"], "confirmation candidate family"
    )
    if (
        candidate_family_id == _CONTROL_FAMILY
        or candidate_family_id not in discovery["finalist_family_ids"]
    ):
        raise ValueError(
            "confirmation candidate must be a non-control frozen discovery finalist"
        )
    public_ids = payload["public_reference_family_ids"]
    if public_ids != list(_PUBLIC_FAMILIES):
        raise ValueError("public references must be exhaustive K2, K3, and K4")
    if payload["deployed_control_family_id"] != _CONTROL_FAMILY:
        raise ValueError("confirmation control must be K0")
    bootstrap = _validate_bootstrap(payload["bootstrap"])
    raw_batches = payload["score_batches"]
    if not isinstance(raw_batches, list) or not raw_batches:
        raise ValueError("confirmation policy requires score batches")
    batches = [_validate_score_batch(row, confirmation=True) for row in raw_batches]
    if batches != sorted(batches, key=lambda row: row["batch_id"]):
        raise ValueError("confirmation score batches must be sorted")
    if len({row["batch_id"] for row in batches}) != len(batches) or len(
        {row["plan_canonical_sha256"] for row in batches}
    ) != len(batches):
        raise ValueError("confirmation batches contain duplicate ids or plans")

    confirmation_rows = [row for row in batches if row["phase"] == "confirmation"]
    expected_confirmation_roles = {
        (fixture, seed_role)
        for fixture in _CONFIRMATION_FIXTURES
        for seed_role in ("A", "B")
    }
    observed_confirmation_roles = {
        (row["fixture_id"], row["seed_role"]) for row in confirmation_rows
    }
    if observed_confirmation_roles != expected_confirmation_roles or len(
        confirmation_rows
    ) != len(expected_confirmation_roles):
        raise ValueError("confirmation requires C1-C4 at both predeclared seeds")
    seal_by_fixture = {row["fixture_id"]: row for row in seal["fixtures"]}
    tournament_hours = float(plan_state["document"]["tournament_envelope_hours"])
    for row in confirmation_rows:
        expected_seed = (
            plan_state["seed_a"] if row["seed_role"] == "A" else plan_state["seed_b"]
        )
        sealed = seal_by_fixture[row["fixture_id"]]
        if (
            row["seed"] != expected_seed
            or row["hours"] != tournament_hours
            or row["dataset_boundary"] is not None
            or row["fixture_manifest_sha256"] != sealed["fixture_manifest_sha256"]
            or row["fixture_approval_sha256"] != sealed["fixture_approval_sha256"]
        ):
            raise ValueError("confirmation batch escaped its sealed concept/seed")

    boundary_rows = [row for row in batches if row["phase"] == "boundary"]
    expected_cells = {
        (float(hours), boundary)
        for hours in plan_state["document"]["confirmation_contract"]["boundary_hours"]
        for boundary in plan_state["document"]["confirmation_contract"][
            "dataset_boundaries"
        ]
    }
    observed_cells = {(row["hours"], row["dataset_boundary"]) for row in boundary_rows}
    if observed_cells != expected_cells or len(boundary_rows) != len(expected_cells):
        raise ValueError("confirmation policy lacks the complete 3x2 boundary matrix")
    if any(
        row["seed_role"] != "A" or row["seed"] != plan_state["seed_a"]
        for row in boundary_rows
    ):
        raise ValueError("boundary mechanics cells must use frozen Seed A")
    return {
        "schema": 2,
        "kind": "forge-krea-confirmation-decision-policy",
        "phase": "confirmation",
        "prepared_by": prepared_by,
        "discovery_plan": dict(payload["discovery_plan"]),
        "discovery_execution_authorization": dict(
            payload["discovery_execution_authorization"]
        ),
        "confirmation_fixture_seal": dict(payload["confirmation_fixture_seal"]),
        "discovery_decision": dict(payload["discovery_decision"]),
        "candidate_family_id": candidate_family_id,
        "score_batches": batches,
        "public_reference_family_ids": list(_PUBLIC_FAMILIES),
        "deployed_control_family_id": _CONTROL_FAMILY,
        "bootstrap": bootstrap,
    }


def seal_confirmation_policy(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _validate_confirmation_policy_payload(
        _object(payload, "confirmation policy")
    )
    if normalized != payload:
        raise ValueError("confirmation policy is not canonically normalized")
    return {
        **normalized,
        "policy_sha256": krea_provenance.canonical_sha256(normalized),
    }


def validate_confirmation_policy(value: dict[str, Any]) -> dict[str, Any]:
    value = _object(value, "confirmation policy")
    required = {
        "schema",
        "kind",
        "phase",
        "prepared_by",
        "discovery_plan",
        "discovery_execution_authorization",
        "confirmation_fixture_seal",
        "discovery_decision",
        "candidate_family_id",
        "score_batches",
        "public_reference_family_ids",
        "deployed_control_family_id",
        "bootstrap",
        "policy_sha256",
    }
    _exact(value, required, "confirmation policy")
    body = {key: item for key, item in value.items() if key != "policy_sha256"}
    normalized = _validate_confirmation_policy_payload(body)
    if normalized != body or value["policy_sha256"] != krea_provenance.canonical_sha256(
        body
    ):
        raise ValueError("confirmation policy digest/normalization mismatch")
    return value


def build_approval(
    policy: dict[str, Any],
    *,
    reviewer_identity: str | None = None,
    technical_reviewer_actor: dict[str, Any] | None = None,
    approved_at_utc: str,
) -> dict[str, Any]:
    if policy.get("kind") == "forge-krea-discovery-decision-policy":
        validate_policy(policy)
        phase = "discovery"
    elif policy.get("kind") == "forge-krea-agent-discovery-decision-policy":
        validate_policy(policy)
        if reviewer_identity is not None or technical_reviewer_actor is None:
            raise ValueError(
                "agent discovery approval requires its prebound technical reviewer"
            )
        reviewer, context = _validate_agent_governance(
            {
                **policy,
                "technical_reviewer_actor": technical_reviewer_actor,
            },
            actor_field="technical_reviewer_actor",
            delegated_actor_name="discovery_decision_reviewer",
        )
        _require_distinct_actors(
            reviewer,
            policy["technical_preparer_actor"],
            label="decision policy preparer and reviewer",
        )
        _, seal, _ = _binding(
            policy["confirmation_fixture_seal"], "confirmation fixture seal"
        )
        _require_distinct_actors(
            reviewer,
            _actor(seal["technical_custodian_actor"], "fixture seal custodian"),
            label="decision policy reviewer and fixture custodian",
        )
        body = {
            "schema": 3,
            "kind": "forge-krea-agent-discovery-decision-policy-approval",
            "phase": "discovery",
            "policy_sha256": policy["policy_sha256"],
            "technical_reviewer_actor": reviewer,
            "accountable_owner_identity": context["owner"],
            "owner_ratification_sha256": context["owner_ratification_sha256"],
            "fixture_admission_envelope": dict(
                context["authorization"]["fixture_admission_envelope"]
            ),
            "discovery_execution_authorization": dict(
                policy["discovery_execution_authorization"]
            ),
            "delegated_review_contract": krea_delegated_review_contract.binding(),
            "agent_review_is_not_human_review": True,
            "approved_at_utc": _timestamp(approved_at_utc, "approved_at_utc"),
            "decision": "approved",
        }
        return {**body, "approval_sha256": krea_provenance.canonical_sha256(body)}
    elif policy.get("kind") == "forge-krea-confirmation-decision-policy":
        validate_confirmation_policy(policy)
        phase = "confirmation"
    else:
        raise ValueError("unsupported decision policy for approval")
    if technical_reviewer_actor is not None or reviewer_identity is None:
        raise ValueError("legacy decision approval requires a named human reviewer")
    reviewer = _named_human(reviewer_identity, "reviewer_identity")
    if reviewer == policy["prepared_by"]:
        raise ValueError("policy reviewer must be independent from its preparer")
    _, seal, _ = _binding(
        policy["confirmation_fixture_seal"], "confirmation fixture seal"
    )
    if seal.get("schema") == 1 and reviewer == seal["reviewer_identity"]:
        raise ValueError("policy reviewer must differ from the fixture sealer")
    body = {
        "schema": 2,
        "kind": f"forge-krea-{phase}-decision-policy-approval",
        "phase": phase,
        "policy_sha256": policy["policy_sha256"],
        "reviewer_identity": reviewer,
        "approved_at_utc": _timestamp(approved_at_utc, "approved_at_utc"),
        "decision": "approved",
    }
    return {**body, "approval_sha256": krea_provenance.canonical_sha256(body)}


def validate_approval(
    value: dict[str, Any], *, policy: dict[str, Any]
) -> dict[str, Any]:
    if policy.get("kind") == "forge-krea-discovery-decision-policy":
        validate_policy(policy)
        phase = "discovery"
    elif policy.get("kind") == "forge-krea-agent-discovery-decision-policy":
        validate_policy(policy)
        return _validate_agent_discovery_approval(value, policy=policy)
    elif policy.get("kind") == "forge-krea-confirmation-decision-policy":
        validate_confirmation_policy(policy)
        phase = "confirmation"
    else:
        raise ValueError("unsupported decision policy for approval")
    value = _object(value, "decision policy approval")
    _exact(
        value,
        {
            "schema",
            "kind",
            "phase",
            "policy_sha256",
            "reviewer_identity",
            "approved_at_utc",
            "decision",
            "approval_sha256",
        },
        "decision policy approval",
    )
    body = {key: item for key, item in value.items() if key != "approval_sha256"}
    if (
        value["schema"] != 2
        or value["kind"] != f"forge-krea-{phase}-decision-policy-approval"
        or value["phase"] != phase
        or value["policy_sha256"] != policy["policy_sha256"]
        or value["decision"] != "approved"
        or value["approval_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("decision approval does not bind this policy")
    reviewer = _named_human(value["reviewer_identity"], "reviewer_identity")
    _timestamp(value["approved_at_utc"], "approved_at_utc")
    if reviewer == policy["prepared_by"]:
        raise ValueError("decision reviewer is not independent")
    _, seal, _ = _binding(
        policy["confirmation_fixture_seal"], "confirmation fixture seal"
    )
    if seal.get("schema") == 1 and reviewer == seal["reviewer_identity"]:
        raise ValueError("decision reviewer equals the fixture sealer")
    approved = _timestamp_value(value["approved_at_utc"], "approved_at_utc")
    if approved <= _timestamp_value(seal["sealed_at_utc"], "fixture sealed_at_utc"):
        raise ValueError("policy approval predates its sealed fixture commitments")
    if phase == "confirmation":
        _, discovery, _ = _binding(policy["discovery_decision"], "discovery decision")
        if approved <= _timestamp_value(
            discovery["decided_at_utc"], "discovery decided_at_utc"
        ):
            raise ValueError("confirmation policy was approved before discovery froze")
    return value


def _validate_agent_discovery_approval(
    value: dict[str, Any], *, policy: dict[str, Any]
) -> dict[str, Any]:
    value = _object(value, "agent discovery policy approval")
    _exact(
        value,
        {
            "schema",
            "kind",
            "phase",
            "policy_sha256",
            "technical_reviewer_actor",
            "accountable_owner_identity",
            "owner_ratification_sha256",
            "fixture_admission_envelope",
            "discovery_execution_authorization",
            "delegated_review_contract",
            "agent_review_is_not_human_review",
            "approved_at_utc",
            "decision",
            "approval_sha256",
        },
        "agent discovery policy approval",
    )
    body = {key: item for key, item in value.items() if key != "approval_sha256"}
    if (
        value["schema"] != 3
        or value["kind"] != "forge-krea-agent-discovery-decision-policy-approval"
        or value["phase"] != "discovery"
        or value["policy_sha256"] != policy["policy_sha256"]
        or value["decision"] != "approved"
        or value["discovery_execution_authorization"]
        != policy["discovery_execution_authorization"]
        or value["approval_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("agent decision approval does not bind this policy")
    reviewer, context = _validate_agent_governance(
        value,
        actor_field="technical_reviewer_actor",
        delegated_actor_name="discovery_decision_reviewer",
    )
    _require_distinct_actors(
        reviewer,
        policy["technical_preparer_actor"],
        label="decision policy preparer and reviewer",
    )
    _, seal, _ = _binding(
        policy["confirmation_fixture_seal"], "confirmation fixture seal"
    )
    _require_distinct_actors(
        reviewer,
        _actor(seal["technical_custodian_actor"], "fixture seal custodian"),
        label="decision policy reviewer and fixture custodian",
    )
    if (
        context["owner"] != policy["accountable_owner_identity"]
        or context["owner_ratification_sha256"] != policy["owner_ratification_sha256"]
    ):
        raise ValueError("agent approval owner differs from decision policy")
    approved = _timestamp_value(value["approved_at_utc"], "approved_at_utc")
    if approved <= _timestamp_value(seal["sealed_at_utc"], "fixture sealed_at_utc"):
        raise ValueError("policy approval predates its sealed fixture commitments")
    if approved <= _timestamp_value(
        context["authorization"]["authorized_at_utc"], "authorized_at_utc"
    ):
        raise ValueError("policy approval predates discovery authorization")
    return value


def _validate_decision_aggregate_bindings(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("decision aggregate bindings must be a list")
    normalized: dict[str, dict[str, Any]] = {}
    previous = None
    for index, raw in enumerate(value):
        label = f"aggregate_bindings[{index}]"
        row = _object(raw, label)
        _exact(
            row,
            {
                "batch_id",
                "phase",
                "fixture_id",
                "seed_role",
                "hours",
                "dataset_boundary",
                "path",
                "file_sha256",
                "aggregate_sha256",
                "plan_canonical_sha256",
                "sealed_plan_approval_sha256",
                "score_plan_reviewer",
                "decision_evidence",
            },
            label,
        )
        batch_id = _identifier(row["batch_id"], f"{label}.batch_id")
        phase = _identifier(row["phase"], f"{label}.phase")
        _identifier(row["fixture_id"], f"{label}.fixture_id")
        seed_role = _identifier(row["seed_role"], f"{label}.seed_role")
        if phase not in {"discovery", "confirmation", "boundary"}:
            raise ValueError(f"{label}.phase is invalid")
        if seed_role not in {"A", "B"}:
            raise ValueError(f"{label}.seed_role is invalid")
        aggregate_path = _portable_relative_path(row["path"], f"{label}.path")
        if len(aggregate_path.parts) != 1:
            raise ValueError(f"{label}.path must be relative to its archive root")
        for key in (
            "file_sha256",
            "aggregate_sha256",
            "plan_canonical_sha256",
            "sealed_plan_approval_sha256",
        ):
            _digest(row[key], f"{label}.{key}")
        score_reviewer = row["score_plan_reviewer"]
        if isinstance(score_reviewer, dict):
            krea_delegated_review_contract.validate_actor(
                "exact_score_plan_reviewer", score_reviewer
            )
        else:
            _named_human(score_reviewer, f"{label}.score_plan_reviewer")
        evidence = _validate_decision_evidence_binding(row["decision_evidence"])
        if row["hours"] is not None:
            _decimal(
                row["hours"],
                f"{label}.hours",
                minimum=Decimal("0.01"),
                maximum=Decimal("24"),
            )
        if row["dataset_boundary"] is not None:
            _identifier(row["dataset_boundary"], f"{label}.dataset_boundary")
        if batch_id in normalized or (previous is not None and batch_id <= previous):
            raise ValueError("aggregate bindings must be unique and sorted by batch_id")
        previous = batch_id
        normalized[batch_id] = {
            **row,
            "path": aggregate_path.as_posix(),
            "decision_evidence": evidence,
        }
    return normalized


def _validate_discovery_curves(
    value: Any,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    curves = _object(value, "discovery curve_results")
    normalized: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for batch_id, raw_families in curves.items():
        _identifier(batch_id, "curve batch_id")
        families = _object(raw_families, f"curve batch {batch_id}")
        normalized_families = {}
        for family_id, raw_rows in families.items():
            _identifier(family_id, "curve family_id")
            if not isinstance(raw_rows, list) or not raw_rows:
                raise ValueError("discovery curve family must be non-empty")
            rows = []
            previous = None
            for index, raw_row in enumerate(raw_rows):
                label = f"curve_results.{batch_id}.{family_id}[{index}]"
                row = _object(raw_row, label)
                _exact(
                    row,
                    {
                        "candidate_id",
                        "candidate_sha256",
                        "step",
                        "fraction_numerator",
                        "fraction_denominator",
                        "fraction",
                        "image_exposures",
                        "weighted_loss",
                        "relative_improvement_over_zero",
                    },
                    label,
                )
                _identifier(row["candidate_id"], f"{label}.candidate_id")
                _digest(row["candidate_sha256"], f"{label}.candidate_sha256")
                step = _positive_int(row["step"], f"{label}.step")
                numerator = _positive_int(
                    row["fraction_numerator"], f"{label}.fraction_numerator"
                )
                denominator = _positive_int(
                    row["fraction_denominator"], f"{label}.fraction_denominator"
                )
                exposures = _positive_int(
                    row["image_exposures"], f"{label}.image_exposures"
                )
                fraction = _decimal(
                    row["fraction"],
                    f"{label}.fraction",
                    minimum=Decimal("0"),
                    maximum=Decimal("1"),
                )
                loss = _finite_loss(row["weighted_loss"], f"{label}.weighted_loss")
                improvement = _decimal(
                    row["relative_improvement_over_zero"],
                    f"{label}.relative_improvement_over_zero",
                    minimum=Decimal("-100"),
                    maximum=Decimal("1"),
                )
                if (
                    numerator != step
                    or denominator < numerator
                    or abs(fraction - Decimal(numerator) / Decimal(denominator))
                    > Decimal("1e-12")
                ):
                    raise ValueError(f"{label} has inconsistent step/fraction")
                ordering = (fraction, step, row["candidate_id"])
                if previous is not None and ordering <= previous:
                    raise ValueError("discovery curve rows are duplicate or unsorted")
                previous = ordering
                rows.append(
                    {
                        **row,
                        "fraction": fraction,
                        "weighted_loss": loss,
                        "relative_improvement_over_zero": improvement,
                        "image_exposures": exposures,
                    }
                )
            normalized_families[family_id] = rows
        normalized[batch_id] = normalized_families
    return normalized


def _validate_discovery_families(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("discovery family_results must be a list")
    normalized = {}
    previous = None
    for index, raw in enumerate(value):
        label = f"family_results[{index}]"
        row = _object(raw, label)
        _exact(
            row,
            {
                "family_id",
                "control",
                "concept_relative_improvement",
                "worst_case_relative_improvement",
                "mean_relative_improvement",
                "paired_cluster_bootstrap",
            },
            label,
        )
        family = _identifier(row["family_id"], f"{label}.family_id")
        if previous is not None and family <= previous:
            raise ValueError("family_results must be unique and sorted")
        previous = family
        concepts = _object(row["concept_relative_improvement"], f"{label}.concepts")
        if set(concepts) != set(_DISCOVERY_FIXTURES):
            raise ValueError("family result must contain exactly D1 and D2")
        values = {
            fixture: _decimal(
                concepts[fixture],
                f"{label}.{fixture}",
                minimum=Decimal("-100"),
                maximum=Decimal("1"),
            )
            for fixture in _DISCOVERY_FIXTURES
        }
        worst = _decimal(
            row["worst_case_relative_improvement"],
            f"{label}.worst_case",
            minimum=Decimal("-100"),
            maximum=Decimal("1"),
        )
        mean = _decimal(
            row["mean_relative_improvement"],
            f"{label}.mean",
            minimum=Decimal("-100"),
            maximum=Decimal("1"),
        )
        if abs(worst - min(values.values())) > Decimal("1e-12") or abs(
            mean - sum(values.values()) / len(values)
        ) > Decimal("1e-12"):
            raise ValueError("family result summary does not recompute")
        bootstrap = _object(row["paired_cluster_bootstrap"], f"{label}.bootstrap")
        _exact(bootstrap, {"point_estimate", "lower", "upper"}, f"{label}.bootstrap")
        # The generation label is separately bound, so only enforce valid CI
        # ordering and the exact point estimate here.
        lower = _decimal(
            bootstrap["lower"],
            f"{label}.bootstrap.lower",
            minimum=Decimal("-100"),
            maximum=Decimal("1"),
        )
        upper = _decimal(
            bootstrap["upper"],
            f"{label}.bootstrap.upper",
            minimum=Decimal("-100"),
            maximum=Decimal("1"),
        )
        point = _decimal(
            bootstrap["point_estimate"],
            f"{label}.bootstrap.point",
            minimum=Decimal("-100"),
            maximum=Decimal("1"),
        )
        if lower > point or point > upper or abs(point - mean) > Decimal("1e-12"):
            raise ValueError("family bootstrap interval/point estimate is invalid")
        if row["control"] is not (family == _CONTROL_FAMILY):
            raise ValueError("family control flag is invalid")
        normalized[family] = {**row, "concepts": values}
    return normalized


def _validate_discovery_record(value: dict[str, Any]) -> dict[str, Any]:
    value = _object(value, "discovery decision record")
    body = {key: item for key, item in value.items() if key != "decision_sha256"}
    common = {
        "schema",
        "kind",
        "phase",
        "decided_at_utc",
        "policy_sha256",
        "policy_file_sha256",
        "policy_approval_sha256",
        "policy_approval_file_sha256",
        "discovery_plan_file_sha256",
        "confirmation_fixture_seal_sha256",
        "aggregate_bindings",
        "bootstrap",
        "outcome",
        "blockers",
        "seed_b_trigger",
        "curve_results",
        "family_results",
        "finalist_family_ids",
        "checkpoint_rules",
        "all_family_checkpoint_rules",
        "production_mutation_authorized",
        "release_review_required",
        "decision_sha256",
    }
    agent_record = value.get("schema") == 3
    if agent_record:
        common |= {
            "decision_reviewer_actor",
            "accountable_owner_identity",
            "owner_ratification_sha256",
            "fixture_admission_envelope",
            "discovery_execution_authorization",
            "delegated_review_contract",
            "agent_review_is_not_human_review",
        }
    else:
        common.add("decision_reviewer_identity")
    outcome = value.get("outcome")
    frozen_only = {
        "seeds_used",
        "D1_winner_family_id",
        "D2_winner_family_id",
        "minimax_regret",
    }
    expected = common | (frozen_only if outcome == "finalists_frozen" else set())
    _exact(value, expected, "discovery decision record")
    if (
        value["schema"] not in {2, 3}
        or value["kind"]
        != (
            "forge-krea-agent-discovery-decision-record"
            if agent_record
            else "forge-krea-discovery-decision-record"
        )
        or value["phase"] != "discovery"
        or value["production_mutation_authorized"] is not False
        or value["release_review_required"] is not True
        or value["outcome"] not in {"seed_b_required", "finalists_frozen", "no-go"}
        or value["decision_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("discovery decision record is invalid")
    _timestamp(value["decided_at_utc"], "discovery decided_at_utc")
    for key in (
        "policy_sha256",
        "policy_file_sha256",
        "policy_approval_sha256",
        "policy_approval_file_sha256",
        "discovery_plan_file_sha256",
        "confirmation_fixture_seal_sha256",
    ):
        _digest(value[key], key)
    if agent_record:
        _validate_agent_governance(
            value,
            actor_field="decision_reviewer_actor",
            delegated_actor_name="discovery_decision_reviewer",
        )
    else:
        _named_human(value["decision_reviewer_identity"], "decision reviewer")
    _validate_bootstrap(value["bootstrap"])
    bindings = _validate_decision_aggregate_bindings(value["aggregate_bindings"])
    if not isinstance(value["blockers"], list) or any(
        not isinstance(item, str) or not item for item in value["blockers"]
    ):
        raise ValueError("discovery blockers are invalid")
    curves = _validate_discovery_curves(value["curve_results"])
    families = _validate_discovery_families(value["family_results"])
    finalists = value["finalist_family_ids"]
    rules = value["checkpoint_rules"]
    all_rules = value["all_family_checkpoint_rules"]
    if value["outcome"] == "finalists_frozen":
        if (
            not isinstance(finalists, list)
            or not finalists
            or len(finalists) > 4
            or _CONTROL_FAMILY not in finalists
            or len(finalists) != len(set(finalists))
            or not isinstance(rules, dict)
            or set(rules) != set(finalists)
            or not isinstance(all_rules, dict)
            or not set(finalists).issubset(all_rules)
            or set(all_rules) != set(families)
            or value["blockers"] != []
            or value["seeds_used"] not in [["A"], ["A", "B"]]
        ):
            raise ValueError("frozen discovery finalists/checkpoint rules are invalid")
        for family_id, rule in all_rules.items():
            _identifier(family_id, "checkpoint-rule family")
            _exact(
                _object(rule, f"checkpoint rule {family_id}"),
                {
                    "method",
                    "mapping_rule",
                    "tie_band",
                    "tie_breaker",
                    "cross_run_tie_breaker",
                    "target_fraction",
                    "maximum_within_family_regret",
                    "mean_within_family_regret",
                    "fixture_scores",
                    "actual_mappings",
                },
                f"checkpoint rule {family_id}",
            )
            if (
                rule["mapping_rule"]
                != "nearest actual candidate; ties choose earlier step"
                or rule["tie_band"] != float(_DISCOVERY_TIE)
                or rule["tie_breaker"]
                != "earliest actual step among candidates within 0.01 of best"
                or rule["cross_run_tie_breaker"]
                != (
                    "minimum maximum mapped step, then minimum mean mapped step, "
                    "then earliest target fraction"
                )
            ):
                raise ValueError(
                    "checkpoint rule escaped the frozen mapping/tie policy"
                )
            _decimal(
                rule["target_fraction"],
                f"{family_id}.target_fraction",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
            )
        expected_roles = {
            (fixture, seed_role)
            for fixture in _DISCOVERY_FIXTURES
            for seed_role in value["seeds_used"]
        }
        observed_roles = {
            (row["fixture_id"], row["seed_role"]) for row in bindings.values()
        }
        if (
            set(curves) != set(bindings)
            or observed_roles != expected_roles
            or len(bindings) != len(expected_roles)
            or any(row["phase"] != "discovery" for row in bindings.values())
            or any(
                set(batch_curves) != set(families) for batch_curves in curves.values()
            )
        ):
            raise ValueError(
                "frozen discovery curve coverage does not match its batches"
            )

        # Recompute each curve's zero-relative values and the concept summaries
        # from the self-contained record.  This prevents a newly self-hashed
        # record from changing winners while retaining old evidence bindings.
        selected_by_batch: dict[str, dict[str, Decimal]] = {}
        reconstructed_analyses: dict[tuple[str, str], dict[str, Any]] = {}
        for batch_id, batch_curves in curves.items():
            implied_zeros = []
            selected_by_batch[batch_id] = {}
            pseudo_candidates = []
            for family_id, rows in batch_curves.items():
                scored = []
                for row in rows:
                    denominator = Decimal("1") - row["relative_improvement_over_zero"]
                    if denominator <= 0:
                        raise ValueError("curve cannot imply a finite positive zero")
                    implied_zeros.append(row["weighted_loss"] / denominator)
                    scored.append((row["relative_improvement_over_zero"], row))
                    pseudo_candidates.append({**row, "family_id": family_id})
                best = max(score for score, _ in scored)
                near = [item for item in scored if best - item[0] <= _DISCOVERY_TIE]
                selected = min(
                    near,
                    key=lambda item: (
                        item[1]["fraction"],
                        item[1]["step"],
                        item[1]["candidate_id"],
                    ),
                )
                selected_by_batch[batch_id][family_id] = selected[0]
            zero = implied_zeros[0]
            if zero <= 0 or any(
                abs(item - zero) > Decimal("1e-10") for item in implied_zeros[1:]
            ):
                raise ValueError("curve relative improvements do not share one zero")
            binding = bindings[batch_id]
            reconstructed_analyses[(binding["fixture_id"], binding["seed_role"])] = {
                "batch_id": batch_id,
                "aggregate": {
                    "candidates": pseudo_candidates,
                    "zero": {"weighted_loss": zero},
                },
            }
        concept_scores = {
            family: {
                fixture: sum(
                    selected_by_batch[batch_id][family]
                    for batch_id, binding in bindings.items()
                    if binding["fixture_id"] == fixture
                )
                / len(value["seeds_used"])
                for fixture in _DISCOVERY_FIXTURES
            }
            for family in families
        }
        for family, result in families.items():
            if any(
                abs(result["concepts"][fixture] - concept_scores[family][fixture])
                > Decimal("1e-10")
                for fixture in _DISCOVERY_FIXTURES
            ):
                raise ValueError(
                    "family concept summaries do not recompute from curves"
                )
            expected_ci = _bootstrap_ci(
                concept_scores[family],
                label=f"discovery:{family}:{','.join(value['seeds_used'])}",
            )
            actual_ci = result["paired_cluster_bootstrap"]
            if any(
                abs(Decimal(str(actual_ci[key])) - Decimal(str(expected_ci[key])))
                > Decimal("1e-10")
                for key in ("point_estimate", "lower", "upper")
            ):
                raise ValueError("family bootstrap does not recompute from concepts")
        noncontrols = sorted(set(families) - {_CONTROL_FAMILY})
        winners = {
            fixture: min(
                noncontrols,
                key=lambda family: (-concept_scores[family][fixture], family),
            )
            for fixture in _DISCOVERY_FIXTURES
        }
        if (
            value["D1_winner_family_id"] != winners["D1"]
            or value["D2_winner_family_id"] != winners["D2"]
        ):
            raise ValueError("recorded discovery winners do not recompute")
        best_by_fixture = {
            fixture: max(concept_scores[family][fixture] for family in noncontrols)
            for fixture in _DISCOVERY_FIXTURES
        }
        regret = {
            family: max(
                best_by_fixture[fixture] - concept_scores[family][fixture]
                for fixture in _DISCOVERY_FIXTURES
            )
            for family in noncontrols
        }
        if set(value["minimax_regret"]) != set(regret) or any(
            abs(Decimal(str(value["minimax_regret"][family])) - score)
            > Decimal("1e-10")
            for family, score in regret.items()
        ):
            raise ValueError("recorded minimax regret does not recompute")
        expected_finalists: list[str] = []
        for family in (winners["D1"], winners["D2"]):
            if family not in expected_finalists:
                expected_finalists.append(family)
        remaining = [
            family for family in noncontrols if family not in expected_finalists
        ]
        if remaining:
            expected_finalists.append(
                min(remaining, key=lambda family: (regret[family], family))
            )
        expected_finalists.append(_CONTROL_FAMILY)
        if finalists != expected_finalists:
            raise ValueError("recorded finalists do not follow the frozen rule")
        for family, expected_rule in all_rules.items():
            recomputed_rule = _checkpoint_rule(
                family,
                analyses=reconstructed_analyses,
                fixtures=_DISCOVERY_FIXTURES,
                seed_roles=tuple(value["seeds_used"]),
                targets=(
                    Decimal("0.1"),
                    Decimal("0.25"),
                    Decimal("0.5"),
                    Decimal("0.75"),
                    Decimal("0.9"),
                    Decimal("1.0"),
                ),
            )
            if expected_rule != recomputed_rule:
                raise ValueError("checkpoint rule does not recompute from full curves")
    elif finalists != [] or rules != {} or all_rules != {}:
        raise ValueError("non-final discovery outcome cannot carry finalists")
    if value["outcome"] == "seed_b_required":
        trigger = _object(value["seed_b_trigger"], "Seed-B trigger")
        _exact(
            trigger,
            {
                "triggered",
                "reasons",
                "noncontrols_inside_band",
                "noncontrols_inside_band_by_fixture",
                "material_reversals",
            },
            "Seed-B trigger",
        )
        if trigger.get("triggered") is not True or not value["blockers"]:
            raise ValueError("seed_b_required lacks a triggered, blocked Seed-B record")
        if (
            set(curves) != set(bindings)
            or {(row["fixture_id"], row["seed_role"]) for row in bindings.values()}
            != {("D1", "A"), ("D2", "A")}
            or any(row["phase"] != "discovery" for row in bindings.values())
        ):
            raise ValueError("Seed-B request lacks complete D1/D2 Seed-A curves")
        family_sets = [set(batch) for batch in curves.values()]
        if (
            not family_sets
            or any(items != family_sets[0] for items in family_sets[1:])
            or _CONTROL_FAMILY not in family_sets[0]
            or not set(_PUBLIC_FAMILIES).issubset(family_sets[0])
        ):
            raise ValueError("Seed-B request curve family coverage is incomplete")
    elif value["outcome"] == "no-go":
        if value["seed_b_trigger"] is not None or not value["blockers"]:
            raise ValueError("no-go must carry blockers and no Seed-B trigger")
    else:
        trigger = _object(value["seed_b_trigger"], "Seed-B trigger")
        _exact(
            trigger,
            {
                "triggered",
                "reasons",
                "noncontrols_inside_band",
                "noncontrols_inside_band_by_fixture",
                "material_reversals",
            },
            "Seed-B trigger",
        )
        if trigger.get("triggered") not in {True, False}:
            raise ValueError("frozen discovery has an invalid Seed-B trigger")
    return value


def _finite_loss(value: Any, label: str) -> Decimal:
    return _decimal(value, label, minimum=Decimal("0"), maximum=Decimal("1"))


def _candidate_row(value: Any, *, text_weight: Decimal, label: str) -> dict[str, Any]:
    row = _object(value, label)
    required = {
        "candidate_id",
        "arm_id",
        "mode",
        "family_id",
        "candidate_sha256",
        "candidate_bytes",
        "execution_plan_sha256",
        "run_completion_sha256",
        "step",
        "fraction_numerator",
        "fraction_denominator",
        "image_exposures",
        "binding_manifest_sha256",
        "zero_control_manifest_sha256",
        "result_file",
        "result_file_sha256",
        "result_canonical_sha256",
        "weighted_loss",
        "text_mean",
        "blank_mean",
        "paired_rows",
        "mechanics",
    }
    _exact(row, required, label)
    candidate_id = _identifier(row["candidate_id"], f"{label}.candidate_id")
    mode = row["mode"]
    if mode not in {"local_run_candidate", "zero_lora_control"}:
        raise ValueError(f"{label}.mode is unsupported")
    candidate_bytes = _positive_int(row["candidate_bytes"], f"{label}.bytes")
    candidate_sha = _digest(row["candidate_sha256"], f"{label}.sha256")
    binding_sha = _digest(
        row["binding_manifest_sha256"], f"{label}.binding_manifest_sha256"
    )
    result_file = row["result_file"]
    if (
        not isinstance(result_file, str)
        or Path(result_file).name != result_file
        or result_file in {"", ".", ".."}
    ):
        raise ValueError(f"{label}.result_file is unsafe")
    result_file_sha = _digest(row["result_file_sha256"], f"{label}.result_file_sha256")
    result_canonical_sha = _digest(
        row["result_canonical_sha256"], f"{label}.result_canonical_sha256"
    )
    weighted = _finite_loss(row["weighted_loss"], f"{label}.weighted_loss")
    text_mean = _finite_loss(row["text_mean"], f"{label}.text_mean")
    blank_mean = _finite_loss(row["blank_mean"], f"{label}.blank_mean")
    expected_weighted = (
        text_weight * text_mean + (Decimal("1") - text_weight) * blank_mean
    )
    if abs(weighted - expected_weighted) > Decimal("1e-12"):
        raise ValueError(f"{label}.weighted_loss does not recompute")

    paired = row["paired_rows"]
    if not isinstance(paired, list) or not paired:
        raise ValueError(f"{label}.paired_rows must be non-empty")
    normalized_pairs = []
    identities = set()
    for index, raw_pair in enumerate(paired):
        pair = _object(raw_pair, f"{label}.paired_rows[{index}]")
        if not _LOSS_KEYS.issubset(pair) or set(pair) == set(_LOSS_KEYS):
            raise ValueError(f"{label}.paired_rows[{index}] lacks bound row identity")
        text_loss = _finite_loss(
            pair["text_guided_loss"], f"{label}.paired_rows[{index}].text"
        )
        blank_loss = _finite_loss(
            pair["blank_prompt_loss"], f"{label}.paired_rows[{index}].blank"
        )
        identity = {key: item for key, item in pair.items() if key not in _LOSS_KEYS}
        identity_sha = krea_provenance.canonical_sha256(identity)
        if identity_sha in identities:
            raise ValueError(f"{label}.paired_rows contains duplicate row identities")
        identities.add(identity_sha)
        normalized_pairs.append(
            {
                "identity": identity,
                "identity_sha256": identity_sha,
                "text_guided_loss": text_loss,
                "blank_prompt_loss": blank_loss,
                "weighted_loss": (
                    text_weight * text_loss + (Decimal("1") - text_weight) * blank_loss
                ),
            }
        )
    computed_text = sum(item["text_guided_loss"] for item in normalized_pairs) / len(
        normalized_pairs
    )
    computed_blank = sum(item["blank_prompt_loss"] for item in normalized_pairs) / len(
        normalized_pairs
    )
    if abs(text_mean - computed_text) > Decimal("1e-12") or abs(
        blank_mean - computed_blank
    ) > Decimal("1e-12"):
        raise ValueError(f"{label} means do not recompute from paired rows")

    if mode == "zero_lora_control":
        null_fields = (
            "arm_id",
            "family_id",
            "execution_plan_sha256",
            "run_completion_sha256",
            "step",
            "fraction_numerator",
            "fraction_denominator",
            "image_exposures",
            "mechanics",
        )
        if any(row[key] is not None for key in null_fields):
            raise ValueError("zero control carries local-run identity")
        arm_id = family_id = None
        step = numerator = denominator = exposures = None
        mechanics = None
        execution_plan_sha = run_completion_sha = None
        zero_control_manifest_sha = _digest(
            row["zero_control_manifest_sha256"],
            f"{label}.zero_control_manifest_sha256",
        )
    else:
        if row["zero_control_manifest_sha256"] is not None:
            raise ValueError("local candidate carries a zero-control manifest")
        zero_control_manifest_sha = None
        arm_id = _identifier(row["arm_id"], f"{label}.arm_id")
        family_id = _identifier(row["family_id"], f"{label}.family_id")
        if arm_id != family_id:
            raise ValueError("local candidate arm_id and family_id differ")
        execution_plan_sha = _digest(
            row["execution_plan_sha256"], f"{label}.execution_plan_sha256"
        )
        run_completion_sha = _digest(
            row["run_completion_sha256"], f"{label}.run_completion_sha256"
        )
        step = _positive_int(row["step"], f"{label}.step")
        numerator = _positive_int(
            row["fraction_numerator"], f"{label}.fraction_numerator"
        )
        denominator = _positive_int(
            row["fraction_denominator"], f"{label}.fraction_denominator"
        )
        exposures = _positive_int(row["image_exposures"], f"{label}.image_exposures")
        if numerator != step or denominator < numerator:
            raise ValueError("candidate fraction is inconsistent with its step")
        mechanics = _object(row["mechanics"], f"{label}.mechanics")
        expected_mechanics = {
            "natural_completion": True,
            "upload_ready": True,
            "clean_telemetry": True,
        }
        if mechanics != expected_mechanics:
            raise ValueError("local candidate lacks clean upload-ready completion")
    return {
        "candidate_id": candidate_id,
        "arm_id": arm_id,
        "mode": mode,
        "family_id": family_id,
        "candidate_sha256": candidate_sha,
        "candidate_bytes": candidate_bytes,
        "execution_plan_sha256": execution_plan_sha,
        "run_completion_sha256": run_completion_sha,
        "step": step,
        "fraction_numerator": numerator,
        "fraction_denominator": denominator,
        "image_exposures": exposures,
        "binding_manifest_sha256": binding_sha,
        "zero_control_manifest_sha256": zero_control_manifest_sha,
        "result_file": result_file,
        "result_file_sha256": result_file_sha,
        "result_canonical_sha256": result_canonical_sha,
        "weighted_loss": weighted,
        "text_mean": text_mean,
        "blank_mean": blank_mean,
        "paired_rows": normalized_pairs,
        "mechanics": mechanics,
    }


def _validate_campaign_adapter(value: Any) -> dict[str, Any]:
    """Validate the complete schema-2 campaign embedded in a score aggregate.

    The decision layer deliberately does not trust ``coverage.complete`` alone.
    The adapter carries the pre-score campaign's complete run/candidate ledger,
    allowing this consumer to prove that every sealed candidate was scored.
    """

    campaign = _object(value, "aggregate campaign adapter")
    _exact(
        campaign,
        {
            "manifest_sha256",
            "file_sha256",
            "fixture_manifest_sha256",
            "discovery_plan_sha256",
            "zero_control_manifest_sha256",
            "decision_contract",
            "confirmation_contract",
            "runs",
        },
        "aggregate campaign adapter",
    )
    for key in (
        "manifest_sha256",
        "file_sha256",
        "fixture_manifest_sha256",
        "discovery_plan_sha256",
        "zero_control_manifest_sha256",
    ):
        _digest(campaign[key], f"campaign.{key}")
    if campaign["decision_contract"] != _DISCOVERY_CAMPAIGN_CONTRACT:
        raise ValueError("campaign discovery contract is not the frozen contract")
    if campaign["confirmation_contract"] != _CONFIRMATION_CAMPAIGN_CONTRACT:
        raise ValueError("campaign confirmation contract is not the frozen contract")
    raw_runs = campaign["runs"]
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("campaign adapter must contain its complete run ledger")
    runs: list[dict[str, Any]] = []
    seen_arms: set[str] = set()
    seen_completions: set[str] = set()
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for run_index, raw_run in enumerate(raw_runs):
        label = f"campaign.runs[{run_index}]"
        run = _object(raw_run, label)
        _exact(
            run,
            {
                "arm_id",
                "execution_plan_sha256",
                "run_completion_sha256",
                "candidates",
            },
            label,
        )
        arm_id = _identifier(run["arm_id"], f"{label}.arm_id")
        execution_sha = _digest(
            run["execution_plan_sha256"], f"{label}.execution_plan_sha256"
        )
        completion_sha = _digest(
            run["run_completion_sha256"], f"{label}.run_completion_sha256"
        )
        if arm_id in seen_arms or completion_sha in seen_completions:
            raise ValueError("campaign contains a duplicate arm or completion")
        seen_arms.add(arm_id)
        seen_completions.add(completion_sha)
        raw_candidates = run["candidates"]
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError(f"{label} has no candidates")
        candidates: list[dict[str, Any]] = []
        for candidate_index, raw_candidate in enumerate(raw_candidates):
            candidate_label = f"{label}.candidates[{candidate_index}]"
            candidate = _object(raw_candidate, candidate_label)
            _exact(
                candidate,
                {"candidate_id", "sha256", "bytes", "step", "fraction"},
                candidate_label,
            )
            candidate_id = _identifier(
                candidate["candidate_id"], f"{candidate_label}.candidate_id"
            )
            candidate_sha = _digest(candidate["sha256"], f"{candidate_label}.sha256")
            candidate_bytes = _positive_int(
                candidate["bytes"], f"{candidate_label}.bytes"
            )
            step = _positive_int(candidate["step"], f"{candidate_label}.step")
            fraction = _object(candidate["fraction"], f"{candidate_label}.fraction")
            _exact(
                fraction, {"numerator", "denominator"}, f"{candidate_label}.fraction"
            )
            numerator = _positive_int(
                fraction["numerator"], f"{candidate_label}.fraction.numerator"
            )
            denominator = _positive_int(
                fraction["denominator"], f"{candidate_label}.fraction.denominator"
            )
            if numerator != step or denominator < numerator:
                raise ValueError(f"{candidate_label} has an invalid fraction")
            if candidate_id in seen_ids or candidate_sha in seen_hashes:
                raise ValueError("campaign candidates contain duplicate ids or bytes")
            seen_ids.add(candidate_id)
            seen_hashes.add(candidate_sha)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "sha256": candidate_sha,
                    "bytes": candidate_bytes,
                    "step": step,
                    "fraction": {"numerator": numerator, "denominator": denominator},
                }
            )
        if candidates != sorted(
            candidates, key=lambda row: (row["step"], row["sha256"])
        ):
            raise ValueError("campaign candidates must be sorted by step/hash")
        runs.append(
            {
                "arm_id": arm_id,
                "execution_plan_sha256": execution_sha,
                "run_completion_sha256": completion_sha,
                "candidates": candidates,
            }
        )
    if runs != sorted(runs, key=lambda row: row["arm_id"]):
        raise ValueError("campaign runs must be sorted by arm_id")
    manifest_body = {
        "schema": 2,
        "kind": "forge-krea-exact-score-campaign",
        "fixture_manifest_sha256": campaign["fixture_manifest_sha256"],
        "discovery_plan_sha256": campaign["discovery_plan_sha256"],
        "runs": runs,
        "zero_control_manifest_sha256": campaign["zero_control_manifest_sha256"],
        "decision_contract": campaign["decision_contract"],
        "confirmation_contract": campaign["confirmation_contract"],
    }
    manifest_sha = krea_provenance.canonical_sha256(manifest_body)
    manifest = {**manifest_body, "manifest_sha256": manifest_sha}
    file_sha = hashlib.sha256(
        krea_provenance.canonical_bytes(manifest) + b"\n"
    ).hexdigest()
    if (
        campaign["manifest_sha256"] != manifest_sha
        or campaign["file_sha256"] != file_sha
    ):
        raise ValueError("campaign adapter does not reproduce its sealed manifest")
    return {**campaign, "runs": runs}


def _validate_fixture_adapter(value: Any) -> dict[str, Any]:
    fixture = _object(value, "aggregate fixture adapter")
    _exact(
        fixture,
        {
            "manifest_sha256",
            "file_sha256",
            "concept_id",
            "experimental_role",
            "evaluation_dataset_sha256",
        },
        "aggregate fixture adapter",
    )
    for key in ("manifest_sha256", "file_sha256", "evaluation_dataset_sha256"):
        _digest(fixture[key], f"fixture.{key}")
    if not isinstance(fixture["concept_id"], str) or not fixture["concept_id"].strip():
        raise ValueError("fixture concept_id is empty")
    _identifier(fixture["experimental_role"], "fixture experimental_role")
    return dict(fixture)


def _campaign_candidates_from_scores(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in candidates:
        if row["mode"] == "zero_lora_control":
            continue
        key = (
            row["family_id"],
            row["execution_plan_sha256"],
            row["run_completion_sha256"],
        )
        grouped.setdefault(key, []).append(
            {
                "candidate_id": row["candidate_id"],
                "sha256": row["candidate_sha256"],
                "bytes": row["candidate_bytes"],
                "step": row["step"],
                "fraction": {
                    "numerator": row["fraction_numerator"],
                    "denominator": row["fraction_denominator"],
                },
            }
        )
    runs = []
    for (arm_id, execution_sha, completion_sha), rows in grouped.items():
        rows.sort(key=lambda row: (row["step"], row["sha256"]))
        runs.append(
            {
                "arm_id": arm_id,
                "execution_plan_sha256": execution_sha,
                "run_completion_sha256": completion_sha,
                "candidates": rows,
            }
        )
    return sorted(runs, key=lambda row: row["arm_id"])


def _validate_score_plan_evidence(
    *,
    plan: dict[str, Any],
    plan_raw: bytes,
    approval: dict[str, Any],
    approval_raw: bytes,
    aggregate: Mapping[str, Any],
) -> None:
    """Prove the separately approved plan is the plan the aggregate reports."""

    if plan_raw != krea_provenance.canonical_bytes(plan) + b"\n":
        raise ValueError("decision-evidence score plan is not canonical JSON")
    if approval_raw != krea_provenance.canonical_bytes(approval) + b"\n":
        raise ValueError("decision-evidence score-plan approval is not canonical JSON")
    _require_keys(
        plan,
        {
            "schema",
            "kind",
            "fixture_manifest",
            "fixture_approval",
            "campaign_manifest",
            "candidates",
            "evaluator",
            "sealed_plan_approval",
        },
        "decision-evidence score plan",
    )
    if plan["schema"] != 2 or plan["kind"] != "forge-krea-exact-score-plan":
        raise ValueError("decision-evidence score plan identity is invalid")
    if krea_provenance.canonical_sha256(plan) != aggregate["plan_canonical_sha256"]:
        raise ValueError("decision-evidence score plan differs from the aggregate")

    sealed = _object(plan["sealed_plan_approval"], "plan approval binding")
    _exact(sealed, {"path", "sha256"}, "plan approval binding")
    if sealed["sha256"] != aggregate["sealed_plan_approval_sha256"]:
        raise ValueError("score plan points to another approval")
    plan_bindings = {}
    raw_candidates = plan["candidates"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("decision-evidence score plan has no candidates")
    for index, raw in enumerate(raw_candidates):
        label = f"decision-evidence score plan candidate[{index}]"
        row = _object(raw, label)
        _require_keys(
            row,
            {"id", "arm_id", "path", "sha256", "candidate_binding"},
            label,
        )
        candidate_id = _identifier(row["id"], f"{label}.id")
        _identifier(row["arm_id"], f"{label}.arm_id")
        if not isinstance(row["path"], str) or not row["path"]:
            raise ValueError(f"{label}.path is empty")
        candidate_sha = _digest(row["sha256"], f"{label}.sha256")
        binding = _object(row["candidate_binding"], f"{label}.candidate_binding")
        _exact(binding, {"path", "sha256"}, f"{label}.candidate_binding")
        binding_sha = _digest(binding["sha256"], f"{label}.candidate_binding.sha256")
        if candidate_id in plan_bindings:
            raise ValueError("decision-evidence score plan repeats a candidate")
        plan_bindings[candidate_id] = {
            "candidate_sha256": candidate_sha,
            "binding_manifest_sha256": binding_sha,
        }

    aggregate_candidates = {row["candidate_id"]: row for row in aggregate["candidates"]}
    if set(plan_bindings) != set(aggregate_candidates):
        raise ValueError("score plan and aggregate candidate coverage differ")
    for candidate_id, planned in plan_bindings.items():
        scored = aggregate_candidates[candidate_id]
        if planned != {
            "candidate_sha256": scored["candidate_sha256"],
            "binding_manifest_sha256": scored["binding_manifest_sha256"],
        }:
            raise ValueError(
                f"score plan candidate {candidate_id} differs from the aggregate"
            )

    approval_candidates = [
        {
            "id": row["candidate_id"],
            "candidate_binding": {
                "mode": row["mode"],
                "binding_manifest_sha256": row["binding_manifest_sha256"],
            },
        }
        for row in aggregate["candidates"]
    ]
    approval_candidates.sort(key=lambda row: row["id"])
    evaluator = _object(plan["evaluator"], "decision-evidence plan evaluator")
    document = aggregate["document"]
    common_envelope = document.get("common_training_envelope")
    common_authorization_sha256 = (
        common_envelope.get("discovery_execution_authorization_sha256")
        if isinstance(common_envelope, dict)
        else None
    )
    approval_summary = krea_batch._validate_v2_approval(
        approval,
        approval_raw,
        plan=plan,
        candidates=approval_candidates,
        evaluator=evaluator,
        common_authorization_sha256=common_authorization_sha256,
    )
    summary = _object(document["sealed_plan_approval"], "aggregate plan approval")
    if summary != approval_summary:
        raise ValueError("aggregate approval summary differs from raw approval")
    if (
        document["plan"].get("approved_payload_sha256")
        != approval["plan_payload_sha256"]
    ):
        raise ValueError("aggregate approved-plan payload binding is wrong")
    if document.get("batch_runner_sha256") != approval["batch_runner_sha256"]:
        raise ValueError("aggregate batch runner differs from the approved plan")


def _validate_raw_evaluator_result(
    *,
    result: dict[str, Any],
    aggregate_row: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> None:
    """Cross-check one raw exact-evaluator result against its aggregate row."""

    _require_keys(
        result,
        {
            "schema",
            "evaluator",
            "candidate_sha256",
            "candidate_bytes",
            "model_type",
            "dataset_sha256",
            "image_count",
            "scored_rows",
            "text_mean",
            "blank_mean",
            "text_weight",
            "weighted_loss",
            "direction",
        },
        "raw exact-evaluator result",
    )
    if (
        result["schema"] != 2
        or result["evaluator"] != "god_krea2_img2img_exact"
        or result["model_type"] != "krea2"
        or result["direction"] != "min"
        or result["candidate_sha256"] != aggregate_row["candidate_sha256"]
        or result["candidate_bytes"] != aggregate_row["candidate_bytes"]
        or result["dataset_sha256"] != aggregate["fixture"]["evaluation_dataset_sha256"]
        or result["image_count"] != len(result["scored_rows"])
    ):
        raise ValueError("raw exact-evaluator result identity differs from aggregate")
    document_row = next(
        row
        for row in aggregate["document"]["candidates"]
        if row["candidate_id"] == aggregate_row["candidate_id"]
    )
    for key, result_key in (
        ("weighted_loss", "weighted_loss"),
        ("text_mean", "text_mean"),
        ("blank_mean", "blank_mean"),
        ("paired_rows", "scored_rows"),
    ):
        if document_row[key] != result[result_key]:
            raise ValueError(
                f"aggregate candidate {aggregate_row['candidate_id']} differs from "
                f"its raw evaluator result at {key}"
            )
    if (
        _decimal(
            result["text_weight"],
            "raw evaluator text_weight",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        != aggregate["text_weight"]
    ):
        raise ValueError("raw evaluator result text weight differs from aggregate")


def _reload_decision_evidence(
    *, aggregate_path: Path, aggregate: Mapping[str, Any]
) -> dict[str, Any]:
    """Reload the portable raw evidence bundle and prove every aggregate row."""

    reference = _object(
        aggregate["document"].get("decision_evidence"),
        "aggregate decision_evidence",
    )
    _exact(
        reference,
        {
            "archive_path",
            "manifest_path",
            "manifest_file_sha256",
            "manifest_sha256",
        },
        "aggregate decision_evidence",
    )
    archive_relative = _portable_relative_path(
        reference["archive_path"], "aggregate decision_evidence archive_path"
    )
    if len(archive_relative.parts) != 1:
        raise ValueError("decision-evidence archive must be beside the aggregate")
    archive = aggregate_path.parent / archive_relative
    if archive.is_symlink() or not archive.is_dir():
        raise ValueError("decision-evidence archive is absent or unsafe")
    manifest_path = _archive_member(
        archive,
        reference["manifest_path"],
        "decision-evidence manifest",
    )
    manifest, manifest_file_sha = _load_canonical(
        manifest_path, "decision-evidence manifest"
    )
    body = {key: item for key, item in manifest.items() if key != "manifest_sha256"}
    _exact(
        manifest,
        {
            "schema",
            "kind",
            "path_rule",
            "score_plan",
            "score_plan_approval",
            "evaluator_results",
            "manifest_sha256",
        },
        "decision-evidence manifest",
    )
    if (
        manifest["schema"] != 1
        or manifest["kind"] != "forge-krea-decision-evidence-bundle"
        or manifest["path_rule"] != "relative_to_this_manifest_parent"
        or manifest["manifest_sha256"] != krea_provenance.canonical_sha256(body)
        or manifest_file_sha != reference["manifest_file_sha256"]
        or manifest["manifest_sha256"] != reference["manifest_sha256"]
    ):
        raise ValueError("decision-evidence manifest identity is invalid")
    plan_entry = _manifest_file_entry(manifest["score_plan"], "evidence score plan")
    approval_entry = _manifest_file_entry(
        manifest["score_plan_approval"], "evidence score-plan approval"
    )
    plan_path = _archive_member(archive, plan_entry["path"], "evidence score plan")
    plan, plan_file_sha, plan_raw = _load_json_evidence(
        plan_path, "evidence score plan"
    )
    approval_path = _archive_member(
        archive, approval_entry["path"], "evidence score-plan approval"
    )
    approval, approval_file_sha, approval_raw = _load_json_evidence(
        approval_path, "evidence score-plan approval"
    )
    if (
        plan_file_sha != plan_entry["file_sha256"]
        or krea_provenance.canonical_sha256(plan) != plan_entry["canonical_sha256"]
        or approval_file_sha != approval_entry["file_sha256"]
        or krea_provenance.canonical_sha256(approval)
        != approval_entry["canonical_sha256"]
        or plan_file_sha != aggregate["document"]["plan"].get("raw_sha256")
        or approval_file_sha != aggregate["sealed_plan_approval_sha256"]
    ):
        raise ValueError("decision-evidence plan/approval bytes do not match aggregate")
    _validate_score_plan_evidence(
        plan=plan,
        plan_raw=plan_raw,
        approval=approval,
        approval_raw=approval_raw,
        aggregate=aggregate,
    )

    raw_results = manifest["evaluator_results"]
    if not isinstance(raw_results, list):
        raise ValueError("decision-evidence evaluator_results is not a list")
    aggregate_rows = {row["candidate_id"]: row for row in aggregate["candidates"]}
    results = []
    previous = None
    for index, raw in enumerate(raw_results):
        label = f"decision-evidence evaluator_results[{index}]"
        row = _object(raw, label)
        _exact(
            row,
            {"candidate_id", "path", "file_sha256", "canonical_sha256"},
            label,
        )
        candidate_id = _identifier(row["candidate_id"], f"{label}.candidate_id")
        if previous is not None and candidate_id <= previous:
            raise ValueError("decision-evidence results are duplicate or unsorted")
        previous = candidate_id
        if candidate_id not in aggregate_rows:
            raise ValueError("decision-evidence result is not in the aggregate")
        entry = _manifest_file_entry(
            {key: row[key] for key in ("path", "file_sha256", "canonical_sha256")},
            label,
        )
        result_path = _archive_member(archive, entry["path"], label)
        if result_path.name != aggregate_rows[candidate_id]["result_file"]:
            raise ValueError("decision-evidence result filename differs from aggregate")
        result, file_sha, _ = _load_json_evidence(result_path, label)
        if (
            file_sha != entry["file_sha256"]
            or file_sha != aggregate_rows[candidate_id]["result_file_sha256"]
            or krea_provenance.canonical_sha256(result) != entry["canonical_sha256"]
            or entry["canonical_sha256"]
            != aggregate_rows[candidate_id]["result_canonical_sha256"]
        ):
            raise ValueError("decision-evidence evaluator result bytes do not match")
        _validate_raw_evaluator_result(
            result=result,
            aggregate_row=aggregate_rows[candidate_id],
            aggregate=aggregate,
        )
        results.append({"candidate_id": candidate_id, **entry})
    if [row["candidate_id"] for row in results] != sorted(aggregate_rows):
        raise ValueError("decision-evidence evaluator result coverage is incomplete")
    return _validate_decision_evidence_binding(
        {
            "archive_path": archive_relative.as_posix(),
            "manifest_path": reference["manifest_path"],
            "manifest_file_sha256": manifest_file_sha,
            "manifest_sha256": manifest["manifest_sha256"],
            "score_plan": plan_entry,
            "score_plan_approval": approval_entry,
            "evaluator_results": results,
        }
    )


def _aggregate(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = _load_canonical(path, "exact-score aggregate")
    required = {
        "schema",
        "kind",
        "coverage",
        "direction",
        "plan",
        "campaign_manifest_sha256",
        "fixture_manifest_sha256",
        "fixture_approval_sha256",
        "sealed_plan_approval_sha256",
        "sealed_plan_approval",
        "evaluation_envelope",
        "fixture_contract",
        "campaign",
        "fixture",
        "training_run_envelopes",
        "candidates",
        "decision_evidence",
        "aggregate_sha256",
    }
    _require_keys(value, required, "exact-score aggregate")
    body = {key: item for key, item in value.items() if key != "aggregate_sha256"}
    if (
        value["schema"] != 2
        or value["kind"] != "forge-krea-exact-score-batch"
        or value["direction"] != "min"
        or value["aggregate_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("exact-score aggregate identity is invalid")
    coverage = _object(value["coverage"], "aggregate coverage")
    _exact(coverage, {"planned", "completed", "complete"}, "aggregate coverage")
    candidates = value["candidates"]
    if (
        coverage["complete"] is not True
        or isinstance(coverage["planned"], bool)
        or not isinstance(coverage["planned"], int)
        or coverage["planned"] <= 0
        or coverage["planned"] != coverage["completed"]
        or not isinstance(candidates, list)
        or coverage["completed"] != len(candidates)
    ):
        raise ValueError("exact-score aggregate coverage is incomplete")
    plan = _object(value["plan"], "aggregate plan")
    _digest(plan.get("canonical_sha256"), "aggregate plan canonical SHA")
    _digest(plan.get("raw_sha256"), "aggregate plan raw SHA")
    _digest(plan.get("approved_payload_sha256"), "aggregate approved plan payload")
    for key in (
        "campaign_manifest_sha256",
        "fixture_manifest_sha256",
        "fixture_approval_sha256",
        "sealed_plan_approval_sha256",
    ):
        _digest(value[key], f"aggregate {key}")
    approval = _object(value["sealed_plan_approval"], "score-plan approval summary")
    if approval.get("decision") != "approved":
        raise ValueError("score-plan approval summary is not approved")
    if "technical_reviewer_actor" in approval:
        score_plan_reviewer: Any = krea_delegated_review_contract.validate_actor(
            "exact_score_plan_reviewer", approval["technical_reviewer_actor"]
        )
        if (
            set(approval)
            != {
                "technical_reviewer_actor",
                "accountable_owner_identity",
                "decision",
                "agent_review_is_not_human_review",
            }
            or approval.get("agent_review_is_not_human_review") is not True
        ):
            raise ValueError("agent score-plan approval summary is incomplete")
        krea_fixture.named_human(
            approval.get("accountable_owner_identity"),
            "score-plan accountable owner",
        )
    else:
        _exact(
            approval,
            {"decision", "reviewer_identity"},
            "score-plan approval summary",
        )
        score_plan_reviewer = _named_human(
            approval.get("reviewer_identity"), "score-plan reviewer"
        )
    evaluation = _object(value["evaluation_envelope"], "evaluation envelope")
    text_weight = _decimal(
        evaluation.get("text_weight"),
        "evaluation text_weight",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    fixture_contract = _object(value["fixture_contract"], "fixture contract")
    _exact(
        fixture_contract,
        {
            "fixture_manifest_identity_sha256",
            "training_pair_count",
            "evaluation_row_count",
            "training_dataset_sha256",
            "evaluation_dataset_sha256",
            "cross_fixture_review_sha256",
        },
        "fixture contract",
    )
    for key in (
        "fixture_manifest_identity_sha256",
        "training_dataset_sha256",
        "evaluation_dataset_sha256",
        "cross_fixture_review_sha256",
    ):
        _digest(fixture_contract[key], f"fixture_contract.{key}")
    training_pairs = _positive_int(
        fixture_contract["training_pair_count"], "fixture training_pair_count"
    )
    evaluation_rows = _positive_int(
        fixture_contract["evaluation_row_count"], "fixture evaluation_row_count"
    )
    normalized_candidates = [
        _candidate_row(row, text_weight=text_weight, label=f"candidate[{index}]")
        for index, row in enumerate(candidates)
    ]
    ids = [row["candidate_id"] for row in normalized_candidates]
    shas = [row["candidate_sha256"] for row in normalized_candidates]
    if len(ids) != len(set(ids)) or len(shas) != len(set(shas)):
        raise ValueError("aggregate candidates contain duplicate ids or bytes")
    zeros = [row for row in normalized_candidates if row["mode"] == "zero_lora_control"]
    if len(zeros) != 1:
        raise ValueError("aggregate must contain exactly one explicit zero control")
    zero_identities = [row["identity"] for row in zeros[0]["paired_rows"]]
    for row in normalized_candidates:
        if [item["identity"] for item in row["paired_rows"]] != zero_identities:
            raise ValueError(
                "candidate rows are not exactly paired to the zero control"
            )
    campaign = _validate_campaign_adapter(value["campaign"])
    fixture = _validate_fixture_adapter(value["fixture"])
    if (
        value["campaign_manifest_sha256"] != campaign["file_sha256"]
        or value["fixture_manifest_sha256"] != fixture["file_sha256"]
        or campaign["fixture_manifest_sha256"] != fixture["manifest_sha256"]
        or fixture_contract["fixture_manifest_identity_sha256"]
        != fixture["manifest_sha256"]
        or fixture_contract["evaluation_dataset_sha256"]
        != fixture["evaluation_dataset_sha256"]
    ):
        raise ValueError("aggregate campaign/fixture adapters are not coherently bound")
    score_runs = _campaign_candidates_from_scores(normalized_candidates)
    if score_runs != campaign["runs"]:
        raise ValueError("aggregate cherry-picks or invents sealed campaign candidates")
    if (
        zeros[0]["zero_control_manifest_sha256"]
        != campaign["zero_control_manifest_sha256"]
    ):
        raise ValueError("aggregate zero control differs from the sealed campaign")
    envelopes = value["training_run_envelopes"]
    if not isinstance(envelopes, list) or not envelopes:
        raise ValueError("aggregate lacks training-run envelopes")
    envelope_keys = []
    for index, raw_envelope in enumerate(envelopes):
        envelope = _object(raw_envelope, f"training_run_envelopes[{index}]")
        _require_keys(
            envelope,
            {"arm_id", "execution_plan_sha256"},
            f"training_run_envelopes[{index}]",
        )
        envelope_keys.append(
            (
                _identifier(envelope["arm_id"], "training envelope arm"),
                _digest(
                    envelope["execution_plan_sha256"],
                    "training envelope execution plan",
                ),
            )
        )
    expected_envelopes = [
        (row["arm_id"], row["execution_plan_sha256"]) for row in campaign["runs"]
    ]
    if envelope_keys != expected_envelopes:
        raise ValueError("training-run envelopes do not cover the sealed campaign")
    return {
        "document": value,
        "file_sha256": file_sha,
        "plan_canonical_sha256": plan["canonical_sha256"],
        "campaign_manifest_sha256": value["campaign_manifest_sha256"],
        "fixture_manifest_sha256": value["fixture_manifest_sha256"],
        "fixture_approval_sha256": value["fixture_approval_sha256"],
        "sealed_plan_approval_sha256": value["sealed_plan_approval_sha256"],
        "score_plan_reviewer": score_plan_reviewer,
        "aggregate_sha256": value["aggregate_sha256"],
        "candidates": normalized_candidates,
        "zero": zeros[0],
        "text_weight": text_weight,
        "fixture_contract": {
            **fixture_contract,
            "training_pair_count": training_pairs,
            "evaluation_row_count": evaluation_rows,
        },
        "campaign": campaign,
        "fixture": fixture,
        "training_run_envelopes": envelopes,
    }, file_sha


def _load_policy_approval(
    *, policy_path: Path, approval_path: Path, confirmation: bool
) -> tuple[dict[str, Any], str, dict[str, Any], str, dict[str, Any]]:
    policy, policy_file_sha = _load_canonical(policy_path, "decision policy")
    if confirmation:
        validate_confirmation_policy(policy)
    else:
        validate_policy(policy)
    approval, approval_file_sha = _load_canonical(
        approval_path, "decision policy approval"
    )
    validate_approval(approval, policy=policy)
    plan_state, _, seal = _bound_plan_and_seal(policy)
    return (
        policy,
        policy_file_sha,
        approval,
        approval_file_sha,
        {
            "plan": plan_state,
            "seal": seal,
        },
    )


def _expected_fixture_counts(
    *, plan_state: Mapping[str, Any], fixture_id: str, boundary: Any
) -> tuple[int, int, int]:
    """Resolve counts without treating distinct C1-C4 fixtures as size aliases."""

    if fixture_id in _DISCOVERY_FIXTURES:
        if boundary is not None:
            raise ValueError("discovery fixture cannot declare a boundary alias")
        return _DISCOVERY_FIXTURE_COUNTS[fixture_id]
    if fixture_id in _CONFIRMATION_FIXTURES:
        if boundary is not None:
            raise ValueError("confirmation fixture cannot declare a boundary alias")
        shape = plan_state["confirmation_shape_contract"][fixture_id]
        training_pairs = shape["training_pairs"]
        return training_pairs, training_pairs, shape["evaluation_rows"]
    if boundary not in _BOUNDARY_FIXTURE_COUNTS:
        raise ValueError("score batch does not map to a frozen fixture contract")
    return _BOUNDARY_FIXTURE_COUNTS[boundary]


def _match_aggregates(
    *, policy: dict[str, Any], aggregate_paths: Iterable[Path]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    plan_state, _, seal = _bound_plan_and_seal(policy)
    confirmation_policy = policy["phase"] == "confirmation"
    discovery: dict[str, Any] | None = None
    if confirmation_policy:
        _, discovery, _ = _binding(policy["discovery_decision"], "discovery decision")
        _validate_discovery_record(discovery)
    sealed_confirmation = {row["fixture_id"]: row for row in seal["fixtures"]}
    expected_by_plan = {
        row["plan_canonical_sha256"]: row for row in policy["score_batches"]
    }
    observed: dict[str, dict[str, Any]] = {}
    bindings = []
    for raw_path in aggregate_paths:
        path = _safe_file(raw_path, "exact-score aggregate")
        aggregate, file_sha = _aggregate(path)
        decision_evidence = _reload_decision_evidence(
            aggregate_path=path, aggregate=aggregate
        )
        expected = expected_by_plan.get(aggregate["plan_canonical_sha256"])
        if expected is None or expected["batch_id"] in observed:
            raise ValueError("unexpected or duplicate exact-score aggregate")
        for aggregate_key, policy_key in (
            ("campaign_manifest_sha256", "campaign_manifest_sha256"),
            ("fixture_manifest_sha256", "fixture_manifest_sha256"),
            ("fixture_approval_sha256", "fixture_approval_sha256"),
            ("sealed_plan_approval_sha256", "sealed_plan_approval_sha256"),
        ):
            if aggregate[aggregate_key] != expected[policy_key]:
                raise ValueError(
                    f"aggregate {expected['batch_id']} differs at {aggregate_key}"
                )
        campaign = aggregate["campaign"]
        if campaign["discovery_plan_sha256"] != policy["discovery_plan"]["sha256"]:
            raise ValueError("aggregate campaign is bound to another discovery plan")
        fixture_adapter = aggregate["fixture"]
        if fixture_adapter["experimental_role"] != expected["fixture_id"]:
            raise ValueError("aggregate fixture role differs from the decision batch")
        campaign_arms = [row["arm_id"] for row in campaign["runs"]]
        if not confirmation_policy:
            expected_arms = list(plan_state["arm_ids"])
        elif expected["phase"] == "confirmation":
            assert discovery is not None
            expected_arms = sorted(
                set(discovery["finalist_family_ids"])
                | set(_PUBLIC_FAMILIES)
                | {_CONTROL_FAMILY}
            )
            counts = {
                arm: sum(
                    candidate["family_id"] == arm
                    for candidate in aggregate["candidates"]
                    if candidate["mode"] == "local_run_candidate"
                )
                for arm in expected_arms
            }
            if any(not 1 <= count <= 3 for count in counts.values()):
                raise ValueError(
                    "confirmation must score locked/final/at-most-one-guard per family"
                )
        else:
            expected_arms = [policy["candidate_family_id"]]
            local_candidates = [
                candidate
                for candidate in aggregate["candidates"]
                if candidate["mode"] == "local_run_candidate"
            ]
            if len(local_candidates) != 1:
                raise ValueError(
                    "boundary mechanics batch must expose exactly one chosen artifact"
                )
            envelope = aggregate["training_run_envelopes"][0]
            evidence = _object(
                envelope.get("candidate_decision"), "boundary candidate decision"
            )
            _exact(
                evidence,
                {
                    "mode",
                    "selected_candidate_sha256",
                    "decision_completed_before_export_reserve",
                    "fallback_used",
                },
                "boundary candidate decision",
            )
            if evidence != {
                "mode": "frozen_checkpoint_rule",
                "selected_candidate_sha256": local_candidates[0]["candidate_sha256"],
                "decision_completed_before_export_reserve": True,
                "fallback_used": False,
            }:
                raise ValueError(
                    "boundary candidate decision is late, fallback-dependent, or unbound"
                )
        if campaign_arms != sorted(expected_arms):
            raise ValueError(
                "aggregate campaign arm coverage is not exact: "
                f"expected={sorted(expected_arms)}, actual={campaign_arms}"
            )
        contract = aggregate["fixture_contract"]
        if (
            contract["cross_fixture_review_sha256"]
            != seal["cross_fixture_review_sha256"]
        ):
            raise ValueError(
                "aggregate lacks the independently bound cross-fixture review"
            )
        if contract["training_dataset_sha256"] == contract["evaluation_dataset_sha256"]:
            raise ValueError("aggregate training and evaluation datasets are not split")
        fixture_id = expected["fixture_id"]
        boundary = expected["dataset_boundary"]
        lower, upper, eval_count = _expected_fixture_counts(
            plan_state=plan_state,
            fixture_id=fixture_id,
            boundary=boundary,
        )
        if (
            not lower <= contract["training_pair_count"] <= upper
            or contract["evaluation_row_count"] != eval_count
        ):
            raise ValueError(
                f"aggregate {expected['batch_id']} violates frozen fixture counts"
            )
        if fixture_id in _DISCOVERY_FIXTURES:
            expected_identity = plan_state["document"]["discovery_tasks"][fixture_id][
                "fixture_split_manifest_sha256"
            ]
            if contract["fixture_manifest_identity_sha256"] != expected_identity:
                raise ValueError("discovery aggregate fixture identity drifted")
        elif fixture_id in _CONFIRMATION_FIXTURES:
            if (
                contract["fixture_manifest_identity_sha256"]
                != sealed_confirmation[fixture_id]["identity_commitment_sha256"]
            ):
                raise ValueError(
                    "confirmation aggregate does not open its sealed fixture"
                )
        observed[expected["batch_id"]] = {**aggregate, "batch": expected}
        bindings.append(
            {
                "batch_id": expected["batch_id"],
                "phase": expected["phase"],
                "fixture_id": expected["fixture_id"],
                "seed_role": expected["seed_role"],
                "hours": expected["hours"],
                "dataset_boundary": expected["dataset_boundary"],
                "path": path.name,
                "file_sha256": file_sha,
                "aggregate_sha256": aggregate["aggregate_sha256"],
                "plan_canonical_sha256": aggregate["plan_canonical_sha256"],
                "sealed_plan_approval_sha256": aggregate["sealed_plan_approval_sha256"],
                "score_plan_reviewer": aggregate["score_plan_reviewer"],
                "decision_evidence": decision_evidence,
            }
        )
    return observed, sorted(bindings, key=lambda row: row["batch_id"])


def _candidate_fraction(row: Mapping[str, Any]) -> Decimal:
    return Decimal(row["fraction_numerator"]) / Decimal(row["fraction_denominator"])


def _relative_improvement(
    candidate: Mapping[str, Any], zero: Mapping[str, Any]
) -> Decimal:
    zero_loss = zero["weighted_loss"]
    if zero_loss <= 0:
        raise ValueError("zero-control weighted loss must be positive")
    return (zero_loss - candidate["weighted_loss"]) / zero_loss


def _public_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "candidate_sha256": row["candidate_sha256"],
        "step": row["step"],
        "fraction_numerator": row["fraction_numerator"],
        "fraction_denominator": row["fraction_denominator"],
        "fraction": float(_candidate_fraction(row)),
        "image_exposures": row["image_exposures"],
        "weighted_loss": float(row["weighted_loss"]),
    }


def _curves(
    aggregate: Mapping[str, Any], *, expected_arm_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for candidate in aggregate["candidates"]:
        if candidate["mode"] == "zero_lora_control":
            continue
        by_family.setdefault(candidate["family_id"], []).append(candidate)
    if set(by_family) != set(expected_arm_ids):
        raise ValueError(
            "aggregate family coverage is not exhaustive: "
            f"missing={sorted(set(expected_arm_ids)-set(by_family))}, "
            f"extra={sorted(set(by_family)-set(expected_arm_ids))}"
        )
    zero = aggregate["zero"]
    result = {}
    for family_id in expected_arm_ids:
        rows = sorted(
            by_family[family_id],
            key=lambda row: (
                _candidate_fraction(row),
                row["step"],
                row["candidate_id"],
            ),
        )
        fractions = [_candidate_fraction(row) for row in rows]
        if len(fractions) != len(set(fractions)):
            raise ValueError(f"family {family_id} has duplicate checkpoint fractions")
        scored = [(_relative_improvement(row, zero), row) for row in rows]
        best = max(item[0] for item in scored)
        near = [item for item in scored if best - item[0] <= _DISCOVERY_TIE]
        selected_score, selected = min(
            near,
            key=lambda item: (
                _candidate_fraction(item[1]),
                item[1]["step"],
                item[1]["candidate_id"],
            ),
        )
        result[family_id] = {
            "curve": [
                {
                    **_public_candidate(row),
                    "relative_improvement_over_zero": float(score),
                }
                for score, row in scored
            ],
            "selected": selected,
            "selected_relative_improvement": selected_score,
        }
    return result


def _map_target(rows: Sequence[dict[str, Any]], target: Decimal) -> dict[str, Any]:
    return min(
        rows,
        key=lambda row: (
            abs(_candidate_fraction(row) - target),
            row["step"],
            row["candidate_id"],
        ),
    )


def _bootstrap_ci(clusters: Mapping[str, Decimal], *, label: str) -> dict[str, float]:
    if not clusters:
        raise ValueError("bootstrap requires concept clusters")
    names = sorted(clusters)
    salt = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(_BOOTSTRAP_SEED ^ salt)
    samples = []
    for _ in range(_BOOTSTRAP_RESAMPLES):
        draw = [clusters[names[rng.randrange(len(names))]] for _ in names]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    alpha = (Decimal("1") - _CONFIDENCE) / Decimal("2")
    lower_index = int(alpha * len(samples))
    upper_index = int((Decimal("1") - alpha) * len(samples)) - 1
    point = sum(clusters.values()) / len(clusters)
    return {
        "point_estimate": float(point),
        "lower": float(samples[max(0, lower_index)]),
        "upper": float(samples[min(len(samples) - 1, upper_index)]),
    }


def _output_path(output: Path, *, phase: str) -> Path:
    output = Path(os.path.abspath(os.path.expanduser(output)))
    expected_prefix = f"krea-{phase}-decision"
    if (
        output.name in _FORBIDDEN_OUTPUTS
        or not _OUTPUT_NAME.fullmatch(output.name)
        or not output.name.startswith(expected_prefix)
    ):
        raise ValueError("decision output must be a non-production JSON record")
    repository_root = Path(__file__).resolve().parents[2]
    production_directory = repository_root / "forge"
    if output == production_directory or production_directory in output.parents:
        raise ValueError("decision output cannot target the production package")
    current = output.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"decision output has a symlink ancestor: {current}")
        current = current.parent
    if os.path.lexists(output) or os.path.lexists(Path(f"{output}.tmp")):
        raise FileExistsError(f"refusing existing decision output: {output}")
    return output


def _publish_record(output: Path, body: dict[str, Any]) -> dict[str, Any]:
    phase = body.get("phase")
    if phase not in {"discovery", "confirmation"}:
        raise ValueError("decision record phase is invalid")
    output = _output_path(output, phase=phase)
    record = {**body, "decision_sha256": krea_provenance.canonical_sha256(body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{output}.tmp")
    payload = krea_provenance.canonical_bytes(record) + b"\n"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, output)
    temporary.unlink()
    directory_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return record


def _analysis_by_role(
    observed: Mapping[str, dict[str, Any]],
    *,
    expected_arm_ids: Sequence[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    analyses: dict[tuple[str, str], dict[str, Any]] = {}
    for batch in observed.values():
        role = (batch["batch"]["fixture_id"], batch["batch"]["seed_role"])
        if role in analyses:
            raise ValueError(f"duplicate aggregate role: {role}")
        analyses[role] = {
            "batch_id": batch["batch"]["batch_id"],
            "curves": _curves(batch, expected_arm_ids=expected_arm_ids),
            "aggregate": batch,
        }
    return analyses


def _concept_family_scores(
    analyses: Mapping[tuple[str, str], dict[str, Any]],
    *,
    fixtures: Sequence[str],
    seed_roles: Sequence[str],
    family_ids: Sequence[str],
) -> dict[str, dict[str, Decimal]]:
    result: dict[str, dict[str, Decimal]] = {family: {} for family in family_ids}
    for fixture in fixtures:
        for family in family_ids:
            values = []
            for seed_role in seed_roles:
                analysis = analyses.get((fixture, seed_role))
                if analysis is None:
                    raise ValueError(f"missing analysis for {fixture}/{seed_role}")
                values.append(
                    analysis["curves"][family]["selected_relative_improvement"]
                )
            result[family][fixture] = sum(values) / len(values)
    return result


def _material_rank_reversal(
    concept_scores: Mapping[str, Mapping[str, Decimal]],
    *,
    noncontrols: Sequence[str],
) -> list[dict[str, Any]]:
    reversals = []
    for index, left in enumerate(noncontrols):
        for right in noncontrols[index + 1 :]:
            d1 = concept_scores[left]["D1"] - concept_scores[right]["D1"]
            d2 = concept_scores[left]["D2"] - concept_scores[right]["D2"]
            if (d1 > _DISCOVERY_TIE and d2 < -_DISCOVERY_TIE) or (
                d1 < -_DISCOVERY_TIE and d2 > _DISCOVERY_TIE
            ):
                reversals.append(
                    {
                        "left_family_id": left,
                        "right_family_id": right,
                        "D1_separation": float(d1),
                        "D2_separation": float(d2),
                    }
                )
    return reversals


def _checkpoint_rule(
    family_id: str,
    *,
    analyses: Mapping[tuple[str, str], dict[str, Any]],
    fixtures: Sequence[str],
    seed_roles: Sequence[str],
    targets: Sequence[Decimal],
) -> dict[str, Any]:
    target_rows = []
    per_fixture_best: dict[str, Decimal] = {}
    for fixture in fixtures:
        scores = []
        for target in targets:
            values = []
            for seed_role in seed_roles:
                raw_rows = [
                    row
                    for row in analyses[(fixture, seed_role)]["aggregate"]["candidates"]
                    if row["family_id"] == family_id
                ]
                mapped = _map_target(raw_rows, target)
                zero = analyses[(fixture, seed_role)]["aggregate"]["zero"]
                values.append(_relative_improvement(mapped, zero))
            scores.append(sum(values) / len(values))
        per_fixture_best[fixture] = max(scores)

    for target in targets:
        fixture_scores = {}
        actual_mappings = []
        for fixture in fixtures:
            values = []
            for seed_role in seed_roles:
                analysis = analyses[(fixture, seed_role)]
                raw_rows = [
                    row
                    for row in analysis["aggregate"]["candidates"]
                    if row["family_id"] == family_id
                ]
                mapped = _map_target(raw_rows, target)
                score = _relative_improvement(mapped, analysis["aggregate"]["zero"])
                values.append(score)
                actual_mappings.append(
                    {
                        "batch_id": analysis["batch_id"],
                        "fixture_id": fixture,
                        "seed_role": seed_role,
                        **_public_candidate(mapped),
                    }
                )
            fixture_scores[fixture] = sum(values) / len(values)
        regrets = {
            fixture: per_fixture_best[fixture] - fixture_scores[fixture]
            for fixture in fixtures
        }
        target_rows.append(
            {
                "target_fraction": target,
                "maximum_within_family_regret": max(regrets.values()),
                "mean_within_family_regret": sum(regrets.values()) / len(regrets),
                "fixture_scores": fixture_scores,
                "actual_mappings": sorted(
                    actual_mappings, key=lambda row: row["batch_id"]
                ),
            }
        )
    best_regret = min(row["maximum_within_family_regret"] for row in target_rows)
    # The frozen checkpoint rule is intentionally conservative: candidates
    # within the absolute 1% relative-improvement band are treated as tied, and
    # the actual earlier checkpoint wins.  Target fractions are only reporting
    # coordinates; emitted steps remain authoritative.
    near = [
        row
        for row in target_rows
        if row["maximum_within_family_regret"] - best_regret <= _DISCOVERY_TIE
    ]
    selected = min(
        near,
        key=lambda row: (
            max(mapping["step"] for mapping in row["actual_mappings"]),
            sum(mapping["step"] for mapping in row["actual_mappings"])
            / len(row["actual_mappings"]),
            row["target_fraction"],
        ),
    )
    return {
        "method": "minimize_D1_D2_maximum_within_family_regret",
        "mapping_rule": "nearest actual candidate; ties choose earlier step",
        "tie_band": float(_DISCOVERY_TIE),
        "tie_breaker": "earliest actual step among candidates within 0.01 of best",
        "cross_run_tie_breaker": (
            "minimum maximum mapped step, then minimum mean mapped step, "
            "then earliest target fraction"
        ),
        "target_fraction": float(selected["target_fraction"]),
        "maximum_within_family_regret": float(selected["maximum_within_family_regret"]),
        "mean_within_family_regret": float(selected["mean_within_family_regret"]),
        "fixture_scores": {
            key: float(value) for key, value in selected["fixture_scores"].items()
        },
        "actual_mappings": selected["actual_mappings"],
    }


def _base_discovery_body(
    *,
    policy: dict[str, Any],
    policy_file_sha: str,
    approval: dict[str, Any],
    approval_file_sha: str,
    state: dict[str, Any],
    aggregate_bindings: list[dict[str, Any]],
    decided_at_utc: str,
) -> dict[str, Any]:
    decided_at_utc = _timestamp(decided_at_utc, "decided_at_utc")
    if _timestamp_value(decided_at_utc, "decided_at_utc") <= _timestamp_value(
        approval["approved_at_utc"], "approved_at_utc"
    ):
        raise ValueError("discovery decision predates its policy approval")
    agent_policy = policy.get("schema") == 3
    governance = (
        {
            "decision_reviewer_actor": dict(approval["technical_reviewer_actor"]),
            "accountable_owner_identity": policy["accountable_owner_identity"],
            "owner_ratification_sha256": policy["owner_ratification_sha256"],
            "fixture_admission_envelope": dict(policy["fixture_admission_envelope"]),
            "discovery_execution_authorization": dict(
                policy["discovery_execution_authorization"]
            ),
            "delegated_review_contract": krea_delegated_review_contract.binding(),
            "agent_review_is_not_human_review": True,
        }
        if agent_policy
        else {"decision_reviewer_identity": approval["reviewer_identity"]}
    )
    return {
        "schema": 3 if agent_policy else 2,
        "kind": (
            "forge-krea-agent-discovery-decision-record"
            if agent_policy
            else "forge-krea-discovery-decision-record"
        ),
        "phase": "discovery",
        "decided_at_utc": decided_at_utc,
        "policy_sha256": policy["policy_sha256"],
        "policy_file_sha256": policy_file_sha,
        "policy_approval_sha256": approval["approval_sha256"],
        "policy_approval_file_sha256": approval_file_sha,
        **governance,
        "discovery_plan_file_sha256": policy["discovery_plan"]["sha256"],
        "confirmation_fixture_seal_sha256": state["seal"]["seal_sha256"],
        "aggregate_bindings": aggregate_bindings,
        "bootstrap": policy["bootstrap"],
        "production_mutation_authorized": False,
        "release_review_required": True,
    }


def _publish_discovery_record(output: Path, body: dict[str, Any]) -> dict[str, Any]:
    record = _publish_record(output, body)
    _validate_discovery_record(record)
    return record


def decide(
    *,
    policy_path: Path,
    approval_path: Path,
    aggregate_paths: Iterable[Path],
    output: Path,
    decided_at_utc: str | None = None,
) -> dict[str, Any]:
    """Run discovery only; this function can never authorize promotion."""

    policy, policy_file_sha, approval, approval_file_sha, state = _load_policy_approval(
        policy_path=policy_path,
        approval_path=approval_path,
        confirmation=False,
    )
    observed, aggregate_bindings = _match_aggregates(
        policy=policy, aggregate_paths=aggregate_paths
    )
    base = _base_discovery_body(
        policy=policy,
        policy_file_sha=policy_file_sha,
        approval=approval,
        approval_file_sha=approval_file_sha,
        state=state,
        aggregate_bindings=aggregate_bindings,
        decided_at_utc=decided_at_utc or _now(),
    )
    by_role = {
        (row["fixture_id"], row["seed_role"]): row for row in policy["score_batches"]
    }
    missing_a = [
        by_role[(fixture, "A")]["batch_id"]
        for fixture in _DISCOVERY_FIXTURES
        if by_role[(fixture, "A")]["batch_id"] not in observed
    ]
    if missing_a:
        return _publish_discovery_record(
            output,
            {
                **base,
                "outcome": "no-go",
                "blockers": [
                    f"missing required Seed-A batch: {item}" for item in missing_a
                ],
                "seed_b_trigger": None,
                "curve_results": {},
                "family_results": [],
                "finalist_family_ids": [],
                "checkpoint_rules": {},
                "all_family_checkpoint_rules": {},
            },
        )

    arm_ids = state["plan"]["arm_ids"]
    noncontrols = [family for family in arm_ids if family != _CONTROL_FAMILY]
    analyses = _analysis_by_role(observed, expected_arm_ids=arm_ids)
    seed_a_scores = _concept_family_scores(
        analyses,
        fixtures=_DISCOVERY_FIXTURES,
        seed_roles=("A",),
        family_ids=arm_ids,
    )
    inside_by_fixture = {}
    for fixture in _DISCOVERY_FIXTURES:
        best = max(seed_a_scores[family][fixture] for family in noncontrols)
        inside_by_fixture[fixture] = sorted(
            family
            for family in noncontrols
            if best - seed_a_scores[family][fixture] <= _DISCOVERY_TIE
        )
    inside = sorted(set().union(*inside_by_fixture.values()))
    reversals = _material_rank_reversal(seed_a_scores, noncontrols=noncontrols)
    trigger_reasons = []
    if any(len(families) >= 3 for families in inside_by_fixture.values()):
        trigger_reasons.append("three_or_more_noncontrols_inside_0.01_band")
    if reversals:
        trigger_reasons.append("material_D1_D2_rank_reversal")
    seed_b_triggered = bool(trigger_reasons)
    b_rows = {
        (row["fixture_id"], row["seed_role"]): row
        for row in policy["score_batches"]
        if row["seed_role"] == "B"
    }
    missing_b = [
        b_rows.get((fixture, "B"), {}).get("batch_id", f"{fixture}/B-not-predeclared")
        for fixture in _DISCOVERY_FIXTURES
        if (fixture, "B") not in b_rows
        or b_rows[(fixture, "B")]["batch_id"] not in observed
    ]
    trigger_record = {
        "triggered": seed_b_triggered,
        "reasons": trigger_reasons,
        "noncontrols_inside_band": inside,
        "noncontrols_inside_band_by_fixture": inside_by_fixture,
        "material_reversals": reversals,
    }
    curve_results = {
        analysis["batch_id"]: {
            family: data["curve"] for family, data in analysis["curves"].items()
        }
        for analysis in analyses.values()
        if analysis["batch_id"] in observed
    }
    if seed_b_triggered and missing_b:
        return _publish_discovery_record(
            output,
            {
                **base,
                "outcome": "seed_b_required",
                "blockers": [
                    f"missing required Seed-B batch: {item}" for item in missing_b
                ],
                "seed_b_trigger": trigger_record,
                "curve_results": curve_results,
                "family_results": [],
                "finalist_family_ids": [],
                "checkpoint_rules": {},
                "all_family_checkpoint_rules": {},
            },
        )

    used_seed_roles = ("A", "B") if seed_b_triggered else ("A",)
    concept_scores = _concept_family_scores(
        analyses,
        fixtures=_DISCOVERY_FIXTURES,
        seed_roles=used_seed_roles,
        family_ids=arm_ids,
    )
    family_results = []
    for family in arm_ids:
        values = concept_scores[family]
        family_results.append(
            {
                "family_id": family,
                "control": family == _CONTROL_FAMILY,
                "concept_relative_improvement": {
                    key: float(value) for key, value in values.items()
                },
                "worst_case_relative_improvement": float(min(values.values())),
                "mean_relative_improvement": float(sum(values.values()) / len(values)),
                "paired_cluster_bootstrap": _bootstrap_ci(
                    values, label=f"discovery:{family}:{','.join(used_seed_roles)}"
                ),
            }
        )
    winners = {
        fixture: min(
            noncontrols,
            key=lambda family: (-concept_scores[family][fixture], family),
        )
        for fixture in _DISCOVERY_FIXTURES
    }
    best_by_fixture = {
        fixture: max(concept_scores[family][fixture] for family in noncontrols)
        for fixture in _DISCOVERY_FIXTURES
    }
    regret = {
        family: max(
            best_by_fixture[fixture] - concept_scores[family][fixture]
            for fixture in _DISCOVERY_FIXTURES
        )
        for family in noncontrols
    }
    finalists: list[str] = []

    def add(family: str) -> None:
        if family not in finalists:
            finalists.append(family)

    add(winners["D1"])
    add(winners["D2"])
    remaining = [family for family in noncontrols if family not in finalists]
    if remaining:
        add(min(remaining, key=lambda family: (regret[family], family)))
    add(_CONTROL_FAMILY)
    if len(finalists) > 4:
        raise RuntimeError("frozen finalist rule exceeded maximum_finalists=4")
    all_checkpoint_rules = {
        family: _checkpoint_rule(
            family,
            analyses=analyses,
            fixtures=_DISCOVERY_FIXTURES,
            seed_roles=used_seed_roles,
            targets=state["plan"]["report_targets"],
        )
        for family in arm_ids
    }
    checkpoint_rules = {family: all_checkpoint_rules[family] for family in finalists}
    record = _publish_discovery_record(
        output,
        {
            **base,
            "outcome": "finalists_frozen",
            "blockers": [],
            "seed_b_trigger": trigger_record,
            "seeds_used": list(used_seed_roles),
            "curve_results": curve_results,
            "family_results": sorted(family_results, key=lambda row: row["family_id"]),
            "D1_winner_family_id": winners["D1"],
            "D2_winner_family_id": winners["D2"],
            "minimax_regret": {
                family: float(value) for family, value in sorted(regret.items())
            },
            "finalist_family_ids": finalists,
            "checkpoint_rules": checkpoint_rules,
            "all_family_checkpoint_rules": all_checkpoint_rules,
        },
    )
    return record


decide_discovery = decide


def _locked_candidate(
    aggregate: Mapping[str, Any],
    *,
    family_id: str,
    checkpoint_rule: Mapping[str, Any],
    confirmation: bool,
) -> dict[str, Any]:
    rows = [
        row
        for row in aggregate["candidates"]
        if row["mode"] == "local_run_candidate" and row["family_id"] == family_id
    ]
    if not rows:
        raise ValueError(f"aggregate lacks frozen family {family_id}")
    fractions = [_candidate_fraction(row) for row in rows]
    if len(fractions) != len(set(fractions)):
        raise ValueError(f"family {family_id} repeats a checkpoint fraction")
    if confirmation:
        if len(rows) > 3:
            raise ValueError(
                f"confirmation family {family_id} exceeds locked/final/one-guard scope"
            )
        final_rows = [row for row in rows if _candidate_fraction(row) == Decimal("1")]
        if len(final_rows) != 1:
            raise ValueError(
                f"confirmation family {family_id} lacks exactly one natural final"
            )
    target = _decimal(
        checkpoint_rule["target_fraction"],
        f"{family_id} frozen checkpoint target",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    selected = _map_target(rows, target)
    return {
        "family_id": family_id,
        "target_fraction": float(target),
        "candidate": selected,
        "candidate_public": _public_candidate(selected),
    }


def _relative_reduction(
    candidate: Mapping[str, Any], control: Mapping[str, Any]
) -> Decimal:
    control_loss = control["weighted_loss"]
    if control_loss <= 0:
        raise ValueError("control weighted loss must be positive")
    return (control_loss - candidate["weighted_loss"]) / control_loss


def _relative_excess(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> Decimal:
    reference_loss = reference["weighted_loss"]
    if reference_loss <= 0:
        raise ValueError("reference weighted loss must be positive")
    return (candidate["weighted_loss"] - reference_loss) / reference_loss


def _validate_public_candidate(value: Any, label: str) -> dict[str, Any]:
    row = _object(value, label)
    _exact(
        row,
        {
            "candidate_id",
            "candidate_sha256",
            "step",
            "fraction_numerator",
            "fraction_denominator",
            "fraction",
            "image_exposures",
            "weighted_loss",
        },
        label,
    )
    _identifier(row["candidate_id"], f"{label}.candidate_id")
    _digest(row["candidate_sha256"], f"{label}.candidate_sha256")
    step = _positive_int(row["step"], f"{label}.step")
    numerator = _positive_int(row["fraction_numerator"], f"{label}.fraction_numerator")
    denominator = _positive_int(
        row["fraction_denominator"], f"{label}.fraction_denominator"
    )
    exposures = _positive_int(row["image_exposures"], f"{label}.image_exposures")
    fraction = _decimal(
        row["fraction"],
        f"{label}.fraction",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    loss = _finite_loss(row["weighted_loss"], f"{label}.weighted_loss")
    if (
        numerator != step
        or denominator < numerator
        or abs(fraction - Decimal(numerator) / Decimal(denominator)) > Decimal("1e-12")
    ):
        raise ValueError(f"{label} has inconsistent step/fraction")
    return {
        **row,
        "step": step,
        "fraction_numerator": numerator,
        "fraction_denominator": denominator,
        "fraction": fraction,
        "image_exposures": exposures,
        "weighted_loss": loss,
    }


def _validate_confirmation_record(value: dict[str, Any]) -> dict[str, Any]:
    value = _object(value, "confirmation decision record")
    body = {key: item for key, item in value.items() if key != "decision_sha256"}
    required = {
        "schema",
        "kind",
        "phase",
        "decided_at_utc",
        "policy_sha256",
        "policy_file_sha256",
        "policy_approval_sha256",
        "policy_approval_file_sha256",
        "decision_reviewer_identity",
        "discovery_plan_file_sha256",
        "discovery_decision_sha256",
        "confirmation_fixture_seal_sha256",
        "aggregate_bindings",
        "bootstrap",
        "candidate_family_id",
        "outcome",
        "blockers",
        "metrics",
        "gates",
        "boundary_results",
        "field_parity_ready",
        "round1_ready",
        "win_ready",
        "production_mutation_authorized",
        "release_review_required",
        "decision_sha256",
    }
    _exact(value, required, "confirmation decision record")
    if (
        value["schema"] != 2
        or value["kind"] != "forge-krea-confirmation-decision-record"
        or value["phase"] != "confirmation"
        or value["outcome"] not in {"PASS", "FAIL", "no-go"}
        or value["production_mutation_authorized"] is not False
        or value["release_review_required"] is not True
        or value["win_ready"] is not False
        or value["decision_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("confirmation decision record is invalid")
    _timestamp(value["decided_at_utc"], "confirmation decided_at_utc")
    for key in (
        "policy_sha256",
        "policy_file_sha256",
        "policy_approval_sha256",
        "policy_approval_file_sha256",
        "discovery_plan_file_sha256",
        "discovery_decision_sha256",
        "confirmation_fixture_seal_sha256",
    ):
        _digest(value[key], key)
    _named_human(value["decision_reviewer_identity"], "decision reviewer")
    _identifier(value["candidate_family_id"], "confirmation candidate family")
    _validate_bootstrap(value["bootstrap"])
    bindings = _validate_decision_aggregate_bindings(value["aggregate_bindings"])
    if not isinstance(value["blockers"], list) or any(
        not isinstance(item, str) or not item for item in value["blockers"]
    ):
        raise ValueError("confirmation blockers are invalid")
    if not isinstance(value["metrics"], dict) or not isinstance(value["gates"], dict):
        raise ValueError("confirmation metrics/gates are invalid")
    if not isinstance(value["boundary_results"], list):
        raise ValueError("confirmation boundary results are invalid")
    expected_ready = value["outcome"] == "PASS"
    if (
        value["field_parity_ready"] is not expected_ready
        or value["round1_ready"] is not expected_ready
    ):
        raise ValueError("confirmation readiness flags disagree with outcome")
    if value["outcome"] == "no-go":
        if not value["blockers"] or value["metrics"] or value["gates"]:
            raise ValueError("no-go must be blocked and must not manufacture metrics")
        if value["boundary_results"] != []:
            raise ValueError("no-go cannot claim boundary results")
    else:
        expected_gates = {
            "all_confirmation_runs_complete",
            "control_superiority_95pct",
            "public_reference_noninferiority_95pct",
            "three_of_four_point_wins_or_ties",
            "no_concept_regression_over_0.03",
            "boundary_matrix_clean",
            "decision_before_export_reserve_without_fallback",
            "stage2_production_surface_ratified",
        }
        if set(value["gates"]) != expected_gates or any(
            not isinstance(item, bool) for item in value["gates"].values()
        ):
            raise ValueError("confirmation gate set is incomplete")
        if expected_ready != all(value["gates"].values()):
            raise ValueError("confirmation PASS/FAIL disagrees with its gates")
        if bool(value["blockers"]) == expected_ready:
            raise ValueError("confirmation blockers disagree with outcome")
        _recompute_confirmation_record(value, bindings=bindings)
    return value


def _recompute_confirmation_record(
    value: Mapping[str, Any], *, bindings: Mapping[str, Mapping[str, Any]]
) -> None:
    metrics = _object(value["metrics"], "confirmation metrics")
    _exact(
        metrics,
        {
            "control_relative_reduction_cluster_bootstrap",
            "public_reference_relative_excess_cluster_bootstrap",
            "point_wins_or_ties",
            "point_win_or_tie_cap",
            "concept_regression_cap",
            "concept_results",
            "strongest_public_reference_rule",
        },
        "confirmation metrics",
    )
    if (
        metrics["point_win_or_tie_cap"] != float(_FIELD_CAP)
        or metrics["concept_regression_cap"] != float(_CONCEPT_CAP)
        or metrics["strongest_public_reference_rule"]
        != _CONFIRMATION_CAMPAIGN_CONTRACT["strongest_public_reference_rule"]
    ):
        raise ValueError("confirmation metrics changed a frozen threshold")
    concepts = metrics["concept_results"]
    if not isinstance(concepts, list) or [
        row.get("fixture_id") if isinstance(row, dict) else None for row in concepts
    ] != list(_CONFIRMATION_FIXTURES):
        raise ValueError("confirmation metrics must contain ordered C1-C4")
    concept_reduction: dict[str, Decimal] = {}
    concept_excess: dict[str, Decimal] = {}
    quality_batch_ids = set()
    for concept_index, raw_concept in enumerate(concepts):
        label = f"concept_results[{concept_index}]"
        concept = _object(raw_concept, label)
        _exact(
            concept,
            {
                "fixture_id",
                "relative_reduction_vs_K0",
                "relative_excess_vs_public_reference",
                "point_win_or_tie",
                "episodes",
            },
            label,
        )
        fixture = concept["fixture_id"]
        episodes = concept["episodes"]
        if not isinstance(episodes, list) or len(episodes) != 2:
            raise ValueError(f"{label} must contain paired Seed A/B episodes")
        reductions = []
        excesses = []
        observed_roles = []
        for episode_index, raw_episode in enumerate(episodes):
            episode_label = f"{label}.episodes[{episode_index}]"
            episode = _object(raw_episode, episode_label)
            _exact(
                episode,
                {
                    "batch_id",
                    "fixture_id",
                    "seed_role",
                    "candidate",
                    "control",
                    "public_reference_candidates",
                    "strongest_public_reference_family_id",
                    "strongest_public_reference",
                    "relative_reduction_vs_K0",
                    "relative_excess_vs_public_reference",
                },
                episode_label,
            )
            batch_id = _identifier(episode["batch_id"], f"{episode_label}.batch_id")
            quality_batch_ids.add(batch_id)
            if (
                episode["fixture_id"] != fixture
                or episode["seed_role"] not in {"A", "B"}
                or batch_id not in bindings
                or bindings[batch_id]["phase"] != "confirmation"
                or bindings[batch_id]["fixture_id"] != fixture
                or bindings[batch_id]["seed_role"] != episode["seed_role"]
            ):
                raise ValueError("confirmation episode escaped its bound concept/seed")
            observed_roles.append(episode["seed_role"])
            candidate = _validate_public_candidate(
                episode["candidate"], f"{episode_label}.candidate"
            )
            control = _validate_public_candidate(
                episode["control"], f"{episode_label}.control"
            )
            raw_references = _object(
                episode["public_reference_candidates"],
                f"{episode_label}.public_reference_candidates",
            )
            if set(raw_references) != set(_PUBLIC_FAMILIES):
                raise ValueError("confirmation episode lacks exhaustive K2-K4")
            references = {
                family: _validate_public_candidate(
                    raw_references[family], f"{episode_label}.references.{family}"
                )
                for family in _PUBLIC_FAMILIES
            }
            strongest_family = min(
                _PUBLIC_FAMILIES,
                key=lambda family: (references[family]["weighted_loss"], family),
            )
            strongest = _validate_public_candidate(
                episode["strongest_public_reference"],
                f"{episode_label}.strongest_public_reference",
            )
            if (
                episode["strongest_public_reference_family_id"] != strongest_family
                or strongest != references[strongest_family]
            ):
                raise ValueError("strongest public reference does not recompute")
            reduction = _relative_reduction(candidate, control)
            excess = _relative_excess(candidate, references[strongest_family])
            recorded_reduction = _decimal(
                episode["relative_reduction_vs_K0"],
                f"{episode_label}.reduction",
                minimum=Decimal("-100"),
                maximum=Decimal("1"),
            )
            recorded_excess = _decimal(
                episode["relative_excess_vs_public_reference"],
                f"{episode_label}.excess",
                minimum=Decimal("-1"),
                maximum=Decimal("100"),
            )
            if abs(recorded_reduction - reduction) > Decimal("1e-10") or abs(
                recorded_excess - excess
            ) > Decimal("1e-10"):
                raise ValueError("confirmation episode metrics do not recompute")
            reductions.append(reduction)
            excesses.append(excess)
        if observed_roles != ["A", "B"]:
            raise ValueError("confirmation episodes must be ordered Seed A then B")
        reduction = sum(reductions) / len(reductions)
        excess = sum(excesses) / len(excesses)
        recorded_reduction = _decimal(
            concept["relative_reduction_vs_K0"],
            f"{label}.reduction",
            minimum=Decimal("-100"),
            maximum=Decimal("1"),
        )
        recorded_excess = _decimal(
            concept["relative_excess_vs_public_reference"],
            f"{label}.excess",
            minimum=Decimal("-1"),
            maximum=Decimal("100"),
        )
        if (
            abs(recorded_reduction - reduction) > Decimal("1e-10")
            or abs(recorded_excess - excess) > Decimal("1e-10")
            or concept["point_win_or_tie"] is not (excess <= _FIELD_CAP)
        ):
            raise ValueError("confirmation concept metrics do not recompute")
        concept_reduction[fixture] = reduction
        concept_excess[fixture] = excess
    expected_control_ci = _bootstrap_ci(
        concept_reduction,
        label=f"confirmation:{value['candidate_family_id']}:vs:K0",
    )
    expected_public_ci = _bootstrap_ci(
        concept_excess,
        label=f"confirmation:{value['candidate_family_id']}:vs:K2-K4",
    )
    for actual, expected_ci, label in (
        (
            metrics["control_relative_reduction_cluster_bootstrap"],
            expected_control_ci,
            "control bootstrap",
        ),
        (
            metrics["public_reference_relative_excess_cluster_bootstrap"],
            expected_public_ci,
            "public bootstrap",
        ),
    ):
        actual = _object(actual, label)
        _exact(actual, {"point_estimate", "lower", "upper"}, label)
        if any(
            abs(Decimal(str(actual[key])) - Decimal(str(expected_ci[key])))
            > Decimal("1e-10")
            for key in ("point_estimate", "lower", "upper")
        ):
            raise ValueError(f"{label} does not recompute")
    wins = sum(excess <= _FIELD_CAP for excess in concept_excess.values())
    if metrics["point_wins_or_ties"] != wins:
        raise ValueError("confirmation point-win count does not recompute")

    boundary_batch_ids = set()
    cells = []
    for index, raw_boundary in enumerate(value["boundary_results"]):
        label = f"boundary_results[{index}]"
        boundary = _object(raw_boundary, label)
        _exact(
            boundary,
            {
                "batch_id",
                "fixture_id",
                "hours",
                "dataset_boundary",
                "candidate",
                "mechanics",
                "candidate_decision",
                "passed",
            },
            label,
        )
        batch_id = _identifier(boundary["batch_id"], f"{label}.batch_id")
        boundary_batch_ids.add(batch_id)
        _identifier(boundary["fixture_id"], f"{label}.fixture_id")
        hours = _decimal(
            boundary["hours"],
            f"{label}.hours",
            minimum=Decimal("0.5"),
            maximum=Decimal("1"),
        )
        dataset_boundary = _identifier(
            boundary["dataset_boundary"], f"{label}.dataset_boundary"
        )
        if (
            batch_id not in bindings
            or bindings[batch_id]["phase"] != "boundary"
            or bindings[batch_id]["fixture_id"] != boundary["fixture_id"]
            or Decimal(str(bindings[batch_id]["hours"])) != hours
            or bindings[batch_id]["dataset_boundary"] != dataset_boundary
        ):
            raise ValueError("boundary result escaped its aggregate binding")
        candidate = _validate_public_candidate(
            boundary["candidate"], f"{label}.candidate"
        )
        mechanics = _object(boundary["mechanics"], f"{label}.mechanics")
        decision = _object(
            boundary["candidate_decision"], f"{label}.candidate_decision"
        )
        if (
            mechanics
            != {
                "natural_completion": True,
                "upload_ready": True,
                "clean_telemetry": True,
            }
            or decision
            != {
                "mode": "frozen_checkpoint_rule",
                "selected_candidate_sha256": candidate["candidate_sha256"],
                "decision_completed_before_export_reserve": True,
                "fallback_used": False,
            }
            or boundary["passed"] is not True
        ):
            raise ValueError("boundary mechanics/decision did not pass")
        cells.append((float(hours), dataset_boundary))
    expected_cells = {
        (float(hours), boundary)
        for hours in (Decimal("0.5"), Decimal("0.75"), Decimal("1.0"))
        for boundary in ("small", "large")
    }
    if set(cells) != expected_cells or len(cells) != len(expected_cells):
        raise ValueError("confirmation record lacks the complete 3x2 boundary matrix")
    if (
        quality_batch_ids | boundary_batch_ids != set(bindings)
        or quality_batch_ids & boundary_batch_ids
    ):
        raise ValueError(
            "confirmation record does not cover every bound aggregate once"
        )

    expected_gates = {
        "all_confirmation_runs_complete": len(quality_batch_ids) == 8,
        "control_superiority_95pct": Decimal(str(expected_control_ci["lower"])) > 0,
        "public_reference_noninferiority_95pct": Decimal(
            str(expected_public_ci["upper"])
        )
        <= _FIELD_CAP,
        "three_of_four_point_wins_or_ties": wins >= 3,
        "no_concept_regression_over_0.03": max(concept_excess.values()) <= _CONCEPT_CAP,
        "boundary_matrix_clean": len(boundary_batch_ids) == 6,
        "decision_before_export_reserve_without_fallback": len(boundary_batch_ids) == 6,
        "stage2_production_surface_ratified": False,
    }
    if value["gates"] != expected_gates:
        raise ValueError("confirmation gates do not recompute")
    expected_blockers = [
        f"failed confirmation gate: {name}"
        for name, passed in expected_gates.items()
        if not passed
    ]
    if value["blockers"] != expected_blockers:
        raise ValueError("confirmation blockers do not recompute")


def _base_confirmation_body(
    *,
    policy: dict[str, Any],
    policy_file_sha: str,
    approval: dict[str, Any],
    approval_file_sha: str,
    state: dict[str, Any],
    aggregate_bindings: list[dict[str, Any]],
    decided_at_utc: str,
) -> dict[str, Any]:
    decided = _timestamp(decided_at_utc, "decided_at_utc")
    if _timestamp_value(decided, "decided_at_utc") <= _timestamp_value(
        approval["approved_at_utc"], "approved_at_utc"
    ):
        raise ValueError("confirmation decision did not follow policy approval")
    _, discovery, discovery_file_sha = _binding(
        policy["discovery_decision"], "discovery decision"
    )
    return {
        "schema": 2,
        "kind": "forge-krea-confirmation-decision-record",
        "phase": "confirmation",
        "decided_at_utc": decided,
        "policy_sha256": policy["policy_sha256"],
        "policy_file_sha256": policy_file_sha,
        "policy_approval_sha256": approval["approval_sha256"],
        "policy_approval_file_sha256": approval_file_sha,
        "decision_reviewer_identity": approval["reviewer_identity"],
        "discovery_plan_file_sha256": policy["discovery_plan"]["sha256"],
        "discovery_decision_sha256": discovery_file_sha,
        "confirmation_fixture_seal_sha256": state["seal"]["seal_sha256"],
        "aggregate_bindings": aggregate_bindings,
        "bootstrap": policy["bootstrap"],
        "candidate_family_id": policy["candidate_family_id"],
        "production_mutation_authorized": False,
        "release_review_required": True,
        "win_ready": False,
    }


def _publish_confirmation_record(output: Path, body: dict[str, Any]) -> dict[str, Any]:
    record = _publish_record(output, body)
    _validate_confirmation_record(record)
    return record


def decide_confirmation(
    *,
    policy_path: Path,
    approval_path: Path,
    aggregate_paths: Iterable[Path],
    output: Path,
    decided_at_utc: str | None = None,
) -> dict[str, Any]:
    """Evaluate the sealed confirmation matrix without authorizing deployment."""

    policy, policy_file_sha, approval, approval_file_sha, state = _load_policy_approval(
        policy_path=policy_path,
        approval_path=approval_path,
        confirmation=True,
    )
    observed, aggregate_bindings = _match_aggregates(
        policy=policy, aggregate_paths=aggregate_paths
    )
    base = _base_confirmation_body(
        policy=policy,
        policy_file_sha=policy_file_sha,
        approval=approval,
        approval_file_sha=approval_file_sha,
        state=state,
        aggregate_bindings=aggregate_bindings,
        decided_at_utc=decided_at_utc or _now(),
    )
    missing = [
        row["batch_id"]
        for row in policy["score_batches"]
        if row["batch_id"] not in observed
    ]
    if missing:
        return _publish_confirmation_record(
            output,
            {
                **base,
                "outcome": "no-go",
                "blockers": [
                    f"missing required confirmation batch: {item}" for item in missing
                ],
                "metrics": {},
                "gates": {},
                "boundary_results": [],
                "field_parity_ready": False,
                "round1_ready": False,
            },
        )

    _, discovery, _ = _binding(policy["discovery_decision"], "discovery decision")
    all_rules = discovery["all_family_checkpoint_rules"]
    candidate_family = policy["candidate_family_id"]
    quality_by_role: dict[tuple[str, str], dict[str, Any]] = {}
    for batch in observed.values():
        expected = batch["batch"]
        if expected["phase"] != "confirmation":
            continue
        family_ids = sorted(
            set(discovery["finalist_family_ids"])
            | set(_PUBLIC_FAMILIES)
            | {_CONTROL_FAMILY}
        )
        locked = {
            family: _locked_candidate(
                batch,
                family_id=family,
                checkpoint_rule=all_rules[family],
                confirmation=True,
            )
            for family in family_ids
        }
        references = [locked[family] for family in _PUBLIC_FAMILIES]
        strongest = min(
            references,
            key=lambda item: (item["candidate"]["weighted_loss"], item["family_id"]),
        )
        candidate = locked[candidate_family]["candidate"]
        control = locked[_CONTROL_FAMILY]["candidate"]
        reference = strongest["candidate"]
        role = (expected["fixture_id"], expected["seed_role"])
        quality_by_role[role] = {
            "batch_id": expected["batch_id"],
            "fixture_id": expected["fixture_id"],
            "seed_role": expected["seed_role"],
            "candidate": locked[candidate_family]["candidate_public"],
            "control": locked[_CONTROL_FAMILY]["candidate_public"],
            "public_reference_candidates": {
                item["family_id"]: item["candidate_public"] for item in references
            },
            "strongest_public_reference_family_id": strongest["family_id"],
            "strongest_public_reference": strongest["candidate_public"],
            "relative_reduction_vs_K0": _relative_reduction(candidate, control),
            "relative_excess_vs_public_reference": _relative_excess(
                candidate, reference
            ),
        }

    concept_reduction: dict[str, Decimal] = {}
    concept_excess: dict[str, Decimal] = {}
    concept_results = []
    for fixture in _CONFIRMATION_FIXTURES:
        episodes = [quality_by_role[(fixture, role)] for role in ("A", "B")]
        reduction = sum(
            episode["relative_reduction_vs_K0"] for episode in episodes
        ) / len(episodes)
        excess = sum(
            episode["relative_excess_vs_public_reference"] for episode in episodes
        ) / len(episodes)
        concept_reduction[fixture] = reduction
        concept_excess[fixture] = excess
        concept_results.append(
            {
                "fixture_id": fixture,
                "relative_reduction_vs_K0": float(reduction),
                "relative_excess_vs_public_reference": float(excess),
                "point_win_or_tie": excess <= _FIELD_CAP,
                "episodes": [
                    {
                        **{
                            key: value
                            for key, value in episode.items()
                            if key
                            not in {
                                "relative_reduction_vs_K0",
                                "relative_excess_vs_public_reference",
                            }
                        },
                        "relative_reduction_vs_K0": float(
                            episode["relative_reduction_vs_K0"]
                        ),
                        "relative_excess_vs_public_reference": float(
                            episode["relative_excess_vs_public_reference"]
                        ),
                    }
                    for episode in episodes
                ],
            }
        )
    control_ci = _bootstrap_ci(
        concept_reduction, label=f"confirmation:{candidate_family}:vs:K0"
    )
    public_ci = _bootstrap_ci(
        concept_excess, label=f"confirmation:{candidate_family}:vs:K2-K4"
    )

    boundary_results = []
    for batch in observed.values():
        expected = batch["batch"]
        if expected["phase"] != "boundary":
            continue
        selected = next(
            row for row in batch["candidates"] if row["mode"] == "local_run_candidate"
        )
        envelope = batch["training_run_envelopes"][0]
        boundary_results.append(
            {
                "batch_id": expected["batch_id"],
                "fixture_id": expected["fixture_id"],
                "hours": expected["hours"],
                "dataset_boundary": expected["dataset_boundary"],
                "candidate": _public_candidate(selected),
                "mechanics": dict(selected["mechanics"]),
                "candidate_decision": dict(envelope["candidate_decision"]),
                "passed": True,
            }
        )
    boundary_results.sort(key=lambda row: (row["hours"], row["dataset_boundary"]))
    wins_or_ties = sum(item["point_win_or_tie"] for item in concept_results)
    gates = {
        "all_confirmation_runs_complete": len(quality_by_role) == 8,
        "control_superiority_95pct": Decimal(str(control_ci["lower"])) > 0,
        "public_reference_noninferiority_95pct": (
            Decimal(str(public_ci["upper"])) <= _FIELD_CAP
        ),
        "three_of_four_point_wins_or_ties": wins_or_ties >= 3,
        "no_concept_regression_over_0.03": max(concept_excess.values()) <= _CONCEPT_CAP,
        "boundary_matrix_clean": len(boundary_results) == 6
        and all(item["passed"] for item in boundary_results),
        "decision_before_export_reserve_without_fallback": len(boundary_results) == 6
        and all(
            item["candidate_decision"]["decision_completed_before_export_reserve"]
            and not item["candidate_decision"]["fallback_used"]
            for item in boundary_results
        ),
        "stage2_production_surface_ratified": False,
    }
    passed = all(gates.values())
    blockers = [
        f"failed confirmation gate: {name}" for name, ok in gates.items() if not ok
    ]
    metrics = {
        "control_relative_reduction_cluster_bootstrap": control_ci,
        "public_reference_relative_excess_cluster_bootstrap": public_ci,
        "point_wins_or_ties": wins_or_ties,
        "point_win_or_tie_cap": float(_FIELD_CAP),
        "concept_regression_cap": float(_CONCEPT_CAP),
        "concept_results": concept_results,
        "strongest_public_reference_rule": _CONFIRMATION_CAMPAIGN_CONTRACT[
            "strongest_public_reference_rule"
        ],
    }
    return _publish_confirmation_record(
        output,
        {
            **base,
            "outcome": "PASS" if passed else "FAIL",
            "blockers": blockers,
            "metrics": metrics,
            "gates": gates,
            "boundary_results": boundary_results,
            "field_parity_ready": passed,
            "round1_ready": passed,
        },
    )


def _cli_control(path: Path, label: str) -> dict[str, Any]:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    value = _object(value, label)
    if raw != krea_provenance.canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline")
    return value


def _cli_publish(path: Path, value: dict[str, Any]) -> None:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(krea_provenance.canonical_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal-discovery-policy")
    seal.add_argument("--payload", required=True, type=Path)
    seal.add_argument("--output", required=True, type=Path)
    approve = commands.add_parser("approve-discovery-policy")
    approve.add_argument("--policy", required=True, type=Path)
    approve.add_argument("--technical-actor", required=True, type=Path)
    approve.add_argument("--approved-at-utc", required=True)
    approve.add_argument("--output", required=True, type=Path)
    decide_parser = commands.add_parser("decide-discovery")
    decide_parser.add_argument("--policy", required=True, type=Path)
    decide_parser.add_argument("--approval", required=True, type=Path)
    decide_parser.add_argument("--aggregate", action="append", required=True, type=Path)
    decide_parser.add_argument("--decided-at-utc", required=True)
    decide_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "seal-discovery-policy":
        result = seal_discovery_policy(_cli_control(args.payload, "policy payload"))
        _cli_publish(args.output, result)
    elif args.command == "approve-discovery-policy":
        result = build_approval(
            _cli_control(args.policy, "discovery policy"),
            technical_reviewer_actor=_cli_control(
                args.technical_actor, "decision technical actor"
            ),
            approved_at_utc=args.approved_at_utc,
        )
        _cli_publish(args.output, result)
    else:
        result = decide_discovery(
            policy_path=args.policy,
            approval_path=args.approval,
            aggregate_paths=args.aggregate,
            output=args.output,
            decided_at_utc=args.decided_at_utc,
        )
    print(krea_provenance.canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
