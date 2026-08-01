#!/usr/bin/env python3
"""Fail-closed finalist freeze for the authorised Seed-A density set + Seed-B.

This adapter exists because the legacy recovery freeze requires all 92 Seed-A
scores and therefore cannot represent the pre-authorised 59/68/70 density
contracts.  It consumes only receipt-validated D1/D2 discovery evidence.  It
never reads sealed C1-C4 fixtures and grants no confirmation, release,
production, or deployment authority.

The scientific policy remains the one pinned before scoring:

* choose the largest receipt-complete density plan at the fixed cutoff;
* derive checkpoint curves/rules from Seed-A only;
* pool Seed-B only as an all-or-none, exact-coordinate replication anchor;
* apply the inclusive absolute 0.01 family tie band, then depth, D1/D2 spread,
  and K2 > K3 > K4 > K5 > K1, in that order.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

try:
    from . import krea_decision
    from . import krea_density_gate
    from . import krea_provenance
    from . import krea_recovery_evidence
    from . import krea_waiver_finalist_freeze
except ImportError:  # pragma: no cover - direct script execution.
    import krea_decision  # type: ignore[no-redef]
    import krea_density_gate  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_recovery_evidence  # type: ignore[no-redef]
    import krea_waiver_finalist_freeze  # type: ignore[no-redef]


SCHEMA = 2
FREEZE_KIND = "forge-krea-density-seedb-finalist-freeze"
CUTOFF_UTC = "2026-08-01T18:00:00Z"
DECISION_COMMIT = "39d8676a3022a57c192d87aa8876f4506e0e8345"
COMPATIBILITY_FIX_COMMIT = "80a898fc94bfd0e4655b1aa8fee1806362ae218e"
FAILURE_SCHEMA = "sn56.pin39.direct-a59-structural-failure.v1"
FAILURE_CLASS = "STRUCTURAL_GATE_CANNOT_REPRESENT_AUTHORIZED_DENSITY_SET"
SEEDB_SCHEMA = "sn56.seedb.score-bridge-results.v1"
SEEDB_KIND = "forge-krea-seedb-sparse-replication-results"

FIXTURES = ("D1", "D2")
FAMILIES = ("K0", "K1", "K2", "K3", "K4", "K5")
NONCONTROLS = ("K1", "K2", "K3", "K4", "K5")
PREFERENCE = ("K2", "K3", "K4", "K5", "K1")
TIE_BAND = Decimal("0.01")
REPORT_TARGETS = tuple(
    Decimal(value) for value in ("0.1", "0.25", "0.5", "0.75", "0.9", "1.0")
)
EXPECTED_PLAN_COUNTS = ((11, 70), (9, 68), (0, 59))
AUTHORITY = dict(krea_waiver_finalist_freeze.AUTHORITY)
FALSE_CLAIMS = dict(krea_waiver_finalist_freeze.FALSE_CLAIMS)

_SHA = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_PATH_PARTS = {"c1", "c2", "c3", "c4"}
_FAILURE_KEYS = {
    "schema",
    "state",
    "decision_commit",
    "compatibility_fix_commit",
    "failure_class",
    "selection_eligible",
    "expected_exact_scores",
    "pinned_gate_returncode",
    "started_utc",
    "started_unix_ns",
    "ended_utc",
    "ended_unix_ns",
    "recovery_index_file_sha256",
    "recovery_index_semantic_sha256",
    "coverage_ledger_file_sha256",
    "stdout_file_sha256",
    "stderr_file_sha256",
    "adapter_work_authorized_by_structural_failure",
    "b14_fabricated_or_required",
    "c1c4_accessed",
}
_B_RESULT_KEYS = {
    "schema",
    "kind",
    "created_at_utc",
    "plan",
    "row_count",
    "complete",
    "rows",
    "seed_a_seed_b_comparison",
    "claim_limits",
    "results_sha256",
}
_B_ROW_KEYS = {
    "ordinal",
    "task_id",
    "fixture",
    "family",
    "label",
    "step",
    "fraction",
    "candidate",
    "weighted_loss",
    "source_seed_a",
    "queue_descriptor",
    "known_state",
    "output_directory",
    "status",
    "result",
    "evidence",
    "validated",
}
_FILE_BINDING_KEYS = {"path", "bytes", "file_sha256"}
_B_VALIDATED_KEYS = {
    "returncode_zero",
    "candidate_bytes_and_sha256",
    "fixture_dataset_sha256",
    "fixture_manifest_and_exact_row_identity",
    "scored_rows_and_prompt_count",
    "evaluator_and_source_commits",
    "evidence_manifest_rehashed",
}
_B_COMPARISON_KEYS = {
    "fixture",
    "family",
    "step",
    "seed_a_relative_improvement",
    "seed_b_relative_improvement",
    "absolute_seed_delta",
    "replication_anchor",
    "pooling_eligible_at_replication_anchor",
    "final_checkpoint_comparison",
}


class DensitySeedBFreezeError(ValueError):
    """Raised when any discovery input or frozen output fails closed."""


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DensitySeedBFreezeError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise DensitySeedBFreezeError(
            f"{label} keys differ: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise DensitySeedBFreezeError(f"{label} must be a lowercase SHA-256")
    return value


def _timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise DensitySeedBFreezeError(f"{label} must be canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise DensitySeedBFreezeError(f"{label} must be canonical UTC") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise DensitySeedBFreezeError(f"{label} must be canonical UTC")
    return value


def _time(value: Any, label: str) -> datetime:
    return datetime.strptime(_timestamp(value, label), "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )


def _decimal(value: Any, label: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise DensitySeedBFreezeError(f"{label} must be finite numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise DensitySeedBFreezeError(f"{label} must be finite numeric") from exc
    if not result.is_finite() or (positive and result <= 0):
        raise DensitySeedBFreezeError(f"{label} must be finite numeric")
    return result


def _reject_c_path(path: Path, label: str) -> None:
    if {part.casefold() for part in path.parts} & _FORBIDDEN_PATH_PARTS:
        raise DensitySeedBFreezeError(f"{label} points at prohibited sealed C evidence")


def _reject_c_paths_in_value(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_c_paths_in_value(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_c_paths_in_value(item, f"{label}[{index}]")
    elif isinstance(value, str) and value.startswith("/"):
        _reject_c_path(Path(value), label)


def _safe_path(path_value: Path | str, label: str, *, must_exist: bool) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path_value))))
    _reject_c_path(path, label)
    current = path if must_exist else path.parent
    while current != current.parent:
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            current = current.parent
            continue
        if stat.S_ISLNK(mode):
            raise DensitySeedBFreezeError(f"{label} has a symlink component: {current}")
        current = current.parent
    if must_exist and not path.is_file():
        raise DensitySeedBFreezeError(f"{label} must be a regular file: {path}")
    return path


def _binding(path_value: Path | str, label: str) -> dict[str, Any]:
    path = _safe_path(path_value, label, must_exist=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise DensitySeedBFreezeError(f"{label} changed while read")
    return {
        "path": str(path),
        "bytes": after.st_size,
        "file_sha256": digest.hexdigest(),
    }


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DensitySeedBFreezeError(f"JSON repeats key: {key}")
        result[key] = value
    return result


def _load_json(
    path_value: Path | str, label: str, *, canonical: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _binding(path_value, label)
    raw = Path(binding["path"]).read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DensitySeedBFreezeError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise DensitySeedBFreezeError(f"{label} must be a JSON object")
    _reject_c_paths_in_value(value, label)
    if canonical and raw != krea_provenance.canonical_bytes(value) + b"\n":
        raise DensitySeedBFreezeError(f"{label} must be canonical JSON plus newline")
    return value, binding


def _load_env(path_value: Path | str, label: str) -> tuple[dict[str, str], dict[str, Any]]:
    binding = _binding(path_value, label)
    raw = Path(binding["path"]).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DensitySeedBFreezeError(f"{label} must be UTF-8") from exc
    if not text.endswith("\n"):
        raise DensitySeedBFreezeError(f"{label} lacks a terminal newline")
    result: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise DensitySeedBFreezeError(f"{label} contains malformed row")
        key, value = line.split("=", 1)
        if not key or key in result or "\x00" in value:
            raise DensitySeedBFreezeError(f"{label} contains duplicate/unsafe row")
        result[key] = value
    _reject_c_paths_in_value(result, label)
    return result, binding


def _validate_binding(value: Any, label: str) -> dict[str, Any]:
    binding = _object(value, label)
    _exact(binding, _FILE_BINDING_KEYS, label)
    if (
        isinstance(binding["bytes"], bool)
        or not isinstance(binding["bytes"], int)
        or binding["bytes"] < 0
    ):
        raise DensitySeedBFreezeError(f"{label} byte count is invalid")
    _digest(binding["file_sha256"], f"{label} file_sha256")
    observed = _binding(binding["path"], label)
    if observed != binding:
        raise DensitySeedBFreezeError(f"{label} bytes drifted")
    return dict(binding)


def _artifact_binding(
    path: Path | str, *, semantic_key: str, semantic_value: str, label: str
) -> dict[str, Any]:
    return {
        **_binding(path, label),
        semantic_key: _digest(semantic_value, f"{label} semantic SHA"),
    }


def _validate_failure(
    path: Path, *, a59_index: Mapping[str, Any], a59_binding: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    value, binding = _load_env(path, "structural failure record")
    _exact(value, _FAILURE_KEYS, "structural failure record")
    if (
        value["schema"] != FAILURE_SCHEMA
        or value["state"] != "EXPECTED_FAIL_CONFIRMED"
        or value["decision_commit"] != DECISION_COMMIT
        or value["compatibility_fix_commit"] != COMPATIBILITY_FIX_COMMIT
        or value["failure_class"] != FAILURE_CLASS
        or value["selection_eligible"] != "59"
        or value["expected_exact_scores"] != "92"
        or not value["pinned_gate_returncode"].isdigit()
        or int(value["pinned_gate_returncode"]) == 0
        or value["adapter_work_authorized_by_structural_failure"] != "true"
        or value["b14_fabricated_or_required"] != "false"
        or value["c1c4_accessed"] != "false"
        or value["recovery_index_file_sha256"] != a59_binding["file_sha256"]
        or value["recovery_index_semantic_sha256"] != a59_index["index_sha256"]
        or value["coverage_ledger_file_sha256"]
        != a59_index["coverage_ledger"]["file_sha256"]
    ):
        raise DensitySeedBFreezeError("structural failure identity or authority drifted")
    for key in (
        "recovery_index_file_sha256",
        "recovery_index_semantic_sha256",
        "coverage_ledger_file_sha256",
        "stdout_file_sha256",
        "stderr_file_sha256",
    ):
        _digest(value[key], key)
    if (
        not value["started_unix_ns"].isdigit()
        or not value["ended_unix_ns"].isdigit()
        or int(value["ended_unix_ns"]) <= int(value["started_unix_ns"])
        or _time(value["ended_utc"], "failure ended_utc")
        <= _time(value["started_utc"], "failure started_utc")
    ):
        raise DensitySeedBFreezeError("structural failure chronology is invalid")
    return value, {
        **binding,
        "failure_semantic_sha256": krea_provenance.canonical_sha256(value),
    }


def _load_density_triplet(
    label: str,
    *,
    plan_path: Path,
    sidecar_path: Path,
    decision_path: Path | None,
) -> dict[str, Any]:
    plan_value, plan_file = _load_json(plan_path, f"{label} plan", canonical=True)
    try:
        plan = krea_density_gate.validate_plan(plan_value)
    except (OSError, ValueError) as exc:
        raise DensitySeedBFreezeError(f"{label} plan does not replay: {exc}") from exc
    sidecar_value, sidecar_file = _load_json(
        sidecar_path, f"{label} sidecar", canonical=True
    )
    try:
        sidecar = krea_density_gate.validate_sidecar(plan, sidecar_value)
    except (OSError, ValueError) as exc:
        raise DensitySeedBFreezeError(f"{label} sidecar does not replay: {exc}") from exc
    additional = plan["selection_policy"].get("requested_additional_target_count")
    expected_count = dict(EXPECTED_PLAN_COUNTS).get(additional)
    if (
        plan["contract"] != krea_density_gate.TARGETED_CONTRACT
        or expected_count is None
        or plan["selected_count"] != expected_count
    ):
        raise DensitySeedBFreezeError(f"{label} density cardinality is not 59/68/70")
    result = {
        "label": label,
        "additional_target_count": additional,
        "selected_count": plan["selected_count"],
        "plan": plan,
        "sidecar": sidecar,
        "plan_binding": {
            **plan_file,
            "target_plan_sha256": plan["target_plan_sha256"],
        },
        "sidecar_binding": {
            **sidecar_file,
            "sidecar_sha256": sidecar["sidecar_sha256"],
        },
        "decision": None,
        "decision_binding": {
            "state": "absent_at_freeze",
            "path": None if decision_path is None else str(decision_path),
        },
    }
    if decision_path is not None:
        decision_value, decision_file = _load_json(
            decision_path, f"{label} decision input", canonical=True
        )
        try:
            decision = krea_density_gate.validate_decision_input(
                decision_value, plan_value=plan, sidecar_value=sidecar
            )
        except (OSError, ValueError) as exc:
            raise DensitySeedBFreezeError(
                f"{label} decision input does not replay: {exc}"
            ) from exc
        result["decision"] = decision
        result["decision_binding"] = {
            **decision_file,
            "decision_input_sha256": decision["decision_input_sha256"],
        }
    return result


def _cutoff_complete(bundle: Mapping[str, Any], cutoff: datetime) -> bool:
    decision = bundle["decision"]
    if decision is None:
        return False
    try:
        index, observed_file_sha = krea_recovery_evidence.load_index(
            Path(decision["final_recovery_index"]["path"])
        )
    except (OSError, ValueError) as exc:
        raise DensitySeedBFreezeError(
            f"{bundle['label']} final recovery index does not replay: {exc}"
        ) from exc
    binding = decision["final_recovery_index"]
    if (
        observed_file_sha != binding["file_sha256"]
        or index["index_sha256"] != binding["index_sha256"]
        or index["coverage_ledger"]["file_sha256"]
        != binding["coverage_ledger_file_sha256"]
    ):
        raise DensitySeedBFreezeError("density final recovery snapshot binding drifted")
    indexed = {row["task_id"]: row for row in index["artifacts"]}
    selected = {
        row["task_id"] for row in bundle["plan"]["rows"] if row["selected"]
    }
    if set(row["task_id"] for row in decision["candidate_rows_for_krea_decision"]) != selected:
        raise DensitySeedBFreezeError("density decision selected set drifted")
    for task_id in selected:
        artifact = indexed.get(task_id)
        validated = artifact.get("validated_artifact") if isinstance(artifact, dict) else None
        status = validated.get("status") if isinstance(validated, dict) else None
        if (
            artifact is None
            or artifact.get("selection_eligible") is not True
            or not isinstance(status, dict)
            or status.get("returncode") != 0
            or _time(status.get("ended_utc"), f"{task_id} ended_utc") > cutoff
        ):
            return False
    return True


def _analysis(decision: Mapping[str, Any]) -> tuple[
    dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]
]:
    rows = decision.get("candidate_rows_for_krea_decision")
    if not isinstance(rows, list):
        raise DensitySeedBFreezeError("density decision rows must be a list")
    by_fixture: dict[str, list[dict[str, Any]]] = {fixture: [] for fixture in FIXTURES}
    zeros: dict[str, dict[str, Any]] = {}
    public: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = _object(raw, "density decision row")
        task_id = row.get("task_id")
        fixture = row.get("fixture")
        family = row.get("family")
        if (
            not isinstance(task_id, str)
            or task_id in public
            or fixture not in FIXTURES
            or (family is not None and family not in FAMILIES)
            or row.get("seed_role") != "A"
        ):
            raise DensitySeedBFreezeError("density decision row identity drifted")
        projected = dict(row)
        projected["weighted_loss"] = _decimal(
            row.get("weighted_loss"), f"{task_id} weighted_loss", positive=True
        )
        public[task_id] = projected
        if family is None:
            if fixture in zeros or row.get("step") != 0:
                raise DensitySeedBFreezeError("density zero baseline is invalid")
            zeros[fixture] = projected
        else:
            final_step = krea_density_gate._GEOMETRY[(fixture, family)][1]
            if not isinstance(row.get("step"), int) or not 0 < row["step"] <= final_step:
                raise DensitySeedBFreezeError("density candidate step is invalid")
            by_fixture[fixture].append(
                {
                    "candidate_id": task_id,
                    "candidate_sha256": row["candidate_binding"]["file_sha256"],
                    "step": row["step"],
                    "image_exposures": row["image_exposures"],
                    "fraction_numerator": row["step"],
                    "fraction_denominator": final_step,
                    "weighted_loss": projected["weighted_loss"],
                    "mode": "local_run_candidate",
                    "family_id": family,
                }
            )
    if set(zeros) != set(FIXTURES):
        raise DensitySeedBFreezeError("density decision lacks D1/D2 zeros")
    analyses: dict[tuple[str, str], dict[str, Any]] = {}
    for fixture in FIXTURES:
        aggregate = {
            "candidates": by_fixture[fixture],
            "zero": {
                "candidate_id": zeros[fixture]["task_id"],
                "candidate_sha256": zeros[fixture]["candidate_binding"]["file_sha256"],
                "step": 0,
                "image_exposures": 0,
                "weighted_loss": zeros[fixture]["weighted_loss"],
                "mode": "zero_lora_control",
                "family_id": "ZERO",
            },
        }
        try:
            curves = krea_decision._curves(aggregate, expected_arm_ids=FAMILIES)
        except (KeyError, ValueError) as exc:
            raise DensitySeedBFreezeError(f"density curve selection failed: {exc}") from exc
        analyses[(fixture, "A")] = {
            "batch_id": f"density-{fixture}-A",
            "aggregate": aggregate,
            "curves": curves,
        }
    return analyses, public


def _relative(loss: Decimal, zero: Decimal) -> Decimal:
    if zero <= 0:
        raise DensitySeedBFreezeError("zero loss must be positive")
    return (zero - loss) / zero


def _source_anchor(
    anchor_analyses: Mapping[tuple[str, str], Mapping[str, Any]],
    public: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str | None], dict[str, Any]]:
    result: dict[tuple[str, str | None], dict[str, Any]] = {}
    for fixture in FIXTURES:
        aggregate = anchor_analyses[(fixture, "A")]["aggregate"]
        zero_id = aggregate["zero"]["candidate_id"]
        result[(fixture, None)] = dict(public[zero_id])
        for family in FAMILIES:
            selected = anchor_analyses[(fixture, "A")]["curves"][family]["selected"]
            result[(fixture, family)] = dict(public[selected["candidate_id"]])
    return result


def _validate_seedb_full(
    path: Path | None, *, anchors: Mapping[tuple[str, str | None], Mapping[str, Any]], cutoff: datetime
) -> tuple[dict[tuple[str, str | None], dict[str, Any]] | None, dict[str, Any]]:
    if path is None:
        return None, {
            "state": "absent_at_cutoff",
            "pooling_eligible": False,
            "partial_rows_are_descriptive_only": True,
        }
    value, file_binding = _load_json(path, "Seed-B bridge results", canonical=True)
    # A genuinely partial operational record may be preserved for provenance,
    # but no scalar from it is admitted.  A record claiming full completion is
    # validated exhaustively; corruption is never silently downgraded.
    if value.get("schema") == SEEDB_SCHEMA and (
        value.get("complete") is not True or value.get("row_count") != 14
    ):
        rows = value.get("rows")
        return None, {
            **file_binding,
            "state": "partial_at_cutoff",
            "observed_row_count": len(rows) if isinstance(rows, list) else 0,
            "pooling_eligible": False,
            "partial_rows_are_descriptive_only": True,
        }
    _exact(value, _B_RESULT_KEYS, "Seed-B bridge results")
    body = {key: item for key, item in value.items() if key != "results_sha256"}
    if (
        value["schema"] != SEEDB_SCHEMA
        or value["kind"] != SEEDB_KIND
        or value["row_count"] != 14
        or value["complete"] is not True
        or value["results_sha256"] != krea_provenance.canonical_sha256(body)
        or _time(value["created_at_utc"], "Seed-B results created_at_utc") > cutoff
    ):
        raise DensitySeedBFreezeError("Seed-B result identity/completeness drifted")
    plan_binding = _object(value["plan"], "Seed-B plan binding")
    _exact(plan_binding, _FILE_BINDING_KEYS | {"plan_sha256"}, "Seed-B plan binding")
    plan_file = _validate_binding(
        {key: plan_binding[key] for key in _FILE_BINDING_KEYS}, "Seed-B plan"
    )
    plan, _ = _load_json(plan_file["path"], "Seed-B plan", canonical=True)
    plan_body = {key: item for key, item in plan.items() if key != "plan_sha256"}
    if (
        plan.get("schema") != "sn56.seedb.score-bridge-plan.v1"
        or plan.get("kind") != "forge-krea-seedb-sparse-replication-bridge"
        or plan.get("seed_role") != "B"
        or plan.get("hard_cap") != 14
        or plan.get("row_count") != 14
        or plan.get("plan_sha256") != krea_provenance.canonical_sha256(plan_body)
        or plan_binding["plan_sha256"] != plan["plan_sha256"]
        or plan.get("decision_pin", {}).get("commit") != DECISION_COMMIT
        or plan.get("selection_scope", {}).get("excluded_tiers")
        != ["EXHAUSTIVE_BACKFILL", "TARGETED_BACKFILL"]
    ):
        raise DensitySeedBFreezeError("Seed-B plan identity drifted")
    plan_rows = {row.get("task_id"): row for row in plan.get("rows", []) if isinstance(row, dict)}
    if len(plan_rows) != 14:
        raise DensitySeedBFreezeError("Seed-B plan does not contain exactly 14 rows")
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != 14:
        raise DensitySeedBFreezeError("Seed-B results do not contain exactly 14 rows")
    indexed: dict[tuple[str, str | None], dict[str, Any]] = {}
    for ordinal, raw in enumerate(rows, 1):
        row = _object(raw, f"Seed-B rows[{ordinal}]")
        _exact(row, _B_ROW_KEYS, f"Seed-B rows[{ordinal}]")
        fixture, family = row["fixture"], row["family"]
        source = plan_rows.get(row["task_id"])
        if (
            row["ordinal"] != ordinal
            or fixture not in FIXTURES
            or (family is not None and family not in FAMILIES)
            or (fixture, family) in indexed
            or not isinstance(source, dict)
            or source.get("fixture") != fixture
            or source.get("family") != family
            or source.get("label") != row["label"]
            or source.get("step") != row["step"]
            or source.get("fraction") != row["fraction"]
            or source.get("candidate") != row["candidate"]
            or source.get("source_seed_a") != row["source_seed_a"]
            or source.get("final_checkpoint_eligible") is not False
            or source.get("checkpoint_role")
            not in {"replication_anchor", "replication_anchor_zero"}
        ):
            raise DensitySeedBFreezeError("Seed-B result escaped its sealed plan")
        _validate_binding(row["candidate"], f"Seed-B {row['task_id']} candidate")
        for key in ("queue_descriptor", "known_state", "status", "result"):
            _validate_binding(row[key], f"Seed-B {row['task_id']} {key}")
        known, _ = _load_env(
            row["known_state"]["path"], f"Seed-B {row['task_id']} known state"
        )
        descriptor, _ = _load_env(
            row["queue_descriptor"]["path"],
            f"Seed-B {row['task_id']} queue descriptor",
        )
        if (
            known.get("state") != "DONE"
            or known.get("descriptor") != row["queue_descriptor"]["path"]
            or descriptor.get("task_id") != row["task_id"]
            or descriptor.get("fixture") != fixture
            or descriptor.get("candidate") != row["candidate"]["path"]
            or descriptor.get("candidate_sha256")
            != row["candidate"]["file_sha256"]
            or descriptor.get("seed_role") != "B"
            or descriptor.get("bridge_plan_sha256") != plan["plan_sha256"]
            or descriptor.get("coverage_tier") != "PRIMARY"
            or descriptor.get("output_dir") != row["output_directory"]
        ):
            raise DensitySeedBFreezeError("Seed-B queue/known receipt chain drifted")
        status, _ = _load_env(row["status"]["path"], f"Seed-B {row['task_id']} status")
        if (
            status.get("returncode") != "0"
            or status.get("fixture") != fixture
            or status.get("candidate_sha256") != row["candidate"]["file_sha256"]
            or _time(status.get("ended_utc"), "Seed-B ended_utc") > cutoff
        ):
            raise DensitySeedBFreezeError("Seed-B status is late or nonzero")
        result, _ = _load_json(
            row["result"]["path"], f"Seed-B {row['task_id']} exact result", canonical=False
        )
        expected_rows = 24 if fixture == "D1" else 40
        commits = plan.get("evaluator", {}).get("source_commits")
        if (
            result.get("evaluator") != "god_krea2_img2img_exact"
            or result.get("candidate_sha256") != row["candidate"]["file_sha256"]
            or result.get("staged_candidate_sha256") != row["candidate"]["file_sha256"]
            or result.get("candidate_bytes") != row["candidate"]["bytes"]
            or result.get("dataset_sha256")
            != plan.get("fixtures", {}).get(fixture, {}).get("sha256")
            or _decimal(result.get("weighted_loss"), "Seed-B exact loss", positive=True)
            != _decimal(row["weighted_loss"], "Seed-B published loss", positive=True)
            or not isinstance(result.get("scored_rows"), list)
            or len(result["scored_rows"]) != expected_rows
            or result.get("runtime", {}).get("comfy_history", {}).get("prompt_count")
            != expected_rows * 10
            or result.get("source", {}).get("expected_commits")
            != {
                "comfyui": commits.get("comfyui"),
                "god": commits.get("god"),
                "tooling_nodes": commits.get("tooling_nodes"),
            }
        ):
            raise DensitySeedBFreezeError("Seed-B exact result identity drifted")
        evidence = _object(row["evidence"], "Seed-B evidence")
        _exact(evidence, {"manifest", "files"}, "Seed-B evidence")
        _validate_binding(evidence["manifest"], "Seed-B evidence manifest")
        if not isinstance(evidence["files"], list):
            raise DensitySeedBFreezeError("Seed-B evidence files must be a list")
        observed_evidence: dict[str, str] = {}
        for evidence_row in evidence["files"]:
            evidence_row = _object(evidence_row, "Seed-B evidence file")
            _exact(evidence_row, {"name"} | _FILE_BINDING_KEYS, "Seed-B evidence file")
            _validate_binding(
                {key: evidence_row[key] for key in _FILE_BINDING_KEYS},
                f"Seed-B evidence {evidence_row['name']}",
            )
            if evidence_row["name"] in observed_evidence:
                raise DensitySeedBFreezeError("Seed-B evidence repeats a file")
            observed_evidence[evidence_row["name"]] = evidence_row["file_sha256"]
        manifest_text = Path(evidence["manifest"]["path"]).read_text(encoding="utf-8")
        manifest_rows: dict[str, str] = {}
        for line in manifest_text.splitlines():
            match = re.fullmatch(r"([0-9a-f]{64})  (/.+)", line)
            if match is None:
                raise DensitySeedBFreezeError("Seed-B evidence manifest is malformed")
            name = Path(match.group(2)).name
            if name in manifest_rows:
                raise DensitySeedBFreezeError("Seed-B evidence manifest repeats a file")
            manifest_rows[name] = match.group(1)
        if manifest_rows != observed_evidence:
            raise DensitySeedBFreezeError("Seed-B evidence manifest bindings drifted")
        validated = row["validated"]
        if (
            not isinstance(validated, dict)
            or set(validated) != _B_VALIDATED_KEYS
            or any(item is not True for item in validated.values())
        ):
            raise DensitySeedBFreezeError("Seed-B validation claims are incomplete")
        anchor = anchors[(fixture, family)]
        if (
            row["source_seed_a"].get("task_id") != anchor["task_id"]
            or row["step"] != anchor["step"]
            or (family is not None and row["fraction"] != str(
                Decimal(anchor["step"])
                / Decimal(krea_density_gate._GEOMETRY[(fixture, family)][1])
            ))
        ):
            raise DensitySeedBFreezeError("Seed-B row does not match the Seed-A anchor")
        indexed[(fixture, family)] = {
            **row,
            "weighted_loss_decimal": _decimal(
                row["weighted_loss"], "Seed-B weighted loss", positive=True
            ),
        }
    expected_roles = {
        (fixture, family)
        for fixture in FIXTURES
        for family in (*FAMILIES, None)
    }
    if set(indexed) != expected_roles:
        raise DensitySeedBFreezeError("Seed-B result coverage is not exactly 12+2")
    comparison = value["seed_a_seed_b_comparison"]
    if not isinstance(comparison, list) or len(comparison) != 12:
        raise DensitySeedBFreezeError("Seed-B comparison table is not exactly 12 rows")
    expected_comparison = []
    for row in rows:
        if row["family"] is None:
            continue
        published = _object(
            comparison[len(expected_comparison)], "Seed-B comparison row"
        )
        _exact(published, _B_COMPARISON_KEYS, "Seed-B comparison row")
        zero = indexed[(row["fixture"], None)]["weighted_loss_decimal"]
        b_relative = (zero - indexed[(row["fixture"], row["family"])]["weighted_loss_decimal"]) / zero
        a_relative = _decimal(
            row["source_seed_a"]["relative_improvement_over_zero"],
            "Seed-A source relative improvement",
        )
        expected_comparison.append(
            {
                "fixture": row["fixture"],
                "family": row["family"],
                "step": row["step"],
                "seed_a_relative_improvement": str(a_relative),
                "seed_b_relative_improvement": str(b_relative),
                "absolute_seed_delta": str(abs(a_relative - b_relative)),
                "replication_anchor": {
                    "source_seed_a_task_id": row["source_seed_a"]["task_id"],
                    "fixture": row["fixture"],
                    "family": row["family"],
                    "step": row["step"],
                    "fraction": row["fraction"],
                    "coordinate_match": True,
                },
                "pooling_eligible_at_replication_anchor": True,
                "final_checkpoint_comparison": {
                    "status": "REQUIRES_TARGETED_FREEZE_COMPARISON",
                    "pool_only_on_exact_fixture_family_step_fraction_match": True,
                    "anchor_mismatch_must_be_named_by_freeze": True,
                },
            }
        )
    if comparison != expected_comparison:
        raise DensitySeedBFreezeError("Seed-B comparison table does not recompute")
    if value["claim_limits"] != [
        "sparse Seed-B replication only; not an exhaustive Seed-B curve",
        "no C1-C4 evidence consumed",
        "no confirmation, release, or deployment authority",
    ]:
        raise DensitySeedBFreezeError("Seed-B claim limits drifted")
    return indexed, {
        **file_binding,
        "state": "complete_14_at_cutoff",
        "results_sha256": value["results_sha256"],
        "plan_file_sha256": plan_binding["file_sha256"],
        "plan_sha256": plan_binding["plan_sha256"],
        "pooling_eligible": True,
        "partial_rows_are_descriptive_only": True,
    }


def _rule_depth(rule: Mapping[str, Any]) -> Decimal:
    mappings = rule.get("actual_mappings")
    if not isinstance(mappings, list) or not mappings:
        raise DensitySeedBFreezeError("checkpoint rule lacks A-only mappings")
    return sum(Decimal(row["image_exposures"]) for row in mappings) / Decimal(
        len(mappings)
    )


def _pick(
    families: Sequence[str],
    *,
    primary: Mapping[str, Decimal],
    concept: Mapping[str, Mapping[str, Decimal]],
    rules: Mapping[str, Mapping[str, Any]],
    label: str,
    invocations: list[dict[str, Any]],
) -> str:
    best = max(primary[family] for family in families)
    near = [family for family in families if best - primary[family] <= TIE_BAND]
    ordered = sorted(
        near,
        key=lambda family: (
            -_rule_depth(rules[family]),
            abs(concept[family]["D1"] - concept[family]["D2"]),
            PREFERENCE.index(family),
            family,
        ),
    )
    chosen = ordered[0]
    if len(near) > 1:
        invocations.append(
            {
                "selection": label,
                "best_primary": float(best),
                "inside_inclusive_absolute_0p01_band": sorted(near),
                "ordered_after_depth_spread_preference": ordered,
                "chosen": chosen,
            }
        )
    return chosen


def _rank_agreement(
    a: Mapping[str, Decimal], b: Mapping[str, Decimal]
) -> dict[str, Any]:
    ordered_a = sorted(FAMILIES, key=lambda family: (-a[family], family))
    ordered_b = sorted(FAMILIES, key=lambda family: (-b[family], family))
    agree = total = 0
    for index, left in enumerate(FAMILIES):
        for right in FAMILIES[index + 1 :]:
            total += 1
            if (a[left] - a[right]) * (b[left] - b[right]) >= 0:
                agree += 1
    return {
        "seed_a_order": ordered_a,
        "seed_b_order": ordered_b,
        "exact_order_agreement": ordered_a == ordered_b,
        "pairwise_direction_agreement": agree / total,
        "pair_count": total,
    }


def _derive(
    *,
    anchor_decision: Mapping[str, Any],
    chosen_decision: Mapping[str, Any],
    seedb_rows: Mapping[tuple[str, str | None], Mapping[str, Any]] | None,
) -> dict[str, Any]:
    anchor_analyses, anchor_public = _analysis(anchor_decision)
    chosen_analyses, chosen_public = _analysis(chosen_decision)
    anchors = _source_anchor(anchor_analyses, anchor_public)
    rules = {
        family: krea_decision._checkpoint_rule(
            family,
            analyses=chosen_analyses,
            fixtures=FIXTURES,
            seed_roles=("A",),
            targets=REPORT_TARGETS,
        )
        for family in FAMILIES
    }
    concept: dict[str, dict[str, Decimal]] = {family: {} for family in FAMILIES}
    per_seed: dict[str, dict[str, Any]] = {family: {} for family in FAMILIES}
    mismatches: list[dict[str, Any]] = []
    full_b = seedb_rows is not None
    for fixture in FIXTURES:
        a_zero = _decimal(
            anchors[(fixture, None)]["weighted_loss"], "Seed-A zero", positive=True
        )
        b_zero = (
            seedb_rows[(fixture, None)]["weighted_loss_decimal"] if full_b else None
        )
        for family in FAMILIES:
            anchor = anchors[(fixture, family)]
            a_anchor = _relative(
                _decimal(anchor["weighted_loss"], "Seed-A anchor", positive=True), a_zero
            )
            b_anchor = (
                _relative(seedb_rows[(fixture, family)]["weighted_loss_decimal"], b_zero)
                if full_b and b_zero is not None
                else None
            )
            primary = (a_anchor + b_anchor) / Decimal(2) if b_anchor is not None else a_anchor
            concept[family][fixture] = primary
            curve = chosen_analyses[(fixture, "A")]["curves"][family]
            curve_selected = curve["selected"]
            mapping = next(
                row
                for row in rules[family]["actual_mappings"]
                if row["fixture_id"] == fixture
            )
            curve_values = [
                Decimal(str(row["relative_improvement_over_zero"]))
                for row in anchor_analyses[(fixture, "A")]["curves"][family]["curve"]
            ]
            sign_agreement = (
                None
                if b_anchor is None
                else (a_anchor >= 0) == (b_anchor >= 0)
            )
            within_a_range = (
                None
                if b_anchor is None
                else min(curve_values) <= b_anchor <= max(curve_values)
            )
            mismatch = (
                anchor["task_id"] != mapping["candidate_id"]
                or anchor["step"] != mapping["step"]
            )
            if mismatch:
                mismatches.append(
                    {
                        "fixture": fixture,
                        "family": family,
                        "A_anchor_task_id": anchor["task_id"],
                        "A_anchor_step": anchor["step"],
                        "A_final_policy_candidate_id": mapping["candidate_id"],
                        "A_final_policy_step": mapping["step"],
                    }
                )
            per_seed[family][fixture] = {
                "A_anchor_relative_improvement": float(a_anchor),
                "B_anchor_relative_improvement": (
                    None if b_anchor is None else float(b_anchor)
                ),
                "absolute_A_B_anchor_delta": (
                    None if b_anchor is None else float(abs(a_anchor - b_anchor))
                ),
                "primary_relative_improvement": float(primary),
                "A_curve_selected": {
                    "candidate_id": curve_selected["candidate_id"],
                    "candidate_sha256": curve_selected["candidate_sha256"],
                    "step": curve_selected["step"],
                    "relative_improvement": float(
                        curve["selected_relative_improvement"]
                    ),
                },
                "A_final_policy_mapping": mapping,
                "anchor_final_mismatch": mismatch,
                "A_B_sign_agreement": sign_agreement,
                "B_anchor_inside_A_anchor_curve_range": within_a_range,
            }
    score_table = {}
    for family in FAMILIES:
        values = concept[family]
        ci = krea_decision._bootstrap_ci(values, label=f"density-seedb-{family}")
        score_table[family] = {
            "D1_primary_relative_improvement": float(values["D1"]),
            "D2_primary_relative_improvement": float(values["D2"]),
            "worst_fixture_primary": float(min(values.values())),
            "mean_fixture_primary": float(sum(values.values()) / Decimal(2)),
            "cluster_bootstrap_95pct": ci,
            "bootstrap_resamples": krea_decision._BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": krea_decision._BOOTSTRAP_SEED,
            "clusters": ["D1", "D2"],
            "seed_count_note": "n=2 training seeds is not a meaningful seed-level CI",
            "supplemental_row_paired_interval": {
                "available": False,
                "reason": "freeze consumes aggregate exact-score contracts; no compatible paired row-loss contract was predeclared",
            },
        }
    invocations: list[dict[str, Any]] = []
    winners = {
        fixture: _pick(
            NONCONTROLS,
            primary={family: concept[family][fixture] for family in NONCONTROLS},
            concept=concept,
            rules=rules,
            label=f"{fixture}_winner",
            invocations=invocations,
        )
        for fixture in FIXTURES
    }
    best_by_fixture = {
        fixture: max(concept[family][fixture] for family in NONCONTROLS)
        for fixture in FIXTURES
    }
    regret = {
        family: max(
            best_by_fixture[fixture] - concept[family][fixture]
            for fixture in FIXTURES
        )
        for family in NONCONTROLS
    }
    finalists: list[str] = []
    for fixture in FIXTURES:
        if winners[fixture] not in finalists:
            finalists.append(winners[fixture])
    remaining = [family for family in NONCONTROLS if family not in finalists]
    if remaining:
        third = _pick(
            remaining,
            primary={family: -regret[family] for family in remaining},
            concept=concept,
            rules=rules,
            label="third_noncontrol_minimax_regret",
            invocations=invocations,
        )
        finalists.append(third)
    if len(finalists) > 3:
        raise DensitySeedBFreezeError("more than three non-control finalists selected")
    finalists.append("K0")
    ranking = {}
    if full_b:
        for fixture in FIXTURES:
            a_scores = {
                family: _decimal(
                    per_seed[family][fixture]["A_anchor_relative_improvement"],
                    "Seed-A anchor score",
                )
                for family in FAMILIES
            }
            b_scores = {
                family: _decimal(
                    per_seed[family][fixture]["B_anchor_relative_improvement"],
                    "Seed-B anchor score",
                )
                for family in FAMILIES
            }
            ranking[fixture] = _rank_agreement(a_scores, b_scores)
    agreement_rows = [
        per_seed[family][fixture]
        for family in FAMILIES
        for fixture in FIXTURES
    ]
    agreement_statement = (
        {
            "status": "full_14_row_exact_anchor_replication_available",
            "sign_agreement_count": sum(
                row["A_B_sign_agreement"] is True for row in agreement_rows
            ),
            "sign_comparison_count": 12,
            "inside_A_curve_range_count": sum(
                row["B_anchor_inside_A_anchor_curve_range"] is True
                for row in agreement_rows
            ),
            "range_comparison_count": 12,
        }
        if full_b
        else {
            "status": "B14_incomplete_at_cutoff_A_anchor_used_uniformly",
            "sign_agreement_count": None,
            "sign_comparison_count": 0,
            "inside_A_curve_range_count": None,
            "range_comparison_count": 0,
        }
    )
    universe = sorted(set(finalists) | {"K0", "K2", "K3", "K4"})
    exact_per_family = {
        family: (
            2
            if any(item["family"] == family for item in mismatches)
            else 1
        )
        for family in universe
    }
    return {
        "selection_algorithm": {
            "primary": (
                "mean exact-coordinate Seed-A/Seed-B anchor RI"
                if full_b
                else "Seed-A anchor RI uniformly for every family"
            ),
            "checkpoint_curves_and_rules_seed_roles": ["A"],
            "family_tie_band": 0.01,
            "family_tie_band_semantics": "inclusive absolute relative-improvement difference",
            "ordered_tie_axes": [
                "greater_A_only_final_policy_mapped_image_exposures",
                "smaller_global_D1_D2_primary_spread",
                "K2>K3>K4>K5>K1",
            ],
            "raw_loss_selection_used": False,
        },
        "score_table": score_table,
        "family_relative_improvements": per_seed,
        "seed_a_seed_b_agreement": {
            "full_B14_pooled": full_b,
            "partial_B_mixed": False,
            "agreement_statement": agreement_statement,
            "ranking_agreement_by_fixture": ranking,
            "anchor_final_mismatch_count": len(mismatches),
            "anchor_final_mismatches": mismatches,
        },
        "D1_winner_family_id": winners["D1"],
        "D2_winner_family_id": winners["D2"],
        "minimax_regret": {family: float(regret[family]) for family in NONCONTROLS},
        "finalist_family_ids": finalists,
        "checkpoint_rules": {family: rules[family] for family in finalists},
        "all_family_checkpoint_rules": rules,
        "tie_break_invocations": invocations,
        "workload": {
            "family_universe": universe,
            "confirmation_training_count": 8 * len(universe),
            "confirmation_formula": "C1-C4 x Seed-A/Seed-B x family_universe",
            "minimum_exact_scores_per_family": exact_per_family,
            "optional_guard_score": {
                "enabled": False,
                "allowed_only_if_predeclared": True,
            },
            "boundary_variants_per_family": 6,
            "sealed_confirmation_content_consumed": False,
        },
    }


def _build(
    *,
    plan0_path: Path,
    sidecar0_path: Path,
    decision0_path: Path,
    plan9_path: Path,
    sidecar9_path: Path,
    decision9_path: Path | None,
    plan11_path: Path,
    sidecar11_path: Path,
    decision11_path: Path | None,
    a59_index_path: Path,
    failure_path: Path,
    seedb_results_path: Path | None,
    frozen_at_utc: str,
) -> dict[str, Any]:
    frozen_at = _time(frozen_at_utc, "frozen_at_utc")
    cutoff = _time(CUTOFF_UTC, "cutoff_utc")
    if frozen_at < cutoff:
        raise DensitySeedBFreezeError("freeze cannot predate the fixed cutoff")
    try:
        a59, a59_file_sha = krea_recovery_evidence.load_index(a59_index_path)
    except (OSError, ValueError) as exc:
        raise DensitySeedBFreezeError(f"A59 recovery snapshot does not replay: {exc}") from exc
    a59_binding = {
        "path": str(_safe_path(a59_index_path, "A59 snapshot", must_exist=True)),
        "bytes": Path(a59_index_path).stat().st_size,
        "file_sha256": a59_file_sha,
        "index_sha256": a59["index_sha256"],
        "coverage_ledger_file_sha256": a59["coverage_ledger"]["file_sha256"],
    }
    eligible = sum(row.get("selection_eligible") is True for row in a59["artifacts"])
    if eligible != 59 or a59.get("coverage", {}).get("selection_eligible") != 59:
        raise DensitySeedBFreezeError("A59 snapshot is not exactly 59 eligible rows")
    failure, failure_binding = _validate_failure(
        failure_path, a59_index=a59, a59_binding=a59_binding
    )
    bundles = [
        _load_density_triplet(
            "plan11",
            plan_path=plan11_path,
            sidecar_path=sidecar11_path,
            decision_path=decision11_path,
        ),
        _load_density_triplet(
            "plan9",
            plan_path=plan9_path,
            sidecar_path=sidecar9_path,
            decision_path=decision9_path,
        ),
        _load_density_triplet(
            "plan0",
            plan_path=plan0_path,
            sidecar_path=sidecar0_path,
            decision_path=decision0_path,
        ),
    ]
    observed_counts = [(item["additional_target_count"], item["selected_count"]) for item in bundles]
    if observed_counts != list(EXPECTED_PLAN_COUNTS):
        raise DensitySeedBFreezeError("density plans are mixed or mislabeled")
    recovery_bindings = [item["plan"]["recovery_index"] for item in bundles]
    if any(binding != recovery_bindings[0] for binding in recovery_bindings[1:]):
        raise DensitySeedBFreezeError("density plans do not share one immutable A59 snapshot")
    expected_recovery = {
        "path": a59_binding["path"],
        "file_sha256": a59_binding["file_sha256"],
        "index_sha256": a59_binding["index_sha256"],
        "coverage_ledger_file_sha256": a59_binding["coverage_ledger_file_sha256"],
    }
    if recovery_bindings[0] != expected_recovery:
        raise DensitySeedBFreezeError("density plans are not bound to the supplied A59 snapshot")
    completeness = {
        bundle["label"]: _cutoff_complete(bundle, cutoff) for bundle in bundles
    }
    complete = [bundle for bundle in bundles if completeness[bundle["label"]]]
    if not complete or completeness["plan0"] is not True:
        raise DensitySeedBFreezeError("mandatory A59 plan is not receipt-complete by cutoff")
    chosen = complete[0]
    anchor_bundle = bundles[-1]
    anchor_analyses, anchor_public = _analysis(anchor_bundle["decision"])
    anchors = _source_anchor(anchor_analyses, anchor_public)
    seedb_rows, seedb_binding = _validate_seedb_full(
        seedb_results_path, anchors=anchors, cutoff=cutoff
    )
    derived = _derive(
        anchor_decision=anchor_bundle["decision"],
        chosen_decision=chosen["decision"],
        seedb_rows=seedb_rows,
    )
    density_provenance = []
    for bundle in bundles:
        density_provenance.append(
            {
                "label": bundle["label"],
                "additional_target_count": bundle["additional_target_count"],
                "selected_count": bundle["selected_count"],
                "complete_by_cutoff": completeness[bundle["label"]],
                "plan": bundle["plan_binding"],
                "sidecar": bundle["sidecar_binding"],
                "decision_input": bundle["decision_binding"],
            }
        )
    body = {
        "schema": SCHEMA,
        "kind": FREEZE_KIND,
        "frozen_at_utc": _timestamp(frozen_at_utc, "frozen_at_utc"),
        "cutoff_utc": CUTOFF_UTC,
        "decision_commit": DECISION_COMMIT,
        "compatibility_fix_commit": COMPATIBILITY_FIX_COMMIT,
        "structural_failure": {
            **failure_binding,
            "failure_class": failure["failure_class"],
            "adapter_work_authorized": True,
        },
        "a59_recovery_snapshot": a59_binding,
        "density_plan_ladder": density_provenance,
        "chosen_density_plan": {
            "label": chosen["label"],
            "selected_count": chosen["selected_count"],
            "additional_target_count": chosen["additional_target_count"],
            "selection_rule": "highest receipt-complete plan: plan11 > plan9 > plan0",
            "rows_outside_chosen_plan_influence_selection": False,
        },
        "seed_b_bridge": {
            **seedb_binding,
            "all_or_none_pooling": True,
            "final_checkpoint_eligible": False,
            "checkpoint_curve_influence": False,
            "checkpoint_rule_influence": False,
            "tie_depth_influence": False,
        },
        "outcome": "finalists_frozen",
        "blockers": [],
        "claims": dict(FALSE_CLAIMS),
        "authority": dict(AUTHORITY),
        "strict_confirmation_admission_required": True,
        "c1c4_content_read": False,
        "agent_review_required": True,
        "agent_review_is_not_human_review": True,
        **derived,
    }
    return {**body, "freeze_sha256": krea_provenance.canonical_sha256(body)}


def freeze_finalists(**kwargs: Any) -> dict[str, Any]:
    output = Path(kwargs.pop("output"))
    value = _build(**kwargs)
    target = _safe_path(output, "density Seed-B freeze output", must_exist=False)
    if os.path.lexists(target) or os.path.lexists(Path(f"{target}.tmp")):
        raise FileExistsError(f"refusing existing freeze output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{target}.tmp")
    payload = krea_provenance.canonical_bytes(value) + b"\n"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, target)
    temporary.unlink()
    return value


def validate_freeze(path: Path) -> dict[str, Any]:
    value, _ = _load_json(path, "density Seed-B freeze", canonical=True)
    if value.get("schema") != SCHEMA or value.get("kind") != FREEZE_KIND:
        raise DensitySeedBFreezeError("density Seed-B freeze identity drifted")
    ladder = value.get("density_plan_ladder")
    if not isinstance(ladder, list) or len(ladder) != 3:
        raise DensitySeedBFreezeError("density Seed-B freeze plan ladder drifted")
    by_label = {row.get("label"): row for row in ladder if isinstance(row, dict)}
    if set(by_label) != {"plan0", "plan9", "plan11"}:
        raise DensitySeedBFreezeError("density Seed-B freeze plan labels drifted")
    seedb = _object(value.get("seed_b_bridge"), "seed_b_bridge")
    seedb_path = (
        seedb.get("path")
        if seedb.get("state") in {"complete_14_at_cutoff", "partial_at_cutoff"}
        else None
    )
    expected = _build(
        plan0_path=Path(by_label["plan0"]["plan"]["path"]),
        sidecar0_path=Path(by_label["plan0"]["sidecar"]["path"]),
        decision0_path=Path(by_label["plan0"]["decision_input"]["path"]),
        plan9_path=Path(by_label["plan9"]["plan"]["path"]),
        sidecar9_path=Path(by_label["plan9"]["sidecar"]["path"]),
        decision9_path=(
            Path(by_label["plan9"]["decision_input"]["path"])
            if by_label["plan9"]["decision_input"].get("state") != "absent_at_freeze"
            else None
        ),
        plan11_path=Path(by_label["plan11"]["plan"]["path"]),
        sidecar11_path=Path(by_label["plan11"]["sidecar"]["path"]),
        decision11_path=(
            Path(by_label["plan11"]["decision_input"]["path"])
            if by_label["plan11"]["decision_input"].get("state") != "absent_at_freeze"
            else None
        ),
        a59_index_path=Path(value["a59_recovery_snapshot"]["path"]),
        failure_path=Path(value["structural_failure"]["path"]),
        seedb_results_path=Path(seedb_path) if seedb_path else None,
        frozen_at_utc=value["frozen_at_utc"],
    )
    if value != expected:
        raise DensitySeedBFreezeError("density Seed-B freeze does not replay exactly")
    return value


def _optional_path(value: str | None) -> Path | None:
    return None if value is None else Path(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("freeze")
    for name in (
        "plan0",
        "sidecar0",
        "decision0",
        "plan9",
        "sidecar9",
        "plan11",
        "sidecar11",
        "a59-index",
        "failure-record",
        "output",
    ):
        create.add_argument(f"--{name}", type=Path, required=True)
    create.add_argument("--decision9", type=Path)
    create.add_argument("--decision11", type=Path)
    create.add_argument("--seed-b-results", type=Path)
    create.add_argument("--frozen-at-utc", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        value = validate_freeze(args.freeze)
    else:
        value = freeze_finalists(
            plan0_path=args.plan0,
            sidecar0_path=args.sidecar0,
            decision0_path=args.decision0,
            plan9_path=args.plan9,
            sidecar9_path=args.sidecar9,
            decision9_path=args.decision9,
            plan11_path=args.plan11,
            sidecar11_path=args.sidecar11,
            decision11_path=args.decision11,
            a59_index_path=args.a59_index,
            failure_path=args.failure_record,
            seedb_results_path=args.seed_b_results,
            frozen_at_utc=args.frozen_at_utc,
            output=args.output,
        )
    print(value["freeze_sha256"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AUTHORITY",
    "CUTOFF_UTC",
    "DensitySeedBFreezeError",
    "FALSE_CLAIMS",
    "FREEZE_KIND",
    "SCHEMA",
    "freeze_finalists",
    "validate_freeze",
]
