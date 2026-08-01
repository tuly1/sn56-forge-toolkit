#!/usr/bin/env python3
"""Deterministic, fail-closed score-density gate for the Krea recovery grid.

The targeted contract does not redefine the recovery universe.  It seals the
canonical 92 Seed-A rows, selects all sparse/zero rows, permanently promotes
the D1/K1 step-522 relief neighbour, and may select at most eleven of the
remaining exhaustive rows as targeted backfill.  The legacy ``exhaustive92``
contract remains available as a separate, explicit plan type.

Public plan and decision paths consume a published recovery index through
``krea_recovery_evidence.load_index``.  That validator rehashes the ledger,
receipts, candidates, results, and evidence before any scalar loss reaches
this boundary.
"""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence

try:
    from . import krea_decision
    from . import krea_provenance
    from . import krea_recovery_evidence
except ImportError:  # pragma: no cover - direct script execution.
    import krea_decision  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_recovery_evidence  # type: ignore[no-redef]


SCHEMA = 2
TARGETED_CONTRACT = "targeted-density-v1"
EXHAUSTIVE92_CONTRACT = "exhaustive92"
TARGET_PLAN_KIND = "forge-krea-density-target-plan"
EXHAUSTIVE_PLAN_KIND = "forge-krea-density-exhaustive92-plan"
SIDECAR_KIND = "forge-krea-density-plan-sidecar"
DECISION_INPUT_KIND = "forge-krea-density-decision-input"

SPARSE_PRIMARY = "SPARSE_PRIMARY"
INDEPENDENT_ZERO = "INDEPENDENT_ZERO"
EXHAUSTIVE_BACKFILL = "EXHAUSTIVE_BACKFILL"
RELIEF_NEIGHBOR_PROMOTED = "RELIEF_NEIGHBOR_PROMOTED"
TARGETED_BACKFILL = "TARGETED_BACKFILL"
RELIEF_TASK_ID = "d1-k1-step522"
SEED_ROLE = "A"

_TIE_UNCERTAINTY_BAND = Decimal("0.01")
_TIE_ORDERED_AXES = (
    "greater_selected_image_exposures",
    "smaller_D1_D2_relative_improvement_spread",
    "predeclared_family_preference",
)
_TIE_FAMILY_PREFERENCE = ("K2", "K3", "K4", "K5", "K1")


def _tie_policy_document() -> dict[str, Any]:
    """Return an isolated JSON projection of the frozen tie policy."""

    return {
        "uncertainty_band": float(_TIE_UNCERTAINTY_BAND),
        "ordered_axes": list(_TIE_ORDERED_AXES),
        "family_preference": list(_TIE_FAMILY_PREFERENCE),
    }


def _assert_external_tie_policy() -> None:
    if krea_decision.FAMILY_TIE_BREAK_POLICY != _tie_policy_document():
        raise RuntimeError(
            "Krea density tie policy differs from the frozen decision policy"
        )


_assert_external_tie_policy()

_SHA = re.compile(r"[0-9a-f]{64}")
_FIXTURES = ("D1", "D2")
_FAMILIES = ("K0", "K1", "K2", "K3", "K4", "K5")
_GEOMETRY = {
    ("D1", "K0"): (53, 260),
    ("D1", "K1"): (87, 691),
    ("D1", "K2"): (87, 691),
    ("D1", "K3"): (78, 618),
    ("D1", "K4"): (37, 290),
    ("D1", "K5"): (87, 691),
    ("D2", "K0"): (74, 367),
    ("D2", "K1"): (67, 531),
    ("D2", "K2"): (67, 531),
    ("D2", "K3"): (59, 472),
    ("D2", "K4"): (27, 210),
    ("D2", "K5"): (67, 531),
}
_UNIVERSE_KEYS = {
    "task_id",
    "fixture",
    "family",
    "label",
    "seed_role",
    "step",
    "image_exposures",
    "universe_tier",
}
_PLAN_ROW_KEYS = _UNIVERSE_KEYS | {
    "ledger_coverage_tier",
    "selected",
    "plan_tier",
}
_RECOVERY_BINDING_KEYS = {
    "path",
    "file_sha256",
    "index_sha256",
    "coverage_ledger_file_sha256",
}
_PLAN_KEYS = {
    "schema",
    "kind",
    "contract",
    "seed_role",
    "canonical_universe_sha256",
    "family_tie_break_policy",
    "recovery_index",
    "primary_losses",
    "selection_policy",
    "rows",
    "selected_count",
    "targeted_backfill_count_including_relief",
    "target_plan_sha256",
}
_SIDECAR_KEYS = {
    "schema",
    "kind",
    "contract",
    "target_plan_sha256",
    "canonical_universe_sha256",
    "rows",
    "sidecar_sha256",
}
_DECISION_INPUT_KEYS = {
    "schema",
    "kind",
    "contract",
    "target_plan_sha256",
    "sidecar_sha256",
    "final_recovery_index",
    "score_count",
    "candidate_rows_for_krea_decision",
    "complete",
    "decision_input_sha256",
}


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(
            f"{label} keys differ: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )


def _finite_loss(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _decimal_loss(value: Any, label: str) -> Decimal:
    """Recover the canonical decimal value represented by a validated scalar."""

    return Decimal(str(_finite_loss(value, label)))


def _canonical_universe() -> tuple[Mapping[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for fixture in _FIXTURES:
        for family in _FAMILIES:
            cadence, final_step = _GEOMETRY[(fixture, family)]
            multiplier = 1
            while multiplier * cadence < final_step:
                step = multiplier * cadence
                task_id = f"{fixture.lower()}-{family.lower()}-step{step}"
                tier = (
                    SPARSE_PRIMARY
                    if multiplier % 2 == 1
                    else EXHAUSTIVE_BACKFILL
                )
                if task_id == RELIEF_TASK_ID:
                    tier = RELIEF_NEIGHBOR_PROMOTED
                rows.append(
                    {
                        "task_id": task_id,
                        "fixture": fixture,
                        "family": family,
                        "label": f"step-{step}",
                        "seed_role": SEED_ROLE,
                        "step": step,
                        # Recovery runs use batch=1, repeats=1 and grad-accum=1.
                        "image_exposures": step,
                        "universe_tier": tier,
                    }
                )
                multiplier += 1
            rows.append(
                {
                    "task_id": f"{fixture.lower()}-{family.lower()}-final{final_step}",
                    "fixture": fixture,
                    "family": family,
                    "label": f"final-{final_step}",
                    "seed_role": SEED_ROLE,
                    "step": final_step,
                    "image_exposures": final_step,
                    "universe_tier": SPARSE_PRIMARY,
                }
            )
        rows.append(
            {
                "task_id": f"{fixture.lower()}-zero-baseline",
                "fixture": fixture,
                "family": None,
                "label": "zero-baseline",
                "seed_role": SEED_ROLE,
                "step": 0,
                "image_exposures": 0,
                "universe_tier": INDEPENDENT_ZERO,
            }
        )
    return tuple(MappingProxyType(row) for row in rows)


CANONICAL_UNIVERSE = _canonical_universe()
CANONICAL_UNIVERSE_SHA256 = krea_provenance.canonical_sha256(
    [dict(row) for row in CANONICAL_UNIVERSE]
)
_UNIVERSE_BY_ID = {row["task_id"]: row for row in CANONICAL_UNIVERSE}
_SPARSE_IDS = tuple(
    row["task_id"]
    for row in CANONICAL_UNIVERSE
    if row["universe_tier"] == SPARSE_PRIMARY
)


def _assert_internal_universe() -> None:
    tiers: dict[str, int] = {}
    for row in CANONICAL_UNIVERSE:
        tiers[row["universe_tier"]] = tiers.get(row["universe_tier"], 0) + 1
    if (
        len(CANONICAL_UNIVERSE) != 92
        or len(_UNIVERSE_BY_ID) != 92
        or tiers
        != {
            SPARSE_PRIMARY: 56,
            EXHAUSTIVE_BACKFILL: 33,
            RELIEF_NEIGHBOR_PROMOTED: 1,
            INDEPENDENT_ZERO: 2,
        }
    ):
        raise RuntimeError("internal Krea density universe drifted")


_assert_internal_universe()


def _file_binding_projection(value: Any, label: str) -> dict[str, Any]:
    binding = _object(value, label)
    result = {key: binding.get(key) for key in ("path", "bytes", "file_sha256")}
    if (
        not isinstance(result["path"], str)
        or not result["path"].startswith("/")
        or os.path.normpath(result["path"]) != result["path"]
        or isinstance(result["bytes"], bool)
        or not isinstance(result["bytes"], int)
        or result["bytes"] < 0
        or not isinstance(result["file_sha256"], str)
        or _SHA.fullmatch(result["file_sha256"]) is None
    ):
        raise ValueError(f"{label} is malformed")
    return result


def _result_binding_projection(value: Any, label: str) -> dict[str, Any]:
    binding = _object(value, label)
    result = _file_binding_projection(binding, label)
    semantic = binding.get("semantic_sha256")
    if not isinstance(semantic, str) or _SHA.fullmatch(semantic) is None:
        raise ValueError(f"{label} semantic binding is malformed")
    result["semantic_sha256"] = semantic
    return result


def _validated_score_projection(
    row: Mapping[str, Any], task_id: str
) -> dict[str, Any]:
    validated = row.get("validated_artifact")
    if row.get("selection_eligible") is not True or not isinstance(validated, dict):
        raise ValueError(f"recovery row is not selection eligible: {task_id}")
    result = _object(validated.get("result"), f"{task_id} validated result")
    candidate_binding = _file_binding_projection(
        validated.get("candidate"), f"{task_id} candidate binding"
    )
    if candidate_binding["file_sha256"] != row.get("expected_candidate_sha256"):
        raise ValueError(f"selected candidate binding differs: {task_id}")
    return {
        "weighted_loss": _finite_loss(result.get("weighted_loss"), task_id),
        "candidate_binding": candidate_binding,
        "result_binding": _result_binding_projection(
            result, f"{task_id} result binding"
        ),
        "evidence_binding": _file_binding_projection(
            validated.get("evidence"), f"{task_id} evidence binding"
        ),
        "receipt_binding": _file_binding_projection(
            validated.get("receipt"), f"{task_id} receipt binding"
        ),
    }


def _primary_evidence_from_recovery_rows(
    rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"task_id": task_id, **_validated_score_projection(rows[task_id], task_id)}
        for task_id in _SPARSE_IDS
    ]


_PRIMARY_EVIDENCE_KEYS = {
    "task_id",
    "weighted_loss",
    "candidate_binding",
    "result_binding",
    "evidence_binding",
    "receipt_binding",
}


def _bound_primary_evidence(
    value: Any,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if not isinstance(value, list):
        raise ValueError("plan primary_losses must be a list")
    if len(value) != len(_SPARSE_IDS):
        raise ValueError("plan must bind all 56 sparse primary evidence rows")
    normalized: list[dict[str, Any]] = []
    losses: dict[str, float] = {}
    for index, raw in enumerate(value):
        row = _object(raw, f"primary_losses[{index}]")
        _exact(row, _PRIMARY_EVIDENCE_KEYS, f"primary_losses[{index}]")
        expected_id = _SPARSE_IDS[index]
        if row["task_id"] != expected_id or expected_id in losses:
            raise ValueError("plan primary evidence differs from canonical sparse order")
        for key in ("candidate_binding", "evidence_binding", "receipt_binding"):
            _exact(
                _object(row[key], f"primary_losses[{index}].{key}"),
                {"path", "bytes", "file_sha256"},
                f"primary_losses[{index}].{key}",
            )
        _exact(
            _object(
                row["result_binding"],
                f"primary_losses[{index}].result_binding",
            ),
            {"path", "bytes", "file_sha256", "semantic_sha256"},
            f"primary_losses[{index}].result_binding",
        )
        normalized_row = {
            "task_id": expected_id,
            "weighted_loss": _finite_loss(row["weighted_loss"], expected_id),
            "candidate_binding": _file_binding_projection(
                row["candidate_binding"],
                f"primary_losses[{index}].candidate_binding",
            ),
            "result_binding": _result_binding_projection(
                row["result_binding"],
                f"primary_losses[{index}].result_binding",
            ),
            "evidence_binding": _file_binding_projection(
                row["evidence_binding"],
                f"primary_losses[{index}].evidence_binding",
            ),
            "receipt_binding": _file_binding_projection(
                row["receipt_binding"],
                f"primary_losses[{index}].receipt_binding",
            ),
        }
        normalized.append(normalized_row)
        losses[expected_id] = normalized_row["weighted_loss"]
    return normalized, losses


def _family_preference(family: str) -> int:
    try:
        return _TIE_FAMILY_PREFERENCE.index(family)
    except ValueError as exc:
        raise ValueError(f"family {family} is absent from FAMILY_TIE_BREAK_POLICY") from exc


def _peak_for_cell(
    fixture: str, family: str, losses: Mapping[str, float]
) -> dict[str, Any]:
    sparse = [
        row
        for row in CANONICAL_UNIVERSE
        if row["fixture"] == fixture
        and row["family"] == family
        and row["universe_tier"] == SPARSE_PRIMARY
    ]
    minimum = min(
        _decimal_loss(losses[row["task_id"]], row["task_id"]) for row in sparse
    )
    # Density targets the empirical peak, not every checkpoint inside the
    # family-level uncertainty band.  Exact-equality ties retain the frozen
    # first tie axis: greater selected image exposures.
    tied = [
        row
        for row in sparse
        if _decimal_loss(losses[row["task_id"]], row["task_id"]) == minimum
    ]
    return max(tied, key=lambda row: (row["image_exposures"], row["task_id"]))


def _adjacent_backfills(
    fixture: str,
    family: str,
    peak: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in CANONICAL_UNIVERSE
        if row["fixture"] == fixture
        and row["family"] == family
        and row["universe_tier"] == EXHAUSTIVE_BACKFILL
    ]
    return sorted(
        rows,
        key=lambda row: (
            abs(row["image_exposures"] - peak["image_exposures"]),
            # Greater exposures is the frozen first tie axis.
            -row["image_exposures"],
            row["task_id"],
        ),
    )


def _rank_cells(
    cells: Sequence[tuple[str, str]],
    losses: Mapping[str, float],
) -> list[tuple[str, str]]:
    """Rank cells with the frozen uncertainty band and family preference."""

    remaining = list(cells)
    result: list[tuple[str, str]] = []
    while remaining:
        peak_losses: dict[tuple[str, str], Decimal] = {}
        for cell in remaining:
            peak = _peak_for_cell(*cell, losses)
            peak_losses[cell] = _decimal_loss(
                losses[peak["task_id"]], peak["task_id"]
            )
        raw_best = min(peak_losses.values())
        tied = [
            cell
            for cell in remaining
            if peak_losses[cell] <= raw_best + _TIE_UNCERTAINTY_BAND
        ]
        chosen = min(
            tied,
            key=lambda cell: (
                _family_preference(cell[1]),
                _FIXTURES.index(cell[0]),
            ),
        )
        result.append(chosen)
        remaining.remove(chosen)
    return result


def _selection(
    losses: Mapping[str, float], additional_target_count: int
) -> set[str]:
    if (
        isinstance(additional_target_count, bool)
        or not isinstance(additional_target_count, int)
        or not 0 <= additional_target_count <= 11
    ):
        raise ValueError("additional_target_count must be an integer in [0,11]")

    selected = {
        row["task_id"]
        for row in CANONICAL_UNIVERSE
        if row["universe_tier"] in {SPARSE_PRIMARY, INDEPENDENT_ZERO}
    }
    selected.add(RELIEF_TASK_ID)
    if additional_target_count == 0:
        return selected

    cells = [
        (fixture, family)
        for fixture in _FIXTURES
        for family in _FAMILIES[1:]
        if (fixture, family) != ("D1", "K1")
    ]
    ranked_cells = _rank_cells(cells, losses)
    base_candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell in cells:
        peak = _peak_for_cell(*cell, losses)
        candidates = _adjacent_backfills(*cell, peak)
        if not candidates:
            raise ValueError(f"{cell[0]}-{cell[1]} lacks adjacent backfill")
        base_candidates[cell] = candidates

    base_count = min(additional_target_count, len(cells))
    for cell in ranked_cells[:base_count]:
        selected.add(base_candidates[cell][0]["task_id"])

    bonus_count = additional_target_count - base_count
    if bonus_count:
        # At most one extra neighbour is awarded per fixture.  Rank those two
        # fixture bonuses by the same frozen family policy.
        bonus_cells: list[tuple[str, str]] = []
        for fixture in _FIXTURES:
            fixture_cells = [cell for cell in cells if cell[0] == fixture]
            winner = _rank_cells(fixture_cells, losses)[0]
            if len(base_candidates[winner]) < 2:
                raise ValueError(f"{winner[0]}-{winner[1]} lacks a fixture bonus")
            bonus_cells.append(winner)
        for cell in _rank_cells(bonus_cells, losses)[:bonus_count]:
            selected.add(base_candidates[cell][1]["task_id"])
    return selected


def _seal(body: Mapping[str, Any], digest_key: str) -> dict[str, Any]:
    sealed = dict(body)
    sealed[digest_key] = krea_provenance.canonical_sha256(sealed)
    return sealed


def _recovery_binding(
    path: Path, index: Mapping[str, Any], file_sha256: str
) -> dict[str, str]:
    return {
        "path": str(path),
        "file_sha256": file_sha256,
        "index_sha256": index["index_sha256"],
        "coverage_ledger_file_sha256": index["coverage_ledger"]["file_sha256"],
    }


def _validate_recovery_binding(value: Any) -> dict[str, str]:
    binding = _object(value, "recovery index binding")
    _exact(binding, _RECOVERY_BINDING_KEYS, "recovery index binding")
    path = binding["path"]
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or os.path.normpath(path) != path
        or any(
            not isinstance(binding[key], str)
            or _SHA.fullmatch(binding[key]) is None
            for key in _RECOVERY_BINDING_KEYS - {"path"}
        )
    ):
        raise ValueError("recovery index binding is malformed")
    return binding


def _adapt_recovery_index(index: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("recovery index artifacts must be a list")
    by_id = {
        row.get("task_id"): row for row in artifacts if isinstance(row, dict)
    }
    if set(by_id) != set(_UNIVERSE_BY_ID) or len(artifacts) != 92:
        raise ValueError("recovery index differs from the canonical 92-row universe")
    adapted: dict[str, dict[str, Any]] = {}
    for expected in CANONICAL_UNIVERSE:
        task_id = expected["task_id"]
        row = by_id[task_id]
        expected_family = expected["family"] or "ZERO"
        expected_final = expected["label"].startswith("final-") or expected["family"] is None
        expected_zero = expected["family"] is None
        ledger_tier = row.get("coverage_tier")
        if task_id == RELIEF_TASK_ID:
            # The row was originally scored under the exhaustive ledger and
            # was later promoted by the authorised density relief.  Both
            # immutable histories are legitimate inputs; retain whichever
            # state the receipt-validated index actually records.
            valid_ledger_tiers = {
                EXHAUSTIVE_BACKFILL,
                RELIEF_NEIGHBOR_PROMOTED,
            }
        else:
            valid_ledger_tiers = {expected["universe_tier"]}
        if (
            row.get("fixture_id") != expected["fixture"]
            or row.get("family_id") != expected_family
            or row.get("step") != expected["step"]
            or row.get("is_final") is not expected_final
            or row.get("zero_control") is not expected_zero
            or ledger_tier not in valid_ledger_tiers
            or not isinstance(row.get("selection_eligible"), bool)
        ):
            raise ValueError(f"recovery index target identity differs: {task_id}")
        adapted[task_id] = {**row, "ledger_coverage_tier": ledger_tier}
    return adapted


def _load_recovery_index(
    path_value: Path | str,
) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, Any]]]:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path_value))))
    index, file_sha256 = krea_recovery_evidence.load_index(path)
    binding = _recovery_binding(path, index, file_sha256)
    _validate_recovery_binding(binding)
    return index, binding, _adapt_recovery_index(index)


def _build_plan(
    primary_evidence: Sequence[Mapping[str, Any]],
    recovery_index_binding: Mapping[str, Any],
    ledger_tiers: Mapping[str, str],
    *,
    contract: str = TARGETED_CONTRACT,
    additional_target_count: int = 11,
) -> dict[str, Any]:
    """Private normalized builder; public callers must use a recovery index."""

    _assert_external_tie_policy()
    bound_primary_evidence, losses = _bound_primary_evidence(
        list(primary_evidence)
    )
    binding = _validate_recovery_binding(recovery_index_binding)
    if set(ledger_tiers) != set(_UNIVERSE_BY_ID):
        raise ValueError("ledger tier coverage differs from canonical universe")
    if contract == TARGETED_CONTRACT:
        selected = _selection(losses, additional_target_count)
        kind = TARGET_PLAN_KIND
        policy = {
            "mode": "peak-adjacent-unsampled",
            "direction": "min",
            "required_noncontrol_cells_other_than_fixed_relief": 9,
            "fixture_bonus_limit": 2,
            "requested_additional_target_count": additional_target_count,
            "fixed_relief_task_id": RELIEF_TASK_ID,
            "live_geometry": {
                "batch_size": 1,
                "dataset_repeats": 1,
                "gradient_accumulation": 1,
                "image_exposures_equal_step": True,
            },
        }
    elif contract == EXHAUSTIVE92_CONTRACT:
        if additional_target_count != 11:
            raise ValueError("exhaustive92 does not accept a targeted row count")
        selected = set(_UNIVERSE_BY_ID)
        kind = EXHAUSTIVE_PLAN_KIND
        policy = {
            "mode": "legacy-exhaustive92",
            "fixed_relief_task_id": RELIEF_TASK_ID,
            "live_geometry": {
                "batch_size": 1,
                "dataset_repeats": 1,
                "gradient_accumulation": 1,
                "image_exposures_equal_step": True,
            },
        }
    else:
        raise ValueError(f"unsupported density contract: {contract}")

    rows: list[dict[str, Any]] = []
    for source in CANONICAL_UNIVERSE:
        is_selected = source["task_id"] in selected
        plan_tier = source["universe_tier"]
        if (
            contract == TARGETED_CONTRACT
            and is_selected
            and source["universe_tier"] == EXHAUSTIVE_BACKFILL
        ):
            plan_tier = TARGETED_BACKFILL
        rows.append(
            {
                **source,
                "ledger_coverage_tier": ledger_tiers[source["task_id"]],
                "selected": is_selected,
                "plan_tier": plan_tier,
            }
        )
    targeted_count = sum(
        row["selected"]
        and row["plan_tier"] in {TARGETED_BACKFILL, RELIEF_NEIGHBOR_PROMOTED}
        for row in rows
    )
    plan = _seal(
        {
            "schema": SCHEMA,
            "kind": kind,
            "contract": contract,
            "seed_role": SEED_ROLE,
            "canonical_universe_sha256": CANONICAL_UNIVERSE_SHA256,
            "family_tie_break_policy": _tie_policy_document(),
            "recovery_index": dict(binding),
            "primary_losses": bound_primary_evidence,
            "selection_policy": policy,
            "rows": rows,
            "selected_count": sum(row["selected"] for row in rows),
            "targeted_backfill_count_including_relief": targeted_count,
        },
        "target_plan_sha256",
    )
    return _validate_plan_structural(plan)


def _validate_universe_projection(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != len(CANONICAL_UNIVERSE):
        raise ValueError("plan must retain the complete canonical 92-row universe")
    for index, (raw, expected) in enumerate(zip(rows, CANONICAL_UNIVERSE, strict=True)):
        row = _object(raw, f"rows[{index}]")
        _exact(row, _PLAN_ROW_KEYS, f"rows[{index}]")
        if {key: row[key] for key in _UNIVERSE_KEYS} != expected:
            raise ValueError(f"rows[{index}] differs from the canonical Seed-A universe")
        valid_ledger_tiers = (
            {EXHAUSTIVE_BACKFILL, RELIEF_NEIGHBOR_PROMOTED}
            if expected["task_id"] == RELIEF_TASK_ID
            else {expected["universe_tier"]}
        )
        if row["ledger_coverage_tier"] not in valid_ledger_tiers:
            raise ValueError(f"rows[{index}] ledger coverage tier differs")
        if not isinstance(row["selected"], bool):
            raise ValueError(f"rows[{index}].selected must be boolean")


def _validate_plan_structural(value: Any) -> dict[str, Any]:
    _assert_external_tie_policy()
    plan = _object(value, "density plan")
    _exact(plan, _PLAN_KEYS, "density plan")
    contract = plan["contract"]
    expected_kind = {
        TARGETED_CONTRACT: TARGET_PLAN_KIND,
        EXHAUSTIVE92_CONTRACT: EXHAUSTIVE_PLAN_KIND,
    }.get(contract)
    if (
        plan["schema"] != SCHEMA
        or plan["kind"] != expected_kind
        or plan["seed_role"] != SEED_ROLE
        or plan["canonical_universe_sha256"] != CANONICAL_UNIVERSE_SHA256
        or plan["family_tie_break_policy"] != _tie_policy_document()
    ):
        raise ValueError("density plan identity or frozen policy drifted")
    _validate_recovery_binding(plan["recovery_index"])
    if not isinstance(plan["rows"], list):
        raise ValueError("density plan rows must be a list")
    _validate_universe_projection(plan["rows"])
    _, bound_losses = _bound_primary_evidence(plan["primary_losses"])
    body = {key: item for key, item in plan.items() if key != "target_plan_sha256"}
    if plan["target_plan_sha256"] != krea_provenance.canonical_sha256(body):
        raise ValueError("density plan hash seal differs")
    rows = plan["rows"]
    selected = [row for row in rows if row["selected"]]
    targeted = [
        row for row in selected
        if row["plan_tier"] in {TARGETED_BACKFILL, RELIEF_NEIGHBOR_PROMOTED}
    ]
    if (
        plan["selected_count"] != len(selected)
        or plan["targeted_backfill_count_including_relief"] != len(targeted)
    ):
        raise ValueError("density plan counts differ")
    policy = _object(plan["selection_policy"], "selection policy")
    geometry = {
        "batch_size": 1,
        "dataset_repeats": 1,
        "gradient_accumulation": 1,
        "image_exposures_equal_step": True,
    }
    if contract == TARGETED_CONTRACT:
        additional = policy.get("requested_additional_target_count")
        if (
            policy.get("mode") != "peak-adjacent-unsampled"
            or policy.get("direction") != "min"
            or policy.get("required_noncontrol_cells_other_than_fixed_relief") != 9
            or policy.get("fixture_bonus_limit") != 2
            or policy.get("fixed_relief_task_id") != RELIEF_TASK_ID
            or policy.get("live_geometry") != geometry
            or isinstance(additional, bool)
            or not isinstance(additional, int)
            or not 0 <= additional <= 11
            or len(selected) != 59 + additional
            or len(targeted) != 1 + additional
        ):
            raise ValueError("targeted density selection cardinality drifted")
        expected = _selection(bound_losses, additional)
        if {row["task_id"] for row in selected} != expected:
            raise ValueError("targeted rows differ from sealed primary-loss selection")
        for row in rows:
            mandatory = row["universe_tier"] in {SPARSE_PRIMARY, INDEPENDENT_ZERO} or row["task_id"] == RELIEF_TASK_ID
            expected_tier = row["universe_tier"]
            if row["selected"] and expected_tier == EXHAUSTIVE_BACKFILL:
                expected_tier = TARGETED_BACKFILL
            if (mandatory and not row["selected"]) or row["plan_tier"] != expected_tier:
                raise ValueError("targeted plan mandatory row/tier differs")
            if row["family"] == "K0" and row["plan_tier"] == TARGETED_BACKFILL:
                raise ValueError("K0 cannot be a targeted backfill")
    elif (
        policy
        != {
            "mode": "legacy-exhaustive92",
            "fixed_relief_task_id": RELIEF_TASK_ID,
            "live_geometry": geometry,
        }
        or len(selected) != 92
        or any(row["plan_tier"] != row["universe_tier"] for row in rows)
    ):
        raise ValueError("legacy exhaustive92 contract drifted")
    return plan


def build_plan_from_recovery_index(
    recovery_index_path: Path | str,
    *,
    contract: str = TARGETED_CONTRACT,
    additional_target_count: int = 11,
) -> dict[str, Any]:
    """Build a plan only from receipt-validated recovery evidence."""

    _, binding, rows = _load_recovery_index(recovery_index_path)
    primary_evidence = _primary_evidence_from_recovery_rows(rows)
    return _build_plan(
        primary_evidence,
        binding,
        {task_id: row["ledger_coverage_tier"] for task_id, row in rows.items()},
        contract=contract,
        additional_target_count=additional_target_count,
    )


def validate_plan(value: Any) -> dict[str, Any]:
    plan = _validate_plan_structural(value)
    _, binding, rows = _load_recovery_index(plan["recovery_index"]["path"])
    primary_evidence = _primary_evidence_from_recovery_rows(rows)
    additional = plan["selection_policy"].get("requested_additional_target_count", 11)
    expected = _build_plan(
        primary_evidence,
        binding,
        {task_id: row["ledger_coverage_tier"] for task_id, row in rows.items()},
        contract=plan["contract"],
        additional_target_count=additional,
    )
    if plan != expected:
        raise ValueError("density plan differs from its revalidated recovery index")
    return plan


def _build_sidecar(plan: Mapping[str, Any]) -> dict[str, Any]:
    return _seal(
        {
            "schema": SCHEMA,
            "kind": SIDECAR_KIND,
            "contract": plan["contract"],
            "target_plan_sha256": plan["target_plan_sha256"],
            "canonical_universe_sha256": CANONICAL_UNIVERSE_SHA256,
            "rows": [
                {
                    "task_id": row["task_id"],
                    "selected": row["selected"],
                    "ledger_coverage_tier": row["ledger_coverage_tier"],
                    "plan_tier": row["plan_tier"],
                    "step": row["step"],
                    "image_exposures": row["image_exposures"],
                }
                for row in plan["rows"]
            ],
        },
        "sidecar_sha256",
    )


def build_sidecar(plan_value: Any) -> dict[str, Any]:
    return _build_sidecar(validate_plan(plan_value))


def _validate_sidecar_structural(
    plan: Mapping[str, Any], sidecar_value: Any
) -> dict[str, Any]:
    sidecar = _object(sidecar_value, "density sidecar")
    _exact(sidecar, _SIDECAR_KEYS, "density sidecar")
    if sidecar != _build_sidecar(plan):
        raise ValueError("density sidecar is not exactly equivalent to its target plan")
    return sidecar


def validate_sidecar(plan_value: Any, sidecar_value: Any) -> dict[str, Any]:
    return _validate_sidecar_structural(validate_plan(plan_value), sidecar_value)


def _safe_new_file(path: Path | str, label: str) -> Path:
    target = Path(os.path.abspath(os.fspath(path)))
    for current in target.parents:
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} parent contains a symlink")
    return target


def write_artifacts(
    plan_path: Path | str, sidecar_path: Path | str, plan_value: Any
) -> tuple[str, str]:
    plan = validate_plan(plan_value)
    targets = (
        (_safe_new_file(plan_path, "plan path"), plan),
        (_safe_new_file(sidecar_path, "sidecar path"), _build_sidecar(plan)),
    )
    if targets[0][0] == targets[1][0]:
        raise ValueError("plan and sidecar paths must differ")
    written: list[Path] = []
    try:
        for path, document in targets:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(krea_provenance.canonical_bytes(document) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            written.append(path)
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path, _ in targets)  # type: ignore[return-value]


def _build_decision_input(
    plan: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    final_binding: Mapping[str, Any],
    indexed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build only from an index already revalidated by ``load_index``."""

    decision_rows: list[dict[str, Any]] = []
    bound_primary = {
        row["task_id"]: {key: item for key, item in row.items() if key != "task_id"}
        for row in plan["primary_losses"]
    }
    for target in plan["rows"]:
        if not target["selected"]:
            continue
        row = indexed[target["task_id"]]
        validated = row.get("validated_artifact")
        if row.get("selection_eligible") is not True or not isinstance(validated, dict):
            raise ValueError(f"selected recovery row is not selection eligible: {target['task_id']}")
        if row["ledger_coverage_tier"] != target["ledger_coverage_tier"]:
            raise ValueError(f"selected ledger tier differs: {target['task_id']}")
        score = _validated_score_projection(row, target["task_id"])
        if (
            target["task_id"] in bound_primary
            and score != bound_primary[target["task_id"]]
        ):
            raise ValueError(
                "final sparse receipt/candidate/result/evidence binding differs "
                "from the plan-bound recovery index"
            )
        decision_rows.append(
            {
                "task_id": target["task_id"],
                "fixture": target["fixture"],
                "family": target["family"],
                "seed_role": SEED_ROLE,
                "ledger_coverage_tier": row["ledger_coverage_tier"],
                "plan_tier": target["plan_tier"],
                "step": target["step"],
                "image_exposures": target["step"],
                "weighted_loss": score["weighted_loss"],
                "candidate_binding": score["candidate_binding"],
                "result_binding": score["result_binding"],
                "evidence_binding": score["evidence_binding"],
                "receipt_binding": score["receipt_binding"],
            }
        )
    if len(decision_rows) != plan["selected_count"]:
        raise ValueError("selected recovery coverage is incomplete")
    return _seal(
        {
            "schema": SCHEMA,
            "kind": DECISION_INPUT_KIND,
            "contract": plan["contract"],
            "target_plan_sha256": plan["target_plan_sha256"],
            "sidecar_sha256": sidecar["sidecar_sha256"],
            "final_recovery_index": final_binding,
            "score_count": len(decision_rows),
            "candidate_rows_for_krea_decision": decision_rows,
            "complete": True,
        },
        "decision_input_sha256",
    )


def build_decision_input_from_recovery_index(
    plan_value: Any,
    sidecar_value: Any,
    recovery_index_path: Path | str,
) -> dict[str, Any]:
    """Expose selected candidates only after every receipt revalidates."""

    plan = validate_plan(plan_value)
    sidecar = _validate_sidecar_structural(plan, sidecar_value)
    _, final_binding, indexed = _load_recovery_index(recovery_index_path)
    return _build_decision_input(plan, sidecar, final_binding, indexed)


def validate_decision_input(
    value: Any,
    *,
    plan_value: Any,
    sidecar_value: Any,
) -> dict[str, Any]:
    """Replay a persisted decision input against both bound recovery indexes."""

    plan = validate_plan(plan_value)
    sidecar = _validate_sidecar_structural(plan, sidecar_value)
    record = _object(value, "density decision input")
    _exact(record, _DECISION_INPUT_KEYS, "density decision input")
    body = {
        key: item for key, item in record.items() if key != "decision_input_sha256"
    }
    if (
        record["schema"] != SCHEMA
        or record["kind"] != DECISION_INPUT_KIND
        or record["contract"] != plan["contract"]
        or record["target_plan_sha256"] != plan["target_plan_sha256"]
        or record["sidecar_sha256"] != sidecar["sidecar_sha256"]
        or record["complete"] is not True
        or isinstance(record["score_count"], bool)
        or not isinstance(record["score_count"], int)
        or record["score_count"] != plan["selected_count"]
        or record["decision_input_sha256"]
        != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("density decision input identity or completeness differs")
    final_binding = _validate_recovery_binding(record["final_recovery_index"])
    _, observed_binding, indexed = _load_recovery_index(final_binding["path"])
    if observed_binding != final_binding:
        raise ValueError("final recovery index binding differs")
    expected = _build_decision_input(plan, sidecar, observed_binding, indexed)
    if record != expected:
        raise ValueError("density decision input does not replay")
    return dict(record)


__all__ = [
    "CANONICAL_UNIVERSE",
    "CANONICAL_UNIVERSE_SHA256",
    "EXHAUSTIVE92_CONTRACT",
    "TARGETED_CONTRACT",
    "build_decision_input_from_recovery_index",
    "build_plan_from_recovery_index",
    "build_sidecar",
    "validate_decision_input",
    "validate_plan",
    "validate_sidecar",
    "write_artifacts",
]
