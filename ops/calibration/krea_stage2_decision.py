#!/usr/bin/env python3
"""Recompute the fixed Krea Stage-2 confirmation and boundary decision.

The record produced here is scientific evidence only.  A PASS does not mutate
the production repository and does not authorize release or deployment.
"""

from __future__ import annotations

from decimal import Decimal
import hashlib
from typing import Any, Mapping

try:
    from . import krea_confirmation_admission
    from . import krea_decision
    from . import krea_provenance
    from . import krea_stage2_score
except ImportError:  # pragma: no cover
    import krea_confirmation_admission  # type: ignore[no-redef]
    import krea_decision  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_stage2_score  # type: ignore[no-redef]


KIND = "forge-krea-stage2-confirmation-decision"
SCHEMA = 1
FIXTURES = ("C1", "C2", "C3", "C4")
SEED_ROLES = ("A", "B")
PUBLIC = ("K2", "K3", "K4")
CONTROL = "K0"
FIELD_CAP = Decimal("0.01")
CONCEPT_CAP = Decimal("0.03")
BOUNDARY_MECHANICS = {
    "natural_completion": True,
    "planned_steps_completed": True,
    "upload_ready": True,
    "clean_telemetry": True,
    "decision_completed_before_export_reserve": True,
    "fallback_used": False,
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


def _canonical_file_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(krea_provenance.canonical_bytes(value) + b"\n").hexdigest()


def _validate_authority(controls: Mapping[str, Any]) -> dict[str, Any]:
    controls = _object(controls, "Stage-2 authority controls")
    keys = {
        "request",
        "request_file_sha256",
        "ratification",
        "ratification_file_sha256",
        "reveal",
        "reveal_file_sha256",
        "materialization",
        "materialization_file_sha256",
        "gpu_execution_authorization",
        "gpu_execution_authorization_file_sha256",
        "production_identity",
        "production_identity_file_sha256",
    }
    _exact(controls, keys, "Stage-2 authority controls")
    authorization = krea_confirmation_admission.validate_gpu_execution_authorization(
        controls["gpu_execution_authorization"],
        request=controls["request"],
        ratification=controls["ratification"],
        reveal=controls["reveal"],
        materialization=controls["materialization"],
        request_file_sha256=controls["request_file_sha256"],
        ratification_file_sha256=controls["ratification_file_sha256"],
        reveal_file_sha256=controls["reveal_file_sha256"],
        materialization_file_sha256=controls["materialization_file_sha256"],
        production_identity=controls["production_identity"],
        production_identity_file_sha256=controls["production_identity_file_sha256"],
    )
    if controls["gpu_execution_authorization_file_sha256"] != _canonical_file_sha(
        authorization
    ):
        raise ValueError("GPU authorization file SHA does not bind its record")
    return authorization


def _expected_authority_bindings(
    controls: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "waiver_finalist_freeze": {
            "file_sha256": authorization["waiver_freeze_file_sha256"],
            "freeze_sha256": authorization["waiver_freeze_sha256"],
        },
        "confirmation_materialization": {
            "file_sha256": controls["materialization_file_sha256"],
            "materialization_sha256": authorization["materialization_sha256"],
        },
        "owner_ratification": {
            "file_sha256": controls["ratification_file_sha256"],
            "ratification_sha256": authorization["ratification_sha256"],
        },
        "gpu_execution_authorization": {
            "file_sha256": controls["gpu_execution_authorization_file_sha256"],
            "gpu_execution_authorization_sha256": authorization[
                "gpu_execution_authorization_sha256"
            ],
        },
        "production_identity": {
            "file_sha256": controls["production_identity_file_sha256"],
            "production_identity_sha256": authorization["production_identity_sha256"],
        },
        "production_image_id": authorization["image_id"],
    }


def _validated_inputs(
    *,
    plans: Mapping[str, dict[str, Any]],
    aggregates: Mapping[str, dict[str, Any]],
    cell_controls: Mapping[str, Mapping[str, Any]],
    authority_controls: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    authorization = _validate_authority(authority_controls)
    expected_authority = _expected_authority_bindings(authority_controls, authorization)
    if set(plans) != set(aggregates) or set(plans) != set(cell_controls):
        raise ValueError(
            "Stage-2 decision plan, aggregate, and replay-control cells differ"
        )
    resolved_plans: dict[str, dict[str, Any]] = {}
    resolved_aggregates: dict[str, dict[str, Any]] = {}
    for cell in sorted(plans):
        controls = _object(cell_controls[cell], f"Stage-2 replay controls {cell}")
        _exact(
            controls,
            {
                "run_controls_by_family",
                "fixture_manifest",
                "fixture_manifest_file_sha256",
                "score_files_by_family",
            },
            f"Stage-2 replay controls {cell}",
        )
        plan = krea_stage2_score.validate_plan_with_run_controls(
            plans[cell],
            controls_by_family=controls["run_controls_by_family"],
        )
        if plan["cell_id"] != cell:
            raise ValueError("Stage-2 plan map key differs from cell id")
        for field, expected in expected_authority.items():
            if plan[field] != expected:
                raise ValueError(f"Stage-2 plan authority differs at {field}")
        if plan["created_at_utc"] <= authorization["authorized_at_utc"]:
            raise ValueError("Stage-2 score plan must postdate GPU authorization")
        aggregate = krea_stage2_score.validate_aggregate_with_score_files(
            aggregates[cell],
            plan=plan,
            fixture_manifest=controls["fixture_manifest"],
            fixture_manifest_file_sha256=controls["fixture_manifest_file_sha256"],
            score_files_by_family=controls["score_files_by_family"],
        )
        resolved_plans[cell] = plan
        resolved_aggregates[cell] = aggregate
    expected_confirmation = {
        f"{fixture}-{role}" for fixture in FIXTURES for role in SEED_ROLES
    }
    expected_boundary = {
        f"B-{hours}-{size}"
        for hours in ("0p5", "0p75", "1")
        for size in ("small", "large")
    }
    if set(resolved_plans) != expected_confirmation | expected_boundary:
        raise ValueError("Stage-2 decision requires the exact 8+6 cell matrix")
    candidate_families = {
        plan["candidate_family_id"] for plan in resolved_plans.values()
    }
    if len(candidate_families) != 1:
        raise ValueError("Stage-2 cells do not share one candidate family")
    candidate_family = next(iter(candidate_families))
    return (
        resolved_plans,
        resolved_aggregates,
        authorization,
        {**expected_authority, "candidate_family_id": candidate_family},
    )


def _losses(aggregate: Mapping[str, Any]) -> dict[str, Decimal]:
    return {
        receipt["family_id"]: Decimal(str(receipt["result"]["weighted_loss"]))
        for receipt in aggregate["receipts"]
    }


def _compute(
    plans: Mapping[str, dict[str, Any]],
    aggregates: Mapping[str, dict[str, Any]],
    *,
    candidate_family: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bool]]:
    concept_reductions: dict[str, Decimal] = {}
    concept_excesses: dict[str, Decimal] = {}
    concepts: list[dict[str, Any]] = []
    for fixture in FIXTURES:
        episodes: list[dict[str, Any]] = []
        reductions: list[Decimal] = []
        excesses: list[Decimal] = []
        for role in SEED_ROLES:
            cell = f"{fixture}-{role}"
            losses = _losses(aggregates[cell])
            expected = {candidate_family, CONTROL, *PUBLIC}
            if set(losses) != expected:
                raise ValueError("confirmation aggregate family coverage differs")
            candidate = losses[candidate_family]
            control = losses[CONTROL]
            if control <= 0:
                raise ValueError("K0 control loss must be positive")
            strongest_family = min(PUBLIC, key=lambda family: (losses[family], family))
            strongest = losses[strongest_family]
            if strongest <= 0:
                raise ValueError("public-reference loss must be positive")
            reduction = (control - candidate) / control
            excess = (candidate - strongest) / strongest
            reductions.append(reduction)
            excesses.append(excess)
            episodes.append(
                {
                    "cell_id": cell,
                    "seed_role": role,
                    "candidate_loss": float(candidate),
                    "control_loss": float(control),
                    "public_reference_losses": {
                        family: float(losses[family]) for family in PUBLIC
                    },
                    "strongest_public_reference_family_id": strongest_family,
                    "strongest_public_reference_loss": float(strongest),
                    "relative_reduction_vs_K0": float(reduction),
                    "relative_excess_vs_public_reference": float(excess),
                }
            )
        reduction = sum(reductions) / Decimal(len(reductions))
        excess = sum(excesses) / Decimal(len(excesses))
        concept_reductions[fixture] = reduction
        concept_excesses[fixture] = excess
        concepts.append(
            {
                "fixture_id": fixture,
                "relative_reduction_vs_K0": float(reduction),
                "relative_excess_vs_public_reference": float(excess),
                "point_win_or_tie": excess <= FIELD_CAP,
                "episodes": episodes,
            }
        )
    control_ci = krea_decision._bootstrap_ci(
        concept_reductions, label=f"stage2:{candidate_family}:vs:K0"
    )
    public_ci = krea_decision._bootstrap_ci(
        concept_excesses, label=f"stage2:{candidate_family}:vs:K2-K4"
    )
    boundary_results: list[dict[str, Any]] = []
    for cell in sorted(
        (name for name in plans if name.startswith("B-")),
        key=lambda name: (plans[name]["hours"], name),
    ):
        plan = plans[cell]
        candidate_rows = {row["family_id"]: row for row in plan["candidates"]}
        row = candidate_rows[candidate_family]
        mechanics = row.get("mechanics")
        passed = mechanics == BOUNDARY_MECHANICS
        boundary_results.append(
            {
                "cell_id": cell,
                "hours": plan["hours"],
                "dataset_boundary": "small" if cell.endswith("small") else "large",
                "candidate_sha256": row["candidate_sha256"],
                "mechanics": mechanics,
                "passed": passed,
            }
        )
    wins = sum(row["point_win_or_tie"] for row in concepts)
    gates = {
        "all_confirmation_runs_complete": len(concepts) == 4
        and all(len(row["episodes"]) == 2 for row in concepts),
        "control_superiority_95pct": Decimal(str(control_ci["lower"])) > 0,
        "public_reference_noninferiority_95pct": Decimal(str(public_ci["upper"]))
        <= FIELD_CAP,
        "three_of_four_point_wins_or_ties": wins >= 3,
        "no_concept_regression_over_0.03": max(concept_excesses.values())
        <= CONCEPT_CAP,
        "boundary_matrix_clean": len(boundary_results) == 6
        and all(row["passed"] for row in boundary_results),
        "decision_before_export_reserve_without_fallback": len(boundary_results) == 6
        and all(row["mechanics"] == BOUNDARY_MECHANICS for row in boundary_results),
        "stage2_production_surface_ratified": True,
    }
    metrics = {
        "control_relative_reduction_cluster_bootstrap": control_ci,
        "public_reference_relative_excess_cluster_bootstrap": public_ci,
        "point_wins_or_ties": wins,
        "point_win_or_tie_cap": float(FIELD_CAP),
        "concept_regression_cap": float(CONCEPT_CAP),
        "concept_results": concepts,
        "strongest_public_reference_rule": (
            "minimum exact loss among exhaustive K2-K4 local reproductions "
            "for the same concept and seed"
        ),
    }
    return metrics, boundary_results, gates


def _aggregate_bindings(
    plans: Mapping[str, Mapping[str, Any]],
    aggregates: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "cell_id": cell,
            "score_plan_sha256": plans[cell]["plan_sha256"],
            "aggregate_sha256": aggregates[cell]["aggregate_sha256"],
        }
        for cell in sorted(plans)
    ]


def build_decision(
    *,
    plans: Mapping[str, dict[str, Any]],
    aggregates: Mapping[str, dict[str, Any]],
    cell_controls: Mapping[str, Mapping[str, Any]],
    authority_controls: Mapping[str, Any],
    decided_at_utc: str,
) -> dict[str, Any]:
    resolved_plans, resolved_aggregates, authorization, common = _validated_inputs(
        plans=plans,
        aggregates=aggregates,
        cell_controls=cell_controls,
        authority_controls=authority_controls,
    )
    candidate_family = common["candidate_family_id"]
    metrics, boundary, gates = _compute(
        resolved_plans,
        resolved_aggregates,
        candidate_family=candidate_family,
    )
    passed = all(gates.values())
    blockers = [
        f"failed confirmation gate: {key}" for key, ok in gates.items() if not ok
    ]
    decided_at_utc = krea_stage2_score.krea_stage2_execution._utc(
        decided_at_utc, "Stage-2 decision time"
    )
    if decided_at_utc <= max(
        aggregate["emitted_at_utc"] for aggregate in resolved_aggregates.values()
    ):
        raise ValueError("Stage-2 decision must postdate every score aggregate")
    body = {
        "schema": SCHEMA,
        "kind": KIND,
        "phase": "confirmation",
        "decided_at_utc": decided_at_utc,
        "candidate_family_id": candidate_family,
        "outcome": "PASS" if passed else "FAIL",
        "blockers": blockers,
        "authority": {
            "gpu_execution_authorization_sha256": authorization[
                "gpu_execution_authorization_sha256"
            ],
            "gpu_execution_authorization_file_sha256": authority_controls[
                "gpu_execution_authorization_file_sha256"
            ],
            "production_identity_sha256": authorization["production_identity_sha256"],
            "production_image_id": authorization["image_id"],
            "waiver_freeze_sha256": authorization["waiver_freeze_sha256"],
            "materialization_sha256": authorization["materialization_sha256"],
        },
        "aggregate_bindings": _aggregate_bindings(resolved_plans, resolved_aggregates),
        "metrics": metrics,
        "gates": gates,
        "boundary_results": boundary,
        "confirmation_passed": passed,
        "release_review_required": True,
        "production_mutation_authorized": False,
        "release_authorized": False,
        "deployment_authorized": False,
        "win_guaranteed": False,
    }
    record = {**body, "decision_sha256": krea_provenance.canonical_sha256(body)}
    return validate_decision(
        record,
        plans=resolved_plans,
        aggregates=resolved_aggregates,
        cell_controls=cell_controls,
        authority_controls=authority_controls,
    )


def validate_decision(
    value: Any,
    *,
    plans: Mapping[str, dict[str, Any]],
    aggregates: Mapping[str, dict[str, Any]],
    cell_controls: Mapping[str, Mapping[str, Any]],
    authority_controls: Mapping[str, Any],
) -> dict[str, Any]:
    record = _object(value, "Stage-2 decision")
    keys = {
        "schema",
        "kind",
        "phase",
        "decided_at_utc",
        "candidate_family_id",
        "outcome",
        "blockers",
        "authority",
        "aggregate_bindings",
        "metrics",
        "gates",
        "boundary_results",
        "confirmation_passed",
        "release_review_required",
        "production_mutation_authorized",
        "release_authorized",
        "deployment_authorized",
        "win_guaranteed",
        "decision_sha256",
    }
    _exact(record, keys, "Stage-2 decision")
    body = {key: item for key, item in record.items() if key != "decision_sha256"}
    if (
        record["schema"] != SCHEMA
        or record["kind"] != KIND
        or record["phase"] != "confirmation"
        or record["outcome"] not in {"PASS", "FAIL"}
        or record["decision_sha256"] != krea_provenance.canonical_sha256(body)
        or record["release_review_required"] is not True
        or any(
            record[key] is not False
            for key in (
                "production_mutation_authorized",
                "release_authorized",
                "deployment_authorized",
                "win_guaranteed",
            )
        )
    ):
        raise ValueError("Stage-2 decision identity or authority differs")
    resolved_plans, resolved_aggregates, authorization, common = _validated_inputs(
        plans=plans,
        aggregates=aggregates,
        cell_controls=cell_controls,
        authority_controls=authority_controls,
    )
    metrics, boundary, gates = _compute(
        resolved_plans,
        resolved_aggregates,
        candidate_family=common["candidate_family_id"],
    )
    passed = all(gates.values())
    expected_authority = {
        "gpu_execution_authorization_sha256": authorization[
            "gpu_execution_authorization_sha256"
        ],
        "gpu_execution_authorization_file_sha256": authority_controls[
            "gpu_execution_authorization_file_sha256"
        ],
        "production_identity_sha256": authorization["production_identity_sha256"],
        "production_image_id": authorization["image_id"],
        "waiver_freeze_sha256": authorization["waiver_freeze_sha256"],
        "materialization_sha256": authorization["materialization_sha256"],
    }
    expected_blockers = [
        f"failed confirmation gate: {key}" for key, ok in gates.items() if not ok
    ]
    if (
        record["candidate_family_id"] != common["candidate_family_id"]
        or record["outcome"] != ("PASS" if passed else "FAIL")
        or record["blockers"] != expected_blockers
        or record["authority"] != expected_authority
        or record["aggregate_bindings"]
        != _aggregate_bindings(resolved_plans, resolved_aggregates)
        or record["metrics"] != metrics
        or record["gates"] != gates
        or record["boundary_results"] != boundary
        or record["confirmation_passed"] is not passed
    ):
        raise ValueError("Stage-2 decision does not recompute")
    decided_at_utc = krea_stage2_score.krea_stage2_execution._utc(
        record["decided_at_utc"], "Stage-2 decision time"
    )
    if decided_at_utc <= max(
        aggregate["emitted_at_utc"] for aggregate in resolved_aggregates.values()
    ):
        raise ValueError("Stage-2 decision must postdate every score aggregate")
    return dict(record)
