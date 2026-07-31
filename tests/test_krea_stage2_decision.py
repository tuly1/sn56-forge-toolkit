from __future__ import annotations

from copy import deepcopy

import pytest

from ops.calibration import krea_provenance
from ops.calibration import krea_stage2_decision as decision


MECHANICS = {
    "natural_completion": True,
    "planned_steps_completed": True,
    "upload_ready": True,
    "clean_telemetry": True,
    "decision_completed_before_export_reserve": True,
    "fallback_used": False,
}


def _sha(label: str) -> str:
    return krea_provenance.canonical_sha256({"label": label})


def _matrix(*, candidate_loss: float = 0.80) -> tuple[dict, dict]:
    plans = {}
    aggregates = {}
    for fixture in decision.FIXTURES:
        for role in decision.SEED_ROLES:
            cell = f"{fixture}-{role}"
            plans[cell] = {
                "cell_id": cell,
                "hours": "0.75",
                "candidate_family_id": "K1",
                "candidates": [
                    {"family_id": family, "mechanics": MECHANICS}
                    for family in ("K0", "K1", "K2", "K3", "K4")
                ],
                "plan_sha256": _sha(f"plan-{cell}"),
            }
            losses = {
                "K0": 1.0,
                "K1": candidate_loss,
                "K2": 0.805,
                "K3": 0.82,
                "K4": 0.83,
            }
            aggregates[cell] = {
                "receipts": [
                    {"family_id": family, "result": {"weighted_loss": loss}}
                    for family, loss in sorted(losses.items())
                ],
                "emitted_at_utc": "2026-08-01T00:00:02Z",
                "aggregate_sha256": _sha(f"aggregate-{cell}"),
            }
    for hours, value in (("0p5", "0.5"), ("0p75", "0.75"), ("1", "1.0")):
        for size in ("small", "large"):
            cell = f"B-{hours}-{size}"
            plans[cell] = {
                "cell_id": cell,
                "hours": value,
                "candidate_family_id": "K1",
                "candidates": [
                    {
                        "family_id": "K1",
                        "candidate_sha256": _sha(f"candidate-{cell}"),
                        "mechanics": dict(MECHANICS),
                    }
                ],
                "plan_sha256": _sha(f"plan-{cell}"),
            }
            aggregates[cell] = {
                "receipts": [],
                "emitted_at_utc": "2026-08-01T00:00:02Z",
                "aggregate_sha256": _sha(f"aggregate-{cell}"),
            }
    return plans, aggregates


def _authorization() -> dict:
    return {
        "authorized_at_utc": "2026-08-01T00:00:00Z",
        "waiver_freeze_file_sha256": _sha("freeze-file"),
        "waiver_freeze_sha256": _sha("freeze"),
        "materialization_file_sha256": _sha("materialization-file"),
        "materialization_sha256": _sha("materialization"),
        "ratification_file_sha256": _sha("ratification-file"),
        "ratification_sha256": _sha("ratification"),
        "gpu_execution_authorization_sha256": _sha("gpu-authorization"),
        "production_identity_file_sha256": _sha("identity-file"),
        "production_identity_sha256": _sha("identity"),
        "image_id": f"sha256:{_sha('image')}",
    }


def _common() -> dict:
    authorization = _authorization()
    return {
        "waiver_finalist_freeze": {
            "file_sha256": authorization["waiver_freeze_file_sha256"],
            "freeze_sha256": authorization["waiver_freeze_sha256"],
        },
        "confirmation_materialization": {
            "file_sha256": authorization["materialization_file_sha256"],
            "materialization_sha256": authorization["materialization_sha256"],
        },
        "owner_ratification": {
            "file_sha256": authorization["ratification_file_sha256"],
            "ratification_sha256": authorization["ratification_sha256"],
        },
        "gpu_execution_authorization": {
            "file_sha256": _sha("gpu-authorization-file"),
            "gpu_execution_authorization_sha256": authorization[
                "gpu_execution_authorization_sha256"
            ],
        },
        "production_identity": {
            "file_sha256": authorization["production_identity_file_sha256"],
            "production_identity_sha256": authorization["production_identity_sha256"],
        },
        "production_image_id": authorization["image_id"],
        "candidate_family_id": "K1",
    }


def test_compute_passes_only_when_quality_and_boundary_gates_all_pass() -> None:
    plans, aggregates = _matrix()
    metrics, boundary, gates = decision._compute(
        plans, aggregates, candidate_family="K1"
    )
    assert all(gates.values())
    assert metrics["point_wins_or_ties"] == 4
    assert len(boundary) == 6

    failed_plans = deepcopy(plans)
    failed_plans["B-0p5-small"]["candidates"][0]["mechanics"]["fallback_used"] = True
    _, boundary, gates = decision._compute(
        failed_plans, aggregates, candidate_family="K1"
    )
    assert gates["boundary_matrix_clean"] is False
    assert gates["decision_before_export_reserve_without_fallback"] is False
    assert (
        next(row for row in boundary if row["cell_id"] == "B-0p5-small")["passed"]
        is False
    )


def test_compute_rejects_candidate_that_regresses_against_public_field() -> None:
    plans, aggregates = _matrix(candidate_loss=0.85)
    metrics, _boundary, gates = decision._compute(
        plans, aggregates, candidate_family="K1"
    )
    assert metrics["point_wins_or_ties"] == 0
    assert gates["public_reference_noninferiority_95pct"] is False
    assert gates["three_of_four_point_wins_or_ties"] is False
    assert gates["no_concept_regression_over_0.03"] is False


def test_validated_inputs_replays_every_run_and_exact_score_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plans, aggregates = _matrix()
    authorization = _authorization()
    common = _common()
    for plan in plans.values():
        plan.update(
            {
                key: value
                for key, value in common.items()
                if key != "candidate_family_id"
            }
        )
        plan["created_at_utc"] = "2026-08-01T00:00:01Z"
    controls = {
        cell: {
            "run_controls_by_family": {"cell": cell},
            "fixture_manifest": {"cell": cell},
            "fixture_manifest_file_sha256": _sha(f"fixture-{cell}"),
            "score_files_by_family": {"cell": cell},
        }
        for cell in plans
    }
    replayed_runs = []
    replayed_scores = []
    monkeypatch.setattr(decision, "_validate_authority", lambda _value: authorization)
    monkeypatch.setattr(
        decision,
        "_expected_authority_bindings",
        lambda _controls, _authorization: {
            key: value for key, value in common.items() if key != "candidate_family_id"
        },
    )

    def replay_plan(value, *, controls_by_family):
        replayed_runs.append((value["cell_id"], controls_by_family))
        return value

    def replay_aggregate(
        value,
        *,
        plan,
        fixture_manifest,
        fixture_manifest_file_sha256,
        score_files_by_family,
    ):
        replayed_scores.append(
            (
                plan["cell_id"],
                fixture_manifest,
                fixture_manifest_file_sha256,
                score_files_by_family,
            )
        )
        return value

    monkeypatch.setattr(
        decision.krea_stage2_score, "validate_plan_with_run_controls", replay_plan
    )
    monkeypatch.setattr(
        decision.krea_stage2_score,
        "validate_aggregate_with_score_files",
        replay_aggregate,
    )
    resolved_plans, resolved_aggregates, _, resolved_common = (
        decision._validated_inputs(
            plans=plans,
            aggregates=aggregates,
            cell_controls=controls,
            authority_controls={},
        )
    )
    assert set(resolved_plans) == set(plans)
    assert set(resolved_aggregates) == set(aggregates)
    assert len(replayed_runs) == len(plans) == 14
    assert len(replayed_scores) == len(plans)
    assert resolved_common["candidate_family_id"] == "K1"


def test_decision_is_non_authorizing_and_must_postdate_all_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plans, aggregates = _matrix()
    authorization = _authorization()
    common = _common()
    monkeypatch.setattr(
        decision,
        "_validated_inputs",
        lambda **_kwargs: (plans, aggregates, authorization, common),
    )
    authority_controls = {
        "gpu_execution_authorization_file_sha256": _sha("gpu-authorization-file")
    }
    record = decision.build_decision(
        plans=plans,
        aggregates=aggregates,
        cell_controls={},
        authority_controls=authority_controls,
        decided_at_utc="2026-08-01T00:00:03Z",
    )
    assert record["outcome"] == "PASS"
    assert record["confirmation_passed"] is True
    assert record["release_authorized"] is False
    assert record["deployment_authorized"] is False
    assert record["win_guaranteed"] is False

    with pytest.raises(ValueError, match="postdate every score aggregate"):
        decision.build_decision(
            plans=plans,
            aggregates=aggregates,
            cell_controls={},
            authority_controls=authority_controls,
            decided_at_utc="2026-08-01T00:00:02Z",
        )
