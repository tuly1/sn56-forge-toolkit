from __future__ import annotations

from pathlib import Path

import pytest

from ops.calibration import krea_stage2_decision as legacy_decision
from ops.calibration import krea_stage2_endgame_decision as decision


def _aggregate(family: str, candidate_loss: float) -> dict:
    losses = {
        "K0": 1.0,
        "K2": 0.84,
        "K3": 0.85,
        "K4": 0.86,
        family: candidate_loss,
    }
    return {
        "receipts": [
            {"family_id": key, "result": {"weighted_loss": value}}
            for key, value in sorted(losses.items())
        ]
    }


def _aggregates(family: str, candidate_loss: float = 0.82) -> dict:
    return {
        f"C{fixture}-{seed}": _aggregate(family, candidate_loss)
        for fixture in range(1, 5)
        for seed in ("A", "B")
    }


def _boundary(clean: bool = True) -> dict:
    mechanics = dict(legacy_decision.BOUNDARY_MECHANICS)
    if not clean:
        mechanics["decision_completed_before_export_reserve"] = False
    return {
        cell: {
            "hours": hours,
            "candidate_sha256": str(index + 1) * 64,
            "mechanics": mechanics,
            "run_evidence_file_sha256": str(index + 2) * 64,
            "run_evidence_sha256": str(index + 3) * 64,
        }
        for index, (cell, hours) in enumerate(
            (
                ("B-0p5-small", "0.5"),
                ("B-0p5-large", "0.5"),
                ("B-0p75-small", "0.75"),
                ("B-0p75-large", "0.75"),
                ("B-1-small", "1.0"),
                ("B-1-large", "1.0"),
            )
        )
    }


def test_each_frozen_policy_uses_the_predeclared_confirmation_gates() -> None:
    for family in decision.ACTIVE_POLICIES:
        result = decision._compute_policy(
            family=family,
            aggregates=_aggregates(family),
            boundary=_boundary(),
        )
        assert result["candidate_family_id"] == family
        assert result["outcome"] == "PASS"
        assert result["confirmation_passed"] is True
        assert all(result["gates"].values())


def test_boundary_surprise_fails_policy_instead_of_being_hidden() -> None:
    result = decision._compute_policy(
        family="K1", aggregates=_aggregates("K1"), boundary=_boundary(clean=False)
    )

    assert result["outcome"] == "FAIL"
    assert result["gates"]["boundary_matrix_clean"] is False
    assert result["gates"]["decision_before_export_reserve_without_fallback"] is False


def test_decision_never_invents_router_or_release_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        decision,
        "_load_scored_groups",
        lambda queue: (
            {"K1": _aggregates("K1"), "K5": _aggregates("K5")},
            [],
            "2026-08-02T10:00:00Z",
        ),
    )
    monkeypatch.setattr(
        decision,
        "_boundary_by_policy",
        lambda **kwargs: {"K1": _boundary(), "K5": _boundary()},
    )
    freeze = {
        "D1_winner_family_id": "K1",
        "D2_winner_family_id": "K5",
        "all_family_checkpoint_rules": {
            "K1": {"target_fraction": 0.9},
            "K5": {"target_fraction": 0.5},
        },
    }
    matrix = {
        "matrix_sha256": "1" * 64,
        "active_variant_family_ids": ["K1", "K5"],
        "freeze": {"file_sha256": "2" * 64, "freeze_sha256": "3" * 64},
        "production_identity": {
            "file_sha256": "4" * 64,
            "production_identity_sha256": "5" * 64,
        },
        "production_image_id": "sha256:" + "6" * 64,
    }
    body = decision._decision_body(
        matrix=matrix,
        plan_set={"plan_set_sha256": "7" * 64},
        authority={"waiver_finalist_freeze": freeze},
        training_gate={
            "completed_at_utc": "2026-08-02T10:01:00Z",
            "gate_sha256": "8" * 64,
        },
        score_queue={"score_queue_sha256": "9" * 64},
        score_gate={
            "completed_at_utc": "2026-08-02T10:02:00Z",
            "gate_sha256": "a" * 64,
        },
        decided_at_utc="2026-08-02T10:03:00Z",
    )

    assert body["overall_confirmation_passed"] is True
    assert body["production_dataset_count_router_predeclared"] is False
    assert body["production_routing_authority"] is False
    assert body["release_family_selected"] is None
    assert body["release_review_required"] is True
    assert body["release_authorized"] is False


def test_missing_completion_gate_fails_before_any_gate_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = {"matrix_sha256": "1" * 64}
    plan_set = {"plan_set_sha256": "2" * 64}
    monkeypatch.setattr(
        decision.krea_stage2_endgame_matrix,
        "validate_matrix",
        lambda value, **kwargs: matrix,
    )
    monkeypatch.setattr(
        decision.training, "validate_plan_set", lambda value, matrix: plan_set
    )
    monkeypatch.setattr(
        decision.training,
        "_validate_authority_bundle",
        lambda value: {
            "waiver_finalist_freeze": {},
            "production_identity": {},
        },
    )
    monkeypatch.setattr(
        decision.scoring,
        "_validate_queue",
        lambda value: {
            "matrix_sha256": "1" * 64,
            "training_plan_set_sha256": "2" * 64,
        },
    )
    monkeypatch.setattr(
        decision.training,
        "seal_exact60_gate",
        lambda **kwargs: pytest.fail("missing-gate path attempted a replay"),
    )

    with pytest.raises(ValueError, match="pre-existing completion gates"):
        decision.build_decision(
            matrix=matrix,
            plan_set=plan_set,
            authority_bundle={},
            training_gate_path=tmp_path / "missing-training.json",
            score_queue={},
            score_gate_path=tmp_path / "missing-score.json",
            decided_at_utc="2026-08-02T10:00:00Z",
        )
