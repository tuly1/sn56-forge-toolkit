"""Adversarial tests for the receipt-index-backed Krea density gate."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping

import pytest

_CALIBRATION = Path(__file__).parents[1] / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))

import krea_density_gate as gate  # noqa: E402
import krea_provenance  # noqa: E402


def _digest(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _file_binding(task_id: str, kind: str) -> dict[str, Any]:
    return {
        "path": f"/receipt-fixture/{task_id}/{kind}.json",
        "bytes": len(task_id) + len(kind) + 1,
        "file_sha256": _digest(f"{task_id}:{kind}"),
    }


def _artifact(source: Mapping[str, Any], *, loss: float = 1.0) -> dict[str, Any]:
    task_id = source["task_id"]
    candidate = _file_binding(task_id, "candidate")
    result = {
        **_file_binding(task_id, "result"),
        "semantic_sha256": _digest(f"{task_id}:result-semantic"),
        "weighted_loss": loss,
    }
    return {
        "task_id": task_id,
        "fixture_id": source["fixture"],
        "family_id": source["family"] or "ZERO",
        "step": source["step"],
        "is_final": source["label"].startswith("final-")
        or source["family"] is None,
        "zero_control": source["family"] is None,
        "coverage_tier": (
            gate.EXHAUSTIVE_BACKFILL
            if task_id == gate.RELIEF_TASK_ID
            else source["universe_tier"]
        ),
        "selection_eligible": True,
        "expected_candidate_sha256": candidate["file_sha256"],
        "validated_artifact": {
            "candidate": candidate,
            "result": result,
            "evidence": _file_binding(task_id, "evidence"),
            "receipt": _file_binding(task_id, "receipt"),
        },
    }


def _recovery_index(label: str) -> dict[str, Any]:
    return {
        "coverage_ledger": {
            "file_sha256": _digest(f"{label}:coverage-ledger")
        },
        "artifacts": [_artifact(row) for row in gate.CANONICAL_UNIVERSE],
    }


@pytest.fixture
def recovery_indexes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    initial_path = tmp_path / "initial-recovery-index.json"
    final_path = tmp_path / "final-recovery-index.json"
    initial = _recovery_index("initial")
    final = _recovery_index("final")
    indexes = {
        initial_path: initial,
        final_path: final,
    }
    calls: list[Path] = []

    def load_index(path: Path) -> tuple[dict[str, Any], str]:
        normalized = Path(path)
        calls.append(normalized)
        value = deepcopy(indexes[normalized])
        body = {key: item for key, item in value.items() if key != "index_sha256"}
        # The density boundary treats load_index as the receipt validator.  The
        # fixture returns fresh hash identities so byte drift remains visible
        # without recreating the recovery module's filesystem graph.
        value["index_sha256"] = _digest(repr(body))
        return value, _digest(f"file:{repr(value)}")

    monkeypatch.setattr(gate.krea_recovery_evidence, "load_index", load_index)
    return {
        "initial_path": initial_path,
        "final_path": final_path,
        "initial": initial,
        "final": final,
        "calls": calls,
    }


def _artifact_by_id(index: dict[str, Any], task_id: str) -> dict[str, Any]:
    return next(row for row in index["artifacts"] if row["task_id"] == task_id)


def _set_loss(index: dict[str, Any], task_id: str, value: float) -> None:
    _artifact_by_id(index, task_id)["validated_artifact"]["result"][
        "weighted_loss"
    ] = value


def _restore(target: dict[str, Any], snapshot: dict[str, Any]) -> None:
    target.clear()
    target.update(deepcopy(snapshot))


def _reseal(value: dict[str, Any], key: str) -> dict[str, Any]:
    body = {name: item for name, item in value.items() if name != key}
    return {**body, key: krea_provenance.canonical_sha256(body)}


def _plan(
    recovery: dict[str, Any],
    *,
    additional_target_count: int = 11,
    contract: str = gate.TARGETED_CONTRACT,
) -> dict[str, Any]:
    return gate.build_plan_from_recovery_index(
        recovery["initial_path"],
        contract=contract,
        additional_target_count=additional_target_count,
    )


def _decision(
    recovery: dict[str, Any], plan: dict[str, Any], sidecar: dict[str, Any]
) -> dict[str, Any]:
    return gate.build_decision_input_from_recovery_index(
        plan,
        sidecar,
        recovery["final_path"],
    )


def test_canonical_universe_has_exact_seed_a_tiers_and_geometry() -> None:
    assert len(gate.CANONICAL_UNIVERSE) == 92
    assert len({row["task_id"] for row in gate.CANONICAL_UNIVERSE}) == 92
    counts: dict[str, int] = {}
    for row in gate.CANONICAL_UNIVERSE:
        counts[row["universe_tier"]] = counts.get(row["universe_tier"], 0) + 1
        assert row["seed_role"] == "A"
        assert row["image_exposures"] == row["step"]
    assert counts == {
        gate.SPARSE_PRIMARY: 56,
        gate.INDEPENDENT_ZERO: 2,
        gate.RELIEF_NEIGHBOR_PROMOTED: 1,
        gate.EXHAUSTIVE_BACKFILL: 33,
    }
    relief = next(
        row for row in gate.CANONICAL_UNIVERSE if row["task_id"] == gate.RELIEF_TASK_ID
    )
    assert relief == {
        "task_id": "d1-k1-step522",
        "fixture": "D1",
        "family": "K1",
        "label": "step-522",
        "seed_role": "A",
        "step": 522,
        "image_exposures": 522,
        "universe_tier": gate.RELIEF_NEIGHBOR_PROMOTED,
    }


@pytest.mark.parametrize("additional", [0, 1, 8, 9, 10, 11])
def test_receipt_index_target_plan_is_bounded_and_deterministic(
    recovery_indexes: dict[str, Any], additional: int
) -> None:
    first = _plan(recovery_indexes, additional_target_count=additional)
    second = _plan(recovery_indexes, additional_target_count=additional)
    assert first == second
    assert first["selected_count"] == 59 + additional
    assert first["targeted_backfill_count_including_relief"] == 1 + additional
    assert first["targeted_backfill_count_including_relief"] <= 12
    assert first["target_plan_sha256"] == krea_provenance.canonical_sha256(
        {key: value for key, value in first.items() if key != "target_plan_sha256"}
    )
    assert recovery_indexes["calls"] == [
        recovery_indexes["initial_path"],
        recovery_indexes["initial_path"],
    ]


def test_returned_plan_and_public_universe_cannot_mutate_frozen_policy(
    recovery_indexes: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _plan(recovery_indexes)
    frozen_policy = deepcopy(first["family_tie_break_policy"])
    first["family_tie_break_policy"]["uncertainty_band"] = 99.0
    first["family_tie_break_policy"]["family_preference"].reverse()

    second = _plan(recovery_indexes)
    assert second["family_tie_break_policy"] == frozen_policy
    assert gate.krea_decision.FAMILY_TIE_BREAK_POLICY == frozen_policy
    with pytest.raises(TypeError):
        gate.CANONICAL_UNIVERSE[0]["image_exposures"] = 999

    monkeypatch.setitem(
        gate.krea_decision.FAMILY_TIE_BREAK_POLICY, "uncertainty_band", 0.02
    )
    with pytest.raises(RuntimeError, match="differs from the frozen"):
        _plan(recovery_indexes)


def test_full_target_plan_covers_nine_cells_plus_two_fixture_bonuses(
    recovery_indexes: dict[str, Any],
) -> None:
    plan = _plan(recovery_indexes)
    targeted = [
        row for row in plan["rows"] if row["plan_tier"] == gate.TARGETED_BACKFILL
    ]
    assert all(row["family"] != "K0" for row in targeted)
    counts: dict[tuple[str, str], int] = {}
    for row in targeted:
        cell = (row["fixture"], row["family"])
        counts[cell] = counts.get(cell, 0) + 1
    assert set(counts) == {
        (fixture, family)
        for fixture in ("D1", "D2")
        for family in ("K1", "K2", "K3", "K4", "K5")
        if (fixture, family) != ("D1", "K1")
    }
    assert sorted(counts.values()) == [1] * 7 + [2, 2]
    assert counts[("D1", "K2")] == 2
    assert counts[("D2", "K2")] == 2


def test_relief_promotion_keeps_logical_and_real_ledger_tiers_distinct(
    recovery_indexes: dict[str, Any],
) -> None:
    plan = _plan(recovery_indexes, additional_target_count=0)
    relief = next(
        row for row in plan["rows"] if row["task_id"] == gate.RELIEF_TASK_ID
    )
    assert relief["universe_tier"] == gate.RELIEF_NEIGHBOR_PROMOTED
    assert relief["plan_tier"] == gate.RELIEF_NEIGHBOR_PROMOTED
    assert relief["ledger_coverage_tier"] == gate.EXHAUSTIVE_BACKFILL

    sidecar = gate.build_sidecar(plan)
    decision = _decision(recovery_indexes, plan, sidecar)
    decision_relief = next(
        row
        for row in decision["candidate_rows_for_krea_decision"]
        if row["task_id"] == gate.RELIEF_TASK_ID
    )
    assert decision_relief["plan_tier"] == gate.RELIEF_NEIGHBOR_PROMOTED
    assert decision_relief["ledger_coverage_tier"] == gate.EXHAUSTIVE_BACKFILL


def test_peak_adjacency_uses_exact_minimum_not_family_uncertainty_band(
    recovery_indexes: dict[str, Any],
) -> None:
    sparse = [
        row
        for row in gate.CANONICAL_UNIVERSE
        if row["fixture"] == "D1"
        and row["family"] == "K2"
        and row["universe_tier"] == gate.SPARSE_PRIMARY
    ]
    _set_loss(recovery_indexes["initial"], sparse[0]["task_id"], 0.030)
    for index, row in enumerate(sparse[1:], start=1):
        _set_loss(
            recovery_indexes["initial"], row["task_id"], 0.031 + index / 1000
        )
    plan = _plan(recovery_indexes, additional_target_count=9)
    selected = {
        row["task_id"]
        for row in plan["rows"]
        if row["selected"] and row["plan_tier"] == gate.TARGETED_BACKFILL
    }
    assert "d1-k2-step174" in selected
    assert "d1-k2-step522" not in selected


def test_exact_decimal_uncertainty_boundary_is_inclusive(
    recovery_indexes: dict[str, Any],
) -> None:
    for row in gate.CANONICAL_UNIVERSE:
        if row["fixture"] == "D1" and row["universe_tier"] == gate.SPARSE_PRIMARY:
            if row["family"] == "K3":
                _set_loss(recovery_indexes["initial"], row["task_id"], 0.0045)
            elif row["family"] == "K2":
                _set_loss(recovery_indexes["initial"], row["task_id"], 0.0145)

    plan = _plan(recovery_indexes, additional_target_count=1)
    targeted = [
        row for row in plan["rows"] if row["plan_tier"] == gate.TARGETED_BACKFILL
    ]
    assert len(targeted) == 1
    # K2 is exactly 0.01 above K3 and therefore remains inside the inclusive
    # band; the frozen family preference chooses K2.
    assert targeted[0]["fixture"] == "D1"
    assert targeted[0]["family"] == "K2"


def test_plan_builder_rejects_partial_duplicate_nonfinite_and_bad_target_counts(
    recovery_indexes: dict[str, Any],
) -> None:
    source = recovery_indexes["initial"]
    original = deepcopy(source)

    source["artifacts"].pop()
    with pytest.raises(ValueError, match="canonical 92-row universe"):
        _plan(recovery_indexes)
    _restore(source, original)

    source["artifacts"].append(deepcopy(source["artifacts"][0]))
    with pytest.raises(ValueError, match="canonical 92-row universe"):
        _plan(recovery_indexes)
    _restore(source, original)

    sparse_id = next(
        row["task_id"]
        for row in gate.CANONICAL_UNIVERSE
        if row["universe_tier"] == gate.SPARSE_PRIMARY
    )
    _set_loss(source, sparse_id, float("nan"))
    with pytest.raises(ValueError, match="finite"):
        _plan(recovery_indexes)
    _restore(source, original)

    for count in (-1, 12, True):
        with pytest.raises(ValueError, match=r"\[0,11\]"):
            _plan(recovery_indexes, additional_target_count=count)


def test_plan_builder_rejects_bad_receipt_identity_and_ineligible_sparse_row(
    recovery_indexes: dict[str, Any],
) -> None:
    source = recovery_indexes["initial"]
    original = deepcopy(source)
    sparse = next(
        row
        for row in source["artifacts"]
        if row["coverage_tier"] == gate.SPARSE_PRIMARY
    )

    sparse["selection_eligible"] = False
    with pytest.raises(ValueError, match="not selection eligible"):
        _plan(recovery_indexes)
    _restore(source, original)

    _artifact_by_id(source, sparse["task_id"])["step"] += 1
    with pytest.raises(ValueError, match="target identity differs"):
        _plan(recovery_indexes)
    _restore(source, original)

    _artifact_by_id(source, sparse["task_id"])["coverage_tier"] = (
        gate.EXHAUSTIVE_BACKFILL
    )
    with pytest.raises(ValueError, match="target identity differs"):
        _plan(recovery_indexes)


def test_plan_rejects_seed_b_duplicate_k0_selection_and_hash_tampering(
    recovery_indexes: dict[str, Any],
) -> None:
    original = _plan(recovery_indexes)

    changed = deepcopy(original)
    changed["seed_role"] = "B"
    changed = _reseal(changed, "target_plan_sha256")
    with pytest.raises(ValueError, match="identity or frozen policy"):
        gate.validate_plan(changed)

    changed = deepcopy(original)
    changed["rows"][1] = deepcopy(changed["rows"][0])
    changed = _reseal(changed, "target_plan_sha256")
    with pytest.raises(ValueError, match="canonical Seed-A"):
        gate.validate_plan(changed)

    changed = deepcopy(original)
    k0 = next(
        row
        for row in changed["rows"]
        if row["family"] == "K0"
        and row["universe_tier"] == gate.EXHAUSTIVE_BACKFILL
    )
    selected_target = next(
        row for row in changed["rows"] if row["plan_tier"] == gate.TARGETED_BACKFILL
    )
    selected_target["selected"] = False
    selected_target["plan_tier"] = gate.EXHAUSTIVE_BACKFILL
    k0["selected"] = True
    k0["plan_tier"] = gate.TARGETED_BACKFILL
    changed = _reseal(changed, "target_plan_sha256")
    with pytest.raises(ValueError, match="sealed primary-loss selection"):
        gate.validate_plan(changed)

    changed = deepcopy(original)
    selected_target = next(
        row for row in changed["rows"] if row["plan_tier"] == gate.TARGETED_BACKFILL
    )
    replacement = next(
        row
        for row in changed["rows"]
        if not row["selected"]
        and row["family"] != "K0"
        and row["universe_tier"] == gate.EXHAUSTIVE_BACKFILL
    )
    selected_target["selected"] = False
    selected_target["plan_tier"] = gate.EXHAUSTIVE_BACKFILL
    replacement["selected"] = True
    replacement["plan_tier"] = gate.TARGETED_BACKFILL
    changed = _reseal(changed, "target_plan_sha256")
    with pytest.raises(ValueError, match="sealed primary-loss selection"):
        gate.validate_plan(changed)

    changed = deepcopy(original)
    changed["target_plan_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash seal"):
        gate.validate_plan(changed)

    changed = deepcopy(original)
    changed["rows"][0]["step"] += 1
    changed = _reseal(changed, "target_plan_sha256")
    with pytest.raises(ValueError, match="canonical Seed-A"):
        gate.validate_plan(changed)


def test_plan_validation_replays_its_bound_receipt_index(
    recovery_indexes: dict[str, Any],
) -> None:
    plan = _plan(recovery_indexes)
    sparse_id = plan["primary_losses"][0]["task_id"]
    _set_loss(recovery_indexes["initial"], sparse_id, 0.125)
    with pytest.raises(ValueError, match="revalidated recovery index"):
        gate.validate_plan(plan)


@pytest.mark.parametrize(
    "binding_key",
    ["candidate_binding", "result_binding", "evidence_binding", "receipt_binding"],
)
def test_plan_rejects_resealed_sparse_loss_and_binding_tampering(
    recovery_indexes: dict[str, Any], binding_key: str
) -> None:
    original = _plan(recovery_indexes)

    changed = deepcopy(original)
    changed["primary_losses"][0]["weighted_loss"] = 1.001
    changed = _reseal(changed, "target_plan_sha256")
    with pytest.raises(ValueError, match="revalidated recovery index"):
        gate.validate_plan(changed)

    changed = deepcopy(original)
    changed["primary_losses"][0][binding_key]["file_sha256"] = _digest(
        f"tampered:{binding_key}"
    )
    changed = _reseal(changed, "target_plan_sha256")
    with pytest.raises(ValueError, match="revalidated recovery index"):
        gate.validate_plan(changed)


def test_sidecar_is_hash_sealed_and_exactly_equivalent(
    recovery_indexes: dict[str, Any],
) -> None:
    plan = _plan(recovery_indexes)
    sidecar = gate.build_sidecar(plan)
    assert gate.validate_sidecar(plan, sidecar) == sidecar

    changed = deepcopy(sidecar)
    changed["rows"][0]["selected"] = False
    changed = _reseal(changed, "sidecar_sha256")
    with pytest.raises(ValueError, match="exactly equivalent"):
        gate.validate_sidecar(plan, changed)

    other_plan = _plan(recovery_indexes, additional_target_count=0)
    with pytest.raises(ValueError, match="exactly equivalent"):
        gate.validate_sidecar(other_plan, sidecar)


def test_decision_builder_exposes_every_and_only_selected_receipt_row(
    recovery_indexes: dict[str, Any],
) -> None:
    plan = _plan(recovery_indexes, additional_target_count=0)
    sidecar = gate.build_sidecar(plan)
    decision = _decision(recovery_indexes, plan, sidecar)
    expected_ids = [row["task_id"] for row in plan["rows"] if row["selected"]]

    assert decision["complete"] is True
    assert decision["score_count"] == 59
    assert "finalists" not in decision
    assert [
        row["task_id"] for row in decision["candidate_rows_for_krea_decision"]
    ] == expected_ids
    assert sum(
        row["family"] is None
        for row in decision["candidate_rows_for_krea_decision"]
    ) == 2
    # The final receipt index is deliberately over-complete; the density gate
    # still exports only the plan-selected subset.
    assert all(row["selection_eligible"] for row in recovery_indexes["final"]["artifacts"])
    assert (
        gate.validate_decision_input(
            decision, plan_value=plan, sidecar_value=sidecar
        )
        == decision
    )


def test_decision_builder_rejects_partial_duplicate_and_bad_candidate_receipts(
    recovery_indexes: dict[str, Any],
) -> None:
    plan = _plan(recovery_indexes, additional_target_count=0)
    sidecar = gate.build_sidecar(plan)
    source = recovery_indexes["final"]
    original = deepcopy(source)
    selected_id = next(row["task_id"] for row in plan["rows"] if row["selected"])

    _artifact_by_id(source, selected_id)["selection_eligible"] = False
    with pytest.raises(ValueError, match="selected recovery row is not selection eligible"):
        _decision(recovery_indexes, plan, sidecar)
    _restore(source, original)

    source["artifacts"].pop()
    with pytest.raises(ValueError, match="canonical 92-row universe"):
        _decision(recovery_indexes, plan, sidecar)
    _restore(source, original)

    source["artifacts"][-1] = deepcopy(source["artifacts"][0])
    with pytest.raises(ValueError, match="canonical 92-row universe"):
        _decision(recovery_indexes, plan, sidecar)
    _restore(source, original)

    validated = _artifact_by_id(source, selected_id)["validated_artifact"]
    validated["candidate"]["file_sha256"] = _digest("wrong-candidate")
    with pytest.raises(ValueError, match="selected candidate binding differs"):
        _decision(recovery_indexes, plan, sidecar)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fixture_id", "D9"),
        ("family_id", "K9"),
        ("step", 999),
        ("is_final", True),
        ("zero_control", True),
        ("coverage_tier", gate.EXHAUSTIVE_BACKFILL),
    ],
)
def test_decision_builder_rejects_receipt_identity_and_geometry_drift(
    recovery_indexes: dict[str, Any], field: str, value: object
) -> None:
    plan = _plan(recovery_indexes, additional_target_count=0)
    sidecar = gate.build_sidecar(plan)
    selected = next(
        row
        for row in plan["rows"]
        if row["selected"]
        and row["family"] == "K0"
        and row["universe_tier"] == gate.SPARSE_PRIMARY
        and not row["label"].startswith("final-")
    )
    _artifact_by_id(recovery_indexes["final"], selected["task_id"])[field] = value
    with pytest.raises(ValueError, match="target identity differs"):
        _decision(recovery_indexes, plan, sidecar)


def test_decision_builder_rejects_nonfinite_and_plan_bound_sparse_loss_drift(
    recovery_indexes: dict[str, Any],
) -> None:
    plan = _plan(recovery_indexes, additional_target_count=0)
    sidecar = gate.build_sidecar(plan)
    sparse_id = plan["primary_losses"][0]["task_id"]
    source = recovery_indexes["final"]
    original = deepcopy(source)

    _set_loss(source, sparse_id, float("inf"))
    with pytest.raises(ValueError, match="finite"):
        _decision(recovery_indexes, plan, sidecar)
    _restore(source, original)

    _set_loss(source, sparse_id, 1.001)
    with pytest.raises(ValueError, match="plan-bound recovery index"):
        _decision(recovery_indexes, plan, sidecar)


@pytest.mark.parametrize(
    "binding_key", ["candidate", "result", "evidence", "receipt"]
)
def test_decision_builder_requires_full_sparse_binding_continuity(
    recovery_indexes: dict[str, Any], binding_key: str
) -> None:
    plan = _plan(recovery_indexes, additional_target_count=0)
    sidecar = gate.build_sidecar(plan)
    sparse_id = plan["primary_losses"][0]["task_id"]
    artifact = _artifact_by_id(recovery_indexes["final"], sparse_id)
    artifact["validated_artifact"][binding_key]["file_sha256"] = _digest(
        f"replacement:{binding_key}"
    )
    if binding_key == "candidate":
        # Preserve the final index's own candidate/ledger agreement so this
        # probes initial-to-final continuity rather than the local SHA guard.
        artifact["expected_candidate_sha256"] = artifact["validated_artifact"][
            "candidate"
        ]["file_sha256"]
    with pytest.raises(ValueError, match="receipt/candidate/result/evidence"):
        _decision(recovery_indexes, plan, sidecar)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed_role", "B"),
        ("plan_tier", gate.EXHAUSTIVE_BACKFILL),
        ("image_exposures", 999),
        ("weighted_loss", 999.0),
    ],
)
def test_validate_decision_input_rejects_row_replay_drift(
    recovery_indexes: dict[str, Any], field: str, value: object
) -> None:
    plan = _plan(recovery_indexes, additional_target_count=0)
    sidecar = gate.build_sidecar(plan)
    decision = _decision(recovery_indexes, plan, sidecar)
    changed = deepcopy(decision)
    changed["candidate_rows_for_krea_decision"][0][field] = value
    changed = _reseal(changed, "decision_input_sha256")
    with pytest.raises(ValueError, match="does not replay"):
        gate.validate_decision_input(
            changed, plan_value=plan, sidecar_value=sidecar
        )


def test_validate_decision_input_rejects_completeness_hash_and_final_index_drift(
    recovery_indexes: dict[str, Any],
) -> None:
    plan = _plan(recovery_indexes, additional_target_count=0)
    sidecar = gate.build_sidecar(plan)
    decision = _decision(recovery_indexes, plan, sidecar)

    changed = deepcopy(decision)
    changed["score_count"] -= 1
    changed = _reseal(changed, "decision_input_sha256")
    with pytest.raises(ValueError, match="identity or completeness"):
        gate.validate_decision_input(
            changed, plan_value=plan, sidecar_value=sidecar
        )

    changed = deepcopy(decision)
    changed["decision_input_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity or completeness"):
        gate.validate_decision_input(
            changed, plan_value=plan, sidecar_value=sidecar
        )

    selected_id = decision["candidate_rows_for_krea_decision"][-1]["task_id"]
    _set_loss(recovery_indexes["final"], selected_id, 7.0)
    with pytest.raises(ValueError, match="final recovery index binding differs"):
        gate.validate_decision_input(
            decision, plan_value=plan, sidecar_value=sidecar
        )


def test_legacy_exhaustive92_is_a_distinct_complete_receipt_contract(
    recovery_indexes: dict[str, Any],
) -> None:
    plan = _plan(recovery_indexes, contract=gate.EXHAUSTIVE92_CONTRACT)
    assert plan["kind"] == gate.EXHAUSTIVE_PLAN_KIND
    assert plan["selected_count"] == 92
    assert all(row["selected"] for row in plan["rows"])
    assert all(row["plan_tier"] == row["universe_tier"] for row in plan["rows"])
    sidecar = gate.build_sidecar(plan)
    decision = _decision(recovery_indexes, plan, sidecar)
    assert decision["score_count"] == 92
    assert (
        gate.validate_decision_input(
            decision, plan_value=plan, sidecar_value=sidecar
        )
        == decision
    )
    with pytest.raises(ValueError, match="does not accept a targeted row count"):
        _plan(
            recovery_indexes,
            contract=gate.EXHAUSTIVE92_CONTRACT,
            additional_target_count=0,
        )


def test_write_artifacts_uses_canonical_json_and_create_only(
    recovery_indexes: dict[str, Any], tmp_path: Path
) -> None:
    plan = _plan(recovery_indexes, additional_target_count=0)
    plan_path = tmp_path / "plan.json"
    sidecar_path = tmp_path / "sidecar.json"
    file_hashes = gate.write_artifacts(plan_path, sidecar_path, plan)
    assert file_hashes == tuple(
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (plan_path, sidecar_path)
    )
    assert plan_path.read_bytes() == krea_provenance.canonical_bytes(plan) + b"\n"
    with pytest.raises(FileExistsError):
        gate.write_artifacts(plan_path, tmp_path / "other.json", plan)
