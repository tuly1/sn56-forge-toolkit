"""Governance tests for the Week-5 evidence ledger."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ops.calibration import evidence_ledger


_LEDGER = (
    Path(__file__).parents[1] / "ops" / "calibration" / "week5" / "evidence-ledger.json"
)


def _document():
    return json.loads(_LEDGER.read_text(encoding="utf-8"))


def test_current_ledger_is_schema_valid_but_named_dris_are_missing():
    document = _document()
    evidence_ledger.validate_ledger(document)
    with pytest.raises(evidence_ledger.LedgerValidationError, match="names missing"):
        evidence_ledger.validate_ledger(document, require_named_dris=True)


def test_unknown_readiness_cannot_be_null_or_missing():
    null = _document()
    null["models"][0]["round1_ready"] = None
    with pytest.raises(evidence_ledger.LedgerValidationError, match="must be boolean"):
        evidence_ledger.validate_ledger(null)

    missing = _document()
    del missing["models"][0]["boss_ready"]
    with pytest.raises(evidence_ledger.LedgerValidationError, match="schema mismatch"):
        evidence_ledger.validate_ledger(missing)


def test_win_ready_cannot_be_asserted_while_any_gate_is_red():
    document = _document()
    document["models"][0]["win_ready"] = True
    with pytest.raises(evidence_ledger.LedgerValidationError, match="conjunction"):
        evidence_ledger.validate_ledger(document)


def test_all_win_gates_require_win_ready_true():
    document = _document()
    model = document["models"][0]
    for key in (
        "mechanics_evidence_pass",
        "mechanics_ready",
        "quality_result_pass",
        "quality_evidenced",
        "field_parity_result_pass",
        "field_parity_ready",
        "round1_ready",
        "boss_ready",
        "public_bundle_scrub_pass",
    ):
        model[key] = True
    model["fixtures"] = 4
    model["training_seeds"] = 1
    model["evaluation_rows"] = 24
    model["public_arms_reproduced"] = ["K2"]
    model["post_reserve_window_utilization"] = 0.91
    model["selector_status"] = "deterministic_policy_validated"
    model["worst_cell_regret"] = 0.005
    model["worst_cell_regret_cap"] = 0.01
    model["owner"] = "Alice Owner"
    model["reviewer"] = "Rita Reviewer"
    model["open_risks"] = []
    for key in (
        "public_arm_provenance_manifest_sha256",
        "evidence_manifest_sha256",
        "quality_result_sha256",
        "field_parity_result_sha256",
        "public_bundle_sha256",
        "rules_contract_sha",
        "round1_evidence_sha256",
        "boss_evidence_sha256",
    ):
        model[key] = "a" * 64
    with pytest.raises(evidence_ledger.LedgerValidationError, match="conjunction"):
        evidence_ledger.validate_ledger(document)
    model["win_ready"] = True
    evidence_ledger.validate_ledger(document)


def test_role_labels_do_not_satisfy_human_dri_gate():
    document = _document()
    document["assignments"] = {
        key: "Response Engineer" for key in document["assignments"]
    }
    evidence_ledger.validate_ledger(document)
    with pytest.raises(evidence_ledger.LedgerValidationError, match="names missing"):
        evidence_ledger.validate_ledger(document, require_named_dris=True)


def test_named_people_can_close_only_the_identity_check_not_a_gpu_gate():
    document = _document()
    document["assignments"] = {
        key: f"Person {index}"
        for index, key in enumerate(document["assignments"], start=1)
    }
    evidence_ledger.validate_ledger(document, require_named_dris=True)


def test_exact_schema_rejects_unreviewed_extension():
    document = copy.deepcopy(_document())
    document["narrative_override"] = True
    with pytest.raises(evidence_ledger.LedgerValidationError, match="schema mismatch"):
        evidence_ledger.validate_ledger(document)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_utilization_must_be_finite_probability(value):
    document = _document()
    document["models"][0]["post_reserve_window_utilization"] = value
    with pytest.raises(evidence_ledger.LedgerValidationError):
        evidence_ledger.validate_ledger(document)


def test_scrub_pass_and_bundle_hash_cannot_disagree():
    document = _document()
    document["models"][0]["public_bundle_scrub_pass"] = True
    with pytest.raises(evidence_ledger.LedgerValidationError, match="must agree"):
        evidence_ledger.validate_ledger(document)


def test_release_cannot_be_frozen_by_deleting_blocker_prose():
    document = _document()
    document["release_blockers"] = []
    with pytest.raises(evidence_ledger.LedgerValidationError, match="cannot be empty"):
        evidence_ledger.validate_ledger(document)

    document = _document()
    document["status"] = "release_frozen"
    with pytest.raises(evidence_ledger.LedgerValidationError, match="release_frozen"):
        evidence_ledger.validate_ledger(document)


def test_intermediate_readiness_claims_require_bound_evidence():
    document = _document()
    document["models"][0]["mechanics_ready"] = True
    with pytest.raises(evidence_ledger.LedgerValidationError, match="mechanics_ready"):
        evidence_ledger.validate_ledger(document)

    document = _document()
    model = document["models"][0]
    model["mechanics_evidence_pass"] = True
    model["evidence_manifest_sha256"] = "a" * 64
    model["mechanics_ready"] = True
    model["quality_evidenced"] = True
    with pytest.raises(
        evidence_ledger.LedgerValidationError, match="quality_evidenced"
    ):
        evidence_ledger.validate_ledger(document)


def test_learning_entry_waiver_records_risk_but_does_not_turn_gates_green():
    document = _document()
    document["status"] = "learning_entry_with_waiver"
    for model in document["models"][1:]:
        model["in_scope_for_entry"] = False
    document["models"][0]["waiver"] = {
        "approved_by": "Alice Owner",
        "approved_at_utc": "2026-07-28T00:00:00Z",
        "consequence": "Krea may fail Round 1 because field parity is still red.",
        "evidence_sha256": "b" * 64,
    }
    evidence_ledger.validate_ledger(document)
    assert document["models"][0]["win_ready"] is False


def test_timestamps_and_arm_ids_are_canonical():
    document = _document()
    document["updated_at_utc"] = "yesterday"
    with pytest.raises(evidence_ledger.LedgerValidationError, match="RFC3339"):
        evidence_ledger.validate_ledger(document)

    document = _document()
    model = document["models"][0]
    model["public_arms_reproduced"] = ["K2", "K2"]
    with pytest.raises(evidence_ledger.LedgerValidationError, match="duplicates"):
        evidence_ledger.validate_ledger(document)
