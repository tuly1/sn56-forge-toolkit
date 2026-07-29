"""Fail-closed tests for executable D1/D2 review and source selection."""

from __future__ import annotations

import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_CALIBRATION = Path(__file__).parents[1] / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))

import krea_provenance as provenance  # noqa: E402
import krea_review_split as split  # noqa: E402


def _group(role: str, index: int, creator: str) -> dict[str, str]:
    source_id = f"{role.lower()}-{index:03d}"
    return {
        "source_id": source_id,
        "creator_id": creator,
        "burst_id": f"burst-{source_id}",
        "scene_id": f"scene-{source_id}",
        "play_root_id": f"play-{source_id}",
        "play_component_id": f"component-{source_id}",
        "accession_family_id": f"accession-{source_id}",
        "human_similarity_cluster_id": f"human-{source_id}",
    }


def _review(monkeypatch: pytest.MonkeyPatch | None = None) -> dict:
    policy, policy_sha, amendment, amendment_sha = split._policy()
    d1 = []
    for index in range(84):
        candidate = index < 68
        creator_index = index % 30 if index < 60 else index - 60
        creator = f"creator-{creator_index:02d}"
        group = _group("D1", index, creator)
        if index == 30:
            group["scene_id"] = "scene-d1-000"
            group["human_similarity_cluster_id"] = "human-d1-000"
        d1.append(
            {
                "source_id": f"d1-{index:03d}",
                "disposition": (
                    "CANDIDATE_ONLY_NOT_ADMITTED" if candidate else "EXCLUDE"
                ),
                "group_identity": group,
                "width": 1200 if index % 3 else 800,
                "height": 800 if index % 3 else 1200,
                "quality_grade": ("A", "B", "C")[index % 3],
                "normalized_factual_caption_sha256": provenance.canonical_sha256(
                    ["d1-caption", index]
                ),
            }
        )
    d2 = []
    for index in range(222):
        group = _group("D2", index, f"not-applicable-{index}")
        if index == 1:
            group["play_component_id"] = "component-d2-000"
        d2.append(
            {
                "source_id": f"d2-{index:03d}",
                "disposition": (
                    "CANDIDATE_ONLY_NOT_ADMITTED" if index < 221 else "EXCLUDE"
                ),
                "group_identity": group,
                "quality_grade": ("A", "B", "C")[index % 3],
            }
        )
    body = {
        "schema": 1,
        "kind": split._REVIEW_KIND,
        "selection_policy_sha256": policy_sha,
        "selection_amendment_sha256": amendment_sha,
        "tool_identity": {
            "algorithm": "xlsx-review-export-v1",
            "source_sha256": split._file_sha256(Path(split.__file__).resolve()),
        },
        "workbook": {"sha256": "0" * 64},
        "owner_signoff": {"summary_sha256": "1" * 64},
        "source_evidence": {"D1": {}, "D2": {}},
        "records": {"D1": d1, "D2": d2},
        "queued_pair_reviews": {"D1": [], "D2": []},
        "counts": {
            "D1": {
                "reviewed": 84,
                "candidates": 68,
                "excluded": 16,
                "queued_pairs_reviewed": 0,
            },
            "D2": {
                "reviewed": 222,
                "candidates": 221,
                "excluded": 1,
                "queued_pairs_reviewed": 0,
            },
        },
        "selection_state": "review_validated_split_pending",
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": split._CLAIM_LIMIT,
    }
    review = {**body, "review_sha256": provenance.canonical_sha256(body)}
    if monkeypatch is not None:
        test_amendment = deepcopy(amendment)
        payload_fields = test_amendment["executable_review_binding"]["payload_fields"]
        test_amendment["executable_review_binding"]["payload_sha256"] = (
            provenance.canonical_sha256(
                {field: review[field] for field in payload_fields}
            )
        )
        monkeypatch.setattr(
            split,
            "_policy",
            lambda: (policy, policy_sha, test_amendment, amendment_sha),
        )
    return review


def test_d1_reference_selector_is_exact_deterministic_and_leak_free(monkeypatch):
    review = _review(monkeypatch)
    first = split.select_d1(review)
    second = split.select_d1(review)

    assert first == second
    assert len(first["training_source_ids"]) == 18
    assert len(first["evaluation_source_ids"]) == 24
    assert len(first["unused_accepted_reserve_source_ids"]) >= 8
    assert not {"d1-000", "d1-030"}.issubset(
        set(first["training_source_ids"] + first["evaluation_source_ids"])
    )
    assert first["objective"]["distinct_selected_creators"] == 30
    assert first["objective"]["maximum_selected_rows_per_creator"] <= 3


def test_d1_dp_preserves_quality_when_a_later_creator_equalizes_maximum():
    """A lower prefix maximum must not erase a better eventual optimum."""

    _, policy_sha, _, _ = split._policy()
    by_creator = {}
    grades = {
        "creator-a": ("A", "A", "A"),
        "creator-b": ("A", "A"),
        "creator-c": ("A", "C"),
        "creator-d": ("A", "A", "A"),
    }
    for creator, creator_grades in grades.items():
        rows = []
        for index, grade in enumerate(creator_grades):
            source_id = f"{creator}-{index}"
            rows.append(
                {
                    "source_id": source_id,
                    "width": 1200,
                    "height": 800,
                    "quality_grade": grade,
                    "normalized_factual_caption_sha256": provenance.canonical_sha256(
                        ["counterexample", source_id]
                    ),
                    "group_identity": {
                        "scene_id": f"scene-{source_id}",
                        "human_similarity_cluster_id": f"human-{source_id}",
                    },
                }
            )
        by_creator[creator] = rows

    objective, state, value, training, evaluation = split._d1_dynamic_program(
        by_creator,
        policy_sha=policy_sha,
        duplicate_bits={},
        training_target=9,
        evaluation_target=0,
    )

    assert objective[:5] == (-4, 3, 0, -9, 0)
    assert state[-1] == 3
    assert value[1] == 9
    assert "creator-a-2" in training
    assert "creator-c-1" not in training
    assert evaluation == ()


def test_d1_validator_rejects_a_self_rehashed_non_reference_plan(monkeypatch):
    review = _review(monkeypatch)
    plan = split.select_d1(review)
    plan["training_source_ids"][0], plan["evaluation_source_ids"][0] = (
        plan["evaluation_source_ids"][0],
        plan["training_source_ids"][0],
    )
    body = {key: value for key, value in plan.items() if key != "split_sha256"}
    plan["split_sha256"] = provenance.canonical_sha256(body)

    with pytest.raises(ValueError, match="reference selector"):
        split.validate_d1_split(plan, review)


def test_d2_commitment_and_hmac_selection_are_deterministic(monkeypatch):
    review = _review(monkeypatch)
    secret = bytes(range(32))
    committed_at = (datetime.now(timezone.utc) - timedelta(seconds=5)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    commitment = split.build_d2_commitment(
        review,
        reviewer_identity="Independent Reviewer",
        committed_at_utc=committed_at,
        secret=secret,
    )
    first = split.select_d2(review, commitment, secret=secret)
    second = split.select_d2(review, commitment, secret=secret)

    assert first == second
    assert len(first["training_source_ids"]) == 36
    assert len(first["evaluation_source_ids"]) == 40
    assert not {"d2-000", "d2-001"}.issubset(
        set(first["training_source_ids"] + first["evaluation_source_ids"])
    )
    with pytest.raises(ValueError, match="open the commitment"):
        split.select_d2(review, commitment, secret=b"x" * 32)


def test_review_and_formula_boundaries_fail_closed(monkeypatch):
    review = _review(monkeypatch)
    tampered = deepcopy(review)
    tampered["records"]["D1"][0]["disposition"] = "EXCLUDE"
    with pytest.raises(ValueError, match="digest"):
        split.validate_review(tampered)

    missing = deepcopy(review)
    del missing["workbook"]
    body = {key: value for key, value in missing.items() if key != "review_sha256"}
    missing["review_sha256"] = provenance.canonical_sha256(body)
    with pytest.raises(ValueError, match="keys mismatch"):
        split.validate_review(missing)

    fabricated = deepcopy(review)
    fabricated["records"]["D1"][0]["quality_grade"] = "C"
    body = {key: value for key, value in fabricated.items() if key != "review_sha256"}
    fabricated["review_sha256"] = provenance.canonical_sha256(body)
    with pytest.raises(ValueError, match="boundary or digest"):
        split.validate_review(fabricated)

    rows = [
        {"__row__": "1", "A": "id", "B": "decision", "__formulas__": ""},
        {"__row__": "2", "A": "one", "B": "ok", "__formulas__": "B"},
    ]
    with pytest.raises(ValueError, match="formulas mismatch"):
        split._table(
            rows,
            header_row=1,
            headers=["id", "decision"],
            expected_count=1,
            label="test",
            formula_columns=set(),
        )
    with pytest.raises(ValueError, match="unsafe XLSX"):
        split._zip_member("../sheet.xml")


def test_policy_and_amendment_are_canonical_and_non_authorizing():
    policy, policy_sha, amendment, amendment_sha = split._policy()
    assert (
        policy_sha == "2d6ea8d8065935c974dd891e4c214bcc4427d5a3ad9b1ddc67d36daaecd65701"
    )
    assert amendment["base_policy_sha256"] == policy_sha
    assert amendment["gpu_execution_authorized"] is False
    assert len(amendment_sha) == 64
    assert policy["admission_authorized"] is False
