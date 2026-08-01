from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from ops.calibration import krea_provenance
from ops.calibration import krea_stage2_endgame_matrix as matrix_module
from ops.calibration import krea_stage2_endgame_scoring as scoring


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _matrix() -> dict:
    freeze = {
        "D1_winner_family_id": "K1",
        "D2_winner_family_id": "K5",
        "freeze_sha256": _sha("freeze"),
    }
    identity = {
        "production_identity_sha256": _sha("identity"),
        "forge": {"commit_sha1": "1" * 40, "tree_sha1": "2" * 40},
        "container_image": {"image_id": "sha256:" + "3" * 64},
    }
    body = matrix_module._matrix_body(
        freeze=freeze,
        freeze_file_sha256=_sha("freeze-file"),
        production_identity=identity,
        production_identity_file_sha256=_sha("identity-file"),
        created_at_utc="2026-08-01T18:21:00Z",
    )
    return matrix_module.validate_matrix(
        {**body, "matrix_sha256": krea_provenance.canonical_sha256(body)}
    )


def _queue(tmp_path: Path) -> dict:
    groups = []
    for row in scoring._groups(_matrix()):
        root = tmp_path / "groups" / row["group_key"]
        groups.append(
            {
                **row,
                "group_path": str(root / "group.json"),
                "aggregate_path": str(root / "aggregate.json"),
            }
        )
    body = {
        "schema": 1,
        "kind": scoring.QUEUE_KIND,
        "config_sha256": _sha("config"),
        "matrix_sha256": _sha("matrix"),
        "training_plan_set_sha256": _sha("training"),
        "expected_group_count": 16,
        "groups": groups,
        "streaming_materialization": True,
        "boundary_scoring_enabled": False,
        "release_authorized": False,
        "production_mutation_authorized": False,
    }
    return {**body, "score_queue_sha256": krea_provenance.canonical_sha256(body)}


def test_score_groups_are_exactly_sixteen_five_family_plans() -> None:
    groups = scoring._groups(_matrix())

    assert len(groups) == 16
    assert {row["candidate_family_id"] for row in groups} == {"K1", "K5"}
    assert all(len(row["family_ids"]) == 5 for row in groups)
    assert all(
        set(row["family_ids"]) == {"K0", "K1", "K2", "K3", "K4"}
        for row in groups
        if row["candidate_family_id"] == "K1"
    )
    assert all(
        set(row["family_ids"]) == {"K0", "K2", "K3", "K4", "K5"}
        for row in groups
        if row["candidate_family_id"] == "K5"
    )


def test_score_queue_rejects_family_drift(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    assert scoring._validate_queue(queue) == queue

    drifted = deepcopy(queue)
    drifted["groups"][0]["family_ids"] = ["K0", "K1", "K2", "K3", "K5"]
    body = {key: value for key, value in drifted.items() if key != "score_queue_sha256"}
    drifted["score_queue_sha256"] = krea_provenance.canonical_sha256(body)
    with pytest.raises(ValueError, match="family coverage"):
        scoring._validate_queue(drifted)


def test_streaming_claims_only_materialized_groups_and_keeps_gpus_serial(
    tmp_path: Path,
) -> None:
    queue = _queue(tmp_path)
    for row in queue["groups"][:6]:
        path = Path(row["group_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")

    claims = scoring.claim_ready_groups(
        score_queue=queue,
        claims_root=tmp_path / "claims",
        claimed_at_utc="2026-08-01T20:00:00Z",
        scheduler_instance_id="score-scheduler-a",
    )
    assert len(claims) == 4
    assert {row["gpu_device"] for row in claims} == {0, 1, 2, 3}
    assert {row["group_key"] for row in claims} == {
        row["group_key"] for row in queue["groups"][:4]
    }
    assert (
        scoring.claim_ready_groups(
            score_queue=queue,
            claims_root=tmp_path / "claims",
            claimed_at_utc="2026-08-01T20:01:00Z",
            scheduler_instance_id="score-scheduler-b",
        )
        == []
    )


def test_score_gate_never_launches_missing_work(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot launch work"):
        scoring.seal_score_gate(
            score_queue=_queue(tmp_path),
            output=tmp_path / "gate.json",
            completed_at_utc="2026-08-01T22:00:00Z",
        )


def test_shared_gpu_lock_fails_closed_on_overlap(tmp_path: Path) -> None:
    with scoring.krea_stage2_endgame_orchestrator.gpu_execution_lock(tmp_path, 2):
        with pytest.raises(RuntimeError, match="already executing"):
            with scoring.krea_stage2_endgame_orchestrator.gpu_execution_lock(
                tmp_path, 2
            ):
                pass


def test_shared_gpu_lock_rejects_symlink_root(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)

    with pytest.raises(ValueError, match="not a real directory"):
        with scoring.krea_stage2_endgame_orchestrator.gpu_execution_lock(alias, 0):
            pass
