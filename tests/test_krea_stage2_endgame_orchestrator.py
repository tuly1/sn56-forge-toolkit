from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from ops.calibration import krea_provenance
from ops.calibration import krea_stage2_endgame_matrix as matrix_module
from ops.calibration import krea_stage2_endgame_orchestrator as orchestrator


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _matrix() -> dict:
    freeze = {
        "D1_winner_family_id": "K1",
        "D2_winner_family_id": "K5",
        "freeze_sha256": _sha("freeze"),
    }
    identity = {"production_identity_sha256": _sha("identity")}
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


def _plan_set(value: dict, tmp_path: Path) -> dict:
    rows = []
    for row in value["rows"]:
        root = tmp_path / "rows" / row["row_key"]
        rows.append(
            {
                "row_key": row["row_key"],
                "wave_id": row["wave_id"],
                "gpu_device": row["gpu_device"],
                "timing_key": orchestrator._timing_key(
                    row, row["family_id"] if row["family_id"] != "K0" else "K1"
                ),
                "plan": {
                    "path": str(root / "plan.json"),
                    "file_sha256": _sha("plan-file-" + row["row_key"]),
                    "plan_sha256": _sha("plan-" + row["row_key"]),
                },
                "approval": {
                    "path": str(root / "approval.json"),
                    "file_sha256": _sha("approval-file-" + row["row_key"]),
                    "approval_sha256": _sha("approval-" + row["row_key"]),
                },
                "output_dir": str(root / "run"),
                "completion_path": str(root / "completion.json"),
                "run_evidence_path": str(root / "run-evidence.json"),
                "score_hook_path": str(root / "score-hook.json"),
                "receipt_path": str(root / "row-receipt.json"),
            }
        )
    queues = {
        str(gpu): [row["row_key"] for row in rows if row["gpu_device"] == gpu]
        for gpu in orchestrator.GPU_IDS
    }
    body = {
        "schema": 1,
        "kind": orchestrator.PLAN_SET_KIND,
        "config_sha256": _sha("config"),
        "matrix_sha256": value["matrix_sha256"],
        "production_image_id": matrix_module.PRODUCTION_IMAGE_ID,
        "training_count": 60,
        "score_stream_count": 60,
        "gpu_queues": queues,
        "waves": value["waves"],
        "rows": rows,
        "strict_authority_per_row": True,
        "waiver_path_available": False,
        "release_authorized": False,
        "production_mutation_authorized": False,
    }
    return {**body, "plan_set_sha256": krea_provenance.canonical_sha256(body)}


def test_timing_program_is_exactly_ten_envelopes_by_three_classes() -> None:
    value = _matrix()
    keys = orchestrator._expected_timing_keys(value)

    assert len(keys) == 30
    assert any(key.startswith("C1__gpu0__h0p75__") for key in keys)
    assert any(key.startswith("B-0p5-large__gpu1__h0p5__") for key in keys)
    assert any(key.startswith("B-1-small__gpu0__h1p0__") for key in keys)
    with pytest.raises(ValueError, match="timing catalog is not exact"):
        orchestrator._timing_catalog({}, matrix=value)


def test_plan_set_encodes_collision_free_serial_gpu_queues(tmp_path: Path) -> None:
    value = _matrix()
    plan_set = _plan_set(value, tmp_path)

    assert orchestrator.validate_plan_set(plan_set, matrix=value) == plan_set
    flattened = [key for queue in plan_set["gpu_queues"].values() for key in queue]
    assert len(flattened) == len(set(flattened)) == 60

    drifted = deepcopy(plan_set)
    drifted["gpu_queues"]["0"][0] = drifted["gpu_queues"]["1"][0]
    body = {key: item for key, item in drifted.items() if key != "plan_set_sha256"}
    drifted["plan_set_sha256"] = krea_provenance.canonical_sha256(body)
    with pytest.raises(ValueError, match="queue is not serial"):
        orchestrator.validate_plan_set(drifted, matrix=value)


def test_scheduler_claims_one_row_per_gpu_in_first_open_wave(tmp_path: Path) -> None:
    value = _matrix()
    plan_set = _plan_set(value, tmp_path)
    claims_root = tmp_path / "claims"

    claims = orchestrator.claim_next(
        plan_set=plan_set,
        matrix=value,
        claims_root=claims_root,
        claimed_at_utc="2026-08-01T18:31:00Z",
        scheduler_instance_id="scheduler-a",
    )
    assert len(claims) == 4
    assert {claim["gpu_device"] for claim in claims} == {0, 1, 2, 3}
    assert len({claim["row_key"] for claim in claims}) == 4
    assert {claim["wave_id"] for claim in claims} == {value["waves"][0]["wave_id"]}

    # Outstanding create-only claims keep each GPU serial.
    assert (
        orchestrator.claim_next(
            plan_set=plan_set,
            matrix=value,
            claims_root=claims_root,
            claimed_at_utc="2026-08-01T18:32:00Z",
            scheduler_instance_id="scheduler-b",
        )
        == []
    )


def test_exact60_gate_never_launches_missing_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _matrix()
    plan_set = _plan_set(value, tmp_path)
    monkeypatch.setattr(
        orchestrator.krea_stage2_endgame_matrix,
        "run_row",
        lambda **_kwargs: pytest.fail("gate attempted to launch a missing row"),
    )

    with pytest.raises(ValueError, match="cannot launch work"):
        orchestrator.seal_exact60_gate(
            plan_set=plan_set,
            matrix=value,
            authority_bundle={},
            output=tmp_path / "gate.json",
            completed_at_utc="2026-08-01T19:00:00Z",
        )


def test_candidate_catalog_rehashes_live_bytes_and_requires_freeze_identity(
    tmp_path: Path,
) -> None:
    catalog = {}
    rules = {}
    for family in matrix_module.FAMILY_ORDER:
        artifact = tmp_path / f"{family}.safetensors"
        artifact.write_bytes((family + "-artifact").encode())
        digest = krea_provenance.file_sha256(artifact)
        candidate_id = family.lower() + "-candidate"
        catalog[family] = {
            "candidate_id": candidate_id,
            "step": 10,
            "path": str(artifact),
        }
        rules[family] = {
            "actual_mappings": [
                {"candidate_id": candidate_id, "candidate_sha256": digest, "step": 10}
            ]
        }
    freeze = {"all_family_checkpoint_rules": rules}

    resolved = orchestrator._candidate_catalog(catalog, freeze=freeze)
    assert set(resolved) == set(matrix_module.FAMILY_ORDER)
    assert all(row["bytes"] > 0 for row in resolved.values())

    (tmp_path / "K1.safetensors").write_bytes(b"drift")
    with pytest.raises(ValueError, match="absent from the freeze"):
        orchestrator._candidate_catalog(catalog, freeze=freeze)
