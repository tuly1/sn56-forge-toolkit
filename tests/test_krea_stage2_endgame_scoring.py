from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import zipfile

import pytest

from ops.calibration import krea_provenance
from ops.calibration import krea_stage2_endgame_matrix as matrix_module
from ops.calibration import krea_stage2_endgame_scoring as scoring
from ops.calibration import krea_stage2_legacy_confirmation as legacy


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


def _legacy_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict]:
    from PIL import Image

    role = "C1"
    root = tmp_path / role
    root.mkdir()
    listed = []
    shape = legacy.amendment.SHAPE_CONTRACT[role]
    for holdout, count in (
        (False, shape["training_pairs"]),
        (True, shape["evaluation_rows"]),
    ):
        prefix = "holdout/" if holdout else ""
        for index in range(1, count + 1):
            image_name = f"{prefix}image-{index:03d}.jpg"
            prompt_name = f"{prefix}image-{index:03d}.txt"
            image_path = root / image_name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8 + index, 9 + index), (index, 1, 2)).save(
                image_path, format="JPEG"
            )
            (root / prompt_name).write_text(f"natural C1 caption {index}\n")
            for relative in (image_name, prompt_name):
                raw = (root / relative).read_bytes()
                listed.append((hashlib.sha256(raw).hexdigest(), relative))
    checksum = root / "MANIFEST-C1.sha256"
    checksum.write_text("".join(f"{digest}  {name}\n" for digest, name in listed))
    with zipfile.ZipFile(root / "c1_tourn.zip", "w") as archive:
        for _digest, relative in listed:
            if "/" not in relative:
                archive.write(root / relative, relative)
    (root / "LICENSES.txt").write_text("test public-domain fixture\n")
    patched = dict(legacy.amendment.MANIFEST_FILE_SHA256S)
    patched[role] = hashlib.sha256(checksum.read_bytes()).hexdigest()
    monkeypatch.setattr(legacy.amendment, "MANIFEST_FILE_SHA256S", patched)
    wrapper = legacy.build_wrapper(
        role_root=root,
        role=role,
        created_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return root / legacy.WRAPPER_NAME, wrapper


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


def test_real_legacy_wrapper_dispatches_for_training_and_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper_path, wrapper = _legacy_wrapper(tmp_path, monkeypatch)

    training_view, training_binding = (
        scoring.krea_stage2_endgame_orchestrator._fixture_binding(wrapper_path, "C1")
    )
    score_view, score_binding, dataset = scoring._fixture_score_view(
        wrapper_path, wrapper_path.parent / "holdout", "C1"
    )

    assert training_view == score_view == legacy.score_view(wrapper)
    assert training_view["trigger_token"] is None
    assert training_binding["manifest_sha256"] == wrapper["wrapper_sha256"]
    assert score_binding["manifest_sha256"] == wrapper["wrapper_sha256"]
    assert training_binding["file_sha256"] == score_binding["file_sha256"]
    assert dataset == wrapper_path.parent / "holdout"


def test_receipt_result_validation_gets_full_legacy_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wrapper_path, wrapper = _legacy_wrapper(tmp_path, monkeypatch)
    candidate_path = tmp_path / "last.safetensors"
    candidate_path.write_bytes(b"candidate")
    candidate_sha = krea_provenance.file_sha256(candidate_path)
    result_path = tmp_path / "result.json"
    result_path.write_text("{}\n")
    identity = wrapper["evaluation_dataset_identity"]
    plan = {
        "plan_sha256": _sha("plan"),
        "phase": "confirmation",
        "cell_id": "C1-A",
        "fixture_id": "C1",
        "seed_role": "A",
        "candidates": [
            {
                "family_id": "K0",
                "candidate_id": "c1-a-k0",
                "candidate_sha256": candidate_sha,
            }
        ],
        "fixture_manifest": {
            "file_sha256": scoring._file_sha(wrapper),
            "manifest_sha256": wrapper["wrapper_sha256"],
        },
        "evaluation_dataset_sha256": identity["sha256"],
        "evaluation_row_count": len(identity["rows"]),
        "evaluator_contract": {},
        "evaluation_dataset_path": str(tmp_path / "holdout"),
    }
    monkeypatch.setattr(
        scoring.krea_stage2_score, "validate_plan", lambda _value: plan
    )
    monkeypatch.setattr(
        scoring.krea_stage2_score,
        "validate_receipt",
        lambda value, **_kwargs: value,
    )
    observed = {}

    def validate_result(_result, *, fixture_manifest, **_kwargs):
        # This is the production nesting: result validation resolves the
        # fixture again.  Passing legacy.score_view(wrapper) here raises the
        # exact live key-mismatch; passing the original wrapper is correct.
        observed["fixture"] = fixture_manifest
        resolved = scoring.krea_stage2_score._fixture_score_view(fixture_manifest)
        return {
            "weighted_loss": 0.1,
            "text_mean": 0.1,
            "blank_mean": 0.1,
            "row_identity_sha256": _sha("rows"),
            "evaluator_contract_sha256": _sha("contract"),
            "dataset_sha256": resolved["evaluation_dataset_identity"]["sha256"],
            "row_count": len(resolved["evaluation_dataset_identity"]["rows"]),
            "prompt_count": len(resolved["evaluation_dataset_identity"]["rows"]) * 10,
        }

    monkeypatch.setattr(
        scoring.krea_stage2_score, "_validate_result", validate_result
    )

    scoring.krea_stage2_score.build_receipt(
        plan=plan,
        family_id="K0",
        candidate_path=candidate_path,
        fixture_manifest=wrapper,
        fixture_manifest_file_sha256=scoring._file_sha(wrapper),
        result_path=result_path,
        status_file_sha256=_sha("status"),
        evidence_manifest_file_sha256=_sha("evidence"),
        completed_at_utc="2026-08-02T02:29:02Z",
    )

    assert observed["fixture"] == wrapper


def test_live_row_replay_rehashes_real_legacy_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper_path, wrapper = _legacy_wrapper(tmp_path, monkeypatch)
    binding = {
        "path": str(wrapper_path),
        "file_sha256": scoring._file_sha(wrapper),
        "manifest_sha256": wrapper["wrapper_sha256"],
    }
    receipts = {
        "config_control": {"receipt": "config"},
        "training_terminal": {"receipt": "terminal"},
        "checkpoint_selection": {"receipt": "selection"},
    }
    plan = {"fixture_manifest": binding}
    completion = {
        "gpu_device": 0,
        "config_control_receipt": receipts["config_control"],
        "training_terminal_receipt": receipts["training_terminal"],
        "checkpoint_selection_receipt": receipts["checkpoint_selection"],
        "ended_at_utc": "2026-08-01T20:00:00Z",
    }
    monkeypatch.setattr(
        matrix_module.krea_stage2_execution,
        "validate_private_run_receipts",
        lambda _plan: receipts,
    )
    captured = {}

    def fake_build_run_evidence(**kwargs):
        captured.update(kwargs)
        return {"evidence": "ok"}

    monkeypatch.setattr(
        matrix_module.krea_stage2_training_evidence,
        "build_run_evidence",
        fake_build_run_evidence,
    )

    result = matrix_module._replay_live_run(
        row={"gpu_device": 0},
        plan=plan,
        approval={},
        completion=completion,
        output_dir=tmp_path,
    )

    assert result == {"evidence": "ok"}
    assert captured["fixture_manifest"] == {
        "file_sha256": binding["file_sha256"],
        "manifest_sha256": wrapper["wrapper_sha256"],
    }


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


def test_materializer_publishes_one_group_as_soon_as_five_rows_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = _matrix()
    rows = []
    ready = {f"confirmation-C1-A-K{index}" for index in range(5)}
    for source in matrix["rows"]:
        if source["phase"] != "confirmation":
            continue
        root = tmp_path / "training" / source["row_key"]
        receipt = root / "receipt.json"
        if source["row_key"] in ready:
            root.mkdir(parents=True)
            receipt.write_text("{}\n")
        rows.append(
            {
                "row_key": source["row_key"],
                "receipt_path": str(receipt),
            }
        )
    training = {"plan_set_sha256": _sha("training"), "rows": rows}
    fixture_path = tmp_path / "C1-wrapper.json"
    fixture_path.write_text("{}\n")
    dataset = tmp_path / "holdout"
    dataset.mkdir()
    config = {
        "config_sha256": _sha("config"),
        "matrix": str(tmp_path / "matrix.json"),
        "training_plan_set": str(tmp_path / "training.json"),
        "fixture_manifests": {"C1": str(fixture_path)},
        "evaluation_datasets": {"C1": str(dataset)},
        "evaluator_contract": {"contract": True},
        "score_plan_created_at_utc": "2026-08-01T20:00:00Z",
    }
    monkeypatch.setattr(scoring, "_validate_config", lambda value: dict(value))
    monkeypatch.setattr(
        scoring,
        "_load",
        lambda path, _label: (matrix if Path(path).name == "matrix.json" else training),
    )
    monkeypatch.setattr(
        scoring.krea_stage2_endgame_matrix, "validate_matrix", lambda value: value
    )
    monkeypatch.setattr(
        scoring.krea_stage2_endgame_orchestrator,
        "validate_plan_set",
        lambda value, **_kwargs: value,
    )

    def run_control(row):
        family = row["row_key"].rsplit("-", 1)[1]
        candidate = tmp_path / f"{family}.safetensors"
        candidate.write_bytes(family.encode())
        plan = {
            "seed": 42565431,
            "hours": "0.75",
            "waiver_finalist_freeze": {
                "file_sha256": _sha("f"),
                "freeze_sha256": _sha("fs"),
            },
            "confirmation_materialization": {
                "file_sha256": _sha("m"),
                "materialization_sha256": _sha("ms"),
            },
            "owner_ratification": {
                "file_sha256": _sha("r"),
                "ratification_sha256": _sha("rs"),
            },
            "gpu_execution_authorization": {
                "file_sha256": _sha("g"),
                "gpu_execution_authorization_sha256": _sha("gs"),
            },
            "production_identity": {
                "file_sha256": _sha("p"),
                "production_identity_sha256": _sha("ps"),
            },
            "production_image_id": "sha256:" + "3" * 64,
        }
        return {
            "run_evidence_path": str(tmp_path / f"{family}-evidence.json"),
            "execution_plan": plan,
            "execution_approval": {},
            "run_completion": {},
            "candidate_path": candidate,
        }, {}

    monkeypatch.setattr(scoring, "_run_control", run_control)
    monkeypatch.setattr(
        scoring.krea_stage2_score,
        "build_candidate_row",
        lambda family_id, **_kwargs: {"family_id": family_id},
    )
    monkeypatch.setattr(
        scoring.krea_stage2_score,
        "seal_plan",
        lambda payload: {
            **payload,
            "plan_sha256": krea_provenance.canonical_sha256(payload),
        },
    )
    monkeypatch.setattr(
        scoring.krea_stage2_score,
        "validate_plan_with_run_controls",
        lambda plan, **_kwargs: plan,
    )
    identity = {"sha256": _sha("eval"), "rows": [{"image": "a.jpg"}]}
    monkeypatch.setattr(
        scoring,
        "_fixture_score_view",
        lambda *_args: (
            {"evaluation_dataset_identity": identity},
            {"file_sha256": _sha("fixture-file"), "manifest_sha256": _sha("fixture")},
            dataset,
        ),
    )

    result = scoring.materialize_ready_score_plans(
        config, output_root=tmp_path / "scores"
    )

    assert [row["group_key"] for row in result["materialized_groups"]] == [
        "score-C1-A-K1"
    ]
    assert not Path(result["queue"]["groups"][1]["group_path"]).exists()


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
