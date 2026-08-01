from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from ops.calibration import krea_provenance
from ops.calibration import krea_stage2_endgame_matrix as matrix
from ops.calibration import krea_stage2_production_identity as production


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _identity(*, image_id: str = matrix.PRODUCTION_IMAGE_ID) -> dict:
    return production.build(
        forge={
            "commit_sha1": matrix.SOURCE_COMMIT,
            "tree_sha1": matrix.SOURCE_TREE,
            "worktree_state": "clean-including-untracked",
        },
        container_image={
            "image_id": image_id,
            "repo_digest": "registry.invalid/forge@sha256:" + "d" * 64,
        },
        dockerfile={
            "path": production.DOCKERFILE_PATH,
            "sha256": _sha("dockerfile"),
            "bytes": 100,
        },
        runtime_inputs=[
            {"path": path, "sha256": _sha(path), "bytes": index + 1}
            for index, path in enumerate(production.RUNTIME_INPUT_PATHS)
        ],
        base_model={
            "model_id": production.KREA_MODEL_ID,
            "revision": "e" * 40,
            "training_identity_sha256": _sha("training"),
            "asset_attestation_sha256": _sha("assets"),
            "text_encoder_id": production.KREA_TEXT_ENCODER_ID,
            "text_encoder_revision": "f" * 40,
        },
        runtime_contract={
            "runtime_identity_sha256": _sha("runtime"),
            "venv_tree_manifest_sha256": _sha("venv"),
            "trainer_identity_sha256": _sha("trainer"),
            "measurement_tool_sha256": _sha("measurement"),
            "jit_enabled": True,
        },
        captured_at_utc="2026-08-01T18:20:00Z",
    )


def _freeze(d1: str, d2: str) -> dict:
    body = {
        "schema": 2,
        "kind": "forge-krea-density-seedb-finalist-freeze",
        "D1_winner_family_id": d1,
        "D2_winner_family_id": d2,
        # Deliberate distractions: matrix policy must not come from these.
        "finalist_family_ids": ["K4", "K3", "K0"],
        "workload": {"family_universe": ["K4"]},
    }
    return {**body, "freeze_sha256": krea_provenance.canonical_sha256(body)}


def _matrix(d1: str = "K1", d2: str = "K5") -> tuple[dict, dict, dict]:
    freeze = _freeze(d1, d2)
    identity = _identity()
    body = matrix._matrix_body(
        freeze=freeze,
        freeze_file_sha256=matrix._canonical_file_sha(freeze),
        production_identity=identity,
        production_identity_file_sha256=matrix._canonical_file_sha(identity),
        created_at_utc="2026-08-01T18:21:00Z",
    )
    value = {**body, "matrix_sha256": krea_provenance.canonical_sha256(body)}
    resolved = matrix.validate_matrix(
        value, freeze=freeze, production_identity=identity
    )
    return resolved, freeze, identity


def test_distinct_winners_produce_exact_48_plus_12_matrix() -> None:
    value, _freeze_record, _identity_record = _matrix("K1", "K5")

    assert value["active_variant_family_ids"] == ["K1", "K5"]
    assert value["family_execution_universe"] == list(matrix.FAMILY_ORDER)
    assert value["confirmation_training_count"] == 48
    assert value["boundary_training_count"] == 12
    assert value["training_count"] == 60
    assert [len(wave["row_keys"]) for wave in value["waves"]] == [40, 8, 6, 6]
    assert len({row["row_key"] for row in value["rows"]}) == 60

    shared = [
        row
        for row in value["rows"]
        if row["phase"] == "confirmation" and row["family_id"] == "K2"
    ]
    assert len(shared) == 8
    assert all(row["candidate_policy_family_ids"] == ["K1", "K5"] for row in shared)
    assert all(row["family_role"] == "public_reference" for row in shared)


@pytest.mark.parametrize(
    ("winner", "confirmation", "boundary", "wave_sizes"),
    [
        ("K1", 40, 6, [40, 6]),
        ("K2", 32, 6, [32, 6]),
    ],
)
def test_same_winner_is_one_policy_and_one_decision(
    winner: str, confirmation: int, boundary: int, wave_sizes: list[int]
) -> None:
    value, _freeze_record, _identity_record = _matrix(winner, winner)

    assert value["active_variant_family_ids"] == [winner]
    assert value["confirmation_training_count"] == confirmation
    assert value["boundary_training_count"] == boundary
    assert [len(wave["row_keys"]) for wave in value["waves"]] == wave_sizes


def test_fixed_fixture_gpu_mapping_and_wave_deduplication() -> None:
    value, _freeze_record, _identity_record = _matrix()
    for row in value["rows"]:
        expected = (
            matrix.FIXTURE_GPU[row["fixture_id"]]
            if row["phase"] == "confirmation"
            else matrix.BOUNDARY_GPU[row["cell_id"]]
        )
        assert row["gpu_device"] == expected
    first_wave = value["waves"][0]["row_keys"]
    assert not any(key.endswith("-K5") for key in first_wave)
    second_wave = value["waves"][1]["row_keys"]
    assert len(second_wave) == 8
    assert all(key.endswith("-K5") for key in second_wave)


def test_matrix_recomputes_from_winners_and_rejects_posthoc_policy_edit() -> None:
    value, freeze, identity = _matrix()
    drifted = deepcopy(value)
    drifted["active_variant_family_ids"] = ["K2"]
    body = {key: item for key, item in drifted.items() if key != "matrix_sha256"}
    drifted["matrix_sha256"] = krea_provenance.canonical_sha256(body)

    with pytest.raises(ValueError, match="does not recompute"):
        matrix.validate_matrix(drifted, freeze=freeze, production_identity=identity)


def test_matrix_hard_rejects_noncertified_image() -> None:
    wrong = _identity(image_id="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="fresh c000"):
        matrix._validate_production_identity(wrong)


def test_matrix_publication_is_create_only(tmp_path: Path) -> None:
    value, _freeze_record, _identity_record = _matrix()
    output = tmp_path / "matrix.json"
    assert matrix.publish_matrix(value, output) == value
    with pytest.raises(FileExistsError):
        matrix.publish_matrix(value, output)


def _row_controls(
    value: dict, identity: dict, freeze: dict
) -> tuple[str, dict, dict, dict, dict]:
    row = value["rows"][0]
    family = row["family_id"]
    plan = {
        "phase": row["phase"],
        "cell_id": row["cell_id"],
        "fixture_id": row["fixture_id"],
        "seed_role": row["seed_role"],
        "family_role": row["family_role"],
        "candidate_universe": [
            {
                "candidate_id": "selected",
                "family_id": family,
                "sha256": _sha("candidate"),
                "bytes": 1,
                "step": 1,
                "zero_control": False,
            }
        ],
        "training_candidate_id": "selected",
        "calibration_profile": family,
        "waiver_finalist_freeze": value["freeze"],
        "production_identity": value["production_identity"],
        "production_image_id": matrix.PRODUCTION_IMAGE_ID,
        "plan_sha256": _sha("plan"),
    }
    approval = {"approval_sha256": _sha("approval")}
    completion = {"completion_sha256": _sha("completion")}
    authority = {
        "waiver_finalist_freeze": freeze,
        "production_identity": identity,
        "production_identity_file_sha256": value["production_identity"][
            "file_sha256"
        ],
    }
    return row["row_key"], plan, approval, completion, authority


def test_run_row_is_fixed_gpu_create_only_and_replay_validates_before_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value, freeze, identity = _matrix()
    row_key, plan, approval, completion, authority = _row_controls(
        value, identity, freeze
    )
    monkeypatch.setattr(
        matrix.krea_stage2_execution,
        "validate_plan_with_authority",
        lambda supplied, **_kwargs: supplied,
    )
    monkeypatch.setattr(
        matrix.krea_stage2_execution,
        "validate_approval",
        lambda supplied, **_kwargs: supplied,
    )
    monkeypatch.setattr(
        matrix.krea_stage2_execution,
        "validate_completion",
        lambda supplied, **_kwargs: supplied,
    )
    observed: list[int] = []

    def fake_run_cell(**kwargs):
        observed.append(kwargs["gpu_device"])
        kwargs["completion_path"].write_bytes(
            krea_provenance.canonical_bytes(completion) + b"\n"
        )
        return completion

    completion_path = tmp_path / "completion.json"
    receipt_path = tmp_path / "receipt.json"
    receipt, replayed = matrix.run_row(
        matrix=value,
        row_key=row_key,
        plan=plan,
        approval=approval,
        authority_bundle=authority,
        output_dir=tmp_path / "run",
        completion_path=completion_path,
        receipt_path=receipt_path,
        run_cell=fake_run_cell,
    )
    assert replayed is False
    assert observed == [value["rows"][0]["gpu_device"]]
    assert receipt["strict_authority_replayed"] is True
    assert receipt["waiver_used"] is False

    replay, replayed = matrix.run_row(
        matrix=value,
        row_key=row_key,
        plan=plan,
        approval=approval,
        authority_bundle=authority,
        output_dir=tmp_path / "must-not-run",
        completion_path=completion_path,
        receipt_path=receipt_path,
        run_cell=lambda **_kwargs: pytest.fail("existing row was rerun"),
    )
    assert replayed is True
    assert replay == receipt

    receipt_path.chmod(0o644)
    receipt_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        matrix.run_row(
            matrix=value,
            row_key=row_key,
            plan=plan,
            approval=approval,
            authority_bundle=authority,
            output_dir=tmp_path / "must-not-run-2",
            completion_path=completion_path,
            receipt_path=receipt_path,
            run_cell=lambda **_kwargs: pytest.fail("drifted receipt was rerun"),
        )
