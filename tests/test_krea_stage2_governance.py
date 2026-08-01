"""Adversarial coverage for the additive Krea Stage-2 governance slice."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from ops.calibration import krea_confirmation_admission as admission
from ops.calibration import krea_stage2_delegated_review_contract as contract
from ops.calibration import krea_stage2_execution_surface_policy as surface
from ops.calibration import krea_stage2_production_identity as production


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _time(base: datetime, seconds: int) -> str:
    return (base + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _production_identity(captured_at_utc: str) -> dict:
    return production.build(
        forge={
            "commit_sha1": "a" * 40,
            "tree_sha1": "b" * 40,
            "worktree_state": "clean-including-untracked",
        },
        container_image={
            "image_id": "sha256:" + "c" * 64,
            "repo_digest": "registry.example/forge@sha256:" + "d" * 64,
        },
        dockerfile={
            "path": production.DOCKERFILE_PATH,
            "sha256": _sha("Dockerfile"),
            "bytes": 123,
        },
        runtime_inputs=[
            {"path": path, "sha256": _sha(path), "bytes": index + 1}
            for index, path in enumerate(production.RUNTIME_INPUT_PATHS)
        ],
        base_model={
            "model_id": production.KREA_MODEL_ID,
            "revision": "e" * 40,
            "training_identity_sha256": _sha("training-assets"),
            "asset_attestation_sha256": _sha("asset-attestation"),
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
        captured_at_utc=captured_at_utc,
    )


def _governance_chain(tmp_path: Path) -> dict:
    base = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=20)
    identity = _production_identity(_time(base, 0))
    identity_file_sha = hashlib.sha256(
        production.canonical_bytes(identity) + b"\n"
    ).hexdigest()
    sealed_root = tmp_path / "sealed"
    rows = []
    for role in sorted(admission._ALL_ROLES):
        relative = f"{role}/payload.bin"
        payload = f"sealed test payload for {role}\n".encode("utf-8")
        path = sealed_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        rows.append(
            {
                "role": role,
                "relative_path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    request = admission.build_request(
        production_identity=identity,
        production_identity_file_sha256=identity_file_sha,
        waiver_freeze_sha256=_sha("waiver-freeze-semantic"),
        waiver_freeze_file_sha256=_sha("waiver-freeze-file"),
        public_commitment_sha256s={
            role: _sha(f"public-{role}") for role in admission._CONFIRMATION_ROLES
        },
        boundary_fixture_manifest_sha256s={
            role: _sha(f"manifest-{role}") for role in surface.BOUNDARY_ROLES
        },
        sealed_inventory_sha256=_sha("sealed-inventory"),
        sealed_inventory_file_sha256=_sha("sealed-inventory-file"),
        sealed_root_locator_sha256=admission.sealed_root_locator_sha256(sealed_root),
        sealed_files=rows,
        prepared_at_utc=_time(base, 1),
    )
    ratification = admission.ratify(
        request,
        production_identity=identity,
        production_identity_file_sha256=identity_file_sha,
        sealed_root=sealed_root,
        owner_identity=admission.OWNER_IDENTITY,
        ratified_at_utc=_time(base, 2),
    )
    ratification_file_sha = admission.canonical_file_sha256(ratification)
    reveal = admission.authorize_reveal(
        request,
        ratification,
        ratification_file_sha256=ratification_file_sha,
        production_identity=identity,
        production_identity_file_sha256=identity_file_sha,
        sealed_root=sealed_root,
        actor=contract.actor("confirmation_reveal_reviewer"),
        revealed_at_utc=_time(base, 3),
    )
    return {
        "base": base,
        "identity": identity,
        "identity_file_sha": identity_file_sha,
        "sealed_root": sealed_root,
        "request": request,
        "ratification": ratification,
        "ratification_file_sha": ratification_file_sha,
        "reveal": reveal,
        "reveal_file_sha": admission.canonical_file_sha256(reveal),
    }


def test_stage2_policy_and_delegated_contract_are_exact_and_fail_closed() -> None:
    assert surface.validate(surface.POLICY)["execution_surface"] == (
        "immutable_production_docker_image"
    )
    assert contract.load()["contract_sha256"] == contract.CONTRACT_SHA256
    assert contract.binding()["file_sha256"] == contract.CONTRACT_FILE_SHA256
    actors = contract.load()["actors"].values()
    assert len({actor["actor_id"] for actor in actors}) == 3

    drifted = deepcopy(surface.POLICY)
    drifted["execution_surface"] = "host"
    with pytest.raises(ValueError, match="policy drifted"):
        surface.validate(drifted)


def test_production_identity_binds_every_exact_input_and_publishes_create_only(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    identity = _production_identity(_time(now, -1))
    output = tmp_path / "production-identity.json"

    published = production.publish(identity, output)

    assert (
        published["production_identity_sha256"]
        == identity["production_identity_sha256"]
    )
    assert published["image_id"] == "sha256:" + "c" * 64
    with pytest.raises(FileExistsError):
        production.publish(identity, output)
    for mutation in (
        lambda value: value["forge"].update(worktree_state="dirty"),
        lambda value: value["forge"].update(tree_sha1="0" * 40),
        lambda value: value["container_image"].update(image_id="latest"),
        lambda value: value["dockerfile"].update(path="Dockerfile"),
        lambda value: value["runtime_inputs"].pop(),
    ):
        drifted = deepcopy(identity)
        mutation(drifted)
        with pytest.raises(ValueError):
            production.validate(drifted)


def _staged_assets(tmp_path: Path) -> tuple[Path, Path, Path]:
    base = tmp_path / "assets" / "krea"
    encoder = tmp_path / "assets" / "qwen"
    (base / "weights").mkdir(parents=True)
    (encoder / "tokenizer").mkdir(parents=True)
    (base / "config.json").write_text('{"model":"krea"}\n', encoding="utf-8")
    (base / "weights" / "model.safetensors").write_bytes(b"K" * (3 * 1024 * 1024 + 7))
    (encoder / "config.json").write_text('{"model":"qwen"}\n', encoding="utf-8")
    (encoder / "tokenizer" / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    staging = tmp_path / "krea-stage.json"
    staging.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "local_dir": "/cache/models/krea--Krea-2-Raw",
                        "repo_id": production.KREA_MODEL_ID,
                        "resolved_path": "/cache/models/krea--Krea-2-Raw",
                        "revision": "1" * 40,
                    },
                    {
                        "local_dir": "/cache/hf_cache/Qwen--Qwen3-VL-4B-Instruct",
                        "repo_id": production.KREA_TEXT_ENCODER_ID,
                        "resolved_path": "/cache/hf_cache/Qwen--Qwen3-VL-4B-Instruct",
                        "revision": "2" * 40,
                    },
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return base, encoder, staging


def test_asset_attestation_binds_staging_bytes_membership_and_live_samples(
    tmp_path: Path,
) -> None:
    base, encoder, staging = _staged_assets(tmp_path)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    record = production.capture_asset_attestation(
        base_model_path=base,
        text_encoder_path=encoder,
        staging_manifest_path=staging,
        captured_at_utc=_time(now, -1),
    )
    assert (
        production.verify_live_asset_attestation(
            record,
            base_model_path=base,
            text_encoder_path=encoder,
        )
        == record
    )
    assert record["base_model"]["revision"] == "1" * 40
    assert record["text_encoder"]["revision"] == "2" * 40
    output = tmp_path / "asset-attestation.json"
    binding = production.publish_asset_attestation(record, output)
    assert binding["attestation_sha256"] == record["attestation_sha256"]
    assert production.load_asset_attestation(output) == record

    (base / "unexpected-empty-directory").mkdir()
    with pytest.raises(ValueError, match="directory membership"):
        production.verify_live_asset_attestation(
            record,
            base_model_path=base,
            text_encoder_path=encoder,
        )


def test_asset_attestation_rejects_content_staging_and_symlink_drift(
    tmp_path: Path,
) -> None:
    base, encoder, staging = _staged_assets(tmp_path)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    record = production.capture_asset_attestation(
        base_model_path=base,
        text_encoder_path=encoder,
        staging_manifest_path=staging,
        captured_at_utc=_time(now, -1),
    )
    (encoder / "config.json").write_text('{"model":"wrong"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bytes/stat"):
        production.verify_live_asset_attestation(
            record,
            base_model_path=base,
            text_encoder_path=encoder,
        )

    base2, encoder2, staging2 = _staged_assets(tmp_path / "symlink-case")
    (base2 / "alias.json").symlink_to(base2 / "config.json")
    with pytest.raises(ValueError, match="symlink"):
        production.capture_asset_attestation(
            base_model_path=base2,
            text_encoder_path=encoder2,
            staging_manifest_path=staging2,
            captured_at_utc=_time(now, -1),
        )


def test_capture_rejects_dirty_worktree_and_binds_real_commit_tree(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    for relative in (production.DOCKERFILE_PATH, *production.RUNTIME_INPUT_PATHS):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"input {relative}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Stage2 Test",
            "-c",
            "user.email=stage2@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    now = datetime.now(timezone.utc).replace(microsecond=0)

    identity = production.capture(
        repository,
        image_id="sha256:" + "1" * 64,
        repo_digest="registry.example/forge@sha256:" + "2" * 64,
        runtime_input_paths=production.RUNTIME_INPUT_PATHS,
        base_model={
            "model_id": production.KREA_MODEL_ID,
            "revision": "e" * 40,
            "training_identity_sha256": _sha("training-assets"),
            "asset_attestation_sha256": _sha("asset-attestation"),
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
        captured_at_utc=_time(now, -1),
    )

    expected_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_tree = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert identity["forge"] == {
        "commit_sha1": expected_commit,
        "tree_sha1": expected_tree,
        "worktree_state": "clean-including-untracked",
    }

    (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        production.capture(
            repository,
            image_id="sha256:" + "1" * 64,
            repo_digest="registry.example/forge@sha256:" + "2" * 64,
            runtime_input_paths=production.RUNTIME_INPUT_PATHS,
            base_model={
                "model_id": production.KREA_MODEL_ID,
                "revision": "e" * 40,
                "training_identity_sha256": _sha("training-assets"),
                "asset_attestation_sha256": _sha("asset-attestation"),
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
            captured_at_utc=_time(now, -1),
        )


def test_ratification_validates_public_governance_before_root_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = _governance_chain(tmp_path)
    drifted = deepcopy(chain["request"])
    drifted["policy_sha256"] = "0" * 64
    body = {key: value for key, value in drifted.items() if key != "request_sha256"}
    drifted["request_sha256"] = admission.canonical_sha256(body)
    monkeypatch.setattr(
        admission,
        "_resolve_sealed_root",
        lambda _value: pytest.fail("invalid public evidence resolved sealed root"),
    )

    with pytest.raises(ValueError, match="request drifted"):
        admission.ratify(
            drifted,
            production_identity=chain["identity"],
            production_identity_file_sha256=chain["identity_file_sha"],
            sealed_root=chain["sealed_root"],
            owner_identity=admission.OWNER_IDENTITY,
            ratified_at_utc=_time(chain["base"], 4),
        )


def test_reveal_validates_actor_before_root_resolution_and_never_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = _governance_chain(tmp_path)
    bad_actor = contract.actor("confirmation_reveal_reviewer")
    bad_actor["actor_id"] += "-impostor"
    monkeypatch.setattr(
        admission,
        "_resolve_sealed_root",
        lambda _value: pytest.fail("invalid reveal actor resolved sealed root"),
    )
    monkeypatch.setattr(
        admission,
        "_read_sealed_file",
        lambda *_args: pytest.fail("reveal read sealed content"),
    )

    with pytest.raises(ValueError, match="owner-ratified delegated actor"):
        admission.authorize_reveal(
            chain["request"],
            chain["ratification"],
            ratification_file_sha256=chain["ratification_file_sha"],
            production_identity=chain["identity"],
            production_identity_file_sha256=chain["identity_file_sha"],
            sealed_root=chain["sealed_root"],
            actor=bad_actor,
            revealed_at_utc=_time(chain["base"], 4),
        )


def test_only_materialization_reads_committed_sealed_files(tmp_path: Path) -> None:
    chain = _governance_chain(tmp_path)
    output = tmp_path / "admitted"

    record = admission.materialize(
        chain["request"],
        chain["ratification"],
        chain["reveal"],
        request_file_sha256=admission.canonical_file_sha256(chain["request"]),
        ratification_file_sha256=chain["ratification_file_sha"],
        reveal_file_sha256=chain["reveal_file_sha"],
        production_identity=chain["identity"],
        production_identity_file_sha256=chain["identity_file_sha"],
        sealed_root=chain["sealed_root"],
        output_dir=output,
        actor=contract.actor("confirmation_materialization_reviewer"),
        materialized_at_utc=_time(chain["base"], 4),
    )

    assert record["admission_authorized"] is True
    assert record["gpu_execution_authorized"] is False
    assert record["image_id"] == chain["identity"]["container_image"]["image_id"]
    assert record["ratification_sha256"] == chain["ratification"]["ratification_sha256"]
    assert (output / "materialization.json").is_file()
    for row in chain["request"]["sealed_files"]:
        assert (
            hashlib.sha256((output / row["relative_path"]).read_bytes()).hexdigest()
            == row["sha256"]
        )

    materialization_file_sha = admission.canonical_file_sha256(record)
    authorization = admission.build_gpu_execution_authorization(
        chain["request"],
        chain["ratification"],
        chain["reveal"],
        record,
        request_file_sha256=admission.canonical_file_sha256(chain["request"]),
        ratification_file_sha256=chain["ratification_file_sha"],
        reveal_file_sha256=chain["reveal_file_sha"],
        materialization_file_sha256=materialization_file_sha,
        production_identity=chain["identity"],
        production_identity_file_sha256=chain["identity_file_sha"],
        owner_identity=admission.OWNER_IDENTITY,
        authorized_at_utc=_time(chain["base"], 5),
    )
    assert authorization["gpu_execution_authorized"] is True
    assert authorization["production_mutation_authorized"] is False
    assert authorization["release_authorized"] is False
    assert authorization["materialization_sha256"] == record["materialization_sha256"]
    assert (
        admission.validate_gpu_execution_authorization(
            authorization,
            request=chain["request"],
            ratification=chain["ratification"],
            reveal=chain["reveal"],
            materialization=record,
            request_file_sha256=admission.canonical_file_sha256(chain["request"]),
            ratification_file_sha256=chain["ratification_file_sha"],
            reveal_file_sha256=chain["reveal_file_sha"],
            materialization_file_sha256=materialization_file_sha,
            production_identity=chain["identity"],
            production_identity_file_sha256=chain["identity_file_sha"],
        )
        == authorization
    )

    with pytest.raises(ValueError, match="only the named owner"):
        admission.build_gpu_execution_authorization(
            chain["request"],
            chain["ratification"],
            chain["reveal"],
            record,
            request_file_sha256=admission.canonical_file_sha256(chain["request"]),
            ratification_file_sha256=chain["ratification_file_sha"],
            reveal_file_sha256=chain["reveal_file_sha"],
            materialization_file_sha256=materialization_file_sha,
            production_identity=chain["identity"],
            production_identity_file_sha256=chain["identity_file_sha"],
            owner_identity=contract.actor("confirmation_materialization_reviewer")[
                "display_name"
            ],
            authorized_at_utc=_time(chain["base"], 5),
        )


def test_materialization_validates_reveal_before_root_resolution_or_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = _governance_chain(tmp_path)
    drifted = deepcopy(chain["reveal"])
    drifted["image_id"] = "sha256:" + "f" * 64
    body = {key: value for key, value in drifted.items() if key != "reveal_sha256"}
    drifted["reveal_sha256"] = admission.canonical_sha256(body)
    monkeypatch.setattr(
        admission,
        "_resolve_sealed_root",
        lambda _value: pytest.fail("invalid reveal resolved sealed root"),
    )
    monkeypatch.setattr(
        admission,
        "_read_sealed_file",
        lambda *_args: pytest.fail("invalid reveal read sealed content"),
    )

    with pytest.raises(ValueError, match="reveal authorization drifted"):
        admission.materialize(
            chain["request"],
            chain["ratification"],
            drifted,
            request_file_sha256=admission.canonical_file_sha256(chain["request"]),
            ratification_file_sha256=chain["ratification_file_sha"],
            reveal_file_sha256=admission.canonical_file_sha256(drifted),
            production_identity=chain["identity"],
            production_identity_file_sha256=chain["identity_file_sha"],
            sealed_root=chain["sealed_root"],
            output_dir=tmp_path / "must-not-exist",
            actor=contract.actor("confirmation_materialization_reviewer"),
            materialized_at_utc=_time(chain["base"], 4),
        )


def test_admission_public_record_publish_is_canonical_and_create_only(
    tmp_path: Path,
) -> None:
    chain = _governance_chain(tmp_path)
    output = tmp_path / "request.json"

    published = admission.publish(chain["request"], output)

    assert published["request_sha256"] == chain["request"]["request_sha256"]
    assert published["file_sha256"] == admission.canonical_file_sha256(chain["request"])
    assert admission.load(output) == chain["request"]
    with pytest.raises(FileExistsError):
        admission.publish(chain["request"], output)

    overclaim = deepcopy(chain["request"])
    overclaim["admission_authorized"] = True
    body = {key: value for key, value in overclaim.items() if key != "request_sha256"}
    overclaim["request_sha256"] = admission.canonical_sha256(body)
    with pytest.raises(ValueError, match="request drifted"):
        admission.publish(overclaim, tmp_path / "overclaim.json")
    assert not (tmp_path / "overclaim.json").exists()
