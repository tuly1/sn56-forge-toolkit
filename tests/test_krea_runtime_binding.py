"""Contracts for the additive Week-5 runtime/profile binding CLI."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from ops.calibration import krea_budget
from ops.calibration import krea_provenance
from ops.calibration import krea_runtime_binding as binding


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write(path: Path, value: dict, *, canonical: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if canonical:
        path.write_bytes(krea_provenance.canonical_bytes(value) + b"\n")
    else:
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _discovery(tmp_path: Path) -> tuple[Path, tuple[str, ...]]:
    classes = ("A-class", "B-class", "C-class")
    geometry = {
        "A-class": (32, 32, "adamw8bit", "mse", 2),
        "B-class": (32, 32, "adamw8bit", "mae", 3),
        "C-class": (64, 64, "automagic", "mse", 2),
    }
    value = {
        "schema": 2,
        "kind": "sn56-week5-krea-discovery-freeze",
        "model": "krea/Krea-2-Raw",
        "model_type": "krea2",
        "gpu_execution_authorized": False,
        "arms": [
            {
                "id": f"K{index}",
                "throughput_equivalence_class": class_name,
                "rank": geometry[class_name][0],
                "alpha": geometry[class_name][1],
                "optimizer": geometry[class_name][2],
                "loss": geometry[class_name][3],
                "guidance": geometry[class_name][4],
            }
            for index, class_name in enumerate(classes)
        ],
    }
    return _write(tmp_path / "discovery.json", value, canonical=False), classes


def _fixture(tmp_path: Path, role: str, count: int, shape: str) -> dict[str, str]:
    manifest = {
        "experimental_role": role,
        "concept_id": f"concept-{role}",
        "training_rows": [{"row_id": f"{role}-{index}"} for index in range(count)],
        "training_dataset_shape_sha256": shape,
        "manifest_sha256": _sha(f"manifest-{role}"),
    }
    approval = {"approval_sha256": _sha(f"approval-{role}")}
    return {
        "manifest": str(_write(tmp_path / role / "manifest.json", manifest)),
        "approval": str(_write(tmp_path / role / "approval.json", approval)),
    }


def _profile(
    tmp_path: Path, fixture: str, class_name: str, count: int, shape: str
) -> Path:
    geometry = {
        "A-class": (32, 32, "adamw8bit", "mse", 2.0),
        "B-class": (32, 32, "adamw8bit", "mae", 3.0),
        "C-class": (64, 64, "automagic", "mse", 2.0),
    }[class_name]
    envelope = krea_budget.seal_execution_envelope(
        equivalence_class=class_name,
        network_rank=geometry[0],
        network_alpha=geometry[1],
        optimizer=geometry[2],
        optimizer_config_sha256=_sha("optimizer"),
        loss=geometry[3],
        differential_guidance_enabled=True,
        guidance_scale=geometry[4],
        training_pair_count=count,
        training_dataset_shape_sha256=shape,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        data_parallel_replicas=1,
        resolution_policy_sha256=_sha("resolution"),
        precision_policy_sha256=_sha("precision"),
        cache_latents_to_disk=False,
        cache_text_embeddings=True,
        compile_enabled=False,
        jit_enabled=True,
        dataloader_workers=2,
        base_model_identity_sha256=_sha("base"),
        runtime_identity_sha256=_sha("runtime"),
        host_execution_identity_sha256=_sha("host"),
        execution_surface="staged_host_venv",
        execution_scope="discovery_only",
        venv_tree_manifest_sha256=_sha("venv-tree"),
        reference_container_image_sha256=_sha("container"),
        gpu_identity_sha256=_sha("gpu"),
        trainer_identity_sha256=_sha("trainer"),
        measurement_tool_sha256=_sha("measurement"),
    )
    profile = krea_budget.seal_throughput_profile(
        execution_envelope=envelope,
        raw_sample_manifest_sha256=_sha(f"raw-{fixture}-{class_name}"),
        startup_sample_count=3,
        update_sample_count=100,
        save_sample_count=8,
        startup_upper_bound_s=10,
        update_upper_bound_s=1,
        save_upper_bound_s=2,
        bound_method="observed-max-plus-predeclared-margin",
        margin_policy_sha256=_sha(f"margin-{fixture}-{class_name}"),
        end_to_end_validation_count=1,
        end_to_end_validation_sha256=_sha(f"e2e-{fixture}-{class_name}"),
        framework_stop_boundary_s=225,
        framework_stop_boundary_source_sha256=_sha("boundary"),
        selection_mode="offline_post_training",
        selection_scorer_identity_sha256=None,
        selection_scoring_reserve_s=0,
        finalization_reserve_s=30,
        upload_reserve_s=30,
    )
    return _write(tmp_path / fixture / f"{class_name}.profile.json", profile)


def _profile_payload(tmp_path: Path, monkeypatch) -> tuple[dict, tuple[str, ...]]:
    monkeypatch.setattr(binding.krea_fixture, "validate_manifest", lambda value: value)
    monkeypatch.setattr(
        binding.krea_fixture,
        "validate_approval",
        lambda value, fixture_manifest: value,
    )
    discovery, classes = _discovery(tmp_path)
    authorization_path = _write(
        tmp_path / "discovery-authorization.json",
        {"authorization_sha256": _sha("authorization")},
    )
    authorization_file_sha = krea_provenance.file_sha256(authorization_path)
    monkeypatch.setattr(
        binding.krea_discovery_authorization,
        "load_binding",
        lambda _value: (
            authorization_path,
            {"authorization_sha256": _sha("authorization")},
            authorization_file_sha,
        ),
    )
    monkeypatch.setattr(
        binding.krea_discovery_authorization,
        "assert_matches_discovery",
        lambda *args, **kwargs: None,
    )
    fixture_shapes = {"D1": (18, _sha("shape-D1")), "D2": (36, _sha("shape-D2"))}
    fixtures = {
        role: _fixture(tmp_path, role, count, shape)
        for role, (count, shape) in fixture_shapes.items()
    }
    profiles = {
        role: {
            class_name: str(_profile(tmp_path, role, class_name, count, shape))
            for class_name in classes
        }
        for role, (count, shape) in fixture_shapes.items()
    }
    return {
        "discovery_plan": str(discovery),
        "discovery_execution_authorization": str(authorization_path),
        "fixtures": fixtures,
        "profiles": profiles,
    }, classes


def test_profile_index_binds_six_fixture_scoped_shapes(tmp_path, monkeypatch):
    payload, classes = _profile_payload(tmp_path, monkeypatch)

    result = binding.build_profile_index(payload)

    assert binding.validate_profile_index(result) == result
    assert result["required_profile_count"] == 6
    assert result["throughput_equivalence_classes"] == list(classes)
    assert result["gpu_execution_authorized"] is False
    assert result["fixtures"]["D1"]["training_pair_count"] == 18
    assert result["fixtures"]["D2"]["training_pair_count"] == 36
    for class_name in classes:
        assert (
            result["fixtures"]["D1"]["profiles"][class_name]["profile_sha256"]
            != result["fixtures"]["D2"]["profiles"][class_name]["profile_sha256"]
        )


def test_profile_index_rejects_cross_fixture_profile_shortcut(tmp_path, monkeypatch):
    payload, classes = _profile_payload(tmp_path, monkeypatch)
    payload["profiles"]["D2"][classes[0]] = payload["profiles"]["D1"][classes[0]]

    with pytest.raises(ValueError, match="escaped fixture shape"):
        binding.build_profile_index(payload)


def test_profile_index_never_rewrites_discovery_freeze(tmp_path, monkeypatch):
    payload, _ = _profile_payload(tmp_path, monkeypatch)
    discovery = Path(payload["discovery_plan"])
    before = discovery.read_bytes()

    binding.build_profile_index(payload)

    assert discovery.read_bytes() == before


def _preflight_payload() -> dict:
    return {
        "maximum_load_per_effective_cpu": 0.5,
        "minimum_available_memory_bytes": 64 * 1024**3,
        "minimum_checkpoint_free_bytes": 350 * 1024**3,
        "maximum_gpu_utilization_percent": 5,
        "minimum_free_gpu_memory_mib": 78_000,
        "maximum_foreign_compute_processes": 0,
        "storage_probe_bytes": 16 * 1024**2,
        "minimum_checkpoint_write_mib_s": 100,
        "minimum_checkpoint_read_mib_s": 100,
        "maximum_checkpoint_fsync_s": 5,
    }


def test_preflight_policy_cli_is_canonical_mode_0600_and_create_only(tmp_path):
    payload_path = _write(tmp_path / "policy.payload.json", _preflight_payload())
    output = tmp_path / "policy.json"

    assert (
        binding.main(
            [
                "seal-preflight-policy",
                "--payload",
                str(payload_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    document = json.loads(output.read_bytes())
    assert output.read_bytes() == krea_provenance.canonical_bytes(document) + b"\n"
    assert os.stat(output).st_mode & 0o777 == 0o600
    assert binding.validate_preflight_policy(document) == document
    with pytest.raises(FileExistsError):
        binding.main(
            [
                "seal-preflight-policy",
                "--payload",
                str(payload_path),
                "--output",
                str(output),
            ]
        )


def test_preflight_policy_forbids_any_foreign_gpu_process_allowance():
    payload = _preflight_payload()
    payload["maximum_foreign_compute_processes"] = 1
    with pytest.raises(ValueError, match="owner-ratified"):
        binding.build_preflight_policy(payload)


def test_host_manifest_wrapper_dispatches_ratified_builder(tmp_path, monkeypatch):
    policy = binding.build_preflight_policy(_preflight_payload())
    policy_path = _write(tmp_path / "policy.json", policy)
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    expected = {
        "host_execution_identity_sha256": _sha("host-manifest"),
        "gpu_execution_authorized": False,
    }
    observed = {}

    receipt = _write(tmp_path / "receipt.json", {})

    def build_manifest(*, checkpoint_path, preflight_policy, bootstrap_receipt_path):
        observed["checkpoint"] = checkpoint_path
        observed["policy"] = preflight_policy
        observed["receipt"] = bootstrap_receipt_path
        return expected

    monkeypatch.setattr(binding.krea_host_identity, "build_manifest", build_manifest)
    output = tmp_path / "host.json"

    assert (
        binding.main(
            [
                "build-host-manifest",
                "--checkpoint-path",
                str(checkpoint),
                "--preflight-policy",
                str(policy_path),
                "--bootstrap-receipt",
                str(receipt),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert observed == {
        "checkpoint": checkpoint,
        "policy": policy,
        "receipt": receipt,
    }
    assert json.loads(output.read_bytes()) == expected


def test_execution_plan_wrapper_requires_index_before_publication(
    tmp_path, monkeypatch
):
    payload = {"arm_id": "K1", "gpu_execution_authorized": False}
    payload_path = _write(tmp_path / "plan.payload.json", payload)
    index_path = _write(tmp_path / "index.json", {"index_sha256": _sha("index")})
    plan = {**payload, "plan_sha256": _sha("plan")}
    monkeypatch.setattr(binding, "_load_profile_index", lambda _path: {"index": True})
    from ops.calibration import krea_execution_plan

    monkeypatch.setattr(krea_execution_plan, "seal_plan", lambda value: plan)
    checked = []
    monkeypatch.setattr(
        binding,
        "validate_plan_against_profile_index",
        lambda value, profile_index: checked.append((value, profile_index)),
    )
    output = tmp_path / "plan.json"

    assert (
        binding.main(
            [
                "seal-execution-plan",
                "--payload",
                str(payload_path),
                "--profile-index",
                str(index_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert checked == [(plan, {"index": True})]
    assert json.loads(output.read_bytes()) == plan
