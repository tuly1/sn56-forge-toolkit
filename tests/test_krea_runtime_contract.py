from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
from types import SimpleNamespace

import pytest

from forge import adaptive_timing, config, krea_runtime, recipe
from forge.tasks import aitoolkit, checkpoints
from forge.data.schema import ImageSpec


SOURCE_RUN_ID = "runtime-contract:" + "a" * 32


def _write_training_safetensor(path: Path, *, step: int) -> Path:
    metadata = {"training_info": json.dumps({"step": step, "epoch": 1})}
    header = json.dumps(
        {
            "__metadata__": metadata,
            "weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            },
        }
    ).encode("utf-8")
    path.write_bytes(
        struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.0)
    )
    return path


def _spec(model_type: str = "krea2") -> ImageSpec:
    return ImageSpec.build(
        task_id="runtime-contract",
        model="krea/Krea-2-Raw",
        model_type=model_type,
        expected_repo_name="contract-repo",
        trigger_word="AetherTest UI",
        dataset_zip=None,
    )


def _manifest(tmp_path: Path, *, false_capability: str | None = None) -> Path:
    capabilities = {
        name: True for name in krea_runtime.RUNTIME_MANIFEST_CAPABILITIES
    }
    if false_capability is not None:
        wire_name = krea_runtime._RUNTIME_CAPABILITY_WIRE_ALIASES.get(
            false_capability, false_capability
        )
        capabilities[wire_name] = False
    value = {
        "schema": 1,
        "runtime_contract_id": krea_runtime.RUNTIME_CONTRACT_ID,
        "base_commit": krea_runtime.PINNED_BASE_COMMIT,
        "capabilities": capabilities,
        "evidence": {
            name: f"tests/{name}.py"
            for name in krea_runtime.RUNTIME_MANIFEST_CAPABILITIES
        },
    }
    path = tmp_path / krea_runtime.CAPABILITY_MANIFEST_FILENAME
    path.write_text(json.dumps(value), encoding="utf-8")
    identity = {
        "schema": 1,
        "runtime_repository": krea_runtime.OWNED_RUNTIME_REPOSITORY,
        "runtime_commit": krea_runtime.OWNED_RUNTIME_COMMIT,
        "capability_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    _identity_path(path).write_text(json.dumps(identity), encoding="utf-8")
    return path


def _identity_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(krea_runtime.RUNTIME_IDENTITY_FILENAME)


def _activate(monkeypatch, tmp_path: Path, bundle: str) -> Path:
    path = _manifest(tmp_path)
    monkeypatch.setenv(krea_runtime.BUNDLE_ENV, bundle)
    monkeypatch.setenv(
        krea_runtime.OWNED_KREA_RUNTIME_DIR_ENV, str(tmp_path)
    )
    return path


def _timing_profile(bundle: str, *, startup_seconds: float = 120.0):
    bundle_sha = krea_runtime.bundle_contract_sha256(bundle)
    document = adaptive_timing._seal_profile_document(
        {
            "schema": adaptive_timing.PROFILE_SCHEMA,
            "kind": adaptive_timing.PROFILE_KIND,
            "evidence_scope": "lab-only",
            "bundle_id": bundle,
            "bundle_sha256": bundle_sha,
            "model_type": "krea2",
            "measured_dataset_size": 18,
            "dataset_regime": adaptive_timing.dataset_regime(18),
            "seconds_per_step": 1.3,
            "startup_seconds": startup_seconds,
            "measurement": {
                "completed_steps": 1000,
                "training_elapsed_seconds": startup_seconds + 1300.0,
                "first_checkpoint_step": 200,
                "first_checkpoint_elapsed_seconds": startup_seconds + 260.0,
            },
            "provenance": {
                "source_run_id": SOURCE_RUN_ID,
                "source_record_sha256": "b" * 64,
                "runtime_commit": krea_runtime.OWNED_RUNTIME_COMMIT,
                "measured_at_utc": "2026-08-04T12:00:00Z",
                "accelerator_identity": "NVIDIA H100 PCIe|81559-MiB",
                "accelerator_identity_evidence": "operator-attested",
            },
        }
    )
    return adaptive_timing.validate_profile(
        document,
        expected_bundle_id=bundle,
        expected_bundle_sha256=bundle_sha,
        expected_model_type="krea2",
        current_dataset_size=18,
        expected_dataset_regime=adaptive_timing.dataset_regime(18),
        expected_accelerator_identity="NVIDIA H100 PCIe|81559-MiB",
    )


def test_incumbent_path_is_the_same_object_and_never_reads_manifest(monkeypatch):
    monkeypatch.delenv(krea_runtime.BUNDLE_ENV, raising=False)
    monkeypatch.setenv("FORGE_KREA_CAPABILITY_MANIFEST", "/does/not/exist.json")
    original = {"sentinel": [1, 2, 3]}

    observed, manifest = krea_runtime.apply(original, "krea2")

    assert observed is original
    assert manifest is None


def test_incumbent_config_is_golden_equal_to_deployed_084ea914(monkeypatch):
    monkeypatch.delenv(krea_runtime.BUNDLE_ENV, raising=False)

    cfg = config.build_config(_spec(), num_images=18, hours_to_complete=0.75)
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()

    # Generated independently from the immutable deployed fallback for this
    # exact task shape. This catches a "dormant" feature changing the incumbent.
    assert hashlib.sha256(canonical).hexdigest() == (
        "9ab3612f0bc52e93b24987dac89444bf062e77516c5aa8ca4e07c45779140396"
    )


def test_non_krea_config_ignores_krea_experiment_environment(monkeypatch):
    monkeypatch.setenv(krea_runtime.BUNDLE_ENV, krea_runtime.LEADER_BUNDLE)
    original = {"sentinel": True}

    observed, manifest = krea_runtime.apply(original, "ideogram4")

    assert observed is original
    assert manifest is None


@pytest.mark.parametrize(
    "model_type", ["ideogram4", "qwen-image", "z-image", "flux"]
)
def test_non_krea_runtime_always_uses_exact_incumbent_tree(
    monkeypatch, model_type
):
    monkeypatch.setenv(krea_runtime.BUNDLE_ENV, krea_runtime.LEADER_BUNDLE)
    monkeypatch.setenv(krea_runtime.INCUMBENT_RUNTIME_DIR_ENV, "/incumbent")
    monkeypatch.setenv(krea_runtime.OWNED_KREA_RUNTIME_DIR_ENV, "/owned-krea")

    assert krea_runtime.runtime_directory(model_type) == "/incumbent"


def test_only_experimental_krea_uses_owned_runtime(monkeypatch):
    monkeypatch.setenv(krea_runtime.INCUMBENT_RUNTIME_DIR_ENV, "/incumbent")
    monkeypatch.setenv(krea_runtime.OWNED_KREA_RUNTIME_DIR_ENV, "/owned-krea")

    assert krea_runtime.runtime_directory(
        "krea2", krea_runtime.INCUMBENT_BUNDLE
    ) == "/incumbent"
    for bundle in (
        krea_runtime.LEADER_BUNDLE,
        krea_runtime.LEADER_COMFY_TE_BUNDLE,
        krea_runtime.MAE_BUNDLE,
    ):
        assert krea_runtime.runtime_directory("krea2", bundle) == "/owned-krea"

    incumbent_contract = krea_runtime.bundle_contract_document(
        krea_runtime.INCUMBENT_BUNDLE
    )
    assert incumbent_contract["runtime_repository"] == (
        "https://github.com/ostris/ai-toolkit.git"
    )
    assert incumbent_contract["runtime_commit"] == krea_runtime.PINNED_BASE_COMMIT


def test_attestation_paths_are_derived_from_selected_runtime_and_ignore_legacy_env(
    tmp_path, monkeypatch
):
    path = _activate(monkeypatch, tmp_path, krea_runtime.LEADER_BUNDLE)
    monkeypatch.setenv(
        "FORGE_KREA_CAPABILITY_MANIFEST", "/parallel/attestation.json"
    )
    monkeypatch.setenv(
        "FORGE_KREA_RUNTIME_IDENTITY", "/parallel/identity.json"
    )

    runtime_dir, manifest_path, identity_path = (
        krea_runtime.runtime_attestation_paths(
            "krea2", krea_runtime.LEADER_BUNDLE
        )
    )

    assert runtime_dir == str(tmp_path)
    assert manifest_path == str(path)
    assert identity_path == str(_identity_path(path))
    assert krea_runtime.load_capability_manifest(
        model_type="krea2", bundle=krea_runtime.LEADER_BUNDLE
    )["runtime_contract_id"] == krea_runtime.RUNTIME_CONTRACT_ID


def test_git_verifier_rejects_a_dirty_selected_runtime_tree(tmp_path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    subprocess.run(["git", "init", "-q", str(runtime_dir)], check=True)
    subprocess.run(
        ["git", "-C", str(runtime_dir), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(runtime_dir), "config", "user.name", "Test"],
        check=True,
    )
    (runtime_dir / "run.py").write_text("print('clean')\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(runtime_dir), "add", "run.py"], check=True)
    subprocess.run(
        ["git", "-C", str(runtime_dir), "commit", "-qm", "runtime"],
        check=True,
    )
    repository = "https://github.com/example/runtime.git"
    subprocess.run(
        ["git", "-C", str(runtime_dir), "remote", "add", "origin", repository],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(runtime_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    krea_runtime._verify_git_checkout(
        str(runtime_dir),
        expected_commit=commit,
        expected_repository=repository,
        runner=subprocess.run,
    )
    (runtime_dir / "run.py").write_text("print('mutated')\n", encoding="utf-8")

    with pytest.raises(
        krea_runtime.KreaRuntimeContractError, match="working tree"
    ):
        krea_runtime._verify_git_checkout(
            str(runtime_dir),
            expected_commit=commit,
            expected_repository=repository,
            runner=subprocess.run,
        )


def test_stable_bundle_ids_carry_honest_source_derived_claims():
    rank1 = krea_runtime.bundle_claim_document(krea_runtime.LEADER_BUNDLE)
    rank3 = krea_runtime.bundle_claim_document(krea_runtime.MAE_BUNDLE)

    assert rank1["source_config_sha256"] == krea_runtime.PUBLIC_RANK1_CONFIG_SHA256
    assert rank3["source_config_sha256"] == krea_runtime.PUBLIC_RANK3_CONFIG_SHA256
    assert rank1["source_repository"] == krea_runtime.PUBLIC_RANK1_REPOSITORY
    assert rank1["source_revision"] == krea_runtime.PUBLIC_RANK1_REVISION
    assert rank1["source_config_path"] == krea_runtime.PUBLIC_CONFIG_PATH
    assert rank3["source_repository"] == krea_runtime.PUBLIC_RANK3_REPOSITORY
    assert rank3["source_revision"] == krea_runtime.PUBLIC_RANK3_REVISION
    assert rank3["source_config_path"] == krea_runtime.PUBLIC_CONFIG_PATH
    assert rank1["byte_equivalent_to_source_config"] is False
    assert rank3["byte_equivalent_to_source_config"] is False
    assert "source-derived" in rank1["classification"]
    assert "source-derived" in rank3["classification"]


def test_unknown_bundle_is_fatal(monkeypatch):
    monkeypatch.setenv(krea_runtime.BUNDLE_ENV, "leader-typo")

    with pytest.raises(krea_runtime.KreaRuntimeContractError, match="unknown"):
        krea_runtime.requested_bundle("krea2")


def test_timing_probe_requires_literal_opt_in(monkeypatch):
    monkeypatch.delenv(krea_runtime.TIMING_PROBE_ENV, raising=False)
    assert krea_runtime.timing_probe_enabled() is False
    monkeypatch.setenv(krea_runtime.TIMING_PROBE_ENV, "1")
    assert krea_runtime.timing_probe_enabled() is True
    monkeypatch.setenv(krea_runtime.TIMING_PROBE_ENV, "yes")
    with pytest.raises(krea_runtime.KreaRuntimeContractError, match="literal"):
        krea_runtime.timing_probe_enabled()


@pytest.mark.parametrize("missing", krea_runtime.REQUIRED_CAPABILITIES)
def test_leader_fails_closed_for_each_missing_runtime_capability(
    tmp_path, monkeypatch, missing
):
    path = _manifest(tmp_path, false_capability=missing)
    monkeypatch.setenv(krea_runtime.BUNDLE_ENV, krea_runtime.LEADER_BUNDLE)
    monkeypatch.setenv(
        krea_runtime.OWNED_KREA_RUNTIME_DIR_ENV, str(tmp_path)
    )

    with pytest.raises(krea_runtime.KreaRuntimeContractError, match=missing):
        config.build_config(_spec(), num_images=18, hours_to_complete=0.75)


def test_capability_manifest_rejects_unknown_claim(tmp_path, monkeypatch):
    path = _manifest(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["capabilities"]["magic_yaml_key"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setenv(krea_runtime.BUNDLE_ENV, krea_runtime.LEADER_BUNDLE)
    monkeypatch.setenv(krea_runtime.OWNED_KREA_RUNTIME_DIR_ENV, str(tmp_path))

    with pytest.raises(krea_runtime.KreaRuntimeContractError, match="differ"):
        krea_runtime.load_capability_manifest()


def test_rank1_source_derived_bundle_applies_declared_fields(tmp_path, monkeypatch):
    _activate(monkeypatch, tmp_path, krea_runtime.LEADER_BUNDLE)

    cfg = config.build_config(_spec(), num_images=18, hours_to_complete=0.75)
    p = cfg["config"]["process"][0]
    train = p["train"]

    assert p["network"] == {"type": "lora", "linear": 32, "linear_alpha": 32}
    assert p["datasets"][0]["caption_dropout_rate"] == 0.05
    assert p["datasets"][0]["cache_latents_to_disk"] is True
    assert p["save"]["save_every"] == 200
    assert p["save"]["max_step_saves_to_keep"] == 12
    assert train["train_text_encoder"] is True
    assert train["unet_lr"] == pytest.approx(1e-4)
    assert train["text_encoder_lr"] == pytest.approx(2.5e-7)
    assert train["timestep_type"] == "krea2_eval_sigmas"
    assert train["differential_guidance_scale"] == pytest.approx(12.0)
    assert train["ema_config"] == {"use_ema": True, "ema_decay": 0.995}
    assert train["lr_scheduler"] == "cosine_by_group"
    assert train["multires_noise_iterations"] == 6
    assert train["multires_noise_discount"] == pytest.approx(0.3)
    assert train["sn56_strict_krea_fields"] is True
    assert "sn56_krea_comfy_text_encoder_export" not in train


def test_comfy_te_bundle_is_distinct_and_only_changes_export_contract(
    tmp_path, monkeypatch
):
    _activate(monkeypatch, tmp_path, krea_runtime.LEADER_BUNDLE)
    leader = config.build_config(_spec(), num_images=18, hours_to_complete=0.75)
    monkeypatch.setenv(
        krea_runtime.BUNDLE_ENV, krea_runtime.LEADER_COMFY_TE_BUNDLE
    )
    effective = config.build_config(
        _spec(), num_images=18, hours_to_complete=0.75
    )
    leader_train = leader["config"]["process"][0]["train"]
    effective_train = effective["config"]["process"][0]["train"]

    assert effective_train.pop("sn56_krea_comfy_text_encoder_export") is True
    assert effective_train == leader_train
    assert krea_runtime.bundle_contract_sha256(
        krea_runtime.LEADER_COMFY_TE_BUNDLE
    ) != krea_runtime.bundle_contract_sha256(krea_runtime.LEADER_BUNDLE)


def test_leader_overlay_never_partially_mutates_incumbent(tmp_path, monkeypatch):
    monkeypatch.delenv(krea_runtime.BUNDLE_ENV, raising=False)
    incumbent = config.build_config(_spec(), num_images=18, hours_to_complete=0.75)
    snapshot = json.loads(json.dumps(incumbent))
    path = _manifest(tmp_path, false_capability="ema_checkpoint_resume")
    env = {
        krea_runtime.BUNDLE_ENV: krea_runtime.LEADER_BUNDLE,
        krea_runtime.OWNED_KREA_RUNTIME_DIR_ENV: str(tmp_path),
    }

    with pytest.raises(krea_runtime.KreaRuntimeContractError):
        krea_runtime.apply(incumbent, "krea2", environ=env)

    assert incumbent == snapshot


def test_rank3_source_derived_bundle_applies_declared_fields(tmp_path, monkeypatch):
    _activate(monkeypatch, tmp_path, krea_runtime.MAE_BUNDLE)

    cfg = config.build_config(_spec(), num_images=18, hours_to_complete=0.75)
    p = cfg["config"]["process"][0]
    train = p["train"]

    assert p["network"] == {"type": "lora", "linear": 32, "linear_alpha": 32}
    assert "caption_dropout_rate" not in p["datasets"][0]
    assert train["loss_type"] == "mae"
    assert train["differential_guidance_scale"] == pytest.approx(3.0)
    assert train["timestep_type"] == "linear"
    assert train["train_text_encoder"] is False
    assert train["ema_config"] == {"use_ema": False}


def test_effective_runtime_record_hash_binds_exact_generated_config(
    tmp_path, monkeypatch
):
    manifest_path = _activate(
        monkeypatch, tmp_path, krea_runtime.LEADER_BUNDLE
    )
    cfg = config.build_config(_spec(), num_images=18, hours_to_complete=0.75)
    config_path = tmp_path / "task.yaml"
    config.write_config(cfg, str(config_path))
    manifest = krea_runtime.load_capability_manifest()

    record = krea_runtime.emit_effective_runtime_record(
        cfg,
        "krea2",
        str(config_path),
        manifest,
        source_run_id=SOURCE_RUN_ID,
        timing_probe=True,
        current_dataset_size=18,
        current_accelerator_identity="NVIDIA H100 PCIe|81559-MiB",
    )
    on_disk = json.loads(
        (tmp_path / "task.yaml.effective-runtime.json").read_text(encoding="utf-8")
    )

    assert on_disk == record
    assert record["generated_config_sha256"] == hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    declared = record.pop("record_sha256")
    assert declared == krea_runtime._canonical_sha256(record)
    assert set(record["effective"]) == {
        "planned_steps",
        "normalized_config_projection",
    }
    assert record["timing"]["mode"] == "bootstrap_probe_unmeasured"
    assert record["timing"]["measured_dataset_size"] is None
    assert record["timing"]["current_dataset_size"] == 18
    assert record["timing"]["dataset_regime"] == "small-11-24"
    assert record["timing"]["accelerator_identity"] == (
        "NVIDIA H100 PCIe|81559-MiB"
    )
    assert record["runtime_commit"] == krea_runtime.OWNED_RUNTIME_COMMIT
    assert record["timing"]["runtime_commit"] == krea_runtime.OWNED_RUNTIME_COMMIT
    assert record["bundle_claim"]["byte_equivalent_to_source_config"] is False
    assert record["source_run_id"] == SOURCE_RUN_ID
    assert record["model_type"] == "krea2"
    assert krea_runtime.COMPONENT_RECOVERY_CAPABILITY in record["capabilities"]
    assert "ema_checkpoint_resume" not in record["capabilities"]
    assert record["runtime_manifest_capability_aliases"] == {
        krea_runtime.COMPONENT_RECOVERY_CAPABILITY: "ema_checkpoint_resume"
    }
    assert record["capability_manifest_file_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert record["capability_manifest_semantic_sha256"] == (
        krea_runtime._canonical_sha256(manifest)
    )
    # Task paths, trigger strings, and credentials do not enter this sidecar.
    assert "AetherTest" not in json.dumps(record)


def test_bootstrap_emitter_persistence_and_profile_producer_are_schema_compatible(
    tmp_path, monkeypatch
):
    _activate(monkeypatch, tmp_path, krea_runtime.LEADER_BUNDLE)
    cfg = config.build_config(_spec(), num_images=18, hours_to_complete=0.75)
    config_path = tmp_path / "bootstrap.yaml"
    config.write_config(cfg, str(config_path))
    planned = cfg["config"]["process"][0]["train"]["steps"]
    save_root = tmp_path / "bootstrap-save"
    scope = checkpoints.begin_run(str(save_root), "contract-repo")
    scope = checkpoints.set_planned_steps(str(save_root), scope, planned)
    source_run_id = f"runtime-contract:{scope['attempt_nonce']}"
    krea_runtime.emit_effective_runtime_record(
        cfg,
        "krea2",
        str(config_path),
        krea_runtime.load_capability_manifest(),
        source_run_id=source_run_id,
        timing_probe=True,
        current_dataset_size=18,
        current_accelerator_identity="NVIDIA H100 PCIe|81559-MiB",
    )
    observation = adaptive_timing.emit_bootstrap_first_checkpoint_observation(
        bundle_id=krea_runtime.LEADER_BUNDLE,
        checkpoint_step=200,
        elapsed_since_launch_s=400.0,
        active_planned_steps=planned,
        event_sink=lambda *_args, **_kwargs: None,
    )
    krea_runtime.persist_first_checkpoint_observation(
        str(config_path), observation
    )
    artifact = _write_training_safetensor(
        save_root / "contract-repo.safetensors", step=planned
    )
    krea_runtime.persist_training_completion_observation(
        str(config_path),
        artifact_path=str(artifact),
        save_root=str(save_root),
        scope=scope,
        training_elapsed_seconds=float(planned * 2),
        returncode=0,
        stopped_by_deadline=False,
    )
    source_path = Path(str(config_path) + ".effective-runtime.json")

    profile = adaptive_timing.produce_profile_document(
        str(source_path),
        source_run_id=source_run_id,
        bundle_id=krea_runtime.LEADER_BUNDLE,
        model_type="krea2",
        measured_dataset_size=18,
        measured_at_utc="2026-08-04T18:00:00Z",
        runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="NVIDIA H100 PCIe, 81559\n",
            stderr="",
        ),
    )

    assert profile["measurement"]["completed_steps"] == planned
    assert profile["seconds_per_step"] == pytest.approx(2.0)
    assert profile["provenance"]["source_record_sha256"] == hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()


def test_effective_record_lifecycle_is_ordered_and_terminal_immutable(
    tmp_path, monkeypatch
):
    _activate(monkeypatch, tmp_path, krea_runtime.LEADER_BUNDLE)
    spec = _spec()
    cfg = config.build_config(spec, num_images=18, hours_to_complete=0.75)
    config_path = tmp_path / "lifecycle.yaml"
    config.write_config(cfg, str(config_path))
    planned = cfg["config"]["process"][0]["train"]["steps"]
    save_root = tmp_path / "lifecycle-save"
    scope = checkpoints.begin_run(str(save_root), spec.expected_repo_name)
    scope = checkpoints.set_planned_steps(str(save_root), scope, planned)
    krea_runtime.emit_effective_runtime_record(
        cfg,
        "krea2",
        str(config_path),
        krea_runtime.load_capability_manifest(),
        source_run_id=f"{spec.task_id}:{scope['attempt_nonce']}",
        timing_probe=True,
        current_dataset_size=18,
        current_accelerator_identity="NVIDIA H100 PCIe|81559-MiB",
    )
    artifact = _write_training_safetensor(
        save_root / f"{spec.expected_repo_name}.safetensors",
        step=planned,
    )
    record_path = Path(str(config_path) + ".effective-runtime.json")
    bootstrap_bytes = record_path.read_bytes()

    old_draft = json.loads(bootstrap_bytes)
    old_draft["schema"] = 4
    old_draft.pop("record_sha256")
    old_draft["record_sha256"] = krea_runtime._canonical_sha256(old_draft)
    record_path.write_bytes(krea_runtime._canonical_bytes(old_draft))
    with pytest.raises(
        krea_runtime.KreaRuntimeContractError, match="schema is unsupported"
    ):
        krea_runtime.persist_first_checkpoint_observation(
            str(config_path),
            adaptive_timing.emit_bootstrap_first_checkpoint_observation(
                bundle_id=krea_runtime.LEADER_BUNDLE,
                checkpoint_step=200,
                elapsed_since_launch_s=300.0,
                active_planned_steps=planned,
                event_sink=lambda *_args, **_kwargs: None,
            ),
        )
    record_path.write_bytes(bootstrap_bytes)

    with pytest.raises(
        krea_runtime.KreaRuntimeContractError, match="out of order"
    ):
        krea_runtime.persist_training_completion_observation(
            str(config_path),
            artifact_path=str(artifact),
            save_root=str(save_root),
            scope=scope,
            training_elapsed_seconds=1000.0,
            returncode=0,
            stopped_by_deadline=False,
        )
    assert record_path.read_bytes() == bootstrap_bytes

    observation = adaptive_timing.emit_bootstrap_first_checkpoint_observation(
        bundle_id=krea_runtime.LEADER_BUNDLE,
        checkpoint_step=200,
        elapsed_since_launch_s=300.0,
        active_planned_steps=planned,
        event_sink=lambda *_args, **_kwargs: None,
    )
    krea_runtime.persist_first_checkpoint_observation(
        str(config_path), observation
    )
    completed = krea_runtime.persist_training_completion_observation(
        str(config_path),
        artifact_path=str(artifact),
        save_root=str(save_root),
        scope=scope,
        training_elapsed_seconds=1000.0,
        returncode=0,
        stopped_by_deadline=False,
    )
    assert completed["lifecycle"] == "terminal"
    assert completed["training_completion_observation"]["natural_completion"] is True
    terminal_bytes = record_path.read_bytes()

    with pytest.raises(
        krea_runtime.KreaRuntimeContractError, match="out of order"
    ):
        krea_runtime.persist_training_completion_observation(
            str(config_path),
            artifact_path=str(artifact),
            save_root=str(save_root),
            scope=scope,
            training_elapsed_seconds=1001.0,
            returncode=0,
            stopped_by_deadline=False,
        )
    with pytest.raises(
        krea_runtime.KreaRuntimeContractError, match="out of order"
    ):
        krea_runtime.persist_first_checkpoint_observation(
            str(config_path), observation
        )
    assert record_path.read_bytes() == terminal_bytes


def test_terminal_artifact_symlink_and_wrong_scope_abort(tmp_path, monkeypatch):
    _activate(monkeypatch, tmp_path, krea_runtime.LEADER_BUNDLE)
    spec = _spec()
    cfg = config.build_config(spec, num_images=18, hours_to_complete=0.75)
    config_path = tmp_path / "artifact-scope.yaml"
    config.write_config(cfg, str(config_path))
    planned = cfg["config"]["process"][0]["train"]["steps"]
    save_root = tmp_path / "artifact-scope-save"
    scope = checkpoints.begin_run(str(save_root), spec.expected_repo_name)
    scope = checkpoints.set_planned_steps(str(save_root), scope, planned)
    krea_runtime.emit_effective_runtime_record(
        cfg,
        "krea2",
        str(config_path),
        krea_runtime.load_capability_manifest(),
        source_run_id=f"{spec.task_id}:{scope['attempt_nonce']}",
        timing_probe=True,
        current_dataset_size=18,
        current_accelerator_identity="NVIDIA H100 PCIe|81559-MiB",
    )
    observation = adaptive_timing.emit_bootstrap_first_checkpoint_observation(
        bundle_id=krea_runtime.LEADER_BUNDLE,
        checkpoint_step=200,
        elapsed_since_launch_s=300.0,
        active_planned_steps=planned,
        event_sink=lambda *_args, **_kwargs: None,
    )
    krea_runtime.persist_first_checkpoint_observation(
        str(config_path), observation
    )
    target = _write_training_safetensor(
        tmp_path / "outside-terminal.safetensors", step=planned
    )
    link = save_root / f"{spec.expected_repo_name}.safetensors"
    link.symlink_to(target)

    with pytest.raises(krea_runtime.KreaRuntimeContractError):
        krea_runtime.persist_training_completion_observation(
            str(config_path),
            artifact_path=str(link),
            save_root=str(save_root),
            scope=scope,
            training_elapsed_seconds=1000.0,
            returncode=0,
            stopped_by_deadline=False,
        )

    link.unlink()
    artifact = _write_training_safetensor(link, step=planned)
    wrong_scope = {**scope, "attempt_nonce": "b" * 32}
    with pytest.raises(
        krea_runtime.KreaRuntimeContractError, match="scope identity mismatch"
    ):
        krea_runtime.persist_training_completion_observation(
            str(config_path),
            artifact_path=str(artifact),
            save_root=str(save_root),
            scope=wrong_scope,
            training_elapsed_seconds=1000.0,
            returncode=0,
            stopped_by_deadline=False,
        )


def test_terminal_artifact_stale_same_process_scope_aborts(tmp_path):
    save_root = tmp_path / "same-process-scope"
    scope_a = checkpoints.begin_run(str(save_root), "contract-repo")
    scope_a = checkpoints.set_planned_steps(str(save_root), scope_a, 1000)
    scope_b = checkpoints.begin_run(str(save_root), "contract-repo")
    scope_b = checkpoints.set_planned_steps(str(save_root), scope_b, 1000)
    artifact = _write_training_safetensor(
        save_root / "contract-repo.safetensors", step=1000
    )

    from forge.tasks.integrity import inspect_training_artifact

    evidence = inspect_training_artifact(str(artifact))
    assert not checkpoints.descriptor_is_current_lora(
        str(save_root), str(artifact), scope_a, evidence.file_identity
    )
    assert checkpoints.descriptor_is_current_lora(
        str(save_root), str(artifact), scope_b, evidence.file_identity
    )


def test_experimental_record_emission_is_mandatory(tmp_path, monkeypatch):
    _activate(monkeypatch, tmp_path, krea_runtime.LEADER_BUNDLE)
    cfg = config.build_config(_spec(), num_images=18, hours_to_complete=0.75)
    config_path = tmp_path / "missing-parent" / "task.yaml"
    config_path.parent.mkdir()
    config.write_config(cfg, str(config_path))
    monkeypatch.setattr(krea_runtime, "_atomic_json", lambda *_args: (_ for _ in ()).throw(OSError()))

    with pytest.raises(
        krea_runtime.KreaRuntimeContractError, match="could not be emitted"
    ):
        krea_runtime.emit_effective_runtime_record(
            cfg,
            "krea2",
            str(config_path),
            krea_runtime.load_capability_manifest(),
            source_run_id=SOURCE_RUN_ID,
            timing_probe=True,
            current_dataset_size=18,
            current_accelerator_identity="NVIDIA H100 PCIe|81559-MiB",
        )


def test_operator_attested_profile_is_bound_to_bundle_in_effective_record(
    tmp_path, monkeypatch
):
    _activate(monkeypatch, tmp_path, krea_runtime.LEADER_BUNDLE)
    profile = _timing_profile(krea_runtime.LEADER_BUNDLE)
    cfg = config.build_config(
        _spec(),
        num_images=18,
        hours_to_complete=0.75,
        throughput_profile=profile,
    )
    config_path = tmp_path / "measured.yaml"
    config.write_config(cfg, str(config_path))

    record = krea_runtime.emit_effective_runtime_record(
        cfg,
        "krea2",
        str(config_path),
        krea_runtime.load_capability_manifest(),
        source_run_id=SOURCE_RUN_ID,
        throughput_profile=profile,
        current_dataset_size=18,
    )

    assert record["timing"] == {
        "mode": "operator_attested_profile",
        "profile_sha256": profile.profile_sha256,
        "runtime_commit": profile.runtime_commit,
        "measured_dataset_size": 18,
        "current_dataset_size": 18,
        "dataset_regime": adaptive_timing.dataset_regime(18),
        "accelerator_identity": profile.accelerator_identity,
        "accelerator_identity_evidence": "operator-attested",
    }
    assert record["bundle_contract_sha256"] == profile.bundle_sha256


def test_effective_record_preserves_measured_and_current_regime_sizes(
    tmp_path, monkeypatch
):
    _activate(monkeypatch, tmp_path, krea_runtime.LEADER_BUNDLE)
    profile = _timing_profile(krea_runtime.LEADER_BUNDLE)
    cfg = config.build_config(
        _spec(), num_images=20, hours_to_complete=0.75,
        throughput_profile=profile,
    )
    config_path = tmp_path / "same-regime.yaml"
    config.write_config(cfg, str(config_path))

    record = krea_runtime.emit_effective_runtime_record(
        cfg,
        "krea2",
        str(config_path),
        krea_runtime.load_capability_manifest(),
        source_run_id=SOURCE_RUN_ID,
        throughput_profile=profile,
        current_dataset_size=20,
    )

    assert record["timing"]["measured_dataset_size"] == 18
    assert record["timing"]["current_dataset_size"] == 20
    assert record["timing"]["dataset_regime"] == "small-11-24"


def test_effective_record_rejects_profile_from_another_runtime(
    tmp_path, monkeypatch
):
    _activate(monkeypatch, tmp_path, krea_runtime.LEADER_BUNDLE)
    profile = _timing_profile(krea_runtime.LEADER_BUNDLE)
    foreign_profile = adaptive_timing.ThroughputProfile(
        **{
            **profile.__dict__,
            "runtime_commit": "c" * 40,
        }
    )
    cfg = config.build_config(
        _spec(),
        num_images=18,
        hours_to_complete=0.75,
        throughput_profile=foreign_profile,
    )
    config_path = tmp_path / "foreign-runtime.yaml"
    config.write_config(cfg, str(config_path))

    with pytest.raises(krea_runtime.KreaRuntimeContractError, match="binding"):
        krea_runtime.emit_effective_runtime_record(
            cfg,
            "krea2",
            str(config_path),
            krea_runtime.load_capability_manifest(),
            source_run_id=SOURCE_RUN_ID,
            throughput_profile=foreign_profile,
            current_dataset_size=18,
        )


def test_experimental_effective_record_uses_reviewed_release_constant(
    tmp_path, monkeypatch
):
    _activate(monkeypatch, tmp_path, krea_runtime.LEADER_BUNDLE)
    cfg = config.build_config(_spec(), 18, 0.75)
    config_path = tmp_path / "static.yaml"
    config.write_config(cfg, str(config_path))

    record = krea_runtime.emit_effective_runtime_record(
        cfg,
        "krea2",
        str(config_path),
        krea_runtime.load_capability_manifest(),
        source_run_id=SOURCE_RUN_ID,
        current_dataset_size=18,
    )

    timing = record["timing"]
    assert record["schema"] == krea_runtime.EFFECTIVE_RUNTIME_SCHEMA
    assert record["lifecycle"] == "release_constant"
    assert timing["mode"] == "reviewed_release_constant"
    assert timing["profile_sha256"] is None
    assert timing["release_timing_policy"]["seconds_per_step"] == 2.2
    assert timing["release_timing_policy_sha256"] == krea_runtime._canonical_sha256(
        timing["release_timing_policy"]
    )


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("dataset", "resolution", [384]),
        ("dataset", "cache_latents_to_disk", False),
        ("train", "batch_size", 2),
        ("train", "optimizer", "adamw"),
        ("train", "optimizer_params", {"weight_decay": 0.2}),
        ("train", "dtype", "fp32"),
        ("train", "gradient_checkpointing", False),
        ("train", "do_differential_guidance", False),
        ("train", "ema_config", {"use_ema": False}),
        ("train", "noise_scheduler", "ddpm"),
    ],
)
def test_timing_contract_rejects_throughput_semantic_drift(
    tmp_path, monkeypatch, section, field, replacement
):
    _activate(monkeypatch, tmp_path, krea_runtime.LEADER_BUNDLE)
    profile = _timing_profile(krea_runtime.LEADER_BUNDLE)
    cfg = config.build_config(
        _spec(), 18, 0.75, throughput_profile=profile
    )
    p = cfg["config"]["process"][0]
    target = p["datasets"][0] if section == "dataset" else p["train"]
    target[field] = replacement
    config_path = tmp_path / f"drift-{section}-{field}.yaml"
    config.write_config(cfg, str(config_path))

    with pytest.raises(
        krea_runtime.KreaRuntimeContractError,
        match="timing contract",
    ):
        krea_runtime.emit_effective_runtime_record(
            cfg,
            "krea2",
            str(config_path),
            krea_runtime.load_capability_manifest(),
            source_run_id=SOURCE_RUN_ID,
            throughput_profile=profile,
            current_dataset_size=18,
        )


def test_runtime_identity_must_match_exact_owned_commit(tmp_path, monkeypatch):
    path = _manifest(tmp_path)
    identity_path = _identity_path(path)
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["runtime_commit"] = "f" * 40
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    monkeypatch.setenv(krea_runtime.BUNDLE_ENV, krea_runtime.LEADER_BUNDLE)
    monkeypatch.setenv(krea_runtime.OWNED_KREA_RUNTIME_DIR_ENV, str(tmp_path))

    with pytest.raises(krea_runtime.KreaRuntimeContractError, match="identity"):
        krea_runtime.load_capability_manifest()


def test_default_incumbent_needs_no_effective_runtime_side_effect():
    assert not krea_runtime.should_emit_effective_runtime_record(
        bundle=krea_runtime.INCUMBENT_BUNDLE
    )
    assert krea_runtime.should_emit_effective_runtime_record(
        bundle=krea_runtime.INCUMBENT_BUNDLE,
        timing_probe=True,
    )
    assert krea_runtime.should_emit_effective_runtime_record(
        bundle=krea_runtime.LEADER_BUNDLE
    )


def _localize_spec(monkeypatch, tmp_path: Path, spec: ImageSpec) -> None:
    values = {
        "save_root": str(tmp_path / "checkpoints" / spec.expected_repo_name),
        "training_folder": str(tmp_path / "checkpoints"),
        "config_path": str(tmp_path / "configs" / "task.yaml"),
        "cached_model_dir": str(tmp_path / "model"),
        "cached_zip_path": str(tmp_path / "dataset.zip"),
        "dataset_images_dir": str(tmp_path / "images"),
        "dataset_holdout_dir": str(tmp_path / "holdout"),
    }
    for name, value in values.items():
        monkeypatch.setattr(
            type(spec), name, property(lambda _self, v=value: v)
        )


def test_default_incumbent_runner_does_not_emit_record(
    tmp_path, monkeypatch
):
    spec = _spec()
    _localize_spec(monkeypatch, tmp_path, spec)
    monkeypatch.delenv(krea_runtime.BUNDLE_ENV, raising=False)
    monkeypatch.delenv(krea_runtime.TIMING_PROBE_ENV, raising=False)
    monkeypatch.delenv(adaptive_timing.PROFILE_ENV, raising=False)
    monkeypatch.setattr(
        aitoolkit.dataset,
        "prepare_aitoolkit_dataset",
        lambda *_args, **_kwargs: (spec.dataset_images_dir, 18),
    )
    monkeypatch.setattr(aitoolkit.holdout, "budget_allows", lambda *_args: False)
    monkeypatch.setattr(aitoolkit.holdout, "enabled_for", lambda *_args: False)
    monkeypatch.setattr(aitoolkit, "_run_toolkit", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(aitoolkit, "_finalize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        krea_runtime,
        "emit_effective_runtime_record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("default incumbent must not emit a runtime record")
        ),
    )

    class Deadline:
        def remaining(self):
            return 10_000.0

        def remaining_hard(self):
            return 10_000.0

    aitoolkit.run(spec, Deadline())

    assert not Path(spec.config_path + ".effective-runtime.json").exists()


def test_production_runner_ignores_host_bound_profile_and_probe_environment(
    tmp_path, monkeypatch
):
    spec = _spec()
    _localize_spec(monkeypatch, tmp_path, spec)
    _activate(monkeypatch, tmp_path, krea_runtime.LEADER_BUNDLE)
    monkeypatch.setenv(adaptive_timing.PROFILE_ENV, "/host/lab-profile.json")
    monkeypatch.setenv(adaptive_timing.SOURCE_RECORD_ENV, "/host/raw-record.json")
    monkeypatch.setenv(krea_runtime.TIMING_PROBE_ENV, "1")
    monkeypatch.setattr(
        adaptive_timing,
        "load_bundle_profile",
        lambda **_kwargs: pytest.fail("production loaded a host-bound profile"),
    )
    monkeypatch.setattr(
        adaptive_timing,
        "current_accelerator_identity",
        lambda **_kwargs: pytest.fail("production probed host identity"),
    )
    monkeypatch.setattr(
        krea_runtime,
        "timing_probe_enabled",
        lambda *_args, **_kwargs: pytest.fail("production entered timing-probe mode"),
    )
    monkeypatch.setattr(
        aitoolkit.dataset,
        "prepare_aitoolkit_dataset",
        lambda *_args, **_kwargs: (spec.dataset_images_dir, 18),
    )
    monkeypatch.setattr(aitoolkit.holdout, "budget_allows", lambda *_args: False)
    monkeypatch.setattr(aitoolkit.holdout, "enabled_for", lambda *_args: False)
    monkeypatch.setattr(aitoolkit, "_run_toolkit", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(aitoolkit, "_finalize", lambda *_args, **_kwargs: None)

    class Deadline:
        def remaining(self):
            return 10_000.0

        def remaining_hard(self):
            return 10_000.0

    aitoolkit.run(spec, Deadline())

    record = json.loads(
        Path(spec.config_path + ".effective-runtime.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["lifecycle"] == "release_constant"
    assert record["timing"]["mode"] == "reviewed_release_constant"
    assert record["timing"]["release_timing_policy"]["seconds_per_step"] == 2.2
    assert "accelerator_identity" not in record["timing"]


def test_attested_tree_different_from_executed_tree_aborts_before_launch(
    tmp_path, monkeypatch
):
    spec = _spec()
    _localize_spec(monkeypatch, tmp_path, spec)
    attested = tmp_path / "attested-runtime"
    executed = tmp_path / "executed-runtime"
    attested.mkdir()
    executed.mkdir()
    _activate(monkeypatch, attested, krea_runtime.LEADER_BUNDLE)
    (attested / "run.py").write_text("pass\n", encoding="utf-8")
    (executed / "run.py").write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(
        krea_runtime,
        "_verify_git_checkout",
        lambda runtime_dir, **_kwargs: (
            None
            if runtime_dir == str(attested)
            else pytest.fail("verifier received a parallel runtime path")
        ),
    )
    monkeypatch.setattr(
        aitoolkit.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "subprocess launched after runtime identity mismatch"
        ),
    )
    Path(spec.save_root).mkdir(parents=True)
    scope = checkpoints.begin_run(spec.save_root, spec.expected_repo_name)

    class Deadline:
        def remaining(self):
            return 10_000.0

    with pytest.raises(
        krea_runtime.KreaRuntimeContractError,
        match="attested.*differs.*executable",
    ):
        aitoolkit._run_toolkit(
            str(tmp_path / "unused.yaml"),
            Deadline(),
            spec,
            scope,
            timing_bundle=krea_runtime.LEADER_BUNDLE,
            toolkit_dir=str(executed),
        )


def test_caller_cannot_downgrade_experimental_runtime_verification(
    tmp_path, monkeypatch
):
    spec = _spec()
    _localize_spec(monkeypatch, tmp_path, spec)
    monkeypatch.setenv(krea_runtime.BUNDLE_ENV, krea_runtime.LEADER_BUNDLE)
    monkeypatch.setattr(
        aitoolkit.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "subprocess launched after bundle inconsistency"
        ),
    )
    Path(spec.save_root).mkdir(parents=True)
    scope = checkpoints.begin_run(spec.save_root, spec.expected_repo_name)

    class Deadline:
        def remaining(self):
            return 10_000.0

    with pytest.raises(
        krea_runtime.KreaRuntimeContractError,
        match="timing bundle differs.*selected runtime bundle",
    ):
        aitoolkit._run_toolkit(
            str(tmp_path / "unused.yaml"),
            Deadline(),
            spec,
            scope,
            timing_bundle=krea_runtime.INCUMBENT_BUNDLE,
            toolkit_dir=str(tmp_path / "arbitrary-runtime"),
        )


def test_timing_source_run_id_includes_exact_attempt_nonce():
    nonce = "a" * 32

    assert aitoolkit._timing_source_run_id(
        _spec(), {"attempt_nonce": nonce}
    ) == f"runtime-contract:{nonce}"


def test_integrated_fake_process_persists_first_and_terminal_observations(
    tmp_path, monkeypatch
):
    spec = _spec()
    _localize_spec(monkeypatch, tmp_path, spec)
    toolkit_dir = tmp_path / "fake-toolkit"
    toolkit_dir.mkdir()
    _activate(monkeypatch, toolkit_dir, krea_runtime.LEADER_BUNDLE)
    profile = _timing_profile(
        krea_runtime.LEADER_BUNDLE, startup_seconds=0.0
    )
    cfg = config.build_config(
        spec, 18, 0.75, throughput_profile=profile
    )
    Path(spec.config_path).parent.mkdir(parents=True)
    config.write_config(cfg, spec.config_path)
    meta_updates = []
    monkeypatch.setattr(
        aitoolkit.telemetry,
        "set_meta",
        lambda **kwargs: meta_updates.append(kwargs),
    )
    Path(spec.save_root).mkdir(parents=True)
    scope = checkpoints.begin_run(spec.save_root, spec.expected_repo_name)
    planned = cfg["config"]["process"][0]["train"]["steps"]
    scope = checkpoints.set_planned_steps(spec.save_root, scope, planned)
    source_run_id = aitoolkit._timing_source_run_id(spec, scope)
    krea_runtime.emit_effective_runtime_record(
        cfg,
        "krea2",
        spec.config_path,
        krea_runtime.load_capability_manifest(),
        source_run_id=source_run_id,
        throughput_profile=profile,
        current_dataset_size=18,
    )
    config_before = hashlib.sha256(Path(spec.config_path).read_bytes()).hexdigest()

    checkpoint_path = Path(spec.save_root) / (
        f"{spec.expected_repo_name}_000000200.safetensors"
    )
    terminal_path = Path(spec.save_root) / f"{spec.expected_repo_name}.safetensors"
    env_marker = tmp_path / "python-bytecode-env.txt"
    fake_script = f'''import json, os, struct, time\nfrom pathlib import Path\ndef write(path, step):\n    metadata = {{"training_info": json.dumps({{"step": step, "epoch": 1}})}}\n    header = json.dumps({{"__metadata__": metadata, "weight": {{"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}}}).encode()\n    Path(path).write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.0))\nPath({str(env_marker)!r}).write_text(os.environ.get("PYTHONDONTWRITEBYTECODE", ""))\ntime.sleep(0.08)\nwrite({str(checkpoint_path)!r}, 200)\ntime.sleep(0.15)\nwrite({str(terminal_path)!r}, {planned})\nprint("{planned}/{planned} loss=0.1", flush=True)\nprint("Saved checkpoint to {str(terminal_path)}", flush=True)\n'''
    (toolkit_dir / "run.py").write_text(fake_script, encoding="utf-8")
    monkeypatch.setattr(aitoolkit, "_AI_TOOLKIT_DIR", str(toolkit_dir))
    monkeypatch.setattr(aitoolkit, "_POLL_SECONDS", 0.01)
    monkeypatch.setattr(
        krea_runtime,
        "verify_selected_runtime",
        lambda *_args, **_kwargs: str(toolkit_dir),
    )
    calls = []
    original_emit = adaptive_timing.emit_first_checkpoint_observation

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return original_emit(*args, **kwargs)

    monkeypatch.setattr(
        adaptive_timing, "emit_first_checkpoint_observation", capture
    )

    class Deadline:
        def remaining(self):
            return 10_000.0

    result = aitoolkit._run_toolkit(
        spec.config_path,
        Deadline(),
        spec,
        scope,
        throughput_profile=profile,
        active_planned_steps=planned,
        future_target_steps=recipe.size_target_steps("krea2", 18, planned),
        total_budget_s=2700.0,
        timing_record_required=True,
        timing_bundle=krea_runtime.LEADER_BUNDLE,
        toolkit_dir=str(toolkit_dir),
    )

    record = json.loads(
        Path(spec.config_path + ".effective-runtime.json").read_text(
            encoding="utf-8"
        )
    )
    assert result is False
    assert len(calls) == 1
    assert record["first_checkpoint_observation"]["checkpoint_step"] == 200
    assert record["first_checkpoint_observation"]["active_planned_steps"] == planned
    assert record["first_checkpoint_observation"]["active_plan_mutable"] is False
    completion = record["training_completion_observation"]
    assert record["lifecycle"] == "terminal"
    assert completion["training_elapsed_seconds"] > 0
    assert completion["returncode"] == 0
    assert completion["stopped_by_deadline"] is False
    assert completion["natural_completion"] is True
    assert completion["artifact_path"] == str(terminal_path.resolve())
    assert completion["artifact_name"] == terminal_path.name
    assert completion["artifact_size_bytes"] == terminal_path.stat().st_size
    assert completion["artifact_sha256"] == hashlib.sha256(
        terminal_path.read_bytes()
    ).hexdigest()
    assert completion["artifact_loadable"] is True
    assert completion["artifact_checkpoint_step"] == planned
    assert completion["completed_steps"] == planned
    assert completion["scope_attempt_nonce"] == scope["attempt_nonce"]
    terminal_stat = terminal_path.stat()
    assert completion["artifact_file_identity"] == {
        "device": terminal_stat.st_dev,
        "inode": terminal_stat.st_ino,
        "size": terminal_stat.st_size,
        "mtime_ns": terminal_stat.st_mtime_ns,
        "ctime_ns": terminal_stat.st_ctime_ns,
    }
    assert env_marker.read_text(encoding="utf-8") == "1"
    assert meta_updates[-1] == {
        "krea_effective_runtime_record_sha256": record["record_sha256"]
    }
    assert hashlib.sha256(Path(spec.config_path).read_bytes()).hexdigest() == config_before


def test_clean_log_with_phantom_terminal_artifact_aborts(tmp_path, monkeypatch):
    spec = _spec()
    _localize_spec(monkeypatch, tmp_path, spec)
    toolkit_dir = tmp_path / "phantom-toolkit"
    toolkit_dir.mkdir()
    _activate(monkeypatch, toolkit_dir, krea_runtime.LEADER_BUNDLE)
    profile = _timing_profile(krea_runtime.LEADER_BUNDLE, startup_seconds=0.0)
    cfg = config.build_config(spec, 18, 0.75, throughput_profile=profile)
    Path(spec.config_path).parent.mkdir(parents=True)
    config.write_config(cfg, spec.config_path)
    Path(spec.save_root).mkdir(parents=True)
    scope = checkpoints.begin_run(spec.save_root, spec.expected_repo_name)
    planned = cfg["config"]["process"][0]["train"]["steps"]
    scope = checkpoints.set_planned_steps(spec.save_root, scope, planned)
    krea_runtime.emit_effective_runtime_record(
        cfg,
        "krea2",
        spec.config_path,
        krea_runtime.load_capability_manifest(),
        source_run_id=aitoolkit._timing_source_run_id(spec, scope),
        throughput_profile=profile,
        current_dataset_size=18,
    )
    _write_training_safetensor(
        Path(spec.save_root)
        / f"{spec.expected_repo_name}_000000200.safetensors",
        step=200,
    )
    (toolkit_dir / "run.py").write_text(
        f'print("{planned - 1}/{planned} loss=0.1")\n'
        'print("Saved checkpoint to /phantom/final.safetensors")\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(aitoolkit, "_POLL_SECONDS", 0.01)
    monkeypatch.setattr(
        krea_runtime,
        "verify_selected_runtime",
        lambda *_args, **_kwargs: str(toolkit_dir),
    )

    class Deadline:
        def remaining(self):
            return 10_000.0

    with pytest.raises(
        adaptive_timing.TimingProfileError,
        match="did not produce a current-run terminal artifact",
    ):
        aitoolkit._run_toolkit(
            spec.config_path,
            Deadline(),
            spec,
            scope,
            throughput_profile=profile,
            active_planned_steps=planned,
            future_target_steps=planned,
            total_budget_s=2700.0,
            timing_record_required=True,
            timing_bundle=krea_runtime.LEADER_BUNDLE,
            toolkit_dir=str(toolkit_dir),
        )

    record = json.loads(
        Path(spec.config_path + ".effective-runtime.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["lifecycle"] == "first_checkpoint"
    assert record["first_checkpoint_observation"] is not None
    assert record["training_completion_observation"] is None


def test_experimental_profile_requires_post_run_checkpoint_observation(
    tmp_path, monkeypatch
):
    spec = _spec()
    _localize_spec(monkeypatch, tmp_path, spec)
    _activate(monkeypatch, tmp_path, krea_runtime.LEADER_BUNDLE)
    profile = _timing_profile(krea_runtime.LEADER_BUNDLE, startup_seconds=0.0)
    cfg = config.build_config(spec, 18, 0.75, throughput_profile=profile)
    Path(spec.config_path).parent.mkdir(parents=True)
    config.write_config(cfg, spec.config_path)
    krea_runtime.emit_effective_runtime_record(
        cfg,
        "krea2",
        spec.config_path,
        krea_runtime.load_capability_manifest(),
        source_run_id=SOURCE_RUN_ID,
        throughput_profile=profile,
        current_dataset_size=18,
    )
    Path(spec.save_root).mkdir(parents=True)
    scope = checkpoints.begin_run(spec.save_root, spec.expected_repo_name)
    planned = cfg["config"]["process"][0]["train"]["steps"]
    scope = checkpoints.set_planned_steps(spec.save_root, scope, planned)
    toolkit_dir = tmp_path / "no-checkpoint-toolkit"
    toolkit_dir.mkdir()
    (toolkit_dir / "run.py").write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(aitoolkit, "_POLL_SECONDS", 0.01)
    monkeypatch.setattr(
        krea_runtime,
        "verify_selected_runtime",
        lambda *_args, **_kwargs: str(toolkit_dir),
    )

    class Deadline:
        def remaining(self):
            return 10_000.0

    with pytest.raises(
        adaptive_timing.TimingProfileError,
        match="required first-checkpoint timing observation",
    ):
        aitoolkit._run_toolkit(
            spec.config_path,
            Deadline(),
            spec,
            scope,
            throughput_profile=profile,
            active_planned_steps=planned,
            future_target_steps=recipe.size_target_steps(
                "krea2", 18, planned
            ),
            total_budget_s=2700.0,
            timing_record_required=True,
            timing_bundle=krea_runtime.LEADER_BUNDLE,
            toolkit_dir=str(toolkit_dir),
        )


def test_bootstrap_probe_requires_post_run_checkpoint_observation(
    tmp_path, monkeypatch
):
    spec = _spec()
    _localize_spec(monkeypatch, tmp_path, spec)
    monkeypatch.setenv(krea_runtime.BUNDLE_ENV, krea_runtime.LEADER_BUNDLE)
    Path(spec.config_path).parent.mkdir(parents=True)
    Path(spec.config_path).write_text("{}\n", encoding="utf-8")
    Path(spec.save_root).mkdir(parents=True)
    scope = checkpoints.begin_run(spec.save_root, spec.expected_repo_name)
    scope = checkpoints.set_planned_steps(spec.save_root, scope, 1000)
    toolkit_dir = tmp_path / "empty-bootstrap-toolkit"
    toolkit_dir.mkdir()
    (toolkit_dir / "run.py").write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(aitoolkit, "_POLL_SECONDS", 0.01)
    monkeypatch.setattr(
        krea_runtime,
        "verify_selected_runtime",
        lambda *_args, **_kwargs: str(toolkit_dir),
    )

    class Deadline:
        def remaining(self):
            return 10_000.0

    with pytest.raises(
        adaptive_timing.TimingProfileError,
        match="required first-checkpoint timing observation",
    ):
        aitoolkit._run_toolkit(
            spec.config_path,
            Deadline(),
            spec,
            scope,
            active_planned_steps=1000,
            timing_record_required=True,
            timing_probe=True,
            timing_bundle=krea_runtime.LEADER_BUNDLE,
            toolkit_dir=str(toolkit_dir),
        )


def test_incumbent_timing_observation_persistence_is_best_effort(monkeypatch):
    observation = object()
    events = []
    monkeypatch.setattr(
        krea_runtime,
        "persist_first_checkpoint_observation",
        lambda *_args: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(
        aitoolkit.telemetry,
        "event",
        lambda name, **fields: events.append((name, fields)),
    )

    assert aitoolkit._persist_first_checkpoint_observation(
        "/missing/config.yaml", observation, required=False
    ) is False
    assert events == [
        (
            "krea_first_checkpoint_observation_persist_failed",
            {"error_type": "OSError"},
        )
    ]

    with pytest.raises(OSError, match="disk full"):
        aitoolkit._persist_first_checkpoint_observation(
            "/missing/config.yaml", observation, required=True
        )
