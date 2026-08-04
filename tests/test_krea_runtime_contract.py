from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import pytest

from forge import adaptive_timing, config, krea_runtime, recipe
from forge.tasks import aitoolkit, checkpoints
from forge.data.schema import ImageSpec


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
    capabilities = {name: True for name in krea_runtime.REQUIRED_CAPABILITIES}
    if false_capability is not None:
        capabilities[false_capability] = False
    value = {
        "schema": 1,
        "runtime_contract_id": krea_runtime.RUNTIME_CONTRACT_ID,
        "base_commit": krea_runtime.PINNED_BASE_COMMIT,
        "capabilities": capabilities,
        "evidence": {
            name: f"tests/{name}.py"
            for name in krea_runtime.REQUIRED_CAPABILITIES
        },
    }
    path = tmp_path / "capabilities.json"
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
    return manifest_path.with_name("runtime-identity.json")


def _activate(monkeypatch, tmp_path: Path, bundle: str) -> Path:
    path = _manifest(tmp_path)
    monkeypatch.setenv(krea_runtime.BUNDLE_ENV, bundle)
    monkeypatch.setenv(krea_runtime.CAPABILITY_MANIFEST_ENV, str(path))
    monkeypatch.setenv(
        krea_runtime.RUNTIME_IDENTITY_ENV, str(_identity_path(path))
    )
    return path


def _timing_profile(bundle: str, *, startup_seconds: float = 120.0):
    bundle_sha = krea_runtime.bundle_contract_sha256(bundle)
    document = adaptive_timing.seal_profile_document(
        {
            "schema": adaptive_timing.PROFILE_SCHEMA,
            "kind": adaptive_timing.PROFILE_KIND,
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
                "source_run_id": "week6-bootstrap",
                "source_record_sha256": "b" * 64,
                "runtime_commit": krea_runtime.OWNED_RUNTIME_COMMIT,
                "measured_at_utc": "2026-08-04T12:00:00Z",
                "accelerator_identity": "NVIDIA H100 PCIe|81559-MiB",
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
    monkeypatch.setenv(
        krea_runtime.CAPABILITY_MANIFEST_ENV, "/does/not/exist/capabilities.json"
    )
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


def test_stable_bundle_ids_carry_honest_source_derived_claims():
    rank1 = krea_runtime.bundle_claim_document(krea_runtime.LEADER_BUNDLE)
    rank3 = krea_runtime.bundle_claim_document(krea_runtime.MAE_BUNDLE)

    assert rank1["source_config_sha256"] == krea_runtime.PUBLIC_RANK1_CONFIG_SHA256
    assert rank3["source_config_sha256"] == krea_runtime.PUBLIC_RANK3_CONFIG_SHA256
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
    monkeypatch.setenv(krea_runtime.CAPABILITY_MANIFEST_ENV, str(path))
    monkeypatch.setenv(
        krea_runtime.RUNTIME_IDENTITY_ENV, str(_identity_path(path))
    )

    with pytest.raises(krea_runtime.KreaRuntimeContractError, match=missing):
        config.build_config(_spec(), num_images=18, hours_to_complete=0.75)


def test_capability_manifest_rejects_unknown_claim(tmp_path, monkeypatch):
    path = _manifest(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["capabilities"]["magic_yaml_key"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setenv(krea_runtime.CAPABILITY_MANIFEST_ENV, str(path))

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
        krea_runtime.CAPABILITY_MANIFEST_ENV: str(path),
        krea_runtime.RUNTIME_IDENTITY_ENV: str(_identity_path(path)),
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
    assert record["capability_manifest_file_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert record["capability_manifest_semantic_sha256"] == (
        krea_runtime._canonical_sha256(manifest)
    )
    # Task paths, trigger strings, and credentials do not enter this sidecar.
    assert "AetherTest" not in json.dumps(record)


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
            timing_probe=True,
            current_dataset_size=18,
            current_accelerator_identity="NVIDIA H100 PCIe|81559-MiB",
        )


def test_measured_profile_is_bound_to_bundle_in_effective_record(
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
        throughput_profile=profile,
        current_dataset_size=18,
    )

    assert record["timing"] == {
        "mode": "measured_profile",
        "profile_sha256": profile.profile_sha256,
        "runtime_commit": profile.runtime_commit,
        "measured_dataset_size": 18,
        "current_dataset_size": 18,
        "dataset_regime": adaptive_timing.dataset_regime(18),
        "accelerator_identity": profile.accelerator_identity,
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
            throughput_profile=foreign_profile,
            current_dataset_size=18,
        )


def test_experimental_effective_record_rejects_unlabeled_static_timing(
    tmp_path, monkeypatch
):
    _activate(monkeypatch, tmp_path, krea_runtime.LEADER_BUNDLE)
    cfg = config.build_config(_spec(), 18, 0.75)
    config_path = tmp_path / "static.yaml"
    config.write_config(cfg, str(config_path))

    with pytest.raises(krea_runtime.KreaRuntimeContractError, match="measured timing"):
        krea_runtime.emit_effective_runtime_record(
            cfg,
            "krea2",
            str(config_path),
            krea_runtime.load_capability_manifest(),
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
            throughput_profile=profile,
            current_dataset_size=18,
        )


def test_runtime_identity_must_match_exact_owned_commit(tmp_path, monkeypatch):
    path = _manifest(tmp_path)
    identity_path = _identity_path(path)
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["runtime_commit"] = "f" * 40
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    monkeypatch.setenv(krea_runtime.CAPABILITY_MANIFEST_ENV, str(path))
    monkeypatch.setenv(krea_runtime.RUNTIME_IDENTITY_ENV, str(identity_path))

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


def test_first_durable_checkpoint_is_observed_once_and_persisted(
    tmp_path, monkeypatch
):
    spec = _spec()
    _localize_spec(monkeypatch, tmp_path, spec)
    _activate(monkeypatch, tmp_path, krea_runtime.LEADER_BUNDLE)
    profile = _timing_profile(
        krea_runtime.LEADER_BUNDLE, startup_seconds=0.0
    )
    cfg = config.build_config(
        spec, 18, 0.75, throughput_profile=profile
    )
    Path(spec.config_path).parent.mkdir(parents=True)
    config.write_config(cfg, spec.config_path)
    krea_runtime.emit_effective_runtime_record(
        cfg,
        "krea2",
        spec.config_path,
        krea_runtime.load_capability_manifest(),
        throughput_profile=profile,
        current_dataset_size=18,
    )
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
    config_before = hashlib.sha256(Path(spec.config_path).read_bytes()).hexdigest()

    toolkit_dir = tmp_path / "fake-toolkit"
    toolkit_dir.mkdir()
    checkpoint_path = Path(spec.save_root) / (
        f"{spec.expected_repo_name}_000000200.safetensors"
    )
    fake_script = f'''import json, struct, time\nfrom pathlib import Path\ntime.sleep(0.08)\nheader = json.dumps({{"weight": {{"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}}}).encode()\nPath({str(checkpoint_path)!r}).write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.0))\ntime.sleep(0.15)\n'''
    (toolkit_dir / "run.py").write_text(fake_script, encoding="utf-8")
    monkeypatch.setattr(aitoolkit, "_AI_TOOLKIT_DIR", str(toolkit_dir))
    monkeypatch.setattr(aitoolkit, "_POLL_SECONDS", 0.01)
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
    assert meta_updates[-1] == {
        "krea_effective_runtime_record_sha256": record["record_sha256"]
    }
    assert hashlib.sha256(Path(spec.config_path).read_bytes()).hexdigest() == config_before


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
            toolkit_dir=str(toolkit_dir),
        )


def test_bootstrap_probe_requires_post_run_checkpoint_observation(
    tmp_path, monkeypatch
):
    spec = _spec()
    _localize_spec(monkeypatch, tmp_path, spec)
    Path(spec.config_path).parent.mkdir(parents=True)
    Path(spec.config_path).write_text("{}\n", encoding="utf-8")
    Path(spec.save_root).mkdir(parents=True)
    scope = checkpoints.begin_run(spec.save_root, spec.expected_repo_name)
    scope = checkpoints.set_planned_steps(spec.save_root, scope, 1000)
    toolkit_dir = tmp_path / "empty-bootstrap-toolkit"
    toolkit_dir.mkdir()
    (toolkit_dir / "run.py").write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(aitoolkit, "_POLL_SECONDS", 0.01)

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
