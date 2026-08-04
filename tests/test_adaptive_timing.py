from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from forge import adaptive_timing, krea_runtime, recipe


BUNDLE_SHA = "a" * 64
SOURCE_SHA = "b" * 64
RUNTIME_COMMIT = "c" * 40
DATASET_SIZE = 24
DATASET_REGIME = "small-11-24"
ACCELERATOR_IDENTITY = "NVIDIA H100 PCIe|81559-MiB"
SOURCE_RUN_ID = "week6-timing-probe:" + "a" * 32


def _nvidia_runner(identity: str = ACCELERATOR_IDENTITY):
    name, memory = identity.split("|", 1)
    memory = memory.removesuffix("-MiB")

    def run(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=f"{name}, {memory}\n",
            stderr="",
        )

    return run


def _write_source_record(
    tmp_path, *, bundle=krea_runtime.LEADER_BUNDLE, **overrides
):
    record = {
        "schema": 3,
        "runtime_contract_id": krea_runtime.RUNTIME_CONTRACT_ID,
        "source_run_id": SOURCE_RUN_ID,
        "model_type": "krea2",
        "runtime_repository": krea_runtime.OWNED_RUNTIME_REPOSITORY,
        "runtime_commit": krea_runtime.OWNED_RUNTIME_COMMIT,
        "bundle": bundle,
        "bundle_claim": krea_runtime.bundle_claim_document(bundle),
        "bundle_contract_sha256": krea_runtime.bundle_contract_sha256(bundle),
        "generated_config_sha256": "d" * 64,
        "capability_manifest_file_sha256": "e" * 64,
        "capability_manifest_semantic_sha256": "f" * 64,
        "capabilities": sorted(krea_runtime.REQUIRED_CAPABILITIES),
        "runtime_manifest_capability_aliases": krea_runtime.bundle_contract_document(
            bundle
        )["runtime_manifest_capability_aliases"],
        "timing": {
            "mode": "bootstrap_probe_unmeasured",
            "profile_sha256": None,
            "runtime_commit": krea_runtime.OWNED_RUNTIME_COMMIT,
            "measured_dataset_size": None,
            "current_dataset_size": 24,
            "dataset_regime": "small-11-24",
            "accelerator_identity": ACCELERATOR_IDENTITY,
        },
        "effective": {
            "planned_steps": 1000,
            "normalized_config_projection": krea_runtime.bundle_contract_document(
                bundle
            )["normalized_config_projection"],
        },
        "first_checkpoint_observation": {
            "bundle_id": bundle,
            "timing_profile_sha256": None,
            "observation_mode": "bootstrap_raw_first_checkpoint",
            "checkpoint_step": 200,
            "elapsed_since_launch_s": 300.0,
            "active_planned_steps": 1000,
            "active_plan_mutable": False,
            "active_plan_action": "observe_only_fixed_subprocess",
        },
        "training_completion_observation": {
            "reported_last_step": 1000,
            "training_elapsed_seconds": 1500.0,
            "returncode": 0,
            "stopped_by_deadline": False,
            "natural_completion": True,
        },
    }
    for key, value in overrides.items():
        if "." in key:
            section, field = key.split(".", 1)
            record[section][field] = value
        else:
            record[key] = value
    record["record_sha256"] = adaptive_timing._runtime_record_semantic_sha256(
        record
    )
    path = tmp_path / f"{bundle}.probe.effective-runtime.json"
    path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def _profile_document(**overrides):
    value = {
        "schema": adaptive_timing.PROFILE_SCHEMA,
        "kind": adaptive_timing.PROFILE_KIND,
        "bundle_id": "leader-v1",
        "bundle_sha256": BUNDLE_SHA,
        "model_type": "krea2",
        "measured_dataset_size": DATASET_SIZE,
        "dataset_regime": DATASET_REGIME,
        "seconds_per_step": 1.3,
        "startup_seconds": 120.0,
        "measurement": {
            "completed_steps": 824,
            "training_elapsed_seconds": 1195.0,
            "first_checkpoint_step": 165,
            "first_checkpoint_elapsed_seconds": 345.0,
        },
        "provenance": {
            "source_run_id": SOURCE_RUN_ID,
            "source_record_sha256": SOURCE_SHA,
            "runtime_commit": RUNTIME_COMMIT,
            "measured_at_utc": "2026-08-03T16:30:00Z",
            "accelerator_identity": ACCELERATOR_IDENTITY,
        },
    }
    value.update(overrides)
    return adaptive_timing._seal_profile_document(value)


def _write_profile(tmp_path, **overrides):
    path = tmp_path / "leader-v1-throughput.json"
    path.write_text(
        json.dumps(_profile_document(**overrides), sort_keys=True),
        encoding="utf-8",
    )
    return path


def _load(path):
    return adaptive_timing.load_profile(
        str(path),
        expected_bundle_id="leader-v1",
        expected_bundle_sha256=BUNDLE_SHA,
        expected_model_type="krea2",
        current_dataset_size=DATASET_SIZE,
        expected_dataset_regime=DATASET_REGIME,
        expected_accelerator_identity=ACCELERATOR_IDENTITY,
    )


def test_no_profile_preserves_incumbent_recipe_outputs(monkeypatch):
    monkeypatch.delenv(adaptive_timing.PROFILE_ENV, raising=False)

    profile = adaptive_timing.load_bundle_profile(
        bundle_id="incumbent-v1",
        bundle_sha256="d" * 64,
        model_type="krea2",
        current_dataset_size=DATASET_SIZE,
        dataset_regime=DATASET_REGIME,
        required=False,
    )

    assert profile is None
    assert recipe.size_scaled_steps("krea2", 24, 0.75, 2000) == 824
    assert recipe.size_scaled_steps("krea2", 24, 1.0, 2000) == 1172


def test_experimental_bundle_requires_profile(monkeypatch):
    monkeypatch.delenv(adaptive_timing.PROFILE_ENV, raising=False)

    with pytest.raises(
        adaptive_timing.TimingProfileError, match="required for bundle"
    ):
        adaptive_timing.load_bundle_profile(
            bundle_id="leader-v1",
            bundle_sha256=BUNDLE_SHA,
            model_type="krea2",
            current_dataset_size=DATASET_SIZE,
            dataset_regime=DATASET_REGIME,
            required=True,
        )


def test_profile_producer_hashes_and_crosschecks_raw_completed_run(tmp_path):
    source = _write_source_record(tmp_path)

    document = adaptive_timing.produce_profile_document(
        str(source),
        source_run_id=SOURCE_RUN_ID,
        bundle_id=krea_runtime.LEADER_BUNDLE,
        model_type="krea2",
        measured_dataset_size=24,
        measured_at_utc="2026-08-04T18:00:00Z",
        runner=_nvidia_runner(),
    )
    profile_path = tmp_path / "produced-profile.json"
    profile_path.write_text(json.dumps(document), encoding="utf-8")
    profile = adaptive_timing.load_profile(
        str(profile_path),
        expected_bundle_id=krea_runtime.LEADER_BUNDLE,
        expected_bundle_sha256=krea_runtime.bundle_contract_sha256(
            krea_runtime.LEADER_BUNDLE
        ),
        expected_model_type="krea2",
        current_dataset_size=24,
        expected_dataset_regime="small-11-24",
        expected_accelerator_identity=ACCELERATOR_IDENTITY,
    )

    assert profile.source_record_sha256 == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert profile.source_run_id == SOURCE_RUN_ID
    assert profile.completed_steps == 1000
    assert profile.first_checkpoint_step == 200
    assert profile.seconds_per_step == pytest.approx(1.5)
    assert profile.startup_seconds == 0.0


def test_profile_producer_accepts_mae_bundle_capability_subset(tmp_path):
    source = _write_source_record(
        tmp_path, bundle=krea_runtime.MAE_BUNDLE
    )

    document = adaptive_timing.produce_profile_document(
        str(source),
        source_run_id=SOURCE_RUN_ID,
        bundle_id=krea_runtime.MAE_BUNDLE,
        model_type="krea2",
        measured_dataset_size=24,
        measured_at_utc="2026-08-04T18:00:00Z",
        runner=_nvidia_runner(),
    )

    assert document["bundle_id"] == krea_runtime.MAE_BUNDLE
    assert document["provenance"]["runtime_commit"] == (
        krea_runtime.OWNED_RUNTIME_COMMIT
    )


@pytest.mark.parametrize(
    ("overrides", "kwargs", "message"),
    [
        (
            {"source_run_id": "another-run:" + "b" * 32},
            {},
            "run id mismatch",
        ),
        (
            {"timing.current_dataset_size": 25},
            {},
            "timing identity mismatch",
        ),
        (
            {"first_checkpoint_observation.checkpoint_step": 1001},
            {},
            "first-checkpoint observation mismatch",
        ),
        (
            {"training_completion_observation.natural_completion": False},
            {},
            "did not complete naturally",
        ),
        (
            {"timing.accelerator_identity": "NVIDIA H100 SXM|81559-MiB"},
            {},
            "timing identity mismatch",
        ),
    ],
)
def test_profile_producer_rejects_cross_record_provenance(
    tmp_path, overrides, kwargs, message
):
    source = _write_source_record(tmp_path, **overrides)

    with pytest.raises(adaptive_timing.TimingProfileError, match=message):
        adaptive_timing.produce_profile_document(
            str(source),
            source_run_id=SOURCE_RUN_ID,
            bundle_id=krea_runtime.LEADER_BUNDLE,
            model_type="krea2",
            measured_dataset_size=24,
            measured_at_utc="2026-08-04T18:00:00Z",
            runner=_nvidia_runner(),
            **kwargs,
        )


def test_accelerator_environment_cannot_override_live_observation(monkeypatch):
    monkeypatch.setenv(
        "FORGE_KREA_ACCELERATOR_IDENTITY", "spoofed-gpu|1-MiB"
    )

    assert adaptive_timing.current_accelerator_identity(
        runner=_nvidia_runner()
    ) == ACCELERATOR_IDENTITY


def test_profile_producer_rejects_source_record_symlink(tmp_path):
    source = _write_source_record(tmp_path)
    link = tmp_path / "source-record-link.json"
    link.symlink_to(source)

    with pytest.raises(
        adaptive_timing.TimingProfileError,
        match="source runtime record unavailable",
    ):
        adaptive_timing.produce_profile_document(
            str(link),
            source_run_id=SOURCE_RUN_ID,
            bundle_id=krea_runtime.LEADER_BUNDLE,
            model_type="krea2",
            measured_dataset_size=24,
            measured_at_utc="2026-08-04T18:00:00Z",
            runner=_nvidia_runner(),
        )


def test_valid_profile_is_bound_and_changes_only_explicit_recipe_call(tmp_path):
    profile = _load(_write_profile(tmp_path))

    incumbent = recipe.size_scaled_steps("krea2", 24, 0.75, 2000)
    measured = recipe.size_scaled_steps(
        "krea2",
        24,
        0.75,
        2000,
        throughput_profile=profile,
    )

    assert incumbent == 824
    assert measured == 1200
    assert profile.source_record_sha256 == SOURCE_SHA
    assert profile.runtime_commit == RUNTIME_COMMIT


def test_profile_reuse_is_regime_bound_not_exact_pair_count(tmp_path):
    profile = adaptive_timing.load_profile(
        str(_write_profile(tmp_path)),
        expected_bundle_id="leader-v1",
        expected_bundle_sha256=BUNDLE_SHA,
        expected_model_type="krea2",
        current_dataset_size=18,
        expected_dataset_regime=adaptive_timing.dataset_regime(18),
        expected_accelerator_identity=ACCELERATOR_IDENTITY,
    )

    assert profile.measured_dataset_size == 24
    assert profile.dataset_regime == adaptive_timing.dataset_regime(18)


def test_explicit_recipe_profile_fails_closed_on_wrong_contract(tmp_path):
    profile = _load(_write_profile(tmp_path))

    with pytest.raises(adaptive_timing.TimingProfileError, match="model type"):
        recipe.size_scaled_steps(
            "ideogram4",
            24,
            0.75,
            2000,
            throughput_profile=profile,
        )
    with pytest.raises(adaptive_timing.TimingProfileError, match="invalid"):
        recipe.size_scaled_steps(
            "krea2",
            24,
            0.75,
            2000,
            throughput_profile=object(),
        )


@pytest.mark.parametrize(
    (
        "expected_bundle",
        "expected_sha",
        "expected_model",
        "expected_size",
        "expected_regime",
        "expected_accelerator",
        "message",
    ),
    [
        ("mae-g3-v1", BUNDLE_SHA, "krea2", DATASET_SIZE, DATASET_REGIME, ACCELERATOR_IDENTITY, "bundle id mismatch"),
        ("leader-v1", "d" * 64, "krea2", DATASET_SIZE, DATASET_REGIME, ACCELERATOR_IDENTITY, "bundle digest mismatch"),
        ("leader-v1", BUNDLE_SHA, "ideogram4", DATASET_SIZE, DATASET_REGIME, ACCELERATOR_IDENTITY, "model type mismatch"),
        ("leader-v1", BUNDLE_SHA, "krea2", 25, "medium-25-50", ACCELERATOR_IDENTITY, "dataset regime mismatch"),
        ("leader-v1", BUNDLE_SHA, "krea2", DATASET_SIZE, "medium-25-50", ACCELERATOR_IDENTITY, "current dataset regime is inconsistent"),
        ("leader-v1", BUNDLE_SHA, "krea2", DATASET_SIZE, DATASET_REGIME, "NVIDIA H100 SXM|81559-MiB", "accelerator identity mismatch"),
    ],
)
def test_profile_rejects_cross_bundle_or_model_reuse(
    tmp_path,
    expected_bundle,
    expected_sha,
    expected_model,
    expected_size,
    expected_regime,
    expected_accelerator,
    message,
):
    path = _write_profile(tmp_path)

    with pytest.raises(adaptive_timing.TimingProfileError, match=message):
        adaptive_timing.load_profile(
            str(path),
            expected_bundle_id=expected_bundle,
            expected_bundle_sha256=expected_sha,
            expected_model_type=expected_model,
            current_dataset_size=expected_size,
            expected_dataset_regime=expected_regime,
            expected_accelerator_identity=expected_accelerator,
        )


def test_profile_rejects_tampering_and_unknown_fields(tmp_path):
    path = _write_profile(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["seconds_per_step"] = 0.7
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(adaptive_timing.TimingProfileError, match="digest mismatch"):
        _load(path)

    value = _profile_document()
    value["provenance"]["unreviewed_note"] = "not in schema"
    path.write_text(json.dumps(adaptive_timing._seal_profile_document(value)))
    with pytest.raises(adaptive_timing.TimingProfileError, match="fields differ"):
        _load(path)


def test_profile_rejects_a_self_hashed_but_unmeasured_rate(tmp_path):
    path = _write_profile(tmp_path, seconds_per_step=0.7)

    with pytest.raises(
        adaptive_timing.TimingProfileError, match="recorded measurement"
    ):
        _load(path)


def test_profile_rejects_boolean_schema_version(tmp_path):
    path = _write_profile(tmp_path, schema=True)

    with pytest.raises(
        adaptive_timing.TimingProfileError, match="unsupported.*contract"
    ):
        _load(path)


def test_profile_rejects_symlink(tmp_path):
    source = _write_profile(tmp_path)
    link = tmp_path / "profile-link.json"
    link.symlink_to(source)

    with pytest.raises(adaptive_timing.TimingProfileError, match="regular file"):
        _load(link)


def test_first_checkpoint_correction_is_future_only_and_emits_telemetry(tmp_path):
    profile = _load(_write_profile(tmp_path))
    events = []

    observation = adaptive_timing.emit_first_checkpoint_observation(
        profile,
        event_sink=lambda name, **fields: events.append((name, fields)),
        active_planned_steps=824,
        future_target_steps=1200,
        checkpoint_step=165,
        elapsed_since_launch_s=285.0,
        total_budget_s=2700.0,
        export_reserve_s=180.0,
        safety=0.85,
    )

    assert observation.observed_seconds_per_step == pytest.approx(1.0)
    assert observation.correction == "faster"
    assert observation.active_planned_steps == 824
    assert observation.active_plan_mutable is False
    assert observation.active_plan_action == "observe_only_fixed_subprocess"
    assert observation.active_plan_exceeds_observed_budget is False
    assert observation.future_recommended_steps == 1200
    assert observation.future_step_delta == 376
    assert events[0][0] == adaptive_timing.FIRST_CHECKPOINT_EVENT
    assert events[0][1]["active_plan_mutable"] is False
    assert events[0][1]["future_recommended_steps"] == 1200


def test_slower_first_checkpoint_recommends_less_only_for_future_run(tmp_path):
    profile = _load(_write_profile(tmp_path))

    observation = adaptive_timing.observe_first_checkpoint(
        profile,
        active_planned_steps=1200,
        future_target_steps=1200,
        checkpoint_step=200,
        elapsed_since_launch_s=520.0,
        total_budget_s=2700.0,
        export_reserve_s=180.0,
        safety=0.85,
    )

    assert observation.observed_seconds_per_step == pytest.approx(2.0)
    assert observation.correction == "slower"
    assert observation.active_planned_steps == 1200
    assert observation.active_plan_mutable is False
    assert observation.active_plan_exceeds_observed_budget is True
    assert observation.future_budget_cap_steps == 997
    assert observation.future_recommended_steps == 997


def test_first_checkpoint_rejects_impossible_observation(tmp_path):
    profile = _load(_write_profile(tmp_path))

    with pytest.raises(
        adaptive_timing.TimingProfileError, match="before profiled startup"
    ):
        adaptive_timing.observe_first_checkpoint(
            profile,
            active_planned_steps=824,
            future_target_steps=1200,
            checkpoint_step=165,
            elapsed_since_launch_s=100.0,
            total_budget_s=2700.0,
            export_reserve_s=180.0,
            safety=0.85,
        )


def test_bootstrap_first_checkpoint_emits_raw_persistable_evidence():
    events = []

    observation = adaptive_timing.emit_bootstrap_first_checkpoint_observation(
        bundle_id="leader-v1",
        checkpoint_step=200,
        elapsed_since_launch_s=341.25,
        active_planned_steps=1000,
        event_sink=lambda name, **fields: events.append((name, fields)),
    )

    assert observation.observation_mode == "bootstrap_raw_first_checkpoint"
    assert observation.telemetry_fields()["timing_profile_sha256"] is None
    assert events == [
        (
            adaptive_timing.FIRST_CHECKPOINT_EVENT,
            observation.telemetry_fields(),
        )
    ]
