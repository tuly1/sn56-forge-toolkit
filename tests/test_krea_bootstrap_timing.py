"""Adversarial contracts for the acyclic first-GPU Krea timing path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest

from ops.calibration import krea_budget
from ops.calibration import krea_execution_plan
from ops.calibration import krea_provenance
from ops.calibration import krea_timing_probe
from ops.calibration import run_krea_ladder


SHA = {letter: letter * 64 for letter in "0123456789abcdef"}


def _agent_actor(instance: str, role: str) -> dict[str, str]:
    return {
        "actor_class": "agent",
        "actor_id": f"codex-{instance}",
        "display_name": "Codex timing reviewer",
        "role": role,
        "review_instance_id": instance,
        "identity_assurance": (
            "self-declared-agent-identity-not-human-or-cryptographic-authentication"
        ),
    }


def _envelope(*, tool_sha=None):
    return krea_budget.seal_execution_envelope(
        equivalence_class="A-rank32-adamw8bit-mse-guidance2",
        network_rank=32,
        network_alpha=32,
        optimizer="adamw8bit",
        optimizer_config_sha256=SHA["1"],
        loss="mse",
        differential_guidance_enabled=True,
        guidance_scale=2.0,
        training_pair_count=18,
        training_dataset_shape_sha256=SHA["2"],
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        data_parallel_replicas=1,
        resolution_policy_sha256=SHA["3"],
        precision_policy_sha256=SHA["4"],
        cache_latents_to_disk=False,
        cache_text_embeddings=True,
        compile_enabled=False,
        jit_enabled=True,
        dataloader_workers=2,
        base_model_identity_sha256=SHA["5"],
        runtime_identity_sha256=SHA["6"],
        host_execution_identity_sha256=SHA["7"],
        execution_surface="staged_host_venv",
        execution_scope="discovery_only",
        venv_tree_manifest_sha256=SHA["8"],
        reference_container_image_sha256=SHA["f"],
        gpu_identity_sha256=SHA["9"],
        trainer_identity_sha256=SHA["a"],
        measurement_tool_sha256=tool_sha or SHA["b"],
    )


def _sample(capture, observation, start, duration_ns, units=1):
    return {
        "capture_id": capture,
        "observation_id": observation,
        "duration_s": duration_ns / 1_000_000_000,
        "units": units,
        "started_monotonic_ns": start,
        "ended_monotonic_ns": start + duration_ns,
    }


def _raw_evidence():
    envelope = _envelope()
    commands = []
    for index in range(3):
        started_unix_ns = 1_800_000_000_000_000_000 + index * 10_000_000_000
        commands.append(
            {
                "capture_id": f"capture-{index}",
                "argv": ["/usr/bin/probe", "--fixed"],
                "executable_path": "/usr/bin/probe",
                "executable_sha256": SHA["c"],
                "returncode": 0,
                "started_unix_ns": started_unix_ns,
                "ended_unix_ns": started_unix_ns + 9_000_000_000,
                "event_stream_sha256": f"{index + 1:x}" * 64,
            }
        )
    samples = {
        "startup": [
            _sample(f"capture-{i}", f"startup-{i}", 1000 + i * 100, 10_000_000_000)
            for i in range(3)
        ],
        "optimizer_update": [
            _sample("capture-0", "updates", 2000, 200_000_000_000, units=100)
        ],
        "checkpoint_save": [
            _sample("capture-0", "saves", 3000, 8_000_000_000, units=8)
        ],
        "finalization": [_sample("capture-0", "final", 4000, 30_000_000_000)],
        "upload": [_sample("capture-0", "upload", 5000, 20_000_000_000)],
    }
    raw = krea_budget.seal_timing_sample_manifest(
        execution_envelope=envelope,
        probe_contract_sha256=SHA["d"],
        measurement_tool_sha256=SHA["b"],
        command_captures=commands,
        samples=samples,
        seed_bindings=[{"role": "A", "seed": 42565431}],
    )
    margin = krea_budget.seal_margin_policy(
        reviewer_identity="Jordan Example",
        approved_at_utc="2026-07-28T01:00:00Z",
        frozen_before_capture=True,
        multiplicative_margin={name: 1.25 for name in samples},
        additive_margin_s={name: 0.5 for name in samples},
    )
    e2e = krea_budget.seal_end_to_end_validation(
        execution_envelope_sha256=envelope["execution_envelope_sha256"],
        probe_contract_sha256=SHA["d"],
        runs=[
            {
                "run_id": "heldout-1",
                "seed_role": "A",
                "seed": 42565431,
                "hard_budget_s": 2700.0,
                "outer_wall_clock_s": 2000.0,
                "natural_completion": True,
                "upload_ready": True,
                "failure_or_fallback_telemetry": False,
                "run_record_sha256": SHA["e"],
            }
        ],
    )
    return raw, margin, e2e


def test_profile_is_recomputed_from_receipt_clock_blocks_not_opaque_claims():
    raw, margin, e2e = _raw_evidence()
    profile = krea_budget.seal_throughput_profile_from_evidence(
        raw_sample_manifest=raw,
        margin_policy=margin,
        end_to_end_validation=e2e,
        framework_stop_boundary_s=225.0,
        framework_stop_boundary_source_sha256=SHA["0"],
    )

    assert profile["startup_sample_count"] == 3
    assert profile["update_sample_count"] == 100
    assert profile["save_sample_count"] == 8
    assert profile["startup_upper_bound_s"] == 13.0
    assert profile["update_upper_bound_s"] == 3.0
    assert profile["save_upper_bound_s"] == 1.75
    assert profile["finalization_reserve_s"] == 38.0
    assert profile["upload_reserve_s"] == 25.5
    assert profile["raw_sample_manifest_sha256"] == raw["raw_sample_manifest_sha256"]


def test_schema2_timing_approval_requires_fresh_agent_review(monkeypatch):
    prior = _agent_actor("fixture-review", "independent_reviewer")
    authorization = {
        "file_sha256": SHA["1"],
        "authorization_sha256": SHA["2"],
        "document": {
            "accountable_owner_identity": "Jordan Example",
            "authorized_at_utc": "2026-07-28T00:00:00Z",
            "fixture_admission_envelope": {"owner_ratification_sha256": SHA["3"]},
        },
    }
    resolved = {
        "fixture": {"governance": {"independent_agent_review": {"actor": prior}}},
        "host_execution_manifest": {"host_execution_identity_sha256": SHA["4"]},
        "discovery_execution_authorization": authorization,
    }
    plan = {"schema": 2, "probe_contract_sha256": SHA["5"]}
    monkeypatch.setattr(
        krea_execution_plan, "validate_timing_probe_plan", lambda _plan: resolved
    )
    monkeypatch.setattr(
        krea_execution_plan.krea_discovery_authorization,
        "validate_technical_actor",
        lambda _authorization, actor, **_kwargs: actor,
    )

    with pytest.raises(ValueError, match="fresh technical review actor"):
        krea_execution_plan.build_timing_probe_approval(
            plan,
            reviewer_identity=None,
            approved_at_utc="2026-07-28T00:01:00Z",
            technical_reviewer_actor=prior,
        )

    approval = krea_execution_plan.build_timing_probe_approval(
        plan,
        reviewer_identity=None,
        approved_at_utc="2026-07-28T00:01:00Z",
        technical_reviewer_actor=_agent_actor(
            "timing-review", "timing_probe_execution_reviewer"
        ),
    )
    assert (
        krea_execution_plan.validate_timing_probe_approval(approval, plan=plan)
        == approval
    )
    predates = json.loads(json.dumps(approval))
    predates["approved_at_utc"] = "2026-07-27T23:59:59Z"
    predates_body = {
        key: value for key, value in predates.items() if key != "approval_sha256"
    }
    predates["approval_sha256"] = krea_provenance.canonical_sha256(predates_body)
    with pytest.raises(ValueError, match="predates discovery authorization"):
        krea_execution_plan.validate_timing_probe_approval(predates, plan=plan)


def test_margin_policy_must_be_frozen_before_every_governed_capture():
    raw, _, e2e = _raw_evidence()
    late_margin = krea_budget.seal_margin_policy(
        reviewer_identity="Jordan Example",
        approved_at_utc="2030-01-01T00:00:00Z",
        frozen_before_capture=True,
        multiplicative_margin={name: 1.25 for name in raw["samples"]},
        additive_margin_s={name: 0.5 for name in raw["samples"]},
    )
    with pytest.raises(
        krea_budget.TimingEvidenceError,
        match="approved after capture began",
    ):
        krea_budget.seal_throughput_profile_from_evidence(
            raw_sample_manifest=raw,
            margin_policy=late_margin,
            end_to_end_validation=e2e,
            framework_stop_boundary_s=225.0,
            framework_stop_boundary_source_sha256=SHA["0"],
        )


def test_margin_policy_rejects_impossible_calendar_timestamp():
    with pytest.raises(krea_budget.TimingEvidenceError, match="real UTC timestamp"):
        krea_budget.seal_margin_policy(
            reviewer_identity="Jordan Example",
            approved_at_utc="2026-02-30T00:00:00Z",
            frozen_before_capture=True,
            multiplicative_margin={name: 1.25 for name in krea_budget._TIMING_METRICS},
            additive_margin_s={name: 0.5 for name in krea_budget._TIMING_METRICS},
        )


def test_raw_evidence_rejects_fabricated_counts_duration_and_tool_identity():
    raw, _, _ = _raw_evidence()
    undercount = json.loads(json.dumps(raw))
    undercount["samples"]["optimizer_update"][0]["units"] = 99
    undercount.pop("raw_sample_manifest_sha256")
    with pytest.raises(krea_budget.TimingEvidenceError, match="fewer than 100"):
        krea_budget.seal_timing_sample_manifest(
            execution_envelope=undercount["execution_envelope"],
            probe_contract_sha256=undercount["probe_contract_sha256"],
            measurement_tool_sha256=undercount["measurement_tool_sha256"],
            command_captures=undercount["command_captures"],
            samples=undercount["samples"],
            seed_bindings=undercount["seed_bindings"],
        )

    duration = json.loads(json.dumps(raw))
    duration["samples"]["optimizer_update"][0]["duration_s"] = 0.001
    duration.pop("raw_sample_manifest_sha256")
    with pytest.raises(krea_budget.TimingEvidenceError, match="monotonic"):
        krea_budget.seal_timing_sample_manifest(
            execution_envelope=duration["execution_envelope"],
            probe_contract_sha256=duration["probe_contract_sha256"],
            measurement_tool_sha256=duration["measurement_tool_sha256"],
            command_captures=duration["command_captures"],
            samples=duration["samples"],
            seed_bindings=duration["seed_bindings"],
        )

    with pytest.raises(krea_budget.TimingEvidenceError, match="producer"):
        krea_budget.seal_timing_sample_manifest(
            execution_envelope=raw["execution_envelope"],
            probe_contract_sha256=raw["probe_contract_sha256"],
            measurement_tool_sha256=SHA["f"],
            command_captures=raw["command_captures"],
            samples=raw["samples"],
            seed_bindings=raw["seed_bindings"],
        )


def test_marker_pairing_uses_parent_receipt_time_and_fails_unpaired():
    contract = SHA["d"]
    capture = "capture-1"
    begin = {
        "schema": 1,
        "kind": "forge-krea-timing-marker",
        "probe_contract_sha256": contract,
        "capture_id": capture,
        "observation_id": "updates",
        "metric": "optimizer_update",
        "state": "begin",
        "units": 100,
    }
    end = {**begin, "state": "end"}
    samples, _ = krea_timing_probe._pair_markers(
        [
            (10, krea_provenance.canonical_bytes(begin)),
            (2_000_000_010, krea_provenance.canonical_bytes(end)),
        ],
        contract_sha=contract,
        capture_id=capture,
    )
    assert samples["optimizer_update"][0]["duration_s"] == 2.0
    with pytest.raises(ValueError, match="never ended"):
        krea_timing_probe._pair_markers(
            [(10, krea_provenance.canonical_bytes(begin))],
            contract_sha=contract,
            capture_id=capture,
        )


def test_runtime_compute_identity_excludes_seed_but_binds_it_separately(monkeypatch):
    monkeypatch.setattr(run_krea_ladder.importlib.metadata, "distributions", lambda: [])
    monkeypatch.setattr(
        run_krea_ladder,
        "_run_text",
        lambda _command, cwd=None: "NVIDIA H100, GPU-1, 999.0",
    )
    monkeypatch.setenv("SEED", "1")
    monkeypatch.setenv("PYTHONHASHSEED", "1")
    first = run_krea_ladder._runtime_fingerprint()
    monkeypatch.setenv("SEED", "2")
    monkeypatch.setenv("PYTHONHASHSEED", "2")
    second = run_krea_ladder._runtime_fingerprint()
    assert first["sha256"] == second["sha256"]
    assert first["stochastic_controls_sha256"] != second["stochastic_controls_sha256"]
    assert first["stochastic_controls"] == {
        "seed_env": "1",
        "pythonhashseed_env": "1",
    }


def test_root_run_paths_are_confined_to_campaign_controls_and_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_root = tmp_path / "campaign"
    controls = campaign_root / "controls"
    controls.mkdir(parents=True)
    monkeypatch.setattr(run_krea_ladder, "_CAMPAIGN_ROOT", campaign_root)
    monkeypatch.setattr(run_krea_ladder, "_CONTROL_ROOT", controls)
    args = argparse.Namespace(
        campaign_dir=campaign_root / "D1",
        execution_plan=controls / "D1.plan.json",
        execution_approval=controls / "D1.approval.json",
        timing_probe_plan=None,
        timing_probe_approval=None,
    )
    run_krea_ladder._validate_execution_paths(args)
    args.campaign_dir = tmp_path / "outside"
    with pytest.raises(ValueError, match="campaign-dir must be below"):
        run_krea_ladder._validate_execution_paths(args)
    args.campaign_dir = campaign_root / "D1"
    args.execution_plan = tmp_path / "attacker.plan.json"
    with pytest.raises(ValueError, match="execution plan must be below"):
        run_krea_ladder._validate_execution_paths(args)

    assert krea_execution_plan._lexical_child(
        "/campaign/controls/probe.json", Path("/campaign/controls")
    )
    assert not krea_execution_plan._lexical_child(
        "/campaign/controls/../../etc/probe.json", Path("/campaign/controls")
    )
    assert not krea_execution_plan._lexical_child(
        "/etc/campaign.json", Path("/campaign")
    )


def _recipe_for_k1():
    values = {
        "planned_steps": 100,
        "submitted_step": None,
        "learning_rate": 0.0001,
        "rank": 32,
        "alpha": 32,
        "optimizer": "adamw8bit",
        "optimizer_parameters": {"weight_decay": 0.0001},
        "loss": "mse",
        "guidance": {"enabled": True, "scale": 2},
        "scheduler": "flowmatch",
        "dropout": 0.05,
        "gradient_accumulation": 1,
        "effective_batch": 1,
        "ema": {"enabled": False, "decay": 0.99},
        "save_cadence": 12,
        "selector": None,
    }
    return {
        "fields": {name: {"effective_value": value} for name, value in values.items()}
    }


def test_discovery_binding_parses_arm_fixture_seed_class_and_allowed_axes(tmp_path):
    discovery_path = Path("ops/calibration/week5/krea-discovery-plan.json")
    discovery = json.loads(discovery_path.read_text())
    discovery["discovery_tasks"]["D1"]["identity"] = "fixture-one"
    discovery["discovery_tasks"]["D1"]["fixture_split_manifest_sha256"] = SHA["1"]
    local = tmp_path / "discovery.json"
    local.write_text(json.dumps(discovery, indent=2) + "\n")
    binding = {
        "path": str(local),
        "sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
    }
    result = krea_execution_plan.validate_discovery_semantics(
        binding,
        arm_id="K1",
        fixture_id="D1",
        fixture_manifest_sha256=SHA["1"],
        training_pair_count=18,
        seed_role="A",
        seed=42565431,
        throughput_equivalence_class="A-rank32-adamw8bit-mse-guidance2",
        execution_recipe=_recipe_for_k1(),
        schedule_mode="measured_budget_fill",
        predeclared_recipe_axes=["planned_steps", "save_cadence"],
        basis_mode="internal",
    )
    assert result["arm"]["id"] == "K1"

    with pytest.raises(ValueError, match="seed"):
        krea_execution_plan.validate_discovery_semantics(
            binding,
            arm_id="K1",
            fixture_id="D1",
            fixture_manifest_sha256=SHA["1"],
            training_pair_count=18,
            seed_role="B",
            seed=42565431,
            throughput_equivalence_class="A-rank32-adamw8bit-mse-guidance2",
            execution_recipe=_recipe_for_k1(),
            schedule_mode="measured_budget_fill",
            predeclared_recipe_axes=["planned_steps", "save_cadence"],
            basis_mode="internal",
        )
    with pytest.raises(ValueError, match="axes"):
        krea_execution_plan.validate_discovery_semantics(
            binding,
            arm_id="K1",
            fixture_id="D1",
            fixture_manifest_sha256=SHA["1"],
            training_pair_count=18,
            seed_role="A",
            seed=42565431,
            throughput_equivalence_class="A-rank32-adamw8bit-mse-guidance2",
            execution_recipe=_recipe_for_k1(),
            schedule_mode="measured_budget_fill",
            predeclared_recipe_axes=["learning_rate"],
            basis_mode="internal",
        )


def test_public_basis_rebinds_every_primary_source(monkeypatch, tmp_path):
    source = {
        "schema": 1,
        "kind": "forge-krea-public-arm-provenance",
        "source_arm_id": "K2",
        "source": {"url": "https://example.invalid/K2", "revision": "1" * 40},
        "official_context": {"task_id": "task"},
        "files": {"source_config": {}, "source_artifact": {}},
        "fields": {"observed": {}, "unsupported": [], "adapted": []},
        "evaluator_sha": None,
        "matched_concept": {"available": False},
        "adaptation_target": {"mode": "local_reproduction"},
        "normalized_recipe": {},
        "review_assertion": {"status": "unreviewed"},
        "manifest_sha256": SHA["1"],
    }
    approval = {
        "schema": 1,
        "kind": "forge-krea-source-normalization-approval",
        "source_arm_id": "K2",
        "provenance_manifest_sha256": SHA["1"],
        "reviewer_identity": "Jordan Example",
        "decision": "approved",
        "assertions": {
            "source_fields_reviewed": True,
            "unsupported_fields_reviewed": True,
            "adaptations_reviewed": True,
            "source_artifact_identity_reviewed": True,
            "claim_limits_reviewed": True,
        },
    }
    bindings = {
        name: {"path": str(tmp_path / name), "sha256": SHA["2"]}
        for name in (
            "source_config",
            "source_artifact",
            "field_ledger",
            "task_raw",
            "tournament_raw",
            "revision_manifest",
        )
    }
    for binding in bindings.values():
        Path(binding["path"]).write_bytes(b"x")
    monkeypatch.setattr(
        krea_execution_plan,
        "_load_binding",
        lambda value, label: (
            (Path("/source"), source, SHA["3"])
            if "source provenance" in label
            else (Path("/approval"), approval, SHA["4"])
        ),
    )
    monkeypatch.setattr(
        krea_execution_plan,
        "_file_binding",
        lambda value, label: (Path(value["path"]), value["sha256"]),
    )
    seen = {}
    monkeypatch.setattr(
        krea_execution_plan.krea_provenance,
        "validate_manifest",
        lambda manifest, **kwargs: seen.update(kwargs) or manifest,
    )
    monkeypatch.setattr(
        krea_execution_plan.krea_public_source,
        "build_metadata",
        lambda *args, **kwargs: {"machine_derived": True},
    )
    monkeypatch.setattr(
        krea_execution_plan.krea_provenance,
        "build_manifest",
        lambda *args, **kwargs: json.loads(json.dumps(source)),
    )
    monkeypatch.setattr(
        krea_execution_plan.krea_provenance,
        "normalize_execution_recipe",
        lambda recipe, source_recipe: recipe,
    )
    result = krea_execution_plan._arm_basis(
        {
            "mode": "public_submission",
            "source_provenance": {"path": "/source", "sha256": SHA["3"]},
            "source_normalization_approval": {"path": "/approval", "sha256": SHA["4"]},
            "source_files": bindings,
        },
        arm_id="K2",
        execution_recipe={"fields": {}},
    )
    assert set(seen) == {
        "source_config_path",
        "source_artifact_path",
        "field_ledger_path",
        "task_raw_path",
        "tournament_raw_path",
        "revision_manifest_path",
    }
    assert set(result["rebound_source_files"]) == set(bindings)


def test_public_basis_rejects_self_rehashed_recipe_not_derived_from_fixed_yaml(
    monkeypatch, tmp_path
):
    """An internally canonical manifest cannot rewrite primary YAML facts."""

    source_config = tmp_path / "K2.yaml"
    source_config.write_text("train:\n  lr: 0.000086\n", encoding="utf-8")
    paths = {
        "source_config": source_config,
        "source_artifact": tmp_path / "K2.safetensors",
        "field_ledger": tmp_path / "ledger.json",
        "task_raw": tmp_path / "task.json",
        "tournament_raw": tmp_path / "tournament.json",
        "revision_manifest": tmp_path / "revision.json",
    }
    for name, path in paths.items():
        if name != "source_config":
            path.write_bytes(b"fixed-primary-bytes")
    expected = {
        "schema": 1,
        "kind": "forge-krea-public-arm-provenance",
        "source_arm_id": "K2",
        "source": {"url": "https://example.invalid/K2", "revision": "1" * 40},
        "official_context": {"task_id": "task"},
        "files": {"source_config": {"sha256": "fixed"}, "source_artifact": {}},
        "fields": {
            "observed": {"/config/process/0/train/lr": 0.000086},
            "unsupported": [],
            "adapted": [],
        },
        "evaluator_sha": None,
        "matched_concept": {"available": False},
        "adaptation_target": {"mode": "local_reproduction"},
        "normalized_recipe": {"fields": {"learning_rate": {"source_value": 0.000086}}},
        "review_assertion": {"status": "unreviewed"},
        "manifest_sha256": SHA["1"],
    }
    tampered = json.loads(json.dumps(expected))
    tampered["fields"]["observed"]["/config/process/0/train/lr"] = 0.5
    tampered["normalized_recipe"]["fields"]["learning_rate"]["source_value"] = 0.5
    tampered["manifest_sha256"] = krea_provenance.canonical_sha256(
        {key: value for key, value in tampered.items() if key != "manifest_sha256"}
    )
    approval = {
        "schema": 1,
        "kind": "forge-krea-source-normalization-approval",
        "source_arm_id": "K2",
        "provenance_manifest_sha256": tampered["manifest_sha256"],
        "reviewer_identity": "Jordan Example",
        "decision": "approved",
        "assertions": {
            "source_fields_reviewed": True,
            "unsupported_fields_reviewed": True,
            "adaptations_reviewed": True,
            "source_artifact_identity_reviewed": True,
            "claim_limits_reviewed": True,
        },
    }
    bindings = {
        name: {"path": str(path), "sha256": SHA["2"]} for name, path in paths.items()
    }
    monkeypatch.setattr(
        krea_execution_plan,
        "_load_binding",
        lambda value, label: (
            (tmp_path / "source.json", tampered, SHA["3"])
            if "source provenance" in label
            else (tmp_path / "approval.json", approval, SHA["4"])
        ),
    )
    monkeypatch.setattr(
        krea_execution_plan,
        "_file_binding",
        lambda value, label: (Path(value["path"]), value["sha256"]),
    )
    # Model the old consumer: both the self-digesting manifest and an updated
    # approval validate.  The new primary-byte derivation is what must reject.
    monkeypatch.setattr(
        krea_execution_plan.krea_provenance,
        "validate_manifest",
        lambda manifest, **kwargs: manifest,
    )
    monkeypatch.setattr(
        krea_execution_plan.krea_public_source,
        "build_metadata",
        lambda *args, **kwargs: {"parsed_from_fixed_yaml": True},
    )
    monkeypatch.setattr(
        krea_execution_plan.krea_provenance,
        "build_manifest",
        lambda *args, **kwargs: expected,
    )
    with pytest.raises(ValueError, match="primary-byte re-derivation"):
        krea_execution_plan._arm_basis(
            {
                "mode": "public_submission",
                "source_provenance": {"path": "/source", "sha256": SHA["3"]},
                "source_normalization_approval": {
                    "path": "/approval",
                    "sha256": SHA["4"],
                },
                "source_files": bindings,
            },
            arm_id="K2",
            execution_recipe={"fields": {}},
        )


def test_public_basis_rejects_self_rehashed_disclosure_not_derived_from_primary_bytes(
    monkeypatch, tmp_path
):
    """A rehashed disclosure cannot diverge from the primary-byte derivation."""

    paths = {
        name: tmp_path / name
        for name in (
            "source_config",
            "source_artifact",
            "field_ledger",
            "task_raw",
            "tournament_raw",
            "revision_manifest",
        )
    }
    for path in paths.values():
        path.write_bytes(b"fixed-primary-bytes")
    disclosure = {
        "schema": 1,
        "kind": "forge-krea-local-reproduction-disclosure",
        "execution_authorized": False,
        "adapted_fields": [
            {
                "name": "depth policy",
                "source_recipe_fields": ["planned_steps", "submitted_step"],
                "local_policy": "fill the measured budget",
                "evidence": "derived from the bound public source",
            }
        ],
        "source_unknown_fields": [],
        "predeclared_local_values": [],
        "claim_limit": "disclosure only; not execution approval",
    }
    expected = {
        "schema": 2,
        "kind": "forge-krea-public-arm-provenance",
        "source_arm_id": "K2",
        "source": {"url": "https://example.invalid/K2", "revision": "1" * 40},
        "official_context": {"task_id": "task"},
        "files": {"source_config": {}, "source_artifact": {}},
        "fields": {"observed": {}, "unsupported": [], "adapted": []},
        "evaluator_sha": None,
        "matched_concept": {"available": False},
        "adaptation_target": {"mode": "local_reproduction"},
        "local_reproduction_disclosure": disclosure,
        "normalized_recipe": {},
        "review_assertion": {"status": "unreviewed"},
    }
    expected["manifest_sha256"] = krea_provenance.canonical_sha256(expected)
    tampered = json.loads(json.dumps(expected))
    tampered["local_reproduction_disclosure"]["adapted_fields"][0][
        "local_policy"
    ] = "silently use a different local depth policy"
    tampered["manifest_sha256"] = krea_provenance.canonical_sha256(
        {key: value for key, value in tampered.items() if key != "manifest_sha256"}
    )
    approval = {
        "schema": 1,
        "kind": "forge-krea-source-normalization-approval",
        "source_arm_id": "K2",
        "provenance_manifest_sha256": tampered["manifest_sha256"],
        "reviewer_identity": "Jordan Example",
        "decision": "approved",
        "assertions": {
            "source_fields_reviewed": True,
            "unsupported_fields_reviewed": True,
            "adaptations_reviewed": True,
            "source_artifact_identity_reviewed": True,
            "claim_limits_reviewed": True,
        },
    }
    bindings = {
        name: {"path": str(path), "sha256": SHA["2"]} for name, path in paths.items()
    }
    monkeypatch.setattr(
        krea_execution_plan,
        "_load_binding",
        lambda value, label: (
            (tmp_path / "source.json", tampered, SHA["3"])
            if "source provenance" in label
            else (tmp_path / "approval.json", approval, SHA["4"])
        ),
    )
    monkeypatch.setattr(
        krea_execution_plan,
        "_file_binding",
        lambda value, label: (Path(value["path"]), value["sha256"]),
    )
    # Model a manifest whose own digest and human approval have both been
    # updated.  Only re-derivation from the immutable source bytes exposes it.
    monkeypatch.setattr(
        krea_execution_plan.krea_provenance,
        "validate_manifest",
        lambda manifest, **kwargs: manifest,
    )
    monkeypatch.setattr(
        krea_execution_plan.krea_public_source,
        "build_metadata",
        lambda *args, **kwargs: {"parsed_from_fixed_primary_bytes": True},
    )
    monkeypatch.setattr(
        krea_execution_plan.krea_provenance,
        "build_manifest",
        lambda *args, **kwargs: expected,
    )
    with pytest.raises(ValueError, match="primary-byte re-derivation"):
        krea_execution_plan._arm_basis(
            {
                "mode": "public_submission",
                "source_provenance": {"path": "/source", "sha256": SHA["3"]},
                "source_normalization_approval": {
                    "path": "/approval",
                    "sha256": SHA["4"],
                },
                "source_files": bindings,
            },
            arm_id="K2",
            execution_recipe={"fields": {}},
        )


def test_pre_run_approval_contains_no_future_natural_completion(monkeypatch):
    resolved = {
        "host_execution_manifest": {"host_execution_identity_sha256": SHA["1"]},
        "throughput_profile": {"profile_sha256": SHA["2"]},
    }
    monkeypatch.setattr(krea_execution_plan, "validate_plan", lambda plan: resolved)
    approval = krea_execution_plan.build_approval(
        {"plan_sha256": SHA["3"]},
        reviewer_identity="Jordan Example",
        approved_at_utc="2026-07-28T01:02:03Z",
    )
    assert "h100_certification" not in approval
    assert approval["assertions"]["natural_completion_is_post_run_evidence"] is True
    assert approval["gpu_execution_authorized"] is True


def test_bootstrap_runner_derives_isolated_names_without_a_profile():
    probe = {
        "task_id": "timing-probe",
        "expected_repo_name": "timing-repo",
        "probe_schedule": {
            "planned_steps": 101,
            "save_every": 13,
            "hard_budget_s": 2700,
        },
        "probe_contract_sha256": SHA["1"],
    }
    first = run_krea_ladder._bootstrap_execution_shape(probe, capture_id="first")
    second = run_krea_ladder._bootstrap_execution_shape(probe, capture_id="second")
    assert first["task_id"] != second["task_id"]
    assert first["expected_repo_name"] != second["expected_repo_name"]
    assert first["schedule"] == {
        "mode": "timing_probe_fixed_depth",
        "planned_steps": 101,
        "save_every": 13,
        "candidate_steps": [13, 26, 39, 52, 65, 78, 91, 101],
        "required_landmarks": [],
        "landmark_policy": "none",
    }
    assert first["budget_plan"] == {"hard_budget_s": "2700"}


def test_calibration_forces_exact_final_and_restores_production_selector():
    production_selection = SimpleNamespace(source="training_loss_divergence")
    exact_final = SimpleNamespace(source="exact_final")

    def production_select(*_args):
        return production_selection

    module = SimpleNamespace(
        select=production_select,
        _default_selection=lambda *_args: exact_final,
    )
    with pytest.raises(RuntimeError, match="forced test unwind"):
        with run_krea_ladder._calibration_exact_final_selection(module):
            assert module.select([], "repo", "/checkpoints", {}) is exact_final
            raise RuntimeError("forced test unwind")
    assert module.select is production_select
    assert module.select([], "repo", "/checkpoints", {}) is production_selection


def test_calibration_exact_final_selection_fails_closed_without_natural_final():
    production_select = lambda *_args: SimpleNamespace(source="production")
    module = SimpleNamespace(
        select=production_select,
        _default_selection=lambda *_args: SimpleNamespace(
            source="highest_valid_periodic"
        ),
    )
    with run_krea_ladder._calibration_exact_final_selection(module):
        with pytest.raises(RuntimeError, match="without an exact natural final"):
            module.select([], "repo", "/checkpoints", {})
    assert module.select is production_select


class _FakeScopeClient:
    def __init__(self, *, wait_action="complete"):
        self.pid = 43210
        self.returncode = None
        self.wait_action = wait_action

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if callable(self.wait_action):
            return self.wait_action(self)
        if self.wait_action == "timeout":
            raise subprocess.TimeoutExpired(["systemd-run"], timeout)
        self.returncode = 0
        return 0

    def terminate(self):
        self.returncode = -signal.SIGTERM

    def kill(self):
        self.returncode = -signal.SIGKILL


def _fake_systemd_binaries(tmp_path):
    systemd_run = tmp_path / "systemd-run"
    systemctl = tmp_path / "systemctl"
    for path in (systemd_run, systemctl):
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    return systemd_run, systemctl


def test_timing_scope_wrapper_is_recursive_bounded_and_collected(monkeypatch, tmp_path):
    binaries = _fake_systemd_binaries(tmp_path)
    fake = _FakeScopeClient()
    launched = {}
    monkeypatch.setattr(krea_timing_probe, "_systemd_prerequisites", lambda: binaries)
    monkeypatch.setattr(
        krea_timing_probe.subprocess,
        "Popen",
        lambda argv, **kwargs: launched.update(argv=argv, kwargs=kwargs) or fake,
    )
    monkeypatch.setattr(
        krea_timing_probe, "_validated_process_group", lambda _process: fake.pid
    )
    monkeypatch.setattr(
        krea_timing_probe,
        "_scope_status",
        lambda *args, **kwargs: "active",
    )
    monkeypatch.setattr(
        krea_timing_probe, "_process_group_is_empty", lambda _pgid: True
    )
    monkeypatch.setattr(
        krea_timing_probe, "_wait_scope_collected", lambda *args, **kwargs: True
    )

    returncode, receipt = krea_timing_probe._run_in_transient_scope(
        ["/usr/bin/python3", "worker.py"],
        env={"FORGE_KREA_TIMING_SOCKET": "/tmp/socket"},
        timeout_s=30.0,
        capture_id="capture-a",
    )

    assert returncode == 0
    assert "--scope" in launched["argv"]
    assert "--collect" in launched["argv"]
    assert "--property=KillMode=control-group" in launched["argv"]
    assert "--property=RuntimeMaxSec=40.0s" in launched["argv"]
    assert launched["kwargs"]["start_new_session"] is True
    assert receipt["scope_observed_active"] is True
    assert receipt["recursive_cleanup_proven"] is True
    assert receipt["unit_collected"] is True


def test_capture_child_environment_is_constructed_not_inherited(monkeypatch):
    for name in (
        "PYTORCH_CUDA_ALLOC_CONF",
        "HF_HOME",
        "LD_LIBRARY_PATH",
        "HTTPS_PROXY",
        "OMP_NUM_THREADS",
        "NCCL_DEBUG",
    ):
        monkeypatch.setenv(name, "operator-value")

    environment = krea_timing_probe._capture_child_environment(
        socket_path="/tmp/receipt.sock",
        contract_sha256="b" * 64,
        capture_id="timing-a",
    )

    assert environment == {
        "PATH": "/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "FORGE_KREA_TIMING_SOCKET": "/tmp/receipt.sock",
        "FORGE_KREA_TIMING_PROBE_CONTRACT_SHA256": "b" * 64,
        "FORGE_KREA_TIMING_CAPTURE_ID": "timing-a",
    }


def test_timing_scope_timeout_cleans_recursive_cgroup(monkeypatch, tmp_path):
    binaries = _fake_systemd_binaries(tmp_path)
    fake = _FakeScopeClient(wait_action="timeout")
    cleaned = []
    monkeypatch.setattr(krea_timing_probe, "_systemd_prerequisites", lambda: binaries)
    monkeypatch.setattr(
        krea_timing_probe.subprocess, "Popen", lambda *args, **kwargs: fake
    )
    monkeypatch.setattr(
        krea_timing_probe, "_validated_process_group", lambda _process: fake.pid
    )
    monkeypatch.setattr(
        krea_timing_probe,
        "_scope_status",
        lambda *args, **kwargs: (
            "collected" if fake.returncode is not None else "active"
        ),
    )

    def cleanup(process, **kwargs):
        cleaned.append(kwargs)
        process.returncode = -signal.SIGTERM

    monkeypatch.setattr(krea_timing_probe, "_terminate_scope_and_client", cleanup)
    with pytest.raises(TimeoutError, match="recursive scope cleaned"):
        krea_timing_probe._run_in_transient_scope(
            ["/usr/bin/python3", "escape-with-setsid.py"],
            env={},
            timeout_s=1.0,
            capture_id="timeout-capture",
        )
    assert len(cleaned) == 1
    assert cleaned[0]["unit"].startswith("forge-krea-timing-")


def test_timing_scope_validation_failure_still_cleans_cgroup(monkeypatch, tmp_path):
    binaries = _fake_systemd_binaries(tmp_path)
    fake = _FakeScopeClient()
    cleaned = []
    monkeypatch.setattr(krea_timing_probe, "_systemd_prerequisites", lambda: binaries)
    monkeypatch.setattr(
        krea_timing_probe.subprocess, "Popen", lambda *args, **kwargs: fake
    )
    monkeypatch.setattr(
        krea_timing_probe,
        "_validated_process_group",
        lambda _process: (_ for _ in ()).throw(RuntimeError("untrusted PGID")),
    )

    def cleanup(process, **kwargs):
        cleaned.append(kwargs)
        process.returncode = -signal.SIGTERM

    monkeypatch.setattr(
        krea_timing_probe, "_terminate_scope_without_validated_group", cleanup
    )
    with pytest.raises(RuntimeError, match="untrusted PGID"):
        krea_timing_probe._run_in_transient_scope(
            ["/usr/bin/python3", "worker.py"],
            env={},
            timeout_s=30.0,
            capture_id="validation-failure",
        )
    assert len(cleaned) == 1


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGHUP])
def test_timing_scope_signal_paths_cleanup_before_propagating(
    monkeypatch, tmp_path, stop_signal
):
    binaries = _fake_systemd_binaries(tmp_path)
    handlers = {}

    def install(signum, handler):
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    def interrupted_wait(_process):
        handlers[stop_signal](stop_signal, None)

    fake = _FakeScopeClient(wait_action=interrupted_wait)
    cleaned = []
    monkeypatch.setattr(krea_timing_probe, "_systemd_prerequisites", lambda: binaries)
    monkeypatch.setattr(
        krea_timing_probe.subprocess, "Popen", lambda *args, **kwargs: fake
    )
    monkeypatch.setattr(
        krea_timing_probe, "_validated_process_group", lambda _process: fake.pid
    )
    monkeypatch.setattr(
        krea_timing_probe, "_scope_status", lambda *args, **kwargs: "active"
    )
    monkeypatch.setattr(
        krea_timing_probe.signal, "getsignal", lambda signum: signal.SIG_DFL
    )
    monkeypatch.setattr(krea_timing_probe.signal, "signal", install)

    def cleanup(process, **kwargs):
        cleaned.append(kwargs)
        process.returncode = -stop_signal

    monkeypatch.setattr(krea_timing_probe, "_terminate_scope_and_client", cleanup)
    with pytest.raises(krea_timing_probe._CaptureCancellation):
        krea_timing_probe._run_in_transient_scope(
            ["/usr/bin/python3", "escape-with-setsid.py"],
            env={},
            timeout_s=30.0,
            capture_id=f"signal-{stop_signal.name.lower()}",
        )
    assert len(cleaned) == 1


def test_timing_scope_fails_closed_without_rootful_systemd(monkeypatch):
    monkeypatch.setattr(krea_timing_probe.sys, "platform", "linux")
    monkeypatch.setattr(krea_timing_probe.os, "geteuid", lambda: 1000)
    with pytest.raises(PermissionError, match="rootful"):
        krea_timing_probe._systemd_prerequisites()
