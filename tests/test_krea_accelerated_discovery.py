"""Fail-closed tests for the owner-directed accelerated discovery matrix."""

from __future__ import annotations

import hashlib
from pathlib import Path
import pytest

from ops.calibration import krea_accelerated_discovery as accelerated
from ops.calibration import krea_budget
from ops.calibration import krea_provenance
from ops.calibration import krea_execution_plan
from ops.calibration import krea_runtime_binding as runtime_binding
from ops.calibration import krea_timing_probe
from ops.calibration import krea_training_evidence
from ops.calibration import run_krea_ladder as runner


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _binding(label: str, semantic: str) -> dict[str, str]:
    return {
        "path": f"/sealed/{label}.json",
        "file_sha256": _sha(f"{label}-file"),
        semantic: _sha(f"{label}-semantic"),
    }


def _run_bundle_binding(tmp_path: Path, label: str = "D1-K4-run") -> dict[str, str]:
    body = {"schema": 2, "kind": "forge-krea-run-evidence-bundle", "label": label}
    bundle = {**body, "bundle_sha256": krea_provenance.canonical_sha256(body)}
    path = tmp_path / f"{label}.bundle.json"
    path.write_bytes(krea_provenance.canonical_bytes(bundle) + b"\n")
    return {
        "path": str(path),
        "file_sha256": krea_provenance.file_sha256(path),
        "bundle_sha256": bundle["bundle_sha256"],
    }


def _payload() -> dict:
    return {
        "discovery_plan": _binding("discovery", "discovery_sha256"),
        "discovery_execution_authorization": _binding(
            "authorization", "authorization_sha256"
        ),
        "fixture_admission_envelope": _binding("admission", "envelope_sha256"),
        "measured_profile": _binding("d1-a-profile", "profile_sha256"),
        "historical_host_execution_manifest": _binding(
            "historical-host", "host_execution_identity_sha256"
        ),
        "created_at_utc": "2026-07-30T20:50:00Z",
        "cadence_multiplier": 1,
        "schedule_slip_record": None,
        "supersedes_campaign_sha256": None,
    }


def test_umbrella_seals_exact_twelve_cell_matrix() -> None:
    campaign = accelerated.build_campaign(_payload())

    assert campaign["cell_count"] == 12
    assert [row["cell_id"] for row in campaign["cells"]] == [
        f"{fixture}-K{arm}"
        for fixture in ("D1", "D2")
        for arm in range(6)
    ]
    assert accelerated.campaign_cell(campaign, "D1", "K1")[
        "effective_hard_budget_s"
    ] == 2700
    assert accelerated.campaign_cell(campaign, "D1", "K3")[
        "effective_hard_budget_s"
    ] == 2454
    assert accelerated.campaign_cell(campaign, "D1", "K4")[
        "effective_hard_budget_s"
    ] == 1350
    assert accelerated.campaign_cell(campaign, "D2", "K1")[
        "effective_hard_budget_s"
    ] == 2160
    assert accelerated.campaign_cell(campaign, "D2", "K3")[
        "effective_hard_budget_s"
    ] == 1963
    assert accelerated.campaign_cell(campaign, "D2", "K4")[
        "effective_hard_budget_s"
    ] == 1080


def test_campaign_tampering_cannot_be_self_rehashed() -> None:
    campaign = accelerated.build_campaign(_payload())
    campaign["cells"][0]["runtime_factor"] = "0.25"
    body = {key: value for key, value in campaign.items() if key != "campaign_sha256"}
    campaign["campaign_sha256"] = krea_provenance.canonical_sha256(body)

    with pytest.raises(ValueError, match="exact twelve-cell matrix"):
        accelerated.validate_campaign(campaign)


def test_cadence_relief_requires_positive_bound_slip(tmp_path) -> None:
    payload = _payload()
    payload["cadence_multiplier"] = 2
    payload["supersedes_campaign_sha256"] = _sha("superseded")
    with pytest.raises(ValueError, match="requires a bound schedule-slip"):
        accelerated.build_campaign(payload)

    slip = accelerated.build_schedule_slip(
        campaign_sha256=payload["supersedes_campaign_sha256"],
        observed_at_utc="2026-07-30T21:00:00Z",
        schedule_slip_s=1,
        completed_cell_ids=["D1-K0"],
    )
    slip_path = tmp_path / "slip.json"
    slip_path.write_bytes(krea_provenance.canonical_bytes(slip) + b"\n")
    payload["schedule_slip_record"] = {
        "path": str(slip_path),
        "file_sha256": krea_provenance.file_sha256(slip_path),
        "slip_sha256": slip["slip_sha256"],
    }
    campaign = accelerated.build_campaign(payload)
    assert all(row["cadence_multiplier"] == 2 for row in campaign["cells"])
    assert all(not row["depth_increase_from_cadence_relief"] for row in campaign["cells"])


def test_k4_correction_is_one_way_and_capped(tmp_path: Path) -> None:
    run_bundle = _run_bundle_binding(tmp_path)
    correction = accelerated.build_k4_correction(
        campaign_sha256=_sha("campaign"),
        source_run_bundle=run_bundle,
        predicted_first_checkpoint_s="100",
        observed_first_checkpoint_s="120",
        observed_at_utc="2026-07-30T21:15:00Z",
    )
    assert correction["corrected_runtime_factor"] == "3.75"
    assert correction["factor_decrease_forbidden"] is True
    assert correction["depth_increase_authorized"] is False

    with pytest.raises(ValueError, match="exceeds the preauthorized"):
        accelerated.build_k4_correction(
            campaign_sha256=_sha("campaign"),
            source_run_bundle=run_bundle,
            predicted_first_checkpoint_s="100",
            observed_first_checkpoint_s="200",
            observed_at_utc="2026-07-30T21:15:00Z",
        )


def test_k4_correction_source_is_same_campaign_uncorrected_d1_k4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _run_bundle_binding(tmp_path)
    correction = {"source_run_bundle": source, "campaign_sha256": _sha("campaign")}
    candidate_path = tmp_path / "candidate.json"
    plan_path = tmp_path / "D1-K4.plan.json"
    index_path = tmp_path / "source.index.json"
    for path in (candidate_path, plan_path, index_path):
        path.write_bytes(b"{}\n")
    plan = {
        "plan_sha256": _sha("plan"),
        "arm_id": "K4",
        "discovery_fixture_id": "D1",
        "discovery_profile_index": {
            "path": str(index_path),
            "file_sha256": krea_provenance.file_sha256(index_path),
            "index_sha256": _sha("index"),
        },
    }
    bundle = {
        "bundle_sha256": source["bundle_sha256"],
        "arm_id": "K4",
        "execution_plan_sha256": plan["plan_sha256"],
        "candidate_bindings": [{"binding": {"path": str(candidate_path)}}],
    }
    candidate = {
        "execution_plan": {
            "path": str(plan_path),
            "sha256": krea_provenance.file_sha256(plan_path),
        }
    }
    monkeypatch.setattr(
        krea_training_evidence, "validate_run_evidence", lambda _path: bundle
    )

    def fake_load(path, _label, *, canonical):
        assert canonical is True
        if Path(path) == candidate_path:
            return candidate_path, candidate, krea_provenance.file_sha256(candidate_path)
        assert Path(path) == plan_path
        return plan_path, plan, krea_provenance.file_sha256(plan_path)

    monkeypatch.setattr(runtime_binding, "_load_json", fake_load)
    source_index = {
        "index_sha256": _sha("index"),
        "accelerated_discovery_campaign": {
            "campaign_sha256": correction["campaign_sha256"]
        },
        "k4_correction": None,
    }
    monkeypatch.setattr(
        runtime_binding, "_load_profile_index", lambda _path: source_index
    )
    runtime_binding._validate_k4_source_run(correction)

    source_index["accelerated_discovery_campaign"]["campaign_sha256"] = _sha(
        "another-campaign"
    )
    with pytest.raises(ValueError, match="same campaign's uncorrected"):
        runtime_binding._validate_k4_source_run(correction)


def test_accelerated_index_contains_one_real_profile_and_six_proxy_slots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    classes = (
        "A-rank32-adamw8bit-mse-guidance2",
        "B-rank32-adamw8bit-mae-guidance3",
        "C-rank64-automagic-mse-guidance2",
    )
    discovery = {
        "arms": [
            {
                "throughput_equivalence_class": class_name,
                "rank": 64 if class_name.startswith("C-") else 32,
                "alpha": 64 if class_name.startswith("C-") else 32,
                "optimizer": (
                    "automagic" if class_name.startswith("C-") else "adamw8bit"
                ),
                "loss": "mae" if class_name.startswith("B-") else "mse",
                "guidance": 3 if class_name.startswith("B-") else 2,
            }
            for class_name in classes
        ]
    }
    campaign_payload = _payload()
    campaign_payload["discovery_plan"]["discovery_sha256"] = (
        krea_provenance.canonical_sha256(discovery)
    )
    campaign = accelerated.build_campaign(campaign_payload)
    authorization = {
        "authorization_sha256": _sha("authorization-semantic"),
        "fixture_admission_envelope": {
            **campaign["fixture_admission_envelope"],
            "owner_ratification_sha256": _sha("owner-ratification"),
        },
    }
    profile_record = {
        "path": "/sealed/d1-a-profile.json",
        "file_sha256": _sha("d1-a-profile-file"),
        "profile_sha256": _sha("d1-a-profile-semantic"),
        "execution_envelope_sha256": _sha("profile-envelope"),
        "campaign_runtime_identity_sha256": _sha("campaign-runtime"),
    }

    monkeypatch.setattr(
        runtime_binding,
        "_load_discovery",
        lambda _path: (
            Path("/sealed/discovery.json"),
            discovery,
            _sha("discovery-file"),
            classes,
        ),
    )
    monkeypatch.setattr(
        runtime_binding.krea_discovery_authorization,
        "load_binding",
        lambda _value: (
            Path("/sealed/authorization.json"),
            authorization,
            _sha("authorization-file"),
        ),
    )
    monkeypatch.setattr(
        runtime_binding.krea_discovery_authorization,
        "assert_matches_discovery",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_binding.krea_accelerated_discovery,
        "load_campaign_binding",
        lambda _value: (
            Path("/sealed/campaign.json"),
            campaign,
            _sha("campaign-file"),
        ),
    )

    def fake_fixture(fixture_id: str, _value: object):
        fixture = {
            "training_rows": list(range(18 if fixture_id == "D1" else 36)),
            "training_dataset_shape_sha256": _sha(f"shape-{fixture_id}"),
        }
        record = {
            "manifest": {
                "path": f"/sealed/{fixture_id}-manifest.json",
                "file_sha256": _sha(f"{fixture_id}-manifest-file"),
                "manifest_sha256": _sha(f"{fixture_id}-manifest-semantic"),
            },
            "approval": {
                "path": f"/sealed/{fixture_id}-approval.json",
                "file_sha256": _sha(f"{fixture_id}-approval-file"),
                "approval_sha256": _sha(f"{fixture_id}-approval-semantic"),
            },
            "concept_id": f"concept-{fixture_id}",
            "training_pair_count": len(fixture["training_rows"]),
            "training_dataset_shape_sha256": fixture[
                "training_dataset_shape_sha256"
            ],
        }
        return fixture, record

    monkeypatch.setattr(runtime_binding, "_load_fixture", fake_fixture)
    monkeypatch.setattr(
        runtime_binding,
        "_load_profile",
        lambda *_args, **_kwargs: dict(profile_record),
    )
    monkeypatch.setattr(runtime_binding, "_validate_k4_source_run", lambda _row: None)
    payload = {
        "discovery_plan": "/sealed/discovery.json",
        "discovery_execution_authorization": {
            "path": "/sealed/authorization.json",
            "file_sha256": _sha("authorization-file"),
            "authorization_sha256": _sha("authorization-semantic"),
        },
        "fixtures": {"D1": {}, "D2": {}},
        "accelerated_discovery_campaign": {
            "path": "/sealed/campaign.json",
            "file_sha256": _sha("campaign-file"),
            "campaign_sha256": campaign["campaign_sha256"],
        },
        "measured_profile": "/sealed/d1-a-profile.json",
    }
    index = runtime_binding.build_profile_index(payload)

    assert index["schema"] == 3
    assert index["measured_profile_count"] == 1
    assert index["target_slot_count"] == 6
    assert index["fixtures"]["D1"]["profiles"][classes[0]][
        "source_profile"
    ] == profile_record
    assert index["fixtures"]["D2"]["profiles"][classes[2]][
        "effective_hard_budget_s"
    ] == 1080

    correction = accelerated.build_k4_correction(
        campaign_sha256=campaign["campaign_sha256"],
        source_run_bundle=_run_bundle_binding(tmp_path),
        predicted_first_checkpoint_s="100",
        observed_first_checkpoint_s="120",
        observed_at_utc="2026-07-30T22:15:00Z",
    )
    correction_path = tmp_path / "k4-correction.json"
    correction_path.write_bytes(krea_provenance.canonical_bytes(correction) + b"\n")
    corrected_payload = {
        **payload,
        "k4_correction": {
            "path": str(correction_path),
            "file_sha256": krea_provenance.file_sha256(correction_path),
            "correction_sha256": correction["correction_sha256"],
        },
    }
    corrected = runtime_binding.build_profile_index(corrected_payload)
    corrected_k4 = corrected["fixtures"]["D2"]["profiles"][classes[2]]
    assert corrected_k4["runtime_factor"] == "3.75"
    assert corrected_k4["effective_hard_budget_s"] == 720
    assert corrected_k4["k4_correction_sha256"] == correction[
        "correction_sha256"
    ]

    tampered = {**index, "target_slot_count": 5}
    tampered["index_sha256"] = krea_provenance.canonical_sha256(
        {key: value for key, value in tampered.items() if key != "index_sha256"}
    )
    with pytest.raises(ValueError, match="identity is invalid"):
        runtime_binding.validate_profile_index(tampered)


def _recipe(planned: int, cadence: int) -> dict:
    return {
        "fields": {
            "planned_steps": {"effective_value": planned},
            "save_cadence": {"effective_value": cadence},
        }
    }


def _budget(planned: int, cadence: int, hard: int) -> dict:
    return {
        "max_affordable_steps": planned,
        "save_every": cadence,
        "actual_candidates": [
            {"step": step}
            for step in list(range(cadence, planned, cadence)) + [planned]
        ],
        "hard_budget_s": str(hard),
        "accounting": {"maximum_save_overhead_fraction": "1.0"},
    }


def _profile() -> krea_budget.ThroughputProfile:
    envelope = krea_budget.seal_execution_envelope(
        equivalence_class="a-rank32-adamw8bit-mse-guidance2",
        network_rank=32,
        network_alpha=32,
        optimizer="adamw8bit",
        optimizer_config_sha256=_sha("optimizer"),
        loss="mse",
        differential_guidance_enabled=True,
        guidance_scale=2.0,
        training_pair_count=24,
        training_dataset_shape_sha256=_sha("dataset-shape"),
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
        venv_tree_manifest_sha256=_sha("venv"),
        reference_container_image_sha256=_sha("container"),
        gpu_identity_sha256=_sha("gpu"),
        trainer_identity_sha256=_sha("trainer"),
        measurement_tool_sha256=_sha("measurement"),
    )
    document = krea_budget.seal_throughput_profile(
        execution_envelope=envelope,
        raw_sample_manifest_sha256=_sha("raw"),
        startup_sample_count=3,
        update_sample_count=100,
        save_sample_count=8,
        startup_upper_bound_s=1.0,
        update_upper_bound_s=1.0,
        save_upper_bound_s=1.0,
        bound_method="observed-max-plus-predeclared-margin",
        margin_policy_sha256=_sha("margin"),
        end_to_end_validation_count=1,
        end_to_end_validation_sha256=_sha("heldout"),
        framework_stop_boundary_s=225.0,
        framework_stop_boundary_source_sha256=_sha("boundary"),
        selection_mode="offline_post_training",
        selection_scorer_identity_sha256=None,
        selection_scoring_reserve_s=0.0,
        finalization_reserve_s=1.0,
        upload_reserve_s=1.0,
    )
    return krea_budget.load_throughput_profile(document)


def test_real_profile_exposes_runtime_identity_only_through_envelope() -> None:
    profile = _profile()
    assert not hasattr(profile, "runtime_identity_sha256")
    assert profile.execution_envelope.runtime_identity_sha256 == _sha("runtime")
    assert "profile.runtime_identity_sha256" not in Path(runner.__file__).read_text()


def test_every_other_checkpoint_requires_slip_bound_campaign_cadence() -> None:
    cell = {"cadence_multiplier": 2, "fixture_id": "D1", "arm_id": "K1"}
    schedule = {
        "mode": "measured_budget_fill",
        "planned_steps": 100,
        "save_every": 26,
        "candidate_steps": [26, 52, 78, 100],
        "required_landmarks": [],
        "landmark_policy": "none",
    }
    assert krea_execution_plan._schedule(
        schedule,
        recipe=_recipe(100, 26),
        budget_plan=_budget(100, 13, 1000),
        profile=_profile(),
        accelerated_cell=cell,
    ) == schedule

    with pytest.raises(ValueError, match="cadence multiplier"):
        krea_execution_plan._schedule(
            schedule,
            recipe=_recipe(100, 26),
            budget_plan=_budget(100, 13, 1000),
            profile=_profile(),
        )


def test_release_control_relief_keeps_depth_and_doubles_exact_cadence() -> None:
    cell = {"cadence_multiplier": 2, "fixture_id": "D2", "arm_id": "K0"}
    schedule = {
        "mode": "release_control",
        "planned_steps": 367,
        "save_every": 148,
        "candidate_steps": [148, 296, 367],
        "required_landmarks": [],
        "landmark_policy": "none",
    }
    assert krea_execution_plan._schedule(
        schedule,
        recipe=_recipe(367, 148),
        budget_plan=_budget(500, 63, 2160),
        profile=_profile(),
        accelerated_cell=cell,
    ) == schedule

    wrong = {**schedule, "planned_steps": 368, "candidate_steps": [148, 296, 368]}
    with pytest.raises(ValueError, match="depth/cadence drifted"):
        krea_execution_plan._schedule(
            wrong,
            recipe=_recipe(368, 148),
            budget_plan=_budget(500, 63, 2160),
            profile=_profile(),
            accelerated_cell=cell,
        )


def test_source_transition_allows_only_the_control_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compatibility = {
        "document": {
            "historical_compatibility": {
                "source_commit": "58822b496019177a02fa6196247ac30e788331bb"
            }
        }
    }
    changed = sorted(runner._ACCELERATED_TRANSITION_ALLOWED_PATHS)
    unsafe = {"deletion": False}

    def fake_run_text(command: list[str], *, cwd=None) -> str:
        if "status" in command:
            return ""
        if "diff" in command:
            if unsafe["deletion"]:
                return "D\tops/calibration/krea_execution_plan.py"
            return "\n".join(f"M\t{path}" for path in changed)
        raise AssertionError(command)

    monkeypatch.setattr(runner, "_run_text", fake_run_text)
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: object())
    runner._validate_control_only_source_transition(compatibility)

    changed.append("forge/tasks/aitoolkit.py")
    with pytest.raises(RuntimeError, match="non-control files"):
        runner._validate_control_only_source_transition(compatibility)

    unsafe["deletion"] = True
    with pytest.raises(RuntimeError, match="unsafe Git change"):
        runner._validate_control_only_source_transition(compatibility)


def test_real_successor_git_transition_is_control_only() -> None:
    compatibility = {
        "document": {
            "historical_compatibility": {
                "source_commit": "58822b496019177a02fa6196247ac30e788331bb"
            }
        }
    }
    runner._validate_control_only_source_transition(compatibility)


def test_admitted_588_timing_probe_uses_exact_historical_git_blobs() -> None:
    admitted_probe = {
        "probe_contract_sha256": (
            "490914141eed9a0d083c870185e8cd832de8c1c47b5108250a260196416dd0d4"
        ),
        "runner_sha256": (
            "b4ac3a6b475c3b59c3344baca103c7c03e5b0c14e9c25b24010906602b6a72df"
        ),
        "measurement_tool_sha256": (
            "e8eaca8495885a94e49e7611c2f9ac26fea3ed07ef29611aa077bb4a0c76ac6c"
        ),
    }
    assert admitted_probe["probe_contract_sha256"] == (
        "490914141eed9a0d083c870185e8cd832de8c1c47b5108250a260196416dd0d4"
    )
    assert krea_execution_plan._historical_timing_source_identities(
        "58822b496019177a02fa6196247ac30e788331bb"
    ) == {key: admitted_probe[key] for key in krea_execution_plan._TIMING_SOURCE_PATHS}
    with pytest.raises(ValueError, match="not authorized"):
        krea_execution_plan._historical_timing_source_identities("0" * 40)


def test_archival_replay_propagates_exact_historical_source_through_all_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_commit = "58822b496019177a02fa6196247ac30e788331bb"
    seen: list[str | None] = []

    def validate_capture(_value):
        seen.append(krea_execution_plan._HISTORICAL_TIMING_REPLAY_SOURCE.get())
        raise RuntimeError("capture-context-observed")

    monkeypatch.setattr(krea_timing_probe, "validate_capture", validate_capture)
    with pytest.raises(RuntimeError, match="capture-context-observed"):
        krea_execution_plan._replay_historical_timing(
            source_commit, krea_timing_probe.raw_from_captures, [{}]
        )
    with pytest.raises(RuntimeError, match="capture-context-observed"):
        krea_execution_plan._replay_historical_timing(
            source_commit, krea_timing_probe.end_to_end_from_records, [{}], []
        )
    with pytest.raises(RuntimeError, match="capture-context-observed"):
        krea_execution_plan._replay_historical_timing(
            None, krea_timing_probe.raw_from_captures, [{}]
        )
    assert seen == [source_commit, source_commit, None]
    assert krea_execution_plan._HISTORICAL_TIMING_REPLAY_SOURCE.get() is None
    assert krea_execution_plan._replay_historical_timing(
        source_commit,
        lambda: krea_execution_plan._HISTORICAL_TIMING_REPLAY_SOURCE.get(),
    ) == source_commit
    assert krea_execution_plan._HISTORICAL_TIMING_REPLAY_SOURCE.get() is None
    with pytest.raises(ValueError, match="not authorized"):
        krea_execution_plan._replay_historical_timing(
            "0" * 40, lambda: None
        )


def test_runner_allows_only_the_plan_derived_proxy_mismatch_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner, "_validate_control_only_source_transition", lambda _value: None
    )
    compatibility = {
        "cell": {
            "timing_evidence_mode": "conservative_proxy_not_measured_equivalence",
            "cadence_multiplier": 2,
        },
        "historical_host": {"host_execution_identity_sha256": _sha("old-host")},
        "proxy_mismatch_fields": [
            "network_alpha",
            "network_rank",
            "optimizer",
            "training_dataset_shape_sha256",
            "training_pair_count",
        ],
    }
    mismatch_names = set(compatibility["proxy_mismatch_fields"]) | {
        "host_execution_identity_sha256",
        "trainer_identity_sha256",
    }
    mismatches = {name: {"expected": "old", "actual": "new"} for name in mismatch_names}
    runner._validate_accelerated_proxy_transition(
        mismatches=mismatches,
        actual_host_execution_identity_sha256=_sha("new-host"),
        historical_host_execution_identity_sha256=_sha("old-host"),
        compatibility=compatibility,
    )

    mismatches["runtime_identity_sha256"] = {"expected": "old", "actual": "new"}
    with pytest.raises(RuntimeError, match="escaped its compatibility"):
        runner._validate_accelerated_proxy_transition(
            mismatches=mismatches,
            actual_host_execution_identity_sha256=_sha("new-host"),
            historical_host_execution_identity_sha256=_sha("old-host"),
            compatibility=compatibility,
        )
