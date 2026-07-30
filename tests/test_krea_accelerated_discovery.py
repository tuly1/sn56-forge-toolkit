"""Fail-closed tests for the owner-directed accelerated discovery matrix."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ops.calibration import krea_accelerated_discovery as accelerated
from ops.calibration import krea_provenance
from ops.calibration import krea_runtime_binding as runtime_binding
from ops.calibration import run_krea_ladder as runner


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _binding(label: str, semantic: str) -> dict[str, str]:
    return {
        "path": f"/sealed/{label}.json",
        "file_sha256": _sha(f"{label}-file"),
        semantic: _sha(f"{label}-semantic"),
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


def test_k4_correction_is_one_way_and_capped() -> None:
    correction = accelerated.build_k4_correction(
        campaign_sha256=_sha("campaign"),
        source_run_bundle_sha256=_sha("D1-K4-run"),
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
            source_run_bundle_sha256=_sha("D1-K4-run"),
            predicted_first_checkpoint_s="100",
            observed_first_checkpoint_s="200",
            observed_at_utc="2026-07-30T21:15:00Z",
        )


def test_accelerated_index_contains_one_real_profile_and_six_proxy_slots(
    monkeypatch: pytest.MonkeyPatch,
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
        "fixture_admission_envelope": campaign["fixture_admission_envelope"],
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

    tampered = {**index, "target_slot_count": 5}
    tampered["index_sha256"] = krea_provenance.canonical_sha256(
        {key: value for key, value in tampered.items() if key != "index_sha256"}
    )
    with pytest.raises(ValueError, match="identity is invalid"):
        runtime_binding.validate_profile_index(tampered)


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
    changed = [
        "ops/calibration/krea_accelerated_discovery.py",
        "ops/calibration/krea_execution_plan.py",
        "ops/calibration/krea_profile_index.py",
        "ops/calibration/krea_runtime_binding.py",
        "ops/calibration/run_krea_ladder.py",
        "ops/calibration/week5/krea-accelerated-discovery-policy.json",
        "tests/test_krea_accelerated_discovery.py",
    ]
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
