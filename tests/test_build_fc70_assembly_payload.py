"""Tests for the exact external fc70 assembly-payload builder."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from campaign_tools import build_fc70_assembly_payload as builder


def _publish(path: Path, value: dict) -> str:
    raw = builder.canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _inputs(tmp_path: Path) -> dict:
    staged = tmp_path / "basis.json"
    staged.write_bytes(b'{"accepted":true}\n')
    staged_row = {
        "relative_path": staged.name,
        "sha256": builder.file_sha(staged),
        "bytes": staged.stat().st_size,
    }
    arm = {
        "arm_basis": {"mode": "accepted"},
        "execution_recipe": {"schema": 1},
    }
    body = {
        "schema": 1,
        "kind": "forge-krea-fc70-arm-inputs",
        "source": "accepted-week5-artifacts-only",
        "arms": {f"K{index}": arm for index in range(6)},
        "staged_files": [staged_row],
    }
    arm_inputs = tmp_path / "fc70-arm-inputs.json"
    _publish(arm_inputs, {**body, "manifest_sha256": builder.semantic_sha(body)})

    probe = tmp_path / "probe.json"
    _publish(probe, {"base_model": builder.EXPECTED_BASE_MODEL})
    names = (
        "margin",
        "raw",
        "e2e-validation",
        "measurement-a",
        "measurement-b",
        "measurement-c",
        "heldout-capture",
        "heldout-run",
    )
    evidence = {}
    for name in names:
        path = tmp_path / f"{name}.json"
        _publish(path, {"name": name})
        evidence[name] = path
    admitted_evaluation = {
        "D1": "800b73ea4918d7bacc74dbc83ce365ec3490f653e5638430eb67e6f5e238301d",
        "D2": "f49c26acaa86a9b3785c29553e6ec0dcf41c7cbf799c61170932b1091b88d441",
    }
    admitted_archives = {
        "D1": "30601eceed1fa5590013aa9eee877b055a8e75986231edbd5045e421c083a201",
        "D2": "da5a647e318a1d1904635df8631458fccb680165fb3378aba2e25fa2b8e30f5e",
    }
    admission_root = tmp_path / "admission.58822b4"
    fixture_slots = {}
    for role in ("D1", "D2"):
        manifest_body = {
            "schema": 2,
            "kind": "forge-krea-curated-fixture",
            "experimental_role": role,
            "training_archive": {"sha256": admitted_archives[role]},
            "evaluation_dataset_identity": {"sha256": admitted_evaluation[role]},
        }
        manifest = {
            **manifest_body,
            "manifest_sha256": builder.semantic_sha(manifest_body),
        }
        relative = f"fixtures/{role}/fixture-manifest.json"
        manifest_path = admission_root / relative
        manifest_file_sha = _publish(manifest_path, manifest)
        fixture_slots[role] = {
            "manifest": {
                "relative_path": relative,
                "file_sha256": manifest_file_sha,
                "manifest_sha256": manifest["manifest_sha256"],
            }
        }
    envelope_body = {
        "schema": 1,
        "kind": "forge-krea-fixture-admission-envelope",
        "admission_authorized": True,
        "gpu_execution_authorized": False,
        "discovery_fixtures": fixture_slots,
    }
    admission_envelope = admission_root / "admission-envelope.json"
    _publish(
        admission_envelope,
        {**envelope_body, "envelope_sha256": builder.semantic_sha(envelope_body)},
    )
    return {
        "admission_envelope": admission_envelope,
        "arm_inputs": arm_inputs,
        "probe": probe,
        **evidence,
    }


def test_build_payload_binds_exact_observed_files(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    payload = builder.build_payload(
        admission_envelope=paths["admission_envelope"],
        arm_inputs=paths["arm_inputs"],
        probe_plan=paths["probe"],
        margin_policy=paths["margin"],
        raw_timing=paths["raw"],
        end_to_end_validation=paths["e2e-validation"],
        measurement_captures=(
            paths["measurement-a"],
            paths["measurement-b"],
            paths["measurement-c"],
        ),
        heldout_captures=(paths["heldout-capture"],),
        heldout_run_records=(paths["heldout-run"],),
    )

    assert payload["base_model"] == builder.EXPECTED_BASE_MODEL
    assert payload["fixtures"]["D1"]["evaluation_dataset"]["sha256"] == (
        "800b73ea4918d7bacc74dbc83ce365ec3490f653e5638430eb67e6f5e238301d"
    )
    assert payload["fixtures"]["D2"]["evaluation_dataset"]["sha256"] == (
        "f49c26acaa86a9b3785c29553e6ec0dcf41c7cbf799c61170932b1091b88d441"
    )
    assert payload["fixtures"]["D1"]["training_archive"]["sha256"] == (
        "30601eceed1fa5590013aa9eee877b055a8e75986231edbd5045e421c083a201"
    )
    assert set(payload["arms"]) == {f"K{index}" for index in range(6)}
    assert payload["timing_evidence"]["probe_contract"] == {
        "path": str(paths["probe"]),
        "sha256": builder.file_sha(paths["probe"]),
    }
    assert [
        row["path"] for row in payload["timing_evidence"]["measurement_captures"]
    ] == [
        str(paths["measurement-a"]),
        str(paths["measurement-b"]),
        str(paths["measurement-c"]),
    ]


def test_arm_staged_file_tamper_fails_closed(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    (tmp_path / "basis.json").write_bytes(b'{"accepted":false}\n')
    with pytest.raises(ValueError, match="identity mismatch"):
        builder.validate_arm_inputs(paths["arm_inputs"])


def test_admitted_manifest_drift_fails_closed(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    manifest = paths["admission_envelope"].parent / "fixtures/D1/fixture-manifest.json"
    manifest.write_bytes(manifest.read_bytes().replace(b"800b73", b"900b73", 1))
    with pytest.raises(ValueError, match="admitted fixture identity"):
        builder.build_payload(
            admission_envelope=paths["admission_envelope"],
            arm_inputs=paths["arm_inputs"],
            probe_plan=paths["probe"],
            margin_policy=paths["margin"],
            raw_timing=paths["raw"],
            end_to_end_validation=paths["e2e-validation"],
            measurement_captures=(
                paths["measurement-a"],
                paths["measurement-b"],
                paths["measurement-c"],
            ),
            heldout_captures=(paths["heldout-capture"],),
            heldout_run_records=(paths["heldout-run"],),
        )


def test_publish_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "payload.json"
    path, digest = builder.publish_create_only(output, {"value": 1})
    assert path == output
    assert digest == builder.file_sha(output)
    with pytest.raises(FileExistsError):
        builder.publish_create_only(output, {"value": 2})
