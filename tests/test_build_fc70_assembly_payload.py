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
    return {"arm_inputs": arm_inputs, "probe": probe, **evidence}


def test_build_payload_binds_exact_observed_files(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    payload = builder.build_payload(
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
    assert payload["fixtures"] == builder.FIXTURES
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


def test_publish_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "payload.json"
    path, digest = builder.publish_create_only(output, {"value": 1})
    assert path == output
    assert digest == builder.file_sha(output)
    with pytest.raises(FileExistsError):
        builder.publish_create_only(output, {"value": 2})

