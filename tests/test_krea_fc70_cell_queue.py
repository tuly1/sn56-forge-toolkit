"""Deterministic matrix tests for the external fc70 cell controller."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from campaign_tools import krea_fc70_cell_queue as queue


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _canonical(path: Path, value: dict) -> None:
    path.write_bytes(queue._canonical_bytes(value) + b"\n")


def _spec() -> dict:
    recipe = {
        "schema": 1,
        "kind": "forge-krea-normalized-recipe",
        "fields": {
            "planned_steps": {
                "classification": "unknown_source_fixed",
                "source_pointers": [],
                "source_value": None,
                "effective_value": 1,
                "evidence": "frozen local execution choice",
            },
            "save_cadence": {
                "classification": "unknown_source_fixed",
                "source_pointers": [],
                "source_value": None,
                "effective_value": 1,
                "evidence": "frozen local execution choice",
            },
        },
    }
    payload = {
        "schema": 1,
        "kind": queue._KIND,
        "task_id_prefix": "week5-krea",
        "expected_repo_prefix": "week5-krea",
        "timing_evidence": {"bound": True},
        "base_model": {"bound": True},
        "fixtures": {
            fixture: {
                "training_archive": {
                    "path": f"/campaign/{fixture}/training.zip",
                    "sha256": _sha(f"{fixture}-archive"),
                },
                "evaluation_dataset": {
                    "path": f"/campaign/{fixture}/evaluation",
                    "sha256": _sha(f"{fixture}-evaluation"),
                },
            }
            for fixture in ("D1", "D2")
        },
        "arms": {
            f"K{arm}": {
                "arm_basis": {"arm": f"K{arm}"},
                "execution_recipe": recipe,
            }
            for arm in range(6)
        },
    }
    return queue.seal_spec(payload)


@pytest.mark.parametrize("cell_id", queue._CELLS)
def test_all_twelve_cells_are_assembled_from_exact_controls(
    tmp_path: Path, cell_id: str
) -> None:
    fixture_id, arm_id = cell_id.split("-", 1)
    discovery_path = tmp_path / "discovery.json"
    _canonical(discovery_path, {"training_seed_a": 42565431})
    runner_path = tmp_path / "ops/calibration/run_krea_ladder.py"
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("# frozen fc70 runner\n")
    profile_index = {
        "discovery_plan": {"path": str(discovery_path)},
        "discovery_execution_authorization": {"bound": True},
        "fixtures": {
            fixture: {
                "manifest": {
                    "path": f"/campaign/{fixture}/manifest.json",
                    "file_sha256": _sha(f"{fixture}-manifest"),
                },
                "approval": {
                    "path": f"/campaign/{fixture}/approval.json",
                    "file_sha256": _sha(f"{fixture}-approval"),
                },
            }
            for fixture in ("D1", "D2")
        },
        "index_sha256": _sha("index"),
    }
    classes = {
        "K0": "A",
        "K1": "A",
        "K2": "A",
        "K3": "B",
        "K4": "C",
        "K5": "A",
    }
    controls = {
        "cell": {
            "cell_id": cell_id,
            "throughput_equivalence_class": classes[arm_id],
        },
        "recipe_overrides": {"planned_steps": 700, "save_cadence": 91},
        "budget_plan": {"hard_budget_s": "2700"},
        "budget_plan_sha256": _sha("budget"),
        "schedule": {"planned_steps": 700, "save_every": 91},
    }
    profile = SimpleNamespace(
        runtime_identity_sha256=_sha("runtime"),
        execution_envelope=SimpleNamespace(
            execution_envelope_sha256=_sha("envelope")
        ),
    )
    modules = {"budget": SimpleNamespace(load_throughput_profile=lambda _v: profile)}
    host_path = tmp_path / "host.json"
    profile_path = tmp_path / "profile.json"
    index_path = tmp_path / "index.json"
    for path in (host_path, profile_path, index_path):
        _canonical(path, {})

    payload = queue._cell_payload(
        cell_id=cell_id,
        spec=_spec(),
        controls=controls,
        profile_index=profile_index,
        profile_index_path=index_path,
        profile_index_file_sha=_sha("index-file"),
        throughput_profile={},
        throughput_profile_path=profile_path,
        throughput_profile_file_sha=_sha("profile-file"),
        host_manifest_path=host_path,
        host_manifest_file_sha=_sha("host-file"),
        forge_root=tmp_path,
        modules=modules,
    )

    assert payload["arm_id"] == arm_id
    assert payload["discovery_fixture_id"] == fixture_id
    assert payload["seed"] == 42565431
    assert payload["throughput_equivalence_class"] == classes[arm_id]
    assert payload["budget_plan"] == controls["budget_plan"]
    assert payload["schedule"] == controls["schedule"]
    assert payload["execution_recipe"]["fields"]["planned_steps"][
        "effective_value"
    ] == 700
    assert payload["execution_recipe"]["fields"]["save_cadence"][
        "effective_value"
    ] == 91
    assert payload["predeclared_recipe_axes"] == queue._AXES[arm_id]
    assert payload["fixture_manifest"]["path"].endswith(
        f"/{fixture_id}/manifest.json"
    )
    assert payload["runner_sha256"] == queue._file_sha(runner_path)


def test_initial_queue_excludes_correction_gated_d2_k4() -> None:
    assert queue._INITIAL_CELLS[0] == "D1-K1"
    assert set(queue._INITIAL_CELLS) == set(queue._CELLS) - {"D2-K4"}


def test_controller_is_sequential_and_stops_on_first_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    rows = [
        {"cell_id": "D1-K0", "campaign_dir": str(runs / "D1-K0"), "argv": ["a"]},
        {"cell_id": "D1-K1", "campaign_dir": str(runs / "D1-K1"), "argv": ["b"]},
    ]
    monkeypatch.setattr(queue, "validate_queue", lambda _path: {"cells": rows})
    monkeypatch.setattr(queue, "_modules", lambda _path: {})
    observed = []

    def fail_first(argv, *, check):
        assert check is True
        observed.append(argv)
        raise RuntimeError("stop")

    monkeypatch.setattr(queue.subprocess, "run", fail_first)
    with pytest.raises(RuntimeError, match="stop"):
        queue.run_queue(tmp_path / "queue.json")
    assert observed == [["a"]]
    assert (runs / "D1-K0").is_dir()
    assert not (runs / "D1-K1").exists()
