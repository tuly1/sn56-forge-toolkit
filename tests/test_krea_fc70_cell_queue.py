"""Deterministic matrix tests for the external fc70 cell controller."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from campaign_tools import krea_fc70_cell_queue as queue
from forge.data.schema import ImageSpec
from ops.calibration import krea_budget
from ops.calibration import run_krea_ladder as runner


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _canonical(path: Path, value: dict) -> None:
    path.write_bytes(queue._canonical_bytes(value) + b"\n")


def _throughput_profile() -> dict:
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
        base_model_identity_sha256=_sha("base-model"),
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
    return krea_budget.seal_throughput_profile(
        execution_envelope=envelope,
        raw_sample_manifest_sha256=_sha("raw-timing"),
        startup_sample_count=3,
        update_sample_count=100,
        save_sample_count=8,
        startup_upper_bound_s=10,
        update_upper_bound_s=1,
        save_upper_bound_s=2,
        bound_method="observed-max-plus-predeclared-margin",
        margin_policy_sha256=_sha("margin"),
        end_to_end_validation_count=1,
        end_to_end_validation_sha256=_sha("heldout"),
        framework_stop_boundary_s=225,
        framework_stop_boundary_source_sha256=_sha("stop-boundary"),
        selection_mode="offline_post_training",
        selection_scorer_identity_sha256=None,
        selection_scoring_reserve_s=0,
        finalization_reserve_s=10,
        upload_reserve_s=10,
    )


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
    discovery_path = tmp_path / queue._TRACKED_DISCOVERY_PLAN
    discovery_path.parent.mkdir(parents=True)
    discovery_path.write_text(
        json.dumps({"training_seed_a": 42565431}, indent=2) + "\n"
    )
    runner_path = tmp_path / "ops/calibration/run_krea_ladder.py"
    runner_path.write_text("# frozen fc70 runner\n")
    profile_index = {
        "discovery_plan": {
            "path": str(discovery_path),
            "file_sha256": queue._file_sha(discovery_path),
        },
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
    throughput_profile = _throughput_profile()
    modules = {"budget": krea_budget}
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
        throughput_profile=throughput_profile,
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
    assert payload["runtime_identity_sha256"] == throughput_profile[
        "execution_envelope"
    ]["runtime_identity_sha256"]
    assert payload["execution_envelope_sha256"] == throughput_profile[
        "execution_envelope"
    ]["execution_envelope_sha256"]


def test_tracked_discovery_plan_rejects_wrong_path_and_digest(
    tmp_path: Path,
) -> None:
    tracked_path = tmp_path / queue._TRACKED_DISCOVERY_PLAN
    tracked_path.parent.mkdir(parents=True)
    tracked_path.write_text('{\n  "training_seed_a": 42565431\n}\n')
    wrong_path = tmp_path / "discovery.json"
    wrong_path.write_bytes(tracked_path.read_bytes())

    with pytest.raises(ValueError, match="not the tracked Forge plan"):
        queue._load_tracked_discovery_plan(
            wrong_path,
            forge_root=tmp_path,
            expected_file_sha256=queue._file_sha(wrong_path),
        )
    with pytest.raises(ValueError, match="differs from the profile index"):
        queue._load_tracked_discovery_plan(
            tracked_path,
            forge_root=tmp_path,
            expected_file_sha256=_sha("wrong discovery plan"),
        )


def test_tracked_discovery_plan_rejects_malformed_json(tmp_path: Path) -> None:
    tracked_path = tmp_path / queue._TRACKED_DISCOVERY_PLAN
    tracked_path.parent.mkdir(parents=True)
    tracked_path.write_text("{not-json}\n")

    with pytest.raises(ValueError, match="is not JSON"):
        queue._load_tracked_discovery_plan(
            tracked_path,
            forge_root=tmp_path,
            expected_file_sha256=queue._file_sha(tracked_path),
        )


def test_initial_queue_excludes_correction_gated_d2_k4() -> None:
    assert queue._INITIAL_CELLS[0] == "D1-K1"
    assert set(queue._INITIAL_CELLS) == set(queue._CELLS) - {"D2-K4"}


def test_queue_uses_the_runners_required_system_python_entry() -> None:
    assert queue._RUNNER_INITIAL_PYTHON == "/usr/bin/python3"
    source = Path(runner.__file__).read_text()
    assert 'os.path.samefile(sys.executable, "/usr/bin/python3")' in source
    assert "/app/venv/bin/python" not in queue._RUNNER_INITIAL_PYTHON


def test_training_archive_is_atomically_staged_at_real_imagespec_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "training.zip"
    source.write_bytes(b"admitted training archive")
    source_sha = queue._file_sha(source)
    fixture = tmp_path / "fixture.json"
    _canonical(
        fixture,
        {"training_archive": {"sha256": source_sha, "bytes": source.stat().st_size}},
    )
    plan = tmp_path / "plan.json"
    _canonical(
        plan,
        {
            "task_id": "week5-krea-D1-K1",
            "expected_repo_name": "week5-krea-D1-K1",
            "base_model": {"model_id": "krea/Krea-2-Raw"},
            "training_archive": {"path": str(source), "sha256": source_sha},
            "fixture_manifest": {
                "path": str(fixture),
                "sha256": queue._file_sha(fixture),
            },
        },
    )
    cache = tmp_path / "cache/datasets"
    cache.mkdir(parents=True)
    monkeypatch.setattr(queue, "_DATASET_CACHE_ROOT", cache)
    target = queue._stage_training_archive(
        {"plan": {"path": str(plan), "sha256": queue._file_sha(plan)}},
        ImageSpec,
    )
    assert target == cache / "week5-krea-D1-K1_tourn.zip"
    assert target.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        queue._stage_training_archive(
            {"plan": {"path": str(plan), "sha256": queue._file_sha(plan)}},
            ImageSpec,
        )


def test_dataset_cleanup_removes_only_exact_transient_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    preserved = dataset / "configs"
    preserved.mkdir()
    for name in queue._DATASET_WORK_NAMES:
        work = dataset / name
        work.mkdir()
        (work / "data").write_text("x")
    monkeypatch.setattr(queue, "_DATASET_ROOT", dataset)
    queue._clean_dataset_work_paths()
    assert preserved.is_dir()
    assert all(not (dataset / name).exists() for name in queue._DATASET_WORK_NAMES)


def test_controller_is_sequential_and_stops_on_first_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    rows = [
        {"cell_id": "D1-K0", "campaign_dir": str(runs / "D1-K0"), "argv": ["a"]},
        {"cell_id": "D1-K1", "campaign_dir": str(runs / "D1-K1"), "argv": ["b"]},
    ]
    monkeypatch.setattr(queue, "validate_queue", lambda _path: {"cells": rows})
    monkeypatch.setattr(queue, "_modules", lambda _path: {"image_spec": ImageSpec})
    monkeypatch.setattr(queue, "_stage_training_archive", lambda *_args: None)
    monkeypatch.setattr(queue, "_verify_completed_condition", lambda _row: None)
    cleanup_called = False

    def cleanup():
        nonlocal cleanup_called
        cleanup_called = True

    monkeypatch.setattr(queue, "_clean_dataset_work_paths", cleanup)
    observed = []

    def fail_first(argv, *, check):
        assert check is True
        observed.append(argv)
        raise RuntimeError("stop")

    monkeypatch.setattr(queue.subprocess, "run", fail_first)
    with pytest.raises(RuntimeError, match="stop"):
        queue.run_queue(tmp_path / "queue.json")
    assert observed == [["a"]]
    assert cleanup_called is False
    assert (runs / "D1-K0").is_dir()
    assert not (runs / "D1-K1").exists()


def test_two_rows_stage_run_verify_and_clean_before_next_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    rows = [
        {"cell_id": cell, "campaign_dir": str(runs / cell), "argv": [cell]}
        for cell in ("D1-K0", "D1-K1")
    ]
    events: list[str] = []
    monkeypatch.setattr(queue, "validate_queue", lambda _path: {"cells": rows})
    monkeypatch.setattr(queue, "_modules", lambda _path: {"image_spec": ImageSpec})
    monkeypatch.setattr(
        queue,
        "_stage_training_archive",
        lambda row, _spec: events.append(f"stage:{row['cell_id']}"),
    )
    monkeypatch.setattr(
        queue.subprocess,
        "run",
        lambda argv, *, check: events.append(f"run:{argv[0]}") if check else None,
    )
    monkeypatch.setattr(
        queue,
        "_verify_completed_condition",
        lambda row: events.append(f"verify:{row['cell_id']}"),
    )
    monkeypatch.setattr(
        queue, "_clean_dataset_work_paths", lambda: events.append("clean")
    )
    queue.run_queue(tmp_path / "queue.json")
    assert events == [
        "stage:D1-K0",
        "run:D1-K0",
        "verify:D1-K0",
        "clean",
        "stage:D1-K1",
        "run:D1-K1",
        "verify:D1-K1",
        "clean",
    ]
