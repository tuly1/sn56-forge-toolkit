from __future__ import annotations

from pathlib import Path

import pytest

from ops.calibration import krea_stage2_endgame_supervisor as supervisor


def _payload(tmp_path: Path) -> dict:
    files = {}
    for name in ("matrix", "plan-set", "authority", "score-config"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n")
        files[name] = str(path)
    return {
        "matrix": files["matrix"],
        "plan_set": files["plan-set"],
        "authority_bundle": files["authority"],
        "training_claims_root": str(tmp_path / "training-claims"),
        "score_config": files["score-config"],
        "score_output_root": str(tmp_path / "scores"),
        "score_claims_root": str(tmp_path / "score-claims"),
        "gpu_lock_root": str(tmp_path / "gpu-locks"),
        "worker_state_root": str(tmp_path / "workers"),
        "training_gate": str(tmp_path / "training-gate.json"),
        "score_gate": str(tmp_path / "score-gate.json"),
        "deadline_utc": "2026-08-02T15:00:00Z",
        "scheduler_instance_id": "endgame-a",
        "poll_interval_seconds": 2,
    }


def test_config_generator_seals_only_absolute_bounded_launch_input(
    tmp_path: Path,
) -> None:
    record = supervisor.build_config(_payload(tmp_path))
    assert supervisor.validate_config(record) == record

    bad = _payload(tmp_path)
    bad["worker_state_root"] = "relative/workers"
    with pytest.raises(ValueError, match="absolute and normalized"):
        supervisor.build_config(bad)


def test_dispatch_reserves_one_ready_score_gpu_then_trains_on_other_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = supervisor.build_config(_payload(tmp_path))
    ready_group = tmp_path / "ready-group.json"
    ready_group.write_text("{}\n")
    queue = {
        "matrix_sha256": "matrix-sha",
        "training_plan_set_sha256": "plan-sha",
        "groups": [
            {
                "group_key": "score-C1-A-K1",
                "group_path": str(ready_group),
                "aggregate_path": str(tmp_path / "aggregate.json"),
            }
        ],
    }
    matrix = {"matrix_sha256": "matrix-sha"}
    plan_set = {"plan_set_sha256": "plan-sha", "rows": []}
    monkeypatch.setattr(supervisor, "validate_config", lambda value: dict(value))
    monkeypatch.setattr(supervisor, "_load", lambda path, _label: {})
    monkeypatch.setattr(
        supervisor.krea_stage2_endgame_matrix, "validate_matrix", lambda value: matrix
    )
    monkeypatch.setattr(
        supervisor.training, "validate_plan_set", lambda value, matrix: plan_set
    )
    monkeypatch.setattr(
        supervisor.training, "_validate_authority_bundle", lambda value: {}
    )
    monkeypatch.setattr(supervisor.scoring, "_validate_config", lambda value: {})
    monkeypatch.setattr(
        supervisor.scoring,
        "materialize_ready_score_plans",
        lambda config, output_root: {"queue": queue},
    )
    monkeypatch.setattr(supervisor.scoring, "_validate_queue", lambda value: value)
    monkeypatch.setattr(supervisor, "_outstanding_specs", lambda **kwargs: [])
    monkeypatch.setattr(
        supervisor, "_reconcile", lambda config, specs, running: (set(), set())
    )
    monkeypatch.setattr(supervisor, "_completion_counts", lambda *_args: (0, 0))
    score_devices = []
    training_devices = []

    def claim_scores(**kwargs):
        score_devices.append(kwargs["gpu_devices"])
        return [{"gpu_device": gpu} for gpu in kwargs["gpu_devices"]]

    def claim_training(**kwargs):
        training_devices.append(kwargs["gpu_devices"])
        return [{"gpu_device": gpu} for gpu in kwargs["gpu_devices"]]

    monkeypatch.setattr(supervisor.scoring, "claim_ready_groups", claim_scores)
    monkeypatch.setattr(supervisor.training, "claim_next", claim_training)

    status = supervisor.dispatch_once(
        config, running={}, now_utc="2026-08-01T21:00:00Z"
    )

    assert score_devices == [[0]]
    assert training_devices == [[1, 2, 3]]
    assert status["state"] == "running"


def test_incomplete_work_fails_at_deadline_without_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = supervisor.build_config(_payload(tmp_path))
    queue = {
        "matrix_sha256": "matrix-sha",
        "training_plan_set_sha256": "plan-sha",
        "groups": [],
    }
    matrix = {"matrix_sha256": "matrix-sha"}
    plan_set = {"plan_set_sha256": "plan-sha", "rows": []}
    monkeypatch.setattr(supervisor, "validate_config", lambda value: dict(value))
    monkeypatch.setattr(supervisor, "_load", lambda path, _label: {})
    monkeypatch.setattr(
        supervisor.krea_stage2_endgame_matrix, "validate_matrix", lambda value: matrix
    )
    monkeypatch.setattr(
        supervisor.training, "validate_plan_set", lambda value, matrix: plan_set
    )
    monkeypatch.setattr(
        supervisor.training, "_validate_authority_bundle", lambda value: {}
    )
    monkeypatch.setattr(supervisor.scoring, "_validate_config", lambda value: {})
    monkeypatch.setattr(
        supervisor.scoring,
        "materialize_ready_score_plans",
        lambda config, output_root: {"queue": queue},
    )
    monkeypatch.setattr(supervisor.scoring, "_validate_queue", lambda value: value)
    monkeypatch.setattr(supervisor, "_outstanding_specs", lambda **kwargs: [])
    monkeypatch.setattr(
        supervisor, "_reconcile", lambda config, specs, running: (set(), set())
    )
    monkeypatch.setattr(supervisor, "_completion_counts", lambda *_args: (59, 15))

    with pytest.raises(RuntimeError, match="deadline reached incomplete"):
        supervisor.dispatch_once(
            config, running={}, now_utc="2026-08-02T15:00:00Z"
        )
