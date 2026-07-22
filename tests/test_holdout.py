"""ML-free contract tests for the Gate-B held-out proxy producer."""

from __future__ import annotations

import hashlib
import io
import json
import os
import signal
import sqlite3
import struct
import subprocess
import time

from PIL import Image
import pytest

from forge import config, recipe
from forge.clock import Deadline
from forge.data import dataset
from forge.data.schema import ImageSpec
from forge.tasks import aitoolkit, checkpoints, holdout


def _png(path, color):
    Image.new("RGB", (32, 32), color).save(path)


def _write_pair(root, name, color, caption):
    root.mkdir(parents=True, exist_ok=True)
    _png(root / f"{name}.png", color)
    (root / f"{name}.txt").write_bytes(caption)


def _write_st(path, tag="x"):
    payload = tag.encode("utf-8")
    header = json.dumps(
        {
            "weight": {
                "dtype": "U8",
                "shape": [len(payload)],
                "data_offsets": [0, len(payload)],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    data = struct.pack("<Q", len(header)) + header + payload
    path.write_bytes(data)
    return data


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec(tmp_path, monkeypatch):
    spec = ImageSpec.build(
        task_id="gate-b",
        model="krea/Krea-2-Raw",
        model_type="krea2",
        expected_repo_name="repo",
        trigger_word=None,
        dataset_zip=None,
    )
    save_root = tmp_path / "checkpoints" / "repo"
    holdout_dir = tmp_path / "holdout"
    monkeypatch.setattr(
        type(spec), "save_root", property(lambda self: str(save_root))
    )
    monkeypatch.setattr(
        type(spec), "dataset_holdout_dir", property(lambda self: str(holdout_dir))
    )
    return spec, save_root, holdout_dir


def _deadline(seconds=3600):
    now = time.monotonic()
    return Deadline(hard_stop=now + seconds, export_reserve_s=180)


def test_reserve_holdout_is_true_post_dedup_split(tmp_path):
    images = tmp_path / "images"
    reserved = tmp_path / "reserved"
    for index in range(10):
        _write_pair(
            images,
            f"image-{index}",
            (index * 10, 20, 200 - index * 10),
            f"caption-{index}".encode(),
        )

    count = dataset.reserve_holdout(str(images), holdout_dir=str(reserved))

    assert count == 1
    assert len(list(images.glob("*.png"))) == 9
    assert len(list(reserved.glob("*.png"))) == 1
    held_image = next(reserved.glob("*.png"))
    assert not (images / held_image.name).exists()
    assert (reserved / f"{held_image.stem}.txt").is_file()


def test_reserve_holdout_keeps_tiny_dataset_whole(tmp_path):
    images = tmp_path / "images"
    reserved = tmp_path / "reserved"
    for index in range(3):
        _write_pair(images, f"i{index}", (index, 1, 2), b"caption")

    assert dataset.reserve_holdout(str(images), holdout_dir=str(reserved)) == 0
    assert len(list(images.glob("*.png"))) == 3
    assert list(reserved.glob("*.png")) == []


def test_reserve_holdout_rolls_back_delete_failure(tmp_path, monkeypatch):
    images = tmp_path / "images"
    reserved = tmp_path / "reserved"
    for index in range(10):
        _write_pair(images, f"i{index}", (index, 2, 3), f"c{index}".encode())
    real_remove = os.remove
    calls = 0

    def fail_second_remove(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("forced delete failure")
        return real_remove(path)

    monkeypatch.setattr(os, "remove", fail_second_remove)

    assert dataset.reserve_holdout(str(images), holdout_dir=str(reserved)) == 0
    assert len(list(images.glob("*.png"))) == 10
    assert len(list(images.glob("*.txt"))) == 10


def test_reserve_holdout_raises_if_rollback_is_not_byte_exact(tmp_path, monkeypatch):
    images = tmp_path / "images"
    reserved = tmp_path / "reserved"
    for index in range(10):
        _write_pair(images, f"i{index}", (index, 2, 3), f"c{index}".encode())
    real_remove = os.remove
    real_copy2 = dataset.shutil.copy2
    remove_calls = 0

    def fail_second_remove(path):
        nonlocal remove_calls
        remove_calls += 1
        if remove_calls == 2:
            raise OSError("forced delete failure")
        return real_remove(path)

    def fail_restore(source, destination, *args, **kwargs):
        if str(source).startswith(str(reserved)) and str(destination).startswith(
            str(images)
        ):
            raise OSError("forced restore failure")
        return real_copy2(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "remove", fail_second_remove)
    monkeypatch.setattr(dataset.shutil, "copy2", fail_restore)

    with pytest.raises(RuntimeError, match="could not be restored byte-exactly"):
        dataset.reserve_holdout(str(images), holdout_dir=str(reserved))


def test_reserve_holdout_rejects_ambiguous_shared_caption_stem(tmp_path):
    images = tmp_path / "images"
    reserved = tmp_path / "reserved"
    images.mkdir()
    _png(images / "same.png", (1, 2, 3))
    Image.new("RGB", (32, 32), (4, 5, 6)).save(images / "same.jpg")
    (images / "same.txt").write_bytes(b"one shared caption")

    assert dataset.reserve_holdout(str(images), holdout_dir=str(reserved)) == 0
    assert (images / "same.png").is_file()
    assert (images / "same.jpg").is_file()
    assert (images / "same.txt").read_bytes() == b"one shared caption"


def test_probe_datasets_score_captioned_and_blank_strata_separately(tmp_path):
    source = tmp_path / "holdout"
    target = tmp_path / "probe"
    _write_pair(source, "a", (1, 2, 3), b"exact caption")
    _write_pair(source, "b", (4, 5, 6), b'{"prompt":"json"}')

    captioned, blank, count = holdout._build_probe_datasets(
        str(source), str(target)
    )

    assert count == 2
    captioned_values = [
        path.read_bytes() for path in sorted((tmp_path / "probe/captioned").glob("*.txt"))
    ]
    blank_values = [
        path.read_bytes() for path in sorted((tmp_path / "probe/blank").glob("*.txt"))
    ]
    assert captioned == str(tmp_path / "probe/captioned")
    assert blank == str(tmp_path / "probe/blank")
    assert captioned_values == [b"exact caption", b'{"prompt":"json"}']
    assert blank_values == [b"", b""]


def test_probe_config_is_isolated_zero_lr_and_deterministic(tmp_path):
    cfg = {
        "job": "extension",
        "config": {
            "name": "repo",
            "process": [
                {
                    "type": "diffusion_trainer",
                    "training_folder": "/app/checkpoints/task",
                    "trigger_word": "TOK",
                    "network": {"type": "lora", "linear": 32},
                    "datasets": [
                        {
                            "folder_path": "/dataset/images",
                            "caption_dropout_rate": 0.05,
                            "resolution": [512, 768, 1024],
                            "flip_x": True,
                        }
                    ],
                    "train": {
                        "steps": 100,
                        "lr": 1e-4,
                        "optimizer_params": {"weight_decay": 0.1},
                    },
                    "save": {"save_every": 20},
                    "logging": {"use_ui_logger": True},
                }
            ],
        },
    }
    candidate = tmp_path / "candidate.safetensors"
    candidate.write_bytes(b"not read by config builder")
    out = holdout._probe_config(
        cfg,
        candidate=str(candidate),
        probe_dir=str(tmp_path / "probe"),
        candidate_root=str(tmp_path / "isolated"),
        expected_points=8,
    )
    process = out["config"]["process"][0]

    assert cfg["config"]["name"] == "repo"  # source was not mutated
    assert out["config"]["name"] == "probe"
    assert process["training_folder"].startswith(str(tmp_path / "isolated"))
    assert process["network"]["pretrained_lora_path"] == str(candidate)
    assert process["network"]["dropout"] == 0.0
    assert process["trigger_word"] is None
    assert process["train"]["steps"] == 9  # one unlogged warm-up + 8 points
    assert process["train"]["lr"] == 0.0
    assert process["train"]["optimizer_params"]["weight_decay"] == 0.0
    assert process["datasets"][0]["caption_dropout_rate"] == 0.0
    assert process["datasets"][0]["resolution"] == [512]
    assert process["datasets"][0]["flip_x"] is False


def test_complete_producer_manifest_drives_selection(tmp_path, monkeypatch):
    spec, save_root, holdout_dir = _spec(tmp_path, monkeypatch)
    save_root.mkdir(parents=True)
    _write_pair(holdout_dir, "held", (1, 2, 3), b"caption")
    scope = checkpoints.begin_run(str(save_root), "repo")
    scope = checkpoints.set_planned_steps(
        str(save_root), scope, 300, model_type="krea2"
    )
    early = save_root / "repo_000000100.safetensors"
    middle = save_root / "repo_000000200.safetensors"
    final = save_root / "repo.safetensors"
    _write_st(early, "early")
    _write_st(middle, "middle")
    _write_st(final, "final")
    monkeypatch.setenv("FORGE_HOLDOUT_SELECTION_TYPES", "krea2")
    monkeypatch.setitem(
        checkpoints._HELDOUT_PROXY_POLICIES,
        ("heldout_diffusion_loss_proxy_v2", "krea2"),
        {
            "name": "test-margin",
            "calibration_id": "unit-test-only",
            "absolute_floor": 0.01,
            "relative_floor": 0.03,
            "dispersion_multiplier": 2.0,
            "min_holdout_pairs": 1,
            "max_holdout_pairs": 4,
            "seed": 42565431,
            "direction": "min",
            "captioned_weight": 0.25,
            "blank_caption_weight": 0.75,
            "probe_epochs": 2,
            "reference_sources": ["exact_final", "highest_valid_periodic"],
        },
    )
    scores = {early.name: 0.3, middle.name: 0.1, final.name: 0.2}

    def fake_scorer(**kwargs):
        score = scores[os.path.basename(kwargs["path"])]
        points = kwargs["expected_stratum_points"]
        return {
            "score": score,
            "points": points * 2,
            "captioned_score": score,
            "blank_caption_score": score,
            "captioned_points": points,
            "blank_points": points,
            "captioned_stddev": 0.0,
            "blank_stddev": 0.0,
        }

    assert holdout.produce(
        spec,
        {"config": {}},
        scope,
        _deadline(),
        holdout_pairs=1,
        scorer=fake_scorer,
    )
    manifest_path = save_root / "forge_holdout_scores.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["complete"] is True
    assert manifest["schema"] == 1
    assert manifest["metric"] == "heldout_diffusion_loss_proxy_v2"
    assert manifest["proxy_not_validator_metric"] is True
    assert manifest["task_id"] == spec.task_id
    assert manifest["expected_repo_name"] == spec.expected_repo_name
    assert manifest["attempt_nonce"] == scope["attempt_nonce"]
    assert manifest["scope_started_unix"] == scope["started_unix"]
    assert {row["checkpoint"] for row in manifest["scores"]} == set(scores)
    assert all(row["sha256"] == _sha256(save_root / row["checkpoint"])
               for row in manifest["scores"])

    record = checkpoints.finalize(str(save_root), "repo", scope)
    assert record["source"] == "heldout_manifest"
    assert record["selected_file"] == middle.name
    assert (save_root / "last.safetensors").read_bytes() == middle.read_bytes()


def test_proxy_manifest_without_frozen_policy_retains_exact_final(
    tmp_path, monkeypatch
):
    spec, save_root, holdout_dir = _spec(tmp_path, monkeypatch)
    save_root.mkdir(parents=True)
    _write_pair(holdout_dir, "held", (1, 2, 3), b"caption")
    scope = checkpoints.begin_run(str(save_root), "repo")
    scope = checkpoints.set_planned_steps(
        str(save_root), scope, 200, model_type="krea2"
    )
    early = save_root / "repo_000000100.safetensors"
    final = save_root / "repo.safetensors"
    _write_st(early, "early")
    _write_st(final, "final")
    monkeypatch.setenv("FORGE_HOLDOUT_SELECTION_TYPES", "krea2")
    scores = {early.name: 0.01, final.name: 0.20}

    def fake_scorer(**kwargs):
        score = scores[os.path.basename(kwargs["path"])]
        points = kwargs["expected_stratum_points"]
        return {
            "score": score,
            "points": points * 2,
            "captioned_score": score,
            "blank_caption_score": score,
            "captioned_points": points,
            "blank_points": points,
            "captioned_stddev": 0.0,
            "blank_stddev": 0.0,
        }

    assert holdout.produce(
        spec,
        {"config": {}},
        scope,
        _deadline(),
        holdout_pairs=1,
        scorer=fake_scorer,
    )

    record = checkpoints.finalize(str(save_root), "repo", scope)

    assert record["source"] == "exact_final"
    assert record["metric_is_proxy_not_validator_metric"] is False
    assert (save_root / "last.safetensors").read_bytes() == final.read_bytes()


def test_calibrated_policy_can_rank_periodic_only_deadline_candidates(
    tmp_path, monkeypatch
):
    spec, save_root, holdout_dir = _spec(tmp_path, monkeypatch)
    save_root.mkdir(parents=True)
    _write_pair(holdout_dir, "held", (1, 2, 3), b"caption")
    scope = checkpoints.begin_run(str(save_root), "repo")
    scope = checkpoints.set_planned_steps(
        str(save_root), scope, 200, model_type="krea2"
    )
    early = save_root / "repo_000000100.safetensors"
    late = save_root / "repo_000000200.safetensors"
    _write_st(early, "early")
    _write_st(late, "late")
    monkeypatch.setenv("FORGE_HOLDOUT_SELECTION_TYPES", "krea2")
    monkeypatch.setitem(
        checkpoints._HELDOUT_PROXY_POLICIES,
        ("heldout_diffusion_loss_proxy_v2", "krea2"),
        {
            "name": "test-periodic-margin",
            "calibration_id": "unit-test-only",
            "absolute_floor": 0.01,
            "relative_floor": 0.03,
            "dispersion_multiplier": 2.0,
            "min_holdout_pairs": 1,
            "max_holdout_pairs": 4,
            "seed": 42565431,
            "direction": "min",
            "captioned_weight": 0.25,
            "blank_caption_weight": 0.75,
            "probe_epochs": 2,
            "reference_sources": ["highest_valid_periodic"],
        },
    )
    scores = {early.name: 0.10, late.name: 0.20}

    def fake_scorer(**kwargs):
        score = scores[os.path.basename(kwargs["path"])]
        points = kwargs["expected_stratum_points"]
        return {
            "score": score,
            "points": points * 2,
            "captioned_score": score,
            "blank_caption_score": score,
            "captioned_points": points,
            "blank_points": points,
            "captioned_stddev": 0.0,
            "blank_stddev": 0.0,
        }

    assert holdout.produce(
        spec,
        {"config": {}},
        scope,
        _deadline(),
        holdout_pairs=1,
        scorer=fake_scorer,
    )
    record = checkpoints.finalize(str(save_root), "repo", scope)

    assert record["source"] == "heldout_manifest"
    assert record["reference_file"] == late.name
    assert record["selected_file"] == early.name
    assert (save_root / "last.safetensors").read_bytes() == early.read_bytes()


def test_calibrated_policy_rejects_inconsistent_stratum_score(
    tmp_path, monkeypatch
):
    spec, save_root, holdout_dir = _spec(tmp_path, monkeypatch)
    save_root.mkdir(parents=True)
    _write_pair(holdout_dir, "held", (1, 2, 3), b"caption")
    scope = checkpoints.begin_run(str(save_root), "repo")
    scope = checkpoints.set_planned_steps(
        str(save_root), scope, 200, model_type="krea2"
    )
    early = save_root / "repo_000000100.safetensors"
    final = save_root / "repo.safetensors"
    _write_st(early, "early")
    _write_st(final, "final")
    monkeypatch.setenv("FORGE_HOLDOUT_SELECTION_TYPES", "krea2")
    monkeypatch.setitem(
        checkpoints._HELDOUT_PROXY_POLICIES,
        ("heldout_diffusion_loss_proxy_v2", "krea2"),
        {
            "name": "test-margin",
            "calibration_id": "unit-test-only",
            "absolute_floor": 0.01,
            "relative_floor": 0.03,
            "dispersion_multiplier": 2.0,
            "min_holdout_pairs": 1,
            "max_holdout_pairs": 4,
            "seed": 42565431,
            "direction": "min",
            "captioned_weight": 0.25,
            "blank_caption_weight": 0.75,
            "probe_epochs": 2,
            "reference_sources": ["exact_final"],
        },
    )

    def fake_scorer(**kwargs):
        score = 0.10 if kwargs["path"] == str(early) else 0.20
        points = kwargs["expected_stratum_points"]
        return {
            "score": score,
            "points": points * 2,
            "captioned_score": score,
            "blank_caption_score": score,
            "captioned_points": points,
            "blank_points": points,
            "captioned_stddev": 0.0,
            "blank_stddev": 0.0,
        }

    assert holdout.produce(
        spec,
        {"config": {}},
        scope,
        _deadline(),
        holdout_pairs=1,
        scorer=fake_scorer,
    )
    manifest_path = save_root / "forge_holdout_scores.json"
    manifest = json.loads(manifest_path.read_text())
    next(row for row in manifest["scores"] if row["checkpoint"] == early.name)[
        "score"
    ] = 0.0
    manifest_path.write_text(json.dumps(manifest))

    record = checkpoints.finalize(str(save_root), "repo", scope)

    assert record["source"] == "exact_final"
    assert (save_root / "last.safetensors").read_bytes() == final.read_bytes()


def test_producer_hash_drift_leaves_no_manifest(tmp_path, monkeypatch):
    spec, save_root, holdout_dir = _spec(tmp_path, monkeypatch)
    save_root.mkdir(parents=True)
    _write_pair(holdout_dir, "held", (1, 2, 3), b"caption")
    scope = checkpoints.begin_run(str(save_root), "repo")
    first = save_root / "repo_000000100.safetensors"
    final = save_root / "repo.safetensors"
    _write_st(first, "first")
    _write_st(final, "final")
    monkeypatch.setenv("FORGE_HOLDOUT_SELECTION_TYPES", "krea2")
    calls = 0

    def drifting_scorer(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            _write_st(first, "changed")
        points = kwargs["expected_stratum_points"]
        return {
            "score": float(calls),
            "points": points * 2,
            "captioned_score": float(calls),
            "blank_caption_score": float(calls),
            "captioned_points": points,
            "blank_points": points,
        }

    assert not holdout.produce(
        spec,
        {"config": {}},
        scope,
        _deadline(),
        holdout_pairs=1,
        scorer=drifting_scorer,
    )
    assert not (save_root / "forge_holdout_scores.json").exists()


def test_producer_worker_failure_leaves_no_manifest(tmp_path, monkeypatch):
    spec, save_root, holdout_dir = _spec(tmp_path, monkeypatch)
    save_root.mkdir(parents=True)
    _write_pair(holdout_dir, "held", (1, 2, 3), b"caption")
    scope = checkpoints.begin_run(str(save_root), "repo")
    _write_st(save_root / "repo_000000100.safetensors", "first")
    _write_st(save_root / "repo.safetensors", "final")
    monkeypatch.setenv("FORGE_HOLDOUT_SELECTION_TYPES", "krea2")

    def failed_scorer(**_kwargs):
        raise RuntimeError("forced scorer failure")

    assert not holdout.produce(
        spec,
        {"config": {}},
        scope,
        _deadline(),
        holdout_pairs=1,
        scorer=failed_scorer,
    )
    assert not (save_root / "forge_holdout_scores.json").exists()


def test_proxy_is_dormant_without_per_arch_allowlist(tmp_path, monkeypatch):
    spec, save_root, holdout_dir = _spec(tmp_path, monkeypatch)
    save_root.mkdir(parents=True)
    _write_pair(holdout_dir, "held", (1, 2, 3), b"caption")
    scope = checkpoints.begin_run(str(save_root), "repo")
    _write_st(save_root / "repo_000000100.safetensors", "first")
    _write_st(save_root / "repo.safetensors", "final")
    monkeypatch.delenv("FORGE_HOLDOUT_SELECTION_TYPES", raising=False)

    assert not holdout.produce(
        spec,
        {"config": {}},
        scope,
        _deadline(),
        holdout_pairs=1,
        scorer=lambda **_kwargs: {},
    )
    assert not (save_root / "forge_holdout_scores.json").exists()


def test_loss_reader_returns_ordered_ui_logger_points(tmp_path):
    db = tmp_path / "loss_log.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE metrics (
                step INTEGER NOT NULL, key TEXT NOT NULL, value_real REAL,
                value_text TEXT, PRIMARY KEY (step, key)
            );
            """
        )
        conn.executemany(
            "INSERT INTO metrics VALUES (?, 'loss/loss', ?, NULL)",
            [(2, 0.3), (0, 0.1), (1, 0.2)],
        )
    assert holdout._loss_values(str(db)) == [0.1, 0.2, 0.3]


def test_manifest_cleanup_failure_never_escapes_finalization(tmp_path, monkeypatch):
    spec, save_root, holdout_dir = _spec(tmp_path, monkeypatch)
    save_root.mkdir(parents=True)
    _write_pair(holdout_dir, "held", (1, 2, 3), b"caption")
    scope = checkpoints.begin_run(str(save_root), "repo")
    monkeypatch.setenv("FORGE_HOLDOUT_SELECTION_TYPES", "krea2")
    monkeypatch.setattr(
        holdout,
        "_remove_manifest",
        lambda _path: (_ for _ in ()).throw(PermissionError("read-only")),
    )

    assert not holdout.produce(
        spec,
        {"config": {}},
        scope,
        _deadline(),
        holdout_pairs=1,
    )


def test_proxy_terminate_falls_back_to_direct_signal(monkeypatch):
    class FakeProc:
        pid = 1234

        def __init__(self):
            self.signals = []
            self.waits = 0

        def send_signal(self, value):
            self.signals.append(value)

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("probe", timeout)
            return -signal.SIGKILL

        def poll(self):
            return -signal.SIGKILL

    proc = FakeProc()
    monkeypatch.setattr(os, "getpgid", lambda _pid: 1234)
    monkeypatch.setattr(
        os,
        "killpg",
        lambda _pgid, _sig: (_ for _ in ()).throw(PermissionError("denied")),
    )

    holdout._terminate(proc)

    assert proc.signals == [signal.SIGTERM, signal.SIGKILL]


def test_scoring_budget_cannot_enable_after_reserve_boundary():
    decision = aitoolkit._latch_scoring_decision(
        None,
        remaining=1000.0,
        reserve_s=900.0,
        candidates_ready=False,
    )
    assert decision is None
    decision = aitoolkit._latch_scoring_decision(
        decision,
        remaining=944.0,
        reserve_s=900.0,
        candidates_ready=False,
    )
    assert decision is False
    decision = aitoolkit._latch_scoring_decision(
        decision,
        remaining=500.0,
        reserve_s=900.0,
        candidates_ready=True,
    )
    assert decision is False


def test_holdout_budget_gate_matches_recipe_and_keeps_multiple_candidates(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FORGE_HOLDOUT_SELECTION_TYPES", "krea2")
    assert not holdout.budget_allows("krea2", 2202.0)
    assert holdout.budget_allows("krea2", 2203.0)

    spec, _save_root, _holdout_dir = _spec(tmp_path, monkeypatch)
    hard_remaining = 2203.0 + recipe.EXPORT_RESERVE_S
    planning_reserve = (
        holdout.scoring_reserve_s("krea2") + holdout.boundary_margin_s()
    ) / recipe.MARGIN
    cfg = config.build_config(
        spec,
        num_images=9,
        hours_to_complete=(hard_remaining - planning_reserve) / 3600.0,
    )
    process = cfg["config"]["process"][0]
    assert process["train"]["steps"] >= 100
    assert process["save"]["save_every"] < process["train"]["steps"]


def test_dormant_holdout_preserves_exact_recipe_budget_and_step_caps(monkeypatch):
    monkeypatch.delenv("FORGE_HOLDOUT_SELECTION_TYPES", raising=False)

    class FixedDeadline:
        def __init__(self, hard_remaining):
            self.hard_remaining = hard_remaining

        def remaining_hard(self):
            return self.hard_remaining

    cases = (
        ("ideogram4", "black-forest-labs/FLUX.1-Krea-dev", 9, 0.25, 86),
        ("krea2", "krea/Krea-2-Raw", 24, 0.5, 300),
    )
    for model_type, model, images, hours, expected_steps in cases:
        assert not holdout.enabled_for(model_type)
        reserve = holdout.scoring_reserve_s(model_type)
        assert reserve == 0.0
        planned_hours = aitoolkit._recipe_hours(
            FixedDeadline(hours * 3600.0),
            reserve,
        )
        assert planned_hours == hours

        spec = ImageSpec.build(
            task_id=f"dormant-{model_type}",
            model=model,
            model_type=model_type,
            expected_repo_name="repo",
            trigger_word=None,
            dataset_zip=None,
        )
        baseline = config.build_config(
            spec,
            num_images=images,
            hours_to_complete=hours,
        )
        observed = config.build_config(
            spec,
            num_images=images,
            hours_to_complete=planned_hours,
        )
        assert observed == baseline
        process = observed["config"]["process"][0]
        assert process["train"]["steps"] == expected_steps


def test_active_holdout_recipe_budget_includes_reserve_and_margin():
    class FixedDeadline:
        def remaining_hard(self):
            return 2203.0 + recipe.EXPORT_RESERVE_S

    scorer_reserve = 900.0
    expected = (
        FixedDeadline().remaining_hard()
        - (scorer_reserve + holdout.boundary_margin_s()) / recipe.MARGIN
    ) / 3600.0
    assert aitoolkit._recipe_hours(FixedDeadline(), scorer_reserve) == expected
