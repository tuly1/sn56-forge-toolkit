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
    training_dir = tmp_path / "training"
    monkeypatch.setattr(
        type(spec), "save_root", property(lambda self: str(save_root))
    )
    monkeypatch.setattr(
        type(spec), "dataset_holdout_dir", property(lambda self: str(holdout_dir))
    )
    monkeypatch.setattr(
        type(spec), "dataset_images_dir", property(lambda self: str(training_dir))
    )
    for index in range(3):
        _write_pair(
            training_dir,
            f"train-{index}",
            (20 + index, 40 + index, 60 + index),
            f"training-caption-{index}".encode(),
        )
    return spec, save_root, holdout_dir


def _deadline(seconds=3600):
    now = time.monotonic()
    return Deadline(hard_stop=now + seconds, export_reserve_s=180)


def _bound_scope(spec, save_root, *, steps=None):
    scope = checkpoints.begin_run(
        str(save_root),
        spec.expected_repo_name,
        task_id=spec.task_id,
    )
    identity = holdout.dataset_split_identity(
        spec.dataset_images_dir,
        spec.dataset_holdout_dir,
    )
    scope = checkpoints.bind_dataset_split(
        str(save_root),
        scope,
        split_sha256=holdout.dataset_split_sha256(identity),
        training_pairs=identity["training_pairs"],
        holdout_pairs=identity["holdout_pairs"],
    )
    if steps is not None:
        scope = checkpoints.set_planned_steps(
            str(save_root), scope, steps, model_type=spec.model_type
        )
    return scope


def _test_policy(*, reference_sources=("exact_final",)):
    return {
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
        "reference_sources": list(reference_sources),
    }


def _produce_two_candidate_manifest(tmp_path, monkeypatch, *, install_policy=True):
    spec, save_root, holdout_dir = _spec(tmp_path, monkeypatch)
    save_root.mkdir(parents=True)
    _write_pair(holdout_dir, "held", (1, 2, 3), b"caption")
    scope = _bound_scope(spec, save_root, steps=200)
    early = save_root / "repo_000000100.safetensors"
    final = save_root / "repo.safetensors"
    _write_st(early, "early")
    _write_st(final, "final")
    monkeypatch.setenv("FORGE_HOLDOUT_SELECTION_TYPES", "krea2")
    if install_policy:
        monkeypatch.setitem(
            checkpoints._HELDOUT_PROXY_POLICIES,
            ("heldout_diffusion_loss_proxy_v2", "krea2"),
            _test_policy(),
        )

    def scorer(**kwargs):
        score = 0.01 if kwargs["path"] == str(early) else 0.20
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
        scorer=scorer,
    )
    return spec, save_root, scope, early, final


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


def test_pair_digest_uses_unambiguous_component_framing(tmp_path):
    first_image = tmp_path / "first.img"
    first_caption = tmp_path / "first.txt"
    second_image = tmp_path / "second.img"
    second_caption = tmp_path / "second.txt"
    first_image.write_bytes(b"a")
    first_caption.write_bytes(b"\0b")
    second_image.write_bytes(b"a\0")
    second_caption.write_bytes(b"b")

    legacy_first = first_image.read_bytes() + b"\0" + first_caption.read_bytes() + b"\0"
    legacy_second = (
        second_image.read_bytes() + b"\0" + second_caption.read_bytes() + b"\0"
    )
    assert legacy_first == legacy_second
    assert dataset._pair_digest(
        str(first_image), str(first_caption)
    ) != dataset._pair_digest(str(second_image), str(second_caption))


@pytest.mark.parametrize("entry_kind", ("orphan_caption", "non_image_symlink"))
def test_strict_dataset_inventory_rejects_every_unbound_child(
    tmp_path, entry_kind
):
    training = tmp_path / "training"
    _write_pair(training, "paired", (1, 2, 3), b"caption")
    if entry_kind == "orphan_caption":
        (training / "orphan.txt").write_bytes(b"orphan")
    else:
        outside = tmp_path / "outside.txt"
        outside.write_bytes(b"outside")
        os.symlink(outside, training / "orphan-link.txt")

    with pytest.raises(RuntimeError):
        dataset.count_flat_pairs(str(training))
    with pytest.raises(RuntimeError):
        holdout._dataset_snapshot(str(training))


def test_dataset_split_identity_is_byte_bound_ordered_and_name_free(tmp_path):
    training = tmp_path / "training"
    reserved = tmp_path / "reserved"
    _write_pair(training, "b", (10, 20, 30), b"caption-b")
    _write_pair(training, "a", (40, 50, 60), b"caption-a")
    _write_pair(reserved, "held", (70, 80, 90), b"held-caption")

    identity = holdout.dataset_split_identity(str(training), str(reserved))
    digest = holdout.dataset_split_sha256(identity)

    assert identity["training_pairs"] == 2
    assert identity["holdout_pairs"] == 1
    assert identity["total_pairs"] == 3
    assert identity["sample_disjoint"] is True
    assert identity["image_disjoint"] is True
    assert all(
        set(row) == {"sample_sha256", "image_sha256", "caption_sha256"}
        for row in identity["training"] + identity["holdout"]
    )
    assert len(digest) == 64

    (training / "a.txt").write_bytes(b"caption-a-mutated")
    mutated = holdout.dataset_split_identity(str(training), str(reserved))
    assert holdout.dataset_split_sha256(mutated) != digest
    assert mutated["training_sequence_sha256"] != identity[
        "training_sequence_sha256"
    ]


def test_dataset_split_rejects_same_image_with_different_caption(tmp_path):
    training = tmp_path / "training"
    reserved = tmp_path / "reserved"
    _write_pair(training, "train", (10, 20, 30), b"training-caption")
    _write_pair(reserved, "held", (10, 20, 30), b"different-caption")

    with pytest.raises(RuntimeError, match="identical images"):
        holdout.dataset_split_identity(str(training), str(reserved))


def test_rejected_holdout_can_be_restored_byte_exactly(tmp_path):
    training = tmp_path / "training"
    reserved = tmp_path / "reserved"
    for index in range(25):
        _write_pair(
            training,
            f"image-{index}",
            (index * 7 % 255, 20, 200 - index * 3),
            f"caption-{index}".encode(),
        )
    before = {
        path.name: path.read_bytes()
        for path in training.iterdir()
        if path.is_file()
    }
    heldout = dataset.reserve_holdout(str(training), holdout_dir=str(reserved))

    restored = dataset.restore_reserved_holdout(
        str(training), holdout_dir=str(reserved)
    )

    assert restored == heldout == 3
    assert not list(reserved.iterdir())
    assert {
        path.name: path.read_bytes()
        for path in training.iterdir()
        if path.is_file()
    } == before


def test_rejected_holdout_refuses_bytes_changed_after_reservation(tmp_path):
    training = tmp_path / "training"
    reserved = tmp_path / "reserved"
    for index in range(10):
        _write_pair(
            training,
            f"image-{index}",
            (index * 10, 20, 200 - index * 10),
            f"caption-{index}".encode(),
        )
    assert dataset.reserve_holdout(str(training), holdout_dir=str(reserved)) == 1
    heldout_caption = next(reserved.glob("*.txt"))
    heldout_caption.write_bytes(heldout_caption.read_bytes() + b"-mutated")

    with pytest.raises(RuntimeError, match="differ from the receipt"):
        dataset.restore_reserved_holdout(
            str(training), holdout_dir=str(reserved)
        )


@pytest.mark.parametrize(
    "mutation",
    ("added_pair", "orphan_image", "training_bytes", "heldout_bytes"),
)
def test_runner_aborts_if_prepared_dataset_drifts_before_split_binding(
    tmp_path, monkeypatch, mutation
):
    spec, _save_root, _holdout_dir = _spec(tmp_path, monkeypatch)
    training = tmp_path / "training"
    monkeypatch.setattr(
        type(spec),
        "training_folder",
        property(lambda self: str(tmp_path / "training-work")),
    )
    for path in training.iterdir():
        path.unlink()
    for index in range(10):
        _write_pair(
            training,
            f"image-{index}",
            (index * 10, 20, 200 - index * 10),
            f"caption-{index}".encode(),
        )

    monkeypatch.setattr(
        aitoolkit,
        "resolve_base_model",
        lambda _path: "/cache/immutable-base",
    )
    monkeypatch.setattr(
        dataset,
        "prepare_aitoolkit_dataset",
        lambda *_args, **_kwargs: (str(training), 10),
    )
    monkeypatch.setattr(holdout, "budget_allows", lambda *_args: True)
    real_reserve = dataset.reserve_holdout

    def reserve_then_mutate(images_dir, *, holdout_dir):
        count = real_reserve(images_dir, holdout_dir=holdout_dir)
        if mutation == "added_pair":
            _write_pair(training, "unexpected", (2, 4, 6), b"unexpected")
        elif mutation == "orphan_image":
            _png(training / "orphan.png", (2, 4, 6))
        elif mutation == "training_bytes":
            caption = next(training.glob("*.txt"))
            caption.write_bytes(caption.read_bytes() + b"-mutated")
        else:
            caption = next((tmp_path / "holdout").glob("*.txt"))
            caption.write_bytes(caption.read_bytes() + b"-mutated")
        return count

    monkeypatch.setattr(dataset, "reserve_holdout", reserve_then_mutate)

    with pytest.raises(RuntimeError, match="receipt|caption inventory"):
        aitoolkit.run(spec, _deadline())


def test_runner_rejects_prepared_count_not_matching_exact_contents(
    tmp_path, monkeypatch
):
    spec, _save_root, _holdout_dir = _spec(tmp_path, monkeypatch)
    training = tmp_path / "training"
    monkeypatch.setattr(
        type(spec),
        "training_folder",
        property(lambda self: str(tmp_path / "training-work")),
    )
    monkeypatch.setattr(
        aitoolkit,
        "resolve_base_model",
        lambda _path: "/cache/immutable-base",
    )
    monkeypatch.setattr(
        dataset,
        "prepare_aitoolkit_dataset",
        lambda *_args, **_kwargs: (str(training), 4),
    )

    with pytest.raises(RuntimeError, match="exact contents"):
        aitoolkit.run(spec, _deadline())


def test_runner_rechecks_training_count_after_failed_reservation(
    tmp_path, monkeypatch
):
    spec, _save_root, _holdout_dir = _spec(tmp_path, monkeypatch)
    training = tmp_path / "training"
    monkeypatch.setattr(
        type(spec),
        "training_folder",
        property(lambda self: str(tmp_path / "training-work")),
    )
    monkeypatch.setattr(
        aitoolkit,
        "resolve_base_model",
        lambda _path: "/cache/immutable-base",
    )
    monkeypatch.setattr(
        dataset,
        "prepare_aitoolkit_dataset",
        lambda *_args, **_kwargs: (str(training), 3),
    )
    monkeypatch.setattr(holdout, "budget_allows", lambda *_args: True)

    def failed_reservation_with_drift(_images_dir, *, holdout_dir):
        _write_pair(training, "unexpected", (2, 4, 6), b"unexpected")
        return 0

    monkeypatch.setattr(
        dataset,
        "reserve_holdout",
        failed_reservation_with_drift,
    )

    with pytest.raises(RuntimeError, match="changed after holdout reservation"):
        aitoolkit.run(spec, _deadline())


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
                    "training_seed": 7,
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
    assert cfg["config"]["process"][0]["training_seed"] == 7
    assert process["training_folder"].startswith(str(tmp_path / "isolated"))
    assert process["training_seed"] == holdout._SEED
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
    scope = _bound_scope(spec, save_root, steps=300)
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
    assert manifest["schema"] == 2
    assert manifest["metric"] == "heldout_diffusion_loss_proxy_v2"
    assert manifest["proxy_not_validator_metric"] is True
    assert manifest["task_id"] == spec.task_id
    assert manifest["expected_repo_name"] == spec.expected_repo_name
    assert manifest["attempt_nonce"] == scope["attempt_nonce"]
    assert manifest["scope_started_unix"] == scope["started_unix"]
    assert manifest["planned_steps"] == 300
    assert manifest["dataset_split_sha256"] == scope["dataset_split_sha256"]
    assert manifest["dataset_split"] == holdout.dataset_split_identity(
        spec.dataset_images_dir, spec.dataset_holdout_dir
    )
    assert {row["checkpoint"] for row in manifest["scores"]} == set(scores)
    assert all(row["sha256"] == _sha256(save_root / row["checkpoint"])
               for row in manifest["scores"])

    record = checkpoints.finalize(str(save_root), "repo", scope)
    assert record["source"] == "heldout_manifest"
    assert record["selected_file"] == middle.name
    assert (save_root / "last.safetensors").read_bytes() == middle.read_bytes()


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("task_id", "other-task"),
        ("expected_repo_name", "other-repo"),
        ("attempt_nonce", "0" * 32),
        ("scope_started_unix", 0.0),
        ("planned_steps", 201),
        ("dataset_split_sha256", "0" * 64),
        ("model_type", "ideogram4"),
    ),
)
def test_consumer_rejects_manifest_not_bound_to_live_scope(
    tmp_path, monkeypatch, field, replacement
):
    _spec_value, save_root, scope, _early, final = _produce_two_candidate_manifest(
        tmp_path, monkeypatch
    )
    manifest_path = save_root / "forge_holdout_scores.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = replacement
    manifest_path.write_text(json.dumps(manifest))

    record = checkpoints.finalize(str(save_root), "repo", scope)

    assert record["source"] == "exact_final"
    assert (save_root / "last.safetensors").read_bytes() == final.read_bytes()


def test_fully_bound_self_declared_exact_metric_cannot_bypass_empty_policy(
    tmp_path, monkeypatch
):
    _spec_value, save_root, scope, _early, final = _produce_two_candidate_manifest(
        tmp_path, monkeypatch, install_policy=False
    )
    manifest_path = save_root / "forge_holdout_scores.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metric"] = "validator_exact_combined"
    manifest["proxy_not_validator_metric"] = False
    manifest_path.write_text(json.dumps(manifest))

    record = checkpoints.finalize(str(save_root), "repo", scope)

    assert checkpoints._HELDOUT_PROXY_POLICIES == {}
    assert record["source"] == "exact_final"
    assert (save_root / "last.safetensors").read_bytes() == final.read_bytes()


@pytest.mark.parametrize(
    ("checkpoint_value", "step_value", "target"),
    (
        ("../repo_000000100.safetensors", 100, "early"),
        ("/tmp/repo_000000100.safetensors", 100, "early"),
        ("subdir/repo_000000100.safetensors", 100, "early"),
        ("..\\repo_000000100.safetensors", 100, "early"),
        (123, 100, "early"),
        ("repo_000000100.safetensors", None, "early"),
        ("repo_000000100.safetensors", "100", "early"),
        ("repo_000000100.safetensors", 100.0, "early"),
        ("repo_000000100.safetensors", True, "early"),
        ("repo_000000100.safetensors", 101, "early"),
        ("repo.safetensors", 199, "final"),
    ),
)
def test_consumer_rejects_unsafe_checkpoint_or_false_step(
    tmp_path, monkeypatch, checkpoint_value, step_value, target
):
    _spec_value, save_root, scope, early, final = _produce_two_candidate_manifest(
        tmp_path, monkeypatch
    )
    manifest_path = save_root / "forge_holdout_scores.json"
    manifest = json.loads(manifest_path.read_text())
    target_name = early.name if target == "early" else final.name
    row = next(row for row in manifest["scores"] if row["checkpoint"] == target_name)
    row["checkpoint"] = checkpoint_value
    row["step"] = step_value
    manifest_path.write_text(json.dumps(manifest))

    record = checkpoints.finalize(str(save_root), "repo", scope)

    assert record["source"] == "exact_final"
    assert (save_root / "last.safetensors").read_bytes() == final.read_bytes()


@pytest.mark.parametrize(
    "mutation",
    ("sample_hash", "aggregate_hash", "duplicate_sample", "reordered_sequence"),
)
def test_consumer_rejects_inconsistent_split_attestation(
    tmp_path, monkeypatch, mutation
):
    _spec_value, save_root, _scope, _early, final = _produce_two_candidate_manifest(
        tmp_path, monkeypatch
    )
    manifest_path = save_root / "forge_holdout_scores.json"
    manifest = json.loads(manifest_path.read_text())
    split = manifest["dataset_split"]
    if mutation == "sample_hash":
        split["training"][0]["sample_sha256"] = "0" * 64
    elif mutation == "aggregate_hash":
        split["training_set_sha256"] = "0" * 64
    elif mutation == "duplicate_sample":
        split["holdout"][0] = dict(split["training"][0])
    else:
        split["training"].reverse()
    manifest["dataset_split_sha256"] = holdout.dataset_split_sha256(split)

    with pytest.raises(ValueError):
        checkpoints._validate_dataset_split_contract(manifest)

    assert not (save_root / "last.safetensors").exists()
    assert final.is_file()


def test_proxy_manifest_without_frozen_policy_retains_exact_final(
    tmp_path, monkeypatch
):
    spec, save_root, holdout_dir = _spec(tmp_path, monkeypatch)
    save_root.mkdir(parents=True)
    _write_pair(holdout_dir, "held", (1, 2, 3), b"caption")
    scope = _bound_scope(spec, save_root, steps=200)
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
    scope = _bound_scope(spec, save_root, steps=200)
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
    scope = _bound_scope(spec, save_root, steps=200)
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
    scope = _bound_scope(spec, save_root, steps=200)
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
    scope = _bound_scope(spec, save_root, steps=200)
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


@pytest.mark.parametrize("step", (0, 300))
def test_producer_rejects_candidate_outside_planned_depth(
    tmp_path, monkeypatch, step
):
    spec, save_root, holdout_dir = _spec(tmp_path, monkeypatch)
    save_root.mkdir(parents=True)
    _write_pair(holdout_dir, "held", (1, 2, 3), b"caption")
    scope = _bound_scope(spec, save_root, steps=200)
    _write_st(save_root / f"repo_{step:09d}.safetensors", "rogue")
    _write_st(save_root / "repo.safetensors", "final")
    monkeypatch.setenv("FORGE_HOLDOUT_SELECTION_TYPES", "krea2")
    calls = 0

    def scorer(**_kwargs):
        nonlocal calls
        calls += 1
        return {}

    assert not holdout.produce(
        spec,
        {"config": {}},
        scope,
        _deadline(),
        holdout_pairs=1,
        scorer=scorer,
    )
    assert calls == 0
    assert not (save_root / "forge_holdout_scores.json").exists()


@pytest.mark.parametrize("step", (0, 300))
def test_consumer_rejects_candidate_outside_planned_depth(
    tmp_path, monkeypatch, step
):
    _spec_value, save_root, scope, _early, final = _produce_two_candidate_manifest(
        tmp_path, monkeypatch
    )
    rogue = save_root / f"repo_{step:09d}.safetensors"
    _write_st(rogue, "rogue")
    manifest_path = save_root / "forge_holdout_scores.json"
    manifest = json.loads(manifest_path.read_text())
    row = dict(manifest["scores"][0])
    row.update(checkpoint=rogue.name, sha256=_sha256(rogue), step=step)
    manifest["scores"].append(row)
    manifest_path.write_text(json.dumps(manifest))

    record = checkpoints.finalize(str(save_root), "repo", scope)

    assert record["source"] == "exact_final"
    assert (save_root / "last.safetensors").read_bytes() == final.read_bytes()


def test_producer_rejects_stale_same_process_scope_before_scorer(
    tmp_path, monkeypatch
):
    spec, save_root, holdout_dir = _spec(tmp_path, monkeypatch)
    save_root.mkdir(parents=True)
    _write_pair(holdout_dir, "held", (1, 2, 3), b"caption")
    stale = _bound_scope(spec, save_root, steps=200)
    checkpoints.begin_run(
        str(save_root), "repo", task_id=spec.task_id
    )
    current_manifest = save_root / "forge_holdout_scores.json"
    current_manifest.write_text('{"owner":"new-attempt"}')
    monkeypatch.setenv("FORGE_HOLDOUT_SELECTION_TYPES", "krea2")
    calls = 0

    def scorer(**_kwargs):
        nonlocal calls
        calls += 1
        return {}

    assert not holdout.produce(
        spec,
        {"config": {}},
        stale,
        _deadline(),
        holdout_pairs=1,
        scorer=scorer,
    )
    assert calls == 0
    assert json.loads(current_manifest.read_text()) == {"owner": "new-attempt"}


def test_producer_rejects_spec_task_mismatch_before_scorer(tmp_path, monkeypatch):
    spec, save_root, holdout_dir = _spec(tmp_path, monkeypatch)
    save_root.mkdir(parents=True)
    _write_pair(holdout_dir, "held", (1, 2, 3), b"caption")
    scope = _bound_scope(spec, save_root, steps=200)
    foreign_spec = ImageSpec.build(
        task_id="other-task",
        model=spec.model,
        model_type=spec.model_type,
        expected_repo_name=spec.expected_repo_name,
        trigger_word=None,
        dataset_zip=None,
    )
    monkeypatch.setenv("FORGE_HOLDOUT_SELECTION_TYPES", "krea2")
    calls = 0

    def scorer(**_kwargs):
        nonlocal calls
        calls += 1
        return {}

    assert not holdout.produce(
        foreign_spec,
        {"config": {}},
        scope,
        _deadline(),
        holdout_pairs=1,
        scorer=scorer,
    )
    assert calls == 0


def test_producer_rechecks_scope_before_manifest_publication(tmp_path, monkeypatch):
    spec, save_root, holdout_dir = _spec(tmp_path, monkeypatch)
    save_root.mkdir(parents=True)
    _write_pair(holdout_dir, "held", (1, 2, 3), b"caption")
    scope = _bound_scope(spec, save_root, steps=200)
    _write_st(save_root / "repo_000000100.safetensors", "early")
    _write_st(save_root / "repo.safetensors", "final")
    monkeypatch.setenv("FORGE_HOLDOUT_SELECTION_TYPES", "krea2")
    calls = 0

    def scorer(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            checkpoints.begin_run(
                str(save_root), "repo", task_id=spec.task_id
            )
            (save_root / "forge_holdout_scores.json").write_text(
                '{"owner":"replacement-attempt"}'
            )
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
        scorer=scorer,
    )
    assert calls == 1
    assert json.loads(
        (save_root / "forge_holdout_scores.json").read_text()
    ) == {"owner": "replacement-attempt"}


@pytest.mark.parametrize("mutate", ("training", "captioned_probe", "blank_probe"))
def test_producer_rejects_dataset_or_probe_drift(
    tmp_path, monkeypatch, mutate
):
    spec, save_root, holdout_dir = _spec(tmp_path, monkeypatch)
    save_root.mkdir(parents=True)
    _write_pair(holdout_dir, "held", (1, 2, 3), b"caption")
    scope = _bound_scope(spec, save_root, steps=200)
    _write_st(save_root / "repo_000000100.safetensors", "early")
    _write_st(save_root / "repo.safetensors", "final")
    monkeypatch.setenv("FORGE_HOLDOUT_SELECTION_TYPES", "krea2")
    calls = 0

    def scorer(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            if mutate == "training":
                path = next(
                    path
                    for path in map(
                        lambda name: os.path.join(spec.dataset_images_dir, name),
                        os.listdir(spec.dataset_images_dir),
                    )
                    if path.endswith(".txt")
                )
                with open(path, "ab") as handle:
                    handle.write(b"-mutated")
            elif mutate == "captioned_probe":
                path = next(
                    os.path.join(kwargs["captioned_dir"], name)
                    for name in os.listdir(kwargs["captioned_dir"])
                    if name.endswith(".txt")
                )
                with open(path, "ab") as handle:
                    handle.write(b"-mutated")
            else:
                path = next(
                    os.path.join(kwargs["blank_dir"], name)
                    for name in os.listdir(kwargs["blank_dir"])
                    if name.endswith(".txt")
                )
                with open(path, "wb") as handle:
                    handle.write(b"not-blank")
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
        scorer=scorer,
    )
    assert calls == (2 if mutate == "training" else 1)
    assert not (save_root / "forge_holdout_scores.json").exists()


def test_ensure_run_rejects_task_mismatch(tmp_path):
    checkpoints.begin_run(str(tmp_path), "repo", task_id="task-a")

    with pytest.raises(RuntimeError, match="another task"):
        checkpoints.ensure_run(str(tmp_path), "repo", task_id="task-b")


def test_set_planned_steps_cannot_resurrect_stale_same_process_scope(tmp_path):
    stale = checkpoints.begin_run(str(tmp_path), "repo", task_id="task-a")
    active = checkpoints.begin_run(str(tmp_path), "repo", task_id="task-a")

    with pytest.raises(RuntimeError, match="stale run scope"):
        checkpoints.set_planned_steps(str(tmp_path), stale, 200)

    assert checkpoints.load_run(str(tmp_path)) == active
    assert "planned_steps" not in active


def test_finalize_rejects_stale_explicit_scope(tmp_path):
    stale = checkpoints.begin_run(str(tmp_path), "repo", task_id="task-a")
    active = checkpoints.begin_run(str(tmp_path), "repo", task_id="task-a")
    candidate = tmp_path / "repo.safetensors"
    _write_st(candidate, "active-final")

    assert checkpoints.current_loras(str(tmp_path), stale) == []
    assert checkpoints.finalize(str(tmp_path), "repo", stale) is None
    assert checkpoints.load_run(str(tmp_path)) == active
    assert candidate.is_file()
    assert not (tmp_path / "last.safetensors").exists()
    assert not (tmp_path / "forge_checkpoint_selection.json").exists()


@pytest.mark.parametrize("step", (0, 300))
def test_default_finalization_rejects_periodic_outside_bound_plan(tmp_path, step):
    scope = checkpoints.begin_run(str(tmp_path), "repo", task_id="task-a")
    scope = checkpoints.set_planned_steps(str(tmp_path), scope, 200)
    rogue = tmp_path / f"repo_{step:09d}.safetensors"
    _write_st(rogue, "rogue")

    assert checkpoints.current_loras(str(tmp_path), scope) == []
    assert checkpoints.current_loras(
        str(tmp_path), scope, enforce_plan=False
    ) == [str(rogue)]
    assert checkpoints.finalize(str(tmp_path), "repo", scope) is None
    assert not (tmp_path / "last.safetensors").exists()


def test_proxy_is_dormant_without_per_arch_allowlist(tmp_path, monkeypatch):
    spec, save_root, holdout_dir = _spec(tmp_path, monkeypatch)
    save_root.mkdir(parents=True)
    _write_pair(holdout_dir, "held", (1, 2, 3), b"caption")
    scope = checkpoints.begin_run(
        str(save_root), "repo", task_id=spec.task_id
    )
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
    scope = checkpoints.begin_run(
        str(save_root), "repo", task_id=spec.task_id
    )
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
