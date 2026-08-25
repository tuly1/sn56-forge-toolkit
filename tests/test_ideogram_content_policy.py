"""Contracts for the validated Week-11 Ideogram product promotion."""

from __future__ import annotations

import builtins
import copy
import hashlib
import json
from pathlib import Path
import struct
import time

import pytest

from forge import config, ideogram_content_policy as policy, telemetry
from forge.data.schema import ImageSpec
from forge.tasks import aitoolkit, checkpoints


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(telemetry, "_PRIVATE_ROOT", str(tmp_path / "private"))
    monkeypatch.setattr(telemetry, "_RUN_NONCE", "content-policy-test")
    monkeypatch.setattr(telemetry, "_BOUND_RUN_KEYS", {})
    monkeypatch.setattr(telemetry, "_LATEST_PRIVATE_RECORDS", {})
    monkeypatch.setattr(telemetry, "_t0", time.monotonic())
    monkeypatch.setattr(
        telemetry,
        "_data",
        {
            "schema": 1,
            "meta": {},
            "env": {},
            "events": [],
            "train_curve": [],
            "eval_curve": [],
            "samples": {},
        },
    )
    checkpoints._ACTIVE_RUNS.clear()


def _spec(model_type: str = "ideogram4") -> ImageSpec:
    return ImageSpec.build(
        task_id="week11-content-policy-test",
        model="black-forest-labs/FLUX.1-Krea-dev",
        model_type=model_type,
        expected_repo_name="content-repo",
        trigger_word="AuraGlow Sphere",
        dataset_zip=None,
    )


def _production(model_type: str = "ideogram4", *, images: int = 24, hours: float = 1.0) -> dict:
    return config.build_config(_spec(model_type), images, hours)


def _product(*, images: int = 24, hours: float = 1.0) -> dict:
    return config.build_config(
        _spec(), images, hours, dataset_category="product"
    )


def _write_safetensors(path: Path, tag: str) -> bytes:
    header = json.dumps(
        {
            "__metadata__": {"tag": tag},
            "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        },
        separators=(",", ":"),
    ).encode("ascii")
    payload = struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)
    path.write_bytes(payload)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_detector_matches_public_threshold_boundaries_and_recursive_scan(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "one.txt").write_text("A PRODUCT packshot", encoding="utf-8")
    for index in range(3):
        (tmp_path / f"plain-{index}.txt").write_text("abstract scene", encoding="utf-8")
    assert policy.detect_category(str(tmp_path), "product") == "product"

    (tmp_path / "plain-3.txt").write_text("abstract scene", encoding="utf-8")
    assert policy.detect_category(str(tmp_path)) == "default"

    (tmp_path / "plain-3.txt").write_text("productivity building", encoding="utf-8")
    assert policy.detect_category(str(tmp_path)) == "default"


def test_detector_preserves_keyword_insertion_order_for_ties(tmp_path: Path) -> None:
    (tmp_path / "tie.txt").write_text("logo product", encoding="utf-8")
    assert policy.detect_category(str(tmp_path)) == "logo"


def test_detector_missing_unreadable_and_frozen_product_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert policy.detect_category(str(tmp_path / "missing")) == "default"
    caption = tmp_path / "fixture.txt"
    caption.write_text(
        '{"trigger":"AuraGlow Sphere","description":"studio shot of the product"}',
        encoding="utf-8",
    )
    assert policy.detect_category(str(tmp_path)) == "product"

    real_open = builtins.open

    def unreadable(path, *args, **kwargs):
        if str(path) == str(caption):
            raise OSError("unreadable")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", unreadable)
    assert policy.detect_category(str(tmp_path)) == "default"


@pytest.mark.parametrize(
    "category",
    [
        None,
        "default",
        "logo",
        "line",
        "watercolor",
        "social",
        "design",
        "style",
        "sketch",
        "lineart",
        "unknown",
    ],
)
def test_every_non_product_category_is_exact_production(category: str | None) -> None:
    baseline = _production()
    candidate = config.build_config(
        _spec(), 24, 1.0, dataset_category=category
    )
    assert candidate == baseline


@pytest.mark.parametrize("model_type", ["flux", "krea2", "z-image", "qwen-image"])
def test_non_ideogram_product_category_is_exact_noop(model_type: str) -> None:
    baseline = _production(model_type)
    candidate = config.build_config(
        _spec(model_type), 24, 1.0, dataset_category="product"
    )
    assert candidate == baseline


def test_line_stays_exact_current_no_cfg_depth_and_schedule() -> None:
    cfg = config.build_config(_spec(), 24, 1.0, dataset_category="line")
    process = cfg["config"]["process"][0]
    train = process["train"]
    assert process["save"]["save_every"] == 200
    assert train["steps"] == 1250
    assert train["lr"] == train["unet_lr"] == 0.000025
    assert train["lr_scheduler"] == "cosine"
    assert train["lr_scheduler_params"] == {"eta_min": 0.0000025}
    assert "do_cfg" not in train
    assert "cfg_scale" not in train


def test_product_projection_is_the_frozen_challenger_after_runtime_normalization() -> None:
    cfg = _product()
    process = cfg["config"]["process"][0]
    train = process["train"]
    assert policy._sha256(policy._trainer_projection(cfg, None)) == (
        "ce226348ca641932de4a0f27d59a69ad8a124825068a4f1e9e9d134166eba4cd"
    )
    assert process["network"]["linear"] == process["network"]["linear_alpha"] == 32
    assert process["datasets"][0]["caption_dropout_rate"] == 0.05
    assert process["datasets"][0]["cache_latents_to_disk"] is True
    assert process["save"] == {
        "dtype": "bf16",
        "save_every": 100,
        "max_step_saves_to_keep": 100,
        "save_format": "diffusers",
        "push_to_hub": False,
    }
    assert train["steps"] == 1514
    assert train["lr"] == train["unet_lr"] == 0.0004
    assert train["train_text_encoder"] is False
    assert train["cache_text_embeddings"] is True
    assert train["min_denoising_steps"] == 250
    assert train["timestep_type"] == "linear"
    assert train["content_or_style"] == "balanced"
    assert train["ema_config"] == {"use_ema": True, "ema_decay": 0.995}
    assert "training_seed" not in process
    for key in (
        "text_encoder_lr",
        "lr_scheduler",
        "lr_scheduler_params",
        "do_cfg",
        "cfg_scale",
    ):
        assert key not in train


def test_product_step_formula_and_bounds() -> None:
    assert policy.product_steps(1.0) == 1514
    assert policy.product_steps(0.01) == 120
    assert policy.product_steps(100.0) == 1700
    assert policy.product_steps(0.5) == int((0.5 * 3600 - 420) / 2.1)


def test_malformed_activation_source_base_category_and_config_fail_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _production()

    activation = dict(policy.PRODUCTION_ACTIVATION)
    activation["activation_sha256"] = "0" * 64
    assert policy.apply(
        baseline,
        "ideogram4",
        "product",
        num_images=24,
        hours_to_complete=1.0,
        activation=activation,
    ) is baseline

    malformed_sources = {**policy.SOURCE_BINDINGS, "decision_sha256": "0" * 64}
    with monkeypatch.context() as scoped:
        scoped.setattr(policy, "SOURCE_BINDINGS", malformed_sources)
        assert policy.apply(
            baseline,
            "ideogram4",
            "product",
            num_images=24,
            hours_to_complete=1.0,
        ) is baseline

    drifted = copy.deepcopy(baseline)
    drifted["config"]["process"][0]["train"]["lr"] = 0.123
    before = copy.deepcopy(drifted)
    assert policy.apply(
        drifted,
        "ideogram4",
        "product",
        num_images=24,
        hours_to_complete=1.0,
    ) is drifted
    assert drifted == before

    assert policy.apply(
        baseline,
        "ideogram4",
        {"product": True},
        num_images=24,
        hours_to_complete=1.0,
    ) is baseline
    malformed = {"config": {}}
    assert policy.apply(
        malformed,
        "ideogram4",
        "product",
        num_images=24,
        hours_to_complete=1.0,
    ) is malformed


def test_corrupt_activation_falls_back_to_exact_current_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _production()
    corrupt = dict(policy.PRODUCTION_ACTIVATION)
    corrupt["policy_sha256"] = "0" * 64
    monkeypatch.setattr(policy, "PRODUCTION_ACTIVATION", corrupt)
    assert _product() == baseline


def test_corrupt_emitted_binding_fails_closed() -> None:
    cfg = _product()
    cfg["meta"]["forge_ideogram_content_policy"]["policy_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content checkpoint binding is invalid"):
        policy.checkpoint_control(cfg)


def test_product_exact_final_outranks_generic_selector(tmp_path: Path) -> None:
    cfg = _product()
    control = policy.checkpoint_control(cfg)
    assert control is not None
    state = checkpoints.begin_run(str(tmp_path), "content-repo")
    state = checkpoints.set_planned_steps(
        str(tmp_path),
        state,
        1514,
        model_type="ideogram4",
        checkpoint_target=control[0],
        checkpoint_selected_step=control[1],
    )
    early = tmp_path / "content-repo_000000100.safetensors"
    final = tmp_path / "content-repo.safetensors"
    _write_safetensors(early, "early")
    final_bytes = _write_safetensors(final, "exact-product-final")
    (tmp_path / "forge_holdout_scores.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "source": "heldout",
                "complete": True,
                "metric": "validator_exact_combined",
                "direction": "min",
                "scores": [
                    {
                        "checkpoint": early.name,
                        "step": 100,
                        "score": 0.01,
                        "sha256": _sha256(early),
                    },
                    {
                        "checkpoint": final.name,
                        "step": 1514,
                        "score": 0.50,
                        "sha256": _sha256(final),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    record = checkpoints.finalize(str(tmp_path), "content-repo", state)
    assert record is not None
    assert record["source"] == "frozen_checkpoint_fraction"
    assert record["selected_step"] == 1514
    assert (tmp_path / "last.safetensors").read_bytes() == final_bytes


def test_category_is_computed_once_before_holdout_and_content_control_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec()
    save_root = tmp_path / "save"
    training_root = tmp_path / "training"
    images = tmp_path / "images"
    images.mkdir()
    monkeypatch.setattr(type(spec), "save_root", property(lambda self: str(save_root)))
    monkeypatch.setattr(
        type(spec), "training_folder", property(lambda self: str(training_root))
    )
    monkeypatch.setattr(
        type(spec), "config_path", property(lambda self: str(tmp_path / "config.yaml"))
    )
    order: list[str] = []
    monkeypatch.setattr(aitoolkit.checkpoints, "ensure_run", lambda *_: {"scope": 1})
    monkeypatch.setattr(
        aitoolkit.dataset,
        "prepare_aitoolkit_dataset",
        lambda *_args, **_kwargs: (str(images), 24),
    )

    def detect(*_args):
        order.append("detect")
        return "product"

    monkeypatch.setattr(aitoolkit.ideogram_content_policy, "detect_category", detect)
    monkeypatch.setattr(aitoolkit.holdout, "budget_allows", lambda *_: True)

    def reserve(*_args, **_kwargs):
        order.append("holdout")
        return 0

    monkeypatch.setattr(aitoolkit.dataset, "reserve_holdout", reserve)
    captured: dict = {}

    def build(*_args, **kwargs):
        captured.update(kwargs)
        return {
            "config": {"process": [{"train": {"steps": 1514}, "save": {"save_every": 100}}]}
        }

    monkeypatch.setattr(aitoolkit, "build_config", build)
    target = {
        "fraction_numerator": 1,
        "fraction_denominator": 1,
        "selection_rule": policy.CHECKPOINT_MAPPING_RULE,
    }
    monkeypatch.setattr(
        aitoolkit.ideogram_content_policy,
        "checkpoint_control",
        lambda _cfg: (target, 1514),
    )

    def production_must_not_run(_cfg):
        raise AssertionError("generic production control ran before product control")

    monkeypatch.setattr(
        aitoolkit.ideogram_release_policy,
        "checkpoint_control",
        production_must_not_run,
    )
    planned: dict = {}

    def set_plan(_root, scope, steps, **kwargs):
        planned.update({"steps": steps, **kwargs})
        return scope

    monkeypatch.setattr(aitoolkit.checkpoints, "set_planned_steps", set_plan)
    monkeypatch.setattr(aitoolkit, "write_config", lambda *_: None)
    monkeypatch.setattr(aitoolkit, "_run_toolkit", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(aitoolkit, "_finalize", lambda *_: None)
    monkeypatch.setattr(aitoolkit.telemetry, "collect_env", lambda: None)
    monkeypatch.setattr(aitoolkit.telemetry, "set_meta", lambda **_: None)
    monkeypatch.setattr(aitoolkit.telemetry, "event", lambda *_args, **_kwargs: None)

    class Deadline:
        def remaining(self) -> float:
            return 3420.0

        def remaining_hard(self) -> float:
            return 3600.0

    aitoolkit.run(spec, Deadline())
    assert order == ["detect", "holdout"]
    assert captured["dataset_category"] == "product"
    assert planned["checkpoint_selected_step"] == 1514
    assert planned["checkpoint_target"] == target


def test_non_ideogram_run_never_calls_detector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec("krea2")
    save_root = tmp_path / "save"
    training_root = tmp_path / "training"
    images = tmp_path / "images"
    images.mkdir()
    monkeypatch.setattr(type(spec), "save_root", property(lambda self: str(save_root)))
    monkeypatch.setattr(
        type(spec), "training_folder", property(lambda self: str(training_root))
    )
    monkeypatch.setattr(
        type(spec), "config_path", property(lambda self: str(tmp_path / "config.yaml"))
    )
    monkeypatch.setattr(aitoolkit.checkpoints, "ensure_run", lambda *_: {"scope": 1})
    monkeypatch.setattr(
        aitoolkit.dataset,
        "prepare_aitoolkit_dataset",
        lambda *_args, **_kwargs: (str(images), 24),
    )

    def detector_must_not_run(*_args):
        raise AssertionError("non-Ideogram captions were scanned")

    monkeypatch.setattr(
        aitoolkit.ideogram_content_policy, "detect_category", detector_must_not_run
    )
    monkeypatch.setattr(aitoolkit.holdout, "budget_allows", lambda *_: False)
    captured: dict = {}

    def build(*_args, **kwargs):
        captured.update(kwargs)
        return {
            "config": {"process": [{"train": {"steps": 100}, "save": {"save_every": 25}}]}
        }

    monkeypatch.setattr(aitoolkit, "build_config", build)
    monkeypatch.setattr(
        aitoolkit.ideogram_content_policy, "checkpoint_control", lambda _cfg: None
    )
    monkeypatch.setattr(
        aitoolkit.ideogram_release_policy, "checkpoint_control", lambda _cfg: None
    )
    monkeypatch.setattr(
        aitoolkit.checkpoints,
        "set_planned_steps",
        lambda _root, scope, _steps, **_kwargs: scope,
    )
    monkeypatch.setattr(aitoolkit, "write_config", lambda *_: None)
    monkeypatch.setattr(aitoolkit, "_run_toolkit", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(aitoolkit, "_finalize", lambda *_: None)
    monkeypatch.setattr(aitoolkit.telemetry, "collect_env", lambda: None)
    monkeypatch.setattr(aitoolkit.telemetry, "set_meta", lambda **_: None)
    monkeypatch.setattr(aitoolkit.telemetry, "event", lambda *_args, **_kwargs: None)

    class Deadline:
        def remaining(self) -> float:
            return 3420.0

        def remaining_hard(self) -> float:
            return 3600.0

    aitoolkit.run(spec, Deadline())
    assert captured["dataset_category"] is None
