"""Contracts for the Week-5 Ideogram production policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import time

import pytest

from forge import config, ideogram_release_policy as policy, telemetry
from forge.data.schema import ImageSpec
from forge.tasks import aitoolkit, checkpoints


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(telemetry, "_PRIVATE_ROOT", str(tmp_path / "private"))
    monkeypatch.setattr(telemetry, "_RUN_NONCE", "ideogram-policy-test")
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
        task_id="week5-ideogram-release-test",
        model="black-forest-labs/FLUX.1-Krea-dev",
        model_type=model_type,
        expected_repo_name="irepo",
        trigger_word=None,
        dataset_zip=None,
    )


def _activation(*, owner_override: bool = False) -> dict:
    body = {
        "schema": 1,
        "kind": policy.ACTIVATION_KIND,
        "policy_sha256": policy.POLICY_SHA256,
        "amendment_sha256": policy.AMENDMENT_SHA256,
        "formal_ideogram_decision_sha256": "a" * 64,
        "scored_exact_final_sha256": "b" * 64,
        "selected_arm": "I-J20",
        "selection_basis": (
            "null_result_owner_override" if owner_override else "clear_win"
        ),
        "owner_override": owner_override,
        "production_mutation_authorized": True,
        "release_authorized": True,
        "deployment_authorized": False,
    }
    return {
        **body,
        "activation_sha256": hashlib.sha256(
            policy._canonical_bytes(body)
        ).hexdigest(),
    }


# What `recipe.size_scaled_steps("ideogram4", 14, 0.75, ...)` materialises for
# the fixture shape below.  Was 107 under the discredited Jul-16 row
# (base 140 / p 0.50 / max 400), briefly 177 under the withdrawn two-point fit
# to the champion's step counts, and is now 421 under `base 500 / p 0.32`.
#
# The fixture shape moved 36 -> 14 pairs at the same 0.75 h ON PURPOSE.  14/0.75
# is the REAL Aug-3 `1365fa1c` shape, and at that shape the SIZE LAW binds
# (421 against a 477 clock cap), so this constant is invariant to `MARGIN` and
# to `SEC_PER_IT["ideogram4"]`.  At 36 pairs the CLOCK binds instead (477 at
# MARGIN 0.92, 432 at 0.85), which would have coupled this release-policy
# contract to a constant another unit is actively revising.
#
# Pinned in ONE place so a depth change shows up as a single deliberate edit
# rather than four silent ones; the depth law itself is guarded in
# tests/test_week6_ideogram_depth.py and tests/test_week6_depth_geometry.py.
PLANNED_STEPS = 421


def _build(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(policy, "PRODUCTION_ACTIVATION", _activation())
    cfg = config.build_config(_spec(), num_images=14, hours_to_complete=0.75)
    assert cfg["config"]["process"][0]["train"]["steps"] == PLANNED_STEPS
    return cfg


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


def _scope_with_control(tmp_path: Path, cfg: dict) -> dict:
    control = policy.checkpoint_control(cfg)
    assert control is not None
    state = checkpoints.begin_run(str(tmp_path), "irepo")
    return checkpoints.set_planned_steps(
        str(tmp_path),
        state,
        cfg["config"]["process"][0]["train"]["steps"],
        model_type="ideogram4",
        checkpoint_target=control[0],
        checkpoint_selected_step=control[1],
    )


def _write_loss_db(path: Path, state: dict, losses: list[float]) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE steps (step INTEGER PRIMARY KEY, wall_time REAL NOT NULL);
            CREATE TABLE metric_keys (
                key TEXT PRIMARY KEY, first_seen_step INTEGER, last_seen_step INTEGER
            );
            CREATE TABLE metrics (
                step INTEGER NOT NULL, key TEXT NOT NULL, value_real REAL,
                value_text TEXT, PRIMARY KEY (step, key)
            );
            """
        )
        conn.execute(
            "INSERT INTO metric_keys VALUES ('loss/loss', 1, ?)", (len(losses),)
        )
        for step, loss in enumerate(losses, 1):
            conn.execute(
                "INSERT INTO steps VALUES (?, ?)",
                (step, state["started_unix"] + step / 1000.0),
            )
            conn.execute(
                "INSERT INTO metrics VALUES (?, 'loss/loss', ?, NULL)",
                (step, loss),
            )


def test_policy_is_dormant_and_non_ideogram_is_exact_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy, "PRODUCTION_ACTIVATION", None)
    assert policy.PRODUCTION_ACTIVATION is None
    baseline = config._apply_overrides(
        config.load_template("ideogram4"), _spec(), 36, 0.75
    )
    assert config.build_config(_spec(), 36, 0.75) == baseline
    assert policy.checkpoint_control(baseline) is None

    monkeypatch.setattr(policy, "PRODUCTION_ACTIVATION", _activation())
    krea_spec = _spec("krea2")
    krea = config._apply_overrides(
        config.load_template("krea2"), krea_spec, 36, 0.75
    )
    assert config.build_config(krea_spec, 36, 0.75) == krea

    monkeypatch.setattr(policy, "PRODUCTION_ACTIVATION", {"schema": 1})
    assert config.build_config(_spec(), 36, 0.75) == baseline


def test_literal_production_activation_is_hash_bound_owner_override() -> None:
    active = policy._validated_activation(policy.PRODUCTION_ACTIVATION)
    assert active is not None
    assert active["formal_ideogram_decision_sha256"] == (
        "deb5bc3dc6590aa4a9ef0a234a5efc5bc25c40c04327810eb3c997c32dc30af4"
    )
    assert active["scored_exact_final_sha256"] == (
        "8d5ab294da5440ed7338ea912144056b7a13a8729d14c3cc05aeebc2cc2a1fde"
    )
    assert active["selection_basis"] == "null_result_owner_override"
    assert active["owner_override"] is True
    # Re-signed for the Week-6 EMA-horizon amendment.  The record no longer
    # authorises the bare I-J20-D2 port: it authorises that port PLUS exactly
    # one named amendment, and `amendment_sha256` is what scopes it.
    assert active["amendment_sha256"] == policy.AMENDMENT_SHA256
    assert active["activation_sha256"] == (
        "b7e436971430f04e216ddf5a4f1599a3f8de2f21e2f9462c1d67245aa0386ba2"
    )
    # The port itself is unchanged: deployment is still NOT authorised by the
    # record, so re-signing did not widen the authority it carries.
    assert active["deployment_authorized"] is False
    assert active["release_authorized"] is True


def test_active_recipe_matches_the_scored_production_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _build(monkeypatch)
    process = cfg["config"]["process"][0]
    train = process["train"]
    assert process["training_seed"] == 20260802
    assert process["datasets"][0]["cache_latents_to_disk"] is True
    assert {
        key: train.get(key)
        for key in (
            "train_text_encoder",
            "lr",
            "unet_lr",
            "text_encoder_lr",
            "lr_scheduler",
            "lr_scheduler_params",
            "ema_config",
            "do_cfg",
            "cfg_scale",
            "steps",
        )
    } == {
        "train_text_encoder": True,
        "lr": 0.000025,
        "unet_lr": 0.000025,
        "text_encoder_lr": 0.0000001,
        "lr_scheduler": "cosine",
        "lr_scheduler_params": {"eta_min": 0.0000025},
        # 0.995 -> 0.99 (Week-6 EMA-horizon amendment).  Guarded in detail by
        # tests/test_week6_ideogram_ema_horizon.py.
        "ema_config": {"use_ema": True, "ema_decay": 0.99},
        "do_cfg": True,
        "cfg_scale": 10.0,
        "steps": PLANNED_STEPS,
    }
    control = policy.checkpoint_control(cfg)
    assert control is not None
    assert control[0] == {
        "fraction_numerator": 1,
        "fraction_denominator": 1,
        "selection_rule": policy.CHECKPOINT_MAPPING_RULE,
    }
    assert control[1] == PLANNED_STEPS

    drifted = json.loads(json.dumps(cfg))
    drifted["config"]["process"][0]["train"]["lr"] = 0.0001
    with pytest.raises(ValueError, match="binding is invalid"):
        policy.checkpoint_control(drifted)


def test_exact_final_outranks_holdout_and_loss_divergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _build(monkeypatch)
    state = _scope_with_control(tmp_path, cfg)
    early = tmp_path / "irepo_000000070.safetensors"
    final = tmp_path / "irepo.safetensors"
    _write_safetensors(early, "early")
    exact_bytes = _write_safetensors(final, "scored-final")
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
                        "step": 70,
                        "score": 0.01,
                        "sha256": _sha256(early),
                    },
                    {
                        "checkpoint": final.name,
                        "step": PLANNED_STEPS,
                        "score": 0.50,
                        "sha256": _sha256(final),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_loss_db(
        tmp_path / "loss_log.db",
        state,
        [0.1] * 70 + [0.9] * (PLANNED_STEPS - 70),
    )

    record = checkpoints.finalize(str(tmp_path), "irepo", state)

    assert record is not None
    assert record["source"] == "frozen_checkpoint_fraction"
    assert record["checkpoint_target_hit"] is True
    assert (tmp_path / "last.safetensors").read_bytes() == exact_bytes


def test_missing_terminal_export_salvages_nearest_current_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _build(monkeypatch)
    state = _scope_with_control(tmp_path, cfg)
    _write_safetensors(tmp_path / "irepo_000000070.safetensors", "early")
    best = _write_safetensors(tmp_path / "irepo_000000140.safetensors", "nearest")

    record = checkpoints.finalize(str(tmp_path), "irepo", state)

    assert record is not None
    assert record["source"] == "frozen_checkpoint_fraction_salvage"
    assert record["selected_step"] == 140
    assert record["checkpoint_target_hit"] is False
    assert (tmp_path / "last.safetensors").read_bytes() == best


def test_aitoolkit_threads_policy_control_into_run_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _build(monkeypatch)
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
    monkeypatch.setattr(aitoolkit.checkpoints, "ensure_run", lambda *_: {"s": 1})
    monkeypatch.setattr(
        aitoolkit.dataset,
        "prepare_aitoolkit_dataset",
        lambda *_args, **_kwargs: (str(images), 36),
    )
    monkeypatch.setattr(aitoolkit.holdout, "budget_allows", lambda *_: False)
    monkeypatch.setattr(aitoolkit, "build_config", lambda *_args, **_kwargs: cfg)
    captured: dict = {}

    def capture(_root, scope, steps, **kwargs):
        captured.update({"scope": scope, "steps": steps, **kwargs})
        return {**scope, "planned_steps": steps}

    monkeypatch.setattr(aitoolkit.checkpoints, "set_planned_steps", capture)
    monkeypatch.setattr(aitoolkit, "write_config", lambda *_: None)
    monkeypatch.setattr(aitoolkit, "_run_toolkit", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(aitoolkit, "_finalize", lambda *_: None)
    monkeypatch.setattr(aitoolkit.telemetry, "collect_env", lambda: None)
    monkeypatch.setattr(aitoolkit.telemetry, "set_meta", lambda **_: None)
    monkeypatch.setattr(aitoolkit.telemetry, "event", lambda *_args, **_kwargs: None)

    class Deadline:
        granted_hours = 0.75

        def remaining(self) -> float:
            return 2520.0

        def remaining_hard(self) -> float:
            return 2700.0

    aitoolkit.run(spec, Deadline())
    assert captured["steps"] == PLANNED_STEPS
    assert captured["model_type"] == "ideogram4"
    assert captured["checkpoint_selected_step"] == PLANNED_STEPS
    assert captured["checkpoint_target"]["fraction_numerator"] == 1


def test_degraded_override_never_activates_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy, "PRODUCTION_ACTIVATION", _activation())
    monkeypatch.setattr(config, "_apply_overrides", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    cfg = config.build_config(_spec(), 36, 0.75)
    assert cfg["config"]["process"][0]["train"]["lr"] == 0.0004
    assert policy.checkpoint_control(cfg) is None
