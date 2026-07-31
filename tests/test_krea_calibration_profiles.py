"""Targeted contracts for the dormant Krea calibration config bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from forge import cli, config, krea_calibration_profiles as profiles
from forge.data.schema import ImageSpec


_EXPECTED_PROFILE_SHA256 = {
    "K0": "f5dd2bdfc765e86932ae90a9c5580f04479eb3f4417537efdb4ca646276f2251",
    "K1": "9ef2312471a5a9a5cad52ee7c35b2a2458a87d1b404dbc67659e369c6df95257",
    "K2": "78f80fd84ce6c5ea763073ca6d11f77e38c996b418e315b81ef2cc83b1221e1c",
    "K3": "31b454af3dc706c0843bc4835021a1274b7a10e9d1dca35d034d16ea621a61ba",
    "K4": "d09a6b32303fabba683490b966e30ba240c49fd1a5942c1a278da6205753cb36",
    "K5": "8d2b63f868439ded257df56900e01ddd6b347546dda4919b5bb8084798137337",
}

_AXES = {
    "K0": (0.0001, 32, 32, "adamw8bit", "mse", 2, 0.05, False),
    "K1": (0.0001, 32, 32, "adamw8bit", "mse", 2, 0.05, False),
    "K2": (0.000086, 32, 32, "adamw8bit", "mse", 2, 0.1, False),
    "K3": (0.0001, 32, 32, "adamw8bit", "mae", 3, 0.05, False),
    "K4": (0.00000086, 64, 64, "automagic", "mse", 2, 0.3, False),
    "K5": (0.0002, 32, 32, "adamw8bit", "mse", 2, 0.05, False),
}


def _spec(model_type: str = "krea2") -> ImageSpec:
    return ImageSpec.build(
        task_id="stage2-test",
        model="krea/Krea-2-Raw",
        model_type=model_type,
        expected_repo_name="stage2-repo",
        trigger_word="TOK",
        dataset_zip=None,
    )


@pytest.fixture(autouse=True)
def _selector_unset(monkeypatch):
    monkeypatch.delenv(profiles.PROFILE_SELECTOR_ENV, raising=False)
    monkeypatch.delenv(profiles.STAGE2_STEPS_ENV, raising=False)
    monkeypatch.delenv(profiles.STAGE2_THROUGHPUT_SHA_ENV, raising=False)
    monkeypatch.delenv(profiles.STAGE2_SEED_ENV, raising=False)
    monkeypatch.delenv(profiles.STAGE2_PLAN_SHA_ENV, raising=False)
    monkeypatch.delenv(profiles.STAGE2_RECEIPT_PATH_ENV, raising=False)
    monkeypatch.delenv(profiles.STAGE2_TARGET_NUMERATOR_ENV, raising=False)
    monkeypatch.delenv(profiles.STAGE2_TARGET_DENOMINATOR_ENV, raising=False)


def _process(cfg: dict) -> dict:
    return cfg["config"]["process"][0]


def test_unset_selector_is_exact_existing_production_config(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(
        profiles.telemetry,
        "event",
        lambda name, **values: events.append((name, values)),
    )
    spec = _spec()
    expected = config._apply_overrides(  # noqa: SLF001 - exact-path regression.
        config.load_template(spec.model_type), spec, 24, 1000
    )

    actual = config.build_config(spec, num_images=24, hours_to_complete=1000)

    assert actual == expected
    assert "forge_krea_calibration_profile" not in actual["meta"]
    assert events == []


@pytest.mark.parametrize("profile_id", tuple(_AXES))
def test_all_frozen_axes_and_candidate_cadences_are_reproduced(
    monkeypatch, profile_id: str
) -> None:
    monkeypatch.setenv(profiles.PROFILE_SELECTOR_ENV, profile_id)

    cfg = config.build_config(_spec(), num_images=24, hours_to_complete=1000)
    process = _process(cfg)
    train = process["train"]
    dataset = process["datasets"][0]
    expected = _AXES[profile_id]

    assert (
        train["lr"],
        process["network"]["linear"],
        process["network"]["linear_alpha"],
        train["optimizer"],
        train["loss_type"],
        train["differential_guidance_scale"],
        dataset["caption_dropout_rate"],
        train["ema_config"]["use_ema"],
    ) == expected
    assert train["do_differential_guidance"] is True
    assert train["ema_config"]["ema_decay"] == 0.99
    assert train["steps"] == 300  # Step depth remains the current planner's.
    if profile_id == "K0":
        assert process["save"]["save_every"] == 61  # Current release cadence.
    else:
        assert process["save"]["save_every"] == 38  # ceil(300 / 8).
    expected_optimizer_params = {"weight_decay": 0.0001}
    if profile_id == "K4":
        expected_optimizer_params.update(
            {"lr_bump": 0.000001, "max_lr": 0.001, "min_lr": 0.0000001}
        )
    assert train["optimizer_params"] == expected_optimizer_params
    assert process["save"]["push_to_hub"] is False

    if profile_id == "K0":
        expected = config._apply_overrides(
            config.load_template("krea2"), _spec(), 24, 1000
        )
        assert cfg == expected
        assert "forge_krea_calibration_profile" not in cfg["meta"]
        return

    binding = cfg["meta"]["forge_krea_calibration_profile"]
    assert binding["profile_id"] == profile_id
    assert binding["profile_sha256"] == _EXPECTED_PROFILE_SHA256[profile_id]
    assert binding["release_selected"] is False
    assert binding["measured_stage2_per_class_throughput_bound"] is False
    assert binding["throughput_profile_sha256"] is None
    assert (
        binding["depth_binding_status"]
        == "pending_measured_stage2_per_class_throughput"
    )


def test_profile_binding_is_pinned_to_the_frozen_discovery_plan() -> None:
    root = Path(__file__).resolve().parents[1]
    freeze = root / "ops/calibration/week5/krea-discovery-plan.json"

    assert hashlib.sha256(freeze.read_bytes()).hexdigest() == (
        profiles.SOURCE_FREEZE_FILE_SHA256
    )
    assert {
        profile_id: profiles.profile_for_id(profile_id).profile_sha256
        for profile_id in profiles.available_profile_ids()
    } == _EXPECTED_PROFILE_SHA256


@pytest.mark.parametrize("value", ["", "k2", " K2", "K2 ", "K6", "winner"])
def test_explicit_unknown_profile_is_rejected(monkeypatch, value: str) -> None:
    events = []
    monkeypatch.setattr(
        profiles.telemetry,
        "event",
        lambda name, **fields: events.append((name, fields)),
    )
    monkeypatch.setenv(profiles.PROFILE_SELECTOR_ENV, value)

    with pytest.raises(profiles.KreaCalibrationProfileError, match="unknown"):
        config.build_config(_spec(), num_images=24, hours_to_complete=1000)

    assert events == [
        (
            "krea_calibration_profile_rejected",
            {"reason": "unknown_profile", "selector_present": True},
        )
    ]


def test_selector_cannot_mutate_a_non_krea_run(monkeypatch) -> None:
    monkeypatch.setenv(profiles.PROFILE_SELECTOR_ENV, "K0")

    with pytest.raises(
        profiles.KreaCalibrationProfileError, match="requires model_type=krea2"
    ):
        config.build_config(_spec("flux"), num_images=24, hours_to_complete=1000)


def test_stage2_depth_is_explicit_plan_bound_and_calibration_only(monkeypatch) -> None:
    profile_sha = "a" * 64
    monkeypatch.setenv(profiles.PROFILE_SELECTOR_ENV, "K2")
    monkeypatch.setenv(profiles.STAGE2_STEPS_ENV, "691")
    monkeypatch.setenv(profiles.STAGE2_THROUGHPUT_SHA_ENV, profile_sha)

    cfg = config.build_config(_spec(), num_images=24, hours_to_complete=0.75)
    process = _process(cfg)
    binding = cfg["meta"]["forge_krea_calibration_profile"]

    assert process["train"]["steps"] == 691
    assert process["save"]["save_every"] == 87
    assert binding["planned_steps"] == 691
    assert binding["planned_steps_source"] == "explicit_owner_ratified_stage2_plan"
    assert binding["throughput_profile_sha256"] == profile_sha
    assert binding["measured_stage2_per_class_throughput_bound"] is True
    assert binding["release_selected"] is False


@pytest.mark.parametrize(
    ("profile_id", "steps", "profile_sha", "match"),
    [
        (None, "691", "a" * 64, "requires an explicit K1-K5"),
        ("K0", "691", "a" * 64, "preserve release-control depth"),
        ("K1", None, "a" * 64, "required together"),
        ("K1", "691", None, "required together"),
        ("K1", "0", "a" * 64, "canonical positive integer"),
        ("K1", "0691", "a" * 64, "canonical positive integer"),
        ("K1", "+691", "a" * 64, "canonical positive integer"),
        ("K1", "5001", "a" * 64, "exceed"),
        ("K1", "691", "A" * 64, "lowercase SHA-256"),
    ],
)
def test_stage2_depth_rejects_partial_unbound_or_unsafe_inputs(
    monkeypatch,
    profile_id: str | None,
    steps: str | None,
    profile_sha: str | None,
    match: str,
) -> None:
    if profile_id is not None:
        monkeypatch.setenv(profiles.PROFILE_SELECTOR_ENV, profile_id)
    if steps is not None:
        monkeypatch.setenv(profiles.STAGE2_STEPS_ENV, steps)
    if profile_sha is not None:
        monkeypatch.setenv(profiles.STAGE2_THROUGHPUT_SHA_ENV, profile_sha)

    with pytest.raises(profiles.KreaCalibrationProfileError, match=match):
        config.build_config(_spec(), num_images=24, hours_to_complete=0.75)


def test_stage2_seed_plan_and_receipt_are_all_or_nothing_and_seed_process(
    monkeypatch,
) -> None:
    plan_sha = "b" * 64
    receipt = f"/run-evidence/{plan_sha}/config-control.json"
    monkeypatch.setenv(profiles.PROFILE_SELECTOR_ENV, "K1")
    monkeypatch.setenv(profiles.STAGE2_SEED_ENV, "42565431")
    with pytest.raises(profiles.KreaCalibrationProfileError, match="required together"):
        config.build_config(_spec(), num_images=24, hours_to_complete=0.75)

    monkeypatch.setenv(profiles.STAGE2_PLAN_SHA_ENV, plan_sha)
    monkeypatch.setenv(profiles.STAGE2_RECEIPT_PATH_ENV, receipt)
    monkeypatch.setenv(profiles.STAGE2_TARGET_NUMERATOR_ENV, "7")
    monkeypatch.setenv(profiles.STAGE2_TARGET_DENOMINATOR_ENV, "8")
    monkeypatch.setenv(profiles.STAGE2_STEPS_ENV, "691")
    monkeypatch.setenv(profiles.STAGE2_THROUGHPUT_SHA_ENV, "a" * 64)
    cfg = config.build_config(_spec(), num_images=24, hours_to_complete=0.75)
    assert _process(cfg)["training_seed"] == 42565431
    selection = cfg["meta"]["forge_krea_checkpoint_selection"]
    assert selection["target_fraction"] == {"numerator": 7, "denominator": 8}
    assert selection["selected_step"] == 609
    assert selection["planned_steps"] == 691


def test_k0_asserts_every_release_axis_before_preserving_config() -> None:
    cfg = config._apply_overrides(config.load_template("krea2"), _spec(), 24, 0.75)
    _process(cfg)["train"]["lr"] = 0.0002
    with pytest.raises(profiles.KreaCalibrationProfileError, match="K0 release"):
        profiles.apply_profile(cfg, profiles.profile_for_id("K0"))


def test_terminal_receipt_proves_exact_final_step_and_is_create_only(
    tmp_path: Path, monkeypatch
) -> None:
    plan_sha = "c" * 64
    parent = tmp_path / plan_sha
    parent.mkdir()
    control_path = parent / "config-control.json"
    checkpoint_selection = {
        "schema": 1,
        "mapping_rule": profiles.STAGE2_CHECKPOINT_MAPPING_RULE,
        "target_fraction": {"numerator": 7, "denominator": 8},
        "planned_steps": 691,
        "selected_step": 609,
        "candidate_steps": [87, 174, 261, 348, 435, 522, 609, 691],
    }
    control_path.write_text(
        json.dumps({"checkpoint_selection": checkpoint_selection}) + "\n"
    )
    control = profiles.Stage2RunControl(
        seed=42565431,
        execution_plan_sha256=plan_sha,
        receipt_path=str(control_path),
        target_fraction_numerator=7,
        target_fraction_denominator=8,
    )
    monkeypatch.setattr(
        profiles, "selected_stage2_run_control", lambda *_args, **_kwargs: control
    )
    receipt = profiles.write_stage2_terminal_receipt(
        profile_id="K1",
        planned_steps=691,
        last_step=691,
        returncode=0,
        stopped_by_deadline=False,
    )
    assert receipt is not None
    assert receipt["planned_steps_completed"] is True
    assert receipt["natural_completion"] is True
    assert receipt["checkpoint_selection"] == checkpoint_selection
    assert json.loads((parent / "training-terminal.json").read_text()) == receipt
    with pytest.raises(FileExistsError):
        profiles.write_stage2_terminal_receipt(
            profile_id="K1",
            planned_steps=691,
            last_step=691,
            returncode=0,
            stopped_by_deadline=False,
        )


def test_stage2_selection_is_preserved_before_public_scrub(
    tmp_path: Path, monkeypatch
) -> None:
    plan_sha = "d" * 64
    evidence = tmp_path / "evidence" / plan_sha
    checkpoints = tmp_path / "checkpoints"
    evidence.mkdir(parents=True)
    checkpoints.mkdir()
    checkpoint_selection = {
        "schema": 1,
        "mapping_rule": profiles.STAGE2_CHECKPOINT_MAPPING_RULE,
        "target_fraction": {"numerator": 7, "denominator": 8},
        "planned_steps": 691,
        "selected_step": 609,
        "candidate_steps": [87, 174, 261, 348, 435, 522, 609, 691],
    }
    control_path = evidence / "config-control.json"
    control_path.write_text(
        json.dumps({"checkpoint_selection": checkpoint_selection}) + "\n"
    )
    control = profiles.Stage2RunControl(
        seed=42565431,
        execution_plan_sha256=plan_sha,
        receipt_path=str(control_path),
        target_fraction_numerator=7,
        target_fraction_denominator=8,
    )
    monkeypatch.setattr(
        profiles, "selected_stage2_run_control", lambda *_args, **_kwargs: control
    )
    selected = checkpoints / "repo_000000609.safetensors"
    promoted = checkpoints / "last.safetensors"
    selected.write_bytes(b"selected-checkpoint")
    promoted.write_bytes(selected.read_bytes())
    digest = hashlib.sha256(promoted.read_bytes()).hexdigest()
    record = {
        "schema": 1,
        "status": "selected_current_run",
        "context": "training",
        "source": "frozen_checkpoint_fraction",
        "selected_file": selected.name,
        "output_file": promoted.name,
        "selected_step": 609,
        "sha256": digest,
        "checkpoint_target": control.checkpoint_target,
        "planned_steps": 691,
    }
    source = checkpoints / "forge_checkpoint_selection.json"
    source.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")))

    preserved = profiles.preserve_stage2_checkpoint_selection(
        profile_id="K1", save_root=str(checkpoints), record=record
    )

    assert preserved is not None
    target = evidence / "forge_checkpoint_selection.json"
    assert target.read_bytes() == source.read_bytes()
    assert preserved["file_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert preserved["selected_checkpoint_sha256"] == digest
    with pytest.raises(FileExistsError):
        profiles.preserve_stage2_checkpoint_selection(
            profile_id="K1", save_root=str(checkpoints), record=record
        )


def test_profile_refuses_to_authorize_network_release() -> None:
    cfg = config.load_template("krea2")
    _process(cfg)["save"]["push_to_hub"] = True

    with pytest.raises(profiles.KreaCalibrationProfileError, match="push_to_hub"):
        profiles.apply_profile(cfg, profiles.profile_for_id("K2"))


def test_explicit_profile_emits_hash_bound_telemetry(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(
        profiles.telemetry,
        "event",
        lambda name, **fields: events.append((name, fields)),
    )
    monkeypatch.setenv(profiles.PROFILE_SELECTOR_ENV, "K3")

    config.build_config(_spec(), num_images=24, hours_to_complete=1000)

    assert len(events) == 1
    name, fields = events[0]
    assert name == "krea_calibration_profile_applied"
    assert fields["profile_id"] == "K3"
    assert fields["profile_sha256"] == _EXPECTED_PROFILE_SHA256["K3"]
    assert fields["planned_steps"] == 300
    assert fields["save_every"] == 38
    assert fields["measured_stage2_per_class_throughput_bound"] is False
    assert fields["release_selected"] is False


def test_unchanged_cli_entrypoint_routes_explicit_environment_profile(
    monkeypatch,
) -> None:
    captured = {}
    monkeypatch.setenv(profiles.PROFILE_SELECTOR_ENV, "K4")

    def _capture_run(spec: ImageSpec, _deadline) -> None:
        captured["cfg"] = config.build_config(
            spec, num_images=24, hours_to_complete=1000
        )

    monkeypatch.setattr(cli, "_run", _capture_run)
    result = cli.main(
        [
            "--task-id",
            "stage2-test",
            "--model",
            "krea/Krea-2-Raw",
            "--model-type",
            "krea2",
            "--expected-repo-name",
            "stage2-repo",
            "--hours-to-complete",
            "0.75",
            "--trigger-word",
            "TOK",
        ]
    )

    assert result == 0
    process = _process(captured["cfg"])
    assert process["network"]["linear"] == 64
    assert process["train"]["optimizer"] == "automagic"
    assert (
        captured["cfg"]["meta"]["forge_krea_calibration_profile"]["profile_id"] == "K4"
    )
