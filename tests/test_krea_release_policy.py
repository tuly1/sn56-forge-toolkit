"""Contracts for the dormant Week-5 Krea production router."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import pytest

from forge import config, krea_release_policy as policy
from forge.clock import Deadline
from forge.data.schema import ImageSpec
from forge.tasks import checkpoints


def _spec(model_type: str = "krea2") -> ImageSpec:
    return ImageSpec.build(
        task_id="week5-release-test",
        model="krea/Krea-2-Raw",
        model_type=model_type,
        expected_repo_name="release-repo",
        trigger_word="TOK",
        dataset_zip=None,
    )


def _activation(
    *,
    release_authorized: bool = True,
    outcomes: dict[str, str] | None = None,
) -> dict:
    outcomes = outcomes or {"K1": "PASS", "K5": "PASS"}
    body = {
        "schema": 1,
        "kind": policy.ACTIVATION_KIND,
        "policy_sha256": policy.POLICY_SHA256,
        "formal_endgame_decision_sha256": "a" * 64,
        "boundary_plan_sha256s": {
            family: {
                cell: hashlib.sha256(f"{family}/{cell}".encode("ascii")).hexdigest()
                for cell in (
                    "B-0p5-small",
                    "B-0p5-large",
                    "B-0p75-small",
                    "B-0p75-large",
                    "B-1-small",
                    "B-1-large",
                )
            }
            for family in ("K1", "K5")
        },
        "release_record_sha256": "b" * 64,
        "overall_confirmation_passed": all(
            value == "PASS" for value in outcomes.values()
        ),
        "policy_outcomes": outcomes,
        "production_mutation_authorized": True,
        "release_authorized": release_authorized,
        "deployment_authorized": False,
    }
    return {
        **body,
        "activation_sha256": hashlib.sha256(policy._canonical_bytes(body)).hexdigest(),
    }


def _build(
    monkeypatch: pytest.MonkeyPatch,
    pairs,
    *,
    remaining_hours: float = 0.74,
    granted_hours: float = 0.75,
    holdout_pairs=0,
) -> dict:
    monkeypatch.setattr(policy, "PRODUCTION_ACTIVATION", _activation())
    return config.build_config(
        _spec(),
        num_images=pairs,
        hours_to_complete=remaining_hours,
        holdout_pairs=holdout_pairs,
        granted_hours=granted_hours,
    )


def _process(cfg: dict) -> dict:
    return cfg["config"]["process"][0]


def _write_safetensors(path: Path, marker: float) -> bytes:
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode("ascii")
    payload = struct.pack("<Q", len(header)) + header + struct.pack("<f", marker)
    path.write_bytes(payload)
    return payload


def test_policy_file_is_canonical_and_hash_bound() -> None:
    path = (
        Path(policy.__file__).resolve().parent
        / "policies"
        / "krea_week5_production_predeclaration.json"
    )
    raw = path.read_bytes()
    artifact = json.loads(raw)

    assert raw == policy._canonical_bytes(artifact) + b"\n"
    assert artifact == policy.POLICY
    body = {key: value for key, value in artifact.items() if key != "policy_sha256"}
    assert hashlib.sha256(policy._canonical_bytes(body)).hexdigest() == (
        artifact["policy_sha256"]
    )
    assert artifact["release_authorized"] is False
    assert artifact["production_mutation_authorized"] is False
    assert artifact["deployment_authorized"] is False


def test_deadline_preserves_third_positional_argument_and_carries_grant() -> None:
    costs = [1.0, 2.0]
    legacy = Deadline(100.0, 10.0, costs)
    assert legacy._step_costs is costs
    assert legacy.granted_hours is None

    current = Deadline.from_hours(
        0.75,
        started_monotonic=100.0,
        export_reserve_s=180.0,
    )
    assert current.granted_hours == 0.75


def test_dormant_policy_preserves_exact_k0_config() -> None:
    spec = _spec()
    expected = config._apply_overrides(config.load_template("krea2"), spec, 24, 0.74)

    actual = config.build_config(
        spec,
        num_images=24,
        hours_to_complete=0.74,
        holdout_pairs=0,
        granted_hours=0.75,
    )

    assert actual == expected
    assert "forge_krea_production_policy" not in actual["meta"]
    assert "forge_krea_checkpoint_selection" not in actual["meta"]


def test_degraded_never_forfeit_config_is_never_labeled_week5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy, "PRODUCTION_ACTIVATION", _activation())

    def fail_overrides(*_args, **_kwargs):
        raise RuntimeError("forced override failure")

    monkeypatch.setattr(config, "_apply_overrides", fail_overrides)
    cfg = config.build_config(
        _spec(),
        num_images=18,
        hours_to_complete=0.74,
        holdout_pairs=0,
        granted_hours=0.75,
    )

    assert cfg["config"]["name"] == "release-repo"
    assert "forge_krea_production_policy" not in cfg["meta"]
    assert "forge_krea_checkpoint_selection" not in cfg["meta"]


@pytest.mark.parametrize(
    ("pairs", "family", "regime", "steps", "target", "selected_step"),
    [
        (18, "K1", "small", 209, {"numerator": 9, "denominator": 10}, 189),
        (27, "K1", "small", 209, {"numerator": 9, "denominator": 10}, 189),
        (28, "K5", "large", 213, {"numerator": 1, "denominator": 2}, 108),
        (36, "K5", "large", 213, {"numerator": 1, "denominator": 2}, 108),
    ],
)
def test_exact_pair_boundary_routes_profile_depth_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    pairs: int,
    family: str,
    regime: str,
    steps: int,
    target: dict,
    selected_step: int,
) -> None:
    cfg = _build(monkeypatch, pairs)
    process = _process(cfg)
    binding = cfg["meta"]["forge_krea_production_policy"]
    checkpoint = cfg["meta"]["forge_krea_checkpoint_selection"]

    assert binding["family"] == family
    assert binding["regime"] == regime
    assert binding["planned_steps"] == steps
    assert binding["remaining_time_step_cap_applied"] is False
    assert process["train"]["steps"] == steps
    assert process["save"]["save_every"] == (steps + 7) // 8
    assert process["train"]["lr"] == (0.0001 if family == "K1" else 0.0002)
    assert checkpoint["target_fraction"] == target
    assert checkpoint["selected_step"] == selected_step
    control = policy.checkpoint_control(cfg)
    assert control is not None
    assert control[0] == {
        "fraction_numerator": target["numerator"],
        "fraction_denominator": target["denominator"],
        "selection_rule": policy.CHECKPOINT_MAPPING_RULE,
    }
    assert control[1] == selected_step


@pytest.mark.parametrize(
    ("granted_hours", "remaining_hours", "boundary_cell", "steps", "selected_step"),
    [
        (0.5, 0.49, "B-0p5-small", 136, 119),
        (0.75, 0.74, "B-0p75-small", 209, 189),
        (1.0, 0.99, "B-1-small", 295, 259),
    ],
)
def test_exact_18_pair_boundary_anchor_is_not_truncated_by_k0_size_law(
    monkeypatch: pytest.MonkeyPatch,
    granted_hours: float,
    remaining_hours: float,
    boundary_cell: str,
    steps: int,
    selected_step: int,
) -> None:
    cfg = _build(
        monkeypatch,
        18,
        remaining_hours=remaining_hours,
        granted_hours=granted_hours,
    )
    binding = cfg["meta"]["forge_krea_production_policy"]

    assert binding["boundary_cell"] == boundary_cell
    assert binding["boundary_planned_steps"] == steps
    assert binding["remaining_time_step_cap"] >= steps
    assert binding["remaining_time_step_cap_applied"] is False
    assert _process(cfg)["train"]["steps"] == steps
    assert cfg["meta"]["forge_krea_checkpoint_selection"]["selected_step"] == (
        selected_step
    )


def test_original_grant_selects_anchor_while_remaining_time_caps_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normal = _build(
        monkeypatch,
        18,
        remaining_hours=0.70,
        granted_hours=0.75,
    )
    assert _process(normal)["train"]["steps"] == 209
    assert normal["meta"]["forge_krea_production_policy"]["boundary_cell"] == (
        "B-0p75-small"
    )

    delayed = _build(
        monkeypatch,
        18,
        remaining_hours=0.08,
        granted_hours=0.75,
    )
    delayed_binding = delayed["meta"]["forge_krea_production_policy"]
    assert delayed_binding["boundary_cell"] == "B-0p75-small"
    assert 1 <= _process(delayed)["train"]["steps"] < 209
    assert delayed_binding["remaining_time_step_cap_applied"] is True


@pytest.mark.parametrize("pairs", [None, True, 0, -1, "18", 18.0])
def test_invalid_or_unavailable_pair_count_preserves_k0(
    monkeypatch: pytest.MonkeyPatch, pairs
) -> None:
    monkeypatch.setattr(policy, "PRODUCTION_ACTIVATION", _activation())
    expected = config._apply_overrides(
        config.load_template("krea2"), _spec(), pairs, 0.74
    )
    actual = config.build_config(
        _spec(),
        num_images=pairs,
        hours_to_complete=0.74,
        holdout_pairs=0,
        granted_hours=0.75,
    )
    assert actual == expected
    assert "forge_krea_production_policy" not in actual["meta"]


@pytest.mark.parametrize("holdout_pairs", [None, True, 1, 2])
def test_policy_requires_holdout_disabled(
    monkeypatch: pytest.MonkeyPatch, holdout_pairs
) -> None:
    cfg = _build(monkeypatch, 18, holdout_pairs=holdout_pairs)
    assert "forge_krea_production_policy" not in cfg["meta"]


def test_invalid_or_unauthorized_activation_preserves_k0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        policy,
        "PRODUCTION_ACTIVATION",
        _activation(release_authorized=False),
    )
    cfg = config.build_config(
        _spec(),
        num_images=18,
        hours_to_complete=0.74,
        holdout_pairs=0,
        granted_hours=0.75,
    )
    assert "forge_krea_production_policy" not in cfg["meta"]


def test_out_of_fixture_range_clamps_and_sub_half_hour_preserves_k0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        _build(monkeypatch, 1)["meta"]["forge_krea_production_policy"]["family"] == "K1"
    )
    assert (
        _build(monkeypatch, 500)["meta"]["forge_krea_production_policy"]["family"]
        == "K5"
    )
    too_short = _build(
        monkeypatch,
        18,
        remaining_hours=0.4,
        granted_hours=0.49,
    )
    assert "forge_krea_production_policy" not in too_short["meta"]


@pytest.mark.parametrize(
    ("outcomes", "small_family", "large_family", "mode"),
    [
        ({"K1": "PASS", "K5": "PASS"}, "K1", "K5", "size_router"),
        ({"K1": "PASS", "K5": "FAIL"}, "K1", "K1", "K1_global"),
        ({"K1": "FAIL", "K5": "PASS"}, "K5", "K5", "K5_global"),
    ],
)
def test_predeclared_surprise_branch_uses_only_passing_policy(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: dict[str, str],
    small_family: str,
    large_family: str,
    mode: str,
) -> None:
    monkeypatch.setattr(
        policy,
        "PRODUCTION_ACTIVATION",
        _activation(outcomes=outcomes),
    )
    small = config.build_config(_spec(), 18, 0.74, holdout_pairs=0, granted_hours=0.75)
    large = config.build_config(_spec(), 36, 0.74, holdout_pairs=0, granted_hours=0.75)
    for cfg, family in ((small, small_family), (large, large_family)):
        binding = cfg["meta"]["forge_krea_production_policy"]
        assert binding["family"] == family
        assert binding["activation_mode"] == mode


def test_neither_policy_passes_keeps_k0(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        policy,
        "PRODUCTION_ACTIVATION",
        _activation(outcomes={"K1": "FAIL", "K5": "FAIL"}),
    )
    cfg = config.build_config(_spec(), 18, 0.74, holdout_pairs=0, granted_hours=0.75)
    assert "forge_krea_production_policy" not in cfg["meta"]


def test_activation_requires_all_twelve_family_qualified_boundary_plans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = _activation()
    activation["boundary_plan_sha256s"]["K5"].pop("B-1-large")
    body = {
        key: value for key, value in activation.items() if key != "activation_sha256"
    }
    activation["activation_sha256"] = hashlib.sha256(
        policy._canonical_bytes(body)
    ).hexdigest()
    monkeypatch.setattr(policy, "PRODUCTION_ACTIVATION", activation)

    cfg = config.build_config(_spec(), 36, 0.74, holdout_pairs=0, granted_hours=0.75)
    assert "forge_krea_production_policy" not in cfg["meta"]


def test_selected_boundary_binding_is_family_qualified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = _activation(outcomes={"K1": "PASS", "K5": "FAIL"})
    monkeypatch.setattr(policy, "PRODUCTION_ACTIVATION", activation)
    cfg = config.build_config(_spec(), 36, 0.74, holdout_pairs=0, granted_hours=0.75)
    binding = cfg["meta"]["forge_krea_production_policy"]
    assert binding["family"] == "K1"
    assert (
        binding["boundary_plan_sha256"]
        == activation["boundary_plan_sha256s"]["K1"]["B-0p75-large"]
    )
    assert (
        binding["boundary_plan_sha256"]
        != activation["boundary_plan_sha256s"]["K5"]["B-0p75-large"]
    )


def test_exact_target_checkpoint_is_promoted(tmp_path: Path) -> None:
    state = checkpoints.begin_run(str(tmp_path), "repo")
    state = checkpoints.set_planned_steps(
        str(tmp_path),
        state,
        136,
        model_type="krea2",
        checkpoint_target={
            "fraction_numerator": 9,
            "fraction_denominator": 10,
            "selection_rule": policy.CHECKPOINT_MAPPING_RULE,
        },
        checkpoint_selected_step=119,
    )
    expected = _write_safetensors(tmp_path / "repo_000000119.safetensors", 1.0)
    _write_safetensors(tmp_path / "repo.safetensors", 2.0)

    record = checkpoints.finalize(str(tmp_path), "repo", state)

    assert record is not None
    assert record["source"] == "frozen_checkpoint_fraction"
    assert record["selected_step"] == 119
    assert record["checkpoint_target_hit"] is True
    assert (tmp_path / "last.safetensors").read_bytes() == expected


def test_target_miss_salvages_nearest_current_checkpoint(tmp_path: Path) -> None:
    state = checkpoints.begin_run(str(tmp_path), "repo")
    state = checkpoints.set_planned_steps(
        str(tmp_path),
        state,
        136,
        model_type="krea2",
        checkpoint_target={
            "fraction_numerator": 9,
            "fraction_denominator": 10,
            "selection_rule": policy.CHECKPOINT_MAPPING_RULE,
        },
        checkpoint_selected_step=119,
    )
    _write_safetensors(tmp_path / "repo_000000068.safetensors", 1.0)
    expected = _write_safetensors(tmp_path / "repo_000000102.safetensors", 2.0)

    record = checkpoints.finalize(str(tmp_path), "repo", state)

    assert record is not None
    assert record["source"] == "frozen_checkpoint_fraction_salvage"
    assert record["selected_step"] == 102
    assert record["checkpoint_target_hit"] is False
    assert (tmp_path / "last.safetensors").read_bytes() == expected


def test_no_current_candidate_preserves_prior_last_under_policy(tmp_path: Path) -> None:
    expected = _write_safetensors(tmp_path / "last.safetensors", 3.0)
    state = checkpoints.begin_run(str(tmp_path), "repo")
    state = checkpoints.set_planned_steps(
        str(tmp_path),
        state,
        136,
        model_type="krea2",
        checkpoint_target={
            "fraction_numerator": 9,
            "fraction_denominator": 10,
            "selection_rule": policy.CHECKPOINT_MAPPING_RULE,
        },
        checkpoint_selected_step=119,
    )

    record = checkpoints.finalize(str(tmp_path), "repo", state)

    assert record is not None
    assert record["source"] == "previous_run_fallback"
    assert (tmp_path / "last.safetensors").read_bytes() == expected


def test_out_of_schedule_current_candidate_cannot_override_prior_last(
    tmp_path: Path,
) -> None:
    expected = _write_safetensors(tmp_path / "last.safetensors", 3.0)
    state = checkpoints.begin_run(str(tmp_path), "repo")
    state = checkpoints.set_planned_steps(
        str(tmp_path),
        state,
        136,
        model_type="krea2",
        checkpoint_target={
            "fraction_numerator": 9,
            "fraction_denominator": 10,
            "selection_rule": policy.CHECKPOINT_MAPPING_RULE,
        },
        checkpoint_selected_step=119,
    )
    _write_safetensors(tmp_path / "repo_000000200.safetensors", 4.0)

    record = checkpoints.finalize(str(tmp_path), "repo", state)

    assert record is not None
    assert record["status"] == "preserved_previous_run"
    assert record["source"] == "previous_run_fallback"
    assert (
        "no checkpoint eligible under the frozen production target" in record["reason"]
    )
    assert record["current_candidates_valid"] == 1
    assert (tmp_path / "last.safetensors").read_bytes() == expected
