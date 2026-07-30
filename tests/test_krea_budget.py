"""Contract tests for the Day-0 measured Krea budget planner."""

from __future__ import annotations

from decimal import Decimal
import json

import pytest

from ops.calibration import krea_budget


_RAW_SAMPLES_SHA = "1" * 64
_RUNTIME_SHA = "2" * 64
_HOST_EXECUTION_SHA = "0" * 64
_RESOLUTION_SHA = "3" * 64
_PRECISION_SHA = "4" * 64
_MARGIN_SHA = "5" * 64
_VALIDATION_SHA = "6" * 64
_OPTIMIZER_SHA = "7" * 64
_DATASET_SHAPE_SHA = "8" * 64
_BASE_SHA = "9" * 64
_CONTAINER_SHA = "a" * 64
_GPU_SHA = "b" * 64
_TRAINER_SHA = "c" * 64
_MEASUREMENT_TOOL_SHA = "d" * 64
_BOUNDARY_SOURCE_SHA = "e" * 64


def _envelope(**overrides):
    values = {
        "equivalence_class": "a-rank32-adamw8bit-mse-guidance2",
        "network_rank": 32,
        "network_alpha": 32,
        "optimizer": "adamw8bit",
        "optimizer_config_sha256": _OPTIMIZER_SHA,
        "loss": "mse",
        "differential_guidance_enabled": True,
        "guidance_scale": 2.0,
        "training_pair_count": 24,
        "training_dataset_shape_sha256": _DATASET_SHAPE_SHA,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "data_parallel_replicas": 1,
        "resolution_policy_sha256": _RESOLUTION_SHA,
        "precision_policy_sha256": _PRECISION_SHA,
        "cache_latents_to_disk": False,
        "cache_text_embeddings": True,
        "compile_enabled": False,
        "jit_enabled": True,
        "dataloader_workers": 2,
        "base_model_identity_sha256": _BASE_SHA,
        "runtime_identity_sha256": _RUNTIME_SHA,
        "host_execution_identity_sha256": _HOST_EXECUTION_SHA,
        "execution_surface": "staged_host_venv",
        "execution_scope": "discovery_only",
        "venv_tree_manifest_sha256": _CONTAINER_SHA,
        "reference_container_image_sha256": "d" * 64,
        "gpu_identity_sha256": _GPU_SHA,
        "trainer_identity_sha256": _TRAINER_SHA,
        "measurement_tool_sha256": _MEASUREMENT_TOOL_SHA,
    }
    values.update(overrides)
    return krea_budget.seal_execution_envelope(**values)


def _record(**overrides):
    values = {
        "execution_envelope": _envelope(),
        "raw_sample_manifest_sha256": _RAW_SAMPLES_SHA,
        "startup_sample_count": 8,
        "update_sample_count": 800,
        "save_sample_count": 16,
        "startup_upper_bound_s": 100.0,
        "update_upper_bound_s": 2.0,
        "save_upper_bound_s": 9.0,
        "bound_method": "observed-max-plus-predeclared-margin",
        "margin_policy_sha256": _MARGIN_SHA,
        "end_to_end_validation_count": 1,
        "end_to_end_validation_sha256": _VALIDATION_SHA,
        "framework_stop_boundary_s": 225.0,
        "framework_stop_boundary_source_sha256": _BOUNDARY_SOURCE_SHA,
        "selection_mode": "offline_post_training",
        "selection_scorer_identity_sha256": None,
        "selection_scoring_reserve_s": 0.0,
        "finalization_reserve_s": 50.0,
        "upload_reserve_s": 20.0,
    }
    values.update(overrides)
    return krea_budget.seal_throughput_profile(**values)


def _profile(**overrides):
    return krea_budget.load_throughput_profile(_record(**overrides))


def test_profile_is_canonical_sha256_bound_and_json_round_trippable():
    record = _record()

    profile = krea_budget.load_throughput_profile(
        json.loads(json.dumps(record, sort_keys=True))
    )

    assert profile.profile_sha256 == record["profile_sha256"]
    assert profile.raw_sample_manifest_sha256 == _RAW_SAMPLES_SHA
    assert profile.execution_envelope.runtime_identity_sha256 == _RUNTIME_SHA
    assert profile.execution_envelope.equivalence_class.startswith("a-rank32")
    assert profile.resolution_policy_sha256 == _RESOLUTION_SHA
    assert profile.precision_policy_sha256 == _PRECISION_SHA
    assert profile.micro_batch_size == 2
    assert profile.gradient_accumulation_steps == 4
    assert profile.data_parallel_replicas == 1
    assert profile.framework_stop_boundary_s == 225.0
    assert profile.to_record() == record


def test_profile_rejects_tampering_missing_fields_and_unknown_fields():
    tampered = _record()
    tampered["update_upper_bound_s"] = 2.1
    with pytest.raises(krea_budget.ProfileValidationError, match="does not match"):
        krea_budget.load_throughput_profile(tampered)

    missing = _record()
    del missing["save_upper_bound_s"]
    with pytest.raises(krea_budget.ProfileValidationError, match="schema mismatch"):
        krea_budget.load_throughput_profile(missing)

    extra = {**_record(), "fallback_update_s": 3.5}
    with pytest.raises(krea_budget.ProfileValidationError, match="schema mismatch"):
        krea_budget.load_throughput_profile(extra)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("raw_sample_manifest_sha256", "ABC"),
        ("execution_envelope", {}),
        ("startup_sample_count", 0),
        ("startup_sample_count", 2),
        ("update_sample_count", True),
        ("update_sample_count", 99),
        ("save_sample_count", 1.5),
        ("save_sample_count", 7),
        ("startup_upper_bound_s", 0),
        ("update_upper_bound_s", float("nan")),
        ("save_upper_bound_s", float("inf")),
        ("bound_method", "empirical-p99"),
        ("margin_policy_sha256", "bad"),
        ("end_to_end_validation_count", 0),
        ("end_to_end_validation_sha256", "bad"),
        ("framework_stop_boundary_s", 224.999),
        ("framework_stop_boundary_s", float("inf")),
        ("framework_stop_boundary_source_sha256", "bad"),
        ("selection_mode", "guess"),
        ("selection_scoring_reserve_s", -1),
        ("finalization_reserve_s", -1),
        ("upload_reserve_s", None),
    ],
)
def test_profile_fails_closed_on_unmeasured_or_invalid_inputs(field, value):
    with pytest.raises(krea_budget.ProfileValidationError):
        _record(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("equivalence_class", "A Rank 32"),
        ("network_rank", 0),
        ("network_alpha", True),
        ("optimizer", "AdamW 8bit"),
        ("optimizer_config_sha256", "bad"),
        ("loss", ""),
        ("differential_guidance_enabled", 1),
        ("guidance_scale", None),
        ("training_pair_count", 0),
        ("training_dataset_shape_sha256", "bad"),
        ("micro_batch_size", 0),
        ("gradient_accumulation_steps", True),
        ("data_parallel_replicas", 1.5),
        ("resolution_policy_sha256", ""),
        ("precision_policy_sha256", "f" * 63),
        ("cache_latents_to_disk", 0),
        ("cache_text_embeddings", None),
        ("compile_enabled", "false"),
        ("jit_enabled", 1),
        ("dataloader_workers", -1),
        ("base_model_identity_sha256", "bad"),
        ("runtime_identity_sha256", "A" * 64),
        ("host_execution_identity_sha256", "bad"),
        ("venv_tree_manifest_sha256", "bad"),
        ("gpu_identity_sha256", "bad"),
        ("trainer_identity_sha256", "bad"),
        ("measurement_tool_sha256", "bad"),
    ],
)
def test_execution_envelope_fails_closed_on_class_or_runtime_drift(field, value):
    with pytest.raises(krea_budget.ProfileValidationError):
        _envelope(**{field: value})


def test_execution_envelope_is_self_bound_and_rejects_tampering():
    record = _envelope()
    envelope = krea_budget.load_execution_envelope(record)
    assert envelope.to_record() == record
    assert envelope.images_per_update == 8

    tampered = json.loads(json.dumps(record))
    tampered["network_rank"] = 64
    with pytest.raises(krea_budget.ProfileValidationError, match="does not match"):
        krea_budget.load_execution_envelope(tampered)

    disabled = _envelope(
        differential_guidance_enabled=False,
        guidance_scale=None,
    )
    assert disabled["guidance_scale"] is None
    with pytest.raises(krea_budget.ProfileValidationError, match="must be null"):
        _envelope(differential_guidance_enabled=False, guidance_scale=2.0)


def test_profile_requires_every_timing_input_instead_of_using_defaults():
    with pytest.raises(TypeError):
        krea_budget.seal_throughput_profile(
            execution_envelope=_envelope(),
            raw_sample_manifest_sha256=_RAW_SAMPLES_SHA,
            startup_sample_count=8,
            update_sample_count=800,
            save_sample_count=16,
            startup_upper_bound_s=100,
            update_upper_bound_s=2,
            save_upper_bound_s=10,
            bound_method="observed-max-plus-predeclared-margin",
            margin_policy_sha256=_MARGIN_SHA,
            end_to_end_validation_count=1,
            end_to_end_validation_sha256=_VALIDATION_SHA,
            framework_stop_boundary_s=225,
            framework_stop_boundary_source_sha256=_BOUNDARY_SOURCE_SHA,
            selection_mode="offline_post_training",
            selection_scorer_identity_sha256=None,
            selection_scoring_reserve_s=0,
            finalization_reserve_s=50,
            # upload_reserve_s deliberately absent
        )


def test_profile_requires_three_distinct_timing_evidence_artifacts():
    with pytest.raises(krea_budget.ProfileValidationError, match="three distinct"):
        _record(margin_policy_sha256=_RAW_SAMPLES_SHA)
    with pytest.raises(krea_budget.ProfileValidationError, match="three distinct"):
        _record(end_to_end_validation_sha256=_MARGIN_SHA)


def test_profile_envelope_digest_changes_for_every_throughput_class_axis():
    baseline = _envelope()
    variants = [
        _envelope(equivalence_class="b-rank32-adamw8bit-mae-guidance3"),
        _envelope(network_rank=64, network_alpha=64),
        _envelope(optimizer="automagic", optimizer_config_sha256="f" * 64),
        _envelope(loss="mae"),
        _envelope(guidance_scale=3.0),
        _envelope(training_pair_count=48),
        _envelope(training_dataset_shape_sha256="f" * 64),
        _envelope(cache_latents_to_disk=True),
        _envelope(compile_enabled=True),
        _envelope(jit_enabled=False),
        _envelope(dataloader_workers=4),
        _envelope(venv_tree_manifest_sha256="f" * 64),
        _envelope(host_execution_identity_sha256="f" * 64),
        _envelope(gpu_identity_sha256="f" * 64),
        _envelope(measurement_tool_sha256="f" * 64),
    ]
    assert all(
        variant["execution_envelope_sha256"] != baseline["execution_envelope_sha256"]
        for variant in variants
    )


def test_selection_reserve_is_bound_to_an_explicit_execution_mode():
    with pytest.raises(krea_budget.ProfileValidationError, match="must have no"):
        _record(selection_scoring_reserve_s=1)
    with pytest.raises(krea_budget.ProfileValidationError, match="must have no"):
        _record(selection_scorer_identity_sha256="7" * 64)
    with pytest.raises(krea_budget.ProfileValidationError, match="positive"):
        _record(
            selection_mode="live_in_budget",
            selection_scorer_identity_sha256="7" * 64,
        )

    live = _record(
        selection_mode="live_in_budget",
        selection_scorer_identity_sha256="7" * 64,
        selection_scoring_reserve_s=30,
    )
    plan = krea_budget.plan_budget(
        krea_budget.load_throughput_profile(live), hard_budget_s=1000
    )
    assert plan.selection_mode == "live_in_budget"
    assert plan.selection_scoring_reserve_s == Decimal("30.0")
    assert plan.max_affordable_steps == 291


def test_schedule_is_uniform_eighth_run_and_records_real_mappings():
    schedule = krea_budget.candidate_schedule(80)

    assert schedule.save_every == 10
    assert schedule.periodic_write_steps == (10, 20, 30, 40, 50, 60, 70, 80)
    assert schedule.periodic_save_count == 8
    assert [candidate.step for candidate in schedule.candidates] == [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
    ]
    assert [candidate.kind for candidate in schedule.candidates] == [
        "periodic",
        "periodic",
        "periodic",
        "periodic",
        "periodic",
        "periodic",
        "periodic",
        "final",
    ]
    assert [
        (mapping.desired, mapping.actual_step) for mapping in schedule.mappings
    ] == [
        ("10%", 10),
        ("25%", 20),
        ("50%", 40),
        ("75%", 60),
        ("90%", 70),
        ("final", 80),
    ]

    record = schedule.to_record()
    assert record["periodic_write_steps"] == [10, 20, 30, 40, 50, 60, 70, 80]
    assert record["actual_candidates"][0] == {
        "kind": "periodic",
        "step": 10,
        "fraction": "0.125",
        "fraction_numerator": 10,
        "fraction_denominator": 80,
        "image_exposures": 10,
    }
    assert record["desired_mappings"][0]["target_fraction"] == "0.1"
    assert record["desired_mappings"][0]["actual_fraction"] == "0.125"
    assert record["desired_mappings"][0]["absolute_fraction_error"] == "0.025"


@pytest.mark.parametrize("steps", [1, 2, 7, 8, 9, 56, 57, 367, 10**12])
def test_schedule_rounding_is_bounded_uniform_and_never_duplicates_final(steps):
    schedule = krea_budget.candidate_schedule(steps)
    periodic = [
        candidate.step
        for candidate in schedule.candidates
        if candidate.kind == "periodic"
    ]

    assert schedule.save_every == (steps + 7) // 8
    assert schedule.periodic_save_count == steps // schedule.save_every
    assert schedule.periodic_write_steps == tuple(
        range(schedule.save_every, steps + 1, schedule.save_every)
    )
    assert len(periodic) <= 7
    assert all(step < steps for step in periodic)
    assert all(
        right - left == schedule.save_every
        for left, right in zip(periodic, periodic[1:])
    )
    assert schedule.candidates[-1].step == steps
    assert schedule.candidates[-1].fraction == Decimal(1)
    assert [mapping.desired for mapping in schedule.mappings] == [
        "10%",
        "25%",
        "50%",
        "75%",
        "90%",
        "final",
    ]


@pytest.mark.parametrize("steps", [0, -1, 1.5, True, None])
def test_schedule_rejects_invalid_steps(steps):
    with pytest.raises(ValueError, match="positive integer"):
        krea_budget.candidate_schedule(steps)


def test_planner_maximizes_steps_and_accounts_for_all_seven_saves():
    # The 225s framework boundary dominates the 70s measured finalization +
    # upload costs. After 100s startup, seven saves cost 63s and 612s remain
    # for 306 updates at 2s each. The apparent outer slack is intentional: the
    # frozen trainer will stop before that wall-clock region.
    plan = krea_budget.plan_budget(_profile(), hard_budget_s=1000)

    assert plan.max_affordable_steps == 306
    assert plan.schedule.save_every == 39
    assert plan.schedule.periodic_save_count == 7
    assert [candidate.step for candidate in plan.schedule.candidates] == [
        39,
        78,
        117,
        156,
        195,
        234,
        273,
        306,
    ]
    assert plan.startup_s == Decimal("100.0")
    assert plan.update_runtime_s == Decimal("612.0")
    assert plan.periodic_save_runtime_s == Decimal("63.0")
    assert plan.selection_scoring_reserve_s == Decimal("0.0")
    assert plan.framework_stop_boundary_s == Decimal("225.0")
    assert plan.effective_training_stop_reserve_s == Decimal("225.0")
    assert plan.finalization_reserve_s == Decimal("50.0")
    assert plan.upload_reserve_s == Decimal("20.0")
    assert plan.planned_runtime_s == Decimal("845.0")
    assert plan.slack_s == Decimal("155.0")
    assert plan.budget_utilization == Decimal("0.845")
    assert plan.update_budget_utilization == Decimal("0.612")
    assert plan.images_per_update == 8
    assert plan.total_image_exposures == 2448

    mappings = {
        mapping.desired: mapping.actual_step for mapping in plan.schedule.mappings
    }
    assert mappings == {
        "10%": 39,
        "25%": 78,
        "50%": 156,
        "75%": 234,
        "90%": 273,
        "final": 306,
    }


def test_save_overhead_reduces_affordable_steps_and_is_not_handwaved():
    cheap = krea_budget.plan_budget(
        _profile(save_upper_bound_s=1.0), hard_budget_s=1000
    )
    expensive = krea_budget.plan_budget(
        _profile(save_upper_bound_s=8.0),
        hard_budget_s=1000,
    )

    assert cheap.max_affordable_steps == 334
    assert expensive.max_affordable_steps == 309
    assert expensive.max_affordable_steps < cheap.max_affordable_steps
    assert expensive.periodic_save_runtime_s == Decimal("56.0")


def test_larger_measured_finalize_upload_reserve_dominates_framework_boundary():
    plan = krea_budget.plan_budget(
        _profile(finalization_reserve_s=210, upload_reserve_s=40),
        hard_budget_s=1000,
    )
    assert plan.framework_stop_boundary_s == Decimal("225.0")
    assert plan.effective_training_stop_reserve_s == Decimal("250.0")
    assert plan.max_affordable_steps < 306


def test_framework_boundary_is_not_double_charged_as_physical_runtime():
    plan = krea_budget.plan_budget(_profile(), hard_budget_s=1000)
    assert plan.effective_training_stop_reserve_s == Decimal("225.0")
    assert plan.finalization_reserve_s + plan.upload_reserve_s == Decimal("70.0")
    assert plan.slack_s == Decimal("155.0")
    assert plan.planned_runtime_s + plan.slack_s == plan.hard_budget_s


@pytest.mark.parametrize("hard_budget_s", [500, 516, 1000, 1016, 4096])
def test_planner_matches_bruteforce_with_actual_terminal_periodic_writes(
    hard_budget_s,
):
    profile = _profile(save_upper_bound_s=1.0)
    plan = krea_budget.plan_budget(profile, hard_budget_s=hard_budget_s)
    feasible = []
    for steps in range(1, hard_budget_s + 1):
        schedule = krea_budget.candidate_schedule(steps)
        training_runtime = steps * 2 + schedule.periodic_save_count
        if training_runtime <= hard_budget_s - 100 - 225:
            feasible.append(steps)
    assert plan.max_affordable_steps == max(feasible)


@pytest.mark.parametrize("update_s", [0.5, 2.75, 11.0])
@pytest.mark.parametrize("save_s", [0.01, 0.2, 1.0])
@pytest.mark.parametrize(("startup_s", "hard_budget_s"), [(1, 500), (33, 1000)])
def test_planner_matches_bruteforce_across_timing_shapes(
    update_s,
    save_s,
    startup_s,
    hard_budget_s,
):
    """Exercise both the small-domain and seven/eight-save fast paths."""

    profile = _profile(
        startup_upper_bound_s=startup_s,
        update_upper_bound_s=update_s,
        save_upper_bound_s=save_s,
    )
    plan = krea_budget.plan_budget(profile, hard_budget_s=hard_budget_s)
    available = Decimal(str(hard_budget_s - startup_s - 225))
    update = Decimal(str(update_s))
    save = Decimal(str(save_s))
    ceiling = int(available // update)
    feasible = []
    for steps in range(1, ceiling + 1):
        schedule = krea_budget.candidate_schedule(steps)
        runtime = Decimal(steps) * update + Decimal(schedule.periodic_save_count) * save
        if runtime <= available:
            feasible.append(steps)

    assert feasible
    assert plan.max_affordable_steps == max(feasible)


def test_planner_rejects_small_nonmonotonic_schedule_that_breaks_save_cap():
    # Fixed=3, update=1, save=10.  Step 8 costs 91 because it writes eight
    # periodic checkpoints; step 9 costs only 52 because ceil(9/8)=2 writes
    # four.  The planner must search the small cadence-boundary domain rather
    # than assuming every successive step costs more.
    profile = _profile(
        startup_upper_bound_s=1,
        update_upper_bound_s=1,
        save_upper_bound_s=10,
        finalization_reserve_s=1,
        upload_reserve_s=1,
    )
    with pytest.raises(
        krea_budget.InsufficientBudgetError,
        match="save-overhead fraction",
    ):
        krea_budget.plan_budget(profile, hard_budget_s=278)


@pytest.mark.parametrize("budget", [None, True, 0, -1, float("nan"), float("inf")])
def test_planner_rejects_invalid_budget_instead_of_falling_back(budget):
    with pytest.raises(ValueError, match="hard_budget_s"):
        krea_budget.plan_budget(_profile(), hard_budget_s=budget)


def test_planner_refuses_budget_that_cannot_cover_one_update():
    with pytest.raises(krea_budget.InsufficientBudgetError):
        krea_budget.plan_budget(_profile(), hard_budget_s=171)


def test_planner_requires_validated_profile_not_an_unbound_mapping():
    with pytest.raises(TypeError, match="validated ThroughputProfile"):
        krea_budget.plan_budget(_record(), hard_budget_s=1000)


def test_planner_enforces_post_reserve_utilization_and_save_overhead():
    low_utilization = _profile(update_upper_bound_s=100, save_upper_bound_s=1)
    with pytest.raises(
        krea_budget.InsufficientBudgetError,
        match="minimum post-reserve training utilization",
    ):
        krea_budget.plan_budget(low_utilization, hard_budget_s=500)

    save_heavy = _profile(update_upper_bound_s=2, save_upper_bound_s=50)
    with pytest.raises(
        krea_budget.InsufficientBudgetError,
        match="save-overhead fraction",
    ):
        krea_budget.plan_budget(save_heavy, hard_budget_s=1000)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_post_reserve_training_utilization", 0),
        ("minimum_post_reserve_training_utilization", 0.89),
        ("minimum_post_reserve_training_utilization", 1.1),
        ("maximum_save_overhead_fraction", True),
        ("maximum_save_overhead_fraction", 0.11),
        ("maximum_save_overhead_fraction", float("nan")),
    ],
)
def test_planner_rejects_invalid_policy_fractions(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError):
        krea_budget.plan_budget(_profile(), hard_budget_s=1000, **kwargs)


def test_plan_record_is_complete_json_and_provenance_bound():
    plan = krea_budget.plan_budget(_profile(), hard_budget_s=1000)
    record = plan.to_record()
    encoded = json.dumps(record, sort_keys=True, allow_nan=False)

    assert record["profile_sha256"] == _record()["profile_sha256"]
    assert record["execution_envelope"] == _envelope()
    assert record["max_affordable_steps"] == 306
    assert record["save_every"] == 39
    assert record["training_geometry"] == {
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "data_parallel_replicas": 1,
        "images_per_update": 8,
        "total_image_exposures": 2448,
        "resolution_policy_sha256": _RESOLUTION_SHA,
        "precision_policy_sha256": _PRECISION_SHA,
    }
    assert record["timing_evidence"] == {
        "raw_sample_manifest_sha256": _RAW_SAMPLES_SHA,
        "bound_method": "observed-max-plus-predeclared-margin",
        "margin_policy_sha256": _MARGIN_SHA,
        "end_to_end_validation_count": 1,
        "end_to_end_validation_sha256": _VALIDATION_SHA,
        "framework_stop_boundary_source_sha256": _BOUNDARY_SOURCE_SHA,
    }
    assert record["selection"] == {
        "mode": "offline_post_training",
        "scorer_identity_sha256": None,
    }
    assert record["actual_candidates"][0]["image_exposures"] == 312
    assert record["accounting"] == {
        "startup_upper_bound_s": "100",
        "update_runtime_upper_bound_s": "612",
        "periodic_save_runtime_upper_bound_s": "63",
        "periodic_save_count": 7,
        "selection_scoring_reserve_s": "0",
        "framework_stop_boundary_s": "225",
        "effective_training_stop_reserve_s": "225",
        "finalization_reserve_s": "50",
        "upload_reserve_s": "20",
        "planned_runtime_s": "845",
        "slack_s": "155",
        "budget_utilization": "0.845",
        "update_budget_utilization": "0.612",
        "post_reserve_training_utilization": "1",
        "minimum_post_reserve_training_utilization": "0.9",
        "save_overhead_fraction": "0.09333333333333333333333333333",
        "maximum_save_overhead_fraction": "0.1",
    }
    assert '"desired_mappings"' in encoded
    assert '"actual_candidates"' in encoded
