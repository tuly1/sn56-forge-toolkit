"""Dormant, environment-selected Krea calibration recipe profiles.

The normal tournament path must remain the release config until a finalist is
explicitly frozen.  Consequently this module has no default: it is consulted
only when ``FORGE_KREA_CALIBRATION_PROFILE`` is present and accepts only the
six frozen Week-5 arm identifiers.

This bridge reproduces the frozen recipe axes and candidate cadence.  It does
*not* claim that the existing production step planner is a measured Stage-2
budget-fill planner.  Until exact production-surface throughput is bound, the
already-planned step count is retained and the config/telemetry binding records
that limitation.  Nothing here downloads, uploads, publishes, or selects a
release profile.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
import re
import stat
from typing import Any, Mapping

from forge import telemetry


PROFILE_SELECTOR_ENV = "FORGE_KREA_CALIBRATION_PROFILE"
STAGE2_STEPS_ENV = "FORGE_KREA_CALIBRATION_STEPS"
STAGE2_THROUGHPUT_SHA_ENV = "FORGE_KREA_CALIBRATION_THROUGHPUT_PROFILE_SHA256"
STAGE2_SEED_ENV = "FORGE_KREA_STAGE2_TRAINING_SEED"
STAGE2_PLAN_SHA_ENV = "FORGE_KREA_STAGE2_EXECUTION_PLAN_SHA256"
STAGE2_RECEIPT_PATH_ENV = "FORGE_KREA_STAGE2_CONTROL_RECEIPT_PATH"
STAGE2_TARGET_NUMERATOR_ENV = "FORGE_KREA_STAGE2_TARGET_FRACTION_NUMERATOR"
STAGE2_TARGET_DENOMINATOR_ENV = "FORGE_KREA_STAGE2_TARGET_FRACTION_DENOMINATOR"
STAGE2_TIMING_PLAN_SHA_ENV = "FORGE_KREA_STAGE2_TIMING_PLAN_SHA256"
STAGE2_TIMING_CONTRACT_SHA_ENV = "FORGE_KREA_STAGE2_TIMING_PROBE_CONTRACT_SHA256"
STAGE2_TIMING_STEPS_ENV = "FORGE_KREA_STAGE2_TIMING_STEPS"
STAGE2_TIMING_SEED_ENV = "FORGE_KREA_STAGE2_TIMING_SEED"
STAGE2_TIMING_RECEIPT_PATH_ENV = "FORGE_KREA_STAGE2_TIMING_RECEIPT_PATH"
STAGE2_CHECKPOINT_MAPPING_RULE = "nearest_current_candidate_ties_choose_earlier_step"
MAX_STAGE2_STEPS = 5000
SOURCE_FREEZE_FILE_SHA256 = (
    "aa57e9bc7d32f04658a820e0cb763ae21df8a6ae7ee30e40d8d5b376a6956ad7"
)
_PROFILE_KIND = "forge-krea-calibration-profile"
_BINDING_KIND = "forge-krea-calibration-profile-binding"
_SCHEMA = 1
_BUDGET_FILL_CADENCE = "discovery-uniform-1/8-with-real-write-accounting"
_RELEASE_CADENCE = (
    "release-c654c4b-kill-safe cadence; preserve production behavior rather "
    "than impose the discovery 1/8 cadence"
)


class KreaCalibrationProfileError(ValueError):
    """An explicit calibration selector or target config is invalid."""


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("checkpoint is not a regular file")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    finally:
        os.close(fd)
    return digest.hexdigest()


@dataclass(frozen=True)
class Stage2DepthOverride:
    steps: int
    throughput_profile_sha256: str


@dataclass(frozen=True)
class Stage2RunControl:
    """The stochastic/receipt controls that make a Stage-2 run auditable."""

    seed: int
    execution_plan_sha256: str
    receipt_path: str
    target_fraction_numerator: int
    target_fraction_denominator: int

    @property
    def checkpoint_target(self) -> dict[str, Any]:
        """Return the exact frozen target schema consumed by finalization."""

        return {
            "fraction_numerator": self.target_fraction_numerator,
            "fraction_denominator": self.target_fraction_denominator,
            "selection_rule": STAGE2_CHECKPOINT_MAPPING_RULE,
        }


@dataclass(frozen=True)
class Stage2TimingBootstrapControl:
    """Pre-profile control used only to measure one frozen Krea class.

    This is deliberately a different type and environment bundle from the
    post-measurement Stage-2 run control.  In particular it has no throughput
    profile field: substituting a timing-plan digest for a profile digest would
    create false evidence and hide the bootstrap dependency cycle.
    """

    timing_plan_sha256: str
    probe_contract_sha256: str
    steps: int
    seed: int
    receipt_path: str

    @property
    def execution_plan_sha256(self) -> str:
        """Compatibility name for the checkpoint journal's run binding."""

        return self.timing_plan_sha256

    @property
    def checkpoint_target(self) -> dict[str, Any]:
        # Timing measures the class, not a finalist checkpoint policy.  Exact
        # final keeps the held-out natural-completion artifact unambiguous.
        return {
            "fraction_numerator": 1,
            "fraction_denominator": 1,
            "selection_rule": STAGE2_CHECKPOINT_MAPPING_RULE,
        }

    @property
    def target_fraction_numerator(self) -> int:
        return 1

    @property
    def target_fraction_denominator(self) -> int:
        return 1


@dataclass(frozen=True)
class KreaCalibrationProfile:
    profile_id: str
    throughput_equivalence_class: str
    depth_policy: str
    candidate_cadence_policy: str
    learning_rate: float
    rank: int
    alpha: int
    optimizer: str
    optimizer_parameters: tuple[tuple[str, float], ...]
    loss: str
    guidance: float
    dropout: float
    ema: bool
    source_unknown_fields: tuple[str, ...] = ()

    @property
    def uses_budget_fill_cadence(self) -> bool:
        return self.candidate_cadence_policy == _BUDGET_FILL_CADENCE

    def frozen_record(self) -> dict[str, Any]:
        """Return the canonical semantic record used for profile binding."""

        return {
            "schema": _SCHEMA,
            "kind": _PROFILE_KIND,
            "source_freeze_file_sha256": SOURCE_FREEZE_FILE_SHA256,
            "profile_id": self.profile_id,
            "throughput_equivalence_class": self.throughput_equivalence_class,
            "depth_policy": self.depth_policy,
            "candidate_cadence_policy": self.candidate_cadence_policy,
            "recipe": {
                "learning_rate": self.learning_rate,
                "rank": self.rank,
                "alpha": self.alpha,
                "optimizer": self.optimizer,
                "optimizer_parameters": dict(self.optimizer_parameters),
                "loss": self.loss,
                "guidance": self.guidance,
                "dropout": self.dropout,
                "ema": self.ema,
            },
            "source_unknown_fields": list(self.source_unknown_fields),
        }

    @property
    def profile_sha256(self) -> str:
        payload = json.dumps(
            self.frozen_record(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


_ADAMW_PARAMETERS = (("weight_decay", 0.0001),)
_AUTOMAGIC_PARAMETERS = (
    ("lr_bump", 0.000001),
    ("max_lr", 0.001),
    ("min_lr", 0.0000001),
    ("weight_decay", 0.0001),
)


def _profile(
    profile_id: str,
    *,
    throughput_class: str = "A-rank32-adamw8bit-mse-guidance2",
    depth_policy: str = "measured-budget-fill",
    cadence_policy: str = _BUDGET_FILL_CADENCE,
    learning_rate: float = 0.0001,
    rank: int = 32,
    alpha: int = 32,
    optimizer: str = "adamw8bit",
    optimizer_parameters: tuple[tuple[str, float], ...] = _ADAMW_PARAMETERS,
    loss: str = "mse",
    guidance: float = 2,
    dropout: float = 0.05,
    ema: bool = False,
    source_unknown_fields: tuple[str, ...] = (),
) -> KreaCalibrationProfile:
    return KreaCalibrationProfile(
        profile_id=profile_id,
        throughput_equivalence_class=throughput_class,
        depth_policy=depth_policy,
        candidate_cadence_policy=cadence_policy,
        learning_rate=learning_rate,
        rank=rank,
        alpha=alpha,
        optimizer=optimizer,
        optimizer_parameters=optimizer_parameters,
        loss=loss,
        guidance=guidance,
        dropout=dropout,
        ema=ema,
        source_unknown_fields=source_unknown_fields,
    )


_PROFILES = {
    "K0": _profile(
        "K0",
        depth_policy="release-c654c4b-static-policy",
        cadence_policy=_RELEASE_CADENCE,
    ),
    "K1": _profile("K1"),
    "K2": _profile(
        "K2",
        depth_policy="measured-budget-fill-with-step-960-landmark-if-budget-safe",
        learning_rate=0.000086,
        dropout=0.1,
    ),
    "K3": _profile(
        "K3",
        throughput_class="B-rank32-adamw8bit-mae-guidance3",
        depth_policy="measured-budget-fill-with-step-1200-landmark-if-budget-safe",
        loss="mae",
        guidance=3,
        # These are the predeclared local adaptations, not recovered source facts.
        source_unknown_fields=("dropout", "ema"),
    ),
    "K4": _profile(
        "K4",
        throughput_class="C-rank64-automagic-mse-guidance2",
        depth_policy="measured-budget-fill-with-step-840-landmark-if-budget-safe",
        learning_rate=0.00000086,
        rank=64,
        alpha=64,
        optimizer="automagic",
        optimizer_parameters=_AUTOMAGIC_PARAMETERS,
        dropout=0.3,
    ),
    "K5": _profile("K5", learning_rate=0.0002),
}


def available_profile_ids() -> tuple[str, ...]:
    return tuple(_PROFILES)


def profile_for_id(profile_id: str) -> KreaCalibrationProfile:
    """Resolve only an exact frozen identifier; aliases/defaults are forbidden."""

    if not isinstance(profile_id, str) or profile_id not in _PROFILES:
        raise KreaCalibrationProfileError(
            "unknown Krea calibration profile; expected one of K0, K1, K2, "
            "K3, K4, K5"
        )
    return _PROFILES[profile_id]


def selected_profile(
    model_type: str, *, environ: Mapping[str, str] | None = None
) -> KreaCalibrationProfile | None:
    """Return the explicitly selected profile, or ``None`` when truly unset."""

    env = os.environ if environ is None else environ
    raw = env.get(PROFILE_SELECTOR_ENV)
    if raw is None:
        return None
    if model_type != "krea2":
        telemetry.event(
            "krea_calibration_profile_rejected",
            reason="selector_requires_krea2",
            selector_present=True,
        )
        raise KreaCalibrationProfileError(
            f"{PROFILE_SELECTOR_ENV} is calibration-only and requires model_type=krea2"
        )
    try:
        return profile_for_id(raw)
    except KreaCalibrationProfileError:
        telemetry.event(
            "krea_calibration_profile_rejected",
            reason="unknown_profile",
            selector_present=True,
        )
        raise


def selected_stage2_depth(
    model_type: str,
    profile: KreaCalibrationProfile | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Stage2DepthOverride | None:
    """Resolve an exact plan-bound depth without creating a production default."""

    env = os.environ if environ is None else environ
    raw_steps = env.get(STAGE2_STEPS_ENV)
    raw_profile_sha = env.get(STAGE2_THROUGHPUT_SHA_ENV)
    if raw_steps is None and raw_profile_sha is None:
        return None

    def reject(reason: str, message: str) -> None:
        telemetry.event(
            "krea_calibration_depth_rejected",
            reason=reason,
            step_override_present=raw_steps is not None,
            throughput_profile_present=raw_profile_sha is not None,
        )
        raise KreaCalibrationProfileError(message)

    if model_type != "krea2" or profile is None:
        reject(
            "profile_required",
            "Stage-2 calibration depth requires an explicit K1-K5 profile",
        )
    if profile.profile_id == "K0":
        reject("k0_depth_is_frozen", "K0 must preserve release-control depth")
    if raw_steps is None or raw_profile_sha is None:
        reject(
            "partial_depth_binding",
            "Stage-2 steps and throughput-profile SHA-256 are required together",
        )
    assert raw_steps is not None and raw_profile_sha is not None
    if re.fullmatch(r"[1-9][0-9]*", raw_steps) is None:
        reject("invalid_steps", "Stage-2 steps must be a canonical positive integer")
    steps = int(raw_steps)
    if steps > MAX_STAGE2_STEPS:
        reject("steps_above_ceiling", "Stage-2 steps exceed the calibration ceiling")
    if re.fullmatch(r"[0-9a-f]{64}", raw_profile_sha) is None:
        reject(
            "invalid_throughput_profile_sha",
            "Stage-2 throughput profile must be a lowercase SHA-256",
        )
    return Stage2DepthOverride(
        steps=steps,
        throughput_profile_sha256=raw_profile_sha,
    )


def selected_stage2_run_control(
    model_type: str,
    profile: KreaCalibrationProfile | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Stage2RunControl | None:
    """Resolve the all-or-nothing Stage-2 seed and private receipt contract.

    Stage-1 profile experiments intentionally do not set these variables.  A
    Stage-2 execution, however, is not allowed to carry a merely documentary
    seed: the exact seed, plan digest, and private receipt path are supplied as
    one indivisible bundle and applied to ai-toolkit's process-level
    ``training_seed`` key.
    """

    env = os.environ if environ is None else environ
    raw = {
        "seed": env.get(STAGE2_SEED_ENV),
        "plan": env.get(STAGE2_PLAN_SHA_ENV),
        "receipt": env.get(STAGE2_RECEIPT_PATH_ENV),
        "target_numerator": env.get(STAGE2_TARGET_NUMERATOR_ENV),
        "target_denominator": env.get(STAGE2_TARGET_DENOMINATOR_ENV),
    }
    present = {key for key, value in raw.items() if value is not None}
    if not present:
        return None

    def reject(reason: str, message: str) -> None:
        telemetry.event(
            "krea_stage2_run_control_rejected",
            reason=reason,
            seed_present=raw["seed"] is not None,
            plan_present=raw["plan"] is not None,
            receipt_present=raw["receipt"] is not None,
            target_numerator_present=raw["target_numerator"] is not None,
            target_denominator_present=raw["target_denominator"] is not None,
        )
        raise KreaCalibrationProfileError(message)

    if present != set(raw):
        reject(
            "partial_control_bundle",
            "Stage-2 seed, plan, receipt, and checkpoint target are required together",
        )
    if model_type != "krea2" or profile is None:
        reject(
            "explicit_profile_required",
            "Stage-2 run controls require an explicit frozen Krea profile",
        )
    assert raw["seed"] is not None
    assert raw["plan"] is not None
    assert raw["receipt"] is not None
    assert raw["target_numerator"] is not None
    assert raw["target_denominator"] is not None
    if re.fullmatch(r"(?:0|[1-9][0-9]*)", raw["seed"]) is None:
        reject("invalid_seed", "Stage-2 seed must be a canonical uint32 integer")
    seed = int(raw["seed"])
    if seed >= 2**32:
        reject("invalid_seed", "Stage-2 seed must be a canonical uint32 integer")
    if re.fullmatch(r"[0-9a-f]{64}", raw["plan"]) is None:
        reject(
            "invalid_plan_sha",
            "Stage-2 execution-plan digest must be a lowercase SHA-256",
        )
    expected_receipt = f"/run-evidence/{raw['plan']}/config-control.json"
    if raw["receipt"] != expected_receipt:
        reject(
            "invalid_receipt_path",
            "Stage-2 control receipt path must be plan-SHA namespaced",
        )
    if (
        re.fullmatch(r"[1-9][0-9]*", raw["target_numerator"]) is None
        or re.fullmatch(r"[1-9][0-9]*", raw["target_denominator"]) is None
    ):
        reject(
            "invalid_checkpoint_target",
            "Stage-2 checkpoint target must be a canonical positive fraction",
        )
    numerator = int(raw["target_numerator"])
    denominator = int(raw["target_denominator"])
    if numerator > denominator or math.gcd(numerator, denominator) != 1:
        reject(
            "invalid_checkpoint_target",
            "Stage-2 checkpoint target must be reduced and no greater than one",
        )
    return Stage2RunControl(
        seed=seed,
        execution_plan_sha256=raw["plan"],
        receipt_path=raw["receipt"],
        target_fraction_numerator=numerator,
        target_fraction_denominator=denominator,
    )


def selected_stage2_timing_bootstrap_control(
    model_type: str,
    profile: KreaCalibrationProfile | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Stage2TimingBootstrapControl | None:
    """Resolve the all-or-nothing pre-profile timing control bundle.

    The ordinary tournament path and post-measurement Stage-2 path never set
    these names.  The bundle cannot be used for K0, another model type, or in
    combination with the ordinary depth/run-control variables.
    """

    env = os.environ if environ is None else environ
    raw = {
        "timing_plan": env.get(STAGE2_TIMING_PLAN_SHA_ENV),
        "probe_contract": env.get(STAGE2_TIMING_CONTRACT_SHA_ENV),
        "steps": env.get(STAGE2_TIMING_STEPS_ENV),
        "seed": env.get(STAGE2_TIMING_SEED_ENV),
        "receipt": env.get(STAGE2_TIMING_RECEIPT_PATH_ENV),
    }
    present = {key for key, value in raw.items() if value is not None}
    if not present:
        return None

    def reject(reason: str, message: str) -> None:
        telemetry.event(
            "krea_stage2_timing_bootstrap_rejected",
            reason=reason,
            timing_plan_present=raw["timing_plan"] is not None,
            probe_contract_present=raw["probe_contract"] is not None,
            steps_present=raw["steps"] is not None,
            seed_present=raw["seed"] is not None,
            receipt_present=raw["receipt"] is not None,
        )
        raise KreaCalibrationProfileError(message)

    if present != set(raw):
        reject(
            "partial_bootstrap_bundle",
            "Stage-2 timing bootstrap requires plan, contract, steps, seed, and "
            "receipt together",
        )
    if model_type != "krea2" or profile is None or profile.profile_id == "K0":
        reject(
            "explicit_noncontrol_profile_required",
            "Stage-2 timing bootstrap requires explicit Krea K1-K5",
        )
    ordinary_names = (
        STAGE2_STEPS_ENV,
        STAGE2_THROUGHPUT_SHA_ENV,
        STAGE2_SEED_ENV,
        STAGE2_PLAN_SHA_ENV,
        STAGE2_RECEIPT_PATH_ENV,
        STAGE2_TARGET_NUMERATOR_ENV,
        STAGE2_TARGET_DENOMINATOR_ENV,
    )
    if any(env.get(name) is not None for name in ordinary_names):
        reject(
            "ordinary_stage2_controls_present",
            "Stage-2 timing bootstrap cannot mix with post-measurement controls",
        )
    assert all(value is not None for value in raw.values())
    if re.fullmatch(r"[0-9a-f]{64}", str(raw["timing_plan"])) is None:
        reject("invalid_timing_plan", "timing plan must be a lowercase SHA-256")
    if re.fullmatch(r"[0-9a-f]{64}", str(raw["probe_contract"])) is None:
        reject(
            "invalid_probe_contract",
            "timing probe contract must be a lowercase SHA-256",
        )
    if raw["timing_plan"] == raw["probe_contract"]:
        reject("collapsed_plan_contract", "timing plan and contract must be distinct")
    if re.fullmatch(r"[1-9][0-9]*", str(raw["steps"])) is None:
        reject("invalid_steps", "timing bootstrap steps must be canonical positive")
    steps = int(str(raw["steps"]))
    if steps > MAX_STAGE2_STEPS:
        reject("steps_above_ceiling", "timing bootstrap steps exceed the ceiling")
    if re.fullmatch(r"(?:0|[1-9][0-9]*)", str(raw["seed"])) is None:
        reject("invalid_seed", "timing bootstrap seed must be canonical uint32")
    seed = int(str(raw["seed"]))
    if seed >= 2**32:
        reject("invalid_seed", "timing bootstrap seed must be canonical uint32")
    expected_receipt = f"/run-evidence/{raw['timing_plan']}/config-control.json"
    if raw["receipt"] != expected_receipt:
        reject(
            "invalid_receipt_path",
            "timing bootstrap receipt must be timing-plan namespaced",
        )
    return Stage2TimingBootstrapControl(
        timing_plan_sha256=str(raw["timing_plan"]),
        probe_contract_sha256=str(raw["probe_contract"]),
        steps=steps,
        seed=seed,
        receipt_path=str(raw["receipt"]),
    )


def apply_stage2_run_control(
    cfg: dict[str, Any], control: Stage2RunControl | None
) -> dict[str, Any]:
    """Apply the exact ai-toolkit seed; leave ordinary production untouched."""

    if control is None:
        return cfg
    if not isinstance(control, Stage2RunControl):
        raise KreaCalibrationProfileError("Stage-2 run control is not validated")
    resolved = copy.deepcopy(cfg)
    root = _mapping(resolved, "config document")
    config = _mapping(root.get("config"), "config")
    processes = config.get("process")
    if not isinstance(processes, list) or len(processes) != 1:
        raise KreaCalibrationProfileError("Krea config must contain one process")
    process = _mapping(processes[0], "Krea process")
    model = _mapping(process.get("model"), "Krea model")
    if model.get("arch") != "krea2":
        raise KreaCalibrationProfileError("Stage-2 seed target is not Krea 2")
    process["training_seed"] = control.seed
    train = _mapping(process.get("train"), "Krea train config")
    save = _mapping(process.get("save"), "Krea save config")
    checkpoint_selection = stage2_checkpoint_selection_binding(
        control,
        planned_steps=_positive_int(train.get("steps"), "planned steps"),
        save_every=_positive_int(save.get("save_every"), "save_every"),
    )
    meta = root.setdefault("meta", {})
    _mapping(meta, "config metadata")[
        "forge_krea_checkpoint_selection"
    ] = checkpoint_selection
    telemetry.event(
        "krea_stage2_training_seed_applied",
        seed=control.seed,
        execution_plan_sha256=control.execution_plan_sha256,
        selected_step=checkpoint_selection["selected_step"],
        target_fraction_numerator=control.target_fraction_numerator,
        target_fraction_denominator=control.target_fraction_denominator,
    )
    return resolved


def apply_stage2_timing_bootstrap_control(
    cfg: dict[str, Any], control: Stage2TimingBootstrapControl | None
) -> dict[str, Any]:
    """Apply a pre-profile measurement depth/seed with explicit disclosure."""

    if control is None:
        return cfg
    if not isinstance(control, Stage2TimingBootstrapControl):
        raise KreaCalibrationProfileError(
            "Stage-2 timing bootstrap control is not validated"
        )
    resolved = copy.deepcopy(cfg)
    root = _mapping(resolved, "config document")
    config = _mapping(root.get("config"), "config")
    processes = config.get("process")
    if not isinstance(processes, list) or len(processes) != 1:
        raise KreaCalibrationProfileError("Krea config must contain one process")
    process = _mapping(processes[0], "Krea process")
    model = _mapping(process.get("model"), "Krea model")
    if model.get("arch") != "krea2":
        raise KreaCalibrationProfileError("timing bootstrap target is not Krea 2")
    binding = _mapping(
        _mapping(root.get("meta"), "config metadata").get(
            "forge_krea_calibration_profile"
        ),
        "Krea profile binding",
    )
    if (
        binding.get("profile_id") == "K0"
        or binding.get("depth_binding_status")
        != "pending_measured_stage2_per_class_throughput"
        or binding.get("throughput_profile_sha256") is not None
    ):
        raise KreaCalibrationProfileError(
            "timing bootstrap requires one unmeasured K1-K5 profile"
        )
    train = _mapping(process.get("train"), "Krea train config")
    save = _mapping(process.get("save"), "Krea save config")
    process["training_seed"] = control.seed
    train["steps"] = control.steps
    save["save_every"] = (control.steps + 7) // 8
    selection = stage2_checkpoint_selection_binding(
        control,
        planned_steps=control.steps,
        save_every=save["save_every"],
    )
    root["meta"]["forge_krea_checkpoint_selection"] = selection
    binding.update(
        planned_steps=control.steps,
        save_every=save["save_every"],
        planned_steps_source="explicit_preprofile_timing_probe",
        depth_binding_status="preprofile_timing_bootstrap",
        throughput_profile_sha256=None,
        measured_stage2_per_class_throughput_bound=False,
        timing_bootstrap={
            "timing_plan_sha256": control.timing_plan_sha256,
            "probe_contract_sha256": control.probe_contract_sha256,
            "release_authorized": False,
        },
    )
    telemetry.event(
        "krea_stage2_timing_bootstrap_applied",
        timing_plan_sha256=control.timing_plan_sha256,
        probe_contract_sha256=control.probe_contract_sha256,
        profile_id=binding["profile_id"],
        steps=control.steps,
        seed=control.seed,
        release_authorized=False,
    )
    return resolved


def stage2_checkpoint_selection_binding(
    control: Stage2RunControl | Stage2TimingBootstrapControl,
    *,
    planned_steps: int,
    save_every: int,
) -> dict[str, Any]:
    """Derive the checkpoint that implements a frozen curve fraction.

    Integer cross-products avoid float drift. An exact distance tie selects
    the earlier current-run checkpoint, matching the frozen decision rule.
    """

    if not isinstance(control, (Stage2RunControl, Stage2TimingBootstrapControl)):
        raise KreaCalibrationProfileError("Stage-2 run control is not validated")
    planned = _positive_int(planned_steps, "planned steps")
    cadence = _positive_int(save_every, "save_every")
    candidate_steps = list(range(cadence, planned, cadence))
    candidate_steps.append(planned)
    candidate_steps = sorted(set(candidate_steps))
    numerator = control.target_fraction_numerator
    denominator = control.target_fraction_denominator
    selected_step = min(
        candidate_steps,
        key=lambda step: (
            abs(step * denominator - planned * numerator),
            step,
        ),
    )
    return {
        "schema": 1,
        "mapping_rule": STAGE2_CHECKPOINT_MAPPING_RULE,
        "target_fraction": {
            "numerator": numerator,
            "denominator": denominator,
        },
        "planned_steps": planned,
        "selected_step": selected_step,
        "candidate_steps": candidate_steps,
    }


def write_stage2_terminal_receipt(
    *,
    profile_id: str,
    planned_steps: int,
    last_step: int | None,
    returncode: int | None,
    stopped_by_deadline: bool,
) -> dict[str, Any] | None:
    """Create the private post-training proof that the full plan completed."""

    profile = profile_for_id(profile_id)
    control = selected_stage2_run_control("krea2", profile)
    timing_bootstrap = selected_stage2_timing_bootstrap_control("krea2", profile)
    if control is not None and timing_bootstrap is not None:
        raise KreaCalibrationProfileError("Stage-2 control modes are not exclusive")
    active_control = control if control is not None else timing_bootstrap
    if active_control is None:
        return None
    if (
        isinstance(planned_steps, bool)
        or not isinstance(planned_steps, int)
        or planned_steps <= 0
    ):
        raise KreaCalibrationProfileError("Stage-2 planned steps are invalid")
    if last_step is not None and (
        isinstance(last_step, bool) or not isinstance(last_step, int) or last_step < 0
    ):
        raise KreaCalibrationProfileError("Stage-2 observed last step is invalid")
    if returncode is not None and (
        isinstance(returncode, bool) or not isinstance(returncode, int)
    ):
        raise KreaCalibrationProfileError("Stage-2 trainer returncode is invalid")
    if not isinstance(stopped_by_deadline, bool):
        raise KreaCalibrationProfileError("Stage-2 deadline state is invalid")
    parent = os.path.dirname(active_control.receipt_path)
    control_receipt = active_control.receipt_path
    if os.path.realpath(parent) != parent or not os.path.isdir(parent):
        raise KreaCalibrationProfileError(
            "Stage-2 terminal receipt parent is not a real directory"
        )
    try:
        with open(control_receipt, "rb") as handle:
            control_bytes = handle.read()
    except OSError as exc:
        raise KreaCalibrationProfileError(
            "Stage-2 config-control receipt is unavailable"
        ) from exc
    try:
        control_record = json.loads(control_bytes)
        checkpoint_selection = control_record["checkpoint_selection"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise KreaCalibrationProfileError(
            "Stage-2 config-control receipt lacks checkpoint selection"
        ) from exc
    naturally_completed = (
        returncode == 0 and stopped_by_deadline is False and last_step == planned_steps
    )
    common = {
        "profile_id": profile.profile_id,
        "profile_sha256": profile.profile_sha256,
        "training_seed": active_control.seed,
        "planned_steps": planned_steps,
        "last_step": last_step,
        "trainer_returncode": returncode,
        "stopped_by_deadline": stopped_by_deadline,
        "planned_steps_completed": last_step == planned_steps,
        "natural_completion": naturally_completed,
        "config_control_file_sha256": hashlib.sha256(control_bytes).hexdigest(),
        "checkpoint_selection": checkpoint_selection,
        "release_authorized": False,
    }
    if timing_bootstrap is None:
        body = {
            "schema": 1,
            "kind": "forge-krea-stage2-training-terminal-receipt",
            "execution_plan_sha256": control.execution_plan_sha256,
            **common,
        }
    else:
        body = {
            "schema": 1,
            "kind": "forge-krea-stage2-timing-bootstrap-terminal-receipt",
            "mode": "preprofile_timing_bootstrap",
            "timing_plan_sha256": timing_bootstrap.timing_plan_sha256,
            "probe_contract_sha256": timing_bootstrap.probe_contract_sha256,
            "throughput_profile_sha256": None,
            "production_mutation_authorized": False,
            **common,
        }
    receipt = {
        **body,
        "receipt_sha256": hashlib.sha256(
            json.dumps(
                body,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }
    payload = (
        json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    output = os.path.join(parent, "training-terminal.json")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(output, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    return receipt


def preserve_stage2_checkpoint_selection(
    *,
    profile_id: str,
    save_root: str,
    record: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Copy the exact private selection proof to the persistent evidence mount.

    The public-bundle scrub intentionally removes the checkpoint-side record
    before validator upload. Stage-2 therefore preserves the already-written
    bytes in its plan-namespaced evidence directory before that scrub runs.
    """

    profile = profile_for_id(profile_id)
    control = selected_stage2_run_control("krea2", profile)
    timing_bootstrap = selected_stage2_timing_bootstrap_control("krea2", profile)
    if control is not None and timing_bootstrap is not None:
        raise KreaCalibrationProfileError("Stage-2 control modes are not exclusive")
    active_control = control if control is not None else timing_bootstrap
    if active_control is None:
        return None
    if not isinstance(record, Mapping):
        raise KreaCalibrationProfileError("Stage-2 checkpoint record is invalid")
    source = os.path.join(save_root, "forge_checkpoint_selection.json")
    try:
        if not os.path.isfile(source) or os.path.islink(source):
            raise OSError
        with open(source, "rb") as handle:
            payload = handle.read()
        parsed = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise KreaCalibrationProfileError(
            "Stage-2 checkpoint-selection source is unavailable"
        ) from exc
    if not isinstance(parsed, dict) or parsed != dict(record):
        raise KreaCalibrationProfileError(
            "Stage-2 checkpoint-selection source differs from finalization"
        )
    try:
        with open(active_control.receipt_path, "rb") as handle:
            config_control = json.load(handle)
        expected = config_control["checkpoint_selection"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise KreaCalibrationProfileError(
            "Stage-2 config-control checkpoint selection is unavailable"
        ) from exc
    selected_file = parsed.get("selected_file")
    output_file = parsed.get("output_file")
    if (
        parsed.get("schema") != 1
        or parsed.get("status") != "selected_current_run"
        or parsed.get("source") != "frozen_checkpoint_fraction"
        or parsed.get("context") != "training"
        or not isinstance(selected_file, str)
        or os.path.basename(selected_file) != selected_file
        or output_file != "last.safetensors"
        or parsed.get("selected_step") != expected.get("selected_step")
        or parsed.get("planned_steps") != expected.get("planned_steps")
        or parsed.get("checkpoint_target") != active_control.checkpoint_target
    ):
        raise KreaCalibrationProfileError(
            "Stage-2 checkpoint selection differs from its frozen target"
        )
    selected_path = os.path.join(save_root, selected_file)
    output_path = os.path.join(save_root, output_file)
    try:
        if (
            os.path.islink(selected_path)
            or os.path.islink(output_path)
            or not os.path.isfile(selected_path)
            or not os.path.isfile(output_path)
        ):
            raise OSError
        selected_sha256 = _file_sha256(selected_path)
        output_sha256 = _file_sha256(output_path)
    except OSError as exc:
        raise KreaCalibrationProfileError(
            "Stage-2 selected or promoted checkpoint is unavailable"
        ) from exc
    if selected_sha256 != output_sha256 or output_sha256 != parsed.get("sha256"):
        raise KreaCalibrationProfileError(
            "Stage-2 promoted checkpoint bytes differ from the selected source"
        )
    parent = os.path.dirname(active_control.receipt_path)
    target = os.path.join(parent, "forge_checkpoint_selection.json")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(target, flags, 0o444)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    return {
        "path": target,
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "selected_checkpoint_sha256": selected_sha256,
        "selected_step": parsed["selected_step"],
    }


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise KreaCalibrationProfileError(f"{label} must be a mapping")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise KreaCalibrationProfileError(f"{label} must be a positive integer")
    return value


def apply_profile(
    cfg: dict[str, Any],
    profile: KreaCalibrationProfile,
    *,
    depth_override: Stage2DepthOverride | None = None,
) -> dict[str, Any]:
    """Apply a frozen profile to a built Krea config and emit its binding.

    A copy is returned so a rejected explicit calibration request cannot leave
    the caller's baseline config partially mutated.
    """

    try:
        frozen = profile_for_id(profile.profile_id)
    except (AttributeError, KreaCalibrationProfileError) as exc:
        raise KreaCalibrationProfileError(
            "profile is not a frozen K0-K5 record"
        ) from exc
    if profile != frozen:
        raise KreaCalibrationProfileError("profile differs from its frozen binding")

    resolved = copy.deepcopy(cfg)
    root = _mapping(resolved, "config document")
    config = _mapping(root.get("config"), "config")
    processes = config.get("process")
    if not isinstance(processes, list) or len(processes) != 1:
        raise KreaCalibrationProfileError("Krea config must contain one process")
    process = _mapping(processes[0], "Krea process")
    model = _mapping(process.get("model"), "Krea model")
    if model.get("arch") != "krea2":
        raise KreaCalibrationProfileError("calibration profile target is not Krea 2")
    save = _mapping(process.get("save"), "Krea save config")
    # A calibration selector is never permission to add a network/release path.
    if save.get("push_to_hub") is not False:
        raise KreaCalibrationProfileError(
            "calibration profile requires push_to_hub to remain false"
        )
    if profile.profile_id == "K0":
        network = _mapping(process.get("network"), "Krea network config")
        train = _mapping(process.get("train"), "Krea train config")
        datasets = process.get("datasets")
        if not isinstance(datasets, list) or len(datasets) != 1:
            raise KreaCalibrationProfileError("Krea config must contain one dataset")
        dataset = _mapping(datasets[0], "Krea dataset config")
        expected = {
            "rank": frozen.rank,
            "alpha": frozen.alpha,
            "optimizer": frozen.optimizer,
            "optimizer_parameters": dict(frozen.optimizer_parameters),
            "loss": frozen.loss,
            "guidance_enabled": True,
            "guidance": frozen.guidance,
            "learning_rate": frozen.learning_rate,
            "dropout": frozen.dropout,
            "ema": {"use_ema": frozen.ema, "ema_decay": 0.99},
        }
        actual = {
            "rank": network.get("linear"),
            "alpha": network.get("linear_alpha"),
            "optimizer": train.get("optimizer"),
            "optimizer_parameters": train.get("optimizer_params"),
            "loss": train.get("loss_type"),
            "guidance_enabled": train.get("do_differential_guidance"),
            "guidance": train.get("differential_guidance_scale"),
            "learning_rate": train.get("lr"),
            "dropout": dataset.get("caption_dropout_rate"),
            "ema": train.get("ema_config"),
        }
        if actual != expected:
            raise KreaCalibrationProfileError(
                "K0 release config differs from its frozen recipe axes"
            )
        # K0 is the literal release control. Selecting it verifies, but does
        # not mutate, those frozen axes or the release depth/save policy.
        return resolved
    network = _mapping(process.get("network"), "Krea network config")
    train = _mapping(process.get("train"), "Krea train config")
    datasets = process.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise KreaCalibrationProfileError("Krea config must contain one dataset")
    dataset = _mapping(datasets[0], "Krea dataset config")
    planned_steps = _positive_int(train.get("steps"), "planned steps")
    current_save_every = _positive_int(save.get("save_every"), "save_every")

    network["linear"] = profile.rank
    network["linear_alpha"] = profile.alpha
    train["lr"] = profile.learning_rate
    train["optimizer"] = profile.optimizer
    train["optimizer_params"] = dict(profile.optimizer_parameters)
    train["loss_type"] = profile.loss
    train["do_differential_guidance"] = True
    train["differential_guidance_scale"] = profile.guidance
    dataset["caption_dropout_rate"] = profile.dropout
    train["ema_config"] = {"use_ema": profile.ema, "ema_decay": 0.99}

    if profile.uses_budget_fill_cadence:
        if depth_override is not None:
            if (
                not isinstance(depth_override, Stage2DepthOverride)
                or depth_override.steps <= 0
                or depth_override.steps > MAX_STAGE2_STEPS
                or re.fullmatch(
                    r"[0-9a-f]{64}", depth_override.throughput_profile_sha256
                )
                is None
            ):
                raise KreaCalibrationProfileError(
                    "Stage-2 depth override differs from its validated binding"
                )
            planned_steps = depth_override.steps
            train["steps"] = planned_steps
        save["save_every"] = (planned_steps + 7) // 8
        if depth_override is None:
            depth_binding_status = "pending_measured_stage2_per_class_throughput"
            planned_steps_source = "existing_production_planner"
            throughput_profile_sha256 = None
        else:
            depth_binding_status = "owner_ratified_stage2_measured_profile"
            planned_steps_source = "explicit_owner_ratified_stage2_plan"
            throughput_profile_sha256 = depth_override.throughput_profile_sha256
    else:
        if depth_override is not None:
            raise KreaCalibrationProfileError("K0 must preserve release-control depth")
        # K0 is the frozen release control, including its current save policy.
        save["save_every"] = current_save_every
        depth_binding_status = "release_control"
        planned_steps_source = "current_production_release_control"
        throughput_profile_sha256 = None

    binding = {
        "schema": _SCHEMA,
        "kind": _BINDING_KIND,
        "calibration_only": True,
        "profile_id": profile.profile_id,
        "profile_sha256": profile.profile_sha256,
        "source_freeze_file_sha256": SOURCE_FREEZE_FILE_SHA256,
        "throughput_equivalence_class": profile.throughput_equivalence_class,
        "depth_policy": profile.depth_policy,
        "candidate_cadence_policy": profile.candidate_cadence_policy,
        "planned_steps": planned_steps,
        "save_every": save["save_every"],
        "planned_steps_source": planned_steps_source,
        "depth_binding_status": depth_binding_status,
        "throughput_profile_sha256": throughput_profile_sha256,
        "measured_stage2_per_class_throughput_bound": depth_override is not None,
        "release_selected": False,
    }
    meta = root.setdefault("meta", {})
    _mapping(meta, "config metadata")["forge_krea_calibration_profile"] = binding

    telemetry.event(
        "krea_calibration_profile_applied",
        profile_id=profile.profile_id,
        profile_sha256=profile.profile_sha256,
        throughput_equivalence_class=profile.throughput_equivalence_class,
        planned_steps=planned_steps,
        save_every=save["save_every"],
        depth_binding_status=depth_binding_status,
        throughput_profile_sha256=throughput_profile_sha256,
        measured_stage2_per_class_throughput_bound=depth_override is not None,
        release_selected=False,
    )
    return resolved
