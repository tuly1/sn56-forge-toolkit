"""Dormant, evidence-bound Week-5 Krea production policy.

The policy is intentionally unavailable unless a release commit contains an
explicit, hash-bound activation record.  Environment variables cannot enable
it.  Until the Stage-2 confirmation decision passes and a separate release
record is authored, :data:`PRODUCTION_ACTIVATION` remains ``None`` and every
call is an exact no-op over the existing K0 production path.

The measured depth table is the conservative boundary-plan surface: no depth
is interpolated.  A granted budget between certified anchors uses the largest
certified budget not exceeding it; budgets above one hour clamp to the one-hour
anchor.  Budgets below the certified half-hour floor fail safely to K0.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re
from typing import Any, Mapping

from forge import recipe, telemetry


POLICY_ID = "week5-krea-two-regime-v1"
POLICY_KIND = "forge-krea-week5-production-router-predeclaration"
ACTIVATION_KIND = "forge-krea-week5-production-router-activation"
CHECKPOINT_MAPPING_RULE = "nearest_current_candidate_ties_choose_earlier_step"
SOURCE_PRODUCTION_COMMIT_SHA1 = "c654c4b24376f7aa9e12dcb82f5e73dcddee3bdb"
SOURCE_STAGE2_PROFILE_SHA256 = {
    "K1": "9ef2312471a5a9a5cad52ee7c35b2a2458a87d1b404dbc67659e369c6df95257",
    "K5": "8d2b63f868439ded257df56900e01ddd6b347546dda4919b5bb8084798137337",
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
_HOURS = (Decimal("0.5"), Decimal("0.75"), Decimal("1.0"))
_HOUR_LABEL = {
    Decimal("0.5"): "0p5",
    Decimal("0.75"): "0p75",
    Decimal("1.0"): "1",
}
_BOUNDARY_STEPS = {
    ("small", Decimal("0.5")): 136,
    ("large", Decimal("0.5")): 139,
    ("small", Decimal("0.75")): 209,
    ("large", Decimal("0.75")): 213,
    ("small", Decimal("1.0")): 295,
    ("large", Decimal("1.0")): 301,
}
_RECIPE = {
    "K1": {
        "learning_rate": 0.0001,
        "rank": 32,
        "alpha": 32,
        "optimizer": "adamw8bit",
        "optimizer_parameters": {"weight_decay": 0.0001},
        "loss": "mse",
        "guidance": 2,
        "dropout": 0.05,
        "ema": False,
    },
    "K5": {
        "learning_rate": 0.0002,
        "rank": 32,
        "alpha": 32,
        "optimizer": "adamw8bit",
        "optimizer_parameters": {"weight_decay": 0.0001},
        "loss": "mse",
        "guidance": 2,
        "dropout": 0.05,
        "ema": False,
    },
}
_TARGET = {"K1": (9, 10), "K5": (1, 2)}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


_POLICY_BODY = {
    "schema": 1,
    "kind": POLICY_KIND,
    "policy_id": POLICY_ID,
    "source_production_commit_sha1": SOURCE_PRODUCTION_COMMIT_SHA1,
    "routing": {
        "input": "post_dataset_preparation_training_pair_count",
        "holdout_required_disabled": True,
        "small": {
            "pair_count_max_inclusive": 27,
            "source_fixture_regime": "D1",
            "family": "K1",
            "target_fraction": {"numerator": 9, "denominator": 10},
        },
        "large": {
            "pair_count_min_inclusive": 28,
            "source_fixture_regime": "D2",
            "family": "K5",
            "target_fraction": {"numerator": 1, "denominator": 2},
        },
        "out_of_fixture_range": "clamp_to_threshold_selected_nearest_regime",
        "invalid_or_unavailable_pair_count": "preserve_K0_exact_final_policy",
    },
    "measured_budget_fill": {
        "source": "sealed_Stage2_boundary_execution_plans",
        "certified_hours": ["0.5", "0.75", "1.0"],
        "uncertified_budget_rule": (
            "use_largest_certified_budget_not_exceeding_grant; clamp_above_1h; "
            "below_0.5h_preserve_K0"
        ),
        "interpolation": False,
        "remaining_time_step_cap": {
            "rule": (
                "minimum(boundary_anchor,max(1,floor((remaining_hard_seconds*"
                "MARGIN-STARTUP_S-EXPORT_RESERVE_S)/SEC_PER_IT[krea2])))"
            ),
            "source": "release-c654c4b recipe time constants",
            "margin": recipe.MARGIN,
            "startup_seconds": recipe.STARTUP_S,
            "export_reserve_seconds": recipe.EXPORT_RESERVE_S,
            "seconds_per_iteration": recipe.SEC_PER_IT["krea2"],
        },
        "steps": {
            "B-0p5-small": 136,
            "B-0p5-large": 139,
            "B-0p75-small": 209,
            "B-0p75-large": 213,
            "B-1-small": 295,
            "B-1-large": 301,
        },
    },
    "profiles": {
        family: {
            "source_stage2_profile_sha256": SOURCE_STAGE2_PROFILE_SHA256[family],
            "recipe": _RECIPE[family],
        }
        for family in ("K1", "K5")
    },
    "checkpoint_policy": {
        "candidate_cadence": "ceil(planned_steps/8)",
        "mapping_rule": CHECKPOINT_MAPPING_RULE,
        "exact_target_preferred": True,
        "target_miss": (
            "salvage_nearest_valid_current_run_candidate_and_record_target_miss"
        ),
        "no_current_candidate": "preserve_valid_prior_last_if_available",
    },
    "activation_contract": {
        "mechanism": "source_control_constant_record_only_no_environment_toggle",
        "conditional_release": {
            "K1_PASS_K5_PASS": "size_router",
            "K1_PASS_K5_FAIL": "K1_global_at_9/10",
            "K1_FAIL_K5_PASS": "K5_global_at_1/2",
            "K1_FAIL_K5_FAIL": "preserve_K0_no_activation",
        },
        "overall_outcome_rule": "overall_confirmation_passed_iff_both_policies_PASS",
        "required_boundary_plan_bindings": {
            family: [
                "B-0p5-small",
                "B-0p5-large",
                "B-0p75-small",
                "B-0p75-large",
                "B-1-small",
                "B-1-large",
            ]
            for family in ("K1", "K5")
        },
        "separate_release_record_required": True,
    },
    "claim_limits": [
        "depth constants are sealed boundary-plan mechanics, not a quality claim",
        "pair-count routing outside the D1/D2 fixture ranges is an explicit clamp",
        "no interpolation or hidden-pool generality is claimed",
        "this predeclaration does not authorize production mutation, release, or deployment",
    ],
    "production_mutation_authorized": False,
    "release_authorized": False,
    "deployment_authorized": False,
}
POLICY_SHA256 = hashlib.sha256(_canonical_bytes(_POLICY_BODY)).hexdigest()
POLICY = {**_POLICY_BODY, "policy_sha256": POLICY_SHA256}

# The release candidate is dormant.  A later reviewed release commit may replace
# ``None`` with a literal record that passes ``_validated_activation``.  Keeping
# this in source control makes activation visible in the commit diff; there is
# deliberately no environment-variable escape hatch.
PRODUCTION_ACTIVATION: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Selection:
    activation_mode: str
    regime: str
    source_fixture_regime: str
    family: str
    pair_count: int
    granted_hours: str
    remaining_hours: str
    certified_hours: str
    boundary_cell: str
    planned_steps: int
    remaining_time_step_cap: int
    target_numerator: int
    target_denominator: int
    boundary_plan_sha256: str
    activation_sha256: str

    @property
    def checkpoint_target(self) -> dict[str, Any]:
        return {
            "fraction_numerator": self.target_numerator,
            "fraction_denominator": self.target_denominator,
            "selection_rule": CHECKPOINT_MAPPING_RULE,
        }


def _validated_activation(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        if not isinstance(value, Mapping):
            raise ValueError("activation is not a mapping")
        record = dict(value)
        required = {
            "schema",
            "kind",
            "policy_sha256",
            "formal_endgame_decision_sha256",
            "boundary_plan_sha256s",
            "release_record_sha256",
            "overall_confirmation_passed",
            "policy_outcomes",
            "production_mutation_authorized",
            "release_authorized",
            "deployment_authorized",
            "activation_sha256",
        }
        if set(record) != required:
            raise ValueError("activation keys differ")
        body = {key: item for key, item in record.items() if key != "activation_sha256"}
        boundaries = record["boundary_plan_sha256s"]
        required_boundaries = _POLICY_BODY["activation_contract"][
            "required_boundary_plan_bindings"
        ]
        if not isinstance(boundaries, Mapping) or set(boundaries) != {"K1", "K5"}:
            raise ValueError("boundary plan coverage differs")
        for family in ("K1", "K5"):
            rows = boundaries[family]
            if not isinstance(rows, Mapping) or set(rows) != set(
                required_boundaries[family]
            ):
                raise ValueError("family boundary plan coverage differs")
        outcomes = record["policy_outcomes"]
        if not isinstance(outcomes, Mapping) or set(outcomes) != {"K1", "K5"}:
            raise ValueError("policy outcome coverage differs")
        if any(value not in {"PASS", "FAIL"} for value in outcomes.values()):
            raise ValueError("policy outcome is invalid")
        both_pass = all(value == "PASS" for value in outcomes.values())
        any_pass = any(value == "PASS" for value in outcomes.values())
        if (
            record["schema"] != 1
            or record["kind"] != ACTIVATION_KIND
            or record["policy_sha256"] != POLICY_SHA256
            or not isinstance(record["overall_confirmation_passed"], bool)
            or record["overall_confirmation_passed"] is not both_pass
            or not any_pass
            or record["production_mutation_authorized"] is not True
            or record["release_authorized"] is not True
            or record["deployment_authorized"] is not False
            or record["activation_sha256"]
            != hashlib.sha256(_canonical_bytes(body)).hexdigest()
        ):
            raise ValueError("activation identity or authority differs")
        for digest in (
            record["formal_endgame_decision_sha256"],
            record["release_record_sha256"],
            *(
                digest
                for family in ("K1", "K5")
                for digest in boundaries[family].values()
            ),
        ):
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ValueError("activation digest is invalid")
        return record
    except Exception as exc:
        telemetry.event(
            "krea_production_policy_inactive",
            reason="invalid_activation_record",
            error_type=type(exc).__name__,
        )
        return None


def _pair_regime(pair_count: Any) -> tuple[str, str, str] | None:
    if (
        isinstance(pair_count, bool)
        or not isinstance(pair_count, int)
        or pair_count <= 0
    ):
        return None
    return ("small", "D1", "K1") if pair_count <= 27 else ("large", "D2", "K5")


def _certified_budget(hours: Any) -> Decimal | None:
    try:
        if isinstance(hours, bool):
            return None
        grant = Decimal(str(hours))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not grant.is_finite() or grant < _HOURS[0]:
        return None
    eligible = [anchor for anchor in _HOURS if anchor <= grant]
    return max(eligible) if eligible else None


def _remaining_time_step_cap(hours: Any) -> tuple[str, int] | None:
    """Return the pure wall-time cap, independent of the legacy size law."""

    try:
        if isinstance(hours, bool):
            return None
        remaining = Decimal(str(hours))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not remaining.is_finite():
        return None
    budget_s = max(Decimal(0), remaining * Decimal(3600))
    train_s = (
        budget_s * Decimal(str(recipe.MARGIN))
        - Decimal(str(recipe.STARTUP_S))
        - Decimal(str(recipe.EXPORT_RESERVE_S))
    )
    if train_s <= 0:
        cap = 1
    else:
        cap = int(train_s / Decimal(str(recipe.SEC_PER_IT["krea2"])))
    return str(remaining.normalize()), max(1, cap)


def select(
    model_type: str,
    *,
    training_pair_count: Any,
    holdout_pairs: Any,
    granted_hours: Any,
    remaining_hours: Any,
    activation: Mapping[str, Any] | None = None,
) -> Selection | None:
    """Resolve the env-unset policy or return ``None`` for exact K0 behavior."""

    active = _validated_activation(
        PRODUCTION_ACTIVATION if activation is None else activation
    )
    if active is None or model_type != "krea2":
        return None
    if holdout_pairs != 0 or isinstance(holdout_pairs, bool):
        return None
    regime = _pair_regime(training_pair_count)
    budget = _certified_budget(granted_hours)
    time_cap = _remaining_time_step_cap(remaining_hours)
    if regime is None or budget is None or time_cap is None:
        return None
    remaining_hours_string, remaining_time_step_cap = time_cap
    label, source_fixture, family = regime
    outcomes = active["policy_outcomes"]
    if outcomes == {"K1": "PASS", "K5": "PASS"}:
        activation_mode = "size_router"
    elif outcomes == {"K1": "PASS", "K5": "FAIL"}:
        activation_mode = "K1_global"
        source_fixture, family = "D1", "K1"
    elif outcomes == {"K1": "FAIL", "K5": "PASS"}:
        activation_mode = "K5_global"
        source_fixture, family = "D2", "K5"
    else:  # Validation rejects this, retained as a defensive K0 fallback.
        return None
    boundary_cell = f"B-{_HOUR_LABEL[budget]}-{label}"
    numerator, denominator = _TARGET[family]
    return Selection(
        activation_mode=activation_mode,
        regime=label,
        source_fixture_regime=source_fixture,
        family=family,
        pair_count=training_pair_count,
        granted_hours=str(Decimal(str(granted_hours)).normalize()),
        remaining_hours=remaining_hours_string,
        certified_hours=str(budget),
        boundary_cell=boundary_cell,
        planned_steps=_BOUNDARY_STEPS[(label, budget)],
        remaining_time_step_cap=remaining_time_step_cap,
        target_numerator=numerator,
        target_denominator=denominator,
        boundary_plan_sha256=active["boundary_plan_sha256s"][family][boundary_cell],
        activation_sha256=active["activation_sha256"],
    )


def _checkpoint_binding(
    selection: Selection, *, planned_steps: int, save_every: int
) -> dict[str, Any]:
    steps = planned_steps
    candidates = list(range(save_every, steps, save_every))
    candidates.append(steps)
    candidates = sorted(set(candidates))
    selected_step = min(
        candidates,
        key=lambda step: (
            abs(
                step * selection.target_denominator - steps * selection.target_numerator
            ),
            step,
        ),
    )
    return {
        "schema": 1,
        "mapping_rule": CHECKPOINT_MAPPING_RULE,
        "target_fraction": {
            "numerator": selection.target_numerator,
            "denominator": selection.target_denominator,
        },
        "planned_steps": steps,
        "selected_step": selected_step,
        "candidate_steps": candidates,
    }


def apply(cfg: dict[str, Any], selection: Selection) -> dict[str, Any]:
    """Apply one validated production profile without calibration env state."""

    if not isinstance(selection, Selection):
        raise ValueError("Krea production selection is not validated")
    resolved = copy.deepcopy(cfg)
    process = resolved["config"]["process"][0]
    if (
        process["model"].get("arch") != "krea2"
        or process["save"].get("push_to_hub") is not False
    ):
        raise ValueError("Krea production policy target is invalid")
    recipe = _RECIPE[selection.family]
    network = process["network"]
    train = process["train"]
    dataset = process["datasets"][0]
    save = process["save"]
    network["linear"] = recipe["rank"]
    network["linear_alpha"] = recipe["alpha"]
    train["lr"] = recipe["learning_rate"]
    train["optimizer"] = recipe["optimizer"]
    train["optimizer_params"] = dict(recipe["optimizer_parameters"])
    train["loss_type"] = recipe["loss"]
    train["do_differential_guidance"] = True
    train["differential_guidance_scale"] = recipe["guidance"]
    train["ema_config"] = {"use_ema": recipe["ema"], "ema_decay": 0.99}
    dataset["caption_dropout_rate"] = recipe["dropout"]
    # The sealed boundary anchor comes from the original grant.  Preserve the
    # existing conservative timing constants as a pure remaining-time safety
    # cap.  The legacy K0 dataset-size law must not truncate measured anchors.
    planned_steps = min(selection.planned_steps, selection.remaining_time_step_cap)
    train["steps"] = planned_steps
    save["save_every"] = (planned_steps + 7) // 8
    checkpoint = _checkpoint_binding(
        selection,
        planned_steps=planned_steps,
        save_every=save["save_every"],
    )
    meta = resolved.setdefault("meta", {})
    meta["forge_krea_production_policy"] = {
        "schema": 1,
        "kind": "forge-krea-production-policy-binding",
        "policy_id": POLICY_ID,
        "policy_sha256": POLICY_SHA256,
        "activation_sha256": selection.activation_sha256,
        "activation_mode": selection.activation_mode,
        "pair_count": selection.pair_count,
        "regime": selection.regime,
        "source_fixture_regime": selection.source_fixture_regime,
        "family": selection.family,
        "source_stage2_profile_sha256": SOURCE_STAGE2_PROFILE_SHA256[selection.family],
        "granted_hours": selection.granted_hours,
        "remaining_hours": selection.remaining_hours,
        "certified_hours": selection.certified_hours,
        "boundary_cell": selection.boundary_cell,
        "boundary_plan_sha256": selection.boundary_plan_sha256,
        "boundary_planned_steps": selection.planned_steps,
        "remaining_time_step_cap": selection.remaining_time_step_cap,
        "planned_steps": planned_steps,
        "remaining_time_step_cap_applied": planned_steps < selection.planned_steps,
        "holdout_disabled": True,
        "release_selected": True,
    }
    meta["forge_krea_checkpoint_selection"] = checkpoint
    telemetry.event(
        "krea_production_policy_applied",
        activation_mode=selection.activation_mode,
        family=selection.family,
        pair_count=selection.pair_count,
        planned_steps=planned_steps,
        target_numerator=selection.target_numerator,
        target_denominator=selection.target_denominator,
        remaining_time_step_cap=selection.remaining_time_step_cap,
        remaining_time_step_cap_applied=planned_steps < selection.planned_steps,
    )
    return resolved


def checkpoint_control(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], int] | None:
    """Return the config-bound production target consumed by finalization."""

    try:
        meta = cfg.get("meta")
        if not isinstance(meta, Mapping):
            return None
        policy = meta.get("forge_krea_production_policy")
        checkpoint = meta.get("forge_krea_checkpoint_selection")
        if policy is None and checkpoint is None:
            return None
        if not isinstance(policy, Mapping) or not isinstance(checkpoint, Mapping):
            raise ValueError("partial Krea production policy binding")
        if (
            policy.get("policy_id") != POLICY_ID
            or policy.get("policy_sha256") != POLICY_SHA256
            or policy.get("release_selected") is not True
            or checkpoint.get("mapping_rule") != CHECKPOINT_MAPPING_RULE
            or checkpoint.get("planned_steps") != policy.get("planned_steps")
        ):
            raise ValueError("Krea production policy binding drifted")
        target = checkpoint.get("target_fraction")
        if not isinstance(target, Mapping) or set(target) != {
            "numerator",
            "denominator",
        }:
            raise ValueError("Krea checkpoint target binding drifted")
        numerator = target["numerator"]
        denominator = target["denominator"]
        selected_step = checkpoint.get("selected_step")
        planned_steps = checkpoint.get("planned_steps")
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in (numerator, denominator, selected_step, planned_steps)
            )
            or numerator > denominator
            or math.gcd(numerator, denominator) != 1
            or selected_step > planned_steps
        ):
            raise ValueError("Krea checkpoint control is invalid")
        return (
            {
                "fraction_numerator": numerator,
                "fraction_denominator": denominator,
                "selection_rule": CHECKPOINT_MAPPING_RULE,
            },
            selected_step,
        )
    except Exception as exc:
        # A partially applied production policy must not silently revert to a
        # different checkpoint.  The caller turns this into the normal handler
        # fallback rather than producing an ambiguously selected artifact.
        raise ValueError("Krea production checkpoint binding is invalid") from exc
