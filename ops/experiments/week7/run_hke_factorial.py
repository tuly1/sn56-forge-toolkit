#!/usr/bin/env python3
"""CPU authority for the Week-7 Krea HKE factor screen.

This module deliberately does not train, score, route, promote, or deploy. It
freezes the experiment contract before a rental, then consumes timing evidence
from the mechanical H100 gate to produce the plan that must be frozen before
quality training:

* materialize the four controlled recipes from one incumbent config; and
* apply the predeclared paired decision rule to exact-score rows.

The clock-fill arms accept internally validated, operator-attested
:class:`ThroughputProfile` objects.
There is no field-derived or hard-coded HKE seconds/step fallback.  A missing,
wrong-bundle, wrong-regime, or wrong-runtime profile is a hard prelaunch stop.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forge import adaptive_timing, krea_runtime, recipe  # noqa: E402


SCHEMA = 1
KIND = "sn56-week7-hke-factorial"
PLAN_STATUS = "TIMING_BOUND_DISCOVERY_PLAN_NO_LAUNCH_AUTHORITY"
MODEL_TYPE = "krea2"
INCUMBENT_BUNDLE = krea_runtime.INCUMBENT_BUNDLE
INCUMBENT_RUNTIME_COMMIT = krea_runtime.PINNED_BASE_COMMIT
INCUMBENT_BUNDLE_SHA256 = krea_runtime.bundle_contract_sha256(INCUMBENT_BUNDLE)
INCUMBENT_TEMPLATE_PATH = REPO_ROOT / "forge" / "templates" / "base_diffusion_krea2.yaml"
ADMISSION_AUTHORITY_PATH = (
    REPO_ROOT / "ops" / "experiments" / "week7" / "hke_fixture_admission.py"
)
# Literal trust anchor, reviewed at the exact source head.  Do not replace this
# with a value calculated at import time: that would let a changed template
# redefine the evidence expected to attest it.
INCUMBENT_TEMPLATE_FILE_SHA256 = (
    "031c4c8b2a6eda25b9db34eff1fdda20a05f04cf42a81b9e0c2515541cb1946b"
)
TRAINING_SEED_A = 42_565_431
PRIMARY_FAMILIES = ("product", "logo_ui")
RELIABILITY_FAMILY = "social"
ARMS: dict[str, dict[str, str]] = {
    "A": {"loss": "mae", "planner": "current_size_law"},
    "B": {"loss": "mse", "planner": "current_size_law"},
    "C": {"loss": "mae", "planner": "measured_clock_fill"},
    "D": {"loss": "mse", "planner": "measured_clock_fill"},
}
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260810
UNCERTAINTY_LEVEL = 0.95
MAX_RELATIVE_REGRESSION = 0.01
EXPECTED_FIXTURE_COUNTS = {
    "social": {"discovery": 10, "confirmation": 8},
    "product": {"discovery": 28, "confirmation": 10},
    "logo_ui": {"discovery": 32, "confirmation": 10},
}


class HKEContractError(RuntimeError):
    """A prelaunch or decision input is missing, malformed, or misbound."""


@dataclass(frozen=True)
class BoundTimingProfile:
    """One validated timing claim plus its experiment-specific binding.

    ``ThroughputProfile`` binds a runtime bundle and accelerator class, but it
    does not by itself say which factorial loss/config produced the timing.
    This outer record closes that gap and is itself content addressed.
    """

    profile: adaptive_timing.ThroughputProfile
    loss: str
    measured_dataset_size: int
    measured_config_sha256: str
    accelerator_identity: str
    binding_sha256: str


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            separators=(",", ": "),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise HKEContractError(f"{label} is not a sha256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise HKEContractError(f"{label} is not a sha256") from exc
    return value


def _require_h100_80gb_identity(value: Any, label: str) -> str:
    text = str(value or "").strip()
    name, separator, memory_field = text.rpartition("|")
    suffix = "-MiB"
    if (
        not separator
        or "h100" not in name.casefold()
        or not memory_field.endswith(suffix)
        or not memory_field[: -len(suffix)].isdigit()
    ):
        raise HKEContractError(f"{label} is not a measured H100 80GB identity")
    memory_mib = int(memory_field[: -len(suffix)])
    if not 78_000 <= memory_mib <= 85_000:
        raise HKEContractError(f"{label} is not a measured H100 80GB identity")
    return text


def _profile_document(
    profile: adaptive_timing.ThroughputProfile,
) -> dict[str, Any]:
    """Reconstruct the schema-3 document whose digest the profile declares."""

    return {
        "schema": adaptive_timing.PROFILE_SCHEMA,
        "kind": adaptive_timing.PROFILE_KIND,
        "bundle_id": profile.bundle_id,
        "bundle_sha256": profile.bundle_sha256,
        "model_type": profile.model_type,
        "measured_dataset_size": profile.measured_dataset_size,
        "dataset_regime": profile.dataset_regime,
        "seconds_per_step": profile.seconds_per_step,
        "startup_seconds": profile.startup_seconds,
        "measurement": {
            "completed_steps": profile.completed_steps,
            "training_elapsed_seconds": profile.training_elapsed_seconds,
            "first_checkpoint_step": profile.first_checkpoint_step,
            "first_checkpoint_elapsed_seconds": (
                profile.first_checkpoint_elapsed_seconds
            ),
        },
        "provenance": {
            "source_run_id": profile.source_run_id,
            "source_record_sha256": profile.source_record_sha256,
            "runtime_commit": profile.runtime_commit,
            "measured_at_utc": profile.measured_at_utc,
            "accelerator_identity": profile.accelerator_identity,
        },
        "profile_sha256": profile.profile_sha256,
    }


def _profile_binding_body(
    profile: adaptive_timing.ThroughputProfile,
    *,
    loss: str,
    measured_dataset_size: int,
    measured_config_sha256: str,
    accelerator_identity: str,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "sn56-week7-hke-timing-profile-binding",
        "loss": loss,
        "bundle_id": INCUMBENT_BUNDLE,
        "bundle_sha256": INCUMBENT_BUNDLE_SHA256,
        "runtime_commit": INCUMBENT_RUNTIME_COMMIT,
        "template_file_sha256": INCUMBENT_TEMPLATE_FILE_SHA256,
        "training_seed": TRAINING_SEED_A,
        "measured_dataset_size": measured_dataset_size,
        "dataset_regime": adaptive_timing.dataset_regime(measured_dataset_size),
        "measured_config_sha256": measured_config_sha256,
        "accelerator_identity": accelerator_identity,
        "profile_sha256": profile.profile_sha256,
        "source_record_sha256": profile.source_record_sha256,
    }


def bind_timing_profile(
    profile: adaptive_timing.ThroughputProfile,
    *,
    loss: str,
    measured_dataset_size: int,
    measured_config_sha256: str,
) -> BoundTimingProfile:
    """Create the content-addressed outer binding used by prelaunch.

    This helper is deterministic, not an authority shortcut.  Consumption
    independently revalidates the inner profile and recomputes this digest.
    """

    body = _profile_binding_body(
        profile,
        loss=loss,
        measured_dataset_size=measured_dataset_size,
        measured_config_sha256=_require_sha256(
            measured_config_sha256, "measured config sha256"
        ),
        accelerator_identity=profile.accelerator_identity,
    )
    return BoundTimingProfile(
        profile=profile,
        loss=loss,
        measured_dataset_size=measured_dataset_size,
        measured_config_sha256=measured_config_sha256,
        accelerator_identity=profile.accelerator_identity,
        binding_sha256=canonical_sha256(body),
    )


def _bound_profile_document(binding: BoundTimingProfile) -> dict[str, Any]:
    """Serialize every byte needed to repeat profile and binding validation."""

    if not isinstance(binding, BoundTimingProfile):
        raise HKEContractError("bound timing profile document is unavailable")
    binding_body = _profile_binding_body(
        binding.profile,
        loss=binding.loss,
        measured_dataset_size=binding.measured_dataset_size,
        measured_config_sha256=binding.measured_config_sha256,
        accelerator_identity=binding.accelerator_identity,
    )
    if binding.binding_sha256 != canonical_sha256(binding_body):
        raise HKEContractError("bound timing profile digest mismatch")
    return {
        "profile": _profile_document(binding.profile),
        "binding": {
            **binding_body,
            "binding_sha256": binding.binding_sha256,
        },
    }


def _validate_bound_profile_document(
    value: Mapping[str, Any],
    *,
    loss: str,
    expected_dataset_size: int,
    expected_config_sha256: str,
) -> BoundTimingProfile:
    """Recreate and validate a serialized timing profile and outer binding."""

    if not isinstance(value, Mapping) or set(value) != {"profile", "binding"}:
        raise HKEContractError(f"serialized {loss} timing profile is malformed")
    binding_value = value["binding"]
    if not isinstance(binding_value, Mapping):
        raise HKEContractError(f"serialized {loss} timing binding is malformed")
    binding_document = dict(binding_value)
    declared_binding = binding_document.pop("binding_sha256", None)
    _require_sha256(declared_binding, f"serialized {loss} timing binding")
    accelerator_identity = _require_h100_80gb_identity(
        binding_document.get("accelerator_identity"),
        f"serialized {loss} timing accelerator",
    )
    try:
        profile = adaptive_timing.validate_profile(
            value["profile"],
            expected_bundle_id=INCUMBENT_BUNDLE,
            expected_bundle_sha256=INCUMBENT_BUNDLE_SHA256,
            expected_model_type=MODEL_TYPE,
            current_dataset_size=expected_dataset_size,
            expected_dataset_regime=adaptive_timing.dataset_regime(
                expected_dataset_size
            ),
            expected_accelerator_identity=accelerator_identity,
        )
    except Exception as exc:
        raise HKEContractError(
            f"serialized {loss} timing profile failed validation"
        ) from exc
    expected_binding = _profile_binding_body(
        profile,
        loss=loss,
        measured_dataset_size=expected_dataset_size,
        measured_config_sha256=expected_config_sha256,
        accelerator_identity=accelerator_identity,
    )
    if (
        binding_document != expected_binding
        or declared_binding != canonical_sha256(expected_binding)
    ):
        raise HKEContractError(f"serialized {loss} timing binding mismatch")
    bound = BoundTimingProfile(
        profile=profile,
        loss=loss,
        measured_dataset_size=expected_dataset_size,
        measured_config_sha256=expected_config_sha256,
        accelerator_identity=accelerator_identity,
        binding_sha256=declared_binding,
    )
    _profile_for_loss(
        {loss: bound},
        loss,
        expected_dataset_size=expected_dataset_size,
        expected_dataset_regime=adaptive_timing.dataset_regime(
            expected_dataset_size
        ),
        expected_config_sha256=expected_config_sha256,
    )
    return bound


def _verify_incumbent_source(base_config: Mapping[str, Any]) -> dict[str, str]:
    """Bind planning to the literal reviewed template bytes and semantics."""

    if sha256_file(INCUMBENT_TEMPLATE_PATH) != INCUMBENT_TEMPLATE_FILE_SHA256:
        raise HKEContractError("incumbent template source hash drifted")
    try:
        parsed = yaml.safe_load(INCUMBENT_TEMPLATE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - guarded by repository tests
        raise HKEContractError("incumbent template source is unreadable") from exc
    if not isinstance(parsed, dict) or dict(base_config) != parsed:
        raise HKEContractError(
            "base config is not the exact reviewed incumbent template"
        )
    return {
        "path": str(INCUMBENT_TEMPLATE_PATH.relative_to(REPO_ROOT)),
        "file_sha256": INCUMBENT_TEMPLATE_FILE_SHA256,
        "semantic_sha256": canonical_sha256(parsed),
    }


def _process_node(config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = config["config"]["process"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise HKEContractError("Krea config has no process node") from exc
    if not isinstance(value, dict):
        raise HKEContractError("Krea process node is not an object")
    return value


def _train_node(config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = config["config"]["process"][0]["train"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HKEContractError("Krea config has no train node") from exc
    if not isinstance(value, dict):
        raise HKEContractError("Krea train node is not an object")
    return value


def _save_node(config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = config["config"]["process"][0]["save"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HKEContractError("Krea config has no save node") from exc
    if not isinstance(value, dict):
        raise HKEContractError("Krea save node is not an object")
    return value


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in sorted(value):
            result.update(_flatten(value[key], f"{prefix}/{key}"))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten(item, f"{prefix}/{index}"))
        return result
    return {prefix or "/": value}


def changed_pointers(left: Mapping[str, Any], right: Mapping[str, Any]) -> set[str]:
    lflat = _flatten(left)
    rflat = _flatten(right)
    return {
        key
        for key in set(lflat) | set(rflat)
        if lflat.get(key, object()) != rflat.get(key, object())
    }


def _profile_for_loss(
    profiles: Mapping[str, BoundTimingProfile],
    loss: str,
    *,
    expected_dataset_size: int,
    expected_dataset_regime: str,
    expected_config_sha256: str,
) -> adaptive_timing.ThroughputProfile:
    binding = profiles.get(loss)
    if not isinstance(binding, BoundTimingProfile):
        raise HKEContractError(f"bound operator-attested {loss} profile is required")
    profile = binding.profile
    if binding.loss != loss:
        raise HKEContractError(f"{loss} profile loss binding mismatch")
    if binding.measured_dataset_size != expected_dataset_size:
        raise HKEContractError(f"{loss} profile dataset-size binding mismatch")
    if binding.measured_config_sha256 != expected_config_sha256:
        raise HKEContractError(f"{loss} profile config binding mismatch")
    if binding.accelerator_identity != profile.accelerator_identity:
        raise HKEContractError(f"{loss} profile accelerator binding mismatch")
    _require_h100_80gb_identity(
        binding.accelerator_identity, f"{loss} profile accelerator"
    )
    if profile.model_type != MODEL_TYPE:
        raise HKEContractError(f"{loss} profile is not Krea2")
    if profile.bundle_id != INCUMBENT_BUNDLE:
        raise HKEContractError(f"{loss} profile bundle id mismatch")
    if profile.bundle_sha256 != INCUMBENT_BUNDLE_SHA256:
        raise HKEContractError(f"{loss} profile bundle digest mismatch")
    if profile.runtime_commit != INCUMBENT_RUNTIME_COMMIT:
        raise HKEContractError(f"{loss} profile is not bound to incumbent runtime")
    if profile.measured_dataset_size != expected_dataset_size:
        raise HKEContractError(f"{loss} profile measured dataset size mismatch")
    if profile.dataset_regime != expected_dataset_regime:
        raise HKEContractError(f"{loss} profile dataset regime mismatch")
    try:
        validated = adaptive_timing.validate_profile(
            _profile_document(profile),
            expected_bundle_id=INCUMBENT_BUNDLE,
            expected_bundle_sha256=INCUMBENT_BUNDLE_SHA256,
            expected_model_type=MODEL_TYPE,
            current_dataset_size=expected_dataset_size,
            expected_dataset_regime=expected_dataset_regime,
            expected_accelerator_identity=binding.accelerator_identity,
        )
    except Exception as exc:
        raise HKEContractError(
            f"{loss} profile failed internal digest/consistency validation"
        ) from exc
    expected_binding = _profile_binding_body(
        validated,
        loss=loss,
        measured_dataset_size=expected_dataset_size,
        measured_config_sha256=expected_config_sha256,
        accelerator_identity=binding.accelerator_identity,
    )
    if binding.binding_sha256 != canonical_sha256(expected_binding):
        raise HKEContractError(f"{loss} profile outer binding digest mismatch")
    return validated


def measured_clock_fill_steps(
    *,
    hours_to_complete: float,
    profile: adaptive_timing.ThroughputProfile,
) -> int:
    """Fill the reviewed window from operator-attested profile throughput.

    The dataset-size target is intentionally absent.  The only ceiling is the
    reviewed Krea policy maximum; safety margin, startup and export reserve are
    retained.  The physical forced-stop gate remains separate and is exercised
    by the later H100 mechanical run.
    """

    if not isinstance(profile, adaptive_timing.ThroughputProfile):
        raise HKEContractError("clock fill requires a validated timing profile")
    if profile.model_type != MODEL_TYPE:
        raise HKEContractError("clock profile model type mismatch")
    if profile.runtime_commit != INCUMBENT_RUNTIME_COMMIT:
        raise HKEContractError("clock profile runtime commit mismatch")
    return _clock_fill_steps(
        hours_to_complete=hours_to_complete,
        seconds_per_step=profile.seconds_per_step,
        startup_seconds=profile.startup_seconds,
    )


def _clock_fill_steps(
    *,
    hours_to_complete: float,
    seconds_per_step: float,
    startup_seconds: float,
) -> int:
    """Materialize one clock plan from already validated conservative inputs."""

    budget = float(hours_to_complete) * 3600.0
    if not math.isfinite(budget) or budget <= 0:
        raise HKEContractError("hours_to_complete must be positive and finite")
    if not math.isfinite(seconds_per_step) or seconds_per_step <= 0:
        raise HKEContractError("seconds_per_step must be positive and finite")
    if not math.isfinite(startup_seconds) or startup_seconds < 0:
        raise HKEContractError("startup_seconds must be non-negative and finite")
    available = (
        budget * recipe.margin_for(MODEL_TYPE)
        - startup_seconds
        - recipe.EXPORT_RESERVE_S
    )
    if available <= 0:
        raise HKEContractError("timing profile leaves no training window")
    steps = int(available / seconds_per_step)
    return max(1, min(int(recipe.STEP_TABLE[MODEL_TYPE]["max"]), steps))


def materialize_current_law_configs(
    base_config: Mapping[str, Any],
    *,
    num_images: int,
    hours_to_complete: float,
) -> dict[str, dict[str, Any]]:
    """Materialize the exact Seed-A timing-source configs for A and B."""

    try:
        count = int(num_images)
    except Exception as exc:
        raise HKEContractError("num_images must be an integer") from exc
    if count <= 0 or isinstance(num_images, bool):
        raise HKEContractError("num_images must be positive")
    original = copy.deepcopy(dict(base_config))
    _process_node(original)["training_seed"] = TRAINING_SEED_A
    template_steps = int(_train_node(original)["steps"])
    template_cadence = int(_save_node(original)["save_every"])
    current_steps = recipe.size_scaled_steps(
        MODEL_TYPE, count, hours_to_complete, template_steps
    )
    result: dict[str, dict[str, Any]] = {}
    for arm_id in ("A", "B"):
        config = copy.deepcopy(original)
        _train_node(config)["loss_type"] = ARMS[arm_id]["loss"]
        _train_node(config)["steps"] = current_steps
        _save_node(config)["save_every"] = recipe.kill_safe_save_every(
            current_steps, template_cadence
        )
        result[arm_id] = config
    return result


def materialize_arms(
    base_config: Mapping[str, Any],
    *,
    num_images: int,
    hours_to_complete: float,
    profiles: Mapping[str, BoundTimingProfile],
) -> dict[str, dict[str, Any]]:
    """Return A-D configs and prove the factorial changed only named fields."""

    try:
        count = int(num_images)
    except Exception as exc:
        raise HKEContractError("num_images must be an integer") from exc
    if count <= 0:
        raise HKEContractError("num_images must be positive")

    original = copy.deepcopy(dict(base_config))
    _process_node(original)["training_seed"] = TRAINING_SEED_A
    template_cadence = int(_save_node(original)["save_every"])
    expected_regime = adaptive_timing.dataset_regime(count)
    # First materialize the two current-law cells.  Their exact bytes are the
    # timing source configs to which the MAE/MSE profiles must bind.
    result = materialize_current_law_configs(
        base_config,
        num_images=count,
        hours_to_complete=hours_to_complete,
    )

    validated_profiles = {
        loss: _profile_for_loss(
            profiles,
            loss,
            expected_dataset_size=count,
            expected_dataset_regime=expected_regime,
            expected_config_sha256=canonical_sha256(
                result["A" if loss == "mae" else "B"]
            ),
        )
        for loss in ("mae", "mse")
    }
    if (
        validated_profiles["mae"].accelerator_identity
        != validated_profiles["mse"].accelerator_identity
    ):
        raise HKEContractError(
            "MAE/MSE timing profiles do not declare the same accelerator"
        )
    # Preserve a true 2x2 factorial: both clock-fill arms use one conservative
    # plan supported by both loss-bound observations.  The slower rate and
    # larger startup are selected independently; this cannot overstate either
    # profile's declared capacity.
    shared_steps = _clock_fill_steps(
        hours_to_complete=hours_to_complete,
        seconds_per_step=max(
            profile.seconds_per_step for profile in validated_profiles.values()
        ),
        startup_seconds=max(
            profile.startup_seconds for profile in validated_profiles.values()
        ),
    )
    for arm_id, source_id in (("C", "A"), ("D", "B")):
        config = copy.deepcopy(result[source_id])
        _train_node(config)["steps"] = shared_steps
        _save_node(config)["save_every"] = recipe.kill_safe_save_every(
            shared_steps, template_cadence
        )
        result[arm_id] = config

    allowed = {
        "/config/process/0/train/loss_type",
        "/config/process/0/train/steps",
        "/config/process/0/save/save_every",
    }
    for arm_id, config in result.items():
        unexpected = changed_pointers(original, config) - allowed
        if unexpected:
            raise HKEContractError(
                f"arm {arm_id} changed fields outside the factorial: {sorted(unexpected)}"
            )
    if changed_pointers(result["A"], result["B"]) != {
        "/config/process/0/train/loss_type"
    }:
        raise HKEContractError("A/B differ by more than loss")
    if changed_pointers(result["C"], result["D"]) != {
        "/config/process/0/train/loss_type"
    }:
        raise HKEContractError("C/D differ by more than loss")
    for left, right in (("A", "C"), ("B", "D")):
        delta = changed_pointers(result[left], result[right])
        if not delta or not delta <= {
            "/config/process/0/train/steps",
            "/config/process/0/save/save_every",
        }:
            raise HKEContractError(f"{left}/{right} planner contrast is invalid")
    return result


def _validate_admission(value: Mapping[str, Any], family: str) -> dict[str, Any]:
    """Validate one receipt embedded in the content-addressed root admission.

    The admission tool is intentionally a separate authority.  This consumer
    carries only the minimum cross-boundary contract and never turns an agent's
    fixture inspection into fictional human approval.
    """

    expected = EXPECTED_FIXTURE_COUNTS[family]
    required = {
        "schema",
        "kind",
        "status",
        "family",
        "counts",
        "candidate_semantic_sha256",
        "human_review_sha256",
        "replay_evidence_sha256",
        "dedup_evidence_sha256",
        "ownership_record_sha256",
        "generator_commit",
        "generator_tree",
        "generator_source_sha256",
        "admission_authority_source_sha256",
        "discovery_row_identity_sha256",
        "confirmation_commitment_sha256",
        "governance",
        "claim_limit",
        "admission_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise HKEContractError(f"{family} fixture admission is unavailable")
    if (
        value.get("schema") != 1
        or value.get("kind") != "sn56-week7-hke-fixture-admission"
    ):
        raise HKEContractError(f"{family} fixture admission contract is unsupported")
    if value.get("status") != "PASS" or value.get("family") != family:
        raise HKEContractError(f"{family} fixture is not admitted")
    counts = value.get("counts")
    if counts != expected:
        raise HKEContractError(f"{family} fixture counts do not match the frozen contract")
    governance = value.get("governance")
    if not isinstance(governance, Mapping):
        raise HKEContractError(f"{family} fixture governance is unavailable")
    if (
        set(governance)
        != {
            "operator_attested_named_human_review",
            "agent_review_is_not_human_review",
            "admission_authorized",
            "gpu_execution_authorized",
            "owner_ratification_required_for_gpu",
        }
        or governance.get("operator_attested_named_human_review") is not True
        or governance.get("agent_review_is_not_human_review") is not True
        or governance.get("admission_authorized") is not True
        or governance.get("gpu_execution_authorized") is not False
        or governance.get("owner_ratification_required_for_gpu") is not True
    ):
        raise HKEContractError(f"{family} fixture governance is not admissible")
    for key in (
        "candidate_semantic_sha256",
        "human_review_sha256",
        "replay_evidence_sha256",
        "dedup_evidence_sha256",
        "ownership_record_sha256",
        "confirmation_commitment_sha256",
        "generator_source_sha256",
        "admission_authority_source_sha256",
        "discovery_row_identity_sha256",
        "admission_sha256",
    ):
        item = value.get(key)
        if not isinstance(item, str) or len(item) != 64:
            raise HKEContractError(f"{family} fixture admission lacks {key}")
        try:
            int(item, 16)
        except ValueError as exc:
            raise HKEContractError(f"{family} fixture admission has invalid {key}") from exc
    for key in ("generator_commit", "generator_tree"):
        item = value.get(key)
        if not isinstance(item, str) or len(item) != 40:
            raise HKEContractError(f"{family} fixture admission lacks {key}")
        try:
            int(item, 16)
        except ValueError as exc:
            raise HKEContractError(f"{family} fixture admission has invalid {key}") from exc
    checked = dict(value)
    declared = checked.pop("admission_sha256")
    if declared != canonical_sha256(checked):
        raise HKEContractError(f"{family} fixture admission digest mismatch")
    if (
        value["admission_authority_source_sha256"]
        != sha256_file(ADMISSION_AUTHORITY_PATH)
    ):
        raise HKEContractError(
            f"{family} fixture admission authority source mismatch"
        )
    return dict(value)


def _validate_admission_set(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the sole fixture authority and derive all family receipts.

    Free-standing receipts are deliberately not accepted.  The admission set
    embeds them before hashing, so one root digest binds the exact family
    receipts consumed by this planner without a circular receipt-to-root link.
    """

    required = {
        "schema",
        "kind",
        "status",
        "candidate_semantic_sha256",
        "human_review_sha256",
        "replay_evidence",
        "replay_evidence_sha256",
        "dedup_evidence_sha256",
        "ownership_record_sha256",
        "generator_revision",
        "family_admission_sha256",
        "receipts",
        "authorization",
        "admission_set_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise HKEContractError("fixture admission-set envelope is malformed")
    checked_root = dict(value)
    declared_root_sha = checked_root.pop("admission_set_sha256")
    _require_sha256(declared_root_sha, "fixture admission-set")
    if declared_root_sha != canonical_sha256(checked_root):
        raise HKEContractError("fixture admission-set digest mismatch")
    if (
        value["schema"] != 1
        or value["kind"] != "sn56-week7-hke-fixture-admission-set"
        or value["status"] != "PASS"
    ):
        raise HKEContractError("fixture admission-set has not passed")
    for key in (
        "candidate_semantic_sha256",
        "human_review_sha256",
        "replay_evidence_sha256",
        "dedup_evidence_sha256",
        "ownership_record_sha256",
    ):
        _require_sha256(value[key], f"fixture admission-set {key}")
    replay = value["replay_evidence"]
    expected_replay_fields = {
        "status",
        "verified_rows",
        "candidate_semantic_sha256",
        "dedup_semantic_sha256",
    }
    if (
        not isinstance(replay, Mapping)
        or set(replay) != expected_replay_fields
        or replay["status"] != "PASS"
        or replay["verified_rows"]
        != sum(sum(counts.values()) for counts in EXPECTED_FIXTURE_COUNTS.values())
        or replay["candidate_semantic_sha256"]
        != value["candidate_semantic_sha256"]
        or replay["dedup_semantic_sha256"] != value["dedup_evidence_sha256"]
        or value["replay_evidence_sha256"] != canonical_sha256(replay)
    ):
        raise HKEContractError("fixture admission-set replay digest mismatch")
    authorization = value["authorization"]
    if authorization != {
        "fixture_admission_authorized": True,
        "gpu_execution_authorized": False,
        "deployment_authorized": False,
    }:
        raise HKEContractError("fixture admission-set authorization is invalid")
    revision = value["generator_revision"]
    revision_fields = {
        "repository",
        "commit",
        "tree",
        "renderer_source_sha256",
        "admission_authority_source_sha256",
        "pinned_remote_refs",
    }
    if not isinstance(revision, Mapping) or set(revision) != revision_fields:
        raise HKEContractError("fixture admission-set generator revision is malformed")
    for key in ("renderer_source_sha256", "admission_authority_source_sha256"):
        _require_sha256(revision[key], f"fixture admission-set generator {key}")
    for key in ("commit", "tree"):
        item = revision[key]
        if not isinstance(item, str) or len(item) != 40:
            raise HKEContractError(f"fixture admission-set generator {key} is invalid")
        try:
            int(item, 16)
        except ValueError as exc:
            raise HKEContractError(
                f"fixture admission-set generator {key} is invalid"
            ) from exc
    if (
        revision["admission_authority_source_sha256"]
        != sha256_file(ADMISSION_AUTHORITY_PATH)
    ):
        raise HKEContractError("fixture admission-set authority source mismatch")
    if (
        not isinstance(revision["repository"], str)
        or not revision["repository"].strip()
        or not isinstance(revision["pinned_remote_refs"], list)
        or not revision["pinned_remote_refs"]
        or any(
            not isinstance(item, str) or not item.strip()
            for item in revision["pinned_remote_refs"]
        )
    ):
        raise HKEContractError("fixture admission-set pinned revision is absent")

    receipts = value["receipts"]
    receipt_hashes = value["family_admission_sha256"]
    expected_families = set(EXPECTED_FIXTURE_COUNTS)
    if (
        not isinstance(receipts, Mapping)
        or set(receipts) != expected_families
        or not isinstance(receipt_hashes, Mapping)
        or set(receipt_hashes) != expected_families
    ):
        raise HKEContractError("fixture admission-set family inventory mismatch")
    validated: dict[str, dict[str, Any]] = {}
    for family in EXPECTED_FIXTURE_COUNTS:
        receipt = _validate_admission(receipts[family], family)
        if receipt_hashes[family] != receipt["admission_sha256"]:
            raise HKEContractError(f"{family} receipt is not bound by admission-set")
        for receipt_key, root_key in (
            ("candidate_semantic_sha256", "candidate_semantic_sha256"),
            ("human_review_sha256", "human_review_sha256"),
            ("replay_evidence_sha256", "replay_evidence_sha256"),
            ("dedup_evidence_sha256", "dedup_evidence_sha256"),
            ("ownership_record_sha256", "ownership_record_sha256"),
        ):
            if receipt[receipt_key] != value[root_key]:
                raise HKEContractError(
                    f"{family} receipt {receipt_key} disagrees with admission-set"
                )
        if (
            receipt["generator_commit"] != revision["commit"]
            or receipt["generator_tree"] != revision["tree"]
            or receipt["generator_source_sha256"]
            != revision["renderer_source_sha256"]
            or receipt["admission_authority_source_sha256"]
            != revision["admission_authority_source_sha256"]
        ):
            raise HKEContractError(
                f"{family} receipt generator disagrees with admission-set"
            )
        validated[family] = receipt
    return {
        "admission_set_sha256": declared_root_sha,
        "receipts": validated,
    }


def _validate_evaluator_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "harness_sha256",
        "god_commit",
        "comfy_commit",
        "defaults_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise HKEContractError("exact evaluator identity is required")
    checked = dict(value)
    for key in ("harness_sha256", "defaults_sha256"):
        _require_sha256(checked[key], f"evaluator {key}")
    for key in ("god_commit", "comfy_commit"):
        item = checked[key]
        if not isinstance(item, str) or len(item) != 40:
            raise HKEContractError(f"evaluator {key} is not a commit")
        try:
            int(item, 16)
        except ValueError as exc:
            raise HKEContractError(f"evaluator {key} is not a commit") from exc
    return checked


def build_prelaunch_plan(
    base_config: Mapping[str, Any],
    *,
    admission_set: Mapping[str, Any],
    profiles_by_family: Mapping[str, Mapping[str, BoundTimingProfile]],
    evaluator_identity: Mapping[str, Any],
    hours_to_complete: float = 0.75,
) -> dict[str, Any]:
    """Build a hash-bound plan only after fixtures and timing evidence exist."""

    if set(profiles_by_family) != set(EXPECTED_FIXTURE_COUNTS):
        raise HKEContractError(
            "prelaunch requires bound operator-attested profiles for every family"
        )
    source_identity = _verify_incumbent_source(base_config)
    evaluator = _validate_evaluator_identity(evaluator_identity)
    evaluator_sha256 = canonical_sha256(evaluator)
    checked_set = _validate_admission_set(admission_set)
    checked = checked_set["receipts"]
    configs: dict[str, Any] = {}
    timing_profiles: dict[str, Any] = {}
    for family, counts in EXPECTED_FIXTURE_COUNTS.items():
        family_profiles = profiles_by_family[family]
        if not isinstance(family_profiles, Mapping) or set(family_profiles) != {
            "mae",
            "mse",
        }:
            raise HKEContractError(
                f"prelaunch requires exact MAE/MSE timing profiles for {family}"
            )
        family_configs = materialize_arms(
            base_config,
            num_images=counts["discovery"],
            hours_to_complete=hours_to_complete,
            profiles=family_profiles,
        )
        selected_arms = ("A", "D") if family == RELIABILITY_FAMILY else tuple(ARMS)
        timing_profiles[family] = {
            loss: _bound_profile_document(family_profiles[loss])
            for loss in ("mae", "mse")
        }
        clock_support = {
            "profile_sha256": {
                loss: family_profiles[loss].profile.profile_sha256
                for loss in ("mae", "mse")
            },
            "binding_sha256": {
                loss: family_profiles[loss].binding_sha256
                for loss in ("mae", "mse")
            },
            "conservative_seconds_per_step": max(
                family_profiles[loss].profile.seconds_per_step
                for loss in ("mae", "mse")
            ),
            "conservative_startup_seconds": max(
                family_profiles[loss].profile.startup_seconds
                for loss in ("mae", "mse")
            ),
        }
        configs[family] = {
            arm: {
                "config": family_configs[arm],
                "config_sha256": canonical_sha256(family_configs[arm]),
                "training_seed": TRAINING_SEED_A,
                "planned_steps": _train_node(family_configs[arm])["steps"],
                "timing_support": (
                    None
                    if ARMS[arm]["planner"] == "current_size_law"
                    else copy.deepcopy(clock_support)
                ),
                "decision_checkpoint": "terminal",
            }
            for arm in selected_arms
        }
    accelerator_identities = {
        profile.accelerator_identity
        for family_profiles in profiles_by_family.values()
        for profile in family_profiles.values()
    }
    if len(accelerator_identities) != 1:
        raise HKEContractError(
            "all timing profiles must come from one H100 80GB accelerator identity"
        )

    body = {
        "schema": SCHEMA,
        "kind": "sn56-week7-hke-prelaunch-plan",
        "status": PLAN_STATUS,
        "contract_sha256": experiment_contract()["contract_sha256"],
        "source": source_identity,
        "training_seed": {
            "role": "Seed-A",
            "value": TRAINING_SEED_A,
        },
        "evaluator": evaluator,
        "evaluator_sha256": evaluator_sha256,
        "hours_to_complete": float(hours_to_complete),
        "admission_set": copy.deepcopy(dict(admission_set)),
        "admission_set_sha256": checked_set["admission_set_sha256"],
        "fixtures": {
            family: {
                "admission_sha256": checked[family]["admission_sha256"],
                "candidate_semantic_sha256": checked[family][
                    "candidate_semantic_sha256"
                ],
                "discovery_row_identity_sha256": checked[family][
                    "discovery_row_identity_sha256"
                ],
                "confirmation_commitment_sha256": checked[family][
                    "confirmation_commitment_sha256"
                ],
                "counts": EXPECTED_FIXTURE_COUNTS[family],
                "generator_commit": checked[family]["generator_commit"],
                "generator_tree": checked[family]["generator_tree"],
            }
            for family in EXPECTED_FIXTURE_COUNTS
        },
        "timing_profiles": timing_profiles,
        "cells": configs,
        "authorization": {
            "cpu_plan_complete": True,
            "gpu_execution_authorized": False,
            "deployment_authorized": False,
        },
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def experiment_contract() -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": KIND,
        "status": "PRELAUNCH_CPU_ONLY",
        "model_type": MODEL_TYPE,
        "incumbent_source": {
            "path": str(INCUMBENT_TEMPLATE_PATH.relative_to(REPO_ROOT)),
            "file_sha256": INCUMBENT_TEMPLATE_FILE_SHA256,
        },
        "training_seed": {"role": "Seed-A", "value": TRAINING_SEED_A},
        "runtime": {
            "bundle": INCUMBENT_BUNDLE,
            "commit": INCUMBENT_RUNTIME_COMMIT,
            "owned_runtime_enabled": False,
        },
        "arms": copy.deepcopy(ARMS),
        "cells": {
            "product": list(ARMS),
            "logo_ui": list(ARMS),
            "social": ["A", "D"],
        },
        "checkpoint_policy": {
            "analysis_phase": "discovery_only_confirmation_sealed",
            "score_all_predeclared_periodic_and_terminal_offline": True,
            "decision_checkpoint": "terminal_only",
            "consume_live_selection_record": False,
            "live_promotion_enabled": False,
        },
        "routing": {
            "semantic_router_enabled": False,
            "production_mutation_authorized": False,
        },
        "clock_fill": {
            "source": (
                "operator_attested_internally_validated_profile_bound_to_"
                "exact_bundle_runtime_dataset_config_and_accelerator"
            ),
            "proof_of_measurement": False,
            "field_or_hke_constant_allowed": False,
            "formula": (
                "floor((hours*3600*margin_for(krea2)-profile.startup_seconds-"
                "EXPORT_RESERVE_S)/profile.seconds_per_step), capped only by "
                "STEP_TABLE[krea2].max"
            ),
        },
        "decision": {
            "metric": "0.25*prompted_loss + 0.75*blank_loss",
            "lower_is_better": True,
            "paired_bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "paired_bootstrap_seed": BOOTSTRAP_SEED,
            "interval": UNCERTAINTY_LEVEL,
            "uncertainty_scope": (
                "row-resampling uncertainty conditional on one fixed training "
                "run and one evaluator row set; excludes seed, rerun, and "
                "hardware variance"
            ),
            "factor_go": (
                "same improving direction on product and logo_ui; paired 95% "
                "CI lower bound >0 on at least one; no relative regression "
                ">=1% on either"
            ),
            "social": "A-vs-D operational reliability only; not a transfer GO",
        },
        "authorization": {
            "fixture_admission_required": True,
            "operator_attested_profiles_required": True,
            "gpu_execution_authorized": False,
            "deployment_authorized": False,
        },
    }
    return {**body, "contract_sha256": canonical_sha256(body)}


def _composite(row: Mapping[str, Any]) -> float:
    try:
        prompted = float(row["prompted_loss"])
        blank = float(row["blank_loss"])
    except Exception as exc:
        raise HKEContractError("score row lacks finite prompted/blank losses") from exc
    if not math.isfinite(prompted) or not math.isfinite(blank):
        raise HKEContractError("score row contains a non-finite loss")
    return 0.25 * prompted + 0.75 * blank


def _validate_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HKEContractError("analysis requires a prelaunch plan")
    plan = dict(value)
    declared = plan.pop("plan_sha256", None)
    required = {
        "schema",
        "kind",
        "status",
        "contract_sha256",
        "source",
        "training_seed",
        "evaluator",
        "evaluator_sha256",
        "hours_to_complete",
        "admission_set",
        "admission_set_sha256",
        "fixtures",
        "timing_profiles",
        "cells",
        "authorization",
    }
    if set(plan) != required:
        raise HKEContractError("prelaunch plan envelope is malformed")
    if declared != canonical_sha256(plan):
        raise HKEContractError("prelaunch plan digest mismatch")
    if (
        plan.get("schema") != SCHEMA
        or plan.get("kind") != "sn56-week7-hke-prelaunch-plan"
        or plan.get("status") != PLAN_STATUS
        or plan.get("contract_sha256") != experiment_contract()["contract_sha256"]
    ):
        raise HKEContractError("prelaunch plan contract mismatch")
    validated_admission_set = _validate_admission_set(plan["admission_set"])
    if (
        plan["admission_set_sha256"]
        != validated_admission_set["admission_set_sha256"]
    ):
        raise HKEContractError("prelaunch admission-set identity mismatch")
    if (
        not isinstance(plan["hours_to_complete"], (int, float))
        or isinstance(plan["hours_to_complete"], bool)
        or not math.isfinite(float(plan["hours_to_complete"]))
        or float(plan["hours_to_complete"]) <= 0
    ):
        raise HKEContractError("prelaunch hours are invalid")
    if plan["authorization"] != {
        "cpu_plan_complete": True,
        "gpu_execution_authorized": False,
        "deployment_authorized": False,
    }:
        raise HKEContractError("prelaunch authorization is invalid")
    source_config = yaml.safe_load(
        INCUMBENT_TEMPLATE_PATH.read_text(encoding="utf-8")
    )
    if plan.get("source") != _verify_incumbent_source(source_config):
        raise HKEContractError("prelaunch plan incumbent source mismatch")
    if plan.get("training_seed") != {
        "role": "Seed-A",
        "value": TRAINING_SEED_A,
    }:
        raise HKEContractError("prelaunch plan seed mismatch")
    evaluator = _validate_evaluator_identity(plan.get("evaluator", {}))
    if plan.get("evaluator_sha256") != canonical_sha256(evaluator):
        raise HKEContractError("prelaunch plan evaluator digest mismatch")
    if set(plan.get("fixtures", {})) != set(EXPECTED_FIXTURE_COUNTS):
        raise HKEContractError("prelaunch plan fixture set mismatch")
    if set(plan.get("timing_profiles", {})) != set(EXPECTED_FIXTURE_COUNTS):
        raise HKEContractError("prelaunch plan timing-profile set mismatch")
    if set(plan.get("cells", {})) != set(EXPECTED_FIXTURE_COUNTS):
        raise HKEContractError("prelaunch plan cell set mismatch")
    plan_accelerator_identities: set[str] = set()
    for family in EXPECTED_FIXTURE_COUNTS:
        expected_current = materialize_current_law_configs(
            source_config,
            num_images=EXPECTED_FIXTURE_COUNTS[family]["discovery"],
            hours_to_complete=float(plan["hours_to_complete"]),
        )
        fixture = plan["fixtures"][family]
        fixture_fields = {
            "admission_sha256",
            "candidate_semantic_sha256",
            "discovery_row_identity_sha256",
            "confirmation_commitment_sha256",
            "counts",
            "generator_commit",
            "generator_tree",
        }
        if not isinstance(fixture, Mapping) or set(fixture) != fixture_fields:
            raise HKEContractError(f"prelaunch plan {family} fixture is malformed")
        for key in (
            "admission_sha256",
            "candidate_semantic_sha256",
            "discovery_row_identity_sha256",
            "confirmation_commitment_sha256",
        ):
            _require_sha256(fixture[key], f"prelaunch {family} fixture {key}")
        if fixture["counts"] != EXPECTED_FIXTURE_COUNTS[family]:
            raise HKEContractError(f"prelaunch plan {family} fixture counts mismatch")
        for key in ("generator_commit", "generator_tree"):
            item = fixture[key]
            if not isinstance(item, str) or len(item) != 40:
                raise HKEContractError(
                    f"prelaunch plan {family} fixture {key} is invalid"
                )
            try:
                int(item, 16)
            except ValueError as exc:
                raise HKEContractError(
                    f"prelaunch plan {family} fixture {key} is invalid"
                ) from exc
        receipt = validated_admission_set["receipts"][family]
        expected_fixture = {
            "admission_sha256": receipt["admission_sha256"],
            "candidate_semantic_sha256": receipt[
                "candidate_semantic_sha256"
            ],
            "discovery_row_identity_sha256": receipt[
                "discovery_row_identity_sha256"
            ],
            "confirmation_commitment_sha256": receipt[
                "confirmation_commitment_sha256"
            ],
            "counts": EXPECTED_FIXTURE_COUNTS[family],
            "generator_commit": receipt["generator_commit"],
            "generator_tree": receipt["generator_tree"],
        }
        if fixture != expected_fixture:
            raise HKEContractError(
                f"prelaunch plan {family} fixture does not reproduce admission"
            )
        timing = plan["timing_profiles"][family]
        if not isinstance(timing, Mapping) or set(timing) != {"mae", "mse"}:
            raise HKEContractError(
                f"prelaunch plan {family} timing profiles are malformed"
            )
        bound_profiles: dict[str, BoundTimingProfile] = {}
        for loss, source_arm in (("mae", "A"), ("mse", "B")):
            bound = _validate_bound_profile_document(
                timing[loss],
                loss=loss,
                expected_dataset_size=EXPECTED_FIXTURE_COUNTS[family][
                    "discovery"
                ],
                expected_config_sha256=canonical_sha256(
                    expected_current[source_arm]
                ),
            )
            bound_profiles[loss] = bound
            plan_accelerator_identities.add(bound.accelerator_identity)
        expected_materialized = materialize_arms(
            source_config,
            num_images=EXPECTED_FIXTURE_COUNTS[family]["discovery"],
            hours_to_complete=float(plan["hours_to_complete"]),
            profiles=bound_profiles,
        )
        expected_arms = {"A", "D"} if family == RELIABILITY_FAMILY else set(ARMS)
        cells = plan["cells"][family]
        if not isinstance(cells, Mapping) or set(cells) != expected_arms:
            raise HKEContractError(f"prelaunch plan {family} arm set mismatch")
        for arm, cell in cells.items():
            cell_fields = {
                "config",
                "config_sha256",
                "training_seed",
                "planned_steps",
                "timing_support",
                "decision_checkpoint",
            }
            if not isinstance(cell, Mapping) or set(cell) != cell_fields:
                raise HKEContractError(f"prelaunch plan {family}/{arm} is invalid")
            if cell.get("training_seed") != TRAINING_SEED_A:
                raise HKEContractError(f"prelaunch plan {family}/{arm} seed mismatch")
            config = cell.get("config")
            if not isinstance(config, Mapping) or cell.get(
                "config_sha256"
            ) != canonical_sha256(config):
                raise HKEContractError(f"prelaunch plan {family}/{arm} config mismatch")
            if cell.get("planned_steps") != _train_node(config).get("steps"):
                raise HKEContractError(f"prelaunch plan {family}/{arm} depth mismatch")
            if (
                _process_node(config).get("training_seed") != TRAINING_SEED_A
                or _train_node(config).get("loss_type") != ARMS[arm]["loss"]
                or cell.get("decision_checkpoint") != "terminal"
            ):
                raise HKEContractError(
                    f"prelaunch plan {family}/{arm} factorial identity mismatch"
                )
            if config != expected_materialized[arm]:
                raise HKEContractError(
                    f"prelaunch plan {family}/{arm} config does not reproduce"
                )
            expected_timing_support = (
                None
                if ARMS[arm]["planner"] == "current_size_law"
                else {
                    "profile_sha256": {
                        loss: bound_profiles[loss].profile.profile_sha256
                        for loss in ("mae", "mse")
                    },
                    "binding_sha256": {
                        loss: bound_profiles[loss].binding_sha256
                        for loss in ("mae", "mse")
                    },
                    "conservative_seconds_per_step": max(
                        bound_profiles[loss].profile.seconds_per_step
                        for loss in ("mae", "mse")
                    ),
                    "conservative_startup_seconds": max(
                        bound_profiles[loss].profile.startup_seconds
                        for loss in ("mae", "mse")
                    ),
                }
            )
            if cell.get("timing_support") != expected_timing_support:
                raise HKEContractError(
                    f"prelaunch plan {family}/{arm} timing profile mismatch"
                )
        if (
            bound_profiles["mae"].accelerator_identity
            != bound_profiles["mse"].accelerator_identity
        ):
            raise HKEContractError(
                f"prelaunch plan {family} timing accelerators disagree"
            )
        if family in PRIMARY_FAMILIES and (
            _train_node(cells["C"]["config"])["steps"]
            != _train_node(cells["D"]["config"])["steps"]
            or _save_node(cells["C"]["config"])["save_every"]
            != _save_node(cells["D"]["config"])["save_every"]
        ):
            raise HKEContractError(
                f"prelaunch plan {family} C/D clock plans disagree"
            )
    if len(plan_accelerator_identities) != 1:
        raise HKEContractError(
            "prelaunch timing profiles do not share one H100 80GB identity"
        )
    return {**plan, "plan_sha256": declared}


def _rows_by_id(
    value: Mapping[str, Any],
    label: str,
    *,
    plan_cell: Mapping[str, Any],
    plan_sha256: str,
    fixture: Mapping[str, Any],
    evaluator_sha256: str,
    expected_row_count: int,
) -> tuple[dict[str, float], str]:
    """Validate one terminal score cell entirely from bound identities."""

    required = {"training_receipt", "attachment_receipt", "score_receipt"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise HKEContractError(f"{label} score cell shape mismatch")

    training = value["training_receipt"]
    training_fields = {
        "schema",
        "kind",
        "status",
        "evidence_class",
        "plan_sha256",
        "config_sha256",
        "training_seed",
        "planned_steps",
        "sha256",
        "bytes",
        "checkpoint_step",
        "completed_steps",
        "source_record_sha256",
        "training_receipt_sha256",
    }
    if not isinstance(training, Mapping) or set(training) != training_fields:
        raise HKEContractError(f"{label} training receipt is malformed")
    training_body = dict(training)
    declared_training = training_body.pop("training_receipt_sha256")
    _require_sha256(declared_training, f"{label} training receipt")
    if declared_training != canonical_sha256(training_body):
        raise HKEContractError(f"{label} training receipt digest mismatch")
    if (
        training["schema"] != 1
        or training["kind"] != "sn56-week7-hke-training-receipt"
        or training["status"] != "OPERATOR_ATTESTED_PASS"
        or training["evidence_class"]
        != "content_bound_operator_attested_not_independent_proof"
        or training["plan_sha256"] != plan_sha256
        or training["config_sha256"] != plan_cell["config_sha256"]
        or training["training_seed"] != TRAINING_SEED_A
        or training["planned_steps"] != plan_cell["planned_steps"]
    ):
        raise HKEContractError(f"{label} training receipt provenance mismatch")
    artifact_sha = _require_sha256(training["sha256"], f"{label} artifact")
    _require_sha256(training["source_record_sha256"], f"{label} source record")
    if (
        isinstance(training["bytes"], bool)
        or not isinstance(training["bytes"], int)
        or training["bytes"] <= 0
    ):
        raise HKEContractError(f"{label} artifact size is invalid")
    planned = plan_cell["planned_steps"]
    if (
        training["checkpoint_step"] != planned
        or training["completed_steps"] != planned
    ):
        raise HKEContractError(f"{label} is not the planned terminal artifact")

    attachment = value["attachment_receipt"]
    attachment_fields = {
        "schema",
        "kind",
        "status",
        "evidence_class",
        "plan_sha256",
        "config_sha256",
        "artifact_sha256",
        "loaded_key_count",
        "unloaded_key_count",
        "log_sha256",
        "attachment_receipt_sha256",
    }
    if not isinstance(attachment, Mapping) or set(attachment) != attachment_fields:
        raise HKEContractError(f"{label} attachment receipt is malformed")
    attachment_body = dict(attachment)
    declared_attachment = attachment_body.pop("attachment_receipt_sha256")
    _require_sha256(declared_attachment, f"{label} attachment receipt")
    if declared_attachment != canonical_sha256(attachment_body):
        raise HKEContractError(f"{label} attachment receipt digest mismatch")
    _require_sha256(attachment["log_sha256"], f"{label} attachment log")
    if (
        attachment["schema"] != 1
        or attachment["kind"] != "sn56-week7-hke-attachment-receipt"
        or attachment["status"] != "OPERATOR_ATTESTED_PASS"
        or attachment["evidence_class"]
        != "content_bound_operator_attested_not_independent_proof"
        or attachment["plan_sha256"] != plan_sha256
        or attachment["config_sha256"] != plan_cell["config_sha256"]
        or attachment["artifact_sha256"] != artifact_sha
        or not isinstance(attachment["loaded_key_count"], int)
        or isinstance(attachment["loaded_key_count"], bool)
        or attachment["loaded_key_count"] <= 0
        or attachment["unloaded_key_count"] != 0
    ):
        raise HKEContractError(f"{label} attachment is not clean and bound")

    score = value["score_receipt"]
    score_fields = {
        "schema",
        "kind",
        "status",
        "evidence_class",
        "plan_sha256",
        "config_sha256",
        "artifact_sha256",
        "evaluator_sha256",
        "fixture_candidate_semantic_sha256",
        "rows",
        "rows_sha256",
        "score_receipt_sha256",
    }
    if not isinstance(score, Mapping) or set(score) != score_fields:
        raise HKEContractError(f"{label} score identity is malformed")
    _require_sha256(score["score_receipt_sha256"], f"{label} score receipt")
    score_body = dict(score)
    declared_score_receipt = score_body.pop("score_receipt_sha256")
    if declared_score_receipt != canonical_sha256(score_body):
        raise HKEContractError(f"{label} score receipt digest mismatch")
    if (
        score["schema"] != 1
        or score["kind"] != "sn56-week7-hke-exact-score-receipt"
        or score["status"] != "OPERATOR_ATTESTED_PASS"
        or score["evidence_class"]
        != "content_bound_operator_attested_not_independent_proof"
        or score["plan_sha256"] != plan_sha256
        or score["config_sha256"] != plan_cell["config_sha256"]
        or score["artifact_sha256"] != artifact_sha
        or score["evaluator_sha256"] != evaluator_sha256
        or score["fixture_candidate_semantic_sha256"]
        != fixture["candidate_semantic_sha256"]
    ):
        raise HKEContractError(f"{label} score provenance mismatch")
    rows = score["rows"]
    if not isinstance(rows, list) or len(rows) != expected_row_count:
        raise HKEContractError(
            f"{label} must contain the exact {expected_row_count} discovery rows"
        )
    if score["rows_sha256"] != canonical_sha256(rows):
        raise HKEContractError(f"{label} score rows digest mismatch")
    result: dict[str, float] = {}
    row_identity: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "row_id",
            "row_sha256",
            "prompted_loss",
            "blank_loss",
        }:
            raise HKEContractError(f"{label} score row shape mismatch")
        row_id = str(row.get("row_id", "")).strip()
        row_sha = _require_sha256(row.get("row_sha256"), f"{label} row")
        identity = f"{row_id}:{row_sha}"
        if not row_id or identity in result:
            raise HKEContractError(f"{label} row identity is missing or duplicated")
        result[identity] = _composite(row)
        row_identity.append({"row_id": row_id, "row_sha256": row_sha})
    return result, canonical_sha256(row_identity)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise HKEContractError("cannot take a percentile of no values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def paired_interval(
    deltas: Sequence[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> tuple[float, float, float]:
    values = [float(value) for value in deltas]
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise HKEContractError("paired bootstrap requires two finite rows")
    if iterations < 100:
        raise HKEContractError("paired bootstrap iteration count is too small")
    rng = random.Random(seed)
    n = len(values)
    means = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(iterations)
    ]
    alpha = (1.0 - UNCERTAINTY_LEVEL) / 2.0
    return (
        sum(values) / n,
        _percentile(means, alpha),
        _percentile(means, 1.0 - alpha),
    )


def _factor_family_effect(
    rows: Mapping[str, Mapping[str, float]], factor: str
) -> dict[str, Any]:
    row_ids = set(rows["A"])
    if any(set(rows[arm]) != row_ids for arm in ARMS):
        raise HKEContractError("factorial arms do not share exact row identity")
    ordered = sorted(row_ids)
    deltas: list[float] = []
    baselines: list[float] = []
    if factor == "loss":
        for row_id in ordered:
            baseline = (rows["A"][row_id] + rows["C"][row_id]) / 2.0
            changed = (rows["B"][row_id] + rows["D"][row_id]) / 2.0
            baselines.append(baseline)
            deltas.append(baseline - changed)
    elif factor == "planner":
        for row_id in ordered:
            baseline = (rows["A"][row_id] + rows["B"][row_id]) / 2.0
            changed = (rows["C"][row_id] + rows["D"][row_id]) / 2.0
            baselines.append(baseline)
            deltas.append(baseline - changed)
    else:
        raise HKEContractError("unknown factor")
    point, lower, upper = paired_interval(deltas)
    baseline_mean = sum(baselines) / len(baselines)
    relative = point / baseline_mean if baseline_mean > 0 else float("-inf")
    return {
        "rows": len(ordered),
        "improvement": point,
        "ci95": [lower, upper],
        "relative_improvement": relative,
        "direction": "improves" if point > 0 else "regresses" if point < 0 else "tie",
    }


def analyze(plan_value: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen discovery rule.  This result never authorizes shipping."""

    plan = _validate_plan(plan_value)
    evidence_fields = {
        "schema",
        "kind",
        "evidence_class",
        "plan_sha256",
        "evaluator_sha256",
        "training_seed",
        "families",
        "evidence_sha256",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != evidence_fields:
        raise HKEContractError("score evidence envelope is malformed")
    evidence_body = dict(evidence)
    declared_evidence = evidence_body.pop("evidence_sha256")
    _require_sha256(declared_evidence, "score evidence")
    if declared_evidence != canonical_sha256(evidence_body):
        raise HKEContractError("score evidence digest mismatch")
    if (
        evidence["schema"] != SCHEMA
        or evidence["kind"] != "sn56-week7-hke-score-evidence"
        or evidence["evidence_class"]
        != "content_bound_operator_attested_not_independent_proof"
        or evidence["plan_sha256"] != plan["plan_sha256"]
        or evidence["evaluator_sha256"] != plan["evaluator_sha256"]
        or evidence["training_seed"] != TRAINING_SEED_A
    ):
        raise HKEContractError("score evidence authority binding mismatch")
    scores = evidence["families"]
    if not isinstance(scores, Mapping) or set(scores) != {
        "product",
        "logo_ui",
        "social",
    }:
        raise HKEContractError("scores must contain product, logo_ui, and social")
    primary: dict[str, dict[str, dict[str, float]]] = {}
    for family in PRIMARY_FAMILIES:
        family_record = scores[family]
        if not isinstance(family_record, Mapping) or set(family_record) != {
            "phase",
            "fixture_admission_sha256",
            "fixture_candidate_semantic_sha256",
            "discovery_row_identity_sha256",
            "arms",
        }:
            raise HKEContractError(f"{family} evidence envelope is malformed")
        fixture = plan["fixtures"][family]
        if (
            family_record["phase"] != "discovery"
            or family_record["fixture_admission_sha256"]
            != fixture["admission_sha256"]
            or family_record["fixture_candidate_semantic_sha256"]
            != fixture["candidate_semantic_sha256"]
            or family_record["discovery_row_identity_sha256"]
            != fixture["discovery_row_identity_sha256"]
        ):
            raise HKEContractError(f"{family} fixture identity mismatch")
        family_scores = family_record["arms"]
        if not isinstance(family_scores, Mapping) or set(family_scores) != set(ARMS):
            raise HKEContractError(f"{family} must contain exact A-D arms")
        primary[family] = {}
        row_sets: set[str] = set()
        for arm in ARMS:
            rows, row_set_sha = _rows_by_id(
                family_scores[arm],
                f"{family}/{arm}",
                plan_cell=plan["cells"][family][arm],
                plan_sha256=plan["plan_sha256"],
                fixture=fixture,
                evaluator_sha256=plan["evaluator_sha256"],
                expected_row_count=fixture["counts"]["discovery"],
            )
            primary[family][arm] = rows
            row_sets.add(row_set_sha)
        if (
            len(row_sets) != 1
            or family_record["discovery_row_identity_sha256"] not in row_sets
        ):
            raise HKEContractError(f"{family} exact score-row identity mismatch")

    factor_results: dict[str, Any] = {}
    for factor in ("loss", "planner"):
        families = {
            family: _factor_family_effect(primary[family], factor)
            for family in PRIMARY_FAMILIES
        }
        same_improving_direction = all(
            families[family]["improvement"] > 0 for family in PRIMARY_FAMILIES
        )
        one_interval_clears_zero = any(
            families[family]["ci95"][0] > 0 for family in PRIMARY_FAMILIES
        )
        no_large_regression = all(
            families[family]["relative_improvement"] > -MAX_RELATIVE_REGRESSION
            for family in PRIMARY_FAMILIES
        )
        factor_results[factor] = {
            "families": families,
            "same_improving_direction": same_improving_direction,
            "one_interval_clears_zero": one_interval_clears_zero,
            "no_family_regression_ge_1pct": no_large_regression,
            "decision": (
                "GO"
                if same_improving_direction
                and one_interval_clears_zero
                and no_large_regression
                else "HOLD"
            ),
        }

    # Report, but do not use, the 2x2 interaction contrast.
    interactions: dict[str, Any] = {}
    for family, rows in primary.items():
        row_ids = sorted(rows["A"])
        deltas = [
            (rows["A"][row_id] - rows["B"][row_id])
            - (rows["C"][row_id] - rows["D"][row_id])
            for row_id in row_ids
        ]
        point, lower, upper = paired_interval(deltas)
        interactions[family] = {
            "contrast": point,
            "ci95": [lower, upper],
            "decision_relevant": False,
        }

    social_record = scores["social"]
    if not isinstance(social_record, Mapping) or set(social_record) != {
        "phase",
        "fixture_admission_sha256",
        "fixture_candidate_semantic_sha256",
        "discovery_row_identity_sha256",
        "arms",
    }:
        raise HKEContractError("social evidence envelope is malformed")
    social_fixture = plan["fixtures"]["social"]
    if (
        social_record["phase"] != "discovery"
        or social_record["fixture_admission_sha256"]
        != social_fixture["admission_sha256"]
        or social_record["fixture_candidate_semantic_sha256"]
        != social_fixture["candidate_semantic_sha256"]
        or social_record["discovery_row_identity_sha256"]
        != social_fixture["discovery_row_identity_sha256"]
    ):
        raise HKEContractError("social fixture identity mismatch")
    social_scores = social_record["arms"]
    if not isinstance(social_scores, Mapping) or set(social_scores) != {"A", "D"}:
        raise HKEContractError("social reliability must contain exact A and D")
    social_rows: dict[str, dict[str, float]] = {}
    social_row_sets: set[str] = set()
    for arm in ("A", "D"):
        rows, row_set_sha = _rows_by_id(
            social_scores[arm],
            f"social/{arm}",
            plan_cell=plan["cells"]["social"][arm],
            plan_sha256=plan["plan_sha256"],
            fixture=social_fixture,
            evaluator_sha256=plan["evaluator_sha256"],
            expected_row_count=social_fixture["counts"]["discovery"],
        )
        social_rows[arm] = rows
        social_row_sets.add(row_set_sha)
    if (
        len(social_row_sets) != 1
        or social_record["discovery_row_identity_sha256"]
        not in social_row_sets
    ):
        raise HKEContractError("social exact score-row identity mismatch")
    if set(social_rows["A"]) != set(social_rows["D"]):
        raise HKEContractError("social A/D row identity mismatch")
    social_delta = [
        social_rows["A"][row] - social_rows["D"][row]
        for row in sorted(social_rows["A"])
    ]
    social_point, social_lower, social_upper = paired_interval(social_delta)

    body = {
        "schema": SCHEMA,
        "kind": "sn56-week7-hke-factorial-decision",
        "status": "OPERATOR_ATTESTED_DISCOVERY_ONLY_NO_SHIP_AUTHORITY",
        "evidence_class": (
            "content_bound_operator_attested_not_independent_measurement_proof"
        ),
        "contract_sha256": experiment_contract()["contract_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "evaluator_sha256": plan["evaluator_sha256"],
        "training_seed": TRAINING_SEED_A,
        "analysis_phase": "discovery",
        "factors": factor_results,
        "interaction": interactions,
        "social_reliability": {
            "operator_attested_terminal_and_attached": True,
            "d_vs_a_improvement": social_point,
            "ci95": [social_lower, social_upper],
            "decision_relevant_to_factor_go": False,
        },
        "authorization": {
            "runtime_bridge_authorized": False,
            "checkpoint_promotion_authorized": False,
            "semantic_routing_authorized": False,
            "deployment_authorized": False,
        },
    }
    return {**body, "decision_sha256": canonical_sha256(body)}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("contract")
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--plan", type=Path, required=True)
    analyze_parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "contract":
        sys.stdout.buffer.write(canonical_bytes(experiment_contract()))
        return 0
    result = analyze(_load_json(args.plan), _load_json(args.evidence))
    sys.stdout.buffer.write(canonical_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
