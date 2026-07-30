#!/usr/bin/env python3
"""Fail-closed owner-directed acceleration for Week-5 Krea discovery.

The ordinary discovery contract requires six measured timing cells.  This
additive compatibility layer records the owner's time-critical decision to
stop after the exact D1/A measurement and use it only as a *conservative
proxy* for the remaining cells.  It does not rewrite, relabel, or claim that
the historical measurement was produced on another fixture or recipe class.

One sealed campaign contains twelve exact cell specifications.  Downstream
execution plans remain per-cell artifacts because a natural-completion record
and its candidates cannot safely span two training runs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

try:
    from . import krea_provenance
except ImportError:  # pragma: no cover - direct script execution.
    import krea_provenance  # type: ignore[no-redef]


_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_POLICY_PATH = Path(__file__).with_name("week5") / "krea-accelerated-discovery-policy.json"
_POLICY_KIND = "forge-krea-owner-accelerated-discovery-policy"
_CAMPAIGN_KIND = "forge-krea-accelerated-discovery-campaign"
_SLIP_KIND = "forge-krea-accelerated-discovery-schedule-slip"
_K4_KIND = "forge-krea-accelerated-discovery-k4-correction"
_FIXTURES = ("D1", "D2")
_ARMS = ("K0", "K1", "K2", "K3", "K4", "K5")
_CLASS_BY_ARM = {
    "K0": "A-rank32-adamw8bit-mse-guidance2",
    "K1": "A-rank32-adamw8bit-mse-guidance2",
    "K2": "A-rank32-adamw8bit-mse-guidance2",
    "K3": "B-rank32-adamw8bit-mae-guidance3",
    "K4": "C-rank64-automagic-mse-guidance2",
    "K5": "A-rank32-adamw8bit-mse-guidance2",
}


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value
    ):
        raise ValueError(f"{label} must be whole-second UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_file(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    raw = path.read_bytes()
    try:
        value = _object(json.loads(raw), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if raw != krea_provenance.canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline")
    return value, hashlib.sha256(raw).hexdigest()


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{label} must be decimal text or a number")
    try:
        result = Decimal(str(value))
    except Exception as exc:  # pragma: no cover - Decimal's exact exception varies.
        raise ValueError(f"{label} is not decimal") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _policy() -> tuple[dict[str, Any], str]:
    policy, file_sha = _canonical_file(_POLICY_PATH, "accelerated discovery policy")
    _exact(
        policy,
        {
            "schema",
            "kind",
            "owner_standing_authorization",
            "historical_timing",
            "base_hard_budget_s",
            "proxy_runtime_factors",
            "cadence_relief",
            "k4_first_checkpoint_correction",
            "claim_limit",
        },
        "accelerated discovery policy",
    )
    owner = _object(policy["owner_standing_authorization"], "owner standing order")
    timing = _object(policy["historical_timing"], "historical timing policy")
    factors = _object(policy["proxy_runtime_factors"], "proxy runtime factors")
    cadence = _object(policy["cadence_relief"], "cadence relief")
    correction = _object(
        policy["k4_first_checkpoint_correction"], "K4 correction policy"
    )
    if (
        policy["schema"] != 1
        or policy["kind"] != _POLICY_KIND
        or owner
        != {
            "accountable_owner_identity": "Atulya Shetty",
            "notes_file_bytes": 3160,
            "notes_file_sha256": (
                "cf3e618dff3983b4a28f5603a6b67b5f834b6785bf198911aea02f42c1162952"
            ),
            "order": 9,
            "scope": (
                "all campaign actions on the path to the win; no per-step owner "
                "ratification"
            ),
        }
        or timing.get("measured_source_commit")
        != "58822b496019177a02fa6196247ac30e788331bb"
        or timing.get("measured_fixture_id") != "D1"
        or timing.get("measured_throughput_equivalence_class")
        != _CLASS_BY_ARM["K1"]
        or timing.get("reuse_mode")
        != "explicit_conservative_proxy_not_measured_equivalence"
        or policy["base_hard_budget_s"] != 2700
        or set(factors) != set(_FIXTURES)
        or any(set(factors[fixture]) != set(_CLASS_BY_ARM.values()) for fixture in _FIXTURES)
        or cadence
        != {
            "allowed_multipliers": [1, 2],
            "activation": (
                "positive-create-only-schedule-slip-record-bound-to-the-campaign"
            ),
            "depth_increase_forbidden": True,
            "normal_multiplier": 1,
            "relief_multiplier": 2,
        }
        or correction
        != {
            "factor_decrease_forbidden": True,
            "margin_multiplier": "1.25",
            "maximum_factor": "4.00",
            "minimum_factor": "2.50",
            "mode": (
                "one-way-correction-for-unlaunched-D2-K4-from-completed-D1-K4-"
                "first-checkpoint"
            ),
            "round_up_increment": "0.05",
        }
    ):
        raise ValueError("accelerated discovery policy differs from owner directive")
    for fixture in _FIXTURES:
        for class_name, raw in factors[fixture].items():
            _decimal(raw, f"{fixture}/{class_name} runtime factor")
    return policy, file_sha


def policy_binding() -> dict[str, str]:
    policy, file_sha = _policy()
    return {
        "path": str(_POLICY_PATH.resolve(strict=True)),
        "file_sha256": file_sha,
        "policy_sha256": krea_provenance.canonical_sha256(policy),
    }


def _binding(value: Any, label: str, semantic_key: str) -> dict[str, str]:
    row = _object(value, label)
    _exact(row, {"path", "file_sha256", semantic_key}, label)
    if not isinstance(row["path"], str) or not row["path"]:
        raise ValueError(f"{label}.path is invalid")
    _digest(row["file_sha256"], f"{label}.file_sha256")
    _digest(row[semantic_key], f"{label}.{semantic_key}")
    return dict(row)


def _canonical_binding(value: Any, label: str, semantic_key: str) -> dict[str, str]:
    row = _binding(value, label, semantic_key)
    path = Path(os.path.abspath(os.path.expanduser(row["path"])))
    document, file_sha = _canonical_file(path, label)
    if (
        file_sha != row["file_sha256"]
        or document.get(semantic_key) != row[semantic_key]
    ):
        raise ValueError(f"{label} binding drifted")
    return {**row, "path": str(path)}


def _cells(*, cadence_multiplier: int) -> list[dict[str, Any]]:
    policy, _ = _policy()
    factors = policy["proxy_runtime_factors"]
    hard = Decimal(policy["base_hard_budget_s"])
    rows: list[dict[str, Any]] = []
    for fixture in _FIXTURES:
        for arm in _ARMS:
            class_name = _CLASS_BY_ARM[arm]
            factor = _decimal(factors[fixture][class_name], "runtime factor")
            effective = int((hard / factor).to_integral_value(rounding=ROUND_FLOOR))
            body = {
                "cell_id": f"{fixture}-{arm}",
                "fixture_id": fixture,
                "arm_id": arm,
                "throughput_equivalence_class": class_name,
                "measured_source_cell": {
                    "fixture_id": "D1",
                    "throughput_equivalence_class": _CLASS_BY_ARM["K1"],
                },
                "timing_evidence_mode": "conservative_proxy_not_measured_equivalence",
                "runtime_factor": format(factor, "f"),
                "base_hard_budget_s": int(hard),
                "effective_hard_budget_s": effective,
                "cadence_multiplier": cadence_multiplier,
                "depth_increase_from_cadence_relief": False,
            }
            rows.append({**body, "cell_sha256": krea_provenance.canonical_sha256(body)})
    return rows


def build_campaign(payload: dict[str, Any]) -> dict[str, Any]:
    """Seal one umbrella containing twelve exact, independently runnable cells."""

    payload = _object(payload, "accelerated campaign payload")
    _exact(
        payload,
        {
            "discovery_plan",
            "discovery_execution_authorization",
            "fixture_admission_envelope",
            "measured_profile",
            "historical_host_execution_manifest",
            "created_at_utc",
            "cadence_multiplier",
            "schedule_slip_record",
            "supersedes_campaign_sha256",
        },
        "accelerated campaign payload",
    )
    policy, policy_file_sha = _policy()
    cadence = payload["cadence_multiplier"]
    if cadence not in policy["cadence_relief"]["allowed_multipliers"]:
        raise ValueError("cadence multiplier is not owner-preauthorized")
    slip_binding = payload["schedule_slip_record"]
    if cadence == 1:
        if slip_binding is not None or payload["supersedes_campaign_sha256"] is not None:
            raise ValueError("normal cadence cannot claim schedule-slip relief")
    else:
        if slip_binding is None:
            raise ValueError("relief cadence requires a bound schedule-slip record")
        _binding(slip_binding, "schedule slip record", "slip_sha256")
        _digest(payload["supersedes_campaign_sha256"], "superseded campaign SHA-256")
    body = {
        "schema": 1,
        "kind": _CAMPAIGN_KIND,
        "owner_directive_policy": {
            "path": str(_POLICY_PATH.resolve(strict=True)),
            "file_sha256": policy_file_sha,
            "policy_sha256": krea_provenance.canonical_sha256(policy),
            "standing_order": 9,
        },
        "discovery_plan": _binding(
            payload["discovery_plan"], "discovery plan", "discovery_sha256"
        ),
        "discovery_execution_authorization": _binding(
            payload["discovery_execution_authorization"],
            "discovery execution authorization",
            "authorization_sha256",
        ),
        "fixture_admission_envelope": _binding(
            payload["fixture_admission_envelope"],
            "fixture admission envelope",
            "envelope_sha256",
        ),
        "measured_profile": {
            **_binding(payload["measured_profile"], "measured profile", "profile_sha256"),
            "fixture_id": "D1",
            "throughput_equivalence_class": _CLASS_BY_ARM["K1"],
        },
        "historical_host_execution_manifest": _binding(
            payload["historical_host_execution_manifest"],
            "historical host execution manifest",
            "host_execution_identity_sha256",
        ),
        "historical_compatibility": {
            "source_commit": policy["historical_timing"]["measured_source_commit"],
            "measured_bytes_remain_immutable": True,
            "proxy_cells_do_not_claim_measured_equivalence": True,
            "new_execution_must_bind_a_fresh_live_host_manifest": True,
        },
        "created_at_utc": _strict_utc(payload["created_at_utc"], "created_at_utc"),
        "cadence_multiplier": cadence,
        "schedule_slip_record": slip_binding,
        "supersedes_campaign_sha256": payload["supersedes_campaign_sha256"],
        "cells": _cells(cadence_multiplier=cadence),
        "cell_count": 12,
        "gpu_execution_authorized": False,
        "claim_limit": policy["claim_limit"],
    }
    record = {**body, "campaign_sha256": krea_provenance.canonical_sha256(body)}
    validate_campaign(record)
    return record


def validate_campaign(value: dict[str, Any]) -> dict[str, Any]:
    value = _object(value, "accelerated discovery campaign")
    _exact(
        value,
        {
            "schema",
            "kind",
            "owner_directive_policy",
            "discovery_plan",
            "discovery_execution_authorization",
            "fixture_admission_envelope",
            "measured_profile",
            "historical_host_execution_manifest",
            "historical_compatibility",
            "created_at_utc",
            "cadence_multiplier",
            "schedule_slip_record",
            "supersedes_campaign_sha256",
            "cells",
            "cell_count",
            "gpu_execution_authorized",
            "claim_limit",
            "campaign_sha256",
        },
        "accelerated discovery campaign",
    )
    policy, policy_file_sha = _policy()
    policy_row = _object(value["owner_directive_policy"], "directive policy binding")
    if policy_row != {
        "path": str(_POLICY_PATH.resolve(strict=True)),
        "file_sha256": policy_file_sha,
        "policy_sha256": krea_provenance.canonical_sha256(policy),
        "standing_order": 9,
    }:
        raise ValueError("campaign does not bind the exact owner directive policy")
    cadence = value["cadence_multiplier"]
    if cadence not in {1, 2}:
        raise ValueError("campaign cadence multiplier is not preauthorized")
    for key, semantic in (
        ("discovery_plan", "discovery_sha256"),
        ("discovery_execution_authorization", "authorization_sha256"),
        ("fixture_admission_envelope", "envelope_sha256"),
        ("historical_host_execution_manifest", "host_execution_identity_sha256"),
    ):
        _binding(value[key], key, semantic)
    measured = _object(value["measured_profile"], "measured profile")
    _exact(
        measured,
        {
            "path",
            "file_sha256",
            "profile_sha256",
            "fixture_id",
            "throughput_equivalence_class",
        },
        "measured profile",
    )
    if (
        measured["fixture_id"] != "D1"
        or measured["throughput_equivalence_class"] != _CLASS_BY_ARM["K1"]
    ):
        raise ValueError("campaign must preserve the exact D1/A measured profile")
    _digest(measured["file_sha256"], "measured profile file SHA-256")
    _digest(measured["profile_sha256"], "measured profile semantic SHA-256")
    compatibility = _object(value["historical_compatibility"], "compatibility")
    if compatibility != {
        "source_commit": "58822b496019177a02fa6196247ac30e788331bb",
        "measured_bytes_remain_immutable": True,
        "proxy_cells_do_not_claim_measured_equivalence": True,
        "new_execution_must_bind_a_fresh_live_host_manifest": True,
    }:
        raise ValueError("historical timing compatibility was weakened")
    _strict_utc(value["created_at_utc"], "created_at_utc")
    if value["cells"] != _cells(cadence_multiplier=cadence) or value["cell_count"] != 12:
        raise ValueError("campaign does not contain the exact twelve-cell matrix")
    if cadence == 1:
        if value["schedule_slip_record"] is not None or value["supersedes_campaign_sha256"] is not None:
            raise ValueError("normal campaign cannot claim cadence relief")
    else:
        binding = _binding(
            value["schedule_slip_record"], "schedule slip record", "slip_sha256"
        )
        slip, file_sha = _canonical_file(Path(binding["path"]), "schedule slip record")
        if file_sha != binding["file_sha256"] or slip["slip_sha256"] != binding["slip_sha256"]:
            raise ValueError("schedule slip binding drifted")
        validate_schedule_slip(slip)
        superseded = _digest(
            value["supersedes_campaign_sha256"], "superseded campaign SHA-256"
        )
        if slip["campaign_sha256"] != superseded:
            raise ValueError("schedule slip does not bind the superseded campaign")
    body = {key: item for key, item in value.items() if key != "campaign_sha256"}
    if (
        value["schema"] != 1
        or value["kind"] != _CAMPAIGN_KIND
        or value["gpu_execution_authorized"] is not False
        or value["claim_limit"] != policy["claim_limit"]
        or value["campaign_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("accelerated campaign identity is invalid")
    return value


def load_campaign_binding(value: Any) -> tuple[Path, dict[str, Any], str]:
    """Open a canonical file+semantic campaign binding without path aliases."""

    binding = _object(value, "accelerated campaign binding")
    _exact(
        binding,
        {"path", "file_sha256", "campaign_sha256"},
        "accelerated campaign binding",
    )
    path = Path(os.path.abspath(os.path.expanduser(binding["path"])))
    campaign, file_sha = _canonical_file(path, "accelerated discovery campaign")
    validate_campaign(campaign)
    if (
        file_sha != _digest(binding["file_sha256"], "campaign file SHA-256")
        or campaign["campaign_sha256"]
        != _digest(binding["campaign_sha256"], "campaign semantic SHA-256")
    ):
        raise ValueError("accelerated campaign binding drifted")
    return path, campaign, file_sha


def campaign_cell(campaign: Mapping[str, Any], fixture_id: str, arm_id: str) -> dict[str, Any]:
    validate_campaign(dict(campaign))
    cell_id = f"{fixture_id}-{arm_id}"
    matches = [row for row in campaign["cells"] if row["cell_id"] == cell_id]
    if len(matches) != 1:
        raise ValueError("accelerated campaign cell is absent or duplicated")
    return dict(matches[0])


def build_schedule_slip(
    *,
    campaign_sha256: str,
    observed_at_utc: str,
    schedule_slip_s: int,
    completed_cell_ids: list[str],
) -> dict[str, Any]:
    if isinstance(schedule_slip_s, bool) or not isinstance(schedule_slip_s, int) or schedule_slip_s <= 0:
        raise ValueError("schedule slip must be a positive whole second")
    if (
        not isinstance(completed_cell_ids, list)
        or completed_cell_ids != sorted(set(completed_cell_ids))
        or any(not _SAFE_ID.fullmatch(item) for item in completed_cell_ids)
    ):
        raise ValueError("completed cell ids are invalid")
    body = {
        "schema": 1,
        "kind": _SLIP_KIND,
        "campaign_sha256": _digest(campaign_sha256, "campaign SHA-256"),
        "observed_at_utc": _strict_utc(observed_at_utc, "observed_at_utc"),
        "schedule_slip_s": schedule_slip_s,
        "completed_cell_ids": completed_cell_ids,
        "decision": "activate_every_other_checkpoint",
        "cadence_multiplier": 2,
        "depth_increase_authorized": False,
    }
    return {**body, "slip_sha256": krea_provenance.canonical_sha256(body)}


def validate_schedule_slip(value: dict[str, Any]) -> dict[str, Any]:
    value = _object(value, "schedule slip")
    _exact(
        value,
        {
            "schema",
            "kind",
            "campaign_sha256",
            "observed_at_utc",
            "schedule_slip_s",
            "completed_cell_ids",
            "decision",
            "cadence_multiplier",
            "depth_increase_authorized",
            "slip_sha256",
        },
        "schedule slip",
    )
    rebuilt = build_schedule_slip(
        campaign_sha256=value["campaign_sha256"],
        observed_at_utc=value["observed_at_utc"],
        schedule_slip_s=value["schedule_slip_s"],
        completed_cell_ids=value["completed_cell_ids"],
    )
    if value != rebuilt:
        raise ValueError("schedule slip is not canonical or weakens relief")
    return value


def build_k4_correction(
    *,
    campaign_sha256: str,
    source_run_bundle: Any,
    predicted_first_checkpoint_s: Any,
    observed_first_checkpoint_s: Any,
    observed_at_utc: str,
) -> dict[str, Any]:
    policy, _ = _policy()
    rule = policy["k4_first_checkpoint_correction"]
    predicted = _decimal(predicted_first_checkpoint_s, "predicted checkpoint seconds")
    observed = _decimal(observed_first_checkpoint_s, "observed checkpoint seconds")
    minimum = Decimal(rule["minimum_factor"])
    maximum = Decimal(rule["maximum_factor"])
    raw = minimum * (observed / predicted) * Decimal(rule["margin_multiplier"])
    increment = Decimal(rule["round_up_increment"])
    corrected = max(minimum, (raw / increment).to_integral_value(rounding=ROUND_CEILING) * increment)
    if corrected > maximum:
        raise ValueError("K4 observation exceeds the preauthorized correction envelope")
    body = {
        "schema": 1,
        "kind": _K4_KIND,
        "campaign_sha256": _digest(campaign_sha256, "campaign SHA-256"),
        "source_cell_id": "D1-K4",
        "target_cell_id": "D2-K4",
        "source_run_bundle": _canonical_binding(
            source_run_bundle, "D1-K4 run-evidence bundle", "bundle_sha256"
        ),
        "predicted_first_checkpoint_s": format(predicted, "f"),
        "observed_first_checkpoint_s": format(observed, "f"),
        "base_runtime_factor": format(minimum, "f"),
        "corrected_runtime_factor": format(corrected, ".2f"),
        "factor_decrease_forbidden": True,
        "depth_increase_authorized": False,
        "observed_at_utc": _strict_utc(observed_at_utc, "observed_at_utc"),
    }
    return {**body, "correction_sha256": krea_provenance.canonical_sha256(body)}


def validate_k4_correction(value: dict[str, Any]) -> dict[str, Any]:
    value = _object(value, "K4 correction")
    _exact(
        value,
        {
            "schema",
            "kind",
            "campaign_sha256",
            "source_cell_id",
            "target_cell_id",
            "source_run_bundle",
            "predicted_first_checkpoint_s",
            "observed_first_checkpoint_s",
            "base_runtime_factor",
            "corrected_runtime_factor",
            "factor_decrease_forbidden",
            "depth_increase_authorized",
            "observed_at_utc",
            "correction_sha256",
        },
        "K4 correction",
    )
    rebuilt = build_k4_correction(
        campaign_sha256=value["campaign_sha256"],
        source_run_bundle=value["source_run_bundle"],
        predicted_first_checkpoint_s=value["predicted_first_checkpoint_s"],
        observed_first_checkpoint_s=value["observed_first_checkpoint_s"],
        observed_at_utc=value["observed_at_utc"],
    )
    if value != rebuilt:
        raise ValueError("K4 correction is not canonical or one-way")
    return value


def load_k4_correction_binding(value: Any) -> tuple[Path, dict[str, Any], str]:
    binding = _object(value, "K4 correction binding")
    _exact(
        binding,
        {"path", "file_sha256", "correction_sha256"},
        "K4 correction binding",
    )
    path = Path(os.path.abspath(os.path.expanduser(binding["path"])))
    correction, file_sha = _canonical_file(path, "K4 correction")
    validate_k4_correction(correction)
    if (
        file_sha != _digest(binding["file_sha256"], "K4 correction file SHA-256")
        or correction["correction_sha256"]
        != _digest(binding["correction_sha256"], "K4 correction semantic SHA-256")
    ):
        raise ValueError("K4 correction binding drifted")
    return path, correction, file_sha


def derive_cell_controls(
    *,
    campaign_path: Path,
    profile_index_path: Path,
    throughput_profile_path: Path,
    fixture_id: str,
    arm_id: str,
) -> dict[str, Any]:
    """Derive the sole budget/schedule overrides for one queued cell."""

    try:
        from . import krea_budget
        from . import krea_runtime_binding
    except ImportError:  # pragma: no cover - direct script execution.
        import krea_budget  # type: ignore[no-redef]
        import krea_runtime_binding  # type: ignore[no-redef]

    campaign, campaign_file_sha = _canonical_file(
        campaign_path, "accelerated discovery campaign"
    )
    validate_campaign(campaign)
    index, index_file_sha = _canonical_file(
        profile_index_path, "accelerated profile index"
    )
    krea_runtime_binding.validate_profile_index(index)
    if (
        index.get("schema") != 3
        or index["accelerated_discovery_campaign"]
        != {
            "path": str(Path(campaign_path).resolve(strict=True)),
            "file_sha256": campaign_file_sha,
            "campaign_sha256": campaign["campaign_sha256"],
        }
    ):
        raise ValueError("cell derivation campaign/index binding drifted")
    cell = campaign_cell(campaign, fixture_id, arm_id)
    slot = index["fixtures"][fixture_id]["profiles"][
        cell["throughput_equivalence_class"]
    ]
    source_profile, source_file_sha = _canonical_file(
        throughput_profile_path, "D1/A source throughput profile"
    )
    profile = krea_budget.load_throughput_profile(source_profile)
    if (
        slot["source_profile"]["path"]
        != str(Path(throughput_profile_path).resolve(strict=True))
        or slot["source_profile"]["file_sha256"] != source_file_sha
        or slot["source_profile"]["profile_sha256"]
        != source_profile["profile_sha256"]
    ):
        raise ValueError("cell derivation source profile drifted")
    effective_cell = dict(cell)
    if cell["cell_id"] == "D2-K4" and index.get("k4_correction") is not None:
        effective_cell = {
            **effective_cell,
            "base_runtime_factor": cell["runtime_factor"],
            "base_effective_hard_budget_s": cell["effective_hard_budget_s"],
            "runtime_factor": slot["runtime_factor"],
            "effective_hard_budget_s": slot["effective_hard_budget_s"],
            "k4_correction": dict(index["k4_correction"]),
        }
    budget = krea_budget.plan_budget(
        profile,
        hard_budget_s=effective_cell["effective_hard_budget_s"],
    ).to_record()
    if arm_id == "K0":
        baseline = {"D1": (260, 53), "D2": (367, 74)}
        planned, base_cadence = baseline[fixture_id]
        if planned > budget["max_affordable_steps"]:
            raise ValueError("accelerated K0 release depth does not fit its cell budget")
        cadence = base_cadence * effective_cell["cadence_multiplier"]
        mode = "release_control"
    else:
        planned = budget["max_affordable_steps"]
        cadence = budget["save_every"] * effective_cell["cadence_multiplier"]
        mode = "measured_budget_fill"
    candidates = list(range(cadence, planned, cadence)) + [planned]
    landmarks = {"K2": 960, "K3": 1200, "K4": 840}
    landmark = landmarks.get(arm_id)
    required_landmarks = [landmark] if landmark in candidates else []
    schedule = {
        "mode": mode,
        "planned_steps": planned,
        "save_every": cadence,
        "candidate_steps": candidates,
        "required_landmarks": required_landmarks,
        "landmark_policy": (
            "preserve_if_budget_safe" if required_landmarks else "none"
        ),
    }
    body = {
        "schema": 1,
        "kind": "forge-krea-accelerated-cell-controls",
        "campaign_sha256": campaign["campaign_sha256"],
        "profile_index_sha256": index["index_sha256"],
        "profile_index_file_sha256": index_file_sha,
        "source_profile_sha256": source_profile["profile_sha256"],
        "cell": effective_cell,
        "budget_plan": budget,
        "budget_plan_sha256": krea_provenance.canonical_sha256(budget),
        "schedule": schedule,
        "recipe_overrides": {
            "planned_steps": planned,
            "save_cadence": cadence,
        },
    }
    return {**body, "controls_sha256": krea_provenance.canonical_sha256(body)}


def _publish(path: Path, value: dict[str, Any]) -> None:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"output has a symlink ancestor: {current}")
        current = current.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = krea_provenance.canonical_bytes(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal-campaign")
    seal.add_argument("--payload", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate-campaign")
    validate.add_argument("--campaign", type=Path, required=True)
    slip = commands.add_parser("seal-schedule-slip")
    slip.add_argument("--payload", type=Path, required=True)
    slip.add_argument("--output", type=Path, required=True)
    correction = commands.add_parser("seal-k4-correction")
    correction.add_argument("--payload", type=Path, required=True)
    correction.add_argument("--output", type=Path, required=True)
    derive = commands.add_parser("derive-cell-controls")
    derive.add_argument("--campaign", type=Path, required=True)
    derive.add_argument("--profile-index", type=Path, required=True)
    derive.add_argument("--throughput-profile", type=Path, required=True)
    derive.add_argument("--fixture", choices=_FIXTURES, required=True)
    derive.add_argument("--arm", choices=_ARMS, required=True)
    derive.add_argument("--output", type=Path, required=True)
    commands.add_parser("show-policy-binding")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "show-policy-binding":
        print(krea_provenance.canonical_bytes(policy_binding()).decode("ascii"))
        return 0
    if args.command == "seal-campaign":
        payload, _ = _canonical_file(args.payload, "accelerated campaign payload")
        campaign = build_campaign(payload)
        _publish(args.output, campaign)
    elif args.command == "seal-schedule-slip":
        payload, _ = _canonical_file(args.payload, "schedule-slip payload")
        _exact(
            payload,
            {
                "campaign_sha256",
                "observed_at_utc",
                "schedule_slip_s",
                "completed_cell_ids",
            },
            "schedule-slip payload",
        )
        slip = build_schedule_slip(**payload)
        _publish(args.output, slip)
        print(
            krea_provenance.canonical_bytes(
                {
                    "status": "PASS",
                    "action": args.command,
                    "slip_sha256": slip["slip_sha256"],
                }
            ).decode("ascii")
        )
        return 0
    elif args.command == "seal-k4-correction":
        payload, _ = _canonical_file(args.payload, "K4 correction payload")
        _exact(
            payload,
            {
                "campaign_sha256",
                "source_run_bundle",
                "predicted_first_checkpoint_s",
                "observed_first_checkpoint_s",
                "observed_at_utc",
            },
            "K4 correction payload",
        )
        correction = build_k4_correction(**payload)
        _publish(args.output, correction)
        print(
            krea_provenance.canonical_bytes(
                {
                    "status": "PASS",
                    "action": args.command,
                    "correction_sha256": correction["correction_sha256"],
                }
            ).decode("ascii")
        )
        return 0
    elif args.command == "derive-cell-controls":
        controls = derive_cell_controls(
            campaign_path=args.campaign,
            profile_index_path=args.profile_index,
            throughput_profile_path=args.throughput_profile,
            fixture_id=args.fixture,
            arm_id=args.arm,
        )
        _publish(args.output, controls)
        print(
            krea_provenance.canonical_bytes(
                {
                    "status": "PASS",
                    "action": args.command,
                    "cell_id": controls["cell"]["cell_id"],
                    "controls_sha256": controls["controls_sha256"],
                }
            ).decode("ascii")
        )
        return 0
    else:
        campaign, _ = _canonical_file(
            args.campaign, "accelerated discovery campaign"
        )
        validate_campaign(campaign)
    print(
        krea_provenance.canonical_bytes(
            {
                "status": "PASS",
                "action": args.command,
                "campaign_sha256": campaign["campaign_sha256"],
            }
        ).decode("ascii")
    )
    return 0


__all__ = [
    "build_campaign",
    "build_k4_correction",
    "build_schedule_slip",
    "campaign_cell",
    "derive_cell_controls",
    "load_campaign_binding",
    "load_k4_correction_binding",
    "policy_binding",
    "validate_campaign",
    "validate_k4_correction",
    "validate_schedule_slip",
]


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests.
    raise SystemExit(main())
