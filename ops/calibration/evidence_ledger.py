#!/usr/bin/env python3
"""Fail-closed structural validation for the Week-5 evidence ledger.

This is experiment/release governance, not tournament runtime code.  The
validator keeps readiness claim structure mechanical: unknown readiness is
represented as ``false``; unnamed accountable humans fail the identity check;
and each readiness flag is derived from explicit, hash-bound result attestations
and tier gates. External review remains responsible for authenticating signers
and the scientific meaning of those attestations.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


_SHA256 = re.compile(r"[0-9a-f]{64}")
_MODEL_TYPES = ("krea2", "ideogram4", "flux", "qwen-image", "z-image")
_TOP_KEYS = frozenset(
    {
        "schema",
        "kind",
        "release_base_commit",
        "updated_at_utc",
        "status",
        "claim_policy",
        "assignments",
        "models",
        "release_blockers",
    }
)
_CLAIM_POLICY_KEYS = frozenset(
    {
        "mechanics_ready",
        "quality_evidenced",
        "field_parity_ready",
        "win_ready",
        "unknown_is_false",
        "waiver_required_for_open_gate",
    }
)
_ASSIGNMENT_KEYS = frozenset(
    {
        "tournament_owner_human",
        "forensics_dri",
        "krea_dri",
        "ideogram_dri",
        "scoring_dri",
        "independent_reviewer",
        "release_dri",
        "operations_dri",
    }
)
_MODEL_KEYS = frozenset(
    {
        "model_type",
        "in_scope_for_entry",
        "tier_scope",
        "mechanics_evidence_pass",
        "mechanics_ready",
        "quality_result_pass",
        "quality_result_sha256",
        "quality_evidenced",
        "field_parity_result_pass",
        "field_parity_result_sha256",
        "field_parity_ready",
        "win_ready",
        "round1_ready",
        "bracket_ready",
        "boss_ready",
        "fixtures",
        "training_seeds",
        "evaluation_rows",
        "public_arms_reproduced",
        "public_arm_provenance_manifest_sha256",
        "post_reserve_window_utilization",
        "selector_status",
        "worst_cell_regret",
        "worst_cell_regret_cap",
        "evidence_manifest_sha256",
        "public_bundle_scrub_pass",
        "public_bundle_sha256",
        "rules_contract_sha",
        "round1_evidence_sha256",
        "bracket_evidence_sha256",
        "boss_evidence_sha256",
        "open_risks",
        "owner",
        "reviewer",
        "waiver",
    }
)
_READINESS_KEYS = (
    "mechanics_ready",
    "quality_evidenced",
    "field_parity_ready",
    "round1_ready",
    "bracket_ready",
    "boss_ready",
    "win_ready",
)
_PLACEHOLDER_WORDS = (
    "codex",
    "role",
    "owner-confirmed",
    "name required",
    "unassigned",
    "response engineer",
    "primary task",
)
_WIN_SELECTOR_STATES = {
    "deterministic_policy_validated",
    "live_selector_validated",
}


class LedgerValidationError(ValueError):
    """The evidence ledger violates its exact schema or claim invariants."""


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise LedgerValidationError(
            f"{label} schema mismatch; missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )


def _optional_sha(value: Any, label: str) -> None:
    if value is not None and (
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
    ):
        raise LedgerValidationError(f"{label} must be null or a lowercase SHA-256")


def _human_name(value: Any) -> bool:
    if not isinstance(value, str) or len(value.strip().split()) < 2:
        return False
    lowered = value.lower()
    return not any(word in lowered for word in _PLACEHOLDER_WORDS)


def _utc_timestamp(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise LedgerValidationError(f"{label} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise LedgerValidationError(
            f"{label} must be an RFC3339 UTC timestamp"
        ) from exc
    if parsed.tzinfo != timezone.utc:
        raise LedgerValidationError(f"{label} must be UTC")


def _win_gate(model: Mapping[str, Any]) -> bool:
    tiers_ready = all(model[f"{tier}_ready"] for tier in model["tier_scope"])
    regret = model["worst_cell_regret"]
    regret_cap = model["worst_cell_regret_cap"]
    regret_pass = regret is not None and regret_cap is not None and regret <= regret_cap
    return all(
        (
            model["mechanics_ready"],
            model["quality_evidenced"],
            model["field_parity_ready"],
            tiers_ready,
            model["fixtures"] >= 4,
            model["training_seeds"] >= 1,
            model["evaluation_rows"] > 0,
            bool(model["public_arms_reproduced"]),
            model["public_arm_provenance_manifest_sha256"] is not None,
            model["post_reserve_window_utilization"] is not None
            and model["post_reserve_window_utilization"] >= 0.9,
            model["evidence_manifest_sha256"] is not None,
            model["public_bundle_scrub_pass"],
            model["public_bundle_sha256"] is not None,
            model["rules_contract_sha"] is not None,
            regret_pass,
            model["selector_status"] in _WIN_SELECTOR_STATES,
            _human_name(model["owner"]),
            _human_name(model["reviewer"]),
            model["waiver"] is None,
            not model["open_risks"],
        )
    )


def validate_ledger(
    document: Mapping[str, Any], *, require_named_dris: bool = False
) -> None:
    if not isinstance(document, Mapping):
        raise LedgerValidationError("ledger must be a JSON object")
    _exact_keys(document, _TOP_KEYS, "ledger")
    if (
        document["schema"] != 1
        or document["kind"] != "sn56-week5-release-evidence-ledger"
    ):
        raise LedgerValidationError("unsupported ledger identity")
    if not isinstance(document["release_base_commit"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", document["release_base_commit"]
    ):
        raise LedgerValidationError("release_base_commit must be a full Git SHA-1")
    if document["status"] not in {
        "day0_open",
        "experiments_open",
        "learning_entry_with_waiver",
        "release_frozen",
    }:
        raise LedgerValidationError("unsupported ledger status")
    _utc_timestamp(document["updated_at_utc"], "updated_at_utc")

    policy = document["claim_policy"]
    assignments = document["assignments"]
    if not isinstance(policy, Mapping) or not isinstance(assignments, Mapping):
        raise LedgerValidationError("claim_policy and assignments must be objects")
    _exact_keys(policy, _CLAIM_POLICY_KEYS, "claim_policy")
    _exact_keys(assignments, _ASSIGNMENT_KEYS, "assignments")
    if policy["unknown_is_false"] is not True:
        raise LedgerValidationError("unknown_is_false must remain true")
    if policy["waiver_required_for_open_gate"] is not True:
        raise LedgerValidationError("waiver_required_for_open_gate must remain true")
    for key in (
        "mechanics_ready",
        "quality_evidenced",
        "field_parity_ready",
        "win_ready",
    ):
        if not isinstance(policy[key], str) or not policy[key].strip():
            raise LedgerValidationError(f"claim_policy.{key} must be explanatory text")
    for key in _ASSIGNMENT_KEYS:
        value = assignments[key]
        if value is not None and not isinstance(value, str):
            raise LedgerValidationError(f"assignments.{key} must be null or text")

    models = document["models"]
    if not isinstance(models, list) or len(models) != len(_MODEL_TYPES):
        raise LedgerValidationError(
            "models must contain exactly the five supported types"
        )
    seen: set[str] = set()
    for index, model in enumerate(models):
        if not isinstance(model, Mapping):
            raise LedgerValidationError(f"models[{index}] must be an object")
        _exact_keys(model, _MODEL_KEYS, f"models[{index}]")
        model_type = model["model_type"]
        if model_type not in _MODEL_TYPES or model_type in seen:
            raise LedgerValidationError("models must contain each supported type once")
        seen.add(model_type)
        if not isinstance(model["in_scope_for_entry"], bool):
            raise LedgerValidationError(
                f"{model_type}.in_scope_for_entry must be boolean"
            )
        tier_scope = model["tier_scope"]
        if (
            not isinstance(tier_scope, list)
            or not tier_scope
            or len(tier_scope) != len(set(tier_scope))
            or not all(tier in {"round1", "bracket", "boss"} for tier in tier_scope)
        ):
            raise LedgerValidationError(
                f"{model_type}.tier_scope must be a unique nonempty tier list"
            )
        for key in (
            "mechanics_evidence_pass",
            "quality_result_pass",
            "field_parity_result_pass",
        ):
            if not isinstance(model[key], bool):
                raise LedgerValidationError(f"{model_type}.{key} must be boolean")
        for key in _READINESS_KEYS:
            if not isinstance(model[key], bool):
                raise LedgerValidationError(
                    f"{model_type}.{key} must be boolean; unknown is false"
                )
        if not isinstance(model["public_bundle_scrub_pass"], bool):
            raise LedgerValidationError(
                f"{model_type}.public_bundle_scrub_pass must be boolean"
            )
        for key in ("fixtures", "training_seeds", "evaluation_rows"):
            value = model[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LedgerValidationError(f"{model_type}.{key} must be >= 0")
        for key in (
            "public_arm_provenance_manifest_sha256",
            "quality_result_sha256",
            "field_parity_result_sha256",
            "evidence_manifest_sha256",
            "public_bundle_sha256",
            "rules_contract_sha",
            "round1_evidence_sha256",
            "bracket_evidence_sha256",
            "boss_evidence_sha256",
        ):
            _optional_sha(model[key], f"{model_type}.{key}")
        if not isinstance(model["public_arms_reproduced"], list) or not all(
            isinstance(value, str) and re.fullmatch(r"[A-Z][A-Z0-9_-]{1,31}", value)
            for value in model["public_arms_reproduced"]
        ):
            raise LedgerValidationError(
                f"{model_type}.public_arms_reproduced must be a canonical arm-ID list"
            )
        if len(model["public_arms_reproduced"]) != len(
            set(model["public_arms_reproduced"])
        ):
            raise LedgerValidationError(
                f"{model_type}.public_arms_reproduced contains duplicates"
            )
        if not isinstance(model["open_risks"], list) or not all(
            isinstance(value, str) and value for value in model["open_risks"]
        ):
            raise LedgerValidationError(
                f"{model_type}.open_risks must be a string list"
            )
        for key in (
            "post_reserve_window_utilization",
            "worst_cell_regret",
            "worst_cell_regret_cap",
        ):
            value = model[key]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise LedgerValidationError(
                    f"{model_type}.{key} must be null or finite numeric"
                )
        utilization = model["post_reserve_window_utilization"]
        if utilization is not None and not 0 <= utilization <= 1:
            raise LedgerValidationError(
                f"{model_type}.post_reserve_window_utilization must be in [0, 1]"
            )
        for key in ("worst_cell_regret", "worst_cell_regret_cap"):
            value = model[key]
            if value is not None and value < 0:
                raise LedgerValidationError(f"{model_type}.{key} must be >= 0")
        if (
            not isinstance(model["selector_status"], str)
            or not model["selector_status"].strip()
        ):
            raise LedgerValidationError(f"{model_type}.selector_status must be text")
        for key in ("owner", "reviewer"):
            if model[key] is not None and not isinstance(model[key], str):
                raise LedgerValidationError(f"{model_type}.{key} must be null or text")
        waiver = model["waiver"]
        if waiver is not None:
            if not isinstance(waiver, Mapping):
                raise LedgerValidationError(
                    f"{model_type}.waiver must be null or object"
                )
            _exact_keys(
                waiver,
                frozenset(
                    {"approved_by", "approved_at_utc", "consequence", "evidence_sha256"}
                ),
                f"{model_type}.waiver",
            )
            if not _human_name(waiver["approved_by"]):
                raise LedgerValidationError(
                    f"{model_type}.waiver approver must be named"
                )
            _utc_timestamp(
                waiver["approved_at_utc"], f"{model_type}.waiver approved_at_utc"
            )
            if (
                not isinstance(waiver["consequence"], str)
                or len(waiver["consequence"].strip()) < 20
            ):
                raise LedgerValidationError(
                    f"{model_type}.waiver must state the likely consequence"
                )
            _optional_sha(waiver["evidence_sha256"], f"{model_type}.waiver evidence")
            if waiver["evidence_sha256"] is None:
                raise LedgerValidationError(f"{model_type}.waiver must bind evidence")
        mechanics_gate = all(
            (
                model["mechanics_evidence_pass"],
                model["evidence_manifest_sha256"] is not None,
            )
        )
        if model["mechanics_ready"] != mechanics_gate:
            raise LedgerValidationError(
                f"{model_type}: mechanics_ready must equal bound mechanics evidence"
            )
        quality_gate = all(
            (
                model["mechanics_ready"],
                model["quality_result_pass"],
                model["quality_result_sha256"] is not None,
                model["fixtures"] >= 2,
                model["training_seeds"] >= 1,
                model["evaluation_rows"] > 0,
            )
        )
        if model["quality_evidenced"] != quality_gate:
            raise LedgerValidationError(
                f"{model_type}: quality_evidenced must equal its evidence conjunction"
            )
        utilization_ready = utilization is not None and utilization >= 0.9
        parity_gate = all(
            (
                model["quality_evidenced"],
                model["field_parity_result_pass"],
                model["field_parity_result_sha256"] is not None,
                model["fixtures"] >= 4,
                bool(model["public_arms_reproduced"]),
                model["public_arm_provenance_manifest_sha256"] is not None,
                utilization_ready,
                model["rules_contract_sha"] is not None,
            )
        )
        if model["field_parity_ready"] != parity_gate:
            raise LedgerValidationError(
                f"{model_type}: field_parity_ready must equal its evidence conjunction"
            )
        for tier in ("round1", "bracket", "boss"):
            tier_gate = all(
                (
                    tier in tier_scope,
                    model["field_parity_ready"],
                    model[f"{tier}_evidence_sha256"] is not None,
                )
            )
            if model[f"{tier}_ready"] != tier_gate:
                raise LedgerValidationError(
                    f"{model_type}.{tier}_ready must equal its scoped evidence gate"
                )
        if model["field_parity_ready"] and model["fixtures"] < 4:
            raise LedgerValidationError(
                f"{model_type}: field parity requires four fixtures"
            )
        if model["public_bundle_scrub_pass"] != (
            model["public_bundle_sha256"] is not None
        ):
            raise LedgerValidationError(
                f"{model_type}: public scrub PASS and bundle SHA must agree"
            )
        if model["win_ready"] != _win_gate(model):
            raise LedgerValidationError(
                f"{model_type}.win_ready must equal the conjunction of every win gate"
            )

    if seen != set(_MODEL_TYPES):
        raise LedgerValidationError("models omit a supported type")
    blockers = document["release_blockers"]
    if not isinstance(blockers, list) or not all(
        isinstance(value, str) and value for value in blockers
    ):
        raise LedgerValidationError("release_blockers must be a string list")
    all_win_ready = all(model["win_ready"] for model in models)
    all_named = all(_human_name(assignments[key]) for key in _ASSIGNMENT_KEYS)
    if not all_win_ready and not blockers:
        raise LedgerValidationError(
            "release_blockers cannot be empty while gates are red"
        )
    if document["status"] == "release_frozen" and (
        not all_win_ready or not all_named or blockers
    ):
        raise LedgerValidationError(
            "release_frozen requires every win gate, named DRI, and no blocker"
        )
    scoped_red = [
        model
        for model in models
        if model["in_scope_for_entry"] and not model["win_ready"]
    ]
    scoped_waivers_complete = all(model["waiver"] is not None for model in scoped_red)
    if document["status"] == "learning_entry_with_waiver" and (
        not scoped_red or not scoped_waivers_complete or not blockers or all_win_ready
    ):
        raise LedgerValidationError(
            "learning_entry_with_waiver requires a recorded waiver, retained "
            "blockers, and red win readiness"
        )

    if require_named_dris:
        unnamed = [key for key in _ASSIGNMENT_KEYS if not _human_name(assignments[key])]
        if unnamed:
            raise LedgerValidationError(
                "accountable human names missing: " + ", ".join(sorted(unnamed))
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--require-named-dris", action="store_true")
    args = parser.parse_args()
    path = args.ledger.expanduser()
    if path.is_symlink() or not path.is_file():
        raise LedgerValidationError("ledger must be a regular non-symlink file")
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_ledger(document, require_named_dris=args.require_named_dris)
    print(
        json.dumps(
            {"passed": True, "named_dri_check": args.require_named_dris},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
