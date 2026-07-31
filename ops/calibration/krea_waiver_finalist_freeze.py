#!/usr/bin/env python3
"""Freeze or independently review recovery-waiver Krea finalists.

This bridge applies the already frozen D1/D2 selection mathematics to a fully
validated recovery evidence index.  It does not retroactively authorize the
recovery campaign.  A three-way tie or material D1/D2 rank reversal still
requests the predeclared Seed B; a waiver cannot turn the evaluator's repeated
generation seeds into a second training seed.

Both outputs are create-only, bind their exact inputs, carry fixed false
authority claims, and are incapable of revealing C1-C4 or authorizing Stage-2,
production mutation, release, or deployment.  Review is a fresh non-human agent
review and is explicitly not a human or deployment approval.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import krea_decision
    from . import krea_delegated_review_contract
    from . import krea_provenance
    from . import krea_recovery_evidence
except ImportError:  # pragma: no cover - direct script execution.
    import krea_decision  # type: ignore[no-redef]
    import krea_delegated_review_contract  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_recovery_evidence  # type: ignore[no-redef]


WAIVER_KIND = "forge-krea-recovery-evidence-finalist-waiver"
FREEZE_KIND = "forge-krea-recovery-waiver-finalist-freeze"
REVIEW_KIND = "forge-krea-recovery-waiver-finalist-freeze-agent-review"
SCHEMA = 1
CONTROL_FAMILY = "K0"
NONCONTROL_FAMILIES = ("K1", "K2", "K3", "K4", "K5")
MAX_NONCONTROL_FINALISTS = 3
TIE_BAND = Decimal("0.01")
REPORT_TARGETS = tuple(
    Decimal(value) for value in ("0.1", "0.25", "0.5", "0.75", "0.9", "1.0")
)
PREPARER_ROLE = "recovery_waiver_finalist_freeze_preparer"
REVIEWER_ROLE = "recovery_waiver_finalist_freeze_reviewer"
OWNER_IDENTITY_ASSURANCE = (
    "interactive-owner-self-attestation-not-cryptographic-or-legal-signature"
)
FALSE_CLAIMS = dict(krea_recovery_evidence.FALSE_CLAIMS)
AUTHORITY = {
    "scope": "waived_D1_D2_recovery_evidence_to_finalist_freeze_only",
    "confirmation_fixture_reveal_authorized": False,
    "execution_authority": False,
    "deployment_authorized": False,
}


class WaiverFreezeError(ValueError):
    """Raised when a waiver, freeze, or review is not fail-closed."""


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise WaiverFreezeError(
            f"{label} keys mismatch: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )


def _timestamp(value: Any, label: str) -> str:
    try:
        return krea_recovery_evidence._timestamp(value, label)
    except krea_recovery_evidence.RecoveryEvidenceError as exc:
        raise WaiverFreezeError(str(exc)) from exc


def _timestamp_value(value: str, label: str) -> datetime:
    return datetime.strptime(_timestamp(value, label), "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )


def _load_control(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        value, binding = krea_recovery_evidence._load_json(path, label, canonical=True)
    except krea_recovery_evidence.RecoveryEvidenceError as exc:
        raise WaiverFreezeError(str(exc)) from exc
    return value, binding["file_sha256"]


def _fresh_actor(value: Any, *, role: str, label: str) -> dict[str, Any]:
    try:
        actor = krea_delegated_review_contract.reject_delegated_actor_reuse(
            value, label=label
        )
    except ValueError as exc:
        raise WaiverFreezeError(str(exc)) from exc
    if actor["role"] != role:
        raise WaiverFreezeError(f"{label} must have role {role}")
    return actor


def validate_waiver(
    value: Any,
    *,
    recovery_index_sha256: str,
    recovery_index_file_sha256: str,
) -> dict[str, Any]:
    """Validate externally supplied owner scope; never create or infer it."""

    if not isinstance(value, dict):
        raise WaiverFreezeError("recovery waiver must be an object")
    _exact(
        value,
        {
            "schema",
            "kind",
            "waiver_id",
            "approved_at_utc",
            "accountable_owner_identity",
            "owner_identity_assurance",
            "recovery_index_sha256",
            "recovery_index_file_sha256",
            "scope",
            "claims",
            "maximum_noncontrol_finalists",
            "independent_agent_review_required",
            "waiver_sha256",
        },
        "recovery waiver",
    )
    body = {key: item for key, item in value.items() if key != "waiver_sha256"}
    owner = krea_delegated_review_contract.load()["accountable_owner_identity"]
    if (
        value["schema"] != SCHEMA
        or value["kind"] != WAIVER_KIND
        or not isinstance(value["waiver_id"], str)
        or not value["waiver_id"].strip()
        or value["accountable_owner_identity"] != owner
        or value["owner_identity_assurance"] != OWNER_IDENTITY_ASSURANCE
        or value["recovery_index_sha256"] != recovery_index_sha256
        or value["recovery_index_file_sha256"] != recovery_index_file_sha256
        or value["scope"]
        != "use_validated_recovery_scores_for_non_authorizing_D1_D2_finalist_freeze"
        or value["claims"] != FALSE_CLAIMS
        or value["maximum_noncontrol_finalists"] != MAX_NONCONTROL_FINALISTS
        or value["independent_agent_review_required"] is not True
        or value["waiver_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise WaiverFreezeError("recovery waiver identity, scope, or claims drifted")
    _timestamp(value["approved_at_utc"], "waiver approved_at_utc")
    return value


def _load_index_and_waiver(
    *, recovery_index_path: Path, waiver_path: Path
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    try:
        index, index_file_sha = krea_recovery_evidence.load_index(recovery_index_path)
    except (OSError, ValueError) as exc:
        raise WaiverFreezeError(str(exc)) from exc
    if index["coverage"]["selection_gate_ready"] is not True:
        raise WaiverFreezeError("recovery index is not 92/92 selection-eligible")
    waiver, waiver_file_sha = _load_control(waiver_path, "recovery evidence waiver")
    validate_waiver(
        waiver,
        recovery_index_sha256=index["index_sha256"],
        recovery_index_file_sha256=index_file_sha,
    )
    return index, index_file_sha, waiver, waiver_file_sha


def _analysis(index: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    by_fixture: dict[str, list[dict[str, Any]]] = {
        fixture: [] for fixture in ("D1", "D2")
    }
    final_steps: dict[tuple[str, str], int] = {}
    zeros: dict[str, dict[str, Any]] = {}
    dataset_shas: dict[str, set[str]] = {fixture: set() for fixture in ("D1", "D2")}
    for artifact in index["artifacts"]:
        if artifact["selection_eligible"] is not True:
            raise WaiverFreezeError(
                "selection analysis received an ineligible artifact"
            )
        validated = artifact["validated_artifact"]
        result = validated["result"]
        fixture = artifact["fixture_id"]
        dataset_shas[fixture].add(result["dataset_sha256"])
        row = {
            "candidate_id": artifact["task_id"],
            "candidate_sha256": artifact["expected_candidate_sha256"],
            "step": artifact["step"],
            "image_exposures": None,
            "weighted_loss": Decimal(str(result["weighted_loss"])),
            "mode": (
                "zero_lora_control"
                if artifact["zero_control"]
                else "local_run_candidate"
            ),
            "family_id": artifact["family_id"],
        }
        if artifact["zero_control"]:
            if fixture in zeros:
                raise WaiverFreezeError(f"fixture {fixture} repeats its zero control")
            zeros[fixture] = row
        else:
            by_fixture[fixture].append(row)
            if artifact["is_final"]:
                key = (fixture, artifact["family_id"])
                if key in final_steps:
                    raise WaiverFreezeError(f"cell {key} repeats its final candidate")
                final_steps[key] = artifact["step"]
    if any(len(shas) != 1 for shas in dataset_shas.values()):
        raise WaiverFreezeError("fixture score rows bind inconsistent datasets")
    expected_cells = {
        (fixture, family)
        for fixture in ("D1", "D2")
        for family in krea_recovery_evidence.FAMILIES
    }
    if set(final_steps) != expected_cells or set(zeros) != {"D1", "D2"}:
        raise WaiverFreezeError(
            "recovery index lacks exact cell finals or zero controls"
        )

    analyses: dict[tuple[str, str], dict[str, Any]] = {}
    for fixture in ("D1", "D2"):
        candidates = []
        for row in by_fixture[fixture]:
            denominator = final_steps[(fixture, row["family_id"])]
            if row["step"] > denominator:
                raise WaiverFreezeError("candidate step exceeds its natural final")
            candidates.append(
                {
                    **row,
                    "fraction_numerator": row["step"],
                    "fraction_denominator": denominator,
                }
            )
        aggregate = {"candidates": candidates, "zero": zeros[fixture]}
        try:
            curves = krea_decision._curves(
                aggregate, expected_arm_ids=krea_recovery_evidence.FAMILIES
            )
        except (KeyError, ValueError) as exc:
            raise WaiverFreezeError(str(exc)) from exc
        analyses[(fixture, "A")] = {
            "batch_id": f"recovery-waiver-{fixture}-A",
            "curves": curves,
            "aggregate": aggregate,
        }
    return analyses


def _derive_selection(index: Mapping[str, Any]) -> dict[str, Any]:
    analyses = _analysis(index)
    families = krea_recovery_evidence.FAMILIES
    try:
        scores = krea_decision._concept_family_scores(
            analyses,
            fixtures=("D1", "D2"),
            seed_roles=("A",),
            family_ids=families,
        )
    except (KeyError, ValueError) as exc:
        raise WaiverFreezeError(str(exc)) from exc
    inside_by_fixture = {}
    for fixture in ("D1", "D2"):
        best = max(scores[family][fixture] for family in NONCONTROL_FAMILIES)
        inside_by_fixture[fixture] = sorted(
            family
            for family in NONCONTROL_FAMILIES
            if best - scores[family][fixture] <= TIE_BAND
        )
    inside = sorted(set().union(*inside_by_fixture.values()))
    reversals = krea_decision._material_rank_reversal(
        scores, noncontrols=NONCONTROL_FAMILIES
    )
    reasons = []
    if any(len(families_inside) >= 3 for families_inside in inside_by_fixture.values()):
        reasons.append("three_or_more_noncontrols_inside_0.01_band")
    if reversals:
        reasons.append("material_D1_D2_rank_reversal")
    trigger = {
        "triggered": bool(reasons),
        "reasons": reasons,
        "noncontrols_inside_band": inside,
        "noncontrols_inside_band_by_fixture": inside_by_fixture,
        "material_reversals": reversals,
        "seed_b_evidence_present": False,
        "waiver_cannot_substitute_for_seed_b": True,
    }
    curve_results = {
        fixture: {family: analysis["curves"][family]["curve"] for family in families}
        for (fixture, role), analysis in analyses.items()
        if role == "A"
    }
    selected_scores = {
        family: {fixture: float(scores[family][fixture]) for fixture in ("D1", "D2")}
        for family in families
    }
    base = {
        "selection_algorithm": {
            "relative_loss_formula": "(L_control-L_candidate)/L_control",
            "control": "independent zero-LoRA exact score per D fixture",
            "seed_roles_used": ["A"],
            "tie_band": float(TIE_BAND),
            "seed_b_trigger": (
                "three or more non-controls inside tie band or material rank reversal"
            ),
            "checkpoint_tie_breaker": (
                "earliest actual step among candidates within 0.01 of best"
            ),
            "maximum_noncontrol_finalists": MAX_NONCONTROL_FINALISTS,
        },
        "seed_b_trigger": trigger,
        "curve_results": curve_results,
        "selected_relative_improvements": selected_scores,
    }
    if trigger["triggered"]:
        return {
            **base,
            "outcome": "seed_b_required",
            "blockers": ["predeclared independent training Seed-B evidence is absent"],
            "D1_winner_family_id": None,
            "D2_winner_family_id": None,
            "minimax_regret": {},
            "finalist_family_ids": [],
            "checkpoint_rules": {},
            "all_family_checkpoint_rules": {},
        }

    winners = {
        fixture: min(
            NONCONTROL_FAMILIES,
            key=lambda family: (-scores[family][fixture], family),
        )
        for fixture in ("D1", "D2")
    }
    best_by_fixture = {
        fixture: max(scores[family][fixture] for family in NONCONTROL_FAMILIES)
        for fixture in ("D1", "D2")
    }
    regret = {
        family: max(
            best_by_fixture[fixture] - scores[family][fixture]
            for fixture in ("D1", "D2")
        )
        for family in NONCONTROL_FAMILIES
    }
    finalists: list[str] = []

    def add(family: str) -> None:
        if family not in finalists:
            finalists.append(family)

    add(winners["D1"])
    add(winners["D2"])
    remaining = [family for family in NONCONTROL_FAMILIES if family not in finalists]
    if remaining:
        add(min(remaining, key=lambda family: (regret[family], family)))
    if len(finalists) > MAX_NONCONTROL_FINALISTS:
        raise WaiverFreezeError("selection exceeded three non-control finalists")
    add(CONTROL_FAMILY)
    try:
        all_rules = {
            family: krea_decision._checkpoint_rule(
                family,
                analyses=analyses,
                fixtures=("D1", "D2"),
                seed_roles=("A",),
                targets=REPORT_TARGETS,
            )
            for family in families
        }
    except (KeyError, ValueError) as exc:
        raise WaiverFreezeError(str(exc)) from exc
    return {
        **base,
        "outcome": "finalists_frozen",
        "blockers": [],
        "D1_winner_family_id": winners["D1"],
        "D2_winner_family_id": winners["D2"],
        "minimax_regret": {
            family: float(value) for family, value in sorted(regret.items())
        },
        "finalist_family_ids": finalists,
        "checkpoint_rules": {family: all_rules[family] for family in finalists},
        "all_family_checkpoint_rules": all_rules,
    }


def _binding(path: Path, *, semantic_key: str, semantic_sha256: str) -> dict[str, Any]:
    path = krea_recovery_evidence._safe_file(path, "freeze input")
    return {
        "path": str(path),
        "file_sha256": krea_provenance.file_sha256(path),
        semantic_key: semantic_sha256,
    }


def _freeze_body(
    *,
    index: Mapping[str, Any],
    index_path: Path,
    waiver: Mapping[str, Any],
    waiver_path: Path,
    preparer_actor: Mapping[str, Any],
    frozen_at_utc: str,
) -> dict[str, Any]:
    if _timestamp_value(frozen_at_utc, "frozen_at_utc") <= _timestamp_value(
        waiver["approved_at_utc"], "waiver approved_at_utc"
    ):
        raise WaiverFreezeError("finalist freeze must postdate the owner waiver")
    return {
        "schema": SCHEMA,
        "kind": FREEZE_KIND,
        "frozen_at_utc": _timestamp(frozen_at_utc, "frozen_at_utc"),
        "recovery_index": _binding(
            index_path,
            semantic_key="index_sha256",
            semantic_sha256=index["index_sha256"],
        ),
        "waiver": _binding(
            waiver_path,
            semantic_key="waiver_sha256",
            semantic_sha256=waiver["waiver_sha256"],
        ),
        "preparer_actor": dict(preparer_actor),
        "agent_review_required": True,
        "agent_review_is_not_human_review": True,
        "claims": dict(FALSE_CLAIMS),
        "authority": dict(AUTHORITY),
        **_derive_selection(index),
    }


def _publish(path: Path, value: dict[str, Any], label: str) -> None:
    try:
        krea_recovery_evidence._publish(path, value)
    except krea_recovery_evidence.RecoveryEvidenceError as exc:
        raise WaiverFreezeError(f"{label}: {exc}") from exc


def freeze_finalists(
    *,
    recovery_index_path: Path,
    waiver_path: Path,
    preparer_actor_path: Path,
    output: Path,
    frozen_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create one immutable, non-authorizing finalist/Seed-B outcome."""

    index, _, waiver, _ = _load_index_and_waiver(
        recovery_index_path=recovery_index_path, waiver_path=waiver_path
    )
    actor_value, _ = _load_control(preparer_actor_path, "freeze preparer actor")
    preparer = _fresh_actor(
        actor_value, role=PREPARER_ROLE, label="freeze preparer actor"
    )
    body = _freeze_body(
        index=index,
        index_path=recovery_index_path,
        waiver=waiver,
        waiver_path=waiver_path,
        preparer_actor=preparer,
        frozen_at_utc=frozen_at_utc
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    value = {**body, "freeze_sha256": krea_provenance.canonical_sha256(body)}
    _publish(output, value, "finalist freeze")
    return value


def _load_freeze(
    *, recovery_index_path: Path, waiver_path: Path, freeze_path: Path
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    index, _, waiver, _ = _load_index_and_waiver(
        recovery_index_path=recovery_index_path, waiver_path=waiver_path
    )
    value, file_sha = _load_control(freeze_path, "recovery finalist freeze")
    if not isinstance(value, dict):
        raise WaiverFreezeError("recovery finalist freeze must be an object")
    body = {key: item for key, item in value.items() if key != "freeze_sha256"}
    if (
        value.get("schema") != SCHEMA
        or value.get("kind") != FREEZE_KIND
        or value.get("freeze_sha256") != krea_provenance.canonical_sha256(body)
        or value.get("claims") != FALSE_CLAIMS
        or value.get("authority") != AUTHORITY
        or value.get("agent_review_required") is not True
        or value.get("agent_review_is_not_human_review") is not True
    ):
        raise WaiverFreezeError("recovery finalist freeze identity/claims drifted")
    preparer = _fresh_actor(
        value.get("preparer_actor"), role=PREPARER_ROLE, label="freeze preparer actor"
    )
    expected = _freeze_body(
        index=index,
        index_path=recovery_index_path,
        waiver=waiver,
        waiver_path=waiver_path,
        preparer_actor=preparer,
        frozen_at_utc=value.get("frozen_at_utc"),
    )
    if body != expected:
        raise WaiverFreezeError("recovery finalist freeze does not recompute exactly")
    finalists = value["finalist_family_ids"]
    noncontrols = [family for family in finalists if family != CONTROL_FAMILY]
    if len(noncontrols) > MAX_NONCONTROL_FINALISTS:
        raise WaiverFreezeError("freeze exceeds three non-control finalists")
    if value["outcome"] == "finalists_frozen" and CONTROL_FAMILY not in finalists:
        raise WaiverFreezeError("frozen finalists omit K0")
    if value["outcome"] == "seed_b_required" and finalists:
        raise WaiverFreezeError("Seed-B-required outcome carries finalists")
    return value, file_sha, index, waiver


def validate_freeze(
    *, recovery_index_path: Path, waiver_path: Path, freeze_path: Path
) -> dict[str, Any]:
    return _load_freeze(
        recovery_index_path=recovery_index_path,
        waiver_path=waiver_path,
        freeze_path=freeze_path,
    )[0]


def review_finalist_freeze(
    *,
    recovery_index_path: Path,
    waiver_path: Path,
    freeze_path: Path,
    reviewer_actor_path: Path,
    output: Path,
    reviewed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Independently rederive and create-only record one fresh agent review."""

    freeze, freeze_file_sha, index, waiver = _load_freeze(
        recovery_index_path=recovery_index_path,
        waiver_path=waiver_path,
        freeze_path=freeze_path,
    )
    actor_value, _ = _load_control(reviewer_actor_path, "freeze reviewer actor")
    reviewer = _fresh_actor(
        actor_value, role=REVIEWER_ROLE, label="freeze reviewer actor"
    )
    preparer = freeze["preparer_actor"]
    if (
        reviewer["actor_id"] == preparer["actor_id"]
        or reviewer["review_instance_id"] == preparer["review_instance_id"]
    ):
        raise WaiverFreezeError("freeze reviewer is not independent from preparer")
    reviewed_at = reviewed_at_utc or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    if _timestamp_value(reviewed_at, "reviewed_at_utc") <= _timestamp_value(
        freeze["frozen_at_utc"], "frozen_at_utc"
    ):
        raise WaiverFreezeError("freeze review must postdate the freeze")
    # Re-derive after all source artifacts have been rehashed by _load_freeze.
    derived = _derive_selection(index)
    reviewed_fields = {
        key: freeze[key]
        for key in (
            "outcome",
            "seed_b_trigger",
            "D1_winner_family_id",
            "D2_winner_family_id",
            "minimax_regret",
            "finalist_family_ids",
            "checkpoint_rules",
            "all_family_checkpoint_rules",
        )
    }
    expected_fields = {key: derived[key] for key in reviewed_fields}
    if reviewed_fields != expected_fields:
        raise WaiverFreezeError("independent freeze rederivation differs")
    body = {
        "schema": SCHEMA,
        "kind": REVIEW_KIND,
        "reviewed_at_utc": _timestamp(reviewed_at, "reviewed_at_utc"),
        "freeze": {
            "path": str(krea_recovery_evidence._safe_file(freeze_path, "freeze")),
            "file_sha256": freeze_file_sha,
            "freeze_sha256": freeze["freeze_sha256"],
        },
        "recovery_index_sha256": index["index_sha256"],
        "waiver_sha256": waiver["waiver_sha256"],
        "reviewer_actor": reviewer,
        "independent_from_preparer": True,
        "reviewer_is_human": False,
        "agent_review_is_not_human_review": True,
        "decision": "verified_exact_recomputation",
        "reviewed_selection_sha256": krea_provenance.canonical_sha256(reviewed_fields),
        "claims": dict(FALSE_CLAIMS),
        "authority": dict(AUTHORITY),
        "deployment_claimed": False,
    }
    value = {**body, "review_sha256": krea_provenance.canonical_sha256(body)}
    _publish(output, value, "finalist freeze agent review")
    return value


def validate_review(
    *,
    recovery_index_path: Path,
    waiver_path: Path,
    freeze_path: Path,
    review_path: Path,
) -> dict[str, Any]:
    freeze, freeze_file_sha, index, waiver = _load_freeze(
        recovery_index_path=recovery_index_path,
        waiver_path=waiver_path,
        freeze_path=freeze_path,
    )
    review, _ = _load_control(review_path, "finalist freeze agent review")
    _exact(
        review,
        {
            "schema",
            "kind",
            "reviewed_at_utc",
            "freeze",
            "recovery_index_sha256",
            "waiver_sha256",
            "reviewer_actor",
            "independent_from_preparer",
            "reviewer_is_human",
            "agent_review_is_not_human_review",
            "decision",
            "reviewed_selection_sha256",
            "claims",
            "authority",
            "deployment_claimed",
            "review_sha256",
        },
        "finalist freeze agent review",
    )
    body = {key: item for key, item in review.items() if key != "review_sha256"}
    reviewer = _fresh_actor(
        review.get("reviewer_actor"), role=REVIEWER_ROLE, label="freeze reviewer actor"
    )
    preparer = freeze["preparer_actor"]
    reviewed_fields = {
        key: freeze[key]
        for key in (
            "outcome",
            "seed_b_trigger",
            "D1_winner_family_id",
            "D2_winner_family_id",
            "minimax_regret",
            "finalist_family_ids",
            "checkpoint_rules",
            "all_family_checkpoint_rules",
        )
    }
    if (
        review.get("schema") != SCHEMA
        or review.get("kind") != REVIEW_KIND
        or review.get("review_sha256") != krea_provenance.canonical_sha256(body)
        or review.get("claims") != FALSE_CLAIMS
        or review.get("authority") != AUTHORITY
        or review.get("deployment_claimed") is not False
        or review.get("reviewer_is_human") is not False
        or review.get("agent_review_is_not_human_review") is not True
        or review.get("independent_from_preparer") is not True
        or review.get("decision") != "verified_exact_recomputation"
        or review.get("recovery_index_sha256") != index["index_sha256"]
        or review.get("waiver_sha256") != waiver["waiver_sha256"]
        or review.get("freeze")
        != {
            "path": str(krea_recovery_evidence._safe_file(freeze_path, "freeze")),
            "file_sha256": freeze_file_sha,
            "freeze_sha256": freeze["freeze_sha256"],
        }
        or review.get("reviewed_selection_sha256")
        != krea_provenance.canonical_sha256(reviewed_fields)
        or reviewer["actor_id"] == preparer["actor_id"]
        or reviewer["review_instance_id"] == preparer["review_instance_id"]
        or _timestamp_value(review.get("reviewed_at_utc"), "reviewed_at_utc")
        <= _timestamp_value(freeze["frozen_at_utc"], "frozen_at_utc")
    ):
        raise WaiverFreezeError("finalist freeze agent review drifted")
    return review


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--recovery-index", required=True, type=Path)
    freeze.add_argument("--waiver", required=True, type=Path)
    freeze.add_argument("--preparer-actor", required=True, type=Path)
    freeze.add_argument("--frozen-at-utc")
    freeze.add_argument("--output", required=True, type=Path)
    review = commands.add_parser("review")
    review.add_argument("--recovery-index", required=True, type=Path)
    review.add_argument("--waiver", required=True, type=Path)
    review.add_argument("--freeze", required=True, type=Path)
    review.add_argument("--reviewer-actor", required=True, type=Path)
    review.add_argument("--reviewed-at-utc")
    review.add_argument("--output", required=True, type=Path)
    validate_freeze_parser = commands.add_parser("validate-freeze")
    validate_freeze_parser.add_argument("--recovery-index", required=True, type=Path)
    validate_freeze_parser.add_argument("--waiver", required=True, type=Path)
    validate_freeze_parser.add_argument("--freeze", required=True, type=Path)
    validate_review_parser = commands.add_parser("validate-review")
    validate_review_parser.add_argument("--recovery-index", required=True, type=Path)
    validate_review_parser.add_argument("--waiver", required=True, type=Path)
    validate_review_parser.add_argument("--freeze", required=True, type=Path)
    validate_review_parser.add_argument("--review", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        value = freeze_finalists(
            recovery_index_path=args.recovery_index,
            waiver_path=args.waiver,
            preparer_actor_path=args.preparer_actor,
            output=args.output,
            frozen_at_utc=args.frozen_at_utc,
        )
        digest = value["freeze_sha256"]
    elif args.command == "review":
        value = review_finalist_freeze(
            recovery_index_path=args.recovery_index,
            waiver_path=args.waiver,
            freeze_path=args.freeze,
            reviewer_actor_path=args.reviewer_actor,
            output=args.output,
            reviewed_at_utc=args.reviewed_at_utc,
        )
        digest = value["review_sha256"]
    elif args.command == "validate-freeze":
        value = validate_freeze(
            recovery_index_path=args.recovery_index,
            waiver_path=args.waiver,
            freeze_path=args.freeze,
        )
        digest = value["freeze_sha256"]
    else:
        value = validate_review(
            recovery_index_path=args.recovery_index,
            waiver_path=args.waiver,
            freeze_path=args.freeze,
            review_path=args.review,
        )
        digest = value["review_sha256"]
    print(digest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
