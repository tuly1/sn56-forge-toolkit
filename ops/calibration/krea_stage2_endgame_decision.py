#!/usr/bin/env python3
"""Recompute the frozen two-policy Stage-2 confirmation decision.

This consumer understands the endgame's K1-for-D1 / K5-for-D2 freeze.  It
strictly replays the exact-60 training gate and the 16-group/80-receipt score
gate, applies the predeclared confirmation tests independently to K1 and K5,
and reports boundary mechanics for each.  It intentionally cannot invent a
dataset-count router or authorize release/deployment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Sequence

try:
    from . import krea_provenance
    from . import krea_stage2_decision as legacy_decision
    from . import krea_stage2_endgame_matrix
    from . import krea_stage2_endgame_orchestrator as training
    from . import krea_stage2_endgame_scoring as scoring
    from . import krea_stage2_execution
    from . import krea_stage2_score
    from . import krea_stage2_training_evidence
except ImportError:  # pragma: no cover - direct CLI execution.
    import krea_provenance  # type: ignore[no-redef]
    import krea_stage2_decision as legacy_decision  # type: ignore[no-redef]
    import krea_stage2_endgame_matrix  # type: ignore[no-redef]
    import krea_stage2_endgame_orchestrator as training  # type: ignore[no-redef]
    import krea_stage2_endgame_scoring as scoring  # type: ignore[no-redef]
    import krea_stage2_execution  # type: ignore[no-redef]
    import krea_stage2_score  # type: ignore[no-redef]
    import krea_stage2_training_evidence  # type: ignore[no-redef]


SCHEMA = 1
KIND = "forge-krea-stage2-endgame-two-policy-decision"
ACTIVE_POLICIES = ("K1", "K5")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(
            f"{label} keys differ: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )


def _load(path: str | Path, label: str) -> dict[str, Any]:
    return krea_stage2_endgame_matrix._load_canonical(path, label)


def _file_sha(value: Mapping[str, Any]) -> str:
    return scoring._file_sha(value)


def _load_scored_groups(
    queue: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    list[dict[str, Any]],
    str,
]:
    by_policy: dict[str, dict[str, dict[str, Any]]] = {
        family: {} for family in ACTIVE_POLICIES
    }
    bindings = []
    latest = ""
    for queued in queue["groups"]:
        group = scoring._validate_group(
            _load(queued["group_path"], "decision score group"), score_queue=queue
        )
        plan = krea_stage2_score.validate_plan(
            _load(group["plan_path"], "decision score plan")
        )
        aggregate = krea_stage2_score.validate_aggregate(
            _load(group["aggregate_path"], "decision score aggregate"), plan=plan
        )
        family = group["candidate_family_id"]
        if family not in by_policy or group["cell_id"] in by_policy[family]:
            raise ValueError("decision score groups duplicate or add a policy cell")
        by_policy[family][group["cell_id"]] = aggregate
        latest = max(latest, aggregate["emitted_at_utc"])
        bindings.append(
            {
                "group_key": group["group_key"],
                "candidate_family_id": family,
                "cell_id": group["cell_id"],
                "plan_sha256": plan["plan_sha256"],
                "aggregate_file_sha256": _file_sha(aggregate),
                "aggregate_sha256": aggregate["aggregate_sha256"],
            }
        )
    expected = {
        f"{fixture}-{seed}"
        for fixture in krea_stage2_endgame_matrix.CONFIRMATION_FIXTURES
        for seed in krea_stage2_endgame_matrix.SEED_ROLES
    }
    if any(set(cells) != expected for cells in by_policy.values()):
        raise ValueError("decision lacks the exact eight score cells per policy")
    return by_policy, bindings, latest


def _boundary_by_policy(
    *, plan_set: Mapping[str, Any], matrix: Mapping[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    matrix_rows = {row["row_key"]: row for row in matrix["rows"]}
    result: dict[str, dict[str, dict[str, Any]]] = {
        family: {} for family in ACTIVE_POLICIES
    }
    for row in plan_set["rows"]:
        source = matrix_rows[row["row_key"]]
        if source["phase"] != "boundary":
            continue
        family = source["family_id"]
        if family not in result or source["cell_id"] in result[family]:
            raise ValueError("boundary evidence duplicates or adds a policy cell")
        plan = krea_stage2_execution.validate_plan(
            _load(row["plan"]["path"], "boundary execution plan")
        )
        approval = krea_stage2_execution.validate_approval(
            _load(row["approval"]["path"], "boundary execution approval"), plan=plan
        )
        completion = krea_stage2_execution.validate_completion(
            _load(row["completion_path"], "boundary run completion"),
            plan=plan,
            approval=approval,
        )
        evidence = krea_stage2_training_evidence.validate_run_evidence(
            _load(row["run_evidence_path"], "boundary run evidence"),
            plan=plan,
            approval=approval,
            completion=completion,
        )
        candidates = [
            item
            for item in evidence["candidate_artifacts"]
            if PurePosixPath(item["path"]).name == "last.safetensors"
        ]
        if len(candidates) != 1:
            raise ValueError("boundary evidence lacks one promoted last.safetensors")
        result[family][source["cell_id"]] = {
            "hours": plan["hours"],
            "candidate_sha256": candidates[0]["sha256"],
            "mechanics": evidence["mechanics"],
            "run_evidence_file_sha256": _file_sha(evidence),
            "run_evidence_sha256": evidence["evidence_sha256"],
        }
    if any(
        set(cells) != set(krea_stage2_endgame_matrix.BOUNDARY_CELLS)
        for cells in result.values()
    ):
        raise ValueError("decision lacks the exact six boundary cells per policy")
    return result


def _compute_policy(
    *,
    family: str,
    aggregates: Mapping[str, dict[str, Any]],
    boundary: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    plans: dict[str, dict[str, Any]] = {cell: {} for cell in aggregates}
    for cell, row in boundary.items():
        plans[cell] = {
            "hours": row["hours"],
            "candidates": [
                {
                    "family_id": family,
                    "candidate_sha256": row["candidate_sha256"],
                    "mechanics": row["mechanics"],
                }
            ],
        }
    metrics, boundary_results, gates = legacy_decision._compute(
        plans,
        aggregates,
        candidate_family=family,
    )
    passed = all(gates.values())
    evidence_bindings = {cell: boundary[cell] for cell in sorted(boundary)}
    return {
        "candidate_family_id": family,
        "outcome": "PASS" if passed else "FAIL",
        "blockers": [
            f"failed confirmation gate: {name}"
            for name, result in gates.items()
            if not result
        ],
        "metrics": metrics,
        "gates": gates,
        "boundary_results": boundary_results,
        "boundary_evidence_bindings": evidence_bindings,
        "confirmation_passed": passed,
    }


def _decision_body(
    *,
    matrix: Mapping[str, Any],
    plan_set: Mapping[str, Any],
    authority: Mapping[str, Any],
    training_gate: Mapping[str, Any],
    score_queue: Mapping[str, Any],
    score_gate: Mapping[str, Any],
    decided_at_utc: str,
) -> dict[str, Any]:
    scored, aggregate_bindings, latest_aggregate = _load_scored_groups(score_queue)
    boundaries = _boundary_by_policy(plan_set=plan_set, matrix=matrix)
    policies = {
        family: _compute_policy(
            family=family,
            aggregates=scored[family],
            boundary=boundaries[family],
        )
        for family in ACTIVE_POLICIES
    }
    decided = krea_stage2_execution._utc(decided_at_utc, "endgame decision time")
    if decided <= max(
        latest_aggregate,
        training_gate["completed_at_utc"],
        score_gate["completed_at_utc"],
    ):
        raise ValueError("endgame decision must postdate both gates and all scores")
    overall = all(row["confirmation_passed"] for row in policies.values())
    freeze = authority["waiver_finalist_freeze"]
    frozen_policy = {
        "D1": {
            "candidate_family_id": "K1",
            "checkpoint_target_fraction": freeze["all_family_checkpoint_rules"][
                "K1"
            ]["target_fraction"],
        },
        "D2": {
            "candidate_family_id": "K5",
            "checkpoint_target_fraction": freeze["all_family_checkpoint_rules"][
                "K5"
            ]["target_fraction"],
        },
    }
    if (
        freeze["D1_winner_family_id"] != "K1"
        or freeze["D2_winner_family_id"] != "K5"
        or matrix["active_variant_family_ids"] != ["K1", "K5"]
    ):
        raise ValueError("endgame decision authority differs from frozen K1/K5 policy")
    blockers = [
        f"{family}: {blocker}"
        for family in ACTIVE_POLICIES
        for blocker in policies[family]["blockers"]
    ]
    return {
        "schema": SCHEMA,
        "kind": KIND,
        "phase": "confirmation",
        "decided_at_utc": decided,
        "matrix_sha256": matrix["matrix_sha256"],
        "training_plan_set_sha256": plan_set["plan_set_sha256"],
        "score_queue_sha256": score_queue["score_queue_sha256"],
        "training_gate": {
            "file_sha256": _file_sha(training_gate),
            "gate_sha256": training_gate["gate_sha256"],
        },
        "score_gate": {
            "file_sha256": _file_sha(score_gate),
            "gate_sha256": score_gate["gate_sha256"],
        },
        "authority": {
            "freeze_file_sha256": matrix["freeze"]["file_sha256"],
            "freeze_sha256": matrix["freeze"]["freeze_sha256"],
            "production_identity_file_sha256": matrix["production_identity"][
                "file_sha256"
            ],
            "production_identity_sha256": matrix["production_identity"][
                "production_identity_sha256"
            ],
            "production_image_id": matrix["production_image_id"],
        },
        "frozen_dataset_regime_policy": frozen_policy,
        "policy_results": policies,
        "aggregate_bindings": aggregate_bindings,
        "outcome": "PASS" if overall else "FAIL",
        "blockers": blockers,
        "overall_confirmation_passed": overall,
        "surprise_review_required": not overall,
        "production_dataset_count_router_predeclared": False,
        "production_routing_authority": False,
        "release_family_selected": None,
        "release_review_required": True,
        "production_mutation_authorized": False,
        "release_authorized": False,
        "deployment_authorized": False,
        "win_guaranteed": False,
    }


def build_decision(
    *,
    matrix: Mapping[str, Any],
    plan_set: Mapping[str, Any],
    authority_bundle: Mapping[str, Any],
    training_gate_path: str | Path,
    score_queue: Mapping[str, Any],
    score_gate_path: str | Path,
    decided_at_utc: str,
) -> dict[str, Any]:
    resolved_matrix = krea_stage2_endgame_matrix.validate_matrix(matrix)
    resolved_plan_set = training.validate_plan_set(plan_set, matrix=resolved_matrix)
    authority = training._validate_authority_bundle(authority_bundle)
    resolved_matrix = krea_stage2_endgame_matrix.validate_matrix(
        resolved_matrix,
        freeze=authority["waiver_finalist_freeze"],
        production_identity=authority["production_identity"],
    )
    queue = scoring._validate_queue(score_queue)
    if (
        queue["matrix_sha256"] != resolved_matrix["matrix_sha256"]
        or queue["training_plan_set_sha256"] != resolved_plan_set["plan_set_sha256"]
    ):
        raise ValueError("decision score queue differs from the training graph")
    if not os.path.lexists(training_gate_path) or not os.path.lexists(score_gate_path):
        raise ValueError("decision requires both pre-existing completion gates")
    existing_training_gate = _load(training_gate_path, "training completion gate")
    training_gate = training.seal_exact60_gate(
        plan_set=resolved_plan_set,
        matrix=resolved_matrix,
        authority_bundle=authority,
        output=training_gate_path,
        completed_at_utc=existing_training_gate["completed_at_utc"],
    )
    existing_score_gate = _load(score_gate_path, "score completion gate")
    score_gate = scoring.seal_score_gate(
        score_queue=queue,
        output=score_gate_path,
        completed_at_utc=existing_score_gate["completed_at_utc"],
    )
    body = _decision_body(
        matrix=resolved_matrix,
        plan_set=resolved_plan_set,
        authority=authority,
        training_gate=training_gate,
        score_queue=queue,
        score_gate=score_gate,
        decided_at_utc=decided_at_utc,
    )
    return {**body, "decision_sha256": krea_provenance.canonical_sha256(body)}


def publish_decision(value: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    record = _object(value, "endgame decision")
    body = {key: item for key, item in record.items() if key != "decision_sha256"}
    if (
        record.get("schema") != SCHEMA
        or record.get("kind") != KIND
        or record.get("decision_sha256") != krea_provenance.canonical_sha256(body)
        or record.get("release_review_required") is not True
        or record.get("production_dataset_count_router_predeclared") is not False
        or record.get("production_routing_authority") is not False
        or record.get("release_family_selected") is not None
        or any(
            record.get(field) is not False
            for field in (
                "production_mutation_authorized",
                "release_authorized",
                "deployment_authorized",
                "win_guaranteed",
            )
        )
    ):
        raise ValueError("endgame decision identity/authority differs")
    return krea_stage2_endgame_matrix._publish_or_replay(
        output, record, "endgame decision"
    )


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--plan-set", required=True, type=Path)
    parser.add_argument("--authority-bundle", required=True, type=Path)
    parser.add_argument("--training-gate", required=True, type=Path)
    parser.add_argument("--score-queue", required=True, type=Path)
    parser.add_argument("--score-gate", required=True, type=Path)
    parser.add_argument("--decided-at-utc", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    try:
        result = build_decision(
            matrix=_load(args.matrix, "decision matrix"),
            plan_set=_load(args.plan_set, "decision plan set"),
            authority_bundle=_load(args.authority_bundle, "decision authority"),
            training_gate_path=args.training_gate,
            score_queue=_load(args.score_queue, "decision score queue"),
            score_gate_path=args.score_gate,
            decided_at_utc=args.decided_at_utc,
        )
        result = publish_decision(result, args.output)
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
