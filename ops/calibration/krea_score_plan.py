#!/usr/bin/env python3
"""Build, approve, and validate exhaustive schema-2 Krea exact-score plans.

``build`` and ``dry-run`` consume immutable stage-three run-evidence bundles,
one zero-control manifest, the reviewed fixture surface, and an evaluator
configuration.  They never create a human approval.  ``approve`` is a separate
named-human action which publishes both the approval artifact and an executable
plan.  ``validate`` replays either the draft's structural checks or the complete
batch validator for an approved plan.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

try:
    from . import batch_evaluate_krea as batch
    from . import krea_fixture
    from . import krea_provenance
    from . import krea_training_evidence
except ImportError:  # pragma: no cover - direct script execution.
    import batch_evaluate_krea as batch  # type: ignore[no-redef]
    import krea_fixture  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_training_evidence  # type: ignore[no-redef]


_DRAFT_KEYS = {
    "schema",
    "kind",
    "dataset",
    "fixture_manifest",
    "fixture_approval",
    "cross_fixture_review",
    "campaign_manifest",
    "decision_context",
    "candidates",
    "evaluator",
}


def _canonical_file(path: Path, value: dict[str, Any]) -> str:
    payload = krea_provenance.canonical_bytes(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _load(path: Path | str, label: str) -> tuple[Path, dict[str, Any], str]:
    safe = batch._safe_file(path, label)
    value, digest, raw = batch._load_json_file(safe, label)
    batch._canonical_control_file(value, raw, label)
    return safe, value, digest


def _binding(path: Path, digest: str) -> dict[str, str]:
    return {"path": str(path), "sha256": digest}


def _bound_document(value: Any, label: str) -> tuple[Path, dict[str, Any], str]:
    reference = batch._object(value, f"{label} binding")
    batch._exact_keys(reference, {"path", "sha256"}, f"{label} binding")
    path, document, file_sha = _load(reference["path"], label)
    if reference != _binding(path, file_sha):
        raise ValueError(f"{label} file binding is not exact")
    return path, document, file_sha


def _bundle_candidates(
    bundle_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], str, str]:
    path, bundle, _ = _load(bundle_path, "stage-three run-evidence bundle")
    validated_bundle = krea_training_evidence.validate_run_evidence(path)
    if validated_bundle != bundle:
        raise ValueError(
            "stage-three validator result differs from the consumed bundle"
        )
    # The strong producer-side validator recomputes the approved execution,
    # run record, exhaustive grid, and safetensors bytes.  The projection below
    # is retained only to normalize that validated evidence into score-plan
    # rows; it is not an alternate/weaker admission path.
    bundle = validated_bundle
    batch._exact_keys(
        bundle,
        {
            "schema",
            "kind",
            "arm_id",
            "execution_plan_sha256",
            "run_completion",
            "candidate_bindings",
            "bundle_sha256",
        },
        "stage-three run-evidence bundle",
    )
    body = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    if (
        bundle["schema"] != 2
        or bundle["kind"] != "forge-krea-run-evidence-bundle"
        or bundle["bundle_sha256"] != krea_provenance.canonical_sha256(body)
        or not isinstance(bundle["arm_id"], str)
        or not batch._SAFE_ID.fullmatch(bundle["arm_id"])
    ):
        raise ValueError(f"invalid run-evidence bundle: {path}")

    completion_binding = batch._object(
        bundle["run_completion"], "bundle run-completion binding"
    )
    batch._exact_keys(
        completion_binding,
        {"path", "sha256"},
        "bundle run-completion binding",
    )
    completion_path, completion, completion_file_sha = _load(
        completion_binding["path"], "run completion"
    )
    if completion_binding != _binding(completion_path, completion_file_sha):
        raise ValueError("bundle run-completion binding is not exact")
    if (
        completion.get("schema") != 3
        or completion.get("kind") != "forge-krea-training-completion"
        or completion.get("arm_id") != bundle["arm_id"]
        or completion.get("execution_plan_sha256") != bundle["execution_plan_sha256"]
        or completion.get("natural_completion") is not True
        or completion.get("in_task_proxy_selection")
        != {"enabled": False, "reserve_s": 0}
    ):
        raise ValueError("bundle does not bind a natural stage-three completion")
    completion_candidates = completion.get("candidates")
    raw_bindings = bundle["candidate_bindings"]
    if (
        not isinstance(completion_candidates, list)
        or not completion_candidates
        or not isinstance(raw_bindings, list)
        or len(raw_bindings) != len(completion_candidates)
    ):
        raise ValueError("bundle candidate coverage is incomplete")

    plan_rows: list[dict[str, Any]] = []
    campaign_candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    execution_plan_file_sha: str | None = None
    discovery_plan_file_sha: str | None = None
    evaluation_dataset_sha: str | None = None
    for index, (raw_row, completion_candidate) in enumerate(
        zip(raw_bindings, completion_candidates, strict=True)
    ):
        label = f"candidate_bindings[{index}]"
        row = batch._object(raw_row, label)
        batch._exact_keys(row, {"candidate_id", "candidate_sha256", "binding"}, label)
        row_binding = batch._object(row["binding"], f"{label}.binding reference")
        batch._exact_keys(row_binding, {"path", "sha256"}, f"{label}.binding reference")
        binding_path, binding, binding_file_sha = _load(
            row_binding["path"], f"{label}.binding"
        )
        if row_binding != _binding(binding_path, binding_file_sha):
            raise ValueError(f"{label} file binding is not exact")
        batch._exact_keys(
            binding,
            {
                "schema",
                "kind",
                "mode",
                "arm_id",
                "candidate_id",
                "candidate",
                "execution_plan",
                "execution_approval",
                "run_completion",
                "run_record",
                "training_log",
                "evaluation_dataset_sha256",
            },
            f"{label}.binding",
        )
        candidate = batch._object(binding.get("candidate"), f"{label}.candidate")
        batch._exact_keys(
            candidate,
            {
                "path",
                "sha256",
                "bytes",
                "step",
                "fraction_numerator",
                "fraction_denominator",
                "aliases",
                "safetensors",
            },
            f"{label}.candidate",
        )
        candidate_id = row["candidate_id"]
        digest = row["candidate_sha256"]
        if (
            binding.get("schema") != 2
            or binding.get("kind") != "forge-krea-local-candidate-binding"
            or binding.get("mode") != "local_run_candidate"
            or binding.get("arm_id") != bundle["arm_id"]
            or binding.get("candidate_id") != candidate_id
            or candidate.get("sha256") != digest
            or binding.get("run_completion") != completion_binding
            or candidate_id in seen_ids
            or digest in seen_hashes
        ):
            raise ValueError(f"{label} identity is inconsistent")
        artifact = batch._safe_file(candidate.get("path"), f"{label} artifact")
        if krea_provenance.file_sha256(
            artifact
        ) != digest or artifact.stat().st_size != candidate.get("bytes"):
            raise ValueError(f"{label} artifact bytes differ from the bundle")

        completion_candidate = batch._object(
            completion_candidate, f"run completion candidate[{index}]"
        )
        projected = {
            "candidate_id": candidate_id,
            "sha256": digest,
            "bytes": candidate["bytes"],
            "step": candidate["step"],
            "fraction": {
                "numerator": candidate["fraction_numerator"],
                "denominator": candidate["fraction_denominator"],
            },
        }
        expected_completion = {
            "candidate_id": completion_candidate.get("candidate_id"),
            "sha256": completion_candidate.get("sha256"),
            "bytes": completion_candidate.get("bytes"),
            "step": completion_candidate.get("step"),
            "fraction": {
                "numerator": completion_candidate.get("fraction_numerator"),
                "denominator": completion_candidate.get("fraction_denominator"),
            },
        }
        if projected != expected_completion:
            raise ValueError("bundle binding differs from its completion candidate")

        execution_binding = batch._object(
            binding["execution_plan"], f"{label}.execution_plan reference"
        )
        batch._exact_keys(
            execution_binding,
            {"path", "sha256"},
            f"{label}.execution_plan reference",
        )
        execution_path, execution, current_plan_file_sha = _load(
            execution_binding["path"], f"{label}.execution_plan"
        )
        if (
            execution_binding != _binding(execution_path, current_plan_file_sha)
            or execution.get("plan_sha256") != bundle["execution_plan_sha256"]
        ):
            raise ValueError("candidate execution-plan binding is inconsistent")
        discovery_binding = batch._object(
            execution.get("discovery_plan"), "execution discovery-plan binding"
        )
        batch._exact_keys(
            discovery_binding,
            {"path", "sha256"},
            "execution discovery-plan binding",
        )
        discovery_path, _discovery, current_discovery_file_sha = _load(
            discovery_binding["path"], "frozen discovery plan"
        )
        if discovery_binding != _binding(discovery_path, current_discovery_file_sha):
            raise ValueError("execution discovery-plan binding is not exact")
        current_evaluation_sha = binding["evaluation_dataset_sha256"]
        if not isinstance(current_evaluation_sha, str) or not batch._SHA256.fullmatch(
            current_evaluation_sha
        ):
            raise ValueError("candidate evaluation-dataset SHA-256 is invalid")
        if execution_plan_file_sha is None:
            execution_plan_file_sha = current_plan_file_sha
            discovery_plan_file_sha = current_discovery_file_sha
            evaluation_dataset_sha = current_evaluation_sha
        elif (
            execution_plan_file_sha != current_plan_file_sha
            or discovery_plan_file_sha != current_discovery_file_sha
            or evaluation_dataset_sha != current_evaluation_sha
        ):
            raise ValueError(
                "one run uses inconsistent execution/discovery/dataset bindings"
            )

        seen_ids.add(candidate_id)
        seen_hashes.add(digest)
        campaign_candidates.append(projected)
        plan_rows.append(
            {
                "id": candidate_id,
                "arm_id": bundle["arm_id"],
                "path": str(artifact),
                "sha256": digest,
                "candidate_binding": _binding(binding_path, binding_file_sha),
            }
        )
    campaign_candidates.sort(key=lambda row: (row["step"], row["sha256"]))
    completion_projection = sorted(
        [
            {
                "candidate_id": row["candidate_id"],
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "step": row["step"],
                "fraction": {
                    "numerator": row["fraction_numerator"],
                    "denominator": row["fraction_denominator"],
                },
            }
            for row in completion_candidates
        ],
        key=lambda row: (row["step"], row["sha256"]),
    )
    if campaign_candidates != completion_projection:
        raise ValueError("bundle does not exhaust the run-completion candidate grid")
    assert discovery_plan_file_sha is not None
    assert evaluation_dataset_sha is not None
    return (
        {
            "arm_id": bundle["arm_id"],
            "execution_plan_sha256": bundle["execution_plan_sha256"],
            "run_completion_sha256": completion_file_sha,
            "candidates": campaign_candidates,
        },
        plan_rows,
        completion,
        discovery_plan_file_sha,
        evaluation_dataset_sha,
    )


def _zero_candidate(
    manifest_path: Path, *, evaluation_dataset_sha256: str
) -> tuple[dict[str, Any], str]:
    path, manifest, file_sha = _load(manifest_path, "zero-control manifest")
    artifact = batch._object(manifest.get("artifact"), "zero-control artifact")
    artifact_path = batch._safe_file(artifact.get("path"), "zero-control artifact")
    krea_training_evidence.validate_zero_control(manifest, artifact_path=artifact_path)
    if manifest.get("evaluation_dataset_sha256") != evaluation_dataset_sha256:
        raise ValueError("zero control belongs to another evaluation fixture")
    return (
        {
            "id": "K0-zero-control",
            "arm_id": "K0",
            "path": str(artifact_path),
            "sha256": artifact["sha256"],
            "candidate_binding": _binding(path, file_sha),
        },
        manifest["manifest_sha256"],
    )


def _decision_context(
    *,
    phase: str,
    campaign: dict[str, Any],
    frozen_discovery_decision: Path | None,
    candidate_family: str | None,
) -> dict[str, Any]:
    if phase != "boundary":
        if frozen_discovery_decision is not None or candidate_family is not None:
            raise ValueError("only boundary mode accepts a frozen discovery decision")
        context = {
            "schema": 1,
            "kind": "forge-krea-exact-score-decision-context",
            "phase": phase,
        }
        return batch._validate_score_decision_context(context, campaign=campaign)
    if frozen_discovery_decision is None:
        raise ValueError("boundary mode requires --frozen-discovery-decision")
    if len(campaign["runs"]) != 1 or len(campaign["runs"][0]["candidates"]) != 1:
        raise ValueError("boundary mode requires one exhaustive run/candidate")
    run = campaign["runs"][0]
    family = candidate_family or run["arm_id"]
    decision_path, decision, decision_file_sha = _load(
        frozen_discovery_decision, "frozen discovery decision"
    )
    rule = batch._object(
        batch._object(decision.get("checkpoint_rules"), "checkpoint rules").get(family),
        "frozen checkpoint rule",
    )
    context = {
        "schema": 1,
        "kind": "forge-krea-exact-score-decision-context",
        "phase": "boundary",
        "frozen_discovery_decision": _binding(decision_path, decision_file_sha),
        "candidate_family_id": family,
        "checkpoint_rule_sha256": krea_provenance.canonical_sha256(rule),
        "selected_candidate": run["candidates"][0],
        "decision_completed_before_export_reserve": True,
        "fallback_used": False,
    }
    normalized = batch._validate_score_decision_context(context, campaign=campaign)
    return {key: value for key, value in normalized.items() if not key.startswith("_")}


def build_documents(
    *,
    bundle_paths: Sequence[Path],
    zero_manifest_path: Path,
    dataset_path: Path,
    fixture_manifest_path: Path,
    fixture_approval_path: Path,
    cross_fixture_review_path: Path,
    evaluator_config_path: Path,
    phase: str,
    campaign_output_path: Path,
    frozen_discovery_decision: Path | None = None,
    candidate_family: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct a sealed campaign and unapproved exact-score plan draft."""

    if phase not in {"discovery", "confirmation", "boundary"}:
        raise ValueError("phase must be discovery, confirmation, or boundary")
    if not bundle_paths:
        raise ValueError("at least one stage-three bundle is required")
    campaign_output_path = Path(
        os.path.abspath(os.path.expanduser(campaign_output_path))
    )
    fixture_path, fixture, fixture_file_sha = _load(
        fixture_manifest_path, "fixture manifest"
    )
    krea_fixture.validate_manifest(fixture)
    approval_path, approval, approval_file_sha = _load(
        fixture_approval_path, "fixture approval"
    )
    krea_fixture.validate_approval(approval, fixture_manifest=fixture)
    cross_path, cross_review, cross_file_sha = _load(
        cross_fixture_review_path, "cross-fixture review"
    )
    batch._validate_cross_fixture_review_surface(cross_review, fixture=fixture)
    evaluator_path, evaluator, _ = _load(evaluator_config_path, "evaluator config")
    del evaluator_path
    evaluator = batch._validate_evaluator(evaluator)
    dataset = batch._safe_directory(dataset_path, "evaluation dataset")
    expected_dataset_sha = fixture["evaluation_dataset_identity"]["sha256"]

    runs: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    discovery_shas: set[str] = set()
    arms: set[str] = set()
    candidate_ids: set[str] = set()
    candidate_hashes: set[str] = set()
    for raw_bundle in bundle_paths:
        run, rows, _completion, discovery_sha, evaluation_sha = _bundle_candidates(
            raw_bundle
        )
        if evaluation_sha != expected_dataset_sha:
            raise ValueError(
                f"run bundle for arm {run['arm_id']} belongs to another evaluation fixture"
            )
        if run["arm_id"] in arms:
            raise ValueError(f"duplicate run bundle for arm {run['arm_id']}")
        for row in rows:
            if row["id"] in candidate_ids or row["sha256"] in candidate_hashes:
                raise ValueError("candidate ids/bytes must be unique across all runs")
            candidate_ids.add(row["id"])
            candidate_hashes.add(row["sha256"])
        arms.add(run["arm_id"])
        runs.append(run)
        candidates.extend(rows)
        discovery_shas.add(discovery_sha)
    if len(discovery_shas) != 1:
        raise ValueError("run bundles do not share one frozen discovery plan")
    runs.sort(key=lambda row: row["arm_id"])
    candidates.sort(key=lambda row: row["id"])

    zero, zero_manifest_sha = _zero_candidate(
        zero_manifest_path, evaluation_dataset_sha256=expected_dataset_sha
    )
    if zero["id"] in candidate_ids or zero["sha256"] in candidate_hashes:
        raise ValueError("zero-control id/bytes collide with a local candidate")
    candidates.append(zero)
    candidates.sort(key=lambda row: row["id"])

    campaign_payload = {
        "schema": 2,
        "kind": "forge-krea-exact-score-campaign",
        "fixture_manifest_sha256": fixture["manifest_sha256"],
        "discovery_plan_sha256": next(iter(discovery_shas)),
        "runs": runs,
        "zero_control_manifest_sha256": zero_manifest_sha,
        "decision_contract": batch._DISCOVERY_DECISION_BINDING,
        "confirmation_contract": batch._CONFIRMATION_DECISION_BINDING,
    }
    campaign = batch.seal_campaign_manifest(campaign_payload)
    context = _decision_context(
        phase=phase,
        campaign=campaign,
        frozen_discovery_decision=frozen_discovery_decision,
        candidate_family=candidate_family,
    )
    campaign_file_sha = hashlib.sha256(
        krea_provenance.canonical_bytes(campaign) + b"\n"
    ).hexdigest()
    draft = {
        "schema": 2,
        "kind": "forge-krea-exact-score-plan",
        "dataset": {"path": str(dataset), "sha256": expected_dataset_sha},
        "fixture_manifest": _binding(fixture_path, fixture_file_sha),
        "fixture_approval": _binding(approval_path, approval_file_sha),
        "cross_fixture_review": _binding(cross_path, cross_file_sha),
        "campaign_manifest": _binding(campaign_output_path, campaign_file_sha),
        "decision_context": context,
        "candidates": candidates,
        "evaluator": evaluator,
    }
    validate_draft(draft, campaign=campaign)
    return campaign, draft


def validate_draft(
    draft: dict[str, Any], *, campaign: dict[str, Any] | None = None
) -> dict[str, Any]:
    batch._exact_keys(draft, _DRAFT_KEYS, "exact-score plan draft")
    if draft["schema"] != 2 or draft["kind"] != "forge-krea-exact-score-plan":
        raise ValueError("exact-score plan draft identity is invalid")
    dataset_spec = batch._object(draft["dataset"], "draft dataset")
    batch._exact_keys(dataset_spec, {"path", "sha256"}, "draft dataset")
    batch._safe_directory(dataset_spec["path"], "draft evaluation dataset")
    if not isinstance(dataset_spec["sha256"], str) or not batch._SHA256.fullmatch(
        dataset_spec["sha256"]
    ):
        raise ValueError("draft evaluation-dataset SHA-256 is invalid")
    _fixture_path, fixture, _fixture_file_sha = _bound_document(
        draft["fixture_manifest"], "fixture manifest"
    )
    krea_fixture.validate_manifest(fixture)
    _approval_path, approval, _approval_file_sha = _bound_document(
        draft["fixture_approval"], "fixture approval"
    )
    krea_fixture.validate_approval(approval, fixture_manifest=fixture)
    _cross_path, cross_review, _cross_file_sha = _bound_document(
        draft["cross_fixture_review"], "cross-fixture review"
    )
    batch._validate_cross_fixture_review_surface(cross_review, fixture=fixture)
    if dataset_spec["sha256"] != fixture["evaluation_dataset_identity"]["sha256"]:
        raise ValueError("draft dataset differs from the approved fixture identity")
    evaluator = batch._validate_evaluator(draft["evaluator"])
    del evaluator
    campaign_binding = batch._object(
        draft["campaign_manifest"], "draft campaign binding"
    )
    batch._exact_keys(campaign_binding, {"path", "sha256"}, "draft campaign binding")
    if campaign is None:
        campaign_path, campaign, campaign_file_sha = _load(
            campaign_binding["path"], "campaign manifest"
        )
        expected_campaign_binding = _binding(campaign_path, campaign_file_sha)
    else:
        expected_campaign_binding = {
            "path": str(
                Path(os.path.abspath(os.path.expanduser(campaign_binding["path"])))
            ),
            "sha256": hashlib.sha256(
                krea_provenance.canonical_bytes(campaign) + b"\n"
            ).hexdigest(),
        }
    if campaign_binding != expected_campaign_binding:
        raise ValueError("draft campaign file binding is not exact")
    batch._validate_campaign_manifest(campaign)
    if campaign["fixture_manifest_sha256"] != fixture["manifest_sha256"]:
        raise ValueError("draft campaign belongs to another approved fixture")
    batch._validate_score_decision_context(draft["decision_context"], campaign=campaign)
    raw_candidates = draft["candidates"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("draft candidates are empty")
    ids = []
    hashes = []
    for index, raw in enumerate(raw_candidates):
        row = batch._object(raw, f"draft candidate[{index}]")
        batch._exact_keys(
            row,
            {"id", "arm_id", "path", "sha256", "candidate_binding"},
            f"draft candidate[{index}]",
        )
        if (
            not isinstance(row["id"], str)
            or not batch._SAFE_ID.fullmatch(row["id"])
            or not isinstance(row["arm_id"], str)
            or not batch._SAFE_ID.fullmatch(row["arm_id"])
            or not isinstance(row["sha256"], str)
            or not batch._SHA256.fullmatch(row["sha256"])
        ):
            raise ValueError("draft candidate identity is invalid")
        artifact = batch._safe_file(row["path"], f"draft candidate[{index}]")
        if krea_provenance.file_sha256(artifact) != row["sha256"]:
            raise ValueError("draft candidate bytes changed")
        _binding_path, _binding_document, _binding_file_sha = _bound_document(
            row["candidate_binding"], f"draft candidate[{index}]"
        )
        ids.append(row["id"])
        hashes.append(row["sha256"])
    if (
        ids != sorted(ids)
        or len(ids) != len(set(ids))
        or len(hashes) != len(set(hashes))
    ):
        raise ValueError("draft candidates are duplicate or unsorted")
    return draft


def publish_build(
    *, output_dir: Path, campaign: dict[str, Any], draft: dict[str, Any]
) -> tuple[Path, Path]:
    output = Path(os.path.abspath(os.path.expanduser(output_dir)))
    expected_campaign_path = output / "campaign.json"
    campaign_binding = batch._object(
        draft.get("campaign_manifest"), "draft campaign binding"
    )
    if campaign_binding.get("path") != str(expected_campaign_path):
        raise ValueError(
            "draft campaign path does not name the campaign published with the draft"
        )
    validate_draft(draft, campaign=campaign)
    batch._reject_symlink_ancestors(output.parent, "score-plan output parent")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing existing score-plan output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _canonical_file(temporary / "campaign.json", campaign)
        _canonical_file(temporary / "score-plan.draft.json", draft)
        os.rename(temporary, output)
        parent_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output / "campaign.json", output / "score-plan.draft.json"


def approve_draft(
    *,
    draft_path: Path,
    reviewer_identity: str,
    approval_output: Path,
    plan_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish a separate human approval and the corresponding executable plan."""

    _, draft, _ = _load(draft_path, "exact-score plan draft")
    validate_draft(draft)
    approval_output = Path(os.path.abspath(os.path.expanduser(approval_output)))
    plan_output = Path(os.path.abspath(os.path.expanduser(plan_output)))
    if approval_output == plan_output:
        raise ValueError("approval and executable plan outputs must be distinct")
    for path in (approval_output, plan_output):
        batch._reject_symlink_ancestors(path.parent, "approval output parent")
        path.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(path) or os.path.lexists(Path(f"{path}.tmp")):
            raise FileExistsError(f"refusing existing approval/plan output: {path}")

    approval = batch.build_sealed_plan_approval(
        draft, reviewer_identity=reviewer_identity
    )
    batch._publish_exclusive(approval_output, approval)
    executable = {
        **draft,
        "sealed_plan_approval": _binding(
            approval_output, krea_provenance.file_sha256(approval_output)
        ),
    }
    # Full validation occurs only after the separately published approval exists
    # and before the executable plan becomes visible.
    batch._validate_plan(executable)
    batch._publish_exclusive(plan_output, executable)
    return approval, executable


def validate_plan(path: Path) -> dict[str, Any]:
    _, plan, _ = _load(path, "approved exact-score plan")
    batch._validate_plan(plan)
    return plan


def _add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle", action="append", required=True, type=Path)
    parser.add_argument("--zero-manifest", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--fixture-manifest", required=True, type=Path)
    parser.add_argument("--fixture-approval", required=True, type=Path)
    parser.add_argument("--cross-fixture-review", required=True, type=Path)
    parser.add_argument("--evaluator-config", required=True, type=Path)
    parser.add_argument(
        "--phase", required=True, choices=("discovery", "confirmation", "boundary")
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frozen-discovery-decision", type=Path)
    parser.add_argument("--candidate-family")


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    _add_build_arguments(commands.add_parser("build"))
    _add_build_arguments(commands.add_parser("dry-run"))
    approve = commands.add_parser("approve")
    approve.add_argument("--draft", required=True, type=Path)
    approve.add_argument("--reviewer-identity", required=True)
    approve.add_argument("--approval-output", required=True, type=Path)
    approve.add_argument("--plan-output", required=True, type=Path)
    validate = commands.add_parser("validate")
    target = validate.add_mutually_exclusive_group(required=True)
    target.add_argument("--draft", type=Path)
    target.add_argument("--plan", type=Path)
    return parser.parse_args(argv)


def _build_from_args(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    output = Path(os.path.abspath(os.path.expanduser(args.output_dir)))
    return build_documents(
        bundle_paths=args.bundle,
        zero_manifest_path=args.zero_manifest,
        dataset_path=args.dataset,
        fixture_manifest_path=args.fixture_manifest,
        fixture_approval_path=args.fixture_approval,
        cross_fixture_review_path=args.cross_fixture_review,
        evaluator_config_path=args.evaluator_config,
        phase=args.phase,
        campaign_output_path=output / "campaign.json",
        frozen_discovery_decision=args.frozen_discovery_decision,
        candidate_family=args.candidate_family,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    if args.command in {"build", "dry-run"}:
        campaign, draft = _build_from_args(args)
        if args.command == "build":
            campaign_path, draft_path = publish_build(
                output_dir=args.output_dir, campaign=campaign, draft=draft
            )
            result = {
                "status": "draft_built_unapproved",
                "campaign": _binding(
                    campaign_path, krea_provenance.file_sha256(campaign_path)
                ),
                "draft": _binding(draft_path, krea_provenance.file_sha256(draft_path)),
                "human_approval_created": False,
            }
        else:
            result = {
                "status": "dry_run_valid",
                "writes_performed": False,
                "campaign_sha256": campaign["manifest_sha256"],
                "draft_canonical_sha256": krea_provenance.canonical_sha256(draft),
                "candidate_count": len(draft["candidates"]),
            }
    elif args.command == "approve":
        approval, plan = approve_draft(
            draft_path=args.draft,
            reviewer_identity=args.reviewer_identity,
            approval_output=args.approval_output,
            plan_output=args.plan_output,
        )
        result = {
            "status": "approved_plan_published",
            "approval_sha256": krea_provenance.canonical_sha256(approval),
            "plan_canonical_sha256": krea_provenance.canonical_sha256(plan),
        }
    elif args.draft is not None:
        _, draft, _ = _load(args.draft, "exact-score plan draft")
        validate_draft(draft)
        result = {"status": "draft_valid", "human_approval_present": False}
    else:
        plan = validate_plan(args.plan)
        result = {
            "status": "approved_plan_valid",
            "plan_canonical_sha256": krea_provenance.canonical_sha256(plan),
        }
    print(krea_provenance.canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
