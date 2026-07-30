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
    from . import krea_execution_surface_policy
    from . import krea_provenance
    from . import krea_historical_training_evidence
    from . import krea_scorer_extension_policy
    from . import krea_training_evidence
except ImportError:  # pragma: no cover - direct script execution.
    import batch_evaluate_krea as batch  # type: ignore[no-redef]
    import krea_fixture  # type: ignore[no-redef]
    import krea_execution_surface_policy  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_historical_training_evidence  # type: ignore[no-redef]
    import krea_scorer_extension_policy  # type: ignore[no-redef]
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
_KREA_ASSET_REPOSITORY = "Comfy-Org/Krea-2"
_KREA_ASSET_REVISION = "952f49d49653cb42e7d6cf7cbfad74738073ec7d"
_KREA_ASSET_SOURCE_PATHS = {
    "diffusion_model": "diffusion_models/krea2_raw_fp8_scaled.safetensors",
    "text_encoder": "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
    "vae": "vae/qwen_image_vae.safetensors",
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
    *,
    historical_validator_identity: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], str, str]:
    path, bundle, _ = _load(bundle_path, "stage-three run-evidence bundle")
    if historical_validator_identity is None:
        validated_bundle = krea_training_evidence.validate_run_evidence(path)
    else:
        validated_bundle = krea_historical_training_evidence.validate_run_evidence(
            path, historical_validator_identity
        )
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
            "discovery_profile_index_sha256",
            "discovery_execution_authorization_sha256",
            "host_bootstrap_receipt_sha256",
            "execution_surface_policy_sha256",
            "execution_surface",
            "execution_scope",
            "run_completion",
            "candidate_bindings",
            "bundle_sha256",
        },
        "stage-three run-evidence bundle",
    )
    expected_execution_surface_policy_sha256 = (
        krea_execution_surface_policy.POLICY["policy_sha256"]
        if historical_validator_identity is None
        else historical_validator_identity["execution_surface_policy_sha256"]
    )
    body = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    if (
        bundle["schema"] != 2
        or bundle["kind"] != "forge-krea-run-evidence-bundle"
        or bundle["bundle_sha256"] != krea_provenance.canonical_sha256(body)
        or not isinstance(bundle["arm_id"], str)
        or not batch._SAFE_ID.fullmatch(bundle["arm_id"])
        or bundle["execution_surface_policy_sha256"]
        != expected_execution_surface_policy_sha256
        or bundle["execution_surface"] != "staged_host_venv"
        or bundle["execution_scope"] != "discovery_only"
        or any(
            not isinstance(bundle[name], str)
            or not batch._SHA256.fullmatch(bundle[name])
            for name in (
                "discovery_profile_index_sha256",
                "discovery_execution_authorization_sha256",
                "host_bootstrap_receipt_sha256",
            )
        )
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
    manifest_path: Path,
    *,
    evaluation_dataset_sha256: str,
    training_evidence_validator: Any = krea_training_evidence,
) -> tuple[dict[str, Any], str]:
    path, manifest, file_sha = _load(manifest_path, "zero-control manifest")
    artifact = batch._object(manifest.get("artifact"), "zero-control artifact")
    artifact_path = batch._safe_file(artifact.get("path"), "zero-control artifact")
    training_evidence_validator.validate_zero_control(
        manifest, artifact_path=artifact_path
    )
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
    historical_training_validator_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct a sealed campaign and unapproved exact-score plan draft."""

    if phase not in {"discovery", "confirmation", "boundary"}:
        raise ValueError("phase must be discovery, confirmation, or boundary")
    if not bundle_paths:
        raise ValueError("at least one stage-three bundle is required")
    campaign_output_path = Path(
        os.path.abspath(os.path.expanduser(campaign_output_path))
    )
    historical_validator_identity = (
        None
        if historical_training_validator_root is None
        else krea_historical_training_evidence.capture_identity(
            historical_training_validator_root
        )
    )
    historical_modules = (
        None
        if historical_validator_identity is None
        else krea_historical_training_evidence.load_modules(
            historical_validator_identity
        )
    )
    fixture_validator = (
        krea_fixture if historical_modules is None else historical_modules["fixture"]
    )
    cross_validator = (
        batch
        if historical_modules is None
        else historical_modules["batch_evaluate"]
    )
    fixture_path, fixture, fixture_file_sha = _load(
        fixture_manifest_path, "fixture manifest"
    )
    fixture_validator.validate_manifest(fixture)
    approval_path, approval, approval_file_sha = _load(
        fixture_approval_path, "fixture approval"
    )
    fixture_validator.validate_approval(approval, fixture_manifest=fixture)
    cross_path, cross_review, cross_file_sha = _load(
        cross_fixture_review_path, "cross-fixture review"
    )
    cross_validator._validate_cross_fixture_review_surface(
        cross_review, fixture=fixture, source_path=cross_path
    )
    evaluator_path, evaluator, _ = _load(evaluator_config_path, "evaluator config")
    del evaluator_path
    evaluator = batch._validate_evaluator(evaluator)
    batch._validate_scorer_fixture_timeout(evaluator, fixture)
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
            raw_bundle,
            historical_validator_identity=historical_validator_identity,
        )
        if evaluation_sha != expected_dataset_sha:
            raise ValueError(
                f"run bundle for arm {run['arm_id']} belongs to another "
                "evaluation fixture"
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
        zero_manifest_path,
        evaluation_dataset_sha256=expected_dataset_sha,
        training_evidence_validator=(
            krea_training_evidence
            if historical_modules is None
            else historical_modules["training_evidence"]
        ),
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
    if historical_validator_identity is not None:
        campaign_payload["historical_training_evidence_validator"] = (
            historical_validator_identity
        )
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
    historical_identity = campaign.get("historical_training_evidence_validator")
    historical_modules = (
        None
        if historical_identity is None
        else krea_historical_training_evidence.load_modules(historical_identity)
    )
    fixture_validator = (
        krea_fixture if historical_modules is None else historical_modules["fixture"]
    )
    cross_validator = (
        batch
        if historical_modules is None
        else historical_modules["batch_evaluate"]
    )
    _fixture_path, fixture, _fixture_file_sha = _bound_document(
        draft["fixture_manifest"], "fixture manifest"
    )
    fixture_validator.validate_manifest(fixture)
    _approval_path, approval, _approval_file_sha = _bound_document(
        draft["fixture_approval"], "fixture approval"
    )
    fixture_validator.validate_approval(approval, fixture_manifest=fixture)
    _cross_path, cross_review, _cross_file_sha = _bound_document(
        draft["cross_fixture_review"], "cross-fixture review"
    )
    cross_validator._validate_cross_fixture_review_surface(
        cross_review, fixture=fixture, source_path=_cross_path
    )
    if dataset_spec["sha256"] != fixture["evaluation_dataset_identity"]["sha256"]:
        raise ValueError("draft dataset differs from the approved fixture identity")
    evaluator = batch._validate_evaluator(draft["evaluator"])
    batch._validate_scorer_fixture_timeout(evaluator, fixture)
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
    reviewer_identity: str | None,
    approval_output: Path,
    plan_output: Path,
    technical_reviewer_actor: dict[str, Any] | None = None,
    discovery_authorization_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish a separately attributed approval and executable plan."""

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

    if technical_reviewer_actor is not None:
        if reviewer_identity is not None or discovery_authorization_path is None:
            raise ValueError(
                "agent score approval requires only a technical actor and authorization"
            )
        authorization_path, authorization, authorization_file_sha = _load(
            discovery_authorization_path, "discovery execution authorization"
        )
        approval = batch.build_agent_sealed_plan_approval(
            draft,
            technical_reviewer_actor=technical_reviewer_actor,
            discovery_execution_authorization={
                "path": str(authorization_path),
                "file_sha256": authorization_file_sha,
                "authorization_sha256": authorization["authorization_sha256"],
            },
        )
    else:
        if reviewer_identity is None or discovery_authorization_path is not None:
            raise ValueError("legacy score approval requires one named human")
        if draft["decision_context"].get("phase") == "discovery":
            for raw_candidate in draft["candidates"]:
                _path, binding, _sha = _bound_document(
                    raw_candidate["candidate_binding"],
                    "candidate binding for approval governance",
                )
                if binding.get("mode") != "local_run_candidate":
                    continue
                _execution_path, execution, _execution_sha = _bound_document(
                    binding["execution_plan"],
                    "candidate execution plan for approval governance",
                )
                if execution.get("discovery_execution_authorization") is not None:
                    raise ValueError(
                        "authorization-bound Stage-1 discovery scoring requires "
                        "the delegated technical reviewer"
                    )
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


def stage_stage1_evaluator_assets(
    *, comfy_root: Path, token: str, receipt_output: Path
) -> dict[str, Any]:
    """Download and verify only the three immutable Stage-1 Krea assets."""

    if not isinstance(token, str) or not token:
        raise ValueError("HF_TOKEN is required for Krea asset staging")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - host dependency gate.
        raise RuntimeError("huggingface_hub is required for asset staging") from exc

    comfy_root = comfy_root.resolve(strict=True)
    contract = krea_execution_surface_policy.POLICY["stage1_exact_scorer_contract"]
    if (comfy_root / "extra_model_paths.yaml").exists():
        raise ValueError("asset staging refuses Comfy extra_model_paths.yaml")
    lora_root = comfy_root / "models" / "loras"
    if os.path.lexists(lora_root):
        batch._empty_real_directory(
            lora_root,
            "asset-staging ComfyUI LoRA directory",
            allowed_zero_byte_placeholder=batch._COMFY_LORA_PLACEHOLDER,
        )
    staged = []
    for name in sorted(contract["assets"]):
        expected = contract["assets"][name]
        destination = comfy_root / expected["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(destination):
            raise FileExistsError(f"refusing existing evaluator asset: {destination}")
        source = Path(
            hf_hub_download(
                repo_id=_KREA_ASSET_REPOSITORY,
                filename=_KREA_ASSET_SOURCE_PATHS[name],
                revision=_KREA_ASSET_REVISION,
                token=token,
            )
        ).resolve(strict=True)
        digest = hashlib.sha256()
        size = 0
        try:
            with source.open("rb") as reader, destination.open("xb") as writer:
                for block in iter(lambda: reader.read(16 * 1024 * 1024), b""):
                    writer.write(block)
                    digest.update(block)
                    size += len(block)
                writer.flush()
                os.fsync(writer.fileno())
            destination.chmod(0o400)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        if digest.hexdigest() != expected["sha256"] or size != expected["bytes"]:
            destination.unlink()
            raise ValueError(f"downloaded evaluator asset differs: {name}")
        staged.append(
            {
                "name": name,
                "source_filename": _KREA_ASSET_SOURCE_PATHS[name],
                "canonical_path": str(destination),
                "sha256": expected["sha256"],
                "bytes": expected["bytes"],
            }
        )
    body = {
        "schema": 1,
        "kind": "forge-krea-stage1-evaluator-asset-stage",
        "repository": _KREA_ASSET_REPOSITORY,
        "revision": _KREA_ASSET_REVISION,
        "assets": staged,
        "credential_recorded": False,
    }
    receipt = {**body, "receipt_sha256": krea_provenance.canonical_sha256(body)}
    _canonical_file(receipt_output, receipt)
    return receipt


def build_stage1_evaluator_config(
    *,
    comfy_root: Path,
    god_root: Path,
    python_path: Path,
    cache_provenance_sha256: str,
    fixture_role: str,
    systemd_run_path: Path = Path("/usr/bin/systemd-run"),
    systemctl_path: Path = Path("/usr/bin/systemctl"),
) -> dict[str, Any]:
    """Materialize the literal owner-bound Stage-1 exact-scorer config."""

    try:
        from . import evaluate_krea_local as local_evaluator
    except ImportError:  # pragma: no cover - direct script execution.
        import evaluate_krea_local as local_evaluator  # type: ignore[no-redef]

    if not isinstance(cache_provenance_sha256, str) or not batch._SHA256.fullmatch(
        cache_provenance_sha256
    ):
        raise ValueError("cache provenance SHA-256 is invalid")
    comfy_root = comfy_root.resolve(strict=True)
    god_root = god_root.resolve(strict=True)
    python_path = python_path.resolve(strict=True)
    contract = krea_execution_surface_policy.POLICY["stage1_exact_scorer_contract"]
    effective_timeouts = krea_scorer_extension_policy.effective_timeouts(
        contract["timeouts_s"], fixture_role
    )
    python_environment = local_evaluator._python_environment(python_path)
    driver_environment = {
        key: python_environment[key]
        for key in (
            "executable",
            "prefix",
            "base_prefix",
            "python",
            "distribution_count",
            "distributions_sha256",
            "normalized_distributions_sha256",
        )
    }
    assets = {
        name: {
            "canonical_path": str(comfy_root / row["relative_path"]),
            "sha256": row["sha256"],
            "bytes": row["bytes"],
        }
        for name, row in contract["assets"].items()
    }
    systemd_run = systemd_run_path.resolve(strict=True)
    systemctl = systemctl_path.resolve(strict=True)
    config = {
        "comfy_root": str(comfy_root),
        "comfy_python": str(python_path),
        "god_root": str(god_root),
        "driver_python": str(python_path),
        "expected_god_commit": contract["god_commit"],
        "expected_comfy_commit": contract["comfy_commit"],
        "expected_tooling_commit": contract["tooling_commit"],
        "expected_evaluator_script_sha256": contract["evaluator_script_sha256"],
        "expected_dataset_identity_module_sha256": contract[
            "dataset_identity_module_sha256"
        ],
        "expected_eval_defaults": contract["eval_defaults"],
        "expected_runtime_identity": {
            "comfy_python_identity_sha256": krea_provenance.canonical_sha256(
                python_environment
            ),
            "driver_python_identity_sha256": krea_provenance.canonical_sha256(
                driver_environment
            ),
        },
        "expected_assets": assets,
        "cache_provenance_sha256": cache_provenance_sha256,
        "containment": {
            "mode": "systemd_transient_service",
            "term_grace_s": contract["timeouts_s"]["containment_term_grace"],
            "systemd_run_path": str(systemd_run),
            "systemd_run_sha256": krea_provenance.file_sha256(systemd_run),
            "systemctl_path": str(systemctl),
            "systemctl_sha256": krea_provenance.file_sha256(systemctl),
            "unit_type": "transient_service",
            "network_policy": {
                "private_network": True,
                "restrict_address_families": ["AF_UNIX", "AF_INET", "AF_INET6"],
                "loopback_allowed": True,
                "outbound_network_blocked": True,
            },
        },
        "base_name": contract["assets"]["diffusion_model"]["basename"],
        "startup_timeout_s": contract["timeouts_s"]["startup"],
        "evaluation_timeout_s": contract["timeouts_s"]["evaluation"],
        "shutdown_timeout_s": contract["timeouts_s"]["shutdown"],
        "scorer_extension_policy": krea_scorer_extension_policy.POLICY,
        "scorer_timeout_profile": fixture_role,
    }
    config["evaluation_timeout_s"] = effective_timeouts["evaluation"]
    normalized = batch._validate_evaluator(config)
    batch._validate_stage1_exact_scorer(normalized)
    return normalized


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
    parser.add_argument(
        "--historical-training-validator-root",
        type=Path,
        help=(
            "exact clean f6ce1ad worktree used only to validate training "
            "evidence; scorer code remains current"
        ),
    )


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    _add_build_arguments(commands.add_parser("build"))
    _add_build_arguments(commands.add_parser("dry-run"))
    approve = commands.add_parser("approve")
    approve.add_argument("--draft", required=True, type=Path)
    approval_actor = approve.add_mutually_exclusive_group(required=True)
    approval_actor.add_argument("--reviewer-identity")
    approval_actor.add_argument("--technical-actor", type=Path)
    approve.add_argument("--discovery-authorization", type=Path)
    approve.add_argument("--approval-output", required=True, type=Path)
    approve.add_argument("--plan-output", required=True, type=Path)
    validate = commands.add_parser("validate")
    target = validate.add_mutually_exclusive_group(required=True)
    target.add_argument("--draft", type=Path)
    target.add_argument("--plan", type=Path)
    evaluator = commands.add_parser("build-stage1-evaluator")
    evaluator.add_argument("--comfy-root", required=True, type=Path)
    evaluator.add_argument("--god-root", required=True, type=Path)
    evaluator.add_argument("--python", required=True, type=Path)
    evaluator.add_argument("--cache-provenance-sha256", required=True)
    evaluator.add_argument("--fixture-role", required=True, choices=("D1", "D2"))
    evaluator.add_argument("--output", required=True, type=Path)
    assets = commands.add_parser("stage-stage1-assets")
    assets.add_argument("--comfy-root", required=True, type=Path)
    assets.add_argument("--receipt", required=True, type=Path)
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
        historical_training_validator_root=args.historical_training_validator_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    if args.command == "stage-stage1-assets":
        receipt = stage_stage1_evaluator_assets(
            comfy_root=args.comfy_root,
            token=os.environ.get("HF_TOKEN", ""),
            receipt_output=args.receipt,
        )
        result = {
            "status": "stage1_exact_evaluator_assets_staged",
            "receipt_sha256": receipt["receipt_sha256"],
        }
    elif args.command == "build-stage1-evaluator":
        evaluator = build_stage1_evaluator_config(
            comfy_root=args.comfy_root,
            god_root=args.god_root,
            python_path=args.python,
            cache_provenance_sha256=args.cache_provenance_sha256,
            fixture_role=args.fixture_role,
        )
        _canonical_file(args.output, evaluator)
        result = {
            "status": "stage1_exact_evaluator_config_published",
            "path": str(args.output),
            "file_sha256": krea_provenance.file_sha256(args.output),
        }
    elif args.command in {"build", "dry-run"}:
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
        technical_actor = None
        if args.technical_actor is not None:
            _, technical_actor, _ = _load(
                args.technical_actor, "exact-score technical actor"
            )
        approval, plan = approve_draft(
            draft_path=args.draft,
            reviewer_identity=args.reviewer_identity,
            approval_output=args.approval_output,
            plan_output=args.plan_output,
            technical_reviewer_actor=technical_actor,
            discovery_authorization_path=args.discovery_authorization,
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
