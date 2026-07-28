"""Operational score-plan builder tests, including a producer/consumer chain."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest


_ROOT = Path(__file__).parents[1]
_CALIBRATION = _ROOT / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))
sys.path.insert(0, str(Path(__file__).parent))

import batch_evaluate_krea as batch  # noqa: E402
import krea_decision  # noqa: E402
import krea_provenance  # noqa: E402
import krea_score_plan as score_plan  # noqa: E402
import test_krea_training_evidence_cli as stage3_test  # noqa: E402
from test_krea_v2_batch_contract import ProducerHarness  # noqa: E402


def _canonical_file(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(krea_provenance.canonical_bytes(value) + b"\n")
    return krea_provenance.file_sha256(path)


class BuilderHarness:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch):
        producer_root = root / "producer"
        producer_root.mkdir()
        self.base = ProducerHarness(producer_root, monkeypatch)
        base = self.base
        monkeypatch.setattr(
            score_plan.krea_fixture, "validate_approval", lambda value, **_kwargs: value
        )
        monkeypatch.setattr(
            score_plan.batch,
            "_validate_cross_fixture_review_surface",
            lambda value, **_kwargs: value,
        )
        monkeypatch.setattr(
            score_plan.krea_training_evidence,
            "validate_zero_control",
            lambda value, **_kwargs: value,
        )
        # This harness predates the operational Stage-3 producer and uses a
        # deliberately abbreviated bundle to isolate score-plan projection.
        # Dedicated tests below exercise the real strong validator boundary.
        monkeypatch.setattr(
            score_plan.krea_training_evidence,
            "validate_run_evidence",
            lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
        )
        monkeypatch.setattr(
            score_plan.batch, "_validate_evaluator", lambda value: dict(value)
        )

        discovery_plan = root / "discovery-plan.json"
        discovery_plan_sha = _canonical_file(discovery_plan, {"status": "frozen"})
        base.discovery_plan_sha = discovery_plan_sha
        execution_plan = root / "execution-plan.json"
        execution_plan_sha = _canonical_file(
            execution_plan,
            {
                "plan_sha256": base.execution_sha,
                "discovery_plan": {
                    "path": str(discovery_plan),
                    "sha256": discovery_plan_sha,
                },
            },
        )
        execution_approval = root / "execution-approval.json"
        execution_approval_sha = _canonical_file(
            execution_approval, {"decision": "approved"}
        )
        run_record = root / "run-record.json"
        run_record_sha = _canonical_file(run_record, {"status": "complete"})
        training_log = root / "training.log"
        training_log.write_bytes(b"complete\n")
        training_log_sha = krea_provenance.file_sha256(training_log)
        completion = root / "run-completion.json"
        completion_value = {
            "schema": 3,
            "kind": "forge-krea-training-completion",
            "arm_id": base.arm,
            "execution_plan_sha256": base.execution_sha,
            "natural_completion": True,
            "in_task_proxy_selection": {"enabled": False, "reserve_s": 0},
            "candidates": [
                {
                    "candidate_id": base.sealed_candidate["candidate_id"],
                    "sha256": base.local_sha,
                    "bytes": base.local_path.stat().st_size,
                    "step": 50,
                    "fraction_numerator": 50,
                    "fraction_denominator": 100,
                    "aliases": ["last.safetensors"],
                    "safetensors": {"test": "identity"},
                }
            ],
        }
        completion_sha = _canonical_file(completion, completion_value)
        local_binding_path = base.candidates[0]["provenance_path"]
        local_binding = {
            "schema": 2,
            "kind": "forge-krea-local-candidate-binding",
            "mode": "local_run_candidate",
            "arm_id": base.arm,
            "candidate_id": base.sealed_candidate["candidate_id"],
            "candidate": {
                "path": str(base.local_path),
                "sha256": base.local_sha,
                "bytes": base.local_path.stat().st_size,
                "step": 50,
                "fraction_numerator": 50,
                "fraction_denominator": 100,
                "aliases": ["last.safetensors"],
                "safetensors": {"test": "identity"},
            },
            "execution_plan": {
                "path": str(execution_plan),
                "sha256": execution_plan_sha,
            },
            "execution_approval": {
                "path": str(execution_approval),
                "sha256": execution_approval_sha,
            },
            "run_completion": {"path": str(completion), "sha256": completion_sha},
            "run_record": {"path": str(run_record), "sha256": run_record_sha},
            "training_log": {
                "path": str(training_log),
                "sha256": training_log_sha,
            },
            "evaluation_dataset_sha256": base.dataset_sha,
        }
        local_binding_sha = _canonical_file(local_binding_path, local_binding)
        base.candidates[0]["provenance_file_sha256"] = local_binding_sha
        base.candidates[0]["candidate_binding"][
            "binding_manifest_sha256"
        ] = local_binding_sha
        base.candidates[0]["candidate_binding"][
            "run_completion_sha256"
        ] = completion_sha

        self.bundle_path = root / "bundle.json"
        bundle_body = {
            "schema": 2,
            "kind": "forge-krea-run-evidence-bundle",
            "arm_id": base.arm,
            "execution_plan_sha256": base.execution_sha,
            "run_completion": {"path": str(completion), "sha256": completion_sha},
            "candidate_bindings": [
                {
                    "candidate_id": base.sealed_candidate["candidate_id"],
                    "candidate_sha256": base.local_sha,
                    "binding": {
                        "path": str(local_binding_path),
                        "sha256": local_binding_sha,
                    },
                }
            ],
        }
        self.bundle = {
            **bundle_body,
            "bundle_sha256": krea_provenance.canonical_sha256(bundle_body),
        }
        _canonical_file(self.bundle_path, self.bundle)

        self.zero_manifest_path = root / "zero-manifest.json"
        zero_body = {
            "schema": 2,
            "kind": "forge-krea-zero-lora-control",
            "mode": "zero_lora_control",
            "artifact": {
                "path": str(base.zero_path),
                "sha256": base.zero_sha,
                "bytes": base.zero_path.stat().st_size,
            },
            "evaluation_dataset_sha256": base.dataset_sha,
        }
        self.zero_manifest = {
            **zero_body,
            "manifest_sha256": base.zero_manifest_sha,
        }
        self.zero_manifest_file_sha = _canonical_file(
            self.zero_manifest_path, self.zero_manifest
        )
        base.candidates[1]["provenance_path"] = self.zero_manifest_path
        base.candidates[1]["provenance_file_sha256"] = self.zero_manifest_file_sha
        base.candidates[1]["candidate_binding"][
            "binding_manifest_sha256"
        ] = self.zero_manifest_file_sha

        self.evaluator_config_path = root / "evaluator.json"
        evaluator = {
            key: deepcopy(value)
            for key, value in base.evaluator.items()
            if not key.startswith("_")
        }
        evaluator["cache_provenance_sha256"] = score_plan.hashlib.sha256(
            b"cache"
        ).hexdigest()
        _canonical_file(self.evaluator_config_path, evaluator)
        self.output_dir = root / "built"

    def kwargs(self, *, phase: str = "boundary") -> dict:
        return {
            "bundle_paths": [self.bundle_path],
            "zero_manifest_path": self.zero_manifest_path,
            "dataset_path": self.base.dataset,
            "fixture_manifest_path": self.base.fixture_path,
            "fixture_approval_path": self.base.fixture_approval_path,
            "cross_fixture_review_path": self.base.cross_review_path,
            "evaluator_config_path": self.evaluator_config_path,
            "phase": phase,
            "campaign_output_path": self.output_dir / "campaign.json",
            "frozen_discovery_decision": (
                self.base.discovery_path if phase == "boundary" else None
            ),
            "candidate_family": self.base.arm if phase == "boundary" else None,
        }

    def build(self) -> tuple[dict, dict, Path, Path]:
        campaign, draft = score_plan.build_documents(**self.kwargs())
        campaign_path, draft_path = score_plan.publish_build(
            output_dir=self.output_dir, campaign=campaign, draft=draft
        )
        return campaign, draft, campaign_path, draft_path


def test_build_is_unapproved_and_dry_run_performs_no_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    harness = BuilderHarness(tmp_path, monkeypatch)
    dry_target = tmp_path / "dry-target"
    args = [
        "dry-run",
        "--bundle",
        str(harness.bundle_path),
        "--zero-manifest",
        str(harness.zero_manifest_path),
        "--dataset",
        str(harness.base.dataset),
        "--fixture-manifest",
        str(harness.base.fixture_path),
        "--fixture-approval",
        str(harness.base.fixture_approval_path),
        "--cross-fixture-review",
        str(harness.base.cross_review_path),
        "--evaluator-config",
        str(harness.evaluator_config_path),
        "--phase",
        "boundary",
        "--output-dir",
        str(dry_target),
        "--frozen-discovery-decision",
        str(harness.base.discovery_path),
        "--candidate-family",
        harness.base.arm,
    ]
    assert score_plan.main(args) == 0
    assert json.loads(capsys.readouterr().out)["writes_performed"] is False
    assert not dry_target.exists()

    _campaign, draft, _campaign_path, draft_path = harness.build()
    assert "sealed_plan_approval" not in draft
    assert set(path.name for path in harness.output_dir.iterdir()) == {
        "campaign.json",
        "score-plan.draft.json",
    }
    assert score_plan.main(["validate", "--draft", str(draft_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "human_approval_present": False,
        "status": "draft_valid",
    }


def test_approval_is_a_separate_command_and_final_plan_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    harness = BuilderHarness(tmp_path, monkeypatch)
    _campaign, _draft, _campaign_path, draft_path = harness.build()
    approval_path = tmp_path / "human-approval.json"
    plan_path = tmp_path / "score-plan.json"
    assert (
        score_plan.main(
            [
                "approve",
                "--draft",
                str(draft_path),
                "--reviewer-identity",
                "Morgan Auditor",
                "--approval-output",
                str(approval_path),
                "--plan-output",
                str(plan_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "approved_plan_published"
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert approval["reviewer_identity"] == "Morgan Auditor"
    assert plan["sealed_plan_approval"] == {
        "path": str(approval_path),
        "sha256": krea_provenance.file_sha256(approval_path),
    }
    assert score_plan.main(["validate", "--plan", str(plan_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "approved_plan_valid"


def test_fake_stage3_builder_to_batch_to_decision_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = BuilderHarness(tmp_path, monkeypatch)
    campaign, _draft, campaign_path, draft_path = harness.build()
    base = harness.base
    approval_path = tmp_path / "round-trip-approval.json"
    _approval, executable = score_plan.approve_draft(
        draft_path=draft_path,
        reviewer_identity="Round Trip Reviewer",
        approval_output=approval_path,
        plan_output=tmp_path / "round-trip-plan.json",
    )
    base.score_approval_path = approval_path
    base.score_approval_sha = krea_provenance.file_sha256(approval_path)
    base.evaluator["_sealed_plan_approval_path"] = str(approval_path)
    base.evaluator["_sealed_plan_approval_sha256"] = base.score_approval_sha
    base.evaluator["_sealed_plan_approval"] = {
        "decision": "approved",
        "reviewer_identity": "Round Trip Reviewer",
    }
    base.evaluator["_plan_payload_sha256"] = batch._plan_payload_sha256(executable)
    base.evaluator["_campaign_manifest_path"] = str(campaign_path)
    base.evaluator["_campaign_manifest_file_sha256"] = krea_provenance.file_sha256(
        campaign_path
    )
    base.evaluator["_campaign_manifest_sha256"] = campaign["manifest_sha256"]
    base.evaluator["_decision_context"] = batch._validate_score_decision_context(
        executable["decision_context"], campaign=campaign
    )
    base.campaign = campaign
    base.campaign_path = campaign_path
    base.campaign_file_sha = krea_provenance.file_sha256(campaign_path)
    # The mocked batch validator returns ``base.candidates`` directly; keep its
    # synthetic zero-control id aligned with the executable score plan that the
    # real validator would return.
    base.candidates[1]["id"] = "K0-zero-control"

    aggregate_path = tmp_path / "builder-aggregate.json"
    aggregate = batch.run_batch(
        executable,
        results_dir=tmp_path / "builder-results",
        output=aggregate_path,
    )
    normalized, _ = krea_decision._aggregate(aggregate_path)
    assert normalized["campaign"]["runs"] == campaign["runs"]
    base.patch_match_context(monkeypatch)
    observed, _bindings = krea_decision._match_aggregates(
        policy=base.boundary_policy(aggregate), aggregate_paths=[aggregate_path]
    )
    assert (
        observed["B-0p5-small"]["zero"]["zero_control_manifest_sha256"]
        == base.zero_manifest_sha
    )


def test_builder_rejects_truncation_and_phase_ambiguity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = BuilderHarness(tmp_path, monkeypatch)
    truncated = deepcopy(harness.bundle)
    truncated["candidate_bindings"] = []
    body = {key: value for key, value in truncated.items() if key != "bundle_sha256"}
    truncated["bundle_sha256"] = krea_provenance.canonical_sha256(body)
    truncated_path = tmp_path / "truncated-bundle.json"
    _canonical_file(truncated_path, truncated)
    kwargs = harness.kwargs()
    kwargs["bundle_paths"] = [truncated_path]
    with pytest.raises(ValueError, match="coverage is incomplete"):
        score_plan.build_documents(**kwargs)

    with pytest.raises(ValueError, match="boundary-ambiguous"):
        score_plan.build_documents(**harness.kwargs(phase="discovery"))


def test_publish_rejects_a_draft_bound_to_another_campaign_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = BuilderHarness(tmp_path, monkeypatch)
    campaign, draft = score_plan.build_documents(**harness.kwargs())
    draft["campaign_manifest"]["path"] = str(tmp_path / "elsewhere.json")
    with pytest.raises(ValueError, match="does not name the campaign"):
        score_plan.publish_build(
            output_dir=harness.output_dir,
            campaign=campaign,
            draft=draft,
        )
    assert not harness.output_dir.exists()


def test_draft_validation_rejects_a_tampered_fixture_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = BuilderHarness(tmp_path, monkeypatch)
    _campaign, _draft, _campaign_path, draft_path = harness.build()
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    draft["fixture_manifest"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fixture manifest file binding"):
        score_plan.validate_draft(draft)


def test_consumer_rejects_fully_rehashed_arbitrary_candidate_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A digest-consistent JSON chain cannot replace safetensors validation."""

    approved_run = stage3_test.approved_run.__wrapped__(tmp_path, monkeypatch)

    # Give the synthetic Stage-3 fixture the discovery binding consumed by the
    # score-plan projection.  Its real plan validator is already replaced by
    # the fixture's strict equality stub, so mutate the shared plan before emit.
    discovery_path = approved_run["root"] / "discovery-plan.json"
    discovery_sha = _canonical_file(discovery_path, {"status": "frozen"})
    approved_run["plan"]["discovery_plan"] = {
        "path": str(discovery_path),
        "sha256": discovery_sha,
    }
    plan_file_sha = _canonical_file(approved_run["plan_path"], approved_run["plan"])
    condition = json.loads(approved_run["condition_path"].read_text())
    condition["execution_plan_file_sha256"] = plan_file_sha
    _canonical_file(approved_run["condition_path"], condition)
    monkeypatch.setattr(
        score_plan.krea_training_evidence.krea_execution_plan,
        "validate_approval",
        lambda value, **_kwargs: value,
    )
    bundle_path = stage3_test._emit(approved_run)
    bundle = json.loads(bundle_path.read_text())

    target_ref = bundle["candidate_bindings"][0]
    target_binding_path = Path(target_ref["binding"]["path"])
    target_binding = json.loads(target_binding_path.read_text())
    target_artifact = Path(target_binding["candidate"]["path"])
    target_artifact.chmod(0o600)
    target_artifact.write_bytes(b"attacker-chosen arbitrary candidate bytes")
    forged_sha = krea_provenance.file_sha256(target_artifact)
    forged_bytes = target_artifact.stat().st_size
    forged_layout = {"attacker_assertion": "not-a-safetensors-identity"}
    target_id = target_ref["candidate_id"]
    target_step = target_binding["candidate"]["step"]
    forged_id = f"step-{target_step}-{forged_sha[:12]}"
    forged_artifact = target_artifact.with_name(f"{forged_id}.safetensors")
    target_artifact.parent.chmod(0o700)
    target_artifact.rename(forged_artifact)
    target_artifact.parent.chmod(0o500)

    # Rehash the run record, completion, every candidate binding, and bundle.
    # This models the exact weakness of a consumer that checks only JSON hashes.
    run_record_path = Path(target_binding["run_record"]["path"])
    run_record = json.loads(run_record_path.read_text())
    aliases = target_binding["candidate"]["aliases"]
    for alias in aliases:
        run_record["current_scope_candidates"][alias["name"]] = forged_sha
        run_record["artifacts"]["candidate_sha256"][alias["name"]] = forged_sha
    run_record_sha = _canonical_file(run_record_path, run_record)

    completion_path = Path(bundle["run_completion"]["path"])
    completion = json.loads(completion_path.read_text())
    completion["run_record_sha256"] = run_record_sha
    completion_target = next(
        row for row in completion["candidates"] if row["candidate_id"] == target_id
    )
    completion_target["candidate_id"] = forged_id
    completion_target["sha256"] = forged_sha
    completion_target["bytes"] = forged_bytes
    completion_target["safetensors"] = forged_layout
    completion_sha = _canonical_file(completion_path, completion)

    for row in bundle["candidate_bindings"]:
        binding_path = Path(row["binding"]["path"])
        binding = json.loads(binding_path.read_text())
        binding["run_record"]["sha256"] = run_record_sha
        binding["run_completion"]["sha256"] = completion_sha
        if binding["candidate_id"] == target_id:
            binding["candidate_id"] = forged_id
            binding["candidate"]["path"] = str(forged_artifact)
            binding["candidate"]["sha256"] = forged_sha
            binding["candidate"]["bytes"] = forged_bytes
            binding["candidate"]["safetensors"] = forged_layout
            row["candidate_id"] = forged_id
            row["candidate_sha256"] = forged_sha
        row["binding"]["sha256"] = _canonical_file(binding_path, binding)
    bundle["run_completion"]["sha256"] = completion_sha
    bundle_body = {
        key: value for key, value in bundle.items() if key != "bundle_sha256"
    }
    bundle["bundle_sha256"] = krea_provenance.canonical_sha256(bundle_body)
    _canonical_file(bundle_path, bundle)

    strong_validator = score_plan.krea_training_evidence.validate_run_evidence
    monkeypatch.setattr(
        score_plan.krea_training_evidence,
        "validate_run_evidence",
        lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )
    # The previous manual projection accepts the fully rehashed forgery.
    projected, *_rest = score_plan._bundle_candidates(bundle_path)
    assert projected["candidates"][0]["sha256"] == forged_sha

    monkeypatch.setattr(
        score_plan.krea_training_evidence,
        "validate_run_evidence",
        strong_validator,
    )
    with pytest.raises(ValueError, match="safetensors"):
        score_plan._bundle_candidates(bundle_path)
