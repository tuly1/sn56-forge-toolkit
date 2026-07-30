"""Producer-to-consumer tests for the schema-2 Krea score-batch contract.

The external Comfy process is replaced with a deterministic evaluator result;
input staging, exact result validation, aggregate publication, and the
``krea_decision`` aggregate/campaign adapters remain real.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


_CALIBRATION = Path(__file__).parents[1] / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))

import batch_evaluate_krea as batch  # noqa: E402
import krea_decision  # noqa: E402
import krea_provenance  # noqa: E402


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical_file(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(krea_provenance.canonical_bytes(value) + b"\n")
    return krea_provenance.file_sha256(path)


def _reseal_aggregate(value: dict) -> dict:
    body = {key: item for key, item in value.items() if key != "aggregate_sha256"}
    return {**body, "aggregate_sha256": krea_provenance.canonical_sha256(body)}


def _reseal_campaign_adapter(adapter: dict) -> dict:
    body = {
        "schema": 2,
        "kind": "forge-krea-exact-score-campaign",
        "fixture_manifest_sha256": adapter["fixture_manifest_sha256"],
        "discovery_plan_sha256": adapter["discovery_plan_sha256"],
        "runs": adapter["runs"],
        "zero_control_manifest_sha256": adapter["zero_control_manifest_sha256"],
        "decision_contract": adapter["decision_contract"],
        "confirmation_contract": adapter["confirmation_contract"],
    }
    manifest_sha = krea_provenance.canonical_sha256(body)
    manifest = {**body, "manifest_sha256": manifest_sha}
    return {
        "manifest_sha256": manifest_sha,
        "file_sha256": hashlib.sha256(
            krea_provenance.canonical_bytes(manifest) + b"\n"
        ).hexdigest(),
        "fixture_manifest_sha256": body["fixture_manifest_sha256"],
        "discovery_plan_sha256": body["discovery_plan_sha256"],
        "zero_control_manifest_sha256": body["zero_control_manifest_sha256"],
        "decision_contract": body["decision_contract"],
        "confirmation_contract": body["confirmation_contract"],
        "runs": body["runs"],
    }


def _rewrite_canonical(path: Path, value: dict) -> str:
    path.chmod(0o600)
    return _canonical_file(path, value)


def _evidence_manifest(output: Path) -> tuple[Path, dict]:
    aggregate = json.loads(output.read_text(encoding="utf-8"))
    reference = aggregate["decision_evidence"]
    path = output.parent / reference["archive_path"] / reference["manifest_path"]
    return path, json.loads(path.read_text(encoding="utf-8"))


class ProducerHarness:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch):
        self.root = root
        self.arm = "K1"
        self.execution_sha = _sha("execution-plan")
        self.completion_sha = _sha("run-completion")
        self.zero_manifest_sha = _sha("zero-control-manifest")
        self.discovery_plan_sha = _sha("discovery-plan-file")
        self.dataset_sha = _sha("evaluation-dataset")
        self.training_sha = _sha("training-dataset")

        self.dataset = root / "evaluation-dataset"
        self.dataset.mkdir()
        (self.dataset / "row-000.png").write_bytes(b"not-decoded-by-this-test")
        self.comfy_root = root / "comfy"
        (self.comfy_root / "models" / "loras").mkdir(parents=True)
        self.god_root = root / "god"
        self.god_root.mkdir()

        self.local_path = root / "K1-step-50.safetensors"
        self.zero_path = root / "zero.safetensors"
        self.local_path.write_bytes(b"local-candidate-bytes")
        self.zero_path.write_bytes(b"zero-control-bytes")
        self.local_sha = krea_provenance.file_sha256(self.local_path)
        self.zero_sha = krea_provenance.file_sha256(self.zero_path)
        self.sealed_candidate = {
            "candidate_id": "K1-step-50",
            "sha256": self.local_sha,
            "bytes": self.local_path.stat().st_size,
            "step": 50,
            "fraction": {"numerator": 50, "denominator": 100},
        }
        self.run = {
            "arm_id": self.arm,
            "execution_plan_sha256": self.execution_sha,
            "run_completion_sha256": self.completion_sha,
            "candidates": [self.sealed_candidate],
        }

        self.fixture_identity_sha = _sha("fixture-identity")
        self.fixture = {
            "manifest_sha256": self.fixture_identity_sha,
            "concept_id": "boundary-concept",
            "experimental_role": "B-0p5-small",
            "training_rows": [{"row_id": f"train-{index:02d}"} for index in range(20)],
            "evaluation_rows": [{"row_id": f"row-{index:03d}"} for index in range(24)],
            "training_dataset_identity": {"sha256": self.training_sha},
            "evaluation_dataset_identity": {"sha256": self.dataset_sha},
        }
        self.fixture_path = root / "fixture.json"
        self.fixture_file_sha = _canonical_file(self.fixture_path, self.fixture)
        self.fixture_approval_path = root / "fixture-approval.json"
        self.fixture_approval_sha = _canonical_file(
            self.fixture_approval_path, {"decision": "approved"}
        )
        self.cross_review_path = root / "cross-review.json"
        self.cross_review_sha = _canonical_file(
            self.cross_review_path, {"decision": "passed"}
        )
        self.score_approval_path = root / "score-approval.json"
        self.score_approval_sha = _canonical_file(
            self.score_approval_path,
            {"decision": "approved", "reviewer_identity": "Riley Reviewer"},
        )

        campaign_body = {
            "schema": 2,
            "kind": "forge-krea-exact-score-campaign",
            "fixture_manifest_sha256": self.fixture_identity_sha,
            "discovery_plan_sha256": self.discovery_plan_sha,
            "runs": [self.run],
            "zero_control_manifest_sha256": self.zero_manifest_sha,
            "decision_contract": batch._DISCOVERY_DECISION_BINDING,
            "confirmation_contract": batch._CONFIRMATION_DECISION_BINDING,
        }
        self.campaign = batch.seal_campaign_manifest(campaign_body)
        self.campaign_path = root / "campaign.json"
        self.campaign_file_sha = _canonical_file(self.campaign_path, self.campaign)

        self.checkpoint_rule = {
            "target_fraction": 0.5,
            "mapping_rule": "nearest actual candidate; ties choose earlier step",
        }
        self.discovery = {
            "outcome": "finalists_frozen",
            "finalist_family_ids": ["K1", "K0"],
            "checkpoint_rules": {"K1": self.checkpoint_rule},
        }
        self.discovery_path = root / "frozen-discovery.json"
        self.discovery_file_sha = _canonical_file(self.discovery_path, self.discovery)
        monkeypatch.setattr(
            krea_decision, "_validate_discovery_record", lambda value: value
        )
        raw_context = {
            "schema": 1,
            "kind": "forge-krea-exact-score-decision-context",
            "phase": "boundary",
            "frozen_discovery_decision": {
                "path": str(self.discovery_path),
                "sha256": self.discovery_file_sha,
            },
            "candidate_family_id": self.arm,
            "checkpoint_rule_sha256": krea_provenance.canonical_sha256(
                self.checkpoint_rule
            ),
            "selected_candidate": self.sealed_candidate,
            "decision_completed_before_export_reserve": True,
            "fallback_used": False,
        }
        self.decision_context = batch._validate_score_decision_context(
            raw_context, campaign=self.campaign
        )

        self.candidates = [
            self._candidate(
                candidate_id=self.sealed_candidate["candidate_id"],
                source_arm_id=self.arm,
                path=self.local_path,
                mode="local_run_candidate",
            ),
            self._candidate(
                candidate_id="zero-control",
                source_arm_id="K0",
                path=self.zero_path,
                mode="zero_lora_control",
            ),
        ]
        self.assets = {
            "diffusion_model": {
                "canonical_path": "/models/krea-base.safetensors",
                "sha256": _sha("base"),
                "bytes": 10,
            },
            "text_encoder": {
                "canonical_path": "/models/text.safetensors",
                "sha256": _sha("text"),
                "bytes": 11,
            },
            "vae": {
                "canonical_path": "/models/vae.safetensors",
                "sha256": _sha("vae"),
                "bytes": 12,
            },
        }
        self.comfy_runtime = {"distributions_sha256": _sha("comfy-runtime")}
        self.driver_runtime = {"distributions_sha256": _sha("driver-runtime")}
        evaluator_script = _CALIBRATION / "evaluate_krea_local.py"
        identity_module = _CALIBRATION / "krea_dataset_identity.py"
        common_training = {"kind": "test-common-training-envelope"}
        self.evaluator = {
            "comfy_root": str(self.comfy_root),
            "comfy_python": sys.executable,
            "god_root": str(self.god_root),
            "driver_python": sys.executable,
            "expected_god_commit": "a" * 40,
            "expected_comfy_commit": "b" * 40,
            "expected_tooling_commit": "c" * 40,
            "expected_evaluator_script_sha256": krea_provenance.file_sha256(
                evaluator_script
            ),
            "expected_dataset_identity_module_sha256": krea_provenance.file_sha256(
                identity_module
            ),
            "expected_eval_defaults": {
                "steps": 28,
                "cfg": 1.0,
                "denoise": 0.85,
                "generations": 1,
                "master_seed": 42,
                "text_weight": 0.5,
            },
            "expected_runtime_identity": {
                "comfy_python_identity_sha256": krea_provenance.canonical_sha256(
                    self.comfy_runtime
                ),
                "driver_python_identity_sha256": krea_provenance.canonical_sha256(
                    self.driver_runtime
                ),
            },
            "expected_assets": self.assets,
            "containment": {
                "mode": "systemd_transient_service",
                "term_grace_s": 0.5,
                "systemd_run_path": "/usr/bin/systemd-run",
                "systemctl_path": "/usr/bin/systemctl",
            },
            "_expected_dataset_identity": None,
            "_common_training_envelope": common_training,
            "_common_training_envelope_sha256": (
                krea_provenance.canonical_sha256(common_training)
            ),
            "_sealed_plan_approval_path": str(self.score_approval_path),
            "_sealed_plan_approval_sha256": self.score_approval_sha,
            "_sealed_plan_approval": {
                "decision": "approved",
                "reviewer_identity": "Riley Reviewer",
            },
            "_plan_payload_sha256": _sha("approved-score-plan-payload"),
            "_batch_runner_sha256": krea_provenance.file_sha256(
                _CALIBRATION / "batch_evaluate_krea.py"
            ),
            "_fixture_manifest_path": str(self.fixture_path),
            "_fixture_manifest_file_sha256": self.fixture_file_sha,
            "_fixture_approval_path": str(self.fixture_approval_path),
            "_fixture_approval_file_sha256": self.fixture_approval_sha,
            "_cross_fixture_review_path": str(self.cross_review_path),
            "_cross_fixture_review_file_sha256": self.cross_review_sha,
            "_campaign_manifest_path": str(self.campaign_path),
            "_campaign_manifest_file_sha256": self.campaign_file_sha,
            "_campaign_manifest_sha256": self.campaign["manifest_sha256"],
            "_decision_context": self.decision_context,
            "_training_run_envelopes": [
                {
                    "arm_id": self.arm,
                    "execution_plan_sha256": self.execution_sha,
                    "budget_plan": {"hard_budget_s": 1800},
                }
            ],
        }
        self.plan = {
            "schema": 2,
            "kind": "forge-krea-exact-score-plan",
            "test_contract": "producer-publication",
        }

        monkeypatch.setattr(
            batch,
            "_validate_plan",
            lambda _plan: (
                self.dataset,
                self.dataset_sha,
                self.candidates,
                self.evaluator,
            ),
        )
        monkeypatch.setattr(
            batch.krea_fixture, "validate_manifest", lambda value: value
        )
        monkeypatch.setattr(batch, "_run_contained", self._mock_evaluator)

    def _candidate(
        self, *, candidate_id: str, source_arm_id: str, path: Path, mode: str
    ) -> dict:
        digest = krea_provenance.file_sha256(path)
        binding_path = self.root / f"{candidate_id}.binding.json"
        binding_file_sha = _canonical_file(binding_path, {"mode": mode})
        if mode == "local_run_candidate":
            normalized = {
                "mode": mode,
                "binding_manifest_sha256": binding_file_sha,
                "execution_plan_sha256": self.execution_sha,
                "run_completion_sha256": self.completion_sha,
                "candidate_sha256": digest,
                "candidate_bytes": path.stat().st_size,
                "candidate_step": 50,
                "candidate_fraction": {"numerator": 50, "denominator": 100},
                "candidate_image_exposures": 50,
                "normalized_recipe": {},
            }
        else:
            normalized = {
                "mode": mode,
                "binding_manifest_sha256": binding_file_sha,
                "zero_control_manifest_sha256": self.zero_manifest_sha,
                "candidate_sha256": digest,
                "candidate_bytes": path.stat().st_size,
                "candidate_step": None,
                "candidate_fraction": None,
                "candidate_image_exposures": 0,
            }
        return {
            "id": candidate_id,
            "source_arm_id": source_arm_id,
            "path": path,
            "sha256": digest,
            "provenance": {"manifest_sha256": None},
            "provenance_path": binding_path,
            "provenance_file_sha256": binding_file_sha,
            "candidate_binding": normalized,
        }

    def _mock_evaluator(
        self, command: list[str], **_kwargs
    ) -> subprocess.CompletedProcess:
        def argument(flag: str) -> str:
            return command[command.index(flag) + 1]

        candidate = Path(argument("--candidate-path"))
        output = Path(argument("--output"))
        log = Path(f"{output}.comfy.log")
        log.write_bytes(b"clean mocked evaluator log\n")
        loss = 0.1 if candidate.name == self.zero_path.name else 0.08
        rows = [
            {
                "index": index,
                "row_id": f"row-{index:03d}",
                "prompt_sha256": _sha(f"prompt-{index}"),
                "generation_seed": index,
                "text_guided_loss": loss,
                "blank_prompt_loss": loss,
            }
            for index in range(24)
        ]
        source = {
            "god": {"commit": "a" * 40, "tree": "d" * 40},
            "comfyui": {"commit": "b" * 40, "tree": "e" * 40},
            "tooling_nodes": {"commit": "c" * 40, "tree": "f" * 40},
            "expected_commits": {
                "god": "a" * 40,
                "comfyui": "b" * 40,
                "tooling_nodes": "c" * 40,
            },
            "god_import_bindings": {},
            "workflow_path": "validator/evaluation/ComfyUI/workflows/krea2.json",
            "workflow_sha256": _sha("workflow"),
            "calibration_shim_sha256": krea_provenance.file_sha256(Path(command[1])),
            "comfy_main_sha256": _sha("comfy-main"),
        }
        runtime = {
            "fresh_comfy_process": True,
            "loopback": "127.0.0.1",
            "cache": "comfy_default_fresh_process",
            "database": "memory",
            "api_nodes_disabled": True,
            "isolated_input_output_temp_user": True,
            "offline_environment": True,
            "custom_node_allowlist": ["comfyui-tooling-nodes"],
            "python": self.comfy_runtime,
            "driver_python": self.driver_runtime,
            "comfy_log_sha256": krea_provenance.file_sha256(log),
            "comfy_log_bytes": log.stat().st_size,
        }
        result = {
            "schema": 2,
            "evaluator": "god_krea2_img2img_exact",
            "candidate": candidate.name,
            "candidate_sha256": krea_provenance.file_sha256(candidate),
            "candidate_bytes": candidate.stat().st_size,
            "staged_candidate_sha256": krea_provenance.file_sha256(candidate),
            "comfy_lora_name": (
                f"candidate-{krea_provenance.file_sha256(candidate)}.safetensors"
            ),
            "model_type": "krea2",
            "dataset": argument("--dataset"),
            "dataset_sha256": self.dataset_sha,
            "image_count": len(rows),
            "scored_rows": rows,
            "base_name": "krea-base.safetensors",
            "asset_sha256": {
                key: value["sha256"] for key, value in self.assets.items()
            },
            "asset_bytes": {key: value["bytes"] for key, value in self.assets.items()},
            "steps": 28,
            "cfg": 1.0,
            "denoise": 0.85,
            "generations": 1,
            "master_seed": 42,
            "seeds": [42 + index for index in range(len(rows))],
            "text_guided_losses": [loss] * len(rows),
            "blank_prompt_losses": [loss] * len(rows),
            "text_mean": loss,
            "blank_mean": loss,
            "text_weight": 0.5,
            "weighted_loss": loss,
            "direction": "min",
            "elapsed_s": 1.0,
            "source": source,
            "runtime": runtime,
        }
        output.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    def publish(self) -> tuple[Path, dict]:
        public_evaluator = {
            key: deepcopy(value)
            for key, value in self.evaluator.items()
            if not key.startswith("_")
        }
        public_evaluator["cache_provenance_sha256"] = _sha("cache-provenance")
        plan_candidates = [
            {
                "id": candidate["id"],
                "arm_id": candidate["source_arm_id"],
                "path": str(candidate["path"]),
                "sha256": candidate["sha256"],
                "candidate_binding": {
                    "path": str(candidate["provenance_path"]),
                    "sha256": candidate["provenance_file_sha256"],
                },
            }
            for candidate in self.candidates
        ]
        self.plan = {
            "schema": 2,
            "kind": "forge-krea-exact-score-plan",
            "dataset": {"path": str(self.dataset), "sha256": self.dataset_sha},
            "fixture_manifest": {
                "path": str(self.fixture_path),
                "sha256": self.fixture_file_sha,
            },
            "fixture_approval": {
                "path": str(self.fixture_approval_path),
                "sha256": self.fixture_approval_sha,
            },
            "cross_fixture_review": {
                "path": str(self.cross_review_path),
                "sha256": self.cross_review_sha,
            },
            "campaign_manifest": {
                "path": str(self.campaign_path),
                "sha256": self.campaign_file_sha,
            },
            "decision_context": {
                key: value
                for key, value in self.decision_context.items()
                if not key.startswith("_")
            },
            "candidates": plan_candidates,
            "evaluator": public_evaluator,
        }
        approval_candidates = [
            {
                "id": candidate["id"],
                "candidate_binding": {
                    "mode": candidate["candidate_binding"]["mode"],
                    "binding_manifest_sha256": candidate["candidate_binding"][
                        "binding_manifest_sha256"
                    ],
                },
            }
            for candidate in self.candidates
        ]
        approval_candidates.sort(key=lambda row: row["id"])
        approval = {
            "schema": 2,
            "kind": "forge-krea-exact-score-plan-approval",
            "decision": "approved",
            "reviewer_identity": "Riley Reviewer",
            **batch._v2_plan_approval_expected(
                self.plan,
                candidates=approval_candidates,
                evaluator=public_evaluator,
            ),
        }
        self.score_approval_sha = _canonical_file(self.score_approval_path, approval)
        self.plan["sealed_plan_approval"] = {
            "path": str(self.score_approval_path),
            "sha256": self.score_approval_sha,
        }
        _canonical_file(self.root / "score-plan.json", self.plan)
        self.evaluator["_sealed_plan_approval_sha256"] = self.score_approval_sha
        self.evaluator["_sealed_plan_approval"] = {
            "decision": "approved",
            "reviewer_identity": "Riley Reviewer",
        }
        self.evaluator["_plan_payload_sha256"] = batch._plan_payload_sha256(self.plan)
        output = self.root / "aggregate.json"
        aggregate = batch.run_batch(
            self.plan,
            results_dir=self.root / "results",
            output=output,
        )
        return output, aggregate

    def boundary_policy(self, aggregate: dict) -> dict:
        return {
            "phase": "confirmation",
            "candidate_family_id": self.arm,
            "discovery_plan": {"sha256": self.discovery_plan_sha},
            "discovery_decision": {
                "path": str(self.discovery_path),
                "sha256": self.discovery_file_sha,
            },
            "score_batches": [
                {
                    "batch_id": "B-0p5-small",
                    "phase": "boundary",
                    "fixture_id": "B-0p5-small",
                    "seed_role": "A",
                    "hours": 0.5,
                    "dataset_boundary": "small",
                    "plan_canonical_sha256": aggregate["plan"]["canonical_sha256"],
                    "campaign_manifest_sha256": self.campaign_file_sha,
                    "fixture_manifest_sha256": self.fixture_file_sha,
                    "fixture_approval_sha256": self.fixture_approval_sha,
                    "sealed_plan_approval_sha256": self.score_approval_sha,
                }
            ],
        }

    def patch_match_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            krea_decision,
            "_bound_plan_and_seal",
            lambda _policy: (
                {"arm_ids": [self.arm], "document": {}},
                {},
                {
                    "fixtures": [],
                    "cross_fixture_review_sha256": self.cross_review_sha,
                },
            ),
        )
        monkeypatch.setattr(
            krea_decision,
            "_binding",
            lambda *_args, **_kwargs: (
                self.discovery_path,
                self.discovery,
                self.discovery_file_sha,
            ),
        )


def test_schema2_publication_round_trips_into_decision_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = ProducerHarness(tmp_path, monkeypatch)
    output, aggregate = harness.publish()

    assert aggregate["campaign"]["runs"] == [harness.run]
    assert [row["arm_id"] for row in aggregate["campaign"]["runs"]] == ["K1"]
    zero = next(
        row for row in aggregate["candidates"] if row["mode"] == "zero_lora_control"
    )
    local = next(
        row for row in aggregate["candidates"] if row["mode"] == "local_run_candidate"
    )
    assert zero["zero_control_manifest_sha256"] == harness.zero_manifest_sha
    assert all(
        zero[key] is None
        for key in (
            "arm_id",
            "family_id",
            "execution_plan_sha256",
            "run_completion_sha256",
            "step",
            "fraction_numerator",
            "fraction_denominator",
            "image_exposures",
            "mechanics",
        )
    )
    assert local["zero_control_manifest_sha256"] is None
    assert aggregate["training_run_envelopes"][0]["candidate_decision"] == {
        "mode": "frozen_checkpoint_rule",
        "selected_candidate_sha256": harness.local_sha,
        "decision_completed_before_export_reserve": True,
        "fallback_used": False,
    }
    assert output.read_bytes() == krea_provenance.canonical_bytes(aggregate) + b"\n"

    campaign = krea_decision._validate_campaign_adapter(aggregate["campaign"])
    assert campaign["runs"] == [harness.run]
    normalized, file_sha = krea_decision._aggregate(output)
    assert file_sha == krea_provenance.file_sha256(output)
    assert normalized["campaign"]["runs"] == [harness.run]
    assert normalized["zero"]["zero_control_manifest_sha256"] == (
        harness.zero_manifest_sha
    )

    harness.patch_match_context(monkeypatch)
    observed, bindings = krea_decision._match_aggregates(
        policy=harness.boundary_policy(aggregate), aggregate_paths=[output]
    )
    assert list(observed) == ["B-0p5-small"]
    assert bindings[0]["aggregate_sha256"] == aggregate["aggregate_sha256"]


def test_decision_consumer_reloads_every_raw_result_and_fails_when_one_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = ProducerHarness(tmp_path, monkeypatch)
    output, aggregate = harness.publish()
    harness.patch_match_context(monkeypatch)
    manifest_path, manifest = _evidence_manifest(output)
    archive = manifest_path.parent
    result = archive / manifest["evaluator_results"][0]["path"]
    result.parent.chmod(0o700)
    result.unlink()

    with pytest.raises(ValueError, match="must be a regular non-symlink file"):
        krea_decision._match_aggregates(
            policy=harness.boundary_policy(aggregate), aggregate_paths=[output]
        )


def test_decision_consumer_rejects_raw_result_byte_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = ProducerHarness(tmp_path, monkeypatch)
    output, aggregate = harness.publish()
    harness.patch_match_context(monkeypatch)
    manifest_path, manifest = _evidence_manifest(output)
    result = manifest_path.parent / manifest["evaluator_results"][0]["path"]
    result.chmod(0o600)
    result.write_bytes(result.read_bytes() + b" \n")

    with pytest.raises(ValueError, match="evaluator result bytes do not match"):
        krea_decision._match_aggregates(
            policy=harness.boundary_policy(aggregate), aggregate_paths=[output]
        )


def test_rehashed_aggregate_semantic_forgery_is_rejected_by_raw_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = ProducerHarness(tmp_path, monkeypatch)
    output, aggregate = harness.publish()
    harness.patch_match_context(monkeypatch)
    forged = deepcopy(aggregate)
    target = forged["candidates"][0]
    for pair in target["paired_rows"]:
        pair["text_guided_loss"] = 0.123
        pair["blank_prompt_loss"] = 0.123
    target["text_mean"] = 0.123
    target["blank_mean"] = 0.123
    target["weighted_loss"] = 0.123
    forged = _reseal_aggregate(forged)
    forged_path = tmp_path / "forged-aggregate.json"
    _canonical_file(forged_path, forged)
    shutil.copytree(
        tmp_path / aggregate["decision_evidence"]["archive_path"],
        tmp_path / f"{forged_path.name}.evidence",
    )
    forged["decision_evidence"]["archive_path"] = f"{forged_path.name}.evidence"
    forged = _reseal_aggregate(forged)
    _canonical_file(forged_path, forged)

    with pytest.raises(ValueError, match="differs from its raw evaluator result"):
        krea_decision._match_aggregates(
            policy=harness.boundary_policy(forged), aggregate_paths=[forged_path]
        )


def test_rehashed_wrong_score_plan_is_rejected_by_original_human_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = ProducerHarness(tmp_path, monkeypatch)
    output, aggregate = harness.publish()
    harness.patch_match_context(monkeypatch)
    manifest_path, manifest = _evidence_manifest(output)
    plan_entry = manifest["score_plan"]
    plan_path = manifest_path.parent / plan_entry["path"]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["candidates"][0]["path"] = "/attacker/replaced-candidate.safetensors"
    plan_file_sha = _rewrite_canonical(plan_path, plan)
    plan_entry["file_sha256"] = plan_file_sha
    plan_entry["canonical_sha256"] = krea_provenance.canonical_sha256(plan)
    manifest_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = krea_provenance.canonical_sha256(manifest_body)
    manifest_file_sha = _rewrite_canonical(manifest_path, manifest)

    forged = deepcopy(aggregate)
    forged["plan"]["raw_sha256"] = plan_file_sha
    forged["plan"]["canonical_sha256"] = plan_entry["canonical_sha256"]
    forged["decision_evidence"]["manifest_file_sha256"] = manifest_file_sha
    forged["decision_evidence"]["manifest_sha256"] = manifest["manifest_sha256"]
    forged = _reseal_aggregate(forged)
    _rewrite_canonical(output, forged)

    with pytest.raises(
        ValueError, match="exact-score approval does not bind the complete batch plan"
    ):
        krea_decision._match_aggregates(
            policy=harness.boundary_policy(forged), aggregate_paths=[output]
        )


def test_aggregate_and_evidence_bundle_remain_verifiable_after_relocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    harness = ProducerHarness(source, monkeypatch)
    output, aggregate = harness.publish()
    harness.patch_match_context(monkeypatch)
    destination = tmp_path / "off-host-archive"
    destination.mkdir()
    relocated = destination / output.name
    shutil.copy2(output, relocated)
    shutil.copytree(
        output.parent / aggregate["decision_evidence"]["archive_path"],
        destination / aggregate["decision_evidence"]["archive_path"],
    )

    observed, bindings = krea_decision._match_aggregates(
        policy=harness.boundary_policy(aggregate), aggregate_paths=[relocated]
    )
    assert list(observed) == ["B-0p5-small"]
    assert bindings[0]["path"] == relocated.name
    assert all(
        not Path(row["path"]).is_absolute()
        for row in bindings[0]["decision_evidence"]["evaluator_results"]
    )


def test_consumer_rejects_campaign_cherry_pick_and_zero_manifest_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = ProducerHarness(tmp_path, monkeypatch)
    _output, aggregate = harness.publish()

    cherry = deepcopy(aggregate)
    cherry["campaign"]["runs"][0]["candidates"][0]["sha256"] = _sha(
        "invented-candidate"
    )
    cherry["campaign"] = _reseal_campaign_adapter(cherry["campaign"])
    cherry["campaign_manifest_sha256"] = cherry["campaign"]["file_sha256"]
    cherry = _reseal_aggregate(cherry)
    cherry_path = tmp_path / "cherry-pick.json"
    _canonical_file(cherry_path, cherry)
    with pytest.raises(ValueError, match="cherry-picks or invents"):
        krea_decision._aggregate(cherry_path)

    zero_tamper = deepcopy(aggregate)
    next(
        row for row in zero_tamper["candidates"] if row["mode"] == "zero_lora_control"
    )["zero_control_manifest_sha256"] = _sha("other-zero-manifest")
    zero_tamper = _reseal_aggregate(zero_tamper)
    zero_path = tmp_path / "zero-tamper.json"
    _canonical_file(zero_path, zero_tamper)
    with pytest.raises(ValueError, match="zero control differs"):
        krea_decision._aggregate(zero_path)


def test_boundary_decision_tamper_is_rejected_by_policy_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = ProducerHarness(tmp_path, monkeypatch)
    _output, aggregate = harness.publish()
    harness.patch_match_context(monkeypatch)
    policy = harness.boundary_policy(aggregate)

    for field, value in (
        ("selected_candidate_sha256", _sha("wrong-selection")),
        ("decision_completed_before_export_reserve", False),
        ("fallback_used", True),
    ):
        tampered = deepcopy(aggregate)
        tampered["training_run_envelopes"][0]["candidate_decision"][field] = value
        tampered = _reseal_aggregate(tampered)
        path = tmp_path / f"tampered-{field}.json"
        _canonical_file(path, tampered)
        with pytest.raises(ValueError, match="late, fallback-dependent, or unbound"):
            krea_decision._match_aggregates(
                policy=policy,
                aggregate_paths=[path],
            )


def test_explicit_boundary_context_blocks_phase_ambiguity_and_rule_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = ProducerHarness(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="boundary-ambiguous"):
        batch._validate_score_decision_context(
            {
                "schema": 1,
                "kind": "forge-krea-exact-score-decision-context",
                "phase": "discovery",
            },
            campaign=harness.campaign,
        )

    context = {
        key: value
        for key, value in harness.decision_context.items()
        if not key.startswith("_")
    }
    context["checkpoint_rule_sha256"] = _sha("different-checkpoint-rule")
    with pytest.raises(ValueError, match="checkpoint rule SHA-256 mismatch"):
        batch._validate_score_decision_context(context, campaign=harness.campaign)

    wrong_campaign_body = {
        key: deepcopy(value)
        for key, value in harness.campaign.items()
        if key != "manifest_sha256"
    }
    wrong_candidate = wrong_campaign_body["runs"][0]["candidates"][0]
    wrong_candidate["step"] = 40
    wrong_candidate["fraction"] = {"numerator": 40, "denominator": 100}
    wrong_campaign = batch.seal_campaign_manifest(wrong_campaign_body)
    context = {
        key: value
        for key, value in harness.decision_context.items()
        if not key.startswith("_")
    }
    context["selected_candidate"] = wrong_candidate
    with pytest.raises(ValueError, match="does not map the frozen checkpoint rule"):
        batch._validate_score_decision_context(context, campaign=wrong_campaign)
