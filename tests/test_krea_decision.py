"""Adversarial contract tests for the Week-5 Krea decision layer.

These tests intentionally build small, synthetic schema-2 score batches.  They
exercise the adapter boundary rather than trusting the batch producer: the
campaign ledger, score rows, fixture identity, policy, approvals, and decision
records all have to agree byte-for-byte.
"""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

import pytest

_CALIBRATION = Path(__file__).parents[1] / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))

import krea_decision  # noqa: E402
import krea_provenance  # noqa: E402


ARMS = ("K0", "K1", "K2", "K3", "K4", "K5")
PUBLIC = ("K2", "K3", "K4")
BOOTSTRAP = {
    "method": "paired_cluster_bootstrap",
    "cluster_unit": "task/concept",
    "confidence": 0.95,
    "resamples": 10_000,
    "seed": 42_565_431,
}
DISCOVERY_CONTRACT = {
    "paired_rows_required": True,
    "discovery_tie_band": 0.01,
    "cluster_unit": "task/concept",
    "bootstrap": "cluster-bootstrap by task/concept",
    "bootstrap_confidence": 0.95,
    "bootstrap_resamples": 10_000,
    "bootstrap_seed": 42_565_431,
    "material_rank_reversal_definition": (
        "any non-control pair switches order across D1/D2 with >0.01 "
        "relative-improvement separation in both directions"
    ),
    "checkpoint_tie_breaker": (
        "earliest actual step among candidates within 0.01 of best"
    ),
}
CONFIRMATION_CONTRACT = {
    "field_parity_noninferiority_cap": 0.01,
    "concept_regression_cap": 0.03,
    "minimum_point_estimate_wins_or_ties": 3,
    "point_win_or_tie_cap": 0.01,
    "strongest_public_reference_rule": (
        "minimum loss among exhaustive approved K2-K4 local public-family "
        "reproductions for the same "
        "concept and seed"
    ),
    "boundary_gate": "mechanics_only_natural_completion_upload_ready_clean",
}


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write(path: Path, value: dict) -> tuple[Path, str]:
    raw = krea_provenance.canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _binding(path: Path, digest: str) -> dict:
    return {"path": str(path), "sha256": digest}


def _agent_actor(name: str) -> dict:
    return krea_decision.krea_delegated_review_contract.actor(name)


def _custodian_actor() -> dict:
    return {
        "actor_class": "agent",
        "actor_id": "codex-week5-sealed-confirmation-custodian",
        "display_name": "Codex Week-5 sealed confirmation custodian (agent)",
        "role": "sealed_confirmation_custodian",
        "review_instance_id": "week5-krea-sealed-confirmation-custody-20260727-v2",
        "identity_assurance": (
            "self-declared-agent-identity-not-human-or-cryptographic-authentication"
        ),
    }


class Harness:
    profile_index_registry: dict[str, tuple[Path, dict, str]] = {}

    def __init__(self, root: Path):
        self.root = root
        source = Path("ops/calibration/week5/krea-discovery-plan.json")
        self.plan = json.loads(source.read_text())
        self.plan["status"] = "draft_blocked_pre_gpu"
        self.plan["gpu_execution_authorized"] = False
        identities = {
            fixture: _sha(f"fixture-internal-{fixture}")
            for fixture in ("C1", "C2", "C3", "C4")
        }
        self.plan["confirmation_contract"]["identities"] = identities
        self.plan_path, self.plan_sha = _write(root / "plan.json", self.plan)
        authorization_payload = {
            "schema": 2,
            "kind": "forge-krea-discovery-execution-authorization",
            "discovery_plan": {
                "path": str(self.plan_path),
                "file_sha256": self.plan_sha,
                "discovery_sha256": krea_provenance.canonical_sha256(self.plan),
            },
            "fixture_admission_envelope": {
                "path": str(root / "synthetic-admission.json"),
                "file_sha256": _sha("fixture-admission-file"),
                "envelope_sha256": _sha("fixture-admission"),
                "owner_ratification_sha256": _sha("owner-ratification"),
            },
            "execution_surface_policy_sha256": _sha("surface-policy"),
            "frozen_status": "draft_blocked_pre_gpu",
            "frozen_gpu_blockers": list(self.plan["gpu_blockers"]),
            "authorized_actions": [
                "bootstrap_timing_probe",
                "profile_indexed_discovery_execution",
            ],
            "authorized_scope": "stage1_discovery_only",
            "status": "sealed_executable",
            "gpu_blockers_closed_for_authorized_scope": True,
            "gpu_execution_authorized": False,
            "technical_reviewer_actor": {
                "actor_class": "agent",
                "actor_id": "codex-krea-runtime-reviewer",
                "display_name": "Codex Krea runtime reviewer",
                "role": "discovery_execution_authorization_reviewer",
                "review_instance_id": "decision-harness-authorization-review",
                "identity_assurance": (
                    "self-declared-agent-identity-not-human-or-cryptographic-"
                    "authentication"
                ),
            },
            "accountable_owner_identity": "Jordan Example",
            "authorized_at_utc": "2026-07-28T00:00:00Z",
            "claim_limit": "synthetic decision-flow fixture",
        }
        self.authorization = krea_decision.seal_discovery_execution_authorization(
            authorization_payload
        )
        self.authorization_path, self.authorization_sha = _write(
            root / "discovery-authorization.json", self.authorization
        )
        self.arm_classes = {
            row["id"]: row["throughput_equivalence_class"] for row in self.plan["arms"]
        }
        classes = sorted(set(self.arm_classes.values()))
        runtime_identity = {
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "data_parallel_replicas": 1,
            "resolution_policy_sha256": _sha("resolution-policy"),
            "precision_policy_sha256": _sha("precision-policy"),
            "cache_latents_to_disk": True,
            "cache_text_embeddings": True,
            "compile_enabled": False,
            "jit_enabled": True,
            "dataloader_workers": 2,
            "base_model_identity_sha256": _sha("base-model"),
            "runtime_identity_sha256": _sha("runtime"),
            "host_execution_identity_sha256": _sha("one-host"),
            "execution_surface": "staged_host_venv",
            "execution_scope": "discovery_only",
            "venv_tree_manifest_sha256": _sha("venv-tree"),
            "reference_container_image_sha256": _sha("container"),
            "gpu_identity_sha256": _sha("gpu"),
            "trainer_identity_sha256": _sha("trainer"),
            "measurement_tool_sha256": _sha("measurement-tool"),
        }
        campaign_runtime_identity = krea_provenance.canonical_sha256(runtime_identity)
        self.profile_envelopes: dict[str, dict[str, dict]] = {}
        indexed_fixtures = {}
        for fixture_id, training_pairs in (("D1", 18), ("D2", 36)):
            self.profile_envelopes[fixture_id] = {}
            profiles = {}
            for class_name in classes:
                class_fields = {
                    "network_rank": 64 if class_name.startswith("C-") else 32,
                    "network_alpha": 64 if class_name.startswith("C-") else 32,
                    "optimizer": (
                        "automagic" if class_name.startswith("C-") else "adamw8bit"
                    ),
                    "loss": "mae" if class_name.startswith("B-") else "mse",
                    "differential_guidance_enabled": True,
                    "guidance_scale": 3.0 if class_name.startswith("B-") else 2.0,
                }
                envelope = {
                    **runtime_identity,
                    **class_fields,
                    "equivalence_class": class_name,
                    "training_pair_count": training_pairs,
                    "training_dataset_shape_sha256": _sha(
                        f"training-shape-{fixture_id}"
                    ),
                    "execution_envelope_sha256": _sha(
                        f"execution-envelope-{fixture_id}-{class_name}"
                    ),
                }
                profile = {
                    "profile_sha256": _sha(f"profile-{fixture_id}-{class_name}"),
                    "execution_envelope": envelope,
                }
                profile_path, profile_file_sha = _write(
                    root / f"profile-{fixture_id}-{_sha(class_name)[:8]}.json",
                    profile,
                )
                self.profile_envelopes[fixture_id][class_name] = envelope
                profiles[class_name] = {
                    "path": str(profile_path),
                    "file_sha256": profile_file_sha,
                    "profile_sha256": profile["profile_sha256"],
                    "execution_envelope_sha256": envelope["execution_envelope_sha256"],
                    "campaign_runtime_identity_sha256": campaign_runtime_identity,
                }
            indexed_fixtures[fixture_id] = {
                "manifest": {
                    "path": str(root / f"fixture-{fixture_id}.json"),
                    "file_sha256": _sha(f"fixture-file-{fixture_id}"),
                    "manifest_sha256": _sha(f"admitted-fixture-{fixture_id}"),
                },
                "approval": {
                    "path": str(root / f"fixture-{fixture_id}.approval.json"),
                    "file_sha256": _sha(f"fixture-approval-{fixture_id}"),
                    "approval_sha256": _sha(f"fixture-approval-semantic-{fixture_id}"),
                },
                "concept_id": f"concept-{fixture_id}",
                "training_pair_count": training_pairs,
                "training_dataset_shape_sha256": _sha(f"training-shape-{fixture_id}"),
                "profiles": profiles,
            }
        index_body = {
            "schema": 2,
            "kind": "forge-krea-discovery-profile-index",
            "discovery_plan": {
                "path": str(self.plan_path),
                "file_sha256": self.plan_sha,
            },
            "discovery_execution_authorization": {
                "path": str(self.authorization_path),
                "file_sha256": self.authorization_sha,
                "authorization_sha256": self.authorization["authorization_sha256"],
            },
            "throughput_equivalence_classes": classes,
            "required_profile_count": 6,
            "cross_fixture_profile_reuse_forbidden": True,
            "campaign_runtime_identity_sha256": campaign_runtime_identity,
            "fixtures": indexed_fixtures,
            "gpu_execution_authorized": False,
        }
        self.profile_index = {
            **index_body,
            "index_sha256": krea_provenance.canonical_sha256(index_body),
        }
        self.profile_index_path, self.profile_index_file_sha = _write(
            root / "discovery-profile-index.json", self.profile_index
        )
        self.profile_index_binding = {
            "path": str(self.profile_index_path),
            "file_sha256": self.profile_index_file_sha,
            "index_sha256": self.profile_index["index_sha256"],
        }
        self.profile_index_registry[str(self.profile_index_path)] = (
            self.profile_index_path,
            self.profile_index,
            self.profile_index_file_sha,
        )
        seal_payload = {
            "schema": 1,
            "kind": "forge-krea-confirmation-fixture-commitments",
            "discovery_protocol_sha256": krea_decision._discovery_protocol_sha(
                self.plan
            ),
            "sealed_at_utc": "2026-07-28T00:00:00Z",
            "reviewer_identity": "Riley Reviewer",
            "sealed_before_discovery_unblinding": True,
            "cross_fixture_review_sha256": _sha("cross-fixture-review"),
            "fixtures": [
                {
                    "fixture_id": fixture,
                    "identity_commitment_sha256": identities[fixture],
                    "fixture_manifest_sha256": _sha(f"fixture-file-{fixture}"),
                    "fixture_approval_sha256": _sha(f"fixture-approval-{fixture}"),
                }
                for fixture in ("C1", "C2", "C3", "C4")
            ],
        }
        self.seal = krea_decision.seal_confirmation_fixture_commitments(seal_payload)
        self.seal_path, self.seal_sha = _write(root / "fixture-seal.json", self.seal)

    def _score_row(
        self,
        *,
        batch_id: str,
        family: str | None,
        step: int | None,
        denominator: int | None,
        loss: float,
        row_count: int,
    ) -> dict:
        zero = family is None
        candidate_id = f"{batch_id}-{'zero' if zero else family + '-' + str(step)}"
        paired = [
            {
                "row_id": f"row-{index:03d}",
                "prompt_sha256": _sha(f"{batch_id}-prompt-{index}"),
                "generation_seed": index,
                "text_guided_loss": loss,
                "blank_prompt_loss": loss,
            }
            for index in range(row_count)
        ]
        return {
            "candidate_id": candidate_id,
            "arm_id": None if zero else family,
            "mode": "zero_lora_control" if zero else "local_run_candidate",
            "family_id": None if zero else family,
            "candidate_sha256": _sha(f"candidate-{candidate_id}"),
            "candidate_bytes": 1024,
            "execution_plan_sha256": (
                None if zero else _sha(f"execution-{batch_id}-{family}")
            ),
            "run_completion_sha256": (
                None if zero else _sha(f"completion-{batch_id}-{family}")
            ),
            "step": step,
            "fraction_numerator": step,
            "fraction_denominator": denominator,
            "image_exposures": None if zero else step * 8,
            "binding_manifest_sha256": _sha(f"binding-{candidate_id}"),
            "zero_control_manifest_sha256": (
                _sha(f"zero-manifest-{batch_id}") if zero else None
            ),
            "result_file": f"{candidate_id}.json",
            "result_file_sha256": _sha(f"result-file-{candidate_id}"),
            "result_canonical_sha256": _sha(f"result-canonical-{candidate_id}"),
            "weighted_loss": loss,
            "text_mean": loss,
            "blank_mean": loss,
            "paired_rows": paired,
            "mechanics": (
                None
                if zero
                else {
                    "natural_completion": True,
                    "upload_ready": True,
                    "clean_telemetry": True,
                }
            ),
        }

    def aggregate(
        self,
        *,
        batch_id: str,
        phase: str,
        fixture_id: str,
        seed_role: str,
        arms: tuple[str, ...],
        losses: dict[str, float],
        hours: float | None = None,
        boundary: str | None = None,
        steps: tuple[int, ...] = (100, 500, 1000),
        denominator: int = 1000,
    ) -> tuple[Path, dict]:
        if fixture_id == "D1":
            training_pairs, evaluation_rows = 18, 24
        elif fixture_id == "D2":
            training_pairs, evaluation_rows = 36, 40
        elif fixture_id in {"C1", "C2", "C3", "C4"}:
            shape = self.plan["confirmation_contract"]["fixture_shape_contract"][
                fixture_id
            ]
            training_pairs = shape["training_pairs"]
            evaluation_rows = shape["evaluation_rows"]
        elif boundary == "small":
            training_pairs, evaluation_rows = 20, 24
        elif boundary == "large":
            training_pairs, evaluation_rows = 40, 40
        else:
            raise ValueError("synthetic aggregate has no fixture-count contract")
        candidates = [
            self._score_row(
                batch_id=batch_id,
                family=None,
                step=None,
                denominator=None,
                loss=0.1,
                row_count=evaluation_rows,
            )
        ]
        for family in arms:
            for step in steps:
                # Earlier checkpoints are meaningfully worse by default so the
                # synthetic best is the final; callers can use one-step batches.
                penalty = 0.03 if step == min(steps) and len(steps) > 1 else 0.0
                if len(steps) > 2 and step not in {min(steps), max(steps)}:
                    penalty = 0.015
                candidates.append(
                    self._score_row(
                        batch_id=batch_id,
                        family=family,
                        step=step,
                        denominator=denominator,
                        loss=losses[family] + penalty,
                        row_count=evaluation_rows,
                    )
                )
        candidates.sort(key=lambda row: row["candidate_id"])
        runs = []
        for family in sorted(arms):
            family_rows = [row for row in candidates if row["family_id"] == family]
            campaign_candidates = [
                {
                    "candidate_id": row["candidate_id"],
                    "sha256": row["candidate_sha256"],
                    "bytes": row["candidate_bytes"],
                    "step": row["step"],
                    "fraction": {
                        "numerator": row["fraction_numerator"],
                        "denominator": row["fraction_denominator"],
                    },
                }
                for row in family_rows
            ]
            campaign_candidates.sort(key=lambda row: (row["step"], row["sha256"]))
            runs.append(
                {
                    "arm_id": family,
                    "execution_plan_sha256": family_rows[0]["execution_plan_sha256"],
                    "run_completion_sha256": family_rows[0]["run_completion_sha256"],
                    "candidates": campaign_candidates,
                }
            )
        fixture_internal = (
            self.profile_index["fixtures"][fixture_id]["manifest"]["manifest_sha256"]
            if fixture_id in {"D1", "D2"}
            else self.plan["confirmation_contract"]["identities"].get(
                fixture_id, _sha(f"fixture-internal-{fixture_id}")
            )
        )
        fixture_file = (
            next(
                row["fixture_manifest_sha256"]
                for row in self.seal["fixtures"]
                if row["fixture_id"] == fixture_id
            )
            if fixture_id in {"C1", "C2", "C3", "C4"}
            else _sha(f"fixture-file-{fixture_id}")
        )
        fixture_approval = (
            next(
                row["fixture_approval_sha256"]
                for row in self.seal["fixtures"]
                if row["fixture_id"] == fixture_id
            )
            if fixture_id in {"C1", "C2", "C3", "C4"}
            else _sha(f"fixture-approval-{fixture_id}")
        )
        campaign_body = {
            "schema": 2,
            "kind": "forge-krea-exact-score-campaign",
            "fixture_manifest_sha256": fixture_internal,
            "discovery_plan_sha256": self.plan_sha,
            "runs": runs,
            "zero_control_manifest_sha256": _sha(f"zero-manifest-{batch_id}"),
            "decision_contract": DISCOVERY_CONTRACT,
            "confirmation_contract": CONFIRMATION_CONTRACT,
        }
        campaign_manifest_sha = krea_provenance.canonical_sha256(campaign_body)
        campaign_manifest = {
            **campaign_body,
            "manifest_sha256": campaign_manifest_sha,
        }
        campaign_file_sha = hashlib.sha256(
            krea_provenance.canonical_bytes(campaign_manifest) + b"\n"
        ).hexdigest()
        campaign = {
            "manifest_sha256": campaign_manifest_sha,
            "file_sha256": campaign_file_sha,
            "fixture_manifest_sha256": fixture_internal,
            "discovery_plan_sha256": self.plan_sha,
            "zero_control_manifest_sha256": campaign_body[
                "zero_control_manifest_sha256"
            ],
            "decision_contract": DISCOVERY_CONTRACT,
            "confirmation_contract": CONFIRMATION_CONTRACT,
            "runs": runs,
        }
        public_evaluator = {
            "expected_god_commit": "a" * 40,
            "expected_comfy_commit": "b" * 40,
            "expected_tooling_commit": "c" * 40,
            "expected_evaluator_script_sha256": _sha("evaluator-script"),
            "expected_dataset_identity_module_sha256": _sha("dataset-identity"),
            "expected_eval_defaults": {
                "steps": 28,
                "cfg": 1.0,
                "denoise": 0.85,
                "generations": 1,
                "master_seed": 42,
                "text_weight": 1.0,
            },
            "expected_runtime_identity": {
                "comfy_python_identity_sha256": _sha("comfy-runtime"),
                "driver_python_identity_sha256": _sha("driver-runtime"),
            },
            "expected_assets": {},
            "cache_provenance_sha256": _sha("cache-provenance"),
            "containment": {"term_grace_s": 1.0},
        }
        plan_candidates = [
            {
                "id": row["candidate_id"],
                "arm_id": row["family_id"] or "K0",
                "path": f"artifacts/{row['candidate_id']}.safetensors",
                "sha256": row["candidate_sha256"],
                "candidate_binding": {
                    "path": f"bindings/{row['candidate_id']}.json",
                    "sha256": row["binding_manifest_sha256"],
                },
            }
            for row in candidates
        ]
        score_plan = {
            "schema": 2,
            "kind": "forge-krea-exact-score-plan",
            "dataset": {
                "path": f"datasets/{fixture_id}",
                "sha256": _sha(f"eval-{fixture_id}"),
            },
            "fixture_manifest": {
                "path": f"fixtures/{fixture_id}.json",
                "sha256": fixture_file,
            },
            "fixture_approval": {
                "path": f"fixtures/{fixture_id}.approval.json",
                "sha256": fixture_approval,
            },
            "cross_fixture_review": {
                "path": "fixtures/cross-review.json",
                "sha256": self.seal["cross_fixture_review_sha256"],
            },
            "campaign_manifest": {
                "path": f"campaigns/{batch_id}.json",
                "sha256": campaign_file_sha,
            },
            "decision_context": {"phase": phase},
            "candidates": plan_candidates,
            "evaluator": public_evaluator,
        }
        approval_candidates = [
            {
                "id": row["candidate_id"],
                "candidate_binding": {
                    "mode": row["mode"],
                    "binding_manifest_sha256": row["binding_manifest_sha256"],
                },
            }
            for row in candidates
        ]
        approval_candidates.sort(key=lambda row: row["id"])
        plan_approval = {
            "schema": 2,
            "kind": "forge-krea-exact-score-plan-approval",
            "decision": "approved",
            "reviewer_identity": "Skyler Auditor",
            **krea_decision.krea_batch._v2_plan_approval_expected(
                score_plan,
                candidates=approval_candidates,
                evaluator=public_evaluator,
            ),
        }
        approval_path, plan_approval_sha = _write(
            self.root / f"score-plan-approval-{batch_id}.json", plan_approval
        )
        score_plan["sealed_plan_approval"] = {
            "path": str(approval_path),
            "sha256": plan_approval_sha,
        }
        plan_path, plan_file_sha = _write(
            self.root / f"score-plan-{batch_id}.json", score_plan
        )
        plan_sha = krea_provenance.canonical_sha256(score_plan)
        campaign_manifest_file_sha = campaign_file_sha
        training_envelopes = [
            {
                "arm_id": run["arm_id"],
                "execution_plan_sha256": run["execution_plan_sha256"],
                **(
                    {
                        "execution_envelope": self.profile_envelopes[fixture_id][
                            self.arm_classes[run["arm_id"]]
                        ],
                        "throughput_profile_sha256": self.profile_index["fixtures"][
                            fixture_id
                        ]["profiles"][self.arm_classes[run["arm_id"]]][
                            "profile_sha256"
                        ],
                    }
                    if fixture_id in {"D1", "D2"}
                    else {}
                ),
                "budget_plan": {"hard_budget_s": 2700},
                "budget_plan_sha256": _sha(f"budget-{batch_id}-{run['arm_id']}"),
                "schedule": {"mode": "natural_completion"},
                **(
                    {
                        "candidate_decision": {
                            "mode": "frozen_checkpoint_rule",
                            "selected_candidate_sha256": next(
                                row["candidate_sha256"]
                                for row in candidates
                                if row["family_id"] == run["arm_id"]
                            ),
                            "decision_completed_before_export_reserve": True,
                            "fallback_used": False,
                        }
                    }
                    if phase == "boundary"
                    else {}
                ),
            }
            for run in runs
        ]
        body = {
            "schema": 2,
            "kind": "forge-krea-exact-score-batch",
            "coverage": {
                "planned": len(candidates),
                "completed": len(candidates),
                "complete": True,
            },
            "direction": "min",
            "plan": {
                "raw_sha256": plan_file_sha,
                "canonical_sha256": plan_sha,
                "approved_payload_sha256": plan_approval["plan_payload_sha256"],
            },
            "campaign_manifest_sha256": campaign_manifest_file_sha,
            "fixture_manifest_sha256": fixture_file,
            "fixture_approval_sha256": fixture_approval,
            "sealed_plan_approval_sha256": plan_approval_sha,
            "sealed_plan_approval": {
                "decision": "approved",
                "reviewer_identity": "Skyler Auditor",
            },
            "batch_runner_sha256": plan_approval["batch_runner_sha256"],
            "evaluation_envelope": {"text_weight": 1.0},
            "fixture_contract": {
                "fixture_manifest_identity_sha256": fixture_internal,
                "training_pair_count": training_pairs,
                "evaluation_row_count": evaluation_rows,
                "training_dataset_sha256": _sha(f"train-{fixture_id}"),
                "evaluation_dataset_sha256": _sha(f"eval-{fixture_id}"),
                "cross_fixture_review_sha256": self.seal["cross_fixture_review_sha256"],
            },
            "campaign": campaign,
            "fixture": {
                "manifest_sha256": fixture_internal,
                "file_sha256": fixture_file,
                "concept_id": f"concept-{fixture_id}",
                "experimental_role": fixture_id,
                "evaluation_dataset_sha256": _sha(f"eval-{fixture_id}"),
            },
            "training_run_envelopes": training_envelopes,
            "candidates": candidates,
        }
        result_root = self.root / f"raw-results-{batch_id}"
        result_root.mkdir()
        completed = []
        for row in candidates:
            result = {
                "schema": 2,
                "evaluator": "god_krea2_img2img_exact",
                "candidate_sha256": row["candidate_sha256"],
                "candidate_bytes": row["candidate_bytes"],
                "model_type": "krea2",
                "dataset_sha256": _sha(f"eval-{fixture_id}"),
                "image_count": len(row["paired_rows"]),
                "scored_rows": row["paired_rows"],
                "text_mean": row["text_mean"],
                "blank_mean": row["blank_mean"],
                "text_weight": 1.0,
                "weighted_loss": row["weighted_loss"],
                "direction": "min",
            }
            result_path, result_file_sha = _write(
                result_root / row["result_file"], result
            )
            result_canonical_sha = krea_provenance.canonical_sha256(result)
            row["result_file_sha256"] = result_file_sha
            row["result_canonical_sha256"] = result_canonical_sha
            completed.append(
                {
                    "candidate_id": row["candidate_id"],
                    "result_file": row["result_file"],
                    "result_file_sha256": result_file_sha,
                    "result_canonical_sha256": result_canonical_sha,
                    "_result_path": str(result_path),
                }
            )
        aggregate_path = self.root / f"aggregate-{batch_id}.json"
        body["decision_evidence"] = (
            krea_decision.krea_batch._publish_decision_evidence_bundle(
                output=aggregate_path,
                plan=score_plan,
                plan_raw=krea_provenance.canonical_bytes(score_plan) + b"\n",
                approval_path=approval_path,
                completed=completed,
            )
        )
        aggregate = {
            **body,
            "aggregate_sha256": krea_provenance.canonical_sha256(body),
        }
        path, _ = _write(aggregate_path, aggregate)
        score_batch = {
            "batch_id": batch_id,
            "phase": phase,
            "fixture_id": fixture_id,
            "seed_role": seed_role,
            "seed": 42_565_431 if seed_role == "A" else 309_817_421,
            "hours": hours,
            "dataset_boundary": boundary,
            "plan_canonical_sha256": plan_sha,
            "sealed_plan_approval_sha256": plan_approval_sha,
            "campaign_manifest_sha256": campaign_manifest_file_sha,
            "fixture_manifest_sha256": fixture_file,
            "fixture_approval_sha256": fixture_approval,
        }
        return path, score_batch

    def discovery_case(
        self,
        *,
        losses_by_fixture: dict[str, dict[str, float]],
        include_seed_b_policy: bool = False,
        include_seed_b_aggregates: bool = False,
        decision_time: str = "2026-07-28T00:02:00Z",
    ) -> tuple[dict, list[Path], Path, Path]:
        paths = []
        batches = []
        for fixture in ("D1", "D2"):
            path, batch = self.aggregate(
                batch_id=f"{fixture}-A",
                phase="discovery",
                fixture_id=fixture,
                seed_role="A",
                arms=ARMS,
                losses=losses_by_fixture[fixture],
            )
            paths.append(path)
            batches.append(batch)
        if include_seed_b_policy:
            for fixture in ("D1", "D2"):
                path, batch = self.aggregate(
                    batch_id=f"{fixture}-B",
                    phase="discovery",
                    fixture_id=fixture,
                    seed_role="B",
                    arms=ARMS,
                    losses=losses_by_fixture[fixture],
                )
                batches.append(batch)
                if include_seed_b_aggregates:
                    paths.append(path)
        batches.sort(key=lambda row: row["batch_id"])
        payload = {
            "schema": 2,
            "kind": "forge-krea-discovery-decision-policy",
            "phase": "discovery",
            "prepared_by": "Priya Engineer",
            "discovery_plan": _binding(self.plan_path, self.plan_sha),
            "discovery_execution_authorization": _binding(
                self.authorization_path, self.authorization_sha
            ),
            "discovery_profile_index": dict(self.profile_index_binding),
            "confirmation_fixture_seal": _binding(self.seal_path, self.seal_sha),
            "score_batches": batches,
            "bootstrap": BOOTSTRAP,
        }
        policy = krea_decision.seal_discovery_policy(payload)
        policy_path, _ = _write(self.root / "discovery-policy.json", policy)
        approval = krea_decision.build_approval(
            policy,
            reviewer_identity="Morgan Auditor",
            approved_at_utc="2026-07-28T00:01:00Z",
        )
        approval_path, _ = _write(self.root / "discovery-approval.json", approval)
        output = self.root / "krea-discovery-decision.json"
        record = krea_decision.decide_discovery(
            policy_path=policy_path,
            approval_path=approval_path,
            aggregate_paths=paths,
            output=output,
            decided_at_utc=decision_time,
        )
        return record, paths, policy_path, approval_path

    def confirmation_case(
        self,
        discovery: dict,
        *,
        quality_losses: dict[str, float],
        omit_batch: str | None = None,
        decision_time: str = "2026-07-28T00:04:00Z",
    ) -> tuple[dict, list[Path], dict]:
        discovery_path, discovery_sha = _write(
            self.root / "frozen-discovery.json", discovery
        )
        candidate = next(
            family for family in discovery["finalist_family_ids"] if family != "K0"
        )
        quality_arms = tuple(
            sorted(set(discovery["finalist_family_ids"]) | set(PUBLIC) | {"K0"})
        )
        paths = []
        batches = []
        for fixture in ("C1", "C2", "C3", "C4"):
            for seed_role in ("A", "B"):
                batch_id = f"{fixture}-{seed_role}"
                path, batch = self.aggregate(
                    batch_id=batch_id,
                    phase="confirmation",
                    fixture_id=fixture,
                    seed_role=seed_role,
                    arms=quality_arms,
                    losses=quality_losses,
                    hours=0.75,
                    steps=(900, 1000),
                )
                batches.append(batch)
                if batch_id != omit_batch:
                    paths.append(path)
        for hours in (0.5, 0.75, 1.0):
            for boundary in ("small", "large"):
                hour_label = str(hours).replace(".", "p")
                batch_id = f"B-{hour_label}-{boundary}"
                path, batch = self.aggregate(
                    batch_id=batch_id,
                    phase="boundary",
                    fixture_id=batch_id,
                    seed_role="A",
                    arms=(candidate,),
                    losses={candidate: quality_losses[candidate]},
                    hours=hours,
                    boundary=boundary,
                    steps=(900,),
                )
                batches.append(batch)
                if batch_id != omit_batch:
                    paths.append(path)
        batches.sort(key=lambda row: row["batch_id"])
        payload = {
            "schema": 2,
            "kind": "forge-krea-confirmation-decision-policy",
            "phase": "confirmation",
            "prepared_by": "Priya Engineer",
            "discovery_plan": _binding(self.plan_path, self.plan_sha),
            "discovery_execution_authorization": _binding(
                self.authorization_path, self.authorization_sha
            ),
            "discovery_profile_index": dict(self.profile_index_binding),
            "confirmation_fixture_seal": _binding(self.seal_path, self.seal_sha),
            "discovery_decision": _binding(discovery_path, discovery_sha),
            "candidate_family_id": candidate,
            "score_batches": batches,
            "public_reference_family_ids": list(PUBLIC),
            "deployed_control_family_id": "K0",
            "bootstrap": BOOTSTRAP,
        }
        policy = krea_decision.seal_confirmation_policy(payload)
        policy_path, _ = _write(self.root / "confirmation-policy.json", policy)
        approval = krea_decision.build_approval(
            policy,
            reviewer_identity="Morgan Auditor",
            approved_at_utc="2026-07-28T00:03:00Z",
        )
        approval_path, _ = _write(self.root / "confirmation-approval.json", approval)
        record = krea_decision.decide_confirmation(
            policy_path=policy_path,
            approval_path=approval_path,
            aggregate_paths=paths,
            output=self.root / "krea-confirmation-decision.json",
            decided_at_utc=decision_time,
        )
        return record, paths, policy


@pytest.fixture
def harness(tmp_path: Path, monkeypatch) -> Harness:
    # Decision-policy tests exercise the policy math and exact authorization
    # binding.  Full admission-bundle rederivation is covered independently by
    # test_krea_discovery_authorization.py.
    monkeypatch.setattr(
        krea_decision.krea_discovery_authorization,
        "validate",
        lambda value: value,
    )
    result = Harness(tmp_path)
    monkeypatch.setattr(
        krea_decision.krea_profile_index,
        "load_binding",
        lambda value: Harness.profile_index_registry[value["path"]],
    )
    return result


def _enable_agent_decision_governance(
    harness: Harness, monkeypatch, *, custodian: dict | None = None
) -> None:
    custodian = deepcopy(custodian or _custodian_actor())
    authorization_body = {
        key: value
        for key, value in harness.authorization.items()
        if key != "authorization_sha256"
    }
    authorization_body["authorized_actions"] = [
        "bootstrap_timing_probe",
        "profile_indexed_discovery_execution",
        "offline_exact_scoring",
        "discovery_decision_evaluation",
    ]
    harness.authorization = {
        **authorization_body,
        "authorization_sha256": krea_provenance.canonical_sha256(authorization_body),
    }
    harness.authorization_path, harness.authorization_sha = _write(
        harness.authorization_path, harness.authorization
    )
    index_body = {
        key: deepcopy(value)
        for key, value in harness.profile_index.items()
        if key != "index_sha256"
    }
    index_body["discovery_execution_authorization"] = {
        "path": str(harness.authorization_path),
        "file_sha256": harness.authorization_sha,
        "authorization_sha256": harness.authorization["authorization_sha256"],
    }
    harness.profile_index = {
        **index_body,
        "index_sha256": krea_provenance.canonical_sha256(index_body),
    }
    harness.profile_index_path, harness.profile_index_file_sha = _write(
        harness.profile_index_path, harness.profile_index
    )
    harness.profile_index_binding = {
        "path": str(harness.profile_index_path),
        "file_sha256": harness.profile_index_file_sha,
        "index_sha256": harness.profile_index["index_sha256"],
    }
    harness.profile_index_registry[str(harness.profile_index_path)] = (
        harness.profile_index_path,
        harness.profile_index,
        harness.profile_index_file_sha,
    )
    admission = {
        "accountable_owner_identity": "Jordan Example",
        "governance": {"sealed_custodian_actor": {"actor": custodian}},
    }
    ratification = {"ratification_sha256": _sha("owner-ratification")}
    monkeypatch.setattr(
        krea_decision.krea_discovery_authorization,
        "_load_admission_binding",
        lambda _value: (
            harness.root / "synthetic-admission.json",
            admission,
            _sha("fixture-admission-file"),
            ratification,
            {},
        ),
    )
    seal_payload = {
        "schema": 2,
        "kind": "forge-krea-agent-confirmation-fixture-commitments",
        "discovery_protocol_sha256": krea_decision._discovery_protocol_sha(
            harness.plan
        ),
        "sealed_at_utc": "2026-07-28T00:00:30Z",
        "technical_custodian_actor": custodian,
        "accountable_owner_identity": "Jordan Example",
        "owner_ratification_sha256": _sha("owner-ratification"),
        "discovery_execution_authorization": _binding(
            harness.authorization_path, harness.authorization_sha
        ),
        "agent_review_is_not_human_review": True,
        "sealed_before_discovery_unblinding": True,
        "cross_fixture_review_sha256": _sha("cross-fixture-review"),
        "fixtures": [
            {
                "fixture_id": fixture,
                "identity_commitment_sha256": harness.plan["confirmation_contract"][
                    "identities"
                ][fixture],
                "fixture_manifest_sha256": _sha(f"fixture-file-{fixture}"),
                "fixture_approval_sha256": _sha(f"fixture-approval-{fixture}"),
            }
            for fixture in ("C1", "C2", "C3", "C4")
        ],
    }
    harness.seal = krea_decision.seal_confirmation_fixture_commitments(seal_payload)
    harness.seal_path, harness.seal_sha = _write(harness.seal_path, harness.seal)


def _agent_discovery_case(
    harness: Harness,
    *,
    losses_by_fixture: dict[str, dict[str, float]],
) -> tuple[dict, list[Path], dict, dict]:
    paths = []
    batches = []
    for fixture in ("D1", "D2"):
        path, batch = harness.aggregate(
            batch_id=f"{fixture}-A",
            phase="discovery",
            fixture_id=fixture,
            seed_role="A",
            arms=ARMS,
            losses=losses_by_fixture[fixture],
        )
        paths.append(path)
        batches.append(batch)
    payload = {
        "schema": 3,
        "kind": "forge-krea-agent-discovery-decision-policy",
        "phase": "discovery",
        "technical_preparer_actor": _agent_actor("discovery_decision_policy_preparer"),
        "accountable_owner_identity": "Jordan Example",
        "owner_ratification_sha256": _sha("owner-ratification"),
        "fixture_admission_envelope": dict(
            harness.authorization["fixture_admission_envelope"]
        ),
        "discovery_plan": _binding(harness.plan_path, harness.plan_sha),
        "discovery_execution_authorization": _binding(
            harness.authorization_path, harness.authorization_sha
        ),
        "discovery_profile_index": dict(harness.profile_index_binding),
        "confirmation_fixture_seal": _binding(harness.seal_path, harness.seal_sha),
        "score_batches": sorted(batches, key=lambda row: row["batch_id"]),
        "bootstrap": BOOTSTRAP,
        "delegated_review_contract": (
            krea_decision.krea_delegated_review_contract.binding()
        ),
        "agent_review_is_not_human_review": True,
    }
    policy = krea_decision.seal_discovery_policy(payload)
    policy_path, _ = _write(harness.root / "agent-discovery-policy.json", policy)
    approval = krea_decision.build_approval(
        policy,
        technical_reviewer_actor=_agent_actor("discovery_decision_reviewer"),
        approved_at_utc="2026-07-28T00:01:00Z",
    )
    approval_path, _ = _write(harness.root / "agent-discovery-approval.json", approval)
    record = krea_decision.decide_discovery(
        policy_path=policy_path,
        approval_path=approval_path,
        aggregate_paths=paths,
        output=harness.root / "krea-discovery-decision-agent.json",
        decided_at_utc="2026-07-28T00:02:00Z",
    )
    return record, paths, policy, approval


def test_discovery_freeze_rejects_status_blocker_and_dummy_profile_rewrites(
    harness: Harness,
):
    for mutation, message in (
        (
            lambda plan: plan.__setitem__("status", "ready"),
            "status",
        ),
        (
            lambda plan: plan.__setitem__("gpu_blockers", []),
            "blockers",
        ),
        (
            lambda plan: plan["budget_contract"][
                "throughput_profiles_by_equivalence_class"
            ].__setitem__("A-rank32-adamw8bit-mse-guidance2", _sha("dummy-profile")),
            "deferred index sentinel",
        ),
        (
            lambda plan: plan["arms"][0].__setitem__(
                "throughput_profile_sha256", _sha("late-flat-profile")
            ),
            "legacy flat hash",
        ),
    ):
        changed = deepcopy(harness.plan)
        mutation(changed)
        with pytest.raises(ValueError, match=message):
            krea_decision._validate_discovery_plan(changed)


def test_decision_binds_index_without_rewriting_the_blocked_freeze(harness: Harness):
    before = harness.plan_path.read_bytes()
    record, _, policy_path, _ = harness.discovery_case(
        losses_by_fixture=_agreement_losses()
    )
    policy = json.loads(policy_path.read_text())

    assert harness.plan_path.read_bytes() == before
    assert harness.plan["gpu_execution_authorized"] is False
    assert all(
        row.get("throughput_profile_sha256") is None
        and row.get("fixture_manifest_sha256") is None
        and row.get("fixture_approval_sha256") is None
        for row in harness.plan["arms"]
    )
    assert policy["discovery_profile_index"] == harness.profile_index_binding
    assert (
        record["discovery_profile_index_sha256"]
        == harness.profile_index["index_sha256"]
    )
    assert (
        record["discovery_profile_index_file_sha256"] == harness.profile_index_file_sha
    )


def test_discovery_policy_rejects_nonindexed_fixture_approval(harness: Harness):
    _, _, policy_path, _ = harness.discovery_case(losses_by_fixture=_agreement_losses())
    body = json.loads(policy_path.read_text())
    body.pop("policy_sha256")
    body["score_batches"][0]["fixture_approval_sha256"] = _sha("unindexed-approval")

    with pytest.raises(ValueError, match="indexed exact fixture approval"):
        krea_decision.seal_discovery_policy(body)


@pytest.mark.parametrize("drift", ["class", "host"])
def test_decision_rejects_profile_class_and_one_host_drift(
    harness: Harness, drift: str
):
    _, aggregate_paths, policy_path, approval_path = harness.discovery_case(
        losses_by_fixture=_agreement_losses()
    )
    aggregate_path = aggregate_paths[0]
    aggregate = json.loads(aggregate_path.read_text())
    envelope = aggregate["training_run_envelopes"][0]["execution_envelope"]
    if drift == "class":
        envelope["equivalence_class"] = "B-rank32-adamw8bit-mae-guidance3"
    else:
        envelope["host_execution_identity_sha256"] = _sha("foreign-host")
    body = {key: value for key, value in aggregate.items() if key != "aggregate_sha256"}
    aggregate = {
        **body,
        "aggregate_sha256": krea_provenance.canonical_sha256(body),
    }
    _write(aggregate_path, aggregate)

    with pytest.raises(ValueError, match="escaped indexed"):
        krea_decision.decide_discovery(
            policy_path=policy_path,
            approval_path=approval_path,
            aggregate_paths=aggregate_paths,
            output=harness.root / f"krea-discovery-decision-{drift}-drift.json",
            decided_at_utc="2026-07-28T00:03:00Z",
        )


def _agreement_losses() -> dict[str, dict[str, float]]:
    return {
        "D1": {
            "K0": 0.096,
            "K1": 0.070,
            "K2": 0.075,
            "K3": 0.080,
            "K4": 0.085,
            "K5": 0.090,
        },
        "D2": {
            "K0": 0.097,
            "K1": 0.071,
            "K2": 0.076,
            "K3": 0.081,
            "K4": 0.086,
            "K5": 0.091,
        },
    }


def test_agent_decision_governance_closes_custody_policy_approval_and_record(
    harness: Harness, monkeypatch
):
    _enable_agent_decision_governance(harness, monkeypatch)
    record, _, policy, approval = _agent_discovery_case(
        harness, losses_by_fixture=_agreement_losses()
    )

    assert harness.seal["schema"] == 2
    assert harness.seal["technical_custodian_actor"] == _custodian_actor()
    assert policy["schema"] == 3
    assert policy["technical_preparer_actor"] == _agent_actor(
        "discovery_decision_policy_preparer"
    )
    assert approval["technical_reviewer_actor"] == _agent_actor(
        "discovery_decision_reviewer"
    )
    assert record["schema"] == 3
    assert record["kind"] == "forge-krea-agent-discovery-decision-record"
    assert record["decision_reviewer_actor"] == approval["technical_reviewer_actor"]
    assert record["accountable_owner_identity"] == "Jordan Example"
    assert record["agent_review_is_not_human_review"] is True
    assert "decision_reviewer_identity" not in record

    # The unchanged legacy confirmation path must be able to consume the
    # agent-governed discovery record without relabeling its actor as human.
    quality = {
        "K0": 0.100,
        "K1": 0.080,
        "K2": 0.082,
        "K3": 0.084,
        "K4": 0.086,
    }
    confirmation, _, _ = harness.confirmation_case(record, quality_losses=quality)
    assert confirmation["schema"] == 2
    assert confirmation["discovery_decision_sha256"]


def test_agent_fixture_seal_rejects_wrong_custodian_owner_action_and_chronology(
    harness: Harness, monkeypatch
):
    _enable_agent_decision_governance(harness, monkeypatch)
    body = {key: value for key, value in harness.seal.items() if key != "seal_sha256"}

    wrong_custodian = deepcopy(body)
    wrong_custodian["technical_custodian_actor"]["actor_id"] = "other-custodian"
    with pytest.raises(ValueError, match="owner-ratified custodian"):
        krea_decision.seal_confirmation_fixture_commitments(wrong_custodian)

    wrong_owner = deepcopy(body)
    wrong_owner["owner_ratification_sha256"] = _sha("other-ratification")
    with pytest.raises(ValueError, match="owner-ratified custodian"):
        krea_decision.seal_confirmation_fixture_commitments(wrong_owner)

    pre_authorization = deepcopy(body)
    pre_authorization["sealed_at_utc"] = "2026-07-28T00:00:00Z"
    with pytest.raises(ValueError, match="sealed after authorization"):
        krea_decision.seal_confirmation_fixture_commitments(pre_authorization)

    authorization = deepcopy(harness.authorization)
    authorization_body = {
        key: value
        for key, value in authorization.items()
        if key != "authorization_sha256"
    }
    authorization_body["authorized_actions"].remove("discovery_decision_evaluation")
    authorization = {
        **authorization_body,
        "authorization_sha256": krea_provenance.canonical_sha256(authorization_body),
    }
    _, file_sha = _write(harness.authorization_path, authorization)
    missing_action = deepcopy(body)
    missing_action["discovery_execution_authorization"]["sha256"] = file_sha
    with pytest.raises(ValueError, match="does not permit"):
        krea_decision.seal_confirmation_fixture_commitments(missing_action)


def test_agent_policy_and_approval_reject_actor_reuse_and_tampering(
    harness: Harness, monkeypatch
):
    _enable_agent_decision_governance(harness, monkeypatch)
    _, _, policy, approval = _agent_discovery_case(
        harness, losses_by_fixture=_agreement_losses()
    )

    policy_body = {
        key: value for key, value in policy.items() if key != "policy_sha256"
    }
    wrong_preparer = deepcopy(policy_body)
    wrong_preparer["technical_preparer_actor"][
        "review_instance_id"
    ] = "unratified-review-instance"
    with pytest.raises(ValueError, match="owner-ratified delegated actor"):
        krea_decision.seal_discovery_policy(wrong_preparer)

    with pytest.raises(ValueError, match="prebound technical reviewer"):
        krea_decision.build_approval(
            policy,
            reviewer_identity="Morgan Auditor",
            approved_at_utc="2026-07-28T00:01:00Z",
        )

    reused = deepcopy(_agent_actor("discovery_decision_reviewer"))
    reused["actor_id"] = policy["technical_preparer_actor"]["actor_id"]
    with pytest.raises(ValueError, match="owner-ratified delegated actor"):
        krea_decision.build_approval(
            policy,
            technical_reviewer_actor=reused,
            approved_at_utc="2026-07-28T00:01:00Z",
        )

    tampered = deepcopy(approval)
    tampered["accountable_owner_identity"] = "Taylor Other"
    tampered_body = {
        key: value for key, value in tampered.items() if key != "approval_sha256"
    }
    tampered["approval_sha256"] = krea_provenance.canonical_sha256(tampered_body)
    with pytest.raises(ValueError, match="owner-ratified authority"):
        krea_decision.validate_approval(tampered, policy=policy)


def test_agent_record_rejects_self_rehashed_actor_and_owner_tampering(
    harness: Harness, monkeypatch
):
    _enable_agent_decision_governance(harness, monkeypatch)
    record, _, _, _ = _agent_discovery_case(
        harness, losses_by_fixture=_agreement_losses()
    )

    for field, value, message in (
        (
            "decision_reviewer_actor",
            {
                **record["decision_reviewer_actor"],
                "review_instance_id": "unratified-decision-instance",
            },
            "owner-ratified delegated actor",
        ),
        ("owner_ratification_sha256", _sha("forged-owner"), "owner-ratified authority"),
    ):
        tampered = deepcopy(record)
        tampered[field] = value
        body = {key: item for key, item in tampered.items() if key != "decision_sha256"}
        tampered["decision_sha256"] = krea_provenance.canonical_sha256(body)
        with pytest.raises(ValueError, match=message):
            krea_decision._validate_discovery_record(tampered)


def test_discovery_agreement_freezes_shared_winner_minimax_and_k0(harness: Harness):
    record, _, _, _ = harness.discovery_case(losses_by_fixture=_agreement_losses())

    assert record["outcome"] == "finalists_frozen"
    assert record["D1_winner_family_id"] == record["D2_winner_family_id"] == "K1"
    assert record["finalist_family_ids"] == ["K1", "K2", "K0"]
    assert set(record["all_family_checkpoint_rules"]) == set(ARMS)
    assert set(record["checkpoint_rules"]) == {"K0", "K1", "K2"}
    assert record["production_mutation_authorized"] is False
    assert record["release_review_required"] is True
    assert len(record["curve_results"]) == 2
    assert all(set(curves) == set(ARMS) for curves in record["curve_results"].values())
    assert all(
        len(curve) == 3
        for curves in record["curve_results"].values()
        for curve in curves.values()
    )


def test_discovery_disagreement_freezes_two_winners_minimax_and_k0(harness: Harness):
    losses = _agreement_losses()
    # Keep the fixture reversal outside the 0.01 uncertainty band; inside it,
    # the predeclared family tie-break intentionally resolves the apparent
    # disagreement before finalist selection.
    losses["D1"].update({"K1": 0.0680, "K2": 0.0710, "K3": 0.082})
    losses["D2"].update({"K1": 0.0710, "K2": 0.0680, "K3": 0.082})
    record, _, _, _ = harness.discovery_case(
        losses_by_fixture=losses,
        include_seed_b_policy=True,
        include_seed_b_aggregates=True,
    )

    assert record["outcome"] == "finalists_frozen"
    assert record["D1_winner_family_id"] == "K1"
    assert record["D2_winner_family_id"] == "K2"
    assert record["finalist_family_ids"] == ["K1", "K2", "K3", "K0"]
    assert len(record["finalist_family_ids"]) == 4


def test_three_way_tie_requires_seed_b_and_never_freezes_from_seed_a(harness: Harness):
    losses = _agreement_losses()
    for fixture in losses.values():
        fixture.update({"K1": 0.070, "K2": 0.0705, "K3": 0.0708})
    record, _, _, _ = harness.discovery_case(losses_by_fixture=losses)

    assert record["outcome"] == "seed_b_required"
    assert record["finalist_family_ids"] == []
    assert record["checkpoint_rules"] == {}
    assert record["seed_b_trigger"]["triggered"] is True
    assert (
        "three_or_more_noncontrols_inside_0.01_band"
        in record["seed_b_trigger"]["reasons"]
    )


def test_material_reversal_requires_seed_b_and_seed_b_allows_freeze(harness: Harness):
    losses = _agreement_losses()
    losses["D1"].update({"K1": 0.060, "K2": 0.085})
    losses["D2"].update({"K1": 0.085, "K2": 0.060})
    required, _, _, _ = harness.discovery_case(losses_by_fixture=losses)
    assert required["outcome"] == "seed_b_required"
    assert "material_D1_D2_rank_reversal" in required["seed_b_trigger"]["reasons"]

    second = Harness(harness.root / "with-b")
    frozen, _, _, _ = second.discovery_case(
        losses_by_fixture=losses,
        include_seed_b_policy=True,
        include_seed_b_aggregates=True,
    )
    assert frozen["outcome"] == "finalists_frozen"
    assert frozen["seeds_used"] == ["A", "B"]


def test_schema2_campaign_ledger_prevents_candidate_cherry_pick(harness: Harness):
    _, paths, policy_path, approval_path = harness.discovery_case(
        losses_by_fixture=_agreement_losses()
    )
    target = paths[0]
    aggregate = json.loads(target.read_text())
    aggregate["candidates"] = [
        row for row in aggregate["candidates"] if row["candidate_id"] != "D1-A-K2-500"
    ]
    aggregate["coverage"] = {
        "planned": len(aggregate["candidates"]),
        "completed": len(aggregate["candidates"]),
        "complete": True,
    }
    body = {key: value for key, value in aggregate.items() if key != "aggregate_sha256"}
    aggregate["aggregate_sha256"] = krea_provenance.canonical_sha256(body)
    _write(target, aggregate)

    with pytest.raises(ValueError, match="cherry-picks or invents"):
        krea_decision.decide_discovery(
            policy_path=policy_path,
            approval_path=approval_path,
            aggregate_paths=paths,
            output=harness.root / "krea-discovery-decision-retry.json",
            decided_at_utc="2026-07-28T00:03:00Z",
        )


def test_schema1_and_self_declared_campaign_contracts_fail_closed(harness: Harness):
    _, paths, policy_path, approval_path = harness.discovery_case(
        losses_by_fixture=_agreement_losses()
    )
    target = paths[0]
    aggregate = json.loads(target.read_text())
    aggregate["schema"] = 1
    body = {key: value for key, value in aggregate.items() if key != "aggregate_sha256"}
    aggregate["aggregate_sha256"] = krea_provenance.canonical_sha256(body)
    _write(target, aggregate)
    with pytest.raises(ValueError, match="identity is invalid"):
        krea_decision.decide_discovery(
            policy_path=policy_path,
            approval_path=approval_path,
            aggregate_paths=paths,
            output=harness.root / "krea-discovery-decision-schema1.json",
            decided_at_utc="2026-07-28T00:03:00Z",
        )


def test_frozen_constants_bootstrap_and_plan_shape_are_not_tunable(harness: Harness):
    bad = deepcopy(harness.plan)
    bad["decision_contract"]["bootstrap_resamples"] = 9999
    with pytest.raises(ValueError, match="differs from frozen protocol"):
        krea_decision._validate_discovery_plan(bad)

    bad = deepcopy(harness.plan)
    bad["confirmation_contract"]["fixture_shape_contract"]["C4"]["evaluation_rows"] = 24
    with pytest.raises(ValueError, match="differs from frozen protocol"):
        krea_decision._validate_discovery_plan(bad)

    bad = deepcopy(harness.plan)
    bad["confirmation_contract"]["fixture_shape_contract"]["C2"][
        "concept_class"
    ] = "generic small fixture"
    with pytest.raises(ValueError, match="differs from frozen protocol"):
        krea_decision._validate_discovery_plan(bad)

    bad = deepcopy(harness.plan)
    bad["confirmation_fixture_commitment"]["commitment_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from publication"):
        krea_decision._validate_discovery_plan(bad)

    bad_bootstrap = {**BOOTSTRAP, "seed": 7}
    with pytest.raises(ValueError, match="frozen policy"):
        krea_decision._validate_bootstrap(bad_bootstrap)

    clusters = {"C1": Decimal("0.01"), "C2": Decimal("0.02")}
    assert krea_decision._bootstrap_ci(clusters, label="fixed") == (
        krea_decision._bootstrap_ci(clusters, label="fixed")
    )


def test_confirmation_counts_are_per_fixture_not_small_large_aliases(
    harness: Harness,
):
    plan_state = krea_decision._validate_discovery_plan(harness.plan)
    assert {
        fixture: krea_decision._expected_fixture_counts(
            plan_state=plan_state,
            fixture_id=fixture,
            boundary=None,
        )
        for fixture in ("C1", "C2", "C3", "C4")
    } == {
        "C1": (20, 20, 6),
        "C2": (45, 45, 6),
        "C3": (30, 30, 8),
        "C4": (12, 12, 5),
    }
    assert krea_decision._expected_fixture_counts(
        plan_state=plan_state,
        fixture_id="B-0p5-small",
        boundary="small",
    ) == (18, 24, 24)
    with pytest.raises(ValueError, match="cannot declare a boundary alias"):
        krea_decision._expected_fixture_counts(
            plan_state=plan_state,
            fixture_id="C1",
            boundary="small",
        )


def test_checkpoint_tie_breaker_chooses_earliest_actual_step_inside_one_percent():
    def row(step: int, score: str) -> dict:
        return {
            "candidate_id": f"K1-{step}",
            "candidate_sha256": _sha(f"K1-{step}"),
            "step": step,
            "fraction_numerator": step,
            "fraction_denominator": 1000,
            "image_exposures": step * 8,
            "weighted_loss": Decimal(score),
        }

    zero = {"weighted_loss": Decimal("0.1")}
    analyses = {}
    for fixture in ("D1", "D2"):
        rows = [row(100, "0.081"), row(500, "0.0805"), row(1000, "0.080")]
        analyses[(fixture, "A")] = {
            "batch_id": f"{fixture}-A",
            "curves": {},
            "aggregate": {
                "candidates": [{**item, "family_id": "K1"} for item in rows],
                "zero": zero,
            },
        }
    rule = krea_decision._checkpoint_rule(
        "K1",
        analyses=analyses,
        fixtures=("D1", "D2"),
        seed_roles=("A",),
        targets=(Decimal("0.1"), Decimal("0.5"), Decimal("1")),
    )
    assert rule["target_fraction"] == 0.1
    assert {row["step"] for row in rule["actual_mappings"]} == {100}


def test_checkpoint_cross_run_tie_break_is_explicit_for_different_grids():
    zero = {"weighted_loss": Decimal("0.1")}
    grids = {
        "D1": ((100, 1000), (500, 1000), (1000, 1000)),
        "D2": ((80, 1200), (640, 1200), (1200, 1200)),
    }
    analyses = {}
    for fixture, grid in grids.items():
        candidates = []
        for step, denominator in grid:
            candidates.append(
                {
                    "candidate_id": f"{fixture}-K1-{step}",
                    "candidate_sha256": _sha(f"{fixture}-K1-{step}"),
                    "family_id": "K1",
                    "step": step,
                    "fraction_numerator": step,
                    "fraction_denominator": denominator,
                    "image_exposures": step * 8,
                    "weighted_loss": Decimal("0.0805"),
                }
            )
        analyses[(fixture, "A")] = {
            "batch_id": f"{fixture}-A",
            "aggregate": {"candidates": candidates, "zero": zero},
        }
    rule = krea_decision._checkpoint_rule(
        "K1",
        analyses=analyses,
        fixtures=("D1", "D2"),
        seed_roles=("A",),
        targets=(Decimal("0.1"), Decimal("0.5"), Decimal("1")),
    )
    assert rule["target_fraction"] == 0.1
    assert [row["step"] for row in rule["actual_mappings"]] == [100, 80]
    assert rule["cross_run_tie_breaker"].startswith("minimum maximum mapped step")


def test_strict_freeze_chronology_rejects_same_second_unblinding(harness: Harness):
    with pytest.raises(ValueError, match="predates its policy approval"):
        harness.discovery_case(
            losses_by_fixture=_agreement_losses(),
            decision_time="2026-07-28T00:01:00Z",
        )

    frozen = Harness(harness.root / "chronology")
    discovery, _, _, _ = frozen.discovery_case(losses_by_fixture=_agreement_losses())
    tampered = deepcopy(discovery)
    tampered["decided_at_utc"] = frozen.seal["sealed_at_utc"]
    decision_body = {
        key: value for key, value in tampered.items() if key != "decision_sha256"
    }
    tampered["decision_sha256"] = krea_provenance.canonical_sha256(decision_body)
    discovery_path, discovery_sha = _write(frozen.root / "same-time.json", tampered)
    with pytest.raises(ValueError, match="not sealed before discovery unblinding"):
        krea_decision.seal_confirmation_policy(
            {
                "schema": 2,
                "kind": "forge-krea-confirmation-decision-policy",
                "phase": "confirmation",
                "prepared_by": "Priya Engineer",
                "discovery_plan": _binding(frozen.plan_path, frozen.plan_sha),
                "discovery_execution_authorization": _binding(
                    frozen.authorization_path, frozen.authorization_sha
                ),
                "discovery_profile_index": dict(frozen.profile_index_binding),
                "confirmation_fixture_seal": _binding(
                    frozen.seal_path, frozen.seal_sha
                ),
                "discovery_decision": _binding(discovery_path, discovery_sha),
                "candidate_family_id": "K1",
                "score_batches": [{}],
                "public_reference_family_ids": list(PUBLIC),
                "deployed_control_family_id": "K0",
                "bootstrap": BOOTSTRAP,
            }
        )


def test_confirmation_passes_all_scientific_gates_but_never_claims_win_or_deploy(
    harness: Harness,
):
    discovery, _, _, _ = harness.discovery_case(losses_by_fixture=_agreement_losses())
    quality = {"K0": 0.100, "K1": 0.080, "K2": 0.082, "K3": 0.084, "K4": 0.086}
    record, _, _ = harness.confirmation_case(discovery, quality_losses=quality)

    assert record["outcome"] == "FAIL"
    assert record["field_parity_ready"] is False
    assert record["round1_ready"] is False
    assert record["win_ready"] is False
    assert record["production_mutation_authorized"] is False
    assert record["gates"]["stage2_production_surface_ratified"] is False
    assert all(
        passed
        for name, passed in record["gates"].items()
        if name != "stage2_production_surface_ratified"
    )
    assert record["metrics"]["point_wins_or_ties"] == 4
    assert len(record["metrics"]["concept_results"]) == 4
    assert len(record["boundary_results"]) == 6
    assert {
        episode["strongest_public_reference_family_id"]
        for concept in record["metrics"]["concept_results"]
        for episode in concept["episodes"]
    } == {"K2"}


def test_confirmation_fails_public_parity_and_concept_regression(harness: Harness):
    discovery, _, _, _ = harness.discovery_case(losses_by_fixture=_agreement_losses())
    quality = {"K0": 0.100, "K1": 0.090, "K2": 0.080, "K3": 0.084, "K4": 0.086}
    record, _, _ = harness.confirmation_case(discovery, quality_losses=quality)

    assert record["outcome"] == "FAIL"
    assert record["field_parity_ready"] is False
    assert record["gates"]["control_superiority_95pct"] is True
    assert record["gates"]["public_reference_noninferiority_95pct"] is False
    assert record["gates"]["no_concept_regression_over_0.03"] is False
    assert record["blockers"]


def test_self_rehashed_confirmation_gate_tampering_is_recomputed(harness: Harness):
    discovery, _, _, _ = harness.discovery_case(losses_by_fixture=_agreement_losses())
    quality = {
        "K0": 0.100,
        "K1": 0.080,
        "K2": 0.082,
        "K3": 0.084,
        "K4": 0.086,
    }
    record, _, _ = harness.confirmation_case(discovery, quality_losses=quality)
    tampered = deepcopy(record)
    tampered["outcome"] = "FAIL"
    tampered["field_parity_ready"] = False
    tampered["round1_ready"] = False
    tampered["gates"]["public_reference_noninferiority_95pct"] = False
    tampered["blockers"] = [
        "failed confirmation gate: public_reference_noninferiority_95pct"
    ]
    body = {key: value for key, value in tampered.items() if key != "decision_sha256"}
    tampered["decision_sha256"] = krea_provenance.canonical_sha256(body)

    with pytest.raises(ValueError, match="gates do not recompute"):
        krea_decision._validate_confirmation_record(tampered)


def test_confirmation_missing_any_c1_c4_or_boundary_batch_is_no_go(harness: Harness):
    discovery, _, _, _ = harness.discovery_case(losses_by_fixture=_agreement_losses())
    quality = {"K0": 0.100, "K1": 0.080, "K2": 0.082, "K3": 0.084, "K4": 0.086}
    record, _, _ = harness.confirmation_case(
        discovery,
        quality_losses=quality,
        omit_batch="C4-B",
    )
    assert record["outcome"] == "no-go"
    assert record["metrics"] == {}
    assert record["gates"] == {}
    assert any("C4-B" in blocker for blocker in record["blockers"])


def test_confirmation_fixture_shape_and_row_counts_are_enforced(harness: Harness):
    discovery, _, _, _ = harness.discovery_case(losses_by_fixture=_agreement_losses())
    quality = {
        "K0": 0.100,
        "K1": 0.080,
        "K2": 0.082,
        "K3": 0.084,
        "K4": 0.086,
    }
    _, paths, _ = harness.confirmation_case(discovery, quality_losses=quality)
    target = next(path for path in paths if path.name == "aggregate-C1-A.json")
    aggregate = json.loads(target.read_text())
    aggregate["fixture_contract"]["evaluation_row_count"] = 23
    body = {key: value for key, value in aggregate.items() if key != "aggregate_sha256"}
    aggregate["aggregate_sha256"] = krea_provenance.canonical_sha256(body)
    _write(target, aggregate)

    with pytest.raises(ValueError, match="violates frozen fixture counts"):
        krea_decision.decide_confirmation(
            policy_path=harness.root / "confirmation-policy.json",
            approval_path=harness.root / "confirmation-approval.json",
            aggregate_paths=paths,
            output=harness.root / "krea-confirmation-decision-retry.json",
            decided_at_utc="2026-07-28T00:05:00Z",
        )


def test_confirmation_policy_requires_all_eight_quality_and_six_boundary_cells(
    harness: Harness,
):
    discovery, _, _, _ = harness.discovery_case(losses_by_fixture=_agreement_losses())
    discovery_path, discovery_sha = _write(
        harness.root / "manual-discovery.json", discovery
    )
    payload = {
        "schema": 2,
        "kind": "forge-krea-confirmation-decision-policy",
        "phase": "confirmation",
        "prepared_by": "Priya Engineer",
        "discovery_plan": _binding(harness.plan_path, harness.plan_sha),
        "discovery_execution_authorization": _binding(
            harness.authorization_path, harness.authorization_sha
        ),
        "discovery_profile_index": dict(harness.profile_index_binding),
        "confirmation_fixture_seal": _binding(harness.seal_path, harness.seal_sha),
        "discovery_decision": _binding(discovery_path, discovery_sha),
        "candidate_family_id": "K1",
        "score_batches": [],
        "public_reference_family_ids": list(PUBLIC),
        "deployed_control_family_id": "K0",
        "bootstrap": BOOTSTRAP,
    }
    with pytest.raises(ValueError, match="requires score batches"):
        krea_decision.seal_confirmation_policy(payload)

    # Start from an otherwise valid sealed policy and independently remove a
    # quality episode and a boundary cell.  Neither omission may be papered
    # over by the decision code.
    valid_root = Harness(harness.root / "valid-policy")
    valid_discovery, _, _, _ = valid_root.discovery_case(
        losses_by_fixture=_agreement_losses()
    )
    quality = {
        "K0": 0.100,
        "K1": 0.080,
        "K2": 0.082,
        "K3": 0.084,
        "K4": 0.086,
    }
    _, _, valid_policy = valid_root.confirmation_case(
        valid_discovery, quality_losses=quality
    )
    policy_body = {
        key: value for key, value in valid_policy.items() if key != "policy_sha256"
    }
    missing_quality = deepcopy(policy_body)
    missing_quality["score_batches"] = [
        row for row in missing_quality["score_batches"] if row["batch_id"] != "C4-B"
    ]
    with pytest.raises(ValueError, match="C1-C4 at both predeclared seeds"):
        krea_decision.seal_confirmation_policy(missing_quality)
    missing_boundary = deepcopy(policy_body)
    missing_boundary["score_batches"] = [
        row
        for row in missing_boundary["score_batches"]
        if row["batch_id"] != "B-1p0-large"
    ]
    with pytest.raises(ValueError, match="complete 3x2 boundary matrix"):
        krea_decision.seal_confirmation_policy(missing_boundary)


def test_boundary_fallback_or_late_decision_is_rejected_before_statistics(
    harness: Harness,
):
    discovery, _, _, _ = harness.discovery_case(losses_by_fixture=_agreement_losses())
    quality = {"K0": 0.100, "K1": 0.080, "K2": 0.082, "K3": 0.084, "K4": 0.086}
    # Build a valid case in another directory so we can tamper before decision.
    other = Harness(harness.root / "boundary-tamper")
    other_discovery, _, _, _ = other.discovery_case(
        losses_by_fixture=_agreement_losses()
    )
    discovery_path, discovery_sha = _write(
        other.root / "frozen-discovery.json", other_discovery
    )
    candidate = "K1"
    paths, batches = [], []
    for fixture in ("C1", "C2", "C3", "C4"):
        for seed in ("A", "B"):
            path, batch = other.aggregate(
                batch_id=f"{fixture}-{seed}",
                phase="confirmation",
                fixture_id=fixture,
                seed_role=seed,
                arms=("K0", "K1", "K2", "K3", "K4"),
                losses=quality,
                hours=0.75,
                steps=(900, 1000),
            )
            paths.append(path)
            batches.append(batch)
    for hours in (0.5, 0.75, 1.0):
        for boundary in ("small", "large"):
            batch_id = f"B-{str(hours).replace('.', 'p')}-{boundary}"
            path, batch = other.aggregate(
                batch_id=batch_id,
                phase="boundary",
                fixture_id=batch_id,
                seed_role="A",
                arms=(candidate,),
                losses={candidate: 0.08},
                hours=hours,
                boundary=boundary,
                steps=(900,),
            )
            paths.append(path)
            batches.append(batch)
    tampered = json.loads(paths[-1].read_text())
    tampered["training_run_envelopes"][0]["candidate_decision"]["fallback_used"] = True
    body = {key: value for key, value in tampered.items() if key != "aggregate_sha256"}
    tampered["aggregate_sha256"] = krea_provenance.canonical_sha256(body)
    _write(paths[-1], tampered)
    payload = {
        "schema": 2,
        "kind": "forge-krea-confirmation-decision-policy",
        "phase": "confirmation",
        "prepared_by": "Priya Engineer",
        "discovery_plan": _binding(other.plan_path, other.plan_sha),
        "discovery_execution_authorization": _binding(
            other.authorization_path, other.authorization_sha
        ),
        "discovery_profile_index": dict(other.profile_index_binding),
        "confirmation_fixture_seal": _binding(other.seal_path, other.seal_sha),
        "discovery_decision": _binding(discovery_path, discovery_sha),
        "candidate_family_id": candidate,
        "score_batches": sorted(batches, key=lambda row: row["batch_id"]),
        "public_reference_family_ids": list(PUBLIC),
        "deployed_control_family_id": "K0",
        "bootstrap": BOOTSTRAP,
    }
    policy = krea_decision.seal_confirmation_policy(payload)
    policy_path, _ = _write(other.root / "confirmation-policy.json", policy)
    approval = krea_decision.build_approval(
        policy,
        reviewer_identity="Morgan Auditor",
        approved_at_utc="2026-07-28T00:03:00Z",
    )
    approval_path, _ = _write(other.root / "confirmation-approval.json", approval)
    with pytest.raises(ValueError, match="late, fallback-dependent, or unbound"):
        krea_decision.decide_confirmation(
            policy_path=policy_path,
            approval_path=approval_path,
            aggregate_paths=paths,
            output=other.root / "krea-confirmation-decision.json",
            decided_at_utc="2026-07-28T00:04:00Z",
        )


def test_confirmation_candidate_must_be_predeclared_noncontrol_finalist(
    harness: Harness,
):
    discovery, _, _, _ = harness.discovery_case(losses_by_fixture=_agreement_losses())
    path, digest = _write(harness.root / "disc.json", discovery)
    payload = {
        "schema": 2,
        "kind": "forge-krea-confirmation-decision-policy",
        "phase": "confirmation",
        "prepared_by": "Priya Engineer",
        "discovery_plan": _binding(harness.plan_path, harness.plan_sha),
        "discovery_execution_authorization": _binding(
            harness.authorization_path, harness.authorization_sha
        ),
        "discovery_profile_index": dict(harness.profile_index_binding),
        "confirmation_fixture_seal": _binding(harness.seal_path, harness.seal_sha),
        "discovery_decision": _binding(path, digest),
        "candidate_family_id": "K5",
        "score_batches": [{}],
        "public_reference_family_ids": list(PUBLIC),
        "deployed_control_family_id": "K0",
        "bootstrap": BOOTSTRAP,
    }
    with pytest.raises(ValueError, match="non-control frozen discovery finalist"):
        krea_decision.seal_confirmation_policy(payload)


def test_self_rehashed_discovery_winner_tampering_cannot_seed_confirmation(
    harness: Harness,
):
    discovery, _, _, _ = harness.discovery_case(losses_by_fixture=_agreement_losses())
    tampered = deepcopy(discovery)
    tampered["D1_winner_family_id"] = "K2"
    body = {key: value for key, value in tampered.items() if key != "decision_sha256"}
    tampered["decision_sha256"] = krea_provenance.canonical_sha256(body)

    with pytest.raises(ValueError, match="winners do not recompute"):
        krea_decision._validate_discovery_record(tampered)


def test_output_cannot_target_production_or_selector_names(
    harness: Harness, tmp_path: Path
):
    record, paths, policy, approval = harness.discovery_case(
        losses_by_fixture=_agreement_losses()
    )
    del record
    production = Path("forge") / "krea-discovery-decision.json"
    with pytest.raises(ValueError, match="production package"):
        krea_decision.decide_discovery(
            policy_path=policy,
            approval_path=approval,
            aggregate_paths=paths,
            output=production,
            decided_at_utc="2026-07-28T00:03:00Z",
        )
    with pytest.raises(ValueError, match="non-production JSON"):
        krea_decision.decide_discovery(
            policy_path=policy,
            approval_path=approval,
            aggregate_paths=paths,
            output=tmp_path / "forge_holdout_scores.json",
            decided_at_utc="2026-07-28T00:03:00Z",
        )


def _tie_analyses(exposures: dict[str, int]) -> dict:
    return {
        (fixture, "A"): {
            "curves": {
                family: {"selected": {"image_exposures": depth}}
                for family, depth in exposures.items()
            }
        }
        for fixture in ("D1", "D2")
    }


def test_family_tie_break_axes_are_depth_consistency_then_fixed_family() -> None:
    families = ("K1", "K2", "K3")

    depth_scores = {
        "K1": {"D1": Decimal("0.205"), "D2": Decimal("0.205")},
        "K2": {"D1": Decimal("0.200"), "D2": Decimal("0.200")},
        "K3": {"D1": Decimal("0.199"), "D2": Decimal("0.199")},
    }
    assert (
        krea_decision._pick_family_within_uncertainty(
            families,
            primary_values={family: depth_scores[family]["D1"] for family in families},
            analyses=_tie_analyses({"K1": 800, "K2": 1000, "K3": 900}),
            concept_scores=depth_scores,
            fixtures=("D1", "D2"),
            seed_roles=("A",),
        )
        == "K2"
    )

    consistency_scores = {
        "K1": {"D1": Decimal("0.205"), "D2": Decimal("0.190")},
        "K2": {"D1": Decimal("0.202"), "D2": Decimal("0.188")},
        "K3": {"D1": Decimal("0.200"), "D2": Decimal("0.199")},
    }
    assert (
        krea_decision._pick_family_within_uncertainty(
            families,
            primary_values={
                family: consistency_scores[family]["D1"] for family in families
            },
            analyses=_tie_analyses({family: 1000 for family in families}),
            concept_scores=consistency_scores,
            fixtures=("D1", "D2"),
            seed_roles=("A",),
        )
        == "K3"
    )

    family_scores = {
        family: {"D1": Decimal("0.2"), "D2": Decimal("0.2")} for family in families
    }
    assert (
        krea_decision._pick_family_within_uncertainty(
            families,
            primary_values={family: family_scores[family]["D1"] for family in families},
            analyses=_tie_analyses({family: 1000 for family in families}),
            concept_scores=family_scores,
            fixtures=("D1", "D2"),
            seed_roles=("A",),
        )
        == "K2"
    )
