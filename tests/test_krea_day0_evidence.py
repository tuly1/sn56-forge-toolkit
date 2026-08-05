"""ML-free tests for Day-0 Krea provenance and exact-score contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

_CALIBRATION = Path(__file__).parents[1] / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _CALIBRATION / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


provenance = _load("krea_provenance")
sys.modules["krea_provenance"] = provenance
batch = _load("batch_evaluate_krea")
sys.modules["batch_evaluate_krea"] = batch
ladder = _load("run_krea_ladder")
sys.modules["run_krea_ladder"] = ladder
producer = _load("krea_training_evidence")

GOD_SHA = "a" * 40
COMFY_SHA = "b" * 40
TOOLING_SHA = "c" * 40
EVALUATION_DATASET_SHA = "d" * 64
TRAINING_DATASET_SHA = "e" * 64
COMFY_RUNTIME = {"distributions_sha256": "f" * 64}
DRIVER_RUNTIME = {"distributions_sha256": "0" * 64}
TRAINING_RUNTIME_SHA = "a" * 64


def test_host_checks_use_sealed_checkpoint_root_before_and_after_run(
    tmp_path, monkeypatch
):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    save_root = checkpoint_root / "task-id" / "repo"
    assert not save_root.parent.exists()
    monkeypatch.setattr(ladder, "_CHECKPOINT_ROOT", checkpoint_root)

    bound_root = ladder._checkpoint_root_for_save(save_root)
    calls = []

    class HostIdentity:
        @staticmethod
        def verify_live(manifest, *, checkpoint_path):
            calls.append(("preflight", manifest, checkpoint_path))
            return {"phase": "preflight"}

        @staticmethod
        def verify_static(manifest, *, checkpoint_path):
            calls.append(("post_run", manifest, checkpoint_path))
            return {"phase": "post_run"}

    manifest = {"sealed": True}
    assert ladder._verify_host_preflight(HostIdentity, manifest, bound_root) == {
        "phase": "preflight"
    }
    assert ladder._verify_host_post_run(HostIdentity, manifest, bound_root) == {
        "phase": "post_run"
    }
    assert calls == [
        ("preflight", manifest, checkpoint_root),
        ("post_run", manifest, checkpoint_root),
    ]


def test_checkpoint_root_binding_rejects_root_and_escape(tmp_path, monkeypatch):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    monkeypatch.setattr(ladder, "_CHECKPOINT_ROOT", checkpoint_root)

    with pytest.raises(ValueError, match="escaped /app/checkpoints"):
        ladder._checkpoint_root_for_save(checkpoint_root)
    with pytest.raises(ValueError, match="escaped /app/checkpoints"):
        ladder._checkpoint_root_for_save(tmp_path / "other" / "repo")


def _recipe():
    def row(
        classification,
        source_value,
        effective_value,
        *,
        pointer=None,
        evidence="immutable source/config evidence",
    ):
        return {
            "classification": classification,
            "source_pointers": [] if pointer is None else [pointer],
            "source_value": source_value,
            "effective_value": effective_value,
            "evidence": evidence,
        }

    return {
        "schema": 1,
        "kind": "forge-krea-normalized-recipe",
        "fields": {
            "planned_steps": row("known", 1500, 1500, pointer="/train/steps"),
            # Source submission facts are retained, but the local training
            # stage may not preselect the checkpoint it will later score.
            "submitted_step": row(
                "unsupported", 1500, None, pointer="/artifact/submitted_step"
            ),
            "learning_rate": row("known", 0.0002, 0.0002, pointer="/train/lr"),
            "rank": row("adapted", 144, 128, pointer="/network/rank"),
            "alpha": row("unknown_source_fixed", None, 128),
            "optimizer": row("unknown_source_fixed", None, "adamw8bit"),
            "optimizer_parameters": row(
                "unknown_source_fixed", None, {"weight_decay": 0.0001}
            ),
            "loss": row("unknown_source_fixed", None, "mse"),
            "guidance": row(
                "unknown_source_fixed", None, {"enabled": True, "scale": 2}
            ),
            "scheduler": row("unknown_source_fixed", None, "flowmatch"),
            "dropout": row("unknown_source_fixed", None, 0.05),
            "gradient_accumulation": row("unknown_source_fixed", None, 1),
            "effective_batch": row("unknown_source_fixed", None, 1),
            "ema": row(
                "adapted",
                {"enabled": True, "decay": 0.99},
                {"enabled": False, "decay": 0.99},
                pointer="/train/ema",
            ),
            "save_cadence": row("unknown_source_fixed", None, 200),
            "selector": row("unknown", None, None),
        },
    }


def _source_recipe():
    execution = _recipe()
    known = {"planned_steps", "submitted_step", "learning_rate", "rank", "ema"}
    fields = {}
    for name, row in execution["fields"].items():
        if name in known:
            fields[name] = {
                "classification": "known",
                "source_pointers": row["source_pointers"],
                "source_value": row["source_value"],
                "effective_value": None,
                "evidence": row["evidence"],
            }
        else:
            fields[name] = {
                "classification": "unknown",
                "source_pointers": [],
                "source_value": None,
                "effective_value": None,
                "evidence": "public source did not disclose this field",
            }
    return {
        "schema": 1,
        "kind": "forge-krea-normalized-recipe",
        "fields": fields,
    }


def _canonical_file(path: Path, value) -> str:
    path.write_bytes(provenance.canonical_bytes(value) + b"\n")
    return provenance.file_sha256(path)


def _fixture_row(row_id, digit, perceptual_hash):
    def digest(label):
        return hashlib.sha256(f"{row_id}:{digit}:{label}".encode()).hexdigest()

    content = {
        "image_sha256": digest("image"),
        "decoded_pixels_sha256": digest("pixels"),
        "caption_sha256": digest("caption"),
        "normalized_caption_sha256": digest("normalized-caption"),
        "width": 64,
        "height": 64,
        "mode": "RGB",
    }
    return {
        "row_id": row_id,
        "content_sha256": provenance.canonical_sha256(content),
        **content,
        "perceptual_hash64": perceptual_hash,
    }


def _metadata(
    source_arm_id: str,
    *,
    approved: bool = True,
    mode: str = "local_reproduction",
    matched_dataset_sha: str | None = None,
    reviewer: str = "Atulya Shetty",
):
    direct = mode == "direct_public_artifact"
    local_disclosure = None
    if not direct:
        local_disclosure = {
            "schema": 1,
            "kind": "forge-krea-local-reproduction-disclosure",
            "execution_authorized": False,
            "adapted_fields": [
                {
                    "name": "depth policy",
                    "source_recipe_fields": ["planned_steps", "submitted_step"],
                    "local_policy": "measured-budget-fill",
                    "evidence": (
                        "Source depth remains immutable; local depth is resolved by "
                        "the predeclared timing policy."
                    ),
                }
            ],
            "source_unknown_fields": [],
            "predeclared_local_values": [],
            "claim_limit": (
                "Machine disclosure only; not human review or execution approval."
            ),
        }
    return {
        "source_arm_id": source_arm_id,
        "source": {
            "url": f"https://huggingface.co/gradients/{source_arm_id}",
            "revision": "1" * 40,
        },
        "official_context": {
            "tournament_id": "tourn_test",
            "task_id": "task-1",
            "hotkey": f"hotkey-{source_arm_id}",
            "submission_id": f"submission-{source_arm_id}",
            "official_rank": 1,
            "official_loss": 0.04,
            "repository": f"gradients/{source_arm_id}",
            "repo_revision": "1" * 40,
            "artifact_repo_path": "checkpoints/last.safetensors",
            "config_repo_path": "checkpoints/config.yaml",
        },
        "fields": {
            "observed": {
                "/artifact/submitted_step": 1500,
                "/network/rank": 144,
                "/train/ema": {"enabled": True, "decay": 0.99},
                "/train/lr": 0.0002,
                "/train/steps": 1500,
            },
            "unsupported": [],
            "adapted": [],
        },
        "evaluator_sha": GOD_SHA,
        "matched_concept": {
            "available": matched_dataset_sha is not None,
            "dataset_sha256": matched_dataset_sha,
            "basis": (
                "harvested task dataset is byte-bound"
                if matched_dataset_sha
                else "hidden public-task dataset was unavailable"
            ),
            "evidence": {
                "public_task_id": "task-1",
                "matched_dataset_recovered": matched_dataset_sha is not None,
            },
        },
        "adaptation_target": {
            "mode": mode,
            "model_type": "krea2",
            "source_artifact_role": "score_candidate" if direct else "reference_only",
            "candidate_role": "source_artifact" if direct else "local_training_output",
            "description": (
                "score only against its matched concept"
                if direct
                else "retrain recipe from the immutable base"
            ),
        },
        "local_reproduction_disclosure": local_disclosure,
        "normalized_recipe": _source_recipe(),
        # This is deliberately an assertion, not a signature/authentication.
        "review_assertion": {
            "status": "approved" if approved else "unreviewed",
            "reviewer_identity": reviewer,
            "notes": "source fields checked",
        },
    }


def _source_provenance(
    root: Path,
    source_arm_id: str,
    *,
    approved=True,
    mode="local_reproduction",
    matched_dataset_sha=None,
    reviewer="Atulya Shetty",
):
    config = root / f"{source_arm_id}.source.yaml"
    artifact = root / f"{source_arm_id}.source.safetensors"
    config.write_bytes(b"train:\n  lr: 0.0002\n")
    artifact.write_bytes(f"public-source-{source_arm_id}".encode())
    task_raw = root / f"{source_arm_id}.task.raw.json"
    tournament_raw = root / f"{source_arm_id}.tournament.raw.json"
    revision_raw = root / f"{source_arm_id}.revision.raw.json"
    _canonical_file(
        task_raw,
        {
            "task_id": "task-1",
            "model_type": "krea2",
            "hotkey_details": [
                {
                    "hotkey": f"hotkey-{source_arm_id}",
                    "submission_id": f"submission-{source_arm_id}",
                    "rank": 1,
                    "test_loss": 0.04,
                    "repo": f"gradients/{source_arm_id}",
                }
            ],
        },
    )
    _canonical_file(
        tournament_raw,
        {
            "tournament_id": "tourn_test",
            "rounds": [
                {
                    "status": "completed",
                    "tasks": [{"task_id": "task-1"}],
                }
            ],
        },
    )
    _canonical_file(
        revision_raw,
        {
            "capture_complete": True,
            "captures": [
                {
                    "path": "checkpoints/config.yaml",
                    "kind": "small",
                    "captured": True,
                    "object_sha256": provenance.file_sha256(config),
                    "bytes": config.stat().st_size,
                },
                {
                    "path": "checkpoints/last.safetensors",
                    "kind": "weight",
                    "captured": True,
                    "object_sha256": provenance.file_sha256(artifact),
                    "bytes": artifact.stat().st_size,
                },
            ],
            "config_absent": False,
            "configs": ["checkpoints/config.yaml"],
            "eligible_weight_plan": [
                {
                    "path": "checkpoints/last.safetensors",
                    "lfs_oid": provenance.file_sha256(artifact),
                    "size": artifact.stat().st_size,
                }
            ],
            "failures": [],
            "processing_complete": True,
            "repo_id": f"gradients/{source_arm_id}",
            "revision": "1" * 40,
            "skipped": [],
            "tree_entry_count": 2,
            "tree_file_count": 2,
            "tree_truncated": False,
            "weights_enabled": True,
        },
    )
    ledger = root / f"{source_arm_id}.field-ledger.json"
    _canonical_file(
        ledger,
        {
            "schema": 1,
            "kind": "sn56-week5-krea-r1-public-field-ledger",
            "task": {
                "task_id": "task-1",
                "tournament_api": "https://api.gradients.io/tournament/tourn_test/details",
                "snapshot": {
                    "tournament_snapshot_sha256": provenance.file_sha256(
                        tournament_raw
                    ),
                    "task_snapshot_sha256": provenance.file_sha256(task_raw),
                },
            },
            "submissions": [
                {
                    "official_rank": 1,
                    "hotkey": f"hotkey-{source_arm_id}",
                    "score": 0.04,
                    "submission_id": f"submission-{source_arm_id}",
                    "repo": f"gradients/{source_arm_id}",
                    "repo_revision": "1" * 40,
                    "config_url": (
                        f"https://huggingface.co/gradients/{source_arm_id}/resolve/"
                        + "1" * 40
                        + "/checkpoints/config.yaml"
                    ),
                    "artifact": {
                        "path": "checkpoints/last.safetensors",
                        "lfs_sha256": provenance.file_sha256(artifact),
                    },
                }
            ],
        },
    )
    manifest = provenance.build_manifest(
        _metadata(
            source_arm_id,
            approved=approved,
            mode=mode,
            matched_dataset_sha=matched_dataset_sha,
            reviewer=reviewer,
        ),
        source_config_path=config,
        source_artifact_path=artifact,
        field_ledger_path=ledger,
        task_raw_path=task_raw,
        tournament_raw_path=tournament_raw,
        revision_manifest_path=revision_raw,
    )
    path = root / f"{source_arm_id}.provenance.json"
    provenance.publish_exclusive(path, manifest)
    return path, manifest, config, artifact


def _approval(root: Path, arm: str, manifest, *, reviewer="Atulya Shetty"):
    path = root / f"{arm}.approval.json"
    digest = _canonical_file(
        path,
        {
            "schema": 1,
            "kind": "forge-krea-source-normalization-approval",
            "decision": "approved",
            "reviewer_identity": reviewer,
            "source_arm_id": arm,
            "provenance_manifest_sha256": manifest["manifest_sha256"],
        },
    )
    return {"path": str(path), "sha256": digest}


def _local_binding(
    root: Path, arm: str, manifest, candidate: Path, *, reviewer="Atulya Shetty"
):
    training_rows = [
        _fixture_row("train-001", "1", "0000000000000000"),
        _fixture_row("train-002", "5", "0000000000000001"),
    ]
    evaluation_rows = [_fixture_row("eval-001", "9", "ffffffffffffffff")]
    near_report = {
        "comparisons": 2,
        "minimum_hamming_distance": 63,
        "matches": [],
    }
    fixture = root / f"{arm}.fixture-split.json"
    fixture_sha = _canonical_file(
        fixture,
        {
            "schema": 1,
            "kind": "forge-krea-fixture-split",
            "concept_id": "concept-week5",
            "concept_evidence_sha256": "5" * 64,
            "training_dataset_sha256": TRAINING_DATASET_SHA,
            "evaluation_dataset_sha256": EVALUATION_DATASET_SHA,
            "training_rows": training_rows,
            "evaluation_rows": evaluation_rows,
            "near_duplicate_policy": {
                "detector": "pillow-rgb-average-hash-8x8",
                "implementation_sha256": "5" * 64,
                "maximum_hamming_distance": 6,
                "report": near_report,
                "report_sha256": provenance.canonical_sha256(near_report),
                "passed": True,
            },
        },
    )
    condition = root / f"{arm}.condition.json"
    condition_sha = _canonical_file(
        condition,
        {
            "schema": 1,
            "kind": "forge-krea-training-condition",
            "source_arm_id": arm,
            "provenance_manifest_sha256": manifest["manifest_sha256"],
            "normalized_recipe": _recipe(),
            "base_model": {
                "model_id": "krea/Krea-2-Raw",
                "revision": "6" * 40,
            },
            "seed": 42565431,
            "training_dataset_sha256": TRAINING_DATASET_SHA,
            "fixture_split_manifest_sha256": fixture_sha,
            "train_rows_sha256": provenance.canonical_sha256(training_rows),
            "runtime_identity_sha256": TRAINING_RUNTIME_SHA,
            "training_geometry": {
                "resolution_policy_sha256": "7" * 64,
                "precision_policy_sha256": "8" * 64,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "data_parallel_replicas": 1,
                "effective_batch_size": 1,
            },
            "predeclared_recipe_axes": [
                "guidance",
                "learning_rate",
                "planned_steps",
            ],
        },
    )
    run_record = root / f"{arm}.run.json"
    training_log = root / f"{arm}.train.log"
    run_record.write_bytes(b'{"trainer":"emitted"}\n')
    training_log.write_bytes(b"natural completion at step 1500\n")
    run_sha = provenance.file_sha256(run_record)
    log_sha = provenance.file_sha256(training_log)
    completion = root / f"{arm}.completion.json"
    completion_sha = _canonical_file(
        completion,
        {
            "schema": 1,
            "kind": "forge-krea-training-completion",
            "source_arm_id": arm,
            "provenance_manifest_sha256": manifest["manifest_sha256"],
            "training_condition_sha256": condition_sha,
            "fixture_split_manifest_sha256": fixture_sha,
            "training_dataset_sha256": TRAINING_DATASET_SHA,
            "candidate_sha256": provenance.file_sha256(candidate),
            "run_record_sha256": run_sha,
            "training_log_sha256": log_sha,
            "natural_completion": True,
        },
    )
    return {
        "mode": "local_reproduction",
        "training_dataset_sha256": TRAINING_DATASET_SHA,
        "evaluation_dataset_sha256": EVALUATION_DATASET_SHA,
        "fixture_split_manifest": {"path": str(fixture), "sha256": fixture_sha},
        "training_condition": {"path": str(condition), "sha256": condition_sha},
        "completion_manifest": {
            "path": str(completion),
            "sha256": completion_sha,
        },
        "run_record": {"path": str(run_record), "sha256": run_sha},
        "training_log": {"path": str(training_log), "sha256": log_sha},
        "source_normalization_approval": _approval(
            root, arm, manifest, reviewer=reviewer
        ),
    }


def _plan(
    root: Path,
    arms=("arm-a", "arm-b"),
    *,
    approved=True,
    mode="local_reproduction",
    matched_dataset_sha=None,
    reviewer="Atulya Shetty",
):
    dataset = root / "evaluation-dataset"
    comfy = root / "comfy"
    god = root / "god"
    for directory in (dataset, comfy, god):
        directory.mkdir()
    (comfy / "models" / "loras").mkdir(parents=True)
    comfy_python = root / "comfy-python"
    comfy_python.write_text("#!/bin/sh\n", encoding="utf-8")
    comfy_python.chmod(0o755)
    rows = []
    for arm in arms:
        prov_path, manifest, _, source_artifact = _source_provenance(
            root,
            arm,
            approved=approved,
            mode=mode,
            matched_dataset_sha=matched_dataset_sha,
            reviewer=reviewer,
        )
        if mode == "direct_public_artifact":
            candidate = source_artifact
            candidate_id = f"{arm}-public"
            binding = {
                "mode": mode,
                "source_normalization_approval": _approval(
                    root, arm, manifest, reviewer=reviewer
                ),
            }
        else:
            candidate = root / f"{arm}.reproduced.safetensors"
            candidate.write_bytes(f"local-output-{arm}".encode())
            candidate_id = f"{arm}-repro"
            binding = _local_binding(root, arm, manifest, candidate, reviewer=reviewer)
        rows.append(
            {
                "id": candidate_id,
                "source_arm_id": arm,
                "path": str(candidate),
                "sha256": provenance.file_sha256(candidate),
                "provenance": str(prov_path),
                "candidate_binding": binding,
            }
        )
    containment_binary = Path(shutil.which("true") or "/usr/bin/true")
    containment_binary_sha = provenance.file_sha256(
        containment_binary.resolve(strict=True)
    )
    plan = {
        "schema": 1,
        "dataset": {"path": str(dataset), "sha256": EVALUATION_DATASET_SHA},
        "candidates": rows,
        "evaluator": {
            "comfy_root": str(comfy),
            "comfy_python": str(comfy_python),
            "god_root": str(god),
            "driver_python": str(Path(sys.executable)),
            "expected_god_commit": GOD_SHA,
            "expected_comfy_commit": COMFY_SHA,
            "expected_tooling_commit": TOOLING_SHA,
            "expected_evaluator_script_sha256": provenance.file_sha256(
                _CALIBRATION / "evaluate_krea_local.py"
            ),
            "expected_dataset_identity_module_sha256": provenance.file_sha256(
                _CALIBRATION / "krea_dataset_identity.py"
            ),
            "expected_eval_defaults": {
                "steps": 28,
                "cfg": 1.0,
                "denoise": 0.85,
                "generations": 1,
                "text_weight": 0.5,
                "master_seed": 42,
            },
            "expected_runtime_identity": {
                "comfy_python_identity_sha256": provenance.canonical_sha256(
                    COMFY_RUNTIME
                ),
                "driver_python_identity_sha256": provenance.canonical_sha256(
                    DRIVER_RUNTIME
                ),
            },
            "expected_assets": {
                "diffusion_model": {
                    "canonical_path": "/cache/models--Comfy-Org--Krea-2.safetensors",
                    "sha256": "c" * 64,
                    "bytes": 1,
                },
                "text_encoder": {
                    "canonical_path": "/cache/qwen_3_4b.safetensors",
                    "sha256": "d" * 64,
                    "bytes": 1,
                },
                "vae": {
                    "canonical_path": "/cache/ae.safetensors",
                    "sha256": "e" * 64,
                    "bytes": 1,
                },
            },
            "cache_provenance_sha256": "9" * 64,
            "containment": {
                "mode": "systemd_transient_service",
                "unit_type": "transient_service",
                "network_policy": {
                    "private_network": True,
                    "restrict_address_families": ["AF_UNIX", "AF_INET", "AF_INET6"],
                    "loopback_allowed": True,
                    "outbound_network_blocked": True,
                },
                "term_grace_s": 0.5,
                "systemd_run_path": str(containment_binary),
                "systemd_run_sha256": containment_binary_sha,
                "systemctl_path": str(containment_binary),
                "systemctl_sha256": containment_binary_sha,
            },
            "port": 8199,
        },
    }
    approval = batch.build_sealed_plan_approval(
        plan, reviewer_identity="Jordan Example"
    )
    approval_path = root / "sealed-plan-approval.json"
    approval_sha = _canonical_file(approval_path, approval)
    plan["sealed_plan_approval"] = {
        "path": str(approval_path),
        "sha256": approval_sha,
    }
    return plan


def _reseal_plan(plan):
    approval_path = Path(plan["sealed_plan_approval"]["path"])
    approval = batch.build_sealed_plan_approval(
        plan, reviewer_identity="Jordan Example"
    )
    plan["sealed_plan_approval"]["sha256"] = _canonical_file(approval_path, approval)


def _rewrite_condition_chain(
    plan, index, *, mutate_condition=None, mutate_fixture=None
):
    row = plan["candidates"][index]
    binding = row["candidate_binding"]
    fixture_path = Path(binding["fixture_split_manifest"]["path"])
    fixture = json.loads(fixture_path.read_text())
    if mutate_fixture is not None:
        mutate_fixture(fixture, binding)
    fixture_sha = _canonical_file(fixture_path, fixture)
    binding["fixture_split_manifest"]["sha256"] = fixture_sha

    condition_path = Path(binding["training_condition"]["path"])
    condition = json.loads(condition_path.read_text())
    condition["fixture_split_manifest_sha256"] = fixture_sha
    if mutate_condition is not None:
        mutate_condition(condition, binding)
    condition_sha = _canonical_file(condition_path, condition)
    binding["training_condition"]["sha256"] = condition_sha

    completion_path = Path(binding["completion_manifest"]["path"])
    completion = json.loads(completion_path.read_text())
    completion["training_condition_sha256"] = condition_sha
    completion["fixture_split_manifest_sha256"] = fixture_sha
    completion["training_dataset_sha256"] = binding["training_dataset_sha256"]
    binding["completion_manifest"]["sha256"] = _canonical_file(
        completion_path, completion
    )
    _reseal_plan(plan)


def _result(command, *, loss=0.03, runtime_patch=None, source_nonce=None):
    def value(flag):
        return command[command.index(flag) + 1]

    candidate = Path(value("--candidate-path"))
    output = Path(value("--output"))
    log = Path(f"{output}.comfy.log")
    log.write_bytes(b"clean evaluator log")
    source = {
        "god": {"commit": GOD_SHA, "tree": "7" * 40},
        "comfyui": {"commit": COMFY_SHA, "tree": "8" * 40},
        "tooling_nodes": {"commit": TOOLING_SHA, "tree": "9" * 40},
        "expected_commits": {
            "god": GOD_SHA,
            "comfyui": COMFY_SHA,
            "tooling_nodes": TOOLING_SHA,
        },
        "god_import_bindings": {},
        "workflow_path": "validator/evaluation/ComfyUI/workflows/krea2.json",
        "workflow_sha256": "a" * 64,
        "calibration_shim_sha256": provenance.file_sha256(Path(command[1])),
        "comfy_main_sha256": "b" * 64,
    }
    if source_nonce is not None:
        source["test_nonce"] = source_nonce
    runtime = {
        "fresh_comfy_process": True,
        "loopback": "127.0.0.1",
        "port": 8199,
        "cache": "comfy_default_fresh_process",
        "database": "memory",
        "api_nodes_disabled": True,
        "isolated_input_output_temp_user": True,
        "offline_environment": True,
        "custom_node_allowlist": ["comfyui-tooling-nodes"],
        "python": COMFY_RUNTIME,
        "driver_python": DRIVER_RUNTIME,
        "comfy_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "comfy_log_bytes": log.stat().st_size,
    }
    runtime.update(runtime_patch or {})
    result = {
        "schema": 2,
        "evaluator": "god_krea2_img2img_exact",
        "candidate": candidate.name,
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "candidate_bytes": candidate.stat().st_size,
        "staged_candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "comfy_lora_name": (
            "candidate-"
            + hashlib.sha256(candidate.read_bytes()).hexdigest()
            + ".safetensors"
        ),
        "model_type": "krea2",
        "dataset": value("--dataset"),
        "dataset_sha256": EVALUATION_DATASET_SHA,
        "image_count": 1,
        "scored_rows": [
            {
                "index": 0,
                "text_guided_loss": loss,
                "blank_prompt_loss": loss,
            }
        ],
        "base_name": "models--Comfy-Org--Krea-2.safetensors",
        "asset_sha256": {
            "diffusion_model": "c" * 64,
            "text_encoder": "d" * 64,
            "vae": "e" * 64,
        },
        "asset_bytes": {"diffusion_model": 1, "text_encoder": 1, "vae": 1},
        "steps": 28,
        "cfg": 1.0,
        "denoise": 0.85,
        "generations": 1,
        "validator_default_generations": 1,
        "seed_mode": "validator-exact-1",
        "master_seed": 42,
        "seeds": [42],
        "text_guided_losses": [loss],
        "blank_prompt_losses": [loss],
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


def _mock_runner(monkeypatch, *, runtime_patch=None, mixed_source=False):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        _result(
            command,
            loss=0.02 + len(calls) / 1000,
            runtime_patch=runtime_patch,
            source_nonce=len(calls) if mixed_source else None,
        )
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(batch, "_run_contained", run)
    return calls


def _completed_condition(root: Path, candidate: Path, train_dir: Path, eval_dir: Path):
    candidate_sha = provenance.file_sha256(candidate)
    resolution_sha = provenance.canonical_sha256([512, 768, 1024])
    precision_sha = provenance.canonical_sha256(
        {"train_dtype": "bf16", "save_dtype": "bf16"}
    )
    budget_plan = {
        "max_affordable_steps": 1500,
        "save_every": 200,
        "training_geometry": {
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "data_parallel_replicas": 1,
            "images_per_update": 1,
            "resolution_policy_sha256": resolution_sha,
            "precision_policy_sha256": precision_sha,
        },
    }
    return {
        "schema": 1,
        "kind": "forge-krea2-calibration-condition",
        "complete": True,
        "task_id": "concept-week5",
        "expected_repo_name": "local-repro",
        "model": "krea/Krea-2-Raw",
        "axes": {"lr": 0.0002, "depth_steps": 1500, "guidance": "on"},
        "fixed_controls": {"training_seed": 42565431},
        "derived": {"save_every": 200},
        "budget": {
            "plan": budget_plan,
            "plan_sha256": provenance.canonical_sha256(budget_plan),
        },
        "allowed_condition_config_differences": {
            "scientific_axes": [
                "/config/process/0/train/lr",
                "/config/process/0/train/steps",
                "/config/process/0/train/do_differential_guidance",
                "/config/process/0/train/differential_guidance_scale",
            ]
        },
        "resolved_config": {
            "config": {
                "process": [
                    {
                        "network": {"linear": 128, "linear_alpha": 128},
                        "save": {"save_every": 200, "dtype": "bf16"},
                        "datasets": [
                            {
                                "resolution": [512, 768, 1024],
                                "caption_dropout_rate": 0.05,
                            }
                        ],
                        "train": {
                            "steps": 1500,
                            "lr": 0.0002,
                            "optimizer": "adamw8bit",
                            "optimizer_params": {"weight_decay": 0.0001},
                            "loss_type": "mse",
                            "noise_scheduler": "flowmatch",
                            "batch_size": 1,
                            "gradient_accumulation": 1,
                            "dtype": "bf16",
                            "do_differential_guidance": True,
                            "differential_guidance_scale": 2,
                            "ema_config": {"use_ema": False, "ema_decay": 0.99},
                        },
                    }
                ]
            }
        },
        "selection": {
            "status": "selected_current_run",
            "source": "exact_final",
            "selected_step": 1500,
            "sha256": candidate_sha,
        },
        "artifacts": {"last_sha256": candidate_sha},
        "telemetry": {
            "schema": 1,
            "events": [
                {
                    "name": "toolkit_end",
                    "returncode": 0,
                    "stopped_by_deadline": False,
                },
                {"name": "toolkit_metrics", "last_step": 1500},
                {"name": "checkpoint_finalized", "status": "selected_current_run"},
                {"name": "run_complete"},
            ],
        },
        "dataset_after_split": {
            "training": ladder._fingerprint_path(train_dir),
            "holdout": ladder._fingerprint_path(eval_dir),
        },
        "provenance": {
            "dataset": {"sha256": "5" * 64},
            "base_assets": {"base_model": {"sha256": "6" * 64}},
            "runtime": {"sha256": TRAINING_RUNTIME_SHA},
        },
    }


def test_provenance_is_pretraining_canonical_and_requires_external_approval(tmp_path):
    path, manifest, config, artifact = _source_provenance(tmp_path, "arm")
    assert set(manifest["files"]) == {"source_config", "source_artifact"}
    assert "candidate" not in manifest
    provenance.validate_manifest(
        json.loads(path.read_text()),
        source_config_path=config,
        source_artifact_path=artifact,
    )
    assert path.read_bytes() == provenance.canonical_bytes(manifest) + b"\n"
    with pytest.raises(FileExistsError):
        provenance.publish_exclusive(path, manifest)

    role_root = tmp_path / "role"
    role_root.mkdir()
    plan = _plan(role_root, arms=("arm",), reviewer="human owner")
    with pytest.raises(ValueError, match="role label"):
        batch._validate_plan(plan)


def test_local_reproduction_disclosure_is_required_and_fail_closed(tmp_path):
    path, manifest, _config, _artifact = _source_provenance(tmp_path, "arm")
    disclosure = manifest["local_reproduction_disclosure"]
    assert disclosure["execution_authorized"] is False
    assert disclosure["adapted_fields"][0]["name"] == "depth policy"

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["local_reproduction_disclosure"]["execution_authorized"] = True
    body = {key: value for key, value in tampered.items() if key != "manifest_sha256"}
    tampered["manifest_sha256"] = provenance.canonical_sha256(body)
    with pytest.raises(ValueError, match="disclosure identity"):
        provenance.validate_manifest(tampered)

    missing = _metadata("another-arm")
    del missing["local_reproduction_disclosure"]
    with pytest.raises(ValueError, match="metadata keys mismatch"):
        provenance.build_manifest(
            missing,
            source_config_path=tmp_path / "arm.source.yaml",
            source_artifact_path=tmp_path / "arm.source.safetensors",
            field_ledger_path=tmp_path / "arm.field-ledger.json",
            task_raw_path=tmp_path / "arm.task.raw.json",
            tournament_raw_path=tmp_path / "arm.tournament.raw.json",
            revision_manifest_path=tmp_path / "arm.revision.raw.json",
        )


def test_source_unknowns_and_predeclared_values_are_separate_contracts():
    source = _source_recipe()
    disclosure = {
        "schema": 1,
        "kind": "forge-krea-local-reproduction-disclosure",
        "execution_authorized": False,
        "adapted_fields": [
            {
                "name": "source-unknown dropout fixed locally",
                "source_recipe_fields": ["dropout"],
                "local_policy": "use the separately declared local control",
                "evidence": "source absence is not a local-value observation",
            }
        ],
        "source_unknown_fields": [
            {
                "field": "dropout",
                "source_classification": "unknown",
                "source_pointers": [],
                "source_value": None,
                "evidence": source["fields"]["dropout"]["evidence"],
            }
        ],
        "predeclared_local_values": [
            {
                "field": "dropout",
                "value": 0.05,
                "basis": "predeclared local control, not source evidence",
            }
        ],
        "claim_limit": "not execution approval",
    }
    target = {
        "mode": "local_reproduction",
        "model_type": "krea2",
        "source_artifact_role": "reference_only",
        "candidate_role": "local_training_output",
        "description": "local test",
    }
    assert (
        provenance._local_reproduction_disclosure(
            disclosure,
            adaptation_target=target,
            source_recipe=source,
        )
        == disclosure
    )

    conflated = json.loads(json.dumps(disclosure))
    conflated["source_unknown_fields"][0]["source_value"] = 0.05
    with pytest.raises(ValueError, match="contradicts source provenance"):
        provenance._local_reproduction_disclosure(
            conflated,
            adaptation_target=target,
            source_recipe=source,
        )

    omitted = json.loads(json.dumps(disclosure))
    omitted["source_unknown_fields"] = []
    omitted["predeclared_local_values"] = []
    with pytest.raises(ValueError, match="must be disclosed exactly once"):
        provenance._local_reproduction_disclosure(
            omitted,
            adaptation_target=target,
            source_recipe=source,
        )


def test_local_reproduction_uses_distinct_disjoint_train_and_eval_sets(tmp_path):
    plan = _plan(tmp_path, arms=("arm",))
    _, eval_sha, candidates, _ = batch._validate_plan(plan)
    binding = candidates[0]["candidate_binding"]
    assert eval_sha == EVALUATION_DATASET_SHA
    assert binding["training_dataset_sha256"] == TRAINING_DATASET_SHA
    assert binding["evaluation_dataset_sha256"] == EVALUATION_DATASET_SHA
    assert binding["training_dataset_sha256"] != binding["evaluation_dataset_sha256"]
    source_sha = candidates[0]["provenance"]["files"]["source_artifact"]["sha256"]
    assert candidates[0]["sha256"] != source_sha

    fixture_path = Path(
        plan["candidates"][0]["candidate_binding"]["fixture_split_manifest"]["path"]
    )
    fixture = json.loads(fixture_path.read_text())
    fixture["evaluation_rows"][0] = dict(fixture["training_rows"][0])
    plan["candidates"][0]["candidate_binding"]["fixture_split_manifest"]["sha256"] = (
        _canonical_file(fixture_path, fixture)
    )
    with pytest.raises(ValueError, match="not disjoint"):
        batch._validate_plan(plan)


def test_condition_and_trainer_completion_are_exact_and_bound(tmp_path):
    plan = _plan(tmp_path, arms=("arm",))
    row = plan["candidates"][0]
    condition_binding = row["candidate_binding"]["training_condition"]
    condition_path = Path(condition_binding["path"])
    condition = json.loads(condition_path.read_text())
    condition["seed"] = 7
    condition["unknown"] = True
    condition_binding["sha256"] = _canonical_file(condition_path, condition)
    with pytest.raises(ValueError, match="keys mismatch"):
        batch._validate_plan(plan)

    completion_root = tmp_path / "completion"
    completion_root.mkdir()
    completion_plan = _plan(completion_root, arms=("arm",))
    completion_binding = completion_plan["candidates"][0]["candidate_binding"][
        "completion_manifest"
    ]
    completion_path = Path(completion_binding["path"])
    completion = json.loads(completion_path.read_text())
    completion["natural_completion"] = False
    completion_binding["sha256"] = _canonical_file(completion_path, completion)
    with pytest.raises(ValueError, match="incomplete or unbound"):
        batch._validate_plan(completion_plan)


def test_direct_public_scoring_requires_matched_concept_and_approval(tmp_path):
    plan = _plan(
        tmp_path,
        arms=("arm",),
        mode="direct_public_artifact",
        matched_dataset_sha=EVALUATION_DATASET_SHA,
    )
    _, _, candidates, _ = batch._validate_plan(plan)
    assert candidates[0]["candidate_binding"]["mode"] == "direct_public_artifact"

    mismatch_root = tmp_path / "mismatch"
    mismatch_root.mkdir()
    mismatch = _plan(
        mismatch_root,
        arms=("arm",),
        mode="direct_public_artifact",
        matched_dataset_sha="1" * 64,
    )
    with pytest.raises(ValueError, match="lacks matched concept"):
        batch._validate_plan(mismatch)


def test_evaluator_identity_requirements_are_fail_closed(tmp_path):
    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    missing = _plan(missing_root, arms=("arm",))
    del missing["evaluator"]["expected_tooling_commit"]
    with pytest.raises(ValueError, match="keys mismatch"):
        batch._validate_plan(missing)

    script_root = tmp_path / "script"
    script_root.mkdir()
    wrong_script = _plan(script_root, arms=("arm",))
    wrong_script["evaluator"]["expected_evaluator_script_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="evaluator-script identity mismatch"):
        batch._validate_plan(wrong_script)

    identity_root = tmp_path / "identity"
    identity_root.mkdir()
    identity_plan = _plan(identity_root, arms=("arm",))
    identity_plan["evaluator"]["expected_runtime_identity"][
        "driver_python_identity_sha256"
    ] = ("2" * 64)
    _reseal_plan(identity_plan)
    _, _, candidates, evaluator = batch._validate_plan(identity_plan)
    result_command = batch._evaluator_command(
        evaluator_script=_CALIBRATION / "evaluate_krea_local.py",
        dataset=Path(identity_plan["dataset"]["path"]),
        candidate=candidates[0],
        result_path=tmp_path / "identity-result.json",
        evaluator=evaluator,
    )
    _result(result_command)
    result = json.loads((tmp_path / "identity-result.json").read_text())
    with pytest.raises(ValueError, match="runtime identity mismatch"):
        batch._validate_result(
            result,
            candidate=candidates[0],
            dataset=Path(identity_plan["dataset"]["path"]),
            expected_dataset_sha256=EVALUATION_DATASET_SHA,
            expected_dataset_identity=None,
            evaluator=evaluator,
            evaluator_script_sha=identity_plan["evaluator"][
                "expected_evaluator_script_sha256"
            ],
            log_path=Path(f"{tmp_path / 'identity-result.json'}.comfy.log"),
        )


@pytest.mark.parametrize(
    ("runtime_patch", "label"),
    [
        ({"offline_environment": False}, "offline"),
        ({"isolated_input_output_temp_user": False}, "isolated"),
        ({"loopback": "0.0.0.0"}, "loopback"),
        ({"database": "disk.sqlite"}, "database"),
        ({"api_nodes_disabled": False}, "api"),
        ({"fresh_comfy_process": False}, "fresh"),
    ],
)
def test_each_runtime_invariant_is_required(
    tmp_path, monkeypatch, runtime_patch, label
):
    plan = _plan(tmp_path, arms=("arm",))
    _mock_runner(monkeypatch, runtime_patch=runtime_patch)
    output = tmp_path / f"{label}.json"
    with pytest.raises(ValueError, match="unsafe evaluator runtime"):
        batch.run_batch(
            plan,
            results_dir=tmp_path / f"{label}-results",
            output=output,
        )
    assert not output.exists()


def test_batch_fresh_process_full_coverage_and_mixed_envelope_rejection(
    tmp_path, monkeypatch
):
    plan = _plan(tmp_path, arms=("z-arm", "a-arm"))
    calls = _mock_runner(monkeypatch)
    output = tmp_path / "aggregate.json"
    aggregate = batch.run_batch(plan, results_dir=tmp_path / "results", output=output)
    assert len(calls) == 2
    assert [row["candidate_id"] for row in aggregate["candidates"]] == [
        "a-arm-repro",
        "z-arm-repro",
    ]
    assert all(
        row["candidate_binding"]["source_normalization_approval"]["reviewer_identity"]
        == "Atulya Shetty"
        for row in aggregate["candidates"]
    )
    assert aggregate["sealed_plan_approval"]["reviewer_identity"] == "Jordan Example"
    assert aggregate["common_training_envelope_sha256"] == provenance.canonical_sha256(
        aggregate["common_training_envelope"]
    )
    assert not (tmp_path / "forge_holdout_scores.json").exists()

    mixed_root = tmp_path / "mixed"
    mixed_root.mkdir()
    mixed = _plan(mixed_root, arms=("arm-a", "arm-b"))
    _mock_runner(monkeypatch, mixed_source=True)
    with pytest.raises(RuntimeError, match="common evaluation envelope"):
        batch.run_batch(
            mixed,
            results_dir=tmp_path / "mixed-results",
            output=tmp_path / "mixed.json",
        )


def test_partial_or_stale_evidence_never_publishes(tmp_path, monkeypatch):
    plan = _plan(tmp_path, arms=("arm",))

    def partial(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        Path(f"{output}.tmp").write_text("partial")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(batch, "_run_contained", partial)
    output = tmp_path / "aggregate.json"
    with pytest.raises(RuntimeError, match="partial result"):
        batch.run_batch(plan, results_dir=tmp_path / "results", output=output)
    assert not output.exists()
    with pytest.raises(ValueError, match="production selection filename"):
        batch.run_batch(
            plan,
            results_dir=tmp_path / "other-results",
            output=tmp_path / "forge_holdout_scores.json",
        )


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="requires POSIX process groups")
def test_timeout_cleanup_kills_spawned_child_process_group(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    code = (
        "import pathlib,subprocess,time;"
        "p=subprocess.Popen(['sleep','60']);"
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(p.pid));"
        "time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5.0
    while not child_pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert child_pid_file.exists()
    child_pid = int(child_pid_file.read_text())
    batch._terminate_process_group(process, term_grace_s=0.2)
    assert process.poll() is not None
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"spawned child survived process-group cleanup: {child_pid}")
