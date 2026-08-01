from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import struct

import pytest

from ops.calibration import krea_provenance
from ops.calibration import krea_stage2_execution as stage2
from ops.calibration import krea_stage2_training_evidence as training_evidence


def _sha(label: str) -> str:
    return krea_provenance.canonical_sha256({"label": label})


def _binding(label: str, semantic_key: str) -> dict[str, str]:
    return {"file_sha256": _sha(f"{label}-file"), semantic_key: _sha(label)}


_TEST_BASE = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=5)


def _time(seconds: int) -> str:
    return (_TEST_BASE + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _actor(role: str = "execution_plan_reviewer") -> dict[str, object]:
    return {
        "actor_id": f"codex-stage2-{role}",
        "display_name": f"Codex Stage2 {role}",
        "role": role,
        "review_instance_id": f"week5-stage2-{role}-20260731",
        "non_human": True,
    }


def _candidates() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": "K1-final691",
            "family_id": "K1",
            "sha256": _sha("candidate"),
            "bytes": 228_587_720,
            "step": 691,
            "zero_control": False,
        },
        {
            "candidate_id": "zero-control",
            "family_id": "ZERO",
            "sha256": _sha("zero"),
            "bytes": 228_587_720,
            "step": None,
            "zero_control": True,
        },
    ]


def _checkpoint_selection(
    *,
    planned_steps: int = 691,
    numerator: int = 7,
    denominator: int = 8,
    checkpoint_rule_sha256: str | None = None,
) -> dict[str, object]:
    schedule = stage2.krea_budget.candidate_schedule(planned_steps)
    selected = min(
        schedule.candidates,
        key=lambda candidate: (
            abs(candidate.step * denominator - numerator * planned_steps),
            candidate.step,
        ),
    )
    return {
        "checkpoint_rule_sha256": checkpoint_rule_sha256 or _sha("checkpoint-rule"),
        "target_fraction": {
            "numerator": numerator,
            "denominator": denominator,
        },
        "selected_step": selected.step,
        "denominator_steps": planned_steps,
        "mapping_rule": "nearest_current_candidate_ties_choose_earlier_step",
    }


def _payload(*, boundary: bool = False, root: Path = Path("/tmp/stage2")) -> dict:
    if boundary:
        phase = "boundary"
        cell = "B-0p5-small"
        fixture = cell
        seed_role = "A"
        hours = "0.5"
    else:
        phase = "confirmation"
        cell = "C1-A"
        fixture = "C1"
        seed_role = "A"
        hours = "0.75"
    task_id = f"stage2-{cell.lower()}"
    expected_repo_name = f"stage2-{cell.lower()}-k1"
    throughput_profile = {
        "path": str(root / "throughput-profile.json"),
        "file_sha256": _sha("throughput-profile-file"),
        "profile_sha256": _sha("throughput-profile"),
    }
    return {
        "schema": 1,
        "kind": stage2.PLAN_KIND,
        "phase": phase,
        "cell_id": cell,
        "fixture_id": fixture,
        "seed_role": seed_role,
        "seed": 42565431,
        "hours": hours,
        "task_id": task_id,
        "expected_repo_name": expected_repo_name,
        "model": "krea/Krea-2-Raw",
        "model_type": "krea2",
        "trigger_word": "SN56" if boundary else None,
        "candidate_universe": _candidates(),
        "training_candidate_id": "K1-final691",
        "family_role": "candidate",
        "calibration_profile": "K1",
        "planned_steps": 691,
        "checkpoint_selection": _checkpoint_selection(),
        "throughput_profile": throughput_profile,
        "throughput_evidence": {
            "raw_samples": {
                "path": str(root / "throughput-raw.json"),
                "file_sha256": _sha("throughput-raw-file"),
                "raw_sample_manifest_sha256": _sha("throughput-raw"),
            },
            "margin_policy": {
                "path": str(root / "throughput-margin.json"),
                "file_sha256": _sha("throughput-margin-file"),
                "margin_policy_sha256": _sha("throughput-margin"),
            },
            "end_to_end_validation": {
                "path": str(root / "throughput-e2e.json"),
                "file_sha256": _sha("throughput-e2e-file"),
                "end_to_end_validation_sha256": _sha("throughput-e2e"),
            },
        },
        "execution_environment_profile": dict(throughput_profile),
        "base_model_identity_sha256": _sha("training-assets"),
        "base_asset_attestation": {
            "path": str(root / "base-asset-attestation.json"),
            "file_sha256": _sha("base-asset-attestation-file"),
            "attestation_sha256": _sha("base-asset-attestation"),
        },
        "fixture_manifest": {
            "path": str(root / "fixture-manifest.json"),
            "file_sha256": _sha("fixture-manifest-file"),
            "manifest_sha256": _sha("fixture-manifest"),
        },
        "waiver_finalist_freeze": _binding("freeze", "freeze_sha256"),
        "confirmation_materialization": _binding(
            "materialization", "materialization_sha256"
        ),
        "owner_ratification": _binding("ratification", "ratification_sha256"),
        "gpu_execution_authorization": _binding(
            "gpu-authorization", "gpu_execution_authorization_sha256"
        ),
        "production_identity": _binding("identity", "production_identity_sha256"),
        "execution_surface_policy_sha256": _sha("policy"),
        "delegated_role_contract_sha256": _sha("roles"),
        "production_image_id": f"sha256:{_sha('image')}",
        "entrypoint_argv": [
            "--task-id",
            task_id,
            "--model",
            "krea/Krea-2-Raw",
            "--model-type",
            "krea2",
            "--expected-repo-name",
            expected_repo_name,
            "--hours-to-complete",
            hours,
        ] + (["--trigger-word", "SN56"] if boundary else []),
        "mounts": [
            {
                "source": str(root / "models"),
                "destination": "/cache/models/krea--Krea-2-Raw",
                "read_only": True,
                "purpose": "base_model",
            },
            {
                "source": str(root / "text-encoder"),
                "destination": "/cache/hf_cache/Qwen--Qwen3-VL-4B-Instruct",
                "read_only": True,
                "purpose": "text_encoder",
            },
            {
                "source": str(root / "datasets"),
                "destination": "/cache/datasets",
                "read_only": True,
                "purpose": "dataset_cache",
            },
            {
                "source": str(root / "checkpoints"),
                "destination": "/app/checkpoints",
                "read_only": False,
                "purpose": "checkpoints",
            },
            {
                "source": str(root / "evidence"),
                "destination": "/run-evidence",
                "read_only": False,
                "purpose": "run_evidence",
            },
        ],
        "network_mode": "none",
        "runtime": "nvidia",
        "created_at_utc": _time(0),
        "gpu_execution_authorized": True,
        "release_authorized": False,
        "production_mutation_authorized": False,
    }


def _plan(*, boundary: bool = False, root: Path = Path("/tmp/stage2")) -> dict:
    return stage2.seal_plan(_payload(boundary=boundary, root=root))


def test_trigger_scope_is_fail_closed_between_legacy_confirmation_and_boundary() -> None:
    confirmation = _payload()
    confirmation["trigger_word"] = "invented-after-freeze"
    confirmation["entrypoint_argv"].extend(
        ["--trigger-word", "invented-after-freeze"]
    )
    with pytest.raises(ValueError, match="preserve the sealed null trigger"):
        stage2.seal_plan(confirmation)

    boundary = _payload(boundary=True)
    boundary["trigger_word"] = None
    index = boundary["entrypoint_argv"].index("--trigger-word")
    del boundary["entrypoint_argv"][index : index + 2]
    with pytest.raises(ValueError, match="boundary trigger_word"):
        stage2.seal_plan(boundary)


def _approval(plan: dict) -> dict:
    return stage2.build_approval(
        plan,
        reviewer_actor=_actor(),
        approved_at_utc=_time(1),
    )


def _completion(plan: dict, approval: dict) -> dict:
    control = {
        "file_sha256": _sha("config-control-file"),
        "receipt_sha256": _sha("config-control-receipt"),
        "config_sha256": _sha("effective-config"),
    }
    terminal = {
        "file_sha256": _sha("training-terminal-file"),
        "receipt_sha256": _sha("training-terminal-receipt"),
    }
    selection = {
        "file_sha256": _sha("checkpoint-selection-file"),
        "receipt_sha256": _sha("checkpoint-selection-receipt"),
    }
    selected_step = plan["checkpoint_selection"]["selected_step"]
    selected_file = (
        f"{plan['expected_repo_name']}.safetensors"
        if selected_step == plan["planned_steps"]
        else f"{plan['expected_repo_name']}_{selected_step:09d}.safetensors"
    )
    body = {
        "schema": 1,
        "kind": stage2.COMPLETION_KIND,
        "execution_plan_sha256": plan["plan_sha256"],
        "execution_approval_sha256": approval["approval_sha256"],
        "production_image_id": plan["production_image_id"],
        "phase": plan["phase"],
        "cell_id": plan["cell_id"],
        "started_at_utc": _time(2),
        "ended_at_utc": _time(3),
        "returncode": 0,
        "natural_completion": True,
        "fallback_used": False,
        "mechanics": {
            "natural_completion": True,
            "planned_steps_completed": True,
            "upload_ready": True,
            "clean_telemetry": True,
            "decision_completed_before_export_reserve": True,
            "fallback_used": False,
        },
        "artifact_manifest": [
            {"path": "stdout.log", "bytes": 12, "sha256": _sha("stdout")},
            {
                "path": f"checkpoints/{selected_file}",
                "bytes": 100,
                "sha256": _sha("selected"),
            },
            {
                "path": "checkpoints/last.safetensors",
                "bytes": 100,
                "sha256": _sha("selected"),
            },
            {
                "path": "evidence/config-control.json",
                "bytes": 100,
                "sha256": control["file_sha256"],
            },
            {
                "path": "evidence/effective-config.yaml",
                "bytes": 100,
                "sha256": control["config_sha256"],
            },
            {
                "path": "evidence/training-terminal.json",
                "bytes": 100,
                "sha256": terminal["file_sha256"],
            },
            {
                "path": "evidence/forge_checkpoint_selection.json",
                "bytes": 100,
                "sha256": selection["file_sha256"],
            },
        ],
        "config_control_receipt": control,
        "training_terminal_receipt": terminal,
        "checkpoint_selection_receipt": selection,
        "postrun_identity_sha256": plan["production_identity"][
            "production_identity_sha256"
        ],
        "network_mode": "none",
        "runtime": "nvidia",
        "gpu_device": 0,
        "strict_discovery_replayed": False,
        "release_authorized": False,
        "production_mutation_authorized": False,
    }
    return {**body, "completion_sha256": krea_provenance.canonical_sha256(body)}


def _reseal_plan(plan: dict) -> dict:
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    return {**body, "plan_sha256": krea_provenance.canonical_sha256(body)}


def _reseal_approval(approval: dict) -> dict:
    body = {key: value for key, value in approval.items() if key != "approval_sha256"}
    return {**body, "approval_sha256": krea_provenance.canonical_sha256(body)}


def _reseal_completion(completion: dict) -> dict:
    body = {
        key: value for key, value in completion.items() if key != "completion_sha256"
    }
    return {**body, "completion_sha256": krea_provenance.canonical_sha256(body)}


def _fake_authority_bundle(plan: dict) -> tuple[dict, dict, dict]:
    plan = deepcopy(plan)
    selected = next(
        row
        for row in plan["candidate_universe"]
        if row["candidate_id"] == plan["training_candidate_id"]
    )
    selected["family_id"] = "K0"
    plan["calibration_profile"] = "K0"
    plan["family_role"] = "control"
    plan["planned_steps"] = 100
    plan["checkpoint_selection"] = _checkpoint_selection(
        planned_steps=100,
        numerator=1,
        denominator=1,
    )
    plan["throughput_profile"] = None
    plan["throughput_evidence"] = None
    plan["expected_repo_name"] = f"stage2-{plan['cell_id'].lower()}-k0"
    repo_index = plan["entrypoint_argv"].index("--expected-repo-name") + 1
    plan["entrypoint_argv"][repo_index] = plan["expected_repo_name"]
    records = {
        "request": {"kind": "request-test"},
        "ratification": {"kind": "ratification-test"},
        "reveal": {"kind": "reveal-test"},
        "materialization": {"kind": "materialization-test", "files": []},
        "production_identity": {"kind": "production-identity-test"},
        "sealed_inventory": {
            "kind": "sealed-inventory-test",
            "inventory_sha256": _sha("authority-inventory"),
        },
    }
    files = {
        name: stage2.krea_confirmation_admission.canonical_file_sha256(record)
        for name, record in records.items()
    }
    authorization = {
        "waiver_freeze_sha256": _sha("authority-freeze"),
        "waiver_freeze_file_sha256": _sha("authority-freeze-file"),
        "materialization_sha256": _sha("authority-materialization"),
        "materialization_file_sha256": files["materialization"],
        "ratification_sha256": _sha("authority-ratification"),
        "ratification_file_sha256": files["ratification"],
        "gpu_execution_authorization_sha256": _sha("authority-gpu"),
        "production_identity_sha256": _sha("authority-production"),
        "production_identity_file_sha256": files["production_identity"],
        "policy_sha256": _sha("authority-policy"),
        "delegated_review_contract_sha256": _sha("authority-contract"),
        "sealed_inventory_sha256": records["sealed_inventory"][
            "inventory_sha256"
        ],
        "sealed_inventory_file_sha256": files["sealed_inventory"],
        "image_id": f"sha256:{_sha('authority-image')}",
    }
    records["gpu_execution_authorization"] = authorization
    files["gpu_execution_authorization"] = (
        stage2.krea_confirmation_admission.canonical_file_sha256(authorization)
    )
    controls = {
        **records,
        **{f"{name}_file_sha256": digest for name, digest in files.items()},
        "waiver_finalist_freeze": {"kind": "freeze-test"},
    }
    plan["waiver_finalist_freeze"] = {
        "file_sha256": authorization["waiver_freeze_file_sha256"],
        "freeze_sha256": authorization["waiver_freeze_sha256"],
    }
    plan["confirmation_materialization"] = {
        "file_sha256": authorization["materialization_file_sha256"],
        "materialization_sha256": authorization["materialization_sha256"],
    }
    plan["owner_ratification"] = {
        "file_sha256": authorization["ratification_file_sha256"],
        "ratification_sha256": authorization["ratification_sha256"],
    }
    plan["gpu_execution_authorization"] = {
        "file_sha256": files["gpu_execution_authorization"],
        "gpu_execution_authorization_sha256": authorization[
            "gpu_execution_authorization_sha256"
        ],
    }
    plan["production_identity"] = {
        "file_sha256": authorization["production_identity_file_sha256"],
        "production_identity_sha256": authorization["production_identity_sha256"],
    }
    plan["execution_surface_policy_sha256"] = authorization["policy_sha256"]
    plan["delegated_role_contract_sha256"] = authorization[
        "delegated_review_contract_sha256"
    ]
    plan["production_image_id"] = authorization["image_id"]
    return _reseal_plan(plan), controls, authorization


def _write_safetensors(path: Path, *, first: bytes = b"\x01\x02\x03\x04") -> None:
    header = {
        "__metadata__": {"source": "stage2-test"},
        "adapter.a": {
            "dtype": "U8",
            "shape": [4],
            "data_offsets": [0, 4],
        },
        "adapter.b": {
            "dtype": "F32",
            "shape": [2],
            "data_offsets": [4, 12],
        },
    }
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + first + b"\x01" * 8)


def _write_stage2_receipts(plan: dict, evidence_root: Path) -> tuple[dict, dict, dict]:
    evidence_root.mkdir(parents=True, exist_ok=True)
    config_bytes = b"config: stage2-test\n"
    config_path = evidence_root / "effective-config.yaml"
    config_path.write_bytes(config_bytes)
    frozen = stage2.krea_calibration_profiles.profile_for_id(
        plan["calibration_profile"]
    )
    runtime_selection = stage2._runtime_checkpoint_selection(plan)
    save_every = (
        stage2.recipe.kill_safe_save_every(plan["planned_steps"], 200)
        if frozen.profile_id == "K0"
        else (plan["planned_steps"] + 7) // 8
    )
    effective = {
        "config_name": plan["expected_repo_name"],
        "training_folder": f"/app/checkpoints/{plan['task_id']}",
        "trigger_word": plan["trigger_word"],
        "model_arch": "krea2",
        "model_name_or_path": "/cache/models/krea--Krea-2-Raw",
        "model_kwargs": {
            "text_encoder_path": "/cache/hf_cache/Qwen--Qwen3-VL-4B-Instruct",
            "vae_path": "/cache/models/krea--Krea-2-Raw",
        },
        "dataset_folder_path": "/dataset/images",
        "network_rank": frozen.rank,
        "network_alpha": frozen.alpha,
        "optimizer": frozen.optimizer,
        "optimizer_params": dict(frozen.optimizer_parameters),
        "loss": frozen.loss,
        "guidance_enabled": True,
        "guidance_scale": frozen.guidance,
        "learning_rate": frozen.learning_rate,
        "dropout": frozen.dropout,
        "ema": {"use_ema": frozen.ema, "ema_decay": 0.99},
        "steps": plan["planned_steps"],
        "save_every": save_every,
        "push_to_hub": False,
        "batch_size": 1,
        "gradient_accumulation": 1,
        "resolution": [512, 768, 1024],
        "train_dtype": "bf16",
        "save_dtype": "bf16",
        "cache_latents_to_disk": False,
        "cache_text_embeddings": False,
        "compile": False,
        "dataloader_workers": 0,
    }
    config_sha = krea_provenance.file_sha256(config_path)
    control_body = {
        "schema": 1,
        "kind": "forge-krea-stage2-config-control-receipt",
        "execution_plan_sha256": plan["plan_sha256"],
        "profile_id": frozen.profile_id,
        "profile_sha256": frozen.profile_sha256,
        "training_seed": plan["seed"],
        "throughput_profile_sha256": plan["throughput_profile"]["profile_sha256"],
        "config_sha256": config_sha,
        "effective_config_file": {
            "path": "effective-config.yaml",
            "bytes": len(config_bytes),
            "sha256": config_sha,
        },
        "effective_recipe": effective,
        "effective_recipe_sha256": krea_provenance.canonical_sha256(effective),
        "checkpoint_selection": runtime_selection,
        "release_authorized": False,
    }
    control = {
        **control_body,
        "receipt_sha256": krea_provenance.canonical_sha256(control_body),
    }
    control_path = evidence_root / "config-control.json"
    control_path.write_bytes(krea_provenance.canonical_bytes(control) + b"\n")
    terminal_body = {
        "schema": 1,
        "kind": "forge-krea-stage2-training-terminal-receipt",
        "execution_plan_sha256": plan["plan_sha256"],
        "profile_id": frozen.profile_id,
        "profile_sha256": frozen.profile_sha256,
        "training_seed": plan["seed"],
        "planned_steps": plan["planned_steps"],
        "last_step": plan["planned_steps"],
        "trainer_returncode": 0,
        "stopped_by_deadline": False,
        "planned_steps_completed": True,
        "natural_completion": True,
        "config_control_file_sha256": krea_provenance.file_sha256(control_path),
        "checkpoint_selection": runtime_selection,
        "release_authorized": False,
    }
    terminal = {
        **terminal_body,
        "receipt_sha256": krea_provenance.canonical_sha256(terminal_body),
    }
    terminal_path = evidence_root / "training-terminal.json"
    terminal_path.write_bytes(krea_provenance.canonical_bytes(terminal) + b"\n")
    selected_step = runtime_selection["selected_step"]
    selected_file = (
        f"{plan['expected_repo_name']}.safetensors"
        if selected_step == plan["planned_steps"]
        else f"{plan['expected_repo_name']}_{selected_step:09d}.safetensors"
    )
    checkpoint_mount = next(
        Path(mount["source"])
        for mount in plan["mounts"]
        if mount["purpose"] == "checkpoints"
    )
    checkpoint_root = checkpoint_mount / plan["task_id"] / plan["expected_repo_name"]
    selected_path = checkpoint_root / selected_file
    promoted_path = checkpoint_root / "last.safetensors"
    _write_safetensors(selected_path)
    _write_safetensors(promoted_path)
    selection_record = {
        "schema": 1,
        "status": "selected_current_run",
        "context": "training",
        "source": "frozen_checkpoint_fraction",
        "selected_file": selected_file,
        "output_file": "last.safetensors",
        "selected_step": selected_step,
        "sha256": krea_provenance.file_sha256(selected_path),
        "reason": "selected the frozen Stage-2 checkpoint fraction",
        "score": None,
        "metric": None,
        "direction": None,
        "training_loss_is_proxy_not_validator_metric": False,
        "metric_is_proxy_not_validator_metric": False,
        "reference_file": None,
        "reference_score": None,
        "score_advantage": None,
        "required_advantage": None,
        "margin_policy": None,
        "calibration_id": None,
        "current_candidates_discovered": 1,
        "current_candidates_valid": 1,
        "created_unix": 1,
        "checkpoint_target": {
            "fraction_numerator": runtime_selection["target_fraction"]["numerator"],
            "fraction_denominator": runtime_selection["target_fraction"]["denominator"],
            "selection_rule": runtime_selection["mapping_rule"],
        },
        "planned_steps": plan["planned_steps"],
    }
    selection_path = evidence_root / "forge_checkpoint_selection.json"
    selection_path.write_bytes(krea_provenance.canonical_bytes(selection_record))
    return (
        {
            "file_sha256": krea_provenance.file_sha256(control_path),
            "receipt_sha256": control["receipt_sha256"],
            "config_sha256": config_sha,
        },
        {
            "file_sha256": krea_provenance.file_sha256(terminal_path),
            "receipt_sha256": terminal["receipt_sha256"],
        },
        {
            "file_sha256": krea_provenance.file_sha256(selection_path),
            "receipt_sha256": krea_provenance.canonical_sha256(selection_record),
        },
    )


def _materialized_completion(
    plan: dict, approval: dict, *, run_root: Path, roots: Path
) -> dict:
    stdout = run_root / "container.stdout"
    stdout.parent.mkdir(parents=True)
    stdout.write_bytes(b"completed\n")
    checkpoint = (
        roots
        / "checkpoints"
        / plan["task_id"]
        / plan["expected_repo_name"]
        / "final.safetensors"
    )
    _write_safetensors(checkpoint)
    evidence_root = roots / "evidence" / plan["plan_sha256"]
    control, terminal, selection = _write_stage2_receipts(plan, evidence_root)
    runtime_selection = stage2._runtime_checkpoint_selection(plan)
    selected_file = (
        f"{plan['expected_repo_name']}.safetensors"
        if runtime_selection["selected_step"] == plan["planned_steps"]
        else (
            f"{plan['expected_repo_name']}_"
            f"{runtime_selection['selected_step']:09d}.safetensors"
        )
    )
    checkpoint_root = checkpoint.parent
    artifacts = [
        {
            "path": "run/container.stdout",
            "bytes": stdout.stat().st_size,
            "sha256": krea_provenance.file_sha256(stdout),
        },
        {
            "path": "checkpoints/final.safetensors",
            "bytes": checkpoint.stat().st_size,
            "sha256": krea_provenance.file_sha256(checkpoint),
        },
    ]
    for name in (selected_file, "last.safetensors"):
        path = checkpoint_root / name
        artifacts.append(
            {
                "path": f"checkpoints/{name}",
                "bytes": path.stat().st_size,
                "sha256": krea_provenance.file_sha256(path),
            }
        )
    for name in (
        "config-control.json",
        "effective-config.yaml",
        "training-terminal.json",
        "forge_checkpoint_selection.json",
    ):
        path = evidence_root / name
        artifacts.append(
            {
                "path": f"evidence/{name}",
                "bytes": path.stat().st_size,
                "sha256": krea_provenance.file_sha256(path),
            }
        )
    body = {
        "schema": 1,
        "kind": stage2.COMPLETION_KIND,
        "execution_plan_sha256": plan["plan_sha256"],
        "execution_approval_sha256": approval["approval_sha256"],
        "production_image_id": plan["production_image_id"],
        "phase": plan["phase"],
        "cell_id": plan["cell_id"],
        "started_at_utc": _time(2),
        "ended_at_utc": _time(3),
        "returncode": 0,
        "natural_completion": True,
        "fallback_used": False,
        "mechanics": {
            "natural_completion": True,
            "planned_steps_completed": True,
            "upload_ready": True,
            "clean_telemetry": True,
            "decision_completed_before_export_reserve": True,
            "fallback_used": False,
        },
        "artifact_manifest": artifacts,
        "config_control_receipt": control,
        "training_terminal_receipt": terminal,
        "checkpoint_selection_receipt": selection,
        "postrun_identity_sha256": plan["production_identity"][
            "production_identity_sha256"
        ],
        "network_mode": "none",
        "runtime": "nvidia",
        "gpu_device": 0,
        "strict_discovery_replayed": False,
        "release_authorized": False,
        "production_mutation_authorized": False,
    }
    return {**body, "completion_sha256": krea_provenance.canonical_sha256(body)}


@pytest.mark.parametrize("boundary", [False, True])
def test_stage2_plan_approval_completion_round_trip(boundary: bool) -> None:
    plan = _plan(boundary=boundary)
    approval = _approval(plan)
    completion = _completion(plan, approval)
    assert stage2.validate_plan(plan) == plan
    assert stage2.validate_approval(approval, plan=plan) == approval
    assert (
        stage2.validate_completion(completion, plan=plan, approval=approval)
        == completion
    )


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda p: p.__setitem__("seed", 42), "seed or hours"),
        (lambda p: p.__setitem__("hours", "1.0"), "seed or hours"),
        (lambda p: p.__setitem__("production_image_id", "latest"), "image id"),
        (lambda p: p.__setitem__("network_mode", "bridge"), "offline"),
        (lambda p: p.__setitem__("release_authorized", True), "release or mutation"),
        (
            lambda p: p["entrypoint_argv"].extend(["--entrypoint", "bash"]),
            "controlled grammar",
        ),
        (
            lambda p: p.__setitem__("candidate_universe", p["candidate_universe"][:-1]),
            "zero control",
        ),
        (
            lambda p: p.__setitem__("calibration_profile", "K2"),
            "calibration profile",
        ),
    ],
)
def test_stage2_plan_rejects_authority_and_matrix_drift(mutator, match: str) -> None:
    plan = _plan()
    mutator(plan)
    plan = _reseal_plan(plan)
    with pytest.raises(ValueError, match=match):
        stage2.validate_plan(plan)


def test_stage2_plan_rejects_duplicate_last_value_wins_cli_argument() -> None:
    plan = _plan()
    plan["entrypoint_argv"].extend(["--model-type", "flux"])
    plan = _reseal_plan(plan)
    with pytest.raises(ValueError, match="controlled grammar"):
        stage2.validate_plan(plan)


def test_stage2_plan_checkpoint_selection_is_reduced_and_maps_real_cadence() -> None:
    plan = _plan()
    assert plan["checkpoint_selection"] == {
        "checkpoint_rule_sha256": _sha("checkpoint-rule"),
        "target_fraction": {"numerator": 7, "denominator": 8},
        "selected_step": 609,
        "denominator_steps": 691,
        "mapping_rule": "nearest_current_candidate_ties_choose_earlier_step",
    }

    full_final = deepcopy(plan)
    full_final["checkpoint_selection"]["selected_step"] = 691
    full_final = _reseal_plan(full_final)
    with pytest.raises(ValueError, match="does not map its declared target"):
        stage2.validate_plan(full_final)

    unreduced = deepcopy(plan)
    unreduced["checkpoint_selection"]["target_fraction"] = {
        "numerator": 14,
        "denominator": 16,
    }
    unreduced = _reseal_plan(unreduced)
    with pytest.raises(ValueError, match="must be reduced"):
        stage2.validate_plan(unreduced)

    zero_target = deepcopy(plan)
    zero_target["checkpoint_selection"]["target_fraction"] = {
        "numerator": 0,
        "denominator": 1,
    }
    zero_target = _reseal_plan(zero_target)
    with pytest.raises(ValueError, match=r"inside \(0,1\]"):
        stage2.validate_plan(zero_target)


def test_stage2_checkpoint_selection_exact_midpoint_chooses_earlier_step() -> None:
    rule = {"target_fraction": 0.1875, "actual_mappings": [{"test": True}]}
    selection = stage2._checkpoint_selection_for_rule(rule, planned_steps=8)
    assert selection["target_fraction"] == {"numerator": 3, "denominator": 16}
    assert selection["selected_step"] == 1
    assert selection["denominator_steps"] == 8


def test_k0_checkpoint_mapping_uses_the_real_release_cadence() -> None:
    rule = {"target_fraction": 0.75, "actual_mappings": [{"test": True}]}

    control = stage2._checkpoint_selection_for_rule(
        rule, planned_steps=100, profile_id="K0"
    )
    uniform = stage2._checkpoint_selection_for_rule(rule, planned_steps=100)

    assert control["selected_step"] == 75
    assert uniform["selected_step"] == 78


@pytest.mark.parametrize(
    "mutation",
    [
        lambda mounts: mounts.append(
            {
                "source": "/tmp/overlay",
                "destination": "/app/forge",
                "read_only": True,
                "purpose": "rogue_overlay",
            }
        ),
        lambda mounts: mounts[0].update(destination="/app"),
        lambda mounts: mounts[0].update(read_only=False),
        lambda mounts: mounts[0].update(source="/tmp/unsafe,ro"),
        lambda mounts: mounts[1].update(purpose="base_model"),
    ],
)
def test_stage2_plan_rejects_mount_overlay_and_schema_bypass(mutation) -> None:
    plan = _plan()
    mutation(plan["mounts"])
    plan = _reseal_plan(plan)
    with pytest.raises(ValueError, match="mount"):
        stage2.validate_plan(plan)


def test_stage2_throughput_profile_recomputes_exact_maximal_depth(
    tmp_path: Path,
) -> None:
    plan = _plan(root=tmp_path)
    fixture = {
        "training_rows": [{"row": index} for index in range(24)],
        "training_dataset_shape_sha256": _sha("training-shape"),
    }
    production_identity = {
        "base_model": {
            "training_identity_sha256": plan["base_model_identity_sha256"],
        },
        "runtime_contract": {
            "runtime_identity_sha256": _sha("runtime"),
            "venv_tree_manifest_sha256": _sha("runtime-tree"),
            "trainer_identity_sha256": _sha("trainer"),
            "measurement_tool_sha256": _sha("measurement-tool"),
            "jit_enabled": True,
        },
    }
    envelope = stage2.krea_budget.seal_execution_envelope(
        equivalence_class="A-rank32-adamw8bit-mse-guidance2",
        network_rank=32,
        network_alpha=32,
        optimizer="adamw8bit",
        optimizer_config_sha256=krea_provenance.canonical_sha256(
            {"weight_decay": 0.0001}
        ),
        loss="mse",
        differential_guidance_enabled=True,
        guidance_scale=2,
        training_pair_count=24,
        training_dataset_shape_sha256=fixture["training_dataset_shape_sha256"],
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        data_parallel_replicas=1,
        resolution_policy_sha256=krea_provenance.canonical_sha256([512, 768, 1024]),
        precision_policy_sha256=krea_provenance.canonical_sha256(
            {"train_dtype": "bf16", "save_dtype": "bf16"}
        ),
        cache_latents_to_disk=False,
        cache_text_embeddings=False,
        compile_enabled=False,
        jit_enabled=True,
        dataloader_workers=0,
        base_model_identity_sha256=plan["base_model_identity_sha256"],
        runtime_identity_sha256=production_identity["runtime_contract"][
            "runtime_identity_sha256"
        ],
        host_execution_identity_sha256=_sha("host"),
        execution_surface="immutable_production_docker_image",
        execution_scope="stage2_throughput_timing_only",
        venv_tree_manifest_sha256=production_identity["runtime_contract"][
            "venv_tree_manifest_sha256"
        ],
        reference_container_image_sha256=plan["production_image_id"].split(":")[1],
        gpu_identity_sha256=_sha("gpu"),
        trainer_identity_sha256=production_identity["runtime_contract"][
            "trainer_identity_sha256"
        ],
        measurement_tool_sha256=production_identity["runtime_contract"][
            "measurement_tool_sha256"
        ],
    )
    captures = []
    for index in range(3):
        start = 1_800_000_000_000_000_000 + index * 10_000_000_000
        captures.append(
            {
                "capture_id": f"capture-{index}",
                "argv": ["/usr/bin/probe", "--fixed"],
                "executable_path": "/usr/bin/probe",
                "executable_sha256": _sha("probe"),
                "returncode": 0,
                "started_unix_ns": start,
                "ended_unix_ns": start + 9_000_000_000,
                "event_stream_sha256": _sha(f"events-{index}"),
            }
        )

    def sample(capture: str, observation: str, start: int, duration: int, units=1):
        return {
            "capture_id": capture,
            "observation_id": observation,
            "duration_s": duration / 1_000_000_000,
            "units": units,
            "started_monotonic_ns": start,
            "ended_monotonic_ns": start + duration,
        }

    samples = {
        "startup": [
            sample(
                f"capture-{index}",
                f"startup-{index}",
                1000 + index * 100,
                10_000_000_000,
            )
            for index in range(3)
        ],
        "optimizer_update": [
            sample("capture-0", "updates", 2000, 200_000_000_000, units=100)
        ],
        "checkpoint_save": [sample("capture-0", "saves", 3000, 8_000_000_000, units=8)],
        "finalization": [sample("capture-0", "final", 4000, 30_000_000_000)],
        "upload": [sample("capture-0", "upload", 5000, 20_000_000_000)],
    }
    raw = stage2.krea_budget.seal_timing_sample_manifest(
        execution_envelope=envelope,
        probe_contract_sha256=_sha("probe-contract"),
        measurement_tool_sha256=production_identity["runtime_contract"][
            "measurement_tool_sha256"
        ],
        command_captures=captures,
        samples=samples,
        seed_bindings=[{"role": "A", "seed": 42565431}],
    )
    margin = stage2.krea_budget.seal_margin_policy(
        reviewer_identity="Jordan Example",
        approved_at_utc="2026-07-28T01:00:00Z",
        frozen_before_capture=True,
        multiplicative_margin={name: 1.25 for name in samples},
        additive_margin_s={name: 0.5 for name in samples},
    )
    e2e = stage2.krea_budget.seal_end_to_end_validation(
        execution_envelope_sha256=envelope["execution_envelope_sha256"],
        probe_contract_sha256=_sha("probe-contract"),
        runs=[
            {
                "run_id": "heldout-1",
                "seed_role": "A",
                "seed": 42565431,
                "hard_budget_s": 2700.0,
                "outer_wall_clock_s": 2000.0,
                "natural_completion": True,
                "upload_ready": True,
                "failure_or_fallback_telemetry": False,
                "run_record_sha256": _sha("heldout"),
            }
        ],
    )
    record = stage2.krea_budget.seal_throughput_profile_from_evidence(
        raw_sample_manifest=raw,
        margin_policy=margin,
        end_to_end_validation=e2e,
        framework_stop_boundary_s=225,
        framework_stop_boundary_source_sha256=_sha("stop-boundary"),
    )
    evidence_records = {
        "raw_samples": (raw, "raw_sample_manifest_sha256", "throughput-raw.json"),
        "margin_policy": (margin, "margin_policy_sha256", "throughput-margin.json"),
        "end_to_end_validation": (
            e2e,
            "end_to_end_validation_sha256",
            "throughput-e2e.json",
        ),
    }
    for field, (document, semantic_key, filename) in evidence_records.items():
        path = tmp_path / filename
        path.write_bytes(krea_provenance.canonical_bytes(document) + b"\n")
        plan["throughput_evidence"][field] = {
            "path": str(path),
            "file_sha256": krea_provenance.file_sha256(path),
            semantic_key: document[semantic_key],
        }
    profile_path = tmp_path / "throughput-profile.json"
    profile_path.write_bytes(krea_provenance.canonical_bytes(record) + b"\n")
    profile = stage2.krea_budget.load_throughput_profile(record)
    planned = stage2.krea_budget.plan_budget(
        profile, hard_budget_s=0.75 * 3600
    ).max_affordable_steps
    plan["planned_steps"] = planned
    plan["checkpoint_selection"] = _checkpoint_selection(planned_steps=planned)
    plan["throughput_profile"] = {
        "path": str(profile_path),
        "file_sha256": krea_provenance.file_sha256(profile_path),
        "profile_sha256": record["profile_sha256"],
    }
    plan["execution_environment_profile"] = dict(plan["throughput_profile"])
    stage2._validate_throughput_depth(
        plan, fixture=fixture, production_identity=production_identity
    )

    plan["planned_steps"] = 5000
    with pytest.raises(ValueError, match="measured maximal budget"):
        stage2._validate_throughput_depth(
            plan, fixture=fixture, production_identity=production_identity
        )


def test_stage2_family_is_bound_to_owner_ratified_finalist_freeze() -> None:
    plan = _plan()
    selected = next(
        row
        for row in plan["candidate_universe"]
        if row["candidate_id"] == plan["training_candidate_id"]
    )
    rules = {}
    for family in ("K0", "K1", "K2", "K3", "K4", "K5"):
        if family == "K1":
            candidate_id = selected["candidate_id"]
            candidate_sha = selected["sha256"]
            step = selected["step"]
        else:
            candidate_id = f"{family}-frozen"
            candidate_sha = _sha(f"{family}-frozen")
            step = 1
        rules[family] = {
            "target_fraction": 0.875,
            "actual_mappings": [
                {
                    "candidate_id": candidate_id,
                    "candidate_sha256": candidate_sha,
                    "step": step,
                }
            ],
        }
    body = {
        "schema": stage2.krea_waiver_finalist_freeze.SCHEMA,
        "kind": stage2.krea_waiver_finalist_freeze.FREEZE_KIND,
        "outcome": "finalists_frozen",
        "blockers": [],
        "claims": stage2.krea_waiver_finalist_freeze.FALSE_CLAIMS,
        "authority": stage2.krea_waiver_finalist_freeze.AUTHORITY,
        "finalist_family_ids": ["K1", "K0"],
        "checkpoint_rules": {family: rules[family] for family in ("K1", "K0")},
        "all_family_checkpoint_rules": rules,
    }
    freeze = {
        **body,
        "freeze_sha256": krea_provenance.canonical_sha256(body),
    }
    request = {
        "waiver_freeze_sha256": freeze["freeze_sha256"],
        "waiver_freeze_file_sha256": __import__("hashlib")
        .sha256(krea_provenance.canonical_bytes(freeze) + b"\n")
        .hexdigest(),
    }
    plan["checkpoint_selection"] = stage2._checkpoint_selection_for_rule(
        rules["K1"], planned_steps=plan["planned_steps"]
    )
    assert (
        stage2._validate_frozen_execution_family(plan, freeze=freeze, request=request)
        == freeze
    )

    density_body = {
        **body,
        "schema": stage2.krea_density_seedb_freeze.SCHEMA,
        "kind": stage2.krea_density_seedb_freeze.FREEZE_KIND,
        "claims": stage2.krea_density_seedb_freeze.FALSE_CLAIMS,
        "authority": stage2.krea_density_seedb_freeze.AUTHORITY,
    }
    density_freeze = {
        **density_body,
        "freeze_sha256": krea_provenance.canonical_sha256(density_body),
    }
    density_request = {
        "waiver_freeze_sha256": density_freeze["freeze_sha256"],
        "waiver_freeze_file_sha256": __import__("hashlib")
        .sha256(krea_provenance.canonical_bytes(density_freeze) + b"\n")
        .hexdigest(),
    }
    assert (
        stage2._validate_frozen_execution_family(
            plan, freeze=density_freeze, request=density_request
        )
        == density_freeze
    )

    drifted_rules = deepcopy(rules)
    drifted_rules["K1"]["target_fraction"] = 0.5
    drifted_body = {
        **body,
        "checkpoint_rules": {family: drifted_rules[family] for family in ("K1", "K0")},
        "all_family_checkpoint_rules": drifted_rules,
    }
    drifted_freeze = {
        **drifted_body,
        "freeze_sha256": krea_provenance.canonical_sha256(drifted_body),
    }
    drifted_request = {
        "waiver_freeze_sha256": drifted_freeze["freeze_sha256"],
        "waiver_freeze_file_sha256": __import__("hashlib")
        .sha256(krea_provenance.canonical_bytes(drifted_freeze) + b"\n")
        .hexdigest(),
    }
    with pytest.raises(ValueError, match="differs from the frozen report rule"):
        stage2._validate_frozen_execution_family(
            plan, freeze=drifted_freeze, request=drifted_request
        )

    relabeled = deepcopy(plan)
    selected = next(
        row
        for row in relabeled["candidate_universe"]
        if row["candidate_id"] == relabeled["training_candidate_id"]
    )
    selected["family_id"] = "K4"
    relabeled["calibration_profile"] = "K4"
    with pytest.raises(ValueError, match="frozen non-control finalist"):
        stage2._validate_frozen_execution_family(
            relabeled, freeze=freeze, request=request
        )


def test_stage2_plan_recomputes_complete_owner_authority_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, controls, authorization = _fake_authority_bundle(_plan())
    observed: list[dict] = []

    def fake_validate(value, **kwargs):
        observed.append({"value": value, **kwargs})
        return value

    monkeypatch.setattr(
        stage2.krea_confirmation_admission,
        "validate_gpu_execution_authorization",
        fake_validate,
    )
    monkeypatch.setattr(
        stage2.krea_stage2_admission_chain,
        "validate_inventory",
        lambda value: value,
    )
    monkeypatch.setattr(
        stage2.krea_stage2_admission_chain,
        "inventory_sealed_files",
        lambda _value: controls["materialization"].get("files"),
    )
    monkeypatch.setattr(
        stage2,
        "_validate_frozen_execution_family",
        lambda plan, **_kwargs: plan,
    )
    monkeypatch.setattr(
        stage2,
        "_validate_fixture_and_archive",
        lambda plan, **_kwargs: {
            "training_rows": [],
            "training_dataset_shape_sha256": _sha("shape"),
        },
    )
    monkeypatch.setattr(
        stage2, "_validate_throughput_depth", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        stage2,
        "_validate_execution_environment_profile",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        stage2, "_validate_live_base_assets", lambda *_args, **_kwargs: {}
    )
    assert (
        stage2.validate_plan_with_authority(plan, authority_controls=controls) == plan
    )
    assert observed[0]["value"] == authorization
    assert observed[0]["materialization"] == controls["materialization"]
    assert observed[0]["production_identity"] == controls["production_identity"]

    drifted = deepcopy(plan)
    drifted["production_image_id"] = f"sha256:{'0' * 64}"
    drifted = _reseal_plan(drifted)
    with pytest.raises(ValueError, match="policy, contract, or image"):
        stage2.validate_plan_with_authority(drifted, authority_controls=controls)


def test_stage2_authority_bundle_rehashes_every_record_before_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, controls, _ = _fake_authority_bundle(_plan())
    controls = deepcopy(controls)
    controls["ratification"]["drift"] = True
    monkeypatch.setattr(
        stage2.krea_confirmation_admission,
        "validate_gpu_execution_authorization",
        lambda *_args, **_kwargs: pytest.fail("drifted record reached admission"),
    )
    with pytest.raises(ValueError, match="ratification file SHA-256"):
        stage2.validate_plan_with_authority(plan, authority_controls=controls)


def test_stage2_boundary_uses_explicit_frozen_selected_profile() -> None:
    plan = _plan(boundary=True)
    assert plan["calibration_profile"] == "K1"
    assert plan["planned_steps"] == 691
    assert plan["throughput_profile"] is not None
    drifted = deepcopy(plan)
    drifted["calibration_profile"] = "K0"
    drifted = _reseal_plan(drifted)
    with pytest.raises(ValueError, match="calibration profile"):
        stage2.validate_plan(drifted)


def test_private_receipt_replay_uses_plan_derived_evidence_root(tmp_path: Path) -> None:
    roots = tmp_path / "roots"
    for name in ("models", "text-encoder", "datasets", "checkpoints", "evidence"):
        (roots / name).mkdir(parents=True)
    plan = _plan(root=roots)
    approval = _approval(plan)
    _materialized_completion(
        plan,
        approval,
        run_root=tmp_path / "run",
        roots=roots,
    )

    replay = stage2.validate_private_run_receipts(plan)
    expected_root = roots / "evidence" / plan["plan_sha256"]
    assert replay["evidence_root"] == str(expected_root)
    assert replay["effective_config_path"] == str(
        expected_root / "effective-config.yaml"
    )
    assert replay["config_control"]["config_sha256"] == krea_provenance.file_sha256(
        expected_root / "effective-config.yaml"
    )
    assert replay["training_terminal"]["receipt_sha256"]
    assert replay["checkpoint_selection"] == {
        "file_sha256": krea_provenance.file_sha256(
            expected_root / "forge_checkpoint_selection.json"
        ),
        "receipt_sha256": krea_provenance.canonical_sha256(
            json.loads((expected_root / "forge_checkpoint_selection.json").read_bytes())
        ),
    }

    (expected_root / "effective-config.yaml").write_text("drifted: true\n")
    with pytest.raises(ValueError, match="bytes drifted"):
        stage2.validate_private_run_receipts(plan)


def test_private_receipt_replay_rejects_missing_or_tampered_selection(
    tmp_path: Path,
) -> None:
    roots = tmp_path / "roots"
    for name in ("models", "text-encoder", "datasets", "checkpoints", "evidence"):
        (roots / name).mkdir(parents=True)
    plan = _plan(root=roots)
    approval = _approval(plan)
    _materialized_completion(
        plan,
        approval,
        run_root=tmp_path / "run",
        roots=roots,
    )
    selection_path = (
        roots / "evidence" / plan["plan_sha256"] / "forge_checkpoint_selection.json"
    )
    original = selection_path.read_bytes()
    selection_path.unlink()
    with pytest.raises(ValueError, match="not a readable JSON file"):
        stage2.validate_private_run_receipts(plan)

    record = json.loads(original)
    record["source"] = "exact_final"
    selection_path.write_bytes(krea_provenance.canonical_bytes(record))
    with pytest.raises(ValueError, match="differs from its frozen target"):
        stage2.validate_private_run_receipts(plan)


def test_private_receipt_replay_rejects_full_final_substitution(
    tmp_path: Path,
) -> None:
    roots = tmp_path / "roots"
    for name in ("models", "text-encoder", "datasets", "checkpoints", "evidence"):
        (roots / name).mkdir(parents=True)
    plan = _plan(root=roots)
    approval = _approval(plan)
    _materialized_completion(
        plan,
        approval,
        run_root=tmp_path / "run",
        roots=roots,
    )
    checkpoint_root = (
        roots / "checkpoints" / plan["task_id"] / plan["expected_repo_name"]
    )
    exact_final = checkpoint_root / f"{plan['expected_repo_name']}.safetensors"
    _write_safetensors(exact_final, first=b"\x09\x08\x07\x06")
    (checkpoint_root / "last.safetensors").write_bytes(exact_final.read_bytes())
    with pytest.raises(ValueError, match="bytes differ from last.safetensors"):
        stage2.validate_private_run_receipts(plan)


@pytest.mark.parametrize(
    ("receipt_name", "match"),
    [
        ("config-control.json", "config-control receipt differs"),
        ("training-terminal.json", "terminal receipt does not prove"),
    ],
)
def test_private_receipts_reject_runtime_checkpoint_binding_drift(
    tmp_path: Path, receipt_name: str, match: str
) -> None:
    roots = tmp_path / "roots"
    for name in ("models", "text-encoder", "datasets", "checkpoints", "evidence"):
        (roots / name).mkdir(parents=True)
    plan = _plan(root=roots)
    approval = _approval(plan)
    _materialized_completion(
        plan,
        approval,
        run_root=tmp_path / "run",
        roots=roots,
    )
    receipt_path = roots / "evidence" / plan["plan_sha256"] / receipt_name
    receipt = json.loads(receipt_path.read_bytes())
    receipt["checkpoint_selection"]["selected_step"] = plan["planned_steps"]
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = krea_provenance.canonical_sha256(body)
    receipt_path.write_bytes(krea_provenance.canonical_bytes(receipt) + b"\n")
    with pytest.raises(ValueError, match=match):
        stage2.validate_private_run_receipts(plan)


def test_stage2_approval_must_be_fresh_agent_and_postdate_plan() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="postdate"):
        stage2.build_approval(
            plan,
            reviewer_actor=_actor(),
            approved_at_utc=plan["created_at_utc"],
        )
    wrong = _actor("fixture_reviewer")
    with pytest.raises(ValueError, match="wrong delegated role"):
        stage2.build_approval(
            plan,
            reviewer_actor=wrong,
            approved_at_utc=_time(1),
        )
    approval = _approval(plan)
    approval["release_authorized"] = True
    approval = _reseal_approval(approval)
    with pytest.raises(ValueError, match="authority flags"):
        stage2.validate_approval(approval, plan=plan)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("returncode", 1, "naturally"),
        ("fallback_used", True, "naturally"),
        ("strict_discovery_replayed", True, "overclaims"),
        ("production_image_id", f"sha256:{'0' * 64}", "authority binding"),
    ],
)
def test_stage2_completion_rejects_failure_fallback_and_overclaim(
    field: str, value, match: str
) -> None:
    plan = _plan()
    approval = _approval(plan)
    completion = _completion(plan, approval)
    completion[field] = value
    completion = _reseal_completion(completion)
    with pytest.raises(ValueError, match=match):
        stage2.validate_completion(completion, plan=plan, approval=approval)


def test_stage2_completion_requires_selection_receipt_and_manifest_binding() -> None:
    plan = _plan()
    approval = _approval(plan)
    completion = _completion(plan, approval)
    del completion["checkpoint_selection_receipt"]
    completion = _reseal_completion(completion)
    with pytest.raises(ValueError, match="checkpoint_selection_receipt"):
        stage2.validate_completion(completion, plan=plan, approval=approval)

    completion = _completion(plan, approval)
    completion["artifact_manifest"] = [
        row
        for row in completion["artifact_manifest"]
        if row["path"] != "evidence/forge_checkpoint_selection.json"
    ]
    completion = _reseal_completion(completion)
    with pytest.raises(ValueError, match="forge_checkpoint_selection"):
        stage2.validate_completion(completion, plan=plan, approval=approval)


def test_stage2_completion_rejects_full_final_promotion_substitution() -> None:
    plan = _plan()
    approval = _approval(plan)
    completion = _completion(plan, approval)
    last = next(
        row
        for row in completion["artifact_manifest"]
        if row["path"] == "checkpoints/last.safetensors"
    )
    last["sha256"] = _sha("full-final-substitution")
    completion = _reseal_completion(completion)
    with pytest.raises(ValueError, match="selected checkpoint promotion"):
        stage2.validate_completion(completion, plan=plan, approval=approval)


def test_run_cell_uses_exact_image_and_never_overrides_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = tmp_path / "roots"
    for name in ("models", "text-encoder", "datasets", "checkpoints", "evidence"):
        (roots / name).mkdir(parents=True)
    plan = _plan(root=roots)
    approval = _approval(plan)
    observed: list[str] = []

    class Result:
        returncode = 0

    def fake_run(command, *, stdout, stderr, check, timeout):
        observed.extend(command)
        stdout.write(b"ok\n")
        checkpoint = (
            roots
            / "checkpoints"
            / plan["task_id"]
            / plan["expected_repo_name"]
            / "last.safetensors"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        _write_safetensors(checkpoint)
        (checkpoint.parent / "forge_run.json").write_text(
            json.dumps(
                {
                    "schema": 2,
                    "kind": "forge-public-run-recorder",
                    "private_record_sha256": _sha("private-run"),
                    "events": [
                        {"t": 100.0, "name": "checkpoint_finalized"},
                        {"t": 101.0, "name": "run_complete"},
                        {"t": 102.0, "name": "public_bundle_ready"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        _write_stage2_receipts(
            plan,
            roots / "evidence" / plan["plan_sha256"],
        )
        return Result()

    monkeypatch.setattr(stage2.subprocess, "run", fake_run)
    authority_checked: list[dict] = []

    def fake_authority(value, *, authority_controls):
        authority_checked.append(authority_controls)
        return stage2.validate_plan(value)

    monkeypatch.setattr(stage2, "validate_plan_with_authority", fake_authority)
    monkeypatch.setattr(
        stage2, "_validate_live_throughput_environment", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        stage2, "_validate_live_base_assets", lambda *_args, **_kwargs: {}
    )
    authority_chain = {
        "test": "owner-authority-chain",
        "production_identity": {},
    }
    completion = stage2.run_cell(
        plan=plan,
        approval=approval,
        authority_controls=authority_chain,
        output_dir=tmp_path / "run",
        completion_path=tmp_path / "completion.json",
        gpu_device=3,
    )
    assert observed[0:2] == ["docker", "run"]
    assert "--entrypoint" not in observed
    assert plan["production_image_id"] in observed
    assert observed.index(plan["production_image_id"]) < observed.index("--task-id")
    assert "FORGE_KREA_CALIBRATION_PROFILE=K1" in observed
    assert "FORGE_KREA_CALIBRATION_STEPS=691" in observed
    assert "FORGE_KREA_STAGE2_TARGET_FRACTION_NUMERATOR=7" in observed
    assert "FORGE_KREA_STAGE2_TARGET_FRACTION_DENOMINATOR=8" in observed
    assert (
        "FORGE_KREA_CALIBRATION_THROUGHPUT_PROFILE_SHA256="
        + plan["throughput_profile"]["profile_sha256"]
        in observed
    )
    assert authority_checked == [authority_chain]
    assert completion["returncode"] == 0
    with pytest.raises(FileExistsError):
        stage2.run_cell(
            plan=plan,
            approval=approval,
            authority_controls=authority_chain,
            output_dir=tmp_path / "run",
            completion_path=tmp_path / "other.json",
            gpu_device=3,
        )


def test_run_cell_rejects_preexisting_checkpoint_root_before_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = tmp_path / "roots"
    for name in ("models", "text-encoder", "datasets", "checkpoints", "evidence"):
        (roots / name).mkdir(parents=True)
    plan = _plan(root=roots)
    approval = _approval(plan)
    stale = roots / "checkpoints" / plan["task_id"] / plan["expected_repo_name"]
    stale.mkdir(parents=True)
    _write_safetensors(stale / "last.safetensors")
    monkeypatch.setattr(
        stage2,
        "validate_plan_with_authority",
        lambda value, **_kwargs: stage2.validate_plan(value),
    )
    monkeypatch.setattr(
        stage2.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("Docker reached with stale checkpoints"),
    )
    monkeypatch.setattr(
        stage2, "_validate_live_throughput_environment", lambda *_args, **_kwargs: None
    )

    with pytest.raises(FileExistsError, match="checkpoint/evidence roots"):
        stage2.run_cell(
            plan=plan,
            approval=approval,
            authority_controls={},
            output_dir=tmp_path / "run",
            completion_path=tmp_path / "completion.json",
            gpu_device=0,
        )


def test_stage2_run_evidence_rehashes_and_binds_every_authority(
    tmp_path: Path,
) -> None:
    roots = tmp_path / "roots"
    for name in ("models", "text-encoder", "datasets", "checkpoints", "evidence"):
        (roots / name).mkdir(parents=True)
    plan = _plan(root=roots)
    approval = _approval(plan)
    run_root = tmp_path / "run"
    completion = _materialized_completion(
        plan, approval, run_root=run_root, roots=roots
    )
    evidence = training_evidence.build_run_evidence(
        plan=plan,
        plan_file_sha256=_sha("plan-file"),
        approval=approval,
        approval_file_sha256=_sha("approval-file"),
        completion=completion,
        completion_file_sha256=_sha("completion-file"),
        run_output_root=run_root,
        fixture_manifest=_binding("fixture", "manifest_sha256"),
        emitted_at_utc=_time(4),
    )
    assert evidence["candidate_artifacts"] == [
        row
        for row in completion["artifact_manifest"]
        if row["path"].startswith("checkpoints/")
        and row["path"].endswith(".safetensors")
    ]
    assert (
        training_evidence.validate_run_evidence(
            evidence, plan=plan, approval=approval, completion=completion
        )
        == evidence
    )

    tampered = deepcopy(evidence)
    tampered["execution_plan"]["unbound"] = _sha("unbound")
    body = {key: value for key, value in tampered.items() if key != "evidence_sha256"}
    tampered["evidence_sha256"] = krea_provenance.canonical_sha256(body)
    with pytest.raises(ValueError, match="execution plan keys differ"):
        training_evidence.validate_run_evidence(
            tampered, plan=plan, approval=approval, completion=completion
        )


def test_stage2_run_evidence_rejects_artifact_drift(tmp_path: Path) -> None:
    roots = tmp_path / "roots"
    for name in ("models", "text-encoder", "datasets", "checkpoints", "evidence"):
        (roots / name).mkdir(parents=True)
    plan = _plan(root=roots)
    approval = _approval(plan)
    run_root = tmp_path / "run"
    completion = _materialized_completion(
        plan, approval, run_root=run_root, roots=roots
    )
    (run_root / "container.stdout").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="bytes drifted"):
        training_evidence.build_run_evidence(
            plan=plan,
            plan_file_sha256=_sha("plan-file"),
            approval=approval,
            approval_file_sha256=_sha("approval-file"),
            completion=completion,
            completion_file_sha256=_sha("completion-file"),
            run_output_root=run_root,
            fixture_manifest=_binding("fixture", "manifest_sha256"),
            emitted_at_utc=_time(4),
        )


def test_stage2_zero_control_is_deterministic_create_only_and_all_zero(
    tmp_path: Path,
) -> None:
    roots = tmp_path / "roots"
    for name in ("models", "text-encoder", "datasets", "checkpoints", "evidence"):
        (roots / name).mkdir(parents=True)
    plan = _plan(root=roots)
    approval = _approval(plan)
    run_root = tmp_path / "run"
    completion = _materialized_completion(
        plan, approval, run_root=run_root, roots=roots
    )
    evidence = training_evidence.build_run_evidence(
        plan=plan,
        plan_file_sha256=_sha("plan-file"),
        approval=approval,
        approval_file_sha256=_sha("approval-file"),
        completion=completion,
        completion_file_sha256=_sha("completion-file"),
        run_output_root=run_root,
        fixture_manifest=_binding("fixture", "manifest_sha256"),
        emitted_at_utc=_time(4),
    )
    template = (
        roots
        / "checkpoints"
        / plan["task_id"]
        / plan["expected_repo_name"]
        / "final.safetensors"
    )
    artifact = tmp_path / "zero" / "zero.safetensors"
    manifest_path = tmp_path / "zero" / "zero.json"
    manifest = training_evidence.emit_zero_control(
        template_artifact=template,
        template_run_evidence=evidence,
        output_artifact=artifact,
        output_manifest=manifest_path,
    )
    assert (
        training_evidence.validate_zero_control(manifest, artifact_path=artifact)
        == manifest
    )
    _, data = training_evidence.krea_training_evidence._read_safetensors(artifact)
    assert data and not any(data)
    with pytest.raises(FileExistsError):
        training_evidence.emit_zero_control(
            template_artifact=template,
            template_run_evidence=evidence,
            output_artifact=artifact,
            output_manifest=manifest_path,
        )


def test_stage2_zero_control_detects_postpublish_tamper(tmp_path: Path) -> None:
    template = tmp_path / "template.safetensors"
    _write_safetensors(template)
    template_sha = krea_provenance.file_sha256(template)
    evidence_body = {
        "kind": training_evidence.RUN_KIND,
        "natural_completion": True,
        "candidate_artifacts": [{"sha256": template_sha}],
    }
    evidence = {
        **evidence_body,
        "evidence_sha256": krea_provenance.canonical_sha256(evidence_body),
    }
    artifact = tmp_path / "zero.safetensors"
    manifest_path = tmp_path / "zero.json"
    manifest = training_evidence.emit_zero_control(
        template_artifact=template,
        template_run_evidence=evidence,
        output_artifact=artifact,
        output_manifest=manifest_path,
    )
    raw = bytearray(artifact.read_bytes())
    raw[-1] = 1
    artifact.write_bytes(raw)
    with pytest.raises(ValueError, match="binding differs"):
        training_evidence.validate_zero_control(manifest, artifact_path=artifact)
