"""Canonical claim boundary for the Week-5 Krea Stage-1 campaign."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


_BODY = {
    "schema": 1,
    "kind": "forge-krea-execution-surface-policy",
    "execution_surface": "staged_host_venv",
    "execution_scope": "discovery_only",
    "claims_forbidden": [
        "absolute_tournament_budget_fill",
        "production_container_throughput_equivalence",
        "release_or_deployment_authorization",
        "tournament_field_parity",
    ],
    "stage1_timing_margin_policy": {
        "multiplicative_margin": {
            "startup": 1.25,
            "optimizer_update": 1.25,
            "checkpoint_save": 1.25,
            "finalization": 1.25,
            "upload": 1.25,
        },
        "additive_margin_s": {
            "startup": 5.0,
            "optimizer_update": 0.05,
            "checkpoint_save": 2.0,
            "finalization": 10.0,
            "upload": 10.0,
        },
    },
    "stage1_host_preflight_policy": {
        "maximum_load_per_effective_cpu": 0.5,
        "minimum_available_memory_bytes": 68719476736,
        "minimum_checkpoint_free_bytes": 375809638400,
        "maximum_gpu_utilization_percent": 5,
        "minimum_free_gpu_memory_mib": 78000,
        "maximum_foreign_compute_processes": 0,
        "storage_probe_bytes": 16777216,
        "minimum_checkpoint_write_mib_s": 100,
        "minimum_checkpoint_read_mib_s": 100,
        "maximum_checkpoint_fsync_s": 5,
    },
    "stage1_exact_scorer_contract": {
        "god_commit": "b026da04b6179cf82945e8736590dd923114342b",
        "comfy_commit": "091b70edda0c062fc9338a1d7e8e2f94f4c0ad0b",
        "tooling_commit": "5d3194f4d4158ab31df7a060e1e4c56fa03f320c",
        "evaluator_script_sha256": (
            "6ba85ccd8fc51a5e4a946da1be5c4f9d241b1868d14ab19430d353e89508122f"
        ),
        "dataset_identity_module_sha256": (
            "632f6ca7d58a0bdb38519bef510d621bef09405bd87f5479fe3bad68d69e955f"
        ),
        "eval_defaults": {
            "steps": 25,
            "cfg": 12,
            "denoise": 0.8,
            "generations": 5,
            "master_seed": 42,
            "text_weight": 0.25,
        },
        "assets": {
            "diffusion_model": {
                "basename": "krea2_raw_fp8_scaled.safetensors",
                "relative_path": "models/diffusion_models/krea2_raw_fp8_scaled.safetensors",
                "sha256": (
                    "48cd5d6c100297968349b41a8e77c6591d1dac18a215807f5f25f59e5c54cd61"
                ),
                "bytes": 13141730784,
            },
            "text_encoder": {
                "basename": "qwen3vl_4b_fp8_scaled.safetensors",
                "relative_path": "models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
                "sha256": (
                    "54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094"
                ),
                "bytes": 5242467968,
            },
            "vae": {
                "basename": "qwen_image_vae.safetensors",
                "relative_path": "models/vae/qwen_image_vae.safetensors",
                "sha256": (
                    "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"
                ),
                "bytes": 253806246,
            },
        },
        "source_trees": {
            "god": "60d5e579aed31b69bf07d0513aace1518c974c30",
            "comfyui": "1936f65713a6a6d88066b0d6127931ec50c1a2c1",
            "tooling_nodes": "c7f2378076420703e933bb7619f5f1d67eb1dbeb",
        },
        "runtime_materialization": {
            "exact_lock": {
                "relative_path": "week5/krea-stage1-exact-scorer-lock.txt",
                "sha256": (
                    "5473a9da95cc729cac65ae0309b1044224a40eb1e8961b77cd0e39eab846bb08"
                ),
                "line_count": 229,
                "resolved_distribution_count": 229,
                "normalized_name_version_sha256": (
                    "ef382bd0c993113f5ae058ff91ac5aa1b42c8de5986d4b7471fd77919c9aae22"
                ),
                "vcs_distribution_versions": {"fiber": "2.6.0"},
            },
            "python_version": "3.10.20",
            "distribution_count": 229,
            "distributions_sha256": (
                "bbcd979cae4ca3cc3e8a35c16c3d1908512bec1b8b7e9a540582122e97648bed"
            ),
            "same_comfy_and_driver_python": True,
            "requirements_sha256": {
                "comfyui": (
                    "5cb303a106455a29613d661fd1d67b0b58556fba71e7b88b6b487da868458eba"
                ),
                "tooling_nodes": (
                    "17938e9b82a1ec0f0985730088f57b73b206e5fcad390002640da246b343609e"
                ),
                "god_validator": (
                    "7cc1ca40ae96917b4c2d35aeb0e0734cc5462dce0aa39da30ee5b6832ddbfb56"
                ),
            },
            "critical_distributions": {
                "torch": "2.9.1+cu128",
                "torchvision": "0.24.1+cu128",
                "torchaudio": "2.9.1+cu128",
            },
            "cuda_runtime_probe": {
                "torch": "2.9.1+cu128",
                "torch_cuda": "12.8",
                "cudnn_version": 91002,
                "bf16_conv3d": True,
                "overlapping_namespace_owners": {
                    "cudnn": "nvidia-cudnn-cu12==9.10.2.21",
                    "cusparselt": "nvidia-cusparselt-cu12==0.7.1",
                    "nccl": "nvidia-nccl-cu12==2.27.5",
                    "nvshmem": "nvidia-nvshmem-cu12==3.3.20",
                },
            },
            "install_order": [
                "comfyui_requirements",
                "force_torch_trio_cu128",
                "tooling_requirements",
                "diffusers_and_huggingface_hub",
                "god_validator_requirements",
                "force_all_overlapping_cuda12_namespace_wheels_last",
            ],
        },
        "timeouts_s": {
            "startup": 300.0,
            # This is the owner-ratified training/scoring base contract.  A
            # scorer-only extension may raise the effective evaluation limit,
            # but must not mutate this policy (and thereby orphan valid
            # training evidence produced under it).
            "evaluation": 3600.0,
            "shutdown": 20.0,
            "containment_term_grace": 20.0,
        },
        "estimated_seconds_per_candidate": 720,
        "execution_scope": "offline_stage1_discovery_only",
    },
    "authorized_technical_agent_roles": [
        "discovery_execution_authorization_reviewer",
        "timing_probe_execution_reviewer",
        "execution_plan_reviewer",
    ],
    "agent_review_is_not_human_review": True,
    "stage2_requirement": (
        "immutable production-Docker evidence requires a separate Forge commit "
        "and fresh named-owner ratification"
    ),
}
POLICY = {**_BODY, "policy_sha256": _sha256(_BODY)}


def validate(value: Any) -> dict[str, Any]:
    if value != POLICY:
        raise ValueError("Stage-1 execution-surface policy drifted")
    return dict(POLICY)


def technical_role(role: Any) -> str:
    if role not in POLICY["authorized_technical_agent_roles"]:
        raise ValueError("technical agent role is not owner-ratifiable for Stage-1")
    return str(role)
