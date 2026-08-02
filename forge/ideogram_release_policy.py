"""Hash-bound Week-5 Ideogram recipe and exact-final checkpoint policy.

The deployed Week-4 recipe remains the source/default.  A literal, reviewed
activation record is required before this module changes any config.  The
activation binds the exact recipe validated on ``I-J20-D2`` and forces the
unnumbered terminal export to remain the production artifact; generic holdout
or training-loss selectors cannot silently replace the scored position.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Mapping

from forge import telemetry


POLICY_ID = "week5-ideogram-effective-unet-port-exact-final-v2"
POLICY_KIND = "forge-ideogram-week5-production-policy"
ACTIVATION_KIND = "forge-ideogram-week5-production-activation"
CHECKPOINT_MAPPING_RULE = "nearest_current_candidate_ties_choose_earlier_step"
SOURCE_VALIDATION_CELL = "I-J20-D2"
SOURCE_CONFIG_FILE_SHA256 = (
    "95578f5e3bdcbda1def0bd66506fad102abf3a02d9f74d4a71b464114ddea190"
)
SOURCE_CONFIG_SEMANTIC_SHA256 = (
    "ea29386cf5c71e7a74dc27d841c9a28f5ce82a69def7e217d2140506a86bb951"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


_EXPECTED_RECIPE = {
    "model_arch": "ideogram4",
    "training_seed": 20260802,
    "network": {"type": "lora", "linear": 32, "linear_alpha": 32},
    "save": {
        "dtype": "bf16",
        "max_step_saves_to_keep": 100,
        "save_format": "diffusers",
        "push_to_hub": False,
    },
    "dataset": {
        "caption_ext": "txt",
        "caption_dropout_rate": 0.05,
        "cache_latents_to_disk": True,
        "is_reg": False,
        "resolution": [512, 768, 1024],
    },
    "train": {
        "batch_size": 1,
        "gradient_accumulation": 1,
        "train_unet": True,
        "train_text_encoder": True,
        "gradient_checkpointing": True,
        "noise_scheduler": "flowmatch",
        "optimizer": "adamw8bit",
        "timestep_type": "linear",
        "optimizer_params": {"weight_decay": 0.0001},
        "cache_text_embeddings": False,
        "lr": 0.000025,
        "unet_lr": 0.000025,
        "text_encoder_lr": 0.0000001,
        "lr_scheduler": "cosine",
        "lr_scheduler_params": {"eta_min": 0.0000025},
        "ema_config": {"use_ema": True, "ema_decay": 0.995},
        "do_cfg": True,
        "cfg_scale": 10.0,
        "disable_sampling": True,
        "dtype": "bf16",
    },
}

_SOURCE_RECIPE = {
    **_EXPECTED_RECIPE,
    "training_seed": None,
    "dataset": {
        **_EXPECTED_RECIPE["dataset"],
        "cache_latents_to_disk": False,
    },
    "train": {
        **_EXPECTED_RECIPE["train"],
        "train_text_encoder": False,
        "lr": 0.0001,
        "unet_lr": None,
        "text_encoder_lr": None,
        "lr_scheduler": None,
        "lr_scheduler_params": None,
        "ema_config": {"use_ema": False, "ema_decay": 0.99},
        "do_cfg": None,
        "cfg_scale": None,
    },
}

_POLICY_BODY = {
    "schema": 1,
    "kind": POLICY_KIND,
    "policy_id": POLICY_ID,
    "source_recipe_projection": _SOURCE_RECIPE,
    "recipe_projection": _EXPECTED_RECIPE,
    "calibration_provenance": {
        "cell": SOURCE_VALIDATION_CELL,
        "derived_config_file_sha256": SOURCE_CONFIG_FILE_SHA256,
        "derived_config_semantic_sha256": SOURCE_CONFIG_SEMANTIC_SHA256,
        "scored_artifact": "I-J20-D2.safetensors",
        "claim": (
            "production-compatible effective-U-Net schedule; the frozen stack "
            "created no trainable text-encoder LoRA modules"
        ),
    },
    "checkpoint_policy": {
        "target_fraction": {"numerator": 1, "denominator": 1},
        "mapping_rule": CHECKPOINT_MAPPING_RULE,
        "calibration_artifact": "unnumbered_exact_final",
    },
    "release_authorized": False,
    "deployment_authorized": False,
}
POLICY_SHA256 = hashlib.sha256(_canonical_bytes(_POLICY_BODY)).hexdigest()

# Populated only by the release commit after the exact-score decision closes.
# There is intentionally no environment-variable activation path.
PRODUCTION_ACTIVATION: Mapping[str, Any] | None = None


def _validated_activation(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        if not isinstance(value, Mapping):
            raise ValueError("activation is not a mapping")
        record = dict(value)
        required = {
            "schema",
            "kind",
            "policy_sha256",
            "formal_ideogram_decision_sha256",
            "scored_exact_final_sha256",
            "selected_arm",
            "selection_basis",
            "owner_override",
            "production_mutation_authorized",
            "release_authorized",
            "deployment_authorized",
            "activation_sha256",
        }
        if set(record) != required:
            raise ValueError("activation keys differ")
        body = {k: v for k, v in record.items() if k != "activation_sha256"}
        if (
            record["schema"] != 1
            or record["kind"] != ACTIVATION_KIND
            or record["policy_sha256"] != POLICY_SHA256
            or record["selected_arm"] != "I-J20"
            or record["selection_basis"]
            not in {"clear_win", "null_result_owner_override"}
            or not isinstance(record["owner_override"], bool)
            or record["owner_override"]
            != (record["selection_basis"] == "null_result_owner_override")
            or record["production_mutation_authorized"] is not True
            or record["release_authorized"] is not True
            or record["deployment_authorized"] is not False
            or record["activation_sha256"]
            != hashlib.sha256(_canonical_bytes(body)).hexdigest()
        ):
            raise ValueError("activation identity or authority differs")
        for key in (
            "formal_ideogram_decision_sha256",
            "scored_exact_final_sha256",
        ):
            digest = record[key]
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ValueError(f"{key} is invalid")
        return record
    except Exception as exc:
        telemetry.event(
            "ideogram_production_policy_inactive",
            reason="invalid_activation_record",
            error_type=type(exc).__name__,
        )
        return None


def _recipe_projection(cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        process = cfg["config"]["process"][0]
        network = process["network"]
        save = process["save"]
        dataset = process["datasets"][0]
        train = process["train"]
        model = process["model"]
        return {
            "model_arch": model["arch"],
            "training_seed": process.get("training_seed"),
            "network": {k: network[k] for k in ("type", "linear", "linear_alpha")},
            "save": {
                k: save[k]
                for k in (
                    "dtype",
                    "max_step_saves_to_keep",
                    "save_format",
                    "push_to_hub",
                )
            },
            "dataset": {
                k: dataset[k]
                for k in (
                    "caption_ext",
                    "caption_dropout_rate",
                    "cache_latents_to_disk",
                    "is_reg",
                    "resolution",
                )
            },
            "train": {
                k: train.get(k)
                for k in (
                    "batch_size",
                    "gradient_accumulation",
                    "train_unet",
                    "train_text_encoder",
                    "gradient_checkpointing",
                    "noise_scheduler",
                    "optimizer",
                    "timestep_type",
                    "optimizer_params",
                    "cache_text_embeddings",
                    "lr",
                    "unet_lr",
                    "text_encoder_lr",
                    "lr_scheduler",
                    "lr_scheduler_params",
                    "ema_config",
                    "do_cfg",
                    "cfg_scale",
                    "disable_sampling",
                    "dtype",
                )
            },
        }
    except (IndexError, KeyError, TypeError):
        return None


def apply(
    cfg: dict[str, Any],
    model_type: str,
    *,
    activation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically apply the validated recipe and its exact-final binding."""

    active = _validated_activation(
        PRODUCTION_ACTIVATION if activation is None else activation
    )
    if (
        active is None
        or model_type != "ideogram4"
        or _recipe_projection(cfg) not in (_SOURCE_RECIPE, _EXPECTED_RECIPE)
    ):
        return cfg
    resolved = copy.deepcopy(cfg)
    process = resolved["config"]["process"][0]
    process["training_seed"] = _EXPECTED_RECIPE["training_seed"]
    process["datasets"][0]["cache_latents_to_disk"] = True
    process["train"].update(
        {
            "train_text_encoder": True,
            "lr": 0.000025,
            "unet_lr": 0.000025,
            "text_encoder_lr": 0.0000001,
            "lr_scheduler": "cosine",
            "lr_scheduler_params": {"eta_min": 0.0000025},
            "ema_config": {"use_ema": True, "ema_decay": 0.995},
            "do_cfg": True,
            "cfg_scale": 10.0,
        }
    )
    if _recipe_projection(resolved) != _EXPECTED_RECIPE:
        return cfg
    steps = process["train"]["steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        return cfg
    resolved.setdefault("meta", {}).update(
        {
            "forge_ideogram_production_policy": {
                "schema": 1,
                "policy_id": POLICY_ID,
                "policy_sha256": POLICY_SHA256,
                "activation_sha256": active["activation_sha256"],
                "formal_ideogram_decision_sha256": active[
                    "formal_ideogram_decision_sha256"
                ],
                "scored_exact_final_sha256": active["scored_exact_final_sha256"],
                "source_validation_cell": SOURCE_VALIDATION_CELL,
                "source_config_semantic_sha256": SOURCE_CONFIG_SEMANTIC_SHA256,
                "release_authorized": True,
                "deployment_authorized": False,
            },
            "forge_ideogram_checkpoint_selection": {
                "schema": 1,
                "mapping_rule": CHECKPOINT_MAPPING_RULE,
                "target_fraction": {"numerator": 1, "denominator": 1},
                "planned_steps": steps,
                "selected_step": steps,
                "calibration_artifact": "unnumbered_exact_final",
            },
        }
    )
    telemetry.event(
        "ideogram_production_checkpoint_bound",
        policy_id=POLICY_ID,
        planned_steps=steps,
    )
    return resolved


def checkpoint_control(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], int] | None:
    """Return the validated 1/1 target consumed by finalization."""

    meta = cfg.get("meta")
    if not isinstance(meta, Mapping):
        return None
    binding = meta.get("forge_ideogram_production_policy")
    checkpoint = meta.get("forge_ideogram_checkpoint_selection")
    if binding is None and checkpoint is None:
        return None
    try:
        if not isinstance(binding, Mapping) or not isinstance(checkpoint, Mapping):
            raise ValueError("partial binding")
        if _recipe_projection(cfg) != _EXPECTED_RECIPE:
            raise ValueError("recipe drifted")
        if (
            binding.get("schema") != 1
            or binding.get("policy_id") != POLICY_ID
            or binding.get("policy_sha256") != POLICY_SHA256
            or binding.get("source_validation_cell") != SOURCE_VALIDATION_CELL
            or binding.get("source_config_semantic_sha256")
            != SOURCE_CONFIG_SEMANTIC_SHA256
            or binding.get("release_authorized") is not True
            or binding.get("deployment_authorized") is not False
            or any(
                not isinstance(binding.get(key), str)
                or _SHA256.fullmatch(binding[key]) is None
                for key in (
                    "activation_sha256",
                    "formal_ideogram_decision_sha256",
                    "scored_exact_final_sha256",
                )
            )
        ):
            raise ValueError("policy binding drifted")
        target = checkpoint.get("target_fraction")
        planned = checkpoint.get("planned_steps")
        selected = checkpoint.get("selected_step")
        if (
            checkpoint.get("schema") != 1
            or checkpoint.get("mapping_rule") != CHECKPOINT_MAPPING_RULE
            or checkpoint.get("calibration_artifact") != "unnumbered_exact_final"
            or target != {"numerator": 1, "denominator": 1}
            or isinstance(planned, bool)
            or not isinstance(planned, int)
            or planned <= 0
            or selected != planned
            or cfg["config"]["process"][0]["train"]["steps"] != planned
            or math.gcd(target["numerator"], target["denominator"]) != 1
        ):
            raise ValueError("checkpoint binding drifted")
        return (
            {
                "fraction_numerator": 1,
                "fraction_denominator": 1,
                "selection_rule": CHECKPOINT_MAPPING_RULE,
            },
            selected,
        )
    except Exception as exc:
        raise ValueError("Ideogram production checkpoint binding is invalid") from exc
