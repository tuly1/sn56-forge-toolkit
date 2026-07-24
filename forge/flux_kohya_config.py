"""Deterministic Kohya configuration for validator-routed standalone FLUX.

G.O.D stages some FLUX bases as a single ``.safetensors`` transformer file and
routes those tasks to ``standalone-image-trainer.dockerfile``.  ai-toolkit's
directory loader cannot consume that cache shape offline; Kohya's FLUX loader
can.  This module contains only the frozen, networkless configuration surface
needed by that legacy-named path.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

from forge import recipe


MAX_TRAIN_STEPS = 250
SAVE_EVERY_STEPS = 25
SAVE_LAST_N_STEPS = 100
# R11's frozen standalone child ran for 1,576.6 seconds before its deadline
# termination.  During that window step 75 was the last durable checkpoint;
# step 100 was interrupted while saving and was not a valid candidate.  Use the
# conservative effective rate (durable steps / full child runtime) and retain
# 20% additional headroom.  This is an operational completion calibration, not
# a claim that the resulting step count is quality-optimal.
R11_LAST_DURABLE_STEPS = 75
R11_OBSERVED_CHILD_RUNTIME_S = 1_576.6
DEADLINE_THROUGHPUT_HEADROOM = 0.80
WINNER_REFERENCE = (
    "https://github.com/gradients-opensource/"
    "god-image-tourn-4aff76a867d2af49-20260720-position-1"
)

AE_PATH = "/app/flux/ae.safetensors"
CLIP_L_PATH = "/app/flux/clip_l.safetensors"
T5XXL_PATH = "/app/flux/t5xxl_fp16.safetensors"
TOKENIZER_CACHE_DIR = "/app/flux/tokenizers"


def build_config(
    *,
    base_model: str,
    train_data_dir: str,
    output_dir: str,
    output_name: str,
    config_file: str,
    steps: int = MAX_TRAIN_STEPS,
) -> dict[str, Any]:
    """Return the frozen operational FLUX recipe.

    The shape follows the most recent public tournament winner's operational
    FLUX configuration (guidance-matched training, rank 128, Lion, 250 steps),
    while checkpoint cadence is made step-based for Forge's deadline fallback.
    """
    steps = max(1, int(steps))
    # Reuse the common fixed-candidate cadence.  Re-scaling to ``steps // 5``
    # would make short deadline-capped runs spend a disproportionate share of
    # their budget writing multi-gigabyte checkpoints.
    save_every = recipe.kill_safe_save_every(steps, SAVE_EVERY_STEPS)
    if steps > 1 and save_every >= steps:
        # The common cadence deliberately changes regimes at 25 steps.  At the
        # boundary it can otherwise place the first periodic save at the exact
        # terminal step, leaving no recovery candidate if throughput degrades.
        save_every = max(1, steps // 2)
    return {
        # Model and immutable support assets baked into the pinned base image.
        "pretrained_model_name_or_path": base_model,
        "ae": AE_PATH,
        "clip_l": CLIP_L_PATH,
        "t5xxl": T5XXL_PATH,
        "t5xxl_max_token_length": 512,
        "tokenizer_cache_dir": TOKENIZER_CACHE_DIR,
        "apply_t5_attn_mask": True,
        # Output. No Hub or tracker keys are emitted: tournament runtime is
        # offline and the validator uploads this directory itself.
        "output_dir": output_dir,
        "output_name": output_name,
        # The pinned Apr-2025 Kohya loader reconstructs an argparse Namespace
        # from TOML and otherwise replaces the CLI's config_file with None before
        # calling splitext. Self-binding the exact path is its required contract.
        "config_file": config_file,
        "save_model_as": "safetensors",
        "save_precision": "float",
        "save_every_n_steps": save_every,
        "save_last_n_steps": max(save_every, min(SAVE_LAST_N_STEPS, steps)),
        # This pinned Kohya revision consumes mem_eff_save directly from the
        # TOML-injected Namespace even though it does not declare an argparse
        # action for it. Omitting it makes checkpoint export raise AttributeError.
        "mem_eff_save": True,
        "no_metadata": True,
        # Precision and memory.
        "mixed_precision": "bf16",
        "full_bf16": True,
        "highvram": True,
        "gradient_checkpointing": True,
        "xformers": True,
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "vae_batch_size": 4,
        # Dataset and bucketing.
        "train_data_dir": train_data_dir,
        "resolution": "1024,1024",
        "bucket_no_upscale": True,
        "bucket_reso_steps": 64,
        "min_bucket_reso": 256,
        "max_bucket_reso": 2048,
        "caption_extension": ".txt",
        "caption_dropout_rate": 0.1,
        # FLUX scoring distribution.
        "guidance_scale": 85.0,
        "timestep_sampling": "sigmoid",
        "discrete_flow_shift": 3.1582,
        "model_prediction_type": "raw",
        "max_timestep": 1000,
        # Operational optimization recipe.
        "max_train_steps": steps,
        "train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "optimizer_type": "Lion",
        "optimizer_args": ["weight_decay=0.005", "betas=(0.9,0.99)"],
        "unet_lr": 5.0e-5,
        "text_encoder_lr": [5.0e-6, 5.0e-6],
        "lr_scheduler": "cosine",
        "lr_scheduler_args": [],
        "lr_scheduler_num_cycles": 1,
        "lr_scheduler_power": 1,
        # Loss and noise.
        "loss_type": "l2",
        "huber_c": 0.1,
        "huber_scale": 1,
        "huber_schedule": "snr",
        "prior_loss_weight": 1,
        # LoRA capacity and trainable blocks.
        "network_module": "networks.lora_flux",
        "network_dim": 128,
        "network_alpha": 64,
        "network_args": [
            "train_double_block_indices=all",
            "train_single_block_indices=all",
            "train_t5xxl=True",
        ],
        # Runtime determinism.
        "max_data_loader_n_workers": 4,
        "seed": 2,
    }


def budgeted_train_steps(
    remaining_soft_s: float,
    *,
    boundary_margin_s: float,
    max_steps: int = MAX_TRAIN_STEPS,
) -> int:
    """Cap the winner-derived recipe to a naturally completable task budget.

    ``remaining_soft_s`` already excludes Forge's export reserve.  The caller
    supplies the additional process-termination boundary margin.  The measured
    rate is intentionally based only on R11's last fully durable checkpoint;
    the interrupted step-100 write is not counted as usable throughput.
    Invalid or exhausted budgets fail safe to one planned step, which still
    leaves the normal fallback/finalization path alive.
    """
    try:
        remaining = float(remaining_soft_s)
        raw_boundary = float(boundary_margin_s)
        ceiling = max(1, int(max_steps))
        if not math.isfinite(remaining) or not math.isfinite(raw_boundary):
            return 1
        boundary = max(0.0, raw_boundary)
        usable = max(0.0, remaining - boundary)
        calibrated = math.floor(
            usable
            * (R11_LAST_DURABLE_STEPS / R11_OBSERVED_CHILD_RUNTIME_S)
            * DEADLINE_THROUGHPUT_HEADROOM
        )
        return max(1, min(ceiling, calibrated))
    except (TypeError, ValueError, OverflowError):
        return 1


def write_config(config: dict[str, Any], path: str) -> None:
    """Atomically serialize the flat Kohya config as TOML."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".tmp"
    try:
        with open(temp, "w", encoding="utf-8") as fh:
            for key, value in config.items():
                if not key or not key.replace("_", "").isalnum():
                    raise ValueError(f"unsafe TOML key: {key!r}")
                fh.write(f"{key} = {_toml_value(value)}\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            os.remove(temp)
        except OSError:
            pass
        raise


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {type(value).__name__}")
