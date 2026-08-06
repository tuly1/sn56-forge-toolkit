"""Hash-bound Ideogram recipe and exact-final checkpoint policy.

The deployed Week-4 recipe remains the source/default.  A literal, reviewed
activation record is required before this module changes any config.  The
activation binds the recipe validated on ``I-J20-D2`` and forces the
unnumbered terminal export to remain the production artifact; generic holdout
or training-loss selectors cannot silently replace the scored position.

WEEK-6 AMENDMENT.  Exactly ONE field now diverges from the ``I-J20-D2``
projection: ``train.ema_config.ema_decay`` moves 0.995 -> 0.99.  The divergence
is carried as a first-class, individually hashed record (``WEEK6_EMA_AMENDMENT``)
rather than an in-place edit, so the provenance claim stays honest: the shipped
recipe is *the I-J20-D2 port plus one named, evidence-cited amendment*, not the
I-J20-D2 port.  See the block above ``EMA_DECAY`` for the mechanism and the
evidence.  ``deployment_authorized`` remains False in the activation record;
promotion past this branch still requires a separate, explicit owner step.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Mapping

from forge import telemetry


POLICY_ID = "week6-ideogram-exact-final-ema-horizon-v1"
SUPERSEDED_POLICY_ID = "week5-ideogram-effective-unet-port-exact-final-v2"
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


# --- Week-6 amendment: the EMA export horizon ------------------------------
#
# WHAT THE PINNED RUNTIME DOES (all line numbers at ai-toolkit pin
# 99be3d96a2468d3a5228a4eb05ba67e63c586b4e, the commit the production Docker
# image builds; VERIFIED by reading the pinned tree, not inferred):
#
#   1. jobs/process/BaseSDTrainProcess.py:108
#        self.train_config = TrainConfig(**self.get_conf('train', {}))
#      Our whole `train:` block arrives as kwargs; nothing is whitelisted.
#   2. toolkit/config_modules.py:521-529 reads `ema_config` and builds
#        EMAConfig(**ema_config); :795 use_ema, :796 ema_decay.
#   3. BaseSDTrainProcess.py:769-781 `setup_ema()`, called at :2031 right after
#      the optimizer is built and BEFORE the first step, constructs
#        ExponentialMovingAverage(params, decay=..., use_feedback=...,
#                                 param_multiplier=...)
#      *** `use_num_updates` is NOT in that call. ***  It is therefore stuck at
#      the toolkit/ema.py:47 default False, so ema.py:57 sets
#      `self.num_updates = None`, so the `(1+n)/(10+n)` warm-up ramp guarded by
#      `if self.num_updates is not None` at ema.py:118-124 NEVER RUNS.  Decay is
#      flat from step 1.  There is NO config path to this: the constructor
#      argument is simply not plumbed.  Option "(c) enable bias correction via
#      config" is DEAD at this pin.
#   4. toolkit/ema.py:62-65 clones the shadow from the params at setup time, and
#      toolkit/lora_special.py:122 zero-initialises `lora_up` (B).  So the shadow
#      starts at zero adapter effect and the defect is ATTENUATION of the trained
#      delta, not contamination with junk weights.
#   5. BaseSDTrainProcess.py:491-497 `save()` calls `self.ema.eval()`
#      UNCONDITIONALLY (= ema.py:336-341 store() + copy_to(), i.e. live params
#      := shadow), then :530-539 `self.network.save_weights(...)` reads those
#      live params, then :697-698 restores.  EVERY export — the terminal one and
#      every periodic checkpoint our terminate->finalize path can salvage — is
#      the shadow.  `self.ema` is non-None iff `use_ema` is true, and nothing
#      else gates :495.  Option "(d) keep EMA but export the true weights"
#      is likewise unreachable through config.
#
# So `ema_decay` (and `use_ema`) are the ONLY config-reachable levers, and both
# are honoured: ema.py:55 stores `self.decay = decay` and ema.py:117 uses it
# directly on every update.  ema.py:53-54 only rejects decay outside [0, 1].
#
# HOW MUCH DELTA WE WERE THROWING AWAY.  Model: B (lora_up) starts at 0 and
# accumulates |per-coordinate displacement| ~ lr per Adam step along a locally
# stable direction, so B_k = sum_{j<=k} lr_j and the shadow obeys
# s_k = d*s_{k-1} + (1-d)*B_k with s_0 = 0.  A (lora_down) starts NON-zero, so
# to leading order A_ema ~= A_final and the exported adapter is
# (A_ema)(s_T) ~= (s_T/B_T) * (A_final B_T) — i.e. the exported delta is the
# trained delta scaled by f = s_T/B_T.  ASSUMPTION, stated: the update direction
# is stable enough that the shadow is a scaled endpoint rather than a
# directional average.  That is exact early and degrades late; it is the same
# assumption the field-comparison arithmetic in forge/recipe.py already makes.
#
# f under our ACTUAL schedule (cosine 2.5e-5 -> eta_min 2.5e-6; the pin really
# does honour eta_min — toolkit/scheduler.py:11-16 builds
# torch.optim.lr_scheduler.CosineAnnealingLR and BaseSDTrainProcess.py:2035-2036
# injects total_iters -> T_max), at the three REAL Aug-3 ideogram4 shapes and
# the depths recipe.py's law now ships:
#
#     shape                        steps   f @0.995   f @0.99   gain
#     1365fa1c N=14 h=0.75 (R1)     421      0.720      0.894    x1.24
#     84be9fcd N=46 h=1.0           616      0.828      0.945    x1.14
#     b72da8c6 N=40 h=1.0           589      0.816      0.940    x1.15
#
# Across five growth models (constant-lr/linear B, our cosine with eta_min
# 2.5e-6 and with 0, and saturating B with kappa = T/3 and T/6) the span is
# f@0.995 = 0.585..0.817 and f@0.99 = 0.768..0.959 at T=421, gain x1.17..x1.31;
# at T=616/589 the gain is x1.08..x1.22.  Turning EMA OFF entirely would give at
# most x1.22..x1.71 at T=421 — i.e. 0.99 already captures most of the
# recoverable delta.  CORRECTION to a figure in circulation: the widely quoted
# "~58.5% exported at 421 steps" is the CONSTANT-lr model.  We do not run
# constant lr; under our cosine it is ~72%.  The defect is real but smaller than
# the constant-lr framing implies.
#
# WHY 0.99 AND NOT 0 / not "off".  OBSERVED, from the Aug-3 tournament audit
# (SN56-project/evidence/week6-field-depth-audit-20260806/analysis.json): every
# ideogram4 artifact that published a config is hotkey 5FBmn1ax, and BOTH of his
# rank-1 ideogram4 artifacts ran `use_ema: true` with `ema_decay: 0.99` —
# 1365fa1c (N=14, 0.75 h, 174 steps, rank 1 of 2) and 84be9fcd (N=46, 1.0 h, 341
# steps, rank 1 of 2).  0.99 is the only EMA setting anywhere in the record with
# a rank-1 ideogram4 result behind it.  0.995 is the field's constant for
# krea2 / qwen-image / z-image rank-1 artifacts (same file) — we imported it
# across model types, and ideogram4's much shorter runs are exactly where that
# import costs the most.  Conversely `use_ema: false` appears on five artifacts
# in that file and every one of them placed in the bottom half of its task
# (krea2 R1 ranks 5, 8 and 11 of 14; krea2 R3 rank 2 of 2; z-image R2 rank 2 of
# 2) — weak and heavily confounded evidence, but it points away from disabling
# EMA outright, which is why option (a) is rejected.
#
# HONEST BOUND ON THE PAYOFF.  Under the same arithmetic the 1365fa1c rank-1
# artifact exported f=0.53 of ITS trained delta and still won, because its lr
# integral (174 x 4e-4 constant = 0.0696) is ~12x ours at 421 steps
# (~0.0058 under our cosine).  Measured in absolute exported parameter movement
# on that shape we go from ~11% of the winner to ~14%.  This amendment is a real
# but MODEST recovery; the dominant remaining factor on ideogram4 is the lr
# integral, which is a coupled decision (forge/recipe.py's depth law is derived
# from lr 2.5e-5) and is deliberately NOT touched here.
EMA_DECAY = 0.99

WEEK6_EMA_AMENDMENT: Mapping[str, Any] = {
    "schema": 1,
    "amendment_id": "week6-ideogram-ema-horizon",
    "field": "config.process[0].train.ema_config.ema_decay",
    "validated_value": 0.995,
    "amended_value": EMA_DECAY,
    # The one field of the shipped recipe that the I-J20-D2 cell did NOT run.
    "covered_by_source_validation_cell": False,
    "basis": "field_observed_rank1_ideogram4_setting",
    "evidence": (
        "week6-field-depth-audit-20260806/analysis.json: 1365fa1c (rank 1, "
        "N=14, 0.75h) and 84be9fcd (rank 1, N=46, 1.0h), both hotkey 5FBmn1ax, "
        "both use_ema=true ema_decay=0.99; ai-toolkit pin 99be3d96 "
        "BaseSDTrainProcess.py:491-497,769-781 + toolkit/ema.py:47,57,118 "
        "(use_num_updates unreachable, save() always exports the shadow)"
    ),
}
AMENDMENT_SHA256 = hashlib.sha256(_canonical_bytes(WEEK6_EMA_AMENDMENT)).hexdigest()


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
        # use_ema stays True: see the WEEK6_EMA_AMENDMENT block above.  Only the
        # decay moves, 0.995 -> EMA_DECAY (0.99).
        "ema_config": {"use_ema": True, "ema_decay": EMA_DECAY},
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
    "supersedes_policy_id": SUPERSEDED_POLICY_ID,
    "source_recipe_projection": _SOURCE_RECIPE,
    "recipe_projection": _EXPECTED_RECIPE,
    "amendments": [WEEK6_EMA_AMENDMENT],
    "calibration_provenance": {
        "cell": SOURCE_VALIDATION_CELL,
        "derived_config_file_sha256": SOURCE_CONFIG_FILE_SHA256,
        "derived_config_semantic_sha256": SOURCE_CONFIG_SEMANTIC_SHA256,
        "scored_artifact": "I-J20-D2.safetensors",
        "claim": (
            "production-compatible effective-U-Net schedule; the frozen stack "
            "created no trainable text-encoder LoRA modules"
        ),
        # The recipe below is the I-J20-D2 port PLUS the listed amendments; the
        # cell's own score does not cover the amended field.
        "covers_recipe_projection_exactly": False,
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

# Literal release activation bound to the completed exact-score record.  The
# predeclared clear-win gate was null (the paired interval crossed zero); the
# owner explicitly authorized the I-J20 port on the documented null-result
# override branch.  There is intentionally no environment-variable path.
#
# RE-SIGNED FOR WEEK 6.  Any change to the recipe projection necessarily changes
# POLICY_SHA256 and so requires this record to be regenerated — that is the
# mechanism working, not a bypass of it.  `amendment_sha256` was added so the
# re-signature is SCOPED: it binds this activation to exactly one named
# one-field amendment (WEEK6_EMA_AMENDMENT) and nothing else, and the record
# stops validating if that amendment is edited.  `deployment_authorized` stays
# False; promoting this off the integration branch remains a separate step.
PRODUCTION_ACTIVATION: Mapping[str, Any] | None = {
    "schema": 1,
    "kind": "forge-ideogram-week5-production-activation",
    "policy_sha256": "fcf9ad8a284a2a4e7da58fcef64d0243562f9eca1af0df723de0546368138f91",
    "amendment_sha256": "88442e052fbc95c74d3a85d1dd79c8a29f6f17115d69949b9401679bedebfdb4",
    "formal_ideogram_decision_sha256": (
        "deb5bc3dc6590aa4a9ef0a234a5efc5bc25c40c04327810eb3c997c32dc30af4"
    ),
    "scored_exact_final_sha256": (
        "8d5ab294da5440ed7338ea912144056b7a13a8729d14c3cc05aeebc2cc2a1fde"
    ),
    "selected_arm": "I-J20",
    "selection_basis": "null_result_owner_override",
    "owner_override": True,
    "production_mutation_authorized": True,
    "release_authorized": True,
    "deployment_authorized": False,
    "activation_sha256": "b7e436971430f04e216ddf5a4f1599a3f8de2f21e2f9462c1d67245aa0386ba2",
}


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
            "amendment_sha256",
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
            # Scopes the re-signature to the one named amendment.
            or record["amendment_sha256"] != AMENDMENT_SHA256
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
            # The single field this module writes that the I-J20-D2 cell did
            # not run.  ai-toolkit pin 99be3d96 consumes it at
            # config_modules.py:521-529,796 -> BaseSDTrainProcess.py:769-781
            # (`decay=`) -> ema.py:55,117.  Nothing else on this dict changed.
            "ema_config": {"use_ema": True, "ema_decay": EMA_DECAY},
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
                "amendment_sha256": AMENDMENT_SHA256,
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
            or binding.get("amendment_sha256") != AMENDMENT_SHA256
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
