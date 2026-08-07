"""Hash-bound Ideogram recipe and exact-final checkpoint policy.

The deployed Week-4 recipe remains the source/default.  A literal, reviewed
activation record is required before this module changes any config.  The
activation binds the recipe validated on ``I-J20-D2`` and forces the
unnumbered terminal export to remain the production artifact; generic holdout
or training-loss selectors cannot silently replace the scored position.

WEEK-6 AMENDMENT VACATED (2026-08-06, unit-3 re-derivation).  The Week-6
amendment moved ``train.ema_config.ema_decay`` 0.995 -> 0.99.  It is now
WITHDRAWN and the field is back at 0.995, so the shipped recipe is once again
the ``I-J20-D2`` port with ZERO substantive divergences.  ``WEEK6_EMA_AMENDMENT``
is retained as a hashed VACATION record rather than deleted, so the audit trail
shows the field moved and moved back.  See the block above ``EMA_DECAY`` for the
evaluator mechanism, the in-family matched pair that decided it, and the
arithmetic.  ``deployment_authorized`` remains False in the activation record;
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


# v2 = the v1 EMA-horizon amendment VACATED.  A new id (rather than an in-place
# edit of v1) so the activation record cannot be reused across the change and
# the supersession chain records that the field moved and moved back.
POLICY_ID = "week6-ideogram-exact-final-ema-horizon-vacated-v2"
SUPERSEDED_POLICY_ID = "week6-ideogram-exact-final-ema-horizon-v1"
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


# --- Week-6 EMA export horizon: amendment PROPOSED, then VACATED -----------
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
# HOW MUCH DELTA THE EXPORT CARRIES.  (This arithmetic is unchanged by the
# vacation and is what the decision below is computed on.  Note the framing
# correction: a lower f is ATTENUATION, and on ideogram4 attenuation is not
# automatically a defect — see (1)-(5) below, where the evaluator's ~7x
# amplification makes a SMALLER exported delta the thing we want.)  Model: B
# (lora_up) starts at 0 and
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
#     shape                        steps   f @0.995   f @0.99   0.99 is
#     (anchor 3cfa1578 N=11 h=.75)  390      0.695      0.880    x1.27
#     1365fa1c N=14 h=0.75 (R1)     421      0.720      0.894    x1.24
#     b72da8c6 N=40 h=1.0           589      0.816      0.940    x1.15
#     84be9fcd N=46 h=1.0           616      0.828      0.945    x1.14
#
# Across five growth models (constant-lr/linear B, our cosine with eta_min
# 2.5e-6 and with 0, and saturating B with kappa = T/3 and T/6) the span is
# f@0.995 = 0.585..0.817 and f@0.99 = 0.768..0.959 at T=421, ratio x1.17..x1.31;
# at T=616/589 the ratio is x1.08..x1.22.  So the SIGN and rough SIZE of the
# 0.995 -> 0.99 strength increase (+14% to +31%) are robust to the growth model;
# only the absolute level of f is model-dependent.  CORRECTION to a figure in
# circulation: the widely quoted "~58.5% exported at 421 steps" is the
# CONSTANT-lr model.  We do not run constant lr; under our cosine it is ~72%.
#
# WHY THE AMENDMENT IS VACATED AND THE FIELD IS BACK AT 0.995.
#
# (1) THE EVALUATOR AMPLIFIES THE ADAPTER ~7x ON THIS TYPE, AND ONLY THIS TYPE.
# OBSERVED, from the G.O.D pin b026da04
# validator/evaluation/comfy_workflows/lora_ideogram4.json:
#   - `Lora_loader` feeds ONLY the positive path (-> `CFG_override` ->
#     `Dual_model_guider.model`).
#   - `Dual_model_guider.model_negative` is `Unconditional_checkpoint_loader`,
#     a SEPARATE checkpoint (`ideogram4_unconditional_fp8_scaled.safetensors`)
#     that the LoRA is never applied to.
#   - `Dual_model_guider` is a DualModelGuider at cfg 8; `CFG_override` drops it
#     to cfg 5 over sigma-percent [0.7, 1.0].  diffusion.py:235-236 writes both
#     from the eval cfg: `Dual_model_guider.cfg = 8` and
#     `CFG_override.cfg = max(cfg - 3, 1) = 5`.  constants.py:31 fixes cfg 8,
#     steps 30, denoise 0.75.
# So the guided prediction is  pred = neg + c*(pos - neg)  with `neg` completely
# LoRA-INDEPENDENT.  A LoRA that shifts the positive branch's prediction by d
# shifts the final prediction by EXACTLY c*d.  With denoise 0.75 the sampled arc
# is sigma-percent [0.25, 1.0], of which (0.70-0.25)/0.75 = 60% runs at c=8 and
# 40% at c=5, i.e. an arc-averaged c ~= 6.8.  INFERRED (the percent-to-sigma map
# is not linear in step index), but the endpoints 8 and 5 are OBSERVED.
# CONTRAST: krea2/qwen-image/z-image use one model on both branches, so the
# common-mode part of the LoRA delta passes at 1x, and at the blank prompt
# (0.75 of the score) positive == negative == "" makes CFG an exact no-op.
# ideogram4 is the only type where the LoRA is amplified in BOTH scored
# conditions, i.e. across 100% of the score.
#
# (2) THEREFORE THE LOSS IS QUADRATIC IN (c * strength), SO OPTIMAL STRENGTH
# GOES AS 1/c AND THE OVERSHOOT PENALTY GOES AS c^2.  Writing the pixel error
# as E0 - c*s*J*d_u and the off-target part as c*s*J*d_h, the plain per-pixel
# MSE the validator computes (diffusion.py:203-209) is
#     MSE(s) = L0 - 2*c*s*G + (c*s)^2*H
# with G = <E0, J d_u> and H = |J d_u|^2 + |J d_h|^2.  Hence c*s* = G/H, i.e.
# s* ~ 1/c, and d2MSE/ds2 = 2*c^2*H.  At c ~= 6.8 versus c = 1 that is a ~46x
# sharper penalty for being too strong.  This is a DERIVATION from (1), not a
# measurement — but (3) measures it.
#
# (3) THE ONE CONTROLLED STRENGTH EXPERIMENT IN THE ENTIRE IDEOGRAM4 RECORD SAYS
# STRONGER IS CATASTROPHIC, AND IT IS IN OUR OWN FAMILY.  Jul-20 R1 task
# 3cfa1578 (ideogram4, N=11, h=0.75, SIXTEEN miners — the largest ideogram4
# field in existence, at the R1 shape).  Two entrants shipped configs that
# differ in ONE variable, verified by diffing the two published config.yaml
# files (hf gradients-io-tournaments/tournament-tourn_4aff76a867d2af49_20260720
# -3cfa1578-5506-4a1d-a79a-2c55abc8958b-<hotkey>/checkpoints/config.yaml):
#     5FNLSgh8  sha256 bf852a1aaa954ad8aaeea0b9522f4b8147dc84a88f1c6aeedde7f1c
#               7201a954c  (same digest recorded independently on 2026-07-22 in
#               SN56-project/SN56-WEEK4-INDEPENDENT-REVIEW-2026-07-22.md:66)
#     5EACrayt  sha256 eeb914952cf4672f8b83a0f2e54237319b1b828d003273a7bb35dc78
#               03128f30
# The ENTIRE diff is the learning rate scaled 2x (lr and unet_lr 2.5e-5 ->
# 5.0e-5, text_encoder_lr 1e-7 -> 8e-7, and the matching min_lr_by_initial_lr
# keys).  IDENTICAL in both: steps 378, ema_decay 0.995, use_ema true,
# lr_scheduler cosine_by_group, network lora linear 32 / linear_alpha 32,
# caption_dropout_rate 0.05, cache_latents_to_disk true, do_cfg true,
# cfg_scale 10.0, timestep_type linear, noise_scheduler flowmatch, adamw8bit,
# weight_decay 1e-4, batch_size 1, gradient_accumulation 1, train_text_encoder
# true, resolution [512,768,1024], dtype bf16, save_every 200.
# RESULT (scores per forge/recipe.py:211-212): 5FNLSgh8 RANK 1 of 16 at
# test_loss 0.0502341; 5EACrayt RANK 13 of 16 at 0.0965093.  +92.1% loss for 2x
# strength at identical depth, in the recipe family this module ships.
# Fitting (2) to that point (at s = 2s* the model gives MSE = L0 exactly, i.e.
# the whole adapter benefit is destroyed) yields the local sensitivity
#     dMSE/MSE(s*) = 0.921 * (s/s* - 1)^2.
#
# (4) WHERE 0.995 AND 0.99 PUT US RELATIVE TO THAT ANCHOR.  Exported strength is
# (lr integral) x f, with f = s_T/B_T from the model documented above.  At the
# anchor's own shape (N=11, h=0.75, where forge/recipe.py's law ships 390 steps
# against the anchor's 378):
#     decay 0.995 -> f 0.695, lr-int 0.005363, exported 0.003728 = 1.047x anchor
#     decay 0.99  -> f 0.880, lr-int 0.005363, exported 0.004721 = 1.326x anchor
# (the anchor itself: 378 steps, f 0.685, lr-int 0.005198, exported 0.003560)
# So 0.995 reproduces the rank-1-of-16 in-family artifact's exported strength to
# within ~5%; the amendment overshoots it by ~33%.  Through (3) that overshoot
# prices at 0.921 * 0.326^2 = +9.8% loss, against +0.2% for 0.995.  Our Aug-3 R1
# elimination margin was 0.97%.  The amendment stakes ~9x that margin on the
# one direction of the strength axis that has a measurement, and the
# measurement is bad.
#
# (5) WHY 5FBmn1ax's 0.99 IS NOT EVIDENCE FOR 0.99 HERE.  OBSERVED (Aug-3
# artifacts, verified from the config.yaml sidecars in
# SN56-project/evidence/week6-field-depth-audit-20260806/raw/): both of his
# rank-1 ideogram4 runs do carry `use_ema: true, ema_decay: 0.99` — at 174 and
# 341 steps, with lr 4e-4 CONSTANT and no scheduler.  But an EMA's averaging
# horizon is decay/(1-decay): 199 steps at 0.995 versus 99 at 0.99.  At his 174
# steps the 0.995 horizon is 114% OF THE WHOLE RUN (f would be 0.334 versus
# 0.530 at 0.99), so 0.99 is forced by his depth, not chosen for ideogram4.  Our
# shapes are 421/589/616, where the 0.995 horizon is 47%/34%/32% of the run and
# the anchor ran it at 378 (53%) and won.  His constant is a property of a
# 16x-hotter, 4x-shallower recipe; importing it is the same cross-family error
# the previous version of this block committed in the other direction.  His
# third ideogram4 run (b72da8c6, 1523 steps) has no `ema_config` at all and
# LOST, so within his own three runs depth and EMA are perfectly confounded and
# nothing about EMA is identified.
#
# WHAT THIS DOES NOT CLAIM.  s* is task-dependent (G and H are properties of the
# held-out images), so the 0.921 coefficient is a local sensitivity from one
# task, not a universal constant, and the 1.04x/1.32x figures are ratios to one
# artifact rather than to a measured optimum.  The asymmetry is what decides it:
# there is no observation anywhere in the ideogram4 record of MORE exported
# strength scoring BETTER, and one clean observation of 2x scoring far worse.
EMA_DECAY = 0.995  # I-J20-D2 / 5FNLSgh8 anchor value; see (1)-(5) above

WEEK6_EMA_AMENDMENT: Mapping[str, Any] = {
    "schema": 1,
    "amendment_id": "week6-ideogram-ema-horizon",
    "field": "config.process[0].train.ema_config.ema_decay",
    "validated_value": 0.995,
    # VACATED: the amended value is back at the validated value, so the shipped
    # recipe no longer diverges from the source cell on this or any field.
    "amended_value": EMA_DECAY,
    "status": "vacated",
    # True again: with the divergence withdrawn, every field of the shipped
    # projection is a field the I-J20-D2 cell actually ran.
    "covered_by_source_validation_cell": True,
    "basis": "vacated_on_in_family_matched_pair_and_evaluator_amplification",
    "evidence": (
        "PROPOSED on: week6-field-depth-audit-20260806/analysis.json — 1365fa1c "
        "(rank 1, N=14, 0.75h) and 84be9fcd (rank 1, N=46, 1.0h), both hotkey "
        "5FBmn1ax, both use_ema=true ema_decay=0.99.  VACATED on: (a) G.O.D pin "
        "b026da04 lora_ideogram4.json — DualModelGuider's negative branch is a "
        "separate un-LoRA'd checkpoint, so the LoRA delta enters the scored "
        "prediction at cfg (8, falling to 5 via CFG_override over the last 30%; "
        "diffusion.py:235-236), arc-averaged ~6.8x, making optimal adapter "
        "strength ~1/c and the overshoot penalty ~c^2; (b) Jul-20 R1 task "
        "3cfa1578 (ideogram4, N=11, h=0.75, 16 miners) 5FNLSgh8 vs 5EACrayt — "
        "published configs identical except lr 2.5e-5 -> 5.0e-5, giving rank 1 "
        "(0.0502341) vs rank 13 (0.0965093), i.e. 2x strength costs +92.1% loss "
        "at identical depth in this module's own recipe family; (c) 5FBmn1ax's "
        "0.99 is forced by his 174-341 step depths, where the 0.995 averaging "
        "horizon (199 steps) meets or exceeds the run length, and his one deep "
        "ideogram4 run carries no ema_config at all and lost.  Runtime facts "
        "unchanged: ai-toolkit pin 99be3d96 BaseSDTrainProcess.py:491-497,"
        "769-781 + toolkit/ema.py:47,57,118 (use_num_updates unreachable, "
        "save() always exports the shadow)."
    ),
    "vacated_on": "2026-08-06",
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
        # use_ema True, decay EMA_DECAY (0.995) — the source cell's own values.
        # See the vacated-amendment block above for why 0.99 was rejected.
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
    # EMPTY: the shipped recipe has no live divergence from the source cell.
    # The withdrawn one is kept alongside, hashed, for the audit trail.
    "amendments": [],
    "vacated_amendments": [WEEK6_EMA_AMENDMENT],
    "calibration_provenance": {
        "cell": SOURCE_VALIDATION_CELL,
        "derived_config_file_sha256": SOURCE_CONFIG_FILE_SHA256,
        "derived_config_semantic_sha256": SOURCE_CONFIG_SEMANTIC_SHA256,
        "scored_artifact": "I-J20-D2.safetensors",
        "claim": (
            "production-compatible effective-U-Net schedule; the frozen stack "
            "created no trainable text-encoder LoRA modules"
        ),
        # True again with the EMA amendment vacated: every field of the
        # projection below is a field the I-J20-D2 cell actually ran.  (The
        # `lr_scheduler: cosine` + `eta_min` spelling versus the field
        # artifact's `cosine_by_group` + `min_lr_by_initial_lr` is the
        # documented effective-schedule port, not a divergence in value: both
        # anneal 2.5e-5 -> 2.5e-6.)
        "covers_recipe_projection_exactly": True,
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
# RE-SIGNED TWICE IN WEEK 6.  Any change to the recipe projection necessarily
# changes POLICY_SHA256 and so requires this record to be regenerated — that is
# the mechanism working, not a bypass of it.  The first re-signature carried the
# EMA-horizon amendment (0.995 -> 0.99); THIS one carries its VACATION, so the
# projection is back to the bare I-J20-D2 port.  `amendment_sha256` keeps the
# signature SCOPED: it binds this activation to exactly one named record — now a
# vacation rather than a live divergence — and stops validating if that record
# is edited.  `deployment_authorized` stays False; promoting this off the
# integration branch remains a separate step.
PRODUCTION_ACTIVATION: Mapping[str, Any] | None = {
    "schema": 1,
    "kind": "forge-ideogram-week5-production-activation",
    "policy_sha256": "aff7ce75bac9f28e40a0e7c8e1991d5a0482107b16b7be5c9699bd3b2189ea51",
    "amendment_sha256": "501311769658b813e61a6bd6a9a2b59d2dc8ca664d70a804c9755216d4267675",
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
    "activation_sha256": "04261257fadfc780fe70f557b1f5b6c6672e09631b804d51d9d28863f0ba348c",
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
