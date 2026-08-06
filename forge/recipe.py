"""Step / save-cadence / wall-clock policy — OUR EDGE.

The eval scores 0.25*caption-guided + 0.75*UNCONDITIONAL L2 reconstruction. Our
depth policy scales steps sublinearly with dataset size (a power law) and then
caps by a wall-time budget so we finish rather than get killed.

WEEK-6 RECALIBRATION (2026-08-06) — read this before changing a number.
======================================================================
Every constant below was re-derived from the COMPLETE Aug-3 tournament field:
all 40 published artifacts across 14 tasks (`tourn_c54bb970b5d0aa91_20260803`),
305 safetensors ``__metadata__.training_info`` headers and 24 published
``checkpoints/config.yaml`` files.  Audit + reproduction scripts:

  ops/experiments/week6/FIELD-DEPTH-LAW-AUDIT.md
  /Users/atulyashetty/Test/SN56-project/evidence/week6-field-depth-audit-20260806/

Two premises that used to live in this file are FALSIFIED and have been deleted:

  1. "over-training is the #1 liability" / "deep training never helped".
     That came from a Jul-16 probe over 8..128 steps on 12 photos — the wrong
     regime entirely.  Five miners published their own per-checkpoint
     reconstruction curves (`.krea2/.zimage/.flux_checkpoint_evaluations.json`,
     already split into the validator's own text/no-text terms).  FIVE CURVES,
     FIVE MINIMA AT THE DEEPEST CHECKPOINT EVALUATED — not one turns over.  On
     the only open field of the tournament (R1 krea2, 14 ranked artifacts),
     Spearman(steps, test_loss) inside the template recipe family (n=9) is
     -0.605: deeper was better.  We shipped 823 steps (39/img) and placed 9/15;
     the template pack's top four ran 1432-2000.
  2. "krea2's do_differential_guidance adds a second guidance forward per step".
     That code is unreachable: `do_differential_guidance` is nested inside
     `if self.train_config.do_guidance_loss:` (SDTrainer.py:692 guard at indent
     8 vs :734 at indent 12) and we never set `do_guidance_loss`.  It has never
     executed, so it cannot justify a slow per-step constant.

The single biggest defect the audit found is NOT the depth law: in 11 of the 14
Aug-3 tasks the `SEC_PER_IT` wall-time cap truncated the size law before the
size law was ever consulted.  See the SEC_PER_IT block below.
"""

from __future__ import annotations

# --- step-scaling table ----------------------------------------------------
# steps = clamp(base * (N / n_ref)**p, min, max), then capped by wall clock.
#
# Provenance tags used below:
#   OBSERVED  = read out of a published artifact's metadata / config
#   INFERRED  = derived (e.g. kohya writes epochs, not steps)
_N_REF = 24  # ~mid of the 10-50 pair range

STEP_TABLE = {
    #            base  n_ref   p    min    max
    # flux — UNCHANGED. The law is already right: uncapped it returns 870 steps
    # at N=15, which is EXACTLY what the rank-1 miner shipped on both Aug-3 flux
    # tasks (5FW2Eaae, 58 epochs x 15 imgs; 5D7iEJm5, 50 x 15 = 750).  Only the
    # clock was wrong (see SEC_PER_IT).  The audit rates flux depth evidence TOO
    # THIN to move (n=4 artifacts, 2 head-to-heads) and the step counts are
    # INFERRED, not observed, because kohya records epochs.  Deliberately left
    # alone.  NOTE: `forge/tasks/flux_kohya.py` routes standalone-checkpoint
    # FLUX bases to kohya, where depth comes from
    # `flux_kohya_config.MAX_TRAIN_STEPS`, NOT from this row — a separate open
    # decision (FIELD-DEPTH-LAW-AUDIT §6.5).
    "flux": dict(base=1100, n_ref=_N_REF, p=0.50, min=500, max=2000),
    # krea2 — was base=1200 p=0.50 min=100 max=2000 (0.64x the field winners).
    # The 8-win operator 5FBmn1ax's krea2 policy is a PURE CLOCK-FILL with NO
    # size term, recovered exactly from his own published step counts:
    #     steps = (hours*3600 - 478) / 1.552
    #     h=1.00 -> 2012, published 2012 on N=42 AND N=43 AND N=50 (identical
    #               across 8x the dataset-size range)
    #     h=0.75 -> 1432, published 1432 on N=21
    # p is flattened 0.50 -> 0.35 to mirror that near-size-independence while
    # keeping a floor for pathologically small sets; `base`/`max` are raised so
    # the RECALIBRATED CLOCK (not the size law) becomes the binding constraint
    # on 1.0 h tasks, which is exactly the champion's behaviour.  `min` 100->600
    # because no 100-step krea2 is competitive in anything we observed — the one
    # 200-step krea2 in the R1 field ranked 13/14.  Replay on the four real
    # Aug-3 krea2 shapes: 1825/1336/1840/1888 vs winners 2012/1000/2012/2012
    # = 1.02x (was 0.64x).
    # RESIDUAL UNCERTAINTY, largest of any row: the R1 winner shipped only 1000
    # steps and beat the 1432/1750/2000 pack — with an entirely different recipe
    # (krea2_eval_sigmas timesteps, TE-LoRA, EMA 0.995, multires noise,
    # cosine_by_group, differential_guidance_scale 12).  Depth is necessary, not
    # sufficient; this moves us from 39 to ~64 steps/img, i.e. into the band of
    # ranks 3-5, not into rank 1.
    "krea2": dict(base=1500, n_ref=_N_REF, p=0.35, min=600, max=2200),
    # ideogram4 — was base=140 p=0.50 min=48 max=400 (the discredited Jul-16
    # experiment), then base=240 p=0.57 min=120 max=1600, a two-point fit to the
    # champion's own published step counts (N=14 -> 174, N=46 -> 341).  BOTH are
    # rejected, and the second for the reason the first one hid:
    #
    # THE CHAMPION'S STEP COUNTS ARE NOT TRANSFERABLE TO OUR RECIPE.  Every
    # ideogram4 config in the field runs `lr: 0.0004` CONSTANT with
    # `ema_decay 0.99` (OBSERVED in 5FBmn1ax's three published config.yaml).
    # `forge.ideogram_release_policy` runs us at lr 2.5e-5 cosine-decayed to
    # 2.5e-6 — a 16x lower peak, ~29x lower mean — with `ema_decay 0.995`.
    # Under Adam the per-coordinate displacement per step is ~lr, so the
    # comparable quantity across two recipes is the lr INTEGRAL, not the step
    # count:
    #     champion, 1365fa1c : 174 steps x 4e-4 constant   = 0.0696
    #     us at the old law's 177 steps                    = 0.00245  (28.5x less)
    #     ...and `save()` exports the EMA, a lagged average rather than the
    #     final iterate, which costs another 2.3x at that depth
    #                                                      = 0.00106  (65.6x less)
    # Shipping "177 because the champion shipped 174" does not reproduce his
    # shallow optimum; it ships ~1.5% of his parameter movement.  That single
    # factor invalidates the two-point fit, so it is gone.
    #
    # WHAT THE FIELD ACTUALLY SUPPORTS — re-derived independently from the
    # safetensors `__metadata__` headers of all SIX Aug-3 ideogram4 artifacts:
    #   1365fa1c product N=14 h=0.75: 174 (rank 1) vs >900 (rank 2, +46.1% loss)
    #       The ONLY clean shallow-vs-deep head-to-head anywhere in ideogram4,
    #       and shallow won big.  But the deep arm published NO config and
    #       stripped its metadata (its depth is only bounded by a last_000000900
    #       checkpoint), so the 46% cannot be attributed to depth alone.
    #   84be9fcd style N=46 h=1.0: 341 (rank 1) vs UNKNOWN (rank 2 published two
    #       files, no `__metadata__`, no checkpoint ladder).  This point carries
    #       ZERO DEPTH INFORMATION and must not be used to calibrate anything.
    #   b72da8c6 style N=40 h=1.0: >=1200 (rank 1) vs 1523 (rank 2, +4.4%).
    #       BOTH arms deep, no shallow arm.  It says 1250 ~= 1523; it does NOT
    #       say that 321 would have lost.
    #   Jul-20 R1 3cfa1578 ideogram4, N=9, SIXTEEN miners — the largest
    #       ideogram4 sample we have (SN56-WEEK3-POSTMORTEM.md §6a): we ran the
    #       shortest run in the field (85 steps) and placed 4/16; the deep
    #       cluster was 722-1000+, the bracket winner 1200, and the recorded
    #       finding is that the img2img metric "did NOT punish overtraining".
    # Net across two tournaments: ideogram4 depth is FLAT and WIDE — 85 to 1250
    # are all competitive — with the deep end favoured in the larger field.
    # There is NO size law to fit: N=9 -> 1200, N=14 -> 174, N=40 -> 1250,
    # N=46 -> 341 is uncorrelated with size.
    #
    # NO FAMILY SPLIT, and NOT because it is infeasible.  `spec.trigger_word is
    # None` separates `style` from every other family 12/12 across the Aug-3
    # configs (both style tasks carry no trigger word; all ten design/social/
    # logo/product configs do), so a style router IS implementable from data the
    # container actually receives.  It is rejected because the two style tasks
    # disagree with EACH OTHER by 4.5x — 341 won at N=46, ~1250 won at N=40 —
    # so there is no style depth to route TO.  Adding a one-task-per-branch
    # parameter to the highest-variance row in this table, which is half the R1
    # draw, is the worst available trade.
    #
    # THE ROW BELOW IS THEREFORE NOT A FIT TO THE FIELD.  It is set from the two
    # constraints we can measure on OUR OWN pipeline:
    #  (a) EMA FLOOR.  ai-toolkit seeds the EMA shadow from the LoRA at init
    #      (B = 0, i.e. zero effect), constructs it with `use_num_updates=False`
    #      so there is no warm-up or bias correction, and `save()` ALWAYS
    #      exports the EMA (`BaseSDTrainProcess.py:566-568,840-851`;
    #      `toolkit/ema.py` __init__/update).  So 0.995^T of every artifact we
    #      upload is the untrained init: 41% at 177 steps, 12% at 421, 4.6% at
    #      616.  Depth is the ONLY lever on this that does not require
    #      re-signing the hash-bound release activation record.
    #  (b) CLOCK CEILING AT OUR COST.  `do_cfg: true` runs the transformer at
    #      batch 2 every step, so SEC_PER_IT is 4.2 for us, not the field's
    #      2.05 -> the 1.0 h cap is 674 steps and the 0.75 h cap is 477.  The
    #      b72da8c6 winning regime (~1250) is UNREACHABLE while do_cfg is on.
    #      That is a do_cfg decision, not a depth decision.
    # base/p are chosen so the SIZE LAW binds ~10-15% below that ceiling on all
    # three real shapes — 421 / 589 / 616 against caps 477 / 674 / 674 — taking
    # grant utilisation from 45-54% to 82-85%.  KNOWN CORNER, stated rather than
    # hidden: on a 0.75 h grant the law crosses the 477 cap at N~21, so a SHORT
    # R1 ideogram task with a mid-size dataset is clock-bound at 477 (432 if the
    # margin reverts to 0.85).  That is the do_cfg tax being visible, not a
    # miscalibration; it is still 1.3x the withdrawn law and it keeps four
    # periodic candidates.  Every 1.0 h shape up to N=50 stays law-bound.
    # Because the law binds on the real shapes and the
    # clock does not, this row is invariant to MARGIN and to any SEC_PER_IT
    # revision, and it absorbs a 21% error in the INFERRED 4.2 s/step constant
    # before anything truncates (the tightest shape, 84be9fcd at 616 steps,
    # breaks only above 5.06 s/step = 2.47x the field's own no-do_cfg bound);
    # a truncation then DEGRADES rather than forfeits
    # (`forge/tasks/aitoolkit.py` _run_toolkit -> _terminate -> _finalize
    # promotes the newest periodic save, and there are four of them).
    # p 0.57 -> 0.32 because the field shows no size signal at all; this mirrors
    # krea2's deliberately flat 0.35 rather than a 2-point fit.  min 120 -> 350
    # binds only below N~8, under the smallest ideogram4 dataset ever observed
    # (N=9, Jul-20).  max 1600 -> 620: 1600 was INERT (the old law topped out at
    # 365 at N=50, so it could never bind within 4x), whereas 620 binds from
    # N >= 47, inside the observed 9-50 range, and caps extrapolation of a
    # recipe we have never run past ~200 steps in a tournament.
    # PRE-COMMIT: this is a reasoned bet, not a measurement.  If ideogram4 is
    # the R1 draw and we lose, the correct next experiment is do_cfg on/off at
    # matched depth (which buys back 2x the reachable depth), NOT another depth
    # change.
    "ideogram4": dict(base=500, n_ref=_N_REF, p=0.32, min=350, max=620),
    # z-image — was base=1100 p=0.50 min=400 max=2000.  Cleanest result in the
    # audit: TWO DIFFERENT rank-1 operators with different recipes imply the
    # same law to 0.1% —
    #     5FBmn1ax N=48 -> 1317  => base = 1317*(24/48)^0.5 = 931
    #     5EACrayt N=39 -> 1188  => base = 1188*(24/39)^0.5 = 932
    # base=930 reproduces both winners to within 2 steps (1315 vs 1317, 1186 vs
    # 1188).  `base` DROPS but shipped depth RISES 860 -> 1315, because the real
    # defect was SEC_PER_IT=3.0 while a field miner completed 2000 steps in the
    # same 1.0 h.  `max` pulled to 1800 to stay out of the flat-2000-template
    # regime, whose entrant lost by 7.9%.
    "z-image": dict(base=930, n_ref=_N_REF, p=0.50, min=350, max=1800),
    # qwen-image — DEPTH WAS ALREADY RIGHT (now/win = 1.00x).  This row changes
    # only so it is right for the right REASON: today we land on the field by
    # accident, because the size law wants 1137/1307 and SEC_PER_IT=4.0 happens
    # to cut it to 1027.  Fix the clock without fixing the law and qwen instantly
    # over-trains by 25%.  The champion's exact recovered law is
    #     steps = 834 * (N/24)^0.51   (949 and 1104 vs published 949 and 1095)
    # so base=840/p=0.51 makes the SIZE LAW the intended binding constraint.
    # `max` 3000 -> 1600: nothing in the field went past 1300 configured / 1095
    # shipped, and FOUR of the six qwen artifacts were deadline-killed by
    # over-scheduling (1150->850, 1300->700, 1300->600).
    "qwen-image": dict(base=840, n_ref=_N_REF, p=0.51, min=300, max=1600),
}

# --- wall-time budget model ------------------------------------------------
# s/step used to cap the size law.  These are now PER-TYPE MEASUREMENTS with an
# explicit pad, not the "guesses to be replaced" they used to be.
#
# Two independent sources of truth:
#   (a) OUR OWN MEASUREMENT on the tournament host.  Our Aug-3 artifact is
#       byte-identified as hotkey 5HLA2QWY via its published forge_run.json:
#       toolkit_start -> toolkit_end = 1041.1 s for 823 krea2 steps
#       = 1.265 s/step.
#   (b) FIELD UPPER BOUNDS.  For any miner whose shipped step count REACHED its
#       configured step count, the run completed inside the budget, so
#       s/step <= (hours*3600 - 478)/shipped_steps is a hard upper bound on the
#       achievable rate on tournament hardware:
#         krea2      <= 1.55   (forge had 2.2  -> 1.42x too slow)
#         ideogram4  <= 2.05   (forge had 3.0  -> 1.46x too slow)
#         z-image    <= 1.56   (forge had 3.0  -> 1.92x too slow)
#         qwen-image <= 4.49   (forge had 4.0  -> the ONLY tight one; unchanged)
#         flux       ~ 1.71    (INFERRED from 5D7iEJm5's 25.7 s/epoch at N=15)
SEC_PER_IT = {
    # flux 2.5 -> 2.0.  Field cadence implies ~1.7 s/step at batch 1; 2.0 is an
    # ~18% pad and restores the size law (870) as the binding constraint.
    # INFERRED, not measured — kohya logs epochs.
    "flux": 2.0,
    # krea2 2.2 -> 1.5.  Above OUR OWN measured 1.265 s/step (a 19% pad) and
    # just under the field's 1.55 bound.  The old 2.2 was justified by
    # `do_differential_guidance`, an unreachable branch that has never executed
    # (see the module docstring).
    "krea2": 1.5,
    # ideogram4 3.0 -> 4.2.  NOTE THIS GOES UP, AND IT IS THE ONE PLACE THE TWO
    # WEEK-6 AUDITS DISAGREE.  The field bound of 2.05 s/step was measured on
    # field configs, which do NOT set `do_cfg`.  OUR config does:
    # `forge.ideogram_release_policy` sets `do_cfg: true` + `cfg_scale: 10.0`,
    # which runs the transformer at BATCH 2 every step (uncond not detached) and
    # adds a second grad-enabled forward through the 8B Qwen3-VL text encoder
    # (PIPELINE-MATERIALIZATION-AUDIT D6).  ~2x the field's per-step cost, so
    # 2.05 * 2 ~= 4.2 is the honest constant FOR OUR PIPELINE.  This costs us
    # nothing on the real shapes: on all three Aug-3 ideogram4 tasks the size law
    # (421/616/589) still binds below the 4.2-based cap (477/674/674), so the
    # materialised steps are identical to what a 2.1 constant would give — while
    # preserving the kill-safety margin that a 2x-optimistic constant would burn.
    "ideogram4": 4.2,
    # z-image 3.0 -> 1.8.  A field miner completed 2000 steps in a 1.0 h task
    # (=> <= 1.56 s/step); 1.8 is a 15% pad.  UNMEASURED ON OUR HOST — we have
    # never run z-image ourselves.  This is the least-verified reduction here.
    "z-image": 1.8,
    # qwen-image UNCHANGED at 4.0.  The only per-type constant the field does
    # NOT contradict (bound 4.49).  Four of six qwen artifacts were
    # deadline-killed; do not make this more optimistic.
    "qwen-image": 4.0,
}
# Fixed reserves.  STARTUP_S covers base-model load + latent/text-embed warmup.
# It is mildly UNDER-modelled today when `cache_latents_to_disk` is on and the
# template's 3-copy `resolution` list is in force, because the VAE pre-encode
# pass then runs over 3N images (PIPELINE-MATERIALIZATION-AUDIT D13).  The
# evaluator-geometry policy in `forge/geometry.py` collapses that to N when it
# is enabled, which removes the under-model rather than papering over it.
STARTUP_S = 300.0
EXPORT_RESERVE_S = 180.0  # mirrors cli._EXPORT_RESERVE_SECONDS
# MARGIN 0.85 -> 0.92.  `size_scaled_steps` computes
# `budget*MARGIN - STARTUP_S - EXPORT_RESERVE_S`, i.e. it took a 15% haircut ON
# TOP OF a 480 s fixed reserve — double-counting.  The champion's recovered
# model is `(budget - 478)` with NO multiplicative margin, and 478 ~= our 480.
# Sensitivity (krea2 @ 1.0 h, SEC 1.5): 0.85->1720, 0.90->1840, 0.92->1888,
# 0.95->1960.  0.92 keeps ~290 s of jitter headroom BEYOND the 480 s reserve.
# Going past 0.95 buys little and erodes the never-forfeit posture.  The
# asymmetry favours scheduling slightly deep: over-scheduling is recoverable
# because `forge/tasks/checkpoints.py` promotes the highest valid periodic save
# to `last.safetensors` on a kill, whereas under-scheduling is not recoverable.
MARGIN = 0.92


def size_scaled_steps(model_type, num_images, hours_to_complete, template_steps):
    """Never raises → falls back to template_steps on any error (INV-1)."""
    try:
        mt = (model_type or "").strip().lower()
        row = STEP_TABLE.get(mt)
        if row is None:
            return int(template_steps)
        n = max(1, int(num_images))
        scaled = row["base"] * (n / row["n_ref"]) ** row["p"]
        scaled = int(round(max(row["min"], min(row["max"], scaled))))

        sit = SEC_PER_IT.get(mt, 3.0)
        budget_s = max(0.0, float(hours_to_complete) * 3600.0)
        train_s = budget_s * MARGIN - STARTUP_S - EXPORT_RESERVE_S
        budget_cap = int(train_s / sit) if train_s > 0 else 1
        return max(1, min(scaled, budget_cap))  # cap may push below `min`
    except Exception:
        try:
            return int(template_steps)
        except Exception:
            return 1000


def projected_wall_s(model_type, steps):
    """Planned wall clock for ``steps`` at the policy rate, incl. fixed reserves.

    This is the quantity the wall-time cap is the inverse of; exposing it makes
    the budget-fit property assertable in tests instead of re-derived by hand.
    Never raises (INV-1).
    """
    try:
        sit = SEC_PER_IT.get((model_type or "").strip().lower(), 3.0)
        return STARTUP_S + max(0, int(steps)) * float(sit) + EXPORT_RESERVE_S
    except Exception:
        return STARTUP_S + EXPORT_RESERVE_S


def first_save_wall_s(model_type, steps, save_every):
    """Planned wall clock at which the FIRST periodic checkpoint lands.

    Kill-safety (INV-2) is "a deadline stop must never find us with nothing
    exported".  Periodic saves are the only mid-run recovery point, so the
    projected time to the first one is the property that has to stay well inside
    the budget even when the box is slower than modelled.  Never raises.
    """
    try:
        sit = SEC_PER_IT.get((model_type or "").strip().lower(), 3.0)
        cadence = max(1, int(save_every))
        first = min(cadence, max(1, int(steps)))
        return STARTUP_S + first * float(sit)
    except Exception:
        return STARTUP_S


def kill_safe_save_every(steps, template_save_every):
    """Budget about four useful periodic candidates plus the exact final.

    Saving is the only mid-run kill-safety, but each tournament save took tens of
    seconds.  A fixed candidate budget is easier to reason about than ``steps//8``:
    target four periodic saves and do not save more often than every 25 steps on
    short jobs.  The first ordinary candidate lands at about 20% of the planned
    run, while the very-short-run branch emits a recovery point near halfway.
    """
    try:
        s = max(1, int(steps))
        template = max(1, int(template_save_every))
        if s < 25:
            # A heavily time-capped run still needs one mid-run recovery point;
            # waiting until its terminal step is not kill-safe.
            return max(1, min(template, max(1, s // 2), s))
        # floor(steps / (4 periodic + 1 final interval)) + 1 keeps the fifth
        # periodic save beyond the terminal step, even when steps is divisible
        # by five.  The 86/367-step tournament shapes produce cadences of 25 and
        # 74 respectively (3 and 4 periodic saves).
        candidate_interval = s // 5 + 1
        return max(
            1,
            min(max(25, candidate_interval), s),
        )
    except Exception:
        try:
            return int(template_save_every)
        except Exception:
            return 100
