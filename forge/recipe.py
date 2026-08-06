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
#
# WHAT `max` IS FOR, stated once so no row has to claim more than it delivers.
# Observed tournament dataset sizes are N = 9..50 (Jul-20 and Aug-3 harvests).
# The N at which each `max` first binds is:
#   ideogram4  48   ACTIVE inside the observed range
#   krea2      72   backstop only
#   flux       80   backstop only
#   qwen-image 85   backstop only
#   z-image    90   backstop only
# So four of the five are ceilings on EXTRAPOLATION, not policy, and changing
# them cannot move any real tournament shape.  A `max` set where it can never
# bind within 4x the observed range is not a decision, it is decoration — the
# reason ideogram4's short-lived `max=1600` was withdrawn (its law topped out at
# 365 at N=50).  `test_step_table_max_binding_sizes` pins this table, so a row
# that quietly becomes inert fails a test.
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
    # keeping a floor for pathologically small sets.  `min` 100->600 because no
    # 100-step krea2 is competitive in anything we observed — the one 200-step
    # krea2 in the R1 field ranked 13/14.  At N=21 the law returns 1431.5 -> 1432,
    # which is EXACTLY the depth 5FBmn1ax configured and completed on the real R1
    # shape (and 5FjDsFGA independently completed the same 1432 there).
    # Replay on the four real Aug-3 krea2 shapes with SEC_PER_IT["krea2"]=1.35:
    # 1432/1825/1840/1939 vs winners 1000/2012/2012/2012 = 1.05x (was 0.64x); the
    # size law now binds on all four, which is the intent (see SEC_PER_IT).
    # RESIDUAL UNCERTAINTY, largest of any row.  Two things are true at once:
    #  * Depth is NOT sufficient.  The R1 winner shipped only 1000 steps and beat
    #    the 1432/1750/2000 pack with an entirely different recipe
    #    (krea2_eval_sigmas timesteps, TE-LoRA, EMA 0.995, multires noise,
    #    cosine_by_group), and 1432 produced rank 4 for one operator and rank 14
    #    for another.
    #  * Depth is real but WEAK relative to recipe variance.  OLS inside the
    #    9-artifact R1 template family is loss = 0.053134 - 1.677e-6*steps with
    #    residual sd 1.16e-3, so 823 -> 1432 moves predicted loss by 1.0e-3
    #    (0.88 sd) while 1336 -> 1432 moves it by only 1.6e-4 (0.14 sd).
    # Interpolating that fit onto the observed R1 loss ladder puts 1432 at rank
    # ~9 (+-1 sd band 5..11), against the rank 9 we actually took at 823.  So the
    # honest claim is: this removes a self-inflicted depth deficit and puts us at
    # a depth two operators demonstrably completed on this exact shape.  It does
    # NOT on its own predict a top-5 finish; the selection/recipe work does.
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
# s/step used to cap the size law.  These are PER-TYPE MEASUREMENTS with an
# explicit pad, not the "guesses to be replaced" they used to be.
#
# HOW A RATE IS READ OUT OF A PUBLISHED ARTIFACT.  `forge/tasks/aitoolkit.py`
# terminates ai-toolkit when `deadline.remaining() <= _STOP_MARGIN_S`, i.e. at
# `budget - EXPORT_RESERVE_S - STOP_MARGIN_S`, and STARTUP_S of what is left is
# model load + latent/text-embed warmup.  So the optimizer-step window is
#
#     W(h) = h*3600 - EXPORT_RESERVE_S - STOP_MARGIN_S - STARTUP_S = h*3600 - 525
#
# and for any published artifact whose config.yaml and checkpoint ladder are
# both readable:
#     shipped == configured (run COMPLETED) => s/step <= W(h)/shipped   [BOUND]
#     shipped <  configured (run KILLED)    => s/step ~= W(h)/shipped   [POINT]
# The killed form is a direct measurement, accurate to one save interval; the
# completed form is only an upper bound (the miner may have had time to spare).
# `FIELD_DEMONSTRATED_DEPTH` below stores the completed-form evidence in the
# exact integer arithmetic the guard test uses, so no float rounding can flip a
# FINISH into a kill.
#
#   type        evidence used for the constant                          -> SEC
#   krea2       OUR OWN 823 steps in 1041.1 s of toolkit_start..end on
#               the tournament host (5HLA2QWY forge_run.json) = 1.265
#               s/step GROSS OF STARTUP; field BOUND 1.519 (5FBmn1ax and
#               5FjDsFGA each completed 1432 on the real R1 shape)      -> 1.35
#   qwen-image  TWO INDEPENDENT operators (5FW2Eaae, 5FpdSckw) with
#               identical configs both configured 1150 on 7421f056
#               (h=1.25) and were both killed with their last save at
#               850 => 3975/850 = 4.68 s/step, a REPRODUCED POINT
#               measurement.  Champion BOUND on the same type is 4.45.  -> 4.70
#   z-image     BOUND 1.538 (5D2Qee4V completed 2000 in 1.0 h).
#               UNMEASURED on our host.                                 -> 1.80
#   ideogram4   BOUND 2.019 (5FBmn1ax completed 1523 in 1.0 h), DOUBLED
#               because our config sets do_cfg (see below).             -> 4.20
#   flux        BOUND 2.500 (rank-1 5FW2Eaae, 58 kohya epochs x N=15 =
#               870 steps in 0.75 h).  INFERRED — kohya logs epochs.    -> 2.00
#
# TWO qwen artifacts imply far slower rates (5FW2Eaae 6.96 on ff643470,
# 5GU4Xkd3 8.13 on 4782f46f) and are NOT used.  Neither is reproduced, and each
# is contradicted by the SAME operator running much faster on another qwen task
# in the same tournament (5FW2Eaae is the 4.68 datum above), so they cannot be a
# stable property of the hardware or the architecture.  A constant that survived
# 8.13 s/step would plan ~400 qwen steps against a field that shipped 949-1095 —
# the exact failure this recalibration exists to undo.
#
# The krea2 artifacts of 5EACrayt and 5FNLSgh8 are also excluded: both carry 504
# text-encoder tensors (they train a TE-LoRA, which we never do), and 5FNLSgh8's
# kill implies 2.72 s/step — a config difference, not a hardware rate.
SEC_PER_IT = {
    # flux 2.5 -> 2.0.  Field cadence implies ~1.7 s/step at batch 1; 2.0 is an
    # ~18% pad and restores the size law (870) as the binding constraint.
    # INFERRED, not measured — kohya logs epochs.
    "flux": 2.0,
    # krea2 2.2 -> 1.35 (was 1.5 at c424362).  The old 2.2 was justified by
    # `do_differential_guidance`, an unreachable branch that has never executed
    # (see the module docstring).
    #
    # WHY 1.35 AND NOT 1.5 OR 1.3.  The three candidates were replayed over all
    # four real Aug-3 krea2 shapes.  1.35 and 1.30 are IDENTICAL in output —
    # both make the SIZE LAW the binding constraint everywhere, which is the
    # stated intent of the krea2 row — while 1.5 is the only one of the three
    # that still lets the clock truncate it:
    #     shape                     law   SEC1.5   SEC1.35   SEC1.30
    #     41025fb5 N=21 h=0.75     1432     1336      1432      1432
    #     3e0fdcde N=42 h=1.00     1825     1825      1825      1825
    #     db9f7244 N=43 h=1.00     1840     1840      1840      1840
    #     f6725c2b N=50 h=1.00     1939     1888      1939      1939
    # The law stops being truncated for any SEC <= 1.399 (0.75 h) and <= 1.460
    # (1.0 h); 1.35 sits 3.6% inside that threshold, and 1.30 buys nothing beyond
    # it while making `projected_wall_s` 4% more optimistic.  So 1.35.
    #
    # SAFETY.  1.35 is a 6.7% pad over our own measured 1.265 s/step, and that
    # 1.265 is GROSS OF STARTUP (it is toolkit_start -> toolkit_end for 823
    # steps), so charging STARTUP_S=300 on top of it is pure additional cushion:
    # net of a 300 s startup our measured rate is (1041.1-300)/823 = 0.90 s/step.
    # At the field's own tightest krea2 bound (1.519 s/step, from two independent
    # operators each completing 1432 steps on the real R1 shape) every planned
    # krea2 depth here still completes — 1432 exactly fills the 0.75 h window by
    # construction, and 1825/1840/1939 sit 4-10% inside the 2024-step 1.0 h
    # window.  See `FIELD_DEMONSTRATED_DEPTH` and the guard test.
    #
    # DOWNSIDE IF WE ARE SLOWER THAN THAT.  A deadline stop DEGRADES depth, it
    # does not forfeit: `_run_toolkit -> _terminate -> _finalize` promotes the
    # newest valid periodic save.  On the R1 shape `kill_safe_save_every(1432)`
    # is 287, so a stop anywhere in (1148, 1432) ships 1148 — still 1.4x the 823
    # we actually shipped on Aug-3, and the fit below puts 1148 at rank ~10, the
    # same band the 1336 that SEC=1.5 would have produced.  The bet is bounded.
    "krea2": 1.35,
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
    # qwen-image 4.0 -> 4.7.  4.0 was carried forward as "the only constant the
    # field does not contradict", but that was read off the champion's COMPLETED
    # run (an upper BOUND of 4.45), and a bound is not a rate.  The field also
    # contains a reproduced POINT measurement, and it is slower than 4.0: on
    # 7421f056 (h=1.25) 5FW2Eaae and 5FpdSckw both configured 1150, both ran the
    # same config, and both were killed with their last periodic save at 850 =>
    # 3975/850 = 4.68 s/step.  Two independent operators agreeing exactly is the
    # strongest rate datum in the qwen set, so 4.7 is the honest constant.
    # This matters beyond the cap: `projected_wall_s` and `first_save_wall_s` —
    # the kill-safety projections — are only meaningful if this is a real rate.
    # At 4.0 the projection claimed 676 s of slack on 7421f056 where the true
    # slack is ~91 s.
    # The compensating change is MARGIN_BY_TYPE["qwen-image"]; the two constants
    # are calibrated JOINTLY and the guard test pins the composite, not either
    # one alone.  Four of six qwen artifacts were deadline-killed — this type has
    # the least headroom in the table and is UNMEASURED on our own host.
    "qwen-image": 4.7,
}
# The extra cushion `forge/tasks/aitoolkit.py` gates termination on, on top of
# the export reserve: training is stopped at `hard_stop - (180 + 45)`.  Mirrors
# `forge.tasks.holdout.boundary_margin_s()`; imported there rather than here to
# keep this module import-free, and pinned equal in the tests.
STOP_MARGIN_S = 45.0
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
# Sensitivity (krea2 @ 1.0 h, SEC 1.35): 0.85->1700, 0.90->1844, 0.92->1901,
# 0.95->1988.  0.92 keeps ~290 s of jitter headroom BEYOND the 480 s reserve.
# The asymmetry favours scheduling slightly deep: over-scheduling is recoverable
# because `forge/tasks/checkpoints.py` promotes the highest valid periodic save
# to `last.safetensors` on a kill, whereas under-scheduling is not recoverable.
#
# THIS IS THE DEFAULT ONLY.  Applying 0.92 GLOBALLY was a regression: it raised
# the qwen cap 1027 -> 1122 (+9.3%) without touching SEC_PER_IT["qwen-image"],
# and qwen is the one type with no clock headroom.  Replayed at the field's own
# reproduced qwen rate (4.68 s/step) the global 0.92 was killed on two of the
# three real qwen shapes: 7421f056 planned 909, stopped at 850, shipped its
# 728-step save; ff643470 planned 1104, stopped at 1042, shipped 884.  A margin
# is a per-type dial because the headroom it is spending is per-type.
MARGIN = 0.92
# Per-type override of MARGIN.  READ THIS WITH SEC_PER_IT — the pair is what is
# calibrated, and the invariant they exist to satisfy is the one the guard test
# asserts: for every type x every real Aug-3 shape, the planned step count must
# still complete at that type's own field-observed rate.
#
#   type        SEC_PER_IT vs its evidence      MARGIN  why
#   flux        2.00 vs 1.71 inferred (+17%)      0.92  law binds; clock inert
#   krea2       1.35 vs 1.265 measured (+7%)      0.92  law binds on all 4 shapes
#   ideogram4   4.20 vs 2.019x2 field (+4%)       0.92  law binds on all 3 shapes
#   z-image     1.80 vs 1.538 field (+17%)        0.92  law binds on both shapes
#   qwen-image  4.70 vs 4.676 reproduced (+0.5%)  0.98  CLOCK BINDS — see below
#
# qwen is the only row where the clock is the active constraint, so it is the
# only row where MARGIN does real work, and its SEC carries essentially no pad
# (the rate is the measurement).  With an honest rate the arithmetic identity
# `budget*M - 480 == budget - 525` gives M = 1 - 45/budget ~= 0.99 for "plan
# exactly to the terminate trigger", so 0.98 IS the cushion, not an absence of
# one: it plans 836 steps into a window the field demonstrated holds 850, and
# 1023 into one it demonstrated holds 1042.  Setting qwen to 0.92 instead would
# plan 778/954/954 — safe, but 7% below what the same evidence says fits, on the
# type where the Aug-3 winners shipped 850/949/1095.
MARGIN_BY_TYPE = {
    "flux": 0.92,
    "krea2": 0.92,
    "ideogram4": 0.92,
    "z-image": 0.92,
    "qwen-image": 0.98,
}

# Deepest run the Aug-3 field is KNOWN to have completed for each type, with the
# budget it completed in: `(hours, steps, provenance)`.  Because the run finished,
# `steps` optimizer steps demonstrably fit inside `W(hours) = hours*3600 - 525`.
# Scaling that linearly is the strictest defensible statement the field supports
# about achievable throughput, and it is the acceptance criterion for every
# constant above: we never plan more steps than the field demonstrated, in the
# same window, at the same architecture.  Kept as integers so the guard test is
# exact integer arithmetic with no float knife-edge.
#
# qwen-image is deliberately NOT the deepest completed run (5FBmn1ax's 1095 in
# 1.5 h).  It is the reproduced KILL at 850 in 1.25 h, which is slower and is a
# measurement rather than a bound — see the SEC_PER_IT header.
FIELD_DEMONSTRATED_DEPTH = {
    "flux": (0.75, 870, "5FW2Eaae 241cda6c rank 1, 58 kohya epochs x N=15"),
    "krea2": (0.75, 1432, "5FBmn1ax + 5FjDsFGA 41025fb5, cfg 1432 shipped 1432"),
    "ideogram4": (1.0, 761, "5FBmn1ax b72da8c6 cfg 1523 shipped 1523, HALVED "
                            "for our do_cfg batch-2 step (2.019 -> 4.038 s/step)"),
    "z-image": (1.0, 2000, "5D2Qee4V b290d171, cfg 2000 shipped 2000"),
    "qwen-image": (1.25, 850, "5FW2Eaae + 5FpdSckw 7421f056, cfg 1150 both "
                              "killed with their last save at 850"),
}


def margin_for(model_type):
    """Per-type clock margin, falling back to the global default.  Never raises."""
    try:
        return float(MARGIN_BY_TYPE.get((model_type or "").strip().lower(), MARGIN))
    except Exception:
        return MARGIN


def training_deadline_s(hours_to_complete):
    """Seconds from container start to the moment `_run_toolkit` terminates.

    `aitoolkit._run_toolkit` stops training when `deadline.remaining()` (already
    net of the export reserve) falls to `_STOP_MARGIN_S`.  This is the real wall
    the plan has to fit inside — `MARGIN` is a policy dial, this is physics.
    Never raises (INV-1).
    """
    try:
        budget = max(0.0, float(hours_to_complete) * 3600.0)
        return max(0.0, budget - EXPORT_RESERVE_S - STOP_MARGIN_S)
    except Exception:
        return 0.0


def completed_steps_at_rate(hours_to_complete, sec_per_it):
    """Optimizer steps that land before the terminate trigger, at ``sec_per_it``.

    The inverse of the budget model, expressed against the REAL deadline rather
    than `budget*MARGIN`, so "does this plan actually finish at rate R?" is one
    call instead of a hand-derivation.  Never raises (INV-1).
    """
    try:
        rate = float(sec_per_it)
        if rate <= 0:
            return 0
        window = training_deadline_s(hours_to_complete) - STARTUP_S
        return max(0, int(window / rate))
    except Exception:
        return 0


def field_demonstrated_steps(model_type, hours_to_complete):
    """Steps the field demonstrably completed, scaled to ``hours_to_complete``.

    Exact integer arithmetic on `FIELD_DEMONSTRATED_DEPTH`: the reference run
    fitted `steps` into `W(ref_hours)`, so the same throughput fits
    `steps * W(hours) // W(ref_hours)` into this budget.  Returns None when the
    type has no field evidence.  Never raises (INV-1).
    """
    try:
        entry = FIELD_DEMONSTRATED_DEPTH.get((model_type or "").strip().lower())
        if entry is None:
            return None
        ref_hours, ref_steps = entry[0], entry[1]
        ref_window = int(round(training_deadline_s(ref_hours) - STARTUP_S))
        window = int(round(training_deadline_s(hours_to_complete) - STARTUP_S))
        if ref_window <= 0 or window <= 0:
            return 0
        return window * int(ref_steps) // ref_window
    except Exception:
        return None


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
        train_s = budget_s * margin_for(mt) - STARTUP_S - EXPORT_RESERVE_S
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
