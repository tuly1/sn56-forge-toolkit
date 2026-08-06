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

Two premises that used to live in this file have been deleted.  ONE IS
FALSIFIED; the other is UNSUPPORTED and the data lean the other way.  The
distinction is deliberate — see the correctness sweep in
FIELD-DEPTH-LAW-AUDIT "Known limitations and unfixed exposure".

  1. UNSUPPORTED (not falsified): "over-training is the #1 liability" / "deep
     training never helped".  That came from a Jul-16 probe over 8..128 steps
     on 12 photos — the wrong regime entirely.  What replaces it is weaker than
     this file used to claim:
       * FIVE published per-checkpoint reconstruction curves
         (`.krea2/.zimage/.flux_checkpoint_evaluations.json`, already split into
         the validator's own text/no-text terms) — but from only TWO operators
         (5EACrayt x3, 5FNLSgh8 x2), so they are not five independent samples.
         FOUR of the five put their minimum at the deepest checkpoint
         evaluated.  The fifth DOES turn over, and it is on this exact R1 krea2
         shape: 5FNLSgh8 scored 800 -> 0.032265 and its 1000-step export ->
         0.032689 (+1.3%) and shipped the 800.  Two of the four also have
         interior bumps (flux ep20-30, z-image @750).
       * On the only open field of the tournament (R1 krea2, 14 ranked
         artifacts), Spearman(steps, test_loss) inside the template recipe
         family (n=9) is -0.605, exact permutation p = 0.045 one-sided / 0.091
         two-sided.  The three excluded artifacts (504 TE tensors) are ranks 1,
         2 and 13; keeping rank 13 gives rho = -0.646 (p = 0.024), so the
         exclusion is CONSERVATIVE, not cherry-picked.
     We shipped 823 steps (39/img) and placed 9/15; the template pack's top four
     ran 1432-2000.  Read this as "the shallow premise is unsupported and the
     data lean deeper", not as proof that deeper is better.
  2. FALSIFIED: "krea2's do_differential_guidance adds a second guidance forward
     per step".
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
    # alone.  NOTE, AND THIS IS BIGGER THAN "a small clock fix": on a
    # standalone-checkpoint FLUX base the validator runs
    # `ops/docker/standalone-image-trainer.dockerfile`, which sets
    # `FORGE_FLUX_BACKEND=kohya`; `dispatch.for_model_type` then routes to
    # `forge/tasks/flux_kohya.py` and depth comes from
    # `flux_kohya_config.budgeted_train_steps` (75 durable steps / 1576.6 s x
    # 0.80 headroom, ceiling `MAX_TRAIN_STEPS = 250`), NOT from this row:
    # 94 steps at 0.75 h, 128 at 1.0 h, 196 at 1.5 h, at train_batch_size 4 x
    # grad_accum 2.  Aug-3 task db5fefc5 (dataautogpt3/FLUX-MonochromeManga)
    # takes that path; 241cda6c (rayonlabs/FLUX.1-dev) does not.  So HALF the
    # observed flux draw never consults STEP_TABLE at all — an open decision
    # (FIELD-DEPTH-LAW-AUDIT §6.5 + "Known limitations").
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
    #    9-artifact R1 template family is loss = 0.053134 - 1.677e-6*steps
    #    (reproduced exactly), so 823 -> 1432 moves predicted loss by 1.0e-3 and
    #    1336 -> 1432 by only 1.6e-4.  Expressed in residual sd, STATE THE
    #    CONVENTION because it changes the headline: the quoted sd 1.16e-3
    #    divides by n, giving 0.88 sd and 0.14 sd; the usual OLS residual sd
    #    (n-2 df) is 1.32e-3, giving 0.78 sd and 0.12 sd.  The n-divisor version
    #    makes the depth effect look ~13% more significant than it is.
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
    #     final iterate, which retains only 43.4% of the path at that depth
    #     (1/0.434 = 2.3x) under OUR cosine 2.5e-5 -> 2.5e-6 schedule
    #                                                      = 0.00106  (65.6x less)
    #     [INFERRED, not measured: the 43.4% is the EMA weight
    #      (1-d)*sum_t d^(T-t)*theta_t / theta_T computed on the cumulative-lr
    #      path with theta_0 = 0, i.e. it assumes displacement/step ~ lr and a
    #      stable update direction.  Under a CONSTANT lr the same formula gives
    #      33.9%.  Neither is a measurement of our adapter.]
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
    #       ideogram4 sample we have: we ran the shortest run in the field
    #       (85 steps) and placed 4/16; the deep cluster was 722-1000+ and the
    #       recorded finding is that the img2img metric "did NOT punish
    #       overtraining" (SN56-WEEK3-POSTMORTEM.md §6a).
    #       CORRECTION (week-6 sweep): this row used to end "...the bracket
    #       winner 1200".  NO CITED SOURCE SAYS THAT, and our own
    #       SN56-WEEK4-INDEPENDENT-REVIEW-2026-07-22.md §2 says the opposite:
    #       on 3cfa1578 the R1 WINNER configured 378 steps at lr 2.5e-5 with
    #       EMA + cosine + TE training, while the lr-4e-4 arm that configured
    #       1200/1650 ranked 8th.  The "1200" appears to have been imported
    #       from the Jul-27 KREA2 R1 table (task 73013636), a different type and
    #       a different tournament.
    #       This matters more than a citation slip: that 378-step winner is the
    #       ONLY field artifact anywhere in OUR OWN lr/EMA/scheduler family, and
    #       it sits BELOW the 421/589/616 this row now ships.  Our law at N=9
    #       returns 365, which is close to it; the divergence is at large N.
    # Net across two tournaments: ideogram4 depth is FLAT and WIDE — 85 to 1523
    # all placed 1st or 2nd somewhere — with NO consistent direction once the
    # Jul-20 winner is read correctly.  There is NO size law to fit: N=9 -> 378,
    # N=14 -> 174, N=40 -> 1250, N=46 -> 341 is uncorrelated with size.
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
    #  (a) EMA ATTENUATION (this was previously written as an "EMA floor" and
    #      the mechanism was stated WRONG; the direction survives, the framing
    #      does not).  ai-toolkit seeds the EMA shadow from the LoRA at init,
    #      constructs it with `use_num_updates=False` so there is no warm-up or
    #      bias correction, and `save()` ALWAYS exports the EMA
    #      (`BaseSDTrainProcess.py:491,495-497` save; `:769-781` setup_ema;
    #      `:2031` call site; `toolkit/ema.py:43-72` __init__, `:100-152`
    #      update, `:336-341` eval).
    #      WHY THE OLD WORDING WAS FALSE: `lora_up` is ZERO-INITIALISED
    #      (`toolkit/lora_special.py:122`), so the adapter's effect at init is
    #      exactly zero.  `0.995^T` is NOT "the untrained-initialisation share
    #      of the export" — there is no untrained signal to retain.  It is the
    #      EMA's memory horizon.
    #      WHAT IT ACTUALLY COSTS: the export is an attenuated copy of the
    #      trained delta.  Under our cosine 2.5e-5 -> 2.5e-6 schedule the
    #      exported adapter is ~43% of the final iterate at 177 steps, ~72% at
    #      421, ~82% at 589, ~83% at 616 (INFERRED — see the model note above;
    #      under a constant lr the same formula gives 34% / 58% / 68% / 69%).
    #      So the penalty is real and depth still fixes it, but the old numbers
    #      ("41% untrained at 177, 4.6% at 616") described a quantity that does
    #      not exist.  Depth is the ONLY lever on this that does not require
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
    # miscalibration; it is 1.3x the withdrawn law at N=50 and 2.1x at N=21, and
    # it keeps four periodic candidates.  Every 1.0 h shape up to N=50 stays
    # law-bound.
    # Because the law binds on the real shapes and the clock does not, this row
    # is invariant to MARGIN and to any SEC_PER_IT revision over the range the
    # tests grid.  HOW MUCH ERROR IT ABSORBS — corrected, because the previous
    # figure ("21%, breaks only above 5.06 s/step = 2.47x the field bound")
    # IGNORED MARGIN and so overstated the cushion:
    #   PLANNING truncation (the constant is wrong -> we plan fewer steps).
    #   `size_scaled_steps` caps at (budget*0.92 - 480)/SEC, so the law survives
    #   only while SEC stays below
    #       84be9fcd 616 steps @1.0h : 4.597  (+9.5% over 4.2, 2.28x the 2.019
    #                                          field no-do_cfg bound)  <-- tightest
    #       1365fa1c 421 steps @0.75h: 4.760  (+13.3%)
    #       b72da8c6 589 steps @1.0h : 4.808  (+14.5%)
    #   RUNTIME kill (the box is actually slower -> the deadline stops us).  The
    #   terminate gate is budget - 180 - 45, less STARTUP_S, so the real rate at
    #   which the tightest shape stops short is 3075/616 = 4.992 (+18.9%).
    # The old 5.06 was neither of these (it dropped STOP_MARGIN_S as well).
    # A truncation DEGRADES rather than forfeits (`forge/tasks/aitoolkit.py`
    # _run_toolkit -> _terminate -> _finalize promotes the newest periodic save,
    # and there are four of them).
    # p 0.57 -> 0.32 because the field shows no size signal at all; this mirrors
    # krea2's deliberately flat 0.35 rather than a 2-point fit.  min 120 -> 350
    # binds only below N~8, under the smallest ideogram4 dataset ever observed
    # (N=9, Jul-20).  max 1600 -> 620: 1600 was INERT (the old law topped out at
    # 365 at N=50, so it could never bind within 4x), whereas 620 first CHANGES
    # the output at N = 48 — not 47, as this comment used to say: at N=47 the
    # law returns 619.975, which rounds to 620 with or without the cap, so the
    # cap is a no-op there.  BE HONEST ABOUT HOW LITTLE THIS DOES: 620 binds
    # only at N = 48, 49, 50 (the top of the observed range) and changes depth
    # by 4 / 8 / 12 steps respectively.  It is barely more than decoration; it
    # is kept because it caps extrapolation of a recipe we have never run past
    # ~200 steps in a tournament, not because it moves any real shape.
    # (`test_step_table_max_binding_sizes` pins the crossover at 48.)
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
    #     steps = 834 * (N/24)^0.51   -> 950 and 1096 vs his published 949 and
    #                                    1095 (this line previously read "949 and
    #                                    1104"; 1104 is what OUR base=840 returns
    #                                    at N=41, not what his 834 returns)
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
#   krea2       OUR OWN 823 steps in 1036.4 s of toolkit_start..end on
#               the tournament host (5HLA2QWY forge_run.json: toolkit_start
#               t=4.7 -> toolkit_end t=1041.1, both RELATIVE TO scope start,
#               so the ELAPSED window is 1036.4 s) = 1.259 s/step GROSS OF
#               STARTUP.  This file previously read "823 steps in 1041.1 s
#               = 1.265", which used an end TIMESTAMP as a duration; the
#               error was 0.5% and in the conservative direction.
#               Field BOUND 1.519 (5FBmn1ax and 5FjDsFGA each completed
#               1432 on the real R1 shape)                              -> 1.35
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
# TWO qwen artifacts imply far slower rates (5FW2Eaae 6.96 s/step on ff643470,
# 5GU4Xkd3 8.13 on 4782f46f) and are NOT used to SET the constant.  THE HONEST
# STATEMENT OF WHY, because the previous one was a non-sequitur:
#   * NOT a measurement-quality argument.  All three qwen kills — the 850 that
#     gives 4.676, and both of these — ran `save_every: 50`, so all three locate
#     the kill inside one 50-step interval.  Same precision.
#   * NOT "each is contradicted by the SAME operator on another qwen task".
#     That holds for exactly ONE of them.  5FW2Eaae entered two qwen tasks with
#     configs identical on every recorded field except the step target (rank
#     128/128, lr 1e-4, adamw8bit, res [512,768,1024], batch 1, EMA 0.995,
#     timestep weighted, save_every 50) and implies 4.676 s/step on 7421f056 and
#     6.964 on ff643470.  5GU4Xkd3 entered only ONE qwen task in this
#     tournament, so nothing of his contradicts his 8.13.
#   * What that pair actually demonstrates is ~1.5x PER-TASK THROUGHPUT VARIANCE
#     for one operator on one recipe (6.964/4.676 = 1.49x).  That is a RISK TO
#     PRICE, not a disqualification.  4.7 is used because the two-operator 4.676
#     is the best point estimate — not because the slow observations are refuted.
# EXPOSURE IF THE SLOW TAIL IS REAL (replayed against the real terminate gate):
#     ff643470 h=1.5  plan 1023 save_every 205: @6.96 -> 700 steps, SHIPS 615
#                                               @8.13 -> 600 steps, SHIPS 410
#     7421f056 h=1.25 plan  836 save_every 168: @6.96 -> 570 steps, SHIPS 504
#     4782f46f h=1.5  plan  957 save_every 192: @6.96 -> 700 steps, SHIPS 576
# i.e. 40-60% of planned depth, DEGRADED not forfeited.  A constant built to
# survive 8.13 would instead plan ~400 qwen steps against a field that shipped
# 949-1095 — the failure this recalibration exists to undo.  The trade is stated,
# not hidden.
#
# The krea2 artifacts of 5EACrayt and 5FNLSgh8 are also excluded: both carry 504
# text-encoder tensors (they train a TE-LoRA, which we never do).  CORRECTION to
# an earlier claim in this file: 5FNLSgh8 was NOT time-killed at 800 and does not
# imply "2.72 s/step".  Its own `.krea2_checkpoint_evaluations.json` scores
# 200/400/600/800 AND a distinct 1000-step `last` export, so the run COMPLETED
# its configured 1000 and its selector promoted the 800 (audit §4.5 hash match).
# It yields NO rate datum at all.
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
    # SAFETY.  1.35 is a 7.2% pad over our own measured 1.259 s/step, and that
    # 1.259 is GROSS OF STARTUP (it is toolkit_start -> toolkit_end for 823
    # steps), so charging STARTUP_S=300 on top of it is pure additional cushion:
    # net of a 300 s startup our measured rate is (1036.4-300)/823 = 0.895
    # s/step.
    # At the field's own tightest krea2 bound (1.519 s/step, from two independent
    # operators each completing 1432 steps on the real R1 shape) every planned
    # krea2 depth here still completes — but SAY HOW TIGHT THAT IS on R1: 1432
    # fills the 0.75 h window EXACTLY by construction (2175/1432 = 1.51885), so
    # the R1 plan truncates at ANY rate above 1.51885 s/step.  That is 20.6% over
    # our own measured 1.259 — a real cushion, but it is a single knife-edge, and
    # the guard test only passes because `field_demonstrated_steps` is exact
    # integer arithmetic.  The 1.0 h shapes are not tight: 1825/1840/1939 sit
    # 4-10% inside the 2024-step window.  See `FIELD_DEMONSTRATED_DEPTH`.
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
    # For the shipped 836-step plan on 7421f056, `budget - projected_wall_s` is
    # 676 s at a 4.0 constant and 91 s at 4.7.  (Neither is "the true slack":
    # against the terminate gate at the field's measured 4.676 the plan lands
    # 14 steps / ~66 s early.  The point is that a constant chosen for comfort
    # makes the projection lie by ~10 minutes.)
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
# Sensitivity — the CLOCK CAP for krea2 @ 1.0 h at SEC 1.35, i.e.
# int((3600*M - 480)/1.35):  0.85->1911, 0.90->2044, 0.92->2097, 0.95->2177,
# 0.98->2257.  (This line used to read 1700/1844/1901/1988; those numbers do not
# reproduce from any shipped constant — the slope implied by the series is
# 3600/SEC = 2892, i.e. SEC ~= 1.245, against a ~944 s fixed reserve.  Corrected
# 2026-08-06 by the week-6 correctness sweep.)  The krea2 law returns at most
# 1939 in the observed size range, so at every margin >= 0.85 the LAW binds and
# this cap is inert for krea2 — which is the intent, and is why the wrong
# numbers changed no decision.
# 0.92 keeps ~290 s of jitter headroom BEYOND the 480 s reserve.
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
