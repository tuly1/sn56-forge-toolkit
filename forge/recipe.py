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
    # ideogram4 — was base=140 p=0.50 min=48 max=400, the direct output of the
    # discredited Jul-16 experiment and the least defensible entry in the table.
    # Two CHAMPION WINS give an exact two-point law:
    #     N=14 -> 174 steps (won by 46.06%) ; N=46 -> 341 steps (won by 27.2%)
    #     => p = ln(341/174)/ln(46/14) = 0.5655 , base = 174/(14/24)^0.5655 = 237.5
    # base=240/p=0.57 reproduces both to within 2%.  `max` 400 -> 1600 not
    # because the law reaches it (N=50 gives only 365) but because 400 makes the
    # 1300-step regime that WON task b72da8c6 structurally unreachable; keeping a
    # discredited hard ceiling is the worse error.
    # RESIDUAL UNCERTAINTY — FLAGGED BY THE AUDIT AS TOO THIN TO CONCLUDE FROM:
    # n=5 artifacts, 3 tasks, 2 usable head-to-heads, INTERNALLY CONTRADICTORY.
    # At N=14 shallow (174) crushed deep (1000) by 46%; at N=40 both entrants
    # were deep (1300 vs 1523) and there is NO shallow arm on that task at all.
    # The champion himself swung 4.5x between two same-size style tasks and the
    # deep arm lost.  We ship this only because the incumbent row is a provable
    # artefact of a discredited experiment: this replaces a bad prior with a
    # weak posterior, NOT with a measurement.  ideogram4 is one of the two R1
    # draws, so it is also the highest-variance line here.  Pre-commit to
    # accepting a loss on ideogram4 without concluding anything from it.
    "ideogram4": dict(base=240, n_ref=_N_REF, p=0.57, min=120, max=1600),
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
    # (177/348/321) binds well below the 4.2-based cap (477/674/674), so the
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
