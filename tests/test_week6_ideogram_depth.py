"""Week-6 regression guard for the ideogram4 depth law specifically.

WHY THIS FILE IS SEPARATE FROM test_week6_depth_geometry.py
===========================================================
Every other row in ``recipe.STEP_TABLE`` is calibrated by REPRODUCING THE FIELD
WINNERS' STEP COUNTS, and ``test_shipped_depth_now_tracks_the_field_winners``
asserts exactly that.  ideogram4 cannot be calibrated that way and the reason is
mechanical, not a matter of taste:

  * Every ideogram4 config published in the Aug-3 field runs ``lr: 0.0004``
    CONSTANT (OBSERVED — 5FBmn1ax's three ``checkpoints/config.yaml``).
  * ``forge.ideogram_release_policy`` runs OUR ideogram4 at ``lr 2.5e-5``
    cosine-decayed to ``eta_min 2.5e-6`` — a 16x lower peak, ~29x lower mean.
    (``unet_lr``/``text_encoder_lr`` are inert; the LoRA branch passes
    ``train_config.lr`` for every group — PIPELINE-MATERIALIZATION-AUDIT §2.1.)

Under Adam the per-coordinate displacement per step is ~lr, so matching a step
count across a 16x lr gap matches nothing.  The two constraints that DO bind on
our own pipeline are asserted here instead:

  (a) EMA attenuation — ``save()`` always exports the EMA shadow, which is
      seeded at the zero-effect init and never bias-corrected, so the export is
      an ATTENUATED copy of the trained delta and the attenuation shrinks with
      depth.  (This bullet used to say ``0.995**steps`` of the export "is
      untrained init".  That is FALSE: ``lora_up`` is zero-initialised
      (``toolkit/lora_special.py:122``), so the init contributes no adapter
      effect at all.  See the per-test docstring below.)
  (b) the do_cfg clock ceiling — ``do_cfg: true`` runs the transformer at batch
      2, so our per-step cost is ~2x the field's and the reachable depth is
      correspondingly halved.

Evidence:
  ops/experiments/week6/FIELD-DEPTH-LAW-AUDIT.md            §6.2
  ops/experiments/week6/PIPELINE-MATERIALIZATION-AUDIT.md   §4.3, D6
  SN56-project/SN56-WEEK3-POSTMORTEM.md                     §6a (Jul-20, 16 miners)
  SN56-project/SN56-WEEK4-INDEPENDENT-REVIEW-2026-07-22.md  §2 (the Jul-20 R1
      ideogram4 WINNER configured 378 steps at lr 2.5e-5 + EMA + cosine + TE —
      the only field artifact in our own recipe family, and it sits BELOW the
      421/589/616 this law ships)
"""

from __future__ import annotations

import math

import pytest

from forge import ideogram_release_policy, recipe


# task, family, n_pairs, hours, rank-1 miner's shipped depth, what we now ship
#
# "SHIPPED" IS RESOLVED BY LFS CONTENT HASH, NOT BY FILENAME.  Several operators
# publish a `checkpoints/last.safetensors` that is byte-identical to a numbered
# rung of their own ladder — they trained deep and shipped a shallower selection.
# Taking the deepest published checkpoint as "shipped" overstates the field, and
# this table did that on two of its three rows before 2026-08-06.
REAL_IDEOGRAM_TASKS = [
    # 1365fa1c: the ONLY clean shallow-vs-deep head-to-head in ideogram4.
    # 174 (rank 1) beat 800 (rank 2) by 46.1% — but the deep arm stripped its
    # metadata and published no config, so it is not a depth-only comparison.
    # CORRECTED: the deep arm is 800, not ">900".  5GU4Xkd3 trained to 900
    # (metadata on last_000000900) and its `last.safetensors` has the same LFS
    # oid as its `last_000000800.safetensors`.
    # n_train 12 (N=14): OBSERVED in the zip central directory.  421 -> 414.
    ("1365fa1c", "product", 12, 0.75, 174, 414),
    # b72da8c6: BOTH arms deep (1100 rank 1 vs 1523 rank 2, +4.4%).  No shallow
    # arm exists on this task, so it cannot show that a shallow run would lose —
    # only that 1100 ~= 1523.
    # CORRECTED 1300 -> 1100: the rank-1 5GU4Xkd3 trained to 1200 and its
    # `last.safetensors` has the same LFS oid as `last_000001100.safetensors`.
    # The rank-2 5FBmn1ax is exact (config `steps: 1523`, metadata step 1523)
    # and is the ONLY ideogram4 config in the field with no `ema_config` block —
    # the same operator ran EMA 0.99 on the two tasks he WON, so depth and EMA
    # are confounded within his own three runs.
    # n_train 36 (N=40).  589 is unchanged: base 500 at N=40 and base 517 at
    # n_train 36 both land on 589, the one coincidence in the table.
    ("b72da8c6", "style", 36, 1.0, 1100, 589),
    # 84be9fcd: 341 (rank 1) vs an opponent who published ONE FILE, no
    # __metadata__ and no checkpoint ladder.  ZERO depth information.
    # n_train 41 (N=46).  616 -> 614.
    ("84be9fcd", "style", 41, 1.0, 341, 614),
]
IDS = [row[0] for row in REAL_IDEOGRAM_TASKS]

# The Jul-20 R1 ideogram4 task (3cfa1578), N=11 h=0.75, SIXTEEN miners — the
# largest ideogram4 sample in existence and the only one at an R1 shape.
# `image_text_pairs` in api.gradients.io/auditing/tasks/3cfa1578-... is 11; the
# "N=9" that appeared in earlier week-6 notes is wrong.
JUL20_R1_N_PAIRS = 11
# ...but the law is handed 11 - ceil(1.1) = 9.  BOTH numbers are right, for
# different quantities, and the "the N=9 in earlier week-6 notes is wrong"
# correction above was itself the error.  PROVED BY OUR OWN RUN: on Jul-20 the
# row was base=140 p=0.50 with the clock inert, and 5HLA2QWY shipped 86 steps.
# 140*(9/24)**0.5 = 85.7 -> 86; 140*(11/24)**0.5 = 94.8 -> 95.
JUL20_R1_N_TRAIN = 9
JUL20_R1_HOURS = 0.75
# 5FNLSgh8, rank 1 of 16, test_loss 0.0502341.  Shipped 378 steps at lr 2.5e-5
# with `cosine_by_group` + `use_ema: true` / `ema_decay: 0.995` + TE-LoRA on a
# rank-32 network.  THE ONLY FIELD ARTIFACT IN OUR OWN lr/EMA/SCHEDULER FAMILY,
# across both tournaments.
JUL20_IN_FAMILY_WINNER_STEPS = 378
# 5EACrayt, rank 13 of 16, test_loss 0.0965093 — the SAME 378 steps, the same
# schedule, the same EMA, the same TE-LoRA, the same rank, at lr 5e-5.
JUL20_MATCHED_DEPTH_CONTROL_STEPS = 378

# The EMA decay the activated release policy binds.  This is now GENUINELY read
# out of the policy — the previous version of this line claimed to do that in a
# comment while hardcoding 0.995, so the two could (and did) drift apart.
EMA_DECAY = ideogram_release_policy._EXPECTED_RECIPE["train"]["ema_config"][
    "ema_decay"
]
# The value that decay is expected to hold, pinned separately so a policy edit
# still trips this file rather than being silently inherited through the line
# above.
#
# WEEK-6 INTEGRATION (2026-08-07): 0.99 -> 0.995.  The Week-6 EMA-horizon
# amendment was VACATED by unit 3 and this pin follows it back to the value the
# in-family field anchor ran (5FNLSgh8, Jul-20 3cfa1578, rank 1 of 16, published
# config sha256 bf852a1a...7201a954c: `steps: 378`, `ema_decay: 0.995`).  The
# integrator re-derived the decision rather than inheriting it — see
# `test_ideogram4_export_stays_inside_the_measured_strength_band` below, whose
# assertion had to be rewritten because the old one was expressed in a quantity
# that rejects the anchor itself.
EXPECTED_EMA_DECAY = 0.995
# The lr the SAME task's rank-13-of-16 entrant ran at the SAME depth (5EACrayt,
# config sha256 eeb91495...03128f30 — byte-identical to the anchor except lr
# 2.5e-5 -> 5.0e-5 and the matching min_lr keys).  test_loss 0.0965093 against
# the anchor's 0.0502341: +92.1% for 2x the exported adapter strength, at
# identical depth, in our own recipe family.  This is the ONLY controlled
# adapter-strength measurement in the entire ideogram4 record and it is what
# the strength band below is anchored on.
MATCHED_PAIR_OVERSTRENGTH_LR = 5.0e-5
MATCHED_PAIR_OVERSTRENGTH_RATIO = 2.0  # = 5.0e-5 / 2.5e-5 at identical depth

# Our shipped schedule, needed by the attenuation model below.
LR0 = 2.5e-5
ETA_MIN = 2.5e-6


def _cap(hours, margin, sec_per_it):
    train_s = hours * 3600.0 * margin - recipe.STARTUP_S - recipe.EXPORT_RESERVE_S
    return int(train_s / sec_per_it) if train_s > 0 else 1


def exported_fraction(steps, decay, lr0=LR0, eta_min=ETA_MIN, cosine=True):
    """Share of the trained LoRA delta that ``save()`` actually exports.

    MODEL, NOT MEASUREMENT.  ``lora_up`` (B) is zero-initialised, so
    ``B_k = sum_{j<=k} lr_j`` under Adam's ~lr per-coordinate displacement, and
    the exported shadow obeys ``s_k = d*s_{k-1} + (1-d)*B_k`` with ``s_0 = 0``.
    ``lora_down`` starts non-zero so ``A_ema ~= A_final`` to leading order, which
    makes the exported adapter ``f = s_T/B_T`` times the trained one.  ASSUMPTION:
    the update direction is stable enough that the shadow is a scaled endpoint
    rather than a directional average — exact early, degrades late.

    Anchored by ``test_the_attenuation_model_matches_its_published_anchors``.
    """
    return _shadow_and_path(steps, decay, lr0, eta_min, cosine)[2]


def _shadow_and_path(steps, decay, lr0=LR0, eta_min=ETA_MIN, cosine=True):
    """(exported shadow magnitude, trained magnitude, their ratio)."""
    b = 0.0
    s = 0.0
    for k in range(steps):
        lr = (
            eta_min + (lr0 - eta_min) * (1 + math.cos(math.pi * k / steps)) / 2
            if cosine
            else lr0
        )
        b += lr
        s = decay * s + (1 - decay) * b
    return s, b, s / b


def exported_strength(steps, decay, lr0=LR0):
    """ABSOLUTE magnitude of the exported (shadow) adapter, not a fraction.

    THIS IS THE QUANTITY THE EVALUATOR SEES, and it is why this file no longer
    asserts on ``exported_fraction``.  On ideogram4 the validator's
    ``DualModelGuider`` takes its NEGATIVE branch from a separate, never-LoRA'd
    checkpoint (G.O.D pin b026da04 ``comfy_workflows/lora_ideogram4.json``:
    ``model_negative`` <- ``Unconditional_checkpoint_loader`` ->
    ``ideogram4_unconditional_fp8_scaled.safetensors``), so a LoRA that moves its
    branch by ``d`` moves the scored prediction by ``c*d`` with c ~= 6.8
    arc-averaged over cfg 8 falling to ``max(cfg-3,1)=5`` from ``start_percent``
    0.7 (``diffusion.py:235-236``).  Optimal strength therefore scales as 1/c and
    the overshoot penalty as c^2 — magnitude is the axis that matters here, and
    ATTENUATION IS HOW WE HIT IT, not a defect to be minimised.

    The eta_min of the shipped schedule is 0.1*lr, so scaling lr scales the whole
    path; that is exactly the perturbation the matched pair applied.
    """
    return _shadow_and_path(steps, decay, lr0, eta_min=0.1 * lr0)[0]


def test_activated_policy_still_carries_the_lr_and_ema_this_law_assumes():
    """The law below is derived FROM these values; pin them.

    THIS TRIPWIRE FIRED TWICE IN WEEK 6 AND WAS DISCHARGED BOTH TIMES.  It was
    written to go red if ``ema_decay`` ever moved off 0.995 so that the depth law
    would be re-derived rather than silently inherited.  It fired when
    ``forge.ideogram_release_policy`` adopted the field's 0.99, and it fired
    again at the week-6 integration merge when unit 3 VACATED that amendment and
    put the decay back on 0.995.

    THE SECOND DISCHARGE FOUND A DEFECT IN THE FIRST.  The re-derivation done
    when 0.99 landed expressed the guard as ``exported_fraction >= 0.85``, a
    floor calibrated at decay 0.99 (T >= 338).  That framing treats EMA
    attenuation as pure loss.  It is not: the in-family rank-1 anchor
    (5FNLSgh8, 378 steps, decay 0.995) exports ``f = 0.685``, so THE 0.85 FLOOR
    REJECTS THE ONE ARTIFACT THIS WHOLE ROW IS CALIBRATED TO REPRODUCE.  A guard
    that fails the target is a broken guard, not a mis-tuned constant, so it was
    replaced rather than re-tuned — see
    ``test_ideogram4_export_stays_inside_the_measured_strength_band``.

    The SHIPPED DEPTHS still do not move: 414/589/614 are set by the size law and
    the do_cfg clock ceiling at either decay.

    ``lr`` is deliberately still pinned hard: it is the coupled decision the
    Week-6 amendment did NOT take, and the depth law IS derived from it.
    """
    train = ideogram_release_policy._EXPECTED_RECIPE["train"]
    assert train["lr"] == 2.5e-5
    assert train["lr_scheduler"] == "cosine"
    assert train["lr_scheduler_params"] == {"eta_min": 2.5e-6}
    assert train["do_cfg"] is True and train["cfg_scale"] == 10.0
    assert train["ema_config"] == {"use_ema": True, "ema_decay": EXPECTED_EMA_DECAY}
    assert EMA_DECAY == EXPECTED_EMA_DECAY
    # The amendment record is retained as a VACATED record rather than deleted,
    # so the audit trail shows the field moved and moved back, and the shipped
    # decay is once again the I-J20-D2 port's own validated value with no live
    # divergence.  A future re-adoption of 0.99 must land a NEW amendment.
    amendment = ideogram_release_policy.WEEK6_EMA_AMENDMENT
    assert amendment["field"] == "config.process[0].train.ema_config.ema_decay"
    assert amendment["status"] == "vacated"
    # Vacated means the record no longer carries a divergence at all: BOTH the
    # validated and the amended value are back on the anchor's 0.995, so the
    # emitted recipe is the I-J20-D2 port with zero live amendments.
    assert amendment["validated_value"] == EXPECTED_EMA_DECAY
    assert amendment["amended_value"] == EXPECTED_EMA_DECAY
    assert amendment["covered_by_source_validation_cell"] is True
    assert ideogram_release_policy._POLICY_BODY["amendments"] == []
    assert amendment in ideogram_release_policy._POLICY_BODY["vacated_amendments"]
    assert (
        ideogram_release_policy.PRODUCTION_ACTIVATION["amendment_sha256"]
        == ideogram_release_policy.AMENDMENT_SHA256
    )


def test_the_attenuation_model_matches_its_published_anchors():
    """Guard the model this file now asserts on against silent drift.

    The constant-lr case has a closed form,
    ``f = (1-d^T) - d(1 - T d^(T-1) + (T-1) d^T) / (T(1-d))``, and the two
    anchors below are the figures derived independently three times in week 6
    (unit report, adversarial review, correctness sweep).  A second copy of this
    model lives in ``tests/test_week6_ideogram_ema_horizon.py``; both are pinned
    to the same anchors, so an edit to either one is caught here.
    """

    def closed_form(steps, decay):
        return (1 - decay**steps) - decay * (
            1 - steps * decay ** (steps - 1) + (steps - 1) * decay**steps
        ) / (steps * (1 - decay))

    for steps, expected in ((177, 0.338689), (421, 0.584607)):
        sim = exported_fraction(steps, 0.995, cosine=False)
        assert sim == pytest.approx(expected, abs=5e-6)
        assert sim == pytest.approx(closed_form(steps, 0.995), abs=1e-9)
    # And the cosine values this file's floor is calibrated against.
    assert exported_fraction(421, 0.99) == pytest.approx(0.8939, abs=5e-4)
    assert exported_fraction(421, 0.995) == pytest.approx(0.7192, abs=5e-4)


@pytest.mark.parametrize("row", REAL_IDEOGRAM_TASKS, ids=IDS)
def test_ideogram4_depth_is_set_by_our_own_pipeline_not_the_field(row):
    """The size law must BIND, and bind below the do_cfg clock ceiling."""
    _task, _family, pairs, hours, _winner, expected = row
    sec = recipe.SEC_PER_IT["ideogram4"]
    steps = recipe.size_scaled_steps("ideogram4", pairs, hours, 2000)
    assert steps == expected

    cap = _cap(hours, recipe.MARGIN, sec)
    assert steps < cap, "the size law, not the clock, must decide ideogram4 depth"
    # ...and it must bind with real room, not by one step.  A cap built on an
    # INFERRED constant (the 2x do_cfg multiplier) is not something to run flat
    # into: qwen-image already showed what that costs.
    assert steps <= 0.95 * cap


@pytest.mark.parametrize("row", REAL_IDEOGRAM_TASKS, ids=IDS)
def test_ideogram4_depth_is_invariant_to_margin_and_sec_per_it(row):
    """The other week-6 fixes must not be able to move this row.

    MARGIN is under revision (globally 0.92 today, possibly per-type tomorrow)
    and SEC_PER_IT["ideogram4"] rests on an INFERRED 2x do_cfg multiplier that
    the two week-6 audits disagreed about.  Because the size law binds, neither
    can change what ideogram4 ships — except for the single tightest shape under
    the most pessimistic combination, and then only marginally.
    """
    _task, _family, pairs, hours, _winner, expected = row
    law = recipe.STEP_TABLE["ideogram4"]
    scaled = law["base"] * (pairs / law["n_ref"]) ** law["p"]
    scaled = int(round(max(law["min"], min(law["max"], scaled))))
    assert scaled == expected

    for margin in (0.85, 0.90, 0.92, 0.95):
        for sec in (2.1, 3.0, 4.2):
            shipped = max(1, min(scaled, _cap(hours, margin, sec)))
            # Worst case in the grid is margin 0.85 x sec 4.2 on the 1.0 h
            # shapes, which clips 616 to 614 — 0.3%.
            assert shipped >= expected - 3


@pytest.mark.parametrize("row", REAL_IDEOGRAM_TASKS, ids=IDS)
def test_ideogram4_export_stays_inside_the_measured_strength_band(row):
    """(a) EMA attenuation — the reason this law is deep and not shallow.

    ``setup_ema`` builds ``ExponentialMovingAverage`` from the LoRA parameters
    at init, with ``use_num_updates=False`` so the ``(1+n)/(10+n)`` warm-up is
    DISABLED and the decay is flat from step 1 (0.99 since the Week-6
    EMA-horizon amendment; it was 0.995 when this test was written).  ``save()`` then calls
    ``ema.eval()`` unconditionally, so every artifact we upload — periodic saves
    included — is the shadow, not the trained weights
    (``BaseSDTrainProcess.py:491,495-497`` save, ``:769-781`` setup_ema,
    ``:2031`` call site; ``toolkit/ema.py:43-72,100-152,336-341``).

    WHAT THIS TEST ACTUALLY PINS, stated precisely because the previous version
    of this docstring was wrong.  It said ``0.995**steps`` is "the share of the
    export that is literally the untrained initialisation".  It is not:
    ``lora_up`` is ZERO-initialised (``toolkit/lora_special.py:122``), so the
    adapter has no effect at init and there is no untrained signal for the
    shadow to retain.  ``0.995**steps`` is the EMA's MEMORY HORIZON, and the
    real cost is ATTENUATION of the trained delta — under our cosine
    2.5e-5 -> 2.5e-6 schedule the export is ~43% of the final iterate at 177
    steps, ~72% at 421 and ~83% at 616 (INFERRED from
    ``(1-d)*sum_t d^(T-t)*theta_t / theta_T`` on the cumulative-lr path; a
    constant-lr path gives 34% / 58% / 69%).

    THE ASSERTION IS NOW A STRENGTH BAND, NOT A FRACTION FLOOR (integrator,
    week-6 merge, 2026-08-07).  Two earlier framings were both wrong:

      * ``EMA_DECAY ** steps <= 0.15`` — a memory-horizon proxy that measures
        nothing the evaluator sees, and toothless past T ~= 189 at 0.99.
      * ``exported_fraction >= 0.85`` — the right direction of concern, wrong
        sign of policy.  Attenuation is not waste; it is the mechanism by which
        a deeper run lands on a competitive exported magnitude.  At decay 0.995
        that floor needs T >= 678, and it FAILS THE ANCHOR: 5FNLSgh8 won rank 1
        of 16 at 378 steps with ``f = 0.685``.

    WHAT IS ASSERTED INSTEAD, and where each end of the band comes from:

      LOWER 0.85x the anchor.  Rejects the retired two-point law
      ``240*(n/24)^0.57``, which is what the original guard existed to reject:
      at n_train 12/36/41 it ships 162/302/326 steps for strength ratios
      0.256/0.712/0.801 — all three fail.  This is a real floor, not decoration.

      UPPER 2.0x the anchor, EXCLUSIVE.  Not a chosen constant: 2.0 is exactly
      where the one measured over-strength artifact sits.  5EACrayt ran the
      anchor's config at 2x lr and the SAME 378 steps, so its exported strength
      is 2.000x by construction, and it scored 0.0965093 against the anchor's
      0.0502341 (rank 13 of 16, +92.1%).  We refuse to ship a modelled strength
      that reaches a magnitude the field has measured to be badly wrong.

    KNIFE-EDGE, STATED RATHER THAN HIDDEN: 84be9fcd ships 1.959x — 98.0% of the
    measured-bad magnitude.  So this test is 2% from red on our DEEPEST shape,
    and that is deliberate: ideogram4 depth cannot be raised again without
    re-arguing this band.

    WHY THE BAND IS NOT EVIDENCE THAT 589/614 ARE TOO DEEP.  The strength model
    is linear in the cumulative lr path and was measured on the LR axis at fixed
    depth.  The field falsifies that extrapolation along the DEPTH axis: on
    b72da8c6 the rank-1 entrant shipped 1100 steps at lr 4e-4 and the rank-2
    shipped 1523 at the same lr — modelled strengths far past 2x, and they took
    the top two places.  More steps at fixed lr is a better-converged adapter,
    not merely a bigger one; the model's own docstring flags it as "exact early,
    degrades late".  The band is therefore a TRIPWIRE forcing re-argument, not a
    claim that the optimum is 1.0x.  See the week-6 release note's "known unfixed
    exposure" for the full statement.
    """
    _task, _family, pairs, hours, _winner, expected = row
    steps = recipe.size_scaled_steps("ideogram4", pairs, hours, 2000)
    assert steps == expected

    anchor = exported_strength(JUL20_IN_FAMILY_WINNER_STEPS, 0.995)
    ratio = exported_strength(steps, EMA_DECAY) / anchor

    assert ratio >= 0.85, (
        f"{_task}: exported strength is only {ratio:.3f}x the in-family rank-1 "
        f"anchor at {steps} steps / decay {EMA_DECAY} — too shallow"
    )
    assert ratio < MATCHED_PAIR_OVERSTRENGTH_RATIO, (
        f"{_task}: exported strength {ratio:.3f}x reaches the magnitude measured "
        f"at rank 13 of 16 (5EACrayt, 2x lr, same depth, +92.1% loss)"
    )
    # The retired c424362 two-point law is strictly rejected by the lower bound.
    old_law = int(round(240 * (pairs / 24) ** 0.57))
    assert exported_strength(old_law, EMA_DECAY) / anchor < 0.85


@pytest.mark.parametrize("row", REAL_IDEOGRAM_TASKS, ids=IDS)
def test_readopting_099_would_breach_the_band_on_a_real_shape(row):
    """Why the Week-6 EMA amendment was vacated, in the band's own units.

    0.99 halves the averaging horizon (99 steps against 199), so the export
    carries a larger share of a path whose magnitude is already set by depth.
    On the two 1.0 h shapes that pushes the modelled strength PAST the measured
    over-strength point; on the 0.75 h shape it does not, which is precisely why
    the amendment looked harmless when it was only checked at R1.
    """
    _task, _family, pairs, hours, _winner, expected = row
    anchor = exported_strength(JUL20_IN_FAMILY_WINNER_STEPS, 0.995)
    at_099 = exported_strength(expected, 0.99) / anchor
    at_0995 = exported_strength(expected, 0.995) / anchor
    assert at_099 > at_0995
    breached = {"b72da8c6": True, "84be9fcd": True, "1365fa1c": False}
    assert (at_099 >= MATCHED_PAIR_OVERSTRENGTH_RATIO) is breached[_task], (
        _task,
        at_099,
    )


@pytest.mark.parametrize("row", REAL_IDEOGRAM_TASKS, ids=IDS)
def test_ideogram4_fits_the_budget_at_the_do_cfg_corrected_rate(row):
    """(b) the do_cfg clock ceiling — the reason this law is not deeper still.

    Asserted at the honest 4.2 s/step, and then again at the rate that would
    make us miss the deadline, which must be >=20% worse than modelled.

    THREE DIFFERENT THRESHOLDS EXIST HERE and the previous docstring blurred
    them into one ("truncates only above 5.06 s/step ... 2.47x the field bound"),
    which made the cushion look about twice as large as it is:

      * PLANNING truncation — the SEC_PER_IT CONSTANT is wrong, so the cap
        ``(budget*MARGIN - 480)/SEC`` falls below the law.  On the tightest real
        shape (84be9fcd, 616 steps @ 1.0 h) that happens above **4.597 s/step**
        = +9.5% over 4.2 = 2.28x the field's 2.019 no-do_cfg bound.  The other
        two shapes break at 4.760 (421 @ 0.75 h) and 4.808 (589 @ 1.0 h).
      * RUNTIME kill — the BOX is slower, so the terminate gate stops us.  That
        gate is ``budget - 180 - 45``, less ``STARTUP_S``, so the tightest shape
        stops short above **4.992 s/step** (+18.9%).  This is what the assertion
        below approximates.
      * The retired 5.06 was the runtime threshold with ``STOP_MARGIN_S``
        dropped as well, quoted as if it were the constant-error budget.

    ``breaking_rate`` below still omits ``STOP_MARGIN_S`` (it is a 45 s / ~1.4%
    optimism on a 1.0 h budget); the assertion is a floor, so the direction is
    safe, but the number it computes is 5.065, not 4.992.
    """
    _task, _family, pairs, hours, _winner, expected = row
    steps = recipe.size_scaled_steps("ideogram4", pairs, hours, 2000)
    budget = hours * 3600.0
    wall = recipe.projected_wall_s("ideogram4", steps)
    assert wall <= budget * recipe.MARGIN
    # Utilisation: the audit's headline ideogram4 defect was that every task
    # left 75-77% of its grant unused.  Fixed, without filling it to the brim.
    assert 0.78 <= wall / budget <= 0.90

    # The rate at which the SOFT stop (hard deadline minus the export reserve)
    # would truncate us.  _run_toolkit gates termination on remaining(), which
    # already subtracts EXPORT_RESERVE_S.
    trainable_s = budget - recipe.EXPORT_RESERVE_S - recipe.STARTUP_S
    breaking_rate = trainable_s / steps
    assert breaking_rate >= 1.20 * recipe.SEC_PER_IT["ideogram4"]


@pytest.mark.parametrize("row", REAL_IDEOGRAM_TASKS, ids=IDS)
def test_ideogram4_truncation_degrades_rather_than_forfeits(row):
    """Overrun is recoverable: _terminate -> _finalize promotes a periodic save.

    That only holds if periodic saves actually exist and land early enough, so
    both are asserted here rather than assumed.  ai-toolkit fires a numbered
    save when ``step % save_every == 0``, then an unnumbered exact final.
    """
    _task, _family, pairs, hours, _winner, expected = row
    steps = recipe.size_scaled_steps("ideogram4", pairs, hours, 2000)
    save_every = recipe.kill_safe_save_every(steps, 200)
    periodic = (steps - 1) // save_every
    assert 3 <= periodic <= 5

    budget = hours * 3600.0
    first = recipe.first_save_wall_s("ideogram4", steps, save_every)
    assert first <= 0.35 * budget
    # Even at DOUBLE the modelled per-step cost the first recovery point is on
    # disk before the soft stop.
    doubled = recipe.STARTUP_S + save_every * recipe.SEC_PER_IT["ideogram4"] * 2
    assert doubled <= budget - recipe.EXPORT_RESERVE_S
    # The deepest surviving candidate after a late kill keeps most of the run.
    assert (periodic * save_every) / steps >= 0.75


def test_where_the_clock_takes_over_from_the_law_is_known_and_bounded():
    """On SHORT grants the clock does bind, and that is stated rather than hidden.

    The law is deliberately sized so it binds on every 1.0 h shape up to N=50
    and on the real 0.75 h shape (N=14).  It does NOT bind on a 0.75 h task with
    N >= ~21, because ``do_cfg`` costs us 4.2 s/step and a 0.75 h grant only
    buys 477 of them.  Round 1 was a 0.75 h task on Aug-3, so this corner is
    reachable on Monday and is asserted, not assumed:

      * the crossover sits at N ~= 21 (below the Aug-3 style-task sizes);
      * at the cap we still ship materially deeper than the withdrawn law;
      * a clock-bound run keeps four periodic candidates, so a deadline stop
        promotes ~80% of the planned depth rather than forfeiting.

    The ~10% rate tolerance of a clock-bound run is a property of MARGIN, not of
    this row, and it is identical for every clock-bound type in the table.
    """
    cap_075 = _cap(0.75, recipe.MARGIN, recipe.SEC_PER_IT["ideogram4"])
    law = recipe.STEP_TABLE["ideogram4"]

    def pure(n):
        return law["base"] * (n / law["n_ref"]) ** law["p"]

    crossover = next(n for n in range(5, 60) if pure(n) > cap_075)
    assert 18 <= crossover <= 24

    for pairs in (24, 36, 50):
        steps = recipe.size_scaled_steps("ideogram4", pairs, 0.75, 2000)
        assert steps == cap_075
        # still far deeper than the withdrawn two-point law would have shipped
        assert steps > int(round(240 * (pairs / 24) ** 0.57))
        save_every = recipe.kill_safe_save_every(steps, 200)
        periodic = (steps - 1) // save_every
        assert periodic >= 3
        assert (periodic * save_every) / steps >= 0.75
    # ...and every 1.0 h shape in the observed size range stays law-bound.
    cap_10 = _cap(1.0, recipe.MARGIN, recipe.SEC_PER_IT["ideogram4"])
    assert recipe.size_scaled_steps("ideogram4", 50, 1.0, 2000) < cap_10


def test_ideogram4_law_is_flat_because_the_field_has_no_size_signal():
    """N=11 -> 378, N=14 -> 174, N=40 -> 1100, N=46 -> 341 across two
    tournaments: the winners' depth is uncorrelated with dataset size.  The
    exponent therefore mirrors krea2's deliberately flat 0.35 rather than the
    0.57 that a 2-point fit to one operator produced.

    The four winning depths are not merely noisy, they are MUTUALLY
    INCONSISTENT under any power law: the two closest points in N are the
    furthest apart in depth (N=40 -> 1100 and N=46 -> 341 is 15% more data for
    3.23x fewer steps, a local exponent of -8.4).  `p` is therefore near-flat
    because N is the wrong instrument, not because a flat exponent fits.
    """
    law = recipe.STEP_TABLE["ideogram4"]
    assert law["p"] <= 0.40
    assert abs(law["p"] - recipe.STEP_TABLE["krea2"]["p"]) <= 0.05

    # No power law through any two of the Aug-3 three lands near the third.
    winners = {14: 174, 46: 341, 40: 1100}
    ns = sorted(winners)
    for i in range(len(ns)):
        for j in range(i + 1, len(ns)):
            n1, n2 = ns[i], ns[j]
            p = math.log(winners[n2] / winners[n1]) / math.log(n2 / n1)
            (n3,) = [n for n in ns if n not in (n1, n2)]
            predicted = winners[n1] * (n3 / n1) ** p
            ratio = predicted / winners[n3]
            assert not (0.7 <= ratio <= 1.4), (
                f"a size law through N={n1},{n2} would predict N={n3} to within "
                f"{ratio:.2f}x — if this ever passes, re-open the size-law "
                f"question, because it means the field acquired a size signal"
            )


def test_ideogram4_matches_the_only_in_family_field_winner():
    """The R1-shape anchor this row is actually calibrated against.

    Every OTHER ideogram4 artifact in the field runs `lr: 0.0004` constant, 16x
    our peak, so its step count is not transferable (that is the whole premise
    of this file).  Exactly ONE is not like that: 5FNLSgh8 on the Jul-20 R1 task
    3cfa1578 ran lr 2.5e-5 with a cosine schedule, EMA on, and a rank-32 LoRA —
    our family — and WON, 1 of 16, at 378 steps on N=11 / n_train 9, h=0.75.

    That is the R1 shape, ideogram4 is half the R1 draw, and it is the only
    depth in either tournament produced by a recipe like ours.  So it is the one
    number this row must not drift away from, and nothing else in this file
    pins the SMALL-N end.

    HOW STRONG THE ANCHOR IS, honestly: n=1.  The band is deliberately wide
    (0.75x..1.35x) — it is a tripwire against a rewrite that lands 4x away, not
    a claim that 378 is optimal.  For calibration, the production pin 084ea914
    (base=140 p=0.50) returns 86 here, which is 0.23x and would fail — and 86
    is not hypothetical, it is what those constants DID ship for us on this
    exact task.

    EVALUATED AT n_train, NOT AT N.  The winner's container held 9 images, and
    so will ours; `base` is set to 378/(9/24)**0.32 = 517.4 so the equality is
    exact rather than "+3.2% at a shape the runtime never sees".
    """
    shipped = recipe.size_scaled_steps(
        "ideogram4", JUL20_R1_N_TRAIN, JUL20_R1_HOURS, 1000
    )
    ratio = shipped / JUL20_IN_FAMILY_WINNER_STEPS
    assert shipped == JUL20_IN_FAMILY_WINNER_STEPS, (
        f"the row no longer reproduces its own calibration anchor: {shipped} "
        f"!= {JUL20_IN_FAMILY_WINNER_STEPS}"
    )
    assert 0.75 <= ratio <= 1.35, (
        f"n_train={JUL20_R1_N_TRAIN} h={JUL20_R1_HOURS} ships {shipped} steps against "
        f"the only in-family field winner's {JUL20_IN_FAMILY_WINNER_STEPS} "
        f"({ratio:.2f}x)"
    )
    # The anchor must be the LAW, not the clock: if the clock were binding here
    # the agreement would be an accident of SEC_PER_IT (which is UNMEASURED for
    # ideogram4) rather than a property of the depth policy.
    law = recipe.STEP_TABLE["ideogram4"]
    pure = law["base"] * (JUL20_R1_N_TRAIN / law["n_ref"]) ** law["p"]
    assert shipped == int(round(max(law["min"], min(law["max"], pure))))


def test_ideogram4_is_not_below_every_depth_that_ever_won():
    """Directional guard: do not re-ship a law that is under the whole field.

    The production pin 084ea914 (base=140 p=0.50 min=48 max=400) returned
    86/99/171/183 against winning depths of 378/174/1100/341 at the n_train
    9/12/36/41 those four tasks deliver (N=11/14/40/46) —
    every residual the same sign, i.e. a BIAS, not scatter.  Our two tournament
    results to date were both on the shallow side of their fields, and the one
    catastrophic ideogram4 artifact on record (5FjDsFGA, Jul-20, +421% loss vs
    rank 1) is a SHALLOW one at 200 steps; there is no observed deep-side
    catastrophe anywhere in this type.

    This does NOT assert that deeper is better — Spearman(steps, loss) over the
    16-artifact Jul-20 field is +0.18 (p = 0.49), i.e. no detectable effect in
    either direction, and the matched-depth pair at 378/378 finished rank 1 and
    rank 13 on lr alone.  It asserts only that we are not uniformly beneath a
    field whose depth we cannot explain.
    """
    # keyed by n_train — the abscissa the law is handed, not the audit N
    winners = {9: 378, 12: 174, 36: 1100, 41: 341}
    hours = {9: 0.75, 12: 0.75, 36: 1.0, 41: 1.0}
    below = [
        n
        for n, w in winners.items()
        if recipe.size_scaled_steps("ideogram4", n, hours[n], 1000) < w
    ]
    assert len(below) < len(winners), (
        "the law is below EVERY winning ideogram4 depth ever observed "
        f"(shapes {sorted(below)}) — that is the 084ea914 failure mode"
    )


def test_ideogram4_bounds_can_actually_bind():
    """``max`` must not be decoration.

    The c424362 row carried ``max: 1600`` while the law topped out at 365 at
    N=50 — it could never bind within 4x, so it encoded nothing.

    ALL BOUNDS BELOW ARE NOW IN n_train, THE LAW'S ACTUAL ARGUMENT.  620 first
    CHANGES the shipped depth at **n_train = 43**, which is the same real task
    it always bit at: the old N-space crossover was 48 and 48 - ceil(4.8) = 43.
    The refit moved the units, not the policy.

    And be honest about the size of the effect: 620 binds only at n_train 43,
    44, 45 (N = 48, 49, 50) and changes depth by 3 / 8 / 12 steps.  That is at
    the very top of the observed range and ABOVE the largest ideogram4 task ever
    seen (84be9fcd, n_train 41), so it is very nearly decoration; it is kept as
    a cap on extrapolating a recipe we have never run past ~200 steps in a
    tournament.  ``min`` is the opposite kind of bound: it guards the unobserved
    small tail, below the n_train 9 minimum ever seen — which is exactly the
    Jul-20 R1 shape this row is anchored on, so ``min`` must NOT bind there.
    """
    law = recipe.STEP_TABLE["ideogram4"]

    def pure(n):
        return law["base"] * (n / law["n_ref"]) ** law["p"]

    # the largest ideogram4 abscissa ever observed is 41; the largest of ANY
    # type is 45 (f6725c2b, N=50).  `max` must sit between them.
    assert pure(41) < law["max"] <= pure(45), "max must bind inside n_train 8..45"
    assert recipe.size_scaled_steps("ideogram4", 45, 1.0, 2000) == law["max"]
    # min binds strictly below the smallest ideogram4 abscissa ever observed
    # (n_train 9, from the N=11 Jul-20 R1 task 3cfa1578).
    assert pure(9) > law["min"] >= pure(7)


def test_no_family_router_was_introduced():
    """A style/product split was considered and REJECTED on the evidence.

    It is not infeasible — ``spec.trigger_word is None`` separates ``style``
    from every other family 12/12 across the Aug-3 configs.  It is rejected
    because the two style tasks disagree with each other by 4.5x (341 won at
    N=46, ~1250 won at N=40), so there is no style depth to route TO.  This test
    exists so that a future reader does not mistake the absence of a router for
    an oversight, and so that adding one has to be a deliberate act.
    """
    assert set(recipe.STEP_TABLE["ideogram4"]) == {
        "base",
        "n_ref",
        "p",
        "min",
        "max",
    }
    import inspect

    src = inspect.getsource(recipe.size_scaled_steps)
    assert "family" not in src and "trigger" not in src
