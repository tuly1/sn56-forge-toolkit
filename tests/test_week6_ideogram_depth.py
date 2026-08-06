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

import pytest

from forge import ideogram_release_policy, recipe


# task, family, n_pairs, hours, rank-1 miner's shipped depth, what we now ship
REAL_IDEOGRAM_TASKS = [
    # 1365fa1c: the ONLY clean shallow-vs-deep head-to-head in ideogram4.
    # 174 (rank 1) beat >900 (rank 2) by 46.1% — but the deep arm stripped its
    # metadata and published no config, so it is not a depth-only comparison.
    ("1365fa1c", "product", 14, 0.75, 174, 421),
    # b72da8c6: BOTH arms deep (>=1200 rank 1 vs 1523 rank 2, +4.4%).  No
    # shallow arm exists on this task, so it cannot show that a shallow run
    # would lose — only that 1250 ~= 1523.
    ("b72da8c6", "style", 40, 1.0, 1300, 589),
    # 84be9fcd: 341 (rank 1) vs an opponent who published TWO FILES, no
    # __metadata__ and no checkpoint ladder.  ZERO depth information.
    ("84be9fcd", "style", 46, 1.0, 341, 616),
]
IDS = [row[0] for row in REAL_IDEOGRAM_TASKS]

# The EMA decay the activated release policy binds.  Not a free constant here:
# it is read back out of the policy so this file fails if the policy moves.
EMA_DECAY = 0.995


def _cap(hours, margin, sec_per_it):
    train_s = hours * 3600.0 * margin - recipe.STARTUP_S - recipe.EXPORT_RESERVE_S
    return int(train_s / sec_per_it) if train_s > 0 else 1


def test_activated_policy_still_carries_the_lr_and_ema_this_law_assumes():
    """The law below is derived FROM these two values; pin them.

    If ``lr`` ever returns to the field's 4e-4, or ``ema_decay`` drops to the
    field's 0.99, the whole argument for this depth changes and the law must be
    re-derived rather than silently inherited.
    """
    train = ideogram_release_policy._EXPECTED_RECIPE["train"]
    assert train["lr"] == 2.5e-5
    assert train["lr_scheduler"] == "cosine"
    assert train["lr_scheduler_params"] == {"eta_min": 2.5e-6}
    assert train["ema_config"] == {"use_ema": True, "ema_decay": EMA_DECAY}
    assert train["do_cfg"] is True and train["cfg_scale"] == 10.0


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
def test_ideogram4_export_is_not_dominated_by_the_untrained_init(row):
    """(a) EMA attenuation — the reason this law is deep and not shallow.

    ``setup_ema`` builds ``ExponentialMovingAverage`` from the LoRA parameters
    at init, with ``use_num_updates=False`` so the ``(1+n)/(10+n)`` warm-up is
    DISABLED and the decay is a flat 0.995 from step 1.  ``save()`` then calls
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

    The assertion below is therefore a DEPTH FLOOR expressed through a monotone
    proxy, not a measurement of dilution, and the local name ``untrained_share``
    is a misnomer left in place because this unit may only edit docstrings (see
    the integration request in the week-6 correctness-sweep report).  The bound
    0.15 corresponds to steps >= ~379.  Direction is unchanged by the
    correction: deeper still means less attenuation.
    """
    _task, _family, pairs, hours, _winner, expected = row
    steps = recipe.size_scaled_steps("ideogram4", pairs, hours, 2000)
    assert steps == expected
    untrained_share = EMA_DECAY ** steps
    assert untrained_share <= 0.15, (
        f"{_task}: {100 * untrained_share:.0f}% of the export is untrained init"
    )
    # And strictly better than what the c424362 two-point law would have shipped.
    old_law = int(round(240 * (pairs / 24) ** 0.57))
    assert untrained_share < EMA_DECAY ** old_law


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
    """N=9 -> 1200, N=14 -> 174, N=40 -> ~1250, N=46 -> 341 across two
    tournaments: the winners' depth is uncorrelated with dataset size.  The
    exponent therefore mirrors krea2's deliberately flat 0.35 rather than the
    0.57 that a 2-point fit to one operator produced.
    """
    law = recipe.STEP_TABLE["ideogram4"]
    assert law["p"] <= 0.40
    assert abs(law["p"] - recipe.STEP_TABLE["krea2"]["p"]) <= 0.05


def test_ideogram4_bounds_can_actually_bind():
    """``max`` must not be decoration.

    The c424362 row carried ``max: 1600`` while the law topped out at 365 at
    N=50 — it could never bind within 4x, so it encoded nothing.  620 first
    CHANGES the shipped depth at **N = 48** — not N=47, as this docstring used
    to say: at N=47 the law returns 619.975, which rounds to 620 with or without
    the cap.  ``test_step_table_max_binding_sizes`` pins the crossover at 48.

    And be honest about the size of the effect: 620 binds only at N = 48, 49, 50
    and changes depth by 4 / 8 / 12 steps.  It is inside the observed range
    (f6725c2b ran N=50) but it is very nearly decoration; it is kept as a cap on
    extrapolating a recipe we have never run past ~200 steps in a tournament.
    ``min`` is the opposite kind of bound: it guards the unobserved small tail,
    below the N=9 minimum ever seen.

    NOTE the assertion is ``pure(46) < max <= pure(50)``, which is satisfied by
    the N=47 no-op as well — it does not by itself pin 48.
    """
    law = recipe.STEP_TABLE["ideogram4"]

    def pure(n):
        return law["base"] * (n / law["n_ref"]) ** law["p"]

    assert pure(46) < law["max"] <= pure(50), "max must bind inside 9..50"
    assert recipe.size_scaled_steps("ideogram4", 50, 1.0, 2000) == law["max"]
    # min binds strictly below the smallest ideogram4 dataset ever observed
    # (N=9, Jul-20 R1 task 3cfa1578).
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
