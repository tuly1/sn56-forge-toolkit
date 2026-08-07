"""Week-6 regression guard: depth law, wall-clock model, and training geometry.

WHY THIS FILE EXISTS
====================
A Jul-16 calibration hard-coded shallow depth laws into ``forge/recipe.py`` from
an 8..128-step probe on 12 photos, and NOTHING CAUGHT IT FOR FIVE WEEKS.  krea2
stayed capped at 400 steps until hours before the Aug-3 tournament; the other
four types were never revisited at all.  We shipped 823 steps into an R1 field
whose template-recipe pack ran 1432-2000 and were eliminated by 0.97%.

Every number asserted here is anchored to a REAL Aug-3 tournament artifact or to
pinned ai-toolkit / evaluator source, so the next silent miscalibration fails a
test instead of a tournament.

Evidence:
  ops/experiments/week6/FIELD-DEPTH-LAW-AUDIT.md            (40 artifacts, 14 tasks)
  ops/experiments/week6/PIPELINE-MATERIALIZATION-AUDIT.md   (what actually runs)
  SN56-project/evidence/week6-field-depth-audit-20260806/    (scripts + payloads)
  SN56-project/evidence/week6-tournament-dataset-harvest-20260806/ (task shapes)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
from PIL import Image

from forge import geometry, ideogram_release_policy, recipe
from forge.config import build_config
from forge.data.schema import ImageSpec


# --------------------------------------------------------------------------- #
# The fourteen real Aug-3 task shapes.
#
# task / round / model_type / family
#   / n_pairs        = `image_text_pairs` in the auditing record (the FULL set)
#   / n_train        = images inside train_data.zip = WHAT THE CONTAINER GETS
#   / hours_to_complete
#   / rank-1 miner's shipped steps
#   / steps forge shipped on the Aug-3 pin 084ea914 (BEFORE)
#   / steps forge ships now (AFTER)
#   / save_every / expected single-scalar resolution under the geometry policy
#   / (on-geometry images, total images) under that resolution
#   / source dimension histogram, straight out of the harvest task-meta.json
#
# `None` resolution == the type is structurally excluded from the geometry
# policy (ideogram4; see forge/geometry.py) and must keep the template list.
#
# THE `n_train` COLUMN IS NEW (2026-08-07) AND IT IS THE REASON THIS FILE WAS
# GREEN WHILE THE CONTAINER SHIPPED SOMETHING ELSE.  The validator withholds a
# 10% holdout for its own scoring and hands the miner only the remainder, so
# `image_text_pairs` is NOT the number `recipe.size_scaled_steps` receives:
#
#     forge/tasks/aitoolkit.py:56  total_pairs = pairs inside train_data.zip
#     forge/tasks/aitoolkit.py:78  pairs = total_pairs - holdout_pairs  (0 in prod)
#     forge/tasks/aitoolkit.py:97  build_config(spec, num_images=pairs, ...)
#
# Every value below is OBSERVED, not modelled: the zip END-OF-CENTRAL-DIRECTORY
# of each task's `training_data` URL was read by HTTP range request (directory
# only, no image payload).  All fourteen satisfy n_train == N - ceil(0.10*N)
# exactly.  `_build` now passes n_train, so a regression in the plumbing fails
# here instead of in a tournament.
#
# KNOWN GAP, STATED RATHER THAN PAPERED OVER: the `dims` histogram and therefore
# the two geometry columns are still the FULL n_pairs distribution, because
# WHICH images the validator withholds is not recoverable from any artifact we
# hold (the zip directory gives names and byte sizes, not pixel dimensions).
# That affects only the geometry columns, which no depth assertion reads;
# `num_images` — the only input the depth law takes — is exact.
# --------------------------------------------------------------------------- #
REAL_TASKS = [
    ("41025fb5", 1, "krea2", "design", 21, 18, 0.75, 1000, 824, 1432, 287,
     887, (21, 21), {(1024, 768): 21}),
    ("7421f056", 2, "qwen-image", "design", 28, 25, 1.25, 850, 836, 836, 168,
     887, (28, 28), {(1024, 768): 28}),
    ("84be9fcd", 2, "ideogram4", "style", 46, 41, 1.0, 341, 183, 614, 123,
     None, None, {(1408, 768): 45, (768, 1376): 1}),
    ("b290d171", 2, "z-image", "design", 39, 35, 1.0, 1188, 860, 1188, 238,
     747, (37, 39), {(1408, 768): 37, (768, 1376): 2}),
    # WINNER COLUMN CORRECTED 750 -> 650 and 870 -> 754.  kohya records EPOCHS,
    # so these are `epochs x images`, and the image count is 13 (the zip), not
    # 15 (the auditing record): 5D7iEJm5 ran 50 epochs, 5FW2Eaae ran 58.  This
    # is the one type where the abscissa error also corrupted the ANCHOR, which
    # is why flux depth moves DOWN here while every other type moves up.
    ("db5fefc5", 2, "flux", "product", 15, 13, 0.75, 650, 726, 754, 151,
     873, (0, 15), {(1195, 896): 15}),
    ("241cda6c", 3, "flux", "product", 15, 13, 0.75, 754, 726, 754, 151,
     873, (0, 15), {(1195, 896): 15}),
    ("db9f7244", 3, "krea2", "design", 43, 38, 1.0, 2012, 1172, 1860, 373,
     758, (26, 43), {(768, 1376): 18, (1408, 768): 17, (1376, 768): 8}),
    ("ff643470", 4, "qwen-image", "social", 41, 36, 1.5, 1095, 1027, 1023, 205,
     887, (41, 41), {(1024, 768): 41}),
    ("1365fa1c", 5, "ideogram4", "product", 14, 12, 0.75, 174, 99, 414, 83,
     None, None, {(1195, 896): 13, (1376, 768): 1}),
    ("3e0fdcde", 5, "krea2", "design", 42, 37, 1.0, 2012, 1172, 1843, 369,
     887, (42, 42), {(1024, 768): 42}),
    ("4782f46f", 5, "qwen-image", "logo", 31, 27, 1.5, 949, 1027, 947, 190,
     747, (31, 31), {(1408, 768): 31}),
    ("b2582457", 5, "z-image", "social", 48, 43, 1.0, 1317, 860, 1317, 264,
     887, (48, 48), {(1024, 768): 48}),
    # 1300 -> 1100 (2026-08-06 pre-tournament re-derivation).  The rank-1
    # 5GU4Xkd3 trained to 1200 and published a `checkpoints/last.safetensors`
    # whose LFS oid is EQUAL to its own `last_000001100.safetensors`, i.e. it
    # SELECTED and shipped the 1100 rung.  Neither 1300 nor ">=1200" is what any
    # artifact contains.  ideogram4 is excluded from the winner-ratio assertions
    # below, so this column is documentation for this row — but it was wrong.
    ("b72da8c6", 5, "ideogram4", "style", 40, 36, 1.0, 1100, 171, 589, 118,
     None, None, {(1024, 768): 40}),
    ("f6725c2b", 5, "krea2", "design", 50, 45, 1.0, 2012, 1172, 1974, 395,
     887, (50, 50), {(1024, 768): 50}),
]

# The validator's split rule, OBSERVED to hold on all fourteen rows above.  A
# helper rather than a comment so the claim is executable.
def _validator_n_train(n_pairs):
    return n_pairs - math.ceil(0.10 * n_pairs)

TEMPLATE_RESOLUTION = [512, 768, 1024]
IDS = [f"{row[0]}-{row[2]}" for row in REAL_TASKS]


@dataclass(frozen=True)
class _GeoSpec(ImageSpec):
    """ImageSpec with a redirectable dataset dir (the real one is /dataset/images)."""

    images_dir: str = ""

    @property
    def dataset_images_dir(self) -> str:
        return self.images_dir


@pytest.fixture(scope="module")
def dataset_dirs(tmp_path_factory):
    """One flat image+caption dir per real task, at the real source dimensions."""
    root = tmp_path_factory.mktemp("week6-shapes")
    out = {}
    for row in REAL_TASKS:
        task, dims = row[0], row[13]
        d = root / task
        d.mkdir()
        index = 0
        for (width, height), count in sorted(dims.items()):
            for _ in range(count):
                Image.new("RGB", (width, height), (7, 9, 11)).save(
                    d / f"{index:03d}.png", compress_level=1
                )
                (d / f"{index:03d}.txt").write_text("a caption")
                index += 1
        out[task] = str(d)
    return out


def _build(task, images_dir, *, geometry_types=None, monkeypatch=None):
    row = next(r for r in REAL_TASKS if r[0] == task)
    _t, _r, model_type, _fam, _n_pairs, n_train, hours = row[:7]
    if geometry_types is None:
        monkeypatch.delenv("FORGE_EVAL_GEOMETRY_TYPES", raising=False)
    else:
        monkeypatch.setenv("FORGE_EVAL_GEOMETRY_TYPES", geometry_types)
    spec = _GeoSpec(
        task_id=task,
        model="rayonlabs/Test-Base",
        model_type=model_type,
        expected_repo_name=f"tournament-week6-{task}",
        images_dir=images_dir,
    )
    # n_train, NOT n_pairs.  `forge/tasks/aitoolkit.py:97` passes the count of
    # pairs it unpacked from train_data.zip; passing the auditing record's N
    # here is what let this suite certify depths the container never emitted.
    return build_config(spec, num_images=n_train, hours_to_complete=hours)


# --------------------------------------------------------------------------- #
# 1. The policy tables themselves — pinned to their field evidence.
# --------------------------------------------------------------------------- #
def test_step_table_is_the_week6_field_calibration():
    """The exact table. Changing a number here must be a deliberate, argued act."""
    assert recipe.STEP_TABLE == {
        # 1100 -> 1024.  The rank-1 miner ran 58 kohya EPOCHS over the 13 images
        # in train_data.zip = 754 steps, not the 870 that 58 x N=15 implies.
        # 1024*sqrt(13/24) = 754.  Both the abscissa AND the anchor were wrong,
        # and they pointed opposite ways: at 1100 we emitted 810, i.e. 7.4% DEEP
        # of the anchor, while looking 6.9% short of a documented 870 that no
        # miner ever ran.
        "flux": dict(base=1024, n_ref=24, p=0.50, min=500, max=2000),
        # 5FBmn1ax's krea2 policy is pure clock-fill: 2012 steps on N=42, 43 AND
        # 50 (8x the size range, identical depth), 1432 at h=0.75.
        # 1500 -> 1584 = 1432/(18/24)^0.35: the R1 task hands the container 18
        # images, so 1500 emitted 1356 while this file asserted 1432.
        "krea2": dict(base=1584, n_ref=24, p=0.35, min=600, max=2200),
        # NOT a fit to the field: the champion runs lr 4e-4 constant and we run
        # 2.5e-5 cosine, so his step counts are not transferable (28.5x less lr
        # integral at matched steps).  Set instead from our own EMA attenuation
        # and our own do_cfg clock ceiling, and normalised to the ONE in-family
        # field winner.  500 -> 517 = 378/(9/24)^0.32 — an abscissa correction
        # only; p/min/max are untouched and the HELD adjudication is unchanged.
        "ideogram4": dict(base=517, n_ref=24, p=0.32, min=350, max=620),
        # Two INDEPENDENT rank-1 operators: 1317 and 1188 at n_train 43 and 35
        # imply base 983.9 / 983.8 at p=0.5 — agreement to 0.01%, TIGHTER than
        # the 0.11% the same two artifacts showed when fitted at N=48/39 (931 /
        # 932).  base 984 reproduces both winners EXACTLY.
        "z-image": dict(base=984, n_ref=24, p=0.50, min=350, max=1800),
        # 5FBmn1ax: 892*(n/24)^0.51 gives 947 and 1097 at n_train 27 and 36,
        # against his published 949 and 1095 (+-0.2%).  `p` held at 0.51; the
        # exponent re-recovered at the corrected abscissa is 0.497, worth <=0.4%
        # over the whole observed range.
        "qwen-image": dict(base=892, n_ref=24, p=0.51, min=300, max=1600),
    }


def test_the_validator_holdout_rule_that_sets_the_abscissa():
    """n_train == N - ceil(0.10*N) on all fourteen real Aug-3 tasks.

    OBSERVED, not modelled.  The `n_train` column was read out of each task's
    `train_data.zip` END-OF-CENTRAL-DIRECTORY over HTTP range requests (zip
    directory only; no image payload fetched, and the quarantined `test_data`
    archives were not touched).  Fourteen out of fourteen, no exceptions.

    This is pinned as a test because the entire refit rests on it: if the
    validator ever changes its holdout fraction, every `base` in STEP_TABLE is
    stale by (new/old)**p and NOTHING ELSE WOULD NOTICE.
    """
    for row in REAL_TASKS:
        task, n_pairs, n_train = row[0], row[4], row[5]
        assert n_train == _validator_n_train(n_pairs), task
        # The `ceil` makes the withheld share strictly more than 10% at small N:
        # the observed band is 0.857 (N=21 and N=14) to 0.900 (N=40 and N=50).
        assert 0.85 <= n_train / n_pairs <= 0.90, task

    # And the reason the fix is a recalibration rather than plumbing: the map is
    # NOT invertible, so a container holding n_train images cannot recover N.
    # The collision at 18 is our own R1 shape.
    collisions = {}
    for n_pairs in range(2, 120):
        collisions.setdefault(_validator_n_train(n_pairs), []).append(n_pairs)
    assert collisions[18] == [20, 21]
    assert any(len(v) > 1 for v in collisions.values())


def test_step_table_max_binding_sizes():
    """Each row's `max` must be honest about whether it can bind at all.

    Observed tournament dataset sizes are N = 9..50, i.e. n_train = 8..45, and
    THE CROSSOVER IS QUOTED IN n_train BECAUSE THAT IS THE LAW'S ARGUMENT.  A
    `max` that the law cannot reach within 4x that range is a ceiling on
    extrapolation, not a policy — which is what made the short-lived ideogram4
    `max=1600` an inert change (its law topped out at 365 at N=50).  Pinning the
    crossover here means a row that silently becomes decoration fails a test
    instead of reading as a decision.

    THE ABSCISSA REFIT DID NOT MOVE THREE OF THESE IN REAL TERMS.  ideogram4,
    qwen-image and z-image previously crossed at N = 48 / 85 / 90, and
    n_train(48)=43, n_train(85)=76, n_train(90)=81 — exactly the values below.
    Those three `max` values still bite at the same real dataset, which is the
    check that the refit changed units and not policy.  krea2 (72 -> 69 in N
    terms) and flux (80 -> 103) did move, because their refits were not pure
    unit changes; both stay far outside anything ever observed.
    """
    crossover = {}
    for model_type, row in recipe.STEP_TABLE.items():
        first = next(
            (
                n
                for n in range(1, 401)
                if row["base"] * (n / row["n_ref"]) ** row["p"] >= row["max"]
            ),
            None,
        )
        crossover[model_type] = first
    assert crossover == {
        # The one row whose `max` is ACTIVE at the top of the observed range
        # (n_train 43 == the N=48 task).
        "ideogram4": 43,
        # Backstops against pathological n, and labelled as such in recipe.py.
        "krea2": 62,
        "qwen-image": 76,
        "z-image": 81,
        "flux": 92,
    }
    # Nothing may be inert the way ideogram4's 1600 was: unreachable within 4x
    # the largest observed dataset.
    for model_type, first in crossover.items():
        assert first is not None and first <= 200, model_type


def test_discredited_jul16_premises_are_not_reintroduced():
    """The two false claims that caused the miscalibration must stay deleted.

    1. "deep training never helped" / "over-training is the #1 liability" — five
       published field reconstruction curves all bottom out at their DEEPEST
       checkpoint, and Spearman(steps, loss) inside the template family on the
       only 14-way field is -0.605.
    2. `do_differential_guidance` "adds a second guidance forward per step" — it
       is nested under `if self.train_config.do_guidance_loss:` and has never
       executed, so it cannot justify a slow per-step constant.
    """
    source = open(recipe.__file__, encoding="utf-8").read().lower()
    for claim in (
        "over-training is the #1 liability",
        "deep training never helped",
        "do_differential_guidance",
    ):
        start = 0
        while True:
            at = source.find(claim, start)
            if at < 0:
                break
            # The claim may only appear inside an explicit refutation, never as
            # a live justification for a number.
            window = source[max(0, at - 1200): at + 1200]
            assert (
                "falsified" in window
                or "unreachable" in window
                or "never executed" in window
            ), f"{claim!r} reappears in recipe.py without its refutation"
            start = at + 1


@pytest.mark.parametrize(
    ("model_type", "rate", "kind", "note"),
    [
        # Rates read out of the published artifacts with the SAME arithmetic the
        # module documents: W(h) = h*3600 - 225 (terminate trigger) - 300
        # (STARTUP_S), then W(h)/shipped.
        #
        # The two kinds are NOT interchangeable, and reading them as if they were
        # is how qwen-image ended up 14% optimistic:
        #   MEASURED — a run that was KILLED, or our own instrumented run.  This
        #              IS a rate.  SEC_PER_IT below it is planning work nobody
        #              has shown fits, so MEASURED sets a hard FLOOR.
        #   BOUND    — a run that COMPLETED.  It only says the true rate was at
        #              most this; the miner may have had time to spare.  A bound
        #              cannot set a floor.  It is used the other way round, as a
        #              limit on how conservative we are allowed to be.
        ("krea2", 1041.1 / 823, "MEASURED",
         "OUR OWN 823 steps in 1041.1 s on the tournament host (5HLA2QWY)"),
        ("qwen-image", 3975 / 850, "MEASURED",
         "5FW2Eaae and 5FpdSckw, identical configs, both killed at 850/1150"),
        ("z-image", 3075 / 2000, "BOUND",
         "5D2Qee4V completed 2000 steps in the same 1.0 h"),
        ("flux", 2175 / 870, "BOUND",
         "rank-1 5FW2Eaae, 58 kohya epochs x N=15, INFERRED"),
        ("ideogram4", 2 * 3075 / 1523, "BOUND",
         "5FBmn1ax completed 1523 in 1.0 h; DOUBLED for our do_cfg batch-2 step"),
    ],
)
def test_sec_per_it_is_never_faster_than_its_own_evidence(
    model_type, rate, kind, note
):
    """A rate constant may pad its evidence; it may never outrun a measurement.

    Direction matters and is easy to get backwards: a SMALLER constant means
    MORE planned steps.  Both failure modes are bounded here — being under a
    MEASURED rate is what would have got qwen killed on two of three real shapes,
    and being far over any of them is what threw away ~40% of every krea2 and
    z-image budget for five weeks.
    """
    sec = recipe.SEC_PER_IT[model_type]
    if kind == "MEASURED":
        assert sec >= rate, (
            f"{model_type}: {sec} s/step is FASTER than the MEASURED "
            f"{rate:.3f} — {note}"
        )
    assert sec / rate <= 1.20, (
        f"{model_type}: {sec / rate:.3f}x its evidence — that much pad is how "
        f"depth was thrown away for five weeks ({note})"
    )


def test_krea2_rate_sits_between_our_measurement_and_the_field_bound():
    """The one type with BOTH kinds of evidence, and they bracket the constant.

    Our own instrumented run says 1.265 s/step; the field's tightest krea2 bound
    says the champion was no slower than 1.519 on the same shape.  1.35 is inside
    that bracket: padded over what we measured, and still faster than the slowest
    rate consistent with a completed field run — so the clock never truncates the
    size law while remaining a rate the field has shown is achievable.
    """
    ours = 1041.1 / 823
    field_bound = 2175 / 1432  # 5FBmn1ax/5FjDsFGA completed 1432 in 0.75 h
    assert ours < recipe.SEC_PER_IT["krea2"] < field_bound


def test_krea2_sec_per_it_pads_our_own_measurement():
    """5HLA2QWY (us) published 1041.1 s for 823 steps = 1.265 s/step.

    That 1041.1 s is `toolkit_start -> toolkit_end`, i.e. GROSS OF STARTUP, so
    the constant is padded twice: 1.35 is 6.7% over the gross rate, and the
    budget model then charges STARTUP_S = 300 s on top of it.  Net of a 300 s
    startup the same artifact implies 0.90 s/step.
    """
    gross = 1041.1 / 823
    assert recipe.SEC_PER_IT["krea2"] > gross
    assert recipe.SEC_PER_IT["krea2"] / gross == pytest.approx(1.07, abs=0.02)
    net_of_startup = (1041.1 - recipe.STARTUP_S) / 823
    assert recipe.SEC_PER_IT["krea2"] / net_of_startup == pytest.approx(1.50, abs=0.03)


def test_krea2_rate_makes_the_size_law_bind_not_the_clock():
    """The krea2 1.5 -> 1.35 decision, restated as the property it buys.

    1.5 was the only one of the three candidates (1.5 / 1.35 / 1.30) that let
    the clock truncate the law, and it truncated exactly the R1 shape: 1432 (two
    field operators completed exactly that depth there) down to 1336, a depth
    nobody in the field ran.  1.35 and 1.30 are identical in output, so 1.30's
    extra optimism buys nothing and costs `projected_wall_s` 4% of its honesty.
    """
    thresholds = []
    for row in REAL_TASKS:
        if row[2] != "krea2":
            continue
        n_train, hours, after = row[5], row[6], row[9]
        law = _pure_law("krea2", n_train)
        assert after == law, f"{row[0]}: clock truncated the krea2 law to {after}"
        window = hours * 3600.0 * recipe.margin_for("krea2") - 480.0
        thresholds.append(window / law)
    # The largest rate at which every real krea2 shape is still size-bound.
    # 0.75 h/N=21 gives 1.399 and 1.0 h/N=50 gives 1.461, so the binding
    # threshold is 1.399 and 1.35 sits 3.6% inside it.
    assert min(thresholds) == pytest.approx(1.399, abs=0.005)
    assert recipe.SEC_PER_IT["krea2"] < min(thresholds)


def test_ideogram4_sec_per_it_is_deliberately_above_the_field_bound():
    """The field's 2.05 s/step bound does NOT apply to our ideogram4 config.

    `forge.ideogram_release_policy` sets `do_cfg: true`, which runs the
    transformer at batch 2 every step and adds a second grad-enabled forward
    through the 8B text encoder (PIPELINE-MATERIALIZATION-AUDIT D6) — roughly
    2x the field's per-step cost. Being honest about that costs nothing: the
    size law binds on all three real ideogram4 shapes either way.
    """
    assert recipe.SEC_PER_IT["ideogram4"] == pytest.approx(2.05 * 2, abs=0.15)
    for row in REAL_TASKS:
        if row[2] != "ideogram4":
            continue
        n_train, hours, after = row[5], row[6], row[9]
        law = _pure_law("ideogram4", n_train)
        cap = _clock_cap("ideogram4", hours)
        assert law < cap, "ideogram4 must be size-bound, not clock-bound"
        assert after == law


def test_margin_stops_double_counting_the_fixed_reserve():
    """0.85 applied a 15% haircut ON TOP OF a 480 s fixed reserve."""
    assert recipe.MARGIN == 0.92
    assert recipe.STARTUP_S + recipe.EXPORT_RESERVE_S == 480.0
    # The champion's recovered clock model is (budget - 478) with no
    # multiplicative margin at all; 0.92 keeps ~290 s of jitter headroom on a
    # 1.0 h task BEYOND that reserve.
    headroom = 3600.0 * (1 - recipe.MARGIN)
    assert 250.0 <= headroom <= 400.0


def test_margin_is_per_type_because_the_headroom_it_spends_is_per_type():
    """0.92 GLOBALLY was a regression, and it landed on exactly one type.

    Four of the five rows have a size law that binds well below the clock, so
    their margin is inert.  qwen-image is the one type where the clock is the
    active constraint AND whose rate constant carries no pad, so a global
    +8% margin went straight into planned steps: the qwen cap rose 1027 -> 1122
    and two of the three real qwen shapes stopped being able to finish.  The
    guard against a repeat is `test_every_shape_finishes_at_its_field_rate`.
    """
    assert set(recipe.MARGIN_BY_TYPE) == set(recipe.SEC_PER_IT)
    for model_type, margin in recipe.MARGIN_BY_TYPE.items():
        assert recipe.margin_for(model_type) == margin
        assert 0.80 <= margin <= 0.99
    # Unknown types and junk fall back to the default rather than raising.
    assert recipe.margin_for("not-a-type") == recipe.MARGIN
    assert recipe.margin_for(None) == recipe.MARGIN
    # Only qwen-image departs from the default, and it departs upward, because
    # its SEC_PER_IT is the measurement rather than a padded one.
    departures = {
        model_type: margin
        for model_type, margin in recipe.MARGIN_BY_TYPE.items()
        if margin != recipe.MARGIN
    }
    assert departures == {"qwen-image": 0.98}
    # A margin of 1 - STOP_MARGIN_S/budget would plan exactly to the terminate
    # trigger; qwen's 0.98 is strictly inside that, i.e. it is a cushion.
    for hours in (1.25, 1.5):
        plan_to_the_wall = 1.0 - recipe.STOP_MARGIN_S / (hours * 3600.0)
        assert recipe.margin_for("qwen-image") < plan_to_the_wall


def test_stop_margin_mirrors_the_production_terminate_gate():
    """`aitoolkit._run_toolkit` terminates at `remaining() <= boundary_margin_s()`.

    `recipe.training_deadline_s` is only the real deadline if this stays equal to
    the value the runner actually gates on; drift would silently make every
    finish/kill projection in this file optimistic.
    """
    from forge.tasks import aitoolkit, holdout

    assert recipe.STOP_MARGIN_S == holdout.boundary_margin_s()
    assert recipe.STOP_MARGIN_S == aitoolkit._STOP_MARGIN_S
    assert recipe.training_deadline_s(1.0) == 3600.0 - 180.0 - 45.0


def _pure_law(model_type, pairs):
    row = recipe.STEP_TABLE[model_type]
    scaled = row["base"] * (pairs / row["n_ref"]) ** row["p"]
    return int(round(max(row["min"], min(row["max"], scaled))))


def _clock_cap(model_type, hours):
    train_s = (
        hours * 3600.0 * recipe.margin_for(model_type)
        - recipe.STARTUP_S
        - recipe.EXPORT_RESERVE_S
    )
    return int(train_s / recipe.SEC_PER_IT[model_type])


# --------------------------------------------------------------------------- #
# 2. Materialisation over the five types x the real task shapes.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("row", REAL_TASKS, ids=IDS)
def test_materialized_steps(row, dataset_dirs, monkeypatch):
    """The real build_config, on the real shape, must ship the audited depth."""
    task, _rnd, model_type, _fam, _n_pairs, n_train, hours = row[:7]
    before, after = row[8], row[9]
    cfg = _build(task, dataset_dirs[task], monkeypatch=monkeypatch)
    steps = cfg["config"]["process"][0]["train"]["steps"]
    assert steps == after, f"{task}: expected {after}, got {steps} (was {before})"
    assert steps == recipe.size_scaled_steps(
        model_type, n_train, hours, cfg["config"]["process"][0]["train"]["steps"]
    )
    # ...and it is NOT what the law returns at the auditing record's N.  That
    # equality is the bug this column exists to prevent, so assert the
    # inequality wherever the two differ at all (they coincide only where the
    # clock cap binds both).
    at_full_n = recipe.size_scaled_steps(model_type, _n_pairs, hours, steps)
    assert at_full_n >= steps


@pytest.mark.parametrize("row", REAL_TASKS, ids=IDS)
def test_save_cadence_leaves_four_periodic_candidates(row, dataset_dirs, monkeypatch):
    task = row[0]
    after, save_every = row[9], row[10]
    cfg = _build(task, dataset_dirs[task], monkeypatch=monkeypatch)
    process = cfg["config"]["process"][0]
    assert process["save"]["save_every"] == save_every
    # ai-toolkit fires a numbered save at the START of a step when
    # step_num % save_every == 0 and step_num != 0, then an unnumbered exact
    # final after the loop (BaseSDTrainProcess.py:2332,2596-2601).
    periodic = (after - 1) // save_every
    assert periodic >= 3, f"{task}: only {periodic} mid-run recovery points"
    assert periodic <= 5


@pytest.mark.parametrize("row", REAL_TASKS, ids=IDS)
def test_projected_wall_clock_fits_the_budget(row, dataset_dirs, monkeypatch):
    """Planned wall clock, incl. startup + export reserve, must fit with margin."""
    task, hours, after = row[0], row[6], row[9]
    model_type = row[2]
    cfg = _build(task, dataset_dirs[task], monkeypatch=monkeypatch)
    steps = cfg["config"]["process"][0]["train"]["steps"]
    assert steps == after
    budget = hours * 3600.0
    margin = recipe.margin_for(model_type)
    wall = recipe.projected_wall_s(model_type, steps)
    assert wall <= budget * margin + 1.0, f"{task}: {wall:.0f}s of {budget:.0f}s"
    # `wall` already contains the 180 s export reserve, so the remainder is pure
    # jitter headroom.  MARGIN sets its floor.
    slack = budget - wall
    assert slack >= budget * (1.0 - margin) - 1.0
    # The load-bearing property is not a round number of seconds of slack — it is
    # that TRAINING ends before `_run_toolkit` terminates it, which happens
    # `EXPORT_RESERVE_S + STOP_MARGIN_S` before the hard kill.  Asserting the
    # magic 200 s instead is what let the qwen regression through: at the old
    # optimistic 4.0 s/step every qwen row cleared 200 s of "slack" that the real
    # rate did not have.
    train_end = recipe.STARTUP_S + steps * recipe.SEC_PER_IT[model_type]
    assert train_end <= recipe.training_deadline_s(hours), (
        f"{task}: training ends at {train_end:.0f}s, terminated at "
        f"{recipe.training_deadline_s(hours):.0f}s"
    )


@pytest.mark.parametrize("row", REAL_TASKS, ids=IDS)
def test_first_periodic_save_is_kill_safe(row, dataset_dirs, monkeypatch):
    """INV-2: a deadline stop must never find us with nothing exported.

    Periodic saves are the ONLY mid-run recovery point, so the projected time to
    the FIRST one has to stay deep inside the budget even when the box runs much
    slower than the policy models.  Asserted at the policy rate AND at 2x it.
    """
    task, hours, after, save_every = row[0], row[6], row[9], row[10]
    model_type = row[2]
    cfg = _build(task, dataset_dirs[task], monkeypatch=monkeypatch)
    process = cfg["config"]["process"][0]
    assert process["train"]["steps"] == after
    budget = hours * 3600.0
    first = recipe.first_save_wall_s(model_type, after, process["save"]["save_every"])
    assert first <= 0.35 * budget, f"{task}: first save at {100*first/budget:.0f}%"
    # At twice the modelled per-step cost the first candidate is still on disk
    # well before the soft stop (budget - 180 s export reserve).
    doubled = recipe.STARTUP_S + save_every * recipe.SEC_PER_IT[model_type] * 2
    assert doubled <= budget - recipe.EXPORT_RESERVE_S


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        # krea2 exceeds 1.0 only because of the R1 shape, where the rank-1
        # artifact is the shallowest thing on the task (1000) and six others
        # completed 1278-2000; our 1432 is the champion's own depth there.
        # 1.05 -> 1.06 in the abscissa refit: the three 1.0 h shapes each moved
        # ~5% closer to the 2012 they are measured against, and the R1 ratio is
        # pinned at exactly 1432/1000 by construction.
        ("krea2", 1.06),
        # EXACTLY 1.00 now, on both shapes, not on average: base 984 reproduces
        # 1188 at n_train 35 and 1317 at n_train 43 to the step.
        ("z-image", 1.00),
        # qwen 1.03 -> 0.975: the clock, calibrated to the field's own
        # reproduced 4.68 s/step, will not fund 1104 on ff643470.  Planning it
        # anyway ships 884 (see test_every_shape_finishes_at_its_field_rate).
        ("qwen-image", 0.975),
        ("flux", 1.08),
        # ideogram4 IS DELIBERATELY ABSENT.  Matching the field winners' STEP
        # COUNTS is not a valid target for this type: every field ideogram4
        # config runs lr 4e-4 constant and ours runs 2.5e-5 cosine, so equal
        # steps mean 28.5x less parameter movement.  Its depth is asserted
        # against our own EMA floor and clock ceiling instead — see
        # test_ideogram4_depth_is_set_by_our_own_pipeline_not_the_field.
    ],
)
def test_shipped_depth_now_tracks_the_field_winners(model_type, expected):
    ratios = [
        row[9] / row[7] for row in REAL_TASKS if row[2] == model_type
    ]
    assert sum(ratios) / len(ratios) == pytest.approx(expected, abs=0.01)


def test_every_type_moved_off_the_shallow_edge_of_its_band():
    """Before: only qwen-image was inside the winners' band. After: all five."""
    for row in REAL_TASKS:
        model_type, winner, before, after = row[2], row[7], row[8], row[9]
        if row[0] == "b72da8c6":
            continue  # the one uninformative task; see the parametrised test
        if model_type == "ideogram4":
            # Excluded on purpose: the field's ideogram4 step counts are not a
            # band we should be inside, because they were produced at 16x our
            # learning rate.  Asserted separately against our own constraints.
            continue
        assert after >= before or model_type == "qwen-image"
        # 41025fb5 is the one shape whose rank-1 artifact is also its SHALLOWEST
        # (1000 steps, against 1278/1400/1432/1432/1750/2000 from the rest of the
        # field), so "ratio to rank-1" is the wrong ceiling there: 1432/1000 =
        # 1.43 is the champion's own depth on that exact task.  Everywhere else
        # 1.40 still holds.
        ceiling = 1.45 if row[0] == "41025fb5" else 1.40
        assert 0.85 <= after / winner <= ceiling, f"{row[0]} {model_type}"


# --------------------------------------------------------------------------- #
# 2b. THE CLOCK GUARD.  No type may be planned above its own observed
#     throughput — the property the global MARGIN 0.85 -> 0.92 broke.
# --------------------------------------------------------------------------- #
def _shipped_after_stop(planned, stopped_at, save_every):
    """What lands in `last.safetensors` if training is terminated at `stopped_at`.

    Mirrors ai-toolkit + `forge/tasks/checkpoints.py`: a numbered save fires at
    the START of step k for every k % save_every == 0, and `_finalize` promotes
    the highest valid one.  A completed run additionally writes the unnumbered
    exact final, which wins.
    """
    if planned <= stopped_at:
        return planned
    return (stopped_at // save_every) * save_every


@pytest.mark.parametrize("row", REAL_TASKS, ids=IDS)
def test_every_shape_finishes_at_its_field_rate(row, dataset_dirs, monkeypatch):
    """The invariant SEC_PER_IT and MARGIN_BY_TYPE exist to satisfy.

    For every type x every real Aug-3 shape, the planned step count must still
    complete when the box runs at the SLOWEST rate that type's own published
    artifacts support.  `recipe.field_demonstrated_steps` is exact integer
    arithmetic over `FIELD_DEMONSTRATED_DEPTH`, so no float rounding can flip a
    verdict.

    This is the test that would have caught the regression.  With MARGIN 0.92
    applied globally and SEC_PER_IT["qwen-image"] left at 4.0, qwen planned 909
    on 7421f056 and 1104 on ff643470, against field-demonstrated windows of 850
    and 1042; the runs would have been terminated mid-flight and shipped their
    728- and 884-step periodic saves — 13% and 20% shallower than simply
    planning what fits.
    """
    task, model_type, hours, after, save_every = (
        row[0], row[2], row[6], row[9], row[10]
    )
    cfg = _build(task, dataset_dirs[task], monkeypatch=monkeypatch)
    steps = cfg["config"]["process"][0]["train"]["steps"]
    assert steps == after

    demonstrated = recipe.field_demonstrated_steps(model_type, hours)
    assert demonstrated is not None
    assert steps <= demonstrated, (
        f"{task} {model_type}: plans {steps} steps but the field only "
        f"demonstrates {demonstrated} in a {hours} h window — a stop would ship "
        f"{_shipped_after_stop(steps, demonstrated, save_every)}"
    )
    # ...and at the policy's own rate, which must never be the more optimistic
    # of the two (test_sec_per_it_is_never_faster_than_its_own_evidence).
    at_policy_rate = recipe.completed_steps_at_rate(
        hours, recipe.SEC_PER_IT[model_type]
    )
    assert steps <= at_policy_rate


def test_the_qwen_regression_is_pinned_as_a_counterexample():
    """Freeze the exact numbers, so "just bump MARGIN" cannot come back quietly.

    c424362 raised MARGIN to 0.92 for every type at once.  qwen-image is the only
    row whose clock actually binds, so it was the only row that moved, and it
    moved past what the field shows fits.

    THE NUMBERS MOVED IN THE ABSCISSA REFIT and that is expected: this is a
    counterfactual on TODAY's law ("what would the broken margin plan now"),
    not a recording of what commit c424362 emitted.  Fed the n_train the
    container actually receives, and with `base` refitted to match, the plans
    are 911 and 1097 where they used to read 909 and 1104.  The conclusion is
    unchanged and so is the mechanism: both still overrun the window the field
    demonstrated, and both still degrade to a periodic save rather than forfeit.
    """
    broken_margin, broken_sec = 0.92, 4.0
    n_train = {"7421f056": 25, "ff643470": 36}
    for task, hours, planned_then, stops_at, ships in (
        ("7421f056", 1.25, 911, 850, 732),
        ("ff643470", 1.5, 1097, 1042, 880),
    ):
        cap = int((hours * 3600.0 * broken_margin - 480.0) / broken_sec)
        law = _pure_law("qwen-image", n_train[task])
        assert min(law, cap) == planned_then
        assert recipe.field_demonstrated_steps("qwen-image", hours) == stops_at
        save_every = recipe.kill_safe_save_every(planned_then, 250)
        assert _shipped_after_stop(planned_then, stops_at, save_every) == ships
        # What ships now instead, having planned inside the window.
        now = recipe.size_scaled_steps("qwen-image", n_train[task], hours, 0)
        assert now <= stops_at and now > ships


@pytest.mark.parametrize("row", REAL_TASKS, ids=IDS)
def test_no_shape_can_forfeit_even_far_below_its_modelled_rate(row):
    """INV-2 at the refitted depths: a slow box DEGRADES, it never ships nothing.

    The abscissa refit made four of five types deeper, so the question "did we
    just buy a forfeit path?" has to be answered by replay rather than by
    argument.  Every real Aug-3 shape is run against the terminate gate at 1x,
    1.25x, 1.5x, 2x and 3x its type's policy rate — 3x is far outside anything
    any artifact supports (the worst field observation anywhere is qwen's 8.13
    s/step, 1.73x its policy 4.7) — and in every cell a numbered periodic save
    must already be on disk.

    THE ONE SHAPE THAT LOSES CUSHION IN THIS COMMIT is 41025fb5: planning 1432
    instead of 1356 moves the rate at which R1 truncates from 1.604 to 1.519
    s/step.  That is deliberate and is exactly the knife-edge recipe.py's
    SEC_PER_IT["krea2"] block documents — 1432 fills the 0.75 h window by
    construction, because it IS the depth two operators completed there.  It
    still degrades to 1148 rather than forfeiting, which is what this asserts.
    """
    model_type, hours, planned, save_every = row[2], row[6], row[9], row[10]
    policy_rate = recipe.SEC_PER_IT[model_type]
    window = recipe.training_deadline_s(hours) - recipe.STARTUP_S

    for multiple in (1.0, 1.25, 1.5, 2.0, 3.0):
        rate = policy_rate * multiple
        reached = recipe.completed_steps_at_rate(hours, rate)
        shipped = _shipped_after_stop(planned, reached, save_every)
        assert shipped > 0, (
            f"{row[0]} {model_type}: at {multiple}x the policy rate "
            f"({rate:.2f} s/step) the run reaches step {reached} and the first "
            f"periodic save is at {save_every} — NOTHING WOULD BE EXPORTED"
        )
        # ...and it must land inside the window, not merely be scheduled.
        first_save_s = recipe.STARTUP_S + min(save_every, planned) * rate
        assert first_save_s <= recipe.training_deadline_s(hours), (
            f"{row[0]} {model_type}: at {multiple}x the first periodic save "
            f"projects to {first_save_s:.0f}s, past the terminate gate at "
            f"{recipe.training_deadline_s(hours):.0f}s"
        )
    assert window > 0

    # At 1x the plan must actually complete: that is the whole point of pairing
    # SEC_PER_IT with MARGIN_BY_TYPE.
    assert recipe.completed_steps_at_rate(hours, policy_rate) >= planned


def test_krea2_overrun_degrades_depth_instead_of_forfeiting(tmp_path):
    """Kill-safety for the 1.35 decision, exercised through the real finalizer.

    The R1 plan is 1432 steps with `save_every` 287.  If the box is slower than
    the field's tightest krea2 bound the run is terminated, and the only thing
    standing between us and an empty upload is `checkpoints.finalize` promoting
    the newest periodic save.  Verified here at the chosen rate AND at the two
    rejected candidates, so "is 1.35 kill-safe?" is answered by execution rather
    than by argument.
    """
    from forge.tasks import checkpoints

    pairs, hours, repo = 21, 0.75, "repo"
    for sec_per_it in (1.30, 1.35, 1.50):
        planned = min(
            _pure_law("krea2", pairs),
            int((hours * 3600.0 * recipe.margin_for("krea2") - 480.0) / sec_per_it),
        )
        save_every = recipe.kill_safe_save_every(planned, 250)
        # Terminate the run 1 step before the plan: the worst case that still
        # loses a whole save interval.
        stopped_at = planned - 1
        shipped = _shipped_after_stop(planned, stopped_at, save_every)
        assert 0 < shipped < planned

        root = tmp_path / f"sec-{sec_per_it}"
        root.mkdir()
        state = checkpoints.ensure_run(str(root), repo)
        state = checkpoints.set_planned_steps(str(root), state, planned)
        newest = None
        for step in range(save_every, stopped_at + 1, save_every):
            newest = _write_st(root / f"{repo}_{step:09d}.safetensors", tag=str(step))
        record = checkpoints.finalize(str(root), repo, state)
        assert record is not None, f"{sec_per_it}: nothing exported after a stop"
        assert (root / "last.safetensors").read_bytes() == newest
        assert record["selected_step"] == shipped
        # The degraded outcome is still deeper than the 823 steps we shipped on
        # Aug-3 — the downside of planning deep is bounded, not catastrophic.
        assert shipped > 823


def _write_st(path, tag=""):
    """Minimal valid one-tensor safetensors file (mirrors tests/test_units.py)."""
    import json as _json
    import struct as _struct

    header = _json.dumps(
        {
            "__metadata__": {"tag": tag},
            "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        }
    ).encode()
    path.write_bytes(_struct.pack("<Q", len(header)) + header + _struct.pack("<f", 0.0))
    return path.read_bytes()


# --------------------------------------------------------------------------- #
# 3. Geometry — the evaluator/ai-toolkit ports, and the resolution fields.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # Every distinct source shape in the Aug-3 harvest, run through the
        # evaluator's own adjust_image_size (long edge 1024, floor/16, crop).
        ((1024, 768), (1024, 768)),
        ((1195, 896), (1024, 752)),
        ((1408, 768), (1024, 544)),
        ((1376, 768), (1024, 560)),
        ((768, 1376), (560, 1024)),
        # Square and portrait-square control points.
        ((1024, 1024), (1024, 1024)),
        ((2048, 2048), (1024, 1024)),
    ],
)
def test_evaluator_size_matches_the_pinned_evaluator(source, expected):
    assert geometry.evaluator_size(*source) == expected


@pytest.mark.parametrize(
    ("source", "resolution", "divisibility", "expected"),
    [
        # The incumbent 3-copy list on the R1 krea2 shape: only @1024 matches.
        ((1024, 768), 512, 16, (576, 448)),
        ((1024, 768), 768, 16, (896, 656)),
        ((1024, 768), 1024, 16, (1024, 768)),
        # The wide-source failure mode: the @1024 bucket keeps a ~1400 px long
        # edge where the evaluator scores at 1024 — 1.88x the pixel count.
        ((1408, 768), 1024, 16, (1392, 752)),
        # ...and the geometry policy's chosen resolution reproduces the
        # evaluator exactly for the same source.
        ((1408, 768), 747, 16, (1024, 544)),
        ((1024, 768), 887, 16, (1024, 768)),
    ],
)
def test_bucket_size_matches_pinned_ai_toolkit(
    source, resolution, divisibility, expected
):
    assert geometry.bucket_size(*source, resolution, divisibility) == expected


def test_bucket_divisibility_table_matches_the_pinned_archs():
    """data_loader.py:395 overwrites our bucket_tolerance with these values."""
    assert geometry.BUCKET_DIVISIBILITY == {
        "krea2": 16,       # vae_scale_factor 8 * patch 2
        "ideogram4": 16,   # vae_scale_factor 8 * patch 2
        "z-image": 16,     # 8 * 2
        "qwen-image": 32,  # 16 * 2
        "flux": 32,        # legacy StableDiffusion: 2**3 * 2 (is_flux) * 2
    }


def test_geometry_policy_is_off_by_default(monkeypatch):
    """The ENTIRE Aug-3 field shipped the inherited list, winners included."""
    monkeypatch.delenv("FORGE_EVAL_GEOMETRY_TYPES", raising=False)
    for model_type in geometry.BUCKET_DIVISIBILITY:
        assert geometry.enabled_for(model_type) is False


def test_geometry_policy_opt_in_is_per_type(monkeypatch):
    monkeypatch.setenv("FORGE_EVAL_GEOMETRY_TYPES", "krea2, z-image")
    assert geometry.enabled_for("krea2") is True
    assert geometry.enabled_for("z-image") is True
    assert geometry.enabled_for("qwen-image") is False
    assert geometry.enabled_for("flux") is False
    monkeypatch.setenv("FORGE_EVAL_GEOMETRY_TYPES", "*")
    assert geometry.enabled_for("qwen-image") is True
    # ...but never ideogram4, at any setting.
    assert geometry.enabled_for("ideogram4") is False


@pytest.mark.parametrize("row", REAL_TASKS, ids=IDS)
def test_resolution_fields_default_off(row, dataset_dirs, monkeypatch):
    """With the switch unset every type keeps the template's 3-copy list."""
    task = row[0]
    cfg = _build(task, dataset_dirs[task], monkeypatch=monkeypatch)
    dataset = cfg["config"]["process"][0]["datasets"][0]
    assert dataset["resolution"] == TEMPLATE_RESOLUTION
    assert "bucket_tolerance" not in dataset


@pytest.mark.parametrize("row", REAL_TASKS, ids=IDS)
def test_resolution_fields_with_policy_on(row, dataset_dirs, monkeypatch):
    """With the switch on, one scalar resolution + the true bucket divisibility."""
    task, model_type = row[0], row[2]
    expected_resolution, coverage = row[11], row[12]
    cfg = _build(task, dataset_dirs[task], geometry_types="*", monkeypatch=monkeypatch)
    dataset = cfg["config"]["process"][0]["datasets"][0]
    if expected_resolution is None:  # ideogram4: structurally excluded
        assert dataset["resolution"] == TEMPLATE_RESOLUTION
        assert "bucket_tolerance" not in dataset
        return
    assert dataset["resolution"] == expected_resolution
    assert isinstance(dataset["resolution"], int)  # scalar, NOT a one-item list
    assert dataset["bucket_tolerance"] == geometry.BUCKET_DIVISIBILITY[model_type]
    plan = geometry.plan(model_type, geometry.measure_images(dataset_dirs[task]))
    assert (plan["on_geometry"], plan["images"]) == coverage


def test_geometry_policy_lifts_on_geometry_share_across_all_shapes():
    """19.0% -> 89.2%, and 3x fewer materialised samples. Both are the point."""
    incumbent_hits = incumbent_total = 0
    policy_hits = policy_total = 0
    for row in REAL_TASKS:
        model_type, dims = row[2], row[13]
        divisibility = geometry.BUCKET_DIVISIBILITY[model_type]
        flat = [wh for wh, count in dims.items() for _ in range(count)]
        # Incumbent: three dataset copies, one per resolution entry.
        for resolution in TEMPLATE_RESOLUTION:
            for width, height in flat:
                incumbent_total += 1
                if geometry.bucket_size(
                    width, height, resolution, divisibility
                ) == geometry.evaluator_size(width, height):
                    incumbent_hits += 1
        chosen, matched, total = geometry.choose_resolution(flat, divisibility)
        policy_hits += matched
        policy_total += total
    assert (incumbent_hits, incumbent_total) == (270, 1419)
    assert round(100.0 * incumbent_hits / incumbent_total, 1) == 19.0
    assert (policy_hits, policy_total) == (422, 473)
    assert round(100.0 * policy_hits / policy_total, 1) == 89.2
    # One dataset copy instead of three.
    assert policy_total * 3 == incumbent_total


def test_flux_residual_mismatch_is_structural_and_named():
    """flux stays at 0% on-geometry and the plan says exactly why.

    Its 1195x896 sources score at 1024x752, and 752 is not a multiple of 32 —
    the bucket divisibility the LEGACY StableDiffusion class forces for flux.
    No resolution can reach it. The policy still helps: 992x768 is 1.01x the
    scored pixel count where the incumbent @1024 copy is 1.34x.
    """
    dims = [(1195, 896)] * 15
    plan = geometry.plan("flux", dims)
    assert plan["on_geometry"] == 0
    cohort = plan["cohorts"][0]
    assert cohort["evaluator"] == "1024x752"
    assert cohort["bucket"] == "992x768"
    assert cohort["unreachable_reason"] == "evaluator_size_not_multiple_of_32"
    scored = 1024 * 752
    assert 992 * 768 / scored == pytest.approx(0.99, abs=0.02)
    incumbent = geometry.bucket_size(1195, 896, 1024, 32)
    assert incumbent[0] * incumbent[1] / scored == pytest.approx(1.34, abs=0.02)


def test_single_scalar_beats_a_per_aspect_ratio_resolution_list():
    """The rejected alternative, on the worst real shape (R3 krea2, 3 ratios).

    A K-entry list gives every image one on-geometry copy but K-1 off-geometry
    ones, so the SHARE falls to 1/K while the sample count multiplies by K.
    """
    dims = [(768, 1376)] * 18 + [(1408, 768)] * 17 + [(1376, 768)] * 8
    scalar, matched, total = geometry.choose_resolution(dims, 16)
    assert (scalar, matched, total) == (758, 26, 43)
    per_ratio = sorted(
        {
            geometry.choose_resolution([wh], 16)[0]
            for wh in set(dims)
        }
    )
    list_hits = sum(
        1
        for resolution in per_ratio
        for width, height in dims
        if geometry.bucket_size(width, height, resolution, 16)
        == geometry.evaluator_size(width, height)
    )
    list_total = len(per_ratio) * len(dims)
    assert matched / total > list_hits / list_total
    assert total < list_total


# --------------------------------------------------------------------------- #
# 4. ideogram4's structural exclusion, and INV-1 degradation.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "task", [row[0] for row in REAL_TASKS if row[2] == "ideogram4"]
)
def test_ideogram4_release_policy_survives_the_geometry_switch(
    task, dataset_dirs, monkeypatch
):
    """Changing ideogram4's resolution would silently delete the Week-5 recipe.

    `ideogram_release_policy._EXPECTED_RECIPE` hash-binds
    `dataset.resolution == [512, 768, 1024]`. Drift it and `apply()` becomes a
    no-op (lr reverts 2.5e-5 -> 1e-4, EMA/cosine/do_cfg/cache_latents all
    vanish); apply it afterwards instead and `checkpoint_control()` RAISES on
    the drifted projection, which forfeits the task to the untrained fallback.
    `POLICY_SHA256` is bound into the activation record, so `_EXPECTED_RECIPE`
    cannot simply be edited either.
    """
    cfg = _build(task, dataset_dirs[task], geometry_types="*", monkeypatch=monkeypatch)
    process = cfg["config"]["process"][0]
    assert process["datasets"][0]["resolution"] == TEMPLATE_RESOLUTION
    assert "bucket_tolerance" not in process["datasets"][0]
    # The policy fired and its checkpoint binding still validates.
    assert cfg["meta"]["forge_ideogram_production_policy"]["policy_id"]
    assert process["train"]["lr"] == 0.000025
    assert process["train"]["do_cfg"] is True
    control, selected = ideogram_release_policy.checkpoint_control(cfg)
    assert control["fraction_numerator"] == 1
    assert selected == process["train"]["steps"]


def test_geometry_degrades_to_the_template_when_it_cannot_measure(
    tmp_path, monkeypatch
):
    """INV-1: an unreadable/empty dataset dir leaves the template untouched."""
    monkeypatch.setenv("FORGE_EVAL_GEOMETRY_TYPES", "*")
    empty = tmp_path / "empty"
    empty.mkdir()
    for images_dir in (str(empty), str(tmp_path / "does-not-exist")):
        spec = _GeoSpec(
            task_id="degraded",
            model="rayonlabs/Test-Base",
            model_type="krea2",
            expected_repo_name="tournament-week6-degraded",
            images_dir=images_dir,
        )
        # 21 -> 18: the R1 task's n_train, i.e. what the container is handed.
        cfg = build_config(spec, num_images=18, hours_to_complete=0.75)
        dataset = cfg["config"]["process"][0]["datasets"][0]
        assert dataset["resolution"] == TEMPLATE_RESOLUTION
        assert "bucket_tolerance" not in dataset
        # ...and the depth policy still materialised normally.
        assert cfg["config"]["process"][0]["train"]["steps"] == 1432


def test_geometry_entry_points_never_raise():
    """Every public entry point degrades instead of raising (INV-1)."""
    assert geometry.plan("krea2", []) is None
    assert geometry.plan("not-a-type", [(1024, 768)]) is None
    assert geometry.plan("krea2", [(0, 0)]) is None
    assert geometry.plan(None, [(1024, 768)]) is None
    assert geometry.measure_images("/definitely/not/a/path") == []
    with pytest.raises(ValueError):  # the low-level port is allowed to be strict
        geometry.evaluator_size(0, 10)


def test_recipe_helpers_never_raise():
    assert recipe.projected_wall_s("krea2", "not-an-int") > 0
    assert recipe.projected_wall_s(None, 100) > 0
    assert recipe.first_save_wall_s("krea2", None, None) > 0
    assert recipe.size_scaled_steps("krea2", None, None, 2000) == 2000
    assert recipe.margin_for(object()) == recipe.MARGIN
    assert recipe.training_deadline_s("not-a-number") == 0.0
    assert recipe.training_deadline_s(-5) == 0.0
    assert recipe.completed_steps_at_rate(1.0, 0) == 0
    assert recipe.completed_steps_at_rate("x", 1.0) == 0
    assert recipe.field_demonstrated_steps("not-a-type", 1.0) is None
    assert recipe.field_demonstrated_steps(None, 1.0) is None
    assert recipe.field_demonstrated_steps("krea2", 0.0) == 0


def test_projected_wall_is_the_inverse_of_the_clock_cap():
    """The cap and the projection must not drift apart."""
    for model_type in recipe.SEC_PER_IT:
        for hours in (0.75, 1.0, 1.25, 1.5):
            margin = recipe.margin_for(model_type)
            cap = _clock_cap(model_type, hours)
            wall = recipe.projected_wall_s(model_type, cap)
            assert wall <= hours * 3600.0 * margin + 1e-6
            assert math.isclose(
                wall + recipe.SEC_PER_IT[model_type],
                hours * 3600.0 * margin,
                abs_tol=recipe.SEC_PER_IT[model_type] + 1e-6,
            )
