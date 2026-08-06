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
# task / round / model_type / family / n_pairs / hours_to_complete
#   / rank-1 miner's shipped steps
#   / steps forge shipped on the Aug-3 pin 084ea914 (BEFORE)
#   / steps forge ships now (AFTER)
#   / save_every / expected single-scalar resolution under the geometry policy
#   / (on-geometry images, total images) under that resolution
#   / source dimension histogram, straight out of the harvest task-meta.json
#
# `None` resolution == the type is structurally excluded from the geometry
# policy (ideogram4; see forge/geometry.py) and must keep the template list.
# --------------------------------------------------------------------------- #
REAL_TASKS = [
    ("41025fb5", 1, "krea2", "design", 21, 0.75, 1000, 824, 1336, 268,
     887, (21, 21), {(1024, 768): 21}),
    ("7421f056", 2, "qwen-image", "design", 28, 1.25, 850, 836, 909, 182,
     887, (28, 28), {(1024, 768): 28}),
    ("84be9fcd", 2, "ideogram4", "style", 46, 1.0, 341, 194, 616, 124,
     None, None, {(1408, 768): 45, (768, 1376): 1}),
    ("b290d171", 2, "z-image", "design", 39, 1.0, 1188, 860, 1186, 238,
     747, (37, 39), {(1408, 768): 37, (768, 1376): 2}),
    ("db5fefc5", 2, "flux", "product", 15, 0.75, 750, 726, 870, 175,
     873, (0, 15), {(1195, 896): 15}),
    ("241cda6c", 3, "flux", "product", 15, 0.75, 870, 726, 870, 175,
     873, (0, 15), {(1195, 896): 15}),
    ("db9f7244", 3, "krea2", "design", 43, 1.0, 2012, 1172, 1840, 369,
     758, (26, 43), {(768, 1376): 18, (1408, 768): 17, (1376, 768): 8}),
    ("ff643470", 4, "qwen-image", "social", 41, 1.5, 1095, 1027, 1104, 221,
     887, (41, 41), {(1024, 768): 41}),
    ("1365fa1c", 5, "ideogram4", "product", 14, 0.75, 174, 107, 421, 85,
     None, None, {(1195, 896): 13, (1376, 768): 1}),
    ("3e0fdcde", 5, "krea2", "design", 42, 1.0, 2012, 1172, 1825, 366,
     887, (42, 42), {(1024, 768): 42}),
    ("4782f46f", 5, "qwen-image", "logo", 31, 1.5, 949, 1027, 957, 192,
     747, (31, 31), {(1408, 768): 31}),
    ("b2582457", 5, "z-image", "social", 48, 1.0, 1317, 860, 1315, 264,
     887, (48, 48), {(1024, 768): 48}),
    ("b72da8c6", 5, "ideogram4", "style", 40, 1.0, 1300, 181, 589, 118,
     None, None, {(1024, 768): 40}),
    ("f6725c2b", 5, "krea2", "design", 50, 1.0, 2012, 1172, 1888, 378,
     887, (50, 50), {(1024, 768): 50}),
]

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
        task, dims = row[0], row[12]
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
    _t, _r, model_type, _fam, pairs, hours = row[:6]
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
    return build_config(spec, num_images=pairs, hours_to_complete=hours)


# --------------------------------------------------------------------------- #
# 1. The policy tables themselves — pinned to their field evidence.
# --------------------------------------------------------------------------- #
def test_step_table_is_the_week6_field_calibration():
    """The exact table. Changing a number here must be a deliberate, argued act."""
    assert recipe.STEP_TABLE == {
        # UNCHANGED: uncapped this returns 870 at N=15 — exactly the rank-1
        # miner's shipped depth on task 241cda6c. Evidence rated TOO THIN to move.
        "flux": dict(base=1100, n_ref=24, p=0.50, min=500, max=2000),
        # 5FBmn1ax's krea2 policy is pure clock-fill: 2012 steps on N=42, 43 AND
        # 50 (8x the size range, identical depth), 1432 at h=0.75.
        "krea2": dict(base=1500, n_ref=24, p=0.35, min=600, max=2200),
        # NOT a fit to the field: the champion runs lr 4e-4 constant and we run
        # 2.5e-5 cosine, so his step counts are not transferable (28.5x less lr
        # integral at matched steps).  Set instead from our own EMA floor
        # (0.995^T untrained init in every export) and our own do_cfg clock
        # ceiling.  See the recipe.py comment.
        "ideogram4": dict(base=500, n_ref=24, p=0.32, min=350, max=620),
        # Two INDEPENDENT rank-1 operators: 1317@N=48 and 1188@N=39 both imply
        # base 931/932 at p=0.5. Agreement to 0.1%.
        "z-image": dict(base=930, n_ref=24, p=0.50, min=350, max=1800),
        # 5FBmn1ax: steps = 834*(N/24)^0.51 reproduces 949 and 1095 exactly.
        "qwen-image": dict(base=840, n_ref=24, p=0.51, min=300, max=1600),
    }


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
    ("model_type", "field_bound", "note"),
    [
        # Hard upper bounds on the ACHIEVABLE rate: for any miner whose SHIPPED
        # steps reached their CONFIGURED steps, the run fit the budget, so
        # s/step <= (hours*3600 - 478)/shipped.
        ("krea2", 1.55, "our own measured rate on the tournament host is 1.265"),
        ("z-image", 1.56, "a field miner completed 2000 steps in the same 1.0 h"),
        ("qwen-image", 4.49, "the only constant the field does not contradict"),
        ("flux", 1.71, "INFERRED from 5D7iEJm5's 25.7 s/epoch at N=15"),
    ],
)
def test_sec_per_it_tracks_the_field_bound(model_type, field_bound, note):
    """The per-type constant must sit in a narrow band around the field bound.

    Direction matters and is easy to get backwards: a LARGER constant means
    FEWER planned steps.  Below the bound is aggressive (risks a deadline stop);
    far above it is what threw away ~40% of every krea2 and z-image budget for
    five weeks.  Both failure modes are bounded here.
    """
    ratio = recipe.SEC_PER_IT[model_type] / field_bound
    assert 0.88 <= ratio <= 1.20, f"{model_type}: {ratio:.3f}x the bound — {note}"


def test_krea2_sec_per_it_pads_our_own_measurement():
    """5HLA2QWY (us) published 1041.1 s for 823 steps = 1.265 s/step."""
    measured = 1041.1 / 823
    assert recipe.SEC_PER_IT["krea2"] > measured
    assert recipe.SEC_PER_IT["krea2"] / measured == pytest.approx(1.19, abs=0.02)


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
        pairs, hours, after = row[4], row[5], row[8]
        law = _pure_law("ideogram4", pairs)
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


def _pure_law(model_type, pairs):
    row = recipe.STEP_TABLE[model_type]
    scaled = row["base"] * (pairs / row["n_ref"]) ** row["p"]
    return int(round(max(row["min"], min(row["max"], scaled))))


def _clock_cap(model_type, hours):
    train_s = (
        hours * 3600.0 * recipe.MARGIN - recipe.STARTUP_S - recipe.EXPORT_RESERVE_S
    )
    return int(train_s / recipe.SEC_PER_IT[model_type])


# --------------------------------------------------------------------------- #
# 2. Materialisation over the five types x the real task shapes.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("row", REAL_TASKS, ids=IDS)
def test_materialized_steps(row, dataset_dirs, monkeypatch):
    """The real build_config, on the real shape, must ship the audited depth."""
    task, _rnd, model_type, _fam, pairs, hours = row[:6]
    before, after = row[7], row[8]
    cfg = _build(task, dataset_dirs[task], monkeypatch=monkeypatch)
    steps = cfg["config"]["process"][0]["train"]["steps"]
    assert steps == after, f"{task}: expected {after}, got {steps} (was {before})"
    assert steps == recipe.size_scaled_steps(
        model_type, pairs, hours, cfg["config"]["process"][0]["train"]["steps"]
    )


@pytest.mark.parametrize("row", REAL_TASKS, ids=IDS)
def test_save_cadence_leaves_four_periodic_candidates(row, dataset_dirs, monkeypatch):
    task = row[0]
    after, save_every = row[8], row[9]
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
    task, hours, after = row[0], row[5], row[8]
    model_type = row[2]
    cfg = _build(task, dataset_dirs[task], monkeypatch=monkeypatch)
    steps = cfg["config"]["process"][0]["train"]["steps"]
    assert steps == after
    budget = hours * 3600.0
    wall = recipe.projected_wall_s(model_type, steps)
    assert wall <= budget * recipe.MARGIN + 1.0, f"{task}: {wall:.0f}s of {budget:.0f}s"
    # `wall` already contains the 180 s export reserve, so the remainder is pure
    # jitter headroom.  MARGIN sets its floor; the tightest real shape (the
    # clock-bound krea2 tasks) keeps exactly that and nothing less.
    slack = budget - wall
    assert slack >= budget * (1.0 - recipe.MARGIN) - 1.0
    assert slack >= 200.0, f"{task}: only {slack:.0f}s of headroom"


@pytest.mark.parametrize("row", REAL_TASKS, ids=IDS)
def test_first_periodic_save_is_kill_safe(row, dataset_dirs, monkeypatch):
    """INV-2: a deadline stop must never find us with nothing exported.

    Periodic saves are the ONLY mid-run recovery point, so the projected time to
    the FIRST one has to stay deep inside the budget even when the box runs much
    slower than the policy models.  Asserted at the policy rate AND at 2x it.
    """
    task, hours, after, save_every = row[0], row[5], row[8], row[9]
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
        ("krea2", 1.02),
        ("z-image", 1.00),
        ("qwen-image", 1.03),
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
        row[8] / row[6] for row in REAL_TASKS if row[2] == model_type
    ]
    assert sum(ratios) / len(ratios) == pytest.approx(expected, abs=0.01)


def test_every_type_moved_off_the_shallow_edge_of_its_band():
    """Before: only qwen-image was inside the winners' band. After: all five."""
    for row in REAL_TASKS:
        model_type, winner, before, after = row[2], row[6], row[7], row[8]
        if row[0] == "b72da8c6":
            continue  # the one uninformative task; see the parametrised test
        if model_type == "ideogram4":
            # Excluded on purpose: the field's ideogram4 step counts are not a
            # band we should be inside, because they were produced at 16x our
            # learning rate.  Asserted separately against our own constraints.
            continue
        assert after >= before or model_type == "qwen-image"
        assert 0.85 <= after / winner <= 1.40, f"{row[0]} {model_type}"


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
    expected_resolution, coverage = row[10], row[11]
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
        model_type, dims = row[2], row[12]
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
        cfg = build_config(spec, num_images=21, hours_to_complete=0.75)
        dataset = cfg["config"]["process"][0]["datasets"][0]
        assert dataset["resolution"] == TEMPLATE_RESOLUTION
        assert "bucket_tolerance" not in dataset
        # ...and the depth policy still materialised normally.
        assert cfg["config"]["process"][0]["train"]["steps"] == 1336


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


def test_projected_wall_is_the_inverse_of_the_clock_cap():
    """The cap and the projection must not drift apart."""
    for model_type in recipe.SEC_PER_IT:
        for hours in (0.75, 1.0, 1.25, 1.5):
            cap = _clock_cap(model_type, hours)
            wall = recipe.projected_wall_s(model_type, cap)
            assert wall <= hours * 3600.0 * recipe.MARGIN + 1e-6
            assert math.isclose(
                wall + recipe.SEC_PER_IT[model_type],
                hours * 3600.0 * recipe.MARGIN,
                abs_tol=recipe.SEC_PER_IT[model_type] + 1e-6,
            )
