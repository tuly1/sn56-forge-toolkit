"""Week-6 guard: qwen-image LoRA rank is HELD at 32, and here is why.

WHY THIS FILE EXISTS
====================
The Aug-3 field ran qwen-image at network rank 128/141/149 while we ship 32.  A
blind expert audit called that "a plausible real 4x capacity deficit, unremarked
and unexplained".  It was investigated on 2026-08-07 (Unit 4) against the full
qwen record of BOTH retrievable tournaments and DELIBERATELY NOT CHANGED.

Full analysis: `SN56-project/SN56-WEEK6-QWEN-CAPACITY-ANALYSIS-2026-08-07.md`.
The load-bearing findings are re-stated here as executable assertions so the
decision cannot be silently reversed, and so that raising rank without also
re-pricing the clock FAILS instead of quietly truncating the run.

THE FOUR REASONS, EACH PINNED BY A TEST BELOW
---------------------------------------------
 1. RANK IS NOT WHAT SEPARATES WINNERS.  Across 7 qwen tasks in 2 tournaments,
    rank 128 took 5 of 7 first places; rank > 128 took 2, both by ONE operator
    (5FBmn1ax) on ONE day, against opponents DEADLINE-KILLED at 46-54% of plan.
    The only task where a high-rank and a low-rank entrant BOTH completed their
    configured plan is Jul-27 `5722b124`, and there rank 128 at 300 steps beat
    rank 152 at 1197 steps by 8.4%.
        -> test_lower_rank_won_the_only_completed_head_to_head
        -> test_rank_128_holds_the_majority_of_qwen_first_places

 2. ONE OF THE TWO HIGH-RANK WINS IS INSIDE THE NOISE FLOOR.  On `7421f056` two
    entrants published configs identical after removing hotkey-bearing paths
    (rank 128, lr 1e-4, steps 1150, save_every 50, keep 40, capdrop 0.05,
    wd 1e-5, adamw8bit, EMA 0.995, timestep weighted, res [512,768,1024]) and
    both were killed with their last save at step 850.  Their losses differ by
    2.145%.  The rank-149 win on `ff643470` was by 0.82%.
        -> test_high_rank_margin_is_inside_the_same_recipe_noise_floor

 3. THE COUPLED CHANGE IS MAGNITUDE-NEUTRAL BY CONSTRUCTION.  Every field config
    and ours sets `linear_alpha == linear`, so the LoRA scale is exactly 1.0.
    With B initialised to zero and Adam driving each entry at ~lr/step,
    dW = B@A sums r rank-1 terms, so ||dW||_F ~ sqrt(r)*lr*t.  That is precisely
    why the field's recovered law is lr ∝ 1/sqrt(rank): to hold ||dW|| invariant.
    So rank 32 -> 128 WITH the coupled lr buys only extra directions, and
    WITHOUT it is a 2x learning-rate change in disguise.
        -> test_alpha_equals_rank_so_the_magnitude_law_applies
        -> test_coupled_lr_at_rank_128_is_magnitude_neutral

 4. WE ARE NOT ACTUALLY BEHIND.  On E = lr*sqrt(rank)*steps the field's qwen
    winners span 0.339 .. 1.191; our three planned shapes give 0.473 / 0.541 /
    0.579 -- inside the band, 1.40x above the SHALLOWEST observed qwen winner.
        -> test_our_planned_effective_magnitude_sits_inside_the_winner_band

AND THE COST SIDE, SO A FUTURE RANK CHANGE CANNOT BE MADE UNPRICED
------------------------------------------------------------------
    -> test_artifact_size_law_reproduces_our_own_measured_rank32_artifact
    -> test_step_time_multiplier_for_rank_128
    -> test_raising_rank_requires_repricing_sec_per_it

EVIDENCE (all read-only, all re-checkable)
------------------------------------------
Aug-3 configs / ladders / LFS sizes / tensor shapes:
    SN56-project/evidence/week6-field-depth-audit-20260806/{raw,analysis.json}
Aug-3 scores, N, budgets:
    SN56-project/evidence/week6-tournament-dataset-harvest-20260806/tasks/
Jul-27 (recovered live 2026-08-07; the DATASETS are expired, the ARTIFACTS are not):
    api.gradients.io/tournament/tourn_0f7a1d3f6b5b66f9_20260727/details
    api.gradients.io/auditing/tasks/<task_id>
    huggingface.co/gradients-io-tournaments/
        tournament-tourn_0f7a1d3f6b5b66f9_20260727-<task_id>-<hotkey8>
        /raw/main/checkpoints/config.yaml  and  /api/models/<repo>/tree/main/checkpoints
Our own qwen throughput / VRAM / artifact bytes (H100 PCIe, rank 32, 1026/1026
steps completed naturally, NOT deadline-stopped):
    SN56-project/evidence/hyperstack-qwen-forge-operational-20260724/
        forge_run.public.json      toolkit_start t=4.1 -> toolkit_end t=2982.2
        final.json                 planned_steps 1026, selected_step 1026
        training-validation.json   rank 32, 840 pairs, 1680 tensors,
                                   logical_elements 294,912,000, 590,058,840 B
    SN56-project/SN56-HYPERSTACK-CAMPAIGN-RESULTS-2026-07-24.md:137-143
Scoring metric (validator pin b026da04):
    validator/evaluation/constants.py:25-32   qwen: steps 20, cfg 8, denoise 0.93
    validator/evaluation/evaluators/diffusion.py:203-209  plain per-pixel MSE
    validator/evaluation/evaluators/diffusion.py:311-314  blank-prompt branch
    validator/scoring/tasks.py:280-298        0.25*caption + 0.75*blank
"""

import math
import os

import pytest
import yaml

from forge import recipe

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN_TEMPLATE = os.path.join(
    REPO_ROOT, "forge", "templates", "base_diffusion_qwen_image.yaml"
)

# --- the three real Aug-3 qwen shapes (N, hours) --------------------------
# task-audit.json: image_text_pairs length and hours_to_complete.
REAL_QWEN_SHAPES = {
    "7421f056": (28, 1.25),
    "ff643470": (41, 1.50),
    "4782f46f": (31, 1.50),
}

# --- measured artifact geometry -------------------------------------------
# Recovered from the tensor shapes of
#   .../tournament-...-ff643470-...-5FBmn1ax/checkpoints/last.safetensors
# 60 blocks x 14 wrapped linears = 840 LoRA pairs / 1680 tensors.
LORA_PARAMS_PER_RANK = 9_216_000
# Byte model fitted on the field's four distinct qwen artifact sizes
# (r=128 -> 2,359,534,944 / r=141 -> 2,599,151,104 / r=149 -> 2,746,607,184 /
#  r=152 -> 2,801,903,200).  Slope agrees to 1e-5 across all four.
ARTIFACT_BYTES_PER_RANK = 18_432_010
ARTIFACT_BYTES_INTERCEPT = 237_664
# Our own rank-32 export, from training-validation.json.
OUR_RANK32_ARTIFACT_BYTES = 590_058_840
OUR_RANK32_LOGICAL_ELEMENTS = 294_912_000

# --- our own measured qwen throughput -------------------------------------
OUR_TOOLKIT_WINDOW_S = 2982.2 - 4.1  # forge_run.public.json
OUR_COMPLETED_STEPS = 1026  # final.json, stopped_by_deadline: false
OUR_SEC_PER_STEP_GROSS = OUR_TOOLKIT_WINDOW_S / OUR_COMPLETED_STEPS  # 2.9026

# --- the field record: (label, rank, lr, shipped_steps, placed_first) ------
# `shipped_steps` is None where the entrant published only last.safetensors
# with no ladder, so the kill cannot be bracketed.
QWEN_FIELD = [
    # Jul-27, tourn_0f7a1d3f6b5b66f9_20260727
    ("jul27 5722b124 5FpdSckw", 128, 1.00e-4, 300, True),
    ("jul27 5722b124 5GU4Xkd3", 152, 8.83e-5, 1197, False),
    ("jul27 78ca1088 5FpdSckw", 128, 1.00e-4, None, True),
    ("jul27 d21c5b80 5FpdSckw", 128, 1.00e-4, None, True),
    ("jul27 d21c5b80 5GKoYQm7", 128, 1.00e-4, None, False),
    ("jul27 deb3c5c6 5FW2Eaae", 128, 1.00e-4, None, True),
    ("jul27 deb3c5c6 5C7yZ5wg", 128, 1.00e-4, None, False),
    # Aug-3, tourn_c54bb970b5d0aa91_20260803
    ("aug03 7421f056 5FW2Eaae", 128, 1.00e-4, 850, True),
    ("aug03 7421f056 5FpdSckw", 128, 1.00e-4, 850, False),
    ("aug03 ff643470 5FBmn1ax", 149, 8.91e-5, 1095, True),
    ("aug03 ff643470 5FW2Eaae", 128, 1.00e-4, 700, False),
    ("aug03 4782f46f 5FBmn1ax", 141, 9.16e-5, 949, True),
    ("aug03 4782f46f 5GU4Xkd3", 128, 1.00e-4, 600, False),
]


def _template():
    with open(QWEN_TEMPLATE) as fh:
        return yaml.safe_load(fh)["config"]["process"][0]


def _effective_magnitude(rank, lr, steps):
    """E = lr * sqrt(rank) * steps.

    INFERRED, not measured.  With `alpha == rank` the LoRA scale is 1.0; with B
    zero-initialised and Adam driving each entry at ~lr per step, dW = B@A sums
    `rank` rank-1 terms of similar magnitude, so ||dW||_F ~ sqrt(rank)*lr*steps.
    This is the same relation the field's own lr ∝ 1/sqrt(rank) law exists to
    hold invariant, which is the evidence that the field believes it too.
    """
    return lr * math.sqrt(rank) * steps


# --------------------------------------------------------------------------
# 1. The template still says what this analysis was written about.
# --------------------------------------------------------------------------


def test_qwen_template_still_ships_rank_32_lr_1e4():
    """HELD on 2026-08-07 by Unit 4.  Changing these three numbers without
    re-reading SN56-WEEK6-QWEN-CAPACITY-ANALYSIS-2026-08-07.md is the failure
    this test exists to prevent."""
    proc = _template()
    assert proc["network"]["linear"] == 32
    assert proc["network"]["linear_alpha"] == 32
    assert proc["train"]["lr"] == pytest.approx(1e-4)


def test_alpha_equals_rank_so_the_magnitude_law_applies():
    """The whole magnitude argument rests on the LoRA scale being alpha/rank ==
    1.0.  Every field config in QWEN_FIELD sets linear_alpha == linear, and so
    does ours.  If someone decouples them, ||dW|| ~ sqrt(rank)*lr*steps stops
    holding and section 3(c) of the analysis has to be redone."""
    proc = _template()
    assert proc["network"]["linear_alpha"] == proc["network"]["linear"], (
        "alpha != rank: the effective-magnitude law used to reject the rank "
        "change no longer applies; re-derive before shipping."
    )


def test_qwen_template_sets_no_do_cfg():
    """Our measured 2.903 s/step was taken WITHOUT do_cfg, while the field's
    4.676 s/step (the source of SEC_PER_IT=4.70) was taken WITH it.  That gap
    is why the clock has far more room than the constant suggests.  If do_cfg
    is ever added here, every budget number in section 5.3 is void."""
    proc = _template()
    assert not proc["train"].get("do_cfg"), (
        "do_cfg on qwen roughly doubles the per-step cost (recipe.py:521-533 "
        "prices the same effect for ideogram4); the rank-capacity budget "
        "analysis assumed it is off."
    )


# --------------------------------------------------------------------------
# 2. The field record does not support raising rank.
# --------------------------------------------------------------------------


def test_rank_128_holds_the_majority_of_qwen_first_places():
    """5 of 7 first places across both tournaments went to rank 128 -- the same
    value four different hotkeys published verbatim.  Rank is an inherited
    per-architecture ai-toolkit default, not a tuned edge."""
    firsts = [row for row in QWEN_FIELD if row[4]]
    assert len(firsts) == 7, "one first place per qwen task, 7 tasks"
    at_128 = [row for row in firsts if row[1] == 128]
    above_128 = [row for row in firsts if row[1] > 128]
    assert len(at_128) == 5
    assert len(above_128) == 2
    # And both of the high-rank wins are the same operator on the same day.
    assert {row[0].split()[0] for row in above_128} == {"aug03"}
    assert {row[0].split()[2] for row in above_128} == {"5FBmn1ax"}
    # Nobody has ever run qwen below 128.  Our 32 is unprecedented in EITHER
    # direction -- which is why this is an open question, not a known deficit.
    assert min(row[1] for row in QWEN_FIELD) == 128


def test_lower_rank_won_the_deepest_disadvantaged_head_to_head():
    """Jul-27 `5722b124` (N=48, h=1.5): rank 128 shipped 275 steps and beat rank
    152 at 1197 steps by 8.4%.

    CORRECTED AT THE WEEK-6 INTEGRATION MERGE (2026-08-07), and the test renamed
    with it.  This was billed as "the ONLY qwen task where entrants at two
    different ranks BOTH COMPLETED their configured plan", asserting
    `1197/300 == 3.99`.  That is false.  `5FpdSckw`'s config says `steps: 300`,
    but the shipped artifact's own metadata says otherwise — read by HTTP range
    request over the first 8 bytes (LE uint64 header length) plus the header
    JSON of
    `.../20260727-5722b124-...-5FpdSckw/checkpoints/last.safetensors`:
        __metadata__.training_info = {"step": 275, "epoch": 2}
    against `5GU4Xkd3`'s {"step": 1197, "epoch": 9}.  It did NOT complete, and
    there is NO task in the record where a high-rank and a low-rank entrant both
    completed.  The DIRECTION survives and is if anything stronger — 4.35x fewer
    steps, and still an 8.4% win — but the claim as written was wrong and must
    not be recycled.
    """
    winner = 0.09613764286492714  # 5FpdSckw, rank 128, SHIPPED 275 steps
    loser = 0.10422607737567503  # 5GU4Xkd3, rank 152, SHIPPED 1197 steps
    assert winner < loser
    assert (loser / winner - 1) == pytest.approx(0.0841, abs=5e-4)
    # ...and the winner shipped 4.35x fewer steps, so this is not a depth
    # artifact in the high-rank entrant's favour.
    assert 1197 / 275 == pytest.approx(4.353, abs=0.01)


def test_high_rank_margin_is_inside_the_same_recipe_noise_floor():
    """`7421f056` gives a free replicate: two entrants, configs identical after
    removing hotkey-bearing paths, both killed at step 850.  Their spread is the
    run-to-run noise of this metric on qwen.  The rank-149 win on `ff643470` is
    smaller than that, so it is not evidence of a rank effect."""
    # 7421f056: 5FW2Eaae vs 5FpdSckw, byte-identical recipe, both shipped 850.
    noise = 0.09278030969600007 / 0.09083169271833681 - 1
    assert noise == pytest.approx(0.02145, abs=5e-5)
    # ff643470: 5FBmn1ax rank 149 (completed 1095) vs 5FW2Eaae rank 128
    # (KILLED at 700 of 1300).  The rank-149 entrant also shipped 1.56x the
    # depth, and still only won by:
    margin = 0.11688379350934618 / 0.1159371583891911 - 1
    assert margin == pytest.approx(0.00817, abs=5e-5)
    assert margin < noise, (
        "the high-rank win margin must stay inside the same-recipe noise floor "
        "for the 'rank is not the variable' conclusion to hold"
    )


# WITHDRAWN AT THE WEEK-6 INTEGRATION MERGE (2026-08-07):
# `test_our_planned_effective_magnitude_sits_inside_the_winner_band`.
#
# It asserted that our planned `E = lr*sqrt(rank)*steps` sits inside the field
# winners' band [0.339, 1.191] and 1.40x above its shallowest member, and it was
# offered as the unit's central positive claim ("we are not behind").  It is
# withdrawn rather than re-tuned, for three independent reasons:
#
#  1. THE BAND'S LOWER EDGE IS COMPUTED FROM A DEPTH THAT DID NOT HAPPEN.  It
#     used `5722b124 / 5FpdSckw` at 300 steps, taken from that entrant's config.
#     Its shipped artifact carries `training_info = {"step": 275, "epoch": 2}`
#     (range-read of the safetensors header; verified independently by the
#     reviewer and again by the integrator).  See
#     `test_lower_rank_won_the_deepest_disadvantaged_head_to_head` above.
#  2. THE WINNER SET IS INCOMPLETE.  `assert len(winners) == 4` excluded every
#     entrant that published no numbered ladder, but those depths ARE readable
#     from `__metadata__.training_info` — the reviewer recovered five more,
#     moving the band to roughly [0.226, 1.245].  The integrator verified the
#     technique on 5722b124 but did NOT re-verify the other five, so no
#     corrected band is asserted here.
#  3. IT IS A CATEGORY ERROR EVEN WITH CORRECT NUMBERS.  Our E is computed from
#     a FIXED PLANNED depth; the winners' E is computed from the depth they
#     CHOSE TO SHIP, and several of them shipped far inside their own clock.
#     Comparing the two says nothing about capacity and quietly reassures us
#     about the wrong lever.  The qwen gap the record actually supports is
#     checkpoint SELECTION and `do_cfg` (13/13 in the field, absent from ours),
#     not rank -- see the week-6 release note's week-7 list.
#
# The HOLD-rank-32 decision does not rest on this test; it rests on
# `test_rank_128_holds_the_majority_of_qwen_first_places`,
# `test_high_rank_margin_is_inside_the_same_recipe_noise_floor` and
# `test_coupled_lr_at_rank_128_is_magnitude_neutral`, none of which are affected.


def test_coupled_lr_at_rank_128_is_magnitude_neutral():
    """The lr that preserves our current ||dW|| at rank 128 is 1e-4*sqrt(32/128)
    = 5.0e-5.  Adopting the field's ABSOLUTE constant instead (1.0877e-3/sqrt(128)
    = 9.61e-5) is a ~1.92x magnitude change bundled with the capacity change --
    two coupled unvalidated moves, not one.  This test states both numbers so
    the distinction cannot be lost again."""
    ours = _effective_magnitude(32, 1e-4, 1)
    neutral_lr = 1e-4 * math.sqrt(32 / 128)
    assert neutral_lr == pytest.approx(5.0e-5, rel=1e-9)
    assert _effective_magnitude(128, neutral_lr, 1) == pytest.approx(ours, rel=1e-9)

    field_constant = 1.0877e-3  # 5FBmn1ax aug03 (r=141 and r=149 agree to 0.01%)
    field_lr_at_128 = field_constant / math.sqrt(128)
    assert field_lr_at_128 == pytest.approx(9.614e-5, rel=1e-3)
    ratio = _effective_magnitude(128, field_lr_at_128, 1) / ours
    assert ratio == pytest.approx(1.923, rel=1e-3), (
        "adopting the field's absolute lr constant at rank 128 is a 1.92x "
        "magnitude change, not a pure capacity change"
    )
    # The same 1.92x is reachable at rank 32 for free, by lr alone.
    free_lr = field_constant / math.sqrt(32)
    assert free_lr == pytest.approx(1.9228e-4, rel=1e-3)
    assert _effective_magnitude(32, free_lr, 1) / ours == pytest.approx(
        ratio, rel=1e-9
    )


# --------------------------------------------------------------------------
# 3. Cost side: a future rank change cannot be made unpriced.
# --------------------------------------------------------------------------


def test_artifact_size_law_reproduces_our_own_measured_rank32_artifact():
    """bytes(r) = 18,432,010*r + 237,664, fitted on the field's r=128/141/149/152
    sizes, predicts our own rank-32 export to within metadata slack.  This is
    what makes the 4x artifact-size claim a measurement rather than an estimate."""
    predicted = ARTIFACT_BYTES_PER_RANK * 32 + ARTIFACT_BYTES_INTERCEPT
    assert abs(predicted - OUR_RANK32_ARTIFACT_BYTES) < 8_192
    assert LORA_PARAMS_PER_RANK * 32 == OUR_RANK32_LOGICAL_ELEMENTS
    # rank 128 is 4.00x the bytes of rank 32.
    at_128 = ARTIFACT_BYTES_PER_RANK * 128 + ARTIFACT_BYTES_INTERCEPT
    assert at_128 / OUR_RANK32_ARTIFACT_BYTES == pytest.approx(4.0, abs=0.01)
    # ...and it must still clear EXPORT_RESERVE_S as a single HF LFS push.
    required_mb_s = (at_128 / 1e6) / recipe.EXPORT_RESERVE_S
    assert required_mb_s == pytest.approx(13.1, abs=0.3), (
        "rank 128 needs ~13 MB/s sustained export throughput vs ~3.3 MB/s "
        "today; this is UNMEASURED on the tournament host"
    )


def test_step_time_multiplier_for_rank_128():
    """Sequence-applied base params per block are 226,492,416 and LoRA params
    per rank are 110,592 -- ratio exactly 2048 -- so LoRA GEMM FLOPs are r/2048
    of base.  With gradient checkpointing the base costs ~3 forward-equivalents
    and the LoRA ~4, giving a step cost proportional to (3 + r/512).  The
    img_mod.1/txt_mod.1 projections hold 28% of LoRA params but ~0 FLOPs, which
    is why the artifact grows 4x while compute grows ~6%."""

    def mult(r):
        return 3.0 + r / 512.0

    assert mult(128) / mult(32) == pytest.approx(1.0612, abs=1e-4)
    assert mult(64) / mult(32) == pytest.approx(1.0204, abs=1e-4)


def test_raising_rank_requires_repricing_sec_per_it():
    """SEC_PER_IT["qwen-image"] = 4.70 was derived from FIELD artifacts that run
    rank 128 AND do_cfg (recipe.py:538-557).  It is not our rate: our own
    completed 1026-step run measured 2.903 s/step gross of startup at rank 32
    with no do_cfg.  If anyone raises the template rank, the constant must be
    re-derived rather than inherited -- this test fails until they do."""
    proc = _template()
    rank = proc["network"]["linear"]
    sec = recipe.SEC_PER_IT["qwen-image"]
    if rank == 32:
        assert sec == pytest.approx(4.70), (
            "the qwen clock constant moved while rank stayed 32; re-read "
            "SN56-WEEK6-QWEN-CAPACITY-ANALYSIS-2026-08-07.md section 5"
        )
        return
    required = 4.70 * (3.0 + rank / 512.0) / (3.0 + 32 / 512.0)
    assert sec >= required - 1e-9, (
        f"rank was raised to {rank} without re-pricing SEC_PER_IT['qwen-image']: "
        f"needs >= {required:.3f} s/step, found {sec:.3f}.  qwen is the one type "
        f"where the clock binds (recipe.py:608-618)."
    )


def test_our_measured_rate_leaves_room_for_rank_128_on_every_real_shape():
    """The affordability half of the answer.  Rank 128 is NOT rejected because
    it does not fit -- at our own measured rate it fits every real qwen shape
    with 40-70% of the optimizer window unused.  It is rejected because the
    evidence says it buys nothing (see the tests above).  If this ever fails,
    the recommendation gets STRONGER, not weaker."""
    proc = _template()
    net_rate_r32 = (OUR_TOOLKIT_WINDOW_S - recipe.STARTUP_S) / OUR_COMPLETED_STEPS
    net_rate_r128 = net_rate_r32 * (3.0 + 128 / 512.0) / (3.0 + 32 / 512.0)
    for label, (n, hours) in REAL_QWEN_SHAPES.items():
        plan = recipe.size_scaled_steps(
            "qwen-image", n, hours, proc["train"]["steps"]
        )
        window = (
            hours * 3600.0
            - recipe.EXPORT_RESERVE_S
            - recipe.STOP_MARGIN_S
            - recipe.STARTUP_S
        )
        fits_r128 = int(window / net_rate_r128)
        assert fits_r128 > plan, (
            f"{label}: rank 128 would no longer complete the planned {plan} "
            f"steps at our measured rate (fits {fits_r128})"
        )
        # ...and there is real slack, not a knife edge.
        assert fits_r128 / plan > 1.5


def test_the_metric_is_mostly_an_unconditional_reconstruction():
    """Documentation-as-assertion for the reason capacity is the wrong knob.
    qwen-image is scored 0.25*caption + 0.75*blank; in the blank branch
    positive == negative == "" so CFG is inert, leaving a prompt-free
    reconstruction at denoise 0.93 -- the highest of any type.  That is a
    first-moment (prior-location) objective, which cannot use extra LoRA
    directions, and rank 32 already gives ~6-10M trainable params per training
    image on N = 28-50."""
    blank_weight = 0.75
    qwen_denoise = 0.93
    assert blank_weight > 0.5
    # denoise by type (constants.py:25-32): krea2 0.80, ideogram4 0.75,
    # flux 0.80, z-image 0.90, qwen-image 0.93 -- qwen destroys the most.
    assert qwen_denoise == max(0.80, 0.75, 0.80, 0.93, 0.90)
    for n, _hours in REAL_QWEN_SHAPES.values():
        params_per_image = OUR_RANK32_LOGICAL_ELEMENTS / n
        assert params_per_image > 5e6
