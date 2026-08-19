"""Week-9 recipe pins: the Aug-24 changes, frozen with their evidence.

WHY THIS FILE EXISTS
====================
The week-9 changes respond to two losses:
  * Aug-17: three ideogram4 tasks, we finished LAST on all three
    (0.0765/0.2013/0.1212 vs field best 0.0185/0.0271/0.0298) — the do_cfg
    poison (evidence/week9-ideogram4-lane-20260818/REPORT.md §1-§4).
  * Aug-10: three krea2 tasks, 10th/5th/6th — the §4.4 flat-law/cadence
    correction was computed then and never shipped
    (evidence/aug10-loss-forensics-20260812/AUG10-LOSS-FORENSICS.md §4.4).
Plus the z-image boss-winner deltas (evidence/week9-zimage-lane-20260818/
REPORT.md §3.3) and the eval-band timestep alignment
(evidence/week9-research-scout-20260818/REPORT.md §1 F1.1).

Each pin below is the week-9 decision in executable form, so reverting any of
them is a deliberate act that must re-argue the evidence, not a silent drift.
Full arithmetic: evidence/week9-recipe-impl-20260818/CHANGES.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from PIL import Image

from forge import recipe
from forge.config import build_config, load_template
from forge.data.schema import ImageSpec

ALL_TYPES = ("flux", "krea2", "ideogram4", "z-image", "qwen-image")


@dataclass(frozen=True)
class _Spec(ImageSpec):
    images_dir: str = ""

    @property
    def dataset_images_dir(self) -> str:
        return self.images_dir


@pytest.fixture(scope="module")
def images_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("week9-images")
    for i in range(4):
        Image.new("RGB", (1024, 768), (7, 9, 11)).save(
            d / f"{i:03d}.png", compress_level=1
        )
        (d / f"{i:03d}.txt").write_text("a caption")
    return str(d)


def _built_process(model_type, pairs, hours, images_dir):
    spec = _Spec(
        task_id="week9-pins",
        model="rayonlabs/Test-Base",
        model_type=model_type,
        expected_repo_name="tournament-week9-pins",
        images_dir=images_dir,
    )
    cfg = build_config(spec, num_images=pairs, hours_to_complete=hours)
    return cfg["config"]["process"][0]


# --------------------------------------------------------------------------- #
# A. ideogram4 — the do_cfg poison stays dead.
# --------------------------------------------------------------------------- #
def test_no_template_carries_do_cfg():
    """No template may carry do_cfg/cfg_scale.  Absent = toolkit defaults
    False/1.0 (config_modules.py:489/:491 at pin 99be3d96) = single-branch
    training.  For ideogram4 this is the load-bearing week-9 removal: do_cfg
    trained pred = uncond + 10*(cond - uncond) with detach_unconditional=False
    (SDTrainer.py:1269-1271, base_model.py:947-949) — a -9x gradient on the
    blank-prompt branch that is 75% of the score, amplified ~8x by the
    ideogram4 evaluator's un-LoRA'd negative branch."""
    for model_type in ALL_TYPES:
        train = load_template(model_type)["config"]["process"][0]["train"]
        assert "do_cfg" not in train, model_type
        assert "cfg_scale" not in train, model_type


@pytest.mark.parametrize(
    "pairs,hours", [(16, 1.0), (10, 0.75)], ids=["aug17-1.0h", "aug17-0.75h"]
)
def test_emitted_ideogram4_config_has_no_cfg_objective(pairs, hours, images_dir):
    """The EMITTED config — through the release policy, which used to INJECT
    do_cfg true / cfg_scale 10.0 — carries neither key.  The policy's
    projection now REQUIRES absence (ideogram_release_policy._EXPECTED_RECIPE
    do_cfg/cfg_scale None; post-apply verification enforces it)."""
    process = _built_process("ideogram4", pairs, hours, images_dir)
    assert "do_cfg" not in process["train"]
    assert "cfg_scale" not in process["train"]
    # ...and the I-J20 recipe minus the poison still applied (not the
    # degraded template path): lr/EMA prove the policy fired.
    assert process["train"]["lr"] == 2.5e-5
    assert process["train"]["ema_config"] == {"use_ema": True, "ema_decay": 0.995}


# --------------------------------------------------------------------------- #
# B. min_denoising_steps — eval-band alignment, field values per type.
# --------------------------------------------------------------------------- #
def test_min_denoising_steps_field_values():
    """ideogram4 250 / z-image 100 — the field-winner values, traced at pin
    99be3d96 (config_modules.py:369 -> BaseSDTrainProcess.py:1148,:1276-1293
    -> timesteps = linspace(1000,1) DESCENDING, samplers/
    custom_flowmatch_sampler.py:107-118): min=k excludes the k highest-noise
    timesteps, i.e. exactly the band above each type's evaluator renoise point
    (ideogram4 denoise 0.75 -> t>~750; z-image denoise 0.90 -> t>~900).
    Evidence: week9-research-scout REPORT §1 F1.1 (SpeeD, CVPR 2025) +
    ideogram lane §2.2 (5D7iEJm5/5EACrayt/5FNLSgh8/5HKEAZxF all ship 250) +
    z-image lane §2.3 L7 (5FBmn1ax rank-1 ships 100).

    krea2 deliberately has NO value: no krea2 field value was ever observed
    (scout doctrine: field-parity only where observed).  The forensics' #5
    proposed TESTING 200 on krea2 but sequenced it after the self-eval
    harness because the effect is inside the 3.7% noise floor — it is an
    untested GPU A/B, not a ship.  flux/qwen: out of week-9 scope.
    """
    expected = {
        "ideogram4": 250,
        "z-image": 100,
        "krea2": None,
        "flux": None,
        "qwen-image": None,
    }
    for model_type, value in expected.items():
        train = load_template(model_type)["config"]["process"][0]["train"]
        if value is None:
            assert "min_denoising_steps" not in train, model_type
        else:
            assert train.get("min_denoising_steps") == value, model_type


def test_min_denoising_steps_survives_the_ideogram_release_policy(images_dir):
    """The release policy deep-copies and updates specific keys; the template's
    min_denoising_steps must reach the emitted config unchanged."""
    process = _built_process("ideogram4", 16, 1.0, images_dir)
    assert process["train"]["min_denoising_steps"] == 250


def test_zimage_template_caption_dropout_pin():
    """caption_dropout_rate == 0.05 in the z-image template (boss-winner
    delta: 5GU4Xkd3 3-for-3 on the last three z-image boss tasks with our
    template + this key + flat 1000; z-image lane REPORT §3.3).  The emitted-
    config effectiveness gate is pinned in test_blank_prompt_training.py."""
    dataset = load_template("z-image")["config"]["process"][0]["datasets"][0]
    assert dataset.get("caption_dropout_rate") == 0.05


# --------------------------------------------------------------------------- #
# C. Step-law emissions at the Aug-10 / Aug-17 anchor shapes.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "model_type,pairs,hours,expected,basis",
    [
        # ideogram4 — Aug-17 anchor shapes (lane REPORT §4.1, recomputed:
        # CHANGES.md §3A; law-bound both).  n inferred by inverting our own
        # Aug-17 save cadence (114/228/342/456 -> n~16; 96/192/288/384 -> n~10).
        ("ideogram4", 16, 1.0, 1098, "law 1250*(16/24)^0.32 = 1097.9"),
        ("ideogram4", 10, 0.75, 945, "law 1250*(10/24)^0.32 = 944.58 -> round"),
        # ideogram4 — Aug-3 big shapes: clock-bound at the no-cfg caps.
        ("ideogram4", 41, 1.0, 1348, "cap int((3312-480)/2.1)"),
        ("ideogram4", 12, 0.75, 954, "cap int((2484-480)/2.1)"),
        # krea2 — the real Aug-10 shapes; §4.4's own table, replayed exactly.
        ("krea2", 10, 0.75, 1360, "cap int((2484-400-180)/1.40); field 1333"),
        ("krea2", 32, 1.0, 1850, "flat law; field plan 1833"),
        ("krea2", 28, 1.0, 1850, "flat law; field plan 1833"),
        # z-image — the last three boss shapes (winner shipped flat 1000 on
        # all three) + the two untouched Aug-3 1.0h anchors.
        ("z-image", 15, 0.75, 1000, "floor (raw law 777.9)"),
        ("z-image", 18, 0.75, 1000, "floor (raw law 852.2)"),
        ("z-image", 20, 0.75, 1000, "floor (raw law 898.3)"),
        ("z-image", 35, 1.0, 1188, "anchor 5EACrayt b290d171, untouched"),
        ("z-image", 43, 1.0, 1317, "anchor 5FBmn1ax b2582457, untouched"),
    ],
)
def test_week9_step_law_emissions(model_type, pairs, hours, expected, basis):
    assert recipe.size_scaled_steps(model_type, pairs, hours, 2000) == expected, basis


def test_week9_constants():
    """The supporting constants, pinned in one place."""
    assert recipe.SEC_PER_IT["ideogram4"] == 2.1  # do_cfg x2 removed; 2.019*1.04
    assert recipe.SEC_PER_IT["krea2"] == 1.40  # measured 1.3746 + 1.8% pad
    assert recipe.startup_for("krea2") == 400.0  # measured 373 s + pad (§4.4)
    for other in ("flux", "ideogram4", "z-image", "qwen-image"):
        assert recipe.startup_for(other) == 300.0, other  # scope containment
    assert recipe.FIELD_DEMONSTRATED_DEPTH["ideogram4"][:2] == (1.0, 1523)
    assert recipe.STEP_TABLE["krea2"] == dict(
        base=1850, n_ref=24, p=0.00, min=600, max=2200
    )
    assert recipe.STEP_TABLE["ideogram4"] == dict(
        base=1250, n_ref=24, p=0.32, min=350, max=1650
    )
    assert recipe.STEP_TABLE["z-image"] == dict(
        base=984, n_ref=24, p=0.50, min=1000, max=1800
    )
    # Out-of-scope rows byte-identical to the base commit (75a0a20c).
    assert recipe.STEP_TABLE["flux"] == dict(
        base=1024, n_ref=24, p=0.50, min=500, max=2000
    )
    assert recipe.STEP_TABLE["qwen-image"] == dict(
        base=892, n_ref=24, p=0.51, min=300, max=1600
    )
    assert recipe.SEC_PER_IT["flux"] == 2.0
    assert recipe.SEC_PER_IT["z-image"] == 1.8
    assert recipe.SEC_PER_IT["qwen-image"] == 4.7


# --------------------------------------------------------------------------- #
# D. Fixed checkpoint cadence — the selection ladder.
# --------------------------------------------------------------------------- #
def test_fixed_save_cadence():
    """krea2/ideogram4 save at the field's fixed 200 (forensics §4.4; the
    Aug-17 selectors shipped their 200-ladder argmin on all six entries);
    below one interval the adaptive kill-safe rule stands; other types and
    typeless callers keep the week-6 adaptive rule byte-identically."""
    assert recipe.FIXED_SAVE_EVERY == {"krea2": 200, "ideogram4": 200}
    for fixed_type, steps in (("krea2", 1850), ("krea2", 1360),
                              ("ideogram4", 1348), ("ideogram4", 954)):
        assert recipe.kill_safe_save_every(steps, 250, fixed_type) == 200
    # Short clock-capped runs still get an early recovery point.
    assert recipe.kill_safe_save_every(165, 250, "ideogram4") == 34  # 165//5+1
    assert recipe.kill_safe_save_every(199, 250, "krea2") == 40  # 199//5+1
    # Legacy behavior preserved for other types / no model_type.
    assert recipe.kill_safe_save_every(2000, 250) == 401
    assert recipe.kill_safe_save_every(2000, 250, "qwen-image") == 401
    assert recipe.kill_safe_save_every(1023, 250, "qwen-image") == 205


@pytest.mark.parametrize(
    "model_type,pairs,hours,steps",
    [("krea2", 10, 0.75, 1360), ("ideogram4", 16, 1.0, 1098)],
)
def test_emitted_save_cadence_is_the_fixed_200(
    model_type, pairs, hours, steps, images_dir
):
    process = _built_process(model_type, pairs, hours, images_dir)
    assert process["train"]["steps"] == steps
    assert process["save"]["save_every"] == 200
    # The ladder is retained on disk, not rolled away.
    assert process["save"]["max_step_saves_to_keep"] >= (steps - 1) // 200


# --------------------------------------------------------------------------- #
# E. krea2 dead keys stay deleted.
# --------------------------------------------------------------------------- #
def test_krea2_differential_guidance_keys_stay_deleted():
    """Confirmed-inert at pin 99be3d96 (SDTrainer.py:734 unreachable under
    do_guidance_loss:692 default False; sole consumption site by git grep) and
    deleted per the forensics STOP list.  Had do_guidance_loss ever been set,
    they would have silently activated at scale 2 — absence is the guard."""
    train = load_template("krea2")["config"]["process"][0]["train"]
    assert "do_differential_guidance" not in train
    assert "differential_guidance_scale" not in train
    assert "do_guidance_loss" not in train
