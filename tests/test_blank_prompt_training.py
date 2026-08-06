"""Week-6 guard: caption dropout, cached text embeddings, and the blank prompt.

WHY THIS FILE EXISTS
====================
75% of the image score is a BLANK-PROMPT reconstruction MSE
(validator/evaluation/evaluators/diffusion.py:203-318 and
validator/scoring/tasks.py:280-298 at validator pin b026da04), so "does the
model ever see an empty caption during training?" looks like a free win, and
`caption_dropout_rate` looks like the lever.

It was investigated for z-image and qwen-image on 2026-08-06 and DELIBERATELY
NOT SHIPPED.  Two independent reasons, both recorded as executable assertions
below so the decision cannot be silently reversed:

  1. THE FIELD DID NOT USE IT.  Every retrievable Aug-3 rank-1 config from
     HF org `gradients-io-tournaments` was read.  Effective caption dropout
     appears in exactly ONE of nine ai-toolkit configs (krea2 41025fb5).  Both
     z-image winners and all three qwen-image winners ran dropout 0.

  2. FOR QWEN IT IS INERT, AND UN-INERTING IT DOES NOT FIT THE CLOCK.  At
     ai-toolkit pin 99be3d96a2468d3a5228a4eb05ba67e63c586b4e,
     `toolkit/dataloader_mixins.py:387` reads

         if self.dataset_config.caption_dropout_rate > 0 \
                 and not short_caption \
                 and not self.dataset_config.cache_text_embeddings:

     so caption dropout is hard-disabled whenever text embeddings are cached.
     `jobs/process/BaseSDTrainProcess.py:149-151` forces the dataset flag on
     from `train.cache_text_embeddings`, which our qwen template sets (as did
     3/3 qwen winners).  Turning caching off to re-enable dropout puts a
     Qwen2.5-VL-7B text-encoder forward back on EVERY step
     (`extensions_built_in/sd_trainer/SDTrainer.py:1628-1647` vs the cached
     branch at 1571-1588) and keeps that encoder resident instead of unloading
     it (`SDTrainer.py:307-345` -> `toolkit/unloader.py:44`).  qwen-image is the
     one type with no clock headroom: see
     `test_qwen_budget_cannot_absorb_a_live_text_encoder`.

Evidence (all read-only, all re-checkable):
  HF gradients-io-tournaments/tournament-tourn_c54bb970b5d0aa91_20260803-<task>-<hotkey>
      /raw/main/checkpoints/config.yaml            (the rank-1 configs)
  SN56-project/evidence/week6-tournament-dataset-harvest-20260806/tasks/
                                                   (the real task shapes)
  ai-toolkit pin 99be3d96a2468d3a5228a4eb05ba67e63c586b4e   (the semantics)
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from PIL import Image

from forge import recipe
from forge.config import build_config
from forge.data.schema import ImageSpec

# --------------------------------------------------------------------------- #
# The Aug-3 field, as read off the rank-1 config.yaml of every task whose
# trained_model_repository the harvest recorded.  `dropout` is the value of
# `datasets[0].caption_dropout_rate` (None == key absent, which is toolkit
# default 0.0 per toolkit/config_modules.py:919); `cache_te` is
# `train.cache_text_embeddings` (None == key absent, default False per
# toolkit/config_modules.py:544).
#
# `effective` is the conjunction the toolkit actually evaluates at
# dataloader_mixins.py:387 — dropout only does something when it is > 0 AND
# text embeddings are NOT cached.
#
#   task       type        dropout  cache_te  effective
# --------------------------------------------------------------------------- #
AUG3_WINNER_CAPTION_DROPOUT = {
    # z-image — BOTH winners ran zero dropout.  This is the finding that
    # argues against adding it to our z-image template.
    ("b290d171", "z-image"): (None, None, False),
    ("b2582457", "z-image"): (None, None, False),
    # qwen-image — all three winners cached text embeddings.  7421f056 DID set
    # caption_dropout_rate 0.05, and it did nothing: cache_text_embeddings was
    # true, so line 387's `not cache_text_embeddings` was False.  This is the
    # exact mistake the non-inertness guard below exists to prevent.
    ("7421f056", "qwen-image"): (0.05, True, False),
    ("ff643470", "qwen-image"): (None, True, False),
    ("4782f46f", "qwen-image"): (None, True, False),
    # The other ai-toolkit types, for context: the one and only effective
    # caption dropout anywhere in the Aug-3 field is krea2 41025fb5.
    ("41025fb5", "krea2"): (0.05, False, True),
    ("db9f7244", "krea2"): (None, False, False),
    ("84be9fcd", "ideogram4"): (None, False, False),
    ("1365fa1c", "ideogram4"): (None, False, False),
}

# Every ai-toolkit type forge can emit.  flux is included: `forge/tasks/
# dispatch.py:19` routes flux to kohya ONLY under FORGE_FLUX_BACKEND=kohya, so
# the ai-toolkit template is the default path for it too.
AITOOLKIT_TYPES = ("flux", "krea2", "ideogram4", "z-image", "qwen-image")

# One real Aug-3 shape per type, from the harvest task-meta.json:
#   type -> (task, n_pairs, hours_to_complete, (width, height))
REAL_SHAPE = {
    "flux": ("db5fefc5", 15, 0.75, (1195, 896)),
    "krea2": ("41025fb5", 21, 0.75, (1024, 768)),
    "ideogram4": ("84be9fcd", 46, 1.0, (1408, 768)),
    "z-image": ("b290d171", 39, 1.0, (1408, 768)),
    "qwen-image": ("7421f056", 28, 1.25, (1024, 768)),
}

# The two real qwen shapes where the CLOCK, not the depth law, is the binding
# constraint: (task, n_pairs, hours).  See recipe.MARGIN_BY_TYPE's header.
QWEN_CLOCK_BOUND_SHAPES = [("7421f056", 28, 1.25), ("ff643470", 41, 1.5)]


@dataclass(frozen=True)
class _Spec(ImageSpec):
    """ImageSpec with a redirectable dataset dir (the real one is /dataset/images)."""

    images_dir: str = ""

    @property
    def dataset_images_dir(self) -> str:
        return self.images_dir


@pytest.fixture(scope="module")
def generated_configs(tmp_path_factory):
    """The config forge would actually emit, per type, on a real Aug-3 shape."""
    root = tmp_path_factory.mktemp("blank-prompt")
    out = {}
    for model_type in AITOOLKIT_TYPES:
        task, pairs, hours, (width, height) = REAL_SHAPE[model_type]
        images = root / task
        images.mkdir()
        for index in range(pairs):
            Image.new("RGB", (width, height), (7, 9, 11)).save(
                images / f"{index:03d}.png", compress_level=1
            )
            (images / f"{index:03d}.txt").write_text("a caption")
        spec = _Spec(
            task_id=task,
            model="rayonlabs/Test-Base",
            model_type=model_type,
            expected_repo_name=f"tournament-week6-{task}",
            images_dir=str(images),
        )
        out[model_type] = build_config(
            spec, num_images=pairs, hours_to_complete=hours
        )
    return out


def _dataset_and_train(cfg):
    process = cfg["config"]["process"][0]
    return process["datasets"][0], process["train"]


# --------------------------------------------------------------------------- #
# 1. The invariant: a caption dropout we ship must actually do something.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_type", AITOOLKIT_TYPES)
def test_shipped_caption_dropout_is_never_inert(generated_configs, model_type):
    """`caption_dropout_rate > 0` requires `cache_text_embeddings` falsy.

    toolkit/dataloader_mixins.py:387 at pin 99be3d96 gates the dropout roll on
    `not self.dataset_config.cache_text_embeddings`, and
    jobs/process/BaseSDTrainProcess.py:149-151 copies `train.cache_text_embeddings`
    down onto every dataset.  A config that sets both is telling the operator it
    trains on blank prompts while training on none — which is precisely what the
    Aug-3 qwen winner 7421f056 shipped.

    This is a PROPERTY, not a pinned value: any type may legitimately change its
    dropout or its caching, but never into this combination.

    THE FLAG IS READ AT BOTH LEVELS (integrator, week-6 merge).  The first
    version of this test read only `train.cache_text_embeddings`, which admitted
    the exact defect it exists to prevent: the gate at dataloader_mixins.py:387
    reads the DATASET's flag, and BaseSDTrainProcess.py:148-151 only propagates
    train -> dataset when train is TRUE — it never clears a dataset-level flag.
    A config with dataset-level caching on and train-level off is therefore
    inert, and the train-only read passed it.  Effective caching is the OR.
    """
    dataset, train = _dataset_and_train(generated_configs[model_type])
    dropout = float(dataset.get("caption_dropout_rate", 0.0) or 0.0)
    cached = bool(train.get("cache_text_embeddings", False)) or bool(
        dataset.get("cache_text_embeddings", False)
    )
    assert not (dropout > 0 and cached), (
        f"{model_type}: caption_dropout_rate={dropout} is INERT because "
        "cache_text_embeddings is true at the train and/or dataset level "
        "(dataloader_mixins.py:387 reads dataset_config.cache_text_embeddings; "
        "BaseSDTrainProcess.py:148-151 copies train->dataset but never clears). "
        "Either drop the dropout key or turn caching off — but read "
        "test_qwen_budget_cannot_absorb_a_live_text_encoder before turning "
        "caching off on qwen-image."
    )


# --------------------------------------------------------------------------- #
# 2. z-image: no caption dropout, because the field ran none.
# --------------------------------------------------------------------------- #
def test_zimage_ships_no_caption_dropout(generated_configs):
    """z-image trains on captions only — matching 2/2 Aug-3 z-image winners.

    Both rank-1 z-image configs (tasks b290d171 and b2582457) omit
    `caption_dropout_rate` entirely, i.e. toolkit default 0.0
    (toolkit/config_modules.py:919).  The metric being 75% blank-prompt is an
    argument for dropout in the abstract; the two miners who actually won
    z-image tasks under that metric did not use it, and we cannot measure the
    alternative before the Monday tournament.  Absence here is a decision.
    """
    dataset, train = _dataset_and_train(generated_configs["z-image"])
    assert "caption_dropout_rate" not in dataset, (
        "z-image gained a caption_dropout_rate. The Aug-3 field evidence "
        "(b290d171, b2582457: both absent) does not support it. If new "
        "evidence does, update AUG3_WINNER_CAPTION_DROPOUT and this docstring "
        "in the same commit."
    )
    # The mechanism is live for z-image (unlike qwen) — nothing caches text
    # embeddings away — so if this is ever revisited the change is a one-liner
    # and costs no wall clock.  Pin the precondition so that stays true.
    assert not bool(train.get("cache_text_embeddings", False))
    assert not bool(train.get("unload_text_encoder", False))


# --------------------------------------------------------------------------- #
# 3. qwen-image: keep cached text embeddings, ship no dropout.
# --------------------------------------------------------------------------- #
def test_qwen_keeps_cached_text_embeddings_and_ships_no_dropout(generated_configs):
    """qwen-image caches text embeddings — matching 3/3 Aug-3 qwen winners.

    Consequence, by dataloader_mixins.py:387, is that caption dropout cannot
    work for qwen at all.  We therefore do not set it, rather than setting it
    and pretending.
    """
    dataset, train = _dataset_and_train(generated_configs["qwen-image"])
    assert train.get("cache_text_embeddings") is True, (
        "qwen-image stopped caching text embeddings. 3/3 Aug-3 qwen winners "
        "cached, and un-caching puts a Qwen2.5-VL-7B forward on every step on "
        "the one type with ~1% clock slack — see "
        "test_qwen_budget_cannot_absorb_a_live_text_encoder."
    )
    assert "caption_dropout_rate" not in dataset, (
        "qwen-image gained a caption_dropout_rate. With "
        "cache_text_embeddings=true it is INERT (dataloader_mixins.py:387) — "
        "the exact no-op the Aug-3 winner 7421f056 shipped."
    )


# --------------------------------------------------------------------------- #
# 4. Why the qwen fix was rejected: the clock, quantified.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task,pairs,hours", QWEN_CLOCK_BOUND_SHAPES)
def test_qwen_budget_cannot_absorb_a_live_text_encoder(
    generated_configs, task, pairs, hours
):
    """Disabling `cache_text_embeddings` must buy itself out of ~1% of slack.

    With caching ON the text encoder runs once per IMAGE during
    `setup_epoch` -> `cache_text_embeddings` (toolkit/dataloader_mixins.py:1874)
    and is then unloaded to CPU (SDTrainer.py:330-340 -> unloader.py:44).  With
    caching OFF it runs once per STEP under no_grad (SDTrainer.py:1628-1647) and
    stays resident on the GPU (qwen_image.py:167-172, 221 — a qfloat8
    Qwen2.5-VL-7B with the visual tower stripped).

    So the recurring cost of the "fix" is `(steps - cached_items)` extra 7B text
    encoder forwards, and the whole budget for them is the slack the planner
    leaves between `steps * SEC_PER_IT` and the terminate trigger.

    `cached_items` is `pairs * len(resolution)`, NOT `pairs` (integrator, week-6
    merge).  `resolution: [512, 768, 1024]` in the qwen template is expanded by
    toolkit/config_modules.py:1050-1062 into one dataset COPY per resolution, so
    the setup-time cached pass covers 3N items, not N.  The correction moves the
    per-forward budget UP (fewer extra forwards to amortise the same slack) —
    i.e. it works against the rejection — and the rejection still holds.

    This test asserts the per-forward budget is under 100 ms, i.e. too tight to
    bet a tournament on for an encoder we have NEVER timed on our own host.

    WHAT A FAILURE HERE ACTUALLY MEANS.  This quantity is dominated by the
    planner, not by the encoder: when the wall-clock cap binds, slack collapses
    to roughly `(1 - MARGIN_BY_TYPE['qwen-image']) * budget - STARTUP_S`.  So it
    goes RED when we become MORE conservative (measured: margin 0.98 passes at
    ~61/74 ms, 0.96 fails at ~171/183 ms), which is the opposite of a reason to
    turn caching off.  If it fails, first check whether `MARGIN_BY_TYPE` or
    `SEC_PER_IT` moved; re-derive this bound against the new planner. Do NOT
    read a failure as authorisation to disable `cache_text_embeddings`, and
    never on tournament eve — that still requires a measurement of the qwen
    encoder on our own host, which we do not have.
    """
    qwen_dataset, _train = _dataset_and_train(generated_configs["qwen-image"])
    resolution = qwen_dataset.get("resolution", 512)
    dataset_copies = len(resolution) if isinstance(resolution, list) else 1
    assert dataset_copies == 3, (
        "the qwen resolution list changed; re-derive the cached-item count "
        f"(config_modules.py:1050-1062 forks one dataset copy per resolution): "
        f"{resolution}"
    )
    steps = recipe.size_scaled_steps("qwen-image", pairs, hours, 3000)
    window_s = recipe.training_deadline_s(hours) - recipe.STARTUP_S
    slack_s = window_s - steps * recipe.SEC_PER_IT["qwen-image"]
    extra_forwards = steps - pairs * dataset_copies

    assert extra_forwards > 0
    assert slack_s > 0, f"{task}: planner already over-books the window"
    budget_per_forward_s = slack_s / extra_forwards
    assert budget_per_forward_s < 0.100, (
        f"{task}: qwen now has {slack_s:.0f}s of slack over {extra_forwards} "
        f"extra text-encoder forwards ({budget_per_forward_s * 1000:.0f} ms "
        f"each) at MARGIN_BY_TYPE={recipe.MARGIN_BY_TYPE['qwen-image']} and "
        f"SEC_PER_IT={recipe.SEC_PER_IT['qwen-image']}. Re-derive this bound "
        "against whichever of those two moved; this is NOT authorisation to "
        "disable cache_text_embeddings."
    )


def test_qwen_is_the_type_with_the_least_clock_slack():
    """The premise of the rejection: qwen has less headroom than any other type.

    Recomputed from the shipped constants rather than asserted as a number, so
    it tracks `recipe.SEC_PER_IT` / `recipe.MARGIN_BY_TYPE` instead of going
    stale beside them.
    """
    slack_fraction = {}
    for model_type in AITOOLKIT_TYPES:
        _task, pairs, hours, _dims = REAL_SHAPE[model_type]
        steps = recipe.size_scaled_steps(model_type, pairs, hours, 3000)
        window_s = recipe.training_deadline_s(hours) - recipe.STARTUP_S
        used_s = steps * recipe.SEC_PER_IT[model_type]
        slack_fraction[model_type] = (window_s - used_s) / window_s

    assert min(slack_fraction, key=slack_fraction.get) == "qwen-image", (
        f"qwen-image is no longer the tightest type: {slack_fraction}"
    )
    assert slack_fraction["qwen-image"] < 0.05
    # z-image, by contrast, is depth-law bound with a fifth of its window spare:
    # a caption-dropout change there would cost no wall clock at all.  The
    # reason we are not making it is evidence, not budget.
    assert slack_fraction["z-image"] > 0.20


# --------------------------------------------------------------------------- #
# 5. The field record itself, so the argument is auditable without re-fetching.
# --------------------------------------------------------------------------- #
def test_recorded_field_evidence_is_self_consistent():
    """`effective` must equal the toolkit's own gate: rate > 0 AND not cached.

    Guards the table above against a transcription error that would otherwise
    silently reverse the conclusion it supports.
    """
    for key, (dropout, cache_te, effective) in AUG3_WINNER_CAPTION_DROPOUT.items():
        computed = bool(dropout or 0.0) and not bool(cache_te)
        assert computed == effective, f"{key}: recorded effective={effective}"

    zimage = [
        v for (_t, mt), v in AUG3_WINNER_CAPTION_DROPOUT.items() if mt == "z-image"
    ]
    qwen = [
        v for (_t, mt), v in AUG3_WINNER_CAPTION_DROPOUT.items() if mt == "qwen-image"
    ]
    assert len(zimage) == 2 and not any(v[2] for v in zimage)
    assert len(qwen) == 3 and not any(v[2] for v in qwen)
    assert all(v[1] is True for v in qwen), "3/3 qwen winners cached text embeddings"

    effective_total = sum(1 for v in AUG3_WINNER_CAPTION_DROPOUT.values() if v[2])
    assert effective_total == 1, (
        "Exactly one Aug-3 ai-toolkit winner (krea2 41025fb5) ran an effective "
        "caption dropout. If this changes, the z-image/qwen decision changes."
    )
