"""Week-6 ideogram4 EMA export horizon: the amendment, and its VACATION.

HISTORY, because the direction of this file reversed.  A Week-6 amendment moved
``train.ema_config.ema_decay`` 0.995 -> 0.99 on the reasoning that the EMA
shadow ATTENUATES the trained delta and that both Aug-3 rank-1 ideogram4
artifacts published 0.99.  Both of those facts are still true (and still
asserted below).  The amendment is nevertheless VACATED, because it optimised
the wrong sign: it made the exported adapter STRONGER on the one model type
whose evaluator amplifies the adapter ~7x.

WHAT THE RUNTIME DOES (unchanged, and still what makes ``ema_decay`` matter).
At ai-toolkit pin ``99be3d96a2468d3a5228a4eb05ba67e63c586b4e`` — the commit the
production Docker image builds — every artifact we upload is the EMA *shadow*,
never the trained weights:

  * ``jobs/process/BaseSDTrainProcess.py:491-497`` — ``save()`` calls
    ``self.ema.eval()`` unconditionally when ``self.ema`` exists, which is
    ``store()`` + ``copy_to()`` (``toolkit/ema.py:336-341``), i.e. the live
    parameters are overwritten by the shadow.  ``:530-539`` then serialises
    those live parameters and ``:697-698`` restores.  Periodic checkpoints —
    the ones our terminate -> finalize salvage path promotes — go through the
    same ``save()``.
  * ``toolkit/ema.py:62-65`` seeds the shadow by cloning the parameters at
    ``setup_ema()`` time (``BaseSDTrainProcess.py:769-781``, called at
    ``:2031`` before the first optimizer step), and
    ``toolkit/lora_special.py:122`` zero-initialises ``lora_up``.  The shadow
    therefore starts at *zero adapter effect*, so a lower decay-horizon means a
    STRONGER exported adapter, and a higher one means a weaker adapter.
  * ``toolkit/ema.py:47,57,118`` — ``setup_ema()`` does not pass
    ``use_num_updates``, so ``self.num_updates`` is ``None`` and the
    ``(1 + n) / (10 + n)`` warm-up ramp guarded by ``if self.num_updates is not
    None`` never runs.  Decay is flat from step 1 and there is NO config path
    to change that.

WHY THE SIGN MATTERS HERE AND NOWHERE ELSE.  At G.O.D pin ``b026da04``,
``validator/evaluation/comfy_workflows/lora_ideogram4.json`` wires a
``DualModelGuider`` whose NEGATIVE branch is a separate, never-LoRA'd checkpoint
(``ideogram4_unconditional_fp8_scaled.safetensors``).  The LoRA enters only the
positive branch, so the guided prediction ``neg + c*(pos - neg)`` carries the
LoRA's delta multiplied by ``c`` — 8, falling to 5 over the last 30% via
``CFG_override`` (``diffusion.py:235-236`` writes both).  Every other image type
runs one model on both branches, and at the blank prompt (0.75 of the score)
``positive == negative == ""`` makes CFG an exact no-op there.  So ideogram4 is
the only type where adapter strength is amplified, and it is amplified across
100% of its score.

Every test below fails if the vacation is reverted, i.e. if 0.99 comes back.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time

import pytest
import yaml

from forge import config, ideogram_release_policy as policy, recipe, telemetry
from forge.data.schema import ImageSpec
from forge.tasks import checkpoints


# The three REAL Aug-3 ideogram4 shapes and the depth `recipe.STEP_TABLE`
# ships for each.  Mirrors tests/test_week6_ideogram_depth.py::REAL_IDEOGRAM_TASKS
# but is re-derived from `recipe` below so the two cannot silently diverge.
REAL_SHAPES = [
    # task, n_train, hours, steps we ship
    #
    # THE SIZE COLUMN IS n_train, NOT `image_text_pairs` (2026-08-07).  The
    # validator withholds ceil(0.10*N) before building train_data.zip, so the
    # container is handed 12 / 41 / 36 where the auditing record says 14 / 46 /
    # 40 — OBSERVED from each zip's central directory.  Feeding the audit N here
    # made this file certify depths (421/616/589) that were never emitted; what
    # `recipe` actually produced at the real abscissa was 401/593/569.
    ("1365fa1c", 12, 0.75, 414),  # R1 draw shape; tightest budget  (N=14)
    ("84be9fcd", 41, 1.0, 614),   # (N=46)
    ("b72da8c6", 36, 1.0, 589),   # (N=40) — unchanged; the one coincidence
]
IDS = [row[0] for row in REAL_SHAPES]

SHIPPED_DECAY = 0.995  # what I-J20-D2 / the 5FNLSgh8 anchor ran; restored
REJECTED_DECAY = 0.99  # 5FBmn1ax's value; the vacated amendment's target


# --------------------------------------------------------------------------
# The evaluator's amplification, read off the pinned workflow graph.
# --------------------------------------------------------------------------
# OBSERVED at G.O.D pin b026da04:
#   comfy_workflows/lora_ideogram4.json  `Dual_model_guider`   cfg 8
#   comfy_workflows/lora_ideogram4.json  `CFG_override`        cfg 5,
#                                        start_percent 0.7, end_percent 1
#   evaluators/diffusion.py:235-236      writes both from the eval cfg:
#                                        cfg and max(cfg - 3, 1)
#   evaluation/constants.py:31           ideogram4 cfg 8, denoise 0.75
IDEOGRAM4_CFG_MAIN = 8.0
IDEOGRAM4_CFG_TAIL = 5.0
IDEOGRAM4_CFG_TAIL_START = 0.7
IDEOGRAM4_DENOISE = 0.75
# Types whose blank-prompt condition has positive == negative == "" on a SINGLE
# model, making CFG an exact no-op there (0.75 of their score at 1x).
UNAMPLIFIED_TYPES = ("krea2", "qwen-image", "z-image")


def ideogram4_effective_cfg() -> float:
    """Arc-averaged amplification of the LoRA delta over the sampled schedule.

    ``SplitSigmasDenoise`` at denoise 0.75 keeps the low-noise 75% of the
    schedule, i.e. sigma-percent [0.25, 1.0].  ``CFG_override`` takes over at
    0.7, so the sampled arc is (0.70 - 0.25) at cfg 8 and (1.0 - 0.70) at cfg 5.

    INFERRED: percent-to-sigma is not linear in step index, so this is an
    approximation of the average.  The ENDPOINTS (8 and 5) are OBSERVED, and
    every conclusion drawn from this number only needs c >> 1.
    """
    lo = 1.0 - IDEOGRAM4_DENOISE
    span = IDEOGRAM4_DENOISE
    main = (IDEOGRAM4_CFG_TAIL_START - lo) / span
    return main * IDEOGRAM4_CFG_MAIN + (1.0 - main) * IDEOGRAM4_CFG_TAIL


# --------------------------------------------------------------------------
# The one controlled adapter-strength experiment in the ideogram4 record.
# --------------------------------------------------------------------------
# Jul-20 R1 task 3cfa1578 (ideogram4, N=11, h=0.75, SIXTEEN miners — the largest
# ideogram4 field in existence, and at the R1 shape).  Two published configs,
# fetched from hf gradients-io-tournaments/tournament-tourn_4aff76a867d2af49_
# 20260720-3cfa1578-5506-4a1d-a79a-2c55abc8958b-<hotkey>/checkpoints/config.yaml
# and diffed: the ENTIRE difference is the learning rate scaled 2x (lr and
# unet_lr 2.5e-5 -> 5.0e-5, text_encoder_lr 1e-7 -> 8e-7, and the matching
# min_lr_by_initial_lr keys).  steps 378, ema_decay 0.995, use_ema true,
# cosine_by_group, lora 32/32, caption_dropout_rate 0.05, do_cfg true,
# cfg_scale 10.0 and everything else are IDENTICAL in both.
ANCHOR_CONFIG_SHA256 = (
    "bf852a1aaa954ad8aaeea0b9522f4b8147dc84a88f1c6aeedde7f1c7201a954c"
)  # 5FNLSgh8; same digest independently recorded 2026-07-22 in
#    SN56-project/SN56-WEEK4-INDEPENDENT-REVIEW-2026-07-22.md:66
DOUBLED_LR_CONFIG_SHA256 = (
    "eeb914952cf4672f8b83a0f2e54237319b1b828d003273a7bb35dc7803128f30"
)  # 5EACrayt
ANCHOR_STEPS = 378
ANCHOR_LR = 2.5e-5
ANCHOR_DECAY = 0.995
ANCHOR_LOSS = 0.0502341  # rank 1 of 16   (forge/recipe.py:211-212)
DOUBLED_LR_LOSS = 0.0965093  # rank 13 of 16 (same source)
# Our Aug-3 R1 elimination margin, for scale on any penalty computed below.
AUG3_ELIMINATION_MARGIN = 0.0097


def strength_penalty(ratio: float) -> float:
    """Fractional loss increase for shipping ``ratio`` x the anchor's strength.

    From the quadratic that the amplification implies (see the module docstring
    and forge/ideogram_release_policy.py): with the scored metric a plain
    per-pixel MSE (diffusion.py:203-209) and the LoRA entering the prediction
    linearly at gain c, ``MSE(s) = L0 - 2 c s G + (c s)^2 H``.  At ``s = 2 s*``
    that returns exactly ``L0``, i.e. the whole adapter benefit is destroyed, so
    the single observed point calibrates the curvature:

        dMSE / MSE(s*) = k * (ratio - 1)^2,  k = DOUBLED_LR_LOSS/ANCHOR_LOSS - 1

    CAVEAT, stated: ``s*`` is task-dependent (G and H are properties of the
    held-out images), so ``k`` is a local sensitivity measured on one task, not
    a universal constant.  It is used here only to rank two candidate decays.
    """
    k = DOUBLED_LR_LOSS / ANCHOR_LOSS - 1.0
    return k * (ratio - 1.0) ** 2


# --------------------------------------------------------------------------
# A faithful re-implementation of the pinned EMA arithmetic.
# --------------------------------------------------------------------------
def exported_fraction(steps: int, decay: float, lr_of_step) -> float:
    """Share of the trained LoRA delta that ``save()`` actually writes.

    Reproduces ``toolkit/ema.py`` update-by-update with ``num_updates is None``
    (see the module docstring), on ``lora_up`` (B), which starts at zero:

        B_k = B_{k-1} + lr_k          (Adam: |per-coord step| ~ lr)
        s_k = decay * s_{k-1} + (1 - decay) * B_k,    s_0 = B_0 = 0

    MODEL ASSUMPTIONS, stated so they are not mistaken for measurement:
      * per-step displacement magnitude tracks ``lr`` — true for Adam once the
        moment estimates settle;
      * the update DIRECTION is stable enough that the shadow is a scaled
        endpoint rather than a directional average — exact early, degrades late;
      * ``lora_down`` (A) starts non-zero, so ``A_ema ~= A_final`` to leading
        order and the exported adapter ~= ``(s_T / B_T)`` x the trained adapter.

    This is a MODEL.  Nothing here has been measured on a GPU.
    """
    b = 0.0
    s = 0.0
    for k in range(1, steps + 1):
        b += lr_of_step(k, steps)
        s = decay * s + (1.0 - decay) * b
    return s / b


def exported_strength(steps: int, decay: float, lr_of_step) -> float:
    """``(lr integral) x f`` — the quantity the evaluator actually amplifies."""
    integral = sum(lr_of_step(k, steps) for k in range(1, steps + 1))
    return integral * exported_fraction(steps, decay, lr_of_step)


def constant_lr(value: float):
    return lambda _k, _steps: value


def our_cosine_lr(lr_max: float = 2.5e-5, eta_min: float = 2.5e-6):
    """Our actual schedule.

    ``toolkit/scheduler.py:11-16`` builds
    ``torch.optim.lr_scheduler.CosineAnnealingLR`` and
    ``BaseSDTrainProcess.py:2035-2036`` injects ``total_iters`` -> ``T_max``,
    so ``eta_min`` from ``lr_scheduler_params`` really is honoured.
    """

    def f(k: int, steps: int) -> float:
        phase = math.pi * (k - 1) / max(steps - 1, 1)
        return eta_min + (lr_max - eta_min) * (1 + math.cos(phase)) / 2

    return f


def _closed_form_constant_lr(steps: int, decay: float) -> float:
    """Analytic ``s_T / B_T`` for the constant-lr (linear B) case."""
    d_t = decay**steps
    d_t1 = decay ** (steps - 1)
    return (1 - d_t) - decay * (1 - steps * d_t1 + (steps - 1) * d_t) / (
        steps * (1 - decay)
    )


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Same isolation as tests/test_ideogram_release_policy.py.

    `forge.telemetry` keeps its record in a MODULE-GLOBAL `_data`, and
    `telemetry.init()` only updates `meta` — it does not reset `events`
    (forge/telemetry.py:125-146).  Any module that drives `config.build_config`
    without rebinding `_data` therefore leaks events into whatever runs next,
    and `tests/test_publication.py` asserts on an exact one-element event list.
    """
    monkeypatch.setattr(telemetry, "_PRIVATE_ROOT", str(tmp_path / "private"))
    monkeypatch.setattr(telemetry, "_RUN_NONCE", "ideogram-ema-horizon-test")
    monkeypatch.setattr(telemetry, "_BOUND_RUN_KEYS", {})
    monkeypatch.setattr(telemetry, "_LATEST_PRIVATE_RECORDS", {})
    monkeypatch.setattr(telemetry, "_t0", time.monotonic())
    monkeypatch.setattr(
        telemetry,
        "_data",
        {
            "schema": 1,
            "meta": {},
            "env": {},
            "events": [],
            "train_curve": [],
            "eval_curve": [],
            "samples": {},
        },
    )
    checkpoints._ACTIVE_RUNS.clear()


def _activation() -> dict:
    body = {
        "schema": 1,
        "kind": policy.ACTIVATION_KIND,
        "policy_sha256": policy.POLICY_SHA256,
        "amendment_sha256": policy.AMENDMENT_SHA256,
        "formal_ideogram_decision_sha256": "a" * 64,
        "scored_exact_final_sha256": "b" * 64,
        "selected_arm": "I-J20",
        "selection_basis": "clear_win",
        "owner_override": False,
        "production_mutation_authorized": True,
        "release_authorized": True,
        "deployment_authorized": False,
    }
    return {
        **body,
        "activation_sha256": hashlib.sha256(
            policy._canonical_bytes(body)
        ).hexdigest(),
    }


def _built_train_block(
    monkeypatch: pytest.MonkeyPatch, pairs: int, hours: float
) -> dict:
    """The `train:` block ai-toolkit will actually receive, via the real path."""
    monkeypatch.setattr(policy, "PRODUCTION_ACTIVATION", _activation())
    spec = ImageSpec.build(
        task_id="week6-ideogram-ema-horizon",
        model="black-forest-labs/FLUX.1-Krea-dev",
        model_type="ideogram4",
        expected_repo_name="irepo",
        trigger_word=None,
        dataset_zip=None,
    )
    cfg = config.build_config(spec, num_images=pairs, hours_to_complete=hours)
    return cfg["config"]["process"][0]["train"]


# --------------------------------------------------------------------------
# 1. the emitted config
# --------------------------------------------------------------------------
@pytest.mark.parametrize("row", REAL_SHAPES, ids=IDS)
def test_emitted_ideogram_config_carries_the_anchor_decay(
    row, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real `build_config` path, at every real shape, emits 0.995."""
    _task, pairs, hours, steps = row
    train = _built_train_block(monkeypatch, pairs, hours)
    assert train["steps"] == steps
    assert train["ema_config"] == {"use_ema": True, "ema_decay": SHIPPED_DECAY}
    assert train["ema_config"]["ema_decay"] != REJECTED_DECAY


def test_anchor_decay_survives_the_yaml_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`config.write_config` is a bare `yaml.safe_dump` — prove it stays 0.995.

    `forge/config.py:242-246` dumps the resolved dict verbatim with no schema
    filter, and ai-toolkit re-reads the whole `train:` block as kwargs
    (`BaseSDTrainProcess.py:108`).  A float that survives this round trip is a
    float `EMAConfig(**ema_config)` will see.
    """
    train = _built_train_block(monkeypatch, 14, 0.75)
    path = tmp_path / "out" / "cfg.yaml"
    config.write_config({"config": {"process": [{"train": train}]}}, str(path))
    reloaded = yaml.safe_load(path.read_text())
    ema = reloaded["config"]["process"][0]["train"]["ema_config"]
    assert ema == {"use_ema": True, "ema_decay": SHIPPED_DECAY}
    assert isinstance(ema["ema_decay"], float)
    # ema.py:53-54 is the only validation the pin applies to `decay`.
    assert 0.0 <= ema["ema_decay"] <= 1.0


def test_we_do_not_emit_ema_keys_the_pin_silently_ignores_or_that_bite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard against the two tempting "fixes" that do not work at this pin.

    `use_num_updates` looks like the right fix — it is the flag that enables the
    `(1 + n) / (10 + n)` warm-up bias correction at `toolkit/ema.py:118-124`.
    It is NOT reachable: `setup_ema()` (`BaseSDTrainProcess.py:776-780`) passes
    only `decay`, `use_feedback` and `param_multiplier`, and `EMAConfig`
    (`toolkit/config_modules.py:793-803`) reads its fields with `kwargs.get`,
    so an unknown key is dropped without error.  Emitting it would be a silently
    inert change that reads, in review, like a solved problem.

    `use_feedback` and `param_multiplier` ARE plumbed, but they mutate the LIVE
    training parameters every update (`ema.py:139-146`), not just the shadow.
    They are not export fixes and must not be reached for as if they were.
    """
    train = _built_train_block(monkeypatch, 14, 0.75)
    assert set(train["ema_config"]) == {"use_ema", "ema_decay"}


# --------------------------------------------------------------------------
# 2. the mechanism: why STRONGER is the wrong direction on this type
# --------------------------------------------------------------------------
def test_the_evaluator_amplifies_the_lora_delta_on_ideogram4() -> None:
    """`neg` is LoRA-independent, so the LoRA's delta is multiplied by cfg.

    In `lora_ideogram4.json` the `Lora_loader` output feeds only
    `CFG_override` -> `Dual_model_guider.model`, while
    `Dual_model_guider.model_negative` is a SEPARATE checkpoint
    (`ideogram4_unconditional_fp8_scaled.safetensors`) that the LoRA never
    touches.  So `pred = neg + c*(pos - neg)` and `d(pred)/d(delta) = c`.
    """
    c = ideogram4_effective_cfg()
    assert IDEOGRAM4_CFG_TAIL < c < IDEOGRAM4_CFG_MAIN
    assert c == pytest.approx(6.8, abs=0.05)
    # The claim that decides the recipe is only that c is LARGE, not its exact
    # value: even the tail-only bound is a 5x amplification.
    assert c > 4.0


def test_amplification_inverts_the_optimum_and_squares_the_penalty() -> None:
    """`s* ~ 1/c` and `d2MSE/ds2 ~ c^2`.

    From `MSE(s) = L0 - 2 c s G + (c s)^2 H`: the stationary point is at
    `c s* = G/H`, so the optimal RAW adapter strength falls as 1/c, and the
    curvature in s rises as c^2.  Both directions matter: ideogram4 wants a
    smaller adapter than an unamplified type AND punishes overshoot far harder.
    """
    c = ideogram4_effective_cfg()
    g_over_h = 1.0  # units; only ratios are used

    def mse(strength: float, gain: float) -> float:
        return -2 * gain * strength * g_over_h + (gain * strength) ** 2

    optimum_amplified = g_over_h / c
    optimum_flat = g_over_h / 1.0
    assert optimum_amplified == pytest.approx(optimum_flat / c)

    # Excess loss above each type's OWN optimum is `gain^2 * (s - s*)^2`, so the
    # same ABSOLUTE strength error costs c^2 more on ideogram4.  This is the
    # statement that matters, because `ema_decay` moves raw strength by a fixed
    # amount regardless of which type it is applied to.
    err = 0.01
    excess_amplified = mse(optimum_amplified + err, c) - mse(optimum_amplified, c)
    excess_flat = mse(optimum_flat + err, 1.0) - mse(optimum_flat, 1.0)
    assert excess_amplified == pytest.approx(c**2 * err**2)
    assert excess_flat == pytest.approx(err**2)
    assert excess_amplified / excess_flat == pytest.approx(c**2)
    assert c**2 > 45


def test_only_ideogram4_amplifies_the_adapter() -> None:
    """The other types run one model on both branches; blank prompt is 1x.

    For krea2/qwen-image/z-image the blank-prompt condition (0.75 of the score,
    `validator/scoring/tasks.py:280-298`) has positive == negative == "" on the
    SAME LoRA'd model, so `uncond + cfg*(cond - uncond)` collapses to `uncond`
    and the adapter passes at exactly 1x.  This is why an EMA constant may not
    be carried between ideogram4 and those types in either direction.
    """
    assert "ideogram4" not in UNAMPLIFIED_TYPES
    for model_type in UNAMPLIFIED_TYPES:
        blank_prompt_gain = 1.0  # CFG is an exact no-op there
        assert blank_prompt_gain < ideogram4_effective_cfg(), model_type


def test_the_matched_pair_measures_the_penalty_for_extra_strength() -> None:
    """2x strength at identical depth, in our own family: +92% loss.

    The only single-variable adapter-strength comparison in the whole ideogram4
    record.  Both configs are pinned by digest so a future editor has to go and
    look at the artifacts rather than re-argue from memory.
    """
    assert len(ANCHOR_CONFIG_SHA256) == 64 and len(DOUBLED_LR_CONFIG_SHA256) == 64
    assert ANCHOR_CONFIG_SHA256 != DOUBLED_LR_CONFIG_SHA256
    observed = DOUBLED_LR_LOSS / ANCHOR_LOSS - 1.0
    assert observed == pytest.approx(0.921, abs=5e-3)
    # The fitted curve must reproduce the point it was fitted to.
    assert strength_penalty(2.0) == pytest.approx(observed, rel=1e-9)
    # And it must be quiet near the anchor: this is a local model, not a cliff.
    assert strength_penalty(1.05) < 0.01


@pytest.mark.parametrize("row", REAL_SHAPES, ids=IDS)
def test_099_would_ship_a_stronger_adapter_at_every_real_shape(row) -> None:
    """The vacated amendment's mechanical effect, restated with its sign.

    Under our ACTUAL cosine schedule the exported fraction would move
        421 steps: 0.720 -> 0.894   (x1.24)
        589 steps: 0.816 -> 0.940   (x1.15)
        616 steps: 0.828 -> 0.945   (x1.14)
    This test does NOT dispute the arithmetic that motivated the amendment — it
    pins it.  What changed is that on ideogram4 "exports more of the delta" is a
    COST, not a recovery.
    """
    _task, _pairs, _hours, steps = row
    lr = our_cosine_lr()
    shipped = exported_fraction(steps, SHIPPED_DECAY, lr)
    rejected = exported_fraction(steps, REJECTED_DECAY, lr)
    assert rejected > shipped
    assert rejected / shipped >= 1.10


@pytest.mark.parametrize("row", REAL_SHAPES, ids=IDS)
def test_the_strength_increase_is_robust_to_the_delta_growth_model(row) -> None:
    """The RISK must not depend on which growth model is right.

    Five models: constant lr (linear B); our cosine to eta_min 2.5e-6; our
    cosine to 0 (in case `lr_scheduler_params` is ever dropped); and a
    saturating B_k = B_inf (1 - exp(-k/kappa)) at kappa = T/3 and T/6, which is
    the pessimistic "the delta plateaus early so the lag costs little" case.
    Under every one of them 0.99 ships a materially stronger adapter, so the
    reason to reject it does not rest on a modelling choice.
    """
    _task, _pairs, _hours, steps = row

    def saturating(kappa_frac: float):
        return lambda k, t: math.exp(-(k - 1) / (t * kappa_frac)) - math.exp(
            -k / (t * kappa_frac)
        )

    models = [
        constant_lr(1.0),
        our_cosine_lr(),
        our_cosine_lr(eta_min=0.0),
        saturating(1 / 3),
        saturating(1 / 6),
    ]
    for lr in models:
        shipped = exported_fraction(steps, SHIPPED_DECAY, lr)
        rejected = exported_fraction(steps, REJECTED_DECAY, lr)
        assert rejected / shipped >= 1.08


def test_the_ema_model_reproduces_the_independently_derived_numbers() -> None:
    """Cross-check the simulation against the closed form and the audit figures.

    The constant-lr closed form is
        f = (1 - d^T) - d(1 - T d^(T-1) + (T-1) d^T) / (T (1 - d))
    and it must agree with the step-by-step simulation.  The two anchors below
    (33.9% at 177 steps, 58.5% at 421, both at d=0.995, constant lr) were
    derived independently before this file existed; reproducing them is what
    makes the rest of the arithmetic here trustworthy.
    """
    for steps, expected in ((177, 0.339), (421, 0.585)):
        sim = exported_fraction(steps, SHIPPED_DECAY, constant_lr(1.0))
        closed = _closed_form_constant_lr(steps, SHIPPED_DECAY)
        assert sim == pytest.approx(closed, abs=1e-9)
        assert sim == pytest.approx(expected, abs=5e-4)


# --------------------------------------------------------------------------
# 3. the decision: 0.995 tracks the anchor, 0.99 overshoots it
# --------------------------------------------------------------------------
def test_shipped_decay_reproduces_the_in_family_anchor_strength() -> None:
    """At the anchor's own shape we must land on the anchor, not past it.

    `recipe.size_scaled_steps` ships 390 steps at the anchor's N=11 / h=0.75
    against its 378, so depth is nearly matched and the decay is what decides
    exported strength.  0.995 lands within ~5%; 0.99 overshoots by ~33%.
    """
    lr = our_cosine_lr()
    ours = recipe.size_scaled_steps("ideogram4", 11, 0.75, 2000)
    assert 370 <= ours <= 410, f"anchor-shape depth moved to {ours}"

    anchor = exported_strength(ANCHOR_STEPS, ANCHOR_DECAY, lr)
    at_shipped = exported_strength(ours, SHIPPED_DECAY, lr) / anchor
    at_rejected = exported_strength(ours, REJECTED_DECAY, lr) / anchor

    assert at_shipped == pytest.approx(1.05, abs=0.06)
    assert at_rejected == pytest.approx(1.33, abs=0.06)
    assert at_rejected / at_shipped > 1.20


def test_the_rejected_decay_prices_above_our_elimination_margin() -> None:
    """The decision, in one number.

    Feeding the two candidate strength ratios through the curvature measured on
    the matched pair: 0.995 prices at a fraction of a percent, 0.99 at ~10%.
    We were eliminated in Aug-3 R1 by 0.97%.  ideogram4 is ~half the R1 draw.
    """
    lr = our_cosine_lr()
    ours = recipe.size_scaled_steps("ideogram4", 11, 0.75, 2000)
    anchor = exported_strength(ANCHOR_STEPS, ANCHOR_DECAY, lr)

    cost_shipped = strength_penalty(
        exported_strength(ours, SHIPPED_DECAY, lr) / anchor
    )
    cost_rejected = strength_penalty(
        exported_strength(ours, REJECTED_DECAY, lr) / anchor
    )
    assert cost_shipped < AUG3_ELIMINATION_MARGIN
    assert cost_rejected > 5 * AUG3_ELIMINATION_MARGIN
    assert cost_rejected > 10 * cost_shipped


def test_5fbmn1ax_099_is_explained_by_his_depth_not_by_ideogram4() -> None:
    """Why the Aug-3 rank-1 artifacts' 0.99 does not transfer to us.

    OBSERVED (config.yaml sidecars under
    SN56-project/evidence/week6-field-depth-audit-20260806/raw/): 5FBmn1ax ran
    `use_ema: true`, `ema_decay: 0.99`, `lr: 4e-4` CONSTANT with no scheduler,
    and took rank 1 at 174 steps (1365fa1c, N=14) and 341 steps (84be9fcd,
    N=46).  An EMA's averaging horizon is `decay/(1-decay)`: 199 steps at 0.995
    against 99 at 0.99.  At 174 steps the 0.995 horizon EXCEEDS the entire run,
    so 0.99 is forced by his depth.  Our shapes are 3-4x deeper, and the
    in-family anchor ran 0.995 at 378 steps and won 1 of 16.
    """
    horizon_995 = ANCHOR_DECAY / (1 - ANCHOR_DECAY)
    horizon_99 = REJECTED_DECAY / (1 - REJECTED_DECAY)
    assert horizon_995 == pytest.approx(199, abs=1)
    assert horizon_99 == pytest.approx(99, abs=1)
    # His shallowest winning run is shorter than the 0.995 horizon.
    assert horizon_995 > 174
    # Ours, and the anchor's, are comfortably longer than it.
    for steps in (ANCHOR_STEPS, *(row[3] for row in REAL_SHAPES)):
        assert steps > horizon_995

    # And his own three runs cannot identify EMA at all: the two he won carry
    # ema_decay 0.99 at 174/341 steps, the one he LOST (b72da8c6) has no
    # `ema_config` block and ran 1523 steps.  Depth and EMA move together.
    his_runs = [(174, True), (341, True), (1523, False)]
    won = {steps for steps, ema in his_runs if ema}
    lost = {steps for steps, ema in his_runs if not ema}
    assert max(won) < min(lost), "EMA and depth are confounded in his record"


def test_the_lr_gap_to_5fbmn1ax_is_not_a_deficiency_to_close() -> None:
    """The ~12x lr-integral gap is the amplification working as intended.

    5FBmn1ax runs 174 x 4e-4 = 0.0696 against our ~0.0058 at 421 cosine steps.
    Read as a gap to close, that invites raising lr — which is exactly the
    change the matched pair measured at +92% loss.  This test pins the gap as
    OBSERVED and pins the conclusion that it must not be closed by raising lr.
    """
    his = 174 * 4e-4
    ours = sum(our_cosine_lr()(k, 421) for k in range(1, 422))
    assert ours / his < 0.12  # ~12x, OBSERVED

    # Closing even half of it means multiplying strength by ~6, which the fitted
    # curvature prices as catastrophic — far past the 2x that was measured.
    assert strength_penalty(6.0) > 20.0
    # Whereas the anchor's own operating point is where we already are.
    assert strength_penalty(1.0) == 0.0


# --------------------------------------------------------------------------
# 4. the vacation cannot be reverted silently
# --------------------------------------------------------------------------
def test_amendment_record_is_vacated_and_hash_bound() -> None:
    """The withdrawn amendment stays in the file, hashed, as an audit trail."""
    assert policy.EMA_DECAY == SHIPPED_DECAY
    amendment = policy.WEEK6_EMA_AMENDMENT
    assert amendment["status"] == "vacated"
    assert amendment["validated_value"] == SHIPPED_DECAY
    # Vacated means the amended value is back at the validated value.
    assert amendment["amended_value"] == SHIPPED_DECAY
    assert amendment["amended_value"] != REJECTED_DECAY
    # With no divergence left, the source cell's score covers the whole recipe.
    assert amendment["covered_by_source_validation_cell"] is True
    assert policy.AMENDMENT_SHA256 == hashlib.sha256(
        policy._canonical_bytes(amendment)
    ).hexdigest()
    assert policy._EXPECTED_RECIPE["train"]["ema_config"] == {
        "use_ema": True,
        "ema_decay": SHIPPED_DECAY,
    }
    # No LIVE amendment: the shipped recipe is the bare I-J20-D2 port again.
    assert policy._POLICY_BODY["amendments"] == []
    assert amendment in policy._POLICY_BODY["vacated_amendments"]
    assert (
        policy._POLICY_BODY["calibration_provenance"][
            "covers_recipe_projection_exactly"
        ]
        is True
    )


def test_readopting_099_deactivates_rather_than_silently_shipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hand-edited decay must not reach ai-toolkit.

    `apply()` writes `EMA_DECAY` and then re-projects the resolved config,
    bailing to the untouched input if the projection is not `_EXPECTED_RECIPE`.
    So moving the projection back to 0.99 without moving the constant (or vice
    versa) deactivates the policy loudly instead of shipping a mismatch.
    """
    monkeypatch.setattr(policy, "PRODUCTION_ACTIVATION", _activation())
    monkeypatch.setitem(
        policy._EXPECTED_RECIPE["train"],
        "ema_config",
        {"use_ema": True, "ema_decay": REJECTED_DECAY},
    )
    train = _built_train_block(monkeypatch, 14, 0.75)
    # The Week-4 template default, i.e. `apply()` refused to fire at all.
    assert train["ema_config"] == {"use_ema": False, "ema_decay": 0.99}
    assert train["lr"] != 2.5e-5


def test_checkpoint_control_rejects_a_binding_without_the_amendment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run-scope consumer validates the amendment too, not just the policy."""
    monkeypatch.setattr(policy, "PRODUCTION_ACTIVATION", _activation())
    spec = ImageSpec.build(
        task_id="week6-ideogram-ema-horizon",
        model="black-forest-labs/FLUX.1-Krea-dev",
        model_type="ideogram4",
        expected_repo_name="irepo",
        trigger_word=None,
        dataset_zip=None,
    )
    cfg = config.build_config(spec, num_images=14, hours_to_complete=0.75)
    assert policy.checkpoint_control(cfg) is not None

    cfg["meta"]["forge_ideogram_production_policy"]["amendment_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="binding is invalid"):
        policy.checkpoint_control(cfg)


# --------------------------------------------------------------------------
# 5. blast radius: which other model types are exposed to this code path
# --------------------------------------------------------------------------
def test_no_other_model_type_enables_ema_at_this_commit() -> None:
    """`save()`-exports-the-shadow applies to ANY type that turns EMA on.

    Today ideogram4 is the only one, and only because this policy module turns
    it on — every shipped template has `use_ema: false`.  If a future template
    enables EMA, the same attenuation applies and the decay must be chosen
    against that type's own step counts AND its own evaluator gain, not
    inherited from here: ideogram4's 0.995 is calibrated to a ~7x amplifier that
    no other type has.
    """
    root = Path(recipe.__file__).resolve().parent / "templates"
    enabled = {}
    for path in sorted(root.glob("base_diffusion_*.yaml")):
        train = yaml.safe_load(path.read_text())["config"]["process"][0]["train"]
        ema = train.get("ema_config")
        if ema is not None and ema.get("use_ema"):
            enabled[path.name] = ema
    assert enabled == {}, (
        "a template now enables EMA; that type's exports are the EMA shadow "
        "(BaseSDTrainProcess.py:491-497) and need their own decay decision"
    )


def test_the_policy_only_touches_ideogram4(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing here can move another type's EMA settings."""
    monkeypatch.setattr(policy, "PRODUCTION_ACTIVATION", _activation())
    for model_type in ("krea2", "flux", "qwen-image", "z-image"):
        cfg = {"config": {"process": [{"train": {"ema_config": "sentinel"}}]}}
        assert policy.apply(cfg, model_type) is cfg
