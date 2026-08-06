"""Week-6 amendment: the ideogram4 EMA export horizon.

WHAT IS BEING DEFENDED.  At ai-toolkit pin
``99be3d96a2468d3a5228a4eb05ba67e63c586b4e`` — the commit the production Docker
image builds, and the only runtime available on Monday — every artifact we
upload is the EMA *shadow*, never the trained weights:

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
    therefore starts at *zero adapter effect*: the defect is attenuation of the
    trained delta, not junk weights.
  * ``toolkit/ema.py:47,57,118`` — ``setup_ema()`` does not pass
    ``use_num_updates``, so ``self.num_updates`` is ``None`` and the
    ``(1 + n) / (10 + n)`` warm-up ramp guarded by ``if self.num_updates is not
    None`` never runs.  Decay is flat from step 1 and there is NO config path
    to change that.

So ``ema_decay`` is one of only two config-reachable levers (the other being
``use_ema``), and the amendment moves it 0.995 -> 0.99 — the value BOTH rank-1
ideogram4 artifacts in the Aug-3 field actually published.

Every test below fails if the amendment is reverted.
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
    # task, n_pairs, hours, steps we ship
    ("1365fa1c", 14, 0.75, 421),  # R1 draw shape; tightest budget
    ("84be9fcd", 46, 1.0, 616),
    ("b72da8c6", 40, 1.0, 589),
]
IDS = [row[0] for row in REAL_SHAPES]

VALIDATED_DECAY = 0.995  # what I-J20-D2 ran, and what this amendment replaces
AMENDED_DECAY = 0.99  # 5FBmn1ax's published rank-1 ideogram4 value


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
def test_emitted_ideogram_config_carries_the_amended_decay(
    row, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real `build_config` path, at every real shape, emits 0.99."""
    _task, pairs, hours, steps = row
    train = _built_train_block(monkeypatch, pairs, hours)
    assert train["steps"] == steps
    assert train["ema_config"] == {"use_ema": True, "ema_decay": AMENDED_DECAY}
    assert train["ema_config"]["ema_decay"] != VALIDATED_DECAY


def test_amended_decay_survives_the_yaml_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`config.write_config` is a bare `yaml.safe_dump` — prove it stays 0.99.

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
    assert ema == {"use_ema": True, "ema_decay": AMENDED_DECAY}
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
# 2. the quantified effect
# --------------------------------------------------------------------------
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
        sim = exported_fraction(steps, VALIDATED_DECAY, constant_lr(1.0))
        closed = _closed_form_constant_lr(steps, VALIDATED_DECAY)
        assert sim == pytest.approx(closed, abs=1e-9)
        assert sim == pytest.approx(expected, abs=5e-4)


@pytest.mark.parametrize("row", REAL_SHAPES, ids=IDS)
def test_amendment_recovers_delta_at_every_real_shape(row) -> None:
    """0.99 exports strictly more of the trained delta than 0.995, everywhere.

    Under our ACTUAL cosine schedule the exported fraction moves
        421 steps: 0.720 -> 0.894   (x1.24)
        616 steps: 0.828 -> 0.945   (x1.14)
        589 steps: 0.816 -> 0.940   (x1.15)
    The lower bounds asserted here are deliberately slack; what must not move is
    the DIRECTION and the fact that 0.995 falls below the bound 0.99 clears.
    """
    _task, _pairs, _hours, steps = row
    lr = our_cosine_lr()
    before = exported_fraction(steps, VALIDATED_DECAY, lr)
    after = exported_fraction(steps, AMENDED_DECAY, lr)
    assert after > before
    assert after >= 0.88, "the amended decay must export ~90%+ of the delta"
    assert before < 0.85, "reverting to 0.995 must visibly cost delta"
    assert after / before >= 1.10


@pytest.mark.parametrize("row", REAL_SHAPES, ids=IDS)
def test_the_gain_is_robust_to_the_delta_growth_model(row) -> None:
    """The conclusion must not depend on which growth model is right.

    Five models: constant lr (linear B); our cosine to eta_min 2.5e-6; our
    cosine to 0 (in case `lr_scheduler_params` is ever dropped); and a
    saturating B_k = B_inf (1 - exp(-k/kappa)) at kappa = T/3 and T/6, which is
    the pessimistic "the delta plateaus early so the lag costs little" case.
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
        before = exported_fraction(steps, VALIDATED_DECAY, lr)
        after = exported_fraction(steps, AMENDED_DECAY, lr)
        assert after > before
        assert after / before >= 1.08


def test_disabling_ema_would_add_little_over_the_amendment() -> None:
    """Why the amendment stops at 0.99 instead of `use_ema: false`.

    At the R1 shape (421 steps, our cosine) the remaining headroom above 0.99 is
    only ~1.12x, against the ~1.24x the amendment already takes.  Option (a)
    would spend the entire remaining deviation budget from the field's published
    rank-1 ideogram4 recipe to buy the smaller half of the effect.
    """
    lr = our_cosine_lr()
    at_995 = exported_fraction(421, VALIDATED_DECAY, lr)
    at_99 = exported_fraction(421, AMENDED_DECAY, lr)
    no_ema = 1.0
    taken = at_99 / at_995
    remaining = no_ema / at_99
    assert taken > remaining, "the amendment must capture the larger share"
    assert remaining < 1.15


def test_the_field_rank1_artifacts_also_shipped_an_attenuated_export() -> None:
    """Attenuation is not by itself what separates us from the winner.

    OBSERVED (week6-field-depth-audit-20260806/analysis.json): 5FBmn1ax ran
    `use_ema: true`, `ema_decay: 0.99`, `lr: 4e-4` CONSTANT, and took rank 1 at
    174 steps (1365fa1c, N=14) and 341 steps (84be9fcd, N=46).  Under the same
    model those artifacts exported only 53% and 72% of their own trained delta.

    They won anyway because their lr INTEGRAL dwarfs ours: 174 x 4e-4 = 0.0696
    against our 421 cosine steps ~ 0.0058, a ~12x gap.  This test exists so the
    amendment is never oversold: it closes an export leak, not the lr gap, and
    the lr gap is the larger remaining factor on ideogram4.
    """
    winner_f = exported_fraction(174, AMENDED_DECAY, constant_lr(4e-4))
    assert winner_f == pytest.approx(0.530, abs=5e-3)

    winner_trained = 174 * 4e-4
    ours_trained = sum(our_cosine_lr()(k, 421) for k in range(1, 422))
    assert ours_trained / winner_trained < 0.12

    ours_exported = ours_trained * exported_fraction(421, AMENDED_DECAY, our_cosine_lr())
    winner_exported = winner_trained * winner_f
    # We move from ~11% to ~14% of the winner's exported parameter movement.
    assert 0.12 < ours_exported / winner_exported < 0.17


# --------------------------------------------------------------------------
# 3. the amendment cannot be reverted silently
# --------------------------------------------------------------------------
def test_amendment_record_is_hash_bound_to_the_shipped_decay() -> None:
    """Editing the decay without re-signing deactivates the whole policy."""
    assert policy.EMA_DECAY == AMENDED_DECAY
    assert policy.WEEK6_EMA_AMENDMENT["amended_value"] == AMENDED_DECAY
    assert policy.WEEK6_EMA_AMENDMENT["validated_value"] == VALIDATED_DECAY
    # The amendment is explicitly NOT covered by the I-J20-D2 score.  If this
    # ever flips to True someone has quietly re-labelled an unvalidated field as
    # validated.
    assert policy.WEEK6_EMA_AMENDMENT["covered_by_source_validation_cell"] is False
    assert policy.AMENDMENT_SHA256 == hashlib.sha256(
        policy._canonical_bytes(policy.WEEK6_EMA_AMENDMENT)
    ).hexdigest()
    assert policy._EXPECTED_RECIPE["train"]["ema_config"] == {
        "use_ema": True,
        "ema_decay": AMENDED_DECAY,
    }
    # The amendment is inside the hashed policy body, so POLICY_SHA256 — and
    # therefore the activation record — moves with it.
    assert policy.WEEK6_EMA_AMENDMENT in policy._POLICY_BODY["amendments"]


def test_reverting_the_decay_deactivates_rather_than_silently_ships(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hand-edited decay must not reach ai-toolkit.

    `apply()` re-projects the resolved config and bails to the untouched input
    if the projection is not `_EXPECTED_RECIPE`, so a decay that disagrees with
    the hash-bound recipe cannot be emitted with the policy's blessing.
    """
    monkeypatch.setattr(policy, "PRODUCTION_ACTIVATION", _activation())
    monkeypatch.setitem(
        policy._EXPECTED_RECIPE["train"],
        "ema_config",
        {"use_ema": True, "ema_decay": VALIDATED_DECAY},
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
# 4. blast radius: which other model types are exposed to this code path
# --------------------------------------------------------------------------
def test_no_other_model_type_enables_ema_at_this_commit() -> None:
    """`save()`-exports-the-shadow applies to ANY type that turns EMA on.

    Today ideogram4 is the only one, and only because this policy module turns
    it on — every shipped template has `use_ema: false`.  If a future template
    enables EMA, the same attenuation applies and the decay must be chosen
    against that type's own step counts, not inherited from here.  This test is
    the tripwire.
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
