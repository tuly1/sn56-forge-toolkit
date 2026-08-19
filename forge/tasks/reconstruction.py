"""Held-out reconstruction scorer with the validator's exact metric semantics.

The validator scores one file (``checkpoint*/…last.safetensors``) by img2img
reconstruction of 2-5 hidden images: per image it draws the fixed seed list
``generate_reproducible_seeds(master_seed=42, n)``, runs a caption-guided and a
blank-prompt pass per seed, takes per-pixel MSE against the preprocessed source
image, means over seeds then images per pass, and combines the two passes at
0.25 (caption) / 0.75 (blank).  Sources, copied at upstream ``f7caab6c`` into
``evidence/week9-flux-lane-20260818/upstream-f7caab6c/``:

* geometry        image_io.py:16-40  (``adjust_image_size``)
* seeds           diffusion.py:35-37, 308
* loss            diffusion.py:203-209  (``calculate_l2_loss``)
* eval loop shape diffusion.py:289-318
* weighting       constants.py:24 (0.25) + validator/scoring/tasks.py:280-298
* per-type params constants.py:26-33 (``EVAL_DEFAULTS``)

This module reproduces those semantics against OUR OWN reserved holdout pairs
(never validator test rows — operating note 13) to rank the current run's
checkpoint ladder.  It is honest about what it is NOT: the validator renders
through ComfyUI on fp8-scaled eval bases; we render through the pinned
ai-toolkit bf16 training stack.  The metric is therefore declared
``proxy_not_validator_metric`` and named :data:`METRIC`, and promotion stays
behind the consumer's calibration gates (see ``forge.tasks.checkpoints``).

Structure:

* Pure, CPU-testable primitives: geometry, seeds, loss, weighting, per-type
  parameter table, greedy-soup arithmetic.  No torch import at module level.
* :func:`score_candidates` — the batch producer entry used by
  ``forge.tasks.holdout.produce``.  It spawns ONE worker subprocess that loads
  the model stack once and scores every candidate (a per-candidate subprocess
  would reload a ~12B transformer 5-8x and blow the reserve — krea2 lane
  REPORT §3.2), polls it against the run deadline, and kills it with margin.
  Any failure path returns no rows, so the producer writes no manifest and the
  true final checkpoint ships unchanged.
* ``python -m forge.tasks.reconstruction --worker order.json`` — the GPU
  worker.  All torch / ai-toolkit imports live behind that flag.

FLUX SEED NOTE (evidence/week9-flux-lane-20260818/REPORT.md §2.2, mechanism
OBSERVED): the flux evaluator's per-generation seed edit lands on a
BasicScheduler node that has no seed input; the actual noise source is a
RandomNoise node with a hard-coded ``noise_seed`` baked into lora_flux.json.
All 10 flux "generations" are therefore the identical deterministic render, so
ONE generation per image per pass reproduces the flux metric's mean exactly.
"""

from __future__ import annotations

import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

_FINALIZE_MARGIN_S = 30.0
# Do not even launch the worker without a useful window: model load alone is
# expected to take 60-120 s (INFERRED, krea2 lane §3.4; GPU runbook gate Q1).
_MIN_START_S = 120.0
_POLL_SECONDS = 2.0
# The worker reserves this much of its own budget for writing results and for
# the greedy-soup loop's abort-to-argmin path.
_WORKER_RESULT_MARGIN_S = 20.0

METRIC = "heldout_reconstruction_mse_v1"
MASTER_SEED = 42
CAPTIONED_WEIGHT = 0.25
BLANK_WEIGHT = 0.75

# Per-type sampler/guidance parameters mirroring the upstream evaluator's
# EVAL_DEFAULTS (constants.py:26-33 @ f7caab6c) plus the workflow facts needed
# to reproduce each type's render.  ``validator_generations`` is what the
# validator runs; ``scorer_generations`` is what WE run per image per pass:
#
# * flux = 1: the validator's flux seed is dead (module docstring), so its 10
#   generations are one deterministic render repeated; 1 reproduces the mean.
# * krea2 = 2: the full 0.25/0.75 metric at 2 of the validator's 5 seeds is
#   the only variant that fits the 900 s reserve at a 5-8 rung ladder
#   (krea2 lane REPORT §3.4: 2-seed full ≈ 600-840 s; 5-seed ≈ 1920 s).
# * ideogram4 = 2: same cost frame, but its DualModelGuider doubles per-step
#   forwards (un-LoRA'd negative branch), priced into the raised 900 s reserve
#   in forge.tasks.holdout.  Field rung deltas are 10-56% (ideogram4 lane §3),
#   so 2-seed rank noise is second-order.  INFERRED; GPU runbook gates Q1/Q3.
#
# ``schedule_semantics`` records which ComfyUI node family the validator's
# workflow uses so the worker reproduces the right denoise-window convention:
# * "ksampler_trailing": KSampler-style — sigma schedule built for ``steps``,
#   denoise d executes the last round(steps*d) of them (krea2/ideogram4
#   workflows edit a Sampler/Scheduler+denoise input; krea2 lane §3.1).
# * "basic_scheduler_stretched": BasicScheduler — schedule built for
#   int(steps/denoise) and the last ``steps`` are executed (flux lane §2.2).
# Both conventions are INFERRED from ComfyUI semantics, not re-read at a pin:
# GPU runbook gate Q7 (sigma-schedule parity) decides them.
EVAL_PARAMS: dict[str, dict[str, Any]] = {
    "krea2": {
        "steps": 25,
        "cfg": 12,
        "denoise": 0.8,
        "validator_generations": 5,
        "scorer_generations": 2,
        "sampler": "euler",
        "scheduler": "simple",
        "schedule_semantics": "ksampler_trailing",
        "guidance": "cfg_dual_pass",
    },
    "ideogram4": {
        "steps": 30,
        "cfg": 8,
        "denoise": 0.75,
        "validator_generations": 5,
        "scorer_generations": 2,
        "sampler": "euler",
        "scheduler": "simple",
        "schedule_semantics": "ksampler_trailing",
        # diffusion.py:224-238: Dual_model_guider cfg=8 with the negative
        # branch produced by the UN-LoRA'd model; CFG_override = max(cfg-3, 1).
        "guidance": "dual_model_unlora_negative",
        "cfg_override": 5,
    },
    "flux": {
        "steps": 35,
        # FluxGuidance embedder value, not a dual-pass CFG scale — the flux
        # workflow has no negative branch (flux lane §2.2).
        "cfg": 100,
        "denoise": 0.8,
        "validator_generations": 10,
        "scorer_generations": 1,
        "sampler": "dpmpp_2m",
        "scheduler": "sgm_uniform",
        "schedule_semantics": "basic_scheduler_stretched",
        "guidance": "distilled_embed_single_pass",
    },
}

_SOUP_ENV = "FORGE_RECON_SOUP_TYPES"


def implemented_types() -> frozenset[str]:
    return frozenset(EVAL_PARAMS)


def uses_reconstruction(model_type: str) -> bool:
    """Whether this architecture's holdout scoring is the reconstruction scorer."""
    return (model_type or "").strip().lower() in EVAL_PARAMS


def soup_enabled_for(model_type: str) -> bool:
    """Greedy-soup flag (scout REPORT §Q3 F3.1).  DEFAULT OFF per type.

    Argmin remains the shipped default this week; the soup only runs when this
    env names the type, and even then it must beat the argmin on the offline
    metric to be considered by the consumer.
    """
    mt = (model_type or "").strip().lower()
    if mt not in EVAL_PARAMS:
        return False
    raw = os.environ.get(_SOUP_ENV, "")
    allowed = {value.strip().lower() for value in raw.split(",") if value.strip()}
    return "*" in allowed or mt in allowed


def generate_reproducible_seeds(master_seed: int, n: int) -> list[int]:
    """Exact port of the evaluator's seed derivation (diffusion.py:35-37)."""
    random.seed(master_seed)
    return [random.randint(0, 2**32 - 1) for _ in range(n)]


def scorer_seeds(model_type: str) -> list[int]:
    """The first ``scorer_generations`` of the validator's own seed sequence.

    The validator draws ``generate_reproducible_seeds(42, n)`` per image
    (diffusion.py:308) — the same list every image — so a fidelity-preserving
    subset is a PREFIX of that exact list, not fresh seeds.
    """
    params = EVAL_PARAMS[(model_type or "").strip().lower()]
    return generate_reproducible_seeds(MASTER_SEED, params["validator_generations"])[
        : params["scorer_generations"]
    ]


def evaluator_size(width: int, height: int) -> tuple[int, int]:
    """Final scored dimensions for a (width, height) source (image_io.py:16-40)."""
    w, h = int(width), int(height)
    if w <= 0 or h <= 0:
        raise ValueError("non-positive image dimension")
    if w > h:
        new_w = 1024
        new_h = int((h / w) * 1024)
    else:
        new_h = 1024
        new_w = int((w / h) * 1024)
    floor_w = (new_w // 16) * 16
    floor_h = (new_h // 16) * 16
    return (min(new_w, floor_w), min(new_h, floor_h))


def preprocess_image(image):
    """Exact port of the evaluator's ``adjust_image_size`` (image_io.py:16-40).

    Order matters and is preserved: LANCZOS resize of the long edge to 1024
    (short edge truncated by ``int``), THEN floor both dims to a multiple of
    16, THEN centre-crop the RESIZED image to the floored dims with
    ``left=(w-cw)//2, top=(h-ch)//2``.  Sub-1024 images are UPSCALED, exactly
    as the evaluator does.  Takes and returns a PIL image.
    """
    from PIL import Image  # local: keep module import ML/Pillow-free for tools

    width, height = image.size
    if width > height:
        new_width = 1024
        new_height = int((height / width) * 1024)
    else:
        new_height = 1024
        new_width = int((width / height) * 1024)

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    new_width = (new_width // 16) * 16
    new_height = (new_height // 16) * 16

    width, height = image.size
    crop_width = min(width, new_width)
    crop_height = min(height, new_height)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    image = image.crop((left, top, left + crop_width, top + crop_height))
    return image


def image_mse(test_image, generated_image) -> float:
    """Exact port of ``calculate_l2_loss`` (diffusion.py:203-209)."""
    import numpy as np

    test = np.array(test_image.convert("RGB")) / 255.0
    generated = np.array(generated_image.convert("RGB")) / 255.0
    if test.shape != generated.shape:
        raise ValueError("Images must have the same dimensions to calculate L2 loss.")
    return float(np.mean((test - generated) ** 2))


def combine_pass_means(
    text_per_image: Sequence[float], blank_per_image: Sequence[float]
) -> tuple[float, float, float]:
    """Combine per-image pass means exactly as the validator does.

    ``eval_loop`` (diffusion.py:289-318) means over seeds per image (done by
    the caller), collects one value per image per pass; scoring then means over
    images per pass and weights 0.25/0.75.  Returns
    (combined, captioned_mean, blank_mean).
    """
    if not text_per_image or len(text_per_image) != len(blank_per_image):
        raise ValueError("per-image pass lists must be equal-length and non-empty")
    captioned = statistics.fmean(text_per_image)
    blank = statistics.fmean(blank_per_image)
    return (
        CAPTIONED_WEIGHT * captioned + BLANK_WEIGHT * blank,
        captioned,
        blank,
    )


def holdout_pairs_list(holdout_dir: str) -> list[tuple[str, str]]:
    """(image_path, caption_path) pairs in the reserved holdout directory."""
    pairs: list[tuple[str, str]] = []
    for name in sorted(os.listdir(holdout_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        image = os.path.join(holdout_dir, name)
        caption = os.path.join(holdout_dir, stem + ".txt")
        if os.path.isfile(image) and os.path.isfile(caption):
            pairs.append((image, caption))
    return pairs


# ---------------------------------------------------------------------------
# Greedy soup (Wortsman et al., ICML 2022 — scout REPORT §Q3 F3.1)
# ---------------------------------------------------------------------------


def soup_average(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Uniform average of LoRA state dicts; refuses any key/shape/dtype drift.

    Works on anything ndarray-like (torch tensors on the GPU worker, numpy in
    CPU tests).  Averaging checkpoints whose tensor key sets or shapes differ
    would silently combine different adapters — that is a hard error, never a
    best-effort merge.
    """
    if len(states) < 2:
        raise ValueError("soup needs at least two member states")
    reference = states[0]
    ref_keys = set(reference.keys())
    for index, state in enumerate(states[1:], start=1):
        keys = set(state.keys())
        if keys != ref_keys:
            missing = sorted(ref_keys ^ keys)[:5]
            raise ValueError(f"soup member {index} tensor keys differ (e.g. {missing})")
        for key in ref_keys:
            a, b = reference[key], state[key]
            if tuple(getattr(a, "shape", ())) != tuple(getattr(b, "shape", ())):
                raise ValueError(f"soup member {index} shape differs at {key!r}")
            if str(getattr(a, "dtype", "")) != str(getattr(b, "dtype", "")):
                raise ValueError(f"soup member {index} dtype differs at {key!r}")
    scale = 1.0 / len(states)
    out: dict[str, Any] = {}
    for key in reference.keys():
        total = states[0][key] * scale
        for state in states[1:]:
            total = total + state[key] * scale
        out[key] = total
    return out


def greedy_soup(
    ranked_names: Sequence[str],
    scores_by_name: Mapping[str, float],
    load_state: Callable[[str], Mapping[str, Any]],
    score_state: Callable[[Mapping[str, Any]], float],
    *,
    remaining_s: Callable[[], float],
    margin_s: float,
    est_eval_s: float,
) -> dict[str, Any]:
    """Greedy soup over an already-scored, best-first candidate ranking.

    Starts from the best rung; for each next-best rung, averages it into the
    accepted set and keeps the merge iff the offline score strictly improves.
    By construction the result is argmin-or-better ON THE OFFLINE METRIC.

    Budget contract (abort-to-argmin, NOT abort-to-last): every re-score costs
    a full evaluator pass, so before each trial we require
    ``remaining_s() > margin_s + est_eval_s``; on violation we STOP and return
    whatever has been accepted so far with status ``aborted_budget``.  Rung
    scores are already complete when this runs, so an abort here still leaves
    the argmin selection fully usable.

    Returns ``{"members", "state", "score", "status", "trials"}`` where
    ``state`` is None unless at least one merge was accepted.
    """
    if not ranked_names:
        raise ValueError("greedy soup needs a non-empty ranking")
    members = [ranked_names[0]]
    best_score = float(scores_by_name[ranked_names[0]])
    best_state: Mapping[str, Any] | None = None
    status = "no_improvement"
    trials = 0
    for name in ranked_names[1:]:
        if remaining_s() <= margin_s + est_eval_s:
            status = "aborted_budget"
            break
        trial_states = [load_state(member) for member in members] + [load_state(name)]
        trial_state = soup_average(trial_states)
        trial_score = float(score_state(trial_state))
        trials += 1
        if not math.isfinite(trial_score):
            raise ValueError("soup trial produced a non-finite score")
        if trial_score < best_score:
            members = members + [name]
            best_score = trial_score
            best_state = trial_state
            status = "improved"
    if best_state is None:
        return {
            "members": list(members),
            "state": None,
            "score": best_score,
            "status": status,
            "trials": trials,
        }
    return {
        "members": list(members),
        "state": best_state,
        "score": best_score,
        "status": status,
        "trials": trials,
    }


# ---------------------------------------------------------------------------
# Producer-side batch entry (fail-closed; used by forge.tasks.holdout.produce)
# ---------------------------------------------------------------------------


def _worker_cmd(order_path: str) -> list[str]:
    """Subprocess command line.  Module-level so tests can monkeypatch it."""
    return [sys.executable, "-m", "forge.tasks.reconstruction", "--worker", order_path]


def score_candidates(
    *,
    model_type: str,
    candidates: Sequence[str],
    holdout_dir: str,
    cfg: dict[str, Any],
    temp_root: str,
    deadline,
    soup_output_path: str | None = None,
) -> dict[str, Any]:
    """Score every candidate in one worker subprocess; raise on ANY defect.

    Raising is the fail-closed contract: the caller (``holdout.produce``)
    converts any exception into "no manifest", after which finalization ships
    the true final checkpoint unchanged.  This function never writes into
    ``save_root``; the optional soup artifact goes to ``soup_output_path``
    inside ``temp_root`` and is placed by the caller only after validation.
    """
    mt = (model_type or "").strip().lower()
    if mt not in EVAL_PARAMS:
        raise ValueError(f"reconstruction scorer does not implement {model_type!r}")
    if len(candidates) < 2:
        raise ValueError("reconstruction scoring needs at least two candidates")
    pairs = holdout_pairs_list(holdout_dir)
    if not pairs:
        raise RuntimeError("no holdout pairs available for reconstruction scoring")
    if deadline.remaining() <= _FINALIZE_MARGIN_S + _MIN_START_S:
        raise TimeoutError(
            "insufficient soft-deadline budget to start the reconstruction worker"
        )

    params = EVAL_PARAMS[mt]
    process = cfg["config"]["process"][0]
    order = {
        "schema": 1,
        "model_type": mt,
        "metric": METRIC,
        "candidates": [os.path.abspath(path) for path in candidates],
        "holdout_pairs": [
            {"image": image, "caption": caption} for image, caption in pairs
        ],
        "params": dict(params),
        "seeds": scorer_seeds(mt),
        "master_seed": MASTER_SEED,
        "captioned_weight": CAPTIONED_WEIGHT,
        "blank_caption_weight": BLANK_WEIGHT,
        # The worker rebuilds the training model stack from the same resolved
        # locations the trainer used (name_or_path + injected TE/VAE paths).
        "model": {
            "name_or_path": process.get("model", {}).get("name_or_path"),
            "model_kwargs": dict(process.get("model", {}).get("model_kwargs", {})),
            "arch": process.get("model", {}).get("arch"),
        },
        "network": {
            key: process.get("network", {}).get(key)
            for key in ("type", "linear", "linear_alpha")
        },
        # Self-budget: the worker must finish EVERYTHING (bar the optional
        # soup) inside this window or exit non-zero.  Partial coverage is a
        # failure, never a manifest.
        "budget_s": max(0.0, deadline.remaining() - _FINALIZE_MARGIN_S),
        "result_margin_s": _WORKER_RESULT_MARGIN_S,
        "soup": {
            "enabled": bool(soup_enabled_for(mt) and soup_output_path),
            "output_path": soup_output_path,
        },
    }
    order_path = os.path.join(temp_root, "recon-order.json")
    result_path = os.path.join(temp_root, "recon-result.json")
    order["result_path"] = result_path
    with open(order_path, "w", encoding="utf-8") as fh:
        json.dump(order, fh, sort_keys=True)

    log_path = os.path.join(temp_root, "recon-worker.log")
    # cwd = the directory containing the ``forge`` package, so the ``-m``
    # import resolves identically no matter where the trainer was launched.
    package_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            _worker_cmd(order_path),
            cwd=package_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while proc.poll() is None:
            if deadline.remaining() <= _FINALIZE_MARGIN_S:
                _terminate(proc)
                raise TimeoutError("reconstruction worker exceeded the deadline guard")
            time.sleep(_POLL_SECONDS)
    if proc.returncode != 0:
        tail = _tail(log_path)
        raise RuntimeError(
            f"reconstruction worker failed (rc={proc.returncode}): {tail}"
        )

    with open(result_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return validate_worker_payload(payload, order)


def validate_worker_payload(
    payload: dict[str, Any], order: dict[str, Any]
) -> dict[str, Any]:
    """Hard-validate the worker's result against the order.  Raises on defect."""
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        raise ValueError("worker payload schema mismatch")
    if payload.get("metric") != order["metric"]:
        raise ValueError("worker payload metric mismatch")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("worker payload rows must be a list")
    expected_names = [os.path.basename(path) for path in order["candidates"]]
    seen = [str(row.get("checkpoint")) for row in rows]
    if sorted(seen) != sorted(expected_names) or len(set(seen)) != len(seen):
        raise ValueError(
            "worker rows must cover every candidate exactly once: "
            f"got {sorted(seen)}, expected {sorted(expected_names)}"
        )
    pairs = len(order["holdout_pairs"])
    generations = int(order["params"]["scorer_generations"])
    expected_stratum_points = pairs * generations
    for row in rows:
        score = float(row.get("score"))
        captioned = float(row.get("captioned_score"))
        blank = float(row.get("blank_caption_score"))
        if any(not math.isfinite(v) or v < 0.0 for v in (score, captioned, blank)):
            raise ValueError(f"non-finite/negative score for {row.get('checkpoint')!r}")
        recombined = CAPTIONED_WEIGHT * captioned + BLANK_WEIGHT * blank
        if not math.isclose(score, recombined, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"combined score does not match strata for {row.get('checkpoint')!r}"
            )
        if (
            int(row.get("captioned_points")) != expected_stratum_points
            or int(row.get("blank_caption_points")) != expected_stratum_points
            or int(row.get("points")) != expected_stratum_points * 2
        ):
            raise ValueError(
                f"point coverage mismatch for {row.get('checkpoint')!r}: expected "
                f"{expected_stratum_points} per stratum"
            )
        for key in ("captioned_stddev", "blank_stddev"):
            spread = float(row.get(key))
            if not math.isfinite(spread) or spread < 0.0:
                raise ValueError(f"invalid dispersion for {row.get('checkpoint')!r}")
    soup = payload.get("soup")
    if soup is not None:
        _validate_soup_payload(soup, rows, order)
    return {"rows": rows, "soup": soup, "scorer": payload.get("scorer") or {}}


def _validate_soup_payload(
    soup: dict[str, Any], rows: list[dict[str, Any]], order: dict[str, Any]
) -> None:
    """A defective soup is a hard error — the caller then drops to argmin."""
    scores = {str(row["checkpoint"]): float(row["score"]) for row in rows}
    members = soup.get("members")
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError("soup payload must list at least two members")
    if any(str(member) not in scores for member in members):
        raise ValueError("soup members must be scored candidates")
    if len(set(members)) != len(members):
        raise ValueError("soup members must be unique")
    soup_score = float(soup.get("score"))
    argmin = min(scores.values())
    if not math.isfinite(soup_score) or soup_score >= argmin:
        raise ValueError(
            "soup score must strictly beat the argmin rung "
            f"({soup_score!r} vs {argmin!r})"
        )
    output_path = str(soup.get("output_path") or "")
    expected = str(order.get("soup", {}).get("output_path") or "")
    if not expected or output_path != expected or not os.path.isfile(output_path):
        raise ValueError("soup artifact path is missing or unexpected")


def _tail(log_path: str, n: int = 5) -> str:
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as fh:
            return " | ".join(line.strip() for line in fh.readlines()[-n:])[-500:]
    except Exception:
        return "<no worker log>"


def _terminate(proc: subprocess.Popen) -> None:
    def _signal_group(sig: int) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            proc.send_signal(sig)

    _signal_group(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_group(signal.SIGKILL)
    proc.wait(timeout=10)
    if proc.poll() is None:
        raise RuntimeError("reconstruction worker survived SIGKILL")


# ---------------------------------------------------------------------------
# GPU worker (runs as ``python -m forge.tasks.reconstruction --worker order``)
# ---------------------------------------------------------------------------
# Everything below imports torch / the pinned ai-toolkit lazily.  It cannot be
# exercised on CPU; the GPU-validation runbook
# (evidence/week9-selection-impl-20260818/GPU-VALIDATION-RUNBOOK.md) is the
# authority on whether these semantics match a ComfyUI reference render before
# any promotion flag is flipped.


def _worker_main(order_path: str) -> int:
    started = time.monotonic()
    with open(order_path, encoding="utf-8") as fh:
        order = json.load(fh)
    budget_s = float(order["budget_s"])
    result_margin_s = float(order.get("result_margin_s", _WORKER_RESULT_MARGIN_S))

    def remaining() -> float:
        return max(0.0, budget_s - (time.monotonic() - started))

    backend = _load_backend(order)
    from PIL import Image

    prepared = []
    for pair in order["holdout_pairs"]:
        with Image.open(pair["image"]) as raw:
            image = preprocess_image(raw)
        with open(pair["caption"], encoding="utf-8") as fh:
            caption = fh.read()
        prepared.append((image, caption))

    seeds = [int(seed) for seed in order["seeds"]]
    rows: list[dict[str, Any]] = []
    eval_costs: list[float] = []
    candidate_paths = list(order["candidates"])
    for path in candidate_paths:
        per_candidate = len(candidate_paths) - len(rows)
        if remaining() <= result_margin_s + (
            statistics.fmean(eval_costs) if eval_costs else 0.0
        ):
            # Partial coverage can never become a manifest; fail loudly so the
            # producer records the timeout and ships the true final.
            print(
                f"RECON_WORKER_BUDGET_EXHAUSTED remaining={remaining():.1f}s "
                f"candidates_left={per_candidate}",
                flush=True,
            )
            return 3
        t0 = time.monotonic()
        backend.attach_lora(path)
        row = _score_attached(backend, prepared, seeds, order)
        row["checkpoint"] = os.path.basename(path)
        rows.append(row)
        eval_costs.append(time.monotonic() - t0)

    soup_section = None
    soup_cfg = order.get("soup") or {}
    if soup_cfg.get("enabled"):
        soup_section = _worker_soup(
            backend,
            rows,
            candidate_paths,
            prepared,
            seeds,
            order,
            remaining=remaining,
            margin_s=result_margin_s,
            est_eval_s=(statistics.fmean(eval_costs) if eval_costs else 60.0),
        )

    payload = {
        "schema": 1,
        "metric": order["metric"],
        "rows": rows,
        "soup": soup_section,
        "scorer": {
            "backend": type(backend).__name__,
            "eval_cost_s": [round(cost, 2) for cost in eval_costs],
            "elapsed_s": round(time.monotonic() - started, 2),
        },
    }
    temp = order["result_path"] + ".tmp"
    with open(temp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temp, order["result_path"])
    return 0


def _score_attached(backend, prepared, seeds, order) -> dict[str, Any]:
    """Score the currently attached adapter over images x seeds x passes."""
    text_per_image: list[float] = []
    blank_per_image: list[float] = []
    all_text: list[float] = []
    all_blank: list[float] = []
    for image, caption in prepared:
        text_losses = []
        blank_losses = []
        for seed in seeds:
            generated = backend.reconstruct(image, caption, seed)
            text_losses.append(image_mse(image, generated))
            generated = backend.reconstruct(image, "", seed)
            blank_losses.append(image_mse(image, generated))
        text_per_image.append(statistics.fmean(text_losses))
        blank_per_image.append(statistics.fmean(blank_losses))
        all_text.extend(text_losses)
        all_blank.extend(blank_losses)
    combined, captioned, blank = combine_pass_means(text_per_image, blank_per_image)
    return {
        "score": combined,
        "captioned_score": captioned,
        "blank_caption_score": blank,
        "captioned_points": len(all_text),
        "blank_caption_points": len(all_blank),
        "points": len(all_text) + len(all_blank),
        "captioned_stddev": statistics.pstdev(all_text) if len(all_text) > 1 else 0.0,
        "blank_stddev": statistics.pstdev(all_blank) if len(all_blank) > 1 else 0.0,
    }


def _worker_soup(
    backend,
    rows,
    candidate_paths,
    prepared,
    seeds,
    order,
    *,
    remaining,
    margin_s,
    est_eval_s,
):
    """Greedy soup over the scored rungs.  NEVER raises into the rung result."""
    try:
        import safetensors.torch

        by_name = {os.path.basename(path): path for path in candidate_paths}
        ranked = [
            str(row["checkpoint"])
            for row in sorted(rows, key=lambda row: float(row["score"]))
        ]
        scores = {str(row["checkpoint"]): float(row["score"]) for row in rows}
        state_cache: dict[str, Any] = {}

        def load_state(name: str):
            if name not in state_cache:
                state_cache[name] = safetensors.torch.load_file(by_name[name])
            return state_cache[name]

        def score_state(state) -> float:
            backend.attach_lora_state(state)
            return float(_score_attached(backend, prepared, seeds, order)["score"])

        result = greedy_soup(
            ranked,
            scores,
            load_state,
            score_state,
            remaining_s=remaining,
            margin_s=margin_s,
            est_eval_s=est_eval_s,
        )
        if result["state"] is None:
            return {
                "status": result["status"],
                "members": result["members"],
                "score": None,
                "output_path": None,
                "trials": result["trials"],
            }
        output_path = order["soup"]["output_path"]
        safetensors.torch.save_file(dict(result["state"]), output_path)
        return {
            "status": result["status"],
            "members": result["members"],
            "score": result["score"],
            "output_path": output_path,
            "trials": result["trials"],
        }
    except BaseException as exc:  # abort-to-argmin: rung rows stay authoritative
        print(f"RECON_SOUP_FAILED {type(exc).__name__}: {exc}", flush=True)
        return {
            "status": f"soup_failed:{type(exc).__name__}",
            "members": [],
            "score": None,
            "output_path": None,
            "trials": 0,
        }


def _load_backend(order: dict[str, Any]):
    """Instantiate the per-type render backend on the pinned ai-toolkit stack.

    GPU-GATED: these imports and every call they enable execute only inside the
    validator/runbook container (AI_TOOLKIT_DIR on sys.path, torch + weights
    present).  The runbook's smoke is the proof they work; nothing here runs in
    CPU unit tests.
    """
    toolkit_dir = os.environ.get("AI_TOOLKIT_DIR", "/app/ai-toolkit")
    if toolkit_dir not in sys.path:
        sys.path.insert(0, toolkit_dir)
    model_type = order["model_type"]
    if model_type == "krea2":
        return _ToolkitBackend(order, arch="krea2")
    if model_type == "ideogram4":
        return _Ideogram4Backend(order)
    if model_type == "flux":
        return _FluxBackend(order)
    raise ValueError(f"no reconstruction backend for {model_type!r}")


class _ToolkitBackend:
    """img2img reconstruction through the pinned ai-toolkit model primitives.

    Uses the BaseModel quartet that krea2/ideogram4 expose at pin 99be3d96
    (verified in workspaces/repos/ai-toolkit at that commit):
    ``get_prompt_embeds`` / ``encode_images`` / ``get_noise_prediction`` /
    ``decode_latents``, plus ``LoRASpecialNetwork`` for adapter attach/detach.
    The denoise loop follows the per-type ``schedule_semantics`` in EVAL_PARAMS;
    GPU runbook gate Q7 owns sigma-schedule parity with the ComfyUI reference.
    """

    def __init__(self, order: dict[str, Any], arch: str):
        self._order = order
        self._params = order["params"]
        self._arch = arch
        self._network = None
        self._model = self._build_model(order, arch)

    # -- model/adapter management -------------------------------------------
    def _build_model(self, order, arch):
        import torch
        from toolkit.config_modules import ModelConfig
        from toolkit.util.get_model import get_model_class

        model_cfg = dict(order["model"].get("model_kwargs", {}))
        config = ModelConfig(
            name_or_path=order["model"]["name_or_path"],
            arch=arch,
            dtype="bf16",
            **{k: v for k, v in model_cfg.items() if v is not None},
        )
        model_class = get_model_class(config)
        model = model_class(
            device=torch.device("cuda"),
            model_config=config,
            dtype="bf16",
        )
        model.load_model()
        return model

    def attach_lora(self, path: str):
        """Attach adapter weights, mirroring the pin's own wiring.

        BaseSDTrainProcess.py:1767-1812 @ 99be3d96 is the authoritative
        pattern: construct LoRASpecialNetwork(text_encoder, unet, lora_dim,
        alpha, train_*, is_transformer, base_model), force_to fp32, hand it to
        ``sd.network``, ``_update_torch_multiplier()``, then POSITIONAL
        ``apply_to(text_encoder, unet, train_text_encoder, train_unet)``;
        weights load via ``network.load_weights(path)`` (:866-868).
        """
        import torch
        from toolkit.lora_special import LoRASpecialNetwork

        if self._network is None:
            network_cfg = self._order.get("network") or {}
            unet = self._model.get_model_to_train()
            network = LoRASpecialNetwork(
                text_encoder=None,
                unet=unet,
                lora_dim=int(network_cfg.get("linear") or 32),
                multiplier=1.0,
                alpha=int(network_cfg.get("linear_alpha") or 32),
                train_unet=False,
                train_text_encoder=False,
                is_transformer=getattr(self._model, "is_transformer", True),
                base_model=self._model,
            )
            network.force_to(self._model.device_torch, dtype=torch.float32)
            self._model.network = network
            network._update_torch_multiplier()
            network.apply_to(None, unet, False, True)
            network.eval()
            self._network = network
        self._network.load_weights(path)
        self.set_adapter_active(True)

    def attach_lora_state(self, state):
        """Soup states arrive in memory; persist once and reuse the path loader
        (``load_weights`` at the pin reads files, not dicts)."""
        import safetensors.torch

        temp = os.path.join(
            tempfile.gettempdir(), f"recon-soup-state-{os.getpid()}.safetensors"
        )
        safetensors.torch.save_file(dict(state), temp)
        self.attach_lora(temp)

    def set_adapter_active(self, active: bool):
        # ToolkitNetworkMixin drives module multipliers off is_active; refresh
        # them after every toggle.  GPU runbook gate E2 proves the toggle has a
        # real effect (score must change when the negative branch is LoRA'd).
        if self._network is not None:
            self._network.is_active = bool(active)
            self._network._update_torch_multiplier()

    # -- render ---------------------------------------------------------------
    def reconstruct(self, image, prompt: str, seed: int):
        import torch

        params = self._params
        latents = self._encode(image)
        sigmas = self._sigma_window(params)
        noise = torch.randn(
            latents.shape,
            generator=torch.Generator(device="cpu").manual_seed(int(seed)),
            device="cpu",
            dtype=torch.float32,
        ).to(latents.device, latents.dtype)
        start_sigma = float(sigmas[0])
        current = latents * (1.0 - start_sigma) + noise * start_sigma
        cond = self._model.get_prompt_embeds(prompt)
        uncond = self._model.get_prompt_embeds("")
        cfg = float(params["cfg"])
        for index in range(len(sigmas) - 1):
            sigma, sigma_next = float(sigmas[index]), float(sigmas[index + 1])
            timestep = torch.tensor([sigma * 1000.0], device=current.device)
            velocity = self._guided_velocity(current, timestep, cond, uncond, cfg, prompt)
            current = current + velocity * (sigma_next - sigma)
        return self._decode(current)

    def _guided_velocity(self, latent, timestep, cond, uncond, cfg, prompt):
        cond_v = self._model.get_noise_prediction(
            latent_model_input=latent, timestep=timestep, text_embeddings=cond
        )
        if prompt == "":
            # Blank pass: prompt == negative prompt == "", so CFG cancels
            # exactly (u + s*(c-u) == u) — one forward is the exact value
            # (krea2 lane REPORT §3.1).
            return cond_v
        uncond_v = self._model.get_noise_prediction(
            latent_model_input=latent, timestep=timestep, text_embeddings=uncond
        )
        return uncond_v + cfg * (cond_v - uncond_v)

    def _sigma_window(self, params):
        steps = int(params["steps"])
        denoise = float(params["denoise"])
        if params["schedule_semantics"] == "basic_scheduler_stretched":
            total = int(steps / denoise)
            executed = steps
        else:  # ksampler_trailing
            total = steps
            executed = max(1, round(steps * denoise))
        # "simple" scheduler: uniform in flow time from 1 -> 0 over ``total``
        # steps; execute the LAST ``executed`` of them (start from the sigma at
        # depth ``denoise``).  Parity with ComfyUI + each arch's mu-shift is
        # GPU runbook gate Q7.
        full = [1.0 - index / total for index in range(total + 1)]
        return full[total - executed :]

    def _encode(self, image):
        import numpy as np
        import torch

        array = np.array(image.convert("RGB")).astype("float32") / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1) * 2.0 - 1.0
        return self._model.encode_images([tensor])

    def _decode(self, latents):
        from PIL import Image
        import numpy as np

        decoded = self._model.decode_latents(latents)
        array = decoded[0].float().clamp(-1, 1).permute(1, 2, 0).cpu().numpy()
        array = ((array + 1.0) / 2.0 * 255.0).round().astype("uint8")
        return Image.fromarray(array)


class _Ideogram4Backend(_ToolkitBackend):
    """Ideogram4: DualModelGuider semantics (diffusion.py:224-238).

    The negative branch comes from the UN-LoRA'd model on BOTH passes, so the
    blank pass does NOT cancel: output = neg + cfg*(cond - neg) with the
    adapter active only on the cond branch — the ~8x amplification channel the
    onboarding describes.  Costs two forwards per step on both passes (priced
    into the raised 900 s ideogram4 reserve).
    """

    def __init__(self, order: dict[str, Any]):
        super().__init__(order, arch="ideogram4")

    def _guided_velocity(self, latent, timestep, cond, uncond, cfg, prompt):
        embeds = cond if prompt != "" else uncond
        self.set_adapter_active(True)
        cond_v = self._model.get_noise_prediction(
            latent_model_input=latent, timestep=timestep, text_embeddings=embeds
        )
        self.set_adapter_active(False)
        neg_v = self._model.get_noise_prediction(
            latent_model_input=latent, timestep=timestep, text_embeddings=uncond
        )
        self.set_adapter_active(True)
        return neg_v + cfg * (cond_v - neg_v)


class _FluxBackend:
    """Flux: single-branch render with a FluxGuidance embed (no negative pass).

    Flux routes through ai-toolkit's legacy StableDiffusion class at the pin
    (forge/geometry.py BUCKET_DIVISIBILITY note), which exposes
    ``encode_prompt`` / ``encode_images`` / ``predict_noise`` /
    ``decode_latents``.  guidance_scale is the distilled-guidance embedder
    value (workflow FluxGuidance=100), not a dual-pass CFG.
    """

    def __init__(self, order: dict[str, Any]):
        import torch
        from toolkit.config_modules import ModelConfig
        from toolkit.stable_diffusion_model import StableDiffusion

        self._order = order
        self._params = order["params"]
        model_cfg = dict(order["model"].get("model_kwargs", {}))
        config = ModelConfig(
            name_or_path=order["model"]["name_or_path"],
            is_flux=True,
            dtype="bf16",
            **{k: v for k, v in model_cfg.items() if v is not None},
        )
        self._model = StableDiffusion(
            device=torch.device("cuda"), model_config=config, dtype="bf16"
        )
        self._model.load_model()
        self._network = None

    def attach_lora(self, path: str):
        # Same pinned wiring as _ToolkitBackend.attach_lora, on the legacy
        # class's unet handle with is_flux=True.
        import torch
        from toolkit.lora_special import LoRASpecialNetwork

        if self._network is None:
            network_cfg = self._order.get("network") or {}
            unet = self._model.unet
            network = LoRASpecialNetwork(
                text_encoder=None,
                unet=unet,
                lora_dim=int(network_cfg.get("linear") or 32),
                multiplier=1.0,
                alpha=int(network_cfg.get("linear_alpha") or 32),
                train_unet=False,
                train_text_encoder=False,
                is_flux=True,
                base_model=self._model,
            )
            network.force_to(self._model.device_torch, dtype=torch.float32)
            self._model.network = network
            network._update_torch_multiplier()
            network.apply_to(None, unet, False, True)
            network.eval()
            self._network = network
        self._network.load_weights(path)
        self._network.is_active = True
        self._network._update_torch_multiplier()

    def attach_lora_state(self, state):
        import safetensors.torch

        temp = os.path.join(
            tempfile.gettempdir(), f"recon-soup-state-{os.getpid()}.safetensors"
        )
        safetensors.torch.save_file(dict(state), temp)
        self.attach_lora(temp)

    def reconstruct(self, image, prompt: str, seed: int):
        import torch

        params = self._params
        latents = self._encode(image)
        steps = int(params["steps"])
        denoise = float(params["denoise"])
        total = int(steps / denoise)  # basic_scheduler_stretched
        full = [1.0 - index / total for index in range(total + 1)]
        sigmas = full[total - steps :]
        noise = torch.randn(
            latents.shape,
            generator=torch.Generator(device="cpu").manual_seed(int(seed)),
            device="cpu",
            dtype=torch.float32,
        ).to(latents.device, latents.dtype)
        start_sigma = float(sigmas[0])
        current = latents * (1.0 - start_sigma) + noise * start_sigma
        embeds = self._model.encode_prompt(prompt)
        for index in range(len(sigmas) - 1):
            sigma, sigma_next = float(sigmas[index]), float(sigmas[index + 1])
            timestep = torch.tensor([sigma * 1000.0], device=current.device)
            pred = self._model.predict_noise(
                latents=current,
                text_embeddings=embeds,
                timestep=timestep,
                guidance_scale=float(params["cfg"]),
            )
            current = current + pred * (sigma_next - sigma)
        return self._decode(current)

    def _encode(self, image):
        import numpy as np
        import torch

        array = np.array(image.convert("RGB")).astype("float32") / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1) * 2.0 - 1.0
        return self._model.encode_images([tensor])

    def _decode(self, latents):
        from PIL import Image
        import numpy as np

        decoded = self._model.decode_latents(latents)
        array = decoded[0].float().clamp(-1, 1).permute(1, 2, 0).cpu().numpy()
        array = ((array + 1.0) / 2.0 * 255.0).round().astype("uint8")
        return Image.fromarray(array)


def _main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "--worker":
        return _worker_main(argv[1])
    print("usage: python -m forge.tasks.reconstruction --worker <order.json>")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
