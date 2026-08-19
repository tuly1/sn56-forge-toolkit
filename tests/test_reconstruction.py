"""CPU contract tests for the week-9 reconstruction selection scorer.

Covers: evaluator-geometry parity, seed determinism, weighting math,
producer fail-closed behaviour (exception AND timeout), atomic promotion,
per-type enablement, selection cadence, greedy soup, and the promotion gates.

Real-image fixtures come from the Aug-17 tournament harvest's TRAIN rows
(evidence/aug17-tournament-dataset-harvest-20260818/*/pairs-train/) — never
from any QUARANTINE-* directory (operating note 13).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
import time

import numpy as np
from PIL import Image
import pytest

from forge import recipe
from forge.clock import Deadline
from forge.data.schema import ImageSpec
from forge.tasks import checkpoints, holdout, reconstruction

# SN56-project/evidence/... — five levels above this file both in the main
# checkout (workspaces/repos/forge-toolkit/tests) and in a worktree
# (workspaces/worktrees/<name>/tests).
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
_HARVEST = os.path.join(
    _PROJECT_ROOT, "evidence", "aug17-tournament-dataset-harvest-20260818"
)


def _harvest_images(limit=4):
    """Real tournament TRAIN images (skips entirely if the harvest is absent)."""
    out = []
    if not os.path.isdir(_HARVEST):
        return out
    for task in sorted(os.listdir(_HARVEST)):
        if task.startswith("QUARANTINE"):
            continue
        pairs = os.path.join(_HARVEST, task, "pairs-train")
        if not os.path.isdir(pairs):
            continue
        for name in sorted(os.listdir(pairs)):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                out.append(os.path.join(pairs, name))
                if len(out) >= limit:
                    return out
    return out


# ---------------------------------------------------------------------------
# Reference implementation: VERBATIM upstream logic (image_io.py:16-40 at
# f7caab6c, copied from evidence/week9-flux-lane-20260818/upstream-f7caab6c/).
# Our port must match this byte-for-byte in behaviour.
# ---------------------------------------------------------------------------


def _upstream_adjust_image_size(image):
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
    right = left + crop_width
    bottom = top + crop_height
    image = image.crop((left, top, right, bottom))

    return image


_ODD_SIZES = [
    (1024, 1024),  # exact-1024 square: identity
    (2000, 1000),  # clean 2:1 downscale
    (1023, 767),   # just under 1024: still rescaled to long edge 1024
    (1024, 767),   # exact long edge, odd short edge
    (500, 300),    # sub-1024: UPSCALED (the evaluator upscales, port must too)
    (300, 500),    # portrait sub-1024
    (777, 999),    # odd portrait
    (999, 777),    # odd landscape
    (1919, 1079),  # 1080p-ish odd
    (33, 17),      # degenerate tiny
    (1024, 1008),  # already /16 after resize
    (1025, 1024),  # barely landscape
]


def _noise_image(size, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(
        rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8), "RGB"
    )


@pytest.mark.parametrize("size", _ODD_SIZES)
def test_preprocess_matches_upstream_reference_pixels(size):
    image = _noise_image(size, seed=size[0] * 31 + size[1])
    ours = reconstruction.preprocess_image(image)
    reference = _upstream_adjust_image_size(image)
    assert ours.size == reference.size
    assert np.array_equal(np.array(ours), np.array(reference))
    # evaluator_size predicts the scored dims without touching pixels
    assert ours.size == reconstruction.evaluator_size(*size)
    # scored dims are always /16 with the long edge exactly 1024
    w, h = ours.size
    assert w % 16 == 0 and h % 16 == 0
    assert max(w, h) <= 1024


def test_preprocess_expected_dims_precomputed():
    # Hand-computed from the upstream arithmetic (CHANGES.md §1).
    assert reconstruction.evaluator_size(1024, 1024) == (1024, 1024)
    assert reconstruction.evaluator_size(2000, 1000) == (1024, 512)
    assert reconstruction.evaluator_size(1023, 767) == (1024, 752)
    assert reconstruction.evaluator_size(500, 300) == (1024, 608)
    assert reconstruction.evaluator_size(300, 500) == (608, 1024)
    assert reconstruction.evaluator_size(777, 999) == (784, 1024)


def test_preprocess_matches_upstream_on_real_tournament_images():
    images = _harvest_images(limit=4)
    if not images:
        pytest.skip("aug17 harvest not present on this machine")
    for path in images:
        with Image.open(path) as raw:
            image = raw.convert("RGB")
            ours = reconstruction.preprocess_image(image)
            reference = _upstream_adjust_image_size(image)
        assert ours.size == reference.size
        assert np.array_equal(np.array(ours), np.array(reference))


def test_image_mse_matches_upstream_semantics():
    a = _noise_image((64, 48), seed=1)
    b = _noise_image((64, 48), seed=2)
    ours = reconstruction.image_mse(a, b)
    reference = float(
        np.mean((np.array(a) / 255.0 - np.array(b) / 255.0) ** 2)
    )
    assert ours == pytest.approx(reference, rel=1e-12)
    assert reconstruction.image_mse(a, a) == 0.0
    with pytest.raises(ValueError):
        reconstruction.image_mse(a, _noise_image((48, 64)))


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

# Exact expected sequence for random.seed(42) (Mersenne Twister is stable
# across CPython versions); first 10 as the flux evaluator would draw them.
_EXPECTED_SEEDS_42 = [
    2746317213,
    1181241943,
    958682846,
    3163119785,
    1812140441,
    127978094,
    939042955,
    2340505846,
    946785248,
    2530876844,
]


def test_seed_sequence_is_the_validators_exactly():
    assert (
        reconstruction.generate_reproducible_seeds(42, 10) == _EXPECTED_SEEDS_42
    )
    # deterministic across repeated calls (the evaluator reseeds per image)
    assert (
        reconstruction.generate_reproducible_seeds(42, 10) == _EXPECTED_SEEDS_42
    )
    assert (
        reconstruction.generate_reproducible_seeds(42, 5)
        == _EXPECTED_SEEDS_42[:5]
    )


def test_scorer_seeds_are_a_prefix_of_the_validator_sequence():
    # krea2/ideogram4: validator runs 5 generations; we run the first 2 seeds.
    assert reconstruction.scorer_seeds("krea2") == _EXPECTED_SEEDS_42[:2]
    assert reconstruction.scorer_seeds("ideogram4") == _EXPECTED_SEEDS_42[:2]
    # flux: the validator's per-generation seed edit lands on a node with no
    # seed input (flux lane REPORT §2.2) — one generation reproduces the mean.
    assert reconstruction.scorer_seeds("flux") == _EXPECTED_SEEDS_42[:1]
    assert reconstruction.EVAL_PARAMS["flux"]["scorer_generations"] == 1
    assert reconstruction.EVAL_PARAMS["flux"]["validator_generations"] == 10


def test_eval_params_mirror_upstream_constants():
    # constants.py:26-33 @ f7caab6c
    assert reconstruction.EVAL_PARAMS["krea2"]["steps"] == 25
    assert reconstruction.EVAL_PARAMS["krea2"]["cfg"] == 12
    assert reconstruction.EVAL_PARAMS["krea2"]["denoise"] == 0.8
    assert reconstruction.EVAL_PARAMS["ideogram4"]["steps"] == 30
    assert reconstruction.EVAL_PARAMS["ideogram4"]["cfg"] == 8
    assert reconstruction.EVAL_PARAMS["ideogram4"]["denoise"] == 0.75
    assert reconstruction.EVAL_PARAMS["ideogram4"]["cfg_override"] == 5
    assert reconstruction.EVAL_PARAMS["flux"]["steps"] == 35
    assert reconstruction.EVAL_PARAMS["flux"]["cfg"] == 100
    assert reconstruction.EVAL_PARAMS["flux"]["denoise"] == 0.8
    assert reconstruction.CAPTIONED_WEIGHT == 0.25
    assert reconstruction.BLANK_WEIGHT == 0.75


def test_combine_pass_means_matches_eval_loop_shape():
    combined, captioned, blank = reconstruction.combine_pass_means(
        [0.2, 0.4], [0.1, 0.3]
    )
    assert captioned == pytest.approx(0.3)
    assert blank == pytest.approx(0.2)
    assert combined == pytest.approx(0.25 * 0.3 + 0.75 * 0.2)
    with pytest.raises(ValueError):
        reconstruction.combine_pass_means([0.1], [])


# ---------------------------------------------------------------------------
# Enablement / reserves / cadence
# ---------------------------------------------------------------------------


def test_per_type_enablement(monkeypatch):
    monkeypatch.setenv("FORGE_HOLDOUT_SELECTION_TYPES", "krea2,ideogram4,flux")
    for mt in ("krea2", "ideogram4", "flux"):
        assert holdout.enabled_for(mt)
        assert holdout.scoring_reserve_s(mt) == 900.0
    # z-image / qwen-image are OUT this week, even under a wildcard
    monkeypatch.setenv("FORGE_HOLDOUT_SELECTION_TYPES", "*")
    assert not holdout.enabled_for("z-image")
    assert not holdout.enabled_for("qwen-image")
    monkeypatch.delenv("FORGE_HOLDOUT_SELECTION_TYPES")
    for mt in ("krea2", "ideogram4", "flux"):
        assert not holdout.enabled_for(mt)
        assert holdout.scoring_reserve_s(mt) == 0.0


def test_reconstruction_routing_is_type_scoped():
    for mt in ("krea2", "ideogram4", "flux"):
        assert reconstruction.uses_reconstruction(mt)
    for mt in ("z-image", "qwen-image", "sdxl", "", None):
        assert not reconstruction.uses_reconstruction(mt)


def test_selection_cadence_dormant_equivalence(monkeypatch):
    monkeypatch.delenv("FORGE_HOLDOUT_SELECTION_TYPES", raising=False)
    for steps in (24, 86, 367, 456, 944, 1432, 2000):
        assert recipe.selection_save_every(
            "krea2", steps, 250
        ) == recipe.kill_safe_save_every(steps, 250)


def test_selection_cadence_active_gives_dense_ladder(monkeypatch):
    monkeypatch.setenv("FORGE_HOLDOUT_SELECTION_TYPES", "krea2,ideogram4,flux")
    assert recipe.selection_save_every("krea2", 1432, 250) == 200
    assert recipe.selection_save_every("ideogram4", 944, 250) == 200
    assert recipe.selection_save_every("flux", 1002, 250) == 200
    # short plans keep at least one mid-run rung and never go below 25
    assert recipe.selection_save_every("krea2", 90, 250) == 46
    assert recipe.selection_save_every("krea2", 60, 250) == 31
    assert recipe.selection_save_every("krea2", 20, 250) == (
        recipe.kill_safe_save_every(20, 250)
    )
    # non-enabled type stays legacy even when the env is set
    assert recipe.selection_save_every(
        "z-image", 1432, 250
    ) == recipe.kill_safe_save_every(1432, 250)


# ---------------------------------------------------------------------------
# Greedy soup arithmetic (scout REPORT §Q3 F3.1)
# ---------------------------------------------------------------------------


def _state(**tensors):
    return {key: np.asarray(value, dtype=np.float32) for key, value in tensors.items()}


def test_soup_average_is_uniform_and_key_exact():
    a = _state(w=[1.0, 2.0], b=[[1.0]])
    b = _state(w=[3.0, 4.0], b=[[3.0]])
    soup = reconstruction.soup_average([a, b])
    assert np.allclose(soup["w"], [2.0, 3.0])
    assert np.allclose(soup["b"], [[2.0]])
    with pytest.raises(ValueError, match="keys differ"):
        reconstruction.soup_average([a, _state(w=[1.0, 2.0])])
    with pytest.raises(ValueError, match="shape differs"):
        reconstruction.soup_average([a, _state(w=[1.0, 2.0, 3.0], b=[[1.0]])])
    with pytest.raises(ValueError):
        reconstruction.soup_average([a])


def test_greedy_soup_improves_synthetic_case():
    # Score = distance of the averaged weight to a target of 2.0.  Rungs at
    # 1.0 / 3.2 / 0.4: best single is 1.0 (score 1.0); soup(1.0, 3.2) averages
    # to 2.1 (score 0.1) -> accepted; adding 0.4 worsens -> rejected.
    states = {
        "r1": _state(w=[1.0]),
        "r2": _state(w=[3.2]),
        "r3": _state(w=[0.4]),
    }
    scores = {"r1": 1.0, "r2": 1.2, "r3": 1.6}

    def score_state(state):
        return abs(float(state["w"][0]) - 2.0)

    result = reconstruction.greedy_soup(
        ["r1", "r2", "r3"],
        scores,
        lambda name: states[name],
        score_state,
        remaining_s=lambda: 1e9,
        margin_s=10.0,
        est_eval_s=1.0,
    )
    assert result["status"] == "improved"
    assert result["members"] == ["r1", "r2"]
    assert result["score"] == pytest.approx(0.1)
    assert np.allclose(result["state"]["w"], [2.1])
    assert result["trials"] == 2


def test_greedy_soup_no_improvement_returns_argmin():
    states = {"r1": _state(w=[2.0]), "r2": _state(w=[9.0])}
    result = reconstruction.greedy_soup(
        ["r1", "r2"],
        {"r1": 0.0, "r2": 7.0},
        lambda name: states[name],
        lambda state: abs(float(state["w"][0]) - 2.0),
        remaining_s=lambda: 1e9,
        margin_s=10.0,
        est_eval_s=1.0,
    )
    assert result["status"] == "no_improvement"
    assert result["state"] is None
    assert result["members"] == ["r1"]


def test_greedy_soup_aborts_to_argmin_on_budget():
    # remaining() below margin+estimate before the first trial: no re-score is
    # ever attempted, the accepted set stays the argmin rung.
    states = {"r1": _state(w=[1.0]), "r2": _state(w=[3.0])}
    calls = {"n": 0}

    def score_state(state):
        calls["n"] += 1
        return 0.0

    result = reconstruction.greedy_soup(
        ["r1", "r2"],
        {"r1": 1.0, "r2": 2.0},
        lambda name: states[name],
        score_state,
        remaining_s=lambda: 30.0,
        margin_s=20.0,
        est_eval_s=15.0,
    )
    assert result["status"] == "aborted_budget"
    assert result["state"] is None
    assert result["members"] == ["r1"]
    assert calls["n"] == 0


# ---------------------------------------------------------------------------
# Producer fail-closed + manifest + promotion flows
# ---------------------------------------------------------------------------


def _write_st(path, tag="x"):
    payload = tag.encode("utf-8")
    header = json.dumps(
        {
            "weight": {
                "dtype": "U8",
                "shape": [len(payload)],
                "data_offsets": [0, len(payload)],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    data = struct.pack("<Q", len(header)) + header + payload
    path.write_bytes(data)
    return data


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec(tmp_path, monkeypatch, model_type="krea2"):
    spec = ImageSpec.build(
        task_id="week9-recon",
        model="krea/Krea-2-Raw",
        model_type=model_type,
        expected_repo_name="repo",
        trigger_word=None,
        dataset_zip=None,
    )
    save_root = tmp_path / "checkpoints" / "repo"
    holdout_dir = tmp_path / "holdout"
    monkeypatch.setattr(
        type(spec), "save_root", property(lambda self: str(save_root))
    )
    monkeypatch.setattr(
        type(spec), "dataset_holdout_dir", property(lambda self: str(holdout_dir))
    )
    return spec, save_root, holdout_dir


def _deadline(seconds=3600.0):
    now = time.monotonic()
    return Deadline(hard_stop=now + seconds, export_reserve_s=180)


def _holdout_pairs(holdout_dir, count=2):
    holdout_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new("RGB", (48, 32), (index * 40, 10, 200)).save(
            holdout_dir / f"h{index}.png"
        )
        (holdout_dir / f"h{index}.txt").write_text(f"caption {index}")
    return count


def _cfg():
    return {
        "config": {
            "process": [
                {
                    "model": {
                        "name_or_path": "/cache/models/base",
                        "arch": "krea2",
                        "model_kwargs": {"text_encoder_path": "/cache/te"},
                    },
                    "network": {"type": "lora", "linear": 32, "linear_alpha": 32},
                }
            ]
        }
    }


def _seed_run(tmp_path, monkeypatch, model_type="krea2", pairs=2):
    monkeypatch.setenv("FORGE_HOLDOUT_SELECTION_TYPES", "krea2,ideogram4,flux")
    spec, save_root, holdout_dir = _spec(tmp_path, monkeypatch, model_type)
    holdout_pairs = _holdout_pairs(holdout_dir, pairs)
    scope = checkpoints.begin_run(str(save_root), "repo")
    scope = checkpoints.set_planned_steps(
        str(save_root), scope, 400, model_type=model_type
    )
    _write_st(save_root / "repo_000000200.safetensors", tag="rung200")
    _write_st(save_root / "repo.safetensors", tag="final")
    return spec, save_root, holdout_dir, scope, holdout_pairs


def _fake_rows(save_root, best="repo_000000200.safetensors", pairs=2, gens=2):
    points = pairs * gens
    names = ["repo_000000200.safetensors", "repo.safetensors"]
    rows = []
    for name in names:
        score = 0.05 if name == best else 0.09
        rows.append(
            {
                "checkpoint": name,
                "score": score,
                "captioned_score": score,
                "blank_caption_score": score,
                "captioned_points": points,
                "blank_caption_points": points,
                "points": points * 2,
                "captioned_stddev": 0.001,
                "blank_stddev": 0.001,
            }
        )
    return rows


def test_produce_writes_reconstruction_manifest(tmp_path, monkeypatch):
    spec, save_root, holdout_dir, scope, pairs = _seed_run(tmp_path, monkeypatch)

    def fake_recon(**kwargs):
        assert kwargs["model_type"] == "krea2"
        assert len(kwargs["candidates"]) == 2
        return {"rows": _fake_rows(save_root), "soup": None, "scorer": {"backend": "t"}}

    assert holdout.produce(
        spec, _cfg(), scope, _deadline(), holdout_pairs=pairs, recon_scorer=fake_recon
    )
    manifest = json.loads((save_root / "forge_holdout_scores.json").read_text())
    assert manifest["metric"] == "heldout_reconstruction_mse_v1"
    assert manifest["proxy_not_validator_metric"] is True
    assert manifest["master_seed"] == 42
    assert manifest["seeds_used"] == _EXPECTED_SEEDS_42[:2]
    assert manifest["generations"] == 2
    assert manifest["eval_params"]["steps"] == 25
    assert manifest["eval_params"]["denoise"] == 0.8
    assert manifest["captioned_weight"] == 0.25
    assert manifest["blank_caption_weight"] == 0.75
    assert len(manifest["scores"]) == 2
    by_name = {row["checkpoint"]: row for row in manifest["scores"]}
    assert by_name["repo_000000200.safetensors"]["sha256"] == _sha256(
        save_root / "repo_000000200.safetensors"
    )
    assert by_name["repo_000000200.safetensors"]["step"] == 200
    # geometry recorded for both holdout images (48x32 -> 1024x672)
    assert manifest["eval_geometry"] == [[1024, 672], [1024, 672]]


def test_produce_fail_closed_on_scorer_exception(tmp_path, monkeypatch):
    spec, save_root, holdout_dir, scope, pairs = _seed_run(tmp_path, monkeypatch)
    final_bytes = (save_root / "repo.safetensors").read_bytes()

    def broken(**kwargs):
        raise RuntimeError("scorer exploded")

    assert not holdout.produce(
        spec, _cfg(), scope, _deadline(), holdout_pairs=pairs, recon_scorer=broken
    )
    assert not (save_root / "forge_holdout_scores.json").exists()
    # finalization then ships the true final unchanged
    record = checkpoints.finalize(str(save_root), "repo", scope)
    assert record["source"] == "exact_final"
    assert (save_root / "last.safetensors").read_bytes() == final_bytes


def test_produce_fail_closed_on_incomplete_coverage(tmp_path, monkeypatch):
    spec, save_root, holdout_dir, scope, pairs = _seed_run(tmp_path, monkeypatch)

    def partial(**kwargs):
        return {"rows": _fake_rows(save_root)[:1], "soup": None, "scorer": {}}

    assert not holdout.produce(
        spec, _cfg(), scope, _deadline(), holdout_pairs=pairs, recon_scorer=partial
    )
    assert not (save_root / "forge_holdout_scores.json").exists()


def test_produce_fail_closed_on_nonfinite_score(tmp_path, monkeypatch):
    spec, save_root, holdout_dir, scope, pairs = _seed_run(tmp_path, monkeypatch)

    def nonfinite(**kwargs):
        rows = _fake_rows(save_root)
        rows[0]["score"] = float("nan")
        return {"rows": rows, "soup": None, "scorer": {}}

    assert not holdout.produce(
        spec, _cfg(), scope, _deadline(), holdout_pairs=pairs, recon_scorer=nonfinite
    )
    assert not (save_root / "forge_holdout_scores.json").exists()


def test_produce_fail_closed_on_candidate_byte_drift(tmp_path, monkeypatch):
    spec, save_root, holdout_dir, scope, pairs = _seed_run(tmp_path, monkeypatch)

    def drifting(**kwargs):
        _write_st(save_root / "repo.safetensors", tag="mutated-during-scoring")
        return {"rows": _fake_rows(save_root), "soup": None, "scorer": {}}

    assert not holdout.produce(
        spec, _cfg(), scope, _deadline(), holdout_pairs=pairs, recon_scorer=drifting
    )
    assert not (save_root / "forge_holdout_scores.json").exists()


def test_score_candidates_preflight_timeout(tmp_path, monkeypatch):
    spec, save_root, holdout_dir, scope, pairs = _seed_run(tmp_path, monkeypatch)
    candidates = [
        str(save_root / "repo_000000200.safetensors"),
        str(save_root / "repo.safetensors"),
    ]
    with pytest.raises(TimeoutError):
        reconstruction.score_candidates(
            model_type="krea2",
            candidates=candidates,
            holdout_dir=str(holdout_dir),
            cfg=_cfg(),
            temp_root=str(tmp_path / "t"),
            deadline=_deadline(seconds=120.0 + 180.0),  # < margin + min start
        )


def test_worker_timeout_kills_and_fails_closed(tmp_path, monkeypatch):
    """Real subprocess path: a hung worker is killed at the deadline guard and
    produce() leaves no manifest; finalize ships the exact final."""
    spec, save_root, holdout_dir, scope, pairs = _seed_run(tmp_path, monkeypatch)
    (tmp_path / "t").mkdir(exist_ok=True)
    stub = tmp_path / "hang.py"
    stub.write_text("import time\ntime.sleep(600)\n")
    monkeypatch.setattr(
        reconstruction,
        "_worker_cmd",
        lambda order_path: [sys.executable, str(stub), order_path],
    )
    # soft remaining = 190s (>150 pre-flight) but the poll guard trips at 30s
    # remaining; the Deadline here compresses that wait to ~4 wall seconds.
    deadline = Deadline(
        hard_stop=time.monotonic() + 34.0 + 180.0, export_reserve_s=180
    )
    monkeypatch.setattr(reconstruction, "_MIN_START_S", 2.0)
    started = time.monotonic()
    assert not holdout.produce(
        spec, _cfg(), scope, deadline, holdout_pairs=pairs
    )
    assert time.monotonic() - started < 60.0
    assert not (save_root / "forge_holdout_scores.json").exists()
    record = checkpoints.finalize(str(save_root), "repo", scope)
    assert record["source"] == "exact_final"


def test_worker_nonzero_exit_fails_closed(tmp_path, monkeypatch):
    spec, save_root, holdout_dir, scope, pairs = _seed_run(tmp_path, monkeypatch)
    stub = tmp_path / "die.py"
    stub.write_text("raise SystemExit(3)\n")
    monkeypatch.setattr(
        reconstruction,
        "_worker_cmd",
        lambda order_path: [sys.executable, str(stub), order_path],
    )
    assert not holdout.produce(
        spec, _cfg(), scope, _deadline(), holdout_pairs=pairs
    )
    assert not (save_root / "forge_holdout_scores.json").exists()


# ---------------------------------------------------------------------------
# Consumer: shadow vs promotion vs soup
# ---------------------------------------------------------------------------


def _produce_ok(tmp_path, monkeypatch, soup=None, model_type="krea2"):
    spec, save_root, holdout_dir, scope, pairs = _seed_run(
        tmp_path, monkeypatch, model_type=model_type
    )

    def fake_recon(**kwargs):
        result = {"rows": _fake_rows(save_root), "soup": None, "scorer": {}}
        if soup is not None:
            soup_path = os.path.join(kwargs["temp_root"], "soup-candidate.safetensors")
            os.makedirs(kwargs["temp_root"], exist_ok=True)
            from pathlib import Path

            _write_st(Path(soup_path), tag="soup-weights")
            result["soup"] = dict(soup, output_path=soup_path)
        return result

    assert holdout.produce(
        spec, _cfg(), scope, _deadline(), holdout_pairs=pairs, recon_scorer=fake_recon
    )
    return spec, save_root, scope


def test_shadow_mode_manifest_is_telemetry_only(tmp_path, monkeypatch):
    """Without FORGE_RECON_PROMOTION_TYPES the manifest must not change the
    shipped artifact: the exact final is promoted, argmin recorded only."""
    monkeypatch.delenv("FORGE_RECON_PROMOTION_TYPES", raising=False)
    spec, save_root, scope = _produce_ok(tmp_path, monkeypatch)
    final_bytes = (save_root / "repo.safetensors").read_bytes()
    record = checkpoints.finalize(str(save_root), "repo", scope)
    assert record["source"] == "exact_final"
    assert (save_root / "last.safetensors").read_bytes() == final_bytes


def test_promotion_env_ships_the_argmin_atomically(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_RECON_PROMOTION_TYPES", "krea2")
    spec, save_root, scope = _produce_ok(tmp_path, monkeypatch)
    rung_bytes = (save_root / "repo_000000200.safetensors").read_bytes()
    record = checkpoints.finalize(str(save_root), "repo", scope)
    assert record["source"] == "heldout_manifest"
    assert record["selected_file"] == "repo_000000200.safetensors"
    assert record["metric_is_proxy_not_validator_metric"] is True
    # atomic-copy semantics: last.safetensors byte-equals the argmin rung and
    # the record's hash matches the promoted bytes
    last = save_root / "last.safetensors"
    assert last.read_bytes() == rung_bytes
    assert record["sha256"] == _sha256(last)
    assert not (save_root / "last.safetensors.tmp").exists()


def test_promotion_env_is_type_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_RECON_PROMOTION_TYPES", "ideogram4")
    spec, save_root, scope = _produce_ok(tmp_path, monkeypatch)  # krea2 run
    record = checkpoints.finalize(str(save_root), "repo", scope)
    assert record["source"] == "exact_final"


def test_promotion_respects_advantage_gate(tmp_path, monkeypatch):
    """A best rung within 0.5% of the exact final must NOT displace it."""
    monkeypatch.setenv("FORGE_RECON_PROMOTION_TYPES", "krea2")
    spec, save_root, holdout_dir, scope, pairs = _seed_run(tmp_path, monkeypatch)

    def fake_recon(**kwargs):
        rows = _fake_rows(save_root)
        rows[0]["score"] = rows[0]["captioned_score"] = rows[0][
            "blank_caption_score"
        ] = 0.089999  # < 0.5% better than the final's 0.09
        return {"rows": rows, "soup": None, "scorer": {}}

    assert holdout.produce(
        spec, _cfg(), scope, _deadline(), holdout_pairs=pairs, recon_scorer=fake_recon
    )
    record = checkpoints.finalize(str(save_root), "repo", scope)
    assert record["source"] == "heldout_proxy_guarded_default"
    assert record["selected_file"] == "repo.safetensors"


def test_promotion_rejects_tampered_manifest_params(tmp_path, monkeypatch):
    """A manifest scored under drifted evaluator params cannot promote."""
    monkeypatch.setenv("FORGE_RECON_PROMOTION_TYPES", "krea2")
    spec, save_root, scope = _produce_ok(tmp_path, monkeypatch)
    manifest_path = save_root / "forge_holdout_scores.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["eval_params"]["denoise"] = 0.5
    manifest_path.write_text(json.dumps(manifest))
    record = checkpoints.finalize(str(save_root), "repo", scope)
    assert record["source"] == "exact_final"


def _soup_section(save_root):
    return {
        "status": "improved",
        "members": ["repo_000000200.safetensors", "repo.safetensors"],
        "score": 0.03,
        "trials": 1,
    }


def test_soup_ships_only_with_both_env_gates(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_RECON_PROMOTION_TYPES", "krea2")
    monkeypatch.setenv("FORGE_RECON_SOUP_TYPES", "krea2")
    spec, save_root, scope = _produce_ok(
        tmp_path, monkeypatch, soup=_soup_section(None)
    )
    soup_file = save_root / "repo.soup.safetensors"
    assert soup_file.exists()
    record = checkpoints.finalize(str(save_root), "repo", scope)
    assert record["source"] == "heldout_soup"
    assert (save_root / "last.safetensors").read_bytes() == soup_file.read_bytes()


def test_soup_without_soup_env_falls_back_to_argmin(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_RECON_PROMOTION_TYPES", "krea2")
    monkeypatch.delenv("FORGE_RECON_SOUP_TYPES", raising=False)
    spec, save_root, scope = _produce_ok(
        tmp_path, monkeypatch, soup=_soup_section(None)
    )
    record = checkpoints.finalize(str(save_root), "repo", scope)
    assert record["source"] == "heldout_manifest"
    assert record["selected_file"] == "repo_000000200.safetensors"


def test_soup_with_tampered_bytes_falls_back_to_argmin(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_RECON_PROMOTION_TYPES", "krea2")
    monkeypatch.setenv("FORGE_RECON_SOUP_TYPES", "krea2")
    spec, save_root, scope = _produce_ok(
        tmp_path, monkeypatch, soup=_soup_section(None)
    )
    _write_st(save_root / "repo.soup.safetensors", tag="tampered")
    record = checkpoints.finalize(str(save_root), "repo", scope)
    assert record["source"] == "heldout_manifest"
    assert record["selected_file"] == "repo_000000200.safetensors"


def test_soup_that_does_not_beat_argmin_is_rejected_by_producer(
    tmp_path, monkeypatch
):
    """validate_worker_payload refuses a soup score >= the argmin rung."""
    order = {
        "metric": reconstruction.METRIC,
        "candidates": ["/x/a.safetensors", "/x/b.safetensors"],
        "holdout_pairs": [1, 2],
        "params": reconstruction.EVAL_PARAMS["krea2"],
        "soup": {"enabled": True, "output_path": "/tmp/nope"},
    }
    rows = [
        {
            "checkpoint": "a.safetensors",
            "score": 0.05,
            "captioned_score": 0.05,
            "blank_caption_score": 0.05,
            "captioned_points": 4,
            "blank_caption_points": 4,
            "points": 8,
            "captioned_stddev": 0.0,
            "blank_stddev": 0.0,
        },
        {
            "checkpoint": "b.safetensors",
            "score": 0.06,
            "captioned_score": 0.06,
            "blank_caption_score": 0.06,
            "captioned_points": 4,
            "blank_caption_points": 4,
            "points": 8,
            "captioned_stddev": 0.0,
            "blank_stddev": 0.0,
        },
    ]
    payload = {
        "schema": 1,
        "metric": reconstruction.METRIC,
        "rows": rows,
        "soup": {
            "status": "improved",
            "members": ["a.safetensors", "b.safetensors"],
            "score": 0.05,  # NOT strictly better than argmin
            "output_path": "/tmp/nope",
        },
        "scorer": {},
    }
    with pytest.raises(ValueError, match="strictly beat the argmin"):
        reconstruction.validate_worker_payload(payload, order)


# ---------------------------------------------------------------------------
# Exact-metric hole closure (week-9)
# ---------------------------------------------------------------------------


def _exact_manifest(tmp_path, monkeypatch):
    state = checkpoints.begin_run(str(tmp_path), "repo")
    state = checkpoints.set_planned_steps(
        str(tmp_path), state, 400, model_type="krea2"
    )
    _write_st(tmp_path / "repo_000000100.safetensors", tag="best")
    _write_st(tmp_path / "repo.safetensors", tag="final")
    (tmp_path / "forge_holdout_scores.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "source": "heldout",
                "complete": True,
                "metric": "validator_exact_combined",
                "direction": "min",
                "scores": [
                    {
                        "checkpoint": "repo_000000100.safetensors",
                        "score": 0.05,
                        "sha256": _sha256(tmp_path / "repo_000000100.safetensors"),
                    },
                    {
                        "checkpoint": "repo.safetensors",
                        "score": 0.12,
                        "sha256": _sha256(tmp_path / "repo.safetensors"),
                    },
                ],
            }
        )
    )
    return state


def test_exact_metric_manifest_is_gated_by_env(tmp_path, monkeypatch):
    monkeypatch.delenv("FORGE_EXACT_HELDOUT_METRIC_TYPES", raising=False)
    state = _exact_manifest(tmp_path, monkeypatch)
    record = checkpoints.finalize(str(tmp_path), "repo", state)
    # hole closed: a stray exact-named manifest no longer promotes
    assert record["source"] == "exact_final"


def test_exact_metric_manifest_promotes_with_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_EXACT_HELDOUT_METRIC_TYPES", "krea2")
    state = _exact_manifest(tmp_path, monkeypatch)
    record = checkpoints.finalize(str(tmp_path), "repo", state)
    assert record["source"] == "heldout_manifest"
    assert record["selected_file"] == "repo_000000100.safetensors"
