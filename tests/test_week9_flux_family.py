"""Week-9 flux field-family port: snapshot->kohya routing, depth law, parity,
and the sacred fallback (evidence/week9-flux-family-port-20260819/CHANGES.md).

Evidence anchors:
- beta Phase D (evidence/week9-gpu-campaign-20260819/beta/REPORT.md §5): the
  field kohya dim128 TE-inclusive family beat our aitoolkit family by 19.55%
  relative; its exact recipe is pinned at tests/data/week9_flux_family/
  phaseD-field.toml (sha256 174f7359..., byte copy of the beta scripts file).
- flux lane REPORT §3.3: field winners ship 540-754 presentations across
  n_train 13-27; the curve is flat 40->58 epochs at n=12.
"""

from __future__ import annotations

import json
import os
import random
import struct
import zipfile

import pytest

from forge import flux_kohya_config
from forge.data.schema import ImageSpec
from forge.tasks import flux_kohya

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "week9_flux_family"
)
_FIELD_TOML = os.path.join(_DATA_DIR, "phaseD-field.toml")
_GOLDEN = os.path.join(_DATA_DIR, "golden_week9rc_flux_aitoolkit_config.yaml")


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _spec(**values) -> ImageSpec:
    fields = {
        "task_id": "flux-task",
        "model": "org/snapshot-flux",
        "model_type": "flux",
        "expected_repo_name": "flux-output",
        "trigger_word": "TOK",
        "dataset_zip": None,
    }
    fields.update(values)
    return ImageSpec.build(**fields)


def _noise_png_bytes(seed: int) -> bytes:
    """Perceptually distinct image so the dedup pass keeps every pair."""
    import io

    from PIL import Image

    rng = random.Random(seed)
    img = Image.frombytes("RGB", (16, 16), rng.randbytes(16 * 16 * 3))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _dataset_zip(tmp_path, pairs: int = 12) -> str:
    archive = tmp_path / "task_tourn.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for index in range(pairs):
            zf.writestr(f"nested/{index:03d}.png", _noise_png_bytes(index))
            zf.writestr(f"nested/{index:03d}.txt", f"a distinct caption {index}")
    return str(archive)


def _bfl_checkpoint(path, *, size=flux_kohya._SNAPSHOT_MIN_CHECKPOINT_BYTES + 1):
    """Sparse file with a genuine safetensors header carrying BFL flux keys."""
    header = json.dumps(
        {
            "double_blocks.0.img_attn.qkv.weight": {
                "dtype": "BF16",
                "shape": [1],
                "data_offsets": [0, 2],
            }
        }
    ).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header)) + header + b"\x00\x00")
    os.truncate(path, size)
    return str(path)


def _snapshot_model_dir(tmp_path, *, checkpoint_name="flux1-dev.safetensors"):
    """A rayonlabs/FLUX.1-dev-shaped snapshot: root BFL file + diffusers tree."""
    model_dir = tmp_path / "model"
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    (model_dir / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(
        b"component weights"
    )
    checkpoint = _bfl_checkpoint(model_dir / checkpoint_name)
    return model_dir, checkpoint


def _sd_scripts_dir(tmp_path, monkeypatch):
    script_root = tmp_path / "sd-scripts"
    script_root.mkdir()
    (script_root / "flux_train_network.py").write_text("# test", encoding="utf-8")
    monkeypatch.setattr(flux_kohya, "_SD_SCRIPTS_DIR", str(script_root))
    return script_root


def _ready_runtime(tmp_path, monkeypatch):
    """Materialize every surface _kohya_runtime_ready() preflights."""
    _sd_scripts_dir(tmp_path, monkeypatch)
    assets = tmp_path / "flux-assets"
    tokenizers = assets / "tokenizers"
    tokenizers.mkdir(parents=True)
    for name in ("ae.safetensors", "clip_l.safetensors", "t5xxl_fp16.safetensors"):
        (assets / name).write_bytes(b"asset")
    monkeypatch.setattr(
        flux_kohya.flux_kohya_config, "AE_PATH", str(assets / "ae.safetensors")
    )
    monkeypatch.setattr(
        flux_kohya.flux_kohya_config,
        "CLIP_L_PATH",
        str(assets / "clip_l.safetensors"),
    )
    monkeypatch.setattr(
        flux_kohya.flux_kohya_config,
        "T5XXL_PATH",
        str(assets / "t5xxl_fp16.safetensors"),
    )
    monkeypatch.setattr(
        flux_kohya.flux_kohya_config, "TOKENIZER_CACHE_DIR", str(tokenizers)
    )


class _Deadline:
    def __init__(self, remaining: float, remaining_hard: float | None = None):
        self._remaining = remaining
        self._remaining_hard = (
            remaining + 180.0 if remaining_hard is None else remaining_hard
        )

    def remaining(self) -> float:
        return self._remaining

    def remaining_hard(self) -> float:
        return self._remaining_hard


def _clear_feature_env(monkeypatch):
    for gate in (
        "FORGE_FLUX_SNAPSHOT_BACKEND",
        "FORGE_HOLDOUT_SELECTION_TYPES",
        "FORGE_EVALGRID_SNAP_TYPES",
        "FORGE_EVAL_GEOMETRY_TYPES",
        "FORGE_TEMPLATES_DIR",
    ):
        monkeypatch.delenv(gate, raising=False)


# ---------------------------------------------------------------------------
# depth law (CHANGES.md §5)
# ---------------------------------------------------------------------------


def test_field_epoch_law_reproduces_the_beta_field_arm_depth():
    # n=12 at 58 epochs is EXACTLY the beta Phase D field arm: 87 opt steps.
    assert flux_kohya_config.field_epoch_steps(12) == 87
    # Real Aug-3 shape (n=13): 54 epochs -> 702 presentations -> 108 steps.
    assert flux_kohya_config.field_epoch_steps(13) == 108
    # Aug-10 R2 shape (n~=27): 26 epochs -> 702 presentations -> 91 steps.
    assert flux_kohya_config.field_epoch_steps(27) == 91
    # Epoch cap binds below n=12: never deeper than 58 epochs.
    assert flux_kohya_config.field_epoch_steps(1) == 29  # 58 * 1 batch / ga2
    assert flux_kohya_config.field_epoch_steps(8) == 58  # 58 * 2 batches / ga2


def test_field_epoch_law_presentations_stay_in_the_winner_band():
    # Field winners shipped 540-754 presentations (flux lane REPORT §3.3);
    # tonight's curve was flat from 480.  Over every observed tournament shape
    # the law's epoch target lands inside that band.
    for n in range(12, 28):
        epochs = min(58, -(-696 // n))  # the law's epoch choice
        presentations = epochs * n
        assert 540 <= presentations <= 754, (n, presentations)


def test_field_epoch_law_never_exceeds_the_kohya_step_ceiling():
    for n in list(range(1, 64)) + [100, 250, 500, 1000, 2000]:
        assert (
            flux_kohya_config.field_epoch_steps(n)
            <= flux_kohya_config.MAX_TRAIN_STEPS
        )


def test_field_floor_is_the_flat_band_shallow_edge():
    assert flux_kohya_config.field_floor_steps(12) == 60  # 40 ep x 12 = 480
    assert flux_kohya_config.field_floor_steps(13) == 74  # 37 ep
    assert flux_kohya_config.field_floor_steps(27) == 63  # 18 ep
    assert flux_kohya_config.field_floor_steps(1) == 20  # 40 ep cap
    for n in range(1, 40):
        assert flux_kohya_config.field_floor_steps(
            n
        ) <= flux_kohya_config.field_epoch_steps(n)


def test_field_law_degrades_to_the_anchor_depth_on_invalid_input():
    assert flux_kohya_config.field_epoch_steps("bogus") == 87
    assert flux_kohya_config.field_epoch_steps(None) == 87
    assert flux_kohya_config.field_floor_steps("bogus") == 60
    assert flux_kohya_config.field_floor_steps(None) == 60


def test_epochs_to_steps_uses_kohya_batch_arithmetic():
    # 87 max_train_steps completed exactly 58 epochs over 12 images at
    # batch 4 x ga 2 tonight (beta REPORT §5): 58 * ceil(12/4) / 2.
    assert flux_kohya_config._epochs_to_steps(58, 12) == 87
    # Accumulation carries across epochs; partial batches count as batches.
    assert flux_kohya_config._epochs_to_steps(54, 13) == 108
    assert flux_kohya_config._epochs_to_steps(1, 1) == 1


# ---------------------------------------------------------------------------
# snapshot checkpoint resolver
# ---------------------------------------------------------------------------


def test_snapshot_resolver_finds_the_root_bfl_checkpoint(tmp_path):
    model_dir, checkpoint = _snapshot_model_dir(tmp_path)
    assert (
        flux_kohya.resolve_snapshot_kohya_checkpoint(str(model_dir)) == checkpoint
    )


def test_snapshot_resolver_requires_the_evaluator_size_rule(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    small = model_dir / "flux1-dev.safetensors"
    _bfl_checkpoint(small, size=1024 * 1024)  # BFL header but far below 10GiB
    assert flux_kohya.resolve_snapshot_kohya_checkpoint(str(model_dir)) is None


def test_snapshot_resolver_rejects_non_bfl_checkpoints(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    path = model_dir / "combined-diffusers.safetensors"
    header = json.dumps(
        {
            "transformer_blocks.0.attn.to_q.weight": {
                "dtype": "BF16",
                "shape": [1],
                "data_offsets": [0, 2],
            }
        }
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x00")
    os.truncate(path, flux_kohya._SNAPSHOT_MIN_CHECKPOINT_BYTES + 1)
    assert flux_kohya.resolve_snapshot_kohya_checkpoint(str(model_dir)) is None


def test_snapshot_resolver_picks_the_largest_like_the_evaluator(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _bfl_checkpoint(
        model_dir / "smaller.safetensors",
        size=flux_kohya._SNAPSHOT_MIN_CHECKPOINT_BYTES + 1,
    )
    bigger = _bfl_checkpoint(
        model_dir / "bigger.safetensors",
        size=flux_kohya._SNAPSHOT_MIN_CHECKPOINT_BYTES + 2,
    )
    assert flux_kohya.resolve_snapshot_kohya_checkpoint(str(model_dir)) == bigger


def test_snapshot_resolver_skips_shards_symlinks_and_bad_dirs(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard = model_dir / "model-00001-of-00002.safetensors"
    _bfl_checkpoint(shard)
    assert flux_kohya.resolve_snapshot_kohya_checkpoint(str(model_dir)) is None
    shard.unlink()

    outside = _bfl_checkpoint(tmp_path / "outside.safetensors")
    (model_dir / "linked.safetensors").symlink_to(outside)
    assert flux_kohya.resolve_snapshot_kohya_checkpoint(str(model_dir)) is None

    assert (
        flux_kohya.resolve_snapshot_kohya_checkpoint(str(tmp_path / "missing"))
        is None
    )


# ---------------------------------------------------------------------------
# routing: flux -> kohya when healthy, -> aitoolkit on any failure
# ---------------------------------------------------------------------------


def test_snapshot_routes_to_kohya_when_route_is_healthy(
    monkeypatch, tmp_path
):
    from forge.tasks import aitoolkit

    _clear_feature_env(monkeypatch)
    model_dir, checkpoint = _snapshot_model_dir(tmp_path)
    spec = _spec()
    monkeypatch.setattr(
        type(spec), "cached_model_dir", property(lambda self: str(model_dir))
    )
    monkeypatch.setattr(
        flux_kohya, "_kohya_runtime_ready", lambda: (True, "ready")
    )
    calls = []
    monkeypatch.setattr(
        flux_kohya,
        "_train_with_kohya",
        lambda s, d, base_model, *, layout: calls.append((base_model, layout))
        or True,
    )
    monkeypatch.setattr(
        aitoolkit,
        "run",
        lambda *_: pytest.fail("healthy kohya route must not reach ai-toolkit"),
    )
    monkeypatch.setattr(flux_kohya.telemetry, "set_meta", lambda **_: None)
    monkeypatch.setattr(flux_kohya.telemetry, "event", lambda *a, **k: None)

    flux_kohya.run(spec, _Deadline(2340.0))
    assert calls == [(checkpoint, "snapshot_directory")]


@pytest.mark.parametrize(
    "failure",
    ["env_opt_out", "no_checkpoint", "runtime_missing", "crash", "below_floor"],
)
def test_snapshot_falls_back_to_aitoolkit_on_any_route_failure(
    monkeypatch, tmp_path, failure
):
    from forge.tasks import aitoolkit

    _clear_feature_env(monkeypatch)
    model_dir, _checkpoint = _snapshot_model_dir(tmp_path)
    spec = _spec()
    monkeypatch.setattr(
        type(spec), "cached_model_dir", property(lambda self: str(model_dir))
    )
    monkeypatch.setattr(
        flux_kohya, "_kohya_runtime_ready", lambda: (True, "ready")
    )
    monkeypatch.setattr(
        flux_kohya,
        "_train_with_kohya",
        lambda *a, **k: pytest.fail("kohya must not train in this failure mode"),
    )

    if failure == "env_opt_out":
        monkeypatch.setenv("FORGE_FLUX_SNAPSHOT_BACKEND", "aitoolkit")
    elif failure == "no_checkpoint":
        monkeypatch.setattr(
            flux_kohya, "resolve_snapshot_kohya_checkpoint", lambda _dir: None
        )
    elif failure == "runtime_missing":
        monkeypatch.setattr(
            flux_kohya,
            "_kohya_runtime_ready",
            lambda: (False, "kohya_script_missing"),
        )
    elif failure == "crash":
        def _crash(*_args, **_kwargs):
            raise RuntimeError("kohya exploded")

        monkeypatch.setattr(flux_kohya, "_train_with_kohya", _crash)
    elif failure == "below_floor":
        monkeypatch.setattr(
            flux_kohya, "_train_with_kohya", lambda *a, **k: False
        )

    deadline = _Deadline(2340.0)
    fallback_calls = []
    monkeypatch.setattr(
        aitoolkit,
        "run",
        lambda s, d: fallback_calls.append((s, d)),
    )
    events = []
    monkeypatch.setattr(flux_kohya.telemetry, "set_meta", lambda **_: None)
    monkeypatch.setattr(
        flux_kohya.telemetry,
        "event",
        lambda name, **values: events.append((name, values)),
    )

    flux_kohya.run(spec, deadline)

    # The fallback receives the SAME spec and deadline objects.
    assert fallback_calls == [(spec, deadline)]
    assert ("flux_backend_selected", {"backend": "aitoolkit",
                                      "cache_layout": "snapshot_directory"}) in events


def test_standalone_layout_still_routes_to_kohya_unconditionally(
    monkeypatch, tmp_path
):
    _clear_feature_env(monkeypatch)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    standalone = model_dir / "base.safetensors"
    standalone.write_bytes(b"model")
    spec = _spec()
    monkeypatch.setattr(
        type(spec), "cached_model_dir", property(lambda self: str(model_dir))
    )
    calls = []
    monkeypatch.setattr(
        flux_kohya,
        "_run_standalone_kohya",
        lambda s, d, model: calls.append(model),
    )
    monkeypatch.setattr(flux_kohya.telemetry, "event", lambda *a, **k: None)

    flux_kohya.run(spec, _Deadline(2340.0))
    assert calls == [str(standalone)]


# ---------------------------------------------------------------------------
# budgeting through the real _train_with_kohya
# ---------------------------------------------------------------------------


def _wire_real_train(monkeypatch, tmp_path, spec):
    monkeypatch.setattr(
        type(spec), "save_root", property(lambda self: str(tmp_path / "save"))
    )
    monkeypatch.setattr(
        type(spec),
        "training_folder",
        property(lambda self: str(tmp_path / "training")),
    )
    monkeypatch.setattr(
        type(spec),
        "dataset_images_dir",
        property(lambda self: str(tmp_path / "dataset" / "images")),
    )
    monkeypatch.setattr(
        type(spec),
        "cached_zip_path",
        property(lambda self, _z=_dataset_zip(tmp_path): _z),
    )
    monkeypatch.setattr(
        type(spec),
        "config_path",
        property(lambda self: str(tmp_path / "configs" / "flux-task.yaml")),
    )


def test_snapshot_budget_below_floor_declines_before_training(
    monkeypatch, tmp_path
):
    _clear_feature_env(monkeypatch)
    spec = _spec()
    _wire_real_train(monkeypatch, tmp_path, spec)
    monkeypatch.setattr(
        flux_kohya, "_run_kohya", lambda *a, **k: pytest.fail("must not train")
    )
    events = []
    monkeypatch.setattr(flux_kohya.telemetry, "set_meta", lambda **_: None)
    monkeypatch.setattr(
        flux_kohya.telemetry,
        "event",
        lambda name, **values: events.append((name, values)),
    )

    # 1200s soft remaining -> R11 budget 43 steps < floor 60 (n=12).
    outcome = flux_kohya._train_with_kohya(
        spec,
        _Deadline(1200.0),
        str(tmp_path / "flux1-dev.safetensors"),
        layout="snapshot_directory",
    )

    assert outcome is False
    floor_events = [v for n, v in events if n == "kohya_snapshot_budget_below_floor"]
    assert floor_events == [
        {
            "budget_steps": 43,
            "floor_steps": 60,
            "pairs": 12,
            "remaining_soft_s": 1200.0,
        }
    ]


def test_snapshot_plan_is_field_depth_capped_by_the_r11_budget(
    monkeypatch, tmp_path
):
    _clear_feature_env(monkeypatch)
    spec = _spec()
    _wire_real_train(monkeypatch, tmp_path, spec)
    monkeypatch.setattr(flux_kohya, "_run_kohya", lambda *a, **k: None)
    monkeypatch.setattr(
        flux_kohya.checkpoints,
        "finalize",
        lambda *a, **k: {
            "status": "selected_current_run",
            "source": "exact_final",
            "selected_step": 87,
        },
    )
    events = []
    monkeypatch.setattr(flux_kohya.telemetry, "set_meta", lambda **_: None)
    monkeypatch.setattr(
        flux_kohya.telemetry,
        "event",
        lambda name, **values: events.append((name, values)),
    )

    # 2340s soft remaining (a 0.75h task after export reserve and parent
    # startup): R11 budget 87 steps; field law 87 for n=12 -> plan 87, the
    # exact beta Phase D field arm depth.
    outcome = flux_kohya._train_with_kohya(
        spec,
        _Deadline(2340.0),
        str(tmp_path / "flux1-dev.safetensors"),
        layout="snapshot_directory",
    )

    assert outcome is True
    budget = next(v for n, v in events if n == "kohya_step_budgeted")
    assert budget["planned_steps"] == 87
    assert budget["budget_steps"] == 87
    assert budget["field_law_steps"] == 87
    assert budget["pairs"] == 12

    config_text = (tmp_path / "save" / "config.toml").read_text(encoding="utf-8")
    config = _parse_flat_toml(config_text)
    assert config["max_train_steps"] == 87
    assert config["network_dim"] == 128
    assert config["network_alpha"] == 64


def test_tight_budget_between_floor_and_target_ships_the_budget(
    monkeypatch, tmp_path
):
    _clear_feature_env(monkeypatch)
    spec = _spec()
    _wire_real_train(monkeypatch, tmp_path, spec)
    monkeypatch.setattr(flux_kohya, "_run_kohya", lambda *a, **k: None)
    monkeypatch.setattr(
        flux_kohya.checkpoints,
        "finalize",
        lambda *a, **k: {
            "status": "selected_current_run",
            "source": "highest_valid_periodic",
            "selected_step": 50,
        },
    )
    events = []
    monkeypatch.setattr(flux_kohya.telemetry, "set_meta", lambda **_: None)
    monkeypatch.setattr(
        flux_kohya.telemetry,
        "event",
        lambda name, **values: events.append((name, values)),
    )

    # 2000s -> R11 budget 74 steps: >= floor 60, < target 87 -> plan 74.
    outcome = flux_kohya._train_with_kohya(
        spec,
        _Deadline(2000.0),
        str(tmp_path / "flux1-dev.safetensors"),
        layout="snapshot_directory",
    )

    assert outcome is True
    budget = next(v for n, v in events if n == "kohya_step_budgeted")
    assert budget["planned_steps"] == 74
    assert budget["budget_steps"] == 74
    assert budget["field_law_steps"] == 87


# ---------------------------------------------------------------------------
# config emission parity with the beta field arm (CHANGES.md §4)
# ---------------------------------------------------------------------------


def _parse_flat_toml(text: str) -> dict:
    """The emitted kohya TOML is flat with JSON-compatible values."""
    parsed = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition(" = ")
        assert separator, f"unparseable TOML line: {line!r}"
        parsed[key.strip()] = json.loads(value)
    return parsed


# Keys that legitimately differ from the beta field arm's TOML.  Every entry
# must be justified in CHANGES.md §4 (parity table).
_PATH_KEYS = {
    "pretrained_model_name_or_path",  # resolved per-task snapshot checkpoint
    "output_dir",
    "output_name",
    "config_file",
    "train_data_dir",
}
_DOCUMENTED_DELTAS = {
    "max_train_steps",  # law-derived (87 for the Phase D shape, asserted below)
    "save_every_n_steps",  # kill-safe cadence 25 vs the arm's curve-read 15
    "seed",  # both arbitrary: field configs are scrubbed (toml header)
}
_OURS_ONLY_KEYS = {
    "save_last_n_steps",  # retention pruning; no training effect
}


def test_emitted_config_matches_the_field_arm_substantive_keys():
    with open(_FIELD_TOML, encoding="utf-8") as fh:
        field = _parse_flat_toml(fh.read())
    ours = flux_kohya_config.build_config(
        base_model="/cache/models/rayonlabs--FLUX.1-dev/flux1-dev.safetensors",
        train_data_dir="/dataset/images",
        output_dir="/app/checkpoints/task/repo",
        output_name="repo",
        config_file="/app/checkpoints/task/repo/config.toml",
        steps=flux_kohya_config.field_epoch_steps(12),
    )

    # Key sets match exactly, modulo the documented retention-only extra.
    assert set(field) - set(ours) == set()
    assert set(ours) - set(field) == _OURS_ONLY_KEYS

    mismatched = {
        key: (field[key], ours[key])
        for key in field
        if key not in _PATH_KEYS
        and key not in _DOCUMENTED_DELTAS
        and field[key] != ours[key]
    }
    assert mismatched == {}

    # The documented deltas hold their expected values (not just "different").
    assert ours["max_train_steps"] == 87 == field["max_train_steps"]
    assert ours["save_every_n_steps"] == 25 and field["save_every_n_steps"] == 15
    assert ours["seed"] == 2 and field["seed"] == 1
    assert ours["save_last_n_steps"] == 87  # keeps every rung of an 87-step run

    # Spot-assert the substantive family keys the beta A/B validated.
    assert ours["network_module"] == "networks.lora_flux"
    assert ours["network_dim"] == 128
    assert ours["network_alpha"] == 64
    assert ours["network_args"] == [
        "train_double_block_indices=all",
        "train_single_block_indices=all",
        "train_t5xxl=True",
    ]
    assert ours["optimizer_type"] == "Lion"
    assert ours["optimizer_args"] == ["weight_decay=0.005", "betas=(0.9,0.99)"]
    assert ours["unet_lr"] == pytest.approx(5e-5)
    assert ours["text_encoder_lr"] == pytest.approx([5e-6, 5e-6])
    assert ours["train_batch_size"] == 4
    assert ours["gradient_accumulation_steps"] == 2
    assert ours["caption_dropout_rate"] == 0.1
    assert ours["guidance_scale"] == 85.0
    assert ours["timestep_sampling"] == "sigmoid"
    assert ours["discrete_flow_shift"] == 3.1582
    assert ours["model_prediction_type"] == "raw"
    assert ours["resolution"] == "1024,1024"
    assert ours["loss_type"] == "l2"


# ---------------------------------------------------------------------------
# THE SACRED FALLBACK: kohya crash -> byte-identical week9-rc aitoolkit config
# ---------------------------------------------------------------------------


def test_kohya_crash_falls_back_to_byte_identical_week9rc_aitoolkit_config(
    monkeypatch, tmp_path
):
    """End to end: the kohya route is attempted, crashes mid-launch, and the
    fallback emits EXACTLY the config bytes week9-rc's flux path emits for the
    same task shape (golden generated at c70611f — see regen_golden.py for
    provenance; FORGE-GOLDEN-ROOT marks the run-specific filesystem root)."""
    from forge.tasks import aitoolkit

    _clear_feature_env(monkeypatch)
    spec = _spec()
    model_dir, _checkpoint = _snapshot_model_dir(tmp_path)
    monkeypatch.setattr(
        type(spec), "cached_model_dir", property(lambda self: str(model_dir))
    )
    _wire_real_train(monkeypatch, tmp_path, spec)
    monkeypatch.setattr(
        type(spec),
        "dataset_holdout_dir",
        property(lambda self: str(tmp_path / "dataset" / "holdout")),
    )
    _ready_runtime(tmp_path, monkeypatch)

    # Crash the kohya child AT LAUNCH, after the real resolver, preflight,
    # dataset preparation, budgeting, and config write all ran.
    monkeypatch.setattr(flux_kohya, "_start_gpu_sampler", lambda *a: None)

    def _no_gpu(*_args, **_kwargs):
        raise OSError("CUDA initialization failed")

    monkeypatch.setattr(flux_kohya.subprocess, "Popen", _no_gpu)

    # The fallback trains for real in production; here we stop at the config
    # write, which is the byte-identity subject.
    monkeypatch.setattr(aitoolkit, "_run_toolkit", lambda *a, **k: False)
    monkeypatch.setattr(aitoolkit, "_finalize", lambda *a, **k: None)
    monkeypatch.setattr(aitoolkit.telemetry, "collect_env", lambda: None)

    deadline = _Deadline(2340.0, remaining_hard=2520.0)
    flux_kohya.run(spec, deadline)

    with open(_GOLDEN, "rb") as fh:
        golden = fh.read()
    expected = (
        golden.replace(b"FORGE-GOLDEN-ROOT/checkpoints/flux-task", str(tmp_path / "training").encode())
        .replace(b"FORGE-GOLDEN-ROOT/dataset/images", str(tmp_path / "dataset" / "images").encode())
        .replace(b"FORGE-GOLDEN-ROOT/cache/model", str(model_dir).encode())
    )
    with open(spec.config_path, "rb") as fh:
        produced = fh.read()
    assert produced == expected
