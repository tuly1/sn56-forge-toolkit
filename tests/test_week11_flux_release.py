from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import tomllib

import pytest

from forge import flux_kohya_config
from forge.data.schema import ImageSpec
from forge.tasks import flux_kohya


_GOLDEN_ROOT = Path(__file__).parent / "data" / "week11_flux_kohya"
_GOLDENS = {
    "241cda6c-seed-1.toml": {
        "sha256": "050e8f941af7d52ec265ed32603f4c4e8c3844dd095dc239af35d3a15de9c14c",
        "base_model": "/models/flux1-dev.safetensors",
        "train_data_dir": "/campaign/datasets/241cda6c/train",
        "output_dir": "/campaign/checkpoints/241cda6c/seed-1",
        "output_name": "flux-favorite-241cda6c-s1",
        "config_file": "/campaign/configs/241cda6c-seed-1.toml",
    },
    "db5fefc5-seed-1.toml": {
        "sha256": "fb2840e643a5f827fa2f961cb21cfc499b60605c3ba8235b093c3af23cb93822",
        "base_model": "/models/FLUX-DEV_MonochromeManga.safetensors",
        "train_data_dir": "/campaign/datasets/db5fefc5/train",
        "output_dir": "/campaign/checkpoints/db5fefc5/seed-1",
        "output_name": "flux-favorite-db5fefc5-s1",
        "config_file": "/campaign/configs/db5fefc5-seed-1.toml",
    },
}


class _Deadline:
    def __init__(self, remaining: float):
        self._remaining = remaining

    def remaining(self) -> float:
        return self._remaining


def _spec(**values) -> ImageSpec:
    fields = {
        "task_id": "week11-flux-task",
        "model": "org/snapshot-flux",
        "model_type": "flux",
        "expected_repo_name": "flux-output",
        "trigger_word": "TOK",
        "dataset_zip": None,
    }
    fields.update(values)
    return ImageSpec.build(**fields)


def _write_tensor(path: Path, name: str = "weight") -> bytes:
    header = json.dumps(
        {
            name: {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            }
        }
    ).encode()
    payload = struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.0)
    path.write_bytes(payload)
    return payload


def _large_header_file(
    path: Path,
    *,
    tensor_name: str = "double_blocks.0.img_attn.qkv.weight",
    malformed: bool = False,
) -> str:
    if malformed:
        path.write_bytes(struct.pack("<Q", 9) + b"{not-json")
    else:
        _write_tensor(path, tensor_name)
    os.truncate(path, flux_kohya._SNAPSHOT_MIN_CHECKPOINT_BYTES + 1)
    return str(path)


def _snapshot(tmp_path: Path, kind: str = "eligible") -> tuple[Path, str | None]:
    model = tmp_path / f"model-{kind}"
    transformer = model / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"nested")

    expected = None
    if kind == "eligible":
        expected = _large_header_file(model / "flux1-dev.safetensors")
    elif kind == "malformed":
        _large_header_file(model / "flux1-dev.safetensors", malformed=True)
    elif kind == "non_bfl":
        _large_header_file(
            model / "flux1-dev.safetensors",
            tensor_name="transformer_blocks.0.attn.to_q.weight",
        )
    elif kind == "shard":
        _large_header_file(model / "model-00001-of-00002.safetensors")
    elif kind == "multiple":
        _large_header_file(model / "flux-a.safetensors")
        _large_header_file(model / "flux-b.safetensors")
    elif kind == "eligible_plus_shard":
        _large_header_file(model / "flux1-dev.safetensors")
        _large_header_file(model / "model-00001-of-00002.safetensors")
    elif kind == "symlink":
        outside = tmp_path / "outside.safetensors"
        _large_header_file(outside)
        (model / "flux1-dev.safetensors").symlink_to(outside)
    else:  # pragma: no cover - test helper guard
        raise AssertionError(kind)
    return model, expected


def _wire_spec(monkeypatch, spec: ImageSpec, tmp_path: Path, model: Path) -> Path:
    save_root = tmp_path / "save"
    monkeypatch.setattr(
        type(spec), "cached_model_dir", property(lambda self: str(model))
    )
    monkeypatch.setattr(
        type(spec), "save_root", property(lambda self: str(save_root))
    )
    monkeypatch.setattr(
        type(spec),
        "training_folder",
        property(lambda self: str(tmp_path / "training")),
    )
    monkeypatch.setattr(
        type(spec),
        "dataset_images_dir",
        property(lambda self: str(tmp_path / "dataset-images")),
    )
    monkeypatch.setattr(
        type(spec), "cached_zip_path", property(lambda self: str(tmp_path / "task.zip"))
    )
    monkeypatch.setattr(
        type(spec),
        "config_path",
        property(lambda self: str(tmp_path / "configs" / "task.yaml")),
    )
    return save_root


def _quiet_telemetry(monkeypatch) -> None:
    monkeypatch.setattr(flux_kohya.telemetry, "set_meta", lambda **_: None)
    monkeypatch.setattr(flux_kohya.telemetry, "event", lambda *_, **__: None)


@pytest.mark.parametrize("filename", sorted(_GOLDENS))
def test_week11_snapshot_toml_matches_both_frozen_seed1_configs(
    tmp_path, filename
):
    binding = _GOLDENS[filename]
    golden_path = _GOLDEN_ROOT / filename
    golden_bytes = golden_path.read_bytes()
    assert hashlib.sha256(golden_bytes).hexdigest() == binding["sha256"]

    with golden_path.open("rb") as handle:
        frozen = tomllib.load(handle)
    emitted = flux_kohya_config.build_week11_snapshot_config(
        base_model=binding["base_model"],
        train_data_dir=binding["train_data_dir"],
        output_dir=binding["output_dir"],
        output_name=binding["output_name"],
        config_file=binding["config_file"],
    )
    assert emitted == frozen
    assert emitted["max_train_steps"] == 94
    assert emitted["seed"] == 1
    assert emitted["network_dim"] == 128
    assert emitted["network_alpha"] == 64
    assert emitted["network_args"] == [
        "train_double_block_indices=all",
        "train_single_block_indices=all",
        "train_t5xxl=True",
    ]

    rendered = tmp_path / filename
    flux_kohya_config.write_config(emitted, str(rendered))
    assert rendered.read_bytes() == golden_bytes


def test_standalone_config_default_is_unchanged_seed2_and_budgeted_surface():
    config = flux_kohya_config.build_config(
        base_model="base",
        train_data_dir="dataset",
        output_dir="output",
        output_name="repo",
        config_file="output/config.toml",
        steps=59,
    )
    assert config["seed"] == 2
    assert config["max_train_steps"] == 59
    assert config["save_every_n_steps"] == 25
    assert not hasattr(flux_kohya_config, "field_epoch_steps")


@pytest.mark.parametrize(
    "kind",
    [
        "malformed",
        "non_bfl",
        "shard",
        "multiple",
        "eligible_plus_shard",
        "symlink",
    ],
)
def test_snapshot_resolver_accepts_only_one_direct_regular_nonsharded_bfl_file(
    monkeypatch, tmp_path, kind
):
    monkeypatch.setattr(flux_kohya, "_SNAPSHOT_MIN_CHECKPOINT_BYTES", 1024)
    model, _ = _snapshot(tmp_path, kind)
    assert flux_kohya.resolve_snapshot_kohya_checkpoint(str(model)) is None


def test_snapshot_resolver_accepts_exact_eligible_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(flux_kohya, "_SNAPSHOT_MIN_CHECKPOINT_BYTES", 1024)
    model, expected = _snapshot(tmp_path)
    assert flux_kohya.resolve_flux_cache_layout(str(model)) == (
        "snapshot_directory",
        None,
    )
    assert flux_kohya.resolve_snapshot_kohya_checkpoint(str(model)) == expected


def test_eligible_snapshot_routes_to_exact_week11_kohya(monkeypatch, tmp_path):
    from forge.tasks import aitoolkit

    monkeypatch.setattr(flux_kohya, "_SNAPSHOT_MIN_CHECKPOINT_BYTES", 1024)
    model, expected = _snapshot(tmp_path)
    spec = _spec()
    _wire_spec(monkeypatch, spec, tmp_path, model)
    _quiet_telemetry(monkeypatch)
    monkeypatch.setattr(flux_kohya, "_kohya_runtime_ready", lambda: (True, "ready"))
    calls = []
    monkeypatch.setattr(
        flux_kohya,
        "_run_snapshot_kohya",
        lambda selected, deadline, checkpoint, **kwargs: calls.append(
            (selected, deadline, checkpoint, kwargs)
        ),
    )
    monkeypatch.setattr(
        aitoolkit,
        "run",
        lambda *_: pytest.fail("eligible snapshot must use the validated Kohya route"),
    )
    deadline = _Deadline(3000.0)

    flux_kohya.run(spec, deadline)

    assert calls == [
        (spec, deadline, expected, {"remaining_soft_s": 3000.0})
    ]


@pytest.mark.parametrize("failure", ["malformed", "shard", "symlink", "runtime", "budget"])
def test_ineligible_snapshot_or_runtime_budget_falls_back_to_aitoolkit(
    monkeypatch, tmp_path, failure
):
    from forge.tasks import aitoolkit

    monkeypatch.setattr(flux_kohya, "_SNAPSHOT_MIN_CHECKPOINT_BYTES", 1024)
    model, _ = _snapshot(tmp_path, failure if failure in {"malformed", "shard", "symlink"} else "eligible")
    spec = _spec()
    _wire_spec(monkeypatch, spec, tmp_path, model)
    _quiet_telemetry(monkeypatch)
    monkeypatch.setattr(
        flux_kohya,
        "_kohya_runtime_ready",
        lambda: (False, "kohya_script_missing") if failure == "runtime" else (True, "ready"),
    )
    monkeypatch.setattr(
        flux_kohya,
        "_run_snapshot_kohya",
        lambda *_args, **_kwargs: pytest.fail("ineligible route must not train"),
    )
    calls = []
    monkeypatch.setattr(aitoolkit, "run", lambda selected, deadline: calls.append((selected, deadline)))
    deadline = _Deadline(50.0 if failure == "budget" else 3000.0)

    flux_kohya.run(spec, deadline)

    assert calls == [(spec, deadline)]


@pytest.mark.parametrize("failure", ["config", "training"])
def test_snapshot_config_or_training_failure_resets_then_falls_back(
    monkeypatch, tmp_path, failure
):
    from forge.tasks import aitoolkit

    monkeypatch.setattr(flux_kohya, "_SNAPSHOT_MIN_CHECKPOINT_BYTES", 1024)
    model, _ = _snapshot(tmp_path)
    spec = _spec()
    save_root = _wire_spec(monkeypatch, spec, tmp_path, model)
    _quiet_telemetry(monkeypatch)
    monkeypatch.setattr(flux_kohya, "_kohya_runtime_ready", lambda: (True, "ready"))
    monkeypatch.setattr(
        flux_kohya.dataset,
        "prepare_kohya_flux_dataset",
        lambda *_args, **_kwargs: (str(tmp_path / "dataset-images"), 13),
    )
    if failure == "config":
        monkeypatch.setattr(
            flux_kohya.flux_kohya_config,
            "build_week11_snapshot_config",
            lambda **_: (_ for _ in ()).throw(ValueError("bad config")),
        )
    else:
        monkeypatch.setattr(
            flux_kohya,
            "_run_kohya",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("crash")),
        )
    calls = []

    def fallback(selected, deadline):
        assert not any(
            path.name.startswith(spec.expected_repo_name)
            for path in save_root.glob("*")
        )
        calls.append((selected, deadline))

    monkeypatch.setattr(aitoolkit, "run", fallback)
    deadline = _Deadline(3000.0)

    flux_kohya.run(spec, deadline)

    assert calls == [(spec, deadline)]


def test_snapshot_partial_periodic_is_quarantined_then_falls_back(
    monkeypatch, tmp_path
):
    from forge.tasks import aitoolkit

    monkeypatch.setattr(flux_kohya, "_SNAPSHOT_MIN_CHECKPOINT_BYTES", 1024)
    model, _ = _snapshot(tmp_path)
    spec = _spec()
    save_root = _wire_spec(monkeypatch, spec, tmp_path, model)
    _quiet_telemetry(monkeypatch)
    monkeypatch.setattr(flux_kohya, "_kohya_runtime_ready", lambda: (True, "ready"))
    monkeypatch.setattr(
        flux_kohya.dataset,
        "prepare_kohya_flux_dataset",
        lambda *_args, **_kwargs: (str(tmp_path / "dataset-images"), 13),
    )
    periodic_name = f"{spec.expected_repo_name}-step00000025.safetensors"

    def periodic_only(*_args, **_kwargs):
        _write_tensor(save_root / periodic_name)
        return 0, False

    monkeypatch.setattr(flux_kohya, "_run_kohya", periodic_only)
    calls = []

    def fallback(selected, deadline):
        assert not (save_root / periodic_name).exists()
        assert list(tmp_path.rglob(periodic_name)), "partial must remain recoverable"
        calls.append((selected, deadline))

    monkeypatch.setattr(aitoolkit, "run", fallback)
    deadline = _Deadline(3000.0)

    flux_kohya.run(spec, deadline)

    assert calls == [(spec, deadline)]
    assert not (save_root / "last.safetensors").exists()


@pytest.mark.parametrize("outcome", [(1, False), (0, True)])
def test_snapshot_non_natural_terminal_is_quarantined_then_falls_back(
    monkeypatch, tmp_path, outcome
):
    from forge.tasks import aitoolkit

    monkeypatch.setattr(flux_kohya, "_SNAPSHOT_MIN_CHECKPOINT_BYTES", 1024)
    model, _ = _snapshot(tmp_path)
    spec = _spec()
    save_root = _wire_spec(monkeypatch, spec, tmp_path, model)
    _quiet_telemetry(monkeypatch)
    monkeypatch.setattr(flux_kohya, "_kohya_runtime_ready", lambda: (True, "ready"))
    monkeypatch.setattr(
        flux_kohya.dataset,
        "prepare_kohya_flux_dataset",
        lambda *_args, **_kwargs: (str(tmp_path / "dataset-images"), 13),
    )
    final_name = f"{spec.expected_repo_name}.safetensors"

    def ambiguous_final(*_args, **_kwargs):
        _write_tensor(save_root / final_name)
        return outcome

    monkeypatch.setattr(flux_kohya, "_run_kohya", ambiguous_final)
    calls = []

    def fallback(selected, deadline):
        assert not (save_root / final_name).exists()
        assert list(tmp_path.rglob(final_name)), "failed final must remain recoverable"
        calls.append((selected, deadline))

    monkeypatch.setattr(aitoolkit, "run", fallback)
    deadline = _Deadline(3000.0)

    flux_kohya.run(spec, deadline)

    assert calls == [(spec, deadline)]
    assert not (save_root / "last.safetensors").exists()


def test_snapshot_accepts_only_natural_exact_step94_final(monkeypatch, tmp_path):
    from forge.tasks import aitoolkit

    monkeypatch.setattr(flux_kohya, "_SNAPSHOT_MIN_CHECKPOINT_BYTES", 1024)
    model, _ = _snapshot(tmp_path)
    spec = _spec()
    save_root = _wire_spec(monkeypatch, spec, tmp_path, model)
    _quiet_telemetry(monkeypatch)
    monkeypatch.setattr(flux_kohya, "_kohya_runtime_ready", lambda: (True, "ready"))
    monkeypatch.setattr(
        flux_kohya.dataset,
        "prepare_kohya_flux_dataset",
        lambda *_args, **_kwargs: (str(tmp_path / "dataset-images"), 13),
    )
    final_bytes = b""

    def exact_final(*_args, **_kwargs):
        nonlocal final_bytes
        final_bytes = _write_tensor(
            save_root / f"{spec.expected_repo_name}.safetensors"
        )
        return 0, False

    monkeypatch.setattr(flux_kohya, "_run_kohya", exact_final)
    monkeypatch.setattr(
        aitoolkit,
        "run",
        lambda *_: pytest.fail("natural exact final must not fall back"),
    )

    flux_kohya.run(spec, _Deadline(3000.0))

    assert (save_root / "last.safetensors").read_bytes() == final_bytes
    record = json.loads(
        (save_root / "forge_checkpoint_selection.json").read_text(encoding="utf-8")
    )
    assert record["source"] == "exact_final"
    assert record["selected_step"] == 94
