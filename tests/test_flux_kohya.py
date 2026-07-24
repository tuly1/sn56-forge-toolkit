from __future__ import annotations

import io
import json
import os
import signal
import struct
import zipfile

import pytest

from forge import flux_kohya_config
from forge.data import dataset
from forge.data.schema import ImageSpec
from forge.tasks import checkpoints, dispatch, flux_kohya


def _spec(**values) -> ImageSpec:
    fields = {
        "task_id": "flux-task",
        "model": "org/standalone-flux",
        "model_type": "flux",
        "expected_repo_name": "flux-output",
        "trigger_word": "TOK",
        "dataset_zip": None,
    }
    fields.update(values)
    return ImageSpec.build(**fields)


def _png_bytes(color=(10, 20, 30)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_safetensors(path, tag: str = "test") -> bytes:
    header = json.dumps(
        {
            "__metadata__": {"tag": tag},
            "weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            },
        }
    ).encode()
    value = struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.0)
    path.write_bytes(value)
    return value


def test_dispatch_uses_kohya_only_in_legacy_flux_image(monkeypatch):
    monkeypatch.delenv("FORGE_FLUX_BACKEND", raising=False)
    assert dispatch.for_model_type("flux").__module__ == "forge.tasks.aitoolkit"

    monkeypatch.setenv("FORGE_FLUX_BACKEND", "kohya")
    assert dispatch.for_model_type("flux") is flux_kohya.run
    assert dispatch.for_model_type("krea2").__module__ == "forge.tasks.aitoolkit"


def test_standalone_model_resolution_requires_exactly_one_direct_file(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    only = model_dir / "base.safetensors"
    only.write_bytes(b"model")
    (model_dir / "README.md").write_text("metadata", encoding="utf-8")
    nested = model_dir / "nested"
    nested.mkdir()
    (nested / "ignored.safetensors").write_bytes(b"nested")

    assert flux_kohya.resolve_standalone_model_file(str(model_dir)) == str(only)

    (model_dir / "second.safetensors").write_bytes(b"second")
    with pytest.raises(RuntimeError, match="exactly one"):
        flux_kohya.resolve_standalone_model_file(str(model_dir))


def test_standalone_model_resolution_rejects_symlink_only(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"model")
    (model_dir / "linked.safetensors").symlink_to(outside)

    with pytest.raises(RuntimeError, match="found 0"):
        flux_kohya.resolve_standalone_model_file(str(model_dir))


def test_kohya_dataset_layout_and_trigger_injection(tmp_path):
    archive = tmp_path / "task.zip"
    root = tmp_path / "images"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("nested/a.png", _png_bytes())
        zf.writestr("nested/a.txt", "a monochrome illustration")
        zf.writestr("nested/b.png", _png_bytes((30, 20, 10)))
        zf.writestr("nested/b.txt", "TOK, already present")
        zf.writestr("nested/c.png", _png_bytes((30, 10, 20)))
        # Missing caption becomes an empty sidecar in the common unpacker.
        zf.writestr("nested/d.png", _png_bytes((20, 30, 10)))
        zf.writestr("nested/d.txt", "a poster mentioning TOK later")

    train_dir, pairs = dataset.prepare_kohya_flux_dataset(
        str(archive), images_root=str(root), trigger_word="TOK"
    )

    concept = root / "1_TOK"
    assert train_dir == str(root)
    assert pairs == 4
    assert sorted(path.name for path in concept.iterdir()) == [
        "a.png",
        "a.txt",
        "b.png",
        "b.txt",
        "c.png",
        "c.txt",
        "d.png",
        "d.txt",
    ]
    assert (concept / "a.txt").read_bytes() == b"TOK, a monochrome illustration"
    assert (concept / "b.txt").read_bytes() == b"TOK, already present"
    assert (concept / "c.txt").read_bytes() == b"TOK"
    assert (concept / "d.txt").read_bytes() == b"TOK, a poster mentioning TOK later"
    assert not (tmp_path / "images__forge_flux_flat").exists()


def test_caption_trigger_requires_an_exact_leading_token():
    assert dataset._caption_starts_with_trigger(b"art, ink drawing", b"art")
    assert dataset._caption_starts_with_trigger(b"  ART poster", b"art")
    assert not dataset._caption_starts_with_trigger(b"cartoon poster", b"art")
    assert not dataset._caption_starts_with_trigger(b"a poster about art", b"art")


def test_operational_config_is_fixed_offline_and_kill_safe(tmp_path):
    config = flux_kohya_config.build_config(
        base_model="/cache/models/base/model.safetensors",
        train_data_dir="/dataset/images",
        output_dir="/app/checkpoints/task/repo",
        output_name="repo",
        config_file="/dataset/configs/task.toml",
    )

    assert config["max_train_steps"] == 250
    assert config["save_every_n_steps"] == 25
    assert config["save_last_n_steps"] == 100
    assert config["guidance_scale"] == 85.0
    assert config["optimizer_type"] == "Lion"
    assert config["unet_lr"] == pytest.approx(5e-5)
    assert config["text_encoder_lr"] == pytest.approx([5e-6, 5e-6])
    assert config["network_dim"] == 128
    assert config["network_alpha"] == 64
    assert config["seed"] == 2
    assert config["max_data_loader_n_workers"] == 4
    assert config["config_file"] == "/dataset/configs/task.toml"
    assert config["mem_eff_save"] is True
    assert "noise_offset_type" not in config
    assert not any("huggingface" in key or "wandb" in key for key in config)

    path = tmp_path / "config" / "task.toml"
    flux_kohya_config.write_config(config, str(path))
    text = path.read_text(encoding="utf-8")
    assert 'pretrained_model_name_or_path = "/cache/models/base/model.safetensors"' in text
    assert 'network_args = ["train_double_block_indices=all"' in text
    assert not (path.parent / "task.toml.tmp").exists()


def test_kohya_executes_toml_from_private_publication_scope(monkeypatch):
    spec = _spec()
    monkeypatch.setattr(
        type(spec),
        "save_root",
        property(lambda self: "/app/checkpoints/flux-task/flux-output"),
    )
    config_path = "/app/checkpoints/flux-task/flux-output/config.toml"
    assert flux_kohya._config_path(spec) == config_path

    command = flux_kohya._command(
        config_path,
        script="/app/sd-scripts/flux_train_network.py",
    )
    assert command[:3] == [
        os.sys.executable,
        "-m",
        "accelerate.commands.launch",
    ]
    assert command[-3:] == [
        "/app/sd-scripts/flux_train_network.py",
        "--config_file",
        config_path,
    ]


def test_kohya_periodic_checkpoint_is_current_and_promotable(tmp_path):
    state = checkpoints.begin_run(str(tmp_path), "repo")
    state = checkpoints.set_planned_steps(str(tmp_path), state, 250, model_type="flux")
    periodic = tmp_path / "repo-step00000175.safetensors"
    periodic_bytes = _write_safetensors(periodic, "periodic")

    assert checkpoints.current_loras(str(tmp_path), state) == [str(periodic)]
    record = checkpoints.finalize(str(tmp_path), "repo", state)

    assert (tmp_path / "last.safetensors").read_bytes() == periodic_bytes
    assert record["source"] == "highest_valid_periodic"
    assert record["selected_step"] == 175


def test_kohya_exact_final_beats_periodics(tmp_path):
    state = checkpoints.begin_run(str(tmp_path), "repo")
    state = checkpoints.set_planned_steps(str(tmp_path), state, 250, model_type="flux")
    _write_safetensors(tmp_path / "repo-step00000225.safetensors", "periodic")
    final = _write_safetensors(tmp_path / "repo.safetensors", "final")

    record = checkpoints.finalize(str(tmp_path), "repo", state)

    assert (tmp_path / "last.safetensors").read_bytes() == final
    assert record["source"] == "exact_final"
    assert record["selected_step"] == 250


def test_kohya_truncated_newest_falls_back_to_valid_periodic(tmp_path):
    state = checkpoints.begin_run(str(tmp_path), "repo")
    valid = _write_safetensors(
        tmp_path / "repo-step00000175.safetensors", "valid"
    )
    (tmp_path / "repo-step00000200.safetensors").write_bytes(b"truncated")

    record = checkpoints.finalize(str(tmp_path), "repo", state)

    assert (tmp_path / "last.safetensors").read_bytes() == valid
    assert record["source"] == "highest_valid_periodic"
    assert record["selected_step"] == 175


def test_terminate_signals_the_process_group(monkeypatch):
    calls = []

    class Process:
        pid = 123

        def wait(self, timeout):
            calls.append(("wait", timeout))
            if timeout == 5:
                raise flux_kohya.subprocess.TimeoutExpired("kohya", timeout)

        def send_signal(self, sig):
            calls.append(("direct", sig))

    monkeypatch.setattr(flux_kohya.os, "getpgid", lambda pid: 456)
    monkeypatch.setattr(
        flux_kohya.os,
        "killpg",
        lambda pgid, sig: calls.append(("group", pgid, sig)),
    )

    flux_kohya._terminate(Process())

    assert calls == [
        ("group", 456, signal.SIGTERM),
        ("wait", 5),
        ("group", 456, signal.SIGKILL),
        ("wait", 10),
    ]


def test_kohya_log_parser_handles_scientific_loss(tmp_path):
    log = tmp_path / "kohya.log"
    log.write_text(
        "steps: 70%|#######| 175/250 [avr_loss=2.531e-02]\n",
        encoding="utf-8",
    )
    assert flux_kohya._parse_log(str(log)) == (pytest.approx(0.02531), 175)
