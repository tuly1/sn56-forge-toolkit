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


def test_dispatch_uses_shape_aware_flux_handler_only_in_legacy_image(monkeypatch):
    monkeypatch.delenv("FORGE_FLUX_BACKEND", raising=False)
    assert dispatch.for_model_type("flux").__module__ == "forge.tasks.aitoolkit"

    monkeypatch.setenv("FORGE_FLUX_BACKEND", "kohya")
    assert dispatch.for_model_type("flux") is flux_kohya.run
    assert dispatch.for_model_type("krea2").__module__ == "forge.tasks.aitoolkit"


def test_standalone_model_resolution_matches_normalized_downloader_shape(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    only = model_dir / "base.safetensors"
    only.write_bytes(b"model")
    # huggingface_hub can retain nested local-dir metadata; G.O.D's normalizer
    # removes other root files, and its legacy resolver counts root files only.
    nested = model_dir / ".cache" / "huggingface"
    nested.mkdir(parents=True)
    (nested / "download.json").write_text("{}", encoding="utf-8")

    assert flux_kohya.resolve_standalone_model_file(str(model_dir)) == str(only)

    (model_dir / "README.md").write_text("metadata", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact-one direct"):
        flux_kohya.resolve_standalone_model_file(str(model_dir))


def test_flux_cache_layout_routes_diffusers_snapshot_to_aitoolkit(tmp_path):
    model_dir = tmp_path / "model"
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    weight = model_dir / "transformer" / "diffusion_pytorch_model.safetensors"
    weight.write_bytes(b"model")

    assert flux_kohya.resolve_flux_cache_layout(str(model_dir)) == (
        "snapshot_directory",
        None,
    )


def test_one_root_checkpoint_ignores_arbitrary_leftover_asset_directory(tmp_path):
    model_dir = tmp_path / "model"
    (model_dir / "examples").mkdir(parents=True)
    root_checkpoint = model_dir / "root.safetensors"
    root_checkpoint.write_bytes(b"root model")
    (model_dir / "examples" / "README.md").write_text(
        "{}", encoding="utf-8"
    )
    (model_dir / "examples" / "nested.safetensors").write_bytes(
        b"unused nested model"
    )

    assert flux_kohya.resolve_flux_cache_layout(str(model_dir)) == (
        "standalone_checkpoint",
        str(root_checkpoint),
    )


def test_one_root_checkpoint_with_diffusers_component_routes_snapshot(tmp_path):
    model_dir = tmp_path / "model"
    transformer = model_dir / "transformer"
    transformer.mkdir(parents=True)
    (model_dir / "root.safetensors").write_bytes(b"root model")
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    (transformer / "nested.safetensors").write_bytes(b"component model")

    assert flux_kohya.resolve_flux_cache_layout(str(model_dir)) == (
        "snapshot_directory",
        None,
    )


def test_flux_cache_layout_routes_sharded_snapshot_to_aitoolkit(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors.index.json").write_text(
        '{"weight_map":{"transformer.weight":"model-00001-of-00001.safetensors"}}',
        encoding="utf-8",
    )
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"model")

    assert flux_kohya.resolve_flux_cache_layout(str(model_dir)) == (
        "snapshot_directory",
        None,
    )


def test_single_shard_named_weight_routes_snapshot_per_downloader_contract(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"model")

    assert flux_kohya.resolve_flux_cache_layout(str(model_dir)) == (
        "snapshot_directory",
        None,
    )


def test_nested_weight_index_routes_snapshot(tmp_path):
    model_dir = tmp_path / "model"
    assets = model_dir / "assets"
    assets.mkdir(parents=True)
    (model_dir / "root.safetensors").write_bytes(b"root model")
    (assets / "model.safetensors.index.json").write_text(
        '{"weight_map":{"weight":"root.safetensors"}}', encoding="utf-8"
    )

    assert flux_kohya.resolve_flux_cache_layout(str(model_dir)) == (
        "snapshot_directory",
        None,
    )


@pytest.mark.parametrize("cache_kind", ["empty", "weight_free", "empty_weight"])
def test_flux_cache_layout_rejects_incomplete_cache(tmp_path, cache_kind):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    if cache_kind == "weight_free":
        (model_dir / "README.md").write_text("metadata", encoding="utf-8")
    elif cache_kind == "empty_weight":
        (model_dir / "model.safetensors").touch()

    with pytest.raises(RuntimeError):
        flux_kohya.resolve_flux_cache_layout(str(model_dir))


def test_standalone_layout_ignores_unreferenced_nested_symlink(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    only = model_dir / "base.safetensors"
    only.write_bytes(b"model")
    nested = model_dir / ".cache"
    nested.mkdir()
    (nested / "outside").symlink_to(tmp_path / "outside")

    assert flux_kohya.resolve_flux_cache_layout(str(model_dir)) == (
        "standalone_checkpoint",
        str(only),
    )


def test_snapshot_layout_rejects_nested_symlink(tmp_path):
    model_dir = tmp_path / "model"
    transformer = model_dir / "transformer"
    transformer.mkdir(parents=True)
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    (transformer / "weights.safetensors").write_bytes(b"model")
    (transformer / "outside").symlink_to(tmp_path / "outside")

    with pytest.raises(RuntimeError, match="symlink"):
        flux_kohya.resolve_flux_cache_layout(str(model_dir))


def test_legacy_flux_handler_routes_cache_shape_without_guessing(monkeypatch, tmp_path):
    from forge.tasks import aitoolkit

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    spec = _spec()
    monkeypatch.setattr(
        type(spec), "cached_model_dir", property(lambda self: str(model_dir))
    )
    calls = []
    monkeypatch.setattr(
        flux_kohya,
        "_run_standalone_kohya",
        lambda selected_spec, deadline, model: calls.append(("kohya", model)),
    )
    monkeypatch.setattr(
        aitoolkit,
        "run",
        lambda selected_spec, deadline: calls.append(("aitoolkit", None)),
    )
    monkeypatch.setattr(flux_kohya.telemetry, "set_meta", lambda **values: None)
    monkeypatch.setattr(flux_kohya.telemetry, "event", lambda *args, **values: None)

    standalone = model_dir / "base.safetensors"
    standalone.write_bytes(b"model")
    flux_kohya.run(spec, object())
    assert calls == [("kohya", str(standalone))]

    calls.clear()
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    flux_kohya.run(spec, object())
    assert calls == [("aitoolkit", None)]


def test_kohya_subprocess_uses_only_its_pinned_dependency_paths(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/opt/sn56/ai-toolkit-python")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setenv(
        "FORGE_KOHYA_PYTHONPATH",
        "/home/.local/lib/python3.10/site-packages",
    )
    monkeypatch.setenv(
        "FORGE_KOHYA_LD_LIBRARY_PATH",
        "/usr/local/cuda/lib:/usr/local/cuda/lib64",
    )
    monkeypatch.setenv("FORGE_KOHYA_LD_PRELOAD", "libtcmalloc.so")
    monkeypatch.setenv("FORGE_KOHYA_PROTOBUF_IMPLEMENTATION", "python")
    monkeypatch.setenv("FORGE_KOHYA_PATH", "/kohya/bin:/usr/bin")

    env = flux_kohya._kohya_subprocess_env()

    assert env["PYTHONPATH"] == "/home/.local/lib/python3.10/site-packages"
    assert env["LD_LIBRARY_PATH"] == "/usr/local/cuda/lib:/usr/local/cuda/lib64"
    assert env["LD_PRELOAD"] == "libtcmalloc.so"
    assert env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] == "python"
    assert env["PATH"] == "/kohya/bin:/usr/bin"


def test_standalone_path_does_not_attest_the_parent_dependency_graph(
    monkeypatch, tmp_path
):
    spec = _spec()
    monkeypatch.setattr(
        type(spec), "save_root", property(lambda self: str(tmp_path / "output"))
    )
    monkeypatch.setattr(
        type(spec),
        "training_folder",
        property(lambda self: str(tmp_path / "training")),
    )
    monkeypatch.setattr(
        flux_kohya.checkpoints, "ensure_run", lambda *args: {"scope": "test"}
    )
    monkeypatch.setattr(
        flux_kohya.dataset,
        "prepare_kohya_flux_dataset",
        lambda *args, **kwargs: (str(tmp_path / "training"), 1),
    )
    planned = []

    def set_planned_steps(_root, _scope, steps, **kwargs):
        planned.append((steps, kwargs))
        return {"scope": "test"}

    monkeypatch.setattr(
        flux_kohya.checkpoints, "set_planned_steps", set_planned_steps
    )
    configs = []

    def build_config(**kwargs):
        configs.append(kwargs)
        return {"save_every_n_steps": 25}

    monkeypatch.setattr(
        flux_kohya.flux_kohya_config,
        "build_config",
        build_config,
    )
    monkeypatch.setattr(
        flux_kohya.flux_kohya_config, "write_config", lambda *args: None
    )
    monkeypatch.setattr(flux_kohya, "_run_kohya", lambda *args: None)
    monkeypatch.setattr(
        flux_kohya.checkpoints,
        "finalize",
        lambda *args, **kwargs: {
            "status": "selected_current_run",
            "source": "exact_final",
            "selected_step": 59,
        },
    )
    events = []
    monkeypatch.setattr(flux_kohya.telemetry, "set_meta", lambda **kwargs: None)
    monkeypatch.setattr(
        flux_kohya.telemetry,
        "event",
        lambda name, **kwargs: events.append((name, kwargs)),
    )
    monkeypatch.setattr(
        flux_kohya.telemetry,
        "collect_env",
        lambda: pytest.fail("parent ai-toolkit graph must not describe Kohya"),
    )
    monkeypatch.setattr(
        flux_kohya.telemetry,
        "note_peak_memory",
        lambda: pytest.fail("parent Torch allocator did not train Kohya"),
    )

    class Deadline:
        def remaining(self):
            # 0.5h hard budget minus Forge's 180s export reserve.
            return 1620.0

    flux_kohya._run_standalone_kohya(
        spec,
        Deadline(),
        str(tmp_path / "base.safetensors"),
    )

    assert planned == [(59, {"model_type": "flux"})]
    assert configs[0]["steps"] == 59
    budget = next(values for name, values in events if name == "kohya_step_budgeted")
    assert budget["planned_steps"] == 59
    assert budget["remaining_soft_s"] == 1620.0


def test_standalone_model_resolution_rejects_symlink_only(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"model")
    (model_dir / "linked.safetensors").symlink_to(outside)

    with pytest.raises(RuntimeError, match="symlink"):
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


def test_kohya_missing_caption_without_trigger_is_nonempty_but_blank(tmp_path):
    archive = tmp_path / "task.zip"
    root = tmp_path / "images"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("nested/a.png", _png_bytes())

    train_dir, pairs = dataset.prepare_kohya_flux_dataset(
        str(archive), images_root=str(root), trigger_word=None
    )

    caption = root / "1_style" / "a.txt"
    assert train_dir == str(root)
    assert pairs == 1
    assert caption.read_bytes() == b"\n"
    assert caption.read_text(encoding="utf-8").strip() == ""
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
    assert config["save_every_n_steps"] == 51
    assert config["save_last_n_steps"] == 100
    assert config["guidance_scale"] == 85.0
    assert config["optimizer_type"] == "Lion"
    assert config["unet_lr"] == pytest.approx(5e-5)
    assert config["text_encoder_lr"] == pytest.approx([5e-6, 5e-6])
    assert config["network_dim"] == 128
    assert config["network_alpha"] == 64
    assert config["seed"] == 2
    assert config["max_data_loader_n_workers"] == 4
    assert config["tokenizer_cache_dir"] == "/app/flux/tokenizers"
    assert config["config_file"] == "/dataset/configs/task.toml"
    assert config["mem_eff_save"] is True
    assert "noise_offset_type" not in config
    assert not any("huggingface" in key or "wandb" in key for key in config)

    path = tmp_path / "config" / "task.toml"
    flux_kohya_config.write_config(config, str(path))
    text = path.read_text(encoding="utf-8")
    assert 'pretrained_model_name_or_path = "/cache/models/base/model.safetensors"' in text
    assert 'network_args = ["train_double_block_indices=all"' in text
    assert 'tokenizer_cache_dir = "/app/flux/tokenizers"' in text
    assert not (path.parent / "task.toml.tmp").exists()


@pytest.mark.parametrize(
    ("remaining_soft_s", "expected"),
    [
        (1620.0, 59),
        (3420.0, 128),
        (7200.0, 250),
        (45.0, 1),
        (44.0, 1),
        (float("nan"), 1),
        (float("inf"), 1),
        (None, 1),
        ("invalid", 1),
    ],
)
def test_kohya_step_budget_uses_only_durable_r11_throughput(
    remaining_soft_s, expected
):
    assert flux_kohya_config.budgeted_train_steps(
        remaining_soft_s,
        boundary_margin_s=45.0,
    ) == expected


@pytest.mark.parametrize("boundary", [float("nan"), float("inf"), "invalid"])
def test_kohya_step_budget_rejects_invalid_boundary(boundary):
    assert flux_kohya_config.budgeted_train_steps(
        1620.0,
        boundary_margin_s=boundary,
    ) == 1


@pytest.mark.parametrize(
    ("steps", "save_every"),
    [(25, 12), (59, 25), (128, 26), (250, 51)],
)
def test_deadline_capped_config_uses_fixed_candidate_save_cadence(
    steps, save_every
):
    config = flux_kohya_config.build_config(
        base_model="/cache/models/base/model.safetensors",
        train_data_dir="/dataset/images",
        output_dir="/app/checkpoints/task/repo",
        output_name="repo",
        config_file="/dataset/configs/task.toml",
        steps=steps,
    )

    assert config["max_train_steps"] == steps
    assert config["save_every_n_steps"] == save_every
    assert save_every < steps
    assert steps % save_every != 0


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
    assert "current run's exact final" in record["reason"]
    assert "ai-toolkit" not in record["reason"]


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


def test_kohya_log_parser_keeps_loss_but_does_not_infer_progress(tmp_path):
    log = tmp_path / "kohya.log"
    log.write_text(
        "steps: 70%|#######| 175/250 [avr_loss=2.531e-02]\n"
        "unrelated cache pass: 51/64\n",
        encoding="utf-8",
    )
    assert flux_kohya._parse_loss(str(log)) == pytest.approx(0.02531)
    assert not hasattr(flux_kohya, "_parse_log")


def test_kohya_metrics_never_publish_a_guessed_step(monkeypatch, tmp_path):
    script_root = tmp_path / "sd-scripts"
    script_root.mkdir()
    (script_root / "flux_train_network.py").write_text("# test", encoding="utf-8")
    log_path = tmp_path / "kohya.log"
    events = []

    class Process:
        returncode = 0

        def poll(self):
            return 0

    class Deadline:
        def remaining(self):
            return 3600

    def popen(command, **kwargs):
        kwargs["stdout"].write(
            "steps: 70%|#######| 175/250 [avr_loss=2.531e-02]\n"
            "unrelated cache pass: 51/64\n"
        )
        kwargs["stdout"].flush()
        return Process()

    monkeypatch.setattr(flux_kohya, "_SD_SCRIPTS_DIR", str(script_root))
    monkeypatch.setattr(flux_kohya, "_log_path", lambda spec: str(log_path))
    monkeypatch.setattr(flux_kohya, "_start_gpu_sampler", lambda *args: None)
    monkeypatch.setattr(flux_kohya.subprocess, "Popen", popen)
    monkeypatch.setattr(
        flux_kohya.telemetry,
        "event",
        lambda name, **values: events.append((name, values)),
    )
    monkeypatch.setattr(flux_kohya.telemetry, "sample", lambda *args: None)
    monkeypatch.setattr(
        flux_kohya.telemetry,
        "train_point",
        lambda *args: pytest.fail("an unbound loss must not become a train point"),
    )

    flux_kohya._run_kohya("/tmp/config.toml", Deadline(), _spec(), {})

    assert [event for event in events if event[0] == "kohya_metrics"] == [
        ("kohya_metrics", {"loss": pytest.approx(0.02531)})
    ]
