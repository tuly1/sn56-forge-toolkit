"""ML-free unit tests for sn56-forge-toolkit (no torch / ai-toolkit import).

Covers the two highest-risk correctness points — config.name==repo and the
finalize-to-last.safetensors guarantee — plus the dataset byte-exact caption
invariant, the size-scaling recipe, and the never-crash CLI funnel.
"""

from __future__ import annotations

import io
import os
import zipfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Templates now ship inside the package (forge/templates/), not at repo root.
TEMPLATES_DIR = os.path.join(REPO_ROOT, "forge", "templates")

from forge import config, recipe  # noqa: E402
from forge.data import dataset  # noqa: E402
from forge.data.schema import ImageSpec  # noqa: E402
from forge.tasks import aitoolkit, checkpoints, dispatch, fallback  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _spec(**kw):
    base = dict(
        task_id="t1",
        model="stabilityai/x",
        model_type="flux",
        expected_repo_name="myrepo",
        trigger_word="tok",
        dataset_zip=None,
    )
    base.update(kw)
    return ImageSpec.build(**base)


def _png_bytes(color):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="PNG")
    return buf.getvalue()


def _make_zip(path, pairs, nested=True, captions=None, missing_caption_for=None):
    """pairs: list of (basename, color). captions: dict basename->str."""
    captions = captions or {}
    prefix = "dataset_top/" if nested else ""
    with zipfile.ZipFile(path, "w") as zf:
        for i, (name, color) in enumerate(pairs):
            zf.writestr(f"{prefix}{name}.png", _png_bytes(color))
            if missing_caption_for and name in missing_caption_for:
                continue
            cap = captions.get(name, f"caption {i}")
            zf.writestr(f"{prefix}{name}.txt", cap)


# --------------------------------------------------------------------------- #
# 1. schema paths
# --------------------------------------------------------------------------- #
def test_schema_paths():
    s = _spec()
    assert s.cached_model_dir == "/cache/models/stabilityai--x"
    assert s.cached_zip_path == "/cache/datasets/t1_tourn.zip"
    assert s.dataset_images_dir == "/dataset/images"
    assert s.training_folder == "/app/checkpoints/t1"
    assert s.save_root == "/app/checkpoints/t1/myrepo"
    assert s.output_dir == s.save_root
    assert s.config_path == "/dataset/configs/t1.yaml"


def test_toolkit_log_path_is_run_unique_and_not_under_upload_root(tmp_path, monkeypatch):
    s = _spec(expected_repo_name="repo")
    config_path = tmp_path / "configs" / "task.yaml"
    save_root = tmp_path / "checkpoints" / "repo"
    save_root.mkdir(parents=True)
    monkeypatch.setattr(type(s), "config_path", property(lambda self: str(config_path)))
    monkeypatch.setattr(type(s), "save_root", property(lambda self: str(save_root)))
    first = aitoolkit._toolkit_log_path(s)
    second = aitoolkit._toolkit_log_path(s)

    assert first != second
    assert str(save_root) not in first
    assert str(config_path.parent / "forge-logs") in first
    assert "repo" in os.path.basename(first)


def test_toolkit_log_parser_preserves_exponent_and_terminal_step(tmp_path):
    log = tmp_path / "toolkit.log"
    log.write_text(
        "krea: 97%|#########7| 35/36 [loss: 2.531e-02]\n"
        "Saved checkpoint to /outputs/krea.safetensors\n",
        encoding="utf-8",
    )

    loss, step = aitoolkit._parse_toolkit_log(str(log))

    assert loss == pytest.approx(0.02531)
    assert step == 36


def test_schema_model_type_normalized():
    s = _spec(model_type="  FLUX ")
    assert s.model_type == "flux"


# --------------------------------------------------------------------------- #
# 2. dataset flat unpack
# --------------------------------------------------------------------------- #
def test_dataset_flat_unpack(tmp_path):
    zip_path = tmp_path / "t1_tourn.zip"
    imgs = tmp_path / "images"
    colors = [(i * 20, 40, 200 - i * 15) for i in range(4)]
    _make_zip(zip_path, [(f"img{i}", colors[i]) for i in range(4)], nested=True)

    out_dir, n = dataset.prepare_aitoolkit_dataset(
        str(zip_path), images_dir=str(imgs), trigger_word="tok"
    )
    assert out_dir == str(imgs)
    assert n == 4
    files = sorted(os.listdir(imgs))
    assert len([f for f in files if f.endswith(".png")]) == 4
    assert len([f for f in files if f.endswith(".txt")]) == 4
    # no {repeats}_concept subdir, no subdirs at all
    assert all(os.path.isfile(os.path.join(imgs, f)) for f in files)
    # extract temp cleaned up
    assert not os.path.isdir(str(imgs).rstrip("/") + "__extract")
    assert not os.path.isdir(str(imgs).rstrip("/") + "__flat")


def test_dataset_multi_subdir_all_kept(tmp_path):
    # Images split across sibling/per-concept folders must ALL be collected
    # (mirrors the validator's rglob staging), not truncated to one subdir.
    zip_path = tmp_path / "t1_tourn.zip"
    imgs = tmp_path / "images"
    with zipfile.ZipFile(zip_path, "w") as zf:
        # same basename in two folders → collision-rename must keep both
        zf.writestr("concept_a/img.png", _png_bytes((200, 10, 10)))
        zf.writestr("concept_a/img.txt", "a caption")
        zf.writestr("concept_b/img.png", _png_bytes((10, 10, 200)))
        zf.writestr("concept_b/img.txt", "b caption")
        zf.writestr("concept_b/extra.png", _png_bytes((10, 200, 10)))
        zf.writestr("concept_b/extra.txt", "extra caption")
    _out, n = dataset.prepare_aitoolkit_dataset(str(zip_path), images_dir=str(imgs))
    assert n == 3  # nothing dropped
    pngs = sorted(f for f in os.listdir(imgs) if f.endswith(".png"))
    assert len(pngs) == 3
    # captions stay paired to their (possibly renamed) image stem, byte-exact
    caps = {
        open(os.path.join(imgs, f), "rb").read()
        for f in os.listdir(imgs)
        if f.endswith(".txt")
    }
    assert {b"a caption", b"b caption", b"extra caption"} == caps


# --------------------------------------------------------------------------- #
# 3. dedup called
# --------------------------------------------------------------------------- #
def test_dedup_invoked(tmp_path, monkeypatch):
    zip_path = tmp_path / "t1_tourn.zip"
    imgs = tmp_path / "images"
    _make_zip(zip_path, [(f"img{i}", (i, i, i)) for i in range(3)], nested=True)

    called = {}
    import forge.data.dedup as dedup_mod

    def _fake(image_dir):
        called["dir"] = image_dir
        return 0

    monkeypatch.setattr(dedup_mod, "dedup_dataset", _fake)
    dataset.prepare_aitoolkit_dataset(str(zip_path), images_dir=str(imgs))
    assert "dir" in called
    # dedup runs on the descended source dir, not the flat target
    assert called["dir"] != str(imgs)


def test_dedup_real_reduces(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("numpy")
    zip_path = tmp_path / "t1_tourn.zip"
    imgs = tmp_path / "images"
    # 20 identical images → dedup should thin toward the keep floor
    _make_zip(zip_path, [(f"img{i:02d}", (10, 10, 10)) for i in range(20)], nested=True)
    _out, n = dataset.prepare_aitoolkit_dataset(str(zip_path), images_dir=str(imgs))
    assert n < 20


# --------------------------------------------------------------------------- #
# 4/5/6. captions byte-exact / json preserved / missing -> empty
# --------------------------------------------------------------------------- #
def test_caption_byte_exact_no_trigger(tmp_path):
    zip_path = tmp_path / "t1_tourn.zip"
    imgs = tmp_path / "images"
    cap = "a red car, sunset"
    _make_zip(
        zip_path, [("img0", (200, 0, 0))], nested=True, captions={"img0": cap}
    )
    dataset.prepare_aitoolkit_dataset(
        str(zip_path), images_dir=str(imgs), trigger_word="MYTRIGGER"
    )
    data = open(os.path.join(imgs, "img0.txt"), "rb").read()
    assert data == cap.encode("utf-8")
    assert b"MYTRIGGER" not in data


def test_caption_json_preserved(tmp_path):
    zip_path = tmp_path / "t1_tourn.zip"
    imgs = tmp_path / "images"
    js = '{"prompt":"a cat","style":"photo"}'
    _make_zip(zip_path, [("img0", (5, 5, 5))], nested=True, captions={"img0": js})
    dataset.prepare_aitoolkit_dataset(str(zip_path), images_dir=str(imgs))
    data = open(os.path.join(imgs, "img0.txt"), "rb").read()
    assert data == js.encode("utf-8")


def test_missing_caption_empty_file(tmp_path):
    zip_path = tmp_path / "t1_tourn.zip"
    imgs = tmp_path / "images"
    _make_zip(
        zip_path,
        [("img0", (9, 9, 9))],
        nested=True,
        missing_caption_for={"img0"},
    )
    _out, n = dataset.prepare_aitoolkit_dataset(str(zip_path), images_dir=str(imgs))
    assert n == 1
    dst = os.path.join(imgs, "img0.txt")
    assert os.path.isfile(dst)
    assert os.path.getsize(dst) == 0


# --------------------------------------------------------------------------- #
# 7/8. recipe
# --------------------------------------------------------------------------- #
def test_recipe_step_scaling():
    # no budget pressure (large hours)
    assert recipe.size_scaled_steps("flux", 24, 1000, 2000) == 1100
    s10 = recipe.size_scaled_steps("flux", 10, 1000, 2000)
    s50 = recipe.size_scaled_steps("flux", 50, 1000, 2000)
    assert s10 < s50
    # clamps (krea2 re-calibrated Jul 16: near-flat score curve -> shallow band)
    assert recipe.size_scaled_steps("krea2", 1, 1000, 2000) == 100  # min floor
    assert recipe.size_scaled_steps("krea2", 500, 1000, 2000) == 400  # max
    # z-image now has its own law (champion base 1100 @ n_ref)
    assert recipe.size_scaled_steps("z-image", 24, 1000, 2000) == 1100
    # unknown type -> template
    assert recipe.size_scaled_steps("sd3", 24, 1000, 2000) == 2000
    # budget cap drives well below scaled, never < 1
    capped = recipe.size_scaled_steps("krea2", 24, 0.01, 2000)
    assert capped >= 1
    assert capped < 1500


def test_recipe_budget_cap_example():
    # krea2 @ 24 imgs now scales to its calibrated base (300) well under the
    # 1h budget cap (~737), so the SIZE law binds, not the clock
    v = recipe.size_scaled_steps("krea2", 24, 1.0, 2000)
    assert v == 300
    # the clock still binds on a tight budget
    v = recipe.size_scaled_steps("krea2", 24, 0.2, 2000)
    assert v < 300


def test_recipe_save_every():
    # Fixed four-candidate budget, including runs longer than template cadence.
    assert recipe.kill_safe_save_every(2000, 250) == 401
    assert recipe.kill_safe_save_every(700, 250) == 141
    # Actual Jul-20 tournament shapes: three/four periodic saves, respectively.
    assert recipe.kill_safe_save_every(86, 250) == 25
    assert (86 - 1) // recipe.kill_safe_save_every(86, 250) == 3
    assert recipe.kill_safe_save_every(367, 200) == 74
    assert (367 - 1) // recipe.kill_safe_save_every(367, 200) == 4
    assert recipe.kill_safe_save_every(24, 250) == 12  # one mid-run recovery point
    assert recipe.kill_safe_save_every(2, 250) == 1


# --------------------------------------------------------------------------- #
# 9. config build per type
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _templates_env(monkeypatch):
    monkeypatch.setattr(config, "_TEMPLATES_DIR", TEMPLATES_DIR)


def test_config_flux():
    s = _spec(model_type="flux", expected_repo_name="repoA", trigger_word="tok")
    cfg = config.build_config(s, num_images=24, hours_to_complete=1000)
    assert cfg["config"]["name"] == "repoA"
    p = cfg["config"]["process"][0]
    assert p["training_folder"] == "/app/checkpoints/t1"
    assert p["datasets"][0]["folder_path"] == "/dataset/images"
    assert p["model"]["name_or_path"] == "/cache/models/stabilityai--x"
    assert p["train"]["steps"] == 1100
    assert p["trigger_word"] == "tok"
    # arch preserved
    assert p["model"]["is_flux"] is True
    # Four periodic candidates plus final: floor(1100/5) + 1 = 221.
    assert p["save"]["save_every"] == 221


def test_config_krea2():
    s = _spec(model_type="krea2", expected_repo_name="krepo", trigger_word="tok")
    cfg = config.build_config(s, num_images=24, hours_to_complete=1000)
    p = cfg["config"]["process"][0]
    assert cfg["config"]["name"] == "krepo"
    assert p["train"]["do_differential_guidance"] is True
    assert p["train"]["loss_type"] == "mse"
    assert p["network"]["lokr_full_rank"] is True
    mk = p["model"]["model_kwargs"]
    assert mk["text_encoder_path"] == "/cache/hf_cache/Qwen--Qwen3-VL-4B-Instruct"
    # vae_path is the model DIR (Krea2Model appends /vae itself)
    assert mk["vae_path"] == "/cache/models/stabilityai--x"


def test_config_ideogram4():
    s = _spec(model_type="ideogram4", expected_repo_name="irepo", trigger_word=None)
    cfg = config.build_config(s, num_images=24, hours_to_complete=1000)
    p = cfg["config"]["process"][0]
    assert cfg["config"]["name"] == "irepo"
    assert p["network"]["lokr_full_rank"] is True
    assert (
        p["model"]["unconditional_lora_path"]
        == "/cache/hf_cache/ideogram_4_unconditional_lora_r16.safetensors"
    )
    mk = p["model"]["model_kwargs"]
    assert mk["text_encoder_path"] == "/cache/hf_cache/Qwen--Qwen3-VL-8B-Instruct"
    assert "vae_path" not in mk
    assert p["trigger_word"] is None


def test_config_write_roundtrip(tmp_path):
    import yaml

    s = _spec(model_type="flux")
    cfg = config.build_config(s, num_images=24, hours_to_complete=1000)
    out = tmp_path / "cfg" / "t1.yaml"
    config.write_config(cfg, str(out))
    reloaded = yaml.safe_load(open(out))
    assert reloaded["config"]["name"] == "myrepo"


def test_resolve_base_model_single_safetensors(tmp_path):
    # The authoritative runtime entrypoint passes the DIR unconditionally so the
    # ai-toolkit loader can resolve per-arch subfolders — never collapse to a file.
    d = tmp_path / "model"
    d.mkdir()
    f = d / "weights.safetensors"
    f.write_bytes(b"x")
    assert config.resolve_base_model(str(d)) == str(d)


def test_resolve_base_model_dir(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "a.safetensors").write_bytes(b"x")
    (d / "b.safetensors").write_bytes(b"x")
    # multiple files -> stay dir
    assert config.resolve_base_model(str(d)) == str(d)


# --------------------------------------------------------------------------- #
# 10. finalize picks/creates last.safetensors
# --------------------------------------------------------------------------- #
def _write_st(path, tag=""):
    """Write a minimal valid one-tensor safetensors file and return its bytes."""
    import json as _json
    import struct as _struct

    header = _json.dumps(
        {
            "__metadata__": {"tag": tag},
            "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        }
    ).encode()
    path.write_bytes(_struct.pack("<Q", len(header)) + header + _struct.pack("<f", 0.0))
    return path.read_bytes()


def _read_selection(root):
    import json

    return json.loads((root / "forge_checkpoint_selection.json").read_text())


def _sha256(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_loss_db(path, state, losses):
    """Write the real ai-toolkit recorder schema used in tournament artifacts."""
    import sqlite3

    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE steps (step INTEGER PRIMARY KEY, wall_time REAL NOT NULL);
            CREATE TABLE metric_keys (
                key TEXT PRIMARY KEY, first_seen_step INTEGER, last_seen_step INTEGER
            );
            CREATE TABLE metrics (
                step INTEGER NOT NULL, key TEXT NOT NULL, value_real REAL,
                value_text TEXT, PRIMARY KEY (step, key)
            );
            """
        )
        conn.execute(
            "INSERT INTO metric_keys VALUES ('loss/loss', 1, ?)", (len(losses),)
        )
        for step, loss in enumerate(losses, 1):
            conn.execute(
                "INSERT INTO steps VALUES (?, ?)",
                (step, state["started_unix"] + step / 1000.0),
            )
            conn.execute(
                "INSERT INTO metrics VALUES (?, 'loss/loss', ?, NULL)",
                (step, loss),
            )


def test_default_selection_prefers_current_exact_final(tmp_path):
    state = checkpoints.begin_run(str(tmp_path), "repo9")
    state = checkpoints.set_planned_steps(str(tmp_path), state, 500)
    _write_st(tmp_path / "repo9_000000500.safetensors", tag="periodic")
    final_bytes = _write_st(tmp_path / "repo9.safetensors", tag="final")
    record = checkpoints.finalize(str(tmp_path), "repo9", state)
    assert (tmp_path / "last.safetensors").read_bytes() == final_bytes
    assert record["source"] == "exact_final"
    assert record["selected_step"] == 500
    assert record["sha256"]


def test_default_selection_highest_valid_periodic(tmp_path):
    state = checkpoints.begin_run(str(tmp_path), "repo")
    _write_st(tmp_path / "repo_000000200.safetensors", tag="old")
    newest = _write_st(tmp_path / "repo_000000600.safetensors", tag="new")
    record = checkpoints.finalize(str(tmp_path), "repo", state)
    assert (tmp_path / "last.safetensors").read_bytes() == newest
    assert record["source"] == "highest_valid_periodic"
    assert record["selected_step"] == 600


def test_selection_skips_truncated_current_newest(tmp_path):
    state = checkpoints.begin_run(str(tmp_path), "repo")
    ok = _write_st(tmp_path / "repo_000000200.safetensors", tag="ok")
    (tmp_path / "repo_000000600.safetensors").write_bytes(b"truncated")
    record = checkpoints.finalize(str(tmp_path), "repo", state)
    assert (tmp_path / "last.safetensors").read_bytes() == ok
    assert record["current_candidates_discovered"] == 2
    assert record["current_candidates_valid"] == 1


def test_same_task_retry_ignores_stale_last_and_higher_step(tmp_path):
    old_last = _write_st(tmp_path / "last.safetensors", tag="previous")
    stale_high = _write_st(tmp_path / "repo_000000900.safetensors", tag="stale")
    state = checkpoints.begin_run(str(tmp_path), "repo")
    current = _write_st(tmp_path / "repo_000000100.safetensors", tag="current")
    record = checkpoints.finalize(str(tmp_path), "repo", state)
    assert old_last != current != stale_high
    assert (tmp_path / "last.safetensors").read_bytes() == current
    assert record["selected_step"] == 100
    assert record["current_candidates_discovered"] == 1


def test_begin_run_blocks_aitoolkit_zero_step_autoresume(tmp_path):
    """Every model and fixed-name training state must be absent at launch."""
    prior_exact = _write_st(tmp_path / "repo.safetensors", tag="prior-final")
    _write_st(tmp_path / "repo_000000900.safetensors", tag="prior-periodic")
    (tmp_path / "repo.pt").write_bytes(b"old-state")
    (tmp_path / "repo_state").mkdir()
    (tmp_path / "optimizer.pt").write_bytes(b"old-optimizer")
    (tmp_path / "learnable_snr.json").write_text("{}", encoding="utf-8")

    state = checkpoints.begin_run(str(tmp_path), "repo")

    assert state["quarantine_complete"] is True
    assert set(state["quarantined"]) == {
        "learnable_snr.json",
        "optimizer.pt",
        "repo.pt",
        "repo.safetensors",
        "repo_000000900.safetensors",
        "repo_state",
    }
    assert not any(path.name.startswith("repo") for path in tmp_path.iterdir())
    assert not (tmp_path / "optimizer.pt").exists()
    assert not (tmp_path / "learnable_snr.json").exists()
    assert (tmp_path / "last.safetensors").read_bytes() == prior_exact
    # Handler reuse must not start a second journal or disturb the fallback.
    assert checkpoints.ensure_run(str(tmp_path), "repo") == state


def test_same_size_overwrite_with_restored_mtime_is_current_via_ctime(tmp_path):
    import time

    path = tmp_path / "repo_000000100.safetensors"
    old = _write_st(path, tag="old1")
    old_stat = path.stat()
    state = checkpoints.begin_run(str(tmp_path), "repo")
    # Model an unusual deterministic writer that restores mtime and produces the
    # same byte length.  ctime still changes on supported validator filesystems.
    time.sleep(0.002)
    new = _write_st(path, tag="new1")
    assert len(old) == len(new)
    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    assert path.stat().st_mtime_ns == old_stat.st_mtime_ns
    assert checkpoints.current_loras(str(tmp_path), state) == [str(path)]


def test_failed_retry_explicitly_preserves_previous_last(tmp_path):
    prior = _write_st(tmp_path / "last.safetensors", tag="previous")
    _write_st(tmp_path / "repo_000000900.safetensors", tag="stale-periodic")
    state = checkpoints.begin_run(str(tmp_path), "repo")
    (tmp_path / "repo_000000100.safetensors").write_bytes(b"failed-partial-save")
    record = checkpoints.finalize(str(tmp_path), "repo", state)
    assert (tmp_path / "last.safetensors").read_bytes() == prior
    assert record["status"] == "preserved_previous_run"
    assert record["source"] == "previous_run_fallback"
    assert record["current_candidates_discovered"] == 1
    assert record["current_candidates_valid"] == 0
    assert "no valid checkpoint" in record["reason"]


def test_heldout_manifest_selects_scored_checkpoint(tmp_path):
    import json

    state = checkpoints.begin_run(str(tmp_path), "repo")
    best = _write_st(tmp_path / "repo_000000100.safetensors", tag="best")
    _write_st(tmp_path / "repo_000000200.safetensors", tag="worse")
    _write_st(tmp_path / "repo.safetensors", tag="final")
    (tmp_path / "forge_holdout_scores.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "source": "heldout",
                "complete": True,
                "metric": "validator_combined_proxy",
                "direction": "min",
                "scores": [
                    {
                        "checkpoint": "repo_000000100.safetensors",
                        "score": 0.05,
                        "sha256": _sha256(tmp_path / "repo_000000100.safetensors"),
                    },
                    {
                        "checkpoint": "repo_000000200.safetensors",
                        "score": 0.08,
                        "sha256": _sha256(tmp_path / "repo_000000200.safetensors"),
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
    record = checkpoints.finalize(str(tmp_path), "repo", state)
    assert (tmp_path / "last.safetensors").read_bytes() == best
    assert record["source"] == "heldout_manifest"
    assert record["selected_step"] == 100
    assert record["score"] == 0.05


def test_incomplete_heldout_manifest_cannot_override_final(tmp_path):
    import json

    state = checkpoints.begin_run(str(tmp_path), "repo")
    early = tmp_path / "repo_000000100.safetensors"
    _write_st(early, tag="early")
    final = _write_st(tmp_path / "repo.safetensors", tag="final")
    (tmp_path / "forge_holdout_scores.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "source": "heldout",
                "complete": False,
                "direction": "min",
                "scores": [
                    {
                        "checkpoint": early.name,
                        "score": 0.01,
                        "sha256": _sha256(early),
                    }
                ],
            }
        )
    )

    record = checkpoints.finalize(str(tmp_path), "repo", state)

    assert (tmp_path / "last.safetensors").read_bytes() == final
    assert record["source"] == "exact_final"


def test_stale_holdout_manifest_is_ignored_on_retry(tmp_path):
    import json

    (tmp_path / "forge_holdout_scores.json").write_text(
        json.dumps(
            {
                "source": "heldout",
                "direction": "min",
                "scores": [
                    {"checkpoint": "repo_000000100.safetensors", "score": 0.01}
                ],
            }
        )
    )
    state = checkpoints.begin_run(str(tmp_path), "repo")
    _write_st(tmp_path / "repo_000000100.safetensors", tag="periodic")
    final = _write_st(tmp_path / "repo.safetensors", tag="final")
    record = checkpoints.finalize(str(tmp_path), "repo", state)
    assert (tmp_path / "last.safetensors").read_bytes() == final
    assert record["source"] == "exact_final"


def test_clear_sustained_training_loss_divergence_selects_early(tmp_path):
    state = checkpoints.begin_run(str(tmp_path), "repo")
    _write_st(tmp_path / "repo_000000045.safetensors", tag="45")
    best = _write_st(tmp_path / "repo_000000090.safetensors", tag="90")
    _write_st(tmp_path / "repo_000000135.safetensors", tag="135")
    _write_st(tmp_path / "repo.safetensors", tag="final")
    losses = [0.16] * 45 + [0.10] * 45 + [0.18] * 45 + [0.60] * 45
    _write_loss_db(tmp_path / "loss_log.db", state, losses)
    record = checkpoints.finalize(str(tmp_path), "repo", state)
    assert (tmp_path / "last.safetensors").read_bytes() == best
    assert record["source"] == "training_loss_divergence"
    assert record["selected_step"] == 90
    assert record["training_loss_is_proxy_not_validator_metric"] is True


def test_stable_improving_training_loss_keeps_final(tmp_path):
    state = checkpoints.begin_run(str(tmp_path), "repo")
    _write_st(tmp_path / "repo_000000045.safetensors", tag="45")
    _write_st(tmp_path / "repo_000000090.safetensors", tag="90")
    _write_st(tmp_path / "repo_000000135.safetensors", tag="135")
    final = _write_st(tmp_path / "repo.safetensors", tag="final")
    losses = [1.0 - step * 0.004 for step in range(1, 181)]
    _write_loss_db(tmp_path / "loss_log.db", state, losses)
    record = checkpoints.finalize(str(tmp_path), "repo", state)
    assert (tmp_path / "last.safetensors").read_bytes() == final
    assert record["source"] == "exact_final"


def test_loss_reader_accepts_current_uncheckpointed_sqlite_wal(tmp_path):
    import sqlite3

    db = tmp_path / "loss_log.db"
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE steps (step INTEGER PRIMARY KEY, wall_time REAL NOT NULL);
            CREATE TABLE metric_keys (
                key TEXT PRIMARY KEY, first_seen_step INTEGER, last_seen_step INTEGER
            );
            CREATE TABLE metrics (
                step INTEGER NOT NULL, key TEXT NOT NULL, value_real REAL,
                value_text TEXT, PRIMARY KEY (step, key)
            );
            INSERT INTO metric_keys VALUES ('loss/loss', 1, 1);
            """
        )
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    state = checkpoints.begin_run(str(tmp_path), "repo")
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "INSERT INTO steps VALUES (?, ?)",
            (1, state["started_unix"] + 0.1),
        )
        conn.execute(
            "INSERT INTO metrics VALUES (?, 'loss/loss', ?, NULL)", (1, 0.25)
        )
        conn.commit()
        assert (tmp_path / "loss_log.db-wal").is_file()
        # Model the deadline state where only WAL freshness is observable.
        state["before"]["loss_log.db"] = checkpoints._signature(str(db))

        assert checkpoints._loss_points(str(tmp_path), state) == [(1, 0.25)]
    finally:
        conn.close()


def test_atomic_records_and_promotion_fsync_parent_directory(tmp_path, monkeypatch):
    import stat

    real_fsync = os.fsync
    fsynced_directory = []

    def _spy(fd):
        fsynced_directory.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _spy)
    state = checkpoints.begin_run(str(tmp_path), "repo")
    _write_st(tmp_path / "repo_000000100.safetensors", tag="current")
    checkpoints.finalize(str(tmp_path), "repo", state)
    assert any(fsynced_directory)


def test_valid_safetensors():
    from forge.tasks.integrity import valid_safetensors
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path

        good = Path(d) / "g.safetensors"
        _write_st(good)
        assert valid_safetensors(str(good))
        bad = Path(d) / "b.safetensors"
        bad.write_bytes(b"xx")
        assert not valid_safetensors(str(bad))
        # truncated data section: header promises tensor bytes the file lacks
        import json as _json
        import struct as _struct

        hdr = _json.dumps(
            {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
        ).encode()
        cut = Path(d) / "c.safetensors"
        cut.write_bytes(_struct.pack("<Q", len(hdr)) + hdr + b"\x00" * 8)
        assert not valid_safetensors(str(cut))


def test_valid_safetensors_rejects_metadata_only_and_invalid_offsets(tmp_path):
    import json
    import struct

    from forge.tasks.integrity import valid_safetensors

    metadata_header = json.dumps({"__metadata__": {"format": "pt"}}).encode()
    metadata_only = tmp_path / "metadata-only.safetensors"
    metadata_only.write_bytes(struct.pack("<Q", len(metadata_header)) + metadata_header)
    assert not valid_safetensors(str(metadata_only))

    bad_header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [-1, 3]}}
    ).encode()
    bad_offsets = tmp_path / "bad-offsets.safetensors"
    bad_offsets.write_bytes(struct.pack("<Q", len(bad_header)) + bad_header + b"abc")
    assert not valid_safetensors(str(bad_offsets))

    invalid_metadata_header = json.dumps(
        {
            "__metadata__": {"step": 1},
            "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        }
    ).encode()
    invalid_metadata = tmp_path / "invalid-metadata.safetensors"
    invalid_metadata.write_bytes(
        struct.pack("<Q", len(invalid_metadata_header))
        + invalid_metadata_header
        + b"\0" * 4
    )
    assert not valid_safetensors(str(invalid_metadata))


@pytest.mark.parametrize(
    "header,payload",
    [
        (
            {"weight": {"dtype": "BOGUS", "shape": [1], "data_offsets": [0, 4]}},
            b"\0" * 4,
        ),
        (
            {"weight": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]}},
            b"\0" * 4,
        ),
        (
            {
                "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                "b": {"dtype": "F32", "shape": [1], "data_offsets": [2, 6]},
            },
            b"\0" * 6,
        ),
        (
            {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}},
            b"\0" * 4,
        ),
        (
            {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}},
            b"\0" * 8,
        ),
    ],
)
def test_valid_safetensors_rejects_bad_dtype_shape_and_layout(
    tmp_path, header, payload
):
    import json
    import struct

    from forge.tasks.integrity import valid_safetensors

    encoded = json.dumps(header).encode()
    path = tmp_path / "malformed.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)

    assert not valid_safetensors(str(path))


def test_finalize_creates_last(tmp_path, monkeypatch):
    s = _spec(expected_repo_name="repo")
    save_root = tmp_path / "repo"
    save_root.mkdir()
    scope = checkpoints.begin_run(str(save_root), "repo")
    _write_st(save_root / "repo_000000700.safetensors", tag="weights")
    final_bytes = _write_st(save_root / "repo.safetensors", tag="final")
    monkeypatch.setattr(type(s), "save_root", property(lambda self: str(save_root)))
    monkeypatch.setattr(type(s), "output_dir", property(lambda self: str(save_root)))
    aitoolkit._finalize(s, scope)
    last = save_root / "last.safetensors"
    assert last.is_file()
    assert last.read_bytes() == final_bytes


def test_finalize_raises_when_empty(tmp_path, monkeypatch):
    s = _spec(expected_repo_name="repo")
    save_root = tmp_path / "repo"
    save_root.mkdir()
    scope = checkpoints.begin_run(str(save_root), "repo")
    monkeypatch.setattr(type(s), "save_root", property(lambda self: str(save_root)))
    with pytest.raises(RuntimeError):
        aitoolkit._finalize(s, scope)


# --------------------------------------------------------------------------- #
# 11. fallback promote
# --------------------------------------------------------------------------- #
def test_fallback_promotes(tmp_path, monkeypatch):
    s = _spec(expected_repo_name="repo")
    save_root = tmp_path / "repo"
    save_root.mkdir()
    checkpoints.begin_run(str(save_root), "repo")
    expected = _write_st(save_root / "repo_000000300.safetensors", tag="current")
    monkeypatch.setattr(type(s), "save_root", property(lambda self: str(save_root)))
    fallback.emit_untrained_copy(s)
    assert (save_root / "last.safetensors").read_bytes() == expected


def test_fallback_empty_no_raise(tmp_path, monkeypatch):
    s = _spec(expected_repo_name="repo")
    save_root = tmp_path / "repo"
    save_root.mkdir()
    checkpoints.begin_run(str(save_root), "repo")
    monkeypatch.setattr(type(s), "save_root", property(lambda self: str(save_root)))
    # must not raise
    fallback.emit_untrained_copy(s)
    assert not (save_root / "last.safetensors").exists()


# --------------------------------------------------------------------------- #
# 12. dispatch + cli never-crash
# --------------------------------------------------------------------------- #
def test_dispatch_known_and_unknown():
    assert dispatch.for_model_type("flux") is not None
    assert dispatch.for_model_type("krea2") is not None
    assert dispatch.for_model_type("ideogram4") is not None
    assert dispatch.for_model_type("z-image") is not None
    assert dispatch.for_model_type("qwen-image") is not None
    assert dispatch.for_model_type("sdxl") is None  # retired type stays unknown


def test_cli_never_crash_missing_cache():
    from forge import cli

    rc = cli.main(
        [
            "--task-id", "nope",
            "--model", "stabilityai/x",
            "--model-type", "flux",
            "--expected-repo-name", "repoZ",
            "--hours-to-complete", "0.01",
        ]
    )
    assert rc == 0


def test_cli_unknown_type_fallback():
    from forge import cli

    rc = cli.main(
        [
            "--task-id", "nope2",
            "--model", "stabilityai/x",
            "--model-type", "z-image",
            "--expected-repo-name", "repoY",
            "--hours-to-complete", "0.01",
            "--extra-unknown-flag", "junk",
        ]
    )
    assert rc == 0


# --------------------------------------------------------------------------- #
# 12. review-fix regressions: stem collisions, caption case, argparse funnel
# --------------------------------------------------------------------------- #
def test_collect_flat_stem_collision_no_caption_crosswire(tmp_path):
    from forge.data.dataset import _collect_flat

    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "b").mkdir(parents=True)
    (src / "a" / "x.jpg").write_bytes(b"img-a")
    (src / "a" / "x.txt").write_text("caption-a")
    (src / "b" / "x.png").write_bytes(b"img-b")
    (src / "b" / "x.txt").write_text("caption-b")
    dest = tmp_path / "flat"
    assert _collect_flat(str(src), str(dest))
    caps = sorted(p.read_text() for p in dest.glob("*.txt"))
    assert caps == ["caption-a", "caption-b"]  # neither clobbered
    imgs = sorted(p.name for p in dest.iterdir() if p.suffix != ".txt")
    stems = {p.rsplit(".", 1)[0] for p in imgs}
    assert len(stems) == 2  # distinct stems → unambiguous pairing


def test_collect_flat_caption_extension_case(tmp_path):
    from forge.data.dataset import _collect_flat

    src = tmp_path / "src"
    src.mkdir()
    (src / "IMG0.jpg").write_bytes(b"i")
    (src / "IMG0.TXT").write_text("upper-ext caption")
    dest = tmp_path / "flat"
    assert _collect_flat(str(src), str(dest))
    assert (dest / "IMG0.txt").read_text() == "upper-ext caption"


def test_fallback_skips_truncated_newest(tmp_path, monkeypatch):
    s = _spec(expected_repo_name="repo")
    save_root = tmp_path / "repo"
    save_root.mkdir()
    checkpoints.begin_run(str(save_root), "repo")
    ok_bytes = _write_st(save_root / "repo_000000100.safetensors", tag="ok")
    (save_root / "repo_000000900.safetensors").write_bytes(b"trunc")
    monkeypatch.setattr(type(s), "save_root", property(lambda self: str(save_root)))
    fallback.emit_untrained_copy(s)
    assert (save_root / "last.safetensors").read_bytes() == ok_bytes


def test_parse_never_raises_systemexit():
    from forge import cli

    # malformed/missing args must not escape as SystemExit(2)
    ns = cli._parse(["--hours-to-complete"])  # dangling value
    assert ns.task_id is None
    assert cli.main(["--model-type", "flux"]) == 0  # missing task-id → clean 0


# --------------------------------------------------------------------------- #
# 13. z-image / qwen-image coverage (knockout types)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mt,tmpl_arch", [("z-image", "zimage:turbo"),
                                          ("qwen-image", "qwen_image")])
def test_new_types_config(mt, tmpl_arch):
    s = _spec(model_type=mt, expected_repo_name="repoX")
    cfg = config.build_config(s, num_images=20, hours_to_complete=1.0)
    assert cfg["config"]["name"] == "repoX"
    p = cfg["config"]["process"][0]
    assert p["model"]["arch"] == tmpl_arch
    assert p["model"]["name_or_path"] == s.cached_model_dir
    assert p["datasets"][0]["folder_path"] == "/dataset/images"
    assert 1 <= p["train"]["steps"] <= 3000
    # validator-staged aux paths survive untouched
    if mt == "z-image":
        assert p["model"]["assistant_lora_path"] == (
            "/cache/hf_cache/zimage_turbo_training_adapter_v2.safetensors"
        )
    else:
        assert p["model"]["qtype"].endswith("qwen_image_torchao_uint3.safetensors")


def test_new_types_dispatchable():
    from forge.tasks import dispatch as d

    for mt in ("z-image", "qwen-image"):
        s = _spec(model_type=mt)
        assert s.model_type == mt


def test_image_dockerfiles_repin_torchcodec_for_torch26():
    """Keep ai-toolkit's media extension on the ABI G.O.D actually ships."""
    names = (
        "standalone-image-trainer.dockerfile",
        "standalone-image-toolkit-trainer.dockerfile",
    )
    contents = []
    for name in names:
        path = os.path.join(REPO_ROOT, "ops", "docker", name)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        assert "torch==2.6.0" in text
        assert "torchcodec==0.2.1" in text
        contents.append(text)
    assert contents[0] == contents[1]
