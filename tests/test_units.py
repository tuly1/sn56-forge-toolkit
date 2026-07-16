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
from forge.tasks import aitoolkit, dispatch, fallback  # noqa: E402


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
    # capped at 100 so the FIRST checkpoint always lands early (kill-safety)
    assert recipe.kill_safe_save_every(2000, 250) == 100
    assert recipe.kill_safe_save_every(700, 250) == 87  # steps//8
    assert recipe.kill_safe_save_every(2, 250) == 1  # floor >= 1


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
    # save_every kill-safe, capped at 100 (min(250, 1100//8=137, 100) = 100)
    assert p["save"]["save_every"] == 100


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
    """Write a minimal VALID safetensors file (header-only) and return its bytes."""
    import json as _json
    import struct as _struct

    header = _json.dumps({"__metadata__": {"tag": tag}}).encode()
    path.write_bytes(_struct.pack("<Q", len(header)) + header)
    return path.read_bytes()


def test_pick_final_prefers_exact_repo(tmp_path):
    root = tmp_path
    p1 = root / "repo_000000700.safetensors"
    p2 = root / "repo.safetensors"
    _write_st(p1)
    _write_st(p2)
    loras = [str(p1), str(p2)]
    assert aitoolkit._pick_final(loras, "repo") == str(p2)


def test_pick_final_highest_step_when_no_exact(tmp_path):
    root = tmp_path
    p1 = root / "repo_000000200.safetensors"
    p2 = root / "repo_000000600.safetensors"
    _write_st(p1)
    _write_st(p2)
    assert aitoolkit._pick_final([str(p1), str(p2)], "repo") == str(p2)


def test_pick_final_digit_trap(tmp_path):
    # repo name ends in a digit — exact-match branch must win over step regex
    root = tmp_path
    exact = root / "repo9.safetensors"
    periodic = root / "repo9_000000500.safetensors"
    _write_st(exact)
    _write_st(periodic)
    assert aitoolkit._pick_final([str(periodic), str(exact)], "repo9") == str(exact)


def test_pick_final_skips_truncated_newest(tmp_path):
    # deadline kill mid-save: newest file truncated → step down to valid older
    ok = tmp_path / "repo_000000200.safetensors"
    trunc = tmp_path / "repo_000000600.safetensors"
    _write_st(ok)
    trunc.write_bytes(b"x")  # not a valid safetensors
    assert aitoolkit._pick_final([str(ok), str(trunc)], "repo") == str(ok)


def test_pick_final_skips_corrupt_exact_final(tmp_path):
    exact = tmp_path / "repo.safetensors"
    periodic = tmp_path / "repo_000000500.safetensors"
    exact.write_bytes(b"x")  # corrupt final save
    _write_st(periodic)
    assert aitoolkit._pick_final([str(exact), str(periodic)], "repo") == str(periodic)


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


def test_finalize_creates_last(tmp_path, monkeypatch):
    s = _spec(expected_repo_name="repo")
    save_root = tmp_path / "repo"
    save_root.mkdir()
    _write_st(save_root / "repo_000000700.safetensors", tag="weights")
    final_bytes = _write_st(save_root / "repo.safetensors", tag="final")
    monkeypatch.setattr(type(s), "save_root", property(lambda self: str(save_root)))
    monkeypatch.setattr(type(s), "output_dir", property(lambda self: str(save_root)))
    aitoolkit._finalize(s)
    last = save_root / "last.safetensors"
    assert last.is_file()
    assert last.read_bytes() == final_bytes


def test_finalize_raises_when_empty(tmp_path, monkeypatch):
    s = _spec(expected_repo_name="repo")
    save_root = tmp_path / "repo"
    save_root.mkdir()
    monkeypatch.setattr(type(s), "save_root", property(lambda self: str(save_root)))
    with pytest.raises(RuntimeError):
        aitoolkit._finalize(s)


# --------------------------------------------------------------------------- #
# 11. fallback promote
# --------------------------------------------------------------------------- #
def test_fallback_promotes(tmp_path, monkeypatch):
    s = _spec(expected_repo_name="repo")
    save_root = tmp_path / "repo"
    save_root.mkdir()
    (save_root / "repo_000000300.safetensors").write_bytes(b"w")
    monkeypatch.setattr(type(s), "save_root", property(lambda self: str(save_root)))
    fallback.emit_untrained_copy(s)
    assert (save_root / "last.safetensors").is_file()


def test_fallback_empty_no_raise(tmp_path, monkeypatch):
    s = _spec(expected_repo_name="repo")
    save_root = tmp_path / "repo"
    save_root.mkdir()
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
