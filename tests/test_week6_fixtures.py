"""Contract tests for the week-6 real-data fixture builder.

These are ML-free.  Most run against a synthetic harvest-shaped store built in
``tmp_path`` so that duplicate handling and the seal can be exercised on inputs
we control; the real Aug-3 harvest contains no duplicate pairs, so it cannot
prove the duplicate rule on its own.  A small number of tests do read the real
harvest, read-only, to pin the split of the task we were eliminated on.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "ops" / "experiments" / "week6" / "build_fixtures.py"
)
_spec = importlib.util.spec_from_file_location("week6_build_fixtures", _MODULE_PATH)
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)

REAL_HARVEST = bf.DEFAULT_HARVEST_ROOT
PRIMARY_TASK = "41025fb5-8473-40c6-a88d-20c0bb303edc"

# Filesystem audit trail.  Recording is off unless a test pushes a sink, so the
# hook costs one truthiness check per event for the rest of the session.
_AUDIT_SINK: list[list[str]] = []
_AUDITED_EVENTS = frozenset(
    {"open", "os.scandir", "os.listdir", "os.stat", "os.mkdir", "shutil.copyfile"}
)


def _audit(event: str, args) -> None:
    if not _AUDIT_SINK or event not in _AUDITED_EVENTS:
        return
    sink = _AUDIT_SINK[-1]
    for arg in args:
        if isinstance(arg, str):
            sink.append(arg)
        elif isinstance(arg, (bytes, Path)):
            sink.append(str(arg))


sys.addaudithook(_audit)

real_harvest_only = pytest.mark.skipif(
    not (REAL_HARVEST / "tasks" / PRIMARY_TASK / "pairs").is_dir(),
    reason="Aug-3 harvest store is not present on this machine",
)


# --------------------------------------------------------------------------
# synthetic harvest store
# --------------------------------------------------------------------------


def _blocky(seed: int, size: tuple[int, int]) -> Image.Image:
    """Low-entropy deterministic image: PNG stays small, content stays distinct."""
    width, height = size
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    tile_h = -(-height // 6)
    tile_w = -(-width // 8)
    array = np.repeat(np.repeat(blocks, tile_h, axis=0), tile_w, axis=1)[:height, :width]
    return Image.fromarray(array, mode="RGB")


def make_task(
    root: Path,
    task_id: str,
    n_pairs: int,
    size: tuple[int, int] = (1024, 768),
    exact_dup: tuple[int, int] | None = None,
    near_dup: tuple[int, int] | None = None,
    trigger: str = "SynthUI",
) -> Path:
    task_dir = root / "tasks" / task_id
    pairs = task_dir / "pairs"
    pairs.mkdir(parents=True)
    for index in range(n_pairs):
        stem = f"{index:03d}"
        image = _blocky(1000 + index, size)
        image.save(pairs / f"{stem}.png")
        (pairs / f"{stem}.txt").write_text(
            f"{trigger}: synthetic caption {stem}", encoding="utf-8"
        )
    if exact_dup is not None:
        source, target = exact_dup
        shutil.copyfile(pairs / f"{source:03d}.png", pairs / f"{target:03d}.png")
    if near_dup is not None:
        source, target = near_dup
        array = np.asarray(Image.open(pairs / f"{source:03d}.png").convert("RGB")).copy()
        array[0, 0] = (array[0, 0].astype(int) + 3) % 256
        Image.fromarray(array, mode="RGB").save(pairs / f"{target:03d}.png")

    (task_dir / "task-meta.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "round": 1,
                "status": "success",
                "task_type": "ImageTask",
                "is_organic": False,
                "model_type": "krea2",
                "base_model": "krea/Krea-2-Raw",
                "ds": f"design_{task_id}",
                "ds_short": f"design_{task_id[:8]}",
                "family": "design",
                "n_pairs": n_pairs,
                "hours_to_complete": 1.0,
                "trigger_word": trigger,
            }
        ),
        encoding="utf-8",
    )
    lines = []
    for path in sorted(pairs.iterdir()):
        lines.append(f"{bf.sha256_file(path)}  pairs/{path.name}")
    (task_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return task_dir


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """A two-discovery / two-confirmation store with planted duplicates."""
    harvest = tmp_path / "harvest"
    make_task(harvest, "disc-small", 21)
    make_task(harvest, "disc-wide", 24, size=(1408, 768))
    make_task(harvest, "conf-a", 26, exact_dup=(0, 1))
    make_task(harvest, "conf-b", 28, near_dup=(0, 1))
    # Mirrors the sealed directory in the real store; must never be opened.
    quarantine = harvest / "QUARANTINE-test-data"
    quarantine.mkdir()
    (quarantine / "sealed.zip").write_bytes(b"do not read")
    monkeypatch.setattr(
        bf,
        "PRIMARY_TASKS",
        {"discovery": ["disc-small", "disc-wide"], "confirmation": ["conf-a", "conf-b"]},
    )
    return harvest


def build_into(harvest: Path, out: Path, seed: int = bf.DEFAULT_SEED) -> dict:
    return bf.build(harvest, out, seed, _MODULE_PATH)


def tree_hashes(root: Path, skip: set[str] | None = None) -> dict[str, str]:
    skip = skip or set()
    return {
        p.relative_to(root).as_posix(): bf.sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.relative_to(root).as_posix() not in skip
    }


# --------------------------------------------------------------------------
# preprocessing conformance
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        ((1024, 768), (1024, 768)),
        ((1408, 768), (1024, 544)),
        ((1195, 896), (1024, 752)),
        ((768, 1376), (560, 1024)),
        ((1376, 768), (1024, 560)),
    ],
)
def test_transform_matches_pinned_validator(source, expected):
    """Every resolution cohort measured in the Aug-3 harvest, pinned.

    Expected values are derived from the pinned validator source
    (image_io.py:16-39 @ b026da04), not from our own output.
    """
    out = bf.adjust_image_size(_blocky(1, source))
    assert out.size == expected
    assert bf.describe_adjustment(source)["output"] == list(expected)


def test_floor_is_16_not_8():
    """1195x896 is the cohort that separates the two rules.

    Long-edge scale gives 1024x767. Flooring to 16 yields 752; flooring to 8
    would yield 760. The validator floors to 16.
    """
    described = bf.describe_adjustment((1195, 896))
    assert described["lanczos_resize_to"] == [1024, 767]
    assert described["floor16"] == [1024, 752]
    assert described["output"] == [1024, 752]
    assert (767 // 8) * 8 == 760 != 752


def test_transform_uses_lanczos_and_center_crop():
    described = bf.describe_adjustment((1408, 768))
    # 14 rows removed, 7 from the top: integer-floor center crop.
    assert described["rows_cropped"] == 14
    assert described["crop_box"] == [0, 7, 1024, 551]


def test_identity_cohort_is_pixel_identical(tmp_path):
    """1024x768 is 57% of the corpus; the validator's resize is a no-op there."""
    path = tmp_path / "a.png"
    _blocky(7, (1024, 768)).save(path)
    _, record = bf.preprocess_to_png_bytes(path)
    assert record["geometry_is_identity"] is True
    assert record["pixels_identical_to_source"] is True


# --------------------------------------------------------------------------
# split policy
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_pairs,expected_holdout,downgraded",
    [(21, 9, True), (22, 10, False), (42, 10, False), (43, 10, False), (50, 10, False)],
)
def test_plan_split(n_pairs, expected_holdout, downgraded):
    holdout, note = bf.plan_split(n_pairs)
    assert holdout == expected_holdout
    assert (note is not None) is downgraded
    assert n_pairs - holdout >= bf.TRAIN_MIN
    assert bf.HOLDOUT_MIN <= holdout <= bf.HOLDOUT_TARGET


def test_plan_split_refuses_when_task_is_too_small():
    with pytest.raises(bf.FixtureError):
        bf.plan_split(19)


def test_holdout_size_bounds_in_built_fixtures(synthetic, tmp_path):
    build_into(synthetic, tmp_path / "out")
    for manifest_path in (tmp_path / "out" / "discovery").rglob("manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        holdout_n = manifest["split"]["holdout_n"]
        assert bf.HOLDOUT_MIN <= holdout_n <= bf.HOLDOUT_TARGET
        assert manifest["split"]["train_n"] >= bf.TRAIN_MIN
        assert holdout_n == len(manifest["split"]["holdout"])


def test_holdout_downgrade_is_recorded(synthetic, tmp_path):
    build_into(synthetic, tmp_path / "out")
    manifest = json.loads(
        (tmp_path / "out" / "discovery" / "disc-small" / "manifest.json").read_text("utf-8")
    )
    assert manifest["split"]["holdout_n"] == 9
    assert manifest["split"]["train_n"] == 12
    assert manifest["split"]["holdout_downgraded"] is True
    assert "12-row training minimum" in manifest["split"]["holdout_downgrade_note"]


# --------------------------------------------------------------------------
# leakage
# --------------------------------------------------------------------------


def test_no_train_holdout_overlap_and_full_coverage(synthetic, tmp_path):
    out = tmp_path / "out"
    build_into(synthetic, out)
    bf.unseal(out, json.loads((out / "SEALED" / "SEAL.json").read_text())["commitment_sha256"])
    for manifest_path in out.rglob("manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        holdout = set(manifest["split"]["holdout"])
        train = set(manifest["split"]["train"])
        assert not (holdout & train)
        assert holdout | train == {i["stem"] for i in manifest["images"]}
        # and the emitted files agree with the manifest
        task_dir = manifest_path.parent
        on_disk_holdout = {p.stem for p in (task_dir / "holdout" / "source").glob("*.png")}
        on_disk_train = {p.stem for p in (task_dir / "train" / "source").glob("*.png")}
        assert on_disk_holdout == holdout
        assert on_disk_train == train
        assert not (on_disk_holdout & on_disk_train)


def test_no_image_sha256_appears_on_both_sides(synthetic, tmp_path):
    out = tmp_path / "out"
    build_into(synthetic, out)
    manifest = json.loads(
        (out / "discovery" / "disc-wide" / "manifest.json").read_text("utf-8")
    )
    by_side: dict[str, set[str]] = {"train": set(), "holdout": set()}
    for record in manifest["images"]:
        by_side[record["split"]].add(record["source"]["sha256"])
    assert not (by_side["train"] & by_side["holdout"])


def test_exact_duplicates_are_placed_on_the_same_side(synthetic, tmp_path):
    out = tmp_path / "out"
    build_into(synthetic, out)
    bf.unseal(out, json.loads((out / "SEALED" / "SEAL.json").read_text())["commitment_sha256"])
    manifest = json.loads(
        (out / "confirmation" / "conf-a" / "manifest.json").read_text("utf-8")
    )
    assert ["000", "001"] in manifest["dedup"]["duplicate_groups"]
    assert manifest["dedup"]["exact_duplicate_pairs"]
    sides = {r["split"] for r in manifest["images"] if r["stem"] in {"000", "001"}}
    assert len(sides) == 1


def test_near_duplicates_are_placed_on_the_same_side(synthetic, tmp_path):
    out = tmp_path / "out"
    build_into(synthetic, out)
    bf.unseal(out, json.loads((out / "SEALED" / "SEAL.json").read_text())["commitment_sha256"])
    manifest = json.loads(
        (out / "confirmation" / "conf-b" / "manifest.json").read_text("utf-8")
    )
    assert ["000", "001"] in manifest["dedup"]["duplicate_groups"]
    near = manifest["dedup"]["near_duplicate_pairs"]
    assert near and not manifest["dedup"]["exact_duplicate_pairs"]
    assert near[0]["dhash_hamming"] <= bf.DHASH_THRESHOLD
    assert near[0]["norm_mse"] <= bf.DUP_MSE_THRESHOLD
    sides = {r["split"] for r in manifest["images"] if r["stem"] in {"000", "001"}}
    assert len(sides) == 1


def test_duplicate_screen_needs_both_criteria():
    """dHash alone over-groups this corpus; the pixel check is what makes it safe."""
    assert bf.DHASH_THRESHOLD == 6
    assert bf.DUP_MSE_THRESHOLD == 0.005
    assert "AND" in bf.screen_duplicates([], [], [])[1]["rule"]


# --------------------------------------------------------------------------
# determinism and no-re-roll
# --------------------------------------------------------------------------


def test_build_is_byte_identical_across_runs(synthetic, tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    build_into(synthetic, first)
    build_into(synthetic, second)
    # README embeds the absolute output root by design.
    left = tree_hashes(first, skip={"README.md"})
    right = tree_hashes(second, skip={"README.md"})
    assert left == right
    assert len(left) > 100


def test_verify_reports_identical(synthetic, tmp_path):
    out = tmp_path / "out"
    build_into(synthetic, out)
    result = bf.verify(out, synthetic, bf.DEFAULT_SEED, _MODULE_PATH)
    assert result["identical"] is True
    assert result["differing"] == []
    assert result["files_compared"] > 100


def test_build_refuses_to_overwrite(synthetic, tmp_path):
    out = tmp_path / "out"
    build_into(synthetic, out)
    before = tree_hashes(out)
    with pytest.raises(bf.FixtureError, match="refusing to overwrite"):
        build_into(synthetic, out)
    assert tree_hashes(out) == before


def test_a_different_seed_produces_a_different_split(synthetic, tmp_path):
    build_into(synthetic, tmp_path / "a", seed=bf.DEFAULT_SEED)
    build_into(synthetic, tmp_path / "b", seed=bf.DEFAULT_SEED + 1)
    splits = []
    for root in ("a", "b"):
        manifest = json.loads(
            (tmp_path / root / "discovery" / "disc-wide" / "manifest.json").read_text("utf-8")
        )
        assert manifest["seed"] == (
            bf.DEFAULT_SEED if root == "a" else bf.DEFAULT_SEED + 1
        )
        splits.append(manifest["split"]["holdout"])
    assert splits[0] != splits[1]


def test_manifest_carries_no_wall_clock(synthetic, tmp_path):
    out = tmp_path / "out"
    build_into(synthetic, out)
    for name in ("FIXTURES-INDEX.json", "SEALED/SEAL.json"):
        assert "20" + "26-" not in (out / name).read_text("utf-8").replace(
            "week6-tournament-dataset-harvest-20260806", ""
        ).replace("week6-fixtures-20260806", "")
    manifest = (out / "discovery" / "disc-small" / "manifest.json").read_text("utf-8")
    for banned in ("created_at", "built_at", "timestamp"):
        assert banned not in manifest


def test_manifest_records_required_provenance(synthetic, tmp_path):
    out = tmp_path / "out"
    build_into(synthetic, out)
    manifest_path = out / "discovery" / "disc-wide" / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))

    assert manifest["task"]["task_id"] == "disc-wide"
    assert manifest["task"]["ds_short"]
    assert manifest["task"]["model_type"] == "krea2"
    assert manifest["trigger_verification"]["phrase"] == "SynthUI"
    assert manifest["seed"] == bf.DEFAULT_SEED
    assert manifest["builder_version"] == bf.BUILDER_VERSION
    assert manifest["preprocessing"]["divisibility"] == 16
    assert manifest["preprocessing"]["pin"]["commit"].startswith("b026da04")

    for record in manifest["images"]:
        assert len(record["source"]["sha256"]) == 64
        assert record["source"]["width"] == 1408 and record["source"]["height"] == 768
        assert record["preprocessed"]["output"] == [1024, 544]
        assert record["split"] in {"train", "holdout"}

    declared = (out / "discovery" / "disc-wide" / "manifest.sha256").read_text().split()[0]
    assert declared == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert declared == bf.sha256_bytes(bf.canonical_json(manifest))


# --------------------------------------------------------------------------
# the seal
# --------------------------------------------------------------------------


def test_confirmation_is_unreadable_until_unsealed(synthetic, tmp_path):
    out = tmp_path / "out"
    result = build_into(synthetic, out)

    assert not (out / "confirmation").exists()
    with pytest.raises(bf.SealedFixtureError):
        bf.load_group(out, "confirmation")
    # discovery stays readable
    assert len(bf.load_group(out, "discovery")) == 2

    # no confirmation image or manifest exists in readable form anywhere
    readable = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert not [r for r in readable if r.startswith("confirmation/")]
    assert not [r for r in readable if r.endswith(".png") and "conf-" in r]
    blob = (out / "SEALED" / "confirmation.sealed").read_bytes()
    assert b"\x89PNG" not in blob
    assert b"sn56.week6.fixture-manifest" not in blob
    assert b"conf-a" not in blob

    # only the commitment is published
    assert len(result["confirmation_commitment_sha256"]) == 64


def test_seal_does_not_publish_split_assignments(synthetic, tmp_path):
    out = tmp_path / "out"
    build_into(synthetic, out)
    commitment = json.loads((out / "SEALED" / "SEAL.json").read_text())["commitment_sha256"]
    published = (out / "SEALED" / "SEAL.json").read_text("utf-8") + (
        out / "FIXTURES-INDEX.json"
    ).read_text("utf-8")

    bf.unseal(out, commitment)
    manifest = json.loads((out / "confirmation" / "conf-a" / "manifest.json").read_text())
    holdout = manifest["split"]["holdout"]
    assert len(holdout) >= bf.HOLDOUT_MIN
    # the published files never named which stems were held out
    assert "holdout" not in json.loads(
        (out / "FIXTURES-INDEX.json").read_text()
    )["confirmation"]
    assert f'"{holdout[0]}"' not in published


def test_unseal_rejects_a_wrong_commitment(synthetic, tmp_path):
    out = tmp_path / "out"
    build_into(synthetic, out)
    with pytest.raises(bf.FixtureError, match="commitment is wrong"):
        bf.unseal(out, "0" * 64)
    assert not (out / "confirmation").exists()


def test_unseal_detects_a_tampered_container(synthetic, tmp_path):
    out = tmp_path / "out"
    result = build_into(synthetic, out)
    container = out / "SEALED" / "confirmation.sealed"
    blob = bytearray(container.read_bytes())
    blob[-1] ^= 0xFF
    container.write_bytes(bytes(blob))
    with pytest.raises(bf.FixtureError, match="has been modified"):
        bf.unseal(out, result["confirmation_commitment_sha256"])


def test_unseal_matches_the_commitment_and_leaves_a_receipt(synthetic, tmp_path):
    out = tmp_path / "out"
    result = build_into(synthetic, out)
    commitment = result["confirmation_commitment_sha256"]

    unsealed = bf.unseal(out, commitment)
    assert unsealed["files_written"] > 0

    combined = (out / "confirmation" / "confirmation-manifest.json").read_bytes()
    assert bf.sha256_bytes(combined) == commitment

    receipt = json.loads((out / "confirmation" / "UNSEAL-RECEIPT.json").read_text())
    assert receipt["commitment_sha256"] == commitment
    assert receipt["unsealed_at_utc"]

    assert len(bf.load_group(out, "confirmation")) == 2


def test_unseal_refuses_to_run_twice(synthetic, tmp_path):
    out = tmp_path / "out"
    result = build_into(synthetic, out)
    bf.unseal(out, result["confirmation_commitment_sha256"])
    with pytest.raises(bf.FixtureError, match="refusing to unseal twice"):
        bf.unseal(out, result["confirmation_commitment_sha256"])


# --------------------------------------------------------------------------
# the source store is read-only
# --------------------------------------------------------------------------


def test_harvest_store_is_not_mutated(synthetic, tmp_path):
    before = tree_hashes(synthetic)
    out = tmp_path / "out"
    build_into(synthetic, out)
    bf.unseal(out, json.loads((out / "SEALED" / "SEAL.json").read_text())["commitment_sha256"])
    assert tree_hashes(synthetic) == before


def test_builder_rejects_a_corrupted_harvest(synthetic, tmp_path):
    """SHA256SUMS is cross-checked, so a silently edited harvest fails closed."""
    victim = synthetic / "tasks" / "disc-small" / "pairs" / "000.png"
    _blocky(999, (1024, 768)).save(victim)
    with pytest.raises(bf.FixtureError, match="SHA256SUMS mismatch"):
        build_into(synthetic, tmp_path / "out")


def test_builder_rejects_an_unpaired_image(tmp_path, monkeypatch):
    harvest = tmp_path / "harvest"
    make_task(harvest, "solo", 21)
    (harvest / "tasks" / "solo" / "pairs" / "000.txt").unlink()
    monkeypatch.setattr(bf, "PRIMARY_TASKS", {"discovery": ["solo"], "confirmation": []})
    with pytest.raises(bf.FixtureError, match="no caption"):
        build_into(harvest, tmp_path / "out")


# --------------------------------------------------------------------------
# the real Aug-3 harvest
# --------------------------------------------------------------------------


@real_harvest_only
def test_real_primary_task_split_is_pinned(tmp_path):
    """The Aug-3 R1 task we were eliminated on: 21 pairs -> 9 holdout / 12 train."""
    result = bf.build_task(
        REAL_HARVEST, PRIMARY_TASK, "discovery", bf.DEFAULT_SEED, tmp_path / "t", "x" * 64
    )
    manifest = result["manifest"]
    assert manifest["task"]["model_type"] == "krea2"
    assert manifest["task"]["ds_short"] == "design_aetherflow_ui"
    assert manifest["task"]["round"] == 1
    assert manifest["trigger_verification"]["phrase"] == "AetherFlow UI"
    assert manifest["trigger_verification"]["captions_containing_phrase_verbatim"] == 21
    assert manifest["split"]["holdout_n"] == 9
    assert manifest["split"]["train_n"] == 12
    assert manifest["split"]["holdout_downgraded"] is True
    assert manifest["dedup"]["n_multi_member_groups"] == 0
    assert manifest["resolution_distribution"] == {
        "before_preprocessing": {"1024x768": 21},
        "after_preprocessing": {"1024x768": 21},
    }
    assert all(r["preprocessed"]["pixels_identical_to_source"] for r in manifest["images"])


@real_harvest_only
def test_real_mixed_resolution_task_is_transformed(tmp_path):
    """db9f7244 carries three non-1024 cohorts; every one is resized for scoring."""
    result = bf.build_task(
        REAL_HARVEST,
        "db9f7244-b3c8-4f85-96d9-7184af8b4179",
        "discovery",
        bf.DEFAULT_SEED,
        tmp_path / "t",
        "x" * 64,
    )
    manifest = result["manifest"]
    assert manifest["resolution_distribution"] == {
        "before_preprocessing": {"1376x768": 8, "1408x768": 17, "768x1376": 18},
        "after_preprocessing": {"1024x560": 8, "1024x544": 17, "560x1024": 18},
    }
    assert manifest["split"]["holdout_n"] == 10
    assert manifest["split"]["train_n"] == 33
    assert not any(r["preprocessed"]["geometry_is_identity"] for r in manifest["images"])


def test_builder_never_touches_quarantine_or_anything_outside_pairs(synthetic, tmp_path):
    """Behavioural, not textual: audit every filesystem access during a build."""
    sink: list[str] = []
    _AUDIT_SINK.append(sink)
    try:
        build_into(synthetic, tmp_path / "out")
    finally:
        _AUDIT_SINK.pop()

    root = str(synthetic)
    touched = sorted({p for p in sink if p.startswith(root)})
    assert touched, "the audit hook recorded nothing; the test would be vacuous"
    assert not [p for p in touched if "QUARANTINE" in p]

    allowed_suffixes = ("/task-meta.json", "/SHA256SUMS")
    for path in touched:
        rel = Path(path).relative_to(synthetic).as_posix()
        if rel in {"", ".", "tasks"} or rel.startswith("tasks/") and rel.count("/") == 1:
            continue  # the store root and task directories themselves
        assert (
            "/pairs/" in f"/{rel}"
            or rel.endswith("/pairs")
            or path.endswith(allowed_suffixes)
        ), f"builder touched an unexpected harvest path: {rel}"

    # and the quarantined payload is byte-unchanged
    assert (synthetic / "QUARANTINE-test-data" / "sealed.zip").read_bytes() == b"do not read"
