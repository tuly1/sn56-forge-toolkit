#!/usr/bin/env python3
"""Build deterministic train/holdout fixtures from the Aug-3 tournament harvest.

Week-5's confirmation step was non-discriminative because its holdout was two
or three images wide.  This builder fixes that: it cuts an 8-10 image holdout
out of a real harvested task, preprocesses the holdout exactly the way the
pinned validator does, and records every hash needed to prove the split was
never re-rolled.

Three properties are load-bearing.

*Read-only source.*  The harvest store is opened for reading and never written,
renamed, or created within.  Nothing under ``QUARANTINE-test-data`` is touched;
this builder only ever descends into ``tasks/<task_id>/pairs``.

*Determinism.*  The split is a pure function of ``(seed, task_id,
builder_version, the set of near-duplicate groups)``.  There is no RNG, no
wall-clock, and no dependence on filesystem ordering: groups are ranked by
SHA-256 of a fixed key string.  Re-running the builder on the same inputs
reproduces every output byte for byte, and the builder refuses to write into a
directory that already has contents, so a split cannot be quietly re-rolled.

*Discovery / confirmation separation.*  Discovery fixtures are plain files.
Confirmation fixtures are packed into a single opaque container and their
contents are not present on disk in readable form.  Building emits only the
commitment -- the SHA-256 of the confirmation manifest.  An explicit ``unseal``
invocation carrying that commitment is required before the confirmation images
can be read, and it leaves a dated receipt.

The seal is a *procedural* barrier, not secrecy.  The commitment is published
at build time and is also the container key, and the split itself is a
deterministic function of the recorded public seed.  Anyone holding this script
and the harvest can recompute the confirmation split without unsealing.  What
the seal buys is that reading it is a deliberate, recorded act rather than an
accident of a directory walk -- which is the discipline the confirmation arm
actually needs.  It should not be described as any stronger than that.

Preprocessing provenance
------------------------
``adjust_image_size`` below is a verbatim port of the pinned validator's
``validator/evaluation/image_io.py:16-39`` at
``gradients-ai/G.O.D @ b026da04b6179cf82945e8736590dd923114342b``
(file SHA-256 ``ae1b1a506ddba77208fe7f6264f41af919b721b75c464a4710edb4804f0ad286``,
read directly from a checkout of that commit).  In the pinned tree it is called
exactly once -- ``validator/evaluation/evaluators/diffusion.py:302``, on each
held-out image immediately before scoring -- and never on training data.

Two details there are easy to get wrong and are recorded in every manifest:
the validator floors dimensions to a multiple of **16**, not 8; and it resamples
with **LANCZOS**, whereas ai-toolkit's dataloader resamples with BICUBIC.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import itertools
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

BUILDER_VERSION = "1.0.0"
MANIFEST_SCHEMA = "sn56.week6.fixture-manifest"
MANIFEST_SCHEMA_VERSION = 1

DEFAULT_SEED = 20260806
DEFAULT_HARVEST_ROOT = Path(
    "/Users/atulyashetty/Test/SN56-project/evidence/"
    "week6-tournament-dataset-harvest-20260806"
)
DEFAULT_OUT_ROOT = DEFAULT_HARVEST_ROOT.parent / "week6-fixtures-20260806"

# Split policy.  A holdout of 8-10 is wide enough to separate arms; week-5's
# 2-3 image holdout was not.  TRAIN_MIN is the floor below which the training
# side stops being a workable LoRA dataset.
HOLDOUT_TARGET = 10
HOLDOUT_MIN = 8
TRAIN_MIN = 12

# Near-duplicate screen.  Both criteria must fire.  dHash alone over-groups
# this corpus: synthetic UI mockups share gross layout, so pairs at Hamming
# distance 2-6 routinely differ by a pixel MSE of 0.05-0.22, which is two
# orders of magnitude above anything that could be called a duplicate.
DHASH_BITS = 64
DHASH_THRESHOLD = 6
DUP_MSE_NORM_SIDE = 256
DUP_MSE_THRESHOLD = 0.005

VALIDATOR_PIN = {
    "repo": "gradients-ai/G.O.D",
    "commit": "b026da04b6179cf82945e8736590dd923114342b",
    "file": "validator/evaluation/image_io.py",
    "file_sha256": "ae1b1a506ddba77208fe7f6264f41af919b721b75c464a4710edb4804f0ad286",
    "function": "adjust_image_size",
    "lines": "16-39",
    "call_sites": ["validator/evaluation/evaluators/diffusion.py:302"],
    "call_site_count_in_pinned_tree": 1,
    "loss": "validator/evaluation/evaluators/diffusion.py:203-209 "
    "(convert RGB, /255.0, mean squared error; raises on shape mismatch)",
}

SEAL_MAGIC = b"SN56SFAR1\n"
SEAL_SCHEMA = "sn56.week6.fixture-seal"
INDEX_SCHEMA = "sn56.week6.fixture-index"

PRIMARY_TASKS = {
    "discovery": [
        "41025fb5-8473-40c6-a88d-20c0bb303edc",
        "db9f7244-b3c8-4f85-96d9-7184af8b4179",
    ],
    "confirmation": [
        "3e0fdcde-3dcc-4fb8-b255-a6d778e61cbb",
        "f6725c2b-7e31-493c-8830-31b20fd5db78",
    ],
}


class FixtureError(RuntimeError):
    """A fail-closed builder refusal."""


class SealedFixtureError(FixtureError):
    """The confirmation group was read before it was unsealed."""


# --------------------------------------------------------------------------
# validator preprocessing -- verbatim port, see module docstring
# --------------------------------------------------------------------------


def adjust_image_size(image: Image.Image) -> Image.Image:
    """Port of the pinned validator's ``adjust_image_size``.

    Kept structurally identical to upstream, including the ``int()``
    truncation, the divisor of 16, and the integer-floor crop offsets.  Do not
    "clean this up" -- any deviation silently decouples our fixtures from what
    the scorer actually sees.
    """
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


def describe_adjustment(size: tuple[int, int]) -> dict:
    """Reproduce the intermediate geometry of ``adjust_image_size`` for the record."""
    width, height = size
    if width > height:
        scaled_w = 1024
        scaled_h = int((height / width) * 1024)
    else:
        scaled_h = 1024
        scaled_w = int((width / height) * 1024)
    floor_w = (scaled_w // 16) * 16
    floor_h = (scaled_h // 16) * 16
    crop_w = min(scaled_w, floor_w)
    crop_h = min(scaled_h, floor_h)
    left = (scaled_w - crop_w) // 2
    top = (scaled_h - crop_h) // 2
    return {
        "source": [width, height],
        "lanczos_resize_to": [scaled_w, scaled_h],
        "floor16": [floor_w, floor_h],
        "crop_box": [left, top, left + crop_w, top + crop_h],
        "output": [crop_w, crop_h],
        "rows_cropped": scaled_h - crop_h,
        "cols_cropped": scaled_w - crop_w,
        "geometry_is_identity": [crop_w, crop_h] == [width, height],
    }


def preprocess_to_png_bytes(path: Path) -> tuple[bytes, dict]:
    """Apply the validator transform and encode the way the validator would.

    ``image_to_base64`` saves with ``image.format if image.format else "PNG"``.
    ``resize``/``crop`` return images whose ``format`` is ``None``, so the
    reference the scorer compares against is always a lossless PNG.  We write
    that same PNG.
    """
    with Image.open(path) as src:
        src.load()
        source_size = src.size
        source_mode = src.mode
        source_format = src.format
        adjusted = adjust_image_size(src)
        pixels_identical = adjusted.size == source_size and np.array_equal(
            np.asarray(adjusted), np.asarray(src)
        )
        buffer = io.BytesIO()
        # format is None after resize/crop -> the validator's PNG fallback.
        adjusted.save(buffer, format=adjusted.format if adjusted.format else "PNG")
        payload = buffer.getvalue()
        record = describe_adjustment(source_size)
        record.update(
            {
                "source_mode": source_mode,
                "source_format": source_format,
                "output_mode": adjusted.mode,
                "output_format_used": adjusted.format if adjusted.format else "PNG",
                "mode_converted": adjusted.mode != source_mode,
                "pixels_identical_to_source": bool(pixels_identical),
            }
        )
    return payload, record


# --------------------------------------------------------------------------
# hashing / duplicate screen
# --------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj) -> bytes:
    """Byte-stable JSON.  Every manifest hash in this tree is over this form."""
    return (
        json.dumps(obj, sort_keys=True, ensure_ascii=True, indent=2, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")


def dhash64(path: Path) -> int:
    """64-bit difference hash over a 9x8 grayscale reduction (LANCZOS)."""
    with Image.open(path) as image:
        reduced = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        array = np.asarray(reduced, dtype=np.int16)
    bits = (array[:, 1:] > array[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def normalized_pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize(
            (DUP_MSE_NORM_SIDE, DUP_MSE_NORM_SIDE), Image.Resampling.LANCZOS
        )
        return np.asarray(resized, dtype=np.float64) / 255.0


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def screen_duplicates(stems: Sequence[str], image_paths: Sequence[Path], shas: Sequence[str]):
    """Group images that must not be split across train/holdout.

    Rule: identical SHA-256, or (dHash Hamming <= threshold AND normalized
    pixel MSE <= threshold).  Both near-duplicate criteria must fire; see the
    constant block for why dHash alone is not sufficient on this corpus.
    """
    n = len(stems)
    hashes = [dhash64(p) for p in image_paths]
    pixels = [normalized_pixels(p) for p in image_paths]

    uf = UnionFind(n)
    exact_pairs: list[dict] = []
    near_pairs: list[dict] = []
    all_pairs: list[dict] = []

    for i, j in itertools.combinations(range(n), 2):
        distance = hamming(hashes[i], hashes[j])
        mse = float(np.mean((pixels[i] - pixels[j]) ** 2))
        record = {
            "a": stems[i],
            "b": stems[j],
            "dhash_hamming": distance,
            "norm_mse": round(mse, 6),
        }
        all_pairs.append(record)
        if shas[i] == shas[j]:
            exact_pairs.append(record)
            uf.union(i, j)
        elif distance <= DHASH_THRESHOLD and mse <= DUP_MSE_THRESHOLD:
            near_pairs.append(record)
            uf.union(i, j)

    buckets: dict[int, list[str]] = {}
    for index, stem in enumerate(stems):
        buckets.setdefault(uf.find(index), []).append(stem)
    groups = sorted((sorted(members) for members in buckets.values()), key=lambda m: m[0])

    closest_by_mse = sorted(all_pairs, key=lambda r: (r["norm_mse"], r["a"], r["b"]))[:5]
    closest_by_dhash = sorted(
        all_pairs, key=lambda r: (r["dhash_hamming"], r["norm_mse"], r["a"], r["b"])
    )[:5]

    report = {
        "dhash": {
            "bits": DHASH_BITS,
            "reduction": "9x8 grayscale, PIL LANCZOS",
            "threshold_hamming": DHASH_THRESHOLD,
        },
        "pixel": {
            "normalization": f"{DUP_MSE_NORM_SIDE}x{DUP_MSE_NORM_SIDE} RGB LANCZOS, /255.0",
            "metric": "mean squared error",
            "threshold": DUP_MSE_THRESHOLD,
        },
        "rule": (
            "identical sha256 OR (dhash_hamming <= "
            f"{DHASH_THRESHOLD} AND norm_mse <= {DUP_MSE_THRESHOLD}); "
            "members of a group are always placed on the same side"
        ),
        "n_images": n,
        "n_groups": len(groups),
        "n_multi_member_groups": sum(1 for g in groups if len(g) > 1),
        "exact_duplicate_pairs": exact_pairs,
        "near_duplicate_pairs": near_pairs,
        "duplicate_groups": [g for g in groups if len(g) > 1],
        "closest_pairs_by_norm_mse": closest_by_mse,
        "closest_pairs_by_dhash": closest_by_dhash,
    }
    return groups, report


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------


def plan_split(n_pairs: int) -> tuple[int, str | None]:
    """Choose the holdout size.  Returns ``(holdout_n, downgrade_note)``."""
    for holdout in range(HOLDOUT_TARGET, HOLDOUT_MIN - 1, -1):
        if n_pairs - holdout >= TRAIN_MIN:
            if holdout == HOLDOUT_TARGET:
                return holdout, None
            note = (
                f"{n_pairs} pairs cannot support the {HOLDOUT_TARGET}-image holdout "
                f"target while leaving the {TRAIN_MIN}-row training minimum "
                f"({n_pairs} - {HOLDOUT_TARGET} = {n_pairs - HOLDOUT_TARGET}). "
                f"Downgraded to the largest holdout that leaves >= {TRAIN_MIN} train rows: "
                f"holdout={holdout}, train={n_pairs - holdout}."
            )
            return holdout, note
    raise FixtureError(
        f"task has {n_pairs} pairs; no holdout in [{HOLDOUT_MIN}, {HOLDOUT_TARGET}] "
        f"leaves the {TRAIN_MIN}-row training minimum"
    )


def group_rank_key(seed: int, task_id: str, group: Sequence[str]) -> str:
    material = f"{seed}|{task_id}|{BUILDER_VERSION}|{'+'.join(group)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def assign_split(seed: int, task_id: str, groups: Sequence[Sequence[str]], holdout_n: int):
    """Deterministically fill the holdout from rank-ordered duplicate groups."""
    ranked = sorted(groups, key=lambda g: (group_rank_key(seed, task_id, g), "+".join(g)))
    holdout: list[str] = []
    train: list[str] = []
    for group in ranked:
        if len(holdout) + len(group) <= holdout_n:
            holdout.extend(group)
        else:
            train.extend(group)
    holdout.sort()
    train.sort()
    if len(holdout) < HOLDOUT_MIN:
        raise FixtureError(
            f"duplicate grouping left only {len(holdout)} holdout images "
            f"(minimum {HOLDOUT_MIN}) for task {task_id}"
        )
    if len(train) < TRAIN_MIN:
        raise FixtureError(
            f"duplicate grouping left only {len(train)} train rows "
            f"(minimum {TRAIN_MIN}) for task {task_id}"
        )
    ordering = [
        {"group": list(g), "rank_key": group_rank_key(seed, task_id, g)} for g in ranked
    ]
    return holdout, train, ordering


# --------------------------------------------------------------------------
# harvest reading (read-only)
# --------------------------------------------------------------------------

_TRIGGER_FIELDS = (
    ("trigger_word", lambda v: v if isinstance(v, str) else None),
    ("trigger", lambda v: v.get("declared_trigger_word") if isinstance(v, dict) else None),
    (
        "trigger_phrase",
        lambda v: v.get("audit_trigger_word") if isinstance(v, dict) else None,
    ),
)


def read_task_meta(task_dir: Path) -> dict:
    """Read ``task-meta.json`` tolerantly.

    The harvest was written by several agents and the three tasks we use carry
    three different meta schemas, so every field is looked up by candidate list
    rather than by a fixed path.  Image geometry is deliberately *not* taken
    from here -- it is measured from the files.
    """
    meta_path = task_dir / "task-meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    trigger = None
    trigger_field = None
    for field, extract in _TRIGGER_FIELDS:
        if field in meta:
            value = extract(meta[field])
            if value:
                trigger = value
                trigger_field = field
                break

    n_pairs = meta.get("n_pairs")
    if n_pairs is None:
        n_pairs = meta.get("n_pairs_expected")

    sums_path = task_dir / "SHA256SUMS"
    return {
        "task_id": meta.get("task_id"),
        "round": meta.get("round"),
        "tournament_id": meta.get("tournament_id"),
        "status": meta.get("status"),
        "task_type": meta.get("task_type"),
        "is_organic": meta.get("is_organic"),
        "model_type": meta.get("model_type"),
        "base_model": meta.get("base_model"),
        "ds": meta.get("ds"),
        "ds_short": meta.get("ds_short"),
        "family": meta.get("family"),
        "boss_round": meta.get("boss_round"),
        "n_pairs_declared": n_pairs,
        "hours_to_complete": meta.get("hours_to_complete"),
        "winner_hotkey": meta.get("winner_hotkey"),
        "trigger_phrase": trigger,
        "trigger_phrase_meta_field": trigger_field,
        "task_meta_sha256": sha256_file(meta_path),
        "sha256sums_sha256": sha256_file(sums_path) if sums_path.exists() else None,
    }


def read_harvest_sums(task_dir: Path) -> dict[str, str]:
    sums_path = task_dir / "SHA256SUMS"
    if not sums_path.exists():
        return {}
    table: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition("  ")
        if digest and name:
            table[name.strip()] = digest.strip()
    return table


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def list_pairs(task_dir: Path) -> list[tuple[str, Path, Path]]:
    pairs_dir = task_dir / "pairs"
    if not pairs_dir.is_dir():
        raise FixtureError(f"no pairs/ directory under {task_dir}")
    found: dict[str, Path] = {}
    for entry in sorted(pairs_dir.iterdir()):
        if entry.suffix.lower() in IMAGE_EXTENSIONS and entry.is_file():
            if entry.stem in found:
                raise FixtureError(f"two images share the stem {entry.stem} in {pairs_dir}")
            found[entry.stem] = entry
    rows = []
    for stem in sorted(found):
        caption = pairs_dir / f"{stem}.txt"
        if not caption.is_file():
            raise FixtureError(f"image {found[stem]} has no caption {caption}")
        rows.append((stem, found[stem], caption))
    if not rows:
        raise FixtureError(f"no image/caption pairs under {pairs_dir}")
    return rows


# --------------------------------------------------------------------------
# per-task build
# --------------------------------------------------------------------------


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FixtureError(f"refusing to overwrite {path}")
    path.write_bytes(payload)


def build_task(
    harvest_root: Path,
    task_id: str,
    group: str,
    seed: int,
    out_dir: Path,
    builder_sha256: str,
) -> dict:
    task_dir = harvest_root / "tasks" / task_id
    if not task_dir.is_dir():
        raise FixtureError(f"task {task_id} not found under {harvest_root / 'tasks'}")

    meta = read_task_meta(task_dir)
    declared_sums = read_harvest_sums(task_dir)
    rows = list_pairs(task_dir)
    stems = [r[0] for r in rows]
    image_paths = [r[1] for r in rows]
    caption_paths = [r[2] for r in rows]

    shas = [sha256_file(p) for p in image_paths]
    caption_shas = [sha256_file(p) for p in caption_paths]

    sums_mismatches = []
    for stem, path, digest in zip(stems, image_paths, shas):
        declared = declared_sums.get(f"pairs/{path.name}")
        if declared is not None and declared != digest:
            sums_mismatches.append({"file": f"pairs/{path.name}", "declared": declared, "measured": digest})
    if sums_mismatches:
        raise FixtureError(f"harvest SHA256SUMS mismatch for {task_id}: {sums_mismatches}")

    groups, dedup_report = screen_duplicates(stems, image_paths, shas)
    holdout_n, downgrade_note = plan_split(len(stems))
    holdout, train, ordering = assign_split(seed, task_id, groups, holdout_n)

    holdout_set = set(holdout)
    train_set = set(train)
    overlap = holdout_set & train_set
    if overlap:
        raise FixtureError(f"train/holdout overlap in {task_id}: {sorted(overlap)}")
    if holdout_set | train_set != set(stems):
        raise FixtureError(f"split does not cover every pair in {task_id}")

    group_of = {}
    for index, members in enumerate(groups):
        for stem in members:
            group_of[stem] = index
    for members in groups:
        sides = {("holdout" if s in holdout_set else "train") for s in members}
        if len(sides) != 1:
            raise FixtureError(f"duplicate group split across sides in {task_id}: {members}")

    images: list[dict] = []
    trigger = meta["trigger_phrase"]
    trigger_hits = 0

    for stem, image_path, caption_path, image_sha, caption_sha in zip(
        stems, image_paths, caption_paths, shas, caption_shas
    ):
        side = "holdout" if stem in holdout_set else "train"
        caption_bytes = caption_path.read_bytes()
        caption_text = caption_bytes.decode("utf-8")
        if trigger and trigger in caption_text:
            trigger_hits += 1

        with Image.open(image_path) as probe:
            width, height = probe.size
            source_format = probe.format
            source_mode = probe.mode

        adjusted_png, adjust_record = preprocess_to_png_bytes(image_path)

        source_image_rel = f"{side}/source/{image_path.name}"
        source_caption_rel = f"{side}/source/{caption_path.name}"
        prep_image_rel = f"{side}/validator_geometry/{stem}.png"
        prep_caption_rel = f"{side}/validator_geometry/{caption_path.name}"

        _write(out_dir / source_image_rel, image_path.read_bytes())
        _write(out_dir / source_caption_rel, caption_bytes)
        _write(out_dir / prep_image_rel, adjusted_png)
        _write(out_dir / prep_caption_rel, caption_bytes)

        images.append(
            {
                "stem": stem,
                "split": side,
                "duplicate_group": group_of[stem],
                "source": {
                    "harvest_path": f"tasks/{task_id}/pairs/{image_path.name}",
                    "sha256": image_sha,
                    "bytes": image_path.stat().st_size,
                    "width": width,
                    "height": height,
                    "format": source_format,
                    "mode": source_mode,
                    "divisible_by_8": (width % 8 == 0 and height % 8 == 0),
                    "divisible_by_16": (width % 16 == 0 and height % 16 == 0),
                },
                "caption": {
                    "harvest_path": f"tasks/{task_id}/pairs/{caption_path.name}",
                    "sha256": caption_sha,
                    "bytes": len(caption_bytes),
                    "chars": len(caption_text),
                    "contains_trigger_phrase": bool(trigger and trigger in caption_text),
                },
                "preprocessed": {
                    **adjust_record,
                    "sha256": sha256_bytes(adjusted_png),
                    "bytes": len(adjusted_png),
                },
                "outputs": {
                    "source_image": source_image_rel,
                    "source_caption": source_caption_rel,
                    "validator_geometry_image": prep_image_rel,
                    "validator_geometry_caption": prep_caption_rel,
                },
            }
        )

    resolution_before: dict[str, int] = {}
    resolution_after: dict[str, int] = {}
    for record in images:
        before = f"{record['source']['width']}x{record['source']['height']}"
        after = "x".join(str(v) for v in record["preprocessed"]["output"])
        resolution_before[before] = resolution_before.get(before, 0) + 1
        resolution_after[after] = resolution_after.get(after, 0) + 1

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "builder_source_sha256": builder_sha256,
        "environment": {
            "pillow": Image.__version__,
            "numpy": np.__version__,
            "note": "recorded because LANCZOS resampling and PNG encoding are "
            "library-version dependent; byte-identity claims hold within a version",
        },
        "group": group,
        "seed": seed,
        "task": meta,
        "harvest": {
            "root": str(harvest_root),
            "task_dir": f"tasks/{task_id}",
            "access": "read-only",
            "n_pairs_measured": len(stems),
            "sha256sums_cross_checked": bool(declared_sums),
            "sha256sums_entries_matched": sum(
                1 for p in image_paths if f"pairs/{p.name}" in declared_sums
            ),
        },
        "trigger_verification": {
            "phrase": trigger,
            "meta_field": meta["trigger_phrase_meta_field"],
            "captions_containing_phrase_verbatim": trigger_hits,
            "captions_total": len(stems),
        },
        "preprocessing": {
            "transform": "validator.adjust_image_size",
            "pin": VALIDATOR_PIN,
            "steps": [
                "isotropic scale so the LONG edge is exactly 1024; the short edge "
                "uses int() truncation",
                "PIL LANCZOS resampling",
                "floor BOTH dimensions to a multiple of 16",
                "center crop with integer-floor offsets (bias up to 1px toward top/left)",
            ],
            "divisibility": 16,
            "divisibility_note": (
                "The validator floors to a multiple of 16, not 8 "
                "(image_io.py:28-29). Sources that are not divisible by 8 are not "
                "special-cased anywhere upstream: they go through the same "
                "long-edge scale then floor-16 crop, which is what this builder "
                "reproduces. No source in this task required a different path."
            ),
            "resample": "PIL.Image.Resampling.LANCZOS",
            "output_encoding": (
                "PNG, lossless. image_to_base64 saves with image.format or PNG; "
                "resize/crop clear Image.format, so the scored reference is always PNG."
            ),
            "mode_conversion": (
                "none inside adjust_image_size; calculate_l2_loss converts to RGB "
                "at scoring time"
            ),
            "applied_to": {
                "holdout/validator_geometry": True,
                "train/validator_geometry": True,
                "holdout/source": False,
                "train/source": False,
            },
            "field_fidelity_note": (
                "In the pinned validator this transform runs on held-out images only "
                "and never on training data (the entrypoint copies training files "
                "verbatim). train/source is therefore the field-faithful training "
                "input; train/validator_geometry is the geometry-aligned arm. Both "
                "are emitted so the arms are explicit; neither is a silent "
                "substitution."
            ),
        },
        "dedup": dedup_report,
        "split": {
            "policy": {
                "holdout_target": HOLDOUT_TARGET,
                "holdout_min": HOLDOUT_MIN,
                "train_min": TRAIN_MIN,
            },
            "holdout_n": len(holdout),
            "train_n": len(train),
            "holdout_downgraded": downgrade_note is not None,
            "holdout_downgrade_note": downgrade_note,
            "determinism": (
                "duplicate groups ranked ascending by "
                "sha256('<seed>|<task_id>|<builder_version>|<stems joined by +>'), "
                "then filled into the holdout in rank order; no RNG, no clock"
            ),
            "holdout": holdout,
            "train": train,
            "group_ranking": ordering,
        },
        "resolution_distribution": {
            "before_preprocessing": resolution_before,
            "after_preprocessing": resolution_after,
        },
        "images": images,
    }

    manifest_bytes = canonical_json(manifest)
    manifest_sha = sha256_bytes(manifest_bytes)
    _write(out_dir / "manifest.json", manifest_bytes)
    _write(out_dir / "manifest.sha256", f"{manifest_sha}  manifest.json\n".encode("utf-8"))

    return {
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "summary": {
            "task_id": task_id,
            "round": meta["round"],
            "model_type": meta["model_type"],
            "ds_short": meta["ds_short"],
            "family": meta["family"],
            "trigger_phrase": trigger,
            "n_pairs": len(stems),
            "holdout_n": len(holdout),
            "train_n": len(train),
            "holdout_downgraded": downgrade_note is not None,
            "duplicate_groups": dedup_report["n_multi_member_groups"],
            "resolutions_before": resolution_before,
            "resolutions_after": resolution_after,
            "manifest_sha256": manifest_sha,
        },
    }


# --------------------------------------------------------------------------
# sealed container
# --------------------------------------------------------------------------


def keystream(key: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(hashlib.sha256(key + struct.pack(">Q", counter)).digest())
        counter += 1
    return bytes(out[:length])


def pack_container(root: Path) -> bytes:
    """Deterministic archive of ``root``: sorted index + concatenated payloads.

    A hand-rolled format rather than tar, because tar carries mtime/uid/gid and
    format-dependent padding that would leak nondeterminism into the seal.
    """
    entries = []
    blobs = []
    offset = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        payload = path.read_bytes()
        rel = path.relative_to(root).as_posix()
        entries.append(
            {"path": rel, "offset": offset, "size": len(payload), "sha256": sha256_bytes(payload)}
        )
        blobs.append(payload)
        offset += len(payload)
    header = canonical_json({"format": "SN56SFAR1", "entries": entries})
    return SEAL_MAGIC + struct.pack(">Q", len(header)) + header + b"".join(blobs)


def unpack_container(blob: bytes, dest: Path) -> list[dict]:
    if not blob.startswith(SEAL_MAGIC):
        raise FixtureError(
            "container did not decode to a valid archive -- the supplied "
            "commitment is wrong"
        )
    cursor = len(SEAL_MAGIC)
    (header_len,) = struct.unpack(">Q", blob[cursor : cursor + 8])
    cursor += 8
    header = json.loads(blob[cursor : cursor + header_len].decode("utf-8"))
    body = blob[cursor + header_len :]
    written = []
    for entry in header["entries"]:
        payload = body[entry["offset"] : entry["offset"] + entry["size"]]
        if sha256_bytes(payload) != entry["sha256"]:
            raise FixtureError(f"payload hash mismatch for {entry['path']}")
        target = dest / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        written.append(entry)
    return written


def seal_directory(staged: Path, commitment: str, out_root: Path) -> dict:
    container = pack_container(staged)
    stream = keystream(commitment.encode("ascii"), len(container))
    sealed = bytes(a ^ b for a, b in zip(container, stream))
    sealed_dir = out_root / "SEALED"
    _write(sealed_dir / "confirmation.sealed", sealed)
    return {
        "container_sha256": sha256_bytes(sealed),
        "container_bytes": len(sealed),
        "plaintext_sha256": sha256_bytes(container),
    }


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def _require_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FixtureError(
            f"refusing to overwrite existing fixtures at {path}; "
            "a split that has been built must never be re-rolled -- "
            "use `verify` to check reproducibility, or choose a new --out-root"
        )


def build(harvest_root: Path, out_root: Path, seed: int, builder_path: Path) -> dict:
    if not (harvest_root / "tasks").is_dir():
        raise FixtureError(f"{harvest_root} does not look like the harvest store")
    _require_empty(out_root)

    builder_sha = sha256_file(builder_path)
    out_root.mkdir(parents=True, exist_ok=True)

    discovery_summaries = []
    for task_id in PRIMARY_TASKS["discovery"]:
        result = build_task(
            harvest_root,
            task_id,
            "discovery",
            seed,
            out_root / "discovery" / task_id,
            builder_sha,
        )
        summary = dict(result["summary"])
        summary["dir"] = f"discovery/{task_id}"
        discovery_summaries.append(summary)

    with tempfile.TemporaryDirectory(prefix="week6-seal-") as staging:
        staged = Path(staging) / "confirmation"
        confirmation_manifests = []
        confirmation_summaries = []
        for task_id in PRIMARY_TASKS["confirmation"]:
            result = build_task(
                harvest_root, task_id, "confirmation", seed, staged / task_id, builder_sha
            )
            confirmation_manifests.append(result["manifest"])
            confirmation_summaries.append(result["summary"])

        combined = {
            "schema": "sn56.week6.confirmation-manifest",
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "builder_version": BUILDER_VERSION,
            "builder_source_sha256": builder_sha,
            "seed": seed,
            "tasks": confirmation_manifests,
        }
        combined_bytes = canonical_json(combined)
        commitment = sha256_bytes(combined_bytes)
        (staged / "confirmation-manifest.json").write_bytes(combined_bytes)

        seal_info = seal_directory(staged, commitment, out_root)

    seal = {
        "schema": SEAL_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "builder_source_sha256": builder_sha,
        "seed": seed,
        "commitment_sha256": commitment,
        "commitment_definition": (
            "sha256 of the canonical JSON bytes of confirmation-manifest.json, "
            "which embeds the full per-task manifest of both confirmation tasks"
        ),
        "container_file": "SEALED/confirmation.sealed",
        "container_sha256": seal_info["container_sha256"],
        "container_bytes": seal_info["container_bytes"],
        "container_format": (
            "SN56SFAR1 (sorted index + concatenated payloads) XORed with "
            "SHA256(commitment_ascii || uint64_be counter)"
        ),
        "sealed_tasks": [
            {
                "task_id": s["task_id"],
                "round": s["round"],
                "model_type": s["model_type"],
                "ds_short": s["ds_short"],
                "n_pairs": s["n_pairs"],
                "holdout_n": s["holdout_n"],
                "train_n": s["train_n"],
            }
            for s in confirmation_summaries
        ],
        "unseal_command": (
            "python3 ops/experiments/week6/build_fixtures.py unseal "
            "--out-root <root> --commitment <commitment_sha256>"
        ),
        "read_barrier_note": (
            "Confirmation images and per-image split assignments are not present "
            "on disk in readable form until `unseal` runs, and unsealing writes a "
            "dated receipt. This is a procedural barrier, not secrecy: the "
            "commitment is published here and is also the container key, and the "
            "split is a deterministic function of the recorded public seed, so a "
            "determined reader holding this builder can recompute it. Its purpose "
            "is to make reading the confirmation set a deliberate, logged act."
        ),
    }
    _write(out_root / "SEALED" / "SEAL.json", canonical_json(seal))

    index = {
        "schema": INDEX_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "builder_source_sha256": builder_sha,
        "seed": seed,
        "harvest_root": str(harvest_root),
        "harvest_access": "read-only; only tasks/<id>/{pairs,task-meta.json,SHA256SUMS} were read",
        "quarantine_note": "QUARANTINE-test-data was never opened, listed, or referenced",
        "split_policy": {
            "holdout_target": HOLDOUT_TARGET,
            "holdout_min": HOLDOUT_MIN,
            "train_min": TRAIN_MIN,
        },
        "discovery": discovery_summaries,
        "confirmation": {
            "sealed": True,
            "commitment_sha256": commitment,
            "seal_file": "SEALED/SEAL.json",
            "container_file": "SEALED/confirmation.sealed",
            "task_ids": list(PRIMARY_TASKS["confirmation"]),
        },
    }
    _write(out_root / "FIXTURES-INDEX.json", canonical_json(index))
    _write(out_root / "README.md", _readme(out_root, commitment).encode("utf-8"))

    return {
        "out_root": str(out_root),
        "discovery": discovery_summaries,
        "confirmation_commitment_sha256": commitment,
        "confirmation_sealed_tasks": seal["sealed_tasks"],
        "container_sha256": seal_info["container_sha256"],
        "container_bytes": seal_info["container_bytes"],
    }


def _readme(out_root: Path, commitment: str) -> str:
    return f"""# Week-6 real-data fixtures ({BUILDER_VERSION})

Built by `ops/experiments/week6/build_fixtures.py` on branch
`claude/week6-real-fixture-experiment` from the Aug-3 tournament harvest.
The harvest store was read only; nothing in it was created, modified, or removed,
and `QUARANTINE-test-data` was never opened.

## Layout

- `FIXTURES-INDEX.json` - every discovery group, sizes, manifest hashes.
- `discovery/<task_id>/` - readable now.
  - `manifest.json` / `manifest.sha256`
  - `train/source/`, `holdout/source/` - byte-identical copies of the harvested pairs.
  - `train/validator_geometry/`, `holdout/validator_geometry/` - the same pairs after
    the pinned validator's `adjust_image_size` (long edge to 1024, LANCZOS, floor to a
    multiple of **16**, center crop), saved as lossless PNG.
- `SEALED/` - the confirmation group, packed into one opaque container.
- `confirmation/` - does not exist until `unseal` runs.

`holdout/validator_geometry` is what the scorer actually compares against.
`train/source` is what the tournament's training container actually received
(the entrypoint copies training files verbatim; the validator transform is
applied to held-out images only). `train/validator_geometry` exists so the
geometry-alignment arm is explicit rather than a silent resize.

## Confirmation commitment

    {commitment}

To read the confirmation set:

    python3 ops/experiments/week6/build_fixtures.py unseal \\
        --out-root {out_root} \\
        --commitment {commitment}

This is a procedural barrier, not secrecy. The commitment is published above and
is also the container key, and the split is a deterministic function of the
recorded public seed, so anyone with this builder can recompute it. The point is
that reading the confirmation set is a deliberate act that leaves a receipt.

## Reproducibility

    python3 ops/experiments/week6/build_fixtures.py verify --out-root {out_root}

rebuilds into a scratch directory and compares every byte. The builder refuses to
write into a non-empty output root, so a split cannot be re-rolled in place.
Split assignment uses no RNG and no wall clock: duplicate groups are ranked by
`sha256("<seed>|<task_id>|<builder_version>|<stems>")`.
"""


def unseal(out_root: Path, commitment: str) -> dict:
    seal_path = out_root / "SEALED" / "SEAL.json"
    if not seal_path.exists():
        raise FixtureError(f"no seal at {seal_path}")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))

    dest = out_root / "confirmation"
    if dest.exists():
        raise FixtureError(f"{dest} already exists; refusing to unseal twice")

    container_path = out_root / "SEALED" / "confirmation.sealed"
    sealed = container_path.read_bytes()
    measured = sha256_bytes(sealed)
    if measured != seal["container_sha256"]:
        raise FixtureError(
            f"sealed container has been modified: expected {seal['container_sha256']}, "
            f"measured {measured}"
        )

    commitment = commitment.strip().lower()
    stream = keystream(commitment.encode("ascii"), len(sealed))
    plain = bytes(a ^ b for a, b in zip(sealed, stream))

    staging = Path(tempfile.mkdtemp(prefix="week6-unseal-"))
    try:
        entries = unpack_container(plain, staging)
        combined = (staging / "confirmation-manifest.json").read_bytes()
        recomputed = sha256_bytes(combined)
        if recomputed != commitment:
            raise FixtureError(
                "confirmation manifest does not match the commitment "
                f"(manifest hashes to {recomputed})"
            )
        if recomputed != seal["commitment_sha256"]:
            raise FixtureError("commitment in SEAL.json disagrees with the sealed manifest")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    receipt = {
        "schema": "sn56.week6.unseal-receipt",
        "commitment_sha256": commitment,
        "container_sha256": measured,
        "files_written": len(entries),
        "unsealed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "note": "wall-clock appears only in this receipt; it is an event log and is "
        "excluded from every manifest and from the determinism check",
    }
    (dest / "UNSEAL-RECEIPT.json").write_bytes(canonical_json(receipt))
    return {
        "dest": str(dest),
        "files_written": len(entries),
        "commitment_sha256": commitment,
    }


def load_group(out_root: Path, group: str) -> list[Path]:
    """Fixture accessor for the experiment runner.

    Raises ``SealedFixtureError`` if the confirmation group has not been
    unsealed, so a runner cannot read it by accident.
    """
    if group == "discovery":
        base = out_root / "discovery"
        if not base.is_dir():
            raise FixtureError(f"no discovery fixtures at {base}")
        return sorted(p for p in base.iterdir() if p.is_dir())
    if group == "confirmation":
        base = out_root / "confirmation"
        if not base.is_dir():
            raise SealedFixtureError(
                "the confirmation group is sealed; run "
                "`build_fixtures.py unseal --out-root <root> --commitment <sha256>` "
                "before reading it"
            )
        return sorted(p for p in base.iterdir() if p.is_dir())
    raise FixtureError(f"unknown group {group!r}")


def _tree_bytes(root: Path, skip: Iterable[str] = ()) -> dict[str, str]:
    skipped = set(skip)
    table = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel in skipped:
            continue
        table[rel] = sha256_file(path)
    return table


def verify(out_root: Path, harvest_root: Path, seed: int, builder_path: Path) -> dict:
    """Rebuild into scratch and compare every byte against the existing tree."""
    with tempfile.TemporaryDirectory(prefix="week6-verify-") as scratch:
        replica_root = Path(scratch) / "week6-fixtures"
        build(harvest_root, replica_root, seed, builder_path)
        # README embeds the output path by design; compare it separately.
        original = _tree_bytes(out_root, skip={"README.md"})
        replica = _tree_bytes(replica_root, skip={"README.md"})

    only_original = sorted(set(original) - set(replica))
    only_replica = sorted(set(replica) - set(original))
    differing = sorted(k for k in set(original) & set(replica) if original[k] != replica[k])
    return {
        "files_compared": len(set(original) & set(replica)),
        "identical": not (only_original or only_replica or differing),
        "missing_in_replica": only_original,
        "unexpected_in_replica": only_replica,
        "differing": differing,
        "excluded": ["README.md (embeds the absolute output path)"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["build", "unseal", "verify", "status"])
    parser.add_argument("--harvest-root", type=Path, default=DEFAULT_HARVEST_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--commitment", default=None)
    args = parser.parse_args(argv)

    builder_path = Path(__file__).resolve()

    try:
        if args.command == "build":
            result = build(args.harvest_root, args.out_root, args.seed, builder_path)
        elif args.command == "unseal":
            if not args.commitment:
                raise FixtureError("unseal requires --commitment")
            result = unseal(args.out_root, args.commitment)
        elif args.command == "verify":
            result = verify(args.out_root, args.harvest_root, args.seed, builder_path)
        else:
            index_path = args.out_root / "FIXTURES-INDEX.json"
            result = {
                "out_root": str(args.out_root),
                "built": index_path.exists(),
                "confirmation_unsealed": (args.out_root / "confirmation").is_dir(),
            }
            if index_path.exists():
                result["index"] = json.loads(index_path.read_text(encoding="utf-8"))
    except FixtureError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
