"""Canonical Krea exact-evaluator dataset identity (CPU-only)."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable


_SHA256_HEX = __import__("re").compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _real_directory(path: Path) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"dataset path has a symlink ancestor: {current}")
        current = current.parent
    if not path.is_dir():
        raise ValueError(f"dataset path is not a real directory: {path}")
    return path


def _read_stable_regular(path: Path, label: str) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"{label} is not a safe regular file: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not regular: {path}")
        chunks = []
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        identity = lambda item: (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
        )
        if identity(before) != identity(after):
            raise RuntimeError(f"{label} changed while read: {path}")
        return b"".join(chunks), after
    finally:
        os.close(descriptor)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def capture_dataset(
    dataset: Path,
    *,
    list_supported_images: Callable[[str, tuple[str, ...]], list[str]],
    extensions: tuple[str, ...],
) -> dict[str, Any]:
    """Capture the byte/dimension/order identity consumed by exact scoring."""

    dataset = _real_directory(dataset)
    if (
        not isinstance(extensions, tuple)
        or not extensions
        or any(
            not isinstance(item, str)
            or not item.startswith(".")
            or item != item.lower()
            or "/" in item
            or "\\" in item
            for item in extensions
        )
        or len(extensions) != len(set(extensions))
    ):
        raise ValueError("dataset extensions are invalid")
    entries = list(dataset.iterdir())
    unsafe = [
        entry.name
        for entry in entries
        if entry.is_symlink() or not stat.S_ISREG(entry.lstat().st_mode)
    ]
    if unsafe:
        raise RuntimeError({"dataset_unsafe_entries": sorted(unsafe)})
    evaluator_order = list_supported_images(str(dataset), extensions)
    if (
        not isinstance(evaluator_order, list)
        or not evaluator_order
        or any(not isinstance(name, str) for name in evaluator_order)
        or len(evaluator_order) != len(set(evaluator_order))
    ):
        raise RuntimeError("image enumerator returned an invalid or empty image list")
    for name in evaluator_order:
        if (
            name in {".", ".."}
            or Path(name).name != name
            or "/" in name
            or "\\" in name
            or Path(name).suffix.lower() not in extensions
        ):
            raise RuntimeError(f"image enumerator returned an unsafe name: {name!r}")
    image_names = set(evaluator_order)
    stems: set[str] = set()
    prompt_names: set[str] = set()
    expected_prompt_names = {
        f"{os.path.splitext(image_name)[0]}.txt" for image_name in evaluator_order
    }
    regular_names = {entry.name for entry in entries}
    expected_names = image_names | expected_prompt_names
    unexpected = regular_names - expected_names
    missing = expected_names - regular_names
    if unexpected or missing:
        raise RuntimeError(
            {
                "unexpected_dataset_files": sorted(unexpected),
                "missing_dataset_files": sorted(missing),
            }
        )
    rows = []
    pil_image = importlib.import_module("PIL.Image")
    for index, image_name in enumerate(evaluator_order):
        image_path = dataset / image_name
        stem = os.path.splitext(image_name)[0]
        if stem in stems:
            raise RuntimeError(f"ambiguous duplicate image stem in dataset: {stem}")
        stems.add(stem)
        prompt_name = f"{stem}.txt"
        prompt_path = dataset / prompt_name
        image_bytes, image_stat = _read_stable_regular(image_path, "dataset image")
        prompt_bytes, prompt_stat = _read_stable_regular(prompt_path, "paired prompt")
        try:
            prompt = prompt_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"paired prompt is not UTF-8: {prompt_path}") from exc
        if not prompt.strip():
            raise RuntimeError(f"paired prompt is empty: {prompt_path}")
        prompt_names.add(prompt_name)
        with pil_image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
        with pil_image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            image_format = image.format
            mode = image.mode
        if width <= 0 or height <= 0:
            raise RuntimeError(f"image has invalid dimensions: {image_path}")
        rows.append(
            {
                "index": index,
                "image": image_name,
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "image_bytes": image_stat.st_size,
                "image_width": width,
                "image_height": height,
                "image_format": image_format,
                "image_mode": mode,
                "prompt": prompt_name,
                "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                "prompt_bytes": prompt_stat.st_size,
            }
        )
    if prompt_names != expected_prompt_names:
        raise RuntimeError("dataset prompt pairing changed during capture")
    identity = {"evaluator_order": evaluator_order, "rows": rows}
    identity["sha256"] = _json_sha256(identity)
    return identity


def validate_identity(value: dict[str, Any]) -> dict[str, Any]:
    """Validate an evaluator-order identity without requiring the live files."""

    if not isinstance(value, dict) or set(value) != {
        "evaluator_order",
        "rows",
        "sha256",
    }:
        raise ValueError("dataset identity schema mismatch")
    order = value["evaluator_order"]
    rows = value["rows"]
    if (
        not isinstance(order, list)
        or not order
        or any(not isinstance(item, str) or not item for item in order)
        or len(order) != len(set(order))
        or not isinstance(rows, list)
        or len(rows) != len(order)
    ):
        raise ValueError("dataset identity order/rows are invalid")
    expected_keys = {
        "index",
        "image",
        "image_sha256",
        "image_bytes",
        "image_width",
        "image_height",
        "image_format",
        "image_mode",
        "prompt",
        "prompt_sha256",
        "prompt_bytes",
    }
    prompts: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise ValueError("dataset identity row schema mismatch")
        if row["index"] != index or row["image"] != order[index]:
            raise ValueError("dataset identity row order mismatch")
        for field in ("image", "image_format", "image_mode", "prompt"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"dataset identity row {field} is empty")
        for field in ("image_sha256", "prompt_sha256"):
            if not isinstance(row[field], str) or not _SHA256_HEX.fullmatch(row[field]):
                raise ValueError(f"dataset identity row {field} is invalid")
        for field in ("image_bytes", "image_width", "image_height", "prompt_bytes"):
            if (
                isinstance(row[field], bool)
                or not isinstance(row[field], int)
                or row[field] <= 0
            ):
                raise ValueError(f"dataset identity row {field} is invalid")
        expected_prompt = f"{os.path.splitext(row['image'])[0]}.txt"
        if row["prompt"] != expected_prompt or row["prompt"] in prompts:
            raise ValueError("dataset identity prompt pairing is invalid")
        prompts.add(row["prompt"])
    body = {"evaluator_order": order, "rows": rows}
    if (
        not isinstance(value["sha256"], str)
        or not _SHA256_HEX.fullmatch(value["sha256"])
        or value["sha256"] != _json_sha256(body)
    ):
        raise ValueError("dataset identity digest mismatch")
    return value
