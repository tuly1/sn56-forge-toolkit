"""Capture the exact clean Forge/container identity used by Krea Stage-2."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
from typing import Any, Mapping, Sequence


KIND = "forge-krea-stage2-production-identity"
ASSET_ATTESTATION_KIND = "forge-krea-stage2-base-asset-attestation"
SCHEMA = 1
DOCKERFILE_PATH = "ops/docker/standalone-image-toolkit-trainer.dockerfile"
RUNTIME_INPUT_PATHS = (
    "ops/docker/image-runtime-lock.txt",
    "ops/docker/image-runtime-phase1-constraints.txt",
    "ops/docker/verify_image_runtime.py",
)
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_REPO_DIGEST = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
_MODEL_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
KREA_MODEL_ID = "krea/Krea-2-Raw"
KREA_TEXT_ENCODER_ID = "Qwen/Qwen3-VL-4B-Instruct"
_ASSET_DESTINATIONS = {
    "base_model": "/cache/models/krea--Krea-2-Raw",
    "text_encoder": "/cache/hf_cache/Qwen--Qwen3-VL-4B-Instruct",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = keys - set(value)
    extra = set(value) - keys
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _git_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _GIT_SHA.fullmatch(value):
        raise ValueError(f"{label} must be a full lowercase Git SHA-1")
    return value


def _utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value
    ):
        raise ValueError(f"{label} must be canonical whole-second UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{label} is not real UTC") from exc
    if parsed < datetime(2020, 1, 1, tzinfo=timezone.utc) or parsed > datetime.now(
        timezone.utc
    ) + timedelta(seconds=60):
        raise ValueError(f"{label} is outside accepted evidence time bounds")
    return value


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a canonical relative path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or str(pure) != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    return value


def _reject_symlink_ancestors(
    path: Path, label: str, *, include_leaf: bool = True
) -> None:
    current = path if include_leaf else path.parent
    while True:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink component: {current}")
        if current == current.parent:
            return
        current = current.parent


def _file_identity(value: Any, label: str) -> dict[str, Any]:
    row = _object(value, label)
    _exact(row, {"path", "sha256", "bytes"}, label)
    path = _relative_path(row["path"], f"{label}.path")
    digest = _digest(row["sha256"], f"{label}.sha256")
    size = row["bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{label}.bytes must be a positive integer")
    return {"path": path, "sha256": digest, "bytes": size}


def _asset_root(value: str | Path, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(str(value))))
    _reject_symlink_ancestors(path, label)
    if not path.is_dir() or not stat.S_ISDIR(path.stat().st_mode):
        raise ValueError(f"{label} must be a real directory")
    return path


_SAMPLE_BAND_BYTES = 1024 * 1024


def _stat_record(info: os.stat_result) -> dict[str, int]:
    return {
        "dev": int(info.st_dev),
        "ino": int(info.st_ino),
        "mode": int(info.st_mode & 0o7777),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
        "nlink": int(info.st_nlink),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _stable_tuple(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_uid),
        int(info.st_gid),
        int(info.st_nlink),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _sample_hash(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    if size <= 3 * _SAMPLE_BAND_BYTES:
        spans = [(0, size)]
    else:
        spans = [
            (0, _SAMPLE_BAND_BYTES),
            ((size - _SAMPLE_BAND_BYTES) // 2, _SAMPLE_BAND_BYTES),
            (size - _SAMPLE_BAND_BYTES, _SAMPLE_BAND_BYTES),
        ]
    for offset, length in spans:
        raw = os.pread(descriptor, length, offset)
        if len(raw) != length:
            raise ValueError("asset sample read was truncated")
        digest.update(f"{offset}:{length}\0".encode("ascii"))
        digest.update(raw)
    return digest.hexdigest()


def _capture_asset_tree(
    root: Path,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    """Hash every file once under a race-resistant no-symlink contract."""

    root = _asset_root(root, "asset tree")
    tree_digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    directories: list[str] = []
    seen_inodes: set[tuple[int, int]] = set()

    def visit(directory: Path, logical: str) -> None:
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"asset tree directory is not stable: {directory}")
        tree_digest.update(f"D\0{logical}\0".encode("utf-8"))
        if logical:
            directories.append(logical)
        with os.scandir(directory) as entries:
            children = sorted(entries, key=lambda item: item.name)
        for child in children:
            path = Path(child.path)
            relative = f"{logical}/{child.name}" if logical else child.name
            before = path.lstat()
            if stat.S_ISLNK(before.st_mode):
                raise ValueError(f"asset tree contains a symlink: {path}")
            if stat.S_ISDIR(before.st_mode):
                visit(path, relative)
                continue
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"asset tree contains a special entry: {path}")
            inode = (int(before.st_dev), int(before.st_ino))
            if inode in seen_inodes:
                raise ValueError("asset tree contains duplicate hard-linked content")
            seen_inodes.add(inode)
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                opened = os.fstat(descriptor)
                if _stable_tuple(opened) != _stable_tuple(before):
                    raise ValueError("asset file changed while it was opened")
                file_digest = hashlib.sha256()
                tree_digest.update(f"F\0{relative}\0{opened.st_size}\0".encode("utf-8"))
                consumed = 0
                while True:
                    block = os.read(descriptor, 8 * 1024 * 1024)
                    if not block:
                        break
                    consumed += len(block)
                    file_digest.update(block)
                    tree_digest.update(block)
                sample_sha256 = _sample_hash(descriptor, int(opened.st_size))
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            if consumed != opened.st_size or _stable_tuple(after) != _stable_tuple(
                opened
            ):
                raise ValueError("asset file changed while it was hashed")
            files.append(
                {
                    "path": relative,
                    "bytes": int(opened.st_size),
                    "sha256": file_digest.hexdigest(),
                    "sample_sha256": sample_sha256,
                    "stat": _stat_record(opened),
                }
            )

    visit(root, "")
    files.sort(key=lambda row: row["path"])
    content = {
        "kind": "directory",
        "sha256": tree_digest.hexdigest(),
        "files": len(files),
        "bytes": sum(row["bytes"] for row in files),
        "symlinks": 0,
    }
    directories.sort()
    return content, directories, files


def _verify_asset_tree(root: Path, expected: Mapping[str, Any]) -> None:
    """Rewalk exact membership/stat rows and sample every file pre/post cell."""

    root = _asset_root(root, "live asset tree")
    observed_paths: list[str] = []
    observed_directories: list[str] = []
    expected_rows = {row["path"]: row for row in expected["files"]}
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        current = Path(directory)
        for name in dirnames:
            child = current / name
            if child.is_symlink() or not child.is_dir():
                raise ValueError("live asset directory membership changed")
            observed_directories.append(child.relative_to(root).as_posix())
        for name in filenames:
            path = current / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or relative not in expected_rows:
                raise ValueError("live asset file membership changed")
            row = expected_rows[relative]
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                before = os.fstat(descriptor)
                sample_sha256 = _sample_hash(descriptor, int(before.st_size))
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            if (
                _stable_tuple(before) != _stable_tuple(after)
                or int(before.st_size) != row["bytes"]
                or _stat_record(before) != row["stat"]
                or sample_sha256 != row["sample_sha256"]
            ):
                raise ValueError("live asset bytes/stat changed after attestation")
            observed_paths.append(relative)
    if len(observed_paths) != len(set(observed_paths)) or sorted(
        observed_paths
    ) != sorted(expected_rows):
        raise ValueError("live asset exact membership differs from attestation")
    if sorted(observed_directories) != expected["directories"]:
        raise ValueError("live asset directory membership differs from attestation")


def _load_staging_manifest(path_value: str | Path) -> dict[str, Any]:
    """Derive immutable repo revisions from the reviewed asset-stage record."""

    path = Path(os.path.abspath(os.path.expanduser(str(path_value))))
    _reject_symlink_ancestors(path, "asset staging manifest")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            chunks = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ValueError("asset staging manifest is unavailable") from exc
    raw = b"".join(chunks)
    if (
        not stat.S_ISREG(before.st_mode)
        or _stable_tuple(before) != _stable_tuple(after)
        or len(raw) != before.st_size
    ):
        raise ValueError("asset staging manifest changed while read")
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("asset staging manifest is not JSON") from exc
    document = _object(document, "asset staging manifest")
    _exact(document, {"assets"}, "asset staging manifest")
    assets = document["assets"]
    if not isinstance(assets, list):
        raise ValueError("asset staging manifest assets must be an array")
    expected = {
        "base_model": KREA_MODEL_ID,
        "text_encoder": KREA_TEXT_ENCODER_ID,
    }
    normalized: list[dict[str, str]] = []
    for name in ("base_model", "text_encoder"):
        matches = [
            row
            for row in assets
            if isinstance(row, dict) and row.get("repo_id") == expected[name]
        ]
        if len(matches) != 1:
            raise ValueError("asset staging manifest lacks one exact required repo")
        row = matches[0]
        revision = row.get("revision")
        if (
            not isinstance(revision, str)
            or _MODEL_REVISION.fullmatch(revision) is None
            or row.get("local_dir") != _ASSET_DESTINATIONS[name]
            or row.get("resolved_path") != _ASSET_DESTINATIONS[name]
        ):
            raise ValueError("asset staging manifest repo binding differs")
        normalized.append(
            {
                "name": name,
                "repo_id": expected[name],
                "revision": revision,
                "container_root": _ASSET_DESTINATIONS[name],
            }
        )
    return {
        "path": str(path),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "assets": normalized,
        "assets_sha256": canonical_sha256(normalized),
    }


def capture_asset_attestation(
    *,
    base_model_path: str | Path,
    text_encoder_path: str | Path,
    staging_manifest_path: str | Path,
    captured_at_utc: str,
) -> dict[str, Any]:
    """Content-hash the exact two read-only training assets once per host."""

    staging = _load_staging_manifest(staging_manifest_path)
    revisions = {row["name"]: row["revision"] for row in staging["assets"]}
    sources = {
        "base_model": _asset_root(base_model_path, "base model asset"),
        "text_encoder": _asset_root(text_encoder_path, "text encoder asset"),
    }
    assets: list[dict[str, Any]] = []
    content_identities: dict[str, dict[str, Any]] = {}
    for name in ("base_model", "text_encoder"):
        content, directories, files = _capture_asset_tree(sources[name])
        content_identities[name] = content
        assets.append(
            {
                "name": name,
                "source_path": str(sources[name]),
                "mount_destination": _ASSET_DESTINATIONS[name],
                "content_identity": content,
                "directories": directories,
                "directories_manifest_sha256": canonical_sha256(directories),
                "files": files,
                "files_manifest_sha256": canonical_sha256(files),
                "stat_tree_sha256": canonical_sha256(
                    [
                        {
                            "path": row["path"],
                            "bytes": row["bytes"],
                            "stat": row["stat"],
                        }
                        for row in files
                    ]
                ),
            }
        )
    body = {
        "schema": SCHEMA,
        "kind": ASSET_ATTESTATION_KIND,
        "captured_at_utc": _utc(captured_at_utc, "asset capture time"),
        "source_staging_manifest": staging,
        "base_model": {
            "model_id": KREA_MODEL_ID,
            "revision": revisions["base_model"],
        },
        "text_encoder": {
            "model_id": KREA_TEXT_ENCODER_ID,
            "revision": revisions["text_encoder"],
        },
        "assets": assets,
        "training_identity_sha256": canonical_sha256(content_identities),
        "release_authorized": False,
    }
    return validate_asset_attestation(
        {**body, "attestation_sha256": canonical_sha256(body)}
    )


def validate_asset_attestation(value: Any) -> dict[str, Any]:
    record = _object(value, "base asset attestation")
    _exact(
        record,
        {
            "schema",
            "kind",
            "captured_at_utc",
            "source_staging_manifest",
            "base_model",
            "text_encoder",
            "assets",
            "training_identity_sha256",
            "release_authorized",
            "attestation_sha256",
        },
        "base asset attestation",
    )
    body = {key: item for key, item in record.items() if key != "attestation_sha256"}
    if (
        record["schema"] != SCHEMA
        or record["kind"] != ASSET_ATTESTATION_KIND
        or record["attestation_sha256"] != canonical_sha256(body)
        or record["release_authorized"] is not False
    ):
        raise ValueError("base asset attestation header differs")
    _utc(record["captured_at_utc"], "asset capture time")
    staging = _object(record["source_staging_manifest"], "source staging manifest")
    _exact(
        staging,
        {"path", "file_sha256", "assets", "assets_sha256"},
        "source staging manifest",
    )
    staging_path = Path(str(staging["path"]))
    if (
        not staging_path.is_absolute()
        or ".." in staging_path.parts
        or str(staging_path) != os.path.abspath(str(staging["path"]))
    ):
        raise ValueError("source staging manifest path is not canonical")
    _digest(staging["file_sha256"], "source staging manifest file")
    if not isinstance(staging["assets"], list) or staging[
        "assets_sha256"
    ] != canonical_sha256(staging["assets"]):
        raise ValueError("source staging manifest asset digest differs")
    expected_models = {
        "base_model": (KREA_MODEL_ID, record["base_model"]),
        "text_encoder": (KREA_TEXT_ENCODER_ID, record["text_encoder"]),
    }
    for label, (expected_id, raw) in expected_models.items():
        model = _object(raw, label)
        _exact(model, {"model_id", "revision"}, label)
        if (
            model["model_id"] != expected_id
            or not isinstance(model["revision"], str)
            or _MODEL_REVISION.fullmatch(model["revision"]) is None
        ):
            raise ValueError(f"{label} identity is not immutable")
    expected_staging_assets = [
        {
            "name": name,
            "repo_id": expected_id,
            "revision": record[name]["revision"],
            "container_root": _ASSET_DESTINATIONS[name],
        }
        for name, expected_id in (
            ("base_model", KREA_MODEL_ID),
            ("text_encoder", KREA_TEXT_ENCODER_ID),
        )
    ]
    if staging["assets"] != expected_staging_assets:
        raise ValueError("source staging manifest revisions differ")
    assets = record["assets"]
    if not isinstance(assets, list) or len(assets) != 2:
        raise ValueError("base asset attestation must contain exactly two assets")
    normalized_content: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for index, raw in enumerate(assets):
        row = _object(raw, f"assets[{index}]")
        _exact(
            row,
            {
                "name",
                "source_path",
                "mount_destination",
                "content_identity",
                "directories",
                "directories_manifest_sha256",
                "files",
                "files_manifest_sha256",
                "stat_tree_sha256",
            },
            f"assets[{index}]",
        )
        name = row["name"]
        if name not in _ASSET_DESTINATIONS or name in seen:
            raise ValueError("base asset name is unknown or duplicated")
        source = Path(str(row["source_path"]))
        if (
            not source.is_absolute()
            or ".." in source.parts
            or str(source) != os.path.abspath(str(row["source_path"]))
            or row["mount_destination"] != _ASSET_DESTINATIONS[name]
        ):
            raise ValueError("base asset source/destination is not canonical")
        content = _object(row["content_identity"], "asset content identity")
        _exact(
            content,
            {"kind", "sha256", "files", "bytes", "symlinks"},
            "asset content identity",
        )
        if content["kind"] != "directory":
            raise ValueError("base asset content must identify a directory")
        _digest(content["sha256"], "asset content sha256")
        for field in ("files", "bytes", "symlinks"):
            if (
                isinstance(content[field], bool)
                or not isinstance(content[field], int)
                or content[field] < 0
            ):
                raise ValueError("base asset content counts are invalid")
        if content["files"] <= 0 or content["bytes"] <= 0 or content["symlinks"] != 0:
            raise ValueError("base asset content is empty")
        directories = row["directories"]
        if (
            not isinstance(directories, list)
            or any(
                not isinstance(item, str)
                or _relative_path(item, "asset directory") != item
                for item in directories
            )
            or directories != sorted(set(directories))
            or row["directories_manifest_sha256"] != canonical_sha256(directories)
        ):
            raise ValueError("asset directory manifest differs")
        files = row["files"]
        if not isinstance(files, list) or len(files) != content["files"]:
            raise ValueError("asset file manifest count differs")
        normalized_files: list[dict[str, Any]] = []
        paths: set[str] = set()
        inodes: set[tuple[int, int]] = set()
        for file_index, raw_file in enumerate(files):
            file_row = _object(raw_file, f"asset file[{file_index}]")
            _exact(
                file_row,
                {"path", "bytes", "sha256", "sample_sha256", "stat"},
                f"asset file[{file_index}]",
            )
            path = _relative_path(file_row["path"], "asset file path")
            if path in paths:
                raise ValueError("asset file path is duplicated")
            size = file_row["bytes"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError("asset file byte count is invalid")
            _digest(file_row["sha256"], "asset file sha256")
            _digest(file_row["sample_sha256"], "asset file sample sha256")
            stat_row = _object(file_row["stat"], "asset file stat")
            stat_keys = {
                "dev",
                "ino",
                "mode",
                "uid",
                "gid",
                "nlink",
                "mtime_ns",
                "ctime_ns",
            }
            _exact(stat_row, stat_keys, "asset file stat")
            if any(
                isinstance(stat_row[key], bool)
                or not isinstance(stat_row[key], int)
                or stat_row[key] < 0
                for key in stat_keys
            ):
                raise ValueError("asset file stat is invalid")
            inode = (stat_row["dev"], stat_row["ino"])
            if inode in inodes:
                raise ValueError("asset file inode is duplicated")
            paths.add(path)
            inodes.add(inode)
            normalized_files.append(dict(file_row))
        if normalized_files != sorted(normalized_files, key=lambda item: item["path"]):
            raise ValueError("asset files must be sorted by path")
        if sum(item["bytes"] for item in normalized_files) != content["bytes"]:
            raise ValueError("asset file bytes differ from content identity")
        if row["files_manifest_sha256"] != canonical_sha256(normalized_files):
            raise ValueError("asset files manifest digest differs")
        expected_stat_sha = canonical_sha256(
            [
                {"path": item["path"], "bytes": item["bytes"], "stat": item["stat"]}
                for item in normalized_files
            ]
        )
        if row["stat_tree_sha256"] != expected_stat_sha:
            raise ValueError("asset stat-tree digest differs")
        seen.add(name)
        normalized_content[name] = dict(content)
    if seen != set(_ASSET_DESTINATIONS):
        raise ValueError("base asset attestation set differs")
    if record["training_identity_sha256"] != canonical_sha256(normalized_content):
        raise ValueError("base asset training identity differs")
    return dict(record)


def verify_live_asset_attestation(
    value: Any,
    *,
    base_model_path: str | Path,
    text_encoder_path: str | Path,
) -> dict[str, Any]:
    """Cheaply prove the content-hashed host trees have not changed in place."""

    record = validate_asset_attestation(value)
    if (
        _load_staging_manifest(record["source_staging_manifest"]["path"])
        != record["source_staging_manifest"]
    ):
        raise ValueError("source staging manifest changed after attestation")
    sources = {
        "base_model": _asset_root(base_model_path, "live base model asset"),
        "text_encoder": _asset_root(text_encoder_path, "live text encoder asset"),
    }
    rows = {row["name"]: row for row in record["assets"]}
    for name, source in sources.items():
        if str(source) != rows[name]["source_path"]:
            raise ValueError("live asset source differs from its attestation")
        _verify_asset_tree(source, rows[name])
    return record


def publish_asset_attestation(value: Any, output: str | Path) -> dict[str, Any]:
    record = validate_asset_attestation(value)
    path = Path(os.path.abspath(os.path.expanduser(str(output))))
    _reject_symlink_ancestors(path, "asset attestation output")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(record) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return asset_attestation_binding(path)


def load_asset_attestation(path: str | Path) -> dict[str, Any]:
    source = Path(os.path.abspath(os.path.expanduser(str(path))))
    _reject_symlink_ancestors(source, "asset attestation")
    if source.is_symlink() or not source.is_file():
        raise ValueError("asset attestation must be a regular file")
    raw = source.read_bytes()
    try:
        record = validate_asset_attestation(json.loads(raw))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("asset attestation is not JSON") from exc
    if raw != canonical_bytes(record) + b"\n":
        raise ValueError("asset attestation is not canonical JSON")
    return record


def asset_attestation_binding(path: str | Path) -> dict[str, Any]:
    source = Path(os.path.abspath(os.path.expanduser(str(path))))
    record = load_asset_attestation(source)
    return {
        "path": str(source),
        "file_sha256": hashlib.sha256(canonical_bytes(record) + b"\n").hexdigest(),
        "attestation_sha256": record["attestation_sha256"],
    }


def build(
    *,
    forge: Mapping[str, Any],
    container_image: Mapping[str, Any],
    dockerfile: Mapping[str, Any],
    runtime_inputs: Sequence[Mapping[str, Any]],
    base_model: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    captured_at_utc: str,
) -> dict[str, Any]:
    """Build and validate a portable exact production identity record."""

    forge_value = _object(dict(forge), "Forge identity")
    _exact(
        forge_value,
        {"commit_sha1", "tree_sha1", "worktree_state"},
        "Forge identity",
    )
    normalized_forge = {
        "commit_sha1": _git_sha(forge_value["commit_sha1"], "Forge commit"),
        "tree_sha1": _git_sha(forge_value["tree_sha1"], "Forge tree"),
        "worktree_state": forge_value["worktree_state"],
    }
    if normalized_forge["worktree_state"] != "clean-including-untracked":
        raise ValueError("Stage-2 production identity requires a clean worktree")

    image = _object(dict(container_image), "container image identity")
    _exact(image, {"image_id", "repo_digest"}, "container image identity")
    if not isinstance(image["image_id"], str) or not _IMAGE_ID.fullmatch(
        image["image_id"]
    ):
        raise ValueError("container image_id must be an immutable sha256 identity")
    if not isinstance(image["repo_digest"], str) or not _REPO_DIGEST.fullmatch(
        image["repo_digest"]
    ):
        raise ValueError("container repo_digest must be an immutable repo digest")
    normalized_image = dict(image)

    dockerfile_value = _file_identity(dockerfile, "Dockerfile identity")
    if dockerfile_value["path"] != DOCKERFILE_PATH:
        raise ValueError("Stage-2 must bind the production toolkit Dockerfile")
    if isinstance(runtime_inputs, (str, bytes)) or not runtime_inputs:
        raise ValueError("runtime_inputs must be a non-empty complete manifest")
    normalized_inputs = [
        _file_identity(dict(row), f"runtime_inputs[{index}]")
        for index, row in enumerate(runtime_inputs)
    ]
    paths = [row["path"] for row in normalized_inputs]
    if tuple(paths) != RUNTIME_INPUT_PATHS:
        raise ValueError("runtime_inputs must bind the exact production input set")

    model = _object(dict(base_model), "base model identity")
    _exact(
        model,
        {
            "model_id",
            "revision",
            "training_identity_sha256",
            "asset_attestation_sha256",
            "text_encoder_id",
            "text_encoder_revision",
        },
        "base model identity",
    )
    if (
        model["model_id"] != KREA_MODEL_ID
        or not isinstance(model["revision"], str)
        or _MODEL_REVISION.fullmatch(model["revision"]) is None
        or model["text_encoder_id"] != KREA_TEXT_ENCODER_ID
        or not isinstance(model["text_encoder_revision"], str)
        or _MODEL_REVISION.fullmatch(model["text_encoder_revision"]) is None
    ):
        raise ValueError("Stage-2 base model is not immutable Krea-2-Raw")
    normalized_model = {
        "model_id": KREA_MODEL_ID,
        "revision": model["revision"],
        "training_identity_sha256": _digest(
            model["training_identity_sha256"], "base model training identity"
        ),
        "asset_attestation_sha256": _digest(
            model["asset_attestation_sha256"], "base model asset attestation"
        ),
        "text_encoder_id": KREA_TEXT_ENCODER_ID,
        "text_encoder_revision": model["text_encoder_revision"],
    }
    runtime = _object(dict(runtime_contract), "container runtime contract")
    runtime_keys = {
        "runtime_identity_sha256",
        "venv_tree_manifest_sha256",
        "trainer_identity_sha256",
        "measurement_tool_sha256",
        "jit_enabled",
    }
    _exact(runtime, runtime_keys, "container runtime contract")
    if runtime["jit_enabled"] is not True:
        raise ValueError("Stage-2 production runtime must enable the measured JIT path")
    normalized_runtime = {
        key: _digest(runtime[key], f"container runtime {key}")
        for key in runtime_keys - {"jit_enabled"}
    }
    normalized_runtime["jit_enabled"] = True

    body = {
        "schema": SCHEMA,
        "kind": KIND,
        "captured_at_utc": _utc(captured_at_utc, "capture time"),
        "forge": normalized_forge,
        "container_image": normalized_image,
        "dockerfile": dockerfile_value,
        "runtime_inputs": normalized_inputs,
        "runtime_inputs_sha256": canonical_sha256(normalized_inputs),
        "base_model": normalized_model,
        "runtime_contract": normalized_runtime,
    }
    record = {**body, "production_identity_sha256": canonical_sha256(body)}
    return validate(record)


def validate(value: Any) -> dict[str, Any]:
    record = _object(value, "Stage-2 production identity")
    _exact(
        record,
        {
            "schema",
            "kind",
            "captured_at_utc",
            "forge",
            "container_image",
            "dockerfile",
            "runtime_inputs",
            "runtime_inputs_sha256",
            "base_model",
            "runtime_contract",
            "production_identity_sha256",
        },
        "Stage-2 production identity",
    )
    if record["schema"] != SCHEMA or record["kind"] != KIND:
        raise ValueError("unsupported Stage-2 production identity")
    # Validate each leaf without recursively calling ``build``.
    _utc(record["captured_at_utc"], "capture time")
    forge = _object(record["forge"], "Forge identity")
    _exact(forge, {"commit_sha1", "tree_sha1", "worktree_state"}, "Forge identity")
    _git_sha(forge["commit_sha1"], "Forge commit")
    _git_sha(forge["tree_sha1"], "Forge tree")
    if forge["worktree_state"] != "clean-including-untracked":
        raise ValueError("Stage-2 production identity requires a clean worktree")
    image = _object(record["container_image"], "container image identity")
    _exact(image, {"image_id", "repo_digest"}, "container image identity")
    if not isinstance(image["image_id"], str) or not _IMAGE_ID.fullmatch(
        image["image_id"]
    ):
        raise ValueError("container image_id must be an immutable sha256 identity")
    if not isinstance(image["repo_digest"], str) or not _REPO_DIGEST.fullmatch(
        image["repo_digest"]
    ):
        raise ValueError("container repo_digest must be an immutable repo digest")
    dockerfile = _file_identity(record["dockerfile"], "Dockerfile identity")
    if dockerfile["path"] != DOCKERFILE_PATH:
        raise ValueError("Stage-2 must bind the production toolkit Dockerfile")
    raw_inputs = record["runtime_inputs"]
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ValueError("runtime_inputs must be a non-empty complete manifest")
    inputs = [
        _file_identity(row, f"runtime_inputs[{index}]")
        for index, row in enumerate(raw_inputs)
    ]
    paths = [row["path"] for row in inputs]
    if tuple(paths) != RUNTIME_INPUT_PATHS:
        raise ValueError("runtime_inputs must bind the exact production input set")
    if record["runtime_inputs_sha256"] != canonical_sha256(inputs):
        raise ValueError("runtime input manifest digest mismatch")
    model = _object(record["base_model"], "base model identity")
    _exact(
        model,
        {
            "model_id",
            "revision",
            "training_identity_sha256",
            "asset_attestation_sha256",
            "text_encoder_id",
            "text_encoder_revision",
        },
        "base model identity",
    )
    if (
        model["model_id"] != KREA_MODEL_ID
        or not isinstance(model["revision"], str)
        or _MODEL_REVISION.fullmatch(model["revision"]) is None
        or model["text_encoder_id"] != KREA_TEXT_ENCODER_ID
        or not isinstance(model["text_encoder_revision"], str)
        or _MODEL_REVISION.fullmatch(model["text_encoder_revision"]) is None
    ):
        raise ValueError("Stage-2 base model is not immutable Krea-2-Raw")
    _digest(model["training_identity_sha256"], "base model training identity")
    _digest(model["asset_attestation_sha256"], "base model asset attestation")
    runtime = _object(record["runtime_contract"], "container runtime contract")
    runtime_keys = {
        "runtime_identity_sha256",
        "venv_tree_manifest_sha256",
        "trainer_identity_sha256",
        "measurement_tool_sha256",
        "jit_enabled",
    }
    _exact(runtime, runtime_keys, "container runtime contract")
    if runtime["jit_enabled"] is not True:
        raise ValueError("Stage-2 production runtime must enable the measured JIT path")
    for key in runtime_keys - {"jit_enabled"}:
        _digest(runtime[key], f"container runtime {key}")
    body = {
        key: item for key, item in record.items() if key != "production_identity_sha256"
    }
    if record["production_identity_sha256"] != canonical_sha256(body):
        raise ValueError("Stage-2 production identity digest mismatch")
    return record


def _safe_root(value: str | Path) -> Path:
    raw = str(value)
    path = Path(os.path.abspath(os.path.expanduser(raw)))
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"Forge repository has a symlink component: {current}")
        current = current.parent
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Forge repository must be a real directory")
    return path


def _git(root: Path, *arguments: str) -> bytes:
    executable = shutil.which("git")
    if executable is None:
        raise ValueError("Git executable is unavailable")
    environment = {
        "PATH": os.path.dirname(executable) + ":/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    try:
        return subprocess.run(
            [
                executable,
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.autocrlf=false",
                "-C",
                str(root),
                *arguments,
            ],
            check=True,
            capture_output=True,
            timeout=30,
            env=environment,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Forge Git identity could not be read safely") from exc


def _hash_tracked_file(root: Path, relative: str, label: str) -> dict[str, Any]:
    portable = _relative_path(relative, label)
    _git(root, "ls-files", "--error-unmatch", "--", portable)
    path = root.joinpath(*PurePosixPath(portable).parts)
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"{label} must be a tracked regular non-symlink file")
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"{label} must not be empty")
    return {
        "path": portable,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def capture(
    repository: str | Path,
    *,
    image_id: str,
    repo_digest: str,
    runtime_input_paths: Sequence[str],
    base_model: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    captured_at_utc: str,
) -> dict[str, Any]:
    """Capture a clean repository and every declared image-build input."""

    root = _safe_root(repository)
    try:
        top = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip())
        commit = _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
        tree = _git(root, "rev-parse", "--verify", "HEAD^{tree}").decode().strip()
    except UnicodeDecodeError as exc:
        raise ValueError("Forge Git identity is not UTF-8/ASCII") from exc
    if top.resolve() != root.resolve():
        raise ValueError("repository is not the Forge Git root")
    cache_rows = _git(root, "ls-files", "-v", "-z").split(b"\0")
    if any(row and not row.startswith(b"H ") for row in cache_rows):
        raise ValueError(
            "Forge index contains skip-worktree, assume-unchanged, or "
            "non-canonical cache state"
        )
    if _git(root, "status", "--porcelain=v2", "-z", "--untracked-files=all"):
        raise ValueError("Forge worktree must be clean, including untracked files")
    dockerfile = _hash_tracked_file(root, DOCKERFILE_PATH, "Dockerfile")
    if isinstance(runtime_input_paths, (str, bytes)) or not runtime_input_paths:
        raise ValueError("runtime_input_paths must be a non-empty sequence")
    normalized = tuple(
        _relative_path(item, "runtime input path") for item in runtime_input_paths
    )
    if normalized != RUNTIME_INPUT_PATHS:
        raise ValueError("runtime input paths must be the exact production input set")
    inputs = [_hash_tracked_file(root, item, "runtime input") for item in normalized]
    try:
        final_commit = (
            _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
        )
        final_tree = _git(root, "rev-parse", "--verify", "HEAD^{tree}").decode().strip()
    except UnicodeDecodeError as exc:
        raise ValueError("final Forge Git identity is not ASCII") from exc
    if (
        final_commit != commit
        or final_tree != tree
        or _git(root, "status", "--porcelain=v2", "-z", "--untracked-files=all")
    ):
        raise ValueError("Forge identity changed or became dirty during capture")
    return build(
        forge={
            "commit_sha1": commit,
            "tree_sha1": tree,
            "worktree_state": "clean-including-untracked",
        },
        container_image={"image_id": image_id, "repo_digest": repo_digest},
        dockerfile=dockerfile,
        runtime_inputs=inputs,
        base_model=base_model,
        runtime_contract=runtime_contract,
        captured_at_utc=captured_at_utc,
    )


def publish(value: Any, output: str | Path) -> dict[str, Any]:
    """Create one canonical identity file without overwriting evidence."""

    record = validate(value)
    path = Path(os.path.abspath(os.path.expanduser(output)))
    _reject_symlink_ancestors(path, "production identity output")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(path, "production identity output")
    payload = canonical_bytes(record) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return binding(path)


def load(path: str | Path) -> dict[str, Any]:
    source = Path(os.path.abspath(os.path.expanduser(path)))
    _reject_symlink_ancestors(source, "production identity")
    if not source.is_file() or not stat.S_ISREG(source.stat().st_mode):
        raise ValueError("production identity must be a regular non-symlink file")
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("production identity is not JSON") from exc
    record = validate(value)
    if raw != canonical_bytes(record) + b"\n":
        raise ValueError("production identity is not canonical JSON")
    return record


def binding(path: str | Path) -> dict[str, Any]:
    source = Path(os.path.abspath(os.path.expanduser(path)))
    record = load(source)
    return {
        "path": str(source),
        "file_sha256": hashlib.sha256(canonical_bytes(record) + b"\n").hexdigest(),
        "production_identity_sha256": record["production_identity_sha256"],
        "image_id": record["container_image"]["image_id"],
    }
