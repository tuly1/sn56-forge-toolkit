#!/usr/bin/env python3
"""Harvest public safetensors headers without downloading tensor bodies.

The input is an exact-tournament, create-only snapshot produced by
``safe_harvest_sync.py``.  Repository names and immutable revisions are taken
from hash-verified ``hf-tree`` observations; mutable Hugging Face ``main`` is
never queried.  Each network response must honor an exact byte Range.  A
server that returns ``200`` (and would therefore send a whole checkpoint) is
closed before its response body is read.

Only the eight-byte safetensors prefix and JSON header are requested.  The
weight body is never fetched.  Output is create-only and binds every parsed
record back to the public repository, immutable revision, task, path, LFS
SHA-256, and source tree-observation hashes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
from typing import Any, Callable, Iterable, Mapping, Protocol
from urllib.parse import quote
from urllib.request import Request, urlopen

try:
    from safe_harvest_sync import (
        IntegrityError,
        Publisher,
        SyncError,
        canonical_json,
        contains_forbidden_body,
        contains_forbidden_path,
        sha256_bytes,
        utc_iso,
        utc_now,
        utc_token,
        _open_regular_beneath,
    )
except ImportError:  # pragma: no cover - supports package-style test imports
    from .safe_harvest_sync import (
        IntegrityError,
        Publisher,
        SyncError,
        canonical_json,
        contains_forbidden_body,
        contains_forbidden_path,
        sha256_bytes,
        utc_iso,
        utc_now,
        utc_token,
        _open_regular_beneath,
    )


SCHEMA = "sn56.week7.public-safetensors-headers"
SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")
TASK_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
TOURNAMENT_RE = re.compile(r"tourn_[a-z0-9]+_[0-9]{8}")
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_TREE_BYTES = 64 * 1024 * 1024


class Response(Protocol):
    status: int
    headers: Mapping[str, str]

    def read(self, amount: int = -1) -> bytes: ...
    def close(self) -> None: ...


OpenRequest = Callable[[Request, float], Response]


def _default_open(request: Request, timeout: float) -> Response:
    return urlopen(request, timeout=timeout)  # type: ignore[return-value]


def _read_regular(root: Path, relative: str, limit: int) -> bytes:
    fd = _open_regular_beneath(root, relative)
    try:
        body = bytearray()
        while True:
            block = os.read(fd, min(1024 * 1024, limit + 1 - len(body)))
            if not block:
                return bytes(body)
            body.extend(block)
            if len(body) > limit:
                raise IntegrityError("metadata exceeds its safety ceiling")
    finally:
        os.close(fd)


def _cas_relative(digest: str) -> str:
    if SHA256_RE.fullmatch(digest) is None:
        raise IntegrityError("invalid SHA-256 CAS identity")
    return f"objects/sha256/{digest[:2]}/{digest}"


def _cas_bytes(root: Path, digest: str, limit: int = MAX_TREE_BYTES) -> bytes:
    body = _read_regular(root, _cas_relative(digest), limit)
    if sha256_bytes(body) != digest:
        raise IntegrityError("CAS bytes do not match their SHA-256 identity")
    return body


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return str(value)
    return None


def fetch_exact_range(
    url: str,
    start: int,
    end: int,
    total: int,
    *,
    timeout: float,
    opener: OpenRequest = _default_open,
) -> bytes:
    """Fetch one exact range, rejecting full-body and malformed responses."""
    if start < 0 or end < start or total <= end:
        raise IntegrityError("invalid requested byte range")
    request = Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "Range": f"bytes={start}-{end}",
            "User-Agent": "sn56-week7-public-header-harvester/1",
        },
        method="GET",
    )
    response = opener(request, timeout)
    try:
        # Check status and headers before any read.  A 200 may represent a
        # multi-gigabyte model body and is never consumed, even partially.
        if response.status != 206:
            raise IntegrityError(f"range request returned HTTP {response.status}, not 206")
        expected_range = f"bytes {start}-{end}/{total}"
        if _header_value(response.headers, "Content-Range") != expected_range:
            raise IntegrityError("range response has an unexpected Content-Range")
        expected_length = end - start + 1
        length = _header_value(response.headers, "Content-Length")
        if length is None or int(length) != expected_length:
            raise IntegrityError("range response has an unexpected Content-Length")
        body = response.read(expected_length + 1)
        if len(body) != expected_length:
            raise IntegrityError("range response body length is not exact")
        return body
    finally:
        response.close()


def fetch_safetensors_header(
    url: str,
    total: int,
    *,
    timeout: float = 30.0,
    opener: OpenRequest = _default_open,
) -> tuple[bytes, dict[str, Any]]:
    if total <= 8:
        raise IntegrityError("safetensors file is too small")
    prefix = fetch_exact_range(url, 0, 7, total, timeout=timeout, opener=opener)
    header_length = struct.unpack("<Q", prefix)[0]
    if header_length <= 1 or header_length > MAX_HEADER_BYTES:
        raise IntegrityError("safetensors header length is outside the safety bounds")
    if 8 + header_length > total:
        raise IntegrityError("safetensors header extends beyond the declared LFS object")
    header = fetch_exact_range(
        url,
        8,
        7 + header_length,
        total,
        timeout=timeout,
        opener=opener,
    )
    if contains_forbidden_body(header):
        raise IntegrityError("safetensors header matched prohibited dataset terminology")
    try:
        parsed = json.loads(header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError("safetensors header is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise IntegrityError("safetensors header root is not an object")
    return prefix + header, parsed


def _load_root_identity(root: Path, tournament_id: str) -> dict[str, Any]:
    value = json.loads(_read_regular(root, "ROOT-IDENTITY.json", 1024 * 1024))
    if not isinstance(value, dict) or value.get("tournament_id") != tournament_id:
        raise IntegrityError("input root identity does not match the selected tournament")
    return value


def _tree_observations(root: Path, tournament_id: str) -> Iterable[tuple[str, dict[str, Any], bytes]]:
    prefix = root / "observations" / "hf-tree" / "gradients-io-tournaments"
    if not prefix.is_dir():
        return
    marker = f"tournament-{tournament_id}-"
    for path in sorted(prefix.rglob("*.json")):
        relative = path.relative_to(root).as_posix()
        if marker not in relative:
            continue
        wrapper_body = _read_regular(root, relative, 4 * 1024 * 1024)
        wrapper_sha = sha256_bytes(wrapper_body)
        try:
            wrapper = json.loads(wrapper_body)
        except json.JSONDecodeError as exc:
            raise IntegrityError("hf-tree observation wrapper is not JSON") from exc
        if not isinstance(wrapper, dict) or wrapper.get("source") != "hf-tree":
            raise IntegrityError("hf-tree observation has an invalid source")
        digest = wrapper.get("content_sha256")
        if not isinstance(digest, str) or wrapper.get("object") != _cas_relative(digest):
            raise IntegrityError("hf-tree observation has an invalid CAS binding")
        body = _cas_bytes(root, digest)
        yield relative, {**wrapper, "observation_sha256": wrapper_sha}, body


def _tree_identity(wrapper: Mapping[str, Any], tournament_id: str) -> tuple[str, str, str]:
    key = wrapper.get("key")
    if not isinstance(key, str):
        raise IntegrityError("hf-tree observation has no key")
    parts = key.split("/")
    if len(parts) < 4:
        raise IntegrityError("hf-tree key is malformed")
    repo = "/".join(parts[:2])
    revision = parts[2]
    if REVISION_RE.fullmatch(revision) is None:
        raise IntegrityError("hf-tree key does not bind an immutable revision")
    marker = f"tournament-{tournament_id}-"
    if marker not in repo:
        raise IntegrityError("hf-tree repository is outside the selected tournament")
    task_match = TASK_RE.search(repo)
    if task_match is None:
        raise IntegrityError("hf-tree repository does not identify a task")
    return repo, revision, task_match.group(0)


def collect_candidates(root: Path, tournament_id: str) -> list[dict[str, Any]]:
    """Derive unique public safetensors objects from verified tree snapshots."""
    candidates: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for observation_path, wrapper, tree_body in _tree_observations(root, tournament_id):
        repo, revision, task_id = _tree_identity(wrapper, tournament_id)
        try:
            rows = json.loads(tree_body)
        except json.JSONDecodeError as exc:
            raise IntegrityError("hf-tree CAS object is not JSON") from exc
        if not isinstance(rows, list):
            raise IntegrityError("hf-tree response is not a list")
        for row in rows:
            if not isinstance(row, dict) or row.get("type") != "file":
                continue
            path = row.get("path")
            lfs = row.get("lfs")
            if not isinstance(path, str) or not path.lower().endswith(".safetensors"):
                continue
            if contains_forbidden_path(path):
                # Never request a prohibited path, even when a compromised or
                # unexpectedly broad upstream tree snapshot contains one.
                continue
            if not isinstance(lfs, dict):
                raise IntegrityError("safetensors tree row lacks LFS identity")
            oid = lfs.get("oid")
            size = lfs.get("size")
            if not isinstance(oid, str) or SHA256_RE.fullmatch(oid) is None:
                raise IntegrityError("safetensors LFS SHA-256 is invalid")
            if not isinstance(size, int) or size <= 8:
                raise IntegrityError("safetensors LFS size is invalid")
            key = (repo, revision, path, oid)
            candidate = {
                "repository": repo,
                "revision": revision,
                "task_id": task_id,
                "path": path,
                "lfs_sha256": oid,
                "lfs_bytes": size,
                "tree_observation": observation_path,
                "tree_observation_sha256": wrapper["observation_sha256"],
                "tree_content_sha256": wrapper["content_sha256"],
            }
            previous = candidates.get(key)
            if previous is not None and previous != candidate:
                raise IntegrityError("duplicate checkpoint identity has conflicting provenance")
            candidates[key] = candidate
    return [candidates[key] for key in sorted(candidates)]


def _tensor_summary(parsed: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    metadata = parsed.get("__metadata__")
    tensors: list[dict[str, Any]] = []
    for name, value in sorted(parsed.items()):
        if name == "__metadata__":
            continue
        if not isinstance(value, dict):
            raise IntegrityError("safetensors tensor entry is not an object")
        dtype = value.get("dtype")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list):
            raise IntegrityError("safetensors tensor entry is incomplete")
        if not all(isinstance(item, int) and item >= 0 for item in shape):
            raise IntegrityError("safetensors tensor shape is invalid")
        if len(offsets) != 2 or not all(isinstance(item, int) and item >= 0 for item in offsets):
            raise IntegrityError("safetensors tensor offsets are invalid")
        tensors.append({"name": name, "dtype": dtype, "shape": shape, "data_offsets": offsets})
    if not tensors:
        raise IntegrityError("safetensors header contains no tensors")
    return metadata, tensors


def harvest(
    input_root: Path,
    output_root: Path,
    tournament_id: str,
    *,
    observed_at: dt.datetime,
    timeout: float = 30.0,
    opener: OpenRequest = _default_open,
) -> dict[str, Any]:
    if TOURNAMENT_RE.fullmatch(tournament_id) is None:
        raise SyncError("invalid tournament ID")
    _load_root_identity(input_root, tournament_id)
    candidates = collect_candidates(input_root, tournament_id)
    publisher = Publisher(output_root)

    identity = canonical_json(
        {
            "schema": "sn56.week7.public-safetensors-header-root",
            "schema_version": 1,
            "input_root": str(input_root.absolute()),
            "tournament_id": tournament_id,
        }
    )
    staged_identity = publisher.stage_bytes(identity)
    publisher.publish(staged_identity, "ROOT-IDENTITY.json")
    staged_identity.discard()

    associations: list[dict[str, Any]] = []
    objects_seen: set[str] = set()
    for candidate in candidates:
        repo = candidate["repository"]
        revision = candidate["revision"]
        path = candidate["path"]
        url = (
            "https://huggingface.co/"
            + quote(repo, safe="/")
            + "/resolve/"
            + revision
            + "/"
            + quote(path, safe="/")
        )
        raw_header, parsed = fetch_safetensors_header(
            url,
            candidate["lfs_bytes"],
            timeout=timeout,
            opener=opener,
        )
        header_sha = sha256_bytes(raw_header)
        raw_stage = publisher.stage_bytes(raw_header)
        publisher.publish(raw_stage, _cas_relative(header_sha))
        raw_stage.discard()
        metadata, tensors = _tensor_summary(parsed)
        record = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "lfs_sha256": candidate["lfs_sha256"],
            "lfs_bytes": candidate["lfs_bytes"],
            "header_sha256": header_sha,
            "header_bytes": len(raw_header),
            "header_object": _cas_relative(header_sha),
            "metadata": metadata,
            "tensor_count": len(tensors),
            "tensors": tensors,
        }
        record_body = canonical_json(record)
        record_sha = sha256_bytes(record_body)
        record_relative = f"records/{candidate['lfs_sha256'][:2]}/{candidate['lfs_sha256']}.json"
        record_stage = publisher.stage_bytes(record_body)
        publisher.publish(record_stage, record_relative)
        record_stage.discard()
        objects_seen.add(candidate["lfs_sha256"])
        associations.append(
            {
                **candidate,
                "resolve_url": url,
                "record": record_relative,
                "record_sha256": record_sha,
                "header_sha256": header_sha,
            }
        )

    stamp = utc_token(observed_at)
    index = {
        "schema": f"{SCHEMA}.inventory",
        "schema_version": SCHEMA_VERSION,
        "observed_at": utc_iso(observed_at),
        "tournament_id": tournament_id,
        "candidate_count": len(candidates),
        "unique_lfs_objects": len(objects_seen),
        "associations": associations,
    }
    index_body = canonical_json(index)
    index_stage = publisher.stage_bytes(index_body)
    index_relative = f"inventories/{stamp}.json"
    publisher.publish(index_stage, index_relative)
    index_stage.discard()
    return {
        "status": "COMPLETE",
        "tournament_id": tournament_id,
        "candidate_count": len(candidates),
        "unique_lfs_objects": len(objects_seen),
        "inventory": index_relative,
        "inventory_sha256": sha256_bytes(index_body),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tournament-id", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = harvest(
            args.input_root,
            args.output_root,
            args.tournament_id,
            observed_at=utc_now(),
            timeout=args.timeout,
        )
    except (OSError, ValueError, SyncError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
