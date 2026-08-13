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
from pathlib import Path, PurePosixPath
import re
import struct
import sys
from typing import Any, Callable, Iterable, Mapping, Protocol
from urllib.parse import quote, unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

try:
    from safe_harvest_sync import (
        IntegrityError,
        Publisher,
        SyncError,
        canonical_json,
        checked_relative,
        contains_forbidden_body,
        contains_forbidden_path,
        sha256_bytes,
        observation_timestamp,
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
        checked_relative,
        contains_forbidden_body,
        contains_forbidden_path,
        sha256_bytes,
        observation_timestamp,
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
HF_ORGANIZATION = "gradients-io-tournaments"
TREE_PAGE_RE = re.compile(r"page-[0-9]+")
OBSERVATION_FILE_RE = re.compile(
    r"[0-9]{8}T[0-9]{6}\.[0-9]{6}Z-([0-9a-f]{12}|unchanged)\.json"
)


class Response(Protocol):
    status: int
    headers: Mapping[str, str]

    def read(self, amount: int = -1) -> bytes: ...
    def close(self) -> None: ...


OpenRequest = Callable[[Request, float], Response]


def _validate_hf_transport_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise IntegrityError("public safetensors transport URL is malformed") from exc
    host = (parsed.hostname or "").casefold()
    if (
        parsed.scheme.casefold() != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or not (host == "huggingface.co" or host.endswith(".hf.co"))
    ):
        raise IntegrityError("public safetensors transport left the Hugging Face HTTPS boundary")


class _ValidatedHFRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        _validate_hf_transport_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_HF_OPENER = build_opener(_ValidatedHFRedirects())


def _default_open(request: Request, timeout: float) -> Response:
    _validate_hf_transport_url(request.full_url)
    response = _HF_OPENER.open(request, timeout=timeout)
    final_url = response.geturl()
    _validate_hf_transport_url(final_url)
    return response  # type: ignore[return-value]


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


def _canonical_relative(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise IntegrityError(f"{label} is not a string")
    try:
        normalized = checked_relative(value).as_posix()
    except SyncError as exc:
        raise IntegrityError(f"{label} is not a safe relative path") from exc
    if normalized != value:
        raise IntegrityError(f"{label} is not a canonical relative path")
    return normalized


def _checked_checkpoint_path(value: Any) -> str:
    """Return one unambiguous relative tree path, rejecting encoded traversal."""
    path = _canonical_relative(value, "safetensors tree path")
    decoded = path
    # A quoted tree path is quoted once more when placed into the resolve URL.
    # Check every possible unquoting layer so nested %25 encodings cannot turn
    # into a dot segment, absolute path, backslash, or NUL downstream.
    for _ in range(len(path) + 1):
        next_decoded = unquote(decoded)
        if next_decoded == decoded:
            return path
        _canonical_relative(next_decoded, "decoded safetensors tree path")
        decoded = next_decoded
    raise IntegrityError("safetensors tree path has excessive nested encoding")


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


def _tree_identity(
    wrapper: Mapping[str, Any], tournament_id: str, observation_path: str
) -> tuple[str, str, str]:
    key = _canonical_relative(wrapper.get("key"), "hf-tree key")
    parts = key.split("/")
    if (
        len(parts) != 4
        or parts[0] != HF_ORGANIZATION
        or TREE_PAGE_RE.fullmatch(parts[3]) is None
    ):
        raise IntegrityError("hf-tree key is malformed or outside the public tournament organization")
    if int(parts[3].removeprefix("page-")) < 1:
        raise IntegrityError("hf-tree page numbering must start at one")
    revision = parts[2]
    if REVISION_RE.fullmatch(revision) is None:
        raise IntegrityError("hf-tree key does not bind an immutable revision")

    repo_name = parts[1]
    marker = f"tournament-{tournament_id}-"
    if not repo_name.startswith(marker):
        raise IntegrityError("hf-tree repository is outside the selected tournament")
    repository_tail = repo_name[len(marker) :]
    if len(repository_tail) <= 37 or repository_tail[36] != "-":
        raise IntegrityError("hf-tree repository does not identify an exact task and owner")
    task_id = repository_tail[:36]
    owner = repository_tail[37:]
    if TASK_RE.fullmatch(task_id) is None or re.fullmatch(
        r"[1-9A-HJ-NP-Za-km-z]{8}", owner
    ) is None:
        raise IntegrityError("hf-tree repository does not identify an exact task and owner")

    expected_parent = f"observations/hf-tree/{key}"
    observed = PurePosixPath(observation_path)
    if observed.parent.as_posix() != expected_parent:
        raise IntegrityError("hf-tree key does not match its observation path")
    if OBSERVATION_FILE_RE.fullmatch(observed.name) is None:
        raise IntegrityError("hf-tree observation filename is not append-only watcher form")
    return f"{HF_ORGANIZATION}/{repo_name}", revision, task_id


def _load_tree_observation(
    root: Path, observation_path: str, tournament_id: str
) -> tuple[dict[str, Any], bytes, tuple[str, str, str]]:
    relative = _canonical_relative(observation_path, "hf-tree observation path")
    wrapper_body = _read_regular(root, relative, 4 * 1024 * 1024)
    wrapper_sha = sha256_bytes(wrapper_body)
    try:
        wrapper = json.loads(wrapper_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError("hf-tree observation wrapper is not JSON") from exc
    if not isinstance(wrapper, dict) or wrapper.get("source") != "hf-tree":
        raise IntegrityError("hf-tree observation has an invalid source")
    identity = _tree_identity(wrapper, tournament_id, relative)
    normalized_observed_at = observation_timestamp(relative, wrapper)
    digest = wrapper.get("content_sha256")
    if not isinstance(digest, str) or wrapper.get("object") != _cas_relative(digest):
        raise IntegrityError("hf-tree observation has an invalid CAS binding")
    filename_match = OBSERVATION_FILE_RE.fullmatch(PurePosixPath(relative).name)
    if filename_match is None or filename_match.group(1) not in {
        digest[:12],
        "unchanged",
    }:
        raise IntegrityError("hf-tree observation filename does not bind its content digest")
    body = _cas_bytes(root, digest)
    content_bytes = wrapper.get("content_bytes")
    if (
        not isinstance(content_bytes, int)
        or isinstance(content_bytes, bool)
        or content_bytes != len(body)
    ):
        raise IntegrityError("hf-tree observation has an invalid content length")
    return {
        **wrapper,
        "observed_at": normalized_observed_at,
        "observation_sha256": wrapper_sha,
    }, body, identity


def _tree_observations(root: Path, tournament_id: str) -> Iterable[tuple[str, dict[str, Any], bytes]]:
    prefix = root / "observations" / "hf-tree" / HF_ORGANIZATION
    if not prefix.is_dir():
        return
    marker = f"tournament-{tournament_id}-"
    for path in sorted(prefix.rglob("*.json")):
        relative = _canonical_relative(path.relative_to(root).as_posix(), "hf-tree observation path")
        parts = relative.split("/")
        if len(parts) < 4 or not parts[3].startswith(marker):
            continue
        wrapper, body, _ = _load_tree_observation(root, relative, tournament_id)
        yield relative, wrapper, body


def _tree_rows(tree_body: bytes) -> list[Any]:
    try:
        rows = json.loads(tree_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError("hf-tree CAS object is not JSON") from exc
    if not isinstance(rows, list):
        raise IntegrityError("hf-tree response is not a list")
    return rows


def _safetensors_tree_entry(row: Any) -> tuple[str, str, int] | None:
    if not isinstance(row, dict):
        raise IntegrityError("hf-tree response contains a malformed entry")
    if row.get("type") != "file":
        return None
    raw_path = row.get("path")
    if not isinstance(raw_path, str):
        raise IntegrityError("hf-tree file entry has no path")
    if not raw_path.lower().endswith(".safetensors"):
        return None
    path = _checked_checkpoint_path(raw_path)
    lfs = row.get("lfs")
    if not isinstance(lfs, dict):
        raise IntegrityError("safetensors tree row lacks LFS identity")
    oid = lfs.get("oid")
    size = lfs.get("size")
    row_size = row.get("size")
    if not isinstance(oid, str) or SHA256_RE.fullmatch(oid) is None:
        raise IntegrityError("safetensors LFS SHA-256 is invalid")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 8
        or not isinstance(row_size, int)
        or isinstance(row_size, bool)
        or row_size != size
    ):
        raise IntegrityError("safetensors tree/LFS size is invalid or conflicting")
    return path, oid, size


def _provenance_sort_key(value: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(value.get("observed_at", "")),
        str(value.get("tree_observation", "")),
        str(value.get("tree_observation_sha256", "")),
        str(value.get("tree_content_sha256", "")),
    )


def collect_candidates(
    root: Path, tournament_id: str, allowed_task_ids: set[str]
) -> list[dict[str, Any]]:
    """Derive unique public safetensors objects from verified tree snapshots."""
    candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
    pages: dict[tuple[str, str], dict[int, str]] = {}
    for observation_path, wrapper, tree_body in _tree_observations(root, tournament_id):
        repo, revision, task_id = _tree_identity(wrapper, tournament_id, observation_path)
        if task_id not in allowed_task_ids:
            # The source root can continue growing after the selected round.
            # Never turn a same-tournament but out-of-scope task into a Range
            # request merely because its public tree was appended locally.
            continue
        page = int(str(wrapper["key"]).rsplit("/", 1)[1].removeprefix("page-"))
        page_digest = wrapper["content_sha256"]
        prior_page = pages.setdefault((repo, revision), {}).get(page)
        if prior_page is not None and prior_page != page_digest:
            raise IntegrityError("hf-tree page has conflicting repeated content")
        pages[(repo, revision)][page] = page_digest
        for row in _tree_rows(tree_body):
            entry = _safetensors_tree_entry(row)
            if entry is None:
                continue
            path, oid, size = entry
            if contains_forbidden_path(path):
                # Never request a prohibited path, even when a compromised or
                # unexpectedly broad upstream tree snapshot contains one.
                continue
            key = (repo, revision, path)
            provenance = {
                "tree_observation": observation_path,
                "tree_observation_sha256": wrapper["observation_sha256"],
                "tree_content_sha256": wrapper["content_sha256"],
                "observed_at": wrapper.get("observed_at")
                if isinstance(wrapper.get("observed_at"), str)
                else "",
            }
            previous = candidates.get(key)
            if previous is None:
                previous = {
                    "repository": repo,
                    "revision": revision,
                    "task_id": task_id,
                    "path": path,
                    "lfs_sha256": oid,
                    "lfs_bytes": size,
                    "tree_observations": [],
                }
                candidates[key] = previous
            elif previous["lfs_sha256"] != oid or previous["lfs_bytes"] != size:
                raise IntegrityError("duplicate checkpoint path has conflicting LFS size or OID")
            if provenance not in previous["tree_observations"]:
                previous["tree_observations"].append(provenance)

    for observed_pages in pages.values():
        if sorted(observed_pages) != list(range(1, max(observed_pages) + 1)):
            raise IntegrityError("hf-tree page set is not contiguous from page one")

    result: list[dict[str, Any]] = []
    for key in sorted(candidates):
        candidate = candidates[key]
        provenance = sorted(candidate["tree_observations"], key=_provenance_sort_key)
        latest = provenance[-1]
        candidate["tree_observations"] = provenance
        candidate.update(
            {
                "tree_observation": latest["tree_observation"],
                "tree_observation_sha256": latest["tree_observation_sha256"],
                "tree_content_sha256": latest["tree_content_sha256"],
            }
        )
        result.append(candidate)
    return result


def _validate_candidate_association(
    root: Path, tournament_id: str, candidate: Mapping[str, Any]
) -> str:
    """Re-bind an association to each referenced wrapper, CAS, and tree row."""
    path = _checked_checkpoint_path(candidate.get("path"))
    repo = candidate.get("repository")
    revision = candidate.get("revision")
    task_id = candidate.get("task_id")
    oid = candidate.get("lfs_sha256")
    size = candidate.get("lfs_bytes")
    provenance = candidate.get("tree_observations")
    if contains_forbidden_path(path):
        raise IntegrityError("checkpoint association path is prohibited")
    if (
        not isinstance(repo, str)
        or not isinstance(revision, str)
        or not isinstance(task_id, str)
        or not isinstance(oid, str)
        or SHA256_RE.fullmatch(oid) is None
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 8
    ):
        raise IntegrityError("checkpoint association identity is malformed")
    if (
        not isinstance(provenance, list)
        or not provenance
        or not all(isinstance(item, dict) for item in provenance)
    ):
        raise IntegrityError("checkpoint association has no tree provenance")
    ordered = sorted(provenance, key=_provenance_sort_key)
    if provenance != ordered:
        raise IntegrityError("checkpoint association tree provenance is not canonical")
    latest = ordered[-1]
    if any(
        candidate.get(field) != latest.get(field)
        for field in (
            "tree_observation",
            "tree_observation_sha256",
            "tree_content_sha256",
        )
    ):
        raise IntegrityError("checkpoint association latest tree provenance is inconsistent")

    for item in provenance:
        observation_path = _canonical_relative(
            item.get("tree_observation"), "checkpoint tree observation path"
        )
        wrapper, tree_body, identity = _load_tree_observation(
            root, observation_path, tournament_id
        )
        if identity != (repo, revision, task_id):
            raise IntegrityError("checkpoint association contradicts its tree repository identity")
        if (
            wrapper["observation_sha256"] != item.get("tree_observation_sha256")
            or wrapper["content_sha256"] != item.get("tree_content_sha256")
            or (
                wrapper.get("observed_at")
                if isinstance(wrapper.get("observed_at"), str)
                else ""
            )
            != item.get("observed_at")
        ):
            raise IntegrityError("checkpoint association contradicts its tree provenance fields")

        matched = False
        for row in _tree_rows(tree_body):
            entry = _safetensors_tree_entry(row)
            if entry is None or entry[0] != path:
                continue
            matched = True
            if entry[1] != oid or entry[2] != size:
                raise IntegrityError("checkpoint association contradicts its referenced tree entry")
        if not matched:
            raise IntegrityError("checkpoint association path is absent from its referenced tree")
    return path


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
    task_ids: Iterable[str],
    observed_at: dt.datetime,
    timeout: float = 30.0,
    opener: OpenRequest = _default_open,
) -> dict[str, Any]:
    if TOURNAMENT_RE.fullmatch(tournament_id) is None:
        raise SyncError("invalid tournament ID")
    allowed_task_ids = set(task_ids)
    if not allowed_task_ids or any(
        not isinstance(task_id, str) or TASK_RE.fullmatch(task_id) is None
        for task_id in allowed_task_ids
    ):
        raise SyncError("an exact non-empty task-ID allowlist is required")
    _load_root_identity(input_root, tournament_id)
    candidates = collect_candidates(input_root, tournament_id, allowed_task_ids)
    publisher = Publisher(output_root)

    identity = canonical_json(
        {
            "schema": "sn56.week7.public-safetensors-header-root",
            "schema_version": 1,
            "input_root": str(input_root.absolute()),
            "tournament_id": tournament_id,
            "task_ids": sorted(allowed_task_ids),
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
        # Re-read the source evidence immediately before URL construction so
        # no association can reach a Range GET unless its exact wrapper, CAS,
        # and tree entry still bind repo/revision/path/OID/size.
        path = _validate_candidate_association(input_root, tournament_id, candidate)
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
        "task_ids": sorted(allowed_task_ids),
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
    parser.add_argument(
        "--task-id",
        action="append",
        required=True,
        dest="task_ids",
        help="exact in-scope task UUID; repeat once per selected task",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = harvest(
            args.input_root,
            args.output_root,
            args.tournament_id,
            task_ids=args.task_ids,
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
