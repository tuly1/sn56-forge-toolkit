#!/usr/bin/env python3
"""Capture and inventory only the public optimizer-visible training archives.

This collector deliberately has a much narrower authority than the tournament
watcher.  Its input is a checksum-verified final safe-API snapshot whose image
tournament Round 1 is complete.  That snapshot supplies the exact Round-1 task
IDs; the collector requests only each corresponding public auditing endpoint
and, from the full response held in memory, selects only ``training_data``.

The public task endpoint returns one JSON envelope that may also contain
``test_data`` and ``image_text_pairs`` metadata.  Those fields are never
selected, persisted, or dereferenced; only ``training_data`` can authorize a
second request. Downloaded training ZIPs are bounded, hashed while streaming
into a private temporary file, validated and inventoried without extraction,
then published into a create-only SHA-256 CAS. The output is research evidence,
never fixture admission: licensing and third-party rights remain explicitly
unverified.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import unicodedata
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid
import zipfile

try:
    from safe_harvest_sync import (
        IntegrityError,
        Publisher,
        Staged,
        SyncError,
        _check_root_components,
        _open_regular_beneath,
        canonical_json,
        contains_forbidden_path,
        sha256_bytes,
        utc_iso,
        utc_now,
        utc_token,
    )
except ImportError:  # pragma: no cover - supports package-style imports
    from .safe_harvest_sync import (
        IntegrityError,
        Publisher,
        Staged,
        SyncError,
        _check_root_components,
        _open_regular_beneath,
        canonical_json,
        contains_forbidden_path,
        sha256_bytes,
        utc_iso,
        utc_now,
        utc_token,
    )


SCHEMA = "sn56.week7.public-training-archive"
SCHEMA_VERSION = 1
INVENTORY_SCHEMA = f"{SCHEMA}.inventory"
RECEIPT_SCHEMA = f"{SCHEMA}.receipt"
ROOT_SCHEMA = f"{SCHEMA}.root"
SAFE_API_SCHEMA = "sn56.week7.safe-public-api-snapshot"
SAFE_API_SCHEMA_VERSION = 1
API_ROOT = "https://api.gradients.io"
TOURNAMENT_RE = re.compile(r"tourn_[a-z0-9]+_[0-9]{8}")
TASK_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
TERMINAL_TASK_STATUSES = frozenset(
    {
        "success",
        "completed",
        "failed",
        "failure",
        "terminated",
        "cancelled",
        "canceled",
        "error",
    }
)
IMAGE_SUFFIXES = frozenset({".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
CAPTION_SUFFIXES = frozenset({".txt"})

# These ceilings are intentionally independent.  A small compressed ZIP may
# expand dramatically, and one oversized member is independently suspicious.
MAX_API_BYTES = 64 * 1024 * 1024
MAX_COMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 8192
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
CHUNK = 1024 * 1024
TRAINING_ARCHIVE_HOSTS = frozenset({"s3.eu-central-003.backblazeb2.com"})

RIGHTS = {
    "public_access": "observed",
    "license_and_third_party_rights": "unverified",
    "allowed_use": "research-analysis-only",
    "fixture_admission": "not admitted",
}


class Response(Protocol):
    status: int
    headers: Mapping[str, str]

    def read(self, amount: int = -1) -> bytes: ...
    def close(self) -> None: ...


OpenRequest = Callable[[Request, float], Response]


class _RejectRedirects(HTTPRedirectHandler):
    """Reject redirects before urllib can issue a request to the new target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        del req, fp, code, msg, headers, newurl
        raise IntegrityError("public evidence request attempted a redirect")


_NO_REDIRECT_OPENER = build_opener(_RejectRedirects())


def _default_open(request: Request, timeout: float) -> Response:
    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)  # type: ignore[return-value]


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == wanted:
            return str(value)
    return None


def _read_regular(path: Path, limit: int) -> bytes:
    """Read one regular file without following any path-component symlink."""
    absolute = path.absolute()
    parent = _check_root_components(absolute.parent)
    fd = _open_regular_beneath(parent, absolute.name)
    try:
        body = bytearray()
        while True:
            remaining = limit + 1 - len(body)
            block = os.read(fd, min(CHUNK, max(1, remaining)))
            if not block:
                return bytes(body)
            body.extend(block)
            if len(body) > limit:
                raise IntegrityError("input exceeds its safety ceiling")
    finally:
        os.close(fd)


def _load_checked_snapshot(
    snapshot_path: Path,
    tournament_id: str,
    checksum_path: Path | None = None,
) -> tuple[dict[str, Any], str, list[str]]:
    """Verify snapshot bytes/schema/identity and return exact completed-R1 tasks."""
    if TOURNAMENT_RE.fullmatch(tournament_id) is None:
        raise SyncError("invalid tournament ID")
    snapshot = _read_regular(snapshot_path, MAX_API_BYTES)
    digest = sha256_bytes(snapshot)
    sidecar_path = checksum_path or snapshot_path.with_suffix(".sha256")
    sidecar = _read_regular(sidecar_path, 1024).decode("ascii", errors="strict")
    match = re.fullmatch(r"([0-9a-f]{64})  ([^/\0\r\n]+)\n?", sidecar)
    if match is None or match.group(2) != snapshot_path.name:
        raise IntegrityError("safe API checksum sidecar is malformed or targets another file")
    if match.group(1) != digest:
        raise IntegrityError("safe API snapshot checksum mismatch")
    try:
        value = json.loads(
            snapshot,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise IntegrityError("safe API snapshot is not JSON") from exc
    if not isinstance(value, dict):
        raise IntegrityError("safe API snapshot root is not an object")
    if (
        value.get("schema") != SAFE_API_SCHEMA
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != SAFE_API_SCHEMA_VERSION
    ):
        raise IntegrityError("safe API snapshot schema/version is unsupported")
    tournament = value.get("tournament")
    if (
        not isinstance(tournament, dict)
        or tournament.get("tournament_id") != tournament_id
        or tournament.get("tournament_type") != "image"
    ):
        raise IntegrityError("safe API snapshot tournament identity/type mismatch")
    rounds = tournament.get("rounds")
    if not isinstance(rounds, list):
        raise IntegrityError("safe API snapshot has no rounds list")
    round_one = [
        row
        for row in rounds
        if isinstance(row, dict)
        and isinstance(row.get("round_number"), int)
        and not isinstance(row.get("round_number"), bool)
        and row.get("round_number") == 1
    ]
    if len(round_one) != 1 or str(round_one[0].get("status", "")).casefold() != "completed":
        raise IntegrityError("safe API snapshot Round 1 is not uniquely completed")
    tasks = round_one[0].get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise IntegrityError("completed Round 1 has no task list")
    task_ids: list[str] = []
    for row in tasks:
        task_id = row.get("task_id") if isinstance(row, dict) else None
        if not isinstance(task_id, str) or TASK_RE.fullmatch(task_id) is None:
            raise IntegrityError("completed Round 1 contains an invalid task ID")
        if row.get("task_type") != "ImageTask":
            raise IntegrityError("completed Round 1 contains a non-image task")
        task_ids.append(task_id)
    if len(set(task_ids)) != len(task_ids):
        raise IntegrityError("completed Round 1 contains duplicate task IDs")

    # The sanitized task-detail section must agree with the round membership.
    details = value.get("tasks")
    if not isinstance(details, list):
        raise IntegrityError("safe API snapshot has no task-detail list")
    detail_by_id: dict[str, dict[str, Any]] = {}
    for row in details:
        if not isinstance(row, dict) or not isinstance(row.get("task_id"), str):
            raise IntegrityError("safe API snapshot contains a malformed task detail")
        detail_id = row["task_id"]
        if detail_id in detail_by_id:
            raise IntegrityError("safe API snapshot contains duplicate task details")
        detail_by_id[detail_id] = row
    if any(task_id not in detail_by_id for task_id in task_ids):
        raise IntegrityError("safe API snapshot task details do not cover exact Round 1")
    for task_id in task_ids:
        if detail_by_id[task_id].get("task_type") != "ImageTask":
            raise IntegrityError("safe API task detail is not an image task")
    return value, digest, sorted(task_ids)


def _fetch_json(
    url: str,
    *,
    timeout: float,
    opener: OpenRequest,
) -> tuple[dict[str, Any], str, int]:
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "sn56-week7-training-archive/1"},
        method="GET",
    )
    try:
        response = opener(request, timeout)
    except OSError as exc:
        raise IntegrityError("exact public task endpoint request failed") from exc
    try:
        if response.status != 200:
            raise IntegrityError(f"exact public task endpoint returned HTTP {response.status}")
        body = response.read(MAX_API_BYTES + 1)
        if len(body) > MAX_API_BYTES:
            raise IntegrityError("exact public task response exceeds the safety ceiling")
    finally:
        response.close()
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError("exact public task response is not JSON") from exc
    if not isinstance(value, dict):
        raise IntegrityError("exact public task response root is not an object")
    return value, sha256_bytes(body), len(body)


def _select_training_url(value: Mapping[str, Any], task_id: str) -> str | None:
    """Validate the exact terminal task and select *only* ``training_data``."""
    if value.get("task_id") != task_id:
        raise IntegrityError("public task response identity mismatch")
    status = value.get("status")
    if not isinstance(status, str) or status.casefold() not in TERMINAL_TASK_STATUSES:
        raise IntegrityError("public task response is not terminal")
    # This is intentionally the only dataset-bearing key accessed in the full
    # in-memory task response.  In particular, image_text_pairs and test_data
    # remain opaque and cannot influence a request.
    training_data = value.get("training_data")
    if training_data is None or training_data == "":
        return None
    if not isinstance(training_data, str):
        raise IntegrityError("public training_data is neither a URL nor absent")
    return training_data


def _validated_source_url(url: str) -> tuple[str, str]:
    """Return request URL and provenance URL stripped of query/userinfo."""
    if any(ord(char) < 32 or char.isspace() for char in url):
        raise IntegrityError("public training_data URL contains whitespace/control bytes")
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise IntegrityError("public training_data URL is malformed") from exc
    if parsed.scheme.casefold() != "https" or not host or not parsed.path:
        raise IntegrityError("public training_data must be an absolute HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise IntegrityError("public training_data URL must not contain userinfo")
    if port not in (None, 443):
        raise IntegrityError("public training_data URL must use the default HTTPS port")
    if host.casefold() not in TRAINING_ARCHIVE_HOSTS:
        raise IntegrityError("public training_data URL uses an unapproved storage authority")
    if parsed.fragment:
        raise IntegrityError("public training_data URL must not contain a fragment")
    if contains_forbidden_path(parsed.path):
        raise IntegrityError("public training_data URL path matches a prohibited dataset surface")
    redacted = urlunsplit(("https", host.casefold(), parsed.path, "", ""))
    return url, redacted


def _download_to_private_stage(
    publisher: Publisher,
    url: str,
    *,
    timeout: float,
    opener: OpenRequest,
) -> tuple[Staged, dict[str, Any]]:
    request_url, redacted_url = _validated_source_url(url)
    request = Request(
        request_url,
        headers={
            "Accept": "application/zip, application/octet-stream",
            "Accept-Encoding": "identity",
            "User-Agent": "sn56-week7-training-archive/1",
        },
        method="GET",
    )
    try:
        response = opener(request, timeout)
    except OSError as exc:
        # The presigned query may contain credentials.  Keep it out of the
        # stable error message even when the transport exception repeats it.
        raise IntegrityError("public training archive request failed") from exc
    temp_path = publisher.partial / f"{uuid.uuid4().hex}.archive.part"
    fd: int | None = None
    digest = hashlib.sha256()
    size = 0
    try:
        if response.status != 200:
            raise IntegrityError(f"public training archive returned HTTP {response.status}")
        encoding = _header(response.headers, "Content-Encoding")
        if encoding is not None and encoding.casefold() not in {"", "identity"}:
            raise IntegrityError("public training archive used a content encoding")
        declared_raw = _header(response.headers, "Content-Length")
        declared: int | None = None
        if declared_raw is not None:
            try:
                declared = int(declared_raw)
            except ValueError as exc:
                raise IntegrityError("public training archive Content-Length is invalid") from exc
            if declared < 0 or declared > MAX_COMPRESSED_BYTES:
                raise IntegrityError("public training archive exceeds the compressed-size ceiling")
        fd = os.open(
            temp_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            size += len(block)
            if size > MAX_COMPRESSED_BYTES:
                raise IntegrityError("public training archive exceeds the compressed-size ceiling")
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(fd, view)
                if written <= 0:  # pragma: no cover - defensive regular-file guard
                    raise IntegrityError("private archive staging write made no progress")
                view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = None
        if declared is not None and declared != size:
            raise IntegrityError("public training archive Content-Length does not match received bytes")
        if size == 0:
            raise IntegrityError("public training archive is empty")
        staged = Staged(temp_path, digest.hexdigest(), size)
        return staged, {
            "url": redacted_url,
            "http_status": 200,
            "content_type": _header(response.headers, "Content-Type"),
            "declared_content_length": declared,
        }
    except Exception:
        if fd is not None:
            os.close(fd)
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        response.close()


def _safe_member_path(info: zipfile.ZipInfo) -> tuple[str, bool]:
    raw = info.filename
    is_directory = info.is_dir()
    candidate = raw[:-1] if is_directory and raw.endswith("/") else raw
    if (
        not candidate
        or "\x00" in candidate
        or "\\" in candidate
        or candidate.startswith("/")
        or candidate.startswith("~")
    ):
        raise IntegrityError("training ZIP contains an unsafe member path")
    parts = candidate.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise IntegrityError("training ZIP contains an unsafe member path")
    normalized = PurePosixPath(candidate).as_posix()
    if normalized != candidate or contains_forbidden_path(normalized):
        raise IntegrityError("training ZIP contains an unsafe or prohibited member path")
    return normalized, is_directory


def _member_kind(path: str, is_directory: bool) -> str:
    if is_directory:
        return "directory"
    suffix = PurePosixPath(path).suffix.casefold()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in CAPTION_SUFFIXES:
        return "caption"
    return "other"


def _inventory_zip_file(path_or_file: Any) -> dict[str, Any]:
    """Inventory a bounded ZIP without extracting any member to the filesystem."""
    try:
        archive = zipfile.ZipFile(path_or_file, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise IntegrityError("public training archive is not a valid ZIP") from exc
    with archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_ARCHIVE_MEMBERS:
            raise IntegrityError("training ZIP member count is outside the safety bounds")
        validated: list[tuple[zipfile.ZipInfo, str, bool, str]] = []
        seen: set[str] = set()
        total = 0
        for info in infos:
            normalized, is_directory = _safe_member_path(info)
            identity = unicodedata.normalize("NFC", normalized).casefold()
            if identity in seen:
                raise IntegrityError("training ZIP contains duplicate/case-colliding members")
            seen.add(identity)
            if info.flag_bits & 0x1:
                raise IntegrityError("training ZIP contains an encrypted member")
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise IntegrityError("training ZIP contains a symlink")
            file_type = stat.S_IFMT(mode)
            allowed_types = {0, stat.S_IFDIR if is_directory else stat.S_IFREG}
            if file_type not in allowed_types:
                raise IntegrityError("training ZIP contains a non-regular member")
            if info.file_size < 0 or info.compress_size < 0:
                raise IntegrityError("training ZIP member has a negative size")
            if is_directory and info.file_size != 0:
                raise IntegrityError("training ZIP directory has a nonzero body")
            if info.file_size > MAX_MEMBER_BYTES:
                raise IntegrityError("training ZIP member exceeds the per-member ceiling")
            total += info.file_size
            if total > MAX_UNCOMPRESSED_BYTES:
                raise IntegrityError("training ZIP exceeds the uncompressed-size ceiling")
            validated.append((info, normalized, is_directory, _member_kind(normalized, is_directory)))

        members: list[dict[str, Any]] = []
        image_hashes: set[str] = set()
        image_stems: set[tuple[str, str]] = set()
        caption_stems: set[tuple[str, str]] = set()
        image_count = 0
        caption_count = 0
        for index, (info, normalized, is_directory, kind) in enumerate(validated):
            member_sha: str | None = None
            if not is_directory:
                member_digest = hashlib.sha256()
                actual = 0
                try:
                    with archive.open(info, "r") as member:
                        while True:
                            block = member.read(CHUNK)
                            if not block:
                                break
                            actual += len(block)
                            if actual > info.file_size or actual > MAX_MEMBER_BYTES:
                                raise IntegrityError("training ZIP member expanded beyond its declared bound")
                            member_digest.update(block)
                except (OSError, RuntimeError, NotImplementedError, zipfile.BadZipFile) as exc:
                    raise IntegrityError("training ZIP member failed bounded CRC/decompression") from exc
                if actual != info.file_size:
                    raise IntegrityError("training ZIP member size differs from central directory")
                member_sha = member_digest.hexdigest()
                pure = PurePosixPath(normalized)
                stem_key = (pure.parent.as_posix(), pure.stem)
                if kind == "image":
                    image_count += 1
                    image_hashes.add(member_sha)
                    image_stems.add(stem_key)
                elif kind == "caption":
                    caption_count += 1
                    caption_stems.add(stem_key)
            members.append(
                {
                    "central_directory_index": index,
                    "path": normalized,
                    "kind": kind,
                    "uncompressed_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "sha256": member_sha,
                    "crc32": f"{info.CRC:08x}",
                    "compression_method": info.compress_type,
                }
            )
        if image_count == 0:
            raise IntegrityError("training ZIP contains no recognized image members")
        return {
            "member_count": len(members),
            "uncompressed_bytes": total,
            "image_member_count": image_count,
            "caption_member_count": caption_count,
            "paired_stem_count": len(image_stems & caption_stems),
            "post_exact_byte_dedup_image_count": len(image_hashes),
            "members": members,
        }


def _inventory_zip(path: Path) -> dict[str, Any]:
    return _inventory_zip_file(path)


def _publish_bytes(publisher: Publisher, body: bytes, relative: str) -> tuple[str, str]:
    staged = publisher.stage_bytes(body)
    try:
        publisher.publish(staged, relative)
    finally:
        staged.discard()
    return relative, sha256_bytes(body)


def capture(
    snapshot_path: Path,
    output_root: Path,
    tournament_id: str,
    *,
    observed_at: dt.datetime,
    checksum_path: Path | None = None,
    timeout: float = 30.0,
    opener: OpenRequest = _default_open,
) -> dict[str, Any]:
    _, snapshot_sha, task_ids = _load_checked_snapshot(
        snapshot_path, tournament_id, checksum_path
    )
    publisher = Publisher(output_root)
    root_identity = canonical_json(
        {
            "schema": ROOT_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "tournament_id": tournament_id,
            "authority": "exact completed-R1 public training_data URLs only",
        }
    )
    _publish_bytes(publisher, root_identity, "ROOT-IDENTITY.json")

    stamp = utc_token(observed_at)
    task_records: list[dict[str, Any]] = []
    partial = False
    for task_id in task_ids:
        endpoint = f"{API_ROOT}/auditing/tasks/{task_id}"
        task_value, response_sha, response_bytes = _fetch_json(
            endpoint, timeout=timeout, opener=opener
        )
        training_url = _select_training_url(task_value, task_id)
        base_receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "observed_at": utc_iso(observed_at),
            "tournament_id": tournament_id,
            "task_id": task_id,
            "task_status": task_value["status"],
            "source_task_endpoint": endpoint,
            "source_response_sha256": response_sha,
            "source_response_bytes": response_bytes,
            "input_safe_api_snapshot_sha256": snapshot_sha,
            "rights": RIGHTS,
            "exclusion_contract": {
                "selected_field": "training_data only",
                "task_envelope_received_before_allowlist": True,
                "never_selected_persisted_or_dereferenced": [
                    "test_data",
                    "image_text_pairs",
                    "hidden",
                    "holdout",
                    "quarantine",
                    "evaluation rows",
                ],
            },
        }
        if training_url is None:
            partial = True
            receipt = {
                **base_receipt,
                "status": "PARTIAL",
                "reason": "public training_data URL absent",
                "archive": None,
                "inventory": None,
            }
        else:
            staged, transfer = _download_to_private_stage(
                publisher, training_url, timeout=timeout, opener=opener
            )
            try:
                zip_inventory = _inventory_zip(staged.path)
                archive_relative = f"objects/sha256/{staged.digest[:2]}/{staged.digest}"
                publisher.publish(staged, archive_relative)
                receipt = {
                    **base_receipt,
                    "status": "COMPLETE",
                    "reason": None,
                    "source_training_archive": transfer,
                    "archive": {
                        "sha256": staged.digest,
                        "bytes": staged.size,
                        "object": archive_relative,
                    },
                    "inventory": zip_inventory,
                }
            finally:
                staged.discard()
        receipt_body = canonical_json(receipt)
        receipt_relative = f"receipts/{task_id}/{stamp}.json"
        _, receipt_sha = _publish_bytes(publisher, receipt_body, receipt_relative)
        task_records.append(
            {
                "task_id": task_id,
                "status": receipt["status"],
                "receipt": receipt_relative,
                "receipt_sha256": receipt_sha,
                "archive_sha256": receipt["archive"]["sha256"] if receipt["archive"] else None,
                "image_member_count": (
                    receipt["inventory"]["image_member_count"] if receipt["inventory"] else None
                ),
                "post_exact_byte_dedup_image_count": (
                    receipt["inventory"]["post_exact_byte_dedup_image_count"]
                    if receipt["inventory"]
                    else None
                ),
            }
        )

    status = "PARTIAL" if partial else "COMPLETE"
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "observed_at": utc_iso(observed_at),
        "status": status,
        "tournament_id": tournament_id,
        "round_number": 1,
        "round_status": "completed",
        "input_safe_api_snapshot": {
            "filename": snapshot_path.name,
            "sha256": snapshot_sha,
        },
        "task_count": len(task_ids),
        "complete_task_count": sum(row["status"] == "COMPLETE" for row in task_records),
        "partial_task_count": sum(row["status"] == "PARTIAL" for row in task_records),
        "tasks": task_records,
        "rights": RIGHTS,
        "privacy_boundary": (
            "only public optimizer-visible training_data archives were requested; "
            "the task API envelope was received before allowlisting, while image_text_pairs, "
            "test, hidden, and evaluation fields were not selected or persisted and their asset "
            "URLs were never dereferenced"
        ),
    }
    inventory_body = canonical_json(inventory)
    inventory_relative = f"inventories/{stamp}.json"
    _, inventory_sha = _publish_bytes(publisher, inventory_body, inventory_relative)
    sidecar_body = f"{inventory_sha}  {Path(inventory_relative).name}\n".encode("ascii")
    sidecar_relative = f"inventories/{stamp}.sha256"
    _publish_bytes(publisher, sidecar_body, sidecar_relative)
    return {
        "status": status,
        "tournament_id": tournament_id,
        "task_count": len(task_ids),
        "inventory": inventory_relative,
        "inventory_sha256": inventory_sha,
        "checksum": sidecar_relative,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-snapshot", required=True, type=Path)
    parser.add_argument("--api-checksum", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tournament-id", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = capture(
            args.api_snapshot,
            args.output_root,
            args.tournament_id,
            observed_at=utc_now(),
            checksum_path=args.api_checksum,
            timeout=args.timeout,
        )
    except (OSError, UnicodeError, ValueError, SyncError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
