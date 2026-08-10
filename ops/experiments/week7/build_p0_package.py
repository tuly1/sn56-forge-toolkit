#!/usr/bin/env python3
"""Build a hash-bound, offline Week-7 P0 Round-1 evidence package.

The finalizer consumes already-sanitized, public-only evidence surfaces:

* one exact safe-public-API snapshot and its checksum sidecar;
* one exact public-safetensors header inventory; and
* one exact-tournament raw-watcher root with a COMPLETE sync ledger;
* optionally, one checksum-bound exact-R1 public training-archive inventory.

It performs no network requests.  Every consumed local byte string is opened
without following symlinks and checked against its declared SHA-256 identity.
The program refuses to produce a package until the selected tournament is an
image tournament and Round 1 is completed.

Training ZIPs are never extracted or buffered: COMPLETE archive identities are
verified through bounded descriptor-anchored streaming, while only the
collector's hash-bound member inventory is projected.  Other dataset bodies
are outside this tool's contract.  Paths or metadata that match the owner's
hidden/test/holdout/quarantine exclusion vocabulary are rejected before any
referenced object is opened.  The output contains only public task state,
submission metadata, public repository/config/checkpoint provenance, and
cautious byte-identity observations.  In particular, a numbered checkpoint is
called a "final" only when the public bytes establish that fact; normally the
report uses the narrower term ``highest_observed_numbered_step``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlsplit

import yaml

try:
    from capture_public_training_archives import _inventory_zip_file
except ImportError:  # pragma: no cover - direct spec loading in tests/tools
    _capture_path = Path(__file__).with_name("capture_public_training_archives.py")
    _capture_spec = importlib.util.spec_from_file_location(
        "sn56_week7_capture_public_training_archives", _capture_path
    )
    if _capture_spec is None or _capture_spec.loader is None:
        raise
    _capture_module = importlib.util.module_from_spec(_capture_spec)
    sys.modules[_capture_spec.name] = _capture_module
    _capture_spec.loader.exec_module(_capture_module)
    _inventory_zip_file = _capture_module._inventory_zip_file

try:
    from safe_harvest_sync import (
        IntegrityError,
        LocalSource,
        Publisher,
        SyncError,
        canonical_json,
        checked_relative,
        contains_forbidden_body,
        contains_forbidden_path,
        sha256_bytes,
        utc_iso,
        utc_now,
        utc_token,
        validate_observation_identity,
        validate_public_request_provenance,
        observation_timestamp,
        TournamentScope,
        validate_scoped_observation_body,
        _check_root_components,
        _open_regular_beneath,
    )
except ImportError:  # pragma: no cover - package-style imports in tests
    from .safe_harvest_sync import (
        IntegrityError,
        LocalSource,
        Publisher,
        SyncError,
        canonical_json,
        checked_relative,
        contains_forbidden_body,
        contains_forbidden_path,
        sha256_bytes,
        utc_iso,
        utc_now,
        utc_token,
        validate_observation_identity,
        validate_public_request_provenance,
        observation_timestamp,
        TournamentScope,
        validate_scoped_observation_body,
        _check_root_components,
        _open_regular_beneath,
    )


SCHEMA = "sn56.week7.p0-round1-package"
SCHEMA_VERSION = 1
API_SCHEMA = "sn56.week7.safe-public-api-snapshot"
API_SCHEMA_VERSION = 1
WATCHER_ROOT_SCHEMA = "sn56.week7.harvest-root-identity"
WATCHER_ROOT_SCHEMA_VERSION = 1
WATCHER_LEDGER_SCHEMA = "sn56.week7.safe-harvest-sync"
WATCHER_LEDGER_SCHEMA_VERSION = 2
HEADER_ROOT_SCHEMA = "sn56.week7.public-safetensors-header-root"
HEADER_ROOT_SCHEMA_VERSION = 1
HEADER_SCHEMA = "sn56.week7.public-safetensors-headers.inventory"
HEADER_SCHEMA_VERSION = 1
HEADER_RECORD_SCHEMA = "sn56.week7.public-safetensors-headers"
HEADER_RECORD_SCHEMA_VERSION = 1
TRAINING_ARCHIVE_SCHEMA = "sn56.week7.public-training-archive"
TRAINING_ARCHIVE_SCHEMA_VERSION = 1
TRAINING_ARCHIVE_ROOT_SCHEMA = f"{TRAINING_ARCHIVE_SCHEMA}.root"
TRAINING_ARCHIVE_INVENTORY_SCHEMA = f"{TRAINING_ARCHIVE_SCHEMA}.inventory"
TRAINING_ARCHIVE_RECEIPT_SCHEMA = f"{TRAINING_ARCHIVE_SCHEMA}.receipt"
TOURNAMENT_RE = re.compile(r"tourn_[a-z0-9]+_[0-9]{8}")
TASK_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")
NUMBERED_CHECKPOINT_RE = re.compile(r"(?:^|[_-])(\d{1,12})\.safetensors$", re.IGNORECASE)
HF_TREE_PAGE_RE = re.compile(r"page-([0-9]+)")
FILTERED_SNAPSHOT_RE = re.compile(
    r"snapshots/[0-9]{8}T[0-9]{6}\.[0-9]{6}Z/(?:events\.filtered\.jsonl|state-selection\.json)"
)
PUBLIC_REPO_OWNER_RE = re.compile(r"[1-9A-HJ-NP-Za-km-z]{8}")
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_CONFIG_BYTES = 2 * 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024 * 1024 + 8
MAX_TRAINING_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
TRAINING_ARCHIVE_HOSTS = frozenset({"s3.eu-central-003.backblazeb2.com"})
TRAINING_IMAGE_SUFFIXES = frozenset(
    {".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
TRAINING_CAPTION_SUFFIXES = frozenset({".txt"})
ALLOWED_HF_OBSERVATION_SOURCES = frozenset(
    {"hf-model", "hf-revision-manifest", "hf-tree", "hf-file"}
)
TERMINAL_TASK_STATUSES = {
    "success",
    "completed",
    "failed",
    "failure",
    "terminated",
    "cancelled",
    "canceled",
    "error",
}
TRAINING_ARCHIVE_TERMINAL_TASK_STATUSES = TERMINAL_TASK_STATUSES
TRAINING_ARCHIVE_RIGHTS = {
    "public_access": "observed",
    "license_and_third_party_rights": "unverified",
    "allowed_use": "research-analysis-only",
    "fixture_admission": "not admitted",
}
TRAINING_ARCHIVE_EXCLUSION_CONTRACT = {
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
}

SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "C64": 8,
    "C128": 16,
}


def _read_regular(root: Path, relative: str, limit: int = MAX_JSON_BYTES) -> bytes:
    """Read a bounded regular file beneath a descriptor-anchored real root."""
    fd = _open_regular_beneath(root, relative)
    try:
        result = bytearray()
        while True:
            remaining = limit + 1 - len(result)
            if remaining <= 0:
                raise IntegrityError("local evidence file exceeds its safety ceiling")
            block = os.read(fd, min(1024 * 1024, remaining))
            if not block:
                return bytes(result)
            result.extend(block)
    finally:
        os.close(fd)


def _read_path(path: Path, limit: int = MAX_JSON_BYTES) -> bytes:
    root = _check_root_components(path.parent)
    return _read_regular(root, path.name, limit)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json_bytes(body: bytes, label: str) -> Any:
    if contains_forbidden_body(body):
        raise IntegrityError(f"{label} matched prohibited dataset terminology")
    try:
        return json.loads(
            body,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise IntegrityError(f"{label} is not valid JSON") from exc


def _sha_relative(digest: str) -> str:
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise IntegrityError("invalid SHA-256 identity")
    return f"objects/sha256/{digest[:2]}/{digest}"


def _normalized_relative(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise IntegrityError(f"{label} is not a string")
    normalized = checked_relative(value).as_posix()
    if normalized != value:
        raise IntegrityError(f"{label} is not normalized")
    if contains_forbidden_path(normalized):
        raise IntegrityError(f"{label} contains a prohibited path")
    return normalized


def _parse_utc_timestamp(value: Any, label: str) -> dt.datetime:
    if not isinstance(value, str):
        raise IntegrityError(f"{label} is absent")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise IntegrityError(f"{label} is invalid") from exc
    if parsed.tzinfo is None:
        raise IntegrityError(f"{label} is not timezone-aware")
    return parsed.astimezone(dt.timezone.utc)


def _verify_checksum_sidecar(snapshot: Path, body: bytes) -> tuple[str, str]:
    sidecar = snapshot.with_suffix(".sha256")
    checksum_body = _read_path(sidecar, 4096)
    expected = f"{sha256_bytes(body)}  {snapshot.name}\n".encode("ascii")
    if checksum_body != expected:
        raise IntegrityError("safe API snapshot checksum sidecar is invalid")
    return str(sidecar.absolute()), sha256_bytes(checksum_body)


def load_api_snapshot(snapshot_path: Path, tournament_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    body = _read_path(snapshot_path)
    checksum_path, checksum_sha = _verify_checksum_sidecar(snapshot_path, body)
    value = _json_bytes(body, "safe API snapshot")
    if (
        not isinstance(value, dict)
        or value.get("schema") != API_SCHEMA
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != API_SCHEMA_VERSION
    ):
        raise IntegrityError("safe API snapshot schema is invalid")
    observed_at = _parse_utc_timestamp(value.get("observed_at"), "safe API observed_at")
    if snapshot_path.stem != observed_at.strftime("%Y%m%dT%H%M%S.%fZ"):
        raise IntegrityError("safe API observed_at does not match its snapshot filename")
    tournament = value.get("tournament")
    if not isinstance(tournament, dict) or tournament.get("tournament_id") != tournament_id:
        raise IntegrityError("safe API snapshot tournament identity mismatch")
    if tournament.get("tournament_type") != "image":
        raise IntegrityError("selected tournament is not an image tournament")
    rounds = tournament.get("rounds")
    if not isinstance(rounds, list):
        raise IntegrityError("safe API snapshot has no rounds list")
    r1 = [
        row
        for row in rounds
        if isinstance(row, dict)
        and type(row.get("round_number")) is int
        and row.get("round_number") == 1
    ]
    if len(r1) != 1:
        raise IntegrityError("safe API snapshot does not contain exactly one Round 1")
    if str(r1[0].get("status", "")).lower() != "completed":
        raise IntegrityError("Round 1 is not completed")
    all_task_ids: list[str] = []
    for round_value in rounds:
        if not isinstance(round_value, dict) or not isinstance(round_value.get("tasks"), list):
            raise IntegrityError("safe API tournament round/task shape is invalid")
        for task in round_value["tasks"]:
            task_id = task.get("task_id") if isinstance(task, dict) else None
            if not isinstance(task_id, str) or TASK_RE.fullmatch(task_id) is None:
                raise IntegrityError("safe API tournament contains an invalid task identity")
            all_task_ids.append(task_id)
    if len(all_task_ids) != len(set(all_task_ids)):
        raise IntegrityError("safe API tournament contains duplicate task identities")
    task_details = value.get("tasks")
    if not isinstance(task_details, list):
        raise IntegrityError("safe API snapshot has no task details")
    detail_ids = [
        row.get("task_id") if isinstance(row, dict) else None for row in task_details
    ]
    if (
        not all(isinstance(task_id, str) and TASK_RE.fullmatch(task_id) for task_id in detail_ids)
        or sorted(detail_ids) != sorted(all_task_ids)
    ):
        raise IntegrityError("safe API task details do not exactly match tournament tasks")
    sources = value.get("source_responses")
    if not isinstance(sources, list):
        raise IntegrityError("safe API snapshot has no source-response provenance")
    expected_urls = {
        f"https://api.gradients.io/tournament/{tournament_id}/details",
        *(f"https://api.gradients.io/auditing/tasks/{task_id}" for task_id in all_task_ids),
    }
    observed_urls: set[str] = set()
    for source in sources:
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("url"), str)
            or not isinstance(source.get("response_sha256"), str)
            or SHA256_RE.fullmatch(source["response_sha256"]) is None
            or type(source.get("response_bytes")) is not int
            or source["response_bytes"] <= 0
        ):
            raise IntegrityError("safe API source-response provenance is malformed")
        if source["url"] in observed_urls:
            raise IntegrityError("safe API source-response provenance is duplicated")
        observed_urls.add(source["url"])
    if observed_urls != expected_urls:
        raise IntegrityError("safe API source responses do not bind the exact endpoint set")
    return value, {
        "path": str(snapshot_path.absolute()),
        "sha256": sha256_bytes(body),
        "bytes": len(body),
        "checksum_path": checksum_path,
        "checksum_sha256": checksum_sha,
        "observed_at": utc_iso(observed_at),
    }


def _latest_complete_ledger(root: Path) -> tuple[str, bytes, dict[str, Any], bytes]:
    source = LocalSource(root)
    entries = source.inventory()
    candidates = sorted(
        row.path
        for row in entries
        if row.kind == "file" and re.fullmatch(r"ledgers/[^/]+\.json", row.path)
    )
    if not candidates:
        raise IntegrityError("raw-watcher root has no checksum-bound ledger")
    relative = candidates[-1]
    body = _read_regular(source.root, relative)
    # The producer records its exclusion vocabulary (hidden, holdout,
    # evaluation, and similar terms) inside the ledger's policy block.  That
    # descriptor is not dataset material.  Parse it as metadata, then validate
    # every declared path and referenced body separately below.  Select the
    # newest attempt, not the newest successful attempt: a later PARTIAL sync
    # is a known terminal-evidence failure and must never be hidden by an older
    # COMPLETE ledger.
    value = _descriptor_json(body, "raw-watcher ledger")
    if (
        value.get("schema") != WATCHER_LEDGER_SCHEMA
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != WATCHER_LEDGER_SCHEMA_VERSION
    ):
        raise IntegrityError("newest raw-watcher ledger schema is invalid")
    sidecar_relative = relative.removesuffix(".json") + ".sha256"
    sidecar = _read_regular(source.root, sidecar_relative, 4096)
    expected = f"{sha256_bytes(body)}  {PurePosixPath(relative).name}\n".encode("ascii")
    if sidecar != expected:
        raise IntegrityError("raw-watcher ledger checksum sidecar is invalid")
    if value.get("complete") is not True:
        raise IntegrityError("newest raw-watcher ledger is not COMPLETE")
    return relative, body, value, sidecar


def _observation_belongs_to_scope(relative: str, tournament_id: str, task_ids: set[str]) -> bool:
    parts = checked_relative(relative).parts
    if len(parts) < 4 or parts[0] != "observations":
        return False
    source = parts[1]
    if source == "gradients-tournament":
        return len(parts) == 4 and parts[2] == tournament_id
    if source == "acceptance":
        return len(parts) == 4 and parts[2] == tournament_id
    if source == "gradients-task":
        return len(parts) == 4 and parts[2] in task_ids
    if source == "fixtures":
        # This namespace comes from the full image_text_pairs pool, not the
        # optimizer-visible training_data.zip partition, and may contain rows
        # withheld for evaluation.  It is never part of the P0 package.
        return False
    if source in ALLOWED_HF_OBSERVATION_SOURCES:
        if len(parts) < 5 or parts[2] != "gradients-io-tournaments":
            return False
        repo_name = parts[3]
        marker = f"tournament-{tournament_id}-"
        if not repo_name.startswith(marker):
            return False
        tail = repo_name[len(marker) :]
        if len(tail) <= 37 or tail[36] != "-":
            return False
        task_id, owner = tail[:36], tail[37:]
        if (
            task_id not in task_ids
            or PUBLIC_REPO_OWNER_RE.fullmatch(owner) is None
        ):
            return False
        if source == "hf-model":
            return len(parts) == 5
        if len(parts) < 6 or REVISION_RE.fullmatch(parts[4]) is None:
            return False
        if source == "hf-revision-manifest":
            return len(parts) == 6
        if source == "hf-tree":
            return len(parts) == 7 and HF_TREE_PAGE_RE.fullmatch(parts[5]) is not None
        return source == "hf-file" and len(parts) >= 7
    return False


def _declared_nested_objects(value: Any) -> list[tuple[str, str | None]]:
    result: list[tuple[str, str | None]] = []

    def visit(item: Any, inherited_path: str | None = None) -> None:
        if isinstance(item, dict):
            path = next(
                (
                    item[key]
                    for key in ("path", "rfilename", "filename")
                    if isinstance(item.get(key), str)
                ),
                inherited_path,
            )
            digest = item.get("object_sha256")
            if isinstance(digest, str):
                result.append((digest, path))
            for child in item.values():
                visit(child, path)
        elif isinstance(item, list):
            for child in item:
                visit(child, inherited_path)

    visit(value)
    return result


def _declared_raw_weight_objects(value: Any) -> set[str]:
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            path = item.get("path")
            digest = item.get("object_sha256")
            decoded_path = path
            if isinstance(decoded_path, str):
                for _ in range(len(decoded_path) + 1):
                    next_decoded = unquote(decoded_path)
                    if next_decoded == decoded_path:
                        break
                    decoded_path = next_decoded
                else:
                    raise IntegrityError("HF file path has excessive nested encoding")
            if (
                isinstance(decoded_path, str)
                and decoded_path.lower().endswith(
                    (".safetensors", ".ckpt", ".pt", ".pth", ".bin")
                )
                and isinstance(digest, str)
            ):
                found.add(_sha_relative(digest))
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return found


def verify_watcher_root(
    root: Path,
    tournament_id: str,
    r1_task_ids: set[str],
    *,
    api_observed_at: dt.datetime,
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    """Verify the selected ledger and return exact-R1 observation bodies."""
    source = LocalSource(root)
    identity_body = _read_regular(source.root, "ROOT-IDENTITY.json", 1024 * 1024)
    identity = _json_bytes(identity_body, "raw-watcher root identity")
    if (
        not isinstance(identity, dict)
        or identity.get("schema") != WATCHER_ROOT_SCHEMA
        or type(identity.get("schema_version")) is not int
        or identity.get("schema_version") != WATCHER_ROOT_SCHEMA_VERSION
        or identity.get("tournament_id") != tournament_id
    ):
        raise IntegrityError("raw-watcher root identity mismatch")

    ledger_path, ledger_body, ledger, checksum_body = _latest_complete_ledger(source.root)
    scope = ledger.get("scope")
    if not isinstance(scope, dict) or scope.get("tournament_id") != tournament_id:
        raise IntegrityError("raw-watcher ledger scope mismatch")
    ledger_tasks = scope.get("task_ids")
    if (
        not isinstance(ledger_tasks, list)
        or not all(isinstance(task_id, str) and TASK_RE.fullmatch(task_id) for task_id in ledger_tasks)
        or ledger_tasks != sorted(r1_task_ids)
    ):
        raise IntegrityError("raw-watcher ledger task scope is not the exact canonical Round-1 set")
    if ledger.get("status") != "COMPLETE" or ledger.get("errors") not in ([], None):
        raise IntegrityError("raw-watcher ledger is not clean and complete")
    if ledger.get("root_identity_sha256") != sha256_bytes(identity_body):
        raise IntegrityError("raw-watcher ledger does not bind its root identity")
    ledger_observed_at = _parse_utc_timestamp(
        ledger.get("observed_at"), "raw-watcher ledger observed_at"
    )
    if PurePosixPath(ledger_path).stem != ledger_observed_at.strftime(
        "%Y%m%dT%H%M%S.%fZ"
    ):
        raise IntegrityError("raw-watcher ledger observed_at does not match its filename")
    if ledger_observed_at < api_observed_at:
        raise IntegrityError("raw-watcher ledger predates the completed API snapshot")
    derived_observation = _normalized_relative(
        scope.get("derived_from_observation"), "raw-watcher scope observation"
    )
    derived_digest = scope.get("derived_from_content_sha256")
    if not isinstance(derived_digest, str) or SHA256_RE.fullmatch(derived_digest) is None:
        raise IntegrityError("raw-watcher scope content identity is invalid")
    files = ledger.get("files")
    if not isinstance(files, list):
        raise IntegrityError("raw-watcher ledger has no file inventory")

    expected: dict[str, tuple[str, int]] = {}
    for row in files:
        if not isinstance(row, dict):
            raise IntegrityError("raw-watcher ledger file row is invalid")
        relative = row.get("path")
        digest = row.get("sha256")
        size = row.get("bytes")
        if not isinstance(relative, str) or not isinstance(size, int):
            raise IntegrityError("raw-watcher ledger file identity is incomplete")
        checked_relative(relative)
        if contains_forbidden_path(relative):
            # Fail before opening the named object.  A sanitized exact root must
            # never claim one of these paths.
            raise IntegrityError("raw-watcher ledger contains a prohibited path")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise IntegrityError("raw-watcher ledger file digest is invalid")
        if relative in expected:
            raise IntegrityError("raw-watcher ledger contains a duplicate path")
        expected[relative] = (digest, size)
    if derived_observation not in expected:
        raise IntegrityError("raw-watcher scope observation is outside the selected ledger")

    verified: dict[str, bytes] = {}

    def read_expected(relative: str) -> bytes:
        identity = expected.get(relative)
        if identity is None:
            raise IntegrityError("raw-watcher observation references a file outside its ledger")
        body = _read_regular(source.root, relative)
        if len(body) != identity[1] or sha256_bytes(body) != identity[0]:
            raise IntegrityError("raw-watcher ledger file hash/size mismatch")
        if contains_forbidden_body(body):
            raise IntegrityError("raw-watcher ledger file contains prohibited material")
        verified[relative] = body
        return body

    # Metadata first.  This lets us identify a prohibited nested object by its
    # declared path before opening the opaque SHA-named CAS file.
    observation_wrappers: dict[str, dict[str, Any]] = {}
    prohibited_nested: set[str] = set()
    raw_weight_objects: set[str] = set()
    referenced_cas: set[str] = set()
    exact_scope = TournamentScope(
        tournament_id,
        frozenset(r1_task_ids),
        derived_observation,
        derived_digest,
    )
    for relative in sorted(expected):
        if not relative.startswith("observations/"):
            continue
        if not _observation_belongs_to_scope(relative, tournament_id, r1_task_ids):
            raise IntegrityError("raw-watcher ledger contains an out-of-scope observation")
        wrapper_body = read_expected(relative)
        wrapper = _json_bytes(wrapper_body, "raw-watcher observation wrapper")
        if not isinstance(wrapper, dict):
            raise IntegrityError("raw-watcher observation wrapper is not an object")
        wrapper_source, wrapper_key = validate_observation_identity(relative, wrapper)
        validate_public_request_provenance(wrapper_source, wrapper_key, wrapper)
        wrapper_observed_at = _parse_utc_timestamp(
            observation_timestamp(relative, wrapper), "raw-watcher observation observed_at"
        )
        if wrapper_observed_at > ledger_observed_at:
            raise IntegrityError("raw-watcher observation postdates its ledger")
        digest = wrapper.get("content_sha256")
        object_relative = _sha_relative(digest)
        if wrapper.get("object") != object_relative:
            raise IntegrityError("raw-watcher observation has an invalid CAS binding")
        referenced_cas.add(object_relative)
        content = verified.get(object_relative) or read_expected(object_relative)
        if type(wrapper.get("content_bytes")) is not int or wrapper.get(
            "content_bytes"
        ) != len(content):
            raise IntegrityError("raw-watcher observation content length is invalid")
        try:
            content_value = json.loads(content)
        except json.JSONDecodeError:
            content_value = None
        validate_scoped_observation_body(
            wrapper_source, wrapper_key, content_value, exact_scope
        )
        nested_objects = (
            _declared_nested_objects(content_value)
            if wrapper.get("source") in ALLOWED_HF_OBSERVATION_SOURCES
            else []
        )
        for nested_digest, declared_path in nested_objects:
            nested_relative = _sha_relative(nested_digest)
            if declared_path is None:
                raise IntegrityError("raw-watcher HF nested object has no declared path")
            if contains_forbidden_path(declared_path):
                prohibited_nested.add(nested_relative)
            else:
                referenced_cas.add(nested_relative)
        if wrapper.get("source") in ALLOWED_HF_OBSERVATION_SOURCES:
            raw_weight_objects.update(_declared_raw_weight_objects(content_value))
        observation_wrappers[relative] = wrapper

    for relative in sorted(expected):
        if relative in verified:
            continue
        if relative.startswith("objects/sha256/") and relative not in referenced_cas:
            raise IntegrityError("raw-watcher ledger contains an unreferenced CAS object")
        if relative in prohibited_nested:
            # Crucially, no open occurs for the named object.
            raise IntegrityError("raw-watcher metadata references a prohibited nested object")
        if relative in raw_weight_objects:
            # The header inventory is the authorized, bounded safetensors
            # surface.  A full weight body in the watcher root is an input
            # contract violation, not something this CPU finalizer hashes.
            raise IntegrityError("raw-watcher root contains a raw weight body; refusing to read it")
        if not relative.startswith("objects/sha256/") and FILTERED_SNAPSHOT_RE.fullmatch(
            relative
        ) is None:
            raise IntegrityError("raw-watcher ledger contains an unknown file namespace")
        read_expected(relative)

    # Refuse a moving frontier: every exact-scope observation currently present
    # must be named by the selected COMPLETE ledger.  Symlinks are never opened.
    for entry in source.inventory():
        if not entry.path.startswith("observations/"):
            continue
        if not _observation_belongs_to_scope(entry.path, tournament_id, set(ledger_tasks)):
            continue
        if contains_forbidden_path(entry.path):
            raise IntegrityError("raw-watcher root contains a prohibited observation path")
        if entry.kind != "file":
            raise IntegrityError("raw-watcher root contains a linked/non-regular observation")
        if entry.path not in verified:
            raise IntegrityError("raw-watcher root advanced beyond the selected COMPLETE ledger")

    observations: list[dict[str, Any]] = []
    for relative, wrapper_body in sorted(verified.items()):
        if not relative.startswith("observations/"):
            continue
        if not _observation_belongs_to_scope(relative, tournament_id, r1_task_ids):
            continue
        wrapper = observation_wrappers[relative]
        digest = wrapper.get("content_sha256")
        object_relative = _sha_relative(digest)
        if wrapper.get("object") != object_relative:
            raise IntegrityError("raw-watcher observation has an invalid CAS binding")
        object_body = verified.get(object_relative)
        if object_body is None or sha256_bytes(object_body) != digest:
            raise IntegrityError("raw-watcher observation CAS is absent or invalid")
        if type(wrapper.get("content_bytes")) is not int or wrapper.get(
            "content_bytes"
        ) != len(object_body):
            raise IntegrityError("raw-watcher observation content length is invalid")
        observations.append(
            {
                "path": relative,
                "wrapper_sha256": sha256_bytes(wrapper_body),
                "wrapper": wrapper,
                "content": object_body,
                "content_sha256": digest,
            }
        )

    terminal_scope = next(
        (row for row in observations if row["path"] == derived_observation), None
    )
    if terminal_scope is None:
        raise IntegrityError("raw-watcher terminal scope observation was not reconstructed")
    if scope.get("derived_from_content_sha256") != terminal_scope["content_sha256"]:
        raise IntegrityError("raw-watcher scope content identity is inconsistent")
    terminal_value = _json_bytes(
        terminal_scope["content"], "raw-watcher terminal tournament observation"
    )
    if (
        not isinstance(terminal_value, dict)
        or terminal_value.get("tournament_id") != tournament_id
        or terminal_value.get("tournament_type") != "image"
    ):
        raise IntegrityError("raw-watcher terminal body is not the selected image tournament")
    rounds = terminal_value.get("rounds") if isinstance(terminal_value, dict) else None
    terminal_r1 = [
        row
        for row in rounds or []
        if isinstance(row, dict)
        and type(row.get("round_number")) is int
        and row.get("round_number") == 1
    ]
    if len(terminal_r1) != 1 or str(terminal_r1[0].get("status", "")).lower() != "completed":
        raise IntegrityError("raw-watcher scope is not bound to a completed Round 1")
    terminal_tasks = terminal_r1[0].get("tasks")
    if not isinstance(terminal_tasks, list):
        raise IntegrityError("raw-watcher terminal Round 1 has no task list")
    terminal_task_ids = [
        row.get("task_id") for row in terminal_tasks if isinstance(row, dict)
    ]
    if (
        len(terminal_task_ids) != len(terminal_tasks)
        or len(set(terminal_task_ids)) != len(terminal_task_ids)
        or any(not isinstance(task, str) or TASK_RE.fullmatch(task) is None for task in terminal_task_ids)
        or set(terminal_task_ids) != r1_task_ids
    ):
        raise IntegrityError("raw-watcher terminal Round-1 task membership mismatches the API")

    provenance = {
        "root": str(source.root),
        "root_identity_sha256": sha256_bytes(identity_body),
        "ledger": ledger_path,
        "ledger_sha256": sha256_bytes(ledger_body),
        "ledger_checksum_sha256": sha256_bytes(checksum_body),
        "ledger_observed_at": utc_iso(ledger_observed_at),
        "terminal_scope_observation": derived_observation,
        "verified_file_count": len(verified),
        "verified_file_bytes": sum(len(body) for body in verified.values()),
        "r1_observation_count": len(observations),
    }
    return observations, verified, provenance


def _validated_safetensors_summary(
    parsed: Mapping[str, Any], *, lfs_bytes: int, raw_header_bytes: int
) -> tuple[Any, list[dict[str, Any]]]:
    """Re-derive a strict tensor inventory from exact header and LFS bytes."""
    metadata = parsed.get("__metadata__")
    if metadata is not None and (
        not isinstance(metadata, dict)
        or not all(isinstance(key, str) and isinstance(value, str) for key, value in metadata.items())
    ):
        raise IntegrityError("public safetensors metadata is malformed")
    data_bytes = lfs_bytes - raw_header_bytes
    if data_bytes < 0:
        raise IntegrityError("public safetensors header exceeds its LFS object")
    tensors: list[dict[str, Any]] = []
    intervals: list[tuple[int, int, str]] = []
    for name, value in sorted(parsed.items()):
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not name or not isinstance(value, dict):
            raise IntegrityError("public safetensors tensor entry is malformed")
        dtype = value.get("dtype")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if dtype not in SAFETENSORS_DTYPE_BYTES:
            raise IntegrityError("public safetensors tensor dtype is unsupported")
        if not isinstance(shape, list) or not all(
            type(dimension) is int and dimension >= 0 for dimension in shape
        ):
            raise IntegrityError("public safetensors tensor shape is invalid")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(type(offset) is int and offset >= 0 for offset in offsets)
            or offsets[0] > offsets[1]
            or offsets[1] > data_bytes
        ):
            raise IntegrityError("public safetensors tensor offsets are invalid")
        elements = 1
        for dimension in shape:
            elements *= dimension
        if offsets[1] - offsets[0] != elements * SAFETENSORS_DTYPE_BYTES[dtype]:
            raise IntegrityError("public safetensors tensor byte span contradicts dtype/shape")
        intervals.append((offsets[0], offsets[1], name))
        tensors.append(
            {"name": name, "dtype": dtype, "shape": shape, "data_offsets": offsets}
        )
    if not tensors:
        raise IntegrityError("public safetensors header contains no tensors")
    prior_end = 0
    for start, end, _name in sorted(intervals):
        if start != prior_end:
            raise IntegrityError("public safetensors tensor byte ranges are not contiguous")
        prior_end = end
    if prior_end != data_bytes:
        raise IntegrityError("public safetensors tensor ranges do not cover the data buffer")
    return metadata, tensors


def load_header_inventory(
    root: Path,
    inventory_path: Path,
    tournament_id: str,
    r1_task_ids: set[str],
    watcher_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    header_root = _check_root_components(root)
    identity_body = _read_regular(header_root, "ROOT-IDENTITY.json", 1024 * 1024)
    identity = _json_bytes(identity_body, "header root identity")
    if (
        not isinstance(identity, dict)
        or identity.get("schema") != HEADER_ROOT_SCHEMA
        or type(identity.get("schema_version")) is not int
        or identity.get("schema_version") != HEADER_ROOT_SCHEMA_VERSION
        or identity.get("tournament_id") != tournament_id
    ):
        raise IntegrityError("public-header root identity mismatch")
    if identity.get("input_root") != str(_check_root_components(watcher_root)):
        raise IntegrityError("public-header root is bound to a different watcher root")
    identity_tasks = identity.get("task_ids")
    if (
        not isinstance(identity_tasks, list)
        or identity_tasks != sorted(r1_task_ids)
        or len(identity_tasks) != len(set(identity_tasks))
        or any(not isinstance(task, str) or TASK_RE.fullmatch(task) is None for task in identity_tasks)
    ):
        raise IntegrityError("public-header root task allowlist mismatches exact Round 1")
    try:
        relative = inventory_path.absolute().relative_to(header_root).as_posix()
    except ValueError as exc:
        raise IntegrityError("header inventory is outside its declared root") from exc
    inventory_body = _read_regular(header_root, relative)
    inventory = _json_bytes(inventory_body, "public-header inventory")
    if (
        not isinstance(inventory, dict)
        or inventory.get("schema") != HEADER_SCHEMA
        or type(inventory.get("schema_version")) is not int
        or inventory.get("schema_version") != HEADER_SCHEMA_VERSION
        or inventory.get("tournament_id") != tournament_id
    ):
        raise IntegrityError("public-header inventory identity/schema mismatch")
    if inventory.get("task_ids") != sorted(r1_task_ids):
        raise IntegrityError("public-header inventory task allowlist mismatches exact Round 1")
    associations = inventory.get("associations")
    if not isinstance(associations, list) or inventory.get("candidate_count") != len(associations):
        raise IntegrityError("public-header inventory count is invalid")

    verified: list[dict[str, Any]] = []
    unique_lfs: set[str] = set()
    seen_associations: set[tuple[str, str, str, str]] = set()
    for association in associations:
        if not isinstance(association, dict):
            raise IntegrityError("public-header association is invalid")
        task_id = association.get("task_id")
        if task_id not in r1_task_ids:
            raise IntegrityError("public-header association is outside exact Round 1")
        path = _normalized_relative(
            association.get("path"), "public-header association path"
        )
        repo = association.get("repository")
        revision = association.get("revision")
        lfs_sha = association.get("lfs_sha256")
        lfs_bytes = association.get("lfs_bytes")
        if (
            not isinstance(repo, str)
            or not isinstance(revision, str)
            or REVISION_RE.fullmatch(revision) is None
            or not isinstance(lfs_sha, str)
            or SHA256_RE.fullmatch(lfs_sha) is None
            or not isinstance(lfs_bytes, int)
            or lfs_bytes <= 8
        ):
            raise IntegrityError("public-header association identity is invalid")
        if _public_repo_task(repo, tournament_id, r1_task_ids) != task_id:
            raise IntegrityError("public-header association repository/task identity is invalid")
        association_key = (repo, revision, path, lfs_sha)
        if association_key in seen_associations:
            raise IntegrityError("public-header inventory contains a duplicate association")
        seen_associations.add(association_key)
        tree_observations = association.get("tree_observations")
        if (
            not isinstance(tree_observations, list)
            or not tree_observations
            or not all(isinstance(row, dict) for row in tree_observations)
        ):
            raise IntegrityError("public-header association lacks tree provenance")
        provenance_keys: list[tuple[str, str, str, str]] = []
        for provenance in tree_observations:
            observation = _normalized_relative(
                provenance.get("tree_observation"), "public-header tree observation path"
            )
            observation_sha = provenance.get("tree_observation_sha256")
            content_sha = provenance.get("tree_content_sha256")
            observed_at = provenance.get("observed_at")
            if (
                not isinstance(observation_sha, str)
                or SHA256_RE.fullmatch(observation_sha) is None
                or not isinstance(content_sha, str)
                or SHA256_RE.fullmatch(content_sha) is None
                or not isinstance(observed_at, str)
            ):
                raise IntegrityError("public-header tree provenance is malformed")
            provenance_keys.append((observed_at, observation, observation_sha, content_sha))
        if provenance_keys != sorted(provenance_keys) or len(set(provenance_keys)) != len(
            provenance_keys
        ):
            raise IntegrityError("public-header tree provenance is not canonical")
        latest = tree_observations[-1]
        if any(
            association.get(field) != latest.get(field)
            for field in (
                "tree_observation",
                "tree_observation_sha256",
                "tree_content_sha256",
            )
        ):
            raise IntegrityError("public-header latest tree provenance is inconsistent")
        record_relative = _normalized_relative(
            association.get("record"), "public-header record path"
        )
        record_body = _read_regular(header_root, record_relative)
        if sha256_bytes(record_body) != association.get("record_sha256"):
            raise IntegrityError("public-header record hash mismatch")
        record = _json_bytes(record_body, "public-header record")
        if (
            not isinstance(record, dict)
            or record.get("schema") != HEADER_RECORD_SCHEMA
            or type(record.get("schema_version")) is not int
            or record.get("schema_version") != HEADER_RECORD_SCHEMA_VERSION
        ):
            raise IntegrityError("public-header record schema is invalid")
        if record.get("lfs_sha256") != lfs_sha or record.get("lfs_bytes") != lfs_bytes:
            raise IntegrityError("public-header record LFS identity mismatch")
        header_sha = record.get("header_sha256")
        if association.get("header_sha256") != header_sha:
            raise IntegrityError("public-header association contradicts its record header hash")
        header_relative = _normalized_relative(
            record.get("header_object"), "public-header raw header path"
        )
        if header_relative != _sha_relative(header_sha):
            raise IntegrityError("public-header raw header CAS binding is invalid")
        raw_header = _read_regular(header_root, header_relative, MAX_HEADER_BYTES)
        if sha256_bytes(raw_header) != header_sha or len(raw_header) != record.get("header_bytes"):
            raise IntegrityError("public-header raw header hash/size mismatch")
        if len(raw_header) < 9:
            raise IntegrityError("public-header raw header is too short")
        declared = struct.unpack("<Q", raw_header[:8])[0]
        if declared != len(raw_header) - 8:
            raise IntegrityError("public-header raw header length prefix mismatch")
        parsed = _json_bytes(raw_header[8:], "public safetensors header")
        if not isinstance(parsed, dict):
            raise IntegrityError("public safetensors header is not an object")
        metadata, tensors = _validated_safetensors_summary(
            parsed, lfs_bytes=lfs_bytes, raw_header_bytes=len(raw_header)
        )
        tensor_count = len(tensors)
        if (
            metadata != record.get("metadata")
            or tensor_count != record.get("tensor_count")
            or tensors != record.get("tensors")
        ):
            raise IntegrityError("public-header parsed record contradicts raw header bytes")
        unique_lfs.add(lfs_sha)
        verified.append(
            {
                "repository": repo,
                "revision": revision,
                "task_id": task_id,
                "path": path,
                "lfs_sha256": lfs_sha,
                "lfs_bytes": lfs_bytes,
                "header_sha256": header_sha,
                "header_bytes": len(raw_header),
                "metadata": record.get("metadata"),
                "tensor_count": tensor_count,
                "record": record_relative,
                "record_sha256": sha256_bytes(record_body),
                "tree_observation": association.get("tree_observation"),
                "tree_observation_sha256": association.get("tree_observation_sha256"),
                "tree_observations": tree_observations,
            }
        )
    if inventory.get("unique_lfs_objects") != len(
        {row.get("lfs_sha256") for row in associations if isinstance(row, dict)}
    ):
        raise IntegrityError("public-header inventory unique-object count is invalid")
    return verified, {
        "root": str(header_root),
        "root_identity_sha256": sha256_bytes(identity_body),
        "inventory": relative,
        "inventory_sha256": sha256_bytes(inventory_body),
        "r1_association_count": len(verified),
        "r1_unique_lfs_objects": len(unique_lfs),
    }


def _public_repo_task(repo: str, tournament_id: str, task_ids: set[str]) -> str:
    prefix = f"gradients-io-tournaments/tournament-{tournament_id}-"
    if not repo.startswith(prefix):
        raise IntegrityError("repository is outside the exact public tournament organization")
    tail = repo[len(prefix) :]
    if len(tail) <= 37 or tail[36] != "-":
        raise IntegrityError("repository does not contain an exact task and owner")
    task_id = tail[:36]
    owner = tail[37:]
    if (
        task_id not in task_ids
        or TASK_RE.fullmatch(task_id) is None
        or PUBLIC_REPO_OWNER_RE.fullmatch(owner) is None
    ):
        raise IntegrityError("repository task/owner identity is invalid")
    return task_id


def _repo_from_api(
    value: Any, tournament_id: str, task_id: str, participant_hotkey: str
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise IntegrityError("participant repository is not a string")
    prefix = "https://huggingface.co/"
    repo = value.removeprefix(prefix).rstrip("/")
    if _public_repo_task(repo, tournament_id, {task_id}) != task_id:
        raise IntegrityError("participant repository is outside the selected task")
    if (
        not isinstance(participant_hotkey, str)
        or len(participant_hotkey) < 8
        or PUBLIC_REPO_OWNER_RE.fullmatch(participant_hotkey[:8]) is None
        or repo.rsplit("-", 1)[-1] != participant_hotkey[:8]
    ):
        raise IntegrityError("participant repository prefix contradicts its hotkey")
    return repo


def _parse_observed_at(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _recipe_projection(config_body: bytes) -> tuple[dict[str, Any] | None, str | None]:
    """Return only score-relevant, path-free scalar/list fields."""
    if len(config_body) > MAX_CONFIG_BYTES:
        return None, "config exceeds the projection safety ceiling"
    if contains_forbidden_body(config_body):
        raise IntegrityError("public config matched prohibited dataset terminology")
    try:
        value = yaml.safe_load(config_body)
    except yaml.YAMLError:
        return None, "config is not valid safe YAML"
    if not isinstance(value, dict):
        return None, "config root is not a mapping"
    processes = value.get("config", {}).get("process") if isinstance(value.get("config"), dict) else None
    if not isinstance(processes, list) or not processes or not isinstance(processes[0], dict):
        return None, "config has no canonical first process"
    process = processes[0]
    sections = {
        "network": ("type", "linear", "linear_alpha"),
        "save": (
            "dtype",
            "save_every",
            "max_step_saves_to_keep",
            "save_format",
            "push_to_hub",
        ),
        "train": (
            "batch_size",
            "steps",
            "gradient_accumulation",
            "train_unet",
            "train_text_encoder",
            "text_encoder_lr",
            "gradient_checkpointing",
            "noise_scheduler",
            "optimizer",
            "timestep_type",
            "content_or_style",
            "cache_text_embeddings",
            "lr",
            "unet_lr",
            "lr_scheduler",
            "dtype",
            "loss_type",
            "do_differential_guidance",
            "differential_guidance_scale",
            "multires_noise_iterations",
            "multires_noise_discount",
            "noise_offset",
        ),
        "model": ("arch", "quantize", "quantize_te", "low_vram"),
        "sample": (
            "sampler",
            "sample_every",
            "width",
            "height",
            "seed",
            "walk_seed",
            "guidance_scale",
            "steps",
        ),
    }
    fields: dict[str, Any] = {}
    for section, names in sections.items():
        mapping = process.get(section)
        if not isinstance(mapping, dict):
            continue
        for name in names:
            item = mapping.get(name)
            if item is None or not isinstance(item, (str, int, float, bool)):
                continue
            fields[f"config.process[0].{section}.{name}"] = item
    nested_sections = {
        "ema_config": ("use_ema", "ema_decay", "update_after_step", "update_every"),
        "optimizer_params": (
            "weight_decay",
            "betas",
            "eps",
            "d_coef",
            "use_bias_correction",
            "safeguard_warmup",
            "stochastic_rounding",
        ),
        "lr_scheduler_params": (
            "warmup_steps",
            "num_cycles",
            "power",
            "min_lr",
            "min_lr_ratio",
            "lr_end",
            "lr_min",
            "eta_min",
            "decay_steps",
        ),
    }
    train = process.get("train")
    if isinstance(train, dict):
        for section, names in nested_sections.items():
            mapping = train.get(section)
            if not isinstance(mapping, dict):
                continue
            for name in names:
                item = mapping.get(name)
                if isinstance(item, (str, int, float, bool)) or (
                    isinstance(item, list)
                    and len(item) <= 16
                    and all(isinstance(part, (str, int, float, bool)) for part in item)
                ):
                    fields[f"config.process[0].train.{section}.{name}"] = item
    for name in ("sn56_runtime_bundle", "runtime_bundle", "strict_runtime"):
        item = process.get(name)
        if isinstance(item, (str, int, float, bool)):
            fields[f"config.process[0].{name}"] = item
    datasets = process.get("datasets")
    if isinstance(datasets, list) and datasets and isinstance(datasets[0], dict):
        for name in ("caption_dropout_rate", "cache_latents_to_disk", "resolution"):
            item = datasets[0].get(name)
            if isinstance(item, (str, int, float, bool)) or (
                isinstance(item, list)
                and len(item) <= 16
                and all(isinstance(part, (str, int, float, bool)) for part in item)
            ):
                fields[f"config.process[0].datasets[0].{name}"] = item
    projection = {"kind": "narrow-score-relevant-allowlist", "fields": fields}
    if contains_forbidden_body(canonical_json(projection)):
        raise IntegrityError("recipe projection unexpectedly matched prohibited terminology")
    return projection, None


def _metadata_step(metadata: Any) -> int | None:
    if not isinstance(metadata, dict):
        return None
    candidate: Any = metadata.get("training_info")
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except json.JSONDecodeError:
            return None
    if isinstance(candidate, dict):
        step = candidate.get("step")
        if isinstance(step, int) and not isinstance(step, bool) and step >= 0:
            return step
    direct = metadata.get("step")
    if isinstance(direct, int) and not isinstance(direct, bool) and direct >= 0:
        return direct
    return None


def _path_step(path: str) -> int | None:
    match = NUMBERED_CHECKPOINT_RE.search(PurePosixPath(path).name)
    return int(match.group(1)) if match else None


def checkpoint_identity_summary(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in sorted(checkpoints, key=lambda row: (row["path"], row["lfs_sha256"])):
        name = PurePosixPath(item["path"]).name.lower()
        role_labels = []
        if name == "last.safetensors":
            role_labels.append("last")
        if "final" in name:
            role_labels.append("filename-final")
        if "best" in name or "selected" in name:
            role_labels.append("filename-selected")
        if "soup" in name or "merged" in name:
            role_labels.append("filename-soup-or-merged")
        rows.append(
            {
                **item,
                "numbered_step_from_filename": _path_step(item["path"]),
                "step_from_header_metadata": _metadata_step(item.get("metadata")),
                "filename_role_labels": role_labels,
            }
        )
    by_oid: dict[str, list[str]] = {}
    for row in rows:
        by_oid.setdefault(row["lfs_sha256"], []).append(row["path"])
    identity_groups = [
        {"lfs_sha256": oid, "paths": sorted(paths), "byte_identical": True}
        for oid, paths in sorted(by_oid.items())
        if len(paths) > 1
    ]
    lasts = [row for row in rows if "last" in row["filename_role_labels"]]
    numbered = [row for row in rows if row["numbered_step_from_filename"] is not None]
    highest_numbered = max(
        (row["numbered_step_from_filename"] for row in numbered), default=None
    )
    comparisons: list[dict[str, Any]] = []
    for last in lasts:
        matches = sorted(
            {
                row["numbered_step_from_filename"]
                for row in numbered
                if row["lfs_sha256"] == last["lfs_sha256"]
            }
        )
        classification = (
            "last-matches-earlier-numbered-checkpoint"
            if matches and highest_numbered is not None and max(matches) < highest_numbered
            else "last-matches-highest-observed-numbered-checkpoint"
            if highest_numbered is not None and highest_numbered in matches
            else "last-matches-numbered-checkpoint"
            if matches
            else "last-distinct-from-observed-numbered-checkpoints"
            if numbered
            else "no-numbered-comparison-available"
        )
        comparisons.append(
            {
                "last_path": last["path"],
                "last_lfs_sha256": last["lfs_sha256"],
                "submitted_artifact_step_from_header": last["step_from_header_metadata"],
                "byte_identical_numbered_steps": matches,
                "highest_observed_numbered_step": highest_numbered,
                "classification": classification,
            }
        )
    return {
        "checkpoints": rows,
        "byte_identity_groups": identity_groups,
        "last_vs_numbered": comparisons,
        "interpretation_limit": (
            "filename roles and LFS equality are observations; highest numbered is not asserted "
            "to be the training final, and header step is not asserted to be the configured plan"
        ),
    }


def _safe_tree_entries(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise IntegrityError("HF tree response is not a list")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise IntegrityError("HF tree entry is not an object")
        path = _normalized_relative(item.get("path"), "HF tree entry path")
        kind = item.get("type")
        size = item.get("size")
        if kind not in {"file", "directory"} or not isinstance(size, int) or size < 0:
            raise IntegrityError("HF tree entry type/size is invalid")
        row: dict[str, Any] = {"path": path, "type": kind, "bytes": size}
        oid = item.get("oid")
        if isinstance(oid, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", oid):
            row["git_or_xet_oid"] = oid
        lfs = item.get("lfs")
        if lfs is not None:
            if not isinstance(lfs, dict):
                raise IntegrityError("HF tree LFS identity is invalid")
            lfs_oid = lfs.get("oid")
            lfs_size = lfs.get("size")
            if (
                not isinstance(lfs_oid, str)
                or SHA256_RE.fullmatch(lfs_oid) is None
                or not isinstance(lfs_size, int)
                or lfs_size < 0
            ):
                raise IntegrityError("HF tree LFS identity is invalid")
            if size != lfs_size:
                raise IntegrityError("HF tree file size conflicts with its LFS size")
            row["lfs_sha256"] = lfs_oid
            row["lfs_bytes"] = lfs_size
        result.append(row)
    return result


def _merged_tree_summary(pages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for page in pages:
        for row in page["entries"]:
            prior = by_path.get(row["path"])
            if prior is not None and prior != row:
                raise IntegrityError("HF tree pages contain conflicting duplicate paths")
            by_path[row["path"]] = row
    return [by_path[path] for path in sorted(by_path)]


def _canonical_tree_pages(pages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse agreeing repeats and require a contiguous 1-based page set."""
    by_page: dict[int, dict[str, Any]] = {}
    for row in pages:
        number = row.get("page_number")
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            raise IntegrityError("HF tree page number is invalid")
        prior = by_page.get(number)
        if prior is not None and (
            prior.get("content_sha256") != row.get("content_sha256")
            or prior.get("entries") != row.get("entries")
        ):
            raise IntegrityError("HF tree page has conflicting repeated observations")
        if prior is None or row.get("observed_at", "") > prior.get("observed_at", ""):
            by_page[number] = row
    if by_page and sorted(by_page) != list(range(1, max(by_page) + 1)):
        raise IntegrityError("HF tree page set is not contiguous from page one")
    return [by_page[number] for number in sorted(by_page)]


def _public_telemetry_projection(path: str, body: bytes) -> dict[str, Any] | None:
    """Project only the already-scrubbed public Forge recorder JSON."""
    if PurePosixPath(path).name != "forge_run.json":
        return None
    value = _json_bytes(body, "public Forge run recorder")
    if not isinstance(value, dict) or value.get("kind") != "forge-public-run-recorder":
        raise IntegrityError("public Forge run recorder identity is invalid")
    if value.get("schema") != 2 or not isinstance(value.get("events"), list):
        raise IntegrityError("public Forge run recorder schema is invalid")
    events: list[dict[str, Any]] = []
    for event in value["events"]:
        if (
            not isinstance(event, dict)
            or not isinstance(event.get("name"), str)
            or not isinstance(event.get("t"), (int, float))
            or isinstance(event.get("t"), bool)
        ):
            raise IntegrityError("public Forge run recorder event is invalid")
        events.append({"name": event["name"], "t": event["t"]})
    private_sha = value.get("private_record_sha256")
    if not isinstance(private_sha, str) or SHA256_RE.fullmatch(private_sha) is None:
        raise IntegrityError("public Forge run recorder private binding is invalid")
    return {
        "kind": value["kind"],
        "schema": value["schema"],
        "events": events,
        "private_record_sha256": private_sha,
    }


def _manifest_records(
    observations: Iterable[dict[str, Any]],
    verified_watcher_files: Mapping[str, bytes],
    tournament_id: str,
    r1_task_ids: set[str],
    header_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    headers_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in header_rows:
        provenance = row.get("tree_observations")
        if not isinstance(provenance, list) or not provenance:
            raise IntegrityError("public-header association has no tree provenance")
        for source_row in provenance:
            if not isinstance(source_row, dict):
                raise IntegrityError("public-header tree provenance row is invalid")
            tree_observation = source_row.get("tree_observation")
            if (
                not isinstance(tree_observation, str)
                or contains_forbidden_path(tree_observation)
                or tree_observation not in verified_watcher_files
                or sha256_bytes(verified_watcher_files[tree_observation])
                != source_row.get("tree_observation_sha256")
            ):
                raise IntegrityError("public-header association tree-observation binding is invalid")
            wrapper_body = verified_watcher_files[tree_observation]
            wrapper = _json_bytes(wrapper_body, "public-header source tree wrapper")
            if not isinstance(wrapper, dict):
                raise IntegrityError("public-header source tree wrapper identity is invalid")
            source, key = validate_observation_identity(tree_observation, wrapper)
            wrapper_observed_at = observation_timestamp(tree_observation, wrapper)
            if source_row.get("observed_at") != wrapper_observed_at:
                raise IntegrityError(
                    "public-header source tree observation timestamp is inconsistent"
                )
            _parse_utc_timestamp(
                source_row["observed_at"], "public-header tree observed_at"
            )
            parts = key.split("/")
            if (
                source != "hf-tree"
                or len(parts) != 4
                or "/".join(parts[:2]) != row["repository"]
                or parts[2] != row["revision"]
                or HF_TREE_PAGE_RE.fullmatch(parts[3]) is None
            ):
                raise IntegrityError("public-header source tree wrapper scope is invalid")
            tree_digest = wrapper.get("content_sha256")
            if (
                not isinstance(tree_digest, str)
                or SHA256_RE.fullmatch(tree_digest) is None
                or tree_digest != source_row.get("tree_content_sha256")
                or wrapper.get("object") != _sha_relative(tree_digest)
            ):
                raise IntegrityError("public-header source tree wrapper CAS binding is invalid")
            tree_body = verified_watcher_files.get(_sha_relative(tree_digest))
            if tree_body is None or sha256_bytes(tree_body) != tree_digest:
                raise IntegrityError("public-header source tree content is absent or invalid")
            try:
                tree_value = json.loads(tree_body)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise IntegrityError("public-header source tree content is not JSON") from exc
            matching_tree_rows = [
                entry
                for entry in _safe_tree_entries(tree_value)
                if entry["type"] == "file" and entry["path"] == row["path"]
            ]
            if len(matching_tree_rows) != 1:
                raise IntegrityError("public-header source tree does not contain its checkpoint")
            matching_tree = matching_tree_rows[0]
            if (
                matching_tree.get("lfs_sha256") != row["lfs_sha256"]
                or matching_tree.get("lfs_bytes") != row["lfs_bytes"]
            ):
                raise IntegrityError("public-header association contradicts its source tree entry")
        key = (row["repository"], row["revision"], row["path"], row["lfs_sha256"])
        if key in headers_by_key and headers_by_key[key] != row:
            raise IntegrityError("conflicting public-header associations")
        headers_by_key[key] = row

    repos: dict[str, dict[str, Any]] = {}
    for observation in observations:
        wrapper = observation["wrapper"]
        source = wrapper.get("source")
        try:
            content = json.loads(observation["content"])
        except json.JSONDecodeError as exc:
            raise IntegrityError("raw-watcher observation CAS is not JSON") from exc
        if source == "hf-model":
            if not isinstance(content, dict):
                raise IntegrityError("hf-model response is not an object")
            repo = content.get("id") or content.get("modelId")
            revision = content.get("sha")
            if not isinstance(repo, str) or not isinstance(revision, str):
                raise IntegrityError("hf-model response has no immutable head")
            task = _public_repo_task(repo, tournament_id, r1_task_ids)
            if wrapper.get("key") != repo:
                raise IntegrityError("hf-model wrapper key contradicts its response identity")
            if content.get("modelId") is not None and content.get("modelId") != repo:
                raise IntegrityError("hf-model response contains conflicting repository identities")
            if REVISION_RE.fullmatch(revision) is None:
                raise IntegrityError("hf-model response revision is not immutable")
            record = repos.setdefault(
                repo,
                {
                    "task_id": task,
                    "heads": [],
                    "manifests": {},
                    "trees": [],
                    "file_observations": [],
                },
            )
            record["heads"].append(
                {
                    "revision": revision,
                    "observed_at": observation_timestamp(observation["path"], wrapper),
                    "observation": observation["path"],
                    "observation_sha256": observation["wrapper_sha256"],
                }
            )
        elif source == "hf-revision-manifest":
            if not isinstance(content, dict):
                raise IntegrityError("HF revision manifest is not an object")
            repo = content.get("repo_id")
            revision = content.get("revision")
            if not isinstance(repo, str) or not isinstance(revision, str):
                raise IntegrityError("HF revision manifest has no repository/revision")
            task = _public_repo_task(repo, tournament_id, r1_task_ids)
            if wrapper.get("key") != f"{repo}/{revision}":
                raise IntegrityError("HF revision-manifest key contradicts its response identity")
            if REVISION_RE.fullmatch(revision) is None:
                raise IntegrityError("HF revision manifest revision is not immutable")
            record = repos.setdefault(
                repo,
                {
                    "task_id": task,
                    "heads": [],
                    "manifests": {},
                    "trees": [],
                    "file_observations": [],
                },
            )
            prior = record["manifests"].get(revision)
            manifest_record = {
                "body": content,
                "observed_at": observation_timestamp(observation["path"], wrapper),
                "observation": observation["path"],
                "observation_sha256": observation["wrapper_sha256"],
                "content_sha256": observation["content_sha256"],
            }
            if prior is None or manifest_record["observed_at"] > prior["observed_at"]:
                record["manifests"][revision] = manifest_record
        elif source == "hf-tree":
            key = wrapper.get("key")
            if not isinstance(key, str):
                raise IntegrityError("hf-tree observation has no repository/revision key")
            parts = key.split("/")
            if (
                len(parts) != 4
                or parts[0] != "gradients-io-tournaments"
                or HF_TREE_PAGE_RE.fullmatch(parts[3]) is None
            ):
                raise IntegrityError("hf-tree observation key is malformed")
            repo = "/".join(parts[:2])
            revision = parts[2]
            page_number = int(HF_TREE_PAGE_RE.fullmatch(parts[3]).group(1))
            if page_number < 1:
                raise IntegrityError("hf-tree page numbering must start at one")
            task = _public_repo_task(repo, tournament_id, r1_task_ids)
            if REVISION_RE.fullmatch(revision) is None:
                raise IntegrityError("hf-tree observation is outside the selected R1 scope")
            record = repos.setdefault(
                repo,
                {
                    "task_id": task,
                    "heads": [],
                    "manifests": {},
                    "trees": [],
                    "file_observations": [],
                },
            )
            record["trees"].append(
                {
                    "revision": revision,
                    "observed_at": observation_timestamp(observation["path"], wrapper),
                    "observation": observation["path"],
                    "observation_sha256": observation["wrapper_sha256"],
                    "content_sha256": observation["content_sha256"],
                    "page_number": page_number,
                    "entries": _safe_tree_entries(content),
                }
            )
        elif source == "hf-file":
            if not isinstance(content, dict):
                raise IntegrityError("HF file observation is not an object")
            repo = content.get("repo_id")
            revision = content.get("revision")
            if (
                not isinstance(repo, str)
                or not isinstance(revision, str)
                or REVISION_RE.fullmatch(revision) is None
            ):
                raise IntegrityError("HF file observation is outside the selected R1")
            task = _public_repo_task(repo, tournament_id, r1_task_ids)
            path = _normalized_relative(content.get("path"), "HF file observation path")
            if wrapper.get("key") != f"{repo}/{revision}/{path}":
                raise IntegrityError("HF file wrapper key contradicts its response identity")
            record = repos.setdefault(
                repo,
                {
                    "task_id": task,
                    "heads": [],
                    "manifests": {},
                    "trees": [],
                    "file_observations": [],
                },
            )
            record["file_observations"].append(
                {
                    "revision": revision,
                    "path": path,
                    "kind": content.get("kind"),
                    "declared_bytes": content.get("declared_size"),
                    "captured_bytes": content.get("bytes"),
                    "captured_object_sha256": content.get("object_sha256"),
                    "observed_at": observation_timestamp(observation["path"], wrapper),
                    "observation": observation["path"],
                    "observation_sha256": observation["wrapper_sha256"],
                    "content_sha256": observation["content_sha256"],
                }
            )

    missing: list[dict[str, Any]] = []
    public: dict[str, Any] = {}
    for repo, source_record in sorted(repos.items()):
        heads = sorted(source_record["heads"], key=lambda row: (row["observed_at"], row["revision"]))
        observed_head = heads[-1]["revision"] if heads else None
        manifest_revisions = set(source_record["manifests"])
        head_revisions = {row["revision"] for row in source_record["heads"]}
        tree_revisions = {row["revision"] for row in source_record["trees"]}
        file_revisions = {row["revision"] for row in source_record["file_observations"]}
        header_revisions = {
            revision
            for header_repo, revision, _path, _oid in headers_by_key
            if header_repo == repo
        }
        if not heads:
            missing.append(
                {
                    "class": "public-repository-has-no-model-head",
                    "task_id": source_record["task_id"],
                    "repository": repo,
                }
            )
        for revision in sorted(tree_revisions - manifest_revisions):
            missing.append(
                {
                    "class": "tree-revision-has-no-manifest",
                    "task_id": source_record["task_id"],
                    "repository": repo,
                    "revision": revision,
                }
            )
        for revision in sorted(file_revisions - manifest_revisions):
            missing.append(
                {
                    "class": "file-revision-has-no-manifest",
                    "task_id": source_record["task_id"],
                    "repository": repo,
                    "revision": revision,
                }
            )
        for revision in sorted(header_revisions - manifest_revisions):
            missing.append(
                {
                    "class": "header-revision-has-no-manifest",
                    "task_id": source_record["task_id"],
                    "repository": repo,
                    "revision": revision,
                }
            )
        for revision in sorted(head_revisions - manifest_revisions):
            missing.append(
                {
                    "class": "public-head-revision-has-no-manifest",
                    "task_id": source_record["task_id"],
                    "repository": repo,
                    "revision": revision,
                }
            )
        revisions: list[dict[str, Any]] = []
        for revision, raw_record in sorted(source_record["manifests"].items()):
            manifest = raw_record["body"]
            for required in (
                "capture_complete",
                "processing_complete",
                "tree_truncated",
                "tree_entry_count",
                "tree_file_count",
                "captures",
                "config_absent",
                "configs",
                "eligible_weight_plan",
                "failures",
                "skipped",
            ):
                if required not in manifest:
                    raise IntegrityError("HF revision manifest lacks a required completeness field")
            captures = manifest["captures"]
            configs = manifest["configs"]
            weights = manifest["eligible_weight_plan"]
            if not isinstance(captures, list) or not isinstance(configs, list) or not isinstance(weights, list):
                raise IntegrityError("HF revision manifest arrays are invalid")
            normalized_config_paths: list[str] = []
            for config_path in configs:
                normalized = _normalized_relative(config_path, "HF config path")
                if normalized in normalized_config_paths:
                    raise IntegrityError("HF revision manifest contains a duplicate config path")
                normalized_config_paths.append(normalized)
            config_absent = manifest.get("config_absent")
            if type(config_absent) is not bool or config_absent != (
                len(normalized_config_paths) == 0
            ):
                raise IntegrityError("HF revision manifest config-absence state is inconsistent")
            capture_by_path: dict[str, dict[str, Any]] = {}
            excluded_capture_count = 0
            for capture in captures:
                if not isinstance(capture, dict) or not isinstance(capture.get("path"), str):
                    raise IntegrityError("HF revision manifest capture is malformed")
                path = _normalized_relative(capture["path"], "HF manifest capture path")
                if contains_forbidden_path(path):
                    excluded_capture_count += 1
                    continue
                if path in capture_by_path:
                    raise IntegrityError("HF revision manifest contains a duplicate capture path")
                object_sha = capture.get("object_sha256")
                if capture.get("captured") is True:
                    if not isinstance(object_sha, str) or SHA256_RE.fullmatch(object_sha) is None:
                        raise IntegrityError("captured HF file has no valid object identity")
                    relative = _sha_relative(object_sha)
                    object_body = verified_watcher_files.get(relative)
                    if object_body is None or sha256_bytes(object_body) != object_sha:
                        raise IntegrityError("HF manifest capture object is absent or invalid")
                    if capture.get("bytes") not in (None, len(object_body)):
                        raise IntegrityError("HF manifest capture object size mismatch")
                capture_by_path[path] = capture

            tree_pages = _canonical_tree_pages(
                row for row in source_record["trees"] if row["revision"] == revision
            )
            file_tree = _merged_tree_summary(tree_pages)
            if not tree_pages:
                missing.append(
                    {
                        "class": "repository-revision-tree-not-captured",
                        "task_id": source_record["task_id"],
                        "repository": repo,
                        "revision": revision,
                    }
                )
            tree_entry_count = manifest.get("tree_entry_count")
            tree_file_count = manifest.get("tree_file_count")
            if (
                type(tree_entry_count) is not int
                or tree_entry_count < 0
                or type(tree_file_count) is not int
                or tree_file_count < 0
            ):
                raise IntegrityError("HF revision manifest tree counts are invalid")
            if tree_entry_count != len(file_tree) or tree_file_count != sum(
                row["type"] == "file" for row in file_tree
            ):
                missing.append(
                    {
                        "class": "manifest-tree-counts-conflict-with-immutable-tree",
                        "task_id": source_record["task_id"],
                        "repository": repo,
                        "revision": revision,
                    }
                )

            tree_files_by_path = {
                row["path"]: row for row in file_tree if row["type"] == "file"
            }
            file_observations_by_path: dict[str, list[dict[str, Any]]] = {}
            for file_observation in source_record["file_observations"]:
                if file_observation["revision"] != revision:
                    continue
                file_observations_by_path.setdefault(
                    file_observation["path"], []
                ).append(file_observation)
            for path, capture in sorted(capture_by_path.items()):
                tree_file = tree_files_by_path.get(path)
                if tree_file is None:
                    missing.append(
                        {
                            "class": "manifest-capture-absent-from-immutable-tree",
                            "task_id": source_record["task_id"],
                            "repository": repo,
                            "revision": revision,
                            "path": path,
                        }
                    )
                    continue
                declared_size = capture.get("declared_size")
                if declared_size is None:
                    missing.append(
                        {
                            "class": "manifest-capture-declared-size-absent",
                            "task_id": source_record["task_id"],
                            "repository": repo,
                            "revision": revision,
                            "path": path,
                        }
                    )
                    effective_declared_size = tree_file["bytes"]
                elif type(declared_size) is int and declared_size >= 0:
                    effective_declared_size = declared_size
                else:
                    raise IntegrityError("HF manifest capture declared size is invalid")
                if effective_declared_size != tree_file["bytes"]:
                    raise IntegrityError("HF manifest capture size contradicts immutable tree")
                captured = capture.get("captured") is True
                object_sha = capture.get("object_sha256")
                object_bytes = capture.get("bytes")
                if captured:
                    body = verified_watcher_files[_sha_relative(object_sha)]
                    if type(object_bytes) is not int or object_bytes != len(body):
                        raise IntegrityError("HF manifest capture byte count is invalid")
                    if object_bytes != tree_file["bytes"]:
                        raise IntegrityError("HF captured bytes contradict immutable tree size")
                observations_for_path = file_observations_by_path.get(path, [])
                if not observations_for_path:
                    missing.append(
                        {
                            "class": "manifest-capture-has-no-hf-file-observation",
                            "task_id": source_record["task_id"],
                            "repository": repo,
                            "revision": revision,
                            "path": path,
                        }
                    )
                    continue
                for file_observation in observations_for_path:
                    if (
                        file_observation.get("kind") != capture.get("kind")
                        or file_observation.get("declared_bytes") != tree_file["bytes"]
                        or file_observation.get("captured_bytes") != object_bytes
                        or file_observation.get("captured_object_sha256") != object_sha
                    ):
                        raise IntegrityError(
                            "HF file observation contradicts its manifest capture"
                        )
            for path in sorted(set(file_observations_by_path) - set(capture_by_path)):
                missing.append(
                    {
                        "class": "hf-file-observation-absent-from-manifest-captures",
                        "task_id": source_record["task_id"],
                        "repository": repo,
                        "revision": revision,
                        "path": path,
                    }
                )

            tree_config_paths = {
                row["path"]
                for row in file_tree
                if row["type"] == "file"
                and PurePosixPath(row["path"]).name.casefold() == "config.yaml"
            }
            manifest_config_paths = set(normalized_config_paths)
            for path in sorted(tree_config_paths - manifest_config_paths):
                missing.append(
                    {
                        "class": "tree-config-omitted-from-manifest",
                        "task_id": source_record["task_id"],
                        "repository": repo,
                        "revision": revision,
                        "path": path,
                    }
                )
            for path in sorted(manifest_config_paths - tree_config_paths):
                missing.append(
                    {
                        "class": "manifest-config-absent-from-immutable-tree",
                        "task_id": source_record["task_id"],
                        "repository": repo,
                        "revision": revision,
                        "path": path,
                    }
                )
            for path in sorted(tree_config_paths):
                capture = capture_by_path.get(path)
                if capture is None or capture.get("captured") is not True:
                    missing.append(
                        {
                            "class": "tree-config-bytes-not-captured",
                            "task_id": source_record["task_id"],
                            "repository": repo,
                            "revision": revision,
                            "path": path,
                        }
                    )

            plan_by_path: dict[str, tuple[str, int]] = {}
            for weight in weights:
                if not isinstance(weight, dict):
                    raise IntegrityError("HF eligible checkpoint row is malformed")
                path = _normalized_relative(weight.get("path"), "HF checkpoint plan path")
                oid = weight.get("lfs_oid")
                size = weight.get("size")
                if not isinstance(path, str) or contains_forbidden_path(path):
                    raise IntegrityError("HF checkpoint plan contains a prohibited/invalid path")
                if (
                    not isinstance(oid, str)
                    or SHA256_RE.fullmatch(oid) is None
                    or not isinstance(size, int)
                    or size <= 8
                ):
                    raise IntegrityError("HF checkpoint plan has an invalid LFS identity")
                if path in plan_by_path:
                    raise IntegrityError("HF checkpoint plan contains a duplicate path")
                plan_by_path[path] = (oid, size)

            tree_by_path: dict[str, tuple[str, int]] = {}
            for tree_entry in file_tree:
                path = tree_entry["path"]
                if tree_entry["type"] != "file" or not path.lower().endswith(".safetensors"):
                    continue
                oid = tree_entry.get("lfs_sha256")
                size = tree_entry.get("lfs_bytes")
                if (
                    not isinstance(oid, str)
                    or SHA256_RE.fullmatch(oid) is None
                    or not isinstance(size, int)
                    or size <= 8
                ):
                    raise IntegrityError("HF safetensors tree entry lacks a valid LFS identity")
                if path in tree_by_path and tree_by_path[path] != (oid, size):
                    raise IntegrityError("HF tree pages conflict on a checkpoint identity")
                tree_by_path[path] = (oid, size)

            for path, identity in sorted(plan_by_path.items()):
                tree_identity = tree_by_path.get(path)
                if tree_identity is None:
                    missing.append(
                        {
                            "class": "eligible-checkpoint-absent-from-immutable-tree",
                            "task_id": source_record["task_id"],
                            "repository": repo,
                            "revision": revision,
                            "path": path,
                        }
                    )
                elif tree_identity != identity:
                    raise IntegrityError("checkpoint plan and immutable tree identities disagree")
            for path in sorted(set(tree_by_path) - set(plan_by_path)):
                missing.append(
                    {
                        "class": "tree-checkpoint-omitted-from-eligible-plan",
                        "task_id": source_record["task_id"],
                        "repository": repo,
                        "revision": revision,
                        "path": path,
                    }
                )

            # The immutable tree is authoritative for public checkpoint
            # completeness.  Include every tree checkpoint even when the
            # watcher's optimization-oriented eligible plan omitted it.
            checkpoint_rows: list[dict[str, Any]] = []
            for path, (oid, size) in sorted(tree_by_path.items()):
                header = headers_by_key.get((repo, revision, path, oid))
                if header is None:
                    missing.append(
                        {
                            "class": "checkpoint-header-not-captured",
                            "task_id": source_record["task_id"],
                            "repository": repo,
                            "revision": revision,
                            "path": path,
                        }
                    )
                    checkpoint_rows.append(
                        {
                            "path": path,
                            "lfs_sha256": oid,
                            "lfs_bytes": size,
                            "header_available": False,
                            "header_sha256": None,
                            "metadata": None,
                            "tensor_count": None,
                        }
                    )
                else:
                    checkpoint_rows.append(
                        {
                            "path": path,
                            "lfs_sha256": oid,
                            "lfs_bytes": size,
                            "header_available": True,
                            "header_sha256": header["header_sha256"],
                            "metadata": header["metadata"],
                            "tensor_count": header["tensor_count"],
                            "header_record_sha256": header["record_sha256"],
                        }
                    )

            config_rows: list[dict[str, Any]] = []
            for config_path in normalized_config_paths:
                if PurePosixPath(config_path).name.lower() != "config.yaml":
                    continue
                capture = capture_by_path.get(config_path)
                if not capture or capture.get("captured") is not True:
                    missing.append(
                        {
                            "class": "config-listed-but-bytes-not-captured",
                            "task_id": source_record["task_id"],
                            "repository": repo,
                            "revision": revision,
                        }
                    )
                    continue
                object_sha = capture.get("object_sha256")
                config_body = verified_watcher_files[_sha_relative(object_sha)]
                projection, error = _recipe_projection(config_body)
                config_rows.append(
                    {
                        "path": config_path,
                        "sha256": object_sha,
                        "bytes": len(config_body),
                        "recipe_projection": projection,
                        "projection_error": error,
                    }
                )
                if error:
                    missing.append(
                        {
                            "class": "config-recipe-projection-unavailable",
                            "task_id": source_record["task_id"],
                            "repository": repo,
                            "revision": revision,
                        }
                    )

            capture_summaries: list[dict[str, Any]] = []
            telemetry: list[dict[str, Any]] = []
            for path, capture in sorted(capture_by_path.items()):
                object_sha = capture.get("object_sha256")
                captured = capture.get("captured") is True
                summary = {
                    "path": path,
                    "kind": capture.get("kind"),
                    "captured": captured,
                    "declared_bytes": capture.get("declared_size"),
                    "captured_bytes": capture.get("bytes"),
                    "object_sha256": object_sha if captured else None,
                }
                capture_summaries.append(summary)
                if not captured or not isinstance(object_sha, str):
                    continue
                body = verified_watcher_files[_sha_relative(object_sha)]
                projection = _public_telemetry_projection(path, body)
                name = PurePosixPath(path).name.lower()
                if projection is not None or name in {
                    "success.txt",
                    "forge_run.json",
                    "checkpoint_scope.json",
                    "forge_checkpoint_selection.json",
                    ".krea2_checkpoint_evaluations.json",
                }:
                    telemetry.append(
                        {
                            "path": path,
                            "sha256": object_sha,
                            "bytes": len(body),
                            "projection": projection,
                        }
                    )

            manifest_complete = (
                manifest.get("capture_complete") is True
                and manifest.get("processing_complete") is True
                and manifest.get("tree_truncated") is False
                and manifest.get("failures") == []
            )
            if not manifest_complete:
                missing.append(
                    {
                        "class": "repository-revision-capture-partial-or-truncated",
                        "task_id": source_record["task_id"],
                        "repository": repo,
                        "revision": revision,
                    }
                )
            revisions.append(
                {
                    "revision": revision,
                    "is_latest_observed_head": revision == observed_head,
                    "manifest_complete": manifest_complete,
                    "capture_complete": manifest.get("capture_complete"),
                    "processing_complete": manifest.get("processing_complete"),
                    "tree_truncated": manifest.get("tree_truncated"),
                    "failures": manifest.get("failures"),
                    "skipped_count": len(manifest.get("skipped", []))
                    if isinstance(manifest.get("skipped"), list)
                    else None,
                    "config_absent": manifest.get("config_absent"),
                    "configs": config_rows,
                    "checkpoint_identity": checkpoint_identity_summary(checkpoint_rows),
                    "file_tree": file_tree,
                    "file_tree_page_provenance": [
                        {key: value for key, value in row.items() if key != "entries"}
                        for row in sorted(
                            tree_pages,
                            key=lambda row: (
                                row["observed_at"],
                                row["observation"],
                            ),
                        )
                    ],
                    "public_file_captures": capture_summaries,
                    "public_run_telemetry": telemetry,
                    "public_file_observations": sorted(
                        [
                            row
                            for row in source_record["file_observations"]
                            if row["revision"] == revision
                        ],
                        key=lambda row: (row["observed_at"], row["path"], row["observation"]),
                    ),
                    "excluded_capture_count": excluded_capture_count,
                    "manifest_observation": raw_record["observation"],
                    "manifest_observation_sha256": raw_record["observation_sha256"],
                    "manifest_content_sha256": raw_record["content_sha256"],
                }
            )
        if observed_head is not None and observed_head not in source_record["manifests"]:
            missing.append(
                {
                    "class": "latest-public-head-has-no-revision-manifest",
                    "task_id": source_record["task_id"],
                    "repository": repo,
                    "revision": observed_head,
                }
            )
        if not source_record["manifests"]:
            missing.append(
                {
                    "class": "public-repository-has-no-revision-manifest",
                    "task_id": source_record["task_id"],
                    "repository": repo,
                }
            )
        public[repo] = {
            "repository": repo,
            "task_id": source_record["task_id"],
            "latest_observed_head": observed_head,
            "head_observations": heads,
            "tree_observations": sorted(
                [
                    {key: value for key, value in row.items() if key != "entries"}
                    for row in source_record["trees"]
                ],
                key=lambda row: (row["observed_at"], row["revision"], row["observation"]),
            ),
            "revisions": revisions,
        }
    return public, missing


def _descriptor_json(body: bytes, label: str) -> dict[str, Any]:
    """Decode an evidence descriptor without dataset-body vocabulary checks.

    Collector receipts explicitly document the surfaces they did not inspect, so
    their metadata contains exclusion words by design.  Only descriptor fields
    selected below are projected into the P0 package.
    """
    try:
        value = json.loads(
            body,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise IntegrityError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"{label} is not an object")
    return value


def _require_schema(value: Mapping[str, Any], schema: str, label: str) -> None:
    version = value.get("schema_version")
    if (
        value.get("schema") != schema
        or type(version) is not int
        or version != TRAINING_ARCHIVE_SCHEMA_VERSION
    ):
        raise IntegrityError(f"{label} schema/version is invalid")


def _relative_beneath(root: Path, path: Path, label: str) -> str:
    try:
        relative = path.absolute().relative_to(root).as_posix()
    except ValueError as exc:
        raise IntegrityError(f"{label} is outside its declared root") from exc
    return _normalized_relative(relative, label)


def _plain_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise IntegrityError(f"{label} is invalid")
    return value


def _stream_regular_identity(
    root: Path,
    relative: str,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    """Hash one archive through a descriptor without retaining its byte body."""
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise IntegrityError("public training archive SHA-256 identity is invalid")
    if expected_bytes <= 0 or expected_bytes > MAX_TRAINING_ARCHIVE_BYTES:
        raise IntegrityError("public training archive size identity is invalid")
    fd = _open_regular_beneath(root, relative)
    digest = hashlib.sha256()
    observed_bytes = 0
    try:
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            observed_bytes += len(block)
            if observed_bytes > expected_bytes:
                raise IntegrityError("public training archive size mismatch")
            digest.update(block)
    finally:
        os.close(fd)
    if observed_bytes != expected_bytes or digest.hexdigest() != expected_sha256:
        raise IntegrityError("public training archive SHA/size mismatch")


def _verify_and_reinventory_training_zip(
    root: Path, relative: str, expected_sha256: str, expected_bytes: int
) -> dict[str, Any]:
    """Hash and independently inventory one ZIP through the same stable fd."""
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise IntegrityError("public training archive SHA-256 identity is invalid")
    if expected_bytes <= 0 or expected_bytes > MAX_TRAINING_ARCHIVE_BYTES:
        raise IntegrityError("public training archive size identity is invalid")
    fd = _open_regular_beneath(root, relative)
    try:
        handle = os.fdopen(fd, "rb", closefd=False)
        digest = hashlib.sha256()
        observed_bytes = 0
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            observed_bytes += len(block)
            if observed_bytes > expected_bytes:
                raise IntegrityError("public training archive size mismatch")
            digest.update(block)
        if observed_bytes != expected_bytes or digest.hexdigest() != expected_sha256:
            raise IntegrityError("public training archive SHA/size mismatch")
        handle.seek(0)
        try:
            return _inventory_zip_file(handle)
        except IntegrityError:
            raise
        except (OSError, ValueError) as exc:
            raise IntegrityError("public training archive reinventory failed") from exc
    finally:
        try:
            handle.close()
        except UnboundLocalError:
            pass
        os.close(fd)


def _training_zip_counts(value: Any) -> tuple[int, int]:
    """Validate receipt member identities and derive both requested image counts."""
    if not isinstance(value, dict):
        raise IntegrityError("COMPLETE public training receipt has no ZIP inventory")
    members = value.get("members")
    if not isinstance(members, list) or not members:
        raise IntegrityError("public training ZIP member inventory is invalid")
    if _plain_nonnegative_int(value.get("member_count"), "training ZIP member count") != len(
        members
    ):
        raise IntegrityError("public training ZIP member count mismatch")

    image_hashes: set[str] = set()
    image_stems: set[tuple[str, str]] = set()
    caption_stems: set[tuple[str, str]] = set()
    seen_paths: set[str] = set()
    image_count = 0
    caption_count = 0
    uncompressed_bytes = 0
    for index, member in enumerate(members):
        if not isinstance(member, dict):
            raise IntegrityError("public training ZIP member row is invalid")
        if member.get("central_directory_index") != index or type(
            member.get("central_directory_index")
        ) is not int:
            raise IntegrityError("public training ZIP central-directory order is invalid")
        path = _normalized_relative(member.get("path"), "public training ZIP member path")
        identity = path.casefold()
        if identity in seen_paths:
            raise IntegrityError("public training ZIP member inventory contains a duplicate path")
        seen_paths.add(identity)
        kind = member.get("kind")
        if kind not in {"image", "caption", "other", "directory"}:
            raise IntegrityError("public training ZIP member kind is invalid")
        suffix = PurePosixPath(path).suffix.casefold()
        expected_kind = (
            "directory"
            if kind == "directory"
            else "image"
            if suffix in TRAINING_IMAGE_SUFFIXES
            else "caption"
            if suffix in TRAINING_CAPTION_SUFFIXES
            else "other"
        )
        if kind != expected_kind:
            raise IntegrityError("public training ZIP member kind contradicts its path")
        member_bytes = _plain_nonnegative_int(
            member.get("uncompressed_bytes"), "training ZIP member size"
        )
        _plain_nonnegative_int(member.get("compressed_bytes"), "training ZIP compressed size")
        uncompressed_bytes += member_bytes
        member_sha = member.get("sha256")
        if kind == "directory":
            if member_sha is not None or member_bytes != 0:
                raise IntegrityError("public training ZIP directory identity is invalid")
            continue
        if not isinstance(member_sha, str) or SHA256_RE.fullmatch(member_sha) is None:
            raise IntegrityError("public training ZIP member SHA-256 identity is invalid")
        pure = PurePosixPath(path)
        stem = (pure.parent.as_posix(), pure.stem)
        if kind == "image":
            image_count += 1
            image_hashes.add(member_sha)
            image_stems.add(stem)
        elif kind == "caption":
            caption_count += 1
            caption_stems.add(stem)

    declared_image_count = _plain_nonnegative_int(
        value.get("image_member_count"), "training ZIP image count"
    )
    declared_dedup_count = _plain_nonnegative_int(
        value.get("post_exact_byte_dedup_image_count"),
        "training ZIP post-byte-dedup image count",
    )
    checks = (
        (declared_image_count, image_count, "public training ZIP image count mismatch"),
        (
            declared_dedup_count,
            len(image_hashes),
            "public training ZIP post-byte-dedup count mismatch",
        ),
        (
            _plain_nonnegative_int(value.get("caption_member_count"), "training ZIP caption count"),
            caption_count,
            "public training ZIP caption count mismatch",
        ),
        (
            _plain_nonnegative_int(value.get("paired_stem_count"), "training ZIP paired-stem count"),
            len(image_stems & caption_stems),
            "public training ZIP paired-stem count mismatch",
        ),
        (
            _plain_nonnegative_int(
                value.get("uncompressed_bytes"), "training ZIP uncompressed byte count"
            ),
            uncompressed_bytes,
            "public training ZIP uncompressed byte count mismatch",
        ),
    )
    for declared, derived, message in checks:
        if declared != derived:
            raise IntegrityError(message)
    if image_count == 0 or declared_dedup_count == 0:
        raise IntegrityError("public training ZIP contains no recognized image members")
    return image_count, declared_dedup_count


def load_public_training_archive_inventory(
    root: Path,
    inventory_path: Path,
    tournament_id: str,
    r1_task_ids: set[str],
    *,
    api_snapshot_path: Path,
    api_snapshot_sha256: str,
    api_observed_at: dt.datetime,
    checksum_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Verify and project the exact completed-R1 public training receipts."""
    training_root = _check_root_components(root)
    root_identity_body = _read_regular(training_root, "ROOT-IDENTITY.json", 1024 * 1024)
    root_identity = _descriptor_json(root_identity_body, "public training root identity")
    _require_schema(root_identity, TRAINING_ARCHIVE_ROOT_SCHEMA, "public training root identity")
    if root_identity.get("tournament_id") != tournament_id:
        raise IntegrityError("public training root tournament identity mismatch")

    inventory_relative = _relative_beneath(
        training_root, inventory_path, "public training inventory path"
    )
    if (
        not inventory_relative.startswith("inventories/")
        or not inventory_relative.endswith(".json")
    ):
        raise IntegrityError("public training inventory path is outside its descriptor namespace")
    inventory_body = _read_regular(training_root, inventory_relative)
    inventory_sha = sha256_bytes(inventory_body)
    inventory = _descriptor_json(inventory_body, "public training inventory")
    _require_schema(inventory, TRAINING_ARCHIVE_INVENTORY_SCHEMA, "public training inventory")

    selected_checksum = checksum_path or inventory_path.with_suffix(".sha256")
    checksum_relative = _relative_beneath(
        training_root, selected_checksum, "public training inventory checksum path"
    )
    checksum_body = _read_regular(training_root, checksum_relative, 4096)
    expected_checksum = (
        f"{inventory_sha}  {PurePosixPath(inventory_relative).name}\n".encode("ascii")
    )
    if checksum_body != expected_checksum:
        raise IntegrityError("public training inventory checksum sidecar is invalid")

    if (
        inventory.get("tournament_id") != tournament_id
        or inventory.get("round_number") != 1
        or type(inventory.get("round_number")) is not int
        or inventory.get("round_status") != "completed"
    ):
        raise IntegrityError("public training inventory tournament/Round-1 identity mismatch")
    inventory_observed_at = _parse_utc_timestamp(
        inventory.get("observed_at"), "public training inventory observed_at"
    )
    if inventory_observed_at < api_observed_at:
        raise IntegrityError("public training inventory predates its safe API snapshot")
    snapshot_identity = inventory.get("input_safe_api_snapshot")
    if (
        not isinstance(snapshot_identity, dict)
        or snapshot_identity.get("filename") != api_snapshot_path.name
        or snapshot_identity.get("sha256") != api_snapshot_sha256
    ):
        raise IntegrityError("public training inventory safe API snapshot identity mismatch")
    if inventory.get("rights") != TRAINING_ARCHIVE_RIGHTS:
        raise IntegrityError("public training inventory rights boundary is invalid")

    task_rows = inventory.get("tasks")
    if not isinstance(task_rows, list):
        raise IntegrityError("public training inventory has no task rows")
    if _plain_nonnegative_int(inventory.get("task_count"), "public training task count") != len(
        task_rows
    ):
        raise IntegrityError("public training inventory task count mismatch")
    if len(task_rows) != len(r1_task_ids):
        raise IntegrityError("public training inventory does not contain exactly one row per R1 task")

    results: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    complete_count = 0
    verified_archive_bytes = 0
    for task_row in task_rows:
        if not isinstance(task_row, dict):
            raise IntegrityError("public training inventory task row is invalid")
        task_id = task_row.get("task_id")
        if (
            not isinstance(task_id, str)
            or TASK_RE.fullmatch(task_id) is None
            or task_id not in r1_task_ids
            or task_id in seen_tasks
        ):
            raise IntegrityError("public training inventory task identity mismatch or duplicate")
        seen_tasks.add(task_id)
        row_status = task_row.get("status")
        if row_status not in {"COMPLETE", "PARTIAL"}:
            raise IntegrityError("public training inventory task status is invalid")
        receipt_relative = _normalized_relative(
            task_row.get("receipt"), "public training receipt path"
        )
        receipt_parts = PurePosixPath(receipt_relative).parts
        if (
            len(receipt_parts) != 3
            or receipt_parts[:2] != ("receipts", task_id)
            or not receipt_parts[2].endswith(".json")
        ):
            raise IntegrityError("public training receipt path/task binding is invalid")
        receipt_sha = task_row.get("receipt_sha256")
        if not isinstance(receipt_sha, str) or SHA256_RE.fullmatch(receipt_sha) is None:
            raise IntegrityError("public training receipt SHA-256 identity is invalid")
        receipt_body = _read_regular(training_root, receipt_relative)
        if sha256_bytes(receipt_body) != receipt_sha:
            raise IntegrityError("public training receipt checksum mismatch")
        receipt = _descriptor_json(receipt_body, "public training receipt")
        _require_schema(receipt, TRAINING_ARCHIVE_RECEIPT_SCHEMA, "public training receipt")
        if (
            receipt.get("tournament_id") != tournament_id
            or receipt.get("task_id") != task_id
            or receipt.get("status") != row_status
        ):
            raise IntegrityError("public training receipt tournament/task/status identity mismatch")
        if receipt.get("observed_at") != inventory.get("observed_at"):
            raise IntegrityError("public training receipt observation identity mismatch")
        if receipt.get("input_safe_api_snapshot_sha256") != api_snapshot_sha256:
            raise IntegrityError("public training receipt safe API snapshot identity mismatch")
        if receipt.get("source_task_endpoint") != f"https://api.gradients.io/auditing/tasks/{task_id}":
            raise IntegrityError("public training receipt exact-task endpoint identity mismatch")
        response_sha = receipt.get("source_response_sha256")
        if not isinstance(response_sha, str) or SHA256_RE.fullmatch(response_sha) is None:
            raise IntegrityError("public training receipt source-response identity is invalid")
        if _plain_nonnegative_int(
            receipt.get("source_response_bytes"), "public training source-response byte count"
        ) == 0:
            raise IntegrityError("public training receipt source response is empty")
        task_status = receipt.get("task_status")
        if (
            not isinstance(task_status, str)
            or task_status.casefold() not in TRAINING_ARCHIVE_TERMINAL_TASK_STATUSES
        ):
            raise IntegrityError("public training receipt task status is not terminal")
        if receipt.get("rights") != TRAINING_ARCHIVE_RIGHTS:
            raise IntegrityError("public training receipt rights boundary is invalid")
        if receipt.get("exclusion_contract") != TRAINING_ARCHIVE_EXCLUSION_CONTRACT:
            raise IntegrityError("public training receipt exclusion contract is invalid")

        source: dict[str, Any] = {
            "inventory": inventory_relative,
            "inventory_sha256": inventory_sha,
            "receipt": receipt_relative,
            "receipt_sha256": receipt_sha,
        }
        if row_status == "COMPLETE":
            transfer = receipt.get("source_training_archive")
            if not isinstance(transfer, dict):
                raise IntegrityError("COMPLETE public training receipt has no transfer provenance")
            transfer_url = transfer.get("url")
            if not isinstance(transfer_url, str):
                raise IntegrityError("public training transfer URL provenance is invalid")
            try:
                transfer_parts = urlsplit(transfer_url)
            except ValueError as exc:
                raise IntegrityError("public training transfer URL provenance is malformed") from exc
            try:
                transfer_port = transfer_parts.port
            except ValueError as exc:
                raise IntegrityError(
                    "public training transfer URL provenance is malformed"
                ) from exc
            if (
                transfer_parts.scheme != "https"
                or not transfer_parts.hostname
                or transfer_parts.hostname.casefold() not in TRAINING_ARCHIVE_HOSTS
                or transfer_port not in (None, 443)
                or not transfer_parts.path
                or transfer_parts.username is not None
                or transfer_parts.password is not None
                or transfer_parts.query
                or transfer_parts.fragment
                or contains_forbidden_path(transfer_parts.path)
                or transfer.get("http_status") != 200
            ):
                raise IntegrityError("public training transfer provenance violates its boundary")
            archive = receipt.get("archive")
            if not isinstance(archive, dict):
                raise IntegrityError("COMPLETE public training receipt has no archive identity")
            archive_sha = archive.get("sha256")
            archive_bytes = _plain_nonnegative_int(
                archive.get("bytes"), "public training archive byte count"
            )
            declared_length = transfer.get("declared_content_length")
            if declared_length is not None and (
                type(declared_length) is not int or declared_length != archive_bytes
            ):
                raise IntegrityError("public training transfer length provenance is invalid")
            content_type = transfer.get("content_type")
            if content_type is not None and not isinstance(content_type, str):
                raise IntegrityError("public training transfer content type is invalid")
            archive_relative = _normalized_relative(
                archive.get("object"), "public training archive object path"
            )
            if (
                not isinstance(archive_sha, str)
                or SHA256_RE.fullmatch(archive_sha) is None
                or archive_relative != _sha_relative(archive_sha)
                or task_row.get("archive_sha256") != archive_sha
            ):
                raise IntegrityError("public training archive CAS/task-row identity mismatch")
            actual_inventory = _verify_and_reinventory_training_zip(
                training_root, archive_relative, archive_sha, archive_bytes
            )
            if receipt.get("inventory") != actual_inventory:
                raise IntegrityError(
                    "public training receipt inventory contradicts the verified ZIP bytes"
                )
            image_count, dedup_count = _training_zip_counts(actual_inventory)
            if (
                task_row.get("image_member_count") != image_count
                or type(task_row.get("image_member_count")) is not int
                or task_row.get("post_exact_byte_dedup_image_count") != dedup_count
                or type(task_row.get("post_exact_byte_dedup_image_count")) is not int
                or receipt.get("reason") is not None
            ):
                raise IntegrityError("public training inventory row contradicts its COMPLETE receipt")
            complete_count += 1
            verified_archive_bytes += archive_bytes
            source["archive"] = {
                "object": archive_relative,
                "sha256": archive_sha,
                "bytes": archive_bytes,
            }
            results.append(
                {
                    "task_id": task_id,
                    "status": "COMPLETE",
                    "counts_available": True,
                    "reason": None,
                    "actual_train_zip_image_count": image_count,
                    "post_byte_dedup_image_count": dedup_count,
                    "rights": TRAINING_ARCHIVE_RIGHTS,
                    "sources": [source],
                }
            )
        else:
            reason = receipt.get("reason")
            if (
                not isinstance(reason, str)
                or not reason
                or receipt.get("archive") is not None
                or receipt.get("inventory") is not None
                or task_row.get("archive_sha256") is not None
                or task_row.get("image_member_count") is not None
                or task_row.get("post_exact_byte_dedup_image_count") is not None
            ):
                raise IntegrityError("PARTIAL public training receipt is not honestly partial")
            results.append(
                {
                    "task_id": task_id,
                    "status": "PARTIAL",
                    "counts_available": False,
                    "reason": reason,
                    "actual_train_zip_image_count": None,
                    "post_byte_dedup_image_count": None,
                    "rights": TRAINING_ARCHIVE_RIGHTS,
                    "sources": [source],
                }
            )
            missing.append(
                {
                    "class": "public-training-archive-counts-unavailable",
                    "task_id": task_id,
                    "reason": reason,
                }
            )

    if seen_tasks != r1_task_ids:
        raise IntegrityError("public training inventory does not contain exact R1 task identities")
    partial_count = len(task_rows) - complete_count
    if (
        _plain_nonnegative_int(
            inventory.get("complete_task_count"), "public training complete-task count"
        )
        != complete_count
        or _plain_nonnegative_int(
            inventory.get("partial_task_count"), "public training partial-task count"
        )
        != partial_count
    ):
        raise IntegrityError("public training inventory aggregate task counts are invalid")
    derived_status = "COMPLETE" if partial_count == 0 else "PARTIAL"
    if inventory.get("status") != derived_status:
        raise IntegrityError("public training inventory aggregate status is dishonest")

    provenance = {
        "root": str(training_root),
        "root_identity_sha256": sha256_bytes(root_identity_body),
        "inventory": inventory_relative,
        "inventory_sha256": inventory_sha,
        "inventory_checksum": checksum_relative,
        "inventory_checksum_sha256": sha256_bytes(checksum_body),
        "status": derived_status,
        "r1_task_count": len(task_rows),
        "complete_task_count": complete_count,
        "partial_task_count": partial_count,
        "verified_complete_archive_count": complete_count,
        "verified_complete_archive_bytes": verified_archive_bytes,
    }
    return sorted(results, key=lambda row: row["task_id"]), missing, provenance


def public_fixture_inventory(
    observations: Iterable[dict[str, Any]],
    verified_watcher_files: Mapping[str, bytes],
    r1_task_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return an honest placeholder; watcher full-pool rows are never opened.

    A separate, provenance-bound collector supplies the authoritative public
    ``training_data.zip`` inventory.  Keeping this placeholder fail-visible
    prevents an image_text_pairs count from being relabeled as optimizer input
    if that external receipt is absent.
    """
    del observations, verified_watcher_files
    rights = {
        "access": "publicly_accessible",
        "license_and_third_party_rights": "unverified",
        "allowed_use": "research-analysis-only",
        "fixture_admission": "not admitted",
    }
    results = [
        {
            "task_id": task_id,
            "counts_available": False,
            "reason": "authoritative public training_data.zip receipt was not supplied",
            "actual_train_zip_image_count": None,
            "post_byte_dedup_image_count": None,
            "rights": rights,
            "sources": [],
        }
        for task_id in sorted(r1_task_ids)
    ]
    missing = [
        {"class": "public-training-archive-counts-unavailable", "task_id": task_id}
        for task_id in sorted(r1_task_ids)
    ]
    return results, missing


def _score_row_keyed(rows: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise IntegrityError(f"{label} participant rows are absent")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("hotkey"), str):
            raise IntegrityError(f"{label} participant row is malformed")
        hotkey = row["hotkey"]
        if hotkey in result:
            raise IntegrityError(f"{label} contains a duplicate hotkey")
        for field in ("repo", "submission_id", "score_reason"):
            value = row.get(field)
            if value is not None and not isinstance(value, str):
                raise IntegrityError(f"{label} contains an invalid {field}")
        for field in ("test_loss", "synth_loss", "quality_score"):
            value = row.get(field)
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise IntegrityError(f"{label} contains an invalid {field}")
        rank = row.get("rank")
        if rank is not None and (type(rank) is not int or rank < 0):
            raise IntegrityError(f"{label} contains an invalid rank")
        submitted = row.get("repo") is not None or row.get("submission_id") is not None
        scored = any(
            row.get(field) is not None and row.get(field) != 0
            for field in ("test_loss", "synth_loss", "quality_score")
        ) or (
            isinstance(row.get("score_reason"), str)
            and bool(row["score_reason"].strip())
        )
        expected_state = "scored" if scored else "submitted" if submitted else "pending"
        if row.get("derived_state") != expected_state:
            raise IntegrityError(f"{label} contains an invalid derived_state")
        result[hotkey] = row
    return result


def build_package(
    api_snapshot: Path,
    watcher_root: Path,
    header_root: Path,
    header_inventory: Path,
    output_root: Path,
    tournament_id: str,
    *,
    observed_at: dt.datetime,
    training_archive_root: Path | None = None,
    training_archive_inventory: Path | None = None,
    training_archive_checksum: Path | None = None,
) -> dict[str, Any]:
    if TOURNAMENT_RE.fullmatch(tournament_id) is None:
        raise SyncError("invalid tournament ID")
    if (training_archive_root is None) != (training_archive_inventory is None):
        raise SyncError(
            "public training archive root and inventory must be supplied together"
        )
    if training_archive_checksum is not None and training_archive_root is None:
        raise SyncError("public training archive checksum requires its root and inventory")
    api, api_provenance = load_api_snapshot(api_snapshot, tournament_id)
    api_observed_at = _parse_utc_timestamp(
        api.get("observed_at"), "safe API observed_at"
    )
    r1 = next(
        row
        for row in api["tournament"]["rounds"]
        if type(row.get("round_number")) is int and row.get("round_number") == 1
    )
    round_tasks = r1.get("tasks")
    if not isinstance(round_tasks, list) or not round_tasks:
        raise IntegrityError("completed Round 1 contains no tasks")
    task_ids = {
        row.get("task_id") for row in round_tasks if isinstance(row, dict) and isinstance(row.get("task_id"), str)
    }
    if len(task_ids) != len(round_tasks) or any(TASK_RE.fullmatch(task) is None for task in task_ids):
        raise IntegrityError("Round-1 task identities are missing, duplicated, or invalid")
    for round_task in round_tasks:
        if not isinstance(round_task, dict) or round_task.get("task_type") != "ImageTask":
            raise IntegrityError("Round-1 contains a non-image or malformed task row")
    task_api = api.get("tasks")
    if not isinstance(task_api, list):
        raise IntegrityError("safe API snapshot contains no task details")
    detail_by_task: dict[str, dict[str, Any]] = {}
    for detail in task_api:
        if isinstance(detail, dict) and detail.get("task_id") in task_ids:
            if detail["task_id"] in detail_by_task:
                raise IntegrityError("safe API snapshot duplicates a Round-1 task")
            detail_by_task[detail["task_id"]] = detail
    if set(detail_by_task) != task_ids:
        raise IntegrityError("safe API snapshot lacks exact details for every Round-1 task")

    observations, verified_watcher_files, watcher_provenance = verify_watcher_root(
        watcher_root,
        tournament_id,
        task_ids,
        api_observed_at=api_observed_at,
    )
    header_rows, header_provenance = load_header_inventory(
        header_root, header_inventory, tournament_id, task_ids, watcher_root
    )
    public_repos, missing = _manifest_records(
        observations,
        verified_watcher_files,
        tournament_id,
        task_ids,
        header_rows,
    )
    training_archive_provenance: dict[str, Any] | None = None
    if training_archive_root is None:
        fixture_inventory, fixture_missing = public_fixture_inventory(
            observations, verified_watcher_files, task_ids
        )
    else:
        assert training_archive_inventory is not None
        fixture_inventory, fixture_missing, training_archive_provenance = (
            load_public_training_archive_inventory(
                training_archive_root,
                training_archive_inventory,
                tournament_id,
                task_ids,
                api_snapshot_path=api_snapshot,
                api_snapshot_sha256=api_provenance["sha256"],
                api_observed_at=api_observed_at,
                checksum_path=training_archive_checksum,
            )
        )
    missing.extend(fixture_missing)

    tasks: list[dict[str, Any]] = []
    all_round_participants = r1.get("participants")
    if not isinstance(all_round_participants, list) or not all(
        isinstance(item, str) and item for item in all_round_participants
    ):
        raise IntegrityError("Round-1 participant list is invalid")
    if len(all_round_participants) != len(set(all_round_participants)):
        raise IntegrityError("Round-1 participant list contains duplicates")
    for round_task in sorted(round_tasks, key=lambda row: row["task_id"]):
        task_id = round_task["task_id"]
        detail = detail_by_task[task_id]
        if detail.get("task_type") != "ImageTask":
            raise IntegrityError("task detail is not an image task")
        if detail.get("task_type") != round_task.get("task_type"):
            raise IntegrityError("Round-1 task type conflicts with task detail")
        if str(detail.get("status", "")).lower() not in TERMINAL_TASK_STATUSES:
            missing.append(
                {
                    "class": "task-api-not-terminal",
                    "task_id": task_id,
                    "status": detail.get("status"),
                }
            )
        detail_rows = _score_row_keyed(detail.get("participants"), "task API")
        round_rows = _score_row_keyed(round_task.get("participant_scores", []), "tournament round")
        participant_hotkeys = sorted(set(all_round_participants) | set(detail_rows) | set(round_rows))
        participants: list[dict[str, Any]] = []
        for hotkey in participant_hotkeys:
            detail_row = detail_rows.get(hotkey)
            round_row = round_rows.get(hotkey)
            conflict_fields: list[str] = []
            for field in (
                "repo",
                "submission_id",
                "test_loss",
                "synth_loss",
                "quality_score",
                "rank",
                "score_reason",
                "derived_state",
            ):
                left = detail_row.get(field) if detail_row else None
                right = round_row.get(field) if round_row else None
                if left is not None and right is not None and left != right:
                    conflict_fields.append(field)
            if conflict_fields:
                missing.append(
                    {
                        "class": "public-api-surface-conflict",
                        "task_id": task_id,
                        "hotkey": hotkey,
                        "fields": conflict_fields,
                    }
                )
            repo_value = next(
                (
                    row.get("repo")
                    for row in (detail_row, round_row)
                    if row is not None and row.get("repo") is not None
                ),
                None,
            )
            repo = _repo_from_api(repo_value, tournament_id, task_id, hotkey)
            repo_evidence = public_repos.get(repo) if repo else None
            if repo is not None and repo_evidence is None:
                missing.append(
                    {
                        "class": "submitted-repository-not-captured",
                        "task_id": task_id,
                        "hotkey": hotkey,
                        "repository": repo,
                    }
                )
            if hotkey not in detail_rows:
                missing.append(
                    {"class": "round-participant-missing-from-task-api", "task_id": task_id, "hotkey": hotkey}
                )
            state = "no-public-submission"
            for candidate in (detail_row, round_row):
                candidate_state = candidate.get("derived_state") if candidate else None
                if candidate_state == "scored":
                    state = "scored"
                    break
                if candidate_state == "submitted":
                    state = "submitted"
            participants.append(
                {
                    "hotkey": hotkey,
                    "derived_state": state,
                    "task_api": detail_row,
                    "tournament_round": round_row,
                    "repository": repo,
                    "repository_evidence_available": repo_evidence is not None,
                }
            )
        tasks.append(
            {
                "task_id": task_id,
                "task_type": detail.get("task_type"),
                "model_id": detail.get("model_id"),
                "model_type": detail.get("model_type"),
                "hours_to_complete": detail.get("hours_to_complete"),
                "task_status_at_snapshot": detail.get("status"),
                "round_winner": round_task.get("winner"),
                "public_training_archive_present": detail.get("public_training_archive_present"),
                "participants": participants,
            }
        )

    api_repos = {
        row["repository"]
        for task in tasks
        for row in task["participants"]
        if row["repository"] is not None
    }
    unlinked_public_attempts = [
        {
            "task_id": evidence["task_id"],
            "repository": repo,
            "classification": "public-retry-or-unlinked-attempt",
            "latest_observed_head": evidence["latest_observed_head"],
        }
        for repo, evidence in sorted(public_repos.items())
        if repo not in api_repos
    ]

    # Stable ordering and de-duplication make a repeated run auditable.
    missing = sorted(
        {canonical_json(row): row for row in missing}.values(),
        key=lambda row: canonical_json(row),
    )
    package = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(observed_at),
        "status": "COMPLETE" if not missing else "PARTIAL",
        "tournament_id": tournament_id,
        "tournament_type": "image",
        "round": {
            "round_number": 1,
            "round_id": r1.get("round_id"),
            "round_type": r1.get("round_type"),
            "status": "completed",
            "participant_count": len(all_round_participants),
            "task_count": len(tasks),
            "tasks": tasks,
        },
        "public_repositories": [
            {
                **public_repos[key],
                "api_linkage": "linked-final-public-submission"
                if key in api_repos
                else "public-retry-or-unlinked-attempt",
            }
            for key in sorted(public_repos)
        ],
        "public_unlinked_or_retry_attempts": unlinked_public_attempts,
        "public_training_archive_inventory": fixture_inventory,
        "completeness": {
            "missing_or_partial_surfaces": missing,
            "missing_or_partial_count": len(missing),
            "prohibited_dataset_asset_urls_dereferenced": 0,
            "prohibited_dataset_rows_persisted_or_projected": 0,
            "task_api_envelope_received_before_allowlist": training_archive_provenance
            is not None,
            "raw_weight_bodies_downloaded_or_read": 0,
        },
        "inputs": {
            "safe_public_api": api_provenance,
            "raw_watcher": watcher_provenance,
            "public_safetensors_headers": header_provenance,
            "public_training_archives": training_archive_provenance,
        },
        "interpretation_limits": [
            "test_loss and rank are public API observations, not independently recomputed scores",
            "LFS SHA equality proves public byte identity only",
            "header training step is an artifact metadata observation, not proof of the configured plan",
            "highest observed numbered checkpoint is not labeled the training final",
            "config projection is a narrow allowlist and not a claim of full effective-runtime equivalence",
            "watcher image_text_pairs/full-pool fixture rows are excluded unopened",
            "training archives, when supplied, are public optimizer-visible archives with unverified rights and are not fixture-admitted",
            "the public task API envelope is received before allowlisting; prohibited dataset fields are not selected, persisted, projected, or dereferenced",
            "no test, hidden, holdout, quarantine, or evaluation-derived row assets are present",
        ],
    }
    body = canonical_json(package)
    publisher = Publisher(output_root)
    stamp = utc_token(observed_at)
    relative = f"packages/{stamp}.json"
    staged = publisher.stage_bytes(body)
    try:
        publisher.publish(staged, relative)
    finally:
        staged.discard()
    checksum = f"{sha256_bytes(body)}  {stamp}.json\n".encode("ascii")
    checksum_staged = publisher.stage_bytes(checksum)
    try:
        publisher.publish(checksum_staged, f"packages/{stamp}.sha256")
    finally:
        checksum_staged.discard()
    return {
        "status": package["status"],
        "package": relative,
        "package_sha256": sha256_bytes(body),
        "r1_task_count": len(tasks),
        "public_repository_count": len(public_repos),
        "missing_or_partial_count": len(missing),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-snapshot", required=True, type=Path)
    parser.add_argument("--watcher-root", required=True, type=Path)
    parser.add_argument("--header-root", required=True, type=Path)
    parser.add_argument("--header-inventory", required=True, type=Path)
    parser.add_argument("--training-archive-root", type=Path)
    parser.add_argument("--training-archive-inventory", type=Path)
    parser.add_argument("--training-archive-checksum", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tournament-id", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_package(
            args.api_snapshot,
            args.watcher_root,
            args.header_root,
            args.header_inventory,
            args.output_root,
            args.tournament_id,
            observed_at=utc_now(),
            training_archive_root=args.training_archive_root,
            training_archive_inventory=args.training_archive_inventory,
            training_archive_checksum=args.training_archive_checksum,
        )
    except (OSError, ValueError, SyncError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
