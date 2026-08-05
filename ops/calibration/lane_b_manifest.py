#!/usr/bin/env python3
"""Compile immutable Lane-B generation evidence into source manifests.

The generator's plan describes intent; its hard-linked SUCCESS receipts bind
that intent to provider requests; the image and caption files are the bytes
that will actually enter a fixture.  This compiler requires all three to
agree, validates the terminal generation summary, and emits only canonical,
create-only source manifests accepted by ``lane_b_fixture_package.py``.

An approved rights status is an explicit invocation input.  It is never
inferred from a URL or copied from an editable side file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping
import unicodedata
from urllib.parse import urlsplit
import uuid

from PIL import Image, UnidentifiedImageError

import lane_b_fixture_package as fixture_package
import lane_b_generate as generator


_SUMMARY_KEYS = {
    "schema",
    "kind",
    "status",
    "purpose",
    "social_relief",
    "plan_file_sha256",
    "row_count",
    "success_count",
    "success_receipt_sha256",
}
_RECEIPT_KEYS = {
    "schema",
    "kind",
    "state",
    "origin",
    "fixture_id",
    "row_id",
    "attempt",
    "client_request_id",
    "provider_request_id",
    "request_id",
    "provider",
    "model",
    "endpoint",
    "request_started_at",
    "request_finished_at",
    "plan_file_sha256",
    "row_plan_sha256",
    "request_payload_sha256",
    "prompt",
    "prompt_sha256",
    "caption_sha256",
    "rights_reference",
    "rights_status",
    "reference_sha256",
    "response_mime_type",
    "response_image_sha256",
    "output_path",
    "output_image_sha256",
    "output_image_bytes",
    "caption_path",
    "width",
    "height",
    "http_status",
    "error",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RECEIPT_TIME = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z"
)
_RECEIPT_KIND = "sn56-week6-lane-b-generation-attempt"
_SUMMARY_KIND = "sn56-week6-lane-b-generation-summary"
_SOURCE_KIND = "forge-week6-lane-b-source-manifest"
_REFERENCE_PROVENANCE_KIND = "sn56-week6-lane-b-reference-provenance"
_RIGHTS_STATUSES = frozenset(fixture_package._RIGHTS_STATUSES)


class ManifestCompileError(RuntimeError):
    """A fail-closed generation-evidence or publication error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ManifestCompileError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestCompileError(f"{label} must be a JSON object")
    missing = keys - set(value)
    extra = set(value) - keys
    if missing or extra:
        raise ManifestCompileError(
            f"{label} schema mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    return value


def _safe_directory(path: Path | str, label: str) -> Path:
    absolute = Path(os.path.abspath(os.path.expanduser(path)))
    current = absolute
    while current != current.parent:
        if current.is_symlink():
            raise ManifestCompileError(f"{label} has a symlink component: {current}")
        current = current.parent
    if not absolute.is_dir() or absolute.is_symlink():
        raise ManifestCompileError(f"{label} must be a real directory: {absolute}")
    return absolute


def _read_regular(path: Path | str, label: str) -> tuple[bytes, os.stat_result]:
    absolute = Path(os.path.abspath(os.path.expanduser(path)))
    current = absolute.parent
    while current != current.parent:
        if current.is_symlink():
            raise ManifestCompileError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ManifestCompileError(f"cannot safely open {label}: {absolute}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ManifestCompileError(f"{label} is not a regular file: {absolute}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise ManifestCompileError(f"{label} changed while read: {absolute}")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise ManifestCompileError(f"{label} changed size while read: {absolute}")
    return raw, before


def _read_json(path: Path | str, label: str) -> tuple[Any, bytes, os.stat_result]:
    raw, metadata = _read_regular(path, label)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ManifestCompileError(f"{label} contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestCompileError(f"{label} is not valid UTF-8 JSON") from exc
    return value, raw, metadata


def _approved_rights_reference(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or value != unicodedata.normalize("NFC", value)
    ):
        raise ManifestCompileError(f"{label} must be non-empty canonical text")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username is not None:
        raise ManifestCompileError(f"{label} must be an absolute HTTPS URL")
    return value


def _receipt_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or _RECEIPT_TIME.fullmatch(value) is None:
        raise ManifestCompileError(f"{label} is not canonical microsecond UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ManifestCompileError(f"{label} is not real UTC") from exc
    return parsed


def _whole_second(value: datetime) -> str:
    return value.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _manifest_text(value: Any, label: str, *, exact_whitespace: bool) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestCompileError(f"{label} must be non-empty text")
    if value != unicodedata.normalize("NFC", value):
        raise ManifestCompileError(f"{label} must use NFC Unicode")
    if not exact_whitespace and value != " ".join(value.split()):
        raise ManifestCompileError(f"{label} must use canonical whitespace")
    return value


def _decoded_dimensions(raw: bytes, label: str) -> tuple[int, int]:
    try:
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > 100_000_000:
                raise ManifestCompileError(f"{label} has unsafe dimensions")
            image.verify()
    except ManifestCompileError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ManifestCompileError(f"{label} is not a loadable image") from exc
    return width, height


def _canonical_ratio(width: int, height: int) -> str:
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def _request_body(row: Mapping[str, Any]) -> tuple[str, bytes, str | None]:
    if row["provider"] == "openai":
        endpoint, _, body = generator._openai_request(row, "unused-local-key")
        return endpoint, body, None
    endpoint, _, body, reference_sha = generator._gemini_request(
        row, "unused-local-key"
    )
    return endpoint, body, reference_sha


def _generator_label(rows: list[dict[str, Any]], fixture_id: str) -> str:
    facts = {(row["provider"], row["reference_sha256"] is not None) for row in rows}
    labels = {
        frozenset({("openai", False)}): "GPT Image 2",
        frozenset({("gemini", False)}): "Nano Banana 2",
        frozenset({("gemini", True)}): (
            "GPT Image 2 reference → Nano Banana 2 edits"
        ),
    }
    label = labels.get(frozenset(facts))
    expected = fixture_package._FIXTURES[fixture_id]["generator_labels"]
    if label is None or label not in expected:
        raise ManifestCompileError(
            f"{fixture_id} provider/reference facts do not derive an approved "
            "generator label"
        )
    return label


def _validate_receipt(
    *,
    fixture_id: str,
    row: Mapping[str, Any],
    plan_file_sha: str,
    generation_root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    row_id = row["id"]
    fixture_root = generation_root / fixture_id
    success_path = fixture_root / f"{row_id}.success.json"
    value, _, success_stat = _read_json(
        success_path, f"SUCCESS receipt {fixture_id}/{row_id}"
    )
    receipt = _exact_object(
        value, _RECEIPT_KEYS, f"SUCCESS receipt {fixture_id}/{row_id}"
    )
    if (
        receipt["schema"] != 1
        or receipt["kind"] != _RECEIPT_KIND
        or receipt["state"] != "SUCCESS"
    ):
        raise ManifestCompileError(f"{fixture_id}/{row_id} is not a v1 SUCCESS")
    if receipt["origin"] != "real":
        raise ManifestCompileError(
            f"{fixture_id}/{row_id} receipt origin is not real provider evidence"
        )
    if receipt["rights_status"] != "approved-provider-terms":
        raise ManifestCompileError(
            f"{fixture_id}/{row_id} receipt rights status is not approved-provider-terms"
        )
    attempt = receipt["attempt"]
    if (
        isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or not 1 <= attempt <= 9
    ):
        raise ManifestCompileError(f"{fixture_id}/{row_id} attempt is invalid")
    attempt_path = fixture_root / "attempts" / f"{row_id}.attempt-{attempt:03d}.json"
    attempt_value, _, attempt_stat = _read_json(
        attempt_path, f"attempt receipt {fixture_id}/{row_id}"
    )
    if attempt_value != receipt or (
        success_stat.st_dev,
        success_stat.st_ino,
    ) != (attempt_stat.st_dev, attempt_stat.st_ino):
        raise ManifestCompileError(
            f"{fixture_id}/{row_id} SUCCESS is not the immutable attempt hard link"
        )

    row_plan_sha = _sha256(
        _canonical_bytes({"fixture_id": fixture_id, "row": row})
    )
    equality = {
        "fixture_id": fixture_id,
        "row_id": row_id,
        "provider": row["provider"],
        "model": row["model"],
        "prompt": row["prompt"],
        "rights_reference": row["rights_reference"],
        "plan_file_sha256": plan_file_sha,
        "row_plan_sha256": row_plan_sha,
    }
    for field, expected in equality.items():
        if receipt[field] != expected:
            raise ManifestCompileError(
                f"{fixture_id}/{row_id} receipt {field} does not match the plan"
            )

    endpoint, request_body, actual_reference_sha = _request_body(row)
    if receipt["endpoint"] != endpoint:
        raise ManifestCompileError(f"{fixture_id}/{row_id} endpoint mismatch")
    if receipt["request_payload_sha256"] != _sha256(request_body):
        raise ManifestCompileError(f"{fixture_id}/{row_id} request payload mismatch")
    if receipt["reference_sha256"] != actual_reference_sha:
        raise ManifestCompileError(f"{fixture_id}/{row_id} reference hash mismatch")

    expected_image = generation_root / fixture_id / "images" / f"{row_id}.png"
    expected_caption = generation_root / fixture_id / "captions" / f"{row_id}.txt"
    if receipt["output_path"] != str(expected_image):
        raise ManifestCompileError(
            f"{fixture_id}/{row_id} output path is not canonical"
        )
    if receipt["caption_path"] != str(expected_caption):
        raise ManifestCompileError(
            f"{fixture_id}/{row_id} caption path is not canonical"
        )
    image, _ = _read_regular(expected_image, f"output image {fixture_id}/{row_id}")
    caption, _ = _read_regular(expected_caption, f"caption {fixture_id}/{row_id}")
    image_sha = _sha256(image)
    caption_sha = _sha256(caption)
    width, height = _decoded_dimensions(image, f"output image {fixture_id}/{row_id}")
    if caption != row["caption"].encode("utf-8"):
        raise ManifestCompileError(f"{fixture_id}/{row_id} caption bytes mismatch")
    try:
        caption_text = caption.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestCompileError(
            f"{fixture_id}/{row_id} caption is not UTF-8"
        ) from exc
    _manifest_text(
        caption_text, f"{fixture_id}/{row_id}.caption", exact_whitespace=True
    )
    _manifest_text(
        receipt["prompt"], f"{fixture_id}/{row_id}.prompt", exact_whitespace=True
    )
    derived = {
        "output_image_sha256": image_sha,
        "caption_sha256": caption_sha,
        "output_image_bytes": len(image),
        "width": width,
        "height": height,
        "prompt_sha256": _sha256(row["prompt"].encode("utf-8")),
    }
    for field, expected in derived.items():
        if receipt[field] != expected:
            raise ManifestCompileError(
                f"{fixture_id}/{row_id} receipt {field} mismatches actual bytes"
            )
    for field in ("response_image_sha256",):
        _digest(receipt[field], f"{fixture_id}/{row_id}.{field}")
    if receipt["response_mime_type"] not in {
        "image/png",
        "image/jpeg",
        "image/webp",
    }:
        raise ManifestCompileError(f"{fixture_id}/{row_id} response MIME is invalid")
    if (
        isinstance(receipt["http_status"], bool)
        or not isinstance(receipt["http_status"], int)
        or not 200 <= receipt["http_status"] < 300
        or receipt["error"] is not None
    ):
        raise ManifestCompileError(f"{fixture_id}/{row_id} SUCCESS status is invalid")
    try:
        client_id = str(uuid.UUID(receipt["client_request_id"]))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ManifestCompileError(
            f"{fixture_id}/{row_id} client request ID is invalid"
        ) from exc
    if client_id != receipt["client_request_id"]:
        raise ManifestCompileError(
            f"{fixture_id}/{row_id} client request ID is not canonical"
        )
    provider_id = receipt["provider_request_id"]
    if provider_id is not None and (
        not isinstance(provider_id, str) or not provider_id.strip()
    ):
        raise ManifestCompileError(
            f"{fixture_id}/{row_id} provider request ID is invalid"
        )
    if receipt["request_id"] != (provider_id or client_id):
        raise ManifestCompileError(f"{fixture_id}/{row_id} request ID mismatch")
    _manifest_text(
        receipt["request_id"],
        f"{fixture_id}/{row_id}.request_id",
        exact_whitespace=False,
    )
    started = _receipt_time(
        receipt["request_started_at"], f"{fixture_id}/{row_id}.request_started_at"
    )
    finished = _receipt_time(
        receipt["request_finished_at"], f"{fixture_id}/{row_id}.request_finished_at"
    )
    if finished < started:
        raise ManifestCompileError(
            f"{fixture_id}/{row_id} request timeline is reversed"
        )
    rights_reference = _approved_rights_reference(
        receipt["rights_reference"], f"{fixture_id}/{row_id}.rights_reference"
    )

    source_row = {
        "row_id": row_id,
        "image_path": expected_image.relative_to(generation_root).as_posix(),
        "caption_path": expected_caption.relative_to(generation_root).as_posix(),
        "provider": receipt["provider"],
        "model": receipt["model"],
        "request_time_utc": _whole_second(finished),
        "request_id": receipt["request_id"],
        "prompt": receipt["prompt"],
        "output_sha256": image_sha,
        "rights_reference": rights_reference,
        "rights_status": None,
        "reference_sha256": actual_reference_sha,
        "width": width,
        "height": height,
        "aspect_ratio": _canonical_ratio(width, height),
        "image_sha256": image_sha,
        "caption_sha256": caption_sha,
    }
    receipt_canonical_sha = _sha256(_canonical_bytes(receipt))
    return source_row, receipt_canonical_sha, receipt


def _validate_summary(
    summary_path: Path,
    *,
    plan_file_sha: str,
    purpose: str,
    social_relief: bool,
    receipt_hashes: list[str],
) -> str:
    value, raw, _ = _read_json(summary_path, "generation summary")
    summary = _exact_object(value, _SUMMARY_KEYS, "generation summary")
    expected = {
        "schema": 1,
        "kind": _SUMMARY_KIND,
        "status": "complete",
        "purpose": purpose,
        "social_relief": social_relief,
        "plan_file_sha256": plan_file_sha,
        "row_count": len(receipt_hashes),
        "success_count": len(receipt_hashes),
        "success_receipt_sha256": sorted(receipt_hashes),
    }
    if summary != expected:
        raise ManifestCompileError(
            "generation summary does not exactly bind the current plan and receipts"
        )
    return _sha256(raw)


def _validate_reference_evidence(
    plan_path: Path | str,
    generation_root: Path | str,
    *,
    rights_status: str,
) -> dict[str, dict[str, Any]]:
    """Return product-reference provenance only after validating its full chain."""

    if rights_status != "approved-provider-terms":
        raise ManifestCompileError(
            "reference assets require approved-provider-terms rights status"
        )
    plan_raw, _ = _read_regular(plan_path, "reference generation plan")
    plan_file_sha = _sha256(plan_raw)
    try:
        plan, generator_plan_sha = generator.load_plan(Path(plan_path))
    except (generator.GenerationError, OSError) as exc:
        raise ManifestCompileError(
            "reference generation plan failed generator validation"
        ) from exc
    if generator_plan_sha != plan_file_sha:
        raise ManifestCompileError("reference generation plan changed across validation")
    if plan["purpose"] != "reference-assets" or plan["social_relief"] is not False:
        raise ManifestCompileError("reference plan must be an unrelieved reference-assets plan")
    expected_fixtures = {"W6-PRODUCT-A", "W6-PRODUCT-B"}
    if {item["id"] for item in plan["fixtures"]} != expected_fixtures:
        raise ManifestCompileError("reference plan must contain exactly both product fixtures")

    root = _safe_directory(generation_root, "reference generation root")
    receipt_hashes: list[str] = []
    result: dict[str, dict[str, Any]] = {}
    for fixture in plan["fixtures"]:
        fixture_id = fixture["id"]
        if len(fixture["rows"]) != 1:
            raise ManifestCompileError(
                f"{fixture_id} reference plan must contain exactly one canonical row"
            )
        planned_row = fixture["rows"][0]
        source_row, receipt_sha, receipt = _validate_receipt(
            fixture_id=fixture_id,
            row=planned_row,
            plan_file_sha=plan_file_sha,
            generation_root=root,
        )
        receipt_hashes.append(receipt_sha)
        output_path = root / source_row["image_path"]
        reference_rights = _approved_rights_reference(
            receipt["rights_reference"],
            f"{fixture_id}.reference_rights_reference",
        )
        if reference_rights != generator._PROVIDER_RIGHTS["openai"]:
            raise ManifestCompileError(
                f"{fixture_id} reference receipt does not cite the pinned OpenAI terms"
            )
        reference_rights_status = receipt["rights_status"]
        if reference_rights_status != rights_status:
            raise ManifestCompileError(
                f"{fixture_id} reference receipt rights status does not match approval"
            )
        result[fixture_id] = {
            "schema": 1,
            "kind": _REFERENCE_PROVENANCE_KIND,
            "fixture_id": fixture_id,
            "row_id": planned_row["id"],
            "provider": receipt["provider"],
            "model": receipt["model"],
            "reference_plan_file_sha256": plan_file_sha,
            "reference_generation_summary_sha256": None,
            "reference_success_receipt_sha256": receipt_sha,
            "reference_request_payload_sha256": _digest(
                receipt["request_payload_sha256"],
                f"{fixture_id}.reference_request_payload_sha256",
            ),
            "reference_output_path": str(output_path),
            "reference_output_sha256": source_row["output_sha256"],
            "reference_rights_reference": reference_rights,
            "reference_rights_status": reference_rights_status,
        }
    summary_sha = _validate_summary(
        root / "generation-summary.json",
        plan_file_sha=plan_file_sha,
        purpose=plan["purpose"],
        social_relief=False,
        receipt_hashes=receipt_hashes,
    )
    for provenance in result.values():
        provenance["reference_generation_summary_sha256"] = summary_sha
    return result


def _publish_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    succeeded = False
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset : offset + 1024 * 1024])
        os.fsync(descriptor)
        succeeded = True
    finally:
        os.close(descriptor)
        if not succeeded:
            path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def compile_manifests(
    plan_path: Path | str,
    generation_root: Path | str,
    manifest_dir: Path | str,
    *,
    reference_plan_path: Path | str,
    reference_generation_root: Path | str,
    rights_status: str,
    social_relief: bool = False,
) -> list[Path]:
    """Validate one completed materialization and publish its source manifests."""

    if rights_status not in _RIGHTS_STATUSES:
        raise ManifestCompileError("rights_status is not an explicitly approved status")
    if rights_status != "approved-provider-terms":
        raise ManifestCompileError(
            "API-generated Lane-B fixtures require approved-provider-terms"
        )
    plan_raw, _ = _read_regular(plan_path, "generation plan")
    plan_file_sha = _sha256(plan_raw)
    try:
        plan, generator_plan_sha = generator.load_plan(Path(plan_path))
    except (generator.GenerationError, OSError) as exc:
        raise ManifestCompileError(
            "generation plan failed generator validation"
        ) from exc
    if generator_plan_sha != plan_file_sha:
        raise ManifestCompileError("generation plan changed across validation")
    if plan["purpose"] != "materialization":
        raise ManifestCompileError("only a materialization plan may produce manifests")
    if plan["social_relief"] is not social_relief:
        raise ManifestCompileError(
            "social relief must be explicitly selected and match the generation plan"
        )
    root = _safe_directory(generation_root, "generation root")
    reference_provenance = _validate_reference_evidence(
        reference_plan_path,
        reference_generation_root,
        rights_status=rights_status,
    )
    rows_by_fixture: dict[str, list[dict[str, Any]]] = {}
    receipt_hashes: list[str] = []
    seen_requests: set[tuple[str, str]] = set()
    for fixture in plan["fixtures"]:
        rows: list[dict[str, Any]] = []
        for planned_row in fixture["rows"]:
            row, receipt_sha, _ = _validate_receipt(
                fixture_id=fixture["id"],
                row=planned_row,
                plan_file_sha=plan_file_sha,
                generation_root=root,
            )
            request_identity = (row["provider"], row["request_id"])
            if request_identity in seen_requests:
                raise ManifestCompileError(
                    "duplicate provider/request ID across SUCCESS receipts"
                )
            seen_requests.add(request_identity)
            row["rights_status"] = rights_status
            rows.append(row)
            receipt_hashes.append(receipt_sha)
        rows_by_fixture[fixture["id"]] = rows
    _validate_summary(
        root / "generation-summary.json",
        plan_file_sha=plan_file_sha,
        purpose=plan["purpose"],
        social_relief=social_relief,
        receipt_hashes=receipt_hashes,
    )

    manifests: list[tuple[Path, dict[str, Any]]] = []
    destination = Path(os.path.abspath(os.path.expanduser(manifest_dir)))
    if destination.exists() or destination.is_symlink():
        raise ManifestCompileError(f"manifest directory already exists: {destination}")
    fixtures = fixture_package._FIXTURES
    for fixture_id in sorted(rows_by_fixture):
        facts = fixtures[fixture_id]
        rows = rows_by_fixture[fixture_id]
        provenance = reference_provenance.get(fixture_id)
        if fixture_id.startswith("W6-PRODUCT"):
            if provenance is None:
                raise ManifestCompileError(
                    f"{fixture_id} has no validated reference provenance"
                )
            planned_references = {
                (
                    row["reference"]["path"],
                    row["reference"]["sha256"],
                )
                for fixture in plan["fixtures"]
                if fixture["id"] == fixture_id
                for row in fixture["rows"]
            }
            expected_reference = {
                (
                    provenance["reference_output_path"],
                    provenance["reference_output_sha256"],
                )
            }
            if planned_references != expected_reference:
                raise ManifestCompileError(
                    f"{fixture_id} reference is not the validated reference-assets output"
                )
        published_provenance = (
            {
                key: value
                for key, value in provenance.items()
                if key != "reference_output_path"
            }
            if provenance is not None
            else None
        )
        manifests.append(
            (
                destination / f"{fixture_id}.json",
                {
                    "schema": 1,
                    "kind": _SOURCE_KIND,
                    "fixture_id": fixture_id,
                    "family": facts["family"],
                    "subtype": facts["subtype"],
                    "generator_label": _generator_label(rows, fixture_id),
                    "reference_provenance": published_provenance,
                    "total_pairs": len(rows),
                    "rows": rows,
                },
            )
        )
    _safe_directory(destination.parent, "manifest parent")
    try:
        os.mkdir(destination, mode=0o700)
    except OSError as exc:
        raise ManifestCompileError(
            f"cannot create manifest directory: {destination}"
        ) from exc
    published: list[Path] = []
    try:
        for path, value in manifests:
            _publish_exclusive(path, value)
            published.append(path)
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        destination.rmdir()
        raise
    return published


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--generation-root", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--reference-plan", required=True)
    parser.add_argument("--reference-generation-root", required=True)
    parser.add_argument(
        "--rights-status",
        required=True,
        choices=("approved-provider-terms",),
    )
    parser.add_argument("--social-relief", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    manifests = compile_manifests(
        args.plan,
        args.generation_root,
        args.manifest_dir,
        reference_plan_path=args.reference_plan,
        reference_generation_root=args.reference_generation_root,
        rights_status=args.rights_status,
        social_relief=args.social_relief,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest_count": len(manifests),
                "manifests": [str(path) for path in manifests],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
