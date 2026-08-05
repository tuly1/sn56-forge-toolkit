#!/usr/bin/env python3
"""Generate hash-bound Week-6 Lane-B fixture images.

This utility is deliberately separate from the trainer and release path.  It
supports the two source-generator families declared by the Lane-B protocol,
writes one immutable receipt per network attempt, and can resume only from a
previous SUCCESS whose plan, image, and caption bytes still match.

Credentials are read only from ``OPENAI_API_KEY`` or ``GEMINI_API_KEY`` in the
process environment.  They are never accepted on the command line or written
to evidence.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Any, Callable, Mapping
import urllib.error
from urllib.parse import urlsplit
import urllib.request
import uuid

from PIL import Image

import lane_b_plan as planner


_OPENAI_MODEL = "gpt-image-2-2026-04-21"
_GEMINI_MODEL = "gemini-3.1-flash-image"
_OPENAI_ENDPOINT = "https://api.openai.com/v1/images/generations"
_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1/models/"
    f"{_GEMINI_MODEL}:generateContent"
)
_PROVIDER_RIGHTS = {
    "openai": "https://openai.com/policies/services-agreement/",
    "gemini": "https://ai.google.dev/gemini-api/terms",
}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ALLOWED_FIXTURES = {
    "W6-DESIGN-GPT-A": 21,
    "W6-DESIGN-NB-B": 20,
    "W6-PRODUCT-A": 21,
    "W6-PRODUCT-B": 20,
    "W6-SOCIAL-A": 18,
    "W6-DESIGN-GPT-LARGE": 48,
}
_MATERIALIZATION_PROVIDERS = {
    "W6-DESIGN-GPT-A": "openai",
    "W6-DESIGN-NB-B": "gemini",
    "W6-PRODUCT-A": "gemini",
    "W6-PRODUCT-B": "gemini",
    "W6-SOCIAL-A": "gemini",
    "W6-DESIGN-GPT-LARGE": "openai",
}
_OPENAI_SIZES = {"1024x1024", "1536x1024", "1024x1536"}
_GEMINI_ASPECTS = {
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
}
_SUCCESS_RECEIPT_KEYS = {
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


class GenerationError(RuntimeError):
    """A fail-closed plan, transport, or evidence error."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


Transport = Callable[[str, Mapping[str, str], bytes, float], HttpResponse]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _reject_symlink_components(path: Path, label: str, *, include_final: bool) -> None:
    current = path if include_final else path.parent
    while current != current.parent:
        if current.is_symlink():
            raise GenerationError(f"{label} has a symlink component: {current}")
        current = current.parent


def _stable_regular(path: Path, label: str) -> bytes:
    _reject_symlink_components(path, label, include_final=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GenerationError(f"cannot open {label} as a regular file: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise GenerationError(f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in identity):
            raise GenerationError(f"{label} changed while read: {path}")
        data = b"".join(chunks)
        if len(data) != before.st_size:
            raise GenerationError(f"{label} size changed while read: {path}")
        return data
    finally:
        os.close(descriptor)


def _read_json(path: Path, label: str) -> tuple[Any, str]:
    data = _stable_regular(path, label)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise GenerationError(f"{label} contains duplicate key: {key}")
            value[key] = item
        return value

    try:
        return (
            json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicates),
            _sha256_bytes(data),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GenerationError(f"{label} is not valid UTF-8 JSON: {path}") from exc


def _publish_exclusive(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace evidence: {path}")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
    succeeded = False
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset : offset + 1024 * 1024])
        os.fsync(descriptor)
        succeeded = True
    finally:
        os.close(descriptor)
    if succeeded:
        _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise GenerationError(f"cannot fsync non-directory: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_json(path: Path, value: Any) -> None:
    _publish_exclusive(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _ensure_directory(path: Path) -> None:
    _reject_symlink_components(path, "output directory", include_final=True)
    parent = path.parent
    if path != parent:
        if not parent.exists() or parent.is_symlink() or not parent.is_dir():
            raise GenerationError(f"output parent is not a real directory: {parent}")
    if path.exists():
        mode = path.lstat().st_mode
        if not stat.S_ISDIR(mode) or path.is_symlink():
            raise GenerationError(f"output component is not a real directory: {path}")
        return
    created = False
    try:
        path.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        # A concurrent worker may have created the shared fixed-name directory.
        # Accept only the same safe type after the race; row data is serialized
        # separately by the per-row flock below.
        pass
    if path.is_symlink() or not path.is_dir():
        raise GenerationError(f"created output is not a real directory: {path}")
    if created:
        _fsync_directory(parent)


def _validate_text(value: Any, label: str, *, maximum: int = 32_000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GenerationError(f"{label} must be non-empty UTF-8 text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise GenerationError(f"{label} must be non-empty UTF-8 text") from exc
    if len(encoded) > maximum:
        raise GenerationError(f"{label} must be non-empty UTF-8 text")
    return value


def _validate_rights_reference(value: Any) -> str:
    value = _validate_text(value, "rights_reference", maximum=4_000)
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username is not None:
        raise GenerationError("rights_reference must be an absolute HTTPS URL")
    return value


def _normalize_row(row: Any, *, fixture_id: str, purpose: str) -> dict[str, Any]:
    required = {
        "id",
        "provider",
        "model",
        "prompt",
        "caption",
        "rights_reference",
        "aspect_ratio",
        "quality",
        "reference",
    }
    if not isinstance(row, dict) or set(row) != required:
        raise GenerationError(f"row schema mismatch in {fixture_id}")
    row_id = row["id"]
    if not isinstance(row_id, str) or not _SAFE_ID.fullmatch(row_id):
        raise GenerationError(f"unsafe row id in {fixture_id}")
    provider = row["provider"]
    model = row["model"]
    if provider == "openai":
        if model != _OPENAI_MODEL:
            raise GenerationError("OpenAI rows must use the pinned GPT Image 2 snapshot")
        if row["reference"] is not None:
            raise GenerationError("OpenAI Lane-B rows are generation-only")
        if row["aspect_ratio"] not in _OPENAI_SIZES:
            raise GenerationError("OpenAI aspect_ratio must be an explicit pixel size")
    elif provider == "gemini":
        if model != _GEMINI_MODEL:
            raise GenerationError("Gemini rows must use Nano Banana 2")
        if row["aspect_ratio"] not in _GEMINI_ASPECTS:
            raise GenerationError("unsupported Gemini aspect ratio")
    else:
        raise GenerationError(f"unsupported provider: {provider}")
    if row["quality"] not in {"medium", "high"}:
        raise GenerationError("quality must be medium or high")
    reference = row["reference"]
    if reference is not None:
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise GenerationError("reference schema mismatch")
        if not isinstance(reference["path"], str) or not Path(reference["path"]).is_absolute():
            raise GenerationError("reference path must be absolute")
        if not isinstance(reference["sha256"], str) or not _SHA256.fullmatch(
            reference["sha256"]
        ):
            raise GenerationError("reference SHA-256 is invalid")
    elif provider == "gemini" and fixture_id.startswith("W6-PRODUCT") and purpose != "smoke":
        raise GenerationError("product materialization rows require a reference image")
    rights_reference = _validate_rights_reference(row["rights_reference"])
    if rights_reference != _PROVIDER_RIGHTS[provider]:
        raise GenerationError(
            f"{provider} rows must cite the pinned provider terms URL"
        )
    return {
        "id": row_id,
        "provider": provider,
        "model": model,
        "prompt": _validate_text(row["prompt"], "prompt"),
        "caption": _validate_text(row["caption"], "caption", maximum=8_000),
        "rights_reference": rights_reference,
        "aspect_ratio": row["aspect_ratio"],
        "quality": row["quality"],
        "reference": reference,
    }


def load_plan(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha = _read_json(path, "generation plan")
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "kind",
        "purpose",
        "social_relief",
        "fixtures",
    }:
        raise GenerationError("generation plan schema mismatch")
    if value["schema"] != 1 or value["kind"] != "sn56-week6-lane-b-generation-plan":
        raise GenerationError("generation plan identity mismatch")
    purpose = value["purpose"]
    if purpose not in {"smoke", "reference-assets", "materialization"}:
        raise GenerationError(
            "plan purpose must be smoke, reference-assets, or materialization"
        )
    social_relief = value["social_relief"]
    if not isinstance(social_relief, bool) or (
        purpose != "materialization" and social_relief
    ):
        raise GenerationError("social_relief is valid only for materialization")
    fixtures = value["fixtures"]
    if not isinstance(fixtures, list) or not fixtures:
        raise GenerationError("plan fixtures must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    seen_fixtures: set[str] = set()
    seen_rows: set[str] = set()
    product_reference_shas: dict[str, str] = {}
    for fixture in fixtures:
        if not isinstance(fixture, dict) or set(fixture) != {"id", "rows"}:
            raise GenerationError("fixture schema mismatch")
        fixture_id = fixture["id"]
        if fixture_id not in _ALLOWED_FIXTURES or fixture_id in seen_fixtures:
            raise GenerationError(f"unknown or duplicate fixture: {fixture_id}")
        rows = fixture["rows"]
        if not isinstance(rows, list) or not rows:
            raise GenerationError(f"fixture {fixture_id} has no rows")
        if purpose == "materialization" and len(rows) != _ALLOWED_FIXTURES[fixture_id]:
            raise GenerationError(f"fixture {fixture_id} has the wrong materialization count")
        normalized_rows = []
        for row in rows:
            normalized_row = _normalize_row(row, fixture_id=fixture_id, purpose=purpose)
            if purpose == "materialization":
                required_provider = _MATERIALIZATION_PROVIDERS[fixture_id]
                if normalized_row["provider"] != required_provider:
                    raise GenerationError(
                        f"fixture {fixture_id} requires provider {required_provider}"
                    )
                if fixture_id.startswith("W6-PRODUCT"):
                    reference_sha = normalized_row["reference"]["sha256"]
                    previous = product_reference_shas.setdefault(
                        fixture_id, reference_sha
                    )
                    if previous != reference_sha:
                        raise GenerationError(
                            f"fixture {fixture_id} must use one canonical reference"
                        )
                elif normalized_row["reference"] is not None:
                    raise GenerationError(
                        f"fixture {fixture_id} must not use reference images"
                    )
            if purpose == "reference-assets" and (
                not fixture_id.startswith("W6-PRODUCT")
                or normalized_row["provider"] != "openai"
            ):
                raise GenerationError(
                    "reference-assets plans contain only OpenAI product references"
                )
            global_id = f"{fixture_id}/{normalized_row['id']}"
            if global_id in seen_rows:
                raise GenerationError(f"duplicate global row id: {global_id}")
            seen_rows.add(global_id)
            normalized_rows.append(normalized_row)
        seen_fixtures.add(fixture_id)
        normalized.append({"id": fixture_id, "rows": normalized_rows})
    if purpose == "materialization":
        required_fixtures = set(_ALLOWED_FIXTURES)
        if social_relief:
            required_fixtures.remove("W6-SOCIAL-A")
        if seen_fixtures != required_fixtures:
            raise GenerationError(
                "materialization fixture set does not match the declared relief mode"
            )
        references = set(product_reference_shas.values())
        if len(references) != 2:
            raise GenerationError(
                "product fixtures A and B must use distinct canonical references"
            )
        if not social_relief:
            social_rows = next(
                fixture["rows"]
                for fixture in normalized
                if fixture["id"] == "W6-SOCIAL-A"
            )
            ratios = {ratio: 0 for ratio in ("1:1", "4:5", "16:9")}
            for row in social_rows:
                if row["aspect_ratio"] not in ratios:
                    raise GenerationError(
                        "W6-SOCIAL-A supports only 1:1, 4:5, and 16:9"
                    )
                ratios[row["aspect_ratio"]] += 1
            if ratios != {"1:1": 6, "4:5": 6, "16:9": 6}:
                raise GenerationError(
                    "W6-SOCIAL-A must contain six rows at each required ratio"
                )
    normalized_plan = {
        "schema": 1,
        "kind": "sn56-week6-lane-b-generation-plan",
        "purpose": purpose,
        "social_relief": social_relief,
        "fixtures": normalized,
    }
    if purpose != "smoke":
        try:
            planner.validate_frozen_blueprint(normalized_plan)
        except planner.PlanError as exc:
            raise GenerationError(
                "generation plan does not match the frozen blueprint"
            ) from exc
    return normalized_plan, file_sha


def _http_transport(
    url: str, headers: Mapping[str, str], body: bytes, timeout: float
) -> HttpResponse:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(response.status, dict(response.headers.items()), response.read())
    except urllib.error.HTTPError as exc:
        return HttpResponse(exc.code, dict(exc.headers.items()), exc.read(64 * 1024))


def _decode_json_response(response: HttpResponse) -> Any:
    try:
        return json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GenerationError("provider response is not valid UTF-8 JSON") from exc


def _openai_request(row: Mapping[str, Any], key: str) -> tuple[str, dict[str, str], bytes]:
    payload = {
        "model": row["model"],
        "prompt": row["prompt"],
        "n": 1,
        "size": row["aspect_ratio"],
        "quality": row["quality"],
        "output_format": "png",
    }
    return (
        _OPENAI_ENDPOINT,
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        _canonical_bytes(payload),
    )


def _gemini_request(
    row: Mapping[str, Any], key: str
) -> tuple[str, dict[str, str], bytes, str | None]:
    parts: list[dict[str, Any]] = [{"text": row["prompt"]}]
    reference_sha: str | None = None
    if row["reference"] is not None:
        reference_path = Path(row["reference"]["path"])
        reference_bytes = _stable_regular(reference_path, "reference image")
        reference_sha = _sha256_bytes(reference_bytes)
        if reference_sha != row["reference"]["sha256"]:
            raise GenerationError("reference image SHA-256 mismatch")
        try:
            with Image.open(BytesIO(reference_bytes)) as image:
                image.verify()
                mime = Image.MIME.get(image.format or "")
        except Exception as exc:
            raise GenerationError("reference image is not loadable") from exc
        if mime not in {"image/png", "image/jpeg", "image/webp"}:
            raise GenerationError("unsupported reference image MIME type")
        parts.append(
            {
                "inline_data": {
                    "mime_type": mime,
                    "data": base64.b64encode(reference_bytes).decode("ascii"),
                }
            }
        )
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "responseFormat": {
                "image": {
                    "aspectRatio": row["aspect_ratio"],
                    "imageSize": "1K" if row["quality"] == "medium" else "2K",
                }
            },
        },
    }
    return (
        _GEMINI_ENDPOINT,
        {"x-goog-api-key": key, "Content-Type": "application/json"},
        _canonical_bytes(payload),
        reference_sha,
    )


def _extract_image(provider: str, payload: Any) -> tuple[bytes, str]:
    encoded: Any = None
    mime = "image/png"
    if provider == "openai":
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            encoded = data[0].get("b64_json")
    else:
        candidates = payload.get("candidates") if isinstance(payload, dict) else None
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
            content = candidates[0].get("content")
            parts = content.get("parts") if isinstance(content, dict) else None
            images = []
            if isinstance(parts, list):
                for part in parts:
                    if not isinstance(part, dict) or part.get("thought") is True:
                        continue
                    inline = part.get("inlineData", part.get("inline_data"))
                    if isinstance(inline, dict) and isinstance(inline.get("data"), str):
                        images.append(inline)
            if len(images) == 1:
                encoded = images[0]["data"]
                mime = images[0].get("mimeType", images[0].get("mime_type", mime))
    if not isinstance(encoded, str) or mime not in {
        "image/png",
        "image/jpeg",
        "image/webp",
    }:
        raise GenerationError("provider response did not contain exactly one supported image")
    try:
        return base64.b64decode(encoded, validate=True), mime
    except ValueError as exc:
        raise GenerationError("provider image is not valid base64") from exc


def _normalize_png(image_bytes: bytes) -> tuple[bytes, int, int]:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            width, height = image.size
            if width <= 0 or height <= 0:
                raise GenerationError("provider image has invalid dimensions")
            normalized = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            output = BytesIO()
            normalized.save(output, format="PNG", optimize=False, compress_level=6)
            return output.getvalue(), width, height
    except GenerationError:
        raise
    except Exception as exc:
        raise GenerationError("provider image is not loadable") from exc


def _assert_expected_ratio(row: Mapping[str, Any], width: int, height: int) -> None:
    raw = row["aspect_ratio"]
    separator = "x" if row["provider"] == "openai" else ":"
    expected_width, expected_height = (int(item) for item in raw.split(separator))
    relative_error = abs(width * expected_height - height * expected_width) / (
        height * expected_width
    )
    if relative_error > 0.01:
        raise GenerationError(
            f"provider image ratio differs from requested {row['aspect_ratio']}"
        )


def _provider_request_id(headers: Mapping[str, str]) -> str | None:
    lowered = {key.lower(): value for key, value in headers.items()}
    for key in ("x-request-id", "x-goog-request-id", "x-guploader-uploadid"):
        value = lowered.get(key)
        if value:
            return value
    return None


def _validate_success_receipt(
    receipt_path: Path,
    *,
    fixture_id: str,
    row: Mapping[str, Any],
    row_plan_sha: str,
    plan_file_sha: str,
    origin: str,
) -> dict[str, Any]:
    value, _ = _read_json(receipt_path, "success receipt")
    if (
        not isinstance(value, dict)
        or set(value) != _SUCCESS_RECEIPT_KEYS
        or value.get("schema") != 1
        or value.get("kind") != "sn56-week6-lane-b-generation-attempt"
        or value.get("state") != "SUCCESS"
        or value.get("origin") != origin
        or value.get("rights_status") != "approved-provider-terms"
        or value.get("fixture_id") != fixture_id
        or value.get("row_id") != row["id"]
        or value.get("provider") != row["provider"]
        or value.get("model") != row["model"]
        or value.get("plan_file_sha256") != plan_file_sha
        or value.get("row_plan_sha256") != row_plan_sha
    ):
        raise GenerationError("existing success receipt does not match this row plan")
    expected_image = (receipt_path.parent / "images" / f"{row['id']}.png").absolute()
    expected_caption = (receipt_path.parent / "captions" / f"{row['id']}.txt").absolute()
    image_path = Path(value.get("output_path", ""))
    caption_path = Path(value.get("caption_path", ""))
    if image_path != expected_image or caption_path != expected_caption:
        raise GenerationError("existing success receipt points outside its row scope")
    image_data = _stable_regular(image_path, "resumed output image")
    caption_data = _stable_regular(caption_path, "resumed caption")
    if _sha256_bytes(image_data) != value.get("output_image_sha256"):
        raise GenerationError("resumed output image hash mismatch")
    if _sha256_bytes(caption_data) != value.get("caption_sha256"):
        raise GenerationError("resumed caption hash mismatch")
    if caption_data != row["caption"].encode("utf-8"):
        raise GenerationError("resumed caption bytes differ from the plan")
    if len(image_data) != value.get("output_image_bytes"):
        raise GenerationError("resumed output image size mismatch")
    png, width, height = _normalize_png(image_data)
    if png != image_data or (width, height) != (value.get("width"), value.get("height")):
        raise GenerationError("resumed output image encoding or dimensions mismatch")
    _assert_expected_ratio(row, width, height)
    return value


def _attempt_count(attempts_dir: Path, row_id: str) -> int:
    pattern = re.compile(re.escape(row_id) + r"\.attempt-([0-9]{3})\.json")
    numbers = []
    for path in attempts_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0)


def _run_row_locked(
    row: Mapping[str, Any],
    *,
    fixture_id: str,
    plan_file_sha: str,
    fixture_dir: Path,
    transport: Transport,
    origin: str,
    timeout_s: float,
    max_attempts: int,
) -> dict[str, Any]:
    images_dir = fixture_dir / "images"
    captions_dir = fixture_dir / "captions"
    attempts_dir = fixture_dir / "attempts"
    for directory in (images_dir, captions_dir, attempts_dir):
        _ensure_directory(directory)
    success_path = fixture_dir / f"{row['id']}.success.json"
    row_plan_sha = _canonical_sha256({"fixture_id": fixture_id, "row": row})
    if success_path.exists() or success_path.is_symlink():
        return _validate_success_receipt(
            success_path,
            fixture_id=fixture_id,
            row=row,
            row_plan_sha=row_plan_sha,
            plan_file_sha=plan_file_sha,
            origin=origin,
        )

    existing_attempts = _attempt_count(attempts_dir, row["id"])
    if existing_attempts >= max_attempts:
        raise GenerationError(f"attempt budget exhausted for {fixture_id}/{row['id']}")
    key_name = "OPENAI_API_KEY" if row["provider"] == "openai" else "GEMINI_API_KEY"
    key = os.environ.get(key_name)
    if not key or not key.strip():
        raise GenerationError(f"{key_name} is required for {row['provider']} rows")

    last_error: Exception | None = None
    for attempt in range(existing_attempts + 1, max_attempts + 1):
        client_request_id = str(uuid.uuid4())
        started = _utc_now()
        status: int | None = None
        provider_request_id: str | None = None
        reference_sha: str | None = None
        url: str | None = None
        body: bytes | None = None
        attempt_path = attempts_dir / f"{row['id']}.attempt-{attempt:03d}.json"
        image_path = images_dir / f"{row['id']}.png"
        caption_path = captions_dir / f"{row['id']}.txt"
        image_created = False
        caption_created = False
        try:
            if row["provider"] == "openai":
                url, headers, body = _openai_request(row, key)
            else:
                url, headers, body, reference_sha = _gemini_request(row, key)
            response = transport(url, headers, body, timeout_s)
            status = response.status
            provider_request_id = _provider_request_id(response.headers)
            if not 200 <= status < 300:
                raise GenerationError(f"provider HTTP status {status}")
            response_value = _decode_json_response(response)
            raw_image, response_mime = _extract_image(row["provider"], response_value)
            png, width, height = _normalize_png(raw_image)
            _assert_expected_ratio(row, width, height)
            _publish_exclusive(image_path, png)
            image_created = True
            try:
                _publish_exclusive(caption_path, row["caption"].encode("utf-8"))
                caption_created = True
            except Exception:
                if image_created:
                    image_path.unlink(missing_ok=True)
                raise
            receipt = {
                "schema": 1,
                "kind": "sn56-week6-lane-b-generation-attempt",
                "state": "SUCCESS",
                "origin": origin,
                "fixture_id": fixture_id,
                "row_id": row["id"],
                "attempt": attempt,
                "client_request_id": client_request_id,
                "provider_request_id": provider_request_id,
                "request_id": provider_request_id or client_request_id,
                "provider": row["provider"],
                "model": row["model"],
                "endpoint": url,
                "request_started_at": started,
                "request_finished_at": _utc_now(),
                "plan_file_sha256": plan_file_sha,
                "row_plan_sha256": row_plan_sha,
                "request_payload_sha256": _sha256_bytes(body),
                "prompt": row["prompt"],
                "prompt_sha256": _sha256_bytes(row["prompt"].encode()),
                "caption_sha256": _sha256_bytes(row["caption"].encode()),
                "rights_reference": row["rights_reference"],
                "rights_status": "approved-provider-terms",
                "reference_sha256": reference_sha,
                "response_mime_type": response_mime,
                "response_image_sha256": _sha256_bytes(raw_image),
                "output_path": str(image_path.absolute()),
                "output_image_sha256": _sha256_bytes(png),
                "output_image_bytes": len(png),
                "caption_path": str(caption_path.absolute()),
                "width": width,
                "height": height,
                "http_status": status,
                "error": None,
            }
            _publish_json(attempt_path, receipt)
            try:
                os.link(attempt_path, success_path)
                _fsync_directory(attempt_path.parent)
                if success_path.parent != attempt_path.parent:
                    _fsync_directory(success_path.parent)
            except Exception:
                if image_created:
                    image_path.unlink(missing_ok=True)
                if caption_created:
                    caption_path.unlink(missing_ok=True)
                raise
            return _validate_success_receipt(
                success_path,
                fixture_id=fixture_id,
                row=row,
                row_plan_sha=row_plan_sha,
                plan_file_sha=plan_file_sha,
                origin=origin,
            )
        except Exception as exc:
            last_error = exc
            if not success_path.exists():
                if image_created:
                    image_path.unlink(missing_ok=True)
                if caption_created:
                    caption_path.unlink(missing_ok=True)
            failure = {
                "schema": 1,
                "kind": "sn56-week6-lane-b-generation-attempt",
                "state": "FAIL",
                "origin": origin,
                "fixture_id": fixture_id,
                "row_id": row["id"],
                "attempt": attempt,
                "client_request_id": client_request_id,
                "provider_request_id": provider_request_id,
                "request_id": provider_request_id or client_request_id,
                "provider": row["provider"],
                "model": row["model"],
                "endpoint": url,
                "request_started_at": started,
                "request_finished_at": _utc_now(),
                "plan_file_sha256": plan_file_sha,
                "row_plan_sha256": row_plan_sha,
                "request_payload_sha256": (
                    _sha256_bytes(body) if body is not None else None
                ),
                "prompt_sha256": _sha256_bytes(row["prompt"].encode()),
                "rights_status": "approved-provider-terms",
                "reference_sha256": reference_sha,
                "http_status": status,
                "error": {
                    "class": type(exc).__name__,
                    "message_sha256": _sha256_bytes(str(exc).encode()),
                },
            }
            if not attempt_path.exists():
                _publish_json(attempt_path, failure)
            transient = isinstance(exc, (urllib.error.URLError, TimeoutError)) or status in {
                408,
                409,
                429,
            } or (
                isinstance(status, int) and 500 <= status <= 599
            )
            if not transient:
                break
            time.sleep(min(2 ** (attempt - 1), 8))
    raise GenerationError(f"generation failed for {fixture_id}/{row['id']}") from last_error


@contextmanager
def _row_lock(fixture_dir: Path, row_id: str):
    locks_dir = fixture_dir / ".locks"
    _ensure_directory(locks_dir)
    lock_path = locks_dir / f"{row_id}.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise GenerationError(f"cannot open row lock safely: {lock_path}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise GenerationError(f"row lock is not a regular file: {lock_path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _run_row(
    row: Mapping[str, Any],
    *,
    fixture_id: str,
    plan_file_sha: str,
    fixture_dir: Path,
    transport: Transport,
    origin: str = "synthetic",
    timeout_s: float,
    max_attempts: int,
) -> dict[str, Any]:
    with _row_lock(fixture_dir, row["id"]):
        return _run_row_locked(
            row,
            fixture_id=fixture_id,
            plan_file_sha=plan_file_sha,
            fixture_dir=fixture_dir,
            transport=transport,
            origin=origin,
            timeout_s=timeout_s,
            max_attempts=max_attempts,
        )


def execute(
    plan_path: Path,
    output_dir: Path,
    *,
    transport: Transport | None = None,
    timeout_s: float = 180.0,
    max_attempts: int = 3,
) -> dict[str, Any]:
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise GenerationError("timeout must be finite and positive")
    if not 1 <= max_attempts <= 9:
        raise GenerationError("max_attempts must be in [1, 9]")
    plan, plan_file_sha = load_plan(plan_path.resolve(strict=True))
    if transport is None:
        transport = _http_transport
        origin = "real"
    else:
        origin = "synthetic"
    output_dir = output_dir.absolute()
    _ensure_directory(output_dir)
    results = []
    for fixture in plan["fixtures"]:
        fixture_dir = output_dir / fixture["id"]
        _ensure_directory(fixture_dir)
        for row in fixture["rows"]:
            results.append(
                _run_row(
                    row,
                    fixture_id=fixture["id"],
                    plan_file_sha=plan_file_sha,
                    fixture_dir=fixture_dir,
                    transport=transport,
                    origin=origin,
                    timeout_s=timeout_s,
                    max_attempts=max_attempts,
                )
            )
    summary = {
        "schema": 1,
        "kind": "sn56-week6-lane-b-generation-summary",
        "status": "complete",
        "purpose": plan["purpose"],
        "social_relief": plan["social_relief"],
        "plan_file_sha256": plan_file_sha,
        "row_count": len(results),
        "success_count": sum(row["state"] == "SUCCESS" for row in results),
        "success_receipt_sha256": sorted(
            _canonical_sha256(row) for row in results
        ),
    }
    summary_path = output_dir / "generation-summary.json"
    try:
        _publish_json(summary_path, summary)
    except FileExistsError:
        existing, _ = _read_json(summary_path, "generation summary")
        if existing != summary:
            raise GenerationError("existing generation summary differs")
    return summary


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    plan_path = Path(args.plan).expanduser()
    if args.validate_only:
        plan, file_sha = load_plan(plan_path.resolve(strict=True))
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "purpose": plan["purpose"],
                    "social_relief": plan["social_relief"],
                    "fixture_count": len(plan["fixtures"]),
                    "row_count": sum(len(item["rows"]) for item in plan["fixtures"]),
                    "plan_file_sha256": file_sha,
                },
                sort_keys=True,
            )
        )
        return 0
    summary = execute(
        plan_path,
        Path(args.output_dir).expanduser(),
        timeout_s=args.timeout_s,
        max_attempts=args.max_attempts,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
