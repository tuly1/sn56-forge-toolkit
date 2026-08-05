#!/usr/bin/env python3
"""Validate, split, and publish blinded Week-6 Lane-B fixture packages.

The source manifests are immutable, private records.  This module validates
their exact schema and the bytes they name before reading the custodian key.
It then derives a deterministic 90/10 split and publishes separate train and
holdout manifests plus a blinded acceptance record.  The acceptance record is
written last and is the package's terminal marker.

The custodian key is input-only: it is never hashed into, copied into, or
otherwise represented by any output record.

Path checks in this lab-only utility reject static symlink/FIFO substitution.
They assume the 0700 working tree is owned by one trusted operator and is not
being concurrently replaced by a hostile process running as that same UID.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import hmac
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import unicodedata
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from PIL import Image, UnidentifiedImageError

import lane_b_plan as planner


_SOURCE_KIND = "forge-week6-lane-b-source-manifest"
_SPLIT_KIND = "forge-week6-lane-b-split-manifest"
_ACCEPTANCE_KIND = "forge-week6-lane-b-blinded-acceptance"
_EXPORT_KIND = "forge-week6-lane-b-split-byte-export"
_SCHEMA = 1
_OPENAI_MODEL = "gpt-image-2-2026-04-21"
_GEMINI_MODEL = "gemini-3.1-flash-image"
_OPENAI_RIGHTS = "https://openai.com/policies/services-agreement/"
_GEMINI_RIGHTS = "https://ai.google.dev/gemini-api/terms"
_SPLIT_ALGORITHM = "hmac-sha256-fixture-id-plus-manifest-sha-fisher-yates-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ROW_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_RIGHTS_STATUSES = frozenset(
    {
        "approved-public-domain",
        "approved-cc0",
        "approved-cc-by",
        "approved-cc-by-sa",
        "approved-provider-terms",
    }
)

# These are the complete first-wave matrix from the amended 2026-08-04 spec.
# The generator label remains manifest data because W6-SOCIAL-A deliberately
# allows either approved provider.  Identity, family, subtype, and cardinality
# do not vary.
_FIXTURES: dict[str, dict[str, Any]] = {
    "W6-DESIGN-GPT-A": {
        "family": "design/UI",
        "subtype": "desktop/web",
        "total_pairs": 21,
        "generator_labels": ("GPT Image 2",),
    },
    "W6-DESIGN-NB-B": {
        "family": "design/UI",
        "subtype": "mobile app",
        "total_pairs": 20,
        "generator_labels": ("Nano Banana 2",),
    },
    "W6-PRODUCT-A": {
        "family": "product",
        "subtype": "physical product identity A",
        "total_pairs": 21,
        "generator_labels": ("GPT Image 2 reference → Nano Banana 2 edits",),
    },
    "W6-PRODUCT-B": {
        "family": "product",
        "subtype": "physical product identity B",
        "total_pairs": 20,
        "generator_labels": ("GPT Image 2 reference → Nano Banana 2 edits",),
    },
    "W6-SOCIAL-A": {
        "family": "social",
        "subtype": "mixed 1:1, 4:5, 16:9",
        "total_pairs": 18,
        "generator_labels": ("Nano Banana 2",),
    },
    "W6-DESIGN-GPT-LARGE": {
        "family": "design/UI",
        "subtype": "desktop/web",
        "total_pairs": 48,
        "generator_labels": ("GPT Image 2",),
    },
}

_SOURCE_KEYS = {
    "schema",
    "kind",
    "fixture_id",
    "family",
    "subtype",
    "generator_label",
    "total_pairs",
    "rows",
    "reference_provenance",
}
_ROW_KEYS = {
    "row_id",
    "image_path",
    "caption_path",
    "provider",
    "model",
    "request_time_utc",
    "request_id",
    "prompt",
    "output_sha256",
    "rights_reference",
    "rights_status",
    "width",
    "height",
    "aspect_ratio",
    "image_sha256",
    "caption_sha256",
    "reference_sha256",
}
_PLAN_KEYS = {"schema", "kind", "purpose", "social_relief", "fixtures"}
_PLAN_FIXTURE_KEYS = {"id", "rows"}
_PLAN_ROW_KEYS = {
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
_QA_KIND = "forge-week6-lane-b-human-visual-qa"
_QA_KEYS = {
    "schema",
    "kind",
    "state",
    "reviewer",
    "reviewed_at_utc",
    "generation_plan_sha256",
    "social_relief",
    "fixture_count",
    "fixtures",
}
_QA_FIXTURE_KEYS = {
    "fixture_id",
    "source_manifest_sha256",
    "row_count",
    "rows",
}
_QA_ROW_KEYS = {"row_id", "image_sha256", "caption_sha256", "checks"}
_QA_CHECK_KEYS = {
    "required_visible_text",
    "caption_pixel_match",
    "product_identity_preserved",
    "prohibited_brand_person_absent",
    "useful_fixture",
    "perceptual_duplicate_screen",
}
_REFERENCE_PROVENANCE_KEYS = {
    "schema",
    "kind",
    "fixture_id",
    "row_id",
    "provider",
    "model",
    "reference_plan_file_sha256",
    "reference_generation_summary_sha256",
    "reference_success_receipt_sha256",
    "reference_request_payload_sha256",
    "reference_output_sha256",
    "reference_rights_reference",
    "reference_rights_status",
}


def canonical_bytes(value: Any) -> bytes:
    """Return the one accepted JSON encoding for every manifest."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = keys - set(value)
    extra = set(value) - keys
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _text(value: Any, label: str, *, exact_whitespace: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != unicodedata.normalize("NFC", value):
        raise ValueError(f"{label} must use NFC Unicode")
    if not exact_whitespace and value != " ".join(value.split()):
        raise ValueError(f"{label} must use canonical whitespace")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _canonical_utc(value: Any, label: str) -> str:
    value = _text(value, label)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        raise ValueError(f"{label} must be canonical whole-second UTC")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{label} must be real UTC") from exc
    return value


def _relative_path(value: Any, label: str, suffixes: set[str]) -> PurePosixPath:
    value = _text(value, label)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix.casefold() not in suffixes
        or path.as_posix() != value
    ):
        raise ValueError(f"{label} is not a safe canonical relative path")
    return path


def _safe_directory(path: Path | str, label: str, *, create: bool = False) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    if create and not path.exists():
        parent = path.parent
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError(f"{label} parent must be a real directory: {parent}")
        path.mkdir(mode=0o700)
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink component: {current}")
        current = current.parent
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"{label} must be a real directory: {path}")
    return path


def _read_regular(path: Path | str, label: str) -> bytes:
    """Read one regular file through a no-follow descriptor and bind identity."""

    path = Path(os.path.abspath(os.path.expanduser(path)))
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened safely: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
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
    )
    if identity(before) != identity(after):
        raise RuntimeError(f"{label} changed while read: {path}")
    return b"".join(chunks)


def _read_canonical_json(path: Path | str, label: str) -> tuple[dict[str, Any], str]:
    raw = _read_regular(path, label)
    try:
        value = _object(json.loads(raw), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid JSON") from exc
    if raw != canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline")
    return value, hashlib.sha256(raw).hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_ratio(width: int, height: int) -> str:
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def _decoded_dimensions(raw: bytes, label: str) -> tuple[int, int]:
    """Decode enough of an image to bind manifest geometry to actual bytes."""

    try:
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > 100_000_000:
                raise ValueError(f"{label} has unsafe decoded dimensions")
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError(f"{label} is not a valid supported image") from exc
    return width, height


def _rights_reference(value: Any, label: str) -> str:
    value = _text(value, label, exact_whitespace=True)
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username is not None:
        raise ValueError(f"{label} must be an absolute HTTPS terms/rights reference")
    return value


def _validate_row(
    value: Any,
    *,
    fixture_id: str,
    index: int,
    fixture_root: Path,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    row = _object(value, f"{fixture_id}.rows[{index}]")
    _exact(row, _ROW_KEYS, f"{fixture_id}.rows[{index}]")
    row_id = _text(row["row_id"], f"{fixture_id}.rows[{index}].row_id")
    if _ROW_ID.fullmatch(row_id) is None:
        raise ValueError(f"{fixture_id}.rows[{index}].row_id is not canonical")
    image_path = _relative_path(
        row["image_path"],
        f"{fixture_id}.{row_id}.image_path",
        {".jpg", ".jpeg", ".png", ".webp"},
    )
    caption_path = _relative_path(
        row["caption_path"], f"{fixture_id}.{row_id}.caption_path", {".txt"}
    )
    if (
        image_path.stem != row_id
        or caption_path.stem != row_id
        or image_path.parent.name != "images"
        or caption_path.parent.name != "captions"
        or image_path.parent.parent != PurePosixPath(fixture_id)
        or caption_path.parent.parent != PurePosixPath(fixture_id)
    ):
        raise ValueError(
            f"{fixture_id}.{row_id} image/caption paths must match the generated row scope"
        )

    provider = _text(row["provider"], f"{fixture_id}.{row_id}.provider")
    model = _text(row["model"], f"{fixture_id}.{row_id}.model")
    provider_contracts = {
        ("openai", _OPENAI_MODEL): _OPENAI_RIGHTS,
        ("gemini", _GEMINI_MODEL): _GEMINI_RIGHTS,
    }
    expected_rights_reference = provider_contracts.get((provider, model))
    if expected_rights_reference is None:
        raise ValueError(
            f"{fixture_id}.{row_id} provider/model is outside the frozen Lane-B contract"
        )
    _canonical_utc(row["request_time_utc"], f"{fixture_id}.{row_id}.request_time_utc")
    request_id = _text(row["request_id"], f"{fixture_id}.{row_id}.request_id")
    _text(
        row["prompt"],
        f"{fixture_id}.{row_id}.prompt",
        exact_whitespace=True,
    )
    output_sha = _digest(row["output_sha256"], f"{fixture_id}.{row_id}.output_sha256")
    rights_reference = _rights_reference(
        row["rights_reference"], f"{fixture_id}.{row_id}.rights_reference"
    )
    if rights_reference != expected_rights_reference:
        raise ValueError(
            f"{fixture_id}.{row_id}.rights_reference must equal the pinned provider terms"
        )
    rights_status = _text(
        row["rights_status"], f"{fixture_id}.{row_id}.rights_status"
    )
    if rights_status != "approved-provider-terms":
        raise ValueError(
            f"{fixture_id}.{row_id}.rights_status must be approved-provider-terms"
        )
    width = _positive_integer(row["width"], f"{fixture_id}.{row_id}.width")
    height = _positive_integer(row["height"], f"{fixture_id}.{row_id}.height")
    expected_ratio = _canonical_ratio(width, height)
    if row["aspect_ratio"] != expected_ratio:
        raise ValueError(
            f"{fixture_id}.{row_id}.aspect_ratio must be {expected_ratio}"
        )
    image_sha = _digest(row["image_sha256"], f"{fixture_id}.{row_id}.image_sha256")
    caption_sha = _digest(
        row["caption_sha256"], f"{fixture_id}.{row_id}.caption_sha256"
    )
    reference_sha = row["reference_sha256"]
    if fixture_id.startswith("W6-PRODUCT"):
        reference_sha = _digest(
            reference_sha, f"{fixture_id}.{row_id}.reference_sha256"
        )
    elif reference_sha is not None:
        raise ValueError(f"{fixture_id}.{row_id}.reference_sha256 must be null")

    image = _read_regular(fixture_root / image_path, f"{fixture_id}.{row_id}.image")
    caption = _read_regular(
        fixture_root / caption_path, f"{fixture_id}.{row_id}.caption"
    )
    if not image:
        raise ValueError(f"{fixture_id}.{row_id}.image is empty")
    if not caption:
        raise ValueError(f"{fixture_id}.{row_id}.caption is empty")
    if _sha256(image) != image_sha:
        raise ValueError(f"{fixture_id}.{row_id}.image hash mismatch")
    if output_sha != image_sha:
        raise ValueError(
            f"{fixture_id}.{row_id}.output_sha256 must bind the exact image bytes"
        )
    if _sha256(caption) != caption_sha:
        raise ValueError(f"{fixture_id}.{row_id}.caption hash mismatch")
    decoded_width, decoded_height = _decoded_dimensions(
        image, f"{fixture_id}.{row_id}.image"
    )
    if (width, height) != (decoded_width, decoded_height):
        raise ValueError(
            f"{fixture_id}.{row_id} declared dimensions do not match decoded image"
        )
    try:
        caption_text = caption.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{fixture_id}.{row_id}.caption must be UTF-8") from exc
    if not caption_text.strip() or caption_text != unicodedata.normalize("NFC", caption_text):
        raise ValueError(f"{fixture_id}.{row_id}.caption must be non-empty NFC text")

    # Return a fresh dictionary so downstream records cannot retain aliases to
    # a caller-owned mutable object.
    return dict(row), {
        "image": image,
        "caption": caption,
        "provider_request": f"{provider}\0{request_id}".encode("utf-8"),
    }


def _validate_generator_provenance(
    fixture_id: str,
    generator_label: str,
    rows: Sequence[dict[str, Any]],
) -> None:
    """Bind the declared fixture generator to every row's provider/model."""

    expected = {
        "GPT Image 2": ("openai", _OPENAI_MODEL),
        "Nano Banana 2": ("gemini", _GEMINI_MODEL),
        "GPT Image 2 reference → Nano Banana 2 edits": (
            "gemini",
            _GEMINI_MODEL,
        ),
    }[generator_label]
    for row in rows:
        actual = (row["provider"], row["model"])
        if actual != expected:
            raise ValueError(
                f"{fixture_id}.{row['row_id']} provider/model {actual!r} does not "
                f"match generator_label {generator_label!r}"
            )
    if fixture_id.startswith("W6-PRODUCT"):
        references = {row["reference_sha256"] for row in rows}
        if len(references) != 1:
            raise ValueError(f"{fixture_id} must bind one canonical reference SHA")
    if fixture_id == "W6-SOCIAL-A":
        ratios = _distribution(row["aspect_ratio"] for row in rows)
        if ratios != {"16:9": 6, "1:1": 6, "4:5": 6}:
            raise ValueError(
                "W6-SOCIAL-A must contain six rows at each required ratio"
            )


def validate_source_manifest(
    manifest_path: Path | str, fixture_root: Path | str
) -> dict[str, Any]:
    """Validate one immutable source manifest and every named byte."""

    fixture_root = _safe_directory(fixture_root, "fixture root")
    manifest, manifest_sha = _read_canonical_json(manifest_path, "source manifest")
    _exact(manifest, _SOURCE_KEYS, "source manifest")
    if manifest.get("schema") != _SCHEMA or manifest.get("kind") != _SOURCE_KIND:
        raise ValueError("source manifest schema/kind is unsupported")
    fixture_id = _text(manifest["fixture_id"], "fixture_id")
    expected = _FIXTURES.get(fixture_id)
    if expected is None:
        raise ValueError(f"unknown Lane-B fixture_id: {fixture_id}")
    for key in ("family", "subtype", "total_pairs"):
        if manifest.get(key) != expected[key]:
            raise ValueError(
                f"{fixture_id}.{key} must equal the frozen specification value "
                f"{expected[key]!r}"
            )
    generator_label = _text(
        manifest["generator_label"], f"{fixture_id}.generator_label"
    )
    if generator_label not in expected["generator_labels"]:
        raise ValueError(
            f"{fixture_id}.generator_label must be one of the frozen labels "
            f"{expected['generator_labels']!r}"
        )
    rows = manifest["rows"]
    if not isinstance(rows, list) or len(rows) != expected["total_pairs"]:
        raise ValueError(
            f"{fixture_id}.rows must contain exactly {expected['total_pairs']} rows"
        )

    validated: list[dict[str, Any]] = []
    byte_records: list[dict[str, bytes]] = []
    row_ids: set[str] = set()
    image_paths: set[str] = set()
    caption_paths: set[str] = set()
    image_hashes: set[str] = set()
    caption_hashes: set[str] = set()
    provider_requests: set[bytes] = set()
    for index, value in enumerate(rows):
        row, byte_record = _validate_row(
            value, fixture_id=fixture_id, index=index, fixture_root=fixture_root
        )
        checks = (
            (row_ids, row["row_id"], "row_id"),
            (image_paths, row["image_path"], "image_path"),
            (caption_paths, row["caption_path"], "caption_path"),
            (image_hashes, row["image_sha256"], "image_sha256"),
            (caption_hashes, row["caption_sha256"], "caption_sha256"),
            (provider_requests, byte_record["provider_request"], "provider/request_id"),
        )
        for seen, item, label in checks:
            if item in seen:
                raise ValueError(f"{fixture_id} duplicate {label}: {item!r}")
            seen.add(item)
        validated.append(row)
        byte_records.append(byte_record)
    _validate_generator_provenance(fixture_id, generator_label, validated)
    reference_provenance = manifest["reference_provenance"]
    if fixture_id.startswith("W6-PRODUCT"):
        reference = _object(
            reference_provenance, f"{fixture_id}.reference_provenance"
        )
        _exact(
            reference,
            _REFERENCE_PROVENANCE_KEYS,
            f"{fixture_id}.reference_provenance",
        )
        if (
            reference.get("schema") != 1
            or reference.get("kind")
            != "sn56-week6-lane-b-reference-provenance"
            or reference.get("fixture_id") != fixture_id
            or reference.get("provider") != "openai"
            or reference.get("model") != _OPENAI_MODEL
            or reference.get("reference_rights_reference") != _OPENAI_RIGHTS
            or reference.get("reference_rights_status")
            != "approved-provider-terms"
        ):
            raise ValueError(f"{fixture_id}.reference_provenance identity mismatch")
        _text(reference["row_id"], f"{fixture_id}.reference_provenance.row_id")
        for key in (
            "reference_plan_file_sha256",
            "reference_generation_summary_sha256",
            "reference_success_receipt_sha256",
            "reference_request_payload_sha256",
            "reference_output_sha256",
        ):
            _digest(reference[key], f"{fixture_id}.reference_provenance.{key}")
        expected_reference_sha = validated[0]["reference_sha256"]
        if reference["reference_output_sha256"] != expected_reference_sha:
            raise ValueError(
                f"{fixture_id}.reference_provenance output does not bind product rows"
            )
    elif reference_provenance is not None:
        raise ValueError(f"{fixture_id}.reference_provenance must be null")
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "rows": validated,
        "bytes": byte_records,
    }


def _expected_fixture_ids(social_relief: bool) -> set[str]:
    if not isinstance(social_relief, bool):
        raise ValueError("social_relief must be a boolean")
    expected = set(_FIXTURES)
    if social_relief:
        expected.remove("W6-SOCIAL-A")
    return expected


def _validate_generation_plan_binding(
    generation_plan_path: Path | str,
    validated: Sequence[dict[str, Any]],
    *,
    social_relief: bool,
) -> str:
    """Bind source-manifest rows to the exact pre-generation plan bytes."""

    plan, plan_sha = _read_canonical_json(generation_plan_path, "generation plan")
    _exact(plan, _PLAN_KEYS, "generation plan")
    if (
        plan.get("schema") != 1
        or plan.get("kind") != "sn56-week6-lane-b-generation-plan"
        or plan.get("purpose") != "materialization"
        or plan.get("social_relief") is not social_relief
    ):
        raise ValueError("generation plan identity or relief mode mismatch")
    try:
        planner.validate_frozen_blueprint(plan)
    except planner.PlanError as exc:
        raise ValueError("generation plan differs from the frozen blueprint") from exc
    fixtures = plan["fixtures"]
    if not isinstance(fixtures, list):
        raise ValueError("generation plan fixtures must be a list")
    expected_ids = _expected_fixture_ids(social_relief)
    source_by_id = {item["manifest"]["fixture_id"]: item for item in validated}
    plan_by_id: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(fixtures):
        fixture = _object(value, f"generation plan fixtures[{index}]")
        _exact(fixture, _PLAN_FIXTURE_KEYS, f"generation plan fixtures[{index}]")
        fixture_id = _text(fixture["id"], f"generation plan fixtures[{index}].id")
        if fixture_id in plan_by_id:
            raise ValueError(f"duplicate generation-plan fixture: {fixture_id}")
        plan_by_id[fixture_id] = fixture
    if set(plan_by_id) != expected_ids or set(source_by_id) != expected_ids:
        raise ValueError("generation plan and source manifests must bind the exact fixture set")

    for fixture_id in _FIXTURES:
        if fixture_id not in expected_ids:
            continue
        planned_rows = plan_by_id[fixture_id]["rows"]
        source_item = source_by_id[fixture_id]
        source_rows = source_item["rows"]
        source_bytes = source_item["bytes"]
        if not isinstance(planned_rows, list) or len(planned_rows) != len(source_rows):
            raise ValueError(f"{fixture_id} generation-plan row count mismatch")
        source_by_row = {
            row["row_id"]: (row, byte_record)
            for row, byte_record in zip(source_rows, source_bytes, strict=True)
        }
        seen: set[str] = set()
        for index, value in enumerate(planned_rows):
            row = _object(value, f"generation plan {fixture_id}.rows[{index}]")
            _exact(row, _PLAN_ROW_KEYS, f"generation plan {fixture_id}.rows[{index}]")
            row_id = _text(row["id"], f"generation plan {fixture_id}.rows[{index}].id")
            if row_id in seen or row_id not in source_by_row:
                raise ValueError(f"{fixture_id} generation-plan row identity mismatch")
            seen.add(row_id)
            source_row, byte_record = source_by_row[row_id]
            equality = {
                "provider": source_row["provider"],
                "model": source_row["model"],
                "prompt": source_row["prompt"],
                "rights_reference": source_row["rights_reference"],
            }
            for field, expected in equality.items():
                if row[field] != expected:
                    raise ValueError(
                        f"{fixture_id}.{row_id} generation-plan {field} mismatch"
                    )
            caption = _text(
                row["caption"],
                f"generation plan {fixture_id}.{row_id}.caption",
                exact_whitespace=True,
            ).encode("utf-8")
            if caption != byte_record["caption"]:
                raise ValueError(
                    f"{fixture_id}.{row_id} generation-plan caption mismatch"
                )
            _text(
                row["aspect_ratio"],
                f"generation plan {fixture_id}.{row_id}.aspect_ratio",
            )
            if row["quality"] not in {"medium", "high"}:
                raise ValueError(
                    f"generation plan {fixture_id}.{row_id}.quality is invalid"
                )
            reference = row["reference"]
            if fixture_id.startswith("W6-PRODUCT"):
                if (
                    not isinstance(reference, dict)
                    or set(reference) != {"path", "sha256"}
                    or not isinstance(reference["path"], str)
                    or not Path(reference["path"]).is_absolute()
                    or reference["sha256"] != source_row["reference_sha256"]
                ):
                    raise ValueError(
                        f"{fixture_id}.{row_id} generation-plan reference mismatch"
                    )
            elif reference is not None:
                raise ValueError(
                    f"{fixture_id}.{row_id} generation-plan reference must be null"
                )
        if seen != set(source_by_row):
            raise ValueError(f"{fixture_id} generation plan omits source rows")
    return plan_sha


def _validate_visual_qa_receipt(
    qa_receipt_path: Path | str,
    validated: Sequence[dict[str, Any]],
    *,
    generation_plan_sha256: str,
    social_relief: bool,
) -> dict[str, str]:
    """Validate the private, pre-split human visual-QA receipt."""

    receipt, receipt_sha = _read_canonical_json(
        qa_receipt_path, "human visual-QA receipt"
    )
    _exact(receipt, _QA_KEYS, "human visual-QA receipt")
    expected_ids = _expected_fixture_ids(social_relief)
    if (
        receipt.get("schema") != 1
        or receipt.get("kind") != _QA_KIND
        or receipt.get("state") != "PASS"
        or receipt.get("generation_plan_sha256") != generation_plan_sha256
        or receipt.get("social_relief") is not social_relief
        or receipt.get("fixture_count") != len(expected_ids)
    ):
        raise ValueError("human visual-QA receipt identity or state mismatch")
    reviewer = _text(receipt["reviewer"], "human visual-QA receipt reviewer")
    reviewed_at = _canonical_utc(
        receipt["reviewed_at_utc"], "human visual-QA receipt reviewed_at_utc"
    )
    latest_generation = max(
        row["request_time_utc"]
        for source in validated
        for row in source["rows"]
    )
    if reviewed_at < latest_generation:
        raise ValueError(
            "human visual-QA receipt must not predate generated source bytes"
        )
    fixtures = receipt["fixtures"]
    if not isinstance(fixtures, list) or len(fixtures) != len(expected_ids):
        raise ValueError("human visual-QA receipt fixture count mismatch")
    source_by_id = {item["manifest"]["fixture_id"]: item for item in validated}
    qa_by_id: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(fixtures):
        fixture = _object(value, f"human visual-QA fixtures[{index}]")
        _exact(fixture, _QA_FIXTURE_KEYS, f"human visual-QA fixtures[{index}]")
        fixture_id = _text(fixture["fixture_id"], f"human visual-QA fixtures[{index}].id")
        if fixture_id in qa_by_id:
            raise ValueError(f"duplicate human visual-QA fixture: {fixture_id}")
        qa_by_id[fixture_id] = fixture
    if set(qa_by_id) != expected_ids:
        raise ValueError("human visual-QA receipt fixture set mismatch")

    for fixture_id in _FIXTURES:
        if fixture_id not in expected_ids:
            continue
        source = source_by_id[fixture_id]
        qa_fixture = qa_by_id[fixture_id]
        if (
            qa_fixture["source_manifest_sha256"] != source["manifest_sha256"]
            or qa_fixture["row_count"] != len(source["rows"])
        ):
            raise ValueError(f"{fixture_id} human visual-QA source binding mismatch")
        qa_rows = qa_fixture["rows"]
        if not isinstance(qa_rows, list) or len(qa_rows) != len(source["rows"]):
            raise ValueError(f"{fixture_id} human visual-QA row count mismatch")
        source_by_row = {row["row_id"]: row for row in source["rows"]}
        seen: set[str] = set()
        for index, value in enumerate(qa_rows):
            row = _object(value, f"human visual-QA {fixture_id}.rows[{index}]")
            _exact(row, _QA_ROW_KEYS, f"human visual-QA {fixture_id}.rows[{index}]")
            row_id = _text(row["row_id"], f"human visual-QA {fixture_id}.rows[{index}].id")
            if row_id in seen or row_id not in source_by_row:
                raise ValueError(f"{fixture_id} human visual-QA row identity mismatch")
            seen.add(row_id)
            source_row = source_by_row[row_id]
            if (
                row["image_sha256"] != source_row["image_sha256"]
                or row["caption_sha256"] != source_row["caption_sha256"]
            ):
                raise ValueError(
                    f"{fixture_id}.{row_id} human visual-QA byte binding mismatch"
                )
            checks = _object(
                row["checks"], f"human visual-QA {fixture_id}.{row_id}.checks"
            )
            _exact(
                checks,
                _QA_CHECK_KEYS,
                f"human visual-QA {fixture_id}.{row_id}.checks",
            )
            expected_checks = {key: "PASS" for key in _QA_CHECK_KEYS}
            expected_checks["product_identity_preserved"] = (
                True if fixture_id.startswith("W6-PRODUCT") else None
            )
            if checks != expected_checks:
                raise ValueError(
                    f"{fixture_id}.{row_id} human visual-QA checks are incomplete or failing"
                )
        if seen != set(source_by_row):
            raise ValueError(f"{fixture_id} human visual-QA omits source rows")
    return {
        "receipt_sha256": receipt_sha,
        "generation_plan_sha256": generation_plan_sha256,
        "reviewer": reviewer,
        "reviewed_at_utc": reviewed_at,
    }


def _deterministic_shuffle(items: Sequence[int], seed: bytes) -> list[int]:
    """Fisher-Yates using a versioned HMAC stream, independent of Python RNG."""

    result = list(items)
    counter = 0
    for index in range(len(result) - 1, 0, -1):
        block = hmac.new(
            seed,
            b"lane-b-shuffle-v1\0" + counter.to_bytes(8, "big"),
            hashlib.sha256,
        ).digest()
        swap = int.from_bytes(block, "big") % (index + 1)
        result[index], result[swap] = result[swap], result[index]
        counter += 1
    return result


def _split_rows(
    fixture_id: str,
    manifest_sha256: str,
    rows: Sequence[dict[str, Any]],
    custodian_key: bytearray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed = hmac.new(
        custodian_key,
        fixture_id.encode("utf-8") + manifest_sha256.encode("ascii"),
        hashlib.sha256,
    ).digest()
    order = _deterministic_shuffle(range(len(rows)), seed)
    holdout_count = (len(rows) + 9) // 10
    holdout_indices = set(order[:holdout_count])
    train = [dict(row) for index, row in enumerate(rows) if index not in holdout_indices]
    holdout = [dict(row) for index, row in enumerate(rows) if index in holdout_indices]
    return train, holdout


def _manifest_record(
    *,
    source: dict[str, Any],
    source_sha256: str,
    split: str,
    rows: list[dict[str, Any]],
    export_semantic_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": _SCHEMA,
        "kind": _SPLIT_KIND,
        "fixture_id": source["fixture_id"],
        "family": source["family"],
        "subtype": source["subtype"],
        "generator_label": source["generator_label"],
        "split": split,
        "source_manifest_sha256": source_sha256,
        "split_algorithm": _SPLIT_ALGORITHM,
        "export_semantic_sha256": export_semantic_sha256,
        "row_count": len(rows),
        "rows": rows,
    }


def _write_exclusive_bytes(path: Path, payload: bytes) -> tuple[str, int]:
    """Create one independent byte copy, fsync it, then verify exact reread."""

    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset : offset + 1024 * 1024])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    copied = _read_regular(path, f"exported split byte {path.name}")
    if copied != payload:
        raise RuntimeError(f"exported split byte differs from validated source: {path}")
    return _sha256(copied), len(copied)


def _materialize_split_bytes(
    split_directory: Path,
    *,
    fixture_id: str,
    split: str,
    rows: Sequence[dict[str, Any]],
    byte_records: Mapping[str, dict[str, bytes]],
) -> tuple[list[dict[str, Any]], str]:
    """Publish a trainer-visible same-stem tree and bind its exact semantics."""

    fixture_directory = split_directory / fixture_id
    fixture_directory.mkdir(mode=0o700)
    exported_rows: list[dict[str, Any]] = []
    exported_files: list[dict[str, Any]] = []
    for row in rows:
        row_id = row["row_id"]
        source = byte_records[row_id]
        image_suffix = PurePosixPath(row["image_path"]).suffix.casefold()
        image_name = f"{row_id}{image_suffix}"
        caption_name = f"{row_id}.txt"
        image_path = fixture_directory / image_name
        caption_path = fixture_directory / caption_name
        image_sha, image_size = _write_exclusive_bytes(image_path, source["image"])
        caption_sha, caption_size = _write_exclusive_bytes(
            caption_path, source["caption"]
        )
        if image_sha != row["image_sha256"] or caption_sha != row["caption_sha256"]:
            raise RuntimeError(f"{fixture_id}/{row_id} exported split hash mismatch")
        exported = dict(row)
        exported["image_path"] = f"{fixture_id}/{image_name}"
        exported["caption_path"] = f"{fixture_id}/{caption_name}"
        exported_rows.append(exported)
        exported_files.append(
            {
                "row_id": row_id,
                "image_path": exported["image_path"],
                "image_sha256": image_sha,
                "image_bytes": image_size,
                "caption_path": exported["caption_path"],
                "caption_sha256": caption_sha,
                "caption_bytes": caption_size,
            }
        )
    _fsync_directory(fixture_directory)
    semantic = {
        "schema": _SCHEMA,
        "kind": _EXPORT_KIND,
        "fixture_id": fixture_id,
        "split": split,
        "files": exported_files,
    }
    return exported_rows, _sha256(canonical_bytes(semantic))


def _caption_stats(raw_captions: Iterable[bytes]) -> dict[str, int]:
    lengths = [len(raw.decode("utf-8").rstrip("\r\n")) for raw in raw_captions]
    return {
        "count": len(lengths),
        "min_unicode_codepoints": min(lengths),
        "max_unicode_codepoints": max(lengths),
        "sum_unicode_codepoints": sum(lengths),
    }


def _distribution(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _atomic_create(path: Path, value: dict[str, Any]) -> str:
    """Create one canonical record atomically and never replace a path."""

    payload = canonical_bytes(value) + b"\n"
    temporary = path.parent / f".{path.name}.partial-{os.getpid()}"
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {path}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_key(path: Path | str) -> bytearray:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"custodian key has a symlink ancestor: {current}")
        current = current.parent
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"custodian key cannot be opened safely: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("custodian key must be a regular file")
        if before.st_uid != os.geteuid():
            raise ValueError("custodian key must be owned by the effective UID")
        if stat.S_IMODE(before.st_mode) != 0o600:
            raise ValueError("custodian key mode must be exactly 0600")
        if before.st_nlink != 1:
            raise ValueError("custodian key must have exactly one hard link")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 4097)
            if not block:
                break
            chunks.append(block)
            if sum(len(item) for item in chunks) > 4096:
                break
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
        item.st_uid,
        item.st_mode,
        item.st_nlink,
    )
    if identity(before) != identity(after):
        raise RuntimeError("custodian key changed while read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise RuntimeError("custodian key changed size while read")
    if len(raw) < 32 or len(raw) > 4096:
        raise ValueError("custodian key must contain between 32 and 4096 bytes")
    return bytearray(raw)


def _reject_cross_fixture_duplicates(validated: Sequence[dict[str, Any]]) -> None:
    seen: dict[tuple[str, Any], str] = {}
    for item in validated:
        fixture_id = item["manifest"]["fixture_id"]
        for row, byte_record in zip(item["rows"], item["bytes"], strict=True):
            fields = {
                "image_sha256": row["image_sha256"],
                "caption_sha256": row["caption_sha256"],
                "output_sha256": row["output_sha256"],
                "provider/request_id": byte_record["provider_request"],
            }
            for label, value in fields.items():
                key = (label, value)
                previous = seen.get(key)
                if previous is not None and previous != fixture_id:
                    raise ValueError(
                        f"cross-fixture duplicate {label} between {previous} and "
                        f"{fixture_id}"
                    )
                seen[key] = fixture_id
    product_references = {
        item["manifest"]["fixture_id"]: item["rows"][0]["reference_sha256"]
        for item in validated
        if item["manifest"]["fixture_id"].startswith("W6-PRODUCT")
    }
    if (
        set(product_references) == {"W6-PRODUCT-A", "W6-PRODUCT-B"}
        and len(set(product_references.values())) != 2
    ):
        raise ValueError("product fixtures A and B must use distinct references")


def validate_admission_inputs(
    manifest_paths: Sequence[Path | str],
    fixture_root: Path | str,
    generation_plan_path: Path | str,
    qa_receipt_path: Path | str,
    *,
    social_relief: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Validate source bytes, plan binding, and private pre-split human QA."""

    expected_ids = _expected_fixture_ids(social_relief)
    if len(manifest_paths) != len(expected_ids):
        raise ValueError(
            f"exactly {len(expected_ids)} source manifests are required for "
            f"social_relief={social_relief}"
        )
    validated = [
        validate_source_manifest(manifest_path, fixture_root)
        for manifest_path in manifest_paths
    ]
    by_id = {item["manifest"]["fixture_id"]: item for item in validated}
    if len(by_id) != len(validated) or set(by_id) != expected_ids:
        raise ValueError(
            "source manifest fixture IDs must be exactly "
            + ", ".join(fixture for fixture in _FIXTURES if fixture in expected_ids)
        )
    _reject_cross_fixture_duplicates(validated)
    generation_plan_sha = _validate_generation_plan_binding(
        generation_plan_path, validated, social_relief=social_relief
    )
    qa = _validate_visual_qa_receipt(
        qa_receipt_path,
        validated,
        generation_plan_sha256=generation_plan_sha,
        social_relief=social_relief,
    )
    return validated, qa


def publish_package(
    manifest_paths: Sequence[Path | str],
    fixture_root: Path | str,
    custodian_key_path: Path | str,
    output_directory: Path | str,
    *,
    generation_plan_path: Path | str,
    qa_receipt_path: Path | str,
    social_relief: bool = False,
) -> dict[str, Any]:
    """Validate the declared fixture set and publish a create-only package."""

    expected_ids = _expected_fixture_ids(social_relief)
    validated, visual_qa = validate_admission_inputs(
        manifest_paths,
        fixture_root,
        generation_plan_path,
        qa_receipt_path,
        social_relief=social_relief,
    )
    by_id = {item["manifest"]["fixture_id"]: item for item in validated}

    output = Path(os.path.abspath(os.path.expanduser(output_directory)))
    parent = _safe_directory(output.parent, "output parent", create=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite package: {output}")
    key = _load_key(custodian_key_path)
    acceptance_rows: list[dict[str, Any]] = []
    output_created = False
    output_identity: tuple[int, int] | None = None
    try:
        # mkdir is the package-level create-only claim.  It is deliberately
        # inside the cleanup scope but guarded so a losing race cannot remove
        # a directory another process created.
        output.mkdir(mode=0o700)
        output_created = True
        created = output.lstat()
        output_identity = (created.st_dev, created.st_ino)
        train_directory = output / "train"
        holdout_directory = output / "holdout"
        train_directory.mkdir(mode=0o700)
        holdout_directory.mkdir(mode=0o700)
        for fixture_id in _FIXTURES:
            if fixture_id not in expected_ids:
                continue
            item = by_id[fixture_id]
            source = item["manifest"]
            source_sha = item["manifest_sha256"]
            train_rows, holdout_rows = _split_rows(
                fixture_id, source_sha, item["rows"], key
            )
            expected_holdout = (source["total_pairs"] + 9) // 10
            if (
                len(holdout_rows) != expected_holdout
                or len(train_rows) + len(holdout_rows) != source["total_pairs"]
            ):
                raise AssertionError("internal split cardinality mismatch")
            byte_records = {
                row["row_id"]: byte_record
                for row, byte_record in zip(
                    item["rows"], item["bytes"], strict=True
                )
            }
            train_rows, train_export_sha = _materialize_split_bytes(
                train_directory,
                fixture_id=fixture_id,
                split="train",
                rows=train_rows,
                byte_records=byte_records,
            )
            holdout_rows, holdout_export_sha = _materialize_split_bytes(
                holdout_directory,
                fixture_id=fixture_id,
                split="holdout",
                rows=holdout_rows,
                byte_records=byte_records,
            )
            train_record = _manifest_record(
                source=source,
                source_sha256=source_sha,
                split="train",
                rows=train_rows,
                export_semantic_sha256=train_export_sha,
            )
            holdout_record = _manifest_record(
                source=source,
                source_sha256=source_sha,
                split="holdout",
                rows=holdout_rows,
                export_semantic_sha256=holdout_export_sha,
            )
            train_sha = _atomic_create(
                train_directory / f"{fixture_id}.json", train_record
            )
            holdout_sha = _atomic_create(
                holdout_directory / f"{fixture_id}.json", holdout_record
            )
            acceptance_rows.append(
                {
                    "fixture_id": fixture_id,
                    "family": source["family"],
                    "subtype": source["subtype"],
                    "generator_label": source["generator_label"],
                    "total_count": source["total_pairs"],
                    "train_count": len(train_rows),
                    "holdout_count": len(holdout_rows),
                    "source_manifest_sha256": source_sha,
                    "train_manifest_sha256": train_sha,
                    "holdout_manifest_sha256": holdout_sha,
                    "train_export_semantic_sha256": train_export_sha,
                    "holdout_export_semantic_sha256": holdout_export_sha,
                    "caption_length_statistics": _caption_stats(
                        record["caption"] for record in item["bytes"]
                    ),
                    "aspect_ratio_distribution": _distribution(
                        row["aspect_ratio"] for row in item["rows"]
                    ),
                    "rights_status_distribution": _distribution(
                        row["rights_status"] for row in item["rows"]
                    ),
                }
            )
        _fsync_directory(train_directory)
        _fsync_directory(holdout_directory)
        acceptance = {
            "schema": _SCHEMA,
            "kind": _ACCEPTANCE_KIND,
            "split_algorithm": _SPLIT_ALGORITHM,
            "fixture_count": len(acceptance_rows),
            "social_relief": social_relief,
            "fixtures": acceptance_rows,
            "blinding": {
                "reveals_holdout_filenames": False,
                "reveals_holdout_captions": False,
                "reveals_holdout_pixels": False,
                "custodian_key_persisted": False,
            },
            "visual_qa": {
                "state": "pre-split-human-review-pass",
                "generation_plan_sha256": visual_qa["generation_plan_sha256"],
                "receipt_sha256": visual_qa["receipt_sha256"],
                "reviewer": visual_qa["reviewer"],
                "reviewed_at_utc": visual_qa["reviewed_at_utc"],
            },
            "acceptance_state": "blinded-package-valid",
        }
        # This terminal marker is always written last.  Consumers must reject
        # a package directory that lacks it.
        _atomic_create(output / "blinded-acceptance.json", acceptance)
        _fsync_directory(output)
        _fsync_directory(parent)
        return acceptance
    except BaseException:
        # A partial directory is deliberately removed before the terminal
        # acceptance marker exists.  Existing output directories are never
        # reused or overwritten.
        if output_created and output_identity is not None:
            try:
                current = output.lstat()
            except FileNotFoundError:
                current = None
            if (
                current is not None
                and stat.S_ISDIR(current.st_mode)
                and not output.is_symlink()
                and (current.st_dev, current.st_ino) == output_identity
            ):
                shutil.rmtree(output, ignore_errors=True)
        raise
    finally:
        for index in range(len(key)):
            key[index] = 0


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate all source bytes")
    publish = subparsers.add_parser("publish", help="split and publish the package")
    for command in (validate, publish):
        command.add_argument("--fixture-root", required=True, type=Path)
        command.add_argument("--manifest", required=True, action="append", type=Path)
        command.add_argument("--generation-plan", required=True, type=Path)
        command.add_argument("--qa-receipt", required=True, type=Path)
        command.add_argument(
            "--social-relief",
            action="store_true",
            help="accept the predeclared five-fixture package without W6-SOCIAL-A",
        )
    publish.add_argument("--custodian-key", required=True, type=Path)
    publish.add_argument("--output-directory", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    if args.command == "validate":
        validate_admission_inputs(
            args.manifest,
            args.fixture_root,
            args.generation_plan,
            args.qa_receipt,
            social_relief=args.social_relief,
        )
        print("PASS")
        return 0
    publish_package(
        args.manifest,
        args.fixture_root,
        args.custodian_key,
        args.output_directory,
        generation_plan_path=args.generation_plan,
        qa_receipt_path=args.qa_receipt,
        social_relief=args.social_relief,
    )
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
