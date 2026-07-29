#!/usr/bin/env python3
"""Freeze AIC object and IIIF metadata for unapproved Krea D2 candidates.

This is a deliberately narrow, CPU-only pre-admission stage.  It consumes a
validated :mod:`krea_source_curation` D2 harvest and retrieves exactly two JSON
documents for every machine-eligible candidate: the selected AIC artwork
response and the corresponding IIIF ``info.json``.  It never retrieves image
bytes, creates captions or splits, records a rights/admission approval, or
authorizes GPU work.

The enrichment is useful because the search harvest is only a discovery view.
This stage freezes the selected-object view, the API licence/configuration
blocks, and the IIIF service geometry before a human curation decision is made.
All coverage and raw bytes are re-derived by :func:`validate_enrichment`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

try:
    from . import krea_provenance
    from . import krea_source_curation as source_curation
except ImportError:  # pragma: no cover - direct script execution.
    import krea_provenance  # type: ignore[no-redef]
    import krea_source_curation as source_curation  # type: ignore[no-redef]


_SCHEMA = 1
_KIND = "forge-krea-aic-selected-object-iiif-enrichment"
_STATE = "candidate_unreviewed_metadata_enrichment"
_CLAIM_LIMIT = (
    "selected-object-and-iiif-metadata-freeze-only-no-image-rights-caption-"
    "split-admission-or-gpu-approval"
)
_USER_AGENT = (
    "SN56-Krea-fixture-curation/1.0 "
    "(public research; https://github.com/tuly1/sn56-forge-toolkit)"
)
_API_ORIGIN = "https://api.artic.edu"
_API_ROOT = _API_ORIGIN + "/api/v1/artworks"
_IIIF_ROOT = "https://www.artic.edu/iiif/2"
_API_HOST = "api.artic.edu"
_IIIF_HOST = "www.artic.edu"
_IIIF_CONTEXT = "http://iiif.io/api/image/2/context.json"
_IIIF_PROTOCOL = "http://iiif.io/api/image"
_IIIF_PROFILE = "http://iiif.io/api/image/2/level1.json"
_DERIVATIVE_WIDTH = 1686
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MINIMUM_DELAY_S = 1.0
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IMAGE_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_SOURCE_ID = re.compile(r"aic-[1-9][0-9]*")
_API_FIELDS = (
    "id",
    "api_model",
    "api_link",
    "title",
    "alt_titles",
    "thumbnail",
    "main_reference_number",
    "date_start",
    "date_end",
    "date_display",
    "artist_id",
    "artist_title",
    "artist_display",
    "place_of_origin",
    "dimensions",
    "medium_display",
    "inscriptions",
    "credit_line",
    "publication_history",
    "provenance_text",
    "is_public_domain",
    "copyright_notice",
    "department_title",
    "classification_title",
    "classification_titles",
    "image_id",
    "alt_image_ids",
    "updated_at",
)
_EXPECTED_LICENSE_LINKS = [
    "https://creativecommons.org/publicdomain/zero/1.0/",
    "https://www.artic.edu/terms",
]
_HUMAN_GATES = {
    "caption_policy": "pending",
    "concept_suitability": "pending",
    "exhaustive_similarity": "pending",
    "group_identity": "pending",
    "independent_admission": "pending",
    "rights_acceptance": "pending",
    "visual_qc": "pending",
}


def _exact(value: dict[str, Any], keys: set[str], label: str) -> None:
    missing = keys - set(value)
    extra = set(value) - keys
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _canonical_utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value
    ):
        raise ValueError(f"{label} must be canonical UTC (YYYY-MM-DDTHH:MM:SSZ)")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{label} is not a real UTC timestamp") from exc
    now = datetime.now(timezone.utc)
    if parsed < datetime(2020, 1, 1, tzinfo=timezone.utc) or parsed > now + timedelta(
        seconds=60
    ):
        raise ValueError(f"{label} is outside the accepted evidence time bounds")
    return value


def _safe_existing_directory(path: Path, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real non-symlink directory")
    return path


def _safe_file(path: Path, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a non-empty regular non-symlink file")
    return path


def _new_directory(path: Path, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    if path.exists() or path.is_symlink():
        raise ValueError(f"{label} must not already exist")
    _safe_existing_directory(path.parent, f"{label} parent")
    path.mkdir(mode=0o750)
    return path


def _atomic_create(path: Path, payload: bytes, mode: int = 0o640) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to replace existing output: {path}")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short output write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_create(path: Path, value: Any) -> None:
    _atomic_create(path, krea_provenance.canonical_bytes(value) + b"\n")


def _read_canonical(path: Path, label: str) -> dict[str, Any]:
    raw = _safe_file(path, label).read_bytes()
    try:
        value = _object(json.loads(raw), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid JSON") from exc
    if raw != krea_provenance.canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline")
    return value


def _raw_identity(relative_path: str, payload: bytes) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _validate_request_url(url: str, *, host: str, query_allowed: bool) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or (parsed.query and not query_allowed)
    ):
        raise ValueError("AIC request URL is outside the exact HTTPS host contract")
    return url


def _artwork_url(object_id: int) -> str:
    _positive_int(object_id, "AIC object ID")
    query = urlencode({"fields": ",".join(_API_FIELDS)})
    return f"{_API_ROOT}/{object_id}?{query}"


def _iiif_info_url(image_id: str) -> str:
    if not isinstance(image_id, str) or not _IMAGE_ID.fullmatch(image_id):
        raise ValueError("AIC image ID must be a canonical lower-case UUID")
    return f"{_IIIF_ROOT}/{image_id}/info.json"


def _derivative_url(image_id: str) -> str:
    _iiif_info_url(image_id)
    return f"{_IIIF_ROOT}/{image_id}/full/{_DERIVATIVE_WIDTH},/0/default.jpg"


def _parse_raw_json(raw: bytes, label: str) -> dict[str, Any]:
    if not raw or len(raw) > _MAX_JSON_BYTES:
        raise ValueError(f"{label} is empty or exceeds the JSON byte cap")
    try:
        value = _object(json.loads(raw), label)
        # Reject NaN/Infinity accepted by Python's permissive JSON decoder.
        krea_provenance.canonical_bytes(value)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite JSON") from exc
    return value


def _fetch_json(url: str) -> bytes:
    request = Request(
        url, headers={"Accept": "application/json", "User-Agent": _USER_AGENT}
    )
    with urlopen(request, timeout=60) as response:  # noqa: S310 - exact hosts checked.
        if response.geturl() != url:
            raise ValueError("AIC metadata request redirected away from its exact URL")
        content_type = response.headers.get_content_type()
        if content_type not in {"application/json", "application/ld+json"}:
            raise ValueError(f"AIC metadata returned unexpected MIME {content_type}")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > _MAX_JSON_BYTES:
            raise ValueError("AIC metadata exceeds the JSON byte cap")
        raw = response.read(_MAX_JSON_BYTES + 1)
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError("AIC metadata exceeds the JSON byte cap")
    return raw


def _api_blocks(payload: dict[str, Any], label: str) -> dict[str, Any]:
    info = _object(payload.get("info"), f"{label}.info")
    config = _object(payload.get("config"), f"{label}.config")
    _exact(info, {"license_text", "license_links", "version"}, f"{label}.info")
    _exact(config, {"iiif_url", "website_url"}, f"{label}.config")
    license_text = info.get("license_text")
    license_links = info.get("license_links")
    version = info.get("version")
    if (
        not isinstance(license_text, str)
        or "description" not in license_text
        or "Creative Commons Zero (CC0) 1.0" not in license_text
        or license_links != _EXPECTED_LICENSE_LINKS
        or not isinstance(version, str)
        or not version.strip()
        or config.get("iiif_url") != _IIIF_ROOT
        or config.get("website_url") != "http://www.artic.edu"
    ):
        raise ValueError(f"{label} has an unexpected API licence/configuration block")
    return {
        "info": info,
        "config": config,
        "info_sha256": krea_provenance.canonical_sha256(info),
        "config_sha256": krea_provenance.canonical_sha256(config),
        "license_block_sha256": krea_provenance.canonical_sha256(
            {
                "license_text": license_text,
                "license_links": license_links,
                "version": version,
            }
        ),
    }


def _load_d2_harvest(
    harvest_dir: Path,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    """Return validated eligible rows, their exact search objects, and API blocks."""

    harvest_dir = _safe_existing_directory(harvest_dir, "D2 harvest package")
    harvest = source_curation.validate_harvest(harvest_dir)
    if (
        harvest.get("experimental_role") != "D2"
        or harvest.get("concept_id") != "tsukioka-kogyo-nogaku-zue"
        or harvest.get("admission_state") != "candidate_unreviewed"
        or harvest.get("gpu_execution_authorized") is not False
    ):
        raise ValueError("enrichment requires an intact unapproved D2 harvest")

    raw_records = harvest.get("raw_responses")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("D2 harvest has no frozen search pages")
    objects: dict[int, dict[str, Any]] = {}
    first_blocks: dict[str, Any] | None = None
    total: int | None = None
    total_pages = len(raw_records)
    observed = 0
    for page_index, record in enumerate(raw_records, start=1):
        record = _object(record, "D2 raw-response identity")
        name = f"source-response-{page_index:03d}.json"
        if record.get("name") != name:
            raise ValueError("D2 source pages are not contiguous")
        raw = _safe_file(harvest_dir / name, "D2 source page").read_bytes()
        payload = _parse_raw_json(raw, f"D2 source page {page_index}")
        _exact(
            payload,
            {"preference", "pagination", "data", "info", "config"},
            f"D2 source page {page_index}",
        )
        pagination = _object(payload["pagination"], "D2 pagination")
        _exact(
            pagination,
            {"total", "limit", "offset", "total_pages", "current_page"},
            "D2 pagination",
        )
        page_total = _positive_int(pagination.get("total"), "D2 pagination.total")
        limit = _positive_int(pagination.get("limit"), "D2 pagination.limit")
        offset = pagination.get("offset")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("D2 pagination.offset must be a non-negative integer")
        if (
            limit != 100
            or offset != (page_index - 1) * limit
            or pagination.get("total_pages") != total_pages
            or pagination.get("current_page") != page_index
            or total_pages != math.ceil(page_total / limit)
        ):
            raise ValueError("D2 pagination/continuation coverage is incoherent")
        if total is None:
            total = page_total
        elif total != page_total:
            raise ValueError("D2 pagination total changed across frozen pages")
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("D2 source page data must be an array")
        expected_page_count = min(limit, max(0, page_total - offset))
        if len(data) != expected_page_count:
            raise ValueError(
                "D2 source page does not provide complete pagination coverage"
            )
        blocks = _api_blocks(payload, f"D2 source page {page_index}")
        if first_blocks is None:
            first_blocks = blocks
        elif blocks != first_blocks:
            raise ValueError(
                "AIC licence/configuration blocks changed across harvest pages"
            )
        for item_index, raw_item in enumerate(data):
            item = _object(raw_item, f"D2 source object {page_index}:{item_index}")
            # The search endpoint adds a null Elasticsearch score even when an
            # exact ``fields`` projection is requested.  It is search transport
            # metadata, not part of the selected-artwork endpoint's data body.
            _exact(item, set(_API_FIELDS) | {"_score"}, "D2 source object")
            if item["_score"] is not None:
                raise ValueError("D2 exact-filter search unexpectedly returned a score")
            object_id = _positive_int(item.get("id"), "D2 source object.id")
            if object_id in objects:
                raise ValueError("D2 source pages contain a duplicate object ID")
            objects[object_id] = {field: item[field] for field in _API_FIELDS}
        observed += len(data)
    if total is None or observed != total or len(objects) != total:
        raise ValueError("D2 source search coverage is incomplete or duplicated")

    harvest_rows = harvest.get("candidate_rows")
    if not isinstance(harvest_rows, list) or len(harvest_rows) != len(objects):
        raise ValueError("D2 normalized rows do not cover every frozen source object")
    row_ids = [
        row.get("provider_object_id") for row in harvest_rows if isinstance(row, dict)
    ]
    if len(row_ids) != len(objects) or set(row_ids) != set(objects):
        raise ValueError("D2 normalized/object coverage differs from the frozen search")
    eligible: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in harvest_rows:
        row = _object(row, "D2 normalized candidate row")
        if row.get("eligibility") == {"passed": True, "reasons": []}:
            object_id = _positive_int(
                row.get("provider_object_id"), "provider_object_id"
            )
            eligible.append((row, objects[object_id]))
    eligible.sort(key=lambda pair: pair[0]["source_id"])
    if not eligible or first_blocks is None:
        raise ValueError("D2 harvest has no eligible objects or API blocks")
    source_ids = [row["source_id"] for row, _ in eligible]
    image_ids = [row["provider_image_id"] for row, _ in eligible]
    if (
        any(
            not isinstance(value, str) or not _SOURCE_ID.fullmatch(value)
            for value in source_ids
        )
        or len(source_ids) != len(set(source_ids))
        or any(
            not isinstance(value, str) or not _IMAGE_ID.fullmatch(value)
            for value in image_ids
        )
        or len(image_ids) != len(set(image_ids))
    ):
        raise ValueError(
            "eligible D2 source/image identities are invalid or duplicated"
        )
    return harvest, eligible, first_blocks


def _validate_artwork_payload(
    payload: dict[str, Any],
    *,
    expected_object: dict[str, Any],
    expected_blocks: dict[str, Any],
) -> dict[str, Any]:
    _exact(payload, {"data", "info", "config"}, "selected artwork response")
    data = _object(payload.get("data"), "selected artwork response.data")
    _exact(data, set(_API_FIELDS), "selected artwork response.data")
    if data != expected_object:
        raise ValueError(
            "selected artwork metadata differs from the frozen search harvest"
        )
    if data.get("is_public_domain") is not True:
        raise ValueError("selected artwork is no longer exactly public-domain")
    if _api_blocks(payload, "selected artwork response") != expected_blocks:
        raise ValueError("selected artwork API blocks differ from the frozen harvest")
    return data


def _validate_iiif_payload(
    payload: dict[str, Any],
    *,
    row: dict[str, Any],
) -> dict[str, Any]:
    _exact(
        payload,
        {"@context", "@id", "protocol", "width", "height", "sizes", "tiles", "profile"},
        "IIIF info.json",
    )
    image_id = row["provider_image_id"]
    expected_id = f"{_IIIF_ROOT}/{image_id}"
    width = _positive_int(payload.get("width"), "IIIF width")
    height = _positive_int(payload.get("height"), "IIIF height")
    if (
        payload.get("@context") != _IIIF_CONTEXT
        or payload.get("@id") != expected_id
        or payload.get("protocol") != _IIIF_PROTOCOL
        or width != row.get("native_width")
        or height != row.get("native_height")
        or width < _DERIVATIVE_WIDTH
        or row.get("download_url") != _derivative_url(image_id)
    ):
        raise ValueError(
            "IIIF service identity/geometry violates the frozen D2 contract"
        )

    sizes = payload.get("sizes")
    if not isinstance(sizes, list) or not sizes:
        raise ValueError("IIIF sizes must be a non-empty array")
    normalized_sizes: list[tuple[int, int]] = []
    for size in sizes:
        size = _object(size, "IIIF size")
        _exact(size, {"width", "height"}, "IIIF size")
        size_width = _positive_int(size.get("width"), "IIIF size.width")
        size_height = _positive_int(size.get("height"), "IIIF size.height")
        if size_width > width or size_height > height:
            raise ValueError("IIIF advertised size exceeds native geometry")
        normalized_sizes.append((size_width, size_height))
    if (
        len(normalized_sizes) != len(set(normalized_sizes))
        or (width, height) not in normalized_sizes
    ):
        raise ValueError("IIIF sizes are duplicated or omit native geometry")

    tiles = payload.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        raise ValueError("IIIF tiles must be a non-empty array")
    for tile in tiles:
        tile = _object(tile, "IIIF tile")
        _exact(tile, {"width", "height", "scaleFactors"}, "IIIF tile")
        _positive_int(tile.get("width"), "IIIF tile.width")
        _positive_int(tile.get("height"), "IIIF tile.height")
        factors = tile.get("scaleFactors")
        if (
            not isinstance(factors, list)
            or not factors
            or any(
                isinstance(factor, bool) or not isinstance(factor, int) or factor <= 0
                for factor in factors
            )
            or len(factors) != len(set(factors))
        ):
            raise ValueError("IIIF tile scaleFactors are invalid or duplicated")

    profile = payload.get("profile")
    if (
        not isinstance(profile, list)
        or len(profile) != 2
        or profile[0] != _IIIF_PROFILE
    ):
        raise ValueError("IIIF profile is not the expected Image API 2 level-1 profile")
    capabilities = _object(profile[1], "IIIF profile capabilities")
    _exact(
        capabilities,
        {"formats", "maxArea", "qualities", "supports"},
        "IIIF profile capabilities",
    )
    formats = capabilities.get("formats")
    qualities = capabilities.get("qualities")
    supports = capabilities.get("supports")
    if (
        not isinstance(formats, list)
        or "jpg" not in formats
        or not isinstance(qualities, list)
        or "default" not in qualities
        or not isinstance(supports, list)
        or "sizeByW" not in supports
        or isinstance(capabilities.get("maxArea"), bool)
        or not isinstance(capabilities.get("maxArea"), int)
        or capabilities["maxArea"] < width * height
    ):
        raise ValueError(
            "IIIF service cannot guarantee the exact 1686px JPEG derivative"
        )
    return payload


def _response_record(
    *,
    sequence: int,
    url: str,
    relative_path: str,
    raw: bytes,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "method": "GET",
        "url": url,
        "raw": _raw_identity(relative_path, raw),
        "payload_sha256": krea_provenance.canonical_sha256(payload),
    }


def enrich(
    harvest_dir: Path,
    output_dir: Path,
    *,
    retrieved_at_utc: str,
    delay_s: float = _MINIMUM_DELAY_S,
    fetcher: Callable[[str], bytes] = _fetch_json,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Create a new immutable D2 enrichment namespace.

    ``fetcher`` and ``sleeper`` are dependency-injection seams for offline tests;
    production callers should leave both at their defaults.
    """

    if (
        isinstance(delay_s, bool)
        or not isinstance(delay_s, (int, float))
        or not math.isfinite(float(delay_s))
        or delay_s < _MINIMUM_DELAY_S
    ):
        raise ValueError("AIC metadata request delay must be finite and at least 1.0s")
    retrieved_at_utc = _canonical_utc(retrieved_at_utc, "retrieved_at_utc")
    harvest, eligible, api_blocks = _load_d2_harvest(harvest_dir)
    output_dir = _new_directory(output_dir, "AIC enrichment output")
    responses_dir = output_dir / "responses"
    responses_dir.mkdir(mode=0o750)

    rows: list[dict[str, Any]] = []
    sequence = 0

    def fetch(url: str, *, host: str, query_allowed: bool) -> bytes:
        nonlocal sequence
        _validate_request_url(url, host=host, query_allowed=query_allowed)
        if sequence:
            sleeper(float(delay_s))
        sequence += 1
        raw = fetcher(url)
        if not isinstance(raw, bytes):
            raise ValueError("metadata fetcher must return bytes")
        return raw

    for index, (row, expected_object) in enumerate(eligible, start=1):
        source_id = row["source_id"]
        object_id = row["provider_object_id"]
        image_id = row["provider_image_id"]

        artwork_url = _artwork_url(object_id)
        artwork_raw = fetch(artwork_url, host=_API_HOST, query_allowed=True)
        artwork_payload = _parse_raw_json(artwork_raw, f"selected artwork {source_id}")
        artwork_data = _validate_artwork_payload(
            artwork_payload,
            expected_object=expected_object,
            expected_blocks=api_blocks,
        )
        artwork_name = f"artwork-{index:06d}-{source_id}.json"
        artwork_relative = "responses/" + artwork_name
        _atomic_create(responses_dir / artwork_name, artwork_raw)
        artwork_record = _response_record(
            sequence=sequence,
            url=artwork_url,
            relative_path=artwork_relative,
            raw=artwork_raw,
            payload=artwork_payload,
        )
        artwork_record["data_sha256"] = krea_provenance.canonical_sha256(artwork_data)

        info_url = _iiif_info_url(image_id)
        info_raw = fetch(info_url, host=_IIIF_HOST, query_allowed=False)
        info_payload = _parse_raw_json(info_raw, f"IIIF info {source_id}")
        _validate_iiif_payload(info_payload, row=row)
        info_name = f"iiif-{index:06d}-{source_id}.json"
        info_relative = "responses/" + info_name
        _atomic_create(responses_dir / info_name, info_raw)
        info_record = _response_record(
            sequence=sequence,
            url=info_url,
            relative_path=info_relative,
            raw=info_raw,
            payload=info_payload,
        )

        rows.append(
            {
                "source_id": source_id,
                "provider_object_id": object_id,
                "provider_image_id": image_id,
                "artwork_response": artwork_record,
                "iiif_info_response": info_record,
                "derivative_url": _derivative_url(image_id),
            }
        )

    source_ids = [row["source_id"] for row in rows]
    body = {
        "schema": _SCHEMA,
        "kind": _KIND,
        "enrichment_state": _STATE,
        "experimental_role": "D2",
        "concept_id": harvest["concept_id"],
        "retrieved_at_utc": retrieved_at_utc,
        "source_harvest_sha256": harvest["harvest_sha256"],
        "source_normalizer_contract_sha256": harvest[
            "normalizer_contract_sha256"
        ],
        "source_policy_sha256": harvest["source_policy_sha256"],
        "enrichment_tool_sha256": krea_provenance.file_sha256(
            Path(__file__).resolve(strict=True)
        ),
        "request_policy": {
            "sequential": True,
            "minimum_delay_s": float(delay_s),
            "request_count": sequence,
            "artwork_api_host": _API_HOST,
            "iiif_api_host": _IIIF_HOST,
            "artwork_fields": list(_API_FIELDS),
            "maximum_json_bytes": _MAX_JSON_BYTES,
        },
        "api_blocks": api_blocks,
        "derivative_policy": {
            "iiif_base_url": _IIIF_ROOT,
            "region": "full",
            "width": _DERIVATIVE_WIDTH,
            "height": "preserve-aspect-ratio",
            "rotation": 0,
            "quality": "default",
            "format": "jpg",
            "image_bytes_retrieved": False,
        },
        "coverage": {
            "eligible_count": len(source_ids),
            "enriched_count": len(source_ids),
            "eligible_source_ids_sha256": krea_provenance.canonical_sha256(source_ids),
            "missing_source_ids": [],
            "unexpected_source_ids": [],
            "duplicate_source_ids": [],
            "duplicate_image_ids": [],
        },
        "rows": rows,
        "claim_limit": _CLAIM_LIMIT,
        "human_gates": _HUMAN_GATES,
        "human_approvals": [],
        "rights_approved": False,
        "captions_created": False,
        "splits_created": False,
        "fixture_manifest_created": False,
        "jpeg_derivatives_downloaded": False,
        "gpu_execution_authorized": False,
    }
    manifest = {
        **body,
        "enrichment_sha256": krea_provenance.canonical_sha256(body),
    }
    _canonical_create(output_dir / "enrichment.json", manifest)
    validate_enrichment(harvest_dir, output_dir)
    return manifest


def _validate_raw_identity(
    enrichment_dir: Path,
    record: dict[str, Any],
    *,
    expected_relative_path: str,
    label: str,
) -> tuple[bytes, dict[str, Any]]:
    _exact(
        record,
        {"sequence", "method", "url", "raw", "payload_sha256"}
        | ({"data_sha256"} if label == "selected artwork" else set()),
        f"{label} response record",
    )
    raw_identity = _object(record.get("raw"), f"{label} raw identity")
    _exact(raw_identity, {"relative_path", "bytes", "sha256"}, f"{label} raw identity")
    if raw_identity.get("relative_path") != expected_relative_path:
        raise ValueError(f"{label} relative path is not canonical")
    path = _safe_file(enrichment_dir / expected_relative_path, f"{label} raw response")
    raw = path.read_bytes()
    if (
        isinstance(raw_identity.get("bytes"), bool)
        or raw_identity.get("bytes") != len(raw)
        or not isinstance(raw_identity.get("sha256"), str)
        or not _SHA256.fullmatch(raw_identity["sha256"])
        or raw_identity["sha256"] != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError(f"{label} raw-response identity mismatch")
    payload = _parse_raw_json(raw, label)
    if record.get("payload_sha256") != krea_provenance.canonical_sha256(payload):
        raise ValueError(f"{label} canonical payload hash mismatch")
    return raw, payload


def validate_enrichment(harvest_dir: Path, enrichment_dir: Path) -> dict[str, Any]:
    """Re-derive and validate an immutable unapproved AIC enrichment package."""

    harvest, eligible, api_blocks = _load_d2_harvest(harvest_dir)
    enrichment_dir = _safe_existing_directory(enrichment_dir, "AIC enrichment package")
    manifest = _read_canonical(
        enrichment_dir / "enrichment.json", "AIC enrichment manifest"
    )
    expected_keys = {
        "schema",
        "kind",
        "enrichment_state",
        "experimental_role",
        "concept_id",
        "retrieved_at_utc",
        "source_harvest_sha256",
        "source_normalizer_contract_sha256",
        "source_policy_sha256",
        "enrichment_tool_sha256",
        "request_policy",
        "api_blocks",
        "derivative_policy",
        "coverage",
        "rows",
        "claim_limit",
        "human_gates",
        "human_approvals",
        "rights_approved",
        "captions_created",
        "splits_created",
        "fixture_manifest_created",
        "jpeg_derivatives_downloaded",
        "gpu_execution_authorized",
        "enrichment_sha256",
    }
    _exact(manifest, expected_keys, "AIC enrichment manifest")
    body = {key: value for key, value in manifest.items() if key != "enrichment_sha256"}
    false_flags = {
        "rights_approved",
        "captions_created",
        "splits_created",
        "fixture_manifest_created",
        "jpeg_derivatives_downloaded",
        "gpu_execution_authorized",
    }
    if (
        manifest.get("schema") != _SCHEMA
        or manifest.get("kind") != _KIND
        or manifest.get("enrichment_state") != _STATE
        or manifest.get("experimental_role") != "D2"
        or manifest.get("concept_id") != harvest["concept_id"]
        or manifest.get("source_harvest_sha256") != harvest["harvest_sha256"]
        or manifest.get("source_normalizer_contract_sha256")
        != harvest["normalizer_contract_sha256"]
        or manifest.get("source_policy_sha256") != harvest["source_policy_sha256"]
        or manifest.get("enrichment_tool_sha256")
        != krea_provenance.file_sha256(Path(__file__).resolve(strict=True))
        or manifest.get("api_blocks") != api_blocks
        or manifest.get("claim_limit") != _CLAIM_LIMIT
        or manifest.get("human_gates") != _HUMAN_GATES
        or manifest.get("human_approvals") != []
        or any(manifest.get(flag) is not False for flag in false_flags)
        or manifest.get("enrichment_sha256") != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("enrichment is not an intact unapproved metadata-only record")
    _canonical_utc(manifest.get("retrieved_at_utc"), "retrieved_at_utc")

    request_policy = _object(manifest.get("request_policy"), "request_policy")
    _exact(
        request_policy,
        {
            "sequential",
            "minimum_delay_s",
            "request_count",
            "artwork_api_host",
            "iiif_api_host",
            "artwork_fields",
            "maximum_json_bytes",
        },
        "request_policy",
    )
    delay = request_policy.get("minimum_delay_s")
    if (
        request_policy.get("sequential") is not True
        or isinstance(delay, bool)
        or not isinstance(delay, (int, float))
        or not math.isfinite(float(delay))
        or delay < _MINIMUM_DELAY_S
        or request_policy.get("request_count") != 2 * len(eligible)
        or request_policy.get("artwork_api_host") != _API_HOST
        or request_policy.get("iiif_api_host") != _IIIF_HOST
        or request_policy.get("artwork_fields") != list(_API_FIELDS)
        or request_policy.get("maximum_json_bytes") != _MAX_JSON_BYTES
    ):
        raise ValueError("enrichment request policy violates the frozen contract")

    expected_derivative_policy = {
        "iiif_base_url": _IIIF_ROOT,
        "region": "full",
        "width": _DERIVATIVE_WIDTH,
        "height": "preserve-aspect-ratio",
        "rotation": 0,
        "quality": "default",
        "format": "jpg",
        "image_bytes_retrieved": False,
    }
    if manifest.get("derivative_policy") != expected_derivative_policy:
        raise ValueError(
            "enrichment derivative policy is not exact 1686px JPEG metadata-only"
        )

    expected_source_ids = [row["source_id"] for row, _ in eligible]
    expected_coverage = {
        "eligible_count": len(eligible),
        "enriched_count": len(eligible),
        "eligible_source_ids_sha256": krea_provenance.canonical_sha256(
            expected_source_ids
        ),
        "missing_source_ids": [],
        "unexpected_source_ids": [],
        "duplicate_source_ids": [],
        "duplicate_image_ids": [],
    }
    if manifest.get("coverage") != expected_coverage:
        raise ValueError("enrichment coverage is incomplete, duplicated, or unexpected")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != len(eligible):
        raise ValueError("enrichment rows do not exactly cover eligible candidates")

    expected_files = {"enrichment.json"}
    seen_source_ids: set[str] = set()
    seen_image_ids: set[str] = set()
    for index, (manifest_row, expected_pair) in enumerate(
        zip(rows, eligible, strict=True), start=1
    ):
        manifest_row = _object(manifest_row, "enrichment row")
        _exact(
            manifest_row,
            {
                "source_id",
                "provider_object_id",
                "provider_image_id",
                "artwork_response",
                "iiif_info_response",
                "derivative_url",
            },
            "enrichment row",
        )
        expected_row, expected_object = expected_pair
        source_id = expected_row["source_id"]
        object_id = expected_row["provider_object_id"]
        image_id = expected_row["provider_image_id"]
        if (
            manifest_row.get("source_id") != source_id
            or manifest_row.get("provider_object_id") != object_id
            or manifest_row.get("provider_image_id") != image_id
            or manifest_row.get("derivative_url") != _derivative_url(image_id)
            or source_id in seen_source_ids
            or image_id in seen_image_ids
        ):
            raise ValueError("enrichment row identity/order is invalid or duplicated")
        seen_source_ids.add(source_id)
        seen_image_ids.add(image_id)

        artwork_relative = f"responses/artwork-{index:06d}-{source_id}.json"
        artwork_record = _object(
            manifest_row.get("artwork_response"), "artwork_response"
        )
        _, artwork_payload = _validate_raw_identity(
            enrichment_dir,
            artwork_record,
            expected_relative_path=artwork_relative,
            label="selected artwork",
        )
        if (
            artwork_record.get("sequence") != 2 * index - 1
            or artwork_record.get("method") != "GET"
            or artwork_record.get("url") != _artwork_url(object_id)
        ):
            raise ValueError("selected artwork request sequence/URL is not exact")
        artwork_data = _validate_artwork_payload(
            artwork_payload,
            expected_object=expected_object,
            expected_blocks=api_blocks,
        )
        if artwork_record.get("data_sha256") != krea_provenance.canonical_sha256(
            artwork_data
        ):
            raise ValueError("selected artwork data hash mismatch")

        info_relative = f"responses/iiif-{index:06d}-{source_id}.json"
        info_record = _object(
            manifest_row.get("iiif_info_response"), "iiif_info_response"
        )
        _, info_payload = _validate_raw_identity(
            enrichment_dir,
            info_record,
            expected_relative_path=info_relative,
            label="IIIF info",
        )
        if (
            info_record.get("sequence") != 2 * index
            or info_record.get("method") != "GET"
            or info_record.get("url") != _iiif_info_url(image_id)
        ):
            raise ValueError("IIIF request sequence/URL is not exact")
        _validate_iiif_payload(info_payload, row=expected_row)
        expected_files.update({artwork_relative, info_relative})

    actual_files: set[str] = set()
    for path in enrichment_dir.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("enrichment namespace contains an unsafe filesystem entry")
        if path.is_file():
            actual_files.add(path.relative_to(enrichment_dir).as_posix())
    if actual_files != expected_files:
        raise ValueError("enrichment namespace has missing or unexpected files")
    return manifest


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("enrich")
    create.add_argument("--harvest", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--retrieved-at-utc", required=True)
    create.add_argument("--delay-s", type=float, default=_MINIMUM_DELAY_S)
    validate = commands.add_parser("validate")
    validate.add_argument("--harvest", required=True, type=Path)
    validate.add_argument("--enrichment", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    if args.command == "enrich":
        manifest = enrich(
            args.harvest,
            args.output,
            retrieved_at_utc=args.retrieved_at_utc,
            delay_s=args.delay_s,
        )
    else:
        manifest = validate_enrichment(args.harvest, args.enrichment)
    print(krea_provenance.canonical_bytes(manifest).decode("ascii"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
