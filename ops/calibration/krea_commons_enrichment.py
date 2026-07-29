#!/usr/bin/env python3
"""Freeze and validate pending D1 Wikimedia Commons enrichment evidence.

This module adds current page, revision, file, rights-metadata, category, and
wikitext identities to an already validated D1 candidate harvest.  Its only
state transition is from no enrichment record to an explicitly unapproved,
pending-review enrichment record.  It has no fixture-admission or execution
authorization operation.

The network collector writes the exact response bytes before deriving the
manifest.  :func:`validate_enrichment` independently re-derives every row from
those bytes and from the bound harvest package.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from . import krea_provenance
    from . import krea_source_curation as source_curation
except ImportError:  # pragma: no cover - direct script execution.
    import krea_provenance  # type: ignore[no-redef]
    import krea_source_curation as source_curation  # type: ignore[no-redef]


_SCHEMA = 1
_KIND = "forge-krea-commons-enrichment-unapproved"
_STATUS = "pending_named_human_rights_and_visual_review"
_ROLE = "D1"
_CONCEPT_ID = "fontana-del-moro"
_API = "https://commons.wikimedia.org/w/api.php"
_CATEGORY = "Category:Fontana del Moro (Rome)"
_BATCH_LIMIT = 50
_MAX_CONTINUATIONS_PER_BATCH = 100
_USER_AGENT = (
    "SN56-Krea-fixture-curation/1.0 "
    "(public research; https://github.com/tuly1/sn56-forge-toolkit)"
)
_CLAIM_LIMIT = (
    "metadata-consistency-only-rights-and-visual-review-remain-pending-no-"
    "fixture-or-execution-authorization"
)
_SHA1_HEX = re.compile(r"[0-9a-f]{40}")
_REVISION_SHA1 = re.compile(r"[0-9a-z]{31,40}")
_EXT_METADATA_FIELDS = (
    "License",
    "LicenseShortName",
    "UsageTerms",
    "LicenseUrl",
    "AttributionRequired",
    "Copyrighted",
    "Restrictions",
    "Artist",
    "Credit",
    "Permission",
    "ImageDescription",
    "DateTimeOriginal",
    "Categories",
)
_BASE_PARAMETERS = {
    "action": "query",
    "prop": "imageinfo|revisions|categories",
    "iiprop": "url|sha1|size|mime|mediatype|extmetadata",
    "rvprop": "ids|timestamp|sha1|content",
    "rvslots": "main",
    "cllimit": "max",
    "format": "json",
    "formatversion": "2",
}


def _by_urls(version: str) -> tuple[str, ...]:
    return tuple(
        f"{scheme}://creativecommons.org/licenses/by/{version}{suffix}"
        for scheme in ("http", "https")
        for suffix in ("", "/")
    )


_RIGHTS_MATRIX: dict[str, dict[str, Any]] = {
    "pd": {
        "short_name": "Public domain",
        "usage_terms": "Public domain",
        "attribution_required": "false",
        "copyrighted": "False",
        "license_urls": (None, ""),
        "category": None,
        "wikitext_tokens": (),
    },
    "cc0": {
        "short_name": "CC0",
        "usage_terms": "Creative Commons Zero, Public Domain Dedication",
        "attribution_required": "false",
        "copyrighted": "True",
        "license_urls": tuple(
            f"{scheme}://creativecommons.org/publicdomain/zero/1.0{suffix}"
            for scheme in ("http", "https")
            for suffix in ("", "/", "/deed.en")
        ),
        "category": "Category:CC-Zero",
        "wikitext_tokens": ("cc-zero", "cc0"),
    },
    **{
        f"cc-by-{version}": {
            "short_name": f"CC BY {version}",
            "usage_terms": f"Creative Commons Attribution {version}",
            "attribution_required": "true",
            "copyrighted": "True",
            "license_urls": _by_urls(version),
            "category": f"Category:CC-BY-{version}",
            "wikitext_tokens": (f"cc-by-{version}",),
        }
        for version in ("2.0", "2.5", "3.0", "4.0")
    },
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


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _metadata_value(extmetadata: dict[str, Any], field: str) -> str | None:
    item = extmetadata.get(field)
    if item is None:
        return None
    item = _object(item, f"extmetadata.{field}")
    value = item.get("value")
    if not isinstance(value, str):
        raise ValueError(f"extmetadata.{field}.value must be a string")
    return value


def _canonical_sha256(value: Any) -> str:
    return krea_provenance.canonical_sha256(value)


def _file_sha256(path: Path) -> str:
    return krea_provenance.file_sha256(path)


def _batches(values: list[int], size: int = _BATCH_LIMIT) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _request_parameters(
    page_ids: list[int], continuation: dict[str, str] | None = None
) -> dict[str, str]:
    if not page_ids or len(page_ids) > _BATCH_LIMIT:
        raise ValueError("Commons page-ID batch must contain between 1 and 50 IDs")
    if len(set(page_ids)) != len(page_ids) or page_ids != sorted(page_ids):
        raise ValueError("Commons page-ID batch must be unique and sorted")
    parameters = {**_BASE_PARAMETERS, "pageids": "|".join(map(str, page_ids))}
    if continuation is not None:
        if not continuation:
            raise ValueError("Commons continuation must not be empty")
        for key, value in continuation.items():
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(value, str)
                or not value
                or key in parameters
            ):
                raise ValueError("Commons continuation contains an invalid key/value")
        parameters.update(continuation)
    return parameters


def _request_url(parameters: dict[str, str]) -> str:
    return _API + "?" + urlencode(parameters)


def _fetch(request: Request, *, timeout_s: float = 60.0) -> bytes:
    with urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - fixed host
        final_url = response.geturl()
        if not final_url.startswith(_API + "?"):
            raise ValueError("Commons metadata request redirected off the fixed API")
        return response.read()


def _continuation(payload: dict[str, Any]) -> dict[str, str] | None:
    value = payload.get("continue")
    if value is None:
        return None
    value = _object(value, "Commons response.continue")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or not isinstance(item, str) or not item:
            raise ValueError("Commons response continuation is invalid")
        result[key] = item
    if not result:
        raise ValueError("Commons response continuation must not be empty")
    return result


def _eligible_rows(harvest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = harvest.get("candidate_rows")
    if not isinstance(rows, list):
        raise ValueError("harvest candidate_rows must be an array")
    eligible = [
        _object(row, "harvest candidate row")
        for row in rows
        if _object(row, "harvest candidate row").get("eligibility", {}).get("passed")
        is True
    ]
    if not eligible:
        raise ValueError("D1 harvest has no machine-eligible rows")
    eligible.sort(key=lambda row: row["provider_object_id"])
    page_ids = [row.get("provider_object_id") for row in eligible]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in page_ids):
        raise ValueError("eligible D1 rows must carry integer page IDs")
    if len(set(page_ids)) != len(page_ids):
        raise ValueError("eligible D1 harvest repeats a page ID")
    return eligible


def _harvest_file_index(
    harvest_dir: Path, harvest: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    raw_records = harvest["raw_responses"]
    if not isinstance(raw_records, list) or len(raw_records) != 1:
        raise ValueError("D1 harvest must bind exactly one category response")
    name = raw_records[0]["name"]
    raw_path = source_curation._safe_file(harvest_dir / name, "D1 raw harvest")
    payload = _object(json.loads(raw_path.read_bytes()), "D1 raw harvest response")
    pages = _object(payload.get("query"), "D1 raw harvest query").get("pages")
    if not isinstance(pages, list):
        raise ValueError("D1 raw harvest pages must be an array")
    result: dict[int, dict[str, Any]] = {}
    for raw_page in pages:
        page = _object(raw_page, "D1 raw harvest page")
        page_id = _positive_integer(page.get("pageid"), "D1 raw harvest page ID")
        infos = page.get("imageinfo")
        if not isinstance(infos, list) or len(infos) != 1:
            raise ValueError("D1 raw harvest page must bind one current file")
        info = _object(infos[0], "D1 raw harvest imageinfo")
        size = _positive_integer(info.get("size"), "D1 raw harvest file size")
        result[page_id] = {"size": size}
    if len(result) != len(pages):
        raise ValueError("D1 raw harvest repeats a page ID")
    return result


def _parse_categories(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    titles: list[str] = []
    for raw in value:
        item = _object(raw, label)
        if item.get("ns") != 14:
            raise ValueError(f"{label} contains a non-category namespace")
        title = item.get("title")
        if not isinstance(title, str) or not title.startswith("Category:"):
            raise ValueError(f"{label} contains an invalid category title")
        titles.append(title)
    if len(set(titles)) != len(titles):
        raise ValueError(f"{label} contains duplicate categories")
    return sorted(titles)


def _page_core(page: dict[str, Any]) -> dict[str, Any]:
    required = (
        "pageid",
        "ns",
        "title",
        "imagerepository",
        "revisions",
        "imageinfo",
    )
    missing = [key for key in required if key not in page]
    if missing:
        raise ValueError(f"Commons enrichment page lacks fields: {missing}")
    return {key: page[key] for key in required}


def _merge_batch_pages(
    payloads: list[dict[str, Any]], expected_page_ids: list[int]
) -> dict[int, dict[str, Any]]:
    expected = set(expected_page_ids)
    merged: dict[int, dict[str, Any]] = {}
    for payload in payloads:
        if "error" in payload:
            raise ValueError("Commons returned an API error")
        query = _object(payload.get("query"), "Commons enrichment response.query")
        pages = query.get("pages")
        if not isinstance(pages, list):
            raise ValueError("Commons enrichment pages must be an array")
        response_seen: set[int] = set()
        for raw_page in pages:
            page = _object(raw_page, "Commons enrichment page")
            page_id = _positive_integer(page.get("pageid"), "Commons page ID")
            if page_id not in expected or page_id in response_seen:
                raise ValueError(
                    "Commons response has an unexpected or duplicate page ID"
                )
            response_seen.add(page_id)
            categories = _parse_categories(
                page.get("categories"), f"Commons page {page_id} categories"
            )
            core = _page_core(page)
            if page_id not in merged:
                merged[page_id] = {**core, "categories": categories}
            else:
                if _page_core(merged[page_id]) != core:
                    raise ValueError(
                        "Commons page identity changed across continuation responses"
                    )
                merged[page_id]["categories"] = sorted(
                    set(merged[page_id]["categories"]).union(categories)
                )
    if set(merged) != expected:
        raise ValueError("Commons enrichment did not cover every requested page ID")
    return merged


def _wikitext_tokens(wikitext: str) -> set[str]:
    tokens: set[str] = set()
    for match in re.finditer(r"\{\{(.{1,4096}?)\}\}", wikitext, re.DOTALL):
        for part in match.group(1).split("|"):
            token = " ".join(part.strip().casefold().replace("_", " ").split())
            if token:
                tokens.add(token)
    return tokens


def _has_template_token(tokens: set[str], expected: tuple[str, ...]) -> bool:
    return any(
        token == name or token.startswith(name + " ")
        for token in tokens
        for name in expected
    )


def _recognized_pd_category(title: str) -> bool:
    return title.startswith("Category:PD-") or title in {
        "Category:Author died more than 100 years ago public domain images",
        "Category:CC-PD-Mark",
        "Category:CC-Zero",
        "Category:Public domain",
        "Category:Public domain files",
    }


def _recognized_pd_template(tokens: set[str]) -> bool:
    return any(
        token == "pd"
        or token.startswith("pd-")
        or token == "public domain"
        or token.startswith("public domain ")
        or token == "copyrighted free use"
        for token in tokens
    )


def _rights_record(
    extmetadata: dict[str, Any], categories: list[str], wikitext: str
) -> dict[str, Any]:
    frozen = {
        field: _metadata_value(extmetadata, field) for field in _EXT_METADATA_FIELDS
    }
    license_id = frozen["License"]
    matrix = _RIGHTS_MATRIX.get(license_id) if isinstance(license_id, str) else None
    if matrix is None:
        raise ValueError("Commons rights metadata uses a non-allowlisted license ID")
    expected = {
        "LicenseShortName": matrix["short_name"],
        "UsageTerms": matrix["usage_terms"],
        "AttributionRequired": matrix["attribution_required"],
        "Copyrighted": matrix["copyrighted"],
    }
    for field, value in expected.items():
        if frozen[field] != value:
            raise ValueError(f"Commons {license_id} rights tuple disagrees at {field}")
    if frozen["LicenseUrl"] not in matrix["license_urls"]:
        raise ValueError(f"Commons {license_id} uses a non-allowlisted LicenseUrl")
    if frozen["Restrictions"] != "":
        raise ValueError("Commons Restrictions must be present and empty")
    if not isinstance(frozen["Artist"], str) or not frozen["Artist"].strip():
        raise ValueError("Commons Artist metadata must be present")

    page_categories = set(categories)
    ext_category_value = frozen["Categories"]
    if not isinstance(ext_category_value, str) or not ext_category_value.strip():
        raise ValueError("Commons extmetadata Categories must be present")
    ext_categories = {
        "Category:" + item.strip()
        for item in ext_category_value.split("|")
        if item.strip()
    }
    if _CATEGORY not in page_categories or _CATEGORY not in ext_categories:
        raise ValueError("Commons page is not consistently bound to the D1 category")
    if not ext_categories.issubset(page_categories):
        raise ValueError("Commons extmetadata Categories disagree with page categories")

    template_tokens = _wikitext_tokens(wikitext)
    if license_id == "pd":
        if not any(_recognized_pd_category(item) for item in page_categories):
            raise ValueError("Commons public-domain row lacks a recognized PD category")
        if not _recognized_pd_template(template_tokens):
            raise ValueError("Commons public-domain row lacks a recognized PD template")
        rights_signal = "recognized_pd_template_and_category_pending_named_review"
    else:
        required_category = matrix["category"]
        # CommonsMetadata's Categories field intentionally omits some hidden
        # license categories.  The page-categories property includes them, so
        # the license signal is bound there and in the full wikitext; the
        # extmetadata category list is still frozen and checked as a subset.
        if required_category not in page_categories:
            raise ValueError(f"Commons {license_id} category signal is inconsistent")
        if not _has_template_token(template_tokens, matrix["wikitext_tokens"]):
            raise ValueError(f"Commons {license_id} wikitext signal is inconsistent")
        rights_signal = "exact_cc_tuple_template_and_category_pending_named_review"
    return {
        "fields": frozen,
        "page_categories": categories,
        "extmetadata_categories": sorted(ext_categories),
        "mechanical_signal": rights_signal,
        "review_state": "pending_named_human_review",
    }


def _derive_row(
    harvest_row: dict[str, Any], harvest_file: dict[str, Any], page: dict[str, Any]
) -> dict[str, Any]:
    source_id = harvest_row["source_id"]
    page_id = _positive_integer(page.get("pageid"), f"{source_id} page ID")
    if (
        page_id != harvest_row["provider_object_id"]
        or page.get("ns") != 6
        or page.get("title") != harvest_row["title"]
        or page.get("imagerepository") != "local"
    ):
        raise ValueError(f"{source_id} page identity disagrees with the harvest")

    revisions = page.get("revisions")
    if not isinstance(revisions, list) or len(revisions) != 1:
        raise ValueError(f"{source_id} must expose exactly one latest revision")
    revision = _object(revisions[0], f"{source_id} revision")
    revision_id = _positive_integer(revision.get("revid"), f"{source_id} revid")
    parent_id = revision.get("parentid")
    if isinstance(parent_id, bool) or not isinstance(parent_id, int) or parent_id < 0:
        raise ValueError(f"{source_id} revision parentid is invalid")
    revision_sha1 = revision.get("sha1")
    if not isinstance(revision_sha1, str) or not _REVISION_SHA1.fullmatch(
        revision_sha1
    ):
        raise ValueError(f"{source_id} revision sha1 is invalid")
    if (
        revision_id != harvest_row["revision_id"]
        or revision.get("timestamp") != harvest_row["revision_timestamp"]
    ):
        raise ValueError(f"{source_id} latest revision disagrees with the harvest")
    slots = _object(revision.get("slots"), f"{source_id} revision slots")
    main = _object(slots.get("main"), f"{source_id} main slot")
    wikitext = main.get("content")
    if (
        main.get("contentmodel") != "wikitext"
        or main.get("contentformat") != "text/x-wiki"
        or not isinstance(wikitext, str)
        or not wikitext.strip()
    ):
        raise ValueError(f"{source_id} does not expose full main-slot wikitext")

    infos = page.get("imageinfo")
    if not isinstance(infos, list) or len(infos) != 1:
        raise ValueError(f"{source_id} must expose exactly one current file")
    info = _object(infos[0], f"{source_id} imageinfo")
    file_sha1 = info.get("sha1")
    file_size = _positive_integer(info.get("size"), f"{source_id} file size")
    width = _positive_integer(info.get("width"), f"{source_id} width")
    height = _positive_integer(info.get("height"), f"{source_id} height")
    if (
        not isinstance(file_sha1, str)
        or not _SHA1_HEX.fullmatch(file_sha1)
        or file_sha1 != harvest_row["provider_content_sha1"]
        or info.get("url") != harvest_row["download_url"]
        or file_size != harvest_file["size"]
        or width != harvest_row["native_width"]
        or height != harvest_row["native_height"]
        or info.get("mime") != harvest_row["provider_mime"]
        or info.get("mime") != "image/jpeg"
        or info.get("mediatype") != "BITMAP"
        or width < 768
        or height < 768
    ):
        raise ValueError(
            f"{source_id} current JPEG identity disagrees with the harvest"
        )

    extmetadata = _object(info.get("extmetadata"), f"{source_id} extmetadata")
    categories = page["categories"]
    rights = _rights_record(extmetadata, categories, wikitext)
    fields = rights["fields"]
    license_url = fields["LicenseUrl"] or ""
    if (
        fields["LicenseShortName"] != harvest_row["license_name"]
        or license_url != harvest_row["license_url"]
        or fields["UsageTerms"] != harvest_row["usage_terms"]
        or source_curation._plain_metadata(extmetadata.get("Artist"))
        != harvest_row["creator"]
        or source_curation._plain_metadata(extmetadata.get("Credit"))
        != harvest_row["credit"]
    ):
        raise ValueError(
            f"{source_id} rights/creator metadata disagrees with the harvest"
        )
    creator_hint = source_curation._d1_creator_id(extmetadata.get("Artist"))
    if creator_hint != harvest_row["creator_id_hint"] or creator_hint == "text:unknown":
        raise ValueError(f"{source_id} creator hint is not stable across enrichment")

    return {
        "source_id": source_id,
        "provider_object_id": page_id,
        "page_title": page["title"],
        "source_page_url": harvest_row["source_page_url"],
        "latest_revision": {
            "revid": revision_id,
            "parentid": parent_id,
            "timestamp": revision["timestamp"],
            "sha1": revision_sha1,
            "main_slot_contentmodel": main["contentmodel"],
            "main_slot_contentformat": main["contentformat"],
            "main_slot_bytes": len(wikitext.encode("utf-8")),
            "main_slot_sha256": hashlib.sha256(wikitext.encode("utf-8")).hexdigest(),
        },
        "current_file": {
            "url": info["url"],
            "sha1": file_sha1,
            "size": file_size,
            "width": width,
            "height": height,
            "mime": info["mime"],
            "mediatype": info["mediatype"],
        },
        "rights_metadata": rights,
        "creator_id_hint": creator_hint,
        "consistency_state": "machine_consistent_human_review_pending",
    }


def _derive_rows(
    harvest_rows: list[dict[str, Any]],
    harvest_files: dict[int, dict[str, Any]],
    batch_payloads: list[tuple[list[int], list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    pages: dict[int, dict[str, Any]] = {}
    for page_ids, payloads in batch_payloads:
        for page_id, page in _merge_batch_pages(payloads, page_ids).items():
            if page_id in pages:
                raise ValueError("Commons enrichment repeats a page across batches")
            pages[page_id] = page
    expected_ids = {row["provider_object_id"] for row in harvest_rows}
    if set(pages) != expected_ids:
        raise ValueError(
            "Commons enrichment coverage differs from eligible harvest rows"
        )
    rows = [
        _derive_row(
            row,
            harvest_files[row["provider_object_id"]],
            pages[row["provider_object_id"]],
        )
        for row in harvest_rows
    ]
    rows.sort(key=lambda row: row["source_id"])
    return rows


def _raw_record(
    *,
    name: str,
    raw: bytes,
    batch_index: int,
    continuation_index: int,
    page_ids: list[int],
    parameters: dict[str, str],
) -> dict[str, Any]:
    return {
        "name": name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "batch_index": batch_index,
        "continuation_index": continuation_index,
        "page_ids": page_ids,
        "request_parameters": parameters,
    }


def _manifest_body(
    *,
    harvest: dict[str, Any],
    harvest_file_sha256: str,
    retrieved_at_utc: str,
    raw_responses: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_ids = sorted(row["source_id"] for row in rows)
    return {
        "schema": _SCHEMA,
        "kind": _KIND,
        "status": _STATUS,
        "experimental_role": _ROLE,
        "concept_id": _CONCEPT_ID,
        "retrieved_at_utc": source_curation._canonical_utc(
            retrieved_at_utc, "retrieved_at_utc"
        ),
        "source_policy_sha256": harvest["source_policy_sha256"],
        "enrichment_tool_sha256": _file_sha256(Path(__file__).resolve(strict=True)),
        "harvest_binding": {
            "harvest_sha256": harvest["harvest_sha256"],
            "harvest_file_sha256": harvest_file_sha256,
            "eligible_count": len(rows),
            "eligible_source_ids_sha256": _canonical_sha256(source_ids),
        },
        "request_contract": {
            "api": _API,
            "maximum_page_ids_per_batch": _BATCH_LIMIT,
            "base_parameters": _BASE_PARAMETERS,
            "continuation_policy": "follow_every_returned_continuation",
        },
        "raw_responses": raw_responses,
        "candidate_rows": rows,
        "counts": {"requested": len(rows), "mechanically_consistent": len(rows)},
        "claim_limit": _CLAIM_LIMIT,
        "human_review_state": "pending",
        "human_records": [],
        "fixture_manifest_created": False,
        "execution_authorized": False,
    }


def enrich_d1(
    harvest_dir: Path,
    output_dir: Path,
    *,
    retrieved_at_utc: str,
    fetch: Callable[[Request], bytes] | None = None,
) -> dict[str, Any]:
    """Fetch and freeze pending enrichment for every eligible D1 harvest row."""

    harvest_dir = source_curation._safe_existing_directory(
        harvest_dir, "D1 harvest package"
    )
    harvest = source_curation.validate_harvest(harvest_dir)
    if harvest["experimental_role"] != _ROLE or harvest["concept_id"] != _CONCEPT_ID:
        raise ValueError("enrichment requires the exact D1 Fontana del Moro harvest")
    harvest_rows = _eligible_rows(harvest)
    harvest_files = _harvest_file_index(harvest_dir, harvest)
    output_dir = source_curation._new_directory(output_dir, "D1 enrichment output")
    fetcher = fetch or _fetch
    raw_records: list[dict[str, Any]] = []
    batch_payloads: list[tuple[list[int], list[dict[str, Any]]]] = []
    response_number = 0
    for batch_index, page_ids in enumerate(
        _batches([row["provider_object_id"] for row in harvest_rows]), start=1
    ):
        continuation: dict[str, str] | None = None
        continuation_index = 0
        seen_continuations: set[str] = set()
        payloads: list[dict[str, Any]] = []
        while True:
            if continuation_index >= _MAX_CONTINUATIONS_PER_BATCH:
                raise ValueError("Commons continuation count exceeds the safety bound")
            parameters = _request_parameters(page_ids, continuation)
            raw = fetcher(
                Request(_request_url(parameters), headers={"User-Agent": _USER_AGENT})
            )
            if not isinstance(raw, bytes) or not raw:
                raise ValueError("Commons metadata fetch returned no bytes")
            response_number += 1
            name = f"source-response-{response_number:03d}.json"
            source_curation._atomic_create(output_dir / name, raw)
            try:
                payload = _object(json.loads(raw), "Commons enrichment response")
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Commons enrichment response is not JSON") from exc
            payloads.append(payload)
            raw_records.append(
                _raw_record(
                    name=name,
                    raw=raw,
                    batch_index=batch_index,
                    continuation_index=continuation_index,
                    page_ids=page_ids,
                    parameters=parameters,
                )
            )
            continuation = _continuation(payload)
            if continuation is None:
                break
            digest = _canonical_sha256(continuation)
            if digest in seen_continuations:
                raise ValueError("Commons repeated a continuation token")
            seen_continuations.add(digest)
            continuation_index += 1
        batch_payloads.append((page_ids, payloads))

    rows = _derive_rows(harvest_rows, harvest_files, batch_payloads)
    manifest_body = _manifest_body(
        harvest=harvest,
        harvest_file_sha256=_file_sha256(harvest_dir / "harvest.json"),
        retrieved_at_utc=retrieved_at_utc,
        raw_responses=raw_records,
        rows=rows,
    )
    manifest = {
        **manifest_body,
        "enrichment_sha256": _canonical_sha256(manifest_body),
    }
    source_curation._canonical_create(output_dir / "enrichment.json", manifest)
    validate_enrichment(harvest_dir, output_dir)
    return manifest


def _read_raw_responses(
    enrichment_dir: Path,
    raw_records: list[Any],
    expected_batches: list[list[int]],
) -> list[tuple[list[int], list[dict[str, Any]]]]:
    record_index = 0
    batch_payloads: list[tuple[list[int], list[dict[str, Any]]]] = []
    for batch_index, page_ids in enumerate(expected_batches, start=1):
        continuation: dict[str, str] | None = None
        continuation_index = 0
        seen_continuations: set[str] = set()
        payloads: list[dict[str, Any]] = []
        while True:
            if record_index >= len(raw_records):
                raise ValueError("enrichment has too few raw responses")
            raw_record = _object(raw_records[record_index], "raw response identity")
            _exact(
                raw_record,
                {
                    "name",
                    "bytes",
                    "sha256",
                    "batch_index",
                    "continuation_index",
                    "page_ids",
                    "request_parameters",
                },
                "raw response identity",
            )
            expected_name = f"source-response-{record_index + 1:03d}.json"
            expected_parameters = _request_parameters(page_ids, continuation)
            if (
                raw_record["name"] != expected_name
                or raw_record["batch_index"] != batch_index
                or raw_record["continuation_index"] != continuation_index
                or raw_record["page_ids"] != page_ids
                or raw_record["request_parameters"] != expected_parameters
            ):
                raise ValueError("raw response request identity is inconsistent")
            path = source_curation._safe_file(
                enrichment_dir / expected_name, "Commons raw enrichment response"
            )
            raw = path.read_bytes()
            if (
                raw_record["bytes"] != len(raw)
                or raw_record["sha256"] != hashlib.sha256(raw).hexdigest()
            ):
                raise ValueError("Commons raw enrichment response identity mismatch")
            try:
                payload = _object(json.loads(raw), "Commons enrichment response")
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Commons raw enrichment response is not JSON") from exc
            payloads.append(payload)
            record_index += 1
            continuation = _continuation(payload)
            if continuation is None:
                break
            digest = _canonical_sha256(continuation)
            if digest in seen_continuations:
                raise ValueError("Commons repeated a continuation token")
            seen_continuations.add(digest)
            continuation_index += 1
            if continuation_index >= _MAX_CONTINUATIONS_PER_BATCH:
                raise ValueError("Commons continuation count exceeds the safety bound")
        batch_payloads.append((page_ids, payloads))
    if record_index != len(raw_records):
        raise ValueError("enrichment has extraneous raw responses")
    return batch_payloads


def validate_enrichment(harvest_dir: Path, enrichment_dir: Path) -> dict[str, Any]:
    """Re-derive a D1 enrichment record from its harvest and frozen raw bytes."""

    harvest_dir = source_curation._safe_existing_directory(
        harvest_dir, "D1 harvest package"
    )
    enrichment_dir = source_curation._safe_existing_directory(
        enrichment_dir, "D1 enrichment package"
    )
    harvest = source_curation.validate_harvest(harvest_dir)
    if harvest["experimental_role"] != _ROLE or harvest["concept_id"] != _CONCEPT_ID:
        raise ValueError("enrichment requires the exact D1 Fontana del Moro harvest")
    manifest = source_curation._read_canonical(
        enrichment_dir / "enrichment.json", "Commons enrichment manifest"
    )
    expected_keys = {
        "schema",
        "kind",
        "status",
        "experimental_role",
        "concept_id",
        "retrieved_at_utc",
        "source_policy_sha256",
        "enrichment_tool_sha256",
        "harvest_binding",
        "request_contract",
        "raw_responses",
        "candidate_rows",
        "counts",
        "claim_limit",
        "human_review_state",
        "human_records",
        "fixture_manifest_created",
        "execution_authorized",
        "enrichment_sha256",
    }
    _exact(manifest, expected_keys, "Commons enrichment manifest")
    body = {key: value for key, value in manifest.items() if key != "enrichment_sha256"}
    if (
        manifest["schema"] != _SCHEMA
        or manifest["kind"] != _KIND
        or manifest["status"] != _STATUS
        or manifest["experimental_role"] != _ROLE
        or manifest["concept_id"] != _CONCEPT_ID
        or manifest["source_policy_sha256"] != harvest["source_policy_sha256"]
        or manifest["enrichment_tool_sha256"]
        != _file_sha256(Path(__file__).resolve(strict=True))
        or manifest["request_contract"]
        != {
            "api": _API,
            "maximum_page_ids_per_batch": _BATCH_LIMIT,
            "base_parameters": _BASE_PARAMETERS,
            "continuation_policy": "follow_every_returned_continuation",
        }
        or manifest["claim_limit"] != _CLAIM_LIMIT
        or manifest["human_review_state"] != "pending"
        or manifest["human_records"] != []
        or manifest["fixture_manifest_created"] is not False
        or manifest["execution_authorized"] is not False
        or manifest["enrichment_sha256"] != _canonical_sha256(body)
    ):
        raise ValueError("Commons enrichment is not an intact pending record")
    source_curation._canonical_utc(manifest["retrieved_at_utc"], "retrieved_at_utc")

    harvest_rows = _eligible_rows(harvest)
    harvest_files = _harvest_file_index(harvest_dir, harvest)
    source_ids = sorted(row["source_id"] for row in harvest_rows)
    expected_binding = {
        "harvest_sha256": harvest["harvest_sha256"],
        "harvest_file_sha256": _file_sha256(harvest_dir / "harvest.json"),
        "eligible_count": len(harvest_rows),
        "eligible_source_ids_sha256": _canonical_sha256(source_ids),
    }
    if manifest["harvest_binding"] != expected_binding:
        raise ValueError("Commons enrichment harvest binding is stale")
    raw_records = manifest["raw_responses"]
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("Commons enrichment must bind raw responses")
    expected_batches = list(
        _batches([row["provider_object_id"] for row in harvest_rows])
    )
    batch_payloads = _read_raw_responses(enrichment_dir, raw_records, expected_batches)
    derived_rows = _derive_rows(harvest_rows, harvest_files, batch_payloads)
    if manifest["candidate_rows"] != derived_rows:
        raise ValueError("Commons enrichment rows do not rederive from frozen bytes")
    expected_counts = {
        "requested": len(derived_rows),
        "mechanically_consistent": len(derived_rows),
    }
    if manifest["counts"] != expected_counts:
        raise ValueError("Commons enrichment counts do not rederive")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze or validate pending D1 Commons enrichment evidence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--harvest-dir", type=Path, required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--retrieved-at-utc", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--harvest-dir", type=Path, required=True)
    validate.add_argument("--enrichment-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "collect":
        record = enrich_d1(
            args.harvest_dir,
            args.output_dir,
            retrieved_at_utc=args.retrieved_at_utc,
        )
    elif args.command == "validate":
        record = validate_enrichment(args.harvest_dir, args.enrichment_dir)
    else:  # pragma: no cover - argparse enforces choices.
        raise AssertionError("unreachable command")
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
