#!/usr/bin/env python3
"""Freeze and inspect unapproved public-source candidates for Krea D1/D2.

This module deliberately stops before fixture admission.  It can freeze public
metadata, retrieve the corresponding image bytes under a hard byte budget, and
build machine-derived duplicate/QC evidence.  It cannot create rights,
caption, similarity, split, or fixture approval records and it never authorizes
GPU execution.

The final admitted fixture remains the responsibility of :mod:`krea_fixture`,
which requires distinct named-human records and re-hashes the exact staged
training/evaluation bytes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import html
import inspect
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
import unicodedata
import warnings
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

try:
    from . import krea_provenance
except ImportError:  # pragma: no cover - direct script execution.
    import krea_provenance  # type: ignore[no-redef]


_SCHEMA = 2
_HARVEST_KIND = "forge-krea-source-candidate-harvest"
_MATERIALIZATION_KIND = "forge-krea-source-candidate-materialization"
_INSPECTION_KIND = "forge-krea-source-candidate-inspection"
_UNAPPROVED = "candidate_unreviewed"
_USER_AGENT = (
    "SN56-Krea-fixture-curation/1.0 "
    "(public research; https://github.com/tuly1/sn56-forge-toolkit)"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHA1 = re.compile(r"[0-9a-f]{40}")
_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
_SAFE_SOURCE_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,127}")
_D1_CATEGORY = "Category:Fontana del Moro (Rome)"
_D1_API = "https://commons.wikimedia.org/w/api.php"
_D2_API = "https://api.artic.edu/api/v1/artworks/search"
_D2_ARTIST_ID = 26646
_D2_ARTIST = "Tsukioka Kôgyo"
_D2_SERIES = "Pictures of No Performances (Nogaku Zue)"
_D2_FIELDS = (
    "id,api_model,api_link,title,alt_titles,thumbnail,main_reference_number,"
    "date_start,date_end,date_display,artist_id,artist_title,artist_display,"
    "place_of_origin,dimensions,medium_display,inscriptions,credit_line,"
    "publication_history,provenance_text,is_public_domain,copyright_notice,"
    "department_title,classification_title,classification_titles,image_id,"
    "alt_image_ids,updated_at"
)
_D2_PAGE_LIMIT = 100
_D2_IIIF_WIDTH = 1686
_MAXIMUM_DECODED_PIXELS = 25_000_000
_D2_HARD_EXCLUSIONS = {
    154905: "series_paratext_index_page",
    154971: "series_paratext_index_page",
    155014: "series_paratext_index_page",
    155326: "series_paratext_index_page",
    155412: "series_paratext_frontispiece",
    155398: "domain_outlier_properties_plate",
}
_D1_HARD_REJECTS = {
    12758919: "off_domain_museum_model",
    12758925: "off_domain_museum_model",
    35799064: "off_domain_historical_print",
    84796749: "off_domain_historical_print",
    61135226: "off_domain_historical_print",
    76125648: "people_dominant_frame",
}
_D1_QUARANTINES = {
    66539312: "subject_identity_neptune_fountain_label",
    66539352: "subject_identity_neptune_fountain_label",
    22819106: "subject_identity_neptune_fountain_label",
    374687: "filename_description_subject_conflict",
}
_D1_BURST_GROUPS = {
    page_id: group
    for group, page_ids in {
        "commons-user-argenberg-session": (94356422, 94356423),
        "commons-user-carlomorino-session": (374687, 384687),
        "commons-user-daderot-session": (78782113, 78782114, 78782119),
        "commons-user-fa2010-session": (12758919, 12758925),
        "commons-user-jean-pol-grandmont-session": (
            84856131,
            84868150,
            84868331,
            84872662,
            84879897,
        ),
        "commons-user-jebulon-session": (30817295, 30817360),
        "commons-user-sailko-session": (
            76704539,
            76704541,
            76704542,
            76704548,
            76704549,
            76704551,
            76704555,
            76704556,
            76704557,
            76704561,
            76704564,
            76704565,
            76704567,
            76704570,
            76704571,
            76704573,
            76704574,
            76704575,
            76704576,
            76704578,
            76704580,
        ),
        "commons-user-sig-sg-510-session": (56011847, 56011849, 56011850, 56011851),
        "flickr-101561334-n08-session": (101211643, 101211644),
        "flickr-34585612-n00-session": (76176062, 76176304, 76177252, 76177287),
        "flickr-62091376-n03-session": (66539312, 66539352),
        "flickr-72746018-n00-session": (24668869, 24668870),
        "panoramio-5152111-session": (60306420, 60306474, 60306478),
        "text-joonas-lyytinen-session": (824102, 824103),
    }.items()
    for page_id in page_ids
}
_POLICY_PATH = Path(__file__).with_name("week5") / "krea-curation-source-policy.json"
_SELECTION_POLICY_PATH = (
    Path(__file__).with_name("week5") / "krea-curation-selection-policy.json"
)
_ALLOWED_IMAGE_MIMES = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/tiff"}
)
_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/tiff": ".tif",
}
_CLAIM_LIMIT = (
    "metadata-byte-and-machine-screening-only-no-rights-similarity-caption-"
    "split-or-admission-approval"
)
_HUMAN_GATES = {
    "caption_policy": "pending",
    "concept_suitability": "pending",
    "exhaustive_similarity": "pending",
    "group_identity": "pending",
    "independent_admission": "pending",
    "rights_acceptance": "pending",
    "visual_qc": "pending",
}


def _canonical_sha256(value: Any) -> str:
    return krea_provenance.canonical_sha256(value)


def _normalizer_contract_sha256() -> str:
    functions = (
        _canonical_sha256,
        _object,
        _normalize,
        _alnum_words,
        _plain_metadata,
        d1_request_url,
        d2_query,
        _validated_download_url,
        _supported_commons_license,
        _d1_creator_id,
        _d1_rows,
        _d2_play_root,
        _d2_play_aliases,
        _d2_accession_family,
        _d2_rows,
    )
    contract = {
        "schema": 1,
        "functions": [
            {"name": function.__name__, "source": inspect.getsource(function)}
            for function in functions
        ],
        "constants": {
            "d1_category": _D1_CATEGORY,
            "d1_api": _D1_API,
            "d1_hard_rejects": _D1_HARD_REJECTS,
            "d1_quarantines": _D1_QUARANTINES,
            "d1_burst_groups": _D1_BURST_GROUPS,
            "d2_artist_id": _D2_ARTIST_ID,
            "d2_api": _D2_API,
            "d2_artist": _D2_ARTIST,
            "d2_series": _D2_SERIES,
            "d2_fields": _D2_FIELDS,
            "d2_page_limit": _D2_PAGE_LIMIT,
            "d2_hard_exclusions": _D2_HARD_EXCLUSIONS,
            "d2_iiif_width": _D2_IIIF_WIDTH,
            "allowed_image_mimes": sorted(_ALLOWED_IMAGE_MIMES),
            "sha1_pattern": _SHA1.pattern,
            "uuid_pattern": _UUID.pattern,
        },
        "source_policy_sha256": _source_policy()[1],
    }
    return _canonical_sha256(contract)


def _file_sha256(path: Path) -> str:
    return krea_provenance.file_sha256(path)


def _file_sha1(path: Path) -> str:
    digest = hashlib.sha1()  # noqa: S324 - provider identity is SHA-1.
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return " ".join(value.split())


def _canonical_utc(value: Any, label: str) -> str:
    text = _text(value, label)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", text):
        raise ValueError(f"{label} must be canonical UTC (YYYY-MM-DDTHH:MM:SSZ)")
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{label} is not a real UTC timestamp") from exc
    now = datetime.now(timezone.utc)
    if parsed < datetime(2020, 1, 1, tzinfo=timezone.utc) or parsed > now + timedelta(
        seconds=60
    ):
        raise ValueError(f"{label} is outside the accepted evidence time bounds")
    return text


def _utc_datetime(value: Any, label: str) -> datetime:
    return datetime.strptime(
        _canonical_utc(value, label), "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)


def _source_policy() -> tuple[dict[str, Any], str]:
    path = _safe_file(_POLICY_PATH, "source curation policy")
    raw = path.read_bytes()
    try:
        policy = _object(json.loads(raw), "source curation policy")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("source curation policy must be JSON") from exc
    if raw != krea_provenance.canonical_bytes(policy) + b"\n":
        raise ValueError("source curation policy must be canonical JSON plus one newline")
    _exact(
        policy,
        {
            "schema",
            "kind",
            "status",
            "claim_limit",
            "human_gates",
            "hard_caps",
            "profiles",
        },
        "source curation policy",
    )
    expected_profiles = {
        "D1": {
            "allowed_download_hosts": ["upload.wikimedia.org"],
            "concept_id": "fontana-del-moro",
            "minimum_machine_eligible_rows": 50,
            "minimum_native_height": 768,
            "minimum_native_width": 768,
            "source_locator": "https://commons.wikimedia.org/wiki/Category:Fontana_del_Moro_(Rome)",
            "source_system": "Wikimedia Commons",
        },
        "D2": {
            "allowed_download_hosts": ["www.artic.edu"],
            "concept_id": "tsukioka-kogyo-nogaku-zue",
            "iiif_width": _D2_IIIF_WIDTH,
            "minimum_machine_eligible_components": 88,
            "minimum_machine_eligible_rows": 88,
            "minimum_native_height": 768,
            "minimum_native_width": 768,
            "source_locator": "https://www.artic.edu/open-access",
            "source_system": "Art Institute of Chicago",
        },
    }
    if (
        policy["schema"] != 1
        or policy["kind"] != "forge-krea-source-curation-policy"
        or policy["status"] != "curation_only_no_admission"
        or policy["claim_limit"] != _CLAIM_LIMIT
        or policy["human_gates"] != _HUMAN_GATES
        or policy["hard_caps"]
        != {
            "maximum_decoded_pixels": _MAXIMUM_DECODED_PIXELS,
            "maximum_file_bytes": 134217728,
            "maximum_persisted_bytes": 4294967296,
        }
        or policy["profiles"] != expected_profiles
    ):
        raise ValueError("source curation policy differs from the frozen implementation")
    return policy, hashlib.sha256(raw).hexdigest()


def _selection_policy() -> tuple[dict[str, Any], str]:
    path = _safe_file(_SELECTION_POLICY_PATH, "curation selection policy")
    raw = path.read_bytes()
    try:
        policy = _object(json.loads(raw), "curation selection policy")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("curation selection policy must be JSON") from exc
    if raw != krea_provenance.canonical_bytes(policy) + b"\n":
        raise ValueError("curation selection policy must be canonical JSON plus one newline")
    expected = {
        "schema": 1,
        "kind": "forge-krea-curation-selection-policy",
        "status": "pre_admission_policy_frozen",
        "admission_authorized": False,
        "claim_limit": (
            "machine-screening-and-split-objective-only-human-rights-visual-"
            "caption-similarity-and-admission-review-remain-required"
        ),
        "crop_policy": "no_crop",
        "human_review": "exhaustive_over_final_rows",
        "perceptual_screen": {
            "algorithm": "rgb-luma-average-hash-8x8-bilinear-after-exif-transpose",
            "automatic_union_maximum_hamming_distance": 6,
            "human_review_queue_maximum_hamming_distance": 10,
        },
        "concepts": {
            "D1": {
                "creator_disjoint_between_splits": True,
                "exact_evaluation_rows": 24,
                "exact_training_rows": 18,
                "maximum_selected_rows_per_creator": 3,
                "minimum_accepted_unused_reserve_rows": 8,
                "selection_objective": [
                    "maximize_distinct_selected_creators",
                    "minimize_maximum_per_creator_contribution",
                    "minimize_train_eval_stratum_distance",
                    "maximize_frozen_quality_grades",
                    "policy_digest_plus_source_id_sha256_tiebreak",
                ],
                "whole_group_fields": [
                    "creator_id",
                    "burst_id",
                    "scene_id",
                    "human_similarity_cluster_id",
                ],
            },
            "D2": {
                "exact_evaluation_rows": 40,
                "exact_training_rows": 36,
                "maximum_selected_rows_per_play_component": 1,
                "private_split_key_state": "pending_named_reviewer_commitment",
                "whole_group_fields": [
                    "play_root_id",
                    "accession_family",
                    "scene_id",
                    "human_similarity_cluster_id",
                ],
            },
        },
    }
    if policy != expected:
        raise ValueError("curation selection policy violates the frozen boundary")
    return policy, hashlib.sha256(raw).hexdigest()


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(plain.casefold().split())


def _alnum_words(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", _normalize(value)).split())


def _plain_metadata(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    raw = value.get("value")
    if not isinstance(raw, str):
        return ""
    without_tags = re.sub(r"<[^>]*>", " ", html.unescape(raw))
    return " ".join(without_tags.split())


def _d1_creator_id(raw_artist: Any) -> str:
    raw = raw_artist.get("value") if isinstance(raw_artist, dict) else ""
    if not isinstance(raw, str):
        raw = ""
    links = [html.unescape(item) for item in re.findall(r'href="([^"]+)"', raw)]
    identities: list[str] = []
    for link in links:
        parsed = urlsplit("https:" + link if link.startswith("//") else link)
        host = (parsed.hostname or "").casefold()
        path = parsed.path
        query = parsed.query
        user_match = re.search(r"/wiki/User:([^/?#]+)", path, flags=re.IGNORECASE)
        if user_match is None:
            user_match = re.search(r"(?:^|&)title=User:([^&]+)", query, flags=re.IGNORECASE)
        if "commons.wikimedia.org" == host and user_match is not None:
            name = _alnum_words(user_match.group(1).replace("_", " "))
            identities.append("commons-user:" + name)
            continue
        flickr = re.search(r"/(?:people)/([^/?#]+)", path, flags=re.IGNORECASE)
        if "flickr.com" in host and flickr is not None:
            identities.append("flickr:" + flickr.group(1).casefold())
            continue
        panoramio = re.search(r"panoramio\.com/user/(\d+)", path, flags=re.IGNORECASE)
        if host == "web.archive.org" and panoramio is not None:
            identities.append("panoramio:" + panoramio.group(1))
            continue
        if host == "500px.com":
            name = path.strip("/").casefold()
            if name:
                identities.append("500px:" + name)
                continue
        if host.endswith("wikipedia.org"):
            name = path.rsplit("/", 1)[-1]
            name = re.sub(r"^[a-z]{2,3}:", "", name, flags=re.IGNORECASE)
            identities.append("wikipedia:" + _alnum_words(name.replace("_", " ")))
    if identities:
        return "+".join(sorted(set(identities)))
    visible = _alnum_words(_plain_metadata(raw_artist).replace("_", " "))
    return "text:" + (visible or "unknown")


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
    parent = _safe_existing_directory(path.parent, f"{label} parent")
    path.mkdir(mode=0o750)
    if path.parent != parent:
        raise AssertionError("normalized output parent changed")
    return path


def _atomic_create(path: Path, payload: bytes, mode: int = 0o640) -> None:
    path = Path(os.path.abspath(path))
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
    path = _safe_file(path, label)
    raw = path.read_bytes()
    try:
        value = _object(json.loads(raw), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid JSON") from exc
    if raw != krea_provenance.canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline")
    return value


def _harvest_retrieval_binding(harvest: dict[str, Any]) -> dict[str, str]:
    urls = [
        {"source_id": row["source_id"], "download_url": row["download_url"]}
        for row in harvest["candidate_rows"]
        if row["eligibility"]["passed"]
    ]
    urls.sort(key=lambda row: row["source_id"])
    return {
        "harvest_sha256": harvest["harvest_sha256"],
        "eligible_source_urls_sha256": _canonical_sha256(urls),
    }


def validate_retrieval_authorization(
    path: Path, *, harvest: dict[str, Any], requested_maximum_bytes: int
) -> dict[str, Any]:
    record = _read_canonical(path, "retrieval scope authorization")
    expected = {
        "schema",
        "kind",
        "owner_identity",
        "authorized_at_utc",
        "roles",
        "maximum_persisted_bytes",
        "decision",
        "acknowledgements",
        "source_policy_sha256",
        "harvest_bindings",
        "authorization_sha256",
    }
    _exact(record, expected, "retrieval scope authorization")
    body = {key: value for key, value in record.items() if key != "authorization_sha256"}
    owner = _text(record["owner_identity"], "owner_identity")
    if len(owner.split()) < 2 or not all(
        any(character.isalpha() for character in word) for word in owner.split()
    ):
        raise ValueError("retrieval owner must be a named human identity")
    roles = record["roles"]
    role = harvest["experimental_role"]
    cap = record["maximum_persisted_bytes"]
    acknowledgements = _object(record["acknowledgements"], "acknowledgements")
    expected_acknowledgements = {
        "aic_public_domain_images_may_have_third_party_rights": True,
        "commons_cc_by_attribution_must_be_preserved": True,
        "commons_sharealike_material_is_excluded": True,
        "download_does_not_approve_fixture_admission": True,
        "named_rights_review_still_required": True,
    }
    hard_cap = _source_policy()[0]["hard_caps"]["maximum_persisted_bytes"]
    bindings = _object(record["harvest_bindings"], "harvest_bindings")
    _exact(bindings, {"D1", "D2"}, "harvest_bindings")
    for binding_role, binding in bindings.items():
        binding = _object(binding, f"harvest_bindings.{binding_role}")
        _exact(
            binding,
            {"harvest_sha256", "eligible_source_urls_sha256"},
            f"harvest_bindings.{binding_role}",
        )
        if any(
            not isinstance(binding[key], str) or not _SHA256.fullmatch(binding[key])
            for key in binding
        ):
            raise ValueError("retrieval harvest binding contains an invalid digest")
    if (
        record["schema"] != 1
        or record["kind"] != "forge-krea-curation-retrieval-scope-authorization"
        or record["decision"]
        != "authorize_public_candidate_retrieval_for_curation_only"
        or roles != ["D1", "D2"]
        or role not in roles
        or isinstance(cap, bool)
        or not isinstance(cap, int)
        or cap <= 0
        or cap > hard_cap
        or requested_maximum_bytes > cap
        or record["source_policy_sha256"] != harvest["source_policy_sha256"]
        or bindings[role] != _harvest_retrieval_binding(harvest)
        or acknowledgements != expected_acknowledgements
        or record["authorization_sha256"] != _canonical_sha256(body)
    ):
        raise ValueError("retrieval scope authorization is absent, invalid, or too narrow")
    authorized_at = _utc_datetime(record["authorized_at_utc"], "authorized_at_utc")
    harvested_at = _utc_datetime(harvest["retrieved_at_utc"], "harvest.retrieved_at_utc")
    if authorized_at < harvested_at:
        raise ValueError("retrieval authorization predates the bound source harvest")
    return record


def _fetch(
    request: Request, *, timeout_s: float = 60.0, maximum_bytes: int = 16 * 1024 * 1024
) -> bytes:
    requested = urlsplit(request.full_url)
    if (
        requested.scheme != "https"
        or requested.hostname not in {"commons.wikimedia.org", "api.artic.edu"}
        or requested.port not in {None, 443}
        or requested.username is not None
        or requested.password is not None
        or requested.fragment
    ):
        raise ValueError("metadata request is outside the exact HTTPS host contract")
    with urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - checked above.
        if response.geturl() != request.full_url:
            raise ValueError("metadata request redirected away from its exact URL")
        if response.headers.get_content_type() not in {
            "application/json",
            "application/ld+json",
        }:
            raise ValueError("metadata response is not JSON")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = response.read(min(1024 * 1024, maximum_bytes + 1 - total))
            if not block:
                break
            total += len(block)
            if total > maximum_bytes:
                raise ValueError("metadata response exceeds the byte cap")
            chunks.append(block)
        if total <= 0:
            raise ValueError("metadata response is empty")
        return b"".join(chunks)


def d1_request_url() -> str:
    query = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": _D1_CATEGORY,
        "gcmtype": "file",
        "gcmlimit": "500",
        "prop": "imageinfo|revisions",
        "iiprop": "url|sha1|size|mime|extmetadata",
        "rvprop": "ids|timestamp",
        "format": "json",
        "formatversion": "2",
    }
    return _D1_API + "?" + urlencode(query)


def d2_query(page: int) -> dict[str, Any]:
    if isinstance(page, bool) or not isinstance(page, int) or page <= 0:
        raise ValueError("D2 page must be a positive integer")
    return {
        "query": {
            "bool": {
                "filter": [
                    {"term": {"artist_id": _D2_ARTIST_ID}},
                    {"term": {"artist_title.keyword": _D2_ARTIST}},
                    {"term": {"is_public_domain": True}},
                    {"exists": {"field": "image_id"}},
                    {
                        "wildcard": {
                            "title.keyword": {
                                "value": f'*, from the series "{_D2_SERIES}"'
                            }
                        }
                    },
                ]
            }
        },
        "fields": _D2_FIELDS.split(","),
        "limit": _D2_PAGE_LIMIT,
        "page": page,
        "sort": [{"id": "asc"}],
    }


def _supported_commons_license(name: str, url: str) -> bool:
    normalized = _normalize(name)
    normalized_url = url.rstrip("/")
    if normalized == "public domain":
        return normalized_url in {
            "",
            "https://creativecommons.org/publicdomain/mark/1.0",
            "http://creativecommons.org/publicdomain/mark/1.0",
        }
    if normalized == "cc0":
        return normalized_url in {
            "https://creativecommons.org/publicdomain/zero/1.0",
            "http://creativecommons.org/publicdomain/zero/1.0",
            "https://creativecommons.org/publicdomain/zero/1.0/deed.en",
            "http://creativecommons.org/publicdomain/zero/1.0/deed.en",
        }
    match = re.fullmatch(r"cc by ([0-9]+(?:\.[0-9]+)?)(?: [a-z]{2})?", normalized)
    if match is None:
        return False
    version = match.group(1)
    return normalized_url in {
        f"https://creativecommons.org/licenses/by/{version}",
        f"http://creativecommons.org/licenses/by/{version}",
    }


def _d1_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    query = _object(payload.get("query"), "Commons response.query")
    pages = query.get("pages")
    if not isinstance(pages, list):
        raise ValueError("Commons response pages must be an array")
    rows: list[dict[str, Any]] = []
    for raw_page in pages:
        page = _object(raw_page, "Commons page")
        page_id = page.get("pageid")
        title = page.get("title")
        revisions = page.get("revisions")
        infos = page.get("imageinfo")
        if (
            isinstance(page_id, bool)
            or not isinstance(page_id, int)
            or page_id <= 0
            or not isinstance(title, str)
            or not title.startswith("File:")
            or not isinstance(revisions, list)
            or len(revisions) != 1
            or not isinstance(infos, list)
            or len(infos) != 1
        ):
            raise ValueError("Commons candidate lacks one page/revision/image identity")
        revision = _object(revisions[0], "Commons revision")
        info = _object(infos[0], "Commons imageinfo")
        ext = _object(info.get("extmetadata"), "Commons extmetadata")
        revision_id = revision.get("revid")
        revision_timestamp = revision.get("timestamp")
        width = info.get("width")
        height = info.get("height")
        mime = info.get("mime")
        source_sha1 = info.get("sha1")
        original_url = info.get("url")
        license_name = _plain_metadata(ext.get("LicenseShortName"))
        license_url = _plain_metadata(ext.get("LicenseUrl"))
        reasons: list[str] = []
        if not _supported_commons_license(license_name, license_url):
            reasons.append("license_name_url_pair_not_allowlisted_pd_cc0_or_cc_by")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
            or width < 768
            or height < 768
        ):
            reasons.append("native_dimensions_below_768_on_one_or_both_axes")
        if mime not in _ALLOWED_IMAGE_MIMES:
            reasons.append("unsupported_image_mime")
        if not isinstance(source_sha1, str) or not _SHA1.fullmatch(source_sha1):
            reasons.append("missing_or_invalid_provider_sha1")
        if not isinstance(original_url, str):
            reasons.append("missing_original_url")
        else:
            try:
                _validated_download_url(original_url, "D1")
            except ValueError:
                reasons.append("original_url_outside_download_allowlist")
        if (
            isinstance(revision_id, bool)
            or not isinstance(revision_id, int)
            or revision_id <= 0
            or not isinstance(revision_timestamp, str)
        ):
            reasons.append("missing_revision_identity")
        rows.append(
            {
                "source_id": f"commons-{page_id}",
                "provider_object_id": page_id,
                "title": title,
                "source_page_url": "https://commons.wikimedia.org/wiki/"
                + quote(title.replace(" ", "_"), safe=":()_-"),
                "download_url": original_url,
                "provider_content_sha1": source_sha1,
                "revision_id": revision_id,
                "revision_timestamp": revision_timestamp,
                "native_width": width,
                "native_height": height,
                "provider_mime": mime,
                "license_name": license_name,
                "license_url": license_url,
                "usage_terms": _plain_metadata(ext.get("UsageTerms")),
                "creator": _plain_metadata(ext.get("Artist")),
                "creator_id_hint": _d1_creator_id(ext.get("Artist")),
                "credit": _plain_metadata(ext.get("Credit")),
                "attribution": _plain_metadata(ext.get("Attribution")),
                "burst_id_hint": _D1_BURST_GROUPS.get(
                    page_id, f"unreviewed-singleton-commons-{page_id}"
                ),
                "preliminary_human_triage_hint": (
                    {
                        "status": "hard_reject",
                        "reason": _D1_HARD_REJECTS[page_id],
                    }
                    if page_id in _D1_HARD_REJECTS
                    else {
                        "status": "quarantine",
                        "reason": _D1_QUARANTINES[page_id],
                    }
                    if page_id in _D1_QUARANTINES
                    else {"status": "pending_visual_qc", "reason": None}
                ),
                "eligibility": {"passed": not reasons, "reasons": reasons},
            }
        )
    rows.sort(key=lambda row: row["source_id"])
    if len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("Commons response contains duplicate page IDs")
    return rows


def _d2_play_root(title: str) -> str:
    marker = ', from the series "Pictures of No Performances (Nogaku Zue)"'
    if marker not in title:
        raise ValueError("AIC title does not bind the exact selected series")
    root, suffix = title.rsplit(marker, 1)
    if suffix.strip():
        raise ValueError("AIC title has unexpected text after exact series marker")
    return " ".join(root.split())


def _d2_play_aliases(play_root: str) -> list[str]:
    """Return conservative catalog aliases used only as grouping hints."""

    normalized = _normalize(play_root)
    normalized = re.sub(r"\(\s*kyogen\s*\)", " ", normalized, flags=re.IGNORECASE)
    aliases = {_alnum_words(normalized)}
    if re.search(r"\bor\b", normalized):
        aliases.update(
            _alnum_words(part)
            for part in re.split(r"\bor\b", normalized)
            if part.strip()
        )
    for inner in re.findall(r"\(([^)]*)\)", normalized):
        inner_alias = _alnum_words(inner)
        outer = _alnum_words(re.sub(r"\([^)]*\)", " ", normalized))
        if inner_alias and inner_alias != "kyogen":
            aliases.add(inner_alias)
            aliases.add(" ".join(part for part in (outer, inner_alias) if part))
    aliases.discard("")
    return sorted(aliases)


def _d2_accession_family(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(\d{4}\.\d+)\.(\d+)([a-z])?", value.strip())
    if match is None:
        return None
    return f"{match.group(1)}.{int(match.group(2))}"


def _d2_rows(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in pages:
        data = page.get("data")
        if not isinstance(data, list):
            raise ValueError("AIC response data must be an array")
        for raw in data:
            item = _object(raw, "AIC artwork")
            object_id = item.get("id")
            if (
                isinstance(object_id, bool)
                or not isinstance(object_id, int)
                or object_id <= 0
            ):
                raise ValueError("AIC result lacks a positive object ID")
            title = item.get("title")
            artist = item.get("artist_title")
            artist_id = item.get("artist_id")
            image_id = item.get("image_id")
            reference_number = item.get("main_reference_number")
            thumbnail = item.get("thumbnail")
            thumbnail = thumbnail if isinstance(thumbnail, dict) else {}
            width = thumbnail.get("width")
            height = thumbnail.get("height")
            reasons: list[str] = []
            if artist_id != _D2_ARTIST_ID:
                reasons.append("artist_id_not_exact_tsukioka_kogyo")
            if _normalize(str(artist or "")) != _normalize(_D2_ARTIST):
                reasons.append("artist_not_exact_tsukioka_kogyo")
            if item.get("is_public_domain") is not True:
                reasons.append("object_not_public_domain")
            if not isinstance(image_id, str) or not _UUID.fullmatch(image_id):
                reasons.append("missing_image_id")
            try:
                play_root = _d2_play_root(str(title or ""))
            except ValueError:
                play_root = ""
                reasons.append("title_not_exact_selected_series")
            if _normalize(play_root).startswith("index page"):
                reasons.append("series_paratext_index_page")
            hard_exclusion = _D2_HARD_EXCLUSIONS.get(object_id)
            if hard_exclusion is not None and hard_exclusion not in reasons:
                reasons.append(hard_exclusion)
            accession_family = _d2_accession_family(reference_number)
            if accession_family is None:
                reasons.append("accession_family_not_scene_plate")
            if (
                isinstance(width, bool)
                or not isinstance(width, int)
                or isinstance(height, bool)
                or not isinstance(height, int)
                or width < 768
                or height < 768
            ):
                reasons.append("native_dimensions_below_768_on_one_or_both_axes")
            download_url = (
                f"https://www.artic.edu/iiif/2/{image_id}/full/"
                f"{_D2_IIIF_WIDTH},/0/default.jpg"
                if isinstance(image_id, str) and _UUID.fullmatch(image_id)
                else None
            )
            rows.append(
                {
                    "source_id": f"aic-{object_id}",
                    "provider_object_id": object_id,
                    "provider_image_id": image_id,
                    "main_reference_number": reference_number,
                    "accession_family_key": accession_family,
                    "title": title,
                    "play_root": play_root,
                    "normalized_play_root": _alnum_words(play_root),
                    "automatic_play_aliases": _d2_play_aliases(play_root),
                    "source_page_url": f"https://www.artic.edu/artworks/{object_id}",
                    "download_url": download_url,
                    "native_width": width,
                    "native_height": height,
                    "provider_mime": "image/jpeg",
                    "license_name": "CC0 / public domain object image",
                    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                    "artist_title": artist,
                    "artist_id": artist_id,
                    "artist_display": item.get("artist_display"),
                    "date_display": item.get("date_display"),
                    "date_start": item.get("date_start"),
                    "date_end": item.get("date_end"),
                    "classification_title": item.get("classification_title"),
                    "classification_titles": item.get("classification_titles"),
                    "medium_display": item.get("medium_display"),
                    "api_model": item.get("api_model"),
                    "api_link": item.get("api_link"),
                    "alt_titles": item.get("alt_titles"),
                    "place_of_origin": item.get("place_of_origin"),
                    "dimensions": item.get("dimensions"),
                    "inscriptions": item.get("inscriptions"),
                    "credit_line": item.get("credit_line"),
                    "publication_history": item.get("publication_history"),
                    "provenance_text": item.get("provenance_text"),
                    "copyright_notice": item.get("copyright_notice"),
                    "department_title": item.get("department_title"),
                    "alt_image_ids": item.get("alt_image_ids"),
                    "updated_at": item.get("updated_at"),
                    "provider_thumbnail_alt_text": thumbnail.get("alt_text"),
                    "eligibility": {"passed": not reasons, "reasons": reasons},
                }
            )
    rows.sort(key=lambda row: row["source_id"])
    if len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("AIC responses contain duplicate artwork IDs")
    image_ids = [row["provider_image_id"] for row in rows]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("AIC responses reuse an image UUID across artwork IDs")
    parents = {row["source_id"]: row["source_id"] for row in rows}

    def find(item: str) -> str:
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for index, left in enumerate(rows):
        left_aliases = set(left["automatic_play_aliases"])
        for right in rows[index + 1 :]:
            if left_aliases.intersection(right["automatic_play_aliases"]) or (
                left["accession_family_key"] is not None
                and left["accession_family_key"] == right["accession_family_key"]
            ):
                union(left["source_id"], right["source_id"])
    components: dict[str, list[str]] = {}
    for source_id in parents:
        components.setdefault(find(source_id), []).append(source_id)
    component_by_source = {}
    for members in components.values():
        component_id = "play-component-" + _canonical_sha256(sorted(members))
        for source_id in members:
            component_by_source[source_id] = component_id
    for row in rows:
        row["automatic_play_component_id"] = component_by_source[row["source_id"]]
    return rows


def _harvest_body(
    *,
    role: str,
    concept_id: str,
    source_system: str,
    source_locator: str,
    retrieved_at_utc: str,
    request_contract: dict[str, Any],
    raw_responses: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if role not in {"D1", "D2"}:
        raise ValueError("candidate role must be D1 or D2")
    eligible = sum(row["eligibility"]["passed"] is True for row in rows)
    source_policy, policy_sha256 = _source_policy()
    body = {
        "schema": _SCHEMA,
        "kind": _HARVEST_KIND,
        "experimental_role": role,
        "concept_id": concept_id,
        "source_system": source_system,
        "source_locator": source_locator,
        "retrieved_at_utc": _canonical_utc(retrieved_at_utc, "retrieved_at_utc"),
        "request_contract": request_contract,
        "source_policy_sha256": policy_sha256,
        "normalizer_contract_sha256": _normalizer_contract_sha256(),
        "raw_responses": raw_responses,
        "candidate_rows": rows,
        "counts": {
            "observed": len(rows),
            "eligible": eligible,
            "rejected": len(rows) - eligible,
        },
        "admission_state": _UNAPPROVED,
        "claim_limit": _CLAIM_LIMIT,
        "human_gates": source_policy["human_gates"],
        "human_approvals": [],
        "fixture_manifest_created": False,
        "gpu_execution_authorized": False,
    }
    return {**body, "harvest_sha256": _canonical_sha256(body)}


def _raw_record(name: str, payload: bytes) -> dict[str, Any]:
    return {"name": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def harvest_d1(output_dir: Path, *, retrieved_at_utc: str) -> dict[str, Any]:
    output_dir = _new_directory(output_dir, "D1 harvest output")
    url = d1_request_url()
    raw = _fetch(Request(url, headers={"User-Agent": _USER_AGENT}))
    raw_name = "source-response-001.json"
    _atomic_create(output_dir / raw_name, raw)
    try:
        payload = _object(json.loads(raw), "Commons response")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Commons returned non-JSON metadata") from exc
    if payload.get("continue") is not None:
        raise ValueError("Commons category exceeded the one-request frozen contract")
    manifest = _harvest_body(
        role="D1",
        concept_id="fontana-del-moro",
        source_system="Wikimedia Commons",
        source_locator="https://commons.wikimedia.org/wiki/Category:Fontana_del_Moro_(Rome)",
        retrieved_at_utc=retrieved_at_utc,
        request_contract={"method": "GET", "url": url},
        raw_responses=[_raw_record(raw_name, raw)],
        rows=_d1_rows(payload),
    )
    _canonical_create(output_dir / "harvest.json", manifest)
    validate_harvest(output_dir)
    return manifest


def harvest_d2(output_dir: Path, *, retrieved_at_utc: str) -> dict[str, Any]:
    output_dir = _new_directory(output_dir, "D2 harvest output")
    raw_records: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    page = 1
    total_pages: int | None = None
    while total_pages is None or page <= total_pages:
        query = d2_query(page)
        request_bytes = krea_provenance.canonical_bytes(query)
        raw = _fetch(
            Request(
                _D2_API,
                data=request_bytes,
                headers={"Content-Type": "application/json", "User-Agent": _USER_AGENT},
                method="POST",
            )
        )
        name = f"source-response-{page:03d}.json"
        _atomic_create(output_dir / name, raw)
        raw_records.append(_raw_record(name, raw))
        try:
            payload = _object(json.loads(raw), f"AIC response page {page}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("AIC returned non-JSON metadata") from exc
        pagination = _object(payload.get("pagination"), "AIC pagination")
        total = pagination.get("total")
        limit = pagination.get("limit")
        offset = pagination.get("offset")
        observed_total_pages = pagination.get("total_pages")
        current_page = pagination.get("current_page")
        if (
            isinstance(total, bool)
            or not isinstance(total, int)
            or total <= 0
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit != _D2_PAGE_LIMIT
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset != (page - 1) * _D2_PAGE_LIMIT
            or isinstance(observed_total_pages, bool)
            or not isinstance(observed_total_pages, int)
            or observed_total_pages <= 0
            or isinstance(current_page, bool)
            or not isinstance(current_page, int)
            or current_page != page
        ):
            raise ValueError("AIC pagination identity is invalid")
        if total_pages is None:
            total_pages = observed_total_pages
        elif total_pages != observed_total_pages:
            raise RuntimeError("AIC pagination changed during the frozen harvest")
        if total_pages > 50:
            raise ValueError("AIC exact query unexpectedly exceeds 50 pages")
        payloads.append(payload)
        page += 1
    manifest = _harvest_body(
        role="D2",
        concept_id="tsukioka-kogyo-nogaku-zue",
        source_system="Art Institute of Chicago",
        source_locator="https://www.artic.edu/open-access",
        retrieved_at_utc=retrieved_at_utc,
        request_contract={
            "method": "POST",
            "url": _D2_API,
            "page_queries": [d2_query(index) for index in range(1, page)],
        },
        raw_responses=raw_records,
        rows=_d2_rows(payloads),
    )
    _canonical_create(output_dir / "harvest.json", manifest)
    validate_harvest(output_dir)
    return manifest


def validate_harvest(package_dir: Path) -> dict[str, Any]:
    package_dir = _safe_existing_directory(package_dir, "harvest package")
    manifest = _read_canonical(package_dir / "harvest.json", "harvest manifest")
    expected = {
        "schema",
        "kind",
        "experimental_role",
        "concept_id",
        "source_system",
        "source_locator",
        "retrieved_at_utc",
        "request_contract",
        "source_policy_sha256",
        "normalizer_contract_sha256",
        "raw_responses",
        "candidate_rows",
        "counts",
        "admission_state",
        "claim_limit",
        "human_gates",
        "human_approvals",
        "fixture_manifest_created",
        "gpu_execution_authorized",
        "harvest_sha256",
    }
    _exact(manifest, expected, "harvest manifest")
    body = {key: value for key, value in manifest.items() if key != "harvest_sha256"}
    source_policy, policy_sha256 = _source_policy()
    if (
        manifest["schema"] != _SCHEMA
        or manifest["kind"] != _HARVEST_KIND
        or manifest["admission_state"] != _UNAPPROVED
        or manifest["claim_limit"] != _CLAIM_LIMIT
        or manifest["human_gates"] != source_policy["human_gates"]
        or manifest["human_approvals"] != []
        or manifest["fixture_manifest_created"] is not False
        or manifest["gpu_execution_authorized"] is not False
        or manifest["source_policy_sha256"] != policy_sha256
        or manifest["normalizer_contract_sha256"] != _normalizer_contract_sha256()
        or manifest["harvest_sha256"] != _canonical_sha256(body)
    ):
        raise ValueError("harvest manifest is not an intact unapproved candidate record")
    _canonical_utc(manifest["retrieved_at_utc"], "retrieved_at_utc")
    raw_records = manifest["raw_responses"]
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("harvest must bind at least one raw response")
    payloads: list[dict[str, Any]] = []
    for index, raw_record in enumerate(raw_records, start=1):
        raw_record = _object(raw_record, "raw response identity")
        _exact(raw_record, {"name", "bytes", "sha256"}, "raw response identity")
        expected_name = f"source-response-{index:03d}.json"
        if raw_record["name"] != expected_name:
            raise ValueError("raw response names must be contiguous and canonical")
        path = _safe_file(package_dir / expected_name, "raw response")
        raw = path.read_bytes()
        if (
            raw_record["bytes"] != len(raw)
            or raw_record["sha256"] != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("raw response identity mismatch")
        try:
            payloads.append(_object(json.loads(raw), "raw response"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("raw response is not JSON") from exc
    role = manifest["experimental_role"]
    request_contract = _object(manifest["request_contract"], "request_contract")
    if role == "D1":
        if (
            len(payloads) != 1
            or payloads[0].get("continue") is not None
            or payloads[0].get("batchcomplete") is not True
            or request_contract != {"method": "GET", "url": d1_request_url()}
            or manifest["concept_id"] != "fontana-del-moro"
            or manifest["source_system"] != "Wikimedia Commons"
            or manifest["source_locator"]
            != "https://commons.wikimedia.org/wiki/Category:Fontana_del_Moro_(Rome)"
        ):
            raise ValueError("D1 source/request contract mismatch")
        derived_rows = _d1_rows(payloads[0])
    elif role == "D2":
        expected_queries = [d2_query(index) for index in range(1, len(payloads) + 1)]
        if (
            request_contract
            != {"method": "POST", "url": _D2_API, "page_queries": expected_queries}
            or manifest["concept_id"] != "tsukioka-kogyo-nogaku-zue"
            or manifest["source_system"] != "Art Institute of Chicago"
            or manifest["source_locator"] != "https://www.artic.edu/open-access"
        ):
            raise ValueError("D2 source/request contract mismatch")
        expected_total: int | None = None
        observed_rows = 0
        for index, payload in enumerate(payloads, start=1):
            pagination = _object(payload.get("pagination"), "AIC pagination")
            total = pagination.get("total")
            total_pages = pagination.get("total_pages")
            limit = pagination.get("limit")
            offset = pagination.get("offset")
            current_page = pagination.get("current_page")
            data = payload.get("data")
            if (
                isinstance(total, bool)
                or not isinstance(total, int)
                or total <= 0
                or isinstance(total_pages, bool)
                or not isinstance(total_pages, int)
                or total_pages != len(payloads)
                or isinstance(limit, bool)
                or not isinstance(limit, int)
                or limit != _D2_PAGE_LIMIT
                or isinstance(offset, bool)
                or not isinstance(offset, int)
                or offset != (index - 1) * _D2_PAGE_LIMIT
                or isinstance(current_page, bool)
                or not isinstance(current_page, int)
                or current_page != index
                or not isinstance(data, list)
                or (index < len(payloads) and len(data) != _D2_PAGE_LIMIT)
                or len(data) > _D2_PAGE_LIMIT
            ):
                raise ValueError("D2 frozen pagination is incomplete or inconsistent")
            if expected_total is None:
                expected_total = total
            elif expected_total != total:
                raise ValueError("D2 total changed across frozen response pages")
            observed_rows += len(data)
        if observed_rows != expected_total:
            raise ValueError("D2 frozen pagination does not cover its declared total")
        derived_rows = _d2_rows(payloads)
    else:
        raise ValueError("harvest role must be D1 or D2")
    if manifest["candidate_rows"] != derived_rows:
        raise ValueError("candidate rows do not rederive from frozen raw responses")
    eligible = sum(row["eligibility"]["passed"] is True for row in derived_rows)
    expected_counts = {
        "observed": len(derived_rows),
        "eligible": eligible,
        "rejected": len(derived_rows) - eligible,
    }
    if manifest["counts"] != expected_counts:
        raise ValueError("harvest counts do not rederive")
    minimum = source_policy["profiles"][role]["minimum_machine_eligible_rows"]
    if eligible < minimum:
        raise ValueError("harvest has an insufficient machine-eligible candidate pool")
    if role == "D1":
        creator_counts: dict[str, int] = {}
        for row in derived_rows:
            if not row["eligibility"]["passed"]:
                continue
            creator = row.get("creator_id_hint")
            if not isinstance(creator, str) or creator == "text:unknown":
                raise ValueError("D1 harvest has an unresolved creator identity")
            creator_counts[creator] = creator_counts.get(creator, 0) + 1
        selected_capacity = sum(min(count, 3) for count in creator_counts.values())
        if selected_capacity < 42:
            raise ValueError("D1 creator-cap capacity cannot fill the 18/24 split")
    else:
        eligible_components = {
            row["automatic_play_component_id"]
            for row in derived_rows
            if row["eligibility"]["passed"]
        }
        component_minimum = source_policy["profiles"][role][
            "minimum_machine_eligible_components"
        ]
        if len(eligible_components) < component_minimum:
            raise ValueError("D2 harvest has too few eligible play components")
    return manifest


def _validated_download_url(url: Any, role: str) -> str:
    url = _text(url, "download_url")
    parsed = urlsplit(url)
    policy, _ = _source_policy()
    allowed = set(policy["profiles"][role]["allowed_download_hosts"])
    if (
        parsed.scheme != "https"
        or parsed.hostname not in allowed
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("download URL is outside the role-specific HTTPS allowlist")
    return url


def _source_enrichment_identity(
    harvest_dir: Path, enrichment_dir: Path, role: str
) -> dict[str, str]:
    if role == "D1":
        try:
            from . import krea_commons_enrichment as enrichment_module
        except ImportError:  # pragma: no cover - direct script execution.
            import krea_commons_enrichment as enrichment_module  # type: ignore[no-redef]
    elif role == "D2":
        try:
            from . import krea_aic_enrichment as enrichment_module
        except ImportError:  # pragma: no cover - direct script execution.
            import krea_aic_enrichment as enrichment_module  # type: ignore[no-redef]
    else:
        raise ValueError("source enrichment role must be D1 or D2")
    manifest = enrichment_module.validate_enrichment(harvest_dir, enrichment_dir)
    manifest_path = _safe_file(
        enrichment_dir / "enrichment.json", "source enrichment manifest"
    )
    semantic_digest = manifest.get("enrichment_sha256")
    if not isinstance(semantic_digest, str) or not _SHA256.fullmatch(semantic_digest):
        raise ValueError("source enrichment lacks a canonical semantic digest")
    return {
        "kind": manifest["kind"],
        "enrichment_sha256": semantic_digest,
        "manifest_file_sha256": _file_sha256(manifest_path),
    }


def _download_one(
    *,
    url: str,
    destination: Path,
    expected_mime: str,
    maximum_bytes: int,
) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    temporary: Path | None = None
    try:
        with urlopen(request, timeout=120) as response:  # noqa: S310 - allowlisted caller
            final = urlsplit(response.geturl())
            initial = urlsplit(url)
            if (
                final.scheme != "https"
                or final.hostname != initial.hostname
                or final.port not in {None, 443}
                or final.username is not None
                or final.password is not None
                or final.query
                or final.fragment
            ):
                raise ValueError("image download redirected outside its source host")
            mime = response.headers.get_content_type()
            if mime not in _ALLOWED_IMAGE_MIMES or mime != expected_mime:
                raise ValueError(f"unexpected image MIME: {mime}")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError as exc:
                    raise ValueError("image Content-Length is invalid") from exc
                if declared_length <= 0 or declared_length > maximum_bytes:
                    raise ValueError("image exceeds the per-file byte cap")
            descriptor, raw_temporary = tempfile.mkstemp(
                prefix=".partial-", dir=destination.parent
            )
            temporary = Path(raw_temporary)
            digest = hashlib.sha256()
            legacy_digest = hashlib.sha1()  # noqa: S324 - provider identity is SHA-1.
            total = 0
            try:
                while True:
                    block = response.read(min(1024 * 1024, maximum_bytes + 1 - total))
                    if not block:
                        break
                    total += len(block)
                    if total > maximum_bytes:
                        raise ValueError("image exceeds the per-file byte cap")
                    digest.update(block)
                    legacy_digest.update(block)
                    view = memoryview(block)
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise OSError("short image write")
                        view = view[written:]
                if total <= 0:
                    raise ValueError("image download is empty")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.chmod(temporary, 0o640)
            if destination.exists() or destination.is_symlink():
                raise ValueError("refusing to replace an image output")
            os.replace(temporary, destination)
            temporary = None
            return {
                "bytes": total,
                "sha256": digest.hexdigest(),
                "sha1": legacy_digest.hexdigest(),
                "mime": mime,
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
            }
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def materialize(
    harvest_dir: Path,
    output_dir: Path,
    *,
    retrieved_at_utc: str,
    maximum_total_bytes: int,
    maximum_file_bytes: int,
    retrieval_authorization: Path,
    source_enrichment_dir: Path,
    delay_s: float | None = None,
) -> dict[str, Any]:
    harvest = validate_harvest(harvest_dir)
    source_policy, policy_sha256 = _source_policy()
    if (
        isinstance(maximum_total_bytes, bool)
        or not isinstance(maximum_total_bytes, int)
        or maximum_total_bytes <= 0
        or isinstance(maximum_file_bytes, bool)
        or not isinstance(maximum_file_bytes, int)
        or maximum_file_bytes <= 0
        or maximum_file_bytes > maximum_total_bytes
    ):
        raise ValueError("materialization byte caps must be positive and coherent")
    hard_caps = source_policy["hard_caps"]
    if (
        maximum_total_bytes > hard_caps["maximum_persisted_bytes"]
        or maximum_file_bytes > hard_caps["maximum_file_bytes"]
    ):
        raise ValueError("requested byte caps exceed the frozen source policy")
    role = harvest["experimental_role"]
    enrichment_identity = _source_enrichment_identity(
        harvest_dir, source_enrichment_dir, role
    )
    authorization = validate_retrieval_authorization(
        retrieval_authorization,
        harvest=harvest,
        requested_maximum_bytes=maximum_total_bytes,
    )
    retrieved_at = _canonical_utc(retrieved_at_utc, "retrieved_at_utc")
    if _utc_datetime(
        authorization["authorized_at_utc"], "authorized_at_utc"
    ) > _utc_datetime(retrieved_at, "retrieved_at_utc"):
        raise ValueError("materialization retrieval time predates authorization")
    minimum_delay = 1.0 if role == "D2" else 0.25
    delay = minimum_delay if delay_s is None else delay_s
    if (
        not isinstance(delay, (int, float))
        or isinstance(delay, bool)
        or not math.isfinite(float(delay))
        or delay < minimum_delay
    ):
        raise ValueError(f"{role} download delay must be at least {minimum_delay}s")
    output_dir = _new_directory(output_dir, "materialization output")
    images_dir = output_dir / "images"
    images_dir.mkdir(mode=0o750)
    selected = [row for row in harvest["candidate_rows"] if row["eligibility"]["passed"]]
    materialized_rows: list[dict[str, Any]] = []
    total = 0
    for index, row in enumerate(selected):
        source_id = row["source_id"]
        if not isinstance(source_id, str) or not _SAFE_SOURCE_ID.fullmatch(source_id):
            raise ValueError("candidate source_id is unsafe")
        mime = row["provider_mime"]
        if mime not in _EXTENSIONS:
            raise ValueError("candidate MIME lacks a canonical extension")
        url = _validated_download_url(row["download_url"], role)
        destination = images_dir / (source_id + _EXTENSIONS[mime])
        remaining = maximum_total_bytes - total
        if remaining <= 0:
            raise ValueError("materialization exhausted the total byte cap")
        identity = _download_one(
            url=url,
            destination=destination,
            expected_mime=mime,
            maximum_bytes=min(maximum_file_bytes, remaining),
        )
        total += identity["bytes"]
        materialized_rows.append(
            {
                "source_id": source_id,
                "relative_path": "images/" + destination.name,
                "source_url": url,
                **identity,
            }
        )
        if index + 1 < len(selected):
            time.sleep(delay)
    body = {
        "schema": _SCHEMA,
        "kind": _MATERIALIZATION_KIND,
        "experimental_role": role,
        "concept_id": harvest["concept_id"],
        "source_harvest_sha256": harvest["harvest_sha256"],
        "source_policy_sha256": policy_sha256,
        "source_enrichment": enrichment_identity,
        "retrieval_authorization_sha256": authorization["authorization_sha256"],
        "retrieval_authorization_file_sha256": _file_sha256(
            retrieval_authorization
        ),
        "retrieval_owner_identity": authorization["owner_identity"],
        "retrieved_at_utc": retrieved_at,
        "download_policy": {
            "sequential": True,
            "minimum_delay_s": delay,
            "maximum_total_bytes": maximum_total_bytes,
            "maximum_file_bytes": maximum_file_bytes,
            "actual_total_bytes": total,
        },
        "rows": materialized_rows,
        "admission_state": _UNAPPROVED,
        "claim_limit": _CLAIM_LIMIT,
        "human_gates": source_policy["human_gates"],
        "human_approvals": [],
        "fixture_manifest_created": False,
        "gpu_execution_authorized": False,
    }
    manifest = {**body, "materialization_sha256": _canonical_sha256(body)}
    _canonical_create(output_dir / "materialization.json", manifest)
    validate_materialization(
        harvest_dir,
        output_dir,
        retrieval_authorization=retrieval_authorization,
        source_enrichment_dir=source_enrichment_dir,
    )
    return manifest


def validate_materialization(
    harvest_dir: Path,
    materialization_dir: Path,
    *,
    retrieval_authorization: Path,
    source_enrichment_dir: Path,
) -> dict[str, Any]:
    harvest = validate_harvest(harvest_dir)
    materialization_dir = _safe_existing_directory(
        materialization_dir, "materialization package"
    )
    manifest = _read_canonical(
        materialization_dir / "materialization.json", "materialization manifest"
    )
    expected = {
        "schema",
        "kind",
        "experimental_role",
        "concept_id",
        "source_harvest_sha256",
        "source_policy_sha256",
        "source_enrichment",
        "retrieval_authorization_sha256",
        "retrieval_authorization_file_sha256",
        "retrieval_owner_identity",
        "retrieved_at_utc",
        "download_policy",
        "rows",
        "admission_state",
        "claim_limit",
        "human_gates",
        "human_approvals",
        "fixture_manifest_created",
        "gpu_execution_authorized",
        "materialization_sha256",
    }
    _exact(manifest, expected, "materialization manifest")
    body = {
        key: value for key, value in manifest.items() if key != "materialization_sha256"
    }
    source_policy, policy_sha256 = _source_policy()
    raw_policy = _object(manifest.get("download_policy"), "download_policy")
    requested_cap = raw_policy.get("maximum_total_bytes")
    if isinstance(requested_cap, bool) or not isinstance(requested_cap, int):
        raise ValueError("materialization download cap is invalid")
    authorization = validate_retrieval_authorization(
        retrieval_authorization,
        harvest=harvest,
        requested_maximum_bytes=requested_cap,
    )
    enrichment_identity = _source_enrichment_identity(
        harvest_dir,
        source_enrichment_dir,
        harvest["experimental_role"],
    )
    if (
        manifest["schema"] != _SCHEMA
        or manifest["kind"] != _MATERIALIZATION_KIND
        or manifest["experimental_role"] != harvest["experimental_role"]
        or manifest["concept_id"] != harvest["concept_id"]
        or manifest["source_harvest_sha256"] != harvest["harvest_sha256"]
        or manifest["source_policy_sha256"] != policy_sha256
        or manifest["source_enrichment"] != enrichment_identity
        or manifest["retrieval_authorization_sha256"]
        != authorization["authorization_sha256"]
        or manifest["retrieval_authorization_file_sha256"]
        != _file_sha256(retrieval_authorization)
        or manifest["retrieval_owner_identity"] != authorization["owner_identity"]
        or manifest["admission_state"] != _UNAPPROVED
        or manifest["claim_limit"] != _CLAIM_LIMIT
        or manifest["human_gates"] != source_policy["human_gates"]
        or manifest["human_approvals"] != []
        or manifest["fixture_manifest_created"] is not False
        or manifest["gpu_execution_authorized"] is not False
        or manifest["materialization_sha256"] != _canonical_sha256(body)
    ):
        raise ValueError("materialization is not an intact unapproved candidate record")
    materialized_at = _utc_datetime(
        manifest["retrieved_at_utc"], "retrieved_at_utc"
    )
    if materialized_at < _utc_datetime(
        authorization["authorized_at_utc"], "authorized_at_utc"
    ):
        raise ValueError("materialization record predates retrieval authorization")
    eligible = {
        row["source_id"]: row
        for row in harvest["candidate_rows"]
        if row["eligibility"]["passed"]
    }
    rows = manifest["rows"]
    if not isinstance(rows, list) or len(rows) != len(eligible):
        raise ValueError("materialization must cover every eligible candidate exactly")
    total = 0
    seen: set[str] = set()
    expected_row_keys = {
        "source_id",
        "relative_path",
        "source_url",
        "bytes",
        "sha256",
        "sha1",
        "mime",
        "etag",
        "last_modified",
    }
    for row in rows:
        row = _object(row, "materialized row")
        _exact(row, expected_row_keys, "materialized row")
        source_id = row["source_id"]
        if source_id in seen or source_id not in eligible:
            raise ValueError("materialization has a duplicate or unknown source ID")
        seen.add(source_id)
        expected_extension = _EXTENSIONS[eligible[source_id]["provider_mime"]]
        expected_relative = f"images/{source_id}{expected_extension}"
        if (
            row["relative_path"] != expected_relative
            or row["source_url"]
            != _validated_download_url(
                eligible[source_id]["download_url"],
                manifest["experimental_role"],
            )
            or row["mime"] != eligible[source_id]["provider_mime"]
        ):
            raise ValueError("materialized row does not match its source candidate")
        path = _safe_file(materialization_dir / expected_relative, "materialized image")
        if (
            isinstance(row["bytes"], bool)
            or not isinstance(row["bytes"], int)
            or row["bytes"] != path.stat().st_size
            or not isinstance(row["sha256"], str)
            or not _SHA256.fullmatch(row["sha256"])
            or row["sha256"] != _file_sha256(path)
            or not isinstance(row["sha1"], str)
            or not _SHA1.fullmatch(row["sha1"])
            or row["sha1"] != _file_sha1(path)
        ):
            raise ValueError("materialized image identity mismatch")
        provider_sha1 = eligible[source_id].get("provider_content_sha1")
        if provider_sha1 is not None and row["sha1"] != provider_sha1:
            raise ValueError("materialized image differs from the provider SHA-1")
        total += row["bytes"]
    download_policy = _object(manifest["download_policy"], "download_policy")
    _exact(
        download_policy,
        {
            "sequential",
            "minimum_delay_s",
            "maximum_total_bytes",
            "maximum_file_bytes",
            "actual_total_bytes",
        },
        "download_policy",
    )
    if (
        download_policy["sequential"] is not True
        or isinstance(download_policy["minimum_delay_s"], bool)
        or not isinstance(download_policy["minimum_delay_s"], (int, float))
        or not math.isfinite(float(download_policy["minimum_delay_s"]))
        or download_policy["minimum_delay_s"]
        < (1.0 if manifest["experimental_role"] == "D2" else 0.25)
        or isinstance(download_policy["maximum_total_bytes"], bool)
        or not isinstance(download_policy["maximum_total_bytes"], int)
        or download_policy["maximum_total_bytes"] <= 0
        or download_policy["maximum_total_bytes"]
        > source_policy["hard_caps"]["maximum_persisted_bytes"]
        or isinstance(download_policy["maximum_file_bytes"], bool)
        or not isinstance(download_policy["maximum_file_bytes"], int)
        or download_policy["maximum_file_bytes"] <= 0
        or download_policy["maximum_file_bytes"]
        > source_policy["hard_caps"]["maximum_file_bytes"]
        or download_policy["maximum_file_bytes"]
        > download_policy["maximum_total_bytes"]
        or download_policy["actual_total_bytes"] != total
        or total > download_policy["maximum_total_bytes"]
        or any(row["bytes"] > download_policy["maximum_file_bytes"] for row in rows)
    ):
        raise ValueError("materialization violates its byte/rate policy")
    images_dir = _safe_existing_directory(
        materialization_dir / "images", "images directory"
    )
    root_entries = sorted(path.name for path in materialization_dir.iterdir())
    if root_entries != ["images", "materialization.json"]:
        raise ValueError("materialization namespace has unexpected root entries")
    image_entries = list(images_dir.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in image_entries):
        raise ValueError("materialization image directory has unsafe entries")
    actual_files = sorted(path.name for path in image_entries)
    expected_files = sorted(Path(row["relative_path"]).name for row in rows)
    if actual_files != expected_files:
        raise ValueError("materialization image directory has missing or unexpected files")
    return manifest


def _average_hash64(image: Any) -> str:
    from PIL import Image

    reduced = image.convert("L").resize((8, 8), resample=Image.Resampling.BILINEAR)
    values = list(reduced.getdata())
    mean = sum(values) / len(values)
    bits = 0
    for value in values:
        bits = (bits << 1) | int(value >= mean)
    return f"{bits:016x}"


def _union_clusters(rows: list[dict[str, Any]], threshold: int) -> list[list[str]]:
    parents = {row["source_id"]: row["source_id"] for row in rows}

    def find(item: str) -> str:
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)

    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            distance = (
                int(left["perceptual_hash64"], 16)
                ^ int(right["perceptual_hash64"], 16)
            ).bit_count()
            if distance <= threshold:
                union(left["source_id"], right["source_id"])
    clusters: dict[str, list[str]] = {}
    for source_id in parents:
        clusters.setdefault(find(source_id), []).append(source_id)
    return sorted(
        (sorted(members) for members in clusters.values() if len(members) > 1),
        key=lambda members: members[0],
    )


def _derive_inspection_body(
    harvest_dir: Path,
    materialization_dir: Path,
    *,
    retrieval_authorization: Path,
    source_enrichment_dir: Path,
    inspected_at_utc: str,
    cluster_hamming_threshold: int = 6,
    review_hamming_threshold: int = 10,
) -> dict[str, Any]:
    harvest = validate_harvest(harvest_dir)
    materialization = validate_materialization(
        harvest_dir,
        materialization_dir,
        retrieval_authorization=retrieval_authorization,
        source_enrichment_dir=source_enrichment_dir,
    )
    inspected_at = _canonical_utc(inspected_at_utc, "inspected_at_utc")
    if _utc_datetime(inspected_at, "inspected_at_utc") < _utc_datetime(
        materialization["retrieved_at_utc"], "materialization.retrieved_at_utc"
    ):
        raise ValueError("inspection predates materialization")
    selection_policy, selection_policy_sha256 = _selection_policy()
    frozen_screen = selection_policy["perceptual_screen"]
    if (
        isinstance(cluster_hamming_threshold, bool)
        or not isinstance(cluster_hamming_threshold, int)
        or isinstance(review_hamming_threshold, bool)
        or not isinstance(review_hamming_threshold, int)
        or not 0 <= cluster_hamming_threshold <= review_hamming_threshold <= 64
        or cluster_hamming_threshold
        != frozen_screen["automatic_union_maximum_hamming_distance"]
        or review_hamming_threshold
        != frozen_screen["human_review_queue_maximum_hamming_distance"]
    ):
        raise ValueError("inspection thresholds must equal the frozen 6/10 policy")
    source_policy, source_policy_sha256 = _source_policy()
    maximum_decoded_pixels = source_policy["hard_caps"]["maximum_decoded_pixels"]
    if maximum_decoded_pixels != _MAXIMUM_DECODED_PIXELS:
        raise ValueError("decoded-pixel cap differs from the frozen implementation")
    try:
        from PIL import (
            Image,
            ImageOps,
            UnidentifiedImageError,
            __version__ as pillow_version,
        )
    except ImportError as exc:  # pragma: no cover - workspace runtime has Pillow.
        raise RuntimeError("Pillow is required for candidate inspection") from exc
    materialization_dir = _safe_existing_directory(
        materialization_dir, "materialization package"
    )
    source_by_id = {row["source_id"]: row for row in harvest["candidate_rows"]}
    rows: list[dict[str, Any]] = []
    for materialized in materialization["rows"]:
        path = _safe_file(
            materialization_dir / materialized["relative_path"], "candidate image"
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as opened:
                    if getattr(opened, "n_frames", 1) != 1:
                        raise ValueError("multi-frame images are not accepted")
                    if opened.width * opened.height > maximum_decoded_pixels:
                        raise ValueError("decoded image exceeds the frozen pixel cap")
                    opened.verify()
                with Image.open(path) as opened:
                    image_format = opened.format
                    if image_format not in {"JPEG", "PNG", "WEBP", "TIFF"}:
                        raise ValueError("decoded image format is unsupported")
                    oriented = ImageOps.exif_transpose(opened)
                    rgb = oriented.convert("RGB")
                    width, height = rgb.size
                    if width * height > maximum_decoded_pixels:
                        raise ValueError("decoded image exceeds the frozen pixel cap")
                    if width < 768 or height < 768:
                        raise ValueError("decoded image is below 768 on one or both axes")
                    expected_format = {
                        "image/jpeg": "JPEG",
                        "image/png": "PNG",
                        "image/webp": "WEBP",
                        "image/tiff": "TIFF",
                    }[materialized["mime"]]
                    if image_format != expected_format:
                        raise ValueError("decoded format contradicts the bound MIME")
                    pixels = rgb.tobytes()
                    perceptual = _average_hash64(rgb)
        except (
            Image.DecompressionBombWarning,
            Image.DecompressionBombError,
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:
            raise ValueError(f"candidate image is not safely decodable: {path.name}") from exc
        source = source_by_id[materialized["source_id"]]
        rows.append(
            {
                "source_id": materialized["source_id"],
                "relative_path": materialized["relative_path"],
                "byte_sha256": materialized["sha256"],
                "decoded_rgb_sha256": hashlib.sha256(pixels).hexdigest(),
                "width": width,
                "height": height,
                "mode": "RGB",
                "decoder": {"library": "Pillow", "version": pillow_version},
                "decoded_format": image_format,
                "perceptual_hash64": perceptual,
                "provider_title": source["title"],
                "automatic_group_hints": {
                    "creator_id": source.get("creator_id_hint"),
                    "burst_id": source.get("burst_id_hint"),
                    "play_root": source.get("normalized_play_root"),
                    "play_component": source.get("automatic_play_component_id"),
                    "accession_family": source.get("accession_family_key"),
                },
                "human_qc_state": "pending",
                "human_similarity_cluster_id": None,
                "draft_caption": None,
                "proposed_split": None,
            }
        )
    rows.sort(key=lambda row: row["source_id"])
    exact_pixel_groups: dict[str, list[str]] = {}
    review_pairs: list[dict[str, Any]] = []
    for row in rows:
        exact_pixel_groups.setdefault(row["decoded_rgb_sha256"], []).append(row["source_id"])
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            distance = (
                int(left["perceptual_hash64"], 16)
                ^ int(right["perceptual_hash64"], 16)
            ).bit_count()
            same_play = bool(
                left["automatic_group_hints"]["play_component"]
                and left["automatic_group_hints"]["play_component"]
                == right["automatic_group_hints"]["play_component"]
            )
            if distance <= review_hamming_threshold or same_play:
                review_pairs.append(
                    {
                        "left": left["source_id"],
                        "right": right["source_id"],
                        "hamming_distance": distance,
                        "same_play_component": same_play,
                    }
                )
    exact_duplicates = sorted(
        (sorted(members) for members in exact_pixel_groups.values() if len(members) > 1),
        key=lambda members: members[0],
    )
    automatic_clusters = _union_clusters(rows, cluster_hamming_threshold)
    return {
        "schema": _SCHEMA,
        "kind": _INSPECTION_KIND,
        "experimental_role": harvest["experimental_role"],
        "concept_id": harvest["concept_id"],
        "source_harvest_sha256": harvest["harvest_sha256"],
        "source_policy_sha256": source_policy_sha256,
        "selection_policy_sha256": selection_policy_sha256,
        "materialization_sha256": materialization["materialization_sha256"],
        "inspected_at_utc": inspected_at,
        "tool_identity": {
            "source_sha256": _file_sha256(Path(__file__).resolve(strict=True)),
            "perceptual_hash": selection_policy["perceptual_screen"]["algorithm"],
            "cluster_maximum_hamming_distance": cluster_hamming_threshold,
            "human_review_queue_maximum_hamming_distance": review_hamming_threshold,
            "maximum_decoded_pixels": maximum_decoded_pixels,
        },
        "rows": rows,
        "machine_qc": {
            "decoded_count": len(rows),
            "exact_normalized_pixel_duplicate_groups": exact_duplicates,
            "automatic_perceptual_clusters": automatic_clusters,
            "human_review_queue": review_pairs,
            "human_review_queue_count": len(review_pairs),
        },
        "admission_state": _UNAPPROVED,
        "claim_limit": _CLAIM_LIMIT,
        "human_gates": _HUMAN_GATES,
        "human_review_required": True,
        "human_approvals": [],
        "fixture_manifest_created": False,
        "gpu_execution_authorized": False,
    }


def inspect_candidates(
    harvest_dir: Path,
    materialization_dir: Path,
    output_path: Path,
    *,
    retrieval_authorization: Path,
    source_enrichment_dir: Path,
    inspected_at_utc: str,
    cluster_hamming_threshold: int = 6,
    review_hamming_threshold: int = 10,
) -> dict[str, Any]:
    body = _derive_inspection_body(
        harvest_dir,
        materialization_dir,
        retrieval_authorization=retrieval_authorization,
        source_enrichment_dir=source_enrichment_dir,
        inspected_at_utc=inspected_at_utc,
        cluster_hamming_threshold=cluster_hamming_threshold,
        review_hamming_threshold=review_hamming_threshold,
    )
    record = {**body, "inspection_sha256": _canonical_sha256(body)}
    _canonical_create(output_path, record)
    validate_inspection(
        harvest_dir,
        materialization_dir,
        output_path,
        retrieval_authorization=retrieval_authorization,
        source_enrichment_dir=source_enrichment_dir,
    )
    return record


def validate_inspection(
    harvest_dir: Path,
    materialization_dir: Path,
    inspection_path: Path,
    *,
    retrieval_authorization: Path,
    source_enrichment_dir: Path,
) -> dict[str, Any]:
    record = _read_canonical(inspection_path, "candidate inspection")
    expected_keys = {
        "schema",
        "kind",
        "experimental_role",
        "concept_id",
        "source_harvest_sha256",
        "source_policy_sha256",
        "selection_policy_sha256",
        "materialization_sha256",
        "inspected_at_utc",
        "tool_identity",
        "rows",
        "machine_qc",
        "admission_state",
        "claim_limit",
        "human_gates",
        "human_review_required",
        "human_approvals",
        "fixture_manifest_created",
        "gpu_execution_authorized",
        "inspection_sha256",
    }
    _exact(record, expected_keys, "candidate inspection")
    body = {
        key: value for key, value in record.items() if key != "inspection_sha256"
    }
    if record["inspection_sha256"] != _canonical_sha256(body):
        raise ValueError("candidate inspection semantic digest mismatch")
    tool_identity = _object(record["tool_identity"], "inspection tool identity")
    _exact(
        tool_identity,
        {
            "source_sha256",
            "perceptual_hash",
            "cluster_maximum_hamming_distance",
            "human_review_queue_maximum_hamming_distance",
            "maximum_decoded_pixels",
        },
        "inspection tool identity",
    )
    derived = _derive_inspection_body(
        harvest_dir,
        materialization_dir,
        retrieval_authorization=retrieval_authorization,
        source_enrichment_dir=source_enrichment_dir,
        inspected_at_utc=record["inspected_at_utc"],
        cluster_hamming_threshold=tool_identity[
            "cluster_maximum_hamming_distance"
        ],
        review_hamming_threshold=tool_identity[
            "human_review_queue_maximum_hamming_distance"
        ],
    )
    if body != derived:
        raise ValueError("candidate inspection does not rederive from bound image bytes")
    return record


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("harvest-d1", "harvest-d2"):
        command = commands.add_parser(name)
        command.add_argument("--output-dir", required=True, type=Path)
        command.add_argument("--retrieved-at-utc", required=True)
    validate = commands.add_parser("validate-harvest")
    validate.add_argument("--package-dir", required=True, type=Path)
    materialize_command = commands.add_parser("materialize")
    materialize_command.add_argument("--harvest-dir", required=True, type=Path)
    materialize_command.add_argument("--output-dir", required=True, type=Path)
    materialize_command.add_argument("--retrieved-at-utc", required=True)
    materialize_command.add_argument("--maximum-total-bytes", type=int, required=True)
    materialize_command.add_argument("--maximum-file-bytes", type=int, required=True)
    materialize_command.add_argument(
        "--retrieval-authorization", required=True, type=Path
    )
    materialize_command.add_argument("--source-enrichment", required=True, type=Path)
    materialize_command.add_argument("--delay-s", type=float)
    validate_materialized = commands.add_parser("validate-materialization")
    validate_materialized.add_argument("--harvest-dir", required=True, type=Path)
    validate_materialized.add_argument("--materialization-dir", required=True, type=Path)
    validate_materialized.add_argument(
        "--retrieval-authorization", required=True, type=Path
    )
    validate_materialized.add_argument(
        "--source-enrichment", required=True, type=Path
    )
    inspect_command = commands.add_parser("inspect")
    inspect_command.add_argument("--harvest-dir", required=True, type=Path)
    inspect_command.add_argument("--materialization-dir", required=True, type=Path)
    inspect_command.add_argument("--output", required=True, type=Path)
    inspect_command.add_argument("--retrieval-authorization", required=True, type=Path)
    inspect_command.add_argument("--source-enrichment", required=True, type=Path)
    inspect_command.add_argument("--inspected-at-utc", required=True)
    inspect_command.add_argument("--cluster-hamming-threshold", type=int, default=6)
    inspect_command.add_argument("--review-hamming-threshold", type=int, default=10)
    validate_inspection_command = commands.add_parser("validate-inspection")
    validate_inspection_command.add_argument("--harvest-dir", required=True, type=Path)
    validate_inspection_command.add_argument(
        "--materialization-dir", required=True, type=Path
    )
    validate_inspection_command.add_argument("--inspection", required=True, type=Path)
    validate_inspection_command.add_argument(
        "--retrieval-authorization", required=True, type=Path
    )
    validate_inspection_command.add_argument(
        "--source-enrichment", required=True, type=Path
    )
    return parser.parse_args()


def main() -> int:
    args = _parse()
    if args.command == "harvest-d1":
        result = harvest_d1(args.output_dir, retrieved_at_utc=args.retrieved_at_utc)
    elif args.command == "harvest-d2":
        result = harvest_d2(args.output_dir, retrieved_at_utc=args.retrieved_at_utc)
    elif args.command == "validate-harvest":
        result = validate_harvest(args.package_dir)
    elif args.command == "materialize":
        result = materialize(
            args.harvest_dir,
            args.output_dir,
            retrieved_at_utc=args.retrieved_at_utc,
            maximum_total_bytes=args.maximum_total_bytes,
            maximum_file_bytes=args.maximum_file_bytes,
            retrieval_authorization=args.retrieval_authorization,
            source_enrichment_dir=args.source_enrichment,
            delay_s=args.delay_s,
        )
    elif args.command == "validate-materialization":
        result = validate_materialization(
            args.harvest_dir,
            args.materialization_dir,
            retrieval_authorization=args.retrieval_authorization,
            source_enrichment_dir=args.source_enrichment,
        )
    elif args.command == "inspect":
        result = inspect_candidates(
            args.harvest_dir,
            args.materialization_dir,
            args.output,
            retrieval_authorization=args.retrieval_authorization,
            source_enrichment_dir=args.source_enrichment,
            inspected_at_utc=args.inspected_at_utc,
            cluster_hamming_threshold=args.cluster_hamming_threshold,
            review_hamming_threshold=args.review_hamming_threshold,
        )
    elif args.command == "validate-inspection":
        result = validate_inspection(
            args.harvest_dir,
            args.materialization_dir,
            args.inspection,
            retrieval_authorization=args.retrieval_authorization,
            source_enrichment_dir=args.source_enrichment,
        )
    else:  # pragma: no cover - argparse enforces this.
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess smoke.
    raise SystemExit(main())
