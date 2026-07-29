"""Offline adversarial tests for D2 AIC selected-object/IIIF enrichment."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any

import pytest


_CALIBRATION = Path(__file__).parents[1] / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))

import krea_aic_enrichment as enrichment  # noqa: E402
import krea_provenance as provenance  # noqa: E402
import krea_source_curation as curation  # noqa: E402


_AT = "2026-07-28T00:00:00Z"
_INFO = {
    "license_text": (
        "The `description` field in this response is licensed under a Creative "
        "Commons Attribution 4.0 Generic License (CC-By) and the Terms and "
        "Conditions of artic.edu. All other data in this response is licensed "
        "under a Creative Commons Zero (CC0) 1.0 designation and the Terms and "
        "Conditions of artic.edu."
    ),
    "license_links": [
        "https://creativecommons.org/publicdomain/zero/1.0/",
        "https://www.artic.edu/terms",
    ],
    "version": "1.14",
}
_CONFIG = {
    "iiif_url": "https://www.artic.edu/iiif/2",
    "website_url": "http://www.artic.edu",
}


def _object(object_id: int) -> dict[str, Any]:
    image_id = f"00000000-0000-0000-0000-{object_id:012d}"
    values: dict[str, Any] = {
        "id": object_id,
        "api_model": "artworks",
        "api_link": f"https://api.artic.edu/api/v1/artworks/{object_id}",
        "title": (
            f'Play {object_id}, from the series "Pictures of No Performances '
            '(Nogaku Zue)"'
        ),
        "alt_titles": None,
        "thumbnail": {"width": 3000, "height": 2000, "alt_text": "A print."},
        "main_reference_number": f"1939.2258.{object_id}",
        "date_start": 1893,
        "date_end": 1903,
        "date_display": "1898",
        "artist_id": 26646,
        "artist_title": "Tsukioka Kôgyo",
        "artist_display": "Tsukioka Kogyo\nJapanese, 1869-1927",
        "place_of_origin": "Japan",
        "dimensions": "25 × 37 cm",
        "medium_display": "Color woodblock print",
        "inscriptions": None,
        "credit_line": "Frederick W. Gookin Collection",
        "publication_history": None,
        "provenance_text": None,
        "is_public_domain": True,
        "copyright_notice": None,
        "department_title": "Arts of Asia",
        "classification_title": "woodblock print",
        "classification_titles": ["woodblock print", "print", "asian art"],
        "image_id": image_id,
        "alt_image_ids": [],
        "updated_at": "2026-03-22T21:38:09-05:00",
    }
    assert set(values) == set(enrichment._API_FIELDS)
    return values


def _search_payload(
    objects: list[dict[str, Any]],
    *,
    pagination_override: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pagination = {
        "total": len(objects),
        "limit": 100,
        "offset": 0,
        "total_pages": 1,
        "current_page": 1,
    }
    pagination.update(pagination_override or {})
    payload: dict[str, Any] = {
        "preference": None,
        "pagination": pagination,
        "data": [{"_score": None, **item} for item in objects],
        "info": _INFO,
        "config": _CONFIG,
    }
    payload.update(extra or {})
    return payload


def _canonical(path: Path, value: object) -> None:
    path.write_bytes(provenance.canonical_bytes(value) + b"\n")


def _harvest(
    tmp_path: Path,
    *,
    pagination_override: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[Path, dict[int, dict[str, Any]]]:
    package = tmp_path / "harvest"
    package.mkdir(parents=True)
    objects = [_object(200000 + index) for index in range(88)]
    payload = _search_payload(
        objects, pagination_override=pagination_override, extra=extra
    )
    raw = provenance.canonical_bytes(payload)
    (package / "source-response-001.json").write_bytes(raw)
    manifest = curation._harvest_body(
        role="D2",
        concept_id="tsukioka-kogyo-nogaku-zue",
        source_system="Art Institute of Chicago",
        source_locator="https://www.artic.edu/open-access",
        retrieved_at_utc=_AT,
        request_contract={
            "method": "POST",
            "url": curation._D2_API,
            "page_queries": [curation.d2_query(1)],
        },
        raw_responses=[curation._raw_record("source-response-001.json", raw)],
        rows=curation._d2_rows([payload]),
    )
    _canonical(package / "harvest.json", manifest)
    return package, {item["id"]: item for item in objects}


def _iiif(image_id: str, *, width: int = 3000, height: int = 2000) -> dict[str, Any]:
    return {
        "@context": "http://iiif.io/api/image/2/context.json",
        "@id": f"https://www.artic.edu/iiif/2/{image_id}",
        "protocol": "http://iiif.io/api/image",
        "width": width,
        "height": height,
        "sizes": [
            {"width": 750, "height": 500},
            {"width": 1500, "height": 1000},
            {"width": width, "height": height},
        ],
        "tiles": [
            {
                "width": 256,
                "height": 256,
                "scaleFactors": [1, 2, 4, 8, 16, 32],
            }
        ],
        "profile": [
            "http://iiif.io/api/image/2/level1.json",
            {
                "formats": ["jpg"],
                "maxArea": width * height,
                "qualities": ["default", "gray"],
                "supports": ["sizeByW", "cors"],
            },
        ],
    }


class _FakeAIC:
    def __init__(self, objects: dict[int, dict[str, Any]]) -> None:
        self.objects = objects
        self.urls: list[str] = []
        self.mutate_artwork: callable | None = None
        self.mutate_iiif: callable | None = None

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        if url.startswith("https://api.artic.edu/api/v1/artworks/"):
            match = re.search(r"/artworks/(\d+)\?", url)
            assert match is not None
            item = json.loads(json.dumps(self.objects[int(match.group(1))]))
            payload = {"data": item, "info": _INFO, "config": _CONFIG}
            if self.mutate_artwork is not None:
                self.mutate_artwork(payload)
            return provenance.canonical_bytes(payload)
        match = re.fullmatch(
            r"https://www\.artic\.edu/iiif/2/([0-9a-f-]+)/info\.json", url
        )
        assert match is not None
        payload = _iiif(match.group(1))
        if self.mutate_iiif is not None:
            self.mutate_iiif(payload)
        return provenance.canonical_bytes(payload)


def _enrich(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any], _FakeAIC, list[float]]:
    harvest, objects = _harvest(tmp_path)
    output = tmp_path / "enrichment"
    fake = _FakeAIC(objects)
    delays: list[float] = []
    manifest = enrichment.enrich(
        harvest,
        output,
        retrieved_at_utc=_AT,
        fetcher=fake,
        sleeper=delays.append,
    )
    return harvest, output, manifest, fake, delays


def test_enrichment_freezes_exact_complete_pending_metadata_only_package(
    tmp_path: Path,
) -> None:
    harvest, output, manifest, fake, delays = _enrich(tmp_path)
    assert enrichment.validate_enrichment(harvest, output) == manifest
    assert manifest["coverage"] == {
        "eligible_count": 88,
        "enriched_count": 88,
        "eligible_source_ids_sha256": provenance.canonical_sha256(
            [f"aic-{200000 + index}" for index in range(88)]
        ),
        "missing_source_ids": [],
        "unexpected_source_ids": [],
        "duplicate_source_ids": [],
        "duplicate_image_ids": [],
    }
    assert len(fake.urls) == 176
    assert len(delays) == 175
    assert set(delays) == {1.0}
    assert manifest["request_policy"]["sequential"] is True
    assert manifest["derivative_policy"]["width"] == 1686
    assert manifest["jpeg_derivatives_downloaded"] is False
    assert manifest["human_gates"] == enrichment._HUMAN_GATES
    assert manifest["human_approvals"] == []
    assert manifest["gpu_execution_authorized"] is False
    assert not list(output.rglob("*.jpg"))


def test_selected_object_must_match_every_frozen_harvest_field(tmp_path: Path) -> None:
    harvest, objects = _harvest(tmp_path)
    fake = _FakeAIC(objects)
    fake.mutate_artwork = lambda payload: payload["data"].update(title="Changed")
    with pytest.raises(ValueError, match="differs from the frozen search harvest"):
        enrichment.enrich(
            harvest,
            tmp_path / "enrichment",
            retrieved_at_utc=_AT,
            fetcher=fake,
            sleeper=lambda _: None,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(width=2999),
        lambda payload: payload.update(**{"@id": "https://evil.example/image"}),
        lambda payload: payload["profile"][1]["supports"].remove("sizeByW"),
        lambda payload: payload.update(next="continuation-is-not-allowed"),
    ],
)
def test_iiif_identity_capability_and_continuation_fail_closed(
    tmp_path: Path, mutation: Any
) -> None:
    harvest, objects = _harvest(tmp_path)
    fake = _FakeAIC(objects)
    fake.mutate_iiif = mutation
    with pytest.raises(ValueError, match="IIIF"):
        enrichment.enrich(
            harvest,
            tmp_path / "enrichment",
            retrieved_at_utc=_AT,
            fetcher=fake,
            sleeper=lambda _: None,
        )


def test_api_license_and_config_blocks_must_match_search_harvest(
    tmp_path: Path,
) -> None:
    harvest, objects = _harvest(tmp_path)
    fake = _FakeAIC(objects)
    fake.mutate_artwork = lambda payload: payload["info"].update(version="9.9")
    with pytest.raises(ValueError, match="API blocks differ"):
        enrichment.enrich(
            harvest,
            tmp_path / "enrichment",
            retrieved_at_utc=_AT,
            fetcher=fake,
            sleeper=lambda _: None,
        )


@pytest.mark.parametrize(
    ("pagination_override", "extra", "match"),
    [
        ({"current_page": 2}, None, "pagination"),
        ({"total": 89}, None, "pagination"),
        (None, {"next_url": "https://api.artic.edu/next"}, "keys mismatch"),
    ],
)
def test_search_pagination_coverage_and_continuation_fail_closed(
    tmp_path: Path,
    pagination_override: dict[str, Any] | None,
    extra: dict[str, Any] | None,
    match: str,
) -> None:
    harvest, _ = _harvest(
        tmp_path, pagination_override=pagination_override, extra=extra
    )
    with pytest.raises(ValueError, match=match):
        enrichment._load_d2_harvest(harvest)


def test_raw_tamper_and_self_asserted_approval_are_rejected(tmp_path: Path) -> None:
    harvest, output, _, _, _ = _enrich(tmp_path)
    first = output / "responses" / "artwork-000001-aic-200000.json"
    first.write_bytes(first.read_bytes() + b" ")
    with pytest.raises(ValueError, match="raw-response identity mismatch"):
        enrichment.validate_enrichment(harvest, output)

    other = tmp_path / "other"
    harvest2, output2, manifest, _, _ = _enrich(other)
    manifest["rights_approved"] = True
    body = {key: value for key, value in manifest.items() if key != "enrichment_sha256"}
    manifest["enrichment_sha256"] = provenance.canonical_sha256(body)
    _canonical(output2 / "enrichment.json", manifest)
    with pytest.raises(ValueError, match="unapproved metadata-only"):
        enrichment.validate_enrichment(harvest2, output2)


def test_namespace_delay_and_host_contracts_are_fail_closed(tmp_path: Path) -> None:
    harvest, objects = _harvest(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="must not already exist"):
        enrichment.enrich(
            harvest,
            existing,
            retrieved_at_utc=_AT,
            fetcher=_FakeAIC(objects),
            sleeper=lambda _: None,
        )
    with pytest.raises(ValueError, match="at least 1.0s"):
        enrichment.enrich(
            harvest,
            tmp_path / "new",
            retrieved_at_utc=_AT,
            delay_s=0.999,
            fetcher=_FakeAIC(objects),
            sleeper=lambda _: None,
        )
    with pytest.raises(ValueError, match="exact HTTPS host contract"):
        enrichment._validate_request_url(
            "https://evil.example/info.json", host="www.artic.edu", query_allowed=False
        )
