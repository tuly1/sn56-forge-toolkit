"""Offline adversarial tests for pending D1 Commons enrichment evidence."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

import pytest


_CALIBRATION = Path(__file__).parents[1] / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))

import krea_commons_enrichment as enrichment  # noqa: E402
import krea_provenance as provenance  # noqa: E402
import krea_source_curation as source_curation  # noqa: E402


_AT = "2026-07-28T00:00:00Z"


def _ext(value: str) -> dict[str, str]:
    return {"value": value}


def _source_ext(page_id: int) -> dict[str, dict[str, str]]:
    return {
        "LicenseShortName": _ext("CC BY 4.0"),
        "LicenseUrl": _ext("https://creativecommons.org/licenses/by/4.0"),
        "UsageTerms": _ext("Creative Commons Attribution 4.0"),
        "Artist": _ext(f"Photographer {page_id}"),
        "Credit": _ext("Own work"),
        "Attribution": _ext(f"Photographer {page_id}"),
    }


def _source_page(page_id: int) -> dict:
    return {
        "pageid": page_id,
        "title": f"File:Fontana-{page_id}.jpg",
        "revisions": [{"revid": page_id + 1000, "timestamp": "2026-07-27T00:00:00Z"}],
        "imageinfo": [
            {
                "url": f"https://upload.wikimedia.org/test/{page_id}.jpg",
                "sha1": f"{page_id:040x}"[-40:],
                "size": 100_000 + page_id,
                "width": 1600,
                "height": 1200,
                "mime": "image/jpeg",
                "extmetadata": _source_ext(page_id),
            }
        ],
    }


def _canonical(path: Path, value: object) -> None:
    path.write_bytes(provenance.canonical_bytes(value) + b"\n")


def _harvest_package(tmp_path: Path, count: int = 50) -> Path:
    package = tmp_path / "harvest"
    package.mkdir()
    payload = {
        "batchcomplete": True,
        "query": {"pages": [_source_page(index) for index in range(1, count + 1)]},
    }
    raw = json.dumps(payload).encode("utf-8")
    (package / "source-response-001.json").write_bytes(raw)
    manifest = source_curation._harvest_body(
        role="D1",
        concept_id="fontana-del-moro",
        source_system="Wikimedia Commons",
        source_locator=(
            "https://commons.wikimedia.org/wiki/Category:Fontana_del_Moro_(Rome)"
        ),
        retrieved_at_utc=_AT,
        request_contract={"method": "GET", "url": source_curation.d1_request_url()},
        raw_responses=[source_curation._raw_record("source-response-001.json", raw)],
        rows=source_curation._d1_rows(payload),
    )
    _canonical(package / "harvest.json", manifest)
    return package


def _rights_ext(
    page_id: int,
    *,
    license_id: str = "cc-by-4.0",
    short_name: str = "CC BY 4.0",
    usage_terms: str = "Creative Commons Attribution 4.0",
    license_url: str | None = "https://creativecommons.org/licenses/by/4.0",
    attribution_required: str = "true",
    copyrighted: str = "True",
    restrictions: str = "",
    categories: str = "Fontana del Moro (Rome)",
) -> dict[str, dict[str, str]]:
    values: dict[str, str | None] = {
        "License": license_id,
        "LicenseShortName": short_name,
        "UsageTerms": usage_terms,
        "LicenseUrl": license_url,
        "AttributionRequired": attribution_required,
        "Copyrighted": copyrighted,
        "Restrictions": restrictions,
        "Artist": f"Photographer {page_id}",
        "Credit": "Own work",
        "Permission": "",
        "ImageDescription": "A fountain.",
        "DateTimeOriginal": "2020-01-01 00:00",
        "Categories": categories,
    }
    return {key: _ext(value) for key, value in values.items() if value is not None}


def _enrichment_page(
    page_id: int,
    *,
    category_titles: list[str] | None = None,
    wikitext: str = "{{Information|author=Photographer}}\n{{cc-by-4.0}}",
    rights: dict | None = None,
) -> dict:
    categories = category_titles or [
        "Category:Fontana del Moro (Rome)",
        "Category:CC-BY-4.0",
    ]
    return {
        "pageid": page_id,
        "ns": 6,
        "title": f"File:Fontana-{page_id}.jpg",
        "imagerepository": "local",
        "revisions": [
            {
                "revid": page_id + 1000,
                "parentid": page_id + 999,
                "timestamp": "2026-07-27T00:00:00Z",
                "sha1": "a" * 40,
                "slots": {
                    "main": {
                        "contentmodel": "wikitext",
                        "contentformat": "text/x-wiki",
                        "content": wikitext,
                    }
                },
            }
        ],
        "imageinfo": [
            {
                "url": f"https://upload.wikimedia.org/test/{page_id}.jpg",
                "sha1": f"{page_id:040x}"[-40:],
                "size": 100_000 + page_id,
                "width": 1600,
                "height": 1200,
                "mime": "image/jpeg",
                "mediatype": "BITMAP",
                "extmetadata": rights or _rights_ext(page_id),
            }
        ],
        "categories": [{"ns": 14, "title": title} for title in categories],
    }


def _payload(
    page_ids: list[int],
    *,
    continuation: dict[str, str] | None = None,
    mutate: Callable[[list[dict[str, Any]]], None] | None = None,
) -> bytes:
    pages = [_enrichment_page(page_id) for page_id in page_ids]
    if mutate is not None:
        mutate(pages)
    value: dict = {"batchcomplete": True, "query": {"pages": pages}}
    if continuation is not None:
        value["continue"] = continuation
    return json.dumps(value).encode("utf-8")


class _Fetcher:
    def __init__(self, responses: list[bytes]) -> None:
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request) -> bytes:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected network request")
        return self.responses.pop(0)


def test_collect_and_public_validator_bind_exact_harvest_and_query(
    tmp_path: Path,
) -> None:
    harvest = _harvest_package(tmp_path)
    fetcher = _Fetcher([_payload(list(range(1, 51)))])
    output = tmp_path / "enrichment"
    record = enrichment.enrich_d1(harvest, output, retrieved_at_utc=_AT, fetch=fetcher)

    assert record["kind"] == "forge-krea-commons-enrichment-unapproved"
    assert record["status"] == "pending_named_human_rights_and_visual_review"
    assert record["human_records"] == []
    assert record["fixture_manifest_created"] is False
    assert record["execution_authorized"] is False
    assert record["counts"] == {"requested": 50, "mechanically_consistent": 50}
    assert enrichment.validate_enrichment(harvest, output) == record

    query = parse_qs(urlsplit(fetcher.requests[0].full_url).query)
    assert len(query["pageids"][0].split("|")) == 50
    assert query["prop"] == ["imageinfo|revisions|categories"]
    assert query["iiprop"] == ["url|sha1|size|mime|mediatype|extmetadata"]
    assert query["rvprop"] == ["ids|timestamp|sha1|content"]
    assert query["rvslots"] == ["main"]
    assert query["cllimit"] == ["max"]


def test_collector_batches_at_fifty_and_covers_every_eligible_id(
    tmp_path: Path,
) -> None:
    harvest = _harvest_package(tmp_path, count=51)
    fetcher = _Fetcher([_payload(list(range(1, 51))), _payload([51])])
    output = tmp_path / "enrichment"
    record = enrichment.enrich_d1(harvest, output, retrieved_at_utc=_AT, fetch=fetcher)
    assert [
        len(parse_qs(urlsplit(request.full_url).query)["pageids"][0].split("|"))
        for request in fetcher.requests
    ] == [50, 1]
    assert len(record["candidate_rows"]) == 51


def test_every_commons_continuation_is_frozen_replayed_and_merged(
    tmp_path: Path,
) -> None:
    harvest = _harvest_package(tmp_path)
    page_ids = list(range(1, 51))

    def first_half(pages: list[dict]) -> None:
        for page in pages:
            page["categories"] = [
                {"ns": 14, "title": "Category:Fontana del Moro (Rome)"}
            ]

    def second_half(pages: list[dict]) -> None:
        for page in pages:
            page["categories"] = [{"ns": 14, "title": "Category:CC-BY-4.0"}]

    fetcher = _Fetcher(
        [
            _payload(
                page_ids,
                continuation={"continue": "||", "clcontinue": "1|CC-BY-4.0"},
                mutate=first_half,
            ),
            _payload(page_ids, mutate=second_half),
        ]
    )
    output = tmp_path / "enrichment"
    record = enrichment.enrich_d1(harvest, output, retrieved_at_utc=_AT, fetch=fetcher)
    assert len(record["raw_responses"]) == 2
    assert record["raw_responses"][1]["continuation_index"] == 1
    second_query = parse_qs(urlsplit(fetcher.requests[1].full_url).query)
    assert second_query["continue"] == ["||"]
    assert second_query["clcontinue"] == ["1|CC-BY-4.0"]
    enrichment.validate_enrichment(harvest, output)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda pages: pages[0]["imageinfo"][0].update({"sha1": "f" * 40}),
            "current JPEG identity disagrees",
        ),
        (
            lambda pages: pages[0]["imageinfo"][0]["extmetadata"].update(
                {"Restrictions": _ext("trademark")}
            ),
            "Restrictions must be present and empty",
        ),
        (
            lambda pages: pages[0]["imageinfo"][0]["extmetadata"].update(
                {"AttributionRequired": _ext("false")}
            ),
            "rights tuple disagrees",
        ),
        (
            lambda pages: pages[0]["imageinfo"][0]["extmetadata"].update(
                {
                    "License": _ext("cc-by-sa-4.0"),
                    "LicenseShortName": _ext("CC BY-SA 4.0"),
                }
            ),
            "non-allowlisted license ID",
        ),
    ],
)
def test_file_and_exact_rights_matrix_drift_fail_closed(
    tmp_path: Path, mutation, message: str
) -> None:
    harvest = _harvest_package(tmp_path)
    fetcher = _Fetcher([_payload(list(range(1, 51)), mutate=mutation)])
    with pytest.raises(ValueError, match=message):
        enrichment.enrich_d1(
            harvest,
            tmp_path / "enrichment",
            retrieved_at_utc=_AT,
            fetch=fetcher,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda pages: pages[0].update({"ns": 0}),
            "page identity disagrees",
        ),
        (
            lambda pages: pages[0]["revisions"][0]["slots"]["main"].update(
                {"content": "{{Information}}"}
            ),
            "wikitext signal is inconsistent",
        ),
        (
            lambda pages: pages[0].update(
                {"categories": [{"ns": 14, "title": "Category:CC-BY-4.0"}]}
            ),
            "not consistently bound to the D1 category",
        ),
        (
            lambda pages: pages[0]["revisions"][0].update(
                {"revid": pages[0]["revisions"][0]["revid"] + 1}
            ),
            "latest revision disagrees",
        ),
    ],
)
def test_page_category_wikitext_and_revision_drift_fail_closed(
    tmp_path: Path, mutation, message: str
) -> None:
    harvest = _harvest_package(tmp_path)
    fetcher = _Fetcher([_payload(list(range(1, 51)), mutate=mutation)])
    with pytest.raises(ValueError, match=message):
        enrichment.enrich_d1(
            harvest,
            tmp_path / "enrichment",
            retrieved_at_utc=_AT,
            fetch=fetcher,
        )


def test_public_domain_requires_exact_tuple_and_pd_template_category(
    tmp_path: Path,
) -> None:
    harvest = _harvest_package(tmp_path)
    source_raw_path = harvest / "source-response-001.json"
    source_payload = json.loads(source_raw_path.read_bytes())
    source_page = source_payload["query"]["pages"][0]
    source_ext = source_page["imageinfo"][0]["extmetadata"]
    source_ext["LicenseShortName"] = _ext("Public domain")
    source_ext["LicenseUrl"] = _ext("")
    source_ext["UsageTerms"] = _ext("Public domain")
    source_raw = json.dumps(source_payload).encode("utf-8")
    source_raw_path.write_bytes(source_raw)
    source_manifest = json.loads((harvest / "harvest.json").read_bytes())
    source_manifest["raw_responses"] = [
        source_curation._raw_record("source-response-001.json", source_raw)
    ]
    source_manifest["candidate_rows"] = source_curation._d1_rows(source_payload)
    body = {
        key: value for key, value in source_manifest.items() if key != "harvest_sha256"
    }
    source_manifest["harvest_sha256"] = provenance.canonical_sha256(body)
    _canonical(harvest / "harvest.json", source_manifest)

    def make_pd(pages: list[dict]) -> None:
        page = pages[0]
        page["categories"] = [
            {"ns": 14, "title": "Category:Fontana del Moro (Rome)"},
            {"ns": 14, "title": "Category:PD-author-example"},
        ]
        page["revisions"][0]["slots"]["main"]["content"] = "{{PD-author|Example}}"
        page["imageinfo"][0]["extmetadata"] = _rights_ext(
            1,
            license_id="pd",
            short_name="Public domain",
            usage_terms="Public domain",
            license_url=None,
            attribution_required="false",
            copyrighted="False",
            categories="Fontana del Moro (Rome)|PD-author-example",
        )

    fetcher = _Fetcher([_payload(list(range(1, 51)), mutate=make_pd)])
    output = tmp_path / "enrichment"
    record = enrichment.enrich_d1(harvest, output, retrieved_at_utc=_AT, fetch=fetcher)
    assert record["candidate_rows"][0]["rights_metadata"]["mechanical_signal"].endswith(
        "pending_named_review"
    )


def test_raw_tamper_and_self_rehashed_authorization_are_rejected(
    tmp_path: Path,
) -> None:
    harvest = _harvest_package(tmp_path)
    output = tmp_path / "enrichment"
    enrichment.enrich_d1(
        harvest,
        output,
        retrieved_at_utc=_AT,
        fetch=_Fetcher([_payload(list(range(1, 51)))]),
    )
    raw_path = output / "source-response-001.json"
    original_raw = raw_path.read_bytes()
    raw_path.write_bytes(original_raw + b" ")
    with pytest.raises(ValueError, match="response identity mismatch"):
        enrichment.validate_enrichment(harvest, output)

    raw_path.write_bytes(original_raw)
    manifest_path = output / "enrichment.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["execution_authorized"] = True
    body = {key: value for key, value in manifest.items() if key != "enrichment_sha256"}
    manifest["enrichment_sha256"] = provenance.canonical_sha256(body)
    _canonical(manifest_path, manifest)
    with pytest.raises(ValueError, match="not an intact pending record"):
        enrichment.validate_enrichment(harvest, output)


def test_cli_has_only_collection_and_validation_paths() -> None:
    parser = enrichment._parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) == {"collect", "validate"}
