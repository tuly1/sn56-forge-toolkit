"""CPU-only tests for the unapproved D1/D2 source-curation boundary."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys

import pytest


_CALIBRATION = Path(__file__).parents[1] / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))

import krea_provenance as provenance  # noqa: E402
import krea_source_curation as curation  # noqa: E402


_AT = "2026-07-28T00:00:00Z"


def _ext(value: str) -> dict[str, str]:
    return {"value": value}


def _commons_page(
    page_id: int,
    *,
    license_name: str = "CC BY 4.0",
    license_url: str = "https://creativecommons.org/licenses/by/4.0",
    width: int = 1600,
    height: int = 1200,
    content_sha1: str | None = None,
) -> dict:
    return {
        "pageid": page_id,
        "title": f"File:Fontana-{page_id}.jpg",
        "revisions": [{"revid": page_id + 1000, "timestamp": "2026-07-27T00:00:00Z"}],
        "imageinfo": [
            {
                "url": f"https://upload.wikimedia.org/test/{page_id}.jpg",
                "sha1": content_sha1 or f"{page_id:040x}"[-40:],
                "width": width,
                "height": height,
                "mime": "image/jpeg",
                "extmetadata": {
                    "LicenseShortName": _ext(license_name),
                    "LicenseUrl": _ext(license_url),
                    "UsageTerms": _ext(license_name),
                    "Artist": _ext(f"Photographer {page_id}"),
                    "Credit": _ext("Own work"),
                    "Attribution": _ext(f"Photographer {page_id}"),
                },
            }
        ],
    }


def _aic_row(
    object_id: int,
    *,
    title_root: str | None = None,
    image_id: str | None = None,
    public_domain: object = True,
    artist_id: object = 26646,
    artist_title: str = "Tsukioka Kôgyo",
    width: object = 3000,
    height: object = 2026,
) -> dict:
    root = title_root or f"Play {object_id}"
    return {
        "id": object_id,
        "title": (
            f'{root}, from the series "Pictures of No Performances (Nogaku Zue)"'
        ),
        "artist_id": artist_id,
        "artist_title": artist_title,
        "artist_display": "Tsukioka Kogyo\nJapanese, 1869-1927",
        "image_id": image_id or f"00000000-0000-0000-0000-{object_id:012d}",
        "is_public_domain": public_domain,
        "thumbnail": {"width": width, "height": height, "alt_text": "A print."},
        "main_reference_number": f"1939.2258.{object_id}",
        "date_display": "1898",
        "classification_title": "woodblock print",
        "medium_display": "Color woodblock print",
    }


def _canonical(path: Path, value: object) -> None:
    path.write_bytes(provenance.canonical_bytes(value) + b"\n")


def _d1_package(tmp_path: Path) -> Path:
    package = tmp_path / "d1"
    package.mkdir()
    payload = {
        "batchcomplete": True,
        "query": {"pages": [_commons_page(index) for index in range(1, 51)]},
    }
    raw = json.dumps(payload).encode("utf-8")
    (package / "source-response-001.json").write_bytes(raw)
    manifest = curation._harvest_body(
        role="D1",
        concept_id="fontana-del-moro",
        source_system="Wikimedia Commons",
        source_locator=(
            "https://commons.wikimedia.org/wiki/Category:Fontana_del_Moro_(Rome)"
        ),
        retrieved_at_utc=_AT,
        request_contract={"method": "GET", "url": curation.d1_request_url()},
        raw_responses=[curation._raw_record("source-response-001.json", raw)],
        rows=curation._d1_rows(payload),
    )
    _canonical(package / "harvest.json", manifest)
    return package


def _retrieval_authorization(
    path: Path,
    harvest: dict,
    *,
    authorized_at_utc: str = "2026-07-28T00:01:00Z",
) -> Path:
    body = {
        "schema": 1,
        "kind": "forge-krea-curation-retrieval-scope-authorization",
        "owner_identity": "Atulya Shetty",
        "authorized_at_utc": authorized_at_utc,
        "roles": ["D1", "D2"],
        "maximum_persisted_bytes": 4294967296,
        "decision": "authorize_public_candidate_retrieval_for_curation_only",
        "acknowledgements": {
            "aic_public_domain_images_may_have_third_party_rights": True,
            "commons_cc_by_attribution_must_be_preserved": True,
            "commons_sharealike_material_is_excluded": True,
            "download_does_not_approve_fixture_admission": True,
            "named_rights_review_still_required": True,
        },
        "source_policy_sha256": harvest["source_policy_sha256"],
        "harvest_bindings": {
            "D1": curation._harvest_retrieval_binding(harvest),
            "D2": {
                "harvest_sha256": "a" * 64,
                "eligible_source_urls_sha256": "b" * 64,
            },
        },
    }
    _canonical(
        path,
        {**body, "authorization_sha256": provenance.canonical_sha256(body)},
    )
    return path


class _FakeHeaders:
    def __init__(self, mime: str, *, content_length: int | None = None) -> None:
        self._mime = mime
        self._values = {
            "Content-Length": (
                None if content_length is None else str(content_length)
            ),
            "ETag": '"frozen"',
            "Last-Modified": "Tue, 28 Jul 2026 00:00:00 GMT",
        }

    def get_content_type(self) -> str:
        return self._mime

    def get(self, name: str, default: object = None) -> object:
        value = self._values.get(name)
        return default if value is None else value


class _FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        final_url: str,
        mime: str = "image/jpeg",
        content_length: int | None = None,
    ) -> None:
        self._stream = io.BytesIO(payload)
        self._final_url = final_url
        self.headers = _FakeHeaders(mime, content_length=content_length)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def geturl(self) -> str:
        return self._final_url

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


def _materialized_d1_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider_payload: bytes = b"frozen-provider-image",
    downloaded_payload: bytes | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Build a fully validated small-byte D1 materialization via the public API."""

    downloaded = provider_payload if downloaded_payload is None else downloaded_payload
    package = tmp_path / "source"
    package.mkdir()
    source_sha1 = hashlib.sha1(provider_payload).hexdigest()
    payload = {
        "batchcomplete": True,
        "query": {
            "pages": [
                _commons_page(index, content_sha1=source_sha1)
                for index in range(1, 51)
            ]
        },
    }
    raw = json.dumps(payload).encode("utf-8")
    (package / "source-response-001.json").write_bytes(raw)
    manifest = curation._harvest_body(
        role="D1",
        concept_id="fontana-del-moro",
        source_system="Wikimedia Commons",
        source_locator=(
            "https://commons.wikimedia.org/wiki/Category:Fontana_del_Moro_(Rome)"
        ),
        retrieved_at_utc=_AT,
        request_contract={"method": "GET", "url": curation.d1_request_url()},
        raw_responses=[curation._raw_record("source-response-001.json", raw)],
        rows=curation._d1_rows(payload),
    )
    _canonical(package / "harvest.json", manifest)
    harvest = curation.validate_harvest(package)
    authorization = _retrieval_authorization(
        tmp_path / "authorization.json", harvest
    )
    enrichment = tmp_path / "enrichment"
    enrichment.mkdir()
    enrichment_identity = {
        "kind": "test-source-enrichment",
        "enrichment_sha256": "c" * 64,
        "manifest_file_sha256": "d" * 64,
    }
    monkeypatch.setattr(
        curation,
        "_source_enrichment_identity",
        lambda *_args, **_kwargs: enrichment_identity,
    )

    def download(*, destination: Path, expected_mime: str, **_kwargs: object) -> dict:
        destination.write_bytes(downloaded)
        return {
            "bytes": len(downloaded),
            "sha256": hashlib.sha256(downloaded).hexdigest(),
            "sha1": hashlib.sha1(downloaded).hexdigest(),
            "mime": expected_mime,
            "etag": None,
            "last_modified": None,
        }

    monkeypatch.setattr(curation, "_download_one", download)
    monkeypatch.setattr(curation.time, "sleep", lambda _delay: None)
    materialization = tmp_path / "materialization"
    curation.materialize(
        package,
        materialization,
        retrieved_at_utc="2026-07-28T00:02:00Z",
        maximum_total_bytes=64 * 1024 * 1024,
        maximum_file_bytes=1024 * 1024,
        retrieval_authorization=authorization,
        source_enrichment_dir=enrichment,
        delay_s=0.25,
    )
    return package, materialization, authorization, enrichment


def test_d1_requires_consistent_non_sharealike_license_name_and_url() -> None:
    payload = {
        "query": {
            "pages": [
                _commons_page(1),
                _commons_page(
                    2,
                    license_name="CC BY-SA 4.0",
                    license_url="https://creativecommons.org/licenses/by/4.0",
                ),
                _commons_page(
                    3,
                    license_name="CC BY 4.0",
                    license_url="https://creativecommons.org/licenses/by-sa/4.0",
                ),
                _commons_page(
                    4,
                    license_name="CC0",
                    license_url="http://creativecommons.org/publicdomain/zero/1.0/deed.en",
                ),
            ]
        }
    }
    rows = {row["source_id"]: row for row in curation._d1_rows(payload)}
    assert rows["commons-1"]["eligibility"]["passed"] is True
    assert rows["commons-4"]["eligibility"]["passed"] is True
    assert rows["commons-2"]["eligibility"]["passed"] is False
    assert rows["commons-3"]["eligibility"]["passed"] is False


def test_d1_bool_dimension_and_missing_revision_fail_closed() -> None:
    page = _commons_page(1, width=True)
    page["revisions"] = []
    with pytest.raises(ValueError, match="one page/revision/image identity"):
        curation._d1_rows({"query": {"pages": [page]}})


def test_d1_disallowed_original_url_cannot_count_as_machine_eligible() -> None:
    page = _commons_page(1)
    page["imageinfo"][0]["url"] = "https://evil.example/fontana.jpg"
    row = curation._d1_rows({"query": {"pages": [page]}})[0]
    assert row["eligibility"] == {
        "passed": False,
        "reasons": ["original_url_outside_download_allowlist"],
    }


def test_d2_query_is_exact_stable_and_not_fuzzy() -> None:
    query = curation.d2_query(3)
    filters = query["query"]["bool"]["filter"]
    assert {"term": {"artist_id": 26646}} in filters
    assert {"term": {"artist_title.keyword": "Tsukioka Kôgyo"}} in filters
    assert {"term": {"is_public_domain": True}} in filters
    assert {"exists": {"field": "image_id"}} in filters
    assert query["sort"] == [{"id": "asc"}]
    assert query["page"] == 3
    assert "q" not in query


def test_d2_machine_filter_rejects_paratext_short_or_truthy_pd() -> None:
    rows = curation._d2_rows(
        [
            {
                "data": [
                    _aic_row(1),
                    _aic_row(2, title_root="Index Page, prints .1-.50 (Vol.1)"),
                    _aic_row(3, height=707),
                    _aic_row(4, public_domain=1),
                    _aic_row(5, artist_id=True),
                ]
            }
        ]
    )
    by_id = {row["source_id"]: row for row in rows}
    assert by_id["aic-1"]["eligibility"]["passed"] is True
    assert "series_paratext_index_page" in by_id["aic-2"]["eligibility"]["reasons"]
    assert "native_dimensions_below_768_on_one_or_both_axes" in by_id[
        "aic-3"
    ]["eligibility"]["reasons"]
    assert "object_not_public_domain" in by_id["aic-4"]["eligibility"]["reasons"]
    assert "artist_id_not_exact_tsukioka_kogyo" in by_id["aic-5"][
        "eligibility"
    ]["reasons"]


def test_d2_duplicate_object_or_reused_image_uuid_fails_closed() -> None:
    with pytest.raises(ValueError, match="duplicate artwork IDs"):
        curation._d2_rows([{"data": [_aic_row(1), _aic_row(1)]}])
    with pytest.raises(ValueError, match="reuse an image UUID"):
        curation._d2_rows(
            [
                {
                    "data": [
                        _aic_row(1, image_id="same-image"),
                        _aic_row(2, image_id="same-image"),
                    ]
                }
            ]
        )


def test_d2_aliases_and_accession_leafs_form_indivisible_components() -> None:
    tadanori = _aic_row(154985, title_root="Tadanori")
    alternative = _aic_row(155395, title_root="Tadanori or Toshinari")
    left_leaf = _aic_row(155369, title_root="Eboshi-ori")
    right_leaf = _aic_row(155664, title_root="Eboshi-ori")
    left_leaf["main_reference_number"] = "1939.2258.185b"
    right_leaf["main_reference_number"] = "1939.2258.185a"
    rows = {
        row["source_id"]: row
        for row in curation._d2_rows(
            [{"data": [tadanori, alternative, left_leaf, right_leaf]}]
        )
    }
    assert (
        rows["aic-154985"]["automatic_play_component_id"]
        == rows["aic-155395"]["automatic_play_component_id"]
    )
    assert rows["aic-155369"]["accession_family_key"] == "1939.2258.185"
    assert (
        rows["aic-155369"]["automatic_play_component_id"]
        == rows["aic-155664"]["automatic_play_component_id"]
    )


def test_harvest_rederives_raw_bytes_and_cannot_self_assert_approval(
    tmp_path: Path,
) -> None:
    package = _d1_package(tmp_path)
    record = curation.validate_harvest(package)
    assert record["counts"] == {"observed": 50, "eligible": 50, "rejected": 0}
    assert record["human_gates"] == curation._HUMAN_GATES
    assert record["human_approvals"] == []
    assert record["fixture_manifest_created"] is False
    assert record["gpu_execution_authorized"] is False

    record["human_approvals"] = [{"reviewer": "Invented Person"}]
    body = {key: value for key, value in record.items() if key != "harvest_sha256"}
    record["harvest_sha256"] = provenance.canonical_sha256(body)
    _canonical(package / "harvest-mutated.json", record)
    (package / "harvest.json").write_bytes(
        (package / "harvest-mutated.json").read_bytes()
    )
    with pytest.raises(ValueError, match="intact unapproved candidate"):
        curation.validate_harvest(package)


def test_raw_response_tamper_is_detected(tmp_path: Path) -> None:
    package = _d1_package(tmp_path)
    (package / "source-response-001.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="raw response identity mismatch"):
        curation.validate_harvest(package)


def test_d1_continuation_is_rejected_even_after_self_rehash(tmp_path: Path) -> None:
    package = _d1_package(tmp_path)
    payload = json.loads((package / "source-response-001.json").read_bytes())
    payload["continue"] = {"continue": "||", "gcmcontinue": "unsafe"}
    raw = json.dumps(payload).encode("utf-8")
    (package / "source-response-001.json").write_bytes(raw)
    manifest = json.loads((package / "harvest.json").read_bytes())
    manifest["raw_responses"] = [
        curation._raw_record("source-response-001.json", raw)
    ]
    body = {key: value for key, value in manifest.items() if key != "harvest_sha256"}
    manifest["harvest_sha256"] = provenance.canonical_sha256(body)
    _canonical(package / "harvest.json", manifest)
    with pytest.raises(ValueError, match="D1 source/request contract mismatch"):
        curation.validate_harvest(package)


def test_d2_declared_total_must_equal_frozen_page_coverage(tmp_path: Path) -> None:
    package = tmp_path / "d2"
    package.mkdir()
    payload = {
        "pagination": {
            "total": 100,
            "limit": 100,
            "offset": 0,
            "total_pages": 1,
            "current_page": 1,
        },
        "data": [_aic_row(index) for index in range(1000, 1090)],
    }
    raw = json.dumps(payload).encode("utf-8")
    (package / "source-response-001.json").write_bytes(raw)
    rows = curation._d2_rows([payload])
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
        rows=rows,
    )
    _canonical(package / "harvest.json", manifest)
    with pytest.raises(ValueError, match="does not cover its declared total"):
        curation.validate_harvest(package)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_pages", True),
        ("limit", 100.0),
        ("offset", False),
        ("current_page", True),
    ],
)
def test_d2_pagination_types_and_offset_fail_after_self_rehash(
    tmp_path: Path, field: str, value: object
) -> None:
    package = tmp_path / "d2-pagination"
    package.mkdir()
    payload = {
        "pagination": {
            "total": 90,
            "limit": 100,
            "offset": 0,
            "total_pages": 1,
            "current_page": 1,
        },
        "data": [_aic_row(index) for index in range(1000, 1090)],
    }
    raw = json.dumps(payload).encode("utf-8")
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
    curation.validate_harvest(package)

    payload["pagination"][field] = value
    mutated_raw = json.dumps(payload).encode("utf-8")
    (package / "source-response-001.json").write_bytes(mutated_raw)
    manifest["raw_responses"] = [
        curation._raw_record("source-response-001.json", mutated_raw)
    ]
    body = {key: value for key, value in manifest.items() if key != "harvest_sha256"}
    manifest["harvest_sha256"] = provenance.canonical_sha256(body)
    _canonical(package / "harvest.json", manifest)
    with pytest.raises(ValueError, match="pagination is incomplete or inconsistent"):
        curation.validate_harvest(package)


@pytest.mark.parametrize(
    ("role", "url"),
    [
        ("D1", "http://upload.wikimedia.org/file.jpg"),
        ("D1", "https://evil.example/file.jpg"),
        ("D2", "https://user:secret@www.artic.edu/image.jpg"),
        ("D2", "https://www.artic.edu/image.jpg#fragment"),
    ],
)
def test_download_url_allowlists_fail_closed(role: str, url: str) -> None:
    with pytest.raises(ValueError, match="allowlist"):
        curation._validated_download_url(url, role)


def test_policy_is_bound_and_has_no_approval_transition() -> None:
    policy, digest = curation._source_policy()
    assert policy["status"] == "curation_only_no_admission"
    assert policy["human_gates"] == curation._HUMAN_GATES
    assert policy["claim_limit"] == curation._CLAIM_LIMIT
    assert len(digest) == 64
    source = Path(curation.__file__).read_text(encoding="utf-8")
    assert "approve" not in {
        "harvest-d1",
        "harvest-d2",
        "validate-harvest",
        "materialize",
        "validate-materialization",
        "inspect",
    }
    assert "gpu_execution_authorized\": True" not in source


def test_retrieval_requires_exact_named_owner_acknowledgements(tmp_path: Path) -> None:
    harvest = curation.validate_harvest(_d1_package(tmp_path))
    body = {
        "schema": 1,
        "kind": "forge-krea-curation-retrieval-scope-authorization",
        "owner_identity": "Atulya Shetty",
        "authorized_at_utc": _AT,
        "roles": ["D1", "D2"],
        "maximum_persisted_bytes": 4294967296,
        "decision": "authorize_public_candidate_retrieval_for_curation_only",
        "acknowledgements": {
            "aic_public_domain_images_may_have_third_party_rights": True,
            "commons_cc_by_attribution_must_be_preserved": True,
            "commons_sharealike_material_is_excluded": True,
            "download_does_not_approve_fixture_admission": True,
            "named_rights_review_still_required": True,
        },
        "source_policy_sha256": harvest["source_policy_sha256"],
        "harvest_bindings": {
            "D1": curation._harvest_retrieval_binding(harvest),
            "D2": {
                "harvest_sha256": "a" * 64,
                "eligible_source_urls_sha256": "b" * 64,
            },
        },
    }
    record = {**body, "authorization_sha256": provenance.canonical_sha256(body)}
    path = tmp_path / "authorization.json"
    _canonical(path, record)
    assert (
        curation.validate_retrieval_authorization(
            path, harvest=harvest, requested_maximum_bytes=1024
        )["owner_identity"]
        == "Atulya Shetty"
    )
    record["acknowledgements"][
        "named_rights_review_still_required"
    ] = False
    body = {key: value for key, value in record.items() if key != "authorization_sha256"}
    record["authorization_sha256"] = provenance.canonical_sha256(body)
    _canonical(tmp_path / "authorization-bad.json", record)
    with pytest.raises(ValueError, match="absent, invalid, or too narrow"):
        curation.validate_retrieval_authorization(
            tmp_path / "authorization-bad.json",
            harvest=harvest,
            requested_maximum_bytes=1024,
        )


def test_inspection_thresholds_cannot_override_frozen_policy(tmp_path: Path) -> None:
    # The threshold check occurs only after validated materialization. Directly
    # assert the policy loader rejects any hidden policy mutation instead.
    policy, _ = curation._selection_policy()
    assert policy["perceptual_screen"] == {
        "algorithm": "rgb-luma-average-hash-8x8-bilinear-after-exif-transpose",
        "automatic_union_maximum_hamming_distance": 6,
        "human_review_queue_maximum_hamming_distance": 10,
    }


def test_automatic_cluster_is_only_a_machine_hint() -> None:
    rows = [
        {"source_id": "a", "perceptual_hash64": "0000000000000000"},
        {"source_id": "b", "perceptual_hash64": "0000000000000001"},
        {"source_id": "c", "perceptual_hash64": "ffffffffffffffff"},
    ]
    assert curation._union_clusters(rows, 1) == [["a", "b"]]


def test_download_one_binds_redirect_host_mime_bytes_and_content_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"exact-downloaded-image-bytes"
    source_url = "https://upload.wikimedia.org/source/image.jpg"
    response = _FakeResponse(
        payload,
        final_url="https://upload.wikimedia.org/revision/image.jpg",
        content_length=len(payload),
    )
    monkeypatch.setattr(curation, "urlopen", lambda *_args, **_kwargs: response)
    destination = tmp_path / "image.jpg"

    identity = curation._download_one(
        url=source_url,
        destination=destination,
        expected_mime="image/jpeg",
        maximum_bytes=len(payload),
    )

    assert destination.read_bytes() == payload
    assert identity == {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "sha1": hashlib.sha1(payload).hexdigest(),
        "mime": "image/jpeg",
        "etag": '"frozen"',
        "last_modified": "Tue, 28 Jul 2026 00:00:00 GMT",
    }


@pytest.mark.parametrize(
    ("response", "maximum_bytes", "message"),
    [
        (
            _FakeResponse(
                b"image",
                final_url="https://evil.example/image.jpg",
            ),
            1024,
            "redirected outside its source host",
        ),
        (
            _FakeResponse(
                b"image",
                final_url="https://upload.wikimedia.org/image.jpg",
                mime="text/html",
            ),
            1024,
            "unexpected image MIME",
        ),
        (
            _FakeResponse(
                b"image",
                final_url="https://upload.wikimedia.org/image.jpg",
                content_length=2048,
            ),
            1024,
            "per-file byte cap",
        ),
        (
            _FakeResponse(
                b"one-byte-too-many",
                final_url="https://upload.wikimedia.org/image.jpg",
            ),
            len(b"one-byte-too-many") - 1,
            "per-file byte cap",
        ),
    ],
)
def test_download_one_fails_closed_and_removes_partials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: _FakeResponse,
    maximum_bytes: int,
    message: str,
) -> None:
    monkeypatch.setattr(curation, "urlopen", lambda *_args, **_kwargs: response)
    destination = tmp_path / "image.jpg"
    with pytest.raises(ValueError, match=message):
        curation._download_one(
            url="https://upload.wikimedia.org/image.jpg",
            destination=destination,
            expected_mime="image/jpeg",
            maximum_bytes=maximum_bytes,
        )
    assert not destination.exists()
    assert list(tmp_path.glob(".partial-*")) == []


def test_authorization_must_follow_the_exact_bound_harvest(tmp_path: Path) -> None:
    harvest = curation.validate_harvest(_d1_package(tmp_path))
    authorization = _retrieval_authorization(
        tmp_path / "authorization.json",
        harvest,
        authorized_at_utc="2026-07-27T23:59:59Z",
    )
    with pytest.raises(ValueError, match="predates the bound source harvest"):
        curation.validate_retrieval_authorization(
            authorization,
            harvest=harvest,
            requested_maximum_bytes=1024,
        )


def test_materialization_time_must_follow_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harvest_dir = _d1_package(tmp_path)
    harvest = curation.validate_harvest(harvest_dir)
    authorization = _retrieval_authorization(
        tmp_path / "authorization.json",
        harvest,
        authorized_at_utc="2026-07-28T00:02:00Z",
    )
    enrichment = tmp_path / "enrichment"
    enrichment.mkdir()
    monkeypatch.setattr(
        curation,
        "_source_enrichment_identity",
        lambda *_args, **_kwargs: {
            "kind": "test-source-enrichment",
            "enrichment_sha256": "c" * 64,
            "manifest_file_sha256": "d" * 64,
        },
    )
    with pytest.raises(ValueError, match="retrieval time predates authorization"):
        curation.materialize(
            harvest_dir,
            tmp_path / "materialization",
            retrieved_at_utc="2026-07-28T00:01:59Z",
            maximum_total_bytes=1024,
            maximum_file_bytes=512,
            retrieval_authorization=authorization,
            source_enrichment_dir=enrichment,
        )


def test_provider_sha1_is_rederived_after_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="differs from the provider SHA-1"):
        _materialized_d1_package(
            tmp_path,
            monkeypatch,
            provider_payload=b"provider-original",
            downloaded_payload=b"different-download",
        )


@pytest.mark.parametrize("unsafe_entry", ["unexpected-root", "image-symlink"])
def test_materialization_namespace_rejects_unexpected_or_symlink_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_entry: str,
) -> None:
    harvest, materialization, authorization, enrichment = _materialized_d1_package(
        tmp_path, monkeypatch
    )
    if unsafe_entry == "unexpected-root":
        (materialization / "not-in-manifest.txt").write_text("rogue", encoding="utf-8")
        message = "unexpected root entries"
    else:
        (materialization / "images" / "rogue.jpg").symlink_to(
            materialization / "images" / "commons-1.jpg"
        )
        message = "unsafe entries"
    with pytest.raises(ValueError, match=message):
        curation.validate_materialization(
            harvest,
            materialization,
            retrieval_authorization=authorization,
            source_enrichment_dir=enrichment,
        )


def test_inspection_rejects_thresholds_other_than_frozen_six_and_ten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        curation,
        "validate_harvest",
        lambda _path: {"experimental_role": "D1"},
    )
    monkeypatch.setattr(
        curation,
        "validate_materialization",
        lambda *_args, **_kwargs: {
            "retrieved_at_utc": "2026-07-28T00:02:00Z"
        },
    )
    with pytest.raises(ValueError, match="frozen 6/10 policy"):
        curation._derive_inspection_body(
            tmp_path,
            tmp_path,
            retrieval_authorization=tmp_path / "authorization.json",
            source_enrichment_dir=tmp_path,
            inspected_at_utc="2026-07-28T00:03:00Z",
            cluster_hamming_threshold=7,
            review_hamming_threshold=10,
        )


def test_inspection_enforces_explicit_decoded_pixel_cap_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Image = pytest.importorskip("PIL.Image")

    materialization_dir = tmp_path / "materialization"
    images_dir = materialization_dir / "images"
    images_dir.mkdir(parents=True)
    image_path = images_dir / "commons-1.jpg"
    # 25,005,000 pixels: just above the exact 25,000,000 frozen cap while
    # remaining below Pillow's independently configurable bomb threshold.
    Image.new("L", (5001, 5000), color=0).save(image_path, format="JPEG")
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        curation,
        "validate_harvest",
        lambda _path: {
            "experimental_role": "D1",
            "concept_id": "fontana-del-moro",
            "harvest_sha256": "a" * 64,
            "candidate_rows": [
                {
                    "source_id": "commons-1",
                    "title": "Fontana del Moro",
                    "creator": "test:creator",
                }
            ],
        },
    )
    monkeypatch.setattr(
        curation,
        "validate_materialization",
        lambda *_args, **_kwargs: {
            "retrieved_at_utc": "2026-07-28T00:02:00Z",
            "materialization_sha256": "b" * 64,
            "rows": [
                {
                    "source_id": "commons-1",
                    "relative_path": "images/commons-1.jpg",
                    "sha256": image_sha256,
                    "mime": "image/jpeg",
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="not safely decodable") as caught:
        curation._derive_inspection_body(
            tmp_path,
            materialization_dir,
            retrieval_authorization=tmp_path / "authorization.json",
            source_enrichment_dir=tmp_path,
            inspected_at_utc="2026-07-28T00:03:00Z",
        )
    assert "frozen pixel cap" in str(caught.value.__cause__)


def test_inspection_validation_rederives_bytes_and_rejects_self_rehashed_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Image = pytest.importorskip("PIL.Image")

    stream = io.BytesIO()
    Image.new("RGB", (768, 768), color=(40, 80, 120)).save(stream, format="JPEG")
    image_bytes = stream.getvalue()
    harvest, materialization, authorization, enrichment = _materialized_d1_package(
        tmp_path,
        monkeypatch,
        provider_payload=image_bytes,
    )
    inspection = tmp_path / "inspection.json"
    record = curation.inspect_candidates(
        harvest,
        materialization,
        inspection,
        retrieval_authorization=authorization,
        source_enrichment_dir=enrichment,
        inspected_at_utc="2026-07-28T00:03:00Z",
    )
    assert record["rows"][0]["automatic_group_hints"] == {
        "creator_id": "text:photographer 1",
        "burst_id": "unreviewed-singleton-commons-1",
        "play_root": None,
        "play_component": None,
        "accession_family": None,
    }
    assert (
        curation.validate_inspection(
            harvest,
            materialization,
            inspection,
            retrieval_authorization=authorization,
            source_enrichment_dir=enrichment,
        )
        == record
    )

    record["rows"][0]["perceptual_hash64"] = "0000000000000000"
    body = {key: value for key, value in record.items() if key != "inspection_sha256"}
    record["inspection_sha256"] = provenance.canonical_sha256(body)
    _canonical(inspection, record)
    with pytest.raises(ValueError, match="does not rederive from bound image bytes"):
        curation.validate_inspection(
            harvest,
            materialization,
            inspection,
            retrieval_authorization=authorization,
            source_enrichment_dir=enrichment,
        )
