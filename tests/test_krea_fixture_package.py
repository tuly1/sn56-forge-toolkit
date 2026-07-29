"""Fail-closed tests for the Week-5 pre-admission fixture packager."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import zipfile

import pytest


_CALIBRATION = Path(__file__).parents[1] / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _CALIBRATION / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


provenance = _load("krea_provenance")
sys.modules["krea_provenance"] = provenance
review_split = _load("krea_review_split")
sys.modules["krea_review_split"] = review_split
package = _load("krea_fixture_package")


def _review_row(role: str, source_id: str, *, phash: int, burst: str) -> dict:
    return {
        "source_id": source_id,
        "review_notes": "rear view; bright daylight",
        "group_identity": {
            "source_id": source_id,
            "creator_id": f"creator:{source_id}",
            "burst_id": burst,
            "scene_id": f"scene:{source_id}",
            "play_root_id": (
                f"play {source_id}" if role == "D2" else f"not-applicable:{source_id}"
            ),
            "human_similarity_cluster_id": f"human:{source_id}",
            "play_component_id": f"component:{source_id}",
            "accession_family_id": f"accession:{source_id}",
        },
        "perceptual_hash64": f"{phash:016x}",
    }


def test_caption_contract_omits_training_trigger_and_uses_it_once_for_eval() -> None:
    d1 = _review_row("D1", "commons-1", phash=1, burst="one")
    train = package._caption_bytes("D1", "training", d1).decode()
    evaluation = package._caption_bytes("D1", "evaluation", d1).decode()

    assert package._TRIGGERS["D1"] not in train
    assert evaluation.startswith(package._TRIGGERS["D1"] + ", ")
    assert evaluation.count(package._TRIGGERS["D1"]) == 1
    assert not any(
        term in train.casefold() for term in package._D1_FORBIDDEN_CAPTION_TERMS
    )

    d2 = _review_row("D2", "aic-1", phash=2, burst="two")
    d2_train = package._caption_bytes("D2", "training", d2).decode()
    assert package._TRIGGERS["D2"] not in d2_train
    assert "tsukioka" not in d2_train.casefold()
    assert "nogaku zue" not in d2_train.casefold()


def test_cc_by_and_pd_rights_obligations_remain_distinct() -> None:
    cc = package._rights_obligations(
        {
            "rights_decision": "approve_cc_by_obligations_recorded",
            "license_url": "https://creativecommons.org/licenses/by/4.0",
        }
    )
    pd = package._rights_obligations(
        {"rights_decision": "approve_pd_or_cc0", "license_url": ""}
    )

    assert cc["attribution_required"] is True
    assert cc["share_alike_required"] is False
    assert pd["attribution_required"] is False
    assert pd["preserve_provenance_as_project_policy"] is True

    with pytest.raises(ValueError, match="unapproved rights"):
        package._rights_obligations(
            {"rights_decision": "needs_escalation", "license_url": ""}
        )


def test_similarity_screen_prior_review_precedes_flag_rule() -> None:
    rows = {
        "a": _review_row("D1", "a", phash=0, burst="shared"),
        "b": _review_row("D1", "b", phash=0, burst="shared"),
    }
    review = {
        "review_sha256": "1" * 64,
        "queued_pair_reviews": {
            "D1": [
                {
                    "left_source_id": "a",
                    "right_source_id": "b",
                    "relationship_decision": "distinct",
                    "pair_id": "D1-pair-1",
                }
            ]
        },
    }

    # Exercise the precedence on the local algorithm without invoking the
    # production D1 cardinality assertions.
    prior = {
        package._pair_key(item["left_source_id"], item["right_source_id"]): item
        for item in review["queued_pair_reviews"]["D1"]
    }
    key = package._pair_key("a", "b")
    distance = (
        int(rows["a"]["perceptual_hash64"], 16)
        ^ int(rows["b"]["perceptual_hash64"], 16)
    ).bit_count()
    shared = [
        field
        for field in ("accession_family_id", "burst_id")
        if rows["a"]["group_identity"][field] == rows["b"]["group_identity"][field]
    ]
    assert distance == 0 and shared == ["burst_id"]
    assert prior[key]["relationship_decision"] == "distinct"


def test_deterministic_zip_has_fixed_metadata_and_exact_pairs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.jpg").write_bytes(b"image-one")
    (source / "one.txt").write_bytes(b"caption-one\n")
    (source / "two.png").write_bytes(b"image-two")
    (source / "two.txt").write_bytes(b"caption-two\n")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    package._deterministic_zip(source, first)
    package._deterministic_zip(source, second)

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["one.jpg", "one.txt", "two.png", "two.txt"]
        assert all(
            info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist()
        )
        assert all(
            info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()
        )


def test_deterministic_zip_rejects_unpaired_input(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.jpg").write_bytes(b"image")
    with pytest.raises(ValueError, match="exact image/caption pairs"):
        package._deterministic_zip(source, tmp_path / "bad.zip")


def test_candidate_manifest_language_cannot_authorize_execution() -> None:
    assert "pre-admission" in package._KIND
    assert "remain-required" in package._CLAIM_LIMIT
    assert package._SCHEMA == 1


def test_governance_rejection_is_recursive() -> None:
    package._reject_governance_escalation(
        {
            "admission_authorized": False,
            "nested": [{"gpu_execution_authorized": False}],
        }
    )
    with pytest.raises(ValueError, match="nested.*gpu_execution_authorized"):
        package._reject_governance_escalation(
            {
                "admission_authorized": False,
                "nested": [{"gpu_execution_authorized": True}],
            }
        )


def test_group_projection_is_role_exact() -> None:
    d1 = package._group_projection(
        "D1", _review_row("D1", "commons-1", phash=1, burst="one")
    )
    d2 = package._group_projection(
        "D2", _review_row("D2", "aic-1", phash=2, burst="two")
    )

    assert set(d1) == set(package._BASE_GROUP_FIELDS)
    assert set(d2) == set(package._BASE_GROUP_FIELDS + package._D2_GROUP_FIELDS)


def test_split_rejects_selected_non_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(package._EXPECTED_COUNTS, "D1", (1, 1))
    first = _review_row("D1", "a", phash=1, burst="one")
    second = _review_row("D1", "b", phash=2, burst="two")
    first["disposition"] = "EXCLUDE"
    second["disposition"] = "CANDIDATE_ONLY_NOT_ADMITTED"
    review = {"review_sha256": "1" * 64, "records": {"D1": [first, second]}}
    body = {
        "schema": 1,
        "kind": "forge-krea-source-split-plan",
        "experimental_role": "D1",
        "source_review_sha256": review["review_sha256"],
        "training_source_ids": ["a"],
        "evaluation_source_ids": ["b"],
        "admission_authorized": False,
        "gpu_execution_authorized": False,
    }
    split = {
        **body,
        "split_sha256": provenance.canonical_sha256(body),
    }

    with pytest.raises(ValueError, match="rejected or escalated"):
        package._validate_split(split, role="D1", review=review)


@pytest.mark.parametrize("member", ["../escape.jpg", "nested/escape.jpg"])
def test_archive_identity_rejects_non_literal_members(
    tmp_path: Path, member: str
) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_STORED) as archive:
        info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = (package.stat.S_IFREG | 0o600) << 16
        archive.writestr(info, b"unsafe")

    with pytest.raises(ValueError, match="unsafe member"):
        package._archive_identity(archive_path)


def test_archive_identity_rejects_noncanonical_metadata(tmp_path: Path) -> None:
    archive_path = tmp_path / "wrong-mode.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_STORED) as archive:
        info = zipfile.ZipInfo("one.jpg", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = (package.stat.S_IFREG | 0o644) << 16
        archive.writestr(info, b"image")

    with pytest.raises(ValueError, match="unsafe member"):
        package._archive_identity(archive_path)


def test_exact_topology_rejects_nested_manifest() -> None:
    expected = [{"path": "D1/training.zip"}]
    observed = [*expected, {"path": "extra/package-manifest.json"}]

    with pytest.raises(ValueError, match="extra/package-manifest.json"):
        package._validate_exact_topology(observed, {"D1/training.zip"})


def test_package_inventory_rejects_non_regular_entries(tmp_path: Path) -> None:
    fifo = tmp_path / "unexpected.pipe"
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="non-regular"):
        package._package_files(tmp_path)
