"""CPU-only contracts for the Week-7 procedural HKE candidate renderer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "ops"
    / "experiments"
    / "week7"
    / "hke_procedural_renderer.py"
)
SPEC = importlib.util.spec_from_file_location("hke_procedural_renderer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)

DISCOVERY_KEY = b"week7-discovery-key-material-0001"
CONFIRMATION_KEY = b"week7-confirm-key-material-0000002"
GENERATOR_COMMIT = "a" * 40
GENERATOR_TREE = "b" * 40
RIGHTS = {
    "author_record": "SN56 first-party procedural renderer",
    "rights_owner": "SN56 project owner",
    "license_or_use_grant": "first-party training and evaluation fixture use",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def _tree_identity(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, dict]:
    root = tmp_path_factory.mktemp("week7-hke-renderer")
    public = root / "public"
    custodian = root / "custodian"
    result = renderer.build_candidate(
        public_output=public,
        custodian_output=custodian,
        discovery_key=DISCOVERY_KEY,
        confirmation_key=CONFIRMATION_KEY,
        generator_commit=GENERATOR_COMMIT,
        generator_tree=GENERATOR_TREE,
        **RIGHTS,
    )
    return public, custodian, result


def test_exact_shapes_candidate_governance_and_row_evidence(built) -> None:
    public, custodian, result = built
    assert result["status"] == "candidate_unreviewed"
    assert result["governance"] == {
        "human_review": "not_performed",
        "agent_is_not_human": True,
        "admission_authorized": False,
        "gpu_execution_authorized": False,
    }
    expected = {
        "W7-HKE-SOCIAL-A": (10, 8),
        "W7-HKE-PRODUCT-A": (28, 10),
        "W7-HKE-LOGO-UI-A": (32, 10),
    }
    confirmation = _json(custodian / "CONFIRMATION-MANIFEST.json")
    total = 0
    for fixture_id, (discovery_count, confirmation_count) in expected.items():
        public_family = result["families"][fixture_id]
        private_family = confirmation["families"][fixture_id]
        assert public_family["discovery"]["row_count"] == discovery_count
        assert len(public_family["discovery"]["rows"]) == discovery_count
        assert public_family["confirmation"]["row_count"] == confirmation_count
        assert private_family["row_count"] == confirmation_count
        total += discovery_count + confirmation_count
        for row in public_family["discovery"]["rows"] + private_family["rows"]:
            assert row["parameters_sha256"] == renderer.semantic_sha256(row["parameters"])
            assert row["group_identity_sha256"] == renderer.semantic_sha256(
                row["group_identity"]
            )
            assert row["rights_declaration"] == renderer.RIGHTS_DECLARATION
            assert row["visible_glyph_transcript"]
    assert total == 98
    assert result["cross_candidate_evidence"]["row_count"] == 98
    assert result["cross_candidate_evidence"]["pair_comparisons"] == 4753
    assert result["cross_candidate_evidence"]["exact_image_duplicate_count"] == 0
    assert result["cross_candidate_evidence"]["decoded_pixel_duplicate_count"] == 0
    assert result["cross_candidate_evidence"]["normalized_caption_duplicate_count"] == 0
    assert result["cross_candidate_evidence"]["group_identity_duplicate_count"] == 0
    assert result["cross_candidate_evidence"]["perceptual_near_duplicate_count"] == 0
    assert result["rights_record"]["rights_owner"] == RIGHTS["rights_owner"]
    assert renderer.verify_candidate(public, custodian)["verified_rows"] == 98


def test_confirmation_is_custodian_only_and_public_manifest_leaks_no_membership(
    built,
) -> None:
    public, custodian, result = built
    public_bytes = (public / "CANDIDATE-MANIFEST.json").read_bytes()
    confirmation = _json(custodian / "CONFIRMATION-MANIFEST.json")
    assert not (public / "confirmation").exists()
    assert (custodian / "confirmation").is_dir()
    assert str(custodian).encode() not in public_bytes
    assert result["confirmation"]["public_membership_disclosed"] is False
    for family in confirmation["families"].values():
        for row in family["rows"]:
            caption = (
                custodian / "confirmation" / row["relative_caption_path"]
            ).read_bytes()
            for private_value in (
                row["row_id"],
                row["relative_image_path"],
                row["relative_caption_path"],
                row["image_sha256"],
                row["caption_sha256"],
                row["parameters_sha256"],
                row["row_record_sha256"],
                caption.decode("utf-8").strip(),
            ):
                assert private_value.encode("ascii") not in public_bytes


def test_fresh_build_is_byte_deterministic_and_replay_passes(tmp_path: Path) -> None:
    first_public, first_custodian = tmp_path / "p1", tmp_path / "c1"
    second_public, second_custodian = tmp_path / "p2", tmp_path / "c2"
    first = renderer.build_candidate(
        public_output=first_public,
        custodian_output=first_custodian,
        discovery_key=DISCOVERY_KEY,
        confirmation_key=CONFIRMATION_KEY,
        generator_commit=GENERATOR_COMMIT,
        generator_tree=GENERATOR_TREE,
        **RIGHTS,
    )
    second = renderer.build_candidate(
        public_output=second_public,
        custodian_output=second_custodian,
        discovery_key=DISCOVERY_KEY,
        confirmation_key=CONFIRMATION_KEY,
        generator_commit=GENERATOR_COMMIT,
        generator_tree=GENERATOR_TREE,
        **RIGHTS,
    )
    assert first == second
    assert _tree_identity(first_public) == _tree_identity(second_public)
    assert _tree_identity(first_custodian) == _tree_identity(second_custodian)
    assert renderer.verify_replay(
        public_output=first_public,
        custodian_output=first_custodian,
        discovery_key=DISCOVERY_KEY,
        confirmation_key=CONFIRMATION_KEY,
    ) == {
        "status": "PASS",
        "verified_rows": 98,
        "candidate_semantic_sha256": first["semantic_sha256"],
        "dedup_semantic_sha256": first["cross_candidate_evidence"][
            "semantic_sha256"
        ],
    }


def test_keys_and_domains_are_distinct_and_wrong_key_fails_replay(built) -> None:
    public, custodian, _ = built
    fixture = renderer.FIXTURE_CONTRACT[0]
    discovery_image, _, discovery_row = renderer._render(
        fixture, "discovery", 0, DISCOVERY_KEY
    )
    confirmation_image, _, confirmation_row = renderer._render(
        fixture, "confirmation", 0, CONFIRMATION_KEY
    )
    assert discovery_row["parameters"] != confirmation_row["parameters"]
    assert discovery_image != confirmation_image
    assert discovery_row["seed_commitment_sha256"] != confirmation_row[
        "seed_commitment_sha256"
    ]
    with pytest.raises(renderer.FixtureError, match="replay mismatch"):
        renderer.verify_replay(
            public_output=public,
            custodian_output=custodian,
            discovery_key=b"wrong-discovery-key-material-0000",
            confirmation_key=CONFIRMATION_KEY,
        )
    with pytest.raises(renderer.FixtureError, match="must be distinct"):
        renderer.build_candidate(
            public_output=public.parent / "same-key-public",
            custodian_output=public.parent / "same-key-custodian",
            discovery_key=DISCOVERY_KEY,
            confirmation_key=DISCOVERY_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            **RIGHTS,
        )


def test_confirmation_membership_caption_and_visible_token_require_private_key() -> None:
    fixture = renderer.FIXTURE_CONTRACT[0]
    _, first_caption, first_row = renderer._render(
        fixture, "confirmation", 0, CONFIRMATION_KEY
    )
    _, second_caption, second_row = renderer._render(
        fixture,
        "confirmation",
        0,
        b"independent-confirm-key-material-0003",
    )
    assert first_row["row_id"] != second_row["row_id"]
    assert first_row["relative_image_path"] != second_row["relative_image_path"]
    assert first_row["relative_caption_path"] != second_row["relative_caption_path"]
    assert first_caption != second_caption
    assert first_row["caption_sha256"] != second_row["caption_sha256"]
    assert first_row["visible_glyph_transcript"] != second_row[
        "visible_glyph_transcript"
    ]
    assert first_row["group_identity"] != second_row["group_identity"]
    predictable = (
        f"{str(fixture['fixture_id']).lower()}-confirmation-001"
    )
    assert first_row["row_id"] != predictable
    assert second_row["row_id"] != predictable


def test_create_only_disjoint_outputs_and_tamper_detection(
    built, tmp_path: Path
) -> None:
    public, custodian, _ = built
    with pytest.raises(FileExistsError, match="refusing to replace"):
        renderer.build_candidate(
            public_output=public,
            custodian_output=custodian,
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            **RIGHTS,
        )
    with pytest.raises(renderer.FixtureError, match="disjoint trees"):
        renderer.build_candidate(
            public_output=tmp_path / "nested",
            custodian_output=tmp_path / "nested" / "custodian",
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            **RIGHTS,
        )
    copied_public = tmp_path / "copied-public"
    copied_custodian = tmp_path / "copied-custodian"
    shutil.copytree(public, copied_public)
    shutil.copytree(custodian, copied_custodian)
    manifest = _json(copied_public / "DISCOVERY-MANIFEST.json")
    row = next(iter(manifest["families"].values()))["rows"][0]
    target = copied_public / "discovery" / row["relative_image_path"]
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(renderer.FixtureError, match="root inventory mismatch"):
        renderer.verify_candidate(copied_public, copied_custodian)

    clean_public = tmp_path / "extra-public"
    clean_custodian = tmp_path / "extra-custodian"
    shutil.copytree(public, clean_public)
    shutil.copytree(custodian, clean_custodian)
    (clean_public / "unexpected-private-review.json").write_text("{}\n")
    with pytest.raises(renderer.FixtureError, match="root inventory mismatch"):
        renderer.verify_candidate(clean_public, clean_custodian)


def test_source_and_records_exclude_forbidden_input_surfaces(built) -> None:
    _, _, result = built
    assert result["forbidden_inventories"] == renderer.FORBIDDEN_INVENTORIES
    assert all(value == [] for value in result["forbidden_inventories"].values())
    source = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "import random",
        "import time",
        "import urllib",
        "import requests",
        "ImageFont",
        "os.environ",
        "os.getenv",
    ):
        assert forbidden not in source
    assert result["generator"]["network_access"] == "no-network-code-path"
    assert result["generator"]["ambient_inputs"] == []
    assert any(item.startswith("Pillow==") for item in result["generator"]["dependencies"])


def test_owner_and_use_record_are_mandatory_before_candidate_creation(tmp_path: Path) -> None:
    invalid = dict(RIGHTS)
    invalid["rights_owner"] = "TBD"
    with pytest.raises(renderer.FixtureError, match="rights owner"):
        renderer.build_candidate(
            public_output=tmp_path / "public",
            custodian_output=tmp_path / "custodian",
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            generator_commit=GENERATOR_COMMIT,
            generator_tree=GENERATOR_TREE,
            **invalid,
        )


def test_declarative_contract_matches_the_renderer_and_keeps_authority_closed() -> None:
    contract = json.loads(
        MODULE_PATH.with_name("hke_fixture_contract.json").read_text(encoding="utf-8")
    )
    expected = {
        item["family"]: {
            "discovery": item["discovery_count"],
            "confirmation": item["confirmation_count"],
        }
        for item in renderer.FIXTURE_CONTRACT
    }
    actual = {
        family: {
            "discovery": value["discovery_pairs"],
            "confirmation": value["confirmation_pairs"],
        }
        for family, value in contract["families"].items()
    }
    assert actual == expected
    assert contract["authorization"] == {
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "agent_review_is_not_human_review": True,
        "human_review_evidence_class": "operator_attested",
        "owner_ratification_required_for_gpu": True,
    }
