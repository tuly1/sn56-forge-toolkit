"""Contracts for the deterministic Week-6 Lane-B plan builder."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys

import pytest
from PIL import Image


_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT / "ops" / "calibration"))

import lane_b_generate as generator  # noqa: E402
import lane_b_plan as planner  # noqa: E402


EXPECTED_COUNTS = {
    "W6-DESIGN-GPT-A": 21,
    "W6-DESIGN-NB-B": 20,
    "W6-PRODUCT-A": 21,
    "W6-PRODUCT-B": 20,
    "W6-SOCIAL-A": 18,
    "W6-DESIGN-GPT-LARGE": 48,
}
TRIGGERS = {
    "W6-DESIGN-GPT-A": "Zefqara Grid 7C4F",
    "W6-DESIGN-NB-B": "Qelvara Mobile 82D1",
    "W6-PRODUCT-A": "Qorvex Loop A7K4",
    "W6-PRODUCT-B": "Vaskora Pivot B2M9",
    "W6-SOCIAL-A": "Pryqela Social 51D9",
    "W6-DESIGN-GPT-LARGE": "Nexqari Panel 4E6B",
}


def _reference(path: Path, payload: bytes) -> Path:
    color = tuple(hashlib.sha256(payload).digest()[:3])
    output = BytesIO()
    Image.new("RGB", (16, 16), color).save(output, format="PNG")
    path.write_bytes(output.getvalue())
    return path


def _materialization(tmp_path: Path, *, relief: bool = False) -> dict:
    return planner.build_plan(
        "materialization",
        social_relief=relief,
        product_a_reference=_reference(tmp_path / "a.png", b"reference-a"),
        product_b_reference=_reference(tmp_path / "b.png", b"reference-b"),
    )


def _fixtures(plan: dict) -> dict[str, list[dict]]:
    return {fixture["id"]: fixture["rows"] for fixture in plan["fixtures"]}


def test_materialization_is_deterministic_exact_and_generator_loadable(
    tmp_path: Path,
) -> None:
    first = _materialization(tmp_path)
    second = planner.build_plan(
        "materialization",
        product_a_reference=tmp_path / "a.png",
        product_b_reference=tmp_path / "b.png",
    )
    assert planner.canonical_bytes(first) == planner.canonical_bytes(second)
    fixtures = _fixtures(first)
    assert {fixture_id: len(rows) for fixture_id, rows in fixtures.items()} == EXPECTED_COUNTS
    assert sum(map(len, fixtures.values())) == 148

    output = tmp_path / "plan.json"
    file_sha = planner.publish_plan(output, first)
    loaded, observed_sha = generator.load_plan(output)
    assert loaded == first
    assert observed_sha == file_sha
    assert output.read_bytes() == planner.canonical_bytes(first) + b"\n"
    assert stat_mode(output) == 0o600


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_provider_model_aspect_and_rights_mappings_are_frozen(tmp_path: Path) -> None:
    fixtures = _fixtures(_materialization(tmp_path))
    expected = {
        "W6-DESIGN-GPT-A": ("openai", "gpt-image-2-2026-04-21"),
        "W6-DESIGN-NB-B": ("gemini", "gemini-3.1-flash-image"),
        "W6-PRODUCT-A": ("gemini", "gemini-3.1-flash-image"),
        "W6-PRODUCT-B": ("gemini", "gemini-3.1-flash-image"),
        "W6-SOCIAL-A": ("gemini", "gemini-3.1-flash-image"),
        "W6-DESIGN-GPT-LARGE": ("openai", "gpt-image-2-2026-04-21"),
    }
    allowed_aspects = {
        "openai": {"1536x1024", "1024x1024"},
        "gemini": {"9:16", "4:5", "4:3", "1:1", "3:2", "3:4", "16:9"},
    }
    for fixture_id, rows in fixtures.items():
        assert {(row["provider"], row["model"]) for row in rows} == {expected[fixture_id]}
        for row in rows:
            assert row["aspect_ratio"] in allowed_aspects[row["provider"]]
            assert row["rights_reference"].startswith("https://")
            if row["provider"] == "openai":
                assert row["rights_reference"] == "https://openai.com/policies/services-agreement/"
            else:
                assert row["rights_reference"] == "https://ai.google.dev/gemini-api/terms"
    assert {row["aspect_ratio"] for row in fixtures["W6-SOCIAL-A"]} == {
        "1:1",
        "4:5",
        "16:9",
    }


def test_product_rows_bind_exact_references_and_are_direct_edits(tmp_path: Path) -> None:
    a = _reference(tmp_path / "a.png", b"canonical-product-a")
    b = _reference(tmp_path / "b.png", b"canonical-product-b")
    fixtures = _fixtures(
        planner.build_plan(
            "materialization",
            product_a_reference=a,
            product_b_reference=b,
        )
    )
    expected = {
        "W6-PRODUCT-A": {
            "path": str(a),
            "sha256": hashlib.sha256(a.read_bytes()).hexdigest(),
        },
        "W6-PRODUCT-B": {
            "path": str(b),
            "sha256": hashlib.sha256(b.read_bytes()).hexdigest(),
        },
    }
    for fixture_id in ("W6-PRODUCT-A", "W6-PRODUCT-B"):
        assert {json.dumps(row["reference"], sort_keys=True) for row in fixtures[fixture_id]} == {
            json.dumps(expected[fixture_id], sort_keys=True)
        }
        assert all(
            "attached canonical reference image directly" in row["prompt"]
            and "never use or infer from any previously generated edit" in row["prompt"]
            for row in fixtures[fixture_id]
        )


def test_triggers_visible_strings_and_prompts_are_unique(tmp_path: Path) -> None:
    fixtures = _fixtures(_materialization(tmp_path))
    prompts = []
    captions = []
    ids = []
    for fixture_id, rows in fixtures.items():
        trigger = TRIGGERS[fixture_id]
        for row in rows:
            assert row["prompt"].count(trigger) == 1
            assert row["caption"].count(trigger) == 1
            if fixture_id.startswith("W6-DESIGN") or fixture_id == "W6-SOCIAL-A":
                assert "Visible strings (render each exactly once):" in row["prompt"]
                assert "Visible strings:" in row["caption"]
            prompts.append(row["prompt"])
            captions.append(row["caption"])
            ids.append(fixture_id + "/" + row["id"])
    assert len(prompts) == len(set(prompts)) == 148
    assert len(captions) == len(set(captions)) == 148
    assert len(ids) == len(set(ids)) == 148


def test_balanced_blueprint_grid_cardinalities(tmp_path: Path) -> None:
    fixtures = _fixtures(_materialization(tmp_path))
    assert {row["id"].rsplit("-", 1)[-1] for row in fixtures["W6-DESIGN-GPT-A"]} == {
        "overview",
        "focused",
        "exception",
    }
    assert {row["id"].rsplit("-", 1)[-1] for row in fixtures["W6-DESIGN-NB-B"]} == {
        "home",
        "search",
        "detail",
        "offline",
    }
    assert {row["id"].rsplit("-", 1)[-1] for row in fixtures["W6-DESIGN-GPT-LARGE"]} == {
        "overview",
        "filter",
        "detail",
        "comparison",
        "success",
        "exception",
    }
    social_counts = {}
    for row in fixtures["W6-SOCIAL-A"]:
        social_counts[row["aspect_ratio"]] = social_counts.get(row["aspect_ratio"], 0) + 1
    assert social_counts == {"1:1": 6, "4:5": 6, "16:9": 6}


def test_public_seed_contract_and_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(name, "must-not-be-read")
    observed = planner._public_seed("W6-DESIGN-GPT-A")
    expected = hashlib.sha256(
        b"SN56-W6-LANEB-v1" + b"W6-DESIGN-GPT-A"
    ).digest()
    assert observed == expected
    reference_plan = planner.build_plan("reference-assets")
    encoded = planner.canonical_bytes(reference_plan)
    assert b"must-not-be-read" not in encoded
    assert [fixture["id"] for fixture in reference_plan["fixtures"]] == [
        "W6-PRODUCT-A",
        "W6-PRODUCT-B",
    ]
    for fixture in reference_plan["fixtures"]:
        row = fixture["rows"][0]
        assert row["provider"] == "openai"
        assert row["model"] == "gpt-image-2-2026-04-21"
        assert row["reference"] is None
        assert row["prompt"].count(TRIGGERS[fixture["id"]]) == 1
        assert row["caption"].count(TRIGGERS[fixture["id"]]) == 1


def test_social_relief_omits_only_social(tmp_path: Path) -> None:
    fixtures = _fixtures(_materialization(tmp_path, relief=True))
    assert list(fixtures) == [
        "W6-DESIGN-GPT-A",
        "W6-DESIGN-NB-B",
        "W6-PRODUCT-A",
        "W6-PRODUCT-B",
        "W6-DESIGN-GPT-LARGE",
    ]
    assert sum(map(len, fixtures.values())) == 130


def test_fail_closed_reference_inputs_and_create_only_publication(tmp_path: Path) -> None:
    with pytest.raises(planner.PlanError, match="requires both"):
        planner.build_plan("materialization")
    relative = Path("not-absolute.png")
    with pytest.raises(planner.PlanError, match="absolute"):
        planner.build_plan(
            "materialization",
            product_a_reference=relative,
            product_b_reference=relative,
        )
    reference = _reference(tmp_path / "reference.png", b"reference")
    symlink = tmp_path / "reference-link.png"
    symlink.symlink_to(reference)
    with pytest.raises(planner.PlanError, match="open .* safely"):
        planner.build_plan(
            "materialization",
            product_a_reference=symlink,
            product_b_reference=reference,
        )
    duplicate_a = _reference(tmp_path / "duplicate-a.png", b"same-reference")
    duplicate_b = tmp_path / "duplicate-b.png"
    duplicate_b.write_bytes(duplicate_a.read_bytes())
    with pytest.raises(planner.PlanError, match="must differ"):
        planner.build_plan(
            "materialization",
            product_a_reference=duplicate_a,
            product_b_reference=duplicate_b,
        )

    plan = planner.build_plan("reference-assets")
    output = tmp_path / "plan.json"
    planner.publish_plan(output, plan)
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        planner.publish_plan(output, plan)
    assert output.read_bytes() == original


def test_invalid_mode_combinations_fail_closed(tmp_path: Path) -> None:
    reference = _reference(tmp_path / "reference.png", b"reference")
    with pytest.raises(planner.PlanError, match="accepts no"):
        planner.build_plan("reference-assets", social_relief=True)
    with pytest.raises(planner.PlanError, match="accepts no"):
        planner.build_plan("reference-assets", product_a_reference=reference)
    with pytest.raises(planner.PlanError, match="purpose"):
        planner.build_plan("smoke")
    with pytest.raises(planner.PlanError, match="boolean"):
        planner.build_plan("materialization", social_relief=1)  # type: ignore[arg-type]
