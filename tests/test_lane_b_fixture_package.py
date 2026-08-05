"""Fail-closed tests for the Week-6 Lane-B fixture package boundary."""

from __future__ import annotations

import hashlib
import importlib.util
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import pytest
from PIL import Image


_CALIBRATION = Path(__file__).parents[1] / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))


def _load(name: str):
    specification = importlib.util.spec_from_file_location(
        name, _CALIBRATION / f"{name}.py"
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


package = _load("lane_b_fixture_package")
planner = _load("lane_b_plan")


def _write_canonical(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(package.canonical_bytes(value) + b"\n")


def _source_package(tmp_path: Path) -> tuple[Path, list[Path], dict[str, dict]]:
    root = tmp_path / "fixtures"
    root.mkdir(parents=True)
    reference_directory = root / ".references"
    reference_directory.mkdir()
    reference_paths: dict[str, Path] = {}
    for index, fixture_id in enumerate(("W6-PRODUCT-A", "W6-PRODUCT-B"), 1):
        path = reference_directory / f"{fixture_id}.png"
        image = BytesIO()
        Image.new("RGB", (8, 8), (200, index * 31, index * 47)).save(
            image, format="PNG"
        )
        path.write_bytes(image.getvalue())
        reference_paths[fixture_id] = path.absolute()
    generation_plan = planner.build_plan(
        "materialization",
        product_a_reference=reference_paths["W6-PRODUCT-A"],
        product_b_reference=reference_paths["W6-PRODUCT-B"],
    )
    manifests: list[Path] = []
    values: dict[str, dict] = {}
    global_index = 0
    for planned_fixture in generation_plan["fixtures"]:
        fixture_id = planned_fixture["id"]
        expected = package._FIXTURES[fixture_id]
        generator_label = expected["generator_labels"][0]
        rows = []
        for local_index, planned in enumerate(planned_fixture["rows"]):
            global_index += 1
            row_id = planned["id"]
            image_relative = f"{fixture_id}/images/{row_id}.png"
            caption_relative = f"{fixture_id}/captions/{row_id}.txt"
            image_path = root / image_relative
            caption_path = root / caption_relative
            separator = "x" if planned["provider"] == "openai" else ":"
            ratio_width, ratio_height = (
                int(value) for value in planned["aspect_ratio"].split(separator)
            )
            divisor = math.gcd(ratio_width, ratio_height)
            width, height = ratio_width // divisor * 8, ratio_height // divisor * 8
            image_buffer = BytesIO()
            Image.new(
                "RGB",
                (width, height),
                (
                    global_index % 256,
                    (global_index // 256) % 256,
                    (global_index * 17) % 256,
                ),
            ).save(image_buffer, format="PNG")
            image = image_buffer.getvalue()
            caption = planned["caption"].encode("utf-8")
            image_path.parent.mkdir(parents=True, exist_ok=True)
            caption_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(image)
            caption_path.write_bytes(caption)
            image_sha = hashlib.sha256(image).hexdigest()
            rows.append(
                {
                    "row_id": row_id,
                    "image_path": image_relative,
                    "caption_path": caption_relative,
                    "provider": planned["provider"],
                    "model": planned["model"],
                    "request_time_utc": "2026-08-05T12:00:00Z",
                    "request_id": f"request-{global_index:04d}",
                    "prompt": planned["prompt"],
                    "output_sha256": image_sha,
                    "rights_reference": (
                        planned["rights_reference"]
                    ),
                    "rights_status": "approved-provider-terms",
                    "width": width,
                    "height": height,
                    "aspect_ratio": package._canonical_ratio(width, height),
                    "image_sha256": image_sha,
                    "caption_sha256": hashlib.sha256(caption).hexdigest(),
                    "reference_sha256": (
                        planned["reference"]["sha256"]
                        if planned["reference"] is not None
                        else None
                    ),
                }
            )
        value = {
            "schema": 1,
            "kind": package._SOURCE_KIND,
            "fixture_id": fixture_id,
            "family": expected["family"],
            "subtype": expected["subtype"],
            "generator_label": generator_label,
            "total_pairs": expected["total_pairs"],
            "rows": rows,
            "reference_provenance": (
                {
                    "schema": 1,
                    "kind": "sn56-week6-lane-b-reference-provenance",
                    "fixture_id": fixture_id,
                    "row_id": f"{fixture_id.casefold()}-reference",
                    "provider": "openai",
                    "model": package._OPENAI_MODEL,
                    "reference_plan_file_sha256": (
                        "1" if fixture_id.endswith("-A") else "2"
                    )
                    * 64,
                    "reference_generation_summary_sha256": (
                        "3" if fixture_id.endswith("-A") else "4"
                    )
                    * 64,
                    "reference_success_receipt_sha256": (
                        "5" if fixture_id.endswith("-A") else "6"
                    )
                    * 64,
                    "reference_request_payload_sha256": (
                        "7" if fixture_id.endswith("-A") else "8"
                    )
                    * 64,
                    "reference_output_sha256": rows[0]["reference_sha256"],
                    "reference_rights_reference": package._OPENAI_RIGHTS,
                    "reference_rights_status": "approved-provider-terms",
                }
                if fixture_id.startswith("W6-PRODUCT")
                else None
            ),
        }
        manifest = tmp_path / "manifests" / f"{fixture_id}.json"
        _write_canonical(manifest, value)
        manifests.append(manifest)
        values[fixture_id] = value
    return root, manifests, values


@pytest.fixture
def source_package(tmp_path: Path) -> tuple[Path, list[Path], dict[str, dict]]:
    return _source_package(tmp_path)


def _key(tmp_path: Path, payload: bytes = bytes(range(32))) -> Path:
    path = tmp_path / "custodian-key.bin"
    path.write_bytes(payload)
    os.chmod(path, 0o600)
    return path


def _read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _export_semantic_sha(
    split_root: Path, fixture_id: str, split: str, rows: list[dict[str, Any]]
) -> str:
    files = []
    for row in rows:
        image = (split_root / row["image_path"]).read_bytes()
        caption = (split_root / row["caption_path"]).read_bytes()
        files.append(
            {
                "row_id": row["row_id"],
                "image_path": row["image_path"],
                "image_sha256": hashlib.sha256(image).hexdigest(),
                "image_bytes": len(image),
                "caption_path": row["caption_path"],
                "caption_sha256": hashlib.sha256(caption).hexdigest(),
                "caption_bytes": len(caption),
            }
        )
    semantic = {
        "schema": 1,
        "kind": package._EXPORT_KIND,
        "fixture_id": fixture_id,
        "split": split,
        "files": files,
    }
    return hashlib.sha256(package.canonical_bytes(semantic)).hexdigest()


def _admission_records(
    base: Path,
    root: Path,
    manifests: list[Path],
    values: dict[str, dict],
    *,
    social_relief: bool = False,
) -> tuple[Path, Path]:
    base.mkdir(parents=True, exist_ok=True)
    selected_ids = [
        fixture_id
        for fixture_id in package._FIXTURES
        if not (social_relief and fixture_id == "W6-SOCIAL-A")
    ]
    manifest_by_id = {path.stem: path for path in manifests}
    plan_value = planner.build_plan(
        "materialization",
        social_relief=social_relief,
        product_a_reference=(root / ".references" / "W6-PRODUCT-A.png").absolute(),
        product_b_reference=(root / ".references" / "W6-PRODUCT-B.png").absolute(),
    )
    plan = base / "generation-plan.json"
    _write_canonical(plan, plan_value)
    plan_sha = hashlib.sha256(plan.read_bytes()).hexdigest()

    fixtures = []
    for fixture_id in selected_ids:
        qa_rows = []
        for source in values[fixture_id]["rows"]:
            checks = {
                "required_visible_text": "PASS",
                "caption_pixel_match": "PASS",
                "product_identity_preserved": (
                    True if fixture_id.startswith("W6-PRODUCT") else None
                ),
                "prohibited_brand_person_absent": "PASS",
                "useful_fixture": "PASS",
                "perceptual_duplicate_screen": "PASS",
            }
            qa_rows.append(
                {
                    "row_id": source["row_id"],
                    "image_sha256": source["image_sha256"],
                    "caption_sha256": source["caption_sha256"],
                    "checks": checks,
                }
            )
        fixtures.append(
            {
                "fixture_id": fixture_id,
                "source_manifest_sha256": hashlib.sha256(
                    manifest_by_id[fixture_id].read_bytes()
                ).hexdigest(),
                "row_count": len(qa_rows),
                "rows": qa_rows,
            }
        )
    qa_value = {
        "schema": 1,
        "kind": package._QA_KIND,
        "state": "PASS",
        "reviewer": "independent-human-reviewer",
        "reviewed_at_utc": "2026-08-05T18:00:00Z",
        "generation_plan_sha256": plan_sha,
        "social_relief": social_relief,
        "fixture_count": len(fixtures),
        "fixtures": fixtures,
    }
    qa = base / "visual-qa.json"
    _write_canonical(qa, qa_value)
    return plan, qa


def _publish(
    manifests: list[Path],
    root: Path,
    key: Path,
    output: Path,
    values: dict[str, dict],
    *,
    social_relief: bool = False,
):
    plan, qa = _admission_records(
        output.parent / f"{output.name}-admission",
        root,
        manifests,
        values,
        social_relief=social_relief,
    )
    return package.publish_package(
        manifests,
        root,
        key,
        output,
        generation_plan_path=plan,
        qa_receipt_path=qa,
        social_relief=social_relief,
    )


def test_publish_exact_matrix_counts_blinding_and_determinism(
    source_package: tuple[Path, list[Path], dict[str, dict]], tmp_path: Path
) -> None:
    root, manifests, values = source_package
    key_bytes = bytes(range(32))
    key = _key(tmp_path, key_bytes)
    first = tmp_path / "published-one"
    second = tmp_path / "published-two"

    acceptance = _publish(manifests, root, key, first, values)
    _publish(list(reversed(manifests)), root, key, second, values)

    assert acceptance["fixture_count"] == 6
    assert acceptance["social_relief"] is False
    assert acceptance["blinding"] == {
        "reveals_holdout_filenames": False,
        "reveals_holdout_captions": False,
        "reveals_holdout_pixels": False,
        "custodian_key_persisted": False,
    }
    assert acceptance["visual_qa"]["state"] == "pre-split-human-review-pass"
    assert acceptance["visual_qa"]["reviewer"] == "independent-human-reviewer"
    assert acceptance["visual_qa"]["reviewed_at_utc"] == "2026-08-05T18:00:00Z"
    expected_qa = tmp_path / "published-one-admission" / "visual-qa.json"
    assert acceptance["visual_qa"]["receipt_sha256"] == hashlib.sha256(
        expected_qa.read_bytes()
    ).hexdigest()
    assert set(acceptance["visual_qa"]) == {
        "state",
        "generation_plan_sha256",
        "receipt_sha256",
        "reviewer",
        "reviewed_at_utc",
    }
    assert not any(path.name == "visual-qa.json" for path in first.rglob("*"))
    assert (first / "blinded-acceptance.json").read_bytes() == (
        second / "blinded-acceptance.json"
    ).read_bytes()

    expected_counts = {
        "W6-DESIGN-GPT-A": (18, 3),
        "W6-DESIGN-NB-B": (18, 2),
        "W6-PRODUCT-A": (18, 3),
        "W6-PRODUCT-B": (18, 2),
        "W6-SOCIAL-A": (16, 2),
        "W6-DESIGN-GPT-LARGE": (43, 5),
    }
    acceptance_raw = (first / "blinded-acceptance.json").read_bytes()
    assert all(key_bytes not in path.read_bytes() for path in first.rglob("*.json"))
    for fixture in acceptance["fixtures"]:
        fixture_id = fixture["fixture_id"]
        train = _read(first / "train" / f"{fixture_id}.json")
        holdout = _read(first / "holdout" / f"{fixture_id}.json")
        assert (train["row_count"], holdout["row_count"]) == expected_counts[
            fixture_id
        ]
        assert {row["row_id"] for row in train["rows"]}.isdisjoint(
            row["row_id"] for row in holdout["rows"]
        )
        assert train["source_manifest_sha256"] == holdout[
            "source_manifest_sha256"
        ]
        assert train["export_semantic_sha256"] == fixture[
            "train_export_semantic_sha256"
        ] == _export_semantic_sha(first / "train", fixture_id, "train", train["rows"])
        assert holdout["export_semantic_sha256"] == fixture[
            "holdout_export_semantic_sha256"
        ] == _export_semantic_sha(
            first / "holdout", fixture_id, "holdout", holdout["rows"]
        )
        source_by_id = {
            row["row_id"]: row for row in values[fixture_id]["rows"]
        }
        for split_name, split_record in (("train", train), ("holdout", holdout)):
            split_root = first / split_name
            for exported in split_record["rows"]:
                source = source_by_id[exported["row_id"]]
                image_path = split_root / exported["image_path"]
                caption_path = split_root / exported["caption_path"]
                assert image_path.stem == caption_path.stem == exported["row_id"]
                assert image_path.read_bytes() == (root / source["image_path"]).read_bytes()
                assert caption_path.read_bytes() == (
                    root / source["caption_path"]
                ).read_bytes()
                assert image_path.stat().st_ino != (root / source["image_path"]).stat().st_ino
                assert caption_path.stat().st_ino != (
                    root / source["caption_path"]
                ).stat().st_ino
        train_paths = [
            path.relative_to(first / "train").as_posix()
            for path in (first / "train").rglob("*")
        ]
        train_payloads = [
            path.read_bytes() for path in (first / "train").rglob("*") if path.is_file()
        ]
        train_manifest_bytes = (first / "train" / f"{fixture_id}.json").read_bytes()
        for holdout_row in holdout["rows"]:
            assert holdout_row["row_id"].encode() not in acceptance_raw
            assert holdout_row["image_path"].encode() not in acceptance_raw
            assert holdout_row["caption_path"].encode() not in acceptance_raw
            assert holdout_row["image_sha256"].encode() not in acceptance_raw
            assert holdout_row["caption_sha256"].encode() not in acceptance_raw
            source = source_by_id[holdout_row["row_id"]]
            assert not any(holdout_row["row_id"] in path for path in train_paths)
            assert holdout_row["row_id"].encode() not in train_manifest_bytes
            assert (root / source["image_path"]).read_bytes() not in train_payloads
            assert (root / source["caption_path"]).read_bytes() not in train_payloads
        assert fixture["caption_length_statistics"]["count"] == fixture[
            "total_count"
        ]
        if fixture_id == "W6-SOCIAL-A":
            assert fixture["aspect_ratio_distribution"] == {
                "1:1": 6,
                "4:5": 6,
                "16:9": 6,
            }
        else:
            expected_distribution: dict[str, int] = {}
            for row in values[fixture_id]["rows"]:
                ratio = row["aspect_ratio"]
                expected_distribution[ratio] = expected_distribution.get(ratio, 0) + 1
            assert fixture["aspect_ratio_distribution"] == dict(
                sorted(expected_distribution.items())
            )
        assert fixture["rights_status_distribution"] == {
            "approved-provider-terms": fixture["total_count"]
        }

    # The export is an independent byte copy: later source mutation cannot
    # alter the staged trainer tree.
    first_train_manifest = _read(first / "train" / "W6-DESIGN-GPT-A.json")
    exported = first_train_manifest["rows"][0]
    source = next(
        row
        for row in values["W6-DESIGN-GPT-A"]["rows"]
        if row["row_id"] == exported["row_id"]
    )
    exported_path = first / "train" / exported["image_path"]
    exported_before = exported_path.read_bytes()
    (root / source["image_path"]).write_bytes(b"mutated-after-publication")
    assert exported_path.read_bytes() == exported_before


def test_different_key_changes_membership_but_not_source_validation(
    source_package: tuple[Path, list[Path], dict[str, dict]], tmp_path: Path
) -> None:
    root, manifests, values = source_package
    package.validate_source_manifest(manifests[0], root)
    _publish(
        manifests,
        root,
        _key(tmp_path, b"a" * 32),
        tmp_path / "first",
        values,
    )
    second_key = tmp_path / "other-key.bin"
    second_key.write_bytes(b"b" * 32)
    os.chmod(second_key, 0o600)
    _publish(manifests, root, second_key, tmp_path / "second", values)

    first_holdouts = [
        [row["row_id"] for row in _read(path)["rows"]]
        for path in sorted((tmp_path / "first" / "holdout").glob("*.json"))
    ]
    second_holdouts = [
        [row["row_id"] for row in _read(path)["rows"]]
        for path in sorted((tmp_path / "second" / "holdout").glob("*.json"))
    ]
    assert first_holdouts != second_holdouts


def test_rejects_noncanonical_or_mutable_source_manifest(
    source_package: tuple[Path, list[Path], dict[str, dict]], tmp_path: Path
) -> None:
    root, manifests, values = source_package
    manifest = manifests[0]
    manifest.write_text(json.dumps(values[next(iter(package._FIXTURES))], indent=2))
    with pytest.raises(ValueError, match="canonical JSON"):
        package.validate_source_manifest(manifest, root)

    manifest.unlink()
    target = tmp_path / "real.json"
    _write_canonical(target, values[next(iter(package._FIXTURES))])
    manifest.symlink_to(target)
    with pytest.raises(ValueError, match="cannot be opened safely"):
        package.validate_source_manifest(manifest, root)

    fifo = tmp_path / "manifest.fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="must be a regular file"):
        package.validate_source_manifest(fifo, root)


def test_rejects_wrong_count_and_incomplete_matrix(
    source_package: tuple[Path, list[Path], dict[str, dict]], tmp_path: Path
) -> None:
    root, manifests, values = source_package
    fixture_id = "W6-DESIGN-GPT-A"
    values[fixture_id]["rows"].pop()
    values[fixture_id]["total_pairs"] = 20
    _write_canonical(tmp_path / "short.json", values[fixture_id])
    with pytest.raises(ValueError, match="total_pairs must equal"):
        package.validate_source_manifest(tmp_path / "short.json", root)

    with pytest.raises(ValueError, match="exactly 6 source manifests"):
        package.publish_package(
            manifests[:-1],
            root,
            _key(tmp_path),
            tmp_path / "published",
            generation_plan_path=tmp_path / "unused-plan.json",
            qa_receipt_path=tmp_path / "unused-qa.json",
        )


def test_social_relief_accepts_only_the_predeclared_five_fixture_set(
    source_package: tuple[Path, list[Path], dict[str, dict]], tmp_path: Path
) -> None:
    root, manifests, values = source_package
    reduced = [path for path in manifests if path.stem != "W6-SOCIAL-A"]
    key = _key(tmp_path)
    acceptance = _publish(
        reduced,
        root,
        key,
        tmp_path / "published-relief",
        values,
        social_relief=True,
    )
    assert acceptance["fixture_count"] == 5
    assert acceptance["social_relief"] is True
    assert {row["fixture_id"] for row in acceptance["fixtures"]} == {
        fixture_id for fixture_id in package._FIXTURES if fixture_id != "W6-SOCIAL-A"
    }
    with pytest.raises(ValueError, match="exactly 6 source manifests"):
        package.publish_package(
            reduced,
            root,
            key,
            tmp_path / "strict-package",
            generation_plan_path=tmp_path / "unused-plan.json",
            qa_receipt_path=tmp_path / "unused-qa.json",
        )


def test_rejects_generator_label_provider_or_model_mismatch(
    source_package: tuple[Path, list[Path], dict[str, dict]], tmp_path: Path
) -> None:
    root, _, values = source_package
    fixture_id = "W6-DESIGN-GPT-A"
    values[fixture_id]["rows"][0]["provider"] = "gemini"
    values[fixture_id]["rows"][0]["model"] = package._GEMINI_MODEL
    values[fixture_id]["rows"][0]["rights_reference"] = package._GEMINI_RIGHTS
    manifest = tmp_path / "wrong-generator.json"
    _write_canonical(manifest, values[fixture_id])
    with pytest.raises(ValueError, match="does not match generator_label"):
        package.validate_source_manifest(manifest, root)


@pytest.mark.parametrize(
    "rights_reference",
    [
        "https://example.com/rights",
        "https://creativecommons.org/publicdomain/zero/1.0/",
        "https://creativecommons.org/licenses/by/4.0/",
    ],
)
def test_rejects_nonprovider_rights_urls(
    source_package: tuple[Path, list[Path], dict[str, dict]],
    tmp_path: Path,
    rights_reference: str,
) -> None:
    root, _, values = source_package
    fixture_id = "W6-DESIGN-GPT-A"
    values[fixture_id]["rows"][0]["rights_reference"] = rights_reference
    manifest = tmp_path / "wrong-rights-url.json"
    _write_canonical(manifest, values[fixture_id])
    with pytest.raises(ValueError, match="pinned provider terms"):
        package.validate_source_manifest(manifest, root)


@pytest.mark.parametrize(
    "rights_status",
    ["approved-public-domain", "approved-cc0", "approved-cc-by", "approved-cc-by-sa"],
)
def test_rejects_nonprovider_rights_statuses(
    source_package: tuple[Path, list[Path], dict[str, dict]],
    tmp_path: Path,
    rights_status: str,
) -> None:
    root, _, values = source_package
    fixture_id = "W6-DESIGN-GPT-A"
    values[fixture_id]["rows"][0]["rights_status"] = rights_status
    manifest = tmp_path / "wrong-rights-status.json"
    _write_canonical(manifest, values[fixture_id])
    with pytest.raises(ValueError, match="must be approved-provider-terms"):
        package.validate_source_manifest(manifest, root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reference_rights_reference", "https://example.com/terms"),
        ("reference_rights_status", "approved-cc0"),
    ],
)
def test_rejects_unbound_product_reference_rights(
    source_package: tuple[Path, list[Path], dict[str, dict]],
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    root, _, values = source_package
    fixture_id = "W6-PRODUCT-A"
    values[fixture_id]["reference_provenance"][field] = value
    manifest = tmp_path / f"bad-reference-{field}.json"
    _write_canonical(manifest, values[fixture_id])
    with pytest.raises(ValueError, match="reference_provenance identity mismatch"):
        package.validate_source_manifest(manifest, root)


def test_rejects_missing_or_failing_visual_qa_receipt(
    source_package: tuple[Path, list[Path], dict[str, dict]], tmp_path: Path
) -> None:
    root, manifests, values = source_package
    plan, qa = _admission_records(tmp_path / "admission", root, manifests, values)
    qa.unlink()
    with pytest.raises(ValueError, match="cannot be opened safely"):
        package.validate_admission_inputs(manifests, root, plan, qa)

    plan, qa = _admission_records(tmp_path / "admission", root, manifests, values)
    receipt = _read(qa)
    receipt["state"] = "FAIL"
    _write_canonical(qa, receipt)
    with pytest.raises(ValueError, match="identity or state mismatch"):
        package.validate_admission_inputs(manifests, root, plan, qa)


def test_rejects_incomplete_or_byte_forged_visual_qa_receipt(
    source_package: tuple[Path, list[Path], dict[str, dict]], tmp_path: Path
) -> None:
    root, manifests, values = source_package
    plan, qa = _admission_records(tmp_path / "admission", root, manifests, values)
    receipt = _read(qa)
    receipt["fixtures"][0]["rows"].pop()
    _write_canonical(qa, receipt)
    with pytest.raises(ValueError, match="row count mismatch"):
        package.validate_admission_inputs(manifests, root, plan, qa)

    plan, qa = _admission_records(tmp_path / "admission", root, manifests, values)
    receipt = _read(qa)
    receipt["fixtures"][0]["rows"][1] = receipt["fixtures"][0]["rows"][0]
    _write_canonical(qa, receipt)
    with pytest.raises(ValueError, match="row identity mismatch"):
        package.validate_admission_inputs(manifests, root, plan, qa)

    plan, qa = _admission_records(tmp_path / "admission", root, manifests, values)
    receipt = _read(qa)
    receipt["fixtures"][0]["rows"][0]["image_sha256"] = "f" * 64
    _write_canonical(qa, receipt)
    with pytest.raises(ValueError, match="byte binding mismatch"):
        package.validate_admission_inputs(manifests, root, plan, qa)

    plan, qa = _admission_records(tmp_path / "admission", root, manifests, values)
    receipt = _read(qa)
    receipt["fixtures"][0]["rows"][0]["caption_sha256"] = "d" * 64
    _write_canonical(qa, receipt)
    with pytest.raises(ValueError, match="byte binding mismatch"):
        package.validate_admission_inputs(manifests, root, plan, qa)

    plan, qa = _admission_records(tmp_path / "admission", root, manifests, values)
    receipt = _read(qa)
    receipt["fixtures"][0]["source_manifest_sha256"] = "e" * 64
    _write_canonical(qa, receipt)
    with pytest.raises(ValueError, match="source binding mismatch"):
        package.validate_admission_inputs(manifests, root, plan, qa)


def test_rejects_inapplicable_or_failed_semantic_visual_checks(
    source_package: tuple[Path, list[Path], dict[str, dict]], tmp_path: Path
) -> None:
    root, manifests, values = source_package
    plan, qa = _admission_records(tmp_path / "admission", root, manifests, values)
    receipt = _read(qa)
    receipt["fixtures"][0]["rows"][0]["checks"][
        "product_identity_preserved"
    ] = True
    _write_canonical(qa, receipt)
    with pytest.raises(ValueError, match="checks are incomplete or failing"):
        package.validate_admission_inputs(manifests, root, plan, qa)

    plan, qa = _admission_records(tmp_path / "admission", root, manifests, values)
    receipt = _read(qa)
    receipt["fixtures"][0]["rows"][0]["checks"][
        "required_visible_text"
    ] = "NOT_APPLICABLE"
    _write_canonical(qa, receipt)
    with pytest.raises(ValueError, match="checks are incomplete or failing"):
        package.validate_admission_inputs(manifests, root, plan, qa)

    plan, qa = _admission_records(tmp_path / "admission", root, manifests, values)
    receipt = _read(qa)
    product = next(
        fixture
        for fixture in receipt["fixtures"]
        if fixture["fixture_id"].startswith("W6-PRODUCT")
    )
    product["rows"][0]["checks"]["product_identity_preserved"] = None
    _write_canonical(qa, receipt)
    with pytest.raises(ValueError, match="checks are incomplete or failing"):
        package.validate_admission_inputs(manifests, root, plan, qa)


def test_rejects_visual_qa_that_predates_generated_bytes(
    source_package: tuple[Path, list[Path], dict[str, dict]], tmp_path: Path
) -> None:
    root, manifests, values = source_package
    plan, qa = _admission_records(tmp_path / "admission", root, manifests, values)
    receipt = _read(qa)
    receipt["reviewed_at_utc"] = "2026-08-05T11:59:59Z"
    _write_canonical(qa, receipt)
    with pytest.raises(ValueError, match="must not predate generated source bytes"):
        package.validate_admission_inputs(manifests, root, plan, qa)


@pytest.mark.parametrize("field", ["id", "prompt", "caption"])
def test_rejects_generation_plan_forgery_even_with_rebound_qa_hash(
    source_package: tuple[Path, list[Path], dict[str, dict]],
    tmp_path: Path,
    field: str,
) -> None:
    root, manifests, values = source_package
    plan, qa = _admission_records(tmp_path / "admission", root, manifests, values)
    forged_plan = _read(plan)
    forged_plan["fixtures"][0]["rows"][0][field] = {
        "id": "arbitrary-safe-id",
        "prompt": "Arbitrary but structurally valid replacement prompt.",
        "caption": "Arbitrary but structurally valid replacement caption.",
    }[field]
    _write_canonical(plan, forged_plan)
    receipt = _read(qa)
    receipt["generation_plan_sha256"] = hashlib.sha256(plan.read_bytes()).hexdigest()
    _write_canonical(qa, receipt)
    with pytest.raises(ValueError, match="differs from the frozen blueprint"):
        package.validate_admission_inputs(manifests, root, plan, qa)


def test_rejects_product_reference_drift_and_cross_fixture_reuse(
    source_package: tuple[Path, list[Path], dict[str, dict]], tmp_path: Path
) -> None:
    root, manifests, values = source_package
    fixture_id = "W6-PRODUCT-A"
    values[fixture_id]["rows"][0]["reference_sha256"] = "c" * 64
    drifted = tmp_path / "drifted-reference.json"
    _write_canonical(drifted, values[fixture_id])
    with pytest.raises(ValueError, match="one canonical reference"):
        package.validate_source_manifest(drifted, root)

    root, manifests, values = _source_package(tmp_path / "same-reference")
    fixture_id = "W6-PRODUCT-B"
    reused_reference = values["W6-PRODUCT-A"]["rows"][0]["reference_sha256"]
    for row in values[fixture_id]["rows"]:
        row["reference_sha256"] = reused_reference
    values[fixture_id]["reference_provenance"][
        "reference_output_sha256"
    ] = reused_reference
    reused = tmp_path / "same-reference" / "reused.json"
    _write_canonical(reused, values[fixture_id])
    manifests = [reused if path.stem == fixture_id else path for path in manifests]
    with pytest.raises(ValueError, match="must use distinct references"):
        package.publish_package(
            manifests,
            root,
            _key(tmp_path / "same-reference"),
            tmp_path / "same-reference" / "published",
            generation_plan_path=tmp_path / "unused-plan.json",
            qa_receipt_path=tmp_path / "unused-qa.json",
        )


def test_rejects_social_ratio_coverage_drift(
    source_package: tuple[Path, list[Path], dict[str, dict]], tmp_path: Path
) -> None:
    root, _, values = source_package
    fixture_id = "W6-SOCIAL-A"
    row = next(
        candidate
        for candidate in values[fixture_id]["rows"]
        if candidate["aspect_ratio"] == "1:1"
    )
    image = BytesIO()
    Image.new("RGB", (8, 10), (99, 88, 77)).save(image, format="PNG")
    raw = image.getvalue()
    (root / row["image_path"]).write_bytes(raw)
    row["width"], row["height"], row["aspect_ratio"] = 8, 10, "4:5"
    row["image_sha256"] = row["output_sha256"] = hashlib.sha256(raw).hexdigest()
    manifest = tmp_path / "bad-social-ratios.json"
    _write_canonical(manifest, values[fixture_id])
    with pytest.raises(ValueError, match="six rows at each required ratio"):
        package.validate_source_manifest(manifest, root)


def test_rejects_byte_hash_aspect_rights_and_phantom_paths(
    source_package: tuple[Path, list[Path], dict[str, dict]], tmp_path: Path
) -> None:
    root, _, values = source_package
    fixture_id = "W6-DESIGN-GPT-A"
    row = values[fixture_id]["rows"][0]

    row["image_sha256"] = "0" * 64
    _write_canonical(tmp_path / "bad-hash.json", values[fixture_id])
    with pytest.raises(ValueError, match="image hash mismatch"):
        package.validate_source_manifest(tmp_path / "bad-hash.json", root)

    row["image_sha256"] = row["output_sha256"]
    row["aspect_ratio"] = "1200:800"
    _write_canonical(tmp_path / "bad-ratio.json", values[fixture_id])
    with pytest.raises(ValueError, match="aspect_ratio must be 3:2"):
        package.validate_source_manifest(tmp_path / "bad-ratio.json", root)

    row["aspect_ratio"] = "3:2"
    row["width"] = 1200
    row["height"] = 800
    _write_canonical(tmp_path / "bad-dimensions.json", values[fixture_id])
    with pytest.raises(ValueError, match="declared dimensions do not match"):
        package.validate_source_manifest(tmp_path / "bad-dimensions.json", root)

    row["width"] = 12
    row["height"] = 8
    row["rights_status"] = "unknown"
    _write_canonical(tmp_path / "bad-rights.json", values[fixture_id])
    with pytest.raises(ValueError, match="approved-provider-terms"):
        package.validate_source_manifest(tmp_path / "bad-rights.json", root)

    row["rights_status"] = "approved-provider-terms"
    actual_image = root / row["image_path"]
    actual_image.write_bytes(b"not-an-image")
    row["image_sha256"] = hashlib.sha256(b"not-an-image").hexdigest()
    row["output_sha256"] = row["image_sha256"]
    _write_canonical(tmp_path / "bad-image.json", values[fixture_id])
    with pytest.raises(ValueError, match="not a valid supported image"):
        package.validate_source_manifest(tmp_path / "bad-image.json", root)

    # Restore the entire clean source package before exercising a phantom path.
    root, _, values = _source_package(tmp_path / "restored")
    row = values[fixture_id]["rows"][0]
    (root / row["image_path"]).unlink()
    (root / row["caption_path"]).unlink()
    _write_canonical(tmp_path / "phantom.json", values[fixture_id])
    with pytest.raises(ValueError, match="cannot be opened safely"):
        package.validate_source_manifest(tmp_path / "phantom.json", root)


def test_rejects_within_and_cross_fixture_duplicates(
    source_package: tuple[Path, list[Path], dict[str, dict]], tmp_path: Path
) -> None:
    root, manifests, values = source_package
    first_id = "W6-DESIGN-GPT-A"
    first = values[first_id]["rows"][0]
    second = values[first_id]["rows"][1]
    second["request_id"] = first["request_id"]
    _write_canonical(tmp_path / "duplicate-local.json", values[first_id])
    with pytest.raises(ValueError, match="duplicate provider/request_id"):
        package.validate_source_manifest(tmp_path / "duplicate-local.json", root)

    # Rebuild a clean package and bind one row in a second fixture to the exact
    # same image bytes and hash as a row in the first fixture.
    root, manifests, values = _source_package(tmp_path / "fresh")
    first = values[first_id]["rows"][0]
    second_id, cross = next(
        (fixture_id, row)
        for fixture_id, manifest in values.items()
        if fixture_id != first_id
        for row in manifest["rows"]
        if (row["width"], row["height"]) == (first["width"], first["height"])
    )
    source_image = root / first["image_path"]
    target_image = root / cross["image_path"]
    target_image.write_bytes(source_image.read_bytes())
    cross["image_sha256"] = first["image_sha256"]
    cross["output_sha256"] = first["image_sha256"]
    replacement = tmp_path / "fresh" / "cross.json"
    _write_canonical(replacement, values[second_id])
    manifests = [replacement if path.name == f"{second_id}.json" else path for path in manifests]
    with pytest.raises(ValueError, match="cross-fixture duplicate image_sha256"):
        package.publish_package(
            manifests,
            root,
            _key(tmp_path / "fresh"),
            tmp_path / "fresh" / "published",
            generation_plan_path=tmp_path / "unused-plan.json",
            qa_receipt_path=tmp_path / "unused-qa.json",
        )


def test_publish_is_create_only_and_failed_publish_leaves_no_package(
    source_package: tuple[Path, list[Path], dict[str, dict]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifests, values = source_package
    key = _key(tmp_path)
    output = tmp_path / "published"
    _publish(manifests, root, key, output, values)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _publish(manifests, root, key, output, values)

    real_atomic = package._atomic_create
    calls = 0

    def fail_second(path: Path, value: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated publish failure")
        return real_atomic(path, value)

    monkeypatch.setattr(package, "_atomic_create", fail_second)
    failed = tmp_path / "failed"
    with pytest.raises(OSError, match="simulated publish failure"):
        _publish(manifests, root, key, failed, values)
    assert not failed.exists()


def test_corrupt_export_copy_aborts_and_removes_partial_package(
    source_package: tuple[Path, list[Path], dict[str, dict]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifests, values = source_package
    original = package._write_exclusive_bytes
    corrupted = False

    def corrupt_first_image(path: Path, payload: bytes) -> tuple[str, int]:
        nonlocal corrupted
        if not corrupted and path.suffix != ".txt":
            corrupted = True
            return original(path, payload + b"corruption")
        return original(path, payload)

    monkeypatch.setattr(package, "_write_exclusive_bytes", corrupt_first_image)
    output = tmp_path / "corrupt-copy"
    with pytest.raises(RuntimeError, match="exported split hash mismatch"):
        _publish(manifests, root, _key(tmp_path), output, values)
    assert corrupted and not output.exists()


def test_key_descriptor_policy_rejects_mode_owner_links_and_special_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    permissive_dir = tmp_path / "permissive"
    permissive_dir.mkdir()
    permissive = _key(permissive_dir)
    os.chmod(permissive, 0o640)
    with pytest.raises(ValueError, match="mode must be exactly 0600"):
        package._load_key(permissive)

    foreign_dir = tmp_path / "foreign"
    foreign_dir.mkdir()
    foreign = _key(foreign_dir)
    monkeypatch.setattr(package.os, "geteuid", lambda: os.stat(foreign).st_uid + 1)
    with pytest.raises(ValueError, match="owned by the effective UID"):
        package._load_key(foreign)
    monkeypatch.undo()

    hardlink_dir = tmp_path / "hardlink"
    hardlink_dir.mkdir()
    hardlinked = _key(hardlink_dir)
    os.link(hardlinked, hardlink_dir / "second-name.bin")
    with pytest.raises(ValueError, match="exactly one hard link"):
        package._load_key(hardlinked)

    symlink_dir = tmp_path / "symlink"
    symlink_dir.mkdir()
    target = _key(symlink_dir)
    link = symlink_dir / "key-link.bin"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="cannot be opened safely"):
        package._load_key(link)

    fifo = tmp_path / "key.fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        package._load_key(fifo)


def test_loaded_key_buffer_is_zeroed_after_publish(
    source_package: tuple[Path, list[Path], dict[str, dict]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifests, values = source_package
    original = package._load_key
    captured: list[bytearray] = []

    def capture(path: Path) -> bytearray:
        value = original(path)
        captured.append(value)
        return value

    monkeypatch.setattr(package, "_load_key", capture)
    _publish(manifests, root, _key(tmp_path), tmp_path / "zeroed", values)
    assert len(captured) == 1 and not any(captured[0])


def test_key_constraints_and_no_path_traversal(
    source_package: tuple[Path, list[Path], dict[str, dict]], tmp_path: Path
) -> None:
    root, manifests, values = source_package
    with pytest.raises(ValueError, match="between 32 and 4096"):
        _publish(
            manifests,
            root,
            _key(tmp_path, b"short"),
            tmp_path / "bad-key",
            values,
        )
    assert not (tmp_path / "bad-key").exists()

    fixture_id = "W6-DESIGN-GPT-A"
    values[fixture_id]["rows"][0]["image_path"] = "../escape.png"
    _write_canonical(tmp_path / "traversal.json", values[fixture_id])
    with pytest.raises(ValueError, match="safe canonical relative path"):
        package.validate_source_manifest(tmp_path / "traversal.json", root)
