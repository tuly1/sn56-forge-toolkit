"""End-to-end contracts for Lane-B receipt-to-manifest compilation."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image
import pytest


_CALIBRATION = Path(__file__).parents[1] / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))

import lane_b_fixture_package as package  # noqa: E402
import lane_b_generate as generator  # noqa: E402
import lane_b_manifest as compiler  # noqa: E402
import lane_b_plan as planner  # noqa: E402


def _png(index: int, size: tuple[int, int]) -> bytes:
    output = BytesIO()
    Image.new(
        "RGB",
        size,
        (index % 251, (index * 29) % 251, (index * 83) % 251),
    ).save(output, format="PNG")
    return output.getvalue()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _row(
    fixture_id: str,
    index: int,
    *,
    provider: str,
    aspect_ratio: str,
    reference: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "id": f"row-{index:03d}",
        "provider": provider,
        "model": (
            generator._OPENAI_MODEL
            if provider == "openai"
            else generator._GEMINI_MODEL
        ),
        "prompt": f"Render synthetic identity {fixture_id} variation {index:03d}.",
        "caption": f"Synthetic {fixture_id} composition {index:03d}.",
        "rights_reference": (
            "https://openai.com/policies/services-agreement/"
            if provider == "openai"
            else "https://ai.google.dev/gemini-api/terms"
        ),
        "aspect_ratio": aspect_ratio,
        "quality": "medium",
        "reference": reference,
    }


def _write_plan(
    tmp_path: Path,
    references: dict[str, dict[str, str]],
    *,
    social_relief: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    plan = planner.build_plan(
        "materialization",
        social_relief=social_relief,
        product_a_reference=Path(references["W6-PRODUCT-A"]["path"]),
        product_b_reference=Path(references["W6-PRODUCT-B"]["path"]),
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(planner.canonical_bytes(plan) + b"\n")
    return plan_path, tmp_path / "generated", plan


def _write_reference_plan(tmp_path: Path) -> Path:
    plan = planner.build_plan("reference-assets")
    path = tmp_path / "reference-plan.json"
    path.write_bytes(planner.canonical_bytes(plan) + b"\n")
    return path


def _materialize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    social_relief: bool = False,
    synthetic_materialization: bool = False,
    copied_references: bool = False,
) -> tuple[Path, Path, dict[str, Any], Path, Path]:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-never-persist")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-never-persist")
    call = 0

    def transport(url, _headers, body, _timeout):
        nonlocal call
        call += 1
        payload = json.loads(body)
        if "openai.com" in url:
            requested = payload["size"]
            size = {
                "1024x1024": (32, 32),
                "1536x1024": (48, 32),
                "1024x1536": (32, 48),
            }[requested]
            encoded = base64.b64encode(_png(call, size)).decode("ascii")
            response = {"data": [{"b64_json": encoded}]}
            headers = {"x-request-id": f"openai-request-{call:03d}"}
        else:
            requested = payload["generationConfig"]["responseFormat"]["image"][
                "aspectRatio"
            ]
            size = {
                "1:1": (32, 32),
                "3:2": (48, 32),
                "3:4": (24, 32),
                "4:3": (32, 24),
                "4:5": (32, 40),
                "9:16": (27, 48),
                "16:9": (48, 27),
            }[requested]
            encoded = base64.b64encode(_png(call, size)).decode("ascii")
            response = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": encoded,
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
            headers = {"x-goog-request-id": f"gemini-request-{call:03d}"}
        return generator.HttpResponse(200, headers, json.dumps(response).encode())

    monkeypatch.setattr(generator, "_http_transport", transport)
    reference_plan = _write_reference_plan(tmp_path)
    reference_plan_value = json.loads(reference_plan.read_bytes())
    reference_root = tmp_path / "reference-generated"
    reference_summary = generator.execute(reference_plan, reference_root)
    assert reference_summary["row_count"] == 2
    references: dict[str, dict[str, str]] = {}
    for fixture_id in ("W6-PRODUCT-A", "W6-PRODUCT-B"):
        row_id = next(
            fixture["rows"][0]["id"]
            for fixture in reference_plan_value["fixtures"]
            if fixture["id"] == fixture_id
        )
        reference_path = (
            reference_root / fixture_id / "images" / f"{row_id}.png"
        )
        if copied_references:
            copied = tmp_path / f"copied-{fixture_id}-reference.png"
            copied.write_bytes(reference_path.read_bytes())
            reference_path = copied
        raw = reference_path.read_bytes()
        references[fixture_id] = {
            "path": str(reference_path),
            "sha256": _sha256(raw),
        }
    plan_path, generation_root, plan = _write_plan(
        tmp_path, references, social_relief=social_relief
    )
    if synthetic_materialization:
        summary = generator.execute(
            plan_path, generation_root, transport=transport
        )
    else:
        summary = generator.execute(plan_path, generation_root)
    assert summary["row_count"] == sum(len(row["rows"]) for row in plan["fixtures"])
    return plan_path, generation_root, plan, reference_plan, reference_root


def test_compiles_full_generation_and_package_accepts_every_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, generated, _, reference_plan, reference_root = _materialize(
        tmp_path, monkeypatch
    )
    manifests = compiler.compile_manifests(
        plan,
        generated,
        tmp_path / "manifests",
        reference_plan_path=reference_plan,
        reference_generation_root=reference_root,
        rights_status="approved-provider-terms",
    )

    assert len(manifests) == 6
    for manifest_path in manifests:
        raw = manifest_path.read_bytes()
        value = json.loads(raw)
        assert raw == package.canonical_bytes(value) + b"\n"
        validated = package.validate_source_manifest(manifest_path, generated)
        assert validated["manifest"]["fixture_id"] == manifest_path.stem
        for row in validated["rows"]:
            assert row["rights_status"] == "approved-provider-terms"
            assert "fixture_id" not in row
    product = json.loads(
        (tmp_path / "manifests" / "W6-PRODUCT-A.json").read_bytes()
    )
    assert len({row["reference_sha256"] for row in product["rows"]}) == 1
    assert product["generator_label"] == (
        "GPT Image 2 reference → Nano Banana 2 edits"
    )
    provenance = product["reference_provenance"]
    assert provenance == {
        "schema": 1,
        "kind": "sn56-week6-lane-b-reference-provenance",
        "fixture_id": "W6-PRODUCT-A",
        "row_id": "pa-canonical-reference",
        "provider": "openai",
        "model": generator._OPENAI_MODEL,
        "reference_plan_file_sha256": _sha256(reference_plan.read_bytes()),
        "reference_generation_summary_sha256": _sha256(
            (reference_root / "generation-summary.json").read_bytes()
        ),
        "reference_success_receipt_sha256": provenance[
            "reference_success_receipt_sha256"
        ],
        "reference_request_payload_sha256": provenance[
            "reference_request_payload_sha256"
        ],
        "reference_output_sha256": product["rows"][0]["reference_sha256"],
        "reference_rights_reference": generator._PROVIDER_RIGHTS["openai"],
        "reference_rights_status": "approved-provider-terms",
    }
    assert len(provenance["reference_success_receipt_sha256"]) == 64
    assert len(provenance["reference_request_payload_sha256"]) == 64
    non_product = json.loads(
        (tmp_path / "manifests" / "W6-DESIGN-GPT-A.json").read_bytes()
    )
    assert non_product["reference_provenance"] is None


def test_actual_output_tamper_and_receipt_plan_mismatch_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, generated, _, reference_plan, reference_root = _materialize(
        tmp_path, monkeypatch
    )
    row_id = json.loads(plan.read_bytes())["fixtures"][0]["rows"][0]["id"]
    image = generated / "W6-DESIGN-GPT-A" / "images" / f"{row_id}.png"
    original_image = image.read_bytes()
    image.write_bytes(_png(250, (32, 32)))
    with pytest.raises(compiler.ManifestCompileError, match="actual bytes"):
        compiler.compile_manifests(
            plan,
            generated,
            tmp_path / "tampered-image-manifests",
            reference_plan_path=reference_plan,
            reference_generation_root=reference_root,
            rights_status="approved-provider-terms",
        )

    image.write_bytes(original_image)
    fixture = generated / "W6-DESIGN-GPT-A"
    success = fixture / f"{row_id}.success.json"
    attempt = fixture / "attempts" / f"{row_id}.attempt-001.json"
    receipt_raw = success.read_bytes()
    success.unlink()
    success.write_bytes(receipt_raw)
    with pytest.raises(
        compiler.ManifestCompileError, match="immutable attempt hard link"
    ):
        compiler.compile_manifests(
            plan,
            generated,
            tmp_path / "copied-receipt-manifests",
            reference_plan_path=reference_plan,
            reference_generation_root=reference_root,
            rights_status="approved-provider-terms",
        )

    # Restore the hard link, then mutate both names through that inode.  A
    # byte-stable link is necessary but cannot override the current plan.
    success.unlink()
    success.hardlink_to(attempt)
    receipt = json.loads(success.read_bytes())
    receipt["prompt"] = "A forged parallel prompt."
    success.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    with pytest.raises(compiler.ManifestCompileError, match="prompt does not match"):
        compiler.compile_manifests(
            plan,
            generated,
            tmp_path / "forged-receipt-manifests",
            reference_plan_path=reference_plan,
            reference_generation_root=reference_root,
            rights_status="approved-provider-terms",
        )


def test_summary_mismatch_and_implicit_social_relief_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, generated, _, reference_plan, reference_root = _materialize(
        tmp_path, monkeypatch, social_relief=True
    )
    with pytest.raises(compiler.ManifestCompileError, match="explicitly selected"):
        compiler.compile_manifests(
            plan,
            generated,
            tmp_path / "implicit-relief",
            reference_plan_path=reference_plan,
            reference_generation_root=reference_root,
            rights_status="approved-provider-terms",
        )

    manifests = compiler.compile_manifests(
        plan,
        generated,
        tmp_path / "explicit-relief",
        reference_plan_path=reference_plan,
        reference_generation_root=reference_root,
        rights_status="approved-provider-terms",
        social_relief=True,
    )
    assert len(manifests) == 5
    assert not any(path.stem == "W6-SOCIAL-A" for path in manifests)

    summary = generated / "generation-summary.json"
    value = json.loads(summary.read_bytes())
    value["success_count"] -= 1
    summary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(compiler.ManifestCompileError, match="summary does not exactly"):
        compiler.compile_manifests(
            plan,
            generated,
            tmp_path / "bad-summary",
            reference_plan_path=reference_plan,
            reference_generation_root=reference_root,
            rights_status="approved-provider-terms",
            social_relief=True,
        )


def test_refuses_unapproved_rights_and_existing_manifest_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, generated, _, reference_plan, reference_root = _materialize(
        tmp_path, monkeypatch, social_relief=True
    )
    with pytest.raises(compiler.ManifestCompileError, match="explicitly approved"):
        compiler.compile_manifests(
            plan,
            generated,
            tmp_path / "rights",
            reference_plan_path=reference_plan,
            reference_generation_root=reference_root,
            rights_status="unknown",
            social_relief=True,
        )
    with pytest.raises(compiler.ManifestCompileError, match="provider-terms"):
        compiler.compile_manifests(
            plan,
            generated,
            tmp_path / "misclassified-rights",
            reference_plan_path=reference_plan,
            reference_generation_root=reference_root,
            rights_status="approved-cc-by",
            social_relief=True,
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(compiler.ManifestCompileError, match="already exists"):
        compiler.compile_manifests(
            plan,
            generated,
            existing,
            reference_plan_path=reference_plan,
            reference_generation_root=reference_root,
            rights_status="approved-provider-terms",
            social_relief=True,
        )


def test_compiler_rejects_synthetic_transport_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, generated, _, reference_plan, reference_root = _materialize(
        tmp_path,
        monkeypatch,
        synthetic_materialization=True,
    )
    receipt = json.loads(
        (
            generated
            / "W6-DESIGN-GPT-A"
            / (
                json.loads(plan.read_bytes())["fixtures"][0]["rows"][0]["id"]
                + ".success.json"
            )
        ).read_bytes()
    )
    assert receipt["origin"] == "synthetic"
    with pytest.raises(compiler.ManifestCompileError, match="origin is not real"):
        compiler.compile_manifests(
            plan,
            generated,
            tmp_path / "synthetic-manifests",
            reference_plan_path=reference_plan,
            reference_generation_root=reference_root,
            rights_status="approved-provider-terms",
        )


def test_product_reference_must_be_exact_validated_reference_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, generated, _, reference_plan, reference_root = _materialize(
        tmp_path,
        monkeypatch,
        copied_references=True,
    )
    with pytest.raises(
        compiler.ManifestCompileError,
        match="reference is not the validated reference-assets output",
    ):
        compiler.compile_manifests(
            plan,
            generated,
            tmp_path / "copied-reference-manifests",
            reference_plan_path=reference_plan,
            reference_generation_root=reference_root,
            rights_status="approved-provider-terms",
        )


def test_reference_output_tamper_aborts_before_manifest_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, generated, _, reference_plan, reference_root = _materialize(
        tmp_path, monkeypatch
    )
    reference = (
        reference_root
        / "W6-PRODUCT-A"
        / "images"
        / "pa-canonical-reference.png"
    )
    reference.write_bytes(_png(249, (32, 32)))
    destination = tmp_path / "tampered-reference-manifests"
    with pytest.raises(
        compiler.ManifestCompileError,
        match="generation plan failed generator validation",
    ):
        compiler.compile_manifests(
            plan,
            generated,
            destination,
            reference_plan_path=reference_plan,
            reference_generation_root=reference_root,
            rights_status="approved-provider-terms",
        )
    assert not destination.exists()


def test_reference_rights_status_is_derived_from_actual_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, generated, _, reference_plan, reference_root = _materialize(
        tmp_path, monkeypatch
    )
    receipt_path = (
        reference_root / "W6-PRODUCT-A" / "pa-canonical-reference.success.json"
    )
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt["rights_status"] == "approved-provider-terms"
    receipt["rights_status"] = "approved-cc0"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    with pytest.raises(
        compiler.ManifestCompileError,
        match="receipt rights status is not approved-provider-terms",
    ):
        compiler.compile_manifests(
            plan,
            generated,
            tmp_path / "forged-reference-rights",
            reference_plan_path=reference_plan,
            reference_generation_root=reference_root,
            rights_status="approved-provider-terms",
        )


@pytest.mark.parametrize("field", ["id", "prompt", "caption"])
def test_compiler_rejects_correct_cardinality_nonblueprint_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    plan, generated, _, reference_plan, reference_root = _materialize(
        tmp_path, monkeypatch
    )
    forged = json.loads(plan.read_bytes())
    forged["fixtures"][0]["rows"][0][field] = {
        "id": "arbitrary-safe-id",
        "prompt": "Arbitrary but structurally valid replacement prompt.",
        "caption": "Arbitrary but structurally valid replacement caption.",
    }[field]
    plan.write_bytes(planner.canonical_bytes(forged) + b"\n")
    destination = tmp_path / f"forged-{field}-manifests"
    with pytest.raises(
        compiler.ManifestCompileError,
        match="generation plan failed generator validation",
    ):
        compiler.compile_manifests(
            plan,
            generated,
            destination,
            reference_plan_path=reference_plan,
            reference_generation_root=reference_root,
            rights_status="approved-provider-terms",
        )
    assert not destination.exists()
