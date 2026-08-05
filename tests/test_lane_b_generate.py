"""Contracts for the isolated Week-6 Lane-B image generator."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import os
from pathlib import Path
import stat
import sys

from PIL import Image
import pytest


_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT / "ops" / "calibration"))

import lane_b_generate as generator  # noqa: E402
import lane_b_plan as planner  # noqa: E402


def _png(
    color: tuple[int, int, int] = (20, 40, 60),
    size: tuple[int, int] = (32, 32),
) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def _row(
    *,
    provider: str = "openai",
    row_id: str = "row-001",
    reference: dict | None = None,
) -> dict:
    return {
        "id": row_id,
        "provider": provider,
        "model": (
            "gpt-image-2-2026-04-21"
            if provider == "openai"
            else "gemini-3.1-flash-image"
        ),
        "prompt": "Render the fictitious Aster Quay planning dashboard.",
        "caption": "Aster Quay planning dashboard with a blue status rail",
        "rights_reference": (
            "https://openai.com/policies/services-agreement/"
            if provider == "openai"
            else "https://ai.google.dev/gemini-api/terms"
        ),
        "aspect_ratio": "1024x1024" if provider == "openai" else "4:3",
        "quality": "medium",
        "reference": reference,
    }


def _plan(path: Path, row: dict, *, fixture_id: str = "W6-DESIGN-GPT-A") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "sn56-week6-lane-b-generation-plan",
                "purpose": "smoke",
                "social_relief": False,
                "fixtures": [{"id": fixture_id, "rows": [row]}],
            }
        ),
        encoding="utf-8",
    )
    return path


def _openai_response(image: bytes, *, status: int = 200) -> generator.HttpResponse:
    body = json.dumps(
        {"data": [{"b64_json": base64.b64encode(image).decode("ascii")}]}
    ).encode()
    return generator.HttpResponse(status, {"x-request-id": "req_fixture_123"}, body)


def _gemini_response(image: bytes) -> generator.HttpResponse:
    body = json.dumps(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "internal annotation", "thought": True},
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": base64.b64encode(image).decode("ascii"),
                                }
                            },
                        ]
                    }
                }
            ]
        }
    ).encode()
    return generator.HttpResponse(200, {"x-goog-request-id": "goog_fixture_456"}, body)


def test_openai_snapshot_generation_is_hash_bound_resumable_and_secret_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "sk-test-do-not-persist"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    plan = _plan(tmp_path / "plan.json", _row())
    output = tmp_path / "output"
    calls: list[dict] = []

    def transport(url, headers, body, timeout):
        assert url == "https://api.openai.com/v1/images/generations"
        assert headers["Authorization"] == f"Bearer {key}"
        assert timeout == 10.0
        payload = json.loads(body)
        assert payload == {
            "model": "gpt-image-2-2026-04-21",
            "n": 1,
            "output_format": "png",
            "prompt": "Render the fictitious Aster Quay planning dashboard.",
            "quality": "medium",
            "size": "1024x1024",
        }
        calls.append(payload)
        return _openai_response(_png())

    first = generator.execute(
        plan, output, transport=transport, timeout_s=10.0, max_attempts=2
    )
    second = generator.execute(
        plan, output, transport=transport, timeout_s=10.0, max_attempts=2
    )
    assert first == second
    assert first["row_count"] == first["success_count"] == 1
    assert len(calls) == 1

    receipt_path = output / "W6-DESIGN-GPT-A" / "row-001.success.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["origin"] == "synthetic"
    assert receipt["provider_request_id"] == "req_fixture_123"
    assert receipt["prompt"] == "Render the fictitious Aster Quay planning dashboard."
    assert receipt["width"] == 32 and receipt["height"] == 32
    assert receipt["output_image_sha256"] == generator._sha256_bytes(
        Path(receipt["output_path"]).read_bytes()
    )
    for evidence in output.rglob("*.json"):
        assert key not in evidence.read_text()


def test_gemini_reference_edit_uses_nano_banana_two_and_binds_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "gemini-test-do-not-persist"
    monkeypatch.setenv("GEMINI_API_KEY", key)
    reference = tmp_path / "reference.png"
    reference.write_bytes(_png((100, 80, 60)))
    reference_sha = generator._sha256_bytes(reference.read_bytes())
    row = _row(
        provider="gemini",
        reference={"path": str(reference), "sha256": reference_sha},
    )
    row["quality"] = "high"
    plan = _plan(tmp_path / "plan.json", row, fixture_id="W6-PRODUCT-A")
    captured: dict = {}

    def transport(url, headers, body, _timeout):
        assert url.endswith("gemini-3.1-flash-image:generateContent")
        assert headers["x-goog-api-key"] == key
        captured.update(json.loads(body))
        return _gemini_response(_png((2, 4, 8), (32, 24)))

    summary = generator.execute(plan, tmp_path / "output", transport=transport)
    assert summary["status"] == "complete"
    parts = captured["contents"][0]["parts"]
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    assert generator._sha256_bytes(base64.b64decode(parts[1]["inline_data"]["data"])) == reference_sha
    image_config = captured["generationConfig"]["responseFormat"]["image"]
    assert image_config == {"aspectRatio": "4:3", "imageSize": "2K"}
    receipt = json.loads(
        (tmp_path / "output" / "W6-PRODUCT-A" / "row-001.success.json").read_text()
    )
    assert receipt["reference_sha256"] == reference_sha
    assert receipt["provider_request_id"] == "goog_fixture_456"
    assert receipt["origin"] == "synthetic"


def test_default_transport_is_real_and_injected_transport_is_synthetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    plan = _plan(tmp_path / "plan.json", _row())
    monkeypatch.setattr(
        generator, "_http_transport", lambda *_args: _openai_response(_png())
    )
    generator.execute(plan, tmp_path / "real-output")
    real_receipt = json.loads(
        (
            tmp_path
            / "real-output"
            / "W6-DESIGN-GPT-A"
            / "row-001.success.json"
        ).read_text()
    )
    assert real_receipt["origin"] == "real"

    generator.execute(
        plan,
        tmp_path / "synthetic-output",
        transport=lambda *_args: _openai_response(_png()),
    )
    synthetic_receipt = json.loads(
        (
            tmp_path
            / "synthetic-output"
            / "W6-DESIGN-GPT-A"
            / "row-001.success.json"
        ).read_text()
    )
    assert synthetic_receipt["origin"] == "synthetic"
    with pytest.raises(generator.GenerationError, match="does not match this row plan"):
        generator.execute(plan, tmp_path / "synthetic-output")


def test_transient_provider_failure_is_immutable_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "sk-test-retry-secret"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.setattr(generator.time, "sleep", lambda _seconds: None)
    plan = _plan(tmp_path / "plan.json", _row())
    responses = [
        generator.HttpResponse(429, {"x-request-id": "req_rate"}, b'{"error":{}}'),
        _openai_response(_png()),
    ]

    def transport(*_args):
        return responses.pop(0)

    generator.execute(
        plan, tmp_path / "output", transport=transport, max_attempts=2
    )
    attempts = tmp_path / "output" / "W6-DESIGN-GPT-A" / "attempts"
    failed = json.loads((attempts / "row-001.attempt-001.json").read_text())
    passed = json.loads((attempts / "row-001.attempt-002.json").read_text())
    assert failed["state"] == "FAIL" and failed["http_status"] == 429
    assert failed["origin"] == "synthetic"
    assert failed["provider_request_id"] == "req_rate"
    assert passed["state"] == "SUCCESS"
    assert key not in json.dumps(failed)


def test_resume_fails_closed_if_output_bytes_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    plan = _plan(tmp_path / "plan.json", _row())
    output = tmp_path / "output"
    generator.execute(plan, output, transport=lambda *_args: _openai_response(_png()))
    image = output / "W6-DESIGN-GPT-A" / "images" / "row-001.png"
    image.write_bytes(b"tampered")
    with pytest.raises(generator.GenerationError, match="hash mismatch"):
        generator.execute(plan, output, transport=lambda *_args: pytest.fail())


def test_resume_is_bound_to_exact_plan_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    row = _row()
    plan = _plan(tmp_path / "plan.json", row)
    output = tmp_path / "output"
    generator.execute(plan, output, transport=lambda *_args: _openai_response(_png()))
    changed = json.loads(plan.read_text())
    changed["fixtures"].append(
        {"id": "W6-DESIGN-GPT-LARGE", "rows": [_row(row_id="row-002")]}
    )
    plan.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(generator.GenerationError, match="does not match this row plan"):
        generator.execute(plan, output, transport=lambda *_args: pytest.fail())


def test_failed_exclusive_publish_never_deletes_preexisting_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    plan = _plan(tmp_path / "plan.json", _row())
    fixture = tmp_path / "output" / "W6-DESIGN-GPT-A"
    (fixture / "images").mkdir(parents=True)
    preexisting = fixture / "images" / "row-001.png"
    preexisting.write_bytes(b"owner-data")
    with pytest.raises(generator.GenerationError, match="generation failed"):
        generator.execute(
            plan,
            tmp_path / "output",
            transport=lambda *_args: _openai_response(_png()),
            max_attempts=1,
        )
    assert preexisting.read_bytes() == b"owner-data"


def test_row_lock_serializes_concurrent_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    row = _row()
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    calls = 0

    def transport(*_args):
        nonlocal calls
        calls += 1
        return _openai_response(_png())

    def run():
        return generator._run_row(
            row,
            fixture_id="W6-DESIGN-GPT-A",
            plan_file_sha="f" * 64,
            fixture_dir=fixture,
            transport=transport,
            timeout_s=10.0,
            max_attempts=1,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _index: run(), range(2)))
    assert first == second
    assert calls == 1


def test_plan_rejects_unpinned_models_and_incomplete_materialization(
    tmp_path: Path,
) -> None:
    bad_model = _row()
    bad_model["model"] = "gpt-image-2"
    path = _plan(tmp_path / "bad-model.json", bad_model)
    with pytest.raises(generator.GenerationError, match="pinned GPT Image 2"):
        generator.load_plan(path)

    materialization = {
        "schema": 1,
        "kind": "sn56-week6-lane-b-generation-plan",
        "purpose": "materialization",
        "social_relief": False,
        "fixtures": [{"id": "W6-DESIGN-GPT-A", "rows": [_row()]}],
    }
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps(materialization), encoding="utf-8")
    with pytest.raises(generator.GenerationError, match="wrong materialization count"):
        generator.load_plan(incomplete)


def test_plan_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    plan = tmp_path / "duplicate.json"
    plan.write_text(
        '{"schema":1,"schema":1,"kind":"sn56-week6-lane-b-generation-plan",'
        '"purpose":"smoke","social_relief":false,"fixtures":[]}',
        encoding="utf-8",
    )
    with pytest.raises(generator.GenerationError, match="duplicate key"):
        generator.load_plan(plan)


@pytest.mark.parametrize("field", ["id", "prompt", "caption"])
def test_non_smoke_plan_must_match_frozen_blueprint(
    tmp_path: Path, field: str
) -> None:
    reference_a = tmp_path / "reference-a.png"
    reference_b = tmp_path / "reference-b.png"
    reference_a.write_bytes(_png((11, 22, 33)))
    reference_b.write_bytes(_png((44, 55, 66)))
    plan = planner.build_plan(
        "materialization",
        product_a_reference=reference_a.absolute(),
        product_b_reference=reference_b.absolute(),
    )
    plan["fixtures"][0]["rows"][0][field] = {
        "id": "arbitrary-safe-id",
        "prompt": "Arbitrary but structurally valid replacement prompt.",
        "caption": "Arbitrary but structurally valid replacement caption.",
    }[field]
    path = tmp_path / f"forged-{field}.json"
    path.write_bytes(planner.canonical_bytes(plan) + b"\n")
    with pytest.raises(generator.GenerationError, match="frozen blueprint"):
        generator.load_plan(path)


def test_provider_output_must_match_requested_aspect_ratio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    plan = _plan(tmp_path / "plan.json", _row())
    portrait = BytesIO()
    Image.new("RGB", (24, 32), (1, 2, 3)).save(portrait, format="PNG")
    with pytest.raises(generator.GenerationError, match="generation failed") as caught:
        generator.execute(
            plan,
            tmp_path / "output",
            transport=lambda *_args: _openai_response(portrait.getvalue()),
            max_attempts=1,
        )
    assert isinstance(caught.value.__cause__, generator.GenerationError)
    assert "ratio differs" in str(caught.value.__cause__)
    fixture = tmp_path / "output" / "W6-DESIGN-GPT-A"
    assert not (fixture / "images" / "row-001.png").exists()
    assert not (fixture / "captions" / "row-001.txt").exists()


def test_exclusive_publication_fsyncs_file_and_parent(tmp_path: Path, monkeypatch) -> None:
    observed: list[str] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        observed.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(descriptor)

    monkeypatch.setattr(generator.os, "fsync", recording_fsync)
    generator._publish_exclusive(tmp_path / "evidence.bin", b"bound-bytes")
    assert observed == ["file", "directory"]


def test_new_directory_entry_fsyncs_parent_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed_directories: list[tuple[int, int]] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if stat.S_ISDIR(metadata.st_mode):
            observed_directories.append((metadata.st_dev, metadata.st_ino))
        real_fsync(descriptor)

    monkeypatch.setattr(generator.os, "fsync", recording_fsync)
    parent_identity = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    created = tmp_path / "new-evidence-directory"
    generator._ensure_directory(created)
    assert observed_directories == [parent_identity]

    observed_directories.clear()
    generator._ensure_directory(created)
    assert observed_directories == []


def test_success_link_fsyncs_source_and_target_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    plan = _plan(tmp_path / "plan.json", _row())
    output = tmp_path / "output"
    observed_directories: set[tuple[int, int]] = set()
    real_fsync = os.fsync
    real_link = os.link
    link_created = False

    def recording_link(source: Path, target: Path) -> None:
        nonlocal link_created
        real_link(source, target)
        link_created = True

    def recording_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if link_created and stat.S_ISDIR(metadata.st_mode):
            observed_directories.add((metadata.st_dev, metadata.st_ino))
        real_fsync(descriptor)

    monkeypatch.setattr(generator.os, "link", recording_link)
    monkeypatch.setattr(generator.os, "fsync", recording_fsync)
    generator.execute(plan, output, transport=lambda *_args: _openai_response(_png()))
    fixture = output / "W6-DESIGN-GPT-A"
    attempts = fixture / "attempts"
    expected = {
        (fixture.stat().st_dev, fixture.stat().st_ino),
        (attempts.stat().st_dev, attempts.stat().st_ino),
    }
    assert expected <= observed_directories


def test_summary_publication_race_accepts_only_exact_existing_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    plan = _plan(tmp_path / "plan.json", _row())
    output = tmp_path / "output"
    original = generator._publish_json
    raced = False

    def publish_with_exact_race(path: Path, value: dict) -> None:
        nonlocal raced
        if path.name == "generation-summary.json" and not raced:
            raced = True
            original(path, value)
            raise FileExistsError(path)
        original(path, value)

    monkeypatch.setattr(generator, "_publish_json", publish_with_exact_race)
    summary = generator.execute(
        plan, output, transport=lambda *_args: _openai_response(_png())
    )
    assert raced and json.loads((output / "generation-summary.json").read_bytes()) == summary

    different_output = tmp_path / "different-output"
    different_output.mkdir()
    (different_output / "generation-summary.json").write_text("{}\n")
    with pytest.raises(generator.GenerationError, match="summary differs"):
        generator.execute(
            plan,
            different_output,
            transport=lambda *_args: _openai_response(_png()),
        )
