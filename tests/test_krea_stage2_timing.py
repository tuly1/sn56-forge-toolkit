from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import zipfile

import pytest

from ops.calibration import krea_budget
from ops.calibration import krea_confirmation_admission as admission
from ops.calibration import krea_fixture
from ops.calibration import krea_provenance
from ops.calibration import krea_stage2_delegated_review_contract as review_contract
from ops.calibration import krea_stage2_production_identity as production
from ops.calibration import krea_stage2_timing as timing
from ops.calibration import krea_stage2_timing_collector as collector


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _file_sha(value: dict) -> str:
    return hashlib.sha256(krea_provenance.canonical_bytes(value) + b"\n").hexdigest()


def _write_canonical(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(krea_provenance.canonical_bytes(value) + b"\n")


def _list_supported_images(path: str, extensions: tuple[str, ...]) -> list[str]:
    return sorted(
        item.name
        for item in Path(path).iterdir()
        if item.is_file() and item.name.lower().endswith(extensions)
    )


def _group(role: str, label: str) -> dict[str, str]:
    values = {
        "source_id": f"source-{label}",
        "creator_id": f"creator-{label}",
        "burst_id": f"burst-{label}",
        "scene_id": f"scene-{label}",
        "play_root_id": f"play-{label}",
        "human_similarity_cluster_id": f"human-{label}",
    }
    if role == "D2":
        values.update(
            play_component_id=f"component-{label}",
            accession_family_id=f"accession-{label}",
        )
    return values


def _fixture(tmp_path: Path, role: str, train_count: int, eval_count: int) -> dict:
    from PIL import Image

    root = tmp_path / f"fixture-{role}"
    training = root / "training"
    evaluation = root / "evaluation"
    training.mkdir(parents=True)
    evaluation.mkdir()
    row_groups: dict[str, dict[str, dict[str, str]]] = {
        "training": {},
        "evaluation": {},
    }
    for split, directory, count, offset in (
        ("training", training, train_count, 0),
        ("evaluation", evaluation, eval_count, 10_000),
    ):
        for index in range(count):
            name = f"{split}-{index:03d}.png"
            rng = random.Random(offset + index + sum(map(ord, role)))
            pixels = bytes(rng.randrange(256) for _ in range(32 * 32 * 3))
            Image.frombytes("RGB", (32, 32), pixels).save(directory / name)
            (directory / name.replace(".png", ".txt")).write_text(
                f"synthetic calibration caption {role} {split} {index}\n",
                encoding="utf-8",
            )
            row_groups[split][name] = _group(role, f"{role}-{split}-{index}")
    archive = root / "training.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(training.iterdir()):
            bundle.write(path, path.name)

    _, training_rows = krea_fixture._rows(
        training,
        role=role,
        list_supported_images=_list_supported_images,
        extensions=(".png",),
        row_groups=row_groups["training"],
    )
    _, evaluation_rows = krea_fixture._rows(
        evaluation,
        role=role,
        list_supported_images=_list_supported_images,
        extensions=(".png",),
        row_groups=row_groups["evaluation"],
    )
    rights = {
        "schema": 1,
        "kind": "forge-krea-source-rights-review",
        "owner": "Synthetic Test Owner",
        "locator": "https://example.invalid/synthetic",
        "reviewer_identity": "Jordan Rights",
        "reviewed_at_utc": "2026-07-29T00:00:01Z",
        "decision": "approved_for_calibration",
        "assertions": {
            "lawful_access": True,
            "calibration_use_allowed": True,
            "redistribution_reviewed": True,
            "sensitive_content_absent": True,
        },
    }
    captions = {
        "schema": 1,
        "kind": "forge-krea-caption-review",
        "concept_id": f"fixture-{role.lower()}",
        "trigger_token": "SN56",
        "reviewer_identity": "Jordan Caption",
        "reviewed_at_utc": "2026-07-29T00:00:02Z",
        "decision": "approved",
        "training_row_ids": sorted(row["row_id"] for row in training_rows),
        "evaluation_row_ids": sorted(row["row_id"] for row in evaluation_rows),
        "assertions": {
            "manual_review_complete": True,
            "captions_match_images": True,
            "trigger_usage_consistent": True,
            "evaluation_leakage_absent": True,
        },
    }
    all_row_ids = [row["row_id"] for row in training_rows + evaluation_rows]
    similarity = {
        "schema": 1,
        "kind": "forge-krea-human-similarity-review",
        "concept_id": f"fixture-{role.lower()}",
        "experimental_role": role,
        "reviewer_identity": "Jordan Similarity",
        "reviewed_at_utc": "2026-07-29T00:00:03Z",
        "decision": "passed",
        "reviewed_pairs": krea_fixture._reviewed_pairs(all_row_ids),
        "flagged_pairs": [],
    }
    rights_path = root / "rights.json"
    caption_path = root / "captions.json"
    similarity_path = root / "similarity.json"
    _write_canonical(rights_path, rights)
    _write_canonical(caption_path, captions)
    _write_canonical(similarity_path, similarity)
    manifest = krea_fixture.build_manifest(
        {
            "concept_id": f"fixture-{role.lower()}",
            "experimental_role": role,
            "trigger_token": "SN56",
            "source_owner": rights["owner"],
            "source_locator": rights["locator"],
            "source_retrieved_at_utc": "2026-07-29T00:00:00Z",
            "preparer_identity": "Jordan Preparer",
            "god_commit": "a" * 40,
            "near_duplicate_hamming_threshold": 0,
            "group_disjoint_fields": sorted(
                krea_fixture._BASE_GROUP_DISJOINT_FIELDS
                | (krea_fixture._D2_GROUP_FIELDS if role == "D2" else set())
            ),
            "training_row_groups": row_groups["training"],
            "evaluation_row_groups": row_groups["evaluation"],
            "similarity_reviewer_identity": similarity["reviewer_identity"],
        },
        training_dir=training,
        evaluation_dir=evaluation,
        training_archive=archive,
        rights_record=rights_path,
        caption_policy=caption_path,
        similarity_review_record=similarity_path,
        list_supported_images=_list_supported_images,
        extensions=(".png",),
    )
    assert krea_fixture.validate_manifest(manifest) == manifest
    return manifest


def _assets_and_identity(tmp_path: Path) -> tuple[dict, dict]:
    base = tmp_path / "base-model"
    text = tmp_path / "text-encoder"
    base.mkdir()
    text.mkdir()
    (base / "weights.bin").write_bytes(b"base model test bytes\n")
    (text / "weights.bin").write_bytes(b"text encoder test bytes\n")
    staging_path = tmp_path / "staging.json"
    staging = {
        "assets": [
            {
                "repo_id": production.KREA_MODEL_ID,
                "revision": "e" * 40,
                "local_dir": production._ASSET_DESTINATIONS["base_model"],
                "resolved_path": production._ASSET_DESTINATIONS["base_model"],
            },
            {
                "repo_id": production.KREA_TEXT_ENCODER_ID,
                "revision": "f" * 40,
                "local_dir": production._ASSET_DESTINATIONS["text_encoder"],
                "resolved_path": production._ASSET_DESTINATIONS["text_encoder"],
            },
        ]
    }
    _write_canonical(staging_path, staging)
    assets = production.capture_asset_attestation(
        base_model_path=base,
        text_encoder_path=text,
        staging_manifest_path=staging_path,
        captured_at_utc="2026-07-30T00:00:00Z",
    )
    identity = production.build(
        forge={
            "commit_sha1": "1" * 40,
            "tree_sha1": "2" * 40,
            "worktree_state": "clean-including-untracked",
        },
        container_image={
            "image_id": "sha256:" + "3" * 64,
            "repo_digest": "registry.example/forge@sha256:" + "4" * 64,
        },
        dockerfile={
            "path": production.DOCKERFILE_PATH,
            "sha256": _sha("Dockerfile"),
            "bytes": 123,
        },
        runtime_inputs=[
            {"path": path, "sha256": _sha(path), "bytes": index + 1}
            for index, path in enumerate(production.RUNTIME_INPUT_PATHS)
        ],
        base_model={
            "model_id": production.KREA_MODEL_ID,
            "revision": "e" * 40,
            "training_identity_sha256": assets["training_identity_sha256"],
            "asset_attestation_sha256": assets["attestation_sha256"],
            "text_encoder_id": production.KREA_TEXT_ENCODER_ID,
            "text_encoder_revision": "f" * 40,
        },
        runtime_contract={
            "runtime_identity_sha256": _sha("runtime"),
            "venv_tree_manifest_sha256": _sha("venv"),
            "trainer_identity_sha256": _sha("trainer"),
            "measurement_tool_sha256": _sha("measurement-tool"),
            "jit_enabled": True,
        },
        captured_at_utc="2026-07-30T00:00:10Z",
    )
    return assets, identity


def _host() -> dict:
    body = {
        "schema": 1,
        "kind": "forge-krea-stage2-live-host-identity",
        "machine_id_sha256": _sha("machine"),
        "boot_id_sha256": _sha("boot"),
        "kernel_release": "6.8.0-test",
        "machine": "x86_64",
        "cpu_affinity_ids": [0, 1],
        "memory_total_bytes": 128 * 1024**3,
        "checkpoint_device": {"st_dev": 42, "major": 0, "minor": 42},
    }
    return {
        **body,
        "host_execution_identity_sha256": krea_provenance.canonical_sha256(body),
    }


def _gpu(uuid: str = "GPU-test-a") -> dict:
    body = {
        "schema": 1,
        "kind": "forge-krea-stage2-live-gpu-identity",
        "uuid": uuid,
        "name": "NVIDIA H100 PCIe",
        "driver_version": "570.00",
        "memory_total_mib": "81559",
        "compute_capability": "9.0",
        "pci_bus_id": "00000000:01:00.0",
    }
    return {**body, "gpu_identity_sha256": krea_provenance.canonical_sha256(body)}


def _margin() -> dict:
    return krea_budget.seal_margin_policy(
        reviewer_identity="Jordan Margin",
        approved_at_utc="2026-07-30T00:01:00Z",
        frozen_before_capture=True,
        multiplicative_margin={metric: 1.25 for metric in timing._TIMING_METRICS},
        additive_margin_s={metric: 0.5 for metric in timing._TIMING_METRICS},
    )


def _admission_chain(
    tmp_path: Path,
    *,
    identity: dict,
    fixtures: dict[str, dict],
) -> dict:
    sealed_root = tmp_path / "sealed"
    rows = []
    role_file_sha: dict[str, str] = {}
    for role in sorted(admission._ALL_ROLES):
        relative = f"{role}/manifest.json"
        path = sealed_root / relative
        if role in fixtures:
            raw = krea_provenance.canonical_bytes(fixtures[role]) + b"\n"
        else:
            raw = krea_provenance.canonical_bytes({"role": role}) + b"\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        role_file_sha[role] = digest
        rows.append(
            {
                "role": role,
                "relative_path": relative,
                "sha256": digest,
                "bytes": len(raw),
            }
        )
    identity_file_sha = _file_sha(identity)
    request = admission.build_request(
        production_identity=identity,
        production_identity_file_sha256=identity_file_sha,
        waiver_freeze_sha256=_sha("waiver-semantic"),
        waiver_freeze_file_sha256=_sha("waiver-file"),
        public_commitment_sha256s={
            role: role_file_sha[role] for role in sorted(admission._CONFIRMATION_ROLES)
        },
        boundary_fixture_manifest_sha256s={
            role: (
                fixtures[role]["manifest_sha256"]
                if role in fixtures
                else _sha(f"boundary-{role}")
            )
            for role in sorted(admission._BOUNDARY_ROLES)
        },
        sealed_inventory_sha256=_sha("sealed-inventory"),
        sealed_inventory_file_sha256=_sha("sealed-inventory-file"),
        sealed_root_locator_sha256=admission.sealed_root_locator_sha256(sealed_root),
        sealed_files=rows,
        prepared_at_utc="2026-07-30T00:02:00Z",
    )
    request_file_sha = admission.canonical_file_sha256(request)
    ratification = admission.ratify(
        request,
        production_identity=identity,
        production_identity_file_sha256=identity_file_sha,
        sealed_root=sealed_root,
        owner_identity=admission.OWNER_IDENTITY,
        ratified_at_utc="2026-07-30T00:03:00Z",
    )
    ratification_file_sha = admission.canonical_file_sha256(ratification)
    reveal = admission.authorize_reveal(
        request,
        ratification,
        ratification_file_sha256=ratification_file_sha,
        production_identity=identity,
        production_identity_file_sha256=identity_file_sha,
        sealed_root=sealed_root,
        actor=review_contract.actor("confirmation_reveal_reviewer"),
        revealed_at_utc="2026-07-30T00:04:00Z",
    )
    reveal_file_sha = admission.canonical_file_sha256(reveal)
    materialization = admission.materialize(
        request,
        ratification,
        reveal,
        request_file_sha256=request_file_sha,
        ratification_file_sha256=ratification_file_sha,
        reveal_file_sha256=reveal_file_sha,
        production_identity=identity,
        production_identity_file_sha256=identity_file_sha,
        sealed_root=sealed_root,
        output_dir=tmp_path / "materialized",
        actor=review_contract.actor("confirmation_materialization_reviewer"),
        materialized_at_utc="2026-07-30T00:05:00Z",
    )
    materialization_file_sha = admission.canonical_file_sha256(materialization)
    authorization = admission.build_gpu_execution_authorization(
        request,
        ratification,
        reveal,
        materialization,
        request_file_sha256=request_file_sha,
        ratification_file_sha256=ratification_file_sha,
        reveal_file_sha256=reveal_file_sha,
        materialization_file_sha256=materialization_file_sha,
        production_identity=identity,
        production_identity_file_sha256=identity_file_sha,
        owner_identity=admission.OWNER_IDENTITY,
        authorized_at_utc="2026-07-30T00:06:00Z",
    )
    return {
        "request": request,
        "request_file_sha256": request_file_sha,
        "ratification": ratification,
        "ratification_file_sha256": ratification_file_sha,
        "reveal": reveal,
        "reveal_file_sha256": reveal_file_sha,
        "materialization": materialization,
        "materialization_file_sha256": materialization_file_sha,
        "gpu_execution_authorization": authorization,
        "gpu_execution_authorization_file_sha256": admission.canonical_file_sha256(
            authorization
        ),
    }


def _controls(
    *,
    fixture: dict,
    assets: dict,
    identity: dict,
    authority: dict,
) -> dict:
    probe = timing.seal_probe_contract(
        created_at_utc="2026-07-30T00:00:30Z",
        production_image_id=identity["container_image"]["image_id"],
        measurement_tool_sha256=identity["runtime_contract"]["measurement_tool_sha256"],
        collector_executable_sha256=_sha("separate-receipt-collector"),
        executable_sha256=_sha("docker-cli"),
        gpu_device=0,
        fixture_role=fixture["experimental_role"],
        fixture_manifest_sha256=fixture["manifest_sha256"],
        training_archive_sha256=fixture["training_archive"]["sha256"],
        training_archive_bytes=fixture["training_archive"]["bytes"],
        profile_id="K1",
        hard_budget_s=2700.0,
        mount_sources={
            "base_model": "/test/base-model",
            "text_encoder": "/test/text-encoder",
            "dataset_cache": "/test/datasets",
            "checkpoints": "/test/checkpoints",
            "run_evidence": "/test/evidence",
        },
        trigger_word=(
            None
            if fixture["experimental_role"] in timing._CONFIRMATION_ROLES
            else fixture["trigger_token"]
        ),
    )
    host = _host()
    gpu = _gpu()
    margin = _margin()
    return {
        "fixture_manifest": fixture,
        "fixture_manifest_file_sha256": _file_sha(fixture),
        "fixture_manifest_file_bytes": len(
            krea_provenance.canonical_bytes(fixture) + b"\n"
        ),
        "production_identity": identity,
        "production_identity_file_sha256": _file_sha(identity),
        "asset_attestation": assets,
        "asset_attestation_file_sha256": _file_sha(assets),
        "probe_contract": probe,
        "probe_contract_file_sha256": _file_sha(probe),
        "live_host_identity": host,
        "live_host_identity_file_sha256": _file_sha(host),
        "live_gpu_identity": gpu,
        "live_gpu_identity_file_sha256": _file_sha(gpu),
        "margin_policy": margin,
        "margin_policy_file_sha256": _file_sha(margin),
        "content_authority_controls": authority,
    }


def _events(index: int, canary: str) -> list[dict]:
    rows = []
    clock = 1_000_000_000_000 + index * 100_000_000_000
    counters = {
        "startup": 1,
        "optimizer_update": 34,
    }
    for metric, units in counters.items():
        token = f"span-{index}-{metric}-{canary}"
        rows.append(
            timing.seal_event(
                sequence=len(rows),
                span_token=token,
                metric=metric,
                state="begin",
                counter_value=0,
                received_monotonic_ns=clock,
            )
        )
        clock += 1_000_000_000
        rows.append(
            timing.seal_event(
                sequence=len(rows),
                span_token=token,
                metric=metric,
                state="end",
                counter_value=units,
                received_monotonic_ns=clock,
            )
        )
        clock += 1_000_000_000
    for save_index in range(3):
        token = f"span-{index}-checkpoint_save-{save_index}-{canary}"
        rows.append(
            timing.seal_event(
                sequence=len(rows),
                span_token=token,
                metric="checkpoint_save",
                state="begin",
                counter_value=0,
                received_monotonic_ns=clock,
            )
        )
        clock += 1_000_000_000
        rows.append(
            timing.seal_event(
                sequence=len(rows),
                span_token=token,
                metric="checkpoint_save",
                state="end",
                counter_value=1,
                received_monotonic_ns=clock,
            )
        )
        clock += 1_000_000_000
    for metric in ("finalization", "upload"):
        token = f"span-{index}-{metric}-{canary}"
        rows.append(
            timing.seal_event(
                sequence=len(rows),
                span_token=token,
                metric=metric,
                state="begin",
                counter_value=0,
                received_monotonic_ns=clock,
            )
        )
        clock += 1_000_000_000
        rows.append(
            timing.seal_event(
                sequence=len(rows),
                span_token=token,
                metric=metric,
                state="end",
                counter_value=1,
                received_monotonic_ns=clock,
            )
        )
        clock += 1_000_000_000
    return rows


def _receipt(plan: dict, controls: dict, index: int, *, heldout: bool = False) -> dict:
    role = "held_out_end_to_end" if heldout else "timing_measurement"
    monotonic_start = 999_000_000_000 + index * 100_000_000_000
    monotonic_end = monotonic_start + 60_000_000_000
    unix_start = 1_785_372_000_000_000_000 + index * 100_000_000_000
    unix_end = unix_start + 60_000_000_000
    run = timing.seal_run_receipt(
        measurement_role=role,
        artifact_manifest_file_sha256=_sha(f"artifact-file-{index}"),
        artifact_manifest_sha256=_sha(f"artifact-{index}"),
        recorded_unix_ns=unix_end + 1_000_000_000,
    )
    raw = timing.seal_raw_receipt(
        plan,
        probe_contract=controls["probe_contract"],
        receipt_ordinal=3 if heldout else index,
        command={
            "argv": timing.render_probe_command(
                controls["probe_contract"],
                timing_plan_sha256=plan["plan_sha256"],
                receipt_ordinal=3 if heldout else index,
            ),
            "executable_id": timing._EXECUTABLE_ID,
            "executable_path": timing._EXECUTABLE_PATH,
            "executable_sha256": controls["probe_contract"]["executable_sha256"],
            "returncode": 0,
            "started_unix_ns": unix_start,
            "ended_unix_ns": unix_end,
            "started_monotonic_ns": monotonic_start,
            "ended_monotonic_ns": monotonic_end,
            "production_image_id": plan["production_image_id"],
            "network_mode": "none",
            "runtime": "nvidia",
        },
        events=_events(index, "PRIVATE-CANARY"),
        run_receipt=run,
    )
    return {
        "record": raw,
        "file_sha256": _file_sha(raw),
        "receipt_sha256": raw["receipt_sha256"],
    }


def _receipt_manifest(receipts: list[dict], controls: dict) -> dict:
    collector = timing.seal_collector_identity(
        created_at_utc="2026-07-30T00:08:00Z",
        collector_executable_sha256=_sha("separate-receipt-collector"),
        measurement_tool_sha256=controls["probe_contract"]["measurement_tool_sha256"],
    )
    return timing.seal_receipt_manifest(
        created_at_utc="2026-07-30T01:00:00Z",
        collector_identity=collector,
        collector_identity_file_sha256=_file_sha(collector),
        receipt_bindings=receipts,
    )


def _manifest_args(manifest: dict) -> dict:
    return {
        "receipt_manifest": manifest,
        "expected_receipt_manifest_file_sha256": _file_sha(manifest),
        "expected_receipt_manifest_sha256": manifest["receipt_manifest_sha256"],
    }


def _rehash_receipt(binding: dict) -> None:
    binding["record"]["event_stream_sha256"] = krea_provenance.canonical_sha256(
        binding["record"]["events"]
    )
    body = {
        key: value
        for key, value in binding["record"].items()
        if key != "receipt_sha256"
    }
    binding["record"]["receipt_sha256"] = krea_provenance.canonical_sha256(body)
    binding["file_sha256"] = _file_sha(binding["record"])
    binding["receipt_sha256"] = binding["record"]["receipt_sha256"]


@pytest.fixture(scope="module")
def real_surface(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("stage2-timing-real")
    c1 = _fixture(root, "C1", 20, 6)
    boundary = _fixture(root, "B-0p5-small", 18, 24)
    assets, identity = _assets_and_identity(root)
    authority = _admission_chain(
        root, identity=identity, fixtures={"C1": c1, "B-0p5-small": boundary}
    )
    return {
        "root": root,
        "C1": c1,
        "B-0p5-small": boundary,
        "assets": assets,
        "identity": identity,
        "authority": authority,
    }


def _plan_and_receipts(
    real_surface: dict, role: str = "C1"
) -> tuple[dict, dict, list[dict], dict]:
    controls = _controls(
        fixture=real_surface[role],
        assets=real_surface["assets"],
        identity=real_surface["identity"],
        authority=real_surface["authority"],
    )
    plan = timing.build_plan(
        controls=controls,
        profile_id="K1",
        hard_budget_s=2700.0,
        created_at_utc="2026-07-30T00:07:00Z",
    )
    receipts = [_receipt(plan, controls, index) for index in range(3)]
    receipts.append(_receipt(plan, controls, 10, heldout=True))
    return plan, controls, receipts, _receipt_manifest(receipts, controls)


def _restore_writable(root: Path) -> None:
    if root.exists():
        os.chmod(root, 0o700)


def test_real_c1_and_boundary_chains_validate_without_monkeypatch(
    real_surface: dict,
) -> None:
    c1_plan, _, _, _ = _plan_and_receipts(real_surface, "C1")
    boundary_plan, _, _, _ = _plan_and_receipts(real_surface, "B-0p5-small")

    assert c1_plan["sealed_content_authority"]["fixture_commitment"]["mode"] == (
        "confirmation_manifest_file_sha256"
    )
    assert boundary_plan["sealed_content_authority"]["fixture_commitment"]["mode"] == (
        "boundary_manifest_semantic_sha256"
    )


def test_probe_preserves_legacy_null_trigger_and_boundary_trigger() -> None:
    common = {
        "created_at_utc": "2026-07-30T00:00:30Z",
        "production_image_id": "sha256:" + "3" * 64,
        "measurement_tool_sha256": _sha("measurement-tool"),
        "collector_executable_sha256": _sha("collector"),
        "executable_sha256": _sha("docker"),
        "gpu_device": 0,
        "fixture_manifest_sha256": _sha("fixture"),
        "training_archive_sha256": _sha("archive"),
        "training_archive_bytes": 123,
        "profile_id": "K1",
        "hard_budget_s": 2700.0,
        "mount_sources": {
            "base_model": "/test/base-model",
            "text_encoder": "/test/text-encoder",
            "dataset_cache": "/test/datasets",
            "checkpoints": "/test/checkpoints",
            "run_evidence": "/test/evidence",
        },
    }
    confirmation = timing.seal_probe_contract(
        **common,
        fixture_role="C1",
        trigger_word=None,
    )
    assert confirmation["command_fields"]["trigger_word"] is None
    assert "--trigger-word" not in confirmation["command_argv_template"]
    assert timing.validate_probe_contract(confirmation) == confirmation

    boundary = timing.seal_probe_contract(
        **common,
        fixture_role="B-0p75-small",
        trigger_word="SN56",
    )
    trigger_index = boundary["command_argv_template"].index("--trigger-word")
    assert boundary["command_argv_template"][trigger_index + 1] == "SN56"
    with pytest.raises(ValueError, match="trigger word"):
        timing.seal_probe_contract(
            **common,
            fixture_role="B-0p75-small",
            trigger_word=None,
        )


def test_canonical_fixture_schema_never_accepts_null_trigger(real_surface: dict) -> None:
    manifest = deepcopy(real_surface["B-0p5-small"])
    manifest["trigger_token"] = None
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = krea_provenance.canonical_sha256(body)
    with pytest.raises(ValueError, match="trigger_token"):
        krea_fixture.validate_manifest(manifest)


def test_receipt_derived_bundle_replays_and_scrubs_sealed_canaries(
    real_surface: dict, tmp_path: Path
) -> None:
    plan, controls, receipts, manifest = _plan_and_receipts(real_surface)
    root = tmp_path / "bundle"
    try:
        bundle = timing.produce_bundle(
            plan=plan,
            controls=controls,
            **_manifest_args(manifest),
            receipt_bindings=receipts,
            framework_stop_boundary_s=225.0,
            framework_stop_boundary_source_sha256=_sha("stop-boundary"),
            output_root=root,
        )
        binding = timing.bundle_binding(root)
        replay = timing.replay_bundle(
            root,
            expected_bundle_file_sha256=binding["bundle_file_sha256"],
            expected_bundle_sha256=binding["bundle_sha256"],
            controls=controls,
            **_manifest_args(manifest),
            receipt_bindings=receipts,
        )
        persisted = b"".join(path.read_bytes() for path in root.iterdir())
        assert b"COMMAND-SECRET-CANARY" not in persisted
        assert b"PRIVATE-CANARY" not in persisted
        assert root.stat().st_mode & 0o222 == 0
        assert all(path.stat().st_mode & 0o222 == 0 for path in root.iterdir())
        assert bundle["artifact_schema_sha256"] == krea_provenance.canonical_sha256(
            timing._artifact_schema(3, 1)
        )
        assert replay["throughput_profile"]["startup_sample_count"] == 3
        assert replay["throughput_profile"]["update_sample_count"] == 102
        assert replay["throughput_profile"]["save_sample_count"] == 9
        assert [
            json.loads((root / f"measurement-{index:03d}.json").read_text())[
                "receipt_ordinal"
            ]
            for index in range(1, 4)
        ] == [0, 1, 2]
    finally:
        _restore_writable(root)


def test_capture_and_bundle_schemas_are_exact(
    real_surface: dict, tmp_path: Path
) -> None:
    plan, controls, receipts, manifest = _plan_and_receipts(real_surface)
    capture = timing._derive_capture(
        plan, controls=controls, receipt_binding=receipts[0]
    )
    capture["caller_extension"] = True
    with pytest.raises(ValueError, match="keys differ"):
        timing._validate_capture_record(capture, plan=plan)

    root = tmp_path / "exact-schema"
    try:
        bundle = timing.produce_bundle(
            plan=plan,
            controls=controls,
            **_manifest_args(manifest),
            receipt_bindings=receipts,
            framework_stop_boundary_s=225.0,
            framework_stop_boundary_source_sha256=_sha("stop-boundary"),
            output_root=root,
        )
        bundle["caller_extension"] = True
        with pytest.raises(ValueError, match="keys differ"):
            timing.validate_bundle(bundle, root=root)
    finally:
        _restore_writable(root)


def test_valid_but_different_stored_margin_cannot_rebind_bundle(
    real_surface: dict, tmp_path: Path
) -> None:
    plan, controls, receipts, manifest = _plan_and_receipts(real_surface)
    root = tmp_path / "margin-rebind"
    try:
        bundle = timing.produce_bundle(
            plan=plan,
            controls=controls,
            **_manifest_args(manifest),
            receipt_bindings=receipts,
            framework_stop_boundary_s=225.0,
            framework_stop_boundary_source_sha256=_sha("stop-boundary"),
            output_root=root,
        )
        replacement = krea_budget.seal_margin_policy(
            reviewer_identity="Jordan Margin",
            approved_at_utc="2026-07-30T00:01:00Z",
            frozen_before_capture=True,
            multiplicative_margin={metric: 1.5 for metric in timing._TIMING_METRICS},
            additive_margin_s={metric: 0.5 for metric in timing._TIMING_METRICS},
        )
        os.chmod(root, 0o700)
        margin_path = root / "margin-policy.json"
        os.chmod(margin_path, 0o600)
        _write_canonical(margin_path, replacement)
        os.chmod(margin_path, 0o400)
        row = next(
            item for item in bundle["artifacts"] if item["path"] == margin_path.name
        )
        row.update(
            bytes=margin_path.stat().st_size,
            file_sha256=krea_provenance.file_sha256(margin_path),
            semantic_sha256=replacement["margin_policy_sha256"],
        )
        body = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
        bundle["bundle_sha256"] = krea_provenance.canonical_sha256(body)
        bundle_path = root / "bundle.json"
        os.chmod(bundle_path, 0o600)
        _write_canonical(bundle_path, bundle)
        os.chmod(bundle_path, 0o400)
        os.chmod(root, 0o500)
        with pytest.raises(ValueError, match="stored timing margin differs"):
            timing.validate_bundle(bundle, root=root)
    finally:
        _restore_writable(root)


@pytest.mark.parametrize("stale_field", ["duration_s", "units"])
def test_caller_cannot_inject_derived_duration_or_units(
    real_surface: dict, stale_field: str
) -> None:
    plan, controls, receipts, manifest = _plan_and_receipts(real_surface)
    stale = deepcopy(receipts[0])
    stale["record"]["events"][0][stale_field] = 999
    event_body = {
        key: value
        for key, value in stale["record"]["events"][0].items()
        if key != "event_sha256"
    }
    stale["record"]["events"][0]["event_sha256"] = krea_provenance.canonical_sha256(
        event_body
    )
    stale["record"] = {
        **{
            key: value
            for key, value in stale["record"].items()
            if key != "receipt_sha256"
        }
    }
    receipt_body = stale["record"]
    stale["record"]["receipt_sha256"] = krea_provenance.canonical_sha256(receipt_body)
    stale["file_sha256"] = _file_sha(stale["record"])
    stale["receipt_sha256"] = stale["record"]["receipt_sha256"]
    with pytest.raises(ValueError, match="keys differ"):
        timing._derive_capture(plan, controls=controls, receipt_binding=stale)


def test_stale_event_counter_fails_external_receipt_anchor(real_surface: dict) -> None:
    plan, controls, receipts, manifest = _plan_and_receipts(real_surface)
    stale = deepcopy(receipts[0])
    stale["record"]["events"][1]["counter_value"] += 1
    with pytest.raises(ValueError, match="file SHA-256 differs"):
        timing._derive_capture(plan, controls=controls, receipt_binding=stale)


@pytest.mark.parametrize("axis", ["host", "gpu"])
def test_external_live_receipt_substitution_is_rejected(
    real_surface: dict, axis: str
) -> None:
    plan, controls, receipts, manifest = _plan_and_receipts(real_surface)
    moved = deepcopy(controls)
    if axis == "host":
        moved["live_host_identity"] = deepcopy(controls["live_host_identity"])
        moved["live_host_identity"]["boot_id_sha256"] = _sha("other-boot")
        body = {
            key: value
            for key, value in moved["live_host_identity"].items()
            if key != "host_execution_identity_sha256"
        }
        moved["live_host_identity"]["host_execution_identity_sha256"] = (
            krea_provenance.canonical_sha256(body)
        )
        moved["live_host_identity_file_sha256"] = _file_sha(moved["live_host_identity"])
    else:
        moved["live_gpu_identity"] = _gpu("GPU-test-other")
        moved["live_gpu_identity_file_sha256"] = _file_sha(moved["live_gpu_identity"])
    with pytest.raises(ValueError, match="exact control replay"):
        timing._derive_capture(plan, controls=moved, receipt_binding=receipts[0])


@pytest.mark.parametrize("substitution", ["image", "entrypoint", "suffix"])
def test_probe_command_is_exactly_rendered_and_substitutions_fail(
    real_surface: dict, substitution: str
) -> None:
    plan, controls, _, _ = _plan_and_receipts(real_surface)
    probe = deepcopy(controls["probe_contract"])
    image = probe["production_image_id"]
    rendered = timing.render_probe_command(
        probe, timing_plan_sha256=plan["plan_sha256"], receipt_ordinal=0
    )
    assert rendered.count(image) == 1
    assert "--mount" in rendered
    assert "device=0" in rendered
    assert "/bin/true" not in probe["command_argv_template"]
    image_index = probe["command_argv_template"].index(image)
    if substitution == "image":
        probe["command_argv_template"][image_index] = "sha256:" + "9" * 64
    elif substitution == "entrypoint":
        probe["command_argv_template"][image_index:image_index] = [
            "--entrypoint",
            "/bin/sh",
        ]
    else:
        probe["command_argv_template"].append("--unexpected-suffix")
    probe["command_template_sha256"] = krea_provenance.canonical_sha256(
        probe["command_argv_template"]
    )
    body = {
        key: value for key, value in probe.items() if key != "probe_contract_sha256"
    }
    probe["probe_contract_sha256"] = krea_provenance.canonical_sha256(body)
    with pytest.raises(ValueError, match="drifted"):
        timing.validate_probe_contract(probe)


def test_probe_gpu_range_is_exact(real_surface: dict) -> None:
    _, controls, _, _ = _plan_and_receipts(real_surface)
    probe = controls["probe_contract"]
    kwargs = {
        "created_at_utc": probe["created_at_utc"],
        "production_image_id": probe["production_image_id"],
        "measurement_tool_sha256": probe["measurement_tool_sha256"],
        "collector_executable_sha256": probe["collector_executable_sha256"],
        "executable_sha256": probe["executable_sha256"],
        "fixture_role": probe["fixture_manifest"]["role"],
        "fixture_manifest_sha256": probe["fixture_manifest"]["manifest_sha256"],
        "training_archive_sha256": probe["training_archive"]["sha256"],
        "training_archive_bytes": probe["training_archive"]["bytes"],
        "profile_id": probe["command_fields"]["profile_id"],
        "hard_budget_s": probe["command_fields"]["hard_budget_s"],
        "mount_sources": {
            row["purpose"]: row["source_root"] for row in probe["mounts"]
        },
        "trigger_word": probe["command_fields"]["trigger_word"],
    }
    assert (
        timing.seal_probe_contract(gpu_device=3, **kwargs)["command_fields"][
            "gpu_device"
        ]
        == 3
    )
    with pytest.raises(ValueError, match="0, 1, 2, or 3"):
        timing.seal_probe_contract(gpu_device=4, **kwargs)


def test_exact_thirty_matrix_envelopes_render_isolated_three_plus_one_commands(
    real_surface: dict,
) -> None:
    _, controls, _, _ = _plan_and_receipts(real_surface)
    base = controls["probe_contract"]
    assignments = [
        *([(f"C{index}", 2700.0, index - 1) for index in range(1, 5)]),
        ("B-0p5-small", 1800.0, 0),
        ("B-0p5-large", 1800.0, 1),
        ("B-0p75-small", 2700.0, 2),
        ("B-0p75-large", 2700.0, 3),
        ("B-1-small", 3600.0, 0),
        ("B-1-large", 3600.0, 1),
    ]
    contracts = []
    for role, hard_budget_s, gpu in assignments:
        for profile_id in ("K1", "K3", "K4"):
            contract = timing.seal_probe_contract(
                created_at_utc=base["created_at_utc"],
                production_image_id=base["production_image_id"],
                measurement_tool_sha256=base["measurement_tool_sha256"],
                collector_executable_sha256=base["collector_executable_sha256"],
                executable_sha256=base["executable_sha256"],
                gpu_device=gpu,
                fixture_role=role,
                fixture_manifest_sha256=_sha("fixture-" + role),
                training_archive_sha256=_sha("archive-" + role),
                training_archive_bytes=1234,
                profile_id=profile_id,
                hard_budget_s=hard_budget_s,
                mount_sources={
                    row["purpose"]: row["source_root"] for row in base["mounts"]
                },
                trigger_word="SN56",
            )
            assert timing.validate_probe_contract(contract) == contract
            assert [
                row["measurement_role"] for row in contract["receipt_schedule"]
            ] == [
                "timing_measurement",
                "timing_measurement",
                "timing_measurement",
                "held_out_end_to_end",
            ]
            commands = [
                timing.render_probe_command(
                    contract,
                    timing_plan_sha256=_sha(f"plan-{role}-{profile_id}"),
                    receipt_ordinal=ordinal,
                )
                for ordinal in range(4)
            ]
            assert len({tuple(command) for command in commands}) == 4
            assert all(f"device={gpu}" in command for command in commands)
            assert all(
                contract["production_image_id"] in command for command in commands
            )
            contracts.append(contract)
    assert len(contracts) == 30
    assert len({item["probe_contract_sha256"] for item in contracts}) == 30


def test_host_collector_event_stream_is_real_three_save_plus_terminal_chain() -> None:
    stream = collector._EventStream(0)
    stream.emit("startup-r0", "startup", "begin", 1)
    stream.evidence("config-control.json", collector._IN_CLOSE_WRITE)
    for index in range(3):
        name = f"checkpoint_{index:09d}.safetensors"
        stream.checkpoint(name, collector._IN_CREATE)
        stream.checkpoint(name, collector._IN_CLOSE_WRITE)
    stream.evidence("training-terminal.json", collector._IN_CLOSE_WRITE)
    stream.evidence("forge_checkpoint_selection.json", collector._IN_CLOSE_WRITE)
    events = stream.finish()
    samples, _ = timing._derive_samples(
        events, expected_units=timing._EVENT_UNIT_SCHEDULE
    )
    assert len(samples["startup"]) == 1
    assert len(samples["optimizer_update"]) == 1
    assert len(samples["checkpoint_save"]) == 3
    assert len(samples["finalization"]) == 1
    assert len(samples["upload"]) == 1


def test_coherent_counter_inflation_fails_schedule_and_original_manifest(
    real_surface: dict,
) -> None:
    plan, controls, receipts, manifest = _plan_and_receipts(real_surface)
    changed = deepcopy(receipts)
    event = changed[0]["record"]["events"][1]
    event["counter_value"] += 1000
    event_body = {key: value for key, value in event.items() if key != "event_sha256"}
    event["event_sha256"] = krea_provenance.canonical_sha256(event_body)
    _rehash_receipt(changed[0])
    with pytest.raises(ValueError, match="predeclared schedule"):
        timing._derive_capture(plan, controls=controls, receipt_binding=changed[0])
    with pytest.raises(ValueError, match="exactly exhaust"):
        timing._derive_evidence(
            plan=plan,
            controls=controls,
            **_manifest_args(manifest),
            receipt_bindings=changed,
            framework_stop_boundary_s=225.0,
            framework_stop_boundary_source_sha256=_sha("stop-boundary"),
        )


def test_seed_role_canary_is_rejected_and_never_persisted(real_surface: dict) -> None:
    plan, controls, receipts, _ = _plan_and_receipts(real_surface)
    changed = deepcopy(receipts[0])
    changed["record"]["seed_role"] = "PRIVATE-SEED-CANARY"
    _rehash_receipt(changed)
    with pytest.raises(ValueError, match="seed/role differs"):
        timing._derive_capture(plan, controls=controls, receipt_binding=changed)


def test_same_window_renamed_run_is_rejected(real_surface: dict) -> None:
    plan, controls, receipts, _ = _plan_and_receipts(real_surface)
    changed = deepcopy(receipts)
    left = receipts[0]["record"]["command"]
    command = changed[1]["record"]["command"]
    for key in (
        "started_unix_ns",
        "ended_unix_ns",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "invocation_id",
    ):
        command[key] = left[key]
    changed[1]["record"]["run_receipt"] = timing.seal_run_receipt(
        measurement_role="timing_measurement",
        artifact_manifest_file_sha256=_sha("renamed-run-file"),
        artifact_manifest_sha256=_sha("renamed-run-semantic"),
        recorded_unix_ns=command["ended_unix_ns"] + 1_000_000_000,
    )
    _rehash_receipt(changed[1])
    with pytest.raises(ValueError, match="command invocation|intervals overlap"):
        _receipt_manifest(changed, controls)


def test_future_run_receipts_cannot_postdate_anchored_manifest(
    real_surface: dict,
) -> None:
    _, controls, receipts, _ = _plan_and_receipts(real_surface)
    changed = deepcopy(receipts)
    manifest_ns = timing._utc_ns("2026-07-30T01:00:00Z", "manifest")
    for index, binding in enumerate(changed, 1):
        old = binding["record"]["run_receipt"]
        binding["record"]["run_receipt"] = timing.seal_run_receipt(
            measurement_role=binding["record"]["measurement_role"],
            artifact_manifest_file_sha256=old["artifact_manifest_file_sha256"],
            artifact_manifest_sha256=old["artifact_manifest_sha256"],
            recorded_unix_ns=manifest_ns + index * 1_000_000_000,
        )
        _rehash_receipt(binding)
    with pytest.raises(ValueError, match="run chronology"):
        _receipt_manifest(changed, controls)


@pytest.mark.parametrize(
    "collector_created", ["2026-07-30T00:50:00Z", "2026-07-30T01:01:00Z"]
)
def test_collector_identity_must_precede_execution_and_manifest(
    real_surface: dict, collector_created: str
) -> None:
    _, controls, receipts, _ = _plan_and_receipts(real_surface)
    collector = timing.seal_collector_identity(
        created_at_utc=collector_created,
        collector_executable_sha256=_sha("separate-receipt-collector"),
        measurement_tool_sha256=controls["probe_contract"]["measurement_tool_sha256"],
    )
    with pytest.raises(ValueError, match="collector identity postdates"):
        timing.seal_receipt_manifest(
            created_at_utc="2026-07-30T01:00:00Z",
            collector_identity=collector,
            collector_identity_file_sha256=_file_sha(collector),
            receipt_bindings=receipts,
        )


@pytest.mark.parametrize("false_zero", [False, 0.0])
def test_command_returncode_requires_nonbool_integer_zero(
    real_surface: dict, false_zero: object
) -> None:
    plan, controls, receipts, _ = _plan_and_receipts(real_surface)
    changed = deepcopy(receipts[0])
    changed["record"]["command"]["returncode"] = false_zero
    _rehash_receipt(changed)
    with pytest.raises(ValueError, match="allowlisted probe"):
        timing._derive_capture(plan, controls=controls, receipt_binding=changed)


@pytest.mark.parametrize("mode", ["clock", "chronology", "budget"])
def test_clock_chronology_and_predeclared_budget_fail_closed(
    real_surface: dict, mode: str
) -> None:
    plan, controls, receipts, manifest = _plan_and_receipts(real_surface)
    stale = deepcopy(receipts[0])
    command = stale["record"]["command"]
    if mode == "clock":
        command["ended_monotonic_ns"] += timing._CLOCK_TOLERANCE_NS + 1
    elif mode == "chronology":
        command["started_unix_ns"] = timing._utc_ns(plan["created_at_utc"], "plan")
        command["ended_unix_ns"] = command["started_unix_ns"] + 60_000_000_000
    else:
        command["ended_unix_ns"] = command["started_unix_ns"] + 3_000_000_000_000
        command["ended_monotonic_ns"] = (
            command["started_monotonic_ns"] + 3_000_000_000_000
        )
    if mode != "clock":
        unsealed_command = {
            key: value for key, value in command.items() if key != "invocation_id"
        }
        command["invocation_id"] = timing._command(
            unsealed_command,
            probe=controls["probe_contract"],
            timing_plan_sha256=plan["plan_sha256"],
            receipt_ordinal=0,
            image_id=plan["production_image_id"],
            sealed=False,
        )["invocation_id"]
    body = {
        key: value for key, value in stale["record"].items() if key != "receipt_sha256"
    }
    stale["record"]["receipt_sha256"] = krea_provenance.canonical_sha256(body)
    stale["file_sha256"] = _file_sha(stale["record"])
    stale["receipt_sha256"] = stale["record"]["receipt_sha256"]
    expected = {
        "clock": "clocks disagree",
        "chronology": "predates",
        "budget": "hard budget",
    }[mode]
    with pytest.raises(ValueError, match=expected):
        timing._derive_capture(plan, controls=controls, receipt_binding=stale)


def test_reused_monotonic_window_is_rejected_even_with_later_unix_window(
    real_surface: dict,
) -> None:
    plan, controls, receipts, manifest = _plan_and_receipts(real_surface)
    duplicated = deepcopy(receipts)
    duplicated[1]["record"]["events"] = deepcopy(receipts[0]["record"]["events"])
    command = duplicated[1]["record"]["command"]
    command["started_monotonic_ns"] = receipts[0]["record"]["command"][
        "started_monotonic_ns"
    ]
    command["ended_monotonic_ns"] = receipts[0]["record"]["command"][
        "ended_monotonic_ns"
    ]
    unsealed_command = {
        key: value for key, value in command.items() if key != "invocation_id"
    }
    command["invocation_id"] = timing._command(
        unsealed_command,
        probe=controls["probe_contract"],
        timing_plan_sha256=plan["plan_sha256"],
        receipt_ordinal=1,
        image_id=plan["production_image_id"],
        sealed=False,
    )["invocation_id"]
    duplicated[1]["record"]["event_stream_sha256"] = krea_provenance.canonical_sha256(
        duplicated[1]["record"]["events"]
    )
    body = {
        key: value
        for key, value in duplicated[1]["record"].items()
        if key != "receipt_sha256"
    }
    duplicated[1]["record"]["receipt_sha256"] = krea_provenance.canonical_sha256(body)
    duplicated[1]["file_sha256"] = _file_sha(duplicated[1]["record"])
    duplicated[1]["receipt_sha256"] = duplicated[1]["record"]["receipt_sha256"]
    with pytest.raises(ValueError, match="intervals overlap"):
        _receipt_manifest(duplicated, controls)


def test_replay_requires_both_external_bundle_anchors(
    real_surface: dict, tmp_path: Path
) -> None:
    plan, controls, receipts, manifest = _plan_and_receipts(real_surface)
    root = tmp_path / "anchored"
    try:
        timing.produce_bundle(
            plan=plan,
            controls=controls,
            **_manifest_args(manifest),
            receipt_bindings=receipts,
            framework_stop_boundary_s=225.0,
            framework_stop_boundary_source_sha256=_sha("stop-boundary"),
            output_root=root,
        )
        binding = timing.bundle_binding(root)
        with pytest.raises(ValueError, match="external trust anchor"):
            timing.replay_bundle(
                root,
                expected_bundle_file_sha256=_sha("wrong-file"),
                expected_bundle_sha256=binding["bundle_sha256"],
                controls=controls,
                **_manifest_args(manifest),
                receipt_bindings=receipts,
            )
        with pytest.raises(ValueError, match="external trust anchor"):
            timing.replay_bundle(
                root,
                expected_bundle_file_sha256=binding["bundle_file_sha256"],
                expected_bundle_sha256=_sha("wrong-semantic"),
                controls=controls,
                **_manifest_args(manifest),
                receipt_bindings=receipts,
            )
    finally:
        _restore_writable(root)


def test_post_publication_coherent_receipt_rewrite_fails_original_anchors(
    real_surface: dict, tmp_path: Path
) -> None:
    plan, controls, receipts, manifest = _plan_and_receipts(real_surface)
    root = tmp_path / "published"
    try:
        timing.produce_bundle(
            plan=plan,
            controls=controls,
            **_manifest_args(manifest),
            receipt_bindings=receipts,
            framework_stop_boundary_s=225.0,
            framework_stop_boundary_source_sha256=_sha("stop-boundary"),
            output_root=root,
        )
        bundle_anchor = timing.bundle_binding(root)
        changed = deepcopy(receipts)
        event = changed[0]["record"]["events"][3]
        event["counter_value"] += 1
        event_body = {
            key: value for key, value in event.items() if key != "event_sha256"
        }
        event["event_sha256"] = krea_provenance.canonical_sha256(event_body)
        _rehash_receipt(changed[0])
        changed_manifest = _receipt_manifest(changed, controls)
        with pytest.raises(ValueError, match="external trust anchor"):
            timing.replay_bundle(
                root,
                expected_bundle_file_sha256=bundle_anchor["bundle_file_sha256"],
                expected_bundle_sha256=bundle_anchor["bundle_sha256"],
                controls=controls,
                receipt_manifest=changed_manifest,
                expected_receipt_manifest_file_sha256=_file_sha(manifest),
                expected_receipt_manifest_sha256=manifest["receipt_manifest_sha256"],
                receipt_bindings=changed,
            )
    finally:
        _restore_writable(root)


def test_stored_margin_must_equal_exact_control_document(
    real_surface: dict, tmp_path: Path
) -> None:
    plan, controls, receipts, manifest = _plan_and_receipts(real_surface)
    root = tmp_path / "margin"
    try:
        timing.produce_bundle(
            plan=plan,
            controls=controls,
            **_manifest_args(manifest),
            receipt_bindings=receipts,
            framework_stop_boundary_s=225.0,
            framework_stop_boundary_source_sha256=_sha("stop-boundary"),
            output_root=root,
        )
        changed = deepcopy(controls)
        changed["margin_policy"] = krea_budget.seal_margin_policy(
            reviewer_identity="Jordan Margin",
            approved_at_utc="2026-07-30T00:01:00Z",
            frozen_before_capture=True,
            multiplicative_margin={metric: 1.5 for metric in timing._TIMING_METRICS},
            additive_margin_s={metric: 0.5 for metric in timing._TIMING_METRICS},
        )
        changed["margin_policy_file_sha256"] = _file_sha(changed["margin_policy"])
        binding = timing.bundle_binding(root)
        with pytest.raises(ValueError, match="exact control replay"):
            timing.replay_bundle(
                root,
                expected_bundle_file_sha256=binding["bundle_file_sha256"],
                expected_bundle_sha256=binding["bundle_sha256"],
                controls=changed,
                **_manifest_args(manifest),
                receipt_bindings=receipts,
            )
    finally:
        _restore_writable(root)
