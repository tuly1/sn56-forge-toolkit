#!/usr/bin/python3
"""Generate a non-authoritative fixture for the Week-6 wrapper integration run.

The output exercises the current timing/receipt schemas with the exact Forge
tree that the wrapper will consume.  It is synthetic integration input: it
makes no elapsed-time, hardware, model-quality, or production-release claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import struct
import sys
from typing import Any, Sequence


SHA1_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class FixtureError(RuntimeError):
    """The integration-only timing package could not be produced."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_exclusive(path: Path, payload: bytes, *, mode: int = 0o400) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, "fixture write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_validator(path: Path):
    spec = importlib.util.spec_from_file_location("sn56_fixture_validator", path)
    require(spec is not None and spec.loader is not None, "validator import failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def checked_root(value: str) -> Path:
    require(os.path.isabs(value), "materialized repository must be absolute")
    root = Path(value)
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise FixtureError("materialized repository is unavailable") from exc
    require(resolved == root and root.is_dir(), "materialized repository is indirect")
    return root


def create_fixture(args: argparse.Namespace) -> dict[str, str]:
    sys.dont_write_bytecode = True
    root = checked_root(args.materialized_repository)
    require(SHA1_RE.fullmatch(args.release_commit) is not None, "invalid release commit")
    require(SHA1_RE.fullmatch(args.release_tree) is not None, "invalid release tree")
    require(
        SHA256_RE.fullmatch(args.materialized_manifest_sha256) is not None,
        "invalid materialized manifest hash",
    )
    validator_path = root / "ops" / "release" / "sn56-week6-validate-timing-provenance.py"
    require(
        validator_path.is_file() and not validator_path.is_symlink(),
        "exact-tree validator is absent",
    )
    validator = load_validator(validator_path)
    actual_manifest = validator.materialized_tree_manifest_sha256(str(root))
    require(
        actual_manifest == args.materialized_manifest_sha256,
        "materialized tree manifest differs before fixture generation",
    )
    adaptive_timing = validator.load_forge_contract(
        str(root),
        args.release_commit,
        materialized_manifest_sha256=args.materialized_manifest_sha256,
    )
    from forge import krea_runtime
    from forge.tasks.integrity import inspect_training_artifact

    output = Path(args.output_dir)
    require(output.is_absolute(), "fixture output directory must be absolute")
    try:
        parent = output.parent.resolve(strict=True)
    except OSError as exc:
        raise FixtureError("fixture output parent is unavailable") from exc
    require(parent == output.parent and parent.is_dir(), "fixture output parent is indirect")
    try:
        os.mkdir(output, 0o700)
    except OSError as exc:
        raise FixtureError("fixture output directory could not be created exclusively") from exc

    artifact = output / "terminal-artifact.safetensors"
    metadata = {"training_info": json.dumps({"step": 1000, "epoch": 1})}
    header = json.dumps(
        {
            "__metadata__": metadata,
            "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        }
    ).encode("utf-8")
    write_exclusive(
        artifact,
        struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.0),
    )
    artifact_evidence = inspect_training_artifact(str(artifact))
    artifact_sha = sha256_file(artifact)

    source_run_id = "integration-fixture:" + args.release_commit[:32]
    session_id = "week6-clean-head-wrapper-integration"
    bundle_id = krea_runtime.LEADER_BUNDLE
    bundle_sha = krea_runtime.bundle_contract_sha256(bundle_id)
    model_type = "krea2"
    dataset_size = 24
    regime = adaptive_timing.dataset_regime(dataset_size)
    accelerator = "INTEGRATION-STUB-NO-GPU-CLAIM|0-MiB"
    rental_started = "2026-08-07T12:00:00Z"
    training_started = "2026-08-07T12:02:00Z"
    raw_at = "2026-08-07T12:25:00Z"
    profile_at = "2026-08-07T12:26:00Z"
    sealed_at = "2026-08-07T12:27:00Z"
    rental_ended = "2026-08-07T18:00:00Z"
    certificate_scope = "toolkit-krea-only"

    raw: dict[str, Any] = {
        "schema": krea_runtime.EFFECTIVE_RUNTIME_SCHEMA,
        "runtime_contract_id": krea_runtime.RUNTIME_CONTRACT_ID,
        "source_run_id": source_run_id,
        "model_type": model_type,
        "runtime_repository": krea_runtime.OWNED_RUNTIME_REPOSITORY,
        "runtime_commit": krea_runtime.OWNED_RUNTIME_COMMIT,
        "bundle": bundle_id,
        "bundle_claim": krea_runtime.bundle_claim_document(bundle_id),
        "bundle_contract_sha256": bundle_sha,
        "generated_config_sha256": "d" * 64,
        "capability_manifest_file_sha256": "e" * 64,
        "capability_manifest_semantic_sha256": "f" * 64,
        "capabilities": sorted(krea_runtime.REQUIRED_CAPABILITIES),
        "runtime_manifest_capability_aliases": (
            krea_runtime.bundle_contract_document(bundle_id)[
                "runtime_manifest_capability_aliases"
            ]
        ),
        "timing": {
            "mode": "bootstrap_probe_unmeasured",
            "profile_sha256": None,
            "runtime_commit": krea_runtime.OWNED_RUNTIME_COMMIT,
            "measured_dataset_size": None,
            "current_dataset_size": dataset_size,
            "dataset_regime": regime,
            "accelerator_identity": accelerator,
            "accelerator_identity_evidence": "operator-attested",
        },
        "effective": {
            "planned_steps": 1000,
            "normalized_config_projection": (
                krea_runtime.bundle_contract_document(bundle_id)[
                    "normalized_config_projection"
                ]
            ),
        },
        "lifecycle": "terminal",
        "first_checkpoint_observation": {
            "bundle_id": bundle_id,
            "timing_profile_sha256": None,
            "observation_mode": "bootstrap_raw_first_checkpoint",
            "checkpoint_step": 200,
            "elapsed_since_launch_s": 260.0,
            "active_planned_steps": 1000,
            "active_plan_mutable": False,
            "active_plan_action": "observe_only_fixed_subprocess",
        },
        "training_completion_observation": {
            "training_elapsed_seconds": 1300.0,
            "returncode": 0,
            "stopped_by_deadline": False,
            "natural_completion": True,
            "artifact_path": str(artifact),
            "artifact_name": artifact.name,
            "artifact_size_bytes": artifact_evidence.size_bytes,
            "artifact_sha256": artifact_sha,
            "artifact_loadable": True,
            "artifact_checkpoint_step": 1000,
            "completed_steps": 1000,
            "scope_attempt_nonce": args.release_commit[:32],
            "artifact_file_identity": artifact_evidence.file_identity,
        },
    }
    raw["record_sha256"] = krea_runtime._canonical_sha256(raw)
    raw_path = output / "effective-runtime.json"
    write_exclusive(raw_path, krea_runtime._canonical_bytes(raw))
    raw_file_sha = sha256_file(raw_path)

    profile = adaptive_timing.produce_profile_document(
        str(raw_path),
        source_run_id=source_run_id,
        bundle_id=bundle_id,
        model_type=model_type,
        measured_dataset_size=dataset_size,
        measured_at_utc=profile_at,
        expected_accelerator_identity=accelerator,
    )
    profile_path = output / "timing-profile.json"
    write_exclusive(profile_path, canonical_bytes(profile))
    profile_file_sha = sha256_file(profile_path)

    event = {
        "event": validator.EVENT_KIND,
        "gate_session_id": session_id,
        "source_run_id": source_run_id,
        "rental_started_at_utc": rental_started,
        "rental_ended_at_utc": rental_ended,
        "training_started_at_utc": training_started,
        "raw_record_produced_at_utc": raw_at,
        "profile_produced_at_utc": profile_at,
        "sealed_at_utc": sealed_at,
        "profile_file_sha256": profile_file_sha,
        "raw_record_file_sha256": raw_file_sha,
        "terminal_artifact_file_sha256": artifact_sha,
        "profile_semantic_sha256": profile["profile_sha256"],
        "raw_record_semantic_sha256": raw["record_sha256"],
        "forge_commit": args.release_commit,
        "release_tree": args.release_tree,
        "certificate_scope": certificate_scope,
        "bundle_id": bundle_id,
        "bundle_sha256": bundle_sha,
        "model_type": model_type,
        "current_dataset_size": dataset_size,
        "dataset_regime": regime,
        "accelerator_identity": accelerator,
    }
    gate_path = output / "friday-gate.jsonl"
    write_exclusive(gate_path, canonical_bytes(event))
    gate_sha = sha256_file(gate_path)

    validation_args = argparse.Namespace(
        profile=str(profile_path),
        profile_file_sha256=profile_file_sha,
        raw_record=str(raw_path),
        raw_record_file_sha256=raw_file_sha,
        terminal_artifact=str(artifact),
        archived_terminal_artifact=str(artifact),
        terminal_artifact_file_sha256=artifact_sha,
        gate_log=str(gate_path),
        gate_log_file_sha256=gate_sha,
        source_run_id=source_run_id,
        gate_session_id=session_id,
        rental_started_at_utc=rental_started,
        rental_ended_at_utc=rental_ended,
        forge_repository=str(root),
        forge_materialized_manifest_sha256=args.materialized_manifest_sha256,
        forge_commit=args.release_commit,
        release_tree=args.release_tree,
        certificate_scope=certificate_scope,
        bundle_id=bundle_id,
        bundle_sha256=bundle_sha,
        model_type=model_type,
        current_dataset_size=str(dataset_size),
        dataset_regime=regime,
        accelerator_identity=accelerator,
        allow_dirty_forge=False,
        git_self_test_mode=False,
    )
    receipt = validator.validate(validation_args)
    require(receipt.get("state") == "PASS", "generated fixture did not validate")
    receipt_path = output / "fixture-validation-receipt.json"
    write_exclusive(receipt_path, canonical_bytes(receipt))

    require(
        validator.materialized_tree_manifest_sha256(str(root))
        == args.materialized_manifest_sha256,
        "materialized tree changed during fixture generation",
    )
    return {
        "SN56_FIXTURE_ACCELERATOR_IDENTITY": accelerator,
        "SN56_FIXTURE_BUNDLE_ID": bundle_id,
        "SN56_FIXTURE_BUNDLE_SHA256": bundle_sha,
        "SN56_FIXTURE_CURRENT_DATASET_SIZE": str(dataset_size),
        "SN56_FIXTURE_DATASET_REGIME": regime,
        "SN56_FIXTURE_FRIDAY_GATE_LOG": str(gate_path),
        "SN56_FIXTURE_GATE_SESSION_ID": session_id,
        "SN56_FIXTURE_MODEL_TYPE": model_type,
        "SN56_FIXTURE_PROFILE": str(profile_path),
        "SN56_FIXTURE_RAW_RECORD": str(raw_path),
        "SN56_FIXTURE_RENTAL_ENDED_AT_UTC": rental_ended,
        "SN56_FIXTURE_RENTAL_STARTED_AT_UTC": rental_started,
        "SN56_FIXTURE_SOURCE_RUN_ID": source_run_id,
        "SN56_FIXTURE_TERMINAL_ARTIFACT": str(artifact),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--materialized-repository", required=True)
    result.add_argument("--materialized-manifest-sha256", required=True)
    result.add_argument("--release-commit", required=True)
    result.add_argument("--release-tree", required=True)
    result.add_argument("--output-dir", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        values = create_fixture(parser().parse_args(argv))
    except FixtureError as exc:
        print(f"SN56_WEEK6_INTEGRATION_FIXTURE=FAIL reason={exc}", file=sys.stderr)
        return 1
    for key in sorted(values):
        print(f"{key}={values[key]}")
    print("SN56_WEEK6_INTEGRATION_FIXTURE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
