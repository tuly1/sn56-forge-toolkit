#!/usr/bin/env python3
"""Create and validate Stage-2 run evidence and deterministic zero controls."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
from typing import Any

try:
    from . import krea_provenance
    from . import krea_stage2_execution
    from . import krea_training_evidence
except ImportError:  # pragma: no cover
    import krea_provenance  # type: ignore[no-redef]
    import krea_stage2_execution  # type: ignore[no-redef]
    import krea_training_evidence  # type: ignore[no-redef]


RUN_KIND = "forge-krea-stage2-run-evidence"
ZERO_KIND = "forge-krea-stage2-zero-control"
_SHA = re.compile(r"[0-9a-f]{64}")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _exact(value: dict[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(
            f"{label} keys differ: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be SHA-256")
    return value


def _binding(value: Any, label: str, semantic_key: str) -> dict[str, str]:
    value = _object(value, label)
    _exact(value, {"file_sha256", semantic_key}, label)
    _sha(value["file_sha256"], f"{label}.file_sha256")
    _sha(value[semantic_key], f"{label}.{semantic_key}")
    return dict(value)


def _artifact_path(
    plan: dict[str, Any], row: dict[str, Any], run_output_root: Path
) -> Path:
    relative = PurePosixPath(str(row["path"]))
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("artifact path is unsafe")
    parts = relative.parts
    if parts[0] == "run":
        return run_output_root.joinpath(*parts[1:])
    if parts[0] == "checkpoints":
        checkpoint_mount = next(
            Path(mount["source"])
            for mount in plan["mounts"]
            if mount["purpose"] == "checkpoints"
        )
        return (
            checkpoint_mount / plan["task_id"] / plan["expected_repo_name"]
        ).joinpath(*parts[1:])
    if parts[0] == "evidence":
        evidence_mount = next(
            Path(mount["source"])
            for mount in plan["mounts"]
            if mount["purpose"] == "run_evidence"
        )
        return (evidence_mount / plan["plan_sha256"]).joinpath(*parts[1:])
    raise ValueError("artifact path has an unknown Stage-2 root")


def _rehash_artifacts(
    plan: dict[str, Any], completion: dict[str, Any], run_output_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for declared in completion["artifact_manifest"]:
        path = _artifact_path(plan, declared, run_output_root)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Stage-2 artifact is not a regular file: {path}")
        observed = {
            "path": declared["path"],
            "bytes": path.stat().st_size,
            "sha256": krea_provenance.file_sha256(path),
        }
        if observed != declared:
            raise ValueError(f"Stage-2 artifact bytes drifted: {declared['path']}")
        rows.append(observed)
    return rows


def build_run_evidence(
    *,
    plan: dict[str, Any],
    plan_file_sha256: str,
    approval: dict[str, Any],
    approval_file_sha256: str,
    completion: dict[str, Any],
    completion_file_sha256: str,
    run_output_root: Path,
    fixture_manifest: dict[str, str],
    emitted_at_utc: str,
) -> dict[str, Any]:
    resolved_plan = krea_stage2_execution.validate_plan(plan)
    resolved_approval = krea_stage2_execution.validate_approval(
        approval, plan=resolved_plan
    )
    resolved_completion = krea_stage2_execution.validate_completion(
        completion, plan=resolved_plan, approval=resolved_approval
    )
    artifacts = _rehash_artifacts(resolved_plan, resolved_completion, run_output_root)
    private_receipts = krea_stage2_execution.validate_private_run_receipts(
        resolved_plan
    )
    control = private_receipts["config_control"]
    terminal = private_receipts["training_terminal"]
    checkpoint_selection = private_receipts["checkpoint_selection"]
    if (
        control != resolved_completion["config_control_receipt"]
        or terminal != resolved_completion["training_terminal_receipt"]
        or checkpoint_selection != resolved_completion["checkpoint_selection_receipt"]
    ):
        raise ValueError("Stage-2 private receipt bindings drifted")
    candidates = [
        row
        for row in artifacts
        if row["path"].startswith("checkpoints/")
        and row["path"].endswith(".safetensors")
    ]
    if not candidates:
        raise ValueError("Stage-2 run evidence has no checkpoint candidates")
    body = {
        "schema": 1,
        "kind": RUN_KIND,
        "phase": resolved_plan["phase"],
        "cell_id": resolved_plan["cell_id"],
        "fixture_id": resolved_plan["fixture_id"],
        "seed_role": resolved_plan["seed_role"],
        "seed": resolved_plan["seed"],
        "hours": resolved_plan["hours"],
        "training_candidate_id": resolved_plan["training_candidate_id"],
        "execution_plan": {
            "file_sha256": _sha(plan_file_sha256, "plan file"),
            "plan_sha256": resolved_plan["plan_sha256"],
        },
        "execution_approval": {
            "file_sha256": _sha(approval_file_sha256, "approval file"),
            "approval_sha256": resolved_approval["approval_sha256"],
        },
        "run_completion": {
            "file_sha256": _sha(completion_file_sha256, "completion file"),
            "completion_sha256": resolved_completion["completion_sha256"],
        },
        "fixture_manifest": _binding(
            fixture_manifest, "fixture manifest", "manifest_sha256"
        ),
        "waiver_finalist_freeze": resolved_plan["waiver_finalist_freeze"],
        "confirmation_materialization": resolved_plan["confirmation_materialization"],
        "owner_ratification": resolved_plan["owner_ratification"],
        "gpu_execution_authorization": resolved_plan["gpu_execution_authorization"],
        "production_identity": resolved_plan["production_identity"],
        "production_image_id": resolved_plan["production_image_id"],
        "artifacts": artifacts,
        "candidate_artifacts": candidates,
        "mechanics": resolved_completion["mechanics"],
        "emitted_at_utc": emitted_at_utc,
        "natural_completion": True,
        "fallback_used": False,
        "strict_discovery_replayed": False,
        "retroactive_plan_or_approval_claimed": False,
        "release_authorized": False,
        "production_mutation_authorized": False,
    }
    evidence = {**body, "evidence_sha256": krea_provenance.canonical_sha256(body)}
    return validate_run_evidence(
        evidence,
        plan=resolved_plan,
        approval=resolved_approval,
        completion=resolved_completion,
    )


def validate_run_evidence(
    value: Any,
    *,
    plan: dict[str, Any],
    approval: dict[str, Any],
    completion: dict[str, Any],
) -> dict[str, Any]:
    evidence = _object(value, "Stage-2 run evidence")
    keys = {
        "schema",
        "kind",
        "phase",
        "cell_id",
        "fixture_id",
        "seed_role",
        "seed",
        "hours",
        "training_candidate_id",
        "execution_plan",
        "execution_approval",
        "run_completion",
        "fixture_manifest",
        "waiver_finalist_freeze",
        "confirmation_materialization",
        "owner_ratification",
        "gpu_execution_authorization",
        "production_identity",
        "production_image_id",
        "artifacts",
        "candidate_artifacts",
        "mechanics",
        "emitted_at_utc",
        "natural_completion",
        "fallback_used",
        "strict_discovery_replayed",
        "retroactive_plan_or_approval_claimed",
        "release_authorized",
        "production_mutation_authorized",
        "evidence_sha256",
    }
    _exact(evidence, keys, "Stage-2 run evidence")
    body = {key: item for key, item in evidence.items() if key != "evidence_sha256"}
    resolved_plan = krea_stage2_execution.validate_plan(plan)
    resolved_approval = krea_stage2_execution.validate_approval(
        approval, plan=resolved_plan
    )
    resolved_completion = krea_stage2_execution.validate_completion(
        completion, plan=resolved_plan, approval=resolved_approval
    )
    if (
        evidence["schema"] != 1
        or evidence["kind"] != RUN_KIND
        or evidence["evidence_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("Stage-2 run evidence kind/schema/digest differs")
    expected = {
        "phase": resolved_plan["phase"],
        "cell_id": resolved_plan["cell_id"],
        "fixture_id": resolved_plan["fixture_id"],
        "seed_role": resolved_plan["seed_role"],
        "seed": resolved_plan["seed"],
        "hours": resolved_plan["hours"],
        "training_candidate_id": resolved_plan["training_candidate_id"],
        "production_image_id": resolved_plan["production_image_id"],
    }
    if any(evidence[key] != item for key, item in expected.items()):
        raise ValueError("Stage-2 run evidence cell identity differs")
    execution_plan = _binding(
        evidence["execution_plan"], "execution plan", "plan_sha256"
    )
    execution_approval = _binding(
        evidence["execution_approval"], "execution approval", "approval_sha256"
    )
    run_completion = _binding(
        evidence["run_completion"], "run completion", "completion_sha256"
    )
    _binding(evidence["fixture_manifest"], "fixture manifest", "manifest_sha256")
    if (
        execution_plan["plan_sha256"] != resolved_plan["plan_sha256"]
        or execution_approval["approval_sha256"] != resolved_approval["approval_sha256"]
        or run_completion["completion_sha256"]
        != resolved_completion["completion_sha256"]
        or evidence["waiver_finalist_freeze"] != resolved_plan["waiver_finalist_freeze"]
        or evidence["confirmation_materialization"]
        != resolved_plan["confirmation_materialization"]
        or evidence["owner_ratification"] != resolved_plan["owner_ratification"]
        or evidence["gpu_execution_authorization"]
        != resolved_plan["gpu_execution_authorization"]
        or evidence["production_identity"] != resolved_plan["production_identity"]
        or evidence["artifacts"] != resolved_completion["artifact_manifest"]
        or evidence["mechanics"] != resolved_completion["mechanics"]
    ):
        raise ValueError("Stage-2 run evidence authority/artifact binding differs")
    if evidence["candidate_artifacts"] != [
        row
        for row in evidence["artifacts"]
        if row["path"].startswith("checkpoints/")
        and row["path"].endswith(".safetensors")
    ]:
        raise ValueError("Stage-2 candidate artifact projection differs")
    for key in ("natural_completion",):
        if evidence[key] is not True:
            raise ValueError("Stage-2 run did not complete naturally")
    for key in (
        "fallback_used",
        "strict_discovery_replayed",
        "retroactive_plan_or_approval_claimed",
        "release_authorized",
        "production_mutation_authorized",
    ):
        if evidence[key] is not False:
            raise ValueError(f"Stage-2 run evidence overclaims {key}")
    krea_stage2_execution._utc(evidence["emitted_at_utc"], "emitted_at_utc")
    return dict(evidence)


def emit_zero_control(
    *,
    template_artifact: Path,
    template_run_evidence: dict[str, Any],
    output_artifact: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    """Byte-generate an all-zero LoRA preserving the template tensor layout."""

    template_artifact = template_artifact.resolve(strict=True)
    if template_artifact.is_symlink() or not template_artifact.is_file():
        raise ValueError("zero template must be a regular file")
    evidence = _object(template_run_evidence, "template run evidence")
    if (
        evidence.get("kind") != RUN_KIND
        or evidence.get("natural_completion") is not True
    ):
        raise ValueError("zero template requires valid Stage-2 run evidence")
    template_sha = krea_provenance.file_sha256(template_artifact)
    if not any(
        row.get("sha256") == template_sha
        for row in evidence.get("candidate_artifacts", [])
        if isinstance(row, dict)
    ):
        raise ValueError("zero template is absent from Stage-2 run evidence")
    for output in (output_artifact, output_manifest):
        if os.path.lexists(output):
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
    source_header, source_data = krea_training_evidence._read_safetensors(
        template_artifact
    )
    layout = krea_training_evidence._tensor_layout(source_header)
    if not layout or max(
        source_header[row["key"]]["data_offsets"][1] for row in layout
    ) != len(source_data):
        raise ValueError("zero template tensor layout is inconsistent")
    offset = 0
    header: dict[str, Any] = {
        "__metadata__": {
            "forge_control": "stage2-deterministic-zero-lora-v1",
            "template_sha256": template_sha,
            "template_run_evidence_sha256": evidence["evidence_sha256"],
        }
    }
    for row in layout:
        header[row["key"]] = {
            "dtype": row["dtype"],
            "shape": row["shape"],
            "data_offsets": [offset, offset + row["bytes"]],
        }
        offset += row["bytes"]
    header_bytes = json.dumps(
        header, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    with output_artifact.open("xb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        block = b"\0" * min(max(offset, 1), 8 * 1024 * 1024)
        remaining = offset
        while remaining:
            chunk = block[: min(remaining, len(block))]
            handle.write(chunk)
            remaining -= len(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    body = {
        "schema": 1,
        "kind": ZERO_KIND,
        "artifact": {
            "path": str(output_artifact),
            "bytes": output_artifact.stat().st_size,
            "sha256": krea_provenance.file_sha256(output_artifact),
        },
        "template": {
            "path": str(template_artifact),
            "bytes": template_artifact.stat().st_size,
            "sha256": template_sha,
        },
        "template_run_evidence_sha256": evidence["evidence_sha256"],
        "tensor_layout": layout,
        "tensor_layout_sha256": krea_provenance.canonical_sha256(layout),
        "all_tensor_bytes_zero": True,
        "strict_discovery_replayed": False,
        "release_authorized": False,
        "production_mutation_authorized": False,
    }
    manifest = {**body, "manifest_sha256": krea_provenance.canonical_sha256(body)}
    validate_zero_control(manifest, artifact_path=output_artifact)
    payload = krea_provenance.canonical_bytes(manifest) + b"\n"
    with output_manifest.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def validate_zero_control(
    value: Any, *, artifact_path: Path | None = None
) -> dict[str, Any]:
    manifest = _object(value, "Stage-2 zero control")
    keys = {
        "schema",
        "kind",
        "artifact",
        "template",
        "template_run_evidence_sha256",
        "tensor_layout",
        "tensor_layout_sha256",
        "all_tensor_bytes_zero",
        "strict_discovery_replayed",
        "release_authorized",
        "production_mutation_authorized",
        "manifest_sha256",
    }
    _exact(manifest, keys, "Stage-2 zero control")
    body = {key: item for key, item in manifest.items() if key != "manifest_sha256"}
    if (
        manifest["schema"] != 1
        or manifest["kind"] != ZERO_KIND
        or manifest["manifest_sha256"] != krea_provenance.canonical_sha256(body)
        or manifest["tensor_layout_sha256"]
        != krea_provenance.canonical_sha256(manifest["tensor_layout"])
        or manifest["all_tensor_bytes_zero"] is not True
    ):
        raise ValueError("Stage-2 zero-control identity differs")
    for key in (
        "strict_discovery_replayed",
        "release_authorized",
        "production_mutation_authorized",
    ):
        if manifest[key] is not False:
            raise ValueError("Stage-2 zero control overclaims authority")
    artifact = _object(manifest["artifact"], "zero artifact")
    _exact(artifact, {"path", "bytes", "sha256"}, "zero artifact")
    template = _object(manifest["template"], "zero template")
    _exact(template, {"path", "bytes", "sha256"}, "zero template")
    for row, label in ((artifact, "zero artifact"), (template, "zero template")):
        if (
            not isinstance(row["path"], str)
            or not row["path"]
            or isinstance(row["bytes"], bool)
            or not isinstance(row["bytes"], int)
            or row["bytes"] <= 0
        ):
            raise ValueError(f"{label} identity is invalid")
        _sha(row["sha256"], f"{label}.sha256")
    _sha(
        manifest["template_run_evidence_sha256"],
        "template_run_evidence_sha256",
    )
    path = Path(artifact["path"]) if artifact_path is None else artifact_path
    path = path.resolve(strict=True)
    if (
        path.is_symlink()
        or not path.is_file()
        or str(path) != str(Path(artifact["path"]).resolve(strict=True))
        or path.stat().st_size != artifact["bytes"]
        or krea_provenance.file_sha256(path) != artifact["sha256"]
    ):
        raise ValueError("Stage-2 zero artifact binding differs")
    header, data = krea_training_evidence._read_safetensors(path)
    layout = krea_training_evidence._tensor_layout(header)
    metadata = header.get("__metadata__")
    if (
        layout != manifest["tensor_layout"]
        or any(data)
        or not isinstance(metadata, dict)
        or metadata.get("forge_control") != "stage2-deterministic-zero-lora-v1"
        or metadata.get("template_sha256") != manifest["template"]["sha256"]
        or metadata.get("template_run_evidence_sha256")
        != manifest["template_run_evidence_sha256"]
    ):
        raise ValueError("Stage-2 zero tensors are not exactly all zero")
    return dict(manifest)
