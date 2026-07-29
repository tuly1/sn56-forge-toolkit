#!/usr/bin/env python3
"""Stage-three Krea run evidence and explicit zero-LoRA control evidence.

The producer consumes an already-approved stage-two execution plan.  One
natural-completion record enumerates every valid periodic/final artifact from
the active scope; each distinct artifact receives a per-candidate binding.
Checkpoint selection is intentionally absent.  Selection happens only after
the separately sealed exact-score batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile
from typing import Any, Iterable, Sequence

try:
    from . import batch_evaluate_krea as batch
    from . import krea_execution_plan
    from . import krea_fixture
    from . import krea_provenance
except ImportError:  # pragma: no cover - direct script execution.
    import batch_evaluate_krea as batch  # type: ignore[no-redef]
    import krea_execution_plan  # type: ignore[no-redef]
    import krea_fixture  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]


_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _canonical_file(path: Path, value: Any) -> str:
    payload = krea_provenance.canonical_bytes(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _load_canonical(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = batch._safe_file(path, label)
    value, digest, raw = batch._load_json_file(path, label)
    batch._canonical_control_file(value, raw, label)
    return value, digest


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _publish_directory(temporary: Path, output: Path) -> None:
    output = Path(os.path.abspath(os.path.expanduser(output)))
    batch._reject_symlink_ancestors(output.parent, "evidence output parent")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing existing evidence output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    batch._reject_symlink_ancestors(output.parent, "evidence output parent")
    output.mkdir(mode=0o700)
    try:
        for child in sorted(temporary.iterdir(), key=lambda item: item.name):
            os.rename(child, output / child.name)
        temporary.rmdir()
        output_fd = os.open(output, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(output_fd)
        finally:
            os.close(output_fd)
        fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except BaseException:
        # Keep a claimed/partial final directory for audit.  Retrying under the
        # same task id is forbidden until a human resolves the evidence state.
        raise


def load_execution_controls(
    execution_plan_path: Path, execution_approval_path: Path
) -> tuple[dict[str, Any], str, dict[str, Any], str, dict[str, Any]]:
    plan, plan_file_sha = _load_canonical(execution_plan_path, "execution plan")
    resolved = krea_execution_plan.validate_plan(plan)
    approval, approval_file_sha = _load_canonical(
        execution_approval_path, "execution approval"
    )
    krea_execution_plan.validate_approval(approval, plan=plan)
    return plan, plan_file_sha, approval, approval_file_sha, resolved


def _actual_recipe(condition: dict[str, Any]) -> dict[str, Any]:
    process = condition["resolved_config"]["config"]["process"][0]
    train = process["train"]
    dataset = process["datasets"][0]
    network = process["network"]
    save = process["save"]
    ema = train.get("ema_config", {})
    guidance_enabled = bool(train.get("do_differential_guidance", False))
    envelope = (
        condition.get("budget", {})
        .get("throughput_profile", {})
        .get("execution_envelope", {})
    )
    replicas = envelope.get("data_parallel_replicas")
    if replicas != 1:
        raise ValueError("Krea evidence requires exactly one data-parallel replica")
    return {
        "planned_steps": train["steps"],
        "learning_rate": train["lr"],
        "rank": network["linear"],
        "alpha": network["linear_alpha"],
        "optimizer": train["optimizer"],
        "optimizer_parameters": train["optimizer_params"],
        "loss": train["loss_type"],
        "guidance": {
            "enabled": guidance_enabled,
            "scale": (
                train.get("differential_guidance_scale") if guidance_enabled else None
            ),
        },
        "scheduler": train["noise_scheduler"],
        "dropout": dataset["caption_dropout_rate"],
        "gradient_accumulation": train["gradient_accumulation"],
        "effective_batch": (
            train["batch_size"] * train["gradient_accumulation"] * replicas
        ),
        "ema": {"enabled": ema.get("use_ema"), "decay": ema.get("ema_decay")},
        "save_cadence": save["save_every"],
    }


def _validate_completed_condition(
    condition: dict[str, Any],
    *,
    plan: dict[str, Any],
    plan_file_sha: str,
    approval: dict[str, Any],
    approval_file_sha: str,
    resolved: dict[str, Any],
) -> None:
    if (
        condition.get("schema") != 2
        or condition.get("kind") != "forge-krea2-calibration-run"
        or condition.get("complete") is not True
        or condition.get("arm_id") != plan["arm_id"]
        or condition.get("model") != "krea/Krea-2-Raw"
        or condition.get("task_id") != plan["task_id"]
        or condition.get("expected_repo_name") != plan["expected_repo_name"]
        or condition.get("execution_plan_sha256") != plan["plan_sha256"]
        or condition.get("execution_plan_file_sha256") != plan_file_sha
        or condition.get("execution_approval_sha256") != approval["approval_sha256"]
        or condition.get("execution_approval_file_sha256") != approval_file_sha
        or condition.get("in_task_proxy_selection")
        != {"enabled": False, "reserve_s": 0}
    ):
        raise ValueError("run record is not a completed approved Krea execution")
    budget = batch._object(condition.get("budget"), "run budget")
    provenance = batch._object(condition.get("provenance"), "run provenance")
    after_split = batch._object(
        condition.get("dataset_after_split"), "run dataset_after_split"
    )
    attempt = batch._object(condition.get("attempt"), "run attempt")
    if (
        budget.get("plan_sha256") != plan["budget_plan_sha256"]
        or budget.get("plan") != plan["budget_plan"]
        or budget.get("throughput_profile") != resolved["throughput_profile"]
        or provenance.get("host_execution_manifest_sha256")
        != resolved["host_execution_manifest"]["host_execution_identity_sha256"]
        or after_split.get("approved_exact_evaluation_sha256")
        != resolved["fixture"]["evaluation_dataset_identity"]["sha256"]
        or attempt.get("planned_steps") != plan["schedule"]["planned_steps"]
    ):
        raise ValueError("run record contradicts its sealed budget/host/fixture")
    telemetry = condition.get("telemetry")
    if not isinstance(telemetry, dict) or not isinstance(telemetry.get("events"), list):
        raise ValueError("run telemetry is absent")
    names = [row.get("name") for row in telemetry["events"] if isinstance(row, dict)]
    if any(
        isinstance(name, str)
        and (
            name.endswith("_failed")
            or "fallback" in name
            or name.startswith("holdout_")
        )
        for name in names
    ):
        raise ValueError("run telemetry contains failure, fallback, or proxy scoring")
    for required in ("toolkit_start", "toolkit_end", "toolkit_metrics", "run_complete"):
        if names.count(required) != 1:
            raise ValueError(f"run telemetry lacks exactly one {required} event")
    end = next(row for row in telemetry["events"] if row.get("name") == "toolkit_end")
    metrics = next(
        row for row in telemetry["events"] if row.get("name") == "toolkit_metrics"
    )
    planned = plan["schedule"]["planned_steps"]
    if (
        end.get("returncode") != 0
        or end.get("stopped_by_deadline") is not False
        or metrics.get("last_step") != planned
    ):
        raise ValueError("training did not reach natural planned completion")
    actual = _actual_recipe(condition)
    effective = {
        name: row["effective_value"]
        for name, row in plan["execution_recipe"]["fields"].items()
    }
    # submitted_step is per candidate; selector is a later decision.  They must
    # remain unresolved/neutral in the pretraining recipe.
    if effective["submitted_step"] is not None or effective["selector"] is not None:
        raise ValueError(
            "pretraining execution recipe must not choose a submitted step"
        )
    expected = {
        name: value
        for name, value in effective.items()
        if name not in {"submitted_step", "selector"}
    }
    if actual != expected:
        mismatches = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key in actual
            if actual[key] != expected[key]
        }
        raise ValueError(f"resolved config contradicts execution recipe: {mismatches}")


def _candidate_step(path: Path, *, repo: str, planned: int) -> tuple[int, str]:
    if path.name == f"{repo}.safetensors":
        return planned, "exact_final"
    match = re.fullmatch(rf"{re.escape(repo)}_(\d+)\.safetensors", path.name)
    if match is None:
        raise ValueError(f"candidate has an unsupported current-run name: {path.name}")
    step = int(match.group(1))
    if not 0 < step <= planned:
        raise ValueError(f"candidate step is outside the run: {path.name}")
    return step, "periodic"


def _candidate_records(
    paths: Iterable[Path], *, repo: str, planned: int, scope_hashes: dict[str, str]
) -> list[dict[str, Any]]:
    if (
        not isinstance(scope_hashes, dict)
        or not scope_hashes
        or any(
            not isinstance(name, str)
            or Path(name).name != name
            or name in {"", ".", ".."}
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
            for name, digest in scope_hashes.items()
        )
    ):
        raise ValueError("current-scope candidate map is malformed")
    supplied: dict[str, Path] = {}
    for raw in paths:
        path = batch._safe_file(raw, "current-run candidate")
        if path.name in supplied:
            raise ValueError(f"duplicate supplied candidate name: {path.name}")
        supplied[path.name] = path
    if set(supplied) != set(scope_hashes):
        raise ValueError(
            "supplied candidates do not exhaust current scope: "
            f"missing={sorted(set(scope_hashes) - set(supplied))}, "
            f"extra={sorted(set(supplied) - set(scope_hashes))}"
        )
    for name, path in supplied.items():
        if krea_provenance.file_sha256(path) != scope_hashes[name]:
            raise ValueError(f"candidate is stale in current scope: {name}")

    by_sha: dict[str, dict[str, Any]] = {}
    for name in sorted(supplied):
        path = supplied[name]
        if path.name == "last.safetensors":
            # last is a publication alias, not a separately trained candidate.
            continue
        step, role = _candidate_step(path, repo=repo, planned=planned)
        digest = krea_provenance.file_sha256(path)
        if scope_hashes.get(path.name) != digest:
            raise ValueError(f"candidate is absent/stale in current scope: {path.name}")
        row = by_sha.get(digest)
        alias = {"name": path.name, "role": role, "step": step}
        if row is None:
            by_sha[digest] = {
                "candidate_id": f"step-{step}-{digest[:12]}",
                "sha256": digest,
                "step": step,
                "fraction_numerator": step,
                "fraction_denominator": planned,
                "aliases": [alias],
                "canonical_path": str(path),
            }
        else:
            if row["step"] != step:
                raise ValueError(
                    "byte-identical artifacts claim different training steps"
                )
            row["aliases"].append(alias)
    rows = sorted(by_sha.values(), key=lambda row: (row["step"], row["sha256"]))
    if not rows or rows[-1]["step"] != planned:
        raise ValueError("candidate set lacks an exact natural final")
    exact_final_name = f"{repo}.safetensors"
    if exact_final_name not in supplied or "last.safetensors" not in supplied:
        raise ValueError("candidate scope lacks exact-final/publication aliases")
    if scope_hashes["last.safetensors"] != scope_hashes[exact_final_name]:
        raise ValueError("last.safetensors does not equal the exact natural final")
    actual_steps = [row["step"] for row in rows]
    # The caller validates the exact predeclared grid after return.
    if len(actual_steps) != len(set(actual_steps)):
        raise ValueError("candidate set contains ambiguous step identities")
    for row in rows:
        row["aliases"].sort(key=lambda alias: alias["name"])
    return rows


def _training_identity(
    training_dir: Path, *, fixture: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = fixture["training_dataset_identity"]
    observed, rows = krea_fixture._rows(  # Same code as the pre-GPU curator.
        batch._safe_directory(training_dir, "training directory"),
        role=fixture["experimental_role"],
        list_supported_images=lambda _root, _extensions: list(
            expected["evaluator_order"]
        ),
        extensions=tuple(fixture["tool_identity"]["extensions"]),
        row_groups={
            row["relative_image_path"]: row["group_identity"]
            for row in fixture["training_rows"]
        },
    )
    if observed != expected or rows != fixture["training_rows"]:
        raise ValueError("training bytes differ from the approved fixture")
    return observed, rows


def emit_run_evidence(
    *,
    condition_record_path: Path,
    execution_plan_path: Path,
    execution_approval_path: Path,
    candidate_paths: Iterable[Path],
    training_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Emit one run completion plus bindings for every distinct candidate."""

    condition_path = batch._safe_file(condition_record_path, "run record")
    condition, condition_sha = _load_canonical(condition_path, "run record")
    plan_path = batch._safe_file(execution_plan_path, "execution plan")
    approval_path = batch._safe_file(execution_approval_path, "execution approval")
    plan, plan_file_sha, approval, approval_file_sha, resolved = (
        load_execution_controls(plan_path, approval_path)
    )
    _validate_completed_condition(
        condition,
        plan=plan,
        plan_file_sha=plan_file_sha,
        approval=approval,
        approval_file_sha=approval_file_sha,
        resolved=resolved,
    )
    training_identity, training_rows = _training_identity(
        training_dir, fixture=resolved["fixture"]
    )
    scope_hashes = condition.get("current_scope_candidates")
    if not isinstance(scope_hashes, dict) or not scope_hashes:
        raise ValueError("run record lacks current-scope candidate hashes")
    candidates = _candidate_records(
        candidate_paths,
        repo=plan["expected_repo_name"],
        planned=plan["schedule"]["planned_steps"],
        scope_hashes=scope_hashes,
    )
    actual_steps = [row["step"] for row in candidates]
    if actual_steps != plan["schedule"]["candidate_steps"]:
        raise ValueError(
            f"run candidate grid differs from sealed schedule: {actual_steps}"
        )

    output_dir = Path(os.path.abspath(os.path.expanduser(output_dir)))
    batch._reject_symlink_ancestors(output_dir.parent, "run-evidence output parent")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    batch._reject_symlink_ancestors(output_dir.parent, "run-evidence output parent")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        artifacts_dir = temporary / "candidates"
        artifacts_dir.mkdir(mode=0o700)
        final_dir = output_dir
        for candidate in candidates:
            source = Path(candidate["canonical_path"])
            staged_name = f"{candidate['candidate_id']}.safetensors"
            staged = artifacts_dir / staged_name
            copy_identity = batch._copy_verified(
                source, staged, expected_sha256=candidate["sha256"]
            )
            safetensors_identity = _safetensors_identity(staged)
            if safetensors_identity["bytes"] != copy_identity["bytes"]:
                raise RuntimeError(
                    "candidate size changed during safetensors validation"
                )
            os.chmod(staged, 0o400)
            candidate["source_path"] = str(source)
            candidate["canonical_path"] = str(final_dir / "candidates" / staged_name)
            candidate["bytes"] = copy_identity["bytes"]
            candidate["safetensors"] = safetensors_identity
        training_log = condition["telemetry"]
        training_log_path = temporary / "training-log.json"
        training_log_sha = _canonical_file(training_log_path, training_log)
        run_completion = {
            "schema": 3,
            "kind": "forge-krea-training-completion",
            "arm_id": plan["arm_id"],
            "task_id": plan["task_id"],
            "execution_plan_sha256": plan["plan_sha256"],
            "execution_plan_file_sha256": plan_file_sha,
            "execution_approval_sha256": approval["approval_sha256"],
            "execution_approval_file_sha256": approval_file_sha,
            "fixture_manifest_sha256": resolved["fixture"]["manifest_sha256"],
            "training_dataset_sha256": training_identity["sha256"],
            "training_rows_sha256": krea_provenance.canonical_sha256(training_rows),
            "training_archive": {
                "sha256": resolved["fixture"]["training_archive"]["sha256"],
                "bytes": resolved["fixture"]["training_archive"]["bytes"],
            },
            "execution_envelope_sha256": plan["execution_envelope_sha256"],
            "host_execution_identity_sha256": resolved["host_execution_manifest"][
                "host_execution_identity_sha256"
            ],
            "throughput_profile_sha256": resolved["throughput_profile"][
                "profile_sha256"
            ],
            "budget_plan_sha256": plan["budget_plan_sha256"],
            "schedule": plan["schedule"],
            "run_record_sha256": condition_sha,
            "training_log_sha256": training_log_sha,
            "natural_completion": True,
            "in_task_proxy_selection": {"enabled": False, "reserve_s": 0},
            "candidates": [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"canonical_path", "source_path"}
                }
                for row in candidates
            ],
        }
        completion_path = temporary / "run-completion.json"
        completion_sha = _canonical_file(completion_path, run_completion)
        bindings = []
        for index, candidate in enumerate(candidates):
            binding = {
                "schema": 2,
                "kind": "forge-krea-local-candidate-binding",
                "mode": "local_run_candidate",
                "arm_id": plan["arm_id"],
                "candidate_id": candidate["candidate_id"],
                "candidate": {
                    "path": candidate["canonical_path"],
                    "sha256": candidate["sha256"],
                    "bytes": candidate["bytes"],
                    "step": candidate["step"],
                    "fraction_numerator": candidate["fraction_numerator"],
                    "fraction_denominator": candidate["fraction_denominator"],
                    "aliases": candidate["aliases"],
                    "safetensors": candidate["safetensors"],
                },
                "execution_plan": {"path": str(plan_path), "sha256": plan_file_sha},
                "execution_approval": {
                    "path": str(approval_path),
                    "sha256": approval_file_sha,
                },
                "run_completion": {
                    "path": str(final_dir / completion_path.name),
                    "sha256": completion_sha,
                },
                "run_record": {"path": str(condition_path), "sha256": condition_sha},
                "training_log": {
                    "path": str(final_dir / training_log_path.name),
                    "sha256": training_log_sha,
                },
                "evaluation_dataset_sha256": resolved["fixture"][
                    "evaluation_dataset_identity"
                ]["sha256"],
            }
            binding_path = temporary / f"candidate-{index:03d}.json"
            binding_sha = _canonical_file(binding_path, binding)
            bindings.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "candidate_sha256": candidate["sha256"],
                    "binding": {
                        "path": str(final_dir / binding_path.name),
                        "sha256": binding_sha,
                    },
                }
            )
        bundle_body = {
            "schema": 2,
            "kind": "forge-krea-run-evidence-bundle",
            "arm_id": plan["arm_id"],
            "execution_plan_sha256": plan["plan_sha256"],
            "run_completion": {
                "path": str(final_dir / completion_path.name),
                "sha256": completion_sha,
            },
            "candidate_bindings": bindings,
        }
        bundle = {
            **bundle_body,
            "bundle_sha256": krea_provenance.canonical_sha256(bundle_body),
        }
        _canonical_file(temporary / "bundle.json", bundle)
        _publish_directory(temporary, output_dir)
        published_artifacts = output_dir / "candidates"
        os.chmod(published_artifacts, 0o500)
        published_fd = os.open(
            published_artifacts, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(published_fd)
        finally:
            os.close(published_fd)
        validate_run_evidence(output_dir / "bundle.json")
        return bundle
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def _read_safetensors(path: Path) -> tuple[dict[str, Any], bytes]:
    path = batch._safe_file(path, "safetensors artifact")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("safetensors artifact changed while read")
    if len(raw) < 10:
        raise ValueError("safetensors file is truncated")
    header_length = struct.unpack("<Q", raw[:8])[0]
    if header_length <= 1 or 8 + header_length > len(raw):
        raise ValueError("safetensors header length is invalid")
    try:
        header = json.loads(
            raw[8 : 8 + header_length],
            object_pairs_hook=_json_object_without_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("safetensors header is invalid") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    return header, raw[8 + header_length :]


def _tensor_layout(header: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(name for name in header if name != "__metadata__"):
        tensor = header[key]
        if not isinstance(tensor, dict) or set(tensor) != {
            "dtype",
            "shape",
            "data_offsets",
        }:
            raise ValueError(f"invalid safetensors entry: {key}")
        dtype = tensor["dtype"]
        shape = tensor["shape"]
        offsets = tensor["data_offsets"]
        if (
            dtype not in _DTYPE_BYTES
            or not isinstance(shape, list)
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in shape
            )
        ):
            raise ValueError(f"unsupported safetensors tensor: {key}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in offsets
            )
        ):
            raise ValueError(f"invalid safetensors offsets: {key}")
        elements = 1
        for dimension in shape:
            elements *= dimension
        size = elements * _DTYPE_BYTES[dtype]
        if offsets[1] - offsets[0] != size:
            raise ValueError(f"safetensors size/shape mismatch: {key}")
        rows.append({"key": key, "dtype": dtype, "shape": shape, "bytes": size})
    if not rows:
        raise ValueError("zero control template contains no tensors")
    ranges = sorted(
        (
            header[row["key"]]["data_offsets"][0],
            header[row["key"]]["data_offsets"][1],
        )
        for row in rows
    )
    cursor = 0
    for start, end in ranges:
        if start != cursor or end < start:
            raise ValueError("safetensors offsets are overlapping or non-contiguous")
        cursor = end
    return rows


def _safetensors_identity(path: Path) -> dict[str, Any]:
    """Return a strict, content-derived identity for one complete artifact."""

    path = batch._safe_file(path, "candidate safetensors")
    header, data = _read_safetensors(path)
    layout = _tensor_layout(header)
    maximum = max(header[row["key"]]["data_offsets"][1] for row in layout)
    if maximum != len(data):
        raise ValueError("candidate safetensors has trailing or missing tensor bytes")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ValueError("candidate safetensors metadata must be string-to-string")
    header_bytes = json.dumps(
        header, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return {
        "bytes": path.stat().st_size,
        "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
        "metadata": dict(sorted(metadata.items())),
        "metadata_sha256": krea_provenance.canonical_sha256(metadata),
        "tensor_layout_sha256": krea_provenance.canonical_sha256(layout),
        "tensor_count": len(layout),
        "tensor_data_bytes": len(data),
    }


def _validate_candidate_row(
    raw: Any,
    *,
    plan: dict[str, Any],
    artifact_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    row = batch._object(raw, "run-evidence candidate")
    required = {
        "candidate_id",
        "sha256",
        "step",
        "fraction_numerator",
        "fraction_denominator",
        "aliases",
        "bytes",
        "safetensors",
    }
    batch._exact_keys(row, required, "run-evidence candidate")
    candidate_id = row["candidate_id"]
    planned = plan["schedule"]["planned_steps"]
    if (
        not isinstance(candidate_id, str)
        or not _SAFE_ID.fullmatch(candidate_id)
        or _digest(row["sha256"], "candidate") != row["sha256"]
        or row["candidate_id"] != f"step-{row['step']}-{row['sha256'][:12]}"
        or isinstance(row["step"], bool)
        or not isinstance(row["step"], int)
        or row["step"] <= 0
        or row["step"] > planned
        or row["fraction_numerator"] != row["step"]
        or row["fraction_denominator"] != planned
        or isinstance(row["bytes"], bool)
        or not isinstance(row["bytes"], int)
        or row["bytes"] <= 0
    ):
        raise ValueError("run-evidence candidate identity is invalid")
    aliases = row["aliases"]
    if not isinstance(aliases, list) or not aliases:
        raise ValueError("run-evidence candidate aliases are absent")
    scope: dict[str, str] = {}
    normalized_aliases = []
    for raw_alias in aliases:
        alias = batch._object(raw_alias, "run-evidence candidate alias")
        batch._exact_keys(
            alias, {"name", "role", "step"}, "run-evidence candidate alias"
        )
        name = alias["name"]
        if not isinstance(name, str) or Path(name).name != name or name in scope:
            raise ValueError("run-evidence candidate alias name is invalid/duplicate")
        step, role = _candidate_step(
            Path(name), repo=plan["expected_repo_name"], planned=planned
        )
        if alias != {"name": name, "role": role, "step": step} or step != row["step"]:
            raise ValueError("run-evidence candidate alias contradicts its filename")
        scope[name] = row["sha256"]
        normalized_aliases.append(alias)
    if normalized_aliases != sorted(normalized_aliases, key=lambda item: item["name"]):
        raise ValueError("run-evidence candidate aliases are not sorted")
    artifact_path = batch._safe_file(artifact_path, "staged run candidate")
    if (
        krea_provenance.file_sha256(artifact_path) != row["sha256"]
        or artifact_path.stat().st_size != row["bytes"]
        or _safetensors_identity(artifact_path) != row["safetensors"]
    ):
        raise ValueError("staged run candidate bytes/layout differ from binding")
    return row, scope


def _load_local_binding(
    reference: Any, *, label: str
) -> tuple[Path, dict[str, Any], str]:
    path, binding, digest = batch._load_candidate_binding(reference, label)
    batch._exact_keys(
        binding,
        {
            "schema",
            "kind",
            "mode",
            "arm_id",
            "candidate_id",
            "candidate",
            "execution_plan",
            "execution_approval",
            "run_completion",
            "run_record",
            "training_log",
            "evaluation_dataset_sha256",
        },
        label,
    )
    if (
        binding["schema"] != 2
        or binding["kind"] != "forge-krea-local-candidate-binding"
        or binding["mode"] != "local_run_candidate"
    ):
        raise ValueError(f"{label} is not a stage-three local binding")
    return path, binding, digest


def validate_run_evidence(bundle_path: Path) -> dict[str, Any]:
    """Recompute one complete stage-three bundle from referenced bytes.

    Validation is intentionally directory-aware.  A candidate binding cannot
    be detached from its exhaustive bundle and presented as though it proved a
    complete checkpoint grid.
    """

    bundle_path = batch._safe_file(bundle_path, "run-evidence bundle")
    bundle, _bundle_file_sha = _load_canonical(bundle_path, "run-evidence bundle")
    batch._exact_keys(
        bundle,
        {
            "schema",
            "kind",
            "arm_id",
            "execution_plan_sha256",
            "run_completion",
            "candidate_bindings",
            "bundle_sha256",
        },
        "run-evidence bundle",
    )
    body = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    if (
        bundle["schema"] != 2
        or bundle["kind"] != "forge-krea-run-evidence-bundle"
        or bundle["bundle_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("run-evidence bundle identity is invalid")
    references = bundle["candidate_bindings"]
    if not isinstance(references, list) or not references:
        raise ValueError("run-evidence bundle contains no candidates")
    root = bundle_path.parent
    loaded: list[tuple[Path, dict[str, Any], str]] = []
    for index, raw_reference in enumerate(references):
        reference = batch._object(raw_reference, "candidate binding reference")
        batch._exact_keys(
            reference,
            {"candidate_id", "candidate_sha256", "binding"},
            "candidate binding reference",
        )
        path, binding, digest = _load_local_binding(
            reference["binding"], label=f"candidate binding {index}"
        )
        if path != root / f"candidate-{index:03d}.json":
            raise ValueError("candidate binding escaped or reordered its bundle")
        if (
            reference["binding"] != {"path": str(path), "sha256": digest}
            or reference["candidate_id"] != binding["candidate_id"]
            or reference["candidate_sha256"]
            != batch._object(binding["candidate"], "bound candidate").get("sha256")
        ):
            raise ValueError("candidate binding reference is inconsistent")
        loaded.append((path, binding, digest))

    first = loaded[0][1]
    plan_path = batch._safe_file(first["execution_plan"]["path"], "execution plan")
    approval_path = batch._safe_file(
        first["execution_approval"]["path"], "execution approval"
    )
    plan, plan_file_sha, approval, approval_file_sha, resolved = (
        load_execution_controls(plan_path, approval_path)
    )
    if (
        first["execution_plan"] != {"path": str(plan_path), "sha256": plan_file_sha}
        or first["execution_approval"]
        != {"path": str(approval_path), "sha256": approval_file_sha}
        or bundle["arm_id"] != plan["arm_id"]
        or bundle["execution_plan_sha256"] != plan["plan_sha256"]
    ):
        raise ValueError("run-evidence bundle is not bound to its approved plan")

    completion_path, completion, completion_sha = batch._load_candidate_binding(
        bundle["run_completion"], "run completion"
    )
    if completion_path != root / "run-completion.json" or bundle["run_completion"] != {
        "path": str(completion_path),
        "sha256": completion_sha,
    }:
        raise ValueError("run completion escaped or contradicts its bundle")
    batch._exact_keys(
        completion,
        {
            "schema",
            "kind",
            "arm_id",
            "task_id",
            "execution_plan_sha256",
            "execution_plan_file_sha256",
            "execution_approval_sha256",
            "execution_approval_file_sha256",
            "fixture_manifest_sha256",
            "training_dataset_sha256",
            "training_rows_sha256",
            "training_archive",
            "execution_envelope_sha256",
            "host_execution_identity_sha256",
            "throughput_profile_sha256",
            "budget_plan_sha256",
            "schedule",
            "run_record_sha256",
            "training_log_sha256",
            "natural_completion",
            "in_task_proxy_selection",
            "candidates",
        },
        "run completion",
    )
    expected_completion = {
        "schema": 3,
        "kind": "forge-krea-training-completion",
        "arm_id": plan["arm_id"],
        "task_id": plan["task_id"],
        "execution_plan_sha256": plan["plan_sha256"],
        "execution_plan_file_sha256": plan_file_sha,
        "execution_approval_sha256": approval["approval_sha256"],
        "execution_approval_file_sha256": approval_file_sha,
        "fixture_manifest_sha256": resolved["fixture"]["manifest_sha256"],
        "training_dataset_sha256": resolved["fixture"]["training_dataset_identity"][
            "sha256"
        ],
        "training_rows_sha256": krea_provenance.canonical_sha256(
            resolved["fixture"]["training_rows"]
        ),
        "training_archive": {
            "sha256": resolved["fixture"]["training_archive"]["sha256"],
            "bytes": resolved["fixture"]["training_archive"]["bytes"],
        },
        "execution_envelope_sha256": plan["execution_envelope_sha256"],
        "host_execution_identity_sha256": resolved["host_execution_manifest"][
            "host_execution_identity_sha256"
        ],
        "throughput_profile_sha256": resolved["throughput_profile"]["profile_sha256"],
        "budget_plan_sha256": plan["budget_plan_sha256"],
        "schedule": plan["schedule"],
        "natural_completion": True,
        "in_task_proxy_selection": {"enabled": False, "reserve_s": 0},
    }
    if any(completion.get(key) != value for key, value in expected_completion.items()):
        raise ValueError("run completion contradicts its approved execution")

    run_path, condition, condition_sha = batch._load_candidate_binding(
        first["run_record"], "run record"
    )
    log_path, training_log, training_log_sha = batch._load_candidate_binding(
        first["training_log"], "training log"
    )
    _validate_completed_condition(
        condition,
        plan=plan,
        plan_file_sha=plan_file_sha,
        approval=approval,
        approval_file_sha=approval_file_sha,
        resolved=resolved,
    )
    if (
        completion["run_record_sha256"] != condition_sha
        or completion["training_log_sha256"] != training_log_sha
        or training_log != condition["telemetry"]
    ):
        raise ValueError("run completion/log/record bindings are inconsistent")

    completion_rows = completion["candidates"]
    if not isinstance(completion_rows, list) or len(completion_rows) != len(loaded):
        raise ValueError("run completion candidate set is not exhaustive")
    observed_rows: list[dict[str, Any]] = []
    scope: dict[str, str] = {}
    ids: set[str] = set()
    hashes: set[str] = set()
    steps: set[int] = set()
    for index, (_path, binding, _binding_sha) in enumerate(loaded):
        common = {
            "execution_plan": {"path": str(plan_path), "sha256": plan_file_sha},
            "execution_approval": {
                "path": str(approval_path),
                "sha256": approval_file_sha,
            },
            "run_completion": {
                "path": str(completion_path),
                "sha256": completion_sha,
            },
            "run_record": {"path": str(run_path), "sha256": condition_sha},
            "training_log": {"path": str(log_path), "sha256": training_log_sha},
        }
        if (
            binding["arm_id"] != plan["arm_id"]
            or binding["evaluation_dataset_sha256"]
            != resolved["fixture"]["evaluation_dataset_identity"]["sha256"]
            or any(binding[key] != value for key, value in common.items())
        ):
            raise ValueError("candidate binding crossed its run/plan/fixture")
        candidate = batch._object(binding["candidate"], "bound candidate")
        artifact_path = root / "candidates" / f"{binding['candidate_id']}.safetensors"
        if candidate.get("path") != str(artifact_path):
            raise ValueError("candidate artifact escaped its evidence directory")
        row_input = {key: value for key, value in candidate.items() if key != "path"}
        row_input["candidate_id"] = binding["candidate_id"]
        row, row_scope = _validate_candidate_row(
            row_input, plan=plan, artifact_path=artifact_path
        )
        if binding["candidate_id"] != row["candidate_id"]:
            raise ValueError("candidate id differs between binding and artifact")
        if (
            row["candidate_id"] in ids
            or row["sha256"] in hashes
            or row["step"] in steps
            or set(row_scope) & set(scope)
        ):
            raise ValueError("run-evidence candidates are duplicated/ambiguous")
        ids.add(row["candidate_id"])
        hashes.add(row["sha256"])
        steps.add(row["step"])
        scope.update(row_scope)
        observed_rows.append(row)
        if row != completion_rows[index]:
            raise ValueError("candidate binding differs from run completion")
    if observed_rows != sorted(
        observed_rows, key=lambda row: (row["step"], row["sha256"])
    ):
        raise ValueError("run-evidence candidates are not deterministically ordered")
    if [row["step"] for row in observed_rows] != plan["schedule"]["candidate_steps"]:
        raise ValueError("run-evidence candidate grid differs from sealed schedule")
    final_name = f"{plan['expected_repo_name']}.safetensors"
    final_sha = scope.get(final_name)
    if final_sha is None:
        raise ValueError("run-evidence lacks its exact natural final")
    scope["last.safetensors"] = final_sha
    if condition.get("current_scope_candidates") != scope:
        raise ValueError("run-evidence does not exhaust the current run scope")
    artifacts = batch._object(condition.get("artifacts"), "run artifacts")
    if (
        artifacts.get("candidate_sha256")
        != {
            name: digest for name, digest in scope.items() if name != "last.safetensors"
        }
        or artifacts.get("last_sha256") != final_sha
    ):
        raise ValueError("run record artifact ledger contradicts staged evidence")
    return bundle


def _validated_template_binding(
    binding_path: Path,
) -> tuple[Path, dict[str, Any], str]:
    binding_path = batch._safe_file(binding_path, "zero-control template binding")
    bundle = validate_run_evidence(binding_path.parent / "bundle.json")
    path, binding, digest = _load_local_binding(
        {
            "path": str(binding_path),
            "sha256": krea_provenance.file_sha256(binding_path),
        },
        label="zero-control template binding",
    )
    if not any(
        row["binding"] == {"path": str(path), "sha256": digest}
        for row in bundle["candidate_bindings"]
    ):
        raise ValueError("zero-control template binding is absent from its run bundle")
    return path, binding, digest


def emit_zero_control(
    *,
    template_candidate_binding: Path,
    output_artifact: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    """Create a zero LoRA bound to one approved run's staged final template."""

    binding_path, binding, binding_file_sha = _validated_template_binding(
        template_candidate_binding
    )
    batch._exact_keys(
        binding,
        {
            "schema",
            "kind",
            "mode",
            "arm_id",
            "candidate_id",
            "candidate",
            "execution_plan",
            "execution_approval",
            "run_completion",
            "run_record",
            "training_log",
            "evaluation_dataset_sha256",
        },
        "zero-control template binding",
    )
    if (
        binding["schema"] != 2
        or binding["kind"] != "forge-krea-local-candidate-binding"
        or binding["mode"] != "local_run_candidate"
    ):
        raise ValueError("zero control requires a stage-three local candidate binding")
    candidate_identity = batch._object(binding["candidate"], "template candidate")
    batch._exact_keys(
        candidate_identity,
        {
            "path",
            "sha256",
            "bytes",
            "step",
            "fraction_numerator",
            "fraction_denominator",
            "aliases",
            "safetensors",
        },
        "template candidate",
    )
    template = batch._safe_file(candidate_identity["path"], "zero-control template")
    if (
        krea_provenance.file_sha256(template) != candidate_identity["sha256"]
        or template.stat().st_size != candidate_identity["bytes"]
        or _safetensors_identity(template) != candidate_identity["safetensors"]
    ):
        raise ValueError("zero-control template differs from its staged binding")
    plan_path, plan, plan_file_sha = batch._load_candidate_binding(
        binding["execution_plan"], "zero-control execution plan"
    )
    resolved = krea_execution_plan.validate_plan(plan)
    if (
        plan["arm_id"] != binding["arm_id"]
        or binding["evaluation_dataset_sha256"]
        != resolved["fixture"]["evaluation_dataset_identity"]["sha256"]
        or candidate_identity["step"] != plan["schedule"]["planned_steps"]
        or not any(
            alias.get("role") == "exact_final"
            for alias in candidate_identity["aliases"]
        )
    ):
        raise ValueError(
            "zero-control template is not the exact final of its execution fixture"
        )
    base_model = plan["base_model"]
    evaluation_dataset_sha256 = binding["evaluation_dataset_sha256"]
    output_artifact = Path(os.path.abspath(os.path.expanduser(output_artifact)))
    output_manifest = Path(os.path.abspath(os.path.expanduser(output_manifest)))
    if (
        output_artifact == output_manifest
        or output_artifact
        in {
            template,
            binding_path,
            plan_path,
        }
        or output_manifest in {template, binding_path, plan_path}
    ):
        raise ValueError("zero-control inputs and outputs must be distinct")
    for output in (output_artifact, output_manifest):
        batch._reject_symlink_ancestors(output.parent, "zero-control output parent")
        if os.path.lexists(output):
            raise FileExistsError(f"refusing existing zero-control output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        batch._reject_symlink_ancestors(output.parent, "zero-control output parent")
    _digest(evaluation_dataset_sha256, "evaluation dataset")
    base_model = batch._object(base_model, "zero-control base model")
    batch._exact_keys(
        base_model,
        {"model_id", "revision", "training_identity_sha256", "evaluation_assets"},
        "zero-control base model",
    )
    if (
        base_model["model_id"] != "krea/Krea-2-Raw"
        or not isinstance(base_model["revision"], str)
        or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", base_model["revision"])
        or _digest(base_model["training_identity_sha256"], "training base identity")
        != base_model["training_identity_sha256"]
    ):
        raise ValueError("zero control requires the Krea base model")
    assets = batch._object(
        base_model["evaluation_assets"], "zero-control evaluation assets"
    )
    batch._exact_keys(
        assets,
        {"diffusion_model", "text_encoder", "vae"},
        "zero-control evaluation assets",
    )
    for name, raw_asset in assets.items():
        asset = batch._object(raw_asset, f"zero-control asset {name}")
        batch._exact_keys(
            asset,
            {"canonical_path", "sha256", "bytes"},
            f"zero-control asset {name}",
        )
        if (
            not isinstance(asset["canonical_path"], str)
            or not asset["canonical_path"].startswith("/")
            or _digest(asset["sha256"], f"zero-control asset {name}") != asset["sha256"]
            or isinstance(asset["bytes"], bool)
            or not isinstance(asset["bytes"], int)
            or asset["bytes"] <= 0
        ):
            raise ValueError(f"zero-control asset {name} identity is invalid")
    source_header, source_data = _read_safetensors(template)
    layout = _tensor_layout(source_header)
    if max(source_header[row["key"]]["data_offsets"][1] for row in layout) != len(
        source_data
    ):
        raise ValueError("template safetensors data region is inconsistent")
    generator_sha = krea_provenance.file_sha256(Path(__file__).resolve(strict=True))
    offset = 0
    header: dict[str, Any] = {
        "__metadata__": {
            "forge_control": "deterministic-zero-lora-v1",
            "generator_sha256": generator_sha,
            "template_sha256": krea_provenance.file_sha256(template),
            "template_binding_sha256": binding_file_sha,
            "execution_plan_sha256": plan["plan_sha256"],
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
        remaining = offset
        zero_block = b"\0" * min(8 * 1024 * 1024, max(1, offset))
        while remaining:
            block = zero_block[: min(remaining, len(zero_block))]
            handle.write(block)
            remaining -= len(block)
        handle.flush()
        os.fsync(handle.fileno())
    artifact_parent_fd = os.open(
        output_artifact.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(artifact_parent_fd)
    finally:
        os.close(artifact_parent_fd)
    artifact_sha = krea_provenance.file_sha256(output_artifact)
    body = {
        "schema": 2,
        "kind": "forge-krea-zero-lora-control",
        "mode": "zero_lora_control",
        "artifact": {
            "path": str(output_artifact),
            "sha256": artifact_sha,
            "bytes": output_artifact.stat().st_size,
        },
        "template": {
            "path": str(template),
            "sha256": krea_provenance.file_sha256(template),
            "bytes": template.stat().st_size,
            "safetensors": candidate_identity["safetensors"],
        },
        "template_candidate_binding": {
            "path": str(binding_path),
            "sha256": binding_file_sha,
        },
        "execution_plan": {
            "path": str(plan_path),
            "sha256": plan_file_sha,
            "plan_sha256": plan["plan_sha256"],
        },
        "run_completion": binding["run_completion"],
        "generator_sha256": generator_sha,
        "tensor_layout": layout,
        "tensor_layout_sha256": krea_provenance.canonical_sha256(layout),
        "all_tensor_bytes_zero": True,
        "base_model": base_model,
        "evaluation_dataset_sha256": evaluation_dataset_sha256,
    }
    manifest = {**body, "manifest_sha256": krea_provenance.canonical_sha256(body)}
    _canonical_file(output_manifest, manifest)
    manifest_parent_fd = os.open(
        output_manifest.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(manifest_parent_fd)
    finally:
        os.close(manifest_parent_fd)
    validate_zero_control(manifest, artifact_path=output_artifact)
    return manifest


def validate_zero_control(
    manifest: dict[str, Any], *, artifact_path: Path | None = None
) -> dict[str, Any]:
    manifest = batch._object(manifest, "zero-control manifest")
    required = {
        "schema",
        "kind",
        "mode",
        "artifact",
        "template",
        "template_candidate_binding",
        "execution_plan",
        "run_completion",
        "generator_sha256",
        "tensor_layout",
        "tensor_layout_sha256",
        "all_tensor_bytes_zero",
        "base_model",
        "evaluation_dataset_sha256",
        "manifest_sha256",
    }
    batch._exact_keys(manifest, required, "zero-control manifest")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest["schema"] != 2
        or manifest["kind"] != "forge-krea-zero-lora-control"
        or manifest["mode"] != "zero_lora_control"
        or manifest["all_tensor_bytes_zero"] is not True
        or manifest["manifest_sha256"] != krea_provenance.canonical_sha256(body)
        or manifest["tensor_layout_sha256"]
        != krea_provenance.canonical_sha256(manifest["tensor_layout"])
        or manifest["generator_sha256"]
        != krea_provenance.file_sha256(Path(__file__).resolve(strict=True))
    ):
        raise ValueError("zero-control manifest is invalid")
    base = batch._object(manifest["base_model"], "zero-control base model")
    batch._exact_keys(
        base,
        {"model_id", "revision", "training_identity_sha256", "evaluation_assets"},
        "zero-control base model",
    )
    if (
        base["model_id"] != "krea/Krea-2-Raw"
        or not isinstance(base["revision"], str)
        or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", base["revision"])
        or not isinstance(base["training_identity_sha256"], str)
        or not _SHA256.fullmatch(base["training_identity_sha256"])
    ):
        raise ValueError("zero-control base model is not Krea")
    assets = batch._object(base["evaluation_assets"], "zero-control assets")
    batch._exact_keys(
        assets,
        {"diffusion_model", "text_encoder", "vae"},
        "zero-control assets",
    )
    for name, raw_asset in assets.items():
        asset = batch._object(raw_asset, f"zero-control asset {name}")
        batch._exact_keys(
            asset,
            {"canonical_path", "sha256", "bytes"},
            f"zero-control asset {name}",
        )
        if (
            not isinstance(asset["canonical_path"], str)
            or not asset["canonical_path"].startswith("/")
            or not isinstance(asset["sha256"], str)
            or not _SHA256.fullmatch(asset["sha256"])
            or isinstance(asset["bytes"], bool)
            or not isinstance(asset["bytes"], int)
            or asset["bytes"] <= 0
        ):
            raise ValueError(f"zero-control asset {name} identity is invalid")
    _digest(manifest["evaluation_dataset_sha256"], "evaluation dataset")
    binding_reference = batch._object(
        manifest["template_candidate_binding"], "zero-control template binding"
    )
    batch._exact_keys(
        binding_reference, {"path", "sha256"}, "zero-control template binding"
    )
    binding_path, binding, binding_sha = _validated_template_binding(
        Path(binding_reference["path"])
    )
    if binding_reference != {"path": str(binding_path), "sha256": binding_sha}:
        raise ValueError("zero-control template binding SHA mismatch")
    plan_reference = batch._object(
        manifest["execution_plan"], "zero-control execution plan"
    )
    batch._exact_keys(
        plan_reference,
        {"path", "sha256", "plan_sha256"},
        "zero-control execution plan",
    )
    plan_path, plan, plan_file_sha = batch._load_candidate_binding(
        {"path": plan_reference["path"], "sha256": plan_reference["sha256"]},
        "zero-control execution plan",
    )
    resolved = krea_execution_plan.validate_plan(plan)
    if (
        manifest["execution_plan"]
        != {
            "path": str(plan_path),
            "sha256": plan_file_sha,
            "plan_sha256": plan["plan_sha256"],
        }
        or binding.get("execution_plan")
        != {"path": str(plan_path), "sha256": plan_file_sha}
        or binding.get("run_completion") != manifest["run_completion"]
        or binding.get("evaluation_dataset_sha256")
        != manifest["evaluation_dataset_sha256"]
        or resolved["fixture"]["evaluation_dataset_identity"]["sha256"]
        != manifest["evaluation_dataset_sha256"]
        or plan["base_model"] != base
    ):
        raise ValueError("zero control is not bound to the approved run/base/fixture")
    template = batch._object(manifest["template"], "zero-control template")
    batch._exact_keys(
        template,
        {"path", "sha256", "bytes", "safetensors"},
        "zero-control template",
    )
    template_path = batch._safe_file(template["path"], "zero-control template")
    bound_candidate = batch._object(
        binding.get("candidate"), "bound template candidate"
    )
    if (
        bound_candidate.get("step") != plan["schedule"]["planned_steps"]
        or not isinstance(bound_candidate.get("aliases"), list)
        or not any(
            isinstance(alias, dict) and alias.get("role") == "exact_final"
            for alias in bound_candidate["aliases"]
        )
        or template.get("path") != bound_candidate.get("path")
        or template.get("sha256") != bound_candidate.get("sha256")
        or template.get("bytes") != bound_candidate.get("bytes")
        or template.get("safetensors") != bound_candidate.get("safetensors")
        or krea_provenance.file_sha256(template_path) != template.get("sha256")
        or template_path.stat().st_size != template.get("bytes")
        or _safetensors_identity(template_path) != template.get("safetensors")
    ):
        raise ValueError("zero-control template bytes differ from the approved run")
    artifact = batch._object(manifest["artifact"], "zero-control artifact")
    batch._exact_keys(artifact, {"path", "sha256", "bytes"}, "zero-control artifact")
    if (
        not isinstance(artifact["path"], str)
        or not Path(artifact["path"]).is_absolute()
        or not isinstance(artifact["sha256"], str)
        or not _SHA256.fullmatch(artifact["sha256"])
        or isinstance(artifact["bytes"], bool)
        or not isinstance(artifact["bytes"], int)
        or artifact["bytes"] <= 0
    ):
        raise ValueError("zero-control artifact identity is invalid")
    declared_artifact_path = Path(os.path.abspath(os.path.expanduser(artifact["path"])))
    if artifact_path is None:
        artifact_path = declared_artifact_path
    else:
        artifact_path = Path(os.path.abspath(os.path.expanduser(artifact_path)))
        if artifact_path != declared_artifact_path:
            raise ValueError("zero-control artifact path differs from its manifest")
    artifact_path = batch._safe_file(artifact_path, "zero-control artifact")
    if krea_provenance.file_sha256(artifact_path) != artifact.get(
        "sha256"
    ) or artifact_path.stat().st_size != artifact.get("bytes"):
        raise ValueError("zero-control artifact binding mismatch")
    header, data = _read_safetensors(artifact_path)
    layout = _tensor_layout(header)
    metadata = header.get("__metadata__")
    if (
        layout != manifest["tensor_layout"]
        or any(data)
        or not isinstance(metadata, dict)
        or metadata.get("forge_control") != "deterministic-zero-lora-v1"
        or metadata.get("generator_sha256") != manifest["generator_sha256"]
        or metadata.get("template_sha256") != template["sha256"]
        or metadata.get("template_binding_sha256") != binding_sha
        or metadata.get("execution_plan_sha256") != plan["plan_sha256"]
        or max(header[row["key"]]["data_offsets"][1] for row in layout) != len(data)
    ):
        raise ValueError("zero-control tensors are not exactly all zero")
    return manifest


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run-evidence",
        help="emit exhaustive candidate evidence from one approved natural run",
    )
    run.add_argument("--condition-record", required=True, type=Path)
    run.add_argument("--execution-plan", required=True, type=Path)
    run.add_argument("--execution-approval", required=True, type=Path)
    run.add_argument("--training-dir", required=True, type=Path)
    run.add_argument("--candidate", required=True, action="append", type=Path)
    run.add_argument("--output-dir", required=True, type=Path)

    validate_run = commands.add_parser(
        "validate-run-evidence",
        help="recompute a published run bundle, all bindings, and all candidates",
    )
    validate_run.add_argument("--bundle", required=True, type=Path)

    zero = commands.add_parser(
        "zero-control",
        help="emit a deterministic zero LoRA from an exhaustive run's exact final",
    )
    zero.add_argument("--template-candidate-binding", required=True, type=Path)
    zero.add_argument("--output-artifact", required=True, type=Path)
    zero.add_argument("--output-manifest", required=True, type=Path)

    validate_zero = commands.add_parser(
        "validate-zero-control",
        help="recompute a zero-control manifest and its referenced run evidence",
    )
    validate_zero.add_argument("--manifest", required=True, type=Path)
    validate_zero.add_argument("--artifact", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    try:
        if args.command == "run-evidence":
            result = emit_run_evidence(
                condition_record_path=args.condition_record,
                execution_plan_path=args.execution_plan,
                execution_approval_path=args.execution_approval,
                candidate_paths=args.candidate,
                training_dir=args.training_dir,
                output_dir=args.output_dir,
            )
            summary = {
                "status": "emitted",
                "kind": result["kind"],
                "output": str(
                    Path(os.path.abspath(os.path.expanduser(args.output_dir)))
                    / "bundle.json"
                ),
                "bundle_sha256": result["bundle_sha256"],
                "candidate_count": len(result["candidate_bindings"]),
            }
        elif args.command == "validate-run-evidence":
            result = validate_run_evidence(args.bundle)
            summary = {
                "status": "valid",
                "kind": result["kind"],
                "bundle_sha256": result["bundle_sha256"],
                "candidate_count": len(result["candidate_bindings"]),
            }
        elif args.command == "zero-control":
            result = emit_zero_control(
                template_candidate_binding=args.template_candidate_binding,
                output_artifact=args.output_artifact,
                output_manifest=args.output_manifest,
            )
            summary = {
                "status": "emitted",
                "kind": result["kind"],
                "manifest_sha256": result["manifest_sha256"],
                "artifact_sha256": result["artifact"]["sha256"],
            }
        elif args.command == "validate-zero-control":
            manifest, _manifest_file_sha = _load_canonical(
                args.manifest, "zero-control manifest"
            )
            result = validate_zero_control(manifest, artifact_path=args.artifact)
            summary = {
                "status": "valid",
                "kind": result["kind"],
                "manifest_sha256": result["manifest_sha256"],
                "artifact_sha256": result["artifact"]["sha256"],
            }
        else:  # pragma: no cover - argparse owns the command vocabulary.
            raise RuntimeError(f"unsupported command: {args.command}")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
