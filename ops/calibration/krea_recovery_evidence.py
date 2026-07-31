#!/usr/bin/env python3
"""Build a fail-closed index over Week-5 D1/D2 recovery score evidence.

Recovery receipts are scheduling hints, not proof that a score is usable.  This
module records every row (including incomplete/failed rows and its available
logs), but marks a row ``selection_eligible`` only after independently checking
the candidate bytes, rc0 status, exact evaluator result, and evidence manifest.

The resulting index is deliberately non-authorizing.  It never reads C1-C4,
does not claim that the original discovery execution was replayed, and grants
no reveal, GPU, production, release, or deployment authority.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

try:
    from . import krea_provenance
except ImportError:  # pragma: no cover - direct script execution.
    import krea_provenance  # type: ignore[no-redef]


INDEX_KIND = "forge-krea-recovery-exact-score-index"
INDEX_SCHEMA = 1
EXPECTED_EXACT_SCORES = 92
EXPECTED_CANDIDATE_SCORES = 90
EXPECTED_ZERO_SCORES = 2
FIXTURES = ("D1", "D2")
FAMILIES = ("K0", "K1", "K2", "K3", "K4", "K5")
FIXTURE_ROWS = {"D1": 24, "D2": 40}

FALSE_CLAIMS: dict[str, bool] = {
    "strict_discovery_replayed": False,
    "retroactive_plan_or_approval_claimed": False,
    "c1c4_accessed": False,
    "reveal_authority": False,
    "stage2_gpu_execution_authorized": False,
    "production_mutation_authorized": False,
    "release_authorized": False,
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TASK = re.compile(
    r"(?P<fixture>d[12])-(?P<family>k[0-5])-(?P<kind>step|final)(?P<step>[1-9][0-9]*)"
)
_ZERO_TASK = re.compile(r"(?P<fixture>d[12])-zero-baseline")
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_LEDGER_HEADER = (
    "task_id",
    "fixture",
    "cell",
    "label",
    "coverage_tier",
    "expected_candidate",
    "candidate_sha256",
    "state",
)
_RESULT_KEYS = {
    "schema",
    "evaluator",
    "candidate",
    "candidate_sha256",
    "candidate_bytes",
    "staged_candidate_sha256",
    "comfy_lora_name",
    "model_type",
    "dataset",
    "dataset_sha256",
    "image_count",
    "scored_rows",
    "base_name",
    "asset_sha256",
    "asset_bytes",
    "steps",
    "cfg",
    "denoise",
    "generations",
    "master_seed",
    "seeds",
    "text_guided_losses",
    "blank_prompt_losses",
    "text_mean",
    "blank_mean",
    "text_weight",
    "weighted_loss",
    "direction",
    "elapsed_s",
    "source",
    "runtime",
}
_STATUS_KEYS = {
    "started_utc",
    "started_unix_ns",
    "fixture",
    "candidate_sha256",
    "ended_utc",
    "ended_unix_ns",
    "returncode",
}
_EVIDENCE_FILES = {
    "comfy.log",
    "evaluator.stderr",
    "evaluator.stdout",
    "exact-score.json",
    "gpu-telemetry.csv",
    "inputs.sha256",
    "resource-usage.txt",
    "run-status.env",
}
_LOG_NAMES = (
    "comfy.log",
    "evaluator.stderr",
    "evaluator.stdout",
    "gpu-telemetry.csv",
    "resource-usage.txt",
    "run-status.env",
)
_SEALED_PARTS = {"c1", "c2", "c3", "c4"}


class RecoveryEvidenceError(ValueError):
    """Raised when recovery evidence cannot be safely interpreted."""


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RecoveryEvidenceError(f"JSON contains duplicate key: {key}")
        value[key] = item
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise RecoveryEvidenceError(
            f"{label} keys mismatch: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RecoveryEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise RecoveryEvidenceError(f"{label} must be canonical UTC")
    datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return value


def _number(
    value: Any,
    label: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecoveryEvidenceError(f"{label} must be numeric")
    result = float(value)
    if (
        not math.isfinite(result)
        or result < minimum
        or (maximum is not None and result > maximum)
    ):
        raise RecoveryEvidenceError(f"{label} is outside its finite range")
    return result


def _reject_sealed(path: Path | str, label: str) -> None:
    parts = {part.casefold() for part in Path(str(path)).parts}
    if parts & _SEALED_PARTS:
        raise RecoveryEvidenceError(f"{label} points at prohibited sealed C evidence")


def _safe_file(path: Path, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    _reject_sealed(path, label)
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise RecoveryEvidenceError(f"{label} has a symlink component: {current}")
        current = current.parent
    if not path.is_file():
        raise RecoveryEvidenceError(f"{label} is not a regular file: {path}")
    return path


def _safe_directory(path: Path, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    _reject_sealed(path, label)
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise RecoveryEvidenceError(f"{label} has a symlink component: {current}")
        current = current.parent
    if not path.is_dir():
        raise RecoveryEvidenceError(f"{label} is not a real directory: {path}")
    return path


def _file_binding(path: Path, label: str) -> dict[str, Any]:
    path = _safe_file(path, label)
    before = path.stat()
    sha256 = krea_provenance.file_sha256(path)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RecoveryEvidenceError(f"{label} changed while it was hashed")
    return {"path": str(path), "bytes": after.st_size, "file_sha256": sha256}


def _load_json(
    path: Path, label: str, *, canonical: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _file_binding(path, label)
    raw = Path(binding["path"]).read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryEvidenceError(f"{label} is not duplicate-safe JSON") from exc
    if not isinstance(value, dict):
        raise RecoveryEvidenceError(f"{label} must be a JSON object")
    if canonical and raw != krea_provenance.canonical_bytes(value) + b"\n":
        raise RecoveryEvidenceError(f"{label} must be canonical JSON plus one newline")
    return value, binding


def _load_env(path: Path, label: str) -> tuple[dict[str, str], dict[str, Any]]:
    binding = _file_binding(path, label)
    raw = Path(binding["path"]).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecoveryEvidenceError(f"{label} is not UTF-8") from exc
    if not text.endswith("\n"):
        raise RecoveryEvidenceError(f"{label} lacks a terminal newline")
    value: dict[str, str] = {}
    for line in text.splitlines():
        if not line or "=" not in line:
            raise RecoveryEvidenceError(f"{label} contains a malformed row")
        key, item = line.split("=", 1)
        if not key or key in value or "\x00" in item:
            raise RecoveryEvidenceError(f"{label} contains duplicate/unsafe fields")
        value[key] = item
    return value, binding


def _parse_manifest(
    path: Path, label: str
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    binding = _file_binding(path, label)
    raw = Path(binding["path"]).read_bytes()
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RecoveryEvidenceError(f"{label} is not UTF-8") from exc
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (/.+)", line)
        if match is None:
            raise RecoveryEvidenceError(f"{label} contains a malformed checksum row")
        name = Path(match.group(2)).name
        if name in seen:
            raise RecoveryEvidenceError(f"{label} repeats basename {name}")
        seen.add(name)
        rows.append((match.group(1), name))
    if not rows or not raw.endswith(b"\n"):
        raise RecoveryEvidenceError(f"{label} is empty or lacks a terminal newline")
    return rows, binding


def _parse_task(task_id: str) -> dict[str, Any]:
    match = _TASK.fullmatch(task_id)
    if match is not None:
        return {
            "fixture": match.group("fixture").upper(),
            "family": match.group("family").upper(),
            "step": int(match.group("step")),
            "is_final": match.group("kind") == "final",
            "zero_control": False,
        }
    match = _ZERO_TASK.fullmatch(task_id)
    if match is not None:
        return {
            "fixture": match.group("fixture").upper(),
            "family": "ZERO",
            "step": 0,
            "is_final": True,
            "zero_control": True,
        }
    raise RecoveryEvidenceError(f"unsupported recovery task id: {task_id}")


def _load_ledger(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    binding = _file_binding(path, "recovery coverage ledger")
    raw = Path(binding["path"]).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecoveryEvidenceError("recovery coverage ledger is not UTF-8") from exc
    if not text.endswith("\n"):
        raise RecoveryEvidenceError("recovery coverage ledger lacks a terminal newline")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if tuple(reader.fieldnames or ()) != _LEDGER_HEADER:
        raise RecoveryEvidenceError("recovery coverage ledger header drifted")
    rows = [dict(row) for row in reader]
    if len(rows) != EXPECTED_EXACT_SCORES:
        raise RecoveryEvidenceError(
            f"recovery coverage must contain exactly {EXPECTED_EXACT_SCORES} rows"
        )
    task_ids = [row["task_id"] for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise RecoveryEvidenceError("recovery coverage repeats task ids")

    candidates = 0
    zeros = 0
    per_fixture = {fixture: 0 for fixture in FIXTURES}
    per_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        parsed = _parse_task(row["task_id"])
        fixture = parsed["fixture"]
        if row["fixture"] != fixture:
            raise RecoveryEvidenceError(f"{row['task_id']} fixture binding drifted")
        if parsed["zero_control"]:
            zeros += 1
            if row["cell"] != "ALL" or row["label"] != "zero-baseline":
                raise RecoveryEvidenceError(f"{row['task_id']} zero binding drifted")
        else:
            candidates += 1
            expected_cell = f"{fixture}-{parsed['family']}"
            expected_label = (
                f"final-{parsed['step']}"
                if parsed["is_final"]
                else f"step-{parsed['step']}"
            )
            if row["cell"] != expected_cell or row["label"] != expected_label:
                raise RecoveryEvidenceError(
                    f"{row['task_id']} cell/label binding drifted"
                )
            per_cell.setdefault((fixture, parsed["family"]), []).append(parsed)
        if row["candidate_sha256"] != "-":
            _digest(row["candidate_sha256"], f"{row['task_id']} candidate SHA")
        if not row["expected_candidate"] or not row["state"]:
            raise RecoveryEvidenceError(f"{row['task_id']} has empty coverage fields")
        _reject_sealed(row["expected_candidate"], f"{row['task_id']} candidate")
        per_fixture[fixture] += 1

    if (
        candidates != EXPECTED_CANDIDATE_SCORES
        or zeros != EXPECTED_ZERO_SCORES
        or per_fixture != {"D1": 46, "D2": 46}
        or set(per_cell) != {(d, k) for d in FIXTURES for k in FAMILIES}
    ):
        raise RecoveryEvidenceError(
            "recovery coverage does not encode the 90+2 contract"
        )
    for key, parsed_rows in per_cell.items():
        finals = [row for row in parsed_rows if row["is_final"]]
        steps = [row["step"] for row in parsed_rows]
        if (
            len(finals) != 1
            or finals[0]["step"] != max(steps)
            or len(steps) != len(set(steps))
        ):
            raise RecoveryEvidenceError(
                f"coverage cell {key} lacks one unique terminal final"
            )
    return rows, binding


def _log_bindings(output: Path) -> list[dict[str, Any]]:
    if not output.exists():
        return []
    output = _safe_directory(output, "recovery output directory")
    result = []
    for name in _LOG_NAMES:
        path = output / name
        if path.exists():
            binding = _file_binding(path, f"recovery log {name}")
            result.append({"name": name, **binding})
    return result


def _validate_status(
    output: Path, *, fixture: str, candidate_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    value, binding = _load_env(output / "run-status.env", "run status")
    _exact(value, _STATUS_KEYS, "run status")
    if (
        value["fixture"] != fixture
        or value["candidate_sha256"] != candidate_sha256
        or value["returncode"] != "0"
        or not value["started_unix_ns"].isdigit()
        or not value["ended_unix_ns"].isdigit()
        or int(value["ended_unix_ns"]) <= int(value["started_unix_ns"])
    ):
        raise RecoveryEvidenceError("run status is not a matching completed rc0 run")
    started = _timestamp(value["started_utc"], "run started_utc")
    ended = _timestamp(value["ended_utc"], "run ended_utc")
    if datetime.strptime(ended, "%Y-%m-%dT%H:%M:%SZ") <= datetime.strptime(
        started, "%Y-%m-%dT%H:%M:%SZ"
    ):
        raise RecoveryEvidenceError("run status UTC chronology is invalid")
    return {
        "returncode": 0,
        "started_utc": started,
        "ended_utc": ended,
        **binding,
    }, binding


def _validate_result(
    output: Path,
    *,
    fixture: str,
    candidate_path: Path,
    candidate_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value, binding = _load_json(
        output / "exact-score.json", "exact-score result", canonical=False
    )
    _exact(value, _RESULT_KEYS, "exact-score result")
    candidate_binding = _file_binding(candidate_path, "recovery candidate")
    if (
        value["schema"] != 2
        or value["evaluator"] != "god_krea2_img2img_exact"
        or value["model_type"] != "krea2"
        or value["direction"] != "min"
        or value["candidate"] != candidate_path.name
        or value["candidate_sha256"] != candidate_sha256
        or value["staged_candidate_sha256"] != candidate_sha256
        or candidate_binding["file_sha256"] != candidate_sha256
        or value["candidate_bytes"] != candidate_binding["bytes"]
        or value["comfy_lora_name"] != f"candidate-{candidate_sha256}.safetensors"
    ):
        raise RecoveryEvidenceError("exact-score candidate/evaluator identity drifted")
    _reject_sealed(value["dataset"], "exact-score dataset")
    if fixture.casefold() not in {
        part.casefold() for part in Path(value["dataset"]).parts
    }:
        raise RecoveryEvidenceError("exact-score dataset is not bound to its D fixture")
    dataset_sha = _digest(value["dataset_sha256"], "exact-score dataset SHA")
    expected_rows = FIXTURE_ROWS[fixture]
    rows = value["scored_rows"]
    text_losses = value["text_guided_losses"]
    blank_losses = value["blank_prompt_losses"]
    if (
        not isinstance(rows, list)
        or not isinstance(text_losses, list)
        or not isinstance(blank_losses, list)
        or len(rows) != expected_rows
        or value["image_count"] != expected_rows
        or len(text_losses) != expected_rows
        or len(blank_losses) != expected_rows
    ):
        raise RecoveryEvidenceError("exact-score row coverage is incomplete")
    normalized_text = [
        _number(item, f"text loss {index}", maximum=1.0)
        for index, item in enumerate(text_losses)
    ]
    normalized_blank = [
        _number(item, f"blank loss {index}", maximum=1.0)
        for index, item in enumerate(blank_losses)
    ]
    identities: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RecoveryEvidenceError(f"scored row {index} is not an object")
        for key in (
            "index",
            "image",
            "image_sha256",
            "prompt",
            "prompt_sha256",
            "text_guided_loss",
            "blank_prompt_loss",
        ):
            if key not in row:
                raise RecoveryEvidenceError(f"scored row {index} lacks {key}")
        identity = (row["image_sha256"], row["prompt_sha256"])
        _digest(identity[0], f"scored row {index} image SHA")
        _digest(identity[1], f"scored row {index} prompt SHA")
        if row["index"] != index or identity in identities:
            raise RecoveryEvidenceError(
                "scored rows are reordered or duplicate identities"
            )
        identities.add(identity)
        if (
            _number(row["text_guided_loss"], f"row {index} text loss", maximum=1.0)
            != normalized_text[index]
            or _number(row["blank_prompt_loss"], f"row {index} blank loss", maximum=1.0)
            != normalized_blank[index]
        ):
            raise RecoveryEvidenceError(
                "scored row losses differ from aggregate arrays"
            )
    text_mean = sum(normalized_text) / expected_rows
    blank_mean = sum(normalized_blank) / expected_rows
    text_weight = _number(value["text_weight"], "exact-score text weight", maximum=1.0)
    weighted_loss = text_weight * text_mean + (1.0 - text_weight) * blank_mean
    if (
        abs(
            _number(value["text_mean"], "exact-score text mean", maximum=1.0)
            - text_mean
        )
        > 1e-12
        or abs(
            _number(value["blank_mean"], "exact-score blank mean", maximum=1.0)
            - blank_mean
        )
        > 1e-12
        or abs(
            _number(
                value["weighted_loss"],
                "exact-score weighted loss",
                maximum=1.0,
            )
            - weighted_loss
        )
        > 1e-12
    ):
        raise RecoveryEvidenceError("exact-score aggregate losses do not recompute")
    if (
        isinstance(value["generations"], bool)
        or not isinstance(value["generations"], int)
        or value["generations"] <= 0
        or not isinstance(value["seeds"], list)
        or len(value["seeds"]) != value["generations"]
        or any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= 2**32 - 1
            for seed in value["seeds"]
        )
    ):
        raise RecoveryEvidenceError("exact-score seed schedule is invalid")
    for key in ("steps", "generations", "master_seed"):
        if isinstance(value[key], bool) or not isinstance(value[key], int):
            raise RecoveryEvidenceError(f"exact-score {key} is not an integer")
    if value["steps"] <= 0 or value["master_seed"] < 0:
        raise RecoveryEvidenceError("exact-score integer settings are invalid")
    _number(value["cfg"], "exact-score cfg")
    _number(value["denoise"], "exact-score denoise", maximum=1.0)
    _number(value["elapsed_s"], "exact-score elapsed_s")
    if not isinstance(value["base_name"], str) or not value["base_name"]:
        raise RecoveryEvidenceError("exact-score base model name is invalid")
    assets = value["asset_sha256"]
    asset_bytes = value["asset_bytes"]
    if (
        not isinstance(assets, dict)
        or not assets
        or not isinstance(asset_bytes, dict)
        or set(assets) != set(asset_bytes)
    ):
        raise RecoveryEvidenceError("exact-score asset identities are invalid")
    for name, digest in assets.items():
        _digest(digest, f"exact-score asset {name} SHA")
        size = asset_bytes[name]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise RecoveryEvidenceError(f"exact-score asset {name} size is invalid")
    source = value["source"]
    if not isinstance(source, dict):
        raise RecoveryEvidenceError("exact-score source is not an object")
    _exact(
        source,
        {
            "god",
            "comfyui",
            "tooling_nodes",
            "expected_commits",
            "god_import_bindings",
            "workflow_path",
            "workflow_sha256",
            "calibration_shim_sha256",
            "comfy_main_sha256",
        },
        "exact-score source",
    )
    commits: dict[str, str] = {}
    for name in ("god", "comfyui", "tooling_nodes"):
        repository = source[name]
        if not isinstance(repository, dict):
            raise RecoveryEvidenceError(f"exact-score source {name} is not an object")
        _exact(
            repository,
            {"commit", "tree", "tracked_worktree_clean", "nonignored_worktree_clean"},
            f"exact-score source {name}",
        )
        if (
            not isinstance(repository["commit"], str)
            or re.fullmatch(r"[0-9a-f]{40}", repository["commit"]) is None
            or not isinstance(repository["tree"], str)
            or re.fullmatch(r"[0-9a-f]{40}", repository["tree"]) is None
            or repository["tracked_worktree_clean"] is not True
            or repository["nonignored_worktree_clean"] is not True
        ):
            raise RecoveryEvidenceError(f"exact-score source {name} is not clean/bound")
        commits[name] = repository["commit"]
    if source["expected_commits"] != commits:
        raise RecoveryEvidenceError("exact-score expected source commits drifted")
    for key in ("workflow_sha256", "calibration_shim_sha256", "comfy_main_sha256"):
        _digest(source[key], f"exact-score source {key}")
    workflow_path = source["workflow_path"]
    if (
        not isinstance(workflow_path, str)
        or not workflow_path
        or Path(workflow_path).is_absolute()
        or any(part in {"", ".", ".."} for part in Path(workflow_path).parts)
    ):
        raise RecoveryEvidenceError("exact-score workflow path is not relative")
    imports = source["god_import_bindings"]
    if not isinstance(imports, dict) or not imports:
        raise RecoveryEvidenceError("exact-score source import bindings are empty")
    for name, binding_row in imports.items():
        if not isinstance(name, str) or not isinstance(binding_row, dict):
            raise RecoveryEvidenceError("exact-score import binding is malformed")
        _exact(binding_row, {"module", "path", "sha256"}, "exact-score import binding")
        if binding_row["module"] != name or not isinstance(binding_row["path"], str):
            raise RecoveryEvidenceError("exact-score import binding identity drifted")
        _digest(binding_row["sha256"], "exact-score import binding SHA")

    runtime = value["runtime"]
    if not isinstance(runtime, dict):
        raise RecoveryEvidenceError("exact-score runtime is not an object")
    for key, expected in (
        ("fresh_comfy_process", True),
        ("loopback", "127.0.0.1"),
        ("database", "memory"),
        ("api_nodes_disabled", True),
        ("isolated_input_output_temp_user", True),
        ("offline_environment", True),
    ):
        if runtime.get(key) != expected:
            raise RecoveryEvidenceError(
                f"exact-score runtime safety field {key} drifted"
            )
    if runtime.get("custom_node_allowlist") != ["comfyui-tooling-nodes"]:
        raise RecoveryEvidenceError("exact-score custom-node allowlist drifted")
    history = runtime.get("comfy_history")
    prompt_count = expected_rows * value["generations"]
    comfy_log = _safe_file(output / "comfy.log", "exact-score Comfy log")
    if (
        not isinstance(history, dict)
        or history.get("prompt_count") != prompt_count
        or runtime.get("comfy_log_sha256") != krea_provenance.file_sha256(comfy_log)
        or runtime.get("comfy_log_bytes") != comfy_log.stat().st_size
    ):
        raise RecoveryEvidenceError("exact-score prompt count is invalid")
    row_identities = [
        {
            key: item
            for key, item in row.items()
            if key not in {"text_guided_loss", "blank_prompt_loss"}
        }
        for row in rows
    ]
    evaluator_identity = {
        "evaluator": value["evaluator"],
        "model_type": value["model_type"],
        "base_name": value["base_name"],
        "asset_sha256": value["asset_sha256"],
        "asset_bytes": value["asset_bytes"],
        "steps": value["steps"],
        "cfg": value["cfg"],
        "denoise": value["denoise"],
        "generations": value["generations"],
        "master_seed": value["master_seed"],
        "seeds": value["seeds"],
        "text_weight": value["text_weight"],
        "source": source,
    }
    return {
        **binding,
        "semantic_sha256": krea_provenance.canonical_sha256(value),
        "candidate": candidate_binding,
        "dataset_sha256": dataset_sha,
        "image_count": expected_rows,
        "prompt_count": prompt_count,
        "weighted_loss": weighted_loss,
        "text_weight": text_weight,
        "row_identity_sha256": krea_provenance.canonical_sha256(row_identities),
        "evaluator_identity_sha256": krea_provenance.canonical_sha256(
            evaluator_identity
        ),
    }, value


def _validate_evidence(
    output: Path,
    *,
    result_sha256: str,
    candidate_sha256: str,
    candidate_name: str,
) -> dict[str, Any]:
    rows, binding = _parse_manifest(output / "evidence.sha256", "evidence manifest")
    if {name for _, name in rows} != _EVIDENCE_FILES:
        raise RecoveryEvidenceError("evidence manifest file set drifted")
    for expected_sha, name in rows:
        path = _safe_file(output / name, f"evidence file {name}")
        if krea_provenance.file_sha256(path) != expected_sha:
            raise RecoveryEvidenceError(f"evidence file {name} hash drifted")
        if name == "exact-score.json" and expected_sha != result_sha256:
            raise RecoveryEvidenceError("evidence manifest binds another result")
    inputs, inputs_binding = _parse_manifest(output / "inputs.sha256", "input manifest")
    candidate_rows = [
        row for row in inputs if row[0] == candidate_sha256 and row[1] == candidate_name
    ]
    if len(candidate_rows) != 1:
        raise RecoveryEvidenceError(
            "input manifest does not bind exactly one candidate"
        )
    stderr = _safe_file(output / "evaluator.stderr", "evaluator stderr")
    if stderr.stat().st_size != 0:
        raise RecoveryEvidenceError("exact evaluator emitted stderr")
    return {
        **binding,
        "entry_count": len(rows),
        "file_set": sorted(name for _, name in rows),
        "inputs_file_sha256": inputs_binding["file_sha256"],
        "inputs_entry_count": len(inputs),
    }


def _validate_provenance(
    receipt: Mapping[str, str],
    *,
    task_id: str,
    result: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    raw_path = receipt.get("provenance")
    if raw_path is None:
        return None, []
    value, binding = _load_json(
        Path(raw_path), "reconciliation provenance", canonical=True
    )
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        value.get("kind") != "forge-krea-recovered-exact-score-reconciliation"
        or value.get("state") != "COMPLETE"
        or value.get("task", {}).get("task_id") != task_id
        or value.get("receipt_sha256") != krea_provenance.canonical_sha256(body)
        or receipt.get("provenance_sha256") != value.get("receipt_sha256")
        or receipt.get("provenance_file_sha256") != binding["file_sha256"]
    ):
        raise RecoveryEvidenceError("reconciliation provenance identity drifted")
    selected = value.get("selected_result")
    if (
        not isinstance(selected, dict)
        or selected.get("result", {}).get("file_sha256") != result["file_sha256"]
        or selected.get("evidence", {}).get("file_sha256") != evidence["file_sha256"]
        or value.get("strict_discovery_evidence") is not False
        or not isinstance(selected.get("validation"), dict)
        or set(selected["validation"])
        != {
            "candidate_bytes_and_sha256",
            "fixture_manifest_and_dataset_sha256",
            "dataset_file_set_and_prompt_hashes",
            "scored_rows_and_prompt_count",
            "result_loss_and_runtime_identity",
            "run_status_rc0_and_chronology",
            "input_and_evidence_manifests_rehashed",
        }
        or any(item is not True for item in selected["validation"].values())
    ):
        raise RecoveryEvidenceError("reconciliation provenance validation drifted")
    attempts = value.get("attempts")
    if not isinstance(attempts, list):
        raise RecoveryEvidenceError("reconciliation provenance attempts are invalid")
    if sum(item.get("state") == "validated_rc0" for item in attempts) != 1:
        raise RecoveryEvidenceError(
            "reconciliation provenance has duplicate/missing rc0"
        )
    return binding, attempts


def _validate_completed(
    row: Mapping[str, str], receipt_path: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    task_id = row["task_id"]
    parsed = _parse_task(task_id)
    receipt, receipt_binding = _load_env(receipt_path, f"{task_id} receipt")
    required = {
        "task_id",
        "state",
        "fixture",
        "candidate",
        "candidate_sha256",
        "output_dir",
    }
    if (
        not required.issubset(receipt)
        or receipt["task_id"] != task_id
        or receipt["state"] != "COMPLETE"
        or receipt["fixture"] != parsed["fixture"]
        or receipt["candidate_sha256"] != row["candidate_sha256"]
        or row["state"] != "COMPLETE"
    ):
        raise RecoveryEvidenceError("coverage and receipt do not agree on COMPLETE")
    candidate = _safe_file(Path(receipt["candidate"]), f"{task_id} candidate")
    if krea_provenance.file_sha256(candidate) != row["candidate_sha256"]:
        raise RecoveryEvidenceError("candidate bytes differ from coverage")
    output = _safe_directory(Path(receipt["output_dir"]), f"{task_id} output")
    status, _ = _validate_status(
        output,
        fixture=parsed["fixture"],
        candidate_sha256=row["candidate_sha256"],
    )
    result, _ = _validate_result(
        output,
        fixture=parsed["fixture"],
        candidate_path=candidate,
        candidate_sha256=row["candidate_sha256"],
    )
    evidence = _validate_evidence(
        output,
        result_sha256=result["file_sha256"],
        candidate_sha256=row["candidate_sha256"],
        candidate_name=candidate.name,
    )
    for receipt_key, observed in (
        ("result_sha256", result["file_sha256"]),
        ("dataset_sha256", result["dataset_sha256"]),
        ("evidence_manifest_sha256", evidence["file_sha256"]),
    ):
        if receipt_key in receipt and receipt[receipt_key] != observed:
            raise RecoveryEvidenceError(f"receipt {receipt_key} drifted")
    if "prompt_count" in receipt and receipt["prompt_count"] != str(
        result["prompt_count"]
    ):
        raise RecoveryEvidenceError("receipt prompt_count drifted")
    provenance, attempts = _validate_provenance(
        receipt,
        task_id=task_id,
        result=result,
        evidence=evidence,
    )
    selected_attempt = {
        "directory": str(output),
        "state": "validated_rc0",
        "status_file_sha256": status["file_sha256"],
        "result_file_sha256": result["file_sha256"],
    }
    if not attempts:
        attempts = [selected_attempt]
    artifact = {
        "receipt": receipt_binding,
        "output_directory": str(output),
        "candidate": result["candidate"],
        "status": status,
        "result": {key: value for key, value in result.items() if key != "candidate"},
        "evidence": evidence,
        "provenance": provenance,
    }
    return artifact, receipt, attempts


def _artifact_row(row: dict[str, str], receipt_dir: Path) -> dict[str, Any]:
    parsed = _parse_task(row["task_id"])
    receipt_path = receipt_dir / f"{row['task_id']}.env"
    base: dict[str, Any] = {
        "task_id": row["task_id"],
        "fixture_id": parsed["fixture"],
        "family_id": parsed["family"],
        "step": parsed["step"],
        "is_final": parsed["is_final"],
        "zero_control": parsed["zero_control"],
        "coverage_tier": row["coverage_tier"],
        "ledger_state": row["state"],
        "expected_candidate": row["expected_candidate"],
        "expected_candidate_sha256": row["candidate_sha256"],
        "selection_eligible": False,
        "failures": [],
        "attempts": [],
        "log_bindings": [],
        "validated_artifact": None,
    }
    if not receipt_path.exists():
        base["failures"] = ["receipt_missing"]
        return base
    try:
        receipt, receipt_binding = _load_env(receipt_path, f"{row['task_id']} receipt")
        base["receipt"] = receipt_binding
        output_raw = receipt.get("output_dir")
        if output_raw:
            _reject_sealed(output_raw, f"{row['task_id']} output")
            base["log_bindings"] = _log_bindings(Path(output_raw))
        if row["state"] != "COMPLETE" or receipt.get("state") != "COMPLETE":
            base["failures"] = [
                f"coverage_or_receipt_incomplete:{row['state']}:{receipt.get('state', 'missing')}"
            ]
            return base
        validated, _, attempts = _validate_completed(row, receipt_path)
        base["validated_artifact"] = validated
        base["attempts"] = attempts
        base["selection_eligible"] = True
        base["failures"] = [
            item
            for item in attempts
            if item.get("state") in {"nonzero", "incomplete", "status_missing"}
        ]
    except (OSError, RecoveryEvidenceError, ValueError) as exc:
        base["failures"] = [f"validation_error:{exc}"]
    return base


def _publish(output: Path, value: dict[str, Any]) -> None:
    output = Path(os.path.abspath(os.path.expanduser(output)))
    _reject_sealed(output, "recovery index output")
    production = Path(__file__).resolve().parents[2] / "forge"
    if output == production or production in output.parents:
        raise RecoveryEvidenceError(
            "recovery index cannot target the production package"
        )
    current = output.parent
    while current != current.parent:
        if current.is_symlink():
            raise RecoveryEvidenceError("recovery index output has a symlink ancestor")
        current = current.parent
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if os.path.lexists(output) or os.path.lexists(temporary):
        raise FileExistsError(f"refusing existing recovery index output: {output}")
    payload = krea_provenance.canonical_bytes(value) + b"\n"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output)
        temporary.unlink()
        directory_fd = os.open(
            output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_index_document(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecoveryEvidenceError("recovery index must be an object")
    _exact(
        value,
        {
            "schema",
            "kind",
            "indexed_at_utc",
            "coverage_ledger",
            "coverage",
            "artifacts",
            "claims",
            "authority",
            "index_sha256",
        },
        "recovery index",
    )
    body = {key: item for key, item in value.items() if key != "index_sha256"}
    if (
        value["schema"] != INDEX_SCHEMA
        or value["kind"] != INDEX_KIND
        or value["claims"] != FALSE_CLAIMS
        or value["authority"]
        != {
            "scope": "D1_D2_recovery_exact_score_evidence_index_only",
            "selection_freeze_requires_separate_waiver": True,
            "deployment_authorized": False,
        }
        or value["index_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise RecoveryEvidenceError("recovery index identity/authority drifted")
    _timestamp(value["indexed_at_utc"], "indexed_at_utc")
    ledger = value["coverage_ledger"]
    if not isinstance(ledger, dict):
        raise RecoveryEvidenceError("coverage ledger binding is invalid")
    _digest(ledger.get("file_sha256"), "coverage ledger file SHA")
    artifacts = value["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != EXPECTED_EXACT_SCORES:
        raise RecoveryEvidenceError("recovery index artifact count drifted")
    tasks = [row.get("task_id") for row in artifacts if isinstance(row, dict)]
    if len(tasks) != len(artifacts) or len(tasks) != len(set(tasks)):
        raise RecoveryEvidenceError("recovery index repeats or corrupts task ids")
    eligible = sum(row.get("selection_eligible") is True for row in artifacts)
    coverage_complete = sum(row.get("ledger_state") == "COMPLETE" for row in artifacts)
    coverage = value["coverage"]
    expected = {
        "expected_exact_scores": EXPECTED_EXACT_SCORES,
        "candidate_scores": EXPECTED_CANDIDATE_SCORES,
        "independent_zero_scores": EXPECTED_ZERO_SCORES,
        "ledger_complete": coverage_complete,
        "selection_eligible": eligible,
        "selection_gate_ready": eligible == EXPECTED_EXACT_SCORES,
    }
    if coverage != expected:
        raise RecoveryEvidenceError(
            "recovery index coverage summary does not recompute"
        )
    _validate_selection_set(
        artifacts, require_complete=coverage["selection_gate_ready"]
    )
    return value


def _validate_selection_set(
    artifacts: Iterable[Mapping[str, Any]], *, require_complete: bool
) -> None:
    """Require one paired evaluator/dataset identity for every eligible score."""

    eligible = [row for row in artifacts if row.get("selection_eligible") is True]
    evaluator_identities: set[str] = set()
    fixture_datasets: dict[str, set[str]] = {fixture: set() for fixture in FIXTURES}
    fixture_rows: dict[str, set[str]] = {fixture: set() for fixture in FIXTURES}
    zero_shas: set[str] = set()
    for row in eligible:
        validated = row.get("validated_artifact")
        if not isinstance(validated, dict) or not isinstance(
            validated.get("result"), dict
        ):
            raise RecoveryEvidenceError("eligible recovery row lacks validated result")
        result = validated["result"]
        fixture = row.get("fixture_id")
        if fixture not in FIXTURES:
            raise RecoveryEvidenceError("eligible recovery row has invalid fixture")
        evaluator_identities.add(
            _digest(result.get("evaluator_identity_sha256"), "evaluator identity")
        )
        fixture_datasets[fixture].add(
            _digest(result.get("dataset_sha256"), "fixture dataset SHA")
        )
        fixture_rows[fixture].add(
            _digest(result.get("row_identity_sha256"), "fixture row identity SHA")
        )
        if row.get("zero_control") is True:
            zero_shas.add(
                _digest(row.get("expected_candidate_sha256"), "zero candidate SHA")
            )
    if len(evaluator_identities) > 1:
        raise RecoveryEvidenceError(
            "eligible scores use different evaluator identities"
        )
    if any(len(values) > 1 for values in fixture_datasets.values()):
        raise RecoveryEvidenceError(
            "eligible scores use different datasets within a fixture"
        )
    if any(len(values) > 1 for values in fixture_rows.values()):
        raise RecoveryEvidenceError(
            "eligible scores are not paired to identical fixture rows"
        )
    if len(zero_shas) > 1:
        raise RecoveryEvidenceError("D1/D2 eligible zero controls use different bytes")
    if require_complete and (
        len(eligible) != EXPECTED_EXACT_SCORES
        or len(evaluator_identities) != 1
        or any(len(values) != 1 for values in fixture_datasets.values())
        or any(len(values) != 1 for values in fixture_rows.values())
        or len(zero_shas) != 1
    ):
        raise RecoveryEvidenceError("92-score selection set is not coherently paired")


def build_index(
    *,
    coverage_ledger: Path,
    receipt_dir: Path,
    output: Path,
    indexed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Index all 92 rows, but expose scores only after exact rc0 validation."""

    rows, ledger_binding = _load_ledger(coverage_ledger)
    receipt_dir = _safe_directory(receipt_dir, "recovery receipt directory")
    artifacts = [_artifact_row(row, receipt_dir) for row in rows]
    eligible = sum(row["selection_eligible"] is True for row in artifacts)
    ledger_complete = sum(row["ledger_state"] == "COMPLETE" for row in artifacts)
    body = {
        "schema": INDEX_SCHEMA,
        "kind": INDEX_KIND,
        "indexed_at_utc": _timestamp(
            indexed_at_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "indexed_at_utc",
        ),
        "coverage_ledger": ledger_binding,
        "coverage": {
            "expected_exact_scores": EXPECTED_EXACT_SCORES,
            "candidate_scores": EXPECTED_CANDIDATE_SCORES,
            "independent_zero_scores": EXPECTED_ZERO_SCORES,
            "ledger_complete": ledger_complete,
            "selection_eligible": eligible,
            "selection_gate_ready": eligible == EXPECTED_EXACT_SCORES,
        },
        "artifacts": artifacts,
        "claims": dict(FALSE_CLAIMS),
        "authority": {
            "scope": "D1_D2_recovery_exact_score_evidence_index_only",
            "selection_freeze_requires_separate_waiver": True,
            "deployment_authorized": False,
        },
    }
    value = {**body, "index_sha256": krea_provenance.canonical_sha256(body)}
    _validate_index_document(value)
    _publish(output, value)
    return value


def load_index(path: Path) -> tuple[dict[str, Any], str]:
    value, binding = _load_json(path, "recovery evidence index", canonical=True)
    _validate_index_document(value)
    ledger_path = Path(value["coverage_ledger"]["path"])
    if (
        _file_binding(ledger_path, "indexed coverage ledger")
        != value["coverage_ledger"]
    ):
        raise RecoveryEvidenceError("indexed coverage ledger bytes drifted")
    for row in value["artifacts"]:
        if row["selection_eligible"] is True:
            receipt = row.get("receipt")
            if not isinstance(receipt, dict):
                raise RecoveryEvidenceError("eligible row lacks a receipt binding")
            if _file_binding(Path(receipt["path"]), "indexed receipt") != receipt:
                raise RecoveryEvidenceError("indexed receipt bytes drifted")
            ledger_row = {
                "task_id": row["task_id"],
                "state": row["ledger_state"],
                "candidate_sha256": row["expected_candidate_sha256"],
            }
            observed, _, attempts = _validate_completed(
                ledger_row, Path(receipt["path"])
            )
            if observed != row["validated_artifact"] or attempts != row["attempts"]:
                raise RecoveryEvidenceError(
                    f"indexed exact-score artifact drifted: {row['task_id']}"
                )
        for log in row["log_bindings"]:
            if _file_binding(Path(log["path"]), f"indexed log {log['name']}") != {
                key: value for key, value in log.items() if key != "name"
            }:
                raise RecoveryEvidenceError(f"indexed log drifted: {row['task_id']}")
    return value, binding["file_sha256"]


def validate_index(path: Path) -> dict[str, Any]:
    """Validate a published recovery index and all selection-eligible bytes."""

    return load_index(path)[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--coverage-ledger", required=True, type=Path)
    build.add_argument("--receipt-dir", required=True, type=Path)
    build.add_argument("--indexed-at-utc")
    build.add_argument("--output", required=True, type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("--index", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        value = build_index(
            coverage_ledger=args.coverage_ledger,
            receipt_dir=args.receipt_dir,
            output=args.output,
            indexed_at_utc=args.indexed_at_utc,
        )
    else:
        value = validate_index(args.index)
    print(value["index_sha256"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
