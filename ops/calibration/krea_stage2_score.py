#!/usr/bin/env python3
"""Fail-closed exact-score plans and receipts for Krea Stage-2.

This module deliberately does not launch training, reveal fixtures, select a
finalist, or authorize a release.  It binds one admitted Stage-2 cell to the
complete candidate set that must be scored and independently recomputes the
validator-format exact-score result for every candidate.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
from typing import Any, Mapping

try:
    from . import krea_fixture
    from . import krea_provenance
    from . import krea_stage2_execution
    from . import krea_stage2_legacy_confirmation
    from . import krea_stage2_training_evidence
except ImportError:  # pragma: no cover
    import krea_fixture  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_stage2_execution  # type: ignore[no-redef]
    import krea_stage2_legacy_confirmation  # type: ignore[no-redef]
    import krea_stage2_training_evidence  # type: ignore[no-redef]


PLAN_KIND = "forge-krea-stage2-exact-score-plan"
RECEIPT_KIND = "forge-krea-stage2-exact-score-receipt"
AGGREGATE_KIND = "forge-krea-stage2-exact-score-aggregate"
SCHEMA = 1
PUBLIC_REFERENCES = ("K2", "K3", "K4")
CONTROL = "K0"
ASSET_NAMES = ("diffusion_model", "text_encoder", "vae")
_SHA = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_RUN_MECHANICS = {
    "natural_completion": True,
    "planned_steps_completed": True,
    "upload_ready": True,
    "clean_telemetry": True,
    "decision_completed_before_export_reserve": True,
    "fallback_used": False,
}
_SOURCE_KEYS = {
    "god",
    "comfyui",
    "tooling_nodes",
    "expected_commits",
    "god_import_bindings",
    "workflow_path",
    "workflow_sha256",
    "calibration_shim_sha256",
    "comfy_main_sha256",
}
_RUNTIME_KEYS = {
    "fresh_comfy_process",
    "loopback",
    "port",
    "cache",
    "database",
    "api_nodes_disabled",
    "isolated_input_output_temp_user",
    "offline_environment",
    "custom_node_allowlist",
    "startup_timeout_s",
    "evaluation_timeout_s",
    "shutdown_timeout_s",
    "shutdown",
    "python",
    "driver_python",
    "comfy_system_stats",
    "comfy_history",
    "comfy_log",
    "comfy_log_sha256",
    "comfy_log_bytes",
}
_RUNTIME_SOURCE_KEYS = {
    "fresh_comfy_process",
    "loopback",
    "cache",
    "database",
    "api_nodes_disabled",
    "isolated_input_output_temp_user",
    "offline_environment",
    "custom_node_allowlist",
    "python",
    "driver_python",
}
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


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(
            f"{label} keys differ: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} is not a safe identifier")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label} is outside its finite range")
    return result


def _bounded_number(value: Any, label: str) -> float:
    result = _number(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} is outside [0,1]")
    return result


def _absolute_lexical_path(value: Any, label: str) -> str:
    """Reject aliases while retaining the exact path seen by the evaluator.

    Score plans may be replayed outside the container where the dataset is
    mounted, so existence is not required here.  Every existing component is
    nevertheless checked with ``lstat`` and a result must repeat this literal
    path, not merely resolve to the same directory.
    """

    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty absolute path")
    if not value.startswith("/") or os.path.normpath(value) != value:
        raise ValueError(f"{label} must be an absolute canonical lexical path")
    parts = PurePosixPath(value).parts
    if any(part in {"", ".", ".."} for part in parts[1:]):
        raise ValueError(f"{label} contains an unsafe path component")
    path = Path(value)
    for current in (path, *path.parents):
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(f"{label} cannot be inspected") from exc
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} contains a symlink component")
    return value


def _safe_regular_file(value: Path | str, label: str) -> Path:
    raw = os.path.abspath(os.path.expanduser(os.fspath(value)))
    path = Path(raw)
    for current in (path, *path.parents):
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise ValueError(f"{label} is not an inspectable regular file") from exc
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} has a symlink component")
        if current == path and not stat.S_ISREG(mode):
            raise ValueError(f"{label} is not a regular file")
    return path


def _load_canonical_json(
    path: Path | str, label: str
) -> tuple[Path, dict[str, Any], str]:
    safe = _safe_regular_file(path, label)
    try:
        raw = safe.read_bytes()
        value = _object(json.loads(raw), label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if raw != krea_provenance.canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} is not canonical JSON plus one newline")
    return safe, value, krea_provenance.file_sha256(safe)


def _binding(value: Any, label: str, semantic_key: str) -> dict[str, str]:
    row = _object(value, label)
    _exact(row, {"file_sha256", semantic_key}, label)
    return {
        "file_sha256": _sha(row["file_sha256"], f"{label}.file_sha256"),
        semantic_key: _sha(row[semantic_key], f"{label}.{semantic_key}"),
    }


def _fixture_score_view(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate either a canonical B manifest or the admitted legacy C wrapper."""

    document = dict(value)
    if document.get("kind") == krea_stage2_legacy_confirmation.KIND:
        return krea_stage2_legacy_confirmation.score_view(document)
    return krea_fixture.validate_manifest(document)


def _run_mechanics(value: Any, label: str = "run mechanics") -> dict[str, bool]:
    row = _object(value, label)
    _exact(row, set(_RUN_MECHANICS), label)
    if row != _RUN_MECHANICS:
        raise ValueError(f"{label} lacks clean upload-ready boundary mechanics")
    return dict(row)


def _candidate_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("score plan candidates must be non-empty")
    rows: list[dict[str, Any]] = []
    families: set[str] = set()
    candidates: set[str] = set()
    artifacts: set[str] = set()
    for index, raw in enumerate(value):
        row = _object(raw, f"candidates[{index}]")
        _exact(
            row,
            {
                "family_id",
                "training_candidate_id",
                "execution_plan_sha256",
                "execution_approval_sha256",
                "run_completion_sha256",
                "run_evidence_file_sha256",
                "run_evidence_sha256",
                "mechanics",
                "candidate_id",
                "candidate_sha256",
                "candidate_bytes",
                "checkpoint_rule_sha256",
                "checkpoint_target_fraction",
                "checkpoint_mapping_rule",
                "step",
                "fraction_numerator",
                "fraction_denominator",
            },
            f"candidates[{index}]",
        )
        family = _safe_id(row["family_id"], "candidate family")
        candidate = _safe_id(row["candidate_id"], "candidate id")
        _safe_id(row["training_candidate_id"], "training candidate id")
        digest = _sha(row["candidate_sha256"], "candidate sha256")
        for key in (
            "execution_plan_sha256",
            "execution_approval_sha256",
            "run_completion_sha256",
            "run_evidence_file_sha256",
            "run_evidence_sha256",
            "checkpoint_rule_sha256",
        ):
            _sha(row[key], f"candidate {key}")
        target = _object(
            row["checkpoint_target_fraction"],
            f"candidates[{index}].checkpoint_target_fraction",
        )
        _exact(
            target,
            {"numerator", "denominator"},
            f"candidates[{index}].checkpoint_target_fraction",
        )
        target_numerator = target["numerator"]
        target_denominator = target["denominator"]
        if (
            isinstance(target_numerator, bool)
            or not isinstance(target_numerator, int)
            or target_numerator <= 0
            or isinstance(target_denominator, bool)
            or not isinstance(target_denominator, int)
            or target_denominator <= 0
            or target_numerator > target_denominator
            or math.gcd(target_numerator, target_denominator) != 1
        ):
            raise ValueError(
                "candidate checkpoint target must be a positive reduced fraction"
            )
        if (
            row["checkpoint_mapping_rule"]
            != krea_stage2_execution._CHECKPOINT_MAPPING_RULE
        ):
            raise ValueError("candidate checkpoint mapping rule differs")
        _run_mechanics(row["mechanics"], f"candidates[{index}].mechanics")
        if family in families or candidate in candidates or digest in artifacts:
            raise ValueError("score plan repeats a family, candidate, or artifact")
        for key in (
            "candidate_bytes",
            "step",
            "fraction_numerator",
            "fraction_denominator",
        ):
            if (
                isinstance(row[key], bool)
                or not isinstance(row[key], int)
                or row[key] <= 0
            ):
                raise ValueError(f"candidate {key} must be a positive integer")
        if (
            row["fraction_numerator"] != row["step"]
            or row["fraction_denominator"] < row["fraction_numerator"]
        ):
            raise ValueError("candidate step/fraction is inconsistent")
        families.add(family)
        candidates.add(candidate)
        artifacts.add(digest)
        rows.append(dict(row))
    if rows != sorted(rows, key=lambda row: row["family_id"]):
        raise ValueError("score plan candidates must be sorted by family")
    return rows


def _asset_map(value: Any, label: str, *, hashes: bool) -> dict[str, Any]:
    row = _object(value, label)
    _exact(row, set(ASSET_NAMES), label)
    for name in ASSET_NAMES:
        if hashes:
            _sha(row[name], f"{label}.{name}")
        elif (
            isinstance(row[name], bool)
            or not isinstance(row[name], int)
            or row[name] <= 0
        ):
            raise ValueError(f"{label}.{name} must be a positive integer")
    return dict(row)


def _validate_source(value: Any) -> dict[str, Any]:
    source = _object(value, "exact-score source")
    _exact(source, _SOURCE_KEYS, "exact-score source")
    repositories: dict[str, dict[str, Any]] = {}
    for name in ("god", "comfyui", "tooling_nodes"):
        repository = _object(source[name], f"exact-score source {name}")
        _exact(
            repository,
            {
                "commit",
                "tree",
                "tracked_worktree_clean",
                "nonignored_worktree_clean",
            },
            f"exact-score source {name}",
        )
        if (
            repository["tracked_worktree_clean"] is not True
            or repository["nonignored_worktree_clean"] is not True
            or re.fullmatch(r"[0-9a-f]{40}", str(repository["commit"])) is None
            or re.fullmatch(r"[0-9a-f]{40}", str(repository["tree"])) is None
        ):
            raise ValueError("exact-score source is not clean and commit-bound")
        repositories[name] = repository
    commits = _object(source["expected_commits"], "expected source commits")
    _exact(commits, {"god", "comfyui", "tooling_nodes"}, "expected source commits")
    if any(commits[name] != repositories[name]["commit"] for name in repositories):
        raise ValueError("exact-score expected commits differ from source checkouts")
    imports = _object(source["god_import_bindings"], "G.O.D. import bindings")
    if not imports:
        raise ValueError("exact-score G.O.D. import bindings are empty")
    for module, raw in imports.items():
        _safe_id(module, "G.O.D. import module")
        binding = _object(raw, f"G.O.D. import binding {module}")
        _exact(binding, {"module", "path", "sha256"}, f"G.O.D. import binding {module}")
        if not isinstance(binding["path"], str) or not binding["path"]:
            raise ValueError("exact-score G.O.D. import path is empty")
        relative = PurePosixPath(str(binding["path"]))
        if (
            binding["module"] != module
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError("exact-score G.O.D. import binding is unsafe")
        _sha(binding["sha256"], f"G.O.D. import binding {module} sha256")
    if not isinstance(source["workflow_path"], str) or not source["workflow_path"]:
        raise ValueError("exact-score workflow path is empty")
    workflow = PurePosixPath(source["workflow_path"])
    if workflow.is_absolute() or any(
        part in {"", ".", ".."} for part in workflow.parts
    ):
        raise ValueError("exact-score workflow path is unsafe")
    for key in ("workflow_sha256", "calibration_shim_sha256", "comfy_main_sha256"):
        _sha(source[key], f"exact-score source {key}")
    return dict(source)


def _runtime_source(value: Any) -> dict[str, Any]:
    runtime = _object(value, "exact-score runtime")
    _exact(runtime, _RUNTIME_KEYS, "exact-score runtime")
    required = {
        "fresh_comfy_process": True,
        "loopback": "127.0.0.1",
        "cache": "comfy_default_fresh_process",
        "database": "memory",
        "api_nodes_disabled": True,
        "isolated_input_output_temp_user": True,
        "offline_environment": True,
        "custom_node_allowlist": ["comfyui-tooling-nodes"],
    }
    if any(runtime[key] != expected for key, expected in required.items()):
        raise ValueError("exact-score runtime isolation differs")
    for key in ("port", "comfy_log_bytes"):
        upper = 65535 if key == "port" else None
        if (
            isinstance(runtime[key], bool)
            or not isinstance(runtime[key], int)
            or runtime[key] <= 0
            or (upper is not None and runtime[key] > upper)
        ):
            raise ValueError(f"exact-score runtime {key} is invalid")
    for key in ("startup_timeout_s", "evaluation_timeout_s", "shutdown_timeout_s"):
        _number(runtime[key], f"exact-score runtime {key}", positive=True)
    for key in ("python", "driver_python", "comfy_system_stats"):
        if not _object(runtime[key], f"exact-score runtime {key}"):
            raise ValueError(f"exact-score runtime {key} is empty")
    shutdown = _object(runtime["shutdown"], "exact-score shutdown")
    _exact(shutdown, {"returncode", "stop_signal", "forced"}, "exact-score shutdown")
    if (
        isinstance(shutdown["returncode"], bool)
        or not isinstance(shutdown["returncode"], int)
        or shutdown["stop_signal"] != "SIGINT"
        or shutdown["forced"] is not False
    ):
        raise ValueError("exact-score shutdown was not controlled")
    history = _object(runtime["comfy_history"], "Comfy history")
    _exact(history, {"prompt_count", "history_sha256"}, "Comfy history")
    if (
        isinstance(history["prompt_count"], bool)
        or not isinstance(history["prompt_count"], int)
        or history["prompt_count"] <= 0
    ):
        raise ValueError("exact-score prompt count is invalid")
    _sha(history["history_sha256"], "Comfy history")
    _absolute_lexical_path(runtime["comfy_log"], "exact-score Comfy log")
    _sha(runtime["comfy_log_sha256"], "exact-score Comfy log")
    return {key: runtime[key] for key in _RUNTIME_SOURCE_KEYS}


def _evaluator_contract(value: Any) -> dict[str, Any]:
    row = _object(value, "evaluator contract")
    keys = {
        "evaluator",
        "model_type",
        "base_name",
        "asset_sha256",
        "asset_bytes",
        "steps",
        "cfg",
        "denoise",
        "generations",
        "master_seed",
        "seeds",
        "text_weight",
        "source_sha256",
        "runtime_source_sha256",
        "contract_sha256",
    }
    _exact(row, keys, "evaluator contract")
    body = {key: item for key, item in row.items() if key != "contract_sha256"}
    if (
        row["evaluator"] != "god_krea2_img2img_exact"
        or row["model_type"] != "krea2"
        or row["contract_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("evaluator contract identity differs")
    if (
        not isinstance(row["base_name"], str)
        or not row["base_name"]
        or Path(row["base_name"]).name != row["base_name"]
    ):
        raise ValueError("evaluator base name is not one filename")
    _asset_map(row["asset_sha256"], "evaluator asset SHA-256", hashes=True)
    _asset_map(row["asset_bytes"], "evaluator asset bytes", hashes=False)
    _sha(row["source_sha256"], "evaluator source")
    _sha(row["runtime_source_sha256"], "evaluator runtime source")
    for key in ("steps", "generations"):
        if isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] <= 0:
            raise ValueError(f"evaluator {key} must be positive")
    if _number(row["cfg"], "evaluator cfg") < 0:
        raise ValueError("evaluator cfg must be nonnegative")
    _bounded_number(row["denoise"], "evaluator denoise")
    _bounded_number(row["text_weight"], "evaluator text weight")
    if (
        isinstance(row["master_seed"], bool)
        or not isinstance(row["master_seed"], int)
        or not 0 <= row["master_seed"] <= 2**32 - 1
        or not isinstance(row["seeds"], list)
        or len(row["seeds"]) != row["generations"]
        or any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= 2**32 - 1
            for seed in row["seeds"]
        )
        or len(set(row["seeds"])) != len(row["seeds"])
    ):
        raise ValueError("evaluator seed schedule differs")
    return dict(row)


def evaluator_contract_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    result = _object(result, "exact-score result")
    _exact(result, _RESULT_KEYS, "exact-score result")
    source = _validate_source(result["source"])
    runtime_source = _runtime_source(result["runtime"])
    body = {
        "evaluator": result["evaluator"],
        "model_type": result["model_type"],
        "base_name": result["base_name"],
        "asset_sha256": result["asset_sha256"],
        "asset_bytes": result["asset_bytes"],
        "steps": result["steps"],
        "cfg": result["cfg"],
        "denoise": result["denoise"],
        "generations": result["generations"],
        "master_seed": result["master_seed"],
        "seeds": result["seeds"],
        "text_weight": result["text_weight"],
        "source_sha256": krea_provenance.canonical_sha256(source),
        "runtime_source_sha256": krea_provenance.canonical_sha256(runtime_source),
    }
    return _evaluator_contract(
        {**body, "contract_sha256": krea_provenance.canonical_sha256(body)}
    )


def validate_plan(value: Any) -> dict[str, Any]:
    plan = _object(value, "Stage-2 score plan")
    keys = {
        "schema",
        "kind",
        "phase",
        "cell_id",
        "fixture_id",
        "seed_role",
        "seed",
        "hours",
        "candidate_family_id",
        "public_reference_family_ids",
        "control_family_id",
        "candidates",
        "fixture_manifest",
        "evaluation_dataset_sha256",
        "evaluation_dataset_path",
        "evaluation_row_count",
        "evaluator_contract",
        "waiver_finalist_freeze",
        "confirmation_materialization",
        "owner_ratification",
        "gpu_execution_authorization",
        "production_identity",
        "production_image_id",
        "created_at_utc",
        "fallback_allowed",
        "release_authorized",
        "production_mutation_authorized",
        "plan_sha256",
    }
    _exact(plan, keys, "Stage-2 score plan")
    body = {key: item for key, item in plan.items() if key != "plan_sha256"}
    if (
        plan["schema"] != SCHEMA
        or plan["kind"] != PLAN_KIND
        or plan["plan_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("Stage-2 score plan identity differs")
    phase = plan["phase"]
    cell = _safe_id(plan["cell_id"], "score cell")
    fixture = _safe_id(plan["fixture_id"], "score fixture")
    seed_role = plan["seed_role"]
    krea_stage2_execution._validate_cell(
        phase, cell, fixture, seed_role, plan["seed"], str(plan["hours"])
    )
    candidate_family = _safe_id(plan["candidate_family_id"], "candidate family")
    if candidate_family == CONTROL:
        raise ValueError("Stage-2 candidate family cannot be K0")
    if plan["public_reference_family_ids"] != list(PUBLIC_REFERENCES):
        raise ValueError("Stage-2 references must be exhaustive K2-K4")
    if plan["control_family_id"] != CONTROL:
        raise ValueError("Stage-2 control must be K0")
    candidates = _candidate_rows(plan["candidates"])
    observed = {row["family_id"] for row in candidates}
    expected = (
        {candidate_family, CONTROL, *PUBLIC_REFERENCES}
        if phase == "confirmation"
        else {candidate_family}
    )
    if observed != expected:
        raise ValueError("Stage-2 score plan does not exhaust its required families")
    _binding(plan["fixture_manifest"], "fixture manifest", "manifest_sha256")
    _sha(plan["evaluation_dataset_sha256"], "evaluation dataset")
    _absolute_lexical_path(plan["evaluation_dataset_path"], "evaluation dataset path")
    if (
        isinstance(plan["evaluation_row_count"], bool)
        or not isinstance(plan["evaluation_row_count"], int)
        or plan["evaluation_row_count"] <= 0
    ):
        raise ValueError("evaluation row count must be positive")
    _evaluator_contract(plan["evaluator_contract"])
    _binding(plan["waiver_finalist_freeze"], "waiver freeze", "freeze_sha256")
    _binding(
        plan["confirmation_materialization"],
        "confirmation materialization",
        "materialization_sha256",
    )
    _binding(plan["owner_ratification"], "owner ratification", "ratification_sha256")
    _binding(
        plan["gpu_execution_authorization"],
        "GPU execution authorization",
        "gpu_execution_authorization_sha256",
    )
    _binding(
        plan["production_identity"],
        "production identity",
        "production_identity_sha256",
    )
    if (
        not isinstance(plan["production_image_id"], str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", plan["production_image_id"]) is None
    ):
        raise ValueError("production image id is not exact")
    krea_stage2_execution._utc(plan["created_at_utc"], "score plan created_at_utc")
    if (
        plan["fallback_allowed"] is not False
        or plan["release_authorized"] is not False
        or plan["production_mutation_authorized"] is not False
    ):
        raise ValueError("Stage-2 score plan overclaims authority")
    return dict(plan)


def seal_plan(payload: dict[str, Any]) -> dict[str, Any]:
    if "plan_sha256" in payload:
        raise ValueError("unsealed Stage-2 score plan includes a digest")
    plan = {**payload, "plan_sha256": krea_provenance.canonical_sha256(payload)}
    return validate_plan(plan)


def _candidate_from_run_control(
    *,
    family_id: str,
    candidate_id: str,
    control: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one score row only after replaying its Stage-2 run evidence."""

    family = _safe_id(family_id, "candidate family")
    identifier = _safe_id(candidate_id, "candidate id")
    raw = _object(control, f"run control {family}")
    _exact(
        raw,
        {
            "run_evidence_path",
            "execution_plan",
            "execution_approval",
            "run_completion",
            "candidate_path",
        },
        f"run control {family}",
    )
    _evidence_path, evidence, evidence_file_sha = _load_canonical_json(
        raw["run_evidence_path"], f"run evidence {family}"
    )
    execution_plan = _object(raw["execution_plan"], f"execution plan {family}")
    validated = krea_stage2_training_evidence.validate_run_evidence(
        evidence,
        plan=execution_plan,
        approval=_object(raw["execution_approval"], f"execution approval {family}"),
        completion=_object(raw["run_completion"], f"run completion {family}"),
    )
    if validated != evidence:
        raise ValueError("Stage-2 run-evidence validator returned a different document")
    mechanics = _run_mechanics(evidence.get("mechanics"))
    if (
        evidence.get("natural_completion") is not True
        or evidence.get("fallback_used") is not False
    ):
        raise ValueError("Stage-2 run evidence contradicts its mechanics")
    training_candidate_id = _safe_id(
        evidence.get("training_candidate_id"), "training candidate id"
    )
    universe = execution_plan.get("candidate_universe")
    if not isinstance(universe, list):
        raise ValueError("Stage-2 execution plan has no candidate universe")
    selected = [
        row
        for row in universe
        if isinstance(row, dict) and row.get("candidate_id") == training_candidate_id
    ]
    if (
        execution_plan.get("training_candidate_id") != training_candidate_id
        or len(selected) != 1
        or selected[0].get("family_id") != family
    ):
        raise ValueError("score family differs from the validated training candidate")
    phase = evidence.get("phase")
    if phase not in {"confirmation", "boundary"}:
        raise ValueError("score run phase is invalid")
    if execution_plan.get("calibration_profile") != family:
        raise ValueError(f"{phase} score family differs from calibration profile")
    planned_steps = execution_plan.get("planned_steps")
    if (
        isinstance(planned_steps, bool)
        or not isinstance(planned_steps, int)
        or planned_steps <= 0
    ):
        raise ValueError("validated execution plan has no positive planned_steps")
    checkpoint_selection = krea_stage2_execution._checkpoint_selection(
        execution_plan.get("checkpoint_selection"),
        planned_steps=planned_steps,
        profile_id=execution_plan.get("calibration_profile"),
    )
    completion = _object(raw["run_completion"], f"run completion {family}")
    private_receipts = _object(
        krea_stage2_execution.validate_private_run_receipts(execution_plan),
        f"private run receipts {family}",
    )
    if (
        "checkpoint_selection" not in private_receipts
        or "checkpoint_selection_receipt" not in completion
    ):
        raise ValueError(
            "live terminal/config/selection receipts differ from the validated completion"
        )
    live_selection = _binding(
        private_receipts["checkpoint_selection"],
        f"live checkpoint-selection receipt {family}",
        "receipt_sha256",
    )
    completed_selection = _binding(
        completion["checkpoint_selection_receipt"],
        f"completed checkpoint-selection receipt {family}",
        "receipt_sha256",
    )
    if (
        private_receipts.get("config_control")
        != completion.get("config_control_receipt")
        or private_receipts.get("training_terminal")
        != completion.get("training_terminal_receipt")
        or live_selection != completed_selection
    ):
        raise ValueError(
            "live terminal/config/selection receipts differ from the validated completion"
        )
    candidate_path = _safe_regular_file(raw["candidate_path"], f"candidate {family}")
    if candidate_path.name != "last.safetensors":
        raise ValueError("score candidate must be the promoted last.safetensors")
    candidate_sha = krea_provenance.file_sha256(candidate_path)
    candidate_bytes = candidate_path.stat().st_size
    artifacts = evidence.get("candidate_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Stage-2 run evidence has no candidate artifact list")
    final_rows = []
    for artifact in artifacts:
        artifact = _object(artifact, "Stage-2 candidate artifact")
        if PurePosixPath(str(artifact.get("path"))).name == "last.safetensors":
            final_rows.append(artifact)
    if len(final_rows) != 1 or final_rows[0] != {
        "path": final_rows[0].get("path") if final_rows else None,
        "bytes": candidate_bytes,
        "sha256": candidate_sha,
    }:
        raise ValueError("score candidate is not the unique validated last.safetensors")
    for key, semantic_key in (
        ("execution_plan", "plan_sha256"),
        ("execution_approval", "approval_sha256"),
        ("run_completion", "completion_sha256"),
    ):
        _binding(evidence.get(key), f"run evidence {key}", semantic_key)
    row = {
        "family_id": family,
        "training_candidate_id": training_candidate_id,
        "execution_plan_sha256": evidence["execution_plan"]["plan_sha256"],
        "execution_approval_sha256": evidence["execution_approval"]["approval_sha256"],
        "run_completion_sha256": evidence["run_completion"]["completion_sha256"],
        "run_evidence_file_sha256": evidence_file_sha,
        "run_evidence_sha256": _sha(
            evidence.get("evidence_sha256"), "run evidence semantic SHA-256"
        ),
        "mechanics": mechanics,
        "candidate_id": identifier,
        "candidate_sha256": candidate_sha,
        "candidate_bytes": candidate_bytes,
        "checkpoint_rule_sha256": checkpoint_selection["checkpoint_rule_sha256"],
        "checkpoint_target_fraction": checkpoint_selection["target_fraction"],
        "checkpoint_mapping_rule": checkpoint_selection["mapping_rule"],
        "step": checkpoint_selection["selected_step"],
        "fraction_numerator": checkpoint_selection["selected_step"],
        "fraction_denominator": planned_steps,
    }
    return _candidate_rows([row])[0], evidence


def build_candidate_row(
    *,
    family_id: str,
    candidate_id: str,
    run_evidence_path: Path,
    execution_plan: dict[str, Any],
    execution_approval: dict[str, Any],
    run_completion: dict[str, Any],
    candidate_path: Path,
) -> dict[str, Any]:
    """Derive a score-plan row from strongly validated run controls."""

    row, _evidence = _candidate_from_run_control(
        family_id=family_id,
        candidate_id=candidate_id,
        control={
            "run_evidence_path": run_evidence_path,
            "execution_plan": execution_plan,
            "execution_approval": execution_approval,
            "run_completion": run_completion,
            "candidate_path": candidate_path,
        },
    )
    return row


def validate_plan_with_run_controls(
    value: Any, *, controls_by_family: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay every candidate's run controls before a plan reaches a decision."""

    plan = validate_plan(value)
    if not isinstance(controls_by_family, Mapping):
        raise ValueError("run controls must be keyed by family")
    candidates = {row["family_id"]: row for row in plan["candidates"]}
    if set(controls_by_family) != set(candidates):
        raise ValueError("run controls do not exhaust the score-plan families")
    for family, expected in candidates.items():
        observed, evidence = _candidate_from_run_control(
            family_id=family,
            candidate_id=expected["candidate_id"],
            control=controls_by_family[family],
        )
        if observed != expected:
            raise ValueError(f"run control for {family} differs from the score plan")
        expected_identity = {
            "phase": plan["phase"],
            "cell_id": plan["cell_id"],
            "fixture_id": plan["fixture_id"],
            "seed_role": plan["seed_role"],
            "seed": plan["seed"],
            "hours": plan["hours"],
            "fixture_manifest": plan["fixture_manifest"],
            "waiver_finalist_freeze": plan["waiver_finalist_freeze"],
            "confirmation_materialization": plan["confirmation_materialization"],
            "owner_ratification": plan["owner_ratification"],
            "gpu_execution_authorization": plan["gpu_execution_authorization"],
            "production_identity": plan["production_identity"],
            "production_image_id": plan["production_image_id"],
        }
        if any(evidence.get(key) != item for key, item in expected_identity.items()):
            raise ValueError(
                f"run control for {family} differs from the score-plan cell"
            )
    return plan


def _validate_result(
    result: Any,
    *,
    candidate_path: Path,
    candidate: Mapping[str, Any],
    fixture_manifest: Mapping[str, Any],
    evaluator_contract: Mapping[str, Any],
    evaluation_dataset_path: str,
) -> dict[str, Any]:
    result = _object(result, "exact-score result")
    _exact(result, _RESULT_KEYS, "exact-score result")
    candidate_path = _safe_regular_file(candidate_path, "score candidate")
    candidate_sha = krea_provenance.file_sha256(candidate_path)
    if (
        result["schema"] != 2
        or result["evaluator"] != "god_krea2_img2img_exact"
        or result["model_type"] != "krea2"
        or result["direction"] != "min"
        or result["candidate"] != candidate_path.name
        or result["candidate_sha256"] != candidate["candidate_sha256"]
        or result["staged_candidate_sha256"] != candidate_sha
        or candidate_sha != candidate["candidate_sha256"]
        or result["candidate_bytes"] != candidate_path.stat().st_size
        or result["candidate_bytes"] != candidate["candidate_bytes"]
        or result["comfy_lora_name"] != f"candidate-{candidate_sha}.safetensors"
        or result["dataset"] != evaluation_dataset_path
    ):
        raise ValueError("exact-score candidate/evaluator identity differs")
    _number(result["elapsed_s"], "exact-score elapsed_s", positive=True)
    fixture = _fixture_score_view(fixture_manifest)
    identity = fixture["evaluation_dataset_identity"]
    expected_rows = identity["rows"]
    if (
        result["dataset_sha256"] != identity["sha256"]
        or isinstance(result["image_count"], bool)
        or not isinstance(result["image_count"], int)
        or result["image_count"] <= 0
        or result["image_count"] != len(expected_rows)
        or not isinstance(result["scored_rows"], list)
        or not isinstance(result["text_guided_losses"], list)
        or not isinstance(result["blank_prompt_losses"], list)
        or len(result["scored_rows"]) != len(expected_rows)
        or len(result["text_guided_losses"]) != len(expected_rows)
        or len(result["blank_prompt_losses"]) != len(expected_rows)
    ):
        raise ValueError("exact-score dataset coverage differs")
    text: list[float] = []
    blank: list[float] = []
    for index, (expected, observed) in enumerate(
        zip(expected_rows, result["scored_rows"])
    ):
        observed = _object(observed, f"scored_rows[{index}]")
        loss_keys = {"text_guided_loss", "blank_prompt_loss"}
        _exact(observed, set(expected) | loss_keys, f"scored_rows[{index}]")
        observed_identity = {
            key: item for key, item in observed.items() if key not in loss_keys
        }
        if observed_identity != expected:
            raise ValueError("exact-score rows differ from evaluator order")
        text_loss = _bounded_number(observed["text_guided_loss"], "text-guided loss")
        blank_loss = _bounded_number(observed["blank_prompt_loss"], "blank loss")
        if text_loss != _bounded_number(
            result["text_guided_losses"][index], "text loss array"
        ) or blank_loss != _bounded_number(
            result["blank_prompt_losses"][index], "blank loss array"
        ):
            raise ValueError("exact-score row and loss arrays differ")
        text.append(text_loss)
        blank.append(blank_loss)
    text_mean = sum(text) / len(text)
    blank_mean = sum(blank) / len(blank)
    weight = _bounded_number(result["text_weight"], "text weight")
    weighted = weight * text_mean + (1 - weight) * blank_mean
    if (
        not math.isclose(
            _bounded_number(result["text_mean"], "text mean"),
            text_mean,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _bounded_number(result["blank_mean"], "blank mean"),
            blank_mean,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _bounded_number(result["weighted_loss"], "weighted loss"),
            weighted,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("exact-score aggregate losses do not recompute")
    observed_contract = evaluator_contract_from_result(result)
    if observed_contract != _evaluator_contract(evaluator_contract):
        raise ValueError("exact-score evaluator contract differs")
    _validate_source(result["source"])
    _runtime_source(result["runtime"])
    history = result["runtime"]["comfy_history"]
    prompt_count = len(expected_rows) * result["generations"] * 2
    if history["prompt_count"] != prompt_count:
        raise ValueError("exact-score prompt count differs")
    return {
        "weighted_loss": weighted,
        "text_mean": text_mean,
        "blank_mean": blank_mean,
        "row_identity_sha256": krea_provenance.canonical_sha256(expected_rows),
        "evaluator_contract_sha256": observed_contract["contract_sha256"],
        "dataset_sha256": identity["sha256"],
        "row_count": len(expected_rows),
        "prompt_count": prompt_count,
    }


def build_receipt(
    *,
    plan: dict[str, Any],
    family_id: str,
    candidate_path: Path,
    fixture_manifest: dict[str, Any],
    fixture_manifest_file_sha256: str,
    result_path: Path,
    status_file_sha256: str,
    evidence_manifest_file_sha256: str,
    completed_at_utc: str,
) -> dict[str, Any]:
    resolved = validate_plan(plan)
    family = _safe_id(family_id, "receipt family")
    candidates = {row["family_id"]: row for row in resolved["candidates"]}
    if family not in candidates:
        raise ValueError("receipt family is not in the approved score plan")
    manifest = _fixture_score_view(fixture_manifest)
    identity = manifest["evaluation_dataset_identity"]
    if (
        _sha(fixture_manifest_file_sha256, "fixture manifest file")
        != resolved["fixture_manifest"]["file_sha256"]
        or manifest["manifest_sha256"]
        != resolved["fixture_manifest"]["manifest_sha256"]
        or identity["sha256"] != resolved["evaluation_dataset_sha256"]
        or len(identity["rows"]) != resolved["evaluation_row_count"]
    ):
        raise ValueError("receipt fixture manifest differs from score plan")
    result_path = _safe_regular_file(result_path, "exact-score result")
    try:
        result = json.loads(result_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("exact-score result is not JSON") from exc
    computed = _validate_result(
        result,
        candidate_path=candidate_path,
        candidate=candidates[family],
        fixture_manifest=fixture_manifest,
        evaluator_contract=resolved["evaluator_contract"],
        evaluation_dataset_path=resolved["evaluation_dataset_path"],
    )
    body = {
        "schema": SCHEMA,
        "kind": RECEIPT_KIND,
        "score_plan_sha256": resolved["plan_sha256"],
        "phase": resolved["phase"],
        "cell_id": resolved["cell_id"],
        "fixture_id": resolved["fixture_id"],
        "seed_role": resolved["seed_role"],
        "family_id": family,
        "candidate_id": candidates[family]["candidate_id"],
        "candidate_sha256": candidates[family]["candidate_sha256"],
        "result": {
            "file_sha256": krea_provenance.file_sha256(result_path),
            "semantic_sha256": krea_provenance.canonical_sha256(result),
            **computed,
        },
        "status_file_sha256": _sha(status_file_sha256, "status file"),
        "evidence_manifest_file_sha256": _sha(
            evidence_manifest_file_sha256, "evidence manifest file"
        ),
        "completed_at_utc": completed_at_utc,
        "returncode": 0,
        "fallback_used": False,
        "release_authorized": False,
        "production_mutation_authorized": False,
    }
    receipt = {**body, "receipt_sha256": krea_provenance.canonical_sha256(body)}
    return validate_receipt(receipt, plan=resolved)


def validate_receipt(value: Any, *, plan: dict[str, Any]) -> dict[str, Any]:
    receipt = _object(value, "Stage-2 score receipt")
    keys = {
        "schema",
        "kind",
        "score_plan_sha256",
        "phase",
        "cell_id",
        "fixture_id",
        "seed_role",
        "family_id",
        "candidate_id",
        "candidate_sha256",
        "result",
        "status_file_sha256",
        "evidence_manifest_file_sha256",
        "completed_at_utc",
        "returncode",
        "fallback_used",
        "release_authorized",
        "production_mutation_authorized",
        "receipt_sha256",
    }
    _exact(receipt, keys, "Stage-2 score receipt")
    body = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    resolved = validate_plan(plan)
    candidates = {row["family_id"]: row for row in resolved["candidates"]}
    family = _safe_id(receipt["family_id"], "receipt family")
    if (
        receipt["schema"] != SCHEMA
        or receipt["kind"] != RECEIPT_KIND
        or receipt["receipt_sha256"] != krea_provenance.canonical_sha256(body)
        or receipt["score_plan_sha256"] != resolved["plan_sha256"]
        or family not in candidates
        or receipt["candidate_id"] != candidates[family]["candidate_id"]
        or receipt["candidate_sha256"] != candidates[family]["candidate_sha256"]
        or any(
            receipt[key] != resolved[key]
            for key in ("phase", "cell_id", "fixture_id", "seed_role")
        )
    ):
        raise ValueError("Stage-2 score receipt identity differs")
    result = _object(receipt["result"], "receipt result")
    _exact(
        result,
        {
            "file_sha256",
            "semantic_sha256",
            "weighted_loss",
            "text_mean",
            "blank_mean",
            "row_identity_sha256",
            "evaluator_contract_sha256",
            "dataset_sha256",
            "row_count",
            "prompt_count",
        },
        "receipt result",
    )
    for key in (
        "file_sha256",
        "semantic_sha256",
        "row_identity_sha256",
        "evaluator_contract_sha256",
        "dataset_sha256",
    ):
        _sha(result[key], f"receipt result {key}")
    if (
        result["evaluator_contract_sha256"]
        != resolved["evaluator_contract"]["contract_sha256"]
        or result["dataset_sha256"] != resolved["evaluation_dataset_sha256"]
        or isinstance(result["row_count"], bool)
        or not isinstance(result["row_count"], int)
        or result["row_count"] != resolved["evaluation_row_count"]
        or isinstance(result["prompt_count"], bool)
        or not isinstance(result["prompt_count"], int)
        or result["prompt_count"]
        != resolved["evaluation_row_count"]
        * resolved["evaluator_contract"]["generations"]
        * 2
    ):
        raise ValueError("receipt result contract differs")
    for key in ("weighted_loss", "text_mean", "blank_mean"):
        _bounded_number(result[key], f"receipt result {key}")
    _sha(receipt["status_file_sha256"], "receipt status file")
    _sha(receipt["evidence_manifest_file_sha256"], "receipt evidence manifest")
    krea_stage2_execution._utc(receipt["completed_at_utc"], "score completed_at_utc")
    if receipt["completed_at_utc"] <= resolved["created_at_utc"]:
        raise ValueError("Stage-2 score receipt predates its plan")
    if (
        receipt["returncode"] != 0
        or receipt["fallback_used"] is not False
        or receipt["release_authorized"] is not False
        or receipt["production_mutation_authorized"] is not False
    ):
        raise ValueError("Stage-2 score receipt failed or overclaims authority")
    return dict(receipt)


def validate_receipt_with_score_files(
    value: Any,
    *,
    plan: dict[str, Any],
    candidate_path: Path,
    fixture_manifest: dict[str, Any],
    fixture_manifest_file_sha256: str,
    result_path: Path,
) -> dict[str, Any]:
    """Replay result bytes instead of trusting a portable receipt summary."""

    resolved = validate_plan(plan)
    receipt = validate_receipt(value, plan=resolved)
    manifest = _fixture_score_view(fixture_manifest)
    identity = manifest["evaluation_dataset_identity"]
    if (
        _sha(fixture_manifest_file_sha256, "fixture manifest file")
        != resolved["fixture_manifest"]["file_sha256"]
        or manifest["manifest_sha256"]
        != resolved["fixture_manifest"]["manifest_sha256"]
        or identity["sha256"] != resolved["evaluation_dataset_sha256"]
        or len(identity["rows"]) != resolved["evaluation_row_count"]
    ):
        raise ValueError("receipt fixture manifest differs from score plan")
    candidates = {row["family_id"]: row for row in resolved["candidates"]}
    result_path = _safe_regular_file(result_path, "exact-score result")
    try:
        result = _object(json.loads(result_path.read_bytes()), "exact-score result")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("exact-score result is not JSON") from exc
    computed = _validate_result(
        result,
        candidate_path=candidate_path,
        candidate=candidates[receipt["family_id"]],
        fixture_manifest=fixture_manifest,
        evaluator_contract=resolved["evaluator_contract"],
        evaluation_dataset_path=resolved["evaluation_dataset_path"],
    )
    expected = {
        "file_sha256": krea_provenance.file_sha256(result_path),
        "semantic_sha256": krea_provenance.canonical_sha256(result),
        **computed,
    }
    if receipt["result"] != expected:
        raise ValueError("receipt result differs from recomputed exact-score bytes")
    return receipt


def build_aggregate(
    *, plan: dict[str, Any], receipts: list[dict[str, Any]], emitted_at_utc: str
) -> dict[str, Any]:
    resolved = validate_plan(plan)
    if not isinstance(receipts, list):
        raise ValueError("Stage-2 receipts must be an array")
    validated = [validate_receipt(row, plan=resolved) for row in receipts]
    validated.sort(key=lambda row: row["family_id"])
    expected = [row["family_id"] for row in resolved["candidates"]]
    if [row["family_id"] for row in validated] != expected:
        raise ValueError("Stage-2 aggregate lacks one exact receipt per candidate")
    body = {
        "schema": SCHEMA,
        "kind": AGGREGATE_KIND,
        "score_plan_sha256": resolved["plan_sha256"],
        "phase": resolved["phase"],
        "cell_id": resolved["cell_id"],
        "fixture_id": resolved["fixture_id"],
        "seed_role": resolved["seed_role"],
        "seed": resolved["seed"],
        "hours": resolved["hours"],
        "candidate_family_id": resolved["candidate_family_id"],
        "receipts": validated,
        "emitted_at_utc": emitted_at_utc,
        "all_candidates_scored": True,
        "fallback_used": False,
        "release_authorized": False,
        "production_mutation_authorized": False,
    }
    aggregate = {**body, "aggregate_sha256": krea_provenance.canonical_sha256(body)}
    return validate_aggregate(aggregate, plan=resolved)


def validate_aggregate(value: Any, *, plan: dict[str, Any]) -> dict[str, Any]:
    aggregate = _object(value, "Stage-2 score aggregate")
    keys = {
        "schema",
        "kind",
        "score_plan_sha256",
        "phase",
        "cell_id",
        "fixture_id",
        "seed_role",
        "seed",
        "hours",
        "candidate_family_id",
        "receipts",
        "emitted_at_utc",
        "all_candidates_scored",
        "fallback_used",
        "release_authorized",
        "production_mutation_authorized",
        "aggregate_sha256",
    }
    _exact(aggregate, keys, "Stage-2 score aggregate")
    body = {key: item for key, item in aggregate.items() if key != "aggregate_sha256"}
    resolved = validate_plan(plan)
    if (
        aggregate["schema"] != SCHEMA
        or aggregate["kind"] != AGGREGATE_KIND
        or aggregate["aggregate_sha256"] != krea_provenance.canonical_sha256(body)
        or aggregate["score_plan_sha256"] != resolved["plan_sha256"]
        or any(
            aggregate[key] != resolved[key]
            for key in (
                "phase",
                "cell_id",
                "fixture_id",
                "seed_role",
                "seed",
                "hours",
                "candidate_family_id",
            )
        )
    ):
        raise ValueError("Stage-2 score aggregate identity differs")
    if not isinstance(aggregate["receipts"], list):
        raise ValueError("Stage-2 aggregate receipts must be an array")
    receipts = [validate_receipt(row, plan=resolved) for row in aggregate["receipts"]]
    if [row["family_id"] for row in receipts] != [
        row["family_id"] for row in resolved["candidates"]
    ]:
        raise ValueError("Stage-2 aggregate candidate coverage differs")
    krea_stage2_execution._utc(aggregate["emitted_at_utc"], "aggregate emitted_at_utc")
    if receipts and aggregate["emitted_at_utc"] <= max(
        row["completed_at_utc"] for row in receipts
    ):
        raise ValueError("Stage-2 aggregate predates a score receipt")
    if (
        aggregate["all_candidates_scored"] is not True
        or aggregate["fallback_used"] is not False
        or aggregate["release_authorized"] is not False
        or aggregate["production_mutation_authorized"] is not False
    ):
        raise ValueError(
            "Stage-2 score aggregate is incomplete or overclaims authority"
        )
    return dict(aggregate)


def validate_aggregate_with_score_files(
    value: Any,
    *,
    plan: dict[str, Any],
    fixture_manifest: dict[str, Any],
    fixture_manifest_file_sha256: str,
    score_files_by_family: Mapping[str, Any],
) -> dict[str, Any]:
    """Require one live candidate/result pair for every aggregate receipt."""

    resolved = validate_plan(plan)
    aggregate = validate_aggregate(value, plan=resolved)
    families = [row["family_id"] for row in resolved["candidates"]]
    if not isinstance(score_files_by_family, Mapping) or set(
        score_files_by_family
    ) != set(families):
        raise ValueError("score files do not exhaust the aggregate families")
    receipts = {row["family_id"]: row for row in aggregate["receipts"]}
    for family in families:
        control = _object(score_files_by_family[family], f"score files {family}")
        _exact(control, {"candidate_path", "result_path"}, f"score files {family}")
        validate_receipt_with_score_files(
            receipts[family],
            plan=resolved,
            candidate_path=control["candidate_path"],
            fixture_manifest=fixture_manifest,
            fixture_manifest_file_sha256=fixture_manifest_file_sha256,
            result_path=control["result_path"],
        )
    return aggregate
