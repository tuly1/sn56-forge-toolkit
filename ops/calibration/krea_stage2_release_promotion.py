#!/usr/bin/env python3
"""Prove that a confirmed Krea profile is what env-unset production will run.

This is deliberately a post-decision, pre-release gate.  It replays the final
Stage-2 decision and all six boundary runs, then compares their validated
explicit-profile configs with config-only probes from one exact proposed
release image.  A successful proof is evidence for a later release review; it
cannot authorize a repository mutation, release, or deployment.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

import yaml

from forge import krea_calibration_profiles

try:
    from . import krea_fixture
    from . import krea_provenance
    from . import krea_stage2_decision
    from . import krea_stage2_execution
    from . import krea_stage2_production_identity
    from . import krea_stage2_training_evidence
except ImportError:  # pragma: no cover - direct execution/import support.
    import krea_fixture  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_stage2_decision  # type: ignore[no-redef]
    import krea_stage2_execution  # type: ignore[no-redef]
    import krea_stage2_production_identity  # type: ignore[no-redef]
    import krea_stage2_training_evidence  # type: ignore[no-redef]


KIND = "forge-krea-stage2-release-promotion-proof"
PROBE_KIND = "forge-krea-stage2-release-config-probe"
SCHEMA = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

BOUNDARY_CELLS = tuple(
    f"B-{hours}-{size}" for hours in ("0p5", "0p75", "1") for size in ("small", "large")
)
CALIBRATION_ENVIRONMENT = tuple(
    sorted(
        (
            krea_calibration_profiles.PROFILE_SELECTOR_ENV,
            krea_calibration_profiles.STAGE2_STEPS_ENV,
            krea_calibration_profiles.STAGE2_THROUGHPUT_SHA_ENV,
            krea_calibration_profiles.STAGE2_SEED_ENV,
            krea_calibration_profiles.STAGE2_PLAN_SHA_ENV,
            krea_calibration_profiles.STAGE2_RECEIPT_PATH_ENV,
            krea_calibration_profiles.STAGE2_TARGET_NUMERATOR_ENV,
            krea_calibration_profiles.STAGE2_TARGET_DENOMINATOR_ENV,
        )
    )
)
_CLEAN_MECHANICS = {
    "natural_completion": True,
    "planned_steps_completed": True,
    "upload_ready": True,
    "clean_telemetry": True,
    "decision_completed_before_export_reserve": True,
    "fallback_used": False,
}


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(
            f"{label} keys differ: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _image_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IMAGE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact sha256 image id")
    return value


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe identifier")
    return value


def _utc_value(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise ValueError(f"{label} must be canonical whole-second UTC")
    try:
        result = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{label} is not real UTC") from exc
    if result < datetime(2020, 1, 1, tzinfo=timezone.utc) or result > datetime.now(
        timezone.utc
    ) + timedelta(seconds=60):
        raise ValueError(f"{label} is outside accepted evidence time bounds")
    return result


def _canonical_file_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(krea_provenance.canonical_bytes(value) + b"\n").hexdigest()


def _safe_file(path_value: str | Path, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path_value))))
    current = path
    while True:
        try:
            info = current.lstat()
        except OSError as exc:
            raise ValueError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{label} has a symlink component")
        if current == path and not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{label} is not a regular file")
        if current == current.parent:
            break
        current = current.parent
    return path


def _read_file(path_value: str | Path, label: str) -> tuple[Path, bytes, str]:
    path = _safe_file(path_value, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ValueError(f"{label} could not be read") from exc
    raw = b"".join(chunks)
    stable = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    observed = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if stable != observed or len(raw) != before.st_size:
        raise ValueError(f"{label} changed while it was read")
    return path, raw, hashlib.sha256(raw).hexdigest()


def _load_canonical_json(
    path_value: str | Path, label: str
) -> tuple[Path, dict[str, Any], str]:
    path, raw, file_sha = _read_file(path_value, label)
    try:
        value = _object(json.loads(raw), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if raw != krea_provenance.canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} is not canonical JSON plus one newline")
    return path, value, file_sha


def _load_yaml(path_value: str | Path, label: str) -> tuple[Path, dict[str, Any], str]:
    path, raw, file_sha = _read_file(path_value, label)
    try:
        value = _object(yaml.safe_load(raw), label)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"{label} is not YAML") from exc
    return path, value, file_sha


def _process(config: Mapping[str, Any], label: str) -> dict[str, Any]:
    root = _object(config.get("config"), f"{label}.config")
    processes = root.get("process")
    if not isinstance(processes, list) or len(processes) != 1:
        raise ValueError(f"{label} must contain exactly one process")
    return _object(processes[0], f"{label}.process")


def _config_checkpoint_selection(
    meta: Mapping[str, Any],
    *,
    execution_plan: Mapping[str, Any],
    steps: int,
    save_every: int,
    label: str,
) -> dict[str, Any]:
    """Validate the stable checkpoint policy retained in production config."""

    binding = _object(
        meta.get("forge_krea_checkpoint_selection"),
        f"{label} checkpoint selection",
    )
    _exact(
        binding,
        {
            "schema",
            "mapping_rule",
            "target_fraction",
            "planned_steps",
            "selected_step",
            "candidate_steps",
        },
        f"{label} checkpoint selection",
    )
    target = _object(binding["target_fraction"], f"{label} checkpoint target fraction")
    _exact(
        target,
        {"numerator", "denominator"},
        f"{label} checkpoint target fraction",
    )
    numerator = target["numerator"]
    denominator = target["denominator"]
    selected_step = binding["selected_step"]
    candidate_steps = binding["candidate_steps"]
    if (
        binding["schema"] != 1
        or binding["mapping_rule"] != krea_stage2_execution._CHECKPOINT_MAPPING_RULE
        or isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or numerator <= 0
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
        or numerator > denominator
        or math.gcd(numerator, denominator) != 1
        or binding["planned_steps"] != steps
        or isinstance(selected_step, bool)
        or not isinstance(selected_step, int)
        or selected_step <= 0
        or not isinstance(candidate_steps, list)
    ):
        raise ValueError(f"{label} checkpoint selection is invalid")
    expected_candidates = list(range(save_every, steps, save_every))
    expected_candidates.append(steps)
    expected_candidates = sorted(set(expected_candidates))
    expected_selected = min(
        expected_candidates,
        key=lambda step: (abs(step * denominator - steps * numerator), step),
    )
    plan_selection = _object(
        execution_plan.get("checkpoint_selection"),
        "boundary execution checkpoint selection",
    )
    plan_target = _object(
        plan_selection.get("target_fraction"),
        "boundary execution checkpoint target",
    )
    if (
        execution_plan.get("planned_steps") != steps
        or binding["mapping_rule"] != plan_selection.get("mapping_rule")
        or target != plan_target
        or selected_step != plan_selection.get("selected_step")
        or binding["planned_steps"] != plan_selection.get("denominator_steps")
        or candidate_steps != expected_candidates
        or selected_step != expected_selected
    ):
        raise ValueError(f"{label} checkpoint selection differs from its plan")
    return deepcopy(binding)


def _normalize_explicit_config(
    value: Mapping[str, Any],
    *,
    selected_family: str,
    seed: int,
    execution_plan: Mapping[str, Any],
) -> tuple[bytes, int, int, dict[str, Any]]:
    config = deepcopy(dict(value))
    process = _process(config, "explicit boundary config")
    if process.pop("training_seed", None) != seed:
        raise ValueError("explicit boundary config lacks its exact Stage-2 seed")
    meta = _object(config.get("meta"), "explicit boundary config meta")
    binding = _object(
        meta.pop("forge_krea_calibration_profile", None),
        "explicit calibration profile binding",
    )
    if (
        binding.get("profile_id") != selected_family
        or binding.get("calibration_only") is not True
        or binding.get("release_selected") is not False
    ):
        raise ValueError("explicit config does not bind the selected family")
    train = _object(process.get("train"), "explicit boundary train config")
    save = _object(process.get("save"), "explicit boundary save config")
    steps = train.get("steps")
    save_every = save.get("save_every")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in (steps, save_every)
    ):
        raise ValueError("explicit config depth/save policy is invalid")
    checkpoint_selection = _config_checkpoint_selection(
        meta,
        execution_plan=execution_plan,
        steps=steps,
        save_every=save_every,
        label="explicit boundary config",
    )
    return (
        krea_provenance.canonical_bytes(config),
        steps,
        save_every,
        checkpoint_selection,
    )


def _normalize_release_config(
    value: Mapping[str, Any], *, execution_plan: Mapping[str, Any]
) -> tuple[bytes, int, int, dict[str, Any]]:
    config = deepcopy(dict(value))
    process = _process(config, "env-unset release config")
    meta = _object(config.get("meta"), "env-unset release config meta")
    if "forge_krea_calibration_profile" in meta or "training_seed" in process:
        raise ValueError("env-unset release config contains calibration-only state")
    train = _object(process.get("train"), "env-unset release train config")
    save = _object(process.get("save"), "env-unset release save config")
    steps = train.get("steps")
    save_every = save.get("save_every")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in (steps, save_every)
    ):
        raise ValueError("env-unset release depth/save policy is invalid")
    checkpoint_selection = _config_checkpoint_selection(
        meta,
        execution_plan=execution_plan,
        steps=steps,
        save_every=save_every,
        label="env-unset release config",
    )
    return (
        krea_provenance.canonical_bytes(config),
        steps,
        save_every,
        checkpoint_selection,
    )


def _identity_binding(record: Mapping[str, Any], file_sha256: str) -> dict[str, str]:
    identity = krea_stage2_production_identity.validate(dict(record))
    if _sha(file_sha256, "production identity file") != _canonical_file_sha(identity):
        raise ValueError("production identity file SHA-256 differs")
    return {
        "file_sha256": file_sha256,
        "production_identity_sha256": identity["production_identity_sha256"],
        "image_id": identity["container_image"]["image_id"],
        "forge_commit_sha1": identity["forge"]["commit_sha1"],
        "forge_tree_sha1": identity["forge"]["tree_sha1"],
    }


def validate_probe_receipt(
    value: Any,
    *,
    proposed_release_identity: Mapping[str, Any],
    config_path: str | Path,
) -> dict[str, Any]:
    """Validate one config-only receipt emitted by the exact proposed image."""

    receipt = _object(value, "release config probe receipt")
    keys = {
        "schema",
        "kind",
        "cell_id",
        "production_identity_sha256",
        "production_image_id",
        "probe_script_sha256",
        "calibration_environment",
        "image_spec",
        "config_file",
        "rendered_at_utc",
        "release_authorized",
        "production_mutation_authorized",
        "deployment_authorized",
        "receipt_sha256",
    }
    _exact(receipt, keys, "release config probe receipt")
    body = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    identity = krea_stage2_production_identity.validate(dict(proposed_release_identity))
    if (
        receipt["schema"] != SCHEMA
        or receipt["kind"] != PROBE_KIND
        or receipt["receipt_sha256"] != krea_provenance.canonical_sha256(body)
        or receipt["production_identity_sha256"]
        != identity["production_identity_sha256"]
        or receipt["production_image_id"] != identity["container_image"]["image_id"]
        or receipt["release_authorized"] is not False
        or receipt["production_mutation_authorized"] is not False
        or receipt["deployment_authorized"] is not False
    ):
        raise ValueError("release config probe identity or authority differs")
    _safe_id(receipt["cell_id"], "release config probe cell")
    _sha(receipt["probe_script_sha256"], "release config probe script")
    environment = _object(
        receipt["calibration_environment"], "release config probe environment"
    )
    if set(environment) != set(CALIBRATION_ENVIRONMENT) or any(
        environment[name] is not False for name in CALIBRATION_ENVIRONMENT
    ):
        raise ValueError("release config probe did not prove an env-unset surface")
    image_spec = _object(receipt["image_spec"], "release config probe image spec")
    _exact(
        image_spec,
        {
            "task_id",
            "model",
            "model_type",
            "expected_repo_name",
            "trigger_word",
            "hours",
            "training_row_count",
        },
        "release config probe image spec",
    )
    for key in ("task_id", "expected_repo_name"):
        _safe_id(image_spec[key], f"release config probe {key}")
    if image_spec["model"] != "krea/Krea-2-Raw" or image_spec["model_type"] != "krea2":
        raise ValueError("release config probe model differs")
    if (
        not isinstance(image_spec["hours"], str)
        or image_spec["hours"] not in {"0.5", "0.75", "1.0"}
        or isinstance(image_spec["training_row_count"], bool)
        or not isinstance(image_spec["training_row_count"], int)
        or image_spec["training_row_count"] <= 0
    ):
        raise ValueError("release config probe hours/row count differs")
    if image_spec["trigger_word"] is not None and (
        not isinstance(image_spec["trigger_word"], str)
        or not image_spec["trigger_word"]
        or image_spec["trigger_word"] != image_spec["trigger_word"].strip()
    ):
        raise ValueError("release config probe trigger word differs")
    config_file = _object(receipt["config_file"], "release config probe file")
    _exact(config_file, {"name", "bytes", "sha256"}, "release config probe file")
    path, raw, file_sha = _read_file(config_path, "release config probe output")
    if (
        config_file["name"] != path.name
        or isinstance(config_file["bytes"], bool)
        or not isinstance(config_file["bytes"], int)
        or config_file["bytes"] <= 0
        or config_file["bytes"] != len(raw)
        or _sha(config_file["sha256"], "release config probe output") != file_sha
    ):
        raise ValueError("release config probe output bytes differ")
    _utc_value(receipt["rendered_at_utc"], "release config probe rendered time")
    return dict(receipt)


def seal_probe_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in payload:
        raise ValueError("unsealed probe receipt includes a digest")
    body = dict(payload)
    return {**body, "receipt_sha256": krea_provenance.canonical_sha256(body)}


def _fixture_row_count(plan: Mapping[str, Any]) -> int:
    binding = _object(plan["fixture_manifest"], "boundary fixture binding")
    _path, value, file_sha = _load_canonical_json(
        binding["path"], "boundary fixture manifest"
    )
    manifest = krea_fixture.validate_manifest(value)
    if (
        file_sha != binding["file_sha256"]
        or manifest["manifest_sha256"] != binding["manifest_sha256"]
        or manifest["experimental_role"] != plan["fixture_id"]
    ):
        raise ValueError("boundary fixture manifest differs from its execution plan")
    rows = manifest["training_rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("boundary fixture has no training rows")
    return len(rows)


def _probe_control(
    controls: Mapping[str, Any], cell: str
) -> tuple[dict[str, Any], Path]:
    if cell not in controls:
        raise ValueError(f"release config probe is absent for {cell}")
    row = _object(controls[cell], f"release config probe control {cell}")
    _exact(row, {"receipt_path", "config_path"}, f"release config probe control {cell}")
    _path, receipt, _file_sha = _load_canonical_json(
        row["receipt_path"], f"release config probe receipt {cell}"
    )
    return receipt, Path(os.path.abspath(os.path.expanduser(str(row["config_path"]))))


def _completion_checkpoint_promotion(
    *,
    execution_plan: Mapping[str, Any],
    execution_approval_sha256: str,
    completion: Mapping[str, Any],
    private: Mapping[str, Any],
    candidate: Mapping[str, Any],
    run_evidence_file_sha256: str,
    run_evidence_sha256: str,
) -> dict[str, Any]:
    """Bind the score row to the frozen source promoted as ``last``."""

    plan_selection = _object(
        execution_plan.get("checkpoint_selection"),
        "boundary execution checkpoint selection",
    )
    target = _object(
        plan_selection.get("target_fraction"),
        "boundary execution checkpoint target",
    )
    planned_steps = execution_plan["planned_steps"]
    selected_step = plan_selection.get("selected_step")
    selected_name = (
        f"{execution_plan['expected_repo_name']}.safetensors"
        if selected_step == planned_steps
        else f"{execution_plan['expected_repo_name']}_{selected_step:09d}.safetensors"
    )
    if (
        candidate.get("execution_plan_sha256") != execution_plan["plan_sha256"]
        or candidate.get("execution_approval_sha256") != execution_approval_sha256
        or candidate.get("training_candidate_id")
        != execution_plan.get("training_candidate_id")
        or candidate.get("run_completion_sha256") != completion["completion_sha256"]
        or candidate.get("run_evidence_file_sha256") != run_evidence_file_sha256
        or candidate.get("run_evidence_sha256") != run_evidence_sha256
        or candidate.get("checkpoint_rule_sha256")
        != plan_selection.get("checkpoint_rule_sha256")
        or candidate.get("checkpoint_target_fraction") != target
        or candidate.get("checkpoint_mapping_rule")
        != plan_selection.get("mapping_rule")
        or candidate.get("step") != selected_step
        or candidate.get("fraction_numerator") != selected_step
        or candidate.get("fraction_denominator") != planned_steps
    ):
        raise ValueError("boundary score checkpoint policy differs from its plan")
    candidate_sha = _sha(candidate.get("candidate_sha256"), "boundary candidate")
    candidate_bytes = candidate.get("candidate_bytes")
    if (
        isinstance(candidate_bytes, bool)
        or not isinstance(candidate_bytes, int)
        or candidate_bytes <= 0
    ):
        raise ValueError("boundary candidate byte count is invalid")
    manifest = completion.get("artifact_manifest")
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("boundary completion lacks its artifact manifest")
    by_path: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(manifest):
        row = _object(raw, f"boundary artifact manifest row {index}")
        _exact(row, {"path", "bytes", "sha256"}, "boundary artifact manifest row")
        path = row["path"]
        if not isinstance(path, str) or path in by_path:
            raise ValueError("boundary artifact manifest path is invalid")
        by_path[path] = row
    selected_path = f"checkpoints/{selected_name}"
    last_path = "checkpoints/last.safetensors"
    selected = by_path.get(selected_path)
    promoted = by_path.get(last_path)
    selection_receipt = _object(
        private.get("checkpoint_selection"),
        "private checkpoint-selection receipt binding",
    )
    _exact(
        selection_receipt,
        {"file_sha256", "receipt_sha256"},
        "private checkpoint-selection receipt binding",
    )
    for key in selection_receipt:
        _sha(selection_receipt[key], f"private checkpoint-selection {key}")
    receipt_artifact = by_path.get("evidence/forge_checkpoint_selection.json")
    if (
        selected is None
        or promoted is None
        or selected
        != {
            "path": selected_path,
            "bytes": candidate_bytes,
            "sha256": candidate_sha,
        }
        or promoted
        != {
            "path": last_path,
            "bytes": candidate_bytes,
            "sha256": candidate_sha,
        }
        or receipt_artifact is None
        or receipt_artifact.get("bytes", 0) <= 0
        or receipt_artifact.get("sha256") != selection_receipt.get("file_sha256")
    ):
        raise ValueError(
            "boundary completion does not prove frozen checkpoint promotion"
        )
    return {
        "checkpoint_rule_sha256": plan_selection["checkpoint_rule_sha256"],
        "checkpoint_target_fraction": deepcopy(target),
        "checkpoint_mapping_rule": plan_selection["mapping_rule"],
        "selected_step": selected_step,
        "planned_steps": planned_steps,
        "selected_source_path": selected_path,
        "promoted_last_path": last_path,
        "selected_checkpoint_sha256": candidate_sha,
        "selected_checkpoint_bytes": candidate_bytes,
        "selected_source_equals_last": True,
        "checkpoint_selection_file_sha256": selection_receipt["file_sha256"],
        "checkpoint_selection_receipt_sha256": selection_receipt["receipt_sha256"],
    }


def _derive(
    *,
    decision_record: Mapping[str, Any],
    decision_file_sha256: str,
    plans: Mapping[str, dict[str, Any]],
    aggregates: Mapping[str, dict[str, Any]],
    cell_controls: Mapping[str, Mapping[str, Any]],
    authority_controls: Mapping[str, Any],
    proposed_release_identity: Mapping[str, Any],
    proposed_release_identity_file_sha256: str,
    probe_script_sha256: str,
    probe_controls_by_cell: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], datetime]:
    decision = krea_stage2_decision.validate_decision(
        dict(decision_record),
        plans=plans,
        aggregates=aggregates,
        cell_controls=cell_controls,
        authority_controls=authority_controls,
    )
    if (
        decision["outcome"] != "PASS"
        or decision["confirmation_passed"] is not True
        or not decision["gates"]
        or not all(decision["gates"].values())
    ):
        raise ValueError("release promotion requires a fully passing Stage-2 decision")
    decision_file_sha256 = _sha(decision_file_sha256, "Stage-2 decision file")
    if decision_file_sha256 != _canonical_file_sha(decision):
        raise ValueError("Stage-2 decision file SHA-256 differs")
    selected_family = _safe_id(
        decision["candidate_family_id"], "selected Stage-2 family"
    )
    confirmed = krea_stage2_production_identity.validate(
        dict(authority_controls["production_identity"])
    )
    confirmed_file_sha = authority_controls["production_identity_file_sha256"]
    confirmed_binding = _identity_binding(confirmed, confirmed_file_sha)
    if (
        decision["authority"]["production_identity_sha256"]
        != confirmed["production_identity_sha256"]
        or decision["authority"]["production_image_id"]
        != confirmed["container_image"]["image_id"]
    ):
        raise ValueError("Stage-2 decision differs from its confirmed identity")
    proposed = krea_stage2_production_identity.validate(dict(proposed_release_identity))
    proposed_binding = _identity_binding(
        proposed, proposed_release_identity_file_sha256
    )
    for field in (
        "dockerfile",
        "runtime_inputs",
        "runtime_inputs_sha256",
        "base_model",
        "runtime_contract",
    ):
        if proposed[field] != confirmed[field]:
            raise ValueError(
                f"proposed release changes non-promotion production input {field}"
            )
    probe_script_sha256 = _sha(probe_script_sha256, "release config probe script")
    if set(probe_controls_by_cell) != set(BOUNDARY_CELLS):
        raise ValueError("release config probes must cover the exact boundary matrix")
    boundary_rows: list[dict[str, Any]] = []
    latest = max(
        _utc_value(decision["decided_at_utc"], "Stage-2 decision time"),
        _utc_value(proposed["captured_at_utc"], "proposed identity capture time"),
    )
    for cell in BOUNDARY_CELLS:
        score_plan = _object(plans[cell], f"boundary score plan {cell}")
        controls = _object(cell_controls[cell], f"boundary cell controls {cell}")
        run_controls = _object(
            controls["run_controls_by_family"], f"boundary run controls {cell}"
        )
        if selected_family not in run_controls:
            raise ValueError(f"boundary cell {cell} lacks selected-family controls")
        run_control = _object(
            run_controls[selected_family], f"boundary run control {cell}"
        )
        execution_plan = krea_stage2_execution.validate_plan(
            _object(run_control["execution_plan"], f"execution plan {cell}")
        )
        approval = krea_stage2_execution.validate_approval(
            _object(run_control["execution_approval"], f"execution approval {cell}"),
            plan=execution_plan,
        )
        completion = krea_stage2_execution.validate_completion(
            _object(run_control["run_completion"], f"run completion {cell}"),
            plan=execution_plan,
            approval=approval,
        )
        if (
            execution_plan["phase"] != "boundary"
            or execution_plan["cell_id"] != cell
            or execution_plan["calibration_profile"] != selected_family
            or completion["mechanics"] != _CLEAN_MECHANICS
        ):
            raise ValueError(
                f"boundary cell {cell} is not a clean selected-profile run"
            )
        _evidence_path, run_evidence, run_evidence_file_sha = _load_canonical_json(
            run_control["run_evidence_path"], f"boundary run evidence {cell}"
        )
        run_evidence = krea_stage2_training_evidence.validate_run_evidence(
            run_evidence,
            plan=execution_plan,
            approval=approval,
            completion=completion,
        )
        private = krea_stage2_execution.validate_private_run_receipts(execution_plan)
        if (
            private["config_control"] != completion["config_control_receipt"]
            or private["training_terminal"] != completion["training_terminal_receipt"]
            or private.get("checkpoint_selection")
            != completion.get("checkpoint_selection_receipt")
        ):
            raise ValueError(f"boundary cell {cell} private receipts drifted")
        _explicit_path, explicit_config, explicit_file_sha = _load_yaml(
            private["effective_config_path"], f"explicit boundary config {cell}"
        )
        if explicit_file_sha != private["config_control"]["config_sha256"]:
            raise ValueError(f"boundary cell {cell} effective config bytes drifted")
        (
            explicit_normalized,
            explicit_steps,
            explicit_save,
            explicit_checkpoint_selection,
        ) = _normalize_explicit_config(
            explicit_config,
            selected_family=selected_family,
            seed=execution_plan["seed"],
            execution_plan=execution_plan,
        )
        if explicit_steps != execution_plan["planned_steps"]:
            raise ValueError(f"boundary cell {cell} config depth differs from plan")
        receipt, release_config_path = _probe_control(probe_controls_by_cell, cell)
        receipt = validate_probe_receipt(
            receipt,
            proposed_release_identity=proposed,
            config_path=release_config_path,
        )
        if (
            receipt["cell_id"] != cell
            or receipt["probe_script_sha256"] != probe_script_sha256
        ):
            raise ValueError(f"release config probe identity differs for {cell}")
        row_count = _fixture_row_count(execution_plan)
        expected_spec = {
            "task_id": execution_plan["task_id"],
            "model": execution_plan["model"],
            "model_type": execution_plan["model_type"],
            "expected_repo_name": execution_plan["expected_repo_name"],
            "trigger_word": execution_plan["trigger_word"],
            "hours": execution_plan["hours"],
            "training_row_count": row_count,
        }
        if receipt["image_spec"] != expected_spec:
            raise ValueError(f"release config probe inputs differ for {cell}")
        _release_path, release_config, release_file_sha = _load_yaml(
            release_config_path, f"env-unset release config {cell}"
        )
        (
            release_normalized,
            release_steps,
            release_save,
            release_checkpoint_selection,
        ) = _normalize_release_config(
            release_config,
            execution_plan=execution_plan,
        )
        if (
            release_normalized != explicit_normalized
            or release_steps != explicit_steps
            or release_save != explicit_save
            or release_checkpoint_selection != explicit_checkpoint_selection
        ):
            raise ValueError(
                f"env-unset release config differs from selected boundary {cell}"
            )
        candidates = {
            row["family_id"]: row
            for row in score_plan["candidates"]
            if isinstance(row, dict) and "family_id" in row
        }
        candidate = _object(
            candidates.get(selected_family), f"boundary candidate {cell}"
        )
        if candidate.get("mechanics") != _CLEAN_MECHANICS:
            raise ValueError(f"boundary score mechanics differ for {cell}")
        promotion = _completion_checkpoint_promotion(
            execution_plan=execution_plan,
            execution_approval_sha256=approval["approval_sha256"],
            completion=completion,
            private=private,
            candidate=candidate,
            run_evidence_file_sha256=run_evidence_file_sha,
            run_evidence_sha256=run_evidence["evidence_sha256"],
        )
        probe_path, probe_record, probe_file_sha = _load_canonical_json(
            probe_controls_by_cell[cell]["receipt_path"],
            f"release config probe receipt {cell}",
        )
        del probe_path
        if probe_record != receipt:
            raise ValueError(f"release config probe changed during replay for {cell}")
        rendered = _utc_value(receipt["rendered_at_utc"], f"probe time {cell}")
        if rendered < _utc_value(
            proposed["captured_at_utc"], "proposed identity capture time"
        ) or rendered <= _utc_value(decision["decided_at_utc"], "decision time"):
            raise ValueError(f"release config probe chronology differs for {cell}")
        latest = max(latest, rendered)
        boundary_rows.append(
            {
                "cell_id": cell,
                "score_plan_sha256": score_plan["plan_sha256"],
                "execution_plan_sha256": execution_plan["plan_sha256"],
                "execution_approval_sha256": approval["approval_sha256"],
                "completion_sha256": completion["completion_sha256"],
                "run_evidence_file_sha256": run_evidence_file_sha,
                "run_evidence_sha256": run_evidence["evidence_sha256"],
                "config_control_file_sha256": private["config_control"]["file_sha256"],
                "config_control_receipt_sha256": private["config_control"][
                    "receipt_sha256"
                ],
                "effective_config_file_sha256": explicit_file_sha,
                "training_terminal_file_sha256": private["training_terminal"][
                    "file_sha256"
                ],
                "candidate_sha256": _sha(
                    candidate["candidate_sha256"], "boundary candidate"
                ),
                "candidate_bytes": candidate["candidate_bytes"],
                "checkpoint_promotion": promotion,
                "planned_steps": explicit_steps,
                "save_every": explicit_save,
                "explicit_normalized_sha256": hashlib.sha256(
                    explicit_normalized
                ).hexdigest(),
                "probe_receipt_file_sha256": probe_file_sha,
                "probe_receipt_sha256": receipt["receipt_sha256"],
                "release_unset_raw_sha256": release_file_sha,
                "release_normalized_sha256": hashlib.sha256(
                    release_normalized
                ).hexdigest(),
                "normalized_bytes_equal": True,
            }
        )
    common = {
        "decision_binding": {
            "file_sha256": decision_file_sha256,
            "decision_sha256": decision["decision_sha256"],
        },
        "selected_family_id": selected_family,
        "confirmed_production_identity": confirmed_binding,
        "proposed_release_identity": proposed_binding,
        "probe_script_sha256": probe_script_sha256,
    }
    return common, boundary_rows, latest


def build_proof(
    *,
    decision_record: Mapping[str, Any],
    decision_file_sha256: str,
    plans: Mapping[str, dict[str, Any]],
    aggregates: Mapping[str, dict[str, Any]],
    cell_controls: Mapping[str, Mapping[str, Any]],
    authority_controls: Mapping[str, Any],
    proposed_release_identity: Mapping[str, Any],
    proposed_release_identity_file_sha256: str,
    probe_script_sha256: str,
    probe_controls_by_cell: Mapping[str, Any],
    created_at_utc: str,
) -> dict[str, Any]:
    common, boundary_rows, latest = _derive(
        decision_record=decision_record,
        decision_file_sha256=decision_file_sha256,
        plans=plans,
        aggregates=aggregates,
        cell_controls=cell_controls,
        authority_controls=authority_controls,
        proposed_release_identity=proposed_release_identity,
        proposed_release_identity_file_sha256=proposed_release_identity_file_sha256,
        probe_script_sha256=probe_script_sha256,
        probe_controls_by_cell=probe_controls_by_cell,
    )
    created = _utc_value(created_at_utc, "release promotion proof time")
    if created <= latest:
        raise ValueError("release promotion proof must postdate every input")
    gates = {
        "decision_passed": True,
        "exact_boundary_coverage": True,
        "all_boundary_runs_clean": True,
        "proposed_release_tree_clean": True,
        "proposed_image_exact": True,
        "calibration_environment_absent": True,
        "selected_family_baked_into_unset_release": True,
        "all_full_config_normalized_bytes_equal": True,
        "all_depth_and_save_policy_equal": True,
        "all_checkpoint_selection_policies_equal": True,
        "all_private_checkpoint_selection_receipts_bound": True,
        "all_selected_sources_promoted_to_last": True,
    }
    body = {
        "schema": SCHEMA,
        "kind": KIND,
        "created_at_utc": created_at_utc,
        **common,
        "boundary_cells": boundary_rows,
        "gates": gates,
        "release_review_required": True,
        "production_mutation_authorized": False,
        "release_authorized": False,
        "deployment_authorized": False,
    }
    proof = {**body, "proof_sha256": krea_provenance.canonical_sha256(body)}
    return validate_proof(
        proof,
        decision_record=decision_record,
        decision_file_sha256=decision_file_sha256,
        plans=plans,
        aggregates=aggregates,
        cell_controls=cell_controls,
        authority_controls=authority_controls,
        proposed_release_identity=proposed_release_identity,
        proposed_release_identity_file_sha256=proposed_release_identity_file_sha256,
        probe_script_sha256=probe_script_sha256,
        probe_controls_by_cell=probe_controls_by_cell,
    )


def validate_proof(
    value: Any,
    *,
    decision_record: Mapping[str, Any],
    decision_file_sha256: str,
    plans: Mapping[str, dict[str, Any]],
    aggregates: Mapping[str, dict[str, Any]],
    cell_controls: Mapping[str, Mapping[str, Any]],
    authority_controls: Mapping[str, Any],
    proposed_release_identity: Mapping[str, Any],
    proposed_release_identity_file_sha256: str,
    probe_script_sha256: str,
    probe_controls_by_cell: Mapping[str, Any],
) -> dict[str, Any]:
    proof = _object(value, "release promotion proof")
    keys = {
        "schema",
        "kind",
        "created_at_utc",
        "decision_binding",
        "selected_family_id",
        "confirmed_production_identity",
        "proposed_release_identity",
        "probe_script_sha256",
        "boundary_cells",
        "gates",
        "release_review_required",
        "production_mutation_authorized",
        "release_authorized",
        "deployment_authorized",
        "proof_sha256",
    }
    _exact(proof, keys, "release promotion proof")
    body = {key: item for key, item in proof.items() if key != "proof_sha256"}
    if (
        proof["schema"] != SCHEMA
        or proof["kind"] != KIND
        or proof["proof_sha256"] != krea_provenance.canonical_sha256(body)
        or proof["release_review_required"] is not True
        or proof["production_mutation_authorized"] is not False
        or proof["release_authorized"] is not False
        or proof["deployment_authorized"] is not False
    ):
        raise ValueError("release promotion proof identity or authority differs")
    common, boundary_rows, latest = _derive(
        decision_record=decision_record,
        decision_file_sha256=decision_file_sha256,
        plans=plans,
        aggregates=aggregates,
        cell_controls=cell_controls,
        authority_controls=authority_controls,
        proposed_release_identity=proposed_release_identity,
        proposed_release_identity_file_sha256=proposed_release_identity_file_sha256,
        probe_script_sha256=probe_script_sha256,
        probe_controls_by_cell=probe_controls_by_cell,
    )
    for key, expected in common.items():
        if proof[key] != expected:
            raise ValueError(f"release promotion proof binding differs at {key}")
    if proof["boundary_cells"] != boundary_rows:
        raise ValueError("release promotion boundary evidence does not recompute")
    expected_gates = {
        "decision_passed": True,
        "exact_boundary_coverage": True,
        "all_boundary_runs_clean": True,
        "proposed_release_tree_clean": True,
        "proposed_image_exact": True,
        "calibration_environment_absent": True,
        "selected_family_baked_into_unset_release": True,
        "all_full_config_normalized_bytes_equal": True,
        "all_depth_and_save_policy_equal": True,
        "all_checkpoint_selection_policies_equal": True,
        "all_private_checkpoint_selection_receipts_bound": True,
        "all_selected_sources_promoted_to_last": True,
    }
    if proof["gates"] != expected_gates:
        raise ValueError("release promotion gates differ")
    if _utc_value(proof["created_at_utc"], "release promotion proof time") <= latest:
        raise ValueError("release promotion proof chronology differs")
    return dict(proof)
