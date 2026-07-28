#!/usr/bin/env python3
"""Sealed pre-training Krea execution plan and human authorization.

This is stage two of the calibration evidence chain.  Source facts remain in
their own immutable records; this plan records concrete local choices.  It is
created with GPU execution disabled and becomes executable only through a
separate named-human approval that also binds a literal Linux/H100/systemd
certification record.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

try:
    from . import krea_budget
    from . import krea_dataset_identity
    from . import krea_fixture
    from . import krea_host_identity
    from . import krea_provenance
    from . import krea_public_source
except ImportError:  # pragma: no cover - direct script execution.
    import krea_budget  # type: ignore[no-redef]
    import krea_dataset_identity  # type: ignore[no-redef]
    import krea_fixture  # type: ignore[no-redef]
    import krea_host_identity  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_public_source  # type: ignore[no-redef]


_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_IMMUTABLE_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_PUBLIC_APPROVAL_KIND = "forge-krea-source-normalization-approval"
_INTERNAL_MODES = frozenset(
    {"deployed_control", "derived_matched_control", "internal_evidence_challenger"}
)
_DISCOVERY_KIND = "sn56-week5-krea-discovery-freeze"
_TIMING_PROBE_KIND = "forge-krea-bootstrap-timing-probe-plan"
_TIMING_APPROVAL_KIND = "forge-krea-bootstrap-timing-probe-approval"
_EXECUTION_APPROVAL_KIND = "forge-krea-pre-run-execution-approval"
_POSTRUN_CERTIFICATE_KIND = "forge-krea-post-run-natural-completion-certificate"


def _strict_utc(value: Any, label: str) -> str:
    value = _text(value, label)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        raise ValueError(f"{label} must be UTC with whole-second precision")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid UTC timestamp") from exc
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(value: dict[str, Any], required: set[str], label: str) -> None:
    missing = required - set(value)
    extra = set(value) - required
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return " ".join(value.split())


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_file(value: str | Path, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(value)))
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    return path


def _safe_directory(value: str | Path, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(value)))
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real directory: {path}")
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    return path


def _load_binding(value: Any, label: str) -> tuple[Path, dict[str, Any], str]:
    binding = _object(value, label)
    _exact(binding, {"path", "sha256"}, label)
    path = _safe_file(binding["path"], label)
    expected = _digest(binding["sha256"], f"{label}.sha256")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if raw != krea_provenance.canonical_bytes(document) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline")
    return path, _object(document, label), expected


def _file_binding(value: Any, label: str) -> tuple[Path, str]:
    binding = _object(value, label)
    _exact(binding, {"path", "sha256"}, label)
    path = _safe_file(binding["path"], label)
    digest = _digest(binding["sha256"], f"{label}.sha256")
    if krea_provenance.file_sha256(path) != digest:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path, digest


def _json_file_binding(
    value: Any, label: str, *, canonical: bool = False
) -> tuple[Path, dict[str, Any], str]:
    path, digest = _file_binding(value, label)
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    document = _object(document, label)
    if canonical and raw != krea_provenance.canonical_bytes(document) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline")
    return path, document, digest


def build_internal_basis(
    *,
    arm_id: str,
    mode: str,
    description: str,
    evidence_record: dict[str, str],
    release_commit: str,
    parent_arm_id: str | None,
) -> dict[str, Any]:
    if not _SAFE_ID.fullmatch(arm_id):
        raise ValueError("arm_id is invalid")
    if mode not in _INTERNAL_MODES:
        raise ValueError("internal basis mode is invalid")
    if mode == "derived_matched_control":
        if not isinstance(parent_arm_id, str) or not _SAFE_ID.fullmatch(parent_arm_id):
            raise ValueError("derived control requires a parent arm")
    elif parent_arm_id is not None:
        raise ValueError("only a derived control may name a parent arm")
    release_commit = _text(release_commit, "release_commit").lower()
    if not _GIT_SHA.fullmatch(release_commit):
        raise ValueError("release_commit must be a full Git commit")
    evidence_path, _, evidence_file_sha = _load_binding(
        evidence_record, "internal basis evidence record"
    )
    body = {
        "schema": 1,
        "kind": "forge-krea-internal-arm-basis",
        "arm_id": arm_id,
        "mode": mode,
        "description": _text(description, "description"),
        "evidence_record": {
            "path": str(evidence_path),
            "sha256": evidence_file_sha,
        },
        "release_commit": release_commit,
        "parent_arm_id": parent_arm_id,
    }
    return {**body, "basis_sha256": krea_provenance.canonical_sha256(body)}


def validate_internal_basis(value: dict[str, Any], *, arm_id: str) -> dict[str, Any]:
    value = _object(value, "internal arm basis")
    _exact(
        value,
        {
            "schema",
            "kind",
            "arm_id",
            "mode",
            "description",
            "evidence_record",
            "release_commit",
            "parent_arm_id",
            "basis_sha256",
        },
        "internal arm basis",
    )
    rebuilt = build_internal_basis(
        arm_id=value["arm_id"],
        mode=value["mode"],
        description=value["description"],
        evidence_record=value["evidence_record"],
        release_commit=value["release_commit"],
        parent_arm_id=value["parent_arm_id"],
    )
    if value != rebuilt or value["arm_id"] != arm_id:
        raise ValueError("internal arm basis is not canonical or arm-bound")
    return value


def _validate_public_approval(
    value: dict[str, Any], *, source_manifest: dict[str, Any]
) -> None:
    _exact(
        value,
        {
            "schema",
            "kind",
            "source_arm_id",
            "provenance_manifest_sha256",
            "reviewer_identity",
            "decision",
            "assertions",
        },
        "source-normalization approval",
    )
    if (
        value["schema"] != 1
        or value["kind"] != _PUBLIC_APPROVAL_KIND
        or value["source_arm_id"] != source_manifest["source_arm_id"]
        or value["provenance_manifest_sha256"] != source_manifest["manifest_sha256"]
        or value["decision"] != "approved"
    ):
        raise ValueError("source-normalization approval does not bind the source")
    krea_fixture.named_human(value["reviewer_identity"], "reviewer_identity")
    assertions = _object(value["assertions"], "source approval assertions")
    _exact(
        assertions,
        {
            "source_fields_reviewed",
            "unsupported_fields_reviewed",
            "adaptations_reviewed",
            "source_artifact_identity_reviewed",
            "claim_limits_reviewed",
        },
        "source approval assertions",
    )
    if any(item is not True for item in assertions.values()):
        raise ValueError("source-normalization approval assertions did not all pass")


def _arm_basis(
    value: Any, *, arm_id: str, execution_recipe: dict[str, Any]
) -> dict[str, Any]:
    basis = _object(value, "arm basis")
    mode = basis.get("mode")
    if mode == "public_submission":
        _exact(
            basis,
            {
                "mode",
                "source_provenance",
                "source_normalization_approval",
                "source_files",
            },
            "public arm basis",
        )
        _, source, source_file_sha = _load_binding(
            basis["source_provenance"], "source provenance"
        )
        source_files = _object(basis["source_files"], "public source files")
        _exact(
            source_files,
            {
                "source_config",
                "source_artifact",
                "field_ledger",
                "task_raw",
                "tournament_raw",
                "revision_manifest",
            },
            "public source files",
        )
        rebound: dict[str, tuple[Path, str]] = {
            name: _file_binding(binding, f"public source {name}")
            for name, binding in source_files.items()
        }
        # A self-digesting provenance JSON is not proof of its semantic linkage.
        # Re-run the primary-source validators against every bound byte source;
        # otherwise a fabricated semantic_linkage block could be self-hashed and
        # accepted without the official task/tournament/HF observations.
        krea_provenance.validate_manifest(
            source,
            source_config_path=rebound["source_config"][0],
            source_artifact_path=rebound["source_artifact"][0],
            field_ledger_path=rebound["field_ledger"][0],
            task_raw_path=rebound["task_raw"][0],
            tournament_raw_path=rebound["tournament_raw"][0],
            revision_manifest_path=rebound["revision_manifest"][0],
        )
        # ``validate_manifest`` proves that the manifest is internally
        # canonical and linked to the official records.  It cannot, by itself,
        # prove that a self-hashed normalized recipe was actually parsed from
        # the bound YAML/safetensors bytes.  Re-derive the machine-owned
        # semantics from those primary files and require exact agreement.  A
        # later human review assertion is intentionally separate and is the
        # only manifest field allowed to differ from the machine-produced
        # unreviewed record.
        derived_metadata = krea_public_source.build_metadata(
            arm_id,
            source_config_path=rebound["source_config"][0],
            source_artifact_path=rebound["source_artifact"][0],
            field_ledger_path=rebound["field_ledger"][0],
        )
        derived = krea_provenance.build_manifest(
            derived_metadata,
            source_config_path=rebound["source_config"][0],
            source_artifact_path=rebound["source_artifact"][0],
            field_ledger_path=rebound["field_ledger"][0],
            task_raw_path=rebound["task_raw"][0],
            tournament_raw_path=rebound["tournament_raw"][0],
            revision_manifest_path=rebound["revision_manifest"][0],
        )
        semantic_keys = {
            "schema",
            "kind",
            "source_arm_id",
            "source",
            "official_context",
            "files",
            "fields",
            "evaluator_sha",
            "matched_concept",
            "adaptation_target",
            "normalized_recipe",
        }
        mismatches = sorted(
            key for key in semantic_keys if source.get(key) != derived.get(key)
        )
        if mismatches:
            raise ValueError(
                "public source manifest differs from primary-byte re-derivation: "
                f"{mismatches}"
            )
        if source["source_arm_id"] != arm_id:
            raise ValueError("public source arm id differs from execution arm")
        _, source_approval, source_approval_sha = _load_binding(
            basis["source_normalization_approval"], "source approval"
        )
        _validate_public_approval(source_approval, source_manifest=source)
        normalized = krea_provenance.normalize_execution_recipe(
            execution_recipe, source_recipe=source["normalized_recipe"]
        )
        return {
            "mode": mode,
            "source_provenance_file_sha256": source_file_sha,
            "source_manifest_sha256": source["manifest_sha256"],
            "source_normalization_approval_sha256": source_approval_sha,
            "rebound_source_files": {
                name: {"path": str(path), "sha256": digest}
                for name, (path, digest) in sorted(rebound.items())
            },
            "normalized_execution_recipe": normalized,
        }
    if mode == "internal":
        _exact(basis, {"mode", "basis_record"}, "internal arm basis binding")
        _, record, record_file_sha = _load_binding(
            basis["basis_record"], "internal arm basis record"
        )
        validate_internal_basis(record, arm_id=arm_id)
        normalized = krea_provenance.normalize_recipe(execution_recipe)
        return {
            "mode": mode,
            "basis_record_file_sha256": record_file_sha,
            "basis_sha256": record["basis_sha256"],
            "basis_mode": record["mode"],
            "normalized_execution_recipe": normalized,
        }
    raise ValueError("arm basis mode must be public_submission or internal")


def _effective_recipe_values(recipe: dict[str, Any]) -> dict[str, Any]:
    fields = _object(recipe.get("fields"), "normalized recipe fields")
    return {
        name: _object(row, f"recipe field {name}").get("effective_value")
        for name, row in fields.items()
    }


def _discovery_allowed_axes(arm: dict[str, Any]) -> list[str]:
    arm_id = arm.get("id")
    if arm_id == "K0":
        return []
    if arm_id == "K1":
        return ["planned_steps", "save_cadence"]
    if arm_id in {"K2", "K4"}:
        return ["planned_steps", "save_cadence"]
    if arm_id == "K3":
        return ["dropout", "ema", "planned_steps", "save_cadence"]
    if arm_id == "K5":
        return ["learning_rate"]
    raise ValueError(f"discovery plan contains unsupported arm {arm_id!r}")


def validate_discovery_semantics(
    binding: Any,
    *,
    arm_id: str,
    fixture_id: str,
    fixture_manifest_sha256: str,
    training_pair_count: int,
    seed_role: str,
    seed: int,
    throughput_equivalence_class: str,
    execution_recipe: dict[str, Any],
    schedule_mode: str,
    predeclared_recipe_axes: list[str],
    basis_mode: str,
) -> dict[str, Any]:
    """Parse and bind the frozen experiment design, not just its file hash."""

    path, discovery, file_sha = _json_file_binding(binding, "discovery plan")
    if (
        discovery.get("schema") != 2
        or discovery.get("kind") != _DISCOVERY_KIND
        or discovery.get("model") != "krea/Krea-2-Raw"
        or discovery.get("model_type") != "krea2"
        or discovery.get("gpu_execution_authorized") is not False
    ):
        raise ValueError("unsupported or execution-enabled discovery plan")
    tasks = _object(discovery.get("discovery_tasks"), "discovery tasks")
    if fixture_id not in {"D1", "D2"} or fixture_id not in tasks:
        raise ValueError("discovery fixture id must be D1 or D2")
    task = _object(tasks[fixture_id], f"discovery task {fixture_id}")
    if task.get("fixture_split_manifest_sha256") != fixture_manifest_sha256:
        raise ValueError("fixture manifest is not the one frozen in discovery plan")
    pair_range = task.get("required_training_pair_range")
    if (
        not isinstance(pair_range, list)
        or len(pair_range) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) for item in pair_range
        )
        or pair_range[0] <= 0
        or pair_range[0] > pair_range[1]
        or not pair_range[0] <= training_pair_count <= pair_range[1]
    ):
        raise ValueError("fixture training-pair count escaped discovery range")
    if task.get("identity") is None:
        raise ValueError("discovery fixture identity is still unset")

    expected_seed_key = {
        "A": "training_seed_a",
        "B": "training_seed_b_contingency",
    }.get(seed_role)
    if expected_seed_key is None or discovery.get(expected_seed_key) != seed:
        raise ValueError("execution seed does not match its frozen discovery role")
    arms = discovery.get("arms")
    if not isinstance(arms, list):
        raise ValueError("discovery arms must be an array")
    matches = [row for row in arms if isinstance(row, dict) and row.get("id") == arm_id]
    if len(matches) != 1:
        raise ValueError("execution arm is not unique in the discovery plan")
    arm = matches[0]
    if arm.get("throughput_equivalence_class") != throughput_equivalence_class:
        raise ValueError("execution arm escaped its frozen throughput class")
    expected_basis = "public_submission" if arm_id in {"K2", "K3", "K4"} else "internal"
    if basis_mode != expected_basis:
        raise ValueError("arm basis mode contradicts the frozen arm source")
    expected_mode = "release_control" if arm_id == "K0" else "measured_budget_fill"
    if schedule_mode != expected_mode:
        raise ValueError("schedule mode contradicts the frozen arm depth policy")
    expected_axes = _discovery_allowed_axes(arm)
    if predeclared_recipe_axes != expected_axes:
        raise ValueError(
            "execution axes differ from the frozen arm contract: "
            f"expected={expected_axes}, actual={predeclared_recipe_axes}"
        )

    values = _effective_recipe_values(execution_recipe)
    field_map = {
        "learning_rate": "lr",
        "rank": "rank",
        "alpha": "alpha",
        "optimizer": "optimizer",
        "loss": "loss",
        "dropout": "dropout",
    }
    for recipe_name, arm_name in field_map.items():
        expected = arm.get(arm_name)
        if expected is None and arm_id == "K3" and arm_name == "dropout":
            expected = _object(
                arm.get("predeclared_local_values"), "K3 local values"
            ).get("dropout")
        if expected is None or values.get(recipe_name) != expected:
            raise ValueError(
                f"recipe field {recipe_name} contradicts discovery arm {arm_id}"
            )
    expected_guidance = arm.get("guidance")
    guidance = values.get("guidance")
    if guidance != {"enabled": True, "scale": expected_guidance}:
        raise ValueError("recipe guidance contradicts the discovery arm")
    expected_ema = arm.get("ema")
    if expected_ema is None and arm_id == "K3":
        expected_ema = _object(
            arm.get("predeclared_local_values"), "K3 local values"
        ).get("ema")
    ema = values.get("ema")
    if (
        not isinstance(ema, dict)
        or ema.get("enabled") is not expected_ema
        or set(ema) != {"enabled", "decay"}
    ):
        raise ValueError("recipe EMA contradicts the discovery arm")
    if arm_id == "K4":
        optimizer_parameters = values.get("optimizer_parameters")
        expected_parameters = {
            "min_lr": arm["min_lr"],
            "max_lr": arm["max_lr"],
            "lr_bump": arm["lr_bump"],
        }
        if not isinstance(optimizer_parameters, dict) or any(
            optimizer_parameters.get(key) != expected
            for key, expected in expected_parameters.items()
        ):
            raise ValueError("Automagic parameters contradict the frozen K4 arm")
    return {
        "path": path,
        "file_sha256": file_sha,
        "document": discovery,
        "arm": arm,
        "allowed_axes": expected_axes,
        "fixture": task,
        "seed_role": seed_role,
    }


def _schedule(
    value: Any,
    *,
    recipe: dict[str, Any],
    budget_plan: dict[str, Any],
    profile: krea_budget.ThroughputProfile,
) -> dict[str, Any]:
    schedule = _object(value, "schedule")
    _exact(
        schedule,
        {
            "mode",
            "planned_steps",
            "save_every",
            "candidate_steps",
            "required_landmarks",
            "landmark_policy",
        },
        "schedule",
    )
    if schedule["mode"] not in {"release_control", "measured_budget_fill"}:
        raise ValueError("unsupported schedule mode")
    for key in ("planned_steps", "save_every"):
        if (
            isinstance(schedule[key], bool)
            or not isinstance(schedule[key], int)
            or schedule[key] <= 0
        ):
            raise ValueError(f"schedule.{key} must be a positive integer")
    planned = schedule["planned_steps"]
    cadence = schedule["save_every"]
    expected_candidates = list(range(cadence, planned, cadence)) + [planned]
    if schedule["candidate_steps"] != expected_candidates:
        raise ValueError("candidate steps do not match uniform save cadence plus final")
    landmarks = schedule["required_landmarks"]
    if (
        not isinstance(landmarks, list)
        or any(
            isinstance(step, bool) or not isinstance(step, int) or step <= 0
            for step in landmarks
        )
        or landmarks != sorted(set(landmarks))
    ):
        raise ValueError("schedule landmarks are invalid")
    if schedule["landmark_policy"] not in {"none", "preserve_if_budget_safe"}:
        raise ValueError("unsupported landmark policy")
    if schedule["landmark_policy"] == "none" and landmarks:
        raise ValueError("landmarks require preserve_if_budget_safe")
    if any(step <= planned and step not in expected_candidates for step in landmarks):
        raise ValueError("a budget-safe required landmark is absent from candidates")
    fields = recipe["fields"]
    if (
        fields["planned_steps"]["effective_value"] != planned
        or fields["save_cadence"]["effective_value"] != cadence
    ):
        raise ValueError("schedule contradicts execution recipe")
    maximum = budget_plan.get("max_affordable_steps")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise ValueError("budget plan lacks max_affordable_steps")
    if schedule["mode"] == "measured_budget_fill" and planned != maximum:
        raise ValueError("budget-fill schedule does not fill the measured plan")
    if schedule["mode"] == "measured_budget_fill" and (
        cadence != budget_plan.get("save_every")
        or expected_candidates
        != [row["step"] for row in budget_plan.get("actual_candidates", [])]
    ):
        raise ValueError(
            "budget-fill schedule differs from the measured budget cadence"
        )
    if schedule["mode"] == "release_control" and planned > maximum:
        raise ValueError("release-control schedule does not fit the measured plan")
    # A release-control arm can intentionally retain a historical depth and
    # cadence.  Charge *that actual cadence* rather than assuming the discovery
    # planner's ceil(steps/8) schedule.  This closes the K0 under-accounting
    # path where many extra saves could otherwise pass planned<=maximum.
    periodic_save_count = len(range(cadence, planned + 1, cadence))
    hard = float(budget_plan["hard_budget_s"])
    available = (
        hard
        - profile.startup_upper_bound_s
        - profile.selection_scoring_reserve_s
        - max(
            profile.framework_stop_boundary_s,
            profile.finalization_reserve_s + profile.upload_reserve_s,
        )
    )
    charged = (
        planned * profile.update_upper_bound_s
        + periodic_save_count * profile.save_upper_bound_s
    )
    if not math.isfinite(available) or charged > available:
        raise ValueError("actual release schedule exceeds the measured stop boundary")
    maximum_save_fraction = float(
        budget_plan["accounting"]["maximum_save_overhead_fraction"]
    )
    save_fraction = (
        periodic_save_count * profile.save_upper_bound_s / available
        if available > 0
        else math.inf
    )
    if save_fraction > maximum_save_fraction:
        raise ValueError("actual release cadence exceeds the sealed save-I/O cap")
    return dict(schedule)


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    plan = _object(plan, "execution plan")
    _exact(
        plan,
        {
            "schema",
            "kind",
            "arm_id",
            "task_id",
            "expected_repo_name",
            "discovery_plan",
            "discovery_fixture_id",
            "seed_role",
            "fixture_manifest",
            "fixture_approval",
            "training_archive",
            "evaluation_dataset",
            "arm_basis",
            "execution_recipe",
            "throughput_profile",
            "timing_evidence",
            "host_execution_manifest",
            "budget_plan",
            "budget_plan_sha256",
            "schedule",
            "base_model",
            "seed",
            "runtime_identity_sha256",
            "execution_envelope_sha256",
            "throughput_equivalence_class",
            "predeclared_recipe_axes",
            "in_task_proxy_selection",
            "runner_sha256",
            "gpu_execution_authorized",
            "plan_sha256",
        },
        "execution plan",
    )
    if plan["schema"] != 2 or plan["kind"] != "forge-krea-pretraining-execution-plan":
        raise ValueError("unsupported execution plan")
    for key in ("arm_id", "task_id", "expected_repo_name"):
        if not isinstance(plan[key], str) or not _SAFE_ID.fullmatch(plan[key]):
            raise ValueError(f"execution plan {key} is invalid")
    if plan["gpu_execution_authorized"] is not False:
        raise ValueError("execution plan itself must keep GPU authorization false")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan["plan_sha256"] != krea_provenance.canonical_sha256(body):
        raise ValueError("execution plan digest mismatch")

    _, fixture, fixture_file_sha = _load_binding(
        plan["fixture_manifest"], "fixture manifest"
    )
    krea_fixture.validate_manifest(fixture)
    _, fixture_approval, _ = _load_binding(plan["fixture_approval"], "fixture approval")
    krea_fixture.validate_approval(fixture_approval, fixture_manifest=fixture)
    archive_path, archive_sha = _file_binding(
        plan["training_archive"], "training archive"
    )
    if (
        archive_sha != fixture["training_archive"]["sha256"]
        or archive_path.stat().st_size != fixture["training_archive"]["bytes"]
    ):
        raise ValueError("training archive differs from the approved fixture")
    evaluation = _object(plan["evaluation_dataset"], "evaluation dataset")
    _exact(evaluation, {"path", "sha256"}, "evaluation dataset")
    evaluation_path = _safe_directory(evaluation["path"], "evaluation dataset")
    evaluation_sha = _digest(evaluation["sha256"], "evaluation dataset sha256")
    expected_identity = fixture["evaluation_dataset_identity"]
    observed_identity = krea_dataset_identity.capture_dataset(
        evaluation_path,
        list_supported_images=lambda _root, _extensions: list(
            expected_identity["evaluator_order"]
        ),
        extensions=tuple(fixture["tool_identity"]["extensions"]),
    )
    if (
        observed_identity != expected_identity
        or evaluation_sha != expected_identity["sha256"]
    ):
        raise ValueError("evaluation dataset differs from the approved fixture")
    recipe = _arm_basis(
        plan["arm_basis"],
        arm_id=plan["arm_id"],
        execution_recipe=plan["execution_recipe"],
    )["normalized_execution_recipe"]
    profile_path, profile_sha = _file_binding(
        plan["throughput_profile"], "throughput profile"
    )
    try:
        profile = json.loads(profile_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("throughput profile is not JSON") from exc
    if profile.get("profile_sha256") is None:
        raise ValueError("throughput profile lacks its self digest")
    validated_profile = krea_budget.load_throughput_profile(profile)
    timing_evidence = _object(plan["timing_evidence"], "timing evidence")
    _exact(
        timing_evidence,
        {
            "raw_sample_manifest",
            "margin_policy",
            "end_to_end_validation",
            "probe_contract",
            "measurement_captures",
            "heldout_captures",
            "heldout_run_records",
        },
        "timing evidence",
    )
    _, raw_samples, raw_samples_file_sha = _load_binding(
        timing_evidence["raw_sample_manifest"], "raw timing sample manifest"
    )
    _, margin_policy, margin_policy_file_sha = _load_binding(
        timing_evidence["margin_policy"], "timing margin policy"
    )
    _, end_to_end, end_to_end_file_sha = _load_binding(
        timing_evidence["end_to_end_validation"], "end-to-end timing validation"
    )
    _, probe_contract, probe_contract_file_sha = _load_binding(
        timing_evidence["probe_contract"], "timing probe contract"
    )
    validate_timing_probe_plan(probe_contract)
    try:
        from . import krea_timing_probe
    except ImportError:  # pragma: no cover - direct script execution.
        import krea_timing_probe  # type: ignore[no-redef]

    def evidence_array(name: str) -> list[tuple[dict[str, Any], str]]:
        bindings = timing_evidence[name]
        if not isinstance(bindings, list) or not bindings:
            raise ValueError(f"timing evidence {name} must be non-empty")
        rows: list[tuple[dict[str, Any], str]] = []
        for index, binding in enumerate(bindings):
            _, document, file_sha = _load_binding(
                binding, f"timing evidence {name}[{index}]"
            )
            rows.append((document, file_sha))
        return rows

    measurement_capture_rows = evidence_array("measurement_captures")
    heldout_capture_rows = evidence_array("heldout_captures")
    heldout_run_rows = evidence_array("heldout_run_records")
    recomputed_raw = krea_timing_probe.raw_from_captures(
        [document for document, _ in measurement_capture_rows]
    )
    recomputed_e2e = krea_timing_probe.end_to_end_from_records(
        [document for document, _ in heldout_capture_rows], heldout_run_rows
    )
    if recomputed_raw != raw_samples or recomputed_e2e != end_to_end:
        raise ValueError("timing summaries differ from their bound producer records")
    recomputed_profile = krea_budget.seal_throughput_profile_from_evidence(
        raw_sample_manifest=raw_samples,
        margin_policy=margin_policy,
        end_to_end_validation=end_to_end,
        framework_stop_boundary_s=profile["framework_stop_boundary_s"],
        framework_stop_boundary_source_sha256=profile[
            "framework_stop_boundary_source_sha256"
        ],
        selection_mode=profile["selection_mode"],
        selection_scorer_identity_sha256=profile["selection_scorer_identity_sha256"],
        selection_scoring_reserve_s=profile["selection_scoring_reserve_s"],
    )
    if recomputed_profile != profile:
        raise ValueError(
            "throughput profile was not recomputed from bound raw evidence"
        )
    _, host_manifest, host_manifest_file_sha = _load_binding(
        plan["host_execution_manifest"], "host execution manifest"
    )
    krea_host_identity.validate_manifest(host_manifest)
    if profile_sha != plan["throughput_profile"]["sha256"]:
        raise ValueError("throughput profile binding mismatch")
    budget_plan = _object(plan["budget_plan"], "budget plan")
    if plan["budget_plan_sha256"] != krea_provenance.canonical_sha256(budget_plan):
        raise ValueError("budget plan digest mismatch")
    if budget_plan.get("profile_sha256") != profile.get("profile_sha256"):
        raise ValueError("budget plan is not bound to the throughput profile")
    try:
        recomputed_budget = krea_budget.plan_budget(
            validated_profile, hard_budget_s=float(budget_plan["hard_budget_s"])
        ).to_record()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("budget plan cannot be recomputed") from exc
    if recomputed_budget != budget_plan:
        raise ValueError("budget plan differs from the measured planner output")
    if (
        profile.get("selection_mode") != "offline_post_training"
        or profile.get("selection_scoring_reserve_s") != 0
    ):
        raise ValueError("discovery profile must reserve no in-task proxy scorer")
    if plan["in_task_proxy_selection"] != {"enabled": False, "reserve_s": 0}:
        raise ValueError("discovery execution must disable in-task proxy selection")
    for key in (
        "runtime_identity_sha256",
        "execution_envelope_sha256",
        "runner_sha256",
    ):
        _digest(plan[key], key)
    profile_envelope = _object(
        profile.get("execution_envelope"), "throughput execution envelope"
    )
    if (
        profile_envelope.get("runtime_identity_sha256")
        != plan["runtime_identity_sha256"]
    ):
        raise ValueError("profile/runtime identity mismatch")
    # New measured profiles must carry the complete execution envelope.  Old
    # Day-0 profiles fail closed rather than being silently reinterpreted.
    if (
        profile_envelope.get("execution_envelope_sha256")
        != plan["execution_envelope_sha256"]
    ):
        raise ValueError("profile/execution-envelope mismatch")
    if (
        profile_envelope.get("equivalence_class")
        != plan["throughput_equivalence_class"]
    ):
        raise ValueError("profile throughput-equivalence class mismatch")
    if validated_profile.execution_envelope.to_record() != profile_envelope:
        raise ValueError("throughput execution envelope did not normalize exactly")
    if (
        profile_envelope.get("host_execution_identity_sha256")
        != host_manifest["host_execution_identity_sha256"]
    ):
        raise ValueError("profile/host execution identity mismatch")
    if float(profile.get("framework_stop_boundary_s", -1)) < 225.0:
        raise ValueError("profile does not cover Forge's 225-second stop boundary")
    _schedule(
        plan["schedule"],
        recipe=recipe,
        budget_plan=budget_plan,
        profile=validated_profile,
    )
    seed = plan["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("execution seed is invalid")
    base = _object(plan["base_model"], "base_model")
    _exact(
        base,
        {
            "model_id",
            "revision",
            "training_identity_sha256",
            "evaluation_assets",
        },
        "base_model",
    )
    if (
        base["model_id"] != "krea/Krea-2-Raw"
        or not isinstance(base["revision"], str)
        or not _IMMUTABLE_REVISION.fullmatch(base["revision"])
    ):
        raise ValueError("base model identity is not immutable Krea")
    _digest(base["training_identity_sha256"], "base training identity")
    assets = _object(base["evaluation_assets"], "base evaluation assets")
    if set(assets) != {"diffusion_model", "text_encoder", "vae"}:
        raise ValueError("base assets must bind diffusion model, text encoder, and VAE")
    for name, asset in assets.items():
        asset = _object(asset, f"base asset {name}")
        _exact(asset, {"canonical_path", "sha256", "bytes"}, f"base asset {name}")
        _text(asset["canonical_path"], f"base asset {name}.canonical_path")
        _digest(asset["sha256"], f"base asset {name}.sha256")
        if (
            isinstance(asset["bytes"], bool)
            or not isinstance(asset["bytes"], int)
            or asset["bytes"] <= 0
        ):
            raise ValueError(f"base asset {name}.bytes is invalid")
    fields = recipe["fields"]
    values = {name: row["effective_value"] for name, row in fields.items()}
    if values["submitted_step"] is not None or values["selector"] is not None:
        raise ValueError("pretraining plan may not choose a checkpoint or selector")
    envelope = validated_profile.execution_envelope
    guidance = values["guidance"]
    expected_profile_fields = {
        "network_rank": values["rank"],
        "network_alpha": values["alpha"],
        "optimizer": values["optimizer"],
        "optimizer_config_sha256": krea_provenance.canonical_sha256(
            values["optimizer_parameters"]
        ),
        "loss": values["loss"],
        "differential_guidance_enabled": guidance["enabled"],
        "guidance_scale": guidance["scale"],
        "training_pair_count": len(fixture["training_rows"]),
        "training_dataset_shape_sha256": fixture["training_dataset_shape_sha256"],
        "gradient_accumulation_steps": values["gradient_accumulation"],
        "data_parallel_replicas": 1,
        "base_model_identity_sha256": base["training_identity_sha256"],
    }
    denominator = values["gradient_accumulation"]
    micro_batch = values["effective_batch"] // denominator
    if micro_batch * denominator != values["effective_batch"]:
        raise ValueError("execution effective batch is not integral")
    expected_profile_fields["micro_batch_size"] = micro_batch
    mismatches = {
        key: {"expected": expected, "profile": getattr(envelope, key)}
        for key, expected in expected_profile_fields.items()
        if getattr(envelope, key) != expected
    }
    if mismatches:
        raise ValueError(f"recipe/fixture/base escaped measured profile: {mismatches}")
    runner_path = Path(__file__).with_name("run_krea_ladder.py").resolve(strict=True)
    if krea_provenance.file_sha256(runner_path) != plan["runner_sha256"]:
        raise ValueError("execution plan runner SHA differs from local runner")
    axes = plan["predeclared_recipe_axes"]
    if (
        not isinstance(axes, list)
        or axes != sorted(set(axes))
        or any(axis not in recipe["fields"] for axis in axes)
        or any(axis in {"submitted_step", "selector"} for axis in axes)
    ):
        raise ValueError("predeclared recipe axes are invalid")
    basis_mode = _object(plan["arm_basis"], "arm basis").get("mode")
    discovery = validate_discovery_semantics(
        plan["discovery_plan"],
        arm_id=plan["arm_id"],
        fixture_id=plan["discovery_fixture_id"],
        fixture_manifest_sha256=fixture["manifest_sha256"],
        training_pair_count=len(fixture["training_rows"]),
        seed_role=plan["seed_role"],
        seed=seed,
        throughput_equivalence_class=plan["throughput_equivalence_class"],
        execution_recipe=recipe,
        schedule_mode=plan["schedule"]["mode"],
        predeclared_recipe_axes=axes,
        basis_mode=basis_mode,
    )
    if (
        probe_contract["probe_contract_sha256"] != raw_samples["probe_contract_sha256"]
        or probe_contract["probe_contract_sha256"]
        != end_to_end["probe_contract_sha256"]
        or probe_contract["throughput_equivalence_class"]
        != plan["throughput_equivalence_class"]
        or probe_contract["discovery_fixture_id"] != plan["discovery_fixture_id"]
        or probe_contract["execution_envelope"]["execution_envelope_sha256"]
        != plan["execution_envelope_sha256"]
    ):
        raise ValueError("final execution plan escaped its bootstrap timing probe")
    return {
        "fixture": fixture,
        "fixture_manifest_file_sha256": fixture_file_sha,
        "execution_recipe": recipe,
        "training_archive_path": archive_path,
        "evaluation_dataset_path": evaluation_path,
        "throughput_profile_path": profile_path,
        "throughput_profile": profile,
        "host_execution_manifest": host_manifest,
        "host_execution_manifest_file_sha256": host_manifest_file_sha,
        "discovery": discovery,
        "timing_evidence": {
            "raw_sample_manifest_file_sha256": raw_samples_file_sha,
            "margin_policy_file_sha256": margin_policy_file_sha,
            "end_to_end_validation_file_sha256": end_to_end_file_sha,
            "probe_contract_file_sha256": probe_contract_file_sha,
            "measurement_capture_file_sha256": [
                digest for _, digest in measurement_capture_rows
            ],
            "heldout_capture_file_sha256": [
                digest for _, digest in heldout_capture_rows
            ],
            "heldout_run_record_file_sha256": [
                digest for _, digest in heldout_run_rows
            ],
        },
        "schedule": plan["schedule"],
    }


def seal_plan(payload: dict[str, Any]) -> dict[str, Any]:
    if "plan_sha256" in payload:
        raise ValueError("unsealed plan payload must not contain plan_sha256")
    plan = {**payload, "plan_sha256": krea_provenance.canonical_sha256(payload)}
    validate_plan(plan)
    return plan


def validate_timing_probe_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate the executable pre-profile probe contract.

    This contract intentionally has no throughput profile, budget-derived arm
    depth, or post-run certificate.  It is the acyclic first-GPU entry point:
    approved fixture + representative recipe + host capability -> raw timings.
    """

    plan = _object(plan, "timing probe plan")
    _exact(
        plan,
        {
            "schema",
            "kind",
            "arm_id",
            "task_id",
            "expected_repo_name",
            "discovery_plan",
            "discovery_fixture_id",
            "seed_role",
            "seed",
            "fixture_manifest",
            "fixture_approval",
            "training_archive",
            "arm_basis",
            "execution_recipe",
            "host_execution_manifest",
            "base_model",
            "runtime_identity_sha256",
            "execution_envelope",
            "throughput_equivalence_class",
            "predeclared_recipe_axes",
            "probe_schedule",
            "command_argv",
            "runner_sha256",
            "measurement_tool_sha256",
            "gpu_execution_authorized",
            "probe_contract_sha256",
        },
        "timing probe plan",
    )
    if (
        plan["schema"] != 1
        or plan["kind"] != _TIMING_PROBE_KIND
        or plan["gpu_execution_authorized"] is not False
    ):
        raise ValueError("unsupported or self-authorized timing probe plan")
    body = {key: value for key, value in plan.items() if key != "probe_contract_sha256"}
    if plan["probe_contract_sha256"] != krea_provenance.canonical_sha256(body):
        raise ValueError("timing probe contract digest mismatch")
    for key in ("arm_id", "task_id", "expected_repo_name"):
        if not isinstance(plan[key], str) or not _SAFE_ID.fullmatch(plan[key]):
            raise ValueError(f"timing probe {key} is invalid")
    seed = plan["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("timing probe seed is invalid")
    _, fixture, _ = _load_binding(plan["fixture_manifest"], "probe fixture manifest")
    krea_fixture.validate_manifest(fixture)
    _, fixture_approval, _ = _load_binding(
        plan["fixture_approval"], "probe fixture approval"
    )
    krea_fixture.validate_approval(fixture_approval, fixture_manifest=fixture)
    archive_path, archive_sha = _file_binding(
        plan["training_archive"], "probe training archive"
    )
    if (
        archive_sha != fixture["training_archive"]["sha256"]
        or archive_path.stat().st_size != fixture["training_archive"]["bytes"]
    ):
        raise ValueError("probe training archive differs from approved fixture")
    normalized_basis = _arm_basis(
        plan["arm_basis"],
        arm_id=plan["arm_id"],
        execution_recipe=plan["execution_recipe"],
    )
    recipe = normalized_basis["normalized_execution_recipe"]
    _, host, _ = _load_binding(
        plan["host_execution_manifest"], "probe host execution manifest"
    )
    krea_host_identity.validate_manifest(host)
    envelope = krea_budget.load_execution_envelope(plan["execution_envelope"])
    if (
        envelope.equivalence_class != plan["throughput_equivalence_class"]
        or envelope.host_execution_identity_sha256
        != host["host_execution_identity_sha256"]
        or envelope.runtime_identity_sha256 != plan["runtime_identity_sha256"]
        or envelope.execution_envelope_sha256
        != plan["execution_envelope"]["execution_envelope_sha256"]
    ):
        raise ValueError("timing probe execution envelope is not host/runtime bound")
    base = _object(plan["base_model"], "probe base model")
    _exact(
        base,
        {"model_id", "revision", "training_identity_sha256", "evaluation_assets"},
        "probe base model",
    )
    if (
        base["model_id"] != "krea/Krea-2-Raw"
        or not isinstance(base["revision"], str)
        or not _IMMUTABLE_REVISION.fullmatch(base["revision"])
        or envelope.base_model_identity_sha256 != base["training_identity_sha256"]
    ):
        raise ValueError("timing probe base model identity is invalid")
    _digest(base["training_identity_sha256"], "probe base training identity")
    assets = _object(base["evaluation_assets"], "probe base evaluation assets")
    if set(assets) != {"diffusion_model", "text_encoder", "vae"}:
        raise ValueError("probe base assets are incomplete")
    for name, value in assets.items():
        value = _object(value, f"probe base asset {name}")
        _exact(value, {"canonical_path", "sha256", "bytes"}, f"probe base asset {name}")
        _text(value["canonical_path"], f"probe base asset {name}.canonical_path")
        _digest(value["sha256"], f"probe base asset {name}.sha256")
        if (
            isinstance(value["bytes"], bool)
            or not isinstance(value["bytes"], int)
            or value["bytes"] <= 0
        ):
            raise ValueError(f"probe base asset {name}.bytes is invalid")

    values = _effective_recipe_values(recipe)
    schedule = _object(plan["probe_schedule"], "timing probe schedule")
    _exact(
        schedule,
        {
            "planned_steps",
            "save_every",
            "startup_repetitions",
            "hard_budget_s",
            "measurement_role",
        },
        "timing probe schedule",
    )
    planned = schedule["planned_steps"]
    cadence = schedule["save_every"]
    repetitions = schedule["startup_repetitions"]
    hard_budget = schedule["hard_budget_s"]
    if (
        isinstance(planned, bool)
        or not isinstance(planned, int)
        or planned < 100
        or isinstance(cadence, bool)
        or not isinstance(cadence, int)
        or cadence <= 0
        or len(list(range(cadence, planned, cadence)) + [planned]) < 8
        or isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 3
        or isinstance(hard_budget, bool)
        or not isinstance(hard_budget, (int, float))
        or not math.isfinite(float(hard_budget))
        or float(hard_budget) <= 0
        or schedule["measurement_role"] != "timing_and_heldout"
    ):
        raise ValueError("timing probe schedule cannot satisfy sample requirements")
    if values.get("planned_steps") != planned or values.get("save_cadence") != cadence:
        raise ValueError("timing probe schedule contradicts its recipe")
    axes = plan["predeclared_recipe_axes"]
    if not isinstance(axes, list) or axes != sorted(set(axes)):
        raise ValueError("timing probe axes must be a sorted unique list")
    discovery = validate_discovery_semantics(
        plan["discovery_plan"],
        arm_id=plan["arm_id"],
        fixture_id=plan["discovery_fixture_id"],
        fixture_manifest_sha256=fixture["manifest_sha256"],
        training_pair_count=len(fixture["training_rows"]),
        seed_role=plan["seed_role"],
        seed=seed,
        throughput_equivalence_class=plan["throughput_equivalence_class"],
        execution_recipe=recipe,
        schedule_mode=(
            "release_control" if plan["arm_id"] == "K0" else "measured_budget_fill"
        ),
        predeclared_recipe_axes=axes,
        basis_mode=_object(plan["arm_basis"], "probe arm basis").get("mode"),
    )
    command = plan["command_argv"]
    if (
        not isinstance(command, list)
        or not command
        or any(
            not isinstance(item, str) or not item or "\x00" in item for item in command
        )
    ):
        raise ValueError("timing probe command argv is invalid")
    runner_path = Path(__file__).with_name("run_krea_ladder.py").resolve(strict=True)
    tool_path = Path(__file__).with_name("krea_timing_probe.py").resolve(strict=True)
    if (
        len(command) != 8
        or Path(command[1]).expanduser().resolve(strict=True) != runner_path
        or command[2] != "--timing-probe-plan"
        or command[4] != "--timing-probe-approval"
        or command[6] != "--campaign-dir"
        or any(
            not isinstance(command[index], str)
            or not os.path.isabs(os.path.expanduser(command[index]))
            for index in (0, 1, 3, 5, 7)
        )
    ):
        raise ValueError(
            "timing probe command must be the bounded run_krea_ladder bootstrap argv"
        )
    if (
        krea_provenance.file_sha256(runner_path) != plan["runner_sha256"]
        or krea_provenance.file_sha256(tool_path) != plan["measurement_tool_sha256"]
        or envelope.measurement_tool_sha256 != plan["measurement_tool_sha256"]
    ):
        raise ValueError(
            "timing probe code identity differs from local producer/runner"
        )
    return {
        "fixture": fixture,
        "execution_recipe": recipe,
        "host_execution_manifest": host,
        "execution_envelope": envelope,
        "discovery": discovery,
        "training_archive_path": archive_path,
    }


def seal_timing_probe_plan(payload: dict[str, Any]) -> dict[str, Any]:
    if "probe_contract_sha256" in payload:
        raise ValueError("unsealed timing probe payload has a digest")
    plan = {
        **payload,
        "probe_contract_sha256": krea_provenance.canonical_sha256(payload),
    }
    validate_timing_probe_plan(plan)
    return plan


def build_timing_probe_approval(
    plan: dict[str, Any], *, reviewer_identity: str, approved_at_utc: str
) -> dict[str, Any]:
    resolved = validate_timing_probe_plan(plan)
    body = {
        "schema": 1,
        "kind": _TIMING_APPROVAL_KIND,
        "probe_contract_sha256": plan["probe_contract_sha256"],
        "host_execution_identity_sha256": resolved["host_execution_manifest"][
            "host_execution_identity_sha256"
        ],
        "reviewer_identity": krea_fixture.named_human(
            reviewer_identity, "reviewer_identity"
        ),
        "approved_at_utc": _strict_utc(approved_at_utc, "approved_at_utc"),
        "decision": "approved",
        "gpu_execution_authorized": True,
        "assertions": {
            "host_capability_reviewed": True,
            "fixture_and_recipe_reviewed": True,
            "timing_only_no_production_mutation": True,
            "natural_completion_will_be_certified_post_run": True,
        },
    }
    return {**body, "approval_sha256": krea_provenance.canonical_sha256(body)}


def validate_timing_probe_approval(
    approval: dict[str, Any], *, plan: dict[str, Any]
) -> dict[str, Any]:
    resolved = validate_timing_probe_plan(plan)
    approval = _object(approval, "timing probe approval")
    _exact(
        approval,
        {
            "schema",
            "kind",
            "probe_contract_sha256",
            "host_execution_identity_sha256",
            "reviewer_identity",
            "approved_at_utc",
            "decision",
            "gpu_execution_authorized",
            "assertions",
            "approval_sha256",
        },
        "timing probe approval",
    )
    body = {key: value for key, value in approval.items() if key != "approval_sha256"}
    expected_assertions = {
        "host_capability_reviewed": True,
        "fixture_and_recipe_reviewed": True,
        "timing_only_no_production_mutation": True,
        "natural_completion_will_be_certified_post_run": True,
    }
    if (
        approval["schema"] != 1
        or approval["kind"] != _TIMING_APPROVAL_KIND
        or approval["probe_contract_sha256"] != plan["probe_contract_sha256"]
        or approval["host_execution_identity_sha256"]
        != resolved["host_execution_manifest"]["host_execution_identity_sha256"]
        or approval["decision"] != "approved"
        or approval["gpu_execution_authorized"] is not True
        or approval["assertions"] != expected_assertions
        or approval["approval_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("timing probe approval does not authorize this probe")
    krea_fixture.named_human(approval["reviewer_identity"], "reviewer_identity")
    _strict_utc(approval["approved_at_utc"], "approved_at_utc")
    return approval


def build_approval(
    plan: dict[str, Any],
    *,
    reviewer_identity: str,
    approved_at_utc: str,
) -> dict[str, Any]:
    """Create a pre-run approval without demanding evidence from the future."""

    resolved = validate_plan(plan)
    body = {
        "schema": 2,
        "kind": _EXECUTION_APPROVAL_KIND,
        "execution_plan_sha256": plan["plan_sha256"],
        "host_execution_identity_sha256": resolved["host_execution_manifest"][
            "host_execution_identity_sha256"
        ],
        "throughput_profile_sha256": resolved["throughput_profile"]["profile_sha256"],
        "reviewer_identity": krea_fixture.named_human(
            reviewer_identity, "reviewer_identity"
        ),
        "approved_at_utc": _strict_utc(approved_at_utc, "approved_at_utc"),
        "decision": "approved",
        "gpu_execution_authorized": True,
        "assertions": {
            "host_capability_reviewed": True,
            "raw_timing_evidence_reviewed": True,
            "fixture_recipe_and_budget_reviewed": True,
            "natural_completion_is_post_run_evidence": True,
        },
    }
    return {**body, "approval_sha256": krea_provenance.canonical_sha256(body)}


def build_postrun_certificate(
    plan: dict[str, Any],
    *,
    run_record: dict[str, str],
    observed: dict[str, Any],
) -> dict[str, Any]:
    """Seal natural completion after execution; never use it to authorize itself."""

    resolved = validate_plan(plan)
    _exact(run_record, {"path", "sha256"}, "post-run record binding")
    run_path, run_sha = _file_binding(run_record, "post-run record")
    _exact(
        observed,
        {
            "linux_ubuntu_22_04",
            "systemd_runtime_max_enforced",
            "fresh_container",
            "h100_vram_mib",
            "outer_wall_clock_s",
            "hard_budget_s",
            "upload_ready_before_boundary",
            "natural_completion",
            "failure_or_fallback_telemetry",
        },
        "post-run observations",
    )
    body = {
        "schema": 1,
        "kind": _POSTRUN_CERTIFICATE_KIND,
        "runner_sha256": plan["runner_sha256"],
        "execution_envelope_sha256": plan["execution_envelope_sha256"],
        "execution_plan_sha256": plan["plan_sha256"],
        "host_execution_identity_sha256": resolved["host_execution_manifest"][
            "host_execution_identity_sha256"
        ],
        "run_record": {"path": str(run_path), "sha256": run_sha},
        **observed,
    }
    certificate = {
        **body,
        "certificate_sha256": krea_provenance.canonical_sha256(body),
    }
    validate_postrun_certificate(certificate, plan=plan)
    return certificate


def validate_postrun_certificate(
    value: dict[str, Any], *, plan: dict[str, Any]
) -> dict[str, Any]:
    resolved = validate_plan(plan)
    _exact(
        value,
        {
            "schema",
            "kind",
            "runner_sha256",
            "execution_envelope_sha256",
            "execution_plan_sha256",
            "host_execution_identity_sha256",
            "run_record",
            "linux_ubuntu_22_04",
            "systemd_runtime_max_enforced",
            "fresh_container",
            "h100_vram_mib",
            "outer_wall_clock_s",
            "hard_budget_s",
            "upload_ready_before_boundary",
            "natural_completion",
            "failure_or_fallback_telemetry",
            "certificate_sha256",
        },
        "post-run certification",
    )
    body = {key: item for key, item in value.items() if key != "certificate_sha256"}
    vram = value["h100_vram_mib"]
    _file_binding(value["run_record"], "post-run record")
    if (
        value["schema"] != 1
        or value["kind"] != _POSTRUN_CERTIFICATE_KIND
        or value["runner_sha256"] != plan["runner_sha256"]
        or value["execution_envelope_sha256"] != plan["execution_envelope_sha256"]
        or value["execution_plan_sha256"] != plan["plan_sha256"]
        or value["host_execution_identity_sha256"]
        != resolved["host_execution_manifest"]["host_execution_identity_sha256"]
        or value["linux_ubuntu_22_04"] is not True
        or value["systemd_runtime_max_enforced"] is not True
        or value["fresh_container"] is not True
        or isinstance(vram, bool)
        or not isinstance(vram, int)
        or not 78_000 <= vram <= 85_000
        or value["upload_ready_before_boundary"] is not True
        or value["natural_completion"] is not True
        or value["failure_or_fallback_telemetry"] is not False
        or isinstance(value["outer_wall_clock_s"], bool)
        or not isinstance(value["outer_wall_clock_s"], (int, float))
        or isinstance(value["hard_budget_s"], bool)
        or not isinstance(value["hard_budget_s"], (int, float))
        or not math.isfinite(float(value["outer_wall_clock_s"]))
        or not math.isfinite(float(value["hard_budget_s"]))
        or float(value["outer_wall_clock_s"]) <= 0
        or float(value["hard_budget_s"]) <= 0
        or float(value["outer_wall_clock_s"]) > float(value["hard_budget_s"])
        or float(value["hard_budget_s"]) != float(plan["budget_plan"]["hard_budget_s"])
        or value["certificate_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("post-run certificate does not prove natural completion")
    return value


def validate_approval(
    approval: dict[str, Any], *, plan: dict[str, Any]
) -> dict[str, Any]:
    validate_plan(plan)
    approval = _object(approval, "execution approval")
    _exact(
        approval,
        {
            "schema",
            "kind",
            "execution_plan_sha256",
            "host_execution_identity_sha256",
            "throughput_profile_sha256",
            "reviewer_identity",
            "approved_at_utc",
            "decision",
            "gpu_execution_authorized",
            "assertions",
            "approval_sha256",
        },
        "execution approval",
    )
    body = {key: value for key, value in approval.items() if key != "approval_sha256"}
    if (
        approval["schema"] != 2
        or approval["kind"] != _EXECUTION_APPROVAL_KIND
        or approval["execution_plan_sha256"] != plan["plan_sha256"]
        or approval["decision"] != "approved"
        or approval["gpu_execution_authorized"] is not True
        or approval["host_execution_identity_sha256"]
        != _load_binding(plan["host_execution_manifest"], "host execution manifest")[1][
            "host_execution_identity_sha256"
        ]
        or approval["throughput_profile_sha256"]
        != _json_file_binding(plan["throughput_profile"], "throughput profile")[1][
            "profile_sha256"
        ]
        or approval["assertions"]
        != {
            "host_capability_reviewed": True,
            "raw_timing_evidence_reviewed": True,
            "fixture_recipe_and_budget_reviewed": True,
            "natural_completion_is_post_run_evidence": True,
        }
        or approval["approval_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("execution approval does not authorize this plan")
    krea_fixture.named_human(approval["reviewer_identity"], "reviewer_identity")
    _strict_utc(approval["approved_at_utc"], "approved_at_utc")
    return approval
