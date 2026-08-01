#!/usr/bin/env python3
"""Grouped exact-score consumer for the Week-5 Krea Stage-2 endgame.

Each confirmation fixture/seed has two active-policy score plans.  A plan
groups the active candidate with K0 and all three public references, so fixture
and evaluator staging is shared while every candidate still receives an exact
validator-format result, receipt, and live-file replay.  Boundary cells are
mechanics-only and never enter this module.

Training completion and score completion are intentionally separate gates.
This module cannot authorize release or production mutation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Mapping, Sequence

try:
    from . import batch_evaluate_krea
    from . import krea_fixture
    from . import krea_provenance
    from . import krea_stage2_endgame_matrix
    from . import krea_stage2_endgame_orchestrator
    from . import krea_stage2_execution
    from . import krea_stage2_score
except ImportError:  # pragma: no cover - direct CLI execution.
    import batch_evaluate_krea  # type: ignore[no-redef]
    import krea_fixture  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_stage2_endgame_matrix  # type: ignore[no-redef]
    import krea_stage2_endgame_orchestrator  # type: ignore[no-redef]
    import krea_stage2_execution  # type: ignore[no-redef]
    import krea_stage2_score  # type: ignore[no-redef]


SCHEMA = 1
CONFIG_KIND = "forge-krea-stage2-endgame-score-consumer-config"
QUEUE_KIND = "forge-krea-stage2-endgame-score-queue"
GROUP_KIND = "forge-krea-stage2-endgame-score-group"
CLAIM_KIND = "forge-krea-stage2-endgame-score-claim"
STATUS_KIND = "forge-krea-stage2-exact-score-command-status"
EVIDENCE_KIND = "forge-krea-stage2-exact-score-command-evidence"
GATE_KIND = "forge-krea-stage2-endgame-score-completion-gate"
GROUP_COUNT = 16
FAMILIES_PER_GROUP = 5
_SHA = re.compile(r"[0-9a-f]{64}")
_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


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
    if not isinstance(value, str) or _SAFE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a conservative identifier")
    return value


def _file_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(krea_provenance.canonical_bytes(value) + b"\n").hexdigest()


def _load(path: str | Path, label: str) -> dict[str, Any]:
    return krea_stage2_endgame_matrix._load_canonical(path, label)


def _publish_or_replay(
    path: str | Path, value: Mapping[str, Any], label: str
) -> dict[str, Any]:
    return krea_stage2_endgame_matrix._publish_or_replay(path, value, label)


def _score_group_key(cell_id: str, candidate_family: str) -> str:
    return _safe_id(f"score-{cell_id}-{candidate_family}", "score group key")


def _required_families(candidate_family: str) -> list[str]:
    families = {
        candidate_family,
        krea_stage2_score.CONTROL,
        *krea_stage2_score.PUBLIC_REFERENCES,
    }
    if len(families) != FAMILIES_PER_GROUP:
        raise ValueError("active policy overlaps a required reference/control")
    return sorted(families)


def _groups(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    active = matrix["active_variant_family_ids"]
    if active != ["K1", "K5"]:
        raise ValueError("score consumer requires the frozen K1/K5 active policies")
    rows = []
    for fixture in krea_stage2_endgame_matrix.CONFIRMATION_FIXTURES:
        for seed in krea_stage2_endgame_matrix.SEED_ROLES:
            cell = f"{fixture}-{seed}"
            for candidate_family in active:
                rows.append(
                    {
                        "group_key": _score_group_key(cell, candidate_family),
                        "cell_id": cell,
                        "fixture_id": fixture,
                        "seed_role": seed,
                        "candidate_family_id": candidate_family,
                        "family_ids": _required_families(candidate_family),
                    }
                )
    if len(rows) != GROUP_COUNT:
        raise AssertionError("internal score group cardinality drifted")
    return rows


def _expected_group_identity() -> list[tuple[str, str, str, str, str]]:
    return [
        (
            _score_group_key(f"{fixture}-{seed}", family),
            f"{fixture}-{seed}",
            family,
            fixture,
            seed,
        )
        for fixture in krea_stage2_endgame_matrix.CONFIRMATION_FIXTURES
        for seed in krea_stage2_endgame_matrix.SEED_ROLES
        for family in ("K1", "K5")
    ]


def _validate_queue(value: Any) -> dict[str, Any]:
    queue = _object(value, "score queue")
    _exact(
        queue,
        {
            "schema",
            "kind",
            "config_sha256",
            "matrix_sha256",
            "training_plan_set_sha256",
            "expected_group_count",
            "groups",
            "streaming_materialization",
            "boundary_scoring_enabled",
            "release_authorized",
            "production_mutation_authorized",
            "score_queue_sha256",
        },
        "score queue",
    )
    body = {key: item for key, item in queue.items() if key != "score_queue_sha256"}
    if (
        queue["schema"] != SCHEMA
        or queue["kind"] != QUEUE_KIND
        or queue["score_queue_sha256"] != krea_provenance.canonical_sha256(body)
        or queue["expected_group_count"] != GROUP_COUNT
        or queue["streaming_materialization"] is not True
        or queue["boundary_scoring_enabled"] is not False
        or queue["release_authorized"] is not False
        or queue["production_mutation_authorized"] is not False
    ):
        raise ValueError("score queue identity/authority differs")
    for field in ("config_sha256", "matrix_sha256", "training_plan_set_sha256"):
        _sha(queue[field], f"score queue {field}")
    if not isinstance(queue["groups"], list) or len(queue["groups"]) != GROUP_COUNT:
        raise ValueError("score queue does not contain exactly 16 groups")
    expected_keys = {
        "group_key",
        "cell_id",
        "fixture_id",
        "seed_role",
        "candidate_family_id",
        "family_ids",
        "group_path",
        "aggregate_path",
    }
    group_keys = []
    observed_identity = []
    for raw in queue["groups"]:
        row = _object(raw, "score queue group")
        _exact(row, expected_keys, "score queue group")
        group_keys.append(_safe_id(row["group_key"], "score group key"))
        observed_identity.append(
            (
                row["group_key"],
                row["cell_id"],
                row["candidate_family_id"],
                row["fixture_id"],
                row["seed_role"],
            )
        )
        if (
            row["family_ids"] != _required_families(row["candidate_family_id"])
            or len(row["family_ids"]) != FAMILIES_PER_GROUP
        ):
            raise ValueError("score queue group family coverage differs")
        for field in ("group_path", "aggregate_path"):
            path = Path(row[field])
            if not path.is_absolute() or str(path) != os.path.abspath(str(path)):
                raise ValueError(f"score queue {field} is not absolute and normalized")
    if len(set(group_keys)) != GROUP_COUNT:
        raise ValueError("score queue group keys are not unique")
    if observed_identity != _expected_group_identity():
        raise ValueError("score queue is not the exact frozen 16-group order")
    return dict(queue)


def _validate_group(
    value: Any, *, score_queue: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    group = _object(value, "score group")
    keys = {
        "schema",
        "kind",
        "score_queue_sha256",
        "group_key",
        "cell_id",
        "fixture_id",
        "seed_role",
        "candidate_family_id",
        "family_ids",
        "plan_path",
        "plan_file_sha256",
        "plan_sha256",
        "fixture_manifest_path",
        "jobs",
        "aggregate_path",
        "release_authorized",
        "production_mutation_authorized",
        "group_sha256",
    }
    _exact(group, keys, "score group")
    body = {key: item for key, item in group.items() if key != "group_sha256"}
    if (
        group["schema"] != SCHEMA
        or group["kind"] != GROUP_KIND
        or group["group_sha256"] != krea_provenance.canonical_sha256(body)
        or group["family_ids"] != _required_families(group["candidate_family_id"])
        or group["release_authorized"] is not False
        or group["production_mutation_authorized"] is not False
    ):
        raise ValueError("score group identity/authority differs")
    if group["group_key"] != _score_group_key(
        group["cell_id"], group["candidate_family_id"]
    ):
        raise ValueError("score group key differs from its cell/policy")
    for field in ("score_queue_sha256", "plan_file_sha256", "plan_sha256"):
        _sha(group[field], f"score group {field}")
    _safe_id(group["group_key"], "score group key")
    if not isinstance(group["jobs"], list) or len(group["jobs"]) != FAMILIES_PER_GROUP:
        raise ValueError("score group must contain exactly five jobs")
    expected_job_keys = {
        "family_id",
        "candidate_path",
        "result_path",
        "status_path",
        "evidence_manifest_path",
        "receipt_path",
    }
    families = []
    for raw in group["jobs"]:
        job = _object(raw, "score group job")
        _exact(job, expected_job_keys, "score group job")
        families.append(_safe_id(job["family_id"], "score job family"))
        for field in expected_job_keys - {"family_id"}:
            path = Path(job[field])
            if not path.is_absolute() or str(path) != os.path.abspath(str(path)):
                raise ValueError(f"score job {field} is not absolute and normalized")
    if families != group["family_ids"]:
        raise ValueError("score jobs differ from the exact family order")
    for field in ("plan_path", "fixture_manifest_path", "aggregate_path"):
        path = Path(group[field])
        if not path.is_absolute() or str(path) != os.path.abspath(str(path)):
            raise ValueError(f"score group {field} is not absolute and normalized")
    if score_queue is not None:
        queue = _validate_queue(score_queue)
        matches = [
            row for row in queue["groups"] if row["group_key"] == group["group_key"]
        ]
        if (
            len(matches) != 1
            or group["score_queue_sha256"] != queue["score_queue_sha256"]
            or any(
                group[key] != matches[0][key]
                for key in (
                    "cell_id",
                    "fixture_id",
                    "seed_role",
                    "candidate_family_id",
                    "family_ids",
                    "aggregate_path",
                )
            )
            or str(Path(matches[0]["group_path"]))
            != str(Path(group["plan_path"]).parent / "group.json")
        ):
            raise ValueError("score group differs from its immutable queue")
    return dict(group)


def _validate_config(value: Any) -> dict[str, Any]:
    config = _object(value, "score consumer config")
    _exact(
        config,
        {
            "schema",
            "kind",
            "matrix",
            "training_plan_set",
            "fixture_manifests",
            "evaluation_datasets",
            "evaluator_contract",
            "evaluator_script",
            "gpu_surfaces",
            "score_plan_created_at_utc",
            "config_sha256",
        },
        "score consumer config",
    )
    body = {key: item for key, item in config.items() if key != "config_sha256"}
    if (
        config["schema"] != SCHEMA
        or config["kind"] != CONFIG_KIND
        or config["config_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("score consumer config identity differs")
    krea_stage2_score._evaluator_contract(config["evaluator_contract"])
    evaluator_script = Path(
        os.path.abspath(os.path.expanduser(str(config["evaluator_script"])))
    )
    if (
        evaluator_script.is_symlink()
        or not evaluator_script.is_file()
        or evaluator_script.resolve(strict=True)
        != Path(batch_evaluate_krea.__file__)
        .with_name("evaluate_krea_local.py")
        .resolve(strict=True)
    ):
        raise ValueError("score config does not use the pinned exact evaluator")
    surfaces = _object(config["gpu_surfaces"], "score GPU surfaces")
    if set(surfaces) != {str(gpu) for gpu in krea_stage2_endgame_orchestrator.GPU_IDS}:
        raise ValueError("score config must contain GPU surfaces 0-3 exactly")
    required_surface = {
        "driver_python",
        "comfy_root",
        "comfy_python",
        "god_root",
        "expected_god_commit",
        "expected_comfy_commit",
        "expected_tooling_commit",
        "base_name",
        "port",
        "startup_timeout_s",
        "evaluation_timeout_s",
        "shutdown_timeout_s",
        "containment",
    }
    roots = []
    for gpu, surface in surfaces.items():
        surface = _object(surface, f"score GPU {gpu} surface")
        _exact(surface, required_surface, f"score GPU {gpu} surface")
        for field in ("driver_python", "comfy_root", "comfy_python", "god_root"):
            path = Path(os.path.abspath(os.path.expanduser(str(surface[field]))))
            if path.is_symlink() or not path.exists():
                raise ValueError(f"score GPU {gpu} {field} is unavailable")
        roots.append(Path(surface["comfy_root"]).resolve(strict=True))
        if surface["port"] != 8188 + int(gpu):
            raise ValueError("score GPU ports must be fixed to 8188 + device")
        batch_evaluate_krea._timeout_policy(surface)
    if len(roots) != len(set(roots)):
        raise ValueError("score GPUs require isolated Comfy roots")
    return dict(config)


def _row_map(
    plan_set: Mapping[str, Any], matrix: Mapping[str, Any]
) -> dict[tuple[str, str], dict[str, Any]]:
    resolved = krea_stage2_endgame_orchestrator.validate_plan_set(
        plan_set, matrix=matrix
    )
    result = {}
    matrix_rows = {row["row_key"]: row for row in matrix["rows"]}
    for row in resolved["rows"]:
        source = matrix_rows[row["row_key"]]
        if source["phase"] == "confirmation":
            result[(source["cell_id"], source["family_id"])] = row
    if len(result) != matrix["confirmation_training_count"]:
        raise ValueError("confirmation row map is incomplete")
    return result


def _candidate_path(plan: Mapping[str, Any], evidence: Mapping[str, Any]) -> Path:
    matches = [
        row
        for row in evidence["candidate_artifacts"]
        if PurePosixPath(str(row["path"])).name == "last.safetensors"
    ]
    if len(matches) != 1:
        raise ValueError("run evidence lacks one promoted last.safetensors")
    relative = PurePosixPath(matches[0]["path"])
    checkpoint_mount = next(
        Path(row["source"]) for row in plan["mounts"] if row["purpose"] == "checkpoints"
    )
    path = checkpoint_mount / plan["task_id"] / plan["expected_repo_name"]
    if relative.parts[0] != "checkpoints":
        raise ValueError("score candidate is not under the checkpoint root")
    path = path.joinpath(*relative.parts[1:])
    if path.is_symlink() or not path.is_file():
        raise ValueError("score candidate path is not a live regular file")
    if (
        path.stat().st_size != matches[0]["bytes"]
        or krea_provenance.file_sha256(path) != matches[0]["sha256"]
    ):
        raise ValueError("score candidate bytes differ from run evidence")
    return path


def _run_control(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _load(row["plan"]["path"], "score source execution plan")
    approval = _load(row["approval"]["path"], "score source execution approval")
    completion = _load(row["completion_path"], "score source completion")
    evidence = _load(row["run_evidence_path"], "score source run evidence")
    candidate = _candidate_path(plan, evidence)
    return {
        "run_evidence_path": row["run_evidence_path"],
        "execution_plan": plan,
        "execution_approval": approval,
        "run_completion": completion,
        "candidate_path": candidate,
    }, evidence


def _fixture_score_view(
    manifest_path: str | Path, dataset_path: str | Path, expected_role: str
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Load either exact legacy-C wrapper or canonical boundary manifest."""

    path = Path(os.path.abspath(os.path.expanduser(str(manifest_path))))
    document = _load(path, f"score fixture {expected_role}")
    manifest = krea_stage2_score._fixture_score_view(document)
    if manifest["experimental_role"] != expected_role:
        raise ValueError("score fixture role differs")
    dataset = Path(os.path.abspath(os.path.expanduser(str(dataset_path))))
    if dataset.is_symlink() or not dataset.is_dir():
        raise ValueError("score evaluation dataset is unavailable")
    identity = manifest["evaluation_dataset_identity"]
    return (
        manifest,
        {
            "file_sha256": _file_sha(document),
            "manifest_sha256": manifest["manifest_sha256"],
        },
        dataset,
    )


def materialize_ready_score_plans(
    config: Mapping[str, Any], *, output_root: str | Path
) -> dict[str, Any]:
    """Publish every newly-ready 5-family plan; never wait for all 60 rows."""

    supplied = _validate_config(config)
    matrix = krea_stage2_endgame_matrix.validate_matrix(
        _load(supplied["matrix"], "score matrix")
    )
    training = krea_stage2_endgame_orchestrator.validate_plan_set(
        _load(supplied["training_plan_set"], "training plan set"), matrix=matrix
    )
    rows = _row_map(training, matrix)
    root = Path(os.path.abspath(os.path.expanduser(str(output_root))))
    root.mkdir(parents=True, exist_ok=True)
    queue_groups = [
        {
            **group,
            "group_path": str(root / "groups" / group["group_key"] / "group.json"),
            "aggregate_path": str(
                root / "groups" / group["group_key"] / "aggregate.json"
            ),
        }
        for group in _groups(matrix)
    ]
    queue_body = {
        "schema": SCHEMA,
        "kind": QUEUE_KIND,
        "config_sha256": supplied["config_sha256"],
        "matrix_sha256": matrix["matrix_sha256"],
        "training_plan_set_sha256": training["plan_set_sha256"],
        "expected_group_count": GROUP_COUNT,
        "groups": queue_groups,
        "streaming_materialization": True,
        "boundary_scoring_enabled": False,
        "release_authorized": False,
        "production_mutation_authorized": False,
    }
    queue = {
        **queue_body,
        "score_queue_sha256": krea_provenance.canonical_sha256(queue_body),
    }
    queue = _validate_queue(
        _publish_or_replay(root / "score-queue.json", queue, "score queue")
    )
    plan_rows = []
    for group in _groups(matrix):
        group_root = root / "groups" / group["group_key"]
        needed = [rows[(group["cell_id"], family)] for family in group["family_ids"]]
        ready = all(os.path.lexists(row["receipt_path"]) for row in needed)
        plan_path = group_root / "score-plan.json"
        if not ready and not os.path.lexists(plan_path):
            continue
        if not ready:
            raise ValueError("published score plan lost a required training receipt")
        controls = {}
        evidence_by_family = {}
        candidate_rows = []
        for family, row in zip(group["family_ids"], needed, strict=True):
            control, evidence = _run_control(row)
            controls[family] = control
            evidence_by_family[family] = evidence
            candidate_rows.append(
                krea_stage2_score.build_candidate_row(
                    family_id=family,
                    candidate_id=f"{group['cell_id'].lower()}-{family.lower()}",
                    run_evidence_path=Path(control["run_evidence_path"]),
                    execution_plan=control["execution_plan"],
                    execution_approval=control["execution_approval"],
                    run_completion=control["run_completion"],
                    candidate_path=control["candidate_path"],
                )
            )
        candidate_rows.sort(key=lambda row: row["family_id"])
        first = controls[group["family_ids"][0]]["execution_plan"]
        manifest, fixture_binding, dataset = _fixture_score_view(
            supplied["fixture_manifests"][group["fixture_id"]],
            supplied["evaluation_datasets"][group["fixture_id"]],
            group["fixture_id"],
        )
        identity = manifest["evaluation_dataset_identity"]
        payload = {
            "schema": 1,
            "kind": krea_stage2_score.PLAN_KIND,
            "phase": "confirmation",
            "cell_id": group["cell_id"],
            "fixture_id": group["fixture_id"],
            "seed_role": group["seed_role"],
            "seed": first["seed"],
            "hours": first["hours"],
            "candidate_family_id": group["candidate_family_id"],
            "public_reference_family_ids": list(krea_stage2_score.PUBLIC_REFERENCES),
            "control_family_id": krea_stage2_score.CONTROL,
            "candidates": candidate_rows,
            "fixture_manifest": fixture_binding,
            "evaluation_dataset_sha256": identity["sha256"],
            "evaluation_dataset_path": str(dataset),
            "evaluation_row_count": len(identity["rows"]),
            "evaluator_contract": supplied["evaluator_contract"],
            "waiver_finalist_freeze": first["waiver_finalist_freeze"],
            "confirmation_materialization": first["confirmation_materialization"],
            "owner_ratification": first["owner_ratification"],
            "gpu_execution_authorization": first["gpu_execution_authorization"],
            "production_identity": first["production_identity"],
            "production_image_id": first["production_image_id"],
            "created_at_utc": supplied["score_plan_created_at_utc"],
            "fallback_allowed": False,
            "release_authorized": False,
            "production_mutation_authorized": False,
        }
        plan = krea_stage2_score.seal_plan(payload)
        plan = krea_stage2_score.validate_plan_with_run_controls(
            plan, controls_by_family=controls
        )
        group_root.mkdir(parents=True, exist_ok=True)
        _publish_or_replay(plan_path, plan, "grouped exact-score plan")
        jobs = []
        for family in group["family_ids"]:
            family_root = group_root / "families" / family
            jobs.append(
                {
                    "family_id": family,
                    "candidate_path": str(controls[family]["candidate_path"]),
                    "result_path": str(family_root / "result.json"),
                    "status_path": str(family_root / "status.json"),
                    "evidence_manifest_path": str(family_root / "evidence.json"),
                    "receipt_path": str(family_root / "receipt.json"),
                }
            )
        group_body = {
            "schema": SCHEMA,
            "kind": GROUP_KIND,
            "score_queue_sha256": queue["score_queue_sha256"],
            **group,
            "plan_path": str(plan_path),
            "plan_file_sha256": _file_sha(plan),
            "plan_sha256": plan["plan_sha256"],
            "fixture_manifest_path": str(
                Path(supplied["fixture_manifests"][group["fixture_id"]]).resolve(
                    strict=True
                )
            ),
            "jobs": jobs,
            "aggregate_path": str(group_root / "aggregate.json"),
            "release_authorized": False,
            "production_mutation_authorized": False,
        }
        group_record = {
            **group_body,
            "group_sha256": krea_provenance.canonical_sha256(group_body),
        }
        group_record = _validate_group(
            _publish_or_replay(group_root / "group.json", group_record, "score group"),
            score_queue=queue,
        )
        plan_rows.append(
            {
                "group_key": group["group_key"],
                "group_path": str(group_root / "group.json"),
                "group_sha256": group_record["group_sha256"],
                "plan_sha256": plan["plan_sha256"],
            }
        )
    return {"queue": queue, "materialized_groups": plan_rows}


def _command(
    *,
    config: Mapping[str, Any],
    surface: Mapping[str, Any],
    plan: Mapping[str, Any],
    job: Mapping[str, Any],
) -> list[str]:
    fixture = krea_stage2_score._fixture_score_view(
        _load(config["fixture_manifests"][plan["fixture_id"]], "score fixture")
    )
    evaluator = {
        **surface,
        "_expected_dataset_identity": {
            "evaluator_order": [
                row["image"] for row in fixture["evaluation_dataset_identity"]["rows"]
            ]
        },
    }
    return batch_evaluate_krea._evaluator_command(
        evaluator_script=Path(config["evaluator_script"]),
        dataset=Path(plan["evaluation_dataset_path"]),
        candidate={"path": Path(job["candidate_path"])},
        result_path=Path(job["result_path"]),
        evaluator=evaluator,
    )


def _validate_job_evidence(
    *, job: Mapping[str, Any], plan: Mapping[str, Any], gpu_device: int | None
) -> dict[str, Any]:
    family = job["family_id"]
    status_path = Path(job["status_path"])
    evidence_path = Path(job["evidence_manifest_path"])
    status = _load(status_path, "score command status")
    _exact(
        status,
        {
            "schema",
            "kind",
            "score_plan_sha256",
            "family_id",
            "gpu_device",
            "command_sha256",
            "returncode",
            "stdout_sha256",
            "stderr_sha256",
            "fallback_used",
            "release_authorized",
            "status_sha256",
        },
        "score command status",
    )
    status_body = {
        key: value for key, value in status.items() if key != "status_sha256"
    }
    if (
        status["schema"] != SCHEMA
        or status["kind"] != STATUS_KIND
        or status["status_sha256"] != krea_provenance.canonical_sha256(status_body)
        or status["score_plan_sha256"] != plan["plan_sha256"]
        or status["family_id"] != family
        or status["gpu_device"] not in krea_stage2_endgame_orchestrator.GPU_IDS
        or (gpu_device is not None and status["gpu_device"] != gpu_device)
        or status["returncode"] != 0
        or status["fallback_used"] is not False
        or status["release_authorized"] is not False
    ):
        raise ValueError("score command status identity/authority differs")
    for field in ("command_sha256", "stdout_sha256", "stderr_sha256"):
        _sha(status[field], f"score command status {field}")
    evidence = _load(evidence_path, "score command evidence")
    _exact(
        evidence,
        {
            "schema",
            "kind",
            "score_plan_sha256",
            "family_id",
            "candidate_sha256",
            "result_sha256",
            "comfy_log_sha256",
            "status_file_sha256",
            "release_authorized",
            "production_mutation_authorized",
            "evidence_sha256",
        },
        "score command evidence",
    )
    evidence_body = {
        key: value for key, value in evidence.items() if key != "evidence_sha256"
    }
    log_path = Path(str(job["result_path"]) + ".comfy.log")
    if (
        evidence["schema"] != SCHEMA
        or evidence["kind"] != EVIDENCE_KIND
        or evidence["evidence_sha256"]
        != krea_provenance.canonical_sha256(evidence_body)
        or evidence["score_plan_sha256"] != plan["plan_sha256"]
        or evidence["family_id"] != family
        or evidence["candidate_sha256"]
        != krea_provenance.file_sha256(Path(job["candidate_path"]))
        or evidence["result_sha256"]
        != krea_provenance.file_sha256(Path(job["result_path"]))
        or evidence["comfy_log_sha256"] != krea_provenance.file_sha256(log_path)
        or evidence["status_file_sha256"] != krea_provenance.file_sha256(status_path)
        or evidence["release_authorized"] is not False
        or evidence["production_mutation_authorized"] is not False
    ):
        raise ValueError("score command evidence differs from live files")
    return {
        "gpu_device": status["gpu_device"],
        "status_file_sha256": krea_provenance.file_sha256(status_path),
        "evidence_manifest_file_sha256": krea_provenance.file_sha256(evidence_path),
    }


def run_group(
    *,
    config: Mapping[str, Any],
    group: Mapping[str, Any],
    gpu_device: int,
    gpu_lock_root: str | Path,
) -> dict[str, Any]:
    supplied = _validate_config(config)
    if gpu_device not in krea_stage2_endgame_orchestrator.GPU_IDS:
        raise ValueError("score GPU is outside 0-3")
    group = _validate_group(group)
    plan = krea_stage2_score.validate_plan(_load(group["plan_path"], "score plan"))
    if (
        _file_sha(plan) != group["plan_file_sha256"]
        or plan["plan_sha256"] != group["plan_sha256"]
    ):
        raise ValueError("score plan bytes differ from score group")
    surface = supplied["gpu_surfaces"][str(gpu_device)]
    fixture = _load(group["fixture_manifest_path"], "score fixture manifest")
    fixture_file_sha = _file_sha(fixture)
    receipts = []
    score_files = {}
    candidates = {row["family_id"]: row for row in plan["candidates"]}
    for job in group["jobs"]:
        family = job["family_id"]
        receipt_path = Path(job["receipt_path"])
        result_path = Path(job["result_path"])
        if os.path.lexists(receipt_path):
            receipt = _load(receipt_path, "existing score receipt")
            receipt = krea_stage2_score.validate_receipt_with_score_files(
                receipt,
                plan=plan,
                candidate_path=Path(job["candidate_path"]),
                fixture_manifest=fixture,
                fixture_manifest_file_sha256=fixture_file_sha,
                result_path=result_path,
            )
            command_evidence = _validate_job_evidence(
                job=job, plan=plan, gpu_device=None
            )
            if any(
                receipt[key] != command_evidence[key]
                for key in ("status_file_sha256", "evidence_manifest_file_sha256")
            ):
                raise ValueError("score receipt differs from command evidence files")
            receipts.append(receipt)
            score_files[family] = {
                "candidate_path": job["candidate_path"],
                "result_path": job["result_path"],
            }
            continue
        family_root = receipt_path.parent
        family_root.mkdir(parents=True, exist_ok=True)
        candidate_path = Path(job["candidate_path"])
        command = _command(config=supplied, surface=surface, plan=plan, job=job)
        with krea_stage2_endgame_orchestrator.gpu_execution_lock(
            gpu_lock_root, gpu_device
        ):
            staging = batch_evaluate_krea._stage_comfy_lora(
                comfy_root=Path(surface["comfy_root"]),
                candidate=candidate_path,
                candidate_sha256=candidates[family]["candidate_sha256"],
            )
            try:
                process = batch_evaluate_krea._run_contained(
                    command,
                    timeout_s=batch_evaluate_krea._timeout_policy(surface)[
                        "total_candidate_timeout_s"
                    ],
                    candidate_id=f"{group['group_key']}-{family}",
                    containment=surface["containment"],
                    gpu_device=gpu_device,
                )
            finally:
                batch_evaluate_krea._remove_comfy_lora(
                    staging, comfy_root=Path(surface["comfy_root"])
                )
        if process.returncode != 0 or not result_path.is_file():
            raise RuntimeError(
                f"exact scorer failed {group['group_key']} {family}: "
                f"rc={process.returncode}, stderr={process.stderr[-2000:]}"
            )
        status_body = {
            "schema": SCHEMA,
            "kind": STATUS_KIND,
            "score_plan_sha256": plan["plan_sha256"],
            "family_id": family,
            "gpu_device": gpu_device,
            "command_sha256": krea_provenance.canonical_sha256(command),
            "returncode": process.returncode,
            "stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(process.stderr.encode()).hexdigest(),
            "fallback_used": False,
            "release_authorized": False,
        }
        status = {
            **status_body,
            "status_sha256": krea_provenance.canonical_sha256(status_body),
        }
        status_path = Path(job["status_path"])
        _publish_or_replay(status_path, status, "score command status")
        log_path = Path(str(result_path) + ".comfy.log")
        evidence_body = {
            "schema": SCHEMA,
            "kind": EVIDENCE_KIND,
            "score_plan_sha256": plan["plan_sha256"],
            "family_id": family,
            "candidate_sha256": krea_provenance.file_sha256(candidate_path),
            "result_sha256": krea_provenance.file_sha256(result_path),
            "comfy_log_sha256": krea_provenance.file_sha256(log_path),
            "status_file_sha256": krea_provenance.file_sha256(status_path),
            "release_authorized": False,
            "production_mutation_authorized": False,
        }
        evidence_manifest = {
            **evidence_body,
            "evidence_sha256": krea_provenance.canonical_sha256(evidence_body),
        }
        evidence_path = Path(job["evidence_manifest_path"])
        _publish_or_replay(evidence_path, evidence_manifest, "score command evidence")
        completed = datetime.now(timezone.utc).replace(microsecond=0)
        created = datetime.strptime(
            plan["created_at_utc"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
        if completed <= created:
            completed = created + timedelta(seconds=1)
        receipt = krea_stage2_score.build_receipt(
            plan=plan,
            family_id=family,
            candidate_path=candidate_path,
            fixture_manifest=fixture,
            fixture_manifest_file_sha256=fixture_file_sha,
            result_path=result_path,
            status_file_sha256=krea_provenance.file_sha256(status_path),
            evidence_manifest_file_sha256=krea_provenance.file_sha256(evidence_path),
            completed_at_utc=completed.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        _publish_or_replay(receipt_path, receipt, "score receipt")
        command_evidence = _validate_job_evidence(
            job=job, plan=plan, gpu_device=gpu_device
        )
        if any(
            receipt[key] != command_evidence[key]
            for key in ("status_file_sha256", "evidence_manifest_file_sha256")
        ):
            raise ValueError("score receipt differs from command evidence files")
        receipts.append(receipt)
        score_files[family] = {
            "candidate_path": job["candidate_path"],
            "result_path": job["result_path"],
        }
    latest = max(
        datetime.strptime(row["completed_at_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        for row in receipts
    )
    aggregate = krea_stage2_score.build_aggregate(
        plan=plan,
        receipts=receipts,
        emitted_at_utc=(latest + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    aggregate_path = Path(group["aggregate_path"])
    _publish_or_replay(aggregate_path, aggregate, "score aggregate")
    return krea_stage2_score.validate_aggregate_with_score_files(
        aggregate,
        plan=plan,
        fixture_manifest=fixture,
        fixture_manifest_file_sha256=fixture_file_sha,
        score_files_by_family=score_files,
    )


def _validate_claim(value: Any, *, score_queue: Mapping[str, Any]) -> dict[str, Any]:
    queue = _validate_queue(score_queue)
    claim = _object(value, "score claim")
    _exact(
        claim,
        {
            "schema",
            "kind",
            "score_queue_sha256",
            "group_key",
            "group_path",
            "gpu_device",
            "scheduler_instance_id",
            "claimed_at_utc",
            "release_authorized",
            "production_mutation_authorized",
            "claim_sha256",
        },
        "score claim",
    )
    body = {key: item for key, item in claim.items() if key != "claim_sha256"}
    queued = [row for row in queue["groups"] if row["group_key"] == claim["group_key"]]
    if (
        claim["schema"] != SCHEMA
        or claim["kind"] != CLAIM_KIND
        or claim["claim_sha256"] != krea_provenance.canonical_sha256(body)
        or claim["score_queue_sha256"] != queue["score_queue_sha256"]
        or len(queued) != 1
        or claim["group_path"] != queued[0]["group_path"]
        or claim["gpu_device"] not in krea_stage2_endgame_orchestrator.GPU_IDS
        or claim["release_authorized"] is not False
        or claim["production_mutation_authorized"] is not False
    ):
        raise ValueError("score claim identity/authority differs")
    _safe_id(claim["scheduler_instance_id"], "score scheduler instance id")
    krea_stage2_execution._utc(claim["claimed_at_utc"], "score claim time")
    return dict(claim)


def claim_ready_groups(
    *,
    score_queue: Mapping[str, Any],
    claims_root: str | Path,
    claimed_at_utc: str,
    scheduler_instance_id: str,
    gpu_devices: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Assign ready groups to free GPUs without waiting for all training rows."""

    queue = _validate_queue(score_queue)
    root = Path(os.path.abspath(os.path.expanduser(str(claims_root))))
    root.mkdir(parents=True, exist_ok=True)
    active_gpus = set()
    for path in sorted(root.glob("*.json")):
        claim = _validate_claim(_load(path, "existing score claim"), score_queue=queue)
        queued = next(
            row for row in queue["groups"] if row["group_key"] == claim["group_key"]
        )
        if not os.path.lexists(queued["aggregate_path"]):
            active_gpus.add(claim["gpu_device"])
    ready = [
        row
        for row in queue["groups"]
        if os.path.lexists(row["group_path"])
        and not os.path.lexists(row["aggregate_path"])
        and not os.path.lexists(root / f"{row['group_key']}.json")
    ]
    selected_gpus = tuple(
        krea_stage2_endgame_orchestrator.GPU_IDS
        if gpu_devices is None
        else gpu_devices
    )
    if (
        len(selected_gpus) != len(set(selected_gpus))
        or any(
            gpu not in krea_stage2_endgame_orchestrator.GPU_IDS
            for gpu in selected_gpus
        )
    ):
        raise ValueError("score claim GPU subset is invalid")
    claims = []
    for gpu in selected_gpus:
        if gpu in active_gpus or not ready:
            continue
        group = ready.pop(0)
        body = {
            "schema": SCHEMA,
            "kind": CLAIM_KIND,
            "score_queue_sha256": queue["score_queue_sha256"],
            "group_key": group["group_key"],
            "group_path": group["group_path"],
            "gpu_device": gpu,
            "scheduler_instance_id": _safe_id(
                scheduler_instance_id, "score scheduler instance id"
            ),
            "claimed_at_utc": krea_stage2_execution._utc(
                claimed_at_utc, "score claim time"
            ),
            "release_authorized": False,
            "production_mutation_authorized": False,
        }
        claim = {**body, "claim_sha256": krea_provenance.canonical_sha256(body)}
        krea_stage2_endgame_matrix._publish_new(
            root / f"{group['group_key']}.json", claim
        )
        claims.append(_validate_claim(claim, score_queue=queue))
    return claims


def run_score_claim(
    *,
    config: Mapping[str, Any],
    score_queue: Mapping[str, Any],
    claim: Mapping[str, Any],
    gpu_lock_root: str | Path,
) -> dict[str, Any]:
    queue = _validate_queue(score_queue)
    record = _validate_claim(claim, score_queue=queue)
    group = _validate_group(
        _load(record["group_path"], "claimed score group"), score_queue=queue
    )
    if os.path.lexists(group["aggregate_path"]):
        raise ValueError("score claim is already complete")
    return run_group(
        config=config,
        group=group,
        gpu_device=record["gpu_device"],
        gpu_lock_root=gpu_lock_root,
    )


def seal_score_gate(
    *,
    score_queue: Mapping[str, Any],
    output: str | Path,
    completed_at_utc: str,
) -> dict[str, Any]:
    queue = _validate_queue(score_queue)
    missing = [
        group["group_key"]
        for group in queue["groups"]
        if not os.path.lexists(group["group_path"])
        or not os.path.lexists(group["aggregate_path"])
    ]
    if missing:
        raise ValueError(
            "score gate cannot launch work; missing aggregates: " + ",".join(missing)
        )
    aggregates = []
    for queued in queue["groups"]:
        group = _validate_group(
            _load(queued["group_path"], "gate score group"), score_queue=queue
        )
        plan = krea_stage2_score.validate_plan(
            _load(group["plan_path"], "gate score plan")
        )
        if (
            _file_sha(plan) != group["plan_file_sha256"]
            or plan["plan_sha256"] != group["plan_sha256"]
        ):
            raise ValueError("gate score plan bytes differ from score group")
        fixture = _load(group["fixture_manifest_path"], "gate fixture manifest")
        fixture_file_sha = _file_sha(fixture)
        score_files = {
            job["family_id"]: {
                "candidate_path": job["candidate_path"],
                "result_path": job["result_path"],
            }
            for job in group["jobs"]
        }
        aggregate = krea_stage2_score.validate_aggregate_with_score_files(
            _load(group["aggregate_path"], "gate score aggregate"),
            plan=plan,
            fixture_manifest=fixture,
            fixture_manifest_file_sha256=fixture_file_sha,
            score_files_by_family=score_files,
        )
        if len(aggregate["receipts"]) != FAMILIES_PER_GROUP:
            raise ValueError("score aggregate is not the exact five-family set")
        receipts = {row["family_id"]: row for row in aggregate["receipts"]}
        for job in group["jobs"]:
            command_evidence = _validate_job_evidence(
                job=job, plan=plan, gpu_device=None
            )
            receipt = receipts[job["family_id"]]
            if any(
                receipt[key] != command_evidence[key]
                for key in ("status_file_sha256", "evidence_manifest_file_sha256")
            ):
                raise ValueError("gate receipt differs from command evidence files")
        aggregates.append(
            {
                "group_key": group["group_key"],
                "plan_sha256": plan["plan_sha256"],
                "aggregate_file_sha256": krea_provenance.file_sha256(
                    Path(group["aggregate_path"])
                ),
                "aggregate_sha256": aggregate["aggregate_sha256"],
            }
        )
    completed = krea_stage2_execution._utc(
        completed_at_utc, "score gate completion time"
    )
    if completed <= max(
        _load(row["aggregate_path"], "gate aggregate time")["emitted_at_utc"]
        for row in queue["groups"]
    ):
        raise ValueError("score completion gate predates an aggregate")
    body = {
        "schema": SCHEMA,
        "kind": GATE_KIND,
        "score_queue_sha256": queue["score_queue_sha256"],
        "completed_at_utc": completed,
        "group_count": GROUP_COUNT,
        "receipt_count": GROUP_COUNT * FAMILIES_PER_GROUP,
        "aggregates": aggregates,
        "all_live_score_aggregates_replayed": True,
        "boundary_scoring_enabled": False,
        "release_authorized": False,
        "production_mutation_authorized": False,
    }
    gate = {**body, "gate_sha256": krea_provenance.canonical_sha256(body)}
    return _publish_or_replay(output, gate, "score completion gate")


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    materialize = sub.add_parser("materialize-ready")
    materialize.add_argument("--config", required=True, type=Path)
    materialize.add_argument("--output-root", required=True, type=Path)
    run = sub.add_parser("run-group")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--group", required=True, type=Path)
    run.add_argument("--gpu-device", required=True, type=int)
    run.add_argument("--gpu-lock-root", required=True, type=Path)
    claim = sub.add_parser("claim-ready")
    claim.add_argument("--score-queue", required=True, type=Path)
    claim.add_argument("--claims-root", required=True, type=Path)
    claim.add_argument("--claimed-at-utc", required=True)
    claim.add_argument("--scheduler-instance-id", required=True)
    run_claim = sub.add_parser("run-claim")
    run_claim.add_argument("--config", required=True, type=Path)
    run_claim.add_argument("--score-queue", required=True, type=Path)
    run_claim.add_argument("--claim", required=True, type=Path)
    run_claim.add_argument("--gpu-lock-root", required=True, type=Path)
    gate = sub.add_parser("gate")
    gate.add_argument("--score-queue", required=True, type=Path)
    gate.add_argument("--completed-at-utc", required=True)
    gate.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    try:
        if args.command == "materialize-ready":
            result: Any = materialize_ready_score_plans(
                _load(args.config, "score config"), output_root=args.output_root
            )
        elif args.command == "run-group":
            result = run_group(
                config=_load(args.config, "score config"),
                group=_load(args.group, "score group"),
                gpu_device=args.gpu_device,
                gpu_lock_root=args.gpu_lock_root,
            )
        elif args.command == "claim-ready":
            result = claim_ready_groups(
                score_queue=_load(args.score_queue, "score queue"),
                claims_root=args.claims_root,
                claimed_at_utc=args.claimed_at_utc,
                scheduler_instance_id=args.scheduler_instance_id,
            )
        elif args.command == "run-claim":
            result = run_score_claim(
                config=_load(args.config, "score config"),
                score_queue=_load(args.score_queue, "score queue"),
                claim=_load(args.claim, "score claim"),
                gpu_lock_root=args.gpu_lock_root,
            )
        else:
            result = seal_score_gate(
                score_queue=_load(args.score_queue, "score queue"),
                output=args.output,
                completed_at_utc=args.completed_at_utc,
            )
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
