#!/usr/bin/env python3
"""Authority-derived producer and scheduler for the Week-5 Krea endgame.

The module deliberately does not create timing evidence or reveal fixtures.  It
consumes only complete, replayable timing bundles and an admitted Stage-2
authority bundle.  It then creates exactly sixty row plans/approvals, exposes
collision-free serial queues for the four GPUs, streams completed rows to the
exact scorer, and seals an exact-60 completion gate after live replay.

There is no waiver, synthetic-profile, hand-written-plan, or release path.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence

from forge import recipe

try:
    from . import krea_budget
    from . import krea_fixture
    from . import krea_provenance
    from . import krea_stage2_endgame_matrix
    from . import krea_stage2_execution
    from . import krea_stage2_legacy_confirmation
    from . import krea_stage2_production_identity
    from . import krea_stage2_timing
    from . import krea_stage2_training_evidence
except ImportError:  # pragma: no cover - direct CLI execution.
    import krea_budget  # type: ignore[no-redef]
    import krea_fixture  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_stage2_endgame_matrix  # type: ignore[no-redef]
    import krea_stage2_execution  # type: ignore[no-redef]
    import krea_stage2_legacy_confirmation  # type: ignore[no-redef]
    import krea_stage2_production_identity  # type: ignore[no-redef]
    import krea_stage2_timing  # type: ignore[no-redef]
    import krea_stage2_training_evidence  # type: ignore[no-redef]


SCHEMA = 1
CONFIG_KIND = "forge-krea-stage2-endgame-producer-config"
PLAN_SET_KIND = "forge-krea-stage2-endgame-plan-set"
CLAIM_KIND = "forge-krea-stage2-endgame-gpu-claim"
GATE_KIND = "forge-krea-stage2-endgame-exact60-gate"
EXPECTED_ROWS = 60
GPU_IDS = (0, 1, 2, 3)
CLASS_REPRESENTATIVES = ("K1", "K3", "K4")
EXECUTION_PLAN_REVIEWER = {
    "actor_id": "codex-week5-stage2-execution-plan-reviewer",
    "display_name": "Codex Week-5 Stage-2 execution-plan reviewer (agent)",
    "role": "execution_plan_reviewer",
    "review_instance_id": "week5-krea-stage2-execution-plan-review-20260801",
    "non_human": True,
}
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


def _canonical_file_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(krea_provenance.canonical_bytes(value) + b"\n").hexdigest()


def _load(path: str | Path, label: str) -> dict[str, Any]:
    return krea_stage2_endgame_matrix._load_canonical(path, label)


def _publish_or_replay(
    path: str | Path, value: Mapping[str, Any], label: str
) -> dict[str, Any]:
    return krea_stage2_endgame_matrix._publish_or_replay(path, value, label)


def _utc(value: Any, label: str) -> str:
    return krea_stage2_execution._utc(value, label)


def _timing_class(family: str) -> str:
    return krea_stage2_execution.krea_calibration_profiles.profile_for_id(
        family
    ).throughput_equivalence_class


@contextmanager
def gpu_execution_lock(root: str | Path, gpu_device: int):
    """Serialize training and scoring on one physical GPU across processes."""

    if gpu_device not in GPU_IDS:
        raise ValueError("GPU execution lock device is outside 0-3")
    directory = Path(os.path.abspath(os.path.expanduser(str(root))))
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("GPU execution lock root is not a real directory")
    path = directory / f"gpu{gpu_device}.lock"
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("GPU execution lock is not one regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"GPU {gpu_device} is already executing work") from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _expected_timing_keys(matrix: Mapping[str, Any]) -> set[str]:
    fixture_gpu_hours: set[tuple[str, int, str]] = set()
    for row in matrix["rows"]:
        hours = krea_stage2_execution._HOURS[row["cell_id"]]
        fixture_gpu_hours.add((row["fixture_id"], row["gpu_device"], hours))
    classes = {_timing_class(family) for family in CLASS_REPRESENTATIVES}
    return {
        f"{fixture}__gpu{gpu}__h{hours.replace('.', 'p')}__{equivalence}"
        for fixture, gpu, hours in fixture_gpu_hours
        for equivalence in classes
    }


def _timing_key(row: Mapping[str, Any], family: str) -> str:
    hours = krea_stage2_execution._HOURS[row["cell_id"]]
    return (
        f"{row['fixture_id']}__gpu{row['gpu_device']}__h{hours.replace('.', 'p')}__"
        f"{_timing_class(family)}"
    )


def _validate_authority_bundle(value: Any) -> dict[str, Any]:
    controls = _object(value, "Stage-2 authority bundle")
    required = {
        "request",
        "request_file_sha256",
        "ratification",
        "ratification_file_sha256",
        "reveal",
        "reveal_file_sha256",
        "materialization",
        "materialization_file_sha256",
        "gpu_execution_authorization",
        "gpu_execution_authorization_file_sha256",
        "production_identity",
        "production_identity_file_sha256",
        "waiver_finalist_freeze",
        "sealed_inventory",
        "sealed_inventory_file_sha256",
    }
    _exact(controls, required, "Stage-2 authority bundle")
    for name in (
        "request",
        "ratification",
        "reveal",
        "materialization",
        "gpu_execution_authorization",
        "production_identity",
        "sealed_inventory",
    ):
        record = _object(controls[name], f"authority {name}")
        if _canonical_file_sha(record) != _sha(
            controls[f"{name}_file_sha256"], f"authority {name} file"
        ):
            raise ValueError(f"authority {name} file binding differs")
    krea_stage2_endgame_matrix._validate_production_identity(
        controls["production_identity"]
    )
    return dict(controls)


def _artifact_identity(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(os.path.abspath(os.path.expanduser(str(path))))
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be a live regular file")
    size = source.stat().st_size
    if size <= 0:
        raise ValueError(f"{label} is empty")
    return {
        "path": str(source),
        "bytes": size,
        "sha256": krea_provenance.file_sha256(source),
    }


def _candidate_catalog(
    value: Any, *, freeze: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    catalog = _object(value, "candidate artifact catalog")
    if set(catalog) != set(krea_stage2_endgame_matrix.FAMILY_ORDER):
        raise ValueError("candidate catalog must contain K0-K5 exactly")
    rules = _object(freeze["all_family_checkpoint_rules"], "freeze family rules")
    resolved: dict[str, dict[str, Any]] = {}
    for family in krea_stage2_endgame_matrix.FAMILY_ORDER:
        supplied = _object(catalog[family], f"candidate {family}")
        _exact(supplied, {"candidate_id", "step", "path"}, f"candidate {family}")
        identity = _artifact_identity(supplied["path"], f"candidate {family}")
        observed = {
            (
                row.get("candidate_id"),
                row.get("candidate_sha256"),
                row.get("step"),
            )
            for row in rules[family]["actual_mappings"]
            if isinstance(row, dict)
        }
        exact = (supplied["candidate_id"], identity["sha256"], supplied["step"])
        if exact not in observed:
            raise ValueError(f"candidate {family} is absent from the freeze")
        resolved[family] = {
            "candidate_id": _safe_id(supplied["candidate_id"], "candidate id"),
            "family_id": family,
            "sha256": identity["sha256"],
            "bytes": identity["bytes"],
            "step": supplied["step"],
            "zero_control": False,
        }
    return resolved


def _zero_candidate(manifest_path: str | Path) -> dict[str, Any]:
    manifest = _load(manifest_path, "zero-control manifest")
    manifest = krea_stage2_training_evidence.validate_zero_control(manifest)
    artifact = manifest["artifact"]
    return {
        "candidate_id": "zero-control",
        "family_id": "ZERO",
        "sha256": artifact["sha256"],
        "bytes": artifact["bytes"],
        "step": None,
        "zero_control": True,
    }


def _binding(path: Path, semantic_key: str) -> dict[str, str]:
    record = _load(path, path.name)
    return {
        "path": str(path),
        "file_sha256": _canonical_file_sha(record),
        semantic_key: _sha(record[semantic_key], f"{path.name} semantic"),
    }


def _timing_catalog(
    value: Any, *, matrix: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    catalog = _object(value, "timing bundle catalog")
    expected = _expected_timing_keys(matrix)
    if set(catalog) != expected:
        raise ValueError(
            f"timing catalog is not exact: missing={sorted(expected - set(catalog))}, "
            f"extra={sorted(set(catalog) - expected)}"
        )
    resolved: dict[str, dict[str, Any]] = {}
    for key in sorted(catalog):
        entry = _object(catalog[key], f"timing entry {key}")
        _exact(entry, {"bundle_root", "probe_contract"}, f"timing entry {key}")
        root = Path(os.path.abspath(os.path.expanduser(entry["bundle_root"])))
        loaded = krea_stage2_timing.load_timing_bundle(root)
        bundle = loaded["bundle"]
        plan = krea_stage2_timing.validate_plan(loaded["plan"])
        probe_path = Path(
            os.path.abspath(os.path.expanduser(str(entry["probe_contract"])))
        )
        probe = _load(probe_path, f"timing probe contract {key}")
        probe = krea_stage2_timing.validate_probe_contract(probe, plan=plan)
        if (
            plan["probe_contract"]
            != {
                "file_sha256": _canonical_file_sha(probe),
                "probe_contract_sha256": probe["probe_contract_sha256"],
            }
            or plan["production_image_id"] != matrix["production_image_id"]
        ):
            raise ValueError(f"timing entry {key} escaped its probe/image")
        components = key.split("__")
        fixture = components[0]
        gpu = int(components[1].removeprefix("gpu"))
        hours = components[2].removeprefix("h").replace("p", ".")
        equivalence = components[3]
        if (
            plan["fixture_role"] != fixture
            or probe["command_fields"]["gpu_device"] != gpu
            or str(plan["hard_budget_s"]) != str(float(hours) * 3600)
            or plan["calibration_profile"]["throughput_equivalence_class"]
            != equivalence
            or bundle["throughput_equivalence_class"] != equivalence
        ):
            raise ValueError(f"timing entry {key} has a mismatched envelope")
        profile_path = root / "throughput-profile.json"
        raw_path = root / "raw-samples.json"
        margin_path = root / "margin-policy.json"
        e2e_path = root / "end-to-end.json"
        profile = krea_budget.load_throughput_profile(
            _load(profile_path, f"timing profile {key}")
        )
        resolved[key] = {
            "profile": {
                "path": str(profile_path),
                "file_sha256": krea_provenance.file_sha256(profile_path),
                "profile_sha256": profile.profile_sha256,
            },
            "evidence": {
                "raw_samples": _binding(raw_path, "raw_sample_manifest_sha256"),
                "margin_policy": _binding(margin_path, "margin_policy_sha256"),
                "end_to_end_validation": _binding(
                    e2e_path, "end_to_end_validation_sha256"
                ),
            },
            "bundle_sha256": bundle["bundle_sha256"],
            "timing_plan_sha256": plan["plan_sha256"],
            "gpu_device": gpu,
        }
    return resolved


def _fixture_binding(path: str | Path, expected_role: str) -> tuple[dict, dict]:
    source = Path(os.path.abspath(os.path.expanduser(str(path))))
    document = _load(source, f"fixture {expected_role}")
    if expected_role in krea_stage2_legacy_confirmation.ROLES:
        wrapper = krea_stage2_legacy_confirmation.validate_wrapper(document)
        fixture = krea_stage2_legacy_confirmation.score_view(wrapper)
        semantic = wrapper["wrapper_sha256"]
    else:
        fixture = krea_fixture.validate_manifest(document)
        semantic = fixture["manifest_sha256"]
    if fixture["experimental_role"] != expected_role:
        raise ValueError(f"fixture path for {expected_role} contains another role")
    return fixture, {
        "path": str(source),
        "file_sha256": _canonical_file_sha(document),
        "manifest_sha256": semantic,
    }


def _authority_plan_fields(controls: Mapping[str, Any]) -> dict[str, Any]:
    auth = controls["gpu_execution_authorization"]
    return {
        "waiver_finalist_freeze": {
            "file_sha256": auth["waiver_freeze_file_sha256"],
            "freeze_sha256": auth["waiver_freeze_sha256"],
        },
        "confirmation_materialization": {
            "file_sha256": auth["materialization_file_sha256"],
            "materialization_sha256": auth["materialization_sha256"],
        },
        "owner_ratification": {
            "file_sha256": auth["ratification_file_sha256"],
            "ratification_sha256": auth["ratification_sha256"],
        },
        "gpu_execution_authorization": {
            "file_sha256": controls["gpu_execution_authorization_file_sha256"],
            "gpu_execution_authorization_sha256": auth[
                "gpu_execution_authorization_sha256"
            ],
        },
        "production_identity": {
            "file_sha256": auth["production_identity_file_sha256"],
            "production_identity_sha256": auth["production_identity_sha256"],
        },
        "execution_surface_policy_sha256": auth["policy_sha256"],
        "delegated_role_contract_sha256": auth["delegated_review_contract_sha256"],
        "production_image_id": auth["image_id"],
    }


def _plan_for_row(
    *,
    row: Mapping[str, Any],
    matrix: Mapping[str, Any],
    controls: Mapping[str, Any],
    fixture_paths: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
    zero: Mapping[str, Any],
    timing: Mapping[str, Mapping[str, Any]],
    mounts: Mapping[str, Any],
    asset_attestation_path: str | Path,
    created_at_utc: str,
) -> tuple[dict[str, Any], str]:
    family = row["family_id"]
    fixture, fixture_binding = _fixture_binding(
        fixture_paths[row["fixture_id"]], row["fixture_id"]
    )
    timing_key = _timing_key(row, family if family != "K0" else "K1")
    timing_entry = timing[timing_key]
    profile = krea_budget.load_throughput_profile(
        _load(timing_entry["profile"]["path"], "row throughput profile")
    )
    hours = krea_stage2_execution._HOURS[row["cell_id"]]
    if family == "K0":
        planned_steps = recipe.size_scaled_steps(
            "krea2",
            len(fixture["training_dataset_identity"]["rows"]),
            float(hours),
            template_steps=2000,
        )
        throughput_profile = None
        throughput_evidence = None
    else:
        planned_steps = krea_budget.plan_budget(
            profile, hard_budget_s=float(hours) * 3600
        ).max_affordable_steps
        throughput_profile = dict(timing_entry["profile"])
        throughput_evidence = dict(timing_entry["evidence"])
    rule = controls["waiver_finalist_freeze"]["all_family_checkpoint_rules"][family]
    selection = krea_stage2_execution._checkpoint_selection_for_rule(
        rule, planned_steps=planned_steps, profile_id=family
    )
    task_id = f"stage2-{row['cell_id'].lower()}"
    repo_name = f"stage2-{row['cell_id'].lower()}-{family.lower()}"
    mount_contract = []
    destinations = {
        "base_model": ("/cache/models/krea--Krea-2-Raw", True),
        "text_encoder": ("/cache/hf_cache/Qwen--Qwen3-VL-4B-Instruct", True),
        "dataset_cache": ("/cache/datasets", True),
        "checkpoints": ("/app/checkpoints", False),
        "run_evidence": ("/run-evidence", False),
    }
    if set(mounts) != set(destinations):
        raise ValueError("mount source map must contain the five exact purposes")
    for purpose in destinations:
        destination, read_only = destinations[purpose]
        mount_contract.append(
            {
                "source": str(
                    Path(os.path.abspath(os.path.expanduser(str(mounts[purpose]))))
                ),
                "destination": destination,
                "read_only": read_only,
                "purpose": purpose,
            }
        )
    identity = controls["production_identity"]
    asset_path = Path(os.path.abspath(os.path.expanduser(str(asset_attestation_path))))
    asset = krea_stage2_production_identity.load_asset_attestation(asset_path)
    entrypoint_argv = [
        "--task-id",
        task_id,
        "--model",
        "krea/Krea-2-Raw",
        "--model-type",
        "krea2",
        "--expected-repo-name",
        repo_name,
        "--hours-to-complete",
        hours,
    ]
    if fixture["trigger_token"] is not None:
        entrypoint_argv.extend(["--trigger-word", fixture["trigger_token"]])
    payload = {
        "schema": 1,
        "kind": krea_stage2_execution.PLAN_KIND,
        "phase": row["phase"],
        "cell_id": row["cell_id"],
        "fixture_id": row["fixture_id"],
        "seed_role": row["seed_role"],
        "seed": krea_stage2_execution._SEEDS[row["seed_role"]],
        "hours": hours,
        "task_id": task_id,
        "expected_repo_name": repo_name,
        "model": "krea/Krea-2-Raw",
        "model_type": "krea2",
        "trigger_word": fixture["trigger_token"],
        "candidate_universe": [dict(candidates[family]), dict(zero)],
        "training_candidate_id": candidates[family]["candidate_id"],
        "family_role": row["family_role"],
        "calibration_profile": family,
        "planned_steps": planned_steps,
        "checkpoint_selection": selection,
        "throughput_profile": throughput_profile,
        "throughput_evidence": throughput_evidence,
        "execution_environment_profile": dict(timing_entry["profile"]),
        "base_model_identity_sha256": identity["base_model"][
            "training_identity_sha256"
        ],
        "base_asset_attestation": {
            "path": str(asset_path),
            "file_sha256": _canonical_file_sha(asset),
            "attestation_sha256": asset["attestation_sha256"],
        },
        "fixture_manifest": fixture_binding,
        **_authority_plan_fields(controls),
        "entrypoint_argv": entrypoint_argv,
        "mounts": mount_contract,
        "network_mode": "none",
        "runtime": "nvidia",
        "created_at_utc": _utc(created_at_utc, "plan creation time"),
        "gpu_execution_authorized": True,
        "release_authorized": False,
        "production_mutation_authorized": False,
    }
    plan = krea_stage2_execution.seal_plan(payload)
    return plan, timing_key


def _validate_config(value: Any) -> dict[str, Any]:
    config = _object(value, "endgame producer config")
    _exact(
        config,
        {
            "schema",
            "kind",
            "matrix",
            "authority_bundle",
            "fixture_manifests",
            "candidate_artifacts",
            "zero_control_manifest",
            "timing_bundles",
            "mount_sources",
            "asset_attestation",
            "plan_created_at_utc",
            "approval_created_at_utc",
            "config_sha256",
        },
        "endgame producer config",
    )
    body = {key: item for key, item in config.items() if key != "config_sha256"}
    if (
        config["schema"] != SCHEMA
        or config["kind"] != CONFIG_KIND
        or config["config_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("producer config kind/schema/digest differs")
    _utc(config["plan_created_at_utc"], "plan creation time")
    _utc(config["approval_created_at_utc"], "approval creation time")
    return dict(config)


def produce(config: Mapping[str, Any], *, output_root: str | Path) -> dict[str, Any]:
    """Create/replay all sixty plans and approvals from one authority graph."""

    supplied = _validate_config(config)
    matrix = krea_stage2_endgame_matrix.validate_matrix(
        _load(supplied["matrix"], "endgame matrix")
    )
    if matrix["training_count"] != EXPECTED_ROWS or matrix[
        "active_variant_family_ids"
    ] != ["K1", "K5"]:
        raise ValueError("this endgame requires the frozen K1/K5 exact-60 matrix")
    controls = _validate_authority_bundle(
        _load(supplied["authority_bundle"], "authority bundle")
    )
    matrix = krea_stage2_endgame_matrix.validate_matrix(
        matrix,
        freeze=controls["waiver_finalist_freeze"],
        production_identity=controls["production_identity"],
    )
    candidates = _candidate_catalog(
        supplied["candidate_artifacts"],
        freeze=controls["waiver_finalist_freeze"],
    )
    zero = _zero_candidate(supplied["zero_control_manifest"])
    timing = _timing_catalog(supplied["timing_bundles"], matrix=matrix)
    root = Path(os.path.abspath(os.path.expanduser(str(output_root))))
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    actor = dict(EXECUTION_PLAN_REVIEWER)
    for row in matrix["rows"]:
        row_root = root / "rows" / row["row_key"]
        row_root.mkdir(parents=True, exist_ok=True)
        plan, timing_key = _plan_for_row(
            row=row,
            matrix=matrix,
            controls=controls,
            fixture_paths=supplied["fixture_manifests"],
            candidates=candidates,
            zero=zero,
            timing=timing,
            mounts=supplied["mount_sources"],
            asset_attestation_path=supplied["asset_attestation"],
            created_at_utc=supplied["plan_created_at_utc"],
        )
        approval = krea_stage2_execution.build_approval(
            plan,
            reviewer_actor=actor,
            approved_at_utc=supplied["approval_created_at_utc"],
        )
        krea_stage2_endgame_matrix.validate_row_controls(
            matrix=matrix,
            row_key=row["row_key"],
            plan=plan,
            approval=approval,
            authority_bundle=controls,
        )
        plan_path = row_root / "plan.json"
        approval_path = row_root / "approval.json"
        _publish_or_replay(plan_path, plan, "row plan")
        _publish_or_replay(approval_path, approval, "row approval")
        rows.append(
            {
                "row_key": row["row_key"],
                "wave_id": row["wave_id"],
                "gpu_device": row["gpu_device"],
                "timing_key": timing_key,
                "plan": {
                    "path": str(plan_path),
                    "file_sha256": _canonical_file_sha(plan),
                    "plan_sha256": plan["plan_sha256"],
                },
                "approval": {
                    "path": str(approval_path),
                    "file_sha256": _canonical_file_sha(approval),
                    "approval_sha256": approval["approval_sha256"],
                },
                "output_dir": str(row_root / "run"),
                "completion_path": str(row_root / "completion.json"),
                "run_evidence_path": str(row_root / "run-evidence.json"),
                "score_hook_path": str(row_root / "score-hook.json"),
                "receipt_path": str(row_root / "row-receipt.json"),
            }
        )
    queues = {
        str(gpu): [row["row_key"] for row in rows if row["gpu_device"] == gpu]
        for gpu in GPU_IDS
    }
    body = {
        "schema": SCHEMA,
        "kind": PLAN_SET_KIND,
        "config_sha256": supplied["config_sha256"],
        "matrix_sha256": matrix["matrix_sha256"],
        "production_image_id": matrix["production_image_id"],
        "training_count": len(rows),
        "score_stream_count": len(rows),
        "gpu_queues": queues,
        "waves": matrix["waves"],
        "rows": rows,
        "strict_authority_per_row": True,
        "waiver_path_available": False,
        "release_authorized": False,
        "production_mutation_authorized": False,
    }
    plan_set = {**body, "plan_set_sha256": krea_provenance.canonical_sha256(body)}
    validate_plan_set(plan_set, matrix=matrix)
    return _publish_or_replay(root / "plan-set.json", plan_set, "plan set")


def validate_plan_set(value: Any, *, matrix: Mapping[str, Any]) -> dict[str, Any]:
    plan_set = _object(value, "endgame plan set")
    keys = {
        "schema",
        "kind",
        "config_sha256",
        "matrix_sha256",
        "production_image_id",
        "training_count",
        "score_stream_count",
        "gpu_queues",
        "waves",
        "rows",
        "strict_authority_per_row",
        "waiver_path_available",
        "release_authorized",
        "production_mutation_authorized",
        "plan_set_sha256",
    }
    _exact(plan_set, keys, "endgame plan set")
    body = {key: item for key, item in plan_set.items() if key != "plan_set_sha256"}
    if (
        plan_set["schema"] != SCHEMA
        or plan_set["kind"] != PLAN_SET_KIND
        or plan_set["plan_set_sha256"] != krea_provenance.canonical_sha256(body)
        or plan_set["matrix_sha256"] != matrix["matrix_sha256"]
        or plan_set["production_image_id"] != matrix["production_image_id"]
        or plan_set["training_count"] != EXPECTED_ROWS
        or plan_set["score_stream_count"] != EXPECTED_ROWS
        or plan_set["strict_authority_per_row"] is not True
        or plan_set["waiver_path_available"] is not False
        or plan_set["release_authorized"] is not False
        or plan_set["production_mutation_authorized"] is not False
    ):
        raise ValueError("endgame plan-set identity/authority differs")
    rows = plan_set["rows"]
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise ValueError("endgame plan set must contain exactly 60 rows")
    row_keys = [row.get("row_key") for row in rows if isinstance(row, dict)]
    matrix_keys = [row["row_key"] for row in matrix["rows"]]
    if row_keys != matrix_keys or len(set(row_keys)) != EXPECTED_ROWS:
        raise ValueError("plan-set rows differ from the matrix")
    matrix_rows = {row["row_key"]: row for row in matrix["rows"]}
    row_schema = {
        "row_key",
        "wave_id",
        "gpu_device",
        "timing_key",
        "plan",
        "approval",
        "output_dir",
        "completion_path",
        "run_evidence_path",
        "score_hook_path",
        "receipt_path",
    }
    for row in rows:
        _exact(row, row_schema, f"plan-set row {row['row_key']}")
        source = matrix_rows[row["row_key"]]
        if (
            row["wave_id"] != source["wave_id"]
            or row["gpu_device"] != source["gpu_device"]
            or row["timing_key"]
            != _timing_key(
                source, source["family_id"] if source["family_id"] != "K0" else "K1"
            )
        ):
            raise ValueError("plan-set row differs from its matrix/timing envelope")
        for field, semantic in (
            ("plan", "plan_sha256"),
            ("approval", "approval_sha256"),
        ):
            binding = _object(row[field], f"row {field} binding")
            _exact(binding, {"path", "file_sha256", semantic}, f"row {field} binding")
            _sha(binding["file_sha256"], f"row {field} file")
            _sha(binding[semantic], f"row {field} semantic")
            path = Path(binding["path"])
            if not path.is_absolute() or str(path) != os.path.abspath(str(path)):
                raise ValueError(f"row {field} path is not absolute and normalized")
        for field in (
            "output_dir",
            "completion_path",
            "run_evidence_path",
            "score_hook_path",
            "receipt_path",
        ):
            path = Path(row[field])
            if not path.is_absolute() or str(path) != os.path.abspath(str(path)):
                raise ValueError(f"row {field} is not absolute and normalized")
    queues = _object(plan_set["gpu_queues"], "GPU queues")
    if set(queues) != {str(gpu) for gpu in GPU_IDS}:
        raise ValueError("GPU queues differ from the four-device surface")
    flattened = []
    for gpu in GPU_IDS:
        queue = queues[str(gpu)]
        expected = [row["row_key"] for row in rows if row["gpu_device"] == gpu]
        if queue != expected:
            raise ValueError(f"GPU {gpu} queue is not serial matrix order")
        flattened.extend(queue)
    if len(flattened) != EXPECTED_ROWS or set(flattened) != set(row_keys):
        raise ValueError("GPU queues omit or duplicate rows")
    if plan_set["waves"] != matrix["waves"]:
        raise ValueError("plan-set waves differ from the matrix")
    return dict(plan_set)


def claim_next(
    *,
    plan_set: Mapping[str, Any],
    matrix: Mapping[str, Any],
    claims_root: str | Path,
    claimed_at_utc: str,
    scheduler_instance_id: str,
    gpu_devices: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    resolved = validate_plan_set(plan_set, matrix=matrix)
    root = Path(os.path.abspath(os.path.expanduser(str(claims_root))))
    root.mkdir(parents=True, exist_ok=True)
    rows = {row["row_key"]: row for row in resolved["rows"]}
    first_open_wave = None
    for wave in resolved["waves"]:
        if any(
            not os.path.lexists(rows[key]["receipt_path"]) for key in wave["row_keys"]
        ):
            first_open_wave = wave
            break
    if first_open_wave is None:
        return []
    wave_keys = set(first_open_wave["row_keys"])
    selected_gpus = tuple(GPU_IDS if gpu_devices is None else gpu_devices)
    if (
        len(selected_gpus) != len(set(selected_gpus))
        or any(gpu not in GPU_IDS for gpu in selected_gpus)
    ):
        raise ValueError("training claim GPU subset is invalid")
    claims: list[dict[str, Any]] = []
    for gpu in selected_gpus:
        queue = resolved["gpu_queues"][str(gpu)]
        active = [
            key
            for key in queue
            if os.path.lexists(root / f"{key}.json")
            and not os.path.lexists(rows[key]["receipt_path"])
        ]
        if active:
            continue
        candidates = [
            key
            for key in queue
            if key in wave_keys
            and not os.path.lexists(rows[key]["receipt_path"])
            and not os.path.lexists(root / f"{key}.json")
        ]
        if not candidates:
            continue
        key = candidates[0]
        body = {
            "schema": SCHEMA,
            "kind": CLAIM_KIND,
            "plan_set_sha256": resolved["plan_set_sha256"],
            "wave_id": first_open_wave["wave_id"],
            "row_key": key,
            "gpu_device": gpu,
            "scheduler_instance_id": _safe_id(
                scheduler_instance_id, "scheduler instance id"
            ),
            "claimed_at_utc": _utc(claimed_at_utc, "claim time"),
            "waiver_used": False,
            "release_authorized": False,
        }
        claim = {**body, "claim_sha256": krea_provenance.canonical_sha256(body)}
        krea_stage2_endgame_matrix._publish_new(root / f"{key}.json", claim)
        claims.append(claim)
    return claims


def run_claim(
    *,
    claim: Mapping[str, Any],
    plan_set: Mapping[str, Any],
    matrix: Mapping[str, Any],
    authority_bundle: Mapping[str, Any],
    gpu_lock_root: str | Path,
) -> dict[str, Any]:
    resolved = validate_plan_set(plan_set, matrix=matrix)
    record = _validate_claim(claim, plan_set=resolved, matrix=matrix)
    row = next(row for row in resolved["rows"] if row["row_key"] == record["row_key"])
    with gpu_execution_lock(gpu_lock_root, row["gpu_device"]):
        receipt, replayed = krea_stage2_endgame_matrix.run_row(
            matrix=matrix,
            row_key=row["row_key"],
            plan=_load(row["plan"]["path"], "claimed row plan"),
            approval=_load(row["approval"]["path"], "claimed row approval"),
            authority_bundle=authority_bundle,
            output_dir=row["output_dir"],
            completion_path=row["completion_path"],
            run_evidence_path=row["run_evidence_path"],
            score_hook_path=row["score_hook_path"],
            receipt_path=row["receipt_path"],
        )
    return {"receipt": receipt, "replayed_existing": replayed}


def _validate_claim(
    claim: Mapping[str, Any],
    *,
    plan_set: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = validate_plan_set(plan_set, matrix=matrix)
    record = _object(claim, "GPU claim")
    _exact(
        record,
        {
            "schema",
            "kind",
            "plan_set_sha256",
            "wave_id",
            "row_key",
            "gpu_device",
            "scheduler_instance_id",
            "claimed_at_utc",
            "waiver_used",
            "release_authorized",
            "claim_sha256",
        },
        "GPU claim",
    )
    body = {key: item for key, item in record.items() if key != "claim_sha256"}
    if (
        record.get("schema") != SCHEMA
        or record.get("kind") != CLAIM_KIND
        or record.get("claim_sha256") != krea_provenance.canonical_sha256(body)
        or record.get("plan_set_sha256") != resolved["plan_set_sha256"]
        or record.get("waiver_used") is not False
        or record.get("release_authorized") is not False
    ):
        raise ValueError("GPU claim identity/authority differs")
    matches = [row for row in resolved["rows"] if row["row_key"] == record["row_key"]]
    if (
        len(matches) != 1
        or matches[0]["gpu_device"] != record["gpu_device"]
        or matches[0]["wave_id"] != record["wave_id"]
    ):
        raise ValueError("GPU claim differs from its fixed row")
    return dict(record)


def seal_exact60_gate(
    *,
    plan_set: Mapping[str, Any],
    matrix: Mapping[str, Any],
    authority_bundle: Mapping[str, Any],
    output: str | Path,
    completed_at_utc: str,
) -> dict[str, Any]:
    resolved = validate_plan_set(plan_set, matrix=matrix)
    missing = [
        row["row_key"]
        for row in resolved["rows"]
        if not os.path.lexists(row["receipt_path"])
    ]
    if missing:
        raise ValueError(
            "exact-60 gate cannot launch work; missing row receipts: "
            + ",".join(missing)
        )
    receipts = []
    hooks = []
    for row in resolved["rows"]:
        receipt, replayed = krea_stage2_endgame_matrix.run_row(
            matrix=matrix,
            row_key=row["row_key"],
            plan=_load(row["plan"]["path"], "gate row plan"),
            approval=_load(row["approval"]["path"], "gate row approval"),
            authority_bundle=authority_bundle,
            output_dir=row["output_dir"],
            completion_path=row["completion_path"],
            run_evidence_path=row["run_evidence_path"],
            score_hook_path=row["score_hook_path"],
            receipt_path=row["receipt_path"],
        )
        if replayed is not True:
            raise ValueError("exact-60 gate may only replay already completed rows")
        hook = _load(row["score_hook_path"], "gate score hook")
        receipts.append(
            {"row_key": row["row_key"], "receipt_sha256": receipt["receipt_sha256"]}
        )
        hooks.append({"row_key": row["row_key"], "hook_sha256": hook["hook_sha256"]})
    if len(receipts) != EXPECTED_ROWS or len(hooks) != EXPECTED_ROWS:
        raise ValueError("exact-60 gate did not exhaust the matrix")
    body = {
        "schema": SCHEMA,
        "kind": GATE_KIND,
        "plan_set_sha256": resolved["plan_set_sha256"],
        "matrix_sha256": matrix["matrix_sha256"],
        "completed_at_utc": _utc(completed_at_utc, "gate completion time"),
        "training_count": EXPECTED_ROWS,
        "score_stream_count": EXPECTED_ROWS,
        "receipts": receipts,
        "score_hooks": hooks,
        "strict_live_replay_complete": True,
        "waiver_used": False,
        "release_authorized": False,
        "production_mutation_authorized": False,
    }
    gate = {**body, "gate_sha256": krea_provenance.canonical_sha256(body)}
    return _publish_or_replay(output, gate, "exact-60 gate")


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    produce_cmd = sub.add_parser("produce")
    produce_cmd.add_argument("--config", required=True, type=Path)
    produce_cmd.add_argument("--output-root", required=True, type=Path)
    claim = sub.add_parser("claim")
    claim.add_argument("--plan-set", required=True, type=Path)
    claim.add_argument("--matrix", required=True, type=Path)
    claim.add_argument("--claims-root", required=True, type=Path)
    claim.add_argument("--claimed-at-utc", required=True)
    claim.add_argument("--scheduler-instance-id", required=True)
    run = sub.add_parser("run-claim")
    run.add_argument("--claim", required=True, type=Path)
    run.add_argument("--plan-set", required=True, type=Path)
    run.add_argument("--matrix", required=True, type=Path)
    run.add_argument("--authority-bundle", required=True, type=Path)
    run.add_argument("--gpu-lock-root", required=True, type=Path)
    gate = sub.add_parser("gate")
    gate.add_argument("--plan-set", required=True, type=Path)
    gate.add_argument("--matrix", required=True, type=Path)
    gate.add_argument("--authority-bundle", required=True, type=Path)
    gate.add_argument("--completed-at-utc", required=True)
    gate.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    try:
        if args.command == "produce":
            result: Any = produce(
                _load(args.config, "producer config"), output_root=args.output_root
            )
        else:
            matrix = krea_stage2_endgame_matrix.validate_matrix(
                _load(args.matrix, "endgame matrix")
            )
            plan_set = validate_plan_set(
                _load(args.plan_set, "endgame plan set"), matrix=matrix
            )
            if args.command == "claim":
                result = claim_next(
                    plan_set=plan_set,
                    matrix=matrix,
                    claims_root=args.claims_root,
                    claimed_at_utc=args.claimed_at_utc,
                    scheduler_instance_id=args.scheduler_instance_id,
                )
            elif args.command == "run-claim":
                result = run_claim(
                    claim=_load(args.claim, "GPU claim"),
                    plan_set=plan_set,
                    matrix=matrix,
                    authority_bundle=_load(args.authority_bundle, "authority bundle"),
                    gpu_lock_root=args.gpu_lock_root,
                )
            else:
                result = seal_exact60_gate(
                    plan_set=plan_set,
                    matrix=matrix,
                    authority_bundle=_load(args.authority_bundle, "authority bundle"),
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
