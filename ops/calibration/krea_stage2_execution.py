#!/usr/bin/env python3
"""Fail-closed Stage-2 execution plans and production-container receipts.

Stage-1 discovery evidence is intentionally not accepted here.  This module is
the narrow bridge from a reviewed waiver finalist freeze to the separately
ratified production-Docker confirmation surface.  It does not reveal fixtures,
select finalists, score candidates, or authorize a release.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

from forge import krea_calibration_profiles, recipe

try:
    from . import krea_confirmation_admission
    from . import krea_budget
    from . import krea_density_seedb_freeze
    from . import krea_fixture
    from . import krea_provenance
    from . import krea_stage2_production_identity
    from . import krea_stage2_admission_chain
    from . import krea_stage2_legacy_confirmation
    from . import krea_stage2_timing
    from . import krea_waiver_finalist_freeze
except ImportError:  # pragma: no cover - direct CLI execution.
    import krea_confirmation_admission  # type: ignore[no-redef]
    import krea_budget  # type: ignore[no-redef]
    import krea_density_seedb_freeze  # type: ignore[no-redef]
    import krea_fixture  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_stage2_production_identity  # type: ignore[no-redef]
    import krea_stage2_admission_chain  # type: ignore[no-redef]
    import krea_stage2_legacy_confirmation  # type: ignore[no-redef]
    import krea_stage2_timing  # type: ignore[no-redef]
    import krea_waiver_finalist_freeze  # type: ignore[no-redef]


PLAN_KIND = "forge-krea-stage2-cell-plan"
APPROVAL_KIND = "forge-krea-stage2-cell-approval"
COMPLETION_KIND = "forge-krea-stage2-cell-completion"
SCHEMA = 1

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHA1 = re.compile(r"[0-9a-f]{40}")
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_CONFIRMATION = {f"C{index}-{seed}" for index in range(1, 5) for seed in ("A", "B")}
_BOUNDARY = {
    f"B-{hours}-{size}" for hours in ("0p5", "0p75", "1") for size in ("small", "large")
}
_SEEDS = {"A": 42565431, "B": 309817421}
_HOURS = {
    **{cell: "0.75" for cell in _CONFIRMATION},
    **{
        f"B-{label}-{size}": value
        for label, value in (("0p5", "0.5"), ("0p75", "0.75"), ("1", "1.0"))
        for size in ("small", "large")
    },
}
_MOUNT_CONTRACT = {
    "base_model": ("/cache/models/krea--Krea-2-Raw", True),
    "text_encoder": ("/cache/hf_cache/Qwen--Qwen3-VL-4B-Instruct", True),
    "dataset_cache": ("/cache/datasets", True),
    "checkpoints": ("/app/checkpoints", False),
    "run_evidence": ("/run-evidence", False),
}
_PUBLIC_REFERENCE_FAMILIES = {"K2", "K3", "K4"}
_FAMILY_ROLES = {"candidate", "control", "public_reference"}
_STAGE2_PROFILE_SURFACE = "immutable_production_docker_image"
_STAGE2_PROFILE_SCOPE = "stage2_throughput_timing_only"
_KREA_MODEL = "krea/Krea-2-Raw"
_CHECKPOINT_MAPPING_RULE = "nearest_current_candidate_ties_choose_earlier_step"
_TARGET_FRACTION_NUMERATOR_ENV = "FORGE_KREA_STAGE2_TARGET_FRACTION_NUMERATOR"
_TARGET_FRACTION_DENOMINATOR_ENV = "FORGE_KREA_STAGE2_TARGET_FRACTION_DENOMINATOR"

# Single-GPU image-trainer resource/security contract from G.O.D
# b026da04b6179cf82945e8736590dd923114342b (trainer/runtime.py).  The Stage-2
# harness remains offline, so it deliberately keeps ``--network none`` rather
# than joining the validator's internal bridge.
_VALIDATOR_IMAGE_TRAINER_SHM_SIZE = "8g"
_VALIDATOR_IMAGE_TRAINER_MEMORY = "110g"
_VALIDATOR_IMAGE_TRAINER_CPUS = "24"


def _read_identity_file(path: Path, label: str) -> str:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError
        value = path.read_text(encoding="ascii").strip().lower()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if not value:
        raise ValueError(f"{label} is empty")
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def live_stage2_host_identity(checkpoint_path: Path) -> dict[str, Any]:
    """Return the stable host facts shared by timing and a later cell launch."""

    path = Path(os.path.abspath(checkpoint_path))
    if path.is_symlink() or not path.is_dir():
        raise ValueError("checkpoint host identity path must be a real directory")
    mem_total = None
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemTotal:"):
            fields = line.split()
            if len(fields) == 3 and fields[2] == "kB":
                mem_total = int(fields[1]) * 1024
            break
    if mem_total is None or mem_total <= 0:
        raise ValueError("host memory identity is unavailable")
    stat_result = path.stat()
    affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 0))
    )
    if not affinity:
        raise ValueError("host CPU identity is unavailable")
    record = {
        "schema": 1,
        "kind": "forge-krea-stage2-live-host-identity",
        "machine_id_sha256": _read_identity_file(Path("/etc/machine-id"), "machine-id"),
        "boot_id_sha256": _read_identity_file(
            Path("/proc/sys/kernel/random/boot_id"), "boot-id"
        ),
        "kernel_release": platform.release(),
        "machine": platform.machine(),
        "cpu_affinity_ids": affinity,
        "memory_total_bytes": mem_total,
        "checkpoint_device": {
            "st_dev": stat_result.st_dev,
            "major": os.major(stat_result.st_dev),
            "minor": os.minor(stat_result.st_dev),
        },
    }
    return {
        **record,
        "host_execution_identity_sha256": krea_provenance.canonical_sha256(record),
    }


def live_stage2_gpu_identity(gpu_device: int) -> dict[str, Any]:
    """Return the exact selected GPU facts used by Stage-2 timing and execution."""

    if (
        isinstance(gpu_device, bool)
        or not isinstance(gpu_device, int)
        or gpu_device < 0
    ):
        raise ValueError("gpu_device must be a nonnegative integer")
    command = [
        "nvidia-smi",
        "-i",
        str(gpu_device),
        "--query-gpu=uuid,name,driver_version,memory.total,compute_cap,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("selected GPU identity is unavailable") from exc
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise ValueError("selected GPU identity must contain exactly one row")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 6 or any(not field for field in fields):
        raise ValueError("selected GPU identity row is malformed")
    record = {
        "schema": 1,
        "kind": "forge-krea-stage2-live-gpu-identity",
        "uuid": fields[0],
        "name": fields[1],
        "driver_version": fields[2],
        "memory_total_mib": fields[3],
        "compute_capability": fields[4],
        "pci_bus_id": fields[5],
    }
    return {**record, "gpu_identity_sha256": krea_provenance.canonical_sha256(record)}


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(value: dict[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(
            f"{label} keys differ: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise ValueError(f"{label} must be canonical UTC")
    datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return value


def _utc_value(value: str) -> datetime:
    _utc(value, "timestamp")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} is not a safe identifier")
    return value


def _binding(value: Any, label: str, *, semantic_key: str) -> dict[str, str]:
    value = _object(value, label)
    _exact(value, {"file_sha256", semantic_key}, label)
    return {
        "file_sha256": _sha(value["file_sha256"], f"{label}.file_sha256"),
        semantic_key: _sha(value[semantic_key], f"{label}.{semantic_key}"),
    }


def _actor(value: Any, label: str) -> dict[str, str]:
    value = _object(value, label)
    _exact(
        value,
        {"actor_id", "display_name", "role", "review_instance_id", "non_human"},
        label,
    )
    if value["non_human"] is not True:
        raise ValueError(f"{label} must explicitly be a non-human delegated actor")
    return {
        "actor_id": _safe_id(value["actor_id"], f"{label}.actor_id"),
        "display_name": str(value["display_name"]).strip(),
        "role": _safe_id(value["role"], f"{label}.role"),
        "review_instance_id": _safe_id(
            value["review_instance_id"], f"{label}.review_instance_id"
        ),
        "non_human": True,
    }


def _candidate_universe(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("candidate_universe must be a non-empty array")
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    digests: set[str] = set()
    zero_count = 0
    for index, raw in enumerate(value):
        row = _object(raw, f"candidate_universe[{index}]")
        _exact(
            row,
            {"candidate_id", "family_id", "sha256", "bytes", "step", "zero_control"},
            f"candidate_universe[{index}]",
        )
        candidate_id = _safe_id(row["candidate_id"], "candidate_id")
        family = _safe_id(row["family_id"], "family_id")
        digest = _sha(row["sha256"], "candidate sha256")
        if candidate_id in ids or digest in digests:
            raise ValueError("candidate_universe repeats an id or artifact")
        if (
            isinstance(row["bytes"], bool)
            or not isinstance(row["bytes"], int)
            or row["bytes"] <= 0
        ):
            raise ValueError("candidate bytes must be a positive integer")
        if row["step"] is not None and (
            isinstance(row["step"], bool)
            or not isinstance(row["step"], int)
            or row["step"] <= 0
        ):
            raise ValueError("candidate step must be null or a positive integer")
        if not isinstance(row["zero_control"], bool):
            raise ValueError("zero_control must be boolean")
        if row["zero_control"]:
            zero_count += 1
            if family != "ZERO" or row["step"] is not None:
                raise ValueError("zero control must use family ZERO and null step")
        ids.add(candidate_id)
        digests.add(digest)
        rows.append(dict(row))
    if zero_count != 1:
        raise ValueError("candidate_universe must contain exactly one zero control")
    return rows


def _reduced_target_fraction(value: Any, label: str) -> tuple[int, int]:
    """Return one exact reduced rational from a frozen JSON number.

    Frozen checkpoint rules encode their target as a JSON number.  Converting
    through ``str`` preserves that published decimal spelling instead of
    importing the binary representation of a Python float into the authority
    record.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        raise ValueError(f"{label} must be a finite fraction inside [0,1]")
    try:
        fraction = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{label} must be a finite fraction inside [0,1]") from exc
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"{label} must be a finite fraction inside (0,1]")
    return fraction.numerator, fraction.denominator


def _selected_checkpoint_step(
    *,
    planned_steps: int,
    numerator: int,
    denominator: int,
    profile_id: str | None = None,
) -> int:
    """Map a frozen target to the real full-run cadence using integers only."""

    if profile_id == "K0":
        save_every = recipe.kill_safe_save_every(planned_steps, 200)
        candidate_steps = list(range(save_every, planned_steps, save_every))
        candidate_steps.append(planned_steps)
        candidate_steps = sorted(set(candidate_steps))
    else:
        candidate_steps = [
            candidate.step
            for candidate in krea_budget.candidate_schedule(planned_steps).candidates
        ]
    # Every candidate shares denominator ``planned_steps``.  Multiplying the
    # two rationals removes division and floating-point ambiguity.  The step is
    # the second key so an exact midpoint deterministically selects the earlier
    # current-run checkpoint.
    return min(
        candidate_steps,
        key=lambda step: (
            abs(step * denominator - numerator * planned_steps),
            step,
        ),
    )


def _checkpoint_selection_for_rule(
    rule: Any, *, planned_steps: int, profile_id: str | None = None
) -> dict[str, Any]:
    """Derive the sole plan binding allowed for one frozen checkpoint rule."""

    rule = _object(rule, "frozen checkpoint rule")
    numerator, denominator = _reduced_target_fraction(
        rule.get("target_fraction"), "frozen checkpoint target_fraction"
    )
    return {
        "checkpoint_rule_sha256": krea_provenance.canonical_sha256(rule),
        "target_fraction": {
            "numerator": numerator,
            "denominator": denominator,
        },
        "selected_step": _selected_checkpoint_step(
            planned_steps=planned_steps,
            numerator=numerator,
            denominator=denominator,
            profile_id=profile_id,
        ),
        "denominator_steps": planned_steps,
        "mapping_rule": _CHECKPOINT_MAPPING_RULE,
    }


def _checkpoint_selection(
    value: Any, *, planned_steps: int, profile_id: str | None = None
) -> dict[str, Any]:
    selection = _object(value, "checkpoint_selection")
    _exact(
        selection,
        {
            "checkpoint_rule_sha256",
            "target_fraction",
            "selected_step",
            "denominator_steps",
            "mapping_rule",
        },
        "checkpoint_selection",
    )
    target = _object(selection["target_fraction"], "checkpoint target_fraction")
    _exact(target, {"numerator", "denominator"}, "checkpoint target_fraction")
    numerator = target["numerator"]
    denominator = target["denominator"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or numerator <= 0
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
        or numerator > denominator
        or math.gcd(numerator, denominator) != 1
    ):
        raise ValueError("checkpoint target_fraction must be reduced and inside (0,1]")
    selected_step = selection["selected_step"]
    denominator_steps = selection["denominator_steps"]
    if (
        isinstance(selected_step, bool)
        or not isinstance(selected_step, int)
        or selected_step <= 0
        or isinstance(denominator_steps, bool)
        or not isinstance(denominator_steps, int)
        or denominator_steps != planned_steps
        or selected_step > denominator_steps
    ):
        raise ValueError("checkpoint selection step/depth differs from planned_steps")
    expected_step = _selected_checkpoint_step(
        planned_steps=planned_steps,
        numerator=numerator,
        denominator=denominator,
        profile_id=profile_id,
    )
    if (
        selected_step != expected_step
        or selection["mapping_rule"] != _CHECKPOINT_MAPPING_RULE
    ):
        raise ValueError("checkpoint selection does not map its declared target")
    return {
        "checkpoint_rule_sha256": _sha(
            selection["checkpoint_rule_sha256"], "checkpoint rule SHA-256"
        ),
        "target_fraction": {
            "numerator": numerator,
            "denominator": denominator,
        },
        "selected_step": selected_step,
        "denominator_steps": denominator_steps,
        "mapping_rule": _CHECKPOINT_MAPPING_RULE,
    }


def _runtime_checkpoint_selection(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the exact checkpoint binding emitted by the production image.

    The owner-frozen plan binds the target and selected step.  The runtime
    receipt additionally records every checkpoint step that the effective
    config can produce, so replay must derive that list from the effective
    profile cadence instead of trusting either private receipt.
    """

    planned_steps = plan["planned_steps"]
    selection = _checkpoint_selection(
        plan["checkpoint_selection"],
        planned_steps=planned_steps,
        profile_id=plan["calibration_profile"],
    )
    frozen = krea_calibration_profiles.profile_for_id(plan["calibration_profile"])
    save_every = (
        recipe.kill_safe_save_every(planned_steps, 200)
        if frozen.profile_id == "K0"
        else (planned_steps + 7) // 8
    )
    candidate_steps = list(range(save_every, planned_steps, save_every))
    candidate_steps.append(planned_steps)
    candidate_steps = sorted(set(candidate_steps))
    numerator = selection["target_fraction"]["numerator"]
    denominator = selection["target_fraction"]["denominator"]
    selected_step = min(
        candidate_steps,
        key=lambda step: (
            abs(step * denominator - planned_steps * numerator),
            step,
        ),
    )
    if selected_step != selection["selected_step"]:
        raise ValueError(
            "checkpoint selection differs from the effective runtime cadence"
        )
    return {
        "schema": 1,
        "mapping_rule": _CHECKPOINT_MAPPING_RULE,
        "target_fraction": {
            "numerator": numerator,
            "denominator": denominator,
        },
        "planned_steps": planned_steps,
        "selected_step": selected_step,
        "candidate_steps": candidate_steps,
    }


def _mounts(value: Any, *, require_live_sources: bool = False) -> list[dict[str, Any]]:
    """Validate the only four mounts allowed beside the immutable image.

    Docker's bind syntax makes a permissive destination effectively equivalent
    to an entrypoint override: a caller could mount over ``/app/forge`` while
    retaining the approved image digest.  Keep the table exact, and re-check
    live host sources immediately before launch.
    """

    if not isinstance(value, list) or len(value) != len(_MOUNT_CONTRACT):
        raise ValueError("mounts must contain exactly the five production surfaces")
    purposes: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        row = _object(raw, f"mounts[{index}]")
        _exact(
            row, {"source", "destination", "read_only", "purpose"}, f"mounts[{index}]"
        )
        raw_source = row["source"]
        raw_destination = row["destination"]
        if (
            not isinstance(raw_source, str)
            or not raw_source
            or raw_source != raw_source.strip()
            or "\x00" in raw_source
            or "," in raw_source
            or not isinstance(raw_destination, str)
            or not raw_destination
            or raw_destination != raw_destination.strip()
            or "\x00" in raw_destination
            or "," in raw_destination
        ):
            raise ValueError("mount paths must be canonical and Docker-safe")
        source = Path(raw_source)
        destination = PurePosixPath(raw_destination)
        purpose = _safe_id(row["purpose"], "mount purpose")
        expected = _MOUNT_CONTRACT.get(purpose)
        if (
            expected is None
            or not source.is_absolute()
            or not destination.is_absolute()
            or ".." in destination.parts
            or str(source) != os.path.abspath(raw_source)
            or str(destination) != raw_destination
        ):
            raise ValueError("mount paths must be absolute and traversal-free")
        if purpose in purposes:
            raise ValueError("mount purpose is duplicated")
        if not isinstance(row["read_only"], bool):
            raise ValueError("mount read_only must be boolean")
        expected_destination, expected_read_only = expected
        if (
            str(destination) != expected_destination
            or row["read_only"] is not expected_read_only
        ):
            raise ValueError(f"mount {purpose} differs from its fixed destination/mode")
        if require_live_sources:
            current = source
            while True:
                if current.is_symlink():
                    raise ValueError(f"mount source has a symlink component: {source}")
                if current == current.parent:
                    break
                current = current.parent
            try:
                resolved = source.resolve(strict=True)
                mode = source.stat().st_mode
            except OSError as exc:
                raise ValueError(f"mount source is unavailable: {source}") from exc
            if resolved != source or not stat.S_ISDIR(mode):
                raise ValueError(f"mount source must be a real directory: {source}")
        purposes.add(purpose)
        normalized.append(
            {
                "source": str(source),
                "destination": str(destination),
                "read_only": row["read_only"],
                "purpose": purpose,
            }
        )
    if purposes != set(_MOUNT_CONTRACT):
        raise ValueError("mounts differ from the exact production surface set")
    sources = [
        Path(row["source"]).resolve(strict=require_live_sources) for row in normalized
    ]
    for index, left in enumerate(sources):
        for right in sources[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    "mount sources must be pairwise distinct and non-overlapping"
                )
    return normalized


def _validate_cell(
    phase: str, cell_id: str, fixture_id: str, seed_role: str, seed: int, hours: str
) -> None:
    if phase == "confirmation":
        if cell_id not in _CONFIRMATION:
            raise ValueError("confirmation cell is outside C1-C4 x A/B")
        expected_fixture, expected_role = cell_id.split("-", 1)
        if fixture_id != expected_fixture or seed_role != expected_role:
            raise ValueError("confirmation fixture/seed differs from cell")
    elif phase == "boundary":
        if cell_id not in _BOUNDARY or fixture_id != cell_id or seed_role != "A":
            raise ValueError("boundary cell differs from the fixed 3x2 matrix")
    else:
        raise ValueError("Stage-2 phase must be confirmation or boundary")
    if seed != _SEEDS[seed_role] or hours != _HOURS[cell_id]:
        raise ValueError("cell seed or hours differs from the fixed matrix")


def validate_plan(value: Any) -> dict[str, Any]:
    plan = _object(value, "Stage-2 execution plan")
    keys = {
        "schema",
        "kind",
        "phase",
        "cell_id",
        "fixture_id",
        "seed_role",
        "seed",
        "hours",
        "task_id",
        "expected_repo_name",
        "model",
        "model_type",
        "trigger_word",
        "candidate_universe",
        "training_candidate_id",
        "family_role",
        "calibration_profile",
        "planned_steps",
        "checkpoint_selection",
        "throughput_profile",
        "throughput_evidence",
        "execution_environment_profile",
        "base_model_identity_sha256",
        "base_asset_attestation",
        "fixture_manifest",
        "waiver_finalist_freeze",
        "confirmation_materialization",
        "owner_ratification",
        "gpu_execution_authorization",
        "production_identity",
        "execution_surface_policy_sha256",
        "delegated_role_contract_sha256",
        "production_image_id",
        "entrypoint_argv",
        "mounts",
        "network_mode",
        "runtime",
        "created_at_utc",
        "gpu_execution_authorized",
        "release_authorized",
        "production_mutation_authorized",
        "plan_sha256",
    }
    _exact(plan, keys, "Stage-2 execution plan")
    body = {key: item for key, item in plan.items() if key != "plan_sha256"}
    if plan["schema"] != SCHEMA or plan["kind"] != PLAN_KIND:
        raise ValueError("Stage-2 execution plan kind/schema differs")
    if plan["plan_sha256"] != krea_provenance.canonical_sha256(body):
        raise ValueError("Stage-2 execution plan digest differs")
    phase = plan["phase"]
    cell_id = _safe_id(plan["cell_id"], "cell_id")
    fixture_id = _safe_id(plan["fixture_id"], "fixture_id")
    seed_role = plan["seed_role"]
    if seed_role not in _SEEDS:
        raise ValueError("seed_role must be A or B")
    _validate_cell(
        phase, cell_id, fixture_id, seed_role, plan["seed"], str(plan["hours"])
    )
    task_id = _safe_id(plan["task_id"], "task_id")
    expected_repo_name = _safe_id(plan["expected_repo_name"], "expected_repo_name")
    if plan["model_type"] != "krea2" or plan["model"] != _KREA_MODEL:
        raise ValueError("Stage-2 production plan must target exact Krea-2-Raw")
    if phase == "confirmation":
        if plan["trigger_word"] is not None:
            raise ValueError(
                "legacy C1-C4 confirmation must preserve the sealed null trigger"
            )
    elif (
        not isinstance(plan["trigger_word"], str)
        or not plan["trigger_word"]
        or plan["trigger_word"] != plan["trigger_word"].strip()
    ):
        raise ValueError("boundary trigger_word must be non-empty canonical text")
    _sha(plan["base_model_identity_sha256"], "base model identity")
    asset_binding = _object(
        plan["base_asset_attestation"], "base asset attestation binding"
    )
    _exact(
        asset_binding,
        {"path", "file_sha256", "attestation_sha256"},
        "base asset attestation binding",
    )
    asset_path = Path(str(asset_binding["path"]))
    if (
        not asset_path.is_absolute()
        or ".." in asset_path.parts
        or str(asset_path) != os.path.abspath(str(asset_binding["path"]))
    ):
        raise ValueError("base asset attestation path must be absolute and normalized")
    _sha(asset_binding["file_sha256"], "base asset attestation file")
    _sha(asset_binding["attestation_sha256"], "base asset attestation semantic")
    candidates = _candidate_universe(plan["candidate_universe"])
    training_candidate_id = _safe_id(
        plan["training_candidate_id"], "training_candidate_id"
    )
    selected = [
        row for row in candidates if row["candidate_id"] == training_candidate_id
    ]
    if len(selected) != 1 or selected[0]["zero_control"] is not False:
        raise ValueError(
            "training_candidate_id must select one non-zero frozen candidate"
        )
    selected_family = selected[0]["family_id"]
    family_role = plan["family_role"]
    if family_role not in _FAMILY_ROLES:
        raise ValueError("family_role must be candidate, control, or public_reference")
    if family_role == "control" and selected_family != "K0":
        raise ValueError("control execution must select K0")
    if family_role == "public_reference" and selected_family not in (
        _PUBLIC_REFERENCE_FAMILIES
    ):
        raise ValueError("public-reference execution must select K2, K3, or K4")
    if family_role == "candidate" and selected_family == "K0":
        raise ValueError("candidate execution cannot select K0")
    expected_repo = f"stage2-{cell_id.lower()}-{selected_family.lower()}"
    if expected_repo_name != expected_repo:
        raise ValueError(
            "Stage-2 repository namespace must uniquely include cell and family"
        )
    expected_task = f"stage2-{cell_id.lower()}"
    if task_id != expected_task:
        raise ValueError("Stage-2 task namespace must exactly identify its cell")
    fixture_binding = _object(plan["fixture_manifest"], "fixture manifest binding")
    _exact(
        fixture_binding,
        {"path", "file_sha256", "manifest_sha256"},
        "fixture manifest binding",
    )
    fixture_path = Path(str(fixture_binding["path"]))
    if (
        not fixture_path.is_absolute()
        or ".." in fixture_path.parts
        or str(fixture_path) != os.path.abspath(str(fixture_binding["path"]))
    ):
        raise ValueError("fixture manifest path must be absolute and normalized")
    _sha(fixture_binding["file_sha256"], "fixture manifest file")
    _sha(fixture_binding["manifest_sha256"], "fixture manifest semantic")
    if plan["calibration_profile"] != selected_family or selected_family not in {
        "K0",
        "K1",
        "K2",
        "K3",
        "K4",
        "K5",
    }:
        raise ValueError("Stage-2 calibration profile must equal its selected family")
    if phase == "boundary" and family_role != "candidate":
        raise ValueError("boundary execution must exercise the frozen candidate")
    environment_binding = _object(
        plan["execution_environment_profile"],
        "execution environment profile binding",
    )
    _exact(
        environment_binding,
        {"path", "file_sha256", "profile_sha256"},
        "execution environment profile binding",
    )
    environment_path = Path(str(environment_binding["path"]))
    if (
        not environment_path.is_absolute()
        or ".." in environment_path.parts
        or str(environment_path) != os.path.abspath(str(environment_binding["path"]))
    ):
        raise ValueError(
            "execution environment profile path must be absolute and normalized"
        )
    _sha(environment_binding["file_sha256"], "execution environment profile file")
    _sha(
        environment_binding["profile_sha256"],
        "execution environment profile semantic",
    )
    if selected_family == "K0":
        if (
            isinstance(plan["planned_steps"], bool)
            or not isinstance(plan["planned_steps"], int)
            or not 1 <= plan["planned_steps"] <= 5000
            or plan["throughput_profile"] is not None
            or plan["throughput_evidence"] is not None
        ):
            raise ValueError(
                "K0 confirmation must declare its exact release-control depth"
            )
    else:
        if (
            isinstance(plan["planned_steps"], bool)
            or not isinstance(plan["planned_steps"], int)
            or not 1 <= plan["planned_steps"] <= 5000
        ):
            raise ValueError("Stage-2 planned_steps must be inside [1,5000]")
        profile_binding = _object(
            plan["throughput_profile"], "throughput profile binding"
        )
        _exact(
            profile_binding,
            {"path", "file_sha256", "profile_sha256"},
            "throughput profile binding",
        )
        profile_path = Path(str(profile_binding["path"]))
        if (
            not profile_path.is_absolute()
            or ".." in profile_path.parts
            or str(profile_path) != os.path.abspath(str(profile_binding["path"]))
        ):
            raise ValueError("throughput profile path must be absolute and normalized")
        _sha(profile_binding["file_sha256"], "throughput profile file")
        _sha(profile_binding["profile_sha256"], "throughput profile semantic")
        evidence = _object(plan["throughput_evidence"], "throughput evidence")
        _exact(
            evidence,
            {"raw_samples", "margin_policy", "end_to_end_validation"},
            "throughput evidence",
        )
        for field, semantic_key in (
            ("raw_samples", "raw_sample_manifest_sha256"),
            ("margin_policy", "margin_policy_sha256"),
            ("end_to_end_validation", "end_to_end_validation_sha256"),
        ):
            item = _object(evidence[field], f"throughput evidence {field}")
            _exact(
                item,
                {"path", "file_sha256", semantic_key},
                f"throughput evidence {field}",
            )
            item_path = Path(str(item["path"]))
            if (
                not item_path.is_absolute()
                or ".." in item_path.parts
                or str(item_path) != os.path.abspath(str(item["path"]))
            ):
                raise ValueError(
                    "throughput evidence path must be absolute and normalized"
                )
            _sha(item["file_sha256"], f"throughput evidence {field} file")
            _sha(item[semantic_key], f"throughput evidence {field} semantic")
        if environment_binding != profile_binding:
            raise ValueError(
                "non-control execution environment must equal its depth profile"
            )
    _checkpoint_selection(
        plan["checkpoint_selection"],
        planned_steps=plan["planned_steps"],
        profile_id=plan["calibration_profile"],
    )
    _binding(
        plan["waiver_finalist_freeze"],
        "waiver finalist freeze",
        semantic_key="freeze_sha256",
    )
    _binding(
        plan["confirmation_materialization"],
        "confirmation materialization",
        semantic_key="materialization_sha256",
    )
    _binding(
        plan["owner_ratification"],
        "owner ratification",
        semantic_key="ratification_sha256",
    )
    _binding(
        plan["gpu_execution_authorization"],
        "GPU execution authorization",
        semantic_key="gpu_execution_authorization_sha256",
    )
    identity = _binding(
        plan["production_identity"],
        "production identity",
        semantic_key="production_identity_sha256",
    )
    del identity
    _sha(plan["execution_surface_policy_sha256"], "execution surface policy")
    _sha(plan["delegated_role_contract_sha256"], "delegated role contract")
    if (
        not isinstance(plan["production_image_id"], str)
        or _IMAGE_ID.fullmatch(plan["production_image_id"]) is None
    ):
        raise ValueError("production_image_id must be an exact sha256 image id")
    argv = plan["entrypoint_argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
    ):
        raise ValueError("entrypoint_argv must be a non-empty string array")
    expected_argv = [
        "--task-id",
        plan["task_id"],
        "--model",
        plan["model"],
        "--model-type",
        "krea2",
        "--expected-repo-name",
        plan["expected_repo_name"],
        "--hours-to-complete",
        plan["hours"],
    ]
    if plan["trigger_word"] is not None:
        expected_argv.extend(["--trigger-word", plan["trigger_word"]])
    if argv != expected_argv:
        raise ValueError(
            "entrypoint_argv must exactly match the single-use controlled grammar"
        )
    _mounts(plan["mounts"])
    if plan["network_mode"] != "none" or plan["runtime"] != "nvidia":
        raise ValueError("Stage-2 execution must be offline under NVIDIA runtime")
    _utc(plan["created_at_utc"], "created_at_utc")
    if _utc_value(plan["created_at_utc"]) > datetime.now(timezone.utc) + timedelta(
        seconds=60
    ):
        raise ValueError("Stage-2 plan cannot be future-dated")
    if plan["gpu_execution_authorized"] is not True:
        raise ValueError("Stage-2 plan lacks admitted GPU authority")
    if (
        plan["release_authorized"] is not False
        or plan["production_mutation_authorized"] is not False
    ):
        raise ValueError("Stage-2 execution cannot authorize release or mutation")
    return dict(plan)


def seal_plan(payload: dict[str, Any]) -> dict[str, Any]:
    if "plan_sha256" in payload:
        raise ValueError("unsealed plan must omit plan_sha256")
    plan = {**payload, "plan_sha256": krea_provenance.canonical_sha256(payload)}
    return validate_plan(plan)


def _canonical_json_file(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink component: {current}")
        if current == current.parent:
            break
        current = current.parent
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            chunks = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        raw = b"".join(chunks)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or len(raw) != before.st_size:
            raise ValueError(f"{label} changed while it was read")
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not a readable JSON file") from exc
    if not stat.S_ISREG(before.st_mode) or not isinstance(value, dict):
        raise ValueError(f"{label} must be a regular JSON object")
    if raw != krea_provenance.canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline")
    return value, hashlib.sha256(raw).hexdigest()


def _canonical_json_file_without_newline(
    path: Path, label: str
) -> tuple[dict[str, Any], str]:
    """Read the exact canonical-byte format used by checkpoint finalization."""

    path = Path(os.path.abspath(os.path.expanduser(path)))
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink component: {current}")
        if current == current.parent:
            break
        current = current.parent
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
        raw = b"".join(chunks)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or len(raw) != before.st_size:
            raise ValueError(f"{label} changed while it was read")
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not a readable JSON file") from exc
    if not stat.S_ISREG(before.st_mode) or not isinstance(value, dict):
        raise ValueError(f"{label} must be a regular JSON object")
    if raw != krea_provenance.canonical_bytes(value):
        raise ValueError(f"{label} must be canonical JSON without a newline")
    return value, hashlib.sha256(raw).hexdigest()


def _stable_regular_file_identity(path: Path, label: str) -> dict[str, Any]:
    """Hash one immutable-looking regular file without following symlinks."""

    path = Path(os.path.abspath(os.path.expanduser(path)))
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink component: {current}")
        if current == current.parent:
            break
        current = current.parent
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    size = 0
    try:
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{label} must be a regular file")
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or size != before.st_size:
        raise ValueError(f"{label} changed while it was read")
    if size <= 0:
        raise ValueError(f"{label} is empty")
    return {"bytes": size, "sha256": digest.hexdigest()}


def _validate_frozen_execution_family(
    plan: Mapping[str, Any],
    *,
    freeze: Any,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the self-declared execution family to the owner-ratified freeze."""

    record = _object(freeze, "waiver finalist freeze")
    body = {key: item for key, item in record.items() if key != "freeze_sha256"}
    freeze_contract = {
        krea_waiver_finalist_freeze.FREEZE_KIND: (
            krea_waiver_finalist_freeze.SCHEMA,
            krea_waiver_finalist_freeze.FALSE_CLAIMS,
            krea_waiver_finalist_freeze.AUTHORITY,
        ),
        krea_density_seedb_freeze.FREEZE_KIND: (
            krea_density_seedb_freeze.SCHEMA,
            krea_density_seedb_freeze.FALSE_CLAIMS,
            krea_density_seedb_freeze.AUTHORITY,
        ),
    }.get(record.get("kind"))
    expected_file_sha = hashlib.sha256(
        krea_provenance.canonical_bytes(record) + b"\n"
    ).hexdigest()
    if (
        freeze_contract is None
        or record.get("schema") != freeze_contract[0]
        or record.get("freeze_sha256") != krea_provenance.canonical_sha256(body)
        or record.get("freeze_sha256") != request["waiver_freeze_sha256"]
        or expected_file_sha != request["waiver_freeze_file_sha256"]
        or record.get("outcome") != "finalists_frozen"
        or record.get("blockers") != []
        or record.get("claims") != freeze_contract[1]
        or record.get("authority") != freeze_contract[2]
    ):
        raise ValueError("waiver finalist freeze differs from owner-ratified bytes")
    finalists = record.get("finalist_family_ids")
    all_rules = record.get("all_family_checkpoint_rules")
    checkpoint_rules = record.get("checkpoint_rules")
    if (
        not isinstance(finalists, list)
        or not finalists
        or len(finalists) != len(set(finalists))
        or "K0" not in finalists
        or any(
            family not in {"K0", "K1", "K2", "K3", "K4", "K5"} for family in finalists
        )
        or not isinstance(all_rules, dict)
        or set(all_rules) != {"K0", "K1", "K2", "K3", "K4", "K5"}
        or not isinstance(checkpoint_rules, dict)
        or set(checkpoint_rules) != set(finalists)
        or any(checkpoint_rules[family] != all_rules[family] for family in finalists)
    ):
        raise ValueError("waiver finalist freeze family/checkpoint rules are invalid")
    selected = next(
        row
        for row in plan["candidate_universe"]
        if row["candidate_id"] == plan["training_candidate_id"]
    )
    family = selected["family_id"]
    role = plan["family_role"]
    if role == "candidate" and (family == "K0" or family not in finalists):
        raise ValueError(
            "candidate execution family is not a frozen non-control finalist"
        )
    if role == "control" and family != "K0":
        raise ValueError("control execution differs from frozen K0")
    if role == "public_reference" and family not in _PUBLIC_REFERENCE_FAMILIES:
        raise ValueError("public-reference execution differs from frozen K2-K4")
    rule = _object(all_rules[family], f"frozen checkpoint rule {family}")
    mappings = rule.get("actual_mappings")
    if not isinstance(mappings, list) or not mappings:
        raise ValueError("frozen checkpoint rule has no actual mappings")
    identity = (selected["candidate_id"], selected["sha256"], selected["step"])
    observed = {
        (row.get("candidate_id"), row.get("candidate_sha256"), row.get("step"))
        for row in mappings
        if isinstance(row, dict)
    }
    if identity not in observed:
        raise ValueError("training candidate identity is absent from the frozen rule")
    expected_selection = _checkpoint_selection_for_rule(
        rule,
        planned_steps=plan["planned_steps"],
        profile_id=family,
    )
    if plan["checkpoint_selection"] != expected_selection:
        raise ValueError("checkpoint selection differs from the frozen report rule")
    return record


def _validate_fixture_and_archive(
    plan: Mapping[str, Any],
    *,
    materialization: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    binding = plan["fixture_manifest"]
    document, file_sha = _canonical_json_file(
        Path(binding["path"]), "Stage-2 fixture manifest"
    )
    if plan["phase"] == "confirmation":
        wrapper = krea_stage2_legacy_confirmation.validate_wrapper(document)
        if (
            file_sha != binding["file_sha256"]
            or wrapper["wrapper_sha256"] != binding["manifest_sha256"]
            or wrapper["experimental_role"] != plan["fixture_id"]
            or wrapper["trigger_token"] is not None
            or plan["trigger_word"] is not None
        ):
            raise ValueError("Stage-2 legacy wrapper differs from its execution cell")
        fixture = {
            "experimental_role": wrapper["experimental_role"],
            "trigger_token": None,
            "training_rows": [
                {"relative_image_path": row["relative_path"]}
                for row in wrapper["training_media_shapes"]
            ],
            "training_dataset_shape_sha256": wrapper[
                "training_dataset_shape_sha256"
            ],
            "training_archive": {
                key: wrapper["training_archive"][key]
                for key in ("sha256", "bytes")
            },
        }
    else:
        manifest = krea_fixture.validate_manifest(document)
        if (
            file_sha != binding["file_sha256"]
            or manifest["manifest_sha256"] != binding["manifest_sha256"]
            or manifest["experimental_role"] != plan["fixture_id"]
            or manifest["trigger_token"] != plan["trigger_word"]
            or not isinstance(plan["trigger_word"], str)
            or not plan["trigger_word"]
        ):
            raise ValueError("Stage-2 fixture manifest differs from its execution cell")
        fixture = manifest
    committed = [
        row
        for row in materialization["files"]
        if row["role"] == plan["fixture_id"]
        and row["sha256"] == file_sha
        and row["bytes"] == Path(binding["path"]).stat().st_size
    ]
    if len(committed) != 1:
        raise ValueError("fixture manifest is absent from admitted materialization")
    if plan["phase"] == "confirmation":
        if (
            request["public_commitment_sha256s"][plan["fixture_id"]]
            != wrapper["published_checksum_manifest"]["file_sha256"]
        ):
            raise ValueError(
                "confirmation checksum differs from its pre-finalist commitment"
            )
    elif (
        request["boundary_fixture_manifest_sha256s"][plan["fixture_id"]]
        != fixture["manifest_sha256"]
    ):
        raise ValueError("boundary fixture differs from its public admitted manifest")
    mounts = {
        row["purpose"]: row
        for row in _mounts(plan["mounts"], require_live_sources=True)
    }
    archive = Path(mounts["dataset_cache"]["source"]) / (
        str(plan["task_id"]) + "_tourn.zip"
    )
    current = archive
    while True:
        if current.is_symlink():
            raise ValueError("training archive has a symlink component")
        if current == current.parent:
            break
        current = current.parent
    try:
        archive_mode = archive.stat().st_mode
        archive_size = archive.stat().st_size
        archive_sha = krea_provenance.file_sha256(archive)
    except OSError as exc:
        raise ValueError(
            "training archive is absent from the dataset-cache mount"
        ) from exc
    if (
        not stat.S_ISREG(archive_mode)
        or archive_size != fixture["training_archive"]["bytes"]
        or archive_sha != fixture["training_archive"]["sha256"]
    ):
        raise ValueError("mounted training archive differs from the admitted fixture")
    return fixture


def _validate_throughput_depth(
    plan: Mapping[str, Any],
    *,
    fixture: Mapping[str, Any],
    production_identity: Mapping[str, Any],
) -> None:
    binding = plan["throughput_profile"]
    if binding is None:
        return
    profile_record, file_sha = _canonical_json_file(
        Path(binding["path"]), "Stage-2 throughput profile"
    )
    profile = krea_budget.load_throughput_profile(profile_record)
    if (
        file_sha != binding["file_sha256"]
        or profile.profile_sha256 != binding["profile_sha256"]
        or profile.to_record() != profile_record
    ):
        raise ValueError("throughput profile semantic binding drifted")
    evidence_records: dict[str, dict[str, Any]] = {}
    evidence = plan["throughput_evidence"]
    for field, semantic_key in (
        ("raw_samples", "raw_sample_manifest_sha256"),
        ("margin_policy", "margin_policy_sha256"),
        ("end_to_end_validation", "end_to_end_validation_sha256"),
    ):
        item = evidence[field]
        record, observed_file_sha = _canonical_json_file(
            Path(item["path"]), f"Stage-2 throughput evidence {field}"
        )
        if (
            observed_file_sha != item["file_sha256"]
            or record.get(semantic_key) != item[semantic_key]
        ):
            raise ValueError(f"throughput evidence {field} binding drifted")
        evidence_records[field] = record
    raw = krea_budget.load_timing_sample_manifest(evidence_records["raw_samples"])
    margin = krea_budget.load_margin_policy(evidence_records["margin_policy"])
    end_to_end = krea_budget.load_end_to_end_validation(
        evidence_records["end_to_end_validation"]
    )
    rebuilt = krea_budget.seal_throughput_profile_from_evidence(
        raw_sample_manifest=raw,
        margin_policy=margin,
        end_to_end_validation=end_to_end,
        framework_stop_boundary_s=profile.framework_stop_boundary_s,
        framework_stop_boundary_source_sha256=(
            profile.framework_stop_boundary_source_sha256
        ),
        selection_mode=profile.selection_mode,
        selection_scorer_identity_sha256=profile.selection_scorer_identity_sha256,
        selection_scoring_reserve_s=profile.selection_scoring_reserve_s,
    )
    if rebuilt != profile_record:
        raise ValueError("throughput profile is not derived from its bound evidence")
    selected_family = plan["calibration_profile"]
    frozen = krea_calibration_profiles.profile_for_id(selected_family)
    envelope = profile.execution_envelope
    optimizer_sha = krea_provenance.canonical_sha256(dict(frozen.optimizer_parameters))
    expected_image_sha = str(plan["production_image_id"]).split(":", 1)[1]
    expected_resolution_sha = krea_provenance.canonical_sha256([512, 768, 1024])
    expected_precision_sha = krea_provenance.canonical_sha256(
        {"train_dtype": "bf16", "save_dtype": "bf16"}
    )
    base_model = _object(production_identity["base_model"], "production base model")
    runtime_contract = _object(
        production_identity["runtime_contract"], "production runtime contract"
    )
    if (
        envelope.equivalence_class != frozen.throughput_equivalence_class
        or envelope.network_rank != frozen.rank
        or envelope.network_alpha != frozen.alpha
        or envelope.optimizer != frozen.optimizer
        or envelope.optimizer_config_sha256 != optimizer_sha
        or envelope.loss != frozen.loss
        or envelope.differential_guidance_enabled is not True
        or envelope.guidance_scale != frozen.guidance
        or envelope.training_pair_count != len(fixture["training_rows"])
        or envelope.training_dataset_shape_sha256
        != fixture["training_dataset_shape_sha256"]
        or envelope.micro_batch_size != 1
        or envelope.gradient_accumulation_steps != 1
        or envelope.data_parallel_replicas != 1
        or envelope.resolution_policy_sha256 != expected_resolution_sha
        or envelope.precision_policy_sha256 != expected_precision_sha
        or envelope.cache_latents_to_disk is not False
        or envelope.cache_text_embeddings is not False
        or envelope.compile_enabled is not False
        or envelope.jit_enabled is not runtime_contract["jit_enabled"]
        or envelope.dataloader_workers != 0
        or envelope.base_model_identity_sha256 != plan["base_model_identity_sha256"]
        or plan["base_model_identity_sha256"] != base_model["training_identity_sha256"]
        or envelope.execution_surface != _STAGE2_PROFILE_SURFACE
        or envelope.execution_scope != _STAGE2_PROFILE_SCOPE
        or envelope.reference_container_image_sha256 != expected_image_sha
        or envelope.runtime_identity_sha256
        != runtime_contract["runtime_identity_sha256"]
        or envelope.venv_tree_manifest_sha256
        != runtime_contract["venv_tree_manifest_sha256"]
        or envelope.trainer_identity_sha256
        != runtime_contract["trainer_identity_sha256"]
        or envelope.measurement_tool_sha256
        != runtime_contract["measurement_tool_sha256"]
    ):
        raise ValueError(
            "throughput profile does not match the exact family/fixture/image runtime"
        )
    budget = krea_budget.plan_budget(
        profile, hard_budget_s=Decimal(str(plan["hours"])) * Decimal(3600)
    )
    if plan["planned_steps"] != budget.max_affordable_steps:
        raise ValueError("planned_steps differs from the measured maximal budget")


def _validate_execution_environment_profile(
    plan: Mapping[str, Any],
    *,
    fixture: Mapping[str, Any],
    production_identity: Mapping[str, Any],
) -> None:
    """Bind every cell, including K0, to its measured machine/runtime envelope."""

    binding = plan["execution_environment_profile"]
    record, file_sha = _canonical_json_file(
        Path(binding["path"]), "Stage-2 execution environment profile"
    )
    profile = krea_budget.load_throughput_profile(record)
    if (
        file_sha != binding["file_sha256"]
        or profile.profile_sha256 != binding["profile_sha256"]
        or profile.to_record() != record
    ):
        raise ValueError("execution environment profile semantic binding drifted")
    envelope = profile.execution_envelope
    expected_image_sha = str(plan["production_image_id"]).split(":", 1)[1]
    expected_resolution_sha = krea_provenance.canonical_sha256([512, 768, 1024])
    expected_precision_sha = krea_provenance.canonical_sha256(
        {"train_dtype": "bf16", "save_dtype": "bf16"}
    )
    base_model = _object(production_identity["base_model"], "production base model")
    runtime_contract = _object(
        production_identity["runtime_contract"], "production runtime contract"
    )
    if (
        envelope.training_pair_count != len(fixture["training_rows"])
        or envelope.training_dataset_shape_sha256
        != fixture["training_dataset_shape_sha256"]
        or envelope.micro_batch_size != 1
        or envelope.gradient_accumulation_steps != 1
        or envelope.data_parallel_replicas != 1
        or envelope.resolution_policy_sha256 != expected_resolution_sha
        or envelope.precision_policy_sha256 != expected_precision_sha
        or envelope.cache_latents_to_disk is not False
        or envelope.cache_text_embeddings is not False
        or envelope.compile_enabled is not False
        or envelope.jit_enabled is not runtime_contract["jit_enabled"]
        or envelope.dataloader_workers != 0
        or envelope.base_model_identity_sha256 != plan["base_model_identity_sha256"]
        or plan["base_model_identity_sha256"] != base_model["training_identity_sha256"]
        or envelope.execution_surface != _STAGE2_PROFILE_SURFACE
        or envelope.execution_scope != _STAGE2_PROFILE_SCOPE
        or envelope.reference_container_image_sha256 != expected_image_sha
        or envelope.runtime_identity_sha256
        != runtime_contract["runtime_identity_sha256"]
        or envelope.venv_tree_manifest_sha256
        != runtime_contract["venv_tree_manifest_sha256"]
        or envelope.trainer_identity_sha256
        != runtime_contract["trainer_identity_sha256"]
        or envelope.measurement_tool_sha256
        != runtime_contract["measurement_tool_sha256"]
    ):
        raise ValueError(
            "execution environment profile differs from fixture/image/runtime"
        )


def _validate_live_throughput_environment(
    plan: Mapping[str, Any], *, checkpoint_mount: Path, gpu_device: int
) -> None:
    """Rebind host/GPU timing axes immediately before the production launch."""

    binding = plan["execution_environment_profile"]
    record, file_sha = _canonical_json_file(
        Path(binding["path"]), "Stage-2 live throughput profile"
    )
    profile = krea_budget.load_throughput_profile(record)
    if (
        file_sha != binding["file_sha256"]
        or profile.profile_sha256 != binding["profile_sha256"]
    ):
        raise ValueError("live throughput profile binding drifted")
    measured_host, measured_gpu_sha, probe_checkpoint_device = (
        _load_bound_timing_probe_environment(
            profile_path=Path(binding["path"]),
            profile_record=record,
            profile_file_sha256=file_sha,
        )
    )
    host = live_stage2_host_identity(checkpoint_mount)
    gpu = live_stage2_gpu_identity(gpu_device)
    envelope = profile.execution_envelope
    stable_host_fields = {
        key: value
        for key, value in host.items()
        if key not in {"checkpoint_device", "host_execution_identity_sha256"}
    }
    measured_stable_host_fields = {
        key: value
        for key, value in measured_host.items()
        if key not in {"checkpoint_device", "host_execution_identity_sha256"}
    }
    if (
        envelope.host_execution_identity_sha256
        != measured_host["host_execution_identity_sha256"]
        or stable_host_fields != measured_stable_host_fields
        or host["checkpoint_device"] != probe_checkpoint_device
        or envelope.gpu_identity_sha256 != measured_gpu_sha
        or envelope.gpu_identity_sha256 != gpu["gpu_identity_sha256"]
    ):
        raise ValueError("live host/GPU differs from the measured timing envelope")


def _load_bound_timing_probe_environment(
    *,
    profile_path: Path,
    profile_record: Mapping[str, Any],
    profile_file_sha256: str,
) -> tuple[dict[str, Any], str, dict[str, int]]:
    """Recover the actual probe mount from its fully bound timing evidence.

    Older timing capture recorded the host receipt against the persistent
    evidence directory even though the probe wrote checkpoints to ephemeral
    storage.  The timing bundle and its exact probe contract are authoritative:
    this loader validates both and returns the device of the probe's real
    checkpoint source without weakening any other host or GPU identity axis.
    """

    source = Path(os.path.abspath(os.path.expanduser(profile_path)))
    bundle_root = source.parent
    if (
        source.name != "throughput-profile.json"
        or bundle_root.parent.name != "bundles"
    ):
        raise ValueError("live throughput profile is outside a timing bundle")
    loaded = krea_stage2_timing.load_timing_bundle(bundle_root)
    if (
        loaded["root"] != str(bundle_root)
        or loaded["throughput_profile"] != dict(profile_record)
        or krea_provenance.file_sha256(source) != profile_file_sha256
    ):
        raise ValueError("live throughput profile escaped its timing bundle")
    timing_plan = krea_stage2_timing.validate_plan(loaded["plan"])

    timing_root = bundle_root.parent.parent
    prepared_root = timing_root / "prepared"
    if prepared_root.is_symlink() or not prepared_root.is_dir():
        raise ValueError("timing probe controls directory is unavailable")
    probe_binding = _object(timing_plan["probe_contract"], "timing probe binding")
    target_file_sha = _sha(probe_binding["file_sha256"], "timing probe binding file")
    matches: list[tuple[Path, dict[str, Any]]] = []
    for candidate_root in prepared_root.iterdir():
        if candidate_root.is_symlink() or not candidate_root.is_dir():
            raise ValueError("timing probe controls inventory is not real directories")
        candidate = candidate_root / "probe-contract.json"
        if not candidate.exists():
            continue
        probe, candidate_sha = _canonical_json_file(candidate, "timing probe contract")
        if candidate_sha == target_file_sha:
            matches.append((candidate_root, probe))
    if len(matches) != 1:
        raise ValueError("timing probe binding does not resolve exactly once")
    prepared, probe = matches[0]
    controls, _ = _canonical_json_file(
        prepared / "timing-controls.json", "timing probe controls"
    )
    prepared_plan, prepared_plan_sha = _canonical_json_file(
        prepared / "timing-plan.json", "prepared timing plan"
    )
    bundle_plan_path = bundle_root / "timing-plan.json"
    _, bundle_plan_sha = _canonical_json_file(bundle_plan_path, "bundle timing plan")
    if prepared_plan != timing_plan or prepared_plan_sha != bundle_plan_sha:
        raise ValueError("prepared timing plan differs from its sealed bundle")
    krea_stage2_timing.validate_plan_with_controls(timing_plan, controls=controls)
    probe = krea_stage2_timing.validate_probe_contract(probe, plan=timing_plan)
    if (
        controls.get("probe_contract") != probe
        or controls.get("probe_contract_file_sha256") != target_file_sha
        or probe.get("probe_contract_sha256")
        != probe_binding.get("probe_contract_sha256")
    ):
        raise ValueError("timing probe contract differs from bound controls")

    measured_host = _object(
        controls.get("live_host_identity"), "timing host identity"
    )
    host_body = {
        key: value
        for key, value in measured_host.items()
        if key != "host_execution_identity_sha256"
    }
    expected_host_sha = krea_provenance.canonical_sha256(host_body)
    expected_host_file_sha = hashlib.sha256(
        krea_provenance.canonical_bytes(measured_host) + b"\n"
    ).hexdigest()
    live_host_binding = _object(timing_plan["live_host_receipt"], "timing host binding")
    if (
        measured_host.get("host_execution_identity_sha256") != expected_host_sha
        or controls.get("live_host_identity_file_sha256") != expected_host_file_sha
        or live_host_binding
        != {
            "file_sha256": expected_host_file_sha,
            "host_execution_identity_sha256": expected_host_sha,
        }
    ):
        raise ValueError("timing host identity differs from its sealed plan")

    measured_gpu = _object(controls.get("live_gpu_identity"), "timing GPU identity")
    gpu_sha = _sha(measured_gpu.get("gpu_identity_sha256"), "timing GPU identity")
    if timing_plan["live_gpu_receipt"].get("gpu_identity_sha256") != gpu_sha:
        raise ValueError("timing GPU identity differs from its sealed plan")

    checkpoint_mounts = [
        row for row in probe["mounts"] if row.get("purpose") == "checkpoints"
    ]
    if len(checkpoint_mounts) != 1:
        raise ValueError("timing probe checkpoint mount is not unique")
    probe_source = Path(checkpoint_mounts[0]["source_root"])
    probe_identity = live_stage2_host_identity(probe_source)
    return measured_host, gpu_sha, probe_identity["checkpoint_device"]


def _validate_live_base_assets(
    plan: Mapping[str, Any], *, production_identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay the one-time full attestation against the exact live RO mounts."""

    binding = plan["base_asset_attestation"]
    path = Path(binding["path"])
    record = krea_stage2_production_identity.load_asset_attestation(path)
    raw = krea_stage2_production_identity.canonical_bytes(record) + b"\n"
    model = _object(production_identity["base_model"], "production base model")
    if (
        hashlib.sha256(raw).hexdigest() != binding["file_sha256"]
        or record["attestation_sha256"] != binding["attestation_sha256"]
        or record["attestation_sha256"] != model["asset_attestation_sha256"]
        or record["training_identity_sha256"] != model["training_identity_sha256"]
        or record["training_identity_sha256"] != plan["base_model_identity_sha256"]
        or record["base_model"]["model_id"] != model["model_id"]
        or record["base_model"]["revision"] != model["revision"]
        or record["text_encoder"]["model_id"] != model["text_encoder_id"]
        or record["text_encoder"]["revision"] != model["text_encoder_revision"]
    ):
        raise ValueError("base asset attestation differs from production identity")
    mounts = {
        row["purpose"]: row
        for row in _mounts(plan["mounts"], require_live_sources=True)
    }
    return krea_stage2_production_identity.verify_live_asset_attestation(
        record,
        base_model_path=mounts["base_model"]["source"],
        text_encoder_path=mounts["text_encoder"]["source"],
    )


def validate_plan_with_authority(
    value: Any, *, authority_controls: Any
) -> dict[str, Any]:
    """Recompute the complete owner-authorized admission chain for a plan.

    A plan's embedded digests are not authority by themselves.  This validator
    requires every governing record, rehashes its canonical file form, asks the
    admission module to replay the full request -> ratification -> reveal ->
    materialization -> GPU-authorization chain, and only then compares that
    chain to the execution plan.  ``run_cell`` calls this immediately before it
    creates an output directory or invokes Docker.
    """

    plan = validate_plan(value)
    controls = _object(authority_controls, "Stage-2 authority bundle")
    keys = {
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
    _exact(controls, keys, "Stage-2 authority bundle")
    record_names = (
        "request",
        "ratification",
        "reveal",
        "materialization",
        "gpu_execution_authorization",
        "production_identity",
        "sealed_inventory",
    )
    records = {
        name: _object(controls[name], f"authority bundle {name}")
        for name in record_names
    }
    for name in record_names:
        file_key = f"{name}_file_sha256"
        expected_file_sha = krea_confirmation_admission.canonical_file_sha256(
            records[name]
        )
        if _sha(controls[file_key], file_key) != expected_file_sha:
            raise ValueError(f"{name} file SHA-256 does not bind its record")

    authorization = krea_confirmation_admission.validate_gpu_execution_authorization(
        records["gpu_execution_authorization"],
        request=records["request"],
        ratification=records["ratification"],
        reveal=records["reveal"],
        materialization=records["materialization"],
        request_file_sha256=controls["request_file_sha256"],
        ratification_file_sha256=controls["ratification_file_sha256"],
        reveal_file_sha256=controls["reveal_file_sha256"],
        materialization_file_sha256=controls["materialization_file_sha256"],
        production_identity=records["production_identity"],
        production_identity_file_sha256=controls["production_identity_file_sha256"],
    )
    inventory = krea_stage2_admission_chain.validate_inventory(
        records["sealed_inventory"]
    )
    if (
        controls["sealed_inventory_file_sha256"]
        != krea_confirmation_admission.canonical_file_sha256(inventory)
        or authorization["sealed_inventory_sha256"]
        != inventory["inventory_sha256"]
        or authorization["sealed_inventory_file_sha256"]
        != controls["sealed_inventory_file_sha256"]
        or records["materialization"]["files"]
        != krea_stage2_admission_chain.inventory_sealed_files(inventory)
    ):
        raise ValueError("sealed inventory differs from exact owner authority")
    expected = {
        "waiver_finalist_freeze": {
            "file_sha256": authorization["waiver_freeze_file_sha256"],
            "freeze_sha256": authorization["waiver_freeze_sha256"],
        },
        "confirmation_materialization": {
            "file_sha256": authorization["materialization_file_sha256"],
            "materialization_sha256": authorization["materialization_sha256"],
        },
        "owner_ratification": {
            "file_sha256": authorization["ratification_file_sha256"],
            "ratification_sha256": authorization["ratification_sha256"],
        },
        "gpu_execution_authorization": {
            "file_sha256": controls["gpu_execution_authorization_file_sha256"],
            "gpu_execution_authorization_sha256": authorization[
                "gpu_execution_authorization_sha256"
            ],
        },
        "production_identity": {
            "file_sha256": authorization["production_identity_file_sha256"],
            "production_identity_sha256": authorization["production_identity_sha256"],
        },
    }
    for field, binding in expected.items():
        if plan[field] != binding:
            raise ValueError(f"execution plan {field} differs from owner authority")
    if (
        plan["execution_surface_policy_sha256"] != authorization["policy_sha256"]
        or plan["delegated_role_contract_sha256"]
        != authorization["delegated_review_contract_sha256"]
        or plan["production_image_id"] != authorization["image_id"]
    ):
        raise ValueError(
            "execution plan policy, contract, or image differs from authority"
        )
    _validate_frozen_execution_family(
        plan,
        freeze=controls["waiver_finalist_freeze"],
        request=records["request"],
    )
    fixture = _validate_fixture_and_archive(
        plan,
        materialization=records["materialization"],
        request=records["request"],
    )
    if plan["calibration_profile"] == "K0":
        expected_k0_steps = recipe.size_scaled_steps(
            "krea2",
            len(fixture["training_rows"]),
            float(plan["hours"]),
            template_steps=2000,
        )
        if plan["planned_steps"] != expected_k0_steps:
            raise ValueError("K0 planned_steps differs from the frozen release recipe")
    _validate_throughput_depth(
        plan,
        fixture=fixture,
        production_identity=records["production_identity"],
    )
    _validate_execution_environment_profile(
        plan,
        fixture=fixture,
        production_identity=records["production_identity"],
    )
    _validate_live_base_assets(
        plan,
        production_identity=records["production_identity"],
    )
    return plan


def validate_approval(
    value: Any, *, plan: dict[str, Any] | None = None
) -> dict[str, Any]:
    approval = _object(value, "Stage-2 execution approval")
    keys = {
        "schema",
        "kind",
        "execution_plan_sha256",
        "owner_ratification_sha256",
        "gpu_execution_authorization_sha256",
        "production_identity_sha256",
        "reviewer_actor",
        "approved_at_utc",
        "decision",
        "gpu_execution_authorized",
        "release_authorized",
        "production_mutation_authorized",
        "approval_sha256",
    }
    _exact(approval, keys, "Stage-2 execution approval")
    body = {key: item for key, item in approval.items() if key != "approval_sha256"}
    if approval["schema"] != SCHEMA or approval["kind"] != APPROVAL_KIND:
        raise ValueError("Stage-2 approval kind/schema differs")
    if approval["approval_sha256"] != krea_provenance.canonical_sha256(body):
        raise ValueError("Stage-2 approval digest differs")
    _sha(approval["execution_plan_sha256"], "execution plan sha")
    _sha(approval["owner_ratification_sha256"], "owner ratification sha")
    _sha(
        approval["gpu_execution_authorization_sha256"],
        "GPU execution authorization sha",
    )
    _sha(approval["production_identity_sha256"], "production identity sha")
    actor = _actor(approval["reviewer_actor"], "reviewer_actor")
    if actor["role"] != "execution_plan_reviewer":
        raise ValueError("Stage-2 approval uses the wrong delegated role")
    _utc(approval["approved_at_utc"], "approved_at_utc")
    if _utc_value(approval["approved_at_utc"]) > datetime.now(timezone.utc) + timedelta(
        seconds=60
    ):
        raise ValueError("Stage-2 approval cannot be future-dated")
    if (
        approval["decision"] != "approved"
        or approval["gpu_execution_authorized"] is not True
        or approval["release_authorized"] is not False
        or approval["production_mutation_authorized"] is not False
    ):
        raise ValueError("Stage-2 approval authority flags differ")
    if plan is not None:
        resolved = validate_plan(plan)
        if approval["execution_plan_sha256"] != resolved["plan_sha256"]:
            raise ValueError("approval binds a different execution plan")
        if (
            approval["owner_ratification_sha256"]
            != resolved["owner_ratification"]["ratification_sha256"]
        ):
            raise ValueError("approval owner ratification differs from plan")
        if (
            approval["gpu_execution_authorization_sha256"]
            != resolved["gpu_execution_authorization"][
                "gpu_execution_authorization_sha256"
            ]
        ):
            raise ValueError("approval GPU execution authorization differs from plan")
        if (
            approval["production_identity_sha256"]
            != resolved["production_identity"]["production_identity_sha256"]
        ):
            raise ValueError("approval production identity differs from plan")
        if _utc_value(approval["approved_at_utc"]) <= _utc_value(
            resolved["created_at_utc"]
        ):
            raise ValueError("approval must postdate the execution plan")
    return dict(approval)


def build_approval(
    plan: dict[str, Any], *, reviewer_actor: dict[str, Any], approved_at_utc: str
) -> dict[str, Any]:
    resolved = validate_plan(plan)
    body = {
        "schema": SCHEMA,
        "kind": APPROVAL_KIND,
        "execution_plan_sha256": resolved["plan_sha256"],
        "owner_ratification_sha256": resolved["owner_ratification"][
            "ratification_sha256"
        ],
        "gpu_execution_authorization_sha256": resolved["gpu_execution_authorization"][
            "gpu_execution_authorization_sha256"
        ],
        "production_identity_sha256": resolved["production_identity"][
            "production_identity_sha256"
        ],
        "reviewer_actor": _actor(reviewer_actor, "reviewer_actor"),
        "approved_at_utc": approved_at_utc,
        "decision": "approved",
        "gpu_execution_authorized": True,
        "release_authorized": False,
        "production_mutation_authorized": False,
    }
    return validate_approval(
        {**body, "approval_sha256": krea_provenance.canonical_sha256(body)},
        plan=resolved,
    )


def _file_manifest(
    root: Path, *, excluded: set[Path], prefix: str = ""
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path in excluded:
            continue
        if path.is_symlink():
            raise ValueError(f"run output contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = PurePosixPath(prefix, path.relative_to(root).as_posix()).as_posix()
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": krea_provenance.file_sha256(path),
            }
        )
    return rows


def _completion_mechanics(
    checkpoint_root: Path,
    *,
    returncode: int,
    hours: str,
    planned_steps_completed: bool,
) -> dict[str, bool]:
    """Derive boundary mechanics from the production bundle itself."""

    last = checkpoint_root / "last.safetensors"
    recorder = checkpoint_root / "forge_run.json"
    upload_ready = False
    clean_telemetry = False
    decision_before_reserve = False
    fallback_used = True
    try:
        from forge.tasks.integrity import valid_safetensors

        upload_ready = bool(
            returncode == 0
            and last.is_file()
            and not last.is_symlink()
            and valid_safetensors(str(last))
        )
        raw = recorder.read_bytes()
        public = json.loads(raw)
        if (
            not recorder.is_symlink()
            and public.get("schema") == 2
            and public.get("kind") == "forge-public-run-recorder"
            and _SHA256.fullmatch(str(public.get("private_record_sha256")))
        ):
            events = public.get("events")
            if isinstance(events, list) and events:
                names = [
                    row.get("name")
                    for row in events
                    if isinstance(row, dict) and isinstance(row.get("name"), str)
                ]
                forbidden = {
                    "checkpoint_scope_failed",
                    "dispatch_failed",
                    "fallback_failed",
                    "handler_failed",
                    "public_bundle_failed",
                    "spec_build_failed",
                }
                fallback_used = any("fallback" in name for name in names)
                clean_telemetry = (
                    "run_complete" in names
                    and "public_bundle_ready" in names
                    and not fallback_used
                    and not (set(names) & forbidden)
                    and all("failure_class" not in row for row in events)
                )
                finalized = [
                    float(row["t"])
                    for row in events
                    if isinstance(row, dict)
                    and row.get("name") == "checkpoint_finalized"
                    and not isinstance(row.get("t"), bool)
                    and isinstance(row.get("t"), (int, float))
                    and math.isfinite(float(row["t"]))
                ]
                soft_deadline_s = float(hours) * 3600.0 - 180.0
                decision_before_reserve = (
                    len(finalized) == 1
                    and soft_deadline_s > 0
                    and 0 <= finalized[0] <= soft_deadline_s
                )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return {
        "natural_completion": returncode == 0 and planned_steps_completed,
        "planned_steps_completed": planned_steps_completed,
        "upload_ready": upload_ready,
        "clean_telemetry": clean_telemetry,
        "decision_completed_before_export_reserve": decision_before_reserve,
        "fallback_used": fallback_used,
    }


def _validate_terminal_receipt(
    plan: Mapping[str, Any], *, evidence_root: Path, control: Mapping[str, Any]
) -> dict[str, Any]:
    receipt, file_sha = _canonical_json_file(
        evidence_root / "training-terminal.json",
        "Stage-2 training-terminal receipt",
    )
    keys = {
        "schema",
        "kind",
        "execution_plan_sha256",
        "profile_id",
        "profile_sha256",
        "training_seed",
        "planned_steps",
        "last_step",
        "trainer_returncode",
        "stopped_by_deadline",
        "planned_steps_completed",
        "natural_completion",
        "config_control_file_sha256",
        "checkpoint_selection",
        "release_authorized",
        "receipt_sha256",
    }
    _exact(receipt, keys, "Stage-2 training-terminal receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    frozen = krea_calibration_profiles.profile_for_id(plan["calibration_profile"])
    if (
        receipt["schema"] != 1
        or receipt["kind"] != "forge-krea-stage2-training-terminal-receipt"
        or receipt["receipt_sha256"] != krea_provenance.canonical_sha256(body)
        or receipt["execution_plan_sha256"] != plan["plan_sha256"]
        or receipt["profile_id"] != frozen.profile_id
        or receipt["profile_sha256"] != frozen.profile_sha256
        or receipt["training_seed"] != plan["seed"]
        or receipt["planned_steps"] != plan["planned_steps"]
        or receipt["last_step"] != plan["planned_steps"]
        or receipt["trainer_returncode"] != 0
        or receipt["stopped_by_deadline"] is not False
        or receipt["planned_steps_completed"] is not True
        or receipt["natural_completion"] is not True
        or receipt["config_control_file_sha256"] != control["file_sha256"]
        or receipt["checkpoint_selection"] != _runtime_checkpoint_selection(plan)
        or receipt["release_authorized"] is not False
    ):
        raise ValueError("Stage-2 terminal receipt does not prove full completion")
    return {
        "file_sha256": file_sha,
        "receipt_sha256": receipt["receipt_sha256"],
    }


def _validate_control_receipt(
    plan: Mapping[str, Any], *, evidence_root: Path
) -> dict[str, Any]:
    receipt_path = evidence_root / "config-control.json"
    config_path = evidence_root / "effective-config.yaml"
    receipt, file_sha = _canonical_json_file(
        receipt_path, "Stage-2 config-control receipt"
    )
    keys = {
        "schema",
        "kind",
        "execution_plan_sha256",
        "profile_id",
        "profile_sha256",
        "training_seed",
        "throughput_profile_sha256",
        "config_sha256",
        "effective_config_file",
        "effective_recipe",
        "effective_recipe_sha256",
        "checkpoint_selection",
        "release_authorized",
        "receipt_sha256",
    }
    _exact(receipt, keys, "Stage-2 config-control receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    frozen = krea_calibration_profiles.profile_for_id(plan["calibration_profile"])
    expected_throughput = (
        None
        if plan["throughput_profile"] is None
        else plan["throughput_profile"]["profile_sha256"]
    )
    if (
        receipt["schema"] != 1
        or receipt["kind"] != "forge-krea-stage2-config-control-receipt"
        or receipt["receipt_sha256"] != krea_provenance.canonical_sha256(body)
        or receipt["execution_plan_sha256"] != plan["plan_sha256"]
        or receipt["profile_id"] != frozen.profile_id
        or receipt["profile_sha256"] != frozen.profile_sha256
        or receipt["training_seed"] != plan["seed"]
        or receipt["throughput_profile_sha256"] != expected_throughput
        or receipt["checkpoint_selection"] != _runtime_checkpoint_selection(plan)
        or receipt["release_authorized"] is not False
    ):
        raise ValueError("Stage-2 config-control receipt differs from its plan")
    config_file = _object(
        receipt["effective_config_file"], "Stage-2 effective config file"
    )
    _exact(
        config_file,
        {"path", "bytes", "sha256"},
        "Stage-2 effective config file",
    )
    if (
        config_file["path"] != "effective-config.yaml"
        or isinstance(config_file["bytes"], bool)
        or not isinstance(config_file["bytes"], int)
        or config_file["bytes"] <= 0
        or _sha(config_file["sha256"], "effective config SHA-256")
        != config_file["sha256"]
    ):
        raise ValueError("Stage-2 effective config file identity is invalid")
    try:
        if config_path.is_symlink() or not config_path.is_file():
            raise OSError
        config_bytes = config_path.read_bytes()
    except OSError as exc:
        raise ValueError("Stage-2 effective config file is unavailable") from exc
    if (
        len(config_bytes) != config_file["bytes"]
        or hashlib.sha256(config_bytes).hexdigest() != config_file["sha256"]
        or receipt["config_sha256"] != config_file["sha256"]
    ):
        raise ValueError("Stage-2 effective config bytes drifted")
    effective = _object(receipt["effective_recipe"], "Stage-2 effective recipe")
    if receipt["effective_recipe_sha256"] != krea_provenance.canonical_sha256(
        effective
    ):
        raise ValueError("Stage-2 effective recipe digest drifted")
    expected_save_every = (
        recipe.kill_safe_save_every(plan["planned_steps"], 200)
        if frozen.profile_id == "K0"
        else (plan["planned_steps"] + 7) // 8
    )
    expected_effective = {
        "config_name": plan["expected_repo_name"],
        "training_folder": f"/app/checkpoints/{plan['task_id']}",
        "trigger_word": plan["trigger_word"],
        "model_arch": "krea2",
        "model_name_or_path": "/cache/models/krea--Krea-2-Raw",
        "model_kwargs": {
            "text_encoder_path": "/cache/hf_cache/Qwen--Qwen3-VL-4B-Instruct",
            "vae_path": "/cache/models/krea--Krea-2-Raw",
        },
        "dataset_folder_path": "/dataset/images",
        "network_rank": frozen.rank,
        "network_alpha": frozen.alpha,
        "optimizer": frozen.optimizer,
        "optimizer_params": dict(frozen.optimizer_parameters),
        "loss": frozen.loss,
        "guidance_enabled": True,
        "guidance_scale": frozen.guidance,
        "learning_rate": frozen.learning_rate,
        "dropout": frozen.dropout,
        "ema": {"use_ema": frozen.ema, "ema_decay": 0.99},
        "steps": plan["planned_steps"],
        "save_every": expected_save_every,
        "push_to_hub": False,
        "batch_size": 1,
        "gradient_accumulation": 1,
        "resolution": [512, 768, 1024],
        "train_dtype": "bf16",
        "save_dtype": "bf16",
        "cache_latents_to_disk": False,
        "cache_text_embeddings": False,
        "compile": False,
        "dataloader_workers": 0,
    }
    if effective != expected_effective:
        raise ValueError("Stage-2 effective recipe differs from the frozen plan")
    raw_model_path = effective["model_name_or_path"]
    model_path = PurePosixPath(str(raw_model_path))
    model_root = PurePosixPath("/cache/models")
    if (
        not isinstance(raw_model_path, str)
        or not model_path.is_absolute()
        or str(model_path) != raw_model_path
        or ".." in model_path.parts
        or not (model_path == model_root or model_path.is_relative_to(model_root))
    ):
        raise ValueError("Stage-2 effective model escaped the read-only model cache")
    return {
        "file_sha256": file_sha,
        "receipt_sha256": receipt["receipt_sha256"],
        "config_sha256": receipt["config_sha256"],
    }


def _validate_checkpoint_selection_receipt(
    plan: Mapping[str, Any], *, evidence_root: Path
) -> dict[str, str]:
    """Replay the preserved private selection and its promoted checkpoint."""

    receipt, file_sha = _canonical_json_file_without_newline(
        evidence_root / "forge_checkpoint_selection.json",
        "Stage-2 checkpoint-selection receipt",
    )
    keys = {
        "schema",
        "status",
        "context",
        "source",
        "selected_file",
        "output_file",
        "selected_step",
        "sha256",
        "reason",
        "score",
        "metric",
        "direction",
        "training_loss_is_proxy_not_validator_metric",
        "metric_is_proxy_not_validator_metric",
        "reference_file",
        "reference_score",
        "score_advantage",
        "required_advantage",
        "margin_policy",
        "calibration_id",
        "current_candidates_discovered",
        "current_candidates_valid",
        "created_unix",
        "checkpoint_target",
        "planned_steps",
    }
    _exact(receipt, keys, "Stage-2 checkpoint-selection receipt")
    runtime_selection = _runtime_checkpoint_selection(plan)
    selected_step = runtime_selection["selected_step"]
    planned_steps = runtime_selection["planned_steps"]
    target = runtime_selection["target_fraction"]
    expected_target = {
        "fraction_numerator": target["numerator"],
        "fraction_denominator": target["denominator"],
        "selection_rule": _CHECKPOINT_MAPPING_RULE,
    }
    expected_selected_file = (
        f"{plan['expected_repo_name']}.safetensors"
        if selected_step == planned_steps
        else f"{plan['expected_repo_name']}_{selected_step:09d}.safetensors"
    )
    reason = receipt["reason"]
    discovered = receipt["current_candidates_discovered"]
    valid = receipt["current_candidates_valid"]
    created = receipt["created_unix"]
    null_fields = {
        "score",
        "metric",
        "direction",
        "reference_file",
        "reference_score",
        "score_advantage",
        "required_advantage",
        "margin_policy",
        "calibration_id",
    }
    if (
        receipt["schema"] != 1
        or receipt["status"] != "selected_current_run"
        or receipt["context"] != "training"
        or receipt["source"] != "frozen_checkpoint_fraction"
        or receipt["selected_file"] != expected_selected_file
        or receipt["output_file"] != "last.safetensors"
        or receipt["selected_step"] != selected_step
        or receipt["checkpoint_target"] != expected_target
        or receipt["planned_steps"] != planned_steps
        or _sha(receipt["sha256"], "selected checkpoint SHA-256") != receipt["sha256"]
        or not isinstance(reason, str)
        or not reason.strip()
        or receipt["training_loss_is_proxy_not_validator_metric"] is not False
        or receipt["metric_is_proxy_not_validator_metric"] is not False
        or any(receipt[field] is not None for field in null_fields)
        or isinstance(discovered, bool)
        or not isinstance(discovered, int)
        or isinstance(valid, bool)
        or not isinstance(valid, int)
        or valid <= 0
        or discovered < valid
        or isinstance(created, bool)
        or not isinstance(created, int)
        or created <= 0
    ):
        raise ValueError(
            "Stage-2 checkpoint-selection receipt differs from its frozen target"
        )

    checkpoint_mount = next(
        Path(mount["source"])
        for mount in plan["mounts"]
        if mount["purpose"] == "checkpoints"
    )
    checkpoint_root = checkpoint_mount / plan["task_id"] / plan["expected_repo_name"]
    selected_identity = _stable_regular_file_identity(
        checkpoint_root / expected_selected_file,
        "Stage-2 selected checkpoint",
    )
    promoted_identity = _stable_regular_file_identity(
        checkpoint_root / "last.safetensors",
        "Stage-2 promoted last.safetensors",
    )
    if (
        selected_identity != promoted_identity
        or selected_identity["sha256"] != receipt["sha256"]
    ):
        raise ValueError(
            "Stage-2 selected checkpoint bytes differ from last.safetensors"
        )
    try:
        from forge.tasks.integrity import valid_safetensors

        selected_valid = valid_safetensors(
            str(checkpoint_root / expected_selected_file)
        )
        promoted_valid = valid_safetensors(str(checkpoint_root / "last.safetensors"))
    except (OSError, ValueError, TypeError):
        selected_valid = promoted_valid = False
    if not selected_valid or not promoted_valid:
        raise ValueError("Stage-2 checkpoint selection is not valid safetensors")
    return {
        "file_sha256": file_sha,
        "receipt_sha256": krea_provenance.canonical_sha256(receipt),
    }


def validate_private_run_receipts(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Replay the three private receipts for one validated Stage-2 plan.

    Portable run-evidence records bind these receipt digests, but a release
    decision must also prove that the original private files still exist and
    that their effective-config bytes still match the frozen execution plan.
    Keeping that replay here avoids multiple downstream gates reimplementing
    the security-sensitive evidence-root derivation.
    """

    resolved = validate_plan(plan)
    evidence_mount = next(
        Path(mount["source"])
        for mount in resolved["mounts"]
        if mount["purpose"] == "run_evidence"
    )
    evidence_root = evidence_mount / resolved["plan_sha256"]
    control = _validate_control_receipt(resolved, evidence_root=evidence_root)
    terminal = _validate_terminal_receipt(
        resolved,
        evidence_root=evidence_root,
        control=control,
    )
    checkpoint_selection = _validate_checkpoint_selection_receipt(
        resolved,
        evidence_root=evidence_root,
    )
    return {
        "evidence_root": str(evidence_root),
        "effective_config_path": str(evidence_root / "effective-config.yaml"),
        "config_control": control,
        "training_terminal": terminal,
        "checkpoint_selection": checkpoint_selection,
    }


def validate_completion(
    value: Any, *, plan: dict[str, Any], approval: dict[str, Any]
) -> dict[str, Any]:
    completion = _object(value, "Stage-2 completion")
    keys = {
        "schema",
        "kind",
        "execution_plan_sha256",
        "execution_approval_sha256",
        "production_image_id",
        "phase",
        "cell_id",
        "started_at_utc",
        "ended_at_utc",
        "returncode",
        "natural_completion",
        "fallback_used",
        "mechanics",
        "artifact_manifest",
        "config_control_receipt",
        "training_terminal_receipt",
        "checkpoint_selection_receipt",
        "postrun_identity_sha256",
        "network_mode",
        "runtime",
        "gpu_device",
        "strict_discovery_replayed",
        "release_authorized",
        "production_mutation_authorized",
        "completion_sha256",
    }
    _exact(completion, keys, "Stage-2 completion")
    body = {key: item for key, item in completion.items() if key != "completion_sha256"}
    resolved_plan = validate_plan(plan)
    resolved_approval = validate_approval(approval, plan=resolved_plan)
    if completion["schema"] != SCHEMA or completion["kind"] != COMPLETION_KIND:
        raise ValueError("Stage-2 completion kind/schema differs")
    if completion["completion_sha256"] != krea_provenance.canonical_sha256(body):
        raise ValueError("Stage-2 completion digest differs")
    if (
        completion["execution_plan_sha256"] != resolved_plan["plan_sha256"]
        or completion["execution_approval_sha256"]
        != resolved_approval["approval_sha256"]
        or completion["production_image_id"] != resolved_plan["production_image_id"]
        or completion["phase"] != resolved_plan["phase"]
        or completion["cell_id"] != resolved_plan["cell_id"]
        or completion["postrun_identity_sha256"]
        != resolved_plan["production_identity"]["production_identity_sha256"]
    ):
        raise ValueError("Stage-2 completion authority binding differs")
    started = _utc_value(completion["started_at_utc"])
    ended = _utc_value(completion["ended_at_utc"])
    if (
        ended <= started
        or started < _utc_value(resolved_plan["created_at_utc"])
        or started < _utc_value(resolved_approval["approved_at_utc"])
        or ended > datetime.now(timezone.utc) + timedelta(seconds=60)
    ):
        raise ValueError("Stage-2 completion chronology is invalid")
    if (
        completion["returncode"] != 0
        or completion["natural_completion"] is not True
        or completion["fallback_used"] is not False
    ):
        raise ValueError("Stage-2 cell did not complete naturally")
    expected_mechanics = {
        "natural_completion": True,
        "planned_steps_completed": True,
        "upload_ready": True,
        "clean_telemetry": True,
        "decision_completed_before_export_reserve": True,
        "fallback_used": False,
    }
    if completion["mechanics"] != expected_mechanics:
        raise ValueError("Stage-2 cell lacks clean upload-ready boundary mechanics")
    manifest = completion["artifact_manifest"]
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("Stage-2 completion artifact manifest is empty")
    paths: set[str] = set()
    manifest_by_path: dict[str, dict[str, Any]] = {}
    checkpoint_artifacts = 0
    for row in manifest:
        row = _object(row, "artifact manifest row")
        _exact(row, {"path", "bytes", "sha256"}, "artifact manifest row")
        path = PurePosixPath(str(row["path"]))
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("artifact manifest path is unsafe")
        if str(path) in paths:
            raise ValueError("artifact manifest path is duplicated")
        paths.add(str(path))
        manifest_by_path[str(path)] = row
        if (
            isinstance(row["bytes"], bool)
            or not isinstance(row["bytes"], int)
            or row["bytes"] < 0
        ):
            raise ValueError("artifact manifest byte count is invalid")
        _sha(row["sha256"], "artifact sha256")
        if str(path).startswith("checkpoints/") and str(path).endswith(".safetensors"):
            if row["bytes"] <= 0:
                raise ValueError("checkpoint artifact is empty")
            checkpoint_artifacts += 1
    if checkpoint_artifacts == 0:
        raise ValueError("Stage-2 completion has no safetensors checkpoint")
    runtime_selection = _runtime_checkpoint_selection(resolved_plan)
    selected_file = (
        f"{resolved_plan['expected_repo_name']}.safetensors"
        if runtime_selection["selected_step"] == resolved_plan["planned_steps"]
        else (
            f"{resolved_plan['expected_repo_name']}_"
            f"{runtime_selection['selected_step']:09d}.safetensors"
        )
    )
    selected_artifact = manifest_by_path.get(f"checkpoints/{selected_file}")
    promoted_artifact = manifest_by_path.get("checkpoints/last.safetensors")
    if (
        selected_artifact is None
        or promoted_artifact is None
        or selected_artifact["bytes"] <= 0
        or selected_artifact["bytes"] != promoted_artifact["bytes"]
        or selected_artifact["sha256"] != promoted_artifact["sha256"]
    ):
        raise ValueError(
            "Stage-2 completion does not prove selected checkpoint promotion"
        )
    control = _object(
        completion["config_control_receipt"], "config-control receipt binding"
    )
    _exact(
        control,
        {"file_sha256", "receipt_sha256", "config_sha256"},
        "config-control receipt binding",
    )
    for key in control:
        _sha(control[key], f"config-control receipt {key}")
    terminal = _object(
        completion["training_terminal_receipt"],
        "training-terminal receipt binding",
    )
    _exact(
        terminal,
        {"file_sha256", "receipt_sha256"},
        "training-terminal receipt binding",
    )
    for key in terminal:
        _sha(terminal[key], f"training-terminal receipt {key}")
    selection = _object(
        completion["checkpoint_selection_receipt"],
        "checkpoint-selection receipt binding",
    )
    _exact(
        selection,
        {"file_sha256", "receipt_sha256"},
        "checkpoint-selection receipt binding",
    )
    for key in selection:
        _sha(selection[key], f"checkpoint-selection receipt {key}")
    expected_evidence = {
        "evidence/config-control.json": control["file_sha256"],
        "evidence/effective-config.yaml": control["config_sha256"],
        "evidence/training-terminal.json": terminal["file_sha256"],
        "evidence/forge_checkpoint_selection.json": selection["file_sha256"],
    }
    for path, expected_sha256 in expected_evidence.items():
        row = manifest_by_path.get(path)
        if row is None or row["bytes"] <= 0 or row["sha256"] != expected_sha256:
            raise ValueError(f"Stage-2 completion does not bind {path}")
    _sha(completion["postrun_identity_sha256"], "postrun identity")
    if completion["network_mode"] != "none" or completion["runtime"] != "nvidia":
        raise ValueError("Stage-2 completion surface differs")
    if (
        isinstance(completion["gpu_device"], bool)
        or not isinstance(completion["gpu_device"], int)
        or completion["gpu_device"] < 0
    ):
        raise ValueError("gpu_device must be a nonnegative integer")
    if (
        completion["strict_discovery_replayed"] is not False
        or completion["release_authorized"] is not False
        or completion["production_mutation_authorized"] is not False
    ):
        raise ValueError("Stage-2 completion overclaims authority")
    return dict(completion)


def _publish_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = krea_provenance.canonical_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o444)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def run_cell(
    *,
    plan: dict[str, Any],
    approval: dict[str, Any],
    authority_controls: dict[str, Any],
    output_dir: Path,
    completion_path: Path,
    gpu_device: int,
    docker: str = "docker",
) -> dict[str, Any]:
    """Execute one exact-image Stage-2 cell without overriding its entrypoint."""

    resolved = validate_plan_with_authority(plan, authority_controls=authority_controls)
    approved = validate_approval(approval, plan=resolved)
    if os.path.lexists(output_dir) or os.path.lexists(completion_path):
        raise FileExistsError(output_dir)
    live_mounts = _mounts(resolved["mounts"], require_live_sources=True)
    checkpoint_mount = next(
        Path(row["source"]) for row in live_mounts if row["purpose"] == "checkpoints"
    )
    evidence_mount = next(
        Path(row["source"]) for row in live_mounts if row["purpose"] == "run_evidence"
    )
    _validate_live_throughput_environment(
        resolved, checkpoint_mount=checkpoint_mount, gpu_device=gpu_device
    )
    checkpoint_root = (
        checkpoint_mount / resolved["task_id"] / resolved["expected_repo_name"]
    )
    evidence_root = evidence_mount / resolved["plan_sha256"]
    if os.path.lexists(checkpoint_root) or os.path.lexists(evidence_root):
        raise FileExistsError(
            "Stage-2 checkpoint/evidence roots must both be create-only"
        )
    output_dir.mkdir(parents=True, mode=0o755)
    checkpoint_root.mkdir(parents=True, mode=0o755, exist_ok=False)
    evidence_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    checkpoint_root_identity = checkpoint_root.stat()
    evidence_root_identity = evidence_root.stat()
    stdout_path = output_dir / "container.stdout"
    stderr_path = output_dir / "container.stderr"
    container_name = f"forge-stage2-{resolved['plan_sha256'][:24]}"
    command = [
        docker,
        "run",
        "--rm",
        "--name",
        container_name,
        "--runtime",
        "nvidia",
        "--gpus",
        f"device={gpu_device}",
        "--shm-size",
        _VALIDATOR_IMAGE_TRAINER_SHM_SIZE,
        "--memory",
        _VALIDATOR_IMAGE_TRAINER_MEMORY,
        "--cpus",
        _VALIDATOR_IMAGE_TRAINER_CPUS,
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--network",
        "none",
    ]
    for mount in live_mounts:
        # Docker's long --mount grammar spells a read-only bind ``readonly``;
        # writable is the default and has no trailing token.  ``,ro``/``,rw``
        # are only valid in the short ``-v`` syntax.
        read_only = ",readonly" if mount["read_only"] else ""
        command.extend(
            [
                "--mount",
                f"type=bind,src={mount['source']},dst={mount['destination']}"
                + read_only,
            ]
        )
    command.extend(
        [
            "--env",
            "FORGE_KREA_CALIBRATION_PROFILE=" + resolved["calibration_profile"],
            "--env",
            "FORGE_KREA_STAGE2_TRAINING_SEED=" + str(resolved["seed"]),
            "--env",
            "FORGE_KREA_STAGE2_EXECUTION_PLAN_SHA256=" + resolved["plan_sha256"],
            "--env",
            "FORGE_KREA_STAGE2_CONTROL_RECEIPT_PATH=/run-evidence/"
            + resolved["plan_sha256"]
            + "/config-control.json",
            "--env",
            _TARGET_FRACTION_NUMERATOR_ENV
            + "="
            + str(resolved["checkpoint_selection"]["target_fraction"]["numerator"]),
            "--env",
            _TARGET_FRACTION_DENOMINATOR_ENV
            + "="
            + str(resolved["checkpoint_selection"]["target_fraction"]["denominator"]),
        ]
    )
    if resolved["throughput_profile"] is not None:
        command.extend(
            [
                "--env",
                "FORGE_KREA_CALIBRATION_STEPS=" + str(resolved["planned_steps"]),
                "--env",
                "FORGE_KREA_CALIBRATION_THROUGHPUT_PROFILE_SHA256="
                + resolved["throughput_profile"]["profile_sha256"],
            ]
        )
    command.append(resolved["production_image_id"])
    command.extend(resolved["entrypoint_argv"])
    outer_timeout_seconds = int(Decimal(resolved["hours"]) * Decimal(3600)) + 600
    started = datetime.now(timezone.utc).replace(microsecond=0)
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            process = subprocess.run(
                command,
                stdout=stdout,
                stderr=stderr,
                check=False,
                timeout=outer_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            stderr.write(b"\nforge-stage2: outer container deadline exceeded\n")
            stderr.flush()
            subprocess.run(
                [docker, "rm", "-f", container_name],
                stdout=stderr,
                stderr=stderr,
                check=False,
                timeout=60,
            )
            process = subprocess.CompletedProcess(command, 124)
    ended = datetime.now(timezone.utc).replace(microsecond=0)
    if ended <= started:
        ended = started + timedelta(seconds=1)
    try:
        checkpoint_root_after = checkpoint_root.stat()
    except OSError as exc:
        raise RuntimeError("Stage-2 checkpoint root disappeared after Docker") from exc
    if (
        checkpoint_root.is_symlink()
        or not stat.S_ISDIR(checkpoint_root_after.st_mode)
        or (checkpoint_root_after.st_dev, checkpoint_root_after.st_ino)
        != (checkpoint_root_identity.st_dev, checkpoint_root_identity.st_ino)
    ):
        raise RuntimeError("Stage-2 checkpoint root identity changed during Docker")
    try:
        evidence_root_after = evidence_root.stat()
    except OSError as exc:
        raise RuntimeError("Stage-2 evidence root disappeared after Docker") from exc
    if (
        evidence_root.is_symlink()
        or not stat.S_ISDIR(evidence_root_after.st_mode)
        or (evidence_root_after.st_dev, evidence_root_after.st_ino)
        != (evidence_root_identity.st_dev, evidence_root_identity.st_ino)
    ):
        raise RuntimeError("Stage-2 evidence root identity changed during Docker")
    _validate_live_throughput_environment(
        resolved, checkpoint_mount=checkpoint_mount, gpu_device=gpu_device
    )
    _validate_live_base_assets(
        resolved,
        production_identity=_object(
            authority_controls["production_identity"], "production identity"
        ),
    )
    run_manifest = _file_manifest(output_dir, excluded=set(), prefix="run")
    checkpoint_manifest = (
        _file_manifest(checkpoint_root, excluded=set(), prefix="checkpoints")
        if checkpoint_root.is_dir() and not checkpoint_root.is_symlink()
        else []
    )
    evidence_manifest = _file_manifest(evidence_root, excluded=set(), prefix="evidence")
    manifest = [*run_manifest, *checkpoint_manifest, *evidence_manifest]
    try:
        control_receipt = _validate_control_receipt(
            resolved, evidence_root=evidence_root
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        control_receipt = {
            "file_sha256": "0" * 64,
            "receipt_sha256": "0" * 64,
            "config_sha256": "0" * 64,
        }
    try:
        terminal_receipt = _validate_terminal_receipt(
            resolved,
            evidence_root=evidence_root,
            control=control_receipt,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        terminal_receipt = {
            "file_sha256": "0" * 64,
            "receipt_sha256": "0" * 64,
        }
    try:
        checkpoint_selection_receipt = _validate_checkpoint_selection_receipt(
            resolved,
            evidence_root=evidence_root,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        checkpoint_selection_receipt = {
            "file_sha256": "0" * 64,
            "receipt_sha256": "0" * 64,
        }
    private_receipts_complete = all(
        receipt["receipt_sha256"] != "0" * 64
        for receipt in (
            control_receipt,
            terminal_receipt,
            checkpoint_selection_receipt,
        )
    )
    mechanics = _completion_mechanics(
        checkpoint_root,
        returncode=process.returncode,
        hours=resolved["hours"],
        planned_steps_completed=private_receipts_complete,
    )
    body = {
        "schema": SCHEMA,
        "kind": COMPLETION_KIND,
        "execution_plan_sha256": resolved["plan_sha256"],
        "execution_approval_sha256": approved["approval_sha256"],
        "production_image_id": resolved["production_image_id"],
        "phase": resolved["phase"],
        "cell_id": resolved["cell_id"],
        "started_at_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ended_at_utc": ended.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "returncode": process.returncode,
        "natural_completion": mechanics["natural_completion"]
        and bool(checkpoint_manifest),
        "fallback_used": mechanics["fallback_used"],
        "mechanics": mechanics,
        "artifact_manifest": manifest,
        "config_control_receipt": control_receipt,
        "training_terminal_receipt": terminal_receipt,
        "checkpoint_selection_receipt": checkpoint_selection_receipt,
        "postrun_identity_sha256": resolved["production_identity"][
            "production_identity_sha256"
        ],
        "network_mode": "none",
        "runtime": "nvidia",
        "gpu_device": gpu_device,
        "strict_discovery_replayed": False,
        "release_authorized": False,
        "production_mutation_authorized": False,
    }
    completion = {**body, "completion_sha256": krea_provenance.canonical_sha256(body)}
    if (
        process.returncode != 0
        or not checkpoint_manifest
        or control_receipt["receipt_sha256"] == "0" * 64
        or terminal_receipt["receipt_sha256"] == "0" * 64
        or checkpoint_selection_receipt["receipt_sha256"] == "0" * 64
        or not all(
            (
                mechanics["natural_completion"],
                mechanics["upload_ready"],
                mechanics["clean_telemetry"],
                mechanics["decision_completed_before_export_reserve"],
                not mechanics["fallback_used"],
            )
        )
    ):
        # Preserve failure evidence, but never publish a successful completion.
        failure = completion_path.with_suffix(completion_path.suffix + ".failed")
        _publish_new(failure, completion)
        if process.returncode != 0:
            raise RuntimeError(f"Stage-2 container returned {process.returncode}")
        raise RuntimeError("Stage-2 container failed clean upload-ready mechanics")
    validate_completion(completion, plan=resolved, approval=approved)
    _publish_new(completion_path, completion)
    return completion


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return _object(value, str(path))


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-plan")
    validate.add_argument("--plan", required=True, type=Path)
    authority = sub.add_parser("validate-authority")
    authority.add_argument("--plan", required=True, type=Path)
    authority.add_argument("--authority-bundle", required=True, type=Path)
    approval = sub.add_parser("validate-approval")
    approval.add_argument("--plan", required=True, type=Path)
    approval.add_argument("--approval", required=True, type=Path)
    run = sub.add_parser("run-cell")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--approval", required=True, type=Path)
    run.add_argument("--authority-bundle", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--completion", required=True, type=Path)
    run.add_argument("--gpu", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    try:
        plan = validate_plan(_load(args.plan))
        if args.command == "validate-plan":
            result: dict[str, Any] = plan
        elif args.command == "validate-authority":
            result = validate_plan_with_authority(
                plan, authority_controls=_load(args.authority_bundle)
            )
        else:
            approval = validate_approval(_load(args.approval), plan=plan)
            if args.command == "validate-approval":
                result = approval
            else:
                result = run_cell(
                    plan=plan,
                    approval=approval,
                    authority_controls=_load(args.authority_bundle),
                    output_dir=args.output_dir,
                    completion_path=args.completion,
                    gpu_device=args.gpu,
                )
    except (
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
