#!/usr/bin/env python3
"""Create-only, receipt-derived Krea Stage-2 throughput evidence.

The persisted bundle contains only sanitized timing evidence.  Full command
arguments, event span tokens, and sealed-role details remain external inputs to
replay.  Every derived duration, unit count, outcome flag, capture id, run id,
and event id is recomputed from canonical receipt-clock records.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterable, Mapping, Sequence

from forge import krea_calibration_profiles

try:
    from . import krea_budget
    from . import krea_confirmation_admission as admission
    from . import krea_fixture
    from . import krea_provenance
    from . import krea_stage2_production_identity as production
except ImportError:  # pragma: no cover
    import krea_budget  # type: ignore[no-redef]
    import krea_confirmation_admission as admission  # type: ignore[no-redef]
    import krea_fixture  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_stage2_production_identity as production  # type: ignore[no-redef]


SCHEMA = 2
PLAN_KIND = "forge-krea-stage2-timing-plan"
PROBE_CONTRACT_KIND = "forge-krea-stage2-timing-probe-contract"
RAW_RECEIPT_KIND = "forge-krea-stage2-raw-timing-receipt"
EVENT_KIND = "forge-krea-stage2-timing-receipt-event"
RUN_RECEIPT_KIND = "forge-krea-stage2-timing-run-receipt"
COLLECTOR_IDENTITY_KIND = "forge-krea-stage2-receipt-collector-identity"
RECEIPT_MANIFEST_KIND = "forge-krea-stage2-receipt-manifest"
CAPTURE_KIND = "forge-krea-stage2-sanitized-timing-capture"
BUNDLE_KIND = "forge-krea-stage2-timing-evidence-bundle"
AUTHORITY_KIND = "forge-krea-stage2-standing-timing-authority"

OPERATING_NOTES = {
    "path_hint": "SN56-project/SN56-TEAM-OPERATING-NOTES.md",
    "bytes": 4017,
    "sha256": "db3c33441b1daa415de4ceeb5aca3defc865f25371cc7354a13f950cfc6edfdd",
    "order": 11,
}
_AUTHORITY_BODY = {
    "schema": SCHEMA,
    "kind": AUTHORITY_KIND,
    "accountable_owner_identity": "Atulya Shetty",
    "authority_basis": {
        **OPERATING_NOTES,
        "description": "owner authorization is standing and global",
        "identity_assurance": (
            "standing-owner-directive-bound-by-file-hash-not-interactive-"
            "artifact-review"
        ),
    },
    "scope": "stage2_throughput_timing_only",
    "timing_orchestration_authorized": True,
    "sealed_confirmation_content_access_authorized": False,
    "confirmation_training_authorized": False,
    "production_mutation_authorized": False,
    "release_authorized": False,
    "deployment_authorized": False,
    "separate_admission_chain_required_for_sealed_content_or_training": True,
}
STANDING_TIMING_AUTHORITY = {
    **_AUTHORITY_BODY,
    "authority_sha256": krea_provenance.canonical_sha256(_AUTHORITY_BODY),
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA1 = re.compile(r"[0-9a-f]{40}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_TIMING_METRICS = (
    "startup",
    "optimizer_update",
    "checkpoint_save",
    "finalization",
    "upload",
)
_SEALED_ROLES = {f"C{index}" for index in range(1, 5)} | {
    f"B-{hours}-{size}" for hours in ("0p5", "0p75", "1") for size in ("small", "large")
}
_CONFIRMATION_ROLES = {f"C{index}" for index in range(1, 5)}
_MEASUREMENT_RECEIPTS = 3
_HELDOUT_RECEIPTS = 1
_CLOCK_TOLERANCE_NS = 250_000_000
_COMMAND_TEMPLATE_ID = "docker-nvidia-offline-stage2-timing-bootstrap-v2"
_EXECUTABLE_ID = "docker-cli-v1"
_EXECUTABLE_PATH = "/usr/bin/docker"
_COMMAND_PLACEHOLDERS = {
    "container_name": "{container_name}",
    "checkpoint_source": "{checkpoint_source}",
    "evidence_source": "{evidence_source}",
    "timing_plan_sha256": "{timing_plan_sha256}",
    "probe_contract_sha256": "{probe_contract_sha256}",
    "seed": "{seed}",
    "task_id": "{task_id}",
    "expected_repo_name": "{expected_repo_name}",
}
_MOUNT_CONTRACT = {
    "base_model": ("/cache/models/krea--Krea-2-Raw", True),
    "text_encoder": ("/cache/hf_cache/Qwen--Qwen3-VL-4B-Instruct", True),
    "dataset_cache": ("/cache/datasets", True),
    "checkpoints": ("/app/checkpoints", False),
    "run_evidence": ("/run-evidence", False),
}
_COMMAND_TYPED_FIELDS = {
    "network_mode": "none",
    "runtime": "nvidia",
    "entrypoint_mode": "immutable_image_default",
    "in_image_program": "forge.cli",
    "model_type": "krea2",
    "bootstrap_mode": "preprofile_timing_bootstrap",
}
_EVENT_UNIT_SCHEDULE = {
    "startup": [1],
    "optimizer_update": [34],
    "checkpoint_save": [1, 1, 1],
    "finalization": [1],
    "upload": [1],
}
_SEED_VALUES = {"A": 42565431, "B": 309817421, "H": 271828183}
_RECEIPT_SCHEDULE = (
    {
        "ordinal": 0,
        "measurement_role": "timing_measurement",
        "seed_role": "A",
        "seed": _SEED_VALUES["A"],
        "expected_event_units": deepcopy(_EVENT_UNIT_SCHEDULE),
    },
    {
        "ordinal": 1,
        "measurement_role": "timing_measurement",
        "seed_role": "B",
        "seed": _SEED_VALUES["B"],
        "expected_event_units": deepcopy(_EVENT_UNIT_SCHEDULE),
    },
    {
        "ordinal": 2,
        "measurement_role": "timing_measurement",
        "seed_role": "A",
        "seed": _SEED_VALUES["A"],
        "expected_event_units": deepcopy(_EVENT_UNIT_SCHEDULE),
    },
    {
        "ordinal": 3,
        "measurement_role": "held_out_end_to_end",
        "seed_role": "H",
        "seed": _SEED_VALUES["H"],
        "expected_event_units": deepcopy(_EVENT_UNIT_SCHEDULE),
    },
)
_COLLECTOR_ID = "stage2-receipt-collector-v1"
_SELECTION_MODE = "offline_post_training"
_SELECTION_SCORER_IDENTITY = None
_SELECTION_RESERVE_S = 0.0
_TERMINALS = {
    "timing_measurement": ["timing_capture_complete"],
    "held_out_end_to_end": ["natural_completion", "upload_ready"],
}
_CAPTURE_KEYS = {
    "schema",
    "kind",
    "capture_id",
    "receipt_ordinal",
    "measurement_role",
    "source_receipt",
    "timing_plan_sha256",
    "production_image_id",
    "fixture_manifest_sha256",
    "training_dataset_shape_sha256",
    "throughput_equivalence_class",
    "host_execution_identity_sha256",
    "gpu_identity_sha256",
    "base_asset_attestation_sha256",
    "command_policy",
    "command_receipt",
    "event_identity_sha256s",
    "samples",
    "seed_role",
    "seed",
    "hard_budget_s",
    "outer_wall_clock_s",
    "natural_completion",
    "upload_ready",
    "failure_or_fallback_telemetry",
    "run_id",
    "run_record_sha256",
    "run_artifact_manifest",
    "production_mutation_authorized",
    "release_authorized",
    "capture_sha256",
}
_BUNDLE_KEYS = {
    "schema",
    "kind",
    "timing_plan_sha256",
    "production_image_id",
    "training_dataset_shape_sha256",
    "throughput_equivalence_class",
    "receipt_manifest",
    "measurement_capture_count",
    "heldout_capture_count",
    "artifact_schema_sha256",
    "artifacts",
    "profile_sha256",
    "production_mutation_authorized",
    "release_authorized",
    "bundle_sha256",
}
_FIXED_ARTIFACT_SCHEMA = (
    ("timing-plan.json", "plan_sha256"),
    ("margin-policy.json", "margin_policy_sha256"),
    ("measurement-001.json", "capture_sha256"),
    ("measurement-002.json", "capture_sha256"),
    ("measurement-003.json", "capture_sha256"),
    ("heldout-001.json", "capture_sha256"),
    ("raw-samples.json", "raw_sample_manifest_sha256"),
    ("end-to-end.json", "end_to_end_validation_sha256"),
    ("throughput-profile.json", "profile_sha256"),
)


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


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a conservative identifier")
    return value


def _utc(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value) is None
    ):
        raise ValueError(f"{label} must be canonical whole-second UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{label} is not real UTC") from exc
    if parsed > datetime.now(timezone.utc).replace(microsecond=0):
        raise ValueError(f"{label} cannot be future-dated")
    return value


def _utc_ns(value: str, label: str) -> int:
    return int(
        datetime.strptime(_utc(value, label), "%Y-%m-%dT%H:%M:%SZ")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1_000_000_000
    )


def _canonical_file_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(krea_provenance.canonical_bytes(value) + b"\n").hexdigest()


def _canonical_file_bytes(value: Mapping[str, Any]) -> int:
    return len(krea_provenance.canonical_bytes(value) + b"\n")


def _binding(
    *, file_sha256: str, semantic_sha256: str, semantic_key: str
) -> dict[str, str]:
    return {
        "file_sha256": _sha(file_sha256, "binding file SHA-256"),
        semantic_key: _sha(semantic_sha256, semantic_key),
    }


def _validate_binding(value: Any, semantic_key: str, label: str) -> dict[str, str]:
    binding = _object(value, label)
    _exact(binding, {"file_sha256", semantic_key}, label)
    return {
        "file_sha256": _sha(binding["file_sha256"], f"{label} file"),
        semantic_key: _sha(binding[semantic_key], f"{label} semantic"),
    }


def _live_host(value: Any) -> dict[str, Any]:
    record = _object(value, "live Stage-2 host receipt")
    _exact(
        record,
        {
            "schema",
            "kind",
            "machine_id_sha256",
            "boot_id_sha256",
            "kernel_release",
            "machine",
            "cpu_affinity_ids",
            "memory_total_bytes",
            "checkpoint_device",
            "host_execution_identity_sha256",
        },
        "live Stage-2 host receipt",
    )
    if record["schema"] != 1 or record["kind"] != (
        "forge-krea-stage2-live-host-identity"
    ):
        raise ValueError("live Stage-2 host receipt kind differs")
    _sha(record["machine_id_sha256"], "host machine id")
    _sha(record["boot_id_sha256"], "host boot id")
    if any(
        not isinstance(record[field], str) or not record[field]
        for field in ("kernel_release", "machine")
    ):
        raise ValueError("live Stage-2 host platform is empty")
    affinity = record["cpu_affinity_ids"]
    if (
        not isinstance(affinity, list)
        or not affinity
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in affinity
        )
        or affinity != sorted(set(affinity))
    ):
        raise ValueError("live Stage-2 host CPU identity differs")
    memory = record["memory_total_bytes"]
    if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
        raise ValueError("live Stage-2 host memory identity differs")
    device = _object(record["checkpoint_device"], "checkpoint device")
    _exact(device, {"st_dev", "major", "minor"}, "checkpoint device")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in device.values()
    ):
        raise ValueError("live Stage-2 checkpoint device differs")
    body = {
        key: item
        for key, item in record.items()
        if key != "host_execution_identity_sha256"
    }
    if record["host_execution_identity_sha256"] != krea_provenance.canonical_sha256(
        body
    ):
        raise ValueError("live Stage-2 host receipt digest differs")
    return dict(record)


def _live_gpu(value: Any) -> dict[str, Any]:
    record = _object(value, "live Stage-2 GPU receipt")
    _exact(
        record,
        {
            "schema",
            "kind",
            "uuid",
            "name",
            "driver_version",
            "memory_total_mib",
            "compute_capability",
            "pci_bus_id",
            "gpu_identity_sha256",
        },
        "live Stage-2 GPU receipt",
    )
    if (
        record["schema"] != 1
        or record["kind"] != ("forge-krea-stage2-live-gpu-identity")
        or any(
            not isinstance(record[field], str) or not record[field]
            for field in (
                "uuid",
                "name",
                "driver_version",
                "memory_total_mib",
                "compute_capability",
                "pci_bus_id",
            )
        )
    ):
        raise ValueError("live Stage-2 GPU receipt differs")
    body = {key: item for key, item in record.items() if key != "gpu_identity_sha256"}
    if record["gpu_identity_sha256"] != krea_provenance.canonical_sha256(body):
        raise ValueError("live Stage-2 GPU receipt digest differs")
    return dict(record)


def _timing_hours(hard_budget_s: float) -> tuple[float, str, str]:
    hard = krea_budget._require_json_seconds(hard_budget_s, "hard_budget_s")
    values = {
        1800.0: ("0.5", "0p5"),
        2700.0: ("0.75", "0p75"),
        3600.0: ("1.0", "1p0"),
    }
    if hard not in values:
        raise ValueError("Stage-2 timing budget must be 0.5, 0.75, or 1.0 hours")
    hours, label = values[hard]
    return hard, hours, label


def _mount_rows(value: Mapping[str, str]) -> list[dict[str, Any]]:
    mounts = _object(value, "Stage-2 timing mount sources")
    _exact(mounts, set(_MOUNT_CONTRACT), "Stage-2 timing mount sources")
    rows = []
    for purpose, (destination, read_only) in _MOUNT_CONTRACT.items():
        source = mounts[purpose]
        if (
            not isinstance(source, str)
            or not source.startswith("/")
            or os.path.normpath(source) != source
            or "\x00" in source
        ):
            raise ValueError(f"Stage-2 timing {purpose} mount source is invalid")
        rows.append(
            {
                "purpose": purpose,
                "source_root": source,
                "destination": destination,
                "read_only": read_only,
            }
        )
    return rows


def _probe_command_template(
    *,
    production_image_id: str,
    typed_fields: Mapping[str, Any],
    mounts: Sequence[Mapping[str, Any]],
) -> list[str]:
    command = [
        _EXECUTABLE_PATH,
        "run",
        "--rm",
        "--name",
        _COMMAND_PLACEHOLDERS["container_name"],
        "--runtime",
        "nvidia",
        "--gpus",
        f"device={typed_fields['gpu_device']}",
        "--network",
        "none",
    ]
    for mount in mounts:
        source = mount["source_root"]
        if mount["purpose"] == "checkpoints":
            source = _COMMAND_PLACEHOLDERS["checkpoint_source"]
        elif mount["purpose"] == "run_evidence":
            source = _COMMAND_PLACEHOLDERS["evidence_source"]
        specification = f"type=bind,src={source},dst={mount['destination']}"
        if mount["read_only"]:
            specification += ",readonly"
        command.extend(
            [
                "--mount",
                specification,
            ]
        )
    command.extend(
        [
            "--env",
            "FORGE_KREA_CALIBRATION_PROFILE=" + typed_fields["profile_id"],
            "--env",
            "FORGE_KREA_STAGE2_TIMING_PLAN_SHA256="
            + _COMMAND_PLACEHOLDERS["timing_plan_sha256"],
            "--env",
            "FORGE_KREA_STAGE2_TIMING_PROBE_CONTRACT_SHA256="
            + _COMMAND_PLACEHOLDERS["probe_contract_sha256"],
            "--env",
            "FORGE_KREA_STAGE2_TIMING_STEPS=" + str(typed_fields["bootstrap_steps"]),
            "--env",
            "FORGE_KREA_STAGE2_TIMING_SEED=" + _COMMAND_PLACEHOLDERS["seed"],
            "--env",
            "FORGE_KREA_STAGE2_TIMING_RECEIPT_PATH=/run-evidence/"
            + _COMMAND_PLACEHOLDERS["timing_plan_sha256"]
            + "/config-control.json",
            production_image_id,
            "--task-id",
            typed_fields["task_id"],
            "--model",
            "krea/Krea-2-Raw",
            "--dataset-zip",
            "file:///cache/datasets/" + typed_fields["task_id"] + "_tourn.zip",
            "--model-type",
            "krea2",
            "--expected-repo-name",
            typed_fields["expected_repo_name"],
            "--hours-to-complete",
            typed_fields["hours_to_complete"],
        ]
    )
    if typed_fields["trigger_word"] is not None:
        command.extend(["--trigger-word", typed_fields["trigger_word"]])
    return command


def _receipt_namespace(probe: Mapping[str, Any], receipt_ordinal: int) -> str:
    if (
        isinstance(receipt_ordinal, bool)
        or not isinstance(receipt_ordinal, int)
        or not 0 <= receipt_ordinal < len(_RECEIPT_SCHEDULE)
    ):
        raise ValueError("raw timing receipt ordinal differs")
    return f"{probe['probe_contract_sha256'][:24]}-r{receipt_ordinal:02d}"


def receipt_mount_sources(
    probe_contract: Mapping[str, Any], *, receipt_ordinal: int
) -> dict[str, str]:
    """Return the two isolated writable host mount roots for one receipt."""

    probe = validate_probe_contract(probe_contract)
    namespace = _receipt_namespace(probe, receipt_ordinal)
    roots = {row["purpose"]: row["source_root"] for row in probe["mounts"]}
    return {
        "checkpoint_source": os.path.join(roots["checkpoints"], namespace),
        "evidence_source": os.path.join(roots["run_evidence"], namespace),
    }


def render_probe_command(
    probe_contract: Mapping[str, Any],
    *,
    timing_plan_sha256: str,
    receipt_ordinal: int,
) -> list[str]:
    """Realize one exact receipt command from a sealed placeholder template."""

    probe = validate_probe_contract(probe_contract)
    plan_sha = _sha(timing_plan_sha256, "timing plan")
    schedule = probe["receipt_schedule"][receipt_ordinal]
    namespace = _receipt_namespace(probe, receipt_ordinal)
    writable = receipt_mount_sources(probe, receipt_ordinal=receipt_ordinal)
    replacements = {
        _COMMAND_PLACEHOLDERS["container_name"]: "forge-krea-timing-" + namespace,
        _COMMAND_PLACEHOLDERS["checkpoint_source"]: writable["checkpoint_source"],
        _COMMAND_PLACEHOLDERS["evidence_source"]: writable["evidence_source"],
        _COMMAND_PLACEHOLDERS["timing_plan_sha256"]: plan_sha,
        _COMMAND_PLACEHOLDERS["probe_contract_sha256"]: probe["probe_contract_sha256"],
        _COMMAND_PLACEHOLDERS["seed"]: str(schedule["seed"]),
    }
    command = []
    for item in probe["command_argv_template"]:
        rendered = item
        for token, replacement in replacements.items():
            rendered = rendered.replace(token, replacement)
        command.append(rendered)
    if (
        any("{" in item or "}" in item for item in command)
        or command.count(probe["production_image_id"]) != 1
        or "--entrypoint" in command
        or any(item.startswith("--entrypoint=") for item in command)
    ):
        raise AssertionError("internal Stage-2 timing command renderer drifted")
    return command


def seal_probe_contract(
    *,
    created_at_utc: str,
    production_image_id: str,
    measurement_tool_sha256: str,
    collector_executable_sha256: str,
    executable_sha256: str,
    gpu_device: int,
    fixture_role: str,
    fixture_manifest_sha256: str,
    training_archive_sha256: str,
    training_archive_bytes: int,
    profile_id: str,
    hard_budget_s: float,
    mount_sources: Mapping[str, str],
    trigger_word: str | None,
    bootstrap_steps: int = 34,
) -> dict[str, Any]:
    if (
        not isinstance(production_image_id, str)
        or _IMAGE_ID.fullmatch(production_image_id) is None
    ):
        raise ValueError("Stage-2 probe image is not immutable")
    if (
        isinstance(gpu_device, bool)
        or not isinstance(gpu_device, int)
        or not 0 <= gpu_device <= 3
    ):
        raise ValueError("Stage-2 timing probe GPU must be one of 0, 1, 2, or 3")
    role = _safe_id(fixture_role, "timing fixture role")
    if role not in _SEALED_ROLES:
        raise ValueError("Stage-2 timing fixture role is outside C/B")
    frozen, profile = _profile_binding(profile_id)
    hard, hours, hours_label = _timing_hours(hard_budget_s)
    if isinstance(bootstrap_steps, bool) or bootstrap_steps != 34:
        raise ValueError("Stage-2 timing bootstrap requires exactly 34 updates")
    if (
        not isinstance(training_archive_bytes, int)
        or isinstance(training_archive_bytes, bool)
        or training_archive_bytes <= 0
    ):
        raise ValueError("Stage-2 timing training archive byte count is invalid")
    if role in _CONFIRMATION_ROLES:
        if trigger_word is not None:
            raise ValueError("legacy confirmation timing must preserve null trigger")
        trigger = None
    else:
        trigger = _safe_id(trigger_word, "timing trigger word")
    slug = role.lower().replace(".", "p")
    task_id = (
        f"forge-stage2-timing-{slug}-{profile_id.lower()}-h{hours_label}-g{gpu_device}"
    )
    expected_repo_name = task_id + "-output"
    mounts = _mount_rows(mount_sources)
    typed_fields = {
        **_COMMAND_TYPED_FIELDS,
        "gpu_device": gpu_device,
        "fixture_role": role,
        "profile_id": frozen.profile_id,
        "throughput_equivalence_class": profile["throughput_equivalence_class"],
        "hard_budget_s": hard,
        "hours_to_complete": hours,
        "bootstrap_steps": bootstrap_steps,
        "task_id": task_id,
        "expected_repo_name": expected_repo_name,
        "trigger_word": trigger,
    }
    argv = _probe_command_template(
        production_image_id=production_image_id,
        typed_fields=typed_fields,
        mounts=mounts,
    )
    body = {
        "schema": SCHEMA,
        "kind": PROBE_CONTRACT_KIND,
        "created_at_utc": _utc(created_at_utc, "probe contract creation time"),
        "command_template_id": _COMMAND_TEMPLATE_ID,
        "command_fields": typed_fields,
        "command_argv_template": argv,
        "command_template_sha256": krea_provenance.canonical_sha256(argv),
        "executable_id": _EXECUTABLE_ID,
        "executable_path": _EXECUTABLE_PATH,
        "executable_sha256": _sha(executable_sha256, "probe executable"),
        "measurement_tool_sha256": _sha(
            measurement_tool_sha256, "probe measurement tool"
        ),
        "collector_executable_sha256": _sha(
            collector_executable_sha256, "probe collector executable"
        ),
        "production_image_id": production_image_id,
        "fixture_manifest": {
            "role": role,
            "manifest_sha256": _sha(fixture_manifest_sha256, "probe fixture manifest"),
        },
        "training_archive": {
            "sha256": _sha(training_archive_sha256, "probe training archive"),
            "bytes": training_archive_bytes,
        },
        "mounts": mounts,
        "receipt_schedule": deepcopy(list(_RECEIPT_SCHEDULE)),
        "network_mode": "none",
        "runtime": "nvidia",
    }
    return {**body, "probe_contract_sha256": krea_provenance.canonical_sha256(body)}


def validate_probe_contract(
    value: Any, *, plan: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    record = _object(value, "Stage-2 probe contract")
    _exact(
        record,
        {
            "schema",
            "kind",
            "created_at_utc",
            "command_template_id",
            "command_fields",
            "command_argv_template",
            "command_template_sha256",
            "executable_id",
            "executable_path",
            "executable_sha256",
            "measurement_tool_sha256",
            "collector_executable_sha256",
            "production_image_id",
            "fixture_manifest",
            "training_archive",
            "mounts",
            "receipt_schedule",
            "network_mode",
            "runtime",
            "probe_contract_sha256",
        },
        "Stage-2 probe contract",
    )
    expected = seal_probe_contract(
        created_at_utc=record["created_at_utc"],
        production_image_id=record["production_image_id"],
        measurement_tool_sha256=record["measurement_tool_sha256"],
        collector_executable_sha256=record["collector_executable_sha256"],
        executable_sha256=record["executable_sha256"],
        gpu_device=_object(record["command_fields"], "probe command fields")[
            "gpu_device"
        ],
        fixture_role=record["command_fields"]["fixture_role"],
        fixture_manifest_sha256=record["fixture_manifest"]["manifest_sha256"],
        training_archive_sha256=record["training_archive"]["sha256"],
        training_archive_bytes=record["training_archive"]["bytes"],
        profile_id=record["command_fields"]["profile_id"],
        hard_budget_s=record["command_fields"]["hard_budget_s"],
        mount_sources={row["purpose"]: row["source_root"] for row in record["mounts"]},
        trigger_word=record["command_fields"]["trigger_word"],
        bootstrap_steps=record["command_fields"]["bootstrap_steps"],
    )
    if record != expected:
        raise ValueError("Stage-2 probe contract drifted")
    if plan is not None:
        resolved = validate_plan(plan)
        fields = record["command_fields"]
        if (
            record["probe_contract_sha256"]
            != resolved["probe_contract"]["probe_contract_sha256"]
            or record["production_image_id"] != resolved["production_image_id"]
            or record["fixture_manifest"]["role"] != resolved["fixture_role"]
            or record["fixture_manifest"]["manifest_sha256"]
            != resolved["fixture_manifest"]["manifest_sha256"]
            or fields["profile_id"] != resolved["calibration_profile"]["profile_id"]
            or fields["throughput_equivalence_class"]
            != resolved["calibration_profile"]["throughput_equivalence_class"]
            or fields["hard_budget_s"] != resolved["hard_budget_s"]
        ):
            raise ValueError("Stage-2 probe contract differs from timing plan")
    return dict(record)


_CONTROL_KEYS = {
    "fixture_manifest",
    "fixture_manifest_file_sha256",
    "fixture_manifest_file_bytes",
    "production_identity",
    "production_identity_file_sha256",
    "asset_attestation",
    "asset_attestation_file_sha256",
    "probe_contract",
    "probe_contract_file_sha256",
    "live_host_identity",
    "live_host_identity_file_sha256",
    "live_gpu_identity",
    "live_gpu_identity_file_sha256",
    "margin_policy",
    "margin_policy_file_sha256",
    "content_authority_controls",
}
_ADMISSION_CONTROL_KEYS = {
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
}


def _admission_controls(
    value: Any,
    *,
    fixture: Mapping[str, Any],
    fixture_control: Mapping[str, Any],
    fixture_file_sha256: str,
    fixture_file_bytes: int,
    identity: Mapping[str, Any],
    identity_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    controls = _object(value, "sealed-content authority controls")
    _exact(controls, _ADMISSION_CONTROL_KEYS, "sealed-content authority controls")
    records = {
        key: _object(controls[key], key)
        for key in (
            "request",
            "ratification",
            "reveal",
            "materialization",
            "gpu_execution_authorization",
        )
    }
    for key, record_key in (
        ("request_file_sha256", "request"),
        ("ratification_file_sha256", "ratification"),
        ("reveal_file_sha256", "reveal"),
        ("materialization_file_sha256", "materialization"),
        (
            "gpu_execution_authorization_file_sha256",
            "gpu_execution_authorization",
        ),
    ):
        if controls[key] != _canonical_file_sha(records[record_key]):
            raise ValueError(f"{record_key} file SHA-256 differs")
    authorization = admission.validate_gpu_execution_authorization(
        records["gpu_execution_authorization"],
        request=records["request"],
        ratification=records["ratification"],
        reveal=records["reveal"],
        materialization=records["materialization"],
        request_file_sha256=controls["request_file_sha256"],
        ratification_file_sha256=controls["ratification_file_sha256"],
        reveal_file_sha256=controls["reveal_file_sha256"],
        materialization_file_sha256=controls["materialization_file_sha256"],
        production_identity=identity,
        production_identity_file_sha256=identity_file_sha256,
    )
    role = fixture["experimental_role"]
    matches = [
        row
        for row in records["materialization"]["files"]
        if row["role"] == role
        and row["sha256"] == fixture_file_sha256
        and row["bytes"] == fixture_file_bytes
    ]
    if len(matches) != 1:
        raise ValueError("timing fixture is absent from exact admitted materialization")
    request = records["request"]
    if role in _CONFIRMATION_ROLES:
        if fixture_control.get("kind") == (
            "forge-krea-stage2-legacy-confirmation-wrapper"
        ):
            committed_sha256 = _sha(
                _object(
                    fixture_control.get("published_checksum_manifest"),
                    "legacy published checksum manifest",
                ).get("file_sha256"),
                "legacy published checksum manifest file",
            )
            commitment_mode = "legacy_confirmation_checksum_manifest_file_sha256"
        else:
            committed_sha256 = fixture_file_sha256
            commitment_mode = "confirmation_manifest_file_sha256"
        if request["public_commitment_sha256s"][role] != committed_sha256:
            raise ValueError("confirmation timing fixture differs from its commitment")
        fixture_commitment = {
            "mode": commitment_mode,
            "role": role,
            "sha256": committed_sha256,
        }
    elif (
        request["boundary_fixture_manifest_sha256s"].get(role)
        != fixture["manifest_sha256"]
    ):
        raise ValueError("boundary timing fixture differs from its commitment")
    else:
        fixture_commitment = {
            "mode": "boundary_manifest_semantic_sha256",
            "role": role,
            "sha256": fixture["manifest_sha256"],
        }
    binding = {
        "request": _binding(
            file_sha256=controls["request_file_sha256"],
            semantic_sha256=request["request_sha256"],
            semantic_key="request_sha256",
        ),
        "materialization": _binding(
            file_sha256=controls["materialization_file_sha256"],
            semantic_sha256=records["materialization"]["materialization_sha256"],
            semantic_key="materialization_sha256",
        ),
        "gpu_execution_authorization": _binding(
            file_sha256=controls["gpu_execution_authorization_file_sha256"],
            semantic_sha256=authorization["gpu_execution_authorization_sha256"],
            semantic_key="gpu_execution_authorization_sha256",
        ),
        "authorized_at_utc": authorization["authorized_at_utc"],
        "fixture_commitment": fixture_commitment,
    }
    return dict(controls), binding


def _profile_binding(profile_id: str) -> tuple[Any, dict[str, Any]]:
    frozen = krea_calibration_profiles.profile_for_id(profile_id)
    if frozen.profile_id == "K0":
        raise ValueError("K0 has no Stage-2 measured budget-fill profile")
    record = frozen.frozen_record()
    return frozen, {
        "profile_id": frozen.profile_id,
        "profile_sha256": frozen.profile_sha256,
        "throughput_equivalence_class": frozen.throughput_equivalence_class,
        "frozen_record_sha256": krea_provenance.canonical_sha256(record),
    }


def _training_pair_count(fixture: Mapping[str, Any]) -> int:
    rows = fixture.get("training_rows")
    if rows is None and (
        fixture.get("legacy_wrapper_sha256") is not None
        and fixture.get("original_fixture_manifest_reconstructed") is False
    ):
        rows = _object(
            fixture.get("training_dataset_identity"),
            "legacy training dataset identity",
        ).get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("timing fixture has no exact training-pair identity")
    return len(rows)


def _normalize_controls(value: Any) -> dict[str, Any]:
    controls = _object(value, "Stage-2 timing controls")
    _exact(controls, _CONTROL_KEYS, "Stage-2 timing controls")
    fixture_control = dict(_object(controls["fixture_manifest"], "fixture control"))
    if fixture_control.get("kind") == "forge-krea-stage2-legacy-confirmation-wrapper":
        try:
            from . import krea_stage2_legacy_confirmation as legacy
        except ImportError:  # pragma: no cover - direct CLI execution.
            import krea_stage2_legacy_confirmation as legacy  # type: ignore[no-redef]
        wrapper = legacy.validate_wrapper(fixture_control)
        fixture = legacy.score_view(wrapper)
    else:
        fixture = krea_fixture.validate_manifest(fixture_control)
    identity = production.validate(dict(controls["production_identity"]))
    assets = production.validate_asset_attestation(dict(controls["asset_attestation"]))
    probe = validate_probe_contract(controls["probe_contract"])
    host = _live_host(controls["live_host_identity"])
    gpu = _live_gpu(controls["live_gpu_identity"])
    margin = krea_budget.load_margin_policy(dict(controls["margin_policy"]))
    file_pairs = (
        (fixture_control, "fixture_manifest_file_sha256"),
        (identity, "production_identity_file_sha256"),
        (assets, "asset_attestation_file_sha256"),
        (probe, "probe_contract_file_sha256"),
        (host, "live_host_identity_file_sha256"),
        (gpu, "live_gpu_identity_file_sha256"),
        (margin, "margin_policy_file_sha256"),
    )
    if any(_canonical_file_sha(record) != controls[key] for record, key in file_pairs):
        raise ValueError("Stage-2 timing control file binding differs")
    fixture_bytes = controls["fixture_manifest_file_bytes"]
    if (
        isinstance(fixture_bytes, bool)
        or not isinstance(fixture_bytes, int)
        or fixture_bytes != _canonical_file_bytes(fixture_control)
    ):
        raise ValueError("Stage-2 timing fixture byte binding differs")
    if (
        assets["training_identity_sha256"]
        != identity["base_model"]["training_identity_sha256"]
        or assets["attestation_sha256"]
        != identity["base_model"]["asset_attestation_sha256"]
        or probe["measurement_tool_sha256"]
        != identity["runtime_contract"]["measurement_tool_sha256"]
        or probe["production_image_id"] != identity["container_image"]["image_id"]
    ):
        raise ValueError("Stage-2 timing assets/probe differ from production identity")
    role = fixture["experimental_role"]
    if (
        probe["fixture_manifest"]
        != {"role": role, "manifest_sha256": fixture["manifest_sha256"]}
        or probe["training_archive"] != fixture["training_archive"]
    ):
        raise ValueError("Stage-2 timing probe differs from exact fixture/archive")
    authority = controls["content_authority_controls"]
    authority_binding = None
    if role in _SEALED_ROLES:
        if authority is None:
            raise ValueError("sealed Stage-2 timing requires full authority controls")
        authority, authority_binding = _admission_controls(
            authority,
            fixture=fixture,
            fixture_control=fixture_control,
            fixture_file_sha256=controls["fixture_manifest_file_sha256"],
            fixture_file_bytes=fixture_bytes,
            identity=identity,
            identity_file_sha256=controls["production_identity_file_sha256"],
        )
    elif authority is not None:
        raise ValueError("public timing fixture cannot claim sealed-content authority")
    return {
        **dict(controls),
        "fixture_manifest": fixture,
        "fixture_control": fixture_control,
        "production_identity": identity,
        "asset_attestation": assets,
        "probe_contract": probe,
        "live_host_identity": host,
        "live_gpu_identity": gpu,
        "margin_policy": margin,
        "content_authority_controls": authority,
        "authority_binding": authority_binding,
    }


def _execution_envelope(*, frozen: Any, controls: Mapping[str, Any]) -> dict[str, Any]:
    fixture = controls["fixture_manifest"]
    identity = controls["production_identity"]
    runtime = identity["runtime_contract"]
    model = identity["base_model"]
    return krea_budget.seal_execution_envelope(
        equivalence_class=frozen.throughput_equivalence_class,
        network_rank=frozen.rank,
        network_alpha=frozen.alpha,
        optimizer=frozen.optimizer,
        optimizer_config_sha256=krea_provenance.canonical_sha256(
            dict(frozen.optimizer_parameters)
        ),
        loss=frozen.loss,
        differential_guidance_enabled=True,
        guidance_scale=frozen.guidance,
        training_pair_count=_training_pair_count(fixture),
        training_dataset_shape_sha256=fixture["training_dataset_shape_sha256"],
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        data_parallel_replicas=1,
        resolution_policy_sha256=krea_provenance.canonical_sha256([512, 768, 1024]),
        precision_policy_sha256=krea_provenance.canonical_sha256(
            {"train_dtype": "bf16", "save_dtype": "bf16"}
        ),
        cache_latents_to_disk=False,
        cache_text_embeddings=False,
        compile_enabled=False,
        jit_enabled=runtime["jit_enabled"],
        dataloader_workers=0,
        base_model_identity_sha256=model["training_identity_sha256"],
        runtime_identity_sha256=runtime["runtime_identity_sha256"],
        host_execution_identity_sha256=controls["live_host_identity"][
            "host_execution_identity_sha256"
        ],
        execution_surface="immutable_production_docker_image",
        execution_scope="stage2_throughput_timing_only",
        venv_tree_manifest_sha256=runtime["venv_tree_manifest_sha256"],
        reference_container_image_sha256=identity["container_image"]["image_id"].split(
            ":", 1
        )[1],
        gpu_identity_sha256=controls["live_gpu_identity"]["gpu_identity_sha256"],
        trainer_identity_sha256=runtime["trainer_identity_sha256"],
        measurement_tool_sha256=runtime["measurement_tool_sha256"],
    )


def build_plan(
    *,
    controls: Mapping[str, Any],
    profile_id: str,
    hard_budget_s: float,
    created_at_utc: str,
) -> dict[str, Any]:
    resolved = _normalize_controls(controls)
    fixture = resolved["fixture_manifest"]
    identity = resolved["production_identity"]
    probe = resolved["probe_contract"]
    margin = resolved["margin_policy"]
    frozen, profile = _profile_binding(profile_id)
    created = _utc(created_at_utc, "timing plan creation time")
    chronology = [
        _utc_ns(identity["captured_at_utc"], "production identity capture time"),
        _utc_ns(probe["created_at_utc"], "probe contract creation time"),
        _utc_ns(margin["approved_at_utc"], "margin approval time"),
    ]
    if resolved["authority_binding"] is not None:
        chronology.append(
            _utc_ns(
                resolved["authority_binding"]["authorized_at_utc"],
                "GPU authorization time",
            )
        )
    if _utc_ns(created, "timing plan creation time") <= max(chronology):
        raise ValueError("Stage-2 timing plan must follow all bound controls")
    hard = krea_budget._require_json_seconds(hard_budget_s, "hard_budget_s")
    fields = probe["command_fields"]
    if (
        fields["profile_id"] != frozen.profile_id
        or fields["throughput_equivalence_class"] != frozen.throughput_equivalence_class
        or fields["hard_budget_s"] != hard
    ):
        raise ValueError("Stage-2 timing probe profile/budget differs")
    envelope = _execution_envelope(frozen=frozen, controls=resolved)
    forge = dict(identity["forge"])
    body = {
        "schema": SCHEMA,
        "kind": PLAN_KIND,
        "created_at_utc": created,
        "standing_timing_authority": deepcopy(STANDING_TIMING_AUTHORITY),
        "sealed_content_authority": resolved["authority_binding"],
        "production_identity": _binding(
            file_sha256=resolved["production_identity_file_sha256"],
            semantic_sha256=identity["production_identity_sha256"],
            semantic_key="production_identity_sha256",
        ),
        "forge_identity": forge,
        "forge_identity_sha256": krea_provenance.canonical_sha256(forge),
        "production_image_id": identity["container_image"]["image_id"],
        "production_repo_digest": identity["container_image"]["repo_digest"],
        "base_asset_attestation": _binding(
            file_sha256=resolved["asset_attestation_file_sha256"],
            semantic_sha256=resolved["asset_attestation"]["attestation_sha256"],
            semantic_key="attestation_sha256",
        ),
        "base_model_identity_sha256": resolved["asset_attestation"][
            "training_identity_sha256"
        ],
        "fixture_manifest": _binding(
            file_sha256=resolved["fixture_manifest_file_sha256"],
            semantic_sha256=fixture["manifest_sha256"],
            semantic_key="manifest_sha256",
        ),
        "fixture_manifest_file_bytes": resolved["fixture_manifest_file_bytes"],
        "fixture_role": fixture["experimental_role"],
        "training_pair_count": _training_pair_count(fixture),
        "training_dataset_shape_sha256": fixture["training_dataset_shape_sha256"],
        "calibration_profile": profile,
        "execution_envelope": envelope,
        "live_host_receipt": _binding(
            file_sha256=resolved["live_host_identity_file_sha256"],
            semantic_sha256=resolved["live_host_identity"][
                "host_execution_identity_sha256"
            ],
            semantic_key="host_execution_identity_sha256",
        ),
        "live_gpu_receipt": _binding(
            file_sha256=resolved["live_gpu_identity_file_sha256"],
            semantic_sha256=resolved["live_gpu_identity"]["gpu_identity_sha256"],
            semantic_key="gpu_identity_sha256",
        ),
        "probe_contract": _binding(
            file_sha256=resolved["probe_contract_file_sha256"],
            semantic_sha256=probe["probe_contract_sha256"],
            semantic_key="probe_contract_sha256",
        ),
        "command_policy": {
            "command_template_id": probe["command_template_id"],
            "command_template_sha256": probe["command_template_sha256"],
            "executable_id": probe["executable_id"],
            "executable_sha256": probe["executable_sha256"],
        },
        "measurement_tool_sha256": probe["measurement_tool_sha256"],
        "margin_policy": _binding(
            file_sha256=resolved["margin_policy_file_sha256"],
            semantic_sha256=margin["margin_policy_sha256"],
            semantic_key="margin_policy_sha256",
        ),
        "hard_budget_s": hard,
        "receipt_schedule": deepcopy(list(_RECEIPT_SCHEDULE)),
        "expected_measurement_receipts": _MEASUREMENT_RECEIPTS,
        "expected_heldout_receipts": _HELDOUT_RECEIPTS,
        "selection_policy": {
            "selection_mode": _SELECTION_MODE,
            "selection_scorer_identity_sha256": _SELECTION_SCORER_IDENTITY,
            "selection_scoring_reserve_s": _SELECTION_RESERVE_S,
        },
        "network_mode": "none",
        "runtime": "nvidia",
        "fallback_allowed": False,
        "production_mutation_authorized": False,
        "release_authorized": False,
        "deployment_authorized": False,
    }
    plan = validate_plan(
        {**body, "plan_sha256": krea_provenance.canonical_sha256(body)}
    )
    validate_probe_contract(probe, plan=plan)
    return plan


def validate_plan(value: Any) -> dict[str, Any]:
    plan = _object(value, "Stage-2 timing plan")
    keys = {
        "schema",
        "kind",
        "created_at_utc",
        "standing_timing_authority",
        "sealed_content_authority",
        "production_identity",
        "forge_identity",
        "forge_identity_sha256",
        "production_image_id",
        "production_repo_digest",
        "base_asset_attestation",
        "base_model_identity_sha256",
        "fixture_manifest",
        "fixture_manifest_file_bytes",
        "fixture_role",
        "training_pair_count",
        "training_dataset_shape_sha256",
        "calibration_profile",
        "execution_envelope",
        "live_host_receipt",
        "live_gpu_receipt",
        "probe_contract",
        "command_policy",
        "measurement_tool_sha256",
        "margin_policy",
        "hard_budget_s",
        "receipt_schedule",
        "expected_measurement_receipts",
        "expected_heldout_receipts",
        "selection_policy",
        "network_mode",
        "runtime",
        "fallback_allowed",
        "production_mutation_authorized",
        "release_authorized",
        "deployment_authorized",
        "plan_sha256",
    }
    _exact(plan, keys, "Stage-2 timing plan")
    body = {key: item for key, item in plan.items() if key != "plan_sha256"}
    if (
        plan["schema"] != SCHEMA
        or plan["kind"] != PLAN_KIND
        or plan["plan_sha256"] != krea_provenance.canonical_sha256(body)
        or plan["standing_timing_authority"] != STANDING_TIMING_AUTHORITY
    ):
        raise ValueError("Stage-2 timing plan identity/authority differs")
    _utc(plan["created_at_utc"], "timing plan creation time")
    role = _safe_id(plan["fixture_role"], "fixture role")
    if (role in _SEALED_ROLES) != (plan["sealed_content_authority"] is not None):
        raise ValueError("Stage-2 timing sealed authority presence differs")
    if plan["sealed_content_authority"] is not None:
        authority = _object(plan["sealed_content_authority"], "sealed authority")
        _exact(
            authority,
            {
                "request",
                "materialization",
                "gpu_execution_authorization",
                "authorized_at_utc",
                "fixture_commitment",
            },
            "sealed authority",
        )
        _validate_binding(authority["request"], "request_sha256", "request")
        _validate_binding(
            authority["materialization"],
            "materialization_sha256",
            "materialization",
        )
        _validate_binding(
            authority["gpu_execution_authorization"],
            "gpu_execution_authorization_sha256",
            "GPU authorization",
        )
        _utc(authority["authorized_at_utc"], "GPU authorization time")
        commitment = _object(authority["fixture_commitment"], "fixture commitment")
        _exact(commitment, {"mode", "role", "sha256"}, "fixture commitment")
        if commitment["role"] != role or commitment["mode"] not in {
            "confirmation_manifest_file_sha256",
            "legacy_confirmation_checksum_manifest_file_sha256",
            "boundary_manifest_semantic_sha256",
        }:
            raise ValueError("Stage-2 timing fixture commitment differs")
        _sha(commitment["sha256"], "fixture commitment")
    for field, semantic in (
        ("production_identity", "production_identity_sha256"),
        ("base_asset_attestation", "attestation_sha256"),
        ("fixture_manifest", "manifest_sha256"),
        ("live_host_receipt", "host_execution_identity_sha256"),
        ("live_gpu_receipt", "gpu_identity_sha256"),
        ("probe_contract", "probe_contract_sha256"),
        ("margin_policy", "margin_policy_sha256"),
    ):
        _validate_binding(plan[field], semantic, field)
    forge = _object(plan["forge_identity"], "Forge identity")
    if (
        not {"commit_sha1", "tree_sha1", "worktree_state"}.issubset(forge)
        or not isinstance(forge["commit_sha1"], str)
        or _GIT_SHA1.fullmatch(forge["commit_sha1"]) is None
        or not isinstance(forge["tree_sha1"], str)
        or _GIT_SHA1.fullmatch(forge["tree_sha1"]) is None
        or forge["worktree_state"] != "clean-including-untracked"
        or plan["forge_identity_sha256"] != krea_provenance.canonical_sha256(forge)
    ):
        raise ValueError("Stage-2 timing requires an exact clean Forge identity")
    if (
        not isinstance(plan["production_image_id"], str)
        or _IMAGE_ID.fullmatch(plan["production_image_id"]) is None
        or not isinstance(plan["production_repo_digest"], str)
        or "@sha256:" not in plan["production_repo_digest"]
    ):
        raise ValueError("Stage-2 timing image identity is not immutable")
    for field in (
        "base_model_identity_sha256",
        "training_dataset_shape_sha256",
        "measurement_tool_sha256",
    ):
        _sha(plan[field], field)
    for field in (
        "fixture_manifest_file_bytes",
        "training_pair_count",
    ):
        if (
            isinstance(plan[field], bool)
            or not isinstance(plan[field], int)
            or plan[field] <= 0
        ):
            raise ValueError(f"Stage-2 timing {field} is invalid")
    frozen, expected_profile = _profile_binding(
        _object(plan["calibration_profile"], "calibration profile")["profile_id"]
    )
    if plan["calibration_profile"] != expected_profile:
        raise ValueError("Stage-2 timing calibration profile drifted")
    envelope = krea_budget.load_execution_envelope(plan["execution_envelope"])
    if (
        envelope.execution_surface != "immutable_production_docker_image"
        or envelope.execution_scope != "stage2_throughput_timing_only"
        or envelope.training_pair_count != plan["training_pair_count"]
        or envelope.training_dataset_shape_sha256
        != plan["training_dataset_shape_sha256"]
        or envelope.equivalence_class != frozen.throughput_equivalence_class
        or envelope.reference_container_image_sha256
        != plan["production_image_id"].split(":", 1)[1]
        or envelope.base_model_identity_sha256 != plan["base_model_identity_sha256"]
        or envelope.host_execution_identity_sha256
        != plan["live_host_receipt"]["host_execution_identity_sha256"]
        or envelope.gpu_identity_sha256
        != plan["live_gpu_receipt"]["gpu_identity_sha256"]
        or envelope.measurement_tool_sha256 != plan["measurement_tool_sha256"]
    ):
        raise ValueError("Stage-2 timing execution envelope escaped its plan")
    policy = _object(plan["command_policy"], "command policy")
    _exact(
        policy,
        {
            "command_template_id",
            "command_template_sha256",
            "executable_id",
            "executable_sha256",
        },
        "command policy",
    )
    if (
        policy["command_template_id"] != _COMMAND_TEMPLATE_ID
        or policy["executable_id"] != _EXECUTABLE_ID
    ):
        raise ValueError("Stage-2 timing command policy is not allowlisted")
    _sha(policy["command_template_sha256"], "command template")
    _sha(policy["executable_sha256"], "command executable")
    krea_budget._require_json_seconds(plan["hard_budget_s"], "hard_budget_s")
    if (
        plan["expected_measurement_receipts"] != _MEASUREMENT_RECEIPTS
        or plan["expected_heldout_receipts"] != _HELDOUT_RECEIPTS
        or plan["receipt_schedule"] != list(_RECEIPT_SCHEDULE)
        or plan["selection_policy"]
        != {
            "selection_mode": _SELECTION_MODE,
            "selection_scorer_identity_sha256": _SELECTION_SCORER_IDENTITY,
            "selection_scoring_reserve_s": _SELECTION_RESERVE_S,
        }
        or plan["network_mode"] != "none"
        or plan["runtime"] != "nvidia"
        or any(
            plan[field] is not False
            for field in (
                "fallback_allowed",
                "production_mutation_authorized",
                "release_authorized",
                "deployment_authorized",
            )
        )
    ):
        raise ValueError("Stage-2 timing fixed execution policy drifted")
    return dict(plan)


def validate_plan_with_controls(
    value: Any, *, controls: Mapping[str, Any]
) -> dict[str, Any]:
    plan = validate_plan(value)
    rebuilt = build_plan(
        controls=controls,
        profile_id=plan["calibration_profile"]["profile_id"],
        hard_budget_s=plan["hard_budget_s"],
        created_at_utc=plan["created_at_utc"],
    )
    if rebuilt != plan:
        raise ValueError("Stage-2 timing plan differs from exact control replay")
    return plan


def seal_event(
    *,
    sequence: int,
    span_token: str,
    metric: str,
    state: str,
    counter_value: int,
    received_monotonic_ns: int,
) -> dict[str, Any]:
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("timing event sequence is invalid")
    if metric not in _TIMING_METRICS or state not in {"begin", "end"}:
        raise ValueError("timing event metric/state is invalid")
    _safe_id(span_token, "timing span token")
    for value, label in (
        (counter_value, "counter value"),
        (received_monotonic_ns, "receipt clock"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"timing event {label} is invalid")
    body = {
        "schema": SCHEMA,
        "kind": EVENT_KIND,
        "sequence": sequence,
        "span_token": span_token,
        "metric": metric,
        "state": state,
        "counter_value": counter_value,
        "received_monotonic_ns": received_monotonic_ns,
    }
    return {**body, "event_sha256": krea_provenance.canonical_sha256(body)}


def validate_event(value: Any) -> dict[str, Any]:
    event = _object(value, "timing receipt event")
    _exact(
        event,
        {
            "schema",
            "kind",
            "sequence",
            "span_token",
            "metric",
            "state",
            "counter_value",
            "received_monotonic_ns",
            "event_sha256",
        },
        "timing receipt event",
    )
    expected = seal_event(
        sequence=event["sequence"],
        span_token=event["span_token"],
        metric=event["metric"],
        state=event["state"],
        counter_value=event["counter_value"],
        received_monotonic_ns=event["received_monotonic_ns"],
    )
    if event != expected:
        raise ValueError("timing receipt event drifted")
    return dict(event)


def seal_run_receipt(
    *,
    measurement_role: str,
    artifact_manifest_file_sha256: str,
    artifact_manifest_sha256: str,
    recorded_unix_ns: int,
) -> dict[str, Any]:
    if measurement_role not in _TERMINALS:
        raise ValueError("timing run role differs")
    if (
        isinstance(recorded_unix_ns, bool)
        or not isinstance(recorded_unix_ns, int)
        or recorded_unix_ns < 0
    ):
        raise ValueError("timing run receipt clock differs")
    body = {
        "schema": SCHEMA,
        "kind": RUN_RECEIPT_KIND,
        "terminal_events": list(_TERMINALS[measurement_role]),
        "artifact_manifest_file_sha256": _sha(
            artifact_manifest_file_sha256, "artifact manifest file"
        ),
        "artifact_manifest_sha256": _sha(artifact_manifest_sha256, "artifact manifest"),
        "recorded_unix_ns": recorded_unix_ns,
    }
    return {**body, "run_receipt_sha256": krea_provenance.canonical_sha256(body)}


def validate_run_receipt(value: Any, *, measurement_role: str) -> dict[str, Any]:
    receipt = _object(value, "timing run receipt")
    _exact(
        receipt,
        {
            "schema",
            "kind",
            "terminal_events",
            "artifact_manifest_file_sha256",
            "artifact_manifest_sha256",
            "recorded_unix_ns",
            "run_receipt_sha256",
        },
        "timing run receipt",
    )
    expected = seal_run_receipt(
        measurement_role=measurement_role,
        artifact_manifest_file_sha256=receipt["artifact_manifest_file_sha256"],
        artifact_manifest_sha256=receipt["artifact_manifest_sha256"],
        recorded_unix_ns=receipt["recorded_unix_ns"],
    )
    if receipt != expected:
        raise ValueError("timing run receipt drifted")
    return dict(receipt)


def seal_collector_identity(
    *,
    created_at_utc: str,
    collector_executable_sha256: str,
    measurement_tool_sha256: str,
) -> dict[str, Any]:
    executable = _sha(collector_executable_sha256, "receipt collector executable")
    measurement = _sha(measurement_tool_sha256, "receipt measurement tool")
    if executable == measurement:
        raise ValueError("receipt collector must be independent of measurement tool")
    body = {
        "schema": SCHEMA,
        "kind": COLLECTOR_IDENTITY_KIND,
        "created_at_utc": _utc(created_at_utc, "receipt collector creation time"),
        "collector_id": _COLLECTOR_ID,
        "collector_executable_sha256": executable,
        "measurement_tool_sha256": measurement,
    }
    return {
        **body,
        "collector_identity_sha256": krea_provenance.canonical_sha256(body),
    }


def validate_collector_identity(value: Any) -> dict[str, Any]:
    identity = _object(value, "receipt collector identity")
    _exact(
        identity,
        {
            "schema",
            "kind",
            "created_at_utc",
            "collector_id",
            "collector_executable_sha256",
            "measurement_tool_sha256",
            "collector_identity_sha256",
        },
        "receipt collector identity",
    )
    expected = seal_collector_identity(
        created_at_utc=identity["created_at_utc"],
        collector_executable_sha256=identity["collector_executable_sha256"],
        measurement_tool_sha256=identity["measurement_tool_sha256"],
    )
    if identity != expected:
        raise ValueError("receipt collector identity drifted")
    return dict(identity)


_RECEIPT_MANIFEST_ROW_KEYS = {
    "ordinal",
    "measurement_role",
    "seed_role",
    "seed",
    "receipt_file_sha256",
    "receipt_sha256",
    "command_invocation_id",
    "command_started_unix_ns",
    "command_ended_unix_ns",
    "command_started_monotonic_ns",
    "command_ended_monotonic_ns",
    "run_recorded_unix_ns",
    "run_receipt_sha256",
    "run_artifact_manifest_file_sha256",
    "run_artifact_manifest_sha256",
}


def _receipt_manifest_rows(
    receipt_bindings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(receipt_bindings) != len(_RECEIPT_SCHEDULE):
        raise ValueError(
            "receipt manifest requires exactly three measurement and one heldout receipt"
        )
    rows: list[dict[str, Any]] = []
    for ordinal, binding in enumerate(receipt_bindings):
        raw, file_sha, receipt_sha = _receipt_record(binding)
        schedule = _RECEIPT_SCHEDULE[ordinal]
        if (
            raw.get("receipt_ordinal") != ordinal
            or raw.get("measurement_role") != schedule["measurement_role"]
            or raw.get("seed_role") != schedule["seed_role"]
            or raw.get("seed") != schedule["seed"]
        ):
            raise ValueError("receipt manifest row order/seed schedule differs")
        command = _object(raw.get("command"), "manifest command receipt")
        run = _object(raw.get("run_receipt"), "manifest run receipt")
        row = {
            "ordinal": ordinal,
            "measurement_role": schedule["measurement_role"],
            "seed_role": schedule["seed_role"],
            "seed": schedule["seed"],
            "receipt_file_sha256": file_sha,
            "receipt_sha256": receipt_sha,
            "command_invocation_id": _safe_id(
                command.get("invocation_id"), "command invocation id"
            ),
            "command_started_unix_ns": command.get("started_unix_ns"),
            "command_ended_unix_ns": command.get("ended_unix_ns"),
            "command_started_monotonic_ns": command.get("started_monotonic_ns"),
            "command_ended_monotonic_ns": command.get("ended_monotonic_ns"),
            "run_recorded_unix_ns": run.get("recorded_unix_ns"),
            "run_receipt_sha256": _sha(run.get("run_receipt_sha256"), "run receipt"),
            "run_artifact_manifest_file_sha256": _sha(
                run.get("artifact_manifest_file_sha256"),
                "run artifact manifest file",
            ),
            "run_artifact_manifest_sha256": _sha(
                run.get("artifact_manifest_sha256"), "run artifact manifest"
            ),
        }
        for key in (
            "command_started_unix_ns",
            "command_ended_unix_ns",
            "command_started_monotonic_ns",
            "command_ended_monotonic_ns",
            "run_recorded_unix_ns",
        ):
            if isinstance(row[key], bool) or not isinstance(row[key], int):
                raise ValueError("receipt manifest command clock differs")
        if (
            row["command_ended_unix_ns"] <= row["command_started_unix_ns"]
            or row["command_ended_monotonic_ns"] <= row["command_started_monotonic_ns"]
        ):
            raise ValueError("receipt manifest command clock differs")
        rows.append(row)
    return rows


def _validate_manifest_intervals_and_identities(
    rows: Sequence[Mapping[str, Any]]
) -> None:
    for left, right in zip(rows, rows[1:]):
        if (
            right["command_started_unix_ns"] < left["command_ended_unix_ns"]
            or right["command_started_monotonic_ns"]
            < left["command_ended_monotonic_ns"]
        ):
            raise ValueError("timing command intervals overlap or are out of order")
    for fields, label in (
        (("receipt_file_sha256",), "receipt file"),
        (("receipt_sha256",), "receipt semantic"),
        (("command_invocation_id",), "command invocation"),
        (("run_receipt_sha256",), "run receipt"),
        (("run_artifact_manifest_file_sha256",), "run artifact file"),
        (("run_artifact_manifest_sha256",), "run artifact semantic"),
    ):
        identities = [tuple(row[field] for field in fields) for row in rows]
        if len(identities) != len(set(identities)):
            raise ValueError(f"timing {label} identity is not globally unique")


def seal_receipt_manifest(
    *,
    created_at_utc: str,
    collector_identity: Mapping[str, Any],
    collector_identity_file_sha256: str,
    receipt_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    collector = validate_collector_identity(collector_identity)
    collector_file = _sha(
        collector_identity_file_sha256, "receipt collector identity file"
    )
    if collector_file != _canonical_file_sha(collector):
        raise ValueError("receipt collector identity file SHA-256 differs")
    rows = _receipt_manifest_rows(receipt_bindings)
    _validate_manifest_intervals_and_identities(rows)
    created = _utc(created_at_utc, "receipt manifest creation time")
    created_ns = _utc_ns(created, "receipt manifest creation time")
    collector_created_ns = _utc_ns(
        collector["created_at_utc"], "receipt collector creation time"
    )
    if collector_created_ns > created_ns:
        raise ValueError("receipt collector identity postdates receipt manifest")
    if collector_created_ns > min(row["command_started_unix_ns"] for row in rows):
        raise ValueError("receipt collector identity postdates command start")
    if any(
        not row["command_ended_unix_ns"] <= row["run_recorded_unix_ns"] <= created_ns
        for row in rows
    ):
        raise ValueError("receipt manifest run chronology differs")
    body = {
        "schema": SCHEMA,
        "kind": RECEIPT_MANIFEST_KIND,
        "created_at_utc": created,
        "collector_identity": _binding(
            file_sha256=collector_file,
            semantic_sha256=collector["collector_identity_sha256"],
            semantic_key="collector_identity_sha256",
        ),
        "collector": collector,
        "rows": rows,
    }
    return {
        **body,
        "receipt_manifest_sha256": krea_provenance.canonical_sha256(body),
    }


def validate_receipt_manifest(value: Any) -> dict[str, Any]:
    manifest = _object(value, "Stage-2 receipt manifest")
    _exact(
        manifest,
        {
            "schema",
            "kind",
            "created_at_utc",
            "collector_identity",
            "collector",
            "rows",
            "receipt_manifest_sha256",
        },
        "Stage-2 receipt manifest",
    )
    body = {
        key: item for key, item in manifest.items() if key != "receipt_manifest_sha256"
    }
    collector = validate_collector_identity(manifest["collector"])
    binding = _validate_binding(
        manifest["collector_identity"],
        "collector_identity_sha256",
        "receipt collector identity",
    )
    rows = manifest["rows"]
    if (
        manifest["schema"] != SCHEMA
        or manifest["kind"] != RECEIPT_MANIFEST_KIND
        or manifest["receipt_manifest_sha256"] != krea_provenance.canonical_sha256(body)
        or binding["file_sha256"] != _canonical_file_sha(collector)
        or binding["collector_identity_sha256"]
        != collector["collector_identity_sha256"]
        or not isinstance(rows, list)
        or len(rows) != len(_RECEIPT_SCHEDULE)
    ):
        raise ValueError("Stage-2 receipt manifest identity differs")
    _utc(manifest["created_at_utc"], "receipt manifest creation time")
    for ordinal, (row, schedule) in enumerate(zip(rows, _RECEIPT_SCHEDULE)):
        row = _object(row, "receipt manifest row")
        _exact(row, _RECEIPT_MANIFEST_ROW_KEYS, "receipt manifest row")
        if (
            row["ordinal"] != ordinal
            or row["measurement_role"] != schedule["measurement_role"]
            or row["seed_role"] != schedule["seed_role"]
            or row["seed"] != schedule["seed"]
        ):
            raise ValueError("receipt manifest row schedule differs")
        for key in (
            "receipt_file_sha256",
            "receipt_sha256",
            "run_receipt_sha256",
            "run_artifact_manifest_file_sha256",
            "run_artifact_manifest_sha256",
        ):
            _sha(row[key], f"receipt manifest {key}")
        _safe_id(row["command_invocation_id"], "command invocation id")
        for key in (
            "command_started_unix_ns",
            "command_ended_unix_ns",
            "command_started_monotonic_ns",
            "command_ended_monotonic_ns",
            "run_recorded_unix_ns",
        ):
            if isinstance(row[key], bool) or not isinstance(row[key], int):
                raise ValueError("receipt manifest command clock differs")
        if (
            row["command_ended_unix_ns"] <= row["command_started_unix_ns"]
            or row["command_ended_monotonic_ns"] <= row["command_started_monotonic_ns"]
        ):
            raise ValueError("receipt manifest command clock differs")
    _validate_manifest_intervals_and_identities(rows)
    manifest_created_ns = _utc_ns(
        manifest["created_at_utc"], "receipt manifest creation time"
    )
    if (
        _utc_ns(collector["created_at_utc"], "receipt collector creation time")
        > manifest_created_ns
    ):
        raise ValueError("receipt collector identity postdates receipt manifest")
    if _utc_ns(collector["created_at_utc"], "receipt collector creation time") > min(
        row["command_started_unix_ns"] for row in rows
    ):
        raise ValueError("receipt collector identity postdates command start")
    if any(
        not row["command_ended_unix_ns"]
        <= row["run_recorded_unix_ns"]
        <= manifest_created_ns
        for row in rows
    ):
        raise ValueError("receipt manifest run chronology differs")
    return dict(manifest)


_COMMAND_RECEIPT_KEYS = {
    "argv",
    "executable_id",
    "executable_path",
    "executable_sha256",
    "returncode",
    "started_unix_ns",
    "ended_unix_ns",
    "started_monotonic_ns",
    "ended_monotonic_ns",
    "production_image_id",
    "network_mode",
    "runtime",
}


def _command(
    value: Any,
    *,
    probe: Mapping[str, Any],
    timing_plan_sha256: str,
    receipt_ordinal: int,
    image_id: str,
    sealed: bool,
) -> dict[str, Any]:
    command = _object(value, "timing command receipt")
    expected_keys = _COMMAND_RECEIPT_KEYS | ({"invocation_id"} if sealed else set())
    _exact(command, expected_keys, "timing command receipt")
    if (
        command["argv"]
        != render_probe_command(
            probe,
            timing_plan_sha256=timing_plan_sha256,
            receipt_ordinal=receipt_ordinal,
        )
        or command["executable_id"] != probe["executable_id"]
        or command["executable_path"] != probe["executable_path"]
        or command["executable_sha256"] != probe["executable_sha256"]
        or isinstance(command["returncode"], bool)
        or not isinstance(command["returncode"], int)
        or command["returncode"] != 0
        or command["production_image_id"] != image_id
        or command["network_mode"] != "none"
        or command["runtime"] != "nvidia"
    ):
        raise ValueError("timing command escaped its exact allowlisted probe")
    for start_key, end_key, label in (
        ("started_unix_ns", "ended_unix_ns", "unix"),
        ("started_monotonic_ns", "ended_monotonic_ns", "monotonic"),
    ):
        start, end = command[start_key], command[end_key]
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or start < 0
            or isinstance(end, bool)
            or not isinstance(end, int)
            or end <= start
        ):
            raise ValueError(f"timing command {label} clock differs")
    unix_elapsed = command["ended_unix_ns"] - command["started_unix_ns"]
    mono_elapsed = command["ended_monotonic_ns"] - command["started_monotonic_ns"]
    if abs(unix_elapsed - mono_elapsed) > _CLOCK_TOLERANCE_NS:
        raise ValueError("timing command unix/monotonic clocks disagree")
    now_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
    if command["ended_unix_ns"] > now_ns + 60_000_000_000:
        raise ValueError("timing command is future-dated")
    invocation_body = {
        "probe_contract_sha256": probe["probe_contract_sha256"],
        "started_unix_ns": command["started_unix_ns"],
        "ended_unix_ns": command["ended_unix_ns"],
        "started_monotonic_ns": command["started_monotonic_ns"],
        "ended_monotonic_ns": command["ended_monotonic_ns"],
    }
    invocation_id = "inv-" + krea_provenance.canonical_sha256(invocation_body)[:48]
    if sealed and command["invocation_id"] != invocation_id:
        raise ValueError("timing command invocation identity differs")
    return {**dict(command), "invocation_id": invocation_id}


def seal_raw_receipt(
    plan: Mapping[str, Any],
    *,
    probe_contract: Mapping[str, Any],
    receipt_ordinal: int,
    command: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    run_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = validate_plan(plan)
    probe = validate_probe_contract(probe_contract)
    if (
        _canonical_file_sha(probe) != resolved["probe_contract"]["file_sha256"]
        or probe["probe_contract_sha256"]
        != resolved["probe_contract"]["probe_contract_sha256"]
    ):
        raise ValueError("raw receipt probe contract differs from plan")
    if (
        isinstance(receipt_ordinal, bool)
        or not isinstance(receipt_ordinal, int)
        or not 0 <= receipt_ordinal < len(_RECEIPT_SCHEDULE)
    ):
        raise ValueError("raw timing receipt ordinal differs")
    schedule = resolved["receipt_schedule"][receipt_ordinal]
    measurement_role = schedule["measurement_role"]
    normalized_command = _command(
        command,
        probe=probe,
        timing_plan_sha256=resolved["plan_sha256"],
        receipt_ordinal=receipt_ordinal,
        image_id=resolved["production_image_id"],
        sealed=False,
    )
    normalized_events = [validate_event(dict(item)) for item in events]
    if (
        not normalized_events
        or [item["sequence"] for item in normalized_events]
        != list(range(len(normalized_events)))
        or any(
            right["received_monotonic_ns"] <= left["received_monotonic_ns"]
            for left, right in zip(normalized_events, normalized_events[1:])
        )
    ):
        raise ValueError("raw timing receipt event order differs")
    _derive_samples(normalized_events, expected_units=schedule["expected_event_units"])
    run = validate_run_receipt(run_receipt, measurement_role=measurement_role)
    body = {
        "schema": SCHEMA,
        "kind": RAW_RECEIPT_KIND,
        "receipt_ordinal": receipt_ordinal,
        "measurement_role": measurement_role,
        "timing_plan_sha256": resolved["plan_sha256"],
        "probe_contract": dict(resolved["probe_contract"]),
        "live_host_receipt": dict(resolved["live_host_receipt"]),
        "live_gpu_receipt": dict(resolved["live_gpu_receipt"]),
        "seed_role": schedule["seed_role"],
        "seed": schedule["seed"],
        "command": normalized_command,
        "events": normalized_events,
        "event_stream_sha256": krea_provenance.canonical_sha256(normalized_events),
        "run_receipt": run,
    }
    return {**body, "receipt_sha256": krea_provenance.canonical_sha256(body)}


def _receipt_record(value: Any) -> tuple[dict[str, Any], str, str]:
    wrapper = _object(value, "external timing receipt binding")
    _exact(
        wrapper,
        {"record", "file_sha256", "receipt_sha256"},
        "external timing receipt binding",
    )
    record = _object(wrapper["record"], "raw timing receipt")
    if wrapper["file_sha256"] != _canonical_file_sha(record):
        raise ValueError("raw timing receipt file SHA-256 differs")
    if wrapper["receipt_sha256"] != record.get("receipt_sha256"):
        raise ValueError("raw timing receipt semantic binding differs")
    return record, wrapper["file_sha256"], wrapper["receipt_sha256"]


def _derive_samples(
    events: Sequence[Mapping[str, Any]],
    *,
    expected_units: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    units_contract = _object(expected_units, "expected timing event units")
    _exact(units_contract, set(_TIMING_METRICS), "expected timing event units")
    if units_contract != _EVENT_UNIT_SCHEDULE:
        raise ValueError("expected timing event unit schedule drifted")
    samples: dict[str, list[dict[str, Any]]] = {
        metric: [] for metric in _TIMING_METRICS
    }
    open_spans: dict[str, Mapping[str, Any]] = {}
    seen_spans: set[str] = set()
    event_ids: list[str] = []
    for event in events:
        event_ids.append(event["event_sha256"])
        token = event["span_token"]
        if event["state"] == "begin":
            if token in open_spans or token in seen_spans:
                raise ValueError("timing span begins more than once")
            open_spans[token] = event
            continue
        begin = open_spans.pop(token, None)
        if begin is None:
            raise ValueError("timing span ended without a begin receipt")
        if begin["metric"] != event["metric"]:
            raise ValueError("timing span changed metric")
        units = event["counter_value"] - begin["counter_value"]
        if units <= 0:
            raise ValueError("timing span counter did not advance")
        started = begin["received_monotonic_ns"]
        ended = event["received_monotonic_ns"]
        observation = (
            "obs-"
            + krea_provenance.canonical_sha256(
                {"begin": begin["event_sha256"], "end": event["event_sha256"]}
            )[:48]
        )
        samples[event["metric"]].append(
            {
                "observation_id": observation,
                "duration_s": (ended - started) / 1_000_000_000,
                "units": units,
                "started_monotonic_ns": started,
                "ended_monotonic_ns": ended,
            }
        )
        seen_spans.add(token)
    if open_spans or any(not rows for rows in samples.values()):
        raise ValueError("timing receipt has incomplete metric spans")
    if {
        metric: [row["units"] for row in rows] for metric, rows in samples.items()
    } != units_contract:
        raise ValueError(
            "timing receipt event units differ from the predeclared schedule"
        )
    return samples, event_ids


def _derive_capture(
    plan: Mapping[str, Any],
    *,
    controls: Mapping[str, Any],
    receipt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = validate_plan_with_controls(plan, controls=controls)
    normalized_controls = _normalize_controls(controls)
    raw, file_sha, semantic_sha = _receipt_record(receipt_binding)
    expected_keys = {
        "schema",
        "kind",
        "receipt_ordinal",
        "measurement_role",
        "timing_plan_sha256",
        "probe_contract",
        "live_host_receipt",
        "live_gpu_receipt",
        "seed_role",
        "seed",
        "command",
        "events",
        "event_stream_sha256",
        "run_receipt",
        "receipt_sha256",
    }
    _exact(raw, expected_keys, "raw timing receipt")
    body = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    if (
        raw["schema"] != SCHEMA
        or raw["kind"] != RAW_RECEIPT_KIND
        or raw["receipt_sha256"] != krea_provenance.canonical_sha256(body)
        or raw["timing_plan_sha256"] != resolved["plan_sha256"]
        or raw["probe_contract"] != resolved["probe_contract"]
        or raw["live_host_receipt"] != resolved["live_host_receipt"]
        or raw["live_gpu_receipt"] != resolved["live_gpu_receipt"]
    ):
        raise ValueError("raw timing receipt escaped its exact plan")
    probe = normalized_controls["probe_contract"]
    host = normalized_controls["live_host_identity"]
    gpu = normalized_controls["live_gpu_identity"]
    if (
        _canonical_file_sha(host) != raw["live_host_receipt"]["file_sha256"]
        or host["host_execution_identity_sha256"]
        != raw["live_host_receipt"]["host_execution_identity_sha256"]
        or _canonical_file_sha(gpu) != raw["live_gpu_receipt"]["file_sha256"]
        or gpu["gpu_identity_sha256"] != raw["live_gpu_receipt"]["gpu_identity_sha256"]
    ):
        raise ValueError(
            "external live host/GPU receipts differ from raw timing receipt"
        )
    role = raw["measurement_role"]
    ordinal = raw["receipt_ordinal"]
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or not 0 <= ordinal < len(resolved["receipt_schedule"])
    ):
        raise ValueError("raw timing receipt ordinal differs")
    schedule = resolved["receipt_schedule"][ordinal]
    if (
        role != schedule["measurement_role"]
        or raw["seed_role"] != schedule["seed_role"]
        or raw["seed"] != schedule["seed"]
    ):
        raise ValueError("raw timing receipt seed/role differs from fixed plan")
    command = _command(
        raw["command"],
        probe=probe,
        timing_plan_sha256=resolved["plan_sha256"],
        receipt_ordinal=ordinal,
        image_id=resolved["production_image_id"],
        sealed=True,
    )
    if command["started_unix_ns"] <= _utc_ns(resolved["created_at_utc"], "plan time"):
        raise ValueError("timing capture predates its predeclared plan")
    elapsed_ns = command["ended_monotonic_ns"] - command["started_monotonic_ns"]
    hard_ns = int(float(resolved["hard_budget_s"]) * 1_000_000_000)
    if elapsed_ns > hard_ns:
        raise ValueError("timing capture exceeded its predeclared hard budget")
    events = [validate_event(dict(item)) for item in raw["events"]]
    if raw["event_stream_sha256"] != krea_provenance.canonical_sha256(events):
        raise ValueError("raw timing event stream digest differs")
    if any(
        event["received_monotonic_ns"] < command["started_monotonic_ns"]
        or event["received_monotonic_ns"] > command["ended_monotonic_ns"]
        for event in events
    ):
        raise ValueError("timing event escaped the command receipt clock")
    derived, event_ids = _derive_samples(
        events, expected_units=schedule["expected_event_units"]
    )
    capture_id = "cap-" + semantic_sha[:48]
    samples = {
        metric: [
            krea_budget._timing_sample({**row, "capture_id": capture_id}, metric=metric)
            for row in rows
        ]
        for metric, rows in derived.items()
    }
    run = validate_run_receipt(raw["run_receipt"], measurement_role=role)
    if (
        run["recorded_unix_ns"] < command["ended_unix_ns"]
        or run["recorded_unix_ns"]
        > int(datetime.now(timezone.utc).timestamp() * 1_000_000_000) + 60_000_000_000
    ):
        raise ValueError("timing run receipt chronology differs")
    seed_role = _safe_id(raw["seed_role"], "seed role")
    seed = raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("timing receipt seed differs")
    run_id = "run-" + run["run_receipt_sha256"][:48]
    body = {
        "schema": SCHEMA,
        "kind": CAPTURE_KIND,
        "capture_id": capture_id,
        "receipt_ordinal": ordinal,
        "measurement_role": role,
        "source_receipt": {
            "file_sha256": file_sha,
            "receipt_sha256": semantic_sha,
        },
        "timing_plan_sha256": resolved["plan_sha256"],
        "production_image_id": resolved["production_image_id"],
        "fixture_manifest_sha256": resolved["fixture_manifest"]["manifest_sha256"],
        "training_dataset_shape_sha256": resolved["training_dataset_shape_sha256"],
        "throughput_equivalence_class": resolved["calibration_profile"][
            "throughput_equivalence_class"
        ],
        "host_execution_identity_sha256": host["host_execution_identity_sha256"],
        "gpu_identity_sha256": gpu["gpu_identity_sha256"],
        "base_asset_attestation_sha256": resolved["base_asset_attestation"][
            "attestation_sha256"
        ],
        "command_policy": dict(resolved["command_policy"]),
        "command_receipt": {
            "returncode": 0,
            "started_unix_ns": command["started_unix_ns"],
            "ended_unix_ns": command["ended_unix_ns"],
            "started_monotonic_ns": command["started_monotonic_ns"],
            "ended_monotonic_ns": command["ended_monotonic_ns"],
            "event_stream_sha256": raw["event_stream_sha256"],
            "invocation_id": command["invocation_id"],
        },
        "event_identity_sha256s": event_ids,
        "samples": samples,
        "seed_role": seed_role,
        "seed": seed,
        "hard_budget_s": resolved["hard_budget_s"],
        "outer_wall_clock_s": elapsed_ns / 1_000_000_000,
        "natural_completion": role == "held_out_end_to_end",
        "upload_ready": role == "held_out_end_to_end",
        "failure_or_fallback_telemetry": False,
        "run_id": run_id,
        "run_record_sha256": run["run_receipt_sha256"],
        "run_artifact_manifest": {
            "file_sha256": run["artifact_manifest_file_sha256"],
            "artifact_manifest_sha256": run["artifact_manifest_sha256"],
        },
        "production_mutation_authorized": False,
        "release_authorized": False,
    }
    return _validate_capture_record(
        {**body, "capture_sha256": krea_provenance.canonical_sha256(body)},
        plan=resolved,
    )


def _validate_capture_record(value: Any, *, plan: Mapping[str, Any]) -> dict[str, Any]:
    capture = _object(value, "sanitized timing capture")
    _exact(capture, _CAPTURE_KEYS, "sanitized timing capture")
    body = {key: item for key, item in capture.items() if key != "capture_sha256"}
    if (
        capture["schema"] != SCHEMA
        or capture["kind"] != CAPTURE_KIND
        or capture["capture_sha256"] != krea_provenance.canonical_sha256(body)
        or capture["timing_plan_sha256"] != plan["plan_sha256"]
        or capture["production_image_id"] != plan["production_image_id"]
        or capture["fixture_manifest_sha256"]
        != plan["fixture_manifest"]["manifest_sha256"]
        or capture["training_dataset_shape_sha256"]
        != plan["training_dataset_shape_sha256"]
        or capture["throughput_equivalence_class"]
        != plan["calibration_profile"]["throughput_equivalence_class"]
        or capture["host_execution_identity_sha256"]
        != plan["live_host_receipt"]["host_execution_identity_sha256"]
        or capture["gpu_identity_sha256"]
        != plan["live_gpu_receipt"]["gpu_identity_sha256"]
        or capture["base_asset_attestation_sha256"]
        != plan["base_asset_attestation"]["attestation_sha256"]
        or capture["command_policy"] != plan["command_policy"]
        or capture["hard_budget_s"] != plan["hard_budget_s"]
        or capture["production_mutation_authorized"] is not False
        or capture["release_authorized"] is not False
    ):
        raise ValueError("sanitized timing capture differs")
    source = _object(capture["source_receipt"], "capture source receipt")
    _exact(source, {"file_sha256", "receipt_sha256"}, "capture source receipt")
    _sha(source["file_sha256"], "capture source file")
    receipt_sha = _sha(source["receipt_sha256"], "capture source receipt")
    if capture["capture_id"] != "cap-" + receipt_sha[:48]:
        raise ValueError("sanitized timing capture id differs")
    command = _object(capture["command_receipt"], "sanitized command receipt")
    _exact(
        command,
        {
            "returncode",
            "started_unix_ns",
            "ended_unix_ns",
            "started_monotonic_ns",
            "ended_monotonic_ns",
            "event_stream_sha256",
            "invocation_id",
        },
        "sanitized command receipt",
    )
    if command["returncode"] != 0:
        raise ValueError("sanitized command did not succeed")
    for start_key, end_key, label in (
        ("started_unix_ns", "ended_unix_ns", "unix"),
        ("started_monotonic_ns", "ended_monotonic_ns", "monotonic"),
    ):
        start, end = command[start_key], command[end_key]
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or start < 0
            or isinstance(end, bool)
            or not isinstance(end, int)
            or end <= start
        ):
            raise ValueError(f"sanitized command {label} clock differs")
    unix_elapsed = command["ended_unix_ns"] - command["started_unix_ns"]
    mono_elapsed = command["ended_monotonic_ns"] - command["started_monotonic_ns"]
    if (
        abs(unix_elapsed - mono_elapsed) > _CLOCK_TOLERANCE_NS
        or capture["outer_wall_clock_s"] != mono_elapsed / 1_000_000_000
        or capture["outer_wall_clock_s"] > capture["hard_budget_s"]
    ):
        raise ValueError("sanitized command clock/budget differs")
    _sha(command["event_stream_sha256"], "sanitized event stream")
    _safe_id(command["invocation_id"], "sanitized command invocation")
    event_ids = capture["event_identity_sha256s"]
    if (
        not isinstance(event_ids, list)
        or not event_ids
        or any(
            not isinstance(item, str) or _SHA256.fullmatch(item) is None
            for item in event_ids
        )
        or len(event_ids) != len(set(event_ids))
    ):
        raise ValueError("sanitized event identities differ")
    samples = _object(capture["samples"], "sanitized timing samples")
    _exact(samples, set(_TIMING_METRICS), "sanitized timing samples")
    observations: list[str] = []
    for metric in _TIMING_METRICS:
        rows = samples[metric]
        if not isinstance(rows, list) or not rows:
            raise ValueError("sanitized timing samples are empty")
        normalized = [krea_budget._timing_sample(row, metric=metric) for row in rows]
        if normalized != rows or any(
            row["capture_id"] != capture["capture_id"] for row in rows
        ):
            raise ValueError("sanitized timing sample differs")
        observations.extend(row["observation_id"] for row in rows)
    if len(observations) != len(set(observations)):
        raise ValueError("sanitized timing observations are not unique")
    role = capture["measurement_role"]
    ordinal = capture["receipt_ordinal"]
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or not 0 <= ordinal < len(plan["receipt_schedule"])
    ):
        raise ValueError("sanitized timing receipt ordinal differs")
    schedule = plan["receipt_schedule"][ordinal]
    if (
        role != schedule["measurement_role"]
        or capture["seed_role"] != schedule["seed_role"]
        or capture["seed"] != schedule["seed"]
    ):
        raise ValueError("sanitized timing measurement role differs")
    completed = role == "held_out_end_to_end"
    if (
        capture["natural_completion"] is not completed
        or capture["upload_ready"] is not completed
        or capture["failure_or_fallback_telemetry"] is not False
    ):
        raise ValueError("sanitized timing outcome flags differ")
    _safe_id(capture["seed_role"], "sanitized timing seed role")
    seed = capture["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("sanitized timing seed differs")
    run_sha = _sha(capture["run_record_sha256"], "sanitized run record")
    if capture["run_id"] != "run-" + run_sha[:48]:
        raise ValueError("sanitized timing run id differs")
    _validate_binding(
        capture["run_artifact_manifest"],
        "artifact_manifest_sha256",
        "sanitized run artifact manifest",
    )
    return dict(capture)


def _bound_receipt_manifest(
    *,
    receipt_manifest: Mapping[str, Any],
    expected_receipt_manifest_file_sha256: str,
    expected_receipt_manifest_sha256: str,
    receipt_bindings: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = validate_receipt_manifest(receipt_manifest)
    expected_file = _sha(
        expected_receipt_manifest_file_sha256, "expected receipt manifest file"
    )
    expected_semantic = _sha(
        expected_receipt_manifest_sha256, "expected receipt manifest semantic"
    )
    if (
        _canonical_file_sha(manifest) != expected_file
        or manifest["receipt_manifest_sha256"] != expected_semantic
        or manifest["collector"]["measurement_tool_sha256"]
        != plan["measurement_tool_sha256"]
    ):
        raise ValueError("Stage-2 receipt manifest differs from external trust anchor")
    derived_rows = _receipt_manifest_rows(receipt_bindings)
    if manifest["rows"] != derived_rows:
        raise ValueError("Stage-2 receipt manifest does not exactly exhaust receipts")
    return manifest


def _derive_evidence(
    *,
    plan: Mapping[str, Any],
    controls: Mapping[str, Any],
    receipt_manifest: Mapping[str, Any],
    expected_receipt_manifest_file_sha256: str,
    expected_receipt_manifest_sha256: str,
    receipt_bindings: Sequence[Mapping[str, Any]],
    framework_stop_boundary_s: float,
    framework_stop_boundary_source_sha256: str,
) -> dict[str, Any]:
    resolved = validate_plan_with_controls(plan, controls=controls)
    normalized_controls = _normalize_controls(controls)
    manifest = _bound_receipt_manifest(
        receipt_manifest=receipt_manifest,
        expected_receipt_manifest_file_sha256=(expected_receipt_manifest_file_sha256),
        expected_receipt_manifest_sha256=expected_receipt_manifest_sha256,
        receipt_bindings=receipt_bindings,
        plan=resolved,
    )
    captures = [
        _derive_capture(resolved, controls=controls, receipt_binding=item)
        for item in receipt_bindings
    ]
    measurement = sorted(
        (item for item in captures if item["measurement_role"] == "timing_measurement"),
        key=lambda item: item["receipt_ordinal"],
    )
    heldout = sorted(
        (
            item
            for item in captures
            if item["measurement_role"] == "held_out_end_to_end"
        ),
        key=lambda item: item["receipt_ordinal"],
    )
    if len(measurement) != _MEASUREMENT_RECEIPTS or len(heldout) != _HELDOUT_RECEIPTS:
        raise ValueError("Stage-2 timing receipt cardinality differs from fixed plan")
    if [item["receipt_ordinal"] for item in captures] != list(
        range(len(_RECEIPT_SCHEDULE))
    ):
        raise ValueError("Stage-2 timing receipts do not exactly exhaust ordered plan")
    for field in ("capture_id", "run_id"):
        values = [item[field] for item in captures]
        if len(values) != len(set(values)):
            raise ValueError(f"Stage-2 timing {field} is not globally unique")
    receipt_ids = [item["source_receipt"]["receipt_sha256"] for item in captures]
    event_ids = [event for item in captures for event in item["event_identity_sha256s"]]
    if len(receipt_ids) != len(set(receipt_ids)) or len(event_ids) != len(
        set(event_ids)
    ):
        raise ValueError("Stage-2 receipt/event identity is not globally unique")
    invocation_ids = [item["command_receipt"]["invocation_id"] for item in captures]
    artifact_file_ids = [
        item["run_artifact_manifest"]["file_sha256"] for item in captures
    ]
    artifact_semantic_ids = [
        item["run_artifact_manifest"]["artifact_manifest_sha256"] for item in captures
    ]
    if (
        len(invocation_ids) != len(set(invocation_ids))
        or len(artifact_file_ids) != len(set(artifact_file_ids))
        or len(artifact_semantic_ids) != len(set(artifact_semantic_ids))
    ):
        raise ValueError(
            "Stage-2 invocation/run artifact identity is not globally unique"
        )
    ordered_windows = [
        (
            item["command_receipt"]["started_unix_ns"],
            item["command_receipt"]["ended_unix_ns"],
        )
        for item in captures
    ]
    if any(
        right[0] < left[1] for left, right in zip(ordered_windows, ordered_windows[1:])
    ):
        raise ValueError("Stage-2 timing command intervals overlap or are out of order")
    samples = {metric: [] for metric in _TIMING_METRICS}
    command_captures = []
    seed_bindings: dict[str, int] = {}
    for item in measurement:
        receipt = item["command_receipt"]
        command_captures.append(
            {
                "capture_id": item["capture_id"],
                "argv": [
                    "template:" + item["command_policy"]["command_template_id"],
                    "sha256:" + item["command_policy"]["command_template_sha256"],
                ],
                "executable_path": "/sanitized/"
                + item["command_policy"]["executable_id"],
                "executable_sha256": item["command_policy"]["executable_sha256"],
                "returncode": 0,
                "started_unix_ns": receipt["started_unix_ns"],
                "ended_unix_ns": receipt["ended_unix_ns"],
                "event_stream_sha256": receipt["event_stream_sha256"],
            }
        )
        for metric in _TIMING_METRICS:
            samples[metric].extend(item["samples"][metric])
        prior = seed_bindings.setdefault(item["seed_role"], item["seed"])
        if prior != item["seed"]:
            raise ValueError("Stage-2 timing seed role changed")
    raw = krea_budget.seal_timing_sample_manifest(
        execution_envelope=resolved["execution_envelope"],
        probe_contract_sha256=resolved["probe_contract"]["probe_contract_sha256"],
        measurement_tool_sha256=resolved["measurement_tool_sha256"],
        command_captures=command_captures,
        samples=samples,
        seed_bindings=[
            {"role": role, "seed": seed} for role, seed in sorted(seed_bindings.items())
        ],
    )
    e2e = krea_budget.seal_end_to_end_validation(
        execution_envelope_sha256=resolved["execution_envelope"][
            "execution_envelope_sha256"
        ],
        probe_contract_sha256=resolved["probe_contract"]["probe_contract_sha256"],
        runs=[
            {
                "run_id": item["run_id"],
                "seed_role": item["seed_role"],
                "seed": item["seed"],
                "hard_budget_s": item["hard_budget_s"],
                "outer_wall_clock_s": item["outer_wall_clock_s"],
                "natural_completion": item["natural_completion"],
                "upload_ready": item["upload_ready"],
                "failure_or_fallback_telemetry": item["failure_or_fallback_telemetry"],
                "run_record_sha256": item["run_record_sha256"],
            }
            for item in heldout
        ],
    )
    margin = normalized_controls["margin_policy"]
    profile = krea_budget.seal_throughput_profile_from_evidence(
        raw_sample_manifest=raw,
        margin_policy=margin,
        end_to_end_validation=e2e,
        framework_stop_boundary_s=framework_stop_boundary_s,
        framework_stop_boundary_source_sha256=_sha(
            framework_stop_boundary_source_sha256, "framework stop boundary source"
        ),
        selection_mode=_SELECTION_MODE,
        selection_scorer_identity_sha256=_SELECTION_SCORER_IDENTITY,
        selection_scoring_reserve_s=_SELECTION_RESERVE_S,
    )
    return {
        "receipt_manifest": manifest,
        "measurement": measurement,
        "heldout": heldout,
        "raw_samples": raw,
        "end_to_end": e2e,
        "throughput_profile": profile,
        "margin_policy": margin,
    }


def _safe_output_root(path_value: str | Path) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path_value))))
    if path == path.parent or os.path.lexists(path) or not path.parent.is_dir():
        if os.path.lexists(path):
            raise FileExistsError(path)
        raise ValueError("Stage-2 timing output root is invalid")
    current = path.parent
    while True:
        if current.is_symlink():
            raise ValueError("Stage-2 timing output has a symlink ancestor")
        if current == current.parent:
            break
        current = current.parent
    return path


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    payload = krea_provenance.canonical_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)


def _load_canonical(path: Path, label: str) -> dict[str, Any]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
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
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or len(raw) != before.st_size
        ):
            raise ValueError(f"{label} changed while read")
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if (
        not isinstance(value, dict)
        or raw != krea_provenance.canonical_bytes(value) + b"\n"
    ):
        raise ValueError(f"{label} is not canonical JSON plus newline")
    return value


def _artifact_schema(
    measurement_count: int, heldout_count: int
) -> list[tuple[str, str]]:
    if measurement_count != _MEASUREMENT_RECEIPTS or heldout_count != _HELDOUT_RECEIPTS:
        raise ValueError("Stage-2 timing artifact schema cardinality differs")
    return list(_FIXED_ARTIFACT_SCHEMA)


def produce_bundle(
    *,
    plan: Mapping[str, Any],
    controls: Mapping[str, Any],
    receipt_manifest: Mapping[str, Any],
    expected_receipt_manifest_file_sha256: str,
    expected_receipt_manifest_sha256: str,
    receipt_bindings: Sequence[Mapping[str, Any]],
    framework_stop_boundary_s: float,
    framework_stop_boundary_source_sha256: str,
    output_root: str | Path,
) -> dict[str, Any]:
    resolved = validate_plan_with_controls(plan, controls=controls)
    evidence = _derive_evidence(
        plan=resolved,
        controls=controls,
        receipt_manifest=receipt_manifest,
        expected_receipt_manifest_file_sha256=(expected_receipt_manifest_file_sha256),
        expected_receipt_manifest_sha256=expected_receipt_manifest_sha256,
        receipt_bindings=receipt_bindings,
        framework_stop_boundary_s=framework_stop_boundary_s,
        framework_stop_boundary_source_sha256=framework_stop_boundary_source_sha256,
    )
    root = _safe_output_root(output_root)
    root.mkdir(mode=0o700)
    records: dict[str, Mapping[str, Any]] = {
        "timing-plan.json": resolved,
        "margin-policy.json": evidence["margin_policy"],
        "raw-samples.json": evidence["raw_samples"],
        "end-to-end.json": evidence["end_to_end"],
        "throughput-profile.json": evidence["throughput_profile"],
    }
    records.update(
        {
            f"measurement-{index:03d}.json": record
            for index, record in enumerate(evidence["measurement"], 1)
        }
    )
    records.update(
        {
            f"heldout-{index:03d}.json": record
            for index, record in enumerate(evidence["heldout"], 1)
        }
    )
    schema = _artifact_schema(len(evidence["measurement"]), len(evidence["heldout"]))
    artifacts = []
    for name, semantic_key in schema:
        record = records[name]
        _publish(root / name, record)
        artifacts.append(
            {
                "path": name,
                "bytes": (root / name).stat().st_size,
                "file_sha256": krea_provenance.file_sha256(root / name),
                "semantic_sha256": record[semantic_key],
            }
        )
    body = {
        "schema": SCHEMA,
        "kind": BUNDLE_KIND,
        "timing_plan_sha256": resolved["plan_sha256"],
        "production_image_id": resolved["production_image_id"],
        "training_dataset_shape_sha256": resolved["training_dataset_shape_sha256"],
        "throughput_equivalence_class": resolved["calibration_profile"][
            "throughput_equivalence_class"
        ],
        "receipt_manifest": _binding(
            file_sha256=expected_receipt_manifest_file_sha256,
            semantic_sha256=expected_receipt_manifest_sha256,
            semantic_key="receipt_manifest_sha256",
        ),
        "measurement_capture_count": len(evidence["measurement"]),
        "heldout_capture_count": len(evidence["heldout"]),
        "artifact_schema_sha256": krea_provenance.canonical_sha256(schema),
        "artifacts": artifacts,
        "profile_sha256": evidence["throughput_profile"]["profile_sha256"],
        "production_mutation_authorized": False,
        "release_authorized": False,
    }
    bundle = {**body, "bundle_sha256": krea_provenance.canonical_sha256(body)}
    _publish(root / "bundle.json", bundle)
    descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(root, 0o500)
    parent = os.open(root.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return validate_bundle(bundle, root=root)


def validate_bundle(value: Any, *, root: str | Path) -> dict[str, Any]:
    bundle = _object(value, "Stage-2 timing bundle")
    _exact(bundle, _BUNDLE_KEYS, "Stage-2 timing bundle")
    body = {key: item for key, item in bundle.items() if key != "bundle_sha256"}
    if (
        bundle["schema"] != SCHEMA
        or bundle["kind"] != BUNDLE_KIND
        or bundle["bundle_sha256"] != krea_provenance.canonical_sha256(body)
        or bundle["production_mutation_authorized"] is not False
        or bundle["release_authorized"] is not False
    ):
        raise ValueError("Stage-2 timing bundle identity differs")
    for field in (
        "timing_plan_sha256",
        "training_dataset_shape_sha256",
        "artifact_schema_sha256",
        "profile_sha256",
    ):
        _sha(bundle[field], f"timing bundle {field}")
    if (
        not isinstance(bundle["production_image_id"], str)
        or _IMAGE_ID.fullmatch(bundle["production_image_id"]) is None
        or not isinstance(bundle["throughput_equivalence_class"], str)
        or not bundle["throughput_equivalence_class"]
    ):
        raise ValueError("Stage-2 timing bundle execution identity differs")
    _validate_binding(
        bundle["receipt_manifest"],
        "receipt_manifest_sha256",
        "timing bundle receipt manifest",
    )
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(root))))
    current = path
    while True:
        if current.is_symlink():
            raise ValueError("Stage-2 timing bundle has a symlink component")
        if current == current.parent:
            break
        current = current.parent
    if not path.is_dir() or path.stat().st_mode & 0o222:
        raise ValueError("Stage-2 timing bundle root is absent or writable")
    stored_bundle = _load_canonical(path / "bundle.json", "timing bundle")
    if stored_bundle != bundle:
        raise ValueError("Stage-2 timing bundle bytes differ")
    measurement_count = bundle.get("measurement_capture_count")
    heldout_count = bundle.get("heldout_capture_count")
    if measurement_count != _MEASUREMENT_RECEIPTS or heldout_count != _HELDOUT_RECEIPTS:
        raise ValueError("Stage-2 timing bundle capture counts differ")
    schema = _artifact_schema(measurement_count, heldout_count)
    if bundle.get("artifact_schema_sha256") != krea_provenance.canonical_sha256(schema):
        raise ValueError("Stage-2 timing artifact schema differs")
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(schema):
        raise ValueError("Stage-2 timing artifact inventory differs")
    plan = _load_canonical(path / "timing-plan.json", "timing plan")
    validate_plan(plan)
    if (
        bundle["timing_plan_sha256"] != plan["plan_sha256"]
        or bundle["production_image_id"] != plan["production_image_id"]
        or bundle["training_dataset_shape_sha256"]
        != plan["training_dataset_shape_sha256"]
        or bundle["throughput_equivalence_class"]
        != plan["calibration_profile"]["throughput_equivalence_class"]
    ):
        raise ValueError("Stage-2 timing bundle summary differs from plan")
    stored_records: dict[str, dict[str, Any]] = {}
    for row, (name, semantic_key) in zip(artifacts, schema):
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "path",
                "bytes",
                "file_sha256",
                "semantic_sha256",
            }
            or row["path"] != name
        ):
            raise ValueError("Stage-2 timing artifact row differs")
        if (
            isinstance(row["bytes"], bool)
            or not isinstance(row["bytes"], int)
            or row["bytes"] <= 0
        ):
            raise ValueError("Stage-2 timing artifact size differs")
        _sha(row["file_sha256"], "Stage-2 timing artifact file")
        _sha(row["semantic_sha256"], "Stage-2 timing artifact semantic")
        artifact = path / name
        record = _load_canonical(artifact, name)
        stored_records[name] = record
        if (
            artifact.stat().st_mode & 0o222
            or artifact.stat().st_size != row["bytes"]
            or krea_provenance.file_sha256(artifact) != row["file_sha256"]
            or record.get(semantic_key) != row["semantic_sha256"]
        ):
            raise ValueError("Stage-2 timing artifact binding differs")
        if name == "margin-policy.json":
            if (
                krea_budget.load_margin_policy(record) != record
                or row["file_sha256"] != plan["margin_policy"]["file_sha256"]
                or record["margin_policy_sha256"]
                != plan["margin_policy"]["margin_policy_sha256"]
            ):
                raise ValueError("stored timing margin differs")
        elif name.startswith(("measurement-", "heldout-")):
            _validate_capture_record(record, plan=plan)
        elif name == "raw-samples.json":
            krea_budget.load_timing_sample_manifest(record)
        elif name == "end-to-end.json":
            krea_budget.load_end_to_end_validation(record)
        elif name == "throughput-profile.json":
            krea_budget.load_throughput_profile(record)
    raw = stored_records["raw-samples.json"]
    end_to_end = stored_records["end-to-end.json"]
    profile = stored_records["throughput-profile.json"]
    if (
        raw["raw_sample_manifest_sha256"] != profile["raw_sample_manifest_sha256"]
        or end_to_end["end_to_end_validation_sha256"]
        != profile["end_to_end_validation_sha256"]
        or stored_records["margin-policy.json"]["margin_policy_sha256"]
        != profile["margin_policy_sha256"]
        or bundle["profile_sha256"] != profile["profile_sha256"]
    ):
        raise ValueError("Stage-2 timing evidence graph differs")
    expected_names = {name for name, _ in schema} | {"bundle.json"}
    if {item.name for item in path.iterdir()} != expected_names:
        raise ValueError("Stage-2 timing bundle directory inventory differs")
    return dict(bundle)


def bundle_binding(root: str | Path) -> dict[str, str]:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(root))))
    bundle = _load_canonical(path / "bundle.json", "timing bundle")
    validate_bundle(bundle, root=path)
    return {
        "bundle_file_sha256": _canonical_file_sha(bundle),
        "bundle_sha256": bundle["bundle_sha256"],
    }


def load_timing_bundle(root: str | Path) -> dict[str, Any]:
    """Load and fully validate one immutable timing bundle directory."""

    path = Path(os.path.abspath(os.path.expanduser(os.fspath(root))))
    bundle = _load_canonical(path / "bundle.json", "timing bundle")
    validate_bundle(bundle, root=path)
    return {
        "root": str(path),
        "bundle": bundle,
        "plan": _load_canonical(path / "timing-plan.json", "timing plan"),
        "throughput_profile": _load_canonical(
            path / "throughput-profile.json", "throughput profile"
        ),
    }


def replay_bundle(
    root: str | Path,
    *,
    expected_bundle_file_sha256: str,
    expected_bundle_sha256: str,
    controls: Mapping[str, Any],
    receipt_manifest: Mapping[str, Any],
    expected_receipt_manifest_file_sha256: str,
    expected_receipt_manifest_sha256: str,
    receipt_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(root))))
    bundle = _load_canonical(path / "bundle.json", "timing bundle")
    if _canonical_file_sha(bundle) != _sha(
        expected_bundle_file_sha256, "expected bundle file"
    ) or bundle.get("bundle_sha256") != _sha(
        expected_bundle_sha256, "expected bundle semantic"
    ):
        raise ValueError("Stage-2 timing bundle differs from external trust anchor")
    validate_bundle(bundle, root=path)
    expected_manifest_binding = _binding(
        file_sha256=expected_receipt_manifest_file_sha256,
        semantic_sha256=expected_receipt_manifest_sha256,
        semantic_key="receipt_manifest_sha256",
    )
    if bundle["receipt_manifest"] != expected_manifest_binding:
        raise ValueError("Stage-2 receipt manifest differs from published bundle")
    plan = _load_canonical(path / "timing-plan.json", "timing plan")
    plan = validate_plan_with_controls(plan, controls=controls)
    stored_profile = _load_canonical(
        path / "throughput-profile.json", "throughput profile"
    )
    evidence = _derive_evidence(
        plan=plan,
        controls=controls,
        receipt_manifest=receipt_manifest,
        expected_receipt_manifest_file_sha256=(expected_receipt_manifest_file_sha256),
        expected_receipt_manifest_sha256=expected_receipt_manifest_sha256,
        receipt_bindings=receipt_bindings,
        framework_stop_boundary_s=stored_profile["framework_stop_boundary_s"],
        framework_stop_boundary_source_sha256=stored_profile[
            "framework_stop_boundary_source_sha256"
        ],
    )
    expected: dict[str, Mapping[str, Any]] = {
        "margin-policy.json": evidence["margin_policy"],
        "raw-samples.json": evidence["raw_samples"],
        "end-to-end.json": evidence["end_to_end"],
        "throughput-profile.json": evidence["throughput_profile"],
    }
    expected.update(
        {
            f"measurement-{index:03d}.json": record
            for index, record in enumerate(evidence["measurement"], 1)
        }
    )
    expected.update(
        {
            f"heldout-{index:03d}.json": record
            for index, record in enumerate(evidence["heldout"], 1)
        }
    )
    for name, record in expected.items():
        if _load_canonical(path / name, name) != record:
            raise ValueError(f"Stage-2 timing artifact does not replay: {name}")
    if (
        bundle["timing_plan_sha256"] != plan["plan_sha256"]
        or bundle["profile_sha256"] != evidence["throughput_profile"]["profile_sha256"]
    ):
        raise ValueError("Stage-2 timing bundle summary does not replay")
    return {"plan": plan, **evidence, "bundle": bundle}
