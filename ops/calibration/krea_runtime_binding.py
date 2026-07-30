#!/usr/bin/env python3
"""Additive CLIs for host, plan, and fixture-scoped timing bindings.

This module deliberately wraps the already-ratified validators instead of
changing their bytes.  The discovery-profile index is a post-timing artifact:
it binds the immutable pre-profile discovery freeze to six exact
``(D1|D2, A|B|C)`` profiles without creating a discovery-plan/profile hash
cycle.  It grants no GPU authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

try:
    from . import krea_budget
    from . import krea_discovery_authorization
    from . import krea_execution_surface_policy
    from . import krea_fixture
    from . import krea_host_identity
    from . import krea_provenance
except ImportError:  # pragma: no cover - direct script execution.
    import krea_budget  # type: ignore[no-redef]
    import krea_discovery_authorization  # type: ignore[no-redef]
    import krea_execution_surface_policy  # type: ignore[no-redef]
    import krea_fixture  # type: ignore[no-redef]
    import krea_host_identity  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]


_SHA256 = re.compile(r"[0-9a-f]{64}")
_FIXTURES = ("D1", "D2")
_INDEX_KIND = "forge-krea-discovery-profile-index"
_DISCOVERY_KIND = "sn56-week5-krea-discovery-freeze"
_PREFLIGHT_INPUT_KEYS = {
    "maximum_load_per_effective_cpu",
    "minimum_available_memory_bytes",
    "minimum_checkpoint_free_bytes",
    "maximum_gpu_utilization_percent",
    "minimum_free_gpu_memory_mib",
    "maximum_foreign_compute_processes",
    "storage_probe_bytes",
    "minimum_checkpoint_write_mib_s",
    "minimum_checkpoint_read_mib_s",
    "maximum_checkpoint_fsync_s",
}


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = keys - set(value)
    extra = set(value) - keys
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


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


def _load_json(
    value: str | Path, label: str, *, canonical: bool
) -> tuple[Path, dict[str, Any], str]:
    path = _safe_file(value, label)
    raw = path.read_bytes()
    try:
        document = _object(json.loads(raw), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if canonical and raw != krea_provenance.canonical_bytes(document) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline")
    return path, document, hashlib.sha256(raw).hexdigest()


def _publish(path: Path, value: dict[str, Any]) -> None:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"output has a symlink ancestor: {current}")
        current = current.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = krea_provenance.canonical_bytes(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _load_discovery(
    path: str | Path,
) -> tuple[Path, dict[str, Any], str, tuple[str, ...]]:
    source, discovery, file_sha = _load_json(path, "discovery plan", canonical=False)
    if (
        discovery.get("schema") != 2
        or discovery.get("kind") != _DISCOVERY_KIND
        or discovery.get("model") != "krea/Krea-2-Raw"
        or discovery.get("model_type") != "krea2"
        or discovery.get("gpu_execution_authorized") is not False
    ):
        raise ValueError("unsupported or self-authorized discovery freeze")
    arms = discovery.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("discovery freeze has no arms")
    raw_classes = {
        row.get("throughput_equivalence_class") for row in arms if isinstance(row, dict)
    }
    if len(raw_classes) != 3 or any(
        not isinstance(item, str) or not item for item in raw_classes
    ):
        raise ValueError("discovery freeze must define exactly three timing classes")
    classes = sorted(raw_classes)
    return source, discovery, file_sha, tuple(classes)


def build_preflight_policy(payload: dict[str, Any]) -> dict[str, Any]:
    """Bind the exact owner-ratified Stage-1 preflight thresholds."""

    payload = _object(payload, "preflight-policy payload")
    _exact(payload, _PREFLIGHT_INPUT_KEYS, "preflight-policy payload")
    if payload != krea_execution_surface_policy.POLICY["stage1_host_preflight_policy"]:
        raise ValueError("preflight thresholds differ from owner-ratified policy")

    def positive_number(key: str) -> float:
        value = payload[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"preflight policy {key} must be positive and finite")
        return float(value)

    for key in (
        "maximum_load_per_effective_cpu",
        "minimum_checkpoint_write_mib_s",
        "minimum_checkpoint_read_mib_s",
        "maximum_checkpoint_fsync_s",
    ):
        positive_number(key)
    for key in (
        "minimum_available_memory_bytes",
        "minimum_checkpoint_free_bytes",
        "minimum_free_gpu_memory_mib",
    ):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"preflight policy {key} must be a positive integer")
    utilization = payload["maximum_gpu_utilization_percent"]
    if (
        isinstance(utilization, bool)
        or not isinstance(utilization, (int, float))
        or not math.isfinite(float(utilization))
        or not 0 <= float(utilization) <= 100
    ):
        raise ValueError("maximum_gpu_utilization_percent must be between 0 and 100")
    if payload["maximum_foreign_compute_processes"] != 0:
        raise ValueError("Week-5 preflight policy must forbid foreign GPU processes")
    probe_bytes = payload["storage_probe_bytes"]
    if (
        isinstance(probe_bytes, bool)
        or not isinstance(probe_bytes, int)
        or not 4 * 1024 * 1024 <= probe_bytes <= 256 * 1024 * 1024
    ):
        raise ValueError("storage_probe_bytes must be between 4 and 256 MiB")
    return {
        **payload,
        "storage_probe_tool_sha256": krea_provenance.file_sha256(
            Path(krea_host_identity.__file__).resolve(strict=True)
        ),
    }


def validate_preflight_policy(value: dict[str, Any]) -> dict[str, Any]:
    value = _object(value, "preflight policy")
    _exact(
        value,
        _PREFLIGHT_INPUT_KEYS | {"storage_probe_tool_sha256"},
        "preflight policy",
    )
    rebuilt = build_preflight_policy({key: value[key] for key in _PREFLIGHT_INPUT_KEYS})
    if value != rebuilt:
        raise ValueError("preflight policy is not canonical for this ratified tool")
    return value


def _load_fixture(fixture_id: str, value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    item = _object(value, f"{fixture_id} fixture input")
    _exact(item, {"manifest", "approval"}, f"{fixture_id} fixture input")
    manifest_path, manifest, manifest_file_sha = _load_json(
        item["manifest"], f"{fixture_id} fixture manifest", canonical=True
    )
    approval_path, approval, approval_file_sha = _load_json(
        item["approval"], f"{fixture_id} fixture approval", canonical=True
    )
    krea_fixture.validate_manifest(manifest)
    krea_fixture.validate_approval(approval, fixture_manifest=manifest)
    if manifest.get("experimental_role") != fixture_id:
        raise ValueError(f"fixture input {fixture_id} has the wrong role")
    return manifest, {
        "manifest": {
            "path": str(manifest_path),
            "file_sha256": manifest_file_sha,
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "approval": {
            "path": str(approval_path),
            "file_sha256": approval_file_sha,
            "approval_sha256": approval["approval_sha256"],
        },
        "concept_id": manifest["concept_id"],
        "training_pair_count": len(manifest["training_rows"]),
        "training_dataset_shape_sha256": manifest["training_dataset_shape_sha256"],
    }


def _class_contracts(
    discovery: dict[str, Any], classes: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for class_name in classes:
        arms = [
            row
            for row in discovery["arms"]
            if row.get("throughput_equivalence_class") == class_name
        ]
        if not arms:
            raise ValueError(f"timing class {class_name} has no discovery arm")
        candidates = {
            krea_provenance.canonical_sha256(
                {
                    "network_rank": row.get("rank"),
                    "network_alpha": row.get("alpha"),
                    "optimizer": row.get("optimizer"),
                    "loss": row.get("loss"),
                    "differential_guidance_enabled": row.get("guidance") is not None,
                    "guidance_scale": (
                        float(row["guidance"])
                        if row.get("guidance") is not None
                        else None
                    ),
                }
            ): {
                "network_rank": row.get("rank"),
                "network_alpha": row.get("alpha"),
                "optimizer": row.get("optimizer"),
                "loss": row.get("loss"),
                "differential_guidance_enabled": row.get("guidance") is not None,
                "guidance_scale": (
                    float(row["guidance"]) if row.get("guidance") is not None else None
                ),
            }
            for row in arms
        }
        if len(candidates) != 1:
            raise ValueError(f"timing class {class_name} has inconsistent arm geometry")
        contract = next(iter(candidates.values()))
        if (
            isinstance(contract["network_rank"], bool)
            or not isinstance(contract["network_rank"], int)
            or isinstance(contract["network_alpha"], bool)
            or not isinstance(contract["network_alpha"], int)
            or not isinstance(contract["optimizer"], str)
            or not isinstance(contract["loss"], str)
        ):
            raise ValueError(f"timing class {class_name} geometry is incomplete")
        contracts[class_name] = contract
    return contracts


def _load_profile(
    fixture_id: str,
    class_name: str,
    value: Any,
    *,
    fixture: dict[str, Any],
    class_contract: dict[str, Any],
) -> dict[str, Any]:
    path, document, file_sha = _load_json(
        value,
        f"throughput profile {fixture_id}/{class_name}",
        canonical=True,
    )
    profile = krea_budget.load_throughput_profile(document)
    envelope = profile.execution_envelope
    expected = {
        "equivalence_class": class_name,
        "training_pair_count": len(fixture["training_rows"]),
        "training_dataset_shape_sha256": fixture["training_dataset_shape_sha256"],
    }
    observed = {
        "equivalence_class": envelope.equivalence_class,
        "training_pair_count": envelope.training_pair_count,
        "training_dataset_shape_sha256": envelope.training_dataset_shape_sha256,
    }
    if observed != expected:
        raise ValueError(
            f"throughput profile {fixture_id}/{class_name} escaped fixture shape: "
            f"expected={expected}, observed={observed}"
        )
    class_observed = {
        "network_rank": envelope.network_rank,
        "network_alpha": envelope.network_alpha,
        "optimizer": envelope.optimizer,
        "loss": envelope.loss,
        "differential_guidance_enabled": envelope.differential_guidance_enabled,
        "guidance_scale": envelope.guidance_scale,
    }
    if class_observed != class_contract:
        raise ValueError(
            f"throughput profile {fixture_id}/{class_name} has wrong class "
            f"geometry: expected={class_contract}, observed={class_observed}"
        )
    campaign_identity = {
        key: getattr(envelope, key)
        for key in (
            "micro_batch_size",
            "gradient_accumulation_steps",
            "data_parallel_replicas",
            "resolution_policy_sha256",
            "precision_policy_sha256",
            "cache_latents_to_disk",
            "cache_text_embeddings",
            "compile_enabled",
            "jit_enabled",
            "dataloader_workers",
            "base_model_identity_sha256",
            "runtime_identity_sha256",
            "host_execution_identity_sha256",
            "execution_surface",
            "execution_scope",
            "venv_tree_manifest_sha256",
            "reference_container_image_sha256",
            "gpu_identity_sha256",
            "trainer_identity_sha256",
            "measurement_tool_sha256",
        )
    }
    return {
        "path": str(path),
        "file_sha256": file_sha,
        "profile_sha256": document["profile_sha256"],
        "execution_envelope_sha256": envelope.execution_envelope_sha256,
        "campaign_runtime_identity_sha256": krea_provenance.canonical_sha256(
            campaign_identity
        ),
    }


def build_profile_index(payload: dict[str, Any]) -> dict[str, Any]:
    """Build the six-cell post-timing index without modifying the freeze."""

    payload = _object(payload, "profile-index payload")
    _exact(
        payload,
        {
            "discovery_plan",
            "discovery_execution_authorization",
            "fixtures",
            "profiles",
        },
        "profile-index payload",
    )
    discovery_path, discovery, discovery_file_sha, classes = _load_discovery(
        payload["discovery_plan"]
    )
    authorization_path, authorization, authorization_file_sha = (
        krea_discovery_authorization.load_binding(
            payload["discovery_execution_authorization"]
        )
    )
    krea_discovery_authorization.assert_matches_discovery(
        authorization,
        discovery_path=discovery_path,
        discovery=discovery,
        discovery_file_sha256=discovery_file_sha,
        action="profile_indexed_discovery_execution",
    )
    class_contracts = _class_contracts(discovery, classes)
    fixture_inputs = _object(payload["fixtures"], "profile-index fixtures")
    profile_inputs = _object(payload["profiles"], "profile-index profiles")
    if set(fixture_inputs) != set(_FIXTURES) or set(profile_inputs) != set(_FIXTURES):
        raise ValueError("profile index requires exactly D1 and D2")

    fixtures: dict[str, Any] = {}
    profile_digests: dict[str, dict[str, str]] = {}
    campaign_runtime_identities: set[str] = set()
    for fixture_id in _FIXTURES:
        fixture, fixture_record = _load_fixture(fixture_id, fixture_inputs[fixture_id])
        slots = _object(profile_inputs[fixture_id], f"{fixture_id} profiles")
        if set(slots) != set(classes):
            raise ValueError(
                f"{fixture_id} must bind one profile for every timing class"
            )
        profile_records = {
            class_name: _load_profile(
                fixture_id,
                class_name,
                slots[class_name],
                fixture=fixture,
                class_contract=class_contracts[class_name],
            )
            for class_name in classes
        }
        fixtures[fixture_id] = {**fixture_record, "profiles": profile_records}
        profile_digests[fixture_id] = {
            class_name: record["profile_sha256"]
            for class_name, record in profile_records.items()
        }
        campaign_runtime_identities.update(
            row["campaign_runtime_identity_sha256"] for row in profile_records.values()
        )
    for class_name in classes:
        if profile_digests["D1"][class_name] == profile_digests["D2"][class_name]:
            raise ValueError(f"D1 and D2 reused one {class_name} throughput profile")
    if len(campaign_runtime_identities) != 1:
        raise ValueError(
            "six-cell profile index mixes campaign host/runtime identities"
        )

    body = {
        "schema": 2,
        "kind": _INDEX_KIND,
        "discovery_plan": {
            "path": str(discovery_path),
            "file_sha256": discovery_file_sha,
        },
        "discovery_execution_authorization": {
            "path": str(authorization_path),
            "file_sha256": authorization_file_sha,
            "authorization_sha256": authorization["authorization_sha256"],
        },
        "throughput_equivalence_classes": list(classes),
        "required_profile_count": len(_FIXTURES) * len(classes),
        "cross_fixture_profile_reuse_forbidden": True,
        "campaign_runtime_identity_sha256": next(iter(campaign_runtime_identities)),
        "fixtures": fixtures,
        "gpu_execution_authorized": False,
    }
    index = {**body, "index_sha256": krea_provenance.canonical_sha256(body)}
    validate_profile_index(index)
    return index


def validate_profile_index(value: dict[str, Any]) -> dict[str, Any]:
    value = _object(value, "discovery-profile index")
    _exact(
        value,
        {
            "schema",
            "kind",
            "discovery_plan",
            "discovery_execution_authorization",
            "throughput_equivalence_classes",
            "required_profile_count",
            "cross_fixture_profile_reuse_forbidden",
            "campaign_runtime_identity_sha256",
            "fixtures",
            "gpu_execution_authorized",
            "index_sha256",
        },
        "discovery-profile index",
    )
    body = {key: item for key, item in value.items() if key != "index_sha256"}
    if (
        value["schema"] != 2
        or value["kind"] != _INDEX_KIND
        or value["gpu_execution_authorized"] is not False
        or value["cross_fixture_profile_reuse_forbidden"] is not True
        or value["index_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("discovery-profile index identity is invalid")
    discovery_binding = _object(value["discovery_plan"], "index discovery plan")
    _exact(discovery_binding, {"path", "file_sha256"}, "index discovery plan")
    _, discovery, discovery_file_sha, classes = _load_discovery(
        discovery_binding["path"]
    )
    class_contracts = _class_contracts(discovery, classes)
    if discovery_file_sha != _digest(
        discovery_binding["file_sha256"], "index discovery file sha256"
    ):
        raise ValueError("indexed discovery freeze bytes drifted")
    _, authorization, _ = krea_discovery_authorization.load_binding(
        value["discovery_execution_authorization"]
    )
    krea_discovery_authorization.assert_matches_discovery(
        authorization,
        discovery_path=_safe_file(
            discovery_binding["path"], "indexed discovery freeze"
        ),
        discovery=discovery,
        discovery_file_sha256=discovery_file_sha,
        action="profile_indexed_discovery_execution",
    )
    if value["throughput_equivalence_classes"] != list(classes):
        raise ValueError("indexed timing classes differ from discovery freeze")
    if value["required_profile_count"] != len(_FIXTURES) * len(classes):
        raise ValueError("indexed profile cardinality is not six")

    fixtures = _object(value["fixtures"], "indexed fixtures")
    if set(fixtures) != set(_FIXTURES):
        raise ValueError("indexed fixtures must be exactly D1 and D2")
    observed_digests: dict[str, dict[str, str]] = {}
    observed_runtime_identities: set[str] = set()
    for fixture_id in _FIXTURES:
        record = _object(fixtures[fixture_id], f"indexed fixture {fixture_id}")
        _exact(
            record,
            {
                "manifest",
                "approval",
                "concept_id",
                "training_pair_count",
                "training_dataset_shape_sha256",
                "profiles",
            },
            f"indexed fixture {fixture_id}",
        )
        manifest_binding = _object(record["manifest"], f"{fixture_id} manifest")
        approval_binding = _object(record["approval"], f"{fixture_id} approval")
        _exact(
            manifest_binding,
            {"path", "file_sha256", "manifest_sha256"},
            f"{fixture_id} manifest",
        )
        _exact(
            approval_binding,
            {"path", "file_sha256", "approval_sha256"},
            f"{fixture_id} approval",
        )
        fixture, rebuilt = _load_fixture(
            fixture_id,
            {
                "manifest": manifest_binding["path"],
                "approval": approval_binding["path"],
            },
        )
        expected_fixture_record = {
            key: rebuilt[key]
            for key in (
                "manifest",
                "approval",
                "concept_id",
                "training_pair_count",
                "training_dataset_shape_sha256",
            )
        }
        actual_fixture_record = {key: record[key] for key in expected_fixture_record}
        if actual_fixture_record != expected_fixture_record:
            raise ValueError(f"indexed fixture {fixture_id} bytes or shape drifted")
        slots = _object(record["profiles"], f"indexed profiles {fixture_id}")
        if set(slots) != set(classes):
            raise ValueError(f"indexed profiles {fixture_id} are incomplete")
        observed_digests[fixture_id] = {}
        for class_name in classes:
            bound = _object(slots[class_name], f"indexed {fixture_id}/{class_name}")
            _exact(
                bound,
                {
                    "path",
                    "file_sha256",
                    "profile_sha256",
                    "execution_envelope_sha256",
                    "campaign_runtime_identity_sha256",
                },
                f"indexed {fixture_id}/{class_name}",
            )
            rebuilt_profile = _load_profile(
                fixture_id,
                class_name,
                bound["path"],
                fixture=fixture,
                class_contract=class_contracts[class_name],
            )
            if bound != rebuilt_profile:
                raise ValueError(
                    f"indexed throughput profile {fixture_id}/{class_name} drifted"
                )
            observed_digests[fixture_id][class_name] = bound["profile_sha256"]
            observed_runtime_identities.add(bound["campaign_runtime_identity_sha256"])
    for class_name in classes:
        if observed_digests["D1"][class_name] == observed_digests["D2"][class_name]:
            raise ValueError("indexed D1/D2 profiles were improperly shared")
    if len(observed_runtime_identities) != 1 or next(
        iter(observed_runtime_identities)
    ) != _digest(
        value["campaign_runtime_identity_sha256"],
        "campaign runtime identity SHA-256",
    ):
        raise ValueError("indexed profiles do not share one host/runtime identity")
    return value


def _load_profile_index(path: str | Path) -> dict[str, Any]:
    _, value, _ = _load_json(path, "discovery-profile index", canonical=True)
    return validate_profile_index(value)


def validate_plan_against_profile_index(
    plan: dict[str, Any], *, profile_index: dict[str, Any]
) -> None:
    """Require the final plan's already-valid fixture/profile to occupy one cell."""

    try:
        from . import krea_execution_plan
    except ImportError:  # pragma: no cover - direct script execution.
        import krea_execution_plan  # type: ignore[no-redef]

    resolved = krea_execution_plan.validate_plan(plan)
    validate_profile_index(profile_index)
    discovery_path = _safe_file(
        plan["discovery_plan"]["path"], "execution plan discovery freeze"
    )
    if (
        str(discovery_path) != profile_index["discovery_plan"]["path"]
        or krea_provenance.file_sha256(discovery_path)
        != profile_index["discovery_plan"]["file_sha256"]
    ):
        raise ValueError("execution plan and profile index bind different freezes")
    fixture_id = plan["discovery_fixture_id"]
    class_name = plan["throughput_equivalence_class"]
    if fixture_id not in _FIXTURES:
        raise ValueError("execution plan fixture is not D1 or D2")
    fixture_slot = profile_index["fixtures"][fixture_id]
    if (
        resolved["fixture"]["manifest_sha256"]
        != fixture_slot["manifest"]["manifest_sha256"]
        or plan["throughput_profile"]["sha256"]
        != fixture_slot["profiles"][class_name]["file_sha256"]
        or resolved["throughput_profile"]["profile_sha256"]
        != fixture_slot["profiles"][class_name]["profile_sha256"]
    ):
        raise ValueError("execution plan escaped its fixture/class profile-index cell")


def _status(action: str, **values: Any) -> None:
    print(
        krea_provenance.canonical_bytes(
            {"status": "PASS", "action": action, **values}
        ).decode("ascii")
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    host_build = subparsers.add_parser("build-host-manifest")
    host_build.add_argument("--checkpoint-path", type=Path, required=True)
    host_build.add_argument("--preflight-policy", type=Path, required=True)
    host_build.add_argument("--bootstrap-receipt", type=Path, required=True)
    host_build.add_argument("--output", type=Path, required=True)
    policy = subparsers.add_parser("seal-preflight-policy")
    policy.add_argument("--payload", type=Path, required=True)
    policy.add_argument("--output", type=Path, required=True)
    policy_verify = subparsers.add_parser("validate-preflight-policy")
    policy_verify.add_argument("--policy", type=Path, required=True)
    for command in ("verify-host-live", "verify-host-static"):
        verify = subparsers.add_parser(command)
        verify.add_argument("--manifest", type=Path, required=True)
        verify.add_argument("--checkpoint-path", type=Path, required=True)
        verify.add_argument("--output", type=Path)

    index = subparsers.add_parser("seal-profile-index")
    index.add_argument("--payload", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    index_verify = subparsers.add_parser("validate-profile-index")
    index_verify.add_argument("--index", type=Path, required=True)

    plan = subparsers.add_parser("seal-execution-plan")
    plan.add_argument("--payload", type=Path, required=True)
    plan.add_argument("--profile-index", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan_verify = subparsers.add_parser("validate-execution-plan")
    plan_verify.add_argument("--plan", type=Path, required=True)
    plan_verify.add_argument("--profile-index", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "seal-preflight-policy":
        _, payload, _ = _load_json(
            args.payload, "preflight-policy payload", canonical=True
        )
        result = build_preflight_policy(payload)
        _publish(args.output, result)
        _status(
            args.command,
            policy_file_sha256=krea_provenance.file_sha256(args.output),
        )
        return 0
    if args.command == "validate-preflight-policy":
        _, result, file_sha = _load_json(
            args.policy, "preflight policy", canonical=True
        )
        validate_preflight_policy(result)
        _status(args.command, policy_file_sha256=file_sha)
        return 0
    if args.command == "build-host-manifest":
        _, policy, _ = _load_json(
            args.preflight_policy, "host preflight policy", canonical=True
        )
        validate_preflight_policy(policy)
        result = krea_host_identity.build_manifest(
            checkpoint_path=args.checkpoint_path,
            preflight_policy=policy,
            bootstrap_receipt_path=args.bootstrap_receipt,
        )
        _publish(args.output, result)
        _status(
            args.command,
            host_execution_identity_sha256=result["host_execution_identity_sha256"],
        )
        return 0
    if args.command in {"verify-host-live", "verify-host-static"}:
        _, manifest, _ = _load_json(
            args.manifest, "host execution manifest", canonical=True
        )
        if args.command == "verify-host-live":
            result = krea_host_identity.verify_live(
                manifest, checkpoint_path=args.checkpoint_path
            )
        else:
            result = krea_host_identity.verify_static(
                manifest, checkpoint_path=args.checkpoint_path
            )
        if args.output is not None:
            _publish(args.output, result)
        _status(
            args.command,
            host_execution_identity_sha256=manifest["host_execution_identity_sha256"],
        )
        return 0
    if args.command == "seal-profile-index":
        _, payload, _ = _load_json(
            args.payload, "profile-index payload", canonical=True
        )
        result = build_profile_index(payload)
        _publish(args.output, result)
        _status(args.command, index_sha256=result["index_sha256"])
        return 0
    if args.command == "validate-profile-index":
        result = _load_profile_index(args.index)
        _status(args.command, index_sha256=result["index_sha256"])
        return 0
    index = _load_profile_index(args.profile_index)
    if args.command == "seal-execution-plan":
        try:
            from . import krea_execution_plan
        except ImportError:  # pragma: no cover - direct script execution.
            import krea_execution_plan  # type: ignore[no-redef]

        _, payload, _ = _load_json(
            args.payload, "execution-plan payload", canonical=True
        )
        result = krea_execution_plan.seal_plan(payload)
        validate_plan_against_profile_index(result, profile_index=index)
        _publish(args.output, result)
    else:
        _, result, _ = _load_json(args.plan, "sealed execution plan", canonical=True)
        validate_plan_against_profile_index(result, profile_index=index)
    _status(args.command, plan_sha256=result["plan_sha256"])
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests.
    raise SystemExit(main())
