#!/usr/bin/env python3
"""Mandatory discovery-profile index binding for executable Krea plans.

The profile index is deliberately post-timing and non-authorizing.  It binds
the immutable discovery freeze to exactly one fixture/profile cell without
creating a plan/profile hash cycle.  This module is dependency-light so the
execution-plan validator can consume it directly rather than relying on an
optional CLI wrapper.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

try:
    from . import krea_accelerated_discovery
    from . import krea_provenance
    from . import krea_runtime_binding
except ImportError:  # pragma: no cover - direct script execution.
    import krea_accelerated_discovery  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_runtime_binding  # type: ignore[no-redef]


_SHA256 = re.compile(r"[0-9a-f]{64}")


build_profile_index = krea_runtime_binding.build_profile_index
validate_profile_index = krea_runtime_binding.validate_profile_index


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
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


def load_binding(value: Any) -> tuple[Path, dict[str, Any], str]:
    """Load an exact file+semantic profile-index binding."""

    binding = _object(value, "discovery profile index binding")
    _exact(
        binding,
        {"path", "file_sha256", "index_sha256"},
        "discovery profile index binding",
    )
    path = _safe_file(binding["path"], "discovery profile index")
    raw = path.read_bytes()
    try:
        document = _object(json.loads(raw), "discovery profile index")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("discovery profile index is not JSON") from exc
    if raw != krea_provenance.canonical_bytes(document) + b"\n":
        raise ValueError("discovery profile index must be canonical JSON")
    file_sha = hashlib.sha256(raw).hexdigest()
    validate_profile_index(document)
    if file_sha != _digest(
        binding["file_sha256"], "profile-index file SHA-256"
    ) or document["index_sha256"] != _digest(
        binding["index_sha256"], "profile-index semantic SHA-256"
    ):
        raise ValueError("discovery profile index binding drifted")
    return path, document, file_sha


def validate_plan_cell(
    plan: Mapping[str, Any],
    *,
    fixture: Mapping[str, Any],
    throughput_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a plan to occupy its exact frozen fixture/class/profile cell."""

    _, index, file_sha = load_binding(plan["discovery_profile_index"])
    if (
        plan.get("discovery_execution_authorization")
        != index["discovery_execution_authorization"]
    ):
        raise ValueError(
            "execution plan and profile index bind different discovery authority"
        )
    discovery_binding = _object(plan["discovery_plan"], "plan discovery binding")
    discovery_path = _safe_file(discovery_binding["path"], "plan discovery freeze")
    if (
        str(discovery_path) != index["discovery_plan"]["path"]
        or krea_provenance.file_sha256(discovery_path)
        != index["discovery_plan"]["file_sha256"]
    ):
        raise ValueError("execution plan and profile index bind different freezes")
    fixture_id = plan["discovery_fixture_id"]
    class_name = plan["throughput_equivalence_class"]
    if fixture_id not in {"D1", "D2"}:
        raise ValueError("execution plan fixture is not D1 or D2")
    fixture_slot = index["fixtures"][fixture_id]
    profile_slot = fixture_slot["profiles"].get(class_name)
    if profile_slot is None:
        raise ValueError("execution plan timing class is absent from profile index")
    profile_binding = _object(plan["throughput_profile"], "throughput profile")
    approval_binding = _object(plan["fixture_approval"], "fixture approval")
    _exact(approval_binding, {"path", "sha256"}, "fixture approval")
    approval_path = _safe_file(approval_binding["path"], "fixture approval")
    try:
        approval_document = _object(
            json.loads(approval_path.read_bytes()), "fixture approval"
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("fixture approval is not JSON") from exc
    source_profile = (
        profile_slot["source_profile"]
        if index.get("schema") == 3
        else profile_slot
    )
    if (
        fixture["manifest_sha256"] != fixture_slot["manifest"]["manifest_sha256"]
        or approval_binding["sha256"] != fixture_slot["approval"]["file_sha256"]
        or approval_document.get("approval_sha256")
        != fixture_slot["approval"]["approval_sha256"]
        or profile_binding["sha256"] != source_profile["file_sha256"]
        or throughput_profile["profile_sha256"] != source_profile["profile_sha256"]
    ):
        raise ValueError("execution plan escaped its fixture/class profile-index cell")
    accelerated_cell = None
    if index.get("schema") == 3:
        campaign_binding = index["accelerated_discovery_campaign"]
        _, campaign, campaign_file_sha = (
            krea_accelerated_discovery.load_campaign_binding(campaign_binding)
        )
        index_campaign = index["accelerated_discovery_campaign"]
        if (
            campaign_file_sha != index_campaign["file_sha256"]
            or campaign["campaign_sha256"] != index_campaign["campaign_sha256"]
        ):
            raise ValueError("plan and profile index bind different acceleration")
        accelerated_cell = krea_accelerated_discovery.campaign_cell(
            campaign, fixture_id, plan["arm_id"]
        )
        if (
            accelerated_cell["throughput_equivalence_class"] != class_name
            or accelerated_cell["runtime_factor"] != profile_slot["runtime_factor"]
            or accelerated_cell["effective_hard_budget_s"]
            != profile_slot["effective_hard_budget_s"]
            or accelerated_cell["cell_sha256"]
            not in profile_slot["eligible_cell_sha256"]
        ):
            raise ValueError("plan escaped its accelerated campaign cell")
    return {
        "document": index,
        "file_sha256": file_sha,
        "index_sha256": index["index_sha256"],
        "fixture_id": fixture_id,
        "throughput_equivalence_class": class_name,
        "profile_sha256": source_profile["profile_sha256"],
        "accelerated_cell": accelerated_cell,
        "accelerated_campaign_sha256": (
            index["accelerated_discovery_campaign"]["campaign_sha256"]
            if index.get("schema") == 3
            else None
        ),
        "accelerated_campaign": (
            {
                "document": campaign,
                "file_sha256": campaign_file_sha,
                "campaign_sha256": campaign["campaign_sha256"],
                "cell": accelerated_cell,
            }
            if accelerated_cell is not None
            else None
        ),
        "discovery_execution_authorization": dict(
            index["discovery_execution_authorization"]
        ),
    }
