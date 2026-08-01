"""Fail-closed execution boundary for the Week-5 Krea Stage-2 campaign."""

from __future__ import annotations

import hashlib
import json
from typing import Any


BOUNDARY_ROLES = (
    "B-0p5-small",
    "B-0p5-large",
    "B-0p75-small",
    "B-0p75-large",
    "B-1-small",
    "B-1-large",
)


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


_BODY = {
    "schema": 1,
    "kind": "forge-krea-stage2-execution-surface-policy",
    "accountable_owner_identity": "Atulya Shetty",
    "execution_surface": "immutable_production_docker_image",
    "execution_scope": "confirmation_and_boundary_only",
    "confirmation_fixture_roles": ["C1", "C2", "C3", "C4"],
    "boundary_fixture_roles": list(BOUNDARY_ROLES),
    "authorization_sequence": [
        "validate_public_commitments_and_matrix",
        "post_freeze_custodian_hash_copy_and_inventory",
        "fresh_named_owner_ratification",
        "delegated_reveal_authorization",
        "sealed_fixture_materialization",
        "separate_gpu_execution_authorization",
    ],
    "production_identity_requirements": {
        "clean_worktree_including_untracked": True,
        "exact_forge_commit_and_tree": True,
        "immutable_image_id_and_repo_digest": True,
        "exact_dockerfile_bytes": True,
        "complete_sorted_runtime_input_manifest": True,
    },
    "sealed_fixture_rules": {
        "finalist_selection_completed_before_custodian_capture": True,
        "post_freeze_custodian_may_hash_and_copy_into_fresh_root": True,
        "custodian_capture_may_emit_fixture_bytes": False,
        "ratification_and_reveal_validate_before_root_resolution": True,
        "ratification_may_read_sealed_root": False,
        "reveal_authorization_may_read_sealed_root": False,
        "materialization_may_read_fresh_stage2_root": True,
        "public_evidence_contains_commitments_and_inventory_not_fixture_bytes": True,
    },
    "authorized_technical_agent_roles": [
        "confirmation_reveal_reviewer",
        "confirmation_materialization_reviewer",
        "production_identity_reviewer",
    ],
    "agent_review_is_not_human_review": True,
    "fresh_named_owner_ratification_required": True,
    "claims_forbidden": [
        "gpu_execution_without_separate_authorization",
        "production_mutation",
        "release_or_deployment_authorization",
        "tournament_win_or_field_parity",
    ],
}
POLICY = {**_BODY, "policy_sha256": _sha256(_BODY)}


def validate(value: Any) -> dict[str, Any]:
    """Require the exact Stage-2 policy, including its semantic digest."""

    if value != POLICY:
        raise ValueError("Stage-2 execution-surface policy drifted")
    return dict(POLICY)


def boundary_role(value: Any) -> str:
    """Return one of the six predeclared boundary roles; reject aliases."""

    if value not in BOUNDARY_ROLES:
        raise ValueError("fixture role is not an exact Stage-2 boundary role")
    return str(value)


def technical_role(value: Any) -> str:
    if value not in POLICY["authorized_technical_agent_roles"]:
        raise ValueError("technical agent role is not owner-ratifiable for Stage-2")
    return str(value)
