"""Exact scorer-only extension to the immutable Stage-1 execution policy.

The historical training evidence remains validated against its exact
execution-surface policy ``98b59f...`` through the historical-validator
adapter.  The live
scorer extension binds the current scorer surface separately.  Scoring D1
takes about 75 minutes, so the scorer needs a
90-minute process limit.  Keeping this as a separate, exact policy preserves
the training claim boundary while making the effective scorer surface
explicit and downgrade-resistant.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


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
    "kind": "forge-krea-stage1-scorer-extension-policy",
    "base_execution_surface_policy_sha256": (
        "15eacab07996cb3d4678fd512b66c3d6cbf71f221b3fd26ff7d619fc73f0ea8e"
    ),
    "scope": "offline_stage1_exact_scoring_only",
    "changes": {
        "evaluation_timeout_profiles": {
            "D1": {
                "evaluation_rows": 24,
                "generations": 5,
                "comparisons_per_row": 2,
                "prompt_count": 240,
                "measured_runtime_s": 4500.0,
                "evaluation_timeout_s": 5400.0,
                "minimum_headroom_s": 900.0,
            },
            "D2": {
                "evaluation_rows": 40,
                "generations": 5,
                "comparisons_per_row": 2,
                "prompt_count": 400,
                "measured_runtime_s": 7500.0,
                "evaluation_timeout_s": 9000.0,
                "minimum_headroom_s": 1500.0,
            },
        },
        "comfy_lora_placeholder": {
            "relative_path": "models/loras/put_loras_here",
            "required_type": "regular_file",
            "required_bytes": 0,
            "required_link_count": 1,
        },
    },
    "training_policy_changes_forbidden": True,
    "policy_downgrade_or_widening_forbidden": True,
}

POLICY = {**_BODY, "policy_sha256": _sha256(_BODY)}


def validate(value: Any) -> dict[str, Any]:
    if value != POLICY:
        raise ValueError("Stage-1 scorer extension policy drifted")
    return dict(POLICY)


def timeout_profile(role: Any) -> dict[str, Any]:
    profiles = POLICY["changes"]["evaluation_timeout_profiles"]
    if role not in profiles:
        raise ValueError("scorer timeout profile is not admitted")
    return dict(profiles[role])


def validate_fixture_profile(
    role: Any,
    *,
    evaluation_rows: Any,
    generations: Any,
) -> dict[str, Any]:
    profile = timeout_profile(role)
    if (
        evaluation_rows != profile["evaluation_rows"]
        or generations != profile["generations"]
        or profile["prompt_count"]
        != profile["evaluation_rows"]
        * profile["generations"]
        * profile["comparisons_per_row"]
        or profile["evaluation_timeout_s"]
        < profile["measured_runtime_s"] + profile["minimum_headroom_s"]
    ):
        raise ValueError("scorer timeout profile differs from fixture shape")
    return profile


def effective_timeouts(base_timeouts: Any, role: Any) -> dict[str, float]:
    """Apply the one authorized timeout change to an exact base contract."""

    expected_base = {
        "startup": 300.0,
        "evaluation": 3600.0,
        "shutdown": 20.0,
        "containment_term_grace": 20.0,
    }
    if base_timeouts != expected_base:
        raise ValueError("scorer extension refuses an incompatible base policy")
    profile = timeout_profile(role)
    return {**expected_base, "evaluation": profile["evaluation_timeout_s"]}
