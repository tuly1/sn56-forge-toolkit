#!/usr/bin/env python3
"""Owner-ratified, agent-reviewed Krea fixture admission.

This module implements the explicit governance amendment chosen for Week 5:
technical review records may be supplied by honestly labelled agents, while
one named human owner remains accountable and must ratify the exact evidence
interactively.  It never converts an agent into a human reviewer and never
describes the owner's self-attestation as a cryptographic or legal signature.

The authorization sequence is deliberately split:

1. ``prepare`` validates the immutable v2 package and the two original agent
   records, then writes canonical derived evidence, a governance amendment,
   and an inert ratification draft.
2. ``ratify`` is TTY-only and writes the owner's self-attestation.  Neither
   the draft nor the ratification admits a fixture or authorizes a GPU.
3. ``admit-discovery`` builds and validates a portable D1/D2 bundle.  Only its
   final envelope sets ``admission_authorized`` true; GPU authorization stays
   false and must be issued separately by :mod:`krea_execution_plan`.

The discovery path hashes only published C1-C4 commitments.  It does not list,
open, copy, or reveal any sealed C1-C4 content.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
from typing import Any, Mapping

try:
    from . import krea_c1c4_amendment
    from . import krea_dataset_identity
    from . import krea_delegated_review_contract
    from . import krea_execution_surface_policy
    from . import krea_fixture
    from . import krea_fixture_package
    from . import krea_provenance
except ImportError:  # pragma: no cover - direct script execution.
    import krea_c1c4_amendment  # type: ignore[no-redef]
    import krea_dataset_identity  # type: ignore[no-redef]
    import krea_delegated_review_contract  # type: ignore[no-redef]
    import krea_execution_surface_policy  # type: ignore[no-redef]
    import krea_fixture  # type: ignore[no-redef]
    import krea_fixture_package  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]


_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_POLICY_PATH = Path(__file__).with_name("week5") / "krea-governance-policy-v2.json"
_DISCOVERY_PLAN_PATH = Path(__file__).with_name("week5") / "krea-discovery-plan.json"
_POLICY_KIND = "forge-krea-agent-review-owner-ratification-policy"
_SURFACE_KIND = "forge-krea-agent-surface-review"
_INDEPENDENT_KIND = "forge-krea-independent-agent-verification"
_AMENDMENT_KIND = "forge-krea-review-governance-amendment"
_DRAFT_KIND = "forge-krea-owner-ratification-draft"
_PORTABLE_DRAFT_KIND = "forge-krea-portable-owner-ratification-draft"
_RATIFICATION_KIND = "forge-krea-sole-human-owner-ratification"
_ENVELOPE_KIND = "forge-krea-fixture-admission-envelope"
_MODE = "sole-human-owner-ratifies-agent-review-v1"
_AGENT_ASSURANCE = (
    "self-declared-agent-identity-not-human-or-cryptographic-authentication"
)
_OWNER_ASSURANCE = (
    "interactive-owner-self-attestation-not-cryptographic-or-legal-signature"
)
_OWNER = "Atulya Shetty"
_ROLES = ("D1", "D2")
_SURFACES = ("rights", "captions", "similarity")
_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
_CLAIM_LIMIT = (
    "owner-ratified-agent-evidence; Stage-1 is staged-host-venv discovery-only, "
    "not production/release/tournament evidence; Stage-2 requires a separate "
    "Forge commit and fresh named-owner ratification"
)
_SURFACE_SOURCE_FILE_SHA256 = (
    "a7d4b01404822fcbee8ae2143ff8527958851a29828a45bbb8610a0149cd34dc"
)
_INDEPENDENT_SOURCE_FILE_SHA256 = (
    "ded8bd16cfe415118d46dac5d908651caca8f050a8555ffdc8e438ada7bde324"
)
_SUCCESSOR_ADMISSION_COMMIT = "58822b496019177a02fa6196247ac30e788331bb"
_SUCCESSOR_ADMISSION_TREE = "ba569913ceeddab6c425efd97b3dfb39a290a9c5"
_SUCCESSOR_RUNTIME_BASE_COMMIT = "fc70e616b7b9b5ffbd590cf0433609cd4d3528e6"
_SUCCESSOR_ALLOWED_RUNTIME_PATHS = frozenset(
    {
        "ops/calibration/krea_fixture_admission.py",
        "ops/calibration/krea_runtime_binding.py",
        "ops/calibration/run_krea_ladder.py",
    }
)
_GPU_GATE_IMPLEMENTATION_PATHS = {
    "fixture_validator_file_sha256": "ops/calibration/krea_fixture.py",
    "admission_tool_file_sha256": "ops/calibration/krea_fixture_admission.py",
    "execution_plan_file_sha256": "ops/calibration/krea_execution_plan.py",
    "host_identity_file_sha256": "ops/calibration/krea_host_identity.py",
    "runtime_binding_file_sha256": "ops/calibration/krea_runtime_binding.py",
    "host_bootstrap_file_sha256": "ops/calibration/krea_host_bootstrap.py",
    "stage1_runtime_file_sha256": "ops/calibration/krea_stage1_runtime.py",
    "public_source_review_file_sha256": (
        "ops/calibration/krea_public_source_review.py"
    ),
    "execution_surface_policy_file_sha256": (
        "ops/calibration/krea_execution_surface_policy.py"
    ),
    "discovery_authorization_file_sha256": (
        "ops/calibration/krea_discovery_authorization.py"
    ),
    "profile_index_file_sha256": "ops/calibration/krea_profile_index.py",
    "budget_planner_file_sha256": "ops/calibration/krea_budget.py",
    "runner_file_sha256": "ops/calibration/run_krea_ladder.py",
    "training_evidence_file_sha256": (
        "ops/calibration/krea_training_evidence.py"
    ),
    "decision_tool_file_sha256": "ops/calibration/krea_decision.py",
    "score_plan_file_sha256": "ops/calibration/krea_score_plan.py",
    "batch_evaluator_file_sha256": "ops/calibration/batch_evaluate_krea.py",
    "delegated_review_contract_loader_file_sha256": (
        "ops/calibration/krea_delegated_review_contract.py"
    ),
}
_GOD_ORIGIN = "https://github.com/rayonlabs/G.O.D.git"
_GOD_IMAGE_IO = PurePosixPath("validator/evaluation/image_io.py")
_GOD_DATASET_CONSTANTS = PurePosixPath("validator/tasks/datasets/constants.py")
_GOD_ENUMERATOR_AST = ast.dump(
    ast.parse(
        """def list_supported_images(dataset_path: str, extensions: tuple) -> list[str]:
    return [file_name for file_name in os.listdir(dataset_path) if file_name.lower().endswith(extensions)]
"""
    ).body[0],
    include_attributes=False,
)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = keys - set(value)
    extra = set(value) - keys
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _reject_true_authorization_flags(value: Any, label: str) -> None:
    """Reject authority smuggled anywhere inside imported evidence."""

    if isinstance(value, dict):
        for key, nested in value.items():
            if (
                key in {"admission_authorized", "gpu_execution_authorized"}
                and nested is True
            ):
                raise ValueError(f"{label} cannot carry {key}=true")
            _reject_true_authorization_flags(nested, label)
    elif isinstance(value, list):
        for nested in value:
            _reject_true_authorization_flags(nested, label)


def _text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != " ".join(value.split())
    ):
        raise ValueError(f"{label} must be a canonical non-empty string")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _utc(value: Any, label: str) -> str:
    value = _text(value, label)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        raise ValueError(f"{label} must be canonical whole-second UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{label} is not real UTC") from exc
    if parsed > datetime.now(timezone.utc).replace(microsecond=0):
        raise ValueError(f"{label} is in the future")
    return value


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_file(value: Path | str, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(value)))
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _safe_directory(value: Path | str, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(value)))
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real directory: {path}")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _safe_file(path, "hashed file").open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_stable(path: Path, label: str) -> bytes:
    path = _safe_file(path, label)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
    )
    if identity(before) != identity(after):
        raise RuntimeError(f"{label} changed while read")
    return raw


def _json(path: Path, label: str, *, canonical: bool) -> tuple[dict[str, Any], str]:
    raw = _read_stable(path, label)
    try:
        value = _object(json.loads(raw), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if canonical and raw != krea_provenance.canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline")
    return value, hashlib.sha256(raw).hexdigest()


def _write_canonical(path: Path, value: dict[str, Any]) -> None:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {path}")
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


def _binding(path: Path, semantic_key: str, semantic_value: str) -> dict[str, str]:
    return {
        "file_sha256": _file_sha256(path),
        semantic_key: _digest(semantic_value, semantic_key),
    }


def _agent_actor(
    *, actor_id: str, display_name: str, role: str, source_file_sha256: str
) -> dict[str, str]:
    source_file_sha256 = _digest(source_file_sha256, "agent source file SHA-256")
    actor = {
        "actor_class": "agent",
        "actor_id": actor_id,
        "display_name": _text(display_name, "agent display name"),
        "role": role,
        "review_instance_id": f"review-{source_file_sha256[:24]}",
        "identity_assurance": _AGENT_ASSURANCE,
    }
    return krea_fixture._agent_actor(actor, "agent actor")


def _admission_implementation_actor(
    amendment: Mapping[str, Any], *, role: str
) -> dict[str, str]:
    """Derive a producer from the implementation the owner actually ratified.

    Successor validation must not relabel immutable manifests or envelopes with
    the live module hash.  The canonical actor is therefore derived from the
    admission tool SHA stored in the fully validated amendment, whether that
    amendment names the current implementation or the hash-proven 588 ancestor.
    """

    implementation = _object(
        amendment.get("implementation"), "governance implementation binding"
    )
    source_sha256 = _digest(
        implementation.get("admission_tool_file_sha256"),
        "bound admission tool file SHA-256",
    )
    return _agent_actor(
        actor_id="codex-fixture-admission-implementer",
        display_name="Codex (fixture admission implementation agent)",
        role=role,
        source_file_sha256=source_sha256,
    )


def load_sealed_custodian_actor(
    path: Path, *, parent_independent_actor: dict[str, Any]
) -> tuple[dict[str, str], str]:
    """Load the exact non-human custodian identity selected before ratification."""

    actor, file_sha256 = _json(path, "sealed custodian actor", canonical=True)
    actor = krea_fixture._agent_actor(actor, "sealed custodian actor")
    parent = krea_fixture._agent_actor(
        parent_independent_actor, "parent independent review actor"
    )
    krea_fixture._validate_cross_agent_distinct(actor, parent)
    return actor, file_sha256


def load_policy(path: Path = _POLICY_PATH) -> dict[str, Any]:
    policy, _ = _json(path, "governance policy", canonical=True)
    _exact(
        policy,
        {
            "schema",
            "kind",
            "mode",
            "accountable_owner_identity",
            "agent_evidence_may_support_owner_ratification",
            "agent_review_is_not_human_review",
            "independent_human_review_performed",
            "owner_ratification_required",
            "legacy_named_human_contract_unchanged",
            "identity_assurance",
            "authorization_sequence",
            "limitations",
            "policy_sha256",
        },
        "governance policy",
    )
    body = {key: value for key, value in policy.items() if key != "policy_sha256"}
    if (
        policy["schema"] != 1
        or policy["kind"] != _POLICY_KIND
        or policy["mode"] != _MODE
        or policy["accountable_owner_identity"] != _OWNER
        or policy["agent_evidence_may_support_owner_ratification"] is not True
        or policy["agent_review_is_not_human_review"] is not True
        or policy["independent_human_review_performed"] is not False
        or policy["owner_ratification_required"] is not True
        or policy["legacy_named_human_contract_unchanged"] is not True
        or policy["identity_assurance"] != _OWNER_ASSURANCE
        or policy["authorization_sequence"]
        != [
            "agent_technical_evidence",
            "interactive_sole_human_owner_ratification",
            "fixture_admission_envelope",
            "separate_gpu_execution_approval",
        ]
        or not isinstance(policy["limitations"], list)
        or len(policy["limitations"]) != 4
        or policy["policy_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("governance policy differs from the explicit amendment")
    krea_fixture.named_human(policy["accountable_owner_identity"], "owner identity")
    return policy


def _package_inputs(package_root: Path) -> dict[str, Any]:
    root = _safe_directory(package_root, "fixture package")
    package = krea_fixture_package.validate_package(root)
    package_path = root / "package-manifest.json"
    review_path = root / "bundled-review.request.json"
    review, review_file_sha = _json(review_path, "review request", canonical=True)
    candidates = {}
    for role in _ROLES:
        path = root / role / "fixture-manifest.candidate.json"
        candidate, file_sha = _json(path, f"{role} candidate", canonical=True)
        candidates[role] = {
            "path": path,
            "document": candidate,
            "file_sha256": file_sha,
        }
    return {
        "root": root,
        "package": package,
        "package_manifest_file_sha256": _file_sha256(package_path),
        "review_request_file_sha256": review_file_sha,
        "review_request": review,
        "candidates": candidates,
    }


def _ledger_bindings(inputs: dict[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for role in _ROLES:
        bindings = inputs["candidates"][role]["document"]["bindings"]
        for surface, candidate_key in (
            ("rights", "rights_ledger"),
            ("captions", "caption_ledger"),
            ("similarity", "similarity_evidence"),
        ):
            source = bindings[candidate_key]
            result[f"{role}/{surface}"] = {
                "relative_path": source["relative_path"],
                "file_sha256": source["file_sha256"],
                "semantic_sha256": source["semantic_sha256"],
            }
    return result


def _validate_original_records(
    inputs: dict[str, Any], surface_path: Path, independent_path: Path
) -> dict[str, Any]:
    surface, surface_file_sha = _json(
        surface_path, "original surface-agent countersign", canonical=False
    )
    independent, independent_file_sha = _json(
        independent_path, "original independent-agent review", canonical=False
    )
    if surface_file_sha != _SURFACE_SOURCE_FILE_SHA256:
        raise ValueError("surface-agent source record is not the approved exact bytes")
    if independent_file_sha != _INDEPENDENT_SOURCE_FILE_SHA256:
        raise ValueError(
            "independent-agent source record is not the approved exact bytes"
        )
    _exact(
        surface,
        {
            "schema",
            "kind",
            "countersigner",
            "signed_at_utc",
            "package_sha256",
            "file_set_sha256",
            "review_request_sha256",
            "candidate_manifest_sha256s",
            "ledger_file_sha256s",
            "verified",
            "governance_note",
            "scope_limit",
            "fixture_package_v1",
        },
        "original surface-agent countersign",
    )
    _exact(
        independent,
        {
            "schema",
            "kind",
            "reviewer_identity",
            "reviewer_is_human",
            "authorization_provenance",
            "signed_at_utc",
            "independent_of_countersign",
            "package_sha256",
            "file_set_sha256",
            "review_request_sha256",
            "candidate_manifest_sha256s",
            "ledger_file_sha256s",
            "part_1_independent_rederivation",
            "part_2_d2_committed_selection_rederivation",
            "part_3_c1c4_digest_only_acceptance",
            "blocking_findings",
            "approval",
            "fixture_package_v1",
            "files_edited_in_package",
        },
        "original independent-agent review",
    )
    _reject_true_authorization_flags(surface, "surface-agent countersign")
    _reject_true_authorization_flags(independent, "independent-agent review")
    package = inputs["package"]
    expected_ledgers = {
        key.replace("/rights", "/rights-ledger")
        .replace("/captions", "/caption-ledger")
        .replace("/similarity", "/similarity-evidence"): value["file_sha256"]
        for key, value in _ledger_bindings(inputs).items()
    }
    common = {
        "package_sha256": package["package_sha256"],
        "file_set_sha256": package["file_set_sha256"],
        "review_request_sha256": package["review_request_sha256"],
        "candidate_manifest_sha256s": package["candidate_manifest_sha256s"],
        "ledger_file_sha256s": expected_ledgers,
    }
    if (
        surface.get("schema") != 1
        or surface.get("kind") != "forge-krea-response-engineer-countersign"
        or any(surface.get(key) != value for key, value in common.items())
        or "agent" not in str(surface.get("countersigner", "")).casefold()
        or surface.get("scope_limit")
        != (
            "This countersign covers the v2 candidate package only. It does not "
            "authorize admission, GPU execution, or C1-C4 access; those flip at "
            "their own gates per the admission-envelope spec."
        )
    ):
        raise ValueError("surface-agent countersign does not bind the v2 package")
    _utc(surface.get("signed_at_utc"), "surface countersign time")
    verified = _object(surface.get("verified"), "surface countersign verified")
    required_verified = {
        "similarity_pair_binding",
        "machine_screen_accepted",
        "rights_rows",
        "caption_rows",
        "trigger_contract",
        "no_key_material",
        "implementation_commit_pushed",
        "file_set_integrity",
        "d1_training_zip_sha256_matches_handoff",
    }
    if set(verified) != required_verified:
        raise ValueError("surface-agent countersign verification schema changed")
    if (
        independent.get("schema") != 1
        or independent.get("kind") != "forge-krea-independent-reviewer-approval"
        or independent.get("reviewer_is_human") is not False
        or any(independent.get(key) != value for key, value in common.items())
        or surface_file_sha
        not in str(independent.get("independent_of_countersign", ""))
        or independent.get("part_1_independent_rederivation", {}).get("result")
        != "PASS"
        or independent.get("part_2_d2_committed_selection_rederivation", {}).get(
            "result"
        )
        != "PASS"
        or independent.get("part_3_c1c4_digest_only_acceptance", {}).get("result")
        != "PASS"
        or independent.get("approval", {}).get("admission_authorized") is not False
        or independent.get("approval", {}).get("gpu_execution_authorized") is not False
        or independent.get("approval", {}).get("c1c4_revealed") is not False
    ):
        raise ValueError("independent-agent record does not provide exact PASS x3")
    _utc(independent.get("signed_at_utc"), "independent review time")
    findings = independent.get("blocking_findings")
    if not isinstance(findings, list) or {
        row.get("id") for row in findings if isinstance(row, dict)
    } != {"F1", "F2", "F3"}:
        raise ValueError("independent review findings F1/F2/F3 are incomplete")
    return {
        "surface": surface,
        "surface_file_sha256": surface_file_sha,
        "independent": independent,
        "independent_file_sha256": independent_file_sha,
    }


def _seal(body: dict[str, Any], digest_key: str) -> dict[str, Any]:
    if digest_key in body:
        raise ValueError(f"unsealed record contains {digest_key}")
    return {**body, digest_key: krea_provenance.canonical_sha256(body)}


def build_surface_agent_review(
    inputs: dict[str, Any], originals: dict[str, Any]
) -> dict[str, Any]:
    package = inputs["package"]
    actor = _agent_actor(
        actor_id="claude-fable-5-response-engineer",
        display_name="Claude Fable 5 (response engineer, agent)",
        role="surface_reviewer",
        source_file_sha256=originals["surface_file_sha256"],
    )
    surfaces = []
    ledgers = _ledger_bindings(inputs)
    for role in _ROLES:
        candidate = inputs["candidates"][role]["document"]
        for surface in _SURFACES:
            ledger = ledgers[f"{role}/{surface}"]
            assertions: dict[str, Any]
            if surface == "rights":
                assertions = {
                    "every_selected_source_has_a_rights_row": True,
                    "license_and_attribution_obligations_bound": True,
                    "third_party_rights_not_warranted_is_preserved": True,
                }
            elif surface == "captions":
                assertions = {
                    "every_selected_image_has_a_caption_row": True,
                    "agent_reports_caption_image_match": True,
                    "training_captions_rely_on_config_trigger_injection": True,
                    "evaluation_captions_contain_trigger_exactly_once": True,
                }
            else:
                assertions = {
                    "exhaustive_selected_pair_ledger_bound": True,
                    "machine_and_metadata_screen_passed": True,
                    "d1_targeted_pairs_visually_reviewed_by_surface_agent": role
                    == "D1",
                    "independent_agent_did_not_visually_review_d1_targeted_pairs": True,
                }
            surfaces.append(
                {
                    "role": role,
                    "surface": surface,
                    "candidate_manifest_sha256": candidate["candidate_manifest_sha256"],
                    "ledger": ledger,
                    "decision": "accepted_as_agent_technical_evidence",
                    "assertions": assertions,
                }
            )
    body = {
        "schema": 1,
        "kind": _SURFACE_KIND,
        "actor": actor,
        "source_record_file_sha256": originals["surface_file_sha256"],
        "reviewed_at_utc": originals["surface"]["signed_at_utc"],
        "source_package": {
            "package_manifest_file_sha256": inputs["package_manifest_file_sha256"],
            "package_sha256": package["package_sha256"],
            "file_set_sha256": package["file_set_sha256"],
            "review_request_sha256": package["review_request_sha256"],
            "candidate_manifest_sha256s": package["candidate_manifest_sha256s"],
        },
        "surfaces": surfaces,
        "decision": "recommend_owner_ratification",
        "agent_review_is_not_human_review": True,
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": _CLAIM_LIMIT,
    }
    return _seal(body, "review_sha256")


def validate_surface_agent_review(
    record: dict[str, Any], *, inputs: dict[str, Any], originals: dict[str, Any]
) -> dict[str, Any]:
    expected = build_surface_agent_review(inputs, originals)
    if record != expected:
        raise ValueError("surface-agent review is not the canonical derivation")
    return record


def build_independent_agent_review(
    inputs: dict[str, Any],
    originals: dict[str, Any],
    surface_review: dict[str, Any],
    *,
    surface_review_file_sha256: str,
) -> dict[str, Any]:
    package = inputs["package"]
    source = originals["independent"]
    part2 = source["part_2_d2_committed_selection_rederivation"]
    part3 = source["part_3_c1c4_digest_only_acceptance"]
    actor = _agent_actor(
        actor_id="opus-5-independent-reviewer",
        display_name="Opus 5 reviewer (independent agent of record)",
        role="independent_technical_reviewer",
        source_file_sha256=originals["independent_file_sha256"],
    )
    body = {
        "schema": 1,
        "kind": _INDEPENDENT_KIND,
        "actor": actor,
        "source_record_file_sha256": originals["independent_file_sha256"],
        "reviewed_at_utc": source["signed_at_utc"],
        "source_package": {
            "package_manifest_file_sha256": inputs["package_manifest_file_sha256"],
            "package_sha256": package["package_sha256"],
            "file_set_sha256": package["file_set_sha256"],
            "review_request_sha256": package["review_request_sha256"],
            "candidate_manifest_sha256s": package["candidate_manifest_sha256s"],
        },
        "surface_agent_review": {
            "file_sha256": _digest(
                surface_review_file_sha256, "surface review file SHA-256"
            ),
            "review_sha256": surface_review["review_sha256"],
            "actor": surface_review["actor"],
        },
        "technical_results": {
            "package_and_file_set_rederived": True,
            "candidate_and_ledger_bindings_rederived": True,
            "d2_committed_selector_opened_and_exactly_rederived": True,
            "d2_commitment_sha256": part2["commitment_sha256"],
            "d2_split_semantic_sha256": inputs["candidates"]["D2"]["document"][
                "bindings"
            ]["source_split"]["semantic_sha256"],
            "c1c4_commitment_sha256": part3["commitment"],
            "c1c4_manifest_file_sha256s": part3["on_box_manifest_digests_unchanged"]
            | {},
            "c1c4_revealed": False,
            "c1c4_semantic_manifest_mapping_supplied": False,
            "all_six_cross_review_binding_supplied": False,
            "d1_targeted_pairs_visually_reviewed_by_this_actor": False,
        },
        "decision": "technical_pass_recommend_owner_ratification",
        "agent_review_is_not_human_review": True,
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": _CLAIM_LIMIT,
    }
    # The source mapping includes a summary boolean which is not a fixture id.
    body["technical_results"]["c1c4_manifest_file_sha256s"].pop(
        "all_four_match_published", None
    )
    return _seal(body, "review_sha256")


def validate_independent_agent_review(
    record: dict[str, Any],
    *,
    inputs: dict[str, Any],
    originals: dict[str, Any],
    surface_review: dict[str, Any],
    surface_review_file_sha256: str,
) -> dict[str, Any]:
    expected = build_independent_agent_review(
        inputs,
        originals,
        surface_review,
        surface_review_file_sha256=surface_review_file_sha256,
    )
    if record != expected:
        raise ValueError("independent-agent review is not the canonical derivation")
    if record["actor"]["actor_id"] == surface_review["actor"]["actor_id"]:
        raise ValueError("independent and surface agent identities are not distinct")
    return record


def _gpu_gate_implementation_bindings() -> dict[str, str]:
    """Bind every local module that can shape Week-5 GPU authorization.

    This is deliberately an exact, non-optional surface.  Adding, removing, or
    changing any bound module invalidates the governance amendment and therefore
    requires a fresh owner ratification before materialization or GPU approval.
    """

    root = Path(__file__).resolve().parents[2]
    return {
        key: _file_sha256(_safe_file(root / relative, key))
        for key, relative in _GPU_GATE_IMPLEMENTATION_PATHS.items()
    }


def _discovery_public_evidence_bindings() -> dict[str, Any]:
    """Load the sole public-source evidence authority from the frozen plan."""

    plan_path = _safe_file(_DISCOVERY_PLAN_PATH, "Krea discovery plan")
    try:
        plan = json.loads(plan_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Krea discovery plan is not JSON") from exc
    day0 = _object(plan.get("day0_source_evidence"), "Day-0 source evidence")
    evidence = _object(day0.get("evidence_bindings"), "public evidence bindings")
    _exact(
        evidence,
        {"thin_manifest_file_sha256", "public_source_provenance"},
        "public evidence bindings",
    )
    thin_sha = _digest(
        evidence["thin_manifest_file_sha256"],
        "thin evidence manifest file SHA-256",
    )
    sources = _object(
        evidence["public_source_provenance"], "public source provenance bindings"
    )
    _exact(sources, {"K2", "K3", "K4"}, "public source provenance bindings")
    normalized_sources: dict[str, dict[str, str]] = {}
    for arm, binding_value in sorted(sources.items()):
        binding = _object(binding_value, f"{arm} public source binding")
        _exact(
            binding,
            {"file_sha256", "manifest_sha256"},
            f"{arm} public source binding",
        )
        normalized_sources[arm] = {
            "file_sha256": _digest(
                binding["file_sha256"], f"{arm} provenance file SHA-256"
            ),
            "manifest_sha256": _digest(
                binding["manifest_sha256"], f"{arm} manifest semantic SHA-256"
            ),
        }
    return {
        "discovery_plan_file_sha256": _file_sha256(plan_path),
        "thin_manifest_file_sha256": thin_sha,
        "public_source_provenance": normalized_sources,
    }


def _sanitized_forge_git(root: Path, *arguments: str) -> bytes:
    """Run a read-only Git builtin without user/system config or hooks."""

    executable = _safe_file(Path("/usr/bin/git"), "system Git executable")
    environment = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    completed = subprocess.run(
        [
            str(executable),
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.autocrlf=false",
            "-C",
            str(root),
            *arguments,
        ],
        check=True,
        capture_output=True,
        timeout=30,
        env=environment,
    )
    return completed.stdout


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(
        header + raw, usedforsecurity=False
    ).hexdigest()  # Git object identity, not a security digest.


def _forge_repository_identity(root: Path | None = None) -> dict[str, Any]:
    """Bind a clean worktree byte-for-byte to its exact Git commit tree."""

    repository = _safe_directory(
        Path(__file__).resolve().parents[2] if root is None else root,
        "Forge repository",
    )
    try:
        top = _sanitized_forge_git(repository, "rev-parse", "--show-toplevel")
        commit_raw = _sanitized_forge_git(
            repository, "rev-parse", "--verify", "HEAD^{commit}"
        )
        tree_raw = _sanitized_forge_git(
            repository, "rev-parse", "--verify", "HEAD^{tree}"
        )
        status = _sanitized_forge_git(
            repository,
            "status",
            "--porcelain=v2",
            "-z",
            "--untracked-files=all",
        )
        cache_tags = _sanitized_forge_git(repository, "ls-files", "-v", "-z")
        tree_rows = _sanitized_forge_git(
            repository, "ls-tree", "-r", "-z", "--full-tree", "HEAD"
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Forge Git identity could not be read safely") from exc
    try:
        top_path = Path(top.rstrip(b"\n").decode("utf-8"))
        commit = commit_raw.strip().decode("ascii")
        tree = tree_raw.strip().decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Forge Git identity is not portable UTF-8/ASCII") from exc
    if top_path.resolve() != repository.resolve():
        raise ValueError("Forge repository root differs from the governed root")
    if not _GIT_SHA.fullmatch(commit) or not _GIT_SHA.fullmatch(tree):
        raise ValueError("Forge commit or tree identity is invalid")
    if status:
        raise ValueError("Forge worktree must be clean, including untracked files")

    indexed_paths: set[str] = set()
    for row in cache_tags.split(b"\0"):
        if not row:
            continue
        if not row.startswith(b"H "):
            raise ValueError(
                "Forge index contains skip-worktree, assume-unchanged, or "
                "non-canonical cache state"
            )
        try:
            indexed_paths.add(row[2:].decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError("Forge tracked path is not UTF-8") from exc

    tracked_files: list[dict[str, Any]] = []
    tree_paths: set[str] = set()
    for row in tree_rows.split(b"\0"):
        if not row:
            continue
        try:
            metadata, raw_path = row.split(b"\t", 1)
            mode, object_type, blob_sha1 = metadata.decode("ascii").split(" ")
            portable = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("Forge commit tree row is malformed") from exc
        relative = PurePosixPath(portable)
        if (
            relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or object_type != "blob"
            or mode not in {"100644", "100755", "120000"}
            or not _GIT_SHA.fullmatch(blob_sha1)
        ):
            raise ValueError("Forge commit tree contains unsupported content")
        worktree_path = repository.joinpath(*relative.parts)
        if mode == "120000":
            if not worktree_path.is_symlink():
                raise ValueError("Forge worktree symlink differs from commit tree")
            raw = os.fsencode(os.readlink(worktree_path))
        else:
            if worktree_path.is_symlink() or not worktree_path.is_file():
                raise ValueError("Forge tracked file is missing or not regular")
            file_mode = worktree_path.stat().st_mode
            executable = bool(file_mode & stat.S_IXUSR)
            if executable != (mode == "100755"):
                raise ValueError("Forge tracked executable mode differs from tree")
            raw = worktree_path.read_bytes()
        if _git_blob_sha1(raw) != blob_sha1:
            raise ValueError("Forge tracked file bytes differ from commit tree")
        tree_paths.add(portable)
        tracked_files.append(
            {
                "path": portable,
                "mode": mode,
                "git_blob_sha1": blob_sha1,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        )
    if indexed_paths != tree_paths:
        raise ValueError("Forge index path set differs from the commit tree")
    tracked_files.sort(key=lambda row: row["path"])
    return {
        "commit_sha1": commit,
        "tree_sha1": tree,
        "tracked_files": tracked_files,
        "tracked_file_manifest_sha256": krea_provenance.canonical_sha256(tracked_files),
    }


def _forge_repository_identity_at_commit(
    repository: Path, commit_sha1: str
) -> dict[str, Any]:
    """Reconstruct an ancestor identity from immutable Git objects.

    Unlike :func:`_forge_repository_identity`, this never projects ancestor
    bytes into the live worktree.  Every tracked byte is read by object id from
    the current repository's object database and is checked against Git's blob
    identity before its SHA-256 is admitted.
    """

    repository = _safe_directory(repository, "Forge repository")
    commit_sha1 = _text(commit_sha1, "historical Forge commit").lower()
    if not _GIT_SHA.fullmatch(commit_sha1):
        raise ValueError("historical Forge commit is invalid")
    try:
        resolved = _sanitized_forge_git(
            repository, "rev-parse", "--verify", f"{commit_sha1}^{{commit}}"
        ).strip().decode("ascii")
        tree = _sanitized_forge_git(
            repository, "rev-parse", "--verify", f"{commit_sha1}^{{tree}}"
        ).strip().decode("ascii")
        tree_rows = _sanitized_forge_git(
            repository,
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            commit_sha1,
        )
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError) as exc:
        raise ValueError("historical Forge Git identity could not be read") from exc
    if resolved != commit_sha1 or not _GIT_SHA.fullmatch(tree):
        raise ValueError("historical Forge commit or tree identity is invalid")
    tracked_files: list[dict[str, Any]] = []
    for row in tree_rows.split(b"\0"):
        if not row:
            continue
        try:
            metadata, raw_path = row.split(b"\t", 1)
            mode, object_type, blob_sha1 = metadata.decode("ascii").split(" ")
            portable = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("historical Forge tree row is malformed") from exc
        relative = PurePosixPath(portable)
        if (
            relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or object_type != "blob"
            or mode not in {"100644", "100755", "120000"}
            or not _GIT_SHA.fullmatch(blob_sha1)
        ):
            raise ValueError("historical Forge tree contains unsupported content")
        try:
            raw = _sanitized_forge_git(repository, "cat-file", "blob", blob_sha1)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("historical Forge blob could not be read") from exc
        if _git_blob_sha1(raw) != blob_sha1:
            raise ValueError("historical Forge blob identity mismatch")
        tracked_files.append(
            {
                "path": portable,
                "mode": mode,
                "git_blob_sha1": blob_sha1,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        )
    tracked_files.sort(key=lambda item: item["path"])
    return {
        "commit_sha1": commit_sha1,
        "tree_sha1": tree,
        "tracked_files": tracked_files,
        "tracked_file_manifest_sha256": krea_provenance.canonical_sha256(
            tracked_files
        ),
    }


def _implementation_bindings_from_repository_identity(
    identity: Mapping[str, Any],
) -> dict[str, str]:
    rows = identity.get("tracked_files")
    if not isinstance(rows, list):
        raise ValueError("historical repository identity lacks tracked files")
    by_path: dict[str, dict[str, Any]] = {}
    for row_value in rows:
        row = _object(row_value, "historical tracked file")
        portable = row.get("path")
        if not isinstance(portable, str) or portable in by_path:
            raise ValueError("historical repository paths are invalid or duplicated")
        by_path[portable] = row
    result: dict[str, str] = {}
    for key, portable in _GPU_GATE_IMPLEMENTATION_PATHS.items():
        row = by_path.get(portable)
        if row is None or row.get("mode") not in {"100644", "100755"}:
            raise ValueError(f"historical implementation file is absent: {portable}")
        result[key] = _digest(row.get("sha256"), f"historical {portable} SHA-256")
    return result


def _require_fc70_successor(repository: Path, current_commit: str) -> None:
    """Constrain the compatibility path to fc70 plus this bridge surface."""

    try:
        _sanitized_forge_git(
            repository,
            "merge-base",
            "--is-ancestor",
            _SUCCESSOR_ADMISSION_COMMIT,
            _SUCCESSOR_RUNTIME_BASE_COMMIT,
        )
        _sanitized_forge_git(
            repository,
            "merge-base",
            "--is-ancestor",
            _SUCCESSOR_RUNTIME_BASE_COMMIT,
            current_commit,
        )
        changed_raw = _sanitized_forge_git(
            repository,
            "diff",
            "--name-only",
            "-z",
            _SUCCESSOR_RUNTIME_BASE_COMMIT,
            current_commit,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("current Forge runtime is not an fc70 successor") from exc
    try:
        changed = [item.decode("utf-8") for item in changed_raw.split(b"\0") if item]
    except UnicodeDecodeError as exc:
        raise ValueError("successor Forge path is not UTF-8") from exc
    if any(
        not (
            path in _SUCCESSOR_ALLOWED_RUNTIME_PATHS
            or path.startswith("campaign_tools/")
            or path.startswith("tests/")
        )
        for path in changed
    ):
        raise ValueError("fc70 successor changed an unauthorized runtime path")


def build_governance_amendment(
    inputs: dict[str, Any],
    originals: dict[str, Any],
    policy: dict[str, Any],
    surface_review: dict[str, Any],
    independent_review: dict[str, Any],
    *,
    sealed_custodian_actor: dict[str, Any],
    sealed_custodian_actor_file_sha256: str,
    surface_review_file_sha256: str,
    independent_review_file_sha256: str,
    amended_at_utc: str,
) -> dict[str, Any]:
    package = inputs["package"]
    custodian = krea_fixture._agent_actor(
        sealed_custodian_actor, "sealed custodian actor"
    )
    krea_fixture._validate_cross_agent_distinct(custodian, independent_review["actor"])
    prior_actors = [surface_review["actor"], independent_review["actor"], custodian]
    delegated_actors = list(krea_delegated_review_contract.load()["actors"].values())
    prior_ids = {actor["actor_id"] for actor in prior_actors}
    prior_instances = {actor["review_instance_id"] for actor in prior_actors}
    if any(
        actor["actor_id"] in prior_ids or actor["review_instance_id"] in prior_instances
        for actor in delegated_actors
    ):
        raise ValueError("delegated Stage-1 actor reuses prior review identity")
    body = {
        "schema": 1,
        "kind": _AMENDMENT_KIND,
        "amended_at_utc": _utc(amended_at_utc, "amendment time"),
        "mode": _MODE,
        "governance_policy": {
            "file_sha256": _file_sha256(_POLICY_PATH),
            "policy_sha256": policy["policy_sha256"],
        },
        "implementation": _gpu_gate_implementation_bindings(),
        "stage1_delegated_agent_review_contract": (
            krea_delegated_review_contract.binding()
        ),
        "forge_repository_identity": _forge_repository_identity(),
        "public_source_evidence": _discovery_public_evidence_bindings(),
        "source_package": {
            "package_manifest_file_sha256": inputs["package_manifest_file_sha256"],
            "package_sha256": package["package_sha256"],
            "file_set_sha256": package["file_set_sha256"],
            "review_request_file_sha256": inputs["review_request_file_sha256"],
            "review_request_sha256": package["review_request_sha256"],
            "candidate_manifest_sha256s": package["candidate_manifest_sha256s"],
            "ledger_bindings": _ledger_bindings(inputs),
        },
        "original_agent_records": {
            "surface_countersign_file_sha256": originals["surface_file_sha256"],
            "independent_review_file_sha256": originals["independent_file_sha256"],
            "both_actors_are_agents": True,
            "neither_record_is_owner_ratification": True,
        },
        "canonical_agent_evidence": {
            "surface_review": {
                "file_sha256": _digest(
                    surface_review_file_sha256, "surface review file SHA-256"
                ),
                "review_sha256": surface_review["review_sha256"],
                "actor": surface_review["actor"],
            },
            "independent_review": {
                "file_sha256": _digest(
                    independent_review_file_sha256,
                    "independent review file SHA-256",
                ),
                "review_sha256": independent_review["review_sha256"],
                "actor": independent_review["actor"],
            },
        },
        "sealed_custodian_actor": {
            "file_sha256": _digest(
                sealed_custodian_actor_file_sha256,
                "sealed custodian actor file SHA-256",
            ),
            "actor_sha256": krea_provenance.canonical_sha256(custodian),
            "actor": custodian,
        },
        "findings_resolution": {
            "F1": (
                "superseded by explicit agent-review evidence plus sole-human "
                "owner ratification; no named-human surface review is claimed"
            ),
            "F2": (
                "implemented by additive schema-2 governance; legacy schema-1 "
                "named-human validation remains unchanged"
            ),
            "F3": (
                "preserved: the independent agent did not visually review the "
                "15 D1 targeted pairs; owner may ratify reliance on the surface "
                "agent's explicitly bound review without claiming personal review"
            ),
        },
        "record_corrections": {
            "caption_trigger_contract": (
                "training captions omit the trigger because ai-toolkit injects it "
                "from config; evaluation captions contain it exactly once"
            ),
            "c1c4_discovery_scope": (
                "discovery binds published MANIFEST file hashes and aggregate "
                "commitment only; semantic C manifests and all-six cross-review "
                "remain mandatory confirmation inputs and are not claimed here"
            ),
        },
        "accountable_owner_identity": policy["accountable_owner_identity"],
        "owner_ratification_required": True,
        "agent_review_is_not_human_review": True,
        "independent_human_review_performed": False,
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "c1c4_revealed": False,
        "claim_limit": _CLAIM_LIMIT,
    }
    return _seal(body, "amendment_sha256")


def _successor_bound_governance_amendment(
    amendment: Mapping[str, Any], current_expected: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Rebuild the one admitted 588 amendment inside an exact fc70 successor.

    The owner-ratified artifact must keep naming the code and complete Git tree
    that actually produced it.  A successor therefore may not re-seal or
    rewrite those fields to its own bytes.  Instead, this path proves the
    historical commit/tree and every governed implementation file directly
    from immutable Git objects, constrains the live clean checkout to fc70 plus
    this bridge, and then performs the ordinary canonical derivation with only
    those two historical bindings preserved.
    """

    bound_identity = amendment.get("forge_repository_identity")
    if not isinstance(bound_identity, dict):
        return None
    if bound_identity.get("commit_sha1") != _SUCCESSOR_ADMISSION_COMMIT:
        return None
    repository = Path(__file__).resolve().parents[2]
    current_identity = _object(
        current_expected.get("forge_repository_identity"),
        "current Forge repository identity",
    )
    current_commit = current_identity.get("commit_sha1")
    if not isinstance(current_commit, str) or not _GIT_SHA.fullmatch(current_commit):
        raise ValueError("current Forge repository identity is invalid")
    _require_fc70_successor(repository, current_commit)
    historical_identity = _forge_repository_identity_at_commit(
        repository, _SUCCESSOR_ADMISSION_COMMIT
    )
    if (
        historical_identity["tree_sha1"] != _SUCCESSOR_ADMISSION_TREE
        or bound_identity != historical_identity
    ):
        raise ValueError("historical admission repository binding drifted")
    historical_implementation = _implementation_bindings_from_repository_identity(
        historical_identity
    )
    if amendment.get("implementation") != historical_implementation:
        raise ValueError("historical admission implementation binding drifted")
    body = {
        key: value
        for key, value in current_expected.items()
        if key != "amendment_sha256"
    }
    body["implementation"] = historical_implementation
    body["forge_repository_identity"] = historical_identity
    return _seal(body, "amendment_sha256")


def validate_governance_amendment(
    amendment: dict[str, Any],
    *,
    inputs: dict[str, Any],
    originals: dict[str, Any],
    policy: dict[str, Any],
    surface_review: dict[str, Any],
    independent_review: dict[str, Any],
    sealed_custodian_actor: dict[str, Any],
    sealed_custodian_actor_file_sha256: str,
    surface_review_file_sha256: str,
    independent_review_file_sha256: str,
) -> dict[str, Any]:
    expected = build_governance_amendment(
        inputs,
        originals,
        policy,
        surface_review,
        independent_review,
        sealed_custodian_actor=sealed_custodian_actor,
        sealed_custodian_actor_file_sha256=sealed_custodian_actor_file_sha256,
        surface_review_file_sha256=surface_review_file_sha256,
        independent_review_file_sha256=independent_review_file_sha256,
        amended_at_utc=amendment.get("amended_at_utc"),
    )
    if amendment != expected:
        successor_expected = _successor_bound_governance_amendment(
            amendment, expected
        )
        if successor_expected is None or amendment != successor_expected:
            raise ValueError("governance amendment is not the canonical derivation")
    return amendment


def _ratification_decision_bindings(
    *,
    inputs: dict[str, Any],
    amendment: dict[str, Any],
    evaluator_contract: dict[str, Any],
) -> dict[str, Any]:
    package = inputs["package"]
    return {
        "package_sha256": package["package_sha256"],
        "file_set_sha256": package["file_set_sha256"],
        "candidate_manifest_sha256s": package["candidate_manifest_sha256s"],
        "governance_policy_sha256": amendment["governance_policy"]["policy_sha256"],
        "governance_amendment_sha256": amendment["amendment_sha256"],
        "forge_repository_identity": amendment["forge_repository_identity"],
        "public_source_evidence": amendment["public_source_evidence"],
        "surface_agent_review_sha256": amendment["canonical_agent_evidence"][
            "surface_review"
        ]["review_sha256"],
        "independent_agent_review_sha256": amendment["canonical_agent_evidence"][
            "independent_review"
        ]["review_sha256"],
        "sealed_custodian_actor_sha256": amendment["sealed_custodian_actor"][
            "actor_sha256"
        ],
        "god_evaluator_contract_sha256": evaluator_contract["contract_sha256"],
        "god_commit": evaluator_contract["commit"],
        "stage1_execution_surface_policy": {
            "file_sha256": amendment["implementation"][
                "execution_surface_policy_file_sha256"
            ],
            "policy_sha256": krea_execution_surface_policy.POLICY["policy_sha256"],
            "execution_surface": "staged_host_venv",
            "execution_scope": "discovery_only",
        },
        "stage1_delegated_agent_review_contract": (
            amendment["stage1_delegated_agent_review_contract"]
        ),
        "stage2_requirement": krea_execution_surface_policy.POLICY[
            "stage2_requirement"
        ],
    }


def _draft_acknowledgements() -> dict[str, bool]:
    return {
        "agents_are_not_humans": True,
        "no_independent_human_review_occurred": True,
        "owner_does_not_claim_personal_pair_review": True,
        "ratification_is_not_a_cryptographic_or_legal_signature": True,
        "admission_and_gpu_authorization_are_still_separate": True,
        "owner_authorizes_mechanical_gpu_approval_after_envelope_and_host_plan_validation": True,
        "c1c4_remain_sealed": True,
        "stage1_is_discovery_only": True,
        "stage1_is_not_release_or_tournament_evidence": True,
        "stage2_requires_separate_commit_and_fresh_owner_ratification": True,
        "owner_accepts_exact_stage1_timing_margins": True,
        "owner_authorizes_only_prebound_stage1_technical_agents": True,
        "delegated_agents_cannot_change_frozen_rules": True,
    }


def build_portable_ratification_draft(
    *,
    inputs: dict[str, Any],
    originals: dict[str, Any],
    policy: dict[str, Any],
    surface_review: dict[str, Any],
    surface_review_file_sha256: str,
    independent_review: dict[str, Any],
    independent_review_file_sha256: str,
    sealed_custodian_actor: dict[str, Any],
    sealed_custodian_actor_file_sha256: str,
    amendment: dict[str, Any],
    amendment_file_sha256: str,
    evaluator_contract: dict[str, Any],
    evaluator_contract_file_sha256: str,
    prepared_at_utc: str,
) -> dict[str, Any]:
    package = inputs["package"]
    custodian = krea_fixture._agent_actor(
        sealed_custodian_actor, "sealed custodian actor"
    )
    krea_fixture._validate_cross_agent_distinct(custodian, independent_review["actor"])
    if amendment["sealed_custodian_actor"] != {
        "file_sha256": _digest(
            sealed_custodian_actor_file_sha256,
            "sealed custodian actor file SHA-256",
        ),
        "actor_sha256": krea_provenance.canonical_sha256(custodian),
        "actor": custodian,
    }:
        raise ValueError("portable draft custodian differs from the amendment")
    body = {
        "schema": 1,
        "kind": _PORTABLE_DRAFT_KIND,
        "prepared_at_utc": _utc(prepared_at_utc, "portable draft time"),
        "owner_identity": _OWNER,
        "decision_bindings": _ratification_decision_bindings(
            inputs=inputs,
            amendment=amendment,
            evaluator_contract=evaluator_contract,
        ),
        "evidence_files": {
            "package_manifest": {
                "file_sha256": inputs["package_manifest_file_sha256"],
                "package_sha256": package["package_sha256"],
            },
            "surface_source_record_file_sha256": originals["surface_file_sha256"],
            "independent_source_record_file_sha256": originals[
                "independent_file_sha256"
            ],
            "governance_policy": {
                "file_sha256": _file_sha256(_POLICY_PATH),
                "policy_sha256": policy["policy_sha256"],
            },
            "surface_agent_review": {
                "file_sha256": _digest(
                    surface_review_file_sha256, "surface review file SHA-256"
                ),
                "review_sha256": surface_review["review_sha256"],
            },
            "independent_agent_review": {
                "file_sha256": _digest(
                    independent_review_file_sha256,
                    "independent review file SHA-256",
                ),
                "review_sha256": independent_review["review_sha256"],
            },
            "sealed_custodian_actor": {
                "file_sha256": _digest(
                    sealed_custodian_actor_file_sha256,
                    "sealed custodian actor file SHA-256",
                ),
                "actor_sha256": krea_provenance.canonical_sha256(custodian),
            },
            "governance_amendment": {
                "file_sha256": _digest(amendment_file_sha256, "amendment file SHA-256"),
                "amendment_sha256": amendment["amendment_sha256"],
            },
            "god_evaluator_contract": {
                "file_sha256": _digest(
                    evaluator_contract_file_sha256,
                    "evaluator contract file SHA-256",
                ),
                "contract_sha256": evaluator_contract["contract_sha256"],
            },
        },
        "required_phrase": (
            f"RATIFY {package['package_sha256']} AS SOLE HUMAN; "
            "AGENT REVIEW IS NOT HUMAN REVIEW"
        ),
        "acknowledgements": _draft_acknowledgements(),
        "admission_authorized": False,
        "gpu_execution_authorized": False,
    }
    return _seal(body, "draft_sha256")


def validate_portable_ratification_draft(
    draft: dict[str, Any],
    **bindings: Any,
) -> dict[str, Any]:
    expected = build_portable_ratification_draft(
        **bindings,
        prepared_at_utc=draft.get("prepared_at_utc"),
    )
    if draft != expected:
        raise ValueError("portable ratification draft is not the canonical evidence")
    return draft


def build_ratification_draft(
    *,
    inputs: dict[str, Any],
    surface_record_path: Path,
    independent_record_path: Path,
    surface_review_path: Path,
    independent_review_path: Path,
    sealed_custodian_actor_path: Path,
    amendment_path: Path,
    evaluator_contract_path: Path,
    portable_draft_path: Path,
    amendment: dict[str, Any],
    evaluator_contract: dict[str, Any],
    portable_draft: dict[str, Any],
    prepared_at_utc: str,
) -> dict[str, Any]:
    package = inputs["package"]
    body = {
        "schema": 1,
        "kind": _DRAFT_KIND,
        "prepared_at_utc": _utc(prepared_at_utc, "draft time"),
        "owner_identity": _OWNER,
        "inputs": {
            "package_root": str(inputs["root"]),
            "surface_source_record": str(
                _safe_file(surface_record_path, "surface source record")
            ),
            "independent_source_record": str(
                _safe_file(independent_record_path, "independent source record")
            ),
            "surface_agent_review": str(
                _safe_file(surface_review_path, "surface agent review")
            ),
            "independent_agent_review": str(
                _safe_file(independent_review_path, "independent agent review")
            ),
            "sealed_custodian_actor": str(
                _safe_file(sealed_custodian_actor_path, "sealed custodian actor")
            ),
            "governance_amendment": str(
                _safe_file(amendment_path, "governance amendment")
            ),
            "governance_policy": str(_safe_file(_POLICY_PATH, "governance policy")),
            "god_evaluator_contract": str(
                _safe_file(evaluator_contract_path, "G.O.D evaluator contract")
            ),
            "god_evaluator_image_io": str(
                _safe_file(
                    evaluator_contract_path.parent / "image_io.py",
                    "G.O.D evaluator image_io",
                )
            ),
            "god_evaluator_constants": str(
                _safe_file(
                    evaluator_contract_path.parent / "dataset_constants.py",
                    "G.O.D evaluator constants",
                )
            ),
            "portable_ratification_draft": str(
                _safe_file(portable_draft_path, "portable ratification draft")
            ),
        },
        "portable_ratification_draft": {
            "file_sha256": _file_sha256(portable_draft_path),
            "draft_sha256": portable_draft["draft_sha256"],
        },
        "decision_bindings": portable_draft["decision_bindings"],
        "required_phrase": (
            f"RATIFY {package['package_sha256']} AS SOLE HUMAN; "
            "AGENT REVIEW IS NOT HUMAN REVIEW"
        ),
        "acknowledgements": _draft_acknowledgements(),
        "admission_authorized": False,
        "gpu_execution_authorized": False,
    }
    return _seal(body, "draft_sha256")


def _resolve_draft(draft_path: Path) -> dict[str, Any]:
    draft, draft_file_sha = _json(draft_path, "ratification draft", canonical=True)
    if draft.get("schema") != 1 or draft.get("kind") != _DRAFT_KIND:
        raise ValueError("unsupported ratification draft")
    body = {key: value for key, value in draft.items() if key != "draft_sha256"}
    if draft.get("draft_sha256") != krea_provenance.canonical_sha256(body):
        raise ValueError("ratification draft digest mismatch")
    paths = _object(draft.get("inputs"), "ratification draft inputs")
    inputs = _package_inputs(Path(paths["package_root"]))
    originals = _validate_original_records(
        inputs,
        Path(paths["surface_source_record"]),
        Path(paths["independent_source_record"]),
    )
    surface_review, surface_file_sha = _json(
        Path(paths["surface_agent_review"]), "surface agent review", canonical=True
    )
    validate_surface_agent_review(surface_review, inputs=inputs, originals=originals)
    independent_review, independent_file_sha = _json(
        Path(paths["independent_agent_review"]),
        "independent agent review",
        canonical=True,
    )
    validate_independent_agent_review(
        independent_review,
        inputs=inputs,
        originals=originals,
        surface_review=surface_review,
        surface_review_file_sha256=surface_file_sha,
    )
    sealed_custodian_actor_path = _safe_file(
        Path(paths["sealed_custodian_actor"]), "sealed custodian actor"
    )
    sealed_custodian_actor, sealed_custodian_actor_file_sha = (
        load_sealed_custodian_actor(
            sealed_custodian_actor_path,
            parent_independent_actor=independent_review["actor"],
        )
    )
    policy = load_policy(Path(paths["governance_policy"]))
    amendment, amendment_file_sha = _json(
        Path(paths["governance_amendment"]), "governance amendment", canonical=True
    )
    validate_governance_amendment(
        amendment,
        inputs=inputs,
        originals=originals,
        policy=policy,
        surface_review=surface_review,
        independent_review=independent_review,
        sealed_custodian_actor=sealed_custodian_actor,
        sealed_custodian_actor_file_sha256=sealed_custodian_actor_file_sha,
        surface_review_file_sha256=surface_file_sha,
        independent_review_file_sha256=independent_file_sha,
    )
    evaluator_contract_path = Path(paths["god_evaluator_contract"])
    evaluator_root = evaluator_contract_path.parent.parent
    if (
        Path(paths["god_evaluator_image_io"])
        != evaluator_contract_path.parent / "image_io.py"
        or Path(paths["god_evaluator_constants"])
        != evaluator_contract_path.parent / "dataset_constants.py"
    ):
        raise ValueError("ratification draft evaluator paths are not one bundle")
    evaluator_contract, _ = _validate_portable_god_evaluator_contract(evaluator_root)
    portable_draft_path = _safe_file(
        Path(paths["portable_ratification_draft"]), "portable ratification draft"
    )
    portable_draft, portable_draft_file_sha = _json(
        portable_draft_path, "portable ratification draft", canonical=True
    )
    validate_portable_ratification_draft(
        portable_draft,
        inputs=inputs,
        originals=originals,
        policy=policy,
        surface_review=surface_review,
        surface_review_file_sha256=surface_file_sha,
        independent_review=independent_review,
        independent_review_file_sha256=independent_file_sha,
        sealed_custodian_actor=sealed_custodian_actor,
        sealed_custodian_actor_file_sha256=sealed_custodian_actor_file_sha,
        amendment=amendment,
        amendment_file_sha256=amendment_file_sha,
        evaluator_contract=evaluator_contract,
        evaluator_contract_file_sha256=_file_sha256(evaluator_contract_path),
    )
    if portable_draft["prepared_at_utc"] != draft["prepared_at_utc"]:
        raise ValueError("local and portable ratification drafts have different times")
    expected = build_ratification_draft(
        inputs=inputs,
        surface_record_path=Path(paths["surface_source_record"]),
        independent_record_path=Path(paths["independent_source_record"]),
        surface_review_path=Path(paths["surface_agent_review"]),
        independent_review_path=Path(paths["independent_agent_review"]),
        sealed_custodian_actor_path=sealed_custodian_actor_path,
        amendment_path=Path(paths["governance_amendment"]),
        evaluator_contract_path=evaluator_contract_path,
        portable_draft_path=portable_draft_path,
        amendment=amendment,
        evaluator_contract=evaluator_contract,
        portable_draft=portable_draft,
        prepared_at_utc=draft["prepared_at_utc"],
    )
    if draft != expected:
        raise ValueError("ratification draft does not bind the live evidence")
    return {
        "draft": draft,
        "draft_file_sha256": draft_file_sha,
        "inputs": inputs,
        "originals": originals,
        "surface_review": surface_review,
        "surface_review_file_sha256": surface_file_sha,
        "independent_review": independent_review,
        "independent_review_file_sha256": independent_file_sha,
        "sealed_custodian_actor": sealed_custodian_actor,
        "sealed_custodian_actor_path": sealed_custodian_actor_path,
        "sealed_custodian_actor_file_sha256": sealed_custodian_actor_file_sha,
        "policy": policy,
        "amendment": amendment,
        "amendment_file_sha256": amendment_file_sha,
        "evaluator_contract": evaluator_contract,
        "evaluator_contract_path": evaluator_contract_path,
        "portable_draft": portable_draft,
        "portable_draft_path": portable_draft_path,
        "portable_draft_file_sha256": portable_draft_file_sha,
    }


def build_owner_ratification(
    resolved: dict[str, Any], *, ratified_at_utc: str
) -> dict[str, Any]:
    draft = resolved["draft"]
    amendment = resolved["amendment"]
    body = {
        "schema": 1,
        "kind": _RATIFICATION_KIND,
        "owner_identity": _OWNER,
        "owner_identity_assurance": _OWNER_ASSURANCE,
        "ratified_at_utc": _utc(ratified_at_utc, "ratification time"),
        "portable_ratification_draft": {
            "file_sha256": resolved["portable_draft_file_sha256"],
            "draft_sha256": resolved["portable_draft"]["draft_sha256"],
        },
        "governance_amendment": {
            "file_sha256": resolved["amendment_file_sha256"],
            "amendment_sha256": amendment["amendment_sha256"],
        },
        "decision_bindings": draft["decision_bindings"],
        "acknowledgements": {
            "agents_are_not_humans": True,
            "no_independent_human_review_occurred": True,
            "owner_understands_review_scope_and_limitations": True,
            "owner_accepts_accountability_for_using_bound_agent_evidence": True,
            "owner_does_not_claim_personal_pair_review": True,
            "ratification_is_not_a_cryptographic_or_legal_signature": True,
            "owner_authorizes_mechanical_gpu_approval_after_envelope_and_host_plan_validation": True,
            "c1c4_remain_sealed": True,
            "stage1_is_discovery_only": True,
            "stage1_is_not_release_or_tournament_evidence": True,
            "stage2_requires_separate_commit_and_fresh_owner_ratification": True,
            "owner_accepts_exact_stage1_timing_margins": True,
            "owner_authorizes_only_prebound_stage1_technical_agents": True,
            "delegated_agents_cannot_change_frozen_rules": True,
        },
        "decision": "ratified_for_fixture_admission_input",
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": _CLAIM_LIMIT,
    }
    return _seal(body, "ratification_sha256")


def validate_owner_ratification(
    ratification: dict[str, Any], *, resolved: dict[str, Any]
) -> dict[str, Any]:
    expected = build_owner_ratification(
        resolved, ratified_at_utc=ratification.get("ratified_at_utc")
    )
    if ratification != expected:
        raise ValueError("owner ratification does not bind the exact draft")
    ratified = datetime.strptime(
        ratification["ratified_at_utc"], "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)
    prepared = datetime.strptime(
        resolved["draft"]["prepared_at_utc"], "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)
    if ratified < prepared:
        raise ValueError("owner ratification predates its draft")
    return ratification


def prepare_governance(
    *,
    package_root: Path,
    surface_record_path: Path,
    independent_record_path: Path,
    sealed_custodian_actor_path: Path,
    god_checkout: Path,
    god_commit: str,
    output_dir: Path,
    prepared_at_utc: str,
) -> dict[str, Path]:
    output_dir = Path(os.path.abspath(os.path.expanduser(output_dir)))
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, mode=0o700)
    try:
        inputs = _package_inputs(package_root)
        originals = _validate_original_records(
            inputs, surface_record_path, independent_record_path
        )
        policy = load_policy()
        evaluator_contract, evaluator_image_io, evaluator_constants = (
            _build_god_evaluator_contract(god_checkout, expected_commit=god_commit)
        )
        evaluator_dir = output_dir / "evaluator"
        _copy_regular(evaluator_image_io, evaluator_dir / "image_io.py")
        _copy_regular(evaluator_constants, evaluator_dir / "dataset_constants.py")
        evaluator_contract_path = evaluator_dir / "contract.json"
        _write_canonical(evaluator_contract_path, evaluator_contract)
        _validate_portable_god_evaluator_contract(output_dir)
        surface = build_surface_agent_review(inputs, originals)
        surface_path = output_dir / "surface-agent-review.json"
        _write_canonical(surface_path, surface)
        surface_file_sha = _file_sha256(surface_path)
        independent = build_independent_agent_review(
            inputs,
            originals,
            surface,
            surface_review_file_sha256=surface_file_sha,
        )
        independent_path = output_dir / "independent-agent-verification.json"
        _write_canonical(independent_path, independent)
        independent_file_sha = _file_sha256(independent_path)
        custodian, _ = load_sealed_custodian_actor(
            sealed_custodian_actor_path,
            parent_independent_actor=independent["actor"],
        )
        custodian_path = output_dir / "sealed-custodian-actor.json"
        _copy_regular(sealed_custodian_actor_path, custodian_path)
        copied_custodian, custodian_file_sha = load_sealed_custodian_actor(
            custodian_path,
            parent_independent_actor=independent["actor"],
        )
        if copied_custodian != custodian:
            raise RuntimeError("sealed custodian actor changed while copying")
        amendment = build_governance_amendment(
            inputs,
            originals,
            policy,
            surface,
            independent,
            sealed_custodian_actor=custodian,
            sealed_custodian_actor_file_sha256=custodian_file_sha,
            surface_review_file_sha256=surface_file_sha,
            independent_review_file_sha256=independent_file_sha,
            amended_at_utc=prepared_at_utc,
        )
        amendment_path = output_dir / "governance-amendment.json"
        _write_canonical(amendment_path, amendment)
        portable_draft = build_portable_ratification_draft(
            inputs=inputs,
            originals=originals,
            policy=policy,
            surface_review=surface,
            surface_review_file_sha256=surface_file_sha,
            independent_review=independent,
            independent_review_file_sha256=independent_file_sha,
            sealed_custodian_actor=custodian,
            sealed_custodian_actor_file_sha256=custodian_file_sha,
            amendment=amendment,
            amendment_file_sha256=_file_sha256(amendment_path),
            evaluator_contract=evaluator_contract,
            evaluator_contract_file_sha256=_file_sha256(evaluator_contract_path),
            prepared_at_utc=prepared_at_utc,
        )
        portable_draft_path = output_dir / "portable-ratification-draft.json"
        _write_canonical(portable_draft_path, portable_draft)
        draft = build_ratification_draft(
            inputs=inputs,
            surface_record_path=surface_record_path,
            independent_record_path=independent_record_path,
            surface_review_path=surface_path,
            independent_review_path=independent_path,
            sealed_custodian_actor_path=custodian_path,
            amendment_path=amendment_path,
            evaluator_contract_path=evaluator_contract_path,
            portable_draft_path=portable_draft_path,
            amendment=amendment,
            evaluator_contract=evaluator_contract,
            portable_draft=portable_draft,
            prepared_at_utc=prepared_at_utc,
        )
        draft_path = output_dir / "owner-ratification.draft.json"
        _write_canonical(draft_path, draft)
        _resolve_draft(draft_path)
        return {
            "surface_review": surface_path,
            "independent_review": independent_path,
            "sealed_custodian_actor": custodian_path,
            "amendment": amendment_path,
            "evaluator_contract": evaluator_contract_path,
            "portable_draft": portable_draft_path,
            "draft": draft_path,
        }
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def ratify_interactively(*, draft_path: Path, output_path: Path) -> dict[str, Any]:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError("owner ratification requires an interactive TTY")
    resolved = _resolve_draft(draft_path)
    draft = resolved["draft"]
    print("\nSN56 Krea governance ratification")
    print(f"Owner: {_OWNER}")
    print(f"Package: {draft['decision_bindings']['package_sha256']}")
    print(f"G.O.D evaluator: {draft['decision_bindings']['god_commit']}")
    print(
        "Governance amendment: "
        f"{draft['decision_bindings']['governance_amendment_sha256']}"
    )
    print(
        "Agent reviews are technical evidence. No independent human review "
        "occurred, and this is not a cryptographic or legal signature."
    )
    print("This ratification still does not authorize admission or GPU use.\n")
    phrase = input("Type the exact ratification phrase:\n> ")
    if phrase != draft["required_phrase"]:
        raise RuntimeError("ratification phrase did not match exactly")
    ratification = build_owner_ratification(resolved, ratified_at_utc=_now_utc())
    _write_canonical(output_path, ratification)
    validate_owner_ratification(ratification, resolved=resolved)
    return ratification


def _git_output(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
    )
    return completed.stdout.strip()


def _function_ast_and_source(source: str, name: str) -> tuple[str, str]:
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name} definition")
    node = matches[0]
    segment = ast.get_source_segment(source, node)
    if not isinstance(segment, str) or not segment:
        raise ValueError(f"could not recover {name} source")
    return ast.dump(node, include_attributes=False), segment


def _extensions_literal(source: str) -> tuple[str, ...]:
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == "SUPPORTED_IMAGE_FILE_EXTENSIONS"
                    for target in node.targets
                )
            )
            or (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "SUPPORTED_IMAGE_FILE_EXTENSIONS"
            )
        )
    ]
    if len(matches) != 1:
        raise ValueError("expected one SUPPORTED_IMAGE_FILE_EXTENSIONS assignment")
    value = matches[0].value
    extensions = ast.literal_eval(value)
    if (
        not isinstance(extensions, tuple)
        or not extensions
        or any(
            not isinstance(item, str)
            or not item.startswith(".")
            or item != item.lower()
            for item in extensions
        )
    ):
        raise ValueError("G.O.D image extension tuple is unsafe")
    return extensions


def _build_god_evaluator_contract(
    checkout: Path, *, expected_commit: str
) -> tuple[dict[str, Any], Path, Path]:
    checkout = _safe_directory(checkout, "G.O.D checkout")
    if not _GIT_SHA.fullmatch(expected_commit):
        raise ValueError("G.O.D commit must be a full lowercase Git SHA")
    try:
        head = _git_output(checkout, "rev-parse", "--verify", "HEAD")
        origin = _git_output(checkout, "remote", "get-url", "origin")
        status = _git_output(
            checkout, "status", "--porcelain=v1", "--untracked-files=all"
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("G.O.D checkout could not be verified") from exc
    if head != expected_commit or origin.rstrip("/") != _GOD_ORIGIN.rstrip("/"):
        raise ValueError("G.O.D checkout commit or origin differs from the contract")
    if status:
        raise ValueError("G.O.D checkout must be clean, including untracked files")
    image_io_path = _safe_file(checkout / _GOD_IMAGE_IO, "G.O.D image_io.py")
    constants_path = _safe_file(
        checkout / _GOD_DATASET_CONSTANTS, "G.O.D dataset constants"
    )
    image_source = _read_stable(image_io_path, "G.O.D image_io.py").decode("utf-8")
    constants_source = _read_stable(constants_path, "G.O.D dataset constants").decode(
        "utf-8"
    )
    enumerator_ast, enumerator_source = _function_ast_and_source(
        image_source, "list_supported_images"
    )
    if enumerator_ast != _GOD_ENUMERATOR_AST:
        raise ValueError("G.O.D image enumerator semantics changed; review required")
    extensions = _extensions_literal(constants_source)
    body = {
        "schema": 1,
        "kind": "forge-krea-god-evaluator-enumerator-contract",
        "origin": _GOD_ORIGIN,
        "commit": expected_commit,
        "checkout_clean": True,
        "image_io": {
            "upstream_relative_path": _GOD_IMAGE_IO.as_posix(),
            "file_sha256": _file_sha256(image_io_path),
            "callable": "list_supported_images",
            "callable_sha256": hashlib.sha256(
                enumerator_source.encode("utf-8")
            ).hexdigest(),
        },
        "dataset_constants": {
            "upstream_relative_path": _GOD_DATASET_CONSTANTS.as_posix(),
            "file_sha256": _file_sha256(constants_path),
        },
        "extensions": list(extensions),
        "execution_method": "ast-verified-exact-semantics-mirror",
    }
    return _seal(body, "contract_sha256"), image_io_path, constants_path


def _validate_portable_god_evaluator_contract(
    root: Path,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    contract, _ = _json(
        root / "evaluator" / "contract.json",
        "portable G.O.D evaluator contract",
        canonical=True,
    )
    _exact(
        contract,
        {
            "schema",
            "kind",
            "origin",
            "commit",
            "checkout_clean",
            "image_io",
            "dataset_constants",
            "extensions",
            "execution_method",
            "contract_sha256",
        },
        "portable G.O.D evaluator contract",
    )
    image_binding = _object(contract["image_io"], "G.O.D image_io binding")
    constants_binding = _object(
        contract["dataset_constants"], "G.O.D constants binding"
    )
    _exact(
        image_binding,
        {"upstream_relative_path", "file_sha256", "callable", "callable_sha256"},
        "G.O.D image_io binding",
    )
    _exact(
        constants_binding,
        {"upstream_relative_path", "file_sha256"},
        "G.O.D constants binding",
    )
    image_path = _safe_file(root / "evaluator" / "image_io.py", "portable image_io")
    constants_path = _safe_file(
        root / "evaluator" / "dataset_constants.py", "portable dataset constants"
    )
    image_source = _read_stable(image_path, "portable image_io").decode("utf-8")
    constants_source = _read_stable(
        constants_path, "portable dataset constants"
    ).decode("utf-8")
    enumerator_ast, enumerator_source = _function_ast_and_source(
        image_source, "list_supported_images"
    )
    extensions = _extensions_literal(constants_source)
    body = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if (
        contract["schema"] != 1
        or contract["kind"] != "forge-krea-god-evaluator-enumerator-contract"
        or contract["origin"] != _GOD_ORIGIN
        or not _GIT_SHA.fullmatch(contract["commit"])
        or contract["checkout_clean"] is not True
        or image_binding["upstream_relative_path"] != _GOD_IMAGE_IO.as_posix()
        or image_binding["file_sha256"] != _file_sha256(image_path)
        or image_binding["callable"] != "list_supported_images"
        or image_binding["callable_sha256"]
        != hashlib.sha256(enumerator_source.encode("utf-8")).hexdigest()
        or constants_binding["upstream_relative_path"]
        != _GOD_DATASET_CONSTANTS.as_posix()
        or constants_binding["file_sha256"] != _file_sha256(constants_path)
        or enumerator_ast != _GOD_ENUMERATOR_AST
        or contract["extensions"] != list(extensions)
        or contract["execution_method"] != "ast-verified-exact-semantics-mirror"
        or contract["contract_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("portable G.O.D evaluator contract is invalid")
    return contract, extensions


def _evaluator_list_supported_images(
    root: Path, extensions: tuple[str, ...]
) -> list[str]:
    """Execute the reviewed G.O.D list_supported_images semantics exactly."""

    directory = _safe_directory(root, "fixture dataset")
    return [
        file_name
        for file_name in os.listdir(directory)
        if file_name.lower().endswith(extensions)
    ]


def _candidate_row_bridge(
    *,
    candidate_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    split: str,
) -> None:
    candidates = {
        (row["source_id"], row["split"]): row
        for row in candidate_rows
        if row["split"] == split
    }
    finals = {Path(row["relative_image_path"]).stem: row for row in final_rows}
    if set(finals) != {source_id for source_id, _ in candidates}:
        raise ValueError(f"{split} final rows differ from candidate source ids")
    for (source_id, _), candidate in candidates.items():
        final = finals[source_id]
        expected = {
            "relative_image_path": Path(candidate["relative_image_path"]).name,
            "relative_caption_path": Path(candidate["relative_caption_path"]).name,
            "image_sha256": candidate["image_sha256"],
            "decoded_pixels_sha256": candidate["decoded_rgb_sha256"],
            "caption_sha256": candidate["caption_sha256"],
            "normalized_caption_sha256": candidate["normalized_caption_sha256"],
            "width": candidate["width"],
            "height": candidate["height"],
            "perceptual_hash64": candidate["perceptual_hash64"],
            "group_identity": candidate["group_identity"],
        }
        mismatches = {
            key: {"candidate": value, "final": final.get(key)}
            for key, value in expected.items()
            if final.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"{split} final row {source_id} differs from candidate: {mismatches}"
            )


def _dataset_and_rows_from_candidate(
    *,
    root: Path,
    candidate_rows: list[dict[str, Any]],
    split: str,
    evaluator_contract: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build evaluator/rich identities without a second full image decode.

    ``validate_package`` has already recomputed every decoded RGB and pHash.
    Re-decoding the full ~GB package here is redundant and can inflate peak
    memory on the CPU-only admission host.  Pillow header reads still bind the
    evaluator-visible format and original mode, while every byte/dimension and
    decoded identity is copied from—and later compared with—the validated
    candidate rows.
    """

    try:
        from PIL import Image, __version__ as pillow_version
    except ImportError as exc:  # pragma: no cover - production venv has Pillow.
        raise RuntimeError("Pillow is required to admit Krea fixtures") from exc
    extensions = tuple(evaluator_contract["extensions"])
    selected_by_name = {
        Path(row["relative_image_path"]).name: row
        for row in candidate_rows
        if row["split"] == split
    }
    evaluator_order = _evaluator_list_supported_images(root / split, extensions)
    if len(evaluator_order) != len(set(evaluator_order)) or set(evaluator_order) != set(
        selected_by_name
    ):
        raise ValueError(f"{split} evaluator file set differs from candidate rows")
    selected = [selected_by_name[name] for name in evaluator_order]
    identity_rows = []
    rich_rows = []
    order = []
    for index, candidate in enumerate(selected):
        image_name = Path(candidate["relative_image_path"]).name
        prompt_name = Path(candidate["relative_caption_path"]).name
        image_path = _safe_file(root / split / image_name, f"{split} image")
        prompt_path = _safe_file(root / split / prompt_name, f"{split} caption")
        prompt_bytes = _read_stable(prompt_path, f"{split} caption")
        with Image.open(image_path) as opened:
            image_format = opened.format
            image_mode = opened.mode
            dimensions = opened.size
        if dimensions != (candidate["width"], candidate["height"]):
            raise ValueError(f"{split} candidate dimensions changed: {image_name}")
        if (
            _file_sha256(image_path) != candidate["image_sha256"]
            or image_path.stat().st_size != candidate["image_bytes"]
            or hashlib.sha256(prompt_bytes).hexdigest() != candidate["caption_sha256"]
        ):
            raise ValueError(f"{split} candidate bytes changed: {image_name}")
        order.append(image_name)
        identity_rows.append(
            {
                "index": index,
                "image": image_name,
                "image_sha256": candidate["image_sha256"],
                "image_bytes": candidate["image_bytes"],
                "image_width": candidate["width"],
                "image_height": candidate["height"],
                "image_format": image_format,
                "image_mode": image_mode,
                "prompt": prompt_name,
                "prompt_sha256": candidate["caption_sha256"],
                "prompt_bytes": len(prompt_bytes),
            }
        )
        content = {
            "image_sha256": candidate["image_sha256"],
            "decoded_pixels_sha256": candidate["decoded_rgb_sha256"],
            "caption_sha256": candidate["caption_sha256"],
            "normalized_caption_sha256": candidate["normalized_caption_sha256"],
            "width": candidate["width"],
            "height": candidate["height"],
            "mode": "RGB",
        }
        rich_rows.append(
            {
                "row_id": "row-" + krea_provenance.canonical_sha256(content),
                "relative_image_path": image_name,
                "relative_caption_path": prompt_name,
                "content_sha256": krea_provenance.canonical_sha256(content),
                **content,
                "media_type": image_format,
                "perceptual_hash64": candidate["perceptual_hash64"],
                "decoder": {"library": "Pillow", "version": pillow_version},
                "group_identity": candidate["group_identity"],
            }
        )
    identity_body = {"evaluator_order": order, "rows": identity_rows}
    identity = {
        **identity_body,
        "sha256": krea_dataset_identity._json_sha256(identity_body),
    }
    rich_rows.sort(key=lambda row: row["row_id"])
    krea_dataset_identity.validate_identity(identity)
    return identity, rich_rows


def build_agent_governed_manifest(
    *,
    role: str,
    inputs: dict[str, Any],
    resolved: dict[str, Any],
    ratification: dict[str, Any],
    ratification_file_sha256: str,
    evaluator_contract: dict[str, Any],
) -> dict[str, Any]:
    """Derive one production fixture solely from the immutable package bytes."""

    if role not in _ROLES:
        raise ValueError("discovery manifest role must be D1 or D2")
    if "draft" in resolved:
        validate_owner_ratification(ratification, resolved=resolved)
    elif resolved.get("ratification") != ratification:
        raise ValueError("portable ratification was not validated before derivation")
    if evaluator_contract.get(
        "kind"
    ) != "forge-krea-god-evaluator-enumerator-contract" or not _GIT_SHA.fullmatch(
        str(evaluator_contract.get("commit", ""))
    ):
        raise ValueError("verified G.O.D evaluator contract is required")
    candidate = inputs["candidates"][role]["document"]
    root = inputs["root"] / role
    training_identity, training_rows = _dataset_and_rows_from_candidate(
        root=root,
        candidate_rows=candidate["rows"],
        split="training",
        evaluator_contract=evaluator_contract,
    )
    evaluation_identity, evaluation_rows = _dataset_and_rows_from_candidate(
        root=root,
        candidate_rows=candidate["rows"],
        split="evaluation",
        evaluator_contract=evaluator_contract,
    )
    _candidate_row_bridge(
        candidate_rows=candidate["rows"], final_rows=training_rows, split="training"
    )
    _candidate_row_bridge(
        candidate_rows=candidate["rows"],
        final_rows=evaluation_rows,
        split="evaluation",
    )
    group_fields = tuple(
        sorted(
            krea_fixture._BASE_GROUP_DISJOINT_FIELDS
            | (krea_fixture._D2_GROUP_FIELDS if role == "D2" else frozenset())
        )
    )
    report = krea_fixture._duplicates(
        training_rows,
        evaluation_rows,
        threshold=8,
        group_disjoint_fields=group_fields,
    )
    if report["exact_matches"] or report["cross_split_group_matches"]:
        raise ValueError("candidate-to-final derivation found unadjudicable leakage")
    archive = root / "training.zip"
    archive_identity = krea_fixture._archive_identity(
        archive, training_identity=training_identity
    )
    archive_record = candidate["training_archive"]
    if (
        _file_sha256(archive) != archive_record["sha256"]
        or archive.stat().st_size != archive_record["bytes"]
    ):
        raise ValueError("final training archive differs from candidate")
    rights, rights_file_sha = _json(
        root / "rights-ledger.candidate.json", f"{role} rights ledger", canonical=True
    )
    captions, captions_file_sha = _json(
        root / "caption-ledger.candidate.json",
        f"{role} caption ledger",
        canonical=True,
    )
    similarity, similarity_file_sha = _json(
        root / "similarity-evidence.candidate.json",
        f"{role} similarity ledger",
        canonical=True,
    )
    surface = resolved["surface_review"]
    independent = resolved["independent_review"]
    surface_actor = surface["actor"]
    independent_actor = independent["actor"]
    preparer_actor = _admission_implementation_actor(
        resolved["amendment"], role="fixture_implementer"
    )
    materialization, _ = _json(
        inputs["root"]
        / candidate["bindings"]["source_materialization"]["relative_path"],
        f"{role} source materialization",
        canonical=True,
    )
    reviewed_at = surface["reviewed_at_utc"]
    row_ids_train = sorted(row["row_id"] for row in training_rows)
    row_ids_eval = sorted(row["row_id"] for row in evaluation_rows)
    all_row_ids = [row["row_id"] for row in training_rows + evaluation_rows]
    reviewed_pairs = krea_fixture._reviewed_pairs(all_row_ids)
    governance = {
        "mode": _MODE,
        "policy_sha256": resolved["policy"]["policy_sha256"],
        "governance_amendment": {
            "file_sha256": resolved["amendment_file_sha256"],
            "amendment_sha256": resolved["amendment"]["amendment_sha256"],
        },
        "owner_ratification": {
            "file_sha256": _digest(
                ratification_file_sha256, "owner ratification file SHA-256"
            ),
            "ratification_sha256": ratification["ratification_sha256"],
        },
        "source_package": {
            "file_sha256": inputs["package_manifest_file_sha256"],
            "package_sha256": inputs["package"]["package_sha256"],
        },
        "candidate_manifest_sha256": candidate["candidate_manifest_sha256"],
        "surface_agent_review": {
            "file_sha256": resolved["surface_review_file_sha256"],
            "review_sha256": surface["review_sha256"],
            "actor": surface_actor,
        },
        "independent_agent_review": {
            "file_sha256": resolved["independent_review_file_sha256"],
            "review_sha256": independent["review_sha256"],
            "actor": independent_actor,
        },
        "preparer_actor": preparer_actor,
        "accountable_owner_identity": _OWNER,
        "owner_identity_assurance": _OWNER_ASSURANCE,
        "agent_review_is_not_human_review": True,
        "independent_human_review_performed": False,
        "claim_limit": _CLAIM_LIMIT,
    }
    tool_identity = {
        "fixture_module_sha256": _file_sha256(
            Path(krea_fixture.__file__).resolve(strict=True)
        ),
        "dataset_identity_module_sha256": _file_sha256(
            Path(krea_dataset_identity.__file__).resolve(strict=True)
        ),
        "god_commit": evaluator_contract["commit"],
        "enumerator_module": "validator.evaluation.image_io",
        "enumerator_qualname": evaluator_contract["image_io"]["callable"],
        "enumerator_source_sha256": evaluator_contract["image_io"]["file_sha256"],
        "enumerator_callable_sha256": evaluator_contract["image_io"]["callable_sha256"],
        "extensions": evaluator_contract["extensions"],
        "perceptual_hash": "rgb-luma-average-hash-8x8-bilinear-after-exif-transpose",
    }
    body = {
        "schema": 2,
        "kind": "forge-krea-curated-fixture",
        "concept_id": candidate["concept_id"],
        "experimental_role": role,
        "trigger_token": candidate["trigger_token"],
        "caption_policy": {
            "record_sha256": captions_file_sha,
            "reviewer_identity": surface_actor["display_name"],
            "reviewed_at_utc": reviewed_at,
            "decision": "approved",
            "training_row_ids_sha256": krea_provenance.canonical_sha256(row_ids_train),
            "evaluation_row_ids_sha256": krea_provenance.canonical_sha256(row_ids_eval),
            "assertions": {
                "manual_review_complete": True,
                "captions_match_images": True,
                "trigger_usage_consistent": True,
                "evaluation_leakage_absent": True,
            },
        },
        "source_rights": {
            "owner": rights["curation_owner_identity"],
            "locator": rights["source_locator"],
            "retrieved_at_utc": materialization["retrieved_at_utc"],
            "record_sha256": rights_file_sha,
            "reviewer_identity": surface_actor["display_name"],
            "reviewed_at_utc": reviewed_at,
            "decision": "approved_for_calibration",
            "assertions": {
                "lawful_access": True,
                "calibration_use_allowed": True,
                "redistribution_reviewed": True,
                "sensitive_content_absent": True,
            },
        },
        "preparer_identity": preparer_actor["display_name"],
        "training_archive": {
            "sha256": archive_record["sha256"],
            "bytes": archive_record["bytes"],
        },
        "training_archive_identity": archive_identity,
        "training_dataset_identity": training_identity,
        "evaluation_dataset_identity": evaluation_identity,
        "training_dataset_shape_sha256": krea_provenance.canonical_sha256(
            [
                {
                    "width": row["width"],
                    "height": row["height"],
                    "mode": row["mode"],
                    "media_type": row["media_type"],
                }
                for row in training_rows
            ]
        ),
        "evaluation_dataset_shape_sha256": krea_provenance.canonical_sha256(
            [
                {
                    "width": row["width"],
                    "height": row["height"],
                    "mode": row["mode"],
                    "media_type": row["media_type"],
                }
                for row in evaluation_rows
            ]
        ),
        "training_rows": training_rows,
        "evaluation_rows": evaluation_rows,
        "tool_identity": tool_identity,
        "near_duplicate_policy": {
            "maximum_hamming_distance": 8,
            "report": report,
            "report_sha256": krea_provenance.canonical_sha256(report),
            "passed": True,
            "group_disjoint_fields": list(group_fields),
            "human_similarity_review": {
                "reviewer_identity": surface_actor["display_name"],
                "reviewed_at_utc": reviewed_at,
                "record_sha256": similarity_file_sha,
                "method": "owner-ratified-agent-review-plus-pinned-ahash",
                "reviewed_pair_count": len(reviewed_pairs),
                "reviewed_pairs_sha256": krea_provenance.canonical_sha256(
                    reviewed_pairs
                ),
                "decision": "passed",
                "passed": True,
            },
        },
        "governance": governance,
    }
    manifest = _seal(body, "manifest_sha256")
    krea_fixture.validate_manifest(manifest)
    # The canonical ledgers' own semantic hashes remain transitively bound by
    # the canonical surface review and candidate manifest.
    if (
        candidate["bindings"]["rights_ledger"]["file_sha256"] != rights_file_sha
        or candidate["bindings"]["caption_ledger"]["file_sha256"] != captions_file_sha
        or candidate["bindings"]["similarity_evidence"]["file_sha256"]
        != similarity_file_sha
    ):
        raise ValueError("final fixture ledgers escaped candidate bindings")
    return manifest


def _copy_regular(source: Path, destination: Path) -> None:
    source = _safe_file(source, "copied source")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    digest = hashlib.sha256()
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                digest.update(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if digest.hexdigest() != _file_sha256(source):
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"source changed while copied: {source}")


def _copy_tree(source: Path, destination: Path) -> None:
    source = _safe_directory(source, "copied tree")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    destination.mkdir(parents=True, mode=0o700)
    for path in sorted(
        source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()
    ):
        relative = path.relative_to(source)
        if path.is_symlink():
            raise ValueError(f"copied tree contains a symlink: {path}")
        if path.is_dir():
            (destination / relative).mkdir(mode=0o700)
        elif path.is_file() and stat.S_ISREG(path.stat().st_mode):
            _copy_regular(path, destination / relative)
        else:
            raise ValueError(f"copied tree contains an unsafe entry: {path}")


def _inventory(root: Path, *, excluded: set[str] | None = None) -> list[dict[str, Any]]:
    root = _safe_directory(root, "bundle root")
    excluded = set() if excluded is None else excluded
    rows = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"bundle contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError(f"bundle contains an unsafe entry: {relative}")
        if relative not in excluded:
            rows.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    return rows


_MATERIALIZED_SUPPORT_PATHS = frozenset(
    {
        "governance/policy.json",
        "governance/amendment.json",
        "governance/ratification-draft.json",
        "governance/owner-ratification.json",
        "governance/sealed-custodian-actor.json",
        "reviews/surface-agent-review.json",
        "reviews/independent-agent-verification.json",
        "reviews/surface-source-record.json",
        "reviews/independent-source-record.json",
        "confirmation/public-commitment.md",
        "confirmation/shape-amendment.json",
        "confirmation/blinded-acceptance.request.json",
        "evaluator/image_io.py",
        "evaluator/dataset_constants.py",
        "evaluator/contract.json",
        "fixtures/D1/fixture-manifest.json",
        "fixtures/D1/fixture-approval.json",
        "fixtures/D2/fixture-manifest.json",
        "fixtures/D2/fixture-approval.json",
    }
)


def _expected_materialized_paths(package_root: Path) -> set[str]:
    package_paths = {
        f"fixture-package-v2/{row['path']}" for row in _inventory(package_root)
    }
    return package_paths | set(_MATERIALIZED_SUPPORT_PATHS)


def _relative_binding(
    *, root: Path, path: Path, semantic_key: str, semantic_value: str
) -> dict[str, str]:
    relative = path.relative_to(root).as_posix()
    if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
        raise ValueError("bundle binding is not a portable relative path")
    return {
        "relative_path": relative,
        "file_sha256": _file_sha256(path),
        semantic_key: _digest(semantic_value, semantic_key),
    }


def _load_relative_json(
    root: Path,
    binding: dict[str, Any],
    label: str,
    semantic_key: str,
    *,
    canonical: bool = True,
) -> tuple[Path, dict[str, Any]]:
    _exact(
        binding,
        {"relative_path", "file_sha256", semantic_key},
        f"{label} binding",
    )
    relative = PurePosixPath(binding["relative_path"])
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"{label} path is not portable")
    path = _safe_file(root / Path(*relative.parts), label)
    value, file_sha = _json(path, label, canonical=canonical)
    if file_sha != _digest(binding["file_sha256"], f"{label} file SHA-256"):
        raise ValueError(f"{label} file SHA-256 mismatch")
    if value.get(semantic_key) != _digest(binding[semantic_key], semantic_key):
        raise ValueError(f"{label} semantic SHA-256 mismatch")
    return path, value


def _load_relative_file(root: Path, binding: dict[str, Any], label: str) -> Path:
    _exact(binding, {"relative_path", "file_sha256"}, f"{label} binding")
    relative = PurePosixPath(binding["relative_path"])
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"{label} path is not portable")
    path = _safe_file(root / Path(*relative.parts), label)
    if _file_sha256(path) != _digest(binding["file_sha256"], f"{label} SHA-256"):
        raise ValueError(f"{label} file SHA-256 mismatch")
    return path


def build_blinded_acceptance_request(
    *,
    package: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    independent_review: dict[str, Any],
    independent_review_file_sha256: str,
    sealed_custodian_actor: dict[str, Any],
    sealed_custodian_actor_file_sha256: str,
    ratification: dict[str, Any],
    ratification_file_sha256: str,
    public_record_file_sha256: str,
    amendment_file_sha256: str,
    amendment_sha256: str,
    requested_at_utc: str,
) -> dict[str, Any]:
    parent_actor = krea_fixture._agent_actor(
        independent_review["actor"], "parent independent review actor"
    )
    custodian = krea_fixture._agent_actor(
        sealed_custodian_actor, "sealed custodian actor"
    )
    krea_fixture._validate_cross_agent_distinct(custodian, parent_actor)
    body = {
        "schema": 2,
        "kind": "forge-krea-blinded-confirmation-acceptance-request",
        "requested_at_utc": _utc(requested_at_utc, "acceptance request time"),
        "source_package": {
            "package_sha256": package["package_sha256"],
            "file_set_sha256": package["file_set_sha256"],
        },
        "discovery_fixture_manifest_sha256s": {
            role: manifests[role]["manifest_sha256"] for role in _ROLES
        },
        "governance": {
            "parent_independent_review": {
                "file_sha256": _digest(
                    independent_review_file_sha256,
                    "parent independent review file SHA-256",
                ),
                "review_sha256": independent_review["review_sha256"],
                "actor": parent_actor,
            },
            "sealed_custodian_actor": {
                "file_sha256": _digest(
                    sealed_custodian_actor_file_sha256,
                    "sealed custodian actor file SHA-256",
                ),
                "actor_sha256": krea_provenance.canonical_sha256(custodian),
                "actor": custodian,
            },
            "owner_ratification": {
                "file_sha256": _digest(
                    ratification_file_sha256,
                    "owner ratification file SHA-256",
                ),
                "ratification_sha256": ratification["ratification_sha256"],
            },
            "agent_review_is_not_human_review": True,
            "independent_human_review_performed": False,
        },
        "confirmation_commitment": {
            "public_record_file_sha256": _digest(
                public_record_file_sha256, "C public record SHA-256"
            ),
            "commitment_sha256": krea_c1c4_amendment.COMMITMENT_SHA256,
            "published_manifest_file_sha256s": (
                krea_c1c4_amendment.MANIFEST_FILE_SHA256S
            ),
            "shape_amendment_file_sha256": _digest(
                amendment_file_sha256, "C amendment file SHA-256"
            ),
            "shape_amendment_sha256": _digest(
                amendment_sha256, "C amendment semantic SHA-256"
            ),
            "c1c4_revealed": False,
        },
        "required_private_digest_only_fields": list(_BLINDED_PRIVATE_FIELDS),
        "admission_authorized": False,
        "gpu_execution_authorized": False,
    }
    return _seal(body, "request_sha256")


def materialize_discovery_inputs(
    *,
    draft_path: Path,
    ratification_path: Path,
    public_c1c4_record_path: Path,
    shape_amendment_path: Path,
    output_dir: Path,
    materialized_at_utc: str,
) -> dict[str, Any]:
    """Build immutable D1/D2 admission inputs; do not admit them yet."""

    resolved = _resolve_draft(draft_path)
    ratification, ratification_file_sha = _json(
        ratification_path, "owner ratification", canonical=True
    )
    validate_owner_ratification(ratification, resolved=resolved)
    public_record_path = _safe_file(public_c1c4_record_path, "C public record")
    if _file_sha256(public_record_path) != krea_c1c4_amendment.PUBLIC_RECORD_SHA256:
        raise ValueError("C public commitment record SHA-256 mismatch")
    shape_path = _safe_file(shape_amendment_path, "C shape amendment")
    shape, shape_file_sha = _json(shape_path, "C shape amendment", canonical=False)
    krea_c1c4_amendment.validate_amendment(shape)
    if (
        shape_file_sha != krea_c1c4_amendment.AMENDMENT_FILE_SHA256
        or shape["amendment_sha256"] != krea_c1c4_amendment.AMENDMENT_SHA256
    ):
        raise ValueError("C shape amendment differs from the frozen public binding")
    output_dir = Path(os.path.abspath(os.path.expanduser(output_dir)))
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    temporary = output_dir.with_name(output_dir.name + f".tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing to reuse temporary output: {temporary}")
    temporary.mkdir(parents=True, mode=0o700)
    try:
        _copy_tree(resolved["inputs"]["root"], temporary / "fixture-package-v2")
        _copy_regular(_POLICY_PATH, temporary / "governance" / "policy.json")
        _copy_regular(
            Path(resolved["draft"]["inputs"]["governance_amendment"]),
            temporary / "governance" / "amendment.json",
        )
        _copy_regular(
            ratification_path, temporary / "governance" / "owner-ratification.json"
        )
        _copy_regular(
            resolved["sealed_custodian_actor_path"],
            temporary / "governance" / "sealed-custodian-actor.json",
        )
        _copy_regular(
            resolved["portable_draft_path"],
            temporary / "governance" / "ratification-draft.json",
        )
        _copy_regular(
            Path(resolved["draft"]["inputs"]["surface_agent_review"]),
            temporary / "reviews" / "surface-agent-review.json",
        )
        _copy_regular(
            Path(resolved["draft"]["inputs"]["independent_agent_review"]),
            temporary / "reviews" / "independent-agent-verification.json",
        )
        _copy_regular(
            Path(resolved["draft"]["inputs"]["surface_source_record"]),
            temporary / "reviews" / "surface-source-record.json",
        )
        _copy_regular(
            Path(resolved["draft"]["inputs"]["independent_source_record"]),
            temporary / "reviews" / "independent-source-record.json",
        )
        _copy_regular(
            public_record_path, temporary / "confirmation" / "public-commitment.md"
        )
        _copy_regular(shape_path, temporary / "confirmation" / "shape-amendment.json")
        _copy_regular(
            Path(resolved["draft"]["inputs"]["god_evaluator_image_io"]),
            temporary / "evaluator" / "image_io.py",
        )
        _copy_regular(
            Path(resolved["draft"]["inputs"]["god_evaluator_constants"]),
            temporary / "evaluator" / "dataset_constants.py",
        )
        _copy_regular(
            resolved["evaluator_contract_path"],
            temporary / "evaluator" / "contract.json",
        )
        manifests = {
            role: build_agent_governed_manifest(
                role=role,
                inputs=resolved["inputs"],
                resolved=resolved,
                ratification=ratification,
                ratification_file_sha256=ratification_file_sha,
                evaluator_contract=resolved["evaluator_contract"],
            )
            for role in _ROLES
        }
        approvals = {}
        for role in _ROLES:
            manifest_path = temporary / "fixtures" / role / "fixture-manifest.json"
            _write_canonical(manifest_path, manifests[role])
            approval = krea_fixture.build_agent_governed_approval(
                manifests[role],
                technical_reviewer_actor=resolved["independent_review"]["actor"],
                accountable_owner_identity=_OWNER,
                approved_at_utc=ratification["ratified_at_utc"],
            )
            approval_path = temporary / "fixtures" / role / "fixture-approval.json"
            _write_canonical(approval_path, approval)
            krea_fixture.validate_approval(approval, fixture_manifest=manifests[role])
            approvals[role] = approval
        request = build_blinded_acceptance_request(
            package=resolved["inputs"]["package"],
            manifests=manifests,
            independent_review=resolved["independent_review"],
            independent_review_file_sha256=resolved["independent_review_file_sha256"],
            sealed_custodian_actor=resolved["sealed_custodian_actor"],
            sealed_custodian_actor_file_sha256=resolved[
                "sealed_custodian_actor_file_sha256"
            ],
            ratification=ratification,
            ratification_file_sha256=ratification_file_sha,
            public_record_file_sha256=_file_sha256(public_record_path),
            amendment_file_sha256=shape_file_sha,
            amendment_sha256=shape["amendment_sha256"],
            requested_at_utc=materialized_at_utc,
        )
        request_path = temporary / "confirmation" / "blinded-acceptance.request.json"
        _write_canonical(request_path, request)
        files = _inventory(temporary)
        body = {
            "schema": 1,
            "kind": "forge-krea-discovery-admission-input-bundle",
            "materialized_at_utc": _utc(materialized_at_utc, "materialization time"),
            "source_package_sha256": resolved["inputs"]["package"]["package_sha256"],
            "owner_ratification_sha256": ratification["ratification_sha256"],
            "fixture_manifest_sha256s": {
                role: manifests[role]["manifest_sha256"] for role in _ROLES
            },
            "fixture_approval_sha256s": {
                role: approvals[role]["approval_sha256"] for role in _ROLES
            },
            "blinded_acceptance_request_sha256": request["request_sha256"],
            "files": files,
            "file_set_sha256": krea_provenance.canonical_sha256(files),
            "admission_authorized": False,
            "gpu_execution_authorized": False,
            "c1c4_revealed": False,
        }
        materialization = _seal(body, "bundle_sha256")
        _write_canonical(temporary / "input-bundle.json", materialization)
        os.rename(temporary, output_dir)
        return materialization
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


_BLINDED_PRIVATE_FIELDS = [
    "C1-C4 file-to-semantic manifest SHA-256 mapping",
    "all-six D1,D2,C1,C2,C3,C4 semantic manifest map",
    "cross-review file SHA-256, fixture-set SHA-256, pair count and SHA-256",
    "cross-review binding SHA-256",
    "exact pre-ratified sealed-custodian actor",
    "custody and unrevealed assertions",
]


def validate_blinded_acceptance_request(request: dict[str, Any]) -> dict[str, Any]:
    request = _object(request, "blinded acceptance request")
    _exact(
        request,
        {
            "schema",
            "kind",
            "requested_at_utc",
            "source_package",
            "discovery_fixture_manifest_sha256s",
            "governance",
            "confirmation_commitment",
            "required_private_digest_only_fields",
            "admission_authorized",
            "gpu_execution_authorized",
            "request_sha256",
        },
        "blinded acceptance request",
    )
    body = {key: value for key, value in request.items() if key != "request_sha256"}
    source = _object(request["source_package"], "acceptance source package")
    _exact(source, {"package_sha256", "file_set_sha256"}, "acceptance source package")
    for key, value in source.items():
        _digest(value, f"acceptance source package {key}")
    discovery = _object(
        request["discovery_fixture_manifest_sha256s"],
        "discovery fixture manifest hashes",
    )
    _exact(discovery, set(_ROLES), "discovery fixture manifest hashes")
    for role, digest in discovery.items():
        _digest(digest, f"{role} semantic manifest SHA-256")
    governance = _object(request["governance"], "acceptance request governance")
    _exact(
        governance,
        {
            "parent_independent_review",
            "sealed_custodian_actor",
            "owner_ratification",
            "agent_review_is_not_human_review",
            "independent_human_review_performed",
        },
        "acceptance request governance",
    )
    parent = _object(
        governance["parent_independent_review"], "parent independent review"
    )
    custodian_binding = _object(
        governance["sealed_custodian_actor"], "sealed custodian actor binding"
    )
    owner = _object(governance["owner_ratification"], "owner ratification binding")
    _exact(
        parent,
        {"file_sha256", "review_sha256", "actor"},
        "parent independent review",
    )
    _exact(
        custodian_binding,
        {"file_sha256", "actor_sha256", "actor"},
        "sealed custodian actor binding",
    )
    _exact(
        owner,
        {"file_sha256", "ratification_sha256"},
        "owner ratification binding",
    )
    parent_actor = krea_fixture._agent_actor(
        parent["actor"], "parent independent review actor"
    )
    custodian = krea_fixture._agent_actor(
        custodian_binding["actor"], "sealed custodian actor"
    )
    krea_fixture._validate_cross_agent_distinct(custodian, parent_actor)
    for binding, keys, label in (
        (parent, ("file_sha256", "review_sha256"), "parent independent review"),
        (
            custodian_binding,
            ("file_sha256", "actor_sha256"),
            "sealed custodian actor",
        ),
        (owner, ("file_sha256", "ratification_sha256"), "owner ratification"),
    ):
        for key in keys:
            _digest(binding[key], f"{label} {key}")
    commitment = _object(request["confirmation_commitment"], "confirmation commitment")
    _exact(
        commitment,
        {
            "public_record_file_sha256",
            "commitment_sha256",
            "published_manifest_file_sha256s",
            "shape_amendment_file_sha256",
            "shape_amendment_sha256",
            "c1c4_revealed",
        },
        "confirmation commitment",
    )
    for key in (
        "public_record_file_sha256",
        "commitment_sha256",
        "shape_amendment_file_sha256",
        "shape_amendment_sha256",
    ):
        _digest(commitment[key], f"confirmation commitment {key}")
    published = _object(
        commitment["published_manifest_file_sha256s"],
        "published C manifest file hashes",
    )
    _exact(published, {"C1", "C2", "C3", "C4"}, "published C hashes")
    for role, digest in published.items():
        _digest(digest, f"published {role} manifest file SHA-256")
    if (
        request["schema"] != 2
        or request["kind"] != "forge-krea-blinded-confirmation-acceptance-request"
        or custodian_binding["actor_sha256"]
        != krea_provenance.canonical_sha256(custodian)
        or governance["agent_review_is_not_human_review"] is not True
        or governance["independent_human_review_performed"] is not False
        or commitment["public_record_file_sha256"]
        != krea_c1c4_amendment.PUBLIC_RECORD_SHA256
        or commitment["commitment_sha256"] != krea_c1c4_amendment.COMMITMENT_SHA256
        or commitment["published_manifest_file_sha256s"]
        != krea_c1c4_amendment.MANIFEST_FILE_SHA256S
        or commitment["shape_amendment_file_sha256"]
        != krea_c1c4_amendment.AMENDMENT_FILE_SHA256
        or commitment["shape_amendment_sha256"] != krea_c1c4_amendment.AMENDMENT_SHA256
        or commitment["c1c4_revealed"] is not False
        or request["required_private_digest_only_fields"] != _BLINDED_PRIVATE_FIELDS
        or request["admission_authorized"] is not False
        or request["gpu_execution_authorized"] is not False
        or request["request_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("blinded acceptance request is invalid")
    _utc(request["requested_at_utc"], "acceptance request time")
    return request


def build_blinded_acceptance(
    request: dict[str, Any],
    *,
    custodian_actor: dict[str, Any],
    c1c4_semantic_manifest_sha256s: dict[str, str],
    cross_fixture_review: dict[str, Any],
    reviewed_at_utc: str,
) -> dict[str, Any]:
    """Build only the digest-only output of the pre-ratified sealed custodian."""

    validate_blinded_acceptance_request(request)
    governance = request["governance"]
    custodian = krea_fixture._agent_actor(custodian_actor, "sealed custodian actor")
    if custodian != governance["sealed_custodian_actor"]["actor"]:
        raise ValueError("blinded acceptance actor is not the pre-ratified custodian")
    parent = {
        "review_sha256": governance["parent_independent_review"]["review_sha256"],
        "actor": governance["parent_independent_review"]["actor"],
    }
    expected_c_roles = {"C1", "C2", "C3", "C4"}
    if set(c1c4_semantic_manifest_sha256s) != expected_c_roles:
        raise ValueError("blinded acceptance needs exactly C1-C4 semantic hashes")
    for role, digest in c1c4_semantic_manifest_sha256s.items():
        _digest(digest, f"{role} semantic manifest SHA-256")
    all_six = {
        **request["discovery_fixture_manifest_sha256s"],
        **c1c4_semantic_manifest_sha256s,
    }
    krea_fixture.validate_agent_cross_fixture_binding_digest_only(
        cross_fixture_review,
        fixture_manifest_sha256s=all_six,
        parent_independent_review=parent,
        owner_ratification_sha256=governance["owner_ratification"][
            "ratification_sha256"
        ],
        acceptance_request_sha256=request["request_sha256"],
    )
    if cross_fixture_review["actor"] != custodian:
        raise ValueError("cross-fixture review was not made by the bound custodian")
    body = {
        "schema": 2,
        "kind": "forge-krea-blinded-confirmation-acceptance",
        "actor": custodian,
        "parent_independent_review": parent,
        "owner_ratification_sha256": governance["owner_ratification"][
            "ratification_sha256"
        ],
        "reviewed_at_utc": _utc(reviewed_at_utc, "blinded acceptance time"),
        "request_sha256": request["request_sha256"],
        "source_package": request["source_package"],
        "public_confirmation_commitment": request["confirmation_commitment"],
        "fixture_manifest_sha256s": all_six,
        "fixture_manifest_set_sha256": krea_provenance.canonical_sha256(all_six),
        "cross_fixture_review": cross_fixture_review,
        "assertions": {
            "c1c4_file_to_semantic_mapping_verified_in_custody": True,
            "all_six_cross_fixture_review_preexists_discovery_execution": True,
            "c1c4_remain_sealed_and_unrevealed": True,
            "no_c1c4_content_or_path_disclosed": True,
            "d2_selector_key_was_not_accessed_for_this_review": True,
            "d2_committed_selector_opening_bound_to_source_package": True,
            "sealed_custodian_actor_matches_owner_ratified_request": True,
            "custodian_is_distinct_from_parent_independent_actor": True,
            "agent_review_is_not_human_review": True,
        },
        "decision": "accepted_for_d1_d2_discovery_admission",
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "c1c4_revealed": False,
        "claim_limit": (
            "digest-only-agent-custody-acceptance-not-c1c4-admission-content-"
            "disclosure-human-review-quality-proof-or-gpu-authorization"
        ),
    }
    acceptance = _seal(body, "acceptance_sha256")
    validate_blinded_acceptance(acceptance, request=request)
    return acceptance


def validate_blinded_acceptance(
    acceptance: dict[str, Any], *, request: dict[str, Any]
) -> dict[str, Any]:
    validate_blinded_acceptance_request(request)
    acceptance = _object(acceptance, "blinded acceptance")
    _exact(
        acceptance,
        {
            "schema",
            "kind",
            "actor",
            "parent_independent_review",
            "owner_ratification_sha256",
            "reviewed_at_utc",
            "request_sha256",
            "source_package",
            "public_confirmation_commitment",
            "fixture_manifest_sha256s",
            "fixture_manifest_set_sha256",
            "cross_fixture_review",
            "assertions",
            "decision",
            "admission_authorized",
            "gpu_execution_authorized",
            "c1c4_revealed",
            "claim_limit",
            "acceptance_sha256",
        },
        "blinded acceptance",
    )
    body = {
        key: value for key, value in acceptance.items() if key != "acceptance_sha256"
    }
    governance = request["governance"]
    custodian = krea_fixture._agent_actor(
        acceptance["actor"], "blinded acceptance actor"
    )
    expected_custodian = governance["sealed_custodian_actor"]["actor"]
    parent = {
        "review_sha256": governance["parent_independent_review"]["review_sha256"],
        "actor": governance["parent_independent_review"]["actor"],
    }
    manifests = _object(
        acceptance["fixture_manifest_sha256s"], "all-six fixture manifest map"
    )
    if set(manifests) != set(krea_fixture._CROSS_FIXTURE_ROLES):
        raise ValueError("blinded acceptance does not cover exactly all six fixtures")
    for role, digest in manifests.items():
        _digest(digest, f"{role} semantic manifest SHA-256")
    krea_fixture.validate_agent_cross_fixture_binding_digest_only(
        acceptance["cross_fixture_review"],
        fixture_manifest_sha256s=manifests,
        parent_independent_review=parent,
        owner_ratification_sha256=governance["owner_ratification"][
            "ratification_sha256"
        ],
        acceptance_request_sha256=request["request_sha256"],
    )
    request_time = datetime.strptime(
        _utc(request["requested_at_utc"], "acceptance request time"),
        "%Y-%m-%dT%H:%M:%SZ",
    ).replace(tzinfo=timezone.utc)
    acceptance_time = datetime.strptime(
        _utc(acceptance["reviewed_at_utc"], "acceptance time"),
        "%Y-%m-%dT%H:%M:%SZ",
    ).replace(tzinfo=timezone.utc)
    cross_review_time = datetime.strptime(
        acceptance["cross_fixture_review"]["reviewed_at_utc"],
        "%Y-%m-%dT%H:%M:%SZ",
    ).replace(tzinfo=timezone.utc)
    if (
        acceptance["schema"] != 2
        or acceptance["kind"] != "forge-krea-blinded-confirmation-acceptance"
        or custodian != expected_custodian
        or acceptance["cross_fixture_review"]["actor"] != custodian
        or acceptance["parent_independent_review"] != parent
        or acceptance["owner_ratification_sha256"]
        != governance["owner_ratification"]["ratification_sha256"]
        or acceptance["request_sha256"] != request["request_sha256"]
        or acceptance["source_package"] != request["source_package"]
        or acceptance["public_confirmation_commitment"]
        != request["confirmation_commitment"]
        or {role: manifests[role] for role in _ROLES}
        != request["discovery_fixture_manifest_sha256s"]
        or acceptance["fixture_manifest_set_sha256"]
        != krea_provenance.canonical_sha256(manifests)
        or acceptance["assertions"]
        != {
            "c1c4_file_to_semantic_mapping_verified_in_custody": True,
            "all_six_cross_fixture_review_preexists_discovery_execution": True,
            "c1c4_remain_sealed_and_unrevealed": True,
            "no_c1c4_content_or_path_disclosed": True,
            "d2_selector_key_was_not_accessed_for_this_review": True,
            "d2_committed_selector_opening_bound_to_source_package": True,
            "sealed_custodian_actor_matches_owner_ratified_request": True,
            "custodian_is_distinct_from_parent_independent_actor": True,
            "agent_review_is_not_human_review": True,
        }
        or acceptance["decision"] != "accepted_for_d1_d2_discovery_admission"
        or acceptance["admission_authorized"] is not False
        or acceptance["gpu_execution_authorized"] is not False
        or acceptance["c1c4_revealed"] is not False
        or acceptance["claim_limit"]
        != (
            "digest-only-agent-custody-acceptance-not-c1c4-admission-content-"
            "disclosure-human-review-quality-proof-or-gpu-authorization"
        )
        or acceptance["acceptance_sha256"] != krea_provenance.canonical_sha256(body)
        or acceptance_time < request_time
        or acceptance_time < cross_review_time
    ):
        raise ValueError("blinded confirmation acceptance is invalid")
    return acceptance


def _validate_portable_ratification(
    ratification: dict[str, Any],
    *,
    amendment: dict[str, Any],
    amendment_file_sha256: str,
    policy: dict[str, Any],
    inputs: dict[str, Any],
    surface_review: dict[str, Any],
    independent_review: dict[str, Any],
    evaluator_contract: dict[str, Any],
    portable_draft: dict[str, Any],
    portable_draft_file_sha256: str,
) -> dict[str, Any]:
    _exact(
        ratification,
        {
            "schema",
            "kind",
            "owner_identity",
            "owner_identity_assurance",
            "ratified_at_utc",
            "portable_ratification_draft",
            "governance_amendment",
            "decision_bindings",
            "acknowledgements",
            "decision",
            "admission_authorized",
            "gpu_execution_authorized",
            "claim_limit",
            "ratification_sha256",
        },
        "portable owner ratification",
    )
    draft_binding = _object(
        ratification["portable_ratification_draft"],
        "portable ratification draft binding",
    )
    amendment_binding = _object(
        ratification["governance_amendment"], "ratification amendment binding"
    )
    decision_bindings = _object(
        ratification["decision_bindings"], "ratification decision bindings"
    )
    _exact(
        draft_binding,
        {"file_sha256", "draft_sha256"},
        "portable ratification draft binding",
    )
    _exact(
        amendment_binding,
        {"file_sha256", "amendment_sha256"},
        "ratification amendment binding",
    )
    for key, value in draft_binding.items():
        _digest(value, f"portable ratification draft {key}")
    expected_decision_bindings = _ratification_decision_bindings(
        inputs=inputs,
        amendment=amendment,
        evaluator_contract=evaluator_contract,
    )
    _exact(
        decision_bindings,
        set(expected_decision_bindings),
        "ratification decision bindings",
    )
    body = {
        key: value
        for key, value in ratification.items()
        if key != "ratification_sha256"
    }
    expected_acknowledgements = {
        "agents_are_not_humans": True,
        "no_independent_human_review_occurred": True,
        "owner_understands_review_scope_and_limitations": True,
        "owner_accepts_accountability_for_using_bound_agent_evidence": True,
        "owner_does_not_claim_personal_pair_review": True,
        "ratification_is_not_a_cryptographic_or_legal_signature": True,
        "owner_authorizes_mechanical_gpu_approval_after_envelope_and_host_plan_validation": True,
        "c1c4_remain_sealed": True,
        "stage1_is_discovery_only": True,
        "stage1_is_not_release_or_tournament_evidence": True,
        "stage2_requires_separate_commit_and_fresh_owner_ratification": True,
        "owner_accepts_exact_stage1_timing_margins": True,
        "owner_authorizes_only_prebound_stage1_technical_agents": True,
        "delegated_agents_cannot_change_frozen_rules": True,
    }
    if (
        ratification.get("schema") != 1
        or ratification.get("kind") != _RATIFICATION_KIND
        or ratification.get("owner_identity") != _OWNER
        or ratification.get("owner_identity_assurance") != _OWNER_ASSURANCE
        or amendment_binding
        != {
            "file_sha256": amendment_file_sha256,
            "amendment_sha256": amendment["amendment_sha256"],
        }
        or draft_binding
        != {
            "file_sha256": portable_draft_file_sha256,
            "draft_sha256": portable_draft["draft_sha256"],
        }
        or decision_bindings != expected_decision_bindings
        or ratification.get("decision") != "ratified_for_fixture_admission_input"
        or ratification.get("acknowledgements") != expected_acknowledgements
        or ratification.get("admission_authorized") is not False
        or ratification.get("gpu_execution_authorized") is not False
        or ratification.get("claim_limit") != _CLAIM_LIMIT
        or ratification.get("ratification_sha256")
        != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("portable owner ratification is invalid")
    krea_fixture.named_human(ratification["owner_identity"], "owner identity")
    ratified_at = datetime.strptime(
        _utc(ratification["ratified_at_utc"], "ratification time"),
        "%Y-%m-%dT%H:%M:%SZ",
    ).replace(tzinfo=timezone.utc)
    evidence_times = [
        surface_review["reviewed_at_utc"],
        independent_review["reviewed_at_utc"],
        amendment["amended_at_utc"],
        portable_draft["prepared_at_utc"],
    ]
    if any(
        ratified_at
        < datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        for value in evidence_times
    ):
        raise ValueError("owner ratification predates bound agent evidence")
    return ratification


def validate_materialized_inputs(
    root: Path, *, allowed_extra_files: set[str] | None = None
) -> dict[str, Any]:
    root = _safe_directory(root, "materialized discovery inputs")
    materialization, materialization_file_sha = _json(
        root / "input-bundle.json", "input bundle", canonical=True
    )
    body = {
        key: value for key, value in materialization.items() if key != "bundle_sha256"
    }
    extras = set() if allowed_extra_files is None else set(allowed_extra_files)
    if not extras <= {
        "admission-envelope.json",
        "confirmation/blinded-acceptance.json",
    }:
        raise ValueError("unsupported materialized-input exclusion")
    _exact(
        materialization,
        {
            "schema",
            "kind",
            "materialized_at_utc",
            "source_package_sha256",
            "owner_ratification_sha256",
            "fixture_manifest_sha256s",
            "fixture_approval_sha256s",
            "blinded_acceptance_request_sha256",
            "files",
            "file_set_sha256",
            "admission_authorized",
            "gpu_execution_authorized",
            "c1c4_revealed",
            "bundle_sha256",
        },
        "materialized discovery input bundle",
    )
    live_files = _inventory(root, excluded={"input-bundle.json", *extras})
    if (
        materialization.get("schema") != 1
        or materialization.get("kind") != "forge-krea-discovery-admission-input-bundle"
        or materialization.get("files") != live_files
        or materialization.get("file_set_sha256")
        != krea_provenance.canonical_sha256(live_files)
        or materialization.get("bundle_sha256")
        != krea_provenance.canonical_sha256(body)
        or materialization.get("admission_authorized") is not False
        or materialization.get("gpu_execution_authorized") is not False
        or materialization.get("c1c4_revealed") is not False
    ):
        raise ValueError("materialized discovery input bundle is invalid")
    package_root = root / "fixture-package-v2"
    package = krea_fixture_package.validate_package(package_root)
    if {row["path"] for row in live_files} != _expected_materialized_paths(
        package_root
    ):
        raise ValueError("materialized discovery topology is not the literal contract")
    if package["package_sha256"] != materialization["source_package_sha256"]:
        raise ValueError("materialized package differs from bundle binding")
    for key in ("fixture_manifest_sha256s", "fixture_approval_sha256s"):
        values = _object(materialization[key], f"materialization {key}")
        _exact(values, set(_ROLES), f"materialization {key}")
        for role, digest in values.items():
            _digest(digest, f"materialization {role} {key}")
    policy = load_policy(root / "governance" / "policy.json")
    surface_source, surface_source_sha = _json(
        root / "reviews" / "surface-source-record.json",
        "surface source record",
        canonical=False,
    )
    independent_source, independent_source_sha = _json(
        root / "reviews" / "independent-source-record.json",
        "independent source record",
        canonical=False,
    )
    inputs = _package_inputs(package_root)
    originals = _validate_original_records(
        inputs,
        root / "reviews" / "surface-source-record.json",
        root / "reviews" / "independent-source-record.json",
    )
    if (
        originals["surface_file_sha256"] != surface_source_sha
        or originals["independent_file_sha256"] != independent_source_sha
        or surface_source.get("kind") != "forge-krea-response-engineer-countersign"
        or independent_source.get("kind") != "forge-krea-independent-reviewer-approval"
    ):
        raise ValueError("portable original agent records changed")
    surface, surface_file_sha = _json(
        root / "reviews" / "surface-agent-review.json",
        "surface agent review",
        canonical=True,
    )
    validate_surface_agent_review(surface, inputs=inputs, originals=originals)
    independent, independent_file_sha = _json(
        root / "reviews" / "independent-agent-verification.json",
        "independent agent verification",
        canonical=True,
    )
    validate_independent_agent_review(
        independent,
        inputs=inputs,
        originals=originals,
        surface_review=surface,
        surface_review_file_sha256=surface_file_sha,
    )
    custodian, custodian_file_sha = load_sealed_custodian_actor(
        root / "governance" / "sealed-custodian-actor.json",
        parent_independent_actor=independent["actor"],
    )
    amendment, amendment_file_sha = _json(
        root / "governance" / "amendment.json",
        "governance amendment",
        canonical=True,
    )
    validate_governance_amendment(
        amendment,
        inputs=inputs,
        originals=originals,
        policy=policy,
        surface_review=surface,
        independent_review=independent,
        sealed_custodian_actor=custodian,
        sealed_custodian_actor_file_sha256=custodian_file_sha,
        surface_review_file_sha256=surface_file_sha,
        independent_review_file_sha256=independent_file_sha,
    )
    ratification, ratification_file_sha = _json(
        root / "governance" / "owner-ratification.json",
        "owner ratification",
        canonical=True,
    )
    evaluator_contract, _ = _validate_portable_god_evaluator_contract(root)
    portable_draft, portable_draft_file_sha = _json(
        root / "governance" / "ratification-draft.json",
        "portable ratification draft",
        canonical=True,
    )
    validate_portable_ratification_draft(
        portable_draft,
        inputs=inputs,
        originals=originals,
        policy=policy,
        surface_review=surface,
        surface_review_file_sha256=surface_file_sha,
        independent_review=independent,
        independent_review_file_sha256=independent_file_sha,
        sealed_custodian_actor=custodian,
        sealed_custodian_actor_file_sha256=custodian_file_sha,
        amendment=amendment,
        amendment_file_sha256=amendment_file_sha,
        evaluator_contract=evaluator_contract,
        evaluator_contract_file_sha256=_file_sha256(
            root / "evaluator" / "contract.json"
        ),
    )
    _validate_portable_ratification(
        ratification,
        amendment=amendment,
        amendment_file_sha256=amendment_file_sha,
        policy=policy,
        inputs=inputs,
        surface_review=surface,
        independent_review=independent,
        evaluator_contract=evaluator_contract,
        portable_draft=portable_draft,
        portable_draft_file_sha256=portable_draft_file_sha,
    )
    materialized_at = datetime.strptime(
        _utc(materialization["materialized_at_utc"], "materialization time"),
        "%Y-%m-%dT%H:%M:%SZ",
    ).replace(tzinfo=timezone.utc)
    ratified_at = datetime.strptime(
        ratification["ratified_at_utc"], "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)
    if materialized_at < ratified_at:
        raise ValueError("materialization predates owner ratification")
    portable_resolved = {
        "inputs": inputs,
        "policy": policy,
        "surface_review": surface,
        "surface_review_file_sha256": surface_file_sha,
        "independent_review": independent,
        "independent_review_file_sha256": independent_file_sha,
        "amendment": amendment,
        "amendment_file_sha256": amendment_file_sha,
        "portable_draft": portable_draft,
        "portable_draft_file_sha256": portable_draft_file_sha,
        "ratification": ratification,
    }
    manifests = {}
    approvals = {}
    for role in _ROLES:
        manifest, _ = _json(
            root / "fixtures" / role / "fixture-manifest.json",
            f"{role} fixture manifest",
            canonical=True,
        )
        approval, _ = _json(
            root / "fixtures" / role / "fixture-approval.json",
            f"{role} fixture approval",
            canonical=True,
        )
        krea_fixture.validate_manifest(manifest)
        krea_fixture.validate_approval(approval, fixture_manifest=manifest)
        expected_manifest = build_agent_governed_manifest(
            role=role,
            inputs=inputs,
            resolved=portable_resolved,
            ratification=ratification,
            ratification_file_sha256=ratification_file_sha,
            evaluator_contract=evaluator_contract,
        )
        expected_approval = krea_fixture.build_agent_governed_approval(
            expected_manifest,
            technical_reviewer_actor=independent["actor"],
            accountable_owner_identity=_OWNER,
            approved_at_utc=ratification["ratified_at_utc"],
        )
        if manifest != expected_manifest:
            raise ValueError(f"{role} manifest is not the canonical v2 derivation")
        if approval != expected_approval:
            raise ValueError(f"{role} approval is not the canonical derivation")
        if (
            manifest["governance"]["owner_ratification"]
            != {
                "file_sha256": ratification_file_sha,
                "ratification_sha256": ratification["ratification_sha256"],
            }
            or manifest["governance"]["candidate_manifest_sha256"]
            != package["candidate_manifest_sha256s"][role]
            or manifest["manifest_sha256"]
            != materialization["fixture_manifest_sha256s"][role]
            or approval["approval_sha256"]
            != materialization["fixture_approval_sha256s"][role]
        ):
            raise ValueError(f"{role} fixture escaped materialized governance")
        _candidate_row_bridge(
            candidate_rows=inputs["candidates"][role]["document"]["rows"],
            final_rows=manifest["training_rows"],
            split="training",
        )
        _candidate_row_bridge(
            candidate_rows=inputs["candidates"][role]["document"]["rows"],
            final_rows=manifest["evaluation_rows"],
            split="evaluation",
        )
        manifests[role] = manifest
        approvals[role] = approval
    public_path = root / "confirmation" / "public-commitment.md"
    if _file_sha256(public_path) != krea_c1c4_amendment.PUBLIC_RECORD_SHA256:
        raise ValueError("portable C public commitment changed")
    shape, shape_file_sha = _json(
        root / "confirmation" / "shape-amendment.json",
        "portable C shape amendment",
        canonical=False,
    )
    krea_c1c4_amendment.validate_amendment(shape)
    if shape_file_sha != krea_c1c4_amendment.AMENDMENT_FILE_SHA256:
        raise ValueError("portable C shape amendment file changed")
    request, request_file_sha = _json(
        root / "confirmation" / "blinded-acceptance.request.json",
        "blinded acceptance request",
        canonical=True,
    )
    expected_request = build_blinded_acceptance_request(
        package=package,
        manifests=manifests,
        independent_review=independent,
        independent_review_file_sha256=independent_file_sha,
        sealed_custodian_actor=custodian,
        sealed_custodian_actor_file_sha256=custodian_file_sha,
        ratification=ratification,
        ratification_file_sha256=ratification_file_sha,
        public_record_file_sha256=_file_sha256(public_path),
        amendment_file_sha256=shape_file_sha,
        amendment_sha256=shape["amendment_sha256"],
        requested_at_utc=request["requested_at_utc"],
    )
    if (
        request != expected_request
        or request["request_sha256"]
        != materialization["blinded_acceptance_request_sha256"]
    ):
        raise ValueError("blinded acceptance request changed")
    return {
        "materialization": materialization,
        "materialization_file_sha256": materialization_file_sha,
        "package": package,
        "policy": policy,
        "surface_review": surface,
        "surface_review_file_sha256": surface_file_sha,
        "independent_review": independent,
        "independent_review_file_sha256": independent_file_sha,
        "sealed_custodian_actor": custodian,
        "sealed_custodian_actor_file_sha256": custodian_file_sha,
        "amendment": amendment,
        "amendment_file_sha256": amendment_file_sha,
        "portable_draft": portable_draft,
        "portable_draft_file_sha256": portable_draft_file_sha,
        "ratification": ratification,
        "ratification_file_sha256": ratification_file_sha,
        "manifests": manifests,
        "approvals": approvals,
        "request": request,
        "request_file_sha256": request_file_sha,
    }


def _surface_map(
    *, root: Path, surface: dict[str, Any], surface_path: Path
) -> dict[str, dict[str, dict[str, Any]]]:
    rows = {(row["role"], row["surface"]): row for row in surface["surfaces"]}
    if set(rows) != {(role, item) for role in _ROLES for item in _SURFACES}:
        raise ValueError("surface review does not cover exactly six surfaces")
    binding = _relative_binding(
        root=root,
        path=surface_path,
        semantic_key="review_sha256",
        semantic_value=surface["review_sha256"],
    )
    return {
        role: {
            item: {
                **binding,
                "surface_entry_sha256": krea_provenance.canonical_sha256(
                    rows[(role, item)]
                ),
            }
            for item in _SURFACES
        }
        for role in _ROLES
    }


def finalize_discovery_envelope(
    *,
    materialized_root: Path,
    blinded_acceptance_path: Path,
    output_dir: Path,
    admitted_at_utc: str,
) -> dict[str, Any]:
    resolved = validate_materialized_inputs(materialized_root)
    acceptance, _ = _json(
        blinded_acceptance_path, "blinded confirmation acceptance", canonical=True
    )
    validate_blinded_acceptance(
        acceptance,
        request=resolved["request"],
    )
    admitted_at_utc = _utc(admitted_at_utc, "admission time")
    admitted_at = datetime.strptime(admitted_at_utc, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    accepted_at = datetime.strptime(
        acceptance["reviewed_at_utc"], "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)
    if admitted_at < accepted_at:
        raise ValueError("fixture admission predates blinded acceptance")
    output_dir = Path(os.path.abspath(os.path.expanduser(output_dir)))
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    temporary = output_dir.with_name(output_dir.name + f".tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing to reuse temporary output: {temporary}")
    _copy_tree(materialized_root, temporary)
    try:
        acceptance_target = temporary / "confirmation" / "blinded-acceptance.json"
        _copy_regular(blinded_acceptance_path, acceptance_target)
        source_package_root = temporary / "fixture-package-v2"
        package_manifest_path = source_package_root / "package-manifest.json"
        surface_path = temporary / "reviews" / "surface-agent-review.json"
        independent_path = temporary / "reviews" / "independent-agent-verification.json"
        policy_path = temporary / "governance" / "policy.json"
        amendment_path = temporary / "governance" / "amendment.json"
        portable_draft_path = temporary / "governance" / "ratification-draft.json"
        ratification_path = temporary / "governance" / "owner-ratification.json"
        custodian_path = temporary / "governance" / "sealed-custodian-actor.json"
        public_path = temporary / "confirmation" / "public-commitment.md"
        shape_path = temporary / "confirmation" / "shape-amendment.json"
        request_path = temporary / "confirmation" / "blinded-acceptance.request.json"
        fixtures = {}
        for role in _ROLES:
            manifest_path = temporary / "fixtures" / role / "fixture-manifest.json"
            approval_path = temporary / "fixtures" / role / "fixture-approval.json"
            fixture = {
                "manifest": _relative_binding(
                    root=temporary,
                    path=manifest_path,
                    semantic_key="manifest_sha256",
                    semantic_value=resolved["manifests"][role]["manifest_sha256"],
                ),
                "approval": _relative_binding(
                    root=temporary,
                    path=approval_path,
                    semantic_key="approval_sha256",
                    semantic_value=resolved["approvals"][role]["approval_sha256"],
                ),
            }
            fixtures[role] = fixture
        files = _inventory(temporary)
        admission_producer = _admission_implementation_actor(
            resolved["amendment"], role="admission_envelope_producer"
        )
        body = {
            "schema": 1,
            "kind": _ENVELOPE_KIND,
            "phase": "discovery",
            "source_package": {
                "relative_path": "fixture-package-v2",
                "package_manifest": _relative_binding(
                    root=temporary,
                    path=package_manifest_path,
                    semantic_key="package_sha256",
                    semantic_value=resolved["package"]["package_sha256"],
                ),
                "file_set_sha256": resolved["package"]["file_set_sha256"],
                "review_request_sha256": resolved["package"]["review_request_sha256"],
                "candidate_manifest_sha256s": resolved["package"][
                    "candidate_manifest_sha256s"
                ],
            },
            "governance": {
                "mode": _MODE,
                "policy": _relative_binding(
                    root=temporary,
                    path=policy_path,
                    semantic_key="policy_sha256",
                    semantic_value=resolved["policy"]["policy_sha256"],
                ),
                "amendment": _relative_binding(
                    root=temporary,
                    path=amendment_path,
                    semantic_key="amendment_sha256",
                    semantic_value=resolved["amendment"]["amendment_sha256"],
                ),
                "owner_ratification": _relative_binding(
                    root=temporary,
                    path=ratification_path,
                    semantic_key="ratification_sha256",
                    semantic_value=resolved["ratification"]["ratification_sha256"],
                ),
                "portable_ratification_draft": _relative_binding(
                    root=temporary,
                    path=portable_draft_path,
                    semantic_key="draft_sha256",
                    semantic_value=resolved["portable_draft"]["draft_sha256"],
                ),
                "independent_agent_review": _relative_binding(
                    root=temporary,
                    path=independent_path,
                    semantic_key="review_sha256",
                    semantic_value=resolved["independent_review"]["review_sha256"],
                ),
                "sealed_custodian_actor": {
                    "relative_path": custodian_path.relative_to(temporary).as_posix(),
                    "file_sha256": _file_sha256(custodian_path),
                    "actor_sha256": krea_provenance.canonical_sha256(
                        resolved["sealed_custodian_actor"]
                    ),
                    "actor": resolved["sealed_custodian_actor"],
                },
                "agent_review_is_not_human_review": True,
                "independent_human_review_performed": False,
            },
            "surface_countersigns": _surface_map(
                root=temporary,
                surface=resolved["surface_review"],
                surface_path=surface_path,
            ),
            "discovery_fixtures": fixtures,
            "confirmation_commitment": {
                "public_record": {
                    "relative_path": public_path.relative_to(temporary).as_posix(),
                    "file_sha256": _file_sha256(public_path),
                },
                "commitment_sha256": krea_c1c4_amendment.COMMITMENT_SHA256,
                "published_manifest_file_sha256s": (
                    krea_c1c4_amendment.MANIFEST_FILE_SHA256S
                ),
                "shape_amendment": _relative_binding(
                    root=temporary,
                    path=shape_path,
                    semantic_key="amendment_sha256",
                    semantic_value=krea_c1c4_amendment.AMENDMENT_SHA256,
                ),
                "acceptance_request": _relative_binding(
                    root=temporary,
                    path=request_path,
                    semantic_key="request_sha256",
                    semantic_value=resolved["request"]["request_sha256"],
                ),
                "blinded_acceptance": _relative_binding(
                    root=temporary,
                    path=acceptance_target,
                    semantic_key="acceptance_sha256",
                    semantic_value=acceptance["acceptance_sha256"],
                ),
                "c1c4_semantic_manifest_sha256s": {
                    role: acceptance["fixture_manifest_sha256s"][role]
                    for role in ("C1", "C2", "C3", "C4")
                },
                "cross_fixture_review": acceptance["cross_fixture_review"],
                "c1c4_revealed": False,
            },
            "bundle_files": files,
            "bundle_file_set_sha256": krea_provenance.canonical_sha256(files),
            "admission_producer_actor": admission_producer,
            "accountable_owner_identity": _OWNER,
            "admitted_at_utc": admitted_at_utc,
            "decision": "admitted",
            "admission_authorized": True,
            "gpu_execution_authorized": False,
            "claim_limit": (
                "d1-d2-discovery-fixture-integrity-only-not-confirmation-"
                "disclosure-competitiveness-or-gpu-authorization"
            ),
        }
        envelope = _seal(body, "envelope_sha256")
        _write_canonical(temporary / "admission-envelope.json", envelope)
        # Recheck bytes locally, then require the complete validator in a fresh
        # process before this function may return an admitted envelope.  The
        # fresh process keeps the heavyweight package/image rederivation out of
        # this process's peak memory while preventing a producer-only bug from
        # minting apparent admission authority.
        written, _ = _json(
            temporary / "admission-envelope.json",
            "written admission envelope",
            canonical=True,
        )
        if (
            written != envelope
            or _inventory(temporary, excluded={"admission-envelope.json"}) != files
        ):
            raise RuntimeError("written admission envelope failed final byte check")
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve(strict=True)),
                "validate",
                "--bundle",
                str(temporary),
            ],
            capture_output=True,
            text=True,
            timeout=900,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])},
        )
        if (
            completed.returncode != 0
            or completed.stdout.strip() != envelope["envelope_sha256"]
        ):
            detail = completed.stderr.strip().splitlines()[-1:]
            raise RuntimeError(
                "fresh-process admission validation failed"
                + (f": {detail[0]}" if detail else "")
            )
        os.rename(temporary, output_dir)
        return envelope
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _reject_nested_authority(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                key in {"admission_authorized", "gpu_execution_authorized"}
                and item is not False
            ):
                raise ValueError(f"{label} contains nested {key}")
            _reject_nested_authority(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nested_authority(item, f"{label}[{index}]")


def validate_envelope(root_or_path: Path) -> dict[str, Any]:
    candidate = Path(os.path.abspath(os.path.expanduser(root_or_path)))
    root = candidate if candidate.is_dir() else candidate.parent
    envelope_path = (
        root / "admission-envelope.json" if candidate.is_dir() else candidate
    )
    root = _safe_directory(root, "admission bundle")
    envelope, envelope_file_sha = _json(
        envelope_path, "admission envelope", canonical=True
    )
    _exact(
        envelope,
        {
            "schema",
            "kind",
            "phase",
            "source_package",
            "governance",
            "surface_countersigns",
            "discovery_fixtures",
            "confirmation_commitment",
            "bundle_files",
            "bundle_file_set_sha256",
            "admission_producer_actor",
            "accountable_owner_identity",
            "admitted_at_utc",
            "decision",
            "admission_authorized",
            "gpu_execution_authorized",
            "claim_limit",
            "envelope_sha256",
        },
        "admission envelope",
    )
    body = {key: value for key, value in envelope.items() if key != "envelope_sha256"}
    live_files = _inventory(root, excluded={"admission-envelope.json"})
    if (
        envelope["schema"] != 1
        or envelope["kind"] != _ENVELOPE_KIND
        or envelope["phase"] != "discovery"
        or envelope["bundle_files"] != live_files
        or envelope["bundle_file_set_sha256"]
        != krea_provenance.canonical_sha256(live_files)
        or envelope["envelope_sha256"] != krea_provenance.canonical_sha256(body)
        or envelope["accountable_owner_identity"] != _OWNER
        or envelope["decision"] != "admitted"
        or envelope["admission_authorized"] is not True
        or envelope["gpu_execution_authorized"] is not False
        or envelope["claim_limit"]
        != (
            "d1-d2-discovery-fixture-integrity-only-not-confirmation-"
            "disclosure-competitiveness-or-gpu-authorization"
        )
    ):
        raise ValueError("discovery admission envelope is invalid")
    admission_producer = krea_fixture._agent_actor(
        envelope["admission_producer_actor"], "admission producer actor"
    )
    krea_fixture.named_human(
        envelope["accountable_owner_identity"], "accountable owner identity"
    )
    _utc(envelope["admitted_at_utc"], "admitted_at_utc")
    materialized = validate_materialized_inputs(
        root,
        allowed_extra_files={
            "admission-envelope.json",
            "confirmation/blinded-acceptance.json",
        },
    )
    expected_admission_producer = _admission_implementation_actor(
        materialized["amendment"], role="admission_envelope_producer"
    )
    if admission_producer != expected_admission_producer:
        raise ValueError("admission producer differs from the bound implementation")
    expected_live_paths = _expected_materialized_paths(root / "fixture-package-v2") | {
        "input-bundle.json",
        "confirmation/blinded-acceptance.json",
    }
    if {row["path"] for row in live_files} != expected_live_paths:
        raise ValueError("admission envelope topology is not the literal contract")
    source = _object(envelope["source_package"], "source package")
    _exact(
        source,
        {
            "relative_path",
            "package_manifest",
            "file_set_sha256",
            "review_request_sha256",
            "candidate_manifest_sha256s",
        },
        "source package",
    )
    if source.get("relative_path") != "fixture-package-v2":
        raise ValueError("admission source package path is not literal")
    _, bound_package = _load_relative_json(
        root, source["package_manifest"], "package manifest", "package_sha256"
    )
    if (
        source["package_manifest"]["relative_path"]
        != "fixture-package-v2/package-manifest.json"
        or bound_package["package_sha256"] != materialized["package"]["package_sha256"]
        or source["file_set_sha256"] != materialized["package"]["file_set_sha256"]
        or source["review_request_sha256"]
        != materialized["package"]["review_request_sha256"]
        or source["candidate_manifest_sha256s"]
        != materialized["package"]["candidate_manifest_sha256s"]
    ):
        raise ValueError("envelope package binding mismatch")
    governance = _object(envelope["governance"], "envelope governance")
    _exact(
        governance,
        {
            "mode",
            "policy",
            "amendment",
            "owner_ratification",
            "portable_ratification_draft",
            "independent_agent_review",
            "sealed_custodian_actor",
            "agent_review_is_not_human_review",
            "independent_human_review_performed",
        },
        "envelope governance",
    )
    if (
        governance.get("mode") != _MODE
        or governance.get("agent_review_is_not_human_review") is not True
        or governance.get("independent_human_review_performed") is not False
    ):
        raise ValueError("envelope governance is overstated")
    for key, relative_path, semantic_key, expected in (
        (
            "policy",
            "governance/policy.json",
            "policy_sha256",
            materialized["policy"]["policy_sha256"],
        ),
        (
            "amendment",
            "governance/amendment.json",
            "amendment_sha256",
            materialized["amendment"]["amendment_sha256"],
        ),
        (
            "owner_ratification",
            "governance/owner-ratification.json",
            "ratification_sha256",
            materialized["ratification"]["ratification_sha256"],
        ),
        (
            "portable_ratification_draft",
            "governance/ratification-draft.json",
            "draft_sha256",
            materialized["portable_draft"]["draft_sha256"],
        ),
        (
            "independent_agent_review",
            "reviews/independent-agent-verification.json",
            "review_sha256",
            materialized["independent_review"]["review_sha256"],
        ),
    ):
        _, record = _load_relative_json(
            root, governance[key], f"governance {key}", semantic_key
        )
        if (
            governance[key]["relative_path"] != relative_path
            or record[semantic_key] != expected
        ):
            raise ValueError(f"governance {key} semantic mismatch")
    custodian_binding = _object(
        governance["sealed_custodian_actor"], "envelope sealed custodian actor"
    )
    _exact(
        custodian_binding,
        {"relative_path", "file_sha256", "actor_sha256", "actor"},
        "envelope sealed custodian actor",
    )
    custodian_path = _load_relative_file(
        root,
        {
            "relative_path": custodian_binding["relative_path"],
            "file_sha256": custodian_binding["file_sha256"],
        },
        "envelope sealed custodian actor",
    )
    if custodian_binding["relative_path"] != "governance/sealed-custodian-actor.json":
        raise ValueError("sealed custodian actor path is not literal")
    custodian, custodian_file_sha = load_sealed_custodian_actor(
        custodian_path,
        parent_independent_actor=materialized["independent_review"]["actor"],
    )
    if (
        custodian != materialized["sealed_custodian_actor"]
        or custodian_binding["actor"] != custodian
        or custodian_binding["actor_sha256"]
        != krea_provenance.canonical_sha256(custodian)
        or custodian_file_sha != materialized["sealed_custodian_actor_file_sha256"]
    ):
        raise ValueError("envelope sealed custodian actor binding mismatch")
    fixtures = _object(envelope["discovery_fixtures"], "discovery fixtures")
    _exact(fixtures, set(_ROLES), "discovery fixtures")
    for role in _ROLES:
        binding = _object(fixtures[role], f"{role} fixture binding")
        _exact(binding, {"manifest", "approval"}, f"{role} fixture binding")
        manifest_path, manifest = _load_relative_json(
            root, binding["manifest"], f"{role} manifest", "manifest_sha256"
        )
        _, approval = _load_relative_json(
            root, binding["approval"], f"{role} approval", "approval_sha256"
        )
        if (
            binding["manifest"]["relative_path"]
            != f"fixtures/{role}/fixture-manifest.json"
            or binding["approval"]["relative_path"]
            != f"fixtures/{role}/fixture-approval.json"
        ):
            raise ValueError(f"{role} fixture paths are not literal")
        krea_fixture.validate_manifest(manifest)
        krea_fixture.validate_approval(approval, fixture_manifest=manifest)
        if (
            manifest != materialized["manifests"][role]
            or approval != materialized["approvals"][role]
            or _file_sha256(manifest_path) != binding["manifest"]["file_sha256"]
        ):
            raise ValueError(f"{role} envelope fixture mismatch")
    countersigns = _object(envelope["surface_countersigns"], "surface countersigns")
    expected_surfaces = _surface_map(
        root=root,
        surface=materialized["surface_review"],
        surface_path=root / "reviews" / "surface-agent-review.json",
    )
    if countersigns != expected_surfaces:
        raise ValueError("surface countersign map does not cover exact review entries")
    confirmation = _object(
        envelope["confirmation_commitment"], "confirmation commitment"
    )
    _exact(
        confirmation,
        {
            "public_record",
            "commitment_sha256",
            "published_manifest_file_sha256s",
            "shape_amendment",
            "acceptance_request",
            "blinded_acceptance",
            "c1c4_semantic_manifest_sha256s",
            "cross_fixture_review",
            "c1c4_revealed",
        },
        "confirmation commitment",
    )
    if confirmation.get("c1c4_revealed") is not False:
        raise ValueError("discovery envelope reveals C1-C4")
    public_path = _load_relative_file(
        root, confirmation["public_record"], "C public commitment"
    )
    if (
        confirmation["public_record"]["relative_path"]
        != "confirmation/public-commitment.md"
        or _file_sha256(public_path) != krea_c1c4_amendment.PUBLIC_RECORD_SHA256
    ):
        raise ValueError("C public commitment binding mismatch")
    shape_path, shape = _load_relative_json(
        root,
        confirmation["shape_amendment"],
        "C shape amendment",
        "amendment_sha256",
        canonical=False,
    )
    if (
        confirmation["shape_amendment"]["relative_path"]
        != "confirmation/shape-amendment.json"
        or _file_sha256(shape_path) != krea_c1c4_amendment.AMENDMENT_FILE_SHA256
        or shape["amendment_sha256"] != krea_c1c4_amendment.AMENDMENT_SHA256
    ):
        raise ValueError("C shape amendment binding mismatch")
    krea_c1c4_amendment.validate_amendment(shape)
    _, request = _load_relative_json(
        root,
        confirmation["acceptance_request"],
        "acceptance request",
        "request_sha256",
    )
    _, acceptance = _load_relative_json(
        root,
        confirmation["blinded_acceptance"],
        "blinded acceptance",
        "acceptance_sha256",
    )
    if (
        confirmation["acceptance_request"]["relative_path"]
        != "confirmation/blinded-acceptance.request.json"
        or confirmation["blinded_acceptance"]["relative_path"]
        != "confirmation/blinded-acceptance.json"
    ):
        raise ValueError("confirmation acceptance paths are not literal")
    validate_blinded_acceptance(
        acceptance,
        request=request,
    )
    admitted_at = datetime.strptime(
        envelope["admitted_at_utc"], "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)
    accepted_at = datetime.strptime(
        acceptance["reviewed_at_utc"], "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)
    if admitted_at < accepted_at:
        raise ValueError("fixture admission predates blinded acceptance")
    published_c = _object(
        confirmation["published_manifest_file_sha256s"],
        "published C manifest file hashes",
    )
    semantic_c = _object(
        confirmation["c1c4_semantic_manifest_sha256s"],
        "C semantic manifest hashes",
    )
    cross_review = _object(
        confirmation["cross_fixture_review"], "cross-fixture review binding"
    )
    _exact(published_c, {"C1", "C2", "C3", "C4"}, "published C hashes")
    _exact(semantic_c, {"C1", "C2", "C3", "C4"}, "C semantic hashes")
    krea_fixture.validate_agent_cross_fixture_binding_digest_only(
        cross_review,
        fixture_manifest_sha256s=acceptance["fixture_manifest_sha256s"],
        parent_independent_review=acceptance["parent_independent_review"],
        owner_ratification_sha256=acceptance["owner_ratification_sha256"],
        acceptance_request_sha256=request["request_sha256"],
    )
    for label, values in (("published C", published_c), ("semantic C", semantic_c)):
        for role, digest in values.items():
            _digest(digest, f"{label} {role} SHA-256")
    if (
        confirmation.get("commitment_sha256") != krea_c1c4_amendment.COMMITMENT_SHA256
        or published_c != krea_c1c4_amendment.MANIFEST_FILE_SHA256S
        or semantic_c
        != {
            role: acceptance["fixture_manifest_sha256s"][role]
            for role in ("C1", "C2", "C3", "C4")
        }
        or cross_review != acceptance["cross_fixture_review"]
    ):
        raise ValueError("confirmation digest-only binding mismatch")
    for row in live_files:
        parts = PurePosixPath(row["path"]).parts
        if any(part in {"C1", "C2", "C3", "C4"} for part in parts):
            raise ValueError("discovery bundle contains a C1-C4 content path")
        if row["path"].endswith(".json"):
            value, _ = _json(root / row["path"], row["path"], canonical=False)
            _reject_nested_authority(value, row["path"])
    return {
        "envelope": envelope,
        "envelope_file_sha256": envelope_file_sha,
        "fixtures": materialized["manifests"],
        "fixture_approvals": materialized["approvals"],
        "ratification": materialized["ratification"],
        "blinded_acceptance": acceptance,
    }


def authorize_gpu_execution(
    *,
    plan_path: Path,
    admission_envelope_path: Path,
    technical_actor_path: Path,
    output_path: Path,
    approved_at_utc: str,
) -> dict[str, Any]:
    """Issue the separate GPU gate through one exclusive canonical writer."""

    try:
        from . import krea_execution_plan
    except ImportError:  # pragma: no cover - direct script execution.
        import krea_execution_plan  # type: ignore[no-redef]

    plan, _ = _json(plan_path, "Krea execution plan", canonical=True)
    technical_actor, _ = _json(
        technical_actor_path, "technical execution actor", canonical=True
    )
    output_path = Path(os.path.abspath(os.path.expanduser(output_path)))
    approval = krea_execution_plan.build_approval(
        plan,
        reviewer_identity=None,
        approved_at_utc=_utc(approved_at_utc, "GPU approval time"),
        admission_envelope_path=admission_envelope_path,
        approval_output_path=output_path,
        technical_reviewer_actor=technical_actor,
    )
    _write_canonical(output_path, approval)
    krea_execution_plan.validate_approval(
        approval, plan=plan, approval_path=output_path
    )
    return approval


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--package", required=True, type=Path)
    prepare.add_argument("--surface-record", required=True, type=Path)
    prepare.add_argument("--independent-record", required=True, type=Path)
    prepare.add_argument("--sealed-custodian-actor", required=True, type=Path)
    prepare.add_argument("--god-checkout", required=True, type=Path)
    prepare.add_argument("--god-commit", required=True)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--prepared-at-utc")

    ratify = commands.add_parser("ratify")
    ratify.add_argument("--draft", required=True, type=Path)
    ratify.add_argument("--output", required=True, type=Path)

    materialize = commands.add_parser("materialize-discovery")
    materialize.add_argument("--draft", required=True, type=Path)
    materialize.add_argument("--ratification", required=True, type=Path)
    materialize.add_argument("--public-c1c4-record", required=True, type=Path)
    materialize.add_argument("--shape-amendment", required=True, type=Path)
    materialize.add_argument("--output-dir", required=True, type=Path)
    materialize.add_argument("--materialized-at-utc")

    admit = commands.add_parser("admit-discovery")
    admit.add_argument("--materialized-bundle", required=True, type=Path)
    admit.add_argument("--blinded-acceptance", required=True, type=Path)
    admit.add_argument("--output-dir", required=True, type=Path)
    admit.add_argument("--admitted-at-utc")

    validate_inputs = commands.add_parser("validate-inputs")
    validate_inputs.add_argument("--bundle", required=True, type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("--bundle", required=True, type=Path)
    authorize = commands.add_parser("authorize-gpu")
    authorize.add_argument("--plan", required=True, type=Path)
    authorize.add_argument("--admission-envelope", required=True, type=Path)
    authorize.add_argument("--technical-actor", required=True, type=Path)
    authorize.add_argument("--output", required=True, type=Path)
    authorize.add_argument("--approved-at-utc")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "prepare":
        paths = prepare_governance(
            package_root=args.package,
            surface_record_path=args.surface_record,
            independent_record_path=args.independent_record,
            sealed_custodian_actor_path=args.sealed_custodian_actor,
            god_checkout=args.god_checkout,
            god_commit=args.god_commit,
            output_dir=args.output_dir,
            prepared_at_utc=args.prepared_at_utc or _now_utc(),
        )
        print(
            json.dumps(
                {key: str(path) for key, path in paths.items()},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if args.command == "ratify":
        value = ratify_interactively(draft_path=args.draft, output_path=args.output)
        print(
            json.dumps(
                {
                    "owner_identity": value["owner_identity"],
                    "ratification_sha256": value["ratification_sha256"],
                    "admission_authorized": value["admission_authorized"],
                    "gpu_execution_authorized": value["gpu_execution_authorized"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if args.command == "materialize-discovery":
        value = materialize_discovery_inputs(
            draft_path=args.draft,
            ratification_path=args.ratification,
            public_c1c4_record_path=args.public_c1c4_record,
            shape_amendment_path=args.shape_amendment,
            output_dir=args.output_dir,
            materialized_at_utc=args.materialized_at_utc or _now_utc(),
        )
        print(
            json.dumps(
                {
                    "bundle_sha256": value["bundle_sha256"],
                    "fixture_manifest_sha256s": value["fixture_manifest_sha256s"],
                    "admission_authorized": value["admission_authorized"],
                    "gpu_execution_authorized": value["gpu_execution_authorized"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if args.command == "admit-discovery":
        value = finalize_discovery_envelope(
            materialized_root=args.materialized_bundle,
            blinded_acceptance_path=args.blinded_acceptance,
            output_dir=args.output_dir,
            admitted_at_utc=args.admitted_at_utc or _now_utc(),
        )
        print(
            json.dumps(
                {
                    "envelope_sha256": value["envelope_sha256"],
                    "admission_authorized": value["admission_authorized"],
                    "gpu_execution_authorized": value["gpu_execution_authorized"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if args.command == "validate-inputs":
        value = validate_materialized_inputs(args.bundle)
        print(value["materialization"]["bundle_sha256"])
        return 0
    if args.command == "validate":
        value = validate_envelope(args.bundle)
        print(value["envelope"]["envelope_sha256"])
        return 0
    if args.command == "authorize-gpu":
        value = authorize_gpu_execution(
            plan_path=args.plan,
            admission_envelope_path=args.admission_envelope,
            technical_actor_path=args.technical_actor,
            output_path=args.output,
            approved_at_utc=args.approved_at_utc or _now_utc(),
        )
        print(
            json.dumps(
                {
                    "approval_sha256": value["approval_sha256"],
                    "fixture_role": value["fixture_role"],
                    "gpu_execution_authorized": value["gpu_execution_authorized"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests.
    raise SystemExit(main())
