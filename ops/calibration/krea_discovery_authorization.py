#!/usr/bin/env python3
"""Create-only authority for the immutable Week-5 discovery freeze.

The discovery freeze is intentionally non-authorizing and retains the
pre-GPU blocker text that existed when it was published.  This separate
artifact records an agent's technical execution decision under the already
sealed owner ratification for the narrow Stage-1 discovery surface.  It does
not impersonate or invent a second human reviewer.  It deliberately does not
bind the later profile index, avoiding a freeze -> timing -> profile ->
authorization -> freeze hash cycle.  The profile index and executable plans
bind this artifact in the forward direction instead.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

try:
    from . import krea_delegated_review_contract
    from . import krea_execution_surface_policy
    from . import krea_fixture
    from . import krea_provenance
except ImportError:  # pragma: no cover - direct script execution.
    import krea_delegated_review_contract  # type: ignore[no-redef]
    import krea_execution_surface_policy  # type: ignore[no-redef]
    import krea_fixture  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]


_SHA256 = re.compile(r"[0-9a-f]{64}")
_KIND = "forge-krea-discovery-execution-authorization"
_DISCOVERY_KIND = "sn56-week5-krea-discovery-freeze"
_ACTIONS = [
    "bootstrap_timing_probe",
    "profile_indexed_discovery_execution",
    "offline_exact_scoring",
    "discovery_decision_evaluation",
]
_CLAIM_LIMIT = (
    "stage1-staged-host-venv-discovery-only; not production-container, "
    "release, tournament, field-parity, or stage2 authorization"
)


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


def _utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value
    ):
        raise ValueError(f"{label} must be whole-second UTC")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"{label} is not a real UTC timestamp") from exc
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


def _json(path: str | Path, label: str) -> tuple[Path, dict[str, Any], str, bytes]:
    source = _safe_file(path, label)
    raw = source.read_bytes()
    try:
        value = _object(json.loads(raw), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    return source, value, hashlib.sha256(raw).hexdigest(), raw


def _load_discovery_binding(value: Any) -> tuple[Path, dict[str, Any], str]:
    binding = _object(value, "authorization discovery binding")
    _exact(
        binding,
        {"path", "file_sha256", "discovery_sha256"},
        "authorization discovery binding",
    )
    path, discovery, file_sha, _ = _json(binding["path"], "discovery freeze")
    blockers = discovery.get("gpu_blockers")
    if (
        discovery.get("schema") != 2
        or discovery.get("kind") != _DISCOVERY_KIND
        or discovery.get("model") != "krea/Krea-2-Raw"
        or discovery.get("model_type") != "krea2"
        or discovery.get("status") != "draft_blocked_pre_gpu"
        or discovery.get("gpu_execution_authorized") is not False
        or not isinstance(blockers, list)
        or not blockers
        or any(not isinstance(item, str) or not item.strip() for item in blockers)
    ):
        raise ValueError(
            "authorization requires the immutable blocked discovery freeze"
        )
    if file_sha != _digest(
        binding["file_sha256"], "discovery file SHA-256"
    ) or krea_provenance.canonical_sha256(discovery) != _digest(
        binding["discovery_sha256"], "discovery semantic SHA-256"
    ):
        raise ValueError("authorization discovery binding drifted")
    return path, discovery, file_sha


def _load_admission_binding(
    value: Any,
) -> tuple[Path, dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    binding = _object(value, "authorization admission binding")
    _exact(
        binding,
        {
            "path",
            "file_sha256",
            "envelope_sha256",
            "owner_ratification_sha256",
        },
        "authorization admission binding",
    )
    path, envelope, file_sha, raw = _json(binding["path"], "fixture admission envelope")
    if raw != krea_provenance.canonical_bytes(envelope) + b"\n":
        raise ValueError("fixture admission envelope must be canonical JSON")
    governance = _object(envelope.get("governance"), "admission governance")
    ratification = _object(
        governance.get("owner_ratification"), "admission owner ratification"
    )
    if (
        envelope.get("schema") != 1
        or envelope.get("kind") != "forge-krea-fixture-admission-envelope"
        or envelope.get("phase") != "discovery"
        or envelope.get("decision") != "admitted"
        or envelope.get("admission_authorized") is not True
        or envelope.get("gpu_execution_authorized") is not False
        or file_sha != _digest(binding["file_sha256"], "admission file SHA-256")
        or envelope.get("envelope_sha256")
        != _digest(binding["envelope_sha256"], "admission semantic SHA-256")
        or ratification.get("ratification_sha256")
        != _digest(
            binding["owner_ratification_sha256"],
            "owner ratification SHA-256",
        )
    ):
        raise ValueError("fixture admission binding is not admitted or has drifted")
    body = {key: item for key, item in envelope.items() if key != "envelope_sha256"}
    if envelope["envelope_sha256"] != krea_provenance.canonical_sha256(body):
        raise ValueError("fixture admission envelope self digest is invalid")
    # A self-hashed JSON object is not admission evidence.  Reopen the entire
    # bundle through the authoritative validator so package inventory,
    # reviews, ratification, D1/D2 manifests/approvals, and blinded C custody
    # are all rederived before any GPU authority can be consumed.
    try:
        from . import krea_fixture_admission
    except ImportError:  # pragma: no cover - direct script execution.
        import krea_fixture_admission  # type: ignore[no-redef]

    resolved = krea_fixture_admission.validate_envelope(path)
    ratification = _object(resolved.get("ratification"), "validated owner ratification")
    if (
        resolved.get("envelope") != envelope
        or resolved.get("envelope_file_sha256") != file_sha
        or ratification.get("ratification_sha256")
        != binding["owner_ratification_sha256"]
    ):
        raise ValueError("full admission validation differs from authorization binding")
    return path, envelope, file_sha, ratification, resolved


def _prior_admission_actor_sets(
    admission: Mapping[str, Any], admission_resolved: Mapping[str, Any]
) -> tuple[set[str], set[str]]:
    prior_actors: list[dict[str, Any]] = []
    for fixture in _object(
        admission_resolved.get("fixtures"), "admitted fixtures"
    ).values():
        fixture_governance = _object(fixture.get("governance"), "fixture governance")
        for key in ("surface_agent_review", "independent_agent_review"):
            prior_actors.append(
                krea_fixture._agent_actor(
                    _object(fixture_governance.get(key), f"fixture {key}").get("actor"),
                    f"fixture {key} actor",
                )
            )
        prior_actors.append(
            krea_fixture._agent_actor(
                fixture_governance.get("preparer_actor"), "fixture preparer actor"
            )
        )
    envelope_governance = _object(admission.get("governance"), "admission governance")
    prior_actors.extend(
        [
            krea_fixture._agent_actor(
                _object(
                    envelope_governance.get("sealed_custodian_actor"),
                    "sealed custodian binding",
                ).get("actor"),
                "sealed confirmation custodian",
            ),
            krea_fixture._agent_actor(
                admission.get("admission_producer_actor"),
                "admission producer actor",
            ),
        ]
    )
    delegated = krea_delegated_review_contract.load()["actors"].values()
    return (
        {item["actor_id"] for item in prior_actors}
        | {item["actor_id"] for item in delegated},
        {item["review_instance_id"] for item in prior_actors}
        | {item["review_instance_id"] for item in delegated},
    )


def build_payload(
    *,
    discovery_plan_path: str | Path,
    fixture_admission_envelope_path: str | Path,
    technical_reviewer_actor: dict[str, Any],
    authorized_at_utc: str,
) -> dict[str, Any]:
    """Build an agent technical decision under prior owner ratification."""

    discovery_path, discovery, discovery_file_sha, _ = _json(
        discovery_plan_path, "discovery freeze"
    )
    discovery_binding = {
        "path": str(discovery_path),
        "file_sha256": discovery_file_sha,
        "discovery_sha256": krea_provenance.canonical_sha256(discovery),
    }
    # Reuse the strict loader for status/blocker checks.
    _, discovery, _ = _load_discovery_binding(discovery_binding)

    admission_path, admission, admission_file_sha, _ = _json(
        fixture_admission_envelope_path, "fixture admission envelope"
    )
    ratification = _object(
        _object(admission.get("governance"), "admission governance").get(
            "owner_ratification"
        ),
        "admission owner ratification",
    )
    admission_binding = {
        "path": str(admission_path),
        "file_sha256": admission_file_sha,
        "envelope_sha256": admission.get("envelope_sha256"),
        "owner_ratification_sha256": ratification.get("ratification_sha256"),
    }
    _, _, _, _, admission_resolved = _load_admission_binding(admission_binding)
    actor = krea_fixture._agent_actor(
        technical_reviewer_actor, "discovery authorization technical actor"
    )
    prior_actor_ids, prior_review_instances = _prior_admission_actor_sets(
        admission, admission_resolved
    )
    if (
        actor.get("role") != "discovery_execution_authorization_reviewer"
        or actor.get("actor_id") in prior_actor_ids
        or actor.get("review_instance_id") in prior_review_instances
    ):
        raise ValueError(
            "discovery authorization requires a fresh technical review actor"
        )
    return {
        "schema": 2,
        "kind": _KIND,
        "discovery_plan": discovery_binding,
        "fixture_admission_envelope": admission_binding,
        "execution_surface_policy_sha256": krea_execution_surface_policy.POLICY[
            "policy_sha256"
        ],
        "frozen_status": discovery["status"],
        "frozen_gpu_blockers": list(discovery["gpu_blockers"]),
        "authorized_actions": list(_ACTIONS),
        "authorized_scope": "stage1_discovery_only",
        "status": "sealed_executable",
        "gpu_blockers_closed_for_authorized_scope": True,
        "gpu_execution_authorized": False,
        "technical_reviewer_actor": actor,
        "accountable_owner_identity": admission["accountable_owner_identity"],
        "authorized_at_utc": _utc(authorized_at_utc, "authorized_at_utc"),
        "claim_limit": _CLAIM_LIMIT,
    }


def seal(payload: dict[str, Any]) -> dict[str, Any]:
    payload = _object(payload, "discovery authorization payload")
    if "authorization_sha256" in payload:
        raise ValueError("unsealed discovery authorization contains a digest")
    record = {
        **payload,
        "authorization_sha256": krea_provenance.canonical_sha256(payload),
    }
    validate(record)
    return record


def validate(value: dict[str, Any]) -> dict[str, Any]:
    value = _object(value, "discovery execution authorization")
    _exact(
        value,
        {
            "schema",
            "kind",
            "discovery_plan",
            "fixture_admission_envelope",
            "execution_surface_policy_sha256",
            "frozen_status",
            "frozen_gpu_blockers",
            "authorized_actions",
            "authorized_scope",
            "status",
            "gpu_blockers_closed_for_authorized_scope",
            "gpu_execution_authorized",
            "technical_reviewer_actor",
            "accountable_owner_identity",
            "authorized_at_utc",
            "claim_limit",
            "authorization_sha256",
        },
        "discovery execution authorization",
    )
    _, discovery, _ = _load_discovery_binding(value["discovery_plan"])
    _, admission, _, ratification, admission_resolved = _load_admission_binding(
        value["fixture_admission_envelope"]
    )
    decision_bindings = _object(
        ratification.get("decision_bindings"), "ratification decision bindings"
    )
    public_evidence = _object(
        decision_bindings.get("public_source_evidence"),
        "ratified public source evidence",
    )
    surface_policy = _object(
        decision_bindings.get("stage1_execution_surface_policy"),
        "ratified Stage-1 execution surface policy",
    )
    body = {key: item for key, item in value.items() if key != "authorization_sha256"}
    if (
        value["schema"] != 2
        or value["kind"] != _KIND
        or value["execution_surface_policy_sha256"]
        != krea_execution_surface_policy.POLICY["policy_sha256"]
        or value["frozen_status"] != discovery["status"]
        or value["frozen_gpu_blockers"] != discovery["gpu_blockers"]
        or value["authorized_actions"] != _ACTIONS
        or value["authorized_scope"] != "stage1_discovery_only"
        or value["status"] != "sealed_executable"
        or value["gpu_blockers_closed_for_authorized_scope"] is not True
        or value["gpu_execution_authorized"] is not False
        or value["claim_limit"] != _CLAIM_LIMIT
        or value["accountable_owner_identity"]
        != admission.get("accountable_owner_identity")
        or public_evidence.get("discovery_plan_file_sha256")
        != value["discovery_plan"]["file_sha256"]
        or surface_policy.get("policy_sha256")
        != krea_execution_surface_policy.POLICY["policy_sha256"]
        or surface_policy.get("execution_surface") != "staged_host_venv"
        or surface_policy.get("execution_scope") != "discovery_only"
        or value["authorization_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("discovery execution authorization is invalid")
    krea_fixture.named_human(
        value["accountable_owner_identity"], "accountable_owner_identity"
    )
    actor = krea_fixture._agent_actor(
        value["technical_reviewer_actor"],
        "discovery authorization technical actor",
    )
    prior_actor_ids, prior_review_instances = _prior_admission_actor_sets(
        admission, admission_resolved
    )
    if (
        actor.get("role") != "discovery_execution_authorization_reviewer"
        or actor.get("actor_id") in prior_actor_ids
        or actor.get("review_instance_id") in prior_review_instances
    ):
        raise ValueError("discovery authorization lacks a fresh technical actor")
    authorized_at = datetime.strptime(
        _utc(value["authorized_at_utc"], "authorized_at_utc"),
        "%Y-%m-%dT%H:%M:%SZ",
    ).replace(tzinfo=timezone.utc)
    admitted_at = datetime.strptime(
        _utc(admission.get("admitted_at_utc"), "admitted_at_utc"),
        "%Y-%m-%dT%H:%M:%SZ",
    ).replace(tzinfo=timezone.utc)
    if authorized_at < admitted_at:
        raise ValueError("discovery authorization predates fixture admission")
    return value


def load_binding(value: Any) -> tuple[Path, dict[str, Any], str]:
    binding = _object(value, "discovery authorization binding")
    _exact(
        binding,
        {"path", "file_sha256", "authorization_sha256"},
        "discovery authorization binding",
    )
    path, authorization, file_sha, raw = _json(
        binding["path"], "discovery execution authorization"
    )
    if raw != krea_provenance.canonical_bytes(authorization) + b"\n":
        raise ValueError("discovery execution authorization must be canonical JSON")
    validate(authorization)
    if file_sha != _digest(
        binding["file_sha256"], "authorization file SHA-256"
    ) or authorization["authorization_sha256"] != _digest(
        binding["authorization_sha256"], "authorization semantic SHA-256"
    ):
        raise ValueError("discovery execution authorization binding drifted")
    return path, authorization, file_sha


def validate_technical_actor(
    authorization: Mapping[str, Any],
    actor_value: Any,
    *,
    required_role: str,
) -> dict[str, Any]:
    """Validate a post-ratification agent actor without inventing a human."""

    validate(dict(authorization))
    krea_execution_surface_policy.technical_role(required_role)
    actor = krea_fixture._agent_actor(actor_value, f"{required_role} actor")
    prior = krea_fixture._agent_actor(
        authorization["technical_reviewer_actor"],
        "discovery authorization technical actor",
    )
    if (
        actor.get("role") != required_role
        or actor.get("actor_id") == prior.get("actor_id")
        or actor.get("review_instance_id") == prior.get("review_instance_id")
    ):
        raise ValueError(f"{required_role} requires a fresh agent review instance")
    krea_delegated_review_contract.reject_delegated_actor_reuse(
        actor, label=f"{required_role} actor"
    )
    return actor


def assert_matches_discovery(
    authorization: Mapping[str, Any],
    *,
    discovery_path: Path,
    discovery: Mapping[str, Any],
    discovery_file_sha256: str,
    action: str,
) -> None:
    """Require one authorization to close the exact frozen sentinels in use."""

    validate(dict(authorization))
    expected = {
        "path": str(discovery_path),
        "file_sha256": discovery_file_sha256,
        "discovery_sha256": krea_provenance.canonical_sha256(discovery),
    }
    if authorization["discovery_plan"] != expected:
        raise ValueError("discovery authorization names a different freeze")
    if action not in authorization["authorized_actions"]:
        raise ValueError(f"discovery authorization does not permit {action}")
    if (
        discovery.get("status") != "draft_blocked_pre_gpu"
        or discovery.get("gpu_blockers") != authorization["frozen_gpu_blockers"]
        or authorization["gpu_blockers_closed_for_authorized_scope"] is not True
    ):
        raise ValueError("frozen discovery blockers lack exact external closure")
    # The freeze binds stable pre-governance candidate manifests.  The
    # admitted envelope binds the later governance-bearing D manifests.  Link
    # them through governance.candidate_manifest_sha256 without ever
    # back-writing the final manifest SHA into the immutable freeze.
    _, envelope, _, _, _ = _load_admission_binding(
        authorization["fixture_admission_envelope"]
    )
    candidates = _object(
        _object(envelope.get("source_package"), "admission source package").get(
            "candidate_manifest_sha256s"
        ),
        "admitted candidate manifest identities",
    )
    tasks = _object(discovery.get("discovery_tasks"), "discovery tasks")
    if set(candidates) != {"D1", "D2"}:
        raise ValueError("admission does not contain exact D1/D2 candidate identities")
    for role in ("D1", "D2"):
        task = _object(tasks.get(role), f"discovery task {role}")
        if (
            task.get("identity") != candidates[role]
            or task.get("fixture_split_manifest_sha256") != candidates[role]
        ):
            raise ValueError(
                f"discovery task {role} does not bind its admitted candidate manifest"
            )


def assert_fixture_admitted(
    authorization: Mapping[str, Any],
    *,
    role: str,
    fixture: Mapping[str, Any],
    fixture_file_sha256: str,
    fixture_approval: Mapping[str, Any],
    fixture_approval_file_sha256: str,
) -> None:
    """Bind a timing/final plan to the exact admitted D1/D2 artifacts.

    Candidate-manifest identity alone is deliberately insufficient: two later
    governance-bearing manifests can name the same candidate.  Revalidating
    the admission bundle and comparing both file and semantic identities
    closes that substitution path.
    """

    validate(dict(authorization))
    if role not in {"D1", "D2"}:
        raise ValueError("discovery fixture role must be D1 or D2")
    try:
        from . import krea_fixture_admission
    except ImportError:  # pragma: no cover - direct script execution.
        import krea_fixture_admission  # type: ignore[no-redef]

    admission_path = authorization["fixture_admission_envelope"]["path"]
    resolved = krea_fixture_admission.validate_envelope(Path(admission_path))
    envelope = _object(resolved.get("envelope"), "validated admission envelope")
    admitted_fixtures = _object(resolved.get("fixtures"), "validated admitted fixtures")
    admitted_approvals = _object(
        resolved.get("fixture_approvals"), "validated admitted fixture approvals"
    )
    envelope_slots = _object(
        envelope.get("discovery_fixtures"), "admitted discovery fixture bindings"
    )
    slot = _object(envelope_slots.get(role), f"admitted {role} fixture binding")
    manifest_binding = _object(slot.get("manifest"), f"admitted {role} manifest")
    approval_binding = _object(slot.get("approval"), f"admitted {role} approval")
    if (
        fixture.get("experimental_role") != role
        or admitted_fixtures.get(role) != dict(fixture)
        or admitted_approvals.get(role) != dict(fixture_approval)
        or manifest_binding.get("file_sha256")
        != _digest(fixture_file_sha256, f"{role} fixture file SHA-256")
        or manifest_binding.get("manifest_sha256") != fixture.get("manifest_sha256")
        or approval_binding.get("file_sha256")
        != _digest(
            fixture_approval_file_sha256, f"{role} fixture approval file SHA-256"
        )
        or approval_binding.get("approval_sha256")
        != fixture_approval.get("approval_sha256")
    ):
        raise ValueError(
            f"execution fixture/approval differs from exact admitted {role} artifacts"
        )


def publish(path: str | Path, value: dict[str, Any]) -> None:
    """Publish once with O_EXCL; an authorization is never overwritten."""

    validate(value)
    target = Path(os.path.abspath(os.path.expanduser(path)))
    target.parent.mkdir(parents=True, exist_ok=True)
    current = target.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"authorization output has a symlink ancestor: {current}")
        current = current.parent
    payload = krea_provenance.canonical_bytes(value) + b"\n"
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal_parser = sub.add_parser("seal")
    seal_parser.add_argument("--discovery-plan", required=True)
    seal_parser.add_argument("--fixture-admission-envelope", required=True)
    seal_parser.add_argument("--technical-actor", required=True)
    seal_parser.add_argument("--authorized-at-utc", required=True)
    seal_parser.add_argument("--output", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--authorization", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "seal":
        _, actor, _, raw = _json(
            args.technical_actor, "discovery authorization technical actor"
        )
        if raw != krea_provenance.canonical_bytes(actor) + b"\n":
            raise ValueError("technical actor must be canonical JSON")
        record = seal(
            build_payload(
                discovery_plan_path=args.discovery_plan,
                fixture_admission_envelope_path=args.fixture_admission_envelope,
                technical_reviewer_actor=actor,
                authorized_at_utc=args.authorized_at_utc,
            )
        )
        publish(args.output, record)
    else:
        _, record, _, raw = _json(args.authorization, "authorization")
        if raw != krea_provenance.canonical_bytes(record) + b"\n":
            raise ValueError("authorization must be canonical JSON")
        validate(record)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
