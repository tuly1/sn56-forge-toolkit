#!/usr/bin/env python3
"""Sealed pre-training Krea execution plan and human authorization.

This is stage two of the calibration evidence chain.  Source facts remain in
their own immutable records; this plan records concrete local choices.  It is
created with GPU execution disabled and becomes executable only through a
separate named-human approval that also binds a literal Linux/H100/systemd
certification record.
"""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any

try:
    from . import krea_budget
    from . import krea_c1c4_amendment
    from . import krea_dataset_identity
    from . import krea_discovery_authorization
    from . import krea_fixture
    from . import krea_host_identity
    from . import krea_internal_evidence
    from . import krea_provenance
    from . import krea_profile_index
    from . import krea_public_source
except ImportError:  # pragma: no cover - direct script execution.
    import krea_budget  # type: ignore[no-redef]
    import krea_c1c4_amendment  # type: ignore[no-redef]
    import krea_dataset_identity  # type: ignore[no-redef]
    import krea_discovery_authorization  # type: ignore[no-redef]
    import krea_fixture  # type: ignore[no-redef]
    import krea_host_identity  # type: ignore[no-redef]
    import krea_internal_evidence  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_profile_index  # type: ignore[no-redef]
    import krea_public_source  # type: ignore[no-redef]


_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_IMMUTABLE_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_PUBLIC_APPROVAL_KIND = "forge-krea-source-normalization-approval"
_AGENT_PUBLIC_REVIEW_KIND = "forge-krea-agent-source-normalization-review"
_AGENT_PUBLIC_REVIEW_CLAIM_LIMIT = (
    "agent technical review of the normalized public-recipe vocabulary and "
    "disclosed local adaptations; not human review, quality proof, selector "
    "reproduction, fixture admission, or GPU authorization"
)
_AGENT_PUBLIC_REVIEW_ASSERTIONS = {
    "canonical_manifest_valid": True,
    "public_evidence_manifest_verified": True,
    "normalized_recipe_vocabulary_reviewed": True,
    "unknown_and_unsupported_fields_reviewed": True,
    "local_adaptations_reviewed": True,
    "source_artifact_identity_reviewed": True,
    "private_selector_not_claimed_as_reproduced": True,
    "claim_limits_reviewed": True,
}
_INTERNAL_MODES = frozenset(
    {"deployed_control", "derived_matched_control", "internal_evidence_challenger"}
)
_DISCOVERY_KIND = "sn56-week5-krea-discovery-freeze"
_TIMING_PROBE_KIND = "forge-krea-bootstrap-timing-probe-plan"
_TIMING_RUNNER_BOOTSTRAP = (
    "import runpy,sys;sys.path.insert(0,'/app/forge');"
    "runpy.run_module('ops.calibration.run_krea_ladder',run_name='__main__')"
)
_TIMING_APPROVAL_KIND = "forge-krea-bootstrap-timing-probe-approval"
_EXECUTION_APPROVAL_KIND = "forge-krea-pre-run-execution-approval"
_POSTRUN_CERTIFICATE_KIND = "forge-krea-post-run-natural-completion-certificate"
_CAMPAIGN_ROOT = Path("/campaign")
_CONTROL_ROOT = _CAMPAIGN_ROOT / "controls"
_HISTORICAL_TIMING_SOURCE_COMMIT = "58822b496019177a02fa6196247ac30e788331bb"
_TIMING_SOURCE_PATHS = {
    "runner_sha256": "ops/calibration/run_krea_ladder.py",
    "measurement_tool_sha256": "ops/calibration/krea_timing_probe.py",
}
_HISTORICAL_TIMING_REPLAY_SOURCE: ContextVar[str | None] = ContextVar(
    "forge_krea_historical_timing_replay_source", default=None
)


def _historical_timing_source_identities(source_commit: str) -> dict[str, str]:
    """Hash the exact historical Git blobs that produced sealed timing evidence."""

    if source_commit != _HISTORICAL_TIMING_SOURCE_COMMIT:
        raise ValueError("historical timing source commit is not authorized")
    root = Path(__file__).resolve().parents[2]
    identities: dict[str, str] = {}
    for key, relative in _TIMING_SOURCE_PATHS.items():
        try:
            result = subprocess.run(
                [
                    "/usr/bin/git",
                    "-C",
                    str(root),
                    "show",
                    f"{source_commit}:{relative}",
                ],
                check=True,
                capture_output=True,
                env={
                    "PATH": "/usr/bin:/bin",
                    "LANG": "C",
                    "LC_ALL": "C",
                    "GIT_CONFIG_GLOBAL": "/dev/null",
                    "GIT_CONFIG_SYSTEM": "/dev/null",
                },
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("historical timing source is unavailable") from exc
        identities[key] = hashlib.sha256(result.stdout).hexdigest()
    return identities


def _replay_historical_timing(
    source_commit: str | None, callback: Any, /, *args: Any
) -> Any:
    """Scope an authenticated historical identity to archival replay only."""

    if source_commit is None:
        return callback(*args)
    if source_commit != _HISTORICAL_TIMING_SOURCE_COMMIT:
        raise ValueError("historical timing replay source is not authorized")
    active = _HISTORICAL_TIMING_REPLAY_SOURCE.get()
    if active is not None and active != source_commit:
        raise ValueError("historical timing replay source changed mid-validation")
    token = _HISTORICAL_TIMING_REPLAY_SOURCE.set(source_commit)
    try:
        return callback(*args)
    finally:
        _HISTORICAL_TIMING_REPLAY_SOURCE.reset(token)


def _strict_utc(value: Any, label: str) -> str:
    value = _text(value, label)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        raise ValueError(f"{label} must be UTC with whole-second precision")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid UTC timestamp") from exc
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _lexical_child(value: Any, root: Path) -> bool:
    if not isinstance(value, str) or not value or "\x00" in value:
        return False
    path = Path(os.path.abspath(os.path.expanduser(value)))
    return path != root and path.is_relative_to(root)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(value: dict[str, Any], required: set[str], label: str) -> None:
    missing = required - set(value)
    extra = set(value) - required
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return " ".join(value.split())


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


def _safe_directory(value: str | Path, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(value)))
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real directory: {path}")
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    return path


def _load_binding(value: Any, label: str) -> tuple[Path, dict[str, Any], str]:
    binding = _object(value, label)
    _exact(binding, {"path", "sha256"}, label)
    path = _safe_file(binding["path"], label)
    expected = _digest(binding["sha256"], f"{label}.sha256")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if raw != krea_provenance.canonical_bytes(document) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline")
    return path, _object(document, label), expected


def _file_binding(value: Any, label: str) -> tuple[Path, str]:
    binding = _object(value, label)
    _exact(binding, {"path", "sha256"}, label)
    path = _safe_file(binding["path"], label)
    digest = _digest(binding["sha256"], f"{label}.sha256")
    if krea_provenance.file_sha256(path) != digest:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path, digest


def _json_file_binding(
    value: Any, label: str, *, canonical: bool = False
) -> tuple[Path, dict[str, Any], str]:
    path, digest = _file_binding(value, label)
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    document = _object(document, label)
    if canonical and raw != krea_provenance.canonical_bytes(document) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline")
    return path, document, digest


def build_internal_basis(
    *,
    arm_id: str,
    mode: str,
    description: str,
    evidence_record: dict[str, str],
    release_commit: str,
    parent_arm_id: str | None,
) -> dict[str, Any]:
    if not _SAFE_ID.fullmatch(arm_id):
        raise ValueError("arm_id is invalid")
    if mode not in _INTERNAL_MODES:
        raise ValueError("internal basis mode is invalid")
    if mode == "derived_matched_control":
        if not isinstance(parent_arm_id, str) or not _SAFE_ID.fullmatch(parent_arm_id):
            raise ValueError("derived control requires a parent arm")
    elif parent_arm_id is not None:
        raise ValueError("only a derived control may name a parent arm")
    release_commit = _text(release_commit, "release_commit").lower()
    if not _GIT_SHA.fullmatch(release_commit):
        raise ValueError("release_commit must be a full Git commit")
    evidence_path, _, evidence_file_sha = _load_binding(
        evidence_record, "internal basis evidence record"
    )
    body = {
        "schema": 1,
        "kind": "forge-krea-internal-arm-basis",
        "arm_id": arm_id,
        "mode": mode,
        "description": _text(description, "description"),
        "evidence_record": {
            "path": str(evidence_path),
            "sha256": evidence_file_sha,
        },
        "release_commit": release_commit,
        "parent_arm_id": parent_arm_id,
    }
    return {**body, "basis_sha256": krea_provenance.canonical_sha256(body)}


def validate_internal_basis(value: dict[str, Any], *, arm_id: str) -> dict[str, Any]:
    value = _object(value, "internal arm basis")
    _exact(
        value,
        {
            "schema",
            "kind",
            "arm_id",
            "mode",
            "description",
            "evidence_record",
            "release_commit",
            "parent_arm_id",
            "basis_sha256",
        },
        "internal arm basis",
    )
    rebuilt = build_internal_basis(
        arm_id=value["arm_id"],
        mode=value["mode"],
        description=value["description"],
        evidence_record=value["evidence_record"],
        release_commit=value["release_commit"],
        parent_arm_id=value["parent_arm_id"],
    )
    if value != rebuilt or value["arm_id"] != arm_id:
        raise ValueError("internal arm basis is not canonical or arm-bound")
    return value


def _validate_source_review_owner_ratification(
    ratification: Any,
    *,
    ratification_file_sha256: str,
    portable_draft: dict[str, Any],
    portable_draft_file_sha256: str,
    amendment: dict[str, Any],
    amendment_file_sha256: str,
    custodian_actor: dict[str, Any],
    custodian_actor_file_sha256: str,
) -> dict[str, Any]:
    """Validate the minimal relocatable owner-governance authority chain.

    The create-only publisher first validates the original local draft through
    admission's complete canonical resolver.  This portable consumer then
    proves the copied ratification, portable draft, amendment, and explicitly
    non-human custodian are the exact same self-digesting chain.  Final GPU
    approval independently revalidates the complete admitted envelope.
    """

    ratification = _object(ratification, "source-review owner ratification")
    portable_draft = _object(
        portable_draft, "source-review portable ratification draft"
    )
    amendment = _object(amendment, "source-review governance amendment")

    def require_self_digest(document: dict[str, Any], key: str, label: str) -> str:
        digest = _digest(document.get(key), f"{label} semantic SHA-256")
        body = {name: value for name, value in document.items() if name != key}
        if digest != krea_provenance.canonical_sha256(body):
            raise ValueError(f"{label} semantic SHA-256 mismatch")
        return digest

    ratification_sha = require_self_digest(
        ratification, "ratification_sha256", "owner ratification"
    )
    draft_sha = require_self_digest(
        portable_draft, "draft_sha256", "portable ratification draft"
    )
    amendment_sha = require_self_digest(
        amendment, "amendment_sha256", "governance amendment"
    )
    ratification_draft = _object(
        ratification.get("portable_ratification_draft"),
        "owner ratification portable draft binding",
    )
    ratification_amendment = _object(
        ratification.get("governance_amendment"),
        "owner ratification amendment binding",
    )
    _exact(
        ratification_draft,
        {"file_sha256", "draft_sha256"},
        "owner ratification portable draft binding",
    )
    _exact(
        ratification_amendment,
        {"file_sha256", "amendment_sha256"},
        "owner ratification amendment binding",
    )
    decision_bindings = _object(
        ratification.get("decision_bindings"), "owner ratification decisions"
    )
    draft_decisions = _object(
        portable_draft.get("decision_bindings"), "portable draft decisions"
    )
    public_evidence = _object(
        decision_bindings.get("public_source_evidence"),
        "ratified public source evidence",
    )
    _exact(
        public_evidence,
        {
            "discovery_plan_file_sha256",
            "thin_manifest_file_sha256",
            "public_source_provenance",
        },
        "ratified public source evidence",
    )
    _digest(
        public_evidence["discovery_plan_file_sha256"],
        "ratified discovery plan file SHA-256",
    )
    _digest(
        public_evidence["thin_manifest_file_sha256"],
        "ratified thin manifest file SHA-256",
    )
    sources = _object(
        public_evidence["public_source_provenance"],
        "ratified public provenance bindings",
    )
    _exact(sources, {"K2", "K3", "K4"}, "ratified public provenance bindings")
    for arm, raw_binding in sources.items():
        binding = _object(raw_binding, f"ratified {arm} provenance binding")
        _exact(
            binding,
            {"file_sha256", "manifest_sha256"},
            f"ratified {arm} provenance binding",
        )
        _digest(binding["file_sha256"], f"ratified {arm} provenance file SHA-256")
        _digest(
            binding["manifest_sha256"],
            f"ratified {arm} provenance semantic SHA-256",
        )

    amendment_custodian = _object(
        amendment.get("sealed_custodian_actor"), "amendment sealed custodian"
    )
    _exact(
        amendment_custodian,
        {"file_sha256", "actor_sha256", "actor"},
        "amendment sealed custodian",
    )
    custodian = krea_fixture._agent_actor(
        custodian_actor, "source-review sealed custodian actor"
    )
    custodian_sha = krea_provenance.canonical_sha256(custodian)
    portable_evidence = _object(
        portable_draft.get("evidence_files"), "portable draft evidence files"
    )
    portable_custodian = _object(
        portable_evidence.get("sealed_custodian_actor"),
        "portable draft sealed custodian",
    )
    _exact(
        portable_custodian,
        {"file_sha256", "actor_sha256"},
        "portable draft sealed custodian",
    )
    owner = krea_fixture.named_human(
        ratification.get("owner_identity"), "source-review owner identity"
    )
    ratified_at = _strict_utc(
        ratification.get("ratified_at_utc"), "source-review ratification time"
    )
    ratified_time = datetime.strptime(ratified_at, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    evidence_times = [
        _strict_utc(
            portable_draft.get("prepared_at_utc"), "portable draft preparation time"
        ),
        _strict_utc(amendment.get("amended_at_utc"), "amendment time"),
    ]
    if (
        ratification.get("schema") != 1
        or ratification.get("kind") != "forge-krea-sole-human-owner-ratification"
        or portable_draft.get("schema") != 1
        or portable_draft.get("kind") != "forge-krea-portable-owner-ratification-draft"
        or amendment.get("schema") != 1
        or amendment.get("kind") != "forge-krea-review-governance-amendment"
        or ratification_draft
        != {
            "file_sha256": portable_draft_file_sha256,
            "draft_sha256": draft_sha,
        }
        or ratification_amendment
        != {
            "file_sha256": amendment_file_sha256,
            "amendment_sha256": amendment_sha,
        }
        or decision_bindings != draft_decisions
        or decision_bindings.get("governance_amendment_sha256") != amendment_sha
        or amendment.get("public_source_evidence") != public_evidence
        or decision_bindings.get("sealed_custodian_actor_sha256") != custodian_sha
        or amendment_custodian
        != {
            "file_sha256": custodian_actor_file_sha256,
            "actor_sha256": custodian_sha,
            "actor": custodian,
        }
        or portable_custodian
        != {
            "file_sha256": custodian_actor_file_sha256,
            "actor_sha256": custodian_sha,
        }
        or portable_draft.get("owner_identity") != owner
        or ratification.get("decision") != "ratified_for_fixture_admission_input"
        or ratification.get("admission_authorized") is not False
        or ratification.get("gpu_execution_authorized") is not False
        or portable_draft.get("admission_authorized") is not False
        or portable_draft.get("gpu_execution_authorized") is not False
        or amendment.get("admission_authorized") is not False
        or amendment.get("gpu_execution_authorized") is not False
        or any(
            ratified_time
            < datetime.strptime(item, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            for item in evidence_times
        )
    ):
        raise ValueError("portable source-review ratification chain is invalid")
    return {
        "owner_identity": owner,
        "ratification_sha256": ratification_sha,
        "ratification_file_sha256": ratification_file_sha256,
        "ratified_at_utc": ratified_at,
        "public_source_evidence": public_evidence,
        "sealed_custodian_actor_sha256": custodian_sha,
    }


def _thin_evidence_rows(path: Path) -> dict[str, str]:
    """Parse a sha256sum manifest without trusting paths or duplicate rows."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("thin evidence manifest is not UTF-8") from exc
    if not lines:
        raise ValueError("thin evidence manifest is empty")
    rows: dict[str, str] = {}
    for index, line in enumerate(lines):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\x00\r\n]+)", line)
        if match is None:
            raise ValueError(f"thin evidence manifest row {index} is malformed")
        digest, relative_text = match.groups()
        relative = Path(relative_text)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError("thin evidence manifest contains an unsafe path")
        portable = relative.as_posix()
        if portable in rows:
            raise ValueError("thin evidence manifest contains a duplicate path")
        artifact = _safe_file(path.parent / relative, "thin evidence artifact")
        if krea_provenance.file_sha256(artifact) != digest:
            raise ValueError(f"thin evidence artifact SHA-256 mismatch: {portable}")
        rows[portable] = digest
    return rows


def _review_relative_file(review_path: Path, relative_value: Any, label: str) -> Path:
    relative_text = _text(relative_value, f"{label}.relative_path")
    relative = Path(relative_text)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"{label} relative path is unsafe")
    root = review_path.parent.resolve()
    path = _safe_file(root / relative, label)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escaped the review bundle") from exc
    return path


def _portable_binding(
    *, review_path: Path, artifact_path: Path, label: str
) -> dict[str, str]:
    artifact = _safe_file(artifact_path, label)
    relative = Path(os.path.relpath(artifact, review_path.parent))
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"{label} must be inside the review bundle")
    return {
        "relative_path": relative.as_posix(),
        "file_sha256": krea_provenance.file_sha256(artifact),
    }


def _load_review_json(
    *, review_path: Path, binding: Any, label: str, semantic_key: str
) -> tuple[Path, dict[str, Any], str]:
    binding = _object(binding, f"{label} binding")
    _exact(
        binding,
        {"relative_path", "file_sha256", semantic_key},
        f"{label} binding",
    )
    path = _review_relative_file(review_path, binding["relative_path"], label)
    file_sha = _digest(binding["file_sha256"], f"{label} file SHA-256")
    if krea_provenance.file_sha256(path) != file_sha:
        raise ValueError(f"{label} file SHA-256 mismatch")
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if raw != krea_provenance.canonical_bytes(document) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline")
    return path, _object(document, label), file_sha


def build_agent_public_source_review(
    *,
    review_output_path: str | Path,
    source_provenance: dict[str, str],
    thin_evidence_manifest: dict[str, str],
    owner_ratification: dict[str, str],
    portable_ratification_draft: dict[str, str],
    governance_amendment: dict[str, str],
    sealed_custodian_actor: dict[str, str],
    actor: dict[str, Any],
    reviewed_at_utc: str,
) -> dict[str, Any]:
    """Build a non-authorizing agent review from exact bound public evidence."""

    review_path = Path(os.path.abspath(os.path.expanduser(review_output_path)))
    source_path, source, source_file_sha = _load_binding(
        source_provenance, "agent-review source provenance"
    )
    krea_provenance.validate_manifest(source)
    thin_path, thin_sha = _file_binding(
        thin_evidence_manifest, "agent-review thin evidence manifest"
    )
    owner_path, owner, owner_file_sha = _load_binding(
        owner_ratification, "agent-review owner ratification"
    )
    portable_path, portable, portable_file_sha = _load_binding(
        portable_ratification_draft,
        "agent-review portable ratification draft",
    )
    amendment_path, amendment, amendment_file_sha = _load_binding(
        governance_amendment, "agent-review governance amendment"
    )
    custodian_path, custodian, custodian_file_sha = _load_binding(
        sealed_custodian_actor, "agent-review sealed custodian actor"
    )
    owner_summary = _validate_source_review_owner_ratification(
        owner,
        ratification_file_sha256=owner_file_sha,
        portable_draft=portable,
        portable_draft_file_sha256=portable_file_sha,
        amendment=amendment,
        amendment_file_sha256=amendment_file_sha,
        custodian_actor=custodian,
        custodian_actor_file_sha256=custodian_file_sha,
    )
    source_binding = _portable_binding(
        review_path=review_path,
        artifact_path=source_path,
        label="agent-review source provenance",
    )
    expected_source_relative = (
        "public-source-provenance/"
        f"{source['source_arm_id']}-public-source-provenance.json"
    )
    if source_binding["relative_path"] != expected_source_relative:
        raise ValueError("source provenance is not at its canonical bundle path")
    thin_binding = _portable_binding(
        review_path=review_path,
        artifact_path=thin_path,
        label="agent-review thin evidence manifest",
    )
    if thin_binding["relative_path"] != "MANIFEST.sha256":
        raise ValueError("thin evidence manifest is not at the bundle root")

    def semantic_binding(
        *, path: Path, label: str, semantic_key: str, semantic_sha: str
    ) -> dict[str, str]:
        binding = _portable_binding(
            review_path=review_path, artifact_path=path, label=label
        )
        binding[semantic_key] = semantic_sha
        return binding

    body = {
        "schema": 2,
        "kind": _AGENT_PUBLIC_REVIEW_KIND,
        "source_arm_id": source["source_arm_id"],
        "source_provenance": {
            **source_binding,
            "manifest_sha256": source["manifest_sha256"],
        },
        "thin_evidence_manifest": {
            **thin_binding,
        },
        "actor": krea_fixture._agent_actor(actor, "source-normalization review actor"),
        "reviewed_at_utc": _strict_utc(
            reviewed_at_utc, "source-normalization review time"
        ),
        "decision": "technical_pass",
        "assertions": dict(_AGENT_PUBLIC_REVIEW_ASSERTIONS),
        "owner_ratification": semantic_binding(
            path=owner_path,
            label="agent-review owner ratification",
            semantic_key="ratification_sha256",
            semantic_sha=owner_summary["ratification_sha256"],
        ),
        "portable_ratification_draft": semantic_binding(
            path=portable_path,
            label="agent-review portable ratification draft",
            semantic_key="draft_sha256",
            semantic_sha=portable["draft_sha256"],
        ),
        "governance_amendment": semantic_binding(
            path=amendment_path,
            label="agent-review governance amendment",
            semantic_key="amendment_sha256",
            semantic_sha=amendment["amendment_sha256"],
        ),
        "sealed_custodian_actor": semantic_binding(
            path=custodian_path,
            label="agent-review sealed custodian actor",
            semantic_key="actor_sha256",
            semantic_sha=owner_summary["sealed_custodian_actor_sha256"],
        ),
        "agent_review_is_not_human_review": True,
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": _AGENT_PUBLIC_REVIEW_CLAIM_LIMIT,
    }
    review = {**body, "review_sha256": krea_provenance.canonical_sha256(body)}
    _validate_agent_public_review(
        review,
        source_manifest=source,
        source_manifest_file_sha256=source_file_sha,
        review_path=review_path,
    )
    return review


def _validate_agent_public_review(
    value: dict[str, Any],
    *,
    source_manifest: dict[str, Any],
    source_manifest_file_sha256: str,
    review_path: Path,
) -> dict[str, Any]:
    label = "agent source-normalization review"
    _exact(
        value,
        {
            "schema",
            "kind",
            "source_arm_id",
            "source_provenance",
            "thin_evidence_manifest",
            "actor",
            "reviewed_at_utc",
            "decision",
            "assertions",
            "owner_ratification",
            "portable_ratification_draft",
            "governance_amendment",
            "sealed_custodian_actor",
            "agent_review_is_not_human_review",
            "admission_authorized",
            "gpu_execution_authorized",
            "claim_limit",
            "review_sha256",
        },
        label,
    )
    source = _object(value["source_provenance"], "reviewed source provenance")
    _exact(
        source,
        {"relative_path", "file_sha256", "manifest_sha256"},
        "reviewed source provenance",
    )
    expected_relative = (
        "public-source-provenance/"
        f"{source_manifest['source_arm_id']}-public-source-provenance.json"
    )
    if source["relative_path"] != expected_relative:
        raise ValueError("agent review source provenance path is not canonical")
    source_file_sha = _digest(
        source["file_sha256"], "agent review source provenance file SHA-256"
    )
    source_manifest_sha = _digest(
        source["manifest_sha256"], "agent review source manifest SHA-256"
    )
    bundled_source_path = _review_relative_file(
        review_path, source["relative_path"], "agent-review source provenance"
    )
    if krea_provenance.file_sha256(bundled_source_path) != source_file_sha:
        raise ValueError("agent-review source provenance file SHA-256 mismatch")
    try:
        bundled_source_raw = bundled_source_path.read_bytes()
        bundled_source = json.loads(bundled_source_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("agent-review source provenance is not JSON") from exc
    if (
        bundled_source_raw != krea_provenance.canonical_bytes(bundled_source) + b"\n"
        or bundled_source != source_manifest
    ):
        raise ValueError("bundled source provenance differs from execution source")
    thin_binding = _object(
        value["thin_evidence_manifest"], "agent-review thin evidence binding"
    )
    _exact(
        thin_binding,
        {"relative_path", "file_sha256"},
        "agent-review thin evidence binding",
    )
    if thin_binding["relative_path"] != "MANIFEST.sha256":
        raise ValueError("agent-review thin manifest path is not canonical")
    thin_path = _review_relative_file(
        review_path,
        thin_binding["relative_path"],
        "agent-review thin evidence manifest",
    )
    thin_sha = _digest(
        thin_binding["file_sha256"], "agent-review thin manifest file SHA-256"
    )
    if krea_provenance.file_sha256(thin_path) != thin_sha:
        raise ValueError("agent-review thin evidence manifest SHA-256 mismatch")
    rows = _thin_evidence_rows(thin_path)
    _, owner, owner_file_sha = _load_review_json(
        review_path=review_path,
        binding=value["owner_ratification"],
        label="agent-review owner ratification",
        semantic_key="ratification_sha256",
    )
    _, portable, portable_file_sha = _load_review_json(
        review_path=review_path,
        binding=value["portable_ratification_draft"],
        label="agent-review portable ratification draft",
        semantic_key="draft_sha256",
    )
    _, amendment, amendment_file_sha = _load_review_json(
        review_path=review_path,
        binding=value["governance_amendment"],
        label="agent-review governance amendment",
        semantic_key="amendment_sha256",
    )
    _, custodian, custodian_file_sha = _load_review_json(
        review_path=review_path,
        binding=value["sealed_custodian_actor"],
        label="agent-review sealed custodian actor",
        semantic_key="actor_sha256",
    )
    owner_summary = _validate_source_review_owner_ratification(
        owner,
        ratification_file_sha256=owner_file_sha,
        portable_draft=portable,
        portable_draft_file_sha256=portable_file_sha,
        amendment=amendment,
        amendment_file_sha256=amendment_file_sha,
        custodian_actor=custodian,
        custodian_actor_file_sha256=custodian_file_sha,
    )
    public_evidence = owner_summary["public_source_evidence"]
    ratified_source = public_evidence["public_source_provenance"].get(
        source_manifest["source_arm_id"]
    )
    actor = krea_fixture._agent_actor(
        value["actor"], "source-normalization review actor"
    )
    reviewed_at = datetime.strptime(
        _strict_utc(value["reviewed_at_utc"], "source-normalization review time"),
        "%Y-%m-%dT%H:%M:%SZ",
    ).replace(tzinfo=timezone.utc)
    ratified_at = datetime.strptime(
        owner_summary["ratified_at_utc"], "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)
    body = {key: item for key, item in value.items() if key != "review_sha256"}
    if (
        value["schema"] != 2
        or value["kind"] != _AGENT_PUBLIC_REVIEW_KIND
        or value["source_arm_id"] != source_manifest["source_arm_id"]
        or source_file_sha != source_manifest_file_sha256
        or source_manifest_sha != source_manifest["manifest_sha256"]
        or rows.get(expected_relative) != source_manifest_file_sha256
        or public_evidence["thin_manifest_file_sha256"] != thin_sha
        or ratified_source
        != {
            "file_sha256": source_manifest_file_sha256,
            "manifest_sha256": source_manifest["manifest_sha256"],
        }
        or value["decision"] != "technical_pass"
        or value["assertions"] != _AGENT_PUBLIC_REVIEW_ASSERTIONS
        or value["owner_ratification"]["ratification_sha256"]
        != owner_summary["ratification_sha256"]
        or value["portable_ratification_draft"]["draft_sha256"]
        != portable.get("draft_sha256")
        or value["governance_amendment"]["amendment_sha256"]
        != amendment.get("amendment_sha256")
        or value["sealed_custodian_actor"]["actor_sha256"]
        != owner_summary["sealed_custodian_actor_sha256"]
        or reviewed_at < ratified_at
        or value["agent_review_is_not_human_review"] is not True
        or value["admission_authorized"] is not False
        or value["gpu_execution_authorized"] is not False
        or value["claim_limit"] != _AGENT_PUBLIC_REVIEW_CLAIM_LIMIT
        or value["review_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("agent source-normalization review is invalid or unbound")
    return {
        "schema": 2,
        "decision": "technical_pass",
        "review_sha256": value["review_sha256"],
        "actor": actor,
        "owner_identity": owner_summary["owner_identity"],
        "owner_ratification_sha256": owner_summary["ratification_sha256"],
        "sealed_custodian_actor_sha256": owner_summary["sealed_custodian_actor_sha256"],
        "public_source_evidence": public_evidence,
        "claim_limit": _AGENT_PUBLIC_REVIEW_CLAIM_LIMIT,
    }


def _validate_public_approval(
    value: dict[str, Any],
    *,
    source_manifest: dict[str, Any],
    source_manifest_file_sha256: str,
    approval_path: Path | None = None,
) -> dict[str, Any]:
    if value.get("schema") == 2:
        if approval_path is None:
            raise ValueError("schema-2 source review requires its bundle file path")
        return _validate_agent_public_review(
            value,
            source_manifest=source_manifest,
            source_manifest_file_sha256=source_manifest_file_sha256,
            review_path=approval_path,
        )
    _exact(
        value,
        {
            "schema",
            "kind",
            "source_arm_id",
            "provenance_manifest_sha256",
            "reviewer_identity",
            "decision",
            "assertions",
        },
        "source-normalization approval",
    )
    if (
        value["schema"] != 1
        or value["kind"] != _PUBLIC_APPROVAL_KIND
        or value["source_arm_id"] != source_manifest["source_arm_id"]
        or value["provenance_manifest_sha256"] != source_manifest["manifest_sha256"]
        or value["decision"] != "approved"
    ):
        raise ValueError("source-normalization approval does not bind the source")
    krea_fixture.named_human(value["reviewer_identity"], "reviewer_identity")
    assertions = _object(value["assertions"], "source approval assertions")
    _exact(
        assertions,
        {
            "source_fields_reviewed",
            "unsupported_fields_reviewed",
            "adaptations_reviewed",
            "source_artifact_identity_reviewed",
            "claim_limits_reviewed",
        },
        "source approval assertions",
    )
    if any(item is not True for item in assertions.values()):
        raise ValueError("source-normalization approval assertions did not all pass")
    return {
        "schema": 1,
        "decision": "approved",
        "reviewer_identity": value["reviewer_identity"],
    }


def _arm_basis(
    value: Any, *, arm_id: str, execution_recipe: dict[str, Any]
) -> dict[str, Any]:
    basis = _object(value, "arm basis")
    mode = basis.get("mode")
    if mode == "public_submission":
        _exact(
            basis,
            {
                "mode",
                "source_provenance",
                "source_normalization_approval",
                "source_files",
            },
            "public arm basis",
        )
        _, source, source_file_sha = _load_binding(
            basis["source_provenance"], "source provenance"
        )
        source_files = _object(basis["source_files"], "public source files")
        _exact(
            source_files,
            {
                "source_config",
                "source_artifact",
                "field_ledger",
                "task_raw",
                "tournament_raw",
                "revision_manifest",
            },
            "public source files",
        )
        rebound: dict[str, tuple[Path, str]] = {
            name: _file_binding(binding, f"public source {name}")
            for name, binding in source_files.items()
        }
        # A self-digesting provenance JSON is not proof of its semantic linkage.
        # Re-run the primary-source validators against every bound byte source;
        # otherwise a fabricated semantic_linkage block could be self-hashed and
        # accepted without the official task/tournament/HF observations.
        krea_provenance.validate_manifest(
            source,
            source_config_path=rebound["source_config"][0],
            source_artifact_path=rebound["source_artifact"][0],
            field_ledger_path=rebound["field_ledger"][0],
            task_raw_path=rebound["task_raw"][0],
            tournament_raw_path=rebound["tournament_raw"][0],
            revision_manifest_path=rebound["revision_manifest"][0],
        )
        # ``validate_manifest`` proves that the manifest is internally
        # canonical and linked to the official records.  It cannot, by itself,
        # prove that a self-hashed normalized recipe was actually parsed from
        # the bound YAML/safetensors bytes.  Re-derive the machine-owned
        # semantics from those primary files and require exact agreement.  A
        # later human review assertion is intentionally separate and is the
        # only manifest field allowed to differ from the machine-produced
        # unreviewed record.
        derived_metadata = krea_public_source.build_metadata(
            arm_id,
            source_config_path=rebound["source_config"][0],
            source_artifact_path=rebound["source_artifact"][0],
            field_ledger_path=rebound["field_ledger"][0],
        )
        derived = krea_provenance.build_manifest(
            derived_metadata,
            source_config_path=rebound["source_config"][0],
            source_artifact_path=rebound["source_artifact"][0],
            field_ledger_path=rebound["field_ledger"][0],
            task_raw_path=rebound["task_raw"][0],
            tournament_raw_path=rebound["tournament_raw"][0],
            revision_manifest_path=rebound["revision_manifest"][0],
        )
        semantic_keys = {
            "schema",
            "kind",
            "source_arm_id",
            "source",
            "official_context",
            "files",
            "fields",
            "evaluator_sha",
            "matched_concept",
            "adaptation_target",
            "local_reproduction_disclosure",
            "normalized_recipe",
        }
        mismatches = sorted(
            key for key in semantic_keys if source.get(key) != derived.get(key)
        )
        if mismatches:
            raise ValueError(
                "public source manifest differs from primary-byte re-derivation: "
                f"{mismatches}"
            )
        if source["source_arm_id"] != arm_id:
            raise ValueError("public source arm id differs from execution arm")
        source_approval_path, source_approval, source_approval_sha = _load_binding(
            basis["source_normalization_approval"], "source approval"
        )
        source_approval_summary = _validate_public_approval(
            source_approval,
            source_manifest=source,
            source_manifest_file_sha256=source_file_sha,
            approval_path=source_approval_path,
        )
        normalized = krea_provenance.normalize_execution_recipe(
            execution_recipe, source_recipe=source["normalized_recipe"]
        )
        return {
            "mode": mode,
            "source_provenance_file_sha256": source_file_sha,
            "source_manifest_sha256": source["manifest_sha256"],
            "source_normalization_approval_sha256": source_approval_sha,
            "source_normalization_approval": source_approval_summary,
            "rebound_source_files": {
                name: {"path": str(path), "sha256": digest}
                for name, (path, digest) in sorted(rebound.items())
            },
            "normalized_execution_recipe": normalized,
        }
    if mode == "internal":
        expected_keys = {"mode", "basis_record"}
        if arm_id == "K5":
            expected_keys.add("project_root")
        _exact(basis, expected_keys, "internal arm basis binding")
        _, record, record_file_sha = _load_binding(
            basis["basis_record"], "internal arm basis record"
        )
        validate_internal_basis(record, arm_id=arm_id)
        normalized = krea_provenance.normalize_recipe(execution_recipe)
        result = {
            "mode": mode,
            "basis_record_file_sha256": record_file_sha,
            "basis_sha256": record["basis_sha256"],
            "basis_mode": record["mode"],
            "normalized_execution_recipe": normalized,
        }
        if arm_id == "K5":
            project_root = _safe_directory(
                basis["project_root"], "K5 evidence project root"
            )
            evidence_path, evidence, evidence_file_sha = _load_binding(
                record["evidence_record"], "K5 internal evidence record"
            )
            krea_internal_evidence.validate_record(evidence, project_root=project_root)
            anchor = krea_internal_evidence.build_anchor(
                record_path=evidence_path, project_root=project_root
            )
            if anchor["record_file_sha256"] != evidence_file_sha:
                raise ValueError("K5 evidence record binding differs from its anchor")
            result["K5_internal_evidence_anchor"] = anchor
        return result
    raise ValueError("arm basis mode must be public_submission or internal")


def _effective_recipe_values(recipe: dict[str, Any]) -> dict[str, Any]:
    fields = _object(recipe.get("fields"), "normalized recipe fields")
    return {
        name: _object(row, f"recipe field {name}").get("effective_value")
        for name, row in fields.items()
    }


def _discovery_allowed_axes(arm: dict[str, Any]) -> list[str]:
    arm_id = arm.get("id")
    if arm_id == "K0":
        return []
    if arm_id == "K1":
        return ["planned_steps", "save_cadence"]
    if arm_id in {"K2", "K4"}:
        return ["planned_steps", "save_cadence"]
    if arm_id == "K3":
        return ["dropout", "ema", "planned_steps", "save_cadence"]
    if arm_id == "K5":
        return ["learning_rate"]
    raise ValueError(f"discovery plan contains unsupported arm {arm_id!r}")


def validate_discovery_semantics(
    binding: Any,
    *,
    arm_id: str,
    fixture_id: str,
    fixture_manifest_sha256: str,
    fixture_candidate_manifest_sha256: str | None = None,
    training_pair_count: int,
    seed_role: str,
    seed: int,
    throughput_equivalence_class: str,
    execution_recipe: dict[str, Any],
    schedule_mode: str,
    predeclared_recipe_axes: list[str],
    basis_mode: str,
) -> dict[str, Any]:
    """Parse and bind the frozen experiment design, not just its file hash."""

    path, discovery, file_sha = _json_file_binding(binding, "discovery plan")
    if (
        discovery.get("schema") != 2
        or discovery.get("kind") != _DISCOVERY_KIND
        or discovery.get("model") != "krea/Krea-2-Raw"
        or discovery.get("model_type") != "krea2"
        or discovery.get("gpu_execution_authorized") is not False
    ):
        raise ValueError("unsupported or execution-enabled discovery plan")
    # A plan-level digest binding is insufficient if its repo-local amendment
    # has disappeared or changed since the plan was written.  Validate the
    # artifact itself on both timing-probe and execution-plan load paths.
    krea_c1c4_amendment.validate_bound_plan_amendment(discovery)
    tasks = _object(discovery.get("discovery_tasks"), "discovery tasks")
    if fixture_id not in {"D1", "D2"} or fixture_id not in tasks:
        raise ValueError("discovery fixture id must be D1 or D2")
    task = _object(tasks[fixture_id], f"discovery task {fixture_id}")
    frozen_fixture_identity = (
        fixture_candidate_manifest_sha256
        if fixture_candidate_manifest_sha256 is not None
        else fixture_manifest_sha256
    )
    if task.get("fixture_split_manifest_sha256") != frozen_fixture_identity:
        raise ValueError("fixture manifest is not the one frozen in discovery plan")
    pair_range = task.get("required_training_pair_range")
    if (
        not isinstance(pair_range, list)
        or len(pair_range) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) for item in pair_range
        )
        or pair_range[0] <= 0
        or pair_range[0] > pair_range[1]
        or not pair_range[0] <= training_pair_count <= pair_range[1]
    ):
        raise ValueError("fixture training-pair count escaped discovery range")
    if task.get("identity") is None:
        raise ValueError("discovery fixture identity is still unset")
    if (
        fixture_candidate_manifest_sha256 is not None
        and task.get("identity") != fixture_candidate_manifest_sha256
    ):
        raise ValueError(
            "schema-2 discovery identity must be its stable candidate manifest"
        )

    expected_seed_key = {
        "A": "training_seed_a",
        "B": "training_seed_b_contingency",
    }.get(seed_role)
    if expected_seed_key is None or discovery.get(expected_seed_key) != seed:
        raise ValueError("execution seed does not match its frozen discovery role")
    arms = discovery.get("arms")
    if not isinstance(arms, list):
        raise ValueError("discovery arms must be an array")
    matches = [row for row in arms if isinstance(row, dict) and row.get("id") == arm_id]
    if len(matches) != 1:
        raise ValueError("execution arm is not unique in the discovery plan")
    arm = matches[0]
    if arm.get("throughput_equivalence_class") != throughput_equivalence_class:
        raise ValueError("execution arm escaped its frozen throughput class")
    expected_basis = "public_submission" if arm_id in {"K2", "K3", "K4"} else "internal"
    if basis_mode != expected_basis:
        raise ValueError("arm basis mode contradicts the frozen arm source")
    expected_mode = "release_control" if arm_id == "K0" else "measured_budget_fill"
    if schedule_mode != expected_mode:
        raise ValueError("schedule mode contradicts the frozen arm depth policy")
    expected_axes = _discovery_allowed_axes(arm)
    if predeclared_recipe_axes != expected_axes:
        raise ValueError(
            "execution axes differ from the frozen arm contract: "
            f"expected={expected_axes}, actual={predeclared_recipe_axes}"
        )

    values = _effective_recipe_values(execution_recipe)
    field_map = {
        "learning_rate": "lr",
        "rank": "rank",
        "alpha": "alpha",
        "optimizer": "optimizer",
        "loss": "loss",
        "dropout": "dropout",
    }
    for recipe_name, arm_name in field_map.items():
        expected = arm.get(arm_name)
        if expected is None and arm_id == "K3" and arm_name == "dropout":
            expected = _object(
                arm.get("predeclared_local_values"), "K3 local values"
            ).get("dropout")
        if expected is None or values.get(recipe_name) != expected:
            raise ValueError(
                f"recipe field {recipe_name} contradicts discovery arm {arm_id}"
            )
    expected_guidance = arm.get("guidance")
    guidance = values.get("guidance")
    if guidance != {"enabled": True, "scale": expected_guidance}:
        raise ValueError("recipe guidance contradicts the discovery arm")
    expected_ema = arm.get("ema")
    if expected_ema is None and arm_id == "K3":
        expected_ema = _object(
            arm.get("predeclared_local_values"), "K3 local values"
        ).get("ema")
    ema = values.get("ema")
    if (
        not isinstance(ema, dict)
        or ema.get("enabled") is not expected_ema
        or set(ema) != {"enabled", "decay"}
    ):
        raise ValueError("recipe EMA contradicts the discovery arm")
    if arm_id == "K4":
        optimizer_parameters = values.get("optimizer_parameters")
        expected_parameters = {
            "min_lr": arm["min_lr"],
            "max_lr": arm["max_lr"],
            "lr_bump": arm["lr_bump"],
        }
        if not isinstance(optimizer_parameters, dict) or any(
            optimizer_parameters.get(key) != expected
            for key, expected in expected_parameters.items()
        ):
            raise ValueError("Automagic parameters contradict the frozen K4 arm")
    if arm_id == "K5":
        krea_internal_evidence.validate_anchor(arm.get("internal_evidence_anchor"))
    return {
        "path": path,
        "file_sha256": file_sha,
        "document": discovery,
        "arm": arm,
        "allowed_axes": expected_axes,
        "fixture": task,
        "seed_role": seed_role,
    }


def _schedule(
    value: Any,
    *,
    recipe: dict[str, Any],
    budget_plan: dict[str, Any],
    profile: krea_budget.ThroughputProfile,
    accelerated_cell: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schedule = _object(value, "schedule")
    _exact(
        schedule,
        {
            "mode",
            "planned_steps",
            "save_every",
            "candidate_steps",
            "required_landmarks",
            "landmark_policy",
        },
        "schedule",
    )
    if schedule["mode"] not in {"release_control", "measured_budget_fill"}:
        raise ValueError("unsupported schedule mode")
    for key in ("planned_steps", "save_every"):
        if (
            isinstance(schedule[key], bool)
            or not isinstance(schedule[key], int)
            or schedule[key] <= 0
        ):
            raise ValueError(f"schedule.{key} must be a positive integer")
    planned = schedule["planned_steps"]
    cadence = schedule["save_every"]
    expected_candidates = list(range(cadence, planned, cadence)) + [planned]
    if schedule["candidate_steps"] != expected_candidates:
        raise ValueError("candidate steps do not match uniform save cadence plus final")
    landmarks = schedule["required_landmarks"]
    if (
        not isinstance(landmarks, list)
        or any(
            isinstance(step, bool) or not isinstance(step, int) or step <= 0
            for step in landmarks
        )
        or landmarks != sorted(set(landmarks))
    ):
        raise ValueError("schedule landmarks are invalid")
    if schedule["landmark_policy"] not in {"none", "preserve_if_budget_safe"}:
        raise ValueError("unsupported landmark policy")
    if schedule["landmark_policy"] == "none" and landmarks:
        raise ValueError("landmarks require preserve_if_budget_safe")
    if any(step <= planned and step not in expected_candidates for step in landmarks):
        raise ValueError("a budget-safe required landmark is absent from candidates")
    fields = recipe["fields"]
    if (
        fields["planned_steps"]["effective_value"] != planned
        or fields["save_cadence"]["effective_value"] != cadence
    ):
        raise ValueError("schedule contradicts execution recipe")
    maximum = budget_plan.get("max_affordable_steps")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise ValueError("budget plan lacks max_affordable_steps")
    if schedule["mode"] == "measured_budget_fill" and planned != maximum:
        raise ValueError("budget-fill schedule does not fill the measured plan")
    if schedule["mode"] == "measured_budget_fill":
        cadence_multiplier = (
            accelerated_cell["cadence_multiplier"]
            if accelerated_cell is not None
            else 1
        )
        expected_cadence = budget_plan.get("save_every") * cadence_multiplier
        expected_budget_candidates = (
            [row["step"] for row in budget_plan.get("actual_candidates", [])]
            if cadence_multiplier == 1
            else list(range(expected_cadence, planned, expected_cadence))
            + [planned]
        )
        if (
            cadence != expected_cadence
            or expected_candidates != expected_budget_candidates
        ):
            raise ValueError(
                "budget-fill schedule differs from its sealed cadence multiplier"
            )
    if schedule["mode"] == "release_control" and planned > maximum:
        raise ValueError("release-control schedule does not fit the measured plan")
    if schedule["mode"] == "release_control" and accelerated_cell is not None:
        release_baseline = {"D1": (260, 53), "D2": (367, 74)}
        expected_depth, base_cadence = release_baseline[
            accelerated_cell["fixture_id"]
        ]
        if (
            accelerated_cell["arm_id"] != "K0"
            or planned != expected_depth
            or cadence
            != base_cadence * accelerated_cell["cadence_multiplier"]
        ):
            raise ValueError("accelerated release-control depth/cadence drifted")
    # A release-control arm can intentionally retain a historical depth and
    # cadence.  Charge *that actual cadence* rather than assuming the discovery
    # planner's ceil(steps/8) schedule.  This closes the K0 under-accounting
    # path where many extra saves could otherwise pass planned<=maximum.
    periodic_save_count = len(range(cadence, planned + 1, cadence))
    hard = float(budget_plan["hard_budget_s"])
    available = (
        hard
        - profile.startup_upper_bound_s
        - profile.selection_scoring_reserve_s
        - max(
            profile.framework_stop_boundary_s,
            profile.finalization_reserve_s + profile.upload_reserve_s,
        )
    )
    charged = (
        planned * profile.update_upper_bound_s
        + periodic_save_count * profile.save_upper_bound_s
    )
    if not math.isfinite(available) or charged > available:
        raise ValueError("actual release schedule exceeds the measured stop boundary")
    maximum_save_fraction = float(
        budget_plan["accounting"]["maximum_save_overhead_fraction"]
    )
    save_fraction = (
        periodic_save_count * profile.save_upper_bound_s / available
        if available > 0
        else math.inf
    )
    if save_fraction > maximum_save_fraction:
        raise ValueError("actual release cadence exceeds the sealed save-I/O cap")
    return dict(schedule)


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    plan = _object(plan, "execution plan")
    plan_schema = plan.get("schema")
    plan_keys = {
        "schema",
        "kind",
        "arm_id",
        "task_id",
        "expected_repo_name",
        "discovery_plan",
        "discovery_fixture_id",
        "seed_role",
        "fixture_manifest",
        "fixture_approval",
        "training_archive",
        "evaluation_dataset",
        "arm_basis",
        "execution_recipe",
        "throughput_profile",
        "timing_evidence",
        "host_execution_manifest",
        "budget_plan",
        "budget_plan_sha256",
        "schedule",
        "base_model",
        "seed",
        "runtime_identity_sha256",
        "execution_envelope_sha256",
        "throughput_equivalence_class",
        "predeclared_recipe_axes",
        "in_task_proxy_selection",
        "runner_sha256",
        "gpu_execution_authorized",
        "plan_sha256",
    }
    if plan_schema == 3:
        plan_keys.update(
            {
                "discovery_profile_index",
                "discovery_execution_authorization",
            }
        )
    _exact(
        plan,
        plan_keys,
        "execution plan",
    )
    if (
        plan_schema not in {2, 3}
        or plan["kind"] != "forge-krea-pretraining-execution-plan"
    ):
        raise ValueError("unsupported execution plan")
    for key in ("arm_id", "task_id", "expected_repo_name"):
        if not isinstance(plan[key], str) or not _SAFE_ID.fullmatch(plan[key]):
            raise ValueError(f"execution plan {key} is invalid")
    if plan["gpu_execution_authorized"] is not False:
        raise ValueError("execution plan itself must keep GPU authorization false")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan["plan_sha256"] != krea_provenance.canonical_sha256(body):
        raise ValueError("execution plan digest mismatch")

    discovery_authorization = None
    discovery_authorization_file_sha = None
    probe_margin_policy = None
    probe_margin_policy_file_sha = None
    if plan_schema == 3:
        _, discovery_authorization, discovery_authorization_file_sha = (
            krea_discovery_authorization.load_binding(
                plan["discovery_execution_authorization"]
            )
        )

    _, fixture, fixture_file_sha = _load_binding(
        plan["fixture_manifest"], "fixture manifest"
    )
    krea_fixture.validate_manifest(fixture)
    if fixture.get("schema") == 2 and plan_schema != 3:
        raise ValueError(
            "schema-2 discovery fixtures require a schema-3 profile-index-bound plan"
        )
    _, fixture_approval, fixture_approval_file_sha = _load_binding(
        plan["fixture_approval"], "fixture approval"
    )
    krea_fixture.validate_approval(fixture_approval, fixture_manifest=fixture)
    if plan_schema == 3:
        krea_discovery_authorization.assert_fixture_admitted(
            discovery_authorization,
            role=fixture["experimental_role"],
            fixture=fixture,
            fixture_file_sha256=fixture_file_sha,
            fixture_approval=fixture_approval,
            fixture_approval_file_sha256=fixture_approval_file_sha,
        )
    archive_path, archive_sha = _file_binding(
        plan["training_archive"], "training archive"
    )
    if (
        archive_sha != fixture["training_archive"]["sha256"]
        or archive_path.stat().st_size != fixture["training_archive"]["bytes"]
    ):
        raise ValueError("training archive differs from the approved fixture")
    evaluation = _object(plan["evaluation_dataset"], "evaluation dataset")
    _exact(evaluation, {"path", "sha256"}, "evaluation dataset")
    evaluation_path = _safe_directory(evaluation["path"], "evaluation dataset")
    evaluation_sha = _digest(evaluation["sha256"], "evaluation dataset sha256")
    expected_identity = fixture["evaluation_dataset_identity"]
    observed_identity = krea_dataset_identity.capture_dataset(
        evaluation_path,
        list_supported_images=lambda _root, _extensions: list(
            expected_identity["evaluator_order"]
        ),
        extensions=tuple(fixture["tool_identity"]["extensions"]),
    )
    if (
        observed_identity != expected_identity
        or evaluation_sha != expected_identity["sha256"]
    ):
        raise ValueError("evaluation dataset differs from the approved fixture")
    normalized_basis = _arm_basis(
        plan["arm_basis"],
        arm_id=plan["arm_id"],
        execution_recipe=plan["execution_recipe"],
    )
    recipe = normalized_basis["normalized_execution_recipe"]
    profile_path, profile_sha = _file_binding(
        plan["throughput_profile"], "throughput profile"
    )
    try:
        profile = json.loads(profile_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("throughput profile is not JSON") from exc
    if profile.get("profile_sha256") is None:
        raise ValueError("throughput profile lacks its self digest")
    validated_profile = krea_budget.load_throughput_profile(profile)
    profile_index = None
    accelerated_campaign = None
    accelerated_cell = None
    historical_host_manifest = None
    if plan_schema == 3:
        profile_index = krea_profile_index.validate_plan_cell(
            plan,
            fixture=fixture,
            throughput_profile=profile,
        )
        accelerated_campaign = profile_index.get("accelerated_campaign")
        if accelerated_campaign is not None:
            accelerated_cell = accelerated_campaign["cell"]
            if (
                accelerated_cell["fixture_id"] != plan["discovery_fixture_id"]
                or accelerated_cell["arm_id"] != plan["arm_id"]
                or accelerated_cell["throughput_equivalence_class"]
                != plan["throughput_equivalence_class"]
                or accelerated_cell["cadence_multiplier"] not in {1, 2}
            ):
                raise ValueError(
                    "execution plan escaped its accelerated campaign cell"
                )
    timing_evidence = _object(plan["timing_evidence"], "timing evidence")
    _exact(
        timing_evidence,
        {
            "raw_sample_manifest",
            "margin_policy",
            "end_to_end_validation",
            "probe_contract",
            "measurement_captures",
            "heldout_captures",
            "heldout_run_records",
        },
        "timing evidence",
    )
    _, raw_samples, raw_samples_file_sha = _load_binding(
        timing_evidence["raw_sample_manifest"], "raw timing sample manifest"
    )
    _, margin_policy, margin_policy_file_sha = _load_binding(
        timing_evidence["margin_policy"], "timing margin policy"
    )
    normalized_margin_policy = krea_budget.load_margin_policy(margin_policy)
    if plan_schema == 3 and (
        normalized_margin_policy.get("schema") != 2
        or normalized_margin_policy.get("kind")
        != "forge-krea-agent-predeclared-timing-margin-policy"
        or normalized_margin_policy["discovery_execution_authorization"].get(
            "authorization_sha256"
        )
        != discovery_authorization["authorization_sha256"]
    ):
        raise ValueError(
            "schema-3 execution requires its authorization-bound delegated margin"
        )
    _, end_to_end, end_to_end_file_sha = _load_binding(
        timing_evidence["end_to_end_validation"], "end-to-end timing validation"
    )
    _, probe_contract, probe_contract_file_sha = _load_binding(
        timing_evidence["probe_contract"], "timing probe contract"
    )
    historical_probe_source_commit = None
    if accelerated_campaign is not None:
        historical_probe_source_commit = accelerated_campaign["document"][
            "historical_compatibility"
        ]["source_commit"]
    validate_timing_probe_plan(
        probe_contract,
        historical_source_commit=historical_probe_source_commit,
    )
    try:
        from . import krea_timing_probe
    except ImportError:  # pragma: no cover - direct script execution.
        import krea_timing_probe  # type: ignore[no-redef]

    def evidence_array(name: str) -> list[tuple[dict[str, Any], str]]:
        bindings = timing_evidence[name]
        if not isinstance(bindings, list) or not bindings:
            raise ValueError(f"timing evidence {name} must be non-empty")
        rows: list[tuple[dict[str, Any], str]] = []
        for index, binding in enumerate(bindings):
            _, document, file_sha = _load_binding(
                binding, f"timing evidence {name}[{index}]"
            )
            rows.append((document, file_sha))
        return rows

    measurement_capture_rows = evidence_array("measurement_captures")
    heldout_capture_rows = evidence_array("heldout_captures")
    heldout_run_rows = evidence_array("heldout_run_records")
    recomputed_raw = _replay_historical_timing(
        historical_probe_source_commit,
        krea_timing_probe.raw_from_captures,
        [document for document, _ in measurement_capture_rows],
    )
    recomputed_e2e = _replay_historical_timing(
        historical_probe_source_commit,
        krea_timing_probe.end_to_end_from_records,
        [document for document, _ in heldout_capture_rows],
        heldout_run_rows,
    )
    timing_approval_actors: list[dict[str, Any]] = []
    for capture, _ in measurement_capture_rows + heldout_capture_rows:
        approval, _ = krea_timing_probe._load_canonical(
            Path(capture["probe_approval"]["path"]), "timing probe approval"
        )
        if approval.get("schema") == 2:
            timing_approval_actors.append(
                krea_fixture._agent_actor(
                    approval.get("technical_reviewer_actor"),
                    "timing probe approval actor",
                )
            )
    distinct_timing_actors = {
        krea_provenance.canonical_sha256(actor) for actor in timing_approval_actors
    }
    if plan_schema == 3 and len(distinct_timing_actors) != 1:
        raise ValueError(
            "schema-3 final plan requires one common agent timing approval"
        )
    if recomputed_raw != raw_samples or recomputed_e2e != end_to_end:
        raise ValueError("timing summaries differ from their bound producer records")
    recomputed_profile = krea_budget.seal_throughput_profile_from_evidence(
        raw_sample_manifest=raw_samples,
        margin_policy=margin_policy,
        end_to_end_validation=end_to_end,
        framework_stop_boundary_s=profile["framework_stop_boundary_s"],
        framework_stop_boundary_source_sha256=profile[
            "framework_stop_boundary_source_sha256"
        ],
        selection_mode=profile["selection_mode"],
        selection_scorer_identity_sha256=profile["selection_scorer_identity_sha256"],
        selection_scoring_reserve_s=profile["selection_scoring_reserve_s"],
    )
    if recomputed_profile != profile:
        raise ValueError(
            "throughput profile was not recomputed from bound raw evidence"
        )
    _, host_manifest, host_manifest_file_sha = _load_binding(
        plan["host_execution_manifest"], "host execution manifest"
    )
    krea_host_identity.validate_manifest(host_manifest)
    if plan_schema == 3 and host_manifest.get("schema") != 3:
        raise ValueError(
            "schema-3 execution plans require a bootstrap-receipt-bound host manifest"
        )
    if profile_sha != plan["throughput_profile"]["sha256"]:
        raise ValueError("throughput profile binding mismatch")
    budget_plan = _object(plan["budget_plan"], "budget plan")
    if plan["budget_plan_sha256"] != krea_provenance.canonical_sha256(budget_plan):
        raise ValueError("budget plan digest mismatch")
    if budget_plan.get("profile_sha256") != profile.get("profile_sha256"):
        raise ValueError("budget plan is not bound to the throughput profile")
    try:
        recomputed_budget = krea_budget.plan_budget(
            validated_profile, hard_budget_s=float(budget_plan["hard_budget_s"])
        ).to_record()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("budget plan cannot be recomputed") from exc
    if recomputed_budget != budget_plan:
        raise ValueError("budget plan differs from the measured planner output")
    if accelerated_cell is not None and (
        float(budget_plan["hard_budget_s"])
        != float(accelerated_cell["effective_hard_budget_s"])
    ):
        raise ValueError("accelerated cell budget differs from its sealed envelope")
    if (
        profile.get("selection_mode") != "offline_post_training"
        or profile.get("selection_scoring_reserve_s") != 0
    ):
        raise ValueError("discovery profile must reserve no in-task proxy scorer")
    if plan["in_task_proxy_selection"] != {"enabled": False, "reserve_s": 0}:
        raise ValueError("discovery execution must disable in-task proxy selection")
    for key in (
        "runtime_identity_sha256",
        "execution_envelope_sha256",
        "runner_sha256",
    ):
        _digest(plan[key], key)
    profile_envelope = _object(
        profile.get("execution_envelope"), "throughput execution envelope"
    )
    if (
        profile_envelope.get("runtime_identity_sha256")
        != plan["runtime_identity_sha256"]
    ):
        raise ValueError("profile/runtime identity mismatch")
    # New measured profiles must carry the complete execution envelope.  Old
    # Day-0 profiles fail closed rather than being silently reinterpreted.
    if (
        profile_envelope.get("execution_envelope_sha256")
        != plan["execution_envelope_sha256"]
    ):
        raise ValueError("profile/execution-envelope mismatch")
    expected_profile_class = (
        accelerated_campaign["document"]["measured_profile"][
            "throughput_equivalence_class"
        ]
        if accelerated_campaign is not None
        else plan["throughput_equivalence_class"]
    )
    if profile_envelope.get("equivalence_class") != expected_profile_class:
        raise ValueError("profile throughput source class mismatch")
    if validated_profile.execution_envelope.to_record() != profile_envelope:
        raise ValueError("throughput execution envelope did not normalize exactly")
    if accelerated_campaign is not None:
        historical_binding = accelerated_campaign["document"][
            "historical_host_execution_manifest"
        ]
        _, historical_host_manifest, historical_file_sha = _json_file_binding(
            {
                "path": historical_binding["path"],
                "sha256": historical_binding["file_sha256"],
            },
            "historical host execution manifest",
            canonical=True,
        )
        krea_host_identity.validate_manifest(historical_host_manifest)
        if (
            historical_file_sha != historical_binding["file_sha256"]
            or historical_host_manifest["host_execution_identity_sha256"]
            != historical_binding["host_execution_identity_sha256"]
            or profile_envelope.get("host_execution_identity_sha256")
            != historical_host_manifest["host_execution_identity_sha256"]
            or host_manifest["host_execution_identity_sha256"]
            == historical_host_manifest["host_execution_identity_sha256"]
        ):
            raise ValueError("historical host compatibility binding is invalid")
    elif (
        profile_envelope.get("host_execution_identity_sha256")
        != host_manifest["host_execution_identity_sha256"]
    ):
        raise ValueError("profile/host execution identity mismatch")
    bootstrap_runtime = None
    bootstrap_execution_surface = None
    if host_manifest.get("schema") == 3:
        bootstrap_execution_surface = krea_host_identity.bootstrap_execution_surface(
            host_manifest, recapture=False
        )
        bootstrap_runtime = bootstrap_execution_surface["runtime"]
        if (
            profile_envelope.get("execution_surface") != "staged_host_venv"
            or profile_envelope.get("execution_scope") != "discovery_only"
            or profile_envelope.get("venv_tree_manifest_sha256")
            != bootstrap_execution_surface["venv_tree"]["manifest_sha256"]
            or profile_envelope.get("reference_container_image_sha256")
            != bootstrap_runtime["container_image_sha256"]
            or profile_envelope.get("jit_enabled")
            is not bootstrap_runtime["jit_enabled"]
        ):
            raise ValueError(
                "Stage-1 profile surface/runtime identity differs from the bootstrap receipt"
            )
    if float(profile.get("framework_stop_boundary_s", -1)) < 225.0:
        raise ValueError("profile does not cover Forge's 225-second stop boundary")
    _schedule(
        plan["schedule"],
        recipe=recipe,
        budget_plan=budget_plan,
        profile=validated_profile,
        accelerated_cell=accelerated_cell,
    )
    seed = plan["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("execution seed is invalid")
    base = _object(plan["base_model"], "base_model")
    _exact(
        base,
        {
            "model_id",
            "revision",
            "training_identity_sha256",
            "evaluation_assets",
        },
        "base_model",
    )
    if (
        base["model_id"] != "krea/Krea-2-Raw"
        or not isinstance(base["revision"], str)
        or not _IMMUTABLE_REVISION.fullmatch(base["revision"])
    ):
        raise ValueError("base model identity is not immutable Krea")
    _digest(base["training_identity_sha256"], "base training identity")
    assets = _object(base["evaluation_assets"], "base evaluation assets")
    if set(assets) != {"diffusion_model", "text_encoder", "vae"}:
        raise ValueError("base assets must bind diffusion model, text encoder, and VAE")
    for name, asset in assets.items():
        asset = _object(asset, f"base asset {name}")
        _exact(asset, {"canonical_path", "sha256", "bytes"}, f"base asset {name}")
        _text(asset["canonical_path"], f"base asset {name}.canonical_path")
        _digest(asset["sha256"], f"base asset {name}.sha256")
        if (
            isinstance(asset["bytes"], bool)
            or not isinstance(asset["bytes"], int)
            or asset["bytes"] <= 0
        ):
            raise ValueError(f"base asset {name}.bytes is invalid")
    fields = recipe["fields"]
    values = {name: row["effective_value"] for name, row in fields.items()}
    if values["submitted_step"] is not None or values["selector"] is not None:
        raise ValueError("pretraining plan may not choose a checkpoint or selector")
    envelope = validated_profile.execution_envelope
    guidance = values["guidance"]
    expected_profile_fields = {
        "network_rank": values["rank"],
        "network_alpha": values["alpha"],
        "optimizer": values["optimizer"],
        "optimizer_config_sha256": krea_provenance.canonical_sha256(
            values["optimizer_parameters"]
        ),
        "loss": values["loss"],
        "differential_guidance_enabled": guidance["enabled"],
        "guidance_scale": guidance["scale"],
        "training_pair_count": len(fixture["training_rows"]),
        "training_dataset_shape_sha256": fixture["training_dataset_shape_sha256"],
        "gradient_accumulation_steps": values["gradient_accumulation"],
        "data_parallel_replicas": 1,
        "base_model_identity_sha256": base["training_identity_sha256"],
    }
    denominator = values["gradient_accumulation"]
    micro_batch = values["effective_batch"] // denominator
    if micro_batch * denominator != values["effective_batch"]:
        raise ValueError("execution effective batch is not integral")
    expected_profile_fields["micro_batch_size"] = micro_batch
    mismatches = {
        key: {"expected": expected, "profile": getattr(envelope, key)}
        for key, expected in expected_profile_fields.items()
        if getattr(envelope, key) != expected
    }
    proxy_mismatch_fields: list[str] = []
    if mismatches and accelerated_cell is None:
        raise ValueError(f"recipe/fixture/base escaped measured profile: {mismatches}")
    if accelerated_cell is not None:
        proxy_mismatch_fields = sorted(mismatches)
    runner_path = Path(__file__).with_name("run_krea_ladder.py").resolve(strict=True)
    if krea_provenance.file_sha256(runner_path) != plan["runner_sha256"]:
        raise ValueError("execution plan runner SHA differs from local runner")
    axes = plan["predeclared_recipe_axes"]
    if (
        not isinstance(axes, list)
        or axes != sorted(set(axes))
        or any(axis not in recipe["fields"] for axis in axes)
        or any(axis in {"submitted_step", "selector"} for axis in axes)
    ):
        raise ValueError("predeclared recipe axes are invalid")
    basis_mode = _object(plan["arm_basis"], "arm basis").get("mode")
    discovery = validate_discovery_semantics(
        plan["discovery_plan"],
        arm_id=plan["arm_id"],
        fixture_id=plan["discovery_fixture_id"],
        fixture_manifest_sha256=fixture["manifest_sha256"],
        fixture_candidate_manifest_sha256=(
            _object(fixture.get("governance"), "fixture governance").get(
                "candidate_manifest_sha256"
            )
            if fixture.get("schema") == 2
            else None
        ),
        training_pair_count=len(fixture["training_rows"]),
        seed_role=plan["seed_role"],
        seed=seed,
        throughput_equivalence_class=plan["throughput_equivalence_class"],
        execution_recipe=recipe,
        schedule_mode=plan["schedule"]["mode"],
        predeclared_recipe_axes=axes,
        basis_mode=basis_mode,
    )
    if plan_schema == 3:
        krea_discovery_authorization.assert_matches_discovery(
            discovery_authorization,
            discovery_path=discovery["path"],
            discovery=discovery["document"],
            discovery_file_sha256=discovery["file_sha256"],
            action="profile_indexed_discovery_execution",
        )
    if (
        plan["arm_id"] == "K5"
        and normalized_basis["K5_internal_evidence_anchor"]
        != discovery["arm"]["internal_evidence_anchor"]
    ):
        raise ValueError("K5 execution basis differs from the frozen evidence anchor")
    expected_probe_class = (
        accelerated_campaign["document"]["measured_profile"][
            "throughput_equivalence_class"
        ]
        if accelerated_campaign is not None
        else plan["throughput_equivalence_class"]
    )
    expected_probe_fixture = (
        accelerated_campaign["document"]["measured_profile"]["fixture_id"]
        if accelerated_campaign is not None
        else plan["discovery_fixture_id"]
    )
    if (
        probe_contract["probe_contract_sha256"] != raw_samples["probe_contract_sha256"]
        or probe_contract["probe_contract_sha256"]
        != end_to_end["probe_contract_sha256"]
        or probe_contract["throughput_equivalence_class"]
        != expected_probe_class
        or probe_contract["discovery_fixture_id"] != expected_probe_fixture
        or probe_contract["execution_envelope"]["execution_envelope_sha256"]
        != plan["execution_envelope_sha256"]
        or (
            plan_schema == 3
            and probe_contract.get("margin_policy")
            != plan["timing_evidence"]["margin_policy"]
        )
    ):
        raise ValueError("final execution plan escaped its bootstrap timing probe")
    return {
        "fixture": fixture,
        "fixture_manifest_file_sha256": fixture_file_sha,
        "arm_basis": normalized_basis,
        "execution_recipe": recipe,
        "training_archive_path": archive_path,
        "evaluation_dataset_path": evaluation_path,
        "throughput_profile_path": profile_path,
        "throughput_profile": profile,
        "host_execution_manifest": host_manifest,
        "host_execution_manifest_file_sha256": host_manifest_file_sha,
        "host_bootstrap_receipt": host_manifest.get("bootstrap_receipt"),
        "bootstrap_runtime": bootstrap_runtime,
        "bootstrap_execution_surface": bootstrap_execution_surface,
        "discovery_profile_index": profile_index,
        "accelerated_discovery_campaign": (
            accelerated_campaign
            if accelerated_campaign is not None
            else None
        ),
        "historical_host_execution_manifest": historical_host_manifest,
        "accelerated_proxy_mismatch_fields": proxy_mismatch_fields,
        "discovery_execution_authorization": (
            {
                "document": discovery_authorization,
                "file_sha256": discovery_authorization_file_sha,
                "authorization_sha256": discovery_authorization["authorization_sha256"],
            }
            if discovery_authorization is not None
            else None
        ),
        "discovery": discovery,
        "timing_probe_approval_actor": (
            timing_approval_actors[0] if timing_approval_actors else None
        ),
        "timing_evidence": {
            "raw_sample_manifest_file_sha256": raw_samples_file_sha,
            "margin_policy_file_sha256": margin_policy_file_sha,
            "end_to_end_validation_file_sha256": end_to_end_file_sha,
            "probe_contract_file_sha256": probe_contract_file_sha,
            "measurement_capture_file_sha256": [
                digest for _, digest in measurement_capture_rows
            ],
            "heldout_capture_file_sha256": [
                digest for _, digest in heldout_capture_rows
            ],
            "heldout_run_record_file_sha256": [
                digest for _, digest in heldout_run_rows
            ],
        },
        "schedule": plan["schedule"],
    }


def seal_plan(payload: dict[str, Any]) -> dict[str, Any]:
    if "plan_sha256" in payload:
        raise ValueError("unsealed plan payload must not contain plan_sha256")
    plan = {**payload, "plan_sha256": krea_provenance.canonical_sha256(payload)}
    validate_plan(plan)
    return plan


def validate_timing_probe_plan(
    plan: dict[str, Any], *, historical_source_commit: str | None = None
) -> dict[str, Any]:
    """Validate the executable pre-profile probe contract.

    This contract intentionally has no throughput profile, budget-derived arm
    depth, or post-run certificate.  It is the acyclic first-GPU entry point:
    approved fixture + representative recipe + host capability -> raw timings.
    """

    plan = _object(plan, "timing probe plan")
    probe_schema = plan.get("schema")
    probe_keys = {
        "schema",
        "kind",
        "arm_id",
        "task_id",
        "expected_repo_name",
        "discovery_plan",
        "discovery_fixture_id",
        "seed_role",
        "seed",
        "fixture_manifest",
        "fixture_approval",
        "training_archive",
        "arm_basis",
        "execution_recipe",
        "host_execution_manifest",
        "base_model",
        "runtime_identity_sha256",
        "execution_envelope",
        "throughput_equivalence_class",
        "predeclared_recipe_axes",
        "probe_schedule",
        "command_argv",
        "runner_sha256",
        "measurement_tool_sha256",
        "gpu_execution_authorized",
        "probe_contract_sha256",
    }
    if probe_schema == 2:
        probe_keys.update({"discovery_execution_authorization", "margin_policy"})
    _exact(
        plan,
        probe_keys,
        "timing probe plan",
    )
    if (
        probe_schema not in {1, 2}
        or plan["kind"] != _TIMING_PROBE_KIND
        or plan["gpu_execution_authorized"] is not False
    ):
        raise ValueError("unsupported or self-authorized timing probe plan")
    body = {key: value for key, value in plan.items() if key != "probe_contract_sha256"}
    if plan["probe_contract_sha256"] != krea_provenance.canonical_sha256(body):
        raise ValueError("timing probe contract digest mismatch")
    for key in ("arm_id", "task_id", "expected_repo_name"):
        if not isinstance(plan[key], str) or not _SAFE_ID.fullmatch(plan[key]):
            raise ValueError(f"timing probe {key} is invalid")
    seed = plan["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("timing probe seed is invalid")
    _, fixture, fixture_file_sha = _load_binding(
        plan["fixture_manifest"], "probe fixture manifest"
    )
    krea_fixture.validate_manifest(fixture)
    if fixture.get("schema") == 2 and probe_schema != 2:
        raise ValueError(
            "schema-2 timing fixtures require a discovery-authorization-bound probe"
        )
    discovery_authorization = None
    discovery_authorization_file_sha = None
    if probe_schema == 2:
        _, discovery_authorization, discovery_authorization_file_sha = (
            krea_discovery_authorization.load_binding(
                plan["discovery_execution_authorization"]
            )
        )
        _, probe_margin_policy, probe_margin_policy_file_sha = _load_binding(
            plan["margin_policy"], "timing probe margin policy"
        )
        normalized_probe_margin = krea_budget.load_margin_policy(probe_margin_policy)
        if (
            normalized_probe_margin.get("schema") != 2
            or normalized_probe_margin.get("kind")
            != "forge-krea-agent-predeclared-timing-margin-policy"
            or normalized_probe_margin["discovery_execution_authorization"].get(
                "authorization_sha256"
            )
            != discovery_authorization["authorization_sha256"]
        ):
            raise ValueError(
                "schema-2 timing probe requires its authorization-bound delegated margin"
            )
    _, fixture_approval, fixture_approval_file_sha = _load_binding(
        plan["fixture_approval"], "probe fixture approval"
    )
    krea_fixture.validate_approval(fixture_approval, fixture_manifest=fixture)
    if probe_schema == 2:
        krea_discovery_authorization.assert_fixture_admitted(
            discovery_authorization,
            role=fixture["experimental_role"],
            fixture=fixture,
            fixture_file_sha256=fixture_file_sha,
            fixture_approval=fixture_approval,
            fixture_approval_file_sha256=fixture_approval_file_sha,
        )
    archive_path, archive_sha = _file_binding(
        plan["training_archive"], "probe training archive"
    )
    if (
        archive_sha != fixture["training_archive"]["sha256"]
        or archive_path.stat().st_size != fixture["training_archive"]["bytes"]
    ):
        raise ValueError("probe training archive differs from approved fixture")
    normalized_basis = _arm_basis(
        plan["arm_basis"],
        arm_id=plan["arm_id"],
        execution_recipe=plan["execution_recipe"],
    )
    recipe = normalized_basis["normalized_execution_recipe"]
    _, host, _ = _load_binding(
        plan["host_execution_manifest"], "probe host execution manifest"
    )
    krea_host_identity.validate_manifest(host)
    if fixture.get("schema") == 2 and host.get("schema") != 3:
        raise ValueError(
            "schema-2 timing probes require a bootstrap-receipt-bound host manifest"
        )
    envelope = krea_budget.load_execution_envelope(plan["execution_envelope"])
    if (
        envelope.equivalence_class != plan["throughput_equivalence_class"]
        or envelope.host_execution_identity_sha256
        != host["host_execution_identity_sha256"]
        or envelope.runtime_identity_sha256 != plan["runtime_identity_sha256"]
        or envelope.execution_envelope_sha256
        != plan["execution_envelope"]["execution_envelope_sha256"]
    ):
        raise ValueError("timing probe execution envelope is not host/runtime bound")
    bootstrap_runtime = None
    bootstrap_execution_surface = None
    if host.get("schema") == 3:
        bootstrap_execution_surface = krea_host_identity.bootstrap_execution_surface(
            host, recapture=False
        )
        bootstrap_runtime = bootstrap_execution_surface["runtime"]
        if (
            envelope.execution_surface != "staged_host_venv"
            or envelope.execution_scope != "discovery_only"
            or envelope.venv_tree_manifest_sha256
            != bootstrap_execution_surface["venv_tree"]["manifest_sha256"]
            or envelope.reference_container_image_sha256
            != bootstrap_runtime["container_image_sha256"]
            or envelope.jit_enabled is not bootstrap_runtime["jit_enabled"]
        ):
            raise ValueError(
                "timing probe Stage-1 surface/runtime differs from bootstrap receipt"
            )
    if (
        envelope.training_pair_count != len(fixture["training_rows"])
        or envelope.training_dataset_shape_sha256
        != fixture["training_dataset_shape_sha256"]
    ):
        raise ValueError(
            "timing probe execution envelope escaped the approved fixture shape"
        )
    base = _object(plan["base_model"], "probe base model")
    _exact(
        base,
        {"model_id", "revision", "training_identity_sha256", "evaluation_assets"},
        "probe base model",
    )
    if (
        base["model_id"] != "krea/Krea-2-Raw"
        or not isinstance(base["revision"], str)
        or not _IMMUTABLE_REVISION.fullmatch(base["revision"])
        or envelope.base_model_identity_sha256 != base["training_identity_sha256"]
    ):
        raise ValueError("timing probe base model identity is invalid")
    _digest(base["training_identity_sha256"], "probe base training identity")
    assets = _object(base["evaluation_assets"], "probe base evaluation assets")
    if set(assets) != {"diffusion_model", "text_encoder", "vae"}:
        raise ValueError("probe base assets are incomplete")
    for name, value in assets.items():
        value = _object(value, f"probe base asset {name}")
        _exact(value, {"canonical_path", "sha256", "bytes"}, f"probe base asset {name}")
        _text(value["canonical_path"], f"probe base asset {name}.canonical_path")
        _digest(value["sha256"], f"probe base asset {name}.sha256")
        if (
            isinstance(value["bytes"], bool)
            or not isinstance(value["bytes"], int)
            or value["bytes"] <= 0
        ):
            raise ValueError(f"probe base asset {name}.bytes is invalid")

    values = _effective_recipe_values(recipe)
    schedule = _object(plan["probe_schedule"], "timing probe schedule")
    _exact(
        schedule,
        {
            "planned_steps",
            "save_every",
            "startup_repetitions",
            "hard_budget_s",
            "measurement_role",
        },
        "timing probe schedule",
    )
    planned = schedule["planned_steps"]
    cadence = schedule["save_every"]
    repetitions = schedule["startup_repetitions"]
    hard_budget = schedule["hard_budget_s"]
    if (
        isinstance(planned, bool)
        or not isinstance(planned, int)
        or planned < 100
        or isinstance(cadence, bool)
        or not isinstance(cadence, int)
        or cadence <= 0
        or len(list(range(cadence, planned, cadence)) + [planned]) < 8
        or isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 3
        or isinstance(hard_budget, bool)
        or not isinstance(hard_budget, (int, float))
        or not math.isfinite(float(hard_budget))
        or float(hard_budget) <= 0
        or schedule["measurement_role"] != "timing_and_heldout"
    ):
        raise ValueError("timing probe schedule cannot satisfy sample requirements")
    if values.get("planned_steps") != planned or values.get("save_cadence") != cadence:
        raise ValueError("timing probe schedule contradicts its recipe")
    axes = plan["predeclared_recipe_axes"]
    if not isinstance(axes, list) or axes != sorted(set(axes)):
        raise ValueError("timing probe axes must be a sorted unique list")
    discovery = validate_discovery_semantics(
        plan["discovery_plan"],
        arm_id=plan["arm_id"],
        fixture_id=plan["discovery_fixture_id"],
        fixture_manifest_sha256=fixture["manifest_sha256"],
        fixture_candidate_manifest_sha256=(
            _object(fixture.get("governance"), "probe fixture governance").get(
                "candidate_manifest_sha256"
            )
            if fixture.get("schema") == 2
            else None
        ),
        training_pair_count=len(fixture["training_rows"]),
        seed_role=plan["seed_role"],
        seed=seed,
        throughput_equivalence_class=plan["throughput_equivalence_class"],
        execution_recipe=recipe,
        schedule_mode=(
            "release_control" if plan["arm_id"] == "K0" else "measured_budget_fill"
        ),
        predeclared_recipe_axes=axes,
        basis_mode=_object(plan["arm_basis"], "probe arm basis").get("mode"),
    )
    if probe_schema == 2:
        krea_discovery_authorization.assert_matches_discovery(
            discovery_authorization,
            discovery_path=discovery["path"],
            discovery=discovery["document"],
            discovery_file_sha256=discovery["file_sha256"],
            action="bootstrap_timing_probe",
        )
    if (
        plan["arm_id"] == "K5"
        and normalized_basis["K5_internal_evidence_anchor"]
        != discovery["arm"]["internal_evidence_anchor"]
    ):
        raise ValueError("K5 probe basis differs from the frozen evidence anchor")
    command = plan["command_argv"]
    if (
        not isinstance(command, list)
        or not command
        or any(
            not isinstance(item, str) or not item or "\x00" in item for item in command
        )
    ):
        raise ValueError("timing probe command argv is invalid")
    runner_path = Path(__file__).with_name("run_krea_ladder.py").resolve(strict=True)
    tool_path = Path(__file__).with_name("krea_timing_probe.py").resolve(strict=True)
    if (
        len(command) != 10
        or command[0] != "/usr/bin/python3"
        or command[1] != "-I"
        or command[2] != "-c"
        or command[3] != _TIMING_RUNNER_BOOTSTRAP
        or command[4] != "--timing-probe-plan"
        or command[6] != "--timing-probe-approval"
        or command[8] != "--campaign-dir"
        or not _lexical_child(command[5], _CONTROL_ROOT)
        or not _lexical_child(command[7], _CONTROL_ROOT)
        or not _lexical_child(command[9], _CAMPAIGN_ROOT)
    ):
        raise ValueError(
            "timing probe command must be the bounded run_krea_ladder bootstrap argv"
        )
    effective_historical_source = (
        historical_source_commit
        if historical_source_commit is not None
        else _HISTORICAL_TIMING_REPLAY_SOURCE.get()
    )
    expected_code_identities = (
        {
            "runner_sha256": krea_provenance.file_sha256(runner_path),
            "measurement_tool_sha256": krea_provenance.file_sha256(tool_path),
        }
        if effective_historical_source is None
        else _historical_timing_source_identities(effective_historical_source)
    )
    if (
        expected_code_identities["runner_sha256"] != plan["runner_sha256"]
        or expected_code_identities["measurement_tool_sha256"]
        != plan["measurement_tool_sha256"]
        or envelope.measurement_tool_sha256 != plan["measurement_tool_sha256"]
    ):
        raise ValueError(
            "timing probe code identity differs from local producer/runner"
        )
    return {
        "fixture": fixture,
        "arm_basis": normalized_basis,
        "execution_recipe": recipe,
        "host_execution_manifest": host,
        "bootstrap_runtime": bootstrap_runtime,
        "bootstrap_execution_surface": bootstrap_execution_surface,
        "execution_envelope": envelope,
        "discovery": discovery,
        "discovery_execution_authorization": (
            {
                "document": discovery_authorization,
                "file_sha256": discovery_authorization_file_sha,
                "authorization_sha256": discovery_authorization["authorization_sha256"],
            }
            if discovery_authorization is not None
            else None
        ),
        "margin_policy": (
            {
                "document": probe_margin_policy,
                "file_sha256": probe_margin_policy_file_sha,
            }
            if probe_margin_policy is not None
            else None
        ),
        "training_archive_path": archive_path,
    }


def seal_timing_probe_plan(payload: dict[str, Any]) -> dict[str, Any]:
    if "probe_contract_sha256" in payload:
        raise ValueError("unsealed timing probe payload has a digest")
    plan = {
        **payload,
        "probe_contract_sha256": krea_provenance.canonical_sha256(payload),
    }
    validate_timing_probe_plan(plan)
    return plan


def build_timing_probe_approval(
    plan: dict[str, Any],
    *,
    reviewer_identity: str | None,
    approved_at_utc: str,
    technical_reviewer_actor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = validate_timing_probe_plan(plan)
    if plan.get("schema") == 2:
        if reviewer_identity is not None:
            raise ValueError(
                "agent-governed timing probes use a technical actor; "
                "reviewer_identity must be omitted"
            )
        if technical_reviewer_actor is None:
            raise ValueError("schema-2 timing probes require a fresh technical actor")
        governance = _object(
            resolved["fixture"].get("governance"), "probe fixture governance"
        )
        independent = _object(
            governance.get("independent_agent_review"),
            "probe independent agent review",
        )
        prior_actor = krea_fixture._agent_actor(
            independent.get("actor"), "fixture independent reviewer actor"
        )
        authorization = resolved["discovery_execution_authorization"]
        actor = krea_discovery_authorization.validate_technical_actor(
            authorization["document"],
            technical_reviewer_actor,
            required_role="timing_probe_execution_reviewer",
        )
        if actor.get("actor_id") == prior_actor.get("actor_id") or actor.get(
            "review_instance_id"
        ) == prior_actor.get("review_instance_id"):
            raise ValueError("timing approval requires a fresh technical review actor")
        body = {
            "schema": 2,
            "kind": _TIMING_APPROVAL_KIND,
            "probe_contract_sha256": plan["probe_contract_sha256"],
            "host_execution_identity_sha256": resolved["host_execution_manifest"][
                "host_execution_identity_sha256"
            ],
            "discovery_execution_authorization": {
                "file_sha256": authorization["file_sha256"],
                "authorization_sha256": authorization["authorization_sha256"],
            },
            "technical_reviewer_actor": actor,
            "accountable_owner_identity": authorization["document"][
                "accountable_owner_identity"
            ],
            "owner_ratification_sha256": authorization["document"][
                "fixture_admission_envelope"
            ]["owner_ratification_sha256"],
            "approved_at_utc": _strict_utc(approved_at_utc, "approved_at_utc"),
            "decision": "approved",
            "gpu_execution_authorized": True,
            "assertions": {
                "host_capability_reviewed": True,
                "fixture_and_recipe_reviewed": True,
                "agent_review_evidence_bound": True,
                "agents_are_not_humans": True,
                "owner_ratification_bound": True,
                "timing_only_no_production_mutation": True,
                "natural_completion_will_be_certified_post_run": True,
            },
        }
        return {**body, "approval_sha256": krea_provenance.canonical_sha256(body)}
    if reviewer_identity is None:
        raise ValueError("legacy timing probes require a named-human reviewer")
    body = {
        "schema": 1,
        "kind": _TIMING_APPROVAL_KIND,
        "probe_contract_sha256": plan["probe_contract_sha256"],
        "host_execution_identity_sha256": resolved["host_execution_manifest"][
            "host_execution_identity_sha256"
        ],
        "reviewer_identity": krea_fixture.named_human(
            reviewer_identity, "reviewer_identity"
        ),
        "approved_at_utc": _strict_utc(approved_at_utc, "approved_at_utc"),
        "decision": "approved",
        "gpu_execution_authorized": True,
        "assertions": {
            "host_capability_reviewed": True,
            "fixture_and_recipe_reviewed": True,
            "timing_only_no_production_mutation": True,
            "natural_completion_will_be_certified_post_run": True,
        },
    }
    return {**body, "approval_sha256": krea_provenance.canonical_sha256(body)}


def validate_timing_probe_approval(
    approval: dict[str, Any], *, plan: dict[str, Any]
) -> dict[str, Any]:
    resolved = validate_timing_probe_plan(plan)
    approval = _object(approval, "timing probe approval")
    if approval.get("schema") == 2:
        _exact(
            approval,
            {
                "schema",
                "kind",
                "probe_contract_sha256",
                "host_execution_identity_sha256",
                "discovery_execution_authorization",
                "technical_reviewer_actor",
                "accountable_owner_identity",
                "owner_ratification_sha256",
                "approved_at_utc",
                "decision",
                "gpu_execution_authorized",
                "assertions",
                "approval_sha256",
            },
            "agent-governed timing probe approval",
        )
        authorization = resolved.get("discovery_execution_authorization")
        if authorization is None:
            raise ValueError("schema-2 timing approval requires bound authorization")
        governance = _object(
            resolved["fixture"].get("governance"), "probe fixture governance"
        )
        actor = krea_discovery_authorization.validate_technical_actor(
            authorization["document"],
            approval["technical_reviewer_actor"],
            required_role="timing_probe_execution_reviewer",
        )
        prior_actor = krea_fixture._agent_actor(
            _object(
                governance.get("independent_agent_review"),
                "probe independent agent review",
            ).get("actor"),
            "fixture independent reviewer actor",
        )
        expected_assertions = {
            "host_capability_reviewed": True,
            "fixture_and_recipe_reviewed": True,
            "agent_review_evidence_bound": True,
            "agents_are_not_humans": True,
            "owner_ratification_bound": True,
            "timing_only_no_production_mutation": True,
            "natural_completion_will_be_certified_post_run": True,
        }
        body = {
            key: value for key, value in approval.items() if key != "approval_sha256"
        }
        if (
            approval["kind"] != _TIMING_APPROVAL_KIND
            or approval["probe_contract_sha256"] != plan["probe_contract_sha256"]
            or approval["host_execution_identity_sha256"]
            != resolved["host_execution_manifest"]["host_execution_identity_sha256"]
            or approval["discovery_execution_authorization"]
            != {
                "file_sha256": authorization["file_sha256"],
                "authorization_sha256": authorization["authorization_sha256"],
            }
            or actor.get("actor_id") == prior_actor.get("actor_id")
            or actor.get("review_instance_id") == prior_actor.get("review_instance_id")
            or approval["accountable_owner_identity"]
            != authorization["document"]["accountable_owner_identity"]
            or approval["owner_ratification_sha256"]
            != authorization["document"]["fixture_admission_envelope"][
                "owner_ratification_sha256"
            ]
            or approval["decision"] != "approved"
            or approval["gpu_execution_authorized"] is not True
            or approval["assertions"] != expected_assertions
            or approval["approval_sha256"] != krea_provenance.canonical_sha256(body)
        ):
            raise ValueError("agent-governed timing approval does not authorize probe")
        krea_fixture.named_human(
            approval["accountable_owner_identity"], "accountable_owner_identity"
        )
        approved_at = datetime.strptime(
            _strict_utc(approval["approved_at_utc"], "approved_at_utc"),
            "%Y-%m-%dT%H:%M:%SZ",
        ).replace(tzinfo=timezone.utc)
        authorized_at = datetime.strptime(
            _strict_utc(
                authorization["document"]["authorized_at_utc"],
                "authorization authorized_at_utc",
            ),
            "%Y-%m-%dT%H:%M:%SZ",
        ).replace(tzinfo=timezone.utc)
        if approved_at < authorized_at:
            raise ValueError("timing approval predates discovery authorization")
        return approval
    _exact(
        approval,
        {
            "schema",
            "kind",
            "probe_contract_sha256",
            "host_execution_identity_sha256",
            "reviewer_identity",
            "approved_at_utc",
            "decision",
            "gpu_execution_authorized",
            "assertions",
            "approval_sha256",
        },
        "timing probe approval",
    )
    body = {key: value for key, value in approval.items() if key != "approval_sha256"}
    expected_assertions = {
        "host_capability_reviewed": True,
        "fixture_and_recipe_reviewed": True,
        "timing_only_no_production_mutation": True,
        "natural_completion_will_be_certified_post_run": True,
    }
    if (
        approval["schema"] != 1
        or approval["kind"] != _TIMING_APPROVAL_KIND
        or approval["probe_contract_sha256"] != plan["probe_contract_sha256"]
        or approval["host_execution_identity_sha256"]
        != resolved["host_execution_manifest"]["host_execution_identity_sha256"]
        or approval["decision"] != "approved"
        or approval["gpu_execution_authorized"] is not True
        or approval["assertions"] != expected_assertions
        or approval["approval_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("timing probe approval does not authorize this probe")
    krea_fixture.named_human(approval["reviewer_identity"], "reviewer_identity")
    _strict_utc(approval["approved_at_utc"], "approved_at_utc")
    return approval


def build_approval(
    plan: dict[str, Any],
    *,
    reviewer_identity: str | None,
    approved_at_utc: str,
    admission_envelope_path: str | Path | None = None,
    approval_output_path: str | Path | None = None,
    technical_reviewer_actor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a pre-run approval without demanding evidence from the future."""

    resolved = validate_plan(plan)
    if resolved.get("fixture", {"schema": 1}).get("schema") == 2:
        if reviewer_identity is not None:
            raise ValueError(
                "schema-2 fixtures use an explicit technical agent; "
                "reviewer_identity must be omitted"
            )
        if admission_envelope_path is None or approval_output_path is None:
            raise ValueError(
                "schema-2 fixtures require an admission envelope and approval output path"
            )
        try:
            from . import krea_fixture_admission
        except ImportError:  # pragma: no cover - direct script execution.
            import krea_fixture_admission  # type: ignore[no-redef]

        envelope_path = _safe_file(
            admission_envelope_path, "fixture admission envelope"
        )
        envelope_resolved = krea_fixture_admission.validate_envelope(envelope_path)
        source_review = resolved.get("arm_basis", {}).get(
            "source_normalization_approval"
        )
        if (
            source_review is not None
            and source_review.get("schema") == 2
            and source_review.get("owner_ratification_sha256")
            != envelope_resolved["ratification"]["ratification_sha256"]
        ):
            raise ValueError(
                "source review owner ratification differs from fixture admission"
            )
        if technical_reviewer_actor is None:
            raise ValueError("schema-2 fixtures require an explicit technical agent")
        discovery_authorization = resolved.get("discovery_execution_authorization")
        if discovery_authorization is None:
            raise ValueError("schema-4 approval requires discovery authorization")
        technical_actor = krea_discovery_authorization.validate_technical_actor(
            discovery_authorization["document"],
            technical_reviewer_actor,
            required_role="execution_plan_reviewer",
        )
        prior_actor = krea_fixture._agent_actor(
            _object(
                _object(
                    resolved["fixture"].get("governance"),
                    "fixture governance",
                ).get("independent_agent_review"),
                "fixture independent agent review",
            ).get("actor"),
            "fixture independent reviewer actor",
        )
        timing_actor = resolved.get("timing_probe_approval_actor")
        if (
            technical_actor.get("actor_id") == prior_actor.get("actor_id")
            or technical_actor.get("review_instance_id")
            == prior_actor.get("review_instance_id")
            or timing_actor is None
            or technical_actor.get("actor_id") == timing_actor.get("actor_id")
            or technical_actor.get("review_instance_id")
            == timing_actor.get("review_instance_id")
        ):
            raise ValueError(
                "execution approval requires a fresh technical review actor"
            )
        role = resolved["fixture"]["experimental_role"]
        if role not in {"D1", "D2"}:
            raise ValueError("discovery envelope authorizes only D1 or D2")
        fixture_binding = envelope_resolved["envelope"]["discovery_fixtures"][role][
            "manifest"
        ]
        if (
            fixture_binding["file_sha256"]
            != _file_binding(plan["fixture_manifest"], "fixture manifest")[1]
            or fixture_binding["manifest_sha256"]
            != resolved["fixture"]["manifest_sha256"]
        ):
            raise ValueError("admission envelope does not authorize the plan fixture")
        output_path = Path(os.path.abspath(os.path.expanduser(approval_output_path)))
        relative = Path(os.path.relpath(envelope_path, output_path.parent))
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError(
                "admission envelope must be inside the approval output directory"
            )
        profile_index = resolved.get("discovery_profile_index")
        bootstrap_receipt = resolved.get("host_bootstrap_receipt")
        if (
            profile_index is None
            or bootstrap_receipt is None
            or discovery_authorization is None
        ):
            raise ValueError(
                "discovery GPU approval requires authorization, profile-index, "
                "and bootstrap bindings"
            )
        body = {
            "schema": 4,
            "kind": _EXECUTION_APPROVAL_KIND,
            "execution_plan_sha256": plan["plan_sha256"],
            "host_execution_identity_sha256": resolved["host_execution_manifest"][
                "host_execution_identity_sha256"
            ],
            "throughput_profile_sha256": resolved["throughput_profile"][
                "profile_sha256"
            ],
            "discovery_profile_index": {
                "file_sha256": profile_index["file_sha256"],
                "index_sha256": profile_index["index_sha256"],
                "fixture_id": profile_index["fixture_id"],
                "throughput_equivalence_class": profile_index[
                    "throughput_equivalence_class"
                ],
                "profile_sha256": profile_index["profile_sha256"],
            },
            "host_bootstrap_receipt": {
                "file_sha256": bootstrap_receipt["file_sha256"],
                "receipt_sha256": bootstrap_receipt["receipt_sha256"],
                "container_image_sha256": bootstrap_receipt["container_image_sha256"],
            },
            "discovery_execution_authorization": {
                "file_sha256": discovery_authorization["file_sha256"],
                "authorization_sha256": discovery_authorization["authorization_sha256"],
            },
            "fixture_admission_envelope": {
                "relative_path": relative.as_posix(),
                "file_sha256": krea_provenance.file_sha256(envelope_path),
                "envelope_sha256": envelope_resolved["envelope"]["envelope_sha256"],
                "phase": "discovery",
            },
            "fixture_role": role,
            "technical_reviewer_actor": technical_actor,
            "accountable_owner_identity": envelope_resolved["ratification"][
                "owner_identity"
            ],
            "owner_ratification_sha256": envelope_resolved["ratification"][
                "ratification_sha256"
            ],
            "approved_at_utc": _strict_utc(approved_at_utc, "approved_at_utc"),
            "decision": "approved",
            "gpu_execution_authorized": True,
            "assertions": {
                "host_capability_reviewed": True,
                "raw_timing_evidence_reviewed": True,
                "fixture_recipe_and_budget_reviewed": True,
                "fixture_admission_envelope_revalidated": True,
                "owner_authorized_mechanical_gpu_gate": True,
                "profile_index_cell_reviewed": True,
                "bootstrap_receipt_and_local_image_reviewed": True,
                "discovery_execution_authorization_reviewed": True,
                "natural_completion_is_post_run_evidence": True,
            },
        }
        return {
            **body,
            "approval_sha256": krea_provenance.canonical_sha256(body),
        }
    body = {
        "schema": 2,
        "kind": _EXECUTION_APPROVAL_KIND,
        "execution_plan_sha256": plan["plan_sha256"],
        "host_execution_identity_sha256": resolved["host_execution_manifest"][
            "host_execution_identity_sha256"
        ],
        "throughput_profile_sha256": resolved["throughput_profile"]["profile_sha256"],
        "reviewer_identity": krea_fixture.named_human(
            reviewer_identity, "reviewer_identity"
        ),
        "approved_at_utc": _strict_utc(approved_at_utc, "approved_at_utc"),
        "decision": "approved",
        "gpu_execution_authorized": True,
        "assertions": {
            "host_capability_reviewed": True,
            "raw_timing_evidence_reviewed": True,
            "fixture_recipe_and_budget_reviewed": True,
            "natural_completion_is_post_run_evidence": True,
        },
    }
    return {**body, "approval_sha256": krea_provenance.canonical_sha256(body)}


def build_postrun_certificate(
    plan: dict[str, Any],
    *,
    run_record: dict[str, str],
    observed: dict[str, Any],
) -> dict[str, Any]:
    """Seal natural completion after execution; never use it to authorize itself."""

    resolved = validate_plan(plan)
    _exact(run_record, {"path", "sha256"}, "post-run record binding")
    run_path, run_sha = _file_binding(run_record, "post-run record")
    _exact(
        observed,
        {
            "linux_ubuntu_22_04",
            "systemd_runtime_max_enforced",
            "fresh_container",
            "h100_vram_mib",
            "outer_wall_clock_s",
            "hard_budget_s",
            "upload_ready_before_boundary",
            "natural_completion",
            "failure_or_fallback_telemetry",
        },
        "post-run observations",
    )
    body = {
        "schema": 1,
        "kind": _POSTRUN_CERTIFICATE_KIND,
        "runner_sha256": plan["runner_sha256"],
        "execution_envelope_sha256": plan["execution_envelope_sha256"],
        "execution_plan_sha256": plan["plan_sha256"],
        "host_execution_identity_sha256": resolved["host_execution_manifest"][
            "host_execution_identity_sha256"
        ],
        "run_record": {"path": str(run_path), "sha256": run_sha},
        **observed,
    }
    certificate = {
        **body,
        "certificate_sha256": krea_provenance.canonical_sha256(body),
    }
    validate_postrun_certificate(certificate, plan=plan)
    return certificate


def validate_postrun_certificate(
    value: dict[str, Any], *, plan: dict[str, Any]
) -> dict[str, Any]:
    resolved = validate_plan(plan)
    _exact(
        value,
        {
            "schema",
            "kind",
            "runner_sha256",
            "execution_envelope_sha256",
            "execution_plan_sha256",
            "host_execution_identity_sha256",
            "run_record",
            "linux_ubuntu_22_04",
            "systemd_runtime_max_enforced",
            "fresh_container",
            "h100_vram_mib",
            "outer_wall_clock_s",
            "hard_budget_s",
            "upload_ready_before_boundary",
            "natural_completion",
            "failure_or_fallback_telemetry",
            "certificate_sha256",
        },
        "post-run certification",
    )
    body = {key: item for key, item in value.items() if key != "certificate_sha256"}
    vram = value["h100_vram_mib"]
    _file_binding(value["run_record"], "post-run record")
    if (
        value["schema"] != 1
        or value["kind"] != _POSTRUN_CERTIFICATE_KIND
        or value["runner_sha256"] != plan["runner_sha256"]
        or value["execution_envelope_sha256"] != plan["execution_envelope_sha256"]
        or value["execution_plan_sha256"] != plan["plan_sha256"]
        or value["host_execution_identity_sha256"]
        != resolved["host_execution_manifest"]["host_execution_identity_sha256"]
        or value["linux_ubuntu_22_04"] is not True
        or value["systemd_runtime_max_enforced"] is not True
        or value["fresh_container"] is not True
        or isinstance(vram, bool)
        or not isinstance(vram, int)
        or not 78_000 <= vram <= 85_000
        or value["upload_ready_before_boundary"] is not True
        or value["natural_completion"] is not True
        or value["failure_or_fallback_telemetry"] is not False
        or isinstance(value["outer_wall_clock_s"], bool)
        or not isinstance(value["outer_wall_clock_s"], (int, float))
        or isinstance(value["hard_budget_s"], bool)
        or not isinstance(value["hard_budget_s"], (int, float))
        or not math.isfinite(float(value["outer_wall_clock_s"]))
        or not math.isfinite(float(value["hard_budget_s"]))
        or float(value["outer_wall_clock_s"]) <= 0
        or float(value["hard_budget_s"]) <= 0
        or float(value["outer_wall_clock_s"]) > float(value["hard_budget_s"])
        or float(value["hard_budget_s"]) != float(plan["budget_plan"]["hard_budget_s"])
        or value["certificate_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("post-run certificate does not prove natural completion")
    return value


def validate_approval(
    approval: dict[str, Any],
    *,
    plan: dict[str, Any],
    approval_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved = validate_plan(plan)
    approval = _object(approval, "execution approval")
    if approval.get("schema") in {3, 4}:
        if resolved["fixture"].get("schema") != 2 or approval_path is None:
            raise ValueError(
                "schema-3 execution approval requires a schema-2 fixture and its file path"
            )
        if plan.get("schema") == 3 and approval.get("schema") != 4:
            raise ValueError(
                "schema-3 execution plans require schema-4 profile/bootstrap approval"
            )
        approval_keys = {
            "schema",
            "kind",
            "execution_plan_sha256",
            "host_execution_identity_sha256",
            "throughput_profile_sha256",
            "fixture_admission_envelope",
            "fixture_role",
            "technical_reviewer_actor",
            "accountable_owner_identity",
            "owner_ratification_sha256",
            "approved_at_utc",
            "decision",
            "gpu_execution_authorized",
            "assertions",
            "approval_sha256",
        }
        if approval.get("schema") == 4:
            approval_keys.update(
                {
                    "discovery_profile_index",
                    "host_bootstrap_receipt",
                    "discovery_execution_authorization",
                }
            )
        _exact(
            approval,
            approval_keys,
            "execution approval",
        )
        body = {
            key: value for key, value in approval.items() if key != "approval_sha256"
        }
        binding = _object(
            approval["fixture_admission_envelope"], "fixture admission envelope"
        )
        _exact(
            binding,
            {"relative_path", "file_sha256", "envelope_sha256", "phase"},
            "fixture admission envelope",
        )
        relative = Path(binding["relative_path"])
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError("fixture admission envelope path is not portable")
        approval_file = Path(os.path.abspath(os.path.expanduser(approval_path)))
        envelope_path = _safe_file(
            approval_file.parent / relative, "fixture admission envelope"
        )
        if krea_provenance.file_sha256(envelope_path) != binding["file_sha256"]:
            raise ValueError("fixture admission envelope file SHA-256 mismatch")
        try:
            from . import krea_fixture_admission
        except ImportError:  # pragma: no cover - direct script execution.
            import krea_fixture_admission  # type: ignore[no-redef]

        envelope_resolved = krea_fixture_admission.validate_envelope(envelope_path)
        envelope = envelope_resolved["envelope"]
        ratification = envelope_resolved["ratification"]
        source_review = resolved.get("arm_basis", {}).get(
            "source_normalization_approval"
        )
        if (
            source_review is not None
            and source_review.get("schema") == 2
            and source_review.get("owner_ratification_sha256")
            != ratification["ratification_sha256"]
        ):
            raise ValueError(
                "source review owner ratification differs from fixture admission"
            )
        expected_authorization = resolved.get("discovery_execution_authorization")
        if approval.get("schema") == 4 and expected_authorization is None:
            raise ValueError("schema-4 approval lacks discovery authorization")
        technical_actor = (
            krea_discovery_authorization.validate_technical_actor(
                expected_authorization["document"],
                approval["technical_reviewer_actor"],
                required_role="execution_plan_reviewer",
            )
            if approval.get("schema") == 4
            else krea_fixture._agent_actor(
                approval["technical_reviewer_actor"], "technical reviewer actor"
            )
        )
        prior_actor = krea_fixture._agent_actor(
            _object(
                _object(
                    resolved["fixture"].get("governance"),
                    "fixture governance",
                ).get("independent_agent_review"),
                "fixture independent agent review",
            ).get("actor"),
            "fixture independent reviewer actor",
        )
        profile_index_approved = approval.get("discovery_profile_index")
        bootstrap_approved = approval.get("host_bootstrap_receipt")
        authorization_approved = approval.get("discovery_execution_authorization")
        expected_profile_index = resolved.get("discovery_profile_index")
        expected_bootstrap = resolved.get("host_bootstrap_receipt")
        if approval.get("schema") == 4:
            _exact(
                _object(profile_index_approved, "approved profile index"),
                {
                    "file_sha256",
                    "index_sha256",
                    "fixture_id",
                    "throughput_equivalence_class",
                    "profile_sha256",
                },
                "approved profile index",
            )
            _exact(
                _object(bootstrap_approved, "approved bootstrap receipt"),
                {
                    "file_sha256",
                    "receipt_sha256",
                    "container_image_sha256",
                },
                "approved bootstrap receipt",
            )
            _exact(
                _object(
                    authorization_approved,
                    "approved discovery execution authorization",
                ),
                {"file_sha256", "authorization_sha256"},
                "approved discovery execution authorization",
            )
            expected_profile_binding = {
                key: expected_profile_index[key]
                for key in (
                    "file_sha256",
                    "index_sha256",
                    "fixture_id",
                    "throughput_equivalence_class",
                    "profile_sha256",
                )
            }
            expected_bootstrap_binding = {
                key: expected_bootstrap[key]
                for key in (
                    "file_sha256",
                    "receipt_sha256",
                    "container_image_sha256",
                )
            }
            expected_authorization_binding = {
                key: expected_authorization[key]
                for key in ("file_sha256", "authorization_sha256")
            }
        else:
            expected_profile_binding = None
            expected_bootstrap_binding = None
            expected_authorization_binding = None
        role = resolved["fixture"]["experimental_role"]
        fixture_binding = envelope["discovery_fixtures"].get(role)
        if (
            approval["kind"] != _EXECUTION_APPROVAL_KIND
            or approval["execution_plan_sha256"] != plan["plan_sha256"]
            or approval["host_execution_identity_sha256"]
            != resolved["host_execution_manifest"]["host_execution_identity_sha256"]
            or approval["throughput_profile_sha256"]
            != resolved["throughput_profile"]["profile_sha256"]
            or profile_index_approved != expected_profile_binding
            or bootstrap_approved != expected_bootstrap_binding
            or authorization_approved != expected_authorization_binding
            or approval["fixture_role"] != role
            or technical_actor != approval["technical_reviewer_actor"]
            or technical_actor.get("role") != "execution_plan_reviewer"
            or technical_actor.get("review_instance_id")
            == prior_actor.get("review_instance_id")
            or (
                approval.get("schema") == 4
                and (
                    resolved.get("timing_probe_approval_actor") is None
                    or technical_actor.get("actor_id")
                    == resolved["timing_probe_approval_actor"].get("actor_id")
                    or technical_actor.get("review_instance_id")
                    == resolved["timing_probe_approval_actor"].get("review_instance_id")
                )
            )
            or approval["accountable_owner_identity"] != ratification["owner_identity"]
            or approval["owner_ratification_sha256"]
            != ratification["ratification_sha256"]
            or role not in {"D1", "D2"}
            or binding["phase"] != "discovery"
            or binding["envelope_sha256"] != envelope["envelope_sha256"]
            or fixture_binding is None
            or fixture_binding["manifest"]["file_sha256"]
            != _file_binding(plan["fixture_manifest"], "fixture manifest")[1]
            or fixture_binding["manifest"]["manifest_sha256"]
            != resolved["fixture"]["manifest_sha256"]
            or approval["decision"] != "approved"
            or approval["gpu_execution_authorized"] is not True
            or approval["assertions"]
            != {
                "host_capability_reviewed": True,
                "raw_timing_evidence_reviewed": True,
                "fixture_recipe_and_budget_reviewed": True,
                "fixture_admission_envelope_revalidated": True,
                "owner_authorized_mechanical_gpu_gate": True,
                **(
                    {
                        "profile_index_cell_reviewed": True,
                        "bootstrap_receipt_and_local_image_reviewed": True,
                        "discovery_execution_authorization_reviewed": True,
                    }
                    if approval.get("schema") == 4
                    else {}
                ),
                "natural_completion_is_post_run_evidence": True,
            }
            or approval["approval_sha256"] != krea_provenance.canonical_sha256(body)
        ):
            raise ValueError("schema-3 execution approval does not authorize this plan")
        krea_fixture.named_human(
            approval["accountable_owner_identity"], "accountable_owner_identity"
        )
        approved_at = datetime.strptime(
            _strict_utc(approval["approved_at_utc"], "approved_at_utc"),
            "%Y-%m-%dT%H:%M:%SZ",
        ).replace(tzinfo=timezone.utc)
        admitted_at = datetime.strptime(
            envelope["admitted_at_utc"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
        if approved_at < admitted_at:
            raise ValueError("GPU approval predates fixture admission")
        return approval
    if resolved["fixture"].get("schema") != 1:
        raise ValueError("schema-2 fixtures require schema-3 envelope-bound approval")
    _exact(
        approval,
        {
            "schema",
            "kind",
            "execution_plan_sha256",
            "host_execution_identity_sha256",
            "throughput_profile_sha256",
            "reviewer_identity",
            "approved_at_utc",
            "decision",
            "gpu_execution_authorized",
            "assertions",
            "approval_sha256",
        },
        "execution approval",
    )
    body = {key: value for key, value in approval.items() if key != "approval_sha256"}
    if (
        approval["schema"] != 2
        or approval["kind"] != _EXECUTION_APPROVAL_KIND
        or approval["execution_plan_sha256"] != plan["plan_sha256"]
        or approval["decision"] != "approved"
        or approval["gpu_execution_authorized"] is not True
        or approval["host_execution_identity_sha256"]
        != _load_binding(plan["host_execution_manifest"], "host execution manifest")[1][
            "host_execution_identity_sha256"
        ]
        or approval["throughput_profile_sha256"]
        != _json_file_binding(plan["throughput_profile"], "throughput profile")[1][
            "profile_sha256"
        ]
        or approval["assertions"]
        != {
            "host_capability_reviewed": True,
            "raw_timing_evidence_reviewed": True,
            "fixture_recipe_and_budget_reviewed": True,
            "natural_completion_is_post_run_evidence": True,
        }
        or approval["approval_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("execution approval does not authorize this plan")
    krea_fixture.named_human(approval["reviewer_identity"], "reviewer_identity")
    _strict_utc(approval["approved_at_utc"], "approved_at_utc")
    return approval
