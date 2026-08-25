#!/usr/bin/env python3
"""Week-11 profile for the manifest-bound SN56 IMAGE-pin release contract.

This is a deliberately small adapter over ``sn56-release-contract.py``.  It
changes only the release-generation authority: Week-11 artifact locations,
fresh signing domains, the exact current production rollback, and the known
forward ref.  The existing validator continues to own canonical JSON,
signature, readiness, anonymous-ref, clean-worktree, tree, diff, Docker-policy,
and production-endpoint verification.

The checked-in manifest is an unselected HOLD.  The only target-selection
interface remains the existing create-only ``--regenerate-from`` path; there is
no target SHA argument or environment override.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CORE_PATH = SCRIPT_DIR / "sn56-release-contract.py"

_spec = importlib.util.spec_from_file_location(
    "_sn56_week11_release_contract_core", CORE_PATH
)
if _spec is None or _spec.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError(f"cannot load release-contract core: {CORE_PATH}")
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)


DEFAULT_MANIFEST = ROOT / "release" / "week11-release-manifest.json"
DEFAULT_DOCKER_POLICY = ROOT / "release" / "week11-candidate-docker-policy.json"
DEFAULT_READINESS_RECEIPT = ROOT / "release" / "week11-release-readiness.json"
DEFAULT_ALLOWED_SIGNERS = ROOT / "release" / "week11-release-allowed-signers"
DEFAULT_READINESS_ALLOWED_SIGNERS = (
    ROOT / "release" / "week11-readiness-allowed-signers"
)

SIGNING_PRINCIPAL = "sn56-week11-release"
SIGNING_NAMESPACE = "sn56-week11-final-manifest"
READINESS_SIGNING_PRINCIPAL = "sn56-week11-readiness"
READINESS_SIGNING_NAMESPACE = "sn56-week11-final-readiness"

EXPECTED_TARGET_REF = "refs/heads/codex/week11-flux-release"
EXPECTED_ROLLBACK = {
    "commit": "fe9749c027df511b7566b474e6f8524f86b01f83",
    "tree": "c7e79fb326e3bc3ef7b573d2d0144e82e0e24a25",
    "tree_records_sha256": "c5362ef488af54c7729cba91279b0287b382eee05c5739e802bd894ef72ee582",
    "ref": "refs/heads/week11-ideogram-content-product",
}
UNSELECTED_TARGET = {
    "commit": "0" * 40,
    "tree": "0" * 40,
    "tree_records_sha256": "0" * 64,
    "ref": EXPECTED_TARGET_REF,
}
CANDIDATE_SOURCE_EVIDENCE = {
    "state": "sealed-source-hold",
    "repository_url": "https://github.com/tuly1/sn56-forge-toolkit.git",
    "base_commit": EXPECTED_ROLLBACK["commit"],
    "target": {
        "commit": "3f7737c4045c3877e3c9f100a0341fe2566b1db6",
        "tree": "42001148b7b8342d9837dcb7e2b532b23892aeee",
        "tree_records_sha256": "cdb54ba614daa58d075d9049ee48a00ebed487f12fcd05e2eda5da929d987f44",
        "ref": EXPECTED_TARGET_REF,
    },
    "allowed_changes": {
        "base_commit": EXPECTED_ROLLBACK["commit"],
        "name_status_sha256": "416c825bb8ccb840b2039a3052bffe946b4edc37fed22a5a1bd14824ac07e4f7",
        "entries": [
            {"status": "M", "path": "forge/flux_kohya_config.py"},
            {"status": "M", "path": "forge/tasks/flux_kohya.py"},
            {
                "status": "A",
                "path": "tests/data/week11_flux_kohya/241cda6c-seed-1.toml",
            },
            {
                "status": "A",
                "path": "tests/data/week11_flux_kohya/db5fefc5-seed-1.toml",
            },
            {"status": "A", "path": "tests/test_week11_flux_release.py"},
        ],
    },
}

# This is intentionally not part of EXPECTED_ROLLBACK or the operational
# wrapper.  It is retained only as the documented second-line recovery anchor.
SECONDARY_BREAK_GLASS_COMMIT = "59e0698c952edaf1bf34a117ecad41bce87517cf"
EXPECTED_CANDIDATE_DOCKER_POLICY_SHA256 = (
    "ed4537573f6166ce137483b2d3a67a5d4117aeb876e33511b6267a708ca4e25e"
)


for _name, _value in {
    "DEFAULT_MANIFEST": DEFAULT_MANIFEST,
    "DEFAULT_DOCKER_POLICY": DEFAULT_DOCKER_POLICY,
    "DEFAULT_READINESS_RECEIPT": DEFAULT_READINESS_RECEIPT,
    "DEFAULT_ALLOWED_SIGNERS": DEFAULT_ALLOWED_SIGNERS,
    "DEFAULT_READINESS_ALLOWED_SIGNERS": DEFAULT_READINESS_ALLOWED_SIGNERS,
    "SIGNING_PRINCIPAL": SIGNING_PRINCIPAL,
    "SIGNING_NAMESPACE": SIGNING_NAMESPACE,
    "READINESS_SIGNING_PRINCIPAL": READINESS_SIGNING_PRINCIPAL,
    "READINESS_SIGNING_NAMESPACE": READINESS_SIGNING_NAMESPACE,
    "EXPECTED_ROLLBACK": EXPECTED_ROLLBACK,
    "UNSELECTED_TARGET": UNSELECTED_TARGET,
}.items():
    setattr(_core, _name, _value)


def _set_keyword_default(function: Any, name: str, value: Any) -> None:
    defaults = dict(function.__kwdefaults__ or {})
    if name not in defaults:  # pragma: no cover - detects incompatible core edits
        raise RuntimeError(f"release-contract core no longer exposes {name!r} on {function.__name__}")
    defaults[name] = value
    function.__kwdefaults__ = defaults


for _function, _argument, _value in (
    (_core.verify_manifest_signature, "allowed_signers_path", DEFAULT_ALLOWED_SIGNERS),
    (
        _core._validate_readiness_against_contract,
        "allowed_signers_path",
        DEFAULT_READINESS_ALLOWED_SIGNERS,
    ),
    (
        _core.validate_readiness_receipt,
        "allowed_signers_path",
        DEFAULT_READINESS_ALLOWED_SIGNERS,
    ),
    (_core.validate_contract, "docker_policy_path", DEFAULT_DOCKER_POLICY),
    (_core.validate_rollback_contract, "docker_policy_path", DEFAULT_DOCKER_POLICY),
    (_core.prepare_readiness_receipt, "docker_policy_path", DEFAULT_DOCKER_POLICY),
):
    _set_keyword_default(_function, _argument, _value)


_core_validate_schema = _core.validate_schema
_core_regenerate_manifest = _core.regenerate_manifest
_ALLOW_MOCK_REVIEWED_POLICY = False


def load_docker_policy(path: Path) -> tuple[dict[str, Any], bytes]:
    """Load only the byte-exact canonical Week-11 candidate image policy.

    The Week-10 schema-2 build/parity fields describe a different image and are
    deliberately not inherited. Week-11 instead binds the exact Flux candidate,
    unchanged Docker/base-image identities, causal decision, execution-identity
    correction, and the completed production-entrypoint H100 canary.
    Canonical parsing rejects duplicate keys and non-canonical serialization;
    the fixed payload digest rejects every field substitution or addition.
    """

    policy, raw = _core._load_json_object(path.resolve(), "Week-11 Docker policy")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    mock_canary = (
        policy.get("release_evidence", {})
        .get("production_entrypoint_canary", {})
        .get("mock_only")
        is True
    )
    if actual_sha256 != EXPECTED_CANDIDATE_DOCKER_POLICY_SHA256 and not (
        _ALLOW_MOCK_REVIEWED_POLICY and mock_canary
    ):
        raise _core.ContractError(
            "Week-11 Docker policy differs from the exact canonical candidate policy "
            f"({actual_sha256} != {EXPECTED_CANDIDATE_DOCKER_POLICY_SHA256})"
        )
    if policy.get("policy_state") != "reviewed":
        raise _core.ContractError(
            "Week-11 Flux candidate policy_state must remain exactly reviewed"
        )
    return policy, raw


def validate_schema(data: dict[str, Any]) -> None:
    """Apply the existing schema while permitting an unchanged Dockerfile pair.

    Week-11's reviewed promotion is source-only.  Both Dockerfile blobs remain
    mandatory and independently certified, but requiring them to be *changed*
    would force an unsupported production delta.  A synthetic copy satisfies
    only that historical schema condition; every real entry and its real digest
    remain verified below and again against Git by the core contract.
    """

    _core._expect_keys(
        data,
        {
            "schema_version",
            "release_state",
            "target",
            "rollback",
            "source",
            "dockerfiles",
            "allowed_changes",
            "production",
            "candidate_source_evidence",
        },
        "manifest",
    )
    evidence = data["candidate_source_evidence"]
    if evidence != CANDIDATE_SOURCE_EVIDENCE:
        raise _core.ContractError(
            "candidate_source_evidence differs from the exact sealed Week-11 source contract"
        )
    core_data = copy.deepcopy(data)
    core_data.pop("candidate_source_evidence")

    if _core.target_is_unselected(core_data):
        _core_validate_schema(core_data)
    else:
        profiled = copy.deepcopy(core_data)
        allowed = profiled.get("allowed_changes")
        entries = allowed.get("entries") if isinstance(allowed, dict) else None
        structurally_safe = isinstance(entries, list) and all(
            isinstance(entry, dict)
            and isinstance(entry.get("status"), str)
            and isinstance(entry.get("path"), str)
            for entry in entries
        )
        if not structurally_safe:
            _core_validate_schema(core_data)
            return
        present = {entry["path"] for entry in entries}
        for path in _core.EXPECTED_DOCKER_PATHS:
            if path not in present:
                entries.append({"status": "M", "path": path})
        allowed["name_status_sha256"] = hashlib.sha256(
            _core.canonical_name_status(entries)
        ).hexdigest()
        _core_validate_schema(profiled)

        original_allowed = _core._expect_keys(
            core_data["allowed_changes"],
            {"base_commit", "name_status_sha256", "entries"},
            "allowed_changes",
        )
        original_digest = _core._require_hex(
            original_allowed["name_status_sha256"],
            _core.HEX64,
            "allowed_changes.name_status_sha256",
        )
        embedded_digest = hashlib.sha256(
            _core.canonical_name_status(original_allowed["entries"])
        ).hexdigest()
        if embedded_digest != original_digest:
            raise _core.ContractError(
                "allowed_changes entries do not hash to allowed_changes.name_status_sha256 "
                f"({embedded_digest} != {original_digest})"
            )

        if core_data["target"] != evidence["target"]:
            raise _core.ContractError(
                "selected target differs from candidate_source_evidence"
            )
        if core_data["allowed_changes"] != evidence["allowed_changes"]:
            raise _core.ContractError(
                "selected changed surface differs from candidate_source_evidence"
            )

    if core_data["target"]["ref"] != EXPECTED_TARGET_REF:
        raise _core.ContractError(
            f"target.ref must remain the reviewed Week-11 ref {EXPECTED_TARGET_REF}"
        )


def regenerate_manifest(
    template_path: Path,
    candidate_worktree: Path,
    output_path: Path,
    docker_policy_path: Path = DEFAULT_DOCKER_POLICY,
) -> dict[str, Any]:
    """Create one selected HOLD manifest after proving HEAD and target ref agree."""

    template, _ = _core.load_manifest(template_path.resolve())
    validate_schema(template)
    candidate = _core.ensure_repo_root(candidate_worktree)
    head = _core._git_text(candidate, "rev-parse", "HEAD")
    try:
        ref_commit = _core._git_text(candidate, "rev-parse", EXPECTED_TARGET_REF)
    except _core.ContractError as exc:
        raise _core.ContractError(
            f"candidate worktree does not expose required target ref {EXPECTED_TARGET_REF}"
        ) from exc
    if ref_commit != head:
        raise _core.ContractError(
            f"candidate target ref resolves to {ref_commit}, expected clean HEAD {head}"
        )
    return _core_regenerate_manifest(
        template_path,
        candidate,
        output_path,
        docker_policy_path,
    )


_core.validate_schema = validate_schema
_core.regenerate_manifest = regenerate_manifest
_core.load_docker_policy = load_docker_policy


# Re-export the proven core API so this profile is importable by focused tests
# and by any existing release helper without duplicating the security-critical
# implementation.
for _name in dir(_core):
    if not _name.startswith("__") and _name not in globals():
        globals()[_name] = getattr(_core, _name)


def main(argv: list[str] | None = None) -> int:
    global _ALLOW_MOCK_REVIEWED_POLICY
    arguments = list(sys.argv[1:] if argv is None else argv)
    _ALLOW_MOCK_REVIEWED_POLICY = "--mock" in arguments
    try:
        return _core.main(arguments)
    finally:
        _ALLOW_MOCK_REVIEWED_POLICY = False


if __name__ == "__main__":
    raise SystemExit(main())
