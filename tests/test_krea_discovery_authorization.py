"""Fail-closed contracts for the acyclic Stage-1 discovery authority."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from ops.calibration import krea_discovery_authorization as authorization
from ops.calibration import krea_execution_surface_policy
from ops.calibration import krea_fixture_admission
from ops.calibration import krea_provenance


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _actor(instance: str, role: str) -> dict[str, str]:
    return {
        "actor_class": "agent",
        "actor_id": f"codex-krea-runtime-reviewer-{instance}",
        "display_name": "Codex Krea runtime reviewer",
        "role": role,
        "review_instance_id": instance,
        "identity_assurance": (
            "self-declared-agent-identity-not-human-or-cryptographic-authentication"
        ),
    }


def _write(path: Path, value: dict, *, canonical: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if canonical:
        path.write_bytes(krea_provenance.canonical_bytes(value) + b"\n")
    else:
        path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def _artifacts(tmp_path: Path, monkeypatch):
    candidates = {"D1": _sha("candidate-D1"), "D2": _sha("candidate-D2")}
    blockers = ["fixtures are not yet admitted", "timing profiles are not measured"]
    discovery = {
        "schema": 2,
        "kind": "sn56-week5-krea-discovery-freeze",
        "status": "draft_blocked_pre_gpu",
        "model": "krea/Krea-2-Raw",
        "model_type": "krea2",
        "gpu_execution_authorized": False,
        "gpu_blockers": blockers,
        "discovery_tasks": {
            role: {
                "identity": digest,
                "fixture_split_manifest_sha256": digest,
            }
            for role, digest in candidates.items()
        },
    }
    discovery_path = _write(tmp_path / "discovery.json", discovery, canonical=False)
    discovery_file_sha = krea_provenance.file_sha256(discovery_path)
    ratification_sha = _sha("ratification")
    fixtures = {
        role: {
            "schema": 2,
            "experimental_role": role,
            "governance": {
                "candidate_manifest_sha256": candidates[role],
                "surface_agent_review": {
                    "actor": _actor(
                        f"fixture-surface-{role.lower()}", "surface_reviewer"
                    )
                },
                "independent_agent_review": {
                    "actor": _actor(
                        f"fixture-review-{role.lower()}", "independent_reviewer"
                    )
                },
                "preparer_actor": _actor(
                    f"fixture-preparer-{role.lower()}", "fixture_preparer"
                ),
            },
            "manifest_sha256": _sha(f"manifest-{role}"),
        }
        for role in ("D1", "D2")
    }
    fixture_approvals = {
        role: {"approval_sha256": _sha(f"approval-{role}")} for role in ("D1", "D2")
    }
    fixture_file_shas = {
        role: hashlib.sha256(krea_provenance.canonical_bytes(value) + b"\n").hexdigest()
        for role, value in fixtures.items()
    }
    approval_file_shas = {
        role: hashlib.sha256(krea_provenance.canonical_bytes(value) + b"\n").hexdigest()
        for role, value in fixture_approvals.items()
    }
    envelope_body = {
        "schema": 1,
        "kind": "forge-krea-fixture-admission-envelope",
        "phase": "discovery",
        "source_package": {"candidate_manifest_sha256s": candidates},
        "governance": {
            "owner_ratification": {"ratification_sha256": ratification_sha},
            "sealed_custodian_actor": {
                "actor": _actor("sealed-custodian", "sealed_fixture_custodian")
            },
        },
        "discovery_fixtures": {
            role: {
                "manifest": {
                    "file_sha256": fixture_file_shas[role],
                    "manifest_sha256": fixtures[role]["manifest_sha256"],
                },
                "approval": {
                    "file_sha256": approval_file_shas[role],
                    "approval_sha256": fixture_approvals[role]["approval_sha256"],
                },
            }
            for role in ("D1", "D2")
        },
        "accountable_owner_identity": "Jordan Example",
        "admitted_at_utc": "2026-07-28T00:00:00Z",
        "decision": "admitted",
        "admission_authorized": True,
        "gpu_execution_authorized": False,
        "admission_producer_actor": _actor(
            "admission-producer", "fixture_admission_producer"
        ),
    }
    envelope = {
        **envelope_body,
        "envelope_sha256": krea_provenance.canonical_sha256(envelope_body),
    }
    envelope_path = _write(
        tmp_path / "admission" / "admission-envelope.json",
        envelope,
        canonical=True,
    )
    ratification = {
        "ratification_sha256": ratification_sha,
        "decision_bindings": {
            "public_source_evidence": {
                "discovery_plan_file_sha256": discovery_file_sha
            },
            "stage1_execution_surface_policy": {
                "policy_sha256": krea_execution_surface_policy.POLICY["policy_sha256"],
                "execution_surface": "staged_host_venv",
                "execution_scope": "discovery_only",
            },
        },
    }
    resolved = {
        "envelope": envelope,
        "envelope_file_sha256": krea_provenance.file_sha256(envelope_path),
        "ratification": ratification,
        "fixtures": fixtures,
        "fixture_approvals": fixture_approvals,
    }
    monkeypatch.setattr(
        krea_fixture_admission,
        "validate_envelope",
        lambda _path: resolved,
    )
    return discovery_path, envelope_path, resolved


def _sealed(tmp_path: Path, monkeypatch):
    discovery_path, envelope_path, resolved = _artifacts(tmp_path, monkeypatch)
    payload = authorization.build_payload(
        discovery_plan_path=discovery_path,
        fixture_admission_envelope_path=envelope_path,
        technical_reviewer_actor=_actor(
            "authorization-review-1",
            "discovery_execution_authorization_reviewer",
        ),
        authorized_at_utc="2026-07-28T00:01:00Z",
    )
    return authorization.seal(payload), discovery_path, envelope_path, resolved


def test_authorization_is_acyclic_exact_and_create_only(tmp_path, monkeypatch):
    record, discovery_path, _, _ = _sealed(tmp_path, monkeypatch)
    output = tmp_path / "authorization.json"

    authorization.publish(output, record)

    binding = {
        "path": str(output),
        "file_sha256": krea_provenance.file_sha256(output),
        "authorization_sha256": record["authorization_sha256"],
    }
    _, loaded, _ = authorization.load_binding(binding)
    discovery = json.loads(discovery_path.read_bytes())
    authorization.assert_matches_discovery(
        loaded,
        discovery_path=discovery_path,
        discovery=discovery,
        discovery_file_sha256=krea_provenance.file_sha256(discovery_path),
        action="bootstrap_timing_probe",
    )
    assert record["gpu_execution_authorized"] is False
    assert "profile" not in json.dumps(record["discovery_plan"])
    assert os.stat(output).st_mode & 0o777 == 0o400
    with pytest.raises(FileExistsError):
        authorization.publish(output, record)


def test_forged_self_hashed_envelope_cannot_mint_authority(tmp_path, monkeypatch):
    _, discovery_path, envelope_path, _ = _sealed(tmp_path, monkeypatch)
    monkeypatch.setattr(
        krea_fixture_admission,
        "validate_envelope",
        lambda _path: (_ for _ in ()).throw(ValueError("full bundle invalid")),
    )

    with pytest.raises(ValueError, match="full bundle invalid"):
        authorization.build_payload(
            discovery_plan_path=discovery_path,
            fixture_admission_envelope_path=envelope_path,
            technical_reviewer_actor=_actor(
                "authorization-review-1",
                "discovery_execution_authorization_reviewer",
            ),
            authorized_at_utc="2026-07-28T00:01:00Z",
        )


def test_unrelated_admitted_envelope_is_rejected(tmp_path, monkeypatch):
    record, _, _, resolved = _sealed(tmp_path, monkeypatch)
    unrelated = json.loads(json.dumps(resolved))
    unrelated["ratification"]["decision_bindings"]["public_source_evidence"][
        "discovery_plan_file_sha256"
    ] = _sha("other-discovery")
    monkeypatch.setattr(
        krea_fixture_admission,
        "validate_envelope",
        lambda _path: unrelated,
    )

    with pytest.raises(ValueError, match="authorization is invalid"):
        authorization.validate(record)


def test_same_candidate_but_different_valid_fixture_is_rejected(tmp_path, monkeypatch):
    record, _, _, resolved = _sealed(tmp_path, monkeypatch)
    admitted = resolved["fixtures"]["D1"]
    approval = resolved["fixture_approvals"]["D1"]
    fixture_file_sha = resolved["envelope"]["discovery_fixtures"]["D1"]["manifest"][
        "file_sha256"
    ]
    approval_file_sha = resolved["envelope"]["discovery_fixtures"]["D1"]["approval"][
        "file_sha256"
    ]
    authorization.assert_fixture_admitted(
        record,
        role="D1",
        fixture=admitted,
        fixture_file_sha256=fixture_file_sha,
        fixture_approval=approval,
        fixture_approval_file_sha256=approval_file_sha,
    )
    substituted = json.loads(json.dumps(admitted))
    substituted["manifest_sha256"] = _sha("different-governance-bearing-manifest")

    with pytest.raises(ValueError, match="differs from exact admitted"):
        authorization.assert_fixture_admitted(
            record,
            role="D1",
            fixture=substituted,
            fixture_file_sha256=fixture_file_sha,
            fixture_approval=approval,
            fixture_approval_file_sha256=approval_file_sha,
        )


def test_authorization_cannot_predate_admission(tmp_path, monkeypatch):
    discovery_path, envelope_path, _ = _artifacts(tmp_path, monkeypatch)
    payload = authorization.build_payload(
        discovery_plan_path=discovery_path,
        fixture_admission_envelope_path=envelope_path,
        technical_reviewer_actor=_actor(
            "authorization-review-1",
            "discovery_execution_authorization_reviewer",
        ),
        authorized_at_utc="2026-07-27T23:59:59Z",
    )

    with pytest.raises(ValueError, match="predates fixture admission"):
        authorization.seal(payload)


def test_human_identity_cannot_be_passed_as_agent_reviewer(tmp_path, monkeypatch):
    discovery_path, envelope_path, _ = _artifacts(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="JSON object"):
        authorization.build_payload(
            discovery_plan_path=discovery_path,
            fixture_admission_envelope_path=envelope_path,
            technical_reviewer_actor="Jordan Example",  # type: ignore[arg-type]
            authorized_at_utc="2026-07-28T00:01:00Z",
        )
