from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from ops.calibration import krea_confirmation_admission as admission
from ops.calibration import krea_fixture
from ops.calibration import krea_provenance
from ops.calibration import krea_stage2_boundary_derivation as derivation
from ops.calibration import krea_stage2_delegated_review_contract as contract


def _load_timing_helpers():
    path = Path(__file__).with_name("test_krea_stage2_timing.py")
    spec = importlib.util.spec_from_file_location("stage2_timing_test_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _actor(label: str, role: str) -> dict[str, str]:
    return {
        "actor_class": "agent",
        "actor_id": label,
        "display_name": label.replace("-", " ").title(),
        "role": role,
        "review_instance_id": f"review-{label}",
        "identity_assurance": (
            "self-declared-agent-identity-not-human-or-cryptographic-authentication"
        ),
    }


def _schema2(source: dict) -> dict:
    surface = _actor("surface-agent", "surface_reviewer")
    independent = _actor("independent-agent", "independent_technical_reviewer")
    preparer = _actor("preparer-agent", "fixture_implementer")
    value = deepcopy(source)
    value["schema"] = 2
    value["preparer_identity"] = preparer["display_name"]
    value["source_rights"]["reviewer_identity"] = surface["display_name"]
    value["caption_policy"]["reviewer_identity"] = surface["display_name"]
    near = value["near_duplicate_policy"]
    near["maximum_hamming_distance"] = 8
    near["report"] = krea_fixture._duplicates(
        value["training_rows"],
        value["evaluation_rows"],
        threshold=8,
        group_disjoint_fields=tuple(near["group_disjoint_fields"]),
    )
    near["report_sha256"] = krea_provenance.canonical_sha256(near["report"])
    near["human_similarity_review"]["reviewer_identity"] = surface["display_name"]
    near["human_similarity_review"][
        "method"
    ] = "owner-ratified-agent-review-plus-pinned-ahash"
    value["governance"] = {
        "mode": "sole-human-owner-ratifies-agent-review-v1",
        "policy_sha256": _sha("policy"),
        "governance_amendment": {
            "file_sha256": _sha("amendment-file"),
            "amendment_sha256": _sha("amendment"),
        },
        "owner_ratification": {
            "file_sha256": _sha("ratification-file"),
            "ratification_sha256": _sha("ratification"),
        },
        "source_package": {
            "file_sha256": _sha("package-file"),
            "package_sha256": _sha("package"),
        },
        "candidate_manifest_sha256": _sha("candidate"),
        "surface_agent_review": {
            "file_sha256": _sha("surface-file"),
            "review_sha256": _sha("surface"),
            "actor": surface,
        },
        "independent_agent_review": {
            "file_sha256": _sha("independent-file"),
            "review_sha256": _sha("independent"),
            "actor": independent,
        },
        "preparer_actor": preparer,
        "accountable_owner_identity": "Atulya Shetty",
        "owner_identity_assurance": (
            "interactive-owner-self-attestation-not-cryptographic-or-legal-signature"
        ),
        "agent_review_is_not_human_review": True,
        "independent_human_review_performed": False,
        "claim_limit": (
            "owner-ratified-agent-evidence; Stage-1 is staged-host-venv "
            "discovery-only, not production/release/tournament evidence; "
            "Stage-2 requires a separate Forge commit and fresh named-owner "
            "ratification"
        ),
    }
    value.pop("manifest_sha256")
    value["manifest_sha256"] = krea_provenance.canonical_sha256(value)
    return krea_fixture.validate_manifest(value)


def _sources(tmp_path: Path) -> dict[str, tuple[Path, Path, Path]]:
    helpers = _load_timing_helpers()
    result = {}
    for role, train, evaluate in (("D1", 18, 24), ("D2", 36, 40)):
        manifest = _schema2(helpers._fixture(tmp_path, role, train, evaluate))
        approval = krea_fixture.build_agent_governed_approval(
            manifest,
            technical_reviewer_actor=manifest["governance"]["independent_agent_review"][
                "actor"
            ],
            accountable_owner_identity="Atulya Shetty",
            approved_at_utc="2026-08-01T18:30:00Z",
        )
        root = tmp_path / f"fixture-{role}"
        manifest_path = root / "fixture-manifest.json"
        approval_path = root / "fixture-approval.json"
        manifest_path.write_bytes(krea_provenance.canonical_bytes(manifest) + b"\n")
        approval_path.write_bytes(krea_provenance.canonical_bytes(approval) + b"\n")
        result[role] = (manifest_path, approval_path, root)
    return result


def test_builds_all_six_boundaries_without_weakening_d2_group_evidence(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    output = tmp_path / "derived"
    record = derivation.build(
        d1_manifest_path=sources["D1"][0],
        d1_approval_path=sources["D1"][1],
        d1_package_root=sources["D1"][2],
        d2_manifest_path=sources["D2"][0],
        d2_approval_path=sources["D2"][1],
        d2_package_root=sources["D2"][2],
        freeze_binding_path=Path(derivation.__file__).with_name("week5")
        / Path(derivation.FREEZE_BINDING_PATH).name,
        output_dir=output,
        created_at_utc="2026-08-01T18:35:00Z",
    )
    assert len(record["roles"]) == 6
    for row in record["roles"]:
        manifest = json.loads(
            (
                output / "sealed-boundary" / row["role"] / "fixture-manifest.json"
            ).read_text(encoding="utf-8")
        )
        assert krea_fixture.validate_manifest(manifest) == manifest
        assert manifest["boundary_derivation"]["science_selection_input"] is False
        assert (
            manifest["boundary_derivation"]["source_governance_authorizes_boundary"]
            is False
        )
        if row["source_role"] == "D2":
            assert "play_component_id" in manifest["training_rows"][0]["group_identity"]
            assert (
                "accession_family_id"
                in manifest["near_duplicate_policy"]["group_disjoint_fields"]
            )
    with pytest.raises(FileExistsError):
        derivation.build(
            d1_manifest_path=sources["D1"][0],
            d1_approval_path=sources["D1"][1],
            d1_package_root=sources["D1"][2],
            d2_manifest_path=sources["D2"][0],
            d2_approval_path=sources["D2"][1],
            d2_package_root=sources["D2"][2],
            freeze_binding_path=Path(derivation.__file__).with_name("week5")
            / Path(derivation.FREEZE_BINDING_PATH).name,
            output_dir=output,
            created_at_utc="2026-08-01T18:36:00Z",
        )


def test_postfreeze_inventory_is_metadata_only_and_fails_before_root_on_bad_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _sources(tmp_path)
    derived = tmp_path / "derived"
    record = derivation.build(
        d1_manifest_path=sources["D1"][0],
        d1_approval_path=sources["D1"][1],
        d1_package_root=sources["D1"][2],
        d2_manifest_path=sources["D2"][0],
        d2_approval_path=sources["D2"][1],
        d2_package_root=sources["D2"][2],
        freeze_binding_path=Path(derivation.__file__).with_name("week5")
        / Path(derivation.FREEZE_BINDING_PATH).name,
        output_dir=derived,
        created_at_utc="2026-08-01T18:35:00Z",
    )
    sealed = tmp_path / "sealed"
    (sealed).mkdir()
    for role in admission._CONFIRMATION_ROLES:
        path = sealed / role / "fixture-manifest.json"
        path.parent.mkdir()
        path.write_bytes(
            krea_provenance.canonical_bytes(
                {"schema": 99, "experimental_role": role, "secret": "fixture payload"}
            )
            + b"\n"
        )
    for role_root in (derived / "sealed-boundary").iterdir():
        target = sealed / role_root.name
        target.mkdir()
        for source in role_root.rglob("*"):
            if source.is_file():
                destination = target / source.relative_to(role_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
    public = {
        role: hashlib.sha256(
            (sealed / role / "fixture-manifest.json").read_bytes()
        ).hexdigest()
        for role in admission._CONFIRMATION_ROLES
    }
    boundary_semantic = {row["role"]: row["manifest_sha256"] for row in record["roles"]}
    boundary_files = {
        row["role"]: row["manifest_file_sha256"] for row in record["roles"]
    }
    freeze_path = (
        Path(derivation.__file__).with_name("week5")
        / Path(derivation.FREEZE_BINDING_PATH).name
    )
    output = tmp_path / "inventory.json"
    real_validate = admission.krea_fixture.validate_manifest
    monkeypatch.setattr(
        admission.krea_fixture,
        "validate_manifest",
        lambda value: value if value.get("schema") == 99 else real_validate(value),
    )
    inventory = admission.materialize_postfreeze_inventory(
        public_freeze_binding_path=freeze_path,
        remote_reachable_commit_sha1=derivation.FREEZE_BINDING_COMMIT,
        public_commitment_sha256s=public,
        boundary_fixture_manifest_sha256s=boundary_semantic,
        boundary_fixture_manifest_file_sha256s=boundary_files,
        sealed_root=sealed,
        output_path=output,
        actor=contract.actor("confirmation_materialization_reviewer"),
        captured_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    assert admission.validate_postfreeze_inventory(inventory) == inventory
    assert b'"secret":"fixture payload"' not in output.read_bytes()
    assert inventory["fixture_payload_bytes_emitted"] is False

    monkeypatch.setattr(
        admission,
        "_resolve_sealed_root",
        lambda _root: pytest.fail("bad pushed-freeze gate resolved sealed root"),
    )
    with pytest.raises(ValueError, match="not observed remotely"):
        admission.materialize_postfreeze_inventory(
            public_freeze_binding_path=freeze_path,
            remote_reachable_commit_sha1="0" * 40,
            public_commitment_sha256s=public,
            boundary_fixture_manifest_sha256s=boundary_semantic,
            boundary_fixture_manifest_file_sha256s=boundary_files,
            sealed_root=sealed,
            output_path=tmp_path / "must-not-exist.json",
            actor=contract.actor("confirmation_materialization_reviewer"),
            captured_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
