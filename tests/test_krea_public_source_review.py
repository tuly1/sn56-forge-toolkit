"""Fail-closed and relocation contracts for K2-K4 agent source reviews."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from ops.calibration import krea_execution_plan
from ops.calibration import krea_fixture_admission
from ops.calibration import krea_provenance
from ops.calibration import krea_public_source_review


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _seal(body: dict, key: str) -> dict:
    return {**body, key: krea_provenance.canonical_sha256(body)}


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(krea_provenance.canonical_bytes(value) + b"\n")
    return krea_provenance.file_sha256(path)


def _actor(name: str = "codex-public-arm-provenance-auditor") -> dict[str, str]:
    return {
        "actor_class": "agent",
        "actor_id": name,
        "display_name": name.replace("-", " ").title() + " (agent)",
        "role": "source_normalization_reviewer",
        "review_instance_id": f"review-{name}",
        "identity_assurance": (
            "self-declared-agent-identity-not-human-or-cryptographic-authentication"
        ),
    }


def _reseal_review(review: dict) -> None:
    body = {key: value for key, value in review.items() if key != "review_sha256"}
    review["review_sha256"] = krea_provenance.canonical_sha256(body)


def _rewrite_bundle_manifest(root: Path) -> None:
    inventory = krea_public_source_review._bundle_inventory(root)
    (root / "BUNDLE-MANIFEST.sha256").write_text(
        "".join(
            f"{digest}  {relative}\n" for relative, digest in sorted(inventory.items())
        ),
        encoding="ascii",
        newline="",
    )


def _context(monkeypatch, tmp_path: Path, arms=("K2",)) -> dict:
    bundle = tmp_path / "bundle-input"
    source_paths: dict[str, Path] = {}
    sources: dict[str, dict] = {}
    thin_rows: list[str] = []
    source_bindings: dict[str, dict[str, str]] = {}
    for arm in arms:
        source = {
            "source_arm_id": arm,
            "manifest_sha256": _sha(f"{arm}-semantic"),
        }
        source_path = (
            bundle / "public-source-provenance" / f"{arm}-public-source-provenance.json"
        )
        source_file_sha = _write_json(source_path, source)
        relative = f"public-source-provenance/{arm}-public-source-provenance.json"
        thin_rows.append(f"{source_file_sha}  {relative}\n")
        source_paths[arm] = source_path
        sources[arm] = source
        source_bindings[arm] = {
            "file_sha256": source_file_sha,
            "manifest_sha256": source["manifest_sha256"],
        }
    thin_path = bundle / "MANIFEST.sha256"
    thin_path.write_text("".join(thin_rows), encoding="ascii", newline="")
    public_evidence = {
        "discovery_plan_file_sha256": _sha("discovery-plan"),
        "thin_manifest_file_sha256": krea_provenance.file_sha256(thin_path),
        "public_source_provenance": {
            arm: source_bindings.get(
                arm,
                {
                    "file_sha256": _sha(f"{arm}-file"),
                    "manifest_sha256": _sha(f"{arm}-semantic"),
                },
            )
            for arm in ("K2", "K3", "K4")
        },
    }

    governance = bundle / "governance"
    custodian = _actor("sealed-custodian-agent")
    custodian["role"] = "sealed_confirmation_custodian"
    custodian_path = governance / "sealed-custodian-actor.json"
    custodian_file_sha = _write_json(custodian_path, custodian)
    custodian_sha = krea_provenance.canonical_sha256(custodian)
    amendment_body = {
        "schema": 1,
        "kind": "forge-krea-review-governance-amendment",
        "amended_at_utc": "2026-07-30T02:00:00Z",
        "public_source_evidence": public_evidence,
        "sealed_custodian_actor": {
            "file_sha256": custodian_file_sha,
            "actor_sha256": custodian_sha,
            "actor": custodian,
        },
        "admission_authorized": False,
        "gpu_execution_authorized": False,
    }
    amendment = _seal(amendment_body, "amendment_sha256")
    amendment_path = governance / "amendment.json"
    amendment_file_sha = _write_json(amendment_path, amendment)
    decisions = {
        "governance_amendment_sha256": amendment["amendment_sha256"],
        "public_source_evidence": public_evidence,
        "sealed_custodian_actor_sha256": custodian_sha,
    }
    portable_body = {
        "schema": 1,
        "kind": "forge-krea-portable-owner-ratification-draft",
        "prepared_at_utc": "2026-07-30T02:00:30Z",
        "owner_identity": "Atulya Shetty",
        "decision_bindings": decisions,
        "evidence_files": {
            "sealed_custodian_actor": {
                "file_sha256": custodian_file_sha,
                "actor_sha256": custodian_sha,
            }
        },
        "admission_authorized": False,
        "gpu_execution_authorized": False,
    }
    portable = _seal(portable_body, "draft_sha256")
    portable_path = governance / "portable-ratification-draft.json"
    portable_file_sha = _write_json(portable_path, portable)
    ratification_body = {
        "schema": 1,
        "kind": "forge-krea-sole-human-owner-ratification",
        "owner_identity": "Atulya Shetty",
        "ratified_at_utc": "2026-07-30T02:01:04Z",
        "portable_ratification_draft": {
            "file_sha256": portable_file_sha,
            "draft_sha256": portable["draft_sha256"],
        },
        "governance_amendment": {
            "file_sha256": amendment_file_sha,
            "amendment_sha256": amendment["amendment_sha256"],
        },
        "decision_bindings": decisions,
        "decision": "ratified_for_fixture_admission_input",
        "admission_authorized": False,
        "gpu_execution_authorized": False,
    }
    ratification = _seal(ratification_body, "ratification_sha256")
    ratification_path = governance / "owner-ratification.json"
    ratification_file_sha = _write_json(ratification_path, ratification)

    local_draft = {
        "draft_sha256": _sha("local-draft"),
        "inputs": {"governance_amendment": str(amendment_path)},
    }
    local_draft_path = tmp_path / "owner-ratification.draft.json"
    local_draft_file_sha = _write_json(local_draft_path, local_draft)
    resolved = {
        "draft": local_draft,
        "draft_file_sha256": local_draft_file_sha,
        "portable_draft": portable,
        "portable_draft_path": portable_path,
        "sealed_custodian_actor": custodian,
        "sealed_custodian_actor_path": custodian_path,
    }
    calls = []

    def validate_owner(value, *, resolved):
        calls.append((value, resolved))
        return value

    monkeypatch.setattr(
        krea_execution_plan.krea_provenance,
        "validate_manifest",
        lambda value: value,
    )
    monkeypatch.setattr(krea_fixture_admission, "_resolve_draft", lambda path: resolved)
    monkeypatch.setattr(
        krea_fixture_admission, "validate_owner_ratification", validate_owner
    )
    return {
        "bundle": bundle,
        "thin_path": thin_path,
        "sources": sources,
        "source_paths": source_paths,
        "public_evidence": public_evidence,
        "ratification": ratification,
        "ratification_path": ratification_path,
        "ratification_file_sha": ratification_file_sha,
        "portable": portable,
        "portable_path": portable_path,
        "portable_file_sha": portable_file_sha,
        "amendment": amendment,
        "amendment_path": amendment_path,
        "amendment_file_sha": amendment_file_sha,
        "custodian": custodian,
        "custodian_path": custodian_path,
        "custodian_file_sha": custodian_file_sha,
        "local_draft_path": local_draft_path,
        "resolved": resolved,
        "calls": calls,
    }


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": krea_provenance.file_sha256(path)}


def _build_review(context: dict, arm: str = "K2") -> tuple[Path, dict]:
    review_path = context["bundle"] / f"{arm}-source-normalization-review.json"
    review = krea_execution_plan.build_agent_public_source_review(
        review_output_path=review_path,
        source_provenance=_binding(context["source_paths"][arm]),
        thin_evidence_manifest=_binding(context["thin_path"]),
        owner_ratification=_binding(context["ratification_path"]),
        portable_ratification_draft=_binding(context["portable_path"]),
        governance_amendment=_binding(context["amendment_path"]),
        sealed_custodian_actor=_binding(context["custodian_path"]),
        actor=_actor(),
        reviewed_at_utc="2026-07-30T02:02:00Z",
    )
    _write_json(review_path, review)
    return review_path, review


def _validate(review: dict, review_path: Path, context: dict, arm="K2") -> dict:
    return krea_execution_plan._validate_public_approval(
        review,
        source_manifest=context["sources"][arm],
        source_manifest_file_sha256=krea_provenance.file_sha256(
            context["source_paths"][arm]
        ),
        approval_path=review_path,
    )


def test_agent_review_reopens_relocatable_ratification_chain(
    monkeypatch, tmp_path
) -> None:
    context = _context(monkeypatch, tmp_path)
    review_path, review = _build_review(context)

    resolved = _validate(review, review_path, context)

    assert resolved["schema"] == 2
    assert resolved["actor"]["actor_class"] == "agent"
    assert resolved["owner_identity"] == "Atulya Shetty"
    assert resolved["public_source_evidence"] == context["public_evidence"]
    assert all(
        not binding.get("relative_path", "").startswith("/")
        for binding in (
            review["source_provenance"],
            review["thin_evidence_manifest"],
            review["owner_ratification"],
            review["portable_ratification_draft"],
            review["governance_amendment"],
            review["sealed_custodian_actor"],
        )
    )
    assert review["admission_authorized"] is False
    assert review["gpu_execution_authorized"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda review: review["source_provenance"].__setitem__(
            "file_sha256", _sha("wrong-source-file")
        ),
        lambda review: review["source_provenance"].__setitem__(
            "manifest_sha256", _sha("wrong-source-semantic")
        ),
        lambda review: review["assertions"].__setitem__(
            "canonical_manifest_valid", False
        ),
        lambda review: review.__setitem__("admission_authorized", True),
        lambda review: review.__setitem__("gpu_execution_authorized", True),
        lambda review: review["actor"].__setitem__("actor_class", "human"),
        lambda review: review.__setitem__("reviewed_at_utc", "2026-07-30T02:01:03Z"),
        lambda review: review["owner_ratification"].__setitem__(
            "ratification_sha256", _sha("wrong-ratification")
        ),
        lambda review: review.__setitem__("claim_limit", "broader claim"),
    ],
)
def test_agent_review_rejects_rehashed_substitution(
    monkeypatch, tmp_path, mutation
) -> None:
    context = _context(monkeypatch, tmp_path)
    review_path, original = _build_review(context)
    review = deepcopy(original)
    mutation(review)
    _reseal_review(review)

    with pytest.raises(ValueError):
        _validate(review, review_path, context)


def test_agent_review_rejects_mismatched_v2_public_binding(
    monkeypatch, tmp_path
) -> None:
    context = _context(monkeypatch, tmp_path)
    review_path, review = _build_review(context)
    ratification = deepcopy(context["ratification"])
    ratification["decision_bindings"]["public_source_evidence"][
        "public_source_provenance"
    ]["K2"]["manifest_sha256"] = _sha("substituted-v2-source")
    ratification_body = {
        key: value
        for key, value in ratification.items()
        if key != "ratification_sha256"
    }
    ratification["ratification_sha256"] = krea_provenance.canonical_sha256(
        ratification_body
    )
    _write_json(context["ratification_path"], ratification)
    review["owner_ratification"]["file_sha256"] = krea_provenance.file_sha256(
        context["ratification_path"]
    )
    review["owner_ratification"]["ratification_sha256"] = ratification[
        "ratification_sha256"
    ]
    _reseal_review(review)

    with pytest.raises(ValueError, match="ratification chain"):
        _validate(review, review_path, context)


def test_agent_review_rejects_custodian_substitution(monkeypatch, tmp_path) -> None:
    context = _context(monkeypatch, tmp_path)
    review_path, review = _build_review(context)
    substituted = _actor("different-custodian-agent")
    substituted["role"] = "sealed_confirmation_custodian"
    _write_json(context["custodian_path"], substituted)
    review["sealed_custodian_actor"]["file_sha256"] = krea_provenance.file_sha256(
        context["custodian_path"]
    )
    review["sealed_custodian_actor"]["actor_sha256"] = krea_provenance.canonical_sha256(
        substituted
    )
    _reseal_review(review)

    with pytest.raises(ValueError, match="ratification chain"):
        _validate(review, review_path, context)


def test_legacy_named_human_source_approval_is_unchanged() -> None:
    source = {"source_arm_id": "K2", "manifest_sha256": _sha("source")}
    approval = {
        "schema": 1,
        "kind": "forge-krea-source-normalization-approval",
        "source_arm_id": "K2",
        "provenance_manifest_sha256": source["manifest_sha256"],
        "reviewer_identity": "Jordan Example",
        "decision": "approved",
        "assertions": {
            "source_fields_reviewed": True,
            "unsupported_fields_reviewed": True,
            "adaptations_reviewed": True,
            "source_artifact_identity_reviewed": True,
            "claim_limits_reviewed": True,
        },
    }
    assert (
        krea_execution_plan._validate_public_approval(
            approval,
            source_manifest=source,
            source_manifest_file_sha256=_sha("source-file"),
        )["reviewer_identity"]
        == "Jordan Example"
    )
    approval["reviewer_identity"] = "human reviewer"
    with pytest.raises(ValueError, match="role label"):
        krea_execution_plan._validate_public_approval(
            approval,
            source_manifest=source,
            source_manifest_file_sha256=_sha("source-file"),
        )


def test_create_only_bundle_survives_relocation_and_source_deletion(
    monkeypatch, tmp_path
) -> None:
    context = _context(monkeypatch, tmp_path, arms=("K2", "K3", "K4"))
    output = tmp_path / "reviews"
    summary = krea_public_source_review.create_reviews(
        public_evidence_root=context["bundle"],
        owner_ratification_path=context["ratification_path"],
        owner_ratification_draft_path=context["local_draft_path"],
        output_dir=output,
        reviewed_at_utc="2026-07-30T02:02:00Z",
    )
    moved = tmp_path / "relocated" / "portable-reviews"
    moved.parent.mkdir()
    shutil.copytree(output, moved)
    shutil.rmtree(output)
    shutil.rmtree(context["bundle"])

    relocated = krea_public_source_review.validate_bundle(moved)

    assert relocated["review_sha256s"] == summary["review_sha256s"]
    assert (moved / "MANIFEST.sha256").is_file()
    assert (moved / "governance" / "owner-ratification.json").is_file()
    for arm in ("K2", "K3", "K4"):
        raw = (moved / f"{arm}-source-normalization-review.json").read_text()
        assert "/Users/" not in raw
        assert "atulyashetty" not in raw


def test_bundle_rejects_missing_or_substituted_file(monkeypatch, tmp_path) -> None:
    context = _context(monkeypatch, tmp_path, arms=("K2", "K3", "K4"))
    output = tmp_path / "reviews"
    krea_public_source_review.create_reviews(
        public_evidence_root=context["bundle"],
        owner_ratification_path=context["ratification_path"],
        owner_ratification_draft_path=context["local_draft_path"],
        output_dir=output,
        reviewed_at_utc="2026-07-30T02:02:00Z",
    )
    missing = tmp_path / "missing"
    substituted = tmp_path / "substituted"
    shutil.copytree(output, missing)
    shutil.copytree(output, substituted)
    (missing / "governance" / "amendment.json").unlink()
    with pytest.raises(ValueError, match="inventory"):
        krea_public_source_review.validate_bundle(missing)

    source = (
        substituted / "public-source-provenance" / "K2-public-source-provenance.json"
    )
    source.write_bytes(source.read_bytes() + b" ")
    _rewrite_bundle_manifest(substituted)
    with pytest.raises(ValueError):
        krea_public_source_review.validate_bundle(substituted)


def test_create_only_refuses_existing_output(monkeypatch, tmp_path) -> None:
    context = _context(monkeypatch, tmp_path, arms=("K2", "K3", "K4"))
    output = tmp_path / "reviews"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        krea_public_source_review.create_reviews(
            public_evidence_root=context["bundle"],
            owner_ratification_path=context["ratification_path"],
            owner_ratification_draft_path=context["local_draft_path"],
            output_dir=output,
            reviewed_at_utc="2026-07-30T02:02:00Z",
        )
