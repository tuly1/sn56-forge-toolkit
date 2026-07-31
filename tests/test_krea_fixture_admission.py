"""Governance and phase-bound authorization tests for Krea admission."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from ops.calibration import krea_execution_plan
from ops.calibration import krea_fixture
from ops.calibration import krea_fixture_admission as admission
from ops.calibration import krea_provenance


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _utc(offset: int = -60) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _actor(name: str, role: str) -> dict[str, str]:
    return {
        "actor_class": "agent",
        "actor_id": name,
        "display_name": name.replace("-", " ").title(),
        "role": role,
        "review_instance_id": f"review-{name}",
        "identity_assurance": admission._AGENT_ASSURANCE,
    }


def _governance() -> dict:
    surface = _actor("surface-agent", "surface_reviewer")
    independent = _actor("independent-agent", "independent_reviewer")
    preparer = _actor("preparer-agent", "fixture_implementer")
    return {
        "mode": admission._MODE,
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
        "accountable_owner_identity": admission._OWNER,
        "owner_identity_assurance": admission._OWNER_ASSURANCE,
        "agent_review_is_not_human_review": True,
        "independent_human_review_performed": False,
        "claim_limit": admission._CLAIM_LIMIT,
    }


def _minimal_schema2_manifest() -> dict:
    governance = _governance()
    empty_report = {
        "comparisons": 0,
        "minimum_hamming_distance": 64,
        "exact_matches": [],
        "near_matches": [],
        "cross_split_group_matches": [],
    }
    body = {
        "schema": 2,
        "kind": "forge-krea-curated-fixture",
        "concept_id": "fixture",
        "experimental_role": "D1",
        "trigger_token": "trigger",
        "caption_policy": {
            "reviewer_identity": governance["surface_agent_review"]["actor"][
                "display_name"
            ]
        },
        "source_rights": {
            "reviewer_identity": governance["surface_agent_review"]["actor"][
                "display_name"
            ]
        },
        "preparer_identity": governance["preparer_actor"]["display_name"],
        "training_archive": {},
        "training_archive_identity": {},
        "training_dataset_identity": {},
        "evaluation_dataset_identity": {},
        "training_dataset_shape_sha256": _sha("train-shape"),
        "evaluation_dataset_shape_sha256": _sha("eval-shape"),
        "training_rows": [],
        "evaluation_rows": [],
        "tool_identity": {},
        "near_duplicate_policy": {
            "maximum_hamming_distance": 8,
            "report": empty_report,
            "report_sha256": krea_provenance.canonical_sha256(empty_report),
            "passed": True,
            "group_disjoint_fields": [],
            "human_similarity_review": {
                "reviewer_identity": governance["surface_agent_review"]["actor"][
                    "display_name"
                ],
                "method": "owner-ratified-agent-review-plus-pinned-ahash",
            },
        },
        "governance": governance,
    }
    return {**body, "manifest_sha256": krea_provenance.canonical_sha256(body)}


def test_policy_is_canonical_and_explicitly_not_human_review() -> None:
    policy = admission.load_policy()
    assert policy["agent_review_is_not_human_review"] is True
    assert policy["independent_human_review_performed"] is False
    assert policy["legacy_named_human_contract_unchanged"] is True


def test_governance_amendment_binds_full_gpu_gate_surface_and_rejects_drift(
    monkeypatch,
) -> None:
    expected_keys = {
        "fixture_validator_file_sha256",
        "admission_tool_file_sha256",
        "execution_plan_file_sha256",
        "host_identity_file_sha256",
        "runtime_binding_file_sha256",
        "host_bootstrap_file_sha256",
        "stage1_runtime_file_sha256",
        "public_source_review_file_sha256",
        "execution_surface_policy_file_sha256",
        "discovery_authorization_file_sha256",
        "profile_index_file_sha256",
        "budget_planner_file_sha256",
        "runner_file_sha256",
        "training_evidence_file_sha256",
        "decision_tool_file_sha256",
        "score_plan_file_sha256",
        "batch_evaluator_file_sha256",
        "delegated_review_contract_loader_file_sha256",
    }
    original = admission._gpu_gate_implementation_bindings()
    assert set(original) == expected_keys
    assert all(len(value) == 64 for value in original.values())

    real_file_sha = admission._file_sha256

    def changed_file_sha(path: Path) -> str:
        if Path(path).name == "krea_runtime_binding.py":
            return _sha("changed-runtime-binding")
        return real_file_sha(path)

    monkeypatch.setattr(admission, "_file_sha256", changed_file_sha)
    changed = admission._gpu_gate_implementation_bindings()
    assert changed != original
    assert (
        changed["runtime_binding_file_sha256"]
        != original["runtime_binding_file_sha256"]
    )
    assert {
        key: value
        for key, value in changed.items()
        if key != "runtime_binding_file_sha256"
    } == {
        key: value
        for key, value in original.items()
        if key != "runtime_binding_file_sha256"
    }


def test_discovery_plan_binds_exact_k2_k3_k4_public_evidence() -> None:
    binding = admission._discovery_public_evidence_bindings()

    assert binding["thin_manifest_file_sha256"] == (
        "35fc00e534e779da04844a829f2dbfb3cef1451971a2e5a15f8b97f708912868"
    )
    assert binding["public_source_provenance"] == {
        "K2": {
            "file_sha256": "6bbeae58d5e4bcf02ba5fda576875c95c4016ebd2353cc74c20bc95ced311e50",
            "manifest_sha256": "a54145aa233b999330cf145d8e371eaf1790dd0c08bd4abca5b57061d1d63a2e",
        },
        "K3": {
            "file_sha256": "1a29c0f948360c6fdf5e77cf9ad4a32b9b9b02827daad38ceeb6e85c9f469e5b",
            "manifest_sha256": "91a04ed0e94af9859ba1cf19903439fa69016888ed7b254bad96badfe3783eba",
        },
        "K4": {
            "file_sha256": "1553fc7c13a9a3dbcf8e3bdf599e49ade4740d0f80af9056a42ff1bf982bf4da",
            "manifest_sha256": "8507031d5a5a152e4443b83d5bdb0066b6626c74fc707caebd918bb5a5db808e",
        },
    }


def test_forge_repository_identity_requires_exact_clean_tree(tmp_path) -> None:
    repository = tmp_path / "forge"
    repository.mkdir()
    tracked = repository / "tracked.txt"
    tracked.write_text("bound bytes\n", encoding="ascii")
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }

    def git(*arguments: str) -> None:
        subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            env=environment,
        )

    git("init", "-q")
    git("add", "tracked.txt")
    git("commit", "-q", "-m", "fixture")
    identity = admission._forge_repository_identity(repository)
    assert len(identity["commit_sha1"]) == 40
    assert len(identity["tree_sha1"]) == 40
    assert identity["tracked_files"] == [
        {
            "path": "tracked.txt",
            "mode": "100644",
            "git_blob_sha1": identity["tracked_files"][0]["git_blob_sha1"],
            "sha256": hashlib.sha256(b"bound bytes\n").hexdigest(),
            "bytes": 12,
        }
    ]
    assert identity["tracked_file_manifest_sha256"] == (
        krea_provenance.canonical_sha256(identity["tracked_files"])
    )

    (repository / "untracked.txt").write_text("no\n", encoding="ascii")
    with pytest.raises(ValueError, match="clean"):
        admission._forge_repository_identity(repository)
    (repository / "untracked.txt").unlink()

    git("update-index", "--skip-worktree", "tracked.txt")
    with pytest.raises(ValueError, match="skip-worktree"):
        admission._forge_repository_identity(repository)
    git("update-index", "--no-skip-worktree", "tracked.txt")

    git("update-index", "--assume-unchanged", "tracked.txt")
    with pytest.raises(ValueError, match="assume-unchanged"):
        admission._forge_repository_identity(repository)
    git("update-index", "--no-assume-unchanged", "tracked.txt")

    tracked.write_text("substituted\n", encoding="ascii")
    with pytest.raises(ValueError, match="clean|bytes differ"):
        admission._forge_repository_identity(repository)


def test_god_evaluator_contract_requires_exact_clean_checkout(tmp_path) -> None:
    checkout = tmp_path / "G.O.D"
    image_io = checkout / admission._GOD_IMAGE_IO
    constants = checkout / admission._GOD_DATASET_CONSTANTS
    image_io.parent.mkdir(parents=True)
    constants.parent.mkdir(parents=True)
    image_io.write_text(
        "import os\n\n"
        "def list_supported_images(dataset_path: str, extensions: tuple) -> list[str]:\n"
        "    return [file_name for file_name in os.listdir(dataset_path) if file_name.lower().endswith(extensions)]\n",
        encoding="utf-8",
    )
    constants.write_text(
        'SUPPORTED_IMAGE_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg")\n',
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }
    for command in (
        ["git", "init", "-q", str(checkout)],
        ["git", "-C", str(checkout), "remote", "add", "origin", admission._GOD_ORIGIN],
        ["git", "-C", str(checkout), "add", "."],
        ["git", "-C", str(checkout), "commit", "-q", "-m", "fixture"],
    ):
        subprocess.run(command, check=True, env=environment)
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    contract, _, _ = admission._build_god_evaluator_contract(
        checkout, expected_commit=commit
    )
    assert contract["extensions"] == [".png", ".jpg", ".jpeg"]
    assert contract["commit"] == commit

    (checkout / "untracked").write_text("no", encoding="ascii")
    with pytest.raises(ValueError, match="must be clean"):
        admission._build_god_evaluator_contract(checkout, expected_commit=commit)


def test_imported_evidence_cannot_smuggle_nested_authority() -> None:
    admission._reject_true_authorization_flags(
        {"approval": {"admission_authorized": False}}, "evidence"
    )
    with pytest.raises(ValueError, match="gpu_execution_authorized=true"):
        admission._reject_true_authorization_flags(
            {"notes": [{"gpu_execution_authorized": True}]}, "evidence"
        )


def test_schema2_dispatch_keeps_agent_and_owner_roles_separate(monkeypatch) -> None:
    manifest = _minimal_schema2_manifest()
    seen = {}
    monkeypatch.setattr(
        krea_fixture,
        "_validate_legacy_manifest",
        lambda projected: seen.setdefault("projected", projected),
    )

    assert krea_fixture.validate_manifest(manifest) == manifest
    assert seen["projected"]["schema"] == 1
    assert seen["projected"]["preparer_identity"] == "Governance Projection Preparer"
    assert manifest["preparer_identity"] == "Preparer Agent"
    assert manifest["governance"]["accountable_owner_identity"] == admission._OWNER


@pytest.mark.parametrize(
    "mutation",
    ["claims_human", "same_actor", "owner_assurance", "manifest_digest"],
)
def test_schema2_governance_tampering_fails_closed(monkeypatch, mutation) -> None:
    manifest = _minimal_schema2_manifest()
    monkeypatch.setattr(krea_fixture, "_validate_legacy_manifest", lambda value: value)
    if mutation == "claims_human":
        manifest["governance"]["independent_human_review_performed"] = True
    elif mutation == "same_actor":
        manifest["governance"]["independent_agent_review"]["actor"] = deepcopy(
            manifest["governance"]["surface_agent_review"]["actor"]
        )
    elif mutation == "owner_assurance":
        manifest["governance"]["owner_identity_assurance"] = "signed"
    else:
        manifest["manifest_sha256"] = _sha("forged")
    if mutation != "manifest_digest":
        body = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        manifest["manifest_sha256"] = krea_provenance.canonical_sha256(body)
    with pytest.raises(ValueError):
        krea_fixture.validate_manifest(manifest)


def _acceptance_request() -> dict:
    parent = _actor("independent-agent", "independent_technical_reviewer")
    custodian = _actor("sealed-custodian", krea_fixture._SEALED_CUSTODIAN_ROLE)
    body = {
        "schema": 2,
        "kind": "forge-krea-blinded-confirmation-acceptance-request",
        "requested_at_utc": _utc(-120),
        "source_package": {
            "package_sha256": _sha("package"),
            "file_set_sha256": _sha("files"),
        },
        "discovery_fixture_manifest_sha256s": {
            "D1": _sha("D1"),
            "D2": _sha("D2"),
        },
        "governance": {
            "parent_independent_review": {
                "file_sha256": _sha("independent-review-file"),
                "review_sha256": _sha("independent-review"),
                "actor": parent,
            },
            "sealed_custodian_actor": {
                "file_sha256": _sha("sealed-custodian-file"),
                "actor_sha256": krea_provenance.canonical_sha256(custodian),
                "actor": custodian,
            },
            "owner_ratification": {
                "file_sha256": _sha("owner-ratification-file"),
                "ratification_sha256": _sha("owner-ratification"),
            },
            "agent_review_is_not_human_review": True,
            "independent_human_review_performed": False,
        },
        "confirmation_commitment": {
            "public_record_file_sha256": admission.krea_c1c4_amendment.PUBLIC_RECORD_SHA256,
            "commitment_sha256": admission.krea_c1c4_amendment.COMMITMENT_SHA256,
            "published_manifest_file_sha256s": admission.krea_c1c4_amendment.MANIFEST_FILE_SHA256S,
            "shape_amendment_file_sha256": admission.krea_c1c4_amendment.AMENDMENT_FILE_SHA256,
            "shape_amendment_sha256": admission.krea_c1c4_amendment.AMENDMENT_SHA256,
            "c1c4_revealed": False,
        },
        "required_private_digest_only_fields": [
            "C1-C4 file-to-semantic manifest SHA-256 mapping",
            "all-six D1,D2,C1,C2,C3,C4 semantic manifest map",
            "cross-review file SHA-256, fixture-set SHA-256, pair count and SHA-256",
            "cross-review binding SHA-256",
            "exact pre-ratified sealed-custodian actor",
            "custody and unrevealed assertions",
        ],
        "admission_authorized": False,
        "gpu_execution_authorized": False,
    }
    return {**body, "request_sha256": krea_provenance.canonical_sha256(body)}


def _agent_cross_binding(request: dict) -> dict:
    all_six = {
        **request["discovery_fixture_manifest_sha256s"],
        **{role: _sha(role) for role in ("C1", "C2", "C3", "C4")},
    }
    governance = request["governance"]
    parent = {
        "review_sha256": governance["parent_independent_review"]["review_sha256"],
        "actor": governance["parent_independent_review"]["actor"],
    }
    count = 10
    reviewed_pairs_sha = _sha("pairs")
    visual_scope = {
        "performed": False,
        "method": "not-performed",
        "coverage": "none",
        "reviewed_pair_count": 0,
        "reviewed_pairs_sha256": krea_provenance.canonical_sha256([]),
    }
    body = {
        "schema": 2,
        "kind": krea_fixture._AGENT_CROSS_BINDING_KIND,
        "review_file_sha256": _sha("review-file"),
        "review_sha256": _sha("agent-cross-review"),
        "fixture_manifest_sha256s": all_six,
        "fixture_manifest_set_sha256": krea_provenance.canonical_sha256(all_six),
        "actor": governance["sealed_custodian_actor"]["actor"],
        "parent_independent_review": parent,
        "owner_ratification_sha256": governance["owner_ratification"][
            "ratification_sha256"
        ],
        "acceptance_request_sha256": request["request_sha256"],
        "reviewed_at_utc": _utc(-60),
        "reviewed_pair_count": count,
        "reviewed_pairs_sha256": reviewed_pairs_sha,
        "review_scope": {
            "automated": {
                "method": "exact-hash-group-identity-and-pinned-ahash",
                "coverage": "all-cross-role-pairs",
                "reviewed_pair_count": count,
                "reviewed_pairs_sha256": reviewed_pairs_sha,
                "maximum_hamming_distance_policy": "max-of-paired-fixture-thresholds",
            },
            "visual": visual_scope,
        },
        "decision": "passed_agent_automated_cross_fixture_nonoverlap",
        "assertions": krea_fixture._agent_cross_assertions(),
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": krea_fixture._AGENT_CROSS_CLAIM_LIMIT,
    }
    return {**body, "binding_sha256": krea_provenance.canonical_sha256(body)}


def test_blinded_acceptance_binds_all_six_without_revealing_c() -> None:
    request = _acceptance_request()
    actor = request["governance"]["sealed_custodian_actor"]["actor"]
    cross_binding = _agent_cross_binding(request)
    acceptance = admission.build_blinded_acceptance(
        request,
        custodian_actor=actor,
        c1c4_semantic_manifest_sha256s={
            role: _sha(role) for role in ("C1", "C2", "C3", "C4")
        },
        cross_fixture_review=cross_binding,
        reviewed_at_utc=_utc(-60),
    )
    assert (
        admission.validate_blinded_acceptance(acceptance, request=request) == acceptance
    )
    assert (
        acceptance["actor"]
        != request["governance"]["parent_independent_review"]["actor"]
    )
    assert (
        acceptance["owner_ratification_sha256"]
        == request["governance"]["owner_ratification"]["ratification_sha256"]
    )
    assert acceptance["c1c4_revealed"] is False
    assert acceptance["gpu_execution_authorized"] is False

    tampered = deepcopy(acceptance)
    tampered["fixture_manifest_sha256s"]["C4"] = _sha("different")
    with pytest.raises(ValueError):
        admission.validate_blinded_acceptance(tampered, request=request)


@pytest.mark.parametrize(
    "mutation",
    [
        "opus_as_custodian",
        "different_custodian",
        "wrong_parent",
        "wrong_owner",
        "wrong_request",
        "binding_digest",
        "true_authority",
    ],
)
def test_blinded_acceptance_rejects_custodian_or_binding_substitution(
    mutation,
) -> None:
    request = _acceptance_request()
    actor = request["governance"]["sealed_custodian_actor"]["actor"]
    binding = _agent_cross_binding(request)
    if mutation == "opus_as_custodian":
        actor = request["governance"]["parent_independent_review"]["actor"]
    elif mutation == "different_custodian":
        actor = _actor("unratified-custodian", krea_fixture._SEALED_CUSTODIAN_ROLE)
    elif mutation == "wrong_parent":
        binding["parent_independent_review"]["review_sha256"] = _sha("wrong")
    elif mutation == "wrong_owner":
        binding["owner_ratification_sha256"] = _sha("wrong")
    elif mutation == "wrong_request":
        binding["acceptance_request_sha256"] = _sha("wrong")
    elif mutation == "binding_digest":
        binding["binding_sha256"] = _sha("wrong")
    else:
        binding["gpu_execution_authorized"] = True
    with pytest.raises(ValueError):
        admission.build_blinded_acceptance(
            request,
            custodian_actor=actor,
            c1c4_semantic_manifest_sha256s={
                role: _sha(role) for role in ("C1", "C2", "C3", "C4")
            },
            cross_fixture_review=binding,
            reviewed_at_utc=_utc(-60),
        )


def test_ratification_refuses_noninteractive_input(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(admission.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(admission.sys.stdout, "isatty", lambda: True)
    with pytest.raises(RuntimeError, match="interactive TTY"):
        admission.ratify_interactively(
            draft_path=tmp_path / "draft.json", output_path=tmp_path / "ratify.json"
        )
    assert not (tmp_path / "ratify.json").exists()


def test_portable_ratification_binds_real_draft_and_chronology() -> None:
    package = {
        "package_sha256": _sha("package"),
        "file_set_sha256": _sha("file-set"),
        "candidate_manifest_sha256s": {"D1": _sha("D1"), "D2": _sha("D2")},
    }
    inputs = {
        "package": package,
        "package_manifest_file_sha256": _sha("package-file"),
    }
    originals = {
        "surface_file_sha256": _sha("surface-source"),
        "independent_file_sha256": _sha("independent-source"),
    }
    policy = admission.load_policy()
    surface = {"review_sha256": _sha("surface"), "reviewed_at_utc": _utc(-300)}
    independent = {
        "review_sha256": _sha("independent"),
        "reviewed_at_utc": _utc(-240),
        "actor": _actor("independent-agent", "independent_technical_reviewer"),
    }
    amendment = {
        "governance_policy": {"policy_sha256": policy["policy_sha256"]},
        "amendment_sha256": _sha("amendment"),
        "amended_at_utc": _utc(-180),
        "forge_repository_identity": {
            "commit_sha1": "1" * 40,
            "tree_sha1": "2" * 40,
            "tracked_files": [],
            "tracked_file_manifest_sha256": krea_provenance.canonical_sha256([]),
        },
        "public_source_evidence": admission._discovery_public_evidence_bindings(),
        "implementation": admission._gpu_gate_implementation_bindings(),
        "canonical_agent_evidence": {
            "surface_review": {"review_sha256": surface["review_sha256"]},
            "independent_review": {"review_sha256": independent["review_sha256"]},
        },
    }
    custodian = _actor("sealed-custodian", krea_fixture._SEALED_CUSTODIAN_ROLE)
    custodian_file_sha = _sha("sealed-custodian-file")
    amendment["sealed_custodian_actor"] = {
        "file_sha256": custodian_file_sha,
        "actor_sha256": krea_provenance.canonical_sha256(custodian),
        "actor": custodian,
    }
    amendment["stage1_delegated_agent_review_contract"] = (
        admission.krea_delegated_review_contract.binding()
    )
    evaluator = {
        "contract_sha256": _sha("evaluator"),
        "commit": "a" * 40,
    }
    amendment_file_sha = _sha("amendment-file")

    def portable(prepared_at: str) -> dict:
        return admission.build_portable_ratification_draft(
            inputs=inputs,
            originals=originals,
            policy=policy,
            surface_review=surface,
            surface_review_file_sha256=_sha("surface-file"),
            independent_review=independent,
            independent_review_file_sha256=_sha("independent-file"),
            sealed_custodian_actor=custodian,
            sealed_custodian_actor_file_sha256=custodian_file_sha,
            amendment=amendment,
            amendment_file_sha256=amendment_file_sha,
            evaluator_contract=evaluator,
            evaluator_contract_file_sha256=_sha("evaluator-file"),
            prepared_at_utc=prepared_at,
        )

    draft = portable(_utc(-120))
    assert (
        draft["decision_bindings"]["forge_repository_identity"]
        == amendment["forge_repository_identity"]
    )
    assert (
        draft["decision_bindings"]["public_source_evidence"]
        == amendment["public_source_evidence"]
    )
    draft_file_sha = _sha("portable-draft-file")
    resolved = {
        "draft": {"decision_bindings": draft["decision_bindings"]},
        "amendment": amendment,
        "amendment_file_sha256": amendment_file_sha,
        "portable_draft": draft,
        "portable_draft_file_sha256": draft_file_sha,
    }
    ratification = admission.build_owner_ratification(
        resolved, ratified_at_utc=_utc(-60)
    )
    assert (
        ratification["decision_bindings"]["public_source_evidence"]
        == admission._discovery_public_evidence_bindings()
    )
    assert (
        admission._validate_portable_ratification(
            ratification,
            amendment=amendment,
            amendment_file_sha256=amendment_file_sha,
            policy=policy,
            inputs=inputs,
            surface_review=surface,
            independent_review=independent,
            evaluator_contract=evaluator,
            portable_draft=draft,
            portable_draft_file_sha256=draft_file_sha,
        )
        == ratification
    )

    forged = deepcopy(ratification)
    forged["portable_ratification_draft"]["file_sha256"] = _sha("arbitrary")
    forged_body = {
        key: value for key, value in forged.items() if key != "ratification_sha256"
    }
    forged["ratification_sha256"] = krea_provenance.canonical_sha256(forged_body)
    with pytest.raises(ValueError):
        admission._validate_portable_ratification(
            forged,
            amendment=amendment,
            amendment_file_sha256=amendment_file_sha,
            policy=policy,
            inputs=inputs,
            surface_review=surface,
            independent_review=independent,
            evaluator_contract=evaluator,
            portable_draft=draft,
            portable_draft_file_sha256=draft_file_sha,
        )

    postdated = portable(_utc(-30))
    postdated_resolved = {
        **resolved,
        "draft": {"decision_bindings": postdated["decision_bindings"]},
        "portable_draft": postdated,
    }
    postdated_ratification = admission.build_owner_ratification(
        postdated_resolved, ratified_at_utc=_utc(-60)
    )
    with pytest.raises(ValueError, match="predates bound agent evidence"):
        admission._validate_portable_ratification(
            postdated_ratification,
            amendment=amendment,
            amendment_file_sha256=amendment_file_sha,
            policy=policy,
            inputs=inputs,
            surface_review=surface,
            independent_review=independent,
            evaluator_contract=evaluator,
            portable_draft=postdated,
            portable_draft_file_sha256=draft_file_sha,
        )


def test_schema2_execution_approval_requires_and_binds_envelope(
    monkeypatch, tmp_path
) -> None:
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text("fixture", encoding="ascii")
    envelope_dir = tmp_path / "evidence"
    envelope_dir.mkdir()
    envelope_path = envelope_dir / "admission-envelope.json"
    envelope_path.write_text("envelope", encoding="ascii")
    approval_path = tmp_path / "execution-approval.json"
    fixture_file_sha = krea_provenance.file_sha256(fixture_path)
    envelope_file_sha = krea_provenance.file_sha256(envelope_path)
    resolved = {
        "fixture": {
            "schema": 2,
            "experimental_role": "D1",
            "manifest_sha256": _sha("fixture-semantic"),
            "governance": {
                "independent_agent_review": {
                    "actor": _actor("fixture-independent-agent", "independent_reviewer")
                }
            },
        },
        "host_execution_manifest": {"host_execution_identity_sha256": _sha("host")},
        "throughput_profile": {"profile_sha256": _sha("profile")},
        "discovery_profile_index": {
            "file_sha256": _sha("index-file"),
            "index_sha256": _sha("index"),
            "fixture_id": "D1",
            "throughput_equivalence_class": "A-class",
            "profile_sha256": _sha("profile"),
        },
        "host_bootstrap_receipt": {
            "file_sha256": _sha("receipt-file"),
            "receipt_sha256": _sha("receipt"),
            "container_image_sha256": _sha("image"),
        },
        "discovery_execution_authorization": {
            "file_sha256": _sha("authorization-file"),
            "authorization_sha256": _sha("authorization"),
            "document": {},
        },
        "timing_probe_approval_actor": _actor(
            "timing-agent", "timing_probe_execution_reviewer"
        ),
    }
    plan = {
        "schema": 3,
        "plan_sha256": _sha("plan"),
        "fixture_manifest": {"path": str(fixture_path), "sha256": fixture_file_sha},
    }
    envelope = {
        "envelope_sha256": _sha("envelope"),
        "admitted_at_utc": "2026-07-28T00:00:00Z",
        "discovery_fixtures": {
            "D1": {
                "manifest": {
                    "file_sha256": fixture_file_sha,
                    "manifest_sha256": _sha("fixture-semantic"),
                }
            }
        },
    }
    ratification = {
        "owner_identity": admission._OWNER,
        "ratification_sha256": _sha("ratification"),
    }
    monkeypatch.setattr(krea_execution_plan, "validate_plan", lambda value: resolved)
    monkeypatch.setattr(
        krea_execution_plan.krea_discovery_authorization,
        "validate_technical_actor",
        lambda _authorization, actor, **_kwargs: actor,
    )
    monkeypatch.setattr(
        admission,
        "validate_envelope",
        lambda path: {"envelope": envelope, "ratification": ratification},
    )
    with pytest.raises(ValueError, match="require an admission envelope"):
        krea_execution_plan.build_approval(
            plan,
            reviewer_identity=None,
            approved_at_utc="2026-07-29T00:00:00Z",
        )
    with pytest.raises(ValueError, match="reviewer_identity must be omitted"):
        krea_execution_plan.build_approval(
            plan,
            reviewer_identity="Jordan Example",
            approved_at_utc="2026-07-29T00:00:00Z",
            admission_envelope_path=envelope_path,
            approval_output_path=approval_path,
            technical_reviewer_actor=_actor(
                "execution-agent", "execution_plan_reviewer"
            ),
        )
    with pytest.raises(ValueError, match="fresh technical review actor"):
        krea_execution_plan.build_approval(
            plan,
            reviewer_identity=None,
            approved_at_utc="2026-07-29T00:00:00Z",
            admission_envelope_path=envelope_path,
            approval_output_path=approval_path,
            technical_reviewer_actor=resolved["fixture"]["governance"][
                "independent_agent_review"
            ]["actor"],
        )
    approval = krea_execution_plan.build_approval(
        plan,
        reviewer_identity=None,
        approved_at_utc="2026-07-29T00:00:00Z",
        admission_envelope_path=envelope_path,
        approval_output_path=approval_path,
        technical_reviewer_actor=_actor("execution-agent", "execution_plan_reviewer"),
    )
    assert approval["schema"] == 4
    assert approval["fixture_admission_envelope"] == {
        "relative_path": "evidence/admission-envelope.json",
        "file_sha256": envelope_file_sha,
        "envelope_sha256": envelope["envelope_sha256"],
        "phase": "discovery",
    }
    assert approval["accountable_owner_identity"] == admission._OWNER
    assert approval["owner_ratification_sha256"] == ratification["ratification_sha256"]
    assert (
        krea_execution_plan.validate_approval(
            approval, plan=plan, approval_path=approval_path
        )
        == approval
    )

    wrong = deepcopy(approval)
    wrong["fixture_admission_envelope"]["envelope_sha256"] = _sha("wrong")
    wrong_body = {
        key: value for key, value in wrong.items() if key != "approval_sha256"
    }
    wrong["approval_sha256"] = krea_provenance.canonical_sha256(wrong_body)
    with pytest.raises(ValueError):
        krea_execution_plan.validate_approval(
            wrong, plan=plan, approval_path=approval_path
        )


def test_successor_rederives_exact_588_repository_and_implementation() -> None:
    repository = Path(admission.__file__).resolve().parents[2]
    identity = admission._forge_repository_identity_at_commit(
        repository, admission._SUCCESSOR_ADMISSION_COMMIT
    )

    assert identity["commit_sha1"] == admission._SUCCESSOR_ADMISSION_COMMIT
    assert identity["tree_sha1"] == admission._SUCCESSOR_ADMISSION_TREE
    bindings = admission._implementation_bindings_from_repository_identity(identity)
    assert set(bindings) == set(admission._GPU_GATE_IMPLEMENTATION_PATHS)
    tracked = {row["path"]: row for row in identity["tracked_files"]}
    for key, relative in admission._GPU_GATE_IMPLEMENTATION_PATHS.items():
        assert bindings[key] == tracked[relative]["sha256"]


def test_successor_runtime_allowlist_is_the_exact_bridge_surface() -> None:
    assert admission._SUCCESSOR_ALLOWED_RUNTIME_PATHS == {
        "ops/calibration/krea_execution_plan.py",
        "ops/calibration/krea_fixture_admission.py",
        "ops/calibration/krea_runtime_binding.py",
        "ops/calibration/run_krea_ladder.py",
    }


def test_successor_bridge_preserves_only_historical_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical_identity = {
        "commit_sha1": admission._SUCCESSOR_ADMISSION_COMMIT,
        "tree_sha1": admission._SUCCESSOR_ADMISSION_TREE,
        "tracked_files": [],
        "tracked_file_manifest_sha256": _sha("historical-manifest"),
    }
    current_body = {
        "schema": 1,
        "kind": admission._AMENDMENT_KIND,
        "amended_at_utc": "2026-07-29T00:00:00Z",
        "implementation": {"current": _sha("current")},
        "forge_repository_identity": {
            "commit_sha1": _sha("current")[:40],
        },
        "unchanged_evidence": {"record": _sha("evidence")},
    }
    current = {
        **current_body,
        "amendment_sha256": krea_provenance.canonical_sha256(current_body),
    }
    historical_implementation = {"historical": _sha("historical")}
    historical_body = {
        **current_body,
        "implementation": historical_implementation,
        "forge_repository_identity": historical_identity,
    }
    amendment = {
        **historical_body,
        "amendment_sha256": krea_provenance.canonical_sha256(historical_body),
    }
    monkeypatch.setattr(admission, "_require_fc70_successor", lambda *_args: None)
    monkeypatch.setattr(
        admission,
        "_forge_repository_identity_at_commit",
        lambda *_args: historical_identity,
    )
    monkeypatch.setattr(
        admission,
        "_implementation_bindings_from_repository_identity",
        lambda _identity: historical_implementation,
    )

    assert (
        admission._successor_bound_governance_amendment(amendment, current)
        == amendment
    )
    tampered = deepcopy(amendment)
    tampered["implementation"] = {"historical": _sha("tampered")}
    with pytest.raises(ValueError, match="implementation binding drifted"):
        admission._successor_bound_governance_amendment(tampered, current)
