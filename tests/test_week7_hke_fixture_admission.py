"""Human authority boundary for the Week-7 procedural fixtures."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


renderer = load(
    "week7_hke_renderer_admission_tests",
    ROOT / "ops" / "experiments" / "week7" / "hke_procedural_renderer.py",
)
admission = load(
    "week7_hke_admission_tests",
    ROOT / "ops" / "experiments" / "week7" / "hke_fixture_admission.py",
)


DISCOVERY_KEY = b"d" * 32
CONFIRMATION_KEY = b"c" * 32
GENERATOR_COMMIT = "a" * 40
GENERATOR_TREE = "b" * 40
RIGHTS = {
    "author_record": "SN56 first-party procedural renderer",
    "rights_owner": "SN56 project owner",
    "license_or_use_grant": "first-party training and evaluation fixture use",
}


@pytest.fixture(scope="module")
def candidate(tmp_path_factory):
    root = tmp_path_factory.mktemp("hke-admission")
    public = root / "public-boundary" / "public"
    public.parent.mkdir()
    custodian = root / "custodian"
    renderer.build_candidate(
        public_output=public,
        custodian_output=custodian,
        discovery_key=DISCOVERY_KEY,
        confirmation_key=CONFIRMATION_KEY,
        generator_commit=GENERATOR_COMMIT,
        generator_tree=GENERATOR_TREE,
        public_boundary_roots=(public.parent,),
        **RIGHTS,
    )
    return public, custodian


def completed_review(candidate):
    public, custodian = candidate
    draft = admission.build_review_template(
        public, custodian, public_boundary_roots=(public.parent,)
    )
    draft["reviewer_identity"] = "Atulya Shetty"
    draft["reviewed_at_utc"] = "2026-08-10T23:00:00Z"
    draft["decision"] = "PASS"
    for row in draft["rows"]:
        for key in row["checks"]:
            row["checks"][key] = True
    return draft


def _rewrite_semantic(path: Path, value: dict) -> None:
    body = {key: item for key, item in value.items() if key != "semantic_sha256"}
    value["semantic_sha256"] = renderer.semantic_sha256(body)
    path.write_bytes(renderer.canonical_bytes(value) + b"\n")


def generator_probe():
    return {
        "repository": renderer.GENERATOR_REPOSITORY,
        "commit": GENERATOR_COMMIT,
        "tree": GENERATOR_TREE,
        "source_path": renderer.GENERATOR_SOURCE_PATH,
        "renderer_source_sha256": renderer._source_sha256(),
        "admission_authority_path": admission.ADMISSION_SOURCE_PATH,
        "contract_path": renderer.DECLARATIVE_CONTRACT_SOURCE_PATH,
        "contract_source_sha256": renderer._contract_source_sha256(),
        "admission_authority_source_sha256": admission.hashlib.sha256(
            admission.SCRIPT_PATH.read_bytes()
        ).hexdigest(),
        "factor_authority_path": admission.FACTOR_AUTHORITY_SOURCE_PATH,
        "factor_authority_source_sha256": admission.hashlib.sha256(
            admission.FACTOR_AUTHORITY_PATH.read_bytes()
        ).hexdigest(),
        "pinned_remote_refs": ["refs/heads/test-pushed-branch"],
    }


def admitted(candidate):
    public, custodian = candidate
    sealed = admission.seal_review(
        public_root=public,
        custodian_root=custodian,
        public_boundary_roots=(public.parent,),
        draft=completed_review(candidate),
    )
    root, receipts = admission.build_admissions(
        public_root=public,
        custodian_root=custodian,
        discovery_key=DISCOVERY_KEY,
        confirmation_key=CONFIRMATION_KEY,
        public_boundary_roots=(public.parent,),
        sealed_review=sealed,
        generator_identity_probe=generator_probe,
    )
    return public, custodian, sealed, root, receipts


def confirmation_freeze(receipts: dict) -> dict:
    source_plan = {
        "plan_sha256": "1" * 64,
        "frozen_candidate_sha256": "2" * 64,
        "candidate": {
            "arm": "K1",
            "source_plan_sha256": "9" * 64,
            "source_cell_sha256": "a" * 64,
            "bundle_id": "futurebound-mae",
            "bundle_sha256": "4" * 64,
            "runtime_commit": "5" * 40,
            "generated_config": {"config": {"process": [{"train": {"steps": 1200}}]}},
            "generated_config_sha256": admission.semantic_sha256(
                {"config": {"process": [{"train": {"steps": 1200}}]}}
            ),
            "checkpoint_step": 1200,
            "artifact_sha256": "7" * 64,
            "seed": 42,
        },
        "confirmation_commitments": {
            family: {
                pack: record["semantic_commitment_sha256"]
                for pack, record in receipt["packs"].items()
                if record["phase"] == "confirmation"
            }
            for family, receipt in receipts.items()
        },
    }
    source_evidence = {
        "evidence_sha256": "3" * 64,
        "d2_comparisons": {
            "Seed-A": {"direction": "improves"},
            "Seed-B": {"direction": "improves"},
        },
    }
    return _FactorAuthorityStub.freeze_confirmation_candidate(
        source_plan,
        source_evidence,
        frozen_at_utc="2026-08-12T01:00:00Z",
    )


class _FactorAuthorityStub:
    """Small deterministic authority used only by admission unit tests."""

    @staticmethod
    def freeze_confirmation_candidate(plan, evidence, *, frozen_at_utc):
        body = {
            "schema": admission.SCHEMA,
            "kind": admission.CONFIRMATION_FREEZE_KIND,
            "status": "FROZEN_BEFORE_CONFIRMATION_REVEAL",
            "source_d2_plan": copy.deepcopy(plan),
            "source_d2_plan_sha256": plan["plan_sha256"],
            "source_d2_evidence": copy.deepcopy(evidence),
            "d1_frozen_candidate_sha256": plan["frozen_candidate_sha256"],
            "d2_evidence_sha256": evidence["evidence_sha256"],
            "candidate": copy.deepcopy(plan["candidate"]),
            "discovery_decision": {
                "decision": "ADVANCE",
                "confirmation_reveal_authorized": True,
            },
            "d2_comparisons": copy.deepcopy(evidence["d2_comparisons"]),
            "both_finalist_seeds_agree": True,
            "confirmation_commitments": copy.deepcopy(plan["confirmation_commitments"]),
            "confirmation_reveal_authorized": True,
            "frozen_at_utc": frozen_at_utc,
            "checkpoint_promotion_authorized": False,
            "deployment_authorized": False,
        }
        return {
            **body,
            "confirmation_freeze_sha256": admission.semantic_sha256(body),
        }

    @staticmethod
    def _validate_confirmation_freeze(value, *, d2_plan):
        expected = _FactorAuthorityStub.freeze_confirmation_candidate(
            d2_plan,
            value["source_d2_evidence"],
            frozen_at_utc=value["frozen_at_utc"],
        )
        if value != expected:
            raise ValueError("freeze mismatch")
        return value

    @staticmethod
    def build_c2_reveal_authority(plan, evidence):
        body = {
            "schema": admission.SCHEMA,
            "kind": admission.C2_AUTHORITY_KIND,
            "status": "C2_REVEAL_AUTHORIZED_BY_BORDERLINE_C1",
            "source_confirmation_plan": copy.deepcopy(plan),
            "source_confirmation_evidence": copy.deepcopy(evidence),
            "source_confirmation_freeze_sha256": plan["confirmation_freeze_sha256"],
            "c1_evidence_sha256": evidence["evidence_sha256"],
            "c1_comparison": copy.deepcopy(evidence["c1_comparison"]),
            "trigger": {"decision": "REVEAL_C2", "reason": "borderline"},
            "c2_commitment_sha256": plan["c2_commitment_sha256"],
            "c2_reveal_authorized": True,
            "checkpoint_promotion_authorized": False,
            "deployment_authorized": False,
        }
        return {**body, "c2_authority_sha256": admission.semantic_sha256(body)}


@pytest.fixture
def factor_authority_stub(monkeypatch):
    monkeypatch.setattr(
        admission, "_load_factor_authority", lambda: _FactorAuthorityStub
    )


def c2_authority(receipts: dict, freeze: dict) -> dict:
    plan = {
        "confirmation_freeze_sha256": freeze["confirmation_freeze_sha256"],
        "c2_commitment_sha256": receipts["social"]["packs"]["C2"][
            "semantic_commitment_sha256"
        ],
    }
    evidence = {
        "evidence_sha256": "8" * 64,
        "c1_comparison": {
            "relative_improvement": {"composite": 0.035},
            "paired_ci95": [0.001, 0.06],
        },
    }
    return _FactorAuthorityStub.build_c2_reveal_authority(plan, evidence)


def test_template_is_private_pending_and_never_fakes_human_review(candidate):
    public, custodian = candidate
    template = admission.build_review_template(
        public, custodian, public_boundary_roots=(public.parent,)
    )
    assert template["status"] == "PENDING_NAMED_HUMAN_REVIEW"
    assert template["reviewer_identity"] == ""
    assert template["decision"] == "PENDING"
    assert template["rights_record"]["rights_owner"] == RIGHTS["rights_owner"]
    assert len(template["rows"]) == 144
    assert {row["phase"] for row in template["rows"]} == {
        "discovery",
        "confirmation",
    }
    assert {row["pack"] for row in template["rows"]} == {"D1", "D2", "C1", "C2"}
    assert {row["split_role"] for row in template["rows"]} == {
        "training",
        "evaluation",
    }
    assert all(not any(row["checks"].values()) for row in template["rows"])
    assert template["governance"]["agent_review_is_not_human_review"] is True


def test_role_label_and_one_missing_row_check_abort(candidate):
    public, custodian = candidate
    role = completed_review(candidate)
    role["reviewer_identity"] = "human reviewer"
    with pytest.raises(admission.AdmissionError, match="role label"):
        admission.seal_review(
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            draft=role,
        )

    missing = completed_review(candidate)
    missing["rows"][17]["checks"]["caption_semantics_accepted"] = False
    with pytest.raises(admission.AdmissionError, match="unapproved check"):
        admission.seal_review(
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            draft=missing,
        )


def test_exact_row_tampering_aborts_even_if_every_boolean_says_pass(candidate):
    public, custodian = candidate
    forged = completed_review(candidate)
    forged["rows"][73]["image_sha256"] = "0" * 64
    with pytest.raises(admission.AdmissionError, match="identity mismatch"):
        admission.seal_review(
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            draft=forged,
        )


def test_pass_emits_three_admissions_but_keeps_gpu_and_deploy_closed(candidate):
    public, custodian = candidate
    sealed = admission.seal_review(
        public_root=public,
        custodian_root=custodian,
        public_boundary_roots=(public.parent,),
        draft=completed_review(candidate),
    )
    root, receipts = admission.build_admissions(
        public_root=public,
        custodian_root=custodian,
        discovery_key=DISCOVERY_KEY,
        confirmation_key=CONFIRMATION_KEY,
        public_boundary_roots=(public.parent,),
        sealed_review=sealed,
        generator_identity_probe=generator_probe,
    )
    assert set(receipts) == {"social", "product", "logo_ui"}
    assert receipts["social"]["counts"] == {"discovery": 36, "confirmation": 36}
    assert receipts["product"]["counts"] == {"discovery": 18, "confirmation": 18}
    assert receipts["logo_ui"]["counts"] == {"discovery": 18, "confirmation": 18}
    assert set(receipts["social"]["packs"]) == {"D1", "D2", "C1", "C2"}
    for pack in ("D1", "D2"):
        record = receipts["social"]["packs"][pack]
        assert record["phase"] == "discovery"
        assert record["row_count"] == 18
        assert record["training_row_count"] == 10
        assert record["evaluation_row_count"] == 8
        assert record["training_inventory"]["file_count"] == 20
        assert record["evaluation_inventory"]["file_count"] == 16
        assert all(
            "/" not in item["path"] and "\\" not in item["path"]
            for item in record["training_inventory"]["files"]
        )
        assert (
            record["training_inventory_sha256"]
            == record["training_inventory"]["semantic_sha256"]
        )
    for pack in ("C1", "C2"):
        assert receipts["social"]["packs"][pack].keys() == {
            "phase",
            "row_count",
            "training_row_count",
            "evaluation_row_count",
            "semantic_commitment_sha256",
        }
    assert all(item["status"] == "PASS" for item in receipts.values())
    assert all(
        item["governance"]["gpu_execution_authorized"] is False
        for item in receipts.values()
    )
    assert all(
        item["governance"]["operator_attested_named_human_review"] is True
        and item["governance"]["owner_ratification_required_for_gpu"] is True
        for item in receipts.values()
    )
    assert root["receipts"] == receipts
    assert root["authorization"] == {
        "fixture_admission_authorized": True,
        "gpu_execution_authorized": False,
        "deployment_authorized": False,
    }
    public_bytes = admission.canonical_bytes(root) + b"".join(
        admission.canonical_bytes(receipt) for receipt in receipts.values()
    )
    assert b"Atulya Shetty" not in public_bytes
    confirmation = renderer.verify_candidate(
        public, custodian, public_boundary_roots=(public.parent,)
    )["confirmation_manifest"]
    private_row_id = next(iter(next(iter(confirmation["families"].values()))["rows"]))[
        "row_id"
    ].encode("ascii")
    assert private_row_id not in public_bytes


def test_private_review_records_cannot_be_written_in_candidate_or_repo(candidate):
    public, custodian = candidate
    with pytest.raises(admission.AdmissionError, match="must be outside"):
        admission._private_record_path(
            public / "review.json",
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            label="review",
        )
    with pytest.raises(admission.AdmissionError, match="must be outside"):
        admission._private_record_path(
            custodian / "review.json",
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            label="review",
        )
    with pytest.raises(admission.AdmissionError, match="must be outside"):
        admission._private_record_path(
            admission.REPO_ROOT / "review.json",
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            label="review",
        )


def test_private_record_rejects_existing_registered_sibling_worktree(
    candidate, tmp_path, monkeypatch
):
    public, custodian = candidate
    sibling = tmp_path / "sibling-worktree"
    sibling.mkdir()
    monkeypatch.setattr(
        admission.renderer,
        "_registered_worktree_roots",
        lambda: (admission.renderer.EXECUTABLE_REPOSITORY_ROOT, sibling),
    )
    target = sibling / "PRIVATE-CONFIRMATION-REVEAL.json"
    with pytest.raises(admission.AdmissionError, match="registered-worktree"):
        admission._private_record_path(
            target,
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            label="confirmation reveal",
        )
    assert not target.exists()


def test_private_record_rejects_case_alias_of_registered_worktree(
    candidate, tmp_path, monkeypatch
):
    public, custodian = candidate
    sibling = tmp_path / "SiblingWorktree"
    sibling.mkdir()
    alias = sibling.with_name(sibling.name.swapcase())
    try:
        same = alias.samefile(sibling)
    except FileNotFoundError:
        pytest.skip("filesystem is case-sensitive")
    if not same:
        pytest.skip("filesystem is case-sensitive")
    monkeypatch.setattr(
        admission.renderer,
        "_registered_worktree_roots",
        lambda: (admission.renderer.EXECUTABLE_REPOSITORY_ROOT, sibling),
    )
    with pytest.raises(admission.AdmissionError, match="registered-worktree"):
        admission._private_record_path(
            alias / "PRIVATE-CONFIRMATION-REVEAL.json",
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            label="confirmation reveal",
        )


def test_private_record_rejects_missing_prunable_registered_worktree(
    candidate, tmp_path, monkeypatch
):
    public, custodian = candidate
    missing = tmp_path / "missing-prunable-worktree"
    monkeypatch.setattr(
        admission.renderer,
        "_registered_worktree_roots",
        lambda: (admission.renderer.EXECUTABLE_REPOSITORY_ROOT, missing),
    )
    with pytest.raises(admission.AdmissionError, match="registered-worktree"):
        admission._private_record_path(
            missing / "private" / "PRIVATE-CONFIRMATION-REVEAL.json",
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            label="confirmation reveal",
        )


def test_private_record_fails_closed_when_worktree_inventory_fails(
    candidate, tmp_path, monkeypatch
):
    public, custodian = candidate

    def fail_inventory():
        raise admission.renderer.FixtureError("worktree inventory failed closed")

    monkeypatch.setattr(
        admission.renderer, "_registered_worktree_roots", fail_inventory
    )
    with pytest.raises(admission.AdmissionError, match="inventory failed closed"):
        admission._private_record_path(
            tmp_path / "outside" / "review.json",
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            label="review",
        )


def test_private_record_read_rejects_parent_swap_after_validation(
    candidate, tmp_path, monkeypatch
):
    public, custodian = candidate
    safe_parent = tmp_path / "safe-private"
    safe_parent.mkdir()
    record = safe_parent / "review.json"
    record.write_text('{"source":"safe"}', encoding="ascii")
    displaced = tmp_path / "safe-private-displaced"
    worktree = tmp_path / "registered-worktree"
    worktree.mkdir()
    (worktree / record.name).write_text(
        '{"source":"registered-worktree"}', encoding="ascii"
    )
    monkeypatch.setattr(
        admission.renderer,
        "_registered_worktree_roots",
        lambda: (admission.renderer.EXECUTABLE_REPOSITORY_ROOT, worktree),
    )
    original_check = admission._private_record_path
    swapped = False

    def validate_then_swap(*args, **kwargs):
        nonlocal swapped
        checked = original_check(*args, **kwargs)
        if not swapped:
            safe_parent.rename(displaced)
            safe_parent.symlink_to(worktree, target_is_directory=True)
            swapped = True
        return checked

    monkeypatch.setattr(admission, "_private_record_path", validate_then_swap)
    with pytest.raises(admission.AdmissionError, match="cannot safely open"):
        admission._load_private_json(
            record,
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            label="human review draft",
        )
    assert swapped is True


def test_private_record_read_rejects_public_hard_link(candidate, tmp_path) -> None:
    public, custodian = candidate
    private_parent = tmp_path / "private-records-hardlink"
    private_parent.mkdir()
    record = private_parent / "review.json"
    record.write_bytes(admission.canonical_bytes({"source": "private"}))
    public_link = public.parent / "published-review.json"
    os.link(record, public_link)
    assert record.samefile(public_link)
    assert record.stat().st_nlink == 2
    with pytest.raises(admission.AdmissionError, match="single-link regular file"):
        admission._load_private_json(
            record,
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            label="human review draft",
        )


def test_private_record_read_rejects_hard_link_created_mid_read(
    candidate, tmp_path, monkeypatch
) -> None:
    public, custodian = candidate
    private_parent = tmp_path / "private-records-midread-hardlink"
    private_parent.mkdir()
    record = private_parent / "review.json"
    record.write_bytes(admission.canonical_bytes({"source": "private"}))
    public_link = public.parent / "published-midread-review.json"
    original_read = admission.renderer.os.read
    linked = False
    record_identity = (record.stat().st_dev, record.stat().st_ino)

    def link_then_read(descriptor, length):
        nonlocal linked
        metadata = os.fstat(descriptor)
        if not linked and (metadata.st_dev, metadata.st_ino) == record_identity:
            os.link(record, public_link)
            linked = True
        return original_read(descriptor, length)

    monkeypatch.setattr(admission.renderer.os, "read", link_then_read)
    with pytest.raises(admission.AdmissionError, match="descriptor-bound read"):
        admission._load_private_json(
            record,
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            label="human review draft",
        )
    assert linked is True
    assert record.samefile(public_link)


def test_private_record_read_rejects_hard_link_after_first_bound_read(
    candidate, tmp_path, monkeypatch
) -> None:
    public, custodian = candidate
    private_parent = tmp_path / "private-records-postread-hardlink"
    private_parent.mkdir()
    record = private_parent / "review.json"
    record.write_bytes(admission.canonical_bytes({"source": "private"}))
    public_link = public.parent / "published-postread-review.json"
    original_read = admission.renderer._read_regular_at
    reads = 0

    def read_then_link(*args, **kwargs):
        nonlocal reads
        payload = original_read(*args, **kwargs)
        reads += 1
        if reads == 1:
            os.link(record, public_link)
        return payload

    monkeypatch.setattr(admission.renderer, "_read_regular_at", read_then_link)
    with pytest.raises(admission.AdmissionError, match="single-link regular file"):
        admission._load_private_json(
            record,
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            label="human review draft",
        )
    assert reads == 1
    assert record.samefile(public_link)


def test_private_record_publish_rechecks_worktree_inventory_without_unsafe_cleanup(
    candidate, tmp_path, monkeypatch
):
    public, custodian = candidate
    private_parent = tmp_path / "private-records"
    private_parent.mkdir()
    target = private_parent / "PRIVATE-CONFIRMATION-REVEAL.json"
    registered = False
    original_write = admission.renderer._write_exclusive_at

    def worktrees():
        roots = [admission.renderer.EXECUTABLE_REPOSITORY_ROOT]
        if registered:
            roots.append(private_parent)
        return tuple(roots)

    def register_during_publish(*args, **kwargs):
        nonlocal registered
        validation = kwargs["post_write_validation"]

        def register_then_validate(descriptor, metadata):
            nonlocal registered
            registered = True
            validation(descriptor, metadata)

        return original_write(
            *args, **{**kwargs, "post_write_validation": register_then_validate}
        )

    monkeypatch.setattr(admission.renderer, "_registered_worktree_roots", worktrees)
    monkeypatch.setattr(
        admission.renderer, "_write_exclusive_at", register_during_publish
    )
    with pytest.raises(admission.AdmissionError, match="registered-worktree"):
        admission._write_private_new(
            target,
            {"revealed_rows": ["private"]},
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            label="confirmation reveal",
        )
    assert registered is True
    assert target.exists()
    assert target.read_bytes() == b""


def test_private_record_failed_postwrite_scrubs_created_inode_not_replacement(
    candidate, tmp_path, monkeypatch
):
    public, custodian = candidate
    private_parent = tmp_path / "private-records"
    private_parent.mkdir()
    target = private_parent / "PRIVATE-CONFIRMATION-REVEAL.json"
    moved = private_parent / "created-private-record-moved"
    original_assert = admission._assert_private_parent_bound
    calls = 0

    def swap_at_postwrite(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            target.rename(moved)
            target.write_bytes(b"foreign replacement")
            raise admission.AdmissionError("post-write custody changed")
        return original_assert(*args, **kwargs)

    monkeypatch.setattr(admission, "_assert_private_parent_bound", swap_at_postwrite)
    with pytest.raises(admission.AdmissionError, match="post-write custody changed"):
        admission._write_private_new(
            target,
            {"revealed_rows": ["private"]},
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            label="confirmation reveal",
        )
    assert target.read_bytes() == b"foreign replacement"
    assert moved.read_bytes() == b""


def test_private_record_real_postwrite_check_rejects_destination_swap(
    candidate, tmp_path, monkeypatch
):
    public, custodian = candidate
    private_parent = tmp_path / "private-records"
    private_parent.mkdir()
    target = private_parent / "PRIVATE-CONFIRMATION-REVEAL.json"
    moved = private_parent / "created-private-record-moved"
    original_assert = admission._assert_private_parent_bound
    calls = 0

    def swap_then_run_real_check(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            target.rename(moved)
            target.write_bytes(b"foreign replacement")
        return original_assert(*args, **kwargs)

    monkeypatch.setattr(
        admission, "_assert_private_parent_bound", swap_then_run_real_check
    )
    with pytest.raises(
        admission.AdmissionError, match="live target changed during publication"
    ):
        admission._write_private_new(
            target,
            {"revealed_rows": ["private"]},
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            label="confirmation reveal",
        )
    assert target.read_bytes() == b"foreign replacement"
    assert moved.read_bytes() == b""


def test_private_record_publish_rejects_live_parent_replacement_after_link_check(
    candidate, tmp_path, monkeypatch
) -> None:
    public, custodian = candidate
    private_parent = tmp_path / "private-parent-final-race"
    private_parent.mkdir()
    displaced = tmp_path / "private-parent-final-race-displaced"
    target = private_parent / "PRIVATE-CONFIRMATION-REVEAL.json"
    moved_target = displaced / target.name
    original_target_check = admission._assert_private_target_bound
    swapped = False

    def replace_parent_then_check(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            private_parent.rename(displaced)
            private_parent.mkdir()
            target.write_bytes(b"foreign-publication")
            swapped = True
        return original_target_check(*args, **kwargs)

    monkeypatch.setattr(
        admission, "_assert_private_target_bound", replace_parent_then_check
    )
    with pytest.raises(admission.AdmissionError, match="parent changed|live target"):
        admission._write_private_new(
            target,
            {"revealed_rows": ["private"]},
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            label="confirmation reveal",
        )
    assert swapped is True
    assert target.read_bytes() == b"foreign-publication"
    assert moved_target.read_bytes() == b""


def test_private_record_publish_rejects_parent_replacement_after_writer_returns(
    candidate, tmp_path, monkeypatch
) -> None:
    public, custodian = candidate
    private_parent = tmp_path / "private-parent-after-writer"
    private_parent.mkdir()
    displaced = tmp_path / "private-parent-after-writer-displaced"
    target = private_parent / "PRIVATE-CONFIRMATION-REVEAL.json"
    moved_target = displaced / target.name
    original_write = admission.renderer._write_exclusive_at
    swapped = False

    def write_then_replace(*args, **kwargs):
        nonlocal swapped
        result = original_write(*args, **kwargs)
        private_parent.rename(displaced)
        private_parent.mkdir()
        target.write_bytes(b"foreign-after-writer")
        swapped = True
        return result

    monkeypatch.setattr(admission.renderer, "_write_exclusive_at", write_then_replace)
    with pytest.raises(admission.AdmissionError, match="parent changed|live payload"):
        admission._write_private_new(
            target,
            {"revealed_rows": ["private"]},
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            label="confirmation reveal",
        )
    assert swapped is True
    assert target.read_bytes() == b"foreign-after-writer"
    assert moved_target.read_bytes() == b""


def test_private_record_publish_rechecks_parent_after_final_payload_read(
    candidate, tmp_path, monkeypatch
) -> None:
    public, custodian = candidate
    private_parent = tmp_path / "private-parent-after-final-read"
    private_parent.mkdir()
    displaced = tmp_path / "private-parent-after-final-read-displaced"
    target = private_parent / "PRIVATE-CONFIRMATION-REVEAL.json"
    moved_target = displaced / target.name
    original_read = admission.renderer._read_regular_at
    reads = 0

    def read_then_replace(*args, **kwargs):
        nonlocal reads
        payload = original_read(*args, **kwargs)
        reads += 1
        if reads == 2:
            private_parent.rename(displaced)
            private_parent.mkdir()
            target.write_bytes(b"foreign-after-final-read")
        return payload

    monkeypatch.setattr(admission.renderer, "_read_regular_at", read_then_replace)
    with pytest.raises(admission.AdmissionError, match="parent changed"):
        admission._write_private_new(
            target,
            {"revealed_rows": ["private"]},
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            label="confirmation reveal",
        )
    assert reads == 2
    assert target.read_bytes() == b"foreign-after-final-read"
    assert moved_target.read_bytes() == b""


def test_cli_revalidates_private_output_authority_after_helper_returns(
    candidate, tmp_path, monkeypatch
) -> None:
    public, custodian = candidate
    private_parent = tmp_path / "private-cli-output"
    private_parent.mkdir()
    displaced = tmp_path / "private-cli-output-displaced"
    target = private_parent / "PRIVATE-CONFIRMATION-REVEAL.json"
    moved_target = displaced / target.name

    def command(_args):
        admission._write_private_new(
            target,
            {"revealed_rows": ["private"]},
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            label="confirmation reveal",
        )
        private_parent.rename(displaced)
        private_parent.mkdir()
        target.write_bytes(b"foreign-after-helper")
        return 0

    monkeypatch.setattr(admission, "_parse", lambda _argv: object())
    monkeypatch.setattr(admission, "_main_with_held_private_descriptors", command)
    with pytest.raises(admission.AdmissionError, match="parent changed|live target"):
        admission.main([])
    assert target.read_bytes() == b"foreign-after-helper"
    assert moved_target.read_bytes() == b""
    assert admission._ACTIVE_PRIVATE_AUTHORITIES is None


def test_cli_revalidates_private_input_link_count_after_helper_returns(
    candidate, tmp_path, monkeypatch
) -> None:
    public, custodian = candidate
    private_parent = tmp_path / "private-cli-input"
    private_parent.mkdir()
    record = private_parent / "review.json"
    record.write_bytes(admission.canonical_bytes({"source": "private"}))
    public_link = public.parent / "published-after-helper.json"

    def command(_args):
        assert admission._load_private_json(
            record,
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            label="human review draft",
        ) == {"source": "private"}
        os.link(record, public_link)
        return 0

    monkeypatch.setattr(admission, "_parse", lambda _argv: object())
    monkeypatch.setattr(admission, "_main_with_held_private_descriptors", command)
    with pytest.raises(
        admission.AdmissionError, match="live target changed|single-link regular file"
    ):
        admission.main([])
    assert record.samefile(public_link)
    assert record.stat().st_nlink == 2
    assert admission._ACTIVE_PRIVATE_AUTHORITIES is None


def test_admission_rejects_relocated_custodian_inside_evidence_boundary(
    candidate, tmp_path: Path
) -> None:
    public, custodian = candidate
    evidence = tmp_path / "public-evidence"
    evidence.mkdir()
    copied_public = evidence / "candidate"
    relocated_custodian = evidence / "private-confirmation"
    shutil.copytree(public, copied_public)
    shutil.copytree(custodian, relocated_custodian)
    with pytest.raises(
        admission.AdmissionError, match="custodian output must be outside"
    ):
        admission.build_review_template(
            copied_public,
            relocated_custodian,
            public_boundary_roots=(evidence,),
        )


def test_sealed_review_digest_tampering_aborts(candidate):
    public, custodian = candidate
    sealed = admission.seal_review(
        public_root=public,
        custodian_root=custodian,
        public_boundary_roots=(public.parent,),
        draft=completed_review(candidate),
    )
    tampered = copy.deepcopy(sealed)
    tampered["rows"][0]["notes"] = "changed after sealing"
    with pytest.raises(admission.AdmissionError, match="digest mismatch"):
        admission.build_admissions(
            public_root=public,
            custodian_root=custodian,
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            public_boundary_roots=(public.parent,),
            sealed_review=tampered,
            generator_identity_probe=generator_probe,
        )


def test_admission_rejects_attested_tree_that_is_not_the_executed_revision(candidate):
    public, custodian = candidate
    sealed = admission.seal_review(
        public_root=public,
        custodian_root=custodian,
        public_boundary_roots=(public.parent,),
        draft=completed_review(candidate),
    )

    def wrong_tree():
        value = generator_probe()
        value["tree"] = "f" * 40
        return value

    with pytest.raises(admission.AdmissionError, match="generator tree"):
        admission.build_admissions(
            public_root=public,
            custodian_root=custodian,
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            public_boundary_roots=(public.parent,),
            sealed_review=sealed,
            generator_identity_probe=wrong_tree,
        )


def test_admission_rejects_equal_keys_before_trusting_stubbed_replay(
    candidate, monkeypatch
) -> None:
    public, custodian = candidate
    sealed = admission.seal_review(
        public_root=public,
        custodian_root=custodian,
        public_boundary_roots=(public.parent,),
        draft=completed_review(candidate),
    )
    called = False

    def fake_replay(**_kwargs):
        nonlocal called
        called = True
        return {"status": "PASS"}

    monkeypatch.setattr(admission.renderer, "verify_replay", fake_replay)
    with pytest.raises(admission.AdmissionError, match="must be distinct"):
        admission.build_admissions(
            public_root=public,
            custodian_root=custodian,
            discovery_key=DISCOVERY_KEY,
            confirmation_key=DISCOVERY_KEY,
            public_boundary_roots=(public.parent,),
            sealed_review=sealed,
            generator_identity_probe=generator_probe,
        )
    assert called is False


def test_admission_rejects_mismatched_phase_commitment(
    candidate, tmp_path: Path
) -> None:
    public, custodian = candidate
    copied_public = tmp_path / "public-boundary" / "public"
    copied_custodian = tmp_path / "custodian"
    copied_public.parent.mkdir()
    shutil.copytree(public, copied_public)
    shutil.copytree(custodian, copied_custodian)
    confirmation_path = copied_custodian / "CONFIRMATION-MANIFEST.json"
    confirmation = json.loads(confirmation_path.read_text(encoding="ascii"))
    confirmation["phase_key_commitment_sha256"] = renderer._phase_key_commitment(
        DISCOVERY_KEY, renderer.DISCOVERY_DOMAIN
    )
    _rewrite_semantic(confirmation_path, confirmation)
    candidate_path = copied_public / "CANDIDATE-MANIFEST.json"
    candidate_record = json.loads(candidate_path.read_text(encoding="ascii"))
    confirmation_sha = hashlib.sha256(confirmation_path.read_bytes()).hexdigest()
    candidate_record["confirmation"]["custodian_manifest_sha256"] = confirmation_sha
    for family in candidate_record["families"].values():
        family["confirmation"]["custodian_manifest_sha256"] = confirmation_sha
    _rewrite_semantic(candidate_path, candidate_record)
    draft = admission.build_review_template(
        copied_public,
        copied_custodian,
        public_boundary_roots=(copied_public.parent,),
    )
    draft["reviewer_identity"] = "Atulya Shetty"
    draft["reviewed_at_utc"] = "2026-08-11T02:00:00Z"
    draft["decision"] = "PASS"
    for row in draft["rows"]:
        for key in row["checks"]:
            row["checks"][key] = True
    sealed = admission.seal_review(
        public_root=copied_public,
        custodian_root=copied_custodian,
        public_boundary_roots=(copied_public.parent,),
        draft=draft,
    )
    with pytest.raises(admission.AdmissionError, match="phase keys do not match"):
        admission.build_admissions(
            public_root=copied_public,
            custodian_root=copied_custodian,
            discovery_key=DISCOVERY_KEY,
            confirmation_key=CONFIRMATION_KEY,
            public_boundary_roots=(copied_public.parent,),
            sealed_review=sealed,
            generator_identity_probe=generator_probe,
        )


def test_generator_identity_rejects_ambient_contract_drift(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    relative = Path("ops/experiments/week7")
    target = repo / relative
    target.mkdir(parents=True)
    renderer_copy = target / "hke_procedural_renderer.py"
    admission_copy = target / "hke_fixture_admission.py"
    factor_copy = target / "run_hke_factorial.py"
    contract_copy = target / "hke_fixture_contract.json"
    shutil.copy2(Path(renderer.__file__), renderer_copy)
    shutil.copy2(admission.__file__, admission_copy)
    shutil.copy2(admission.FACTOR_AUTHORITY_PATH, factor_copy)
    shutil.copy2(renderer.DECLARATIVE_CONTRACT_PATH, contract_copy)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=SN56 Test",
            "-c",
            "user.email=sn56@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    contract_copy.write_text('{"schema":999}\n', encoding="ascii")
    monkeypatch.setattr(admission, "REPO_ROOT", repo)
    monkeypatch.setattr(admission, "RENDERER_PATH", renderer_copy)
    monkeypatch.setattr(admission, "SCRIPT_PATH", admission_copy)
    monkeypatch.setattr(admission.renderer, "DECLARATIVE_CONTRACT_PATH", contract_copy)
    with pytest.raises(admission.AdmissionError, match="ambient fixture contract"):
        admission._current_generator_identity()


def test_discovery_receipt_rejects_training_evaluation_transplant(candidate) -> None:
    public, custodian = candidate
    verified = renderer.verify_candidate(
        public, custodian, public_boundary_roots=(public.parent,)
    )
    family = copy.deepcopy(
        verified["candidate_manifest"]["families"]["W7-HKE-SOCIAL-A"]
    )
    pack = family["discovery"]["packs"]["D1"]
    transplanted = copy.deepcopy(pack["splits"]["training"]["rows"][:8])
    for row in transplanted:
        row["split_role"] = "evaluation"
    pack["splits"]["evaluation"]["rows"] = transplanted
    with pytest.raises(admission.AdmissionError, match="overlaps by row_id"):
        admission._public_pack_receipts(family)


def test_discovery_receipt_rejects_stable_semantic_duplicate_across_split(
    candidate,
) -> None:
    public, custodian = candidate
    verified = renderer.verify_candidate(
        public, custodian, public_boundary_roots=(public.parent,)
    )
    family = copy.deepcopy(
        verified["candidate_manifest"]["families"]["W7-HKE-PRODUCT-A"]
    )
    pack = family["discovery"]["packs"]["D1"]
    training = pack["splits"]["training"]["rows"][0]
    evaluation = pack["splits"]["evaluation"]["rows"][0]
    evaluation["group_identity_sha256"] = training["group_identity_sha256"]
    with pytest.raises(admission.AdmissionError, match="group_identity_sha256"):
        admission._public_pack_receipts(family)


def test_owner_ratification_requires_named_owner_and_stays_gpu_closed(candidate):
    _public, _custodian, sealed, root, _receipts = admitted(candidate)
    template = admission.build_owner_ratification_template(
        admission_set=root, sealed_review=sealed
    )
    assert template["status"] == "PENDING_NAMED_OWNER_RATIFICATION"
    assert template["owner_identity"] == ""
    assert template["decision"] == "PENDING"
    assert template["admission_set_sha256"] == root["admission_set_sha256"]
    assert template["human_review_sha256"] == sealed["review_sha256"]
    assert template["generator_revision"] == root["generator_revision"]
    assert template["governance"]["gpu_execution_authorized"] is False

    forged = copy.deepcopy(template)
    forged["owner_identity"] = "owner"
    forged["ratified_at_utc"] = "2026-08-12T00:00:00Z"
    forged["decision"] = "RATIFY_FOR_PLAN_CONSUMPTION"
    with pytest.raises(admission.AdmissionError, match="role label"):
        admission.seal_owner_ratification(
            draft=forged, admission_set=root, sealed_review=sealed
        )

    completed = copy.deepcopy(template)
    completed["owner_identity"] = "Atulya Shetty"
    completed["ratified_at_utc"] = "2026-08-12T00:00:00Z"
    completed["decision"] = "RATIFY_FOR_PLAN_CONSUMPTION"
    ratification = admission.seal_owner_ratification(
        draft=completed, admission_set=root, sealed_review=sealed
    )
    assert ratification["status"] == "SEALED_OPERATOR_ATTESTED_OWNER_RATIFICATION"
    assert ratification["governance"]["gpu_execution_authorized"] is False
    assert (
        admission.validate_owner_ratification(
            ratification, admission_set=root, sealed_review=sealed
        )
        == ratification
    )


def test_ratification_cli_requires_a_human_completed_draft(tmp_path: Path) -> None:
    common = [
        "--public-root",
        str(tmp_path / "public-boundary" / "public"),
        "--custodian-root",
        str(tmp_path / "custodian"),
        "--public-boundary-root",
        str(tmp_path / "public-boundary"),
        "--admission-set",
        str(tmp_path / "ADMISSION-SET.json"),
        "--sealed-review",
        str(tmp_path / "sealed-review.json"),
        "--output",
        str(tmp_path / "owner-ratification.json"),
    ]
    template = admission._parse(["ratification-template", *common])
    assert template.command == "ratification-template"
    with pytest.raises(SystemExit):
        admission._parse(["seal-ratification", *common])
    sealed = admission._parse(
        ["seal-ratification", *common, "--draft", str(tmp_path / "draft.json")]
    )
    assert sealed.command == "seal-ratification"
    assert sealed.draft == tmp_path / "draft.json"


def test_stale_schema2_admission_and_ratification_are_rejected(candidate):
    _public, _custodian, sealed, root, _receipts = admitted(candidate)
    stale = copy.deepcopy(root)
    stale["schema"] = 2
    stale_body = dict(stale)
    stale_body.pop("admission_set_sha256")
    stale["admission_set_sha256"] = admission.semantic_sha256(stale_body)
    with pytest.raises(admission.AdmissionError, match="has not passed"):
        admission.build_owner_ratification_template(
            admission_set=stale, sealed_review=sealed
        )


def test_confirmation_reveal_requires_post_d2_freeze_and_exact_commitment(
    candidate, factor_authority_stub
):
    public, custodian, _sealed, root, receipts = admitted(candidate)
    commitment = receipts["social"]["packs"]["C1"]["semantic_commitment_sha256"]
    frozen = confirmation_freeze(receipts)
    verified = renderer.verify_candidate(
        public, custodian, public_boundary_roots=(public.parent,)
    )
    private_pack = verified["confirmation_manifest"]["families"]["W7-HKE-SOCIAL-A"][
        "packs"
    ]["C1"]
    assert commitment == renderer.semantic_sha256(private_pack["rows"])
    assert frozen["confirmation_commitments"]["social"]["C1"] == commitment
    reveal = admission.build_confirmation_reveal(
        public_root=public,
        custodian_root=custodian,
        public_boundary_roots=(public.parent,),
        admission_set=root,
        family="social",
        pack="C1",
        confirmation_authority=frozen,
    )
    assert reveal["privacy"] == "custodian_private_not_public_admission"
    assert len(reveal["revealed_rows"]) == 18
    assert renderer.semantic_sha256(reveal["revealed_rows"]) == commitment
    assert reveal["training_inventory"]["file_count"] == 20
    assert reveal["evaluation_inventory"]["file_count"] == 16
    assert reveal["authorization"]["gpu_execution_authorized"] is False
    assert (
        admission.validate_confirmation_reveal(
            reveal,
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            admission_set=root,
            confirmation_authority=frozen,
        )
        == reveal
    )

    forged_rows = copy.deepcopy(reveal)
    forged_rows["revealed_rows"][0]["row_id"] = "forged-row"
    forged_body = dict(forged_rows)
    forged_body.pop("confirmation_reveal_sha256")
    forged_rows["confirmation_reveal_sha256"] = admission.semantic_sha256(forged_body)
    with pytest.raises(admission.AdmissionError, match="does not reproduce"):
        admission.validate_confirmation_reveal(
            forged_rows,
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            admission_set=root,
            confirmation_authority=frozen,
        )

    pre_freeze = copy.deepcopy(frozen)
    pre_freeze["status"] = "HOLD_NO_CONFIRMATION_REVEAL"
    pre_freeze_body = dict(pre_freeze)
    pre_freeze_body.pop("confirmation_freeze_sha256")
    pre_freeze["confirmation_freeze_sha256"] = admission.semantic_sha256(
        pre_freeze_body
    )
    with pytest.raises(admission.AdmissionError, match="does not reproduce"):
        admission.build_confirmation_reveal(
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            admission_set=root,
            family="social",
            pack="C1",
            confirmation_authority=pre_freeze,
        )

    mismatch = confirmation_freeze(receipts)
    mismatch["confirmation_commitments"]["social"]["C1"] = "f" * 64
    mismatch_body = dict(mismatch)
    mismatch_body.pop("confirmation_freeze_sha256")
    mismatch["confirmation_freeze_sha256"] = admission.semantic_sha256(mismatch_body)
    with pytest.raises(admission.AdmissionError, match="does not reproduce"):
        admission.build_confirmation_reveal(
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            admission_set=root,
            family="social",
            pack="C1",
            confirmation_authority=mismatch,
        )

    bad_hash = copy.deepcopy(frozen)
    bad_hash["confirmation_freeze_sha256"] = "0" * 64
    with pytest.raises(admission.AdmissionError, match="digest mismatch"):
        admission.build_confirmation_reveal(
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            admission_set=root,
            family="social",
            pack="C1",
            confirmation_authority=bad_hash,
        )


def test_product_and_logo_guardrail_reveals_use_same_post_d2_freeze(
    candidate, factor_authority_stub
):
    public, custodian, _sealed, root, receipts = admitted(candidate)
    frozen = confirmation_freeze(receipts)
    for family in ("product", "logo_ui"):
        reveal = admission.build_confirmation_reveal(
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            admission_set=root,
            family=family,
            pack="C1",
            confirmation_authority=frozen,
        )
        assert reveal["family"] == family
        assert reveal["status"] == "PRIVATE_C1_REVEALED_AFTER_D2_CONFIRMATION_FREEZE"
        assert (
            reveal["confirmation_authority_sha256"]
            == frozen["confirmation_freeze_sha256"]
        )


def test_self_rehashed_forged_freeze_cannot_authorize_c1(
    candidate, factor_authority_stub
):
    public, custodian, _sealed, root, receipts = admitted(candidate)
    forged = confirmation_freeze(receipts)
    forged["discovery_decision"]["decision"] = "ADVANCE_POST_HOC"
    forged_body = dict(forged)
    forged_body.pop("confirmation_freeze_sha256")
    forged["confirmation_freeze_sha256"] = admission.semantic_sha256(forged_body)
    with pytest.raises(admission.AdmissionError, match="does not reproduce"):
        admission.build_confirmation_reveal(
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            admission_set=root,
            family="social",
            pack="C1",
            confirmation_authority=forged,
        )


def test_c2_requires_separate_borderline_trigger_authority(
    candidate, factor_authority_stub
):
    public, custodian, _sealed, root, receipts = admitted(candidate)
    frozen = confirmation_freeze(receipts)
    with pytest.raises(admission.AdmissionError, match="C2 authority"):
        admission.build_confirmation_reveal(
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            admission_set=root,
            family="social",
            pack="C2",
            confirmation_authority=frozen,
        )
    authority = c2_authority(receipts, frozen)
    reveal = admission.build_confirmation_reveal(
        public_root=public,
        custodian_root=custodian,
        public_boundary_roots=(public.parent,),
        admission_set=root,
        family="social",
        pack="C2",
        confirmation_authority=authority,
    )
    assert reveal["status"] == "PRIVATE_C2_REVEALED_AFTER_BORDERLINE_C1_TRIGGER"
    assert reveal["confirmation_authority_sha256"] == authority["c2_authority_sha256"]

    forged = copy.deepcopy(authority)
    forged["trigger"]["reason"] = "post-hoc self-asserted"
    forged_body = dict(forged)
    forged_body.pop("c2_authority_sha256")
    forged["c2_authority_sha256"] = admission.semantic_sha256(forged_body)
    with pytest.raises(admission.AdmissionError, match="does not reproduce"):
        admission.build_confirmation_reveal(
            public_root=public,
            custodian_root=custodian,
            public_boundary_roots=(public.parent,),
            admission_set=root,
            family="social",
            pack="C2",
            confirmation_authority=forged,
        )


def _committed_generator_copy(tmp_path: Path):
    repo = tmp_path / "generator-repo"
    relative = Path("ops/experiments/week7")
    target = repo / relative
    target.mkdir(parents=True)
    renderer_copy = target / "hke_procedural_renderer.py"
    admission_copy = target / "hke_fixture_admission.py"
    factor_copy = target / "run_hke_factorial.py"
    contract_copy = target / "hke_fixture_contract.json"
    shutil.copy2(Path(renderer.__file__), renderer_copy)
    shutil.copy2(admission.__file__, admission_copy)
    shutil.copy2(admission.FACTOR_AUTHORITY_PATH, factor_copy)
    shutil.copy2(renderer.DECLARATIVE_CONTRACT_PATH, contract_copy)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=SN56 Test",
            "-c",
            "user.email=sn56@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    module, source_sha = admission._load_committed_source_module(
        repo_root=repo,
        source_path=admission.RENDERER_SOURCE_PATH,
        ambient_path=renderer_copy,
        module_name="week7_hke_renderer_committed_test_copy",
    )
    module.GENERATOR_REPOSITORY = str(repo)
    return repo, renderer_copy, admission_copy, module, source_sha


def test_dirty_renderer_aborts_before_any_ambient_bytes_execute(tmp_path: Path) -> None:
    repo, renderer_copy, _admission_copy, module, _source_sha = (
        _committed_generator_copy(tmp_path)
    )
    committed = renderer_copy.read_bytes()
    renderer_copy.write_bytes(
        committed + b'\nraise AssertionError("ambient renderer executed")\n'
    )
    with pytest.raises(admission.AdmissionError, match="ambient authority differs"):
        admission._load_committed_source_module(
            repo_root=repo,
            source_path=admission.RENDERER_SOURCE_PATH,
            ambient_path=renderer_copy,
            module_name="week7_hke_renderer_dirty_test_copy",
        )
    assert module.GENERATOR_REPOSITORY == str(repo)


def test_dirty_factor_authority_aborts_before_exec(tmp_path: Path) -> None:
    repo = tmp_path / "factor-repo"
    source = repo / admission.FACTOR_AUTHORITY_SOURCE_PATH
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 'committed'\n", encoding="ascii")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=SN56 Test",
            "-c",
            "user.email=sn56@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    source.write_text(
        "raise AssertionError('dirty factor authority executed')\n",
        encoding="ascii",
    )
    with pytest.raises(admission.AdmissionError, match="ambient authority differs"):
        admission._load_committed_source_module(
            repo_root=repo,
            source_path=admission.FACTOR_AUTHORITY_SOURCE_PATH,
            ambient_path=source,
            module_name="week7_dirty_factor_authority_test",
        )


def test_generator_identity_disables_replace_refs_and_drops_inherited_replace_env(
    tmp_path: Path, monkeypatch
) -> None:
    repo, renderer_copy, admission_copy, module, source_sha = _committed_generator_copy(
        tmp_path
    )
    monkeypatch.setattr(admission, "REPO_ROOT", repo)
    monkeypatch.setattr(admission, "RENDERER_PATH", renderer_copy)
    monkeypatch.setattr(admission, "SCRIPT_PATH", admission_copy)
    monkeypatch.setattr(admission, "renderer", module)
    monkeypatch.setattr(admission, "RENDERER_EXECUTED_SOURCE_SHA256", source_sha)
    monkeypatch.setattr(
        admission,
        "ADMISSION_EXECUTED_SOURCE_SHA256",
        hashlib.sha256(admission_copy.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv("GIT_NO_REPLACE_OBJECTS", "0")
    monkeypatch.setenv("GIT_REPLACE_REF_BASE", "refs/replace/attacker")
    real_run = subprocess.run
    observed = []

    def recording_run(*args, **kwargs):
        observed.append(dict(kwargs.get("env", {})))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(admission.subprocess, "run", recording_run)
    identity = admission._current_generator_identity()
    assert identity["renderer_source_sha256"] == source_sha
    assert observed
    assert all(env.get("GIT_NO_REPLACE_OBJECTS") == "1" for env in observed)
    assert all("GIT_REPLACE_REF_BASE" not in env for env in observed)
