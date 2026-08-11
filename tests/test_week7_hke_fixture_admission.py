"""Human authority boundary for the Week-7 procedural fixtures."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
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
    public = root / "public"
    custodian = root / "custodian"
    renderer.build_candidate(
        public_output=public,
        custodian_output=custodian,
        discovery_key=DISCOVERY_KEY,
        confirmation_key=CONFIRMATION_KEY,
        generator_commit=GENERATOR_COMMIT,
        generator_tree=GENERATOR_TREE,
        **RIGHTS,
    )
    return public, custodian


def completed_review(candidate):
    public, custodian = candidate
    draft = admission.build_review_template(public, custodian)
    draft["reviewer_identity"] = "Atulya Shetty"
    draft["reviewed_at_utc"] = "2026-08-10T23:00:00Z"
    draft["decision"] = "PASS"
    for row in draft["rows"]:
        for key in row["checks"]:
            row["checks"][key] = True
    return draft


def generator_probe():
    return {
        "repository": renderer.GENERATOR_REPOSITORY,
        "commit": GENERATOR_COMMIT,
        "tree": GENERATOR_TREE,
        "source_path": renderer.GENERATOR_SOURCE_PATH,
        "renderer_source_sha256": renderer._source_sha256(),
        "admission_authority_source_sha256": admission.hashlib.sha256(
            admission.SCRIPT_PATH.read_bytes()
        ).hexdigest(),
        "pinned_remote_refs": ["refs/heads/test-pushed-branch"],
    }


def test_template_is_private_pending_and_never_fakes_human_review(candidate):
    public, custodian = candidate
    template = admission.build_review_template(public, custodian)
    assert template["status"] == "PENDING_NAMED_HUMAN_REVIEW"
    assert template["reviewer_identity"] == ""
    assert template["decision"] == "PENDING"
    assert template["rights_record"]["rights_owner"] == RIGHTS["rights_owner"]
    assert len(template["rows"]) == 98
    assert {row["phase"] for row in template["rows"]} == {
        "discovery",
        "confirmation",
    }
    assert all(not any(row["checks"].values()) for row in template["rows"])
    assert template["governance"]["agent_review_is_not_human_review"] is True


def test_role_label_and_one_missing_row_check_abort(candidate):
    public, custodian = candidate
    role = completed_review(candidate)
    role["reviewer_identity"] = "human reviewer"
    with pytest.raises(admission.AdmissionError, match="role label"):
        admission.seal_review(
            public_root=public, custodian_root=custodian, draft=role
        )

    missing = completed_review(candidate)
    missing["rows"][17]["checks"]["caption_semantics_accepted"] = False
    with pytest.raises(admission.AdmissionError, match="unapproved check"):
        admission.seal_review(
            public_root=public, custodian_root=custodian, draft=missing
        )


def test_exact_row_tampering_aborts_even_if_every_boolean_says_pass(candidate):
    public, custodian = candidate
    forged = completed_review(candidate)
    forged["rows"][73]["image_sha256"] = "0" * 64
    with pytest.raises(admission.AdmissionError, match="identity mismatch"):
        admission.seal_review(
            public_root=public, custodian_root=custodian, draft=forged
        )


def test_pass_emits_three_admissions_but_keeps_gpu_and_deploy_closed(candidate):
    public, custodian = candidate
    sealed = admission.seal_review(
        public_root=public,
        custodian_root=custodian,
        draft=completed_review(candidate),
    )
    root, receipts = admission.build_admissions(
        public_root=public,
        custodian_root=custodian,
        discovery_key=DISCOVERY_KEY,
        confirmation_key=CONFIRMATION_KEY,
        sealed_review=sealed,
        generator_identity_probe=generator_probe,
    )
    assert set(receipts) == {"social", "product", "logo_ui"}
    assert receipts["social"]["counts"] == {"discovery": 10, "confirmation": 8}
    assert receipts["product"]["counts"] == {"discovery": 28, "confirmation": 10}
    assert receipts["logo_ui"]["counts"] == {"discovery": 32, "confirmation": 10}
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
    confirmation = renderer.verify_candidate(public, custodian)[
        "confirmation_manifest"
    ]
    private_row_id = next(
        iter(next(iter(confirmation["families"].values()))["rows"])
    )["row_id"].encode("ascii")
    assert private_row_id not in public_bytes


def test_private_review_records_cannot_be_written_in_candidate_or_repo(candidate):
    public, custodian = candidate
    with pytest.raises(admission.AdmissionError, match="must be outside"):
        admission._private_record_path(
            public / "review.json",
            public_root=public,
            custodian_root=custodian,
            label="review",
        )
    with pytest.raises(admission.AdmissionError, match="must be outside"):
        admission._private_record_path(
            custodian / "review.json",
            public_root=public,
            custodian_root=custodian,
            label="review",
        )
    with pytest.raises(admission.AdmissionError, match="must be outside"):
        admission._private_record_path(
            admission.REPO_ROOT / "review.json",
            public_root=public,
            custodian_root=custodian,
            label="review",
        )


def test_sealed_review_digest_tampering_aborts(candidate):
    public, custodian = candidate
    sealed = admission.seal_review(
        public_root=public,
        custodian_root=custodian,
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
            sealed_review=tampered,
            generator_identity_probe=generator_probe,
        )


def test_admission_rejects_attested_tree_that_is_not_the_executed_revision(candidate):
    public, custodian = candidate
    sealed = admission.seal_review(
        public_root=public,
        custodian_root=custodian,
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
            sealed_review=sealed,
            generator_identity_probe=wrong_tree,
        )
