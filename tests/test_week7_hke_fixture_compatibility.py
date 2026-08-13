"""Compatibility bridge between immutable fixture and execution authority."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

from ops.experiments.week7 import hke_fixture_compatibility as compatibility

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "week7_hke_factorial_compatibility_fixtures",
    ROOT / "tests" / "test_week7_hke_factorial.py",
)
assert SPEC is not None and SPEC.loader is not None
fixtures = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fixtures
SPEC.loader.exec_module(fixtures)


def test_source_file_constants_are_physical_not_semantic_hashes():
    records = compatibility.SOURCE_RECORDS
    assert records["admission_set"] == {
        "bytes": 47_726,
        "file_sha256": (
            "0a0e60c911f964a0033c9fc655ee6d68dac5dc0ab6d7718505c6cdfe47c271b4"
        ),
        "admission_set_sha256": (
            "44ee883468eabb657e2b6e2a0b20133eee7cd7ae7b70a44de89947e13c94071d"
        ),
    }
    assert records["sealed_human_review"] == {
        "bytes": 178_223,
        "file_sha256": (
            "991a06d6ac3e046bef1f096f9c1c698bed6618785ad54cb34216e56e1a8a2aa2"
        ),
        "review_sha256": (
            "2c0e7754f0a552660f3924242c90e55856922aa749b43cfb5cc790027806b0e2"
        ),
    }
    assert records["owner_ratification"] == {
        "bytes": 1_961,
        "file_sha256": (
            "bfb3265277a56f9a52134f26f137ad220a028d5ad1c912d39b4487370c8e2e6d"
        ),
        "owner_ratification_sha256": (
            "dcc613a1f56366ed9a271d5919a4bbcfd1eeed77054b4e742ec42f3bcd17724e"
        ),
    }


def _records(monkeypatch):
    admission_set = fixtures._admission_set()
    review_body = {
        "schema": 3,
        "kind": "sn56-week7-hke-human-fixture-review",
        "status": "SEALED_OPERATOR_ATTESTED_NAMED_HUMAN_PASS",
        "candidate_semantic_sha256": admission_set["candidate_semantic_sha256"],
        "confirmation_manifest_file_sha256": "6" * 64,
        "decision": "PASS",
        "governance": {
            "agent_review_is_not_human_review": True,
            "confirmation_record_is_private": True,
            "review_must_cover_all_candidate_rows": True,
        },
        "reviewed_at_utc": "2026-08-13T14:16:57Z",
        "reviewer_identity": "Atulya Shetty",
        "rights_record": {"semantic_sha256": admission_set["ownership_record_sha256"]},
        "rows": [{} for _ in range(144)],
    }
    review = {
        **review_body,
        "review_sha256": fixtures.H.fixture_semantic_sha256(review_body),
    }
    admission_set["human_review_sha256"] = review["review_sha256"]
    for family, receipt in admission_set["receipts"].items():
        receipt["human_review_sha256"] = review["review_sha256"]
        fixtures._rehash(receipt, "admission_sha256")
        admission_set["family_admission_sha256"][family] = receipt[
            "admission_sha256"
        ]
    fixtures._rehash(admission_set, "admission_set_sha256")
    ratification = fixtures._ratification(admission_set)
    ratification["reviewer_identity"] = review["reviewer_identity"]
    fixtures._rehash(ratification, "owner_ratification_sha256")
    raw = {
        "admission_set": compatibility.canonical_bytes(admission_set),
        "sealed_human_review": compatibility.canonical_bytes(review),
        "owner_ratification": compatibility.canonical_bytes(ratification),
    }
    source_records = {}
    semantic_fields = {
        "admission_set": "admission_set_sha256",
        "sealed_human_review": "review_sha256",
        "owner_ratification": "owner_ratification_sha256",
    }
    values = {
        "admission_set": admission_set,
        "sealed_human_review": review,
        "owner_ratification": ratification,
    }
    for name, payload in raw.items():
        field = semantic_fields[name]
        source_records[name] = {
            "bytes": len(payload),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            field: values[name][field],
        }
    monkeypatch.setattr(compatibility, "SOURCE_RECORDS", source_records)
    return admission_set, review, ratification, raw


def _target(monkeypatch, admission_set):
    source = admission_set["generator_revision"]
    target = {
        "repository": source["repository"],
        "commit": "d" * 40,
        "tree": "e" * 40,
        "pinned_remote_refs": ["refs/heads/codex/fixture-compatibility"],
        "blob_sha256": {
            source["renderer_source_path"]: source["renderer_source_sha256"],
            source["admission_authority_path"]: source[
                "admission_authority_source_sha256"
            ],
            source["contract_path"]: source["contract_source_sha256"],
            **{
                path: fixtures._sha(path)
                for path in compatibility.EXECUTION_PATHS
            },
        },
    }
    original = compatibility._revision

    def revision(value, paths):
        if value == target["commit"] and tuple(paths) == compatibility.BOUND_PATHS:
            return {
                "commit": target["commit"],
                "tree": target["tree"],
                "blob_sha256": copy.deepcopy(target["blob_sha256"]),
            }
        return original(value, paths)

    monkeypatch.setattr(compatibility, "_revision", revision)
    monkeypatch.setattr(
        compatibility,
        "_remote_refs",
        lambda repository, commit: (
            copy.deepcopy(target["pinned_remote_refs"])
            if repository == target["repository"] and commit == target["commit"]
            else []
        ),
    )
    return target


def _build(monkeypatch):
    admission_set, review, ratification, raw = _records(monkeypatch)
    target = _target(monkeypatch, admission_set)
    receipt = compatibility.build_receipt(
        admission_set=admission_set,
        sealed_review=review,
        owner_ratification=ratification,
        source_raw_files=raw,
        created_at_utc="2026-08-13T18:00:00Z",
        execution_probe=lambda _repository: target,
    )
    return admission_set, ratification, raw, target, receipt


def test_receipt_binds_physical_bytes_and_keeps_gpu_closed(monkeypatch):
    admission_set, ratification, raw, _target_record, receipt = _build(monkeypatch)
    for name, payload in raw.items():
        assert receipt["source_records"][name]["bytes"] == len(payload)
        assert receipt["source_records"][name]["file_sha256"] == hashlib.sha256(
            payload
        ).hexdigest()
    assert receipt["authorization"]["gpu_execution_authorized"] is False
    assert receipt["authorization"]["deployment_authorized"] is False
    assert compatibility.validate_receipt(
        receipt,
        admission_set=admission_set,
        owner_ratification=ratification,
        require_current_execution=False,
    ) == receipt


def test_factor_plan_consumes_receipt_without_rewriting_admission(monkeypatch):
    admission_set, review, ratification, raw = _records(monkeypatch)
    target = _target(monkeypatch, admission_set)
    receipt = compatibility.build_receipt(
        admission_set=admission_set,
        sealed_review=review,
        owner_ratification=ratification,
        source_raw_files=raw,
        created_at_utc="2026-08-13T18:00:00Z",
        execution_probe=lambda _repository: target,
    )
    original_validate = compatibility.validate_receipt

    def validate_without_committed_test_head(value, **kwargs):
        return original_validate(
            value, **kwargs, require_current_execution=False
        )

    monkeypatch.setattr(compatibility, "validate_receipt", validate_without_committed_test_head)
    base = yaml.safe_load(fixtures.H.INCUMBENT_TEMPLATE_PATH.read_text(encoding="utf-8"))
    execution = fixtures._execution()
    for identity in execution.values():
        identity["code_tree"] = target["tree"]
    plan = fixtures.H.build_prelaunch_plan(
        base,
        admission_set=admission_set,
        owner_ratification=ratification,
        fixture_compatibility_receipt=receipt,
        profiles_by_family={"social": {"D1": fixtures._bound_profiles(base, 10)}},
        evaluator_identity=fixtures._evaluator(),
        execution_identities=execution,
    )
    assert plan["admission_set"] == admission_set
    assert plan["admission_set_sha256"] == admission_set["admission_set_sha256"]
    assert plan["fixture_compatibility_receipt"] == receipt
    assert fixtures.H._validate_plan(plan) == plan


def test_one_changed_fixture_authority_blob_requires_readmission(monkeypatch):
    admission_set, review, ratification, raw = _records(monkeypatch)
    target = _target(monkeypatch, admission_set)
    target["blob_sha256"][compatibility.FIXTURE_PATHS[0]] = "f" * 64
    with pytest.raises(compatibility.CompatibilityError, match="re-admission"):
        compatibility.build_receipt(
            admission_set=admission_set,
            sealed_review=review,
            owner_ratification=ratification,
            source_raw_files=raw,
            created_at_utc="2026-08-13T18:00:00Z",
            execution_probe=lambda _repository: target,
        )


def test_rehashed_nonexistent_remote_ref_fails(monkeypatch):
    admission_set, ratification, _raw, _target_record, receipt = _build(monkeypatch)
    forged = copy.deepcopy(receipt)
    forged["execution_authority"]["pinned_remote_refs"] = [
        "refs/heads/does/not/exist"
    ]
    body = dict(forged)
    body.pop("compatibility_receipt_sha256")
    forged["compatibility_receipt_sha256"] = compatibility.semantic_sha256(body)
    with pytest.raises(compatibility.CompatibilityError, match="no longer resolve"):
        compatibility.validate_receipt(
            forged,
            admission_set=admission_set,
            owner_ratification=ratification,
            require_current_execution=False,
        )


def test_impossible_calendar_timestamp_fails(monkeypatch):
    admission_set, review, ratification, raw = _records(monkeypatch)
    target = _target(monkeypatch, admission_set)
    with pytest.raises(compatibility.CompatibilityError, match="timestamp"):
        compatibility.build_receipt(
            admission_set=admission_set,
            sealed_review=review,
            owner_ratification=ratification,
            source_raw_files=raw,
            created_at_utc="2026-19-39T29:59:59Z",
            execution_probe=lambda _repository: target,
        )


def test_file_builder_hashes_physical_not_semantic_bytes(tmp_path, monkeypatch):
    admission_set, review, ratification, raw = _records(monkeypatch)
    target = _target(monkeypatch, admission_set)
    paths = []
    for name, payload in raw.items():
        path = tmp_path / f"{name}.json"
        path.write_bytes(payload)
        path.chmod(0o600)
        paths.append(path)
    output = tmp_path / "compatibility.json"
    receipt = compatibility.create_from_files(
        paths[0],
        paths[1],
        paths[2],
        output,
        "2026-08-13T18:00:00Z",
        execution_probe=lambda _repository: target,
    )
    assert json.loads(output.read_text(encoding="ascii")) == receipt
    file_sha = receipt["source_records"]["admission_set"]["file_sha256"]
    assert file_sha == hashlib.sha256(raw["admission_set"]).hexdigest()
    assert file_sha != admission_set["admission_set_sha256"]


@pytest.mark.parametrize(
    "record", ["admission_set", "sealed_human_review", "owner_ratification"]
)
def test_rehashed_source_record_transplant_fails(record, monkeypatch):
    admission_set, ratification, _raw, _target_record, receipt = _build(monkeypatch)
    forged = copy.deepcopy(receipt)
    forged["source_records"][record]["file_sha256"] = "f" * 64
    body = dict(forged)
    body.pop("compatibility_receipt_sha256")
    forged["compatibility_receipt_sha256"] = compatibility.semantic_sha256(body)
    with pytest.raises(compatibility.CompatibilityError, match="source-file records"):
        compatibility.validate_receipt(
            forged,
            admission_set=admission_set,
            owner_ratification=ratification,
            require_current_execution=False,
        )
