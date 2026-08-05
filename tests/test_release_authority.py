from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "ops" / "release"
WRAPPER = RELEASE / "sn56-week6-final-release-cert.sh"
VALIDATOR = RELEASE / "sn56-week6-validate-timing-provenance.py"
DELEGATE = RELEASE / "sn56-week5-final-release-cert.sh"
FIXTURES = Path(__file__).parent / "fixtures" / "release_authority"
COMMIT = "a" * 40
TREE = "b" * 40
SCOPE = "toolkit-krea-only"


def _authority():
    spec = importlib.util.spec_from_file_location("sn56_release_authority", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pass_receipt(authority, path: Path, *, state: str = "PASS", schema: int = 2):
    value = {
        "schema": schema,
        "kind": authority.RECEIPT_KIND,
        "state": state,
        "evidence_class": "operator-attested",
        "claim_limit": (
            "not-independent-proof-of-elapsed-time-or-hardware-measurement"
        ),
        "certificate_scope": SCOPE,
        "forge": {"repository": "/fixture", "commit": COMMIT, "tree": TREE},
        "files": {},
    }
    value["receipt_sha256"] = hashlib.sha256(
        authority.canonical_bytes(value)
    ).hexdigest()
    path.write_bytes(authority.canonical_bytes(value) + b"\n")


def test_wrapper_is_strict_fixed_authority_with_one_release_commit():
    text = WRAPPER.read_text(encoding="utf-8")

    assert "set -Eeuo pipefail" in text
    assert "SN56_WEEK6_DELEGATED_CERT" not in text
    assert "SN56_RELEASE_FORGE_COMMIT" not in text
    assert "SN56_WEEK6_VALIDATE_ONLY" not in text
    assert "SN56_RELEASE_COMMIT" in text
    assert "--assert-receipt" in text
    assert "--assert-result-env" in text
    assert re.search(r"printf[^\n]*\$\(", text) is None
    assert "sn56.week6.final-release-cert-envelope.v2" in text
    assert "sn56.week6.final-release-cert-envelope.v1" not in text


def test_wrapper_pins_exact_validator_and_delegate_hashes():
    text = WRAPPER.read_text(encoding="utf-8")
    validator = re.search(r"^readonly VALIDATOR_SHA256=([0-9a-f]{64})$", text, re.M)
    delegate = re.search(r"^readonly DELEGATED_SHA256=([0-9a-f]{64})$", text, re.M)

    assert validator is not None and validator.group(1) == _sha(VALIDATOR)
    assert delegate is not None and delegate.group(1) == _sha(DELEGATE)


def test_zero_byte_validator_is_rejected_before_execution(tmp_path):
    authority = _authority()
    source = FIXTURES / "zero-byte-validator.py"

    with pytest.raises(authority.ProvenanceError, match="nonempty regular"):
        authority.stage_validated_file(
            str(source),
            str(tmp_path / "validator.py"),
            label="release validator",
            expected_sha256=_sha(source),
        )


def test_success_exit_without_receipt_is_failure(tmp_path):
    authority = _authority()
    fixture = FIXTURES / "missing-receipt-validator.py"

    with pytest.raises(authority.ProvenanceError, match="receipt.*opened"):
        authority.run_pinned_validator(
            str(fixture),
            _sha(fixture),
            [],
            receipt_path=str(tmp_path / "absent-receipt.json"),
            release_commit=COMMIT,
            release_tree=TREE,
            certificate_scope=SCOPE,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state", "FAIL"),
        ("source_commit", "c" * 40),
        ("source_tree", "d" * 40),
        ("certificate_scope", "wrong-scope"),
    ],
)
def test_delegated_result_env_must_match_release_identity(tmp_path, field, value):
    authority = _authority()
    result = {
        "schema": "sn56.week5.final-release-cert.v2",
        "state": "PASS",
        "source_commit": COMMIT,
        "source_tree": TREE,
        "certificate_scope": SCOPE,
    }
    result[field] = value
    path = tmp_path / "result.env"
    path.write_text(
        "".join(f"{key}={item}\n" for key, item in result.items()),
        encoding="utf-8",
    )

    with pytest.raises(authority.ProvenanceError, match=field):
        authority.assert_delegated_result(
            str(path),
            release_commit=COMMIT,
            release_tree=TREE,
            certificate_scope=SCOPE,
        )


def test_delegated_result_env_positive_and_duplicate_negative(tmp_path):
    authority = _authority()
    path = tmp_path / "result.env"
    path.write_text(
        "schema=sn56.week5.final-release-cert.v2\n"
        "state=PASS\n"
        f"source_commit={COMMIT}\n"
        f"source_tree={TREE}\n"
        f"certificate_scope={SCOPE}\n",
        encoding="utf-8",
    )
    assert authority.assert_delegated_result(
        str(path),
        release_commit=COMMIT,
        release_tree=TREE,
        certificate_scope=SCOPE,
    )["state"] == "PASS"

    path.write_text(path.read_text() + "state=PASS\n", encoding="utf-8")
    with pytest.raises(authority.ProvenanceError, match="duplicates state"):
        authority.assert_delegated_result(
            str(path),
            release_commit=COMMIT,
            release_tree=TREE,
            certificate_scope=SCOPE,
        )


def test_validated_descriptor_never_forwards_replacement_b(tmp_path):
    authority = _authority()
    source = tmp_path / "source"
    replacement = tmp_path / "replacement"
    staged = tmp_path / "staged"
    source.write_bytes(b"A" * 4096)
    replacement.write_bytes(b"B" * 4096)
    a_sha = _sha(source)

    def swap_source_path():
        os.replace(replacement, source)

    try:
        authority.stage_validated_file(
            str(source),
            str(staged),
            label="A/B fixture",
            expected_sha256=a_sha,
            _after_open=swap_source_path,
        )
    except authority.ProvenanceError as exc:
        # Some filesystems update the unlinked A inode's ctime and trigger the
        # conservative identity gate. Others leave the opened inode stable; in
        # that case descriptor copying must forward A, never replacement B.
        assert "changed while" in str(exc)
        assert not staged.exists()
    else:
        assert staged.read_bytes() == b"A" * 4096
        assert source.read_bytes() == b"B" * 4096

    source.write_bytes(b"A" * 4096)
    authority.stage_validated_file(
        str(source),
        str(staged),
        label="A/B fixture",
        expected_sha256=a_sha,
    )

    staged_replacement = tmp_path / "staged-replacement"
    staged_replacement.write_bytes(b"B" * 4096)
    os.replace(staged_replacement, staged)
    with pytest.raises(authority.ProvenanceError, match="hash mismatch"):
        authority.assert_regular_file_hash(
            str(staged), label="forwarded A/B fixture", expected_sha256=a_sha
        )


def test_receipt_requires_current_schema_kind_state_and_identity(tmp_path):
    authority = _authority()
    receipt = tmp_path / "receipt.json"
    _pass_receipt(authority, receipt)
    assert authority.assert_pass_receipt(
        str(receipt),
        release_commit=COMMIT,
        release_tree=TREE,
        certificate_scope=SCOPE,
    )["state"] == "PASS"

    _pass_receipt(authority, receipt, schema=1)
    with pytest.raises(authority.ProvenanceError, match="authoritative PASS"):
        authority.assert_pass_receipt(
            str(receipt),
            release_commit=COMMIT,
            release_tree=TREE,
            certificate_scope=SCOPE,
        )


def test_reviewed_release_constant_is_human_readable_and_single_sourced():
    from forge import recipe

    assert recipe.KREA_RELEASE_TIMING_POLICY == {
        "schema": 1,
        "kind": "forge-reviewed-conservative-timing-constant",
        "model_type": "krea2",
        "seconds_per_step": 2.2,
        "basis": "week5-validator-field-depth-owner-reviewed-2026-08-03",
        "evidence_boundary": (
            "host-bound-lab-profile-never-consumed-by-production"
        ),
    }
    assert recipe.SEC_PER_IT["krea2"] == 2.2


def test_versioned_validator_self_test_covers_current_contract():
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--self-test",
            "--forge-repository",
            str(ROOT),
            "--forge-commit",
            commit,
        ],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "SN56_WEEK6_TIMING_PROVENANCE_SELF_TEST=PASS" in completed.stdout
