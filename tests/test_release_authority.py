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
WORKER = RELEASE / "sn56-week6-final-release-cert-worker.sh"
VALIDATOR = RELEASE / "sn56-week6-validate-timing-provenance.py"
DELEGATE = RELEASE / "sn56-week6-build-gpu-cert.py"
FIXTURES = Path(__file__).parent / "fixtures" / "release_authority"
COMMIT = "a" * 40
TREE = "b" * 40
FORGE_TREE = "c" * 40
SCOPE = "toolkit-krea-only"
MODE = "production"
REPOSITORY = "/fixture"
MATERIALIZED_MANIFEST = "9" * 64
DELEGATED_SOURCE_ARCHIVE = "f" * 64
DELEGATED_SOURCE_MANIFEST = "f" * 64


def _authority():
    spec = importlib.util.spec_from_file_location("sn56_release_authority", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pass_receipt(
    authority,
    path: Path,
    *,
    state: str = "PASS",
    schema: int = 2,
    repository: str = REPOSITORY,
):
    sha = "d" * 64
    value = {
        "schema": schema,
        "kind": authority.RECEIPT_KIND,
        "state": state,
        "evidence_class": "operator-attested",
        "claim_limit": (
            "not-independent-proof-of-elapsed-time-or-hardware-measurement"
        ),
        "gate_session_id": "week6-friday-h100-gate",
        "source_run_id": f"fixture:{'e' * 32}",
        "certificate_scope": SCOPE,
        "forge": {
            "repository": repository,
            "commit": COMMIT,
            "tree": TREE,
            "materialized_manifest_sha256": MATERIALIZED_MANIFEST,
        },
        "scope": {
            "bundle_id": "leader-evaluator-aligned-v1",
            "bundle_sha256": sha,
            "model_type": "krea2",
            "current_dataset_size": 18,
            "dataset_regime": "small",
            "accelerator_identity": "NVIDIA H100 PCIe|81559-MiB",
        },
        "rental_window": {
            "started_at_utc": "2026-08-07T00:00:00Z",
            "ended_at_utc": "2026-08-07T01:00:00Z",
        },
        "gate_event": {
            "line": 1,
            "training_started_at_utc": "2026-08-07T00:01:00Z",
            "raw_record_produced_at_utc": "2026-08-07T00:30:00Z",
            "profile_produced_at_utc": "2026-08-07T00:31:00Z",
            "sealed_at_utc": "2026-08-07T00:32:00Z",
        },
        "files": {
            "profile": {
                "path": "/evidence/profile.json",
                "bytes": 1,
                "file_sha256": sha,
                "semantic_sha256": sha,
            },
            "raw_record": {
                "path": "/evidence/raw.json",
                "bytes": 1,
                "file_sha256": sha,
                "semantic_sha256": sha,
            },
            "terminal_artifact": {
                "path": "/evidence/last.safetensors",
                "bytes": 1,
                "file_sha256": sha,
            },
            "archived_terminal_artifact": {
                "path": "/archive/last.safetensors",
                "bytes": 1,
                "file_sha256": sha,
            },
            "gate_log": {
                "path": "/evidence/gate.jsonl",
                "bytes": 1,
                "file_sha256": sha,
            },
        },
    }
    value["receipt_sha256"] = hashlib.sha256(
        authority.canonical_bytes(value)
    ).hexdigest()
    path.write_bytes(authority.canonical_bytes(value) + b"\n")


def _delegated_result(authority) -> dict[str, str]:
    sha = "f" * 64
    return {
        "schema": authority.DELEGATED_RESULT_SCHEMA,
        "state": "PASS",
        "mode": MODE,
        "certificate_scope": SCOPE,
        "source_commit": COMMIT,
        "source_tree": TREE,
        "forge_tree": FORGE_TREE,
        "source_archive_sha256": sha,
        "source_manifest_sha256": sha,
        "production_manifest_sha256": sha,
        "toolkit_dockerfile_sha256": sha,
        "legacy_dockerfile_sha256": sha,
        "toolkit_image_tag": "sn56/toolkit:test",
        "toolkit_image_id": f"sha256:{sha}",
        "legacy_image_tag": "sn56/legacy:test",
        "legacy_image_id": f"sha256:{sha}",
        "gpu_boundary": "REAL_H100",
        "completed_at_utc": "2026-08-07T00:59:00Z",
    }


def _write_result_env(path: Path, value: dict[str, str]) -> None:
    path.write_text(
        "".join(f"{key}={item}\n" for key, item in value.items()),
        encoding="utf-8",
    )


def test_wrapper_is_strict_fixed_authority_with_one_release_commit():
    text = WRAPPER.read_text(encoding="utf-8")
    worker = WORKER.read_text(encoding="utf-8")

    assert "set -eu" in text
    assert "set -Eeuo pipefail" in worker
    assert "SN56_WEEK6_DELEGATED_CERT" not in text + worker
    assert "SN56_RELEASE_FORGE_COMMIT" not in text + worker
    assert "SN56_WEEK6_VALIDATE_ONLY" not in text + worker
    assert "SN56_RELEASE_COMMIT" in text
    assert "--assert-receipt" in worker
    assert "--assert-result-env" in worker
    assert re.search(r"printf[^\n]*\$\(", text + worker) is None
    assert "sn56.week6.final-release-cert-envelope.v3" in worker
    assert "sn56.week6.final-release-cert-envelope.v2" not in worker
    assert "sn56-week5-final-release-cert.sh" not in text + worker


def test_wrapper_pins_exact_validator_and_delegate_hashes():
    text = WORKER.read_text(encoding="utf-8")
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
            expected_repository=REPOSITORY,
            expected_materialized_manifest_sha256=MATERIALIZED_MANIFEST,
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
        ("forge_tree", "e" * 40),
        ("certificate_scope", "wrong-scope"),
        ("mode", "cpu-integration-stub"),
        ("source_archive_sha256", "1" * 64),
        ("source_manifest_sha256", "2" * 64),
    ],
)
def test_delegated_result_env_must_match_release_identity(tmp_path, field, value):
    authority = _authority()
    result = _delegated_result(authority)
    result[field] = value
    path = tmp_path / "result.env"
    _write_result_env(path, result)

    with pytest.raises(authority.ProvenanceError, match=field):
        authority.assert_delegated_result(
            str(path),
            release_commit=COMMIT,
            release_tree=TREE,
            forge_tree=FORGE_TREE,
            certificate_scope=SCOPE,
            mode=MODE,
            expected_source_archive_sha256=DELEGATED_SOURCE_ARCHIVE,
            expected_source_manifest_sha256=DELEGATED_SOURCE_MANIFEST,
        )


def test_delegated_result_env_positive_and_duplicate_negative(tmp_path):
    authority = _authority()
    path = tmp_path / "result.env"
    _write_result_env(path, _delegated_result(authority))
    assert authority.assert_delegated_result(
        str(path),
        release_commit=COMMIT,
        release_tree=TREE,
        forge_tree=FORGE_TREE,
        certificate_scope=SCOPE,
        mode=MODE,
        expected_source_archive_sha256=DELEGATED_SOURCE_ARCHIVE,
        expected_source_manifest_sha256=DELEGATED_SOURCE_MANIFEST,
    )["state"] == "PASS"

    path.write_text(path.read_text() + "state=PASS\n", encoding="utf-8")
    with pytest.raises(authority.ProvenanceError, match="duplicates state"):
        authority.assert_delegated_result(
            str(path),
            release_commit=COMMIT,
            release_tree=TREE,
            forge_tree=FORGE_TREE,
            certificate_scope=SCOPE,
            mode=MODE,
            expected_source_archive_sha256=DELEGATED_SOURCE_ARCHIVE,
            expected_source_manifest_sha256=DELEGATED_SOURCE_MANIFEST,
        )


def test_delegated_result_env_rejects_unpinned_extra_key(tmp_path):
    authority = _authority()
    path = tmp_path / "result.env"
    result = _delegated_result(authority)
    result["self_declared_repository"] = "/attacker/chosen"
    _write_result_env(path, result)

    with pytest.raises(authority.ProvenanceError, match="pinned schema"):
        authority.assert_delegated_result(
            str(path),
            release_commit=COMMIT,
            release_tree=TREE,
            forge_tree=FORGE_TREE,
            certificate_scope=SCOPE,
            mode=MODE,
            expected_source_archive_sha256=DELEGATED_SOURCE_ARCHIVE,
            expected_source_manifest_sha256=DELEGATED_SOURCE_MANIFEST,
        )


def test_delegated_cpu_integration_mode_cannot_claim_production_pass(tmp_path):
    authority = _authority()
    path = tmp_path / "result.env"
    result = _delegated_result(authority)
    result.update(
        {
            "mode": "cpu-integration",
            "state": "DRY_RUN_PASS",
            "gpu_boundary": "STUBBED_NO_CLAIM",
        }
    )
    _write_result_env(path, result)
    assert authority.assert_delegated_result(
        str(path),
        release_commit=COMMIT,
        release_tree=TREE,
        forge_tree=FORGE_TREE,
        certificate_scope=SCOPE,
        mode="cpu-integration",
        expected_source_archive_sha256=DELEGATED_SOURCE_ARCHIVE,
        expected_source_manifest_sha256=DELEGATED_SOURCE_MANIFEST,
    )["state"] == "DRY_RUN_PASS"

    result["state"] = "PASS"
    _write_result_env(path, result)
    with pytest.raises(authority.ProvenanceError, match="state"):
        authority.assert_delegated_result(
            str(path),
            release_commit=COMMIT,
            release_tree=TREE,
            forge_tree=FORGE_TREE,
            certificate_scope=SCOPE,
            mode="cpu-integration",
            expected_source_archive_sha256=DELEGATED_SOURCE_ARCHIVE,
            expected_source_manifest_sha256=DELEGATED_SOURCE_MANIFEST,
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
        expected_repository=REPOSITORY,
        expected_materialized_manifest_sha256=MATERIALIZED_MANIFEST,
        release_commit=COMMIT,
        release_tree=TREE,
        certificate_scope=SCOPE,
    )["state"] == "PASS"

    _pass_receipt(authority, receipt, schema=1)
    with pytest.raises(authority.ProvenanceError, match="authoritative PASS"):
        authority.assert_pass_receipt(
            str(receipt),
            expected_repository=REPOSITORY,
            expected_materialized_manifest_sha256=MATERIALIZED_MANIFEST,
            release_commit=COMMIT,
            release_tree=TREE,
            certificate_scope=SCOPE,
        )


def test_receipt_repository_is_authority_supplied_not_self_derived(tmp_path):
    authority = _authority()
    receipt = tmp_path / "receipt.json"
    _pass_receipt(authority, receipt, repository="/candidate-selected-itself")

    with pytest.raises(authority.ProvenanceError, match="authoritative PASS"):
        authority.assert_pass_receipt(
            str(receipt),
            expected_repository="/authority-materialized-tree",
            expected_materialized_manifest_sha256=MATERIALIZED_MANIFEST,
            release_commit=COMMIT,
            release_tree=TREE,
            certificate_scope=SCOPE,
        )


def test_receipt_rejects_extra_top_level_and_nested_schema_fields(tmp_path):
    authority = _authority()
    receipt = tmp_path / "receipt.json"
    _pass_receipt(authority, receipt)
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["candidate_extension"] = True
    value["receipt_sha256"] = hashlib.sha256(
        authority.canonical_bytes(
            {key: item for key, item in value.items() if key != "receipt_sha256"}
        )
    ).hexdigest()
    receipt.write_bytes(authority.canonical_bytes(value) + b"\n")
    with pytest.raises(authority.ProvenanceError, match="pinned schema"):
        authority.assert_pass_receipt(
            str(receipt),
            expected_repository=REPOSITORY,
            expected_materialized_manifest_sha256=MATERIALIZED_MANIFEST,
            release_commit=COMMIT,
            release_tree=TREE,
            certificate_scope=SCOPE,
        )

    _pass_receipt(authority, receipt)
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["scope"]["candidate_extension"] = True
    value["receipt_sha256"] = hashlib.sha256(
        authority.canonical_bytes(
            {key: item for key, item in value.items() if key != "receipt_sha256"}
        )
    ).hexdigest()
    receipt.write_bytes(authority.canonical_bytes(value) + b"\n")
    with pytest.raises(authority.ProvenanceError, match="scope keys"):
        authority.assert_pass_receipt(
            str(receipt),
            expected_repository=REPOSITORY,
            expected_materialized_manifest_sha256=MATERIALIZED_MANIFEST,
            release_commit=COMMIT,
            release_tree=TREE,
            certificate_scope=SCOPE,
        )


def test_materialized_tree_manifest_binds_file_bytes_paths_and_modes(tmp_path):
    authority = _authority()
    root = tmp_path / "materialized"
    (root / "forge").mkdir(parents=True)
    regular = root / "forge" / "__init__.py"
    executable = root / "runner.py"
    regular.write_bytes(b"VALUE = 1\n")
    executable.write_bytes(b"#!/usr/bin/python3\n")
    regular.chmod(0o644)
    executable.chmod(0o755)
    rows = [
        f"{hashlib.sha256(regular.read_bytes()).hexdigest()} 100644 forge/__init__.py\n",
        f"{hashlib.sha256(executable.read_bytes()).hexdigest()} 100755 runner.py\n",
    ]
    expected = hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()

    assert authority.materialized_tree_manifest_sha256(str(root)) == expected

    executable.write_bytes(b"#!/usr/bin/python3\nprint('changed')\n")
    assert authority.materialized_tree_manifest_sha256(str(root)) != expected


def test_materialized_tree_manifest_uses_delegate_relative_path_order(tmp_path):
    authority = _authority()
    root = tmp_path / "materialized"
    root.mkdir()
    first = root / "a.txt"
    last = root / "z.txt"
    first.write_bytes(b"VALUE = 1\n")
    last.write_bytes(b"#!/usr/bin/python3\n")
    rows = [
        f"{hashlib.sha256(first.read_bytes()).hexdigest()} 100644 a.txt\n",
        f"{hashlib.sha256(last.read_bytes()).hexdigest()} 100644 z.txt\n",
    ]
    assert rows != sorted(rows), "fixture must distinguish path and digest order"
    expected = hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()

    assert authority.materialized_tree_manifest_sha256(str(root)) == expected


def test_load_forge_contract_imports_exact_manifest_bound_archive_tree(tmp_path):
    authority = _authority()
    root = tmp_path / "materialized"
    package = root / "forge"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    adaptive = package / "adaptive_timing.py"
    adaptive.write_text("ARCHIVE_MARKER = 'exact-tree'\n", encoding="utf-8")
    manifest = authority.materialized_tree_manifest_sha256(str(root))
    program = f"""
import importlib.util
spec = importlib.util.spec_from_file_location('authority', {str(VALIDATOR)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
loaded = module.load_forge_contract(
    {str(root)!r},
    {COMMIT!r},
    materialized_manifest_sha256={manifest!r},
)
if loaded.ARCHIVE_MARKER != 'exact-tree':
    raise SystemExit('wrong Forge module imported')
print(loaded.__file__)
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env={"PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(adaptive)
    with pytest.raises(authority.ProvenanceError, match="manifest differs"):
        authority.load_forge_contract(
            str(root),
            COMMIT,
            materialized_manifest_sha256="0" * 64,
        )


def test_prepare_atomic_envelope_rejects_symlinked_base(tmp_path):
    authority = _authority()
    real_base = tmp_path / "real-base"
    real_base.mkdir()
    linked_base = tmp_path / "linked-base"
    linked_base.symlink_to(real_base, target_is_directory=True)

    with pytest.raises(authority.ProvenanceError, match="without following symlinks"):
        authority.prepare_atomic_envelope(str(linked_base), "release-1")

    assert list(real_base.iterdir()) == []


def test_publish_atomic_envelope_refuses_namespace_collision(tmp_path):
    authority = _authority()
    base = tmp_path / "envelopes"
    stage = Path(authority.prepare_atomic_envelope(str(base), "release-1"))
    (stage / "result.env").write_text("state=PASS\n", encoding="utf-8")
    final = base / "release-1"
    sentinel = final / "sentinel"

    def create_racing_final_namespace():
        final.mkdir()
        sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(authority.ProvenanceError, match="already exists"):
        authority.publish_atomic_envelope(
            str(base),
            "release-1",
            str(stage),
            _before_rename=create_racing_final_namespace,
        )

    assert stage.is_dir()
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_publish_atomic_envelope_failure_leaves_no_final_namespace(tmp_path):
    authority = _authority()
    base = tmp_path / "envelopes"
    stage = Path(authority.prepare_atomic_envelope(str(base), "release-1"))
    (stage / "result.env").write_text("state=PASS\n", encoding="utf-8")

    def fail_before_rename():
        raise RuntimeError("injected publication failure")

    with pytest.raises(RuntimeError, match="injected publication failure"):
        authority.publish_atomic_envelope(
            str(base),
            "release-1",
            str(stage),
            _before_rename=fail_before_rename,
        )

    assert stage.is_dir()
    assert not (base / "release-1").exists()


def test_publish_atomic_envelope_success_is_no_replace_and_durable(tmp_path):
    authority = _authority()
    base = tmp_path / "envelopes"
    stage = Path(authority.prepare_atomic_envelope(str(base), "release-1"))
    (stage / "result.env").write_text("state=PASS\n", encoding="utf-8")
    nested = stage / "nested"
    nested.mkdir()
    (nested / "evidence.json").write_text('{"state":"PASS"}\n', encoding="utf-8")

    published = authority.publish_atomic_envelope(
        str(base), "release-1", str(stage)
    )

    final = base / "release-1"
    assert published == str(final)
    assert not stage.exists()
    assert (final / "result.env").read_text(encoding="utf-8") == "state=PASS\n"
    assert (final / "nested" / "evidence.json").is_file()


def test_atomic_envelope_prepare_and_publish_cli_round_trip(tmp_path):
    base = tmp_path / "envelopes"
    prepared = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--prepare-envelope-base",
            str(base),
            "--prepare-envelope-namespace",
            "release-1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr
    prefix = "SN56_ENVELOPE_STAGE="
    assert prepared.stdout.startswith(prefix)
    stage = Path(prepared.stdout.removeprefix(prefix).strip())
    (stage / "result.env").write_text("state=PASS\n", encoding="utf-8")

    published = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--publish-envelope-base",
            str(base),
            "--publish-envelope-namespace",
            "release-1",
            "--publish-envelope-stage",
            str(stage),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    final = base / "release-1"
    assert published.returncode == 0, published.stderr
    assert published.stdout.strip() == f"SN56_ENVELOPE_PUBLISHED={final}"
    assert (final / "result.env").read_text(encoding="utf-8") == "state=PASS\n"


def test_validator_docstring_names_current_gate_event_version():
    text = VALIDATOR.read_text(encoding="utf-8")
    authority = _authority()

    assert authority.EVENT_KIND in text.split('"""', 2)[1]
    assert "sn56.week6.friday-h100-timing-evidence-sealed.v1" not in text.split(
        '"""', 2
    )[1]


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
