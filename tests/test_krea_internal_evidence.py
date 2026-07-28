"""Adversarial tests for the portable K5 internal-evidence anchor."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest


_CALIBRATION = Path(__file__).parents[1] / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))

import krea_execution_plan  # noqa: E402
import krea_internal_evidence  # noqa: E402
import krea_provenance  # noqa: E402


PATHS = (
    "SN56-GATE-B-H100-RESULTS.md",
    "SN56-WEEK4-GPU-CAMPAIGN-RESULTS-2026-07-23.md",
    "week4-gpu-evidence-2026-07-22/EVIDENCE-ERRATA.md",
)
CONTENTS = {
    PATHS[0]: b"representative Gate-B exact-evaluator evidence\n",
    PATHS[1]: b"representative Week-4 campaign result evidence\n",
    PATHS[2]: b"append-only correction: same fixture and same seed\n",
}


def _canonical(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(krea_provenance.canonical_bytes(value) + b"\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project_copy(tmp_path: Path) -> Path:
    root = tmp_path / "SN56-project"
    for relative in PATHS:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(CONTENTS[relative])
    record = krea_internal_evidence.build_record(project_root=root)
    _canonical(root / "K5-INTERNAL-EVIDENCE-RECORD.json", record)
    return root


def test_canonical_record_reopens_exact_three_relative_files(tmp_path: Path) -> None:
    root = _project_copy(tmp_path)
    path = root / "K5-INTERNAL-EVIDENCE-RECORD.json"
    record, file_sha, size = krea_internal_evidence.load_record(path, project_root=root)
    assert record["schema"] == 2
    assert [row["path"] for row in record["evidence_files"]] == list(PATHS)
    assert [row["bytes"] for row in record["evidence_files"]] == [
        len(CONTENTS[relative]) for relative in PATHS
    ]
    assert file_sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert size == path.stat().st_size
    assert "SN56-WEEK4-GPU-VALIDATION-ERRATUM" not in path.read_text()


def test_record_rejects_traversal_stale_path_and_rehashed_byte_drift(
    tmp_path: Path,
) -> None:
    root = _project_copy(tmp_path)
    record = json.loads((root / "K5-INTERNAL-EVIDENCE-RECORD.json").read_text())

    for invalid in (
        "../SN56-GATE-B-H100-RESULTS.md",
        "SN56-WEEK4-GPU-VALIDATION-ERRATUM-2026-07-22.md",
    ):
        forged = deepcopy(record)
        forged["evidence_files"][0]["path"] = invalid
        body = {key: value for key, value in forged.items() if key != "record_sha256"}
        forged["record_sha256"] = krea_provenance.canonical_sha256(body)
        with pytest.raises(ValueError, match="relative|frozen canonical"):
            krea_internal_evidence.validate_record(forged, project_root=root)

    changed = root / PATHS[0]
    changed.write_bytes(changed.read_bytes() + b"\nattacker rewrite\n")
    with pytest.raises(ValueError, match="bound project bytes"):
        krea_internal_evidence.validate_record(record, project_root=root)


def test_record_rejects_symlinked_evidence_and_implicit_cwd(tmp_path: Path) -> None:
    root = _project_copy(tmp_path)
    target = root / PATHS[2]
    external = tmp_path / "external-errata.md"
    external.write_bytes(CONTENTS[PATHS[2]])
    target.unlink()
    target.symlink_to(external)
    with pytest.raises(ValueError, match="unsafe|symlink"):
        krea_internal_evidence.build_record(project_root=root)
    with pytest.raises(ValueError, match="explicit absolute"):
        krea_internal_evidence.build_record(project_root=Path("SN56-project"))


def test_k5_arm_basis_reopens_record_and_emits_plan_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_copy(tmp_path)
    record_path = root / "K5-INTERNAL-EVIDENCE-RECORD.json"
    record_file_sha = hashlib.sha256(record_path.read_bytes()).hexdigest()
    basis = krea_execution_plan.build_internal_basis(
        arm_id="K5",
        mode="internal_evidence_challenger",
        description="Prior internally measured LR challenger",
        evidence_record={"path": str(record_path), "sha256": record_file_sha},
        release_commit="c654c4b24376f7aa9e12dcb82f5e73dcddee3bdb",
        parent_arm_id=None,
    )
    basis_path = root / "K5-basis.json"
    basis_file_sha = _canonical(basis_path, basis)
    monkeypatch.setattr(
        krea_execution_plan.krea_provenance,
        "normalize_recipe",
        lambda recipe: recipe,
    )
    normalized = krea_execution_plan._arm_basis(
        {
            "mode": "internal",
            "basis_record": {"path": str(basis_path), "sha256": basis_file_sha},
            "project_root": str(root),
        },
        arm_id="K5",
        execution_recipe={"fields": {}},
    )
    assert normalized["K5_internal_evidence_anchor"] == (
        krea_internal_evidence.build_anchor(record_path=record_path, project_root=root)
    )

    with pytest.raises(ValueError, match="project root|missing or unsafe"):
        krea_execution_plan._arm_basis(
            {
                "mode": "internal",
                "basis_record": {
                    "path": str(basis_path),
                    "sha256": basis_file_sha,
                },
                "project_root": str(tmp_path),
            },
            arm_id="K5",
            execution_recipe={"fields": {}},
        )
