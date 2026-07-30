#!/usr/bin/env python3
"""Build exact K0..K5 arm fragments from the accepted Week-5 artifacts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


RELEASE = "c654c4b24376f7aa9e12dcb82f5e73dcddee3bdb"
HOST_CONTROL = Path("/campaign/controls")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def semantic_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def internal_basis(
    *,
    arm_id: str,
    mode: str,
    description: str,
    evidence_path: str,
    evidence_sha: str,
    parent_arm_id: str | None,
) -> dict[str, Any]:
    body = {
        "schema": 1,
        "kind": "forge-krea-internal-arm-basis",
        "arm_id": arm_id,
        "mode": mode,
        "description": description,
        "evidence_record": {"path": evidence_path, "sha256": evidence_sha},
        "release_commit": RELEASE,
        "parent_arm_id": parent_arm_id,
    }
    return {**body, "basis_sha256": semantic_sha(body)}


def public_recipe(source: dict[str, Any], *, arm_id: str) -> dict[str, Any]:
    overrides = {"planned_steps": 100, "save_cadence": 12}
    if arm_id == "K3":
        overrides.update(
            {"dropout": 0.05, "ema": {"enabled": False, "decay": 0.99}}
        )
    if arm_id == "K4":
        overrides["learning_rate"] = 0.00000086
    fields = {}
    for name, source_row in source["normalized_recipe"]["fields"].items():
        state = source_row["classification"]
        if name in {"selector", "submitted_step"}:
            row = {
                "classification": "unsupported",
                "source_pointers": source_row["source_pointers"],
                "source_value": source_row["source_value"],
                "effective_value": None,
                "evidence": "checkpoint choice is downstream",
            }
        elif state == "unknown":
            if name not in overrides:
                raise ValueError(f"unresolved {arm_id} source field: {name}")
            row = {
                "classification": "unknown_source_fixed",
                "source_pointers": [],
                "source_value": None,
                "effective_value": overrides[name],
                "evidence": (
                    f"frozen {arm_id} local control in the Week-5 discovery plan; "
                    "source field remains unknown"
                ),
            }
        else:
            effective = overrides.get(name, source_row["source_value"])
            row = {
                "classification": (
                    "known" if effective == source_row["source_value"] else "adapted"
                ),
                "source_pointers": source_row["source_pointers"],
                "source_value": source_row["source_value"],
                "effective_value": effective,
                "evidence": (
                    source_row["evidence"]
                    if effective == source_row["source_value"]
                    else "frozen Week-5 adaptation; immutable source value preserved"
                ),
            }
        fields[name] = row
    return {
        "schema": 1,
        "kind": "forge-krea-normalized-recipe",
        "fields": fields,
    }


def public_basis(root: Path, arm_id: str, source: dict[str, Any]) -> dict[str, Any]:
    revisions = {
        "K2": "revision-manifests/K2-f4766189-revision.raw.json",
        "K3": "revision-manifests/K3-919e07cd-revision.raw.json",
        "K4": "revision-manifests/K4-71bf349e-revision.raw.json",
    }
    configs = {
        "K2": "raw-configs/K2-rank2-f4766189-config.yaml",
        "K3": "raw-configs/K3-rank3-919e07cd-config.yaml",
        "K4": "raw-configs/K4-rank5-71bf349e-config.yaml",
    }
    provenance = f"public-source-provenance/{arm_id}-public-source-provenance.json"
    approval = f"{arm_id}-source-normalization-review.json"
    task = "official-snapshots/krea-r1-task-latest.raw.json"
    tournament = "official-snapshots/tournament-latest.raw.json"
    revision = revisions[arm_id]
    config = configs[arm_id]
    host_root = HOST_CONTROL / "public-source-review-v4"
    artifact_sha = source["files"]["source_artifact"]["sha256"]
    field_ledger = Path(__file__).resolve().parents[1] / (
        "ops/calibration/week5/krea-r1-field-ledger.json"
    )
    return {
        "mode": "public_submission",
        "source_provenance": {
            "path": str(host_root / provenance),
            "sha256": file_sha(root / provenance),
        },
        "source_normalization_approval": {
            "path": str(host_root / approval),
            "sha256": file_sha(root / approval),
        },
        "source_files": {
            "source_config": {
                "path": str(host_root / config),
                "sha256": file_sha(root / config),
            },
            "source_artifact": {
                "path": str(host_root / "cas" / artifact_sha),
                "sha256": artifact_sha,
            },
            "field_ledger": {
                "path": "/app/forge/ops/calibration/week5/krea-r1-field-ledger.json",
                "sha256": file_sha(field_ledger),
            },
            "task_raw": {
                "path": str(host_root / task),
                "sha256": file_sha(root / task),
            },
            "tournament_raw": {
                "path": str(host_root / tournament),
                "sha256": file_sha(root / tournament),
            },
            "revision_manifest": {
                "path": str(host_root / revision),
                "sha256": file_sha(root / revision),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve(strict=True)
    output = args.output_dir.resolve()
    timing = project / "week5-krea-curation-20260729/timing-controls-staging"
    public = project / "week5-krea-curation-20260729/public-source-review-v4"
    k1_probe = json.loads((timing / "D1-K1-A-timing-probe-payload.json").read_text())
    k3_probe = json.loads(
        (timing / "D1-B-K3-A-timing-probe-payload.candidate.json").read_text()
    )
    k4_probe = json.loads(
        (timing / "D1-C-K4-A-timing-probe-payload.candidate.json").read_text()
    )

    k0_evidence = {
        "schema": 1,
        "kind": "forge-krea-internal-arm-basis-evidence",
        "arm_id": "K0",
        "discovery_plan_file_sha256": k1_probe["discovery_plan"]["sha256"],
        "release_commit": RELEASE,
        "statement": "Deployed c654c4b recipe and static-depth release control.",
    }
    k0_evidence_path = output / "K0-basis-evidence.json"
    publish(k0_evidence_path, k0_evidence)
    k0_basis = internal_basis(
        arm_id="K0",
        mode="deployed_control",
        description="Deployed c654c4b shallow recipe control.",
        evidence_path=str(HOST_CONTROL / k0_evidence_path.name),
        evidence_sha=file_sha(k0_evidence_path),
        parent_arm_id=None,
    )
    k0_basis_path = output / "K0-internal-basis.json"
    publish(k0_basis_path, k0_basis)

    for name in ("D1-K1-basis-evidence.json", "D1-K1-internal-basis.json"):
        shutil.copyfile(timing / name, output / name)

    k5_project = output / "K5-project"
    k5_files = [
        "K5-INTERNAL-EVIDENCE-RECORD.json",
        "SN56-GATE-B-H100-RESULTS.md",
        "SN56-WEEK4-GPU-CAMPAIGN-RESULTS-2026-07-23.md",
        "week4-gpu-evidence-2026-07-22/EVIDENCE-ERRATA.md",
    ]
    for relative in k5_files:
        target = k5_project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / relative, target)
    k5_record = k5_project / "K5-INTERNAL-EVIDENCE-RECORD.json"
    k5_basis = internal_basis(
        arm_id="K5",
        mode="internal_evidence_challenger",
        description="Prior internally measured 2e-4 LR challenger.",
        evidence_path=str(
            HOST_CONTROL / "K5-project/K5-INTERNAL-EVIDENCE-RECORD.json"
        ),
        evidence_sha=file_sha(k5_record),
        parent_arm_id=None,
    )
    k5_basis_path = output / "K5-internal-basis.json"
    publish(k5_basis_path, k5_basis)

    k0_recipe = copy.deepcopy(k1_probe["execution_recipe"])
    k5_recipe = copy.deepcopy(k1_probe["execution_recipe"])
    for row in k0_recipe["fields"].values():
        row["evidence"] = row["evidence"].replace("K1", "K0")
    for row in k5_recipe["fields"].values():
        row["evidence"] = row["evidence"].replace("K1", "K5")
    k5_recipe["fields"]["learning_rate"]["effective_value"] = 0.0002

    sources = {
        arm: json.loads(
            (public / f"public-source-provenance/{arm}-public-source-provenance.json").read_text()
        )
        for arm in ("K2", "K3", "K4")
    }
    arms = {
        "K0": {
            "arm_basis": {
                "mode": "internal",
                "basis_record": {
                    "path": str(HOST_CONTROL / k0_basis_path.name),
                    "sha256": file_sha(k0_basis_path),
                },
            },
            "execution_recipe": k0_recipe,
        },
        "K1": {
            "arm_basis": k1_probe["arm_basis"],
            "execution_recipe": k1_probe["execution_recipe"],
        },
        "K2": {
            "arm_basis": public_basis(public, "K2", sources["K2"]),
            "execution_recipe": public_recipe(sources["K2"], arm_id="K2"),
        },
        "K3": {
            "arm_basis": k3_probe["arm_basis"],
            "execution_recipe": k3_probe["execution_recipe"],
        },
        "K4": {
            "arm_basis": k4_probe["arm_basis"],
            "execution_recipe": k4_probe["execution_recipe"],
        },
        "K5": {
            "arm_basis": {
                "mode": "internal",
                "basis_record": {
                    "path": str(HOST_CONTROL / k5_basis_path.name),
                    "sha256": file_sha(k5_basis_path),
                },
                "project_root": str(HOST_CONTROL / "K5-project"),
            },
            "execution_recipe": k5_recipe,
        },
    }
    body = {
        "schema": 1,
        "kind": "forge-krea-fc70-arm-inputs",
        "source": "accepted-week5-artifacts-only",
        "arms": arms,
        "staged_files": [
            {
                "relative_path": path.relative_to(output).as_posix(),
                "sha256": file_sha(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(output.rglob("*"))
            if path.is_file()
        ],
    }
    manifest = {**body, "manifest_sha256": semantic_sha(body)}
    publish(output / "fc70-arm-inputs.json", manifest)
    print(json.dumps({"output": str(output), "manifest_sha256": manifest["manifest_sha256"]}))


if __name__ == "__main__":
    main()
