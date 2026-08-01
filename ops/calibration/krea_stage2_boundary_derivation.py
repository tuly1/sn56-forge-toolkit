"""Create mechanics-only Stage-2 boundary fixtures from admitted D1/D2 bytes.

This recovery is intentionally narrow.  It does not curate new content, alter
the admitted datasets, or reuse the D1/D2 governance as boundary authority.
It copies the exact admitted bytes, embeds the validated source manifest and
approval, and leaves admission/GPU authority to the fresh Stage-2 chain.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Mapping

try:
    from . import krea_fixture
    from . import krea_provenance
except ImportError:  # pragma: no cover - direct execution.
    import krea_fixture  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]


SCHEMA = 1
KIND = "forge-krea-stage2-boundary-derivation-set"
FREEZE_BINDING_PATH = (
    "ops/calibration/week5/"
    "krea-density-seedb-finalist-freeze-public-binding-2026-08-01.json"
)
FREEZE_BINDING_FILE_SHA256 = (
    "b0fe9af433e0bc76aaf6cace5356efa5824eee0feb5f629c74527fa05ffd3c2a"
)
FREEZE_BINDING_SHA256 = (
    "b6fffaa8d00f94831cd1fef37e3babbd45b0168f2a43df05dc69f323a7a6e561"
)
FREEZE_BINDING_COMMIT = "f8d71ac1d0fcbab9dccf7f5a5a5f904f9f90b237"
_ROLE_SOURCE = dict(krea_fixture._STAGE2_BOUNDARY_SOURCE_ROLES)
_DERIVATION_ACTOR = {
    "actor_class": "agent",
    "actor_id": "codex-week5-stage2-boundary-derivation-implementer",
    "display_name": "Codex Week-5 Stage-2 boundary derivation implementer (agent)",
    "role": "boundary_derivation_implementer",
    "review_instance_id": "week5-krea-stage2-boundary-derivation-20260801-v1",
    "identity_assurance": (
        "self-declared-agent-identity-not-human-or-cryptographic-authentication"
    ),
}


def _utc(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("created_at_utc must be canonical whole-second UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError("created_at_utc must be canonical whole-second UTC") from exc
    if parsed < datetime(2020, 1, 1, tzinfo=timezone.utc) or parsed > datetime.now(
        timezone.utc
    ) + timedelta(seconds=60):
        raise ValueError("created_at_utc is outside accepted bounds")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(path: Path, label: str) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if (
        not isinstance(value, dict)
        or raw != krea_provenance.canonical_bytes(value) + b"\n"
    ):
        raise ValueError(f"{label} is not canonical create-only JSON")
    return value, hashlib.sha256(raw).hexdigest()


def validate_public_freeze_binding(
    binding: Mapping[str, Any], *, file_sha256: str
) -> dict[str, Any]:
    value = dict(binding)
    body = {key: item for key, item in value.items() if key != "binding_sha256"}
    chronology = value.get("chronology_contract")
    if (
        file_sha256 != FREEZE_BINDING_FILE_SHA256
        or value.get("schema") != 1
        or value.get("kind")
        != "forge-krea-density-seedb-finalist-freeze-public-binding"
        or value.get("binding_sha256") != FREEZE_BINDING_SHA256
        or value.get("binding_sha256") != krea_provenance.canonical_sha256(body)
        or not isinstance(chronology, dict)
        or chronology.get("full_freeze_exists_before_binding") is not True
        or chronology.get("reveal_requires_pushed_binding_commit") is not True
        or chronology.get("c1c4_content_read_at_binding") is not False
    ):
        raise ValueError("public finalist-freeze binding is not the pushed gate")
    return value


def _safe_tree(root: Path, label: str) -> Path:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"{label} must be a directory")
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            raise ValueError(f"{label} contains a symlink")
        mode = path.stat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f"{label} contains a special node")
    return root


def _copy_new(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"copy source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    if destination.stat().st_size != source.stat().st_size or _sha(destination) != _sha(
        source
    ):
        raise RuntimeError(f"copied bytes changed: {source}")


def _dataset_files(
    manifest: Mapping[str, Any], split: str
) -> dict[str, tuple[str, int]]:
    rows = manifest[f"{split}_dataset_identity"]["rows"]
    expected: dict[str, tuple[str, int]] = {}
    for row in rows:
        expected[row["image"]] = (row["image_sha256"], row["image_bytes"])
        expected[row["prompt"]] = (row["prompt_sha256"], row["prompt_bytes"])
    return expected


def _validate_package(root: Path, manifest: Mapping[str, Any]) -> None:
    root = _safe_tree(root, "source fixture package")
    for split in ("training", "evaluation"):
        directory = root / split
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"source {split} directory is absent")
        expected = _dataset_files(manifest, split)
        live = {path.name for path in directory.iterdir() if path.is_file()}
        if live != set(expected):
            raise ValueError(f"source {split} file set differs from admitted manifest")
        for name, (digest, size) in expected.items():
            path = directory / name
            if path.stat().st_size != size or _sha(path) != digest:
                raise ValueError(f"source {split} bytes differ: {name}")
    archive = root / "training.zip"
    if (
        archive.is_symlink()
        or not archive.is_file()
        or archive.stat().st_size != manifest["training_archive"]["bytes"]
        or _sha(archive) != manifest["training_archive"]["sha256"]
    ):
        raise ValueError("source training archive differs from admitted manifest")


def _derived_manifest(
    *,
    role: str,
    source: Mapping[str, Any],
    source_file_sha256: str,
    approval: Mapping[str, Any],
    approval_file_sha256: str,
    freeze_file_sha256: str,
) -> dict[str, Any]:
    value = deepcopy(dict(source))
    value["schema"] = 3
    value["experimental_role"] = role
    value["source_governance"] = value.pop("governance")
    value["source_approval"] = deepcopy(dict(approval))
    value["boundary_derivation"] = {
        "mode": krea_fixture._STAGE2_BOUNDARY_DERIVATION_MODE,
        "source_role": _ROLE_SOURCE[role],
        "source_manifest_file_sha256": source_file_sha256,
        "source_manifest_sha256": source["manifest_sha256"],
        "source_approval_file_sha256": approval_file_sha256,
        "source_approval_sha256": approval["approval_sha256"],
        "public_freeze_binding": {
            "path": FREEZE_BINDING_PATH,
            "file_sha256": freeze_file_sha256,
            "binding_sha256": FREEZE_BINDING_SHA256,
            "commit_sha1": FREEZE_BINDING_COMMIT,
        },
        "actor": deepcopy(_DERIVATION_ACTOR),
        "fixture_bytes_changed": False,
        "group_evidence_changed": False,
        "source_governance_is_evidence_only": True,
        "source_governance_authorizes_boundary": False,
        "source_approval_authorizes_boundary": False,
        "fresh_stage2_owner_ratification_required": True,
        "boundary_admission_authorized": False,
        "gpu_execution_authorized": False,
        "science_selection_input": False,
        "claim_limit": krea_fixture._STAGE2_BOUNDARY_DERIVATION_CLAIM_LIMIT,
    }
    value.pop("manifest_sha256")
    value["manifest_sha256"] = krea_provenance.canonical_sha256(value)
    return krea_fixture.validate_manifest(value)


def build(
    *,
    d1_manifest_path: Path,
    d1_approval_path: Path,
    d1_package_root: Path,
    d2_manifest_path: Path,
    d2_approval_path: Path,
    d2_package_root: Path,
    freeze_binding_path: Path,
    output_dir: Path,
    created_at_utc: str,
) -> dict[str, Any]:
    created = _utc(created_at_utc)
    freeze, freeze_file_sha = _canonical(freeze_binding_path, "freeze binding")
    validate_public_freeze_binding(freeze, file_sha256=freeze_file_sha)
    sources: dict[str, dict[str, Any]] = {}
    for role, manifest_path, approval_path, package_root in (
        ("D1", d1_manifest_path, d1_approval_path, d1_package_root),
        ("D2", d2_manifest_path, d2_approval_path, d2_package_root),
    ):
        manifest, manifest_file_sha = _canonical(manifest_path, f"{role} manifest")
        manifest = krea_fixture.validate_manifest(manifest)
        if manifest.get("schema") != 2 or manifest.get("experimental_role") != role:
            raise ValueError(f"{role} source is not the admitted schema-2 fixture")
        approval, approval_file_sha = _canonical(approval_path, f"{role} approval")
        approval = krea_fixture.validate_approval(approval, fixture_manifest=manifest)
        _validate_package(package_root, manifest)
        sources[role] = {
            "manifest": manifest,
            "manifest_file_sha256": manifest_file_sha,
            "approval": approval,
            "approval_file_sha256": approval_file_sha,
            "package_root": package_root.resolve(strict=True),
        }

    destination = output_dir.resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"boundary derivation output already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        rows: list[dict[str, Any]] = []
        sealed = temporary / "sealed-boundary"
        for role in sorted(_ROLE_SOURCE):
            source = sources[_ROLE_SOURCE[role]]
            role_root = sealed / role
            for split in ("training", "evaluation"):
                for path in sorted((source["package_root"] / split).iterdir()):
                    if path.is_file():
                        _copy_new(path, role_root / split / path.name)
            _copy_new(
                source["package_root"] / "training.zip", role_root / "training.zip"
            )
            manifest = _derived_manifest(
                role=role,
                source=source["manifest"],
                source_file_sha256=source["manifest_file_sha256"],
                approval=source["approval"],
                approval_file_sha256=source["approval_file_sha256"],
                freeze_file_sha256=freeze_file_sha,
            )
            manifest_path = role_root / "fixture-manifest.json"
            manifest_path.write_bytes(krea_provenance.canonical_bytes(manifest) + b"\n")
            rows.append(
                {
                    "role": role,
                    "source_role": _ROLE_SOURCE[role],
                    "manifest_file_sha256": _sha(manifest_path),
                    "manifest_sha256": manifest["manifest_sha256"],
                    "training_archive_sha256": manifest["training_archive"]["sha256"],
                    "training_dataset_sha256": manifest["training_dataset_identity"][
                        "sha256"
                    ],
                    "evaluation_dataset_sha256": manifest[
                        "evaluation_dataset_identity"
                    ]["sha256"],
                }
            )
        body = {
            "schema": SCHEMA,
            "kind": KIND,
            "created_at_utc": created,
            "public_freeze_binding": {
                "path": FREEZE_BINDING_PATH,
                "file_sha256": freeze_file_sha,
                "binding_sha256": FREEZE_BINDING_SHA256,
                "commit_sha1": FREEZE_BINDING_COMMIT,
            },
            "roles": rows,
            "mechanics_only": True,
            "source_bytes_changed": False,
            "source_governance_reused_as_boundary_authority": False,
            "fresh_stage2_owner_ratification_required": True,
            "admission_authorized": False,
            "gpu_execution_authorized": False,
            "claim_limit": krea_fixture._STAGE2_BOUNDARY_DERIVATION_CLAIM_LIMIT,
        }
        record = {
            **body,
            "derivation_set_sha256": krea_provenance.canonical_sha256(body),
        }
        (temporary / "boundary-derivation-set.json").write_bytes(
            krea_provenance.canonical_bytes(record) + b"\n"
        )
        os.rename(temporary, destination)
        return record
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1-manifest", required=True, type=Path)
    parser.add_argument("--d1-approval", required=True, type=Path)
    parser.add_argument("--d1-package-root", required=True, type=Path)
    parser.add_argument("--d2-manifest", required=True, type=Path)
    parser.add_argument("--d2-approval", required=True, type=Path)
    parser.add_argument("--d2-package-root", required=True, type=Path)
    parser.add_argument("--freeze-binding", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--created-at-utc", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    value = build(
        d1_manifest_path=args.d1_manifest,
        d1_approval_path=args.d1_approval,
        d1_package_root=args.d1_package_root,
        d2_manifest_path=args.d2_manifest,
        d2_approval_path=args.d2_approval,
        d2_package_root=args.d2_package_root,
        freeze_binding_path=args.freeze_binding,
        output_dir=args.output_dir,
        created_at_utc=args.created_at_utc,
    )
    print(value["derivation_set_sha256"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
