#!/usr/bin/env python3
"""Create-only publisher for K2-K4 agent source-normalization reviews."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any

try:
    from . import krea_execution_plan
    from . import krea_fixture_admission
    from . import krea_provenance
except ImportError:  # pragma: no cover - direct script execution.
    import krea_execution_plan  # type: ignore[no-redef]
    import krea_fixture_admission  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]


_ARMS = ("K2", "K3", "K4")
_ACTOR = {
    "actor_class": "agent",
    "actor_id": "codex-public-arm-provenance-auditor",
    "display_name": "Codex public-arm provenance auditor (agent)",
    "role": "source_normalization_reviewer",
    "review_instance_id": "week5-krea-public-arm-review-20260729",
    "identity_assurance": (
        "self-declared-agent-identity-not-human-or-cryptographic-authentication"
    ),
}


def _binding(path: Path, label: str) -> dict[str, str]:
    path = krea_execution_plan._safe_file(path, label)
    return {
        "path": str(path),
        "sha256": krea_provenance.file_sha256(path),
    }


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    raw = krea_provenance.canonical_bytes(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _copy_exclusive(source: Path, destination: Path, label: str) -> None:
    source = krea_execution_plan._safe_file(source, label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fsync(writer.fileno())


def _bundle_inventory(root: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("review bundle contains a non-regular artifact")
        relative = path.relative_to(root).as_posix()
        if relative == "BUNDLE-MANIFEST.sha256":
            continue
        rows[relative] = krea_provenance.file_sha256(path)
    return rows


def validate_bundle(root: Path) -> dict[str, Any]:
    """Validate a relocated review bundle without any source-host paths."""

    root = krea_execution_plan._safe_directory(root, "public review bundle")
    manifest = krea_execution_plan._safe_file(
        root / "BUNDLE-MANIFEST.sha256", "review bundle manifest"
    )
    expected_rows: dict[str, str] = {}
    for index, line in enumerate(manifest.read_text(encoding="ascii").splitlines()):
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise ValueError(f"review bundle manifest row {index} is malformed")
        digest, relative_text = parts
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("review bundle manifest digest is invalid")
        relative = Path(relative_text)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError("review bundle manifest path is unsafe")
        portable = relative.as_posix()
        if portable in expected_rows:
            raise ValueError("review bundle manifest contains a duplicate path")
        expected_rows[portable] = digest
    if expected_rows != _bundle_inventory(root):
        raise ValueError("review bundle inventory differs from its manifest")

    reviews: dict[str, dict[str, Any]] = {}
    for arm in _ARMS:
        source_path = (
            root / "public-source-provenance" / f"{arm}-public-source-provenance.json"
        )
        review_path = root / f"{arm}-source-normalization-review.json"
        source_raw = source_path.read_bytes()
        review_raw = review_path.read_bytes()
        try:
            source = json.loads(source_raw)
            review = json.loads(review_raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("review bundle contains invalid JSON") from exc
        if (
            source_raw != krea_provenance.canonical_bytes(source) + b"\n"
            or review_raw != krea_provenance.canonical_bytes(review) + b"\n"
        ):
            raise ValueError("review bundle JSON is not canonical")
        krea_execution_plan._validate_agent_public_review(
            review,
            source_manifest=source,
            source_manifest_file_sha256=krea_provenance.file_sha256(source_path),
            review_path=review_path,
        )
        reviews[arm] = review
    return {
        "bundle_root": str(root),
        "bundle_manifest_file_sha256": krea_provenance.file_sha256(manifest),
        "review_file_sha256s": {
            arm: krea_provenance.file_sha256(
                root / f"{arm}-source-normalization-review.json"
            )
            for arm in _ARMS
        },
        "review_sha256s": {arm: reviews[arm]["review_sha256"] for arm in _ARMS},
    }


def create_reviews(
    *,
    public_evidence_root: Path,
    owner_ratification_path: Path,
    owner_ratification_draft_path: Path,
    output_dir: Path,
    reviewed_at_utc: str,
) -> dict[str, Any]:
    """Publish a self-contained, relocatable, create-only review bundle."""

    evidence_root = krea_execution_plan._safe_directory(
        public_evidence_root, "public evidence root"
    )
    source_thin_path = krea_execution_plan._safe_file(
        evidence_root / "MANIFEST.sha256", "thin manifest"
    )
    thin_rows = krea_execution_plan._thin_evidence_rows(source_thin_path)
    ratification_source = krea_execution_plan._safe_file(
        owner_ratification_path, "owner ratification"
    )
    draft_source = krea_execution_plan._safe_file(
        owner_ratification_draft_path, "owner ratification draft"
    )
    resolved = krea_fixture_admission._resolve_draft(draft_source)
    _, ratification, ratification_file_sha = krea_execution_plan._load_binding(
        _binding(ratification_source, "owner ratification"),
        "owner ratification",
    )
    krea_fixture_admission.validate_owner_ratification(ratification, resolved=resolved)
    output = Path(os.path.abspath(os.path.expanduser(output_dir)))
    governed_root = Path(krea_fixture_admission.__file__).resolve().parents[2]
    try:
        output.relative_to(governed_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "review bundle must be outside the ratified Forge Git worktree"
        )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite review output: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("review output parent must be a real directory")
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing to reuse temporary output: {temporary}")
    temporary.mkdir(mode=0o700)
    try:
        for relative_text in thin_rows:
            _copy_exclusive(
                evidence_root / Path(relative_text),
                temporary / Path(relative_text),
                f"thin evidence artifact {relative_text}",
            )
        _copy_exclusive(
            source_thin_path, temporary / "MANIFEST.sha256", "thin manifest"
        )
        governance = temporary / "governance"
        ratification_path = governance / "owner-ratification.json"
        portable_path = governance / "portable-ratification-draft.json"
        amendment_path = governance / "amendment.json"
        custodian_path = governance / "sealed-custodian-actor.json"
        _copy_exclusive(ratification_source, ratification_path, "owner ratification")
        _copy_exclusive(
            resolved["portable_draft_path"],
            portable_path,
            "portable ratification draft",
        )
        _copy_exclusive(
            Path(resolved["draft"]["inputs"]["governance_amendment"]),
            amendment_path,
            "governance amendment",
        )
        _copy_exclusive(
            resolved["sealed_custodian_actor_path"],
            custodian_path,
            "sealed custodian actor",
        )
        copied_thin_binding = _binding(
            temporary / "MANIFEST.sha256", "copied thin manifest"
        )
        copied_ratification_binding = _binding(
            ratification_path, "copied owner ratification"
        )
        copied_portable_binding = _binding(
            portable_path, "copied portable ratification draft"
        )
        copied_amendment_binding = _binding(
            amendment_path, "copied governance amendment"
        )
        copied_custodian_binding = _binding(
            custodian_path, "copied sealed custodian actor"
        )
        reviews: dict[str, dict[str, Any]] = {}
        for arm in _ARMS:
            source_path = (
                temporary
                / "public-source-provenance"
                / f"{arm}-public-source-provenance.json"
            )
            destination = temporary / f"{arm}-source-normalization-review.json"
            review = krea_execution_plan.build_agent_public_source_review(
                review_output_path=destination,
                source_provenance=_binding(
                    source_path, f"{arm} public source provenance"
                ),
                thin_evidence_manifest=copied_thin_binding,
                owner_ratification=copied_ratification_binding,
                portable_ratification_draft=copied_portable_binding,
                governance_amendment=copied_amendment_binding,
                sealed_custodian_actor=copied_custodian_binding,
                actor=_ACTOR,
                reviewed_at_utc=reviewed_at_utc,
            )
            _write_exclusive(destination, review)
            reviews[arm] = review
        inventory = _bundle_inventory(temporary)
        bundle_manifest_path = temporary / "BUNDLE-MANIFEST.sha256"
        with bundle_manifest_path.open("x", encoding="ascii", newline="") as handle:
            handle.writelines(
                f"{digest}  {relative}\n"
                for relative, digest in sorted(inventory.items())
            )
            handle.flush()
            os.fsync(handle.fileno())
        validate_bundle(temporary)
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    summary = validate_bundle(output)
    if ratification_file_sha != krea_provenance.file_sha256(
        output / "governance" / "owner-ratification.json"
    ):
        raise ValueError("published owner ratification bytes changed")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-evidence-root", type=Path, required=True)
    parser.add_argument("--owner-ratification", type=Path, required=True)
    parser.add_argument("--owner-ratification-draft", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reviewed-at-utc",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    args = parser.parse_args(argv)
    summary = create_reviews(
        public_evidence_root=args.public_evidence_root,
        owner_ratification_path=args.owner_ratification,
        owner_ratification_draft_path=args.owner_ratification_draft,
        output_dir=args.output_dir,
        reviewed_at_utc=args.reviewed_at_utc,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI.
    raise SystemExit(main())
