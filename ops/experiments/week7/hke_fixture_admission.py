#!/usr/bin/env python3
"""Human-review and admission boundary for Week-7 HKE fixtures.

The procedural renderer can create and machine-check candidates.  It cannot
approve rights or impersonate a human reviewer.  This module keeps that
boundary explicit:

1. a custodian generates a private review template bound to every exact row;
2. an operator attests that a named human reviewed and filled every check;
3. ``seal-review`` validates that filled draft and publishes a create-only
   private record; and
4. ``admit`` replays the renderer, validates the sealed review, and emits three
   public family receipts.  Those receipts still say GPU execution is false.

The record does not cryptographically authenticate that person, so owner
ratification remains mandatory before any GPU authorization. No command trains,
scores, rents hardware, modifies production, or reveals confirmation membership
in the public admission receipts.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types
from typing import Any, Callable, Mapping, Sequence

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
RENDERER_PATH = SCRIPT_PATH.with_name("hke_procedural_renderer.py")
FACTOR_AUTHORITY_PATH = SCRIPT_PATH.with_name("run_hke_factorial.py")
RENDERER_SOURCE_PATH = "ops/experiments/week7/hke_procedural_renderer.py"
FACTOR_AUTHORITY_SOURCE_PATH = "ops/experiments/week7/run_hke_factorial.py"
ADMISSION_SOURCE_PATH = "ops/experiments/week7/hke_fixture_admission.py"


class AdmissionError(RuntimeError):
    """The human/replay/admission chain is incomplete or misbound."""


def _read_regular_source_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("procedural renderer source is not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        source = b"".join(chunks)
    finally:
        os.close(descriptor)
    return source


def _committed_source_bytes(repo_root: Path, source_path: str) -> bytes:
    """Read one literal HEAD blob without filters, replacements, or Git config."""

    git = Path("/usr/bin/git")
    if not git.is_file():
        raise AdmissionError("fixed /usr/bin/git is unavailable")
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent-sn56-admission-source",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }
    completed = subprocess.run(
        [
            str(git),
            "--no-replace-objects",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "filter.lfs.process=",
            "-c",
            "filter.lfs.clean=",
            "-c",
            "filter.lfs.smudge=",
            "cat-file",
            "blob",
            f"HEAD:{source_path}",
        ],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise AdmissionError(f"committed authority blob is unavailable: {source_path}")
    return completed.stdout


def _load_committed_source_module(
    *,
    repo_root: Path,
    source_path: str,
    ambient_path: Path,
    module_name: str,
) -> tuple[types.ModuleType, str]:
    """Verify ambient bytes, then execute only the literal committed blob.

    The comparison deliberately happens before ``compile``/``exec``.  A dirty,
    foreign, or symlink-swapped authority therefore aborts without running any
    bytes from the ambient checkout.
    """

    committed = _committed_source_bytes(repo_root, source_path)
    ambient = _read_regular_source_bytes(ambient_path)
    if not hmac.compare_digest(committed, ambient):
        raise AdmissionError(
            f"ambient authority differs from committed blob: {source_path}"
        )
    source_sha256 = hashlib.sha256(committed).hexdigest()
    name = module_name
    module = types.ModuleType(name)
    module.__file__ = str(ambient_path)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(committed, str(ambient_path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, source_sha256


renderer, RENDERER_EXECUTED_SOURCE_SHA256 = _load_committed_source_module(
    repo_root=REPO_ROOT,
    source_path=RENDERER_SOURCE_PATH,
    ambient_path=RENDERER_PATH,
    module_name="week7_hke_renderer_for_admission",
)
ADMISSION_EXECUTED_SOURCE_SHA256 = hashlib.sha256(
    _read_regular_source_bytes(SCRIPT_PATH)
).hexdigest()


SCHEMA = 3
REVIEW_KIND = "sn56-week7-hke-human-fixture-review"
ADMISSION_KIND = "sn56-week7-hke-fixture-admission"
ROOT_ADMISSION_KIND = "sn56-week7-hke-fixture-admission-set"
RATIFICATION_KIND = "sn56-week7-hke-owner-ratification"
CONFIRMATION_FREEZE_KIND = "sn56-week7-hke-confirmation-candidate-freeze"
C2_AUTHORITY_KIND = "sn56-week7-hke-c2-reveal-authorization"
CONFIRMATION_REVEAL_KIND = "sn56-week7-hke-private-confirmation-reveal"
REQUIRED_CHECKS = {
    "visual_semantics_accepted",
    "caption_semantics_accepted",
    "first_party_rights_and_training_evaluation_use_accepted",
    "no_third_party_brand_mark_person_character_or_artist_reference",
    "no_external_asset_font_provider_reference_or_tournament_content",
    "cross_family_and_cross_phase_similarity_reviewed",
}
DISALLOWED_REVIEWER_LABELS = {
    "agent",
    "assistant",
    "codex",
    "human",
    "human owner",
    "human reviewer",
    "owner",
    "reviewer",
    "team",
}


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            separators=(",", ": "),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    raw = renderer._read_regular(Path(path), label)
    return _decode_json(raw, label)


def _decode_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"{label} is not JSON") from exc
    if not isinstance(value, dict):
        raise AdmissionError(f"{label} is not an object")
    return value


def _verify_candidate(
    public_root: Path,
    custodian_root: Path,
    public_boundary_roots: Sequence[Path],
) -> dict[str, Any]:
    """Translate renderer custody/integrity failures at the admission boundary."""

    try:
        return renderer.verify_candidate(
            Path(public_root),
            Path(custodian_root),
            public_boundary_roots=public_boundary_roots,
        )
    except renderer.FixtureError as exc:
        raise AdmissionError(str(exc)) from exc


def _write_new(path: Path, value: Any) -> str:
    path = Path(path)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise AdmissionError(f"output parent is unavailable: {path.parent}")
    payload = canonical_bytes(value)
    try:
        renderer._write_exclusive(path, payload)
    except FileExistsError as exc:
        raise AdmissionError(f"refusing to overwrite {path}") from exc
    return hashlib.sha256(payload).hexdigest()


def _private_record_path(
    path: Path,
    *,
    public_root: Path,
    custodian_root: Path,
    public_boundary_roots: Sequence[Path],
    label: str,
) -> Path:
    """Require human-review records to live outside public/candidate trees."""

    path = renderer._absolute(Path(path))
    public_root = renderer._absolute(Path(public_root))
    custodian_root = renderer._absolute(Path(custodian_root))
    try:
        renderer._validate_custody_boundary(
            public_root=public_root,
            custodian_root=custodian_root,
            public_boundary_roots=public_boundary_roots,
        )
    except renderer.FixtureError as exc:
        raise AdmissionError(str(exc)) from exc
    try:
        worktree_roots = renderer._registered_worktree_roots()
        forbidden_roots = (
            public_root,
            custodian_root,
            REPO_ROOT,
            *worktree_roots,
            *(Path(boundary) for boundary in public_boundary_roots),
        )
        overlaps_forbidden = any(
            renderer._paths_overlap(path, root) for root in forbidden_roots
        )
    except renderer.FixtureError as exc:
        raise AdmissionError(str(exc)) from exc
    if overlaps_forbidden:
        raise AdmissionError(
            f"{label} must be outside public, candidate-custodian, repository, "
            "and every registered-worktree tree"
        )
    renderer._require_no_symlink_components(path.parent, f"{label} parent")
    if not path.parent.is_dir():
        raise AdmissionError(f"{label} parent is unavailable")
    return path


def _assert_private_parent_bound(
    parent_descriptor: int,
    path: Path,
    *,
    public_root: Path,
    custodian_root: Path,
    public_boundary_roots: Sequence[Path],
    label: str,
) -> None:
    """Recheck custody and bind the live pathname to one held parent inode."""

    checked = _private_record_path(
        path,
        public_root=public_root,
        custodian_root=custodian_root,
        public_boundary_roots=public_boundary_roots,
        label=label,
    )
    try:
        live_plan = renderer._path_identity_plan(checked.parent)
        held = os.fstat(parent_descriptor)
    except (OSError, renderer.FixtureError) as exc:
        raise AdmissionError(f"{label} parent identity is unavailable") from exc
    live = live_plan[-1]
    if live[1] is None or (live[1], live[2]) != (held.st_dev, held.st_ino):
        raise AdmissionError(f"{label} parent changed after custody validation")


def _open_private_parent(
    path: Path,
    *,
    public_root: Path,
    custodian_root: Path,
    public_boundary_roots: Sequence[Path],
    label: str,
) -> tuple[Path, int]:
    checked = _private_record_path(
        path,
        public_root=public_root,
        custodian_root=custodian_root,
        public_boundary_roots=public_boundary_roots,
        label=label,
    )
    try:
        descriptor = renderer._open_directory_chain_no_symlinks(
            checked.parent, f"{label} parent"
        )
    except renderer.FixtureError as exc:
        raise AdmissionError(str(exc)) from exc
    try:
        _assert_private_parent_bound(
            descriptor,
            checked,
            public_root=public_root,
            custodian_root=custodian_root,
            public_boundary_roots=public_boundary_roots,
            label=label,
        )
    except BaseException:
        os.close(descriptor)
        raise
    return checked, descriptor


def _assert_private_target_bound(
    parent_descriptor: int,
    path: Path,
    created_descriptor: int,
    created_metadata: os.stat_result,
    *,
    public_root: Path,
    custodian_root: Path,
    public_boundary_roots: Sequence[Path],
    label: str,
) -> None:
    """Bind success to the requested live path and the exact created inode."""

    for _ in range(2):
        _assert_private_parent_bound(
            parent_descriptor,
            path,
            public_root=public_root,
            custodian_root=custodian_root,
            public_boundary_roots=public_boundary_roots,
            label=label,
        )
        live_parent: int | None = None
        try:
            live_parent = renderer._open_directory_chain_no_symlinks(
                path.parent, f"{label} live parent"
            )
            held_parent_metadata = os.fstat(parent_descriptor)
            live_parent_metadata = os.fstat(live_parent)
            live_target = os.stat(path.name, dir_fd=live_parent, follow_symlinks=False)
            created_now = os.fstat(created_descriptor)
        except (OSError, renderer.FixtureError) as exc:
            raise AdmissionError(f"{label} live target is unavailable") from exc
        finally:
            if live_parent is not None:
                try:
                    os.close(live_parent)
                except OSError:
                    pass
        if (
            (held_parent_metadata.st_dev, held_parent_metadata.st_ino)
            != (live_parent_metadata.st_dev, live_parent_metadata.st_ino)
            or not stat.S_ISREG(live_target.st_mode)
            or not stat.S_ISREG(created_now.st_mode)
            or live_target.st_nlink != 1
            or created_now.st_nlink != 1
            or (
                live_target.st_dev,
                live_target.st_ino,
                live_target.st_size,
            )
            != (
                created_metadata.st_dev,
                created_metadata.st_ino,
                created_metadata.st_size,
            )
            or (
                created_now.st_dev,
                created_now.st_ino,
                created_now.st_size,
            )
            != (
                created_metadata.st_dev,
                created_metadata.st_ino,
                created_metadata.st_size,
            )
        ):
            raise AdmissionError(f"{label} live target changed during publication")


def _assert_private_payload_bound(
    parent_descriptor: int,
    path: Path,
    expected_payload: bytes,
    *,
    public_root: Path,
    custodian_root: Path,
    public_boundary_roots: Sequence[Path],
    label: str,
) -> None:
    """Rebind a completed read/write to its requested live single-link path."""

    for _ in range(2):
        _assert_private_parent_bound(
            parent_descriptor,
            path,
            public_root=public_root,
            custodian_root=custodian_root,
            public_boundary_roots=public_boundary_roots,
            label=label,
        )
        try:
            live_payload = renderer._read_regular_at(
                parent_descriptor, path.name, label
            )
        except renderer.FixtureError as exc:
            raise AdmissionError(str(exc)) from exc
        if not hmac.compare_digest(live_payload, expected_payload):
            raise AdmissionError(f"{label} live payload changed")


def _load_private_json(
    path: Path,
    *,
    public_root: Path,
    custodian_root: Path,
    public_boundary_roots: Sequence[Path],
    label: str,
) -> dict[str, Any]:
    """Read a private record through the descriptor used for custody checks."""

    checked, parent_descriptor = _open_private_parent(
        path,
        public_root=public_root,
        custodian_root=custodian_root,
        public_boundary_roots=public_boundary_roots,
        label=label,
    )
    try:
        try:
            raw = renderer._read_regular_at(parent_descriptor, checked.name, label)
        except renderer.FixtureError as exc:
            raise AdmissionError(str(exc)) from exc
        _assert_private_parent_bound(
            parent_descriptor,
            checked,
            public_root=public_root,
            custodian_root=custodian_root,
            public_boundary_roots=public_boundary_roots,
            label=label,
        )
        value = _decode_json(raw, label)
        _assert_private_payload_bound(
            parent_descriptor,
            checked,
            raw,
            public_root=public_root,
            custodian_root=custodian_root,
            public_boundary_roots=public_boundary_roots,
            label=label,
        )
        return value
    finally:
        os.close(parent_descriptor)


def _write_private_new(
    path: Path,
    value: Any,
    *,
    public_root: Path,
    custodian_root: Path,
    public_boundary_roots: Sequence[Path],
    label: str,
) -> str:
    """Publish privately through one held parent and recheck worktree custody."""

    checked, parent_descriptor = _open_private_parent(
        path,
        public_root=public_root,
        custodian_root=custodian_root,
        public_boundary_roots=public_boundary_roots,
        label=label,
    )
    payload = canonical_bytes(value)
    retained_descriptor: int | None = None

    def validate_and_retain(descriptor: int, metadata: os.stat_result) -> None:
        nonlocal retained_descriptor
        _assert_private_target_bound(
            parent_descriptor,
            checked,
            descriptor,
            metadata,
            public_root=public_root,
            custodian_root=custodian_root,
            public_boundary_roots=public_boundary_roots,
            label=label,
        )
        retained_descriptor = os.dup(descriptor)

    try:
        _assert_private_parent_bound(
            parent_descriptor,
            checked,
            public_root=public_root,
            custodian_root=custodian_root,
            public_boundary_roots=public_boundary_roots,
            label=label,
        )
        try:
            renderer._write_exclusive_at(
                parent_descriptor,
                checked.name,
                payload,
                post_write_validation=validate_and_retain,
            )
        except FileExistsError as exc:
            raise AdmissionError(f"refusing to overwrite {checked}") from exc
        except renderer.FixtureError as exc:
            raise AdmissionError(str(exc)) from exc
        _assert_private_payload_bound(
            parent_descriptor,
            checked,
            payload,
            public_root=public_root,
            custodian_root=custodian_root,
            public_boundary_roots=public_boundary_roots,
            label=label,
        )
        return hashlib.sha256(payload).hexdigest()
    except BaseException:
        # POSIX has no identity-conditional unlink.  The renderer scrubs the
        # exact created inode through its held descriptor if the post-write
        # custody check fails; this layer never deletes a mutable pathname.
        if retained_descriptor is not None:
            try:
                os.ftruncate(retained_descriptor, 0)
                os.fsync(retained_descriptor)
            except OSError:
                pass
        raise
    finally:
        if retained_descriptor is not None:
            os.close(retained_descriptor)
        os.close(parent_descriptor)


def _rows(verified: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for phase, manifest_name in (
        ("discovery", "discovery_manifest"),
        ("confirmation", "confirmation_manifest"),
    ):
        manifest = verified[manifest_name]
        for fixture_id, family in sorted(manifest["families"].items()):
            for row in family["rows"]:
                result.append(
                    {
                        "fixture_id": fixture_id,
                        "family": row["family"],
                        "phase": phase,
                        "pack": row["pack"],
                        "split_role": row["split_role"],
                        "row_id": row["row_id"],
                        "image_sha256": row["image_sha256"],
                        "caption_sha256": row["caption_sha256"],
                        "decoded_pixels_sha256": row["decoded_pixels_sha256"],
                        "parameters_sha256": row["parameters_sha256"],
                        "group_identity_sha256": row["group_identity_sha256"],
                        "row_record_sha256": row["row_record_sha256"],
                    }
                )
    return result


def build_review_template(
    public_root: Path,
    custodian_root: Path,
    *,
    public_boundary_roots: Sequence[Path],
) -> dict[str, Any]:
    """Build a private, exact-row checklist; this is not a review."""

    verified = _verify_candidate(public_root, custodian_root, public_boundary_roots)
    candidate = verified["candidate_manifest"]
    body = {
        "schema": SCHEMA,
        "kind": REVIEW_KIND,
        "status": "PENDING_NAMED_HUMAN_REVIEW",
        "candidate_semantic_sha256": candidate["semantic_sha256"],
        "rights_record": candidate["rights_record"],
        "confirmation_manifest_file_sha256": candidate["confirmation"][
            "custodian_manifest_sha256"
        ],
        "reviewer_identity": "",
        "reviewed_at_utc": "",
        "rows": [
            {
                **row,
                "checks": {key: False for key in sorted(REQUIRED_CHECKS)},
                "notes": "",
            }
            for row in _rows(verified)
        ],
        "decision": "PENDING",
        "governance": {
            "agent_review_is_not_human_review": True,
            "review_must_cover_all_candidate_rows": True,
            "confirmation_record_is_private": True,
        },
    }
    return body


def _named_human(value: Any) -> str:
    text = " ".join(str(value or "").split())
    lowered_words = {word.casefold() for word in re.findall(r"[A-Za-z]+", text)}
    role_words = {"agent", "assistant", "codex", "human", "owner", "reviewer", "team"}
    if text.casefold() in DISALLOWED_REVIEWER_LABELS or lowered_words & role_words:
        raise AdmissionError("reviewer_identity is a role label, not a named human")
    words = re.findall(r"[A-Za-z][A-Za-z'.-]+", text)
    if len(words) < 2:
        raise AdmissionError("reviewer_identity must name a human")
    return text


def _utc(value: Any) -> str:
    text = str(value or "")
    if not text.endswith("Z"):
        raise AdmissionError("reviewed_at_utc must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise AdmissionError("reviewed_at_utc is invalid") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise AdmissionError("reviewed_at_utc is not UTC")
    return text


def _expected_review_rows(verified: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _rows(verified)


def validate_review_body(
    value: Mapping[str, Any], verified: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a filled human draft before it is sealed."""

    candidate = verified["candidate_manifest"]
    if (
        value.get("schema") != SCHEMA
        or value.get("kind") != REVIEW_KIND
        or value.get("candidate_semantic_sha256") != candidate["semantic_sha256"]
        or value.get("rights_record") != candidate["rights_record"]
        or value.get("confirmation_manifest_file_sha256")
        != candidate["confirmation"]["custodian_manifest_sha256"]
        or value.get("decision") != "PASS"
    ):
        raise AdmissionError("human review identity, binding, or decision is invalid")
    reviewer = _named_human(value.get("reviewer_identity"))
    reviewed_at = _utc(value.get("reviewed_at_utc"))
    rows = value.get("rows")
    expected = _expected_review_rows(verified)
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise AdmissionError("human review must cover exactly all candidate rows")
    for index, (actual, identity) in enumerate(zip(rows, expected, strict=True)):
        if not isinstance(actual, dict):
            raise AdmissionError(f"human review row {index} is not an object")
        projected = {key: actual.get(key) for key in identity}
        if projected != identity:
            raise AdmissionError(f"human review row {index} identity mismatch")
        checks = actual.get("checks")
        if not isinstance(checks, dict) or set(checks) != REQUIRED_CHECKS:
            raise AdmissionError(f"human review row {index} checks are incomplete")
        if any(value is not True for value in checks.values()):
            raise AdmissionError(f"human review row {index} has an unapproved check")
        if not isinstance(actual.get("notes", ""), str):
            raise AdmissionError(f"human review row {index} notes are invalid")
    governance = value.get("governance")
    if governance != {
        "agent_review_is_not_human_review": True,
        "review_must_cover_all_candidate_rows": True,
        "confirmation_record_is_private": True,
    }:
        raise AdmissionError("human review governance was changed")
    return {
        **dict(value),
        "status": "SEALED_OPERATOR_ATTESTED_NAMED_HUMAN_PASS",
        "reviewer_identity": reviewer,
        "reviewed_at_utc": reviewed_at,
    }


def seal_review(
    *,
    public_root: Path,
    custodian_root: Path,
    public_boundary_roots: Sequence[Path],
    draft: Mapping[str, Any],
) -> dict[str, Any]:
    verified = _verify_candidate(public_root, custodian_root, public_boundary_roots)
    body = validate_review_body(draft, verified)
    body.pop("review_sha256", None)
    return {**body, "review_sha256": semantic_sha256(body)}


def validate_sealed_review(
    value: Mapping[str, Any], verified: Mapping[str, Any]
) -> dict[str, Any]:
    if value.get("status") != "SEALED_OPERATOR_ATTESTED_NAMED_HUMAN_PASS":
        raise AdmissionError("sealed human review has not passed")
    declared = value.get("review_sha256")
    body = dict(value)
    body.pop("review_sha256", None)
    if declared != semantic_sha256(body):
        raise AdmissionError("sealed human review digest mismatch")
    return validate_review_body(body, verified)


def _family_counts(candidate: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for value in candidate["families"].values():
        result[value["family"]] = {
            "discovery": value["discovery"]["row_count"],
            "confirmation": value["confirmation"]["row_count"],
        }
    return result


def _split_inventory(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for row in rows:
        files.extend(
            (
                {
                    "path": Path(row["relative_image_path"]).name,
                    "bytes": row["image_bytes"],
                    "sha256": row["image_sha256"],
                },
                {
                    "path": Path(row["relative_caption_path"]).name,
                    "bytes": row["caption_bytes"],
                    "sha256": row["caption_sha256"],
                },
            )
        )
    files.sort(key=lambda item: item["path"])
    if len({item["path"] for item in files}) != len(files):
        raise AdmissionError("fixture split inventory has duplicate flat paths")
    body = {"files": files, "file_count": len(files)}
    return {**body, "semantic_sha256": semantic_sha256(body)}


def _assert_disjoint_split_rows(
    training_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    """Fail closed if membership or stable semantic identity crosses a split."""

    if not training_rows or not evaluation_rows:
        raise AdmissionError(f"{label} training/evaluation split is empty")
    if any(row.get("split_role") != "training" for row in training_rows) or any(
        row.get("split_role") != "evaluation" for row in evaluation_rows
    ):
        raise AdmissionError(f"{label} split-role declaration is inconsistent")
    identity_fields = (
        "row_id",
        "row_record_sha256",
        "image_sha256",
        "decoded_pixels_sha256",
        "caption_sha256",
        "parameters_sha256",
        "group_identity_sha256",
    )
    for field in identity_fields:
        training_values = {row.get(field) for row in training_rows}
        evaluation_values = {row.get(field) for row in evaluation_rows}
        if None in training_values or None in evaluation_values:
            raise AdmissionError(f"{label} split is missing {field}")
        if training_values & evaluation_values:
            raise AdmissionError(
                f"{label} training/evaluation split overlaps by {field}"
            )


def _public_pack_receipts(family_record: Mapping[str, Any]) -> dict[str, Any]:
    packs: dict[str, Any] = {}
    for pack, record in sorted(family_record["discovery"]["packs"].items()):
        training_rows = record["splits"]["training"]["rows"]
        evaluation_rows = record["splits"]["evaluation"]["rows"]
        _assert_disjoint_split_rows(
            training_rows, evaluation_rows, label=f"discovery pack {pack}"
        )
        training_identity = [
            {"row_id": row["row_id"], "row_sha256": row["row_record_sha256"]}
            for row in training_rows
        ]
        evaluation_identity = [
            {"row_id": row["row_id"], "row_sha256": row["row_record_sha256"]}
            for row in evaluation_rows
        ]
        training_inventory = _split_inventory(training_rows)
        evaluation_inventory = _split_inventory(evaluation_rows)
        packs[pack] = {
            "phase": "discovery",
            "row_count": record["row_count"],
            "training_row_count": len(training_rows),
            "evaluation_row_count": len(evaluation_rows),
            "training_row_identity_sha256": semantic_sha256(training_identity),
            "evaluation_row_identity_sha256": semantic_sha256(evaluation_identity),
            "training_inventory": training_inventory,
            "training_inventory_sha256": training_inventory["semantic_sha256"],
            "evaluation_inventory": evaluation_inventory,
            "evaluation_inventory_sha256": evaluation_inventory["semantic_sha256"],
        }
    for pack, record in sorted(family_record["confirmation"]["packs"].items()):
        packs[pack] = {
            "phase": "confirmation",
            "row_count": record["row_count"],
            "training_row_count": record["training_row_count"],
            "evaluation_row_count": record["evaluation_row_count"],
            "semantic_commitment_sha256": record["semantic_commitment_sha256"],
        }
    return packs


def _current_generator_identity() -> dict[str, Any]:
    """Bind admission to the committed and pushed renderer bytes it executes."""

    git = Path("/usr/bin/git")
    if not git.is_file():
        raise AdmissionError("fixed /usr/bin/git is unavailable")
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent-sn56-admission",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }

    def run(arguments: Sequence[str], *, binary: bool = False):
        completed = subprocess.run(
            [
                str(git),
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.hooksPath=/dev/null",
                *arguments,
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=not binary,
        )
        if completed.returncode != 0:
            raise AdmissionError(
                f"generator Git identity command failed: {' '.join(arguments)}"
            )
        return completed.stdout

    commit = str(run(("rev-parse", "--verify", "HEAD^{commit}"))).strip().lower()
    tree = str(run(("rev-parse", "--verify", "HEAD^{tree}"))).strip().lower()
    renderer_blob = run(
        ("cat-file", "blob", f"HEAD:{renderer.GENERATOR_SOURCE_PATH}"),
        binary=True,
    )
    if not isinstance(renderer_blob, bytes):  # pragma: no cover
        raise AdmissionError("committed renderer bytes are unavailable")
    source_sha = hashlib.sha256(renderer_blob).hexdigest()
    if source_sha != RENDERER_EXECUTED_SOURCE_SHA256:
        raise AdmissionError("executed renderer module differs from the committed blob")
    if not hmac.compare_digest(
        renderer_blob, _read_regular_source_bytes(RENDERER_PATH)
    ):
        raise AdmissionError("executed renderer bytes differ from the committed blob")
    authority_blob = run(
        ("cat-file", "blob", f"HEAD:{ADMISSION_SOURCE_PATH}"),
        binary=True,
    )
    if not isinstance(authority_blob, bytes):  # pragma: no cover
        raise AdmissionError("committed admission-authority bytes are unavailable")
    authority_source_sha = hashlib.sha256(authority_blob).hexdigest()
    if (
        authority_source_sha != ADMISSION_EXECUTED_SOURCE_SHA256
        or not hmac.compare_digest(
            authority_blob, _read_regular_source_bytes(SCRIPT_PATH)
        )
    ):
        raise AdmissionError("executed admission authority differs from committed blob")
    factor_blob = run(
        ("cat-file", "blob", f"HEAD:{FACTOR_AUTHORITY_SOURCE_PATH}"),
        binary=True,
    )
    if not isinstance(factor_blob, bytes):  # pragma: no cover
        raise AdmissionError("committed factor-authority bytes are unavailable")
    factor_source_sha = hashlib.sha256(factor_blob).hexdigest()
    if not hmac.compare_digest(
        factor_blob, _read_regular_source_bytes(FACTOR_AUTHORITY_PATH)
    ):
        raise AdmissionError("executed factor authority differs from committed blob")
    contract_path = renderer.DECLARATIVE_CONTRACT_SOURCE_PATH
    contract_blob = run(
        ("cat-file", "blob", f"HEAD:{contract_path}"),
        binary=True,
    )
    if not isinstance(contract_blob, bytes):  # pragma: no cover
        raise AdmissionError("committed fixture-contract bytes are unavailable")
    contract_source_sha = hashlib.sha256(contract_blob).hexdigest()
    ambient_contract = renderer._read_regular(
        renderer.DECLARATIVE_CONTRACT_PATH, "ambient declarative fixture contract"
    )
    if not hmac.compare_digest(contract_blob, ambient_contract):
        raise AdmissionError("ambient fixture contract differs from committed blob")

    remote = subprocess.run(
        [
            str(git),
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "ls-remote",
            "--heads",
            renderer.GENERATOR_REPOSITORY,
        ],
        cwd="/",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if remote.returncode != 0:
        raise AdmissionError("cannot prove generator commit on the pinned remote")
    remote_refs = sorted(
        line.split("\t", 1)[1]
        for line in remote.stdout.splitlines()
        if "\t" in line and line.split("\t", 1)[0].lower() == commit
    )
    if not remote_refs:
        raise AdmissionError("generator commit is not present on the pinned remote")
    return {
        "repository": renderer.GENERATOR_REPOSITORY,
        "commit": commit,
        "tree": tree,
        "source_path": renderer.GENERATOR_SOURCE_PATH,
        "renderer_source_sha256": source_sha,
        "admission_authority_path": ADMISSION_SOURCE_PATH,
        "admission_authority_source_sha256": authority_source_sha,
        "factor_authority_path": FACTOR_AUTHORITY_SOURCE_PATH,
        "factor_authority_source_sha256": factor_source_sha,
        "contract_path": contract_path,
        "contract_source_sha256": contract_source_sha,
        "pinned_remote_refs": remote_refs,
    }


def build_admissions(
    *,
    public_root: Path,
    custodian_root: Path,
    discovery_key: bytes,
    confirmation_key: bytes,
    public_boundary_roots: Sequence[Path],
    sealed_review: Mapping[str, Any],
    generator_identity_probe: Callable[
        [], Mapping[str, Any]
    ] = _current_generator_identity,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        discovery_key = renderer._validate_key(discovery_key, "discovery key")
        confirmation_key = renderer._validate_key(confirmation_key, "confirmation key")
        renderer._require_distinct_phase_keys(discovery_key, confirmation_key)
    except renderer.FixtureError as exc:
        raise AdmissionError(str(exc)) from exc
    phase_commitments = {
        "discovery": renderer._phase_key_commitment(
            discovery_key, renderer.DISCOVERY_DOMAIN
        ),
        "confirmation": renderer._phase_key_commitment(
            confirmation_key, renderer.CONFIRMATION_DOMAIN
        ),
    }
    verified = _verify_candidate(public_root, custodian_root, public_boundary_roots)
    if (
        verified["discovery_manifest"].get("phase_key_commitment_sha256")
        != phase_commitments["discovery"]
        or verified["confirmation_manifest"].get("phase_key_commitment_sha256")
        != phase_commitments["confirmation"]
    ):
        raise AdmissionError("supplied phase keys do not match candidate commitments")
    try:
        replay = renderer.verify_replay(
            public_output=Path(public_root),
            custodian_output=Path(custodian_root),
            discovery_key=discovery_key,
            confirmation_key=confirmation_key,
            public_boundary_roots=public_boundary_roots,
        )
    except renderer.FixtureError as exc:
        raise AdmissionError(str(exc)) from exc
    review = validate_sealed_review(sealed_review, verified)
    candidate = verified["candidate_manifest"]
    live_generator = dict(generator_identity_probe())
    candidate_generator = candidate["generator"]
    for key in (
        "repository",
        "commit",
        "tree",
        "source_path",
        "renderer_source_sha256",
        "contract_path",
        "contract_source_sha256",
    ):
        if live_generator.get(key) != candidate_generator.get(key):
            raise AdmissionError(
                f"candidate generator {key} is not the executed pushed revision"
            )
    if not live_generator.get("pinned_remote_refs"):
        raise AdmissionError("candidate generator has no pinned-remote evidence")
    if live_generator.get("admission_authority_path") != ADMISSION_SOURCE_PATH:
        raise AdmissionError("admission authority path identity is absent")
    authority_source_sha = live_generator.get("admission_authority_source_sha256")
    if not isinstance(authority_source_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", authority_source_sha
    ):
        raise AdmissionError("admission authority source identity is absent")
    if live_generator.get("factor_authority_path") != FACTOR_AUTHORITY_SOURCE_PATH:
        raise AdmissionError("factor authority path identity is absent")
    factor_source_sha = live_generator.get("factor_authority_source_sha256")
    if not isinstance(factor_source_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", factor_source_sha
    ):
        raise AdmissionError("factor authority source identity is absent")
    dedup = verified["dedup_evidence"]
    if any(
        dedup[key] != 0
        for key in (
            "exact_image_duplicate_count",
            "decoded_pixel_duplicate_count",
            "normalized_caption_duplicate_count",
            "group_identity_duplicate_count",
            "perceptual_near_duplicate_count",
        )
    ):
        raise AdmissionError("fixture duplicate screen is not clean")
    replay_evidence = {
        "status": replay["status"],
        "verified_rows": replay["verified_rows"],
        "candidate_semantic_sha256": replay["candidate_semantic_sha256"],
        "dedup_semantic_sha256": replay["dedup_semantic_sha256"],
        "discovery_key_commitment_sha256": replay.get(
            "discovery_key_commitment_sha256"
        ),
        "confirmation_key_commitment_sha256": replay.get(
            "confirmation_key_commitment_sha256"
        ),
        "contract_path": replay.get("contract_path"),
        "contract_source_sha256": replay.get("contract_source_sha256"),
    }
    if (
        replay_evidence["discovery_key_commitment_sha256"]
        != phase_commitments["discovery"]
        or replay_evidence["confirmation_key_commitment_sha256"]
        != phase_commitments["confirmation"]
        or replay_evidence["contract_path"] != live_generator["contract_path"]
        or replay_evidence["contract_source_sha256"]
        != live_generator["contract_source_sha256"]
    ):
        raise AdmissionError("replay phase or contract binding mismatch")
    replay_sha = semantic_sha256(replay_evidence)
    counts = _family_counts(candidate)
    generator_revision = {
        "repository": live_generator["repository"],
        "commit": live_generator["commit"],
        "tree": live_generator["tree"],
        "renderer_source_path": live_generator["source_path"],
        "renderer_source_sha256": live_generator["renderer_source_sha256"],
        "admission_authority_path": live_generator["admission_authority_path"],
        "admission_authority_source_sha256": authority_source_sha,
        "factor_authority_path": live_generator["factor_authority_path"],
        "factor_authority_source_sha256": factor_source_sha,
        "contract_path": live_generator["contract_path"],
        "contract_source_sha256": live_generator["contract_source_sha256"],
        "pinned_remote_refs": live_generator["pinned_remote_refs"],
    }
    generator_revision_sha = semantic_sha256(generator_revision)
    receipts: dict[str, dict[str, Any]] = {}
    by_family = {value["family"]: value for value in candidate["families"].values()}
    for family in ("social", "product", "logo_ui"):
        discovery_identity = [
            {
                "row_id": row["row_id"],
                "row_sha256": row["row_record_sha256"],
            }
            for row in by_family[family]["discovery"]["rows"]
        ]
        body = {
            "schema": SCHEMA,
            "kind": ADMISSION_KIND,
            "status": "PASS",
            "family": family,
            "counts": counts[family],
            "packs": _public_pack_receipts(by_family[family]),
            "candidate_semantic_sha256": candidate["semantic_sha256"],
            "human_review_sha256": sealed_review["review_sha256"],
            "replay_evidence_sha256": replay_sha,
            "dedup_evidence_sha256": dedup["semantic_sha256"],
            "ownership_record_sha256": candidate["rights_record"]["semantic_sha256"],
            "generator_repository": live_generator["repository"],
            "generator_commit": generator_revision["commit"],
            "generator_tree": generator_revision["tree"],
            "generator_source_path": generator_revision["renderer_source_path"],
            "generator_source_sha256": live_generator["renderer_source_sha256"],
            "admission_authority_path": live_generator["admission_authority_path"],
            "admission_authority_source_sha256": authority_source_sha,
            "contract_path": live_generator["contract_path"],
            "contract_source_sha256": live_generator["contract_source_sha256"],
            "discovery_key_commitment_sha256": phase_commitments["discovery"],
            "confirmation_key_commitment_sha256": phase_commitments["confirmation"],
            "confirmation_commitment_sha256": by_family[family]["confirmation"][
                "semantic_commitment_sha256"
            ],
            "discovery_all_row_identity_sha256": semantic_sha256(discovery_identity),
            "generator_revision_sha256": generator_revision_sha,
            "governance": {
                "operator_attested_named_human_review": True,
                "agent_review_is_not_human_review": True,
                "admission_authorized": True,
                "gpu_execution_authorized": False,
                "owner_ratification_required_for_gpu": True,
            },
            "claim_limit": (
                "admitted for this procedural instrument only; field and "
                "opponent-relative transfer remain unproven"
            ),
        }
        receipts[family] = {**body, "admission_sha256": semantic_sha256(body)}
    root_body = {
        "schema": SCHEMA,
        "kind": ROOT_ADMISSION_KIND,
        "status": "PASS",
        "candidate_semantic_sha256": candidate["semantic_sha256"],
        "human_review_sha256": sealed_review["review_sha256"],
        "replay_evidence": replay_evidence,
        "replay_evidence_sha256": replay_sha,
        "dedup_evidence_sha256": dedup["semantic_sha256"],
        "ownership_record_sha256": candidate["rights_record"]["semantic_sha256"],
        "generator_revision": generator_revision,
        "generator_revision_sha256": generator_revision_sha,
        "family_admission_sha256": {
            family: receipt["admission_sha256"] for family, receipt in receipts.items()
        },
        "receipts": receipts,
        "authorization": {
            "fixture_admission_authorized": True,
            "gpu_execution_authorized": False,
            "deployment_authorized": False,
        },
    }
    root = {**root_body, "admission_set_sha256": semantic_sha256(root_body)}
    # Keep the named reviewer and private row identities out of all public receipts.
    public_payload = canonical_bytes(root) + b"".join(
        canonical_bytes(receipt) for receipt in receipts.values()
    )
    reviewer_token = json.dumps(review["reviewer_identity"], ensure_ascii=True)[
        1:-1
    ].encode("ascii")
    if reviewer_token in public_payload:
        raise AdmissionError("public admission leaks the private reviewer identity")
    for family in verified["confirmation_manifest"]["families"].values():
        for row in family["rows"]:
            for key in (
                "row_id",
                "relative_image_path",
                "relative_caption_path",
                "image_sha256",
                "caption_sha256",
                "decoded_pixels_sha256",
                "parameters_sha256",
                "group_identity_sha256",
                "row_record_sha256",
            ):
                if str(row[key]).encode("ascii") in public_payload:
                    raise AdmissionError(
                        "public admission leaks confirmation membership"
                    )
    return root, receipts


def _require_sha256(value: Any, label: str) -> str:
    text = str(value or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise AdmissionError(f"{label} must be an exact SHA-256")
    return text


def _validate_admission_set_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AdmissionError("admission set is not an object")
    expected_keys = {
        "schema",
        "kind",
        "status",
        "candidate_semantic_sha256",
        "human_review_sha256",
        "replay_evidence",
        "replay_evidence_sha256",
        "dedup_evidence_sha256",
        "ownership_record_sha256",
        "generator_revision",
        "generator_revision_sha256",
        "family_admission_sha256",
        "receipts",
        "authorization",
        "admission_set_sha256",
    }
    if set(value) != expected_keys:
        raise AdmissionError("admission-set envelope is malformed")
    body = dict(value)
    declared = _require_sha256(body.pop("admission_set_sha256"), "admission set")
    if declared != semantic_sha256(body):
        raise AdmissionError("admission-set digest mismatch")
    if (
        value.get("schema") != SCHEMA
        or value.get("kind") != ROOT_ADMISSION_KIND
        or value.get("status") != "PASS"
        or value.get("authorization")
        != {
            "fixture_admission_authorized": True,
            "gpu_execution_authorized": False,
            "deployment_authorized": False,
        }
    ):
        raise AdmissionError("admission set has not passed with GPU closed")
    revision = value.get("generator_revision")
    if not isinstance(revision, dict) or set(revision) != {
        "repository",
        "commit",
        "tree",
        "renderer_source_path",
        "renderer_source_sha256",
        "admission_authority_path",
        "admission_authority_source_sha256",
        "factor_authority_path",
        "factor_authority_source_sha256",
        "contract_path",
        "contract_source_sha256",
        "pinned_remote_refs",
    }:
        raise AdmissionError("admission generator revision is absent")
    revision_sha = _require_sha256(
        value.get("generator_revision_sha256"), "generator revision"
    )
    if revision_sha != semantic_sha256(revision):
        raise AdmissionError("admission generator revision digest mismatch")
    renderer._git_sha(revision.get("commit"), "admission generator commit")
    renderer._git_sha(revision.get("tree"), "admission generator tree")
    for key in (
        "renderer_source_sha256",
        "admission_authority_source_sha256",
        "factor_authority_source_sha256",
        "contract_source_sha256",
    ):
        _require_sha256(revision.get(key), f"admission generator {key}")
    replay = value.get("replay_evidence")
    if (
        not isinstance(replay, dict)
        or semantic_sha256(replay) != value.get("replay_evidence_sha256")
        or replay.get("candidate_semantic_sha256")
        != value.get("candidate_semantic_sha256")
        or replay.get("contract_path") != revision.get("contract_path")
        or replay.get("contract_source_sha256")
        != revision.get("contract_source_sha256")
    ):
        raise AdmissionError("admission replay evidence binding mismatch")
    receipts = value.get("receipts")
    receipt_hashes = value.get("family_admission_sha256")
    if (
        not isinstance(receipts, dict)
        or not isinstance(receipt_hashes, dict)
        or set(receipts) != {"social", "product", "logo_ui"}
        or set(receipt_hashes) != set(receipts)
    ):
        raise AdmissionError("admission family receipt inventory mismatch")
    for family, receipt in receipts.items():
        if not isinstance(receipt, dict):
            raise AdmissionError(f"{family} admission receipt is malformed")
        receipt_body = dict(receipt)
        receipt_sha = _require_sha256(
            receipt_body.pop("admission_sha256", None), f"{family} admission"
        )
        if (
            receipt_sha != semantic_sha256(receipt_body)
            or receipt_hashes[family] != receipt_sha
            or receipt.get("schema") != SCHEMA
            or receipt.get("kind") != ADMISSION_KIND
            or receipt.get("status") != "PASS"
            or receipt.get("family") != family
            or receipt.get("candidate_semantic_sha256")
            != value.get("candidate_semantic_sha256")
            or receipt.get("human_review_sha256") != value.get("human_review_sha256")
            or receipt.get("generator_revision_sha256") != revision_sha
            or receipt.get("generator_commit") != revision.get("commit")
            or receipt.get("generator_tree") != revision.get("tree")
            or receipt.get("generator_source_path")
            != revision.get("renderer_source_path")
            or receipt.get("generator_source_sha256")
            != revision.get("renderer_source_sha256")
            or receipt.get("admission_authority_path")
            != revision.get("admission_authority_path")
            or receipt.get("admission_authority_source_sha256")
            != revision.get("admission_authority_source_sha256")
            or receipt.get("contract_path") != revision.get("contract_path")
            or receipt.get("contract_source_sha256")
            != revision.get("contract_source_sha256")
        ):
            raise AdmissionError(f"{family} admission receipt binding mismatch")
        packs = receipt.get("packs")
        if not isinstance(packs, dict):
            raise AdmissionError(f"{family} admission pack receipts are absent")
        for pack, pack_record in packs.items():
            if not isinstance(pack_record, dict):
                raise AdmissionError(f"{family} pack {pack} is malformed")
            phase = pack_record.get("phase")
            if phase == "discovery":
                required = {
                    "phase",
                    "row_count",
                    "training_row_count",
                    "evaluation_row_count",
                    "training_row_identity_sha256",
                    "evaluation_row_identity_sha256",
                    "training_inventory",
                    "training_inventory_sha256",
                    "evaluation_inventory",
                    "evaluation_inventory_sha256",
                }
                if set(pack_record) != required:
                    raise AdmissionError(
                        f"{family} discovery pack {pack} receipt is malformed"
                    )
                if (
                    pack_record.get("training_row_count") != 10
                    or pack_record.get("evaluation_row_count") != 8
                    or pack_record.get("row_count") != 18
                ):
                    raise AdmissionError(
                        f"{family} discovery pack {pack} split counts changed"
                    )
                for split in ("training", "evaluation"):
                    inventory = pack_record.get(f"{split}_inventory")
                    if (
                        not isinstance(inventory, dict)
                        or semantic_sha256(
                            {
                                "files": inventory.get("files"),
                                "file_count": inventory.get("file_count"),
                            }
                        )
                        != inventory.get("semantic_sha256")
                        or pack_record.get(f"{split}_inventory_sha256")
                        != inventory.get("semantic_sha256")
                    ):
                        raise AdmissionError(
                            f"{family} discovery pack {pack} {split} inventory mismatch"
                        )
            elif phase == "confirmation":
                if set(pack_record) != {
                    "phase",
                    "row_count",
                    "training_row_count",
                    "evaluation_row_count",
                    "semantic_commitment_sha256",
                }:
                    raise AdmissionError(
                        f"{family} confirmation pack {pack} leaks or is malformed"
                    )
                if (
                    pack_record.get("training_row_count") != 10
                    or pack_record.get("evaluation_row_count") != 8
                    or pack_record.get("row_count") != 18
                ):
                    raise AdmissionError(
                        f"{family} confirmation pack {pack} split counts changed"
                    )
                _require_sha256(
                    pack_record.get("semantic_commitment_sha256"),
                    f"{family} confirmation pack {pack} commitment",
                )
            else:
                raise AdmissionError(f"{family} pack {pack} phase is invalid")
    return dict(value)


def _validate_private_review_for_ratification(
    value: Mapping[str, Any], admission_set: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AdmissionError("sealed human review is not an object")
    body = dict(value)
    declared = _require_sha256(body.pop("review_sha256", None), "human review")
    reviewer = _named_human(body.get("reviewer_identity"))
    if (
        declared != semantic_sha256(body)
        or value.get("schema") != SCHEMA
        or value.get("kind") != REVIEW_KIND
        or value.get("status") != "SEALED_OPERATOR_ATTESTED_NAMED_HUMAN_PASS"
        or value.get("decision") != "PASS"
        or admission_set.get("human_review_sha256") != declared
    ):
        raise AdmissionError("sealed human review does not bind the admission set")
    return {**dict(value), "reviewer_identity": reviewer}


def build_owner_ratification_template(
    *, admission_set: Mapping[str, Any], sealed_review: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a private owner-completion template; no authority is auto-filled."""

    checked_set = _validate_admission_set_envelope(admission_set)
    review = _validate_private_review_for_ratification(sealed_review, checked_set)
    return {
        "schema": SCHEMA,
        "kind": RATIFICATION_KIND,
        "status": "PENDING_NAMED_OWNER_RATIFICATION",
        "admission_set_sha256": checked_set["admission_set_sha256"],
        "human_review_sha256": review["review_sha256"],
        "reviewer_identity": review["reviewer_identity"],
        "generator_revision": checked_set["generator_revision"],
        "generator_revision_sha256": checked_set["generator_revision_sha256"],
        "owner_identity": "",
        "ratified_at_utc": "",
        "decision": "PENDING",
        "governance": {
            "operator_attested_not_cryptographically_authenticated": True,
            "agent_cannot_complete_owner_fields": True,
            "gpu_execution_authorized": False,
            "deployment_authorized": False,
            "later_plan_must_consume_exact_ratification": True,
        },
    }


def seal_owner_ratification(
    *,
    draft: Mapping[str, Any],
    admission_set: Mapping[str, Any],
    sealed_review: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_owner_ratification_template(
        admission_set=admission_set, sealed_review=sealed_review
    )
    if not isinstance(draft, dict):
        raise AdmissionError("owner ratification draft is not an object")
    if draft.get("status") != "PENDING_NAMED_OWNER_RATIFICATION":
        raise AdmissionError("owner ratification must begin from the pending template")
    immutable = {
        key: value
        for key, value in expected.items()
        if key not in {"status", "owner_identity", "ratified_at_utc", "decision"}
    }
    if any(draft.get(key) != value for key, value in immutable.items()):
        raise AdmissionError("owner ratification binding was changed")
    owner = _named_human(draft.get("owner_identity"))
    ratified_at = _utc(draft.get("ratified_at_utc"))
    if draft.get("decision") != "RATIFY_FOR_PLAN_CONSUMPTION":
        raise AdmissionError("owner ratification decision is not explicit")
    body = {
        **immutable,
        "status": "SEALED_OPERATOR_ATTESTED_OWNER_RATIFICATION",
        "owner_identity": owner,
        "ratified_at_utc": ratified_at,
        "decision": "RATIFY_FOR_PLAN_CONSUMPTION",
    }
    return {**body, "owner_ratification_sha256": semantic_sha256(body)}


def validate_owner_ratification(
    value: Mapping[str, Any],
    *,
    admission_set: Mapping[str, Any],
    sealed_review: Mapping[str, Any],
) -> dict[str, Any]:
    body = dict(value)
    declared = _require_sha256(
        body.pop("owner_ratification_sha256", None), "owner ratification"
    )
    if declared != semantic_sha256(body):
        raise AdmissionError("owner ratification digest mismatch")
    draft = dict(body)
    draft["status"] = "PENDING_NAMED_OWNER_RATIFICATION"
    expected = seal_owner_ratification(
        draft=draft,
        admission_set=admission_set,
        sealed_review=sealed_review,
    )
    if expected != dict(value):
        raise AdmissionError("owner ratification does not reproduce")
    return dict(value)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdmissionError(f"{label} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise AdmissionError(f"{label} must be finite")
    return result


def _validate_candidate_identity(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "arm",
        "source_plan_sha256",
        "source_cell_sha256",
        "bundle_id",
        "bundle_sha256",
        "runtime_commit",
        "generated_config",
        "generated_config_sha256",
        "checkpoint_step",
        "artifact_sha256",
        "seed",
    }:
        raise AdmissionError(f"{label} candidate identity is malformed")
    if not isinstance(value["arm"], str) or not value["arm"]:
        raise AdmissionError(f"{label} candidate arm is absent")
    if not isinstance(value["bundle_id"], str) or not value["bundle_id"]:
        raise AdmissionError(f"{label} candidate bundle is absent")
    for key in (
        "source_plan_sha256",
        "source_cell_sha256",
        "bundle_sha256",
        "generated_config_sha256",
        "artifact_sha256",
    ):
        _require_sha256(value.get(key), f"{label} candidate {key}")
    if (
        not isinstance(value.get("generated_config"), Mapping)
        or semantic_sha256(value["generated_config"])
        != value["generated_config_sha256"]
    ):
        raise AdmissionError(f"{label} candidate config binding mismatch")
    renderer._git_sha(value.get("runtime_commit"), f"{label} runtime commit")
    for key in ("checkpoint_step", "seed"):
        if isinstance(value.get(key), bool) or not isinstance(value.get(key), int):
            raise AdmissionError(f"{label} candidate {key} must be an integer")
    if value["checkpoint_step"] <= 0:
        raise AdmissionError(f"{label} candidate checkpoint step must be positive")
    return dict(value)


def _validate_confirmation_commitments(value: Any) -> dict[str, dict[str, str]]:
    expected = {
        "social": {"C1", "C2"},
        "product": {"C1"},
        "logo_ui": {"C1"},
    }
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise AdmissionError("confirmation commitment inventory is malformed")
    result: dict[str, dict[str, str]] = {}
    for family, packs in expected.items():
        family_value = value.get(family)
        if not isinstance(family_value, Mapping) or set(family_value) != packs:
            raise AdmissionError(
                f"confirmation commitment inventory is malformed for {family}"
            )
        result[family] = {
            pack: _require_sha256(
                family_value.get(pack), f"{family}/{pack} confirmation commitment"
            )
            for pack in sorted(packs)
        }
    return result


def _load_factor_authority() -> types.ModuleType:
    """Load the exact committed experiment authority, never ambient bytes."""

    module, _source_sha256 = _load_committed_source_module(
        repo_root=REPO_ROOT,
        source_path=FACTOR_AUTHORITY_SOURCE_PATH,
        ambient_path=FACTOR_AUTHORITY_PATH,
        module_name="week7_hke_factor_authority_for_admission",
    )
    return module


def _reproduce_confirmation_freeze(value: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute a freeze from its embedded D2 plan/evidence under HEAD code."""

    factor = _load_factor_authority()
    try:
        expected = factor.freeze_confirmation_candidate(
            value.get("source_d2_plan"),
            value.get("source_d2_evidence"),
            frozen_at_utc=str(value.get("frozen_at_utc", "")),
        )
        factor._validate_confirmation_freeze(
            expected,
            d2_plan=expected["source_d2_plan"],
        )
    except Exception as exc:
        raise AdmissionError(
            "confirmation freeze source chain does not validate under the "
            "committed factor authority"
        ) from exc
    if not hmac.compare_digest(canonical_bytes(value), canonical_bytes(expected)):
        raise AdmissionError(
            "confirmation freeze does not reproduce from its embedded source chain"
        )
    return dict(expected)


def _reproduce_c2_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the C2 decision from embedded C1 plan/evidence under HEAD."""

    factor = _load_factor_authority()
    try:
        expected = factor.build_c2_reveal_authority(
            value.get("source_confirmation_plan"),
            value.get("source_confirmation_evidence"),
        )
    except Exception as exc:
        raise AdmissionError(
            "C2 source chain does not validate under the committed factor authority"
        ) from exc
    if not hmac.compare_digest(canonical_bytes(value), canonical_bytes(expected)):
        raise AdmissionError(
            "C2 authority does not reproduce from its embedded source chain"
        )
    return dict(expected)


def validate_confirmation_freeze(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the post-D2 freeze which alone may authorize C1 disclosure."""

    if not isinstance(value, dict):
        raise AdmissionError("confirmation freeze is not an object")
    expected_keys = {
        "schema",
        "kind",
        "status",
        "source_d2_plan",
        "source_d2_plan_sha256",
        "source_d2_evidence",
        "d1_frozen_candidate_sha256",
        "d2_evidence_sha256",
        "candidate",
        "discovery_decision",
        "d2_comparisons",
        "both_finalist_seeds_agree",
        "confirmation_commitments",
        "confirmation_reveal_authorized",
        "frozen_at_utc",
        "checkpoint_promotion_authorized",
        "deployment_authorized",
        "confirmation_freeze_sha256",
    }
    if set(value) != expected_keys:
        raise AdmissionError("confirmation-freeze envelope is malformed")
    body = dict(value)
    declared = _require_sha256(
        body.pop("confirmation_freeze_sha256"), "confirmation freeze"
    )
    if declared != semantic_sha256(body):
        raise AdmissionError("confirmation-freeze digest mismatch")
    reproduced = _reproduce_confirmation_freeze(value)
    if reproduced != dict(value):  # defensive; canonical equality is checked above
        raise AdmissionError("confirmation freeze reproduction changed its value")
    if (
        value.get("schema") != SCHEMA
        or value.get("kind") != CONFIRMATION_FREEZE_KIND
        or value.get("status") != "FROZEN_BEFORE_CONFIRMATION_REVEAL"
        or value.get("confirmation_reveal_authorized") is not True
        or value.get("both_finalist_seeds_agree") is not True
        or value.get("checkpoint_promotion_authorized") is not False
        or value.get("deployment_authorized") is not False
    ):
        raise AdmissionError(
            "candidate has not cleared the post-D2 confirmation freeze"
        )
    for key in (
        "source_d2_plan_sha256",
        "d1_frozen_candidate_sha256",
        "d2_evidence_sha256",
    ):
        _require_sha256(value.get(key), f"confirmation freeze {key}")
    _utc(value.get("frozen_at_utc"))
    _validate_candidate_identity(value.get("candidate"), "confirmation freeze")
    decision = value.get("discovery_decision")
    if (
        not isinstance(decision, Mapping)
        or decision.get("decision") != "ADVANCE"
        or decision.get("confirmation_reveal_authorized") is not True
    ):
        raise AdmissionError("discovery decision does not authorize confirmation")
    comparisons = value.get("d2_comparisons")
    if not isinstance(comparisons, Mapping) or set(comparisons) != {
        "Seed-A",
        "Seed-B",
    }:
        raise AdmissionError("both D2 seed comparisons are required")
    _validate_confirmation_commitments(value.get("confirmation_commitments"))
    return dict(value)


def validate_c2_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the separately predeclared borderline-C1 C2 trigger."""

    if not isinstance(value, dict):
        raise AdmissionError("C2 authority is not an object")
    fields = {
        "schema",
        "kind",
        "status",
        "source_confirmation_plan",
        "source_confirmation_evidence",
        "source_confirmation_freeze_sha256",
        "c1_evidence_sha256",
        "c1_comparison",
        "trigger",
        "c2_commitment_sha256",
        "c2_reveal_authorized",
        "checkpoint_promotion_authorized",
        "deployment_authorized",
        "c2_authority_sha256",
    }
    if set(value) != fields:
        raise AdmissionError("C2 authority envelope is malformed")
    body = dict(value)
    declared = _require_sha256(body.pop("c2_authority_sha256"), "C2 authority")
    if declared != semantic_sha256(body):
        raise AdmissionError("C2 authority digest mismatch")
    reproduced = _reproduce_c2_authority(value)
    if reproduced != dict(value):  # defensive; canonical equality is checked above
        raise AdmissionError("C2 authority reproduction changed its value")
    if (
        value.get("schema") != SCHEMA
        or value.get("kind") != C2_AUTHORITY_KIND
        or value.get("status") != "C2_REVEAL_AUTHORIZED_BY_BORDERLINE_C1"
        or value.get("c2_reveal_authorized") is not True
        or value.get("checkpoint_promotion_authorized") is not False
        or value.get("deployment_authorized") is not False
    ):
        raise AdmissionError("C2 authority does not authorize reveal")
    for key in (
        "source_confirmation_freeze_sha256",
        "c1_evidence_sha256",
        "c2_commitment_sha256",
    ):
        _require_sha256(value.get(key), f"C2 authority {key}")
    trigger = value.get("trigger")
    if not isinstance(trigger, Mapping) or trigger.get("decision") != "REVEAL_C2":
        raise AdmissionError("C2 borderline trigger is not satisfied")
    return dict(value)


def _confirmation_family_record(
    verified: Mapping[str, Any], family: str
) -> tuple[str, Mapping[str, Any]]:
    matches = [
        (fixture_id, record)
        for fixture_id, record in verified["candidate_manifest"]["families"].items()
        if record["family"] == family
    ]
    if len(matches) != 1:
        raise AdmissionError("confirmation family is not unique")
    return matches[0]


def build_confirmation_reveal(
    *,
    public_root: Path,
    custodian_root: Path,
    public_boundary_roots: Sequence[Path],
    admission_set: Mapping[str, Any],
    family: str,
    pack: str,
    confirmation_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Reveal C1 only after D2 freeze, or C2 after its borderline trigger."""

    checked_set = _validate_admission_set_envelope(admission_set)
    if pack == "C1":
        authority = validate_confirmation_freeze(confirmation_authority)
        commitments = authority["confirmation_commitments"]
        authority_sha256 = authority["confirmation_freeze_sha256"]
        authority_kind = CONFIRMATION_FREEZE_KIND
        expected_commitment = commitments.get(family, {}).get(pack)
        status = "PRIVATE_C1_REVEALED_AFTER_D2_CONFIRMATION_FREEZE"
    elif family == "social" and pack == "C2":
        authority = validate_c2_authority(confirmation_authority)
        authority_sha256 = authority["c2_authority_sha256"]
        authority_kind = C2_AUTHORITY_KIND
        expected_commitment = authority["c2_commitment_sha256"]
        status = "PRIVATE_C2_REVEALED_AFTER_BORDERLINE_C1_TRIGGER"
    else:
        raise AdmissionError("confirmation reveal is outside the predeclared protocol")
    verified = _verify_candidate(public_root, custodian_root, public_boundary_roots)
    candidate = verified["candidate_manifest"]
    if checked_set["candidate_semantic_sha256"] != candidate["semantic_sha256"]:
        raise AdmissionError("confirmation candidate disagrees with admission")
    fixture_id, public_family = _confirmation_family_record(verified, family)
    private_family = verified["confirmation_manifest"]["families"][fixture_id]
    private_pack = private_family["packs"].get(pack)
    public_pack = public_family["confirmation"]["packs"].get(pack)
    receipt = checked_set["receipts"].get(family)
    receipt_pack = (
        receipt.get("packs", {}).get(pack) if isinstance(receipt, dict) else None
    )
    if (
        not isinstance(private_pack, dict)
        or not isinstance(public_pack, dict)
        or not isinstance(receipt_pack, dict)
    ):
        raise AdmissionError("confirmation pack is unavailable")
    commitment = public_pack["semantic_commitment_sha256"]
    if (
        commitment != renderer.semantic_sha256(private_pack["rows"])
        or receipt_pack.get("semantic_commitment_sha256") != commitment
        or expected_commitment != commitment
    ):
        raise AdmissionError("confirmation authority commitment mismatch")
    training_rows = private_pack["splits"]["training"]["rows"]
    evaluation_rows = private_pack["splits"]["evaluation"]["rows"]
    training_identity = [
        {"row_id": row["row_id"], "row_sha256": row["row_record_sha256"]}
        for row in training_rows
    ]
    evaluation_identity = [
        {"row_id": row["row_id"], "row_sha256": row["row_record_sha256"]}
        for row in evaluation_rows
    ]
    training_inventory = _split_inventory(training_rows)
    evaluation_inventory = _split_inventory(evaluation_rows)
    _assert_disjoint_split_rows(
        training_rows, evaluation_rows, label=f"confirmation pack {pack}"
    )
    if training_inventory["semantic_sha256"] == evaluation_inventory["semantic_sha256"]:
        raise AdmissionError("confirmation training/evaluation inventories are equal")
    body = {
        "schema": SCHEMA,
        "kind": CONFIRMATION_REVEAL_KIND,
        "status": status,
        "privacy": "custodian_private_not_public_admission",
        "family": family,
        "pack": pack,
        "candidate_semantic_sha256": candidate["semantic_sha256"],
        "admission_set_sha256": checked_set["admission_set_sha256"],
        "family_admission_sha256": receipt["admission_sha256"],
        "confirmation_commitment_sha256": commitment,
        "confirmation_authority_kind": authority_kind,
        "confirmation_authority_sha256": authority_sha256,
        # C1/C2 is private until this authorized reveal.  Once revealed, carry
        # the complete committed row records so downstream experiment
        # validators can reproduce the sealed commitment instead of trusting
        # a parallel, self-declared inventory.
        "revealed_rows": [dict(row) for row in private_pack["rows"]],
        "training_row_identity_sha256": semantic_sha256(training_identity),
        "evaluation_row_identity_sha256": semantic_sha256(evaluation_identity),
        "training_inventory": training_inventory,
        "training_inventory_sha256": training_inventory["semantic_sha256"],
        "evaluation_inventory": evaluation_inventory,
        "evaluation_inventory_sha256": evaluation_inventory["semantic_sha256"],
        "authorization": {
            "confirmation_revealed": True,
            "gpu_execution_authorized": False,
            "deployment_authorized": False,
            "candidate_selection_locked": True,
        },
    }
    return {**body, "confirmation_reveal_sha256": semantic_sha256(body)}


def validate_confirmation_reveal(
    value: Mapping[str, Any],
    *,
    public_root: Path,
    custodian_root: Path,
    public_boundary_roots: Sequence[Path],
    admission_set: Mapping[str, Any],
    confirmation_authority: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AdmissionError("confirmation reveal is not an object")
    expected = build_confirmation_reveal(
        public_root=public_root,
        custodian_root=custodian_root,
        public_boundary_roots=public_boundary_roots,
        admission_set=admission_set,
        family=str(value.get("family", "")),
        pack=str(value.get("pack", "")),
        confirmation_authority=confirmation_authority,
    )
    if not hmac.compare_digest(canonical_bytes(value), canonical_bytes(expected)):
        raise AdmissionError("confirmation reveal does not reproduce")
    return dict(value)


def _read_key(path: Path, label: str) -> bytes:
    return renderer._read_key_file(Path(path), label)


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("review-template")
    seal = sub.add_parser("seal-review")
    admit = sub.add_parser("admit")
    ratification_template = sub.add_parser("ratification-template")
    seal_ratification = sub.add_parser("seal-ratification")
    reveal_confirmation = sub.add_parser("reveal-confirmation")
    for item in (
        template,
        seal,
        admit,
        ratification_template,
        seal_ratification,
        reveal_confirmation,
    ):
        item.add_argument("--public-root", type=Path, required=True)
        item.add_argument("--custodian-root", type=Path, required=True)
        item.add_argument(
            "--public-boundary-root",
            dest="public_boundary_roots",
            action="append",
            type=Path,
            required=True,
            help="public upload/evidence boundary; repeat for every boundary",
        )
    template.add_argument("--output", type=Path, required=True)
    seal.add_argument("--draft", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    admit.add_argument("--sealed-review", type=Path, required=True)
    admit.add_argument("--discovery-key-file", type=Path, required=True)
    admit.add_argument("--confirmation-key-file", type=Path, required=True)
    admit.add_argument("--output-root", type=Path, required=True)
    for item in (ratification_template, seal_ratification):
        item.add_argument("--admission-set", type=Path, required=True)
        item.add_argument("--sealed-review", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
    seal_ratification.add_argument("--draft", type=Path, required=True)
    reveal_confirmation.add_argument("--admission-set", type=Path, required=True)
    reveal_confirmation.add_argument(
        "--confirmation-authority", type=Path, required=True
    )
    reveal_confirmation.add_argument(
        "--family", choices=("social", "product", "logo_ui"), required=True
    )
    reveal_confirmation.add_argument("--pack", choices=("C1", "C2"), required=True)
    reveal_confirmation.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    private_scope = {
        "public_root": args.public_root,
        "custodian_root": args.custodian_root,
        "public_boundary_roots": args.public_boundary_roots,
    }
    if args.command == "review-template":
        value = build_review_template(
            args.public_root,
            args.custodian_root,
            public_boundary_roots=args.public_boundary_roots,
        )
        _write_private_new(
            args.output, value, label="human review template", **private_scope
        )
        return 0
    if args.command == "seal-review":
        draft = _load_private_json(
            args.draft,
            label="human review draft",
            **private_scope,
        )
        value = seal_review(
            public_root=args.public_root,
            custodian_root=args.custodian_root,
            public_boundary_roots=args.public_boundary_roots,
            draft=draft,
        )
        _write_private_new(
            args.output, value, label="sealed human review", **private_scope
        )
        return 0
    if args.command in {"ratification-template", "seal-ratification"}:
        sealed = _load_private_json(
            args.sealed_review,
            label="sealed human review",
            **private_scope,
        )
        admission_set = _load_json(args.admission_set, "admission set")
        if args.command == "ratification-template":
            value = build_owner_ratification_template(
                admission_set=admission_set, sealed_review=sealed
            )
        else:
            draft = _load_private_json(
                args.draft,
                label="owner ratification draft",
                **private_scope,
            )
            value = seal_owner_ratification(
                draft=draft,
                admission_set=admission_set,
                sealed_review=sealed,
            )
        _write_private_new(
            args.output, value, label="owner ratification", **private_scope
        )
        return 0
    if args.command == "reveal-confirmation":
        authority = _load_private_json(
            args.confirmation_authority,
            label="confirmation authority",
            **private_scope,
        )
        value = build_confirmation_reveal(
            public_root=args.public_root,
            custodian_root=args.custodian_root,
            public_boundary_roots=args.public_boundary_roots,
            admission_set=_load_json(args.admission_set, "admission set"),
            family=args.family,
            pack=args.pack,
            confirmation_authority=authority,
        )
        _write_private_new(
            args.output, value, label="confirmation reveal", **private_scope
        )
        return 0
    sealed = _load_private_json(
        args.sealed_review,
        label="sealed human review",
        **private_scope,
    )
    root, receipts = build_admissions(
        public_root=args.public_root,
        custodian_root=args.custodian_root,
        discovery_key=_read_key(args.discovery_key_file, "discovery key"),
        confirmation_key=_read_key(args.confirmation_key_file, "confirmation key"),
        public_boundary_roots=args.public_boundary_roots,
        sealed_review=sealed,
    )
    if renderer._paths_overlap(
        args.output_root, args.public_root
    ) or renderer._paths_overlap(args.output_root, args.custodian_root):
        raise AdmissionError("admission output must be disjoint from candidate trees")
    output_root = renderer._ensure_new_root(args.output_root, "admission output")
    _write_new(output_root / "ADMISSION-SET.json", root)
    for family, receipt in receipts.items():
        _write_new(output_root / f"{family}.json", receipt)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
