#!/usr/bin/env python3
"""Create and replay the strict Krea Stage-2 admission authority chain.

The adapter deliberately does not discover files below the sealed root.  It
accepts a canonical inventory prepared by the fixture custodian, validates all
public inputs, and delegates the first content read to
``krea_confirmation_admission.materialize``.  Its output is an atomic,
create-only directory containing the complete authority bundle consumed by
``krea_stage2_execution``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

try:
    from . import krea_confirmation_admission as admission
    from . import krea_density_seedb_freeze
    from . import krea_stage2_delegated_review_contract as delegated
    from . import krea_stage2_boundary_derivation as boundary_derivation
    from . import krea_stage2_production_identity as production
    from . import krea_stage2_legacy_confirmation as legacy_confirmation
    from . import krea_waiver_finalist_freeze
except ImportError:  # pragma: no cover - direct script execution.
    import krea_confirmation_admission as admission  # type: ignore[no-redef]
    import krea_density_seedb_freeze  # type: ignore[no-redef]
    import krea_stage2_delegated_review_contract as delegated  # type: ignore[no-redef]
    import krea_stage2_boundary_derivation as boundary_derivation  # type: ignore[no-redef]
    import krea_stage2_production_identity as production  # type: ignore[no-redef]
    import krea_stage2_legacy_confirmation as legacy_confirmation  # type: ignore[no-redef]
    import krea_waiver_finalist_freeze  # type: ignore[no-redef]


SCHEMA = 1
SPEC_KIND = "forge-krea-stage2-admission-chain-spec"
INVENTORY_KIND = admission.POSTFREEZE_INVENTORY_KIND
LAYOUT_KIND = "forge-krea-stage2-sealed-inventory-layout"
BOUNDARY_DERIVATION_KIND = boundary_derivation.KIND
DEVIATION_KIND = "forge-krea-stage2-sealed-metadata-access-deviation"
RECEIPT_KIND = "forge-krea-stage2-admission-chain-receipt"
PRIOR_OWNER_KIND = "forge-krea-sole-human-owner-ratification"
OWNER_IDENTITY = admission.OWNER_IDENTITY
_ROLES = admission._ALL_ROLES
_CONFIRMATION_ROLES = admission._CONFIRMATION_ROLES
_BOUNDARY_ROLES = admission._BOUNDARY_ROLES
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHA1 = re.compile(r"[0-9a-f]{40}")
_PUBLIC_FREEZE_COMMIT = "f8d71ac1d0fcbab9dccf7f5a5a5f904f9f90b237"
_PUBLIC_FREEZE_BINDING_FILE_SHA256 = (
    "b0fe9af433e0bc76aaf6cace5356efa5824eee0feb5f629c74527fa05ffd3c2a"
)
_PUBLIC_FREEZE_BINDING_SHA256 = (
    "b6fffaa8d00f94831cd1fef37e3babbd45b0168f2a43df05dc69f323a7a6e561"
)
_PUBLIC_FREEZE_REPOSITORY = "https://github.com/tuly1/sn56-forge-toolkit.git"
_PUBLIC_FREEZE_REF = "refs/heads/codex/week5-krea-stage2-bridge"
_PUBLIC_FREEZE_BINDING_PATH = (
    Path(__file__).resolve().parent
    / "week5"
    / "krea-density-seedb-finalist-freeze-public-binding-2026-08-01.json"
)
_CONFIRMATION_TRANSFER_BINDING = {
    "transport_tar_sha256": (
        "126b794eddf8ca3334cab3dadd6460df0d37043bb5aee9ea008edf4c64f6c304"
    ),
    "source_and_campaign_file_count": 278,
    "relative_path_and_content_merkle_sha256": (
        "9fe17500cf2de9085d5f4af8fe2b068d9082b3c03f6f356d27b5f63fbdb20526"
    ),
    "source_and_campaign_copy_match": True,
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def canonical_file_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value) + b"\n").hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = keys - set(value)
    extra = set(value) - keys
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value
    ) is None:
        raise ValueError(f"{label} must be canonical whole-second UTC")
    datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return value


def _utc_value(value: str) -> datetime:
    return datetime.strptime(_utc(value, "timestamp"), "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a normalized relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    return value


def _absolute_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be an absolute normalized path")
    expanded = os.path.abspath(os.path.expanduser(value))
    if expanded != value or not Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError(f"{label} must be an absolute normalized path")
    return value


def _reject_symlink_ancestors(path: Path, label: str) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink component: {current}")
        if current == current.parent:
            return
        current = current.parent


def _stable_bytes(path: str | Path, label: str) -> bytes:
    source = Path(os.path.abspath(os.path.expanduser(str(path))))
    _reject_symlink_ancestors(source, label)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise ValueError(f"{label} must be a non-empty regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"{label} changed while read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise RuntimeError(f"{label} length changed while read")
    return raw


def _load_canonical(path: str | Path, label: str) -> tuple[dict[str, Any], str]:
    raw = _stable_bytes(path, label)
    try:
        value = _object(json.loads(raw), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if raw != canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} is not canonical JSON plus one newline")
    return value, hashlib.sha256(raw).hexdigest()


def _binding(value: Any, semantic_key: str, label: str) -> dict[str, str]:
    row = _object(value, label)
    _exact(row, {"path", "file_sha256", semantic_key}, label)
    return {
        "path": _absolute_path(row["path"], f"{label}.path"),
        "file_sha256": _digest(row["file_sha256"], f"{label}.file_sha256"),
        semantic_key: _digest(row[semantic_key], f"{label}.{semantic_key}"),
    }


def _positive_size(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _inventory_file(value: Any, label: str) -> dict[str, Any]:
    row = _object(value, label)
    _exact(row, {"relative_path", "sha256", "bytes"}, label)
    return {
        "relative_path": _relative_path(row["relative_path"], f"{label}.relative_path"),
        "sha256": _digest(row["sha256"], f"{label}.sha256"),
        "bytes": _positive_size(row["bytes"], f"{label}.bytes"),
    }


def validate_layout(value: Any) -> dict[str, Any]:
    """Validate the public file-attribution map without touching the seal."""

    layout = _object(value, "sealed inventory layout")
    _exact(
        layout,
        {
            "schema",
            "kind",
            "sealed_root_locator_sha256",
            "roles",
            "supporting_file_roles",
            "layout_sha256",
        },
        "sealed inventory layout",
    )
    roles = _object(layout["roles"], "sealed inventory layout roles")
    if set(roles) != set(_ROLES):
        raise ValueError("sealed inventory layout must cover exactly all Stage-2 roles")
    normalized_roles: dict[str, Any] = {}
    prefixes: list[PurePosixPath] = []
    designated: set[str] = set()
    for role in _ROLES:
        row = _object(roles[role], f"sealed inventory layout role {role}")
        _exact(
            row,
            {
                "root_prefix",
                "manifest_relative_path",
                "manifest_sha256",
                "archive_relative_path",
            },
            f"sealed inventory layout role {role}",
        )
        prefix = _relative_path(row["root_prefix"], f"sealed inventory {role} prefix")
        manifest = _relative_path(
            row["manifest_relative_path"], f"sealed inventory {role} manifest path"
        )
        archive = _relative_path(
            row["archive_relative_path"], f"sealed inventory {role} archive path"
        )
        manifest_semantic = _digest(
            row["manifest_sha256"], f"sealed inventory {role} manifest semantic"
        )
        prefix_parts = PurePosixPath(prefix).parts
        if (
            PurePosixPath(manifest).parts[: len(prefix_parts)] != prefix_parts
            or PurePosixPath(archive).parts[: len(prefix_parts)] != prefix_parts
            or manifest == archive
            or (
                role in _CONFIRMATION_ROLES
                and not manifest.endswith(f"MANIFEST-{role}.sha256")
            )
            or (role in _BOUNDARY_ROLES and not manifest.endswith(".json"))
            or not archive.endswith(".zip")
        ):
            raise ValueError(f"sealed inventory {role} designated files differ from prefix")
        prefixes.append(PurePosixPath(prefix))
        designated.update({manifest, archive})
        normalized_roles[role] = {
            "root_prefix": prefix,
            "manifest_relative_path": manifest,
            "manifest_sha256": manifest_semantic,
            "archive_relative_path": archive,
        }
    if len(prefixes) != len(set(prefixes)) or any(
        left != right
        and (
            left.parts[: len(right.parts)] == right.parts
            or right.parts[: len(left.parts)] == left.parts
        )
        for left in prefixes
        for right in prefixes
    ):
        raise ValueError("sealed inventory role prefixes overlap")
    support = _object(layout["supporting_file_roles"], "supporting file roles")
    normalized_support: dict[str, str] = {}
    for raw_path, role in support.items():
        path = _relative_path(raw_path, "supporting file path")
        if role not in _ROLES or path in designated:
            raise ValueError("supporting file role/path is invalid")
        if any(
            PurePosixPath(path).parts[: len(prefix.parts)] == prefix.parts
            for prefix in prefixes
        ):
            raise ValueError("supporting file is already covered by a role prefix")
        normalized_support[path] = role
    body = {
        "schema": SCHEMA,
        "kind": LAYOUT_KIND,
        "sealed_root_locator_sha256": _digest(
            layout["sealed_root_locator_sha256"], "sealed layout root locator"
        ),
        "roles": normalized_roles,
        "supporting_file_roles": normalized_support,
    }
    expected = {**body, "layout_sha256": canonical_sha256(body)}
    if layout != expected:
        raise ValueError("sealed inventory layout drifted")
    return layout


def validate_boundary_derivation(value: Any) -> dict[str, Any]:
    """Validate the reviewed schema-3 boundary-derivation set exactly."""

    record = _object(value, "boundary derivation")
    _exact(
        record,
        {
            "schema",
            "kind",
            "created_at_utc",
            "public_freeze_binding",
            "roles",
            "mechanics_only",
            "source_bytes_changed",
            "source_governance_reused_as_boundary_authority",
            "fresh_stage2_owner_ratification_required",
            "admission_authorized",
            "gpu_execution_authorized",
            "claim_limit",
            "derivation_set_sha256",
        },
        "boundary derivation",
    )
    created = _utc(record["created_at_utc"], "boundary derivation time")
    if _utc_value(created) <= _utc_value("2026-08-01T18:19:01Z"):
        raise ValueError("boundary derivation predates pushed finalist freeze")
    expected_freeze = {
        "path": boundary_derivation.FREEZE_BINDING_PATH,
        "file_sha256": boundary_derivation.FREEZE_BINDING_FILE_SHA256,
        "binding_sha256": boundary_derivation.FREEZE_BINDING_SHA256,
        "commit_sha1": boundary_derivation.FREEZE_BINDING_COMMIT,
    }
    if record["public_freeze_binding"] != expected_freeze:
        raise ValueError("boundary derivation freeze binding drifted")
    rows = record["roles"]
    if not isinstance(rows, list) or len(rows) != len(_BOUNDARY_ROLES):
        raise ValueError("boundary derivation must cover all six boundary roles")
    normalized_rows: list[dict[str, str]] = []
    for index, item in enumerate(rows):
        row = _object(item, f"boundary derivation roles[{index}]")
        _exact(
            row,
            {
                "role",
                "source_role",
                "manifest_file_sha256",
                "manifest_sha256",
                "training_archive_sha256",
                "training_dataset_sha256",
                "evaluation_dataset_sha256",
            },
            f"boundary derivation roles[{index}]",
        )
        role = row["role"]
        if role not in _BOUNDARY_ROLES:
            raise ValueError("boundary derivation contains an unknown role")
        if row["source_role"] != boundary_derivation._ROLE_SOURCE[role]:
            raise ValueError("boundary derivation source-role mapping drifted")
        normalized_rows.append(
            {
                "role": role,
                "source_role": row["source_role"],
                **{
                    key: _digest(value, f"boundary derivation {role}.{key}")
                    for key, value in row.items()
                    if key not in {"role", "source_role"}
                },
            }
        )
    if [row["role"] for row in normalized_rows] != sorted(_BOUNDARY_ROLES):
        raise ValueError("boundary derivation roles must be unique and sorted")
    if (
        record["schema"] != boundary_derivation.SCHEMA
        or record["kind"] != boundary_derivation.KIND
        or record["mechanics_only"] is not True
        or record["source_bytes_changed"] is not False
        or record["source_governance_reused_as_boundary_authority"] is not False
        or record["fresh_stage2_owner_ratification_required"] is not True
        or record["admission_authorized"] is not False
        or record["gpu_execution_authorized"] is not False
        or not isinstance(record["claim_limit"], str)
        or not record["claim_limit"]
    ):
        raise ValueError("boundary derivation authority contract drifted")
    body = {key: item for key, item in record.items() if key != "derivation_set_sha256"}
    if (
        record["roles"] != normalized_rows
        or record["derivation_set_sha256"] != canonical_sha256(body)
    ):
        raise ValueError("boundary derivation drifted")
    return record


def _verify_public_freeze_binding() -> dict[str, Any]:
    value, file_sha = _load_canonical(_PUBLIC_FREEZE_BINDING_PATH, "public freeze binding")
    body = {key: item for key, item in value.items() if key != "binding_sha256"}
    if (
        file_sha != _PUBLIC_FREEZE_BINDING_FILE_SHA256
        or value.get("binding_sha256") != _PUBLIC_FREEZE_BINDING_SHA256
        or canonical_sha256(body) != _PUBLIC_FREEZE_BINDING_SHA256
        or value.get("kind") != "forge-krea-density-seedb-finalist-freeze-public-binding"
        or value.get("binding_created_at_utc") != "2026-08-01T18:19:01Z"
        or value.get("chronology_contract")
        != {
            "c1c4_content_read_at_binding": False,
            "c1c4_digest_included": False,
            "full_freeze_exists_before_binding": True,
            "reveal_requires_pushed_binding_commit": True,
        }
    ):
        raise ValueError("public freeze binding drifted")
    return value


def _verify_remote_freeze(*, repository_root: str | Path | None = None) -> None:
    """Prove the pushed freeze is an ancestor of the one live remote head."""

    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root).resolve(strict=True)
    )
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--exit-code", _PUBLIC_FREEZE_REPOSITORY, _PUBLIC_FREEZE_REF],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("pushed freeze binding commit is not remotely reachable") from exc
    rows = [line.split() for line in result.stdout.splitlines() if line.strip()]
    if (
        len(rows) != 1
        or len(rows[0]) != 2
        or rows[0][1] != _PUBLIC_FREEZE_REF
        or _SHA1.fullmatch(rows[0][0]) is None
    ):
        raise ValueError("remote freeze ref did not resolve to exactly one commit")
    remote_head = rows[0][0]
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        subprocess.run(
            ["git", "cat-file", "-e", f"{remote_head}^{{commit}}"],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", _PUBLIC_FREEZE_COMMIT, remote_head],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(
            "pushed freeze commit is not a locally proven ancestor of remote head"
        ) from exc
    if status.stdout:
        raise ValueError("freeze ancestry proof requires a clean local repository")


def _stable_file_identity(path: Path, label: str) -> dict[str, Any]:
    source = Path(os.path.abspath(os.path.expanduser(str(path))))
    _reject_symlink_ancestors(source, label)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise ValueError(f"{label} must be a non-empty regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or size != before.st_size:
        raise RuntimeError(f"{label} changed while hashed")
    return {"sha256": digest.hexdigest(), "bytes": size}


def _copy_file_identity(source: Path, target: Path, label: str) -> dict[str, Any]:
    source = Path(os.path.abspath(source))
    _reject_symlink_ancestors(source, label)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(target, f"{label} destination")
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise ValueError(f"{label} must be a non-empty regular file")
        with target.open("xb") as output:
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        after = os.fstat(source_fd)
    finally:
        os.close(source_fd)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or size != before.st_size:
        raise RuntimeError(f"{label} changed while copied")
    return {"sha256": digest.hexdigest(), "bytes": size}


def capture_inventory(
    *,
    confirmation_source_root: str | Path,
    boundary_source_root: str | Path,
    stage2_root: str | Path,
    layout_path: str | Path,
    boundary_derivation_path: str | Path,
    public_commitment_sha256s: Mapping[str, str],
    output: str | Path,
    captured_at_utc: str,
    remote_verifier: Callable[[], None] = _verify_remote_freeze,
) -> dict[str, Any]:
    """Build a fresh C-wrapper/B-schema3 root, then call the reviewed inventory API."""

    public_binding = _verify_public_freeze_binding()
    remote_verifier()
    captured = _utc(captured_at_utc, "inventory capture time")
    if _utc_value(captured) <= _utc_value(public_binding["binding_created_at_utc"]):
        raise ValueError("sealed inventory capture predates pushed freeze binding")
    commitments = admission._role_digests(
        dict(public_commitment_sha256s),
        roles=_CONFIRMATION_ROLES,
        label="inventory public commitment hashes",
    )
    layout, _ = _load_canonical(layout_path, "sealed inventory layout")
    layout = validate_layout(layout)
    derivation, _ = _load_canonical(boundary_derivation_path, "boundary derivation")
    derivation = validate_boundary_derivation(derivation)
    derivation_rows = {row["role"]: row for row in derivation["roles"]}
    destination = Path(os.path.abspath(os.path.expanduser(str(stage2_root))))
    target = Path(os.path.abspath(os.path.expanduser(str(output))))
    locator = admission.sealed_root_locator_sha256(destination)
    if layout["sealed_root_locator_sha256"] != locator:
        raise ValueError("sealed inventory layout differs from sealed root locator")
    if layout["supporting_file_roles"]:
        raise ValueError("fresh Stage-2 root may not carry old-seal support files")
    for role in _CONFIRMATION_ROLES:
        row = layout["roles"][role]
        if (
            row["manifest_sha256"] != legacy_confirmation.PRIOR_SEMANTIC_SHA256S[role]
            or not row["manifest_relative_path"].endswith(f"MANIFEST-{role}.sha256")
        ):
            raise ValueError("legacy C layout differs from blinded prior evidence")
    for role in _BOUNDARY_ROLES:
        row = layout["roles"][role]
        derived = derivation_rows[role]
        if (
            row["manifest_sha256"] != derived["manifest_sha256"]
            or not row["manifest_relative_path"].endswith("fixture-manifest.json")
        ):
            raise ValueError("boundary layout differs from derivation set")
    _reject_symlink_ancestors(destination, "fresh Stage-2 root")
    _reject_symlink_ancestors(target, "sealed inventory output")
    if os.path.lexists(destination) or os.path.lexists(target):
        raise FileExistsError("fresh Stage-2 root or inventory output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)

    # All public/freeze/layout/derivation/actor checks precede source resolution.
    confirmation_root = admission._resolve_sealed_root(confirmation_source_root)
    boundary_root = admission._resolve_sealed_root(boundary_source_root)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    renamed = False
    try:
        for role in _ROLES:
            source_root = confirmation_root if role in _CONFIRMATION_ROLES else boundary_root
            prefix = layout["roles"][role]["root_prefix"]
            role_source = source_root.joinpath(*PurePosixPath(prefix).parts)
            if role_source.is_symlink() or not role_source.is_dir():
                raise ValueError(f"source fixture subtree is absent: {role}")
            found = False
            for path in sorted(role_source.rglob("*")):
                if path.is_symlink():
                    raise ValueError(f"source fixture contains a symlink: {role}")
                if path.is_dir():
                    continue
                if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
                    raise ValueError(f"source fixture contains a special node: {role}")
                role_relative = path.relative_to(role_source)
                target_path = temporary / role / role_relative
                _copy_file_identity(
                    path,
                    target_path,
                    f"post-freeze custodian copy {role}/{role_relative.as_posix()}",
                )
                found = True
            if not found:
                raise ValueError(f"source fixture subtree is empty: {role}")
        wrappers: dict[str, str] = {}
        for role in _CONFIRMATION_ROLES:
            wrapper = legacy_confirmation.build_wrapper(
                role_root=temporary / role,
                role=role,
                created_at_utc=captured,
            )
            wrappers[role] = canonical_file_sha256(wrapper)
        os.rename(temporary, destination)
        renamed = True
        inventory = admission.materialize_postfreeze_inventory(
            public_freeze_binding_path=_PUBLIC_FREEZE_BINDING_PATH,
            remote_reachable_commit_sha1=_PUBLIC_FREEZE_COMMIT,
            public_commitment_sha256s=commitments,
            confirmation_wrapper_file_sha256s=wrappers,
            boundary_fixture_manifest_sha256s={
                role: derivation_rows[role]["manifest_sha256"]
                for role in _BOUNDARY_ROLES
            },
            boundary_fixture_manifest_file_sha256s={
                role: derivation_rows[role]["manifest_file_sha256"]
                for role in _BOUNDARY_ROLES
            },
            sealed_root=destination,
            output_path=target,
            actor=delegated.actor("confirmation_materialization_reviewer"),
            captured_at_utc=captured,
        )
        return validate_inventory(inventory)
    except BaseException:
        if not renamed:
            shutil.rmtree(temporary, ignore_errors=True)
        elif destination.exists() and not target.exists():
            shutil.rmtree(destination, ignore_errors=True)
        raise


def _validate_retired_custom_inventory(value: Any) -> dict[str, Any]:
    inventory = _object(value, "sealed inventory")
    _exact(
        inventory,
        {
            "schema",
            "kind",
            "sealed_root_locator_sha256",
            "capture_authority",
            "layout_sha256",
            "supporting_file_roles",
            "roles",
            "sealed_tree_file_set_sha256",
            "inventory_sha256",
        },
        "sealed inventory",
    )
    roles = _object(inventory["roles"], "sealed inventory roles")
    if set(roles) != set(_ROLES):
        raise ValueError("sealed inventory must cover exactly all Stage-2 roles")
    authority = _object(inventory["capture_authority"], "inventory capture authority")
    _exact(
        authority,
        {
            "captured_at_utc",
            "actor",
            "public_freeze_repository",
            "public_freeze_ref",
            "public_freeze_commit_sha1",
            "public_freeze_binding_file_sha256",
            "public_freeze_binding_sha256",
            "layout_file_sha256",
            "boundary_derivation_file_sha256",
            "boundary_derivation_sha256",
            "confirmation_source_root_locator_sha256",
            "boundary_source_root_locator_sha256",
            "confirmation_source_transfer",
            "public_commitment_sha256s",
            "fixture_content_emitted",
            "selection_was_frozen_before_capture",
            "fresh_stage2_root_built",
            "old_seal_support_files_excluded",
        },
        "inventory capture authority",
    )
    actor = delegated.validate_actor("confirmation_materialization_reviewer", authority["actor"])
    commitments = admission._role_digests(
        authority["public_commitment_sha256s"],
        roles=_CONFIRMATION_ROLES,
        label="inventory public commitment hashes",
    )
    if (
        authority["public_freeze_repository"] != _PUBLIC_FREEZE_REPOSITORY
        or authority["public_freeze_ref"] != _PUBLIC_FREEZE_REF
        or authority["public_freeze_commit_sha1"] != _PUBLIC_FREEZE_COMMIT
        or authority["public_freeze_binding_file_sha256"]
        != _PUBLIC_FREEZE_BINDING_FILE_SHA256
        or authority["public_freeze_binding_sha256"] != _PUBLIC_FREEZE_BINDING_SHA256
        or _digest(authority["layout_file_sha256"], "inventory layout file")
        != authority["layout_file_sha256"]
        or _digest(
            authority["boundary_derivation_file_sha256"],
            "boundary derivation file",
        )
        != authority["boundary_derivation_file_sha256"]
        or _digest(
            authority["boundary_derivation_sha256"],
            "boundary derivation semantic",
        )
        != authority["boundary_derivation_sha256"]
        or _digest(
            authority["confirmation_source_root_locator_sha256"],
            "confirmation source locator",
        )
        != authority["confirmation_source_root_locator_sha256"]
        or _digest(
            authority["boundary_source_root_locator_sha256"],
            "boundary source locator",
        )
        != authority["boundary_source_root_locator_sha256"]
        or authority["fixture_content_emitted"] is not False
        or authority["selection_was_frozen_before_capture"] is not True
        or authority["fresh_stage2_root_built"] is not True
        or authority["old_seal_support_files_excluded"] is not True
        or authority["confirmation_source_transfer"]
        != _CONFIRMATION_TRANSFER_BINDING
    ):
        raise ValueError("inventory capture authority drifted")
    captured_at = _utc(authority["captured_at_utc"], "inventory capture time")
    if _utc_value(captured_at) <= _utc_value("2026-08-01T18:19:01Z"):
        raise ValueError("sealed inventory capture predates pushed freeze binding")
    normalized_authority = {**authority, "actor": actor, "public_commitment_sha256s": commitments}
    _digest(inventory["layout_sha256"], "inventory layout SHA-256")
    support = _object(inventory["supporting_file_roles"], "inventory support roles")
    normalized_support: dict[str, str] = {}
    for raw_path, role in support.items():
        path = _relative_path(raw_path, "inventory support path")
        if role not in _ROLES:
            raise ValueError("inventory support role is invalid")
        normalized_support[path] = role
    if normalized_support:
        raise ValueError("fresh Stage-2 inventory may not include old-seal support files")
    normalized: dict[str, Any] = {}
    paths: list[str] = []
    for role in _ROLES:
        row = _object(roles[role], f"sealed inventory role {role}")
        _exact(row, {"manifest", "archive", "files"}, f"sealed inventory role {role}")
        manifest = _object(row["manifest"], f"sealed inventory {role} manifest")
        archive = _object(row["archive"], f"sealed inventory {role} archive")
        _exact(
            manifest,
            {"relative_path", "file_sha256", "bytes", "manifest_sha256"},
            f"sealed inventory {role} manifest",
        )
        _exact(
            archive,
            {"relative_path", "sha256", "bytes"},
            f"sealed inventory {role} archive",
        )
        normalized_manifest = {
            "relative_path": _relative_path(
                manifest["relative_path"], f"sealed inventory {role} manifest path"
            ),
            "file_sha256": _digest(
                manifest["file_sha256"], f"sealed inventory {role} manifest file"
            ),
            "bytes": _positive_size(
                manifest["bytes"], f"sealed inventory {role} manifest bytes"
            ),
            "manifest_sha256": _digest(
                manifest["manifest_sha256"],
                f"sealed inventory {role} manifest semantic",
            ),
        }
        normalized_archive = {
            "relative_path": _relative_path(
                archive["relative_path"], f"sealed inventory {role} archive path"
            ),
            "sha256": _digest(
                archive["sha256"], f"sealed inventory {role} archive SHA-256"
            ),
            "bytes": _positive_size(
                archive["bytes"], f"sealed inventory {role} archive bytes"
            ),
        }
        if not normalized_manifest["relative_path"].endswith(".json"):
            raise ValueError(f"sealed inventory {role} manifest must be JSON")
        if not normalized_archive["relative_path"].endswith(".zip"):
            raise ValueError(f"sealed inventory {role} archive must be ZIP")
        files = row["files"]
        if not isinstance(files, list) or not files:
            raise ValueError(f"sealed inventory {role} files must be non-empty")
        normalized_files = [
            _inventory_file(item, f"sealed inventory {role} files[{index}]")
            for index, item in enumerate(files)
        ]
        if normalized_files != sorted(normalized_files, key=lambda item: item["relative_path"]):
            raise ValueError(f"sealed inventory {role} files must be sorted")
        if len({item["relative_path"] for item in normalized_files}) != len(normalized_files):
            raise ValueError(f"sealed inventory {role} files must be unique")
        by_path = {item["relative_path"]: item for item in normalized_files}
        if by_path.get(normalized_manifest["relative_path"]) != {
            "relative_path": normalized_manifest["relative_path"],
            "sha256": normalized_manifest["file_sha256"],
            "bytes": normalized_manifest["bytes"],
        } or by_path.get(normalized_archive["relative_path"]) != normalized_archive:
            raise ValueError(
                f"sealed inventory {role} designated files differ from full file list"
            )
        paths.extend(item["relative_path"] for item in normalized_files)
        normalized[role] = {
            "manifest": normalized_manifest,
            "archive": normalized_archive,
            "files": normalized_files,
        }
    if len(paths) != len(set(paths)):
        raise ValueError("sealed inventory paths must be globally unique")
    if set(normalized_support) - set(paths):
        raise ValueError("inventory supporting-file map references absent files")
    if any(
        path not in normalized_support and "/" not in path for path in paths
    ):
        raise ValueError("top-level sealed files require explicit role attribution")
    if any(
        path in normalized_support
        and normalized_support[path] != role
        for role, row in normalized.items()
        for path in (item["relative_path"] for item in row["files"])
    ):
        raise ValueError("supporting-file attribution differs from its inventory role")
    flattened = [
        {"role": role, **item}
        for role, row in normalized.items()
        for item in row["files"]
    ]
    flattened.sort(key=lambda item: (item["role"], item["relative_path"]))
    body = {
        "schema": SCHEMA,
        "kind": INVENTORY_KIND,
        "sealed_root_locator_sha256": _digest(
            inventory["sealed_root_locator_sha256"], "sealed root locator SHA-256"
        ),
        "capture_authority": normalized_authority,
        "layout_sha256": inventory["layout_sha256"],
        "supporting_file_roles": normalized_support,
        "roles": normalized,
        "sealed_tree_file_set_sha256": canonical_sha256(flattened),
    }
    expected = {**body, "inventory_sha256": canonical_sha256(body)}
    if inventory != expected:
        raise ValueError("sealed inventory drifted")
    return inventory


def _retired_custom_inventory_sealed_files(value: Any) -> list[dict[str, Any]]:
    inventory = _validate_retired_custom_inventory(value)
    rows: list[dict[str, Any]] = []
    for role, item in inventory["roles"].items():
        rows.extend({"role": role, **file_row} for file_row in item["files"])
    return admission._sealed_files(
        sorted(rows, key=lambda row: (row["role"], row["relative_path"]))
    )


def validate_inventory(value: Any) -> dict[str, Any]:
    """Validate only the reviewed post-freeze inventory schema."""

    return admission.validate_postfreeze_inventory(value)


def inventory_sealed_files(value: Any) -> list[dict[str, Any]]:
    inventory = validate_inventory(value)
    return admission._sealed_files(list(inventory["files"]))


def validate_deviation(value: Any) -> dict[str, Any]:
    record = _object(value, "sealed metadata-access deviation")
    _exact(
        record,
        {
            "schema",
            "kind",
            "recorded_at_utc",
            "occurrence_window_utc",
            "actor",
            "frozen_selection_binding",
            "operation",
            "observations",
            "impact",
            "corrective_actions",
            "claim_limit",
            "deviation_sha256",
        },
        "sealed metadata-access deviation",
    )
    window = _object(record["occurrence_window_utc"], "deviation occurrence window")
    _exact(
        window,
        {"after_exclusive", "before_exclusive", "precision"},
        "deviation occurrence window",
    )
    after = _utc(window["after_exclusive"], "deviation window start")
    before = _utc(window["before_exclusive"], "deviation window end")
    if _utc_value(after) >= _utc_value(before):
        raise ValueError("deviation occurrence window is empty")
    if window["precision"] != "bounded-window-exact-command-time-unavailable":
        raise ValueError("deviation occurrence-time precision drifted")
    actor = _object(record["actor"], "deviation actor")
    _exact(actor, {"actor_class", "actor_id"}, "deviation actor")
    if actor != {"actor_class": "agent", "actor_id": "root/metagraph_recovery"}:
        raise ValueError("deviation actor drifted")
    binding = _object(record["frozen_selection_binding"], "deviation freeze binding")
    _exact(
        binding,
        {"public_commit_sha1", "public_binding_file_sha256", "public_binding_sha256"},
        "deviation freeze binding",
    )
    if not isinstance(binding["public_commit_sha1"], str) or re.fullmatch(
        r"[0-9a-f]{40}", binding["public_commit_sha1"]
    ) is None:
        raise ValueError("deviation public commit must be a full SHA-1")
    _digest(binding["public_binding_file_sha256"], "deviation binding file")
    _digest(binding["public_binding_sha256"], "deviation binding semantic")
    operation = _object(record["operation"], "deviation operation")
    _exact(
        operation,
        {"host", "sealed_root", "command_classes", "root_metadata_enumerated"},
        "deviation operation",
    )
    if (
        operation["host"] != "bittensor-ops"
        or operation["sealed_root"] != "/opt/sn56-reviewer-sealed"
        or operation["command_classes"]
        != ["find-maxdepth-metadata-listing", "find-filename-pattern-listing"]
        or operation["root_metadata_enumerated"] is not True
    ):
        raise ValueError("deviation operation drifted")
    observations = _object(record["observations"], "deviation observations")
    _exact(
        observations,
        {
            "paths_sizes_modes_owners_mtimes_observed",
            "file_body_bytes_read",
            "caption_text_read",
            "image_pixels_read",
            "files_copied",
            "sealed_root_mutated",
        },
        "deviation observations",
    )
    if observations != {
        "paths_sizes_modes_owners_mtimes_observed": True,
        "file_body_bytes_read": False,
        "caption_text_read": False,
        "image_pixels_read": False,
        "files_copied": False,
        "sealed_root_mutated": False,
    }:
        raise ValueError("deviation observations drifted")
    impact = _object(record["impact"], "deviation impact")
    _exact(
        impact,
        {
            "strict_pre_materialization_barrier_deviation",
            "occurred_after_finalist_freeze",
            "finalist_selection_contaminated",
            "freeze_rerun_required",
        },
        "deviation impact",
    )
    if impact != {
        "strict_pre_materialization_barrier_deviation": True,
        "occurred_after_finalist_freeze": True,
        "finalist_selection_contaminated": False,
        "freeze_rerun_required": False,
    }:
        raise ValueError("deviation impact classification drifted")
    corrective = record["corrective_actions"]
    if corrective != [
        "access-stopped-and-disclosed-immediately",
        "no-further-sealed-root-access-before-authorized-materialization",
        "bind-this-record-into-stage2-admission-chain-receipt",
    ]:
        raise ValueError("deviation corrective actions drifted")
    _utc(record["recorded_at_utc"], "deviation record time")
    body = {key: item for key, item in record.items() if key != "deviation_sha256"}
    if (
        record["schema"] != SCHEMA
        or record["kind"] != DEVIATION_KIND
        or not isinstance(record["claim_limit"], str)
        or record["claim_limit"] == ""
        or record["deviation_sha256"] != canonical_sha256(body)
    ):
        raise ValueError("sealed metadata-access deviation drifted")
    return record


def validate_prior_owner_authorization(value: Any) -> dict[str, Any]:
    record = _object(value, "prior owner ratification")
    expected_keys = {
        "schema",
        "kind",
        "owner_identity",
        "owner_identity_assurance",
        "ratified_at_utc",
        "portable_ratification_draft",
        "governance_amendment",
        "decision_bindings",
        "acknowledgements",
        "decision",
        "admission_authorized",
        "gpu_execution_authorized",
        "claim_limit",
        "ratification_sha256",
    }
    _exact(record, expected_keys, "prior owner ratification")
    acknowledgements = _object(
        record["acknowledgements"], "prior owner ratification acknowledgements"
    )
    required = {
        "owner_authorizes_mechanical_gpu_approval_after_envelope_and_host_plan_validation": True,
        "stage2_requires_separate_commit_and_fresh_owner_ratification": True,
        "owner_accepts_accountability_for_using_bound_agent_evidence": True,
        "ratification_is_not_a_cryptographic_or_legal_signature": True,
    }
    if any(acknowledgements.get(key) is not expected for key, expected in required.items()):
        raise ValueError("prior owner ratification lacks required Stage-2 acknowledgements")
    body = {key: item for key, item in record.items() if key != "ratification_sha256"}
    if (
        record["schema"] != 1
        or record["kind"] != PRIOR_OWNER_KIND
        or record["owner_identity"] != OWNER_IDENTITY
        or record["owner_identity_assurance"]
        != "interactive-owner-self-attestation-not-cryptographic-or-legal-signature"
        or record["decision"] != "ratified_for_fixture_admission_input"
        or record["admission_authorized"] is not False
        or record["gpu_execution_authorized"] is not False
        or record["ratification_sha256"] != canonical_sha256(body)
    ):
        raise ValueError("prior owner ratification drifted")
    _utc(record["ratified_at_utc"], "prior owner ratification time")
    return record


def _validate_freeze(value: Any) -> dict[str, Any]:
    record = _object(value, "finalist freeze")
    body = {key: item for key, item in record.items() if key != "freeze_sha256"}
    contract = {
        krea_waiver_finalist_freeze.FREEZE_KIND: (
            krea_waiver_finalist_freeze.SCHEMA,
            krea_waiver_finalist_freeze.FALSE_CLAIMS,
            krea_waiver_finalist_freeze.AUTHORITY,
        ),
        krea_density_seedb_freeze.FREEZE_KIND: (
            krea_density_seedb_freeze.SCHEMA,
            krea_density_seedb_freeze.FALSE_CLAIMS,
            krea_density_seedb_freeze.AUTHORITY,
        ),
    }.get(record.get("kind"))
    if (
        contract is None
        or record.get("schema") != contract[0]
        or record.get("freeze_sha256") != canonical_sha256(body)
        or record.get("outcome") != "finalists_frozen"
        or record.get("blockers") != []
        or record.get("claims") != contract[1]
        or record.get("authority") != contract[2]
    ):
        raise ValueError("finalist freeze is not an executable frozen decision")
    finalists = record.get("finalist_family_ids")
    all_rules = record.get("all_family_checkpoint_rules")
    selected_rules = record.get("checkpoint_rules")
    families = {"K0", "K1", "K2", "K3", "K4", "K5"}
    if (
        not isinstance(finalists, list)
        or not finalists
        or len(finalists) != len(set(finalists))
        or "K0" not in finalists
        or any(family not in families for family in finalists)
        or not isinstance(all_rules, dict)
        or set(all_rules) != families
        or not isinstance(selected_rules, dict)
        or set(selected_rules) != set(finalists)
        or any(selected_rules[family] != all_rules[family] for family in finalists)
    ):
        raise ValueError("finalist freeze family/checkpoint rules are invalid")
    return record


def validate_spec(value: Any) -> dict[str, Any]:
    spec = _object(value, "admission-chain spec")
    _exact(
        spec,
        {
            "schema",
            "kind",
            "production_identity",
            "waiver_finalist_freeze",
            "prior_owner_ratification",
            "sealed_inventory",
            "sealed_metadata_deviation",
            "sealed_root",
            "timestamps",
            "spec_sha256",
        },
        "admission-chain spec",
    )
    _binding(
        spec["production_identity"],
        "production_identity_sha256",
        "production identity binding",
    )
    _binding(spec["waiver_finalist_freeze"], "freeze_sha256", "finalist freeze binding")
    _binding(
        spec["prior_owner_ratification"],
        "ratification_sha256",
        "prior owner ratification binding",
    )
    _binding(spec["sealed_inventory"], "inventory_sha256", "sealed inventory binding")
    _binding(spec["sealed_metadata_deviation"], "deviation_sha256", "sealed deviation binding")
    _absolute_path(spec["sealed_root"], "sealed_root")
    timestamps = _object(spec["timestamps"], "admission-chain timestamps")
    _exact(
        timestamps,
        {
            "request_prepared_at_utc",
            "owner_ratified_at_utc",
            "reveal_authorized_at_utc",
            "materialized_at_utc",
            "gpu_authorized_at_utc",
        },
        "admission-chain timestamps",
    )
    timestamp_order = (
        "request_prepared_at_utc",
        "owner_ratified_at_utc",
        "reveal_authorized_at_utc",
        "materialized_at_utc",
        "gpu_authorized_at_utc",
    )
    ordered = [_utc(timestamps[key], key) for key in timestamp_order]
    if any(_utc_value(left) >= _utc_value(right) for left, right in zip(ordered, ordered[1:])):
        raise ValueError("admission-chain timestamps must be strictly increasing")
    body = {key: item for key, item in spec.items() if key != "spec_sha256"}
    if (
        spec["schema"] != SCHEMA
        or spec["kind"] != SPEC_KIND
        or spec["spec_sha256"] != canonical_sha256(body)
    ):
        raise ValueError("admission-chain spec drifted")
    return spec


def _load_bound(
    binding_value: Mapping[str, Any], semantic_key: str, label: str
) -> tuple[dict[str, Any], str]:
    binding = _binding(binding_value, semantic_key, f"{label} binding")
    record, file_sha = _load_canonical(binding["path"], label)
    if file_sha != binding["file_sha256"] or record.get(semantic_key) != binding[semantic_key]:
        raise ValueError(f"{label} differs from its declared binding")
    return record, file_sha


def _write_record(path: Path, value: Any) -> str:
    _reject_symlink_ancestors(path, "admission-chain output")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _materialized_manifest_check(output: Path, inventory: Mapping[str, Any]) -> None:
    rows = inventory_sealed_files(inventory)
    by_role = {
        role: [row for row in rows if row["role"] == role]
        for role in _ROLES
    }
    for role in _CONFIRMATION_ROLES:
        expected = inventory["confirmation_wrapper_file_sha256s"][role]
        matches = [row for row in by_role[role] if row["sha256"] == expected]
        if len(matches) != 1:
            raise ValueError(f"materialized {role} wrapper differs from inventory")
        value, file_sha = _load_canonical(
            output / matches[0]["relative_path"], f"materialized {role} wrapper"
        )
        if (
            file_sha != expected
            or legacy_confirmation.validate_wrapper(value)["experimental_role"] != role
        ):
            raise ValueError(f"materialized {role} wrapper differs from inventory")
    for role in _BOUNDARY_ROLES:
        expected = inventory["boundary_fixture_manifest_file_sha256s"][role]
        matches = [row for row in by_role[role] if row["sha256"] == expected]
        if len(matches) != 1:
            raise ValueError(f"materialized {role} manifest differs from inventory")
        value, file_sha = _load_canonical(
            output / matches[0]["relative_path"], f"materialized {role} manifest"
        )
        if (
            file_sha != expected
            or value.get("manifest_sha256")
            != inventory["boundary_fixture_manifest_sha256s"][role]
        ):
            raise ValueError(f"materialized {role} manifest differs from inventory")


def _receipt(
    *,
    spec: Mapping[str, Any],
    spec_file_sha256: str,
    records: Mapping[str, Mapping[str, Any]],
    bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    body = {
        "schema": SCHEMA,
        "kind": RECEIPT_KIND,
        "spec": {"file_sha256": spec_file_sha256, "spec_sha256": spec["spec_sha256"]},
        "prior_owner_ratification": dict(bindings["prior_owner_ratification"]),
        "sealed_inventory": dict(bindings["sealed_inventory"]),
        "post_freeze_custodian_inventory_capture": {
            "actor": delegated.actor("confirmation_materialization_reviewer"),
            "sealed_fixture_bytes_read_and_copied": True,
            "fixture_content_emitted": False,
            "selection_was_frozen_before_capture": True,
            "confirmation_source_transfer": _CONFIRMATION_TRANSFER_BINDING,
        },
        "sealed_metadata_deviation": dict(bindings["sealed_metadata_deviation"]),
        "waiver_finalist_freeze": dict(bindings["waiver_finalist_freeze"]),
        "production_identity": dict(bindings["production_identity"]),
        "request": {
            "file_sha256": admission.canonical_file_sha256(records["request"]),
            "request_sha256": records["request"]["request_sha256"],
        },
        "fresh_stage2_owner_ratification": {
            "file_sha256": admission.canonical_file_sha256(records["ratification"]),
            "ratification_sha256": records["ratification"]["ratification_sha256"],
        },
        "reveal": {
            "file_sha256": admission.canonical_file_sha256(records["reveal"]),
            "reveal_sha256": records["reveal"]["reveal_sha256"],
            "sealed_content_read": False,
        },
        "materialization": {
            "file_sha256": admission.canonical_file_sha256(records["materialization"]),
            "materialization_sha256": records["materialization"]["materialization_sha256"],
            "sealed_content_read_via": (
                "second-read-of-fresh-inventory-bound-root-by-"
                "krea_confirmation_admission.materialize"
            ),
        },
        "gpu_execution_authorization": {
            "file_sha256": admission.canonical_file_sha256(records["gpu_execution_authorization"]),
            "gpu_execution_authorization_sha256": records[
                "gpu_execution_authorization"
            ]["gpu_execution_authorization_sha256"],
        },
        "authority": {
            "admission_authorized": True,
            "gpu_execution_authorized": True,
            "production_mutation_authorized": False,
            "release_authorized": False,
        },
        "owner_authority_interpretation": (
            "prior owner ratification is provenance for standing mechanical execution; "
            "it does not replace the fresh Stage-2 ratification in the exact chain"
        ),
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _validate_receipt(value: Any) -> dict[str, Any]:
    receipt = _object(value, "admission-chain receipt")
    body = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("kind") != RECEIPT_KIND
        or receipt.get("receipt_sha256") != canonical_sha256(body)
        or receipt.get("authority")
        != {
            "admission_authorized": True,
            "gpu_execution_authorized": True,
            "production_mutation_authorized": False,
            "release_authorized": False,
        }
    ):
        raise ValueError("admission-chain receipt drifted")
    return receipt


def _validate_inputs(spec: Mapping[str, Any]) -> dict[str, Any]:
    identity, identity_file_sha = _load_bound(
        spec["production_identity"], "production_identity_sha256", "production identity"
    )
    identity = production.validate(identity)
    freeze, freeze_file_sha = _load_bound(
        spec["waiver_finalist_freeze"], "freeze_sha256", "finalist freeze"
    )
    freeze = _validate_freeze(freeze)
    prior, prior_file_sha = _load_bound(
        spec["prior_owner_ratification"], "ratification_sha256", "prior owner ratification"
    )
    prior = validate_prior_owner_authorization(prior)
    inventory, inventory_file_sha = _load_bound(
        spec["sealed_inventory"], "inventory_sha256", "sealed inventory"
    )
    inventory = validate_inventory(inventory)
    deviation, deviation_file_sha = _load_bound(
        spec["sealed_metadata_deviation"], "deviation_sha256", "sealed metadata deviation"
    )
    deviation = validate_deviation(deviation)
    locator = admission.sealed_root_locator_sha256(spec["sealed_root"])
    if inventory["sealed_root_locator_sha256"] != locator:
        raise ValueError("sealed inventory locator differs from requested sealed root")
    if _utc_value(spec["timestamps"]["owner_ratified_at_utc"]) <= _utc_value(
        prior["ratified_at_utc"]
    ):
        raise ValueError("fresh Stage-2 ratification must follow prior owner ratification")
    if _utc_value(spec["timestamps"]["request_prepared_at_utc"]) <= _utc_value(
        inventory["captured_at_utc"]
    ):
        raise ValueError("Stage-2 request must follow post-freeze inventory capture")
    return {
        "production_identity": identity,
        "waiver_finalist_freeze": freeze,
        "prior_owner_ratification": prior,
        "sealed_inventory": inventory,
        "sealed_metadata_deviation": deviation,
        "file_sha256s": {
            "production_identity": identity_file_sha,
            "waiver_finalist_freeze": freeze_file_sha,
            "prior_owner_ratification": prior_file_sha,
            "sealed_inventory": inventory_file_sha,
            "sealed_metadata_deviation": deviation_file_sha,
        },
    }


def admit(*, spec_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Validate public evidence, then materialize and publish atomically."""

    spec, spec_file_sha = _load_canonical(spec_path, "admission-chain spec")
    spec = validate_spec(spec)
    resolved = _validate_inputs(spec)
    identity = resolved["production_identity"]
    freeze = resolved["waiver_finalist_freeze"]
    inventory = resolved["sealed_inventory"]
    file_shas = resolved["file_sha256s"]
    timestamps = spec["timestamps"]

    request = admission.build_request(
        production_identity=identity,
        production_identity_file_sha256=file_shas["production_identity"],
        waiver_freeze_sha256=freeze["freeze_sha256"],
        waiver_freeze_file_sha256=file_shas["waiver_finalist_freeze"],
        public_commitment_sha256s={
            role: inventory["public_commitment_sha256s"][role]
            for role in _CONFIRMATION_ROLES
        },
        boundary_fixture_manifest_sha256s={
            role: inventory["boundary_fixture_manifest_sha256s"][role]
            for role in _BOUNDARY_ROLES
        },
        sealed_inventory_sha256=inventory["inventory_sha256"],
        sealed_inventory_file_sha256=file_shas["sealed_inventory"],
        sealed_root_locator_sha256=inventory["sealed_root_locator_sha256"],
        sealed_files=inventory_sealed_files(inventory),
        prepared_at_utc=timestamps["request_prepared_at_utc"],
    )
    ratification = admission.ratify(
        request,
        production_identity=identity,
        production_identity_file_sha256=file_shas["production_identity"],
        sealed_root=spec["sealed_root"],
        owner_identity=OWNER_IDENTITY,
        ratified_at_utc=timestamps["owner_ratified_at_utc"],
    )
    ratification_file_sha = admission.canonical_file_sha256(ratification)
    reveal = admission.authorize_reveal(
        request,
        ratification,
        ratification_file_sha256=ratification_file_sha,
        production_identity=identity,
        production_identity_file_sha256=file_shas["production_identity"],
        sealed_root=spec["sealed_root"],
        actor=delegated.actor("confirmation_reveal_reviewer"),
        revealed_at_utc=timestamps["reveal_authorized_at_utc"],
    )

    output = Path(os.path.abspath(os.path.expanduser(str(output_dir))))
    _reject_symlink_ancestors(output, "admission-chain output")
    if os.path.lexists(output):
        raise FileExistsError(f"admission-chain output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(output, "admission-chain output")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _write_record(temporary / "spec.json", spec)
        for name in (
            "production_identity",
            "waiver_finalist_freeze",
            "prior_owner_ratification",
            "sealed_inventory",
            "sealed_metadata_deviation",
        ):
            _write_record(temporary / f"input-{name.replace('_', '-')}.json", resolved[name])
        admission.publish(request, temporary / "request.json")
        admission.publish(ratification, temporary / "ratification.json")
        admission.publish(reveal, temporary / "reveal.json")
        request_file_sha = admission.canonical_file_sha256(request)
        reveal_file_sha = admission.canonical_file_sha256(reveal)
        materialization = admission.materialize(
            request,
            ratification,
            reveal,
            request_file_sha256=request_file_sha,
            ratification_file_sha256=ratification_file_sha,
            reveal_file_sha256=reveal_file_sha,
            production_identity=identity,
            production_identity_file_sha256=file_shas["production_identity"],
            sealed_root=spec["sealed_root"],
            output_dir=temporary / "materialized",
            actor=delegated.actor("confirmation_materialization_reviewer"),
            materialized_at_utc=timestamps["materialized_at_utc"],
        )
        _materialized_manifest_check(temporary / "materialized", inventory)
        materialization_file_sha = admission.canonical_file_sha256(materialization)
        gpu = admission.build_gpu_execution_authorization(
            request,
            ratification,
            reveal,
            materialization,
            request_file_sha256=request_file_sha,
            ratification_file_sha256=ratification_file_sha,
            reveal_file_sha256=reveal_file_sha,
            materialization_file_sha256=materialization_file_sha,
            production_identity=identity,
            production_identity_file_sha256=file_shas["production_identity"],
            owner_identity=OWNER_IDENTITY,
            authorized_at_utc=timestamps["gpu_authorized_at_utc"],
        )
        admission.publish(gpu, temporary / "gpu-execution-authorization.json")
        authority_bundle = {
            "request": request,
            "request_file_sha256": request_file_sha,
            "ratification": ratification,
            "ratification_file_sha256": ratification_file_sha,
            "reveal": reveal,
            "reveal_file_sha256": reveal_file_sha,
            "materialization": materialization,
            "materialization_file_sha256": materialization_file_sha,
            "gpu_execution_authorization": gpu,
            "gpu_execution_authorization_file_sha256": admission.canonical_file_sha256(gpu),
            "production_identity": identity,
            "production_identity_file_sha256": file_shas["production_identity"],
            "waiver_finalist_freeze": freeze,
            "sealed_inventory": inventory,
            "sealed_inventory_file_sha256": file_shas["sealed_inventory"],
        }
        _write_record(temporary / "authority-bundle.json", authority_bundle)
        bindings = {
            name: dict(spec[name])
            for name in (
                "production_identity",
                "waiver_finalist_freeze",
                "prior_owner_ratification",
                "sealed_inventory",
                "sealed_metadata_deviation",
            )
        }
        receipt = _receipt(
            spec=spec,
            spec_file_sha256=spec_file_sha,
            records={
                "request": request,
                "ratification": ratification,
                "reveal": reveal,
                "materialization": materialization,
                "gpu_execution_authorization": gpu,
            },
            bindings=bindings,
        )
        _write_record(temporary / "admission-chain-receipt.json", receipt)
        replay(temporary)
        os.rename(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return replay(output)


def replay(output_dir: str | Path) -> dict[str, Any]:
    """Rehash and exactly replay a published chain without sealed-root access."""

    root = Path(os.path.abspath(os.path.expanduser(str(output_dir))))
    _reject_symlink_ancestors(root, "admission-chain replay root")
    if not root.is_dir() or not stat.S_ISDIR(root.stat().st_mode):
        raise ValueError("admission-chain replay root must be a directory")
    expected_top_level = {
        "spec.json",
        "input-production-identity.json",
        "input-waiver-finalist-freeze.json",
        "input-prior-owner-ratification.json",
        "input-sealed-inventory.json",
        "input-sealed-metadata-deviation.json",
        "request.json",
        "ratification.json",
        "reveal.json",
        "materialized",
        "gpu-execution-authorization.json",
        "authority-bundle.json",
        "admission-chain-receipt.json",
    }
    if {path.name for path in root.iterdir()} != expected_top_level:
        raise ValueError("admission-chain published file set drifted")
    spec, spec_file_sha = _load_canonical(root / "spec.json", "published spec")
    spec = validate_spec(spec)
    names = {
        "production_identity": ("production_identity_sha256", production.validate),
        "waiver_finalist_freeze": ("freeze_sha256", _validate_freeze),
        "prior_owner_ratification": ("ratification_sha256", validate_prior_owner_authorization),
        "sealed_inventory": ("inventory_sha256", validate_inventory),
        "sealed_metadata_deviation": ("deviation_sha256", validate_deviation),
    }
    inputs: dict[str, Any] = {}
    for name, (semantic_key, validator) in names.items():
        record, file_sha = _load_canonical(
            root / f"input-{name.replace('_', '-')}.json", f"published {name}"
        )
        record = validator(record)
        binding = _binding(spec[name], semantic_key, f"spec {name}")
        if file_sha != binding["file_sha256"] or record[semantic_key] != binding[semantic_key]:
            raise ValueError(f"published {name} differs from spec binding")
        inputs[name] = record
    identity = inputs["production_identity"]
    identity_file_sha = spec["production_identity"]["file_sha256"]
    request = admission.load(root / "request.json")
    ratification = admission.load(root / "ratification.json")
    reveal = admission.load(root / "reveal.json")
    materialization = admission.load(root / "materialized" / "materialization.json")
    gpu = admission.load(root / "gpu-execution-authorization.json")
    request_file_sha = admission.canonical_file_sha256(request)
    ratification_file_sha = admission.canonical_file_sha256(ratification)
    reveal_file_sha = admission.canonical_file_sha256(reveal)
    materialization_file_sha = admission.canonical_file_sha256(materialization)
    admission.validate_gpu_execution_authorization(
        gpu,
        request=request,
        ratification=ratification,
        reveal=reveal,
        materialization=materialization,
        request_file_sha256=request_file_sha,
        ratification_file_sha256=ratification_file_sha,
        reveal_file_sha256=reveal_file_sha,
        materialization_file_sha256=materialization_file_sha,
        production_identity=identity,
        production_identity_file_sha256=identity_file_sha,
    )
    inventory = inputs["sealed_inventory"]
    expected_materialized = {
        row["relative_path"] for row in inventory_sealed_files(inventory)
    } | {"materialization.json"}
    observed_materialized: set[str] = set()
    for path in (root / "materialized").rglob("*"):
        if path.is_symlink():
            raise ValueError("materialized fixture tree contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("materialized fixture tree contains a special node")
        observed_materialized.add(path.relative_to(root / "materialized").as_posix())
    if observed_materialized != expected_materialized:
        raise ValueError("materialized fixture file set differs from inventory")
    _materialized_manifest_check(root / "materialized", inventory)
    for row in inventory_sealed_files(inventory):
        raw = _stable_bytes(
            root / "materialized" / row["relative_path"],
            f"materialized fixture {row['relative_path']}",
        )
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("materialized fixture bytes differ from inventory")
    expected_authority = {
        "request": request,
        "request_file_sha256": request_file_sha,
        "ratification": ratification,
        "ratification_file_sha256": ratification_file_sha,
        "reveal": reveal,
        "reveal_file_sha256": reveal_file_sha,
        "materialization": materialization,
        "materialization_file_sha256": materialization_file_sha,
        "gpu_execution_authorization": gpu,
        "gpu_execution_authorization_file_sha256": admission.canonical_file_sha256(gpu),
        "production_identity": identity,
        "production_identity_file_sha256": identity_file_sha,
        "waiver_finalist_freeze": inputs["waiver_finalist_freeze"],
        "sealed_inventory": inventory,
        "sealed_inventory_file_sha256": spec["sealed_inventory"]["file_sha256"],
    }
    authority, _ = _load_canonical(root / "authority-bundle.json", "authority bundle")
    if authority != expected_authority:
        raise ValueError("authority bundle does not exactly replay")
    bindings = {
        name: dict(spec[name])
        for name in (
            "production_identity",
            "waiver_finalist_freeze",
            "prior_owner_ratification",
            "sealed_inventory",
            "sealed_metadata_deviation",
        )
    }
    expected_receipt = _receipt(
        spec=spec,
        spec_file_sha256=spec_file_sha,
        records={
            "request": request,
            "ratification": ratification,
            "reveal": reveal,
            "materialization": materialization,
            "gpu_execution_authorization": gpu,
        },
        bindings=bindings,
    )
    receipt, _ = _load_canonical(
        root / "admission-chain-receipt.json", "admission-chain receipt"
    )
    _validate_receipt(receipt)
    if receipt != expected_receipt:
        raise ValueError("admission-chain receipt does not exactly replay")
    return receipt


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    admit_parser = commands.add_parser("admit")
    admit_parser.add_argument("--spec", required=True, type=Path)
    admit_parser.add_argument("--output-dir", required=True, type=Path)
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("--output-dir", required=True, type=Path)
    inventory_parser = commands.add_parser("validate-inventory")
    inventory_parser.add_argument("--inventory", required=True, type=Path)
    capture_parser = commands.add_parser("capture-inventory")
    capture_parser.add_argument("--confirmation-source-root", required=True, type=Path)
    capture_parser.add_argument("--boundary-source-root", required=True, type=Path)
    capture_parser.add_argument("--stage2-root", required=True, type=Path)
    capture_parser.add_argument("--layout", required=True, type=Path)
    capture_parser.add_argument("--boundary-derivation", required=True, type=Path)
    capture_parser.add_argument("--public-commitments", required=True, type=Path)
    capture_parser.add_argument("--output", required=True, type=Path)
    capture_parser.add_argument("--captured-at-utc", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    try:
        if args.command == "admit":
            result = admit(spec_path=args.spec, output_dir=args.output_dir)
        elif args.command == "replay":
            result = replay(args.output_dir)
        elif args.command == "validate-inventory":
            result, _ = _load_canonical(args.inventory, "sealed inventory")
            result = validate_inventory(result)
        else:
            commitments, _ = _load_canonical(
                args.public_commitments, "public commitment mapping"
            )
            result = capture_inventory(
                confirmation_source_root=args.confirmation_source_root,
                boundary_source_root=args.boundary_source_root,
                stage2_root=args.stage2_root,
                layout_path=args.layout,
                boundary_derivation_path=args.boundary_derivation,
                public_commitment_sha256s=commitments,
                output=args.output,
                captured_at_utc=args.captured_at_utc,
            )
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
