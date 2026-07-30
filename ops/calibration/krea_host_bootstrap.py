#!/usr/bin/env python3
"""Deterministic, fail-closed host layout for Week-5 Krea GPU evidence.

The module is additive and cannot authorize GPU work.  It binds pre-staged,
reviewed source/runtime directories into the fixed paths used by the timing
runbook, checks the literal host contract, and publishes a create-only receipt.
It never installs dependencies, downloads code, or mutates staged sources.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Any, Mapping, Sequence

try:
    from . import krea_host_identity
    from . import krea_provenance
    from . import krea_stage1_runtime
except ImportError:  # pragma: no cover - direct script execution.
    import krea_host_identity  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_stage1_runtime  # type: ignore[no-redef]


_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_SPEC_KIND = "forge-krea-host-bootstrap-spec"
_RECEIPT_KIND = "forge-krea-host-bootstrap-receipt"
_GIB = 1024**3
_FIXED_TARGETS = {
    "forge_repo": "/app/forge",
    "ai_toolkit_repo": "/app/ai-toolkit",
    "venv": "/app/venv",
    "checkpoints": "/app/checkpoints",
    "dataset": "/dataset",
    "cache": "/cache",
    "campaign": "/campaign",
}
_READ_ONLY_BINDINGS = frozenset({"forge_repo", "ai_toolkit_repo", "venv"})
_RUNTIME_CACHE_ROOT = Path("/cache/krea-runtime")
_RUNTIME_CACHE_POLICY = {
    "root": str(_RUNTIME_CACHE_ROOT),
    "namespace_derivation": (
        "timing_plan_file_sha256_plus_capture_id_or_execution_plan_file_sha256"
    ),
    "initial_state": "root-empty-before-bootstrap",
    "cross_capture_or_plan_reuse": False,
    "within_process_reuse": True,
}
_CAMPAIGN_LEAVES = (
    "controls",
    "controls/admission",
    "evidence",
    "krea-timing",
    "krea-discovery",
)
_CALIBRATION_ARTIFACTS = {
    "timing_tool": "ops/calibration/krea_timing_probe.py",
    "runner": "ops/calibration/run_krea_ladder.py",
    "host_identity_tool": "ops/calibration/krea_host_identity.py",
    "runtime_binding_tool": "ops/calibration/krea_runtime_binding.py",
    "bootstrap_tool": "ops/calibration/krea_host_bootstrap.py",
    "stage1_runtime_tool": "ops/calibration/krea_stage1_runtime.py",
    "execution_surface_policy": "ops/calibration/krea_execution_surface_policy.py",
}
_TRUSTED_EXECUTABLE_PATHS = {
    "docker": "/usr/bin/docker",
    "findmnt": "/usr/bin/findmnt",
    "git": "/usr/bin/git",
    "mount": "/usr/bin/mount",
    "nvidia_container_cli": "/usr/bin/nvidia-container-cli",
    "nvidia_smi": "/usr/bin/nvidia-smi",
    "stat": "/usr/bin/stat",
    "system_python": "/usr/bin/python3",
    "umount": "/usr/bin/umount",
}


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = keys - set(value)
    extra = set(value) - keys
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{label} must be a positive finite number")
    return float(value)


def _absolute(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be canonical absolute text")
    pure = PurePosixPath(value)
    if not pure.is_absolute() or str(pure) != value or ".." in pure.parts:
        raise ValueError(f"{label} must be a normalized absolute Linux path")
    return Path(value)


def _no_symlink_ancestors(path: Path, label: str, *, require_exists: bool) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink component: {current}")
        if current == current.parent:
            break
        current = current.parent
    if require_exists and not path.is_dir():
        raise ValueError(f"{label} must be an existing directory: {path}")
    return path


def _safe_file(path: str | Path, label: str) -> Path:
    result = _no_symlink_ancestors(Path(path), label, require_exists=False)
    if result.is_symlink() or not result.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {result}")
    return result


def _safe_venv_python(venv: Path, label: str) -> tuple[Path, Path]:
    """Allow only a Python leaf symlink whose final target stays in the venv."""

    venv = _no_symlink_ancestors(venv, "staged venv", require_exists=True)
    candidate = venv / "bin/python"
    _no_symlink_ancestors(candidate.parent, label, require_exists=True)
    if not os.path.lexists(candidate):
        raise RuntimeError(f"{label} is absent: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
        venv_resolved = venv.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise RuntimeError(f"{label} has a broken or cyclic symlink") from exc
    if not resolved.is_file() or not resolved.is_relative_to(venv_resolved):
        raise RuntimeError(f"{label} resolves outside the staged venv")
    return candidate, resolved


def _load_canonical(path: str | Path, label: str) -> dict[str, Any]:
    path = _safe_file(path, label)
    raw = path.read_bytes()
    try:
        value = _object(json.loads(raw), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if raw != krea_provenance.canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline")
    return value


def _tree_identity(root: Path, label: str) -> dict[str, Any]:
    """Hash every venv entry without following an escaping symlink."""

    root = _no_symlink_ancestors(root, label, require_exists=True)
    root_resolved = root.resolve(strict=True)
    rows: list[list[Any]] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode & 0o7777
        if path.is_symlink():
            try:
                resolved = path.resolve(strict=True)
            except (FileNotFoundError, RuntimeError) as exc:
                raise RuntimeError(
                    f"{label} contains a broken symlink: {relative}"
                ) from exc
            if not resolved.is_relative_to(root_resolved):
                raise RuntimeError(f"{label} symlink escapes tree: {relative}")
            rows.append(
                [
                    relative,
                    "symlink",
                    mode,
                    os.readlink(path),
                    resolved.relative_to(root_resolved).as_posix(),
                ]
            )
        elif path.is_dir():
            rows.append([relative, "directory", mode])
        elif path.is_file():
            rows.append(
                [
                    relative,
                    "file",
                    mode,
                    path.stat().st_size,
                    krea_provenance.file_sha256(path),
                ]
            )
        else:
            raise RuntimeError(f"{label} contains an unsupported file: {relative}")
    return {
        "entry_count": len(rows),
        "manifest_sha256": krea_provenance.canonical_sha256(rows),
    }


def _stage1_tree_identity(value: Any) -> dict[str, Any]:
    """Translate the materializer's rich tree rows into bootstrap identity rows."""

    tree = _object(value, "Stage-1 venv tree manifest")
    _exact(
        tree,
        {"entries", "entry_count", "entries_sha256", "root"},
        "Stage-1 venv tree manifest",
    )
    entries = tree["entries"]
    if not isinstance(entries, list):
        raise ValueError("Stage-1 venv tree entries must be a list")
    rows: list[list[Any]] = []
    for entry in entries:
        entry = _object(entry, "Stage-1 venv tree entry")
        entry_type = entry.get("type")
        if entry_type == "directory":
            _exact(entry, {"path", "type", "mode"}, "Stage-1 directory entry")
            rows.append([entry["path"], "directory", entry["mode"]])
        elif entry_type == "file":
            _exact(
                entry,
                {"path", "type", "mode", "bytes", "sha256"},
                "Stage-1 file entry",
            )
            rows.append(
                [
                    entry["path"],
                    "file",
                    entry["mode"],
                    entry["bytes"],
                    entry["sha256"],
                ]
            )
        elif entry_type == "symlink":
            _exact(
                entry,
                {
                    "path",
                    "type",
                    "mode",
                    "link_target",
                    "resolved_path",
                },
                "Stage-1 symlink entry",
            )
            rows.append(
                [
                    entry["path"],
                    "symlink",
                    entry["mode"],
                    entry["link_target"],
                    entry["resolved_path"],
                ]
            )
        else:
            raise ValueError("Stage-1 venv tree contains an unsupported entry")
    if tree["entry_count"] != len(entries) or tree[
        "entries_sha256"
    ] != krea_provenance.canonical_sha256(entries):
        raise ValueError("Stage-1 venv tree manifest digest drifted")
    return {
        "entry_count": len(rows),
        "manifest_sha256": krea_provenance.canonical_sha256(rows),
    }


def _stage1_runtime_identity(
    spec: dict[str, Any],
    *,
    forge_identity: dict[str, Any],
    ai_toolkit_identity: dict[str, Any],
    venv_tree: dict[str, Any],
    materializer_sha256: str,
) -> dict[str, Any]:
    """Reopen and live-recapture the exact runtime that populated the venv."""

    binding = spec["runtime"]["stage1_runtime_receipt"]
    path = _safe_file(Path(binding["path"]), "Stage-1 runtime receipt")
    raw = path.read_bytes()
    receipt = krea_stage1_runtime.load_receipt(path)
    krea_stage1_runtime.validate_receipt(receipt, recapture=True)
    if (
        krea_provenance.file_sha256(path) != binding["file_sha256"]
        or receipt["receipt_sha256"] != binding["receipt_sha256"]
        or receipt["paths"]["receipt"] != str(path)
        or receipt["paths"]["forge_repo"] != spec["sources"]["forge_repo"]
        or receipt["paths"]["ai_toolkit_repo"] != spec["sources"]["ai_toolkit_repo"]
        or receipt["paths"]["destination"] != spec["sources"]["venv"]
    ):
        raise RuntimeError("Stage-1 runtime receipt binding/path drifted")
    if raw != krea_provenance.canonical_bytes(receipt) + b"\n":
        raise RuntimeError("Stage-1 runtime receipt is not canonical JSON")
    for label, receipt_identity, bootstrap_identity in (
        ("forge", receipt["forge"], forge_identity),
        ("ai-toolkit", receipt["ai_toolkit"], ai_toolkit_identity),
    ):
        if (
            receipt_identity["commit"] != bootstrap_identity["commit"]
            or receipt_identity["tree"] != bootstrap_identity["tree"]
        ):
            raise RuntimeError(f"Stage-1 {label} Git identity drifted")
    if (
        receipt["inputs"]["materializer"]["sha256"] != materializer_sha256
        or _stage1_tree_identity(receipt["tree_manifest"]) != venv_tree
    ):
        raise RuntimeError("Stage-1 materializer or complete venv tree drifted")
    return {
        "path": str(path),
        "file_sha256": binding["file_sha256"],
        "receipt_sha256": binding["receipt_sha256"],
        "venv_tree_entries_sha256": receipt["tree_manifest"]["entries_sha256"],
        "verification": receipt["verification"],
    }


def _publish(path: Path, value: dict[str, Any]) -> None:
    path = _no_symlink_ancestors(path, "output", require_exists=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = krea_provenance.canonical_bytes(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def validate_spec(value: dict[str, Any]) -> dict[str, Any]:
    value = _object(value, "host bootstrap spec")
    _exact(
        value,
        {
            "schema",
            "kind",
            "sources",
            "source_identities",
            "requirements",
            "runtime",
            "gpu_execution_authorized",
            "spec_sha256",
        },
        "host bootstrap spec",
    )
    body = {key: item for key, item in value.items() if key != "spec_sha256"}
    if (
        value["schema"] != 1
        or value["kind"] != _SPEC_KIND
        or value["gpu_execution_authorized"] is not False
        or value["spec_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("host bootstrap spec identity is invalid")
    sources = _object(value["sources"], "bootstrap sources")
    _exact(
        sources,
        set(_FIXED_TARGETS) | {"evidence_root"},
        "bootstrap sources",
    )
    normalized_sources = {
        name: _absolute(path, f"bootstrap source {name}")
        for name, path in sources.items()
    }
    if len(set(normalized_sources.values())) != len(normalized_sources):
        raise ValueError("bootstrap sources must be distinct paths")
    evidence_root = normalized_sources["evidence_root"]
    campaign = normalized_sources["campaign"]
    if not campaign.is_relative_to(evidence_root) or campaign == evidence_root:
        raise ValueError("campaign source must be a child of evidence_root")
    source_items = list(normalized_sources.items())
    for index, (left_name, left) in enumerate(source_items):
        for right_name, right in source_items[index + 1 :]:
            if {left_name, right_name} == {"campaign", "evidence_root"}:
                continue
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError(
                    "bootstrap sources may not have ancestor overlap: "
                    f"{left_name}, {right_name}"
                )
    for source in normalized_sources.values():
        for target in map(Path, _FIXED_TARGETS.values()):
            if (
                source == target
                or source.is_relative_to(target)
                or target.is_relative_to(source)
            ):
                raise ValueError("bootstrap sources and fixed targets may not overlap")

    identities = _object(value["source_identities"], "source identities")
    _exact(identities, {"forge_commit", "ai_toolkit_commit"}, "source identities")
    for name, commit in identities.items():
        if not isinstance(commit, str) or not _GIT_SHA.fullmatch(commit):
            raise ValueError(f"source identity {name} must be a full Git commit")

    requirements = _object(value["requirements"], "host requirements")
    _exact(
        requirements,
        {
            "ubuntu_release",
            "minimum_effective_cpu_capacity",
            "minimum_effective_memory_bytes",
            "minimum_checkpoint_filesystem_bytes",
            "minimum_checkpoint_free_bytes",
            "minimum_evidence_filesystem_bytes",
            "minimum_evidence_free_bytes",
            "minimum_gpu_memory_mib",
            "maximum_gpu_memory_mib",
            "minimum_cuda_version",
            "required_docker_runtime",
            "systemd_pid1_required",
            "unified_cgroup_v2_required",
            "rootful_docker_required",
            "separate_evidence_filesystem_required",
        },
        "host requirements",
    )
    if requirements["ubuntu_release"] != "22.04":
        raise ValueError("Week-5 host must freeze Ubuntu 22.04")
    if (
        _positive_number(
            requirements["minimum_effective_cpu_capacity"],
            "minimum effective CPU capacity",
        )
        < 16
    ):
        raise ValueError("host contract requires at least 16 effective CPUs")
    if (
        _positive_int(
            requirements["minimum_effective_memory_bytes"],
            "minimum effective memory bytes",
        )
        < 64 * _GIB
    ):
        raise ValueError("host contract requires at least 64 GiB effective memory")
    checkpoint_size = _positive_int(
        requirements["minimum_checkpoint_filesystem_bytes"],
        "minimum checkpoint filesystem bytes",
    )
    if checkpoint_size < 500 * _GIB:
        raise ValueError("checkpoint filesystem must be at least 500 GiB")
    checkpoint_free = _positive_int(
        requirements["minimum_checkpoint_free_bytes"],
        "minimum checkpoint free bytes",
    )
    if checkpoint_free < 350 * _GIB or checkpoint_free > checkpoint_size:
        raise ValueError("checkpoint free-space floor exceeds filesystem floor")
    evidence_size = _positive_int(
        requirements["minimum_evidence_filesystem_bytes"],
        "minimum evidence filesystem bytes",
    )
    evidence_free = _positive_int(
        requirements["minimum_evidence_free_bytes"],
        "minimum evidence free bytes",
    )
    if evidence_size < 200 * _GIB:
        raise ValueError("evidence filesystem must be at least 200 GiB")
    if evidence_free < 100 * _GIB or evidence_free > evidence_size:
        raise ValueError("evidence free-space floor exceeds filesystem floor")
    gpu_min = _positive_int(
        requirements["minimum_gpu_memory_mib"], "minimum GPU memory MiB"
    )
    gpu_max = _positive_int(
        requirements["maximum_gpu_memory_mib"], "maximum GPU memory MiB"
    )
    if gpu_min < 78_000 or gpu_max > 85_000 or gpu_min > gpu_max:
        raise ValueError("host contract must retain the literal H100-80GB range")
    if requirements["minimum_cuda_version"] != "12.8":
        raise ValueError("host contract must retain CUDA 12.8")
    if requirements["required_docker_runtime"] != "nvidia":
        raise ValueError("host contract must retain the NVIDIA Docker runtime")
    for key in (
        "systemd_pid1_required",
        "unified_cgroup_v2_required",
        "rootful_docker_required",
        "separate_evidence_filesystem_required",
    ):
        if requirements[key] is not True:
            raise ValueError(f"host requirement {key} must remain true")

    runtime = _object(value["runtime"], "bootstrap runtime")
    _exact(
        runtime,
        {
            "container_image_reference",
            "container_image_sha256",
            "execution_surface",
            "ai_toolkit_dir",
            "jit_enabled",
            "stage1_runtime_receipt",
            "runtime_cache_policy",
        },
        "bootstrap runtime",
    )
    if (
        not isinstance(runtime["container_image_reference"], str)
        or not runtime["container_image_reference"]
        or runtime["container_image_reference"]
        != runtime["container_image_reference"].strip()
        or any(
            character.isspace() for character in runtime["container_image_reference"]
        )
    ):
        raise ValueError("container image reference must be explicit canonical text")
    _digest(runtime["container_image_sha256"], "container image sha256")
    if runtime["container_image_reference"] != (
        f"sha256:{runtime['container_image_sha256']}"
    ):
        raise ValueError(
            "container image reference must be its immutable local image ID"
        )
    if runtime["ai_toolkit_dir"] != _FIXED_TARGETS["ai_toolkit_repo"]:
        raise ValueError("AI_TOOLKIT_DIR must be /app/ai-toolkit")
    if runtime["execution_surface"] != "staged_host_venv":
        raise ValueError("Week-5 execution surface must be the staged host venv")
    if runtime["jit_enabled"] is not True:
        raise ValueError("Stage-1 timing requires the certified CUDA/JIT path")
    stage1_binding = _object(
        runtime["stage1_runtime_receipt"], "Stage-1 runtime receipt binding"
    )
    _exact(
        stage1_binding,
        {"path", "file_sha256", "receipt_sha256"},
        "Stage-1 runtime receipt binding",
    )
    stage1_path = _absolute(stage1_binding["path"], "Stage-1 runtime receipt path")
    controls = normalized_sources["campaign"] / "controls"
    if not stage1_path.is_relative_to(controls) or stage1_path == controls:
        raise ValueError(
            "Stage-1 runtime receipt must be under the durable campaign controls"
        )
    _digest(stage1_binding["file_sha256"], "Stage-1 runtime receipt file SHA-256")
    _digest(stage1_binding["receipt_sha256"], "Stage-1 runtime receipt SHA-256")
    cache_policy = _object(runtime["runtime_cache_policy"], "runtime cache policy")
    if cache_policy != _RUNTIME_CACHE_POLICY:
        raise ValueError("runtime cache policy must retain clean per-plan isolation")
    return value


def seal_spec(payload: dict[str, Any]) -> dict[str, Any]:
    if "spec_sha256" in payload:
        raise ValueError("unsealed bootstrap payload contains spec_sha256")
    spec = {**payload, "spec_sha256": krea_provenance.canonical_sha256(payload)}
    return validate_spec(spec)


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    timeout_seconds: int = 30,
) -> str:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=None if environment is None else dict(environment),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    ).stdout.strip()


def _trusted_executable(name: str) -> tuple[str, dict[str, Any]]:
    """Resolve one fixed, root-owned system executable without consulting PATH."""

    requested = Path(_TRUSTED_EXECUTABLE_PATHS[name])
    if not os.path.lexists(requested):
        raise RuntimeError(f"trusted system executable is absent: {requested}")
    requested_stat = requested.lstat()
    if requested_stat.st_uid != 0 or requested_stat.st_mode & 0o022:
        raise RuntimeError(
            f"trusted system executable path is operator-writable: {requested}"
        )
    try:
        resolved = requested.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise RuntimeError(f"trusted system executable is broken: {requested}") from exc
    resolved_stat = resolved.stat()
    if (
        not stat.S_ISREG(resolved_stat.st_mode)
        or resolved_stat.st_uid != 0
        or resolved_stat.st_mode & 0o022
        or not os.access(resolved, os.X_OK)
        or not resolved.is_relative_to(Path("/usr"))
    ):
        raise RuntimeError(
            f"trusted system executable is not a protected /usr file: {resolved}"
        )
    return str(requested), {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        "sha256": krea_provenance.file_sha256(resolved),
        "mode": resolved_stat.st_mode & 0o7777,
        "uid": resolved_stat.st_uid,
    }


def _trusted_executable_identities() -> dict[str, dict[str, Any]]:
    return {
        name: _trusted_executable(name)[1] for name in sorted(_TRUSTED_EXECUTABLE_PATHS)
    }


def _git_identity(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not (path / ".git").exists():
        raise RuntimeError(f"{label} is not a Git worktree")
    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_ATTR_NOSYSTEM": "1",
    }

    def git(*arguments: str) -> str:
        return subprocess.run(
            [
                _trusted_executable("git")[0],
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                f"safe.directory={path}",
                *arguments,
            ],
            cwd=path,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    if commit != expected:
        raise RuntimeError(f"{label} commit differs from bootstrap spec")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError(f"{label} tracked tree is dirty")
    flags = git("ls-files", "-v").splitlines()
    if any(row and (row[0].islower() or row[0] == "S") for row in flags):
        raise RuntimeError(f"{label} uses assume-unchanged or skip-worktree flags")
    names = git("ls-files", "-z")
    tracked = [name for name in names.split("\x00") if name]
    manifest = []
    for name in tracked:
        file_path = _safe_file(path / name, f"{label} tracked file")
        manifest.append(
            [name, file_path.stat().st_size, krea_provenance.file_sha256(file_path)]
        )
    return {
        "commit": commit,
        "tree": git("rev-parse", "HEAD^{tree}"),
        "tracked_file_count": len(manifest),
        "worktree_manifest_sha256": krea_provenance.canonical_sha256(manifest),
    }


def _filesystem(path: Path, *, require_mountpoint: bool) -> dict[str, Any]:
    args = [
        _trusted_executable("findmnt")[0],
        "--json",
        "--output",
        "SOURCE,TARGET,FSTYPE,OPTIONS,MAJ:MIN",
    ]
    args.extend(["--mountpoint" if require_mountpoint else "--target", str(path)])
    try:
        document = json.loads(_run(args))
        rows = document["filesystems"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"cannot identify filesystem for {path}") from exc
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError(f"filesystem identity for {path} is not singular")
    row = _object(rows[0], f"filesystem {path}")
    if set(row) != {"source", "target", "fstype", "options", "maj:min"}:
        raise RuntimeError(f"filesystem identity for {path} is incomplete")
    stat = path.stat()
    major_minor = f"{os.major(stat.st_dev)}:{os.minor(stat.st_dev)}"
    if row["maj:min"] != major_minor:
        raise RuntimeError(f"findmnt/stat disagree for {path}")
    options = sorted(set(str(row["options"]).split(",")))
    return {
        "source": str(row["source"]),
        "target": str(row["target"]),
        "filesystem_type": str(row["fstype"]),
        "mount_options": options,
        "device_major_minor": major_minor,
        "device_id": stat.st_dev,
    }


def _filesystem_capacity(path: Path) -> tuple[int, int]:
    stat = os.statvfs(path)
    return stat.f_blocks * stat.f_frsize, stat.f_bavail * stat.f_frsize


def _is_mountpoint(path: Path) -> bool:
    """Use findmnt because os.path.ismount cannot reliably see bind mounts."""

    result = subprocess.run(
        [_trusted_executable("findmnt")[0], "--mountpoint", str(path), "--noheadings"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"findmnt could not classify mountpoint {path}")
    return result.returncode == 0


def _docker_identity(
    required_runtime: str,
    *,
    image_reference: str,
    expected_image_sha256: str,
    expected_jit_enabled: bool,
) -> dict[str, Any]:
    if os.environ.get("DOCKER_HOST") or os.environ.get("DOCKER_CONTEXT"):
        raise RuntimeError("Docker host/context overrides are forbidden")
    socket_path = Path("/var/run/docker.sock")
    try:
        socket_mode = socket_path.stat().st_mode
    except FileNotFoundError as exc:
        raise RuntimeError("local rootful Docker socket is absent") from exc
    if not stat.S_ISSOCK(socket_mode):
        raise RuntimeError("local Docker endpoint is not a Unix socket")
    docker_environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "HOME": "/nonexistent-forge-bootstrap",
        "DOCKER_HOST": "unix:///var/run/docker.sock",
        "DOCKER_CONTEXT": "default",
    }

    def docker(*arguments: str, timeout_seconds: int = 30) -> str:
        return _run(
            [_trusted_executable("docker")[0], *arguments],
            environment=docker_environment,
            timeout_seconds=timeout_seconds,
        )

    security = json.loads(docker("info", "--format", "{{json .SecurityOptions}}"))
    runtimes = json.loads(docker("info", "--format", "{{json .Runtimes}}"))
    if not isinstance(security, list) or not isinstance(runtimes, dict):
        raise RuntimeError("docker info returned malformed security/runtime data")
    if "rootless" in json.dumps(security).casefold():
        raise RuntimeError("Docker daemon is rootless")
    if required_runtime not in runtimes:
        raise RuntimeError("Docker daemon lacks the required NVIDIA runtime")
    nvidia_cli = _trusted_executable("nvidia_container_cli")[0]
    nvidia_cli_version = _run([nvidia_cli, "--version"])
    try:
        image = _object(
            json.loads(
                docker("image", "inspect", "--format", "{{json .}}", image_reference)
            ),
            "container image inspection",
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError("Docker image inspection returned malformed JSON") from exc
    expected_image_id = f"sha256:{expected_image_sha256}"
    if image.get("Id") != expected_image_id:
        raise RuntimeError(
            "actual Docker image ID differs from the bootstrap specification"
        )
    repo_digests = image.get("RepoDigests", [])
    if not isinstance(repo_digests, list) or any(
        not isinstance(item, str) for item in repo_digests
    ):
        raise RuntimeError("Docker image RepoDigests are malformed")
    config = _object(image.get("Config"), "container image Config")
    image_environment = config.get("Env", [])
    if not isinstance(image_environment, list) or any(
        not isinstance(item, str) for item in image_environment
    ):
        raise RuntimeError("Docker image environment is malformed")
    if expected_jit_enabled is not True:
        raise RuntimeError("Stage-1 reference image must certify CUDA/JIT enabled")
    smoke = docker(
        "run",
        "--rm",
        "--runtime",
        required_runtime,
        "--gpus",
        "all",
        "--entrypoint",
        _trusted_executable("nvidia_smi")[0],
        expected_image_id,
        "--query-gpu=uuid",
        "--format=csv,noheader,nounits",
    )
    if not smoke.startswith("GPU-") or "\n" in smoke:
        raise RuntimeError("container GPU smoke did not expose exactly one GPU")
    jit_program = (
        "import json,torch;"
        "assert torch.cuda.is_available();"
        "f=lambda x:x.square().add(3.0);"
        "g=torch.compile(f,backend='inductor',fullgraph=True);"
        "x=torch.arange(1024,dtype=torch.float32,device='cuda');"
        "y=g(x);torch.cuda.synchronize();"
        "assert torch.equal(y,f(x));"
        "print(json.dumps({'cuda':True,'result':'PASS','torch':torch.__version__,"
        "'torch_cuda':torch.version.cuda},sort_keys=True,separators=(',',':')))"
    )
    jit_stdout = docker(
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=8g,mode=1777",
        "--runtime",
        required_runtime,
        "--gpus",
        "all",
        "--env",
        "HOME=/tmp/home",
        "--env",
        "XDG_CACHE_HOME=/tmp/xdg",
        "--env",
        "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
        "--env",
        "TRITON_CACHE_DIR=/tmp/triton",
        "--entrypoint",
        "/usr/bin/python3",
        expected_image_id,
        "-I",
        "-c",
        jit_program,
        timeout_seconds=300,
    )
    try:
        jit_smoke = _object(json.loads(jit_stdout), "container CUDA/JIT smoke")
    except json.JSONDecodeError as exc:
        raise RuntimeError("container CUDA/JIT smoke returned malformed JSON") from exc
    if (
        set(jit_smoke) != {"cuda", "result", "torch", "torch_cuda"}
        or jit_smoke["cuda"] is not True
        or jit_smoke["result"] != "PASS"
        or not isinstance(jit_smoke["torch"], str)
        or not isinstance(jit_smoke["torch_cuda"], str)
    ):
        raise RuntimeError("container CUDA/JIT compile smoke did not pass")
    return {
        "server_version": docker("version", "--format", "{{.Server.Version}}"),
        "docker_root_dir": docker("info", "--format", "{{.DockerRootDir}}"),
        "security_options": security,
        "runtimes": sorted(runtimes),
        "rootless": False,
        "nvidia_container_cli": nvidia_cli,
        "nvidia_container_cli_version": nvidia_cli_version,
        "container_image": {
            "reference": image_reference,
            "image_id": expected_image_id,
            "repo_digests": sorted(repo_digests),
            "environment_sha256": krea_provenance.canonical_sha256(
                sorted(image_environment)
            ),
            "gpu_smoke_uuid": smoke,
            "cuda_jit_smoke": jit_smoke,
            "cuda_jit_smoke_stdout_sha256": hashlib.sha256(
                jit_stdout.encode("utf-8")
            ).hexdigest(),
        },
    }


def _gpu_identity() -> dict[str, Any]:
    output = _run(
        [
            _trusted_executable("nvidia_smi")[0],
            "--query-gpu=uuid,name,driver_version,mig.mode.current,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = list(csv.reader(output.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise RuntimeError("host contract requires exactly one complete GPU row")
    fields = [item.strip() for item in rows[0]]
    try:
        total_memory = int(float(fields[4]))
    except ValueError as exc:
        raise RuntimeError("GPU memory is malformed") from exc
    header = _run([_trusted_executable("nvidia_smi")[0]])
    match = re.search(r"CUDA Version:\s*(\d+\.\d+)", header)
    if match is None:
        raise RuntimeError("nvidia-smi does not report a CUDA compatibility version")
    process_result = subprocess.run(
        [
            _trusted_executable("nvidia_smi")[0],
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = (process_result.stdout + process_result.stderr).strip()
    if (
        process_result.returncode != 0
        and "no running processes" not in combined.casefold()
    ):
        raise RuntimeError("cannot establish GPU compute-process occupancy")
    process_rows = [
        row.strip()
        for row in process_result.stdout.splitlines()
        if row.strip() and "no running processes" not in row.casefold()
    ]
    if process_rows:
        raise RuntimeError(f"GPU has foreign compute processes: {process_rows}")
    return {
        "uuid": fields[0],
        "name": fields[1],
        "driver_version": fields[2],
        "mig_mode": fields[3],
        "total_memory_mib": total_memory,
        "cuda_version": match.group(1),
        "compute_processes": [],
    }


def _os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    if "ID" not in values or "VERSION_ID" not in values:
        raise RuntimeError("/etc/os-release lacks ID or VERSION_ID")
    return {"id": values["ID"], "version_id": values["VERSION_ID"]}


def _host_identity(spec: dict[str, Any]) -> dict[str, Any]:
    requirements = spec["requirements"]
    if os.geteuid() != 0:
        raise RuntimeError("host bootstrap requires root")
    pid1 = Path("/proc/1/comm").read_text(encoding="ascii").strip()
    if pid1 != "systemd":
        raise RuntimeError("host bootstrap requires systemd as PID 1")
    cgroup = krea_host_identity._cgroup_constraints()
    if not Path("/sys/fs/cgroup/cgroup.controllers").is_file():
        raise RuntimeError("host bootstrap requires unified cgroup v2")
    os_release = _os_release()
    if os_release != {"id": "ubuntu", "version_id": requirements["ubuntu_release"]}:
        raise RuntimeError("host OS is not the frozen Ubuntu release")

    logical = int(os.cpu_count() or 0)
    affinity = (
        set(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else set(range(logical))
    )
    effective_ids = affinity & set(cgroup["cpuset_cpu_ids"])
    quota = cgroup["cpu_quota_cores"]
    effective_cpu = min(
        [float(len(effective_ids))] + ([] if quota is None else [float(quota)])
    )
    if effective_cpu < float(requirements["minimum_effective_cpu_capacity"]):
        raise RuntimeError("effective CPU capacity is below the host contract")
    mem = krea_host_identity._meminfo()
    memory_limit = cgroup["memory_limit_bytes"]
    effective_memory = min(
        [mem["MemTotal"]] + ([] if memory_limit is None else [memory_limit])
    )
    if effective_memory < requirements["minimum_effective_memory_bytes"]:
        raise RuntimeError("effective memory is below the host contract")

    gpu = _gpu_identity()
    if (
        "H100" not in gpu["name"].upper()
        or gpu["mig_mode"].casefold() != "disabled"
        or not requirements["minimum_gpu_memory_mib"]
        <= gpu["total_memory_mib"]
        <= requirements["maximum_gpu_memory_mib"]
    ):
        raise RuntimeError("GPU is not one non-MIG literal H100-80GB")
    cuda = tuple(int(item) for item in gpu["cuda_version"].split("."))
    required_cuda = tuple(
        int(item) for item in requirements["minimum_cuda_version"].split(".")
    )
    if cuda < required_cuda:
        raise RuntimeError("NVIDIA driver CUDA compatibility is below 12.8")
    docker = _docker_identity(
        requirements["required_docker_runtime"],
        image_reference=spec["runtime"]["container_image_reference"],
        expected_image_sha256=spec["runtime"]["container_image_sha256"],
        expected_jit_enabled=spec["runtime"]["jit_enabled"],
    )
    return {
        "os": os_release,
        "pid1": pid1,
        "cgroup_version": 2,
        "cpu": {
            "logical_cpus": logical,
            "effective_cpu_ids": sorted(effective_ids),
            "cpu_quota_cores": quota,
            "effective_cpu_capacity": effective_cpu,
        },
        "memory": {
            "total_bytes": mem["MemTotal"],
            "cgroup_limit_bytes": memory_limit,
            "effective_capacity_bytes": effective_memory,
        },
        "gpu": gpu,
        "docker": docker,
        "trusted_executables": _trusted_executable_identities(),
    }


def _source_identity(spec: dict[str, Any]) -> dict[str, Any]:
    sources = {
        name: _no_symlink_ancestors(Path(path), name, require_exists=True)
        for name, path in spec["sources"].items()
    }
    forge = sources["forge_repo"]
    ai_toolkit = sources["ai_toolkit_repo"]
    venv_python, resolved_venv_python = _safe_venv_python(
        sources["venv"], "venv Python"
    )
    if not os.access(resolved_venv_python, os.X_OK):
        raise RuntimeError("staged venv Python is not executable")
    venv_tree = _tree_identity(sources["venv"], "staged venv")
    artifacts: dict[str, Any] = {}
    for name, relative in _CALIBRATION_ARTIFACTS.items():
        path = _safe_file(forge / relative, f"calibration artifact {name}")
        artifacts[name] = {
            "relative_path": relative,
            "sha256": krea_provenance.file_sha256(path),
        }
    campaign = sources["campaign"]
    evidence = sources["evidence_root"]
    checkpoint = sources["checkpoints"]
    dataset = sources["dataset"]
    cache = sources["cache"]
    evidence_fs = _filesystem(evidence, require_mountpoint=True)
    campaign_fs = _filesystem(campaign, require_mountpoint=False)
    checkpoint_fs = _filesystem(checkpoint, require_mountpoint=False)
    dataset_fs = _filesystem(dataset, require_mountpoint=False)
    cache_fs = _filesystem(cache, require_mountpoint=False)
    if evidence_fs["filesystem_type"].casefold() in {
        "tmpfs",
        "ramfs",
        "overlay",
        "overlayfs",
    }:
        raise RuntimeError("evidence filesystem is not persistent block storage")
    if campaign.stat().st_dev != evidence.stat().st_dev:
        raise RuntimeError("campaign source is not on the evidence filesystem")
    if evidence.stat().st_dev in {
        checkpoint.stat().st_dev,
        dataset.stat().st_dev,
        cache.stat().st_dev,
    }:
        raise RuntimeError("evidence filesystem is not distinct from volatile storage")
    if len({checkpoint.stat().st_dev, dataset.stat().st_dev, cache.stat().st_dev}) != 1:
        raise RuntimeError(
            "checkpoint, dataset, and cache sources must share qualified volatile storage"
        )
    requirements = spec["requirements"]
    checkpoint_total, checkpoint_free = _filesystem_capacity(checkpoint)
    evidence_total, evidence_free = _filesystem_capacity(evidence)
    if (
        checkpoint_total < requirements["minimum_checkpoint_filesystem_bytes"]
        or checkpoint_free < requirements["minimum_checkpoint_free_bytes"]
    ):
        raise RuntimeError("checkpoint filesystem capacity/free-space gate failed")
    if (
        evidence_total < requirements["minimum_evidence_filesystem_bytes"]
        or evidence_free < requirements["minimum_evidence_free_bytes"]
    ):
        raise RuntimeError("evidence filesystem capacity/free-space gate failed")
    forge_identity = _git_identity(
        forge, spec["source_identities"]["forge_commit"], "Forge repository"
    )
    ai_toolkit_identity = _git_identity(
        ai_toolkit,
        spec["source_identities"]["ai_toolkit_commit"],
        "ai-toolkit repository",
    )
    stage1_runtime = _stage1_runtime_identity(
        spec,
        forge_identity=forge_identity,
        ai_toolkit_identity=ai_toolkit_identity,
        venv_tree=venv_tree,
        materializer_sha256=artifacts["stage1_runtime_tool"]["sha256"],
    )
    return {
        "forge_repo": forge_identity,
        "ai_toolkit_repo": ai_toolkit_identity,
        "venv_python": {
            "relative_path": "bin/python",
            "is_symlink": venv_python.is_symlink(),
            "resolved_relative_path": resolved_venv_python.relative_to(
                sources["venv"].resolve(strict=True)
            ).as_posix(),
            "resolved_sha256": krea_provenance.file_sha256(resolved_venv_python),
        },
        "venv_tree": venv_tree,
        "stage1_runtime": stage1_runtime,
        "calibration_artifacts": artifacts,
        "filesystems": {
            "evidence_root": evidence_fs,
            "campaign": campaign_fs,
            "checkpoints": checkpoint_fs,
            "dataset": dataset_fs,
            "cache": cache_fs,
        },
    }


def _binding_state(spec: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    source_identity = _source_identity(spec)
    for name, target_text in _FIXED_TARGETS.items():
        source = Path(spec["sources"][name])
        target = Path(target_text)
        if not target.is_dir() or not _is_mountpoint(target):
            raise RuntimeError(f"fixed target is not a mountpoint: {target}")
        if not os.path.samefile(source, target):
            raise RuntimeError(
                f"fixed target does not bind its staged source: {target}"
            )
        mount = _filesystem(target, require_mountpoint=True)
        expected_read_only = name in _READ_ONLY_BINDINGS
        if expected_read_only != ("ro" in mount["mount_options"]):
            raise RuntimeError(f"fixed target read-only state drifted: {target}")
        result[name] = {
            "source": str(source),
            "target": target_text,
            "read_only": expected_read_only,
            "filesystem": mount,
        }
    forge_target = Path(_FIXED_TARGETS["forge_repo"])
    venv_target = Path(_FIXED_TARGETS["venv"])
    for name, relative in _CALIBRATION_ARTIFACTS.items():
        target = _safe_file(forge_target / relative, f"mounted artifact {name}")
        expected = source_identity["calibration_artifacts"][name]["sha256"]
        if krea_provenance.file_sha256(target) != expected:
            raise RuntimeError(f"mounted calibration artifact drifted: {name}")
    _, resolved_python = _safe_venv_python(venv_target, "mounted venv Python")
    if not os.access(resolved_python, os.X_OK):
        raise RuntimeError("mounted venv Python is not executable")
    if _tree_identity(venv_target, "mounted venv") != source_identity["venv_tree"]:
        raise RuntimeError("mounted venv tree differs from its staged source identity")
    for leaf in _CAMPAIGN_LEAVES:
        path = _no_symlink_ancestors(
            Path(_FIXED_TARGETS["campaign"]) / leaf,
            f"campaign leaf {leaf}",
            require_exists=True,
        )
        if path.stat().st_dev != Path(_FIXED_TARGETS["campaign"]).stat().st_dev:
            raise RuntimeError(f"campaign leaf escaped evidence filesystem: {leaf}")
    result["runtime_cache"] = _runtime_cache_identity(require_empty=False)
    return result


def _runtime_cache_identity(*, require_empty: bool) -> dict[str, Any]:
    root = _no_symlink_ancestors(
        _RUNTIME_CACHE_ROOT, "runtime cache root", require_exists=True
    )
    cache_mount = Path(_FIXED_TARGETS["cache"])
    if (
        not root.is_dir()
        or root.parent != cache_mount
        or root.stat().st_dev != cache_mount.stat().st_dev
        or root.stat().st_uid != 0
        or root.stat().st_mode & 0o7777 != 0o700
        or not os.access(root, os.R_OK | os.W_OK | os.X_OK)
    ):
        raise RuntimeError("runtime cache root is not protected writable cache storage")
    children = sorted(item.name for item in root.iterdir())
    if require_empty and children:
        raise RuntimeError("runtime cache root is not empty before bootstrap")
    return {
        "path": str(root),
        "device_id": root.stat().st_dev,
        "mode": root.stat().st_mode & 0o7777,
        "uid": root.stat().st_uid,
        "policy": dict(_RUNTIME_CACHE_POLICY),
    }


def _prepare_runtime_cache_root() -> None:
    cache_mount = _no_symlink_ancestors(
        Path(_FIXED_TARGETS["cache"]), "cache binding", require_exists=True
    )
    if os.path.lexists(_RUNTIME_CACHE_ROOT):
        _runtime_cache_identity(require_empty=True)
        return
    _RUNTIME_CACHE_ROOT.mkdir(mode=0o700)
    if _RUNTIME_CACHE_ROOT.parent != cache_mount:
        raise RuntimeError("runtime cache root escaped the fixed cache binding")
    _runtime_cache_identity(require_empty=True)


def _preflight(spec: dict[str, Any], *, require_bindings: bool) -> dict[str, Any]:
    spec = validate_spec(spec)
    host = _host_identity(spec)
    sources = _source_identity(spec)
    docker_root = _no_symlink_ancestors(
        Path(host["docker"]["docker_root_dir"]),
        "DockerRootDir",
        require_exists=True,
    )
    checkpoint_source = Path(spec["sources"]["checkpoints"])
    if docker_root.stat().st_dev != checkpoint_source.stat().st_dev:
        raise RuntimeError("DockerRootDir is outside qualified volatile storage")
    result = {"host": host, "sources": sources}
    if require_bindings:
        result["bindings"] = _binding_state(spec)
    return result


def _ensure_empty_target(path: Path) -> bool:
    path = _no_symlink_ancestors(path, "bind target", require_exists=False)
    created = False
    if not path.exists():
        path.mkdir(parents=True, mode=0o755)
        created = True
    if not path.is_dir() or any(path.iterdir()):
        raise RuntimeError(f"bind target must be an empty directory: {path}")
    return created


def _rollback_bindings(mounted: Sequence[Path]) -> None:
    failures: list[str] = []
    for target in reversed(mounted):
        try:
            result = subprocess.run(
                [_trusted_executable("umount")[0], "--", str(target)],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                failures.append(
                    f"{target}: umount status {result.returncode}: "
                    f"{(result.stderr or result.stdout).strip()}"
                )
        except BaseException as exc:
            failures.append(f"{target}: umount exception {type(exc).__name__}: {exc}")
        try:
            if _is_mountpoint(target):
                failures.append(f"{target}: mount remains active")
        except BaseException as exc:
            failures.append(
                f"{target}: post-umount state unknown {type(exc).__name__}: {exc}"
            )
    if failures:
        targets = ", ".join(str(path) for path in reversed(mounted))
        raise RuntimeError(
            "bootstrap rollback incomplete; GPU execution remains forbidden. "
            f"Manually unmount and audit these targets: {targets}. Details: {failures}"
        )


def _apply_bindings(spec: dict[str, Any]) -> list[Path]:
    mounted: list[Path] = []
    try:
        for name, target_text in _FIXED_TARGETS.items():
            source = Path(spec["sources"][name])
            target = Path(target_text)
            if target.exists() and _is_mountpoint(target):
                if not os.path.samefile(source, target):
                    raise RuntimeError(f"pre-existing mount conflicts at {target}")
                continue
            _ensure_empty_target(target)
            subprocess.run(
                [
                    _trusted_executable("mount")[0],
                    "--bind",
                    "--",
                    str(source),
                    str(target),
                ],
                check=True,
                timeout=30,
            )
            mounted.append(target)
            if name in _READ_ONLY_BINDINGS:
                subprocess.run(
                    [
                        _trusted_executable("mount")[0],
                        "-o",
                        "remount,bind,ro",
                        "--",
                        str(target),
                    ],
                    check=True,
                    timeout=30,
                )
        for leaf in _CAMPAIGN_LEAVES:
            path = Path(_FIXED_TARGETS["campaign"]) / leaf
            if path.is_symlink():
                raise RuntimeError(f"campaign leaf is a symlink: {path}")
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        _prepare_runtime_cache_root()
        return mounted
    except BaseException:
        _rollback_bindings(mounted)
        raise


def build_receipt(spec: dict[str, Any]) -> dict[str, Any]:
    identity = _preflight(spec, require_bindings=True)
    body = {
        "schema": 1,
        "kind": _RECEIPT_KIND,
        "spec": spec,
        "layout_identity": identity,
        "gpu_execution_authorized": False,
    }
    return {**body, "receipt_sha256": krea_provenance.canonical_sha256(body)}


def validate_receipt(value: dict[str, Any], *, recapture: bool) -> dict[str, Any]:
    value = _object(value, "host bootstrap receipt")
    _exact(
        value,
        {
            "schema",
            "kind",
            "spec",
            "layout_identity",
            "gpu_execution_authorized",
            "receipt_sha256",
        },
        "host bootstrap receipt",
    )
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        value["schema"] != 1
        or value["kind"] != _RECEIPT_KIND
        or value["gpu_execution_authorized"] is not False
        or value["receipt_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("host bootstrap receipt identity is invalid")
    spec = validate_spec(_object(value["spec"], "receipt spec"))
    if recapture:
        observed = _preflight(spec, require_bindings=True)
        if observed != value["layout_identity"]:
            raise RuntimeError("host bootstrap layout identity drifted")
    return value


def _status(action: str, **values: Any) -> None:
    print(
        krea_provenance.canonical_bytes(
            {"status": "PASS", "action": action, **values}
        ).decode("ascii")
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal-layout-spec")
    seal.add_argument("--payload", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate-layout-spec")
    validate.add_argument("--spec", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-layout")
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify-layout")
    verify.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "seal-layout-spec":
        payload = _load_canonical(args.payload, "bootstrap-spec payload")
        result = seal_spec(payload)
        _publish(args.output, result)
        _status(args.command, spec_sha256=result["spec_sha256"])
        return 0
    spec = _load_canonical(
        args.spec if hasattr(args, "spec") else args.receipt,
        "host bootstrap spec" if hasattr(args, "spec") else "host bootstrap receipt",
    )
    if args.command == "validate-layout-spec":
        result = validate_spec(spec)
        _status(args.command, spec_sha256=result["spec_sha256"])
        return 0
    if args.command == "prepare-layout":
        spec = validate_spec(spec)
        _preflight(spec, require_bindings=False)
        mounted = _apply_bindings(spec)
        try:
            receipt = build_receipt(spec)
            output = _absolute(str(args.output), "bootstrap receipt output")
            if not output.is_relative_to(Path(_FIXED_TARGETS["campaign"]) / "controls"):
                raise ValueError(
                    "bootstrap receipt must be written under /campaign/controls"
                )
            _publish(output, receipt)
        except BaseException:
            _rollback_bindings(mounted)
            raise
        _status(args.command, receipt_sha256=receipt["receipt_sha256"])
        return 0
    receipt = validate_receipt(spec, recapture=True)
    _status(args.command, receipt_sha256=receipt["receipt_sha256"])
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests.
    raise SystemExit(main())
