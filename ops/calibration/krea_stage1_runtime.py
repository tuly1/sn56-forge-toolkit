#!/usr/bin/env python3
"""Create and attest the Week-5 Stage-1 host Python runtime.

This is intentionally a bounded materializer, not a package manager wrapper.
It reproduces the dependency phase order of the production toolkit Dockerfile
from already-staged, credential-free sources.  It accepts no index override,
credential, Python override, or mutable destination.  Successful execution
publishes one canonical, create-only receipt containing the complete venv tree
manifest and every command's output.

The module is stdlib-only at import time.  Third-party imports happen only in
the two probe subcommands, which are invoked with the newly-created venv.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


KIND = "forge-krea-stage1-runtime-receipt"
PLAN_KIND = "forge-krea-stage1-runtime-plan"
SCHEMA = 1
SYSTEM_PYTHON = Path("/usr/bin/python3")
SYSTEM_GIT = Path("/usr/bin/git")
NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
TRANSIENT_CACHE_PARENT = Path("/var/tmp")
EXPECTED_UBUNTU_ID = "ubuntu"
EXPECTED_UBUNTU_VERSION = "22.04"
EXPECTED_PYTHON = (3, 10)
AI_TOOLKIT_COMMIT = "99be3d96a2468d3a5228a4eb05ba67e63c586b4e"
AI_TOOLKIT_REQUIREMENTS_SHA256 = (
    "6264997796f6a6da55a3b815481cc3cea63a169b01ca55c8b6ee417367b3f5ea"
)
BASE_IMAGE_SHA256 = "c24f8bb95bf1dc8da7cd6158a763f2c9782783ad7648dc4047c5757ef3447db8"
BASE_IMAGE_REFERENCE = "diagonalge/ai-toolkit:latest@sha256:" + BASE_IMAGE_SHA256
PYTORCH_INDEX = "https://download.pytorch.org/whl/cu124"

# A Dockerfile edit is a production-contract edit, not an input to improvise
# around.  Update this constant only alongside an intentional review of the
# command phases below.
TOOLKIT_DOCKERFILE_SHA256 = (
    "762d613b125bd52612f27198b2eee0f8442457993f9680cab27f3439eb68089b"
)
RUNTIME_LOCK_SHA256 = "9c4c15130508c547c67d891f559ca1a513cd62bd5a4b695eb25ceafccd0b850b"
PHASE1_CONSTRAINTS_SHA256 = (
    "864ed2d3c45f86464b189e3f1685e0578eae2af9ecf49e6bb63cadf3a85986ac"
)
RUNTIME_VERIFIER_SHA256 = (
    "533114f9fe9c1e550575061d5b04f9cdc7e661c9047fc36cdd434904eec3500b"
)
EXPECTED_PIP_CHECK_LINE = (
    "easy-dwpose 1.0.3 has requirement huggingface_hub<1.0,>=0.26, "
    "but you have huggingface-hub 1.10.1."
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_SAFE_ENV = {
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/nonexistent",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "PIP_CONFIG_FILE": "/dev/null",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PIP_NO_INPUT": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
}


class Stage1RuntimeError(RuntimeError):
    """The host, inputs, command plan, or resulting runtime failed closed."""


Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class RuntimePaths:
    """Canonical absolute paths used by one materialization."""

    forge_repo: Path
    ai_toolkit_repo: Path
    destination: Path
    receipt: Path

    @property
    def dockerfile(self) -> Path:
        return (
            self.forge_repo / "ops/docker/standalone-image-toolkit-trainer.dockerfile"
        )

    @property
    def runtime_lock(self) -> Path:
        return self.forge_repo / "ops/docker/image-runtime-lock.txt"

    @property
    def constraints(self) -> Path:
        return self.forge_repo / "ops/docker/image-runtime-phase1-constraints.txt"

    @property
    def verifier(self) -> Path:
        return self.forge_repo / "ops/docker/verify_image_runtime.py"

    @property
    def materializer(self) -> Path:
        return self.forge_repo / "ops/calibration/krea_stage1_runtime.py"

    @property
    def requirements(self) -> Path:
        return self.ai_toolkit_repo / "requirements.txt"

    @property
    def venv_python(self) -> Path:
        return self.destination / "bin/python"


@dataclass(frozen=True)
class CommandSpec:
    command_id: str
    phase: str
    argv: tuple[str, ...]
    cwd: str
    attempts: int = 1
    condition: str = "always"

    def json(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "argv": list(self.argv),
            "command_id": self.command_id,
            "condition": self.condition,
            "cwd": self.cwd,
            "phase": self.phase,
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = keys - set(value)
    extra = set(value) - keys
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _normalized_absolute(path: str | Path, label: str) -> Path:
    raw = str(path)
    if not raw or raw != raw.strip():
        raise ValueError(f"{label} must be a non-empty normalized absolute path")
    pure = PurePosixPath(raw)
    if not pure.is_absolute() or str(pure) != raw or ".." in pure.parts:
        raise ValueError(f"{label} must be a normalized absolute POSIX path")
    return Path(raw)


def _reject_symlink_ancestors(
    path: Path, label: str, *, include_leaf: bool = True
) -> None:
    current = path if include_leaf else path.parent
    while True:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink component: {current}")
        if current == current.parent:
            return
        current = current.parent


def _regular_file(path: Path, label: str) -> Path:
    _reject_symlink_ancestors(path, label)
    if not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _directory(path: Path, label: str) -> Path:
    _reject_symlink_ancestors(path, label)
    if not path.is_dir():
        raise ValueError(f"{label} must be an existing non-symlink directory: {path}")
    return path


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _paths(
    forge_repo: str | Path,
    ai_toolkit_repo: str | Path,
    destination: str | Path,
    receipt: str | Path,
    *,
    destination_must_be_absent: bool,
    receipt_must_be_absent: bool,
) -> RuntimePaths:
    paths = RuntimePaths(
        forge_repo=_normalized_absolute(forge_repo, "forge repo"),
        ai_toolkit_repo=_normalized_absolute(ai_toolkit_repo, "ai-toolkit repo"),
        destination=_normalized_absolute(destination, "venv destination"),
        receipt=_normalized_absolute(receipt, "receipt output"),
    )
    _directory(paths.forge_repo, "forge repo")
    _directory(paths.ai_toolkit_repo, "ai-toolkit repo")
    _reject_symlink_ancestors(paths.destination, "venv destination")
    _reject_symlink_ancestors(paths.receipt, "receipt output")
    if destination_must_be_absent and os.path.lexists(paths.destination):
        raise FileExistsError(f"venv destination already exists: {paths.destination}")
    if receipt_must_be_absent and os.path.lexists(paths.receipt):
        raise FileExistsError(f"receipt output already exists: {paths.receipt}")
    if not paths.destination.parent.is_dir():
        raise ValueError("venv destination parent must already exist")
    if not paths.receipt.parent.is_dir():
        raise ValueError("receipt output parent must already exist")
    named = {
        "forge repo": paths.forge_repo,
        "ai-toolkit repo": paths.ai_toolkit_repo,
        "venv destination": paths.destination,
        "receipt output": paths.receipt,
    }
    names = list(named)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            if _overlap(named[left_name], named[right_name]):
                raise ValueError(f"path alias/overlap: {left_name} and {right_name}")
    inputs = [
        paths.dockerfile,
        paths.runtime_lock,
        paths.constraints,
        paths.verifier,
        paths.materializer,
        paths.requirements,
    ]
    identities: set[tuple[int, int]] = set()
    for input_path in inputs:
        _regular_file(input_path, f"input {input_path.name}")
        identity = (input_path.stat().st_dev, input_path.stat().st_ino)
        if identity in identities:
            raise ValueError(f"input path aliases another input: {input_path}")
        identities.add(identity)
    return paths


def _file_identity(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = file_sha256(path)
    after = path.stat()
    before_key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_key = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_key != after_key:
        raise Stage1RuntimeError(f"input changed while hashing: {path}")
    return {"bytes": after.st_size, "path": str(path), "sha256": digest}


def _read_os_release(path: Path = Path("/etc/os-release")) -> dict[str, Any]:
    _regular_file(path, "os-release")
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw = line.split("=", 1)
        value = raw.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        fields[key] = value
    if fields.get("ID") != EXPECTED_UBUNTU_ID:
        raise Stage1RuntimeError("Stage-1 host must be Ubuntu")
    if fields.get("VERSION_ID") != EXPECTED_UBUNTU_VERSION:
        raise Stage1RuntimeError("Stage-1 host must be Ubuntu 22.04")
    return {
        "file": _file_identity(path),
        "id": fields["ID"],
        "version_id": fields["VERSION_ID"],
    }


def _sanitized_environment(paths: RuntimePaths) -> dict[str, str]:
    environment = dict(_SAFE_ENV)
    transient = _transient_cache_root(paths)
    environment["HOME"] = str(transient / "home")
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(transient / "torchinductor")
    environment["TRITON_CACHE_DIR"] = str(transient / "triton")
    return environment


def _transient_cache_root(paths: RuntimePaths) -> Path:
    suffix = hashlib.sha256(str(paths.destination).encode()).hexdigest()[:24]
    return TRANSIENT_CACHE_PARENT / f"sn56-krea-stage1-{suffix}"


def _prepare_transient_cache(paths: RuntimePaths) -> Path:
    parent = _directory(TRANSIENT_CACHE_PARENT, "transient cache parent")
    root = _transient_cache_root(paths)
    _reject_symlink_ancestors(root, "transient materialization cache")
    if os.path.lexists(root):
        raise FileExistsError(f"transient materialization cache exists: {root}")
    root.mkdir(mode=0o700)
    for leaf in ("home", "torchinductor", "triton"):
        (root / leaf).mkdir(mode=0o700)
    if root.parent != parent:
        raise AssertionError("transient cache escaped its fixed parent")
    return root


_PYTHON_IDENTITY_CODE = """\
import json, os, sys, sysconfig
print(json.dumps({
    "cache_tag": sys.implementation.cache_tag,
    "executable": sys.executable,
    "implementation": sys.implementation.name,
    "real_executable": os.path.realpath(sys.executable),
    "soabi": sysconfig.get_config_var("SOABI"),
    "version": list(sys.version_info[:3]),
}, sort_keys=True, separators=(",", ":")))
"""


def command_plan(paths: RuntimePaths) -> tuple[CommandSpec, ...]:
    python = str(paths.venv_python)

    def git(repo: Path, *arguments: str) -> tuple[str, ...]:
        return (
            str(SYSTEM_GIT),
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            f"safe.directory={repo}",
            *arguments,
        )

    return (
        CommandSpec(
            "system-python-identity",
            "preflight",
            (str(SYSTEM_PYTHON), "-I", "-c", _PYTHON_IDENTITY_CODE),
            "/",
        ),
        CommandSpec(
            "forge-head",
            "preflight",
            git(paths.forge_repo, "rev-parse", "--verify", "HEAD"),
            str(paths.forge_repo),
        ),
        CommandSpec(
            "forge-tree",
            "preflight",
            git(paths.forge_repo, "rev-parse", "HEAD^{tree}"),
            str(paths.forge_repo),
        ),
        CommandSpec(
            "forge-status",
            "preflight",
            git(
                paths.forge_repo,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ),
            str(paths.forge_repo),
        ),
        CommandSpec(
            "forge-tracked-manifest",
            "preflight",
            git(paths.forge_repo, "ls-files", "--stage", "-z"),
            str(paths.forge_repo),
        ),
        CommandSpec(
            "forge-tracked-flags",
            "preflight",
            git(paths.forge_repo, "ls-files", "-v", "-z"),
            str(paths.forge_repo),
        ),
        CommandSpec(
            "ai-toolkit-head",
            "preflight",
            git(paths.ai_toolkit_repo, "rev-parse", "--verify", "HEAD"),
            str(paths.ai_toolkit_repo),
        ),
        CommandSpec(
            "ai-toolkit-tree",
            "preflight",
            git(paths.ai_toolkit_repo, "rev-parse", "HEAD^{tree}"),
            str(paths.ai_toolkit_repo),
        ),
        CommandSpec(
            "ai-toolkit-status",
            "preflight",
            git(
                paths.ai_toolkit_repo,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ),
            str(paths.ai_toolkit_repo),
        ),
        CommandSpec(
            "ai-toolkit-tracked-manifest",
            "preflight",
            git(paths.ai_toolkit_repo, "ls-files", "--stage", "-z"),
            str(paths.ai_toolkit_repo),
        ),
        CommandSpec(
            "ai-toolkit-tracked-flags",
            "preflight",
            git(paths.ai_toolkit_repo, "ls-files", "-v", "-z"),
            str(paths.ai_toolkit_repo),
        ),
        CommandSpec(
            "venv-create",
            "materialize",
            (
                str(SYSTEM_PYTHON),
                "-m",
                "venv",
                "--copies",
                str(paths.destination),
            ),
            str(paths.destination.parent),
        ),
        CommandSpec(
            "phase1-ai-toolkit-requirements",
            "materialize",
            (
                python,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--constraint",
                str(paths.constraints),
                "--requirement",
                str(paths.requirements),
            ),
            str(paths.ai_toolkit_repo),
            attempts=5,
        ),
        CommandSpec(
            "phase1-torch-cu124",
            "materialize",
            (
                python,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "torch==2.6.0",
                "torchvision==0.21.0",
                "torchaudio==2.6.0",
                "--index-url",
                PYTORCH_INDEX,
            ),
            str(paths.ai_toolkit_repo),
            attempts=5,
        ),
        CommandSpec(
            "phase1-torchcodec-support",
            "materialize",
            (
                python,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--constraint",
                str(paths.constraints),
                "torchcodec==0.2.1",
                "pyyaml",
                "Pillow",
                "numpy",
                "safetensors",
            ),
            str(paths.ai_toolkit_repo),
            attempts=5,
        ),
        CommandSpec(
            "phase2-certified-runtime-lock",
            "materialize",
            (
                python,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-deps",
                "--extra-index-url",
                PYTORCH_INDEX,
                "--requirement",
                str(paths.runtime_lock),
            ),
            str(paths.ai_toolkit_repo),
            attempts=5,
        ),
        CommandSpec(
            "verify-image-runtime",
            "verify",
            (
                python,
                str(paths.verifier),
                "--lock",
                str(paths.runtime_lock),
                "--constraints",
                str(paths.constraints),
            ),
            str(paths.ai_toolkit_repo),
        ),
        CommandSpec(
            "essential-imports",
            "verify",
            (
                python,
                str(paths.materializer),
                "probe-essential",
                "--ai-toolkit-repo",
                str(paths.ai_toolkit_repo),
            ),
            str(paths.ai_toolkit_repo),
        ),
        CommandSpec(
            "gpu-detect",
            "verify",
            (str(NVIDIA_SMI), "-L"),
            "/",
            condition="nvidia-smi-present",
        ),
        CommandSpec(
            "cuda-jit-smoke",
            "verify",
            (python, str(paths.materializer), "probe-cuda-jit"),
            str(paths.ai_toolkit_repo),
            condition="gpu-present",
        ),
    )


def _contract(paths: RuntimePaths) -> dict[str, Any]:
    plan = [spec.json() for spec in command_plan(paths)]
    return {
        "ai_toolkit_commit": AI_TOOLKIT_COMMIT,
        "base_image_reference": BASE_IMAGE_REFERENCE,
        "base_image_sha256": BASE_IMAGE_SHA256,
        "command_environment": _sanitized_environment(paths),
        "command_plan": plan,
        "command_plan_sha256": canonical_sha256(plan),
        "indexes": {
            "phase1_torch_index_url": PYTORCH_INDEX,
            "phase2_extra_index_url": PYTORCH_INDEX,
        },
        "operating_system": {
            "id": EXPECTED_UBUNTU_ID,
            "version_id": EXPECTED_UBUNTU_VERSION,
        },
        "transient_cache_policy": {
            "execution_surface": False,
            "removed_before_receipt_publication": True,
            "root": str(_transient_cache_root(paths)),
        },
        "python": {
            "major": EXPECTED_PYTHON[0],
            "minor": EXPECTED_PYTHON[1],
            "venv_mode": "--copies",
            "system_executable": str(SYSTEM_PYTHON),
        },
    }


def _validate_pinned_inputs(paths: RuntimePaths) -> dict[str, dict[str, Any]]:
    expected = {
        "toolkit_dockerfile": (paths.dockerfile, TOOLKIT_DOCKERFILE_SHA256),
        "runtime_lock": (paths.runtime_lock, RUNTIME_LOCK_SHA256),
        "phase1_constraints": (paths.constraints, PHASE1_CONSTRAINTS_SHA256),
        "runtime_verifier": (paths.verifier, RUNTIME_VERIFIER_SHA256),
        "ai_toolkit_requirements": (
            paths.requirements,
            AI_TOOLKIT_REQUIREMENTS_SHA256,
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    for label, (path, expected_hash) in expected.items():
        identity = _file_identity(path)
        if identity["sha256"] != expected_hash:
            raise Stage1RuntimeError(
                f"{label} drifted: expected={expected_hash} actual={identity['sha256']}"
            )
        result[label] = identity
    result["materializer"] = _file_identity(paths.materializer)
    return result


def _run_once(spec: CommandSpec, *, runner: Runner, environment: dict[str, str]):
    return runner(
        list(spec.argv),
        cwd=spec.cwd,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _result(
    spec: CommandSpec,
    attempts: list[subprocess.CompletedProcess[str]],
    *,
    status: str = "executed",
) -> dict[str, Any]:
    return {
        "attempts": [
            {
                "returncode": item.returncode,
                "stderr": item.stderr,
                "stderr_sha256": hashlib.sha256(item.stderr.encode()).hexdigest(),
                "stdout": item.stdout,
                "stdout_sha256": hashlib.sha256(item.stdout.encode()).hexdigest(),
            }
            for item in attempts
        ],
        "command_id": spec.command_id,
        "status": status,
    }


def _execute(
    spec: CommandSpec,
    *,
    runner: Runner,
    environment: dict[str, str],
) -> dict[str, Any]:
    attempts: list[subprocess.CompletedProcess[str]] = []
    for _attempt in range(spec.attempts):
        completed = _run_once(spec, runner=runner, environment=environment)
        attempts.append(completed)
        if completed.returncode == 0:
            return _result(spec, attempts)
    last = attempts[-1]
    raise Stage1RuntimeError(
        f"{spec.command_id} failed after {len(attempts)} attempt(s): "
        f"exit={last.returncode} stderr={last.stderr.strip()!r}"
    )


def _sole_attempt(result: dict[str, Any], label: str) -> dict[str, Any]:
    attempts = result.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise Stage1RuntimeError(f"{label} has no command output")
    final = _object(attempts[-1], f"{label} final attempt")
    if final.get("returncode") != 0:
        raise Stage1RuntimeError(f"{label} did not succeed")
    return final


def _parse_json_line(text: str, label: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise Stage1RuntimeError(f"{label} must emit exactly one non-empty JSON line")
    try:
        return _object(json.loads(lines[0]), label)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Stage1RuntimeError(f"{label} emitted invalid JSON") from exc


def _system_python_identity(result: dict[str, Any]) -> dict[str, Any]:
    output = _sole_attempt(result, "system Python identity")
    if output["stderr"]:
        raise Stage1RuntimeError("system Python identity emitted stderr")
    identity = _parse_json_line(output["stdout"], "system Python identity")
    _exact(
        identity,
        {
            "cache_tag",
            "executable",
            "implementation",
            "real_executable",
            "soabi",
            "version",
        },
        "system Python identity",
    )
    if identity["executable"] != str(SYSTEM_PYTHON):
        raise Stage1RuntimeError("system Python did not report /usr/bin/python3")
    version = identity["version"]
    if (
        not isinstance(version, list)
        or len(version) != 3
        or version[:2] != list(EXPECTED_PYTHON)
    ):
        raise Stage1RuntimeError("system Python must be CPython 3.10")
    if identity["implementation"] != "cpython":
        raise Stage1RuntimeError("system Python must be CPython")
    if not isinstance(identity["soabi"], str) or not identity["soabi"].startswith(
        "cpython-310-"
    ):
        raise Stage1RuntimeError("system Python SOABI is not CPython 3.10")
    real = Path(identity["real_executable"])
    _regular_file(real, "resolved system Python")
    if not os.access(real, os.X_OK):
        raise Stage1RuntimeError("resolved system Python is not executable")
    identity["binary"] = _file_identity(real)
    identity["requested_path"] = str(SYSTEM_PYTHON)
    identity["requested_path_symlink"] = SYSTEM_PYTHON.is_symlink()
    if SYSTEM_PYTHON.is_symlink():
        identity["requested_path_link_target"] = os.readlink(SYSTEM_PYTHON)
    else:
        identity["requested_path_link_target"] = None
    return identity


def _tracked_worktree_identity(
    root: Path, tracked_text: str, flags_text: str, label: str
) -> tuple[list[dict[str, Any]], str]:
    staged: dict[str, tuple[str, str]] = {}
    for raw in tracked_text.split("\0"):
        if not raw:
            continue
        try:
            prefix, relative = raw.split("\t", 1)
            mode, blob, stage = prefix.split(" ")
        except ValueError as exc:
            raise Stage1RuntimeError(f"{label} tracked manifest is malformed") from exc
        if mode not in {"100644", "100755"}:
            raise Stage1RuntimeError(
                f"{label} has unsupported tracked mode {mode}: {relative}"
            )
        if not _GIT_SHA.fullmatch(blob) or stage != "0":
            raise Stage1RuntimeError(f"{label} tracked manifest has an invalid row")
        relative_path = Path(relative)
        if (
            not relative
            or relative.startswith("/")
            or ".." in relative_path.parts
            or relative in staged
        ):
            raise Stage1RuntimeError(f"{label} tracked manifest has an unsafe row")
        staged[relative] = (mode, blob)

    flags: dict[str, str] = {}
    for raw in flags_text.split("\0"):
        if not raw:
            continue
        if len(raw) < 3 or raw[1] != " ":
            raise Stage1RuntimeError(f"{label} tracked flags are malformed")
        flag, relative = raw[0], raw[2:]
        if flag != "H":
            raise Stage1RuntimeError(
                f"{label} tracked path has hidden/special index flag {flag}: {relative}"
            )
        if relative in flags:
            raise Stage1RuntimeError(f"{label} tracked flags contain a duplicate path")
        flags[relative] = flag
    if set(flags) != set(staged):
        raise Stage1RuntimeError(f"{label} tracked flags/path set mismatch")
    if not staged:
        raise Stage1RuntimeError(f"{label} tracked manifest is empty")

    entries: list[dict[str, Any]] = []
    for relative in sorted(staged):
        mode, expected_blob = staged[relative]
        path = root / relative
        _regular_file(path, f"{label} tracked file")
        before = path.stat()
        contents = path.read_bytes()
        after = path.stat()
        before_key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_key = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_key != after_key:
            raise Stage1RuntimeError(f"{label} tracked file changed while hashing")
        executable = bool(stat.S_IMODE(after.st_mode) & 0o111)
        if executable != (mode == "100755"):
            raise Stage1RuntimeError(f"{label} tracked file mode drifted: {relative}")
        header = f"blob {len(contents)}\0".encode("ascii")
        actual_blob = hashlib.sha1(header + contents).hexdigest()  # noqa: S324
        if actual_blob != expected_blob:
            raise Stage1RuntimeError(
                f"{label} tracked file differs from index/HEAD: {relative}"
            )
        entries.append(
            {
                "bytes": len(contents),
                "git_blob_sha1": actual_blob,
                "mode": mode,
                "path": relative,
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
        )
    return entries, canonical_sha256(entries)


def _repo_identity(results: Mapping[str, dict[str, Any]], paths: RuntimePaths):
    head = _sole_attempt(results["ai-toolkit-head"], "ai-toolkit HEAD")
    tree = _sole_attempt(results["ai-toolkit-tree"], "ai-toolkit tree")
    status = _sole_attempt(results["ai-toolkit-status"], "ai-toolkit status")
    tracked = _sole_attempt(
        results["ai-toolkit-tracked-manifest"], "ai-toolkit tracked manifest"
    )
    flags = _sole_attempt(
        results["ai-toolkit-tracked-flags"], "ai-toolkit tracked flags"
    )
    for label, output in (
        ("HEAD", head),
        ("tree", tree),
        ("status", status),
        ("tracked manifest", tracked),
        ("tracked flags", flags),
    ):
        if output["stderr"]:
            raise Stage1RuntimeError(f"ai-toolkit {label} emitted stderr")
    commit = head["stdout"].strip()
    tree_sha = tree["stdout"].strip()
    if commit != AI_TOOLKIT_COMMIT:
        raise Stage1RuntimeError(
            f"ai-toolkit commit mismatch: expected={AI_TOOLKIT_COMMIT} actual={commit}"
        )
    if not _GIT_SHA.fullmatch(tree_sha):
        raise Stage1RuntimeError("ai-toolkit tree is not a full Git SHA")
    if status["stdout"] != "":
        raise Stage1RuntimeError("ai-toolkit repository must be exactly clean")
    tracked_entries, tracked_sha = _tracked_worktree_identity(
        paths.ai_toolkit_repo,
        tracked["stdout"],
        flags["stdout"],
        "ai-toolkit",
    )
    return {
        "commit": commit,
        "path": str(paths.ai_toolkit_repo),
        "requirements": _file_identity(paths.requirements),
        "status": "clean-including-untracked",
        "tracked_entries": tracked_entries,
        "tracked_entries_sha256": tracked_sha,
        "tree": tree_sha,
    }


def _forge_identity(results: Mapping[str, dict[str, Any]], paths: RuntimePaths):
    head = _sole_attempt(results["forge-head"], "forge HEAD")
    tree = _sole_attempt(results["forge-tree"], "forge tree")
    status = _sole_attempt(results["forge-status"], "forge status")
    tracked = _sole_attempt(results["forge-tracked-manifest"], "forge tracked manifest")
    flags = _sole_attempt(results["forge-tracked-flags"], "forge tracked flags")
    for label, output in (
        ("HEAD", head),
        ("tree", tree),
        ("status", status),
        ("tracked manifest", tracked),
        ("tracked flags", flags),
    ):
        if output["stderr"]:
            raise Stage1RuntimeError(f"forge {label} emitted stderr")
    commit = head["stdout"].strip()
    tree_sha = tree["stdout"].strip()
    if not _GIT_SHA.fullmatch(commit) or not _GIT_SHA.fullmatch(tree_sha):
        raise Stage1RuntimeError("forge commit/tree is not a full Git SHA")
    if status["stdout"] != "":
        raise Stage1RuntimeError("forge repository must be exactly clean")
    entries, tracked_sha = _tracked_worktree_identity(
        paths.forge_repo, tracked["stdout"], flags["stdout"], "forge"
    )
    required = paths.materializer.relative_to(paths.forge_repo).as_posix()
    if required not in {row["path"] for row in entries}:
        raise Stage1RuntimeError("materializer is not tracked by the forge commit")
    return {
        "commit": commit,
        "path": str(paths.forge_repo),
        "status": "clean-including-untracked",
        "tracked_entries": entries,
        "tracked_entries_sha256": tracked_sha,
        "tree": tree_sha,
    }


def _tree_manifest(root: Path) -> dict[str, Any]:
    _directory(root, "materialized venv")
    root_resolved = root.resolve(strict=True)
    entries: list[dict[str, Any]] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if path.is_symlink():
            try:
                resolved = path.resolve(strict=True)
            except (FileNotFoundError, RuntimeError) as exc:
                raise Stage1RuntimeError(
                    f"venv has broken symlink: {relative}"
                ) from exc
            if not resolved.is_relative_to(root_resolved):
                raise Stage1RuntimeError(f"venv symlink escapes tree: {relative}")
            entries.append(
                {
                    "link_target": os.readlink(path),
                    "mode": mode,
                    "path": relative,
                    "resolved_path": resolved.relative_to(root_resolved).as_posix(),
                    "type": "symlink",
                }
            )
        elif path.is_dir():
            entries.append({"mode": mode, "path": relative, "type": "directory"})
        elif path.is_file():
            entries.append(
                {
                    "bytes": metadata.st_size,
                    "mode": mode,
                    "path": relative,
                    "sha256": file_sha256(path),
                    "type": "file",
                }
            )
        else:
            raise Stage1RuntimeError(f"unsupported venv entry: {relative}")
    return {
        "entries": entries,
        "entry_count": len(entries),
        "entries_sha256": canonical_sha256(entries),
        "root": str(root),
    }


def _validate_tree_manifest(
    value: dict[str, Any], *, expected_root: str, label: str
) -> None:
    _exact(value, {"entries", "entry_count", "entries_sha256", "root"}, label)
    entries = value["entries"]
    if not isinstance(entries, list) or value["entry_count"] != len(entries):
        raise ValueError(f"{label} entry count mismatch")
    if _digest(value["entries_sha256"], f"{label} SHA") != canonical_sha256(entries):
        raise ValueError(f"{label} hash mismatch")
    if value["root"] != expected_root:
        raise ValueError(f"{label} root mismatch")


def _validate_probe_outputs(results: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    runtime = _sole_attempt(results["verify-image-runtime"], "runtime verifier")
    if "SN56_IMAGE_RUNTIME_INVENTORY=PASS" not in runtime["stdout"].splitlines():
        raise Stage1RuntimeError("verify_image_runtime did not emit its PASS sentinel")
    if EXPECTED_PIP_CHECK_LINE not in runtime["stdout"]:
        # The verifier emits the allowed conflict inside its JSON summary.
        raise Stage1RuntimeError(
            "runtime verifier did not attest the sole pip conflict"
        )
    essential = _sole_attempt(results["essential-imports"], "essential imports")
    essential_summary = _parse_json_line(essential["stdout"], "essential imports")
    if essential_summary.get("result") != "PASS":
        raise Stage1RuntimeError("essential import probe did not pass")
    gpu = results["gpu-detect"]
    if gpu["status"] == "skipped-no-nvidia-smi":
        if results["cuda-jit-smoke"]["status"] != "skipped-no-gpu":
            raise Stage1RuntimeError("CUDA/JIT probe skip is inconsistent")
        cuda_summary: dict[str, Any] = {
            "gpu_present": False,
            "result": "SKIPPED_NO_GPU",
        }
    else:
        detected = _sole_attempt(gpu, "GPU detection")
        if not detected["stdout"].strip():
            raise Stage1RuntimeError("nvidia-smi reported no GPU")
        cuda = _sole_attempt(results["cuda-jit-smoke"], "CUDA/JIT smoke")
        cuda_summary = _parse_json_line(cuda["stdout"], "CUDA/JIT smoke")
        if cuda_summary.get("result") != "PASS":
            raise Stage1RuntimeError("CUDA/JIT smoke did not pass")
        cuda_summary["gpu_present"] = True
    return {
        "cuda_jit": cuda_summary,
        "essential_imports": essential_summary,
        "pip_check_expected_sole_conflict": EXPECTED_PIP_CHECK_LINE,
        "runtime_verifier_pass": True,
    }


def _inputs(paths: RuntimePaths) -> dict[str, dict[str, Any]]:
    return _validate_pinned_inputs(paths)


def dry_run(
    forge_repo: str | Path,
    ai_toolkit_repo: str | Path,
    destination: str | Path,
    receipt: str | Path,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    paths = _paths(
        forge_repo,
        ai_toolkit_repo,
        destination,
        receipt,
        destination_must_be_absent=True,
        receipt_must_be_absent=True,
    )
    inputs = _inputs(paths)
    os_release = _read_os_release()
    environment = _sanitized_environment(paths)
    preflight_results: dict[str, dict[str, Any]] = {}
    for spec in command_plan(paths):
        if spec.phase != "preflight":
            continue
        preflight_results[spec.command_id] = _execute(
            spec, runner=runner, environment=environment
        )
    python = _system_python_identity(preflight_results["system-python-identity"])
    forge = _forge_identity(preflight_results, paths)
    ai_toolkit = _repo_identity(preflight_results, paths)
    value = {
        "ai_toolkit": ai_toolkit,
        "contract": _contract(paths),
        "forge": forge,
        "host": {"operating_system": os_release, "system_python": python},
        "inputs": inputs,
        "kind": PLAN_KIND,
        "paths": {
            "ai_toolkit_repo": str(paths.ai_toolkit_repo),
            "destination": str(paths.destination),
            "forge_repo": str(paths.forge_repo),
            "receipt": str(paths.receipt),
        },
        "schema": SCHEMA,
    }
    value["plan_sha256"] = canonical_sha256(value)
    return value


def materialize(
    forge_repo: str | Path,
    ai_toolkit_repo: str | Path,
    destination: str | Path,
    receipt: str | Path,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    paths = _paths(
        forge_repo,
        ai_toolkit_repo,
        destination,
        receipt,
        destination_must_be_absent=True,
        receipt_must_be_absent=True,
    )
    before_inputs = _inputs(paths)
    os_release = _read_os_release()
    environment = _sanitized_environment(paths)
    transient_root = _prepare_transient_cache(paths)
    try:
        return _materialize_prepared(
            paths,
            before_inputs,
            os_release,
            environment,
            transient_root,
            runner,
        )
    finally:
        _remove_transient_cache(transient_root)


def _remove_transient_cache(root: Path) -> None:
    if not os.path.lexists(root):
        return
    if root.is_symlink():
        raise Stage1RuntimeError("transient cache root became a symlink; not removing")
    _reject_symlink_ancestors(root, "transient materialization cache")
    shutil.rmtree(root)
    if os.path.lexists(root):
        raise Stage1RuntimeError("transient materialization cache cleanup failed")


def _materialize_prepared(
    paths: RuntimePaths,
    before_inputs: dict[str, dict[str, Any]],
    os_release: dict[str, Any],
    environment: dict[str, str],
    transient_root: Path,
    runner: Runner,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    specs = command_plan(paths)
    gpu_present = False
    for spec in specs:
        if spec.command_id == "gpu-detect" and not NVIDIA_SMI.is_file():
            result = _result(spec, [], status="skipped-no-nvidia-smi")
        elif spec.command_id == "cuda-jit-smoke" and not gpu_present:
            result = _result(spec, [], status="skipped-no-gpu")
        else:
            result = _execute(spec, runner=runner, environment=environment)
        if spec.command_id == "gpu-detect" and result["status"] == "executed":
            gpu_present = bool(_sole_attempt(result, "GPU detection")["stdout"].strip())
            if not gpu_present:
                raise Stage1RuntimeError("nvidia-smi exists but reported no GPU")
        results.append(result)
        by_id[spec.command_id] = result
        if spec.command_id == "venv-create":
            if not os.path.lexists(paths.venv_python):
                raise Stage1RuntimeError("venv did not create bin/python")
            try:
                resolved_python = paths.venv_python.resolve(strict=True)
                resolved_venv = paths.destination.resolve(strict=True)
            except (FileNotFoundError, RuntimeError) as exc:
                raise Stage1RuntimeError("venv Python is broken or cyclic") from exc
            if not resolved_python.is_relative_to(resolved_venv):
                raise Stage1RuntimeError("venv Python resolves outside the venv")
            _regular_file(resolved_python, "resolved venv Python")

    python = _system_python_identity(by_id["system-python-identity"])
    forge_before = _forge_identity(by_id, paths)
    ai_toolkit_before = _repo_identity(by_id, paths)
    forge_after = _forge_identity_fresh(paths, runner, environment)
    ai_toolkit_after = _repo_identity_fresh(paths, runner, environment)
    if forge_after != forge_before:
        raise Stage1RuntimeError("forge identity changed during materialization")
    if ai_toolkit_after != ai_toolkit_before:
        raise Stage1RuntimeError("ai-toolkit identity changed during materialization")
    after_inputs = _inputs(paths)
    if after_inputs != before_inputs:
        raise Stage1RuntimeError("runtime input files changed during materialization")
    verification = _validate_probe_outputs(by_id)
    tree = _tree_manifest(paths.destination)
    transient_tree = _tree_manifest(transient_root)
    _remove_transient_cache(transient_root)
    value = {
        "ai_toolkit": ai_toolkit_after,
        "command_results": results,
        "contract": _contract(paths),
        "forge": forge_after,
        "host": {"operating_system": os_release, "system_python": python},
        "inputs": after_inputs,
        "kind": KIND,
        "paths": {
            "ai_toolkit_repo": str(paths.ai_toolkit_repo),
            "destination": str(paths.destination),
            "forge_repo": str(paths.forge_repo),
            "receipt": str(paths.receipt),
        },
        "schema": SCHEMA,
        "tree_manifest": tree,
        "transient_materialization_cache": {
            "removed": True,
            "tree_manifest_before_removal": transient_tree,
        },
        "verification": verification,
    }
    value["receipt_sha256"] = canonical_sha256(value)
    _publish(paths.receipt, value)
    return value


def _repo_identity_fresh(
    paths: RuntimePaths, runner: Runner, environment: dict[str, str]
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    for spec in command_plan(paths):
        if not spec.command_id.startswith("ai-toolkit-"):
            continue
        results[spec.command_id] = _execute(
            spec, runner=runner, environment=environment
        )
    return _repo_identity(results, paths)


def _forge_identity_fresh(
    paths: RuntimePaths, runner: Runner, environment: dict[str, str]
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    for spec in command_plan(paths):
        if not spec.command_id.startswith("forge-"):
            continue
        results[spec.command_id] = _execute(
            spec, runner=runner, environment=environment
        )
    return _forge_identity(results, paths)


def _publish(path: Path, value: dict[str, Any]) -> None:
    _reject_symlink_ancestors(path, "receipt output")
    payload = canonical_bytes(value) + b"\n"
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


def _command_results(value: Any, plan: tuple[CommandSpec, ...]) -> None:
    if not isinstance(value, list) or len(value) != len(plan):
        raise ValueError("command_results must cover the exact command plan")
    for result, spec in zip(value, plan, strict=True):
        result = _object(result, f"command result {spec.command_id}")
        _exact(result, {"attempts", "command_id", "status"}, "command result")
        if result["command_id"] != spec.command_id:
            raise ValueError("command_results order/identity drifted")
        if result["status"] not in {
            "executed",
            "skipped-no-gpu",
            "skipped-no-nvidia-smi",
        }:
            raise ValueError("unsupported command result status")
        attempts = result["attempts"]
        if not isinstance(attempts, list) or len(attempts) > spec.attempts:
            raise ValueError("command result attempt count drifted")
        if result["status"] == "executed" and not attempts:
            raise ValueError("executed command has no attempts")
        if result["status"] != "executed" and attempts:
            raise ValueError("skipped command must not have attempts")
        for attempt in attempts:
            attempt = _object(attempt, "command attempt")
            _exact(
                attempt,
                {
                    "returncode",
                    "stderr",
                    "stderr_sha256",
                    "stdout",
                    "stdout_sha256",
                },
                "command attempt",
            )
            if not isinstance(attempt["returncode"], int):
                raise ValueError("command returncode must be an integer")
            for stream in ("stdout", "stderr"):
                if not isinstance(attempt[stream], str):
                    raise ValueError(f"command {stream} must be text")
                if hashlib.sha256(attempt[stream].encode()).hexdigest() != _digest(
                    attempt[f"{stream}_sha256"], f"{stream} SHA"
                ):
                    raise ValueError(f"command {stream} hash mismatch")
        if attempts and attempts[-1]["returncode"] != 0:
            raise ValueError("receipt contains a failed final command")


def validate_receipt(
    value: dict[str, Any], *, recapture: bool = False, runner: Runner = subprocess.run
) -> dict[str, Any]:
    value = _object(value, "Stage-1 receipt")
    _exact(
        value,
        {
            "ai_toolkit",
            "command_results",
            "contract",
            "forge",
            "host",
            "inputs",
            "kind",
            "paths",
            "receipt_sha256",
            "schema",
            "tree_manifest",
            "transient_materialization_cache",
            "verification",
        },
        "Stage-1 receipt",
    )
    if value["schema"] != SCHEMA or value["kind"] != KIND:
        raise ValueError("unsupported Stage-1 receipt")
    expected_receipt_sha = canonical_sha256(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )
    if _digest(value["receipt_sha256"], "receipt SHA") != expected_receipt_sha:
        raise ValueError("Stage-1 receipt self-hash mismatch")
    path_values = _object(value["paths"], "receipt paths")
    _exact(
        path_values,
        {"ai_toolkit_repo", "destination", "forge_repo", "receipt"},
        "receipt paths",
    )
    paths = RuntimePaths(
        forge_repo=_normalized_absolute(path_values["forge_repo"], "forge repo"),
        ai_toolkit_repo=_normalized_absolute(
            path_values["ai_toolkit_repo"], "ai-toolkit repo"
        ),
        destination=_normalized_absolute(path_values["destination"], "destination"),
        receipt=_normalized_absolute(path_values["receipt"], "receipt"),
    )
    if value["contract"] != _contract(paths):
        raise ValueError("Stage-1 command/contract drift")
    plan = command_plan(paths)
    _command_results(value["command_results"], plan)
    tree = _object(value["tree_manifest"], "tree manifest")
    _validate_tree_manifest(
        tree, expected_root=str(paths.destination), label="tree manifest"
    )
    transient = _object(
        value["transient_materialization_cache"], "transient materialization cache"
    )
    _exact(
        transient,
        {"removed", "tree_manifest_before_removal"},
        "transient materialization cache",
    )
    if transient["removed"] is not True:
        raise ValueError("transient materialization cache must be removed")
    transient_tree = _object(
        transient["tree_manifest_before_removal"], "transient cache tree manifest"
    )
    _validate_tree_manifest(
        transient_tree,
        expected_root=str(_transient_cache_root(paths)),
        label="transient cache tree manifest",
    )
    if recapture:
        live_paths = _paths(
            paths.forge_repo,
            paths.ai_toolkit_repo,
            paths.destination,
            paths.receipt,
            destination_must_be_absent=False,
            receipt_must_be_absent=False,
        )
        if _inputs(live_paths) != value["inputs"]:
            raise Stage1RuntimeError("Stage-1 input files drifted")
        if _tree_manifest(live_paths.destination) != tree:
            raise Stage1RuntimeError("materialized venv tree drifted")
        if os.path.lexists(_transient_cache_root(live_paths)):
            raise Stage1RuntimeError("transient materialization cache reappeared")
        environment = _sanitized_environment(live_paths)
        repo = _repo_identity_fresh(live_paths, runner, environment)
        if repo != value["ai_toolkit"]:
            raise Stage1RuntimeError("ai-toolkit repository drifted")
        forge = _forge_identity_fresh(live_paths, runner, environment)
        if forge != value["forge"]:
            raise Stage1RuntimeError("forge repository drifted")
        os_release = _read_os_release()
        if os_release != value["host"]["operating_system"]:
            raise Stage1RuntimeError("host operating-system identity drifted")
        python_spec = command_plan(live_paths)[0]
        python_result = _execute(python_spec, runner=runner, environment=environment)
        if _system_python_identity(python_result) != value["host"]["system_python"]:
            raise Stage1RuntimeError("system Python identity drifted")
    return value


def load_receipt(path: Path) -> dict[str, Any]:
    _regular_file(path, "receipt")
    raw = path.read_bytes()
    try:
        value = _object(json.loads(raw), "receipt")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("receipt is not JSON") from exc
    if raw != canonical_bytes(value) + b"\n":
        raise ValueError("receipt must be canonical JSON plus one newline")
    return value


def probe_essential(ai_toolkit_repo: Path) -> dict[str, Any]:
    repo = _directory(ai_toolkit_repo, "ai-toolkit repo")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import bitsandbytes  # type: ignore[import-not-found]  # noqa: F401
    import diffusers  # type: ignore[import-not-found]
    import numpy  # type: ignore[import-not-found]  # noqa: F401
    import PIL  # type: ignore[import-not-found]  # noqa: F401
    import safetensors  # type: ignore[import-not-found]  # noqa: F401
    import torch  # type: ignore[import-not-found]
    import torchaudio  # type: ignore[import-not-found]  # noqa: F401
    import torchcodec  # type: ignore[import-not-found]  # noqa: F401
    import torchvision  # type: ignore[import-not-found]  # noqa: F401
    import transformers  # type: ignore[import-not-found]
    import yaml  # type: ignore[import-not-found]  # noqa: F401
    import toolkit  # type: ignore[import-not-found]  # noqa: F401

    if torch.__version__ != "2.6.0+cu124" or torch.version.cuda != "12.4":
        raise Stage1RuntimeError("essential import probe loaded the wrong Torch graph")
    return {
        "diffusers": diffusers.__version__,
        "result": "PASS",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
    }


def probe_cuda_jit() -> dict[str, Any]:
    import torch  # type: ignore[import-not-found]

    if not torch.cuda.is_available():
        raise Stage1RuntimeError("CUDA is not available")

    def arithmetic(tensor):
        return tensor.square().add(3.0)

    compiled = torch.compile(arithmetic, backend="inductor", fullgraph=True)
    source = torch.arange(1024, dtype=torch.float32, device="cuda")
    expected = arithmetic(source)
    actual = compiled(source)
    torch.cuda.synchronize()
    if not torch.equal(actual, expected):
        raise Stage1RuntimeError("CUDA/JIT output mismatch")
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "result": "PASS",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--forge-repo", required=True, type=Path)
    parser.add_argument("--ai-toolkit-repo", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    dry_parser = subparsers.add_parser("dry-run")
    _common_arguments(dry_parser)
    materialize_parser = subparsers.add_parser("materialize")
    _common_arguments(materialize_parser)
    validate_parser = subparsers.add_parser("validate-receipt")
    validate_parser.add_argument("--receipt", required=True, type=Path)
    validate_parser.add_argument("--structure-only", action="store_true")
    essential_parser = subparsers.add_parser("probe-essential")
    essential_parser.add_argument("--ai-toolkit-repo", required=True, type=Path)
    subparsers.add_parser("probe-cuda-jit")
    args = parser.parse_args(argv)

    try:
        if args.command == "dry-run":
            result = dry_run(
                args.forge_repo,
                args.ai_toolkit_repo,
                args.destination,
                args.receipt,
            )
        elif args.command == "materialize":
            result = materialize(
                args.forge_repo,
                args.ai_toolkit_repo,
                args.destination,
                args.receipt,
            )
        elif args.command == "validate-receipt":
            result = validate_receipt(
                load_receipt(args.receipt), recapture=not args.structure_only
            )
        elif args.command == "probe-essential":
            result = probe_essential(args.ai_toolkit_repo)
        else:
            result = probe_cuda_jit()
    except (OSError, ValueError, Stage1RuntimeError) as error:
        print(f"SN56_KREA_STAGE1_RUNTIME=FAIL: {error}", file=sys.stderr)
        return 1
    print(canonical_bytes(result).decode("ascii"))
    print("SN56_KREA_STAGE1_RUNTIME=PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
