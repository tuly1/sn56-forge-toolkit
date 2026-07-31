"""Load the one admitted accelerated-discovery validator fail-closed.

Scorer-only fixes must not require re-emitting otherwise valid training
evidence.  This module loads the complete validator graph from the exact clean
c9f30b1 accelerated-discovery successor worktree under an isolated package
name. It
never treats current scorer code as an equivalent training validator and it
has no generic commit allowlist.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import os
import subprocess
import sys
from types import ModuleType
from typing import Any


_COMMIT = "c9f30b14de5358a5fd8e3c2e23a8e6427c2fdb1d"
_TREE = "c2d106ebc9165768ac0dcd3b3d6686056fe5a8c2"
_POLICY_SHA256 = "98b59fd90dbf4ea213c860f873bc472cadc66714c7b9118672de2474f020f5f3"
_MODULE_SHA256 = {
    "krea_training_evidence.py": (
        "68302d97044b47392aebf432f9fdc456e8832de488db56b8157940b927855c3b"
    ),
    "krea_execution_plan.py": (
        "52d4d25eeeaff0d10ab5f0c7939d70471e5ca8d19e2d32865624e604dc501dc8"
    ),
    "krea_execution_surface_policy.py": (
        "29b92928aed6adc5d9d7f59207610f673845e3ab4a196debcca9aba654c786ac"
    ),
    "krea_fixture_admission.py": (
        "2c11b08e03e89c22d0b66530379414916e01c9468a6973c4acda641848a57f11"
    ),
    "krea_runtime_binding.py": (
        "af7a95154eb00c4653bc9ecd1cef465fea69cfc5fd7912126a07573360f2f250"
    ),
    "run_krea_ladder.py": (
        "29402cfda15dee33d69d5a258384fdbfe764386d31aea10e80c8f488b5448593"
    ),
}
_ALIAS = "_forge_krea_training_validator_c9f30b1"
_LOADED_ROOT: Path | None = None


def _git(root: Path, *arguments: str) -> bytes:
    environment = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    try:
        return subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.autocrlf=false",
                "-C",
                str(root),
                *arguments,
            ],
            check=True,
            capture_output=True,
            timeout=30,
            env=environment,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("historical validator Git identity is unreadable") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_identity(root: Path | str) -> dict[str, Any]:
    """Recompute the exact admitted historical worktree identity."""

    candidate = Path(root).expanduser()
    if not candidate.is_absolute():
        raise ValueError("historical validator root must be absolute")
    root_path = candidate.resolve(strict=True)
    if not root_path.is_dir() or candidate.is_symlink():
        raise ValueError("historical validator root must be a real directory")
    try:
        top = Path(
            _git(root_path, "rev-parse", "--show-toplevel")
            .rstrip(b"\n")
            .decode("utf-8")
        ).resolve(strict=True)
        commit = (
            _git(root_path, "rev-parse", "--verify", "HEAD^{commit}")
            .strip()
            .decode("ascii")
        )
        tree = (
            _git(root_path, "rev-parse", "--verify", "HEAD^{tree}")
            .strip()
            .decode("ascii")
        )
    except (UnicodeDecodeError, OSError) as exc:
        raise ValueError("historical validator Git identity is malformed") from exc
    status = _git(
        root_path,
        "status",
        "--porcelain=v2",
        "-z",
        "--untracked-files=all",
    )
    if top != root_path or commit != _COMMIT or tree != _TREE or status:
        raise ValueError(
            "historical validator must be the exact clean c9f30b1 worktree"
        )
    calibration = root_path / "ops" / "calibration"
    observed = {
        name: _file_sha256(calibration / name) for name in sorted(_MODULE_SHA256)
    }
    if observed != _MODULE_SHA256:
        raise ValueError("historical training-validator modules differ from c9f30b1")
    return {
        "schema": 1,
        "kind": "forge-krea-historical-training-evidence-validator",
        "root": str(root_path),
        "commit_sha1": _COMMIT,
        "tree_sha1": _TREE,
        "execution_surface_policy_sha256": _POLICY_SHA256,
        "module_sha256": dict(_MODULE_SHA256),
    }


def validate_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "kind",
        "root",
        "commit_sha1",
        "tree_sha1",
        "execution_surface_policy_sha256",
        "module_sha256",
    }:
        raise ValueError("historical training-validator identity is malformed")
    observed = capture_identity(value.get("root", ""))
    if value != observed:
        raise ValueError("historical training-validator identity drifted")
    return observed


def _namespace(name: str, path: Path) -> ModuleType:
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


def load_modules(identity: Any) -> dict[str, ModuleType]:
    """Load legacy modules without contaminating the current package graph."""

    global _LOADED_ROOT
    normalized = validate_identity(identity)
    root = Path(normalized["root"])
    if _LOADED_ROOT is not None and _LOADED_ROOT != root:
        raise ValueError("one process cannot switch historical validator roots")
    if _ALIAS not in sys.modules:
        _namespace(_ALIAS, root)
        _namespace(f"{_ALIAS}.ops", root / "ops")
        _namespace(f"{_ALIAS}.ops.calibration", root / "ops" / "calibration")
        _LOADED_ROOT = root
    modules = {
        "batch_evaluate": importlib.import_module(
            f"{_ALIAS}.ops.calibration.batch_evaluate_krea"
        ),
        "training_evidence": importlib.import_module(
            f"{_ALIAS}.ops.calibration.krea_training_evidence"
        ),
        "execution_plan": importlib.import_module(
            f"{_ALIAS}.ops.calibration.krea_execution_plan"
        ),
        "discovery_authorization": importlib.import_module(
            f"{_ALIAS}.ops.calibration.krea_discovery_authorization"
        ),
        "delegated_review_contract": importlib.import_module(
            f"{_ALIAS}.ops.calibration.krea_delegated_review_contract"
        ),
        "execution_surface_policy": importlib.import_module(
            f"{_ALIAS}.ops.calibration.krea_execution_surface_policy"
        ),
        "fixture": importlib.import_module(
            f"{_ALIAS}.ops.calibration.krea_fixture"
        ),
        "fixture_admission": importlib.import_module(
            f"{_ALIAS}.ops.calibration.krea_fixture_admission"
        ),
    }
    policy = modules["execution_surface_policy"].POLICY
    if policy.get("policy_sha256") != _POLICY_SHA256:
        raise ValueError("loaded historical execution policy is incompatible")
    return modules


def validate_run_evidence(bundle_path: Path, identity: Any) -> dict[str, Any]:
    modules = load_modules(identity)
    return modules["training_evidence"].validate_run_evidence(bundle_path)


def execution_plan_module(identity: Any) -> ModuleType:
    return load_modules(identity)["execution_plan"]
