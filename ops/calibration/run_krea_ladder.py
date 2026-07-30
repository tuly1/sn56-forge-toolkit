#!/usr/bin/env python3
"""Run one fail-closed Krea2 calibration condition.

This is deliberately stricter than the tournament entry point.  A condition is
accepted only when it starts from empty mutable paths, completes the requested
depth without fallback, scores every current-attempt checkpoint, and matches a
durable campaign envelope.  The envelope makes LR/depth/guidance (plus the save
cadence derived from depth) the only scientific differences between conditions.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import select
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any


_MODEL = "krea/Krea-2-Raw"
_KREA_TEXT_ENCODER = "/cache/hf_cache/Qwen--Qwen3-VL-4B-Instruct"
_CONDITION_NAME = "forge_calibration_condition.json"
_BASELINE_NAME = "krea_ladder_baseline.json"
_REEXEC_MARKER = "FORGE_KREA_CALIBRATION_SEEDED"
_TRUSTED_LAUNCH_MARKER = "FORGE_KREA_STAGE1_TRUSTED_REEXEC"
_RUNTIME_CACHE_ROOT = Path("/cache/krea-runtime")
_RUNTIME_CACHE_ENV = {
    "HOME": "home",
    "XDG_CACHE_HOME": "xdg",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TRITON_CACHE_DIR": "triton",
}
_RUNTIME_CACHE_NAMESPACE_ENV = "FORGE_KREA_RUNTIME_CACHE_NAMESPACE"
_STAGED_VENV_ROOT = Path("/app/venv")
_CAMPAIGN_ROOT = Path("/campaign")
_CONTROL_ROOT = _CAMPAIGN_ROOT / "controls"
_CHECKPOINT_ROOT = Path("/app/checkpoints")
_ISOLATED_MODULE_BOOTSTRAP = (
    "import runpy,sys;sys.path.insert(0,'/app/forge');"
    "runpy.run_module('ops.calibration.run_krea_ladder',run_name='__main__')"
)
_PROBE_SEED = 42565431
_PROBE_EPOCHS = 2
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")

_LR_POINTER = "/config/process/0/train/lr"
_DEPTH_POINTER = "/config/process/0/train/steps"
_GUIDANCE_POINTER = "/config/process/0/train/do_differential_guidance"
_GUIDANCE_SCALE_POINTER = "/config/process/0/train/differential_guidance_scale"
_RANK_POINTER = "/config/process/0/network/linear"
_ALPHA_POINTER = "/config/process/0/network/linear_alpha"
_OPTIMIZER_POINTER = "/config/process/0/train/optimizer"
_OPTIMIZER_PARAMETERS_POINTER = "/config/process/0/train/optimizer_params"
_LOSS_POINTER = "/config/process/0/train/loss_type"
_SCHEDULER_POINTER = "/config/process/0/train/noise_scheduler"
_DROPOUT_POINTER = "/config/process/0/datasets/0/caption_dropout_rate"
_BATCH_POINTER = "/config/process/0/train/batch_size"
_ACCUMULATION_POINTER = "/config/process/0/train/gradient_accumulation"
_EMA_POINTER = "/config/process/0/train/ema_config"
_SAVE_CADENCE_POINTER = "/config/process/0/save/save_every"
_TRAINING_SEED_POINTER = "/config/process/0/training_seed"
_RUN_NAME_POINTER = "/config/name"
_TRAINING_FOLDER_POINTER = "/config/process/0/training_folder"

_SCIENTIFIC_AXIS_POINTERS = (
    _LR_POINTER,
    _DEPTH_POINTER,
    _GUIDANCE_POINTER,
    _GUIDANCE_SCALE_POINTER,
    _RANK_POINTER,
    _ALPHA_POINTER,
    _OPTIMIZER_POINTER,
    _OPTIMIZER_PARAMETERS_POINTER,
    _LOSS_POINTER,
    _SCHEDULER_POINTER,
    _DROPOUT_POINTER,
    _BATCH_POINTER,
    _ACCUMULATION_POINTER,
    _EMA_POINTER,
)
_DERIVED_POINTERS = (_SAVE_CADENCE_POINTER,)
_ISOLATION_POINTERS = (_RUN_NAME_POINTER, _TRAINING_FOLDER_POINTER)
_ALLOWED_BUILDER_MUTATIONS = frozenset(
    (*_SCIENTIFIC_AXIS_POINTERS, *_DERIVED_POINTERS, _TRAINING_SEED_POINTER)
)

_TIMING_ENV = (
    "FORGE_KREA_TIMING_SOCKET",
    "FORGE_KREA_TIMING_PROBE_CONTRACT_SHA256",
    "FORGE_KREA_TIMING_CAPTURE_ID",
)


def _minimal_runtime_environment(
    *,
    timing: dict[str, str] | None = None,
    cache: dict[str, str] | None = None,
    trusted_marker: str | None = None,
    jit_enabled: str | None = None,
    seed: str | None = None,
) -> dict[str, str]:
    """Construct, rather than inherit, the Stage-1 runtime environment."""

    environment = {
        "PATH": "/app/venv/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "AI_TOOLKIT_DIR": "/app/ai-toolkit",
        "FORGE_TEMPLATES_DIR": "/app/forge/forge/templates",
        "FORGE_HOLDOUT_SELECTION_TYPES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if timing:
        if set(timing) != set(_TIMING_ENV) or any(
            not isinstance(value, str) or not value for value in timing.values()
        ):
            raise RuntimeError("timing environment is incomplete or non-canonical")
        environment.update(timing)
    if cache:
        expected_cache_keys = set(_RUNTIME_CACHE_ENV) | {_RUNTIME_CACHE_NAMESPACE_ENV}
        if set(cache) != expected_cache_keys or any(
            not isinstance(value, str) or not value for value in cache.values()
        ):
            raise RuntimeError("runtime-cache environment is incomplete")
        environment.update(cache)
    if trusted_marker is not None:
        if not _SHA256.fullmatch(trusted_marker):
            raise RuntimeError("trusted launch marker is invalid")
        environment[_TRUSTED_LAUNCH_MARKER] = trusted_marker
    if jit_enabled is not None:
        if jit_enabled not in {"0", "1"}:
            raise RuntimeError("JIT environment is invalid")
        environment["FORGE_CALIBRATION_JIT_ENABLED"] = jit_enabled
    if seed is not None:
        if not seed.isdecimal() or not 0 <= int(seed) < 2**32:
            raise RuntimeError("seed environment is invalid")
        environment["PYTHONHASHSEED"] = seed
        environment["SEED"] = seed
        environment[_REEXEC_MARKER] = seed
    return environment


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one clean-room Krea2 ladder condition."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execution-plan", type=Path)
    mode.add_argument("--timing-probe-plan", type=Path)
    parser.add_argument("--execution-approval", type=Path)
    parser.add_argument("--timing-probe-approval", type=Path)
    parser.add_argument(
        "--campaign-dir",
        required=True,
        type=Path,
        help=(
            "shared durable directory containing the immutable campaign "
            "baseline and one record per condition"
        ),
    )
    parser.add_argument("--model", choices=(_MODEL,), default=_MODEL)
    return parser.parse_args()


def _authorized_child_path(value: Path, root: Path, label: str) -> Path:
    """Resolve one operator path and keep it inside its authorized mount."""

    path = value.expanduser().resolve()
    authorized_root = root.resolve(strict=True)
    if path == authorized_root or not path.is_relative_to(authorized_root):
        raise ValueError(f"{label} must be below {authorized_root}: {path}")
    return path


def _validate_execution_paths(args: argparse.Namespace) -> None:
    """Reject root-run control/output paths outside the fixed Stage-1 mounts."""

    _authorized_child_path(args.campaign_dir, _CAMPAIGN_ROOT, "campaign-dir")
    if args.execution_plan is not None:
        _authorized_child_path(args.execution_plan, _CONTROL_ROOT, "execution plan")
    if args.execution_approval is not None:
        _authorized_child_path(
            args.execution_approval, _CONTROL_ROOT, "execution approval"
        )
    if args.timing_probe_plan is not None:
        _authorized_child_path(
            args.timing_probe_plan, _CONTROL_ROOT, "timing probe plan"
        )
    if args.timing_probe_approval is not None:
        _authorized_child_path(
            args.timing_probe_approval, _CONTROL_ROOT, "timing probe approval"
        )


def _preimport_tree_identity(root: Path) -> dict[str, Any]:
    """Recompute staged-venv bytes without importing or executing them."""

    root = root.resolve(strict=True)
    rows: list[list[Any]] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode & 0o7777
        if path.is_symlink():
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise RuntimeError(f"staged venv symlink escapes tree: {relative}")
            rows.append(
                [
                    relative,
                    "symlink",
                    mode,
                    os.readlink(path),
                    resolved.relative_to(root).as_posix(),
                ]
            )
        elif path.is_dir():
            rows.append([relative, "directory", mode])
        elif path.is_file():
            rows.append(
                [relative, "file", mode, path.stat().st_size, _sha256_file(path)]
            )
        else:
            raise RuntimeError(f"staged venv contains unsupported entry: {relative}")
    return {"entry_count": len(rows), "manifest_sha256": _canonical_hash(rows)}


def _runtime_cache_environment(
    plan_file_sha256: str, *, timing_capture_id: str | None, create: bool
) -> dict[str, str]:
    """Return one clean, plan-scoped compiler/cache namespace.

    A namespace left by an earlier attempt is evidence of prior execution and
    fails closed. It is never silently warmed or deleted.
    """

    if not _SHA256.fullmatch(plan_file_sha256):
        raise RuntimeError("runtime cache namespace lacks a plan file SHA-256")
    root = _RUNTIME_CACHE_ROOT
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("sealed runtime cache root is absent or a symlink")
    metadata = root.stat()
    if (
        metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o7777 != 0o700
        or not os.access(root, os.R_OK | os.W_OK | os.X_OK)
    ):
        raise RuntimeError("sealed runtime cache root is not protected and writable")
    if timing_capture_id is not None:
        if not _SAFE_COMPONENT.fullmatch(timing_capture_id):
            raise RuntimeError("runtime cache timing capture id is unsafe")
        namespace_id = _canonical_hash(
            {
                "plan_file_sha256": plan_file_sha256,
                "timing_capture_id": timing_capture_id,
            }
        )
    else:
        namespace_id = plan_file_sha256
    namespace = root / namespace_id
    if create:
        if os.path.lexists(namespace):
            raise RuntimeError(
                "runtime cache namespace already exists; cross-plan/attempt reuse is forbidden"
            )
        namespace.mkdir(mode=0o700)
        for leaf in _RUNTIME_CACHE_ENV.values():
            (namespace / leaf).mkdir(mode=0o700)
    if namespace.is_symlink() or not namespace.is_dir():
        raise RuntimeError("runtime cache namespace is absent or unsafe")
    expected = {
        name: str(namespace / leaf) for name, leaf in _RUNTIME_CACHE_ENV.items()
    }
    expected[_RUNTIME_CACHE_NAMESPACE_ENV] = str(namespace)
    for path_text in expected.values():
        path = Path(path_text)
        if (
            path.is_symlink()
            or not path.is_dir()
            or path.stat().st_uid != os.geteuid()
            or path.stat().st_mode & 0o7777 != 0o700
            or not path.is_relative_to(root)
        ):
            raise RuntimeError("runtime cache namespace leaf is unsafe")
    return expected


def _preimport_canonical_json(
    path: Path, expected_sha: str | None = None
) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha is not None and hashlib.sha256(raw).hexdigest() != expected_sha:
        raise RuntimeError(f"pre-import control digest drifted: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise RuntimeError(f"pre-import control is not canonical JSON: {path}")
    return value


def _preimport_executable_identity(path: Path) -> dict[str, Any]:
    requested = path
    requested_stat = requested.lstat()
    resolved = requested.resolve(strict=True)
    resolved_stat = resolved.stat()
    if (
        requested_stat.st_uid != 0
        or requested_stat.st_mode & 0o022
        or not stat.S_ISREG(resolved_stat.st_mode)
        or resolved_stat.st_uid != 0
        or resolved_stat.st_mode & 0o022
        or not os.access(resolved, os.X_OK)
        or not resolved.is_relative_to(Path("/usr"))
    ):
        raise RuntimeError(f"untrusted system executable: {requested}")
    return {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        "sha256": _sha256_file(resolved),
        "mode": resolved_stat.st_mode & 0o7777,
        "uid": resolved_stat.st_uid,
    }


def _preimport_git_identity(path: Path, expected_commit: str) -> dict[str, Any]:
    git = "/usr/bin/git"
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

    def command(*arguments: str) -> str:
        return subprocess.run(
            [
                git,
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

    commit = command("rev-parse", "HEAD")
    if commit != expected_commit or command(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError(f"pre-import source tree drifted: {path}")
    flags = command("ls-files", "-v").splitlines()
    if any(row and (row[0].islower() or row[0] == "S") for row in flags):
        raise RuntimeError(f"pre-import source tree hides tracked changes: {path}")
    tracked = [name for name in command("ls-files", "-z").split("\x00") if name]
    manifest = []
    for name in tracked:
        file_path = path / name
        if file_path.is_symlink() or not file_path.is_file():
            raise RuntimeError(
                f"pre-import tracked entry is not a regular file: {file_path}"
            )
        manifest.append([name, file_path.stat().st_size, _sha256_file(file_path)])
    return {
        "commit": commit,
        "tree": command("rev-parse", "HEAD^{tree}"),
        "tracked_file_count": len(manifest),
        "worktree_manifest_sha256": _canonical_hash(manifest),
    }


def _trusted_stage1_reexec(args: argparse.Namespace) -> None:
    """Verify receipt/source/venv under system Python, then exec the sealed venv."""

    marker = os.environ.get(_TRUSTED_LAUNCH_MARKER)
    forbidden = {
        name: os.environ[name]
        for name in ("LD_PRELOAD", "PYTHONHOME", "PYTHONPATH", "CUDA_VISIBLE_DEVICES")
        if os.environ.get(name)
    }
    if not marker:
        for name in (
            "XDG_CACHE_HOME",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            _RUNTIME_CACHE_NAMESPACE_ENV,
        ):
            if os.environ.get(name):
                forbidden[name] = os.environ[name]
    supplied_ai_toolkit = os.environ.get("AI_TOOLKIT_DIR")
    if supplied_ai_toolkit not in {None, "", "/app/ai-toolkit"}:
        forbidden["AI_TOOLKIT_DIR"] = supplied_ai_toolkit
    supplied_templates = os.environ.get("FORGE_TEMPLATES_DIR")
    if supplied_templates not in {None, "", "/app/forge/forge/templates"}:
        forbidden["FORGE_TEMPLATES_DIR"] = supplied_templates
    if forbidden:
        raise RuntimeError(
            f"operator environment contains unsealed execution controls: {sorted(forbidden)}"
        )
    # Forge reads this variable at import time.  Install the bootstrap-bound
    # mount before any project import and preserve it through both re-execs.
    os.environ["AI_TOOLKIT_DIR"] = "/app/ai-toolkit"
    os.environ["FORGE_TEMPLATES_DIR"] = "/app/forge/forge/templates"
    plan_path = (args.timing_probe_plan or args.execution_plan).resolve(strict=True)
    plan = _preimport_canonical_json(plan_path)
    plan_file_sha256 = _sha256_file(plan_path)
    binding = plan.get("host_execution_manifest")
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise RuntimeError("Stage-1 plan lacks an exact host-manifest binding")
    host = _preimport_canonical_json(Path(binding["path"]), binding["sha256"])
    receipt_binding = host.get("bootstrap_receipt")
    if not isinstance(receipt_binding, dict):
        raise RuntimeError("Stage-1 host manifest lacks a bootstrap receipt")
    receipt_path = Path(receipt_binding["path"])
    receipt = _preimport_canonical_json(receipt_path, receipt_binding["file_sha256"])
    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if receipt.get("receipt_sha256") != _canonical_hash(receipt_body):
        raise RuntimeError("bootstrap receipt semantic digest drifted")
    spec = receipt.get("spec")
    if not isinstance(spec, dict):
        raise RuntimeError("bootstrap receipt lacks its spec")
    spec_body = {key: value for key, value in spec.items() if key != "spec_sha256"}
    if spec.get("spec_sha256") != _canonical_hash(spec_body):
        raise RuntimeError("bootstrap spec semantic digest drifted")
    expected_cache_policy = {
        "root": str(_RUNTIME_CACHE_ROOT),
        "namespace_derivation": (
            "timing_plan_file_sha256_plus_capture_id_or_execution_plan_file_sha256"
        ),
        "initial_state": "root-empty-before-bootstrap",
        "cross_capture_or_plan_reuse": False,
        "within_process_reuse": True,
    }
    runtime_spec = spec.get("runtime")
    if (
        not isinstance(runtime_spec, dict)
        or runtime_spec.get("runtime_cache_policy") != expected_cache_policy
    ):
        raise RuntimeError("bootstrap receipt lacks the sealed runtime cache policy")
    layout = receipt.get("layout_identity")
    if not isinstance(layout, dict) or not isinstance(layout.get("sources"), dict):
        raise RuntimeError("bootstrap receipt lacks source identity")
    trusted = layout.get("host", {}).get("trusted_executables")
    if not isinstance(trusted, dict):
        raise RuntimeError("bootstrap receipt lacks trusted system executables")
    for name, identity in trusted.items():
        if (
            not isinstance(identity, dict)
            or _preimport_executable_identity(Path(identity.get("requested_path", "")))
            != identity
        ):
            raise RuntimeError(f"trusted system executable drifted: {name}")
    sources = spec.get("sources", {})
    source_identity = layout["sources"]
    binding_identity = layout.get("bindings")
    if (
        not isinstance(binding_identity, dict)
        or not isinstance(binding_identity.get("runtime_cache"), dict)
        or binding_identity["runtime_cache"].get("policy") != expected_cache_policy
    ):
        raise RuntimeError("bootstrap layout lacks the runtime cache binding")
    for key, commit_key in (
        ("forge_repo", "forge_commit"),
        ("ai_toolkit_repo", "ai_toolkit_commit"),
    ):
        observed = _preimport_git_identity(
            Path(sources[key]), spec["source_identities"][commit_key]
        )
        if observed != source_identity[key]:
            raise RuntimeError(f"pre-import {key} identity drifted")
    if _preimport_tree_identity(Path(sources["venv"])) != source_identity["venv_tree"]:
        raise RuntimeError("pre-import staged venv identity drifted")
    if marker:
        cache_capture_id = (
            os.environ.get("FORGE_KREA_TIMING_CAPTURE_ID")
            if args.timing_probe_plan
            else None
        )
        expected_cache_environment = _runtime_cache_environment(
            plan_file_sha256,
            timing_capture_id=cache_capture_id,
            create=False,
        )
        if any(
            os.environ.get(name) != expected
            for name, expected in expected_cache_environment.items()
        ):
            raise RuntimeError("runtime cache environment differs from sealed policy")
        executable = Path(sys.executable).resolve(strict=True)
        if (
            marker != receipt["receipt_sha256"]
            or not executable.is_relative_to(_STAGED_VENV_ROOT.resolve(strict=True))
            or _sha256_file(executable)
            != source_identity["venv_python"]["resolved_sha256"]
        ):
            raise RuntimeError("trusted Stage-1 re-exec marker is invalid")
        return
    if not os.path.samefile(sys.executable, "/usr/bin/python3"):
        raise RuntimeError(
            "Stage-1 must start with /usr/bin/python3 before the staged venv"
        )
    bootstrap_code = (
        "import sys;sys.path.insert(0,'/app/forge');"
        "from ops.calibration import krea_host_bootstrap as m;"
        "raise SystemExit(m.main(['verify-layout','--receipt',sys.argv[1]]))"
    )
    subprocess.run(
        ["/usr/bin/python3", "-I", "-c", bootstrap_code, str(receipt_path)],
        check=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        timeout=180,
    )
    venv_python = _STAGED_VENV_ROOT / "bin/python"
    resolved = venv_python.resolve(strict=True)
    if _sha256_file(resolved) != source_identity["venv_python"]["resolved_sha256"]:
        raise RuntimeError("staged venv Python drifted after receipt recapture")
    cache_capture_id = (
        os.environ.get("FORGE_KREA_TIMING_CAPTURE_ID")
        if args.timing_probe_plan
        else None
    )
    if args.timing_probe_plan and not cache_capture_id:
        raise RuntimeError("timing capture lacks its cache namespace identity")
    if args.execution_plan and os.environ.get("FORGE_KREA_TIMING_CAPTURE_ID"):
        raise RuntimeError("execution plan inherited a timing capture identity")
    cache_environment = _runtime_cache_environment(
        plan_file_sha256,
        timing_capture_id=cache_capture_id,
        create=True,
    )
    timing_environment = None
    if args.timing_probe_plan:
        timing_environment = {name: os.environ.get(name, "") for name in _TIMING_ENV}
    environment = _minimal_runtime_environment(
        timing=timing_environment,
        cache=cache_environment,
        trusted_marker=receipt["receipt_sha256"],
    )
    os.execve(
        str(venv_python),
        [
            str(venv_python),
            "-I",
            "-c",
            _ISOLATED_MODULE_BOOTSTRAP,
            *sys.argv[1:],
        ],
        environment,
    )


def _validate_args(args: argparse.Namespace) -> None:
    for label, value in (
        ("task id", args.task_id),
        ("expected repo name", args.expected_repo_name),
    ):
        if not _SAFE_COMPONENT.fullmatch(value) or value in (".", ".."):
            raise ValueError(f"{label} must be one conservative path component")
    if not math.isfinite(args.lr) or not 0.0 < args.lr <= 1.0:
        raise ValueError("lr must be finite and in (0, 1]")
    if not isinstance(args.steps, int) or not 2 <= args.steps <= 10_000_000:
        raise ValueError("steps must be an integer in [2, 10000000]")
    if not math.isfinite(args.hours) or not 0.0 < args.hours <= 168.0:
        raise ValueError("hours must be finite and in (0, 168]")
    # ai-toolkit's run.py forwards SEED to numpy.random.seed, whose supported
    # range is narrower than torch.manual_seed's.
    if not isinstance(args.seed, int) or not 0 <= args.seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")


def _effective_recipe(
    normalized_recipe: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    fields = normalized_recipe["fields"]
    effective = {name: row["effective_value"] for name, row in fields.items()}
    if any(
        value is None
        for name, value in effective.items()
        if name not in {"submitted_step", "selector"}
    ):
        raise ValueError("local reproduction recipe has unresolved effective fields")
    planned = effective["planned_steps"]
    if (
        isinstance(planned, bool)
        or not isinstance(planned, int)
        or planned != args.steps
    ):
        raise ValueError("recipe planned steps contradict the approved run")
    lr = effective["learning_rate"]
    if (
        isinstance(lr, bool)
        or not isinstance(lr, (int, float))
        or not math.isfinite(float(lr))
        or float(lr) != args.lr
    ):
        raise ValueError("recipe learning rate contradicts the requested run")
    for key in (
        "rank",
        "alpha",
        "gradient_accumulation",
        "effective_batch",
        "save_cadence",
    ):
        value = effective[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"recipe {key} must be a positive integer")
    for key in ("optimizer", "loss", "scheduler"):
        if not isinstance(effective[key], str) or not effective[key].strip():
            raise ValueError(f"recipe {key} must be a non-empty string")
    optimizer_parameters = effective["optimizer_parameters"]
    if not isinstance(optimizer_parameters, dict) or not optimizer_parameters:
        raise ValueError("recipe optimizer_parameters must be a non-empty object")
    dropout = effective["dropout"]
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not math.isfinite(float(dropout))
        or not 0.0 <= float(dropout) <= 1.0
    ):
        raise ValueError("recipe dropout is invalid")
    guidance = effective["guidance"]
    if (
        not isinstance(guidance, dict)
        or set(guidance) != {"enabled", "scale"}
        or not isinstance(guidance["enabled"], bool)
        or (guidance["enabled"] and not isinstance(guidance["scale"], (int, float)))
        or (not guidance["enabled"] and guidance["scale"] is not None)
    ):
        raise ValueError("recipe guidance is invalid")
    if (args.guidance == "on") != guidance["enabled"]:
        raise ValueError("recipe guidance contradicts the requested run")
    ema = effective["ema"]
    if (
        not isinstance(ema, dict)
        or set(ema) != {"enabled", "decay"}
        or not isinstance(ema["enabled"], bool)
        or not isinstance(ema["decay"], (int, float))
    ):
        raise ValueError("recipe EMA is invalid")
    return effective


def _ensure_seeded_process(seed: int) -> None:
    """Restart before Forge imports so PYTHONHASHSEED covers this process too."""
    expected = str(seed)
    if os.environ.get("PYTHONHASHSEED") == expected:
        os.environ["SEED"] = expected
        os.environ[_REEXEC_MARKER] = expected
        return
    if os.environ.get(_REEXEC_MARKER):
        raise RuntimeError("seeded re-exec did not install the requested hash seed")
    timing_environment = None
    if os.environ.get("FORGE_KREA_TIMING_CAPTURE_ID"):
        timing_environment = {name: os.environ.get(name, "") for name in _TIMING_ENV}
    cache_environment = {
        name: os.environ.get(name, "")
        for name in (*_RUNTIME_CACHE_ENV, _RUNTIME_CACHE_NAMESPACE_ENV)
    }
    env = _minimal_runtime_environment(
        timing=timing_environment,
        cache=cache_environment,
        trusted_marker=os.environ.get(_TRUSTED_LAUNCH_MARKER),
        jit_enabled=os.environ.get("FORGE_CALIBRATION_JIT_ENABLED"),
        seed=expected,
    )
    os.execve(
        sys.executable,
        [
            sys.executable,
            "-I",
            "-c",
            _ISOLATED_MODULE_BOOTSTRAP,
            *sys.argv[1:],
        ],
        env,
    )


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _require_clean_paths(paths: list[Path]) -> None:
    leftovers = sorted(str(path) for path in paths if _lexists(path))
    if leftovers:
        raise FileExistsError(
            "calibration refuses preexisting mutable paths: " + ", ".join(leftovers)
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _fingerprint_path(path: Path) -> dict[str, Any]:
    """Hash a file/tree by logical names and bytes, following symlinks safely."""
    if not _lexists(path):
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    totals = {"files": 0, "bytes": 0, "symlinks": 0}

    def visit(actual: Path, logical: str, stack: frozenset[tuple[int, int]]) -> None:
        info = actual.lstat()
        if stat.S_ISLNK(info.st_mode):
            target = os.readlink(actual)
            digest.update(f"L\0{logical}\0{target}\0".encode("utf-8"))
            totals["symlinks"] += 1
            resolved = Path(os.path.realpath(actual))
            if not resolved.exists():
                raise RuntimeError(f"broken symlink in evidence input: {actual}")
            visit_followed(resolved, logical, stack)
            return
        visit_stat(actual, logical, info, stack)

    def visit_followed(
        actual: Path, logical: str, stack: frozenset[tuple[int, int]]
    ) -> None:
        visit_stat(actual, logical, actual.stat(), stack)

    def visit_stat(
        actual: Path,
        logical: str,
        info: os.stat_result,
        stack: frozenset[tuple[int, int]],
    ) -> None:
        if stat.S_ISREG(info.st_mode):
            digest.update(f"F\0{logical}\0{info.st_size}\0".encode("utf-8"))
            with actual.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            totals["files"] += 1
            totals["bytes"] += int(info.st_size)
            return
        if stat.S_ISDIR(info.st_mode):
            inode = (int(info.st_dev), int(info.st_ino))
            if inode in stack:
                raise RuntimeError(f"directory symlink cycle in {actual}")
            digest.update(f"D\0{logical}\0".encode("utf-8"))
            next_stack = stack | {inode}
            with os.scandir(actual) as entries:
                children = sorted(entries, key=lambda entry: entry.name)
            for child in children:
                child_logical = f"{logical}/{child.name}" if logical else child.name
                visit(Path(child.path), child_logical, next_stack)
            return
        raise RuntimeError(f"special filesystem entry is not hashable: {actual}")

    visit(path, "", frozenset())
    kind = "directory" if path.is_dir() else "file"
    return {"kind": kind, "sha256": digest.hexdigest(), **totals}


def _run_text(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout.strip()


def _code_fingerprint(root: Path) -> dict[str, Any]:
    root = Path(_run_text(["git", "-C", str(root), "rev-parse", "--show-toplevel"]))
    raw = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    ).stdout
    allowed_suffixes = {".py", ".yaml", ".yml", ".json", ".toml", ".sh", ".md", ".txt"}
    names = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        name = item.decode("utf-8", errors="strict")
        path = root / name
        if path.is_file() and path.suffix.lower() in allowed_suffixes:
            names.append(name)
    rows = []
    for name in sorted(set(names)):
        path = root / name
        rows.append((name, path.stat().st_size, _sha256_file(path)))
    return {
        "git_head": _run_text(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "git_head_tree": _run_text(
            ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"]
        ),
        "source_files": len(rows),
        "source_bytes": sum(row[1] for row in rows),
        "source_manifest_sha256": _canonical_hash(rows),
    }


def _runtime_fingerprint() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[re.sub(r"[-_.]+", "-", name).lower()] = distribution.version
    try:
        gpu = _run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
    except Exception as exc:
        gpu = [f"unavailable:{type(exc).__name__}"]
    executable = Path(sys.executable).resolve(strict=True)
    # Training seeds are deliberately *not* part of the reusable compute
    # runtime identity.  Seed A and seed B must be able to share a measured
    # throughput profile when every compute axis is identical.  They remain
    # explicit, separately hashed stochastic controls in the run/campaign
    # evidence below.
    compute_payload = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable_sha256": _sha256_file(executable),
        "packages_sha256": _canonical_hash(sorted(packages.items())),
        "package_count": len(packages),
        "gpu": gpu,
    }
    stochastic_controls = {
        "seed_env": os.environ.get("SEED"),
        "pythonhashseed_env": os.environ.get("PYTHONHASHSEED"),
    }
    runtime_cache = {
        "namespace": os.environ.get(_RUNTIME_CACHE_NAMESPACE_ENV),
        "cross_capture_or_plan_reuse": False,
        "within_process_reuse": True,
        "environment": {
            name: os.environ.get(name) for name in sorted(_RUNTIME_CACHE_ENV)
        },
    }
    return {
        **compute_payload,
        "runtime_cache": runtime_cache,
        "stochastic_controls": stochastic_controls,
        "stochastic_controls_sha256": _canonical_hash(stochastic_controls),
        "sha256": _canonical_hash(compute_payload),
    }


def _training_seed_support(ai_toolkit_dir: Path) -> dict[str, Any]:
    source = ai_toolkit_dir / "jobs/process/BaseTrainProcess.py"
    if not source.is_file():
        return {"supported": False, "source": None, "source_sha256": None}
    text = source.read_text(encoding="utf-8", errors="strict")
    supported = bool(re.search(r"get_conf\(\s*['\"]training_seed['\"]", text))
    return {
        "supported": supported,
        "source": "jobs/process/BaseTrainProcess.py",
        "source_sha256": _sha256_file(source),
    }


def _validate_measured_execution_envelope(
    *,
    profile: Any,
    process: dict[str, Any],
    num_images: int,
    fixture: dict[str, Any],
    base_model: dict[str, Any],
    pre: dict[str, Any],
    host_execution_identity_sha256: str,
    venv_tree_manifest_sha256: str,
) -> None:
    """Recompute every locally observable timing-equivalence field."""

    envelope = profile.execution_envelope
    train = process["train"]
    network = process["network"]
    dataset = process["datasets"][0]
    model = process["model"]
    jit_raw = os.environ.get("FORGE_CALIBRATION_JIT_ENABLED")
    if jit_raw not in {"0", "1"}:
        raise RuntimeError(
            "FORGE_CALIBRATION_JIT_ENABLED must explicitly bind the measured JIT path"
        )
    actual = {
        "network_rank": network["linear"],
        "network_alpha": network["linear_alpha"],
        "optimizer": train["optimizer"],
        "optimizer_config_sha256": _canonical_hash(train["optimizer_params"]),
        "loss": train["loss_type"],
        "differential_guidance_enabled": bool(
            train.get("do_differential_guidance", False)
        ),
        "guidance_scale": (
            train.get("differential_guidance_scale")
            if train.get("do_differential_guidance", False)
            else None
        ),
        "training_pair_count": num_images,
        "training_dataset_shape_sha256": fixture["training_dataset_shape_sha256"],
        "micro_batch_size": train["batch_size"],
        "gradient_accumulation_steps": train["gradient_accumulation"],
        "data_parallel_replicas": 1,
        "resolution_policy_sha256": _canonical_hash(dataset["resolution"]),
        "precision_policy_sha256": _canonical_hash(
            {
                "train_dtype": train.get("dtype"),
                "save_dtype": process["save"].get("dtype"),
            }
        ),
        "cache_latents_to_disk": bool(dataset.get("cache_latents_to_disk", False)),
        "cache_text_embeddings": bool(train.get("cache_text_embeddings", False)),
        "compile_enabled": bool(model.get("compile", False)),
        "jit_enabled": jit_raw == "1",
        "dataloader_workers": int(dataset.get("num_workers", 0)),
        "base_model_identity_sha256": pre["training_identity_sha256"],
        "runtime_identity_sha256": pre["runtime"]["sha256"],
        "host_execution_identity_sha256": host_execution_identity_sha256,
        "execution_surface": "staged_host_venv",
        "execution_scope": "discovery_only",
        "venv_tree_manifest_sha256": venv_tree_manifest_sha256,
        "gpu_identity_sha256": _canonical_hash(pre["runtime"]["gpu"]),
        "trainer_identity_sha256": _canonical_hash(
            {
                "forge": pre["code"]["forge"]["source_manifest_sha256"],
                "ai_toolkit": pre["code"]["ai_toolkit"]["source_manifest_sha256"],
            }
        ),
        # Timing is produced by the receipt-clock probe, not this consumer.
        # Recompute the actual producer bytes so a runner cannot masquerade as
        # the measurement tool named by a profile.
        "measurement_tool_sha256": _sha256_file(
            Path(__file__).with_name("krea_timing_probe.py").resolve(strict=True)
        ),
    }
    mismatches = {
        key: {"expected": getattr(envelope, key), "actual": value}
        for key, value in actual.items()
        if getattr(envelope, key) != value
    }
    if mismatches:
        raise RuntimeError(f"run escaped measured throughput envelope: {mismatches}")


def _diff_paths(left: Any, right: Any, pointer: str = "") -> set[str]:
    if type(left) is not type(right):
        return {pointer or "/"}
    if isinstance(left, dict):
        out: set[str] = set()
        for key in sorted(set(left) | set(right), key=str):
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            child = f"{pointer}/{escaped}"
            if key not in left or key not in right:
                out.add(child)
            else:
                out.update(_diff_paths(left[key], right[key], child))
        return out
    if isinstance(left, list):
        if len(left) != len(right):
            return {pointer or "/"}
        out: set[str] = set()
        for index, (old, new) in enumerate(zip(left, right)):
            out.update(_diff_paths(old, new, f"{pointer}/{index}"))
        return out
    return set() if left == right else {pointer or "/"}


def _normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    process = normalized["config"]["process"][0]
    normalized["config"]["name"] = "<run-identity>"
    process["training_folder"] = "<run-isolation-path>"
    train = process["train"]
    network = process["network"]
    dataset = process["datasets"][0]
    train["lr"] = "<axis:lr>"
    train["steps"] = "<axis:depth-steps>"
    train.pop("do_differential_guidance", None)
    train.pop("differential_guidance_scale", None)
    train["__forge_guidance_axis__"] = "<axis:guidance>"
    network["linear"] = "<axis:rank>"
    network["linear_alpha"] = "<axis:alpha>"
    train["optimizer"] = "<axis:optimizer>"
    train["optimizer_params"] = "<axis:optimizer-parameters>"
    train["loss_type"] = "<axis:loss>"
    train["noise_scheduler"] = "<axis:scheduler>"
    dataset["caption_dropout_rate"] = "<axis:dropout>"
    train["batch_size"] = "<axis:micro-batch>"
    train["gradient_accumulation"] = "<axis:gradient-accumulation>"
    train["ema_config"] = "<axis:ema>"
    process["save"]["save_every"] = "<derived:kill-safe-save-cadence>"
    return normalized


def _atomic_json(path: Path, value: Any) -> None:
    """Publish canonical JSON atomically without ever replacing a path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical_bytes(value))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        # link(2) is an atomic no-replace publication on the same filesystem.
        # os.replace would overwrite a concurrent creator after our lexists
        # precheck and could silently splice two attempts together.
        os.link(temp_name, path)
        os.unlink(temp_name)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"evidence path is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"evidence JSON is not an object: {path}")
    return value


def _read_canonical_control(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = path.expanduser().resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is not a regular file: {path}")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise RuntimeError(f"{label} must be canonical JSON plus one newline")
    return value, hashlib.sha256(raw).hexdigest()


def _bootstrap_execution_shape(
    probe: dict[str, Any], *, capture_id: str
) -> dict[str, Any]:
    """Adapt an approved pre-profile probe to the existing one-run machinery."""

    normalized_capture = re.sub(r"[^A-Za-z0-9_.-]", "-", capture_id)
    if not _SAFE_COMPONENT.fullmatch(normalized_capture):
        raise ValueError("timing capture id is not a safe run suffix")
    suffix = hashlib.sha256(normalized_capture.encode("utf-8")).hexdigest()[:12]
    task_prefix = probe["task_id"][:115].rstrip(".-_")
    repo_prefix = probe["expected_repo_name"][:115].rstrip(".-_")
    planned = probe["probe_schedule"]["planned_steps"]
    cadence = probe["probe_schedule"]["save_every"]
    candidate_steps = list(range(cadence, planned, cadence)) + [planned]
    return {
        **probe,
        "task_id": f"{task_prefix}-{suffix}",
        "expected_repo_name": f"{repo_prefix}-{suffix}",
        "schedule": {
            "mode": "timing_probe_fixed_depth",
            "planned_steps": planned,
            "save_every": cadence,
            "candidate_steps": candidate_steps,
            "required_landmarks": [],
            "landmark_policy": "none",
        },
        "budget_plan": {"hard_budget_s": str(probe["probe_schedule"]["hard_budget_s"])},
        "plan_sha256": probe["probe_contract_sha256"],
    }


def _bootstrap_profile(probe: dict[str, Any], krea_budget_module: Any) -> Any:
    envelope = krea_budget_module.load_execution_envelope(probe["execution_envelope"])
    record = {
        "schema": 1,
        "kind": "forge-krea-bootstrap-profile-placeholder",
        "probe_contract_sha256": probe["probe_contract_sha256"],
        "execution_envelope": envelope.to_record(),
        "not_a_throughput_profile": True,
    }
    return SimpleNamespace(
        execution_envelope=envelope,
        runtime_identity_sha256=envelope.runtime_identity_sha256,
        micro_batch_size=envelope.micro_batch_size,
        gradient_accumulation_steps=envelope.gradient_accumulation_steps,
        data_parallel_replicas=envelope.data_parallel_replicas,
        resolution_policy_sha256=envelope.resolution_policy_sha256,
        precision_policy_sha256=envelope.precision_policy_sha256,
        profile_sha256=None,
        to_record=lambda: record,
    )


def _bootstrap_budget(probe: dict[str, Any]) -> Any:
    record = {
        "schema": 1,
        "kind": "forge-krea-bootstrap-fixed-budget",
        "probe_contract_sha256": probe["probe_contract_sha256"],
        "hard_budget_s": str(probe["probe_schedule"]["hard_budget_s"]),
        "planned_steps": probe["probe_schedule"]["planned_steps"],
        "save_every": probe["probe_schedule"]["save_every"],
    }
    return SimpleNamespace(
        max_affordable_steps=probe["probe_schedule"]["planned_steps"],
        to_record=lambda: record,
    )


class _LinuxCheckpointWriteObserver:
    """Observe real safetensors CREATE->CLOSE_WRITE spans via inotify.

    Atomic-rename-only saves are deliberately not guessed: if the backend does
    not expose eight complete write spans, raw-profile assembly fails closed and
    an explicit backend instrumentation hook is required.
    """

    _EVENT = struct.Struct("iIII")
    _IN_CLOSE_WRITE = 0x00000008
    _IN_CREATE = 0x00000100

    def __init__(self, directory: Path, *, capture_id: str, emitter: Any):
        if sys.platform != "linux":
            raise RuntimeError("bootstrap timing requires Linux inotify")
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.capture_id = capture_id
        self.emitter = emitter
        self.stop_event = threading.Event()
        self.errors: list[BaseException] = []
        self.open_writes: dict[str, str] = {}
        self.sequence = 0
        libc = ctypes.CDLL(None, use_errno=True)
        init = libc.inotify_init1
        init.argtypes = [ctypes.c_int]
        init.restype = ctypes.c_int
        self.fd = int(init(os.O_NONBLOCK | os.O_CLOEXEC))
        if self.fd < 0:
            errno_value = ctypes.get_errno()
            raise OSError(errno_value, os.strerror(errno_value))
        add = libc.inotify_add_watch
        add.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add.restype = ctypes.c_int
        watch = int(
            add(
                self.fd,
                os.fsencode(directory),
                self._IN_CREATE | self._IN_CLOSE_WRITE,
            )
        )
        if watch < 0:
            errno_value = ctypes.get_errno()
            os.close(self.fd)
            raise OSError(errno_value, os.strerror(errno_value))
        self.thread = threading.Thread(
            target=self._loop, name="krea-checkpoint-write-observer"
        )

    def start(self) -> None:
        self.thread.start()

    def _loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                readable, _, _ = select.select([self.fd], [], [], 0.1)
                if not readable:
                    continue
                data = os.read(self.fd, 64 * 1024)
                offset = 0
                while offset < len(data):
                    _wd, mask, _cookie, length = self._EVENT.unpack_from(data, offset)
                    offset += self._EVENT.size
                    raw_name = data[offset : offset + length]
                    offset += length
                    name = raw_name.split(b"\0", 1)[0].decode("utf-8", "strict")
                    if not name.endswith(".safetensors") or name == "last.safetensors":
                        continue
                    if mask & self._IN_CREATE:
                        if name in self.open_writes:
                            raise RuntimeError(
                                f"checkpoint write reopened before close: {name}"
                            )
                        self.sequence += 1
                        observation_id = (
                            f"save-{self.sequence:03d}-"
                            f"{hashlib.sha256(name.encode()).hexdigest()[:12]}"
                        )
                        self.open_writes[name] = observation_id
                        self.emitter(
                            observation_id=observation_id,
                            metric="checkpoint_save",
                            state="begin",
                            units=1,
                        )
                    if mask & self._IN_CLOSE_WRITE:
                        observation_id = self.open_writes.pop(name, None)
                        if observation_id is not None:
                            self.emitter(
                                observation_id=observation_id,
                                metric="checkpoint_save",
                                state="end",
                                units=1,
                            )
        except BaseException as exc:
            self.errors.append(exc)

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3)
        os.close(self.fd)
        if self.thread.is_alive():
            raise RuntimeError("checkpoint write observer did not stop")
        if self.errors:
            raise RuntimeError("checkpoint write observer failed") from self.errors[0]
        if self.open_writes:
            raise RuntimeError(
                "checkpoint writes did not close: "
                + ", ".join(sorted(self.open_writes))
            )


def _campaign_baseline(campaign_dir: Path, envelope: dict[str, Any]) -> dict[str, Any]:
    """Atomically establish once, then require byte-semantic envelope equality."""
    baseline_path = campaign_dir / _BASELINE_NAME
    lock_path = campaign_dir / f".{_BASELINE_NAME}.lock"
    campaign_dir.mkdir(parents=True, exist_ok=True)
    if _lexists(lock_path):
        raise RuntimeError(
            f"campaign baseline lock exists; audit before retrying: {lock_path}"
        )
    envelope_sha = _canonical_hash(envelope)
    if _lexists(baseline_path):
        baseline = _read_json(baseline_path)
    else:
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
            os.fsync(lock_fd)
        finally:
            os.close(lock_fd)
        try:
            if _lexists(baseline_path):
                raise RuntimeError("campaign baseline appeared during lock acquisition")
            baseline = {
                "schema": 1,
                "kind": "forge-krea2-calibration-campaign",
                "envelope": envelope,
                "envelope_sha256": envelope_sha,
                "created_unix": int(time.time()),
            }
            _atomic_json(baseline_path, baseline)
        finally:
            os.unlink(lock_path)
            directory_fd = os.open(
                campaign_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    if (
        baseline.get("schema") != 1
        or baseline.get("kind") != "forge-krea2-calibration-campaign"
        or baseline.get("envelope_sha256") != _canonical_hash(baseline.get("envelope"))
        or baseline.get("envelope_sha256") != envelope_sha
        or baseline.get("envelope") != envelope
    ):
        raise RuntimeError("condition does not match the immutable campaign baseline")
    return {
        "path": str(baseline_path.resolve()),
        "file_sha256": _sha256_file(baseline_path),
        "envelope_sha256": envelope_sha,
    }


def _validate_existing_campaign_prefix(
    campaign_dir: Path, expected: dict[str, Any]
) -> None:
    """Reject a known-incompatible campaign before spending the GPU run."""
    baseline_path = campaign_dir / _BASELINE_NAME
    lock_path = campaign_dir / f".{_BASELINE_NAME}.lock"
    if _lexists(lock_path):
        raise RuntimeError(
            f"campaign baseline lock exists; audit before retrying: {lock_path}"
        )
    if not _lexists(baseline_path):
        return
    baseline = _read_json(baseline_path)
    envelope = baseline.get("envelope")
    if (
        baseline.get("schema") != 1
        or baseline.get("kind") != "forge-krea2-calibration-campaign"
        or not isinstance(envelope, dict)
        or baseline.get("envelope_sha256") != _canonical_hash(envelope)
    ):
        raise RuntimeError("existing campaign baseline is malformed")
    mismatches = sorted(
        key for key, value in expected.items() if envelope.get(key) != value
    )
    if mismatches:
        raise RuntimeError(
            "condition differs from the campaign's fixed envelope fields: "
            + ", ".join(mismatches)
        )


def _content_identity(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in ("kind", "sha256", "files", "bytes", "symlinks")}


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    args: argparse.Namespace,
    scope: dict[str, Any],
    candidates: list[Path],
) -> dict[str, str]:
    expected_header = {
        "schema": 1,
        "source": "heldout",
        "complete": True,
        "task_id": args.task_id,
        "expected_repo_name": args.expected_repo_name,
        "attempt_nonce": scope["attempt_nonce"],
        "scope_started_unix": scope["started_unix"],
        "direction": "min",
        "metric": "heldout_diffusion_loss_proxy_v2",
        "proxy_not_validator_metric": True,
        "model_type": "krea2",
        "strata_scored_separately": True,
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"manifest {key!r} is not bound to this attempt")
    holdout_pairs = manifest.get("holdout_pairs")
    epochs = manifest.get("probe_epochs")
    seed = manifest.get("seed")
    if not isinstance(holdout_pairs, int) or holdout_pairs <= 0:
        raise RuntimeError("manifest holdout_pairs must be a positive integer")
    if epochs != _PROBE_EPOCHS:
        raise RuntimeError("manifest probe_epochs differs from the pinned producer")
    if seed != _PROBE_SEED:
        raise RuntimeError("manifest probe seed differs from the pinned producer")
    elapsed = float(manifest.get("elapsed_s"))
    created = manifest.get("created_unix")
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError("manifest elapsed_s is not finite and positive")
    if not isinstance(created, int) or created < int(float(scope["started_unix"])):
        raise RuntimeError("manifest timestamp predates the active scope")
    caption_weight = float(manifest.get("captioned_weight"))
    blank_weight = float(manifest.get("blank_caption_weight"))
    if (
        not math.isfinite(caption_weight)
        or not math.isfinite(blank_weight)
        or not math.isclose(caption_weight, 0.25, rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(blank_weight, 0.75, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise RuntimeError("manifest stratum weights do not match the evaluator")
    rows = manifest.get("scores")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("manifest has no candidate rows")
    candidate_by_name = {path.name: path for path in candidates}
    seen: dict[str, str] = {}
    expected_stratum_points = holdout_pairs * epochs
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("manifest candidate row is not an object")
        name = row.get("checkpoint")
        declared = row.get("sha256")
        if not isinstance(name, str) or Path(name).name != name or name in seen:
            raise RuntimeError("manifest checkpoint names must be unique basenames")
        if name not in candidate_by_name or not isinstance(declared, str):
            raise RuntimeError(f"manifest candidate is not current: {name!r}")
        if name == f"{args.expected_repo_name}.safetensors":
            expected_step = args.steps
        else:
            step_match = re.fullmatch(
                rf"{re.escape(args.expected_repo_name)}_(\d+)\.safetensors",
                name,
            )
            if step_match is None:
                raise RuntimeError(f"manifest candidate name is malformed: {name!r}")
            expected_step = int(step_match.group(1))
        if row.get("step") != expected_step or not 0 < expected_step <= args.steps:
            raise RuntimeError(f"manifest step does not match {name!r}")
        actual = _sha256_file(candidate_by_name[name])
        if not _SHA256.fullmatch(declared) or actual != declared:
            raise RuntimeError(f"manifest candidate hash mismatch: {name!r}")
        finite_fields = (
            "score",
            "captioned_score",
            "blank_caption_score",
            "captioned_stddev",
            "blank_caption_stddev",
        )
        values = {}
        for field in finite_fields:
            values[field] = float(row.get(field))
            if not math.isfinite(values[field]) or values[field] < 0.0:
                raise RuntimeError(f"manifest {field} is invalid for {name!r}")
        if int(row.get("captioned_points")) != expected_stratum_points:
            raise RuntimeError(f"captioned point count is incomplete for {name!r}")
        if int(row.get("blank_caption_points")) != expected_stratum_points:
            raise RuntimeError(f"blank point count is incomplete for {name!r}")
        if int(row.get("points")) != expected_stratum_points * 2:
            raise RuntimeError(f"combined point count is incomplete for {name!r}")
        recomputed = (
            caption_weight * values["captioned_score"]
            + blank_weight * values["blank_caption_score"]
        )
        if not math.isclose(values["score"], recomputed, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"manifest aggregate does not recompute for {name!r}")
        seen[name] = actual
    if set(seen) != set(candidate_by_name):
        raise RuntimeError("manifest does not cover every current-run candidate")
    return seen


def _validate_telemetry(telemetry: dict[str, Any], *, args: argparse.Namespace) -> None:
    if telemetry.get("schema") != 1:
        raise RuntimeError("telemetry schema is not 1")
    meta = telemetry.get("meta")
    events = telemetry.get("events")
    if not isinstance(meta, dict) or not isinstance(events, list):
        raise RuntimeError("telemetry lacks meta/events")
    if meta.get("task_id") != args.task_id or meta.get("model_type") != "krea2":
        raise RuntimeError("telemetry is not bound to this task/model type")
    if meta.get("steps") != args.steps:
        raise RuntimeError("telemetry planned depth differs from the condition")
    names = [event.get("name") for event in events if isinstance(event, dict)]
    forbidden = [
        name
        for name in names
        if isinstance(name, str)
        and (
            name.endswith("_failed") or "fallback" in name or name.endswith("_skipped")
        )
    ]
    if forbidden:
        raise RuntimeError(
            f"failure/fallback telemetry present: {sorted(set(forbidden))}"
        )
    required = {
        "checkpoint_scope_started",
        "dataset_ready",
        "toolkit_start",
        "toolkit_end",
        "toolkit_metrics",
        "checkpoint_selected",
        "checkpoint_finalized",
        "run_complete",
    }
    missing = sorted(name for name in required if names.count(name) != 1)
    if missing:
        raise RuntimeError(
            f"required telemetry events are absent/non-unique: {missing}"
        )
    if any(isinstance(name, str) and name.startswith("holdout_") for name in names):
        raise RuntimeError("discovery run unexpectedly activated in-task proxy scoring")
    toolkit_end = next(event for event in events if event.get("name") == "toolkit_end")
    if (
        toolkit_end.get("returncode") != 0
        or toolkit_end.get("stopped_by_deadline") is not False
    ):
        raise RuntimeError("ai-toolkit did not finish cleanly before the deadline")
    metrics = next(event for event in events if event.get("name") == "toolkit_metrics")
    if metrics.get("last_step") != args.steps:
        raise RuntimeError("ai-toolkit did not reach the requested depth")
    finalized = next(
        event for event in events if event.get("name") == "checkpoint_finalized"
    )
    if finalized.get("status") != "selected_current_run":
        raise RuntimeError("telemetry does not show current-run finalization")


def main() -> int:
    args = _parse()
    _validate_execution_paths(args)
    _trusted_stage1_reexec(args)
    try:
        from . import krea_execution_surface_policy
        from . import krea_training_evidence
    except ImportError:  # pragma: no cover - direct script execution.
        import krea_execution_surface_policy  # type: ignore[no-redef]
        import krea_training_evidence  # type: ignore[no-redef]

    timing_bootstrap = args.timing_probe_plan is not None
    timing_capture_id = os.environ.get("FORGE_KREA_TIMING_CAPTURE_ID")
    if timing_bootstrap:
        if args.timing_probe_approval is None or args.execution_approval is not None:
            raise ValueError("timing bootstrap requires --timing-probe-approval only")
        if not timing_capture_id or not _SAFE_COMPONENT.fullmatch(timing_capture_id):
            raise ValueError(
                "timing bootstrap must run under krea_timing_probe.py capture"
            )
        try:
            from . import krea_execution_plan
            from . import krea_timing_probe
        except ImportError:  # pragma: no cover - direct script execution.
            import krea_execution_plan  # type: ignore[no-redef]
            import krea_timing_probe  # type: ignore[no-redef]
        probe, execution_plan_file_sha = _read_canonical_control(
            args.timing_probe_plan, "timing probe plan"
        )
        execution_approval, execution_approval_file_sha = _read_canonical_control(
            args.timing_probe_approval, "timing probe approval"
        )
        execution_controls = krea_execution_plan.validate_timing_probe_plan(probe)
        krea_execution_plan.validate_timing_probe_approval(
            execution_approval, plan=probe
        )
        execution_plan = _bootstrap_execution_shape(probe, capture_id=timing_capture_id)
        execution_plan_path = args.timing_probe_plan
        execution_approval_path = args.timing_probe_approval
    else:
        if args.execution_approval is None or args.timing_probe_approval is not None:
            raise ValueError("execution mode requires --execution-approval only")
        (
            execution_plan,
            execution_plan_file_sha,
            execution_approval,
            execution_approval_file_sha,
            execution_controls,
        ) = krea_training_evidence.load_execution_controls(
            args.execution_plan, args.execution_approval
        )
        execution_plan_path = args.execution_plan
        execution_approval_path = args.execution_approval
    schedule = execution_plan["schedule"]
    recipe_fields = execution_plan["execution_recipe"]["fields"]
    args.task_id = execution_plan["task_id"]
    args.expected_repo_name = execution_plan["expected_repo_name"]
    args.model = execution_plan["base_model"]["model_id"]
    args.seed = execution_plan["seed"]
    args.steps = schedule["planned_steps"]
    args.lr = recipe_fields["learning_rate"]["effective_value"]
    guidance_value = recipe_fields["guidance"]["effective_value"]
    args.guidance = "on" if guidance_value["enabled"] else "off"
    args.hours = float(execution_plan["budget_plan"]["hard_budget_s"]) / 3600.0
    args.throughput_profile = execution_controls.get("throughput_profile_path")
    sealed_runtime = execution_controls.get("bootstrap_runtime")
    if not isinstance(sealed_runtime, dict):
        raise ValueError("execution controls lack the sealed bootstrap runtime")
    sealed_surface = execution_controls.get("bootstrap_execution_surface")
    if (
        not isinstance(sealed_surface, dict)
        or sealed_runtime.get("execution_surface") != "staged_host_venv"
    ):
        raise ValueError("execution controls do not authorize the staged host venv")
    sealed_venv = sealed_surface.get("venv_python")
    if not isinstance(sealed_venv, dict):
        raise ValueError("execution controls lack the staged venv identity")
    executable = Path(sys.executable).resolve(strict=True)
    expected_executable = (
        Path("/app/venv") / sealed_venv["resolved_relative_path"]
    ).resolve(strict=True)
    if (
        executable != expected_executable
        or _sha256_file(executable) != sealed_venv["resolved_sha256"]
    ):
        raise ValueError("runner Python differs from the bootstrap-bound host venv")
    sealed_environment = {
        "FORGE_CALIBRATION_JIT_ENABLED": (
            "1" if sealed_runtime["jit_enabled"] else "0"
        ),
    }
    for name, expected in sealed_environment.items():
        supplied = os.environ.get(name)
        if supplied is not None and supplied != expected:
            raise ValueError(f"operator environment contradicts sealed {name}")
        os.environ[name] = expected
    _validate_args(args)
    _ensure_seeded_process(args.seed)

    # Start the receipt-clock startup span immediately after the seeded re-exec
    # and before the heavy Forge/project imports.  The end marker is emitted at
    # the first fully resolved training config, so import/configuration time is
    # part of the measured startup reserve rather than silently omitted.
    startup_observation_id = None
    if timing_bootstrap:
        timing_id = hashlib.sha256(timing_capture_id.encode("utf-8")).hexdigest()[:16]
        startup_observation_id = f"startup-{timing_id}"
        krea_timing_probe.emit_marker(
            observation_id=startup_observation_id,
            metric="startup",
            state="begin",
        )

    # Heavy/project imports intentionally occur only after seeded re-exec.
    import yaml

    from forge import telemetry as forge_telemetry
    from forge.cli import main as forge_main
    from forge.data.schema import ImageSpec
    from forge.tasks import aitoolkit, checkpoints, publication
    from forge.tasks.integrity import valid_safetensors

    try:
        from . import krea_budget
        from . import krea_host_identity
    except ImportError:  # pragma: no cover - direct script execution.
        import krea_budget  # type: ignore[no-redef]
        import krea_host_identity  # type: ignore[no-redef]

    campaign_dir = _authorized_child_path(
        args.campaign_dir, _CAMPAIGN_ROOT, "campaign-dir"
    )
    condition_path = campaign_dir / "conditions" / f"{args.task_id}.json"
    condition_claim_path = campaign_dir / "conditions" / f".{args.task_id}.claim"
    evidence_dir = campaign_dir / "evidence" / args.task_id
    spec = ImageSpec.build(
        task_id=args.task_id,
        model=args.model,
        model_type="krea2",
        expected_repo_name=args.expected_repo_name,
        trigger_word=None,
        dataset_zip=None,
    )
    save_root = Path(spec.save_root).resolve()
    checkpoint_root = _CHECKPOINT_ROOT.resolve(strict=True)
    if save_root == checkpoint_root or not save_root.is_relative_to(checkpoint_root):
        raise ValueError(
            "trainer checkpoint namespace escaped /app/checkpoints: " f"{save_root}"
        )
    mutable_paths = [
        Path(spec.config_path),
        Path(spec.training_folder),
        Path(spec.save_root),
        Path(spec.dataset_holdout_dir),
        Path(spec.dataset_images_dir),
        Path(spec.dataset_images_dir + "__extract"),
        Path(spec.dataset_images_dir + "__flat"),
        condition_path,
        condition_claim_path,
        evidence_dir,
    ]
    _require_clean_paths(mutable_paths)

    # Timing evidence is portable only inside the static host/storage/GPU
    # identity on which it was measured.  The live load/RAM/free-space policy
    # is checked before any Forge import can allocate the GPU or mutate its
    # checkpoint namespace.
    checkpoint_parent = Path(spec.save_root).parent.resolve(strict=True)
    host_preflight_before = krea_host_identity.verify_live(
        execution_controls["host_execution_manifest"],
        checkpoint_path=checkpoint_parent,
    )
    condition_claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_fd = os.open(
        condition_claim_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        claim = {
            "schema": 1,
            "kind": "forge-krea-condition-namespace-claim",
            "task_id": args.task_id,
            "execution_plan_sha256": execution_plan["plan_sha256"],
        }
        os.write(claim_fd, _canonical_bytes(claim) + b"\n")
        os.fsync(claim_fd)
    finally:
        os.close(claim_fd)

    # Source facts and internal-control bases were already normalized in the
    # independently approved execution plan.  Only its concrete local recipe
    # may configure this run.
    effective_recipe = _effective_recipe(execution_controls["execution_recipe"], args)

    dataset_zip = Path(spec.cached_zip_path)
    if dataset_zip.is_symlink() or not dataset_zip.is_file():
        raise FileNotFoundError(f"staged dataset must be a regular file: {dataset_zip}")
    if _sha256_file(dataset_zip) != execution_plan["training_archive"]["sha256"]:
        raise RuntimeError("staged training archive differs from the approved fixture")
    ai_toolkit_dir = Path(os.environ.get("AI_TOOLKIT_DIR", "/app/ai-toolkit")).resolve(
        strict=True
    )
    forge_root = Path(__file__).resolve().parents[2]
    for source_root in (forge_root, ai_toolkit_dir):
        if campaign_dir == source_root or campaign_dir.is_relative_to(source_root):
            raise ValueError(
                f"campaign-dir must be outside source trees: {campaign_dir}"
            )
    training_seed = _training_seed_support(ai_toolkit_dir)
    base_paths = {
        "base_model": Path(spec.cached_model_dir),
        "text_encoder": Path(_KREA_TEXT_ENCODER),
    }
    for label, path in base_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} is not staged: {path}")

    pre = {
        "dataset": _fingerprint_path(dataset_zip),
        "base_assets": {
            label: _fingerprint_path(path) for label, path in base_paths.items()
        },
        "code": {
            "forge": _code_fingerprint(forge_root),
            "ai_toolkit": _code_fingerprint(ai_toolkit_dir),
        },
        "runtime": _runtime_fingerprint(),
    }
    pre["training_identity_sha256"] = _canonical_hash(
        {
            label: _content_identity(value)
            for label, value in sorted(pre["base_assets"].items())
        }
    )
    if (
        pre["training_identity_sha256"]
        != execution_plan["base_model"]["training_identity_sha256"]
    ):
        raise RuntimeError("staged training base differs from the sealed base identity")
    if timing_bootstrap:
        profile_path = None
        profile = _bootstrap_profile(probe, krea_budget)
        budget_plan = _bootstrap_budget(probe)
    else:
        profile_path = args.throughput_profile.expanduser().resolve(strict=True)
        profile_document = _read_json(profile_path)
        profile = krea_budget.load_throughput_profile(profile_document)
        budget_plan = krea_budget.plan_budget(
            profile, hard_budget_s=args.hours * 3600.0
        )
        if budget_plan.to_record() != execution_plan["budget_plan"]:
            raise RuntimeError("live budget calculation differs from the sealed plan")
        if execution_plan["schedule"]["mode"] == "measured_budget_fill":
            if budget_plan.max_affordable_steps != args.steps:
                raise RuntimeError("budget-fill depth does not fill the measured plan")
        elif args.steps > budget_plan.max_affordable_steps:
            raise RuntimeError("release-control depth does not fit the measured plan")
    if profile.runtime_identity_sha256 != pre["runtime"]["sha256"]:
        raise RuntimeError("throughput profile runtime does not match this process")
    allowed_differences = {
        "scientific_axes": list(_SCIENTIFIC_AXIS_POINTERS),
        "derived_from_depth": list(_DERIVED_POINTERS),
        "run_isolation_only": list(_ISOLATION_POINTERS),
    }
    campaign_fixed_prefix = {
        "schema": 1,
        "model": args.model,
        "model_type": "krea2",
        "hours": args.hours,
        "training_seed": args.seed,
        "training_seed_support": training_seed,
        "dataset_content": _content_identity(pre["dataset"]),
        "base_assets": {
            label: _content_identity(value)
            for label, value in pre["base_assets"].items()
        },
        "code": pre["code"],
        "runtime_sha256": pre["runtime"]["sha256"],
        "host_execution_identity_sha256": execution_controls["host_execution_manifest"][
            "host_execution_identity_sha256"
        ],
        "execution_envelope": profile.execution_envelope.to_record(),
        "execution_envelope_sha256": profile.execution_envelope.execution_envelope_sha256,
        "throughput_equivalence_class": profile.execution_envelope.equivalence_class,
        "execution_surface_policy": krea_execution_surface_policy.POLICY,
        "throughput_profile_sha256": profile.profile_sha256,
        "budget_plan_sha256": _canonical_hash(budget_plan.to_record()),
        "timing_probe_contract_sha256": (
            probe["probe_contract_sha256"] if timing_bootstrap else None
        ),
        "allowed_condition_config_differences": allowed_differences,
        "in_task_proxy_selection": {"enabled": False, "reserve_s": 0},
    }
    _validate_existing_campaign_prefix(campaign_dir, campaign_fixed_prefix)

    original_build_config = aitoolkit.build_config
    captured: dict[str, Any] = {}
    startup_marker_closed = False

    def _calibration_config(spec_arg, num_images, hours_to_complete):
        nonlocal startup_marker_closed
        baseline = original_build_config(spec_arg, num_images, hours_to_complete)
        resolved = copy.deepcopy(baseline)
        process = resolved["config"]["process"][0]
        train = process["train"]
        network = process["network"]
        dataset_config = process["datasets"][0]
        train["steps"] = effective_recipe["planned_steps"]
        train["lr"] = effective_recipe["learning_rate"]
        network["linear"] = effective_recipe["rank"]
        network["linear_alpha"] = effective_recipe["alpha"]
        train["optimizer"] = effective_recipe["optimizer"]
        train["optimizer_params"] = copy.deepcopy(
            effective_recipe["optimizer_parameters"]
        )
        train["loss_type"] = effective_recipe["loss"]
        train["noise_scheduler"] = effective_recipe["scheduler"]
        dataset_config["caption_dropout_rate"] = effective_recipe["dropout"]
        train["batch_size"] = (
            effective_recipe["effective_batch"]
            // effective_recipe["gradient_accumulation"]
        )
        if (
            train["batch_size"] * effective_recipe["gradient_accumulation"]
            != effective_recipe["effective_batch"]
        ):
            raise RuntimeError("recipe effective batch is not integral")
        train["gradient_accumulation"] = effective_recipe["gradient_accumulation"]
        train["ema_config"] = {
            "use_ema": effective_recipe["ema"]["enabled"],
            "ema_decay": effective_recipe["ema"]["decay"],
        }
        guidance = effective_recipe["guidance"]
        if guidance["enabled"]:
            train["do_differential_guidance"] = True
            train["differential_guidance_scale"] = guidance["scale"]
        else:
            train.pop("do_differential_guidance", None)
            train.pop("differential_guidance_scale", None)
        approved_save_every = execution_plan["schedule"]["save_every"]
        process["save"]["save_every"] = approved_save_every
        if effective_recipe["save_cadence"] != approved_save_every:
            raise RuntimeError("recipe save cadence differs from the sealed schedule")
        if training_seed["supported"]:
            # BaseTrainProcess reads this at process scope, not train scope.
            process["training_seed"] = args.seed
        changed = _diff_paths(baseline, resolved)
        unexpected = changed - _ALLOWED_BUILDER_MUTATIONS
        if unexpected:
            raise RuntimeError(
                f"calibration mutated non-axis config fields: {sorted(unexpected)}"
            )
        if process["save"]["save_every"] != approved_save_every:
            raise RuntimeError("save cadence differs from the sealed schedule")
        resolution_policy_sha = _canonical_hash(dataset_config["resolution"])
        precision_policy_sha = _canonical_hash(
            {
                "train_dtype": train.get("dtype"),
                "save_dtype": process["save"].get("dtype"),
            }
        )
        if (
            train.get("batch_size") != profile.micro_batch_size
            or train.get("gradient_accumulation") != profile.gradient_accumulation_steps
            or profile.data_parallel_replicas != 1
            or effective_recipe["effective_batch"]
            != profile.micro_batch_size * profile.gradient_accumulation_steps
            or resolution_policy_sha != profile.resolution_policy_sha256
            or precision_policy_sha != profile.precision_policy_sha256
        ):
            raise RuntimeError(
                "resolved training geometry differs from the measured profile"
            )
        _validate_measured_execution_envelope(
            profile=profile,
            process=process,
            num_images=int(num_images),
            fixture=execution_controls["fixture"],
            base_model=execution_plan["base_model"],
            pre=pre,
            host_execution_identity_sha256=execution_controls[
                "host_execution_manifest"
            ]["host_execution_identity_sha256"],
            venv_tree_manifest_sha256=sealed_surface["venv_tree"]["manifest_sha256"],
        )
        captured.update(
            baseline=copy.deepcopy(baseline),
            resolved=copy.deepcopy(resolved),
            builder_mutations=sorted(changed),
            num_images=int(num_images),
            derived_hours=float(hours_to_complete),
        )
        if timing_bootstrap and not startup_marker_closed:
            krea_timing_probe.emit_marker(
                observation_id=startup_observation_id,
                metric="startup",
                state="end",
            )
            startup_marker_closed = True
        return resolved

    # Discovery exact scoring is an offline stage.  Enabling Forge's in-task
    # proxy would both alter the train split and reserve ~900 seconds that the
    # sealed budget intentionally assigns to optimizer work.
    os.environ["FORGE_HOLDOUT_SELECTION_TYPES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    aitoolkit.build_config = _calibration_config
    original_run_toolkit = aitoolkit._run_toolkit
    original_finalize = aitoolkit._finalize
    checkpoint_observer = None
    if timing_bootstrap:
        checkpoint_observer = _LinuxCheckpointWriteObserver(
            Path(spec.save_root),
            capture_id=timing_capture_id,
            emitter=krea_timing_probe.emit_marker,
        )
        checkpoint_observer.start()

        def _timed_run_toolkit(*toolkit_args, **toolkit_kwargs):
            observation_id = f"updates-{timing_id}"
            krea_timing_probe.emit_marker(
                observation_id=observation_id,
                metric="optimizer_update",
                state="begin",
                units=args.steps,
            )
            try:
                return original_run_toolkit(*toolkit_args, **toolkit_kwargs)
            finally:
                krea_timing_probe.emit_marker(
                    observation_id=observation_id,
                    metric="optimizer_update",
                    state="end",
                    units=args.steps,
                )

        def _timed_finalize(*finalize_args, **finalize_kwargs):
            observation_id = f"finalize-{timing_id}"
            krea_timing_probe.emit_marker(
                observation_id=observation_id,
                metric="finalization",
                state="begin",
            )
            try:
                return original_finalize(*finalize_args, **finalize_kwargs)
            finally:
                krea_timing_probe.emit_marker(
                    observation_id=observation_id,
                    metric="finalization",
                    state="end",
                )

        aitoolkit._run_toolkit = _timed_run_toolkit
        aitoolkit._finalize = _timed_finalize
    launch_not_before = time.time()
    try:
        return_code = forge_main(
            [
                "--task-id",
                args.task_id,
                "--model",
                args.model,
                "--model-type",
                "krea2",
                "--expected-repo-name",
                args.expected_repo_name,
                "--hours-to-complete",
                str(args.hours),
            ]
        )
    finally:
        aitoolkit.build_config = original_build_config
        aitoolkit._run_toolkit = original_run_toolkit
        aitoolkit._finalize = original_finalize
        if checkpoint_observer is not None:
            checkpoint_observer.close()
    if return_code != 0:
        raise RuntimeError(f"Forge returned nonzero status {return_code}")
    if not captured:
        raise RuntimeError("Forge never resolved the calibration config")
    if timing_bootstrap and not startup_marker_closed:
        raise RuntimeError("timing bootstrap never completed its startup observation")
    if timing_bootstrap and checkpoint_observer.sequence < 8:
        raise RuntimeError(
            "backend exposed fewer than eight real checkpoint write spans; "
            "timing profile remains unavailable"
        )

    upload_observation_id = None
    if timing_bootstrap:
        upload_observation_id = f"upload-ready-{timing_id}"
        krea_timing_probe.emit_marker(
            observation_id=upload_observation_id,
            metric="upload",
            state="begin",
        )

    # Production publication archives private sidecars outside the exact folder
    # G.O.D uploads.  Resolve the same-process ephemeral bundle explicitly; never
    # add a calibration-only switch that could disable the production scrub.
    private_dir = Path(forge_telemetry.private_bundle_dir(spec.save_root))
    config_path = Path(publication.private_artifact_path(spec.save_root, "config.yaml"))
    telemetry_path = Path(forge_telemetry.private_record_path(spec.save_root))
    public_telemetry_path = Path(spec.save_root) / "forge_run.json"
    scope_path = Path(
        publication.private_artifact_path(
            spec.save_root, ".forge_checkpoint_scope.json"
        )
    )
    last_path = Path(spec.save_root) / "last.safetensors"
    selected_paths = [
        config_path,
        telemetry_path,
        public_telemetry_path,
        scope_path,
        last_path,
    ]
    for path in selected_paths:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"required current-run output is absent/unsafe: {path}")
    if private_dir.is_symlink() or not private_dir.is_dir():
        raise RuntimeError(f"private evidence bundle is absent/unsafe: {private_dir}")

    loaded_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if loaded_config != captured["resolved"]:
        raise RuntimeError("persisted YAML differs from the resolved config")
    process = loaded_config["config"]["process"][0]
    if (
        process["train"].get("steps") != args.steps
        or process["train"].get("lr") != args.lr
    ):
        raise RuntimeError("persisted config does not contain the requested axes")
    if training_seed["supported"] and process.get("training_seed") != args.seed:
        raise RuntimeError("ai-toolkit training_seed was not pinned")

    active_scope = checkpoints.load_run(spec.save_root)
    disk_scope = _read_json(scope_path)
    if active_scope is None or active_scope != disk_scope:
        raise RuntimeError("checkpoint scope is not active in this process")
    scope = active_scope
    if (
        scope.get("schema") != 2
        or scope.get("repo") != args.expected_repo_name
        or scope.get("quarantine_complete") is not True
        or scope.get("planned_steps") != args.steps
        or scope.get("model_type") != "krea2"
        or scope.get("before") != {}
        or not isinstance(scope.get("attempt_nonce"), str)
        or not scope["attempt_nonce"]
        or not isinstance(scope.get("process_nonce"), str)
        or not scope["process_nonce"]
        or not isinstance(scope.get("started_unix"), (int, float))
        or scope["started_unix"] < launch_not_before - 1.0
    ):
        raise RuntimeError("checkpoint scope is stale, contaminated, or incomplete")

    candidates = [
        Path(path) for path in checkpoints.current_loras(spec.save_root, scope)
    ]
    if len(candidates) < 2 or not all(valid_safetensors(path) for path in candidates):
        raise RuntimeError("fewer than two valid current-attempt candidates")
    # Keep the complete current-scope namespace, including the publication
    # alias.  Stage three must prove that its caller did not cherry-pick a
    # subset of files out of the successful run.
    current_scope_hashes = {
        candidate.name: _sha256_file(candidate) for candidate in candidates
    }
    candidate_hashes = {
        candidate.name: _sha256_file(candidate)
        for candidate in candidates
        if candidate.name != "last.safetensors"
    }
    expected_candidate_steps = []
    for candidate_name in candidate_hashes:
        if candidate_name == f"{args.expected_repo_name}.safetensors":
            expected_candidate_steps.append(args.steps)
            continue
        match = re.fullmatch(
            rf"{re.escape(args.expected_repo_name)}_(\d+)\.safetensors",
            candidate_name,
        )
        if match is None:
            raise RuntimeError(f"unexpected current-run candidate: {candidate_name}")
        expected_candidate_steps.append(int(match.group(1)))
    if (
        sorted(set(expected_candidate_steps))
        != execution_plan["schedule"]["candidate_steps"]
    ):
        raise RuntimeError("current-run candidates differ from the sealed grid")
    telemetry = _read_json(telemetry_path)
    _validate_telemetry(telemetry, args=args)
    public_telemetry = _read_json(public_telemetry_path)
    if (
        set(public_telemetry) != {"schema", "kind", "private_record_sha256", "events"}
        or public_telemetry.get("schema") != 2
        or public_telemetry.get("kind") != "forge-public-run-recorder"
        or public_telemetry.get("private_record_sha256") != _sha256_file(telemetry_path)
        or not isinstance(public_telemetry.get("events"), list)
    ):
        raise RuntimeError("public telemetry is not the strict hash-bound projection")
    if any(
        not isinstance(event, dict)
        or not {"t", "name"}.issubset(event)
        or not set(event).issubset({"t", "name", "failure_class"})
        for event in public_telemetry["events"]
    ):
        raise RuntimeError("public telemetry event schema is not minimal")
    telemetry_meta = telemetry["meta"]
    if (
        telemetry_meta.get("holdout_pairs") != 0
        or telemetry_meta.get("pairs") != captured["num_images"]
        or telemetry_meta.get("num_images") != captured["num_images"]
        or telemetry_meta.get("total_pairs") != captured["num_images"]
    ):
        raise RuntimeError(
            "telemetry/config dataset counts disagree or proxy split ran"
        )

    exact_final = Path(spec.save_root) / f"{args.expected_repo_name}.safetensors"
    last_sha = _sha256_file(last_path)
    if not exact_final.is_file() or exact_final.name not in candidate_hashes:
        raise RuntimeError("clean completion did not leave the requested exact final")
    exact_final_sha = candidate_hashes[exact_final.name]
    if not valid_safetensors(last_path) or last_sha != exact_final_sha:
        raise RuntimeError("publication alias does not equal the exact natural final")

    # Verify that the resolved config references exactly the pre-hashed Krea
    # model/TE assets (the VAE intentionally aliases the base model directory).
    model_cfg = process["model"]
    kwargs = model_cfg.get("model_kwargs", {})
    if (
        Path(model_cfg.get("name_or_path", "")).resolve()
        != base_paths["base_model"].resolve()
        or Path(kwargs.get("vae_path", "")).resolve()
        != base_paths["base_model"].resolve()
        or Path(kwargs.get("text_encoder_path", "")).resolve()
        != base_paths["text_encoder"].resolve()
    ):
        raise RuntimeError("resolved config references unexpected model assets")

    post = {
        "dataset": _fingerprint_path(dataset_zip),
        "base_assets": {
            label: _fingerprint_path(path) for label, path in base_paths.items()
        },
        "code": {
            "forge": _code_fingerprint(forge_root),
            "ai_toolkit": _code_fingerprint(ai_toolkit_dir),
        },
        "runtime": _runtime_fingerprint(),
    }
    post["training_identity_sha256"] = _canonical_hash(
        {
            label: _content_identity(value)
            for label, value in sorted(post["base_assets"].items())
        }
    )
    if post != pre:
        raise RuntimeError(
            "dataset/base/code/runtime provenance changed during training"
        )

    normalized = _normalized_config(loaded_config)
    envelope = {
        **campaign_fixed_prefix,
        "normalized_control_config_sha256": _canonical_hash(normalized),
    }
    baseline = _campaign_baseline(campaign_dir, envelope)

    condition_record = {
        "schema": 2,
        "kind": (
            "forge-krea2-bootstrap-timing-run"
            if timing_bootstrap
            else "forge-krea2-calibration-run"
        ),
        "complete": True,
        "arm_id": execution_plan["arm_id"],
        "task_id": args.task_id,
        "expected_repo_name": args.expected_repo_name,
        "model": args.model,
        "execution_plan_sha256": execution_plan["plan_sha256"],
        "execution_plan_file_sha256": execution_plan_file_sha,
        "execution_approval_sha256": execution_approval["approval_sha256"],
        "execution_approval_file_sha256": execution_approval_file_sha,
        "discovery_profile_index_sha256": (
            execution_controls["discovery_profile_index"]["index_sha256"]
            if not timing_bootstrap
            else None
        ),
        "discovery_profile_index_file_sha256": (
            execution_controls["discovery_profile_index"]["file_sha256"]
            if not timing_bootstrap
            else None
        ),
        "discovery_execution_authorization_sha256": execution_controls[
            "discovery_execution_authorization"
        ]["authorization_sha256"],
        "discovery_execution_authorization_file_sha256": execution_controls[
            "discovery_execution_authorization"
        ]["file_sha256"],
        "host_bootstrap_receipt_sha256": execution_controls["host_execution_manifest"][
            "bootstrap_receipt"
        ]["receipt_sha256"],
        "host_bootstrap_receipt_file_sha256": execution_controls[
            "host_execution_manifest"
        ]["bootstrap_receipt"]["file_sha256"],
        "timing_probe_contract_sha256": (
            probe["probe_contract_sha256"] if timing_bootstrap else None
        ),
        "timing_capture_id": timing_capture_id if timing_bootstrap else None,
        "namespace_claim_sha256": _sha256_file(condition_claim_path),
        "in_task_proxy_selection": {"enabled": False, "reserve_s": 0},
        "axes": {
            "lr": args.lr,
            "depth_steps": args.steps,
            "guidance": args.guidance,
        },
        "fixed_controls": {
            "hours": args.hours,
            "training_seed": args.seed,
            "pythonhashseed": int(os.environ["PYTHONHASHSEED"]),
            "ai_toolkit_training_seed_supported": training_seed["supported"],
            "ai_toolkit_training_seed": (
                process.get("training_seed") if training_seed["supported"] else None
            ),
        },
        "derived": {"save_every": process["save"]["save_every"]},
        "budget": {
            "throughput_profile_path": (
                str(profile_path) if profile_path is not None else None
            ),
            "throughput_profile_file_sha256": (
                _sha256_file(profile_path) if profile_path is not None else None
            ),
            "throughput_profile": profile.to_record(),
            "plan": budget_plan.to_record(),
            "plan_sha256": _canonical_hash(budget_plan.to_record()),
        },
        "allowed_condition_config_differences": allowed_differences,
        "builder_mutations_from_production_config": captured["builder_mutations"],
        "resolved_config": loaded_config,
        "resolved_config_canonical_sha256": _canonical_hash(loaded_config),
        "resolved_config_file_sha256": _sha256_file(config_path),
        "normalized_control_config_sha256": _canonical_hash(normalized),
        "campaign_baseline": baseline,
        "provenance": {
            "dataset_path": str(dataset_zip.resolve()),
            "dataset": pre["dataset"],
            "base_asset_paths": {
                label: str(path.resolve()) for label, path in base_paths.items()
            },
            "base_assets": pre["base_assets"],
            "code": pre["code"],
            "runtime": pre["runtime"],
            "host_execution_manifest_sha256": execution_controls[
                "host_execution_manifest"
            ]["host_execution_identity_sha256"],
            "host_preflight_before": host_preflight_before,
            "host_observation_after": krea_host_identity.verify_static(
                execution_controls["host_execution_manifest"],
                checkpoint_path=checkpoint_parent,
            ),
        },
        "attempt": {
            "attempt_nonce": scope["attempt_nonce"],
            "process_nonce": scope["process_nonce"],
            "scope_started_unix": scope["started_unix"],
            "planned_steps": scope["planned_steps"],
            "training_pairs": captured["num_images"],
            "recipe_hours_after_scoring_reserve": captured["derived_hours"],
        },
        "dataset_after_split": {
            "training": _fingerprint_path(Path(spec.dataset_images_dir)),
            "approved_exact_evaluation_sha256": execution_controls["fixture"][
                "evaluation_dataset_identity"
            ]["sha256"],
        },
        "artifacts": {
            "scope_sha256": _sha256_file(scope_path),
            "telemetry_sha256": _sha256_file(telemetry_path),
            "public_telemetry_sha256": _sha256_file(public_telemetry_path),
            "last_sha256": last_sha,
            "candidate_sha256": candidate_hashes,
        },
        "current_scope_candidates": current_scope_hashes,
        "telemetry": telemetry,
        "verified_unix": int(time.time()),
    }

    output_condition_path = Path(spec.save_root) / _CONDITION_NAME
    if _lexists(output_condition_path) or _lexists(condition_path):
        raise FileExistsError("condition record already exists; refusing overwrite")
    # The campaign copy is authoritative and is written last.  A crash between
    # writes leaves no campaign-accepted condition and forces manual audit.
    _atomic_json(output_condition_path, condition_record)
    _atomic_json(condition_path, condition_record)
    output_sha = _sha256_file(output_condition_path)
    campaign_sha = _sha256_file(condition_path)
    if output_sha != campaign_sha:
        raise RuntimeError("durable output/campaign condition records differ")

    if timing_bootstrap:
        krea_timing_probe.emit_marker(
            observation_id=upload_observation_id,
            metric="upload",
            state="end",
        )
        print(
            json.dumps(
                {
                    "task_id": args.task_id,
                    "timing_probe_contract_sha256": probe["probe_contract_sha256"],
                    "condition_record": str(condition_path),
                    "condition_record_sha256": campaign_sha,
                    "checkpoint_write_observations": checkpoint_observer.sequence,
                },
                sort_keys=True,
            )
        )
        return 0

    evidence_bundle = krea_training_evidence.emit_run_evidence(
        condition_record_path=condition_path,
        execution_plan_path=execution_plan_path,
        execution_approval_path=execution_approval_path,
        candidate_paths=candidates,
        training_dir=Path(spec.dataset_images_dir),
        output_dir=evidence_dir,
    )

    print(
        json.dumps(
            {
                "task_id": args.task_id,
                "campaign_envelope_sha256": baseline["envelope_sha256"],
                "condition_record_sha256": campaign_sha,
                "candidates": len(candidate_hashes),
                "evidence_bundle": str(evidence_dir / "bundle.json"),
                "bundle_sha256": evidence_bundle["bundle_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
