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
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any


_MODEL = "krea/Krea-2-Raw"
_KREA_TEXT_ENCODER = "/cache/hf_cache/Qwen--Qwen3-VL-4B-Instruct"
_CONDITION_NAME = "forge_calibration_condition.json"
_BASELINE_NAME = "krea_ladder_baseline.json"
_REEXEC_MARKER = "FORGE_KREA_CALIBRATION_SEEDED"
_PROBE_SEED = 42565431
_PROBE_EPOCHS = 2
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")

_LR_POINTER = "/config/process/0/train/lr"
_DEPTH_POINTER = "/config/process/0/train/steps"
_GUIDANCE_POINTER = "/config/process/0/train/do_differential_guidance"
_GUIDANCE_SCALE_POINTER = "/config/process/0/train/differential_guidance_scale"
_SAVE_CADENCE_POINTER = "/config/process/0/save/save_every"
_TRAINING_SEED_POINTER = "/config/process/0/training_seed"
_RUN_NAME_POINTER = "/config/name"
_TRAINING_FOLDER_POINTER = "/config/process/0/training_folder"

_SCIENTIFIC_AXIS_POINTERS = (
    _LR_POINTER,
    _DEPTH_POINTER,
    _GUIDANCE_POINTER,
    _GUIDANCE_SCALE_POINTER,
)
_DERIVED_POINTERS = (_SAVE_CADENCE_POINTER,)
_ISOLATION_POINTERS = (_RUN_NAME_POINTER, _TRAINING_FOLDER_POINTER)
_ALLOWED_BUILDER_MUTATIONS = frozenset(
    (*_SCIENTIFIC_AXIS_POINTERS, *_DERIVED_POINTERS, _TRAINING_SEED_POINTER)
)


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one clean-room Krea2 ladder condition."
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--expected-repo-name", required=True)
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument(
        "--steps", "--depth-steps", dest="steps", required=True, type=int
    )
    parser.add_argument("--guidance", choices=("on", "off"), required=True)
    parser.add_argument("--hours", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
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


def _ensure_seeded_process(seed: int) -> None:
    """Restart before Forge imports so PYTHONHASHSEED covers this process too."""
    expected = str(seed)
    if os.environ.get("PYTHONHASHSEED") == expected:
        os.environ["SEED"] = expected
        os.environ[_REEXEC_MARKER] = expected
        return
    if os.environ.get(_REEXEC_MARKER):
        raise RuntimeError("seeded re-exec did not install the requested hash seed")
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = expected
    env["SEED"] = expected
    env[_REEXEC_MARKER] = expected
    os.execve(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
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
            "git", "-C", str(root), "ls-files", "--cached", "--others",
            "--exclude-standard", "-z",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    ).stdout
    allowed_suffixes = {
        ".py", ".yaml", ".yml", ".json", ".toml", ".sh", ".md", ".txt"
    }
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
    payload = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable_sha256": _sha256_file(executable),
        "packages_sha256": _canonical_hash(sorted(packages.items())),
        "package_count": len(packages),
        "gpu": gpu,
        "seed_env": os.environ.get("SEED"),
        "pythonhashseed_env": os.environ.get("PYTHONHASHSEED"),
    }
    return {**payload, "sha256": _canonical_hash(payload)}


def _training_seed_support(ai_toolkit_dir: Path) -> dict[str, Any]:
    source = ai_toolkit_dir / "jobs/process/BaseTrainProcess.py"
    if not source.is_file():
        return {"supported": False, "source": None, "source_sha256": None}
    text = source.read_text(encoding="utf-8", errors="strict")
    supported = bool(
        re.search(r"get_conf\(\s*['\"]training_seed['\"]", text)
    )
    return {
        "supported": supported,
        "source": "jobs/process/BaseTrainProcess.py",
        "source_sha256": _sha256_file(source),
    }


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
    train["lr"] = "<axis:lr>"
    train["steps"] = "<axis:depth-steps>"
    train.pop("do_differential_guidance", None)
    train.pop("differential_guidance_scale", None)
    train["__forge_guidance_axis__"] = "<axis:guidance>"
    process["save"]["save_every"] = "<derived:kill-safe-save-cadence>"
    return normalized


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical_bytes(value))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
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


def _campaign_baseline(
    campaign_dir: Path, envelope: dict[str, Any]
) -> dict[str, Any]:
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
    return {
        key: value[key]
        for key in ("kind", "sha256", "files", "bytes", "symlinks")
    }


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
    if (
        not isinstance(created, int)
        or created < int(float(scope["started_unix"]))
    ):
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
            "score", "captioned_score", "blank_caption_score",
            "captioned_stddev", "blank_caption_stddev",
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
        if not math.isclose(
            values["score"], recomputed, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise RuntimeError(f"manifest aggregate does not recompute for {name!r}")
        seen[name] = actual
    if set(seen) != set(candidate_by_name):
        raise RuntimeError("manifest does not cover every current-run candidate")
    return seen


def _validate_telemetry(
    telemetry: dict[str, Any], *, args: argparse.Namespace, candidate_count: int
) -> None:
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
        name for name in names
        if isinstance(name, str)
        and (name.endswith("_failed") or "fallback" in name or name.endswith("_skipped"))
    ]
    if forbidden:
        raise RuntimeError(f"failure/fallback telemetry present: {sorted(set(forbidden))}")
    required = {
        "checkpoint_scope_started", "dataset_ready", "holdout_reserved",
        "toolkit_start", "toolkit_end", "toolkit_metrics",
        "holdout_manifest_complete", "checkpoint_selected",
        "checkpoint_finalized", "run_complete",
    }
    missing = sorted(name for name in required if names.count(name) != 1)
    if missing:
        raise RuntimeError(f"required telemetry events are absent/non-unique: {missing}")
    if names.count("holdout_candidate_scored") != candidate_count:
        raise RuntimeError("telemetry candidate count differs from the manifest")
    toolkit_end = next(event for event in events if event.get("name") == "toolkit_end")
    if toolkit_end.get("returncode") != 0 or toolkit_end.get("stopped_by_deadline") is not False:
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
    _validate_args(args)
    _ensure_seeded_process(args.seed)

    # Heavy/project imports intentionally occur only after seeded re-exec.
    import yaml

    from forge import recipe
    from forge.cli import main as forge_main
    from forge.data.schema import ImageSpec
    from forge.tasks import aitoolkit, checkpoints
    from forge.tasks.integrity import valid_safetensors

    campaign_dir = args.campaign_dir.expanduser().resolve()
    for volatile_root in (Path("/app/checkpoints"), Path("/dataset"), Path("/cache")):
        if campaign_dir == volatile_root or campaign_dir.is_relative_to(volatile_root):
            raise ValueError(
                "campaign-dir must be durable and outside trainer/cache paths: "
                f"{campaign_dir}"
            )
    condition_path = campaign_dir / "conditions" / f"{args.task_id}.json"
    spec = ImageSpec.build(
        task_id=args.task_id,
        model=args.model,
        model_type="krea2",
        expected_repo_name=args.expected_repo_name,
        trigger_word=None,
        dataset_zip=None,
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
    ]
    _require_clean_paths(mutable_paths)

    dataset_zip = Path(spec.cached_zip_path)
    if dataset_zip.is_symlink() or not dataset_zip.is_file():
        raise FileNotFoundError(f"staged dataset must be a regular file: {dataset_zip}")
    ai_toolkit_dir = Path(
        os.environ.get("AI_TOOLKIT_DIR", "/app/ai-toolkit")
    ).resolve(strict=True)
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
        "allowed_condition_config_differences": allowed_differences,
        "holdout_activation": "krea2",
    }
    _validate_existing_campaign_prefix(campaign_dir, campaign_fixed_prefix)

    original_build_config = aitoolkit.build_config
    captured: dict[str, Any] = {}

    def _calibration_config(spec_arg, num_images, hours_to_complete):
        baseline = original_build_config(spec_arg, num_images, hours_to_complete)
        resolved = copy.deepcopy(baseline)
        process = resolved["config"]["process"][0]
        train = process["train"]
        train["steps"] = args.steps
        train["lr"] = args.lr
        if args.guidance == "on":
            train["do_differential_guidance"] = True
            train["differential_guidance_scale"] = 2
        else:
            train.pop("do_differential_guidance", None)
            train.pop("differential_guidance_scale", None)
        process["save"]["save_every"] = recipe.kill_safe_save_every(
            args.steps,
            process["save"].get("save_every", 250),
        )
        if training_seed["supported"]:
            # BaseTrainProcess reads this at process scope, not train scope.
            process["training_seed"] = args.seed
        changed = _diff_paths(baseline, resolved)
        unexpected = changed - _ALLOWED_BUILDER_MUTATIONS
        if unexpected:
            raise RuntimeError(
                f"calibration mutated non-axis config fields: {sorted(unexpected)}"
            )
        if process["save"]["save_every"] != recipe.kill_safe_save_every(
            args.steps, baseline["config"]["process"][0]["save"].get("save_every", 250)
        ):
            raise RuntimeError("save cadence is not the declared depth-derived value")
        captured.update(
            baseline=copy.deepcopy(baseline),
            resolved=copy.deepcopy(resolved),
            builder_mutations=sorted(changed),
            num_images=int(num_images),
            derived_hours=float(hours_to_complete),
        )
        return resolved

    os.environ["FORGE_HOLDOUT_SELECTION_TYPES"] = "krea2"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    aitoolkit.build_config = _calibration_config
    launch_not_before = time.time()
    try:
        return_code = forge_main(
            [
                "--task-id", args.task_id,
                "--model", args.model,
                "--model-type", "krea2",
                "--expected-repo-name", args.expected_repo_name,
                "--hours-to-complete", str(args.hours),
            ]
        )
    finally:
        aitoolkit.build_config = original_build_config
    if return_code != 0:
        raise RuntimeError(f"Forge returned nonzero status {return_code}")
    if not captured:
        raise RuntimeError("Forge never resolved the calibration config")

    config_path = Path(spec.config_path)
    manifest_path = Path(spec.save_root) / "forge_holdout_scores.json"
    selection_path = Path(spec.save_root) / "forge_checkpoint_selection.json"
    telemetry_path = Path(spec.save_root) / "forge_run.json"
    scope_path = Path(spec.save_root) / ".forge_checkpoint_scope.json"
    last_path = Path(spec.save_root) / "last.safetensors"
    selected_paths = [
        config_path, manifest_path, selection_path, telemetry_path, scope_path, last_path
    ]
    for path in selected_paths:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"required current-run output is absent/unsafe: {path}")

    loaded_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if loaded_config != captured["resolved"]:
        raise RuntimeError("persisted YAML differs from the resolved config")
    process = loaded_config["config"]["process"][0]
    if process["train"].get("steps") != args.steps or process["train"].get("lr") != args.lr:
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

    candidates = [Path(path) for path in checkpoints.current_loras(spec.save_root, scope)]
    if len(candidates) < 2 or not all(valid_safetensors(path) for path in candidates):
        raise RuntimeError("fewer than two valid current-attempt candidates")
    manifest = _read_json(manifest_path)
    candidate_hashes = _validate_manifest(
        manifest, args=args, scope=scope, candidates=candidates
    )
    selection = _read_json(selection_path)
    telemetry = _read_json(telemetry_path)
    _validate_telemetry(telemetry, args=args, candidate_count=len(candidates))
    telemetry_meta = telemetry["meta"]
    if (
        telemetry_meta.get("holdout_pairs") != manifest["holdout_pairs"]
        or telemetry_meta.get("pairs") != captured["num_images"]
        or telemetry_meta.get("num_images") != captured["num_images"]
        or telemetry_meta.get("total_pairs")
        != captured["num_images"] + manifest["holdout_pairs"]
    ):
        raise RuntimeError("telemetry/config/manifest dataset counts disagree")

    exact_final = Path(spec.save_root) / f"{args.expected_repo_name}.safetensors"
    selected_name = selection.get("selected_file")
    allowed_selection_sources = {"exact_final", "training_loss_divergence"}
    if (
        selection.get("schema") != 1
        or selection.get("status") != "selected_current_run"
        or selection.get("context") != "training"
        or selection.get("source") not in allowed_selection_sources
        or selection.get("output_file") != "last.safetensors"
        or selection.get("current_candidates_discovered") != len(candidates)
        or selection.get("current_candidates_valid") != len(candidates)
        or selected_name not in candidate_hashes
    ):
        raise RuntimeError("selection is not a permitted current-run candidate")
    selection_source = selection["source"]
    if selection_source == "exact_final":
        if selected_name != exact_final.name or selection.get("selected_step") != args.steps:
            raise RuntimeError("exact-final selection identity is inconsistent")
    else:
        selected_match = re.fullmatch(
            rf"{re.escape(args.expected_repo_name)}_(\d+)\.safetensors",
            str(selected_name),
        )
        if (
            selected_match is None
            or selection.get("selected_step") != int(selected_match.group(1))
            or selection.get("training_loss_is_proxy_not_validator_metric") is not True
        ):
            raise RuntimeError("divergence selection identity is inconsistent")
    if (
        not isinstance(selection.get("created_unix"), int)
        or selection["created_unix"] < int(float(scope["started_unix"]))
    ):
        raise RuntimeError("selection timestamp predates the active scope")
    last_sha = _sha256_file(last_path)
    if (
        not valid_safetensors(last_path)
        or last_sha != candidate_hashes[selected_name]
        or selection.get("sha256") != last_sha
    ):
        raise RuntimeError("selection and last.safetensors differ")
    if not exact_final.is_file() or exact_final.name not in candidate_hashes:
        raise RuntimeError("clean completion did not leave the requested exact final")

    # Verify that the resolved config references exactly the pre-hashed Krea
    # model/TE assets (the VAE intentionally aliases the base model directory).
    model_cfg = process["model"]
    kwargs = model_cfg.get("model_kwargs", {})
    if (
        Path(model_cfg.get("name_or_path", "")).resolve() != base_paths["base_model"].resolve()
        or Path(kwargs.get("vae_path", "")).resolve() != base_paths["base_model"].resolve()
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
    if post != pre:
        raise RuntimeError("dataset/base/code/runtime provenance changed during training")

    normalized = _normalized_config(loaded_config)
    envelope = {
        **campaign_fixed_prefix,
        "normalized_control_config_sha256": _canonical_hash(normalized),
    }
    baseline = _campaign_baseline(campaign_dir, envelope)

    condition_record = {
        "schema": 1,
        "kind": "forge-krea2-calibration-condition",
        "complete": True,
        "task_id": args.task_id,
        "expected_repo_name": args.expected_repo_name,
        "model": args.model,
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
            "holdout": _fingerprint_path(Path(spec.dataset_holdout_dir)),
        },
        "artifacts": {
            "scope_sha256": _sha256_file(scope_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "selection_sha256": _sha256_file(selection_path),
            "telemetry_sha256": _sha256_file(telemetry_path),
            "last_sha256": last_sha,
            "candidate_sha256": candidate_hashes,
        },
        "manifest_summary": {
            "metric": manifest["metric"],
            "probe_seed": manifest["seed"],
            "holdout_pairs": manifest["holdout_pairs"],
            "probe_epochs": manifest["probe_epochs"],
            "elapsed_s": manifest.get("elapsed_s"),
            "candidates": len(candidate_hashes),
        },
        # Keep the durable campaign copy self-contained; the H100 output tree is
        # useful corroboration but may be reclaimed after certification.
        "manifest": manifest,
        "selection": selection,
        "telemetry": telemetry,
        "selection_summary": {
            "status": selection["status"],
            "source": selection["source"],
            "selected_file": selection["selected_file"],
            "selected_step": selection["selected_step"],
            "sha256": selection["sha256"],
        },
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

    print(
        json.dumps(
            {
                "task_id": args.task_id,
                "campaign_envelope_sha256": baseline["envelope_sha256"],
                "condition_record_sha256": campaign_sha,
                "candidates": len(candidate_hashes),
                "selected_file": selection["selected_file"],
                "selection_source": selection["source"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
