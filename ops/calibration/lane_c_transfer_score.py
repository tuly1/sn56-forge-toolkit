#!/usr/bin/env python3
"""Run the Week-6 Lane-C reduced-2 transfer screen on one GPU.

This is deliberately a narrow campaign runner around ``evaluate_krea_local``.
It accepts a frozen manifest of exactly fifteen candidate files, stages and
hash-verifies one candidate at a time, starts a fresh evaluator process for
each candidate, and publishes an aggregate only after every result passes the
reduced-2 and fixture-order contract.

The output directory is create-only.  A failed or interrupted run therefore
leaves diagnostic evidence but can never be mistaken for a complete campaign.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import time
from typing import Any


_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_MARKER = "put_loras_here"
_GENERATION_COUNT = 2
_COMPARISONS_PER_ROW = 2
_REDUCED_SEEDS = [2746317213, 1181241943]
_DEFAULT_EXPECTED_CANDIDATES = 15
_LOOPBACK = "127.0.0.1"


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be in [1, 65535]")
    return parsed


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-image-order", required=True)
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument("--comfy-python", required=True)
    parser.add_argument("--driver-python", required=True)
    parser.add_argument("--god-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-god-commit", required=True)
    parser.add_argument("--expected-comfy-commit", required=True)
    parser.add_argument("--expected-tooling-commit", required=True)
    parser.add_argument(
        "--expected-candidate-count",
        type=_positive_int,
        default=_DEFAULT_EXPECTED_CANDIDATES,
    )
    parser.add_argument("--gpu-index", type=_nonnegative_int, default=0)
    parser.add_argument("--port", type=_port, default=8188)
    parser.add_argument(
        "--base-name", default="krea2_raw_fp8_scaled.safetensors"
    )
    parser.add_argument("--startup-timeout-s", type=_positive_float, default=300.0)
    parser.add_argument(
        "--evaluation-timeout-s", type=_positive_float, default=3600.0
    )
    parser.add_argument("--shutdown-timeout-s", type=_positive_float, default=20.0)
    parser.add_argument(
        "--port-reuse-timeout-s", type=_positive_float, default=90.0
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def _read_stable_regular(path: Path, label: str) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"cannot open {label} as a regular file: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in identity):
            raise RuntimeError(f"{label} changed while it was read: {path}")
        data = b"".join(chunks)
        if len(data) != before.st_size:
            raise RuntimeError(f"{label} size changed while it was read: {path}")
        return data, before
    finally:
        os.close(descriptor)


def _read_json(path: Path, label: str) -> tuple[Any, str]:
    data, _ = _read_stable_regular(path, label)
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    return value, hashlib.sha256(data).hexdigest()


def _load_candidates(
    manifest_path: Path, *, expected_count: int
) -> tuple[list[dict[str, Any]], str]:
    value, file_sha256 = _read_json(manifest_path, "candidate manifest")
    if not isinstance(value, dict) or set(value) != {"schema", "kind", "candidates"}:
        raise ValueError("candidate manifest schema mismatch")
    if value["schema"] != 1 or value["kind"] != "sn56-lane-c-transfer-candidates":
        raise ValueError("candidate manifest identity mismatch")
    rows = value["candidates"]
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(
            f"candidate manifest must contain exactly {expected_count} rows"
        )
    candidates: list[dict[str, Any]] = []
    ids: set[str] = set()
    paths: set[Path] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"id", "path", "sha256"}:
            raise ValueError(f"candidate row {index} schema mismatch")
        candidate_id = row["id"]
        raw_path = row["path"]
        expected_sha = row["sha256"]
        if not isinstance(candidate_id, str) or not _SAFE_ID.fullmatch(candidate_id):
            raise ValueError(f"candidate row {index} has an unsafe id")
        if candidate_id in ids:
            raise ValueError(f"duplicate candidate id: {candidate_id}")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise ValueError(f"candidate {candidate_id} path must be absolute")
        candidate_path = Path(raw_path)
        if candidate_path in paths:
            raise ValueError(f"duplicate candidate path: {candidate_path}")
        if not isinstance(expected_sha, str) or not _SHA256.fullmatch(expected_sha):
            raise ValueError(f"candidate {candidate_id} SHA-256 is invalid")
        ids.add(candidate_id)
        paths.add(candidate_path)
        candidates.append(
            {"id": candidate_id, "path": candidate_path, "sha256": expected_sha}
        )
    return candidates, file_sha256


def _load_expected_order(path: Path) -> tuple[list[str], str]:
    data, _ = _read_stable_regular(path, "expected image order")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("expected image order is not UTF-8") from exc
    names = text.splitlines()
    if (
        not names
        or any(not name or Path(name).name != name for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("expected image order must contain unique basenames")
    return names, hashlib.sha256(data).hexdigest()


def _assert_commit(value: str, label: str) -> None:
    if not _COMMIT.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase forty-hex commit")


def _absolute_lexical(value: str) -> Path:
    """Make a path absolute without collapsing a venv interpreter symlink."""

    return Path(os.path.abspath(os.path.expanduser(value)))


def _assert_lora_root_empty(lora_root: Path) -> None:
    if not lora_root.is_dir() or lora_root.is_symlink():
        raise RuntimeError("Comfy LoRA root must be a real directory")
    entries = sorted(lora_root.iterdir(), key=lambda item: item.name)
    if [entry.name for entry in entries] != [_MARKER]:
        raise RuntimeError(
            "Comfy LoRA root must contain only its tracked empty marker"
        )
    marker = entries[0]
    marker_stat = marker.lstat()
    if (
        not stat.S_ISREG(marker_stat.st_mode)
        or marker_stat.st_size != 0
        or marker_stat.st_nlink != 1
    ):
        raise RuntimeError("Comfy LoRA marker identity is invalid")


def _stage_candidate(
    candidate: dict[str, Any], lora_root: Path
) -> tuple[Path, int]:
    source = candidate["path"]
    source_data, source_stat = _read_stable_regular(source, "candidate adapter")
    observed_sha = hashlib.sha256(source_data).hexdigest()
    if observed_sha != candidate["sha256"]:
        raise RuntimeError(f"candidate {candidate['id']} source SHA-256 mismatch")
    target = lora_root / f"candidate-{observed_sha}.safetensors"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o400)
    except OSError as exc:
        raise RuntimeError(f"cannot create candidate staging file: {target}") from exc
    try:
        offset = 0
        while offset < len(source_data):
            offset += os.write(descriptor, source_data[offset : offset + 1024 * 1024])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        staged_stat = os.fstat(descriptor)
        if not stat.S_ISREG(staged_stat.st_mode) or staged_stat.st_size != len(source_data):
            raise RuntimeError("staged candidate identity is invalid")
    except Exception:
        os.close(descriptor)
        target.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    if _sha256(target) != observed_sha:
        target.unlink(missing_ok=True)
        raise RuntimeError("staged candidate SHA-256 mismatch")
    return target, source_stat.st_size


def _fixed_environment(driver_python: Path, output_dir: Path) -> dict[str, str]:
    runtime_home = output_dir / ".runner-home"
    runtime_home.mkdir(mode=0o700)
    return {
        "PATH": f"{driver_python.parent}:/usr/local/bin:/usr/bin:/bin",
        "HOME": str(runtime_home),
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def _wait_for_bindable_port(port: int, *, timeout_s: float) -> dict[str, Any]:
    """Wait until the evaluator's own bind-based freshness gate can pass.

    A clean ComfyUI shutdown can leave the listening port unavailable during
    the kernel's socket teardown window even though no process is listening.
    The evaluator deliberately fails closed on that state.  Mirror its exact
    bind test here between candidates instead of converting a transient
    teardown state into a failed campaign.
    """

    started = time.monotonic()
    deadline = started + timeout_s
    attempts = 0
    while True:
        attempts += 1
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((_LOOPBACK, port))
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise RuntimeError(
                        f"cannot verify ComfyUI port {_LOOPBACK}:{port}"
                    ) from exc
                now = time.monotonic()
                if now >= deadline:
                    raise RuntimeError(
                        f"ComfyUI port {_LOOPBACK}:{port} remained unavailable "
                        f"for {timeout_s}s"
                    ) from exc
            else:
                return {
                    "attempts": attempts,
                    "waited_s": time.monotonic() - started,
                }
        time.sleep(0.25)


def _command(
    args: argparse.Namespace,
    *,
    evaluator_script: Path,
    staged_candidate: Path,
    result_path: Path,
    comfy_log: Path,
    expected_order: list[str],
) -> list[str]:
    command = [
        str(Path(args.driver_python)),
        str(evaluator_script),
        "--dataset",
        str(Path(args.dataset)),
        "--candidate-path",
        str(staged_candidate),
        "--comfy-root",
        str(Path(args.comfy_root)),
        "--comfy-python",
        str(Path(args.comfy_python)),
        "--god-root",
        str(Path(args.god_root)),
        "--output",
        str(result_path),
        "--comfy-log",
        str(comfy_log),
        "--base-name",
        args.base_name,
        "--port",
        str(args.port),
        "--gpu-index",
        str(args.gpu_index),
        "--startup-timeout-s",
        str(args.startup_timeout_s),
        "--evaluation-timeout-s",
        str(args.evaluation_timeout_s),
        "--shutdown-timeout-s",
        str(args.shutdown_timeout_s),
        "--expected-god-commit",
        args.expected_god_commit,
        "--expected-comfy-commit",
        args.expected_comfy_commit,
        "--expected-tooling-commit",
        args.expected_tooling_commit,
        "--generations",
        str(_GENERATION_COUNT),
    ]
    for name in expected_order:
        command.extend(("--expected-image", name))
    return command


def _validate_result(
    result_path: Path,
    *,
    candidate: dict[str, Any],
    staged_candidate: Path,
    dataset: Path,
    expected_order: list[str],
    expected_prompt_count: int,
    gpu_index: int,
    candidate_bytes: int,
    expected_dataset_sha256: str,
) -> dict[str, Any]:
    result, result_file_sha = _read_json(result_path, "candidate result")
    if not isinstance(result, dict):
        raise ValueError("candidate result must be a JSON object")
    scored_rows = result.get("scored_rows")
    runtime = result.get("runtime")
    history = runtime.get("comfy_history") if isinstance(runtime, dict) else None
    checks = {
        "schema": result.get("schema") == 2,
        "evaluator": result.get("evaluator") == "god_krea2_img2img_exact",
        "candidate_sha256": result.get("candidate_sha256") == candidate["sha256"],
        "staged_candidate_sha256": (
            result.get("staged_candidate_sha256") == candidate["sha256"]
        ),
        "candidate_name": result.get("candidate") == staged_candidate.name,
        "candidate_bytes": result.get("candidate_bytes") == candidate_bytes,
        "comfy_lora_name": (
            result.get("comfy_lora_name") == staged_candidate.name
        ),
        "model_type": result.get("model_type") == "krea2",
        "direction": result.get("direction") == "min",
        "dataset": (
            isinstance(result.get("dataset"), str)
            and Path(result["dataset"]).resolve() == dataset
        ),
        "dataset_sha256": result.get("dataset_sha256") == expected_dataset_sha256,
        "image_count": result.get("image_count") == len(expected_order),
        "scored_order": (
            isinstance(scored_rows, list)
            and [row.get("image") for row in scored_rows if isinstance(row, dict)]
            == expected_order
        ),
        "generations": result.get("generations") == _GENERATION_COUNT,
        "validator_default_generations": (
            result.get("validator_default_generations") == 5
        ),
        "seed_mode": result.get("seed_mode") == "reduced-2",
        "seed_schedule": result.get("seeds") == _REDUCED_SEEDS,
        "master_seed": result.get("master_seed") == 42,
        "prompt_count": (
            isinstance(history, dict)
            and history.get("prompt_count") == expected_prompt_count
        ),
        "gpu_index": (
            isinstance(runtime, dict)
            and runtime.get("physical_gpu_index") == gpu_index
            and runtime.get("cuda_visible_devices") == str(gpu_index)
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            f"candidate {candidate['id']} result contract failed: {failed}"
        )
    weighted_loss = result.get("weighted_loss")
    if (
        isinstance(weighted_loss, bool)
        or not isinstance(weighted_loss, (int, float))
        or not math.isfinite(float(weighted_loss))
        or not 0 <= float(weighted_loss) <= 1
    ):
        raise RuntimeError(f"candidate {candidate['id']} weighted loss is invalid")
    return {
        "id": candidate["id"],
        "candidate_sha256": candidate["sha256"],
        "result_path": str(result_path),
        "result_file_sha256": result_file_sha,
        "weighted_loss": weighted_loss,
        "elapsed_s": result.get("elapsed_s"),
    }


def _publish_json(path: Path, value: Any) -> None:
    data = json.dumps(
        value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
    ).encode("ascii") + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run(args: argparse.Namespace) -> dict[str, Any]:
    for name in (
        "expected_god_commit",
        "expected_comfy_commit",
        "expected_tooling_commit",
    ):
        _assert_commit(getattr(args, name), name)
    if not _SHA256.fullmatch(args.expected_dataset_sha256):
        raise ValueError("expected_dataset_sha256 must be lowercase sixty-four hex")
    manifest_path = Path(args.candidate_manifest).expanduser().resolve(strict=True)
    order_path = Path(args.expected_image_order).expanduser().resolve(strict=True)
    dataset = Path(args.dataset).expanduser().resolve(strict=True)
    comfy_root = Path(args.comfy_root).expanduser().resolve(strict=True)
    god_root = Path(args.god_root).expanduser().resolve(strict=True)
    driver_python = _absolute_lexical(args.driver_python)
    comfy_python = _absolute_lexical(args.comfy_python)
    output_dir = Path(args.output_dir).expanduser().absolute()
    evaluator_script = Path(__file__).with_name("evaluate_krea_local.py").resolve(
        strict=True
    )
    if not dataset.is_dir() or not comfy_root.is_dir() or not god_root.is_dir():
        raise ValueError("dataset, Comfy, and G.O.D paths must be directories")
    if not driver_python.is_file() or not comfy_python.is_file():
        raise ValueError("driver and Comfy Python paths must be files")
    if output_dir.exists():
        raise FileExistsError(f"refusing stale output directory: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"output parent is missing: {output_dir.parent}")

    candidates, manifest_file_sha = _load_candidates(
        manifest_path, expected_count=args.expected_candidate_count
    )
    expected_order, order_file_sha = _load_expected_order(order_path)
    # Commands execute from the G.O.D checkout.  Normalize every configured
    # path before that cwd transition so a relative operator spelling cannot
    # silently select a different object.
    args.dataset = str(dataset)
    args.comfy_root = str(comfy_root)
    args.comfy_python = str(comfy_python)
    args.driver_python = str(driver_python)
    args.god_root = str(god_root)
    expected_prompt_count = (
        len(expected_order) * _GENERATION_COUNT * _COMPARISONS_PER_ROW
    )
    lora_root = comfy_root / "models" / "loras"
    _assert_lora_root_empty(lora_root)
    output_dir.mkdir(mode=0o700)
    results_dir = output_dir / "results"
    logs_dir = output_dir / "logs"
    results_dir.mkdir(mode=0o700)
    logs_dir.mkdir(mode=0o700)
    environment = _fixed_environment(driver_python, output_dir)
    completed: list[dict[str, Any]] = []

    for candidate in candidates:
        _assert_lora_root_empty(lora_root)
        port_gate = _wait_for_bindable_port(
            args.port, timeout_s=args.port_reuse_timeout_s
        )
        staged_candidate, source_bytes = _stage_candidate(candidate, lora_root)
        result_path = results_dir / f"{candidate['id']}.json"
        comfy_log = logs_dir / f"{candidate['id']}.comfy.log"
        driver_log = logs_dir / f"{candidate['id']}.driver.log"
        command = _command(
            args,
            evaluator_script=evaluator_script,
            staged_candidate=staged_candidate,
            result_path=result_path,
            comfy_log=comfy_log,
            expected_order=expected_order,
        )
        try:
            with driver_log.open("xb") as log_handle:
                completed_process = subprocess.run(
                    command,
                    cwd=str(god_root),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                log_handle.flush()
                os.fsync(log_handle.fileno())
            if completed_process.returncode != 0:
                raise RuntimeError(
                    f"candidate {candidate['id']} evaluator exited "
                    f"{completed_process.returncode}"
                )
            if source_bytes <= 0 or _sha256(staged_candidate) != candidate["sha256"]:
                raise RuntimeError(
                    f"candidate {candidate['id']} staging changed during scoring"
                )
            summary = _validate_result(
                result_path,
                candidate=candidate,
                staged_candidate=staged_candidate,
                dataset=dataset,
                expected_order=expected_order,
                expected_prompt_count=expected_prompt_count,
                gpu_index=args.gpu_index,
                candidate_bytes=source_bytes,
                expected_dataset_sha256=args.expected_dataset_sha256,
            )
            summary["driver_log_sha256"] = _sha256(driver_log)
            summary["comfy_log_sha256"] = _sha256(comfy_log)
            summary["prelaunch_port_gate"] = port_gate
            completed.append(summary)
            print(
                f"LANE-C DONE {candidate['id']} "
                f"weighted_loss={summary['weighted_loss']}",
                flush=True,
            )
        finally:
            staged_candidate.unlink(missing_ok=True)
            _assert_lora_root_empty(lora_root)

    aggregate = {
        "schema": 1,
        "kind": "sn56-lane-c-transfer-reduced-2-results",
        "status": "complete",
        "candidate_manifest": str(manifest_path),
        "candidate_manifest_file_sha256": manifest_file_sha,
        "expected_image_order": str(order_path),
        "expected_image_order_file_sha256": order_file_sha,
        "dataset": str(dataset),
        "expected_dataset_sha256": args.expected_dataset_sha256,
        "candidate_count": len(completed),
        "image_count": len(expected_order),
        "generations": _GENERATION_COUNT,
        "seed_mode": "reduced-2",
        "expected_prompt_count_per_candidate": expected_prompt_count,
        "gpu_index": args.gpu_index,
        "port_reuse_timeout_s": args.port_reuse_timeout_s,
        "evaluator_script_sha256": _sha256(evaluator_script),
        "results": completed,
    }
    aggregate["aggregate_sha256"] = _canonical_sha256(aggregate)
    _publish_json(output_dir / "aggregate.json", aggregate)
    return aggregate


def main() -> int:
    aggregate = run(_parse())
    print(json.dumps(aggregate, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
