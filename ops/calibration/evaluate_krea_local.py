#!/usr/bin/env python3
"""Certify one local Krea2 LoRA with G.O.D's pinned Comfy evaluator.

G.O.D normally discovers LoRAs through Hugging Face.  This shim changes only
that discovery step: the candidate must already be present under the supplied
ComfyUI checkout with a filename derived from its SHA-256.  The evaluator
workflow, preprocessing, seed schedule, inference loop, and loss calculation
are imported from the explicitly supplied G.O.D checkout.

The run is deliberately fail-closed.  It starts a fresh, loopback-only ComfyUI
process from an explicitly supplied virtual environment, uses fresh input,
output, temp, user, and history state, refuses dirty source trees or
stale evidence paths, binds every score to an ordered image/prompt pair, and
publishes the result without overwriting an existing file.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from types import ModuleType
from typing import Any, Iterator
import urllib.error
import urllib.request


_LOOPBACK = "127.0.0.1"
_DEFAULT_PORT = 8188
_TOOLING_NODE = "comfyui-tooling-nodes"


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be in [1, 65535]")
    return parsed


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument(
        "--comfy-python",
        required=True,
        help="Python executable inside the virtual environment used for ComfyUI",
    )
    parser.add_argument("--god-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--comfy-log",
        help="Fresh log path (default: <output>.comfy.log)",
    )
    parser.add_argument(
        "--base-name",
        default="models--Comfy-Org--Krea-2.safetensors",
    )
    parser.add_argument("--port", type=_port, default=_DEFAULT_PORT)
    parser.add_argument(
        "--startup-timeout-s",
        type=_positive_float,
        default=300.0,
    )
    parser.add_argument(
        "--evaluation-timeout-s",
        type=_positive_float,
        default=3600.0,
    )
    parser.add_argument(
        "--shutdown-timeout-s",
        type=_positive_float,
        default=20.0,
    )
    parser.add_argument("--expected-god-commit")
    parser.add_argument("--expected-comfy-commit")
    parser.add_argument("--expected-tooling-commit")
    return parser.parse_args()


def _absolute_lexical(value: str) -> Path:
    """Return an absolute path without erasing the identity of a final symlink."""

    return Path(os.path.abspath(os.path.expanduser(value)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_text(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        command,
        cwd=str(cwd) if cwd is not None else None,
        stderr=subprocess.STDOUT,
        text=True,
    ).strip()


def _git_snapshot(path: Path, *, expected_commit: str | None) -> dict[str, str]:
    if not (path / ".git").exists():
        # Worktrees can use a .git file, so ask git before declaring failure.
        try:
            _run_text(["git", "-C", str(path), "rev-parse", "--git-dir"])
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"not a Git checkout: {path}") from exc
    commit = _run_text(["git", "-C", str(path), "rev-parse", "HEAD"])
    tree = _run_text(["git", "-C", str(path), "rev-parse", "HEAD^{tree}"])
    object_type = _run_text(
        ["git", "-C", str(path), "cat-file", "-t", commit]
    )
    if object_type != "commit":
        raise RuntimeError(f"Git HEAD is not a commit in {path}")
    status = _run_text(
        [
            "git",
            "-C",
            str(path),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ]
    )
    if status:
        raise RuntimeError(f"Git worktree has non-ignored changes: {path}\n{status}")
    if expected_commit is not None and commit.lower() != expected_commit.lower():
        raise RuntimeError(
            f"Git commit mismatch for {path}: expected {expected_commit}, got {commit}"
        )
    return {"commit": commit, "tree": tree}


def _assert_git_unchanged(
    path: Path,
    before: dict[str, str],
    *,
    expected_commit: str | None,
) -> None:
    after = _git_snapshot(path, expected_commit=expected_commit)
    if after != before:
        raise RuntimeError(f"Git identity changed during evaluation: {path}")


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _module_source(module: ModuleType) -> Path:
    raw = getattr(module, "__file__", None)
    if not raw:
        raise RuntimeError(f"imported module has no source path: {module.__name__}")
    path = Path(raw).resolve()
    if path.suffix in {".pyc", ".pyo"}:
        try:
            path = Path(importlib.util.source_from_cache(str(path))).resolve()
        except ValueError as exc:
            raise RuntimeError(
                f"cannot resolve source for imported module: {module.__name__}"
            ) from exc
    if not path.is_file():
        raise RuntimeError(f"imported module source is missing: {path}")
    return path


def _import_god(god_root: Path) -> tuple[dict[str, ModuleType], dict[str, dict[str, str]]]:
    root_text = str(god_root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()

    names = {
        "constants": "validator.evaluation.constants",
        "image_models": "core.models.image_models",
        "diffusion": "validator.evaluation.evaluators.diffusion",
        "models": "validator.evaluation.models",
        "comfy_gateway": "validator.infrastructure.comfy_gateway",
        "image_io": "validator.evaluation.image_io",
        "dataset_constants": "validator.tasks.datasets.constants",
    }
    modules = {key: importlib.import_module(name) for key, name in names.items()}
    # Audit every transitive module in G.O.D's own namespaces, not only the
    # handful referenced below.  This catches a split import where, for
    # example, ``validator`` came from the supplied checkout but one ``core``
    # dependency was silently satisfied by site-packages.
    bindings: dict[str, dict[str, str]] = {}
    namespace_modules = {
        name: module
        for name, module in sys.modules.items()
        if (
            name == "validator"
            or name.startswith("validator.")
            or name == "core"
            or name.startswith("core.")
        )
        and isinstance(module, ModuleType)
    }
    for name, module in sorted(namespace_modules.items()):
        if getattr(module, "__file__", None) is None:
            locations = [Path(item).resolve() for item in getattr(module, "__path__", ())]
            if not locations or any(
                not _path_is_within(location, god_root) for location in locations
            ):
                raise RuntimeError(
                    f"namespace module resolved outside supplied G.O.D root: {name}"
                )
            continue
        source = _module_source(module)
        if not _path_is_within(source, god_root):
            raise RuntimeError(
                f"{module.__name__} imported outside supplied G.O.D root: {source}"
            )
        bindings[name] = {
            "module": module.__name__,
            "path": str(source.relative_to(god_root)),
            "sha256": _sha256(source),
        }
    return modules, bindings


def _assert_bound_sources_unchanged(
    god_root: Path,
    bindings: dict[str, dict[str, str]],
) -> None:
    for details in bindings.values():
        path = god_root / details["path"]
        if _sha256(path) != details["sha256"]:
            raise RuntimeError(f"imported G.O.D source changed during evaluation: {path}")


def _python_environment(python_path: Path) -> dict[str, Any]:
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise ValueError(f"Comfy Python is not an executable file: {python_path}")
    venv_root = python_path.parent.parent
    venv_marker = venv_root / "pyvenv.cfg"
    conda_marker = venv_root / "conda-meta" / "history"
    if venv_marker.is_file():
        environment_kind = "venv"
        identity_marker = venv_marker
    elif conda_marker.is_file():
        environment_kind = "conda"
        identity_marker = conda_marker
    else:
        raise ValueError(
            "Comfy Python must be inside an identifiable venv or conda "
            f"environment: missing {venv_marker} and {conda_marker}"
        )
    probe = (
        "import hashlib,importlib.metadata as m,json,platform,sys;"
        "rows=sorted((d.metadata.get('Name') or '',d.version) "
        "for d in m.distributions());"
        "encoded=json.dumps(rows,separators=(',',':')).encode();"
        "print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,"
        "'base_prefix':sys.base_prefix,'python':platform.python_version(),"
        "'distribution_count':len(rows),"
        "'distributions_sha256':hashlib.sha256(encoded).hexdigest()}))"
    )
    try:
        info = json.loads(_run_text([str(python_path), "-I", "-c", probe]))
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not inspect Comfy Python: {python_path}") from exc
    if Path(info["prefix"]).resolve() != venv_root.resolve():
        raise RuntimeError(
            f"Comfy Python did not activate supplied venv: {info['prefix']} != {venv_root}"
        )
    info.update(
        {
            "requested_executable": str(python_path),
            "venv_root": str(venv_root),
            "environment_kind": environment_kind,
            "identity_marker": str(identity_marker),
            "identity_marker_sha256": _sha256(identity_marker),
        }
    )
    return info


def _driver_environment() -> dict[str, Any]:
    rows = sorted(
        (distribution.metadata.get("Name") or "", distribution.version)
        for distribution in importlib.metadata.distributions()
    )
    return {
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "python": ".".join(str(item) for item in sys.version_info[:3]),
        "distribution_count": len(rows),
        "distributions_sha256": _json_sha256(rows),
    }


def _capture_dataset(
    dataset: Path,
    *,
    list_supported_images: Any,
    extensions: tuple[str, ...],
) -> dict[str, Any]:
    if not dataset.is_dir():
        raise ValueError(f"dataset path is not a directory: {dataset}")
    entries = list(dataset.iterdir())
    non_files = [entry.name for entry in entries if not entry.is_file()]
    symlinks = [entry.name for entry in entries if entry.is_symlink()]
    if non_files or symlinks:
        raise RuntimeError(
            {"dataset_non_files": sorted(non_files), "dataset_symlinks": sorted(symlinks)}
        )

    evaluator_order = list_supported_images(str(dataset), extensions)
    if (
        not isinstance(evaluator_order, list)
        or not evaluator_order
        or any(not isinstance(name, str) for name in evaluator_order)
        or len(evaluator_order) != len(set(evaluator_order))
    ):
        raise RuntimeError("G.O.D returned an invalid or empty image list")

    image_names = set(evaluator_order)
    stems: set[str] = set()
    prompt_names: set[str] = set()
    rows: list[dict[str, Any]] = []
    for index, image_name in enumerate(evaluator_order):
        image_path = dataset / image_name
        stem = os.path.splitext(image_name)[0]
        if stem in stems:
            raise RuntimeError(f"ambiguous duplicate image stem in dataset: {stem}")
        stems.add(stem)
        prompt_name = f"{stem}.txt"
        prompt_path = dataset / prompt_name
        if not prompt_path.is_file() or prompt_path.is_symlink():
            raise RuntimeError(f"missing or unsafe paired prompt: {prompt_path}")
        try:
            prompt = prompt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"prompt is not valid UTF-8: {prompt_path}") from exc
        if not prompt.strip():
            raise RuntimeError(f"paired prompt is empty: {prompt_path}")
        prompt_names.add(prompt_name)

        # PIL is already a dependency of the imported evaluator.  Verifying the
        # bytes here prevents a corrupt row from becoming an unnamed eval error.
        pil_image = importlib.import_module("PIL.Image")
        try:
            with pil_image.open(image_path) as image:
                image.verify()
            with pil_image.open(image_path) as image:
                width, height = image.size
                image_format = image.format
                mode = image.mode
        except Exception as exc:
            raise RuntimeError(f"invalid image row: {image_path}") from exc
        if width <= 0 or height <= 0:
            raise RuntimeError(f"image has invalid dimensions: {image_path}")
        rows.append(
            {
                "index": index,
                "image": image_name,
                "image_sha256": _sha256(image_path),
                "image_bytes": image_path.stat().st_size,
                "image_width": width,
                "image_height": height,
                "image_format": image_format,
                "image_mode": mode,
                "prompt": prompt_name,
                "prompt_sha256": _sha256(prompt_path),
                "prompt_bytes": prompt_path.stat().st_size,
            }
        )

    regular_names = {entry.name for entry in entries}
    expected_names = image_names | prompt_names
    unexpected = regular_names - expected_names
    missing_expected = expected_names - regular_names
    if unexpected or missing_expected:
        raise RuntimeError(
            {
                "unexpected_dataset_files": sorted(unexpected),
                "missing_dataset_files": sorted(missing_expected),
            }
        )
    identity = {
        "evaluator_order": evaluator_order,
        "rows": rows,
    }
    identity["sha256"] = _json_sha256(identity)
    return identity


def _file_snapshot(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError({"missing_files": missing})
    return {
        name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for name, path in paths.items()
    }


def _assert_files_unchanged(
    paths: dict[str, Path],
    before: dict[str, dict[str, Any]],
) -> None:
    if _file_snapshot(paths) != before:
        raise RuntimeError("candidate, staged LoRA, workflow, shim, or asset changed")


def _assert_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((_LOOPBACK, port))
        except OSError as exc:
            raise RuntimeError(
                f"refusing to reuse occupied ComfyUI port {_LOOPBACK}:{port}"
            ) from exc


def _http_json(port: int, path: str, *, timeout: float) -> Any:
    request = urllib.request.Request(
        f"http://{_LOOPBACK}:{port}{path}",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"ComfyUI {path} returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def _wait_for_fresh_comfy(
    process: subprocess.Popen[bytes],
    *,
    port: int,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"fresh ComfyUI exited during startup ({returncode})")
        try:
            stats = _http_json(port, "/system_stats", timeout=2.0)
            history = _http_json(port, "/history", timeout=2.0)
            if not isinstance(stats, dict):
                raise RuntimeError("ComfyUI returned invalid system stats")
            if history != {}:
                raise RuntimeError("fresh ComfyUI process has non-empty history")
            return stats
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.5)
    raise TimeoutError(
        f"fresh ComfyUI was not ready within {timeout_s}s: {last_error}"
    )


class _EvaluationTimedOut(TimeoutError):
    pass


@contextmanager
def _time_limit(seconds: float) -> Iterator[None]:
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        raise RuntimeError("hard evaluation timeout requires POSIX SIGALRM")

    def _expired(_signum: int, _frame: Any) -> None:
        raise _EvaluationTimedOut(f"evaluation exceeded {seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _stop_process(
    process: subprocess.Popen[bytes],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    if process.poll() is not None:
        return {"returncode": process.returncode, "stop_signal": None, "forced": False}
    phases = (
        (signal.SIGINT, timeout_s),
        (signal.SIGTERM, min(timeout_s, 5.0)),
    )
    for stop_signal, wait_s in phases:
        try:
            os.killpg(process.pid, stop_signal)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=wait_s)
            return {
                "returncode": process.returncode,
                "stop_signal": signal.Signals(stop_signal).name,
                "forced": stop_signal != signal.SIGINT,
            }
        except subprocess.TimeoutExpired:
            continue
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5.0)
    return {
        "returncode": process.returncode,
        "stop_signal": signal.Signals(signal.SIGKILL).name,
        "forced": True,
    }


def _close_gateway(gateway: ModuleType) -> None:
    websocket = getattr(gateway, "ws", None)
    if websocket is not None:
        try:
            websocket.close()
        except Exception:
            pass


def _run_exact_eval(
    diffusion: ModuleType,
    *,
    dataset: Path,
    params: Any,
    generations: int,
) -> tuple[dict[str, Any], list[str]]:
    captured_orders: list[list[str]] = []
    original = diffusion.list_supported_images

    def _capture_order(dataset_path: str, extensions: tuple[str, ...]) -> list[str]:
        names = original(dataset_path, extensions)
        captured_orders.append(list(names))
        return names

    diffusion.list_supported_images = _capture_order
    try:
        raw = diffusion.eval_loop(
            str(dataset),
            params,
            generations=generations,
        )
    finally:
        diffusion.list_supported_images = original
    if len(captured_orders) != 1:
        raise RuntimeError(
            f"expected one evaluator dataset listing, observed {len(captured_orders)}"
        )
    return raw, captured_orders[0]


def _validate_history(history: Any, *, expected_prompts: int) -> dict[str, Any]:
    if not isinstance(history, dict) or len(history) != expected_prompts:
        raise RuntimeError(
            f"ComfyUI history count mismatch: expected {expected_prompts}, "
            f"got {len(history) if isinstance(history, dict) else 'invalid'}"
        )
    for prompt_id, item in history.items():
        if not isinstance(item, dict) or not isinstance(item.get("outputs"), dict):
            raise RuntimeError(f"invalid ComfyUI history entry: {prompt_id}")
        if not item["outputs"]:
            raise RuntimeError(f"ComfyUI history entry has no output: {prompt_id}")
        status = item.get("status")
        if isinstance(status, dict):
            if status.get("status_str") not in (None, "success"):
                raise RuntimeError(f"ComfyUI prompt was not successful: {prompt_id}")
            if status.get("completed") is False:
                raise RuntimeError(f"ComfyUI prompt was incomplete: {prompt_id}")
    return {
        "prompt_count": len(history),
        "history_sha256": _json_sha256(history),
    }


def _publish_exclusive(output: Path, temp: Path, result: dict[str, Any]) -> None:
    if output.exists() or temp.exists():
        raise FileExistsError(f"refusing stale evidence path: {output} or {temp}")
    with temp.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        # Unlike os.replace(), link() cannot overwrite evidence created by a
        # racing or earlier run.
        os.link(temp, output)
        temp.unlink()
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        # Keep the temp evidence for diagnosis; future runs will fail closed.
        raise


def main() -> int:
    args = _parse()
    started = time.monotonic()
    original_cwd = Path.cwd()

    dataset = Path(args.dataset).expanduser().resolve(strict=True)
    candidate = _absolute_lexical(args.candidate_path)
    output = _absolute_lexical(args.output)
    temp_output = Path(f"{output}.tmp")
    comfy_log = _absolute_lexical(args.comfy_log) if args.comfy_log else Path(
        f"{output}.comfy.log"
    )
    comfy_root = Path(args.comfy_root).expanduser().resolve(strict=True)
    god_root = Path(args.god_root).expanduser().resolve(strict=True)
    comfy_python = _absolute_lexical(args.comfy_python)
    tooling_root = comfy_root / "custom_nodes" / _TOOLING_NODE

    output.parent.mkdir(parents=True, exist_ok=True)
    comfy_log.parent.mkdir(parents=True, exist_ok=True)
    evidence_paths = {output, temp_output, comfy_log}
    if len(evidence_paths) != 3:
        raise ValueError("output, temp output, and Comfy log paths must differ")
    stale = [str(path) for path in evidence_paths if path.exists()]
    if stale:
        raise FileExistsError({"stale_evidence_paths": stale})

    if not candidate.is_file():
        raise ValueError("candidate path is invalid")
    base_name = os.path.basename(args.base_name)
    if base_name != args.base_name or base_name in {"", ".", ".."}:
        raise ValueError("base-name must be one ComfyUI filename")
    comfy_main = comfy_root / "main.py"
    if not comfy_main.is_file():
        raise FileNotFoundError(f"ComfyUI entrypoint is missing: {comfy_main}")
    extra_model_paths = comfy_root / "extra_model_paths.yaml"
    if extra_model_paths.exists():
        raise RuntimeError(
            "refusing ComfyUI extra_model_paths.yaml; model resolution must stay "
            "inside the supplied Comfy root"
        )

    expected_commits = {
        "god": args.expected_god_commit,
        "comfyui": args.expected_comfy_commit,
        "tooling_nodes": args.expected_tooling_commit,
    }
    git_before = {
        "god": _git_snapshot(god_root, expected_commit=args.expected_god_commit),
        "comfyui": _git_snapshot(
            comfy_root,
            expected_commit=args.expected_comfy_commit,
        ),
        "tooling_nodes": _git_snapshot(
            tooling_root,
            expected_commit=args.expected_tooling_commit,
        ),
    }
    python_environment = _python_environment(comfy_python)
    driver_environment = _driver_environment()

    os.chdir(god_root)
    try:
        modules, import_bindings = _import_god(god_root)
        constants = modules["constants"]
        image_models = modules["image_models"]
        models = modules["models"]
        diffusion = modules["diffusion"]
        gateway = modules["comfy_gateway"]
        image_io = modules["image_io"]
        dataset_constants = modules["dataset_constants"]

        extensions = tuple(dataset_constants.SUPPORTED_IMAGE_FILE_EXTENSIONS)
        dataset_before = _capture_dataset(
            dataset,
            list_supported_images=image_io.list_supported_images,
            extensions=extensions,
        )
        model_type = image_models.ImageModelType.KREA2.value
        if model_type not in constants.EVAL_DEFAULTS:
            raise RuntimeError("supplied G.O.D checkout has no Krea2 eval defaults")
        defaults = dict(constants.EVAL_DEFAULTS[model_type])
        steps = defaults.get("steps")
        generations = defaults.get("generations")
        cfg = defaults.get("cfg")
        denoise = defaults.get("denoise")
        if (
            not isinstance(steps, int)
            or isinstance(steps, bool)
            or steps <= 0
            or not isinstance(generations, int)
            or isinstance(generations, bool)
            or generations <= 0
            or isinstance(cfg, bool)
            or isinstance(denoise, bool)
            or not math.isfinite(float(cfg))
            or float(cfg) < 0.0
            or not math.isfinite(float(denoise))
            or not 0.0 <= float(denoise) <= 1.0
        ):
            raise RuntimeError(f"invalid Krea2 eval defaults: {defaults}")

        candidate_sha256 = _sha256(candidate)
        candidate_name = f"candidate-{candidate_sha256}.safetensors"
        staged_candidate = comfy_root / "models" / "loras" / candidate_name
        if not staged_candidate.is_file():
            raise FileNotFoundError(f"staged Comfy LoRA is missing: {staged_candidate}")
        if _sha256(staged_candidate) != candidate_sha256:
            raise RuntimeError(
                "candidate bytes do not match the LoRA that ComfyUI would score"
            )

        workflow_path = god_root / constants.LORA_KREA2_WORKFLOW_PATH
        if not _path_is_within(workflow_path.resolve(), god_root):
            raise RuntimeError("G.O.D workflow escaped supplied checkout")
        immutable_paths = {
            "candidate": candidate,
            "staged_candidate": staged_candidate,
            "diffusion_model": (
                comfy_root / "models" / "diffusion_models" / base_name
            ),
            "text_encoder": (
                comfy_root
                / "models"
                / "text_encoders"
                / "qwen3vl_4b_fp8_scaled.safetensors"
            ),
            "vae": comfy_root / "models" / "vae" / "qwen_image_vae.safetensors",
            "workflow": workflow_path,
            "calibration_shim": Path(__file__).resolve(),
            "comfy_main": comfy_main,
            "comfy_environment_marker": Path(
                python_environment["identity_marker"]
            ),
        }
        immutable_before = _file_snapshot(immutable_paths)

        _assert_port_free(args.port)
        process: subprocess.Popen[bytes] | None = None
        shutdown: dict[str, Any] | None = None
        raw: dict[str, Any]
        actual_order: list[str]
        system_stats: dict[str, Any]
        history_evidence: dict[str, Any]
        with tempfile.TemporaryDirectory(prefix="forge-krea-comfy-") as isolation:
            isolation_root = Path(isolation)
            input_dir = isolation_root / "input"
            output_dir = isolation_root / "output"
            temp_dir = isolation_root / "temp-root"
            user_dir = isolation_root / "user"
            for directory in (input_dir, output_dir, temp_dir, user_dir):
                directory.mkdir()

            command = [
                str(comfy_python),
                str(comfy_main),
                "--listen",
                _LOOPBACK,
                "--port",
                str(args.port),
                "--disable-auto-launch",
                "--base-directory",
                str(comfy_root),
                "--input-directory",
                str(input_dir),
                "--output-directory",
                str(output_dir),
                "--temp-directory",
                str(temp_dir),
                "--user-directory",
                str(user_dir),
                "--disable-all-custom-nodes",
                "--whitelist-custom-nodes",
                _TOOLING_NODE,
                "--disable-api-nodes",
                "--database-url",
                "sqlite:///:memory:",
            ]
            child_environment = dict(os.environ)
            child_environment.update(
                {
                    "VIRTUAL_ENV": str(comfy_python.parent.parent),
                    "PATH": f"{comfy_python.parent}{os.pathsep}{child_environment.get('PATH', '')}",
                    "PYTHONNOUSERSITE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "DIFFUSERS_OFFLINE": "1",
                    "HF_HUB_DISABLE_TELEMETRY": "1",
                }
            )
            child_environment.pop("PYTHONPATH", None)

            with comfy_log.open("xb") as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd=str(comfy_root),
                    env=child_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    system_stats = _wait_for_fresh_comfy(
                        process,
                        port=args.port,
                        timeout_s=args.startup_timeout_s,
                    )
                    workflow, _ = diffusion.load_comfy_workflows(model_type)
                    gateway.server_address = f"{_LOOPBACK}:{args.port}"
                    params = models.Img2ImgPayload(
                        ckpt_name=base_name,
                        lora_name=candidate_name,
                        steps=steps,
                        cfg=cfg,
                        denoise=denoise,
                        comfy_template=copy.deepcopy(workflow),
                        is_safetensors=True,
                        model_type=model_type,
                    )
                    with _time_limit(args.evaluation_timeout_s):
                        gateway.connect()
                        if getattr(gateway, "ws", None) is None:
                            raise RuntimeError("G.O.D Comfy gateway did not create a websocket")
                        gateway.ws.settimeout(args.evaluation_timeout_s)
                        raw, actual_order = _run_exact_eval(
                            diffusion,
                            dataset=dataset,
                            params=params,
                            generations=generations,
                        )
                    if actual_order != dataset_before["evaluator_order"]:
                        raise RuntimeError(
                            "G.O.D image order changed between identity capture and scoring"
                        )
                    expected_prompts = (
                        len(dataset_before["rows"]) * generations * 2
                    )
                    history = _http_json(args.port, "/history", timeout=10.0)
                    history_evidence = _validate_history(
                        history,
                        expected_prompts=expected_prompts,
                    )
                    queue = _http_json(args.port, "/queue", timeout=10.0)
                    if not isinstance(queue, dict):
                        raise RuntimeError("ComfyUI returned an invalid queue")
                    if queue.get("queue_running") or queue.get("queue_pending"):
                        raise RuntimeError("ComfyUI queue was not empty after evaluation")
                    if process.poll() is not None:
                        raise RuntimeError(
                            "fresh ComfyUI exited before controlled shutdown "
                            f"({process.returncode})"
                        )
                finally:
                    _close_gateway(gateway)
                    shutdown = _stop_process(
                        process,
                        timeout_s=args.shutdown_timeout_s,
                    )
                log_handle.flush()
                os.fsync(log_handle.fileno())
            if shutdown is None:
                raise RuntimeError("fresh ComfyUI shutdown was not observed")

        if not isinstance(raw, dict):
            raise RuntimeError("G.O.D evaluator returned a non-object result")
        normalized: dict[str, list[float]] = {}
        row_count = len(dataset_before["rows"])
        for key in ("text_guided_losses", "no_text_losses"):
            values = raw.get(key)
            if not isinstance(values, list) or len(values) != row_count:
                raise RuntimeError(f"evaluator returned invalid {key} length")
            converted: list[float] = []
            for value in values:
                if isinstance(value, bool):
                    raise RuntimeError(f"evaluator returned boolean in {key}")
                number = float(value)
                # RGB pixels are normalized into [0, 1] before G.O.D computes
                # MSE, so a valid per-row loss is also bounded by [0, 1].
                if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                    raise RuntimeError(f"evaluator returned invalid {key} value")
                converted.append(number)
            normalized[key] = converted

        dataset_after = _capture_dataset(
            dataset,
            list_supported_images=image_io.list_supported_images,
            extensions=extensions,
        )
        if dataset_after != dataset_before:
            raise RuntimeError("dataset identity changed during evaluation")
        _assert_files_unchanged(immutable_paths, immutable_before)
        _assert_bound_sources_unchanged(god_root, import_bindings)
        _assert_git_unchanged(
            god_root,
            git_before["god"],
            expected_commit=args.expected_god_commit,
        )
        _assert_git_unchanged(
            comfy_root,
            git_before["comfyui"],
            expected_commit=args.expected_comfy_commit,
        )
        _assert_git_unchanged(
            tooling_root,
            git_before["tooling_nodes"],
            expected_commit=args.expected_tooling_commit,
        )
        if _python_environment(comfy_python) != python_environment:
            raise RuntimeError("Comfy virtual-environment identity changed during evaluation")
        if _driver_environment() != driver_environment:
            raise RuntimeError("evaluator driver environment changed during evaluation")

        text_mean = sum(normalized["text_guided_losses"]) / row_count
        blank_mean = sum(normalized["no_text_losses"]) / row_count
        text_weight = float(constants.DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT)
        if not math.isfinite(text_weight) or not 0.0 <= text_weight <= 1.0:
            raise RuntimeError(f"invalid G.O.D text-guided weight: {text_weight}")
        weighted = text_weight * text_mean + (1.0 - text_weight) * blank_mean
        if not all(math.isfinite(value) and value >= 0.0 for value in (
            text_mean,
            blank_mean,
            weighted,
        )):
            raise RuntimeError("aggregate evaluator loss is invalid")
        if any(value > 1.0 for value in (text_mean, blank_mean, weighted)):
            raise RuntimeError("aggregate evaluator loss escaped RGB-MSE bounds")

        seeds = diffusion.generate_reproducible_seeds(
            master_seed=42,
            n=generations,
        )
        if (
            not isinstance(seeds, list)
            or len(seeds) != generations
            or any(
                not isinstance(seed, int)
                or isinstance(seed, bool)
                or not 0 <= seed <= 2**32 - 1
                for seed in seeds
            )
        ):
            raise RuntimeError("G.O.D returned an invalid seed schedule")
        scored_rows = []
        for row, text_loss, blank_loss in zip(
            dataset_before["rows"],
            normalized["text_guided_losses"],
            normalized["no_text_losses"],
            strict=True,
        ):
            scored_rows.append(
                {
                    **row,
                    "text_guided_loss": text_loss,
                    "blank_prompt_loss": blank_loss,
                }
            )

        log_sha256 = _sha256(comfy_log)
        result = {
            "schema": 2,
            "evaluator": "god_krea2_img2img_exact",
            "candidate": candidate.name,
            "candidate_sha256": candidate_sha256,
            "candidate_bytes": immutable_before["candidate"]["bytes"],
            "staged_candidate_sha256": immutable_before["staged_candidate"]["sha256"],
            "comfy_lora_name": candidate_name,
            "model_type": model_type,
            "dataset": str(dataset),
            "dataset_sha256": dataset_before["sha256"],
            "image_count": row_count,
            "scored_rows": scored_rows,
            "base_name": base_name,
            "asset_sha256": {
                name: immutable_before[name]["sha256"]
                for name in ("diffusion_model", "text_encoder", "vae")
            },
            "asset_bytes": {
                name: immutable_before[name]["bytes"]
                for name in ("diffusion_model", "text_encoder", "vae")
            },
            "steps": steps,
            "cfg": cfg,
            "denoise": denoise,
            "generations": generations,
            "seeds": seeds,
            "text_guided_losses": normalized["text_guided_losses"],
            "blank_prompt_losses": normalized["no_text_losses"],
            "text_mean": text_mean,
            "blank_mean": blank_mean,
            "text_weight": text_weight,
            "weighted_loss": weighted,
            "direction": "min",
            "elapsed_s": round(time.monotonic() - started, 3),
            "source": {
                "god": {
                    **git_before["god"],
                    "tracked_worktree_clean": True,
                    "nonignored_worktree_clean": True,
                },
                "comfyui": {
                    **git_before["comfyui"],
                    "tracked_worktree_clean": True,
                    "nonignored_worktree_clean": True,
                },
                "tooling_nodes": {
                    **git_before["tooling_nodes"],
                    "tracked_worktree_clean": True,
                    "nonignored_worktree_clean": True,
                },
                "expected_commits": expected_commits,
                "god_import_bindings": import_bindings,
                "workflow_path": str(workflow_path.relative_to(god_root)),
                "workflow_sha256": immutable_before["workflow"]["sha256"],
                "calibration_shim_sha256": immutable_before["calibration_shim"]["sha256"],
                "comfy_main_sha256": immutable_before["comfy_main"]["sha256"],
            },
            "runtime": {
                "fresh_comfy_process": True,
                "loopback": _LOOPBACK,
                "port": args.port,
                "cache": "comfy_default_fresh_process",
                "database": "memory",
                "api_nodes_disabled": True,
                "isolated_input_output_temp_user": True,
                "offline_environment": True,
                "custom_node_allowlist": [_TOOLING_NODE],
                "startup_timeout_s": args.startup_timeout_s,
                "evaluation_timeout_s": args.evaluation_timeout_s,
                "shutdown_timeout_s": args.shutdown_timeout_s,
                "shutdown": shutdown,
                "python": python_environment,
                "driver_python": driver_environment,
                "comfy_system_stats": system_stats,
                "comfy_history": history_evidence,
                "comfy_log": str(comfy_log),
                "comfy_log_sha256": log_sha256,
                "comfy_log_bytes": comfy_log.stat().st_size,
            },
        }
        _publish_exclusive(output, temp_output, result)
        print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
        return 0
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
