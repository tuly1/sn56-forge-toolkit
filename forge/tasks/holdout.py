"""Fail-closed producer for image checkpoint held-out proxy scores.

The validator's actual image score is a ComfyUI img2img reconstruction metric.
Those evaluator-only FP8 models and support assets are not staged inside the
air-gapped trainer.  This module therefore makes a narrower, honest promise: it
runs the pinned ai-toolkit training objective at zero learning rate on examples
that were removed before training. Captioned and blank-caption strata are run
separately, then combined at the validator's 1:3 weight; this prevents the
pinned logger's omitted first step from changing the claimed weighting. The
result is a deterministic checkpoint-ranking *proxy*, not a validator-score
replica.

The existing consumer in :mod:`forge.tasks.checkpoints` accepts a manifest only
when every valid current-run candidate is represented by an exact SHA-256.  The
producer below preserves that contract: any timeout, missing point, worker
failure, candidate-set change, or hash drift leaves no authoritative manifest.
Finalization then falls through to the exact-final/divergence policy.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import yaml

from forge import recipe, telemetry
from forge.clock import Deadline
from forge.data.schema import ImageSpec
from forge.tasks import checkpoints
from forge.tasks.integrity import valid_safetensors

_AI_TOOLKIT_DIR = os.environ.get("AI_TOOLKIT_DIR", "/app/ai-toolkit")
_MANIFEST_NAME = "forge_holdout_scores.json"
_METRIC = "heldout_diffusion_loss_proxy_v2"
_SEED = 42565431
_PROBE_EPOCHS = 2
_CAPTIONED_WEIGHT = 0.25
_BLANK_WEIGHT = 0.75
_POLL_SECONDS = 2.0
_FINALIZE_MARGIN_S = 30.0
_MIN_CANDIDATE_START_S = 120.0
_IMPLEMENTED_TYPES = frozenset({"krea2", "ideogram4"})
_SCORING_RESERVE_S = {"krea2": 900.0, "ideogram4": 750.0}
_MIN_TRAINING_WINDOW_S = {"krea2": 600.0, "ideogram4": 600.0}
_BOUNDARY_MARGIN_S = 45.0


def enabled_for(model_type: str) -> bool:
    """Whether the still-experimental proxy is enabled for this architecture.

    Activation is explicit because each architecture needs an external
    ComfyUI rank-correlation gate.  A merged producer with no allowlist remains
    dormant and cannot silently change a tournament export.
    """
    model_type = (model_type or "").strip().lower()
    if model_type not in _IMPLEMENTED_TYPES:
        return False
    raw = os.environ.get("FORGE_HOLDOUT_SELECTION_TYPES", "")
    allowed = {value.strip().lower() for value in raw.split(",") if value.strip()}
    return "*" in allowed or model_type in allowed


def scoring_reserve_s(model_type: str) -> float:
    """Initial conservative reserve; replace with measured target-runtime p95."""
    if not enabled_for(model_type):
        return 0.0
    return float(_SCORING_RESERVE_S.get((model_type or "").strip().lower(), 0.0))


def boundary_margin_s() -> float:
    return _BOUNDARY_MARGIN_S


def budget_allows(model_type: str, remaining_soft_s: float) -> bool:
    """Whether splitting data can leave both useful training and full scoring."""
    if not enabled_for(model_type):
        return False
    model_type = (model_type or "").strip().lower()
    try:
        reserve = float(_SCORING_RESERVE_S.get(model_type, 0.0))
        minimum_training = float(_MIN_TRAINING_WINDOW_S.get(model_type, 0.0))
        hard_equivalent = float(remaining_soft_s) + recipe.EXPORT_RESERVE_S
        planned_training = (
            hard_equivalent * recipe.MARGIN
            - reserve
            - _BOUNDARY_MARGIN_S
            - recipe.STARTUP_S
            - recipe.EXPORT_RESERVE_S
        )
        return reserve > 0.0 and planned_training >= minimum_training
    except (TypeError, ValueError):
        return False


def has_scoring_candidates(save_root: str, scope: dict[str, Any]) -> bool:
    """Whether reserving scorer time could currently produce a manifest."""
    try:
        return len(_valid_candidates(save_root, scope)) >= 2
    except Exception:
        return False


def produce(
    spec: ImageSpec,
    cfg: dict[str, Any],
    scope: dict[str, Any],
    deadline: Deadline,
    *,
    holdout_pairs: int,
    scorer: Callable[..., dict[str, Any]] | None = None,
) -> bool:
    """Score every valid current-run candidate and atomically publish a manifest.

    Returns ``True`` only when a complete authoritative manifest was written.
    It never raises into finalization.
    """
    manifest_path = os.path.join(spec.save_root, _MANIFEST_NAME)
    started = time.monotonic()
    temp_root = None
    try:
        _remove_manifest(manifest_path)
        if not enabled_for(spec.model_type):
            telemetry.event(
                "holdout_scoring_skipped",
                reason="model_type_not_allowlisted",
                model_type=spec.model_type,
            )
            return False
        if holdout_pairs <= 0 or not os.path.isdir(spec.dataset_holdout_dir):
            telemetry.event("holdout_scoring_skipped", reason="no_true_holdout")
            return False

        before = _valid_candidates(spec.save_root, scope)
        if len(before) < 2:
            telemetry.event(
                "holdout_scoring_skipped",
                reason="fewer_than_two_valid_candidates",
                candidates=len(before),
            )
            return False
        before_hashes = {os.path.basename(path): _sha256(path) for path in before}

        temp_root = tempfile.mkdtemp(prefix="forge-holdout-proxy-")
        probe_root = os.path.join(temp_root, "datasets")
        captioned_dir, blank_dir, probe_pairs = _build_probe_datasets(
            spec.dataset_holdout_dir,
            probe_root,
        )
        if probe_pairs != holdout_pairs:
            raise RuntimeError(
                f"probe datasets have {probe_pairs} pairs per stratum; expected "
                f"{holdout_pairs}"
            )
        expected_stratum_points = probe_pairs * _PROBE_EPOCHS
        expected_points = expected_stratum_points * 2

        score_one = scorer or _score_candidate
        rows: list[dict[str, Any]] = []
        for index, path in enumerate(before):
            candidates_left = len(before) - index
            required_window = (
                _FINALIZE_MARGIN_S
                + candidates_left * _MIN_CANDIDATE_START_S
            )
            if deadline.remaining() <= required_window:
                raise TimeoutError(
                    "insufficient soft-deadline budget for all remaining probes: "
                    f"need>{required_window:.1f}s for {candidates_left} candidates"
                )
            result = score_one(
                path=path,
                cfg=cfg,
                captioned_dir=captioned_dir,
                blank_dir=blank_dir,
                temp_root=temp_root,
                index=index,
                expected_points=expected_points,
                expected_stratum_points=expected_stratum_points,
                deadline=deadline,
            )
            score = float(result["score"])
            points = int(result["points"])
            captioned_points = int(result["captioned_points"])
            blank_points = int(result["blank_points"])
            if (
                not math.isfinite(score)
                or points != expected_points
                or captioned_points != expected_stratum_points
                or blank_points != expected_stratum_points
            ):
                raise RuntimeError(
                    f"invalid proxy result for {os.path.basename(path)!r}: "
                    f"score={score!r}, points={points}, expected={expected_points}, "
                    f"captioned={captioned_points}, blank={blank_points}"
                )
            row = {
                "checkpoint": os.path.basename(path),
                "sha256": before_hashes[os.path.basename(path)],
                "step": _step_of(path, scope),
                "score": score,
                "points": points,
                "captioned_score": float(result["captioned_score"]),
                "blank_caption_score": float(result["blank_caption_score"]),
                "captioned_points": captioned_points,
                "blank_caption_points": blank_points,
            }
            for source, target in (
                ("captioned_stddev", "captioned_stddev"),
                ("blank_stddev", "blank_caption_stddev"),
            ):
                spread = result.get(source)
                if spread is not None and math.isfinite(float(spread)):
                    row[target] = float(spread)
            rows.append(row)
            telemetry.event(
                "holdout_candidate_scored",
                checkpoint=row["checkpoint"],
                score=score,
                points=points,
            )

        after = _valid_candidates(spec.save_root, scope)
        after_hashes = {os.path.basename(path): _sha256(path) for path in after}
        if list(before_hashes) != list(after_hashes) or before_hashes != after_hashes:
            raise RuntimeError("candidate set or bytes changed while scoring")

        manifest = {
            "schema": 1,
            "source": "heldout",
            "complete": True,
            # Bind the evidence to the checkpoint journal that authorized this
            # exact candidate set.  Consumers still validate every candidate
            # hash, while calibration tooling can additionally prove that a
            # complete manifest was produced by this attempt rather than left
            # behind by a same-task retry.
            "task_id": spec.task_id,
            "expected_repo_name": spec.expected_repo_name,
            "attempt_nonce": scope.get("attempt_nonce"),
            "scope_started_unix": scope.get("started_unix"),
            "direction": "min",
            "metric": _METRIC,
            "proxy_not_validator_metric": True,
            "model_type": spec.model_type,
            "seed": _SEED,
            "holdout_pairs": holdout_pairs,
            "probe_epochs": _PROBE_EPOCHS,
            "captioned_weight": _CAPTIONED_WEIGHT,
            "blank_caption_weight": _BLANK_WEIGHT,
            "strata_scored_separately": True,
            "scores": rows,
            "elapsed_s": round(time.monotonic() - started, 3),
            "created_unix": int(time.time()),
        }
        _atomic_json(manifest_path, manifest)
        telemetry.event(
            "holdout_manifest_complete",
            metric=_METRIC,
            candidates=len(rows),
            holdout_pairs=holdout_pairs,
            elapsed_s=manifest["elapsed_s"],
        )
        return True
    except BaseException as exc:
        try:
            _remove_manifest(manifest_path)
        except BaseException as cleanup_exc:
            telemetry.event(
                "holdout_manifest_cleanup_failed",
                error=f"{type(cleanup_exc).__name__}: {cleanup_exc}",
            )
        telemetry.event(
            "holdout_scoring_failed", error=f"{type(exc).__name__}: {exc}"
        )
        return False
    finally:
        if temp_root:
            shutil.rmtree(temp_root, ignore_errors=True)


def _score_candidate(
    *,
    path: str,
    cfg: dict[str, Any],
    captioned_dir: str,
    blank_dir: str,
    temp_root: str,
    index: int,
    expected_points: int,
    expected_stratum_points: int,
    deadline: Deadline,
) -> dict[str, Any]:
    candidate_root = os.path.join(temp_root, f"candidate-{index:02d}")
    os.makedirs(candidate_root, exist_ok=True)
    captioned_values = _score_stratum(
        path=path,
        cfg=cfg,
        probe_dir=captioned_dir,
        candidate_root=os.path.join(candidate_root, "captioned"),
        probe_name="probe-captioned",
        expected_points=expected_stratum_points,
        deadline=deadline,
    )
    blank_values = _score_stratum(
        path=path,
        cfg=cfg,
        probe_dir=blank_dir,
        candidate_root=os.path.join(candidate_root, "blank"),
        probe_name="probe-blank",
        expected_points=expected_stratum_points,
        deadline=deadline,
    )
    captioned_score = statistics.fmean(captioned_values)
    blank_score = statistics.fmean(blank_values)
    return {
        "score": (
            _CAPTIONED_WEIGHT * captioned_score
            + _BLANK_WEIGHT * blank_score
        ),
        "captioned_score": captioned_score,
        "blank_caption_score": blank_score,
        "captioned_stddev": (
            statistics.pstdev(captioned_values)
            if len(captioned_values) > 1
            else 0.0
        ),
        "blank_stddev": (
            statistics.pstdev(blank_values) if len(blank_values) > 1 else 0.0
        ),
        "captioned_points": len(captioned_values),
        "blank_points": len(blank_values),
        "points": len(captioned_values) + len(blank_values),
    }


def _score_stratum(
    *,
    path: str,
    cfg: dict[str, Any],
    probe_dir: str,
    candidate_root: str,
    probe_name: str,
    expected_points: int,
    deadline: Deadline,
) -> list[float]:
    os.makedirs(candidate_root, exist_ok=True)
    probe_cfg = _probe_config(
        cfg,
        candidate=path,
        probe_dir=probe_dir,
        candidate_root=candidate_root,
        expected_points=expected_points,
        probe_name=probe_name,
    )
    config_path = os.path.join(candidate_root, "probe.yaml")
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(probe_cfg, fh, sort_keys=False)

    log_path = os.path.join(candidate_root, "probe.log")
    env = os.environ.copy()
    env.update(
        {
            "SEED": str(_SEED),
            "PYTHONHASHSEED": str(_SEED),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_DISABLED": "true",
        }
    )
    cmd = [sys.executable, "run.py", config_path]
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=_AI_TOOLKIT_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while proc.poll() is None:
            if deadline.remaining() <= _FINALIZE_MARGIN_S:
                _terminate(proc)
                raise TimeoutError(f"proxy worker timed out for {os.path.basename(path)}")
            time.sleep(_POLL_SECONDS)
    if proc.returncode != 0:
        raise RuntimeError(
            f"proxy worker failed for {os.path.basename(path)} (rc={proc.returncode})"
        )

    save_root = os.path.join(candidate_root, "outputs", probe_name)
    values = _loss_values(os.path.join(save_root, "loss_log.db"))
    if len(values) != expected_points:
        # The pinned logger intentionally omits the first training step.  The
        # generated config runs one extra step, so exact expected coverage here
        # is a hard completeness check, not a best-effort log scrape.
        raise RuntimeError(
            f"proxy recorder has {len(values)} loss points; expected {expected_points}"
        )
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError("proxy recorder contains non-finite loss")
    result = list(values)
    # Optimizer/model outputs are not evidence and can be hundreds of MiB per
    # stratum. Keep only the in-memory losses; the complete manifest is the
    # durable record after all candidates succeed.
    shutil.rmtree(candidate_root, ignore_errors=True)
    return result


def _probe_config(
    cfg: dict[str, Any],
    *,
    candidate: str,
    probe_dir: str,
    candidate_root: str,
    expected_points: int,
    probe_name: str = "probe",
) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["config"]["name"] = probe_name
    process = out["config"]["process"][0]
    process["training_folder"] = os.path.join(candidate_root, "outputs")
    process["sqlite_db_path"] = os.path.join(candidate_root, "aitk.db")
    process["trigger_word"] = None

    network = process.setdefault("network", {})
    network["pretrained_lora_path"] = candidate
    network["dropout"] = 0.0

    datasets = process.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise ValueError("proxy scorer requires exactly one source dataset")
    dataset = copy.deepcopy(datasets[0])
    process["datasets"] = [dataset]
    dataset["folder_path"] = probe_dir
    dataset["caption_dropout_rate"] = 0.0
    dataset["token_dropout_rate"] = 0.0
    dataset["shuffle_tokens"] = False
    dataset["flip_x"] = False
    dataset["flip_y"] = False
    dataset["random_crop"] = False
    dataset["random_scale"] = False
    dataset["augments"] = []
    dataset["num_repeats"] = 1
    dataset["resolution"] = [512]
    dataset["trigger_word"] = None

    train = process["train"]
    train["steps"] = expected_points + 1
    train.pop("start_step", None)
    train["lr"] = 0.0
    train["batch_size"] = 1
    train["gradient_accumulation"] = 1
    train["gradient_accumulation_steps"] = 1
    train["skip_first_sample"] = True
    train["force_first_sample"] = False
    train["disable_sampling"] = True
    train.setdefault("optimizer_params", {})["weight_decay"] = 0.0

    save = process.setdefault("save", {})
    save["save_every"] = expected_points + 2
    save["max_step_saves_to_keep"] = 1
    save["push_to_hub"] = False
    process.setdefault("logging", {})["log_every"] = 1
    process["logging"]["use_ui_logger"] = True
    process["logging"]["use_wandb"] = False
    return out


def _build_probe_datasets(
    holdout_dir: str,
    probe_root: str,
) -> tuple[str, str, int]:
    """Build separate captioned/blank strata over the same held-out images."""
    shutil.rmtree(probe_root, ignore_errors=True)
    captioned_dir = os.path.join(probe_root, "captioned")
    blank_dir = os.path.join(probe_root, "blank")
    os.makedirs(captioned_dir)
    os.makedirs(blank_dir)
    images: list[tuple[str, str, str]] = []
    for name in sorted(os.listdir(holdout_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        image = os.path.join(holdout_dir, name)
        caption = os.path.join(holdout_dir, stem + ".txt")
        if os.path.isfile(image) and os.path.isfile(caption):
            images.append((image, caption, ext.lower()))
    for index, (image, caption, ext) in enumerate(images):
        stem = f"h{index:03d}"
        shutil.copy2(image, os.path.join(captioned_dir, stem + ext))
        shutil.copyfile(caption, os.path.join(captioned_dir, stem + ".txt"))
        shutil.copy2(image, os.path.join(blank_dir, stem + ext))
        with open(os.path.join(blank_dir, stem + ".txt"), "wb"):
            pass
    return captioned_dir, blank_dir, len(images)


def _loss_values(path: str) -> list[float]:
    if not os.path.isfile(path):
        return []
    uri = f"file:{os.path.abspath(path)}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=2.0) as conn:
        rows = conn.execute(
            "SELECT value_real FROM metrics WHERE key = 'loss/loss' ORDER BY step"
        ).fetchall()
    return [float(row[0]) for row in rows if row[0] is not None]


def _valid_candidates(save_root: str, scope: dict[str, Any]) -> list[str]:
    return [
        path
        for path in checkpoints.current_loras(save_root, scope)
        if valid_safetensors(path)
    ]


def _step_of(path: str, scope: dict[str, Any]) -> int | None:
    match = re.search(r"_(\d+)\.safetensors$", os.path.basename(path))
    if match:
        return int(match.group(1))
    try:
        step = int(scope.get("planned_steps"))
        return step if step > 0 else None
    except (TypeError, ValueError):
        return None


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: str, value: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".tmp"
    try:
        with open(temp, "w", encoding="utf-8") as fh:
            json.dump(value, fh, sort_keys=True, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, path)
        _fsync_parent(path)
    except BaseException:
        try:
            os.remove(temp)
        except OSError:
            pass
        raise


def _remove_manifest(path: str) -> None:
    changed = False
    for candidate in (path, path + ".tmp"):
        try:
            os.remove(candidate)
            changed = True
        except FileNotFoundError:
            pass
        except Exception:
            # A manifest we cannot remove must never be replaced or trusted.
            raise
    if changed:
        _fsync_parent(path)


def _fsync_parent(path: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(os.path.dirname(os.path.abspath(path)), flags)
    try:
        try:
            os.fsync(fd)
        except OSError:
            pass
    finally:
        os.close(fd)


def _terminate(proc: subprocess.Popen) -> None:
    def _signal_group(sig: int) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            # A process-group lookup can fail under restricted runtimes. Direct
            # signalling still guarantees the scorer parent is not silently
            # left alive while finalization deletes its working directory.
            proc.send_signal(sig)

    _signal_group(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_group(signal.SIGKILL)
    proc.wait(timeout=10)
    if proc.poll() is None:
        raise RuntimeError("proxy worker survived SIGKILL")
