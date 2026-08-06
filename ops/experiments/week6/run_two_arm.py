#!/usr/bin/env python3
"""Week-6 two-arm (plus zero-LoRA control) Krea2 experiment runner.

WHAT THIS IS
------------
One deterministic, resumable driver for the head-to-head declared in
``ops/experiments/week6/RECIPE-PROVENANCE.md``:

* arm A  ``leader_derived``  — the Aug-3 R1 leader's own published recipe
* arm B  ``incumbent``       — what we shipped on Aug-3, steps pinned to match
* arm Z  ``zero_lora``       — an all-zero LoRA control that trains NOTHING

It trains nothing itself and scores nothing itself.  Training is the pinned
Forge path (``forge.cli.main`` with the arm's frozen config injected in place of
``forge.tasks.aitoolkit.build_config`` — exactly the substitution
``ops/calibration/run_krea_ladder.py`` already makes).  Scoring is the pinned
exact harness ``ops/calibration/evaluate_krea_local.py``, one fresh process per
candidate.  This file is orchestration, evidence, and statistics only.

NON-NEGOTIABLES ENCODED HERE
----------------------------
1. **Predeclaration.**  The decision rule is written to
   ``<workdir>/PREDECLARATION.json`` create-only, before any GPU work.  Every
   later invocation re-derives it and fails loudly on any drift, so the rule
   cannot be edited once results exist.
2. **Blank and prompted stay separate.**  The evaluator's two strata are always
   reported separately alongside the 0.25/0.75 composite.  Per-image rows are
   retained in every cell result — week 5 lost its per-row tables and could not
   reconstruct them.
3. **Unloaded LoRA keys are fatal by default.**  The pinned ComfyUI log is
   parsed for ``lora key not loaded:`` after every score; the count and the
   UNet/text-encoder split are recorded, the arm is flagged, and the run aborts
   unless ``--on-unloaded-keys flag`` is given.  (This is how the leader's 504
   inert text-encoder tensors were found; a silently scored arm is worthless.)
4. **Resumable by content.**  Every cell is keyed by a SHA-256 over its complete
   input identity.  A completed cell is never recomputed; a partial cell is
   moved aside deterministically, never reused.
5. **Deterministic.**  No timestamps in the plan, the predeclaration, or the
   analysis; the bootstrap is seeded and the seed is recorded; measured
   durations appear only in receipts, labelled as measurements.

WHERE THIS RUNS
---------------
On the GPU host, like ``run_krea_ladder.py``.  ``--host`` names that host and
its pinned checkouts; execute mode refuses to run anywhere else.  ``plan``
(dry-run) is host-independent and uses zero GPU.

USAGE
-----
    # dry run: validate everything, cost it, write the predeclaration
    python3 ops/experiments/week6/run_two_arm.py plan \\
        --fixture <fixture-manifest.json> \\
        --arm ops/experiments/week6/arms/leader_derived.json \\
        --arm ops/experiments/week6/arms/incumbent.json \\
        --arm ops/experiments/week6/arms/zero_lora_control.json \\
        --host <gpu-host.json> --workdir <durable-dir>

    # execute (same arguments)
    python3 ops/experiments/week6/run_two_arm.py run ...

    # re-derive the analysis from cells already on disk (no GPU)
    python3 ops/experiments/week6/run_two_arm.py analyze ...
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Sequence


SCHEMA = 1
KIND = "forge-week6-two-arm-experiment"
FIXTURE_KINDS = ("forge-week6-real-fixture", "forge-week6-real-fixture-planonly")
ARM_KIND = "forge-week6-arm"
HOST_KIND = "forge-week6-gpu-host"
PREDECLARATION_KIND = "forge-week6-two-arm-predeclaration"

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
EVALUATOR_PATH = REPO_ROOT / "ops" / "calibration" / "evaluate_krea_local.py"
LADDER_PATH = REPO_ROOT / "ops" / "calibration" / "run_krea_ladder.py"

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
# Prior art: the flux dual-runtime contract used exactly this pattern against
# the same pinned ComfyUI loader.
_UNLOADED_KEY_RE = re.compile(r"lora key not loaded:\s*([^\r\n]+)")
_REEXEC_MARKER = "FORGE_WEEK6_TWO_ARM_SEEDED"

# The scoring harness (evaluate_krea_local.py) is Krea2-only by construction.
SUPPORTED_MODEL_TYPE = "krea2"

# Config pointers the runner is allowed to rewrite when it binds a frozen arm
# recipe to this host and this fixture.  Anything else changing is a bug and
# aborts the cell.  (Mirrors run_krea_ladder's allowed-mutation discipline.)
_BINDING_POINTERS = (
    "/config/name",
    "/config/process/0/training_folder",
    "/config/process/0/trigger_word",
    "/config/process/0/datasets/0/folder_path",
    "/config/process/0/model/name_or_path",
    "/config/process/0/model/model_kwargs/vae_path",
    "/config/process/0/training_seed",
)


# ---------------------------------------------------------------------------
# Predeclared decision rule.  Edit ONLY before a workdir exists; every workdir
# pins the rule it was created with and refuses to run under a different one.
# ---------------------------------------------------------------------------

DECISION_RULE: dict[str, Any] = {
    "rule_id": "week6-two-arm-blank-lower-bound-v1",
    "declared_before_any_result_exists": True,
    "objective_of_record": (
        "validator loss = 0.25 * caption-guided reconstruction + 0.75 * "
        "blank-prompt reconstruction; lower is better"
    ),
    "primary_decision_metric": "blank_prompt_loss (the 0.75-weighted stratum)",
    "checkpoint_selection_per_arm": (
        "the arm's best checkpoint is the one with the lowest composite mean "
        "(0.25*prompted + 0.75*blank); ties break on lowest blank mean, then "
        "lowest step, then lowest candidate sha256"
    ),
    "head_to_head_statistic": (
        "per holdout image i, delta_i = blank_i(other arm's best checkpoint) - "
        "blank_i(this arm's best checkpoint); the statistic is mean(delta_i), "
        "positive meaning this arm reconstructs better"
    ),
    "uncertainty": {
        "method": "nonparametric percentile bootstrap",
        "resample_unit": "holdout IMAGE (not seed, not generation)",
        "iterations": 10000,
        "seed": 20260806,
        "interval": 0.95,
    },
    "win_condition": (
        "an arm wins only if the bootstrap 95% LOWER bound of its head-to-head "
        "blank advantage over the other arm's best checkpoint is strictly "
        "above zero"
    ),
    "null_result": (
        "if no arm satisfies the win condition the result is NULL: no winner is "
        "declared, nothing ships, and the incumbent release pin stands"
    ),
    "disqualifiers": [
        "the arm reported ANY unloaded LoRA key in the pinned ComfyUI log",
        "any planned cell for the arm is missing, failed, or unscored",
        "fewer than 2 holdout images (the image bootstrap is undefined)",
        "the arms disagree on holdout image identity (per-image pairing broken)",
    ],
    "reported_but_not_decisive": [
        "prompted (caption-guided) loss",
        "the 0.25/0.75 composite",
        "paired per-image deltas versus the zero-LoRA control with bootstrap CIs",
        "relative (percentage) deltas versus the zero-LoRA control",
    ],
    "known_limitation": (
        "the pinned harness returns one loss per image already averaged over its "
        "fixed generation seeds, so per-seed values are not recoverable and the "
        "bootstrap is over images only"
    ),
}


# ---------------------------------------------------------------------------
# small deterministic helpers
# ---------------------------------------------------------------------------


class ExperimentError(RuntimeError):
    """Any fail-closed condition in planning or execution."""


class UnloadedLoraKeys(ExperimentError):
    """A scored candidate had LoRA keys the pinned ComfyUI never loaded."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ExperimentError(f"not a regular JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentError(f"JSON root is not an object: {path}")
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    import yaml  # local: keeps the analysis path importable without pyyaml

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ExperimentError(f"not a regular YAML file: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExperimentError(f"YAML root is not a mapping: {path}")
    return value


def write_json_exclusive(path: Path, value: Any) -> str:
    """Create-only publication.  Returns the file sha256.

    Idempotent under resume: if the file exists with byte-identical content the
    call succeeds; differing content is a fail-closed conflict.
    """
    path = Path(path)
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    encoded = payload.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if path.exists():
        existing = path.read_bytes()
        if existing != encoded:
            raise ExperimentError(
                f"refusing to overwrite differing evidence: {path}"
            )
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.tmp"
    if temp.exists():
        temp.unlink()
    with temp.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temp, path)
    temp.unlink()
    return digest


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ExperimentError(message)


def _safe_id(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and _SAFE_ID.fullmatch(value) and value not in (".", ".."),
        f"{label} must be one conservative identifier component, got {value!r}",
    )
    return str(value)


def _sha256_field(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value),
        f"{label} must be a lowercase hex sha256",
    )
    return str(value)


def _positive_float(value: Any, label: str, *, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentError(f"{label} must be a number") from exc
    require(
        math.isfinite(number) and 0.0 < number <= upper,
        f"{label} must be finite and in (0, {upper}]",
    )
    return number


def _positive_int(value: Any, label: str, *, upper: int) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and 0 < value <= upper,
        f"{label} must be an integer in [1, {upper}]",
    )
    return int(value)


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(os.path.expanduser(str(value)))
    if not candidate.is_absolute():
        candidate = base / candidate
    return Path(os.path.abspath(candidate))


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldoutRow:
    index: int
    image: str
    image_sha256: str
    prompt: str
    prompt_sha256: str

    def identity(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "image": self.image,
            "image_sha256": self.image_sha256,
            "prompt": self.prompt,
            "prompt_sha256": self.prompt_sha256,
        }


@dataclass(frozen=True)
class Fixture:
    path: Path
    manifest_sha256: str
    fixture_id: str
    task_id: str
    model: str
    model_type: str
    trigger_word: str | None
    hours_to_complete: float
    plan_only: bool
    train_pairs: int
    train_archive: Path | None
    train_archive_sha256: str | None
    holdout_pairs: int
    holdout_dir: Path | None
    holdout_rows: tuple[HoldoutRow, ...]
    source: dict[str, Any]

    @property
    def holdout_dataset_sha256(self) -> str | None:
        if not self.holdout_rows:
            return None
        return canonical_hash([row.identity() for row in self.holdout_rows])

    def identity(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "task_id": self.task_id,
            "model": self.model,
            "model_type": self.model_type,
            "trigger_word": self.trigger_word,
            "hours_to_complete": self.hours_to_complete,
            "plan_only": self.plan_only,
            "manifest_path": str(self.path),
            "manifest_sha256": self.manifest_sha256,
            "train_pairs": self.train_pairs,
            "train_archive_sha256": self.train_archive_sha256,
            "holdout_pairs": self.holdout_pairs,
            "holdout_dataset_sha256": self.holdout_dataset_sha256,
            "source": self.source,
        }


def load_fixture(path: Path) -> Fixture:
    path = Path(os.path.abspath(os.path.expanduser(str(path))))
    raw = read_json(path)
    base = path.parent
    require(raw.get("schema") == SCHEMA, f"fixture schema must be {SCHEMA}")
    kind = raw.get("kind")
    require(kind in FIXTURE_KINDS, f"fixture kind must be one of {FIXTURE_KINDS}")
    plan_only = bool(raw.get("plan_only", False))
    require(
        plan_only == (kind == "forge-week6-real-fixture-planonly"),
        "fixture 'plan_only' must agree with its kind",
    )

    fixture_id = _safe_id(raw.get("fixture_id"), "fixture_id")
    task_id = _safe_id(raw.get("task_id"), "task_id")
    model = raw.get("model")
    require(
        isinstance(model, str) and model.strip() and "\x00" not in model,
        "fixture 'model' must be the validator's --model string",
    )
    model_type = str(raw.get("model_type") or "").strip().lower()
    require(
        model_type == SUPPORTED_MODEL_TYPE,
        f"this runner only drives {SUPPORTED_MODEL_TYPE}; got {model_type!r}",
    )
    trigger = raw.get("trigger_word")
    require(
        trigger is None or (isinstance(trigger, str) and trigger.strip()),
        "fixture 'trigger_word' must be a non-empty string or null",
    )
    hours = _positive_float(raw.get("hours_to_complete"), "hours_to_complete", upper=168.0)

    train = raw.get("train")
    require(isinstance(train, dict), "fixture 'train' must be an object")
    train_pairs = _positive_int(train.get("pairs"), "train.pairs", upper=100_000)
    train_archive: Path | None = None
    train_archive_sha: str | None = None
    if plan_only:
        require(
            "archive_path" not in train,
            "a plan-only fixture must not declare a training archive",
        )
    else:
        require(
            isinstance(train.get("archive_path"), str),
            "train.archive_path is required on an executable fixture",
        )
        train_archive = _resolve(base, train["archive_path"])
        train_archive_sha = _sha256_field(train.get("archive_sha256"), "train.archive_sha256")

    holdout = raw.get("holdout")
    require(isinstance(holdout, dict), "fixture 'holdout' must be an object")
    holdout_pairs = _positive_int(holdout.get("pairs"), "holdout.pairs", upper=100_000)
    holdout_dir: Path | None = None
    rows: list[HoldoutRow] = []
    if plan_only:
        require(
            not holdout.get("rows"),
            "a plan-only fixture must not declare holdout rows (no split exists)",
        )
    else:
        require(
            isinstance(holdout.get("dir"), str),
            "holdout.dir is required on an executable fixture",
        )
        holdout_dir = _resolve(base, holdout["dir"])
        raw_rows = holdout.get("rows")
        require(
            isinstance(raw_rows, list) and len(raw_rows) == holdout_pairs,
            "holdout.rows must be a list with exactly holdout.pairs entries",
        )
        seen_images: set[str] = set()
        seen_stems: set[str] = set()
        for index, row in enumerate(raw_rows):
            require(isinstance(row, dict), f"holdout row {index} is not an object")
            image = row.get("image")
            require(
                isinstance(image, str)
                and image == os.path.basename(image)
                and image not in ("", ".", "..")
                and image not in seen_images,
                f"holdout row {index} 'image' must be a unique bare filename",
            )
            stem, extension = os.path.splitext(image)
            require(
                extension and stem and stem not in seen_stems,
                f"holdout row {index} image stem must be unique and non-empty",
            )
            prompt = row.get("prompt", f"{stem}.txt")
            require(
                prompt == f"{stem}.txt",
                f"holdout row {index} prompt must be '<image stem>.txt'",
            )
            seen_images.add(image)
            seen_stems.add(stem)
            rows.append(
                HoldoutRow(
                    index=index,
                    image=image,
                    image_sha256=_sha256_field(
                        row.get("image_sha256"), f"holdout row {index} image_sha256"
                    ),
                    prompt=str(prompt),
                    prompt_sha256=_sha256_field(
                        row.get("prompt_sha256"), f"holdout row {index} prompt_sha256"
                    ),
                )
            )

    source = raw.get("source", {})
    require(isinstance(source, dict), "fixture 'source' must be an object when present")

    return Fixture(
        path=path,
        manifest_sha256=sha256_file(path),
        fixture_id=fixture_id,
        task_id=task_id,
        model=str(model),
        model_type=model_type,
        trigger_word=(str(trigger) if isinstance(trigger, str) else None),
        hours_to_complete=hours,
        plan_only=plan_only,
        train_pairs=train_pairs,
        train_archive=train_archive,
        train_archive_sha256=train_archive_sha,
        holdout_pairs=holdout_pairs,
        holdout_dir=holdout_dir,
        holdout_rows=tuple(rows),
        source=source,
    )


@dataclass(frozen=True)
class Arm:
    path: Path
    spec_sha256: str
    arm_id: str
    role: str  # "train" | "zero_lora_control"
    label: str
    expected_repo_name: str | None
    config_path: Path | None
    config_sha256: str | None
    ai_toolkit_commit: str | None
    training_seed: int | None
    steps: int | None
    save_every: int | None
    candidate_path: Path | None
    candidate_sha256: str | None
    zero_lora_record_path: Path | None

    @property
    def trains(self) -> bool:
        return self.role == "train"

    def identity(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "role": self.role,
            "label": self.label,
            "spec_path": str(self.path),
            "spec_sha256": self.spec_sha256,
            "expected_repo_name": self.expected_repo_name,
            "config_path": (str(self.config_path) if self.config_path else None),
            "config_sha256": self.config_sha256,
            "ai_toolkit_commit": self.ai_toolkit_commit,
            "training_seed": self.training_seed,
            "steps": self.steps,
            "save_every": self.save_every,
            "candidate_path": (str(self.candidate_path) if self.candidate_path else None),
            "candidate_sha256": self.candidate_sha256,
            "zero_lora_record_path": (
                str(self.zero_lora_record_path) if self.zero_lora_record_path else None
            ),
        }


def load_arm(path: Path) -> Arm:
    path = Path(os.path.abspath(os.path.expanduser(str(path))))
    raw = read_json(path)
    base = path.parent
    require(raw.get("schema") == SCHEMA, f"arm schema must be {SCHEMA}")
    require(raw.get("kind") == ARM_KIND, f"arm kind must be {ARM_KIND!r}")
    arm_id = _safe_id(raw.get("arm_id"), "arm_id")
    role = raw.get("role")
    require(
        role in ("train", "zero_lora_control"),
        "arm 'role' must be 'train' or 'zero_lora_control'",
    )
    label = raw.get("label", arm_id)
    require(isinstance(label, str) and label.strip(), "arm 'label' must be a string")

    expected_repo_name: str | None = None
    config_path: Path | None = None
    config_sha: str | None = None
    toolkit_commit: str | None = None
    training_seed: int | None = None
    steps: int | None = None
    save_every: int | None = None
    candidate_path: Path | None = None
    candidate_sha: str | None = None
    record_path: Path | None = None

    if role == "train":
        expected_repo_name = _safe_id(
            raw.get("expected_repo_name"), "expected_repo_name"
        )
        require(
            isinstance(raw.get("config_path"), str), "train arm needs 'config_path'"
        )
        config_path = _resolve(base, raw["config_path"])
        require(config_path.is_file(), f"arm config is missing: {config_path}")
        config_sha = sha256_file(config_path)
        declared = raw.get("config_sha256")
        if declared is not None:
            require(
                _sha256_field(declared, "config_sha256") == config_sha,
                f"arm config sha256 mismatch for {arm_id}: declared {declared}, "
                f"actual {config_sha}",
            )
        commit = raw.get("ai_toolkit_commit")
        require(
            isinstance(commit, str) and _GIT_SHA_RE.fullmatch(commit),
            "train arm needs 'ai_toolkit_commit' (full 40-hex commit)",
        )
        toolkit_commit = str(commit)
        training_seed = raw.get("training_seed")
        require(
            isinstance(training_seed, int)
            and not isinstance(training_seed, bool)
            and 0 <= training_seed < 2**32,
            "train arm needs an integer 'training_seed' in [0, 2**32)",
        )
        config = read_yaml(config_path)
        steps, save_every = _config_depth(config, config_path)
    else:
        require(
            isinstance(raw.get("candidate_path"), str),
            "zero-LoRA arm needs 'candidate_path' (built by "
            "build_zero_lora_baseline.py; this runner never fabricates weights)",
        )
        candidate_path = _resolve(base, raw["candidate_path"])
        candidate_sha = _sha256_field(raw.get("candidate_sha256"), "candidate_sha256")
        require(
            isinstance(raw.get("zero_lora_record_path"), str),
            "zero-LoRA arm needs 'zero_lora_record_path' (the builder's record)",
        )
        record_path = _resolve(base, raw["zero_lora_record_path"])

    return Arm(
        path=path,
        spec_sha256=sha256_file(path),
        arm_id=arm_id,
        role=str(role),
        label=str(label),
        expected_repo_name=expected_repo_name,
        config_path=config_path,
        config_sha256=config_sha,
        ai_toolkit_commit=toolkit_commit,
        training_seed=training_seed,
        steps=steps,
        save_every=save_every,
        candidate_path=candidate_path,
        candidate_sha256=candidate_sha,
        zero_lora_record_path=record_path,
    )


def _config_depth(config: dict[str, Any], path: Path) -> tuple[int, int]:
    try:
        process = config["config"]["process"][0]
        train = process["train"]
        save = process["save"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ExperimentError(f"arm config has no config.process[0]: {path}") from exc
    steps = _positive_int(train.get("steps"), f"{path.name} train.steps", upper=10_000_000)
    save_every = _positive_int(
        save.get("save_every"), f"{path.name} save.save_every", upper=10_000_000
    )
    arch = str(process.get("model", {}).get("arch", "")).strip().lower()
    require(
        arch == SUPPORTED_MODEL_TYPE,
        f"{path.name}: model.arch must be {SUPPORTED_MODEL_TYPE!r}, got {arch!r}",
    )
    return steps, save_every


def projected_checkpoints(steps: int, save_every: int) -> list[int]:
    """Periodic saves strictly before the terminal step, plus the exact final.

    ai-toolkit writes ``{name}_{step:09d}.safetensors`` on the cadence and
    ``{name}.safetensors`` at the end.  A periodic save that lands exactly on
    the terminal step is the same artifact as the final, so it is not counted
    twice; execution deduplicates by content hash regardless.
    """
    periodic = [
        step for step in range(save_every, steps, save_every) if 0 < step < steps
    ]
    return periodic + [steps]


@dataclass(frozen=True)
class Host:
    path: Path
    spec_sha256: str
    host_id: str
    hostname: str | None
    ai_toolkit_dir: Path
    comfy_root: Path
    comfy_python: Path
    god_root: Path
    god_commit: str
    comfy_commit: str
    tooling_commit: str
    base_name: str
    gpu_hourly_usd: float
    comfy_port: int

    def identity(self) -> dict[str, Any]:
        return {
            "host_id": self.host_id,
            "hostname": self.hostname,
            "spec_path": str(self.path),
            "spec_sha256": self.spec_sha256,
            "ai_toolkit_dir": str(self.ai_toolkit_dir),
            "comfy_root": str(self.comfy_root),
            "comfy_python": str(self.comfy_python),
            "god_root": str(self.god_root),
            "god_commit": self.god_commit,
            "comfy_commit": self.comfy_commit,
            "tooling_commit": self.tooling_commit,
            "base_name": self.base_name,
            "comfy_port": self.comfy_port,
            "gpu_hourly_usd": self.gpu_hourly_usd,
        }


def load_host(path: Path) -> Host:
    path = Path(os.path.abspath(os.path.expanduser(str(path))))
    raw = read_json(path)
    base = path.parent
    require(raw.get("schema") == SCHEMA, f"host schema must be {SCHEMA}")
    require(raw.get("kind") == HOST_KIND, f"host kind must be {HOST_KIND!r}")
    host_id = _safe_id(raw.get("host_id"), "host_id")
    hostname = raw.get("hostname")
    require(
        hostname is None or (isinstance(hostname, str) and hostname.strip()),
        "host 'hostname' must be a non-empty string or null "
        "(null disables the on-host identity check and is only for lab replay)",
    )
    paths: dict[str, Path] = {}
    for key in ("ai_toolkit_dir", "comfy_root", "comfy_python", "god_root"):
        require(isinstance(raw.get(key), str), f"host '{key}' is required")
        paths[key] = _resolve(base, raw[key])
    commits = {}
    for key in ("god_commit", "comfy_commit", "tooling_commit"):
        value = raw.get(key)
        require(
            isinstance(value, str) and _GIT_SHA_RE.fullmatch(value),
            f"host '{key}' must be a full 40-hex commit",
        )
        commits[key] = str(value)
    base_name = raw.get("base_name", "models--Comfy-Org--Krea-2.safetensors")
    require(
        isinstance(base_name, str) and base_name == os.path.basename(base_name),
        "host 'base_name' must be one ComfyUI filename",
    )
    usd = _positive_float(raw.get("gpu_hourly_usd", 2.0), "gpu_hourly_usd", upper=1000.0)
    port = raw.get("comfy_port", 8188)
    require(
        isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535,
        "host 'comfy_port' must be a TCP port",
    )
    return Host(
        path=path,
        spec_sha256=sha256_file(path),
        host_id=host_id,
        hostname=(str(hostname) if isinstance(hostname, str) else None),
        ai_toolkit_dir=paths["ai_toolkit_dir"],
        comfy_root=paths["comfy_root"],
        comfy_python=paths["comfy_python"],
        god_root=paths["god_root"],
        god_commit=commits["god_commit"],
        comfy_commit=commits["comfy_commit"],
        tooling_commit=commits["tooling_commit"],
        base_name=str(base_name),
        gpu_hourly_usd=usd,
        comfy_port=int(port),
    )


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """Planning constants.  Every field is labelled OBSERVED or INFERRED.

    None of these change what runs; they only price the plan.  Execution
    records the measured values, and ``analyze`` reports plan-vs-measured.
    """

    sec_per_step: float = 2.2
    startup_s: float = 300.0
    export_reserve_s: float = 180.0
    save_overhead_s: float = 30.0
    eval_fixed_overhead_s: float = 247.54
    eval_generations: int = 5
    eval_seconds_per_generation: float = 12.5722
    gpu_hourly_usd: float = 2.0

    def provenance(self) -> dict[str, str]:
        return {
            "sec_per_step": (
                "DEFAULT 2.2 = forge.recipe.SEC_PER_IT['krea2'] at pin 084ea914 — "
                "OBSERVED as a source constant, and it is a deliberately "
                "conservative planning bound, NOT a measured throughput. Prior "
                "week-6 forensics report ~1.27 s/step measured on the Aug-3 "
                "tournament run; pass --sec-per-step to price that instead."
            ),
            "startup_s": "DEFAULT 300 = forge.recipe.STARTUP_S (OBSERVED constant)",
            "export_reserve_s": (
                "DEFAULT 180 = forge.cli._EXPORT_RESERVE_SECONDS (OBSERVED constant)"
            ),
            "save_overhead_s": (
                "DEFAULT 30 — INFERRED from forge/tasks/aitoolkit.py's 'each "
                "tournament save took tens of seconds'; not measured here"
            ),
            "eval_fixed_overhead_s": (
                "DEFAULT 247.54 — OBSERVED. Intercept of a two-point fit over "
                "83 real exact-score records from this same harness at these "
                "same pins (week-5 scorer fleet, NVIDIA H100 PCIe): 24-image "
                "datasets median 3264.86 s, 40-image datasets median 5276.41 s. "
                "This is per-cell ComfyUI startup + model load + teardown."
            ),
            "eval_generations": (
                "DEFAULT 5 — OBSERVED. Every one of those 83 records reports "
                "generations=5, steps=25, cfg=12, denoise=0.8, text_weight=0.25 "
                "as read from validator.evaluation.constants.EVAL_DEFAULTS in "
                "the pinned G.O.D checkout. (forge/recipe.py's '10-seed' "
                "comment is stale; execution re-reads the real value and "
                "records generations_match_plan.)"
            ),
            "eval_seconds_per_generation": (
                "DEFAULT 12.5722 — OBSERVED. Slope of the same two-point fit: "
                "(5276.41-3264.86)/((40-24)*5*2) s per image-generation, where "
                "each image costs generations x 2 strata (prompted + blank). "
                "Measured on H100 PCIe at week-5 fixture resolutions; week-6 "
                "fixtures are 1024x768, so treat it as close but not identical."
            ),
            "gpu_hourly_usd": "the price the task fixed for this estimate",
        }

    def train_seconds(self, steps: int, saves: int) -> float:
        return (
            self.startup_s
            + steps * self.sec_per_step
            + saves * self.save_overhead_s
            + self.export_reserve_s
        )

    def score_seconds(self, holdout_pairs: int) -> float:
        # The harness runs both strata: prompted and blank, over its fixed seeds.
        return (
            self.eval_fixed_overhead_s
            + holdout_pairs
            * self.eval_generations
            * 2
            * self.eval_seconds_per_generation
        )

    def record(self) -> dict[str, Any]:
        return {"values": asdict(self), "provenance": self.provenance()}


# ---------------------------------------------------------------------------
# cell identity
# ---------------------------------------------------------------------------


def train_cell_inputs(
    *, fixture: Fixture, arm: Arm, host: Host, hours: float
) -> dict[str, Any]:
    return {
        "cell_kind": "train",
        "experiment_schema": SCHEMA,
        "fixture": {
            "fixture_id": fixture.fixture_id,
            "task_id": fixture.task_id,
            "manifest_sha256": fixture.manifest_sha256,
            "train_archive_sha256": fixture.train_archive_sha256,
            "train_pairs": fixture.train_pairs,
            "model": fixture.model,
            "model_type": fixture.model_type,
            "trigger_word": fixture.trigger_word,
        },
        "arm": {
            "arm_id": arm.arm_id,
            "spec_sha256": arm.spec_sha256,
            "config_sha256": arm.config_sha256,
            "expected_repo_name": arm.expected_repo_name,
            "ai_toolkit_commit": arm.ai_toolkit_commit,
            "training_seed": arm.training_seed,
            "steps": arm.steps,
            "save_every": arm.save_every,
        },
        "runtime": {
            "hours_to_complete": hours,
            "holdout_selection_types": "",
            "binding_pointers": list(_BINDING_POINTERS),
            "driver_sha256": sha256_file(SCRIPT_PATH),
        },
    }


def score_cell_inputs(
    *,
    fixture: Fixture,
    arm: Arm,
    host: Host,
    candidate_sha256: str,
    candidate_name: str,
    step: int | None,
    generations_planned: int,
) -> dict[str, Any]:
    return {
        "cell_kind": "score",
        "experiment_schema": SCHEMA,
        "fixture": {
            "fixture_id": fixture.fixture_id,
            "manifest_sha256": fixture.manifest_sha256,
            "holdout_dataset_sha256": fixture.holdout_dataset_sha256,
            "holdout_pairs": fixture.holdout_pairs,
        },
        "arm_id": arm.arm_id,
        "candidate": {
            "name": candidate_name,
            "sha256": candidate_sha256,
            "step": step,
        },
        "evaluator": {
            "harness_sha256": sha256_file(EVALUATOR_PATH),
            "god_commit": host.god_commit,
            "comfy_commit": host.comfy_commit,
            "tooling_commit": host.tooling_commit,
            "base_name": host.base_name,
            "generations_planned": generations_planned,
        },
        "driver_sha256": sha256_file(SCRIPT_PATH),
    }


def cell_key(inputs: dict[str, Any]) -> str:
    return canonical_hash(inputs)


def cell_dir(workdir: Path, inputs: dict[str, Any]) -> Path:
    return Path(workdir) / "cells" / str(inputs["cell_kind"]) / cell_key(inputs)


# ---------------------------------------------------------------------------
# cell store (resume)
# ---------------------------------------------------------------------------


RESULT_NAME = "result.json"


def completed_cell(directory: Path, key: str) -> dict[str, Any] | None:
    """Return a previously completed cell result, or None.

    A directory without a well-formed ``result.json`` is NOT complete; the
    caller must move it aside before recomputing.  A result whose recorded key
    disagrees with the recomputed key is a fail-closed corruption.
    """
    result_path = Path(directory) / RESULT_NAME
    if not result_path.is_file():
        return None
    value = read_json(result_path)
    if value.get("cell_key") != key:
        raise ExperimentError(
            f"cell result key mismatch in {result_path}: "
            f"recorded {value.get('cell_key')!r}, expected {key!r}"
        )
    if value.get("complete") is not True:
        raise ExperimentError(f"cell result is present but not complete: {result_path}")
    return value


def preempt_partial_cell(directory: Path) -> Path | None:
    """Deterministically move an incomplete cell aside.  Never reuse it.

    The suffix is the lowest free integer, not a timestamp or a random token,
    so a replay of the same interruption sequence produces the same names.
    """
    directory = Path(directory)
    if not directory.exists():
        return None
    if (directory / RESULT_NAME).is_file():
        return None
    for index in range(1000):
        target = directory.with_name(f"{directory.name}.incomplete.{index:03d}")
        if not target.exists():
            directory.rename(target)
            return target
    raise ExperimentError(f"too many incomplete cell directories beside {directory}")


# ---------------------------------------------------------------------------
# unloaded-key detection
# ---------------------------------------------------------------------------


def classify_lora_key(key: str) -> str:
    low = key.strip().lower()
    if (
        low.startswith("lora_te")
        or low.startswith("lora_te.")
        or "text_encoder" in low
        or "language_model" in low
    ):
        return "text_encoder"
    if low.startswith("lora_unet") or "diffusion_model" in low:
        return "unet"
    return "other"


def parse_unloaded_keys(log_text: str, *, sample: int = 20) -> dict[str, Any]:
    """Count ``lora key not loaded:`` reports in a pinned-ComfyUI log."""
    found = [match.strip() for match in _UNLOADED_KEY_RE.findall(log_text)]
    unique = sorted(set(found))
    by_namespace: dict[str, int] = {"unet": 0, "text_encoder": 0, "other": 0}
    for key in unique:
        by_namespace[classify_lora_key(key)] += 1
    return {
        "total_reports": len(found),
        "unique_keys": len(unique),
        "by_namespace": by_namespace,
        "keys_sha256": canonical_hash(unique),
        "keys_sample": unique[:sample],
        "keys_sample_truncated": len(unique) > sample,
        "clean": len(found) == 0,
        "pattern": _UNLOADED_KEY_RE.pattern,
    }


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def mean(values: Sequence[float]) -> float:
    require(values, "mean of an empty sequence")
    return sum(float(v) for v in values) / len(values)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    iterations: int,
    level: float = 0.95,
) -> dict[str, Any]:
    """Percentile bootstrap of the mean, resampling the IMAGE index.

    Deterministic: ``random.Random`` with the recorded seed, drawn in a fixed
    order.  Returns ``defined: False`` (rather than an interval) when fewer
    than two images exist, because the win condition must then be unreachable.
    """
    n = len(values)
    if n < 2:
        return {
            "defined": False,
            "reason": "fewer than 2 holdout images; the image bootstrap is undefined",
            "n": n,
            "mean": (float(values[0]) if n == 1 else None),
            "seed": seed,
            "iterations": iterations,
            "level": level,
            "method": "percentile bootstrap over holdout images",
        }
    data = [float(value) for value in values]
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += data[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo_index = int(math.floor((1.0 - level) / 2.0 * iterations))
    hi_index = int(math.ceil((1.0 + level) / 2.0 * iterations)) - 1
    lo_index = min(max(lo_index, 0), iterations - 1)
    hi_index = min(max(hi_index, 0), iterations - 1)
    return {
        "defined": True,
        "n": n,
        "mean": sum(data) / n,
        "lo": means[lo_index],
        "hi": means[hi_index],
        "seed": seed,
        "iterations": iterations,
        "level": level,
        "method": "percentile bootstrap over holdout images",
    }


def composite(prompted: float, blank: float) -> float:
    return 0.25 * float(prompted) + 0.75 * float(blank)


# ---------------------------------------------------------------------------
# planning / dry run
# ---------------------------------------------------------------------------


def build_plan(
    *,
    fixture: Fixture,
    arms: Sequence[Arm],
    host: Host,
    cost: CostModel,
    hours: float | None = None,
    workdir: Path,
) -> dict[str, Any]:
    train_arms = [arm for arm in arms if arm.trains]
    control_arms = [arm for arm in arms if not arm.trains]
    require(len(train_arms) == 2, "exactly two training arms are required")
    require(len(control_arms) == 1, "exactly one zero-LoRA control arm is required")
    ids = [arm.arm_id for arm in arms]
    require(len(set(ids)) == len(ids), "arm ids must be unique")
    repos = [arm.expected_repo_name for arm in train_arms]
    require(
        len(set(repos)) == len(repos),
        "training arms must use distinct expected_repo_name values "
        "(they share /app/checkpoints/<task-id>/)",
    )

    effective_hours = float(hours) if hours is not None else fixture.hours_to_complete
    hours_source = (
        "operator override (--hours)"
        if hours is not None
        else "fixture.hours_to_complete (the tournament grant)"
    )

    # Deterministic warnings only.  Anything that depends on which machine the
    # planner runs on belongs in environment_report(), never in the plan: the
    # plan is published create-only and must be byte-identical on replay.
    warnings: list[str] = []
    if fixture.plan_only:
        warnings.append(
            "fixture is PLAN-ONLY: no split, no bytes, no execution. "
            "Costs below are projections from declared pair counts."
        )

    train_cells: list[dict[str, Any]] = []
    projected_score_cells = 0
    train_seconds_total = 0.0
    for arm in train_arms:
        assert arm.steps is not None and arm.save_every is not None
        checkpoints = projected_checkpoints(arm.steps, arm.save_every)
        seconds = cost.train_seconds(arm.steps, len(checkpoints))
        train_seconds_total += seconds
        grant_s = effective_hours * 3600.0
        fits = seconds <= grant_s
        if not fits:
            warnings.append(
                f"arm {arm.arm_id}: {arm.steps} steps priced at "
                f"{cost.sec_per_step} s/step needs {seconds / 3600.0:.3f} h but the "
                f"grant is {effective_hours:.3f} h — the Forge deadline will "
                f"terminate training early and the arms will NOT be depth-matched"
            )
        inputs = train_cell_inputs(
            fixture=fixture, arm=arm, host=host, hours=effective_hours
        )
        key = cell_key(inputs)
        directory = cell_dir(workdir, inputs)
        train_cells.append(
            {
                "arm_id": arm.arm_id,
                "cell_key": key,
                "cell_dir": str(directory),
                "steps": arm.steps,
                "save_every": arm.save_every,
                "projected_checkpoint_steps": checkpoints,
                "projected_checkpoints": len(checkpoints),
                "estimated_seconds": seconds,
                "fits_in_grant": fits,
            }
        )
        projected_score_cells += len(checkpoints)
    projected_score_cells += len(control_arms)

    score_seconds_each = cost.score_seconds(fixture.holdout_pairs)
    score_seconds_total = score_seconds_each * projected_score_cells
    total_seconds = train_seconds_total + score_seconds_total
    total_hours = total_seconds / 3600.0

    estimate = {
        "train_gpu_hours": train_seconds_total / 3600.0,
        "score_gpu_hours": score_seconds_total / 3600.0,
        "total_gpu_hours": total_hours,
        "gpu_hourly_usd": cost.gpu_hourly_usd,
        "total_usd": total_hours * cost.gpu_hourly_usd,
        "score_cells_projected": projected_score_cells,
        "score_seconds_per_cell": score_seconds_each,
        "note": (
            "projection only; it prices the plan, it does not bound the run. "
            "Actual GPU time is recorded per cell as a measurement."
        ),
    }

    return {
        "schema": SCHEMA,
        "kind": KIND + "-plan",
        "driver": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
            "evaluator_path": str(EVALUATOR_PATH),
            "evaluator_sha256": sha256_file(EVALUATOR_PATH),
            "trainer_entrypoint": "forge.cli.main (pinned Forge path, config injected)",
        },
        "workdir": str(Path(workdir)),
        "fixture": fixture.identity(),
        "arms": [arm.identity() for arm in arms],
        "host": host.identity(),
        "hours": {"effective": effective_hours, "source": hours_source},
        "cost_model": cost.record(),
        "train_cells": train_cells,
        "estimate": estimate,
        "warnings": warnings,
        "decision_rule": DECISION_RULE,
        "executable": not fixture.plan_only,
    }


def environment_report(
    *,
    plan: dict[str, Any],
    fixture: Fixture,
    arms: Sequence[Arm],
    host: Host,
    workdir: Path,
) -> dict[str, Any]:
    """Machine-dependent validation.  Never written into the durable plan."""
    checks = _path_checks(fixture, arms, host)
    warnings = [
        f"missing {check['role']}: {check['path']}"
        for check in checks
        if not check["exists"]
    ]
    already = [
        cell["arm_id"]
        for cell in plan["train_cells"]
        if (Path(cell["cell_dir"]) / RESULT_NAME).is_file()
    ]
    return {
        "path_checks": checks,
        "environment_warnings": warnings,
        "train_cells_already_complete": already,
        "workdir_exists": Path(workdir).exists(),
    }


def _path_checks(
    fixture: Fixture, arms: Sequence[Arm], host: Host
) -> list[dict[str, Any]]:
    entries: list[tuple[str, Path | None]] = [
        ("fixture manifest", fixture.path),
        ("fixture train archive", fixture.train_archive),
        ("fixture holdout dir", fixture.holdout_dir),
        ("evaluator harness", EVALUATOR_PATH),
        ("ladder helpers", LADDER_PATH),
        ("host ai_toolkit_dir", host.ai_toolkit_dir),
        ("host comfy_root", host.comfy_root),
        ("host comfy_python", host.comfy_python),
        ("host god_root", host.god_root),
    ]
    for arm in arms:
        entries.append((f"arm {arm.arm_id} spec", arm.path))
        entries.append((f"arm {arm.arm_id} config", arm.config_path))
        entries.append((f"arm {arm.arm_id} candidate", arm.candidate_path))
        entries.append((f"arm {arm.arm_id} zero-LoRA record", arm.zero_lora_record_path))
    checks: list[dict[str, Any]] = []
    for role, path in entries:
        if path is None:
            continue
        checks.append(
            {"role": role, "path": str(path), "exists": Path(path).exists()}
        )
    return checks


def predeclaration(plan: dict[str, Any]) -> dict[str, Any]:
    """The scientific identity the rule is bound to — NOT the whole plan.

    Cost-model knobs and host paths are deliberately excluded: re-pricing the
    plan must not look like tampering with the rule, and tampering with the
    rule must not be able to hide behind a re-pricing.
    """
    fixture = plan["fixture"]
    arms = sorted(
        (
            {
                "arm_id": arm["arm_id"],
                "role": arm["role"],
                "spec_sha256": arm["spec_sha256"],
                "config_sha256": arm["config_sha256"],
            }
            for arm in plan["arms"]
        ),
        key=lambda item: item["arm_id"],
    )
    return {
        "schema": SCHEMA,
        "kind": PREDECLARATION_KIND,
        "decision_rule": DECISION_RULE,
        "bound_to": {
            "fixture_id": fixture["fixture_id"],
            "fixture_manifest_sha256": fixture["manifest_sha256"],
            "holdout_dataset_sha256": fixture["holdout_dataset_sha256"],
            "holdout_pairs": fixture["holdout_pairs"],
            "arms": arms,
        },
    }


def publish_plan(workdir: Path, plan: dict[str, Any]) -> Path:
    """Content-addressed so re-pricing a plan never collides with an old one."""
    digest = canonical_hash(plan)
    path = Path(workdir) / "plans" / f"plan-{digest[:12]}.json"
    write_json_exclusive(path, plan)
    return path


def establish_predeclaration(workdir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Write the rule create-only, or prove the existing one still matches."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "PREDECLARATION.json"
    expected = predeclaration(plan)
    if path.is_file():
        existing = read_json(path)
        if canonical_hash(existing) != canonical_hash(expected):
            raise ExperimentError(
                "PREDECLARATION DRIFT: the decision rule or the scientific "
                f"identity in {path} differs from the one this invocation "
                "derives. The rule is frozen once a workdir exists; use a new "
                "workdir for a new experiment."
            )
        return {"path": str(path), "sha256": sha256_file(path), "created": False}
    digest = write_json_exclusive(path, expected)
    return {"path": str(path), "sha256": digest, "created": True}


def format_estimate(plan: dict[str, Any], environment: dict[str, Any] | None = None) -> str:
    estimate = plan["estimate"]
    lines = [
        "=== WEEK-6 TWO-ARM PLAN ===",
        f"fixture     : {plan['fixture']['fixture_id']} "
        f"(task {plan['fixture']['task_id']}, "
        f"{plan['fixture']['train_pairs']} train / "
        f"{plan['fixture']['holdout_pairs']} holdout pairs)",
        f"executable  : {plan['executable']}",
        f"host        : {plan['host']['host_id']}",
        f"grant       : {plan['hours']['effective']:.3f} h "
        f"({plan['hours']['source']})",
        "",
        "arms:",
    ]
    for cell in plan["train_cells"]:
        lines.append(
            f"  {cell['arm_id']:<16} steps={cell['steps']:<6} "
            f"save_every={cell['save_every']:<5} "
            f"checkpoints={cell['projected_checkpoints']} "
            f"train={cell['estimated_seconds'] / 3600.0:.3f} h "
            f"fits_in_grant={cell['fits_in_grant']}"
        )
    for arm in plan["arms"]:
        if arm["role"] != "train":
            lines.append(f"  {arm['arm_id']:<16} zero-LoRA control (trains nothing)")
    lines += [
        "",
        f"score cells : {estimate['score_cells_projected']} "
        f"x {estimate['score_seconds_per_cell'] / 3600.0:.3f} h",
        f"train hours : {estimate['train_gpu_hours']:.3f}",
        f"score hours : {estimate['score_gpu_hours']:.3f}",
        f"TOTAL       : {estimate['total_gpu_hours']:.3f} GPU-hours",
        f"COST        : ${estimate['total_usd']:.2f} "
        f"at ${estimate['gpu_hourly_usd']:.2f}/GPU-hour",
    ]
    if plan["warnings"]:
        lines += ["", "plan warnings:"]
        lines += [f"  - {item}" for item in plan["warnings"]]
    if environment is not None:
        if environment["train_cells_already_complete"]:
            lines += [
                "",
                "already complete (would be skipped on resume): "
                + ", ".join(environment["train_cells_already_complete"]),
            ]
        if environment["environment_warnings"]:
            lines += ["", "environment warnings (this machine):"]
            lines += [f"  - {item}" for item in environment["environment_warnings"]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


def default_runner(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(argv),
        cwd=(str(cwd) if cwd is not None else None),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def refuse_gpu_runner(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess:
    raise ExperimentError("dry run attempted to execute a subprocess")


def git_head(path: Path, runner: Callable[..., subprocess.CompletedProcess]) -> str:
    result = runner(["git", "-C", str(path), "rev-parse", "HEAD"])
    if result.returncode != 0:
        raise ExperimentError(
            f"cannot read git HEAD in {path}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def preflight(
    *,
    fixture: Fixture,
    arms: Sequence[Arm],
    host: Host,
    runner: Callable[..., subprocess.CompletedProcess],
    hostname: str | None = None,
) -> dict[str, Any]:
    require(
        not fixture.plan_only,
        "refusing to execute a PLAN-ONLY fixture: it has no split and no bytes",
    )
    observed_host = hostname if hostname is not None else socket.gethostname()
    if host.hostname is not None:
        require(
            observed_host == host.hostname,
            f"this is {observed_host!r} but the host spec pins "
            f"{host.hostname!r}; run on the declared GPU host",
        )
    missing = [check for check in _path_checks(fixture, arms, host) if not check["exists"]]
    require(
        not missing,
        "missing required paths: "
        + ", ".join(f"{item['role']}={item['path']}" for item in missing),
    )
    toolkit_head = git_head(host.ai_toolkit_dir, runner)
    for arm in arms:
        if not arm.trains:
            continue
        require(
            toolkit_head.lower() == str(arm.ai_toolkit_commit).lower(),
            f"arm {arm.arm_id} declares ai-toolkit "
            f"{arm.ai_toolkit_commit} but {host.ai_toolkit_dir} is at "
            f"{toolkit_head}. Running an arm on the wrong runtime silently "
            "disables its score-critical settings; refusing.",
        )
    verified = verify_holdout_dir(fixture)
    return {
        "hostname": observed_host,
        "ai_toolkit_head": toolkit_head,
        "holdout": verified,
    }


def verify_holdout_dir(fixture: Fixture) -> dict[str, Any]:
    """The holdout directory must be exactly the manifest, byte for byte."""
    require(fixture.holdout_dir is not None, "fixture has no holdout directory")
    directory = Path(fixture.holdout_dir)
    require(directory.is_dir(), f"holdout dir is not a directory: {directory}")
    expected: dict[str, str] = {}
    for row in fixture.holdout_rows:
        expected[row.image] = row.image_sha256
        expected[row.prompt] = row.prompt_sha256
    actual_names = sorted(entry.name for entry in directory.iterdir())
    require(
        actual_names == sorted(expected),
        "holdout directory contents differ from the fixture manifest: "
        f"unexpected={sorted(set(actual_names) - set(expected))} "
        f"missing={sorted(set(expected) - set(actual_names))}",
    )
    for name, declared in sorted(expected.items()):
        path = directory / name
        require(
            path.is_file() and not path.is_symlink(),
            f"holdout entry is not a regular file: {path}",
        )
        actual = sha256_file(path)
        require(
            actual == declared,
            f"holdout byte mismatch for {name}: manifest {declared}, disk {actual}",
        )
    return {
        "dir": str(directory),
        "files": len(expected),
        "holdout_dataset_sha256": fixture.holdout_dataset_sha256,
        "verified": True,
    }


def run_train_cell(
    *,
    fixture: Fixture,
    arm: Arm,
    host: Host,
    hours: float,
    workdir: Path,
    runner: Callable[..., subprocess.CompletedProcess],
) -> dict[str, Any]:
    inputs = train_cell_inputs(fixture=fixture, arm=arm, host=host, hours=hours)
    key = cell_key(inputs)
    directory = cell_dir(workdir, inputs)
    existing = completed_cell(directory, key)
    if existing is not None:
        existing["resumed"] = True
        return existing
    preempted = preempt_partial_cell(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(directory / "inputs.json", inputs)

    env = dict(os.environ)
    env.update(
        {
            "PYTHONHASHSEED": str(arm.training_seed),
            "SEED": str(arm.training_seed),
            "AI_TOOLKIT_DIR": str(host.ai_toolkit_dir),
            # The fixture already carries the split. Forge's own experimental
            # holdout reservation must NOT take a second bite out of the train
            # set, or the arms stop training on the declared data.
            "FORGE_HOLDOUT_SELECTION_TYPES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    argv = [
        sys.executable,
        str(SCRIPT_PATH),
        "train-cell",
        "--fixture",
        str(fixture.path),
        "--arm",
        str(arm.path),
        "--host",
        str(host.path),
        "--hours",
        repr(hours),
        "--cell-dir",
        str(directory),
        "--cell-key",
        key,
    ]
    started = time.time()
    result = runner(argv, env=env)
    (directory / "train-cell.stdout.log").write_text(
        result.stdout or "", encoding="utf-8"
    )
    (directory / "train-cell.stderr.log").write_text(
        result.stderr or "", encoding="utf-8"
    )
    if result.returncode != 0:
        raise ExperimentError(
            f"training arm {arm.arm_id} failed (rc={result.returncode}); see "
            f"{directory}/train-cell.stderr.log"
        )
    receipt = completed_cell(directory, key)
    require(
        receipt is not None,
        f"training arm {arm.arm_id} exited 0 without publishing a receipt",
    )
    assert receipt is not None
    receipt["resumed"] = False
    receipt["preempted_partial_dir"] = (str(preempted) if preempted else None)
    receipt["driver_observed_wall_clock_s"] = round(time.time() - started, 3)
    return receipt


def stage_candidate(host: Host, candidate: Path, sha256: str) -> Path:
    """Place the candidate where the pinned harness insists on finding it."""
    loras = Path(host.comfy_root) / "models" / "loras"
    require(loras.is_dir(), f"ComfyUI loras dir is missing: {loras}")
    require(
        Path(candidate).is_file(),
        f"candidate to score is missing: {candidate} (a checkpoint recorded by "
        "a completed training cell was deleted or moved)",
    )
    staged = loras / f"candidate-{sha256}.safetensors"
    if staged.exists():
        actual = sha256_file(staged)
        require(
            actual == sha256,
            f"staged candidate {staged} has sha256 {actual}, expected {sha256}",
        )
        return staged
    temp = loras / f".candidate-{sha256}.tmp"
    if temp.exists():
        temp.unlink()
    shutil.copyfile(candidate, temp)
    actual = sha256_file(temp)
    if actual != sha256:
        temp.unlink()
        raise ExperimentError(
            f"staged copy of {candidate} hashed {actual}, expected {sha256}"
        )
    os.link(temp, staged)
    temp.unlink()
    return staged


def run_score_cell(
    *,
    fixture: Fixture,
    arm: Arm,
    host: Host,
    candidate_path: Path,
    candidate_sha256: str,
    candidate_name: str,
    step: int | None,
    cost: CostModel,
    workdir: Path,
    runner: Callable[..., subprocess.CompletedProcess],
    on_unloaded_keys: str = "fail",
) -> dict[str, Any]:
    inputs = score_cell_inputs(
        fixture=fixture,
        arm=arm,
        host=host,
        candidate_sha256=candidate_sha256,
        candidate_name=candidate_name,
        step=step,
        generations_planned=cost.eval_generations,
    )
    key = cell_key(inputs)
    directory = cell_dir(workdir, inputs)
    existing = completed_cell(directory, key)
    if existing is not None:
        existing["resumed"] = True
        _enforce_attachment(existing, on_unloaded_keys)
        return existing
    preempt_partial_cell(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(directory / "inputs.json", inputs)

    staged = stage_candidate(host, Path(candidate_path), candidate_sha256)
    score_path = directory / "score.json"
    comfy_log = directory / "comfy.log"
    argv = [
        str(host.comfy_python),
        str(EVALUATOR_PATH),
        "--dataset",
        str(fixture.holdout_dir),
        "--candidate-path",
        str(candidate_path),
        "--comfy-root",
        str(host.comfy_root),
        "--comfy-python",
        str(host.comfy_python),
        "--god-root",
        str(host.god_root),
        "--output",
        str(score_path),
        "--comfy-log",
        str(comfy_log),
        "--base-name",
        host.base_name,
        "--port",
        str(host.comfy_port),
        "--expected-god-commit",
        host.god_commit,
        "--expected-comfy-commit",
        host.comfy_commit,
        "--expected-tooling-commit",
        host.tooling_commit,
    ]
    started = time.time()
    result = runner(argv)
    elapsed = round(time.time() - started, 3)
    (directory / "evaluator.stdout.log").write_text(
        result.stdout or "", encoding="utf-8"
    )
    (directory / "evaluator.stderr.log").write_text(
        result.stderr or "", encoding="utf-8"
    )
    if result.returncode != 0:
        raise ExperimentError(
            f"exact scorer failed for {arm.arm_id}/{candidate_name} "
            f"(rc={result.returncode}); see {directory}/evaluator.stderr.log"
        )
    require(score_path.is_file(), f"scorer exited 0 without {score_path}")
    score = read_json(score_path)
    log_text = comfy_log.read_text(encoding="utf-8", errors="replace") if comfy_log.is_file() else ""
    require(log_text.strip(), f"pinned ComfyUI log is empty: {comfy_log}")
    unloaded = parse_unloaded_keys(log_text)

    rows = score.get("scored_rows")
    require(
        isinstance(rows, list) and len(rows) == fixture.holdout_pairs,
        "scorer returned an unexpected number of scored rows",
    )
    per_image: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        prompted = float(row["text_guided_loss"])
        blank = float(row["blank_prompt_loss"])
        per_image.append(
            {
                "index": index,
                "image": row["image"],
                "image_sha256": row["image_sha256"],
                "prompted_loss": prompted,
                "blank_loss": blank,
                "composite_loss": composite(prompted, blank),
            }
        )
    expected_images = [row.image for row in fixture.holdout_rows]
    require(
        sorted(item["image"] for item in per_image) == sorted(expected_images),
        "scored images do not match the fixture holdout manifest",
    )

    receipt = {
        "schema": SCHEMA,
        "kind": KIND + "-score-cell",
        "complete": True,
        "cell_key": key,
        "cell_kind": "score",
        "inputs": inputs,
        "arm_id": arm.arm_id,
        "arm_role": arm.role,
        "candidate": {
            "name": candidate_name,
            "path": str(candidate_path),
            "sha256": candidate_sha256,
            "step": step,
            "staged_as": str(staged),
        },
        "score_path": str(score_path),
        "score_sha256": sha256_file(score_path),
        "comfy_log": str(comfy_log),
        "comfy_log_sha256": sha256_file(comfy_log),
        "unloaded_lora_keys": unloaded,
        "attachment_ok": bool(unloaded["clean"]),
        "generations_observed": score.get("generations"),
        "generations_planned": cost.eval_generations,
        "generations_match_plan": score.get("generations") == cost.eval_generations,
        "seeds": score.get("seeds"),
        "text_weight_observed": score.get("text_weight"),
        "means": {
            "prompted": float(score["text_mean"]),
            "blank": float(score["blank_mean"]),
            "composite_from_scorer": float(score["weighted_loss"]),
            "composite_recomputed": composite(
                float(score["text_mean"]), float(score["blank_mean"])
            ),
        },
        "per_image": per_image,
        "measured": {"wall_clock_s": elapsed},
    }
    write_json_exclusive(directory / RESULT_NAME, receipt)
    _enforce_attachment(receipt, on_unloaded_keys)
    receipt["resumed"] = False
    return receipt


def _enforce_attachment(receipt: dict[str, Any], policy: str) -> None:
    unloaded = receipt.get("unloaded_lora_keys") or {}
    if unloaded.get("clean", True):
        return
    message = (
        f"UNLOADED LORA KEYS in arm {receipt.get('arm_id')!r} candidate "
        f"{receipt.get('candidate', {}).get('name')!r}: "
        f"{unloaded.get('unique_keys')} unique keys "
        f"({unloaded.get('by_namespace')}) were never loaded by the pinned "
        f"ComfyUI. Evidence is at {receipt.get('comfy_log')}."
    )
    if policy == "fail":
        raise UnloadedLoraKeys(message + " Re-run with --on-unloaded-keys flag to "
                               "record and continue (the arm is then barred from "
                               "winning by the predeclared rule).")
    print(f"[run_two_arm] WARNING: {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def _align(rows_a: Sequence[dict[str, Any]], rows_b: Sequence[dict[str, Any]]) -> None:
    keys_a = [row["image_sha256"] for row in rows_a]
    keys_b = [row["image_sha256"] for row in rows_b]
    require(
        sorted(keys_a) == sorted(keys_b),
        "cannot pair per-image results: the two cells scored different images",
    )


def _by_image(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["image_sha256"]: row for row in rows}


def paired_deltas(
    better: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]],
    field: str,
) -> list[float]:
    """baseline - better, per image, positive meaning `better` has lower loss."""
    _align(better, baseline)
    left = _by_image(better)
    right = _by_image(baseline)
    order = sorted(left)
    return [float(right[key][field]) - float(left[key][field]) for key in order]


def summarize_cell(cell: dict[str, Any]) -> dict[str, Any]:
    rows = cell["per_image"]
    return {
        "arm_id": cell["arm_id"],
        "candidate": cell["candidate"],
        "cell_key": cell["cell_key"],
        "prompted_mean": mean([row["prompted_loss"] for row in rows]),
        "blank_mean": mean([row["blank_loss"] for row in rows]),
        "composite_mean": mean([row["composite_loss"] for row in rows]),
        "attachment_ok": bool(cell.get("attachment_ok", False)),
        "unloaded_lora_keys": cell.get("unloaded_lora_keys"),
        "images": len(rows),
        "per_image": rows,
    }


def select_best(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    require(summaries, "no scored checkpoints for this arm")
    return sorted(
        summaries,
        key=lambda item: (
            item["composite_mean"],
            item["blank_mean"],
            item["candidate"].get("step") if item["candidate"].get("step") is not None else 1 << 60,
            item["candidate"]["sha256"],
        ),
    )[0]


def analyze(
    *,
    plan: dict[str, Any],
    score_cells: Sequence[dict[str, Any]],
    train_receipts: Sequence[dict[str, Any]] = (),
    incomplete: Sequence[str] = (),
) -> dict[str, Any]:
    rule = DECISION_RULE
    boot = rule["uncertainty"]
    seed = int(boot["seed"])
    iterations = int(boot["iterations"])
    level = float(boot["interval"])

    by_arm: dict[str, list[dict[str, Any]]] = {}
    for cell in score_cells:
        by_arm.setdefault(cell["arm_id"], []).append(summarize_cell(cell))

    control_ids = [arm["arm_id"] for arm in plan["arms"] if arm["role"] != "train"]
    train_ids = [arm["arm_id"] for arm in plan["arms"] if arm["role"] == "train"]
    require(len(control_ids) == 1, "exactly one control arm expected")
    control_id = control_ids[0]

    problems: list[str] = list(incomplete)
    stray = sorted(set(by_arm) - set(train_ids) - {control_id})
    for arm_id in stray:
        problems.append(
            f"scored cell for unknown arm {arm_id!r} is present in the workdir "
            "and was NOT used; the workdir may be mixing experiments"
        )
    control_rows: list[dict[str, Any]] | None = None
    if by_arm.get(control_id):
        control_summary = select_best(by_arm[control_id])
        control_rows = control_summary["per_image"]
    else:
        control_summary = None
        problems.append(f"zero-LoRA control {control_id!r} has no score")

    arms_out: dict[str, Any] = {}
    best_by_arm: dict[str, dict[str, Any]] = {}
    for arm_id in train_ids + [control_id]:
        summaries = sorted(
            by_arm.get(arm_id, []),
            key=lambda item: (
                item["candidate"].get("step") if item["candidate"].get("step") is not None else -1,
                item["candidate"]["sha256"],
            ),
        )
        if not summaries:
            problems.append(f"arm {arm_id!r} has no scored checkpoint")
            arms_out[arm_id] = {
                "role": ("control" if arm_id == control_id else "train"),
                "checkpoints": [],
                "best": None,
                "flagged_unloaded_keys": False,
                "missing": True,
            }
            continue
        checkpoints_out = []
        for summary in summaries:
            entry = {
                "candidate": summary["candidate"],
                "cell_key": summary["cell_key"],
                "prompted_mean": summary["prompted_mean"],
                "blank_mean": summary["blank_mean"],
                "composite_mean": summary["composite_mean"],
                "attachment_ok": summary["attachment_ok"],
                "unloaded_lora_keys": summary["unloaded_lora_keys"],
                "per_image": summary["per_image"],
            }
            if control_rows is not None and arm_id != control_id:
                entry["vs_zero_lora"] = {
                    field: bootstrap_mean_ci(
                        paired_deltas(summary["per_image"], control_rows, f"{field}_loss"),
                        seed=seed,
                        iterations=iterations,
                        level=level,
                    )
                    for field in ("blank", "prompted", "composite")
                }
                entry["vs_zero_lora"]["relative_blank_mean_pct"] = _relative_pct(
                    summary["per_image"], control_rows, "blank_loss"
                )
            checkpoints_out.append(entry)
        best = select_best(summaries)
        best_by_arm[arm_id] = best
        flagged = any(not item["attachment_ok"] for item in summaries)
        if flagged:
            problems.append(
                f"arm {arm_id!r} reported unloaded LoRA keys; it is barred from "
                "winning by the predeclared rule"
            )
        arms_out[arm_id] = {
            "role": ("control" if arm_id == control_id else "train"),
            "checkpoints": checkpoints_out,
            "best": {
                "candidate": best["candidate"],
                "cell_key": best["cell_key"],
                "prompted_mean": best["prompted_mean"],
                "blank_mean": best["blank_mean"],
                "composite_mean": best["composite_mean"],
            },
            "flagged_unloaded_keys": flagged,
            "missing": False,
        }

    head_to_head: dict[str, Any] = {}
    winner: str | None = None
    winner_interval: dict[str, Any] | None = None
    reasons: list[str] = []
    if len(train_ids) == 2 and all(arm_id in best_by_arm for arm_id in train_ids):
        first, second = train_ids
        for arm_id, other_id in ((first, second), (second, first)):
            deltas = paired_deltas(
                best_by_arm[arm_id]["per_image"],
                best_by_arm[other_id]["per_image"],
                "blank_loss",
            )
            interval = bootstrap_mean_ci(
                deltas, seed=seed, iterations=iterations, level=level
            )
            head_to_head[arm_id] = {
                "versus": other_id,
                "statistic": "mean per-image blank-loss advantage",
                "per_image_deltas": deltas,
                "interval": interval,
                "eligible": not arms_out[arm_id]["flagged_unloaded_keys"],
            }
        for arm_id in train_ids:
            entry = head_to_head[arm_id]
            interval = entry["interval"]
            if not interval.get("defined"):
                reasons.append(
                    f"{arm_id}: {interval.get('reason')}"
                )
                continue
            if not entry["eligible"]:
                reasons.append(
                    f"{arm_id}: disqualified (unloaded LoRA keys), interval not applied"
                )
                continue
            if interval["lo"] > 0.0:
                if winner is not None:
                    raise ExperimentError(
                        "both arms cleared the win condition, which is "
                        "arithmetically impossible for symmetric paired deltas; "
                        "the inputs are inconsistent"
                    )
                winner = arm_id
                winner_interval = interval
            else:
                reasons.append(
                    f"{arm_id}: bootstrap 95% lower bound "
                    f"{interval['lo']:.6g} is not above zero"
                )
    else:
        reasons.append("head-to-head not evaluated: an arm has no scored checkpoint")

    if winner is not None and problems:
        # A winner is only allowed when nothing is missing or flagged.
        reasons.append(
            "winner withdrawn: the run has unresolved problems "
            f"({len(problems)}); see 'problems'"
        )
        winner, winner_interval = None, None

    statement = (
        f"WINNER: {winner} — its best checkpoint's blank-prompt advantage over "
        f"the other arm's best checkpoint has a bootstrap 95% interval of "
        f"[{winner_interval['lo']:.6g}, {winner_interval['hi']:.6g}] "
        f"(mean {winner_interval['mean']:.6g}), lower bound above zero."
        if winner is not None and winner_interval is not None
        else (
            "NO WINNER (null result). No arm's blank-prompt advantage over the "
            "other arm has a bootstrap 95% lower bound above zero, so under the "
            "predeclared rule nothing replaces the incumbent."
        )
    )

    return {
        "schema": SCHEMA,
        "kind": KIND + "-analysis",
        "decision_rule": rule,
        "predeclaration_bound_to": predeclaration(plan)["bound_to"],
        "fixture": plan["fixture"],
        "host": plan["host"],
        "cost_model": plan["cost_model"],
        "estimate": plan["estimate"],
        "train_receipts": [
            {
                "arm_id": receipt.get("arm_id"),
                "cell_key": receipt.get("cell_key"),
                "measured": receipt.get("measured"),
                "config_sha256": receipt.get("config", {}).get("resolved_file_sha256"),
                "dataset_sha256": receipt.get("dataset", {}).get(
                    "train_archive_sha256"
                ),
                "steps_reached": receipt.get("measured", {}).get("steps_reached"),
                "checkpoints": receipt.get("checkpoints"),
            }
            for receipt in train_receipts
        ],
        "control_arm_id": control_id,
        "arms": arms_out,
        "head_to_head": head_to_head,
        "winner": winner,
        "winner_interval": winner_interval,
        "null_result": winner is None,
        "winner_statement": statement,
        "reasons": reasons,
        "problems": problems,
        "complete": not problems,
    }


def _relative_pct(
    rows: Sequence[dict[str, Any]], control_rows: Sequence[dict[str, Any]], field: str
) -> float | None:
    _align(rows, control_rows)
    left = _by_image(rows)
    right = _by_image(control_rows)
    values = []
    for key in sorted(left):
        base = float(right[key][field])
        if base <= 0.0:
            return None
        values.append((base - float(left[key][field])) / base * 100.0)
    return mean(values)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def execute(
    *,
    fixture: Fixture,
    arms: Sequence[Arm],
    host: Host,
    cost: CostModel,
    hours: float | None,
    workdir: Path,
    runner: Callable[..., subprocess.CompletedProcess],
    on_unloaded_keys: str = "fail",
    hostname: str | None = None,
) -> dict[str, Any]:
    plan = build_plan(
        fixture=fixture,
        arms=arms,
        host=host,
        cost=cost,
        hours=hours,
        workdir=workdir,
    )
    establish_predeclaration(workdir, plan)
    plan_path = publish_plan(workdir, plan)
    flight = preflight(
        fixture=fixture, arms=arms, host=host, runner=runner, hostname=hostname,
    )
    effective_hours = plan["hours"]["effective"]

    train_receipts: list[dict[str, Any]] = []
    score_cells: list[dict[str, Any]] = []
    for arm in arms:
        if not arm.trains:
            continue
        receipt = run_train_cell(
            fixture=fixture,
            arm=arm,
            host=host,
            hours=effective_hours,
            workdir=workdir,
            runner=runner,
        )
        train_receipts.append(receipt)
        for checkpoint in receipt["checkpoints"]:
            score_cells.append(
                run_score_cell(
                    fixture=fixture,
                    arm=arm,
                    host=host,
                    candidate_path=Path(checkpoint["path"]),
                    candidate_sha256=checkpoint["sha256"],
                    candidate_name=checkpoint["name"],
                    step=checkpoint.get("step"),
                    cost=cost,
                    workdir=workdir,
                    runner=runner,
                    on_unloaded_keys=on_unloaded_keys,
                )
            )
    for arm in arms:
        if arm.trains:
            continue
        _verify_zero_lora(arm)
        score_cells.append(
            run_score_cell(
                fixture=fixture,
                arm=arm,
                host=host,
                candidate_path=Path(str(arm.candidate_path)),
                candidate_sha256=str(arm.candidate_sha256),
                candidate_name="zero-lora-control",
                step=0,
                cost=cost,
                workdir=workdir,
                runner=runner,
                on_unloaded_keys=on_unloaded_keys,
            )
        )

    analysis = analyze(
        plan=plan, score_cells=score_cells, train_receipts=train_receipts
    )
    analysis["preflight"] = flight
    analysis["plan_path"] = str(plan_path)
    digest = canonical_hash(analysis)
    path = Path(workdir) / "analysis" / f"analysis-{digest[:12]}.json"
    write_json_exclusive(path, analysis)
    analysis["analysis_path"] = str(path)
    return analysis


def _verify_zero_lora(arm: Arm) -> None:
    """The control must be a proven all-zero artifact, not just a file."""
    require(
        arm.candidate_path is not None and Path(arm.candidate_path).is_file(),
        f"zero-LoRA candidate is missing: {arm.candidate_path}",
    )
    actual = sha256_file(Path(str(arm.candidate_path)))
    require(
        actual == arm.candidate_sha256,
        f"zero-LoRA candidate sha256 mismatch: declared {arm.candidate_sha256}, "
        f"actual {actual}",
    )
    record = read_json(Path(str(arm.zero_lora_record_path)))
    require(
        record.get("kind") == "zero_lora_safetensors_baseline",
        "zero-LoRA record is not a zero_lora_safetensors_baseline record",
    )
    output = record.get("output", {})
    verification = record.get("verification", {})
    require(
        output.get("sha256") == actual,
        "zero-LoRA record does not describe this artifact",
    )
    require(
        output.get("all_zero") is True
        and verification.get("output_all_zero") is True
        and verification.get("output_all_finite") is True,
        "zero-LoRA record does not prove a finite all-zero artifact",
    )


# ---------------------------------------------------------------------------
# train-cell (runs on the GPU host, inside the seeded child process)
# ---------------------------------------------------------------------------


def _ensure_seeded_process(seed: int) -> None:
    """Re-exec before Forge imports so PYTHONHASHSEED covers this process."""
    expected = str(seed)
    if os.environ.get("PYTHONHASHSEED") == expected:
        os.environ["SEED"] = expected
        os.environ[_REEXEC_MARKER] = expected
        return
    if os.environ.get(_REEXEC_MARKER):
        raise ExperimentError("seeded re-exec did not install the requested hash seed")
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = expected
    env["SEED"] = expected
    env[_REEXEC_MARKER] = expected
    os.execve(sys.executable, [sys.executable, str(SCRIPT_PATH), *sys.argv[1:]], env)


def _load_ladder():
    """Reuse the calibration helpers rather than re-deriving them here."""
    spec = importlib.util.spec_from_file_location("run_krea_ladder", LADDER_PATH)
    require(spec is not None and spec.loader is not None, "cannot load ladder helpers")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def bind_config(
    config: dict[str, Any],
    *,
    spec: Any,
    trigger_word: str | None,
    training_seed: int,
) -> dict[str, Any]:
    """Rebind the five environment-dependent pointers.  Nothing else moves."""
    resolved = copy.deepcopy(config)
    process = resolved["config"]["process"][0]
    resolved["config"]["name"] = spec.expected_repo_name
    process["training_folder"] = spec.training_folder
    if trigger_word is not None:
        process["trigger_word"] = trigger_word
    process["datasets"][0]["folder_path"] = spec.dataset_images_dir
    process["model"]["name_or_path"] = spec.cached_model_dir
    kwargs = process["model"].setdefault("model_kwargs", {})
    if "vae_path" in kwargs:
        kwargs["vae_path"] = spec.cached_model_dir
    process["training_seed"] = training_seed
    return resolved


def train_cell_main(args: argparse.Namespace) -> int:
    fixture = load_fixture(Path(args.fixture))
    arm = load_arm(Path(args.arm))
    host = load_host(Path(args.host))
    require(arm.trains, "train-cell requires a training arm")
    assert arm.training_seed is not None
    _ensure_seeded_process(arm.training_seed)

    hours = float(args.hours)
    inputs = train_cell_inputs(fixture=fixture, arm=arm, host=host, hours=hours)
    key = cell_key(inputs)
    require(
        key == args.cell_key,
        f"train-cell key mismatch: parent {args.cell_key}, child {key}",
    )
    directory = Path(args.cell_dir)
    require(directory.is_dir(), f"cell directory does not exist: {directory}")

    ladder = _load_ladder()

    # Heavy imports only after the seeded re-exec, exactly as the ladder does.
    import yaml

    from forge import telemetry as forge_telemetry
    from forge.cli import main as forge_main
    from forge.data.schema import ImageSpec
    from forge.tasks import aitoolkit, checkpoints
    from forge.tasks.integrity import valid_safetensors

    spec = ImageSpec.build(
        task_id=fixture.task_id,
        model=fixture.model,
        model_type=fixture.model_type,
        expected_repo_name=str(arm.expected_repo_name),
        trigger_word=fixture.trigger_word,
        dataset_zip=None,
    )
    mutable = [
        Path(spec.config_path),
        Path(spec.training_folder),
        Path(spec.save_root),
        Path(spec.dataset_holdout_dir),
        Path(spec.dataset_images_dir),
        Path(spec.dataset_images_dir + "__extract"),
        Path(spec.dataset_images_dir + "__flat"),
    ]
    ladder._require_clean_paths(mutable)

    staged_zip = Path(spec.cached_zip_path)
    staged_zip.parent.mkdir(parents=True, exist_ok=True)
    archive = Path(str(fixture.train_archive))
    if staged_zip.exists():
        actual = sha256_file(staged_zip)
        require(
            actual == fixture.train_archive_sha256,
            f"staged dataset {staged_zip} is sha256 {actual}, fixture declares "
            f"{fixture.train_archive_sha256}",
        )
    else:
        shutil.copyfile(archive, staged_zip)
        actual = sha256_file(staged_zip)
        require(
            actual == fixture.train_archive_sha256,
            f"staged dataset copy hashed {actual}, expected "
            f"{fixture.train_archive_sha256}",
        )

    frozen = read_yaml(Path(str(arm.config_path)))
    captured: dict[str, Any] = {}
    original_build_config = aitoolkit.build_config

    def _arm_config(spec_arg, num_images, hours_to_complete):
        resolved = bind_config(
            frozen,
            spec=spec_arg,
            trigger_word=fixture.trigger_word,
            training_seed=int(str(arm.training_seed)),
        )
        changed = ladder._diff_paths(frozen, resolved)
        unexpected = changed - set(_BINDING_POINTERS)
        if unexpected:
            raise ExperimentError(
                f"arm binding touched non-binding config fields: {sorted(unexpected)}"
            )
        captured.update(
            frozen=copy.deepcopy(frozen),
            resolved=copy.deepcopy(resolved),
            bindings=sorted(changed),
            num_images=int(num_images),
            derived_hours=float(hours_to_complete),
        )
        return resolved

    pre_runtime = ladder._runtime_fingerprint()
    aitoolkit.build_config = _arm_config
    started_unix = time.time()
    started = time.monotonic()
    try:
        return_code = forge_main(
            [
                "--task-id",
                fixture.task_id,
                "--model",
                fixture.model,
                "--model-type",
                fixture.model_type,
                "--expected-repo-name",
                str(arm.expected_repo_name),
                "--hours-to-complete",
                repr(hours),
            ]
        )
    finally:
        aitoolkit.build_config = original_build_config
    elapsed = time.monotonic() - started
    finished_unix = time.time()
    require(return_code == 0, f"Forge returned nonzero status {return_code}")
    require(captured, "Forge never resolved the arm config")

    persisted = yaml.safe_load(Path(spec.config_path).read_text(encoding="utf-8"))
    require(
        persisted == captured["resolved"],
        "persisted YAML differs from the resolved arm config",
    )
    scope = checkpoints.load_run(spec.save_root)
    require(scope is not None, "checkpoint scope is not active in this process")
    candidates = [Path(path) for path in checkpoints.current_loras(spec.save_root, scope)]
    require(candidates, "training produced no current-attempt candidate")
    checkpoints_out: list[dict[str, Any]] = []
    seen_sha: dict[str, str] = {}
    for path in sorted(candidates):
        require(valid_safetensors(str(path)), f"invalid safetensors: {path}")
        digest = sha256_file(path)
        match = re.search(r"_(\d+)\.safetensors$", path.name)
        step = int(match.group(1)) if match else int(str(arm.steps))
        entry = {
            "name": path.name,
            "path": str(path),
            "sha256": digest,
            "bytes": path.stat().st_size,
            "step": step,
            "duplicate_of": seen_sha.get(digest),
        }
        seen_sha.setdefault(digest, path.name)
        checkpoints_out.append(entry)
    unique = [item for item in checkpoints_out if item["duplicate_of"] is None]

    last_path = Path(spec.save_root) / "last.safetensors"
    last_sha = sha256_file(last_path) if last_path.is_file() else None
    selection = checkpoints.current_selection_record(spec.save_root, scope)

    steps_reached: int | None = None
    telemetry_record: dict[str, Any] | None = None
    try:
        telemetry_path = Path(forge_telemetry.private_record_path(spec.save_root))
        if telemetry_path.is_file():
            telemetry_record = read_json(telemetry_path)
            for event in telemetry_record.get("events", []):
                if isinstance(event, dict) and event.get("name") == "toolkit_metrics":
                    value = event.get("last_step")
                    if isinstance(value, int) and not isinstance(value, bool):
                        steps_reached = value
    except Exception:
        telemetry_record = None
    if steps_reached is None:
        steps_reached = max(item["step"] for item in checkpoints_out)

    post_runtime = ladder._runtime_fingerprint()
    receipt = {
        "schema": SCHEMA,
        "kind": KIND + "-train-cell",
        "complete": True,
        "cell_key": key,
        "cell_kind": "train",
        "inputs": inputs,
        "arm_id": arm.arm_id,
        "arm_role": arm.role,
        "expected_repo_name": arm.expected_repo_name,
        "config": {
            "frozen_path": str(arm.config_path),
            "frozen_sha256": arm.config_sha256,
            "resolved_canonical_sha256": canonical_hash(captured["resolved"]),
            "resolved_file_path": spec.config_path,
            "resolved_file_sha256": sha256_file(Path(spec.config_path)),
            "bindings_applied": captured["bindings"],
            "planned_steps": arm.steps,
            "save_every": arm.save_every,
        },
        "dataset": {
            "train_archive_path": str(archive),
            "train_archive_sha256": fixture.train_archive_sha256,
            "staged_zip": str(staged_zip),
            "train_pairs_declared": fixture.train_pairs,
            "num_images_seen_by_recipe": captured["num_images"],
            "images_dir": ladder._fingerprint_path(Path(spec.dataset_images_dir)),
            "holdout_reservation_disabled": True,
        },
        "runtime": {
            "host_id": host.host_id,
            "hostname": socket.gethostname(),
            "ai_toolkit_dir": str(host.ai_toolkit_dir),
            "ai_toolkit_commit_declared": arm.ai_toolkit_commit,
            "pre": pre_runtime,
            "post": post_runtime,
            "unchanged": pre_runtime == post_runtime,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "training_seed": arm.training_seed,
        },
        "checkpoints": unique,
        "checkpoints_all": checkpoints_out,
        "last_safetensors_sha256": last_sha,
        "selection_record": selection,
        "measured": {
            "wall_clock_s": round(elapsed, 3),
            "started_unix": round(started_unix, 3),
            "finished_unix": round(finished_unix, 3),
            "steps_reached": steps_reached,
            "seconds_per_step": (
                round(elapsed / steps_reached, 6) if steps_reached else None
            ),
            "reached_planned_depth": steps_reached == arm.steps,
        },
        "telemetry": telemetry_record,
    }
    write_json_exclusive(directory / RESULT_NAME, receipt)
    print(
        json.dumps(
            {
                "arm_id": arm.arm_id,
                "cell_key": key,
                "steps_reached": steps_reached,
                "checkpoints": len(unique),
                "wall_clock_s": receipt["measured"]["wall_clock_s"],
            },
            sort_keys=True,
        )
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--arm", action="append", default=[], required=True)
    parser.add_argument("--host", "--gpu-host", dest="host", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--sec-per-step", type=float, default=None)
    parser.add_argument("--eval-generations", type=int, default=None)
    parser.add_argument("--eval-seconds-per-generation", type=float, default=None)
    parser.add_argument("--save-overhead-s", type=float, default=None)
    parser.add_argument("--gpu-hourly-usd", type=float, default=None)
    parser.add_argument("--json", action="store_true")


def _cost_from(args: argparse.Namespace, host: Host) -> CostModel:
    defaults = CostModel()
    return CostModel(
        sec_per_step=(
            args.sec_per_step if args.sec_per_step is not None else defaults.sec_per_step
        ),
        startup_s=defaults.startup_s,
        export_reserve_s=defaults.export_reserve_s,
        save_overhead_s=(
            args.save_overhead_s
            if args.save_overhead_s is not None
            else defaults.save_overhead_s
        ),
        eval_fixed_overhead_s=defaults.eval_fixed_overhead_s,
        eval_generations=(
            args.eval_generations
            if args.eval_generations is not None
            else defaults.eval_generations
        ),
        eval_seconds_per_generation=(
            args.eval_seconds_per_generation
            if args.eval_seconds_per_generation is not None
            else defaults.eval_seconds_per_generation
        ),
        gpu_hourly_usd=(
            args.gpu_hourly_usd
            if args.gpu_hourly_usd is not None
            else host.gpu_hourly_usd
        ),
    )


def _load_all(args: argparse.Namespace) -> tuple[Fixture, list[Arm], Host, CostModel]:
    fixture = load_fixture(Path(args.fixture))
    arms = [load_arm(Path(path)) for path in args.arm]
    host = load_host(Path(args.host))
    return fixture, arms, host, _cost_from(args, host)


def plan_main(args: argparse.Namespace) -> int:
    fixture, arms, host, cost = _load_all(args)
    workdir = Path(os.path.abspath(os.path.expanduser(args.workdir)))
    plan = build_plan(
        fixture=fixture, arms=arms, host=host, cost=cost, hours=args.hours,
        workdir=workdir,
    )
    environment = environment_report(
        plan=plan, fixture=fixture, arms=arms, host=host, workdir=workdir
    )
    if args.strict_paths:
        require(
            not environment["environment_warnings"],
            f"strict path check failed: {environment['environment_warnings']}",
        )
    established = establish_predeclaration(workdir, plan)
    plan_path = publish_plan(workdir, plan)
    output = dict(plan)
    output["plan_path"] = str(plan_path)
    output["environment"] = environment
    output["predeclaration"] = established
    output["gpu_used"] = False
    if args.json:
        print(json.dumps(output, allow_nan=False, indent=2, sort_keys=True))
    else:
        print(format_estimate(plan, environment))
        print("")
        print(f"predeclaration: {established['path']} (sha256 {established['sha256']})")
        print(f"plan          : {plan_path}")
        print("GPU used      : none (dry run)")
    return 0


def run_main(args: argparse.Namespace) -> int:
    if args.dry_run:
        return plan_main(args)
    fixture, arms, host, cost = _load_all(args)
    workdir = Path(os.path.abspath(os.path.expanduser(args.workdir)))
    analysis = execute(
        fixture=fixture,
        arms=arms,
        host=host,
        cost=cost,
        hours=args.hours,
        workdir=workdir,
        runner=default_runner,
        on_unloaded_keys=args.on_unloaded_keys,
    )
    print(json.dumps(analysis, allow_nan=False, indent=2, sort_keys=True))
    return 0


def analyze_main(args: argparse.Namespace) -> int:
    fixture, arms, host, cost = _load_all(args)
    workdir = Path(os.path.abspath(os.path.expanduser(args.workdir)))
    plan = build_plan(
        fixture=fixture, arms=arms, host=host, cost=cost, hours=args.hours,
        workdir=workdir,
    )
    establish_predeclaration(workdir, plan)
    score_cells: list[dict[str, Any]] = []
    incomplete: list[str] = []
    score_root = workdir / "cells" / "score"
    if score_root.is_dir():
        for directory in sorted(score_root.iterdir()):
            result = directory / RESULT_NAME
            if not result.is_file():
                incomplete.append(f"incomplete score cell: {directory}")
                continue
            score_cells.append(read_json(result))
    train_receipts: list[dict[str, Any]] = []
    train_root = workdir / "cells" / "train"
    if train_root.is_dir():
        for directory in sorted(train_root.iterdir()):
            result = directory / RESULT_NAME
            if not result.is_file():
                incomplete.append(f"incomplete train cell: {directory}")
                continue
            train_receipts.append(read_json(result))
    analysis = analyze(
        plan=plan,
        score_cells=score_cells,
        train_receipts=train_receipts,
        incomplete=incomplete,
    )
    digest = canonical_hash(analysis)
    path = workdir / "analysis" / f"analysis-{digest[:12]}.json"
    write_json_exclusive(path, analysis)
    analysis["analysis_path"] = str(path)
    print(json.dumps(analysis, allow_nan=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Week-6 two-arm Krea2 experiment runner (plus zero-LoRA control)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan", help="dry run: validate and cost, zero GPU")
    _add_common(plan_parser)
    plan_parser.add_argument("--strict-paths", action="store_true")
    plan_parser.set_defaults(handler=plan_main)

    run_parser = sub.add_parser("run", help="execute the experiment on the GPU host")
    _add_common(run_parser)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--strict-paths", action="store_true")
    run_parser.add_argument(
        "--on-unloaded-keys", choices=("fail", "flag"), default="fail"
    )
    run_parser.set_defaults(handler=run_main)

    analyze_parser = sub.add_parser(
        "analyze", help="re-derive the analysis from cells already on disk"
    )
    _add_common(analyze_parser)
    analyze_parser.add_argument("--strict-paths", action="store_true")
    analyze_parser.set_defaults(handler=analyze_main)

    train_parser = sub.add_parser(
        "train-cell", help="internal: one seeded training arm (invoked by 'run')"
    )
    train_parser.add_argument("--fixture", required=True)
    train_parser.add_argument("--arm", required=True)
    train_parser.add_argument("--host", required=True)
    train_parser.add_argument("--hours", required=True)
    train_parser.add_argument("--cell-dir", required=True)
    train_parser.add_argument("--cell-key", required=True)
    train_parser.set_defaults(handler=train_cell_main)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except ExperimentError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
