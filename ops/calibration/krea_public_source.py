#!/usr/bin/env python3
"""Build one unreviewed Krea public-arm provenance record from raw evidence.

This adapter is deliberately narrow: it accepts only the frozen Week-5 K2-K4
source arms, reparses their immutable ai-toolkit YAML, reads the submitted step
from the safetensors header, and then delegates the full official-record
rebinding and canonical publication to :mod:`krea_provenance`.

The generated review assertion is always ``unreviewed``.  A human approval is
a separate artifact and cannot be manufactured by this command.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any
from urllib.parse import urlsplit

import yaml

try:
    from . import krea_provenance
except ImportError:  # pragma: no cover - direct script execution.
    import krea_provenance  # type: ignore[no-redef]


_ARMS = {
    "K2": {
        "official_rank": 2,
        "expected_revision": "f4766189afc0f0ce46b52ac2991efc5f005ebbfd",
        "expected_hotkey_prefix": "5C7yZ5wg",
    },
    "K3": {
        "official_rank": 3,
        "expected_revision": "919e07cd4505cf64c13a9baef4402f2b42a6fb59",
        "expected_hotkey_prefix": "5EeLcV3L",
    },
    "K4": {
        "official_rank": 5,
        "expected_revision": "71bf349eb44640289b00fc620640a1302cc3c485",
        "expected_hotkey_prefix": "5GKoYQm7",
    },
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RANK4_EXCLUSION_RATIONALE = (
    "Final rank 4 adds no fully disclosed distinct faithful arm: its scored "
    "last.safetensors is in the rank-32 AdamW8bit MSE LR-1e-4 family already "
    "covered by K0/K2, its only potentially orthogonal checkpoint soup was not "
    "submitted or scored, and its differential-guidance setting is unknown."
)
_LOCAL_REPRODUCTION = {
    "K2": {
        "depth_policy": "measured-budget-fill-with-step-960-landmark-if-budget-safe",
        "candidate_cadence_policy": (
            "discovery-uniform-1/8-with-real-write-accounting"
        ),
        "selection_policy": (
            "preserve and exact-score full curve; public holdout algorithm is not "
            "assumed recoverable"
        ),
        "source_unknown_fields": [],
        "predeclared_local_values": [],
    },
    "K3": {
        "depth_policy": "measured-budget-fill-with-step-1200-landmark-if-budget-safe",
        "candidate_cadence_policy": (
            "discovery-uniform-1/8-with-real-write-accounting"
        ),
        "selection_policy": (
            "preserve and exact-score every valid current-attempt candidate offline "
            "after training; the public highest-numbered fallback remains a source "
            "submission fact, not the local selector"
        ),
        "source_unknown_fields": ["dropout", "ema"],
        "predeclared_local_values": [
            {
                "field": "dropout",
                "value": 0.05,
                "basis": (
                    "Predeclared local K0/K1 control value from the draft discovery "
                    "plan; this is not evidence of K3's source dropout."
                ),
            },
            {
                "field": "ema",
                "value": False,
                "basis": (
                    "Predeclared local K0/K1 control value from the draft discovery "
                    "plan; this is not evidence of K3's source EMA setting."
                ),
            },
        ],
    },
    "K4": {
        "depth_policy": "measured-budget-fill-with-step-840-landmark-if-budget-safe",
        "candidate_cadence_policy": (
            "discovery-uniform-1/8-with-real-write-accounting"
        ),
        "selection_policy": (
            "preserve and exact-score full curve; public holdout algorithm is not "
            "assumed recoverable"
        ),
        "source_unknown_fields": [],
        "predeclared_local_values": [],
    },
}


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(_ARMS), required=True)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--source-artifact", required=True, type=Path)
    parser.add_argument("--field-ledger", required=True, type=Path)
    parser.add_argument("--task-raw", required=True, type=Path)
    parser.add_argument("--tournament-raw", required=True, type=Path)
    parser.add_argument("--revision-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _safe_file(path: Path, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a non-empty regular non-symlink file")
    return path


def _json_file(path: Path, label: str) -> dict[str, Any]:
    path = _safe_file(path, label)
    before = path.stat()
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    after = path.stat()
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
    )
    if identity(before) != identity(after) or not isinstance(value, dict):
        raise RuntimeError(f"{label} changed while read or is not an object")
    return value


def _config(path: Path) -> dict[str, Any]:
    path = _safe_file(path, "source config")
    before = path.stat()
    try:
        value = yaml.safe_load(path.read_bytes())
    except yaml.YAMLError as exc:
        raise ValueError("source config is not YAML") from exc
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("source config changed while read")
    if not isinstance(value, dict):
        raise ValueError("source config root is not an object")
    return value


def _artifact_training_info(path: Path) -> dict[str, Any]:
    """Read only the safetensors header through one no-follow descriptor."""

    path = _safe_file(path, "source artifact")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 10:
            raise ValueError(
                "source artifact is not a complete regular safetensors file"
            )
        length_raw = os.read(descriptor, 8)
        if len(length_raw) != 8:
            raise ValueError("source artifact lacks a safetensors header length")
        header_length = struct.unpack("<Q", length_raw)[0]
        if header_length <= 1 or 8 + header_length >= before.st_size:
            raise ValueError("source artifact has an invalid safetensors header length")
        remaining = header_length
        chunks: list[bytes] = []
        while remaining:
            block = os.read(descriptor, min(remaining, 8 * 1024 * 1024))
            if not block:
                raise ValueError("source artifact has a truncated safetensors header")
            chunks.append(block)
            remaining -= len(block)
        try:
            header = json.loads(b"".join(chunks))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("source artifact safetensors header is not JSON") from exc
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError("source artifact changed while its header was read")
    finally:
        os.close(descriptor)
    if not isinstance(header, dict) or not isinstance(header.get("__metadata__"), dict):
        raise ValueError("source artifact lacks safetensors metadata")
    raw = header["__metadata__"].get("training_info")
    if not isinstance(raw, str):
        raise ValueError("source artifact lacks string training_info metadata")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("source artifact training_info is not JSON") from exc
    if (
        not isinstance(value, dict)
        or isinstance(value.get("step"), bool)
        or not isinstance(value.get("step"), int)
        or value["step"] <= 0
    ):
        raise ValueError("source artifact training_info lacks a positive step")
    return value


def _one_process(config: dict[str, Any]) -> dict[str, Any]:
    root = config.get("config")
    rows = root.get("process") if isinstance(root, dict) else None
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("source config must contain exactly one training process")
    process = rows[0]
    if process.get("type") != "diffusion_trainer":
        raise ValueError("source config is not one diffusion trainer")
    model = process.get("model")
    if not isinstance(model, dict) or model.get("arch") != "krea2":
        raise ValueError("source config is not Krea2")
    return process


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _row(value: Any, pointer: str, evidence: str) -> dict[str, Any]:
    return {
        "classification": "known",
        "source_pointers": [pointer],
        "source_value": value,
        "effective_value": None,
        "evidence": evidence,
    }


def _unknown(evidence: str) -> dict[str, Any]:
    return {
        "classification": "unknown",
        "source_pointers": [],
        "source_value": None,
        "effective_value": None,
        "evidence": evidence,
    }


def _local_reproduction_disclosure(
    arm: str, *, normalized_recipe: dict[str, Any]
) -> dict[str, Any]:
    """Expose intended local adaptations without changing source observations."""

    policy = _LOCAL_REPRODUCTION[arm]
    source_fields = normalized_recipe["fields"]
    adapted = [
        {
            "name": "depth policy",
            "source_recipe_fields": ["planned_steps", "submitted_step"],
            "local_policy": policy["depth_policy"],
            "evidence": (
                "Source planned/submitted depth remains in normalized_recipe; the "
                "local run uses this separately predeclared budget-fill policy."
            ),
        },
        {
            "name": "offline exact-scoring selection policy",
            "source_recipe_fields": ["selector"],
            "local_policy": policy["selection_policy"],
            "evidence": (
                "The public selector remains a source observation; local checkpoint "
                "choice is deferred to the separately predeclared exact-score rule."
            ),
        },
        {
            "name": "save cadence and candidate grid",
            "source_recipe_fields": ["save_cadence"],
            "local_policy": policy["candidate_cadence_policy"],
            "evidence": (
                "Source save cadence remains in normalized_recipe; the local grid "
                "uses this separately predeclared cadence policy."
            ),
        },
    ]
    unknown_names = policy["source_unknown_fields"]
    if unknown_names:
        adapted.append(
            {
                "name": (
                    "source-unknown dropout and EMA fixed locally to the K0/K1 "
                    "controls"
                ),
                "source_recipe_fields": ["dropout", "ema"],
                "local_policy": (
                    "Use only the separately listed predeclared_local_values; these "
                    "are local controls, not recovered K3 source facts."
                ),
                "evidence": (
                    "The immutable K3 config omits both fields; source absence and "
                    "local choices are deliberately represented in separate arrays."
                ),
            }
        )
    adapted.sort(key=lambda row: row["name"])
    source_unknowns = [
        {
            "field": name,
            "source_classification": "unknown",
            "source_pointers": [],
            "source_value": None,
            "evidence": source_fields[name]["evidence"],
        }
        for name in sorted(unknown_names)
    ]
    return {
        "schema": 1,
        "kind": "forge-krea-local-reproduction-disclosure",
        "execution_authorized": False,
        "adapted_fields": adapted,
        "source_unknown_fields": source_unknowns,
        "predeclared_local_values": policy["predeclared_local_values"],
        "claim_limit": (
            "Machine-derived disclosure of intended local adaptations only; it is "
            "not human review, execution approval, or evidence that an adapted run "
            "reproduces the public score."
        ),
    }


def build_metadata(
    arm: str,
    *,
    source_config_path: Path,
    source_artifact_path: Path,
    field_ledger_path: Path,
) -> dict[str, Any]:
    """Re-derive one source recipe and its official identity."""

    if arm not in _ARMS:
        raise ValueError(f"unsupported public source arm: {arm}")
    spec = _ARMS[arm]
    ledger = _json_file(field_ledger_path, "field ledger")
    if (
        ledger.get("schema") != 1
        or ledger.get("kind") != "sn56-week5-krea-r1-public-field-ledger"
    ):
        raise ValueError("unsupported field ledger")
    if (
        ledger.get("discovery_arm_selection", {}).get("excluded_final_rank_4")
        != _RANK4_EXCLUSION_RATIONALE
    ):
        raise ValueError("field ledger lacks the frozen final-rank-4 exclusion basis")
    submissions = ledger.get("submissions")
    if not isinstance(submissions, list):
        raise ValueError("field ledger submissions are not an array")
    matches = [
        row
        for row in submissions
        if isinstance(row, dict) and row.get("official_rank") == spec["official_rank"]
    ]
    if len(matches) != 1:
        raise ValueError(f"{arm} does not identify exactly one official submission")
    submission = matches[0]
    revision = submission.get("repo_revision")
    hotkey = submission.get("hotkey")
    if (
        revision != spec["expected_revision"]
        or not isinstance(hotkey, str)
        or not hotkey.startswith(spec["expected_hotkey_prefix"])
    ):
        raise ValueError(f"{arm} official identity differs from the frozen screen")

    process = _one_process(_config(source_config_path))
    train = process.get("train")
    network = process.get("network")
    save = process.get("save")
    datasets = process.get("datasets")
    if (
        not isinstance(train, dict)
        or not isinstance(network, dict)
        or not isinstance(save, dict)
        or not isinstance(datasets, list)
        or len(datasets) != 1
        or not isinstance(datasets[0], dict)
    ):
        raise ValueError("source config lacks one train/network/save/dataset surface")
    dataset = datasets[0]
    submitted = _artifact_training_info(source_artifact_path)["step"]
    ledger_submitted = submission.get("submitted_step")
    if (
        not isinstance(ledger_submitted, dict)
        or ledger_submitted.get("state") != "known"
        or ledger_submitted.get("value") != submitted
    ):
        raise ValueError("artifact header step contradicts the field ledger")

    planned = _positive_int(train.get("steps"), "planned steps")
    if submitted > planned:
        raise ValueError("submitted step exceeds planned steps")
    batch_size = _positive_int(train.get("batch_size"), "batch size")
    accumulation = _positive_int(
        train.get("gradient_accumulation"), "gradient accumulation"
    )
    guidance_enabled = train.get("do_differential_guidance")
    guidance_scale = train.get("differential_guidance_scale")
    if (
        guidance_enabled is not True
        or isinstance(guidance_scale, bool)
        or not isinstance(guidance_scale, (int, float))
        or guidance_scale <= 0
    ):
        raise ValueError("source config lacks enabled positive differential guidance")
    ema_raw = train.get("ema_config")
    dropout_present = "caption_dropout_rate" in dataset
    ema_present = isinstance(ema_raw, dict)
    if arm == "K3" and (dropout_present or ema_present):
        raise ValueError("K3 no longer has the frozen unknown dropout/EMA surface")
    if arm != "K3" and (not dropout_present or not ema_present):
        raise ValueError(f"{arm} lacks its declared dropout/EMA fields")

    selector = submission.get("selection")
    selector_mode = selector.get("mode") if isinstance(selector, dict) else None
    if selector_mode not in {"holdout_selected", "highest_numbered_fallback"}:
        raise ValueError("official selection mode is absent or unsupported")
    optimizer_params = train.get("optimizer_params")
    if not isinstance(optimizer_params, dict) or not optimizer_params:
        raise ValueError("source optimizer parameters are absent")

    learning_rate_evidence = "Immutable source config."
    if str(train.get("optimizer", "")).casefold() == "automagic":
        learning_rate_evidence = (
            "Immutable source config; for Automagic this scalar is a configured "
            "input/base LR, not a realized per-update trajectory."
        )
    known: dict[str, tuple[Any, str, str]] = {
        "planned_steps": (
            planned,
            "/config/process/0/train/steps",
            "Immutable source config.",
        ),
        "submitted_step": (
            submitted,
            "/artifact/__metadata__/training_info/step",
            "Submitted safetensors training_info.step, cross-checked to the field ledger.",
        ),
        "learning_rate": (
            train.get("lr"),
            "/config/process/0/train/lr",
            learning_rate_evidence,
        ),
        "rank": (
            _positive_int(network.get("linear"), "network rank"),
            "/config/process/0/network/linear",
            "Immutable source config.",
        ),
        "alpha": (
            _positive_int(network.get("linear_alpha"), "network alpha"),
            "/config/process/0/network/linear_alpha",
            "Immutable source config.",
        ),
        "optimizer": (
            train.get("optimizer"),
            "/config/process/0/train/optimizer",
            "Immutable source config.",
        ),
        "optimizer_parameters": (
            optimizer_params,
            "/config/process/0/train/optimizer_params",
            "Immutable source config.",
        ),
        "loss": (
            train.get("loss_type"),
            "/config/process/0/train/loss_type",
            "Immutable source config.",
        ),
        "guidance": (
            {"enabled": True, "scale": guidance_scale},
            "/derived/config/process/0/train/differential_guidance",
            "Both do_differential_guidance and differential_guidance_scale are present in the immutable config.",
        ),
        "scheduler": (
            train.get("noise_scheduler"),
            "/config/process/0/train/noise_scheduler",
            "Immutable source config.",
        ),
        "gradient_accumulation": (
            accumulation,
            "/config/process/0/train/gradient_accumulation",
            "Immutable source config.",
        ),
        "effective_batch": (
            batch_size * accumulation,
            "/derived/config/process/0/train/effective_batch",
            "Single-process config batch_size multiplied by gradient_accumulation.",
        ),
        "save_cadence": (
            _positive_int(save.get("save_every"), "save cadence"),
            "/config/process/0/save/save_every",
            "Immutable source config.",
        ),
        "selector": (
            selector_mode,
            "/official/selection/mode",
            "Public file layout/selection record plus validator checkpoint precedence, bound by the field ledger.",
        ),
    }
    if dropout_present:
        known["dropout"] = (
            dataset["caption_dropout_rate"],
            "/config/process/0/datasets/0/caption_dropout_rate",
            "Immutable source config.",
        )
    if ema_present:
        known["ema"] = (
            {"enabled": ema_raw.get("use_ema"), "decay": ema_raw.get("ema_decay")},
            "/config/process/0/train/ema_config",
            "Immutable source config.",
        )
    all_recipe_fields = {
        "planned_steps",
        "submitted_step",
        "learning_rate",
        "rank",
        "alpha",
        "optimizer",
        "optimizer_parameters",
        "loss",
        "guidance",
        "scheduler",
        "dropout",
        "gradient_accumulation",
        "effective_batch",
        "ema",
        "save_cadence",
        "selector",
    }
    unknown_evidence = {
        "dropout": (
            "caption_dropout_rate is absent from the immutable source config; "
            "framework default not imputed."
        ),
        "ema": (
            "ema_config is absent from the immutable source config; framework "
            "default not imputed."
        ),
    }
    fields = {
        name: (
            _row(*known[name])
            if name in known
            else _unknown(
                unknown_evidence.get(
                    name,
                    "Field absent from the immutable source config; framework "
                    "default not imputed.",
                )
            )
        )
        for name in sorted(all_recipe_fields)
    }
    observed = {row[1]: row[0] for row in known.values()}
    normalized_recipe = {
        "schema": 1,
        "kind": "forge-krea-normalized-recipe",
        "fields": fields,
    }
    # Validate the recipe against the same strict vocabulary used downstream.
    classified = {"observed": observed, "unsupported": [], "adapted": []}
    krea_provenance.normalize_recipe(
        normalized_recipe, classified_fields=classified, source_only=True
    )

    task = ledger.get("task")
    if not isinstance(task, dict) or not isinstance(task.get("task_id"), str):
        raise ValueError("field ledger task identity is absent")
    tournament_api = task.get("tournament_api")
    if not isinstance(tournament_api, str):
        raise ValueError("field ledger tournament URL is absent")
    tournament_parts = [
        part for part in urlsplit(tournament_api).path.split("/") if part
    ]
    if len(tournament_parts) < 3 or tournament_parts[-1] != "details":
        raise ValueError("field ledger tournament URL is malformed")
    tournament_id = tournament_parts[-2]
    config_url = submission.get("config_url")
    if not isinstance(config_url, str):
        raise ValueError("official submission lacks its immutable config URL")
    marker = f"/resolve/{revision}/"
    if marker not in config_url:
        raise ValueError("official config URL is not bound to the source revision")
    config_repo_path = config_url.split(marker, 1)[1]
    artifact = submission.get("artifact")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
        raise ValueError("official artifact path is absent")
    repo = submission.get("repo")
    if not isinstance(repo, str):
        raise ValueError("official repository is absent")
    return {
        "source_arm_id": arm,
        "source": {
            "url": f"https://huggingface.co/{repo}",
            "revision": revision,
        },
        "official_context": {
            "tournament_id": tournament_id,
            "task_id": task["task_id"],
            "hotkey": hotkey,
            "submission_id": submission["submission_id"],
            "official_rank": submission["official_rank"],
            "official_loss": submission["score"],
            "repository": repo,
            "repo_revision": revision,
            "artifact_repo_path": artifact["path"],
            "config_repo_path": config_repo_path,
        },
        "fields": classified,
        # The official task/tournament payloads do not disclose the exact
        # evaluator image commit.  The field ledger's f947... revision proves
        # checkpoint-precedence source code only; it is not relabeled here as
        # the task evaluator.
        "evaluator_sha": None,
        "matched_concept": {
            "available": False,
            "dataset_sha256": None,
            "basis": "The hidden public-task concept bytes were not exposed in the immutable submission repository.",
            "evidence": {
                "public_task_id": task["task_id"],
                "matched_dataset_recovered": False,
            },
        },
        "adaptation_target": {
            "mode": "local_reproduction",
            "model_type": "krea2",
            "source_artifact_role": "reference_only",
            "candidate_role": "local_training_output",
            "description": "Retrain the source family from the immutable Krea base on independently sealed fixtures; do not score the public artifact as a matched-concept candidate.",
        },
        "local_reproduction_disclosure": _local_reproduction_disclosure(
            arm, normalized_recipe=normalized_recipe
        ),
        "normalized_recipe": normalized_recipe,
        "review_assertion": {
            "status": "unreviewed",
            "reviewer_identity": "Pending Independent Reviewer",
            "notes": "Machine-derived from sealed public evidence; this record is not execution approval.",
        },
    }


def main() -> int:
    args = _parse()
    metadata = build_metadata(
        args.arm,
        source_config_path=args.source_config,
        source_artifact_path=args.source_artifact,
        field_ledger_path=args.field_ledger,
    )
    manifest = krea_provenance.build_manifest(
        metadata,
        source_config_path=args.source_config,
        source_artifact_path=args.source_artifact,
        field_ledger_path=args.field_ledger,
        task_raw_path=args.task_raw,
        tournament_raw_path=args.tournament_raw,
        revision_manifest_path=args.revision_manifest,
    )
    krea_provenance.publish_exclusive(args.output, manifest)
    print(krea_provenance.canonical_bytes(manifest).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
