#!/usr/bin/env python3
"""Create an immutable provenance record for one public Krea calibration arm.

The record deliberately distinguishes facts observed in the public submission
from fields Forge cannot represent and choices we adapted for a local replay.
It contains no wall-clock value or local input path, so the same inputs produce
the same canonical JSON and digest on every host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVIEW_STATES = frozenset({"unreviewed", "approved", "rejected"})
_RECIPE_CLASSIFICATIONS = frozenset(
    {"known", "unsupported", "adapted", "unknown", "unknown_source_fixed"}
)
_RECIPE_FIELDS = frozenset(
    {
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
)
_RECIPE_KIND = "forge-krea-normalized-recipe"
_OPTIMIZERS = frozenset({"adamw8bit", "adamw", "automagic", "lion"})
_LOSSES = frozenset({"mse", "mae", "l2", "huber"})
_SCHEDULERS = frozenset({"flowmatch", "constant", "linear", "cosine"})
_SELECTORS = frozenset(
    {
        "exact_final",
        "training_loss_divergence",
        "holdout_selected",
        "highest_numbered_fallback",
        "explicit_checkpoint_promotion",
        "last_priority",
        "last_equals_numbered_checkpoint",
    }
)
_KIND = "forge-krea-public-arm-provenance"
_SCHEMA = 1


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--source-artifact", required=True, type=Path)
    parser.add_argument("--field-ledger", required=True, type=Path)
    parser.add_argument("--task-raw", required=True, type=Path)
    parser.add_argument("--tournament-raw", required=True, type=Path)
    parser.add_argument("--revision-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def canonical_bytes(value: Any) -> bytes:
    """Return the sole canonical encoding used for manifests and digests."""

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


def _file_identity(path: Path, label: str) -> dict[str, Any]:
    before = path.stat()
    digest = file_sha256(path)
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError(f"{label} changed while it was hashed")
    return {"name": path.name, "bytes": after.st_size, "sha256": digest}


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact_keys(
    value: dict[str, Any],
    required: set[str],
    label: str,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _field_name(value: Any, label: str) -> str:
    value = _text(value, label)
    if not value.startswith("/") or "//" in value:
        raise ValueError(f"{label} must be an absolute JSON-pointer-like field")
    return value


def _validate_json(value: Any, label: str) -> None:
    try:
        canonical_bytes(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not finite canonical JSON") from exc
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")


def _source(value: Any) -> dict[str, str]:
    source = _require_object(value, "source")
    _exact_keys(source, {"url", "revision"}, "source")
    url = _text(source["url"], "source.url")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "source.url must be a credential-free HTTPS URL without query or fragment"
        )
    revision = _text(source["revision"], "source.revision").lower()
    if not _REVISION.fullmatch(revision):
        raise ValueError(
            "source.revision must be a full 40- or 64-digit hexadecimal revision"
        )
    return {"url": url, "revision": revision}


def _evaluator_sha(value: Any) -> str | None:
    """Normalize a source-task evaluator revision without inventing one.

    Public tournament records do not always disclose the exact evaluator image
    commit.  ``None`` is therefore a valid source-provenance fact.  Legacy
    direct-artifact scoring still fails closed because its consumer requires an
    exact equality with the active evaluator commit.
    """

    if value is None:
        return None
    revision = _text(value, "evaluator_sha").lower()
    if not _REVISION.fullmatch(revision):
        raise ValueError(
            "evaluator_sha must be null or a full 40-/64-digit hexadecimal SHA"
        )
    return revision


def _fields(value: Any) -> dict[str, Any]:
    fields = _require_object(value, "fields")
    _exact_keys(fields, {"observed", "unsupported", "adapted"}, "fields")
    observed = _require_object(fields["observed"], "fields.observed")
    normalized_observed: dict[str, Any] = {}
    for raw_name, raw_value in observed.items():
        name = _field_name(raw_name, "observed field")
        _validate_json(raw_value, f"observed field {name}")
        normalized_observed[name] = raw_value

    unsupported = fields["unsupported"]
    adapted = fields["adapted"]
    if not isinstance(unsupported, list) or not isinstance(adapted, list):
        raise ValueError("fields.unsupported and fields.adapted must be arrays")

    normalized_unsupported: list[dict[str, Any]] = []
    unsupported_names: set[str] = set()
    for index, raw in enumerate(unsupported):
        row = _require_object(raw, f"fields.unsupported[{index}]")
        _exact_keys(row, {"field", "value", "reason"}, f"fields.unsupported[{index}]")
        name = _field_name(row["field"], f"fields.unsupported[{index}].field")
        if name in unsupported_names:
            raise ValueError(f"duplicate unsupported field: {name}")
        unsupported_names.add(name)
        _validate_json(row["value"], f"unsupported field {name}")
        normalized_unsupported.append(
            {
                "field": name,
                "value": row["value"],
                "reason": _text(row["reason"], "reason"),
            }
        )

    normalized_adapted: list[dict[str, Any]] = []
    adapted_names: set[str] = set()
    adapted_targets: set[str] = set()
    for index, raw in enumerate(adapted):
        row = _require_object(raw, f"fields.adapted[{index}]")
        required = {
            "source_field",
            "source_value",
            "target_field",
            "target_value",
            "reason",
        }
        _exact_keys(row, required, f"fields.adapted[{index}]")
        source_field = _field_name(
            row["source_field"], f"fields.adapted[{index}].source_field"
        )
        target_field = _field_name(
            row["target_field"], f"fields.adapted[{index}].target_field"
        )
        if source_field in adapted_names:
            raise ValueError(f"duplicate adapted source field: {source_field}")
        if target_field in adapted_targets:
            raise ValueError(f"duplicate adapted target field: {target_field}")
        adapted_names.add(source_field)
        adapted_targets.add(target_field)
        _validate_json(row["source_value"], f"adapted source field {source_field}")
        _validate_json(row["target_value"], f"adapted target field {target_field}")
        normalized_adapted.append(
            {
                "source_field": source_field,
                "source_value": row["source_value"],
                "target_field": target_field,
                "target_value": row["target_value"],
                "reason": _text(row["reason"], "reason"),
            }
        )

    overlap = set(normalized_observed) & (unsupported_names | adapted_names)
    overlap |= unsupported_names & adapted_names
    target_overlap = adapted_targets & (
        set(normalized_observed) | unsupported_names | adapted_names
    )
    overlap |= target_overlap
    if overlap:
        raise ValueError(f"fields have multiple classifications: {sorted(overlap)}")

    normalized_unsupported.sort(key=canonical_bytes)
    normalized_adapted.sort(key=canonical_bytes)
    return {
        "observed": normalized_observed,
        "unsupported": normalized_unsupported,
        "adapted": normalized_adapted,
    }


def _review_assertion(value: Any) -> dict[str, Any]:
    """Validate self-declared review metadata, not an authenticated signature."""

    review = _require_object(value, "review_assertion")
    _exact_keys(
        review,
        {"status", "reviewer_identity"},
        "review_assertion",
        {"notes"},
    )
    status = _text(review["status"], "review_assertion.status")
    if status not in _REVIEW_STATES:
        raise ValueError(f"unsupported review status: {status}")
    reviewer = _text(review["reviewer_identity"], "review_assertion.reviewer_identity")
    notes = review.get("notes")
    if notes is not None:
        notes = _text(notes, "review_assertion.notes")
    return {"status": status, "reviewer_identity": reviewer, "notes": notes}


def _validate_recipe_value(name: str, value: Any, label: str) -> None:
    if value is None:
        return
    if name in {
        "planned_steps",
        "submitted_step",
        "rank",
        "alpha",
        "gradient_accumulation",
        "effective_batch",
        "save_cadence",
    }:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
        return
    if name == "learning_rate":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 < float(value) <= 1.0
        ):
            raise ValueError(f"{label} must be finite and in (0, 1]")
        return
    if name == "dropout":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{label} must be finite and in [0, 1]")
        return
    if name in {"optimizer", "loss", "scheduler", "selector"}:
        allowed = {
            "optimizer": _OPTIMIZERS,
            "loss": _LOSSES,
            "scheduler": _SCHEDULERS,
            "selector": _SELECTORS,
        }[name]
        if not isinstance(value, str) or value.casefold() not in allowed:
            raise ValueError(f"{label} is unsupported")
        return
    if name == "optimizer_parameters":
        parameters = _require_object(value, label)
        if not parameters:
            raise ValueError(f"{label} must not be empty")
        _validate_json(parameters, label)
        for key, item in parameters.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{label} contains an invalid key")
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"{label}.{key} must be numeric")
            if not math.isfinite(float(item)) or float(item) < 0:
                raise ValueError(f"{label}.{key} must be finite and non-negative")
        return
    if name == "guidance":
        guidance = _require_object(value, label)
        _exact_keys(guidance, {"enabled", "scale"}, label)
        enabled = guidance["enabled"]
        scale = guidance["scale"]
        if not isinstance(enabled, bool):
            raise ValueError(f"{label}.enabled must be boolean")
        if enabled:
            if (
                isinstance(scale, bool)
                or not isinstance(scale, (int, float))
                or not math.isfinite(float(scale))
                or float(scale) <= 0
            ):
                raise ValueError(f"{label}.scale must be finite and positive")
        elif scale is not None:
            raise ValueError(f"{label}.scale must be null when disabled")
        return
    if name == "ema":
        ema = _require_object(value, label)
        _exact_keys(ema, {"enabled", "decay"}, label)
        if not isinstance(ema["enabled"], bool):
            raise ValueError(f"{label}.enabled must be boolean")
        decay = ema["decay"]
        if (
            isinstance(decay, bool)
            or not isinstance(decay, (int, float))
            or not math.isfinite(float(decay))
            or not 0.0 < float(decay) < 1.0
        ):
            raise ValueError(f"{label}.decay must be finite and in (0, 1)")
        return
    raise AssertionError(f"recipe field lacks a validator: {name}")


def normalize_recipe(
    value: Any,
    *,
    classified_fields: dict[str, Any] | None = None,
    source_only: bool = False,
) -> dict[str, Any]:
    """Validate the exact, versioned Krea recipe vocabulary.

    Every score-relevant field has an explicit epistemic classification.  A
    missing key may therefore never be mistaken for a framework default, and
    an extra key may never create an unreviewed experimental axis.
    """

    recipe = _require_object(value, "normalized_recipe")
    _exact_keys(recipe, {"schema", "kind", "fields"}, "normalized_recipe")
    if recipe["schema"] != 1 or recipe["kind"] != _RECIPE_KIND:
        raise ValueError("unsupported normalized_recipe schema or kind")
    fields = _require_object(recipe["fields"], "normalized_recipe.fields")
    _exact_keys(fields, set(_RECIPE_FIELDS), "normalized_recipe.fields")
    normalized: dict[str, dict[str, Any]] = {}
    claimed_source_pointers: dict[str, str] = {}
    for name in sorted(_RECIPE_FIELDS):
        row = _require_object(fields[name], f"normalized_recipe.fields.{name}")
        _exact_keys(
            row,
            {
                "classification",
                "source_pointers",
                "source_value",
                "effective_value",
                "evidence",
            },
            f"normalized_recipe.fields.{name}",
        )
        classification = _text(
            row["classification"],
            f"normalized_recipe.fields.{name}.classification",
        )
        if classification not in _RECIPE_CLASSIFICATIONS:
            raise ValueError(
                f"normalized_recipe.fields.{name} has unsupported classification"
            )
        source_value = row["source_value"]
        effective_value = row["effective_value"]
        source_pointers = row["source_pointers"]
        if (
            not isinstance(source_pointers, list)
            or any(not isinstance(pointer, str) for pointer in source_pointers)
            or source_pointers != sorted(set(source_pointers))
        ):
            raise ValueError(
                f"normalized_recipe.fields.{name}.source_pointers is invalid"
            )
        for pointer in source_pointers:
            _field_name(pointer, f"normalized_recipe.fields.{name}.source_pointer")
            previous = claimed_source_pointers.get(pointer)
            if previous is not None:
                raise ValueError(
                    f"source pointer {pointer} is claimed by both {previous} and {name}"
                )
            claimed_source_pointers[pointer] = name
        _validate_json(source_value, f"normalized_recipe.fields.{name}.source_value")
        _validate_json(
            effective_value, f"normalized_recipe.fields.{name}.effective_value"
        )
        _validate_recipe_value(
            name, source_value, f"normalized_recipe.fields.{name}.source_value"
        )
        _validate_recipe_value(
            name,
            effective_value,
            f"normalized_recipe.fields.{name}.effective_value",
        )
        if source_only:
            if classification not in {"known", "unsupported", "unknown"}:
                raise ValueError(
                    f"source recipe field {name} cannot declare a local adaptation"
                )
            if classification in {"known", "unsupported"}:
                if (
                    not source_pointers
                    or source_value is None
                    or effective_value is not None
                ):
                    raise ValueError(
                        f"source recipe field {name} must bind only its source value"
                    )
            elif (
                source_pointers
                or source_value is not None
                or effective_value is not None
            ):
                raise ValueError(
                    f"unknown source recipe field {name} must use null values"
                )
        elif classification == "known":
            if (
                not source_pointers
                or source_value is None
                or source_value != effective_value
            ):
                raise ValueError(
                    f"known recipe field {name} must preserve a non-null source value"
                )
        elif classification == "unsupported":
            if (
                not source_pointers
                or source_value is None
                or effective_value is not None
            ):
                raise ValueError(
                    f"unsupported recipe field {name} must have source-only value"
                )
        elif classification == "adapted":
            if (
                not source_pointers
                or source_value is None
                or effective_value is None
                or source_value == effective_value
            ):
                raise ValueError(
                    f"adapted recipe field {name} must declare different values"
                )
        elif classification == "unknown":
            if (
                source_pointers
                or source_value is not None
                or effective_value is not None
            ):
                raise ValueError(f"unknown recipe field {name} must use null values")
        elif source_pointers or source_value is not None or effective_value is None:
            raise ValueError(
                f"unknown_source_fixed recipe field {name} requires only an "
                "effective value"
            )
        normalized[name] = {
            "classification": classification,
            "source_pointers": source_pointers,
            "source_value": source_value,
            "effective_value": effective_value,
            "evidence": _text(
                row["evidence"], f"normalized_recipe.fields.{name}.evidence"
            ),
        }
    if classified_fields is not None:
        observed = classified_fields["observed"]
        unsupported = {
            row["field"]: row["value"] for row in classified_fields["unsupported"]
        }
        adapted = {
            row["source_field"]: (row["source_value"], row["target_value"])
            for row in classified_fields["adapted"]
        }
        buckets = {
            "known": observed,
            "unsupported": unsupported,
            "adapted": adapted,
        }
        for name, row in normalized.items():
            classification = row["classification"]
            if classification in {"unknown", "unknown_source_fixed"}:
                continue
            bucket = buckets[classification]
            pointers = row["source_pointers"]
            if any(pointer not in bucket for pointer in pointers):
                raise ValueError(
                    f"normalized recipe field {name} contradicts raw field classification"
                )
            if len(pointers) != 1:
                raise ValueError(
                    f"normalized recipe field {name} must bind one authoritative pointer"
                )
            raw = bucket[pointers[0]]
            if classification == "adapted":
                raw_source, raw_effective = raw
                if (
                    row["source_value"] != raw_source
                    or row["effective_value"] != raw_effective
                ):
                    raise ValueError(
                        f"normalized recipe field {name} contradicts adapted values"
                    )
            elif row["source_value"] != raw:
                raise ValueError(
                    f"normalized recipe field {name} contradicts observed source value"
                )
    for value_key in ("source_value", "effective_value"):
        planned = normalized["planned_steps"][value_key]
        submitted = normalized["submitted_step"][value_key]
        if (
            isinstance(planned, int)
            and not isinstance(planned, bool)
            and isinstance(submitted, int)
            and not isinstance(submitted, bool)
            and submitted > planned
        ):
            raise ValueError(
                f"normalized recipe {value_key} submitted_step exceeds planned_steps"
            )
    return {"schema": 1, "kind": _RECIPE_KIND, "fields": normalized}


def normalize_execution_recipe(
    value: Any, *, source_recipe: dict[str, Any]
) -> dict[str, Any]:
    """Validate concrete local choices against an immutable source recipe."""

    source = normalize_recipe(source_recipe, source_only=True)
    execution = normalize_recipe(value)
    for name, local in execution["fields"].items():
        public = source["fields"][name]
        source_state = public["classification"]
        local_state = local["classification"]
        if name in {"submitted_step", "selector"}:
            if source_state in {"known", "unsupported"}:
                if (
                    local_state != "unsupported"
                    or local["source_pointers"] != public["source_pointers"]
                    or local["source_value"] != public["source_value"]
                    or local["effective_value"] is not None
                ):
                    raise ValueError(
                        f"execution recipe must keep {name} source-only; "
                        "checkpoint choice is downstream"
                    )
            elif local_state != "unknown":
                raise ValueError(
                    f"execution recipe must keep unknown {name} unresolved"
                )
            continue
        if source_state == "known":
            expected_state = (
                "known"
                if local["effective_value"] == public["source_value"]
                else "adapted"
            )
            if (
                local_state != expected_state
                or local["source_pointers"] != public["source_pointers"]
                or local["source_value"] != public["source_value"]
            ):
                raise ValueError(
                    f"execution recipe field {name} contradicts known source facts"
                )
        elif source_state == "unsupported":
            if (
                local_state != "unsupported"
                or local["source_pointers"] != public["source_pointers"]
                or local["source_value"] != public["source_value"]
            ):
                raise ValueError(
                    f"execution recipe field {name} contradicts unsupported source facts"
                )
        elif local_state not in {"unknown", "unknown_source_fixed"}:
            raise ValueError(
                f"execution recipe field {name} overstates an unknown source fact"
            )
    return execution


def _matched_concept(value: Any) -> dict[str, Any]:
    matched = _require_object(value, "matched_concept")
    _exact_keys(
        matched,
        {"available", "dataset_sha256", "basis", "evidence"},
        "matched_concept",
    )
    available = matched["available"]
    if not isinstance(available, bool):
        raise ValueError("matched_concept.available must be boolean")
    dataset_sha256 = matched["dataset_sha256"]
    if available:
        if not isinstance(dataset_sha256, str) or not _SHA256.fullmatch(dataset_sha256):
            raise ValueError("available matched concept requires a full dataset_sha256")
    elif dataset_sha256 is not None:
        raise ValueError("unavailable matched concept must use null dataset_sha256")
    basis = _text(matched["basis"], "matched_concept.basis")
    evidence = _require_object(matched["evidence"], "matched_concept.evidence")
    if not evidence:
        raise ValueError("matched_concept.evidence must not be empty")
    _validate_json(evidence, "matched_concept.evidence")
    return {
        "available": available,
        "dataset_sha256": dataset_sha256,
        "basis": basis,
        "evidence": evidence,
    }


def _adaptation_target(
    value: Any, *, matched_concept: dict[str, Any]
) -> dict[str, str]:
    target = _require_object(value, "adaptation_target")
    _exact_keys(
        target,
        {"mode", "model_type", "source_artifact_role", "candidate_role", "description"},
        "adaptation_target",
    )
    mode = _text(target["mode"], "adaptation_target.mode")
    roles = {
        "local_reproduction": ("reference_only", "local_training_output"),
        "direct_public_artifact": ("score_candidate", "source_artifact"),
    }
    if mode not in roles:
        raise ValueError(f"unsupported adaptation target mode: {mode}")
    model_type = _text(target["model_type"], "adaptation_target.model_type")
    if model_type != "krea2":
        raise ValueError("adaptation_target.model_type must be krea2")
    source_role = _text(
        target["source_artifact_role"], "adaptation_target.source_artifact_role"
    )
    candidate_role = _text(target["candidate_role"], "adaptation_target.candidate_role")
    if (source_role, candidate_role) != roles[mode]:
        raise ValueError(f"adaptation target roles contradict mode {mode}")
    if mode == "direct_public_artifact" and not matched_concept["available"]:
        raise ValueError("direct_public_artifact requires an available matched concept")
    return {
        "mode": mode,
        "model_type": model_type,
        "source_artifact_role": source_role,
        "candidate_role": candidate_role,
        "description": _text(target["description"], "adaptation_target.description"),
    }


def _safe_file(path: Path, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{label} must not be empty: {path}")
    return path


def _load_json(path: Path, label: str) -> dict[str, Any]:
    path = _safe_file(path, label)
    before = path.stat()
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"{label} changed while read")
    return _require_object(value, label)


def _official_context(
    value: Any,
    *,
    field_ledger_path: Path,
    source: dict[str, str],
    source_artifact: dict[str, Any],
    source_config: dict[str, Any],
    task_raw_path: Path,
    tournament_raw_path: Path,
    revision_manifest_path: Path,
) -> dict[str, Any]:
    context = _require_object(value, "official_context")
    _exact_keys(
        context,
        {
            "tournament_id",
            "task_id",
            "hotkey",
            "submission_id",
            "official_rank",
            "official_loss",
            "repository",
            "repo_revision",
            "artifact_repo_path",
            "config_repo_path",
        },
        "official_context",
    )
    for key in (
        "tournament_id",
        "task_id",
        "hotkey",
        "submission_id",
        "repository",
        "artifact_repo_path",
    ):
        _text(context[key], f"official_context.{key}")
    config_path = context["config_repo_path"]
    if config_path is not None:
        _text(config_path, "official_context.config_repo_path")
    for key in ("artifact_repo_path", "config_repo_path"):
        path = context[key]
        if path is not None and (
            path.startswith("/") or ".." in Path(path).parts or "\\" in path
        ):
            raise ValueError(
                f"official_context.{key} must be a safe repo-relative path"
            )
    rank = context["official_rank"]
    loss = context["official_loss"]
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("official_context.official_rank is invalid")
    if (
        isinstance(loss, bool)
        or not isinstance(loss, (int, float))
        or not math.isfinite(float(loss))
        or float(loss) < 0
    ):
        raise ValueError("official_context.official_loss is invalid")
    revision = _text(context["repo_revision"], "official_context.repo_revision").lower()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("official_context.repo_revision must be a full Git commit")

    ledger_path = _safe_file(field_ledger_path, "field ledger")
    ledger = _load_json(ledger_path, "field ledger")
    if (
        ledger.get("schema") != 1
        or ledger.get("kind") != "sn56-week5-krea-r1-public-field-ledger"
    ):
        raise ValueError("unsupported Krea field ledger")
    task = _require_object(ledger.get("task"), "field ledger task")
    snapshot = _require_object(task.get("snapshot"), "field ledger task snapshot")
    submissions = ledger.get("submissions")
    if not isinstance(submissions, list):
        raise ValueError("field ledger submissions must be an array")
    matching = [
        row
        for row in submissions
        if isinstance(row, dict)
        and row.get("hotkey") == context["hotkey"]
        and row.get("submission_id") == context["submission_id"]
    ]
    if len(matching) != 1:
        raise ValueError("official context does not identify one ledger submission")
    row = matching[0]
    artifact = _require_object(row.get("artifact"), "ledger artifact")
    tournament_api = _text(task.get("tournament_api"), "field ledger tournament_api")
    if (
        task.get("task_id") != context["task_id"]
        or f"/tournament/{context['tournament_id']}/details" not in tournament_api
        or row.get("official_rank") != rank
        or row.get("score") != loss
        or row.get("repo") != context["repository"]
        or row.get("repo_revision") != revision
        or artifact.get("path") != context["artifact_repo_path"]
        or artifact.get("lfs_sha256") != source_artifact["sha256"]
        or source["revision"] != revision
        or urlsplit(source["url"]).path.strip("/") != context["repository"]
    ):
        raise ValueError(
            "official context contradicts the bound field ledger/source bytes"
        )
    config_url = row.get("config_url")
    expected_config_path = context["config_repo_path"]
    if expected_config_path is None:
        if config_url is not None:
            raise ValueError(
                "ledger exposes a config but official context omits its path"
            )
    elif (
        not isinstance(config_url, str) or f"/{expected_config_path}" not in config_url
    ):
        raise ValueError("official config path contradicts the field ledger")
    for key in ("tournament_snapshot_sha256", "task_snapshot_sha256"):
        if not isinstance(snapshot.get(key), str) or not _SHA256.fullmatch(
            snapshot[key]
        ):
            raise ValueError(f"field ledger lacks a valid {key}")
    task_raw_path = _safe_file(task_raw_path, "raw official task")
    tournament_raw_path = _safe_file(tournament_raw_path, "raw official tournament")
    revision_manifest_path = _safe_file(
        revision_manifest_path, "raw Hugging Face revision manifest"
    )
    task_raw = _load_json(task_raw_path, "raw official task")
    tournament_raw = _load_json(tournament_raw_path, "raw official tournament")
    revision_manifest = _load_json(
        revision_manifest_path, "raw Hugging Face revision manifest"
    )
    raw_task_sha = file_sha256(task_raw_path)
    raw_tournament_sha = file_sha256(tournament_raw_path)
    if (
        raw_task_sha != snapshot["task_snapshot_sha256"]
        or raw_tournament_sha != snapshot["tournament_snapshot_sha256"]
    ):
        raise ValueError(
            "field ledger raw observation digests do not match bound payloads"
        )
    task_rows = task_raw.get("hotkey_details")
    if not isinstance(task_rows, list):
        raise ValueError("raw official task lacks hotkey_details")
    raw_matches = [
        item
        for item in task_rows
        if isinstance(item, dict)
        and item.get("hotkey") == context["hotkey"]
        and item.get("submission_id") == context["submission_id"]
    ]
    if (
        task_raw.get("task_id") != context["task_id"]
        or task_raw.get("model_type") != "krea2"
        or len(raw_matches) != 1
        or raw_matches[0].get("rank") != rank
        or raw_matches[0].get("test_loss") != loss
        or raw_matches[0].get("repo") != context["repository"]
    ):
        raise ValueError("raw official task contradicts the source arm")
    rounds = tournament_raw.get("rounds")
    if not isinstance(rounds, list):
        raise ValueError("raw tournament lacks rounds")
    round_matches = [
        round_row
        for round_row in rounds
        if isinstance(round_row, dict)
        and round_row.get("status") == "completed"
        and any(
            isinstance(task_row, dict) and task_row.get("task_id") == context["task_id"]
            for task_row in round_row.get("tasks", [])
        )
    ]
    if (
        tournament_raw.get("tournament_id") != context["tournament_id"]
        or len(round_matches) != 1
    ):
        raise ValueError("raw tournament does not bind one completed source round")
    _exact_keys(
        revision_manifest,
        {
            "capture_complete",
            "captures",
            "config_absent",
            "configs",
            "eligible_weight_plan",
            "failures",
            "processing_complete",
            "repo_id",
            "revision",
            "skipped",
            "tree_entry_count",
            "tree_file_count",
            "tree_truncated",
            "weights_enabled",
        },
        "raw Hugging Face revision manifest",
    )
    if (
        revision_manifest["capture_complete"] is not True
        or revision_manifest["processing_complete"] is not True
        or revision_manifest["tree_truncated"] is not False
        or revision_manifest["weights_enabled"] is not True
        or revision_manifest["failures"] != []
        or revision_manifest["skipped"] != []
        or revision_manifest["repo_id"] != context["repository"]
        or revision_manifest["revision"] != revision
    ):
        raise ValueError("Hugging Face revision capture is partial or mismatched")
    captures = revision_manifest["captures"]
    weights = revision_manifest["eligible_weight_plan"]
    configs = revision_manifest["configs"]
    if not all(isinstance(rows, list) for rows in (captures, weights, configs)):
        raise ValueError("Hugging Face revision capture arrays are invalid")
    artifact_matches = [
        item
        for item in captures
        if isinstance(item, dict)
        and item.get("path") == context["artifact_repo_path"]
        and item.get("kind") == "weight"
        and item.get("captured") is True
        and item.get("object_sha256") == source_artifact["sha256"]
        and item.get("bytes") == source_artifact["bytes"]
    ]
    weight_matches = [
        item
        for item in weights
        if isinstance(item, dict)
        and item.get("path") == context["artifact_repo_path"]
        and item.get("lfs_oid") == source_artifact["sha256"]
        and item.get("size") == source_artifact["bytes"]
    ]
    config_matches = [
        item
        for item in captures
        if isinstance(item, dict)
        and item.get("path") == context["config_repo_path"]
        and item.get("kind") == "small"
        and item.get("captured") is True
        and item.get("object_sha256") == source_config["sha256"]
        and item.get("bytes") == source_config["bytes"]
    ]
    if (
        context["config_repo_path"] is None
        or revision_manifest["config_absent"] is not False
        or context["config_repo_path"] not in configs
        or len(artifact_matches) != 1
        or len(weight_matches) != 1
        or len(config_matches) != 1
    ):
        raise ValueError("revision manifest does not prove exact config/artifact bytes")
    normalized_context = dict(context)
    normalized_context["repo_revision"] = revision
    return {
        **normalized_context,
        "official_observations": {
            "tournament_raw": _file_identity(
                tournament_raw_path, "raw official tournament"
            ),
            "task_raw": _file_identity(task_raw_path, "raw official task"),
            "revision_manifest": _file_identity(
                revision_manifest_path, "raw Hugging Face revision manifest"
            ),
        },
        "field_ledger": _file_identity(ledger_path, "field ledger"),
        "semantic_linkage": {
            "task_submission_unique": True,
            "rank_and_loss_match": True,
            "repository_revision_match": True,
            "artifact_path_and_lfs_match": True,
            "source_url_revision_match": True,
            "raw_official_task_match": True,
            "raw_tournament_round_match": True,
            "revision_capture_complete": True,
        },
    }


def _validate_official_context_record(
    value: Any,
    *,
    source: dict[str, str],
    source_artifact: dict[str, Any],
    source_config: dict[str, Any],
    field_ledger_path: Path | None,
    task_raw_path: Path | None,
    tournament_raw_path: Path | None,
    revision_manifest_path: Path | None,
) -> dict[str, Any]:
    record = _require_object(value, "official_context")
    base_keys = {
        "tournament_id",
        "task_id",
        "hotkey",
        "submission_id",
        "official_rank",
        "official_loss",
        "repository",
        "repo_revision",
        "artifact_repo_path",
        "config_repo_path",
    }
    _exact_keys(
        record,
        base_keys | {"official_observations", "field_ledger", "semantic_linkage"},
        "official_context",
    )
    rebound_paths = (
        field_ledger_path,
        task_raw_path,
        tournament_raw_path,
        revision_manifest_path,
    )
    if any(path is not None for path in rebound_paths):
        if any(path is None for path in rebound_paths):
            raise ValueError(
                "official context rebinding requires every raw evidence file"
            )
        rebuilt = _official_context(
            {key: record[key] for key in base_keys},
            field_ledger_path=field_ledger_path,
            source=source,
            source_artifact=source_artifact,
            source_config=source_config,
            task_raw_path=task_raw_path,  # type: ignore[arg-type]
            tournament_raw_path=tournament_raw_path,  # type: ignore[arg-type]
            revision_manifest_path=revision_manifest_path,  # type: ignore[arg-type]
        )
        if rebuilt != record:
            raise ValueError("official context differs from the rebound field ledger")
        return record
    for key in (
        "tournament_id",
        "task_id",
        "hotkey",
        "submission_id",
        "repository",
        "artifact_repo_path",
    ):
        _text(record[key], f"official_context.{key}")
    if record["config_repo_path"] is not None:
        _text(record["config_repo_path"], "official_context.config_repo_path")
    if (
        isinstance(record["official_rank"], bool)
        or not isinstance(record["official_rank"], int)
        or record["official_rank"] <= 0
        or isinstance(record["official_loss"], bool)
        or not isinstance(record["official_loss"], (int, float))
        or not math.isfinite(float(record["official_loss"]))
        or float(record["official_loss"]) < 0
        or record["repo_revision"] != source["revision"]
        or source_artifact["sha256"] == ""
    ):
        raise ValueError("official context has invalid rank/loss/source linkage")
    observations = _require_object(
        record["official_observations"], "official observations"
    )
    _exact_keys(
        observations,
        {"tournament_raw", "task_raw", "revision_manifest"},
        "official observations",
    )
    for name, raw_binding in observations.items():
        raw_binding = _require_object(raw_binding, f"official observation {name}")
        _exact_keys(raw_binding, {"name", "bytes", "sha256"}, f"observation {name}")
        if (
            Path(_text(raw_binding["name"], f"observation {name}.name")).name
            != raw_binding["name"]
            or isinstance(raw_binding["bytes"], bool)
            or not isinstance(raw_binding["bytes"], int)
            or raw_binding["bytes"] <= 0
            or not isinstance(raw_binding["sha256"], str)
            or not _SHA256.fullmatch(raw_binding["sha256"])
        ):
            raise ValueError(f"official observation {name} is invalid")
    ledger = _require_object(record["field_ledger"], "field ledger binding")
    _exact_keys(ledger, {"name", "bytes", "sha256"}, "field ledger binding")
    if (
        Path(_text(ledger["name"], "field ledger name")).name != ledger["name"]
        or isinstance(ledger["bytes"], bool)
        or not isinstance(ledger["bytes"], int)
        or ledger["bytes"] <= 0
        or not isinstance(ledger["sha256"], str)
        or not _SHA256.fullmatch(ledger["sha256"])
    ):
        raise ValueError("field ledger binding is invalid")
    linkage = _require_object(record["semantic_linkage"], "semantic linkage")
    expected_linkage = {
        "task_submission_unique": True,
        "rank_and_loss_match": True,
        "repository_revision_match": True,
        "artifact_path_and_lfs_match": True,
        "source_url_revision_match": True,
        "raw_official_task_match": True,
        "raw_tournament_round_match": True,
        "revision_capture_complete": True,
    }
    if linkage != expected_linkage:
        raise ValueError("official semantic linkage is incomplete")
    return record


def build_manifest(
    metadata: dict[str, Any],
    *,
    source_config_path: Path,
    source_artifact_path: Path,
    field_ledger_path: Path,
    task_raw_path: Path,
    tournament_raw_path: Path,
    revision_manifest_path: Path,
) -> dict[str, Any]:
    """Validate inputs and return a deterministic, self-digesting manifest."""

    metadata = _require_object(metadata, "metadata")
    _exact_keys(
        metadata,
        {
            "source_arm_id",
            "source",
            "official_context",
            "fields",
            "evaluator_sha",
            "matched_concept",
            "adaptation_target",
            "normalized_recipe",
            "review_assertion",
        },
        "metadata",
    )
    source_arm_id = _text(metadata["source_arm_id"], "source_arm_id")
    if not _SAFE_ID.fullmatch(source_arm_id) or source_arm_id in {".", ".."}:
        raise ValueError("source_arm_id must be one conservative path component")
    evaluator_sha = _evaluator_sha(metadata["evaluator_sha"])
    classified_fields = _fields(metadata["fields"])
    if classified_fields["adapted"]:
        raise ValueError("source provenance cannot contain local adapted-field choices")
    normalized_recipe = normalize_recipe(
        metadata["normalized_recipe"],
        classified_fields=classified_fields,
        source_only=True,
    )

    source_config_path = _safe_file(source_config_path, "source_config")
    source_artifact_path = _safe_file(source_artifact_path, "source_artifact")
    if source_config_path == source_artifact_path:
        raise ValueError("source config and source artifact must be distinct files")
    source = _source(metadata["source"])
    source_config_identity = _file_identity(source_config_path, "source_config")
    source_artifact_identity = _file_identity(source_artifact_path, "source_artifact")
    matched_concept = _matched_concept(metadata["matched_concept"])
    body: dict[str, Any] = {
        "schema": _SCHEMA,
        "kind": _KIND,
        "source_arm_id": source_arm_id,
        "source": source,
        "official_context": _official_context(
            metadata["official_context"],
            field_ledger_path=field_ledger_path,
            source=source,
            source_artifact=source_artifact_identity,
            source_config=source_config_identity,
            task_raw_path=task_raw_path,
            tournament_raw_path=tournament_raw_path,
            revision_manifest_path=revision_manifest_path,
        ),
        "files": {
            "source_config": source_config_identity,
            "source_artifact": source_artifact_identity,
        },
        "fields": classified_fields,
        "evaluator_sha": evaluator_sha,
        "matched_concept": matched_concept,
        "adaptation_target": _adaptation_target(
            metadata["adaptation_target"], matched_concept=matched_concept
        ),
        "normalized_recipe": normalized_recipe,
        "review_assertion": _review_assertion(metadata["review_assertion"]),
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def validate_manifest(
    manifest: dict[str, Any],
    *,
    source_config_path: Path | None = None,
    source_artifact_path: Path | None = None,
    field_ledger_path: Path | None = None,
    task_raw_path: Path | None = None,
    tournament_raw_path: Path | None = None,
    revision_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a persisted manifest and optionally re-bind its local files."""

    manifest = _require_object(manifest, "manifest")
    _exact_keys(
        manifest,
        {
            "schema",
            "kind",
            "source_arm_id",
            "source",
            "official_context",
            "files",
            "fields",
            "evaluator_sha",
            "matched_concept",
            "adaptation_target",
            "normalized_recipe",
            "review_assertion",
            "manifest_sha256",
        },
        "manifest",
    )
    digest = manifest["manifest_sha256"]
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("manifest_sha256 is invalid")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if canonical_sha256(body) != digest:
        raise ValueError("manifest_sha256 does not match canonical manifest body")
    if manifest["schema"] != _SCHEMA or manifest["kind"] != _KIND:
        raise ValueError("unsupported provenance schema or kind")

    files = _require_object(manifest["files"], "manifest.files")
    _exact_keys(files, {"source_config", "source_artifact"}, "manifest.files")
    for label in ("source_config", "source_artifact"):
        row = _require_object(files[label], f"manifest.files.{label}")
        _exact_keys(row, {"name", "bytes", "sha256"}, f"manifest.files.{label}")
        _text(row["name"], f"manifest.files.{label}.name")
        if Path(row["name"]).name != row["name"]:
            raise ValueError(f"manifest.files.{label}.name must be a basename")
        if (
            not isinstance(row["bytes"], int)
            or isinstance(row["bytes"], bool)
            or row["bytes"] < 0
        ):
            raise ValueError(f"manifest.files.{label}.bytes is invalid")
        if not isinstance(row["sha256"], str) or not _SHA256.fullmatch(row["sha256"]):
            raise ValueError(f"manifest.files.{label}.sha256 is invalid")

    # Reuse the semantic validators by reconstructing their metadata view.
    rebuilt_metadata = {
        "source_arm_id": manifest["source_arm_id"],
        "source": manifest["source"],
        "official_context": manifest["official_context"],
        "fields": manifest["fields"],
        "evaluator_sha": manifest["evaluator_sha"],
        "matched_concept": manifest["matched_concept"],
        "adaptation_target": manifest["adaptation_target"],
        "normalized_recipe": manifest["normalized_recipe"],
        "review_assertion": manifest["review_assertion"],
    }
    source_arm_id = _text(rebuilt_metadata["source_arm_id"], "source_arm_id")
    if not _SAFE_ID.fullmatch(source_arm_id) or source_arm_id in {".", ".."}:
        raise ValueError("invalid source_arm_id")
    if _source(rebuilt_metadata["source"]) != manifest["source"]:
        raise ValueError("manifest.source is not canonical")
    if (
        _validate_official_context_record(
            rebuilt_metadata["official_context"],
            source=manifest["source"],
            source_artifact=files["source_artifact"],
            source_config=files["source_config"],
            field_ledger_path=field_ledger_path,
            task_raw_path=task_raw_path,
            tournament_raw_path=tournament_raw_path,
            revision_manifest_path=revision_manifest_path,
        )
        != manifest["official_context"]
    ):
        raise ValueError("manifest.official_context is not canonical")
    if _fields(rebuilt_metadata["fields"]) != manifest["fields"]:
        raise ValueError("manifest.fields is not canonical")
    evaluator_sha = _evaluator_sha(rebuilt_metadata["evaluator_sha"])
    if evaluator_sha != manifest["evaluator_sha"]:
        raise ValueError("manifest.evaluator_sha is not canonical")
    matched_concept = _matched_concept(rebuilt_metadata["matched_concept"])
    if matched_concept != manifest["matched_concept"]:
        raise ValueError("manifest.matched_concept is not canonical")
    adaptation_target = _adaptation_target(
        rebuilt_metadata["adaptation_target"], matched_concept=matched_concept
    )
    if adaptation_target != manifest["adaptation_target"]:
        raise ValueError("manifest.adaptation_target is not canonical")
    recipe = normalize_recipe(
        rebuilt_metadata["normalized_recipe"],
        classified_fields=manifest["fields"],
        source_only=True,
    )
    if recipe != manifest["normalized_recipe"]:
        raise ValueError("manifest.normalized_recipe is not canonical")
    if (
        _review_assertion(rebuilt_metadata["review_assertion"])
        != manifest["review_assertion"]
    ):
        raise ValueError("manifest.review_assertion is not canonical")

    for label, path in (
        ("source_config", source_config_path),
        ("source_artifact", source_artifact_path),
    ):
        if path is None:
            continue
        safe = _safe_file(path, label)
        expected = files[label]
        identity = _file_identity(safe, label)
        if (
            identity["bytes"] != expected["bytes"]
            or identity["sha256"] != expected["sha256"]
        ):
            raise ValueError(f"{label} bytes do not match provenance")
    return manifest


def publish_exclusive(output: Path, manifest: dict[str, Any]) -> None:
    output = Path(os.path.abspath(os.path.expanduser(output)))
    current = output.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"provenance output has a symlink ancestor: {current}")
        current = current.parent
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{output}.tmp")
    if os.path.lexists(output) or os.path.lexists(temporary):
        raise FileExistsError(
            f"refusing stale provenance path: {output} or {temporary}"
        )
    payload = canonical_bytes(manifest) + b"\n"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output)
        temporary.unlink()
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        raise


def main() -> int:
    args = _parse()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    manifest = build_manifest(
        metadata,
        source_config_path=args.source_config,
        source_artifact_path=args.source_artifact,
        field_ledger_path=args.field_ledger,
        task_raw_path=args.task_raw,
        tournament_raw_path=args.tournament_raw,
        revision_manifest_path=args.revision_manifest,
    )
    publish_exclusive(args.output, manifest)
    print(canonical_bytes(manifest).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
