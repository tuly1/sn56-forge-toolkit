"""Run-scoped checkpoint discovery, selection, and promotion.

The validator reuses ``/app/checkpoints/{task_id}`` across retries.  ai-toolkit
writes deterministic filenames, so an unscoped directory listing can mistake a
previous attempt's ``last.safetensors`` or high-step periodic save for output
from the current attempt.  This module snapshots the directory before training
and only considers files whose filesystem signature changed afterwards.

Selection is intentionally conservative:

* A current-run ``forge_holdout_scores.json`` is authoritative when it declares
  ``source: heldout`` and scores valid current-run candidates.
* Otherwise, the exact final save wins unless the current run's ai-toolkit
  ``loss_log.db`` shows large, sustained late training-loss divergence.  That
  signal is explicitly recorded as a proxy, never as the validator metric.
* If the current run produces nothing valid, an intact pre-run
  ``last.safetensors`` may be retained, but only with an explicit fallback
  selection record and telemetry event.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
import re
import sqlite3
import statistics
import time
from typing import Any
import uuid

from forge.tasks.integrity import valid_safetensors

_SCOPE_FILE = ".forge_checkpoint_scope.json"
_SELECTION_FILE = "forge_checkpoint_selection.json"
_HOLDOUT_FILE = "forge_holdout_scores.json"
_LOSS_DB_FILE = "loss_log.db"
_SCOPE_SCHEMA = 2
_SELECTION_SCHEMA = 1
# ai-toolkit consumes these fixed-name files even when no repo-prefixed model
# checkpoint remains.  In particular, BaseSDTrainProcess loads ``optimizer.pt``
# unconditionally during optimizer setup; leaving it behind contaminates a
# validator retry with the previous attempt's momentum/variance state.
_AUTO_RESUME_FIXED_NAMES = frozenset({"optimizer.pt", "learnable_snr.json"})
_PROCESS_NONCE = uuid.uuid4().hex
_ACTIVE_RUNS: dict[str, dict[str, Any]] = {}
# Only names whose values are produced by the validator's exact scoring code may
# bypass proxy calibration. Unknown/self-declared metric names fail closed.
_EXACT_HELDOUT_METRICS = frozenset({"validator_exact_combined"})
# Frozen, consumer-owned promotion gates. A producer manifest cannot declare
# its own safety threshold. Entries are added only after exact Comfy evaluator
# calibration, keyed by (metric, model_type); an empty map is telemetry-only.
_HELDOUT_PROXY_POLICIES: dict[tuple[str, str], dict[str, Any]] = {}


@dataclass(frozen=True)
class Selection:
    path: str
    source: str
    reason: str
    step: int | None
    score: float | None = None
    metric: str | None = None
    direction: str | None = None
    is_training_loss_proxy: bool = False
    is_metric_proxy: bool = False
    reference_file: str | None = None
    reference_score: float | None = None
    score_advantage: float | None = None
    required_advantage: float | None = None
    margin_policy: str | None = None
    calibration_id: str | None = None


def begin_run(save_root: str, repo: str) -> dict[str, Any]:
    """Persist a pre-run inventory and return it.

    ai-toolkit itself auto-resumes from the newest direct child matching
    ``{repo}*``.  Merely filtering discovery after training is too late: a stale
    terminal save can make a retry perform zero steps and rewrite stale weights
    with a fresh signature.  Before launch we therefore move every such entry
    to a durable sibling quarantine.  A valid ``last.safetensors`` remains in
    the upload root as the only explicit prior-run fallback, so a hard kill at
    any later point still leaves a scoreable artifact.
    """
    if not repo or os.path.basename(repo) != repo or repo in (".", ".."):
        raise ValueError("checkpoint repo name must be one safe path component")
    key = os.path.abspath(save_root)
    attempt_nonce = uuid.uuid4().hex
    quarantine = os.path.join(_quarantine_root(save_root), attempt_nonce)
    state: dict[str, Any] = {
        "schema": _SCOPE_SCHEMA,
        "repo": repo,
        "process_nonce": _PROCESS_NONCE,
        "attempt_nonce": attempt_nonce,
        "started_unix": time.time(),
        "before": {},
        "quarantine": quarantine,
        "quarantine_complete": False,
    }
    # Install the incomplete attempt in memory before *any* filesystem work.
    # If even mkdir or the first journal write fails, the handler cannot revive
    # a complete scope left by a previous process and train on its optimizer.
    _ACTIVE_RUNS[key] = state
    os.makedirs(save_root, exist_ok=True)
    scope_path = os.path.join(save_root, _SCOPE_FILE)
    # Atomic replacement is the durable tombstone. If persistence is full, the
    # in-process state still fails closed, and a later process ignores the old
    # journal because its process nonce cannot match.
    try:
        _atomic_json(scope_path, state)
    except Exception as exc:
        _event("checkpoint_scope_persist_failed", error=f"{type(exc).__name__}: {exc}")

    _ensure_prior_last(save_root, repo)
    _prune_old_quarantines(save_root)
    before = {
        name: sig
        for name in _tracked_names(save_root)
        if (sig := _signature(os.path.join(save_root, name))) is not None
    }
    state["before"] = before
    _ACTIVE_RUNS[key] = state
    try:
        _atomic_json(scope_path, state)
    except Exception as exc:
        _event("checkpoint_scope_persist_failed", error=f"{type(exc).__name__}: {exc}")

    # Quarantine every direct name ai-toolkit can consume as prior training
    # state: repo-prefixed model/state entries plus its fixed-name optimizer and
    # learnable-SNR files.  This operation is same-filesystem os.replace plus
    # directory fsync; a partial failure aborts the handler rather than allowing
    # contaminated auto-resume.
    moved: list[str] = []
    try:
        candidates = sorted(
            name
            for name in os.listdir(save_root)
            if (
                name.startswith(repo) or name in _AUTO_RESUME_FIXED_NAMES
            )
            and name != os.path.basename(scope_path)
        )
        if candidates:
            os.makedirs(quarantine, exist_ok=True)
            _fsync_parent(os.path.join(quarantine, ".sentinel"))
        for name in candidates:
            source = os.path.join(save_root, name)
            destination = os.path.join(quarantine, name)
            os.replace(source, destination)
            moved.append(name)
            _fsync_parent(source)
            _fsync_parent(destination)
        state["quarantined"] = moved
        state["quarantine_complete"] = True
        _ACTIVE_RUNS[key] = state
        _atomic_json(scope_path, state)
    except BaseException:
        state["quarantined"] = moved
        _ACTIVE_RUNS[key] = state
        try:
            _atomic_json(scope_path, state)
        except Exception:
            pass
        raise
    _event(
        "checkpoint_scope_started",
        prior_files=len([n for n in before if n.endswith(".safetensors")]),
        prior_last="last.safetensors" in before,
        quarantined=len(moved),
    )
    return state


def ensure_run(save_root: str, repo: str) -> dict[str, Any]:
    """Reuse only this process's CLI scope, or create a direct-call scope.

    A completed journal found only on disk belongs to an earlier process and is
    never proof that this retry quarantined its own auto-resume inputs.
    """
    state = _ACTIVE_RUNS.get(os.path.abspath(save_root))
    if state is not None:
        if state.get("repo") != repo:
            raise RuntimeError("active checkpoint scope belongs to another repo")
        if state.get("process_nonce") != _PROCESS_NONCE:
            raise RuntimeError("active checkpoint scope belongs to another process")
        if state.get("quarantine_complete") is not True:
            raise RuntimeError("checkpoint scope initialization is incomplete")
        return state
    return begin_run(save_root, repo)


def load_run(save_root: str) -> dict[str, Any] | None:
    active = _ACTIVE_RUNS.get(os.path.abspath(save_root))
    if active is not None:
        return active if _scope_is_current_process(active) else None
    try:
        with open(os.path.join(save_root, _SCOPE_FILE), encoding="utf-8") as fh:
            state = json.load(fh)
        return state if _scope_is_current_process(state) else None
    except Exception:
        return None


def _scope_is_current_process(state: Any) -> bool:
    return bool(
        isinstance(state, dict)
        and state.get("schema") == _SCOPE_SCHEMA
        and state.get("process_nonce") == _PROCESS_NONCE
        and isinstance(state.get("attempt_nonce"), str)
        and bool(state["attempt_nonce"])
        and isinstance(state.get("before"), dict)
    )


def _scope_is_complete(state: Any) -> bool:
    return _scope_is_current_process(state) and state.get("quarantine_complete") is True


def set_planned_steps(
    save_root: str,
    state: dict[str, Any],
    steps: int,
    *,
    model_type: str | None = None,
) -> dict[str, Any]:
    """Add the planned terminal step so an unnumbered exact final is traceable."""
    updated = dict(state)
    try:
        updated["planned_steps"] = max(1, int(steps))
        if model_type:
            updated["model_type"] = str(model_type).strip().lower()
        _ACTIVE_RUNS[os.path.abspath(save_root)] = updated
        _atomic_json(os.path.join(save_root, _SCOPE_FILE), updated)
    except Exception as exc:
        _event("checkpoint_scope_plan_failed", error=f"{type(exc).__name__}: {exc}")
    return updated


def current_loras(save_root: str, state: dict[str, Any] | None) -> list[str]:
    """Return only validly named LoRAs created or replaced in this run."""
    if not _scope_is_complete(state) or not os.path.isdir(save_root):
        return []
    repo = str(state.get("repo") or "")
    periodic = re.compile(rf"^{re.escape(repo)}_\d+\.safetensors$")
    out: list[str] = []
    for name in os.listdir(save_root):
        if name != f"{repo}.safetensors" and periodic.fullmatch(name) is None:
            continue
        path = os.path.join(save_root, name)
        if os.path.isfile(path) and _is_current(path, state):
            out.append(path)
    return sorted(out)


def finalize(
    save_root: str,
    repo: str,
    state: dict[str, Any] | None = None,
    *,
    context: str = "training",
) -> dict[str, Any] | None:
    """Promote one current-run checkpoint, or explicitly retain a prior last.

    Returns the persisted selection record.  ``None`` means that neither the
    current run nor a prior attempt left a valid artifact.
    """
    state = state or load_run(save_root)
    if state is not None and not _scope_is_complete(state):
        _event("checkpoint_scope_incomplete", context=context)
        return None
    if state is not None and state.get("repo") != repo:
        _event(
            "checkpoint_scope_repo_mismatch",
            expected_repo=repo,
            scoped_repo=state.get("repo"),
        )
        state = None
    candidates = current_loras(save_root, state)
    valid = [path for path in candidates if valid_safetensors(path)]

    if valid:
        selection = select(valid, repo, save_root, state)
        last = os.path.join(save_root, "last.safetensors")
        promoted_sha256 = _atomic_copy(selection.path, last)
        record = _selection_record(
            selection,
            output_path=last,
            output_sha256=promoted_sha256,
            status="selected_current_run",
            context=context,
            discovered=len(candidates),
            valid=len(valid),
        )
        _write_selection(save_root, record)
        _event(
            "checkpoint_selected",
            source=selection.source,
            step=selection.step,
            sha256=record["sha256"],
            current_candidates=len(candidates),
        )
        _cleanup_quarantine(state)
        return record

    last = os.path.join(save_root, "last.safetensors")
    if _is_prior_valid_last(last, state):
        selection = Selection(
            path=last,
            source="previous_run_fallback",
            reason=(
                "current run produced no valid checkpoint; retained the valid "
                "last.safetensors that existed before this run"
            ),
            step=None,
        )
        record = _selection_record(
            selection,
            output_path=last,
            output_sha256=None,
            status="preserved_previous_run",
            context=context,
            discovered=len(candidates),
            valid=0,
        )
        _write_selection(save_root, record)
        _event(
            "checkpoint_previous_run_fallback",
            sha256=record["sha256"],
            current_candidates=len(candidates),
        )
        _cleanup_quarantine(state)
        return record

    _event(
        "checkpoint_unavailable",
        context=context,
        current_candidates=len(candidates),
    )
    return None


def select(
    valid: list[str],
    repo: str,
    save_root: str,
    state: dict[str, Any] | None,
) -> Selection:
    """Choose among already integrity-checked current-run candidates."""
    default = _default_selection(valid, repo, state)

    heldout = _select_from_holdout(valid, save_root, state, default)
    if heldout is not None:
        return heldout

    divergence = _select_from_loss_divergence(valid, default, save_root, state)
    if divergence is not None:
        return divergence
    return default


def _default_selection(
    valid: list[str], repo: str, state: dict[str, Any] | None
) -> Selection:
    exact = os.path.join(os.path.dirname(valid[0]), f"{repo}.safetensors")
    if exact in valid:
        return Selection(
            path=exact,
            source="exact_final",
            reason=(
                "no authoritative held-out selection and no clear sustained "
                "late training-loss divergence; selected ai-toolkit's exact final"
            ),
            step=_planned_steps(state),
        )
    chosen = max(valid, key=lambda path: (_step_of(path), os.path.basename(path)))
    return Selection(
        path=chosen,
        source="highest_valid_periodic",
        reason=(
            "exact final was unavailable; selected the highest-step valid "
            "current-run periodic checkpoint"
        ),
        step=_step_of(chosen),
    )


def _select_from_holdout(
    valid: list[str],
    save_root: str,
    state: dict[str, Any] | None,
    default: Selection,
) -> Selection | None:
    path = os.path.join(save_root, _HOLDOUT_FILE)
    if not _is_current(path, state):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("schema") != 1:
            raise ValueError("schema must be 1")
        if data.get("source") != "heldout":
            raise ValueError("source must be 'heldout'")
        if data.get("complete") is not True:
            raise ValueError("complete must be true")
        direction = str(data.get("direction", "min")).lower()
        if direction not in ("min", "max"):
            raise ValueError("direction must be min or max")
        metric = str(data.get("metric") or "heldout_score")
        is_proxy = (
            data.get("proxy_not_validator_metric") is True
            or metric not in _EXACT_HELDOUT_METRICS
        )
        by_name = {os.path.basename(path): path for path in valid}
        rows = data.get("scores")
        if not isinstance(rows, list):
            raise ValueError("scores must be a list")
        scored: list[tuple[float, str, int | None]] = []
        row_by_path: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        for row in rows:
            try:
                name = os.path.basename(
                    str(row.get("checkpoint") or row.get("file") or "")
                )
                candidate = by_name.get(name)
                if not name or name in seen:
                    raise ValueError("checkpoint names must be unique")
                seen.add(name)
                score = float(row.get("score"))
                declared_sha = str(row.get("sha256") or "").lower()
                if (
                    candidate is None
                    or not math.isfinite(score)
                    or not re.fullmatch(r"[0-9a-f]{64}", declared_sha)
                    or _sha256(candidate) != declared_sha
                ):
                    raise ValueError(f"invalid score/hash for {name!r}")
                step = row.get("step")
                inferred_step = _step_of(candidate)
                if inferred_step < 0:
                    inferred_step = _planned_steps(state)
                try:
                    step = int(step) if step is not None else inferred_step
                except (TypeError, ValueError):
                    step = inferred_step
                scored.append(
                    (
                        score,
                        candidate,
                        step if step is not None and step >= 0 else None,
                    )
                )
                row_by_path[candidate] = row
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid heldout score row: {exc}") from exc
        if seen != set(by_name):
            raise ValueError("scores must cover every valid current-run checkpoint")
        if not scored:
            raise ValueError("no finite scores for valid current-run checkpoints")
        score, candidate, step = (
            min(scored, key=lambda item: item[0])
            if direction == "min"
            else max(scored, key=lambda item: item[0])
        )
        default_row = next(
            (row for row in scored if row[1] == default.path),
            None,
        )
        if default_row is None:
            raise ValueError("default current-run checkpoint has no score")
        default_score, _default_path, _default_step = default_row
        advantage = (
            default_score - score if direction == "min" else score - default_score
        )

        if is_proxy:
            if data.get("proxy_not_validator_metric") is not True:
                _event(
                    "heldout_proxy_telemetry_only",
                    reason="proxy disclosure was missing or false",
                )
                return None
            model_type = str(data.get("model_type") or "").strip().lower()
            policy = _HELDOUT_PROXY_POLICIES.get((metric, model_type))
            if policy is None:
                _event(
                    "heldout_proxy_telemetry_only",
                    reason=(
                        "no frozen calibration policy exists for "
                        f"{model_type or 'unknown'}"
                    ),
                )
                return None
            scoped_model_type = str(
                (state or {}).get("model_type") or ""
            ).strip().lower()
            if not scoped_model_type or model_type != scoped_model_type:
                _event(
                    "heldout_proxy_telemetry_only",
                    reason="manifest model type is not bound to this run scope",
                )
                return None
            allowed_sources = policy.get("reference_sources", ("exact_final",))
            if (
                not isinstance(allowed_sources, (list, tuple, set, frozenset))
                or default.source not in {str(value) for value in allowed_sources}
            ):
                _event(
                    "heldout_proxy_telemetry_only",
                    reason=f"policy does not cover reference source {default.source}",
                )
                return None
            try:
                _validate_proxy_contract(policy, data, row_by_path)
            except (KeyError, TypeError, ValueError) as exc:
                _event(
                    "heldout_proxy_telemetry_only",
                    reason=f"proxy contract invalid: {exc}",
                )
                return None
            if candidate == default.path:
                return Selection(
                    path=default.path,
                    source="heldout_manifest",
                    reason=(
                        f"the conservative default is also best on {metric}; "
                        "no proxy-based checkpoint change was needed"
                    ),
                    step=default.step,
                    score=default_score,
                    metric=metric,
                    direction=direction,
                    is_metric_proxy=True,
                    reference_file=os.path.basename(default.path),
                    reference_score=default_score,
                    score_advantage=0.0,
                    required_advantage=0.0,
                    margin_policy=str(policy.get("name") or "calibrated_proxy_margin"),
                    calibration_id=str(policy.get("calibration_id") or ""),
                )
            try:
                required = _proxy_required_advantage(
                    policy,
                    data,
                    row_by_path[candidate],
                    row_by_path[default.path],
                    default_score,
                )
            except (KeyError, TypeError, ValueError) as exc:
                _event(
                    "heldout_proxy_telemetry_only",
                    reason=f"calibrated-gate evidence was invalid: {exc}",
                )
                return None
            if advantage <= required:
                return _guarded_proxy_default(
                    default,
                    metric,
                    direction,
                    default_score,
                    "proxy advantage did not strictly exceed the calibrated gate",
                    advantage=advantage,
                    required=required,
                    policy=policy,
                )
            return Selection(
                path=candidate,
                source="heldout_manifest",
                reason=(
                    f"selected best {metric}; proxy advantage {advantage:.8g} "
                    f"strictly exceeded calibrated gate {required:.8g}"
                ),
                step=step,
                score=score,
                metric=metric,
                direction=direction,
                is_metric_proxy=True,
                reference_file=os.path.basename(default.path),
                reference_score=default_score,
                score_advantage=advantage,
                required_advantage=required,
                margin_policy=str(policy.get("name") or "calibrated_proxy_margin"),
                calibration_id=str(policy.get("calibration_id") or ""),
            )

        # Non-proxy held-out metrics retain the pre-Gate-B exact-score contract.
        if candidate != default.path and advantage == 0.0:
            return Selection(
                path=default.path,
                source="heldout_manifest_near_tie",
                reason=(
                    f"{metric} tied the conservative default exactly; retained "
                    "the exact-final/default checkpoint"
                ),
                step=default.step,
                score=default_score,
                metric=metric,
                direction=direction,
            )
        return Selection(
            path=candidate,
            source="heldout_manifest",
            reason=(
                f"selected best {metric} from current-run {_HOLDOUT_FILE}; "
                "held-out scores take precedence over final weights"
            ),
            step=step,
            score=score,
            metric=metric,
            direction=direction,
        )
    except Exception as exc:
        _event("holdout_manifest_ignored", error=f"{type(exc).__name__}: {exc}")
        return None


def _guarded_proxy_default(
    default: Selection,
    metric: str,
    direction: str,
    default_score: float,
    detail: str,
    *,
    advantage: float | None = None,
    required: float | None = None,
    policy: dict[str, Any] | None = None,
) -> Selection:
    return Selection(
        path=default.path,
        source="heldout_proxy_guarded_default",
        reason=f"retained conservative default because {detail}",
        step=default.step,
        score=default_score,
        metric=metric,
        direction=direction,
        is_metric_proxy=True,
        reference_file=os.path.basename(default.path),
        reference_score=default_score,
        score_advantage=advantage,
        required_advantage=required,
        margin_policy=(str(policy.get("name")) if policy else None),
        calibration_id=(str(policy.get("calibration_id")) if policy else None),
    )


def _validate_proxy_contract(
    policy: dict[str, Any],
    manifest: dict[str, Any],
    rows: dict[str, dict[str, Any]],
) -> None:
    direction = str(manifest.get("direction") or "").lower()
    if direction != str(policy.get("direction") or "").lower():
        raise ValueError("direction does not match frozen policy")
    captioned_weight = float(manifest.get("captioned_weight"))
    blank_weight = float(manifest.get("blank_caption_weight"))
    expected_captioned = float(policy.get("captioned_weight"))
    expected_blank = float(policy.get("blank_caption_weight"))
    if (
        not math.isclose(captioned_weight, expected_captioned, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(blank_weight, expected_blank, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(captioned_weight + blank_weight, 1.0, abs_tol=1e-12)
        or manifest.get("strata_scored_separately") is not True
    ):
        raise ValueError("stratum weighting does not match frozen policy")
    epochs = int(manifest.get("probe_epochs"))
    if epochs != int(policy.get("probe_epochs")) or epochs <= 0:
        raise ValueError("probe epoch count does not match frozen policy")
    seed = int(manifest.get("seed"))
    if seed != int(policy.get("seed")):
        raise ValueError("probe seed does not match frozen policy")
    holdout_pairs = int(manifest.get("holdout_pairs"))
    minimum_pairs = max(1, int(policy.get("min_holdout_pairs")))
    maximum_pairs = int(policy.get("max_holdout_pairs"))
    if not minimum_pairs <= holdout_pairs <= maximum_pairs:
        raise ValueError("holdout pair count is outside frozen policy")
    expected_points = holdout_pairs * epochs
    for row in rows.values():
        captioned_score = float(row.get("captioned_score"))
        blank_score = float(row.get("blank_caption_score"))
        combined = float(row.get("score"))
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (captioned_score, blank_score, combined)
        ):
            raise ValueError("proxy component score is invalid")
        recomputed = (
            captioned_weight * captioned_score + blank_weight * blank_score
        )
        if not math.isclose(combined, recomputed, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("combined proxy score does not match its strata")
        if (
            int(row.get("captioned_points")) != expected_points
            or int(row.get("blank_caption_points")) != expected_points
            or int(row.get("points")) != expected_points * 2
        ):
            raise ValueError("proxy point count does not match holdout contract")
        for key in ("captioned_stddev", "blank_caption_stddev"):
            spread = float(row.get(key))
            if not math.isfinite(spread) or spread < 0.0:
                raise ValueError("proxy dispersion field is invalid")


def _proxy_required_advantage(
    policy: dict[str, Any],
    manifest: dict[str, Any],
    best_row: dict[str, Any],
    default_row: dict[str, Any],
    default_score: float,
) -> float:
    holdout_pairs = int(manifest.get("holdout_pairs"))
    minimum_pairs = int(policy.get("min_holdout_pairs", 1))
    if holdout_pairs < max(1, minimum_pairs):
        raise ValueError("insufficient independent held-out pairs for proxy policy")

    def _variance(row: dict[str, Any]) -> float:
        captioned_sd = float(row.get("captioned_stddev"))
        blank_sd = float(row.get("blank_caption_stddev"))
        captioned_points = int(row.get("captioned_points"))
        blank_points = int(row.get("blank_caption_points"))
        values = (captioned_sd, blank_sd)
        if (
            any(not math.isfinite(value) or value < 0.0 for value in values)
            or captioned_points <= 0
            or blank_points <= 0
        ):
            raise ValueError("proxy row has invalid dispersion fields")
        # Repeated loss points are not independent images; never inflate n_eff
        # beyond the number of genuinely held-out pairs.
        captioned_n = min(holdout_pairs, captioned_points)
        blank_n = min(holdout_pairs, blank_points)
        captioned_weight = float(policy.get("captioned_weight"))
        blank_weight = float(policy.get("blank_caption_weight"))
        return (
            (captioned_weight ** 2) * (captioned_sd ** 2) / captioned_n
            + (blank_weight ** 2) * (blank_sd ** 2) / blank_n
        )

    absolute = float(policy.get("absolute_floor"))
    relative = float(policy.get("relative_floor"))
    multiplier = float(policy.get("dispersion_multiplier"))
    if (
        any(not math.isfinite(value) or value < 0.0
            for value in (absolute, relative, multiplier))
        or relative > 1.0
    ):
        raise ValueError("invalid frozen proxy policy")
    dispersion = multiplier * math.sqrt(
        _variance(best_row) + _variance(default_row)
    )
    return max(absolute, relative * abs(default_score), dispersion)


def _select_from_loss_divergence(
    valid: list[str],
    default: Selection,
    save_root: str,
    state: dict[str, Any] | None,
) -> Selection | None:
    """Use raw loss only for an unusually large and sustained regression.

    This is a forfeit/overfit guard, not a substitute for the tournament metric.
    It requires at least 30 current-run loss points, two periodic candidates,
    >=35% degradation versus the best candidate window, a three-MAD absolute
    separation, and two consecutive late windows >=25% above the best window.
    """
    points = _loss_points(save_root, state)
    stepped = [(path, _step_of(path)) for path in valid if _step_of(path) >= 0]
    if len(points) < 30 or len(stepped) < 2:
        return None

    max_loss_step = max(step for step, _loss in points)
    default_step = default.step if default.step is not None else max_loss_step
    window = max(8, min(32, len(points) // 12))

    scored: list[tuple[float, float, str, int]] = []
    for path, step in stepped:
        values = [loss for point_step, loss in points if step - window < point_step <= step]
        if len(values) >= max(5, window // 2):
            scored.append((_trimmed_mean(values), _mad(values), path, step))
    if not scored:
        return None
    best_mean, best_mad, best_path, best_step = min(scored, key=lambda row: row[0])
    if default_step - best_step < 2 * window:
        return None

    late_windows: list[float] = []
    for end in (default_step - window, default_step):
        vals = [loss for step, loss in points if end - window < step <= end]
        if len(vals) < max(5, window // 2):
            return None
        late_windows.append(_trimmed_mean(vals))
    late_mean = late_windows[-1]
    separation = late_mean - best_mean
    noise_floor = max(0.01, 3.0 * best_mad)
    clear_ratio = late_mean >= best_mean * 1.35
    clear_absolute = separation >= noise_floor
    sustained = all(mean >= best_mean * 1.25 for mean in late_windows)
    if not (clear_ratio and clear_absolute and sustained):
        return None

    return Selection(
        path=best_path,
        source="training_loss_divergence",
        reason=(
            "raw ai-toolkit training loss showed clear sustained late divergence "
            f"(best-window={best_mean:.6g} at step {best_step}, "
            f"late-window={late_mean:.6g}); this is a conservative proxy, not "
            "the validator's held-out metric"
        ),
        step=best_step,
        score=best_mean,
        metric="raw_training_loss_window",
        direction="min",
        is_training_loss_proxy=True,
    )


def _loss_points(
    save_root: str, state: dict[str, Any] | None
) -> list[tuple[int, float]]:
    path = os.path.join(save_root, _LOSS_DB_FILE)
    wal_path = path + "-wal"
    if state is None or not (
        _is_current(path, state) or _is_current(wal_path, state)
    ):
        return []
    try:
        # ``mode=ro`` prevents selection from mutating or creating the recorder.
        uri = f"file:{os.path.abspath(path)}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=1.0) as conn:
            rows = conn.execute(
                """
                SELECT m.step, m.value_real
                  FROM metrics AS m
                  JOIN steps AS s ON s.step = m.step
                 WHERE m.key = 'loss/loss'
                   AND s.wall_time >= ?
                 ORDER BY m.step
                """,
                (float(state.get("started_unix", 0.0)) - 2.0,),
            ).fetchall()
        out = []
        for step, loss in rows:
            step, loss = int(step), float(loss)
            if step >= 0 and math.isfinite(loss):
                out.append((step, loss))
        return out
    except Exception as exc:
        _event("loss_selection_unavailable", error=f"{type(exc).__name__}: {exc}")
        return []


def _trimmed_mean(values: list[float]) -> float:
    ordered = sorted(values)
    trim = int(len(ordered) * 0.1)
    core = ordered[trim : len(ordered) - trim] if trim and len(ordered) > 2 * trim else ordered
    return statistics.fmean(core)


def _mad(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def _selection_record(
    selection: Selection,
    *,
    output_path: str,
    output_sha256: str | None,
    status: str,
    context: str,
    discovered: int,
    valid: int,
) -> dict[str, Any]:
    return {
        "schema": _SELECTION_SCHEMA,
        "status": status,
        "context": context,
        "source": selection.source,
        "selected_file": os.path.basename(selection.path),
        "output_file": os.path.basename(output_path),
        "selected_step": selection.step,
        "sha256": output_sha256 or _sha256(output_path),
        "reason": selection.reason,
        "score": selection.score,
        "metric": selection.metric,
        "direction": selection.direction,
        "training_loss_is_proxy_not_validator_metric": selection.is_training_loss_proxy,
        "metric_is_proxy_not_validator_metric": selection.is_metric_proxy,
        "reference_file": selection.reference_file,
        "reference_score": selection.reference_score,
        "score_advantage": selection.score_advantage,
        "required_advantage": selection.required_advantage,
        "margin_policy": selection.margin_policy,
        "calibration_id": selection.calibration_id,
        "current_candidates_discovered": discovered,
        "current_candidates_valid": valid,
        "created_unix": int(time.time()),
    }


def _write_selection(save_root: str, record: dict[str, Any]) -> None:
    _atomic_json(os.path.join(save_root, _SELECTION_FILE), record)


def _is_prior_valid_last(path: str, state: dict[str, Any] | None) -> bool:
    if state is None or "last.safetensors" not in state.get("before", {}):
        return False
    # It must still be byte/file-identical to the pre-run artifact.  If it was
    # modified during this run, it is neither a trusted prior nor a candidate.
    return not _is_current(path, state) and valid_safetensors(path)


def _tracked_names(save_root: str) -> list[str]:
    return [
        name
        for name in os.listdir(save_root)
        if name.endswith(".safetensors")
        or name
        in (
            _HOLDOUT_FILE,
            _LOSS_DB_FILE,
            _LOSS_DB_FILE + "-wal",
            _LOSS_DB_FILE + "-shm",
        )
    ]


def _ensure_prior_last(save_root: str, repo: str) -> None:
    """Create the kill-safe fallback before hiding ai-toolkit resume inputs."""
    last = os.path.join(save_root, "last.safetensors")
    if valid_safetensors(last):
        return
    try:
        candidates = [
            os.path.join(save_root, name)
            for name in os.listdir(save_root)
            if name.startswith(repo)
            and name.endswith(".safetensors")
            and os.path.isfile(os.path.join(save_root, name))
            and valid_safetensors(os.path.join(save_root, name))
        ]
    except OSError:
        candidates = []
    if not candidates:
        return
    exact = os.path.join(save_root, f"{repo}.safetensors")
    if exact in candidates:
        chosen = exact
    else:
        numbered = [path for path in candidates if _step_of(path) >= 0]
        chosen = (
            max(numbered, key=lambda path: (_step_of(path), path))
            if numbered
            else max(candidates, key=lambda path: (os.path.getctime(path), path))
        )
    digest = _atomic_copy(chosen, last)
    _event(
        "checkpoint_prior_last_created",
        source=os.path.basename(chosen),
        sha256=digest,
    )


def _cleanup_quarantine(state: dict[str, Any] | None) -> None:
    path = str((state or {}).get("quarantine") or "")
    if not path:
        return
    try:
        import shutil

        shutil.rmtree(path)
        parent = os.path.dirname(path)
        _fsync_parent(os.path.join(parent, ".sentinel"))
        try:
            os.rmdir(parent)
        except OSError:
            pass
    except FileNotFoundError:
        pass
    except Exception as exc:
        _event("checkpoint_quarantine_cleanup_failed", error=f"{type(exc).__name__}: {exc}")


def _prune_old_quarantines(save_root: str) -> None:
    """Remove completed/abandoned prior quarantine copies after fallback exists."""
    root = _quarantine_root(save_root)
    if not os.path.isdir(root):
        return
    try:
        import shutil

        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
        _fsync_parent(os.path.join(root, ".sentinel"))
        try:
            os.rmdir(root)
        except OSError:
            pass
    except Exception as exc:
        # These are sibling directories, never ai-toolkit resume inputs. A
        # cleanup failure costs disk only and must not block quarantining the
        # current attempt's live inputs.
        _event(
            "checkpoint_quarantine_prune_failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def _quarantine_root(save_root: str) -> str:
    absolute = os.path.abspath(save_root)
    suffix = hashlib.sha256(absolute.encode()).hexdigest()[:12]
    return os.path.join(
        os.path.dirname(absolute), f".forge-checkpoint-quarantine-{suffix}"
    )


def _signature(path: str) -> dict[str, int] | None:
    """Cheap run-boundary identity without hashing multi-GB checkpoints.

    Size, inode, mtime, and ctime detect normal ai-toolkit rewrites, including an
    in-place same-size rewrite whose mtime is restored.  On an exotic filesystem
    that exposes neither high-resolution mtime nor ctime, an overwrite within one
    timestamp tick could be classified as prior-run.  That failure is deliberate
    and fail-closed: the new candidate is ignored instead of a stale file being
    promoted.  We avoid hashing every pre-run checkpoint because that would burn
    the same I/O budget this policy is intended to recover.
    """
    try:
        stat = os.stat(path)
        if not os.path.isfile(path):
            return None
        return {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "ctime_ns": int(stat.st_ctime_ns),
            "inode": int(stat.st_ino),
        }
    except Exception:
        return None


def _is_current(path: str, state: dict[str, Any] | None) -> bool:
    if state is None:
        return False
    now = _signature(path)
    if now is None:
        return False
    before = state.get("before", {}).get(os.path.basename(path))
    return before is None or before != now


def _step_of(path: str) -> int:
    match = re.search(r"_(\d+)\.safetensors$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_copy(src: str, dst: str) -> str:
    """Atomically copy and hash in one pass; return the promoted SHA-256."""
    tmp = dst + ".tmp"
    try:
        digest = hashlib.sha256()
        with open(src, "rb") as source, open(tmp, "wb") as target:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                target.write(block)
                digest.update(block)
            target.flush()
            os.fsync(target.fileno())
        os.replace(tmp, dst)
        _fsync_parent(dst)
        return digest.hexdigest()
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise


def _planned_steps(state: dict[str, Any] | None) -> int | None:
    try:
        value = int((state or {}).get("planned_steps"))
        return value if value > 0 else None
    except (TypeError, ValueError):
        return None


def _atomic_json(path: str, value: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(value, fh, sort_keys=True, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_parent(path)
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise


def _fsync_parent(path: str) -> None:
    """Durably commit a rename; tolerate filesystems that reject directory fsync."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(os.path.dirname(os.path.abspath(path)), flags)
    try:
        try:
            os.fsync(fd)
        except OSError as exc:
            unsupported = {
                errno.EINVAL,
                getattr(errno, "ENOTSUP", errno.EINVAL),
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            }
            if exc.errno not in unsupported:
                raise
    finally:
        os.close(fd)


def _event(name: str, **values: Any) -> None:
    try:
        from forge import telemetry

        telemetry.event(name, **values)
    except Exception:
        pass
