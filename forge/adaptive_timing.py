"""Measured, bundle-bound timing profiles for opt-in training experiments.

The incumbent recipe continues to use :mod:`forge.recipe`'s literal timing
constants unless a caller explicitly loads and supplies one of these profiles.
Experimental bundles are different: their caller must set ``required=True`` so
a missing, malformed, or wrongly bound profile stops the experiment instead of
silently running it with another bundle's timing assumption.

The first-checkpoint correction in this module is deliberately observational.
ai-toolkit receives a fixed step count when its subprocess starts; seeing a
faster first checkpoint cannot extend that active process.  The correction
therefore records the active plan unchanged and recommends a budget cap only
for a future launch.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping


PROFILE_ENV = "FORGE_KREA_THROUGHPUT_PROFILE"
PROFILE_KIND = "forge-measured-throughput-profile"
PROFILE_SCHEMA = 1
FIRST_CHECKPOINT_EVENT = "first_checkpoint_timing_observed"
_MAX_PROFILE_BYTES = 64 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_BUNDLE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")

_PROFILE_FIELDS = {
    "schema",
    "kind",
    "bundle_id",
    "bundle_sha256",
    "model_type",
    "seconds_per_step",
    "startup_seconds",
    "measurement",
    "provenance",
    "profile_sha256",
}
_MEASUREMENT_FIELDS = {
    "completed_steps",
    "training_elapsed_seconds",
    "first_checkpoint_step",
    "first_checkpoint_elapsed_seconds",
}
_PROVENANCE_FIELDS = {
    "source_run_id",
    "source_record_sha256",
    "runtime_commit",
    "measured_at_utc",
    "accelerator",
}


class TimingProfileError(RuntimeError):
    """A measured timing profile is absent, invalid, or bound elsewhere."""


@dataclass(frozen=True)
class ThroughputProfile:
    """A validated measurement for one exact experimental bundle.

    ``training_elapsed_seconds`` and ``first_checkpoint_elapsed_seconds`` are
    measured from subprocess launch.  ``startup_seconds`` is the measured
    pre-optimizer portion.  The declared rate must equal
    ``(training_elapsed_seconds - startup_seconds) / completed_steps`` within
    two percent, so a self-consistent JSON hash cannot turn a guess into a
    claimed measurement.
    """

    bundle_id: str
    bundle_sha256: str
    model_type: str
    seconds_per_step: float
    startup_seconds: float
    completed_steps: int
    training_elapsed_seconds: float
    first_checkpoint_step: int
    first_checkpoint_elapsed_seconds: float
    source_run_id: str
    source_record_sha256: str
    runtime_commit: str
    measured_at_utc: str
    accelerator: str
    profile_sha256: str


@dataclass(frozen=True)
class FirstCheckpointCorrection:
    """Observed timing correction whose recommendation applies to future runs."""

    bundle_id: str
    profile_sha256: str
    checkpoint_step: int
    elapsed_since_launch_s: float
    profiled_seconds_per_step: float
    observed_seconds_per_step: float
    observed_to_profile_ratio: float
    correction: str
    active_planned_steps: int
    active_plan_mutable: bool
    active_plan_action: str
    active_plan_exceeds_observed_budget: bool
    future_budget_cap_steps: int
    future_target_steps: int
    future_recommended_steps: int
    future_step_delta: int

    def telemetry_fields(self) -> dict[str, Any]:
        """Return a stable, strategy-light event payload."""

        return {
            "bundle_id": self.bundle_id,
            "timing_profile_sha256": self.profile_sha256,
            "checkpoint_step": self.checkpoint_step,
            "elapsed_since_launch_s": round(self.elapsed_since_launch_s, 3),
            "profiled_seconds_per_step": round(
                self.profiled_seconds_per_step, 6
            ),
            "observed_seconds_per_step": round(
                self.observed_seconds_per_step, 6
            ),
            "observed_to_profile_ratio": round(
                self.observed_to_profile_ratio, 6
            ),
            "correction": self.correction,
            "active_planned_steps": self.active_planned_steps,
            "active_plan_mutable": self.active_plan_mutable,
            "active_plan_action": self.active_plan_action,
            "active_plan_exceeds_observed_budget": (
                self.active_plan_exceeds_observed_budget
            ),
            "future_budget_cap_steps": self.future_budget_cap_steps,
            "future_target_steps": self.future_target_steps,
            "future_recommended_steps": self.future_recommended_steps,
            "future_step_delta": self.future_step_delta,
        }


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value using the project's canonical JSON convention."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seal_profile_document(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with its canonical, self-binding profile digest."""

    document = dict(value)
    document.pop("profile_sha256", None)
    document["profile_sha256"] = canonical_sha256(document)
    return document


def load_bundle_profile(
    *,
    bundle_id: str,
    bundle_sha256: str,
    model_type: str,
    required: bool,
    environ: Mapping[str, str] | None = None,
    path: str | None = None,
) -> ThroughputProfile | None:
    """Load the explicitly configured profile for one bundle.

    ``required=True`` is the fail-closed contract for experimental bundles.
    The incumbent caller uses ``required=False``; with no path configured this
    returns ``None`` and leaves every historical recipe output unchanged.
    """

    env = os.environ if environ is None else environ
    selected_path = path
    if selected_path is None:
        selected_path = str(env.get(PROFILE_ENV, "")).strip() or None
    if selected_path is None:
        if required:
            raise TimingProfileError(
                f"measured throughput profile required for bundle {bundle_id!r}"
            )
        return None
    return load_profile(
        selected_path,
        expected_bundle_id=bundle_id,
        expected_bundle_sha256=bundle_sha256,
        expected_model_type=model_type,
    )


def load_profile(
    path: str,
    *,
    expected_bundle_id: str,
    expected_bundle_sha256: str,
    expected_model_type: str,
) -> ThroughputProfile:
    """Load and validate one exact profile, rejecting all binding drift."""

    try:
        path_stat = os.lstat(path)
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
            raise TimingProfileError("timing profile must be a regular file")
        if path_stat.st_size <= 0 or path_stat.st_size > _MAX_PROFILE_BYTES:
            raise TimingProfileError("timing profile size is invalid")
        with open(path, "rb") as fh:
            raw = fh.read(_MAX_PROFILE_BYTES + 1)
        if len(raw) > _MAX_PROFILE_BYTES:
            raise TimingProfileError("timing profile is too large")
        value = json.loads(raw.decode("utf-8"))
    except TimingProfileError:
        raise
    except Exception as exc:
        raise TimingProfileError(f"timing profile unavailable: {path}") from exc

    return validate_profile(
        value,
        expected_bundle_id=expected_bundle_id,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_model_type=expected_model_type,
    )


def validate_profile(
    value: Any,
    *,
    expected_bundle_id: str,
    expected_bundle_sha256: str,
    expected_model_type: str,
) -> ThroughputProfile:
    """Validate schema, provenance, self-hash, and exact bundle binding."""

    document = _exact_object(value, _PROFILE_FIELDS, "timing profile")
    if (
        isinstance(document["schema"], bool)
        or not isinstance(document["schema"], int)
        or document["schema"] != PROFILE_SCHEMA
        or document["kind"] != PROFILE_KIND
    ):
        raise TimingProfileError("unsupported timing profile contract")

    bundle_id = _bundle_id(document["bundle_id"])
    bundle_sha256 = _sha256(document["bundle_sha256"], "bundle sha256")
    model_type = _text(document["model_type"], "model type", maximum=64).lower()
    if bundle_id != _bundle_id(expected_bundle_id):
        raise TimingProfileError("timing profile bundle id mismatch")
    if bundle_sha256 != _sha256(expected_bundle_sha256, "expected bundle sha256"):
        raise TimingProfileError("timing profile bundle digest mismatch")
    if model_type != _text(
        expected_model_type, "expected model type", maximum=64
    ).lower():
        raise TimingProfileError("timing profile model type mismatch")

    declared_profile_sha = _sha256(
        document["profile_sha256"], "profile sha256"
    )
    body = dict(document)
    body.pop("profile_sha256")
    if canonical_sha256(body) != declared_profile_sha:
        raise TimingProfileError("timing profile digest mismatch")

    seconds_per_step = _finite_positive(
        document["seconds_per_step"], "seconds per step", maximum=3600.0
    )
    startup_seconds = _finite_nonnegative(
        document["startup_seconds"], "startup seconds", maximum=86400.0
    )
    measurement = _exact_object(
        document["measurement"], _MEASUREMENT_FIELDS, "timing measurement"
    )
    completed_steps = _positive_int(
        measurement["completed_steps"], "completed steps"
    )
    training_elapsed_seconds = _finite_positive(
        measurement["training_elapsed_seconds"],
        "training elapsed seconds",
        maximum=604800.0,
    )
    first_checkpoint_step = _positive_int(
        measurement["first_checkpoint_step"], "first checkpoint step"
    )
    first_checkpoint_elapsed_seconds = _finite_positive(
        measurement["first_checkpoint_elapsed_seconds"],
        "first checkpoint elapsed seconds",
        maximum=604800.0,
    )
    if first_checkpoint_step > completed_steps:
        raise TimingProfileError("first checkpoint exceeds completed steps")
    if first_checkpoint_elapsed_seconds > training_elapsed_seconds:
        raise TimingProfileError("first checkpoint exceeds measured run time")
    if first_checkpoint_elapsed_seconds <= startup_seconds:
        raise TimingProfileError("first checkpoint does not follow startup")
    effective_seconds_per_step = (
        training_elapsed_seconds - startup_seconds
    ) / completed_steps
    relative_rate_error = abs(
        effective_seconds_per_step - seconds_per_step
    ) / seconds_per_step
    if relative_rate_error > 0.02:
        raise TimingProfileError(
            "seconds per step does not match the recorded measurement"
        )

    provenance = _exact_object(
        document["provenance"], _PROVENANCE_FIELDS, "timing provenance"
    )
    source_run_id = _text(
        provenance["source_run_id"], "source run id", maximum=256
    )
    source_record_sha256 = _sha256(
        provenance["source_record_sha256"], "source record sha256"
    )
    runtime_commit = _git_commit(provenance["runtime_commit"])
    measured_at_utc = _utc(provenance["measured_at_utc"])
    accelerator = _text(provenance["accelerator"], "accelerator", maximum=256)

    return ThroughputProfile(
        bundle_id=bundle_id,
        bundle_sha256=bundle_sha256,
        model_type=model_type,
        seconds_per_step=seconds_per_step,
        startup_seconds=startup_seconds,
        completed_steps=completed_steps,
        training_elapsed_seconds=training_elapsed_seconds,
        first_checkpoint_step=first_checkpoint_step,
        first_checkpoint_elapsed_seconds=first_checkpoint_elapsed_seconds,
        source_run_id=source_run_id,
        source_record_sha256=source_record_sha256,
        runtime_commit=runtime_commit,
        measured_at_utc=measured_at_utc,
        accelerator=accelerator,
        profile_sha256=declared_profile_sha,
    )


def observe_first_checkpoint(
    profile: ThroughputProfile,
    *,
    active_planned_steps: int,
    future_target_steps: int,
    checkpoint_step: int,
    elapsed_since_launch_s: float,
    total_budget_s: float,
    export_reserve_s: float,
    safety: float,
) -> FirstCheckpointCorrection:
    """Measure current effective throughput and recommend a future-run cap.

    The observation happens after the checkpoint is durable, so the derived
    seconds/step conservatively includes that checkpoint's write overhead.
    ``active_planned_steps`` is returned unchanged because the running
    ai-toolkit subprocess cannot accept a larger fixed step count.
    """

    if not isinstance(profile, ThroughputProfile):
        raise TimingProfileError("first-checkpoint correction needs a profile")
    planned = _positive_int(active_planned_steps, "active planned steps")
    target = _positive_int(future_target_steps, "future target steps")
    observed_step = _positive_int(checkpoint_step, "checkpoint step")
    if observed_step > planned:
        raise TimingProfileError("observed checkpoint exceeds the active plan")
    elapsed = _finite_positive(
        elapsed_since_launch_s,
        "elapsed since launch",
        maximum=604800.0,
    )
    budget = _finite_positive(total_budget_s, "total budget", maximum=604800.0)
    reserve = _finite_nonnegative(
        export_reserve_s, "export reserve", maximum=604800.0
    )
    margin = _finite_positive(safety, "safety", maximum=1.0)
    if elapsed <= profile.startup_seconds:
        raise TimingProfileError("first checkpoint elapsed before profiled startup")

    # The checkpoint write is deliberately left in the numerator.  It makes
    # this a conservative effective throughput estimate rather than assuming an
    # unmeasured I/O subtraction.
    observed_sit = (elapsed - profile.startup_seconds) / observed_step
    if not math.isfinite(observed_sit) or observed_sit <= 0:
        raise TimingProfileError("observed seconds per step is invalid")
    ratio = observed_sit / profile.seconds_per_step
    if ratio < 0.95:
        correction = "faster"
    elif ratio > 1.05:
        correction = "slower"
    else:
        correction = "within_profile_band"

    usable = budget * margin - profile.startup_seconds - reserve
    future_cap = max(1, int(max(0.0, usable) / observed_sit))
    future_recommended = max(1, min(target, future_cap))
    return FirstCheckpointCorrection(
        bundle_id=profile.bundle_id,
        profile_sha256=profile.profile_sha256,
        checkpoint_step=observed_step,
        elapsed_since_launch_s=elapsed,
        profiled_seconds_per_step=profile.seconds_per_step,
        observed_seconds_per_step=observed_sit,
        observed_to_profile_ratio=ratio,
        correction=correction,
        active_planned_steps=planned,
        active_plan_mutable=False,
        active_plan_action="observe_only_fixed_subprocess",
        active_plan_exceeds_observed_budget=planned > future_cap,
        future_budget_cap_steps=future_cap,
        future_target_steps=target,
        future_recommended_steps=future_recommended,
        future_step_delta=future_recommended - planned,
    )


def emit_first_checkpoint_observation(
    profile: ThroughputProfile,
    *,
    event_sink: Callable[..., None] | None = None,
    **observation_kwargs: Any,
) -> FirstCheckpointCorrection:
    """Compute and best-effort emit one first-checkpoint timing event."""

    observation = observe_first_checkpoint(profile, **observation_kwargs)
    sink = event_sink
    if sink is None:
        from forge import telemetry

        sink = telemetry.event
    try:
        sink(FIRST_CHECKPOINT_EVENT, **observation.telemetry_fields())
    except Exception:
        # Telemetry can never change the training result.
        pass
    return observation


def _exact_object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise TimingProfileError(f"{label} fields differ from schema")
    return value


def _text(value: Any, label: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise TimingProfileError(f"{label} is invalid")
    return value


def _bundle_id(value: Any) -> str:
    text = _text(value, "bundle id", maximum=64)
    if _BUNDLE_ID_RE.fullmatch(text) is None:
        raise TimingProfileError("bundle id is invalid")
    return text


def _sha256(value: Any, label: str) -> str:
    text = _text(value, label, maximum=64)
    if _SHA256_RE.fullmatch(text) is None:
        raise TimingProfileError(f"{label} is invalid")
    return text


def _git_commit(value: Any) -> str:
    text = _text(value, "runtime commit", maximum=40)
    if _GIT_COMMIT_RE.fullmatch(text) is None:
        raise TimingProfileError("runtime commit is invalid")
    return text


def _utc(value: Any) -> str:
    text = _text(value, "measurement time", maximum=40)
    if not text.endswith("Z"):
        raise TimingProfileError("measurement time must be UTC")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise TimingProfileError("measurement time is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise TimingProfileError("measurement time must be UTC")
    return text


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TimingProfileError(f"{label} is invalid")
    return value


def _finite_positive(value: Any, label: str, *, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimingProfileError(f"{label} is invalid")
    number = float(value)
    if not math.isfinite(number) or number <= 0 or number > maximum:
        raise TimingProfileError(f"{label} is invalid")
    return number


def _finite_nonnegative(value: Any, label: str, *, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimingProfileError(f"{label} is invalid")
    number = float(value)
    if not math.isfinite(number) or number < 0 or number > maximum:
        raise TimingProfileError(f"{label} is invalid")
    return number
