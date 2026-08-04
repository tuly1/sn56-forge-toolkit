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
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from forge.file_evidence import RegularFileError, read_regular_bytes


PROFILE_ENV = "FORGE_KREA_THROUGHPUT_PROFILE"
SOURCE_RECORD_ENV = "FORGE_KREA_THROUGHPUT_SOURCE_RECORD"
PROFILE_KIND = "forge-measured-throughput-profile"
PROFILE_SCHEMA = 3
FIRST_CHECKPOINT_EVENT = "first_checkpoint_timing_observed"
_MAX_PROFILE_BYTES = 64 * 1024
_MAX_SOURCE_RECORD_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_BUNDLE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_SOURCE_RUN_ID_RE = re.compile(r".+:[0-9a-f]{32}")

_PROFILE_FIELDS = {
    "schema",
    "kind",
    "bundle_id",
    "bundle_sha256",
    "model_type",
    "measured_dataset_size",
    "dataset_regime",
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
    "accelerator_identity",
}
_SOURCE_RECORD_FIELDS = {
    "schema",
    "runtime_contract_id",
    "source_run_id",
    "model_type",
    "runtime_repository",
    "runtime_commit",
    "bundle",
    "bundle_claim",
    "bundle_contract_sha256",
    "generated_config_sha256",
    "capability_manifest_file_sha256",
    "capability_manifest_semantic_sha256",
    "capabilities",
    "runtime_manifest_capability_aliases",
    "timing",
    "effective",
    "lifecycle",
    "first_checkpoint_observation",
    "training_completion_observation",
    "record_sha256",
}
_PROBE_TIMING_FIELDS = {
    "mode",
    "profile_sha256",
    "runtime_commit",
    "measured_dataset_size",
    "current_dataset_size",
    "dataset_regime",
    "accelerator_identity",
}
_FIRST_OBSERVATION_FIELDS = {
    "bundle_id",
    "timing_profile_sha256",
    "observation_mode",
    "checkpoint_step",
    "elapsed_since_launch_s",
    "active_planned_steps",
    "active_plan_mutable",
    "active_plan_action",
}
_COMPLETION_OBSERVATION_FIELDS = {
    "training_elapsed_seconds",
    "returncode",
    "stopped_by_deadline",
    "natural_completion",
    "artifact_path",
    "artifact_name",
    "artifact_size_bytes",
    "artifact_sha256",
    "artifact_loadable",
    "artifact_checkpoint_step",
    "completed_steps",
    "scope_attempt_nonce",
}
_SOURCE_EFFECTIVE_FIELDS = {
    "planned_steps",
    "normalized_config_projection",
}


class TimingProfileError(RuntimeError):
    """A measured timing profile is absent, invalid, or bound elsewhere."""


@dataclass(frozen=True)
class ThroughputProfile:
    """A validated measurement for one exact experimental bundle.

    ``training_elapsed_seconds`` and ``first_checkpoint_elapsed_seconds`` are
    measured from subprocess launch.  Schema 3 conservatively includes startup
    in the rate and therefore records ``startup_seconds`` as zero. The declared
    rate must equal
    ``(training_elapsed_seconds - startup_seconds) / completed_steps`` within
    two percent, so a self-consistent JSON hash cannot turn a guess into a
    claimed measurement.
    """

    bundle_id: str
    bundle_sha256: str
    model_type: str
    measured_dataset_size: int
    dataset_regime: str
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
    accelerator_identity: str
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


@dataclass(frozen=True)
class BootstrapFirstCheckpointObservation:
    """Raw durable-checkpoint timing evidence collected before a profile exists."""

    bundle_id: str
    checkpoint_step: int
    elapsed_since_launch_s: float
    active_planned_steps: int
    observation_mode: str = "bootstrap_raw_first_checkpoint"

    def telemetry_fields(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "timing_profile_sha256": None,
            "observation_mode": self.observation_mode,
            "checkpoint_step": self.checkpoint_step,
            "elapsed_since_launch_s": round(self.elapsed_since_launch_s, 3),
            "active_planned_steps": self.active_planned_steps,
            "active_plan_mutable": False,
            "active_plan_action": "observe_only_fixed_subprocess",
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


def _seal_profile_document(value: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a profile already derived from verified raw evidence."""

    document = dict(value)
    document.pop("profile_sha256", None)
    document["profile_sha256"] = canonical_sha256(document)
    return document


def dataset_regime(dataset_size: int) -> str:
    """Return the stable fixture-size band recorded alongside the exact size."""

    size = _positive_int(dataset_size, "dataset size")
    if size <= 10:
        return "tiny-1-10"
    if size <= 24:
        return "small-11-24"
    if size <= 50:
        return "medium-25-50"
    return "large-51-plus"


def current_accelerator_identity(
    *,
    environ: Mapping[str, str] | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> str:
    """Return a portable GPU-class identity or fail before profile reuse.

    Model name and total memory are read from the device while deliberately
    omitting per-device UUID, allowing reuse on an equivalent card. Environment
    variables are not an identity source and cannot override this observation.
    """

    del environ
    try:
        completed = runner(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if completed.returncode != 0 or len(rows) != 1:
            raise ValueError
        name, memory = (part.strip() for part in rows[0].rsplit(",", 1))
        if not name or not memory.isdigit():
            raise ValueError
        return _text(
            f"{name}|{int(memory)}-MiB",
            "accelerator identity",
            maximum=256,
        )
    except Exception as exc:
        raise TimingProfileError(
            "current accelerator identity could not be established"
        ) from exc


def produce_profile_document(
    source_record_path: str,
    *,
    source_run_id: str,
    bundle_id: str,
    model_type: str,
    measured_dataset_size: int,
    measured_at_utc: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
    expected_accelerator_identity: str | None = None,
) -> dict[str, Any]:
    """Derive a timing profile from one completed raw runtime record.

    The producer—not the operator—derives the recipe/runtime digests, observes
    the accelerator, hashes the source bytes, and cross-checks the declared run,
    dataset, first durable checkpoint, and natural terminal observation.  A
    plausible-looking JSON document without those source observations cannot
    become a schema-3 profile.
    """

    from forge import krea_runtime

    expected_run_id = _source_run_id(source_run_id)
    expected_bundle = _bundle_id(bundle_id)
    expected_model = _text(model_type, "model type", maximum=64).lower()
    expected_size = _positive_int(measured_dataset_size, "measured dataset size")
    expected_regime = dataset_regime(expected_size)
    expected_bundle_sha = krea_runtime.bundle_contract_sha256(expected_bundle)
    expected_runtime_commit = krea_runtime.runtime_commit_for_bundle(
        expected_bundle
    )
    expected_runtime_repository = krea_runtime.runtime_repository_for_bundle(
        expected_bundle
    )
    accelerator_identity = expected_accelerator_identity
    if accelerator_identity is None:
        accelerator_identity = current_accelerator_identity(runner=runner)
    accelerator_identity = _text(
        accelerator_identity, "accelerator identity", maximum=256
    )
    raw, record = _read_source_record(source_record_path)
    document = _exact_object(record, _SOURCE_RECORD_FIELDS, "source runtime record")
    if document["schema"] != 4:
        raise TimingProfileError("unsupported source runtime record schema")

    declared_record_sha = _sha256(
        document["record_sha256"], "source record semantic sha256"
    )
    body = dict(document)
    body.pop("record_sha256")
    if _runtime_record_semantic_sha256(body) != declared_record_sha:
        raise TimingProfileError("source runtime record digest mismatch")
    if document["runtime_contract_id"] != krea_runtime.RUNTIME_CONTRACT_ID:
        raise TimingProfileError("source runtime contract id mismatch")
    if document["source_run_id"] != expected_run_id:
        raise TimingProfileError("source runtime run id mismatch")
    if document["model_type"] != expected_model:
        raise TimingProfileError("source runtime model type mismatch")
    if document["bundle"] != expected_bundle:
        raise TimingProfileError("source runtime bundle id mismatch")
    if document["bundle_contract_sha256"] != expected_bundle_sha:
        raise TimingProfileError("source runtime bundle digest mismatch")
    if (
        document["runtime_repository"] != expected_runtime_repository
        or document["runtime_commit"] != expected_runtime_commit
    ):
        raise TimingProfileError("source runtime identity mismatch")
    expected_contract = krea_runtime.bundle_contract_document(expected_bundle)
    expected_capabilities = (
        []
        if expected_bundle == krea_runtime.INCUMBENT_BUNDLE
        else sorted(krea_runtime.REQUIRED_CAPABILITIES)
    )
    if (
        document["bundle_claim"] != expected_contract["claim"]
        or document["capabilities"] != expected_capabilities
        or document["runtime_manifest_capability_aliases"]
        != expected_contract["runtime_manifest_capability_aliases"]
    ):
        raise TimingProfileError("source runtime contract evidence mismatch")
    if document["lifecycle"] != "terminal":
        raise TimingProfileError("source runtime lifecycle is incomplete")
    _sha256(document["generated_config_sha256"], "generated config sha256")
    if expected_bundle != krea_runtime.INCUMBENT_BUNDLE:
        _sha256(
            document["capability_manifest_file_sha256"],
            "capability manifest file sha256",
        )
        _sha256(
            document["capability_manifest_semantic_sha256"],
            "capability manifest semantic sha256",
        )

    timing = _exact_object(
        document["timing"], _PROBE_TIMING_FIELDS, "source timing identity"
    )
    if (
        timing["mode"] != "bootstrap_probe_unmeasured"
        or timing["profile_sha256"] is not None
        or timing["runtime_commit"] != expected_runtime_commit
        or timing["measured_dataset_size"] is not None
        or timing["current_dataset_size"] != expected_size
        or timing["dataset_regime"] != expected_regime
        or timing["accelerator_identity"] != accelerator_identity
    ):
        raise TimingProfileError("source timing identity mismatch")

    effective = _exact_object(
        document["effective"], _SOURCE_EFFECTIVE_FIELDS, "source effective runtime"
    )
    if (
        effective["normalized_config_projection"]
        != expected_contract["normalized_config_projection"]
    ):
        raise TimingProfileError("source effective runtime projection mismatch")
    planned_steps = effective["planned_steps"]
    planned_steps = _positive_int(planned_steps, "source planned steps")
    first = _exact_object(
        document["first_checkpoint_observation"],
        _FIRST_OBSERVATION_FIELDS,
        "source first-checkpoint observation",
    )
    first_step = _positive_int(first["checkpoint_step"], "first checkpoint step")
    first_elapsed = _finite_positive(
        first["elapsed_since_launch_s"],
        "first checkpoint elapsed seconds",
        maximum=604800.0,
    )
    if (
        first["bundle_id"] != expected_bundle
        or first["timing_profile_sha256"] is not None
        or first["observation_mode"] != "bootstrap_raw_first_checkpoint"
        or first["active_planned_steps"] != planned_steps
        or first["active_plan_mutable"] is not False
        or first["active_plan_action"] != "observe_only_fixed_subprocess"
        or first_step > planned_steps
    ):
        raise TimingProfileError("source first-checkpoint observation mismatch")

    completion = _exact_object(
        document["training_completion_observation"],
        _COMPLETION_OBSERVATION_FIELDS,
        "source training-completion observation",
    )
    elapsed = _finite_positive(
        completion["training_elapsed_seconds"],
        "training elapsed seconds",
        maximum=604800.0,
    )
    artifact_path = _text(
        completion["artifact_path"], "terminal artifact path", maximum=4096
    )
    artifact_name = _text(
        completion["artifact_name"], "terminal artifact name", maximum=255
    )
    artifact_size = _positive_int(
        completion["artifact_size_bytes"], "terminal artifact size"
    )
    artifact_sha = _sha256(
        completion["artifact_sha256"], "terminal artifact sha256"
    )
    artifact_step = _nonnegative_int(
        completion["artifact_checkpoint_step"], "terminal artifact checkpoint step"
    )
    completed_steps = _positive_int(
        completion["completed_steps"], "terminal completed steps"
    )
    scope_attempt_nonce = _attempt_nonce(completion["scope_attempt_nonce"])
    if not expected_run_id.endswith(f":{scope_attempt_nonce}"):
        raise TimingProfileError("source terminal artifact scope mismatch")
    try:
        from forge.tasks.integrity import inspect_training_artifact

        artifact = inspect_training_artifact(artifact_path)
    except Exception as exc:
        raise TimingProfileError("source terminal artifact is unavailable") from exc
    if (
        artifact.path != os.path.abspath(artifact_path)
        or artifact_name != os.path.basename(artifact.path)
        or artifact.size_bytes != artifact_size
        or artifact.sha256 != artifact_sha
        or artifact.checkpoint_step != artifact_step
        or completion["artifact_loadable"] is not True
        or completed_steps != artifact_step
    ):
        raise TimingProfileError("source terminal artifact evidence mismatch")
    if (
        completed_steps != planned_steps
        or artifact_name != os.path.basename(artifact_path)
        or completion["returncode"] != 0
        or completion["stopped_by_deadline"] is not False
        or completion["natural_completion"] is not True
        or first_elapsed > elapsed
    ):
        raise TimingProfileError("source training did not complete naturally")

    measured_at = measured_at_utc
    if measured_at is None:
        measured_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
    measured_at = _utc(measured_at)
    result = _seal_profile_document(
        {
            "schema": PROFILE_SCHEMA,
            "kind": PROFILE_KIND,
            "bundle_id": expected_bundle,
            "bundle_sha256": expected_bundle_sha,
            "model_type": expected_model,
            "measured_dataset_size": expected_size,
            "dataset_regime": expected_regime,
            # Conservatively include startup and terminal-save time in the rate.
            "seconds_per_step": elapsed / planned_steps,
            "startup_seconds": 0.0,
            "measurement": {
                "completed_steps": completed_steps,
                "training_elapsed_seconds": elapsed,
                "first_checkpoint_step": first_step,
                "first_checkpoint_elapsed_seconds": first_elapsed,
            },
            "provenance": {
                "source_run_id": expected_run_id,
                "source_record_sha256": hashlib.sha256(raw).hexdigest(),
                "runtime_commit": expected_runtime_commit,
                "measured_at_utc": measured_at,
                "accelerator_identity": accelerator_identity,
            },
        }
    )
    # Reuse the normal loader's complete semantic validation before returning.
    validate_profile(
        result,
        expected_bundle_id=expected_bundle,
        expected_bundle_sha256=expected_bundle_sha,
        expected_model_type=expected_model,
        current_dataset_size=expected_size,
        expected_dataset_regime=expected_regime,
        expected_accelerator_identity=accelerator_identity,
    )
    return result


def load_bundle_profile(
    *,
    bundle_id: str,
    bundle_sha256: str,
    model_type: str,
    current_dataset_size: int,
    dataset_regime: str,
    required: bool,
    expected_accelerator_identity: str | None = None,
    environ: Mapping[str, str] | None = None,
    path: str | None = None,
    source_record_path: str | None = None,
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
    selected_source_path = source_record_path
    if selected_source_path is None:
        selected_source_path = str(env.get(SOURCE_RECORD_ENV, "")).strip() or None
    if selected_source_path is None:
        raise TimingProfileError(
            f"raw source runtime record required for bundle {bundle_id!r}"
        )
    accelerator_identity = expected_accelerator_identity
    if accelerator_identity is None:
        accelerator_identity = current_accelerator_identity(environ=env)
    return load_profile(
        selected_path,
        source_record_path=selected_source_path,
        expected_bundle_id=bundle_id,
        expected_bundle_sha256=bundle_sha256,
        expected_model_type=model_type,
        current_dataset_size=current_dataset_size,
        expected_dataset_regime=dataset_regime,
        expected_accelerator_identity=accelerator_identity,
    )


def load_profile(
    path: str,
    *,
    source_record_path: str,
    expected_bundle_id: str,
    expected_bundle_sha256: str,
    expected_model_type: str,
    current_dataset_size: int,
    expected_dataset_regime: str,
    expected_accelerator_identity: str,
) -> ThroughputProfile:
    """Load a profile and reproduce it from its mandatory exact raw record."""

    try:
        raw = read_regular_bytes(
            path,
            label="timing profile",
            maximum_size=_MAX_PROFILE_BYTES,
        )
        value = json.loads(raw.decode("utf-8"))
    except TimingProfileError:
        raise
    except RegularFileError as exc:
        raise TimingProfileError(str(exc)) from exc
    except Exception as exc:
        raise TimingProfileError(f"timing profile unavailable: {path}") from exc

    profile = validate_profile(
        value,
        expected_bundle_id=expected_bundle_id,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_model_type=expected_model_type,
        current_dataset_size=current_dataset_size,
        expected_dataset_regime=expected_dataset_regime,
        expected_accelerator_identity=expected_accelerator_identity,
    )
    reproduced = produce_profile_document(
        source_record_path,
        source_run_id=profile.source_run_id,
        bundle_id=profile.bundle_id,
        model_type=profile.model_type,
        measured_dataset_size=profile.measured_dataset_size,
        measured_at_utc=profile.measured_at_utc,
        expected_accelerator_identity=expected_accelerator_identity,
    )
    if reproduced != value:
        raise TimingProfileError(
            "timing profile is not reproduced by source runtime record"
        )
    return profile


def validate_profile(
    value: Any,
    *,
    expected_bundle_id: str,
    expected_bundle_sha256: str,
    expected_model_type: str,
    current_dataset_size: int,
    expected_dataset_regime: str,
    expected_accelerator_identity: str,
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
    measured_dataset_size = _positive_int(
        document["measured_dataset_size"], "measured dataset size"
    )
    profile_dataset_regime = _text(
        document["dataset_regime"], "dataset regime", maximum=64
    )
    if bundle_id != _bundle_id(expected_bundle_id):
        raise TimingProfileError("timing profile bundle id mismatch")
    if bundle_sha256 != _sha256(expected_bundle_sha256, "expected bundle sha256"):
        raise TimingProfileError("timing profile bundle digest mismatch")
    if model_type != _text(
        expected_model_type, "expected model type", maximum=64
    ).lower():
        raise TimingProfileError("timing profile model type mismatch")
    current_size = _positive_int(current_dataset_size, "current dataset size")
    expected_regime = _text(
        expected_dataset_regime, "expected dataset regime", maximum=64
    )
    if expected_regime != dataset_regime(current_size):
        raise TimingProfileError("current dataset regime is inconsistent")
    if profile_dataset_regime != dataset_regime(measured_dataset_size):
        raise TimingProfileError("timing profile dataset regime is inconsistent")
    if profile_dataset_regime != expected_regime:
        raise TimingProfileError("timing profile dataset regime mismatch")

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
    source_run_id = _source_run_id(provenance["source_run_id"])
    source_record_sha256 = _sha256(
        provenance["source_record_sha256"], "source record sha256"
    )
    runtime_commit = _git_commit(provenance["runtime_commit"])
    measured_at_utc = _utc(provenance["measured_at_utc"])
    accelerator_identity = _text(
        provenance["accelerator_identity"],
        "accelerator identity",
        maximum=256,
    )
    if accelerator_identity != _text(
        expected_accelerator_identity,
        "expected accelerator identity",
        maximum=256,
    ):
        raise TimingProfileError("timing profile accelerator identity mismatch")

    return ThroughputProfile(
        bundle_id=bundle_id,
        bundle_sha256=bundle_sha256,
        model_type=model_type,
        measured_dataset_size=measured_dataset_size,
        dataset_regime=profile_dataset_regime,
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
        accelerator_identity=accelerator_identity,
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


def emit_bootstrap_first_checkpoint_observation(
    *,
    bundle_id: str,
    checkpoint_step: int,
    elapsed_since_launch_s: float,
    active_planned_steps: int,
    event_sink: Callable[..., None] | None = None,
) -> BootstrapFirstCheckpointObservation:
    """Persistable raw observation for an explicitly labeled profile bootstrap."""

    planned = _positive_int(active_planned_steps, "active planned steps")
    observed_step = _positive_int(checkpoint_step, "checkpoint step")
    if observed_step > planned:
        raise TimingProfileError("observed checkpoint exceeds the active plan")
    observation = BootstrapFirstCheckpointObservation(
        bundle_id=_bundle_id(bundle_id),
        checkpoint_step=observed_step,
        elapsed_since_launch_s=_finite_positive(
            elapsed_since_launch_s,
            "elapsed since launch",
            maximum=604800.0,
        ),
        active_planned_steps=planned,
    )
    sink = event_sink
    if sink is None:
        from forge import telemetry

        sink = telemetry.event
    try:
        sink(FIRST_CHECKPOINT_EVENT, **observation.telemetry_fields())
    except Exception:
        pass
    return observation


def _read_source_record(path: str) -> tuple[bytes, Any]:
    """Read a bounded regular source record without following a symlink."""

    try:
        raw = read_regular_bytes(
            path,
            label="source runtime record",
            maximum_size=_MAX_SOURCE_RECORD_BYTES,
        )
        return raw, json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise TimingProfileError(
            f"source runtime record unavailable: {path}"
        ) from exc


def _runtime_record_semantic_sha256(value: Any) -> str:
    """Match the effective-runtime record's newline-terminated hash domain."""

    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _source_run_id(value: Any) -> str:
    text = _text(value, "source run id", maximum=256)
    if _SOURCE_RUN_ID_RE.fullmatch(text) is None:
        raise TimingProfileError("source run id is invalid")
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


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TimingProfileError(f"{label} is invalid")
    return value


def _attempt_nonce(value: Any) -> str:
    text = _text(value, "scope attempt nonce", maximum=32)
    if re.fullmatch(r"[0-9a-f]{32}", text) is None:
        raise TimingProfileError("scope attempt nonce is invalid")
    return text


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
