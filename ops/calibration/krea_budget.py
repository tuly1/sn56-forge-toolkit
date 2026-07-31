"""Evidence-bound Krea calibration budget planning.

This module is intentionally outside the production ``forge`` package.  It is
Day-0 experiment plumbing: a measured profile goes in and an auditable schedule
comes out.  There are no timing defaults and no exception-to-guess fallback.

ai-toolkit exposes one ``save_every`` value rather than arbitrary checkpoint
steps.  We therefore ask it to save at an approximately one-eighth cadence,
record the checkpoints that cadence can *actually* produce, and map the desired
10/25/50/75/90/final observations to their nearest real candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import hashlib
import hmac
import json
import math
from numbers import Integral, Real
import re
from typing import Any, Mapping


_PROFILE_SCHEMA_VERSION = 2
_PLAN_SCHEMA_VERSION = 2
_MODEL_TYPE = "krea2"
_MIN_STARTUP_SAMPLES = 3
_MIN_UPDATE_SAMPLES = 100
_MIN_SAVE_SAMPLES = 8
_MIN_FRAMEWORK_STOP_BOUNDARY_S = 225.0
_FROZEN_MINIMUM_UTILIZATION = Decimal("0.90")
_FROZEN_MAXIMUM_SAVE_OVERHEAD = Decimal("0.10")
_ALLOWED_EXECUTION_SURFACES = {
    ("staged_host_venv", "discovery_only"),
    ("immutable_production_docker_image", "stage2_throughput_timing_only"),
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_EXECUTION_ENVELOPE_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "model_type",
        "equivalence_class",
        "network_rank",
        "network_alpha",
        "optimizer",
        "optimizer_config_sha256",
        "loss",
        "differential_guidance_enabled",
        "guidance_scale",
        "training_pair_count",
        "training_dataset_shape_sha256",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "data_parallel_replicas",
        "resolution_policy_sha256",
        "precision_policy_sha256",
        "cache_latents_to_disk",
        "cache_text_embeddings",
        "compile_enabled",
        "jit_enabled",
        "dataloader_workers",
        "base_model_identity_sha256",
        "runtime_identity_sha256",
        "host_execution_identity_sha256",
        "execution_surface",
        "execution_scope",
        "venv_tree_manifest_sha256",
        "reference_container_image_sha256",
        "gpu_identity_sha256",
        "trainer_identity_sha256",
        "measurement_tool_sha256",
    }
)
_EXECUTION_ENVELOPE_KEYS = _EXECUTION_ENVELOPE_PAYLOAD_KEYS | {
    "execution_envelope_sha256"
}
_PROFILE_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "model_type",
        "execution_envelope",
        "raw_sample_manifest_sha256",
        "startup_sample_count",
        "update_sample_count",
        "save_sample_count",
        "startup_upper_bound_s",
        "update_upper_bound_s",
        "save_upper_bound_s",
        "bound_method",
        "margin_policy_sha256",
        "end_to_end_validation_count",
        "end_to_end_validation_sha256",
        "framework_stop_boundary_s",
        "framework_stop_boundary_source_sha256",
        "selection_mode",
        "selection_scorer_identity_sha256",
        "selection_scoring_reserve_s",
        "finalization_reserve_s",
        "upload_reserve_s",
    }
)
_PROFILE_KEYS = _PROFILE_PAYLOAD_KEYS | {"profile_sha256"}
_DESIRED_CANDIDATES = (
    ("10%", Decimal("0.10")),
    ("25%", Decimal("0.25")),
    ("50%", Decimal("0.50")),
    ("75%", Decimal("0.75")),
    ("90%", Decimal("0.90")),
    ("final", Decimal("1")),
)
_TIMING_METRICS = (
    "startup",
    "optimizer_update",
    "checkpoint_save",
    "finalization",
    "upload",
)


class ProfileValidationError(ValueError):
    """A throughput profile is incomplete, malformed, or not content-bound."""


class InsufficientBudgetError(ValueError):
    """The measured budget cannot afford even one update and finalization."""


class TimingEvidenceError(ProfileValidationError):
    """Raw timing observations or their predeclared margin are invalid."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProfileValidationError("profile is not canonical JSON") from exc


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProfileValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise ProfileValidationError(f"{label} must be a conservative identifier")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ProfileValidationError(f"{label} must be a JSON boolean")
    return value


def _require_nonnegative_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ProfileValidationError(f"{label} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ProfileValidationError(f"{label} must be a non-negative integer")
    return result


def _require_count(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ProfileValidationError(f"{label} must be a positive integer")
    result = int(value)
    if result < minimum:
        raise ProfileValidationError(
            f"{label} must contain at least {minimum} observations"
        )
    return result


def _require_json_seconds(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ProfileValidationError(f"{label} must be a positive JSON number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ProfileValidationError(f"{label} must be finite and greater than zero")
    return result


def _nonnegative_json_seconds(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ProfileValidationError(f"{label} must be a non-negative JSON number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ProfileValidationError(f"{label} must be finite and non-negative")
    return result


def _positive_decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{label} must be a positive number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be a positive number") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{label} must be finite and greater than zero")
    return result


def _fraction_decimal(value: Any, label: str) -> Decimal:
    result = _positive_decimal(value, label)
    if result > 1:
        raise ValueError(f"{label} must be in (0, 1]")
    return result


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


@dataclass(frozen=True)
class ExecutionEnvelope:
    """The exact measured condition under which timing may be reused.

    This deliberately binds quality-recipe fields as well as hardware/runtime
    fields. A Rank-32 AdamW profile is not admissible for Rank-64 Automagic,
    and a timing result from a differently shaped dataset is not silently
    treated as portable.
    """

    equivalence_class: str
    network_rank: int
    network_alpha: int
    optimizer: str
    optimizer_config_sha256: str
    loss: str
    differential_guidance_enabled: bool
    guidance_scale: float | None
    training_pair_count: int
    training_dataset_shape_sha256: str
    micro_batch_size: int
    gradient_accumulation_steps: int
    data_parallel_replicas: int
    resolution_policy_sha256: str
    precision_policy_sha256: str
    cache_latents_to_disk: bool
    cache_text_embeddings: bool
    compile_enabled: bool
    jit_enabled: bool
    dataloader_workers: int
    base_model_identity_sha256: str
    runtime_identity_sha256: str
    host_execution_identity_sha256: str
    execution_surface: str
    execution_scope: str
    venv_tree_manifest_sha256: str
    reference_container_image_sha256: str
    gpu_identity_sha256: str
    trainer_identity_sha256: str
    measurement_tool_sha256: str
    execution_envelope_sha256: str

    @property
    def images_per_update(self) -> int:
        return (
            self.micro_batch_size
            * self.gradient_accumulation_steps
            * self.data_parallel_replicas
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "model_type": _MODEL_TYPE,
            "equivalence_class": self.equivalence_class,
            "network_rank": self.network_rank,
            "network_alpha": self.network_alpha,
            "optimizer": self.optimizer,
            "optimizer_config_sha256": self.optimizer_config_sha256,
            "loss": self.loss,
            "differential_guidance_enabled": self.differential_guidance_enabled,
            "guidance_scale": self.guidance_scale,
            "training_pair_count": self.training_pair_count,
            "training_dataset_shape_sha256": self.training_dataset_shape_sha256,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "data_parallel_replicas": self.data_parallel_replicas,
            "resolution_policy_sha256": self.resolution_policy_sha256,
            "precision_policy_sha256": self.precision_policy_sha256,
            "cache_latents_to_disk": self.cache_latents_to_disk,
            "cache_text_embeddings": self.cache_text_embeddings,
            "compile_enabled": self.compile_enabled,
            "jit_enabled": self.jit_enabled,
            "dataloader_workers": self.dataloader_workers,
            "base_model_identity_sha256": self.base_model_identity_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "host_execution_identity_sha256": self.host_execution_identity_sha256,
            "execution_surface": self.execution_surface,
            "execution_scope": self.execution_scope,
            "venv_tree_manifest_sha256": self.venv_tree_manifest_sha256,
            "reference_container_image_sha256": self.reference_container_image_sha256,
            "gpu_identity_sha256": self.gpu_identity_sha256,
            "trainer_identity_sha256": self.trainer_identity_sha256,
            "measurement_tool_sha256": self.measurement_tool_sha256,
            "execution_envelope_sha256": self.execution_envelope_sha256,
        }


def seal_execution_envelope(
    *,
    equivalence_class: str,
    network_rank: int,
    network_alpha: int,
    optimizer: str,
    optimizer_config_sha256: str,
    loss: str,
    differential_guidance_enabled: bool,
    guidance_scale: float | None,
    training_pair_count: int,
    training_dataset_shape_sha256: str,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    data_parallel_replicas: int,
    resolution_policy_sha256: str,
    precision_policy_sha256: str,
    cache_latents_to_disk: bool,
    cache_text_embeddings: bool,
    compile_enabled: bool,
    jit_enabled: bool,
    dataloader_workers: int,
    base_model_identity_sha256: str,
    runtime_identity_sha256: str,
    host_execution_identity_sha256: str,
    execution_surface: str,
    execution_scope: str,
    venv_tree_manifest_sha256: str,
    reference_container_image_sha256: str,
    gpu_identity_sha256: str,
    trainer_identity_sha256: str,
    measurement_tool_sha256: str,
) -> dict[str, Any]:
    """Normalize and self-bind one measured execution-equivalence class."""

    guidance_enabled = _require_bool(
        differential_guidance_enabled, "differential_guidance_enabled"
    )
    if guidance_enabled:
        normalized_guidance: float | None = _require_json_seconds(
            guidance_scale, "guidance_scale"
        )
    else:
        if guidance_scale is not None:
            raise ProfileValidationError(
                "guidance_scale must be null when differential guidance is disabled"
            )
        normalized_guidance = None
    payload: dict[str, Any] = {
        "schema_version": 2,
        "model_type": _MODEL_TYPE,
        "equivalence_class": _require_safe_id(equivalence_class, "equivalence_class"),
        "network_rank": _require_count(network_rank, "network_rank", minimum=1),
        "network_alpha": _require_count(network_alpha, "network_alpha", minimum=1),
        "optimizer": _require_safe_id(optimizer, "optimizer"),
        "optimizer_config_sha256": _require_sha256(
            optimizer_config_sha256, "optimizer_config_sha256"
        ),
        "loss": _require_safe_id(loss, "loss"),
        "differential_guidance_enabled": guidance_enabled,
        "guidance_scale": normalized_guidance,
        "training_pair_count": _require_count(
            training_pair_count, "training_pair_count", minimum=1
        ),
        "training_dataset_shape_sha256": _require_sha256(
            training_dataset_shape_sha256, "training_dataset_shape_sha256"
        ),
        "micro_batch_size": _require_count(
            micro_batch_size, "micro_batch_size", minimum=1
        ),
        "gradient_accumulation_steps": _require_count(
            gradient_accumulation_steps,
            "gradient_accumulation_steps",
            minimum=1,
        ),
        "data_parallel_replicas": _require_count(
            data_parallel_replicas, "data_parallel_replicas", minimum=1
        ),
        "resolution_policy_sha256": _require_sha256(
            resolution_policy_sha256, "resolution_policy_sha256"
        ),
        "precision_policy_sha256": _require_sha256(
            precision_policy_sha256, "precision_policy_sha256"
        ),
        "cache_latents_to_disk": _require_bool(
            cache_latents_to_disk, "cache_latents_to_disk"
        ),
        "cache_text_embeddings": _require_bool(
            cache_text_embeddings, "cache_text_embeddings"
        ),
        "compile_enabled": _require_bool(compile_enabled, "compile_enabled"),
        "jit_enabled": _require_bool(jit_enabled, "jit_enabled"),
        "dataloader_workers": _require_nonnegative_count(
            dataloader_workers, "dataloader_workers"
        ),
        "execution_surface": _require_safe_id(execution_surface, "execution_surface"),
        "execution_scope": _require_safe_id(execution_scope, "execution_scope"),
    }
    if (execution_surface, execution_scope) not in _ALLOWED_EXECUTION_SURFACES:
        raise ProfileValidationError(
            "throughput profiles require an exact admitted Stage-1 discovery "
            "or Stage-2 production-timing surface/scope pair"
        )
    for label, value in (
        ("base_model_identity_sha256", base_model_identity_sha256),
        ("runtime_identity_sha256", runtime_identity_sha256),
        ("host_execution_identity_sha256", host_execution_identity_sha256),
        ("venv_tree_manifest_sha256", venv_tree_manifest_sha256),
        (
            "reference_container_image_sha256",
            reference_container_image_sha256,
        ),
        ("gpu_identity_sha256", gpu_identity_sha256),
        ("trainer_identity_sha256", trainer_identity_sha256),
        ("measurement_tool_sha256", measurement_tool_sha256),
    ):
        payload[label] = _require_sha256(value, label)
    return {
        **payload,
        "execution_envelope_sha256": _payload_sha256(payload),
    }


def load_execution_envelope(document: Mapping[str, Any]) -> ExecutionEnvelope:
    if not isinstance(document, Mapping):
        raise ProfileValidationError("execution_envelope must be a mapping")
    keys = frozenset(document.keys())
    if keys != _EXECUTION_ENVELOPE_KEYS:
        missing = sorted(_EXECUTION_ENVELOPE_KEYS - keys, key=repr)
        extra = sorted(keys - _EXECUTION_ENVELOPE_KEYS, key=repr)
        raise ProfileValidationError(
            "execution envelope schema mismatch; "
            f"missing={missing!r}, extra={extra!r}"
        )
    if document["schema_version"] != 2 or document["model_type"] != _MODEL_TYPE:
        raise ProfileValidationError("unsupported execution envelope schema/model")
    normalized = seal_execution_envelope(
        **{
            key: document[key]
            for key in _EXECUTION_ENVELOPE_PAYLOAD_KEYS
            if key not in {"schema_version", "model_type"}
        }
    )
    claimed = _require_sha256(
        document["execution_envelope_sha256"], "execution_envelope_sha256"
    )
    if not hmac.compare_digest(claimed, normalized["execution_envelope_sha256"]):
        raise ProfileValidationError(
            "execution_envelope_sha256 does not match envelope contents"
        )
    return ExecutionEnvelope(
        **{
            key: normalized[key]
            for key in _EXECUTION_ENVELOPE_KEYS
            if key not in {"schema_version", "model_type"}
        }
    )


@dataclass(frozen=True)
class ThroughputProfile:
    """A validated, content-bound set of conservative timing measurements."""

    execution_envelope: ExecutionEnvelope
    raw_sample_manifest_sha256: str
    startup_sample_count: int
    update_sample_count: int
    save_sample_count: int
    startup_upper_bound_s: float
    update_upper_bound_s: float
    save_upper_bound_s: float
    bound_method: str
    margin_policy_sha256: str
    end_to_end_validation_count: int
    end_to_end_validation_sha256: str
    framework_stop_boundary_s: float
    framework_stop_boundary_source_sha256: str
    selection_mode: str
    selection_scorer_identity_sha256: str | None
    selection_scoring_reserve_s: float
    finalization_reserve_s: float
    upload_reserve_s: float
    profile_sha256: str

    @property
    def execution_envelope_sha256(self) -> str:
        return self.execution_envelope.execution_envelope_sha256

    @property
    def micro_batch_size(self) -> int:
        return self.execution_envelope.micro_batch_size

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.execution_envelope.gradient_accumulation_steps

    @property
    def data_parallel_replicas(self) -> int:
        return self.execution_envelope.data_parallel_replicas

    @property
    def resolution_policy_sha256(self) -> str:
        return self.execution_envelope.resolution_policy_sha256

    @property
    def precision_policy_sha256(self) -> str:
        return self.execution_envelope.precision_policy_sha256

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": _PROFILE_SCHEMA_VERSION,
            "model_type": _MODEL_TYPE,
            "execution_envelope": self.execution_envelope.to_record(),
            "raw_sample_manifest_sha256": self.raw_sample_manifest_sha256,
            "startup_sample_count": self.startup_sample_count,
            "update_sample_count": self.update_sample_count,
            "save_sample_count": self.save_sample_count,
            "startup_upper_bound_s": self.startup_upper_bound_s,
            "update_upper_bound_s": self.update_upper_bound_s,
            "save_upper_bound_s": self.save_upper_bound_s,
            "bound_method": self.bound_method,
            "margin_policy_sha256": self.margin_policy_sha256,
            "end_to_end_validation_count": self.end_to_end_validation_count,
            "end_to_end_validation_sha256": self.end_to_end_validation_sha256,
            "framework_stop_boundary_s": self.framework_stop_boundary_s,
            "framework_stop_boundary_source_sha256": (
                self.framework_stop_boundary_source_sha256
            ),
            "selection_mode": self.selection_mode,
            "selection_scorer_identity_sha256": self.selection_scorer_identity_sha256,
            "selection_scoring_reserve_s": self.selection_scoring_reserve_s,
            "finalization_reserve_s": self.finalization_reserve_s,
            "upload_reserve_s": self.upload_reserve_s,
            "profile_sha256": self.profile_sha256,
        }


def seal_throughput_profile(
    *,
    execution_envelope: Mapping[str, Any],
    raw_sample_manifest_sha256: str,
    startup_sample_count: int,
    update_sample_count: int,
    save_sample_count: int,
    startup_upper_bound_s: float,
    update_upper_bound_s: float,
    save_upper_bound_s: float,
    bound_method: str,
    margin_policy_sha256: str,
    end_to_end_validation_count: int,
    end_to_end_validation_sha256: str,
    framework_stop_boundary_s: float,
    framework_stop_boundary_source_sha256: str,
    selection_mode: str,
    selection_scorer_identity_sha256: str | None,
    selection_scoring_reserve_s: float,
    finalization_reserve_s: float,
    upload_reserve_s: float,
) -> dict[str, Any]:
    """Create a canonical profile record and bind every field with SHA-256.

    Callers must supply every measurement and reserve.  In particular, this
    function never fills absent timing data from a built-in default.
    """

    envelope = load_execution_envelope(execution_envelope)
    payload = {
        "schema_version": _PROFILE_SCHEMA_VERSION,
        "model_type": _MODEL_TYPE,
        "execution_envelope": envelope.to_record(),
        "raw_sample_manifest_sha256": _require_sha256(
            raw_sample_manifest_sha256, "raw_sample_manifest_sha256"
        ),
        "startup_sample_count": _require_count(
            startup_sample_count,
            "startup_sample_count",
            minimum=_MIN_STARTUP_SAMPLES,
        ),
        "update_sample_count": _require_count(
            update_sample_count,
            "update_sample_count",
            minimum=_MIN_UPDATE_SAMPLES,
        ),
        "save_sample_count": _require_count(
            save_sample_count,
            "save_sample_count",
            minimum=_MIN_SAVE_SAMPLES,
        ),
        "startup_upper_bound_s": _require_json_seconds(
            startup_upper_bound_s, "startup_upper_bound_s"
        ),
        "update_upper_bound_s": _require_json_seconds(
            update_upper_bound_s, "update_upper_bound_s"
        ),
        "save_upper_bound_s": _require_json_seconds(
            save_upper_bound_s, "save_upper_bound_s"
        ),
        "bound_method": bound_method,
        "margin_policy_sha256": _require_sha256(
            margin_policy_sha256, "margin_policy_sha256"
        ),
        "end_to_end_validation_count": _require_count(
            end_to_end_validation_count,
            "end_to_end_validation_count",
            minimum=1,
        ),
        "end_to_end_validation_sha256": _require_sha256(
            end_to_end_validation_sha256, "end_to_end_validation_sha256"
        ),
        "framework_stop_boundary_s": _require_json_seconds(
            framework_stop_boundary_s, "framework_stop_boundary_s"
        ),
        "framework_stop_boundary_source_sha256": _require_sha256(
            framework_stop_boundary_source_sha256,
            "framework_stop_boundary_source_sha256",
        ),
        "selection_mode": selection_mode,
        "selection_scorer_identity_sha256": selection_scorer_identity_sha256,
        "selection_scoring_reserve_s": _nonnegative_json_seconds(
            selection_scoring_reserve_s, "selection_scoring_reserve_s"
        ),
        "finalization_reserve_s": _require_json_seconds(
            finalization_reserve_s, "finalization_reserve_s"
        ),
        "upload_reserve_s": _require_json_seconds(upload_reserve_s, "upload_reserve_s"),
    }
    if bound_method != "observed-max-plus-predeclared-margin":
        raise ProfileValidationError("unsupported bound_method")
    if payload["framework_stop_boundary_s"] < _MIN_FRAMEWORK_STOP_BOUNDARY_S:
        raise ProfileValidationError(
            "framework_stop_boundary_s must preserve the frozen 225-second "
            "Forge export-plus-stop boundary"
        )
    evidence_digests = {
        payload["raw_sample_manifest_sha256"],
        payload["margin_policy_sha256"],
        payload["end_to_end_validation_sha256"],
    }
    if len(evidence_digests) != 3:
        raise ProfileValidationError(
            "raw samples, margin policy, and held-out end-to-end validation "
            "must be three distinct evidence artifacts"
        )
    if selection_mode not in {
        "offline_post_training",
        "deterministic_checkpoint",
        "live_in_budget",
    }:
        raise ProfileValidationError("unsupported selection_mode")
    if selection_mode == "live_in_budget":
        _require_sha256(
            selection_scorer_identity_sha256,
            "selection_scorer_identity_sha256",
        )
        if payload["selection_scoring_reserve_s"] <= 0:
            raise ProfileValidationError(
                "live_in_budget selection requires a positive measured reserve"
            )
    elif (
        selection_scorer_identity_sha256 is not None
        or payload["selection_scoring_reserve_s"] != 0
    ):
        raise ProfileValidationError(
            "offline/deterministic selection must have no in-budget scorer reserve"
        )
    return {**payload, "profile_sha256": _payload_sha256(payload)}


def load_throughput_profile(document: Mapping[str, Any]) -> ThroughputProfile:
    """Validate an exact-schema profile before it can influence a run."""

    if not isinstance(document, Mapping):
        raise ProfileValidationError("profile must be a mapping")
    keys = frozenset(document.keys())
    if keys != _PROFILE_KEYS:
        missing = sorted(_PROFILE_KEYS - keys, key=repr)
        extra = sorted(keys - _PROFILE_KEYS, key=repr)
        raise ProfileValidationError(
            f"profile schema mismatch; missing={missing!r}, extra={extra!r}"
        )
    if document["schema_version"] != _PROFILE_SCHEMA_VERSION:
        raise ProfileValidationError("unsupported profile schema_version")
    if document["model_type"] != _MODEL_TYPE:
        raise ProfileValidationError("throughput profile is not for krea2")

    # Normalize first, then hash the normalized payload.  seal/load therefore
    # remain stable across a JSON round trip (where Integral subclasses vanish).
    normalized = seal_throughput_profile(
        execution_envelope=document["execution_envelope"],
        raw_sample_manifest_sha256=document["raw_sample_manifest_sha256"],
        startup_sample_count=document["startup_sample_count"],
        update_sample_count=document["update_sample_count"],
        save_sample_count=document["save_sample_count"],
        startup_upper_bound_s=document["startup_upper_bound_s"],
        update_upper_bound_s=document["update_upper_bound_s"],
        save_upper_bound_s=document["save_upper_bound_s"],
        bound_method=document["bound_method"],
        margin_policy_sha256=document["margin_policy_sha256"],
        end_to_end_validation_count=document["end_to_end_validation_count"],
        end_to_end_validation_sha256=document["end_to_end_validation_sha256"],
        framework_stop_boundary_s=document["framework_stop_boundary_s"],
        framework_stop_boundary_source_sha256=document[
            "framework_stop_boundary_source_sha256"
        ],
        selection_mode=document["selection_mode"],
        selection_scorer_identity_sha256=document["selection_scorer_identity_sha256"],
        selection_scoring_reserve_s=document["selection_scoring_reserve_s"],
        finalization_reserve_s=document["finalization_reserve_s"],
        upload_reserve_s=document["upload_reserve_s"],
    )
    claimed = _require_sha256(document["profile_sha256"], "profile_sha256")
    if not hmac.compare_digest(claimed, normalized["profile_sha256"]):
        raise ProfileValidationError("profile_sha256 does not match profile contents")
    return ThroughputProfile(
        execution_envelope=load_execution_envelope(normalized["execution_envelope"]),
        **{
            key: normalized[key]
            for key in _PROFILE_KEYS
            if key not in {"schema_version", "model_type", "execution_envelope"}
        },
    )


def _exact_mapping_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    keys = frozenset(value)
    if keys != expected:
        raise TimingEvidenceError(
            f"{label} schema mismatch; missing={sorted(expected - keys)!r}, "
            f"extra={sorted(keys - expected)!r}"
        )


def _timing_sample(value: Any, *, metric: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TimingEvidenceError(f"{metric} timing sample must be a mapping")
    _exact_mapping_keys(
        value,
        frozenset(
            {
                "capture_id",
                "observation_id",
                "duration_s",
                "units",
                "started_monotonic_ns",
                "ended_monotonic_ns",
            }
        ),
        f"{metric} timing sample",
    )
    capture_id = _require_safe_id(value["capture_id"], "capture_id")
    observation_id = _require_safe_id(value["observation_id"], "observation_id")
    units = _require_count(value["units"], "timing sample units", minimum=1)
    start = _require_nonnegative_count(
        value["started_monotonic_ns"], "started_monotonic_ns"
    )
    end = _require_nonnegative_count(value["ended_monotonic_ns"], "ended_monotonic_ns")
    if end <= start:
        raise TimingEvidenceError("timing sample must end after it starts")
    duration = _require_json_seconds(value["duration_s"], "timing sample duration_s")
    observed = (end - start) / 1_000_000_000
    # The producer writes duration_s from the same monotonic receipt timestamps.
    # Recompute it here instead of trusting an independently editable number.
    if not math.isclose(duration, observed, rel_tol=0.0, abs_tol=5e-10):
        raise TimingEvidenceError("timing sample duration contradicts monotonic clocks")
    return {
        "capture_id": capture_id,
        "observation_id": observation_id,
        "duration_s": duration,
        "units": units,
        "started_monotonic_ns": start,
        "ended_monotonic_ns": end,
    }


def seal_timing_sample_manifest(
    *,
    execution_envelope: Mapping[str, Any],
    probe_contract_sha256: str,
    measurement_tool_sha256: str,
    command_captures: list[Mapping[str, Any]],
    samples: Mapping[str, list[Mapping[str, Any]]],
    seed_bindings: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal raw receipt-clock measurements from the timing-probe producer.

    ``units`` allows one observed optimizer/save block to cover many real
    operations without manufacturing duplicate pseudo-samples.  Minimum sample
    requirements are enforced using those observed units, while startup still
    requires three independently identified observations.
    """

    envelope = load_execution_envelope(execution_envelope)
    tool_sha = _require_sha256(measurement_tool_sha256, "measurement_tool_sha256")
    if envelope.measurement_tool_sha256 != tool_sha:
        raise TimingEvidenceError(
            "execution envelope does not name the timing producer that sealed samples"
        )
    if not isinstance(command_captures, list) or not command_captures:
        raise TimingEvidenceError("command_captures must contain at least one capture")
    normalized_captures: list[dict[str, Any]] = []
    seen_capture_ids: set[str] = set()
    for item in command_captures:
        if not isinstance(item, Mapping):
            raise TimingEvidenceError("command capture must be a mapping")
        _exact_mapping_keys(
            item,
            frozenset(
                {
                    "capture_id",
                    "argv",
                    "executable_path",
                    "executable_sha256",
                    "returncode",
                    "started_unix_ns",
                    "ended_unix_ns",
                    "event_stream_sha256",
                }
            ),
            "command capture",
        )
        capture_id = _require_safe_id(item["capture_id"], "capture_id")
        if capture_id in seen_capture_ids:
            raise TimingEvidenceError("duplicate command capture id")
        seen_capture_ids.add(capture_id)
        argv = item["argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(arg, str) or "\x00" in arg for arg in argv)
        ):
            raise TimingEvidenceError("command capture argv is invalid")
        executable_path = item["executable_path"]
        if not isinstance(executable_path, str) or not executable_path.startswith("/"):
            raise TimingEvidenceError("captured executable path must be absolute")
        returncode = item["returncode"]
        if isinstance(returncode, bool) or not isinstance(returncode, Integral):
            raise TimingEvidenceError("captured returncode must be an integer")
        if int(returncode) != 0:
            raise TimingEvidenceError("failed timing commands cannot enter a profile")
        started = _require_nonnegative_count(item["started_unix_ns"], "started_unix_ns")
        ended = _require_nonnegative_count(item["ended_unix_ns"], "ended_unix_ns")
        if ended <= started:
            raise TimingEvidenceError("command capture timestamps are invalid")
        normalized_captures.append(
            {
                "capture_id": capture_id,
                "argv": list(argv),
                "executable_path": executable_path,
                "executable_sha256": _require_sha256(
                    item["executable_sha256"], "executable_sha256"
                ),
                "returncode": int(returncode),
                "started_unix_ns": started,
                "ended_unix_ns": ended,
                "event_stream_sha256": _require_sha256(
                    item["event_stream_sha256"], "event_stream_sha256"
                ),
            }
        )
    normalized_captures.sort(key=lambda row: row["capture_id"])

    if not isinstance(samples, Mapping):
        raise TimingEvidenceError("samples must be a mapping")
    _exact_mapping_keys(samples, frozenset(_TIMING_METRICS), "timing samples")
    normalized_samples: dict[str, list[dict[str, Any]]] = {}
    seen_observations: set[tuple[str, str]] = set()
    for metric in _TIMING_METRICS:
        rows = samples[metric]
        if not isinstance(rows, list) or not rows:
            raise TimingEvidenceError(f"{metric} samples must be non-empty")
        normalized = []
        for raw in rows:
            row = _timing_sample(raw, metric=metric)
            if row["capture_id"] not in seen_capture_ids:
                raise TimingEvidenceError("timing sample names an unknown capture")
            identity = (row["capture_id"], row["observation_id"])
            if identity in seen_observations:
                raise TimingEvidenceError("timing observation id is reused")
            seen_observations.add(identity)
            normalized.append(row)
        normalized.sort(key=lambda row: (row["capture_id"], row["observation_id"]))
        normalized_samples[metric] = normalized

    if len(normalized_samples["startup"]) < _MIN_STARTUP_SAMPLES:
        raise TimingEvidenceError(
            "raw timing evidence needs three startup observations"
        )
    if (
        sum(row["units"] for row in normalized_samples["optimizer_update"])
        < _MIN_UPDATE_SAMPLES
    ):
        raise TimingEvidenceError("raw timing evidence covers fewer than 100 updates")
    if (
        sum(row["units"] for row in normalized_samples["checkpoint_save"])
        < _MIN_SAVE_SAMPLES
    ):
        raise TimingEvidenceError("raw timing evidence covers fewer than eight saves")

    if not isinstance(seed_bindings, list) or not seed_bindings:
        raise TimingEvidenceError("seed_bindings must be non-empty")
    normalized_seeds: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    for item in seed_bindings:
        if not isinstance(item, Mapping):
            raise TimingEvidenceError("seed binding must be a mapping")
        _exact_mapping_keys(item, frozenset({"role", "seed"}), "seed binding")
        role = _require_safe_id(item["role"], "seed role")
        seed = item["seed"]
        if (
            role in seen_roles
            or isinstance(seed, bool)
            or not isinstance(seed, Integral)
            or not 0 <= int(seed) < 2**32
        ):
            raise TimingEvidenceError("seed binding is invalid or duplicated")
        seen_roles.add(role)
        normalized_seeds.append({"role": role, "seed": int(seed)})
    normalized_seeds.sort(key=lambda row: row["role"])

    payload = {
        "schema": 1,
        "kind": "forge-krea-raw-timing-sample-manifest",
        "execution_envelope": envelope.to_record(),
        "probe_contract_sha256": _require_sha256(
            probe_contract_sha256, "probe_contract_sha256"
        ),
        "measurement_tool_sha256": tool_sha,
        "command_captures": normalized_captures,
        "samples": normalized_samples,
        # Seeds are evidence-bound, but intentionally outside the reusable
        # compute/runtime identity in execution_envelope.
        "seed_bindings": normalized_seeds,
    }
    return {**payload, "raw_sample_manifest_sha256": _payload_sha256(payload)}


def load_timing_sample_manifest(document: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise TimingEvidenceError("raw timing sample manifest must be a mapping")
    expected = frozenset(
        {
            "schema",
            "kind",
            "execution_envelope",
            "probe_contract_sha256",
            "measurement_tool_sha256",
            "command_captures",
            "samples",
            "seed_bindings",
            "raw_sample_manifest_sha256",
        }
    )
    _exact_mapping_keys(document, expected, "raw timing sample manifest")
    if (
        document["schema"] != 1
        or document["kind"] != "forge-krea-raw-timing-sample-manifest"
    ):
        raise TimingEvidenceError("unsupported raw timing sample manifest")
    normalized = seal_timing_sample_manifest(
        execution_envelope=document["execution_envelope"],
        probe_contract_sha256=document["probe_contract_sha256"],
        measurement_tool_sha256=document["measurement_tool_sha256"],
        command_captures=list(document["command_captures"]),
        samples=document["samples"],
        seed_bindings=list(document["seed_bindings"]),
    )
    claimed = _require_sha256(
        document["raw_sample_manifest_sha256"], "raw_sample_manifest_sha256"
    )
    if not hmac.compare_digest(claimed, normalized["raw_sample_manifest_sha256"]):
        raise TimingEvidenceError("raw timing sample manifest digest mismatch")
    return normalized


def seal_margin_policy(
    *,
    reviewer_identity: str,
    approved_at_utc: str,
    frozen_before_capture: bool,
    multiplicative_margin: Mapping[str, Any],
    additive_margin_s: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(reviewer_identity, str) or len(reviewer_identity.split()) < 2:
        raise TimingEvidenceError("margin policy requires a named human reviewer")
    if not isinstance(approved_at_utc, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", approved_at_utc
    ):
        raise TimingEvidenceError("approved_at_utc must be strict UTC seconds")
    try:
        normalized_approval_time = (
            datetime.strptime(approved_at_utc, "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )
    except ValueError as exc:
        raise TimingEvidenceError(
            "approved_at_utc is not a real UTC timestamp"
        ) from exc
    if frozen_before_capture is not True:
        raise TimingEvidenceError("margin policy must be frozen before timing capture")
    if not isinstance(multiplicative_margin, Mapping) or not isinstance(
        additive_margin_s, Mapping
    ):
        raise TimingEvidenceError("margin maps must be mappings")
    expected = frozenset(_TIMING_METRICS)
    _exact_mapping_keys(multiplicative_margin, expected, "multiplicative margins")
    _exact_mapping_keys(additive_margin_s, expected, "additive margins")
    normalized_multiplier: dict[str, float] = {}
    normalized_additive: dict[str, float] = {}
    for metric in _TIMING_METRICS:
        multiplier = _require_json_seconds(
            multiplicative_margin[metric], f"{metric} multiplicative margin"
        )
        if multiplier < 1.0:
            raise TimingEvidenceError(
                "multiplicative timing margins cannot be below one"
            )
        normalized_multiplier[metric] = multiplier
        normalized_additive[metric] = _nonnegative_json_seconds(
            additive_margin_s[metric], f"{metric} additive margin"
        )
    payload = {
        "schema": 1,
        "kind": "forge-krea-predeclared-timing-margin-policy",
        "reviewer_identity": " ".join(reviewer_identity.split()),
        "approved_at_utc": normalized_approval_time,
        "frozen_before_capture": True,
        "multiplicative_margin": normalized_multiplier,
        "additive_margin_s": normalized_additive,
    }
    return {**payload, "margin_policy_sha256": _payload_sha256(payload)}


def seal_agent_margin_policy(
    *,
    technical_reviewer_actor: Mapping[str, Any],
    discovery_execution_authorization: Mapping[str, Any],
    approved_at_utc: str,
    frozen_before_capture: bool,
    multiplicative_margin: Mapping[str, Any],
    additive_margin_s: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the exact owner-ratified Stage-1 margin via a fresh agent review."""

    try:
        from . import krea_delegated_review_contract
        from . import krea_discovery_authorization
    except ImportError:  # pragma: no cover - direct script execution.
        import krea_delegated_review_contract  # type: ignore[no-redef]
        import krea_discovery_authorization  # type: ignore[no-redef]

    _, authorization, _ = krea_discovery_authorization.load_binding(
        discovery_execution_authorization
    )
    actor = krea_delegated_review_contract.validate_actor(
        "timing_margin_reviewer",
        technical_reviewer_actor,
    )
    authorization_actor = authorization["technical_reviewer_actor"]
    if (
        actor["actor_id"] == authorization_actor["actor_id"]
        or actor["review_instance_id"] == authorization_actor["review_instance_id"]
    ):
        raise TimingEvidenceError("margin reviewer is not fresh from authorization")
    if not isinstance(approved_at_utc, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", approved_at_utc
    ):
        raise TimingEvidenceError("approved_at_utc must be strict UTC seconds")
    try:
        approved = datetime.strptime(approved_at_utc, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        authorized = datetime.strptime(
            authorization["authorized_at_utc"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise TimingEvidenceError("margin approval timestamp is invalid") from exc
    if approved < authorized:
        raise TimingEvidenceError("margin approval predates discovery authorization")
    if frozen_before_capture is not True:
        raise TimingEvidenceError("margin policy must be frozen before timing capture")
    delegated_contract = krea_delegated_review_contract.load()
    expected = delegated_contract["timing_margin_contract"]
    if set(multiplicative_margin) != set(_TIMING_METRICS) or set(
        additive_margin_s
    ) != set(_TIMING_METRICS):
        raise TimingEvidenceError("timing margin maps have the wrong metrics")
    normalized_multiplier = {
        name: _require_json_seconds(
            multiplicative_margin[name], f"{name} multiplicative margin"
        )
        for name in _TIMING_METRICS
    }
    normalized_additive = {
        name: _nonnegative_json_seconds(
            additive_margin_s[name], f"{name} additive margin"
        )
        for name in _TIMING_METRICS
    }
    if {
        "multiplicative_margin": normalized_multiplier,
        "additive_margin_s": normalized_additive,
    } != {
        "multiplicative_margin": expected["multiplicative_margin"],
        "additive_margin_s": expected["additive_margin_s"],
    }:
        raise TimingEvidenceError("timing margins differ from owner-ratified policy")
    payload = {
        "schema": 2,
        "kind": "forge-krea-agent-predeclared-timing-margin-policy",
        "technical_reviewer_actor": actor,
        "accountable_owner_identity": authorization["accountable_owner_identity"],
        "owner_ratification_sha256": authorization["fixture_admission_envelope"][
            "owner_ratification_sha256"
        ],
        "discovery_execution_authorization": dict(discovery_execution_authorization),
        "delegated_review_contract": krea_delegated_review_contract.binding(),
        "agent_review_is_not_human_review": True,
        "approved_at_utc": approved.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frozen_before_capture": True,
        "multiplicative_margin": normalized_multiplier,
        "additive_margin_s": normalized_additive,
    }
    return {**payload, "margin_policy_sha256": _payload_sha256(payload)}


def load_margin_policy(document: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise TimingEvidenceError("timing margin policy must be a mapping")
    if document.get("schema") == 2:
        expected_agent = frozenset(
            {
                "schema",
                "kind",
                "technical_reviewer_actor",
                "accountable_owner_identity",
                "owner_ratification_sha256",
                "discovery_execution_authorization",
                "delegated_review_contract",
                "agent_review_is_not_human_review",
                "approved_at_utc",
                "frozen_before_capture",
                "multiplicative_margin",
                "additive_margin_s",
                "margin_policy_sha256",
            }
        )
        _exact_mapping_keys(document, expected_agent, "timing margin policy")
        normalized = seal_agent_margin_policy(
            technical_reviewer_actor=document["technical_reviewer_actor"],
            discovery_execution_authorization=document[
                "discovery_execution_authorization"
            ],
            approved_at_utc=document["approved_at_utc"],
            frozen_before_capture=document["frozen_before_capture"],
            multiplicative_margin=document["multiplicative_margin"],
            additive_margin_s=document["additive_margin_s"],
        )
        if document.get("agent_review_is_not_human_review") is not True:
            raise TimingEvidenceError("agent margin is misattributed as human review")
        claimed = _require_sha256(
            document["margin_policy_sha256"], "margin_policy_sha256"
        )
        if not hmac.compare_digest(claimed, normalized["margin_policy_sha256"]):
            raise TimingEvidenceError("timing margin policy digest mismatch")
        return normalized
    expected = frozenset(
        {
            "schema",
            "kind",
            "reviewer_identity",
            "approved_at_utc",
            "frozen_before_capture",
            "multiplicative_margin",
            "additive_margin_s",
            "margin_policy_sha256",
        }
    )
    _exact_mapping_keys(document, expected, "timing margin policy")
    if (
        document["schema"] != 1
        or document["kind"] != "forge-krea-predeclared-timing-margin-policy"
    ):
        raise TimingEvidenceError("unsupported timing margin policy")
    normalized = seal_margin_policy(
        reviewer_identity=document["reviewer_identity"],
        approved_at_utc=document["approved_at_utc"],
        frozen_before_capture=document["frozen_before_capture"],
        multiplicative_margin=document["multiplicative_margin"],
        additive_margin_s=document["additive_margin_s"],
    )
    claimed = _require_sha256(document["margin_policy_sha256"], "margin_policy_sha256")
    if not hmac.compare_digest(claimed, normalized["margin_policy_sha256"]):
        raise TimingEvidenceError("timing margin policy digest mismatch")
    return normalized


def seal_end_to_end_validation(
    *,
    execution_envelope_sha256: str,
    probe_contract_sha256: str,
    runs: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(runs, list) or not runs:
        raise TimingEvidenceError("held-out end-to-end validation needs a run")
    normalized_runs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in runs:
        if not isinstance(item, Mapping):
            raise TimingEvidenceError("end-to-end run must be a mapping")
        _exact_mapping_keys(
            item,
            frozenset(
                {
                    "run_id",
                    "seed_role",
                    "seed",
                    "hard_budget_s",
                    "outer_wall_clock_s",
                    "natural_completion",
                    "upload_ready",
                    "failure_or_fallback_telemetry",
                    "run_record_sha256",
                }
            ),
            "end-to-end run",
        )
        run_id = _require_safe_id(item["run_id"], "end-to-end run_id")
        if run_id in seen:
            raise TimingEvidenceError("duplicate end-to-end run_id")
        seen.add(run_id)
        seed = item["seed"]
        if (
            isinstance(seed, bool)
            or not isinstance(seed, Integral)
            or not 0 <= int(seed) < 2**32
        ):
            raise TimingEvidenceError("end-to-end seed is invalid")
        hard = _require_json_seconds(item["hard_budget_s"], "hard_budget_s")
        wall = _require_json_seconds(item["outer_wall_clock_s"], "outer_wall_clock_s")
        if wall > hard:
            raise TimingEvidenceError("end-to-end run exceeded its hard budget")
        if (
            item["natural_completion"] is not True
            or item["upload_ready"] is not True
            or item["failure_or_fallback_telemetry"] is not False
        ):
            raise TimingEvidenceError(
                "end-to-end run is not a clean natural completion"
            )
        normalized_runs.append(
            {
                "run_id": run_id,
                "seed_role": _require_safe_id(item["seed_role"], "seed_role"),
                "seed": int(seed),
                "hard_budget_s": hard,
                "outer_wall_clock_s": wall,
                "natural_completion": True,
                "upload_ready": True,
                "failure_or_fallback_telemetry": False,
                "run_record_sha256": _require_sha256(
                    item["run_record_sha256"], "run_record_sha256"
                ),
            }
        )
    normalized_runs.sort(key=lambda row: row["run_id"])
    payload = {
        "schema": 1,
        "kind": "forge-krea-held-out-end-to-end-timing-validation",
        "execution_envelope_sha256": _require_sha256(
            execution_envelope_sha256, "execution_envelope_sha256"
        ),
        "probe_contract_sha256": _require_sha256(
            probe_contract_sha256, "probe_contract_sha256"
        ),
        "runs": normalized_runs,
    }
    return {**payload, "end_to_end_validation_sha256": _payload_sha256(payload)}


def load_end_to_end_validation(document: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise TimingEvidenceError("end-to-end validation must be a mapping")
    expected = frozenset(
        {
            "schema",
            "kind",
            "execution_envelope_sha256",
            "probe_contract_sha256",
            "runs",
            "end_to_end_validation_sha256",
        }
    )
    _exact_mapping_keys(document, expected, "end-to-end timing validation")
    if (
        document["schema"] != 1
        or document["kind"] != "forge-krea-held-out-end-to-end-timing-validation"
    ):
        raise TimingEvidenceError("unsupported end-to-end timing validation")
    normalized = seal_end_to_end_validation(
        execution_envelope_sha256=document["execution_envelope_sha256"],
        probe_contract_sha256=document["probe_contract_sha256"],
        runs=list(document["runs"]),
    )
    claimed = _require_sha256(
        document["end_to_end_validation_sha256"], "end_to_end_validation_sha256"
    )
    if not hmac.compare_digest(claimed, normalized["end_to_end_validation_sha256"]):
        raise TimingEvidenceError("end-to-end timing validation digest mismatch")
    return normalized


def seal_throughput_profile_from_evidence(
    *,
    raw_sample_manifest: Mapping[str, Any],
    margin_policy: Mapping[str, Any],
    end_to_end_validation: Mapping[str, Any],
    framework_stop_boundary_s: float,
    framework_stop_boundary_source_sha256: str,
    selection_mode: str = "offline_post_training",
    selection_scorer_identity_sha256: str | None = None,
    selection_scoring_reserve_s: float = 0.0,
) -> dict[str, Any]:
    """Recompute a throughput profile from the actual raw observations.

    This is the only Day-1 path that should create a profile.  It deliberately
    derives counts and bounds from the supplied records; callers cannot provide
    an opaque count/digest pair or hand-enter a favorable upper bound.
    """

    raw = load_timing_sample_manifest(raw_sample_manifest)
    margin = load_margin_policy(margin_policy)
    e2e = load_end_to_end_validation(end_to_end_validation)
    envelope = load_execution_envelope(raw["execution_envelope"])
    if (
        e2e["execution_envelope_sha256"] != envelope.execution_envelope_sha256
        or e2e["probe_contract_sha256"] != raw["probe_contract_sha256"]
    ):
        raise TimingEvidenceError(
            "end-to-end evidence escaped the timing probe envelope"
        )
    # Margin timestamps have whole-second precision.  Convert seconds before
    # multiplying so a binary float cannot move the approval across a
    # nanosecond capture boundary.
    approved_seconds = int(
        datetime.strptime(margin["approved_at_utc"], "%Y-%m-%dT%H:%M:%SZ")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    approved_ns = approved_seconds * 1_000_000_000
    first_capture_ns = min(row["started_unix_ns"] for row in raw["command_captures"])
    if approved_ns > first_capture_ns:
        raise TimingEvidenceError(
            "timing margin policy was approved after capture began"
        )

    def bound(metric: str) -> float:
        per_unit = [
            float(row["duration_s"]) / int(row["units"])
            for row in raw["samples"][metric]
        ]
        return max(per_unit) * float(margin["multiplicative_margin"][metric]) + float(
            margin["additive_margin_s"][metric]
        )

    startup_count = len(raw["samples"]["startup"])
    update_count = sum(row["units"] for row in raw["samples"]["optimizer_update"])
    save_count = sum(row["units"] for row in raw["samples"]["checkpoint_save"])
    return seal_throughput_profile(
        execution_envelope=envelope.to_record(),
        raw_sample_manifest_sha256=raw["raw_sample_manifest_sha256"],
        startup_sample_count=startup_count,
        update_sample_count=update_count,
        save_sample_count=save_count,
        startup_upper_bound_s=bound("startup"),
        update_upper_bound_s=bound("optimizer_update"),
        save_upper_bound_s=bound("checkpoint_save"),
        bound_method="observed-max-plus-predeclared-margin",
        margin_policy_sha256=margin["margin_policy_sha256"],
        end_to_end_validation_count=len(e2e["runs"]),
        end_to_end_validation_sha256=e2e["end_to_end_validation_sha256"],
        framework_stop_boundary_s=framework_stop_boundary_s,
        framework_stop_boundary_source_sha256=framework_stop_boundary_source_sha256,
        selection_mode=selection_mode,
        selection_scorer_identity_sha256=selection_scorer_identity_sha256,
        selection_scoring_reserve_s=selection_scoring_reserve_s,
        finalization_reserve_s=bound("finalization"),
        upload_reserve_s=bound("upload"),
    )


@dataclass(frozen=True)
class Candidate:
    step: int
    fraction: Decimal
    kind: str

    def to_record(self, total_steps: int) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "step": self.step,
            "fraction": _decimal_text(self.fraction),
            "fraction_numerator": self.step,
            "fraction_denominator": total_steps,
        }


@dataclass(frozen=True)
class CandidateMapping:
    desired: str
    target_fraction: Decimal
    actual_step: int
    actual_fraction: Decimal
    absolute_fraction_error: Decimal

    def to_record(self) -> dict[str, Any]:
        return {
            "desired": self.desired,
            "target_fraction": _decimal_text(self.target_fraction),
            "actual_step": self.actual_step,
            "actual_fraction": _decimal_text(self.actual_fraction),
            "absolute_fraction_error": _decimal_text(self.absolute_fraction_error),
        }


@dataclass(frozen=True)
class CandidateSchedule:
    total_steps: int
    save_every: int
    periodic_write_steps: tuple[int, ...]
    candidates: tuple[Candidate, ...]
    mappings: tuple[CandidateMapping, ...]

    @property
    def periodic_save_count(self) -> int:
        return len(self.periodic_write_steps)

    def to_record(self, *, images_per_update: int = 1) -> dict[str, Any]:
        return {
            "save_every": self.save_every,
            "periodic_write_steps": list(self.periodic_write_steps),
            "actual_candidates": [
                {
                    **candidate.to_record(self.total_steps),
                    "image_exposures": candidate.step * images_per_update,
                }
                for candidate in self.candidates
            ],
            "desired_mappings": [mapping.to_record() for mapping in self.mappings],
        }


def candidate_schedule(total_steps: int) -> CandidateSchedule:
    """Return the real candidates allowed by one uniform ai-toolkit cadence."""

    if isinstance(total_steps, bool) or not isinstance(total_steps, Integral):
        raise ValueError("total_steps must be a positive integer")
    steps = int(total_steps)
    if steps <= 0:
        raise ValueError("total_steps must be a positive integer")

    # ceil(steps / 8), expressed with integer arithmetic. Candidate identity
    # deduplicates a terminal numbered save from the final export, but runtime
    # accounting below charges both writes when the cadence lands on terminal.
    save_every = (steps + 7) // 8
    # ai-toolkit writes a numbered periodic checkpoint even when the cadence
    # lands exactly on the terminal update, and Forge then writes/promotes its
    # terminal artifact. Charge every real periodic write, while representing a
    # terminal duplicate only once in the candidate set.
    periodic_write_steps = tuple(range(save_every, steps + 1, save_every))
    periodic_steps = tuple(step for step in periodic_write_steps if step < steps)
    candidates = tuple(
        Candidate(step, Decimal(step) / Decimal(steps), "periodic")
        for step in periodic_steps
    ) + (
        Candidate(steps, Decimal(1), "final"),
    )

    mappings: list[CandidateMapping] = []
    for label, target in _DESIRED_CANDIDATES:
        if label == "final":
            selected = candidates[-1]
        else:
            # Earlier wins exact ties, avoiding an accidental claim that a
            # checkpoint reached more of the run than it actually did.
            selected = min(
                candidates,
                key=lambda candidate: (
                    abs(candidate.fraction - target),
                    candidate.step,
                ),
            )
        mappings.append(
            CandidateMapping(
                desired=label,
                target_fraction=target,
                actual_step=selected.step,
                actual_fraction=selected.fraction,
                absolute_fraction_error=abs(selected.fraction - target),
            )
        )

    return CandidateSchedule(
        total_steps=steps,
        save_every=save_every,
        periodic_write_steps=periodic_write_steps,
        candidates=candidates,
        mappings=tuple(mappings),
    )


@dataclass(frozen=True)
class BudgetPlan:
    profile_sha256: str
    execution_envelope: ExecutionEnvelope
    hard_budget_s: Decimal
    max_affordable_steps: int
    schedule: CandidateSchedule
    startup_s: Decimal
    update_runtime_s: Decimal
    periodic_save_runtime_s: Decimal
    selection_scoring_reserve_s: Decimal
    framework_stop_boundary_s: Decimal
    effective_training_stop_reserve_s: Decimal
    finalization_reserve_s: Decimal
    upload_reserve_s: Decimal
    planned_runtime_s: Decimal
    slack_s: Decimal
    budget_utilization: Decimal
    update_budget_utilization: Decimal
    post_reserve_training_utilization: Decimal
    minimum_post_reserve_training_utilization: Decimal
    save_overhead_fraction: Decimal
    maximum_save_overhead_fraction: Decimal

    @property
    def images_per_update(self) -> int:
        return self.execution_envelope.images_per_update

    @property
    def total_image_exposures(self) -> int:
        return self.max_affordable_steps * self.images_per_update

    raw_sample_manifest_sha256: str
    bound_method: str
    margin_policy_sha256: str
    end_to_end_validation_count: int
    end_to_end_validation_sha256: str
    framework_stop_boundary_source_sha256: str
    selection_mode: str
    selection_scorer_identity_sha256: str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": _PLAN_SCHEMA_VERSION,
            "model_type": _MODEL_TYPE,
            "profile_sha256": self.profile_sha256,
            "execution_envelope": self.execution_envelope.to_record(),
            "hard_budget_s": _decimal_text(self.hard_budget_s),
            "max_affordable_steps": self.max_affordable_steps,
            "training_geometry": {
                "micro_batch_size": self.execution_envelope.micro_batch_size,
                "gradient_accumulation_steps": (
                    self.execution_envelope.gradient_accumulation_steps
                ),
                "data_parallel_replicas": (
                    self.execution_envelope.data_parallel_replicas
                ),
                "images_per_update": self.images_per_update,
                "total_image_exposures": self.total_image_exposures,
                "resolution_policy_sha256": (
                    self.execution_envelope.resolution_policy_sha256
                ),
                "precision_policy_sha256": (
                    self.execution_envelope.precision_policy_sha256
                ),
            },
            "timing_evidence": {
                "raw_sample_manifest_sha256": self.raw_sample_manifest_sha256,
                "bound_method": self.bound_method,
                "margin_policy_sha256": self.margin_policy_sha256,
                "end_to_end_validation_count": self.end_to_end_validation_count,
                "end_to_end_validation_sha256": self.end_to_end_validation_sha256,
                "framework_stop_boundary_source_sha256": (
                    self.framework_stop_boundary_source_sha256
                ),
            },
            "selection": {
                "mode": self.selection_mode,
                "scorer_identity_sha256": self.selection_scorer_identity_sha256,
            },
            **self.schedule.to_record(images_per_update=self.images_per_update),
            "accounting": {
                "startup_upper_bound_s": _decimal_text(self.startup_s),
                "update_runtime_upper_bound_s": _decimal_text(self.update_runtime_s),
                "periodic_save_runtime_upper_bound_s": _decimal_text(
                    self.periodic_save_runtime_s
                ),
                "periodic_save_count": self.schedule.periodic_save_count,
                "selection_scoring_reserve_s": _decimal_text(
                    self.selection_scoring_reserve_s
                ),
                "framework_stop_boundary_s": _decimal_text(
                    self.framework_stop_boundary_s
                ),
                "effective_training_stop_reserve_s": _decimal_text(
                    self.effective_training_stop_reserve_s
                ),
                "finalization_reserve_s": _decimal_text(self.finalization_reserve_s),
                "upload_reserve_s": _decimal_text(self.upload_reserve_s),
                "planned_runtime_s": _decimal_text(self.planned_runtime_s),
                "slack_s": _decimal_text(self.slack_s),
                "budget_utilization": _decimal_text(self.budget_utilization),
                "update_budget_utilization": _decimal_text(
                    self.update_budget_utilization
                ),
                "post_reserve_training_utilization": _decimal_text(
                    self.post_reserve_training_utilization
                ),
                "minimum_post_reserve_training_utilization": _decimal_text(
                    self.minimum_post_reserve_training_utilization
                ),
                "save_overhead_fraction": _decimal_text(self.save_overhead_fraction),
                "maximum_save_overhead_fraction": _decimal_text(
                    self.maximum_save_overhead_fraction
                ),
            },
        }


def plan_budget(
    profile: ThroughputProfile,
    *,
    hard_budget_s: Any,
    minimum_post_reserve_training_utilization: Any = Decimal("0.90"),
    maximum_save_overhead_fraction: Any = Decimal("0.10"),
) -> BudgetPlan:
    """Derive the largest naturally completable Krea run from measurements.

    Periodic checkpoint I/O is charged at its evidence-bound conservative upper
    bound. Selection scoring, finalization, and upload each have explicit
    reserves. If one update cannot fit, this function raises rather than
    silently substituting a guessed step count.
    """

    if not isinstance(profile, ThroughputProfile):
        raise TypeError("profile must be a validated ThroughputProfile")
    # Re-validate to prevent a manually constructed dataclass from bypassing the
    # content binding.
    profile = load_throughput_profile(profile.to_record())
    budget = _positive_decimal(hard_budget_s, "hard_budget_s")
    minimum_utilization = _fraction_decimal(
        minimum_post_reserve_training_utilization,
        "minimum_post_reserve_training_utilization",
    )
    maximum_save_fraction = _fraction_decimal(
        maximum_save_overhead_fraction,
        "maximum_save_overhead_fraction",
    )
    if minimum_utilization < _FROZEN_MINIMUM_UTILIZATION:
        raise ValueError(
            "minimum_post_reserve_training_utilization cannot relax the frozen "
            "0.90 policy"
        )
    if maximum_save_fraction > _FROZEN_MAXIMUM_SAVE_OVERHEAD:
        raise ValueError(
            "maximum_save_overhead_fraction cannot relax the frozen 0.10 policy"
        )
    startup = Decimal(str(profile.startup_upper_bound_s))
    update = Decimal(str(profile.update_upper_bound_s))
    save = Decimal(str(profile.save_upper_bound_s))
    selection_scoring = Decimal(str(profile.selection_scoring_reserve_s))
    framework_stop_boundary = Decimal(str(profile.framework_stop_boundary_s))
    finalization = Decimal(str(profile.finalization_reserve_s))
    upload = Decimal(str(profile.upload_reserve_s))
    # Forge's inner deadline stops ai-toolkit at least 225 seconds before the
    # hard kill (180-second export reserve plus 45-second stop margin). That
    # boundary overlaps, rather than adds to, measured finalization/upload.
    # Use the larger obligation, while accounting for any live selector
    # separately because it also consumes the post-training window.
    effective_training_stop_reserve = max(
        framework_stop_boundary,
        finalization + upload,
    )
    available_for_updates_and_saves = (
        budget - startup - selection_scoring - effective_training_stop_reserve
    )

    if available_for_updates_and_saves <= 0:
        raise InsufficientBudgetError(
            "budget does not cover startup, selection scoring, and the larger "
            "of the framework stop boundary or finalization/upload reserves"
        )
    upper_without_saves = int(
        (available_for_updates_and_saves / update).to_integral_value(
            rounding=ROUND_FLOOR
        )
    )
    if upper_without_saves < 1:
        raise InsufficientBudgetError("budget does not cover one measured update")

    def fits(steps: int) -> bool:
        schedule = candidate_schedule(steps)
        training_runtime = (
            Decimal(steps) * update + Decimal(schedule.periodic_save_count) * save
        )
        return training_runtime <= available_for_updates_and_saves

    # From 57 steps onward ceil(steps/8) emits seven periodic writes except
    # exact multiples of eight, which emit eight. The seven-write cap is still
    # an upper bound on affordable depth; at most one decrement is needed when
    # that cap lands on an eight-write boundary.
    after_seven_saves = available_for_updates_and_saves - Decimal(7) * save
    seven_save_cap = (
        int((after_seven_saves / update).to_integral_value(rounding=ROUND_FLOOR))
        if after_seven_saves >= update
        else 0
    )
    if seven_save_cap >= 57:
        max_steps = min(upper_without_saves, seven_save_cap)
        if not fits(max_steps):
            max_steps -= 1
    else:
        # Below 57, cadence boundaries can reduce the save count.  Exhausting
        # this fixed 56-element domain is clearer and safer than assuming
        # monotonicity around a boundary.
        ceiling = min(56, upper_without_saves)
        feasible = [steps for steps in range(1, ceiling + 1) if fits(steps)]
        if not feasible:
            raise InsufficientBudgetError(
                "budget does not cover one update plus required reserves"
            )
        max_steps = max(feasible)

    if not fits(max_steps):
        raise AssertionError("internal planner error: selected plan exceeds budget")
    if fits(max_steps + 1):
        raise AssertionError("internal planner error: selected plan is not maximal")

    schedule = candidate_schedule(max_steps)
    update_runtime = Decimal(max_steps) * update
    save_runtime = Decimal(schedule.periodic_save_count) * save
    planned = (
        startup
        + update_runtime
        + save_runtime
        + selection_scoring
        + finalization
        + upload
    )
    slack = budget - planned
    # The frozen 90% utilization gate measures the post-fixed-reserve window
    # occupied by optimizer updates plus checkpoint I/O. The independent 10%
    # cap prevents save traffic from satisfying that gate by itself.
    post_reserve_utilization = (
        update_runtime + save_runtime
    ) / available_for_updates_and_saves
    save_overhead_fraction = save_runtime / available_for_updates_and_saves
    if post_reserve_utilization < minimum_utilization:
        raise InsufficientBudgetError(
            "maximal plan does not meet minimum post-reserve training utilization: "
            f"{_decimal_text(post_reserve_utilization)} < "
            f"{_decimal_text(minimum_utilization)}"
        )
    if save_overhead_fraction > maximum_save_fraction:
        raise InsufficientBudgetError(
            "candidate cadence exceeds maximum save-overhead fraction: "
            f"{_decimal_text(save_overhead_fraction)} > "
            f"{_decimal_text(maximum_save_fraction)}"
        )
    return BudgetPlan(
        profile_sha256=profile.profile_sha256,
        execution_envelope=profile.execution_envelope,
        hard_budget_s=budget,
        max_affordable_steps=max_steps,
        schedule=schedule,
        startup_s=startup,
        update_runtime_s=update_runtime,
        periodic_save_runtime_s=save_runtime,
        selection_scoring_reserve_s=selection_scoring,
        framework_stop_boundary_s=framework_stop_boundary,
        effective_training_stop_reserve_s=effective_training_stop_reserve,
        finalization_reserve_s=finalization,
        upload_reserve_s=upload,
        planned_runtime_s=planned,
        slack_s=slack,
        budget_utilization=planned / budget,
        update_budget_utilization=update_runtime / budget,
        post_reserve_training_utilization=post_reserve_utilization,
        minimum_post_reserve_training_utilization=minimum_utilization,
        save_overhead_fraction=save_overhead_fraction,
        maximum_save_overhead_fraction=maximum_save_fraction,
        raw_sample_manifest_sha256=profile.raw_sample_manifest_sha256,
        bound_method=profile.bound_method,
        margin_policy_sha256=profile.margin_policy_sha256,
        end_to_end_validation_count=profile.end_to_end_validation_count,
        end_to_end_validation_sha256=profile.end_to_end_validation_sha256,
        framework_stop_boundary_source_sha256=(
            profile.framework_stop_boundary_source_sha256
        ),
        selection_mode=profile.selection_mode,
        selection_scorer_identity_sha256=profile.selection_scorer_identity_sha256,
    )
