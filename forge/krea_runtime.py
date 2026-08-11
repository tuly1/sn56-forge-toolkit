"""Fail-closed Week-6 Krea runtime bundles and effective-runtime records.

The public August-3 rank-1 config contains several keys that the incumbent
ai-toolkit silently ignores.  Copying that YAML onto the incumbent runtime
would therefore run a different experiment while claiming byte equivalence.
The stable ``leader-*`` identifiers below are compatibility IDs for
source-derived candidates, not claims that Forge reproduced every public byte.
This module keeps the deployed ``incumbent-v1`` runtime isolated and makes every
experimental bundle conditional on a capability manifest produced by the
owned, conformance-tested Krea-only runtime fork.

Nothing in this module selects a production winner.  ``leader-v1``,
``leader-comfy-te-v1``, and ``mae-g3-v1`` are calibration candidates only; the
environment must request one
explicitly.  Unknown bundles and incomplete manifests are fatal by design.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from typing import Any, Mapping

import yaml

from forge import telemetry
from forge.file_evidence import RegularFileError, read_regular_bytes

BUNDLE_ENV = "FORGE_KREA_BUNDLE"
TIMING_PROBE_ENV = "FORGE_KREA_TIMING_PROBE"
INCUMBENT_RUNTIME_DIR_ENV = "AI_TOOLKIT_DIR"
OWNED_KREA_RUNTIME_DIR_ENV = "FORGE_KREA_AI_TOOLKIT_DIR"
INCUMBENT_BUNDLE = "incumbent-v1"
LEADER_BUNDLE = "leader-v1"
LEADER_COMFY_TE_BUNDLE = "leader-comfy-te-v1"
MAE_BUNDLE = "mae-g3-v1"
WEEK7_FACTORIAL_NO_MULTIRES_BUNDLE = "week7-factorial-nomr-v1"
WEEK7_FACTORIAL_MULTIRES_BUNDLE = "week7-factorial-mr-v1"
KNOWN_BUNDLES = frozenset(
    {
        INCUMBENT_BUNDLE,
        LEADER_BUNDLE,
        LEADER_COMFY_TE_BUNDLE,
        MAE_BUNDLE,
        WEEK7_FACTORIAL_NO_MULTIRES_BUNDLE,
        WEEK7_FACTORIAL_MULTIRES_BUNDLE,
    }
)

RUNTIME_CONTRACT_ID = "sn56-krea-runtime-v1"
PINNED_BASE_COMMIT = "99be3d96a2468d3a5228a4eb05ba67e63c586b4e"
PINNED_BASE_TREE = "9f7434621f06949655c5ebe13f45247ccf605c05"
OWNED_RUNTIME_COMMIT = "71e133b4e73a716d1094f22355a46be07953b828"
OWNED_RUNTIME_TREE = "61456aab4c4f18712414a54c2ef7333fadedf59e"
OWNED_RUNTIME_REPOSITORY = "https://github.com/tuly1/sn56-ai-toolkit-mirror.git"
INCUMBENT_RUNTIME_REPOSITORY = "https://github.com/ostris/ai-toolkit.git"
DEFAULT_INCUMBENT_RUNTIME_DIR = "/app/ai-toolkit"
DEFAULT_OWNED_KREA_RUNTIME_DIR = "/opt/sn56/krea-ai-toolkit"
CAPABILITY_MANIFEST_FILENAME = "sn56_krea_runtime_capabilities.json"
RUNTIME_IDENTITY_FILENAME = ".sn56-runtime-identity.json"
_MAX_ATTESTATION_BYTES = 256 * 1024

PUBLIC_RANK1_CONFIG_SHA256 = (
    "50fe6eec02281d0e8acf0ea7d3d3b15b3b320a1ad0b6b6d450e17930dbd5dc1c"
)
PUBLIC_RANK3_CONFIG_SHA256 = (
    "eceb74aef768cb1cd62212abd4e05c6fd012da6bfb866106ea75f1af66e72307"
)
PUBLIC_RANK1_REPOSITORY = (
    "https://huggingface.co/gradients-io-tournaments/"
    "tournament-tourn_c54bb970b5d0aa91_20260803-"
    "41025fb5-8473-40c6-a88d-20c0bb303edc-5EACrayt"
)
PUBLIC_RANK1_REVISION = "a28c6a0f64c06bf81e191515a1d80e04fc793b44"
PUBLIC_RANK3_REPOSITORY = (
    "https://huggingface.co/gradients-io-tournaments/"
    "tournament-tourn_c54bb970b5d0aa91_20260803-"
    "41025fb5-8473-40c6-a88d-20c0bb303edc-5FBmn1ax"
)
PUBLIC_RANK3_REVISION = "63f94211664970831c9e3575a3e373a7720f4254"
PUBLIC_CONFIG_PATH = "checkpoints/config.yaml"

_BUNDLE_CLAIMS = {
    INCUMBENT_BUNDLE: {
        "classification": "incumbent-production-control",
        "source_relationship": "deployed-config-and-runtime-control",
        "source_repository": None,
        "source_revision": None,
        "source_config_path": None,
        "source_config_sha256": None,
        "byte_equivalent_to_source_config": True,
    },
    LEADER_BUNDLE: {
        "classification": "source-derived-public-rank1-positive-control",
        "source_relationship": "selected-fields-derived-from-public-config",
        "source_repository": PUBLIC_RANK1_REPOSITORY,
        "source_revision": PUBLIC_RANK1_REVISION,
        "source_config_path": PUBLIC_CONFIG_PATH,
        "source_config_sha256": PUBLIC_RANK1_CONFIG_SHA256,
        "byte_equivalent_to_source_config": False,
    },
    LEADER_COMFY_TE_BUNDLE: {
        "classification": "source-derived-rank1-plus-export-hypothesis",
        "source_relationship": (
            "selected-fields-derived-from-public-config-plus-owned-export-extension"
        ),
        "source_repository": PUBLIC_RANK1_REPOSITORY,
        "source_revision": PUBLIC_RANK1_REVISION,
        "source_config_path": PUBLIC_CONFIG_PATH,
        "source_config_sha256": PUBLIC_RANK1_CONFIG_SHA256,
        "byte_equivalent_to_source_config": False,
    },
    MAE_BUNDLE: {
        "classification": "source-derived-public-rank3-positive-control",
        "source_relationship": "selected-fields-derived-from-public-config",
        "source_repository": PUBLIC_RANK3_REPOSITORY,
        "source_revision": PUBLIC_RANK3_REVISION,
        "source_config_path": PUBLIC_CONFIG_PATH,
        "source_config_sha256": PUBLIC_RANK3_CONFIG_SHA256,
        "byte_equivalent_to_source_config": False,
    },
    WEEK7_FACTORIAL_NO_MULTIRES_BUNDLE: {
        "classification": "week7-dormant-factorial-owned-runtime-control",
        "source_relationship": "reviewed-incumbent-config-on-owned-runtime",
        "source_repository": None,
        "source_revision": None,
        "source_config_path": None,
        "source_config_sha256": None,
        "byte_equivalent_to_source_config": False,
    },
    WEEK7_FACTORIAL_MULTIRES_BUNDLE: {
        "classification": "week7-dormant-factorial-multires-candidate",
        "source_relationship": (
            "reviewed-incumbent-config-plus-predeclared-multires-fields"
        ),
        "source_repository": None,
        "source_revision": None,
        "source_config_path": None,
        "source_config_sha256": None,
        "byte_equivalent_to_source_config": False,
    },
}

# One named assertion per silent-runtime failure found in the Week-6 audit.
COMPONENT_RECOVERY_CAPABILITY = "component_consistent_ema_optimizer_recovery"
REQUIRED_CAPABILITIES = (
    "qwen3vl_text_encoder_lora",
    "optimizer_group_lr_split",
    "krea2_eval_sigmas",
    "cosine_by_group",
    "multires_noise",
    "ungated_differential_guidance",
    COMPONENT_RECOVERY_CAPABILITY,
    "strict_unknown_train_field_rejection",
)

# The immutable schema-v1 runtime tag predates the corrected guarantee name.
# Forge exposes only the honest semantic name; this one explicit translation is
# retained at the wire boundary until a future runtime-schema revision.
_RUNTIME_CAPABILITY_WIRE_ALIASES = {
    COMPONENT_RECOVERY_CAPABILITY: "ema_checkpoint_resume",
}
RUNTIME_MANIFEST_CAPABILITIES = tuple(
    _RUNTIME_CAPABILITY_WIRE_ALIASES.get(name, name) for name in REQUIRED_CAPABILITIES
)

_BUNDLE_CAPABILITIES = {
    LEADER_BUNDLE: REQUIRED_CAPABILITIES,
    LEADER_COMFY_TE_BUNDLE: REQUIRED_CAPABILITIES,
    MAE_BUNDLE: (
        "ungated_differential_guidance",
        "strict_unknown_train_field_rejection",
    ),
    WEEK7_FACTORIAL_NO_MULTIRES_BUNDLE: ("strict_unknown_train_field_rejection",),
    WEEK7_FACTORIAL_MULTIRES_BUNDLE: (
        "multires_noise",
        "strict_unknown_train_field_rejection",
    ),
}


class KreaRuntimeContractError(RuntimeError):
    """An experimental recipe cannot be represented by the active runtime."""


@dataclasses.dataclass
class VerifiedRuntime:
    """Private immutable materialization of the verified runtime tree.

    The selected checkout remains untrusted mutable input.  Verification copies
    its exact bytes into a private directory, verifies the copied Git tree, and
    carries that directory descriptor into ``Popen``.  Neither replacing the
    checkout path nor modifying one of its children can redirect execution.
    """

    runtime_dir: str
    materialized_dir: str
    directory_fd: int

    def launch_cwd(self) -> str:
        for root in ("/proc/self/fd", "/dev/fd"):
            candidate = f"{root}/{self.directory_fd}"
            if os.path.isdir(candidate):
                return candidate
        raise KreaRuntimeContractError(
            "verified runtime descriptor has no executable fd namespace"
        )

    def popen_directory_kwargs(self) -> dict[str, Any]:
        """Return platform-specific, descriptor-bound ``Popen`` arguments."""

        proc_path = f"/proc/self/fd/{self.directory_fd}"
        if os.path.isdir(proc_path):
            return {
                "cwd": proc_path,
                "pass_fds": (self.directory_fd,),
            }

        # macOS exposes directory descriptors under /dev/fd but its subprocess
        # chdir rejects that synthetic path.  No sampler thread exists yet at
        # this boundary, so the fallback performs the single async-safe fchdir
        # operation in the child before exec.  Production Linux takes the
        # /proc path above and does not use preexec_fn.
        return {
            "cwd": None,
            "pass_fds": (self.directory_fd,),
            "preexec_fn": self._fchdir_for_exec,
        }

    def _fchdir_for_exec(self) -> None:
        os.fchdir(self.directory_fd)

    def close(self) -> None:
        if self.directory_fd >= 0:
            os.close(self.directory_fd)
            self.directory_fd = -1
        if self.materialized_dir:
            _remove_materialized_runtime(self.materialized_dir)
            self.materialized_dir = ""

    def __enter__(self) -> "VerifiedRuntime":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def requested_bundle(model_type: str, environ: dict[str, str] | None = None) -> str:
    """Resolve the explicit Krea bundle without affecting any other model type."""
    if (model_type or "").strip().lower() != "krea2":
        return INCUMBENT_BUNDLE
    env = os.environ if environ is None else environ
    value = str(env.get(BUNDLE_ENV, INCUMBENT_BUNDLE)).strip().lower()
    if value not in KNOWN_BUNDLES:
        raise KreaRuntimeContractError(f"unknown Krea bundle: {value!r}")
    return value


def runtime_directory(
    model_type: str,
    bundle: str | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> str:
    """Select the owned fork only for an explicit experimental Krea bundle."""

    env = os.environ if environ is None else environ
    resolved = requested_bundle(model_type, env) if bundle is None else bundle
    if resolved not in KNOWN_BUNDLES:
        raise KreaRuntimeContractError(f"unknown Krea bundle: {resolved!r}")
    is_experimental_krea = (
        model_type or ""
    ).strip().lower() == "krea2" and resolved != INCUMBENT_BUNDLE
    incumbent_dir, owned_dir = _isolated_runtime_directories(env)
    return owned_dir if is_experimental_krea else incumbent_dir


def _isolated_runtime_directories(
    environ: Mapping[str, str],
) -> tuple[str, str]:
    """Resolve both runtime roots and reject aliasing or containment.

    A separately configurable executable and attestation path recreates the
    exact failure this contract is meant to prevent.  Both roots therefore
    enter through one resolver, and even a symlink alias or nested checkout is
    rejected before either runtime can launch.
    """

    resolved: list[str] = []
    for variable, default in (
        (INCUMBENT_RUNTIME_DIR_ENV, DEFAULT_INCUMBENT_RUNTIME_DIR),
        (OWNED_KREA_RUNTIME_DIR_ENV, DEFAULT_OWNED_KREA_RUNTIME_DIR),
    ):
        raw = environ.get(variable, default)
        if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
            raise KreaRuntimeContractError("runtime directory is invalid")
        if not os.path.isabs(raw):
            raise KreaRuntimeContractError("runtime directory must be absolute")
        resolved.append(os.path.realpath(raw))

    incumbent_dir, owned_dir = resolved
    try:
        common = os.path.commonpath((incumbent_dir, owned_dir))
    except ValueError as exc:  # pragma: no cover - unlike roots on Windows
        raise KreaRuntimeContractError("runtime directories are incomparable") from exc
    if common in {incumbent_dir, owned_dir}:
        raise KreaRuntimeContractError(
            "incumbent and owned runtime directories overlap"
        )
    try:
        incumbent_identity = os.stat(incumbent_dir, follow_symlinks=False)
        owned_identity = os.stat(owned_dir, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        if (incumbent_identity.st_dev, incumbent_identity.st_ino) == (
            owned_identity.st_dev,
            owned_identity.st_ino,
        ):
            raise KreaRuntimeContractError(
                "incumbent and owned runtime directories alias the same root"
            )
    return incumbent_dir, owned_dir


def runtime_attestation_paths(
    model_type: str,
    bundle: str,
    *,
    environ: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    """Derive every attestation path from the checkout that will execute."""

    runtime_dir = runtime_directory(model_type, bundle, environ=environ)
    return (
        runtime_dir,
        os.path.join(runtime_dir, CAPABILITY_MANIFEST_FILENAME),
        os.path.join(runtime_dir, RUNTIME_IDENTITY_FILENAME),
    )


def runtime_commit_for_bundle(bundle: str) -> str:
    if bundle not in KNOWN_BUNDLES:
        raise KreaRuntimeContractError(f"unknown Krea bundle: {bundle!r}")
    return PINNED_BASE_COMMIT if bundle == INCUMBENT_BUNDLE else OWNED_RUNTIME_COMMIT


def runtime_repository_for_bundle(bundle: str) -> str:
    if bundle not in KNOWN_BUNDLES:
        raise KreaRuntimeContractError(f"unknown Krea bundle: {bundle!r}")
    return (
        INCUMBENT_RUNTIME_REPOSITORY
        if bundle == INCUMBENT_BUNDLE
        else OWNED_RUNTIME_REPOSITORY
    )


def bundle_claim_document(bundle: str) -> dict[str, Any]:
    """Return an honest source relationship while preserving stable bundle IDs."""

    if bundle not in KNOWN_BUNDLES:
        raise KreaRuntimeContractError(f"unknown Krea bundle: {bundle!r}")
    return copy.deepcopy(_BUNDLE_CLAIMS[bundle])


def timing_probe_enabled(environ: dict[str, str] | None = None) -> bool:
    """Allow one explicitly labeled bootstrap run before a profile exists."""
    env = os.environ if environ is None else environ
    raw = env.get(TIMING_PROBE_ENV)
    if raw is None or raw == "0":
        return False
    if raw == "1":
        return True
    raise KreaRuntimeContractError(f"{TIMING_PROBE_ENV} must be literal 0 or 1")


def apply(
    cfg: dict[str, Any],
    model_type: str,
    *,
    environ: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return ``(config, manifest)`` for the requested runtime bundle.

    The incumbent returns the original object untouched.  Experimental bundles
    are applied to a deep copy only after the complete runtime contract passes.
    This prevents a partial mutation from becoming a plausible-looking run.
    """
    bundle = requested_bundle(model_type, environ)
    if bundle == INCUMBENT_BUNDLE:
        return cfg, None

    manifest = load_capability_manifest(
        model_type=model_type,
        bundle=bundle,
        environ=environ,
    )
    require_capabilities(manifest, _BUNDLE_CAPABILITIES[bundle])
    candidate = copy.deepcopy(cfg)
    if bundle in {LEADER_BUNDLE, LEADER_COMFY_TE_BUNDLE}:
        _apply_source_derived_rank1(candidate)
        if bundle == LEADER_COMFY_TE_BUNDLE:
            candidate["config"]["process"][0]["train"][
                "sn56_krea_comfy_text_encoder_export"
            ] = True
    elif bundle == MAE_BUNDLE:
        _apply_mae(candidate)
    elif bundle in {
        WEEK7_FACTORIAL_NO_MULTIRES_BUNDLE,
        WEEK7_FACTORIAL_MULTIRES_BUNDLE,
    }:
        _apply_week7_factorial(candidate, bundle=bundle)
    else:  # guarded by requested_bundle; defense against future drift.
        raise KreaRuntimeContractError(f"unimplemented Krea bundle: {bundle}")
    _validate_effective_bundle(candidate, bundle)
    if bundle not in {
        WEEK7_FACTORIAL_NO_MULTIRES_BUNDLE,
        WEEK7_FACTORIAL_MULTIRES_BUNDLE,
    } and timing_contract_projection(candidate, bundle=bundle) != (
        _reference_bundle_projection(bundle)
    ):
        raise KreaRuntimeContractError(
            "Krea bundle normalized config projection drifted"
        )
    return candidate, manifest


def materialize_week7_factorial_config(
    cfg: dict[str, Any], *, multires_noise: bool
) -> tuple[dict[str, Any], str]:
    """Materialize a dormant Week-7 factorial cell without claiming launch.

    Capability evidence is necessarily host/runtime specific and is verified by
    :func:`apply` at launch.  This pure planner helper returns the exact config
    and bundle ID that launch must select; it never loads the leader overlays.
    """

    bundle = (
        WEEK7_FACTORIAL_MULTIRES_BUNDLE
        if multires_noise
        else WEEK7_FACTORIAL_NO_MULTIRES_BUNDLE
    )
    candidate = copy.deepcopy(cfg)
    _apply_week7_factorial(candidate, bundle=bundle)
    _validate_effective_bundle(candidate, bundle)
    return candidate, bundle


def require_week7_factorial_capabilities(
    manifest: dict[str, Any], *, multires_noise: bool
) -> None:
    """Fail closed if the selected owned runtime would make a factor inert."""

    bundle = (
        WEEK7_FACTORIAL_MULTIRES_BUNDLE
        if multires_noise
        else WEEK7_FACTORIAL_NO_MULTIRES_BUNDLE
    )
    require_capabilities(manifest, _BUNDLE_CAPABILITIES[bundle])


def load_capability_manifest(
    *,
    model_type: str = "krea2",
    bundle: str | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    resolved = requested_bundle(model_type, env) if bundle is None else bundle
    runtime_dir, path, identity_path = runtime_attestation_paths(
        model_type,
        resolved,
        environ=env,
    )
    try:
        manifest_bytes = _read_regular_attestation(path, "capability manifest")
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as exc:
        raise KreaRuntimeContractError(
            f"Krea capability manifest unavailable: {path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise KreaRuntimeContractError("Krea capability manifest must be an object")
    if set(manifest) != {
        "schema",
        "runtime_contract_id",
        "base_commit",
        "capabilities",
        "evidence",
    }:
        raise KreaRuntimeContractError("Krea capability manifest fields differ")
    if manifest.get("schema") != 1:
        raise KreaRuntimeContractError("unsupported Krea capability manifest schema")
    if manifest.get("runtime_contract_id") != RUNTIME_CONTRACT_ID:
        raise KreaRuntimeContractError("Krea runtime contract id mismatch")
    if manifest.get("base_commit") != PINNED_BASE_COMMIT:
        raise KreaRuntimeContractError("Krea runtime base commit mismatch")
    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, dict):
        raise KreaRuntimeContractError("Krea capability map is missing")
    if set(capabilities) != set(RUNTIME_MANIFEST_CAPABILITIES):
        raise KreaRuntimeContractError(
            "Krea runtime capability names differ from the contract"
        )
    evidence = manifest.get("evidence")
    if (
        not isinstance(evidence, dict)
        or set(evidence) != set(RUNTIME_MANIFEST_CAPABILITIES)
        or any(not isinstance(value, str) or not value for value in evidence.values())
    ):
        raise KreaRuntimeContractError("Krea capability evidence map is invalid")
    _load_runtime_identity(
        identity_path,
        capability_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    return manifest


def _load_runtime_identity(
    path: str,
    *,
    capability_manifest_sha256: str,
) -> dict[str, Any]:
    try:
        identity = json.loads(
            _read_regular_attestation(path, "runtime identity").decode("utf-8")
        )
    except Exception as exc:
        raise KreaRuntimeContractError(
            f"Krea runtime identity unavailable: {path}"
        ) from exc
    expected = {
        "schema": 1,
        "runtime_repository": OWNED_RUNTIME_REPOSITORY,
        "runtime_commit": OWNED_RUNTIME_COMMIT,
        "capability_manifest_sha256": capability_manifest_sha256,
    }
    if identity != expected:
        raise KreaRuntimeContractError("Krea runtime identity mismatch")
    return identity


def require_capabilities(manifest: dict[str, Any], required: tuple[str, ...]) -> None:
    capabilities = manifest.get("capabilities", {})
    missing = [
        name
        for name in required
        if capabilities.get(_RUNTIME_CAPABILITY_WIRE_ALIASES.get(name, name))
        is not True
    ]
    if missing:
        raise KreaRuntimeContractError(
            "Krea runtime lacks required capabilities: " + ", ".join(missing)
        )


def canonical_capabilities(manifest: dict[str, Any]) -> list[str]:
    """Expose semantic capability names, never legacy schema-v1 wire aliases."""

    capabilities = manifest.get("capabilities", {})
    return sorted(
        name
        for name in REQUIRED_CAPABILITIES
        if capabilities.get(_RUNTIME_CAPABILITY_WIRE_ALIASES.get(name, name)) is True
    )


def verify_selected_runtime(
    model_type: str,
    bundle: str,
    *,
    environ: dict[str, str] | None = None,
) -> str:
    """Fail before every launch unless the executable tree is the pinned tree."""

    with open_verified_runtime(
        model_type,
        bundle,
        environ=environ,
    ) as verified:
        return verified.runtime_dir


def open_verified_runtime(
    model_type: str,
    bundle: str,
    *,
    environ: dict[str, str] | None = None,
) -> VerifiedRuntime:
    """Verify and retain the exact directory descriptor used by the child."""

    env = os.environ if environ is None else environ
    runtime_dir = runtime_directory(model_type, bundle, environ=env)
    is_experimental_krea = (
        model_type or ""
    ).strip().lower() == "krea2" and bundle != INCUMBENT_BUNDLE
    manifest: dict[str, Any] | None = None
    if is_experimental_krea:
        manifest = load_capability_manifest(
            model_type=model_type,
            bundle=bundle,
            environ=env,
        )
        require_capabilities(manifest, _BUNDLE_CAPABILITIES[bundle])

    expected_commit = (
        OWNED_RUNTIME_COMMIT if is_experimental_krea else PINNED_BASE_COMMIT
    )
    expected_tree = OWNED_RUNTIME_TREE if is_experimental_krea else PINNED_BASE_TREE
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        root_fd = os.open(runtime_dir, flags)
    except OSError as exc:
        raise KreaRuntimeContractError(
            "selected runtime checkout is unavailable"
        ) from exc
    materialized_dir = ""
    materialized_fd = -1
    expected_identity: dict[str, Any] | None = None
    manifest_bytes = b""
    try:
        path_identity = os.stat(runtime_dir, follow_symlinks=False)
        if _stable_stat_identity(os.fstat(root_fd)) != _stable_stat_identity(
            path_identity
        ):
            raise KreaRuntimeContractError(
                "selected runtime path changed before verification"
            )
        if is_experimental_krea:
            assert manifest is not None
            manifest_stat = os.stat(
                CAPABILITY_MANIFEST_FILENAME,
                dir_fd=root_fd,
                follow_symlinks=False,
            )
            manifest_bytes = _read_fd_regular_bytes(
                root_fd,
                CAPABILITY_MANIFEST_FILENAME,
                manifest_stat,
                maximum_size=_MAX_ATTESTATION_BYTES,
            )
            try:
                descriptor_manifest = json.loads(manifest_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise KreaRuntimeContractError(
                    "descriptor-bound capability manifest is invalid"
                ) from exc
            if descriptor_manifest != manifest:
                raise KreaRuntimeContractError(
                    "capability manifest changed before runtime verification"
                )
            identity_stat = os.stat(
                RUNTIME_IDENTITY_FILENAME,
                dir_fd=root_fd,
                follow_symlinks=False,
            )
            identity_bytes = _read_fd_regular_bytes(
                root_fd,
                RUNTIME_IDENTITY_FILENAME,
                identity_stat,
                maximum_size=_MAX_ATTESTATION_BYTES,
            )
            try:
                descriptor_identity = json.loads(identity_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise KreaRuntimeContractError(
                    "descriptor-bound runtime identity is invalid"
                ) from exc
            expected_identity = {
                "schema": 1,
                "runtime_repository": OWNED_RUNTIME_REPOSITORY,
                "runtime_commit": OWNED_RUNTIME_COMMIT,
                "capability_manifest_sha256": hashlib.sha256(
                    manifest_bytes
                ).hexdigest(),
            }
            if descriptor_identity != expected_identity:
                raise KreaRuntimeContractError(
                    "descriptor-bound runtime identity mismatch"
                )
        _verify_runtime_tree_fd(
            root_fd,
            expected_commit=expected_commit,
            expected_tree_sha1=expected_tree,
        )
        if _stable_stat_identity(os.fstat(root_fd)) != _stable_stat_identity(
            os.stat(runtime_dir, follow_symlinks=False)
        ):
            raise KreaRuntimeContractError(
                "selected runtime path changed during verification"
            )
        materialized_dir = tempfile.mkdtemp(
            prefix="sn56-runtime-exec-",
            dir=os.path.realpath("/tmp"),
        )
        materialized_fd = os.open(materialized_dir, flags)
        _copy_runtime_tree_fd(root_fd, materialized_fd, root=True)
        allowed_root_extras = (
            frozenset({RUNTIME_IDENTITY_FILENAME})
            if is_experimental_krea
            else frozenset()
        )
        materialized_tree = _git_tree_digest(
            materialized_fd,
            root=True,
            allowed_root_extras=allowed_root_extras,
        ).hex()
        if materialized_tree != expected_tree:
            raise KreaRuntimeContractError(
                "materialized runtime executed tree mismatch"
            )
        if is_experimental_krea:
            assert manifest is not None and expected_identity is not None
            copied_manifest_stat = os.stat(
                CAPABILITY_MANIFEST_FILENAME,
                dir_fd=materialized_fd,
                follow_symlinks=False,
            )
            copied_manifest_bytes = _read_fd_regular_bytes(
                materialized_fd,
                CAPABILITY_MANIFEST_FILENAME,
                copied_manifest_stat,
                maximum_size=_MAX_ATTESTATION_BYTES,
            )
            copied_identity_stat = os.stat(
                RUNTIME_IDENTITY_FILENAME,
                dir_fd=materialized_fd,
                follow_symlinks=False,
            )
            copied_identity_bytes = _read_fd_regular_bytes(
                materialized_fd,
                RUNTIME_IDENTITY_FILENAME,
                copied_identity_stat,
                maximum_size=_MAX_ATTESTATION_BYTES,
            )
            if copied_manifest_bytes != manifest_bytes:
                raise KreaRuntimeContractError(
                    "materialized capability manifest mismatch"
                )
            try:
                copied_identity = json.loads(copied_identity_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise KreaRuntimeContractError(
                    "materialized runtime identity is invalid"
                ) from exc
            if copied_identity != expected_identity:
                raise KreaRuntimeContractError("materialized runtime identity mismatch")
        os.fchmod(materialized_fd, 0o500)
        os.close(root_fd)
        root_fd = -1
        return VerifiedRuntime(
            runtime_dir=runtime_dir,
            materialized_dir=materialized_dir,
            directory_fd=materialized_fd,
        )
    except Exception:
        if root_fd >= 0:
            os.close(root_fd)
        if materialized_fd >= 0:
            os.close(materialized_fd)
        _remove_materialized_runtime(materialized_dir)
        raise


def _git_sha1(data: bytes = b"") -> Any:
    """Return SHA-1 for Git object compatibility, including on FIPS hosts."""

    try:
        return hashlib.sha1(data, usedforsecurity=False)
    except TypeError:  # pragma: no cover - Python/OpenSSL compatibility
        return hashlib.sha1(data)


def _stable_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _copy_runtime_tree_fd(source_fd: int, destination_fd: int, *, root: bool) -> None:
    """Copy one verified-tree candidate without following mutable pathnames."""

    source_before = os.fstat(source_fd)
    if not stat.S_ISDIR(source_before.st_mode):
        raise KreaRuntimeContractError("selected runtime contains a non-directory")
    try:
        names = os.listdir(source_fd)
    except OSError as exc:
        raise KreaRuntimeContractError(
            "selected runtime directory cannot be enumerated"
        ) from exc

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    read_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    write_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for name in names:
        if root and name == ".git":
            continue
        encoded_name = os.fsencode(name)
        if not encoded_name or b"\x00" in encoded_name or b"/" in encoded_name:
            raise KreaRuntimeContractError("selected runtime contains an invalid path")
        try:
            expected = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        except OSError as exc:
            raise KreaRuntimeContractError(
                "selected runtime entry changed during materialization"
            ) from exc
        if stat.S_ISLNK(expected.st_mode):
            raise KreaRuntimeContractError("selected runtime contains a symlink")
        if stat.S_ISREG(expected.st_mode):
            input_fd = -1
            output_fd = -1
            try:
                input_fd = os.open(name, read_flags, dir_fd=source_fd)
                try:
                    output_fd = os.open(name, write_flags, 0o600, dir_fd=destination_fd)
                except Exception:
                    os.close(input_fd)
                    input_fd = -1
                    raise
            except OSError as exc:
                raise KreaRuntimeContractError(
                    "selected runtime file cannot be materialized safely"
                ) from exc
            try:
                before = os.fstat(input_fd)
                if _stable_stat_identity(before) != _stable_stat_identity(expected):
                    raise KreaRuntimeContractError(
                        "selected runtime file changed before materialization"
                    )
                copied = 0
                while True:
                    chunk = os.read(input_fd, 1024 * 1024)
                    if not chunk:
                        break
                    view = memoryview(chunk)
                    while view:
                        written = os.write(output_fd, view)
                        if written <= 0:  # pragma: no cover - OS contract
                            raise KreaRuntimeContractError(
                                "materialized runtime write made no progress"
                            )
                        view = view[written:]
                    copied += len(chunk)
                if copied != before.st_size or _stable_stat_identity(
                    os.fstat(input_fd)
                ) != _stable_stat_identity(before):
                    raise KreaRuntimeContractError(
                        "selected runtime file changed during materialization"
                    )
                os.fchmod(output_fd, 0o500 if expected.st_mode & 0o111 else 0o400)
            finally:
                if input_fd >= 0:
                    os.close(input_fd)
                if output_fd >= 0:
                    os.close(output_fd)
        elif stat.S_ISDIR(expected.st_mode):
            input_fd = -1
            output_fd = -1
            try:
                os.mkdir(name, 0o700, dir_fd=destination_fd)
                input_fd = os.open(name, directory_flags, dir_fd=source_fd)
                try:
                    output_fd = os.open(name, directory_flags, dir_fd=destination_fd)
                except Exception:
                    os.close(input_fd)
                    input_fd = -1
                    raise
            except OSError as exc:
                raise KreaRuntimeContractError(
                    "selected runtime directory cannot be materialized safely"
                ) from exc
            try:
                if _stable_stat_identity(os.fstat(input_fd)) != _stable_stat_identity(
                    expected
                ):
                    raise KreaRuntimeContractError(
                        "selected runtime directory changed before materialization"
                    )
                _copy_runtime_tree_fd(input_fd, output_fd, root=False)
                os.fchmod(output_fd, 0o500)
            finally:
                if input_fd >= 0:
                    os.close(input_fd)
                if output_fd >= 0:
                    os.close(output_fd)
        else:
            raise KreaRuntimeContractError(
                "selected runtime contains a non-regular entry"
            )

    if _stable_stat_identity(os.fstat(source_fd)) != _stable_stat_identity(
        source_before
    ):
        raise KreaRuntimeContractError(
            "selected runtime directory changed during materialization"
        )


def _remove_materialized_runtime(path: str) -> None:
    """Remove a private read-only runtime without following external symlinks."""

    if not path:
        return
    try:
        if os.path.islink(path):
            os.unlink(path)
            return
        for current, directories, files in os.walk(path, topdown=False):
            for name in files:
                candidate = os.path.join(current, name)
                if os.path.islink(candidate):
                    os.unlink(candidate)
                else:
                    os.chmod(candidate, 0o600, follow_symlinks=False)
            for name in directories:
                candidate = os.path.join(current, name)
                if os.path.islink(candidate):
                    os.unlink(candidate)
                else:
                    os.chmod(candidate, 0o700, follow_symlinks=False)
            os.chmod(current, 0o700, follow_symlinks=False)
        shutil.rmtree(path)
    except FileNotFoundError:
        return


def _git_blob_digest(directory_fd: int, name: str, expected: os.stat_result) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        file_fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise KreaRuntimeContractError(
            "selected runtime file cannot be opened safely"
        ) from exc
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or (
            _stable_stat_identity(before) != _stable_stat_identity(expected)
        ):
            raise KreaRuntimeContractError(
                "selected runtime file identity changed during verification"
            )
        digest = _git_sha1(f"blob {before.st_size}\0".encode("ascii"))
        observed = 0
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            digest.update(chunk)
        after = os.fstat(file_fd)
        if observed != before.st_size or (
            _stable_stat_identity(after) != _stable_stat_identity(before)
        ):
            raise KreaRuntimeContractError(
                "selected runtime file changed during verification"
            )
        return digest.digest()
    finally:
        os.close(file_fd)


def _git_tree_digest(
    directory_fd: int,
    *,
    root: bool,
    allowed_root_extras: frozenset[str] = frozenset(),
) -> bytes:
    before = os.fstat(directory_fd)
    if not stat.S_ISDIR(before.st_mode):
        raise KreaRuntimeContractError("selected runtime tree contains a non-directory")
    try:
        names = os.listdir(directory_fd)
    except OSError as exc:
        raise KreaRuntimeContractError(
            "selected runtime directory cannot be enumerated"
        ) from exc

    entries: list[tuple[bytes, bytes]] = []
    for name in names:
        if root and (name == ".git" or name in allowed_root_extras):
            continue
        encoded_name = os.fsencode(name)
        if not encoded_name or b"\x00" in encoded_name or b"/" in encoded_name:
            raise KreaRuntimeContractError("selected runtime contains an invalid path")
        try:
            entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise KreaRuntimeContractError(
                "selected runtime entry changed during verification"
            ) from exc
        if stat.S_ISLNK(entry_stat.st_mode):
            raise KreaRuntimeContractError("selected runtime contains a symlink")
        if stat.S_ISREG(entry_stat.st_mode):
            mode = b"100755" if entry_stat.st_mode & 0o111 else b"100644"
            object_digest = _git_blob_digest(directory_fd, name, entry_stat)
            sort_key = encoded_name + b"\x00"
        elif stat.S_ISDIR(entry_stat.st_mode):
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                child_fd = os.open(name, flags, dir_fd=directory_fd)
            except OSError as exc:
                raise KreaRuntimeContractError(
                    "selected runtime directory cannot be opened safely"
                ) from exc
            try:
                if _stable_stat_identity(os.fstat(child_fd)) != _stable_stat_identity(
                    entry_stat
                ):
                    raise KreaRuntimeContractError(
                        "selected runtime directory identity changed"
                    )
                object_digest = _git_tree_digest(child_fd, root=False)
            finally:
                os.close(child_fd)
            mode = b"40000"
            sort_key = encoded_name + b"/"
        else:
            raise KreaRuntimeContractError(
                "selected runtime contains a non-regular entry"
            )
        entries.append((sort_key, mode + b" " + encoded_name + b"\0" + object_digest))

    after = os.fstat(directory_fd)
    if _stable_stat_identity(after) != _stable_stat_identity(before):
        raise KreaRuntimeContractError(
            "selected runtime directory changed during verification"
        )
    body = b"".join(value for _key, value in sorted(entries))
    return _git_sha1(f"tree {len(body)}\0".encode("ascii") + body).digest()


def _verify_runtime_tree(
    runtime_dir: str,
    *,
    expected_commit: str,
    expected_tree_sha1: str,
) -> None:
    """Verify detached HEAD and executed bytes without invoking repository Git.

    Runtime containers need no ``git`` binary for this check.  Hashing the
    files that will execute also prevents repository-local Git configuration,
    attributes, filters, index flags, or aliases from shaping the verdict.
    """

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        root_fd = os.open(runtime_dir, flags)
    except OSError as exc:
        raise KreaRuntimeContractError(
            "selected runtime checkout is unavailable"
        ) from exc
    try:
        _verify_runtime_tree_fd(
            root_fd,
            expected_commit=expected_commit,
            expected_tree_sha1=expected_tree_sha1,
        )
    finally:
        os.close(root_fd)


def _verify_runtime_tree_fd(
    root_fd: int,
    *,
    expected_commit: str,
    expected_tree_sha1: str,
) -> None:
    """Verify a runtime through the same open directory used for execution."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        root_before = os.fstat(root_fd)
        if not stat.S_ISDIR(root_before.st_mode):
            raise KreaRuntimeContractError(
                "selected runtime checkout is not a directory"
            )
        try:
            git_fd = os.open(".git", flags, dir_fd=root_fd)
        except OSError as exc:
            raise KreaRuntimeContractError(
                "selected runtime has no detached Git identity"
            ) from exc
        try:
            head_stat = os.stat("HEAD", dir_fd=git_fd, follow_symlinks=False)
            head_bytes = _read_fd_regular_bytes(
                git_fd, "HEAD", head_stat, maximum_size=128
            )
        finally:
            os.close(git_fd)
        try:
            head = head_bytes.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise KreaRuntimeContractError(
                "selected runtime HEAD is not an exact commit"
            ) from exc
        if head != expected_commit or not re.fullmatch(r"[0-9a-f]{40}", head):
            raise KreaRuntimeContractError("selected runtime commit mismatch")
        allowed_root_extras = (
            frozenset({RUNTIME_IDENTITY_FILENAME})
            if expected_commit == OWNED_RUNTIME_COMMIT
            else frozenset()
        )
        tree_sha1 = _git_tree_digest(
            root_fd,
            root=True,
            allowed_root_extras=allowed_root_extras,
        ).hex()
        if tree_sha1 != expected_tree_sha1:
            raise KreaRuntimeContractError("selected runtime executed tree mismatch")
        if _stable_stat_identity(os.fstat(root_fd)) != _stable_stat_identity(
            root_before
        ):
            raise KreaRuntimeContractError(
                "selected runtime root changed during verification"
            )
    except OSError as exc:
        raise KreaRuntimeContractError(
            "selected runtime changed during verification"
        ) from exc


def _read_fd_regular_bytes(
    directory_fd: int,
    name: str,
    expected: os.stat_result,
    *,
    maximum_size: int,
) -> bytes:
    if expected.st_size > maximum_size:
        raise KreaRuntimeContractError("selected runtime identity is oversized")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        value_fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise KreaRuntimeContractError(
            "selected runtime identity cannot be opened safely"
        ) from exc
    try:
        before = os.fstat(value_fd)
        if not stat.S_ISREG(before.st_mode) or (
            _stable_stat_identity(before) != _stable_stat_identity(expected)
        ):
            raise KreaRuntimeContractError("selected runtime identity changed")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(value_fd, min(64 * 1024, maximum_size + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > maximum_size:
                raise KreaRuntimeContractError("selected runtime identity is oversized")
        if _stable_stat_identity(os.fstat(value_fd)) != _stable_stat_identity(before):
            raise KreaRuntimeContractError("selected runtime identity changed")
        return b"".join(chunks)
    finally:
        os.close(value_fd)


def _read_regular_attestation(path: str, label: str) -> bytes:
    """Read one bounded attestation through the shared path-safe reader."""

    try:
        return read_regular_bytes(
            path,
            label=f"Krea {label}",
            maximum_size=_MAX_ATTESTATION_BYTES,
        )
    except RegularFileError as exc:
        raise KreaRuntimeContractError(f"Krea {label} is not a regular file") from exc


def bundle_contract_document(bundle: str) -> dict[str, Any]:
    """Return the task-independent contract used to bind timing/evidence.

    Steps are intentionally excluded: exposure is governed by the independent
    budget policy. Everything that distinguishes the optimizer/runtime bundle
    is included, so a timing profile cannot be reused after a recipe change.
    """
    if bundle not in KNOWN_BUNDLES:
        raise KreaRuntimeContractError(f"unknown Krea bundle: {bundle!r}")
    return {
        "schema": 2,
        "runtime_contract_id": RUNTIME_CONTRACT_ID,
        "bundle": bundle,
        "claim": bundle_claim_document(bundle),
        "base_commit": PINNED_BASE_COMMIT,
        "runtime_repository": runtime_repository_for_bundle(bundle),
        "runtime_commit": runtime_commit_for_bundle(bundle),
        "required_capabilities": list(_BUNDLE_CAPABILITIES.get(bundle, ())),
        "runtime_manifest_capability_aliases": {
            name: _RUNTIME_CAPABILITY_WIRE_ALIASES[name]
            for name in _BUNDLE_CAPABILITIES.get(bundle, ())
            if name in _RUNTIME_CAPABILITY_WIRE_ALIASES
        },
        "normalized_config_projection": _reference_bundle_projection(bundle),
    }


def bundle_contract_sha256(bundle: str) -> str:
    return _canonical_sha256(bundle_contract_document(bundle))


def should_emit_effective_runtime_record(
    *,
    bundle: str,
    throughput_profile: Any = None,
    timing_probe: bool = False,
) -> bool:
    """Keep the default incumbent execution path free of new I/O/events."""

    return bool(
        bundle != INCUMBENT_BUNDLE or throughput_profile is not None or timing_probe
    )


def emit_effective_runtime_record(
    cfg: dict[str, Any],
    model_type: str,
    config_path: str,
    manifest: dict[str, Any] | None,
    *,
    throughput_profile=None,
    timing_probe: bool = False,
    source_run_id: str | None = None,
    current_dataset_size: int | None = None,
    current_accelerator_identity: str | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Atomically emit the exact effective training contract beside config YAML.

    The record is outside the validator upload root.  Its public telemetry event
    contains only identifiers and hashes; recipe details remain private.
    Emission is mandatory for experimental bundles and best-effort for the
    incumbent so a calibration run cannot proceed with an unrecorded runtime.
    """
    bundle = requested_bundle(model_type, environ)
    if not isinstance(source_run_id, str):
        raise KreaRuntimeContractError("effective runtime source run id is invalid")
    source_run_id = source_run_id.strip()
    task_identity, separator, attempt_nonce = source_run_id.rpartition(":")
    if (
        not separator
        or not task_identity
        or len(source_run_id) > 256
        or len(attempt_nonce) != 32
        or any(character not in "0123456789abcdef" for character in attempt_nonce)
    ):
        raise KreaRuntimeContractError("effective runtime source run id is invalid")
    expected_projection = bundle_contract_document(bundle)[
        "normalized_config_projection"
    ]
    if timing_contract_projection(cfg, bundle=bundle) != expected_projection:
        raise KreaRuntimeContractError(
            "effective config no longer matches the bundle timing contract"
        )
    if bundle != INCUMBENT_BUNDLE and throughput_profile is None and not timing_probe:
        raise KreaRuntimeContractError(
            "experimental Krea execution needs operator-attested timing "
            "or explicit probe mode"
        )
    timing: dict[str, Any]
    if throughput_profile is not None:
        from forge.adaptive_timing import ThroughputProfile, dataset_regime

        if not isinstance(throughput_profile, ThroughputProfile):
            raise KreaRuntimeContractError("invalid operator-attested timing profile")
        expected_digest = bundle_contract_sha256(bundle)
        expected_runtime_commit = runtime_commit_for_bundle(bundle)
        if (
            throughput_profile.bundle_id != bundle
            or throughput_profile.bundle_sha256 != expected_digest
            or throughput_profile.model_type != "krea2"
            or throughput_profile.runtime_commit != expected_runtime_commit
        ):
            raise KreaRuntimeContractError(
                "operator-attested timing profile binding drifted"
            )
        if (
            isinstance(current_dataset_size, bool)
            or not isinstance(current_dataset_size, int)
            or current_dataset_size <= 0
            or dataset_regime(current_dataset_size) != throughput_profile.dataset_regime
        ):
            raise KreaRuntimeContractError(
                "operator-attested timing profile dataset regime drifted"
            )
        timing = {
            "mode": "operator_attested_profile",
            "profile_sha256": throughput_profile.profile_sha256,
            "runtime_commit": throughput_profile.runtime_commit,
            "measured_dataset_size": throughput_profile.measured_dataset_size,
            "current_dataset_size": current_dataset_size,
            "dataset_regime": throughput_profile.dataset_regime,
            "accelerator_identity": throughput_profile.accelerator_identity,
        }
    elif timing_probe:
        from forge.adaptive_timing import dataset_regime

        if (
            isinstance(current_dataset_size, bool)
            or not isinstance(current_dataset_size, int)
            or current_dataset_size <= 0
            or not isinstance(current_accelerator_identity, str)
            or not current_accelerator_identity.strip()
            or len(current_accelerator_identity) > 256
        ):
            raise KreaRuntimeContractError(
                "bootstrap timing run identity is incomplete"
            )
        timing = {
            "mode": "bootstrap_probe_unmeasured",
            "profile_sha256": None,
            "runtime_commit": runtime_commit_for_bundle(bundle),
            "measured_dataset_size": None,
            "current_dataset_size": current_dataset_size,
            "dataset_regime": dataset_regime(current_dataset_size),
            "accelerator_identity": current_accelerator_identity.strip(),
        }
    else:
        timing = {
            "mode": "incumbent_static",
            "profile_sha256": None,
            "runtime_commit": None,
        }
    try:
        config_sha = _sha256_file(config_path)
        manifest_file_sha, manifest_semantic_sha = _capability_manifest_hashes(
            manifest,
            model_type=model_type,
            bundle=bundle,
            environ=environ,
        )
    except Exception as exc:
        if bundle != INCUMBENT_BUNDLE:
            raise KreaRuntimeContractError(
                f"effective config could not be hashed: {config_path}"
            ) from exc
        telemetry.event(
            "krea_effective_runtime_record_failed",
            bundle=bundle,
            error_type=type(exc).__name__,
        )
        return {"schema": 4, "bundle": bundle, "emission_failed": True}
    record: dict[str, Any] = {
        "schema": 4,
        "runtime_contract_id": RUNTIME_CONTRACT_ID,
        "source_run_id": source_run_id,
        "model_type": (model_type or "").strip().lower(),
        "runtime_repository": runtime_repository_for_bundle(bundle),
        "runtime_commit": runtime_commit_for_bundle(bundle),
        "bundle": bundle,
        "bundle_claim": bundle_claim_document(bundle),
        "bundle_contract_sha256": bundle_contract_sha256(bundle),
        "generated_config_sha256": config_sha,
        "capability_manifest_file_sha256": manifest_file_sha,
        "capability_manifest_semantic_sha256": manifest_semantic_sha,
        "capabilities": (
            canonical_capabilities(manifest) if manifest is not None else []
        ),
        "runtime_manifest_capability_aliases": {
            name: wire
            for name, wire in _RUNTIME_CAPABILITY_WIRE_ALIASES.items()
            if name in _BUNDLE_CAPABILITIES.get(bundle, ())
        },
        "timing": timing,
        "effective": _effective_fields(cfg, bundle=bundle),
        "lifecycle": "bootstrap",
        "first_checkpoint_observation": None,
        "training_completion_observation": None,
    }
    record_sha = _canonical_sha256(record)
    record["record_sha256"] = record_sha
    path = config_path + ".effective-runtime.json"
    try:
        _atomic_json(path, record)
    except Exception as exc:
        if bundle != INCUMBENT_BUNDLE:
            raise KreaRuntimeContractError(
                f"effective-runtime record could not be emitted: {path}"
            ) from exc
        telemetry.event(
            "krea_effective_runtime_record_failed",
            bundle=bundle,
            error_type=type(exc).__name__,
        )
        return record
    telemetry.event(
        "krea_effective_runtime_recorded",
        bundle=bundle,
        record_sha256=record_sha,
        generated_config_sha256=config_sha,
        capability_manifest_file_sha256=manifest_file_sha,
        capability_manifest_semantic_sha256=manifest_semantic_sha,
    )
    telemetry.set_meta(
        krea_bundle=bundle,
        krea_effective_runtime_record_sha256=record_sha,
    )
    return record


def _apply_source_derived_rank1(cfg: dict[str, Any]) -> None:
    p = _process(cfg)
    dataset = p["datasets"][0]
    dataset["cache_latents_to_disk"] = True
    dataset["caption_dropout_rate"] = 0.05
    p["network"] = {"type": "lora", "linear": 32, "linear_alpha": 32}
    p["save"]["save_every"] = 200
    # Retain every periodic checkpoint emitted at the source-derived 200-step
    # interval through our 2,000-step ceiling.  The owned cap of 12 is enough
    # for at most ten periodic checkpoints; it is not a promise of 12 saves.
    # This changes evidence retention only, not optimizer numerics or final LoRA.
    p["save"]["max_step_saves_to_keep"] = 12
    steps = p["train"]["steps"]
    # Replace rather than update: otherwise innocuous-looking template keys can
    # drift into the experiment and defeat strict unknown-field accounting.
    p["train"] = {
        "batch_size": 1,
        "cache_text_embeddings": False,
        "differential_guidance_scale": 12.0,
        "disable_sampling": True,
        "do_differential_guidance": True,
        "dtype": "bf16",
        "ema_config": {"use_ema": True, "ema_decay": 0.995},
        "gradient_accumulation": 1,
        "gradient_checkpointing": True,
        "loss_type": "mse",
        "lr": 1e-4,
        "lr_scheduler": "cosine_by_group",
        "lr_scheduler_params": {
            "min_lr_by_initial_lr": {"0.00000025": 0.0, "0.0001": 1e-5}
        },
        "multires_noise_discount": 0.3,
        "multires_noise_iterations": 6,
        "noise_offset": 0.0,
        "noise_scheduler": "flowmatch",
        "optimizer": "adamw8bit",
        "optimizer_params": {"weight_decay": 0.0001},
        "steps": steps,
        "text_encoder_lr": 2.5e-7,
        "timestep_type": "krea2_eval_sigmas",
        "train_text_encoder": True,
        "train_unet": True,
        "unet_lr": 1e-4,
        "sn56_strict_krea_fields": True,
    }


def _apply_mae(cfg: dict[str, Any]) -> None:
    p = _process(cfg)
    dataset = p["datasets"][0]
    dataset["cache_latents_to_disk"] = False
    dataset.pop("caption_dropout_rate", None)
    p["network"] = {"type": "lora", "linear": 32, "linear_alpha": 32}
    p["save"]["save_every"] = 200
    steps = p["train"]["steps"]
    p["train"] = {
        "batch_size": 1,
        "cache_text_embeddings": False,
        "content_or_style": "balanced",
        "differential_guidance_scale": 3.0,
        "disable_sampling": True,
        "do_differential_guidance": True,
        "dtype": "bf16",
        "ema_config": {"use_ema": False},
        "force_first_sample": False,
        "gradient_accumulation": 1,
        "gradient_checkpointing": True,
        "loss_type": "mae",
        "lr": 1e-4,
        "noise_scheduler": "flowmatch",
        "optimizer": "adamw8bit",
        "optimizer_params": {"weight_decay": 0.0001},
        "skip_first_sample": True,
        "steps": steps,
        "timestep_type": "linear",
        "train_text_encoder": False,
        "train_unet": True,
        "unload_text_encoder": False,
        "sn56_strict_krea_fields": True,
    }


def _apply_week7_factorial(cfg: dict[str, Any], *, bundle: str) -> None:
    """Apply only the dormant Week-7 factorial execution contract.

    This deliberately does not call either public-recipe overlay.  The bridge
    and A/B controls run the incumbent recipe bytes on the owned runtime, with
    the strict-field sentinel as the sole governance field.  C/D add exactly
    the two predeclared multires-noise fields on that same owned runtime.
    """

    if bundle not in {
        WEEK7_FACTORIAL_NO_MULTIRES_BUNDLE,
        WEEK7_FACTORIAL_MULTIRES_BUNDLE,
    }:
        raise KreaRuntimeContractError("invalid Week-7 factorial bundle")
    train = _process(cfg)["train"]
    train["sn56_strict_krea_fields"] = True
    if bundle == WEEK7_FACTORIAL_MULTIRES_BUNDLE:
        train["multires_noise_iterations"] = 6
        train["multires_noise_discount"] = 0.3
    else:
        train.pop("multires_noise_iterations", None)
        train.pop("multires_noise_discount", None)


def _validate_effective_bundle(cfg: dict[str, Any], bundle: str) -> None:
    p = _process(cfg)
    train = p["train"]
    if train.get("sn56_strict_krea_fields") is not True:
        raise KreaRuntimeContractError("strict Krea field validation is inactive")
    if bundle not in {
        WEEK7_FACTORIAL_NO_MULTIRES_BUNDLE,
        WEEK7_FACTORIAL_MULTIRES_BUNDLE,
    } and p.get("network") != {
        "type": "lora",
        "linear": 32,
        "linear_alpha": 32,
    }:
        raise KreaRuntimeContractError("Krea bundle network topology drifted")
    mismatched: list[str] = []
    if bundle in {LEADER_BUNDLE, LEADER_COMFY_TE_BUNDLE}:
        expected = {
            "train_text_encoder": True,
            "unet_lr": 1e-4,
            "text_encoder_lr": 2.5e-7,
            "timestep_type": "krea2_eval_sigmas",
            "differential_guidance_scale": 12.0,
            "multires_noise_iterations": 6,
            "multires_noise_discount": 0.3,
        }
    elif bundle == MAE_BUNDLE:
        expected = {
            "train_text_encoder": False,
            "loss_type": "mae",
            "timestep_type": "linear",
            "differential_guidance_scale": 3.0,
        }
    elif bundle == WEEK7_FACTORIAL_NO_MULTIRES_BUNDLE:
        expected = {
            "sn56_strict_krea_fields": True,
        }
        for forbidden in (
            "multires_noise_iterations",
            "multires_noise_discount",
            "text_encoder_lr",
        ):
            if forbidden in train:
                mismatched.append(forbidden)
                break
        if train.get("timestep_type") == "krea2_eval_sigmas":
            mismatched.append("timestep_type")
    elif bundle == WEEK7_FACTORIAL_MULTIRES_BUNDLE:
        expected = {
            "sn56_strict_krea_fields": True,
            "multires_noise_iterations": 6,
            "multires_noise_discount": 0.3,
        }
    else:  # pragma: no cover - guarded by bundle selection.
        raise KreaRuntimeContractError(f"unknown Krea bundle: {bundle!r}")
    mismatched.extend(key for key, value in expected.items() if train.get(key) != value)
    if bool(train.get("sn56_krea_comfy_text_encoder_export", False)) != (
        bundle == LEADER_COMFY_TE_BUNDLE
    ):
        mismatched.append("sn56_krea_comfy_text_encoder_export")
    if mismatched:
        raise KreaRuntimeContractError(
            "Krea bundle effective fields drifted: " + ", ".join(mismatched)
        )


def _process(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        process = cfg["config"]["process"]
        if not isinstance(process, list) or len(process) != 1:
            raise ValueError
        p = process[0]
        if p["model"]["arch"] != "krea2":
            raise ValueError
        return p
    except Exception as exc:
        raise KreaRuntimeContractError("invalid Krea config shape") from exc


def timing_contract_projection(cfg: dict[str, Any], *, bundle: str) -> dict[str, Any]:
    """Normalize one generated config for timing-profile compatibility.

    Only task identity, task-specific paths, trigger text, and the budgeted step
    count are abstracted. Every throughput- or optimizer-relevant value remains
    in the projection, including resolution, cache behavior, batch size,
    optimizer parameters, dtype, checkpointing, noise, guidance, EMA, scheduler,
    and all untouched template fields. Experimental bundle save cadence remains
    exact; incumbent cadence is explicitly normalized because it is derived
    from the independently budgeted step count.
    """

    if bundle not in KNOWN_BUNDLES:
        raise KreaRuntimeContractError(f"unknown Krea bundle: {bundle!r}")
    normalized = copy.deepcopy(cfg)
    try:
        normalized["config"]["name"] = "<task-identity>"
        p = normalized["config"]["process"][0]
        p["training_folder"] = "<task-path>"
        p["trigger_word"] = "<trigger>"
        p["datasets"][0]["folder_path"] = "<task-path>"
        p["model"]["name_or_path"] = "<model-path>"
        model_kwargs = p["model"].get("model_kwargs", {})
        for key in ("text_encoder_path", "vae_path"):
            if key in model_kwargs:
                model_kwargs[key] = "<model-path>"
        p["train"]["steps"] = "<budgeted-steps>"
        if bundle == INCUMBENT_BUNDLE:
            p["save"]["save_every"] = "<kill-safe-derived-from-steps>"
    except Exception as exc:
        raise KreaRuntimeContractError(
            "invalid Krea config for timing projection"
        ) from exc
    return normalized


def _reference_bundle_projection(bundle: str) -> dict[str, Any]:
    template_path = os.path.join(
        os.path.dirname(__file__), "templates", "base_diffusion_krea2.yaml"
    )
    try:
        with open(template_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        p = _process(cfg)
        cfg["config"]["name"] = "reference-task"
        p["training_folder"] = "/reference/checkpoints"
        p["trigger_word"] = "reference-trigger"
        p["datasets"][0]["folder_path"] = "/reference/dataset"
        p["model"]["name_or_path"] = "/reference/model"
        p["model"].setdefault("model_kwargs", {}).update(
            {
                "text_encoder_path": "/reference/qwen",
                "vae_path": "/reference/model",
            }
        )
        p["train"]["steps"] = 1234
        p["save"]["save_every"] = 247
        if bundle in {LEADER_BUNDLE, LEADER_COMFY_TE_BUNDLE}:
            _apply_source_derived_rank1(cfg)
            if bundle == LEADER_COMFY_TE_BUNDLE:
                p["train"]["sn56_krea_comfy_text_encoder_export"] = True
        elif bundle == MAE_BUNDLE:
            _apply_mae(cfg)
        elif bundle in {
            WEEK7_FACTORIAL_NO_MULTIRES_BUNDLE,
            WEEK7_FACTORIAL_MULTIRES_BUNDLE,
        }:
            _apply_week7_factorial(cfg, bundle=bundle)
        elif bundle != INCUMBENT_BUNDLE:
            raise KreaRuntimeContractError(f"unknown Krea bundle: {bundle!r}")
        return timing_contract_projection(cfg, bundle=bundle)
    except KreaRuntimeContractError:
        raise
    except Exception as exc:
        raise KreaRuntimeContractError(
            "could not construct Krea reference projection"
        ) from exc


def _effective_fields(cfg: dict[str, Any], *, bundle: str) -> dict[str, Any]:
    p = _process(cfg)
    return {
        "planned_steps": p["train"]["steps"],
        "normalized_config_projection": timing_contract_projection(cfg, bundle=bundle),
    }


def persist_first_checkpoint_observation(
    config_path: str, observation: Any
) -> dict[str, Any]:
    """Atomically bind the exactly-once timing observation into private evidence."""

    path = config_path + ".effective-runtime.json"
    try:
        record = json.loads(
            _read_regular_attestation(path, "effective runtime record").decode("utf-8")
        )
        if not isinstance(record, dict):
            raise ValueError("record is not an object")
        declared_sha = record.pop("record_sha256")
        if declared_sha != _canonical_sha256(record):
            raise ValueError("record digest mismatch")
        if record.get("generated_config_sha256") != _sha256_file(config_path):
            raise ValueError("generated config binding mismatch")
        if record.get("schema") != 4:
            raise ValueError("effective runtime record schema is unsupported")
        if (
            record.get("lifecycle") != "bootstrap"
            or record.get("first_checkpoint_observation") is not None
            or record.get("training_completion_observation") is not None
        ):
            raise ValueError("first checkpoint observation is out of order")
        fields = observation.telemetry_fields()
        record["first_checkpoint_observation"] = fields
        record["lifecycle"] = "first_checkpoint"
        record["record_sha256"] = _canonical_sha256(record)
        _atomic_json(path, record)
        return record
    except Exception as exc:
        raise KreaRuntimeContractError(
            f"first-checkpoint observation could not be persisted: {exc}"
        ) from exc


def persist_training_completion_observation(
    config_path: str,
    *,
    artifact_path: str,
    save_root: str,
    scope: dict[str, Any],
    training_elapsed_seconds: float,
    returncode: int | None,
    stopped_by_deadline: bool,
) -> dict[str, Any]:
    """Bind the subprocess terminal state into the raw timing record.

    Failed and deadline-stopped probes are recorded rather than disguised as
    completed timing claims. The checkpoint step, bytes, hash, and filesystem
    identity come from one opened current-run safetensors descriptor; log text
    is never completion evidence.
    Profile production accepts only a clean natural completion whose artifact
    records the runtime's completed-step count for the config-bound plan.
    """

    path = config_path + ".effective-runtime.json"
    try:
        record = json.loads(
            _read_regular_attestation(path, "effective runtime record").decode("utf-8")
        )
        if not isinstance(record, dict):
            raise ValueError("record is not an object")
        declared_sha = record.pop("record_sha256")
        if declared_sha != _canonical_sha256(record):
            raise ValueError("record digest mismatch")
        if record.get("generated_config_sha256") != _sha256_file(config_path):
            raise ValueError("generated config binding mismatch")
        if record.get("schema") != 4:
            raise ValueError("effective runtime record schema is unsupported")
        if (
            record.get("lifecycle") != "first_checkpoint"
            or record.get("first_checkpoint_observation") is None
            or record.get("training_completion_observation") is not None
        ):
            raise ValueError("training completion observation is out of order")
        planned_steps = record.get("effective", {}).get("planned_steps")
        if (
            isinstance(planned_steps, bool)
            or not isinstance(planned_steps, int)
            or planned_steps <= 0
        ):
            raise ValueError("planned steps are invalid")
        if (
            isinstance(training_elapsed_seconds, bool)
            or not isinstance(training_elapsed_seconds, (int, float))
            or not math.isfinite(float(training_elapsed_seconds))
            or float(training_elapsed_seconds) <= 0
        ):
            raise ValueError("training elapsed seconds is invalid")
        if returncode is not None and (
            isinstance(returncode, bool) or not isinstance(returncode, int)
        ):
            raise ValueError("return code is invalid")
        if not isinstance(stopped_by_deadline, bool):
            raise ValueError("deadline state is invalid")

        from forge.tasks import checkpoints
        from forge.tasks.integrity import inspect_training_artifact

        if not isinstance(scope, dict):
            raise ValueError("checkpoint scope is invalid")
        attempt_nonce = scope.get("attempt_nonce")
        if (
            not isinstance(attempt_nonce, str)
            or len(attempt_nonce) != 32
            or any(character not in "0123456789abcdef" for character in attempt_nonce)
            or not str(record.get("source_run_id") or "").endswith(f":{attempt_nonce}")
        ):
            raise ValueError("terminal artifact scope identity mismatch")
        artifact = inspect_training_artifact(artifact_path)
        if not checkpoints.descriptor_is_current_lora(
            save_root,
            artifact.path,
            scope,
            artifact.file_identity,
        ):
            raise ValueError("terminal artifact is not from the current run")
        artifact_name = os.path.basename(artifact.path)
        exact_final_name = f"{scope.get('repo')}.safetensors"
        # ai-toolkit increments ``step_num`` at the end of each loop iteration
        # and writes that completed-step count into final-save metadata.
        completed_steps = artifact.checkpoint_step
        natural_completion = bool(
            returncode == 0
            and not stopped_by_deadline
            and artifact_name == exact_final_name
            and completed_steps == planned_steps
        )
        record["training_completion_observation"] = {
            "training_elapsed_seconds": round(float(training_elapsed_seconds), 6),
            "returncode": returncode,
            "stopped_by_deadline": stopped_by_deadline,
            "natural_completion": natural_completion,
            "artifact_path": artifact.path,
            "artifact_name": artifact_name,
            "artifact_size_bytes": artifact.size_bytes,
            "artifact_sha256": artifact.sha256,
            "artifact_loadable": True,
            "artifact_checkpoint_step": artifact.checkpoint_step,
            "completed_steps": completed_steps,
            "scope_attempt_nonce": attempt_nonce,
            "artifact_file_identity": artifact.file_identity,
        }
        record["lifecycle"] = "terminal"
        record["record_sha256"] = _canonical_sha256(record)
        _atomic_json(path, record)
        return record
    except Exception as exc:
        raise KreaRuntimeContractError(
            f"training-completion observation could not be persisted: {exc}"
        ) from exc


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _capability_manifest_hashes(
    manifest: dict[str, Any] | None,
    *,
    model_type: str,
    bundle: str,
    environ: dict[str, str] | None = None,
) -> tuple[str | None, str | None]:
    """Return explicit file-byte and canonical-semantic manifest digests."""

    if manifest is None:
        return None, None
    _runtime_dir, path, _identity_path = runtime_attestation_paths(
        model_type,
        bundle,
        environ=environ,
    )
    raw = _read_regular_attestation(path, "capability manifest")
    if json.loads(raw.decode("utf-8")) != manifest:
        raise KreaRuntimeContractError(
            "capability manifest object differs from its recorded file bytes"
        )
    return hashlib.sha256(raw).hexdigest(), _canonical_sha256(manifest)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: str, value: dict[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".krea-runtime-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(_canonical_bytes(value))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
        dir_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
