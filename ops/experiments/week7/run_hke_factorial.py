#!/usr/bin/env python3
"""CPU authority for the Week-7 Krea HKE factor screen.

This module deliberately does not train, score, route, promote, or deploy. It
freezes the experiment contract before a rental, then consumes timing evidence
from the mechanical H100 gate to produce the plan that must be frozen before
quality training:

* materialize the four controlled recipes from one incumbent config; and
* apply the predeclared paired decision rule to exact-score rows.

The clock-fill arms accept internally validated, operator-attested
:class:`ThroughputProfile` objects.
There is no field-derived or hard-coded HKE seconds/step fallback.  A missing,
wrong-bundle, wrong-regime, or wrong-runtime profile is a hard prelaunch stop.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forge import adaptive_timing, krea_runtime, recipe  # noqa: E402

SCHEMA = 3
KIND = "sn56-week7-hke-factorial"
PLAN_STATUS = "TIMING_BOUND_DISCOVERY_PLAN_NO_LAUNCH_AUTHORITY"
MODEL_TYPE = "krea2"
INCUMBENT_BUNDLE = krea_runtime.INCUMBENT_BUNDLE
INCUMBENT_RUNTIME_COMMIT = krea_runtime.PINNED_BASE_COMMIT
INCUMBENT_BUNDLE_SHA256 = krea_runtime.bundle_contract_sha256(INCUMBENT_BUNDLE)
OWNED_NO_MULTIRES_BUNDLE = krea_runtime.WEEK7_FACTORIAL_NO_MULTIRES_BUNDLE
OWNED_MULTIRES_BUNDLE = krea_runtime.WEEK7_FACTORIAL_MULTIRES_BUNDLE
OWNED_NO_MULTIRES_BUNDLE_SHA256 = krea_runtime.bundle_contract_sha256(
    OWNED_NO_MULTIRES_BUNDLE
)
OWNED_MULTIRES_BUNDLE_SHA256 = krea_runtime.bundle_contract_sha256(
    OWNED_MULTIRES_BUNDLE
)
OWNED_RUNTIME_COMMIT = krea_runtime.OWNED_RUNTIME_COMMIT
KREA2_TEXT_ENCODER_PATH = "/cache/hf_cache/Qwen--Qwen3-VL-4B-Instruct"
INCUMBENT_TEMPLATE_PATH = (
    REPO_ROOT / "forge" / "templates" / "base_diffusion_krea2.yaml"
)
ADMISSION_AUTHORITY_PATH = (
    REPO_ROOT / "ops" / "experiments" / "week7" / "hke_fixture_admission.py"
)
# Literal trust anchor, reviewed at the exact source head.  Do not replace this
# with a value calculated at import time: that would let a changed template
# redefine the evidence expected to attest it.
INCUMBENT_TEMPLATE_FILE_SHA256 = (
    "031c4c8b2a6eda25b9db34eff1fdda20a05f04cf42a81b9e0c2515541cb1946b"
)
TRAINING_SEED_A = 42_565_431
TRAINING_SEED_B = 42_565_432
PRIMARY_FAMILIES = ("product", "logo_ui")
RELIABILITY_FAMILY = "social"
ARMS: dict[str, dict[str, Any]] = {
    "A": {
        "loss": "mae",
        "multires_noise": False,
        "bundle": OWNED_NO_MULTIRES_BUNDLE,
    },
    "B": {
        "loss": "mse",
        "multires_noise": False,
        "bundle": OWNED_NO_MULTIRES_BUNDLE,
    },
    "C": {
        "loss": "mae",
        "multires_noise": True,
        "bundle": OWNED_MULTIRES_BUNDLE,
    },
    "D": {
        "loss": "mse",
        "multires_noise": True,
        "bundle": OWNED_MULTIRES_BUNDLE,
    },
}
R0_STEPS = 1_166
FACTORIAL_STEPS = 1_200
MULTIRES_NOISE_ITERATIONS = 6
MULTIRES_NOISE_DISCOUNT = 0.3
FIXED_NVIDIA_SMI_QUERY = (
    "name",
    "uuid",
    "memory.total",
)
FIXED_NVIDIA_SMI_PATH = "/usr/bin/nvidia-smi"
FIXED_NVIDIA_SMI_COMMAND = (
    FIXED_NVIDIA_SMI_PATH,
    "--query-gpu=" + ",".join(FIXED_NVIDIA_SMI_QUERY),
    "--format=csv,noheader",
)
CAPTURED_H100_EVIDENCE_ORIGIN = "operator_attested_fixed_command_capture"
SUPPORTED_H100_NAMES = frozenset(
    {
        "NVIDIA H100 PCIe",
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H100 SXM",
    }
)
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260810
UNCERTAINTY_LEVEL = 0.95
MAX_RELATIVE_REGRESSION = 0.01
MIN_COMPOSITE_IMPROVEMENT = 0.03
BRIDGE_EQUIVALENCE_TOLERANCE = 0.005
C2_BORDERLINE_COMPOSITE_LOW = 0.03
C2_BORDERLINE_COMPOSITE_HIGH = 0.04
C2_BORDERLINE_CI_LOWER_ABS_MAX = 0.0025
EXPECTED_FIXTURE_COUNTS = {
    "social": {"discovery": 36, "confirmation": 36},
    "product": {"discovery": 18, "confirmation": 18},
    "logo_ui": {"discovery": 18, "confirmation": 18},
}
EXPECTED_PACKS = {
    "social": {
        "D1": {"phase": "discovery", "count": 18, "train": 10, "eval": 8},
        "D2": {"phase": "discovery", "count": 18, "train": 10, "eval": 8},
        "C1": {"phase": "confirmation", "count": 18, "train": 10, "eval": 8},
        "C2": {"phase": "confirmation", "count": 18, "train": 10, "eval": 8},
    },
    "product": {
        "D1": {"phase": "discovery", "count": 18, "train": 10, "eval": 8},
        "C1": {"phase": "confirmation", "count": 18, "train": 10, "eval": 8},
    },
    "logo_ui": {
        "D1": {"phase": "discovery", "count": 18, "train": 10, "eval": 8},
        "C1": {"phase": "confirmation", "count": 18, "train": 10, "eval": 8},
    },
}


class HKEContractError(RuntimeError):
    """A prelaunch or decision input is missing, malformed, or misbound."""


@dataclass(frozen=True)
class BoundTimingProfile:
    """One validated timing claim plus its experiment-specific binding.

    ``ThroughputProfile`` binds a runtime bundle and physical accelerator
    identity, but it does not by itself say which factorial loss/config
    produced the timing.
    This outer record closes that gap and is itself content addressed.
    """

    profile: adaptive_timing.ThroughputProfile
    loss: str
    measured_dataset_size: int
    measured_config_sha256: str
    measured_config: Mapping[str, Any]
    source_record: Mapping[str, Any]
    accelerator_observation: Mapping[str, Any]
    binding_sha256: str

    @property
    def accelerator_identity(self) -> str:
        """Canonical physical-GPU identity from the validated observation."""

        row = self.accelerator_observation["device"]
        return adaptive_timing.accelerator_identity(
            name=row["name"],
            memory_total_mib=row["memory_total_mib"],
            uuid=row["uuid"],
        )


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            separators=(",", ": "),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _runtime_record_bytes(value: Any) -> bytes:
    """Reproduce the compact newline-terminated runtime-sidecar byte domain."""

    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _generated_config_bytes(value: Mapping[str, Any]) -> bytes:
    """Reproduce ``forge.config.write_config`` for an exact config-file bind."""

    return yaml.safe_dump(dict(value), sort_keys=False).encode("utf-8")


def fixture_semantic_sha256(value: Any) -> str:
    """Digest in the admission authority's indented canonical JSON domain."""

    return canonical_sha256(value)


def renderer_semantic_sha256(value: Any) -> str:
    """Digest in the renderer's compact canonical JSON domain."""

    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _literal_revision_identity(
    revision: str, relative_paths: Sequence[str]
) -> dict[str, Any]:
    """Resolve one literal commit/tree and hash exact blobs without filters.

    ``git show HEAD:path`` leaves the object lookup implicit.  A hostile
    ``refs/replace`` entry can then change the object bytes that Git returns.
    Resolve the commit and blob object IDs under ``--no-replace-objects`` and
    read that literal blob with ``cat-file``; clean filters and attributes are
    not consulted by this object-level path.
    """

    paths = [str(item) for item in relative_paths]
    for relative in paths:
        parsed = Path(relative)
        if (
            not relative
            or parsed.is_absolute()
            or "\\" in relative
            or ":" in relative
            or any(part in {"", ".", ".."} for part in parsed.parts)
        ):
            raise HKEContractError(f"committed source path is invalid: {relative}")
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent-sn56-factorial",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }
    command = [
        "/usr/bin/git",
        "--no-replace-objects",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
    ]

    def run(arguments: Sequence[str]) -> bytes:
        completed = subprocess.run(
            [*command, *arguments],
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise HKEContractError(f"committed revision unavailable: {revision}")
        return completed.stdout

    commit = (
        run(("rev-parse", "--verify", f"{revision}^{{commit}}")).decode("ascii").strip()
    )
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise HKEContractError("committed revision is not an exact object ID")
    tree = run(("rev-parse", "--verify", f"{commit}^{{tree}}"))
    tree_sha = tree.decode("ascii").strip()
    if re.fullmatch(r"[0-9a-f]{40}", tree_sha) is None:
        raise HKEContractError("committed tree is not an exact object ID")
    blobs: dict[str, str] = {}
    for relative in paths:
        blob = (
            run(("rev-parse", "--verify", f"{commit}:{relative}"))
            .decode("ascii")
            .strip()
        )
        if re.fullmatch(r"[0-9a-f]{40}", blob) is None:
            raise HKEContractError("committed source blob is not an exact object ID")
        if run(("cat-file", "-t", blob)) != b"blob\n":
            raise HKEContractError("committed source object is not a blob")
        blobs[relative] = hashlib.sha256(run(("cat-file", "blob", blob))).hexdigest()
    return {"commit": commit, "tree": tree_sha, "blob_sha256": blobs}


def _head_blob_sha256(relative_path: str) -> str:
    identity = _literal_revision_identity("HEAD", (relative_path,))
    return identity["blob_sha256"][str(relative_path)]


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise HKEContractError(f"{label} is not a sha256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise HKEContractError(f"{label} is not a sha256") from exc
    return value


def _validate_utc_timestamp(value: Any, label: str) -> str:
    """Require one canonical, calendar-valid whole-second UTC timestamp."""

    if not isinstance(value, str):
        raise HKEContractError(f"{label} timestamp is invalid")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise HKEContractError(f"{label} timestamp is invalid") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise HKEContractError(f"{label} timestamp is invalid")
    return value


def _validate_h100_observation(
    value: Any, label: str, *, require_fixed_capture: bool = False
) -> dict[str, Any]:
    """Validate one embedded, mechanically parseable ``nvidia-smi`` record.

    A device-name substring is not hardware evidence.  The exact query, raw CSV
    bytes, UUID and reported memory are all carried forward and content-bound.
    """

    fields = {
        "schema",
        "kind",
        "query",
        "exit_code",
        "captured_at_utc",
        "evidence_origin",
        "raw_stdout",
        "raw_stdout_sha256",
        "device",
        "observation_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise HKEContractError(f"{label} H100 observation is malformed")
    body = dict(value)
    declared = body.pop("observation_sha256")
    _require_sha256(declared, f"{label} H100 observation")
    if declared != canonical_sha256(body):
        raise HKEContractError(f"{label} H100 observation digest mismatch")
    if (
        value["schema"] != 1
        or value["kind"] != "sn56-nvidia-smi-h100-observation"
        or value["query"] != list(FIXED_NVIDIA_SMI_QUERY)
    ):
        raise HKEContractError(f"{label} H100 query is not the fixed query")
    if value["exit_code"] != 0:
        raise HKEContractError(f"{label} nvidia-smi command did not succeed")
    captured_at = _validate_utc_timestamp(value["captured_at_utc"], f"{label} capture")
    if value["evidence_origin"] not in {
        "synthetic",
        CAPTURED_H100_EVIDENCE_ORIGIN,
    }:
        raise HKEContractError(f"{label} evidence origin is invalid")
    if (
        require_fixed_capture
        and value["evidence_origin"] != CAPTURED_H100_EVIDENCE_ORIGIN
    ):
        raise HKEContractError(
            f"{label} launch evidence must come from the fixed command capture"
        )
    raw = value["raw_stdout"]
    if not isinstance(raw, str) or "\n" in raw.rstrip("\n\r"):
        raise HKEContractError(f"{label} H100 raw output is not one CSV row")
    if value["raw_stdout_sha256"] != hashlib.sha256(raw.encode()).hexdigest():
        raise HKEContractError(f"{label} H100 raw-output digest mismatch")
    device = value["device"]
    if not isinstance(device, Mapping) or set(device) != {
        "name",
        "uuid",
        "memory_total_mib",
    }:
        raise HKEContractError(f"{label} H100 device record is malformed")
    name = device["name"]
    uuid = device["uuid"]
    memory = device["memory_total_mib"]
    if name not in SUPPORTED_H100_NAMES:
        raise HKEContractError(f"{label} is not an allowlisted H100 80GB device")
    if (
        not isinstance(uuid, str)
        or re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            uuid,
        )
        is None
    ):
        raise HKEContractError(f"{label} H100 UUID is malformed")
    if (
        isinstance(memory, bool)
        or not isinstance(memory, int)
        or not 78_000 <= memory <= 85_000
    ):
        raise HKEContractError(f"{label} is not an H100 80GB memory observation")
    expected_raw = f"{name}, {uuid}, {memory} MiB\n"
    if raw != expected_raw:
        raise HKEContractError(f"{label} H100 raw output does not parse exactly")
    return {**body, "observation_sha256": declared}


def h100_observation(
    *,
    name: str,
    uuid: str,
    memory_total_mib: int,
    captured_at_utc: str = "2026-08-11T00:00:00Z",
) -> dict[str, Any]:
    """Construct synthetic H100-shaped evidence for CPU-only tests.

    This constructor cannot claim that ``nvidia-smi`` ran.  Launchable records
    can only be produced by :func:`capture_h100_observation`.
    """

    raw = f"{name}, {uuid}, {memory_total_mib} MiB\n"
    body = {
        "schema": 1,
        "kind": "sn56-nvidia-smi-h100-observation",
        "query": list(FIXED_NVIDIA_SMI_QUERY),
        "exit_code": 0,
        "captured_at_utc": captured_at_utc,
        "evidence_origin": "synthetic",
        "raw_stdout": raw,
        "raw_stdout_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "device": {
            "name": name,
            "uuid": uuid,
            "memory_total_mib": memory_total_mib,
        },
    }
    value = {**body, "observation_sha256": canonical_sha256(body)}
    return _validate_h100_observation(value, "constructed")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def capture_h100_observation() -> dict[str, Any]:
    """Run the fixed absolute ``nvidia-smi`` command and bind its exact row."""

    completed = subprocess.run(
        list(FIXED_NVIDIA_SMI_COMMAND),
        cwd="/",
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": "/nonexistent-sn56-h100-capture",
            "LANG": "C",
            "LC_ALL": "C",
        },
        check=False,
        capture_output=True,
        text=True,
    )
    raw = completed.stdout
    match = re.fullmatch(r"([^,\r\n]+), (GPU-[0-9a-fA-F-]+), ([0-9]+) MiB\n", raw)
    if match is None:
        raise HKEContractError("fixed nvidia-smi command returned an invalid row")
    name, uuid, memory_text = match.groups()
    body = {
        "schema": 1,
        "kind": "sn56-nvidia-smi-h100-observation",
        "query": list(FIXED_NVIDIA_SMI_QUERY),
        "exit_code": completed.returncode,
        "captured_at_utc": _utc_now(),
        "evidence_origin": CAPTURED_H100_EVIDENCE_ORIGIN,
        "raw_stdout": raw,
        "raw_stdout_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "device": {
            "name": name,
            "uuid": uuid,
            "memory_total_mib": int(memory_text),
        },
    }
    value = {**body, "observation_sha256": canonical_sha256(body)}
    return _validate_h100_observation(value, "captured", require_fixed_capture=True)


def _validate_inventory_body(
    value: Any, label: str, *, expected_pairs: int | None = None
) -> dict[str, Any]:
    """Validate a canonical, flat, exact image/caption directory inventory."""

    fields = {"files", "file_count", "semantic_sha256"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise HKEContractError(f"{label} training inventory is malformed")
    body = dict(value)
    declared = body.pop("semantic_sha256")
    _require_sha256(declared, f"{label} training inventory")
    if declared != fixture_semantic_sha256(body):
        raise HKEContractError(f"{label} training inventory digest mismatch")
    files = value["files"]
    if (
        not isinstance(files, list)
        or not files
        or isinstance(value["file_count"], bool)
        or value["file_count"] != len(files)
    ):
        raise HKEContractError(f"{label} training inventory is empty")
    expected_order: list[str] = []
    images_by_stem: dict[str, int] = {}
    captions_by_stem: dict[str, int] = {}
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {"path", "bytes", "sha256"}:
            raise HKEContractError(f"{label} training inventory entry is malformed")
        path = item["path"]
        if (
            not isinstance(path, str)
            or not path
            or Path(path).name != path
            or path in {".", ".."}
        ):
            raise HKEContractError(f"{label} training inventory path is indirect")
        if (
            isinstance(item["bytes"], bool)
            or not isinstance(item["bytes"], int)
            or item["bytes"] <= 0
        ):
            raise HKEContractError(f"{label} training inventory size is invalid")
        _require_sha256(item["sha256"], f"{label} training inventory file")
        suffix = Path(path).suffix.casefold()
        stem = Path(path).stem
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            images_by_stem[stem] = images_by_stem.get(stem, 0) + 1
        elif suffix == ".txt":
            captions_by_stem[stem] = captions_by_stem.get(stem, 0) + 1
        else:
            raise HKEContractError(f"{label} training inventory has an unexpected file")
        expected_order.append(path)
    if expected_order != sorted(expected_order) or len(set(expected_order)) != len(
        expected_order
    ):
        raise HKEContractError(f"{label} training inventory is not unique and ordered")
    stems = set(images_by_stem) | set(captions_by_stem)
    if (
        not stems
        or any(images_by_stem.get(stem) != 1 for stem in stems)
        or any(captions_by_stem.get(stem) != 1 for stem in stems)
        or (expected_pairs is not None and len(stems) != expected_pairs)
        or len(files) != 2 * len(stems)
    ):
        raise HKEContractError(
            f"{label} inventory must have exactly one image and caption per stem"
        )
    return {**body, "semantic_sha256": declared}


def inventory_training_directory(path: Path) -> dict[str, Any]:
    """Hash direct children through no-follow descriptors under one directory FD."""

    root = Path(path)
    root_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        root_fd = os.open(root, root_flags)
    except OSError as exc:
        raise HKEContractError("training directory is unavailable") from exc
    try:
        root_before = os.fstat(root_fd)
        if not stat.S_ISDIR(root_before.st_mode):
            raise HKEContractError("training directory is not a direct real directory")
        try:
            names = sorted(os.listdir(root_fd))
        except OSError as exc:
            raise HKEContractError(
                "training directory changed during inventory"
            ) from exc
        files: list[dict[str, Any]] = []
        file_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        for name in names:
            if not name or Path(name).name != name or name in {".", ".."}:
                raise HKEContractError("training directory entry name is invalid")
            try:
                descriptor = os.open(name, file_flags, dir_fd=root_fd)
            except OSError as exc:
                raise HKEContractError(
                    "training directory contains a symlink or unreadable entry"
                ) from exc
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise HKEContractError(
                        "training directory contains a non-regular entry"
                    )
                digest = hashlib.sha256()
                total = 0
                while True:
                    block = os.read(descriptor, 1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    total += len(block)
                after = os.fstat(descriptor)
                try:
                    linked_after = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                except OSError as exc:
                    raise HKEContractError(
                        "training directory file changed during inventory"
                    ) from exc
                identity_before = (
                    before.st_dev,
                    before.st_ino,
                    before.st_mode,
                    before.st_size,
                )
                identity_after = (
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_size,
                )
                linked_identity = (
                    linked_after.st_dev,
                    linked_after.st_ino,
                    linked_after.st_mode,
                    linked_after.st_size,
                )
                if (
                    identity_before != identity_after
                    or identity_after != linked_identity
                    or total != before.st_size
                ):
                    raise HKEContractError(
                        "training directory file changed during inventory"
                    )
                files.append(
                    {
                        "path": name,
                        "bytes": before.st_size,
                        "sha256": digest.hexdigest(),
                    }
                )
            finally:
                os.close(descriptor)
        root_after = os.fstat(root_fd)
        try:
            path_after = os.lstat(root)
            final_names = sorted(os.listdir(root_fd))
        except OSError as exc:
            raise HKEContractError(
                "training directory changed during inventory"
            ) from exc
        if (
            (root_before.st_dev, root_before.st_ino)
            != (root_after.st_dev, root_after.st_ino)
            or (root_before.st_dev, root_before.st_ino)
            != (path_after.st_dev, path_after.st_ino)
            or names != final_names
        ):
            raise HKEContractError("training directory changed during inventory")
    finally:
        os.close(root_fd)
    body = {"files": files, "file_count": len(files)}
    return _validate_inventory_body(
        {**body, "semantic_sha256": fixture_semantic_sha256(body)}, "generated"
    )


def _validate_execution_identity(value: Any, label: str) -> dict[str, str]:
    fields = {"code_tree", "runtime_tree", "container_digest", "python_executable"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise HKEContractError(f"{label} execution identity is malformed")
    result = dict(value)
    for key in ("code_tree", "runtime_tree"):
        item = result[key]
        if not isinstance(item, str) or len(item) != 40:
            raise HKEContractError(f"{label} {key} is not a Git tree")
        try:
            int(item, 16)
        except ValueError as exc:
            raise HKEContractError(f"{label} {key} is not a Git tree") from exc
    digest = result["container_digest"]
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise HKEContractError(f"{label} container digest is not immutable")
    _require_sha256(digest.removeprefix("sha256:"), f"{label} container digest")
    executable = result["python_executable"]
    if (
        not isinstance(executable, str)
        or not os.path.isabs(executable)
        or os.path.normpath(executable) != executable
    ):
        raise HKEContractError(f"{label} Python executable is not absolute")
    return result


def _cell_identity_body(
    *,
    family: str,
    pack: str,
    phase: str,
    arm: str,
    seed: int,
    candidate_semantic_sha256: str,
    fixture_admission_sha256: str,
    evaluation_row_identity_sha256: str,
    generated_config_sha256: str,
    execution_identity: Mapping[str, str],
    training_inventory_sha256: str,
    evaluation_inventory_sha256: str,
    bundle_id: str,
    bundle_sha256: str,
    runtime_commit: str,
    accelerator_observation_sha256: str,
    accelerator_uuid: str,
) -> dict[str, Any]:
    checked_execution = _validate_execution_identity(execution_identity, "cell")
    for value, label in (
        (candidate_semantic_sha256, "cell candidate"),
        (fixture_admission_sha256, "cell admission"),
        (evaluation_row_identity_sha256, "cell evaluation rows"),
        (generated_config_sha256, "cell generated config"),
        (training_inventory_sha256, "cell training inventory"),
        (evaluation_inventory_sha256, "cell evaluation inventory"),
        (bundle_sha256, "cell runtime bundle"),
        (accelerator_observation_sha256, "cell accelerator observation"),
    ):
        _require_sha256(value, label)
    if family not in EXPECTED_PACKS or pack not in EXPECTED_PACKS[family]:
        raise HKEContractError("cell family/pack is outside the frozen design")
    if phase not in {"bridge", "discovery", "replication", "confirmation", "guardrail"}:
        raise HKEContractError("cell phase is outside the staged protocol")
    if not isinstance(arm, str) or not arm:
        raise HKEContractError("cell arm is outside the staged protocol")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise HKEContractError("cell seed is invalid")
    if not isinstance(accelerator_uuid, str) or not accelerator_uuid.startswith("GPU-"):
        raise HKEContractError("cell accelerator UUID is invalid")
    if bundle_id not in {
        INCUMBENT_BUNDLE,
        OWNED_NO_MULTIRES_BUNDLE,
        OWNED_MULTIRES_BUNDLE,
    }:
        raise HKEContractError("cell runtime bundle is not allowed")
    if bundle_sha256 != krea_runtime.bundle_contract_sha256(bundle_id):
        raise HKEContractError("cell runtime bundle digest mismatch")
    if runtime_commit != krea_runtime.runtime_commit_for_bundle(bundle_id):
        raise HKEContractError("cell runtime commit mismatch")
    return {
        "schema": 1,
        "kind": "sn56-week7-hke-cell-identity",
        "family": family,
        "pack": pack,
        "phase": phase,
        "arm": arm,
        "seed": seed,
        "candidate_semantic_sha256": candidate_semantic_sha256,
        "fixture_admission_sha256": fixture_admission_sha256,
        "evaluation_row_identity_sha256": evaluation_row_identity_sha256,
        "generated_config_sha256": generated_config_sha256,
        "code_tree": checked_execution["code_tree"],
        "runtime_tree": checked_execution["runtime_tree"],
        "container_digest": checked_execution["container_digest"],
        "python_executable": checked_execution["python_executable"],
        "training_inventory_sha256": training_inventory_sha256,
        "evaluation_inventory_sha256": evaluation_inventory_sha256,
        "bundle_id": bundle_id,
        "bundle_sha256": bundle_sha256,
        "runtime_commit": runtime_commit,
        "accelerator_observation_sha256": accelerator_observation_sha256,
        "accelerator_uuid": accelerator_uuid,
    }


def _build_cell_identity(**kwargs: Any) -> dict[str, Any]:
    body = _cell_identity_body(**kwargs)
    return {**body, "cell_sha256": canonical_sha256(body)}


def _validate_cell_identity(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HKEContractError(f"{label} cell identity is unavailable")
    document = dict(value)
    declared = document.pop("cell_sha256", None)
    expected = _cell_identity_body(
        family=document.get("family"),
        pack=document.get("pack"),
        phase=document.get("phase"),
        arm=document.get("arm"),
        seed=document.get("seed"),
        candidate_semantic_sha256=document.get("candidate_semantic_sha256"),
        fixture_admission_sha256=document.get("fixture_admission_sha256"),
        evaluation_row_identity_sha256=document.get("evaluation_row_identity_sha256"),
        generated_config_sha256=document.get("generated_config_sha256"),
        execution_identity={
            "code_tree": document.get("code_tree"),
            "runtime_tree": document.get("runtime_tree"),
            "container_digest": document.get("container_digest"),
            "python_executable": document.get("python_executable"),
        },
        training_inventory_sha256=document.get("training_inventory_sha256"),
        evaluation_inventory_sha256=document.get("evaluation_inventory_sha256"),
        bundle_id=document.get("bundle_id"),
        bundle_sha256=document.get("bundle_sha256"),
        runtime_commit=document.get("runtime_commit"),
        accelerator_observation_sha256=document.get("accelerator_observation_sha256"),
        accelerator_uuid=document.get("accelerator_uuid"),
    )
    if document != expected or declared != canonical_sha256(expected):
        raise HKEContractError(f"{label} cell identity digest mismatch")
    return {**expected, "cell_sha256": declared}


CELL_BINDING_FIELDS = frozenset(
    {
        "family",
        "pack",
        "phase",
        "arm",
        "seed",
        "candidate_semantic_sha256",
        "fixture_admission_sha256",
        "evaluation_row_identity_sha256",
        "generated_config_sha256",
        "code_tree",
        "runtime_tree",
        "container_digest",
        "python_executable",
        "training_inventory_sha256",
        "evaluation_inventory_sha256",
        "bundle_id",
        "bundle_sha256",
        "runtime_commit",
        "cell_sha256",
        "accelerator_observation_sha256",
        "accelerator_uuid",
    }
)


def _receipt_cell_binding(identity: Mapping[str, Any]) -> dict[str, Any]:
    checked = _validate_cell_identity(identity, "plan")
    return {key: checked[key] for key in CELL_BINDING_FIELDS}


def _validate_effective_runtime_record(
    value: Any,
    *,
    expected_run_id: str,
    expected_bundle: str,
    expected_config: Mapping[str, Any],
    expected_config_file_sha256: str,
    expected_source_file_sha256: str | None = None,
    label: str,
) -> dict[str, Any]:
    """Bind a real Forge runtime sidecar to the exact planned config/bundle."""

    fields = {
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
    if not isinstance(value, Mapping) or set(value) != fields:
        raise HKEContractError(f"{label} effective-runtime record is malformed")
    record = copy.deepcopy(dict(value))
    body = dict(record)
    declared = body.pop("record_sha256")
    _require_sha256(declared, f"{label} effective-runtime record")
    if declared != hashlib.sha256(_runtime_record_bytes(body)).hexdigest():
        raise HKEContractError(f"{label} effective-runtime record digest mismatch")
    source_file_sha = hashlib.sha256(_runtime_record_bytes(record)).hexdigest()
    if (
        expected_source_file_sha256 is not None
        and source_file_sha != expected_source_file_sha256
    ):
        raise HKEContractError(f"{label} effective-runtime file digest mismatch")
    _require_sha256(expected_config_file_sha256, f"{label} generated config file")
    if hashlib.sha256(_generated_config_bytes(expected_config)).hexdigest() != (
        expected_config_file_sha256
    ):
        raise HKEContractError(f"{label} generated config file digest is false")
    expected_projection = krea_runtime.timing_contract_projection(
        copy.deepcopy(dict(expected_config)), bundle=expected_bundle
    )
    effective = record.get("effective")
    if not isinstance(effective, Mapping) or set(effective) != {
        "planned_steps",
        "normalized_config_projection",
    }:
        raise HKEContractError(f"{label} effective runtime fields are malformed")
    first = record.get("first_checkpoint_observation")
    completion = record.get("training_completion_observation")
    bootstrap_first_fields = {
        "bundle_id",
        "timing_profile_sha256",
        "observation_mode",
        "checkpoint_step",
        "elapsed_since_launch_s",
        "active_planned_steps",
        "active_plan_mutable",
        "active_plan_action",
    }
    profiled_first_fields = {
        "bundle_id",
        "timing_profile_sha256",
        "checkpoint_step",
        "elapsed_since_launch_s",
        "profiled_seconds_per_step",
        "observed_seconds_per_step",
        "observed_to_profile_ratio",
        "correction",
        "active_planned_steps",
        "active_plan_mutable",
        "active_plan_action",
        "active_plan_exceeds_observed_budget",
        "future_budget_cap_steps",
        "future_target_steps",
        "future_recommended_steps",
        "future_step_delta",
    }
    completion_fields = {
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
        "artifact_file_identity",
    }
    timing = record.get("timing")
    if not isinstance(timing, Mapping):
        raise HKEContractError(f"{label} runtime timing identity is malformed")
    timing_mode = timing.get("mode")
    expected_timing_fields = (
        {"mode", "profile_sha256", "runtime_commit"}
        if timing_mode == "incumbent_static"
        else {
            "mode",
            "profile_sha256",
            "runtime_commit",
            "measured_dataset_size",
            "current_dataset_size",
            "dataset_regime",
            "accelerator_identity",
        }
    )
    expected_first_fields = (
        profiled_first_fields
        if timing_mode == "operator_attested_profile"
        else bootstrap_first_fields
    )
    if (
        set(timing) != expected_timing_fields
        or not isinstance(first, Mapping)
        or set(first) != expected_first_fields
        or not isinstance(completion, Mapping)
        or set(completion) != completion_fields
        or not isinstance(completion.get("artifact_file_identity"), Mapping)
        or set(completion["artifact_file_identity"])
        != {"device", "inode", "size", "mtime_ns", "ctime_ns"}
    ):
        raise HKEContractError(f"{label} runtime observations are malformed")
    planned_steps = _train_node(expected_config)["steps"]
    expected_first_checkpoint_step = min(
        planned_steps, int(_save_node(expected_config)["save_every"])
    )
    contract = krea_runtime.bundle_contract_document(expected_bundle)
    expected_capabilities = (
        []
        if expected_bundle == INCUMBENT_BUNDLE
        else sorted(krea_runtime.REQUIRED_CAPABILITIES)
    )
    expected_aliases = contract["runtime_manifest_capability_aliases"]
    if expected_bundle == INCUMBENT_BUNDLE:
        capability_binding_ok = (
            record["capability_manifest_file_sha256"] is None
            and record["capability_manifest_semantic_sha256"] is None
            and record["capabilities"] == []
            and record["runtime_manifest_capability_aliases"] == {}
        )
    else:
        try:
            _require_sha256(
                record["capability_manifest_file_sha256"],
                f"{label} capability manifest file",
            )
            _require_sha256(
                record["capability_manifest_semantic_sha256"],
                f"{label} capability manifest semantic",
            )
        except HKEContractError:
            capability_binding_ok = False
        else:
            capability_binding_ok = (
                record["capabilities"] == expected_capabilities
                and record["runtime_manifest_capability_aliases"] == expected_aliases
            )
    profile_sha = timing.get("profile_sha256")
    first_profile_sha = first.get("timing_profile_sha256")
    timing_observation_ok = False
    if timing_mode == "operator_attested_profile":
        try:
            _require_sha256(profile_sha, f"{label} timing profile")
        except HKEContractError:
            timing_observation_ok = False
        else:
            try:
                profiled_rate = float(first["profiled_seconds_per_step"])
                observed_rate = float(first["observed_seconds_per_step"])
                observed_ratio = float(first["observed_to_profile_ratio"])
                elapsed_rate = float(first["elapsed_since_launch_s"]) / int(
                    first["checkpoint_step"]
                )
            except (TypeError, ValueError, ZeroDivisionError):
                profiled_rate = observed_rate = observed_ratio = elapsed_rate = math.nan
            expected_correction = (
                "faster"
                if observed_ratio < 0.95
                else ("slower" if observed_ratio > 1.05 else "within_profile_band")
            )
            timing_observation_ok = (
                first_profile_sha == profile_sha
                and timing.get("runtime_commit")
                == krea_runtime.runtime_commit_for_bundle(expected_bundle)
                and isinstance(timing.get("measured_dataset_size"), int)
                and timing.get("measured_dataset_size", 0) > 0
                and isinstance(timing.get("current_dataset_size"), int)
                and timing.get("current_dataset_size", 0) > 0
                and isinstance(timing.get("accelerator_identity"), str)
                and bool(timing.get("accelerator_identity", "").strip())
                and all(
                    math.isfinite(item) and item > 0
                    for item in (profiled_rate, observed_rate, observed_ratio)
                )
                and math.isclose(
                    observed_rate, elapsed_rate, rel_tol=1e-6, abs_tol=1e-6
                )
                and math.isclose(
                    observed_ratio,
                    observed_rate / profiled_rate,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
                and first.get("correction") == expected_correction
                and isinstance(first.get("active_plan_exceeds_observed_budget"), bool)
                and all(
                    isinstance(first.get(key), int)
                    and not isinstance(first.get(key), bool)
                    and first.get(key) > 0
                    for key in (
                        "future_budget_cap_steps",
                        "future_target_steps",
                        "future_recommended_steps",
                    )
                )
                and isinstance(first.get("future_step_delta"), int)
                and not isinstance(first.get("future_step_delta"), bool)
            )
    elif timing_mode == "bootstrap_probe_unmeasured":
        timing_observation_ok = (
            profile_sha is None
            and first_profile_sha is None
            and first.get("observation_mode") == "bootstrap_raw_first_checkpoint"
            and timing.get("runtime_commit")
            == krea_runtime.runtime_commit_for_bundle(expected_bundle)
            and timing.get("measured_dataset_size") is None
        )
    elif timing_mode == "incumbent_static":
        timing_observation_ok = (
            expected_bundle == INCUMBENT_BUNDLE
            and profile_sha is None
            and timing.get("runtime_commit") is None
            and first_profile_sha is None
            and first.get("observation_mode") == "bootstrap_raw_first_checkpoint"
        )
    if (
        first.get("bundle_id") != expected_bundle
        or first.get("active_planned_steps") != planned_steps
        or first.get("active_plan_mutable") is not False
        or first.get("active_plan_action") != "observe_only_fixed_subprocess"
        or isinstance(first.get("checkpoint_step"), bool)
        or not isinstance(first.get("checkpoint_step"), int)
        or not 0 < first["checkpoint_step"] <= planned_steps
        or first["checkpoint_step"] != expected_first_checkpoint_step
        or not isinstance(first.get("elapsed_since_launch_s"), (int, float))
        or not 0
        < float(first["elapsed_since_launch_s"])
        <= float(completion.get("training_elapsed_seconds", 0))
        or completion.get("returncode") != 0
        or completion.get("stopped_by_deadline") is not False
        or completion.get("natural_completion") is not True
        or completion.get("artifact_loadable") is not True
        or not isinstance(completion.get("artifact_path"), str)
        or not os.path.isabs(completion.get("artifact_path", ""))
        or not isinstance(completion.get("artifact_name"), str)
        or os.path.basename(completion.get("artifact_path", ""))
        != completion.get("artifact_name")
        or isinstance(completion.get("artifact_size_bytes"), bool)
        or not isinstance(completion.get("artifact_size_bytes"), int)
        or completion.get("artifact_size_bytes", 0) <= 0
        or completion.get("artifact_checkpoint_step") != planned_steps
        or completion.get("completed_steps") != planned_steps
        or completion.get("artifact_size_bytes")
        != completion["artifact_file_identity"].get("size")
        or not capability_binding_ok
        or not timing_observation_ok
        or re.fullmatch(r"[0-9a-f]{32}", str(completion.get("scope_attempt_nonce", "")))
        is None
        or not expected_run_id.endswith(
            ":" + str(completion.get("scope_attempt_nonce", ""))
        )
    ):
        raise HKEContractError(f"{label} runtime observations are inconsistent")
    _require_sha256(completion.get("artifact_sha256"), f"{label} artifact")
    if (
        record["schema"] != 4
        or record["runtime_contract_id"] != krea_runtime.RUNTIME_CONTRACT_ID
        or record["source_run_id"] != expected_run_id
        or record["model_type"] != MODEL_TYPE
        or record["runtime_repository"]
        != krea_runtime.runtime_repository_for_bundle(expected_bundle)
        or record["runtime_commit"]
        != krea_runtime.runtime_commit_for_bundle(expected_bundle)
        or record["bundle"] != expected_bundle
        or record["bundle_claim"] != krea_runtime.bundle_claim_document(expected_bundle)
        or record["bundle_contract_sha256"]
        != krea_runtime.bundle_contract_sha256(expected_bundle)
        or record["bundle_claim"] != contract["claim"]
        or record["generated_config_sha256"] != expected_config_file_sha256
        or record["lifecycle"] != "terminal"
        or effective["planned_steps"] != planned_steps
        or effective["normalized_config_projection"] != expected_projection
        or not krea_runtime.projection_matches_bundle_contract(
            effective["normalized_config_projection"], bundle=expected_bundle
        )
    ):
        raise HKEContractError(f"{label} effective-runtime binding mismatch")
    return record


def build_cell_execution_order(
    *,
    plan_sha256: str,
    plan_cell: Mapping[str, Any],
    owner_identity: str,
    authorized_at_utc: str,
    source_run_id: str,
) -> dict[str, Any]:
    """Bind owner authorization to one exact planned cell.

    This receipt proves neither the cell's ordinal position, predecessor
    completion, chronology, nor exclusive/non-overlapping execution.  Those
    remain an operator procedure unless a separate run log establishes them.
    """

    _require_sha256(plan_sha256, "execution-order plan")
    identity = _validate_cell_identity(plan_cell["cell_identity"], "execution order")
    if plan_cell.get("cell_sha256") != identity["cell_sha256"]:
        raise HKEContractError("execution-order plan cell digest mismatch")
    _validate_utc_timestamp(authorized_at_utc, "execution order")
    if not isinstance(owner_identity, str) or not owner_identity.strip():
        raise HKEContractError("execution-order owner identity is absent")
    task_identity, separator, attempt_nonce = str(source_run_id).rpartition(":")
    if (
        not separator
        or not task_identity
        or len(str(source_run_id)) > 256
        or re.fullmatch(r"[0-9a-f]{32}", attempt_nonce) is None
    ):
        raise HKEContractError("execution-order source run id is invalid")
    body = {
        "schema": SCHEMA,
        "kind": "sn56-week7-hke-owner-cell-execution-order",
        "status": "OWNER_AUTHORIZED_EXACT_CELL_EXECUTION",
        "plan_sha256": plan_sha256,
        "cell_sha256": identity["cell_sha256"],
        "source_run_id": str(source_run_id),
        "owner_identity": owner_identity.strip(),
        "authorized_at_utc": authorized_at_utc,
        "decision": "AUTHORIZE_GPU_EXECUTION_OF_EXACT_CELL",
        "governance": {
            "operator_attested_not_cryptographically_authenticated": True,
            "gpu_execution_authorized": True,
            "checkpoint_promotion_authorized": False,
            "deployment_authorized": False,
        },
    }
    return {**body, "execution_order_sha256": canonical_sha256(body)}


def _validate_cell_execution_order(
    value: Any,
    *,
    plan_sha256: str,
    plan_cell: Mapping[str, Any],
    expected_owner_identity: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HKEContractError("cell execution order is unavailable")
    expected = build_cell_execution_order(
        plan_sha256=plan_sha256,
        plan_cell=plan_cell,
        owner_identity=str(value.get("owner_identity", "")),
        authorized_at_utc=str(value.get("authorized_at_utc", "")),
        source_run_id=str(value.get("source_run_id", "")),
    )
    if (
        dict(value) != expected
        or expected["owner_identity"] != expected_owner_identity.strip()
    ):
        raise HKEContractError("cell execution order does not reproduce")
    return expected


def _require_receipt_cell_binding(
    receipt: Mapping[str, Any], identity: Mapping[str, Any], label: str
) -> None:
    expected = _receipt_cell_binding(identity)
    actual = {key: receipt.get(key) for key in CELL_BINDING_FIELDS}
    if actual != expected:
        raise HKEContractError(f"{label} receipt cell binding mismatch")


def _validate_training_source_record(
    value: Any,
    plan_cell: Mapping[str, Any],
    fixture: Mapping[str, Any],
    execution_order: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    """Validate the embedded source body; a free hash string is not accepted."""

    fields = {
        "schema",
        "kind",
        "run_id",
        "cell_identity",
        "training_inventory",
        "argv",
        "argv_sha256",
        "config_path",
        "cwd",
        "runtime_directory",
        "runtime_revision",
        "generated_config",
        "generated_config_file_sha256",
        "effective_runtime_record",
        "effective_runtime_record_file_sha256",
        "bundle_environment",
        "execution_order_sha256",
        "source_record_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise HKEContractError(f"{label} source record is malformed")
    body = dict(value)
    declared = body.pop("source_record_sha256")
    _require_sha256(declared, f"{label} source record")
    if declared != canonical_sha256(body):
        raise HKEContractError(f"{label} source record digest mismatch")
    if value["schema"] != 2 or value["kind"] != "sn56-week7-hke-training-source":
        raise HKEContractError(f"{label} source record schema is unsupported")
    if (
        not isinstance(value["run_id"], str)
        or not value["run_id"].strip()
        or value["run_id"] != execution_order["source_run_id"]
    ):
        raise HKEContractError(f"{label} source run id is invalid")
    identity = _validate_cell_identity(plan_cell["cell_identity"], label)
    checked_identity = _validate_cell_identity(value["cell_identity"], label)
    if checked_identity != identity:
        raise HKEContractError(f"{label} source record cell mismatch")
    inventory = _validate_inventory_body(value["training_inventory"], label)
    if inventory["semantic_sha256"] != checked_identity["training_inventory_sha256"]:
        raise HKEContractError(f"{label} source inventory is foreign")
    _require_sha256(value["argv_sha256"], f"{label} source argv")
    argv = value["argv"]
    config_path = value["config_path"]
    cwd = value["cwd"]
    runtime_directory = value["runtime_directory"]
    runtime_revision = value["runtime_revision"]
    if not isinstance(runtime_revision, Mapping) or set(runtime_revision) != {
        "repository",
        "commit",
        "tree",
        "directory",
        "clean",
        "observed_at_utc",
        "revision_sha256",
    }:
        raise HKEContractError(f"{label} runtime revision is malformed")
    runtime_revision_body = dict(runtime_revision)
    runtime_revision_sha = runtime_revision_body.pop("revision_sha256")
    _require_sha256(runtime_revision_sha, f"{label} runtime revision")
    _validate_utc_timestamp(runtime_revision.get("observed_at_utc"), f"{label} runtime")
    if (
        runtime_revision_sha != canonical_sha256(runtime_revision_body)
        or runtime_revision.get("repository")
        != krea_runtime.runtime_repository_for_bundle(identity["bundle_id"])
        or runtime_revision.get("commit") != identity["runtime_commit"]
        or runtime_revision.get("tree") != identity["runtime_tree"]
        or runtime_revision.get("directory") != runtime_directory
        or runtime_revision.get("clean") is not True
    ):
        raise HKEContractError(f"{label} executed runtime tree mismatch")
    if (
        value["generated_config"] != plan_cell["config"]
        or canonical_sha256(value["generated_config"])
        != identity["generated_config_sha256"]
        or value["bundle_environment"]
        != {krea_runtime.BUNDLE_ENV: identity["bundle_id"]}
        or value["execution_order_sha256"] != execution_order["execution_order_sha256"]
        or not isinstance(argv, list)
        or len(argv) != 3
        or any(not isinstance(item, str) or not item for item in argv)
        or argv[0] != identity["python_executable"]
        or argv[1] != "run.py"
        or argv[2] != config_path
        or not isinstance(config_path, str)
        or not os.path.isabs(config_path)
        or not isinstance(runtime_directory, str)
        or not os.path.isabs(runtime_directory)
        or cwd != runtime_directory
        or value["argv_sha256"] != canonical_sha256(argv)
    ):
        raise HKEContractError(f"{label} source executed-config binding mismatch")
    runtime_record = _validate_effective_runtime_record(
        value["effective_runtime_record"],
        expected_run_id=value["run_id"],
        expected_bundle=identity["bundle_id"],
        expected_config=value["generated_config"],
        expected_config_file_sha256=value["generated_config_file_sha256"],
        expected_source_file_sha256=value["effective_runtime_record_file_sha256"],
        label=f"{label} source",
    )
    if runtime_record["runtime_commit"] != identity["runtime_commit"]:
        raise HKEContractError(f"{label} source runtime commit mismatch")
    timing = runtime_record["timing"]
    if not isinstance(timing, Mapping):
        raise HKEContractError(f"{label} source timing identity is malformed")
    support = plan_cell.get("timing_support") or {}
    expected_timing_mode = support.get("execution_timing_mode")
    expected_profile_sha = support.get("profile_sha256")
    if expected_timing_mode == "operator_attested_profile":
        if (
            expected_profile_sha is None
            or timing.get("mode") != "operator_attested_profile"
            or timing.get("profile_sha256") != expected_profile_sha
        ):
            raise HKEContractError(f"{label} source timing profile mismatch")
    elif expected_timing_mode == "incumbent_static":
        if identity["bundle_id"] != INCUMBENT_BUNDLE or timing.get("mode") != (
            "incumbent_static"
        ):
            raise HKEContractError(f"{label} incumbent timing mode mismatch")
    elif expected_timing_mode == "bootstrap_probe_unmeasured":
        if timing.get("mode") != "bootstrap_probe_unmeasured":
            raise HKEContractError(f"{label} source probe timing mode mismatch")
    else:
        raise HKEContractError(f"{label} plan timing mode is unsupported")
    if expected_timing_mode != "operator_attested_profile" and (
        expected_profile_sha is not None
    ):
        raise HKEContractError(f"{label} source probe timing mode mismatch")
    timing_accelerator_identity = timing.get("accelerator_identity")
    if expected_timing_mode != "incumbent_static":
        try:
            timing_accelerator_identity = (
                adaptive_timing.validate_accelerator_identity(
                    timing_accelerator_identity
                )
            )
        except Exception as exc:
            raise HKEContractError(
                f"{label} source timing accelerator identity is invalid"
            ) from exc
    if expected_timing_mode != "incumbent_static" and (
        timing.get("runtime_commit") != identity["runtime_commit"]
        or timing.get("current_dataset_size") != fixture["training_row_count"]
        or timing.get("dataset_regime")
        != adaptive_timing.dataset_regime(fixture["training_row_count"])
        or identity["accelerator_uuid"]
        not in str(timing_accelerator_identity)
    ):
        raise HKEContractError(f"{label} source timing/accelerator mismatch")
    return {**body, "source_record_sha256": declared}


def _profile_document(
    profile: adaptive_timing.ThroughputProfile,
) -> dict[str, Any]:
    """Reconstruct the current profile document whose digest it declares."""

    return {
        "schema": adaptive_timing.PROFILE_SCHEMA,
        "kind": adaptive_timing.PROFILE_KIND,
        "bundle_id": profile.bundle_id,
        "bundle_sha256": profile.bundle_sha256,
        "model_type": profile.model_type,
        "measured_dataset_size": profile.measured_dataset_size,
        "dataset_regime": profile.dataset_regime,
        "seconds_per_step": profile.seconds_per_step,
        "startup_seconds": profile.startup_seconds,
        "measurement": {
            "completed_steps": profile.completed_steps,
            "training_elapsed_seconds": profile.training_elapsed_seconds,
            "first_checkpoint_step": profile.first_checkpoint_step,
            "first_checkpoint_elapsed_seconds": (
                profile.first_checkpoint_elapsed_seconds
            ),
        },
        "provenance": {
            "source_run_id": profile.source_run_id,
            "source_record_sha256": profile.source_record_sha256,
            "source_generated_config_sha256": (profile.source_generated_config_sha256),
            "source_config_projection_sha256": (
                profile.source_config_projection_sha256
            ),
            "source_loss_type": profile.source_loss_type,
            "runtime_commit": profile.runtime_commit,
            "measured_at_utc": profile.measured_at_utc,
            "accelerator_identity": profile.accelerator_identity,
        },
        "profile_sha256": profile.profile_sha256,
    }


def _profile_binding_body(
    profile: adaptive_timing.ThroughputProfile,
    *,
    loss: str,
    measured_dataset_size: int,
    measured_config: Mapping[str, Any],
    source_record: Mapping[str, Any],
    accelerator_observation: Mapping[str, Any],
) -> dict[str, Any]:
    checked_observation = _validate_h100_observation(
        accelerator_observation, "timing profile"
    )
    try:
        expected_bundle_sha256 = krea_runtime.bundle_contract_sha256(profile.bundle_id)
        expected_runtime_commit = krea_runtime.runtime_commit_for_bundle(
            profile.bundle_id
        )
    except Exception as exc:
        raise HKEContractError(
            "timing profile names an unsupported runtime bundle"
        ) from exc
    if (
        profile.bundle_sha256 != expected_bundle_sha256
        or profile.runtime_commit != expected_runtime_commit
    ):
        raise HKEContractError("timing profile bundle/runtime provenance mismatch")
    measured_config_value = copy.deepcopy(dict(measured_config))
    measured_config_sha256 = canonical_sha256(measured_config_value)
    generated_config_file_sha256 = hashlib.sha256(
        _generated_config_bytes(measured_config_value)
    ).hexdigest()
    runtime_record = _validate_effective_runtime_record(
        source_record,
        expected_run_id=profile.source_run_id,
        expected_bundle=profile.bundle_id,
        expected_config=measured_config_value,
        expected_config_file_sha256=generated_config_file_sha256,
        expected_source_file_sha256=profile.source_record_sha256,
        label=f"{loss} timing profile",
    )
    timing = runtime_record["timing"]
    first = runtime_record["first_checkpoint_observation"]
    completion = runtime_record["training_completion_observation"]
    source_projection_sha256 = adaptive_timing.canonical_sha256(
        runtime_record["effective"]["normalized_config_projection"]
    )
    expected_rate = float(completion["training_elapsed_seconds"]) / int(
        completion["completed_steps"]
    )
    if (
        _train_node(measured_config_value).get("loss_type") != loss
        or not isinstance(timing, Mapping)
        or timing.get("mode") != "bootstrap_probe_unmeasured"
        or timing.get("profile_sha256") is not None
        or timing.get("runtime_commit") != profile.runtime_commit
        or timing.get("measured_dataset_size") is not None
        or timing.get("current_dataset_size") != measured_dataset_size
        or timing.get("dataset_regime")
        != adaptive_timing.dataset_regime(measured_dataset_size)
        or timing.get("accelerator_identity") != profile.accelerator_identity
        or profile.source_generated_config_sha256
        != runtime_record["generated_config_sha256"]
        or profile.source_config_projection_sha256 != source_projection_sha256
        or profile.source_loss_type != loss
        or profile.completed_steps != completion["completed_steps"]
        or profile.training_elapsed_seconds != completion["training_elapsed_seconds"]
        or profile.first_checkpoint_step != first["checkpoint_step"]
        or profile.first_checkpoint_elapsed_seconds != first["elapsed_since_launch_s"]
        or profile.startup_seconds != 0.0
        or not math.isclose(
            profile.seconds_per_step, expected_rate, rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        raise HKEContractError("timing source differs from its outer binding")
    return {
        "schema": 2,
        "kind": "sn56-week7-hke-timing-profile-binding",
        "loss": loss,
        "bundle_id": profile.bundle_id,
        "bundle_sha256": profile.bundle_sha256,
        "runtime_commit": profile.runtime_commit,
        "template_file_sha256": INCUMBENT_TEMPLATE_FILE_SHA256,
        "training_seed": TRAINING_SEED_A,
        "measured_dataset_size": measured_dataset_size,
        "dataset_regime": adaptive_timing.dataset_regime(measured_dataset_size),
        "measured_config_sha256": measured_config_sha256,
        "source_generated_config_file_sha256": runtime_record[
            "generated_config_sha256"
        ],
        "accelerator_observation": checked_observation,
        "profile_sha256": profile.profile_sha256,
        "source_record_sha256": profile.source_record_sha256,
    }


def bind_timing_profile(
    profile: adaptive_timing.ThroughputProfile,
    *,
    loss: str,
    measured_dataset_size: int,
    measured_config: Mapping[str, Any],
    source_record: Mapping[str, Any],
    accelerator_observation: Mapping[str, Any],
) -> BoundTimingProfile:
    """Create the content-addressed outer binding used by prelaunch.

    This helper is deterministic, not an authority shortcut.  Consumption
    independently revalidates the inner profile and recomputes this digest.
    """

    body = _profile_binding_body(
        profile,
        loss=loss,
        measured_dataset_size=measured_dataset_size,
        measured_config=measured_config,
        source_record=source_record,
        accelerator_observation=accelerator_observation,
    )
    measured_config_sha256 = canonical_sha256(measured_config)
    bound = BoundTimingProfile(
        profile=profile,
        loss=loss,
        measured_dataset_size=measured_dataset_size,
        measured_config_sha256=measured_config_sha256,
        measured_config=copy.deepcopy(dict(measured_config)),
        source_record=copy.deepcopy(dict(source_record)),
        accelerator_observation=copy.deepcopy(dict(accelerator_observation)),
        binding_sha256=canonical_sha256(body),
    )
    if profile.accelerator_identity != bound.accelerator_identity:
        raise HKEContractError(
            "timing profile accelerator label does not match observation"
        )
    return bound


def _bound_profile_document(binding: BoundTimingProfile) -> dict[str, Any]:
    """Serialize every byte needed to repeat profile and binding validation."""

    if not isinstance(binding, BoundTimingProfile):
        raise HKEContractError("bound timing profile document is unavailable")
    binding_body = _profile_binding_body(
        binding.profile,
        loss=binding.loss,
        measured_dataset_size=binding.measured_dataset_size,
        measured_config=binding.measured_config,
        source_record=binding.source_record,
        accelerator_observation=binding.accelerator_observation,
    )
    if binding.measured_config_sha256 != canonical_sha256(
        binding.measured_config
    ) or binding.binding_sha256 != canonical_sha256(binding_body):
        raise HKEContractError("bound timing profile digest mismatch")
    return {
        "profile": _profile_document(binding.profile),
        "source_record": copy.deepcopy(dict(binding.source_record)),
        "binding": {
            **binding_body,
            "binding_sha256": binding.binding_sha256,
        },
    }


def _validate_bound_profile_document(
    value: Mapping[str, Any],
    *,
    loss: str,
    expected_dataset_size: int,
    expected_config: Mapping[str, Any],
    expected_bundle_id: str = INCUMBENT_BUNDLE,
    expected_runtime_commit: str = INCUMBENT_RUNTIME_COMMIT,
) -> BoundTimingProfile:
    """Recreate and validate a serialized timing profile and outer binding."""

    if not isinstance(value, Mapping) or set(value) != {
        "profile",
        "source_record",
        "binding",
    }:
        raise HKEContractError(f"serialized {loss} timing profile is malformed")
    binding_value = value["binding"]
    if not isinstance(binding_value, Mapping):
        raise HKEContractError(f"serialized {loss} timing binding is malformed")
    binding_document = dict(binding_value)
    declared_binding = binding_document.pop("binding_sha256", None)
    _require_sha256(declared_binding, f"serialized {loss} timing binding")
    accelerator_observation = _validate_h100_observation(
        binding_document.get("accelerator_observation"),
        f"serialized {loss} timing accelerator",
    )
    device = accelerator_observation["device"]
    accelerator_identity = adaptive_timing.accelerator_identity(
        name=device["name"],
        memory_total_mib=device["memory_total_mib"],
        uuid=device["uuid"],
    )
    try:
        profile = adaptive_timing.validate_profile(
            value["profile"],
            expected_bundle_id=expected_bundle_id,
            expected_bundle_sha256=krea_runtime.bundle_contract_sha256(
                expected_bundle_id
            ),
            expected_model_type=MODEL_TYPE,
            current_dataset_size=expected_dataset_size,
            expected_dataset_regime=adaptive_timing.dataset_regime(
                expected_dataset_size
            ),
            expected_accelerator_identity=accelerator_identity,
        )
    except Exception as exc:
        raise HKEContractError(
            f"serialized {loss} timing profile failed validation"
        ) from exc
    expected_binding = _profile_binding_body(
        profile,
        loss=loss,
        measured_dataset_size=expected_dataset_size,
        measured_config=expected_config,
        source_record=value["source_record"],
        accelerator_observation=accelerator_observation,
    )
    if binding_document != expected_binding or declared_binding != canonical_sha256(
        expected_binding
    ):
        raise HKEContractError(f"serialized {loss} timing binding mismatch")
    bound = BoundTimingProfile(
        profile=profile,
        loss=loss,
        measured_dataset_size=expected_dataset_size,
        measured_config_sha256=canonical_sha256(expected_config),
        measured_config=copy.deepcopy(dict(expected_config)),
        source_record=copy.deepcopy(dict(value["source_record"])),
        accelerator_observation=accelerator_observation,
        binding_sha256=declared_binding,
    )
    _profile_for_loss(
        {loss: bound},
        loss,
        expected_dataset_size=expected_dataset_size,
        expected_dataset_regime=adaptive_timing.dataset_regime(expected_dataset_size),
        expected_config=expected_config,
        expected_bundle_id=expected_bundle_id,
        expected_runtime_commit=expected_runtime_commit,
    )
    return bound


def _verify_incumbent_source(base_config: Mapping[str, Any]) -> dict[str, str]:
    """Bind planning to the literal reviewed template bytes and semantics."""

    if sha256_file(INCUMBENT_TEMPLATE_PATH) != INCUMBENT_TEMPLATE_FILE_SHA256:
        raise HKEContractError("incumbent template source hash drifted")
    try:
        parsed = yaml.safe_load(INCUMBENT_TEMPLATE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - guarded by repository tests
        raise HKEContractError("incumbent template source is unreadable") from exc
    if not isinstance(parsed, dict) or dict(base_config) != parsed:
        raise HKEContractError(
            "base config is not the exact reviewed incumbent template"
        )
    return {
        "path": str(INCUMBENT_TEMPLATE_PATH.relative_to(REPO_ROOT)),
        "file_sha256": INCUMBENT_TEMPLATE_FILE_SHA256,
        "semantic_sha256": canonical_sha256(parsed),
    }


def _process_node(config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = config["config"]["process"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise HKEContractError("Krea config has no process node") from exc
    if not isinstance(value, dict):
        raise HKEContractError("Krea process node is not an object")
    return value


def _train_node(config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = config["config"]["process"][0]["train"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HKEContractError("Krea config has no train node") from exc
    if not isinstance(value, dict):
        raise HKEContractError("Krea train node is not an object")
    return value


def _save_node(config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = config["config"]["process"][0]["save"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HKEContractError("Krea config has no save node") from exc
    if not isinstance(value, dict):
        raise HKEContractError("Krea save node is not an object")
    return value


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in sorted(value):
            result.update(_flatten(value[key], f"{prefix}/{key}"))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten(item, f"{prefix}/{index}"))
        return result
    return {prefix or "/": value}


def changed_pointers(left: Mapping[str, Any], right: Mapping[str, Any]) -> set[str]:
    lflat = _flatten(left)
    rflat = _flatten(right)
    return {
        key
        for key in set(lflat) | set(rflat)
        if lflat.get(key, object()) != rflat.get(key, object())
    }


def _profile_for_loss(
    profiles: Mapping[str, BoundTimingProfile],
    loss: str,
    *,
    expected_dataset_size: int,
    expected_dataset_regime: str,
    expected_config: Mapping[str, Any],
    expected_bundle_id: str = INCUMBENT_BUNDLE,
    expected_runtime_commit: str = INCUMBENT_RUNTIME_COMMIT,
) -> adaptive_timing.ThroughputProfile:
    binding = profiles.get(loss)
    if not isinstance(binding, BoundTimingProfile):
        raise HKEContractError(f"bound operator-attested {loss} profile is required")
    profile = binding.profile
    if binding.loss != loss:
        raise HKEContractError(f"{loss} profile loss binding mismatch")
    if binding.measured_dataset_size != expected_dataset_size:
        raise HKEContractError(f"{loss} profile dataset-size binding mismatch")
    expected_config_sha256 = canonical_sha256(expected_config)
    if (
        binding.measured_config_sha256 != expected_config_sha256
        or binding.measured_config != expected_config
    ):
        raise HKEContractError(f"{loss} profile config binding mismatch")
    if binding.accelerator_identity != profile.accelerator_identity:
        raise HKEContractError(f"{loss} profile accelerator binding mismatch")
    _validate_h100_observation(
        binding.accelerator_observation, f"{loss} profile accelerator"
    )
    if profile.model_type != MODEL_TYPE:
        raise HKEContractError(f"{loss} profile is not Krea2")
    if profile.bundle_id != expected_bundle_id:
        raise HKEContractError(f"{loss} profile bundle id mismatch")
    expected_bundle_sha256 = krea_runtime.bundle_contract_sha256(expected_bundle_id)
    if profile.bundle_sha256 != expected_bundle_sha256:
        raise HKEContractError(f"{loss} profile bundle digest mismatch")
    if profile.runtime_commit != expected_runtime_commit:
        raise HKEContractError(f"{loss} profile runtime commit mismatch")
    if profile.measured_dataset_size != expected_dataset_size:
        raise HKEContractError(f"{loss} profile measured dataset size mismatch")
    if profile.dataset_regime != expected_dataset_regime:
        raise HKEContractError(f"{loss} profile dataset regime mismatch")
    try:
        validated = adaptive_timing.validate_profile(
            _profile_document(profile),
            expected_bundle_id=expected_bundle_id,
            expected_bundle_sha256=expected_bundle_sha256,
            expected_model_type=MODEL_TYPE,
            current_dataset_size=expected_dataset_size,
            expected_dataset_regime=expected_dataset_regime,
            expected_accelerator_identity=binding.accelerator_identity,
        )
    except Exception as exc:
        raise HKEContractError(
            f"{loss} profile failed internal digest/consistency validation"
        ) from exc
    expected_binding = _profile_binding_body(
        validated,
        loss=loss,
        measured_dataset_size=expected_dataset_size,
        measured_config=expected_config,
        source_record=binding.source_record,
        accelerator_observation=binding.accelerator_observation,
    )
    if binding.binding_sha256 != canonical_sha256(expected_binding):
        raise HKEContractError(f"{loss} profile outer binding digest mismatch")
    return validated


def measured_clock_fill_steps(
    *,
    hours_to_complete: float,
    profile: adaptive_timing.ThroughputProfile,
) -> int:
    """Fill the reviewed window from operator-attested profile throughput.

    The dataset-size target is intentionally absent.  The only ceiling is the
    reviewed Krea policy maximum; safety margin, startup and export reserve are
    retained.  The physical forced-stop gate remains separate and is exercised
    by the later H100 mechanical run.
    """

    if not isinstance(profile, adaptive_timing.ThroughputProfile):
        raise HKEContractError("clock fill requires a validated timing profile")
    if profile.model_type != MODEL_TYPE:
        raise HKEContractError("clock profile model type mismatch")
    try:
        expected_runtime = krea_runtime.runtime_commit_for_bundle(profile.bundle_id)
        expected_bundle = krea_runtime.bundle_contract_sha256(profile.bundle_id)
    except Exception as exc:
        raise HKEContractError("clock profile runtime bundle is unsupported") from exc
    if (
        profile.runtime_commit != expected_runtime
        or profile.bundle_sha256 != expected_bundle
    ):
        raise HKEContractError("clock profile runtime provenance mismatch")
    return _clock_fill_steps(
        hours_to_complete=hours_to_complete,
        seconds_per_step=profile.seconds_per_step,
        startup_seconds=profile.startup_seconds,
    )


def _clock_fill_steps(
    *,
    hours_to_complete: float,
    seconds_per_step: float,
    startup_seconds: float,
) -> int:
    """Materialize one clock plan from already validated conservative inputs."""

    budget = float(hours_to_complete) * 3600.0
    if not math.isfinite(budget) or budget <= 0:
        raise HKEContractError("hours_to_complete must be positive and finite")
    if not math.isfinite(seconds_per_step) or seconds_per_step <= 0:
        raise HKEContractError("seconds_per_step must be positive and finite")
    if not math.isfinite(startup_seconds) or startup_seconds < 0:
        raise HKEContractError("startup_seconds must be non-negative and finite")
    available = (
        budget * recipe.margin_for(MODEL_TYPE)
        - startup_seconds
        - recipe.EXPORT_RESERVE_S
    )
    if available <= 0:
        raise HKEContractError("timing profile leaves no training window")
    steps = int(available / seconds_per_step)
    return max(1, min(int(recipe.STEP_TABLE[MODEL_TYPE]["max"]), steps))


def _materialize_airgapped_krea_base(
    base_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Inject the same load-bearing offline model paths as production Forge."""

    config = copy.deepcopy(dict(base_config))
    process = _process_node(config)
    model = process.get("model")
    if not isinstance(model, dict):
        raise HKEContractError("Krea model config is unavailable")
    base_model_path = model.get("name_or_path")
    if not isinstance(base_model_path, str) or not base_model_path.startswith("/"):
        raise HKEContractError("Krea base-model path is not absolute")
    kwargs = model.setdefault("model_kwargs", {})
    if not isinstance(kwargs, dict):
        raise HKEContractError("Krea model kwargs are malformed")
    kwargs["text_encoder_path"] = KREA2_TEXT_ENCODER_PATH
    kwargs["vae_path"] = base_model_path
    return config


def materialize_current_law_configs(
    base_config: Mapping[str, Any],
    *,
    num_images: int,
    hours_to_complete: float,
) -> dict[str, dict[str, Any]]:
    """Materialize Seed-A C/D timing-source configs on the slower multires path."""

    try:
        count = int(num_images)
    except Exception as exc:
        raise HKEContractError("num_images must be an integer") from exc
    if count <= 0 or isinstance(num_images, bool):
        raise HKEContractError("num_images must be positive")
    original = _materialize_airgapped_krea_base(base_config)
    _process_node(original)["training_seed"] = TRAINING_SEED_A
    template_cadence = int(_save_node(original)["save_every"])
    result: dict[str, dict[str, Any]] = {}
    for arm_id in ("C", "D"):
        config = copy.deepcopy(original)
        _train_node(config)["loss_type"] = ARMS[arm_id]["loss"]
        _train_node(config)["steps"] = FACTORIAL_STEPS
        _save_node(config)["save_every"] = recipe.kill_safe_save_every(
            FACTORIAL_STEPS, template_cadence
        )
        effective, bundle = krea_runtime.materialize_week7_factorial_config(
            config, multires_noise=True
        )
        if bundle != OWNED_MULTIRES_BUNDLE:
            raise HKEContractError("multires timing source selected the wrong runtime")
        result[arm_id] = effective
    return result


def materialize_arms(
    base_config: Mapping[str, Any],
    *,
    num_images: int,
    hours_to_complete: float,
    profiles: Mapping[str, BoundTimingProfile],
) -> dict[str, dict[str, Any]]:
    """Return the matched 1,200-step loss x multires-noise factorial."""

    try:
        count = int(num_images)
    except Exception as exc:
        raise HKEContractError("num_images must be an integer") from exc
    if count <= 0:
        raise HKEContractError("num_images must be positive")

    original = _materialize_airgapped_krea_base(base_config)
    _process_node(original)["training_seed"] = TRAINING_SEED_A
    expected_regime = adaptive_timing.dataset_regime(count)
    template_cadence = int(_save_node(original)["save_every"])
    result: dict[str, dict[str, Any]] = {}
    for arm_id, arm in ARMS.items():
        config = copy.deepcopy(original)
        _train_node(config)["loss_type"] = arm["loss"]
        _train_node(config)["steps"] = FACTORIAL_STEPS
        _save_node(config)["save_every"] = recipe.kill_safe_save_every(
            FACTORIAL_STEPS, template_cadence
        )
        effective, bundle = krea_runtime.materialize_week7_factorial_config(
            config, multires_noise=bool(arm["multires_noise"])
        )
        if bundle != arm["bundle"]:
            raise HKEContractError(f"arm {arm_id} selected the wrong runtime bundle")
        result[arm_id] = effective

    timing_sources = materialize_current_law_configs(
        base_config, num_images=count, hours_to_complete=hours_to_complete
    )

    validated_profiles = {
        loss: _profile_for_loss(
            profiles,
            loss,
            expected_dataset_size=count,
            expected_dataset_regime=expected_regime,
            expected_config=timing_sources["C" if loss == "mae" else "D"],
            expected_bundle_id=OWNED_MULTIRES_BUNDLE,
            expected_runtime_commit=OWNED_RUNTIME_COMMIT,
        )
        for loss in ("mae", "mse")
    }
    if (
        validated_profiles["mae"].accelerator_identity
        != validated_profiles["mse"].accelerator_identity
    ):
        raise HKEContractError(
            "MAE/MSE timing profiles do not declare the same accelerator"
        )
    for loss, profile in validated_profiles.items():
        available = (
            float(hours_to_complete) * 3600.0 * recipe.margin_for(MODEL_TYPE)
            - profile.startup_seconds
            - recipe.EXPORT_RESERVE_S
        )
        if available < FACTORIAL_STEPS * profile.seconds_per_step:
            raise HKEContractError(
                f"{loss} timing profile cannot fit 1200 steps in the 0.75h gate"
            )

    allowed = {
        "/config/process/0/train/loss_type",
        "/config/process/0/train/steps",
        "/config/process/0/save/save_every",
        "/config/process/0/train/multires_noise_iterations",
        "/config/process/0/train/multires_noise_discount",
        "/config/process/0/train/sn56_strict_krea_fields",
    }
    for arm_id, config in result.items():
        unexpected = changed_pointers(original, config) - allowed
        if unexpected:
            raise HKEContractError(
                f"arm {arm_id} changed fields outside the factorial: {sorted(unexpected)}"
            )
    if changed_pointers(result["A"], result["B"]) != {
        "/config/process/0/train/loss_type"
    }:
        raise HKEContractError("A/B differ by more than loss")
    if changed_pointers(result["C"], result["D"]) != {
        "/config/process/0/train/loss_type"
    }:
        raise HKEContractError("C/D differ by more than loss")
    for left, right in (("A", "C"), ("B", "D")):
        delta = changed_pointers(result[left], result[right])
        if delta != {
            "/config/process/0/train/multires_noise_iterations",
            "/config/process/0/train/multires_noise_discount",
        }:
            raise HKEContractError(f"{left}/{right} multires contrast is invalid")
    return result


def materialize_r0(
    base_config: Mapping[str, Any], *, seed: int = TRAINING_SEED_A
) -> dict[str, Any]:
    """Reconstruct the exact incumbent retry horizon as its own terminal run."""

    config = _materialize_airgapped_krea_base(base_config)
    _process_node(config)["training_seed"] = seed
    _train_node(config)["steps"] = R0_STEPS
    _save_node(config)["save_every"] = recipe.kill_safe_save_every(
        R0_STEPS, int(_save_node(config)["save_every"])
    )
    _train_node(config).pop("multires_noise_iterations", None)
    _train_node(config).pop("multires_noise_discount", None)
    _train_node(config).pop("sn56_strict_krea_fields", None)
    return config


def materialize_runtime_bridge(
    base_config: Mapping[str, Any], *, seed: int = TRAINING_SEED_A
) -> dict[str, dict[str, Any]]:
    """Build the no-multires incumbent/owned equivalence bridge."""

    incumbent = _materialize_airgapped_krea_base(base_config)
    _process_node(incumbent)["training_seed"] = seed
    _train_node(incumbent)["steps"] = FACTORIAL_STEPS
    _save_node(incumbent)["save_every"] = recipe.kill_safe_save_every(
        FACTORIAL_STEPS, int(_save_node(incumbent)["save_every"])
    )
    _train_node(incumbent).pop("multires_noise_iterations", None)
    _train_node(incumbent).pop("multires_noise_discount", None)
    owned, bundle = krea_runtime.materialize_week7_factorial_config(
        incumbent, multires_noise=False
    )
    if bundle != OWNED_NO_MULTIRES_BUNDLE:
        raise HKEContractError("bridge selected the wrong owned runtime bundle")
    if changed_pointers(incumbent, owned) != {
        "/config/process/0/train/sn56_strict_krea_fields"
    }:
        raise HKEContractError("bridge configs differ outside the runtime sentinel")
    return {"incumbent": incumbent, "owned": owned}


def required_checkpoint_steps(config: Mapping[str, Any]) -> list[int]:
    """Return only checkpoints the configured trainer can naturally emit."""

    steps = int(_train_node(config)["steps"])
    cadence = int(_save_node(config)["save_every"])
    if steps <= 0 or cadence <= 0:
        raise HKEContractError("checkpoint schedule is invalid")
    return sorted(set(range(cadence, steps + 1, cadence)) | {steps})


def _validate_admission_v3(value: Mapping[str, Any], family: str) -> dict[str, Any]:
    required = {
        "schema",
        "kind",
        "status",
        "family",
        "counts",
        "packs",
        "candidate_semantic_sha256",
        "human_review_sha256",
        "replay_evidence_sha256",
        "dedup_evidence_sha256",
        "ownership_record_sha256",
        "generator_repository",
        "generator_commit",
        "generator_tree",
        "generator_source_path",
        "generator_source_sha256",
        "admission_authority_path",
        "admission_authority_source_sha256",
        "contract_path",
        "contract_source_sha256",
        "discovery_key_commitment_sha256",
        "confirmation_key_commitment_sha256",
        "confirmation_commitment_sha256",
        "discovery_all_row_identity_sha256",
        "generator_revision_sha256",
        "governance",
        "claim_limit",
        "admission_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise HKEContractError(f"{family} schema-3 fixture admission is unavailable")
    body = dict(value)
    declared = body.pop("admission_sha256")
    if declared != fixture_semantic_sha256(body):
        raise HKEContractError(f"{family} fixture admission digest mismatch")
    if (
        value["schema"] != SCHEMA
        or value["kind"] != "sn56-week7-hke-fixture-admission"
        or value["status"] != "PASS"
        or value["family"] != family
        or value["counts"] != EXPECTED_FIXTURE_COUNTS[family]
    ):
        raise HKEContractError(f"{family} fixture is not admitted")
    for key in (
        "candidate_semantic_sha256",
        "human_review_sha256",
        "replay_evidence_sha256",
        "dedup_evidence_sha256",
        "ownership_record_sha256",
        "generator_source_sha256",
        "admission_authority_source_sha256",
        "contract_source_sha256",
        "discovery_key_commitment_sha256",
        "confirmation_key_commitment_sha256",
        "confirmation_commitment_sha256",
        "discovery_all_row_identity_sha256",
        "generator_revision_sha256",
    ):
        _require_sha256(value[key], f"{family} admission {key}")
    if (
        value["discovery_key_commitment_sha256"]
        == value["confirmation_key_commitment_sha256"]
    ):
        raise HKEContractError(f"{family} phase keys are not separated")
    for key in ("generator_commit", "generator_tree"):
        item = value[key]
        if not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{40}", item) is None:
            raise HKEContractError(f"{family} admission {key} is invalid")
    if (
        value["generator_source_path"]
        != "ops/experiments/week7/hke_procedural_renderer.py"
        or value["admission_authority_path"]
        != "ops/experiments/week7/hke_fixture_admission.py"
        or not isinstance(value["generator_repository"], str)
        or not value["generator_repository"].startswith("https://github.com/")
    ):
        raise HKEContractError(f"{family} generator paths are invalid")
    if value["governance"] != {
        "operator_attested_named_human_review": True,
        "agent_review_is_not_human_review": True,
        "admission_authorized": True,
        "gpu_execution_authorized": False,
        "owner_ratification_required_for_gpu": True,
    }:
        raise HKEContractError(f"{family} fixture governance is not admissible")
    packs = value["packs"]
    if not isinstance(packs, Mapping) or set(packs) != set(EXPECTED_PACKS[family]):
        raise HKEContractError(f"{family} fixture pack inventory mismatch")
    for pack, spec in EXPECTED_PACKS[family].items():
        record = packs[pack]
        if spec["phase"] == "discovery":
            fields = {
                "phase",
                "row_count",
                "training_row_count",
                "evaluation_row_count",
                "training_row_identity_sha256",
                "evaluation_row_identity_sha256",
                "training_inventory",
                "training_inventory_sha256",
                "evaluation_inventory",
                "evaluation_inventory_sha256",
            }
            if not isinstance(record, Mapping) or set(record) != fields:
                raise HKEContractError(f"{family}/{pack} discovery pack is malformed")
            training = _validate_inventory_body(
                record["training_inventory"],
                f"{family}/{pack} training",
                expected_pairs=spec["train"],
            )
            evaluation = _validate_inventory_body(
                record["evaluation_inventory"],
                f"{family}/{pack} evaluation",
                expected_pairs=spec["eval"],
            )
            if (
                record["phase"] != "discovery"
                or record["row_count"] != spec["count"]
                or record["training_row_count"] != spec["train"]
                or record["evaluation_row_count"] != spec["eval"]
                or record["training_inventory_sha256"] != training["semantic_sha256"]
                or record["evaluation_inventory_sha256"]
                != evaluation["semantic_sha256"]
                or record["training_row_identity_sha256"]
                == record["evaluation_row_identity_sha256"]
            ):
                raise HKEContractError(f"{family}/{pack} split binding mismatch")
            _require_sha256(
                record["training_row_identity_sha256"], f"{family}/{pack} train rows"
            )
            _require_sha256(
                record["evaluation_row_identity_sha256"], f"{family}/{pack} eval rows"
            )
            train_hashes = {item["sha256"] for item in training["files"]}
            eval_hashes = {item["sha256"] for item in evaluation["files"]}
            if train_hashes & eval_hashes:
                raise HKEContractError(f"{family}/{pack} train/eval bytes overlap")
        else:
            fields = {
                "phase",
                "row_count",
                "training_row_count",
                "evaluation_row_count",
                "semantic_commitment_sha256",
            }
            if not isinstance(record, Mapping) or set(record) != fields:
                raise HKEContractError(
                    f"{family}/{pack} confirmation pack is malformed"
                )
            if (
                record["phase"] != "confirmation"
                or record["row_count"] != spec["count"]
                or record["training_row_count"] != spec["train"]
                or record["evaluation_row_count"] != spec["eval"]
            ):
                raise HKEContractError(f"{family}/{pack} confirmation shape mismatch")
            _require_sha256(
                record["semantic_commitment_sha256"],
                f"{family}/{pack} confirmation commitment",
            )
    return dict(value)


def _validate_admission(value: Mapping[str, Any], family: str) -> dict[str, Any]:
    """Accept only the current content-addressed fixture receipt schema."""

    if isinstance(value, Mapping) and value.get("schema") == SCHEMA:
        return _validate_admission_v3(value, family)
    raise HKEContractError(f"{family} fixture admission schema is superseded")


def _validate_admission_set_v3(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "kind",
        "status",
        "candidate_semantic_sha256",
        "human_review_sha256",
        "replay_evidence",
        "replay_evidence_sha256",
        "dedup_evidence_sha256",
        "ownership_record_sha256",
        "generator_revision",
        "generator_revision_sha256",
        "family_admission_sha256",
        "receipts",
        "authorization",
        "admission_set_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise HKEContractError("fixture admission-set envelope is malformed")
    body = dict(value)
    declared = body.pop("admission_set_sha256")
    _require_sha256(declared, "fixture admission-set")
    if declared != fixture_semantic_sha256(body):
        raise HKEContractError("fixture admission-set digest mismatch")
    if (
        value["schema"] != SCHEMA
        or value["kind"] != "sn56-week7-hke-fixture-admission-set"
        or value["status"] != "PASS"
        or value["authorization"]
        != {
            "fixture_admission_authorized": True,
            "gpu_execution_authorized": False,
            "deployment_authorized": False,
        }
    ):
        raise HKEContractError("fixture admission-set has not passed")
    for key in (
        "candidate_semantic_sha256",
        "human_review_sha256",
        "replay_evidence_sha256",
        "dedup_evidence_sha256",
        "ownership_record_sha256",
    ):
        _require_sha256(value[key], f"fixture admission-set {key}")
    replay = value["replay_evidence"]
    replay_fields = {
        "status",
        "verified_rows",
        "candidate_semantic_sha256",
        "dedup_semantic_sha256",
        "discovery_key_commitment_sha256",
        "confirmation_key_commitment_sha256",
        "contract_path",
        "contract_source_sha256",
    }
    if (
        not isinstance(replay, Mapping)
        or set(replay) != replay_fields
        or replay["status"] != "PASS"
        or replay["verified_rows"]
        != sum(sum(counts.values()) for counts in EXPECTED_FIXTURE_COUNTS.values())
        or replay["candidate_semantic_sha256"] != value["candidate_semantic_sha256"]
        or replay["dedup_semantic_sha256"] != value["dedup_evidence_sha256"]
        or value["replay_evidence_sha256"] != fixture_semantic_sha256(replay)
        or replay["discovery_key_commitment_sha256"]
        == replay["confirmation_key_commitment_sha256"]
    ):
        raise HKEContractError("fixture admission-set replay mismatch")
    revision = value["generator_revision"]
    revision_fields = {
        "repository",
        "commit",
        "tree",
        "renderer_source_path",
        "renderer_source_sha256",
        "admission_authority_path",
        "admission_authority_source_sha256",
        "factor_authority_path",
        "factor_authority_source_sha256",
        "contract_path",
        "contract_source_sha256",
        "pinned_remote_refs",
    }
    if not isinstance(revision, Mapping) or set(revision) != revision_fields:
        raise HKEContractError("fixture admission-set generator revision is malformed")
    for key in (
        "renderer_source_sha256",
        "admission_authority_source_sha256",
        "factor_authority_source_sha256",
        "contract_source_sha256",
    ):
        _require_sha256(revision[key], f"fixture admission-set generator {key}")
    if value["generator_revision_sha256"] != fixture_semantic_sha256(revision):
        raise HKEContractError("fixture admission-set generator digest mismatch")
    for key in ("commit", "tree"):
        item = revision[key]
        if not isinstance(item, str) or len(item) != 40:
            raise HKEContractError(f"fixture admission-set generator {key} is invalid")
        try:
            int(item, 16)
        except ValueError as exc:
            raise HKEContractError(
                f"fixture admission-set generator {key} is invalid"
            ) from exc
    expected_contract_path = "ops/experiments/week7/hke_fixture_contract.json"
    renderer_path = "ops/experiments/week7/hke_procedural_renderer.py"
    admission_path = "ops/experiments/week7/hke_fixture_admission.py"
    factor_path = "ops/experiments/week7/run_hke_factorial.py"
    literal = _literal_revision_identity(
        revision["commit"],
        (renderer_path, admission_path, factor_path, expected_contract_path),
    )
    refs = revision["pinned_remote_refs"]
    if (
        not isinstance(refs, list)
        or not refs
        or any(
            not isinstance(ref, str)
            or re.fullmatch(r"refs/heads/[A-Za-z0-9._/-]+", ref) is None
            for ref in refs
        )
    ):
        raise HKEContractError("fixture admission-set pinned refs are invalid")
    if not all(
        _literal_revision_identity(ref, ())["commit"] == revision["commit"]
        for ref in refs
    ):
        raise HKEContractError("fixture admission-set commit is not on a pinned ref")
    if (
        literal["commit"] != revision["commit"]
        or literal["tree"] != revision["tree"]
        or revision["renderer_source_path"] != renderer_path
        or revision["admission_authority_path"] != admission_path
        or revision["admission_authority_source_sha256"]
        != literal["blob_sha256"][admission_path]
        or revision["admission_authority_source_sha256"]
        != sha256_file(ADMISSION_AUTHORITY_PATH)
        or revision["factor_authority_path"] != factor_path
        or revision["factor_authority_source_sha256"]
        != literal["blob_sha256"][factor_path]
        or revision["factor_authority_source_sha256"]
        != sha256_file(REPO_ROOT / factor_path)
        or revision["renderer_source_sha256"] != literal["blob_sha256"][renderer_path]
        or revision["renderer_source_sha256"] != sha256_file(REPO_ROOT / renderer_path)
        or revision["contract_path"] != expected_contract_path
        or revision["contract_source_sha256"]
        != literal["blob_sha256"][expected_contract_path]
        or revision["contract_source_sha256"]
        != sha256_file(REPO_ROOT / expected_contract_path)
        or replay["contract_path"] != revision["contract_path"]
        or replay["contract_source_sha256"] != revision["contract_source_sha256"]
    ):
        raise HKEContractError("fixture admission-set source binding mismatch")
    receipts = value["receipts"]
    hashes = value["family_admission_sha256"]
    if (
        not isinstance(receipts, Mapping)
        or set(receipts) != set(EXPECTED_PACKS)
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(EXPECTED_PACKS)
    ):
        raise HKEContractError("fixture admission-set family inventory mismatch")
    validated: dict[str, dict[str, Any]] = {}
    for family in EXPECTED_PACKS:
        receipt = _validate_admission(receipts[family], family)
        if (
            hashes[family] != receipt["admission_sha256"]
            or receipt["candidate_semantic_sha256"]
            != value["candidate_semantic_sha256"]
            or receipt["human_review_sha256"] != value["human_review_sha256"]
            or receipt["replay_evidence_sha256"] != value["replay_evidence_sha256"]
            or receipt["dedup_evidence_sha256"] != value["dedup_evidence_sha256"]
            or receipt["ownership_record_sha256"] != value["ownership_record_sha256"]
            or receipt["discovery_key_commitment_sha256"]
            != replay["discovery_key_commitment_sha256"]
            or receipt["confirmation_key_commitment_sha256"]
            != replay["confirmation_key_commitment_sha256"]
            or receipt["generator_commit"] != revision["commit"]
            or receipt["generator_tree"] != revision["tree"]
            or receipt["generator_repository"] != revision["repository"]
            or receipt["generator_source_path"] != revision["renderer_source_path"]
            or receipt["generator_source_sha256"] != revision["renderer_source_sha256"]
            or receipt["admission_authority_path"]
            != revision["admission_authority_path"]
            or receipt["admission_authority_source_sha256"]
            != revision["admission_authority_source_sha256"]
            or receipt["contract_path"] != revision["contract_path"]
            or receipt["contract_source_sha256"] != revision["contract_source_sha256"]
            or receipt["generator_revision_sha256"]
            != value["generator_revision_sha256"]
        ):
            raise HKEContractError(f"{family} receipt disagrees with admission-set")
        validated[family] = receipt
    return {
        "admission_set_sha256": declared,
        "candidate_semantic_sha256": value["candidate_semantic_sha256"],
        "generator_revision": copy.deepcopy(dict(revision)),
        "receipts": validated,
    }


def _validate_admission_set(value: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only the current root admission envelope."""

    if isinstance(value, Mapping) and value.get("schema") == SCHEMA:
        return _validate_admission_set_v3(value)
    raise HKEContractError("fixture admission-set schema is superseded")


def _validate_evaluator_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "harness_sha256",
        "god_commit",
        "comfy_commit",
        "defaults_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise HKEContractError("exact evaluator identity is required")
    checked = dict(value)
    for key in ("harness_sha256", "defaults_sha256"):
        _require_sha256(checked[key], f"evaluator {key}")
    for key in ("god_commit", "comfy_commit"):
        item = checked[key]
        if not isinstance(item, str) or len(item) != 40:
            raise HKEContractError(f"evaluator {key} is not a commit")
        try:
            int(item, 16)
        except ValueError as exc:
            raise HKEContractError(f"evaluator {key} is not a commit") from exc
    return checked


def _validate_owner_ratification(
    value: Mapping[str, Any], *, admission_set: Mapping[str, Any]
) -> dict[str, Any]:
    fields = {
        "schema",
        "kind",
        "status",
        "admission_set_sha256",
        "human_review_sha256",
        "reviewer_identity",
        "generator_revision",
        "generator_revision_sha256",
        "owner_identity",
        "ratified_at_utc",
        "decision",
        "governance",
        "owner_ratification_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise HKEContractError("owner ratification is unavailable")
    body = dict(value)
    declared = body.pop("owner_ratification_sha256")
    if declared != fixture_semantic_sha256(body):
        raise HKEContractError("owner ratification digest mismatch")
    if (
        value["schema"] != SCHEMA
        or value["kind"] != "sn56-week7-hke-owner-ratification"
        or value["status"] != "SEALED_OPERATOR_ATTESTED_OWNER_RATIFICATION"
        or value["admission_set_sha256"] != admission_set["admission_set_sha256"]
        or value["human_review_sha256"] != admission_set["human_review_sha256"]
        or value["generator_revision_sha256"]
        != fixture_semantic_sha256(admission_set["generator_revision"])
        or value["generator_revision"] != admission_set["generator_revision"]
        or value["decision"] != "RATIFY_FOR_PLAN_CONSUMPTION"
        or value["governance"]
        != {
            "operator_attested_not_cryptographically_authenticated": True,
            "agent_cannot_complete_owner_fields": True,
            "gpu_execution_authorized": False,
            "deployment_authorized": False,
            "later_plan_must_consume_exact_ratification": True,
        }
        or not isinstance(value["reviewer_identity"], str)
        or not value["reviewer_identity"].strip()
        or not isinstance(value["owner_identity"], str)
        or not value["owner_identity"].strip()
        or re.fullmatch(
            r"20\d\d-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\dZ",
            str(value["ratified_at_utc"]),
        )
        is None
    ):
        raise HKEContractError("owner ratification binding is invalid")
    return dict(value)


def _execution_identities(
    value: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != {"incumbent", "owned"}:
        raise HKEContractError("incumbent and owned execution identities are required")
    checked = {
        key: _validate_execution_identity(value[key], f"{key} execution")
        for key in ("incumbent", "owned")
    }
    if checked["incumbent"]["runtime_tree"] != krea_runtime.PINNED_BASE_TREE:
        raise HKEContractError("incumbent execution tree is not pinned")
    if checked["owned"]["runtime_tree"] != krea_runtime.OWNED_RUNTIME_TREE:
        raise HKEContractError("owned execution tree is not pinned")
    if checked["incumbent"]["code_tree"] != checked["owned"]["code_tree"]:
        raise HKEContractError("bridge execution code trees differ")
    return checked


def _plan_cell(
    *,
    family: str,
    pack: str,
    phase: str,
    arm: str,
    seed: int,
    config: Mapping[str, Any],
    bundle_id: str,
    fixture: Mapping[str, Any],
    candidate_semantic_sha256: str,
    fixture_admission_sha256: str,
    execution: Mapping[str, str],
    observation: Mapping[str, Any],
    timing_support: Mapping[str, Any] | None,
) -> dict[str, Any]:
    config_value = copy.deepcopy(dict(config))
    config_sha = canonical_sha256(config_value)
    support = copy.deepcopy(dict(timing_support or {}))
    if "execution_timing_mode" not in support:
        support["execution_timing_mode"] = (
            "operator_attested_profile"
            if support.get("profile_sha256") is not None
            else "bootstrap_probe_unmeasured"
        )
    identity = _build_cell_identity(
        family=family,
        pack=pack,
        phase=phase,
        arm=arm,
        seed=seed,
        candidate_semantic_sha256=candidate_semantic_sha256,
        fixture_admission_sha256=fixture_admission_sha256,
        evaluation_row_identity_sha256=fixture["evaluation_row_identity_sha256"],
        generated_config_sha256=config_sha,
        execution_identity=execution,
        training_inventory_sha256=fixture["training_inventory_sha256"],
        evaluation_inventory_sha256=fixture["evaluation_inventory_sha256"],
        bundle_id=bundle_id,
        bundle_sha256=krea_runtime.bundle_contract_sha256(bundle_id),
        runtime_commit=krea_runtime.runtime_commit_for_bundle(bundle_id),
        accelerator_observation_sha256=observation["observation_sha256"],
        accelerator_uuid=observation["device"]["uuid"],
    )
    return {
        "config": config_value,
        "generated_config_sha256": config_sha,
        "planned_steps": int(_train_node(config_value)["steps"]),
        "required_checkpoint_steps": required_checkpoint_steps(config_value),
        "decision_checkpoint_policy": "score_all_natural_freeze_by_d1_minimum",
        "bundle_id": bundle_id,
        "bundle_sha256": krea_runtime.bundle_contract_sha256(bundle_id),
        "runtime_commit": krea_runtime.runtime_commit_for_bundle(bundle_id),
        "timing_support": support,
        "cell_identity": identity,
        "cell_sha256": identity["cell_sha256"],
    }


def build_prelaunch_plan(
    base_config: Mapping[str, Any],
    *,
    admission_set: Mapping[str, Any],
    owner_ratification: Mapping[str, Any],
    profiles_by_family: Mapping[str, Mapping[str, Mapping[str, BoundTimingProfile]]],
    evaluator_identity: Mapping[str, Any],
    execution_identities: Mapping[str, Mapping[str, str]],
    hours_to_complete: float = 0.75,
) -> dict[str, Any]:
    """Build only the launchable bridge + D1 plan for the staged protocol."""

    source_identity = _verify_incumbent_source(base_config)
    checked_set = _validate_admission_set(admission_set)
    ratification = _validate_owner_ratification(
        owner_ratification, admission_set=admission_set
    )
    evaluator = _validate_evaluator_identity(evaluator_identity)
    evaluator_sha = canonical_sha256(evaluator)
    executions = _execution_identities(execution_identities)
    reviewed_tree = checked_set["generator_revision"]["tree"]
    if any(
        execution["code_tree"] != reviewed_tree for execution in executions.values()
    ):
        raise HKEContractError(
            "execution code tree differs from the reviewed generator revision"
        )
    receipt = checked_set["receipts"]["social"]
    admitted_pack = receipt["packs"]["D1"]
    spec = EXPECTED_PACKS["social"]["D1"]
    training_inventory = _validate_inventory_body(
        admitted_pack["training_inventory"],
        "social/D1 training",
        expected_pairs=spec["train"],
    )
    evaluation_inventory = _validate_inventory_body(
        admitted_pack["evaluation_inventory"],
        "social/D1 evaluation",
        expected_pairs=spec["eval"],
    )
    fixture = {
        "family": "social",
        "pack": "D1",
        "phase": "discovery",
        "row_count": spec["count"],
        "training_row_count": spec["train"],
        "evaluation_row_count": spec["eval"],
        "admission_sha256": receipt["admission_sha256"],
        "candidate_semantic_sha256": receipt["candidate_semantic_sha256"],
        "training_row_identity_sha256": admitted_pack["training_row_identity_sha256"],
        "evaluation_row_identity_sha256": admitted_pack[
            "evaluation_row_identity_sha256"
        ],
        "training_inventory": training_inventory,
        "training_inventory_sha256": training_inventory["semantic_sha256"],
        "evaluation_inventory": evaluation_inventory,
        "evaluation_inventory_sha256": evaluation_inventory["semantic_sha256"],
    }
    try:
        d1_profiles = profiles_by_family["social"]["D1"]
    except (KeyError, TypeError) as exc:
        raise HKEContractError("owned-runtime D1 timing profiles are required") from exc
    if not isinstance(d1_profiles, Mapping) or set(d1_profiles) != {"mae", "mse"}:
        raise HKEContractError("D1 requires exact MAE/MSE timing profiles")
    timing_sources = materialize_current_law_configs(
        base_config, num_images=spec["train"], hours_to_complete=hours_to_complete
    )
    checked_profiles: dict[str, BoundTimingProfile] = {}
    observations: dict[str, dict[str, Any]] = {}
    for loss, arm in (("mae", "C"), ("mse", "D")):
        profile = _profile_for_loss(
            d1_profiles,
            loss,
            expected_dataset_size=spec["train"],
            expected_dataset_regime=adaptive_timing.dataset_regime(spec["train"]),
            expected_config=timing_sources[arm],
            expected_bundle_id=OWNED_MULTIRES_BUNDLE,
            expected_runtime_commit=OWNED_RUNTIME_COMMIT,
        )
        observation = _validate_h100_observation(
            d1_profiles[loss].accelerator_observation,
            f"{loss} timing",
            require_fixed_capture=True,
        )
        checked_profiles[loss] = d1_profiles[loss]
        observations[observation["observation_sha256"]] = observation
        available = (
            float(hours_to_complete) * 3600.0 * recipe.margin_for(MODEL_TYPE)
            - profile.startup_seconds
            - recipe.EXPORT_RESERVE_S
        )
        if available < FACTORIAL_STEPS * profile.seconds_per_step:
            raise HKEContractError(f"{loss} timing cannot fit the core factorial")
    if len(observations) != 1:
        raise HKEContractError("timing profiles must use one real H100 observation")
    observation = next(iter(observations.values()))
    arms = materialize_arms(
        base_config,
        num_images=spec["train"],
        hours_to_complete=hours_to_complete,
        profiles=d1_profiles,
    )
    bridge_configs = materialize_runtime_bridge(base_config)
    r0_config = materialize_r0(base_config)
    cells: dict[str, Any] = {"bridge": {}, "d1_core": {}}
    for name, config, bundle, execution_key in (
        ("incumbent", bridge_configs["incumbent"], INCUMBENT_BUNDLE, "incumbent"),
        ("owned", bridge_configs["owned"], OWNED_NO_MULTIRES_BUNDLE, "owned"),
    ):
        cells["bridge"][name] = _plan_cell(
            family="social",
            pack="D1",
            phase="bridge",
            arm=f"bridge-{name}",
            seed=TRAINING_SEED_A,
            config=config,
            bundle_id=bundle,
            fixture=fixture,
            candidate_semantic_sha256=receipt["candidate_semantic_sha256"],
            fixture_admission_sha256=receipt["admission_sha256"],
            execution=executions[execution_key],
            observation=observation,
            timing_support=None,
        )
    cells["d1_core"]["R0"] = _plan_cell(
        family="social",
        pack="D1",
        phase="discovery",
        arm="R0",
        seed=TRAINING_SEED_A,
        config=r0_config,
        bundle_id=INCUMBENT_BUNDLE,
        fixture=fixture,
        candidate_semantic_sha256=receipt["candidate_semantic_sha256"],
        fixture_admission_sha256=receipt["admission_sha256"],
        execution=executions["incumbent"],
        observation=observation,
        timing_support=None,
    )
    for arm, config in arms.items():
        loss = ARMS[arm]["loss"]
        cells["d1_core"][arm] = _plan_cell(
            family="social",
            pack="D1",
            phase="discovery",
            arm=arm,
            seed=TRAINING_SEED_A,
            config=config,
            bundle_id=ARMS[arm]["bundle"],
            fixture=fixture,
            candidate_semantic_sha256=receipt["candidate_semantic_sha256"],
            fixture_admission_sha256=receipt["admission_sha256"],
            execution=executions["owned"],
            observation=observation,
            timing_support=(
                {
                    "execution_timing_mode": "operator_attested_profile",
                    "profile_sha256": checked_profiles[loss].profile.profile_sha256,
                    "binding_sha256": checked_profiles[loss].binding_sha256,
                    "source_arm": arm,
                }
                if arm in {"C", "D"}
                else {
                    "execution_timing_mode": "bootstrap_probe_unmeasured",
                    "profile_sha256": None,
                    "binding_sha256": None,
                    "conservative_source_profile_sha256": (
                        checked_profiles[loss].profile.profile_sha256
                    ),
                    "conservative_source_binding_sha256": (
                        checked_profiles[loss].binding_sha256
                    ),
                    "conservative_source_arm": "C" if loss == "mae" else "D",
                }
            ),
        )
    sealed_commitments = {
        family: {
            pack: receipt_value["packs"][pack]["semantic_commitment_sha256"]
            for pack, pack_spec in EXPECTED_PACKS[family].items()
            if pack_spec["phase"] == "confirmation"
            for receipt_value in (checked_set["receipts"][family],)
        }
        for family in EXPECTED_PACKS
    }
    body = {
        "schema": SCHEMA,
        "kind": "sn56-week7-hke-staged-prelaunch-plan",
        "status": "OWNER_RATIFICATION_BOUND_PRELAUNCH_ONLY",
        "contract_sha256": experiment_contract()["contract_sha256"],
        "source": source_identity,
        "evaluator": evaluator,
        "evaluator_sha256": evaluator_sha,
        "hours_to_complete": float(hours_to_complete),
        "execution_identities": executions,
        "admission_set": copy.deepcopy(dict(admission_set)),
        "admission_set_sha256": checked_set["admission_set_sha256"],
        "owner_ratification": ratification,
        "owner_ratification_sha256": ratification["owner_ratification_sha256"],
        "fixture": fixture,
        "sealed_confirmation_commitments": sealed_commitments,
        "timing_profiles": {
            loss: _bound_profile_document(checked_profiles[loss])
            for loss in ("mae", "mse")
        },
        "cells": cells,
        "zero_lora_control": {
            "required": True,
            "evaluation_row_identity_sha256": fixture["evaluation_row_identity_sha256"],
            "evaluator_sha256": evaluator_sha,
        },
        "operator_procedure_order": [
            "bridge-incumbent",
            "bridge-owned",
            "R0",
            "A",
            "D",
            "B",
            "C",
        ],
        "order_evidence_class": "operator_procedure_not_machine_verified",
        "stage_machine": experiment_contract()["stage_machine"],
        "authorization": {
            "mechanical_plan_ready": True,
            "separate_owner_gpu_order_required": True,
            "gpu_execution_authorized": False,
            "bridge_launch_authorized": False,
            "d1_factorial_launch_authorized": False,
            "checkpoint_promotion_authorized": False,
            "deployment_authorized": False,
        },
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def experiment_contract() -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": KIND,
        "status": "PRELAUNCH_CPU_ONLY",
        "model_type": MODEL_TYPE,
        "incumbent_source": {
            "path": str(INCUMBENT_TEMPLATE_PATH.relative_to(REPO_ROOT)),
            "file_sha256": INCUMBENT_TEMPLATE_FILE_SHA256,
        },
        "training_seeds": {
            "Seed-A": TRAINING_SEED_A,
            "Seed-B": TRAINING_SEED_B,
        },
        "runtime": {
            "r0_bundle": INCUMBENT_BUNDLE,
            "factorial_no_multires_bundle": OWNED_NO_MULTIRES_BUNDLE,
            "factorial_multires_bundle": OWNED_MULTIRES_BUNDLE,
            "incumbent_commit": INCUMBENT_RUNTIME_COMMIT,
            "owned_commit": OWNED_RUNTIME_COMMIT,
            "bridge_required_before_factorial": True,
            "bridge_relative_tolerance": BRIDGE_EQUIVALENCE_TOLERANCE,
            "all_factorial_arms_same_owned_runtime": True,
            "leader_overlay_allowed": False,
        },
        "arms": copy.deepcopy(ARMS),
        "factorial": {
            "r0_depth": R0_STEPS,
            "depth": FACTORIAL_STEPS,
            "loss": ["mae", "mse"],
            "multires_noise": {
                "off": True,
                "on": {
                    "iterations": MULTIRES_NOISE_ITERATIONS,
                    "discount": MULTIRES_NOISE_DISCOUNT,
                },
            },
            "primary": "FutureBound/social",
            "d1_cells": ["R0", "A", "B", "C", "D", "zero-LoRA"],
            "clock_fill_e": "conditional_after_complete_d1_core_only",
        },
        "stage_machine": [
            {
                "stage": "bridge",
                "input": "social/D1",
                "cells": ["incumbent-no-multires", "owned-no-multires"],
                "gate": "relative composite and each component within 0.5%",
            },
            {
                "stage": "d1_core",
                "input": "social/D1 held-out evaluation rows",
                "cells": ["R0-1166", "A-1200", "B-1200", "C-1200", "D-1200"],
                "gate": "score every natural periodic and terminal checkpoint",
            },
            {
                "stage": "d1_optional_e",
                "input": "frozen best D1 recipe",
                "condition": "core complete and predeclared clock-fill eligibility",
            },
            {
                "stage": "d1_candidate_freeze",
                "output": "hash-bound recipe/checkpoint/artifact before D2",
            },
            {
                "stage": "d2_replication",
                "input": "social/D2",
                "cells": ["incumbent/candidate Seed-A", "incumbent/candidate Seed-B"],
            },
            {
                "stage": "confirmation_freeze",
                "gate": "candidate digest fixed before C1 reveal",
            },
            {
                "stage": "c1_confirmation",
                "input": "private social/C1 revealed only after freeze",
            },
            {
                "stage": "product_logo_guardrails",
                "input": "private product/C1 and logo_ui/C1",
            },
            {
                "stage": "c2_optional",
                "condition": "predeclared borderline C1 result only",
            },
        ],
        "checkpoint_policy": {
            "r0_terminal_exactly_1166": True,
            "factorial_1166_checkpoint_required": False,
            "score_every_periodic_checkpoint": True,
            "score_terminal_checkpoint": True,
            "consume_live_selection_record": False,
            "live_promotion_enabled": False,
        },
        "routing": {
            "semantic_router_enabled": False,
            "production_mutation_authorized": False,
        },
        "timing_gate": {
            "source": (
                "operator_attested_internally_validated_profile_bound_to_"
                "exact_bundle_runtime_dataset_config_and_accelerator"
            ),
            "proof_of_measurement": False,
            "field_or_hke_constant_allowed": False,
            "factorial_depth_source": "fixed_reviewed_1200_steps",
            "profiles_do_not_change_factorial_depth": True,
        },
        "decision": {
            "metric": "0.25*prompted_loss + 0.75*blank_loss",
            "lower_is_better": True,
            "paired_bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "paired_bootstrap_seed": BOOTSTRAP_SEED,
            "interval": UNCERTAINTY_LEVEL,
            "uncertainty_scope": (
                "row-resampling uncertainty conditional on one fixed training "
                "run and one evaluator row set; excludes seed, rerun, and "
                "hardware variance"
            ),
            "minimum_composite_improvement": MIN_COMPOSITE_IMPROVEMENT,
            "same_direction_d1_d2": True,
            "discovery_ci_clears_zero_on_at_least_one_pack": True,
            "maximum_prompted_or_blank_regression": MAX_RELATIVE_REGRESSION,
            "both_finalist_seeds_must_agree": True,
            "c1_minimum_composite_improvement": MIN_COMPOSITE_IMPROVEMENT,
            "c1_ci_must_clear_zero": True,
            "product_logo_maximum_regression_each": MAX_RELATIVE_REGRESSION,
            "c2_trigger": {
                "mode": "borderline-only",
                "composite_relative_improvement_low_inclusive": (
                    C2_BORDERLINE_COMPOSITE_LOW
                ),
                "composite_relative_improvement_high_inclusive": (
                    C2_BORDERLINE_COMPOSITE_HIGH
                ),
                "paired_ci95_lower_absolute_max": C2_BORDERLINE_CI_LOWER_ABS_MAX,
            },
            "c2_acceptance": {
                "same_direction_required": True,
                "maximum_prompted_or_blank_regression": MAX_RELATIVE_REGRESSION,
                "minimum_composite_improvement": None,
                "ci_must_clear_zero": False,
            },
        },
        "authorization": {
            "fixture_admission_required": True,
            "owner_ratification_required": True,
            "external_exact_sha_audit_pass_required_as_process_gate": True,
            "operator_attested_profiles_required": True,
            "gpu_execution_authorized": False,
            "deployment_authorized": False,
        },
    }
    return {**body, "contract_sha256": canonical_sha256(body)}


def _composite(row: Mapping[str, Any]) -> float:
    prompted = row.get("prompted_loss")
    blank = row.get("blank_loss")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (prompted, blank)
    ):
        raise HKEContractError("score row losses must be JSON numbers")
    prompted = float(prompted)
    blank = float(blank)
    if (
        not math.isfinite(prompted)
        or not math.isfinite(blank)
        or prompted < 0
        or blank < 0
    ):
        raise HKEContractError("score row losses must be finite and non-negative")
    return 0.25 * prompted + 0.75 * blank


def _validate_plan_v3(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HKEContractError("analysis requires a staged prelaunch plan")
    body = dict(value)
    declared = body.pop("plan_sha256", None)
    _require_sha256(declared, "staged prelaunch plan")
    if declared != canonical_sha256(body):
        raise HKEContractError("staged prelaunch plan digest mismatch")
    fields = {
        "schema",
        "kind",
        "status",
        "contract_sha256",
        "source",
        "evaluator",
        "evaluator_sha256",
        "hours_to_complete",
        "execution_identities",
        "admission_set",
        "admission_set_sha256",
        "owner_ratification",
        "owner_ratification_sha256",
        "fixture",
        "sealed_confirmation_commitments",
        "timing_profiles",
        "cells",
        "zero_lora_control",
        "operator_procedure_order",
        "order_evidence_class",
        "stage_machine",
        "authorization",
    }
    if set(body) != fields:
        raise HKEContractError("staged prelaunch plan envelope is malformed")
    if (
        body["schema"] != SCHEMA
        or body["kind"] != "sn56-week7-hke-staged-prelaunch-plan"
        or body["status"] != "OWNER_RATIFICATION_BOUND_PRELAUNCH_ONLY"
        or body["contract_sha256"] != experiment_contract()["contract_sha256"]
        or body["stage_machine"] != experiment_contract()["stage_machine"]
    ):
        raise HKEContractError("staged prelaunch plan contract mismatch")
    if body["authorization"] != {
        "mechanical_plan_ready": True,
        "separate_owner_gpu_order_required": True,
        "gpu_execution_authorized": False,
        "bridge_launch_authorized": False,
        "d1_factorial_launch_authorized": False,
        "checkpoint_promotion_authorized": False,
        "deployment_authorized": False,
    }:
        raise HKEContractError("staged prelaunch authorization is invalid")
    source_config = yaml.safe_load(INCUMBENT_TEMPLATE_PATH.read_text(encoding="utf-8"))
    if body["source"] != _verify_incumbent_source(source_config):
        raise HKEContractError("staged plan incumbent source mismatch")
    checked_set = _validate_admission_set(body["admission_set"])
    if checked_set["admission_set_sha256"] != body["admission_set_sha256"]:
        raise HKEContractError("staged plan admission root mismatch")
    ratification = _validate_owner_ratification(
        body["owner_ratification"], admission_set=body["admission_set"]
    )
    if ratification["owner_ratification_sha256"] != body["owner_ratification_sha256"]:
        raise HKEContractError("staged plan ratification mismatch")
    evaluator = _validate_evaluator_identity(body["evaluator"])
    if body["evaluator_sha256"] != canonical_sha256(evaluator):
        raise HKEContractError("staged plan evaluator mismatch")
    executions = _execution_identities(body["execution_identities"])
    timing_sources = materialize_current_law_configs(
        source_config,
        num_images=EXPECTED_PACKS["social"]["D1"]["train"],
        hours_to_complete=float(body["hours_to_complete"]),
    )
    profiles = {
        loss: _validate_bound_profile_document(
            body["timing_profiles"][loss],
            loss=loss,
            expected_dataset_size=EXPECTED_PACKS["social"]["D1"]["train"],
            expected_config=timing_sources["C" if loss == "mae" else "D"],
            expected_bundle_id=OWNED_MULTIRES_BUNDLE,
            expected_runtime_commit=OWNED_RUNTIME_COMMIT,
        )
        for loss in ("mae", "mse")
    }
    for loss, profile in profiles.items():
        _validate_h100_observation(
            profile.accelerator_observation,
            f"serialized {loss} timing",
            require_fixed_capture=True,
        )
    rebuilt = build_prelaunch_plan(
        source_config,
        admission_set=body["admission_set"],
        owner_ratification=ratification,
        profiles_by_family={"social": {"D1": profiles}},
        evaluator_identity=evaluator,
        execution_identities=executions,
        hours_to_complete=float(body["hours_to_complete"]),
    )
    if dict(value) != rebuilt:
        raise HKEContractError("staged prelaunch plan does not reproduce exactly")
    return rebuilt


def _validate_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping) and value.get("schema") == SCHEMA:
        return _validate_plan_v3(value)
    raise HKEContractError("prelaunch plan schema is superseded")


def _validate_hashed_receipt(
    value: Any, *, label: str, digest_field: str, required_fields: set[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != required_fields:
        raise HKEContractError(f"{label} receipt is malformed")
    body = dict(value)
    declared = body.pop(digest_field)
    _require_sha256(declared, f"{label} receipt")
    if declared != canonical_sha256(body):
        raise HKEContractError(f"{label} receipt digest mismatch")
    return {**body, digest_field: declared}


def _validate_score_rows(
    rows: Any, *, fixture: Mapping[str, Any], label: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    if not isinstance(rows, list) or len(rows) != fixture["evaluation_row_count"]:
        raise HKEContractError(f"{label} does not use the held-out row count")
    row_map: dict[str, dict[str, float]] = {}
    identities: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "row_id",
            "row_sha256",
            "prompted_loss",
            "blank_loss",
        }:
            raise HKEContractError(f"{label} row is malformed")
        row_id = row["row_id"]
        row_sha = _require_sha256(row["row_sha256"], f"{label} row")
        if not isinstance(row_id, str) or not row_id:
            raise HKEContractError(f"{label} row id is invalid")
        key = f"{row_id}:{row_sha}"
        if key in row_map:
            raise HKEContractError(f"{label} row is duplicated")
        _composite(row)
        row_map[key] = {
            "prompted_loss": float(row["prompted_loss"]),
            "blank_loss": float(row["blank_loss"]),
        }
        identities.append({"row_id": row_id, "row_sha256": row_sha})
    if fixture_semantic_sha256(identities) != fixture["evaluation_row_identity_sha256"]:
        raise HKEContractError(f"{label} uses foreign or training rows")
    return [dict(row) for row in rows], row_map


def _validate_training_receipt(
    value: Any,
    *,
    plan_sha256: str,
    plan_cell: Mapping[str, Any],
    fixture: Mapping[str, Any],
    execution_order: Mapping[str, Any],
) -> dict[str, Any]:
    fields = set(CELL_BINDING_FIELDS) | {
        "schema",
        "kind",
        "status",
        "evidence_class",
        "plan_sha256",
        "planned_steps",
        "completed_steps",
        "terminal_artifact_sha256",
        "terminal_artifact_bytes",
        "terminal_checkpoint_step",
        "source_record",
        "source_record_sha256",
        "training_receipt_sha256",
    }
    receipt = _validate_hashed_receipt(
        value,
        label="training",
        digest_field="training_receipt_sha256",
        required_fields=fields,
    )
    identity = _validate_cell_identity(plan_cell["cell_identity"], "training")
    _require_receipt_cell_binding(receipt, identity, "training")
    source = _validate_training_source_record(
        receipt["source_record"], plan_cell, fixture, execution_order, "training"
    )
    completion = source["effective_runtime_record"]["training_completion_observation"]
    if (
        receipt["schema"] != SCHEMA
        or receipt["kind"] != "sn56-week7-hke-training-receipt"
        or receipt["status"] != "OPERATOR_ATTESTED_PASS"
        or receipt["evidence_class"]
        != "content_bound_operator_attested_not_independent_proof"
        or receipt["plan_sha256"] != plan_sha256
        or receipt["planned_steps"] != plan_cell["planned_steps"]
        or receipt["completed_steps"] != plan_cell["planned_steps"]
        or receipt["terminal_checkpoint_step"] != plan_cell["planned_steps"]
        or receipt["source_record_sha256"] != source["source_record_sha256"]
        or completion.get("returncode") != 0
        or completion.get("stopped_by_deadline") is not False
        or completion.get("natural_completion") is not True
        or completion.get("artifact_loadable") is not True
        or completion.get("artifact_sha256") != receipt["terminal_artifact_sha256"]
        or completion.get("artifact_size_bytes") != receipt["terminal_artifact_bytes"]
        or completion.get("artifact_checkpoint_step")
        != receipt["terminal_checkpoint_step"]
        or completion.get("completed_steps") != receipt["completed_steps"]
        or not source["run_id"].endswith(
            ":" + str(completion.get("scope_attempt_nonce", ""))
        )
    ):
        raise HKEContractError("training receipt authority binding mismatch")
    _require_sha256(receipt["terminal_artifact_sha256"], "terminal artifact")
    if (
        isinstance(receipt["terminal_artifact_bytes"], bool)
        or not isinstance(receipt["terminal_artifact_bytes"], int)
        or receipt["terminal_artifact_bytes"] <= 0
    ):
        raise HKEContractError("terminal artifact size is invalid")
    expected_inventory = _validate_inventory_body(
        fixture["training_inventory"],
        "training fixture",
        expected_pairs=fixture["training_row_count"],
    )
    if source["training_inventory"] != expected_inventory:
        raise HKEContractError("training source inventory differs from the fixture")
    return receipt


def _validate_checkpoint_receipt(
    value: Any,
    *,
    plan_sha256: str,
    plan_cell: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    fields = set(CELL_BINDING_FIELDS) | {
        "schema",
        "kind",
        "status",
        "plan_sha256",
        "checkpoint_step",
        "artifact_sha256",
        "artifact_bytes",
        "loadable",
        "source_record_sha256",
        "training_receipt_sha256",
        "checkpoint_receipt_sha256",
    }
    receipt = _validate_hashed_receipt(
        value,
        label="checkpoint",
        digest_field="checkpoint_receipt_sha256",
        required_fields=fields,
    )
    identity = _validate_cell_identity(plan_cell["cell_identity"], "checkpoint")
    _require_receipt_cell_binding(receipt, identity, "checkpoint")
    if (
        receipt["schema"] != SCHEMA
        or receipt["kind"] != "sn56-week7-hke-checkpoint-receipt"
        or receipt["status"] != "OPERATOR_ATTESTED_PASS"
        or receipt["plan_sha256"] != plan_sha256
        or receipt["source_record_sha256"] != training_receipt["source_record_sha256"]
        or receipt["training_receipt_sha256"]
        != training_receipt["training_receipt_sha256"]
        or receipt["checkpoint_step"] not in plan_cell["required_checkpoint_steps"]
        or receipt["loadable"] is not True
    ):
        raise HKEContractError("checkpoint receipt authority binding mismatch")
    _require_sha256(receipt["artifact_sha256"], "checkpoint artifact")
    if (
        isinstance(receipt["artifact_bytes"], bool)
        or not isinstance(receipt["artifact_bytes"], int)
        or receipt["artifact_bytes"] <= 0
    ):
        raise HKEContractError("checkpoint artifact size is invalid")
    return receipt


def _validate_attachment_receipt(
    value: Any,
    *,
    plan_sha256: str,
    plan_cell: Mapping[str, Any],
    checkpoint_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    fields = set(CELL_BINDING_FIELDS) | {
        "schema",
        "kind",
        "status",
        "evidence_class",
        "plan_sha256",
        "checkpoint_receipt_sha256",
        "checkpoint_step",
        "artifact_sha256",
        "loaded_key_count",
        "unloaded_key_count",
        "log_sha256",
        "attachment_receipt_sha256",
    }
    receipt = _validate_hashed_receipt(
        value,
        label="attachment",
        digest_field="attachment_receipt_sha256",
        required_fields=fields,
    )
    identity = _validate_cell_identity(plan_cell["cell_identity"], "attachment")
    _require_receipt_cell_binding(receipt, identity, "attachment")
    if (
        receipt["schema"] != SCHEMA
        or receipt["kind"] != "sn56-week7-hke-attachment-receipt"
        or receipt["status"] != "OPERATOR_ATTESTED_PASS"
        or receipt["evidence_class"]
        != "content_bound_operator_attested_not_independent_proof"
        or receipt["plan_sha256"] != plan_sha256
        or receipt["checkpoint_receipt_sha256"]
        != checkpoint_receipt["checkpoint_receipt_sha256"]
        or receipt["checkpoint_step"] != checkpoint_receipt["checkpoint_step"]
        or receipt["artifact_sha256"] != checkpoint_receipt["artifact_sha256"]
        or isinstance(receipt["loaded_key_count"], bool)
        or not isinstance(receipt["loaded_key_count"], int)
        or receipt["loaded_key_count"] <= 0
        or receipt["unloaded_key_count"] != 0
    ):
        raise HKEContractError("attachment receipt authority binding mismatch")
    _require_sha256(receipt["log_sha256"], "attachment log")
    return receipt


def _validate_score_receipt(
    value: Any,
    *,
    plan_sha256: str,
    plan_cell: Mapping[str, Any],
    fixture: Mapping[str, Any],
    evaluator_sha256: str,
    checkpoint_receipt: Mapping[str, Any],
    attachment_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    fields = set(CELL_BINDING_FIELDS) | {
        "schema",
        "kind",
        "status",
        "evidence_class",
        "plan_sha256",
        "checkpoint_receipt_sha256",
        "attachment_receipt_sha256",
        "checkpoint_step",
        "artifact_sha256",
        "evaluator_sha256",
        "rows",
        "rows_sha256",
        "score_receipt_sha256",
    }
    receipt = _validate_hashed_receipt(
        value,
        label="score",
        digest_field="score_receipt_sha256",
        required_fields=fields,
    )
    identity = _validate_cell_identity(plan_cell["cell_identity"], "score")
    _require_receipt_cell_binding(receipt, identity, "score")
    if (
        receipt["schema"] != SCHEMA
        or receipt["kind"] != "sn56-week7-hke-exact-score-receipt"
        or receipt["status"] != "OPERATOR_ATTESTED_PASS"
        or receipt["evidence_class"]
        != "content_bound_operator_attested_not_independent_proof"
        or receipt["plan_sha256"] != plan_sha256
        or receipt["checkpoint_receipt_sha256"]
        != checkpoint_receipt["checkpoint_receipt_sha256"]
        or receipt["attachment_receipt_sha256"]
        != attachment_receipt["attachment_receipt_sha256"]
        or receipt["checkpoint_step"] != checkpoint_receipt["checkpoint_step"]
        or receipt["artifact_sha256"] != checkpoint_receipt["artifact_sha256"]
        or receipt["evaluator_sha256"] != evaluator_sha256
        or receipt["rows_sha256"] != canonical_sha256(receipt["rows"])
    ):
        raise HKEContractError("score receipt authority binding mismatch")
    rows, row_map = _validate_score_rows(
        receipt["rows"], fixture=fixture, label="score receipt"
    )
    receipt["rows"] = rows
    return receipt, row_map


def _plan_owner_identity(plan: Mapping[str, Any]) -> str:
    """Resolve the ratified owner through a staged plan's immutable ancestry."""

    ratification = plan.get("owner_ratification")
    if isinstance(ratification, Mapping):
        owner = ratification.get("owner_identity")
    elif isinstance(plan.get("owner_identity"), str):
        owner = plan.get("owner_identity")
    else:
        owner = None
        for key in (
            "source_prelaunch_plan",
            "source_d2_plan",
            "source_confirmation_plan",
        ):
            source = plan.get(key)
            if isinstance(source, Mapping):
                owner = _plan_owner_identity(source)
                break
    if not isinstance(owner, str) or not owner.strip():
        raise HKEContractError("plan owner identity is unavailable")
    return owner.strip()


def validate_score_curve(
    value: Mapping[str, Any],
    *,
    plan_sha256: str,
    plan_cell: Mapping[str, Any],
    fixture: Mapping[str, Any],
    evaluator_sha256: str,
    expected_owner_identity: str,
) -> dict[int, dict[str, Any]]:
    """Validate one exact-scored, attachment-proven natural checkpoint curve."""

    fields = {
        "schema",
        "kind",
        "status",
        "evidence_class",
        "plan_sha256",
        "cell_sha256",
        "execution_order",
        "execution_order_sha256",
        "training_receipt",
        "checkpoints",
        "curve_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise HKEContractError("score curve is malformed")
    body = dict(value)
    declared = body.pop("curve_sha256")
    if declared != canonical_sha256(body):
        raise HKEContractError("score curve digest mismatch")
    if (
        value["schema"] != SCHEMA
        or value["kind"] != "sn56-week7-hke-score-curve"
        or value["status"] != "OPERATOR_ATTESTED_PASS"
        or value["evidence_class"]
        != "content_bound_operator_attested_not_independent_proof"
        or value["plan_sha256"] != plan_sha256
        or value["cell_sha256"] != plan_cell["cell_sha256"]
    ):
        raise HKEContractError("score curve authority binding mismatch")
    execution_order = _validate_cell_execution_order(
        value["execution_order"],
        plan_sha256=plan_sha256,
        plan_cell=plan_cell,
        expected_owner_identity=expected_owner_identity,
    )
    if value["execution_order_sha256"] != execution_order["execution_order_sha256"]:
        raise HKEContractError("score curve execution-order binding mismatch")
    training_receipt = _validate_training_receipt(
        value["training_receipt"],
        plan_sha256=plan_sha256,
        plan_cell=plan_cell,
        fixture=fixture,
        execution_order=execution_order,
    )
    checkpoints = value["checkpoints"]
    if not isinstance(checkpoints, list):
        raise HKEContractError("score curve checkpoint list is absent")
    expected_steps = plan_cell["required_checkpoint_steps"]
    if [
        row.get("checkpoint_receipt", {}).get("checkpoint_step") for row in checkpoints
    ] != expected_steps:
        raise HKEContractError("score curve is not the natural checkpoint schedule")
    result: dict[int, dict[str, Any]] = {}
    for entry in checkpoints:
        if not isinstance(entry, Mapping) or set(entry) != {
            "checkpoint_receipt",
            "attachment_receipt",
            "score_receipt",
            "entry_sha256",
        }:
            raise HKEContractError("score curve checkpoint entry is malformed")
        entry_body = dict(entry)
        entry_digest = entry_body.pop("entry_sha256")
        if entry_digest != canonical_sha256(entry_body):
            raise HKEContractError("score curve checkpoint entry digest mismatch")
        checkpoint = _validate_checkpoint_receipt(
            entry["checkpoint_receipt"],
            plan_sha256=plan_sha256,
            plan_cell=plan_cell,
            training_receipt=training_receipt,
        )
        attachment = _validate_attachment_receipt(
            entry["attachment_receipt"],
            plan_sha256=plan_sha256,
            plan_cell=plan_cell,
            checkpoint_receipt=checkpoint,
        )
        _score, row_map = _validate_score_receipt(
            entry["score_receipt"],
            plan_sha256=plan_sha256,
            plan_cell=plan_cell,
            fixture=fixture,
            evaluator_sha256=evaluator_sha256,
            checkpoint_receipt=checkpoint,
            attachment_receipt=attachment,
        )
        result[int(checkpoint["checkpoint_step"])] = {
            "artifact_sha256": checkpoint["artifact_sha256"],
            "artifact_bytes": checkpoint["artifact_bytes"],
            "rows": row_map,
        }
    terminal = next(
        checkpoint["checkpoint_receipt"]
        for checkpoint in checkpoints
        if checkpoint["checkpoint_receipt"]["checkpoint_step"]
        == plan_cell["planned_steps"]
    )
    if (
        terminal["artifact_sha256"] != training_receipt["terminal_artifact_sha256"]
        or terminal["artifact_bytes"] != training_receipt["terminal_artifact_bytes"]
    ):
        raise HKEContractError("terminal training artifact differs from checkpoint")
    return result


def analyze_runtime_bridge(
    plan_value: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    plan = _validate_plan(plan_value)
    fields = {
        "schema",
        "kind",
        "plan_sha256",
        "curves",
        "evidence_sha256",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != fields:
        raise HKEContractError("runtime bridge evidence is incomplete")
    body = dict(evidence)
    declared = body.pop("evidence_sha256")
    if declared != canonical_sha256(body):
        raise HKEContractError("runtime bridge evidence digest mismatch")
    if (
        evidence["schema"] != SCHEMA
        or evidence["kind"] != "sn56-week7-hke-runtime-bridge-evidence"
        or evidence["plan_sha256"] != plan["plan_sha256"]
        or not isinstance(evidence["curves"], Mapping)
        or set(evidence["curves"]) != {"incumbent", "owned"}
    ):
        raise HKEContractError("runtime bridge evidence binding mismatch")
    curves = {
        name: validate_score_curve(
            evidence["curves"][name],
            plan_sha256=plan["plan_sha256"],
            plan_cell=plan["cells"]["bridge"][name],
            fixture=plan["fixture"],
            evaluator_sha256=plan["evaluator_sha256"],
            expected_owner_identity=_plan_owner_identity(plan),
        )
        for name in ("incumbent", "owned")
    }
    incumbent_rows = curves["incumbent"][
        plan["cells"]["bridge"]["incumbent"]["planned_steps"]
    ]["rows"]
    owned_rows = curves["owned"][plan["cells"]["bridge"]["owned"]["planned_steps"]][
        "rows"
    ]
    comparison = compare_heldout_rows(incumbent_rows, owned_rows)
    equivalence = all(
        abs(float(comparison["relative_improvement"][component]))
        <= BRIDGE_EQUIVALENCE_TOLERANCE
        for component in ("prompted", "blank", "composite")
    )
    result = {
        "schema": SCHEMA,
        "kind": "sn56-week7-hke-runtime-bridge-decision",
        "status": "PASS" if equivalence else "HOLD",
        "plan_sha256": plan["plan_sha256"],
        "evidence_sha256": evidence["evidence_sha256"],
        "comparison": comparison,
        "equivalence_tolerance": BRIDGE_EQUIVALENCE_TOLERANCE,
        "d1_factorial_mechanically_unblocked": equivalence,
        "separate_owner_gpu_order_required": True,
        "gpu_execution_authorized": False,
        "d1_factorial_launch_authorized": False,
        "checkpoint_promotion_authorized": False,
        "deployment_authorized": False,
    }
    return {**result, "bridge_receipt_sha256": canonical_sha256(result)}


def _validate_zero_lora(
    value: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    fields = {
        "schema",
        "kind",
        "status",
        "plan_sha256",
        "evaluator_sha256",
        "evaluation_row_identity_sha256",
        "rows",
        "rows_sha256",
        "zero_receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise HKEContractError("zero-LoRA evidence is missing")
    body = dict(value)
    declared = body.pop("zero_receipt_sha256")
    if declared != canonical_sha256(body):
        raise HKEContractError("zero-LoRA evidence digest mismatch")
    fixture = plan["fixture"]
    if (
        value["schema"] != SCHEMA
        or value["kind"] != "sn56-week7-hke-zero-lora-score"
        or value["status"] != "OPERATOR_ATTESTED_PASS"
        or value["plan_sha256"] != plan["plan_sha256"]
        or value["evaluator_sha256"] != plan["evaluator_sha256"]
        or value["evaluation_row_identity_sha256"]
        != fixture["evaluation_row_identity_sha256"]
        or not isinstance(value["rows"], list)
        or len(value["rows"]) != fixture["evaluation_row_count"]
        or value["rows_sha256"] != canonical_sha256(value["rows"])
    ):
        raise HKEContractError("zero-LoRA evidence binding mismatch")
    result: dict[str, dict[str, float]] = {}
    identities: list[dict[str, str]] = []
    for row in value["rows"]:
        if not isinstance(row, Mapping) or set(row) != {
            "row_id",
            "row_sha256",
            "prompted_loss",
            "blank_loss",
        }:
            raise HKEContractError("zero-LoRA score row is malformed")
        row_id = row["row_id"]
        row_sha = _require_sha256(row["row_sha256"], "zero-LoRA row")
        _composite(row)
        key = f"{row_id}:{row_sha}"
        if key in result:
            raise HKEContractError("zero-LoRA score row is duplicated")
        result[key] = {
            "prompted_loss": float(row["prompted_loss"]),
            "blank_loss": float(row["blank_loss"]),
        }
        identities.append({"row_id": row_id, "row_sha256": row_sha})
    if fixture_semantic_sha256(identities) != fixture["evaluation_row_identity_sha256"]:
        raise HKEContractError("zero-LoRA uses foreign rows")
    return result


def _validated_d1_core(
    plan_value: Mapping[str, Any],
    d1_evidence: Mapping[str, Any],
    *,
    bridge_evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[int, dict[str, Any]]]]:
    """Validate the bridge, full D1 curves, zero baseline, and evidence envelope."""

    plan = _validate_plan(plan_value)
    bridge = analyze_runtime_bridge(plan, bridge_evidence)
    if bridge["status"] != "PASS":
        raise HKEContractError("runtime bridge has not cleared")
    fields = {
        "schema",
        "kind",
        "status",
        "plan_sha256",
        "curves",
        "zero_lora",
        "evidence_sha256",
    }
    if not isinstance(d1_evidence, Mapping) or set(d1_evidence) != fields:
        raise HKEContractError("D1 core evidence is incomplete")
    evidence_body = dict(d1_evidence)
    evidence_declared = evidence_body.pop("evidence_sha256")
    if evidence_declared != canonical_sha256(evidence_body):
        raise HKEContractError("D1 evidence digest mismatch")
    if (
        d1_evidence["schema"] != SCHEMA
        or d1_evidence["kind"] != "sn56-week7-hke-d1-core-evidence"
        or d1_evidence["status"] != "OPERATOR_ATTESTED_PASS"
        or d1_evidence["plan_sha256"] != plan["plan_sha256"]
        or not isinstance(d1_evidence["curves"], Mapping)
        or set(d1_evidence["curves"]) != {"R0", "A", "B", "C", "D"}
    ):
        raise HKEContractError("D1 evidence authority binding mismatch")
    curves = {
        arm: validate_score_curve(
            d1_evidence["curves"][arm],
            plan_sha256=plan["plan_sha256"],
            plan_cell=plan["cells"]["d1_core"][arm],
            fixture=plan["fixture"],
            evaluator_sha256=plan["evaluator_sha256"],
            expected_owner_identity=_plan_owner_identity(plan),
        )
        for arm in ("R0", "A", "B", "C", "D")
    }
    _validate_zero_lora(d1_evidence["zero_lora"], plan=plan)
    r0_step = plan["cells"]["d1_core"]["R0"]["planned_steps"]
    if r0_step != R0_STEPS or R0_STEPS not in curves["R0"]:
        raise HKEContractError("R0 exact 1166 terminal is absent")
    return plan, bridge, curves


def _best_d1_checkpoint(
    plan: Mapping[str, Any], curves: Mapping[str, Mapping[int, Mapping[str, Any]]]
) -> tuple[str, int, dict[str, Any], dict[str, Any]]:
    r0_rows = curves["R0"][R0_STEPS]["rows"]
    candidates: list[tuple[float, str, int, dict[str, Any]]] = []
    for arm in ARMS:
        for step, record in curves[arm].items():
            mean = _component_means(record["rows"])["composite"]
            candidates.append((mean, arm, step, record))
    _, selected_arm, selected_step, selected = min(
        candidates, key=lambda row: (row[0], row[1], -row[2])
    )
    comparison = compare_heldout_rows(r0_rows, selected["rows"])
    return selected_arm, selected_step, selected, comparison


def _terminal_factorial_effects(
    curves: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Report the predeclared terminal 2x2 effects without selecting on them."""

    means = {
        arm: _component_means(curves[arm][FACTORIAL_STEPS]["rows"])["composite"]
        for arm in ("A", "B", "C", "D")
    }
    mae = (means["A"] + means["C"]) / 2.0
    mse = (means["B"] + means["D"]) / 2.0
    no_multires = (means["A"] + means["B"]) / 2.0
    multires = (means["C"] + means["D"]) / 2.0
    return {
        "checkpoint_step": FACTORIAL_STEPS,
        "lower_is_better": True,
        "cell_composite_losses": means,
        "mse_minus_mae": mse - mae,
        "multires_on_minus_off": multires - no_multires,
        "loss_by_multires_interaction": (means["D"] - means["C"])
        - (means["B"] - means["A"]),
        "interpretation": {
            "positive_mse_minus_mae_favors_mae": True,
            "negative_multires_on_minus_off_favors_multires": True,
        },
    }


def _validate_comparison_summary(value: Any, label: str) -> dict[str, Any]:
    fields = {
        "rows",
        "incumbent_means",
        "candidate_means",
        "relative_improvement",
        "paired_composite_delta",
        "paired_ci95",
        "paired_relative_ci95",
        "direction",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise HKEContractError(f"{label} comparison is malformed")
    if (
        isinstance(value["rows"], bool)
        or not isinstance(value["rows"], int)
        or value["rows"] < 2
    ):
        raise HKEContractError(f"{label} comparison row count is invalid")
    for key in ("incumbent_means", "candidate_means", "relative_improvement"):
        record = value[key]
        if not isinstance(record, Mapping) or set(record) != {
            "prompted",
            "blank",
            "composite",
        }:
            raise HKEContractError(f"{label} comparison components are malformed")
        for number in record.values():
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
            ):
                raise HKEContractError(
                    f"{label} comparison contains a non-finite value"
                )
    for key in ("paired_ci95", "paired_relative_ci95"):
        interval = value[key]
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
                for number in interval
            )
            or float(interval[0]) > float(interval[1])
        ):
            raise HKEContractError(f"{label} comparison interval is malformed")
    delta = value["paired_composite_delta"]
    if (
        isinstance(delta, bool)
        or not isinstance(delta, (int, float))
        or not math.isfinite(float(delta))
    ):
        raise HKEContractError(f"{label} comparison delta is invalid")
    expected_direction = (
        "improves" if delta > 0 else "regresses" if delta < 0 else "tie"
    )
    if value["direction"] != expected_direction:
        raise HKEContractError(f"{label} comparison direction mismatch")
    incumbent = value["incumbent_means"]
    candidate = value["candidate_means"]
    for means_label, means in (("incumbent", incumbent), ("candidate", candidate)):
        expected_composite = 0.25 * means["prompted"] + 0.75 * means["blank"]
        if not math.isclose(
            means["composite"], expected_composite, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise HKEContractError(f"{label} {means_label} composite is inconsistent")
    for component in ("prompted", "blank", "composite"):
        baseline = incumbent[component]
        if baseline <= 0 or not math.isclose(
            value["relative_improvement"][component],
            (baseline - candidate[component]) / baseline,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise HKEContractError(f"{label} relative improvement is inconsistent")
    if not math.isclose(
        delta,
        incumbent["composite"] - candidate["composite"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise HKEContractError(f"{label} paired delta is inconsistent")
    expected_relative_interval = [
        bound / incumbent["composite"] for bound in value["paired_ci95"]
    ]
    if any(
        not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        for actual, expected in zip(
            value["paired_relative_ci95"], expected_relative_interval, strict=True
        )
    ):
        raise HKEContractError(f"{label} relative interval is inconsistent")
    return copy.deepcopy(dict(value))


def _clock_fill_eligibility(
    plan: Mapping[str, Any], *, selected_arm: str, comparison: Mapping[str, Any]
) -> tuple[bool, int, BoundTimingProfile]:
    comparison = _validate_comparison_summary(comparison, "D1")
    loss = ARMS[selected_arm]["loss"]
    timing_source = materialize_current_law_configs(
        yaml.safe_load(INCUMBENT_TEMPLATE_PATH.read_text(encoding="utf-8")),
        num_images=plan["fixture"]["training_row_count"],
        hours_to_complete=float(plan["hours_to_complete"]),
    )["C" if loss == "mae" else "D"]
    profile = _validate_bound_profile_document(
        plan["timing_profiles"][loss],
        loss=loss,
        expected_dataset_size=plan["fixture"]["training_row_count"],
        expected_config=timing_source,
        expected_bundle_id=OWNED_MULTIRES_BUNDLE,
        expected_runtime_commit=OWNED_RUNTIME_COMMIT,
    )
    steps = measured_clock_fill_steps(
        hours_to_complete=float(plan["hours_to_complete"]), profile=profile.profile
    )
    rel = comparison["relative_improvement"]
    eligible = (
        steps > FACTORIAL_STEPS
        and rel["composite"] >= MIN_COMPOSITE_IMPROVEMENT
        and rel["prompted"] > -MAX_RELATIVE_REGRESSION
        and rel["blank"] > -MAX_RELATIVE_REGRESSION
    )
    return eligible, steps, profile


def build_optional_e_plan(
    plan_value: Mapping[str, Any],
    d1_evidence: Mapping[str, Any],
    *,
    bridge_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize E only after the complete D1 core makes it eligible."""

    plan, bridge, curves = _validated_d1_core(
        plan_value, d1_evidence, bridge_evidence=bridge_evidence
    )
    selected_arm, _step, _record, comparison = _best_d1_checkpoint(plan, curves)
    eligible, clock_steps, profile = _clock_fill_eligibility(
        plan, selected_arm=selected_arm, comparison=comparison
    )
    if not eligible:
        raise HKEContractError("optional E is not eligible under the predeclared rule")
    source_cell = plan["cells"]["d1_core"][selected_arm]
    config = copy.deepcopy(source_cell["config"])
    _train_node(config)["steps"] = clock_steps
    _save_node(config)["save_every"] = recipe.kill_safe_save_every(
        clock_steps, int(_save_node(source_cell["config"])["save_every"])
    )
    observation = _validate_h100_observation(
        profile.accelerator_observation,
        "optional E execution",
        require_fixed_capture=True,
    )
    cell = _plan_cell(
        family="social",
        pack="D1",
        phase="discovery",
        arm="E",
        seed=TRAINING_SEED_A,
        config=config,
        bundle_id=source_cell["bundle_id"],
        fixture=plan["fixture"],
        candidate_semantic_sha256=plan["fixture"]["candidate_semantic_sha256"],
        fixture_admission_sha256=plan["fixture"]["admission_sha256"],
        execution={
            key: source_cell["cell_identity"][key]
            for key in (
                "code_tree",
                "runtime_tree",
                "container_digest",
                "python_executable",
            )
        },
        observation=observation,
        timing_support={
            "execution_timing_mode": "bootstrap_probe_unmeasured",
            "profile_sha256": None,
            "binding_sha256": None,
            "conservative_source_profile_sha256": profile.profile.profile_sha256,
            "conservative_source_binding_sha256": profile.binding_sha256,
            "source_arm": selected_arm,
            "d1_core_evidence_sha256": d1_evidence["evidence_sha256"],
        },
    )
    body = {
        "schema": SCHEMA,
        "kind": "sn56-week7-hke-optional-e-plan",
        "status": "OPTIONAL_E_READY_NO_OTHER_AUTHORITY",
        "source_prelaunch_plan_sha256": plan["plan_sha256"],
        "owner_identity": _plan_owner_identity(plan),
        "source_d1_evidence": copy.deepcopy(dict(d1_evidence)),
        "d1_core_evidence_sha256": d1_evidence["evidence_sha256"],
        "source_bridge_evidence": copy.deepcopy(dict(bridge_evidence)),
        "bridge_receipt_sha256": bridge["bridge_receipt_sha256"],
        "source_arm": selected_arm,
        "source_comparison": comparison,
        "clock_fill_steps": clock_steps,
        "fixture": copy.deepcopy(plan["fixture"]),
        "evaluator": copy.deepcopy(plan["evaluator"]),
        "evaluator_sha256": plan["evaluator_sha256"],
        "cell": cell,
        "authorization": {
            "mechanical_plan_ready": True,
            "separate_owner_gpu_order_required": True,
            "gpu_execution_authorized": False,
            "optional_e_execution_authorized": False,
            "checkpoint_promotion_authorized": False,
            "deployment_authorized": False,
        },
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def _validate_optional_e_plan(
    value: Mapping[str, Any], *, prelaunch: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HKEContractError("optional E plan is unavailable")
    fields = {
        "schema",
        "kind",
        "status",
        "source_prelaunch_plan_sha256",
        "owner_identity",
        "source_d1_evidence",
        "d1_core_evidence_sha256",
        "source_bridge_evidence",
        "bridge_receipt_sha256",
        "source_arm",
        "source_comparison",
        "clock_fill_steps",
        "fixture",
        "evaluator",
        "evaluator_sha256",
        "cell",
        "authorization",
        "plan_sha256",
    }
    if set(value) != fields:
        raise HKEContractError("optional E plan envelope is malformed")
    body = dict(value)
    declared = body.pop("plan_sha256", None)
    if declared != canonical_sha256(body):
        raise HKEContractError("optional E plan digest mismatch")
    if (
        value.get("schema") != SCHEMA
        or value.get("kind") != "sn56-week7-hke-optional-e-plan"
        or value.get("status") != "OPTIONAL_E_READY_NO_OTHER_AUTHORITY"
        or value.get("source_prelaunch_plan_sha256") != prelaunch["plan_sha256"]
        or value.get("owner_identity") != _plan_owner_identity(prelaunch)
        or value.get("fixture") != prelaunch["fixture"]
        or value.get("evaluator_sha256") != prelaunch["evaluator_sha256"]
        or value.get("cell", {}).get("cell_identity", {}).get("arm") != "E"
        or value.get("cell", {}).get("planned_steps") <= FACTORIAL_STEPS
        or value.get("authorization")
        != {
            "mechanical_plan_ready": True,
            "separate_owner_gpu_order_required": True,
            "gpu_execution_authorized": False,
            "optional_e_execution_authorized": False,
            "checkpoint_promotion_authorized": False,
            "deployment_authorized": False,
        }
    ):
        raise HKEContractError("optional E plan binding mismatch")
    _validate_comparison_summary(value.get("source_comparison"), "optional E source")
    expected = build_optional_e_plan(
        prelaunch,
        value["source_d1_evidence"],
        bridge_evidence=value["source_bridge_evidence"],
    )
    if canonical_bytes(expected) != canonical_bytes(value):
        raise HKEContractError("optional E plan does not reproduce from D1 evidence")
    return dict(value)


def freeze_d1_candidate(
    plan_value: Mapping[str, Any],
    d1_evidence: Mapping[str, Any],
    *,
    bridge_evidence: Mapping[str, Any],
    frozen_at_utc: str,
    optional_e_plan: Mapping[str, Any] | None = None,
    optional_e_curve: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze one recipe/checkpoint after all conditionally required D1 cells."""

    _validate_utc_timestamp(frozen_at_utc, "candidate freeze")
    plan, bridge, curves = _validated_d1_core(
        plan_value, d1_evidence, bridge_evidence=bridge_evidence
    )
    selected_arm, selected_step, selected, comparison = _best_d1_checkpoint(
        plan, curves
    )
    eligible, _clock_steps, _profile = _clock_fill_eligibility(
        plan, selected_arm=selected_arm, comparison=comparison
    )
    selected_cell = plan["cells"]["d1_core"][selected_arm]
    e_plan_sha256: str | None = None
    e_curve_sha256: str | None = None
    if eligible:
        if optional_e_plan is None or optional_e_curve is None:
            raise HKEContractError(
                "eligible optional E evidence is required before freeze"
            )
        e_plan = _validate_optional_e_plan(optional_e_plan, prelaunch=plan)
        e_curves = validate_score_curve(
            optional_e_curve,
            plan_sha256=e_plan["plan_sha256"],
            plan_cell=e_plan["cell"],
            fixture=e_plan["fixture"],
            evaluator_sha256=e_plan["evaluator_sha256"],
            expected_owner_identity=_plan_owner_identity(e_plan),
        )
        for step, record in e_curves.items():
            if (
                _component_means(record["rows"])["composite"]
                < _component_means(selected["rows"])["composite"]
            ):
                selected_arm, selected_step, selected = "E", step, record
                selected_cell = e_plan["cell"]
                comparison = compare_heldout_rows(
                    curves["R0"][R0_STEPS]["rows"], selected["rows"]
                )
        e_plan_sha256 = e_plan["plan_sha256"]
        e_curve_sha256 = optional_e_curve["curve_sha256"]
    elif optional_e_plan is not None or optional_e_curve is not None:
        raise HKEContractError("optional E was supplied when the gate was not eligible")
    frozen_config = copy.deepcopy(selected_cell["config"])
    _train_node(frozen_config)["steps"] = selected_step
    _save_node(frozen_config)["save_every"] = recipe.kill_safe_save_every(
        selected_step, int(_save_node(selected_cell["config"])["save_every"])
    )
    candidate = {
        "arm": selected_arm,
        "source_plan_sha256": (
            e_plan_sha256 if selected_arm == "E" else plan["plan_sha256"]
        ),
        "source_cell_sha256": selected_cell["cell_sha256"],
        "bundle_id": selected_cell["bundle_id"],
        "bundle_sha256": selected_cell["bundle_sha256"],
        "runtime_commit": selected_cell["runtime_commit"],
        "generated_config": frozen_config,
        "generated_config_sha256": canonical_sha256(frozen_config),
        "checkpoint_step": selected_step,
        "artifact_sha256": selected["artifact_sha256"],
        "seed": TRAINING_SEED_A,
    }
    body = {
        "schema": SCHEMA,
        "kind": "sn56-week7-hke-frozen-candidate",
        "status": "FROZEN_D1_ONLY",
        "source_plan_sha256": plan["plan_sha256"],
        "d1_evidence_sha256": d1_evidence["evidence_sha256"],
        "bridge_receipt_sha256": bridge["bridge_receipt_sha256"],
        "candidate": candidate,
        "d1_comparison": comparison,
        "terminal_factorial_effects": _terminal_factorial_effects(curves),
        "optional_e": {
            "eligible": eligible,
            "plan_sha256": e_plan_sha256,
            "curve_sha256": e_curve_sha256,
        },
        "confirmation_commitments": copy.deepcopy(
            plan["sealed_confirmation_commitments"]
        ),
        "frozen_at_utc": frozen_at_utc,
    }
    _validate_utc_timestamp(frozen_at_utc, "candidate freeze")
    return {**body, "frozen_candidate_sha256": canonical_sha256(body)}


def _validate_frozen_candidate(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    optional_e_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema",
        "kind",
        "status",
        "source_plan_sha256",
        "d1_evidence_sha256",
        "bridge_receipt_sha256",
        "candidate",
        "d1_comparison",
        "terminal_factorial_effects",
        "optional_e",
        "confirmation_commitments",
        "frozen_at_utc",
        "frozen_candidate_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise HKEContractError("D1 frozen candidate is malformed")
    body = dict(value)
    declared = body.pop("frozen_candidate_sha256")
    if declared != canonical_sha256(body):
        raise HKEContractError("D1 frozen candidate digest mismatch")
    if (
        value["schema"] != SCHEMA
        or value["kind"] != "sn56-week7-hke-frozen-candidate"
        or value["status"] != "FROZEN_D1_ONLY"
        or value["source_plan_sha256"] != plan["plan_sha256"]
        or value["confirmation_commitments"] != plan["sealed_confirmation_commitments"]
    ):
        raise HKEContractError("D1 frozen candidate binding mismatch")
    _validate_utc_timestamp(value["frozen_at_utc"], "candidate freeze")
    _validate_comparison_summary(value.get("d1_comparison"), "D1 frozen")
    candidate = value["candidate"]
    candidate_fields = {
        "arm",
        "source_plan_sha256",
        "source_cell_sha256",
        "bundle_id",
        "bundle_sha256",
        "runtime_commit",
        "generated_config",
        "generated_config_sha256",
        "checkpoint_step",
        "artifact_sha256",
        "seed",
    }
    if not isinstance(candidate, Mapping) or set(candidate) != candidate_fields:
        raise HKEContractError("D1 frozen candidate identity is malformed")
    arm = candidate["arm"]
    if arm not in {*ARMS, "E"}:
        raise HKEContractError("D1 frozen candidate arm is invalid")
    if arm == "E":
        source_plan = _validate_optional_e_plan(optional_e_plan, prelaunch=plan)
        source_cell = source_plan["cell"]
    else:
        source_plan = plan
        source_cell = plan["cells"]["d1_core"][arm]
    expected_config = copy.deepcopy(source_cell["config"])
    _train_node(expected_config)["steps"] = candidate["checkpoint_step"]
    _save_node(expected_config)["save_every"] = recipe.kill_safe_save_every(
        candidate["checkpoint_step"],
        int(_save_node(source_cell["config"])["save_every"]),
    )
    if (
        candidate["source_plan_sha256"] != source_plan["plan_sha256"]
        or candidate["source_cell_sha256"] != source_cell["cell_sha256"]
        or candidate["bundle_id"] != source_cell["bundle_id"]
        or candidate["bundle_sha256"] != source_cell["bundle_sha256"]
        or candidate["runtime_commit"] != source_cell["runtime_commit"]
        or candidate["generated_config"] != expected_config
        or candidate["generated_config_sha256"] != canonical_sha256(expected_config)
        or candidate["checkpoint_step"] not in source_cell["required_checkpoint_steps"]
        or candidate["seed"] != TRAINING_SEED_A
    ):
        raise HKEContractError("D1 frozen candidate is not a planned checkpoint")
    _require_sha256(candidate["artifact_sha256"], "D1 frozen artifact")
    return dict(value)


def _public_discovery_fixture(
    prelaunch: Mapping[str, Any], *, family: str, pack: str
) -> dict[str, Any]:
    checked_set = _validate_admission_set(prelaunch["admission_set"])
    receipt = checked_set["receipts"][family]
    record = receipt["packs"][pack]
    spec = EXPECTED_PACKS[family][pack]
    if record["phase"] != "discovery":
        raise HKEContractError("public fixture is not a discovery pack")
    training = _validate_inventory_body(
        record["training_inventory"],
        f"{family}/{pack} training",
        expected_pairs=spec["train"],
    )
    evaluation = _validate_inventory_body(
        record["evaluation_inventory"],
        f"{family}/{pack} evaluation",
        expected_pairs=spec["eval"],
    )
    return {
        "family": family,
        "pack": pack,
        "phase": "discovery",
        "row_count": spec["count"],
        "training_row_count": spec["train"],
        "evaluation_row_count": spec["eval"],
        "admission_sha256": receipt["admission_sha256"],
        "candidate_semantic_sha256": receipt["candidate_semantic_sha256"],
        "training_row_identity_sha256": record["training_row_identity_sha256"],
        "evaluation_row_identity_sha256": record["evaluation_row_identity_sha256"],
        "training_inventory": training,
        "training_inventory_sha256": training["semantic_sha256"],
        "evaluation_inventory": evaluation,
        "evaluation_inventory_sha256": evaluation["semantic_sha256"],
    }


def build_d2_replication_plan(
    prelaunch_value: Mapping[str, Any],
    frozen_candidate: Mapping[str, Any],
    *,
    d1_evidence: Mapping[str, Any],
    bridge_evidence: Mapping[str, Any],
    optional_e_plan: Mapping[str, Any] | None = None,
    optional_e_curve: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize D2 only after reproducing the bridge-bound D1 freeze."""

    prelaunch = _validate_plan(prelaunch_value)
    reproduced = freeze_d1_candidate(
        prelaunch,
        d1_evidence,
        bridge_evidence=bridge_evidence,
        frozen_at_utc=str(frozen_candidate.get("frozen_at_utc", "")),
        optional_e_plan=optional_e_plan,
        optional_e_curve=optional_e_curve,
    )
    if dict(frozen_candidate) != reproduced:
        raise HKEContractError("D1 frozen candidate does not reproduce")
    frozen = _validate_frozen_candidate(
        reproduced, plan=prelaunch, optional_e_plan=optional_e_plan
    )
    fixture = _public_discovery_fixture(prelaunch, family="social", pack="D2")
    observation = _validate_h100_observation(
        next(iter(prelaunch["timing_profiles"].values()))["binding"][
            "accelerator_observation"
        ],
        "D2 execution",
        require_fixed_capture=True,
    )
    selected_source = frozen["candidate"]["generated_config"]
    selected_step = int(frozen["candidate"]["checkpoint_step"])
    cells: dict[str, Any] = {}
    for seed_label, seed in (
        ("Seed-A", TRAINING_SEED_A),
        ("Seed-B", TRAINING_SEED_B),
    ):
        incumbent = materialize_r0(
            yaml.safe_load(INCUMBENT_TEMPLATE_PATH.read_text(encoding="utf-8")),
            seed=seed,
        )
        candidate = copy.deepcopy(selected_source)
        _process_node(candidate)["training_seed"] = seed
        _train_node(candidate)["steps"] = selected_step
        _save_node(candidate)["save_every"] = recipe.kill_safe_save_every(
            selected_step, int(_save_node(selected_source)["save_every"])
        )
        for role, config, bundle, execution_key in (
            ("incumbent", incumbent, INCUMBENT_BUNDLE, "incumbent"),
            (
                "candidate",
                candidate,
                str(frozen["candidate"]["bundle_id"]),
                "owned",
            ),
        ):
            key = f"{role}-{seed_label}"
            cells[key] = _plan_cell(
                family="social",
                pack="D2",
                phase="replication",
                arm=key,
                seed=seed,
                config=config,
                bundle_id=bundle,
                fixture=fixture,
                candidate_semantic_sha256=fixture["candidate_semantic_sha256"],
                fixture_admission_sha256=fixture["admission_sha256"],
                execution=prelaunch["execution_identities"][execution_key],
                observation=observation,
                timing_support=(
                    None
                    if role == "incumbent"
                    else {"frozen_candidate_sha256": frozen["frozen_candidate_sha256"]}
                ),
            )
    body = {
        "schema": SCHEMA,
        "kind": "sn56-week7-hke-d2-replication-plan",
        "status": "D2_READY_CONFIRMATION_SEALED",
        "source_prelaunch_plan": copy.deepcopy(prelaunch),
        "source_prelaunch_plan_sha256": prelaunch["plan_sha256"],
        "source_d1_evidence": copy.deepcopy(dict(d1_evidence)),
        "source_bridge_evidence": copy.deepcopy(dict(bridge_evidence)),
        "source_optional_e_plan": (
            None if optional_e_plan is None else copy.deepcopy(dict(optional_e_plan))
        ),
        "source_optional_e_curve": (
            None if optional_e_curve is None else copy.deepcopy(dict(optional_e_curve))
        ),
        "frozen_candidate": frozen,
        "frozen_candidate_sha256": frozen["frozen_candidate_sha256"],
        "fixture": fixture,
        "evaluator": prelaunch["evaluator"],
        "evaluator_sha256": prelaunch["evaluator_sha256"],
        "cells": cells,
        "operator_procedure_order": [
            "incumbent-Seed-A",
            "candidate-Seed-B",
            "candidate-Seed-A",
            "incumbent-Seed-B",
        ],
        "order_evidence_class": "operator_procedure_not_machine_verified",
        "confirmation_commitments": copy.deepcopy(frozen["confirmation_commitments"]),
        "accelerator_observation": observation,
        "execution_identities": copy.deepcopy(prelaunch["execution_identities"]),
        "authorization": {
            "mechanical_plan_ready": True,
            "separate_owner_gpu_order_required": True,
            "gpu_execution_authorized": False,
            "d2_execution_authorized": False,
            "confirmation_reveal_authorized": False,
            "checkpoint_promotion_authorized": False,
            "deployment_authorized": False,
        },
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def _validate_d2_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HKEContractError("D2 plan is unavailable")
    fields = {
        "schema",
        "kind",
        "status",
        "source_prelaunch_plan",
        "source_prelaunch_plan_sha256",
        "source_d1_evidence",
        "source_bridge_evidence",
        "source_optional_e_plan",
        "source_optional_e_curve",
        "frozen_candidate",
        "frozen_candidate_sha256",
        "fixture",
        "evaluator",
        "evaluator_sha256",
        "cells",
        "operator_procedure_order",
        "order_evidence_class",
        "confirmation_commitments",
        "accelerator_observation",
        "execution_identities",
        "authorization",
        "plan_sha256",
    }
    if set(value) != fields:
        raise HKEContractError("D2 plan envelope is malformed")
    body = dict(value)
    declared = body.pop("plan_sha256", None)
    if declared != canonical_sha256(body):
        raise HKEContractError("D2 plan digest mismatch")
    if (
        value.get("schema") != SCHEMA
        or value.get("kind") != "sn56-week7-hke-d2-replication-plan"
        or value.get("status") != "D2_READY_CONFIRMATION_SEALED"
        or not isinstance(value.get("cells"), Mapping)
        or set(value["cells"])
        != {
            "incumbent-Seed-A",
            "candidate-Seed-A",
            "incumbent-Seed-B",
            "candidate-Seed-B",
        }
        or value.get("confirmation_commitments")
        != value.get("frozen_candidate", {}).get("confirmation_commitments")
        or not isinstance(value.get("accelerator_observation"), Mapping)
        or not isinstance(value.get("execution_identities"), Mapping)
        or value.get("authorization")
        != {
            "mechanical_plan_ready": True,
            "separate_owner_gpu_order_required": True,
            "gpu_execution_authorized": False,
            "d2_execution_authorized": False,
            "confirmation_reveal_authorized": False,
            "checkpoint_promotion_authorized": False,
            "deployment_authorized": False,
        }
    ):
        raise HKEContractError("D2 plan contract mismatch")
    expected = build_d2_replication_plan(
        value["source_prelaunch_plan"],
        value["frozen_candidate"],
        d1_evidence=value["source_d1_evidence"],
        bridge_evidence=value["source_bridge_evidence"],
        optional_e_plan=value["source_optional_e_plan"],
        optional_e_curve=value["source_optional_e_curve"],
    )
    if canonical_bytes(expected) != canonical_bytes(value):
        raise HKEContractError("D2 plan does not reproduce from its source chain")
    return dict(value)


def _discovery_advance_decision(
    d1_comparison: Mapping[str, Any],
    d2_comparisons: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    d1 = _validate_comparison_summary(d1_comparison, "D1 discovery")
    if not isinstance(d2_comparisons, Mapping) or set(d2_comparisons) != {
        "Seed-A",
        "Seed-B",
    }:
        raise HKEContractError("both D2 comparisons are required")
    d2 = {
        seed: _validate_comparison_summary(value, f"D2 {seed}")
        for seed, value in d2_comparisons.items()
    }
    d2_composite = sum(
        value["relative_improvement"]["composite"] for value in d2.values()
    ) / len(d2)
    all_discovery = [d1, *d2.values()]
    gates = {
        "d1_composite_at_least_3pct": d1["relative_improvement"]["composite"]
        >= MIN_COMPOSITE_IMPROVEMENT,
        "d2_mean_composite_at_least_3pct": d2_composite >= MIN_COMPOSITE_IMPROVEMENT,
        "same_direction_d1_d2": all(
            value["direction"] == "improves" for value in all_discovery
        ),
        "one_discovery_ci_clears_zero": any(
            value["paired_ci95"][0] > 0 for value in all_discovery
        ),
        "prompted_and_blank_each_regress_less_than_1pct": all(
            value["relative_improvement"][component] > -MAX_RELATIVE_REGRESSION
            for value in all_discovery
            for component in ("prompted", "blank")
        ),
        "both_finalist_seeds_agree": all(
            value["relative_improvement"]["composite"] > 0 for value in d2.values()
        ),
    }
    advance = all(gates.values())
    body = {
        "schema": SCHEMA,
        "kind": "sn56-week7-hke-discovery-advance-decision",
        "status": "OPERATOR_ATTESTED_PROVENANCE_BOUND",
        "d1_comparison": d1,
        "d2_comparisons": d2,
        "d2_mean_composite_improvement": d2_composite,
        "gates": gates,
        "decision": "ADVANCE" if advance else "HOLD",
        "confirmation_reveal_authorized": advance,
        "checkpoint_promotion_authorized": False,
        "deployment_authorized": False,
    }
    return {**body, "decision_sha256": canonical_sha256(body)}


def freeze_confirmation_candidate(
    d2_plan_value: Mapping[str, Any],
    d2_evidence: Mapping[str, Any],
    *,
    frozen_at_utc: str,
) -> dict[str, Any]:
    """Freeze after two-seed D2 replication, before any C1 identity exists."""

    _validate_utc_timestamp(frozen_at_utc, "confirmation freeze")
    plan = _validate_d2_plan(d2_plan_value)
    fields = {
        "schema",
        "kind",
        "status",
        "plan_sha256",
        "curves",
        "evidence_sha256",
    }
    if not isinstance(d2_evidence, Mapping) or set(d2_evidence) != fields:
        raise HKEContractError("D2 evidence is incomplete")
    body = dict(d2_evidence)
    declared = body.pop("evidence_sha256")
    if declared != canonical_sha256(body):
        raise HKEContractError("D2 evidence digest mismatch")
    if (
        d2_evidence["schema"] != SCHEMA
        or d2_evidence["kind"] != "sn56-week7-hke-d2-score-evidence"
        or d2_evidence["status"] != "OPERATOR_ATTESTED_PASS"
        or d2_evidence["plan_sha256"] != plan["plan_sha256"]
        or not isinstance(d2_evidence["curves"], Mapping)
        or set(d2_evidence["curves"]) != set(plan["cells"])
    ):
        raise HKEContractError("D2 evidence authority binding mismatch")
    curves = {
        key: validate_score_curve(
            d2_evidence["curves"][key],
            plan_sha256=plan["plan_sha256"],
            plan_cell=plan["cells"][key],
            fixture=plan["fixture"],
            evaluator_sha256=plan["evaluator_sha256"],
            expected_owner_identity=_plan_owner_identity(plan),
        )
        for key in plan["cells"]
    }
    comparisons: dict[str, Any] = {}
    for seed_label in ("Seed-A", "Seed-B"):
        incumbent_key = f"incumbent-{seed_label}"
        candidate_key = f"candidate-{seed_label}"
        incumbent_step = plan["cells"][incumbent_key]["planned_steps"]
        candidate_step = plan["cells"][candidate_key]["planned_steps"]
        comparisons[seed_label] = compare_heldout_rows(
            curves[incumbent_key][incumbent_step]["rows"],
            curves[candidate_key][candidate_step]["rows"],
        )
    discovery = _discovery_advance_decision(
        plan["frozen_candidate"]["d1_comparison"], comparisons
    )
    seed_agreement = discovery["gates"]["both_finalist_seeds_agree"]
    advance = discovery["decision"] == "ADVANCE"
    record = {
        "schema": SCHEMA,
        "kind": "sn56-week7-hke-confirmation-candidate-freeze",
        "status": (
            "FROZEN_BEFORE_CONFIRMATION_REVEAL"
            if advance
            else "HOLD_NO_CONFIRMATION_REVEAL"
        ),
        "source_d2_plan": copy.deepcopy(plan),
        "source_d2_plan_sha256": plan["plan_sha256"],
        "source_d2_evidence": copy.deepcopy(dict(d2_evidence)),
        "d1_frozen_candidate_sha256": plan["frozen_candidate_sha256"],
        "d2_evidence_sha256": d2_evidence["evidence_sha256"],
        "candidate": copy.deepcopy(plan["frozen_candidate"]["candidate"]),
        "discovery_decision": discovery,
        "d2_comparisons": comparisons,
        "both_finalist_seeds_agree": seed_agreement,
        "confirmation_commitments": copy.deepcopy(plan["confirmation_commitments"]),
        "confirmation_reveal_authorized": advance,
        "frozen_at_utc": frozen_at_utc,
        "checkpoint_promotion_authorized": False,
        "deployment_authorized": False,
    }
    _validate_utc_timestamp(frozen_at_utc, "confirmation freeze")
    return {
        **record,
        "confirmation_freeze_sha256": canonical_sha256(record),
    }


def _validate_confirmation_freeze(
    value: Mapping[str, Any], *, d2_plan: Mapping[str, Any]
) -> dict[str, Any]:
    fields = {
        "schema",
        "kind",
        "status",
        "source_d2_plan",
        "source_d2_plan_sha256",
        "source_d2_evidence",
        "d1_frozen_candidate_sha256",
        "d2_evidence_sha256",
        "candidate",
        "discovery_decision",
        "d2_comparisons",
        "both_finalist_seeds_agree",
        "confirmation_commitments",
        "confirmation_reveal_authorized",
        "frozen_at_utc",
        "checkpoint_promotion_authorized",
        "deployment_authorized",
        "confirmation_freeze_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise HKEContractError("confirmation freeze is malformed")
    body = dict(value)
    declared = body.pop("confirmation_freeze_sha256")
    if declared != canonical_sha256(body):
        raise HKEContractError("confirmation freeze digest mismatch")
    embedded_d2 = _validate_d2_plan(value["source_d2_plan"])
    if canonical_bytes(embedded_d2) != canonical_bytes(d2_plan):
        raise HKEContractError("confirmation freeze embeds a foreign D2 plan")
    expected_freeze = freeze_confirmation_candidate(
        embedded_d2,
        value["source_d2_evidence"],
        frozen_at_utc=str(value["frozen_at_utc"]),
    )
    if canonical_bytes(expected_freeze) != canonical_bytes(value):
        raise HKEContractError(
            "confirmation freeze does not reproduce from D2 evidence"
        )
    decision = value["discovery_decision"]
    expected_decision = _discovery_advance_decision(
        d2_plan["frozen_candidate"]["d1_comparison"],
        value["d2_comparisons"],
    )
    if (
        value["schema"] != SCHEMA
        or value["kind"] != "sn56-week7-hke-confirmation-candidate-freeze"
        or value["status"] != "FROZEN_BEFORE_CONFIRMATION_REVEAL"
        or value["source_d2_plan_sha256"] != d2_plan["plan_sha256"]
        or value["d1_frozen_candidate_sha256"] != d2_plan["frozen_candidate_sha256"]
        or value["candidate"] != d2_plan["frozen_candidate"]["candidate"]
        or decision != expected_decision
        or decision["decision"] != "ADVANCE"
        or value["confirmation_commitments"] != d2_plan["confirmation_commitments"]
        or value["confirmation_reveal_authorized"] is not True
        or value["both_finalist_seeds_agree"] is not True
        or value["checkpoint_promotion_authorized"] is not False
        or value["deployment_authorized"] is not False
    ):
        raise HKEContractError("confirmation freeze authority binding mismatch")
    _validate_utc_timestamp(value["frozen_at_utc"], "confirmation freeze")
    return dict(value)


def _validate_confirmation_reveal_record(
    value: Mapping[str, Any],
    *,
    family: str,
    pack: str,
    confirmation_freeze: Mapping[str, Any],
    admission_set: Mapping[str, Any],
    expected_authority_kind: str = "sn56-week7-hke-confirmation-candidate-freeze",
    expected_authority_sha256: str | None = None,
) -> dict[str, Any]:
    fields = {
        "schema",
        "kind",
        "status",
        "privacy",
        "family",
        "pack",
        "candidate_semantic_sha256",
        "admission_set_sha256",
        "family_admission_sha256",
        "confirmation_commitment_sha256",
        "confirmation_authority_kind",
        "confirmation_authority_sha256",
        "revealed_rows",
        "training_row_identity_sha256",
        "evaluation_row_identity_sha256",
        "training_inventory",
        "training_inventory_sha256",
        "evaluation_inventory",
        "evaluation_inventory_sha256",
        "authorization",
        "confirmation_reveal_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise HKEContractError(f"{family}/{pack} confirmation reveal is malformed")
    body = dict(value)
    declared = body.pop("confirmation_reveal_sha256")
    if declared != canonical_sha256(body):
        raise HKEContractError(f"{family}/{pack} confirmation reveal digest mismatch")
    checked_set = _validate_admission_set(admission_set)
    receipt = checked_set["receipts"][family]
    receipt_pack = receipt["packs"][pack]
    commitment = confirmation_freeze["confirmation_commitments"][family][pack]
    authority_sha = (
        confirmation_freeze["confirmation_freeze_sha256"]
        if expected_authority_sha256 is None
        else expected_authority_sha256
    )
    expected_status = (
        "PRIVATE_C1_REVEALED_AFTER_D2_CONFIRMATION_FREEZE"
        if pack == "C1"
        else "PRIVATE_C2_REVEALED_AFTER_BORDERLINE_C1_TRIGGER"
    )
    if (
        value["schema"] != SCHEMA
        or value["kind"] != "sn56-week7-hke-private-confirmation-reveal"
        or value["status"] != expected_status
        or value["privacy"] != "custodian_private_not_public_admission"
        or value["family"] != family
        or value["pack"] != pack
        or value["candidate_semantic_sha256"]
        != checked_set["candidate_semantic_sha256"]
        or value["admission_set_sha256"] != checked_set["admission_set_sha256"]
        or value["family_admission_sha256"] != receipt["admission_sha256"]
        or value["confirmation_commitment_sha256"] != commitment
        or receipt_pack.get("semantic_commitment_sha256") != commitment
        or value["confirmation_authority_kind"] != expected_authority_kind
        or value["confirmation_authority_sha256"] != authority_sha
        or value["authorization"]
        != {
            "confirmation_revealed": True,
            "gpu_execution_authorized": False,
            "deployment_authorized": False,
            "candidate_selection_locked": True,
        }
    ):
        raise HKEContractError(f"{family}/{pack} confirmation reveal binding mismatch")
    spec = EXPECTED_PACKS[family][pack]
    rows = value["revealed_rows"]
    if (
        not isinstance(rows, list)
        or len(rows) != spec["count"]
        or renderer_semantic_sha256(rows) != commitment
    ):
        raise HKEContractError(f"{family}/{pack} revealed row commitment mismatch")
    required_row_fields = {
        "row_id",
        "fixture_id",
        "family",
        "phase",
        "pack",
        "split_role",
        "ordinal",
        "relative_image_path",
        "relative_caption_path",
        "parameters",
        "parameters_sha256",
        "seed_commitment_sha256",
        "image_sha256",
        "image_bytes",
        "decoded_pixels_sha256",
        "caption_sha256",
        "caption_bytes",
        "normalized_caption_sha256",
        "width",
        "height",
        "format",
        "mode",
        "visible_glyph_transcript",
        "group_identity",
        "group_identity_sha256",
        "rights_declaration",
        "row_record_sha256",
    }
    by_split: dict[str, list[Mapping[str, Any]]] = {
        "training": [],
        "evaluation": [],
    }
    seen_row_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required_row_fields:
            raise HKEContractError(f"{family}/{pack} revealed row schema mismatch")
        row_body = dict(row)
        declared_row_sha = row_body.pop("row_record_sha256")
        if (
            renderer_semantic_sha256(row_body) != declared_row_sha
            or row["family"] != family
            or row["phase"] != "confirmation"
            or row["pack"] != pack
            or row["split_role"] not in by_split
            or renderer_semantic_sha256(row["parameters"]) != row["parameters_sha256"]
            or renderer_semantic_sha256(row["group_identity"])
            != row["group_identity_sha256"]
        ):
            raise HKEContractError(f"{family}/{pack} revealed row binding mismatch")
        if row["row_id"] in seen_row_ids:
            raise HKEContractError(f"{family}/{pack} revealed row id is duplicated")
        seen_row_ids.add(row["row_id"])
        by_split[row["split_role"]].append(row)
    if (
        len(by_split["training"]) != spec["train"]
        or len(by_split["evaluation"]) != spec["eval"]
    ):
        raise HKEContractError(f"{family}/{pack} revealed split count mismatch")

    def revealed_inventory(split_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        files = [
            record
            for row in split_rows
            for record in (
                {
                    "path": Path(str(row["relative_image_path"])).name,
                    "bytes": row["image_bytes"],
                    "sha256": row["image_sha256"],
                },
                {
                    "path": Path(str(row["relative_caption_path"])).name,
                    "bytes": row["caption_bytes"],
                    "sha256": row["caption_sha256"],
                },
            )
        ]
        files.sort(key=lambda item: item["path"])
        body = {"files": files, "file_count": len(files)}
        return {**body, "semantic_sha256": canonical_sha256(body)}

    expected_training = revealed_inventory(by_split["training"])
    expected_evaluation = revealed_inventory(by_split["evaluation"])
    expected_training_identity = canonical_sha256(
        [
            {"row_id": row["row_id"], "row_sha256": row["row_record_sha256"]}
            for row in by_split["training"]
        ]
    )
    expected_evaluation_identity = canonical_sha256(
        [
            {"row_id": row["row_id"], "row_sha256": row["row_record_sha256"]}
            for row in by_split["evaluation"]
        ]
    )
    training = _validate_inventory_body(
        value["training_inventory"],
        f"{family}/{pack} confirmation training",
        expected_pairs=spec["train"],
    )
    evaluation = _validate_inventory_body(
        value["evaluation_inventory"],
        f"{family}/{pack} confirmation evaluation",
        expected_pairs=spec["eval"],
    )
    if (
        training != expected_training
        or evaluation != expected_evaluation
        or value["training_row_identity_sha256"] != expected_training_identity
        or value["evaluation_row_identity_sha256"] != expected_evaluation_identity
    ):
        raise HKEContractError(
            f"{family}/{pack} revealed rows disagree with inventories"
        )
    for declared_hash, actual, label in (
        (value["training_inventory_sha256"], training["semantic_sha256"], "training"),
        (
            value["evaluation_inventory_sha256"],
            evaluation["semantic_sha256"],
            "evaluation",
        ),
    ):
        if declared_hash != actual:
            raise HKEContractError(f"{family}/{pack} {label} inventory digest mismatch")
    return dict(value)


def _fixture_from_confirmation_reveal(value: Mapping[str, Any]) -> dict[str, Any]:
    family = str(value["family"])
    pack = str(value["pack"])
    spec = EXPECTED_PACKS[family][pack]
    training_rows = [
        row for row in value["revealed_rows"] if row["split_role"] == "training"
    ]
    evaluation_rows = [
        row for row in value["revealed_rows"] if row["split_role"] == "evaluation"
    ]
    return {
        "family": family,
        "pack": pack,
        "phase": "confirmation",
        "row_count": spec["count"],
        "training_row_count": spec["train"],
        "evaluation_row_count": spec["eval"],
        "admission_sha256": value["family_admission_sha256"],
        "candidate_semantic_sha256": value["candidate_semantic_sha256"],
        "training_row_identity_sha256": value["training_row_identity_sha256"],
        "evaluation_row_identity_sha256": value["evaluation_row_identity_sha256"],
        "training_row_identity": [
            {"row_id": row["row_id"], "row_sha256": row["row_record_sha256"]}
            for row in training_rows
        ],
        "evaluation_row_identity": [
            {"row_id": row["row_id"], "row_sha256": row["row_record_sha256"]}
            for row in evaluation_rows
        ],
        "training_inventory": copy.deepcopy(value["training_inventory"]),
        "training_inventory_sha256": value["training_inventory_sha256"],
        "evaluation_inventory": copy.deepcopy(value["evaluation_inventory"]),
        "evaluation_inventory_sha256": value["evaluation_inventory_sha256"],
    }


def build_confirmation_plan(
    d2_plan_value: Mapping[str, Any],
    confirmation_freeze_value: Mapping[str, Any],
    *,
    c1_reveals: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build social confirmation plus product/logo guardrails after D2 freeze."""

    d2_plan = _validate_d2_plan(d2_plan_value)
    freeze = _validate_confirmation_freeze(confirmation_freeze_value, d2_plan=d2_plan)
    if not isinstance(c1_reveals, Mapping) or set(c1_reveals) != {
        "social",
        "product",
        "logo_ui",
    }:
        raise HKEContractError("all three C1 reveal records are required")
    reveals = {
        family: _validate_confirmation_reveal_record(
            c1_reveals[family],
            family=family,
            pack="C1",
            confirmation_freeze=freeze,
            admission_set=d2_plan["source_prelaunch_plan"]["admission_set"],
        )
        for family in ("social", "product", "logo_ui")
    }
    observation = _validate_h100_observation(
        d2_plan["accelerator_observation"],
        "confirmation execution",
        require_fixed_capture=True,
    )
    candidate_source = freeze["candidate"]["generated_config"]
    cells: dict[str, dict[str, Any]] = {}
    fixtures: dict[str, dict[str, Any]] = {}
    for family in ("social", "product", "logo_ui"):
        fixture = _fixture_from_confirmation_reveal(reveals[family])
        fixtures[family] = fixture
        incumbent = materialize_r0(
            yaml.safe_load(INCUMBENT_TEMPLATE_PATH.read_text(encoding="utf-8")),
            seed=TRAINING_SEED_A,
        )
        candidate = copy.deepcopy(candidate_source)
        _process_node(candidate)["training_seed"] = TRAINING_SEED_A
        phase = "confirmation" if family == "social" else "guardrail"
        cells[family] = {}
        for role, config, bundle, execution_key in (
            ("incumbent", incumbent, INCUMBENT_BUNDLE, "incumbent"),
            (
                "candidate",
                candidate,
                freeze["candidate"]["bundle_id"],
                "owned",
            ),
        ):
            cells[family][role] = _plan_cell(
                family=family,
                pack="C1",
                phase=phase,
                arm=f"{family}-{role}",
                seed=TRAINING_SEED_A,
                config=config,
                bundle_id=bundle,
                fixture=fixture,
                candidate_semantic_sha256=fixture["candidate_semantic_sha256"],
                fixture_admission_sha256=fixture["admission_sha256"],
                execution=d2_plan["execution_identities"][execution_key],
                observation=observation,
                timing_support=(
                    None
                    if role == "incumbent"
                    else {
                        "confirmation_freeze_sha256": freeze[
                            "confirmation_freeze_sha256"
                        ]
                    }
                ),
            )
    body = {
        "schema": SCHEMA,
        "kind": "sn56-week7-hke-confirmation-plan",
        "status": "C1_AND_GUARDRAILS_READY_C2_SEALED",
        "source_d2_plan": copy.deepcopy(d2_plan),
        "source_d2_plan_sha256": d2_plan["plan_sha256"],
        "confirmation_freeze": freeze,
        "confirmation_freeze_sha256": freeze["confirmation_freeze_sha256"],
        "reveals": reveals,
        "fixtures": fixtures,
        "evaluator": copy.deepcopy(d2_plan["evaluator"]),
        "evaluator_sha256": d2_plan["evaluator_sha256"],
        "cells": cells,
        "operator_procedure_order": [
            "social-incumbent",
            "product-candidate",
            "logo_ui-incumbent",
            "social-candidate",
            "product-incumbent",
            "logo_ui-candidate",
        ],
        "order_evidence_class": "operator_procedure_not_machine_verified",
        "authorization": {
            "mechanical_plan_ready": True,
            "separate_owner_gpu_order_required": True,
            "gpu_execution_authorized": False,
            "c1_and_guardrail_execution_authorized": False,
            "c2_reveal_authorized": False,
            "checkpoint_promotion_authorized": False,
            "deployment_authorized": False,
        },
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def _validate_confirmation_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HKEContractError("confirmation plan is unavailable")
    fields = {
        "schema",
        "kind",
        "status",
        "source_d2_plan",
        "source_d2_plan_sha256",
        "confirmation_freeze",
        "confirmation_freeze_sha256",
        "reveals",
        "fixtures",
        "evaluator",
        "evaluator_sha256",
        "cells",
        "operator_procedure_order",
        "order_evidence_class",
        "authorization",
        "plan_sha256",
    }
    if set(value) != fields:
        raise HKEContractError("confirmation plan envelope is malformed")
    body = dict(value)
    declared = body.pop("plan_sha256", None)
    if declared != canonical_sha256(body):
        raise HKEContractError("confirmation plan digest mismatch")
    if (
        value.get("schema") != SCHEMA
        or value.get("kind") != "sn56-week7-hke-confirmation-plan"
        or value.get("status") != "C1_AND_GUARDRAILS_READY_C2_SEALED"
        or not isinstance(value.get("cells"), Mapping)
        or set(value["cells"]) != {"social", "product", "logo_ui"}
        or any(
            set(value["cells"][family]) != {"incumbent", "candidate"}
            for family in value["cells"]
        )
        or value.get("confirmation_freeze_sha256")
        != value.get("confirmation_freeze", {}).get("confirmation_freeze_sha256")
        or value.get("authorization")
        != {
            "mechanical_plan_ready": True,
            "separate_owner_gpu_order_required": True,
            "gpu_execution_authorized": False,
            "c1_and_guardrail_execution_authorized": False,
            "c2_reveal_authorized": False,
            "checkpoint_promotion_authorized": False,
            "deployment_authorized": False,
        }
    ):
        raise HKEContractError("confirmation plan contract mismatch")
    d2_plan = _validate_d2_plan(value["source_d2_plan"])
    if value["source_d2_plan_sha256"] != d2_plan["plan_sha256"]:
        raise HKEContractError("confirmation plan D2 source mismatch")
    _validate_confirmation_freeze(value["confirmation_freeze"], d2_plan=d2_plan)
    expected = build_confirmation_plan(
        d2_plan,
        value["confirmation_freeze"],
        c1_reveals=value["reveals"],
    )
    if canonical_bytes(expected) != canonical_bytes(value):
        raise HKEContractError(
            "confirmation plan does not reproduce from its source chain"
        )
    return dict(value)


def _validate_confirmation_evidence(
    plan_value: Mapping[str, Any], evidence: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _validate_confirmation_plan(plan_value)
    fields = {
        "schema",
        "kind",
        "status",
        "plan_sha256",
        "curves",
        "evidence_sha256",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != fields:
        raise HKEContractError("confirmation evidence is incomplete")
    body = dict(evidence)
    declared = body.pop("evidence_sha256")
    if declared != canonical_sha256(body):
        raise HKEContractError("confirmation evidence digest mismatch")
    if (
        evidence["schema"] != SCHEMA
        or evidence["kind"] != "sn56-week7-hke-confirmation-evidence"
        or evidence["status"] != "OPERATOR_ATTESTED_PASS"
        or evidence["plan_sha256"] != plan["plan_sha256"]
        or not isinstance(evidence["curves"], Mapping)
        or set(evidence["curves"]) != set(plan["cells"])
    ):
        raise HKEContractError("confirmation evidence authority binding mismatch")
    comparisons: dict[str, Any] = {}
    for family in ("social", "product", "logo_ui"):
        family_curves = evidence["curves"][family]
        if not isinstance(family_curves, Mapping) or set(family_curves) != {
            "incumbent",
            "candidate",
        }:
            raise HKEContractError(f"{family} confirmation curves are incomplete")
        validated = {
            role: validate_score_curve(
                family_curves[role],
                plan_sha256=plan["plan_sha256"],
                plan_cell=plan["cells"][family][role],
                fixture=plan["fixtures"][family],
                evaluator_sha256=plan["evaluator_sha256"],
                expected_owner_identity=_plan_owner_identity(plan),
            )
            for role in ("incumbent", "candidate")
        }
        incumbent_step = plan["cells"][family]["incumbent"]["planned_steps"]
        candidate_step = plan["cells"][family]["candidate"]["planned_steps"]
        comparisons[family] = compare_heldout_rows(
            validated["incumbent"][incumbent_step]["rows"],
            validated["candidate"][candidate_step]["rows"],
        )
    return plan, comparisons


def build_c2_reveal_authority(
    confirmation_plan_value: Mapping[str, Any],
    confirmation_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the immutable borderline-C1 rule without exposing C2 membership."""

    plan, comparisons = _validate_confirmation_evidence(
        confirmation_plan_value, confirmation_evidence
    )
    c1 = _validate_comparison_summary(comparisons["social"], "social C1")
    relative = c1["relative_improvement"]
    point = relative["composite"]
    borderline = (
        C2_BORDERLINE_COMPOSITE_LOW <= point <= C2_BORDERLINE_COMPOSITE_HIGH
        or abs(float(c1["paired_relative_ci95"][0])) <= C2_BORDERLINE_CI_LOWER_ABS_MAX
    )
    components_safe = all(
        relative[component] > -MAX_RELATIVE_REGRESSION
        for component in ("prompted", "blank")
    )
    reveal = point > 0 and components_safe and borderline
    trigger = {
        "decision": "REVEAL_C2" if reveal else "KEEP_C2_SEALED",
        "composite_band": [
            C2_BORDERLINE_COMPOSITE_LOW,
            C2_BORDERLINE_COMPOSITE_HIGH,
        ],
        "ci_lower_abs_max": C2_BORDERLINE_CI_LOWER_ABS_MAX,
        "positive_composite": point > 0,
        "components_safe": components_safe,
        "borderline": borderline,
    }
    body = {
        "schema": SCHEMA,
        "kind": "sn56-week7-hke-c2-reveal-authorization",
        "status": (
            "C2_REVEAL_AUTHORIZED_BY_BORDERLINE_C1" if reveal else "C2_REMAINS_SEALED"
        ),
        "source_confirmation_plan": copy.deepcopy(plan),
        "source_confirmation_evidence": copy.deepcopy(dict(confirmation_evidence)),
        "source_confirmation_freeze_sha256": plan["confirmation_freeze_sha256"],
        "c1_evidence_sha256": confirmation_evidence["evidence_sha256"],
        "c1_comparison": c1,
        "trigger": trigger,
        "c2_commitment_sha256": plan["confirmation_freeze"]["confirmation_commitments"][
            "social"
        ]["C2"],
        "c2_reveal_authorized": reveal,
        "checkpoint_promotion_authorized": False,
        "deployment_authorized": False,
    }
    return {**body, "c2_authority_sha256": canonical_sha256(body)}


def build_c2_plan(
    d2_plan_value: Mapping[str, Any],
    confirmation_plan_value: Mapping[str, Any],
    confirmation_evidence: Mapping[str, Any],
    *,
    c2_authority: Mapping[str, Any],
    c2_reveal: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize the sealed reserve only when the predeclared trigger fires."""

    d2_plan = _validate_d2_plan(d2_plan_value)
    confirmation_plan, _comparisons = _validate_confirmation_evidence(
        confirmation_plan_value, confirmation_evidence
    )
    if canonical_bytes(d2_plan) != canonical_bytes(confirmation_plan["source_d2_plan"]):
        raise HKEContractError("C2 D2 source disagrees with confirmation plan")
    expected_authority = build_c2_reveal_authority(
        confirmation_plan, confirmation_evidence
    )
    if (
        dict(c2_authority) != expected_authority
        or not expected_authority["c2_reveal_authorized"]
    ):
        raise HKEContractError("C2 authority does not reproduce an eligible trigger")
    freeze = _validate_confirmation_freeze(
        confirmation_plan["confirmation_freeze"], d2_plan=d2_plan
    )
    reveal = _validate_confirmation_reveal_record(
        c2_reveal,
        family="social",
        pack="C2",
        confirmation_freeze=freeze,
        admission_set=d2_plan["source_prelaunch_plan"]["admission_set"],
        expected_authority_kind="sn56-week7-hke-c2-reveal-authorization",
        expected_authority_sha256=expected_authority["c2_authority_sha256"],
    )
    fixture = _fixture_from_confirmation_reveal(reveal)
    observation = _validate_h100_observation(
        d2_plan["accelerator_observation"],
        "C2 execution",
        require_fixed_capture=True,
    )
    incumbent = materialize_r0(
        yaml.safe_load(INCUMBENT_TEMPLATE_PATH.read_text(encoding="utf-8")),
        seed=TRAINING_SEED_B,
    )
    candidate = copy.deepcopy(freeze["candidate"]["generated_config"])
    _process_node(candidate)["training_seed"] = TRAINING_SEED_B
    cells: dict[str, Any] = {}
    for role, config, bundle, execution_key in (
        ("incumbent", incumbent, INCUMBENT_BUNDLE, "incumbent"),
        ("candidate", candidate, freeze["candidate"]["bundle_id"], "owned"),
    ):
        cells[role] = _plan_cell(
            family="social",
            pack="C2",
            phase="confirmation",
            arm=f"social-C2-{role}",
            seed=TRAINING_SEED_B,
            config=config,
            bundle_id=bundle,
            fixture=fixture,
            candidate_semantic_sha256=fixture["candidate_semantic_sha256"],
            fixture_admission_sha256=fixture["admission_sha256"],
            execution=d2_plan["execution_identities"][execution_key],
            observation=observation,
            timing_support=(
                None
                if role == "incumbent"
                else {"c2_authority_sha256": expected_authority["c2_authority_sha256"]}
            ),
        )
    body = {
        "schema": SCHEMA,
        "kind": "sn56-week7-hke-c2-plan",
        "status": "C2_READY_AFTER_BORDERLINE_TRIGGER",
        "source_d2_plan": copy.deepcopy(d2_plan),
        "source_d2_plan_sha256": d2_plan["plan_sha256"],
        "source_confirmation_plan": copy.deepcopy(confirmation_plan),
        "source_confirmation_plan_sha256": confirmation_plan["plan_sha256"],
        "source_confirmation_evidence": copy.deepcopy(dict(confirmation_evidence)),
        "confirmation_evidence_sha256": confirmation_evidence["evidence_sha256"],
        "c2_authority": expected_authority,
        "c2_authority_sha256": expected_authority["c2_authority_sha256"],
        "reveal": reveal,
        "fixture": fixture,
        "evaluator": copy.deepcopy(d2_plan["evaluator"]),
        "evaluator_sha256": d2_plan["evaluator_sha256"],
        "cells": cells,
        "operator_procedure_order": ["candidate", "incumbent"],
        "order_evidence_class": "operator_procedure_not_machine_verified",
        "authorization": {
            "mechanical_plan_ready": True,
            "separate_owner_gpu_order_required": True,
            "gpu_execution_authorized": False,
            "c2_execution_authorized": False,
            "checkpoint_promotion_authorized": False,
            "deployment_authorized": False,
        },
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def _validate_c2_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HKEContractError("C2 plan is unavailable")
    fields = {
        "schema",
        "kind",
        "status",
        "source_d2_plan",
        "source_d2_plan_sha256",
        "source_confirmation_plan",
        "source_confirmation_plan_sha256",
        "source_confirmation_evidence",
        "confirmation_evidence_sha256",
        "c2_authority",
        "c2_authority_sha256",
        "reveal",
        "fixture",
        "evaluator",
        "evaluator_sha256",
        "cells",
        "operator_procedure_order",
        "order_evidence_class",
        "authorization",
        "plan_sha256",
    }
    if set(value) != fields:
        raise HKEContractError("C2 plan envelope is malformed")
    body = dict(value)
    declared = body.pop("plan_sha256", None)
    if declared != canonical_sha256(body):
        raise HKEContractError("C2 plan digest mismatch")
    if (
        value.get("schema") != SCHEMA
        or value.get("kind") != "sn56-week7-hke-c2-plan"
        or value.get("status") != "C2_READY_AFTER_BORDERLINE_TRIGGER"
        or set(value.get("cells", {})) != {"incumbent", "candidate"}
        or value.get("c2_authority_sha256")
        != value.get("c2_authority", {}).get("c2_authority_sha256")
        or value.get("c2_authority", {}).get("c2_reveal_authorized") is not True
        or value.get("authorization")
        != {
            "mechanical_plan_ready": True,
            "separate_owner_gpu_order_required": True,
            "gpu_execution_authorized": False,
            "c2_execution_authorized": False,
            "checkpoint_promotion_authorized": False,
            "deployment_authorized": False,
        }
    ):
        raise HKEContractError("C2 plan contract mismatch")
    expected = build_c2_plan(
        value["source_d2_plan"],
        value["source_confirmation_plan"],
        value["source_confirmation_evidence"],
        c2_authority=value["c2_authority"],
        c2_reveal=value["reveal"],
    )
    if canonical_bytes(expected) != canonical_bytes(value):
        raise HKEContractError("C2 plan does not reproduce from its source chain")
    return dict(value)


def _validate_c2_evidence(
    plan_value: Mapping[str, Any], evidence: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _validate_c2_plan(plan_value)
    fields = {
        "schema",
        "kind",
        "status",
        "plan_sha256",
        "curves",
        "evidence_sha256",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != fields:
        raise HKEContractError("C2 evidence is incomplete")
    body = dict(evidence)
    declared = body.pop("evidence_sha256")
    if declared != canonical_sha256(body):
        raise HKEContractError("C2 evidence digest mismatch")
    if (
        evidence["schema"] != SCHEMA
        or evidence["kind"] != "sn56-week7-hke-c2-evidence"
        or evidence["status"] != "OPERATOR_ATTESTED_PASS"
        or evidence["plan_sha256"] != plan["plan_sha256"]
        or not isinstance(evidence["curves"], Mapping)
        or set(evidence["curves"]) != {"incumbent", "candidate"}
    ):
        raise HKEContractError("C2 evidence authority binding mismatch")
    curves = {
        role: validate_score_curve(
            evidence["curves"][role],
            plan_sha256=plan["plan_sha256"],
            plan_cell=plan["cells"][role],
            fixture=plan["fixture"],
            evaluator_sha256=plan["evaluator_sha256"],
            expected_owner_identity=_plan_owner_identity(plan),
        )
        for role in ("incumbent", "candidate")
    }
    comparison = compare_heldout_rows(
        curves["incumbent"][plan["cells"]["incumbent"]["planned_steps"]]["rows"],
        curves["candidate"][plan["cells"]["candidate"]["planned_steps"]]["rows"],
    )
    return plan, comparison


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise HKEContractError("cannot take a percentile of no values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def paired_interval(
    deltas: Sequence[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> tuple[float, float, float]:
    values = [float(value) for value in deltas]
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise HKEContractError("paired bootstrap requires two finite rows")
    if iterations < 100:
        raise HKEContractError("paired bootstrap iteration count is too small")
    rng = random.Random(seed)
    n = len(values)
    means = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations)
    ]
    alpha = (1.0 - UNCERTAINTY_LEVEL) / 2.0
    return (
        sum(values) / n,
        _percentile(means, alpha),
        _percentile(means, 1.0 - alpha),
    )


def _component_means(rows: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise HKEContractError("comparison has no held-out rows")
    prompted = [float(row["prompted_loss"]) for row in rows.values()]
    blank = [float(row["blank_loss"]) for row in rows.values()]
    composite = [0.25 * p + 0.75 * b for p, b in zip(prompted, blank, strict=True)]
    return {
        "prompted": sum(prompted) / len(prompted),
        "blank": sum(blank) / len(blank),
        "composite": sum(composite) / len(composite),
    }


def compare_heldout_rows(
    incumbent: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Compute paired lower-is-better improvements without collapsing components."""

    if set(incumbent) != set(candidate) or len(incumbent) < 2:
        raise HKEContractError("comparison rows are not exactly paired")
    for rows in (incumbent, candidate):
        for row in rows.values():
            _composite(row)
    incumbent_means = _component_means(incumbent)
    candidate_means = _component_means(candidate)
    improvements: dict[str, float] = {}
    for component in ("prompted", "blank", "composite"):
        baseline = incumbent_means[component]
        if baseline <= 0:
            raise HKEContractError("comparison baseline must be positive")
        improvements[component] = (baseline - candidate_means[component]) / baseline
    deltas = [
        _composite(incumbent[row_id]) - _composite(candidate[row_id])
        for row_id in sorted(incumbent)
    ]
    point, lower, upper = paired_interval(deltas)
    baseline = incumbent_means["composite"]
    return {
        "rows": len(incumbent),
        "incumbent_means": incumbent_means,
        "candidate_means": candidate_means,
        "relative_improvement": improvements,
        "paired_composite_delta": point,
        "paired_ci95": [lower, upper],
        "paired_relative_ci95": [lower / baseline, upper / baseline],
        "direction": "improves" if point > 0 else "regresses" if point < 0 else "tie",
    }


def _evaluate_futurebound_comparisons(comparisons: Mapping[str, Any]) -> dict[str, Any]:
    """Pure calculation used only after the public bound validator runs."""

    required = {"D1", "D2", "C1", "product", "logo_ui"}
    if not isinstance(comparisons, Mapping) or set(comparisons) != required:
        raise HKEContractError("FutureBound comparison inventory is incomplete")
    d2 = comparisons["D2"]
    if not isinstance(d2, Mapping) or set(d2) != {"Seed-A", "Seed-B"}:
        raise HKEContractError("both D2 finalist seeds are required")
    checked = {
        label: _validate_comparison_summary(comparisons[label], label)
        for label in ("D1", "C1", "product", "logo_ui")
    }
    checked_d2 = {
        seed: _validate_comparison_summary(value, f"D2 {seed}")
        for seed, value in d2.items()
    }
    comparisons = {**checked, "D2": checked_d2}
    d2 = checked_d2
    all_primary = [comparisons["D1"], *d2.values(), comparisons["C1"]]
    d2_composite = (
        sum(float(seed["relative_improvement"]["composite"]) for seed in d2.values())
        / 2.0
    )
    component_clear = all(
        float(result["relative_improvement"][component]) > -MAX_RELATIVE_REGRESSION
        for result in all_primary
        for component in ("prompted", "blank")
    )
    gates = {
        "d1_composite_at_least_3pct": comparisons["D1"]["relative_improvement"][
            "composite"
        ]
        >= MIN_COMPOSITE_IMPROVEMENT,
        "d2_composite_at_least_3pct": d2_composite >= MIN_COMPOSITE_IMPROVEMENT,
        "same_direction_d1_d2": comparisons["D1"]["direction"] == "improves"
        and all(seed["direction"] == "improves" for seed in d2.values()),
        "one_discovery_ci_clears_zero": comparisons["D1"]["paired_ci95"][0] > 0
        or any(seed["paired_ci95"][0] > 0 for seed in d2.values()),
        "prompted_and_blank_each_regress_less_than_1pct": component_clear,
        "both_finalist_seeds_agree": all(
            seed["relative_improvement"]["composite"] > 0 for seed in d2.values()
        ),
        "c1_composite_at_least_3pct": comparisons["C1"]["relative_improvement"][
            "composite"
        ]
        >= MIN_COMPOSITE_IMPROVEMENT,
        "c1_ci_clears_zero": comparisons["C1"]["paired_ci95"][0] > 0,
        "product_regression_less_than_1pct": comparisons["product"][
            "relative_improvement"
        ]["composite"]
        > -MAX_RELATIVE_REGRESSION,
        "logo_ui_regression_less_than_1pct": comparisons["logo_ui"][
            "relative_improvement"
        ]["composite"]
        > -MAX_RELATIVE_REGRESSION,
    }
    return {
        "comparisons": copy.deepcopy(dict(comparisons)),
        "gates": gates,
        "decision": "GO" if all(gates.values()) else "HOLD",
    }


def evaluate_futurebound_gates(
    confirmation_plan_value: Mapping[str, Any],
    confirmation_evidence: Mapping[str, Any],
    *,
    c2_plan_value: Mapping[str, Any] | None = None,
    c2_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit a decision only from the complete provenance-validated stage chain."""

    plan, confirmation_comparisons = _validate_confirmation_evidence(
        confirmation_plan_value, confirmation_evidence
    )
    freeze = plan["confirmation_freeze"]
    discovery = freeze["discovery_decision"]
    if discovery.get("decision") != "ADVANCE":
        raise HKEContractError("confirmation plan lacks an advancing discovery gate")
    comparisons = {
        "D1": discovery["d1_comparison"],
        "D2": discovery["d2_comparisons"],
        "C1": confirmation_comparisons["social"],
        "product": confirmation_comparisons["product"],
        "logo_ui": confirmation_comparisons["logo_ui"],
    }
    calculated = _evaluate_futurebound_comparisons(comparisons)
    c2_authority = build_c2_reveal_authority(plan, confirmation_evidence)
    c2_record: dict[str, Any] | None = None
    c2_gates: dict[str, bool] = {}
    if c2_authority["c2_reveal_authorized"]:
        if c2_plan_value is None or c2_evidence is None:
            raise HKEContractError(
                "borderline C1 requires the predeclared C2 replication"
            )
        c2_plan, c2_comparison = _validate_c2_evidence(c2_plan_value, c2_evidence)
        if (
            c2_plan["source_confirmation_plan_sha256"] != plan["plan_sha256"]
            or c2_plan["confirmation_evidence_sha256"]
            != confirmation_evidence["evidence_sha256"]
            or c2_plan["c2_authority"] != c2_authority
        ):
            raise HKEContractError("C2 evidence is not bound to this confirmation")
        c2_gates = {
            "c2_same_direction": c2_comparison["direction"] == "improves",
            "c2_prompted_and_blank_regress_less_than_1pct": all(
                c2_comparison["relative_improvement"][component]
                > -MAX_RELATIVE_REGRESSION
                for component in ("prompted", "blank")
            ),
        }
        c2_record = {
            "plan_sha256": c2_plan["plan_sha256"],
            "evidence_sha256": c2_evidence["evidence_sha256"],
            "comparison": c2_comparison,
        }
    elif c2_plan_value is not None or c2_evidence is not None:
        raise HKEContractError(
            "C2 evidence was supplied without the borderline trigger"
        )
    gates = {**calculated["gates"], **c2_gates}
    decision = (
        "GO" if calculated["decision"] == "GO" and all(gates.values()) else "HOLD"
    )
    result = {
        "schema": SCHEMA,
        "kind": "sn56-week7-hke-provenance-bound-final-decision",
        "status": "OPERATOR_ATTESTED_NO_SHIP_AUTHORITY",
        "confirmation_plan_sha256": plan["plan_sha256"],
        "confirmation_evidence_sha256": confirmation_evidence["evidence_sha256"],
        "confirmation_freeze_sha256": plan["confirmation_freeze_sha256"],
        "comparisons": calculated["comparisons"],
        "c2_authority": c2_authority,
        "c2": c2_record,
        "gates": gates,
        "decision": decision,
        "authorization": {
            "checkpoint_promotion_authorized": False,
            "deployment_authorized": False,
        },
    }
    return {**result, "decision_sha256": canonical_sha256(result)}


def analyze(
    plan_value: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Reject the retired monolithic schema-2 analysis interface."""

    del plan_value, evidence
    raise HKEContractError(
        "monolithic factorial analysis is retired; use the staged validators"
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("contract")
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--plan", type=Path, required=True)
    analyze_parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "contract":
        sys.stdout.buffer.write(canonical_bytes(experiment_contract()))
        return 0
    result = analyze(_load_json(args.plan), _load_json(args.evidence))
    sys.stdout.buffer.write(canonical_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
