"""Static host-performance identity plus thresholded live preflight."""

from __future__ import annotations

import hashlib
import csv
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import time
from typing import Any


_SHA256 = re.compile(r"[0-9a-f]{64}")
_UUID = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-" r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_INSTANCE_ASSURANCE = "operational-fingerprint-not-cryptographic-attestation"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _exact(value: dict[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(keys-set(value))}, "
            f"extra={sorted(set(value)-keys)}"
        )


def _module_sha256() -> str:
    return hashlib.sha256(Path(__file__).resolve(strict=True).read_bytes()).hexdigest()


def _strict_text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{label} must be non-empty canonical text")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite_number(value: Any, label: str, *, positive: bool = True) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (float(value) <= 0 if positive else float(value) < 0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be finite and {qualifier}")
    return float(value)


def _canonical_timestamp(
    value: Any,
    label: str,
    *,
    maximum_age_s: float | None = None,
    maximum_future_s: float = 5,
) -> str:
    text = _strict_text(value, label)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", text):
        raise ValueError(f"{label} must be canonical UTC (YYYY-MM-DDTHH:MM:SSZ)")
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{label} is not a real UTC timestamp") from exc
    now = datetime.now(timezone.utc)
    if parsed < datetime(2020, 1, 1, tzinfo=timezone.utc) or parsed > now + timedelta(
        seconds=maximum_future_s
    ):
        raise ValueError(f"{label} is outside accepted evidence time bounds")
    if maximum_age_s is not None and (now - parsed).total_seconds() > maximum_age_s:
        raise ValueError(f"{label} is stale")
    return text


def _absolute_linux_path(value: Any, label: str) -> str:
    text = _strict_text(value, label)
    pure = PurePosixPath(text)
    if not pure.is_absolute() or str(pure) != text or ".." in pure.parts:
        raise ValueError(f"{label} must be a normalized absolute path")
    return text


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _bootstrap_receipt_binding(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _object(value, "bootstrap receipt binding")
    _exact(
        binding,
        {
            "path",
            "file_sha256",
            "receipt_sha256",
            "container_image_sha256",
        },
        "bootstrap receipt binding",
    )
    path = Path(os.path.abspath(os.path.expanduser(binding["path"])))
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"bootstrap receipt has a symlink component: {current}")
        current = current.parent
    if not path.is_file():
        raise ValueError("bootstrap receipt must be a regular file")
    raw = path.read_bytes()
    try:
        receipt = _object(json.loads(raw), "bootstrap receipt")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("bootstrap receipt is not JSON") from exc
    canonical = (
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if raw != canonical:
        raise ValueError("bootstrap receipt must be canonical JSON")
    try:
        from . import krea_host_bootstrap
    except ImportError:  # pragma: no cover - direct script execution.
        import krea_host_bootstrap  # type: ignore[no-redef]

    krea_host_bootstrap.validate_receipt(receipt, recapture=False)
    image_sha = receipt["spec"]["runtime"]["container_image_sha256"]
    if (
        hashlib.sha256(raw).hexdigest()
        != _digest(binding["file_sha256"], "bootstrap receipt file SHA-256")
        or receipt["receipt_sha256"]
        != _digest(binding["receipt_sha256"], "bootstrap receipt semantic SHA-256")
        or image_sha
        != _digest(binding["container_image_sha256"], "bootstrap image SHA-256")
    ):
        raise ValueError("bootstrap receipt binding drifted")
    return binding, receipt


def validate_manifest(value: dict[str, Any]) -> dict[str, Any]:
    value = _object(value, "host execution manifest")
    schema = value.get("schema")
    manifest_keys = {
        "schema",
        "kind",
        "static",
        "preflight_policy",
        "host_execution_identity_sha256",
    }
    if schema == 3:
        manifest_keys.add("bootstrap_receipt")
    _exact(
        value,
        manifest_keys,
        "host execution manifest",
    )
    static = _object(value["static"], "host static identity")
    _exact(
        static,
        {"instance", "cpu", "memory", "checkpoint_filesystem", "gpu"},
        "host static identity",
    )
    instance = _object(static["instance"], "host instance binding")
    _exact(
        instance,
        {
            "machine_id_sha256",
            "product_uuid_sha256",
            "boot_id_sha256",
            "cgroup_v2_path",
            "assurance",
        },
        "host instance binding",
    )
    for key in ("machine_id_sha256", "product_uuid_sha256", "boot_id_sha256"):
        _digest(instance[key], f"host instance {key}")
    _absolute_linux_path(instance["cgroup_v2_path"], "host cgroup_v2_path")
    if instance["assurance"] != _INSTANCE_ASSURANCE:
        raise ValueError("host instance binding overstates or changes its assurance")
    cpu = _object(static["cpu"], "host cpu")
    _exact(
        cpu,
        {
            "model",
            "logical_cpus",
            "process_affinity_cpu_ids",
            "cgroup_cpuset_cpu_ids",
            "effective_cpu_ids",
            "allowed_logical_cpus",
            "cgroup_cpuset_logical_cpus",
            "cpu_quota_cores",
            "effective_cpu_capacity",
        },
        "host cpu",
    )
    memory = _object(static["memory"], "host memory")
    _exact(
        memory,
        {"total_bytes", "cgroup_limit_bytes", "effective_capacity_bytes"},
        "host memory",
    )
    filesystem = _object(static["checkpoint_filesystem"], "checkpoint filesystem")
    _exact(
        filesystem,
        {
            "checkpoint_path",
            "mount_target",
            "source",
            "filesystem_type",
            "mount_options",
            "device_major_minor",
            "device_id",
        },
        "checkpoint filesystem",
    )
    gpu = _object(static["gpu"], "host gpu")
    _exact(
        gpu,
        {
            "uuid",
            "name",
            "driver_version",
            "mig_mode",
            "power_limit_w",
            "max_sm_clock_mhz",
            "max_memory_clock_mhz",
            "total_memory_mib",
        },
        "host gpu",
    )
    policy = _object(value["preflight_policy"], "host preflight policy")
    _exact(
        policy,
        {
            "maximum_load_per_effective_cpu",
            "minimum_available_memory_bytes",
            "minimum_checkpoint_free_bytes",
            "maximum_gpu_utilization_percent",
            "minimum_free_gpu_memory_mib",
            "maximum_foreign_compute_processes",
            "storage_probe_bytes",
            "minimum_checkpoint_write_mib_s",
            "minimum_checkpoint_read_mib_s",
            "maximum_checkpoint_fsync_s",
            "storage_probe_tool_sha256",
        },
        "host preflight policy",
    )
    logical = _positive_int(cpu["logical_cpus"], "logical_cpus")

    def cpu_ids(key: str) -> list[int]:
        items = cpu[key]
        if (
            not isinstance(items, list)
            or not items
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or not 0 <= item < logical
                for item in items
            )
        ):
            raise ValueError(f"cpu.{key} must be sorted unique host CPU IDs")
        if items != sorted(items) or len(items) != len(set(items)):
            raise ValueError(f"cpu.{key} must be sorted unique host CPU IDs")
        return items

    affinity_ids = cpu_ids("process_affinity_cpu_ids")
    cpuset_ids = cpu_ids("cgroup_cpuset_cpu_ids")
    effective_ids = cpu_ids("effective_cpu_ids")
    expected_ids = sorted(set(affinity_ids) & set(cpuset_ids))
    if effective_ids != expected_ids:
        raise ValueError("effective_cpu_ids do not match affinity/cgroup intersection")
    allowed = _positive_int(cpu["allowed_logical_cpus"], "allowed_logical_cpus")
    cpuset = _positive_int(
        cpu["cgroup_cpuset_logical_cpus"], "cgroup_cpuset_logical_cpus"
    )
    if allowed != len(effective_ids) or cpuset != len(cpuset_ids):
        raise ValueError("effective CPU counts do not match their bound CPU ID sets")
    quota = cpu["cpu_quota_cores"]
    if quota is not None:
        quota = _finite_number(quota, "cpu_quota_cores")
    effective = _finite_number(cpu["effective_cpu_capacity"], "effective_cpu_capacity")
    expected_effective = min([float(allowed)] + ([] if quota is None else [quota]))
    if not math.isclose(effective, expected_effective, rel_tol=0, abs_tol=1e-9):
        raise ValueError("effective_cpu_capacity does not match affinity/cpuset/quota")
    _strict_text(cpu["model"], "cpu.model")

    total_memory = _positive_int(memory["total_bytes"], "total_bytes")
    memory_limit = memory["cgroup_limit_bytes"]
    if memory_limit is not None:
        memory_limit = _positive_int(memory_limit, "cgroup_limit_bytes")
    effective_memory = _positive_int(
        memory["effective_capacity_bytes"], "effective_capacity_bytes"
    )
    expected_memory = min(
        [total_memory] + ([] if memory_limit is None else [memory_limit])
    )
    if effective_memory != expected_memory:
        raise ValueError("effective_capacity_bytes does not match host/cgroup limits")

    checkpoint_path = _absolute_linux_path(
        filesystem["checkpoint_path"], "checkpoint filesystem path"
    )
    mount_target = _absolute_linux_path(
        filesystem["mount_target"], "checkpoint mount target"
    )
    if not PurePosixPath(checkpoint_path).is_relative_to(PurePosixPath(mount_target)):
        raise ValueError("checkpoint path is not on the declared mount target")
    for label, item in (
        ("filesystem.source", filesystem["source"]),
        ("filesystem.filesystem_type", filesystem["filesystem_type"]),
    ):
        _strict_text(item, label)
    if not re.fullmatch(
        r"\d+:\d+",
        _strict_text(filesystem["device_major_minor"], "filesystem.device_major_minor"),
    ):
        raise ValueError("checkpoint filesystem device_major_minor is invalid")
    _positive_int(filesystem["device_id"], "filesystem.device_id")
    options = filesystem["mount_options"]
    if not isinstance(options, list) or any(
        not isinstance(item, str) or not item or item != item.strip()
        for item in options
    ):
        raise ValueError("checkpoint mount options are invalid")
    if options != sorted(options) or len(options) != len(set(options)):
        raise ValueError("checkpoint mount options are invalid")
    if "ro" in options or "rw" not in options:
        raise ValueError("checkpoint filesystem must be mounted read-write")

    gpu_total = _positive_int(gpu["total_memory_mib"], "total_memory_mib")
    for label, item in (
        ("gpu.uuid", gpu["uuid"]),
        ("gpu.name", gpu["name"]),
        ("gpu.driver_version", gpu["driver_version"]),
        ("gpu.mig_mode", gpu["mig_mode"]),
    ):
        _strict_text(item, label)
    for key in ("max_sm_clock_mhz", "max_memory_clock_mhz"):
        _positive_int(gpu[key], f"gpu.{key}")
    _finite_number(gpu["power_limit_w"], "gpu.power_limit_w")

    _finite_number(
        policy["maximum_load_per_effective_cpu"],
        "maximum_load_per_effective_cpu",
    )
    minimum_memory = _positive_int(
        policy["minimum_available_memory_bytes"], "minimum_available_memory_bytes"
    )
    if minimum_memory > effective_memory:
        raise ValueError("minimum_available_memory_bytes exceeds effective capacity")
    _positive_int(
        policy["minimum_checkpoint_free_bytes"], "minimum_checkpoint_free_bytes"
    )
    minimum_gpu = _positive_int(
        policy["minimum_free_gpu_memory_mib"], "minimum_free_gpu_memory_mib"
    )
    if minimum_gpu > gpu_total:
        raise ValueError("minimum_free_gpu_memory_mib exceeds total GPU memory")
    gpu_util = policy["maximum_gpu_utilization_percent"]
    if (
        isinstance(gpu_util, bool)
        or not isinstance(gpu_util, (int, float))
        or not math.isfinite(float(gpu_util))
        or not 0 <= float(gpu_util) <= 100
    ):
        raise ValueError("maximum_gpu_utilization_percent is invalid")
    process_limit = policy["maximum_foreign_compute_processes"]
    if (
        isinstance(process_limit, bool)
        or not isinstance(process_limit, int)
        or process_limit < 0
    ):
        raise ValueError("maximum_foreign_compute_processes is invalid")
    probe_bytes = policy["storage_probe_bytes"]
    if (
        isinstance(probe_bytes, bool)
        or not isinstance(probe_bytes, int)
        or not 4 * 1024 * 1024 <= probe_bytes <= 256 * 1024 * 1024
    ):
        raise ValueError("storage_probe_bytes must be between 4 and 256 MiB")
    for key in (
        "minimum_checkpoint_write_mib_s",
        "minimum_checkpoint_read_mib_s",
        "maximum_checkpoint_fsync_s",
    ):
        _finite_number(policy[key], key)
    expected_probe_tool = _module_sha256()
    if policy["storage_probe_tool_sha256"] != expected_probe_tool:
        raise ValueError("storage probe tool identity differs from this module")
    body = {
        "schema": schema,
        "kind": value["kind"],
        "static": static,
        "preflight_policy": policy,
    }
    if schema == 3:
        binding, receipt = _bootstrap_receipt_binding(value["bootstrap_receipt"])
        body["bootstrap_receipt"] = binding
        receipt_checkpoint = Path(receipt["spec"]["sources"]["checkpoints"])
        receipt_target = Path("/app/checkpoints")
        manifest_checkpoint = Path(static["checkpoint_filesystem"]["checkpoint_path"])
        if not any(
            manifest_checkpoint == root or manifest_checkpoint.is_relative_to(root)
            for root in (receipt_checkpoint, receipt_target)
        ):
            raise ValueError(
                "host manifest checkpoint path escaped its bootstrap receipt"
            )
    if (
        schema not in {2, 3}
        or value["kind"] != "forge-krea-host-execution-identity"
        or not isinstance(value["host_execution_identity_sha256"], str)
        or not _SHA256.fullmatch(value["host_execution_identity_sha256"])
        or value["host_execution_identity_sha256"] != canonical_sha256(body)
    ):
        raise ValueError("host execution identity digest mismatch")
    return value


def bootstrap_runtime(manifest: dict[str, Any], *, recapture: bool) -> dict[str, Any]:
    """Return the schema-3 sealed runtime after optional live recapture."""

    validate_manifest(manifest)
    if manifest.get("schema") != 3:
        raise ValueError("executable host manifest lacks a bootstrap receipt")
    _, receipt = _bootstrap_receipt_binding(manifest["bootstrap_receipt"])
    if recapture:
        try:
            from . import krea_host_bootstrap
        except ImportError:  # pragma: no cover - direct script execution.
            import krea_host_bootstrap  # type: ignore[no-redef]

        krea_host_bootstrap.validate_receipt(receipt, recapture=True)
    return dict(receipt["spec"]["runtime"])


def bootstrap_execution_surface(
    manifest: dict[str, Any], *, recapture: bool
) -> dict[str, Any]:
    """Return the receipt-bound host-venv execution surface identity."""

    runtime = bootstrap_runtime(manifest, recapture=recapture)
    _, receipt = _bootstrap_receipt_binding(manifest["bootstrap_receipt"])
    layout = _object(receipt["layout_identity"], "bootstrap layout identity")
    sources = _object(layout.get("sources"), "bootstrap source identity")
    venv = _object(sources.get("venv_python"), "bootstrap venv identity")
    _exact(
        venv,
        {
            "relative_path",
            "is_symlink",
            "resolved_relative_path",
            "resolved_sha256",
        },
        "bootstrap venv identity",
    )
    venv_tree = _object(sources.get("venv_tree"), "bootstrap venv tree identity")
    _exact(
        venv_tree,
        {"entry_count", "manifest_sha256"},
        "bootstrap venv tree identity",
    )
    _positive_int(venv_tree["entry_count"], "venv tree entry_count")
    _digest(venv_tree["manifest_sha256"], "venv tree manifest SHA-256")
    stage1_runtime = _object(
        sources.get("stage1_runtime"), "bootstrap Stage-1 runtime identity"
    )
    _exact(
        stage1_runtime,
        {
            "path",
            "file_sha256",
            "receipt_sha256",
            "venv_tree_entries_sha256",
            "verification",
        },
        "bootstrap Stage-1 runtime identity",
    )
    for key in (
        "file_sha256",
        "receipt_sha256",
        "venv_tree_entries_sha256",
    ):
        _digest(stage1_runtime[key], f"Stage-1 runtime {key}")
    bindings = _object(layout.get("bindings"), "bootstrap binding identity")
    runtime_cache = _object(
        bindings.get("runtime_cache"), "bootstrap runtime cache identity"
    )
    _exact(
        runtime_cache,
        {"path", "device_id", "mode", "uid", "policy"},
        "bootstrap runtime cache identity",
    )
    if (
        runtime_cache["path"] != "/cache/krea-runtime"
        or runtime_cache["mode"] != 0o700
        or runtime_cache["uid"] != 0
        or not isinstance(runtime_cache["device_id"], int)
        or runtime_cache["device_id"] < 0
    ):
        raise ValueError("bootstrap runtime cache identity is unsafe")
    cache_policy = _object(runtime_cache["policy"], "bootstrap runtime cache policy")
    expected_cache_policy = {
        "root": "/cache/krea-runtime",
        "namespace_derivation": (
            "timing_plan_file_sha256_plus_capture_id_or_execution_plan_file_sha256"
        ),
        "initial_state": "root-empty-before-bootstrap",
        "cross_capture_or_plan_reuse": False,
        "within_process_reuse": True,
    }
    if cache_policy != expected_cache_policy:
        raise ValueError("bootstrap runtime cache policy drifted")
    return {
        "runtime": runtime,
        "venv_python": dict(venv),
        "venv_tree": dict(venv_tree),
        "stage1_runtime": dict(stage1_runtime),
        "runtime_cache": dict(runtime_cache),
    }


def _run(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip()


def _meminfo() -> dict[str, int]:
    rows: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        key, raw = line.split(":", 1)
        fields = raw.split()
        if key in {"MemTotal", "MemAvailable"}:
            if len(fields) != 2 or fields[1] != "kB":
                raise RuntimeError(f"/proc/meminfo {key} is malformed")
            rows[key] = _positive_int(int(fields[0]), f"/proc/meminfo {key}") * 1024
    if set(rows) != {"MemTotal", "MemAvailable"}:
        raise RuntimeError("/proc/meminfo lacks total or available memory")
    return rows


def _cgroup_root() -> Path:
    rows = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    unified = [row.split("::", 1)[1] for row in rows if row.startswith("0::")]
    if len(unified) != 1:
        raise RuntimeError("host contract requires a unified cgroup-v2 identity")
    raw_path = unified[0]
    if not raw_path.startswith("/") or ".." in PurePosixPath(raw_path).parts:
        raise RuntimeError("current cgroup-v2 path is malformed")
    relative = raw_path.lstrip("/")
    base = _cgroup_base()
    root = base / relative
    current = root
    while True:
        if current.is_symlink() or not current.is_dir():
            raise RuntimeError(f"current cgroup path is unsafe: {current}")
        if current == base:
            break
        if current == current.parent or not current.is_relative_to(base):
            raise RuntimeError("current cgroup path escapes cgroup-v2")
        current = current.parent
    return root


def _cgroup_base() -> Path:
    return Path("/sys/fs/cgroup")


def _parse_cpuset(value: str) -> set[int]:
    cpus: set[int] = set()
    text = value.strip()
    if not text:
        raise RuntimeError("cpuset.cpus.effective is empty")
    for item in text.split(","):
        if re.fullmatch(r"\d+", item):
            first = last = int(item)
        elif re.fullmatch(r"\d+-\d+", item):
            first, last = (int(part) for part in item.split("-", 1))
            if first > last:
                raise RuntimeError("cpuset.cpus.effective contains a reversed range")
        else:
            raise RuntimeError("cpuset.cpus.effective is malformed")
        values = set(range(first, last + 1))
        if cpus & values:
            raise RuntimeError("cpuset.cpus.effective contains duplicate CPUs")
        cpus.update(values)
    return cpus


def _cgroup_ancestors(root: Path) -> list[Path]:
    base = _cgroup_base()
    if not root.is_relative_to(base):
        raise RuntimeError("current cgroup path escapes cgroup-v2")
    result = []
    current = root
    while True:
        result.append(current)
        if current == base:
            return result
        current = current.parent


def _cgroup_constraints() -> dict[str, Any]:
    """Resolve effective cpuset/quota/headroom across every cgroup ancestor."""

    root = _cgroup_root()
    cpuset = _parse_cpuset((root / "cpuset.cpus.effective").read_text(encoding="ascii"))
    quota_capacities: list[float] = []
    memory_limits: list[int] = []
    memory_headrooms: list[int] = []
    for ancestor in _cgroup_ancestors(root):
        cpu_fields = (ancestor / "cpu.max").read_text(encoding="ascii").split()
        if len(cpu_fields) != 2:
            raise RuntimeError(f"cgroup cpu.max is malformed: {ancestor}")
        try:
            period = int(cpu_fields[1])
            quota = None if cpu_fields[0] == "max" else int(cpu_fields[0])
        except ValueError as exc:
            raise RuntimeError(f"cgroup cpu.max is malformed: {ancestor}") from exc
        if period <= 0 or (quota is not None and quota <= 0):
            raise RuntimeError(f"cgroup cpu.max is malformed: {ancestor}")
        if quota is not None:
            quota_capacities.append(quota / period)

        memory_raw = (ancestor / "memory.max").read_text(encoding="ascii").strip()
        current_raw = (ancestor / "memory.current").read_text(encoding="ascii").strip()
        try:
            memory_limit = None if memory_raw == "max" else int(memory_raw)
            memory_current = int(current_raw)
        except ValueError as exc:
            raise RuntimeError(
                f"cgroup memory controls are malformed: {ancestor}"
            ) from exc
        if memory_current < 0 or (memory_limit is not None and memory_limit <= 0):
            raise RuntimeError(f"cgroup memory controls are malformed: {ancestor}")
        if memory_limit is not None:
            if memory_current > memory_limit:
                raise RuntimeError("cgroup memory usage exceeds an ancestor limit")
            memory_limits.append(memory_limit)
            memory_headrooms.append(memory_limit - memory_current)
    relative = str(root.relative_to(_cgroup_base()))
    return {
        "path": "/" if relative == "." else "/" + relative,
        "cpuset_cpu_ids": sorted(cpuset),
        "cpuset_logical_cpus": len(cpuset),
        "cpu_quota_cores": min(quota_capacities) if quota_capacities else None,
        "memory_limit_bytes": min(memory_limits) if memory_limits else None,
        "memory_available_bytes": min(memory_headrooms) if memory_headrooms else None,
    }


def _safe_checkpoint_path(value: Path) -> Path:
    raw = Path(os.path.expanduser(value))
    if not raw.is_absolute():
        raise ValueError("checkpoint path must be absolute")
    path = Path(os.path.abspath(raw))
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"checkpoint path has a symlink ancestor: {current}")
        if current == current.parent:
            break
        current = current.parent
    if not path.is_dir():
        raise ValueError(f"checkpoint path must be a real directory: {path}")
    return path


def _host_identifier(path: Path, label: str, pattern: re.Pattern[str]) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} source is not a regular file")
    value = path.read_text(encoding="ascii").strip().lower()
    if not pattern.fullmatch(value):
        raise RuntimeError(f"{label} source is malformed")
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _instance_binding(cgroup_path: str) -> dict[str, str]:
    return {
        "machine_id_sha256": _host_identifier(
            Path("/etc/machine-id"), "machine-id", re.compile(r"[0-9a-f]{32}")
        ),
        "product_uuid_sha256": _host_identifier(
            Path("/sys/devices/virtual/dmi/id/product_uuid"), "product UUID", _UUID
        ),
        "boot_id_sha256": _host_identifier(
            Path("/proc/sys/kernel/random/boot_id"), "boot ID", _UUID
        ),
        "cgroup_v2_path": cgroup_path,
        "assurance": _INSTANCE_ASSURANCE,
    }


def _filesystem_identity(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(
            _run(
                [
                    "findmnt",
                    "--json",
                    "--output",
                    "SOURCE,TARGET,FSTYPE,OPTIONS,MAJ:MIN",
                    "--target",
                    str(path),
                ]
            )
        )
        filesystems = document["filesystems"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("findmnt did not return valid JSON") from exc
    if not isinstance(filesystems, list) or len(filesystems) != 1:
        raise RuntimeError("findmnt did not return exactly one filesystem identity")
    row = _object(filesystems[0], "findmnt filesystem")
    if set(row) != {"source", "target", "fstype", "options", "maj:min"}:
        raise RuntimeError("findmnt did not return a complete filesystem identity")
    stat_result = path.stat()
    major_minor = f"{os.major(stat_result.st_dev)}:{os.minor(stat_result.st_dev)}"
    if row["maj:min"] != major_minor:
        raise RuntimeError("findmnt and stat disagree on checkpoint filesystem")
    target = _absolute_linux_path(row["target"], "findmnt mount target")
    if not PurePosixPath(str(path)).is_relative_to(PurePosixPath(target)):
        raise RuntimeError("findmnt target does not contain checkpoint path")
    options = sorted(set(_strict_text(row["options"], "findmnt options").split(",")))
    return {
        "checkpoint_path": str(path),
        "mount_target": target,
        "source": _strict_text(row["source"], "findmnt source"),
        "filesystem_type": _strict_text(row["fstype"], "findmnt fstype"),
        "mount_options": options,
        "device_major_minor": major_minor,
        "device_id": stat_result.st_dev,
    }


def _gpu_row() -> tuple[dict[str, Any], float, int]:
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,mig.mode.current,power.limit,clocks.max.sm,clocks.max.memory,memory.total,utilization.gpu,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        rows = list(csv.reader(output.splitlines()))
    except csv.Error as exc:
        raise RuntimeError("nvidia-smi did not return valid CSV") from exc
    if len(rows) != 1 or len(rows[0]) != 10:
        raise RuntimeError("host contract requires one complete visible GPU row")
    fields = [item.strip() for item in rows[0]]
    try:
        power = _finite_number(float(fields[4]), "GPU power limit")
        sm_clock = _positive_int(int(float(fields[5])), "GPU max SM clock")
        memory_clock = _positive_int(int(float(fields[6])), "GPU max memory clock")
        total_memory = _positive_int(int(float(fields[7])), "GPU total memory")
        utilization = _finite_number(
            float(fields[8]), "GPU utilization", positive=False
        )
        free_memory = _nonnegative_int(int(float(fields[9])), "GPU free memory")
    except ValueError as exc:
        raise RuntimeError("nvidia-smi returned malformed numeric GPU facts") from exc
    if utilization > 100 or free_memory > total_memory:
        raise RuntimeError("nvidia-smi returned impossible live GPU facts")
    static = {
        "uuid": _strict_text(fields[0], "GPU UUID"),
        "name": _strict_text(fields[1], "GPU name"),
        "driver_version": _strict_text(fields[2], "GPU driver version"),
        "mig_mode": _strict_text(fields[3], "GPU MIG mode"),
        "power_limit_w": power,
        "max_sm_clock_mhz": sm_clock,
        "max_memory_clock_mhz": memory_clock,
        "total_memory_mib": total_memory,
    }
    return static, utilization, free_memory


def _compute_processes(gpu_uuid: str) -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        # NVIDIA returns a nonzero status on some versions when no processes
        # exist.  Accept only the explicit no-process diagnostic.
        combined = (result.stdout + result.stderr).casefold()
        if "no running processes" not in combined:
            raise RuntimeError("cannot establish GPU compute-process occupancy")
        return []
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) == 1 and "no running processes" in rows[0].casefold():
        return []
    try:
        parsed = [next(csv.reader([row]), []) for row in rows]
        if any(len(row) != 2 or row[0].strip() != gpu_uuid for row in parsed):
            raise ValueError("compute process belongs to an unexpected GPU")
        pids = [int(row[1].strip()) for row in parsed]
        if any(pid <= 0 for pid in pids) or len(pids) != len(set(pids)):
            raise ValueError("compute process PIDs are invalid or duplicated")
        return sorted(pids)
    except (ValueError, csv.Error) as exc:
        raise RuntimeError(
            "nvidia-smi returned malformed compute-process PIDs"
        ) from exc


def _storage_probe_block() -> bytes:
    seed = hashlib.sha256(b"forge-krea-storage-probe-v2").digest()
    return (seed * ((1024 * 1024 // len(seed)) + 1))[: 1024 * 1024]


def _storage_probe_content_sha256(byte_count: int) -> str:
    block = _storage_probe_block()
    digest = hashlib.sha256()
    consumed = 0
    while consumed < byte_count:
        chunk = block[: min(len(block), byte_count - consumed)]
        digest.update(chunk)
        consumed += len(chunk)
    return digest.hexdigest()


def _storage_probe(path: Path, *, byte_count: int) -> dict[str, Any]:
    """Measure bounded same-filesystem write/fsync/read cost immediately pre-run."""

    path = _safe_checkpoint_path(path)
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or not 4 * 1024 * 1024 <= byte_count <= 256 * 1024 * 1024
    ):
        raise ValueError("storage probe bytes must be between 4 and 256 MiB")
    descriptor, raw_name = tempfile.mkstemp(prefix=".krea-io-probe-", dir=path)
    probe_path = Path(raw_name)
    block = _storage_probe_block()
    written = 0
    write_digest = hashlib.sha256()
    result: dict[str, Any] | None = None
    write_started = time.monotonic_ns()
    try:
        while written < byte_count:
            chunk = block[: min(len(block), byte_count - written)]
            count = os.write(descriptor, chunk)
            if count <= 0:
                raise RuntimeError("storage probe write made no progress")
            write_digest.update(chunk[:count])
            written += count
        write_finished = time.monotonic_ns()
        fsync_started = write_finished
        os.fsync(descriptor)
        fsync_finished = time.monotonic_ns()
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            os.posix_fadvise(descriptor, 0, written, os.POSIX_FADV_DONTNEED)
            cache_drop_requested = True
        else:
            cache_drop_requested = False
        os.close(descriptor)
        descriptor = -1
        read_fd = os.open(probe_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            read_started = time.monotonic_ns()
            read_total = 0
            read_digest = hashlib.sha256()
            while True:
                chunk = os.read(read_fd, 1024 * 1024)
                if not chunk:
                    break
                read_total += len(chunk)
                read_digest.update(chunk)
            read_finished = time.monotonic_ns()
        finally:
            os.close(read_fd)
        probe_stat = probe_path.stat()
        if (
            read_total != byte_count
            or probe_stat.st_size != byte_count
            or read_digest.hexdigest() != write_digest.hexdigest()
            or read_digest.hexdigest() != _storage_probe_content_sha256(byte_count)
        ):
            raise RuntimeError("storage probe byte content is inconsistent")
        directory_stat = path.stat()
        if probe_stat.st_dev != directory_stat.st_dev:
            raise RuntimeError("storage probe escaped the checkpoint filesystem")
        write_s = (write_finished - write_started) / 1_000_000_000
        fsync_s = (fsync_finished - fsync_started) / 1_000_000_000
        read_s = (read_finished - read_started) / 1_000_000_000
        mib = byte_count / (1024 * 1024)
        if min(write_s, read_s, fsync_s) <= 0:
            raise RuntimeError("storage probe clock produced invalid durations")
        result = {
            "checkpoint_path": str(path),
            "device_major_minor": (
                f"{os.major(directory_stat.st_dev)}:{os.minor(directory_stat.st_dev)}"
            ),
            "bytes": byte_count,
            "content_sha256": read_digest.hexdigest(),
            "write_s": write_s,
            "fsync_s": fsync_s,
            "read_s": read_s,
            "write_mib_s": mib / write_s,
            "read_mib_s": mib / read_s,
            "cache_drop_requested": cache_drop_requested,
            "tool_sha256": _module_sha256(),
            "observed_at_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if probe_path.exists() and not probe_path.is_symlink():
            probe_path.unlink()
        directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    if result is None:  # pragma: no cover - exceptions leave through the try.
        raise RuntimeError("storage probe did not produce a result")
    return result


def observe(
    checkpoint_path: Path, *, storage_probe_bytes: int | None = None
) -> dict[str, Any]:
    """Collect Linux/NVIDIA facts used by the measured-host contract."""

    checkpoint_path = _safe_checkpoint_path(checkpoint_path)
    cpu_model = ""
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    if not cpu_model:
        raise RuntimeError("/proc/cpuinfo lacks a CPU model")
    mem = _meminfo()
    cgroup = _cgroup_constraints()
    logical_cpus = _positive_int(int(os.cpu_count() or 0), "host logical CPUs")
    affinity_ids = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(logical_cpus))
    )
    cpuset_ids = cgroup["cpuset_cpu_ids"]
    effective_ids = sorted(set(affinity_ids) & set(cpuset_ids))
    if (
        not effective_ids
        or any(cpu_id < 0 or cpu_id >= logical_cpus for cpu_id in affinity_ids)
        or any(cpu_id < 0 or cpu_id >= logical_cpus for cpu_id in cpuset_ids)
    ):
        raise RuntimeError("effective process/cgroup CPU set is invalid")
    quota = cgroup["cpu_quota_cores"]
    effective_cpu = min(
        [float(len(effective_ids))] + ([] if quota is None else [float(quota)])
    )
    memory_limit = cgroup["memory_limit_bytes"]
    effective_memory = min(
        [mem["MemTotal"]] + ([] if memory_limit is None else [memory_limit])
    )
    filesystem_identity = _filesystem_identity(checkpoint_path)
    gpu_static, gpu_utilization, free_gpu_memory = _gpu_row()
    filesystem = os.statvfs(checkpoint_path)
    checkpoint_free = filesystem.f_bavail * filesystem.f_frsize
    if checkpoint_free < 0:
        raise RuntimeError("checkpoint filesystem reported negative free space")
    cgroup_memory_available = cgroup["memory_available_bytes"]
    static = {
        "instance": _instance_binding(cgroup["path"]),
        "cpu": {
            "model": cpu_model,
            "logical_cpus": logical_cpus,
            "process_affinity_cpu_ids": affinity_ids,
            "cgroup_cpuset_cpu_ids": cpuset_ids,
            "effective_cpu_ids": effective_ids,
            "allowed_logical_cpus": len(effective_ids),
            "cgroup_cpuset_logical_cpus": cgroup["cpuset_logical_cpus"],
            "cpu_quota_cores": quota,
            "effective_cpu_capacity": effective_cpu,
        },
        "memory": {
            "total_bytes": mem["MemTotal"],
            "cgroup_limit_bytes": memory_limit,
            "effective_capacity_bytes": effective_memory,
        },
        "checkpoint_filesystem": filesystem_identity,
        "gpu": gpu_static,
    }
    live: dict[str, Any] = {
        "observed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "load_1m": os.getloadavg()[0],
        "host_available_memory_bytes": mem["MemAvailable"],
        "cgroup_available_memory_bytes": cgroup_memory_available,
        "available_memory_bytes": min(
            value
            for value in (mem["MemAvailable"], cgroup_memory_available)
            if value is not None
        ),
        "checkpoint_free_bytes": checkpoint_free,
        "gpu_utilization_percent": gpu_utilization,
        "free_gpu_memory_mib": free_gpu_memory,
        "compute_process_pids": _compute_processes(gpu_static["uuid"]),
    }
    if storage_probe_bytes is not None:
        live["checkpoint_storage_probe"] = _storage_probe(
            checkpoint_path, byte_count=storage_probe_bytes
        )
        live["observed_at_utc"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    return {"static": static, "live": live}


def build_manifest(
    *,
    checkpoint_path: Path,
    preflight_policy: dict[str, Any],
    bootstrap_receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Capture and seal a static host identity before timing work.

    Schema 3 is the executable Week-5 form and binds the recaptured host layout,
    local container image ID, and staged source identity from the bootstrap
    receipt.  Schema 2 remains validation-only compatibility evidence.
    """

    policy = dict(_object(preflight_policy, "host preflight policy"))
    supplied_tool = policy.get("storage_probe_tool_sha256")
    if supplied_tool is None:
        policy["storage_probe_tool_sha256"] = _module_sha256()
    elif supplied_tool != _module_sha256():
        raise ValueError("storage probe tool identity differs from this module")
    observed = observe(checkpoint_path)
    body: dict[str, Any] = {
        "schema": 3 if bootstrap_receipt_path is not None else 2,
        "kind": "forge-krea-host-execution-identity",
        "static": observed["static"],
        "preflight_policy": policy,
    }
    if bootstrap_receipt_path is not None:
        receipt_path = Path(os.path.abspath(os.path.expanduser(bootstrap_receipt_path)))
        raw = receipt_path.read_bytes()
        receipt = _object(json.loads(raw), "bootstrap receipt")
        try:
            from . import krea_host_bootstrap
        except ImportError:  # pragma: no cover - direct script execution.
            import krea_host_bootstrap  # type: ignore[no-redef]

        krea_host_bootstrap.validate_receipt(receipt, recapture=True)
        receipt_checkpoint = Path(receipt["spec"]["sources"]["checkpoints"])
        mounted_checkpoint = Path("/app/checkpoints")
        checkpoint_real = Path(checkpoint_path).resolve(strict=True)
        if not any(
            candidate.exists() and os.path.samefile(checkpoint_real, candidate)
            for candidate in (receipt_checkpoint, mounted_checkpoint)
        ):
            raise ValueError(
                "checkpoint path is not the bootstrap-bound checkpoint filesystem"
            )
        body["bootstrap_receipt"] = {
            "path": str(receipt_path),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "receipt_sha256": receipt["receipt_sha256"],
            "container_image_sha256": receipt["spec"]["runtime"][
                "container_image_sha256"
            ],
        }
    manifest = {
        **body,
        "host_execution_identity_sha256": canonical_sha256(body),
    }
    validate_manifest(manifest)
    return manifest


def _validate_storage_probe(
    value: Any, *, manifest: dict[str, Any], observation_time: str
) -> dict[str, Any]:
    probe = _object(value, "storage probe")
    _exact(
        probe,
        {
            "checkpoint_path",
            "device_major_minor",
            "bytes",
            "content_sha256",
            "write_s",
            "fsync_s",
            "read_s",
            "write_mib_s",
            "read_mib_s",
            "cache_drop_requested",
            "tool_sha256",
            "observed_at_utc",
        },
        "storage probe",
    )
    policy = manifest["preflight_policy"]
    filesystem = manifest["static"]["checkpoint_filesystem"]
    if (
        probe["checkpoint_path"] != filesystem["checkpoint_path"]
        or probe["device_major_minor"] != filesystem["device_major_minor"]
        or probe["bytes"] != policy["storage_probe_bytes"]
        or probe["tool_sha256"] != policy["storage_probe_tool_sha256"]
    ):
        raise RuntimeError("storage probe is not bound to this host/filesystem/policy")
    if _digest(
        probe["content_sha256"], "storage probe content_sha256"
    ) != _storage_probe_content_sha256(policy["storage_probe_bytes"]):
        raise RuntimeError(
            "storage probe content does not match the pinned I/O pattern"
        )
    if not isinstance(probe["cache_drop_requested"], bool):
        raise RuntimeError("storage probe cache_drop_requested is malformed")
    write_s = _finite_number(probe["write_s"], "storage probe write_s")
    _finite_number(probe["fsync_s"], "storage probe fsync_s")
    read_s = _finite_number(probe["read_s"], "storage probe read_s")
    write_rate = _finite_number(probe["write_mib_s"], "storage probe write_mib_s")
    read_rate = _finite_number(probe["read_mib_s"], "storage probe read_mib_s")
    mib = probe["bytes"] / (1024 * 1024)
    if not math.isclose(
        write_rate, mib / write_s, rel_tol=1e-9, abs_tol=1e-9
    ) or not math.isclose(read_rate, mib / read_s, rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError(
            "storage probe throughput is inconsistent with its durations"
        )
    probe_time = _canonical_timestamp(
        probe["observed_at_utc"],
        "storage probe observed_at_utc",
        maximum_age_s=300,
    )
    if probe_time > observation_time:
        raise RuntimeError("storage probe timestamp follows the enclosing observation")
    return probe


def _validate_live_observation(
    observed: dict[str, Any], *, manifest: dict[str, Any], require_probe: bool
) -> dict[str, Any]:
    observed = _object(observed, "host observation")
    _exact(observed, {"static", "live"}, "host observation")
    if observed["static"] != manifest["static"]:
        raise RuntimeError(
            "host/storage/GPU identity drifted from the measured profile"
        )
    live = _object(observed["live"], "host live observation")
    live_keys = {
        "observed_at_utc",
        "load_1m",
        "host_available_memory_bytes",
        "cgroup_available_memory_bytes",
        "available_memory_bytes",
        "checkpoint_free_bytes",
        "gpu_utilization_percent",
        "free_gpu_memory_mib",
        "compute_process_pids",
    }
    if require_probe:
        live_keys.add("checkpoint_storage_probe")
    _exact(live, live_keys, "host live observation")
    observation_time = _canonical_timestamp(
        live["observed_at_utc"], "host observed_at_utc", maximum_age_s=300
    )
    _finite_number(live["load_1m"], "host load_1m", positive=False)
    host_available = _nonnegative_int(
        live["host_available_memory_bytes"], "host_available_memory_bytes"
    )
    cgroup_available = live["cgroup_available_memory_bytes"]
    if cgroup_available is not None:
        cgroup_available = _nonnegative_int(
            cgroup_available, "cgroup_available_memory_bytes"
        )
    available = _nonnegative_int(
        live["available_memory_bytes"], "available_memory_bytes"
    )
    if available != min(
        value for value in (host_available, cgroup_available) if value is not None
    ):
        raise RuntimeError(
            "available memory does not reflect effective cgroup headroom"
        )
    static_memory = manifest["static"]["memory"]
    if host_available > static_memory["total_bytes"] or (
        cgroup_available is not None
        and static_memory["cgroup_limit_bytes"] is not None
        and cgroup_available > static_memory["cgroup_limit_bytes"]
    ):
        raise RuntimeError("live memory headroom exceeds its bound capacity")
    _nonnegative_int(live["checkpoint_free_bytes"], "checkpoint_free_bytes")
    utilization = _finite_number(
        live["gpu_utilization_percent"], "gpu_utilization_percent", positive=False
    )
    if utilization > 100:
        raise RuntimeError("GPU utilization exceeds 100 percent")
    free_gpu = _nonnegative_int(live["free_gpu_memory_mib"], "free_gpu_memory_mib")
    if free_gpu > manifest["static"]["gpu"]["total_memory_mib"]:
        raise RuntimeError("free GPU memory exceeds total GPU memory")
    pids = live["compute_process_pids"]
    if not isinstance(pids, list) or any(
        isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in pids
    ):
        raise RuntimeError("GPU compute-process occupancy is malformed")
    if pids != sorted(pids) or len(pids) != len(set(pids)):
        raise RuntimeError("GPU compute-process occupancy is malformed")
    if require_probe:
        _validate_storage_probe(
            live["checkpoint_storage_probe"],
            manifest=manifest,
            observation_time=observation_time,
        )
    return observed


def verify_live(manifest: dict[str, Any], *, checkpoint_path: Path) -> dict[str, Any]:
    manifest = validate_manifest(manifest)
    if manifest.get("schema") == 3:
        bootstrap_runtime(manifest, recapture=True)
    observed = observe(
        checkpoint_path,
        storage_probe_bytes=manifest["preflight_policy"]["storage_probe_bytes"],
    )
    _validate_live_observation(observed, manifest=manifest, require_probe=True)
    policy = manifest["preflight_policy"]
    live = observed["live"]
    effective_cpus = float(manifest["static"]["cpu"]["effective_cpu_capacity"])
    load_per_cpu = live["load_1m"] / effective_cpus
    failures = {}
    if load_per_cpu > policy["maximum_load_per_effective_cpu"]:
        failures["load_per_cpu"] = load_per_cpu
    if live["available_memory_bytes"] < policy["minimum_available_memory_bytes"]:
        failures["available_memory_bytes"] = live["available_memory_bytes"]
    if live["checkpoint_free_bytes"] < policy["minimum_checkpoint_free_bytes"]:
        failures["checkpoint_free_bytes"] = live["checkpoint_free_bytes"]
    if live["gpu_utilization_percent"] > policy["maximum_gpu_utilization_percent"]:
        failures["gpu_utilization_percent"] = live["gpu_utilization_percent"]
    if live["free_gpu_memory_mib"] < policy["minimum_free_gpu_memory_mib"]:
        failures["free_gpu_memory_mib"] = live["free_gpu_memory_mib"]
    foreign = [pid for pid in live["compute_process_pids"] if pid != os.getpid()]
    if len(foreign) > policy["maximum_foreign_compute_processes"]:
        failures["foreign_compute_process_pids"] = foreign
    probe = live["checkpoint_storage_probe"]
    if probe["write_mib_s"] < policy["minimum_checkpoint_write_mib_s"]:
        failures["checkpoint_write_mib_s"] = probe["write_mib_s"]
    if probe["read_mib_s"] < policy["minimum_checkpoint_read_mib_s"]:
        failures["checkpoint_read_mib_s"] = probe["read_mib_s"]
    if probe["fsync_s"] > policy["maximum_checkpoint_fsync_s"]:
        failures["checkpoint_fsync_s"] = probe["fsync_s"]
    if failures:
        raise RuntimeError(f"host live preflight thresholds failed: {failures}")
    return observed


def verify_static(manifest: dict[str, Any], *, checkpoint_path: Path) -> dict[str, Any]:
    """Recheck identity without imposing idle-host thresholds after a run."""

    manifest = validate_manifest(manifest)
    if manifest.get("schema") == 3:
        bootstrap_runtime(manifest, recapture=True)
    observed = observe(checkpoint_path)
    _validate_live_observation(observed, manifest=manifest, require_probe=False)
    return observed
