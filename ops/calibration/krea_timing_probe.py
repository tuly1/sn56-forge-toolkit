#!/usr/bin/env python3
"""Receipt-clock Krea timing producer used before any final arm plan exists.

The child workload reports only begin/end markers over a private Unix datagram
socket.  This parent process timestamps receipt with ``monotonic_ns`` and seals
the exact command, executable bytes, preflight, and intervals.  A child cannot
hand the profile builder a favorable duration or an opaque sample count.

First-GPU sequence (all outputs are exclusive-create, canonical JSON):

1. Seal/approve a ``forge-krea-bootstrap-timing-probe-plan`` with
   :mod:`krea_execution_plan`; it has no throughput profile.
2. Freeze a named-human margin policy.  Its approval timestamp must precede
   every timing sample governed by it.
3. Run ``capture`` for the predeclared measurement command.  Descendants call
   ``emit`` around real startup/update/save/finalize/upload spans.
4. ``assemble-raw`` combines timing captures and recomputes unit counts.
5. Run one separately-designated held-out capture and ``build-e2e``.
6. Run ``build-profile``; it rejects a margin approved after the first raw
   capture.  Only now
   can a final execution plan and its independent human approval be sealed.
7. After each arm runs, create the separate post-run natural-completion
   certificate; it is evidence, never a prerequisite for its own execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable

try:
    from . import krea_budget
    from . import krea_execution_plan
    from . import krea_host_identity
    from . import krea_provenance
except ImportError:  # pragma: no cover - direct script execution.
    import krea_budget  # type: ignore[no-redef]
    import krea_execution_plan  # type: ignore[no-redef]
    import krea_host_identity  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]


_SOCKET_ENV = "FORGE_KREA_TIMING_SOCKET"
_CONTRACT_ENV = "FORGE_KREA_TIMING_PROBE_CONTRACT_SHA256"
_CAPTURE_ENV = "FORGE_KREA_TIMING_CAPTURE_ID"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_UNIT_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_METRICS = frozenset(
    {"startup", "optimizer_update", "checkpoint_save", "finalization", "upload"}
)
_SYSTEMD_RUN_PATH = Path("/usr/bin/systemd-run")
_SYSTEMCTL_PATH = Path("/usr/bin/systemctl")
_SCOPE_TERM_GRACE_S = 10.0


class _CaptureCancellation(BaseException):
    """Turn TERM/HUP into a cleanup-bearing control-flow edge."""

    def __init__(self, signum: int):
        super().__init__(f"timing capture received signal {signum}")
        self.signum = signum


def _canonical_bytes(value: Any) -> bytes:
    return krea_provenance.canonical_bytes(value)


def _sha256_file(path: Path) -> str:
    return krea_provenance.file_sha256(path)


def _safe_file(path: Path, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    return path


def _load_canonical(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = _safe_file(path, label)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if raw != _canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline")
    return value, hashlib.sha256(raw).hexdigest()


def _publish(path: Path, value: dict[str, Any]) -> None:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"output has a symlink ancestor: {current}")
        current = current.parent
    payload = _canonical_bytes(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} must be a conservative identifier")
    return value


def marker(
    *, observation_id: str, metric: str, state: str, units: int = 1
) -> dict[str, Any]:
    """Build a marker from the inherited capture environment."""

    socket_path = os.environ.get(_SOCKET_ENV)
    contract = os.environ.get(_CONTRACT_ENV)
    capture_id = os.environ.get(_CAPTURE_ENV)
    if not socket_path or not contract or not capture_id:
        raise RuntimeError("timing marker emitted outside a timing capture")
    if not _SHA256.fullmatch(contract):
        raise RuntimeError("timing capture contract environment is invalid")
    _safe_id(capture_id, "capture_id")
    _safe_id(observation_id, "observation_id")
    if metric not in _METRICS or state not in {"begin", "end"}:
        raise ValueError("timing marker metric/state is invalid")
    if isinstance(units, bool) or not isinstance(units, int) or units <= 0:
        raise ValueError("timing marker units must be a positive integer")
    return {
        "schema": 1,
        "kind": "forge-krea-timing-marker",
        "probe_contract_sha256": contract,
        "capture_id": capture_id,
        "observation_id": observation_id,
        "metric": metric,
        "state": state,
        "units": units,
    }


def emit_marker(
    *, observation_id: str, metric: str, state: str, units: int = 1
) -> None:
    value = marker(
        observation_id=observation_id, metric=metric, state=state, units=units
    )
    payload = _canonical_bytes(value)
    if len(payload) > 4096:
        raise RuntimeError("timing marker exceeds atomic datagram limit")
    address = os.environ[_SOCKET_ENV]
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
        client.sendto(payload, address)


def _validate_marker(
    value: Any, *, contract_sha: str, capture_id: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "kind",
        "probe_contract_sha256",
        "capture_id",
        "observation_id",
        "metric",
        "state",
        "units",
    }:
        raise ValueError("timing marker schema mismatch")
    units = value["units"]
    if (
        value["schema"] != 1
        or value["kind"] != "forge-krea-timing-marker"
        or value["probe_contract_sha256"] != contract_sha
        or value["capture_id"] != capture_id
        or not isinstance(value["observation_id"], str)
        or not _SAFE_ID.fullmatch(value["observation_id"])
        or value["metric"] not in _METRICS
        or value["state"] not in {"begin", "end"}
        or isinstance(units, bool)
        or not isinstance(units, int)
        or units <= 0
    ):
        raise ValueError("timing marker is invalid or escaped its capture")
    return value


def _pair_markers(
    received: list[tuple[int, bytes]], *, contract_sha: str, capture_id: str
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    stream_rows: list[dict[str, Any]] = []
    open_rows: dict[str, tuple[dict[str, Any], int]] = {}
    samples: dict[str, list[dict[str, Any]]] = {
        metric: [] for metric in sorted(_METRICS)
    }
    seen: set[str] = set()
    for sequence, (receipt_ns, raw) in enumerate(received):
        try:
            decoded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("timing marker is not JSON") from exc
        value = _validate_marker(
            decoded, contract_sha=contract_sha, capture_id=capture_id
        )
        event_sha = hashlib.sha256(raw).hexdigest()
        stream_rows.append(
            {
                "sequence": sequence,
                "received_monotonic_ns": receipt_ns,
                "marker_sha256": event_sha,
                "marker": value,
            }
        )
        observation_id = value["observation_id"]
        if value["state"] == "begin":
            if observation_id in open_rows or observation_id in seen:
                raise ValueError("timing observation begins more than once")
            open_rows[observation_id] = (value, receipt_ns)
            continue
        if observation_id not in open_rows:
            raise ValueError("timing observation ended without a begin marker")
        begin, started = open_rows.pop(observation_id)
        if begin["metric"] != value["metric"] or begin["units"] != value["units"]:
            raise ValueError("timing end marker contradicts its begin marker")
        if receipt_ns <= started:
            raise ValueError("timing receipt clock did not advance")
        seen.add(observation_id)
        duration = (receipt_ns - started) / 1_000_000_000
        samples[value["metric"]].append(
            {
                "capture_id": capture_id,
                "observation_id": observation_id,
                "duration_s": duration,
                "units": value["units"],
                "started_monotonic_ns": started,
                "ended_monotonic_ns": receipt_ns,
            }
        )
    if open_rows:
        raise ValueError(f"timing observations never ended: {sorted(open_rows)}")
    stream_sha = krea_provenance.canonical_sha256(stream_rows)
    for rows in samples.values():
        rows.sort(key=lambda row: row["observation_id"])
    return samples, stream_sha


def _resolve_executable(argv0: str) -> Path:
    candidate = shutil.which(argv0) if "/" not in argv0 else argv0
    if candidate is None:
        raise FileNotFoundError(f"timing command is not executable: {argv0}")
    path = Path(candidate).resolve(strict=True)
    if path.is_symlink() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"timing command is not a regular executable: {path}")
    return path


def _scope_unit_name(capture_id: str, *, driver_pid: int | None = None) -> str:
    """Return the sole transient-scope name for one capture driver."""

    capture_id = _safe_id(capture_id, "capture_id")
    pid = os.getpid() if driver_pid is None else driver_pid
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise RuntimeError(f"unsafe timing capture driver PID: {pid!r}")
    suffix = hashlib.sha256(capture_id.encode("utf-8")).hexdigest()[:12]
    unit = f"forge-krea-timing-{pid}-{suffix}"
    if not _UNIT_COMPONENT.fullmatch(unit):
        raise RuntimeError("generated an unsafe timing systemd unit name")
    return unit


def _systemd_prerequisites() -> tuple[Path, Path]:
    """Resolve the rootful systemd boundary or fail before child launch."""

    if sys.platform != "linux":
        raise RuntimeError("timing capture requires a real Linux systemd host")
    if os.geteuid() != 0:
        raise PermissionError("timing capture requires rootful transient scopes")
    if not Path("/run/systemd/system").is_dir():
        raise RuntimeError("systemd is not PID 1 on the timing capture host")
    systemd_run = _resolve_executable(str(_SYSTEMD_RUN_PATH))
    systemctl = _resolve_executable(str(_SYSTEMCTL_PATH))
    return systemd_run, systemctl


def _scope_status(unit: str, *, systemctl_path: Path) -> str:
    """Return active, inactive, or collected with recursive cgroup proof."""

    if not _UNIT_COMPONENT.fullmatch(unit):
        raise RuntimeError(f"unsafe timing scope unit: {unit!r}")
    result = subprocess.run(
        [
            str(systemctl_path),
            "show",
            "--no-pager",
            "--all",
            "--property=LoadState",
            "--property=ActiveState",
            "--property=ControlGroup",
            f"{unit}.scope",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"timing scope state is indeterminate for {unit}: "
            f"rc={result.returncode}, stderr={result.stderr[-1000:]}"
        )
    properties: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            properties[key] = value
    if set(properties) != {"LoadState", "ActiveState", "ControlGroup"}:
        raise RuntimeError(f"systemd returned incomplete timing-scope state for {unit}")
    load = properties["LoadState"]
    active = properties["ActiveState"]
    control_group = properties["ControlGroup"]
    if load == "not-found":
        if active != "inactive" or control_group:
            raise RuntimeError(f"collected timing scope still exposes state: {unit}")
        return "collected"
    if active not in {
        "active",
        "reloading",
        "activating",
        "deactivating",
        "inactive",
        "failed",
    }:
        raise RuntimeError(
            f"unrecognized timing-scope ActiveState for {unit}: {active}"
        )
    populated: str | None = None
    if control_group:
        cgroup_root = Path("/sys/fs/cgroup").resolve(strict=True)
        events = cgroup_root / control_group.lstrip("/") / "cgroup.events"
        if (
            events.is_symlink()
            or not events.is_file()
            or not events.resolve(strict=True).is_relative_to(cgroup_root)
        ):
            raise RuntimeError(f"cannot prove recursive timing-cgroup state: {unit}")
        values: dict[str, str] = {}
        for line in events.read_text(encoding="ascii").splitlines():
            parts = line.split()
            if len(parts) == 2:
                values[parts[0]] = parts[1]
        populated = values.get("populated")
        if populated not in {"0", "1"}:
            raise RuntimeError(f"invalid timing cgroup.events for {unit}")
    if active in {"active", "reloading", "activating", "deactivating"}:
        return "active"
    if populated == "1":
        return "active"
    return "inactive"


def _scope_signal(
    unit: str, stop_signal: signal.Signals, *, systemctl_path: Path
) -> None:
    if not _UNIT_COMPONENT.fullmatch(unit):
        raise RuntimeError(f"unsafe timing scope unit: {unit!r}")
    result = subprocess.run(
        [
            str(systemctl_path),
            "kill",
            "--kill-who=all",
            f"--signal={stop_signal.name}",
            f"{unit}.scope",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    if result.returncode != 0 and _scope_status(
        unit, systemctl_path=systemctl_path
    ) not in {"inactive", "collected"}:
        raise RuntimeError(
            f"could not signal timing scope {unit}: rc={result.returncode}, "
            f"stderr={result.stderr[-1000:]}"
        )


def _validated_process_group(process: subprocess.Popen[Any]) -> int:
    pid = process.pid
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise RuntimeError(f"unsafe timing systemd-run PID: {pid!r}")
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError as exc:
        raise RuntimeError(
            "systemd-run exited before containment was validated"
        ) from exc
    if pgid != pid or pgid == os.getpgrp() or pgid <= 1:
        raise RuntimeError(
            f"unsafe timing systemd-run process group: pid={pid}, pgid={pgid}"
        )
    return pgid


def _signal_live_process_group(
    process: subprocess.Popen[Any], pgid: int, stop_signal: signal.Signals
) -> None:
    if process.poll() is not None:
        return
    try:
        current = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    if current != pgid or current != process.pid or current == os.getpgrp():
        raise RuntimeError("timing systemd-run process-group identity changed")
    try:
        os.killpg(pgid, stop_signal)
    except ProcessLookupError:
        pass


def _process_group_is_empty(pgid: int) -> bool:
    if isinstance(pgid, bool) or not isinstance(pgid, int) or pgid <= 1:
        raise RuntimeError(f"unsafe timing process group: {pgid!r}")
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError as exc:
        raise RuntimeError(
            f"cannot prove timing process group {pgid} is empty"
        ) from exc
    return False


def _wait_scope_collected(unit: str, *, systemctl_path: Path, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _scope_status(unit, systemctl_path=systemctl_path) == "collected":
            return True
        time.sleep(0.05)
    return _scope_status(unit, systemctl_path=systemctl_path) == "collected"


def _terminate_scope_and_client(
    process: subprocess.Popen[Any],
    *,
    pgid: int,
    unit: str,
    systemctl_path: Path,
    term_grace_s: float,
) -> None:
    """Terminate the recursive cgroup and reap its systemd-run client."""

    cleanup_error: BaseException | None = None
    try:
        _scope_signal(unit, signal.SIGTERM, systemctl_path=systemctl_path)
    except BaseException as exc:
        cleanup_error = exc
    try:
        _signal_live_process_group(process, pgid, signal.SIGTERM)
    except BaseException as exc:
        cleanup_error = cleanup_error or exc
    try:
        process.wait(timeout=term_grace_s)
    except subprocess.TimeoutExpired:
        try:
            _scope_signal(unit, signal.SIGKILL, systemctl_path=systemctl_path)
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
        try:
            _signal_live_process_group(process, pgid, signal.SIGKILL)
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
        try:
            process.wait(timeout=max(1.0, term_grace_s))
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
    try:
        if _scope_status(unit, systemctl_path=systemctl_path) == "active":
            _scope_signal(unit, signal.SIGKILL, systemctl_path=systemctl_path)
    except BaseException as exc:
        cleanup_error = cleanup_error or exc
    deadline = time.monotonic() + max(1.0, term_grace_s)
    while not _process_group_is_empty(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if not _process_group_is_empty(pgid):
        cleanup_error = cleanup_error or RuntimeError(
            f"timing systemd-run group still has survivors: {pgid}"
        )
    try:
        if not _wait_scope_collected(
            unit,
            systemctl_path=systemctl_path,
            timeout_s=max(1.0, term_grace_s),
        ):
            raise RuntimeError(f"timing scope was not collected: {unit}")
    except BaseException as exc:
        cleanup_error = cleanup_error or exc
    if cleanup_error is not None:
        raise RuntimeError(
            f"could not prove complete timing-capture cleanup: {unit}"
        ) from cleanup_error


def _terminate_scope_without_validated_group(
    process: subprocess.Popen[Any],
    *,
    unit: str,
    systemctl_path: Path,
    term_grace_s: float,
) -> None:
    """Clean the cgroup when the systemd-run client PGID is untrusted."""

    cleanup_error: BaseException | None = None
    try:
        _scope_signal(unit, signal.SIGTERM, systemctl_path=systemctl_path)
    except BaseException as exc:
        cleanup_error = exc
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=term_grace_s)
    except subprocess.TimeoutExpired:
        try:
            _scope_signal(unit, signal.SIGKILL, systemctl_path=systemctl_path)
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=max(1.0, term_grace_s))
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
    try:
        if not _wait_scope_collected(
            unit,
            systemctl_path=systemctl_path,
            timeout_s=max(1.0, term_grace_s),
        ):
            raise RuntimeError(f"unvalidated timing scope was not collected: {unit}")
    except BaseException as exc:
        cleanup_error = cleanup_error or exc
    if cleanup_error is not None:
        raise RuntimeError(
            f"could not prove cleanup after containment validation failed: {unit}"
        ) from cleanup_error


def _validate_containment_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "mode",
        "unit",
        "systemd_run_path",
        "systemd_run_sha256",
        "systemctl_path",
        "systemctl_sha256",
        "kill_mode",
        "timeout_stop_s",
        "runtime_max_s",
        "scope_observed_active",
        "recursive_cleanup_proven",
        "unit_collected",
    }:
        raise ValueError("timing containment receipt schema mismatch")
    if (
        value["schema"] != 1
        or value["mode"] != "systemd_transient_scope"
        or not isinstance(value["unit"], str)
        or not value["unit"].endswith(".scope")
        or not _UNIT_COMPONENT.fullmatch(value["unit"][:-6])
        or value["kill_mode"] != "control-group"
        or value["timeout_stop_s"] != _SCOPE_TERM_GRACE_S
        or isinstance(value["runtime_max_s"], bool)
        or not isinstance(value["runtime_max_s"], (int, float))
        or not math.isfinite(float(value["runtime_max_s"]))
        or float(value["runtime_max_s"]) <= _SCOPE_TERM_GRACE_S
        or value["scope_observed_active"] is not True
        or value["recursive_cleanup_proven"] is not True
        or value["unit_collected"] is not True
    ):
        raise ValueError("timing containment receipt is not fail-closed")
    for prefix in ("systemd_run", "systemctl"):
        path = _resolve_executable(value[f"{prefix}_path"])
        digest = value[f"{prefix}_sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"timing containment {prefix} digest is invalid")
        if _sha256_file(path) != digest:
            raise ValueError(f"timing containment {prefix} binary changed")
    return value


def _run_in_transient_scope(
    command: list[str],
    *,
    env: dict[str, str],
    timeout_s: float,
    capture_id: str,
) -> tuple[int, dict[str, Any]]:
    """Run one exact command in a rootful recursive transient scope.

    ``systemd-run --scope`` executes in the caller's environment, so the
    private timing socket is inherited without serializing credentials into a
    service unit.  A scope cgroup contains descendants even after ``setsid``.
    RuntimeMaxSec is a manager-side backstop if this Python producer is killed.
    """

    systemd_run, systemctl = _systemd_prerequisites()
    unit = _scope_unit_name(capture_id)
    runtime_max_s = float(timeout_s) + _SCOPE_TERM_GRACE_S
    wrapped = [
        str(systemd_run),
        "--quiet",
        "--scope",
        "--collect",
        f"--unit={unit}",
        "--property=KillMode=control-group",
        f"--property=TimeoutStopSec={_SCOPE_TERM_GRACE_S}s",
        f"--property=RuntimeMaxSec={runtime_max_s}s",
        "--",
        *command,
    ]
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("timing containment must own main-thread signal cleanup")
    previous_handlers: dict[signal.Signals, Any] = {}

    def cancel(signum: int, _frame: Any) -> None:
        raise _CaptureCancellation(signum)

    for stop_signal in (signal.SIGTERM, signal.SIGHUP):
        previous_handlers[stop_signal] = signal.getsignal(stop_signal)
        signal.signal(stop_signal, cancel)
    process: subprocess.Popen[Any] | None = None
    try:
        process = subprocess.Popen(
            wrapped,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        try:
            pgid = _validated_process_group(process)
        except BaseException:
            _terminate_scope_without_validated_group(
                process,
                unit=unit,
                systemctl_path=systemctl,
                term_grace_s=_SCOPE_TERM_GRACE_S,
            )
            raise
        observed_active = False
        activation_deadline = time.monotonic() + min(10.0, float(timeout_s))
        try:
            while process.poll() is None and time.monotonic() < activation_deadline:
                if _scope_status(unit, systemctl_path=systemctl) == "active":
                    observed_active = True
                    break
                time.sleep(0.05)
            if not observed_active:
                raise RuntimeError("timing transient scope was never observed active")
            try:
                returncode = int(process.wait(timeout=float(timeout_s)))
            except subprocess.TimeoutExpired as exc:
                _terminate_scope_and_client(
                    process,
                    pgid=pgid,
                    unit=unit,
                    systemctl_path=systemctl,
                    term_grace_s=_SCOPE_TERM_GRACE_S,
                )
                raise TimeoutError(
                    "timing probe exceeded its sealed timeout; recursive scope cleaned"
                ) from exc
        except BaseException:
            needs_cleanup = process.poll() is None
            if not needs_cleanup:
                try:
                    needs_cleanup = (
                        _scope_status(unit, systemctl_path=systemctl) != "collected"
                    )
                except BaseException:
                    needs_cleanup = True
            if needs_cleanup:
                _terminate_scope_and_client(
                    process,
                    pgid=pgid,
                    unit=unit,
                    systemctl_path=systemctl,
                    term_grace_s=_SCOPE_TERM_GRACE_S,
                )
            raise
        if not _process_group_is_empty(pgid):
            _terminate_scope_and_client(
                process,
                pgid=pgid,
                unit=unit,
                systemctl_path=systemctl,
                term_grace_s=_SCOPE_TERM_GRACE_S,
            )
            raise RuntimeError("timing systemd-run process group survived normal exit")
        if not _wait_scope_collected(
            unit,
            systemctl_path=systemctl,
            timeout_s=_SCOPE_TERM_GRACE_S,
        ):
            _terminate_scope_and_client(
                process,
                pgid=pgid,
                unit=unit,
                systemctl_path=systemctl,
                term_grace_s=_SCOPE_TERM_GRACE_S,
            )
            raise RuntimeError("timing scope survived normal exit")
        receipt = {
            "schema": 1,
            "mode": "systemd_transient_scope",
            "unit": f"{unit}.scope",
            "systemd_run_path": str(systemd_run),
            "systemd_run_sha256": _sha256_file(systemd_run),
            "systemctl_path": str(systemctl),
            "systemctl_sha256": _sha256_file(systemctl),
            "kill_mode": "control-group",
            "timeout_stop_s": _SCOPE_TERM_GRACE_S,
            "runtime_max_s": runtime_max_s,
            "scope_observed_active": True,
            "recursive_cleanup_proven": True,
            "unit_collected": True,
        }
        _validate_containment_receipt(receipt)
        return returncode, receipt
    finally:
        for stop_signal, previous in previous_handlers.items():
            signal.signal(stop_signal, previous)


def capture(
    *,
    probe_plan_path: Path,
    probe_approval_path: Path,
    checkpoint_path: Path,
    output: Path,
    command: list[str],
    capture_id: str,
    measurement_role: str,
    timeout_s: float,
) -> dict[str, Any]:
    plan, plan_file_sha = _load_canonical(probe_plan_path, "timing probe plan")
    approval, approval_file_sha = _load_canonical(
        probe_approval_path, "timing probe approval"
    )
    resolved = krea_execution_plan.validate_timing_probe_plan(plan)
    krea_execution_plan.validate_timing_probe_approval(approval, plan=plan)
    capture_id = _safe_id(capture_id, "capture_id")
    if measurement_role not in {"timing_measurement", "held_out_end_to_end"}:
        raise ValueError("capture measurement role is invalid")
    if plan["probe_schedule"]["measurement_role"] != "timing_and_heldout":
        raise ValueError("probe plan does not authorize both capture roles")
    if command != plan["command_argv"]:
        raise ValueError("executed argv differs from the sealed timing probe command")
    if Path(command[3]).expanduser().resolve(strict=True) != _safe_file(
        probe_plan_path, "timing probe plan"
    ) or Path(command[5]).expanduser().resolve(strict=True) != _safe_file(
        probe_approval_path, "timing probe approval"
    ):
        raise ValueError("timing command does not consume these exact probe controls")
    if (
        not isinstance(timeout_s, (int, float))
        or isinstance(timeout_s, bool)
        or timeout_s <= 0
    ):
        raise ValueError("capture timeout must be positive")
    if timeout_s > float(plan["probe_schedule"]["hard_budget_s"]):
        raise ValueError("capture timeout exceeds the sealed hard budget")
    checkpoint_path = Path(os.path.abspath(os.path.expanduser(checkpoint_path)))
    preflight = krea_host_identity.verify_live(
        resolved["host_execution_manifest"], checkpoint_path=checkpoint_path
    )
    preflight_sha = krea_provenance.canonical_sha256(preflight)
    executable = _resolve_executable(command[0])

    received: list[tuple[int, bytes]] = []
    receive_errors: list[BaseException] = []
    stop = threading.Event()
    with tempfile.TemporaryDirectory(prefix="forge-krea-timing-") as temporary:
        socket_path = str(Path(temporary) / "events.sock")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        server.bind(socket_path)
        server.settimeout(0.1)

        def receiver() -> None:
            try:
                while not stop.is_set():
                    try:
                        payload = server.recv(4096)
                    except socket.timeout:
                        continue
                    received.append((time.monotonic_ns(), payload))
            except BaseException as exc:  # preserve receiver failures for parent
                receive_errors.append(exc)

        thread = threading.Thread(target=receiver, name="krea-timing-receiver")
        thread.start()
        env = dict(os.environ)
        env[_SOCKET_ENV] = socket_path
        env[_CONTRACT_ENV] = plan["probe_contract_sha256"]
        env[_CAPTURE_ENV] = capture_id
        started_unix_ns = time.time_ns()
        started_monotonic_ns = time.monotonic_ns()
        try:
            returncode, containment = _run_in_transient_scope(
                command,
                env=env,
                timeout_s=float(timeout_s),
                capture_id=capture_id,
            )
        finally:
            ended_monotonic_ns = time.monotonic_ns()
            ended_unix_ns = time.time_ns()
            # Let a final datagram already queued in the kernel be drained.
            time.sleep(0.12)
            stop.set()
            thread.join(timeout=2)
            server.close()
        if thread.is_alive() or receive_errors:
            raise RuntimeError("timing marker receiver did not shut down cleanly")
    if returncode != 0:
        raise RuntimeError(f"timing probe command failed with status {returncode}")
    samples, stream_sha = _pair_markers(
        received,
        contract_sha=plan["probe_contract_sha256"],
        capture_id=capture_id,
    )
    payload = {
        "schema": 1,
        "kind": "forge-krea-timing-command-capture",
        "capture_id": capture_id,
        "measurement_role": measurement_role,
        "probe_contract": {
            "path": str(_safe_file(probe_plan_path, "timing probe plan")),
            "sha256": plan_file_sha,
            "probe_contract_sha256": plan["probe_contract_sha256"],
        },
        "probe_approval": {
            "path": str(_safe_file(probe_approval_path, "timing probe approval")),
            "sha256": approval_file_sha,
            "approval_sha256": approval["approval_sha256"],
        },
        "execution_envelope": plan["execution_envelope"],
        "measurement_tool_sha256": _sha256_file(Path(__file__).resolve(strict=True)),
        "containment": containment,
        "host_preflight": preflight,
        "host_preflight_sha256": preflight_sha,
        "command": {
            "argv": command,
            "executable_path": str(executable),
            "executable_sha256": _sha256_file(executable),
            "returncode": returncode,
            "started_unix_ns": started_unix_ns,
            "ended_unix_ns": ended_unix_ns,
            "started_monotonic_ns": started_monotonic_ns,
            "ended_monotonic_ns": ended_monotonic_ns,
            "event_stream_sha256": stream_sha,
        },
        "samples": samples,
    }
    record = {**payload, "capture_sha256": krea_provenance.canonical_sha256(payload)}
    validate_capture(record)
    _publish(output, record)
    return record


def validate_capture(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "kind",
        "capture_id",
        "measurement_role",
        "probe_contract",
        "probe_approval",
        "execution_envelope",
        "measurement_tool_sha256",
        "containment",
        "host_preflight",
        "host_preflight_sha256",
        "command",
        "samples",
        "capture_sha256",
    }:
        raise ValueError("timing command capture schema mismatch")
    body = {key: item for key, item in value.items() if key != "capture_sha256"}
    if (
        value["schema"] != 1
        or value["kind"] != "forge-krea-timing-command-capture"
        or value["measurement_role"]
        not in {
            "timing_measurement",
            "held_out_end_to_end",
        }
        or value["capture_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("timing command capture digest/kind is invalid")
    plan_binding = value["probe_contract"]
    approval_binding = value["probe_approval"]
    if not isinstance(plan_binding, dict) or set(plan_binding) != {
        "path",
        "sha256",
        "probe_contract_sha256",
    }:
        raise ValueError("capture probe contract binding is invalid")
    if not isinstance(approval_binding, dict) or set(approval_binding) != {
        "path",
        "sha256",
        "approval_sha256",
    }:
        raise ValueError("capture probe approval binding is invalid")
    plan, plan_file_sha = _load_canonical(
        Path(plan_binding["path"]), "timing probe plan"
    )
    approval, approval_file_sha = _load_canonical(
        Path(approval_binding["path"]), "timing probe approval"
    )
    krea_execution_plan.validate_timing_probe_plan(plan)
    krea_execution_plan.validate_timing_probe_approval(approval, plan=plan)
    if (
        plan_file_sha != plan_binding["sha256"]
        or plan["probe_contract_sha256"] != plan_binding["probe_contract_sha256"]
        or approval_file_sha != approval_binding["sha256"]
        or approval["approval_sha256"] != approval_binding["approval_sha256"]
        or value["execution_envelope"] != plan["execution_envelope"]
        or value["measurement_tool_sha256"]
        != _sha256_file(Path(__file__).resolve(strict=True))
        or value["measurement_tool_sha256"] != plan["measurement_tool_sha256"]
        or _validate_containment_receipt(value["containment"]) != value["containment"]
        or not isinstance(value["host_preflight"], dict)
        or not isinstance(value["host_preflight_sha256"], str)
        or not _SHA256.fullmatch(value["host_preflight_sha256"])
        or value["host_preflight_sha256"]
        != krea_provenance.canonical_sha256(value["host_preflight"])
        or value["host_preflight"].get("static")
        != krea_execution_plan.validate_timing_probe_plan(plan)[
            "host_execution_manifest"
        ]["static"]
    ):
        raise ValueError("timing capture escaped its approved plan/tool")
    command = value["command"]
    if not isinstance(command, dict) or set(command) != {
        "argv",
        "executable_path",
        "executable_sha256",
        "returncode",
        "started_unix_ns",
        "ended_unix_ns",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "event_stream_sha256",
    }:
        raise ValueError("timing capture command schema mismatch")
    if (
        command["argv"] != plan["command_argv"]
        or command["returncode"] != 0
        or command["ended_unix_ns"] <= command["started_unix_ns"]
        or command["ended_monotonic_ns"] <= command["started_monotonic_ns"]
        or _sha256_file(
            _safe_file(Path(command["executable_path"]), "captured executable")
        )
        != command["executable_sha256"]
        or not _SHA256.fullmatch(command["event_stream_sha256"])
    ):
        raise ValueError("timing capture command identity/result is invalid")
    samples = value["samples"]
    if not isinstance(samples, dict) or set(samples) != _METRICS:
        raise ValueError("timing capture samples schema mismatch")
    for metric, rows in samples.items():
        if not isinstance(rows, list):
            raise ValueError("timing capture sample rows must be arrays")
        for row in rows:
            # Reuse the profile builder's exact receipt-clock normalization.
            normalized = krea_budget._timing_sample(row, metric=metric)
            if normalized != row or row["capture_id"] != value["capture_id"]:
                raise ValueError("timing capture sample is not canonical/capture-bound")
    return value


def raw_from_captures(captures: Iterable[dict[str, Any]]) -> dict[str, Any]:
    captures = list(captures)
    for capture_record in captures:
        validate_capture(capture_record)
        if capture_record["measurement_role"] != "timing_measurement":
            raise ValueError(
                "raw timing manifest may contain only measurement captures"
            )
    if not captures:
        raise ValueError("at least one timing capture is required")
    contract_shas = {
        item["probe_contract"]["probe_contract_sha256"] for item in captures
    }
    envelopes = {
        krea_provenance.canonical_sha256(item["execution_envelope"])
        for item in captures
    }
    tools = {item["measurement_tool_sha256"] for item in captures}
    if len(contract_shas) != 1 or len(envelopes) != 1 or len(tools) != 1:
        raise ValueError("timing captures do not share one probe/envelope/tool")
    samples: dict[str, list[dict[str, Any]]] = {
        metric: [] for metric in sorted(_METRICS)
    }
    commands = []
    seed_bindings: dict[str, int] = {}
    for index, item in enumerate(captures):
        for metric, rows in item["samples"].items():
            samples[metric].extend(rows)
        command = item["command"]
        commands.append(
            {
                "capture_id": item["capture_id"],
                "argv": command["argv"],
                "executable_path": command["executable_path"],
                "executable_sha256": command["executable_sha256"],
                "returncode": command["returncode"],
                "started_unix_ns": command["started_unix_ns"],
                "ended_unix_ns": command["ended_unix_ns"],
                "event_stream_sha256": command["event_stream_sha256"],
            }
        )
        plan, _ = _load_canonical(
            Path(item["probe_contract"]["path"]), "timing probe plan"
        )
        role = plan["seed_role"]
        seed = plan["seed"]
        if role in seed_bindings and seed_bindings[role] != seed:
            raise ValueError("timing captures disagree on a seed role")
        seed_bindings[role] = seed
    return krea_budget.seal_timing_sample_manifest(
        execution_envelope=captures[0]["execution_envelope"],
        probe_contract_sha256=next(iter(contract_shas)),
        measurement_tool_sha256=next(iter(tools)),
        command_captures=commands,
        samples=samples,
        seed_bindings=[
            {"role": role, "seed": seed} for role, seed in sorted(seed_bindings.items())
        ],
    )


def assemble_raw(*, capture_paths: Iterable[Path], output: Path) -> dict[str, Any]:
    captures = [_load_canonical(path, "timing capture")[0] for path in capture_paths]
    record = raw_from_captures(captures)
    _publish(output, record)
    return record


def end_to_end_from_records(
    captures: Iterable[dict[str, Any]],
    run_records: Iterable[tuple[dict[str, Any], str]],
) -> dict[str, Any]:
    captures = list(captures)
    validated_captures = []
    for value in captures:
        validate_capture(value)
        if value["measurement_role"] != "held_out_end_to_end":
            raise ValueError("end-to-end validation requires held-out captures")
        validated_captures.append(value)
    run_records = list(run_records)
    if not validated_captures or len(validated_captures) != len(run_records):
        raise ValueError("each held-out capture needs exactly one run record")
    runs = []
    contract_shas = set()
    envelope_shas = set()
    for capture_record, (run, run_file_sha) in zip(validated_captures, run_records):
        plan, _ = _load_canonical(
            Path(capture_record["probe_contract"]["path"]), "timing probe plan"
        )
        telemetry = run.get("telemetry")
        events = telemetry.get("events") if isinstance(telemetry, dict) else None
        names = (
            [row.get("name") for row in events if isinstance(row, dict)]
            if isinstance(events, list)
            else []
        )
        toolkit_end = (
            next(
                (
                    row
                    for row in events
                    if isinstance(row, dict) and row.get("name") == "toolkit_end"
                ),
                None,
            )
            if isinstance(events, list)
            else None
        )
        metrics = (
            next(
                (
                    row
                    for row in events
                    if isinstance(row, dict) and row.get("name") == "toolkit_metrics"
                ),
                None,
            )
            if isinstance(events, list)
            else None
        )
        failure = any(
            isinstance(name, str)
            and (
                name.endswith("_failed")
                or "fallback" in name
                or name.startswith("holdout_")
            )
            for name in names
        )
        if (
            run.get("complete") is not True
            or run.get("kind") != "forge-krea2-bootstrap-timing-run"
            or run.get("timing_probe_contract_sha256") != plan["probe_contract_sha256"]
            or run.get("timing_capture_id") != capture_record["capture_id"]
            or not isinstance(toolkit_end, dict)
            or toolkit_end.get("returncode") != 0
            or toolkit_end.get("stopped_by_deadline") is not False
            or not isinstance(metrics, dict)
            or metrics.get("last_step") != plan["probe_schedule"]["planned_steps"]
            or names.count("run_complete") != 1
            or failure
        ):
            raise ValueError("held-out run record is not a clean natural completion")
        command = capture_record["command"]
        wall_s = (
            command["ended_monotonic_ns"] - command["started_monotonic_ns"]
        ) / 1_000_000_000
        contract_shas.add(plan["probe_contract_sha256"])
        envelope_shas.add(plan["execution_envelope"]["execution_envelope_sha256"])
        runs.append(
            {
                "run_id": capture_record["capture_id"],
                "seed_role": plan["seed_role"],
                "seed": plan["seed"],
                "hard_budget_s": plan["probe_schedule"]["hard_budget_s"],
                "outer_wall_clock_s": wall_s,
                "natural_completion": True,
                "upload_ready": True,
                "failure_or_fallback_telemetry": False,
                "run_record_sha256": run_file_sha,
            }
        )
    if len(contract_shas) != 1 or len(envelope_shas) != 1:
        raise ValueError("held-out runs do not share one probe execution envelope")
    return krea_budget.seal_end_to_end_validation(
        execution_envelope_sha256=next(iter(envelope_shas)),
        probe_contract_sha256=next(iter(contract_shas)),
        runs=runs,
    )


def build_e2e(
    *, capture_paths: Iterable[Path], run_record_paths: Iterable[Path], output: Path
) -> dict[str, Any]:
    captures = [
        _load_canonical(path, "held-out timing capture")[0] for path in capture_paths
    ]
    run_records = [
        _load_canonical(path, "held-out probe run record") for path in run_record_paths
    ]
    record = end_to_end_from_records(captures, run_records)
    _publish(output, record)
    return record


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command_name", required=True)
    seal_probe = sub.add_parser(
        "seal-probe", help="seal an unapproved pre-profile timing payload"
    )
    seal_probe.add_argument("--payload", required=True, type=Path)
    seal_probe.add_argument("--output", required=True, type=Path)

    approve_probe = sub.add_parser(
        "approve-probe", help="create the separate named-human probe approval"
    )
    approve_probe.add_argument("--probe-plan", required=True, type=Path)
    approve_probe.add_argument("--reviewer", required=True)
    approve_probe.add_argument("--approved-at-utc", required=True)
    approve_probe.add_argument("--output", required=True, type=Path)

    emit = sub.add_parser("emit", help="emit one child marker to the capture socket")
    emit.add_argument("--observation-id", required=True)
    emit.add_argument("--metric", choices=sorted(_METRICS), required=True)
    emit.add_argument("--state", choices=("begin", "end"), required=True)
    emit.add_argument("--units", type=int, default=1)

    capture_parser = sub.add_parser("capture", help="run the sealed probe command")
    capture_parser.add_argument("--probe-plan", required=True, type=Path)
    capture_parser.add_argument("--probe-approval", required=True, type=Path)
    capture_parser.add_argument("--checkpoint-path", required=True, type=Path)
    capture_parser.add_argument("--output", required=True, type=Path)
    capture_parser.add_argument("--capture-id", required=True)
    capture_parser.add_argument(
        "--measurement-role",
        choices=("timing_measurement", "held_out_end_to_end"),
        required=True,
    )
    capture_parser.add_argument("--timeout-s", required=True, type=float)
    capture_parser.add_argument("argv", nargs=argparse.REMAINDER)

    raw = sub.add_parser("assemble-raw", help="seal raw samples from captures")
    raw.add_argument("--capture", action="append", required=True, type=Path)
    raw.add_argument("--output", required=True, type=Path)

    e2e = sub.add_parser("build-e2e", help="seal held-out natural completion")
    e2e.add_argument("--capture", action="append", required=True, type=Path)
    e2e.add_argument("--run-record", action="append", required=True, type=Path)
    e2e.add_argument("--output", required=True, type=Path)

    margin = sub.add_parser(
        "seal-margin", help="freeze the named-human timing margin before capture"
    )
    margin.add_argument("--reviewer", required=True)
    margin.add_argument("--approved-at-utc", required=True)
    margin.add_argument("--multiplier", required=True, type=float)
    for metric in sorted(_METRICS):
        margin.add_argument(
            f"--{metric.replace('_', '-')}-additive-s",
            dest=f"{metric}_additive_s",
            required=True,
            type=float,
        )
    margin.add_argument("--output", required=True, type=Path)

    profile = sub.add_parser("build-profile", help="recompute a timing profile")
    profile.add_argument("--raw", required=True, type=Path)
    profile.add_argument("--margin", required=True, type=Path)
    profile.add_argument("--e2e", required=True, type=Path)
    profile.add_argument("--framework-stop-boundary-s", required=True, type=float)
    profile.add_argument("--framework-boundary-source-sha256", required=True)
    profile.add_argument("--output", required=True, type=Path)

    validate_probe = sub.add_parser(
        "validate-probe", help="validate a pre-profile probe and approval"
    )
    validate_probe.add_argument("--probe-plan", required=True, type=Path)
    validate_probe.add_argument("--probe-approval", required=True, type=Path)

    verify = sub.add_parser("verify", help="revalidate one sealed timing artifact")
    verify.add_argument(
        "--kind", choices=("capture", "raw", "margin", "e2e", "profile"), required=True
    )
    verify.add_argument("--path", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    if args.command_name == "seal-probe":
        payload_path = _safe_file(args.payload, "timing probe payload")
        try:
            payload = json.loads(payload_path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("timing probe payload is not JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("timing probe payload must be a JSON object")
        plan = krea_execution_plan.seal_timing_probe_plan(payload)
        _publish(args.output, plan)
        return 0
    if args.command_name == "approve-probe":
        plan, _ = _load_canonical(args.probe_plan, "timing probe plan")
        approval = krea_execution_plan.build_timing_probe_approval(
            plan,
            reviewer_identity=args.reviewer,
            approved_at_utc=args.approved_at_utc,
        )
        _publish(args.output, approval)
        return 0
    if args.command_name == "emit":
        emit_marker(
            observation_id=args.observation_id,
            metric=args.metric,
            state=args.state,
            units=args.units,
        )
        return 0
    if args.command_name == "capture":
        argv = list(args.argv)
        if argv and argv[0] == "--":
            argv.pop(0)
        if not argv:
            raise ValueError("capture requires a command after --")
        capture(
            probe_plan_path=args.probe_plan,
            probe_approval_path=args.probe_approval,
            checkpoint_path=args.checkpoint_path,
            output=args.output,
            command=argv,
            capture_id=args.capture_id,
            measurement_role=args.measurement_role,
            timeout_s=args.timeout_s,
        )
        return 0
    if args.command_name == "assemble-raw":
        assemble_raw(capture_paths=args.capture, output=args.output)
        return 0
    if args.command_name == "build-e2e":
        build_e2e(
            capture_paths=args.capture,
            run_record_paths=args.run_record,
            output=args.output,
        )
        return 0
    if args.command_name == "seal-margin":
        record = krea_budget.seal_margin_policy(
            reviewer_identity=args.reviewer,
            approved_at_utc=args.approved_at_utc,
            frozen_before_capture=True,
            multiplicative_margin={name: args.multiplier for name in _METRICS},
            additive_margin_s={
                name: getattr(args, f"{name}_additive_s") for name in _METRICS
            },
        )
        _publish(args.output, record)
        return 0
    if args.command_name == "build-profile":
        raw, _ = _load_canonical(args.raw, "raw timing sample manifest")
        margin, _ = _load_canonical(args.margin, "timing margin policy")
        e2e, _ = _load_canonical(args.e2e, "held-out end-to-end timing validation")
        profile = krea_budget.seal_throughput_profile_from_evidence(
            raw_sample_manifest=raw,
            margin_policy=margin,
            end_to_end_validation=e2e,
            framework_stop_boundary_s=args.framework_stop_boundary_s,
            framework_stop_boundary_source_sha256=args.framework_boundary_source_sha256,
        )
        _publish(args.output, profile)
        return 0
    if args.command_name == "validate-probe":
        plan, _ = _load_canonical(args.probe_plan, "timing probe plan")
        approval, _ = _load_canonical(args.probe_approval, "timing probe approval")
        krea_execution_plan.validate_timing_probe_plan(plan)
        krea_execution_plan.validate_timing_probe_approval(approval, plan=plan)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "probe_contract_sha256": plan["probe_contract_sha256"],
                    "approval_sha256": approval["approval_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command_name == "verify":
        value, file_sha = _load_canonical(args.path, f"{args.kind} artifact")
        if args.kind == "capture":
            validated = validate_capture(value)
            semantic_sha = validated["capture_sha256"]
        elif args.kind == "raw":
            validated = krea_budget.load_timing_sample_manifest(value)
            semantic_sha = validated["raw_sample_manifest_sha256"]
        elif args.kind == "margin":
            validated = krea_budget.load_margin_policy(value)
            semantic_sha = validated["margin_policy_sha256"]
        elif args.kind == "e2e":
            validated = krea_budget.load_end_to_end_validation(value)
            semantic_sha = validated["end_to_end_validation_sha256"]
        else:
            validated_profile = krea_budget.load_throughput_profile(value)
            semantic_sha = validated_profile.profile_sha256
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "kind": args.kind,
                    "file_sha256": file_sha,
                    "semantic_sha256": semantic_sha,
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError("unreachable subcommand")


if __name__ == "__main__":
    raise SystemExit(main())
