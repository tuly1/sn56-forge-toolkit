#!/usr/bin/env python3
"""Literal Linux smoke for Krea's transient-systemd evaluator containment.

This script is intentionally separate from pytest: it must run on a real
systemd host as root.  It exercises the same ``_run_contained`` primitive the
exact-score batch uses, without reading or writing production services.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

try:
    from . import batch_evaluate_krea as batch
except ImportError:  # pragma: no cover - direct script execution.
    import batch_evaluate_krea as batch  # type: ignore[no-redef]


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-target", choices=("SIGTERM", "SIGHUP"))
    parser.add_argument("--pid-file", type=Path)
    return parser.parse_args()


def _containment() -> dict[str, Any]:
    return {
        "mode": "systemd_transient_service",
        "unit_type": "transient_service",
        "network_policy": {
            "private_network": True,
            "restrict_address_families": ["AF_UNIX", "AF_INET", "AF_INET6"],
            "loopback_allowed": True,
            "outbound_network_blocked": True,
        },
        "term_grace_s": 0.5,
        "systemd_run_path": "/usr/bin/systemd-run",
        "systemctl_path": "/usr/bin/systemctl",
    }


def _unit(pid: int, candidate_id: str) -> str:
    suffix = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:12]
    return f"forge-krea-eval-{pid}-{suffix}"


def _gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.05)
    return False


def _unit_collected(unit: str) -> bool:
    result = subprocess.run(
        [
            "/usr/bin/systemctl",
            "show",
            "--no-pager",
            "--all",
            "--property=LoadState",
            "--property=ActiveState",
            "--property=ControlGroup",
            unit,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    values = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values == {
        "LoadState": "not-found",
        "ActiveState": "inactive",
        "ControlGroup": "",
    }


def _signal_target(pid_file: Path) -> int:
    code = (
        "import os,time;"
        f"p={str(pid_file)!r};"
        "open(p,'x').write(str(os.getpid()));"
        "time.sleep(3600)"
    )
    batch._run_contained(
        [sys.executable, "-c", code],
        timeout_s=120.0,
        candidate_id="literal-signal-target",
        containment=_containment(),
    )
    return 99


def _signal_case(root: Path, signame: str) -> dict[str, Any]:
    pid_file = root / f"{signame.lower()}-child.pid"
    command = [
        sys.executable,
        str(Path(__file__).resolve(strict=True)),
        "--signal-target",
        signame,
        "--pid-file",
        str(pid_file),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 10.0
    while (
        not pid_file.is_file()
        and process.poll() is None
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    if not pid_file.is_file():
        stdout, stderr = process.communicate(timeout=2.0)
        raise RuntimeError(
            f"{signame} target failed to start: rc={process.returncode}, "
            f"stdout={stdout!r}, stderr={stderr!r}"
        )
    child_pid = int(pid_file.read_text(encoding="utf-8"))
    unit = _unit(process.pid, "literal-signal-target")
    os.kill(process.pid, getattr(signal, signame))
    stdout, stderr = process.communicate(timeout=10.0)
    if process.returncode == 0 or not _gone(child_pid) or not _unit_collected(unit):
        raise RuntimeError(
            f"{signame} cleanup failed: rc={process.returncode}, child={child_pid}, "
            f"unit={unit}, stdout={stdout!r}, stderr={stderr!r}"
        )
    return {
        "signal": signame,
        "driver_returncode": process.returncode,
        "child_pid_gone": True,
        "unit_collected": True,
    }


def main() -> int:
    args = _parse()
    if args.signal_target:
        if args.pid_file is None:
            raise ValueError("--signal-target requires --pid-file")
        return _signal_target(args.pid_file)
    if os.geteuid() != 0:
        raise PermissionError("literal systemd containment smoke must run as root")
    for binary in ("/usr/bin/systemd-run", "/usr/bin/systemctl"):
        if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
            raise RuntimeError(f"required systemd binary is unavailable: {binary}")

    report: dict[str, Any] = {
        "schema": 1,
        "kind": "forge-krea-literal-systemd-containment-smoke",
        "production_units_touched": False,
        "checks": {},
    }
    normal_id = "literal-normal"
    normal = batch._run_contained(
        [sys.executable, "-c", "print('contained-ok')"],
        timeout_s=5.0,
        candidate_id=normal_id,
        containment=_containment(),
    )
    normal_unit = _unit(os.getpid(), normal_id)
    if (
        normal.returncode != 0
        or normal.stdout.strip() != "contained-ok"
        or not _unit_collected(normal_unit)
    ):
        raise RuntimeError(f"normal containment failed: {normal}")
    report["checks"]["normal_exit"] = {
        "returncode": normal.returncode,
        "unit_collected": True,
    }

    network_code = r"""
import json, os, socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('127.0.0.1', 0))
server.listen(1)
outbound = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
outbound.settimeout(1.0)
try:
    result = outbound.connect_ex(('1.1.1.1', 53))
finally:
    outbound.close()
print(json.dumps({'interfaces': sorted(os.listdir('/sys/class/net')), 'loopback_port': server.getsockname()[1], 'outbound_connect_ex': result}))
server.close()
"""
    network_id = "literal-network"
    network = batch._run_contained(
        [sys.executable, "-c", network_code],
        timeout_s=5.0,
        candidate_id=network_id,
        containment=_containment(),
    )
    network_row = json.loads(network.stdout)
    network_unit = _unit(os.getpid(), network_id)
    if (
        network.returncode != 0
        or network_row["interfaces"] != ["lo"]
        or not isinstance(network_row["loopback_port"], int)
        or network_row["outbound_connect_ex"] == 0
        or not _unit_collected(network_unit)
    ):
        raise RuntimeError(f"network namespace failed: {network}")
    report["checks"]["network_namespace"] = {
        **network_row,
        "unit_collected": True,
    }

    with tempfile.TemporaryDirectory(prefix="krea-containment-smoke-") as temporary:
        root = Path(temporary)
        descendant_file = root / "descendant.pid"
        timeout_code = (
            "import os,time;"
            "p=os.fork();"
            f"(os.setsid(),open({str(descendant_file)!r},'x').write(str(os.getpid())),time.sleep(3600)) if p==0 else time.sleep(3600)"
        )
        timeout_id = "literal-timeout-descendant"
        timed_out = False
        try:
            batch._run_contained(
                [sys.executable, "-c", timeout_code],
                timeout_s=1.0,
                candidate_id=timeout_id,
                containment=_containment(),
            )
        except TimeoutError:
            timed_out = True
        if not timed_out or not descendant_file.is_file():
            raise RuntimeError("timeout case did not start and time out as declared")
        descendant_pid = int(descendant_file.read_text(encoding="utf-8"))
        timeout_unit = _unit(os.getpid(), timeout_id)
        if not _gone(descendant_pid) or not _unit_collected(timeout_unit):
            raise RuntimeError("setsid descendant or timeout unit survived cleanup")
        report["checks"]["timeout_recursive_cleanup"] = {
            "timed_out": True,
            "setsid_descendant_gone": True,
            "unit_collected": True,
        }
        report["checks"]["SIGTERM"] = _signal_case(root, "SIGTERM")
        report["checks"]["SIGHUP"] = _signal_case(root, "SIGHUP")

    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
