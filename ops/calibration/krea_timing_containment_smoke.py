#!/usr/bin/env python3
"""Literal rootful-Linux smoke for Krea timing-capture containment.

This touches only uniquely named transient ``forge-krea-timing-*`` scopes.  It
proves normal collection plus recursive cleanup of a descendant that calls
``setsid()`` on timeout, SIGTERM, and SIGHUP.  It does not inspect or mutate a
trainer repository, endpoint, production service, or GPU.
"""

from __future__ import annotations

import argparse
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
    from . import krea_timing_probe as timing
except ImportError:  # pragma: no cover - literal direct execution.
    import krea_timing_probe as timing  # type: ignore[no-redef]


_ESCAPE_CODE = r"""
import os
from pathlib import Path
import sys
import time

pid_file = Path(sys.argv[1])
child = os.fork()
if child == 0:
    os.setsid()
    descriptor = os.open(pid_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    time.sleep(3600)
else:
    time.sleep(3600)
"""


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-target", choices=("SIGTERM", "SIGHUP"))
    parser.add_argument("--pid-file", type=Path)
    return parser.parse_args()


def _gone(pid: int, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.05)
    return False


def _run_escape(pid_file: Path, *, capture_id: str, timeout_s: float) -> None:
    timing._run_in_transient_scope(
        [sys.executable, "-c", _ESCAPE_CODE, str(pid_file)],
        env=dict(os.environ),
        timeout_s=timeout_s,
        capture_id=capture_id,
    )


def _signal_target(signame: str, pid_file: Path) -> int:
    capture_id = f"literal-{signame.lower()}"
    try:
        _run_escape(pid_file, capture_id=capture_id, timeout_s=120.0)
    except timing._CaptureCancellation as exc:
        if exc.signum != getattr(signal, signame):
            raise RuntimeError("capture cancellation signal changed") from exc
        return 128 + exc.signum
    raise RuntimeError("signal target exited without receiving its signal")


def _signal_case(root: Path, signame: str) -> dict[str, Any]:
    pid_file = root / f"{signame.lower()}-descendant.pid"
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
        (not pid_file.is_file() or pid_file.stat().st_size == 0)
        and process.poll() is None
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    if not pid_file.is_file() or pid_file.stat().st_size == 0:
        stdout, stderr = process.communicate(timeout=2.0)
        raise RuntimeError(
            f"{signame} target failed to start: rc={process.returncode}, "
            f"stdout={stdout!r}, stderr={stderr!r}"
        )
    descendant_pid = int(pid_file.read_text(encoding="ascii"))
    capture_id = f"literal-{signame.lower()}"
    unit = timing._scope_unit_name(capture_id, driver_pid=process.pid)
    os.kill(process.pid, getattr(signal, signame))
    stdout, stderr = process.communicate(timeout=30.0)
    _, systemctl = timing._systemd_prerequisites()
    if (
        process.returncode != 128 + getattr(signal, signame)
        or not _gone(descendant_pid)
        or timing._scope_status(unit, systemctl_path=systemctl) != "collected"
    ):
        raise RuntimeError(
            f"{signame} cleanup failed: rc={process.returncode}, "
            f"descendant={descendant_pid}, stdout={stdout!r}, stderr={stderr!r}"
        )
    return {
        "driver_returncode": process.returncode,
        "setsid_descendant_gone": True,
        "unit": f"{unit}.scope",
        "unit_collected": True,
    }


def main() -> int:
    args = _parse()
    if args.signal_target:
        if args.pid_file is None:
            raise ValueError("--signal-target requires --pid-file")
        return _signal_target(args.signal_target, args.pid_file)
    if args.pid_file is not None:
        raise ValueError("--pid-file is valid only with --signal-target")
    systemd_run, systemctl = timing._systemd_prerequisites()
    report: dict[str, Any] = {
        "schema": 1,
        "kind": "forge-krea-literal-timing-containment-smoke",
        "production_units_touched": False,
        "measurement_tool_sha256": timing._sha256_file(
            Path(timing.__file__).resolve(strict=True)
        ),
        "systemd_run_sha256": timing._sha256_file(systemd_run),
        "systemctl_sha256": timing._sha256_file(systemctl),
        "checks": {},
    }
    normal_id = "literal-normal"
    returncode, receipt = timing._run_in_transient_scope(
        [sys.executable, "-c", "import time; time.sleep(0.5)"],
        env=dict(os.environ),
        timeout_s=5.0,
        capture_id=normal_id,
    )
    if returncode != 0 or receipt["unit_collected"] is not True:
        raise RuntimeError("normal timing scope did not complete and collect")
    report["checks"]["normal_exit"] = {
        "returncode": returncode,
        "unit": receipt["unit"],
        "unit_collected": True,
    }

    with tempfile.TemporaryDirectory(prefix="krea-timing-containment-") as temporary:
        root = Path(temporary)
        timeout_pid = root / "timeout-descendant.pid"
        timeout_id = "literal-timeout"
        timed_out = False
        try:
            _run_escape(timeout_pid, capture_id=timeout_id, timeout_s=1.0)
        except TimeoutError:
            timed_out = True
        if not timed_out or not timeout_pid.is_file():
            raise RuntimeError("timeout case did not launch and time out")
        timeout_descendant = int(timeout_pid.read_text(encoding="ascii"))
        timeout_unit = timing._scope_unit_name(timeout_id)
        if (
            not _gone(timeout_descendant)
            or timing._scope_status(timeout_unit, systemctl_path=systemctl)
            != "collected"
        ):
            raise RuntimeError("setsid timeout descendant or scope survived cleanup")
        report["checks"]["timeout_recursive_cleanup"] = {
            "timed_out": True,
            "setsid_descendant_gone": True,
            "unit": f"{timeout_unit}.scope",
            "unit_collected": True,
        }
        report["checks"]["SIGTERM"] = _signal_case(root, "SIGTERM")
        report["checks"]["SIGHUP"] = _signal_case(root, "SIGHUP")

    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
