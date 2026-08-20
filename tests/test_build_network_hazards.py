"""Week-9 HAZARD 1: build-time network steps must be hang-bounded.

The week-9 rehearsal (evidence/week9-rehearsal-20260819/beta/REPORT.md, incident
I-R3) wedged the toolkit image build for ~72 minutes inside a git clone that pip
ran for a ``git+https`` requirement.  ``retry_network`` retried on FAILURE only:
a stalled-but-open TCP stream never returns, so the loop never fires and the
build hangs forever.  The validator builds these same dockerfiles under its own
limits, so an unbounded build is a forfeit.

These tests are text-and-behaviour tests over the real dockerfiles; they need no
Docker daemon and no network.  They fail on the week9-rc bytes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time

import pytest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOCKER_DIR = os.path.join(_REPO_ROOT, "ops", "docker")
DOCKERFILES = {
    "toolkit": os.path.join(_DOCKER_DIR, "standalone-image-toolkit-trainer.dockerfile"),
    "legacy": os.path.join(_DOCKER_DIR, "standalone-image-trainer.dockerfile"),
}

# The retry loop's own backoff: 5 + 10 + 15 + 20 between five attempts.
_BACKOFF_S = 50
_MAX_ATTEMPTS = 5
# `timeout -k 30` gives the child 30 s to die on SIGTERM before SIGKILL, so a
# single attempt can occupy the cap plus that grace.
_KILL_GRACE_S = 30

# Worst case per command = min(all five attempts hitting the cap, the budget
# elapsing during an attempt that then runs to the cap) + the backoff sleeps.
# The apt loop has no budget, only its five attempts over two bounded calls.
_APT_WORST_S = _MAX_ATTEMPTS * (180 + _KILL_GRACE_S + 300 + _KILL_GRACE_S) + _BACKOFF_S

# Pinned so that inflating a timeout has to move a number a reviewer can see.
EXPECTED_WORST_CASE_S = {"toolkit": 8320, "legacy": 12950}
# Hard ceiling: no image may plan more than 4 h of worst-case network time.
WORST_CASE_CEILING_S = 4 * 3600

# The two operator-facing log lines, byte-for-byte as week9-rc emitted them.
RETRY_LOG_LINES = (
    'echo "SN56_NETWORK_RETRY exhausted attempts=$attempt command=$1 '
    'status=$status" >&2;',
    'echo "SN56_NETWORK_RETRY retry=$((attempt + 1))/5 delay_seconds=$delay '
    'command=$1 status=$status" >&2;',
)

# The exact unbounded invocation week9-rc shipped.  Its presence anywhere is the
# hazard itself.
UNBOUNDED_INVOCATION = '"$@" && return 0;'
TIMEOUT_WRAPPER = 'timeout -k 30 "$attempt_timeout_s" '


def _read(name: str) -> str:
    with open(DOCKERFILES[name], encoding="utf-8") as handle:
        return handle.read()


def _run_instructions(text: str) -> list[str]:
    """Return every RUN instruction as one logical line (continuations joined)."""
    logical: list[str] = []
    current: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if current:
            current.append(stripped)
            if not stripped.endswith("\\"):
                logical.append(" ".join(part.rstrip("\\").strip() for part in current))
                current = []
            continue
        if stripped.startswith("#") or not stripped.startswith("RUN "):
            continue
        if stripped.endswith("\\"):
            current = [stripped]
        else:
            logical.append(stripped.rstrip("\\").strip())
    if current:
        logical.append(" ".join(part.rstrip("\\").strip() for part in current))
    return logical


def _retry_calls(text: str) -> list[tuple[int, int, str]]:
    """Every ``retry_network`` invocation as (per_attempt_s, budget_s, command).

    Scans the joined RUN instructions only, so dockerfile comments that mention
    the helper by name cannot be mistaken for call sites.
    """
    calls: list[tuple[int, int, str]] = []
    for line in _run_instructions(text):
        for match in re.finditer(r"retry_network\s+(\S+)\s+(\S+)\s+(\S+)", line):
            first, second, command = match.groups()
            assert first.isdigit(), f"retry_network call is unbounded: {match.group(0)!r}"
            assert second.isdigit(), f"retry_network call has no budget: {match.group(0)!r}"
            calls.append((int(first), int(second), command))
    return calls


def _retry_layers(text: str) -> list[str]:
    return [line for line in _run_instructions(text) if "retry_network()" in line]


def _worst_case_seconds(name: str) -> int:
    total = 0
    for per_attempt, budget, _command in _retry_calls(_read(name)):
        total += min(
            _MAX_ATTEMPTS * (per_attempt + _KILL_GRACE_S) + _BACKOFF_S,
            budget + per_attempt + _KILL_GRACE_S + _BACKOFF_S,
        )
    if name == "legacy":
        total += _APT_WORST_S
    return total


# --------------------------------------------------------------------------
# Structure: every network attempt is wrapped
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_no_unbounded_retry_network_invocation_survives(name: str) -> None:
    """week9-rc ran the command bare; a hang there could never be retried."""
    text = _read(name)
    assert "retry_network()" in text, "the retry helper must still exist"
    for match in re.finditer(re.escape(UNBOUNDED_INVOCATION), text):
        prefix = text[: match.start()]
        assert prefix.endswith(TIMEOUT_WRAPPER), (
            f"{name}: an unwrapped {UNBOUNDED_INVOCATION!r} survives at "
            f"offset {match.start()}"
        )


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_retry_helper_wraps_each_attempt_in_timeout(name: str) -> None:
    layers = _retry_layers(_read(name))
    assert layers, f"{name}: no retry_network layer found"
    for layer in layers:
        assert TIMEOUT_WRAPPER + UNBOUNDED_INVOCATION in layer
        # A missing `timeout` must fail loudly instead of silently running the
        # command unbounded again.
        assert 'command -v timeout >/dev/null 2>&1 ||' in layer
        assert "return 127;" in layer


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_every_retry_network_call_carries_a_timeout_and_a_budget(name: str) -> None:
    calls = _retry_calls(_read(name))
    assert calls, f"{name}: no retry_network call sites found"
    for per_attempt, budget, command in calls:
        assert 60 <= per_attempt <= 1800, (command, per_attempt)
        assert budget >= per_attempt, (command, per_attempt, budget)
        assert budget <= 3600, (command, budget)


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_git_stall_detection_is_exported_in_every_retry_layer(name: str) -> None:
    """`timeout` bounds the whole pip run; these bound the git clone inside it."""
    for layer in _retry_layers(_read(name)):
        assert "export GIT_HTTP_LOW_SPEED_LIMIT=1000" in layer
        assert "GIT_HTTP_LOW_SPEED_TIME=120" in layer


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_operator_log_lines_are_byte_preserved(name: str) -> None:
    """Operators grep SN56_NETWORK_RETRY; the fix may not reword those lines."""
    text = _read(name)
    for line in RETRY_LOG_LINES:
        assert line in text, line
    assert text.count("SN56_NETWORK_TIMEOUT") >= len(_retry_layers(text))


def test_apt_toolchain_loop_bounds_its_network_calls() -> None:
    """The legacy image's hand-rolled apt loop had the same failure-only retry."""
    text = _read("legacy")
    assert "timeout -k 30 180 apt-get -o Acquire::Retries=3 update" in text
    assert (
        "timeout -k 30 300 apt-get -o Acquire::Retries=3 install "
        "-y --no-install-recommends" in text
    )
    assert 'command -v timeout >/dev/null 2>&1 ||' in text


def test_no_network_run_step_is_left_unbounded() -> None:
    """Every RUN that reaches the network goes through a bounded wrapper."""
    network_tokens = ("git fetch", "pip install", "apt-get update", "apt-get -o")
    for name in DOCKERFILES:
        for line in _run_instructions(_read(name)):
            if not any(token in line for token in network_tokens):
                continue
            for token in ("git fetch", "pip install"):
                for match in re.finditer(re.escape(token), line):
                    prefix = line[: match.start()]
                    assert re.search(r"retry_network \d+ \d+ [^;&|]*$", prefix), (
                        f"{name}: unbounded {token!r} in {line[:120]!r}"
                    )
            for match in re.finditer(r"apt-get -o Acquire::Retries=3", line):
                prefix = line[: match.start()]
                assert prefix.rstrip().endswith(
                    tuple(f"timeout -k 30 {n}" for n in (180, 300))
                ), f"unbounded apt-get in {line[:120]!r}"


# --------------------------------------------------------------------------
# Arithmetic: the bound is small enough to be worth having
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_worst_case_network_time_is_bounded_and_pinned(name: str) -> None:
    worst = _worst_case_seconds(name)
    assert worst == EXPECTED_WORST_CASE_S[name], (
        f"{name} worst-case network seconds changed to {worst}; update "
        "evidence/week9-hazards-20260819/CHANGES.md and this pin together"
    )
    assert worst <= WORST_CASE_CEILING_S


# --------------------------------------------------------------------------
# Behaviour: a hang really does become a bounded failure
# --------------------------------------------------------------------------


def _extract_helper(name: str) -> str:
    """Pull the real retry_network definition out of the real dockerfile."""
    layer = _retry_layers(_read(name))[0]
    start = layer.index("retry_network() {")
    end = layer.index("done; };", start) + len("done; };")
    return layer[start:end]


# A POSIX stand-in for GNU coreutils `timeout`, used only on hosts that have
# none (macOS).  Both dockerfile base images are Debian/Ubuntu, where `timeout`
# is part of the Essential coreutils package, so on Linux the tests below drive
# the real binary instead.  The watcher's >/dev/null 2>&1 matters: it must not
# inherit the caller's pipes, or capture_output() would block on its grace sleep.
_TIMEOUT_SHIM = """#!/bin/sh
kill_after=""
while [ $# -gt 0 ]; do
  case "$1" in
    -k) kill_after=$2; shift 2 ;;
    --kill-after=*) kill_after=${1#--kill-after=}; shift ;;
    *) break ;;
  esac
done
duration=$1; shift
"$@" &
cmd_pid=$!
( sleep "$duration"; kill -TERM "$cmd_pid" 2>/dev/null
  if [ -n "$kill_after" ]; then
    sleep "$kill_after"; kill -KILL "$cmd_pid" 2>/dev/null
  fi ) >/dev/null 2>&1 &
watch_pid=$!
wait "$cmd_pid" 2>/dev/null
rc=$?
kill -TERM "$watch_pid" 2>/dev/null
if [ "$rc" -ge 128 ]; then exit 124; fi
exit "$rc"
"""


@pytest.fixture(name="shell_env")
def _shell_env(tmp_path) -> dict:
    """PATH with a `timeout` available (a POSIX stand-in on hosts without one)."""
    env = dict(os.environ)
    if shutil.which("timeout") is None:
        shim_dir = tmp_path / "bin"
        shim_dir.mkdir()
        shim = shim_dir / "timeout"
        shim.write_text(_TIMEOUT_SHIM, encoding="utf-8")
        shim.chmod(0o755)
        env["PATH"] = f"{shim_dir}:{env['PATH']}"
    return env


def _run_helper(script: str, env: dict, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/sh", "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_helper_passes_a_successful_command_straight_through(
    name: str, shell_env: dict
) -> None:
    helper = _extract_helper(name)
    result = _run_helper(f"{helper} retry_network 5 30 true; echo rc=$?", shell_env, 30)
    assert "rc=0" in result.stdout
    assert "SN56_NETWORK_RETRY" not in result.stderr


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_a_hung_command_becomes_a_retry_then_a_bounded_failure(
    name: str, shell_env: dict
) -> None:
    """The decisive test: week9-rc's helper waits forever here."""
    helper = _extract_helper(name)
    started = time.monotonic()
    # per-attempt cap 1 s, total budget 4 s: attempt 1 times out, the loop
    # retries (5 s backoff), attempt 2 times out and the budget is spent.
    result = _run_helper(
        f"{helper} retry_network 1 4 sleep 300; echo rc=$?", shell_env, 60
    )
    elapsed = time.monotonic() - started

    assert "rc=124" in result.stdout, result.stdout
    assert "SN56_NETWORK_TIMEOUT attempt=1 timeout_seconds=1" in result.stderr
    assert "SN56_NETWORK_RETRY retry=2/5 delay_seconds=5" in result.stderr
    assert "SN56_NETWORK_RETRY exhausted attempts=2" in result.stderr
    assert elapsed < 30, f"bounded failure took {elapsed:.1f}s"


def test_week9_rc_helper_would_have_hung(shell_env: dict) -> None:
    """Red/green control: the shipped RC helper never returns on a hang."""
    rc_helper = (
        "retry_network() { "
        "attempt=1; "
        "while :; do "
        '"$@" && return 0; '
        "status=$?; "
        'if [ "$attempt" -ge 5 ]; then '
        'echo "SN56_NETWORK_RETRY exhausted attempts=$attempt" >&2; '
        'return "$status"; '
        "fi; "
        "delay=$((attempt * 5)); "
        'sleep "$delay"; '
        "attempt=$((attempt + 1)); "
        "done; "
        "}; "
    )
    with pytest.raises(subprocess.TimeoutExpired):
        _run_helper(f"{rc_helper} retry_network sleep 300", shell_env, 5)
