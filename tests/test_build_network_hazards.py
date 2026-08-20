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

# THE WALL.  Upstream G.O.D @ f7caab6c, archived with SHA256SUMS at
# evidence/week9-refutation-review-20260818/upstream-f7caab6c-build-timeout/:
#   trainer/constants.py:41   DOCKER_BUILD_TIMEOUT_MINUTES = 30
#   trainer/runtime.py:107-108  timeout_seconds defaults to that * 60
#   trainer/runtime.py:970-976  the miner build passes no override, no_cache=True
#   trainer/runtime.py:978-983  a failed build completes the task, with NO retry
# Every cap below is sized against this, not against what a slow link would
# like: past 1800 s a slow-but-working build is a DNF too.
VALIDATOR_BUILD_TIMEOUT_S = 1800

# `timeout -k 30` gives the child 30 s to die on SIGTERM before SIGKILL, so one
# attempt can occupy its cap plus that grace.
_KILL_GRACE_S = 30
# The apt loop: 3 attempts over two bounded calls, backoff 5 + 10.
_APT_ATTEMPTS = 3
_APT_WORST_S = _APT_ATTEMPTS * (60 + _KILL_GRACE_S + 150 + _KILL_GRACE_S) + 15

# Observed normal-case maxima across every archived clean build, re-derivable
# with evidence/week9-hazards-20260819/measure_observed_network_durations.py.
# A cap below its step's observed max would manufacture failures on healthy
# builds, so every cap is checked against these.
OBSERVED_MAX_NORMAL_S = {
    "git": 2.5,
    "requirements.txt": 399.4,
    "torch==2.6.0": 4.2,
    "torchcodec==0.2.1": 5.2,
    "image-runtime-lock.txt": 100.5,
    "flux-tokenizer-download": 23.0,
}
MIN_CAP_RATIO = 1.35

# Pinned so that changing a timeout has to move a number a reviewer can see.
# ALL-STALL = every network command burns every attempt.  Both exceed the wall
# and cannot be brought under it without caps that break healthy builds; see
# CHANGES.md ADDENDUM §A2 for the proof and why that is a property of the
# image's own length, not of the retry policy.
EXPECTED_ALL_STALL_S = {"toolkit": 2615, "legacy": 3685}
# SINGLE-STALL = the OBSERVED failure mode: one command stalls, the retry
# succeeds.  This is the number the caps actually control.
EXPECTED_SINGLE_STALL_OVERHEAD_S = {"toolkit": 635, "legacy": 635}
# Observed normal end-to-end build wall time on the validator's own builder
# (classic, no BuildKit).  legacy: OBSERVED 2026-08-19 rehearsal 15:38->16:02Z.
# toolkit: INFERRED, BuildKit layer sum x the legacy classic/BuildKit ratio.
OBSERVED_NORMAL_BUILD_S = {"toolkit": 1155, "legacy": 1440}
# The legacy image cannot absorb even one stalled-and-retried step.  Pinned so
# the build-reduction work has a target and any regression is visible.
LEGACY_SINGLE_STALL_DEFICIT_S = 275
# No single attempt may occupy more than a third of the validator's window.
MAX_PER_ATTEMPT_CAP_S = 600

# The two operator-facing log lines, byte-for-byte as week9-rc emitted them.
# The `/5` became `/$attempt_max` when attempt counts became per-call; the
# greppable prefix `SN56_NETWORK_RETRY retry=` is unchanged, as is the
# `exhausted` line.
RETRY_LOG_LINES = (
    'echo "SN56_NETWORK_RETRY exhausted attempts=$attempt command=$1 '
    'status=$status" >&2;',
    'echo "SN56_NETWORK_RETRY retry=$((attempt + 1))/$attempt_max '
    'delay_seconds=$delay command=$1 status=$status" >&2;',
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


def _retry_calls(text: str) -> list[tuple[int, int, int, str]]:
    """Every ``retry_network`` call as (per_attempt_s, budget_s, attempts, rest).

    Scans the joined RUN instructions only, so dockerfile comments that mention
    the helper by name cannot be mistaken for call sites.
    """
    calls: list[tuple[int, int, int, str]] = []
    for line in _run_instructions(text):
        for match in re.finditer(r"retry_network\s+(\S+)\s+(\S+)\s+(\S+)", line):
            first, second, third = match.groups()
            # Slice the tail instead of capturing it: a greedy trailing group
            # would swallow the next call site in the same RUN instruction.
            rest = line[match.end() : match.end() + 160].strip()
            assert first.isdigit(), f"retry_network call is unbounded: {match.group(0)!r}"
            assert second.isdigit(), f"retry_network call has no budget: {match.group(0)!r}"
            assert third.isdigit(), f"retry_network call has no attempt cap: {match.group(0)!r}"
            calls.append((int(first), int(second), int(third), rest))
    return calls


def _backoff_s(attempts: int) -> int:
    """The loop sleeps 5, 10, 15 ... between attempts."""
    return 5 * attempts * (attempts - 1) // 2


def _command_worst_s(per_attempt: int, budget: int, attempts: int) -> int:
    return min(
        attempts * (per_attempt + _KILL_GRACE_S) + _backoff_s(attempts),
        budget + per_attempt + _KILL_GRACE_S + _backoff_s(attempts),
    )


def _retry_layers(text: str) -> list[str]:
    return [line for line in _run_instructions(text) if "retry_network()" in line]


def _worst_case_seconds(name: str) -> int:
    """Every network command burns every attempt."""
    total = sum(
        _command_worst_s(per_attempt, budget, attempts)
        for per_attempt, budget, attempts, _rest in _retry_calls(_read(name))
    )
    if name == "legacy":
        total += _APT_WORST_S
    return total


def _single_stall_overhead_seconds(name: str) -> int:
    """Extra seconds when ONE command stalls once and the retry then works."""
    return max(
        per_attempt + _KILL_GRACE_S + 5
        for per_attempt, _budget, attempts, _rest in _retry_calls(_read(name))
        if attempts > 1
    )


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
def test_every_retry_network_call_carries_a_timeout_a_budget_and_attempts(
    name: str,
) -> None:
    calls = _retry_calls(_read(name))
    assert calls, f"{name}: no retry_network call sites found"
    for per_attempt, budget, attempts, rest in calls:
        assert 30 <= per_attempt <= MAX_PER_ATTEMPT_CAP_S, (rest, per_attempt)
        assert per_attempt <= budget <= 3 * per_attempt, (rest, per_attempt, budget)
        assert 2 <= attempts <= 3, (rest, attempts)


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_no_cap_is_tight_enough_to_break_a_healthy_build(name: str) -> None:
    """A cap under its step's observed normal maximum manufactures failures."""
    checked = 0
    for per_attempt, _budget, _attempts, rest in _retry_calls(_read(name)):
        for token, observed in OBSERVED_MAX_NORMAL_S.items():
            if token in rest:
                assert per_attempt >= MIN_CAP_RATIO * observed, (
                    f"{name}: cap {per_attempt}s for {token} is below "
                    f"{MIN_CAP_RATIO}x the observed normal max {observed}s"
                )
                checked += 1
                break
    assert checked == len(_retry_calls(_read(name)))


def test_apt_caps_clear_the_observed_apt_layer_maximum() -> None:
    """apt update + install together, against the whole layer's observed max."""
    assert 60 + 150 >= MIN_CAP_RATIO * 49.0


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
    assert "timeout -k 30 60 apt-get -o Acquire::Retries=3 update" in text
    assert (
        "timeout -k 30 150 apt-get -o Acquire::Retries=3 install "
        "-y --no-install-recommends" in text
    )
    # Attempts cut from 5 to 3: five apt attempts cannot fit the 1800 s wall.
    assert 'while [ "$attempt" -le 3 ]' in text
    assert 'if [ "$attempt" -ge 3 ]' in text
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
                    assert re.search(r"retry_network \d+ \d+ \d+ [^;&|]*$", prefix), (
                        f"{name}: unbounded {token!r} in {line[:120]!r}"
                    )
            for match in re.finditer(r"apt-get -o Acquire::Retries=3", line):
                prefix = line[: match.start()]
                assert prefix.rstrip().endswith(
                    tuple(f"timeout -k 30 {n}" for n in (60, 150))
                ), f"unbounded apt-get in {line[:120]!r}"


# --------------------------------------------------------------------------
# Arithmetic: the bound is small enough to be worth having
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_all_stall_worst_case_is_pinned(name: str) -> None:
    worst = _worst_case_seconds(name)
    assert worst == EXPECTED_ALL_STALL_S[name], (
        f"{name} all-stall network seconds changed to {worst}; update "
        "evidence/week9-hazards-20260819/CHANGES.md and this pin together"
    )


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_single_stall_overhead_is_pinned(name: str) -> None:
    overhead = _single_stall_overhead_seconds(name)
    assert overhead == EXPECTED_SINGLE_STALL_OVERHEAD_S[name], (
        f"{name} single-stall overhead changed to {overhead}"
    )


def test_toolkit_absorbs_one_stalled_step_inside_the_validator_window() -> None:
    total = (
        OBSERVED_NORMAL_BUILD_S["toolkit"]
        + EXPECTED_SINGLE_STALL_OVERHEAD_S["toolkit"]
    )
    assert total <= VALIDATOR_BUILD_TIMEOUT_S, total


def test_legacy_single_stall_deficit_is_pinned_not_hidden() -> None:
    """The legacy image cannot absorb one stalled step. Pin the shortfall.

    This is the build-reduction target: cut this many seconds off the legacy
    build and a stalled network step stops being an automatic flux DNF.
    """
    total = (
        OBSERVED_NORMAL_BUILD_S["legacy"]
        + EXPECTED_SINGLE_STALL_OVERHEAD_S["legacy"]
    )
    assert total - VALIDATOR_BUILD_TIMEOUT_S == LEGACY_SINGLE_STALL_DEFICIT_S


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_no_single_attempt_can_eat_the_whole_window(name: str) -> None:
    for per_attempt, _budget, _attempts, rest in _retry_calls(_read(name)):
        assert per_attempt + _KILL_GRACE_S < VALIDATOR_BUILD_TIMEOUT_S / 2, rest


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
    result = _run_helper(f"{helper} retry_network 5 30 3 true; echo rc=$?", shell_env, 30)
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
        f"{helper} retry_network 1 4 3 sleep 300; echo rc=$?", shell_env, 60
    )
    elapsed = time.monotonic() - started

    assert "rc=124" in result.stdout, result.stdout
    assert "SN56_NETWORK_TIMEOUT attempt=1 timeout_seconds=1" in result.stderr
    assert "SN56_NETWORK_RETRY retry=2/3 delay_seconds=5" in result.stderr
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
