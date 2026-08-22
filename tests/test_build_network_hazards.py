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

# MEASURED on the builder production actually uses: the 2026-08-20 timed build,
# DOCKER_BUILDKIT=0 --no-cache, bases pre-pulled.
#   log      evidence/week9-rehearsal-20260819/beta/build-proof/build-567cc73-classic.log
#   extract  evidence/week9-hazards-20260819/measure_classic_build.py
# These REPLACE the BuildKit-derived table this guard used to validate against.
# That was the structural defect the reviewer identified (R4): a guard computed
# from BuildKit numbers cannot catch a cap that is too tight for the classic
# builder.  Each value is the larger of the two images' measurements, and for
# steps whose command is not separately timestamped it is the whole step's
# elapsed time, i.e. an upper bound on the command.
MEASURED_CLASSIC_NORMAL_S = {
    "git": 1.0,                       # both images: fetch+checkout inside 1 s
    "requirements.txt": 215.0,        # legacy 215 s, toolkit 189 s
    "torch==2.6.0": 21.0,             # legacy 21 s, toolkit 4 s
    "torchcodec==0.2.1": 21.0,        # whole step: legacy 21 s, toolkit 15 s
    "image-runtime-lock.txt": 64.0,   # toolkit pip portion; legacy step 112 s
    "flux-tokenizer-download": 25.0,  # whole step, legacy only
}
# A cap below its step's normal duration manufactures failures on healthy
# builds -- which under a hard 1800 s limit with no retry is a self-inflicted
# DNF, strictly worse than the hang it replaced.
MIN_CAP_RATIO = 1.35

# Binding rule adopted 2026-08-20: a single timed build may RAISE a cap, never
# LOWER one -- one measurement has no tail. These are the caps as shipped at
# 567cc73; the guard forbids a future edit from going below them.
CAP_FLOOR_S = {
    "git": 60,
    "requirements.txt": 600,
    "torch==2.6.0": 90,
    "torchcodec==0.2.1": 90,
    "image-runtime-lock.txt": 180,
    "flux-tokenizer-download": 90,
}

# Pinned so that changing a timeout has to move a number a reviewer can see.
# ALL-STALL = every network command burns every attempt.  Both exceed the wall
# and cannot be brought under it without caps that break healthy builds; see
# CHANGES.md ADDENDUM §A2 for the proof and why that is a property of the
# image's own length, not of the retry policy.
EXPECTED_ALL_STALL_S = {"toolkit": 2615, "legacy": 3685}
# SINGLE-STALL = the OBSERVED failure mode: one command stalls, the retry
# succeeds.  This is the number the caps actually control.
EXPECTED_SINGLE_STALL_OVERHEAD_S = {"toolkit": 635, "legacy": 635}
# Latest exact end-to-end warm wall on the validator's classic builder.  The
# legacy datum is the final bd852dc H100 PCIe nocache build (2026-08-20); 1326 s
# was the earlier A100-host observation, and the 245 s host spread is larger
# than the latest warm margin.  One build may raise, never lower, timeout caps.
# Source: evidence/week9-rehearsal-20260819/alpha/vm-evidence-mirror/
# legacy-build-timing.txt (sealed; cited here, never edited by this repair).
MEASURED_NORMAL_BUILD_S = {"toolkit": 678, "legacy": 1571}
LEGACY_PRIOR_WARM_BUILD_S = 1326
LEGACY_OBSERVED_HOST_SPREAD_S = 245
LEGACY_WARM_MARGIN_S = 229
# Cost of a first build on a host whose image store is empty.  The timed run's
# own cold-pull attempt did not evict the bases (PULL_AITOOLKIT_s=1), so it is
# not a cold observation; this is the 2026-08-19 rehearsal's measured pull of
# BOTH bases, 14:07:41Z -> 14:12:13Z (beta/setup/build-images.log).
BASE_PULL_S = 272
# The legacy image still cannot absorb a stall on its most expensive step.
# Pinned so the week-10 build-reduction work has a target.
LEGACY_SINGLE_STALL_DEFICIT_S = 406
LEGACY_COLD_BASELINE_DEFICIT_S = 43
# Exact 2026-08-22 classic `--no-cache` timing evidence from the same
# production-class builder.  The 1872.77 s baseline had a warm ai-toolkit base
# and cold Kohya base; it is explicitly not an empty-store measurement.  The
# first consolidated experiment (7bad549) used the normal store and finished in
# 1521.23 s.  Neither is the required disposable empty-store release gate.
LEGACY_WARM_AI_COLD_KOHYA_CLASSIC_NO_CACHE_S = 1872.77
LEGACY_CONSOLIDATED_NORMAL_STORE_EXPERIMENT_S = 1521.23
# Exact 6f0d0d1 disposable empty-store build: the hard release gate stopped it
# while classic Docker was copying the final ai-toolkit Python tree. All base
# pulls, network installs, and lock verifiers had already passed.
LEGACY_EMPTY_STORE_6F0_GATE_S = 1680.03
# The optimized 80aa841 gate reached the merged offline verifier before its
# hard stop. Its tiny cross-stage /opt copy cost 111.895 s. The next empty-store
# gate (329eaed) proved that direct context sourcing in the same post-transplant
# position still cost 109.946 s: source choice recovered only 1.949 s. The
# measured root grew from 29.2 GB to 43.3 GB across the runtime transplants, so
# the next repair moves both context layers before that growth without changing
# their bytes or final paths.
LEGACY_EMPTY_STORE_80AA841_GATE_S = 1680.005
LEGACY_CROSS_STAGE_OPT_COPY_S = 111.895
LEGACY_EMPTY_STORE_329EAED_GATE_S = 1680.005
LEGACY_POST_TRANSPLANT_CONTEXT_OPT_COPY_S = 109.946
LEGACY_SOURCE_ONLY_RECOVERY_S = 1.949
LEGACY_KOHYA_ROOT_BEFORE_TRANSPLANTS_GB = 29.2
LEGACY_FINAL_ROOT_AFTER_TRANSPLANTS_GB = 43.3
LEGACY_PIP_WHEEL_COPY_COMMITS_S = 112 + 20 + 19 + 19
LEGACY_REMOVABLE_OPT_COPY_COMMITS_S = 19 + 19
LEGACY_REDUNDANT_WORKDIR_COMMIT_S = 22
LEGACY_STAGING_RUN_ALLOWANCE_S = 20
LEGACY_REQUIRED_COLD_MARGIN_S = 120
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


def _token_of(rest: str) -> str | None:
    for token in MEASURED_CLASSIC_NORMAL_S:
        if token in rest:
            return token
    return None


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_no_cap_is_tight_enough_to_break_a_healthy_build(name: str) -> None:
    """The guard the reviewer asked for: validated against the CLASSIC table.

    Every cap must clear MIN_CAP_RATIO x the duration that step actually took
    on the builder production uses.  Computed from BuildKit numbers this test
    could not have caught a classic-too-tight cap at all.
    """
    calls = _retry_calls(_read(name))
    checked = 0
    for per_attempt, _budget, _attempts, rest in calls:
        token = _token_of(rest)
        assert token is not None, f"{name}: unmapped retry_network call {rest[:60]!r}"
        measured = MEASURED_CLASSIC_NORMAL_S[token]
        assert per_attempt >= MIN_CAP_RATIO * measured, (
            f"{name}: cap {per_attempt}s for {token} is below {MIN_CAP_RATIO}x "
            f"the MEASURED classic normal {measured}s -- it would fire on a "
            "healthy build"
        )
        checked += 1
    assert checked == len(calls)


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_caps_are_never_lowered_below_the_measured_baseline(name: str) -> None:
    """Binding rule (2026-08-20): a single timed build may only RAISE a cap."""
    for per_attempt, _budget, _attempts, rest in _retry_calls(_read(name)):
        token = _token_of(rest)
        assert per_attempt >= CAP_FLOOR_S[token], (
            f"{name}: cap {per_attempt}s for {token} is below the 567cc73 "
            f"floor {CAP_FLOOR_S[token]}s; lowering a cap needs new evidence, "
            "not one timed build"
        )


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_measured_cap_headroom_is_pinned(name: str) -> None:
    """Pin the actual margins so a future edit has to move a visible number."""
    margins = {
        _token_of(rest): round(per_attempt / MEASURED_CLASSIC_NORMAL_S[_token_of(rest)], 2)
        for per_attempt, _b, _a, rest in _retry_calls(_read(name))
    }
    assert min(margins.values()) >= 2.5, margins
    assert margins["requirements.txt"] == 2.79, margins


def test_apt_caps_clear_the_measured_apt_step() -> None:
    """apt update + install together, against the measured classic step (55 s)."""
    assert 60 + 150 >= MIN_CAP_RATIO * 55.0


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


def _stall_overhead_s(per_attempt: int) -> int:
    """Seconds wasted when a step stalls once and the retry then succeeds.

    The budget is a START-GATE, not a ceiling: an attempt that begins before
    the budget expires runs to its full per-attempt cap (OBSERVED in the
    negative control -- 195 s elapsed against a 150 s budget).  So the wasted
    time is the whole cap plus the SIGKILL grace plus one backoff sleep.
    """
    return per_attempt + _KILL_GRACE_S + 5


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_measured_warm_build_fits_the_validator_window(name: str) -> None:
    warm = MEASURED_NORMAL_BUILD_S[name]
    assert warm <= VALIDATOR_BUILD_TIMEOUT_S, warm


def test_latest_legacy_warm_margin_and_observed_host_spread_are_pinned() -> None:
    warm = MEASURED_NORMAL_BUILD_S["legacy"]
    assert warm == 1571
    assert VALIDATOR_BUILD_TIMEOUT_S - warm == LEGACY_WARM_MARGIN_S
    assert warm - LEGACY_PRIOR_WARM_BUILD_S == LEGACY_OBSERVED_HOST_SPREAD_S
    assert LEGACY_OBSERVED_HOST_SPREAD_S > LEGACY_WARM_MARGIN_S


def test_toolkit_cold_estimate_still_fits_the_validator_window() -> None:
    cold = MEASURED_NORMAL_BUILD_S["toolkit"] + BASE_PULL_S
    assert cold == 950
    assert cold <= VALIDATOR_BUILD_TIMEOUT_S


def test_legacy_cold_estimate_is_hold_not_a_claimed_pass() -> None:
    """1571 warm + measured 272 pull = 1843 estimate, not a cold observation."""
    cold_estimate = MEASURED_NORMAL_BUILD_S["legacy"] + BASE_PULL_S
    assert cold_estimate == 1843
    assert cold_estimate - VALIDATOR_BUILD_TIMEOUT_S == LEGACY_COLD_BASELINE_DEFICIT_S
    assert cold_estimate > VALIDATOR_BUILD_TIMEOUT_S


def test_legacy_layer_consolidation_has_a_measured_timing_rationale() -> None:
    """The empty-store gate is justified, not replaced, by measured overhead."""
    removed = (
        LEGACY_PIP_WHEEL_COPY_COMMITS_S
        + LEGACY_REMOVABLE_OPT_COPY_COMMITS_S
        + LEGACY_REDUNDANT_WORKDIR_COMMIT_S
    )
    projected = (
        LEGACY_WARM_AI_COLD_KOHYA_CLASSIC_NO_CACHE_S
        - removed
        + LEGACY_STAGING_RUN_ALLOWANCE_S
    )
    assert removed == 230
    assert projected == pytest.approx(1662.77)
    assert LEGACY_CONSOLIDATED_NORMAL_STORE_EXPERIMENT_S <= projected
    assert (
        VALIDATOR_BUILD_TIMEOUT_S - LEGACY_CONSOLIDATED_NORMAL_STORE_EXPERIMENT_S
        >= LEGACY_REQUIRED_COLD_MARGIN_S
    )
    assert LEGACY_EMPTY_STORE_6F0_GATE_S > 1680
    assert LEGACY_EMPTY_STORE_6F0_GATE_S < VALIDATOR_BUILD_TIMEOUT_S
    assert LEGACY_EMPTY_STORE_80AA841_GATE_S > 1680
    assert (
        LEGACY_CROSS_STAGE_OPT_COPY_S - LEGACY_POST_TRANSPLANT_CONTEXT_OPT_COPY_S
        == pytest.approx(LEGACY_SOURCE_ONLY_RECOVERY_S)
    )
    assert LEGACY_SOURCE_ONLY_RECOVERY_S < 2
    assert LEGACY_EMPTY_STORE_329EAED_GATE_S > 1680
    assert (
        LEGACY_FINAL_ROOT_AFTER_TRANSPLANTS_GB
        - LEGACY_KOHYA_ROOT_BEFORE_TRANSPLANTS_GB
        == pytest.approx(14.1)
    )


def test_legacy_layer_consolidation_preserves_exact_runtime_paths() -> None:
    text = _read("legacy")
    required_pairs = (
        ("pip", "pip"),
        ("pip-22.0.2.dist-info", "pip-22.0.2.dist-info"),
        ("wheel", "wheel"),
        ("wheel-0.37.1.egg-info", "wheel-0.37.1.egg-info"),
    )
    assert "site=/usr/local/lib/python3.10/dist-packages" in text
    for source, target in required_pairs:
        assert f'test ! -e "$site/{target}"' in text
        assert (
            f"cp -a /usr/lib/python3/dist-packages/{source} "
            f'"$site/{target}"'
        ) in text
        assert (
            "COPY --from=aitoolkit-runtime "
            f"/usr/lib/python3/dist-packages/{source}/"
        ) not in text
    phase3 = next(
        layer
        for layer in _run_instructions(text)
        if "--requirement /opt/sn56/image-runtime-lock.txt" in layer
    )
    assert "python3 /opt/sn56/verify-image-runtime.py" in phase3
    for source, target in required_pairs:
        assert (
            f"cp -a /usr/lib/python3/dist-packages/{source} "
            f'\"$site/{target}\"'
        ) in phase3
    assert text.count("COPY --from=aitoolkit-runtime") == 2
    first_stage = text.split(
        "FROM diagonalge/kohya_latest:latest@sha256:", maxsplit=1
    )[0]
    assert first_stage.count("\nCOPY ") == 1
    assert first_stage.count("\nRUN ") == 1
    assert "\nWORKDIR " not in first_stage
    assert (
        "COPY ops/docker/image-runtime-lock.txt \\\n"
        "    ops/docker/image-runtime-phase1-constraints.txt \\\n"
        "    ops/docker/verify_image_runtime.py \\\n"
        "    /opt/sn56/"
    ) in first_stage
    ordered = (
        "python3 /opt/sn56/verify-image-runtime.py",
        "--files-only",
        "retry_network 60 150 3 git fetch",
        "--requirement requirements.txt",
        "torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0",
        "torchcodec==0.2.1 pyyaml Pillow numpy safetensors",
        "--requirement /opt/sn56/image-runtime-lock.txt",
        "cp -a /usr/lib/python3/dist-packages/pip",
        'test "$(git rev-parse HEAD)"',
    )
    first_stage_run = _run_instructions(first_stage)[0]
    offsets = [first_stage_run.index(token) for token in ordered]
    lock_offset = offsets[-3]
    locked_verify_offset = first_stage_run.index(
        "python3 /opt/sn56/verify-image-runtime.py", lock_offset
    )
    offsets.insert(-2, locked_verify_offset)
    assert offsets == sorted(offsets)
    final_stage = text.split(
        "FROM diagonalge/kohya_latest:latest@sha256:", maxsplit=1
    )[1]
    assert (
        "COPY ops/docker/image-runtime-lock.txt \\\n"
        "    ops/docker/image-runtime-phase1-constraints.txt \\\n"
        "    ops/docker/verify_image_runtime.py \\\n"
        "    /opt/sn56/"
    ) in final_stage
    assert "\nWORKDIR " not in final_stage
    assert final_stage.count("\nRUN ") == 1
    opt_copy = final_stage.index("COPY ops/docker/image-runtime-lock.txt")
    forge_copy = final_stage.index("COPY forge/ /app/forge/")
    source_copy = final_stage.index(
        "COPY --from=aitoolkit-runtime /app/ai-toolkit/ /app/ai-toolkit/"
    )
    python_copy = final_stage.index(
        "COPY --from=aitoolkit-runtime "
        "/usr/local/lib/python3.10/dist-packages/ "
        "/opt/sn56/ai-toolkit-python/"
    )
    final_run_offset = final_stage.index("RUN set -eu;")
    assert opt_copy < forge_copy < source_copy < python_copy < final_run_offset
    final_verifier = next(
        layer
        for layer in _run_instructions(final_stage)
        if "python3 -m forge.flux_kohya_tokenizers stage" in layer
    )
    final_order = (
        "test -f /opt/sn56/image-runtime-lock.txt",
        "test ! -e /opt/sn56/legacy-aitoolkit-toolchain-lock.txt",
        "chmod 0644 /opt/sn56/image-runtime-lock.txt "
        "/opt/sn56/image-runtime-phase1-constraints.txt "
        "/opt/sn56/verify_image_runtime.py",
        "SN56_NETWORK_TIMEOUT unavailable=timeout command=apt-toolchain",
        "timeout -k 30 60 apt-get",
        "timeout -k 30 150 apt-get",
        ">/opt/sn56/legacy-aitoolkit-toolchain-lock.txt",
        ">/opt/sn56/legacy-os-package-inventory.txt",
        ">/opt/sn56/legacy-os-package-inventory.sha256",
        "rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*",
        "mv /opt/sn56/verify_image_runtime.py",
        "python3 /opt/sn56/verify-image-runtime.py",
        "sha256sum --check --strict",
        "assert torch.__version__ == '2.1.2+cu121'",
        "retry_network 90 120 2 env",
        "python3 -m forge.flux_kohya_tokenizers stage",
        "python3 -m forge.flux_kohya_tokenizers verify",
        "python3 -m forge.verify_flux_kohya_runtime",
    )
    final_offsets = [final_verifier.index(token) for token in final_order]
    assert final_offsets == sorted(final_offsets)


def test_toolkit_absorbs_one_stalled_step_inside_the_validator_window() -> None:
    """Re-pinned against the MEASURED classic wall (678 s), not a model.

    Supersedes the provisional constant that made this green on an optimistic
    1.91x ratio; the standing warning attached to it is withdrawn.
    """
    warm = MEASURED_NORMAL_BUILD_S["toolkit"]
    assert warm + EXPECTED_SINGLE_STALL_OVERHEAD_S["toolkit"] == 1313
    assert warm + EXPECTED_SINGLE_STALL_OVERHEAD_S["toolkit"] <= VALIDATOR_BUILD_TIMEOUT_S
    # and still fits on a cold image store
    cold = warm + BASE_PULL_S
    assert cold + EXPECTED_SINGLE_STALL_OVERHEAD_S["toolkit"] <= VALIDATOR_BUILD_TIMEOUT_S


def test_legacy_single_stall_deficit_is_pinned_not_hidden() -> None:
    """The legacy image still cannot absorb a stall on its most expensive step.

    Latest measured warm wall: 1571 + 635 = 2206 s, 406 s over. This is the
    week-10 build-reduction target.
    """
    total = (
        MEASURED_NORMAL_BUILD_S["legacy"]
        + EXPECTED_SINGLE_STALL_OVERHEAD_S["legacy"]
    )
    assert total - VALIDATOR_BUILD_TIMEOUT_S == LEGACY_SINGLE_STALL_DEFICIT_S


def test_legacy_warm_retry_stalls_except_requirements_still_fit() -> None:
    """Warm retry-network outcomes use the latest exact 1571 s baseline."""
    warm = MEASURED_NORMAL_BUILD_S["legacy"]
    survivable, fatal = [], []
    for per_attempt, _b, _a, rest in _retry_calls(_read("legacy")):
        token = _token_of(rest)
        total = warm + _stall_overhead_s(per_attempt)
        (survivable if total <= VALIDATOR_BUILD_TIMEOUT_S else fatal).append(token)
    # the apt loop is not a retry_network call; one failed attempt costs
    # (60+30) + (150+30) + 5 backoff
    # The separate apt loop is now also fatal: 1571 + 275 = 1846.
    assert warm + 60 + _KILL_GRACE_S + 150 + _KILL_GRACE_S + 5 > VALIDATOR_BUILD_TIMEOUT_S
    assert fatal == ["requirements.txt"], (survivable, fatal)
    assert set(survivable) == {
        "git", "torch==2.6.0", "torchcodec==0.2.1",
        "image-runtime-lock.txt", "flux-tokenizer-download",
    }


def test_legacy_cold_hold_cannot_absorb_any_stall() -> None:
    """The modeled cold baseline is already 43 s outside the build window.

    This is 1571 s observed warm plus a separately measured 272 s base pull,
    not a measured cold run.  It is a build-length HOLD; changing retry caps
    on one datum cannot make the cold baseline fit.
    """
    cold = MEASURED_NORMAL_BUILD_S["legacy"] + BASE_PULL_S
    assert cold == 1843
    assert cold > VALIDATOR_BUILD_TIMEOUT_S
    survivable = {
        _token_of(rest)
        for per_attempt, _b, _a, rest in _retry_calls(_read("legacy"))
        if cold + _stall_overhead_s(per_attempt) <= VALIDATOR_BUILD_TIMEOUT_S
    }
    assert survivable == set()
    # The apt loop's single failed attempt also cannot fit.
    assert cold + 60 + _KILL_GRACE_S + 150 + _KILL_GRACE_S + 5 > VALIDATOR_BUILD_TIMEOUT_S


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
