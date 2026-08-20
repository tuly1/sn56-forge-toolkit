"""Week-9 HAZARD 2: no Kohya child may outlive the FLUX fallback.

``flux_kohya.run`` may fall back to ``aitoolkit.run`` after an ordinary Kohya
failure only when shutdown is positively verified.  Both trainers use the
whole GPU, so an unverified child must instead surface a containment error to
the outer non-trainer fallback; concurrent trainers mean VRAM exhaustion or a
corrupted artifact — a forfeit for the task either way.

week9-rc only terminated the child on the clean deadline path: an exception
anywhere in the poll loop, or a ``_terminate`` whose final ``wait`` timed out,
propagated straight past cleanup and into the fallback with the trainer still
running.  These tests spawn REAL child processes and fail on the week9-rc code.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import types

import pytest

import forge.tasks
from forge.data.schema import ImageSpec
from forge.tasks import flux_kohya


def _spawn_sleeper() -> subprocess.Popen:
    """A real child that will not exit on its own inside the test's lifetime."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _spec() -> ImageSpec:
    return ImageSpec.build(
        task_id="flux-task",
        model="org/snapshot-flux",
        model_type="flux",
        expected_repo_name="flux-output",
        trigger_word="TOK",
        dataset_zip=None,
    )


@pytest.fixture(autouse=True)
def _clean_registry():
    """No failed-containment double may leak into the next test."""
    flux_kohya.reap_kohya_children("test_setup")
    yield
    flux_kohya.reap_kohya_children("test_teardown")
    # Stubborn/throwing doubles intentionally remain owned.  Product code must
    # retain them; the test harness alone may discard them after its assertion.
    with flux_kohya._ACTIVE_CHILDREN_LOCK:
        leftovers = list(flux_kohya._ACTIVE_KOHYA_CHILDREN)
        flux_kohya._ACTIVE_KOHYA_CHILDREN.clear()
    for child in leftovers:
        for identity in child.descendants.values():
            try:
                flux_kohya._signal_process_identity(identity, signal.SIGKILL)
            except Exception:
                pass
        try:
            if child.proc.poll() is None:
                child.proc.kill()
                child.proc.wait(timeout=10)
        except Exception:
            pass


class _Deadline:
    def __init__(self, remaining: float = 3600.0, raise_after: int | None = None):
        self._remaining = remaining
        self._raise_after = raise_after
        self.calls = 0

    def remaining(self) -> float:
        self.calls += 1
        if self._raise_after is not None and self.calls > self._raise_after:
            raise RuntimeError("clock read failed mid-training")
        return self._remaining


def _fake_aitoolkit(observations: list) -> types.ModuleType:
    module = types.ModuleType("forge.tasks.aitoolkit")

    def run(spec, deadline):  # noqa: ANN001 — signature mirrors the real one
        observations.append(
            [
                (proc.pid, proc.poll())
                for proc in getattr(run, "watched", [])
            ]
        )

    module.run = run
    return module


def _install_fake_aitoolkit(monkeypatch, module) -> None:
    """Replace the lazily-imported ai-toolkit module.

    Both halves are required: ``from forge.tasks import aitoolkit`` resolves
    the already-bound package ATTRIBUTE when some earlier test in the session
    has imported the real module, and the sys.modules entry when it has not.
    """
    monkeypatch.setitem(sys.modules, "forge.tasks.aitoolkit", module)
    monkeypatch.setattr(forge.tasks, "aitoolkit", module, raising=False)


# --------------------------------------------------------------------------
# The barrier: every fallback path proves the GPU is free first
# --------------------------------------------------------------------------


def test_started_leader_and_escaped_descendant_are_gone_before_fallback(
    monkeypatch, tmp_path
):
    """Popen -> live setsid descendant -> exception -> reap -> fallback.

    This exercises the transition the empty-runtime smoke missed: the real
    poll loop has started a leader and an escaped descendant before the
    injected failure.  The real snapshot failure handler may reach ai-toolkit
    only after both identities are positively gone.
    """
    script_root = tmp_path / "sd-scripts"
    script_root.mkdir()
    (script_root / "flux_train_network.py").write_text("# test", encoding="utf-8")
    spawned: list[subprocess.Popen] = []
    events: list[tuple[str, dict]] = []
    observations: list = []
    marker = tmp_path / "descendant.pid"
    saw_live_pair = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        spawned.append(proc)
        return proc

    fake = types.ModuleType("forge.tasks.aitoolkit")

    def fallback_run(spec, deadline):  # noqa: ANN001
        descendant_pid = int(marker.read_text().strip())
        observations.append((spawned[0].poll(), _alive(descendant_pid)))

    fake.run = fallback_run

    _install_fake_aitoolkit(monkeypatch, fake)
    monkeypatch.setattr(flux_kohya.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(flux_kohya, "_SD_SCRIPTS_DIR", str(script_root))
    monkeypatch.setattr(flux_kohya, "_log_path", lambda spec: str(tmp_path / "k.log"))
    monkeypatch.setattr(flux_kohya, "_start_gpu_sampler", lambda *args: None)
    monkeypatch.setattr(flux_kohya, "_POLL_SECONDS", 0.05)
    monkeypatch.setattr(
        flux_kohya,
        "_command",
        lambda config_path, script=None: [
            sys.executable,
            "-c",
            (
                "import subprocess, sys, time\n"
                "child = subprocess.Popen(\n"
                "    [sys.executable, '-c', 'import time; time.sleep(300)'],\n"
                "    start_new_session=True,\n"
                ")\n"
                f"open({str(marker)!r}, 'w').write(str(child.pid))\n"
                "time.sleep(300)\n"
            ),
        ],
    )
    monkeypatch.setattr(
        flux_kohya,
        "resolve_flux_cache_layout",
        lambda cached_model_dir: (flux_kohya._SNAPSHOT_LAYOUT, None),
    )
    monkeypatch.setattr(
        flux_kohya,
        "resolve_snapshot_kohya_checkpoint",
        lambda cached_model_dir: "/cache/flux1-dev.safetensors",
    )
    monkeypatch.setattr(flux_kohya, "_kohya_runtime_ready", lambda: (True, "ready"))

    def train(spec, deadline, base_model, *, layout):
        # Everything _train_with_kohya does before the child is irrelevant to
        # this hazard; what matters is that the real poll loop runs and then
        # something inside it fails while the trainer is alive.
        flux_kohya._run_kohya(str(tmp_path / "config.toml"), deadline, spec, {})
        return True

    monkeypatch.setattr(flux_kohya, "_train_with_kohya", train)
    monkeypatch.setattr(flux_kohya.telemetry, "set_meta", lambda **values: None)
    monkeypatch.setattr(flux_kohya.telemetry, "sample", lambda *args: None)
    monkeypatch.setattr(
        flux_kohya.telemetry,
        "event",
        lambda name, **values: events.append((name, values)),
    )

    def register_then_inject(proc, *, token=None):
        # Fail before registration appends anything or the assignment to
        # _run_kohya's local `child` completes.  Finally must recover from the
        # raw Popen descriptor and inherited token; an empty registry cannot
        # be mistaken for proof that the GPU is free.
        wait_until = time.monotonic() + 10
        while time.monotonic() < wait_until:
            if marker.exists() and marker.read_text().strip():
                descendant_pid = int(marker.read_text().strip())
                if proc.poll() is None and _alive(descendant_pid):
                    saw_live_pair.append((proc.pid, descendant_pid))
                    raise RuntimeError("injected failure after Popen before return")
            time.sleep(0.02)
        pytest.fail("leader/escaped descendant never became live")

    monkeypatch.setattr(flux_kohya, "_register_kohya_child", register_then_inject)

    try:
        flux_kohya.run(_spec(), _Deadline())

        assert saw_live_pair, "failure was not injected after both processes started"
        assert observations == [(spawned[0].returncode, False)]
        assert spawned[0].returncode is not None, "leader descriptor was not reaped"
        assert [name for name, _ in events if name == "flux_snapshot_kohya_fallback"]
        assert flux_kohya._ACTIVE_KOHYA_CHILDREN == []
    finally:
        if marker.exists() and marker.read_text().strip():
            try:
                os.kill(int(marker.read_text().strip()), signal.SIGKILL)
            except OSError:
                pass
        for proc in spawned:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)


def test_the_barrier_is_silent_when_no_child_was_left_behind(monkeypatch):
    """Byte-identical fallback contract: a clean skip emits no new event."""
    events: list[tuple[str, dict]] = []
    observations: list = []
    _install_fake_aitoolkit(monkeypatch, _fake_aitoolkit(observations))
    monkeypatch.setattr(
        flux_kohya,
        "resolve_flux_cache_layout",
        lambda cached_model_dir: (flux_kohya._SNAPSHOT_LAYOUT, None),
    )
    monkeypatch.setenv("FORGE_FLUX_SNAPSHOT_BACKEND", "aitoolkit")
    monkeypatch.setattr(flux_kohya.telemetry, "set_meta", lambda **values: None)
    monkeypatch.setattr(
        flux_kohya.telemetry,
        "event",
        lambda name, **values: events.append((name, values)),
    )

    flux_kohya.run(_spec(), _Deadline())

    assert observations, "the opt-out path must still reach ai-toolkit"
    assert [event for event in events if event[0] == "kohya_child_reaped"] == []


# --------------------------------------------------------------------------
# _run_kohya cleans up on the paths week9-rc did not cover
# --------------------------------------------------------------------------


def test_an_exception_in_the_poll_loop_still_reaps_the_child(monkeypatch, tmp_path):
    script_root = tmp_path / "sd-scripts"
    script_root.mkdir()
    (script_root / "flux_train_network.py").write_text("# test", encoding="utf-8")
    spawned: list[subprocess.Popen] = []
    events: list[tuple[str, dict]] = []

    monkeypatch.setattr(flux_kohya, "_SD_SCRIPTS_DIR", str(script_root))
    monkeypatch.setattr(flux_kohya, "_log_path", lambda spec: str(tmp_path / "k.log"))
    monkeypatch.setattr(flux_kohya, "_start_gpu_sampler", lambda *args: None)
    monkeypatch.setattr(flux_kohya, "_POLL_SECONDS", 0.05)
    monkeypatch.setattr(
        flux_kohya,
        "_command",
        lambda config_path, script=None: [
            sys.executable,
            "-c",
            "import time; time.sleep(300)",
        ],
    )
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        spawned.append(proc)
        return proc

    # Recorded through Popen, not through the new registry, so this test also
    # runs (and fails on its own assertion) against the week9-rc module.
    monkeypatch.setattr(flux_kohya.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(
        flux_kohya.telemetry,
        "event",
        lambda name, **values: events.append((name, values)),
    )
    monkeypatch.setattr(flux_kohya.telemetry, "sample", lambda *args: None)

    with pytest.raises(RuntimeError, match="clock read failed"):
        flux_kohya._run_kohya(
            str(tmp_path / "config.toml"),
            _Deadline(raise_after=1),
            _spec(),
            {},
        )

    assert spawned, "no Kohya child was spawned"
    child = spawned[0]
    assert child.poll() is not None, "the child outlived the failing _run_kohya"
    assert child.returncode is not None
    assert ("kohya_child_reaped", {
        "reason": "run_kohya_exit",
        "reaped": 1,
        "survived": 0,
    }) in events
    assert getattr(flux_kohya, "_ACTIVE_KOHYA_CHILDREN", []) == []


def test_deadline_path_reaps_the_started_child(monkeypatch, tmp_path):
    script_root = tmp_path / "sd-scripts"
    script_root.mkdir()
    (script_root / "flux_train_network.py").write_text("# test", encoding="utf-8")
    spawned: list[subprocess.Popen] = []
    events: list[tuple[str, dict]] = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        spawned.append(proc)
        return proc

    class DeadlineStop:
        calls = 0

        def remaining(self):
            self.calls += 1
            return 3600.0 if self.calls == 1 else 0.0

    monkeypatch.setattr(flux_kohya, "_SD_SCRIPTS_DIR", str(script_root))
    monkeypatch.setattr(flux_kohya, "_log_path", lambda spec: str(tmp_path / "k.log"))
    monkeypatch.setattr(flux_kohya, "_start_gpu_sampler", lambda *args: None)
    monkeypatch.setattr(flux_kohya.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(
        flux_kohya,
        "_command",
        lambda config_path, script=None: [
            sys.executable,
            "-c",
            "import time; time.sleep(300)",
        ],
    )
    monkeypatch.setattr(
        flux_kohya.telemetry,
        "event",
        lambda name, **values: events.append((name, values)),
    )
    monkeypatch.setattr(flux_kohya.telemetry, "sample", lambda *args: None)

    flux_kohya._run_kohya(
        str(tmp_path / "config.toml"), DeadlineStop(), _spec(), {}
    )

    assert spawned and spawned[0].returncode is not None
    assert flux_kohya._ACTIVE_KOHYA_CHILDREN == []
    end = next(values for name, values in events if name == "kohya_end")
    assert end["stopped_by_deadline"] is True


def test_terminate_reaps_a_real_child_and_reports_success():
    child = _spawn_sleeper()
    started = time.monotonic()
    assert flux_kohya._terminate(child) is True
    assert child.returncode is not None
    assert time.monotonic() - started < 10


def test_terminate_never_raises_when_the_child_cannot_be_confirmed_dead(monkeypatch):
    """week9-rc let this TimeoutExpired escape and skip every cleanup path."""
    signals: list[int] = []

    class Unkillable:
        pid = 4242
        returncode = None

        def poll(self):
            return None

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("kohya", timeout)

    identity = flux_kohya._ProcessIdentity(4242, "proc:stubborn")
    info = flux_kohya._ProcessInfo(
        identity=identity,
        ppid=1,
        pgid=4242,
        state="D",
    )
    monkeypatch.setattr(
        flux_kohya,
        "_process_table",
        lambda: {4242: info},
    )
    monkeypatch.setattr(flux_kohya, "_process_info", lambda pid: info)
    monkeypatch.setattr(
        flux_kohya.os,
        "killpg",
        lambda pgid, sig: signals.append(sig),
    )

    assert flux_kohya._terminate(Unkillable()) is False
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_reused_descendant_pid_is_never_signalled(monkeypatch):
    old = flux_kohya._ProcessIdentity(4242, "proc:old-start")
    replacement = flux_kohya._ProcessInfo(
        identity=flux_kohya._ProcessIdentity(4242, "proc:new-start"),
        ppid=1,
        pgid=4242,
        state="S",
    )
    opened: list[int] = []
    sent: list[tuple[int, int]] = []
    closed: list[int] = []
    monkeypatch.setattr(flux_kohya.sys, "platform", "linux")
    monkeypatch.setattr(flux_kohya, "_process_info", lambda pid: replacement)
    monkeypatch.setattr(
        flux_kohya.os, "pidfd_open", lambda pid: opened.append(pid) or 77, raising=False
    )
    monkeypatch.setattr(
        flux_kohya.signal,
        "pidfd_send_signal",
        lambda fd, sig, info, flags: sent.append((fd, sig)),
        raising=False,
    )
    monkeypatch.setattr(flux_kohya.os, "close", lambda fd: closed.append(fd))
    monkeypatch.setattr(
        flux_kohya.os,
        "kill",
        lambda *args: pytest.fail("numeric os.kill used on Linux"),
    )

    assert flux_kohya._signal_process_identity(old, signal.SIGKILL) is True
    assert opened == [4242]
    assert sent == [], "PID-reused replacement was mistaken for Kohya"
    assert closed == [77]


def test_linux_descendant_signal_uses_identity_bound_pidfd(monkeypatch):
    identity = flux_kohya._ProcessIdentity(4545, "proc:owned")
    info = flux_kohya._ProcessInfo(
        identity=identity,
        ppid=1,
        pgid=4545,
        state="S",
    )
    sent: list[tuple[int, int, object, int]] = []
    monkeypatch.setattr(flux_kohya.sys, "platform", "linux")
    monkeypatch.setattr(flux_kohya, "_process_info", lambda pid: info)
    monkeypatch.setattr(flux_kohya.os, "pidfd_open", lambda pid: 88, raising=False)
    monkeypatch.setattr(
        flux_kohya.signal,
        "pidfd_send_signal",
        lambda fd, sig, siginfo, flags: sent.append((fd, sig, siginfo, flags)),
        raising=False,
    )
    monkeypatch.setattr(flux_kohya.os, "close", lambda fd: None)
    monkeypatch.setattr(
        flux_kohya.os,
        "kill",
        lambda *args: pytest.fail("numeric os.kill used on Linux"),
    )

    assert flux_kohya._signal_process_identity(identity, signal.SIGTERM) is True
    assert sent == [(88, signal.SIGTERM, None, 0)]


def test_reused_leader_pid_blocks_handoff_without_signalling_group(monkeypatch):
    old = flux_kohya._ProcessIdentity(4343, "proc:original-leader")
    replacement = flux_kohya._ProcessInfo(
        identity=flux_kohya._ProcessIdentity(4343, "proc:replacement"),
        ppid=1,
        pgid=4343,
        state="S",
    )

    class ClaimsLive:
        pid = 4343

        def poll(self):
            return None

    child = flux_kohya._KohyaChild(
        proc=ClaimsLive(),
        pid=4343,
        token=None,
        leader_identity=old,
        pgid=4343,
    )
    group_signals: list[tuple[int, int]] = []
    monkeypatch.setattr(flux_kohya, "_process_table", lambda: {4343: replacement})
    monkeypatch.setattr(flux_kohya, "_process_info", lambda pid: replacement)
    monkeypatch.setattr(
        flux_kohya.os,
        "killpg",
        lambda pgid, sig: group_signals.append((pgid, sig)),
    )
    with flux_kohya._ACTIVE_CHILDREN_LOCK:
        flux_kohya._ACTIVE_KOHYA_CHILDREN.append(child)

    with pytest.raises(flux_kohya.KohyaContainmentError, match="leader_pids=\\[4343\\]"):
        flux_kohya._require_kohya_quiescent("pid_reuse_probe")

    assert group_signals == [], "PID-reused process group was signalled"
    assert flux_kohya._ACTIVE_KOHYA_CHILDREN == [child]


def test_descendant_verification_uncertainty_is_not_success(monkeypatch):
    identity = flux_kohya._ProcessIdentity(5151, "proc:owned")

    class ReapedLeader:
        pid = 4141

        def poll(self):
            return 0

    child = flux_kohya._KohyaChild(
        proc=ReapedLeader(),
        pid=4141,
        token="owned-token",
        descendants={identity.pid: identity},
    )
    monkeypatch.setattr(
        flux_kohya,
        "_refresh_kohya_identities",
        lambda record: (_ for _ in ()).throw(
            flux_kohya._ProcessInspectionError("injected unreadable identity")
        ),
    )

    assert flux_kohya._terminate(child, sweep_descendants=True) is False
    assert "injected unreadable identity" in child.last_error


def test_descendant_barrier_requires_two_consecutive_quiet_scans(monkeypatch):
    late = flux_kohya._ProcessIdentity(5252, "proc:late-fork")
    late_info = flux_kohya._ProcessInfo(
        identity=late,
        ppid=1,
        pgid=5252,
        state="S",
    )

    class ReapedLeader:
        pid = 5151

        def poll(self):
            return 0

    child = flux_kohya._KohyaChild(
        proc=ReapedLeader(), pid=5151, token="quiet-scan-token"
    )
    scans = []
    signals = []

    def refresh(record):
        scans.append(len(scans) + 1)
        if len(scans) == 2:
            record.descendants[late.pid] = late
            return {late.pid: late_info}
        return {}

    monkeypatch.setattr(flux_kohya, "_refresh_kohya_identities", refresh)
    monkeypatch.setattr(
        flux_kohya,
        "_signal_process_identity",
        lambda identity, sig: signals.append((identity.pid, sig)) or True,
    )

    assert flux_kohya._wait_for_descendants(child, timeout=1.0) is True
    assert len(scans) >= 4
    assert signals == [(late.pid, signal.SIGKILL)]


def test_an_unconfirmed_child_stays_registered_for_the_fallback_barrier(monkeypatch):
    """False termination retains descriptor ownership and blocks handoff."""
    calls: list[str] = []

    class Zombie:
        pid = 99
        returncode = None

        def poll(self):
            return None

    zombie = Zombie()
    flux_kohya._register_kohya_child(zombie)
    monkeypatch.setattr(
        flux_kohya,
        "_terminate",
        lambda proc, **kwargs: calls.append("terminate") or False,
    )
    monkeypatch.setattr(flux_kohya.telemetry, "event", lambda name, **values: None)

    assert flux_kohya.reap_kohya_children("probe") == 0
    assert calls == ["terminate"]
    assert len(flux_kohya._ACTIVE_KOHYA_CHILDREN) == 1
    assert flux_kohya._ACTIVE_KOHYA_CHILDREN[0].proc is zombie
    with pytest.raises(
        flux_kohya.KohyaContainmentError,
        match="refusing ai-toolkit handoff.*leader_pids=\\[99\\]",
    ):
        flux_kohya._require_kohya_quiescent("fallback_probe")
    assert calls == ["terminate", "terminate"]
    assert flux_kohya._ACTIVE_KOHYA_CHILDREN[0].proc is zombie


def test_failed_reap_blocks_aitoolkit_on_the_same_gpu(monkeypatch):
    child = _spawn_sleeper()
    observations: list = []
    _install_fake_aitoolkit(monkeypatch, _fake_aitoolkit(observations))
    monkeypatch.setattr(
        flux_kohya,
        "resolve_flux_cache_layout",
        lambda cached_model_dir: (flux_kohya._SNAPSHOT_LAYOUT, None),
    )

    def fail_after_launch(spec, deadline):
        flux_kohya._register_kohya_child(child)
        return False

    monkeypatch.setattr(flux_kohya, "_attempt_snapshot_kohya", fail_after_launch)
    monkeypatch.setattr(flux_kohya, "_terminate", lambda *args, **kwargs: False)
    monkeypatch.setattr(flux_kohya.telemetry, "event", lambda *args, **kwargs: None)
    monkeypatch.setattr(flux_kohya.telemetry, "set_meta", lambda **kwargs: None)

    with pytest.raises(
        flux_kohya.KohyaContainmentError,
        match="outer no-artifact fallback required",
    ):
        flux_kohya.run(_spec(), _Deadline())

    assert observations == [], "ai-toolkit started despite unverified Kohya shutdown"
    assert len(flux_kohya._ACTIVE_KOHYA_CHILDREN) == 1
    assert flux_kohya._ACTIVE_KOHYA_CHILDREN[0].proc is child


def test_termination_helper_exception_is_fail_closed(monkeypatch):
    child = _spawn_sleeper()
    observations: list = []
    _install_fake_aitoolkit(monkeypatch, _fake_aitoolkit(observations))
    monkeypatch.setattr(
        flux_kohya,
        "resolve_flux_cache_layout",
        lambda cached_model_dir: (flux_kohya._SNAPSHOT_LAYOUT, None),
    )

    def fail_after_launch(spec, deadline):
        flux_kohya._register_kohya_child(child)
        return False

    def broken_helper(*args, **kwargs):
        raise RuntimeError("injected termination helper failure")

    monkeypatch.setattr(flux_kohya, "_attempt_snapshot_kohya", fail_after_launch)
    monkeypatch.setattr(flux_kohya, "_terminate", broken_helper)
    monkeypatch.setattr(flux_kohya.telemetry, "event", lambda *args, **kwargs: None)
    monkeypatch.setattr(flux_kohya.telemetry, "set_meta", lambda **kwargs: None)

    with pytest.raises(
        flux_kohya.KohyaContainmentError,
        match="termination helper failed: RuntimeError: injected termination helper",
    ):
        flux_kohya.run(_spec(), _Deadline())

    assert observations == []
    assert flux_kohya._ACTIVE_KOHYA_CHILDREN[0].proc is child


def test_containment_error_is_not_downgraded_to_trainer_fallback(monkeypatch):
    monkeypatch.delenv("FORGE_FLUX_SNAPSHOT_BACKEND", raising=False)
    monkeypatch.setattr(
        flux_kohya,
        "resolve_snapshot_kohya_checkpoint",
        lambda cached_model_dir: "/cache/flux1-dev.safetensors",
    )
    monkeypatch.setattr(flux_kohya, "_kohya_runtime_ready", lambda: (True, "ready"))
    monkeypatch.setattr(
        flux_kohya,
        "_train_with_kohya",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            flux_kohya.KohyaContainmentError("injected containment failure")
        ),
    )
    monkeypatch.setattr(flux_kohya.telemetry, "event", lambda *args, **kwargs: None)

    with pytest.raises(
        flux_kohya.KohyaContainmentError,
        match="injected containment failure",
    ):
        flux_kohya._attempt_snapshot_kohya(_spec(), _Deadline())


def test_reap_releases_an_already_reaped_leader_record(monkeypatch):
    child = _spawn_sleeper()
    child.kill()
    child.wait(timeout=10)
    flux_kohya._register_kohya_child(child)
    events: list[str] = []
    monkeypatch.setattr(
        flux_kohya.telemetry, "event", lambda name, **values: events.append(name)
    )

    assert flux_kohya.reap_kohya_children("probe") == 1
    assert events == ["kohya_child_reaped"]
    assert flux_kohya._ACTIVE_KOHYA_CHILDREN == []


def test_no_kohya_child_registry_leak_after_a_clean_run(monkeypatch, tmp_path):
    script_root = tmp_path / "sd-scripts"
    script_root.mkdir()
    (script_root / "flux_train_network.py").write_text("# test", encoding="utf-8")
    spawned: list[subprocess.Popen] = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(flux_kohya, "_SD_SCRIPTS_DIR", str(script_root))
    monkeypatch.setattr(flux_kohya, "_log_path", lambda spec: str(tmp_path / "k.log"))
    monkeypatch.setattr(flux_kohya, "_start_gpu_sampler", lambda *args: None)
    monkeypatch.setattr(flux_kohya, "_POLL_SECONDS", 0.05)
    monkeypatch.setattr(flux_kohya.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(
        flux_kohya,
        "_command",
        lambda config_path, script=None: [sys.executable, "-c", "pass"],
    )
    monkeypatch.setattr(flux_kohya.telemetry, "event", lambda name, **values: None)
    monkeypatch.setattr(flux_kohya.telemetry, "sample", lambda *args: None)
    monkeypatch.setattr(
        flux_kohya.checkpoints, "current_loras", lambda save_root, scope: ["x"]
    )

    flux_kohya._run_kohya(
        str(tmp_path / "config.toml"), _Deadline(), _spec(), {}
    )

    assert spawned and spawned[0].returncode == 0
    assert flux_kohya._ACTIVE_KOHYA_CHILDREN == []


def test_registry_and_barrier_are_importable_module_surface():
    """The barrier must be reachable from the fallback, not buried in a closure."""
    assert callable(flux_kohya.reap_kohya_children)
    assert isinstance(flux_kohya._ACTIVE_KOHYA_CHILDREN, list)
    assert os.path.basename(flux_kohya.__file__) == "flux_kohya.py"


# --------------------------------------------------------------------------
# Escapes found by the independent reviewer (Z6 A1 / A2)
# --------------------------------------------------------------------------


def _spawn_setsid_grandchild(tmp_path):
    """parent -> grandchild that leaves the process group via setsid().

    Returns (parent Popen, grandchild pid).  This is the shape `os.killpg`
    cannot reach: the grandchild is in its own session, so the group kill
    misses it entirely.
    """
    marker = tmp_path / "grandchild.pid"
    script = (
        "import os, subprocess, sys, time\n"
        "child = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(300)'],\n"
        "    start_new_session=True,\n"
        ")\n"
        f"open({str(marker)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(300)\n"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if marker.exists() and marker.read_text().strip():
            break
        time.sleep(0.05)
    else:  # pragma: no cover - environment failure, not a product defect
        parent.kill()
        parent.wait(timeout=10)
        pytest.fail("the test's own grandchild never started")
    return parent, int(marker.read_text().strip())


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_a_setsid_grandchild_does_not_survive_the_barrier(monkeypatch, tmp_path):
    """Reviewer Z6-A1: os.killpg alone leaves a re-sessioned grandchild alive."""
    parent, grandchild_pid = _spawn_setsid_grandchild(tmp_path)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        flux_kohya.telemetry,
        "event",
        lambda name, **values: events.append((name, values)),
    )
    try:
        flux_kohya._register_kohya_child(parent)
        reaped = flux_kohya.reap_kohya_children("grandchild_probe")

        assert parent.poll() is not None, "the direct child was not reaped"
        # _sweep_descendants already waited for the kill to land.
        assert not _alive(grandchild_pid), (
            "a setsid grandchild outlived the barrier and still holds the GPU"
        )
        assert reaped == 1
    finally:
        for pid in (grandchild_pid,):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=10)


def test_descendant_discovery_finds_a_re_sessioned_grandchild(tmp_path):
    """The snapshot itself must see through setsid, or the sweep is empty."""
    parent, grandchild_pid = _spawn_setsid_grandchild(tmp_path)
    try:
        table = flux_kohya._process_table()
        assert grandchild_pid in flux_kohya._descendant_identities(parent.pid, table)
    finally:
        for pid in (grandchild_pid,):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        parent.kill()
        parent.wait(timeout=10)


def test_descendant_sweep_is_opt_in_only(monkeypatch):
    """A pid we did not spawn must never make us signal unrelated processes."""
    child = _spawn_sleeper()
    monkeypatch.setattr(
        flux_kohya,
        "_signal_owned_descendants",
        lambda record, sig: pytest.fail("descendant signalling ran without opt-in"),
    )
    monkeypatch.setattr(
        flux_kohya,
        "_wait_for_descendants",
        lambda record: pytest.fail("descendant verification ran without opt-in"),
    )

    assert flux_kohya._terminate(child) is True
    assert child.returncode is not None


def test_the_success_path_also_reaps_an_unconfirmed_child(monkeypatch, tmp_path):
    """Reviewer Z6-A2: run() used to return before the barrier on success."""
    child = _spawn_sleeper()
    events: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        flux_kohya,
        "resolve_flux_cache_layout",
        lambda cached_model_dir: (flux_kohya._SNAPSHOT_LAYOUT, None),
    )

    def succeed_but_leak(spec, deadline):
        # _train_with_kohya returned True (finalize found a rung) while
        # _terminate could not confirm the child's death.
        flux_kohya._register_kohya_child(child)
        return True

    monkeypatch.setattr(flux_kohya, "_attempt_snapshot_kohya", succeed_but_leak)
    monkeypatch.setattr(flux_kohya.telemetry, "set_meta", lambda **values: None)
    monkeypatch.setattr(
        flux_kohya.telemetry,
        "event",
        lambda name, **values: events.append((name, values)),
    )

    flux_kohya.run(_spec(), _Deadline())

    assert child.poll() is not None, (
        "a kohya child outlived a SUCCESSFUL flux run and kept holding VRAM"
    )
    assert child.returncode is not None
    assert flux_kohya._ACTIVE_KOHYA_CHILDREN == []
    assert [
        values for name, values in events if name == "kohya_child_reaped"
    ] == [{"reason": "flux_run_exit", "reaped": 1, "survived": 0}]


def test_the_standalone_path_also_ends_with_the_barrier(monkeypatch, tmp_path):
    child = _spawn_sleeper()
    monkeypatch.setattr(
        flux_kohya,
        "resolve_flux_cache_layout",
        lambda cached_model_dir: (flux_kohya._STANDALONE_LAYOUT, "/cache/flux.safetensors"),
    )

    def leaky_standalone(spec, deadline, base_model):
        flux_kohya._register_kohya_child(child)

    monkeypatch.setattr(flux_kohya, "_run_standalone_kohya", leaky_standalone)
    monkeypatch.setattr(flux_kohya.telemetry, "set_meta", lambda **values: None)
    monkeypatch.setattr(flux_kohya.telemetry, "event", lambda name, **values: None)

    flux_kohya.run(_spec(), _Deadline())

    assert child.poll() is not None
    assert flux_kohya._ACTIVE_KOHYA_CHILDREN == []
