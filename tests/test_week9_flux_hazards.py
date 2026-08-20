"""Week-9 HAZARD 2: no Kohya child may outlive the FLUX fallback.

``flux_kohya.run`` falls back to ``aitoolkit.run`` on any Kohya failure.  Both
trainers use the whole GPU, so a Kohya child that survives the transition means
VRAM exhaustion or a corrupted artifact — a forfeit for the task either way.

week9-rc only terminated the child on the clean deadline path: an exception
anywhere in the poll loop, or a ``_terminate`` whose final ``wait`` timed out,
propagated straight past cleanup and into the fallback with the trainer still
running.  These tests spawn REAL child processes and fail on the week9-rc code.
"""

from __future__ import annotations

import os
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
    """No test may leak a registered child into the next one.

    getattr, not a direct call: the behavioural tests below must be runnable
    against the week9-rc module (which has no registry) and fail there on their
    own assertions rather than on a missing attribute.
    """
    reap = getattr(flux_kohya, "reap_kohya_children", None)
    if reap is not None:
        reap("test_setup")
    yield
    if reap is not None:
        reap("test_teardown")


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


def test_surviving_kohya_child_is_reaped_before_the_aitoolkit_fallback(
    monkeypatch, tmp_path
):
    """The decisive test: a crash mid-Kohya must not hand ai-toolkit a live GPU.

    Fully end-to-end over surface that exists on week9-rc too — real Popen, the
    real ``_run_kohya`` poll loop, the real ``_attempt_snapshot_kohya`` failure
    handler, the real ``run`` fallback — so it fails on the RC bytes by finding
    the trainer still alive, not by finding a missing helper.
    """
    script_root = tmp_path / "sd-scripts"
    script_root.mkdir()
    (script_root / "flux_train_network.py").write_text("# test", encoding="utf-8")
    spawned: list[subprocess.Popen] = []
    events: list[tuple[str, dict]] = []
    observations: list = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        spawned.append(proc)
        return proc

    fake = _fake_aitoolkit(observations)
    fake.run.watched = spawned

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
            "import time; time.sleep(300)",
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

    flux_kohya.run(_spec(), _Deadline(raise_after=1))

    assert spawned, "the Kohya route never started a child"
    assert observations, "the fallback never reached ai-toolkit"
    assert observations[0][0][1] is not None, (
        "a Kohya child was still training when ai-toolkit started on the same GPU"
    )
    # Reaped, not merely signalled: no zombie holding the device handle.
    assert spawned[0].returncode is not None
    assert [name for name, _ in events if name == "flux_snapshot_kohya_fallback"]


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


def test_terminate_reaps_a_real_child_and_reports_success():
    child = _spawn_sleeper()
    started = time.monotonic()
    assert flux_kohya._terminate(child) is True
    assert child.returncode is not None
    assert time.monotonic() - started < 10


def test_terminate_never_raises_when_the_child_cannot_be_confirmed_dead(monkeypatch):
    """week9-rc let this TimeoutExpired escape and skip every cleanup path."""

    class Unkillable:
        pid = 4242

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("kohya", timeout)

        def send_signal(self, sig):
            pass

    monkeypatch.setattr(flux_kohya.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(flux_kohya.os, "killpg", lambda pgid, sig: None)

    assert flux_kohya._terminate(Unkillable()) is False


def test_an_unconfirmed_child_stays_registered_for_the_fallback_barrier(monkeypatch):
    """A child we could not kill must not be silently forgotten."""
    calls: list[str] = []

    class Zombie:
        pid = 99
        returncode = None

        def poll(self):
            return None

    zombie = Zombie()
    flux_kohya._register_kohya_child(zombie)
    monkeypatch.setattr(
        flux_kohya, "_terminate", lambda proc: calls.append("terminate") or False
    )
    monkeypatch.setattr(flux_kohya.telemetry, "event", lambda name, **values: None)

    assert flux_kohya.reap_kohya_children("probe") == 0
    assert calls == ["terminate"]
    # The registry is drained by the barrier itself, so a second barrier is a
    # no-op rather than an unbounded retry loop.
    assert flux_kohya._ACTIVE_KOHYA_CHILDREN == []


def test_reap_is_a_no_op_for_children_that_already_exited(monkeypatch):
    child = _spawn_sleeper()
    child.kill()
    child.wait(timeout=10)
    flux_kohya._register_kohya_child(child)
    events: list[str] = []
    monkeypatch.setattr(
        flux_kohya.telemetry, "event", lambda name, **values: events.append(name)
    )

    assert flux_kohya.reap_kohya_children("probe") == 0
    assert events == []


def test_no_kohya_child_registry_leak_after_a_clean_run(monkeypatch, tmp_path):
    script_root = tmp_path / "sd-scripts"
    script_root.mkdir()
    (script_root / "flux_train_network.py").write_text("# test", encoding="utf-8")
    monkeypatch.setattr(flux_kohya, "_SD_SCRIPTS_DIR", str(script_root))
    monkeypatch.setattr(flux_kohya, "_log_path", lambda spec: str(tmp_path / "k.log"))
    monkeypatch.setattr(flux_kohya, "_start_gpu_sampler", lambda *args: None)
    monkeypatch.setattr(flux_kohya, "_POLL_SECONDS", 0.05)
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

    assert flux_kohya._ACTIVE_KOHYA_CHILDREN == []


def test_registry_and_barrier_are_importable_module_surface():
    """The barrier must be reachable from the fallback, not buried in a closure."""
    assert callable(flux_kohya.reap_kohya_children)
    assert isinstance(flux_kohya._ACTIVE_KOHYA_CHILDREN, list)
    assert os.path.basename(flux_kohya.__file__) == "flux_kohya.py"
