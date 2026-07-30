"""Fail-closed contracts for the additive Krea host-layout bootstrap."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from ops.calibration import krea_host_bootstrap as bootstrap
from ops.calibration import krea_stage1_runtime
from ops.calibration import run_krea_ladder as runner
from ops.calibration import krea_provenance


def _payload(tmp_path: Path) -> dict:
    evidence = tmp_path / "evidence-volume"
    return {
        "schema": 1,
        "kind": "forge-krea-host-bootstrap-spec",
        "sources": {
            "forge_repo": str(tmp_path / "stage/forge"),
            "ai_toolkit_repo": str(tmp_path / "stage/ai-toolkit"),
            "venv": str(tmp_path / "stage/venv"),
            "checkpoints": str(tmp_path / "volatile/checkpoints"),
            "dataset": str(tmp_path / "volatile/dataset"),
            "cache": str(tmp_path / "volatile/cache"),
            "campaign": str(evidence / "campaign"),
            "evidence_root": str(evidence),
        },
        "source_identities": {
            "forge_commit": "a" * 40,
            "ai_toolkit_commit": "b" * 40,
        },
        "requirements": {
            "ubuntu_release": "22.04",
            "minimum_effective_cpu_capacity": 16,
            "minimum_effective_memory_bytes": 64 * 1024**3,
            "minimum_checkpoint_filesystem_bytes": 500 * 1024**3,
            "minimum_checkpoint_free_bytes": 350 * 1024**3,
            "minimum_evidence_filesystem_bytes": 200 * 1024**3,
            "minimum_evidence_free_bytes": 100 * 1024**3,
            "minimum_gpu_memory_mib": 78_000,
            "maximum_gpu_memory_mib": 85_000,
            "minimum_cuda_version": "12.8",
            "required_docker_runtime": "nvidia",
            "systemd_pid1_required": True,
            "unified_cgroup_v2_required": True,
            "rootful_docker_required": True,
            "separate_evidence_filesystem_required": True,
        },
        "runtime": {
            "container_image_reference": "sha256:" + "c" * 64,
            "container_image_sha256": "c" * 64,
            "execution_surface": "staged_host_venv",
            "ai_toolkit_dir": "/app/ai-toolkit",
            "jit_enabled": True,
            "stage1_runtime_receipt": {
                "path": str(evidence / "campaign/controls/stage1-runtime.json"),
                "file_sha256": "d" * 64,
                "receipt_sha256": "e" * 64,
            },
            "runtime_cache_policy": dict(bootstrap._RUNTIME_CACHE_POLICY),
        },
        "gpu_execution_authorized": False,
    }


def _write(path: Path, value: dict) -> Path:
    path.write_bytes(krea_provenance.canonical_bytes(value) + b"\n")
    return path


def test_layout_spec_is_fixed_self_bound_and_never_authorizes_gpu(tmp_path):
    result = bootstrap.seal_spec(_payload(tmp_path))

    assert bootstrap.validate_spec(result) == result
    assert result["gpu_execution_authorized"] is False
    assert set(bootstrap._FIXED_TARGETS.values()) == {
        "/app/forge",
        "/app/ai-toolkit",
        "/app/venv",
        "/app/checkpoints",
        "/dataset",
        "/cache",
        "/campaign",
    }
    assert bootstrap._READ_ONLY_BINDINGS == frozenset(
        {"forge_repo", "ai_toolkit_repo", "venv"}
    )
    assert bootstrap._CALIBRATION_ARTIFACTS["timing_tool"] == (
        "ops/calibration/krea_timing_probe.py"
    )
    assert bootstrap._CALIBRATION_ARTIFACTS["runner"] == (
        "ops/calibration/run_krea_ladder.py"
    )


def test_layout_spec_cli_is_canonical_mode_0600_and_create_only(tmp_path):
    payload_path = _write(tmp_path / "layout.payload.json", _payload(tmp_path))
    output = tmp_path / "layout.spec.json"

    assert (
        bootstrap.main(
            [
                "seal-layout-spec",
                "--payload",
                str(payload_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    document = json.loads(output.read_bytes())
    assert output.read_bytes() == krea_provenance.canonical_bytes(document) + b"\n"
    assert os.stat(output).st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        bootstrap.main(
            [
                "seal-layout-spec",
                "--payload",
                str(payload_path),
                "--output",
                str(output),
            ]
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("minimum_effective_cpu_capacity", 15, "at least 16"),
        ("minimum_effective_memory_bytes", 63 * 1024**3, "at least 64"),
        ("minimum_checkpoint_filesystem_bytes", 499 * 1024**3, "at least 500"),
        ("minimum_gpu_memory_mib", 77_999, "literal H100"),
        ("maximum_gpu_memory_mib", 85_001, "literal H100"),
    ],
)
def test_layout_spec_cannot_weaken_minimum_host_contract(tmp_path, field, value, match):
    payload = _payload(tmp_path)
    payload["requirements"][field] = value

    with pytest.raises(ValueError, match=match):
        bootstrap.seal_spec(payload)


def _stage_sources(tmp_path: Path, spec: dict) -> None:
    for source in spec["sources"].values():
        Path(source).mkdir(parents=True, exist_ok=True)
    forge = Path(spec["sources"]["forge_repo"])
    for relative in bootstrap._CALIBRATION_ARTIFACTS.values():
        path = forge / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"artifact: {relative}\n", encoding="utf-8")
    python = Path(spec["sources"]["venv"]) / "bin/python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)


def test_source_preflight_rejects_evidence_on_volatile_filesystem(
    tmp_path, monkeypatch
):
    spec = bootstrap.seal_spec(_payload(tmp_path))
    _stage_sources(tmp_path, spec)
    monkeypatch.setattr(
        bootstrap,
        "_git_identity",
        lambda _path, expected, _label: {"commit": expected},
    )
    monkeypatch.setattr(
        bootstrap,
        "_filesystem",
        lambda path, require_mountpoint: {
            "source": "/dev/test",
            "target": str(path),
            "filesystem_type": "ext4",
            "mount_options": ["rw"],
            "device_major_minor": "1:1",
            "device_id": path.stat().st_dev,
        },
    )
    monkeypatch.setattr(
        bootstrap,
        "_filesystem_capacity",
        lambda _path: (800 * 1024**3, 700 * 1024**3),
    )

    with pytest.raises(RuntimeError, match="not distinct"):
        bootstrap._source_identity(spec)


def test_materialized_physical_sources_bind_into_bootstrap_spec_without_cycle(
    tmp_path, monkeypatch
):
    payload = _payload(tmp_path)
    forge = Path(payload["sources"]["forge_repo"])
    toolkit = Path(payload["sources"]["ai_toolkit_repo"])
    venv = Path(payload["sources"]["venv"])
    for path in (forge, toolkit, venv / "bin"):
        path.mkdir(parents=True, exist_ok=True)
    python = venv / "bin/python"
    python.write_bytes(b"python")
    python.chmod(0o755)
    tree_entries = [
        {"mode": 0o755, "path": "bin", "type": "directory"},
        {
            "bytes": 6,
            "mode": 0o755,
            "path": "bin/python",
            "sha256": hashlib.sha256(b"python").hexdigest(),
            "type": "file",
        },
    ]
    tree = {
        "entries": tree_entries,
        "entry_count": len(tree_entries),
        "entries_sha256": krea_provenance.canonical_sha256(tree_entries),
        "root": str(venv),
    }
    receipt_path = Path(payload["runtime"]["stage1_runtime_receipt"]["path"])
    receipt_path.parent.mkdir(parents=True)
    receipt = {
        "paths": {
            "forge_repo": str(forge),
            "ai_toolkit_repo": str(toolkit),
            "destination": str(venv),
            "receipt": str(receipt_path),
        },
        "forge": {"commit": "a" * 40, "tree": "1" * 40},
        "ai_toolkit": {"commit": "b" * 40, "tree": "2" * 40},
        "inputs": {"materializer": {"sha256": "f" * 64}},
        "tree_manifest": tree,
        "verification": {"runtime_verifier_pass": True},
        "receipt_sha256": "e" * 64,
    }
    receipt_path.write_bytes(krea_provenance.canonical_bytes(receipt) + b"\n")
    payload["runtime"]["stage1_runtime_receipt"] = {
        "path": str(receipt_path),
        "file_sha256": krea_provenance.file_sha256(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
    }
    spec = bootstrap.seal_spec(payload)
    monkeypatch.setattr(krea_stage1_runtime, "load_receipt", lambda _path: receipt)
    monkeypatch.setattr(
        krea_stage1_runtime,
        "validate_receipt",
        lambda value, recapture: value,
    )

    result = bootstrap._stage1_runtime_identity(
        spec,
        forge_identity={"commit": "a" * 40, "tree": "1" * 40},
        ai_toolkit_identity={"commit": "b" * 40, "tree": "2" * 40},
        venv_tree=bootstrap._tree_identity(venv, "test venv"),
        materializer_sha256="f" * 64,
    )

    assert result["receipt_sha256"] == "e" * 64
    assert receipt["paths"]["ai_toolkit_repo"] != "/app/ai-toolkit"


def test_gpu_probe_rejects_foreign_compute_process(monkeypatch):
    monkeypatch.setattr(
        bootstrap,
        "_trusted_executable",
        lambda _name: ("/usr/bin/nvidia-smi", {}),
    )

    def run(command, cwd=None):
        del cwd
        if len(command) > 1 and "--query-gpu=" in command[1]:
            return "GPU-1, NVIDIA H100 PCIe, 570.1, Disabled, 81559"
        if command == ["/usr/bin/nvidia-smi"]:
            return "NVIDIA-SMI 570.1 CUDA Version: 12.8"
        raise AssertionError(command)

    monkeypatch.setattr(bootstrap, "_run", run)
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="GPU-1, 1234\n", stderr=""
        ),
    )

    with pytest.raises(RuntimeError, match="foreign compute"):
        bootstrap._gpu_identity()


def test_receipt_reverification_recaptures_and_detects_drift(tmp_path, monkeypatch):
    spec = bootstrap.seal_spec(_payload(tmp_path))
    original = {"host": {"gpu": "GPU-1"}, "sources": {}, "bindings": {}}
    monkeypatch.setattr(bootstrap, "_preflight", lambda *_args, **_kwargs: original)
    receipt = bootstrap.build_receipt(spec)

    assert bootstrap.validate_receipt(receipt, recapture=True) == receipt
    monkeypatch.setattr(
        bootstrap,
        "_preflight",
        lambda *_args, **_kwargs: {
            "host": {"gpu": "GPU-2"},
            "sources": {},
            "bindings": {},
        },
    )
    with pytest.raises(RuntimeError, match="drifted"):
        bootstrap.validate_receipt(receipt, recapture=True)


def test_layout_sources_may_not_overlap_fixed_targets(tmp_path):
    payload = _payload(tmp_path)
    payload["sources"]["forge_repo"] = "/app/forge/source"

    with pytest.raises(ValueError, match="may not overlap"):
        bootstrap.seal_spec(payload)


def test_venv_python_allows_only_an_internal_leaf_symlink(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    internal = venv / "bin/python3"
    internal.write_text("#!/bin/sh\n", encoding="utf-8")
    internal.chmod(0o755)
    (venv / "bin/python").symlink_to("python3")

    candidate, resolved = bootstrap._safe_venv_python(venv, "venv Python")

    assert candidate.is_symlink()
    assert resolved == internal


def test_venv_python_rejects_symlink_escape(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    outside = tmp_path / "outside-python"
    outside.write_text("#!/bin/sh\n", encoding="utf-8")
    (venv / "bin/python").symlink_to(outside)

    with pytest.raises(RuntimeError, match="outside the staged venv"):
        bootstrap._safe_venv_python(venv, "venv Python")


def test_trusted_system_binary_does_not_consult_operator_path(tmp_path, monkeypatch):
    fake = tmp_path / "git"
    fake.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    requested, identity = bootstrap._trusted_executable("git")

    assert requested == "/usr/bin/git"
    assert identity["requested_path"] == "/usr/bin/git"
    assert identity["resolved_path"] != str(fake)


def test_preimport_venv_drift_rejects_before_any_child_exec(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    forge = tmp_path / "forge"
    toolkit = tmp_path / "toolkit"
    venv = tmp_path / "venv"
    for directory in (forge, toolkit, venv / "bin", venv / "site-packages"):
        directory.mkdir(parents=True, exist_ok=True)
    python = venv / "bin/python"
    python.write_text("python", encoding="utf-8")
    python.chmod(0o755)
    expected_tree = runner._preimport_tree_identity(venv)
    initializer = venv / "site-packages/malicious.pth"
    sentinel = tmp_path / "initializer-ran"
    initializer.write_text(
        f"import pathlib; pathlib.Path({str(sentinel)!r}).write_text('ran')\n",
        encoding="utf-8",
    )
    source_identity = {
        "forge_repo": {"identity": "forge"},
        "ai_toolkit_repo": {"identity": "toolkit"},
        "venv_tree": expected_tree,
        "venv_python": {"resolved_sha256": "0" * 64},
    }
    spec_body = {
        "sources": {
            "forge_repo": str(forge),
            "ai_toolkit_repo": str(toolkit),
            "venv": str(venv),
        },
        "source_identities": {"forge_commit": "a" * 40, "ai_toolkit_commit": "b" * 40},
        "runtime": {
            "runtime_cache_policy": dict(bootstrap._RUNTIME_CACHE_POLICY),
        },
    }
    spec = {**spec_body, "spec_sha256": runner._canonical_hash(spec_body)}
    receipt_body = {
        "spec": spec,
        "layout_identity": {
            "sources": source_identity,
            "bindings": {
                "runtime_cache": {"policy": dict(bootstrap._RUNTIME_CACHE_POLICY)}
            },
            "host": {
                "trusted_executables": {"git": {"requested_path": "/usr/bin/git"}}
            },
        },
    }
    receipt = {**receipt_body, "receipt_sha256": runner._canonical_hash(receipt_body)}

    def write_json(path, value):
        raw = runner._canonical_bytes(value) + b"\n"
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    receipt_path = tmp_path / "receipt.json"
    receipt_file_sha = write_json(receipt_path, receipt)
    host = {
        "bootstrap_receipt": {
            "path": str(receipt_path),
            "file_sha256": receipt_file_sha,
        }
    }
    host_path = tmp_path / "host.json"
    host_sha = write_json(host_path, host)
    plan_path = tmp_path / "plan.json"
    write_json(
        plan_path,
        {"host_execution_manifest": {"path": str(host_path), "sha256": host_sha}},
    )
    monkeypatch.setattr(runner.os.path, "samefile", lambda *_args: True)
    monkeypatch.setattr(
        runner,
        "_preimport_executable_identity",
        lambda _path: {"requested_path": "/usr/bin/git"},
    )
    monkeypatch.setattr(
        runner,
        "_preimport_git_identity",
        lambda path, _commit: source_identity[
            "forge_repo" if path == forge else "ai_toolkit_repo"
        ],
    )
    child_started = False

    def child(*_args, **_kwargs):
        nonlocal child_started
        child_started = True
        raise AssertionError("child execution occurred before venv verification")

    monkeypatch.setattr(runner.subprocess, "run", child)
    args = type("Args", (), {"timing_probe_plan": plan_path, "execution_plan": None})()
    with pytest.raises(RuntimeError, match="pre-import staged venv identity drifted"):
        runner._trusted_stage1_reexec(args)
    assert child_started is False
    assert not sentinel.exists()


def test_runner_rejects_operator_ai_toolkit_override_before_control_reads(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AI_TOOLKIT_DIR", "/tmp/operator-toolkit")
    args = type(
        "Args",
        (),
        {"timing_probe_plan": tmp_path / "absent.json", "execution_plan": None},
    )()

    with pytest.raises(RuntimeError, match="AI_TOOLKIT_DIR"):
        runner._trusted_stage1_reexec(args)


def test_runner_rejects_operator_template_override_before_control_reads(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FORGE_TEMPLATES_DIR", "/tmp/operator-templates")
    args = type(
        "Args",
        (),
        {"timing_probe_plan": tmp_path / "absent.json", "execution_plan": None},
    )()

    with pytest.raises(RuntimeError, match="FORGE_TEMPLATES_DIR"):
        runner._trusted_stage1_reexec(args)


def test_runner_rejects_operator_compiler_cache_override_before_control_reads(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/tmp/operator-cache")
    args = type(
        "Args",
        (),
        {"timing_probe_plan": tmp_path / "absent.json", "execution_plan": None},
    )()

    with pytest.raises(RuntimeError, match="TORCHINDUCTOR_CACHE_DIR"):
        runner._trusted_stage1_reexec(args)


def test_runner_runtime_cache_is_clean_plan_scoped_and_never_reused(
    tmp_path, monkeypatch
):
    root = tmp_path / "runtime-cache"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(runner, "_RUNTIME_CACHE_ROOT", root)
    plan_sha = "a" * 64

    environments = []
    for capture_id in ("timing-a", "timing-b", "timing-c", "heldout-e2e"):
        environments.append(
            runner._runtime_cache_environment(
                plan_sha, timing_capture_id=capture_id, create=True
            )
        )

    assert (
        len({item[runner._RUNTIME_CACHE_NAMESPACE_ENV] for item in environments}) == 4
    )
    assert all(
        Path(path).is_relative_to(root)
        for environment in environments
        for path in environment.values()
    )
    with pytest.raises(RuntimeError, match="reuse is forbidden"):
        runner._runtime_cache_environment(
            plan_sha, timing_capture_id="timing-a", create=True
        )


def test_docker_identity_binds_actual_image_id(monkeypatch):
    monkeypatch.setattr(
        bootstrap,
        "_trusted_executable",
        lambda name: (f"/usr/bin/{name.replace('_', '-')}", {}),
    )

    monkeypatch.setattr(
        bootstrap.Path, "stat", lambda _self: type("S", (), {"st_mode": 0o140000})()
    )

    observed_timeouts = []

    def run(command, cwd=None, environment=None, timeout_seconds=30):
        del cwd, environment
        observed_timeouts.append((tuple(command), timeout_seconds))
        if command[:3] == ["/usr/bin/docker", "image", "inspect"]:
            return json.dumps(
                {
                    "Id": "sha256:" + "c" * 64,
                    "RepoDigests": ["registry.invalid/sn56@sha256:" + "d" * 64],
                    "Config": {"Env": []},
                }
            )
        if command[:2] == ["/usr/bin/docker", "run"]:
            if "--entrypoint" in command:
                return json.dumps(
                    {
                        "cuda": True,
                        "result": "PASS",
                        "torch": "2.6.0",
                        "torch_cuda": "12.4",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            return "GPU-1"
        if (
            command[:2] == ["/usr/bin/docker", "info"]
            and "SecurityOptions" in command[-1]
        ):
            return json.dumps(["name=seccomp"])
        if command[:2] == ["/usr/bin/docker", "info"] and "Runtimes" in command[-1]:
            return json.dumps({"nvidia": {}, "runc": {}})
        if command[:2] == ["/usr/bin/docker", "version"]:
            return "27.0"
        if command[:2] == ["/usr/bin/docker", "info"]:
            return "/var/lib/docker"
        if command[0] == "/usr/bin/nvidia-container-cli":
            return "version 1.17"
        raise AssertionError(command)

    monkeypatch.setattr(bootstrap, "_run", run)
    identity = bootstrap._docker_identity(
        "nvidia",
        image_reference="sha256:" + "c" * 64,
        expected_image_sha256="c" * 64,
        expected_jit_enabled=True,
    )
    assert identity["container_image"]["image_id"] == "sha256:" + "c" * 64
    assert identity["container_image"]["cuda_jit_smoke"]["result"] == "PASS"
    assert any(
        "--entrypoint" in command and timeout == 300
        for command, timeout in observed_timeouts
    )

    with pytest.raises(RuntimeError, match="actual Docker image ID differs"):
        bootstrap._docker_identity(
            "nvidia",
            image_reference="sha256:" + "c" * 64,
            expected_image_sha256="e" * 64,
            expected_jit_enabled=True,
        )


def test_docker_identity_rejects_failed_real_cuda_jit_compile(monkeypatch):
    monkeypatch.setattr(
        bootstrap,
        "_trusted_executable",
        lambda name: (f"/usr/bin/{name.replace('_', '-')}", {}),
    )
    monkeypatch.setattr(
        bootstrap.Path, "stat", lambda _self: type("S", (), {"st_mode": 0o140000})()
    )

    def run(command, cwd=None, environment=None, timeout_seconds=30):
        del cwd, environment, timeout_seconds
        if command[:3] == ["/usr/bin/docker", "image", "inspect"]:
            return json.dumps(
                {"Id": "sha256:" + "c" * 64, "RepoDigests": [], "Config": {"Env": []}}
            )
        if command[:2] == ["/usr/bin/docker", "run"]:
            if "--entrypoint" in command:
                return json.dumps(
                    {
                        "cuda": True,
                        "result": "COMPILE_FAILED",
                        "torch": "2.6.0",
                        "torch_cuda": "12.4",
                    }
                )
            return "GPU-1"
        if (
            command[:2] == ["/usr/bin/docker", "info"]
            and "SecurityOptions" in command[-1]
        ):
            return json.dumps(["name=seccomp"])
        if command[:2] == ["/usr/bin/docker", "info"] and "Runtimes" in command[-1]:
            return json.dumps({"nvidia": {}})
        if command[:2] == ["/usr/bin/docker", "version"]:
            return "27.0"
        if command[:2] == ["/usr/bin/docker", "info"]:
            return "/var/lib/docker"
        if command[0] == "/usr/bin/nvidia-container-cli":
            return "version 1.17"
        raise AssertionError(command)

    monkeypatch.setattr(bootstrap, "_run", run)
    with pytest.raises(RuntimeError, match="compile smoke did not pass"):
        bootstrap._docker_identity(
            "nvidia",
            image_reference="sha256:" + "c" * 64,
            expected_image_sha256="c" * 64,
            expected_jit_enabled=True,
        )


def test_rollback_rechecks_every_new_mount_and_surfaces_failures(tmp_path, monkeypatch):
    targets = [tmp_path / "one", tmp_path / "two"]
    monkeypatch.setattr(
        bootstrap, "_trusted_executable", lambda _name: ("/usr/bin/umount", {})
    )
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 32, stdout="", stderr="busy"
        ),
    )
    monkeypatch.setattr(bootstrap, "_is_mountpoint", lambda _path: True)

    with pytest.raises(RuntimeError, match="rollback incomplete.*Manually unmount"):
        bootstrap._rollback_bindings(targets)


def test_rollback_accepts_only_verified_unmounted_targets(tmp_path, monkeypatch):
    targets = [tmp_path / "one", tmp_path / "two"]
    seen = []
    monkeypatch.setattr(
        bootstrap, "_trusted_executable", lambda _name: ("/usr/bin/umount", {})
    )
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="", stderr=""
        ),
    )
    monkeypatch.setattr(
        bootstrap,
        "_is_mountpoint",
        lambda path: seen.append(path) or False,
    )

    bootstrap._rollback_bindings(targets)
    assert seen == list(reversed(targets))
