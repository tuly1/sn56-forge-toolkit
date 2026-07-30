"""Contracts for the create-only Stage-1 host runtime materializer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ops.calibration import krea_host_bootstrap as bootstrap
from ops.calibration import krea_provenance
from ops.calibration import run_krea_ladder as ladder
from ops.calibration import krea_stage1_runtime as runtime


def _copy_sources(tmp_path: Path, monkeypatch) -> tuple[runtime.RuntimePaths, Path]:
    source_root = Path(runtime.__file__).resolve().parents[2]
    forge = tmp_path / "forge"
    ai_toolkit = tmp_path / "ai-toolkit"
    destination = tmp_path / "runtime-venv"
    receipt = tmp_path / "evidence/campaign/controls/stage1-receipt.json"
    receipt.parent.mkdir(parents=True)
    for relative in (
        "ops/docker/standalone-image-toolkit-trainer.dockerfile",
        "ops/docker/image-runtime-lock.txt",
        "ops/docker/image-runtime-phase1-constraints.txt",
        "ops/docker/verify_image_runtime.py",
        "ops/calibration/krea_stage1_runtime.py",
    ):
        target = forge / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative, target)
    ai_toolkit.mkdir()
    (ai_toolkit / "requirements.txt").write_text("example==1\n", encoding="utf-8")
    monkeypatch.setattr(
        runtime,
        "AI_TOOLKIT_REQUIREMENTS_SHA256",
        runtime.file_sha256(ai_toolkit / "requirements.txt"),
    )
    system_python = tmp_path / "system-python3"
    system_python.write_bytes(b"fake python executable\n")
    system_python.chmod(0o755)
    fake_git = tmp_path / "git"
    fake_git.write_bytes(b"fake git executable\n")
    fake_git.chmod(0o755)
    monkeypatch.setattr(runtime, "SYSTEM_PYTHON", system_python)
    monkeypatch.setattr(runtime, "SYSTEM_GIT", fake_git)
    monkeypatch.setattr(runtime, "NVIDIA_SMI", tmp_path / "absent-nvidia-smi")
    transient_parent = tmp_path / "transient-cache"
    transient_parent.mkdir()
    monkeypatch.setattr(runtime, "TRANSIENT_CACHE_PARENT", transient_parent)
    monkeypatch.setattr(
        runtime,
        "_read_os_release",
        lambda: {
            "file": {
                "bytes": 1,
                "path": "/etc/os-release",
                "sha256": "a" * 64,
            },
            "id": "ubuntu",
            "version_id": "22.04",
        },
    )
    return (
        runtime.RuntimePaths(forge, ai_toolkit, destination, receipt),
        system_python,
    )


def _fake_runner(paths: runtime.RuntimePaths, system_python: Path, *, dirty=False):
    identity = {
        "cache_tag": "cpython-310",
        "executable": str(system_python),
        "implementation": "cpython",
        "real_executable": str(system_python),
        "soabi": "cpython-310-x86_64-linux-gnu",
        "version": [3, 10, 12],
    }

    def run(command, **kwargs):
        argv = [str(item) for item in command]
        cwd = kwargs.get("cwd")
        stdout = ""
        if argv[:3] == [str(system_python), "-I", "-c"]:
            stdout = json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n"
        elif "rev-parse" in argv and argv[-1] == "HEAD":
            stdout = (
                runtime.AI_TOOLKIT_COMMIT
                if cwd == str(paths.ai_toolkit_repo)
                else "c" * 40
            ) + "\n"
        elif "rev-parse" in argv and argv[-1] == "HEAD^{tree}":
            stdout = "b" * 40 + "\n"
        elif "status" in argv:
            stdout = "?? dirty\n" if dirty else ""
        elif "ls-files" in argv:
            relative = (
                "requirements.txt"
                if cwd == str(paths.ai_toolkit_repo)
                else "ops/calibration/krea_stage1_runtime.py"
            )
            contents = (Path(cwd) / relative).read_bytes()
            blob = (
                __import__("hashlib")
                .sha1(f"blob {len(contents)}\0".encode("ascii") + contents)
                .hexdigest()
            )
            stdout = (
                f"100644 {blob} 0\t{relative}\0"
                if "--stage" in argv
                else f"H {relative}\0"
            )
        elif argv[1:4] == ["-m", "venv", "--copies"]:
            python = paths.destination / "bin/python"
            python.parent.mkdir(parents=True)
            shutil.copyfile(system_python, python)
            python.chmod(0o755)
        elif "verify_image_runtime.py" in " ".join(argv):
            summary = {
                "allowed_pip_check_conflicts": [runtime.EXPECTED_PIP_CHECK_LINE],
                "result": "PASS",
            }
            stdout = json.dumps(summary) + "\nSN56_IMAGE_RUNTIME_INVENTORY=PASS\n"
        elif "probe-essential" in argv:
            stdout = json.dumps({"result": "PASS"}) + "\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    return run


def test_plan_reproduces_exact_production_phase_order_and_indexes(
    tmp_path, monkeypatch
):
    paths, _python = _copy_sources(tmp_path, monkeypatch)
    specs = runtime.command_plan(paths)
    by_id = {item.command_id: item for item in specs}

    assert [item.command_id for item in specs if item.phase == "materialize"] == [
        "venv-create",
        "phase1-ai-toolkit-requirements",
        "phase1-torch-cu124",
        "phase1-torchcodec-support",
        "phase2-certified-runtime-lock",
    ]
    assert by_id["venv-create"].argv[:4] == (
        str(runtime.SYSTEM_PYTHON),
        "-m",
        "venv",
        "--copies",
    )
    torch = by_id["phase1-torch-cu124"].argv
    assert ("torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0") == (
        torch[5],
        torch[6],
        torch[7],
    )
    assert torch[-2:] == ("--index-url", runtime.PYTORCH_INDEX)
    phase2 = by_id["phase2-certified-runtime-lock"].argv
    assert phase2[-4:-2] == ("--extra-index-url", runtime.PYTORCH_INDEX)
    assert "--no-deps" in phase2


def test_embedded_system_python_probe_is_valid_python():
    compile(runtime._PYTHON_IDENTITY_CODE, "<stage1-python-identity>", "exec")
    completed = subprocess.run(
        [sys.executable, "-I", "-c", runtime._PYTHON_IDENTITY_CODE],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.stderr == ""
    identity = json.loads(completed.stdout)
    assert set(identity) == {
        "cache_tag",
        "executable",
        "implementation",
        "real_executable",
        "soabi",
        "version",
    }
    assert identity["implementation"] == "cpython"


def test_dry_run_binds_clean_repo_python_inputs_and_sanitized_environment(
    tmp_path, monkeypatch
):
    paths, system_python = _copy_sources(tmp_path, monkeypatch)
    result = runtime.dry_run(
        paths.forge_repo,
        paths.ai_toolkit_repo,
        paths.destination,
        paths.receipt,
        runner=_fake_runner(paths, system_python),
    )

    assert result["kind"] == runtime.PLAN_KIND
    assert result["ai_toolkit"]["commit"] == runtime.AI_TOOLKIT_COMMIT
    assert result["ai_toolkit"]["path"] == str(paths.ai_toolkit_repo)
    assert result["ai_toolkit"]["requirements"] == runtime._file_identity(
        paths.requirements
    )
    assert result["ai_toolkit"]["status"] == "clean-including-untracked"
    assert result["ai_toolkit"]["tree"] == "b" * 40
    assert result["ai_toolkit"]["tracked_entries"][0]["path"] == "requirements.txt"
    assert result["host"]["system_python"]["soabi"].startswith("cpython-310-")
    assert result["contract"]["base_image_sha256"] == runtime.BASE_IMAGE_SHA256
    environment = result["contract"]["command_environment"]
    assert environment["PIP_CONFIG_FILE"] == "/dev/null"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert not any("TOKEN" in key or "KEY" in key for key in environment)
    assert not paths.destination.exists()
    assert not paths.receipt.exists()
    assert not runtime._transient_cache_root(paths).exists()


def test_dirty_ai_toolkit_fails_before_materialization(tmp_path, monkeypatch):
    paths, system_python = _copy_sources(tmp_path, monkeypatch)

    with pytest.raises(runtime.Stage1RuntimeError, match="exactly clean"):
        runtime.dry_run(
            paths.forge_repo,
            paths.ai_toolkit_repo,
            paths.destination,
            paths.receipt,
            runner=_fake_runner(paths, system_python, dirty=True),
        )
    assert not paths.destination.exists()


@pytest.mark.parametrize(
    ("blob", "flag", "match"),
    [
        ("0" * 40, "H", "differs from index/HEAD"),
        (None, "S", "hidden/special index flag"),
        (None, "h", "hidden/special index flag"),
    ],
)
def test_actual_tracked_bytes_and_hidden_index_flags_fail_closed(
    tmp_path, blob, flag, match
):
    root = tmp_path / "repo"
    root.mkdir()
    tracked = root / "run.py"
    tracked.write_text("print('trusted')\n", encoding="utf-8")
    contents = tracked.read_bytes()
    actual_blob = (
        __import__("hashlib")
        .sha1(f"blob {len(contents)}\0".encode("ascii") + contents)
        .hexdigest()
    )

    with pytest.raises(runtime.Stage1RuntimeError, match=match):
        runtime._tracked_worktree_identity(
            root,
            f"100644 {blob or actual_blob} 0\trun.py\0",
            f"{flag} run.py\0",
            "test repo",
        )


def test_physical_staging_path_and_create_only_targets(tmp_path, monkeypatch):
    paths, _python = _copy_sources(tmp_path, monkeypatch)
    resolved = runtime._paths(
        paths.forge_repo,
        paths.ai_toolkit_repo,
        paths.destination,
        paths.receipt,
        destination_must_be_absent=True,
        receipt_must_be_absent=True,
    )
    assert resolved.ai_toolkit_repo == paths.ai_toolkit_repo

    paths.destination.mkdir()
    with pytest.raises(FileExistsError, match="destination already exists"):
        runtime._paths(
            paths.forge_repo,
            paths.ai_toolkit_repo,
            paths.destination,
            paths.receipt,
            destination_must_be_absent=True,
            receipt_must_be_absent=True,
        )


def test_symlink_or_path_alias_is_rejected(tmp_path, monkeypatch):
    paths, _python = _copy_sources(tmp_path, monkeypatch)
    alias = tmp_path / "ai-alias"
    alias.symlink_to(paths.ai_toolkit_repo, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink component"):
        runtime._paths(
            paths.forge_repo,
            alias,
            paths.destination,
            paths.receipt,
            destination_must_be_absent=True,
            receipt_must_be_absent=True,
        )


def test_read_os_release_accepts_standard_ubuntu_symlink_layout(tmp_path):
    root = tmp_path / "ubuntu-root"
    etc = root / "etc"
    canonical = root / "usr/lib/os-release"
    etc.mkdir(parents=True)
    canonical.parent.mkdir(parents=True)
    canonical.write_text('ID=ubuntu\nVERSION_ID="22.04"\n', encoding="utf-8")
    exposed = etc / "os-release"
    exposed.symlink_to("../usr/lib/os-release")

    result = runtime._read_os_release(exposed, canonical_path=canonical)

    assert result["id"] == "ubuntu"
    assert result["version_id"] == "22.04"
    assert result["file"]["path"] == str(canonical)
    assert result["file"]["sha256"] == runtime.file_sha256(canonical)


@pytest.mark.parametrize("indirection", [False, True])
def test_read_os_release_rejects_arbitrary_or_chained_symlink(tmp_path, indirection):
    root = tmp_path / "ubuntu-root"
    etc = root / "etc"
    canonical = root / "usr/lib/os-release"
    attacker = root / "attacker/os-release"
    etc.mkdir(parents=True)
    canonical.parent.mkdir(parents=True)
    attacker.parent.mkdir(parents=True)
    contents = 'ID=ubuntu\nVERSION_ID="22.04"\n'
    canonical.write_text(contents, encoding="utf-8")
    attacker.write_text(contents, encoding="utf-8")
    exposed = etc / "os-release"
    if indirection:
        redirect = root / "redirect"
        redirect.symlink_to("usr/lib/os-release")
        exposed.symlink_to("../redirect")
    else:
        exposed.symlink_to("../attacker/os-release")

    with pytest.raises(ValueError, match="point directly to canonical"):
        runtime._read_os_release(exposed, canonical_path=canonical)


def test_read_os_release_rejects_symlinked_canonical_file(tmp_path):
    root = tmp_path / "ubuntu-root"
    etc = root / "etc"
    canonical = root / "usr/lib/os-release"
    attacker = root / "attacker/os-release"
    etc.mkdir(parents=True)
    canonical.parent.mkdir(parents=True)
    attacker.parent.mkdir(parents=True)
    attacker.write_text('ID=ubuntu\nVERSION_ID="22.04"\n', encoding="utf-8")
    canonical.symlink_to("../../attacker/os-release")
    exposed = etc / "os-release"
    exposed.symlink_to("../usr/lib/os-release")

    with pytest.raises(
        ValueError, match="canonical os-release has a symlink component"
    ):
        runtime._read_os_release(exposed, canonical_path=canonical)


def test_materialization_receipt_is_canonical_create_only_and_detects_tree_drift(
    tmp_path, monkeypatch
):
    paths, system_python = _copy_sources(tmp_path, monkeypatch)
    runtime.dry_run(
        paths.forge_repo,
        paths.ai_toolkit_repo,
        paths.destination,
        paths.receipt,
        runner=_fake_runner(paths, system_python),
    )
    result = runtime.materialize(
        paths.forge_repo,
        paths.ai_toolkit_repo,
        paths.destination,
        paths.receipt,
        runner=_fake_runner(paths, system_python),
    )

    assert paths.receipt.read_bytes() == runtime.canonical_bytes(result) + b"\n"
    assert os.stat(paths.receipt).st_mode & 0o777 == 0o600
    assert result["tree_manifest"]["entry_count"] > 1
    assert result["verification"]["runtime_verifier_pass"] is True
    assert result["transient_materialization_cache"]["removed"] is True
    assert not runtime._transient_cache_root(paths).exists()
    assert runtime.validate_receipt(result) == result
    assert (
        runtime.validate_receipt(
            result,
            recapture=True,
            runner=_fake_runner(paths, system_python),
        )
        == result
    )

    (paths.destination / "tampered").write_text("changed\n", encoding="utf-8")
    with pytest.raises(runtime.Stage1RuntimeError, match="venv tree drifted"):
        runtime.validate_receipt(
            result,
            recapture=True,
            runner=_fake_runner(paths, system_python),
        )


def test_synthetic_stage_receipt_bootstrap_and_runner_preimport_lifecycle(
    tmp_path, monkeypatch
):
    paths, system_python = _copy_sources(tmp_path, monkeypatch)
    fake_runner = _fake_runner(paths, system_python)
    stage1_receipt = runtime.materialize(
        paths.forge_repo,
        paths.ai_toolkit_repo,
        paths.destination,
        paths.receipt,
        runner=fake_runner,
    )
    # This is the real structural/live Stage-1 validator; only the subprocess
    # transport is synthetic so the lifecycle remains CPU-only in CI.
    original_validate = runtime.validate_receipt
    monkeypatch.setattr(
        runtime,
        "validate_receipt",
        lambda value, recapture: original_validate(
            value, recapture=recapture, runner=fake_runner
        ),
    )
    fixed_root = tmp_path / "fixed"
    fixed_targets = {
        "forge_repo": str(fixed_root / "forge"),
        "ai_toolkit_repo": str(fixed_root / "ai-toolkit"),
        "venv": str(fixed_root / "venv"),
        "checkpoints": str(fixed_root / "checkpoints"),
        "dataset": str(fixed_root / "dataset"),
        "cache": str(fixed_root / "cache"),
        "campaign": str(fixed_root / "campaign"),
    }
    runtime_cache_root = Path(fixed_targets["cache"]) / "krea-runtime"
    runtime_cache_policy = {
        **bootstrap._RUNTIME_CACHE_POLICY,
        "root": str(runtime_cache_root),
    }
    monkeypatch.setattr(bootstrap, "_FIXED_TARGETS", fixed_targets)
    monkeypatch.setattr(bootstrap, "_RUNTIME_CACHE_ROOT", runtime_cache_root)
    monkeypatch.setattr(bootstrap, "_RUNTIME_CACHE_POLICY", runtime_cache_policy)
    evidence = tmp_path / "evidence"
    payload = {
        "schema": 1,
        "kind": "forge-krea-host-bootstrap-spec",
        "sources": {
            "forge_repo": str(paths.forge_repo),
            "ai_toolkit_repo": str(paths.ai_toolkit_repo),
            "venv": str(paths.destination),
            "checkpoints": str(tmp_path / "volatile/checkpoints"),
            "dataset": str(tmp_path / "volatile/dataset"),
            "cache": str(tmp_path / "volatile/cache"),
            "campaign": str(evidence / "campaign"),
            "evidence_root": str(evidence),
        },
        "source_identities": {
            "forge_commit": stage1_receipt["forge"]["commit"],
            "ai_toolkit_commit": stage1_receipt["ai_toolkit"]["commit"],
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
            "container_image_reference": "sha256:" + "a" * 64,
            "container_image_sha256": "a" * 64,
            "execution_surface": "staged_host_venv",
            "ai_toolkit_dir": fixed_targets["ai_toolkit_repo"],
            "jit_enabled": True,
            "stage1_runtime_receipt": {
                "path": str(paths.receipt),
                "file_sha256": runtime.file_sha256(paths.receipt),
                "receipt_sha256": stage1_receipt["receipt_sha256"],
            },
            "runtime_cache_policy": dict(runtime_cache_policy),
        },
        "gpu_execution_authorized": False,
    }
    spec = bootstrap.seal_spec(payload)
    venv_tree = bootstrap._tree_identity(paths.destination, "synthetic staged venv")
    bound_runtime = bootstrap._stage1_runtime_identity(
        spec,
        forge_identity={
            "commit": stage1_receipt["forge"]["commit"],
            "tree": stage1_receipt["forge"]["tree"],
        },
        ai_toolkit_identity={
            "commit": stage1_receipt["ai_toolkit"]["commit"],
            "tree": stage1_receipt["ai_toolkit"]["tree"],
        },
        venv_tree=venv_tree,
        materializer_sha256=stage1_receipt["inputs"]["materializer"]["sha256"],
    )
    assert bound_runtime["receipt_sha256"] == stage1_receipt["receipt_sha256"]
    assert ladder._preimport_tree_identity(paths.destination) == venv_tree

    # Compose the next two modules against fixed-target stand-ins. Only mount,
    # host-probe, and exec transports are mocked; binding/receipt/reexec logic
    # remains real.
    for relative in bootstrap._CALIBRATION_ARTIFACTS.values():
        source = paths.forge_repo / relative
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"synthetic tracked artifact {relative}\n")
    for name, target_text in fixed_targets.items():
        target = Path(target_text)
        if name == "forge_repo":
            shutil.copytree(paths.forge_repo, target)
        elif name == "ai_toolkit_repo":
            shutil.copytree(paths.ai_toolkit_repo, target)
        elif name == "venv":
            shutil.copytree(paths.destination, target)
        else:
            target.mkdir(parents=True)
    for leaf in bootstrap._CAMPAIGN_LEAVES:
        (Path(fixed_targets["campaign"]) / leaf).mkdir(parents=True, exist_ok=True)
    runtime_cache_root.mkdir(mode=0o700)
    for name in ("checkpoints", "dataset", "cache"):
        Path(payload["sources"][name]).mkdir(parents=True, exist_ok=True)

    calibration_artifacts = {
        name: {
            "relative_path": relative,
            "sha256": runtime.file_sha256(paths.forge_repo / relative),
        }
        for name, relative in bootstrap._CALIBRATION_ARTIFACTS.items()
    }
    source_identity = {
        "forge_repo": {
            "commit": stage1_receipt["forge"]["commit"],
            "tree": stage1_receipt["forge"]["tree"],
        },
        "ai_toolkit_repo": {
            "commit": stage1_receipt["ai_toolkit"]["commit"],
            "tree": stage1_receipt["ai_toolkit"]["tree"],
        },
        "venv_python": {
            "relative_path": "bin/python",
            "is_symlink": False,
            "resolved_relative_path": "bin/python",
            "resolved_sha256": runtime.file_sha256(paths.destination / "bin/python"),
        },
        "venv_tree": venv_tree,
        "stage1_runtime": bound_runtime,
        "calibration_artifacts": calibration_artifacts,
        "filesystems": {},
    }
    host_identity = {
        "docker": {"docker_root_dir": payload["sources"]["checkpoints"]},
        "trusted_executables": {
            "git": {"requested_path": "/usr/bin/git", "synthetic": True}
        },
    }
    source_by_target = {
        Path(target): Path(payload["sources"][name])
        for name, target in fixed_targets.items()
    }
    real_samefile = os.path.samefile

    def synthetic_samefile(left, right):
        left_path, right_path = Path(left), Path(right)
        if (
            source_by_target.get(left_path) == right_path
            or source_by_target.get(right_path) == left_path
        ):
            return True
        if {str(left_path), str(right_path)} == {sys.executable, "/usr/bin/python3"}:
            return True
        return real_samefile(left, right)

    monkeypatch.setattr(bootstrap.os.path, "samefile", synthetic_samefile)
    monkeypatch.setattr(bootstrap, "_is_mountpoint", lambda _path: True)

    def synthetic_filesystem(path, *, require_mountpoint):
        del require_mountpoint
        read_only = str(path) in {
            fixed_targets[name] for name in bootstrap._READ_ONLY_BINDINGS
        }
        return {
            "source": "/dev/synthetic",
            "target": str(path),
            "filesystem_type": "ext4",
            "mount_options": ["ro" if read_only else "rw"],
            "device_major_minor": "1:1",
            "device_id": 1,
        }

    cache_identity = {
        "path": str(runtime_cache_root),
        "device_id": 1,
        "mode": 0o700,
        "uid": 0,
        "policy": dict(runtime_cache_policy),
    }
    monkeypatch.setattr(bootstrap, "_filesystem", synthetic_filesystem)
    monkeypatch.setattr(bootstrap, "_host_identity", lambda _spec: host_identity)
    monkeypatch.setattr(bootstrap, "_source_identity", lambda _spec: source_identity)
    monkeypatch.setattr(
        bootstrap,
        "_runtime_cache_identity",
        lambda require_empty: cache_identity,
    )

    composed = bootstrap._preflight(spec, require_bindings=True)
    assert set(composed["bindings"]) == set(fixed_targets) | {"runtime_cache"}
    layout_receipt = bootstrap.build_receipt(spec)
    assert layout_receipt["layout_identity"] == composed
    layout_path = Path(fixed_targets["campaign"]) / "controls/layout.receipt.json"
    layout_path.write_bytes(krea_provenance.canonical_bytes(layout_receipt) + b"\n")

    host_path = Path(fixed_targets["campaign"]) / "controls/host.json"
    host_document = {
        "bootstrap_receipt": {
            "path": str(layout_path),
            "file_sha256": runtime.file_sha256(layout_path),
        }
    }
    host_path.write_bytes(krea_provenance.canonical_bytes(host_document) + b"\n")
    plan_path = Path(fixed_targets["campaign"]) / "controls/timing-plan.json"
    plan_document = {
        "host_execution_manifest": {
            "path": str(host_path),
            "sha256": runtime.file_sha256(host_path),
        }
    }
    plan_path.write_bytes(krea_provenance.canonical_bytes(plan_document) + b"\n")

    monkeypatch.setattr(ladder, "_STAGED_VENV_ROOT", Path(fixed_targets["venv"]))
    monkeypatch.setattr(ladder, "_RUNTIME_CACHE_ROOT", runtime_cache_root)
    monkeypatch.setattr(
        ladder,
        "_preimport_executable_identity",
        lambda _path: host_identity["trusted_executables"]["git"],
    )
    monkeypatch.setattr(
        ladder,
        "_preimport_git_identity",
        lambda path, _commit: source_identity[
            "forge_repo" if path == paths.forge_repo else "ai_toolkit_repo"
        ],
    )
    monkeypatch.setattr(
        ladder.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "", ""),
    )

    class ExecReached(RuntimeError):
        pass

    executed = {}

    def capture_exec(path, argv, environment):
        executed.update(path=path, argv=argv, environment=environment)
        raise ExecReached

    monkeypatch.setattr(ladder.os, "execve", capture_exec)
    for name in (
        "LD_PRELOAD",
        "PYTHONHOME",
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
        "XDG_CACHE_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        ladder._RUNTIME_CACHE_NAMESPACE_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FORGE_KREA_TIMING_CAPTURE_ID", "timing-a")
    monkeypatch.setenv("FORGE_KREA_TIMING_SOCKET", "/tmp/timing.sock")
    monkeypatch.setenv("FORGE_KREA_TIMING_PROBE_CONTRACT_SHA256", "a" * 64)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "operator-allocator")
    monkeypatch.setenv("HF_HOME", "/tmp/operator-hf")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/operator-lib")
    args = type("Args", (), {"timing_probe_plan": plan_path, "execution_plan": None})()
    with pytest.raises(ExecReached):
        ladder._trusted_stage1_reexec(args)
    assert executed["path"] == str(Path(fixed_targets["venv"]) / "bin/python")
    assert Path(
        executed["environment"][ladder._RUNTIME_CACHE_NAMESPACE_ENV]
    ).is_relative_to(runtime_cache_root)
    assert "PYTORCH_CUDA_ALLOC_CONF" not in executed["environment"]
    assert "HF_HOME" not in executed["environment"]
    assert "LD_LIBRARY_PATH" not in executed["environment"]


def test_failed_materialization_cleans_only_its_transient_cache(tmp_path, monkeypatch):
    paths, system_python = _copy_sources(tmp_path, monkeypatch)
    successful = _fake_runner(paths, system_python)

    def fail_phase1(command, **kwargs):
        argv = [str(item) for item in command]
        if "--requirement" in argv and str(paths.requirements) in argv:
            return subprocess.CompletedProcess(argv, 2, stdout="", stderr="network\n")
        return successful(command, **kwargs)

    with pytest.raises(runtime.Stage1RuntimeError, match="failed after 5"):
        runtime.materialize(
            paths.forge_repo,
            paths.ai_toolkit_repo,
            paths.destination,
            paths.receipt,
            runner=fail_phase1,
        )
    assert not runtime._transient_cache_root(paths).exists()
    assert not paths.receipt.exists()


def test_minimal_runtime_environment_does_not_inherit_operator_controls(monkeypatch):
    for name, value in {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": "/tmp/operator-hf",
        "HF_HUB_CACHE": "/tmp/operator-hub",
        "TRANSFORMERS_CACHE": "/tmp/operator-transformers",
        "TORCH_HOME": "/tmp/operator-torch",
        "CUDA_HOME": "/tmp/operator-cuda",
        "LD_LIBRARY_PATH": "/tmp/operator-lib",
        "HTTPS_PROXY": "http://operator.invalid",
        "OMP_NUM_THREADS": "99",
        "NCCL_DEBUG": "TRACE",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }.items():
        monkeypatch.setenv(name, value)
    cache = {
        "HOME": "/cache/krea-runtime/run/home",
        "XDG_CACHE_HOME": "/cache/krea-runtime/run/xdg",
        "TORCHINDUCTOR_CACHE_DIR": "/cache/krea-runtime/run/torchinductor",
        "TRITON_CACHE_DIR": "/cache/krea-runtime/run/triton",
        ladder._RUNTIME_CACHE_NAMESPACE_ENV: "/cache/krea-runtime/run",
    }
    environment = ladder._minimal_runtime_environment(
        cache=cache,
        trusted_marker="a" * 64,
        jit_enabled="1",
        seed="42",
    )

    for name in (
        "PYTORCH_CUDA_ALLOC_CONF",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "CUDA_HOME",
        "LD_LIBRARY_PATH",
        "HTTPS_PROXY",
        "OMP_NUM_THREADS",
        "NCCL_DEBUG",
        "CUBLAS_WORKSPACE_CONFIG",
    ):
        assert name not in environment
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["PATH"] == "/app/venv/bin:/usr/bin:/bin"


def test_receipt_rejects_command_drift(tmp_path, monkeypatch):
    paths, system_python = _copy_sources(tmp_path, monkeypatch)
    result = runtime.materialize(
        paths.forge_repo,
        paths.ai_toolkit_repo,
        paths.destination,
        paths.receipt,
        runner=_fake_runner(paths, system_python),
    )
    result["contract"]["indexes"]["phase1_torch_index_url"] = "https://example.com"
    without_hash = {
        key: value for key, value in result.items() if key != "receipt_sha256"
    }
    result["receipt_sha256"] = runtime.canonical_sha256(without_hash)

    with pytest.raises(ValueError, match="command/contract drift"):
        runtime.validate_receipt(result)
