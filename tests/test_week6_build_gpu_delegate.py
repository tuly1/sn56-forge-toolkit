from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
DELEGATE = ROOT / "ops" / "release" / "sn56-week6-build-gpu-cert.py"
INTEGRATION = ROOT / "tests" / "integration" / "sn56-week6-release-dry-run.sh"
GIT = shutil.which("git")


def _delegate():
    spec = importlib.util.spec_from_file_location("sn56_week6_build_gpu_cert", DELEGATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(repository: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    assert GIT is not None
    completed = subprocess.run(
        [GIT, "-c", f"safe.directory={repository}", "-C", str(repository), *arguments],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    return completed.stdout


def _repository(tmp_path: Path):
    module = _delegate()
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "release-test@example.invalid")
    _git(repository, "config", "user.name", "Release Test")
    (repository / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    (repository / "forge").mkdir()
    (repository / "forge" / "contract.py").write_text("VALUE = 'A'\n", encoding="utf-8")
    docker = repository / "ops" / "docker"
    docker.mkdir(parents=True)
    (docker / "standalone-image-toolkit-trainer.dockerfile").write_text(
        "FROM example.invalid/toolkit@sha256:" + "1" * 64 + "\n",
        encoding="utf-8",
    )
    (docker / "standalone-image-trainer.dockerfile").write_text(
        "FROM example.invalid/legacy@sha256:" + "2" * 64 + "\n",
        encoding="utf-8",
    )
    (docker / "image-runtime-lock.txt").write_text("lock\n", encoding="utf-8")
    (docker / "image-runtime-phase1-constraints.txt").write_text(
        "constraints\n", encoding="utf-8"
    )
    (docker / "verify_image_runtime.py").write_text("print('PASS')\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "fixture")
    identity = module.SourceIdentity(
        _git(repository, "rev-parse", "HEAD").decode().strip(),
        _git(repository, "rev-parse", "HEAD^{tree}").decode().strip(),
        _git(repository, "rev-parse", "HEAD:forge").decode().strip(),
    )
    tools = {**module.ABSOLUTE_TOOLS, "git": GIT}
    return module, repository, identity, tools


def test_delegate_has_fixed_policy_neutral_two_subject_matrix_and_no_asserts():
    module = _delegate()
    source = DELEGATE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert module.RESULT_SCHEMA == "sn56.week6.build-gpu-cert.v2"
    assert [(item.name, item.dockerfile) for item in module.SUBJECTS] == [
        ("toolkit", "ops/docker/standalone-image-toolkit-trainer.dockerfile"),
        ("legacy", "ops/docker/standalone-image-trainer.dockerfile"),
    ]
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))
    lowered = source.lower()
    assert "week5" not in lowered
    assert "k5-global" not in lowered
    assert "ideogram-disposition" not in lowered


def test_require_is_an_explicit_exception_even_under_optimized_python():
    module = _delegate()
    with pytest.raises(module.DelegateError, match="deliberate failure"):
        module.require(False, "deliberate failure")

    program = (
        "import importlib.util,sys;"
        f"p={str(DELEGATE)!r};"
        "s=importlib.util.spec_from_file_location('delegate_optimized',p);"
        "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
        "m.require(False,'optimized failure')"
    )
    completed = subprocess.run(
        [sys.executable, "-O", "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "optimized failure" in completed.stderr


def test_cpu_integration_can_never_emit_production_pass():
    module = _delegate()

    assert module.result_state_for_mode(module.PRODUCTION_MODE) == "PASS"
    assert (
        module.result_state_for_mode(module.CPU_INTEGRATION_MODE)
        == "DRY_RUN_PASS"
    )
    with pytest.raises(module.DelegateError, match="unknown"):
        module.result_state_for_mode("stub-but-pass")

    hostile_tools = {name: "/must/not/execute" for name in module.ABSOLUTE_TOOLS}
    assert module.observe_host_gpu(module.CPU_INTEGRATION_MODE, hostile_tools) == {
        "accelerator_identity": "INTEGRATION-STUB-NO-GPU-CLAIM|0-MiB",
        "claim": "none",
        "reason": "physical-gpu-boundary-stubbed-for-cpu-integration",
        "state": "STUBBED",
    }
    assert module.observe_image_gpu(
        module.CPU_INTEGRATION_MODE, hostile_tools, "invalid/on-purpose:tag"
    ) == {
        "claim": "none",
        "reason": "physical-container-gpu-boundary-stubbed-for-cpu-integration",
        "state": "STUBBED",
    }


def test_cpu_surface_imports_torch_and_forge_before_the_gpu_boundary(monkeypatch):
    module = _delegate()
    observed: list[tuple[str, str]] = []

    def successful_import(_tools, image_tag, program, *program_args):
        observed.append((image_tag, program))
        assert not program_args
        return subprocess.CompletedProcess(
            ["python3"],
            0,
            stdout=(
                b'{"forge":"/app/forge/__init__.py","state":"PASS",'
                b'"torch":"2.6.0+cu124"}\n'
            ),
            stderr=b"",
        )

    monkeypatch.setattr(module, "run_image_python", successful_import)
    result = module.observe_image_cpu_imports(
        {"docker": "/usr/bin/docker"}, "sn56/toolkit:exact-head"
    )

    assert result["state"] == "PASS"
    assert result["forge"] == "/app/forge/__init__.py"
    assert observed == [("sn56/toolkit:exact-head", module.IMAGE_CPU_IMPORT_PROGRAM)]
    assert "import torch" in module.IMAGE_CPU_IMPORT_PROGRAM
    assert "import forge" in module.IMAGE_CPU_IMPORT_PROGRAM
    assert "torch.cuda" not in module.IMAGE_CPU_IMPORT_PROGRAM
    certify_source = (
        DELEGATE.read_text(encoding="utf-8")
        .split("def certify_cpu_image_surface", 1)[1]
        .split("def observe_host_gpu", 1)[0]
    )
    assert "observe_image_cpu_imports" in certify_source

    def broken_import(_tools, _image_tag, _program, *_program_args):
        return subprocess.CompletedProcess(
            ["python3"], 0, stdout=b'{"state":"PASS","torch":"2.6.0"}\n', stderr=b""
        )

    monkeypatch.setattr(module, "run_image_python", broken_import)
    with pytest.raises(module.DelegateError, match="unexpected path"):
        module.observe_image_cpu_imports(
            {"docker": "/usr/bin/docker"}, "sn56/toolkit:broken"
        )


def test_fixed_subprocess_environment_does_not_inherit_operator_state(monkeypatch):
    module = _delegate()
    monkeypatch.setenv("BASH_ENV", "/tmp/hostile")
    monkeypatch.setenv("PYTHONOPTIMIZE", "2")
    monkeypatch.setenv("PYTHONPATH", "/tmp/hostile")
    monkeypatch.setenv("DOCKER_HOST", "tcp://hostile.invalid:2375")
    monkeypatch.setenv("HF_TOKEN", "secret")

    value = module.fixed_command_env()
    assert value == module.FIXED_ENV
    for name in ("BASH_ENV", "PYTHONOPTIMIZE", "PYTHONPATH", "DOCKER_HOST", "HF_TOKEN"):
        assert name not in value
    assert all(os.path.isabs(path) for path in module.ABSOLUTE_TOOLS.values())
    assert value["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert value["GIT_CONFIG_NOSYSTEM"] == "1"
    assert value["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert value["GIT_CONFIG_COUNT"] == str(len(module.GIT_CONFIG_OVERRIDES))
    assert value["GIT_CONFIG_KEY_0"] == "core.fsmonitor"
    assert value["GIT_CONFIG_VALUE_0"] == "false"


def test_integration_harness_calls_real_outer_wrapper_and_forbids_pass():
    source = INTEGRATION.read_text(encoding="utf-8")

    assert 'wrapper=${clone}/ops/release/sn56-week6-final-release-cert.sh' in source
    assert '/bin/sh "${wrapper}"' in source
    assert "sn56-week6-build-gpu-cert.py" not in source
    assert "SN56_RELEASE_CERT_MODE=cpu-integration" in source
    assert "SN56_WEEK6_FINAL_RELEASE_CERT=DRY_RUN_PASS" in source
    assert "CPU integration illegally emitted production PASS" in source
    assert "git_isolated()" in source
    assert "GIT_CONFIG_KEY_0=core.fsmonitor" in source
    assert "GIT_CONFIG_VALUE_0=false" in source
    assert "core.fsmonitor=false" in source
    assert source.count("git_isolated ") >= 7
    assert "sleep 2 #" in source
    assert "url.https://attacker.invalid/spoof.git.insteadOf" in source
    assert ".git/info/attributes" in source
    assert "filter.sn56-hostile.clean" in source
    assert "hostile-filter-executed" in source


def test_integration_harness_snapshots_the_complete_pipeline_status():
    source = INTEGRATION.read_text(encoding="utf-8")
    snapshot = 'pipeline_status=("${PIPESTATUS[@]}")'
    wrapper_status = 'wrapper_rc=${pipeline_status[0]}'
    tee_status = 'tee_rc=${pipeline_status[1]}'

    assert snapshot in source
    assert source.index(snapshot) < source.index(wrapper_status) < source.index(tee_status)
    assert "wrapper_rc=${PIPESTATUS[0]}" not in source
    assert "tee_rc=${PIPESTATUS[1]}" not in source

    completed = subprocess.run(
        [
            "/bin/bash",
            "-c",
            """
set -uo pipefail
set +e
/bin/bash -c 'exit 7' | /usr/bin/tee /dev/null
pipeline_status=("${PIPESTATUS[@]}")
wrapper_rc=${pipeline_status[0]}
tee_rc=${pipeline_status[1]}
[[ ${wrapper_rc} -eq 7 && ${tee_rc} -eq 0 ]]
""",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_archive_materialization_ignores_assume_unchanged_worktree_bytes(tmp_path):
    module, repository, identity, tools = _repository(tmp_path)
    tracked = repository / "forge" / "contract.py"
    _git(repository, "update-index", "--assume-unchanged", "forge/contract.py")
    tracked.write_text("VALUE = 'B'\n", encoding="utf-8")

    with pytest.raises(module.DelegateError, match="source checkout has changed"):
        module.verify_source_repository(repository, identity, tools)
    work = tmp_path / "work-a"
    work.mkdir()
    materialized = module.materialize_exact_archive(repository, identity, work, tools)

    assert (materialized.root / "forge" / "contract.py").read_text() == "VALUE = 'A'\n"
    assert tracked.read_text() == "VALUE = 'B'\n"
    assert not (materialized.root / ".git").exists()
    assert re_full_sha(materialized.archive_sha256)
    assert re_full_sha(materialized.file_manifest_sha256)


def test_archive_materialization_ignores_skip_worktree_bytes(tmp_path):
    module, repository, identity, tools = _repository(tmp_path)
    tracked = repository / "forge" / "contract.py"
    _git(repository, "update-index", "--skip-worktree", "forge/contract.py")
    tracked.write_text("VALUE = 'B'\n", encoding="utf-8")

    with pytest.raises(module.DelegateError, match="source checkout has changed"):
        module.verify_source_repository(repository, identity, tools)
    work = tmp_path / "work-b"
    work.mkdir()
    materialized = module.materialize_exact_archive(repository, identity, work, tools)

    assert (materialized.root / "forge" / "contract.py").read_text() == "VALUE = 'A'\n"
    assert tracked.read_text() == "VALUE = 'B'\n"


def test_malicious_local_fsmonitor_cannot_spoof_clean_source(tmp_path):
    module, repository, identity, tools = _repository(tmp_path)
    tracked = repository / "forge" / "contract.py"
    _git(repository, "config", "core.fsmonitor", "sleep 2 #")
    _git(repository, "update-index", "--fsmonitor")
    tracked.write_text("VALUE = 'B'\n", encoding="utf-8")

    started = time.monotonic()
    with pytest.raises(module.DelegateError, match="source checkout has changed"):
        module.verify_source_repository(repository, identity, tools)
    assert time.monotonic() - started < 1.5

    command = module.git_command(tools, repository, "status", "--porcelain=v1")
    assert "core.fsmonitor=false" in command
    assert module.FIXED_ENV["GIT_CONFIG_KEY_0"] == "core.fsmonitor"
    assert module.FIXED_ENV["GIT_CONFIG_VALUE_0"] == "false"


def test_hostile_info_attributes_filter_cannot_spoof_or_execute(tmp_path):
    module, repository, identity, tools = _repository(tmp_path)
    tracked = repository / "forge" / "contract.py"
    marker = tmp_path / "hostile-filter-executed"
    driver = repository / ".git" / "hostile-clean-filter"
    driver.write_text(
        "#!/bin/sh\n"
        "/usr/bin/touch -- \"$1\"\n"
        "/usr/bin/printf \"VALUE = 'A'\\n\"\n",
        encoding="utf-8",
    )
    driver.chmod(0o700)
    info = repository / ".git" / "info"
    info.mkdir(exist_ok=True)
    (info / "attributes").write_text(
        "forge/contract.py filter=hostile\n",
        encoding="utf-8",
    )
    _git(repository, "config", "filter.hostile.clean", f"{driver} {marker}")
    tracked.write_text("VALUE = 'B'\n", encoding="utf-8")

    # Reproduce the displaced implementation's false-clean result and command
    # execution, then prove the release delegate does neither.
    assert _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored=matching",
    ) == b""
    assert marker.is_file()
    marker.unlink()

    with pytest.raises(module.DelegateError, match="source checkout has changed"):
        module.verify_source_repository(repository, identity, tools)

    assert not marker.exists(), "release verification executed the hostile clean filter"


def test_live_h100_identity_is_derived_from_nvidia_smi(monkeypatch):
    module = _delegate()

    def observed(argv, **_kwargs):
        assert argv == [
            "/usr/bin/nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                b"NVIDIA H100 PCIe, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, "
                b"570.86.15, 81559\n"
            ),
            stderr=b"",
        )

    monkeypatch.setattr(module, "run_capture", observed)
    result = module.observe_host_gpu(
        module.PRODUCTION_MODE,
        {"nvidia_smi": "/usr/bin/nvidia-smi"},
    )

    assert result["accelerator_identity"] == "NVIDIA H100 PCIe|81559-MiB"
    assert result["state"] == "PASS"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"NVIDIA H100 PCIe, GPU-a, 570.86.15, 81559\nsecond\n", "exactly one"),
        (b"NVIDIA A100, GPU-aaaaaaaa, 570.86.15, 81559\n", "not an H100"),
        (b"NVIDIA H100 PCIe, invalid, 570.86.15, 81559\n", "UUID"),
        (b"NVIDIA H100 PCIe, GPU-aaaaaaaa, driver, 81559\n", "driver"),
    ],
)
def test_live_gpu_identity_rejects_ambiguous_or_invalid_rows(
    monkeypatch, payload, message
):
    module = _delegate()
    monkeypatch.setattr(
        module,
        "run_capture",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv, 0, stdout=payload, stderr=b""
        ),
    )

    with pytest.raises(module.DelegateError, match=message):
        module.observe_host_gpu(
            module.PRODUCTION_MODE,
            {"nvidia_smi": "/usr/bin/nvidia-smi"},
        )


def test_clean_clone_gate_rejects_ignored_execution_surface(tmp_path):
    module, repository, identity, tools = _repository(tmp_path)
    ignored = repository / "forge" / "__pycache__"
    ignored.mkdir()
    (ignored / "contract.cpython-311.pyc").write_bytes(b"not bytecode")

    with pytest.raises(module.DelegateError, match="ignored surfaces"):
        module.verify_source_repository(repository, identity, tools)


def test_parse_ls_tree_rejects_symlinks_and_gitlinks():
    module = _delegate()
    with pytest.raises(module.DelegateError, match="unsupported committed mode"):
        module.parse_ls_tree(b"120000 blob " + b"a" * 40 + b"\tlink\0")
    with pytest.raises(module.DelegateError, match="unsupported committed object"):
        module.parse_ls_tree(b"160000 commit " + b"a" * 40 + b"\tsubmodule\0")


def test_both_real_dockerfiles_use_only_digest_pinned_bases():
    module = _delegate()
    for subject in module.SUBJECTS:
        bases = module.validate_dockerfile_digest_pins(ROOT / subject.dockerfile)
        assert bases
        assert all("@sha256:" in base for base in bases)


def test_build_command_is_no_cache_for_each_fixed_subject():
    module = _delegate()
    tools = {**module.ABSOLUTE_TOOLS, "docker": "/usr/bin/docker"}
    for subject in module.SUBJECTS:
        command = module.docker_build_command(
            tools, subject, f"sn56/{subject.name}:deadbeef"
        )
        assert command[0] == "/usr/bin/docker"
        assert command[1:3] == ["build", "--no-cache"]
        assert command[command.index("--file") + 1] == subject.dockerfile
        assert command[-1] == "."


def test_build_loop_keeps_the_three_filesystem_pressure_guard():
    module = _delegate()
    source = DELEGATE.read_text(encoding="utf-8")

    assert "pressure_paths={" in source
    assert '"root": (Path("/"), ROOT_PRESSURE_FLOOR)' in source
    assert '"work": (work_base, WORK_PRESSURE_FLOOR)' in source
    assert '"evidence": (evidence.stage_path, EVIDENCE_PRESSURE_FLOOR)' in source
    assert "build-pressure.tsv" in source
    assert module.ROOT_PRESSURE_FLOOR < module.ROOT_START_MIN
    assert module.WORK_PRESSURE_FLOOR < module.WORK_START_MIN
    assert module.EVIDENCE_PRESSURE_FLOOR < module.EVIDENCE_START_MIN


def test_result_env_has_exact_week6_schema_and_stub_state():
    module = _delegate()
    result = {
        "schema": module.RESULT_SCHEMA,
        "state": module.DRY_RUN_PASS_STATE,
        "mode": module.CPU_INTEGRATION_MODE,
        "certificate_scope": "toolkit-krea-only",
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "forge_tree": "c" * 40,
        "source_archive_sha256": "d" * 64,
        "source_manifest_sha256": "e" * 64,
        "production_manifest_sha256": "f" * 64,
        "toolkit_dockerfile_sha256": "1" * 64,
        "legacy_dockerfile_sha256": "2" * 64,
        "toolkit_image_tag": "sn56/toolkit:deadbeef",
        "toolkit_image_id": "sha256:" + "3" * 64,
        "legacy_image_tag": "sn56/legacy:deadbeef",
        "legacy_image_id": "sha256:" + "4" * 64,
        "gpu_boundary": "STUBBED_NO_CLAIM",
        "accelerator_identity": "INTEGRATION-STUB-NO-GPU-CLAIM|0-MiB",
        "completed_at_utc": "2026-08-04T00:00:00Z",
    }
    payload = module.result_env_payload(result).decode("utf-8")

    assert payload.startswith("schema=sn56.week6.build-gpu-cert.v2\n")
    assert "state=DRY_RUN_PASS\n" in payload
    assert "gpu_boundary=STUBBED_NO_CLAIM\n" in payload
    assert "state=PASS\n" not in payload


def test_atomic_evidence_publishes_only_complete_manifest(tmp_path):
    module = _delegate()
    base = tmp_path / "evidence"
    envelope = module.AtomicEvidence(base, "integration-proof")
    envelope.write_bytes("nested/one.txt", b"one\n")
    envelope.write_json("two.json", {"state": "PASS"})
    assert not (base / "integration-proof").exists()

    published = envelope.publish()
    envelope.close()

    assert published == base / "integration-proof"
    assert (published / "nested" / "one.txt").read_bytes() == b"one\n"
    manifest = (published / "MANIFEST.sha256").read_text(encoding="ascii")
    assert hashlib.sha256(b"one\n").hexdigest() in manifest
    assert "nested/one.txt" in manifest
    assert "two.json" in manifest
    assert not os.access(published / "two.json", os.W_OK)

    with pytest.raises(module.DelegateError, match="already exists"):
        module.AtomicEvidence(base, "integration-proof")


def test_atomic_evidence_rejects_symlinked_base_component(tmp_path):
    module = _delegate()
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(module.DelegateError, match="symlink component"):
        module.AtomicEvidence(link / "evidence", "proof")


def test_argument_contract_rejects_stub_pass_aliases():
    module = _delegate()
    base = dict(
        source_checkout="/source",
        release_commit="a" * 40,
        release_tree="b" * 40,
        forge_tree="c" * 40,
        certificate_scope="toolkit-krea-only",
        evidence_base="/evidence",
        evidence_namespace="proof",
        work_base="/work",
        toolkit_image_tag="sn56/toolkit:proof",
        legacy_image_tag="sn56/legacy:proof",
        expected_docker_root="/ephemeral/docker",
        expected_containerd_root="/ephemeral/containerd",
        build_timeout_seconds=10,
    )
    with pytest.raises(module.DelegateError, match="invalid delegate mode"):
        module.validate_args(argparse.Namespace(mode="cpu-integration-pass", **base))


def re_full_sha(value: str) -> bool:
    return len(value) == 64 and set(value) <= set("0123456789abcdef")
