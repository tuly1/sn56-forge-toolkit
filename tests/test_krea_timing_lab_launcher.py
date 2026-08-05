from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest
import yaml

from forge import adaptive_timing, config, krea_runtime
from forge.data.schema import ImageSpec
from ops.calibration import run_krea_timing_lab as launcher


def _runtime_manifest(runtime: Path) -> Path:
    capabilities = {
        name: True for name in krea_runtime.RUNTIME_MANIFEST_CAPABILITIES
    }
    manifest = {
        "schema": 1,
        "runtime_contract_id": krea_runtime.RUNTIME_CONTRACT_ID,
        "base_commit": krea_runtime.PINNED_BASE_COMMIT,
        "capabilities": capabilities,
        "evidence": {
            name: f"tests/{name}.py"
            for name in krea_runtime.RUNTIME_MANIFEST_CAPABILITIES
        },
    }
    manifest_path = runtime / krea_runtime.CAPABILITY_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _fake_run_script() -> str:
    """Supplemental supervisor lifecycle fake; not loader-compatibility proof."""

    return '''import hashlib, json, os, stat, struct, sys, time, yaml
from pathlib import Path
config_path = Path(sys.argv[1])
if config_path.suffix != ".yaml" or not config_path.is_file():
    raise SystemExit("launcher did not execute a real .yaml config")
if stat.S_IMODE(config_path.parent.stat().st_mode) != 0o700:
    raise SystemExit("launcher workspace is not private")
config_payload = config_path.read_bytes()
poison = {"PYTHONPATH", "LD_PRELOAD", "HF_TOKEN", "FORGE_KREA_BUNDLE", "SN56_RELEASE_COMMIT"}
if poison.intersection(os.environ):
    raise SystemExit("caller-controlled poison leaked into child environment")
if os.environ.get("PATH") != "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin":
    raise SystemExit("child PATH is not the fixed supervisor constant")
if os.environ.get("HOME") != str(config_path.parent / "child-home"):
    raise SystemExit("child HOME is not supervisor-owned")
if os.environ.get("TMPDIR") != str(config_path.parent / "child-tmp"):
    raise SystemExit("child TMPDIR is not supervisor-owned")
document = yaml.safe_load(config_payload)
repo_name = document["config"]["name"]
process = document["config"]["process"][0]
save_root = Path(process["training_folder"]) / repo_name
planned_steps = process["train"]["steps"]
checkpoint = save_root / f"{repo_name}_000000200.safetensors"
terminal = save_root / f"{repo_name}.safetensors"
def write(path, step):
    metadata = {"training_info": json.dumps({"step": step, "epoch": 1})}
    header = json.dumps({"__metadata__": metadata, "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode("utf-8")
    Path(path).write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.0))
time.sleep(0.05)
write(checkpoint, 200)
time.sleep(0.08)
write(terminal, planned_steps)
print(f"EXECUTED_CONFIG_SHA256={hashlib.sha256(config_payload).hexdigest()}", flush=True)
print(f"COMMITTED_RUNTIME_A_EXECUTED=1 {planned_steps}/{planned_steps} loss=0.1", flush=True)
'''


def _git(runtime: Path, *arguments: str) -> str:
    completed = subprocess.run(
        [launcher._ABSOLUTE_GIT, "-C", str(runtime), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _initialize_runtime_object_store(
    runtime: Path,
    monkeypatch,
) -> tuple[str, str, bytes]:
    runtime.mkdir()
    committed_run = _fake_run_script().encode("utf-8")
    (runtime / "run.py").write_bytes(committed_run)
    _runtime_manifest(runtime)
    (runtime / ".gitignore").write_text(
        f"{krea_runtime.RUNTIME_IDENTITY_FILENAME}\n",
        encoding="utf-8",
    )
    _git(runtime, "init", "-q")
    _git(runtime, "config", "user.name", "SN56 Test")
    _git(runtime, "config", "user.email", "sn56@example.invalid")
    _git(runtime, "add", ".gitignore", "run.py", krea_runtime.CAPABILITY_MANIFEST_FILENAME)
    _git(runtime, "commit", "-q", "-m", "committed runtime A")
    commit = _git(runtime, "rev-parse", "HEAD^{commit}")
    tree = _git(runtime, "rev-parse", "HEAD^{tree}")
    monkeypatch.setattr(krea_runtime, "OWNED_RUNTIME_COMMIT", commit)
    identity = {
        "schema": 1,
        "runtime_repository": krea_runtime.OWNED_RUNTIME_REPOSITORY,
        "runtime_commit": commit,
        "capability_manifest_sha256": hashlib.sha256(
            (runtime / krea_runtime.CAPABILITY_MANIFEST_FILENAME).read_bytes()
        ).hexdigest(),
    }
    (runtime / krea_runtime.RUNTIME_IDENTITY_FILENAME).write_bytes(
        launcher._canonical_bytes(identity)
    )
    return commit, tree, committed_run


def _arguments(
    *,
    config_path: Path,
    runtime: Path,
    save_root: Path,
    repo_name: str,
    terminal: Path,
    evidence: Path,
) -> list[str]:
    now = datetime.now(timezone.utc)
    rental_started = (now - timedelta(minutes=1)).isoformat().replace(
        "+00:00", "Z"
    )
    rental_ended = (now + timedelta(minutes=5)).isoformat().replace(
        "+00:00", "Z"
    )
    return [
        "--config",
        str(config_path),
        "--runtime-dir",
        str(runtime),
        "--save-root",
        str(save_root),
        "--repo-name",
        repo_name,
        "--task-id",
        "friday-h100-gate",
        "--dataset-size",
        "18",
        "--bundle",
        krea_runtime.LEADER_BUNDLE,
        "--terminal-artifact",
        str(terminal),
        "--output-profile",
        str(evidence / "profile.json"),
        "--output-receipt",
        str(evidence / "receipt.json"),
        "--output-log",
        str(evidence / "training.log"),
        "--output-gate-log",
        str(evidence / "friday-h100-gate.jsonl"),
        "--gate-session-id",
        "week6-friday-h100-gate",
        "--rental-started-at-utc",
        rental_started,
        "--rental-ended-at-utc",
        rental_ended,
        "--timeout-seconds",
        "5",
        "--poll-seconds",
        "0.01",
    ]


def _write_minimal_config(
    path: Path,
    *,
    repo_name: str,
    training_folder: Path,
    planned_steps: int = 600,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "config": {
                    "name": repo_name,
                    "process": [
                        {
                            "training_folder": str(training_folder),
                            "model": {"arch": "krea2"},
                            "train": {"steps": planned_steps},
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _load_exact_pinned_config_loader(tmp_path, monkeypatch):
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "pinned_ai_toolkit"
        / "config.py"
    )
    metadata_path = fixture.with_name("METADATA.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source = fixture.read_bytes()
    assert metadata == {
        "schema": 1,
        "repository": "https://github.com/tuly1/sn56-ai-toolkit-mirror.git",
        "commit": krea_runtime.OWNED_RUNTIME_COMMIT,
        "path": "toolkit/config.py",
        "sha256": "526b26b8017a09974db6135c4990b84317a473f91108da60929db469f3008fe5",
        "retrieved_at_utc": "2026-08-04T00:00:00Z",
    }
    assert hashlib.sha256(source).hexdigest() == metadata["sha256"]

    oyaml_fixture = ModuleType("oyaml")
    oyaml_fixture.SafeLoader = yaml.SafeLoader
    oyaml_fixture.load = yaml.load
    toolkit_fixture = ModuleType("toolkit")
    toolkit_fixture.__path__ = []
    paths_fixture = ModuleType("toolkit.paths")
    paths_fixture.TOOLKIT_ROOT = str(tmp_path / "toolkit-root")
    monkeypatch.setitem(sys.modules, "oyaml", oyaml_fixture)
    monkeypatch.setitem(sys.modules, "toolkit", toolkit_fixture)
    monkeypatch.setitem(sys.modules, "toolkit.paths", paths_fixture)

    specification = importlib.util.spec_from_file_location(
        "sn56_exact_pinned_ai_toolkit_config",
        fixture,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_exact_pinned_loader_accepts_real_yaml_and_rejects_fd_pseudo_path(
    tmp_path, monkeypatch
):
    pinned_config = _load_exact_pinned_config_loader(tmp_path, monkeypatch)
    workspace = tmp_path / "private-supervisor"
    workspace.mkdir(mode=0o700)
    captured = workspace / "captured-config.yaml"
    captured.write_bytes(
        b"job: extension\nconfig:\n  name: pinned-loader-contract\n"
    )

    loaded = pinned_config.get_config(str(captured))

    assert loaded["job"] == "extension"
    assert loaded["config"]["name"] == "pinned-loader-contract"
    descriptor = os.open(captured, os.O_RDONLY)
    try:
        descriptor_path = next(
            f"{root}/{descriptor}"
            for root in ("/proc/self/fd", "/dev/fd")
            if os.path.exists(f"{root}/{descriptor}")
        )
        with pytest.raises(ValueError, match="must be a json or yaml file"):
            pinned_config.get_config(descriptor_path)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("index_flag", ["--assume-unchanged", "--skip-worktree"])
def test_lab_launcher_executes_committed_runtime_a_despite_hidden_worktree_b(
    tmp_path, monkeypatch, index_flag
):
    expected_commit = "a" * 40
    expected_tree = "b" * 40
    monkeypatch.setattr(
        launcher,
        "_git_release_identity",
        lambda: (expected_commit, expected_tree),
    )
    runtime = tmp_path / "owned-runtime"
    runtime_commit, runtime_tree, committed_run = _initialize_runtime_object_store(
        runtime,
        monkeypatch,
    )
    repo_name = "lab-krea"
    save_root = tmp_path / "uploaded-checkpoints" / repo_name
    save_root.parent.mkdir(parents=True)
    config_dir = tmp_path / "lab-input"
    config_dir.mkdir()
    evidence = tmp_path / "lab-evidence"
    spec = ImageSpec.build(
        task_id="friday-h100-gate",
        model="krea/Krea-2-Raw",
        model_type="krea2",
        expected_repo_name=repo_name,
        trigger_word="AetherTest UI",
        dataset_zip=None,
    )
    localized = {
        "save_root": str(save_root),
        "training_folder": str(save_root.parent),
        "config_path": str(config_dir / "task.yaml"),
        "cached_model_dir": str(tmp_path / "model"),
        "cached_zip_path": str(tmp_path / "dataset.zip"),
        "dataset_images_dir": str(tmp_path / "images"),
        "dataset_holdout_dir": str(tmp_path / "holdout"),
    }
    for name, value in localized.items():
        monkeypatch.setattr(
            type(spec),
            name,
            property(lambda _self, fixed=value: fixed),
        )
    monkeypatch.setenv(krea_runtime.BUNDLE_ENV, krea_runtime.LEADER_BUNDLE)
    monkeypatch.setenv(
        krea_runtime.OWNED_KREA_RUNTIME_DIR_ENV,
        str(runtime),
    )
    for poison_key in (
        "PYTHONPATH",
        "LD_PRELOAD",
        "HF_TOKEN",
        "SN56_RELEASE_COMMIT",
    ):
        monkeypatch.setenv(poison_key, "must-not-reach-child")
    accelerator_calls = []

    def live_accelerator(**_kwargs):
        accelerator_calls.append(True)
        return "NVIDIA H100 PCIe|81559-MiB"

    monkeypatch.setattr(
        adaptive_timing,
        "current_accelerator_identity",
        live_accelerator,
    )
    generated = config.build_config(spec, 18, 0.75)
    config_path = Path(spec.config_path)
    config.write_config(generated, str(config_path))
    original_config_payload = config_path.read_bytes()
    original_config_sha256 = hashlib.sha256(original_config_payload).hexdigest()
    object_identity_calls = []
    real_object_identity = launcher._git_object_store_identity

    def observed_object_identity(path):
        object_identity_calls.append(True)
        if len(object_identity_calls) == 1:
            incompatible_b = b"config:\n  name: incompatible-B\n"
            config_path.write_bytes(incompatible_b)
            assert config_path.read_bytes() == incompatible_b
            config_path.write_bytes(original_config_payload)
        return real_object_identity(path)

    monkeypatch.setattr(
        launcher,
        "_git_object_store_identity",
        observed_object_identity,
    )
    planned_steps = generated["config"]["process"][0]["train"]["steps"]
    _git(runtime, "update-index", index_flag, "run.py")
    hidden_b_marker = tmp_path / f"hidden-b-{index_flag.removeprefix('--')}"
    (runtime / "run.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(hidden_b_marker)!r}).write_text('B executed')\n"
        "raise SystemExit(91)\n",
        encoding="utf-8",
    )
    assert _git(runtime, "status", "--porcelain=v1", "--untracked-files=all") == ""
    terminal = save_root / f"{repo_name}.safetensors"
    args = launcher.build_parser().parse_args(
        _arguments(
            config_path=config_path,
            runtime=runtime,
            save_root=save_root,
            repo_name=repo_name,
            terminal=terminal,
            evidence=evidence,
        )
    )

    receipt = launcher.run_lab(args)

    sealed_config_path = Path(receipt["config"]["executed"]["path"])
    raw_path = Path(str(sealed_config_path) + ".effective-runtime.json")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    profile = json.loads((evidence / "profile.json").read_text(encoding="utf-8"))
    persisted_receipt = json.loads(
        (evidence / "receipt.json").read_text(encoding="utf-8")
    )
    gate_lines = (evidence / "friday-h100-gate.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(gate_lines) == 1
    gate_event = json.loads(gate_lines[0])
    assert receipt == persisted_receipt
    assert receipt["state"] == "PASS"
    assert receipt["evidence_scope"] == "lab-only"
    assert raw["schema"] == krea_runtime.EFFECTIVE_RUNTIME_SCHEMA
    assert raw["lifecycle"] == "terminal"
    assert raw["generated_config_sha256"] == original_config_sha256
    assert raw["first_checkpoint_observation"]["checkpoint_step"] == 200
    assert raw["training_completion_observation"]["completed_steps"] == planned_steps
    assert profile["schema"] == adaptive_timing.PROFILE_SCHEMA
    assert profile["evidence_scope"] == "lab-only"
    assert profile["provenance"]["accelerator_identity_evidence"] == (
        "operator-attested"
    )
    assert receipt["terminal_artifact"]["sha256"] == raw[
        "training_completion_observation"
    ]["artifact_sha256"]
    assert config_path.read_bytes() == original_config_payload
    assert sealed_config_path.read_bytes() == original_config_payload
    assert sealed_config_path.suffix == ".yaml"
    assert sealed_config_path.stat().st_mode & 0o777 == 0o400
    assert sealed_config_path.parent.stat().st_mode & 0o777 == 0o700
    assert not Path(str(config_path) + ".effective-runtime.json").exists()
    assert receipt["config"] == {
        "original": {
            "path": str(config_path),
            "captured_sha256": original_config_sha256,
        },
        "executed": {
            "path": str(sealed_config_path),
            "sha256": original_config_sha256,
        },
    }
    expected_event_fields = {
        "event",
        "gate_session_id",
        "source_run_id",
        "rental_started_at_utc",
        "rental_ended_at_utc",
        "training_started_at_utc",
        "raw_record_produced_at_utc",
        "profile_produced_at_utc",
        "sealed_at_utc",
        "profile_file_sha256",
        "raw_record_file_sha256",
        "terminal_artifact_file_sha256",
        "profile_semantic_sha256",
        "raw_record_semantic_sha256",
        "forge_commit",
        "release_tree",
        "certificate_scope",
        "bundle_id",
        "bundle_sha256",
        "model_type",
        "current_dataset_size",
        "dataset_regime",
        "accelerator_identity",
    }
    assert set(gate_event) == expected_event_fields
    assert gate_event["event"] == (
        "sn56.week6.friday-h100-timing-evidence-sealed.v2"
    )
    assert gate_event["certificate_scope"] == "toolkit-krea-only"
    assert gate_event["profile_file_sha256"] == hashlib.sha256(
        (evidence / "profile.json").read_bytes()
    ).hexdigest()
    assert gate_event["raw_record_file_sha256"] == hashlib.sha256(
        raw_path.read_bytes()
    ).hexdigest()
    assert gate_event["terminal_artifact_file_sha256"] == hashlib.sha256(
        terminal.read_bytes()
    ).hexdigest()
    assert gate_event["profile_semantic_sha256"] == profile["profile_sha256"]
    assert gate_event["raw_record_semantic_sha256"] == raw["record_sha256"]
    assert gate_event["bundle_id"] == krea_runtime.LEADER_BUNDLE
    assert gate_event["bundle_sha256"] == krea_runtime.bundle_contract_sha256(
        krea_runtime.LEADER_BUNDLE
    )
    assert gate_event["current_dataset_size"] == 18
    assert gate_event["dataset_regime"] == adaptive_timing.dataset_regime(18)
    assert gate_event["forge_commit"] == expected_commit
    assert gate_event["release_tree"] == expected_tree
    times = [
        _parse_utc(gate_event[key])
        for key in (
            "rental_started_at_utc",
            "training_started_at_utc",
            "raw_record_produced_at_utc",
            "profile_produced_at_utc",
            "sealed_at_utc",
            "rental_ended_at_utc",
        )
    ]
    assert times == sorted(times)
    assert receipt["forge"] == {
        "commit": expected_commit,
        "tree": expected_tree,
    }
    assert receipt["friday_h100_gate_log"]["sha256"] == hashlib.sha256(
        (evidence / "friday-h100-gate.jsonl").read_bytes()
    ).hexdigest()
    assert receipt["friday_h100_gate_log"]["event_sha256"] == receipt[
        "friday_h100_gate_log"
    ]["sha256"]
    runtime_receipt = receipt["runtime"]
    assert runtime_receipt["commit"] == runtime_commit
    assert runtime_receipt["tree"] == runtime_tree
    assert runtime_receipt["object_store_path"] == str(runtime)
    assert len(runtime_receipt["archive_sha256"]) == 64
    assert len(runtime_receipt["materialized_file_manifest_sha256"]) == 64
    assert (
        Path(runtime_receipt["materialized_path"]) / "run.py"
    ).read_bytes() == committed_run
    training_log = (evidence / "training.log").read_text(encoding="utf-8")
    assert f"EXECUTED_CONFIG_SHA256={original_config_sha256}" in training_log
    assert "COMMITTED_RUNTIME_A_EXECUTED=1" in training_log
    assert not hidden_b_marker.exists()
    assert len(accelerator_calls) == 2
    assert len(object_identity_calls) == 3
    assert not list(evidence.glob("*.tmp"))


def test_lab_launcher_rejects_evidence_inside_upload_tree(tmp_path):
    save_root = tmp_path / "uploaded"
    evidence = save_root / "evidence"
    args = launcher.build_parser().parse_args(
        _arguments(
            config_path=tmp_path / "config.yaml",
            runtime=tmp_path / "runtime",
            save_root=save_root,
            repo_name="lab-krea",
            terminal=save_root / "lab-krea.safetensors",
            evidence=evidence,
        )
    )

    with pytest.raises(launcher.LabTimingError, match="outside the uploaded"):
        launcher.run_lab(args)

    assert not evidence.exists()


def test_lab_launcher_rejects_nonempty_save_root_without_touching_sentinel(
    tmp_path,
):
    save_root = tmp_path / "uploaded" / "lab-krea"
    save_root.mkdir(parents=True)
    sentinel = save_root / "existing.safetensors"
    sentinel_payload = b"do-not-quarantine-or-rewrite"
    sentinel.write_bytes(sentinel_payload)
    evidence = tmp_path / "evidence"
    args = launcher.build_parser().parse_args(
        _arguments(
            config_path=tmp_path / "config.yaml",
            runtime=tmp_path / "runtime",
            save_root=save_root,
            repo_name="lab-krea",
            terminal=save_root / "lab-krea.safetensors",
            evidence=evidence,
        )
    )

    with pytest.raises(launcher.LabTimingError, match="must be absent"):
        launcher.run_lab(args)

    assert sentinel.read_bytes() == sentinel_payload
    assert list(save_root.iterdir()) == [sentinel]
    assert not evidence.exists()


def test_lab_launcher_yaml_output_mismatch_aborts_before_save_root_mutation(
    tmp_path,
):
    repo_name = "lab-krea"
    save_root = tmp_path / "uploaded" / repo_name
    save_root.parent.mkdir(parents=True)
    config_path = tmp_path / "input" / "config.yaml"
    _write_minimal_config(
        config_path,
        repo_name=repo_name,
        training_folder=tmp_path / "wrong-output-parent",
    )
    evidence = tmp_path / "evidence"
    args = launcher.build_parser().parse_args(
        _arguments(
            config_path=config_path,
            runtime=tmp_path / "runtime",
            save_root=save_root,
            repo_name=repo_name,
            terminal=save_root / f"{repo_name}.safetensors",
            evidence=evidence,
        )
    )

    with pytest.raises(launcher.LabTimingError, match="output differs"):
        launcher.run_lab(args)

    assert not save_root.exists()
    assert not Path(str(config_path) + ".effective-runtime.json").exists()
    assert not evidence.exists()


def test_lab_launcher_rejects_non_git_runtime_object_store_pre_mutation(tmp_path):
    repo_name = "lab-krea"
    save_root = tmp_path / "uploaded" / repo_name
    save_root.parent.mkdir(parents=True)
    config_path = tmp_path / "input" / "config.yaml"
    _write_minimal_config(
        config_path,
        repo_name=repo_name,
        training_folder=save_root.parent,
    )
    selected_runtime = tmp_path / "selected-runtime"
    selected_runtime.mkdir()
    evidence = tmp_path / "evidence"
    args = launcher.build_parser().parse_args(
        _arguments(
            config_path=config_path,
            runtime=selected_runtime,
            save_root=save_root,
            repo_name=repo_name,
            terminal=save_root / f"{repo_name}.safetensors",
            evidence=evidence,
        )
    )

    with pytest.raises(launcher.LabTimingError, match="object-store verification"):
        launcher.run_lab(args)

    assert not save_root.exists()
    assert not Path(str(config_path) + ".effective-runtime.json").exists()
    assert not evidence.exists()


def test_lab_launcher_persistent_sealed_config_tamper_aborts_before_popen(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        launcher,
        "_git_release_identity",
        lambda: ("a" * 40, "b" * 40),
    )
    repo_name = "lab-krea"
    save_root = tmp_path / "uploaded" / repo_name
    save_root.parent.mkdir(parents=True)
    config_path = tmp_path / "input" / "config.yaml"
    _write_minimal_config(
        config_path,
        repo_name=repo_name,
        training_folder=save_root.parent,
    )
    original_payload = config_path.read_bytes()
    runtime = tmp_path / "runtime"
    _initialize_runtime_object_store(runtime, monkeypatch)
    evidence = tmp_path / "evidence"
    supervised_workspace = evidence / ".receipt.json.supervised"
    object_identity_calls = []
    real_object_identity = launcher._git_object_store_identity

    def tampering_object_identity(path):
        object_identity_calls.append(True)
        result = real_object_identity(path)
        if len(object_identity_calls) == 3:
            sealed_config = next(supervised_workspace.glob("*.yaml"))
            sealed_config.chmod(0o600)
            sealed_config.write_bytes(b"persistent incompatible config B\n")
        return result

    monkeypatch.setattr(
        launcher,
        "_git_object_store_identity",
        tampering_object_identity,
    )
    monkeypatch.setattr(
        krea_runtime,
        "emit_effective_runtime_record",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        adaptive_timing,
        "current_accelerator_identity",
        lambda **_kwargs: "NVIDIA H100 PCIe|81559-MiB",
    )
    args = launcher.build_parser().parse_args(
        _arguments(
            config_path=config_path,
            runtime=runtime,
            save_root=save_root,
            repo_name=repo_name,
            terminal=save_root / f"{repo_name}.safetensors",
            evidence=evidence,
        )
    )

    with pytest.raises(launcher.LabTimingError, match="changed before process"):
        launcher.run_lab(args)

    sealed_config = next(supervised_workspace.glob("*.yaml"))
    assert config_path.read_bytes() == original_payload
    assert sealed_config.read_bytes() == b"persistent incompatible config B\n"
    assert len(object_identity_calls) == 3
    assert not (save_root / f"{repo_name}.safetensors").exists()
    assert not (evidence / "profile.json").exists()
    assert not (evidence / "friday-h100-gate.jsonl").exists()


def test_lab_launcher_rejects_out_of_window_gate_before_mutation(tmp_path):
    save_root = tmp_path / "uploaded" / "lab-krea"
    save_root.parent.mkdir(parents=True)
    evidence = tmp_path / "evidence"
    arguments = _arguments(
        config_path=tmp_path / "config.yaml",
        runtime=tmp_path / "runtime",
        save_root=save_root,
        repo_name="lab-krea",
        terminal=save_root / "lab-krea.safetensors",
        evidence=evidence,
    )
    now = datetime.now(timezone.utc)
    started = (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    ended = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    arguments[arguments.index("--rental-started-at-utc") + 1] = started
    arguments[arguments.index("--rental-ended-at-utc") + 1] = ended
    args = launcher.build_parser().parse_args(arguments)

    with pytest.raises(launcher.LabTimingError, match="outside the declared"):
        launcher.run_lab(args)

    assert not save_root.exists()
    assert not evidence.exists()


def test_lab_launcher_rejects_gate_log_collision_before_mutation(tmp_path):
    save_root = tmp_path / "uploaded" / "lab-krea"
    save_root.parent.mkdir(parents=True)
    evidence = tmp_path / "evidence"
    args = launcher.build_parser().parse_args(
        _arguments(
            config_path=tmp_path / "config.yaml",
            runtime=tmp_path / "runtime",
            save_root=save_root,
            repo_name="lab-krea",
            terminal=save_root / "lab-krea.safetensors",
            evidence=evidence,
        )
    )
    args.output_gate_log = args.output_profile

    with pytest.raises(launcher.LabTimingError, match="must be distinct"):
        launcher.run_lab(args)

    assert not save_root.exists()
    assert not evidence.exists()


def test_git_release_identity_rejects_dirty_or_untracked_execution_surface(
    monkeypatch,
):
    commit = "a" * 40
    tree = "b" * 40

    def fake_git(command, **_kwargs):
        if "status" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="?? untracked-supervisor.py\n",
                stderr="",
            )
        value = tree if "HEAD^{tree}" in command else commit
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"{value}\n",
            stderr="",
        )

    monkeypatch.setattr(launcher.subprocess, "run", fake_git)

    with pytest.raises(launcher.LabTimingError, match="not fully clean"):
        launcher._git_release_identity()


def test_git_release_identity_uses_absolute_git_and_fixed_environment(monkeypatch):
    commit = "a" * 40
    tree = "b" * 40
    observations = []

    def fake_git(command, **kwargs):
        observations.append((command, kwargs.get("env")))
        if "status" in command:
            value = ""
        else:
            value = tree if "HEAD^{tree}" in command else commit
        return subprocess.CompletedProcess(command, 0, stdout=f"{value}\n", stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", fake_git)

    assert launcher._git_release_identity() == (commit, tree)
    assert observations
    for command, environment in observations:
        assert command[0] == launcher._ABSOLUTE_GIT
        assert command[1] == "--no-replace-objects"
        assert environment == launcher._git_environment()


def test_lab_launcher_help_is_explicit_and_deterministic():
    first = launcher.build_parser().format_help()
    second = launcher.build_parser().format_help()

    assert first == second
    assert "LAB ONLY" in first
    assert "raw-record/profile evidence" in first
    assert "--runtime-dir" in first
    assert "--output-profile" in first
    assert "--output-gate-log" in first
    assert "--rental-started-at-utc" in first
    assert launcher.sys.dont_write_bytecode is True


def test_production_package_never_imports_lab_launcher():
    repository = Path(__file__).resolve().parents[1]
    for source in (repository / "forge").rglob("*.py"):
        assert "run_krea_timing_lab" not in source.read_text(encoding="utf-8")
