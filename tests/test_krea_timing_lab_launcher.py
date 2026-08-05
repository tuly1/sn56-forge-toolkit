from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from forge import adaptive_timing, config, krea_runtime
from forge.data.schema import ImageSpec
from ops.calibration import run_krea_timing_lab as launcher


def _runtime_attestations(runtime: Path) -> None:
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
    identity = {
        "schema": 1,
        "runtime_repository": krea_runtime.OWNED_RUNTIME_REPOSITORY,
        "runtime_commit": krea_runtime.OWNED_RUNTIME_COMMIT,
        "capability_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    }
    (runtime / krea_runtime.RUNTIME_IDENTITY_FILENAME).write_text(
        json.dumps(identity),
        encoding="utf-8",
    )


def _fake_run_script(
    save_root: Path,
    repo_name: str,
    planned_steps: int,
    *,
    expected_config_sha256: str,
) -> str:
    checkpoint = save_root / f"{repo_name}_000000200.safetensors"
    terminal = save_root / f"{repo_name}.safetensors"
    return f'''import hashlib, json, struct, sys, time
from pathlib import Path
config_path = Path(sys.argv[1])
if not str(config_path).startswith(("/proc/self/fd/", "/dev/fd/")):
    raise SystemExit("launcher did not execute an inherited config descriptor")
if hashlib.sha256(config_path.read_bytes()).hexdigest() != {expected_config_sha256!r}:
    raise SystemExit("launcher did not execute captured config A")
def write(path, step):
    metadata = {{"training_info": json.dumps({{"step": step, "epoch": 1}})}}
    header = json.dumps({{"__metadata__": metadata, "weight": {{"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}}}).encode("utf-8")
    Path(path).write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.0))
time.sleep(0.05)
write({str(checkpoint)!r}, 200)
time.sleep(0.08)
write({str(terminal)!r}, {planned_steps})
print("{planned_steps}/{planned_steps} loss=0.1", flush=True)
'''


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


def test_lab_launcher_executes_captured_config_a_across_original_a_b_a_swap(
    tmp_path, monkeypatch
):
    expected_commit = "a" * 40
    expected_tree = "b" * 40
    monkeypatch.setattr(
        launcher,
        "_git_release_identity",
        lambda: (expected_commit, expected_tree),
    )
    runtime = tmp_path / "owned-runtime"
    runtime.mkdir()
    _runtime_attestations(runtime)
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
    runtime_verifications = []

    def verify_runtime(*_args, **_kwargs):
        runtime_verifications.append(True)
        if len(runtime_verifications) == 1:
            incompatible_b = b"config:\n  name: incompatible-B\n"
            config_path.write_bytes(incompatible_b)
            assert config_path.read_bytes() == incompatible_b
            config_path.write_bytes(original_config_payload)
        return str(runtime)

    monkeypatch.setattr(
        krea_runtime,
        "verify_selected_runtime",
        verify_runtime,
    )
    planned_steps = generated["config"]["process"][0]["train"]["steps"]
    sealed_config_path = evidence / ".receipt.json.executed-config.yaml"
    (runtime / "run.py").write_text(
        _fake_run_script(
            save_root,
            repo_name,
            planned_steps,
            expected_config_sha256=original_config_sha256,
        ),
        encoding="utf-8",
    )
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
    assert sealed_config_path.stat().st_mode & 0o777 == 0o400
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
    assert len(accelerator_calls) == 2
    assert len(runtime_verifications) == 2
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


def test_lab_launcher_attested_executed_runtime_mismatch_aborts_pre_mutation(
    tmp_path, monkeypatch
):
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
    different_runtime = tmp_path / "different-runtime"
    selected_runtime.mkdir()
    different_runtime.mkdir()
    monkeypatch.setattr(
        krea_runtime,
        "verify_selected_runtime",
        lambda *_args, **_kwargs: str(different_runtime),
    )
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

    with pytest.raises(launcher.LabTimingError, match="attested runtime differs"):
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
    runtime.mkdir()
    _runtime_attestations(runtime)
    evidence = tmp_path / "evidence"
    sealed_config = evidence / ".receipt.json.executed-config.yaml"
    verification_calls = []

    def verify_runtime(*_args, **_kwargs):
        verification_calls.append(True)
        if len(verification_calls) == 2:
            sealed_config.chmod(0o600)
            sealed_config.write_bytes(b"persistent incompatible config B\n")
        return str(runtime)

    monkeypatch.setattr(
        krea_runtime,
        "verify_selected_runtime",
        verify_runtime,
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
    popen_calls = []

    def forbidden_popen(*_args, **_kwargs):
        popen_calls.append(True)
        raise AssertionError("Popen must not receive a tampered sealed config")

    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden_popen)
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

    assert config_path.read_bytes() == original_payload
    assert sealed_config.read_bytes() == b"persistent incompatible config B\n"
    assert len(verification_calls) == 2
    assert popen_calls == []
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
