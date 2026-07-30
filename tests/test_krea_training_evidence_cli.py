"""CPU-only operational tests for stage-three Krea evidence and zero controls."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest


_ROOT = Path(__file__).parents[1]
_CALIBRATION = _ROOT / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))

import krea_training_evidence as evidence  # noqa: E402


def _canonical_file(path: Path, value: object) -> str:
    payload = evidence.krea_provenance.canonical_bytes(value) + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _reseal(path: Path, value: dict, digest_key: str | None = None) -> str:
    if digest_key is not None:
        body = {key: item for key, item in value.items() if key != digest_key}
        value[digest_key] = evidence.krea_provenance.canonical_sha256(body)
    return _canonical_file(path, value)


def _safetensors(path: Path, marker: int = 1) -> None:
    header = {
        "__metadata__": {"format": "pt", "producer": "cpu-contract-test"},
        "lora_unet_block.lora_A.weight": {
            "dtype": "F16",
            "shape": [2, 3],
            "data_offsets": [0, 12],
        },
        "lora_unet_block.lora_B.weight": {
            "dtype": "BF16",
            "shape": [3, 2],
            "data_offsets": [12, 24],
        },
    }
    raw_header = json.dumps(
        header, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    # Real safetensors writers pad the header so the tensor region is aligned.
    raw_header += b" " * ((8 - len(raw_header) % 8) % 8)
    path.write_bytes(
        struct.pack("<Q", len(raw_header))
        + raw_header
        + bytes((marker + index) % 256 for index in range(24))
    )


def _recipe_fields() -> dict:
    values = {
        "planned_steps": 100,
        "submitted_step": None,
        "learning_rate": 0.0001,
        "rank": 128,
        "alpha": 128,
        "optimizer": "adamw8bit",
        "optimizer_parameters": {"weight_decay": 0.0001},
        "loss": "mse",
        "guidance": {"enabled": True, "scale": 2},
        "scheduler": "flowmatch",
        "dropout": 0.05,
        "gradient_accumulation": 1,
        "effective_batch": 1,
        "ema": {"enabled": False, "decay": 0.99},
        "save_cadence": 25,
        "selector": None,
    }
    return {name: {"effective_value": value} for name, value in values.items()}


@pytest.fixture
def approved_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    fixture = {
        "manifest_sha256": "1" * 64,
        "training_dataset_identity": {"sha256": "2" * 64},
        "evaluation_dataset_identity": {"sha256": "3" * 64},
        "training_rows": [{"row_id": "train-001", "content_sha256": "4" * 64}],
        "training_archive": {"sha256": "5" * 64, "bytes": 1234},
    }
    base_model = {
        "model_id": "krea/Krea-2-Raw",
        "revision": "6" * 40,
        "training_identity_sha256": "7" * 64,
        "evaluation_assets": {
            name: {
                "canonical_path": f"/models/{name}.safetensors",
                "sha256": digit * 64,
                "bytes": 1024,
            }
            for name, digit in (
                ("diffusion_model", "8"),
                ("text_encoder", "9"),
                ("vae", "a"),
            )
        },
    }
    budget_plan = {"hard_budget_s": 3600, "save_every": 25}
    profile = {
        "schema": 2,
        "profile_sha256": "b" * 64,
        "execution_envelope": {"data_parallel_replicas": 1},
    }
    plan = {
        "arm_id": "KREA-K2",
        "task_id": "fixture-c1",
        "expected_repo_name": "krea-k2",
        "plan_sha256": "c" * 64,
        "budget_plan": budget_plan,
        "budget_plan_sha256": evidence.krea_provenance.canonical_sha256(budget_plan),
        "schedule": {"planned_steps": 100, "candidate_steps": [25, 50, 100]},
        "execution_recipe": {"fields": _recipe_fields()},
        "execution_envelope_sha256": "d" * 64,
        "base_model": base_model,
    }
    approval = {"approval_sha256": "e" * 64, "reviewer_identity": "Human Reviewer"}
    resolved = {
        "fixture": fixture,
        "host_execution_manifest": {"host_execution_identity_sha256": "f" * 64},
        "throughput_profile": profile,
        "execution_recipe": plan["execution_recipe"],
    }
    plan_path = tmp_path / "execution-plan.json"
    approval_path = tmp_path / "execution-approval.json"
    plan_file_sha = _canonical_file(plan_path, plan)
    approval_file_sha = _canonical_file(approval_path, approval)

    def validate_plan(value):
        assert value == plan
        return copy.deepcopy(resolved)

    def validate_approval(value, *, plan, approval_path=None):
        assert value == approval
        assert plan == approved_plan
        assert approval_path is not None
        return value

    approved_plan = copy.deepcopy(plan)
    monkeypatch.setattr(evidence.krea_execution_plan, "validate_plan", validate_plan)
    monkeypatch.setattr(
        evidence.krea_execution_plan, "validate_approval", validate_approval
    )
    monkeypatch.setattr(
        evidence,
        "_training_identity",
        lambda _path, *, fixture: (
            fixture["training_dataset_identity"],
            fixture["training_rows"],
        ),
    )

    candidates_dir = tmp_path / "raw-candidates"
    candidates_dir.mkdir()
    candidates = []
    for step, marker in ((25, 1), (50, 2)):
        path = candidates_dir / f"krea-k2_{step}.safetensors"
        _safetensors(path, marker)
        candidates.append(path)
    final = candidates_dir / "krea-k2.safetensors"
    last = candidates_dir / "last.safetensors"
    _safetensors(final, 3)
    last.write_bytes(final.read_bytes())
    candidates.extend((final, last))
    scope = {
        path.name: evidence.krea_provenance.file_sha256(path) for path in candidates
    }
    telemetry = {
        "schema": 2,
        "events": [
            {"name": "toolkit_start"},
            {"name": "toolkit_end", "returncode": 0, "stopped_by_deadline": False},
            {"name": "toolkit_metrics", "last_step": 100},
            {"name": "run_complete"},
        ],
    }
    condition = {
        "schema": 2,
        "kind": "forge-krea2-calibration-run",
        "complete": True,
        "arm_id": plan["arm_id"],
        "task_id": plan["task_id"],
        "expected_repo_name": plan["expected_repo_name"],
        "model": "krea/Krea-2-Raw",
        "execution_plan_sha256": plan["plan_sha256"],
        "execution_plan_file_sha256": plan_file_sha,
        "execution_approval_sha256": approval["approval_sha256"],
        "execution_approval_file_sha256": approval_file_sha,
        "in_task_proxy_selection": {"enabled": False, "reserve_s": 0},
        "budget": {
            "plan": budget_plan,
            "plan_sha256": plan["budget_plan_sha256"],
            "throughput_profile": profile,
        },
        "provenance": {
            "host_execution_manifest_sha256": resolved["host_execution_manifest"][
                "host_execution_identity_sha256"
            ]
        },
        "dataset_after_split": {
            "approved_exact_evaluation_sha256": fixture["evaluation_dataset_identity"][
                "sha256"
            ]
        },
        "attempt": {"planned_steps": 100},
        "resolved_config": {
            "config": {
                "process": [
                    {
                        "network": {"linear": 128, "linear_alpha": 128},
                        "save": {"save_every": 25},
                        "datasets": [{"caption_dropout_rate": 0.05}],
                        "train": {
                            "steps": 100,
                            "lr": 0.0001,
                            "optimizer": "adamw8bit",
                            "optimizer_params": {"weight_decay": 0.0001},
                            "loss_type": "mse",
                            "noise_scheduler": "flowmatch",
                            "batch_size": 1,
                            "gradient_accumulation": 1,
                            "do_differential_guidance": True,
                            "differential_guidance_scale": 2,
                            "ema_config": {"use_ema": False, "ema_decay": 0.99},
                        },
                    }
                ]
            }
        },
        "current_scope_candidates": scope,
        "artifacts": {
            "candidate_sha256": {
                name: digest
                for name, digest in scope.items()
                if name != "last.safetensors"
            },
            "last_sha256": scope["last.safetensors"],
        },
        "telemetry": telemetry,
    }
    condition_path = tmp_path / "condition.json"
    _canonical_file(condition_path, condition)
    return {
        "root": tmp_path,
        "plan": plan,
        "resolved": resolved,
        "plan_path": plan_path,
        "approval_path": approval_path,
        "condition_path": condition_path,
        "training_dir": train_dir,
        "candidates": candidates,
        "output": tmp_path / "evidence",
    }


def _emit(approved_run: dict) -> Path:
    code = evidence.main(
        [
            "run-evidence",
            "--condition-record",
            str(approved_run["condition_path"]),
            "--execution-plan",
            str(approved_run["plan_path"]),
            "--execution-approval",
            str(approved_run["approval_path"]),
            "--training-dir",
            str(approved_run["training_dir"]),
            *sum(
                (["--candidate", str(path)] for path in approved_run["candidates"]),
                [],
            ),
            "--output-dir",
            str(approved_run["output"]),
        ]
    )
    assert code == 0
    return approved_run["output"] / "bundle.json"


def test_real_safetensors_layout_and_duplicate_header_rejected(tmp_path: Path):
    artifact = tmp_path / "real.safetensors"
    _safetensors(artifact)
    identity = evidence._safetensors_identity(artifact)
    assert identity["tensor_count"] == 2
    assert identity["tensor_data_bytes"] == 24

    duplicate = tmp_path / "duplicate.safetensors"
    raw_header = (
        b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},'
        b'"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
    )
    duplicate.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + b"x")
    with pytest.raises(ValueError, match="header is invalid"):
        evidence._safetensors_identity(duplicate)


def test_run_cli_emits_exhaustive_grid_and_validates(approved_run: dict, capsys):
    bundle_path = _emit(approved_run)
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["candidate_count"] == 3
    assert emitted["status"] == "emitted"
    bundle = evidence.validate_run_evidence(bundle_path)
    assert [row["candidate_id"] for row in bundle["candidate_bindings"]] == [
        row["candidate_id"]
        for row in sorted(
            bundle["candidate_bindings"],
            key=lambda row: int(row["candidate_id"].split("-")[1]),
        )
    ]
    assert evidence.main(["validate-run-evidence", "--bundle", str(bundle_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "valid"
    before = bundle_path.read_bytes()
    with pytest.raises(FileExistsError, match="refusing existing evidence output"):
        evidence.emit_run_evidence(
            condition_record_path=approved_run["condition_path"],
            execution_plan_path=approved_run["plan_path"],
            execution_approval_path=approved_run["approval_path"],
            candidate_paths=approved_run["candidates"],
            training_dir=approved_run["training_dir"],
            output_dir=approved_run["output"],
        )
    assert bundle_path.read_bytes() == before


@pytest.mark.parametrize("drop_index", [0, 1, 2, 3])
def test_run_cli_rejects_every_incomplete_candidate_scope(
    approved_run: dict, drop_index: int, capsys
):
    candidates = [
        path
        for index, path in enumerate(approved_run["candidates"])
        if index != drop_index
    ]
    output = approved_run["root"] / f"incomplete-{drop_index}"
    argv = [
        "run-evidence",
        "--condition-record",
        str(approved_run["condition_path"]),
        "--execution-plan",
        str(approved_run["plan_path"]),
        "--execution-approval",
        str(approved_run["approval_path"]),
        "--training-dir",
        str(approved_run["training_dir"]),
    ]
    for path in candidates:
        argv.extend(("--candidate", str(path)))
    argv.extend(("--output-dir", str(output)))
    assert evidence.main(argv) == 1
    assert "do not exhaust current scope" in capsys.readouterr().err
    assert not output.exists()


def test_run_validation_rejects_tampered_candidate_bytes(
    approved_run: dict,
):
    bundle_path = _emit(approved_run)
    bundle = json.loads(bundle_path.read_text())
    binding_path = Path(bundle["candidate_bindings"][0]["binding"]["path"])
    binding = json.loads(binding_path.read_text())
    artifact = Path(binding["candidate"]["path"])
    artifact.chmod(0o600)
    raw = bytearray(artifact.read_bytes())
    raw[-1] ^= 1
    artifact.write_bytes(raw)
    with pytest.raises(ValueError, match="bytes/layout differ"):
        evidence.validate_run_evidence(bundle_path)


@pytest.mark.parametrize("field", ["fixture", "base", "run"])
def test_zero_validation_rejects_wrong_fixture_base_or_run_binding(
    approved_run: dict, field: str
):
    bundle_path = _emit(approved_run)
    bundle = json.loads(bundle_path.read_text())
    final_binding = Path(bundle["candidate_bindings"][-1]["binding"]["path"])
    artifact = approved_run["root"] / f"zero-{field}.safetensors"
    manifest_path = approved_run["root"] / f"zero-{field}.json"
    assert (
        evidence.main(
            [
                "zero-control",
                "--template-candidate-binding",
                str(final_binding),
                "--output-artifact",
                str(artifact),
                "--output-manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    manifest = json.loads(manifest_path.read_text())
    if field == "fixture":
        manifest["evaluation_dataset_sha256"] = "0" * 64
    elif field == "base":
        manifest["base_model"]["revision"] = "0" * 40
    else:
        manifest["run_completion"] = {
            "path": str(approved_run["root"] / "other-run.json"),
            "sha256": "0" * 64,
        }
    _reseal(manifest_path, manifest, "manifest_sha256")
    with pytest.raises(ValueError, match="approved run/base/fixture"):
        evidence.validate_zero_control(manifest, artifact_path=artifact)


def test_zero_cli_generates_exact_zero_layout_is_exclusive_and_detects_tamper(
    approved_run: dict, capsys
):
    bundle_path = _emit(approved_run)
    capsys.readouterr()
    bundle = json.loads(bundle_path.read_text())
    final_binding = Path(bundle["candidate_bindings"][-1]["binding"]["path"])
    artifact = approved_run["root"] / "zero.safetensors"
    manifest_path = approved_run["root"] / "zero.json"
    argv = [
        "zero-control",
        "--template-candidate-binding",
        str(final_binding),
        "--output-artifact",
        str(artifact),
        "--output-manifest",
        str(manifest_path),
    ]
    assert evidence.main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "emitted"
    manifest = json.loads(manifest_path.read_text())
    header, data = evidence._read_safetensors(artifact)
    template = json.loads(final_binding.read_text())["candidate"]
    assert not any(data)
    assert evidence._tensor_layout(header) == manifest["tensor_layout"]
    assert (
        manifest["tensor_layout_sha256"]
        == template["safetensors"]["tensor_layout_sha256"]
    )
    assert (
        evidence.main(["validate-zero-control", "--manifest", str(manifest_path)]) == 0
    )
    capsys.readouterr()

    artifact_before = artifact.read_bytes()
    assert evidence.main(argv) == 1
    assert "refusing existing zero-control output" in capsys.readouterr().err
    assert artifact.read_bytes() == artifact_before

    raw = bytearray(artifact_before)
    raw[-1] = 1
    artifact.write_bytes(raw)
    assert (
        evidence.main(["validate-zero-control", "--manifest", str(manifest_path)]) == 1
    )
    assert "artifact binding mismatch" in capsys.readouterr().err


def test_tampered_run_record_is_rejected(approved_run: dict):
    bundle_path = _emit(approved_run)
    bundle = json.loads(bundle_path.read_text())
    binding = json.loads(
        Path(bundle["candidate_bindings"][0]["binding"]["path"]).read_text()
    )
    run_record = Path(binding["run_record"]["path"])
    run_record.write_bytes(run_record.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evidence.validate_run_evidence(bundle_path)


def test_cli_process_exit_codes_are_machine_checkable(tmp_path: Path):
    script = _CALIBRATION / "krea_training_evidence.py"
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True
    )
    assert help_result.returncode == 0
    parse_failure = subprocess.run(
        [sys.executable, str(script), "run-evidence"], capture_output=True, text=True
    )
    assert parse_failure.returncode == 2
    runtime_failure = subprocess.run(
        [
            sys.executable,
            str(script),
            "validate-run-evidence",
            "--bundle",
            str(tmp_path / "missing.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert runtime_failure.returncode == 1
    assert runtime_failure.stderr.startswith("ERROR:")
