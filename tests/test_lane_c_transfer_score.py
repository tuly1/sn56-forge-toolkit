"""Focused contracts for the one-GPU Lane-C reduced-2 runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


_ROOT = Path(__file__).parents[1]
_CALIBRATION = _ROOT / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))

import lane_c_transfer_score as runner  # noqa: E402


def _write_manifest(path: Path, candidates: list[Path]) -> None:
    rows = [
        {
            "id": f"candidate-{index}",
            "path": str(candidate),
            "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        }
        for index, candidate in enumerate(candidates)
    ]
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "sn56-lane-c-transfer-candidates",
                "candidates": rows,
            }
        ),
        encoding="utf-8",
    )


def _surface(tmp_path: Path, *, candidates: int = 2) -> tuple[SimpleNamespace, list[Path]]:
    dataset = tmp_path / "dataset"
    comfy = tmp_path / "ComfyUI"
    god = tmp_path / "G.O.D"
    output_parent = tmp_path / "outputs"
    for directory in (dataset, god, output_parent, comfy / "models" / "loras"):
        directory.mkdir(parents=True)
    (comfy / "models" / "loras" / "put_loras_here").write_bytes(b"")
    order = tmp_path / "order.txt"
    order.write_text("image-b.png\nimage-a.png\n", encoding="utf-8")
    candidate_paths = []
    for index in range(candidates):
        candidate = tmp_path / f"source-{index}.safetensors"
        candidate.write_bytes(f"candidate-{index}".encode())
        candidate_paths.append(candidate)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, candidate_paths)
    args = SimpleNamespace(
        candidate_manifest=str(manifest),
        dataset=str(dataset),
        expected_dataset_sha256="d" * 64,
        expected_image_order=str(order),
        comfy_root=str(comfy),
        comfy_python=sys.executable,
        driver_python=sys.executable,
        god_root=str(god),
        output_dir=str(output_parent / "campaign"),
        expected_god_commit="a" * 40,
        expected_comfy_commit="b" * 40,
        expected_tooling_commit="c" * 40,
        expected_candidate_count=candidates,
        gpu_index=0,
        port=8188,
        base_name="krea2_raw_fp8_scaled.safetensors",
        startup_timeout_s=300.0,
        evaluation_timeout_s=3600.0,
        shutdown_timeout_s=20.0,
    )
    return args, candidate_paths


def _result_from_command(
    command: list[str],
    *,
    seed_mode: str = "reduced-2",
    dataset_sha256: str = "d" * 64,
) -> dict:
    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    images = [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "--expected-image"
    ]
    candidate_path = Path(value("--candidate-path"))
    candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    return {
        "schema": 2,
        "evaluator": "god_krea2_img2img_exact",
        "candidate": candidate_path.name,
        "candidate_sha256": candidate_sha,
        "candidate_bytes": candidate_path.stat().st_size,
        "staged_candidate_sha256": candidate_sha,
        "comfy_lora_name": candidate_path.name,
        "model_type": "krea2",
        "dataset": str(Path(value("--dataset")).resolve()),
        "dataset_sha256": dataset_sha256,
        "image_count": len(images),
        "scored_rows": [{"image": name} for name in images],
        "generations": 2,
        "validator_default_generations": 5,
        "seed_mode": seed_mode,
        "master_seed": 42,
        "seeds": [2746317213, 1181241943],
        "weighted_loss": 0.125,
        "direction": "min",
        "elapsed_s": 1.0,
        "runtime": {
            "comfy_history": {"prompt_count": len(images) * 4},
            "physical_gpu_index": int(value("--gpu-index")),
            "cuda_visible_devices": value("--gpu-index"),
        },
    }


def test_command_forces_reduced_two_gpu_and_exact_image_order(tmp_path: Path) -> None:
    args, candidates = _surface(tmp_path, candidates=1)
    lora_root = Path(args.comfy_root) / "models" / "loras"
    manifest_rows, _ = runner._load_candidates(
        Path(args.candidate_manifest), expected_count=1
    )
    staged, _ = runner._stage_candidate(manifest_rows[0], lora_root)
    try:
        command = runner._command(
            args,
            evaluator_script=_CALIBRATION / "evaluate_krea_local.py",
            staged_candidate=staged,
            result_path=tmp_path / "result.json",
            comfy_log=tmp_path / "comfy.log",
            expected_order=["image-b.png", "image-a.png"],
        )
    finally:
        staged.unlink()
    assert command[command.index("--generations") + 1] == "2"
    assert command[command.index("--gpu-index") + 1] == "0"
    observed_order = [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "--expected-image"
    ]
    assert observed_order == ["image-b.png", "image-a.png"]
    assert candidates[0].name not in command


def test_candidate_manifest_and_staging_fail_closed_on_bad_identity(
    tmp_path: Path,
) -> None:
    args, candidates = _surface(tmp_path, candidates=1)
    rows, _ = runner._load_candidates(Path(args.candidate_manifest), expected_count=1)
    candidates[0].write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="source SHA-256 mismatch"):
        runner._stage_candidate(
            rows[0], Path(args.comfy_root) / "models" / "loras"
        )
    runner._assert_lora_root_empty(Path(args.comfy_root) / "models" / "loras")

    with pytest.raises(ValueError, match="exactly 2"):
        runner._load_candidates(Path(args.candidate_manifest), expected_count=2)


def test_runner_scores_all_candidates_and_publishes_only_complete_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _surface(tmp_path)
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)

        def value(flag: str) -> str:
            return command[command.index(flag) + 1]

        Path(value("--comfy-log")).write_text("fresh comfy\n", encoding="utf-8")
        Path(value("--output")).write_text(
            json.dumps(_result_from_command(command)), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.run(args)
    assert result["status"] == "complete"
    assert result["candidate_count"] == 2
    assert result["image_count"] == 2
    assert result["expected_prompt_count_per_candidate"] == 8
    assert result["seed_mode"] == "reduced-2"
    assert len(commands) == 2
    assert (Path(args.output_dir) / "aggregate.json").is_file()
    runner._assert_lora_root_empty(Path(args.comfy_root) / "models" / "loras")
    with pytest.raises(FileExistsError, match="stale output"):
        runner.run(args)


def test_runner_rejects_wrong_seed_mode_and_never_publishes_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _surface(tmp_path, candidates=1)

    def fake_run(command, **kwargs):
        def value(flag: str) -> str:
            return command[command.index(flag) + 1]

        Path(value("--comfy-log")).write_text("fresh comfy\n", encoding="utf-8")
        Path(value("--output")).write_text(
            json.dumps(_result_from_command(command, seed_mode="validator-exact-5")),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="seed_mode"):
        runner.run(args)
    assert not (Path(args.output_dir) / "aggregate.json").exists()
    runner._assert_lora_root_empty(Path(args.comfy_root) / "models" / "loras")


def test_runner_rejects_wrong_dataset_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _surface(tmp_path, candidates=1)

    def fake_run(command, **kwargs):
        def value(flag: str) -> str:
            return command[command.index(flag) + 1]

        Path(value("--comfy-log")).write_text("fresh comfy\n", encoding="utf-8")
        Path(value("--output")).write_text(
            json.dumps(_result_from_command(command, dataset_sha256="e" * 64)),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="dataset_sha256"):
        runner.run(args)
    assert not (Path(args.output_dir) / "aggregate.json").exists()
    runner._assert_lora_root_empty(Path(args.comfy_root) / "models" / "loras")


def test_expected_order_rejects_blank_duplicate_and_path_entries(tmp_path: Path) -> None:
    for content in ("a.png\n\nb.png\n", "a.png\na.png\n", "../a.png\n"):
        path = tmp_path / hashlib.sha256(content.encode()).hexdigest()
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="unique basenames"):
            runner._load_expected_order(path)
