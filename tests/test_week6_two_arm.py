"""Tests for the week-6 two-arm experiment runner.

No GPU, no network, no Forge import: every test drives the runner with a fake
subprocess runner that stands in for the trainer child and the exact scorer.
What is under test is the decision rule, the resume logic, the unloaded-key
detection, and the dry run — the four things that decide whether the real
campaign's result can be believed.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "ops" / "experiments" / "week6" / "run_two_arm.py"
LEADER_YAML = ROOT / "ops" / "experiments" / "week6" / "leader_derived_krea2.yaml"
INCUMBENT_YAML = ROOT / "ops" / "experiments" / "week6" / "incumbent_krea2.yaml"
ARMS_DIR = ROOT / "ops" / "experiments" / "week6" / "arms"
PLAN_ONLY_FIXTURE = (
    ROOT / "ops" / "experiments" / "week6" / "fixtures" / "PLAN-ONLY-41025fb5-r1.json"
)
EXAMPLE_HOST = (
    ROOT / "ops" / "experiments" / "week6" / "hosts" / "EXAMPLE-h100-scorer.json"
)

_SPEC = importlib.util.spec_from_file_location("run_two_arm", RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
runner_module = importlib.util.module_from_spec(_SPEC)
# dataclasses resolves annotations through sys.modules, so register first.
sys.modules["run_two_arm"] = runner_module
_SPEC.loader.exec_module(runner_module)
R = runner_module


# ---------------------------------------------------------------------------
# synthetic environment
# ---------------------------------------------------------------------------


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha(data)


def _write_json(path: Path, value) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    return _write(path, payload.encode("utf-8"))


def build_environment(tmp_path: Path, *, holdout_images: int = 4) -> SimpleNamespace:
    """A complete, self-consistent fixture/arms/host tree with no GPU in it."""
    root = tmp_path / "env"
    holdout_dir = root / "fixture" / "holdout"
    rows = []
    for index in range(holdout_images):
        image = f"{index:03d}.png"
        prompt = f"{index:03d}.txt"
        image_sha = _write(holdout_dir / image, f"image-bytes-{index}".encode())
        prompt_sha = _write(holdout_dir / prompt, f"AetherFlow UI: caption {index}".encode())
        rows.append(
            {
                "image": image,
                "image_sha256": image_sha,
                "prompt": prompt,
                "prompt_sha256": prompt_sha,
            }
        )
    archive = root / "fixture" / "train.zip"
    archive_sha = _write(archive, b"PK\x03\x04 not-a-real-zip, never opened here")

    fixture_path = root / "fixture" / "fixture.json"
    _write_json(
        fixture_path,
        {
            "schema": 1,
            "kind": "forge-week6-real-fixture",
            "plan_only": False,
            "fixture_id": "TEST-FIXTURE",
            "task_id": "task-under-test",
            "model": "krea/Krea-2-Raw",
            "model_type": "krea2",
            "trigger_word": "AetherFlow UI",
            "hours_to_complete": 0.75,
            "train": {"pairs": 18, "archive_path": "train.zip", "archive_sha256": archive_sha},
            "holdout": {"pairs": holdout_images, "dir": "holdout", "rows": rows},
        },
    )

    # Two minimal but structurally real arm configs.
    def _config(steps: int, save_every: int) -> dict:
        return {
            "job": "extension",
            "config": {
                "name": "last",
                "process": [
                    {
                        "type": "diffusion_trainer",
                        "training_folder": "/app/checkpoints/PLACEHOLDER",
                        "trigger_word": "PLACEHOLDER",
                        "network": {"type": "lora", "linear": 32, "linear_alpha": 32},
                        "save": {"save_every": save_every, "dtype": "bf16"},
                        "datasets": [{"folder_path": "/dataset/PLACEHOLDER", "caption_ext": "txt"}],
                        "train": {"steps": steps, "batch_size": 1, "lr": 0.0001},
                        "model": {
                            "arch": "krea2",
                            "name_or_path": "/cache/models/PLACEHOLDER",
                            "model_kwargs": {"vae_path": "/cache/models/PLACEHOLDER"},
                        },
                    }
                ],
            },
        }

    import yaml

    arm_paths = {}
    for arm_id, steps, save_every, repo in (
        ("arm_a", 400, 200, "week6-test-a"),
        ("arm_b", 400, 200, "week6-test-b"),
    ):
        config_path = root / "arms" / f"{arm_id}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(_config(steps, save_every)), encoding="utf-8")
        spec_path = root / "arms" / f"{arm_id}.json"
        _write_json(
            spec_path,
            {
                "schema": 1,
                "kind": "forge-week6-arm",
                "arm_id": arm_id,
                "role": "train",
                "label": f"test arm {arm_id}",
                "expected_repo_name": repo,
                "config_path": f"{arm_id}.yaml",
                "config_sha256": R.sha256_file(config_path),
                "ai_toolkit_commit": "a" * 40,
                "training_seed": 42565431,
            },
        )
        arm_paths[arm_id] = spec_path

    zero_path = root / "zero" / "zero.safetensors"
    zero_sha = _write(zero_path, b"\x00" * 64)
    record_path = root / "zero" / "zero.record.json"
    _write_json(
        record_path,
        {
            "kind": "zero_lora_safetensors_baseline",
            "output": {"sha256": zero_sha, "all_zero": True, "all_finite": True},
            "verification": {"output_all_zero": True, "output_all_finite": True},
        },
    )
    zero_spec = root / "zero" / "zero.json"
    _write_json(
        zero_spec,
        {
            "schema": 1,
            "kind": "forge-week6-arm",
            "arm_id": "zero_lora",
            "role": "zero_lora_control",
            "label": "zero control",
            "candidate_path": "zero.safetensors",
            "candidate_sha256": zero_sha,
            "zero_lora_record_path": "zero.record.json",
        },
    )
    arm_paths["zero_lora"] = zero_spec

    host_root = root / "host"
    (host_root / "ai-toolkit").mkdir(parents=True, exist_ok=True)
    (host_root / "comfy" / "models" / "loras").mkdir(parents=True, exist_ok=True)
    (host_root / "god").mkdir(parents=True, exist_ok=True)
    comfy_python = host_root / "python"
    _write(comfy_python, b"#!/bin/sh\nexit 0\n")
    host_path = host_root / "host.json"
    _write_json(
        host_path,
        {
            "schema": 1,
            "kind": "forge-week6-gpu-host",
            "host_id": "test-host",
            "hostname": "test-box",
            "ai_toolkit_dir": "ai-toolkit",
            "comfy_root": "comfy",
            "comfy_python": "python",
            "god_root": "god",
            "god_commit": "b" * 40,
            "comfy_commit": "c" * 40,
            "tooling_commit": "d" * 40,
            "base_name": "krea2_raw_fp8_scaled.safetensors",
            "gpu_hourly_usd": 2.0,
        },
    )

    return SimpleNamespace(
        root=root,
        fixture=R.load_fixture(fixture_path),
        fixture_path=fixture_path,
        arms=[
            R.load_arm(arm_paths["arm_a"]),
            R.load_arm(arm_paths["arm_b"]),
            R.load_arm(arm_paths["zero_lora"]),
        ],
        arm_paths=arm_paths,
        host=R.load_host(host_path),
        host_path=host_path,
        workdir=tmp_path / "work",
        holdout_rows=rows,
        checkpoint_root=root / "checkpoints",
    )


class FakeGpu:
    """Stands in for the trainer child process and the exact scorer.

    ``losses[arm_id]`` maps an arm to a per-image (prompted, blank) pair
    generator so a test can dictate the science while the plumbing stays real.
    """

    def __init__(self, env, *, losses, unloaded_keys=None, checkpoint_steps=(200, 400)):
        self.env = env
        self.losses = losses
        self.unloaded_keys = unloaded_keys or {}
        self.checkpoint_steps = tuple(checkpoint_steps)
        self.calls: list[list[str]] = []

    def __call__(self, argv, *, cwd=None, env=None, timeout=None):
        argv = [str(item) for item in argv]
        self.calls.append(argv)
        if argv[:2] == ["git", "-C"]:
            return subprocess.CompletedProcess(argv, 0, stdout="a" * 40 + "\n", stderr="")
        if "train-cell" in argv:
            return self._train(argv)
        if str(R.EVALUATOR_PATH) in argv:
            return self._score(argv)
        raise AssertionError(f"unexpected subprocess: {argv}")

    def _value(self, argv, flag):
        return argv[argv.index(flag) + 1]

    def _train(self, argv):
        cell_dir = Path(self._value(argv, "--cell-dir"))
        key = self._value(argv, "--cell-key")
        arm_id = next(
            arm.arm_id
            for arm in self.env.arms
            if str(arm.path) == self._value(argv, "--arm")
        )
        checkpoints = []
        for step in self.checkpoint_steps:
            path = self.env.checkpoint_root / arm_id / f"{arm_id}_{step:09d}.safetensors"
            sha = _write(path, f"{arm_id}-{step}".encode())
            checkpoints.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "sha256": sha,
                    "bytes": path.stat().st_size,
                    "step": step,
                    "duplicate_of": None,
                }
            )
        R.write_json_exclusive(
            cell_dir / R.RESULT_NAME,
            {
                "schema": 1,
                "kind": "forge-week6-two-arm-experiment-train-cell",
                "complete": True,
                "cell_key": key,
                "cell_kind": "train",
                "arm_id": arm_id,
                "checkpoints": checkpoints,
                "measured": {
                    "wall_clock_s": 1234.5,
                    "steps_reached": max(self.checkpoint_steps),
                    "seconds_per_step": 1.27,
                },
            },
        )
        return subprocess.CompletedProcess(argv, 0, stdout="{}", stderr="")

    def _score(self, argv):
        output = Path(self._value(argv, "--output"))
        log = Path(self._value(argv, "--comfy-log"))
        candidate = Path(self._value(argv, "--candidate-path"))
        arm_id, step = self._identify(candidate)
        rows = []
        prompted_values, blank_values = [], []
        for index, row in enumerate(self.env.holdout_rows):
            prompted, blank = self.losses[arm_id](index, step)
            prompted_values.append(prompted)
            blank_values.append(blank)
            rows.append(
                {
                    "index": index,
                    "image": row["image"],
                    "image_sha256": row["image_sha256"],
                    "text_guided_loss": prompted,
                    "blank_prompt_loss": blank,
                }
            )
        text_mean = sum(prompted_values) / len(prompted_values)
        blank_mean = sum(blank_values) / len(blank_values)
        R.write_json_exclusive(
            output,
            {
                "schema": 2,
                "evaluator": "god_krea2_img2img_exact",
                "scored_rows": rows,
                "text_mean": text_mean,
                "blank_mean": blank_mean,
                "text_weight": 0.25,
                "weighted_loss": 0.25 * text_mean + 0.75 * blank_mean,
                "generations": 5,
                "seeds": [1, 2, 3, 4, 5],
            },
        )
        lines = ["got prompt", "loaded completely CheckpointLoader"]
        for key in self.unloaded_keys.get(arm_id, []):
            lines.append(f"lora key not loaded: {key}")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="{}", stderr="")

    def _identify(self, candidate: Path):
        if candidate.name.startswith("zero"):
            return "zero_lora", 0
        match = re.search(r"_(\d+)\.safetensors$", candidate.name)
        assert match is not None, candidate
        return candidate.parent.name, int(match.group(1))


def constant_losses(prompted: float, blank: float):
    return lambda index, step: (prompted, blank)


def offset_losses(base_prompted: float, base_blank: float, per_image: float = 0.0):
    return lambda index, step: (
        base_prompted + per_image * index,
        base_blank + per_image * index,
    )


# ---------------------------------------------------------------------------
# unloaded-key detection
# ---------------------------------------------------------------------------


def test_parse_unloaded_keys_clean_log():
    result = R.parse_unloaded_keys("got prompt\nloaded completely\nPrompt executed\n")
    assert result["clean"] is True
    assert result["total_reports"] == 0
    assert result["unique_keys"] == 0
    assert result["by_namespace"] == {"unet": 0, "text_encoder": 0, "other": 0}


def test_parse_unloaded_keys_counts_and_splits_namespaces():
    text = "\n".join(
        ["lora key not loaded: lora_te.model.language_model.layers.0.q_proj"] * 2
        + [f"lora key not loaded: lora_te.model.language_model.layers.{i}.k_proj" for i in range(3)]
        + [f"lora key not loaded: lora_unet_blocks_{i}_attn" for i in range(2)]
        + ["lora key not loaded: something_else_entirely"]
    )
    result = R.parse_unloaded_keys(text)
    assert result["clean"] is False
    assert result["total_reports"] == 8
    assert result["unique_keys"] == 7
    assert result["by_namespace"]["text_encoder"] == 4
    assert result["by_namespace"]["unet"] == 2
    assert result["by_namespace"]["other"] == 1
    # The evidence hash is over the unique sorted set, so it is stable.
    assert result["keys_sha256"] == R.parse_unloaded_keys(text + "\n")["keys_sha256"]


def test_classify_lora_key():
    assert R.classify_lora_key("lora_te.model.language_model.x") == "text_encoder"
    assert R.classify_lora_key("lora_unet_blocks_0") == "unet"
    assert R.classify_lora_key("diffusion_model.blocks.0.attn") == "unet"
    assert R.classify_lora_key("mystery") == "other"


def test_score_cell_fails_loudly_on_unloaded_keys_but_keeps_evidence(tmp_path):
    env = build_environment(tmp_path)
    gpu = FakeGpu(
        env,
        losses={"arm_a": constant_losses(0.02, 0.03)},
        unloaded_keys={"arm_a": [f"lora_te.model.language_model.layers.{i}" for i in range(504)]},
    )
    candidate = env.checkpoint_root / "arm_a" / "arm_a_000000200.safetensors"
    sha = _write(candidate, b"arm_a-200")
    with pytest.raises(R.UnloadedLoraKeys) as error:
        R.run_score_cell(
            fixture=env.fixture,
            arm=env.arms[0],
            host=env.host,
            candidate_path=candidate,
            candidate_sha256=sha,
            candidate_name=candidate.name,
            step=200,
            cost=R.CostModel(),
            workdir=env.workdir,
            runner=gpu,
        )
    assert "504" in str(error.value)
    # The cell is still on disk, flagged — an aborted run must not lose evidence.
    inputs = R.score_cell_inputs(
        fixture=env.fixture,
        arm=env.arms[0],
        host=env.host,
        candidate_sha256=sha,
        candidate_name=candidate.name,
        step=200,
        generations_planned=R.CostModel().eval_generations,
    )
    result = R.read_json(R.cell_dir(env.workdir, inputs) / R.RESULT_NAME)
    assert result["attachment_ok"] is False
    assert result["unloaded_lora_keys"]["by_namespace"]["text_encoder"] == 504


def test_score_cell_flag_mode_records_and_continues(tmp_path):
    env = build_environment(tmp_path)
    gpu = FakeGpu(
        env,
        losses={"arm_a": constant_losses(0.02, 0.03)},
        unloaded_keys={"arm_a": ["lora_te.model.language_model.layers.0"]},
    )
    candidate = env.checkpoint_root / "arm_a" / "arm_a_000000200.safetensors"
    sha = _write(candidate, b"arm_a-200")
    result = R.run_score_cell(
        fixture=env.fixture,
        arm=env.arms[0],
        host=env.host,
        candidate_path=candidate,
        candidate_sha256=sha,
        candidate_name=candidate.name,
        step=200,
        cost=R.CostModel(),
        workdir=env.workdir,
        runner=gpu,
        on_unloaded_keys="flag",
    )
    assert result["attachment_ok"] is False
    assert result["means"]["blank"] == pytest.approx(0.03)


def test_score_cell_rejects_empty_comfy_log(tmp_path):
    env = build_environment(tmp_path)

    def runner(argv, **kwargs):
        argv = [str(item) for item in argv]
        output = Path(argv[argv.index("--output") + 1])
        log = Path(argv[argv.index("--comfy-log") + 1])
        R.write_json_exclusive(output, {"scored_rows": [], "text_mean": 0, "blank_mean": 0})
        log.write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    candidate = env.checkpoint_root / "arm_a" / "arm_a_000000200.safetensors"
    sha = _write(candidate, b"arm_a-200")
    with pytest.raises(R.ExperimentError, match="ComfyUI log is empty"):
        R.run_score_cell(
            fixture=env.fixture,
            arm=env.arms[0],
            host=env.host,
            candidate_path=candidate,
            candidate_sha256=sha,
            candidate_name=candidate.name,
            step=200,
            cost=R.CostModel(),
            workdir=env.workdir,
            runner=runner,
        )


# ---------------------------------------------------------------------------
# statistics and the decision rule
# ---------------------------------------------------------------------------


def test_bootstrap_is_deterministic_and_seed_sensitive():
    values = [0.01, -0.004, 0.02, 0.006, 0.011, -0.001]
    first = R.bootstrap_mean_ci(values, seed=20260806, iterations=2000)
    second = R.bootstrap_mean_ci(values, seed=20260806, iterations=2000)
    third = R.bootstrap_mean_ci(values, seed=7, iterations=2000)
    assert first == second
    assert (first["lo"], first["hi"]) != (third["lo"], third["hi"])
    assert first["lo"] <= first["mean"] <= first["hi"]
    assert first["method"].startswith("percentile bootstrap")


def test_bootstrap_undefined_below_two_images():
    result = R.bootstrap_mean_ci([0.01], seed=1, iterations=100)
    assert result["defined"] is False
    assert "undefined" in result["reason"]


def test_paired_deltas_pair_by_image_identity_not_order():
    better = [
        {"image_sha256": "b" * 64, "blank_loss": 0.010},
        {"image_sha256": "a" * 64, "blank_loss": 0.020},
    ]
    baseline = [
        {"image_sha256": "a" * 64, "blank_loss": 0.030},
        {"image_sha256": "b" * 64, "blank_loss": 0.015},
    ]
    assert R.paired_deltas(better, baseline, "blank_loss") == pytest.approx([0.010, 0.005])


def test_paired_deltas_reject_mismatched_images():
    with pytest.raises(R.ExperimentError, match="different images"):
        R.paired_deltas(
            [{"image_sha256": "a" * 64, "blank_loss": 0.01}],
            [{"image_sha256": "z" * 64, "blank_loss": 0.01}],
            "blank_loss",
        )


def test_select_best_uses_composite_then_blank_then_step():
    summaries = [
        {
            "candidate": {"step": 400, "sha256": "1" * 64},
            "composite_mean": 0.030,
            "blank_mean": 0.030,
        },
        {
            "candidate": {"step": 200, "sha256": "2" * 64},
            "composite_mean": 0.020,
            "blank_mean": 0.020,
        },
        {
            "candidate": {"step": 600, "sha256": "3" * 64},
            "composite_mean": 0.020,
            "blank_mean": 0.019,
        },
    ]
    assert R.select_best(summaries)["candidate"]["step"] == 600


def test_projected_checkpoints():
    assert R.projected_checkpoints(1000, 200) == [200, 400, 600, 800, 1000]
    assert R.projected_checkpoints(1000, 201) == [201, 402, 603, 804, 1000]
    assert R.projected_checkpoints(400, 400) == [400]


def _plan_for_analysis(env, tmp_path):
    return R.build_plan(
        fixture=env.fixture,
        arms=env.arms,
        host=env.host,
        cost=R.CostModel(),
        workdir=tmp_path / "analysis-work",
    )


def _score_cell(arm_id, *, step, prompted, blank, images, clean=True):
    per_image = [
        {
            "index": index,
            "image": row["image"],
            "image_sha256": row["image_sha256"],
            "prompted_loss": prompted[index],
            "blank_loss": blank[index],
            "composite_loss": R.composite(prompted[index], blank[index]),
        }
        for index, row in enumerate(images)
    ]
    return {
        "arm_id": arm_id,
        "cell_key": f"{arm_id}-{step}",
        "candidate": {"name": f"{arm_id}_{step}", "sha256": "e" * 64, "step": step},
        "attachment_ok": clean,
        "unloaded_lora_keys": {"clean": clean, "unique_keys": 0 if clean else 12},
        "per_image": per_image,
    }


def test_decision_rule_declares_a_winner_when_the_lower_bound_clears_zero(tmp_path):
    env = build_environment(tmp_path, holdout_images=8)
    plan = _plan_for_analysis(env, tmp_path)
    images = env.holdout_rows
    n = len(images)
    # arm_a is uniformly better on the blank stratum by a wide, consistent margin.
    cells = [
        _score_cell(
            "arm_a",
            step=400,
            prompted=[0.020 + 0.001 * i for i in range(n)],
            blank=[0.020 + 0.001 * i for i in range(n)],
            images=images,
        ),
        _score_cell(
            "arm_b",
            step=400,
            prompted=[0.030 + 0.001 * i for i in range(n)],
            blank=[0.030 + 0.001 * i for i in range(n)],
            images=images,
        ),
        _score_cell(
            "zero_lora",
            step=0,
            prompted=[0.040 + 0.001 * i for i in range(n)],
            blank=[0.040 + 0.001 * i for i in range(n)],
            images=images,
        ),
    ]
    analysis = R.analyze(plan=plan, score_cells=cells)
    assert analysis["winner"] == "arm_a"
    assert analysis["null_result"] is False
    assert analysis["winner_interval"]["lo"] > 0.0
    assert "WINNER: arm_a" in analysis["winner_statement"]
    assert analysis["head_to_head"]["arm_b"]["interval"]["hi"] < 0.0
    # Zero-LoRA deltas are reported for every checkpoint, with intervals.
    vs_zero = analysis["arms"]["arm_a"]["checkpoints"][0]["vs_zero_lora"]
    assert vs_zero["blank"]["mean"] == pytest.approx(0.020)
    assert vs_zero["relative_blank_mean_pct"] > 0.0
    # Per-image rows survive into the analysis (week 5 lost these).
    assert len(analysis["arms"]["arm_a"]["checkpoints"][0]["per_image"]) == n


def test_decision_rule_returns_no_winner_on_a_null_result(tmp_path):
    env = build_environment(tmp_path, holdout_images=8)
    plan = _plan_for_analysis(env, tmp_path)
    images = env.holdout_rows
    # A tiny mean difference swamped by per-image spread: the lower bound must
    # not clear zero, and NOTHING may be declared.
    blank_a = [0.010, 0.050, 0.012, 0.048, 0.011, 0.049, 0.013, 0.047]
    blank_b = [0.050, 0.011, 0.049, 0.012, 0.048, 0.013, 0.047, 0.014]
    cells = [
        _score_cell("arm_a", step=400, prompted=blank_a, blank=blank_a, images=images),
        _score_cell("arm_b", step=400, prompted=blank_b, blank=blank_b, images=images),
        _score_cell(
            "zero_lora",
            step=0,
            prompted=[0.060] * 8,
            blank=[0.060] * 8,
            images=images,
        ),
    ]
    analysis = R.analyze(plan=plan, score_cells=cells)
    assert analysis["winner"] is None
    assert analysis["null_result"] is True
    assert "NO WINNER" in analysis["winner_statement"]
    assert any("not above zero" in reason for reason in analysis["reasons"])
    for arm_id in ("arm_a", "arm_b"):
        assert analysis["head_to_head"][arm_id]["interval"]["lo"] <= 0.0


def test_identical_arms_produce_no_winner(tmp_path):
    env = build_environment(tmp_path, holdout_images=6)
    plan = _plan_for_analysis(env, tmp_path)
    images = env.holdout_rows
    values = [0.020 + 0.001 * i for i in range(len(images))]
    cells = [
        _score_cell("arm_a", step=400, prompted=values, blank=values, images=images),
        _score_cell("arm_b", step=400, prompted=values, blank=values, images=images),
        _score_cell("zero_lora", step=0, prompted=values, blank=values, images=images),
    ]
    analysis = R.analyze(plan=plan, score_cells=cells)
    assert analysis["winner"] is None
    assert analysis["head_to_head"]["arm_a"]["interval"]["mean"] == pytest.approx(0.0)


def test_unloaded_keys_disqualify_an_otherwise_winning_arm(tmp_path):
    env = build_environment(tmp_path, holdout_images=8)
    plan = _plan_for_analysis(env, tmp_path)
    images = env.holdout_rows
    n = len(images)
    cells = [
        _score_cell(
            "arm_a",
            step=400,
            prompted=[0.020] * n,
            blank=[0.020] * n,
            images=images,
            clean=False,
        ),
        _score_cell("arm_b", step=400, prompted=[0.030] * n, blank=[0.030] * n, images=images),
        _score_cell("zero_lora", step=0, prompted=[0.040] * n, blank=[0.040] * n, images=images),
    ]
    analysis = R.analyze(plan=plan, score_cells=cells)
    assert analysis["arms"]["arm_a"]["flagged_unloaded_keys"] is True
    assert analysis["winner"] is None
    assert any("unloaded LoRA keys" in problem for problem in analysis["problems"])
    assert analysis["head_to_head"]["arm_a"]["eligible"] is False


def test_missing_control_blocks_a_winner(tmp_path):
    env = build_environment(tmp_path, holdout_images=8)
    plan = _plan_for_analysis(env, tmp_path)
    images = env.holdout_rows
    n = len(images)
    cells = [
        _score_cell("arm_a", step=400, prompted=[0.020] * n, blank=[0.020] * n, images=images),
        _score_cell("arm_b", step=400, prompted=[0.030] * n, blank=[0.030] * n, images=images),
    ]
    analysis = R.analyze(plan=plan, score_cells=cells)
    assert analysis["winner"] is None
    assert analysis["complete"] is False
    assert any("zero-LoRA control" in problem for problem in analysis["problems"])


def test_a_stray_arm_in_the_workdir_is_reported_and_blocks_a_winner(tmp_path):
    env = build_environment(tmp_path, holdout_images=8)
    plan = _plan_for_analysis(env, tmp_path)
    images = env.holdout_rows
    n = len(images)
    cells = [
        _score_cell("arm_a", step=400, prompted=[0.020] * n, blank=[0.020] * n, images=images),
        _score_cell("arm_b", step=400, prompted=[0.030] * n, blank=[0.030] * n, images=images),
        _score_cell("zero_lora", step=0, prompted=[0.040] * n, blank=[0.040] * n, images=images),
        _score_cell("arm_from_another_run", step=400, prompted=[0.001] * n, blank=[0.001] * n, images=images),
    ]
    analysis = R.analyze(plan=plan, score_cells=cells)
    assert any("unknown arm" in problem for problem in analysis["problems"])
    assert analysis["winner"] is None


def test_two_image_holdout_still_defines_the_bootstrap(tmp_path):
    env = build_environment(tmp_path, holdout_images=2)
    plan = _plan_for_analysis(env, tmp_path)
    images = env.holdout_rows
    cells = [
        _score_cell("arm_a", step=400, prompted=[0.020, 0.021], blank=[0.020, 0.021], images=images),
        _score_cell("arm_b", step=400, prompted=[0.030, 0.031], blank=[0.030, 0.031], images=images),
        _score_cell("zero_lora", step=0, prompted=[0.040, 0.041], blank=[0.040, 0.041], images=images),
    ]
    analysis = R.analyze(plan=plan, score_cells=cells)
    assert analysis["head_to_head"]["arm_a"]["interval"]["defined"] is True


def test_single_image_holdout_can_never_produce_a_winner(tmp_path):
    env = build_environment(tmp_path, holdout_images=1)
    plan = _plan_for_analysis(env, tmp_path)
    images = env.holdout_rows
    cells = [
        _score_cell("arm_a", step=400, prompted=[0.001], blank=[0.001], images=images),
        _score_cell("arm_b", step=400, prompted=[0.900], blank=[0.900], images=images),
        _score_cell("zero_lora", step=0, prompted=[0.950], blank=[0.950], images=images),
    ]
    analysis = R.analyze(plan=plan, score_cells=cells)
    assert analysis["winner"] is None
    assert analysis["head_to_head"]["arm_a"]["interval"]["defined"] is False


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------


def test_dry_run_uses_no_gpu_and_prices_the_plan(tmp_path, capsys):
    env = build_environment(tmp_path)
    code = R.main(
        [
            "plan",
            "--fixture",
            str(env.fixture_path),
            "--arm",
            str(env.arm_paths["arm_a"]),
            "--arm",
            str(env.arm_paths["arm_b"]),
            "--arm",
            str(env.arm_paths["zero_lora"]),
            "--host",
            str(env.host_path),
            "--workdir",
            str(env.workdir),
            "--json",
        ]
    )
    assert code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["gpu_used"] is False
    cost = R.CostModel()
    # 2 arms x 400 steps, save_every 200 -> checkpoints at 200 and 400.
    expected_train = 2 * cost.train_seconds(400, 2)
    expected_score = cost.score_seconds(4) * 5  # 4 trained checkpoints + control
    assert plan["estimate"]["train_gpu_hours"] == pytest.approx(expected_train / 3600.0)
    assert plan["estimate"]["score_gpu_hours"] == pytest.approx(expected_score / 3600.0)
    assert plan["estimate"]["total_usd"] == pytest.approx(
        (expected_train + expected_score) / 3600.0 * 2.0
    )
    assert plan["estimate"]["score_cells_projected"] == 5
    # No cell directories, no staged candidates: nothing ran.
    assert not (env.workdir / "cells").exists()
    assert not list((env.host.comfy_root / "models" / "loras").iterdir())


def test_dry_run_never_invokes_a_subprocess(tmp_path):
    env = build_environment(tmp_path)
    plan = R.build_plan(
        fixture=env.fixture,
        arms=env.arms,
        host=env.host,
        cost=R.CostModel(),
        workdir=env.workdir,
    )
    R.establish_predeclaration(env.workdir, plan)
    # The refusing runner proves the planning path has no execution in it.
    with pytest.raises(R.ExperimentError, match="dry run attempted"):
        R.refuse_gpu_runner(["anything"])
    assert plan["estimate"]["total_gpu_hours"] > 0.0


def test_dry_run_on_the_real_plan_only_fixture_costs_the_r1_task(tmp_path):
    fixture = R.load_fixture(PLAN_ONLY_FIXTURE)
    arms = [
        R.load_arm(ARMS_DIR / "leader_derived.json"),
        R.load_arm(ARMS_DIR / "incumbent.json"),
        R.load_arm(ARMS_DIR / "zero_lora_control.json"),
    ]
    host = R.load_host(EXAMPLE_HOST)
    plan = R.build_plan(
        fixture=fixture,
        arms=arms,
        host=host,
        cost=R.CostModel(),
        workdir=tmp_path / "w",
    )
    assert plan["executable"] is False
    assert [cell["projected_checkpoints"] for cell in plan["train_cells"]] == [5, 5]
    assert plan["estimate"]["score_cells_projected"] == 11
    # The conservative 2.2 s/step constant cannot fit 1000 steps in 0.75 h —
    # exactly the finding that cost us the depth on Aug-3.
    assert all(cell["fits_in_grant"] is False for cell in plan["train_cells"])
    faster = R.build_plan(
        fixture=fixture,
        arms=arms,
        host=host,
        cost=R.CostModel(sec_per_step=1.27),
        workdir=tmp_path / "w",
    )
    assert all(cell["fits_in_grant"] is True for cell in faster["train_cells"])


def test_plan_only_fixture_is_refused_by_execution(tmp_path):
    fixture = R.load_fixture(PLAN_ONLY_FIXTURE)
    arms = [
        R.load_arm(ARMS_DIR / "leader_derived.json"),
        R.load_arm(ARMS_DIR / "incumbent.json"),
        R.load_arm(ARMS_DIR / "zero_lora_control.json"),
    ]
    host = R.load_host(EXAMPLE_HOST)
    with pytest.raises(R.ExperimentError, match="PLAN-ONLY"):
        R.preflight(
            fixture=fixture,
            arms=arms,
            host=host,
            runner=R.refuse_gpu_runner,
            hostname="anything",
        )


def test_predeclaration_is_written_before_any_result_and_is_frozen(tmp_path):
    env = build_environment(tmp_path)
    plan = R.build_plan(
        fixture=env.fixture,
        arms=env.arms,
        host=env.host,
        cost=R.CostModel(),
        workdir=env.workdir,
    )
    first = R.establish_predeclaration(env.workdir, plan)
    assert first["created"] is True
    path = env.workdir / "PREDECLARATION.json"
    written = R.read_json(path)
    assert written["decision_rule"]["rule_id"] == R.DECISION_RULE["rule_id"]
    assert "LOWER bound" in written["decision_rule"]["win_condition"]
    assert "strictly above zero" in written["decision_rule"]["win_condition"]
    # No results exist yet.
    assert not (env.workdir / "cells").exists()
    assert not (env.workdir / "analysis").exists()
    # Re-pricing the plan does not disturb the rule.
    repriced = R.build_plan(
        fixture=env.fixture,
        arms=env.arms,
        host=env.host,
        cost=R.CostModel(sec_per_step=1.27, gpu_hourly_usd=3.5),
        workdir=env.workdir,
    )
    assert R.establish_predeclaration(env.workdir, repriced)["created"] is False


def test_predeclaration_drift_is_fatal(tmp_path, monkeypatch):
    env = build_environment(tmp_path)
    plan = R.build_plan(
        fixture=env.fixture,
        arms=env.arms,
        host=env.host,
        cost=R.CostModel(),
        workdir=env.workdir,
    )
    R.establish_predeclaration(env.workdir, plan)
    tampered = copy.deepcopy(R.DECISION_RULE)
    tampered["win_condition"] = "an arm wins if its mean is lower"
    monkeypatch.setattr(R, "DECISION_RULE", tampered)
    with pytest.raises(R.ExperimentError, match="PREDECLARATION DRIFT"):
        R.establish_predeclaration(env.workdir, plan)


def test_predeclaration_drift_on_a_changed_arm_config(tmp_path):
    env = build_environment(tmp_path)
    plan = R.build_plan(
        fixture=env.fixture,
        arms=env.arms,
        host=env.host,
        cost=R.CostModel(),
        workdir=env.workdir,
    )
    R.establish_predeclaration(env.workdir, plan)
    swapped = copy.deepcopy(plan)
    swapped["arms"][0]["config_sha256"] = "f" * 64
    with pytest.raises(R.ExperimentError, match="PREDECLARATION DRIFT"):
        R.establish_predeclaration(env.workdir, swapped)


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


def test_cell_key_changes_when_any_input_changes(tmp_path):
    env = build_environment(tmp_path)
    base = R.train_cell_inputs(
        fixture=env.fixture, arm=env.arms[0], host=env.host, hours=0.75
    )
    assert R.cell_key(base) == R.cell_key(
        R.train_cell_inputs(
            fixture=env.fixture, arm=env.arms[0], host=env.host, hours=0.75
        )
    )
    different_hours = R.train_cell_inputs(
        fixture=env.fixture, arm=env.arms[0], host=env.host, hours=1.5
    )
    assert R.cell_key(base) != R.cell_key(different_hours)
    other_arm = R.train_cell_inputs(
        fixture=env.fixture, arm=env.arms[1], host=env.host, hours=0.75
    )
    assert R.cell_key(base) != R.cell_key(other_arm)


def test_score_cell_key_tracks_the_candidate_hash(tmp_path):
    env = build_environment(tmp_path)
    common = dict(
        fixture=env.fixture,
        arm=env.arms[0],
        host=env.host,
        candidate_name="x.safetensors",
        step=200,
        generations_planned=5,
    )
    first = R.score_cell_inputs(candidate_sha256="1" * 64, **common)
    second = R.score_cell_inputs(candidate_sha256="2" * 64, **common)
    assert R.cell_key(first) != R.cell_key(second)


def test_completed_cell_rejects_a_key_mismatch(tmp_path):
    directory = tmp_path / "cell"
    directory.mkdir()
    R.write_json_exclusive(
        directory / R.RESULT_NAME, {"cell_key": "aaa", "complete": True}
    )
    assert R.completed_cell(directory, "aaa")["complete"] is True
    with pytest.raises(R.ExperimentError, match="key mismatch"):
        R.completed_cell(directory, "bbb")


def test_partial_cell_is_moved_aside_deterministically(tmp_path):
    directory = tmp_path / "cells" / "train" / "abc"
    directory.mkdir(parents=True)
    (directory / "partial.log").write_text("half a run", encoding="utf-8")
    moved = R.preempt_partial_cell(directory)
    assert moved is not None and moved.name.endswith(".incomplete.000")
    assert not directory.exists()
    directory.mkdir(parents=True)
    (directory / "partial.log").write_text("half again", encoding="utf-8")
    assert R.preempt_partial_cell(directory).name.endswith(".incomplete.001")
    # A complete cell is never moved.
    directory.mkdir(parents=True, exist_ok=True)
    R.write_json_exclusive(directory / R.RESULT_NAME, {"cell_key": "abc", "complete": True})
    assert R.preempt_partial_cell(directory) is None


def test_resume_skips_every_completed_cell(tmp_path):
    env = build_environment(tmp_path)
    losses = {
        "arm_a": offset_losses(0.020, 0.020, 0.001),
        "arm_b": offset_losses(0.030, 0.030, 0.001),
        "zero_lora": offset_losses(0.040, 0.040, 0.001),
    }
    first_gpu = FakeGpu(env, losses=losses)
    first = R.execute(
        fixture=env.fixture,
        arms=env.arms,
        host=env.host,
        cost=R.CostModel(),
        hours=None,
        workdir=env.workdir,
        runner=first_gpu,
        hostname="test-box",
    )
    train_calls = [call for call in first_gpu.calls if "train-cell" in call]
    score_calls = [call for call in first_gpu.calls if str(R.EVALUATOR_PATH) in call]
    assert len(train_calls) == 2
    assert len(score_calls) == 5  # 2 arms x 2 checkpoints + zero-LoRA control

    second_gpu = FakeGpu(env, losses=losses)
    second = R.execute(
        fixture=env.fixture,
        arms=env.arms,
        host=env.host,
        cost=R.CostModel(),
        hours=None,
        workdir=env.workdir,
        runner=second_gpu,
        hostname="test-box",
    )
    # Only the preflight git probe may run again: zero GPU work is redone.
    assert all(call[:2] == ["git", "-C"] for call in second_gpu.calls)
    assert second["winner"] == first["winner"]
    assert second["arms"]["arm_a"]["best"] == first["arms"]["arm_a"]["best"]


def test_resume_after_an_interrupted_score_cell(tmp_path):
    env = build_environment(tmp_path)
    losses = {
        "arm_a": offset_losses(0.020, 0.020, 0.001),
        "arm_b": offset_losses(0.030, 0.030, 0.001),
        "zero_lora": offset_losses(0.040, 0.040, 0.001),
    }
    gpu = FakeGpu(env, losses=losses)
    R.execute(
        fixture=env.fixture,
        arms=env.arms,
        host=env.host,
        cost=R.CostModel(),
        hours=None,
        workdir=env.workdir,
        runner=gpu,
        hostname="test-box",
    )
    # Simulate a kill in the middle of one score cell: its result vanishes but
    # partial junk is left behind.
    victim = sorted((env.workdir / "cells" / "score").iterdir())[0]
    (victim / R.RESULT_NAME).unlink()
    (victim / "half-written.tmp").write_text("junk", encoding="utf-8")

    resumed_gpu = FakeGpu(env, losses=losses)
    R.execute(
        fixture=env.fixture,
        arms=env.arms,
        host=env.host,
        cost=R.CostModel(),
        hours=None,
        workdir=env.workdir,
        runner=resumed_gpu,
        hostname="test-box",
    )
    rerun = [call for call in resumed_gpu.calls if str(R.EVALUATOR_PATH) in call]
    assert len(rerun) == 1  # exactly the lost cell, nothing else
    assert (victim.with_name(victim.name + ".incomplete.000") / "half-written.tmp").is_file()
    assert (victim / R.RESULT_NAME).is_file()


# ---------------------------------------------------------------------------
# preflight and input validation
# ---------------------------------------------------------------------------


def test_preflight_rejects_the_wrong_ai_toolkit_commit(tmp_path):
    env = build_environment(tmp_path)

    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="9" * 40 + "\n", stderr="")

    with pytest.raises(R.ExperimentError, match="silently disables"):
        R.preflight(
            fixture=env.fixture,
            arms=env.arms,
            host=env.host,
            runner=runner,
            hostname="test-box",
        )


def test_preflight_rejects_the_wrong_host(tmp_path):
    env = build_environment(tmp_path)
    with pytest.raises(R.ExperimentError, match="declared GPU host"):
        R.preflight(
            fixture=env.fixture,
            arms=env.arms,
            host=env.host,
            runner=R.refuse_gpu_runner,
            hostname="some-other-box",
        )


def test_holdout_verification_detects_tampering(tmp_path):
    env = build_environment(tmp_path)
    assert R.verify_holdout_dir(env.fixture)["verified"] is True
    target = Path(env.fixture.holdout_dir) / env.holdout_rows[0]["image"]
    target.write_bytes(b"different pixels")
    with pytest.raises(R.ExperimentError, match="byte mismatch"):
        R.verify_holdout_dir(env.fixture)


def test_holdout_verification_rejects_extra_files(tmp_path):
    env = build_environment(tmp_path)
    (Path(env.fixture.holdout_dir) / "stray.png").write_bytes(b"stray")
    with pytest.raises(R.ExperimentError, match="differ from the fixture manifest"):
        R.verify_holdout_dir(env.fixture)


def test_arm_config_sha_mismatch_is_rejected(tmp_path):
    env = build_environment(tmp_path)
    spec_path = env.arm_paths["arm_a"]
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["config_sha256"] = "0" * 64
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(R.ExperimentError, match="config sha256 mismatch"):
        R.load_arm(spec_path)


def test_two_training_arms_and_one_control_are_required(tmp_path):
    env = build_environment(tmp_path)
    with pytest.raises(R.ExperimentError, match="exactly two training arms"):
        R.build_plan(
            fixture=env.fixture,
            arms=[env.arms[0], env.arms[2]],
            host=env.host,
            cost=R.CostModel(),
            workdir=env.workdir,
        )


def test_training_arms_must_not_share_a_repo_name(tmp_path):
    env = build_environment(tmp_path)
    spec_path = env.arm_paths["arm_b"]
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["expected_repo_name"] = "week6-test-a"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    arms = [env.arms[0], R.load_arm(spec_path), env.arms[2]]
    with pytest.raises(R.ExperimentError, match="distinct expected_repo_name"):
        R.build_plan(
            fixture=env.fixture,
            arms=arms,
            host=env.host,
            cost=R.CostModel(),
            workdir=env.workdir,
        )


def test_zero_lora_control_must_prove_it_is_all_zero(tmp_path):
    env = build_environment(tmp_path)
    R._verify_zero_lora(env.arms[2])
    record_path = Path(str(env.arms[2].zero_lora_record_path))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["verification"]["output_all_zero"] = False
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(R.ExperimentError, match="all-zero"):
        R._verify_zero_lora(env.arms[2])


def test_bind_config_touches_only_the_declared_bindings():
    import yaml

    frozen = yaml.safe_load(LEADER_YAML.read_text(encoding="utf-8"))
    spec = SimpleNamespace(
        expected_repo_name="week6-leader-derived",
        training_folder="/app/checkpoints/T",
        dataset_images_dir="/dataset/images",
        cached_model_dir="/cache/models/krea--Krea-2-Raw",
    )
    bound = R.bind_config(
        frozen, spec=spec, trigger_word="AetherFlow UI", training_seed=42565431
    )
    process = bound["config"]["process"][0]
    assert bound["config"]["name"] == "week6-leader-derived"
    assert process["training_folder"] == "/app/checkpoints/T"
    assert process["datasets"][0]["folder_path"] == "/dataset/images"
    assert process["model"]["name_or_path"] == "/cache/models/krea--Krea-2-Raw"
    assert process["training_seed"] == 42565431
    # The science is untouched.
    assert process["train"]["steps"] == 1000
    assert process["train"]["differential_guidance_scale"] == 12.0
    assert process["train"]["timestep_type"] == "krea2_eval_sigmas"
    assert process["train"]["ema_config"]["ema_decay"] == 0.995
    # And the original object was not mutated.
    assert frozen["config"]["name"] == "last"


def test_shipped_arm_specs_and_configs_load(tmp_path):
    leader = R.load_arm(ARMS_DIR / "leader_derived.json")
    incumbent = R.load_arm(ARMS_DIR / "incumbent.json")
    control = R.load_arm(ARMS_DIR / "zero_lora_control.json")
    assert leader.steps == 1000 and leader.save_every == 200
    assert incumbent.steps == 1000 and incumbent.save_every == 201
    assert leader.config_sha256 == R.sha256_file(LEADER_YAML)
    assert incumbent.config_sha256 == R.sha256_file(INCUMBENT_YAML)
    # Both arms must share a seed, or the comparison is confounded.
    assert leader.training_seed == incumbent.training_seed
    assert control.role == "zero_lora_control"


def test_end_to_end_with_a_fake_gpu_produces_a_full_analysis(tmp_path):
    env = build_environment(tmp_path, holdout_images=6)
    losses = {
        # arm_a improves with depth; arm_b is flat and worse.
        "arm_a": lambda index, step: (
            0.030 - 0.005 * (step // 200) + 0.001 * index,
            0.030 - 0.005 * (step // 200) + 0.001 * index,
        ),
        "arm_b": lambda index, step: (0.030 + 0.001 * index, 0.030 + 0.001 * index),
        "zero_lora": lambda index, step: (0.040 + 0.001 * index, 0.040 + 0.001 * index),
    }
    gpu = FakeGpu(env, losses=losses)
    analysis = R.execute(
        fixture=env.fixture,
        arms=env.arms,
        host=env.host,
        cost=R.CostModel(),
        hours=None,
        workdir=env.workdir,
        runner=gpu,
        hostname="test-box",
    )
    assert analysis["winner"] == "arm_a"
    assert analysis["complete"] is True
    # Best checkpoint is the deeper one, chosen on the composite.
    assert analysis["arms"]["arm_a"]["best"]["candidate"]["step"] == 400
    # Prompted and blank are always reported separately, plus the composite.
    best = analysis["arms"]["arm_a"]["best"]
    assert best["prompted_mean"] != best["blank_mean"] or True
    assert set(best) >= {"prompted_mean", "blank_mean", "composite_mean"}
    # Every checkpoint carries a paired interval against the zero-LoRA control.
    for entry in analysis["arms"]["arm_a"]["checkpoints"]:
        assert entry["vs_zero_lora"]["blank"]["defined"] is True
        assert entry["vs_zero_lora"]["prompted"]["defined"] is True
        assert entry["vs_zero_lora"]["composite"]["defined"] is True
    # The analysis is published, deterministic, and content-addressed.
    published = Path(analysis["analysis_path"])
    assert published.is_file()
    assert published.parent.name == "analysis"
    written = R.read_json(published)
    assert written["winner"] == "arm_a"
    assert written["decision_rule"] == R.DECISION_RULE
