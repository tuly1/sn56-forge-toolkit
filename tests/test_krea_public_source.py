"""CPU-only tests for the frozen K2-K4 public provenance adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys

import pytest
import yaml


_CALIBRATION = Path(__file__).parents[1] / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))

import krea_public_source as public_source  # noqa: E402


_LEDGER = _CALIBRATION / "week5" / "krea-r1-field-ledger.json"
_DISCOVERY_PLAN = _CALIBRATION / "week5" / "krea-discovery-plan.json"
_CONFIGS = {
    "K2": Path("K2-rank2-f4766189-config.yaml"),
    "K3": Path("K3-rank3-919e07cd-config.yaml"),
    "K4": Path("K4-rank5-71bf349e-config.yaml"),
}
_CONFIG_SHA256 = {
    "K2": "249aa4bea68c41528439873ad6be1a6e4a53fb369309a83b31b5d73f23904ea9",
    "K3": "bd5bb24cc7c997459eac12799aacd5ab5317f7aeaf86f5c915a65fc727c5a803",
    "K4": "5cac928af4514603a0db25b66b8ef97bb1a61bf4b394db6f5e70667e4690f2d5",
}
_LOCAL_EVIDENCE_CONFIG_ROOT = (
    Path(__file__).parents[2] / "day0-staging" / "krea-r1-thin-evidence" / "raw-configs"
)


def _config(path: Path, arm: str) -> Path:
    values = {
        "K2": {
            "steps": 1140,
            "lr": 8.6e-5,
            "rank": 32,
            "optimizer": "adamw8bit",
            "loss": "mse",
            "guidance": 2,
            "save": 120,
            "dropout": 0.1,
            "ema": {"use_ema": False, "ema_decay": 0.99},
            "optimizer_params": {"weight_decay": 0.0001},
        },
        "K3": {
            "steps": 1432,
            "lr": 1e-4,
            "rank": 32,
            "optimizer": "adamw8bit",
            "loss": "mae",
            "guidance": 3,
            "save": 200,
            "dropout": None,
            "ema": None,
            "optimizer_params": {"weight_decay": 0.0001},
        },
        "K4": {
            "steps": 1140,
            "lr": 8.6e-7,
            "rank": 64,
            "optimizer": "automagic",
            "loss": "mse",
            "guidance": 2,
            "save": 120,
            "dropout": 0.3,
            "ema": {"use_ema": False, "ema_decay": 0.99},
            "optimizer_params": {
                "lr_bump": 1e-6,
                "max_lr": 0.001,
                "min_lr": 1e-7,
                "weight_decay": 0.0001,
            },
        },
    }[arm]
    dataset = {}
    if values["dropout"] is not None:
        dataset["caption_dropout_rate"] = values["dropout"]
    train = {
        "batch_size": 1,
        "differential_guidance_scale": values["guidance"],
        "do_differential_guidance": True,
        "gradient_accumulation": 1,
        "loss_type": values["loss"],
        "lr": values["lr"],
        "noise_scheduler": "flowmatch",
        "optimizer": values["optimizer"],
        "optimizer_params": values["optimizer_params"],
        "steps": values["steps"],
    }
    if values["ema"] is not None:
        train["ema_config"] = values["ema"]
    value = {
        "config": {
            "process": [
                {
                    "type": "diffusion_trainer",
                    "model": {"arch": "krea2"},
                    "datasets": [dataset],
                    "network": {
                        "linear": values["rank"],
                        "linear_alpha": values["rank"],
                    },
                    "save": {"save_every": values["save"]},
                    "train": train,
                }
            ]
        }
    }
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")
    return path


def _artifact(path: Path, step: int) -> Path:
    header = {
        "__metadata__": {"training_info": json.dumps({"step": step, "epoch": 1})},
        "lora.test": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
    }
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0")
    return path


@pytest.mark.parametrize(
    ("arm", "step", "lr", "selector"),
    [
        ("K2", 960, 8.6e-5, "holdout_selected"),
        ("K3", 1200, 1e-4, "highest_numbered_fallback"),
        ("K4", 840, 8.6e-7, "holdout_selected"),
    ],
)
def test_actual_public_configs_rederive_frozen_source_recipe(
    tmp_path: Path, arm: str, step: int, lr: float, selector: str
) -> None:
    metadata = public_source.build_metadata(
        arm,
        source_config_path=_config(tmp_path / f"{arm}.yaml", arm),
        source_artifact_path=_artifact(tmp_path / f"{arm}.safetensors", step),
        field_ledger_path=_LEDGER,
    )
    fields = metadata["normalized_recipe"]["fields"]
    disclosure = metadata["local_reproduction_disclosure"]
    plan = json.loads(_DISCOVERY_PLAN.read_text(encoding="utf-8"))
    plan_arm = next(row for row in plan["arms"] if row["id"] == arm)
    assert metadata["review_assertion"]["status"] == "unreviewed"
    assert metadata["evaluator_sha"] is None
    assert disclosure["execution_authorized"] is False
    assert [row["name"] for row in disclosure["adapted_fields"]] == sorted(
        plan_arm["adapted_fields"]
    )
    assert (
        next(
            row for row in disclosure["adapted_fields"] if row["name"] == "depth policy"
        )["local_policy"]
        == plan_arm["depth_policy"]
    )
    assert fields["submitted_step"]["source_value"] == step
    assert fields["learning_rate"]["source_value"] == lr
    assert fields["selector"]["source_value"] == selector
    assert fields["effective_batch"]["source_value"] == 1
    if arm == "K3":
        assert fields["dropout"]["classification"] == "unknown"
        assert fields["ema"]["classification"] == "unknown"
        assert [
            row["field"] for row in disclosure["source_unknown_fields"]
        ] == plan_arm["unknown_source_fields"]
        assert {
            row["field"]: row["value"] for row in disclosure["predeclared_local_values"]
        } == plan_arm["predeclared_local_values"]
        assert all(
            "not evidence of K3's source" in row["basis"]
            for row in disclosure["predeclared_local_values"]
        )
    else:
        assert fields["dropout"]["classification"] == "known"
        assert fields["ema"]["classification"] == "known"
        assert disclosure["source_unknown_fields"] == []
        assert disclosure["predeclared_local_values"] == []
        assert (
            next(
                row
                for row in disclosure["adapted_fields"]
                if row["name"] == "offline exact-scoring selection policy"
            )["local_policy"]
            == plan_arm["selection_policy"]
        )


def test_automagic_lr_caveat_is_scoped_only_to_automagic(tmp_path: Path) -> None:
    evidence = {}
    for arm, step in (("K2", 960), ("K3", 1200), ("K4", 840)):
        metadata = public_source.build_metadata(
            arm,
            source_config_path=_config(tmp_path / f"{arm}.yaml", arm),
            source_artifact_path=_artifact(tmp_path / f"{arm}.safetensors", step),
            field_ledger_path=_LEDGER,
        )
        evidence[arm] = metadata["normalized_recipe"]["fields"]["learning_rate"][
            "evidence"
        ]
    assert evidence["K2"] == "Immutable source config."
    assert evidence["K3"] == "Immutable source config."
    assert "Automagic" in evidence["K4"]


def test_k3_unknowns_and_local_values_cannot_be_conflated(tmp_path: Path) -> None:
    metadata = public_source.build_metadata(
        "K3",
        source_config_path=_config(tmp_path / "K3.yaml", "K3"),
        source_artifact_path=_artifact(tmp_path / "K3.safetensors", 1200),
        field_ledger_path=_LEDGER,
    )
    disclosure = metadata["local_reproduction_disclosure"]
    source = metadata["normalized_recipe"]["fields"]
    assert source["dropout"]["source_value"] is None
    assert source["ema"]["source_value"] is None
    assert all(
        row["source_value"] is None for row in disclosure["source_unknown_fields"]
    )
    assert {row["field"] for row in disclosure["predeclared_local_values"]} == {
        "dropout",
        "ema",
    }


def test_artifact_step_must_match_official_ledger(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="header step contradicts"):
        public_source.build_metadata(
            "K2",
            source_config_path=_config(tmp_path / "K2.yaml", "K2"),
            source_artifact_path=_artifact(tmp_path / "wrong.safetensors", 959),
            field_ledger_path=_LEDGER,
        )


def test_k3_absent_fields_cannot_be_silently_reclassified(tmp_path: Path) -> None:
    value = yaml.safe_load(_config(tmp_path / "K3-source.yaml", "K3").read_bytes())
    value["config"]["process"][0]["datasets"][0]["caption_dropout_rate"] = 0.0
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown dropout/EMA"):
        public_source.build_metadata(
            "K3",
            source_config_path=changed,
            source_artifact_path=_artifact(tmp_path / "K3.safetensors", 1200),
            field_ledger_path=_LEDGER,
        )


@pytest.mark.parametrize(("arm", "step"), [("K2", 960), ("K3", 1200), ("K4", 840)])
def test_local_sealed_config_bytes_parse_when_evidence_copy_is_present(
    tmp_path: Path, arm: str, step: int
) -> None:
    config = _LOCAL_EVIDENCE_CONFIG_ROOT / _CONFIGS[arm]
    if not config.is_file():
        pytest.skip("off-repository thin evidence copy is not present")
    assert hashlib.sha256(config.read_bytes()).hexdigest() == _CONFIG_SHA256[arm]
    metadata = public_source.build_metadata(
        arm,
        source_config_path=config,
        source_artifact_path=_artifact(tmp_path / f"{arm}.safetensors", step),
        field_ledger_path=_LEDGER,
    )
    assert metadata["source_arm_id"] == arm
