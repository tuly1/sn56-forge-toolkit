"""Deterministic contracts for the Lane-C transfer analyzer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


_ROOT = Path(__file__).parents[1]
_CALIBRATION = _ROOT / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))

import lane_c_transfer_analyze as analyzer  # noqa: E402
import lane_c_transfer_score as runner  # noqa: E402


def _write_json(path: Path, value: dict) -> str:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    root: Path, *, seed_mode: str = "reduced-2"
) -> tuple[Path, Path, dict[str, Path]]:
    root.mkdir()
    ids = [f"miner{i:02d}" for i in range(14)]
    losses = {candidate: 0.04 + index * 0.001 for index, candidate in enumerate(ids)}
    board = {
        "schema": "sn56.week6.lane-c.c0.final-board.v1",
        "scored_board": [
            {
                "hotkey8": candidate,
                "rank": index + 1,
                "test_loss": losses[candidate],
            }
            for index, candidate in enumerate(ids)
        ],
        "frozen_reference": {
            "leader": {"hotkey8": ids[0], "test_loss": losses[ids[0]]},
            "robust": {"hotkey8": ids[3], "test_loss": losses[ids[3]]},
            "ours": {"hotkey8": "5HLA2QWY", "test_loss": 0.048},
            "floor": {"hotkey8": ids[12], "test_loss": losses[ids[12]]},
        },
    }
    # Replace rank-9's synthetic id with the project's frozen hotkey.
    board["scored_board"][8]["hotkey8"] = "5HLA2QWY"
    ids[8] = "5HLA2QWY"
    losses = {
        row["hotkey8"]: row["test_loss"] for row in board["scored_board"]
    }
    board_path = root / "board.json"
    _write_json(board_path, board)

    raw_paths: dict[str, Path] = {}
    summaries = []
    dataset_sha = "d" * 64
    all_ids = ids + ["zero"]
    for arm_index, candidate in enumerate(all_ids):
        baseline = 0.05 + arm_index * 0.001 if candidate != "zero" else 0.075
        blank = [baseline + image_index * 0.00001 for image_index in range(24)]
        text = [value + 0.002 for value in blank]
        blank_mean = sum(blank) / len(blank)
        text_mean = sum(text) / len(text)
        weighted = 0.25 * text_mean + 0.75 * blank_mean
        candidate_sha = hashlib.sha256(candidate.encode()).hexdigest()
        raw = {
            "schema": 2,
            "evaluator": "god_krea2_img2img_exact",
            "model_type": "krea2",
            "candidate_sha256": candidate_sha,
            "dataset_sha256": dataset_sha,
            "seed_mode": seed_mode,
            "generations": 2,
            "validator_default_generations": 5,
            "seeds": [2746317213, 1181241943],
            "text_weight": 0.25,
            "blank_mean": blank_mean,
            "text_mean": text_mean,
            "weighted_loss": weighted,
            "scored_rows": [
                {
                    "index": image_index,
                    "image": f"image-{image_index:02d}.png",
                    "blank_prompt_loss": blank[image_index],
                    "text_guided_loss": text[image_index],
                }
                for image_index in range(24)
            ],
        }
        raw_path = root / f"{candidate}.json"
        raw_sha = _write_json(raw_path, raw)
        raw_paths[candidate] = raw_path
        summaries.append(
            {
                "id": candidate,
                "candidate_sha256": candidate_sha,
                "result_path": str(raw_path),
                "result_file_sha256": raw_sha,
                "weighted_loss": weighted,
            }
        )
    aggregate = {
        "schema": 1,
        "kind": "sn56-lane-c-transfer-reduced-2-results",
        "status": "complete",
        "candidate_count": 15,
        "image_count": 24,
        "generations": 2,
        "seed_mode": "reduced-2",
        "expected_prompt_count_per_candidate": 96,
        "expected_dataset_sha256": dataset_sha,
        "results": list(reversed(summaries)),
    }
    aggregate["aggregate_sha256"] = runner._canonical_sha256(aggregate)
    aggregate_path = root / "aggregate.json"
    _write_json(aggregate_path, aggregate)
    return aggregate_path, board_path, raw_paths


def test_analyzer_is_deterministic_and_classifies_strong_transfer(
    tmp_path: Path,
) -> None:
    aggregate, board, _ = _fixture(tmp_path / "input")
    first = analyzer.analyze(aggregate, board)
    second = analyzer.analyze(aggregate, board)
    assert first == second
    assert first["classification"] == "TRANSFER-VALID"
    assert first["primary_spearman"]["rho"] == pytest.approx(1.0)
    assert first["primary_spearman"]["bootstrap_ci"]["lower"] > 0.20
    assert first["primary_spearman"]["permutation_test"]["permutations"] == 10_000
    assert first["resampling"]["bootstrap"] == {
        "resamples": 2_000,
        "seed": 2_026_080_501,
    }
    assert len(first["arms"]) == 14
    assert first["claim_limits"]["release_authority"] is False

    output_a = tmp_path / "analysis-a.json"
    output_b = tmp_path / "analysis-b.json"
    assert analyzer.main(
        ["--aggregate", str(aggregate), "--c0-board", str(board), "--output", str(output_a)]
    ) == 0
    assert analyzer.main(
        ["--aggregate", str(aggregate), "--c0-board", str(board), "--output", str(output_b)]
    ) == 0
    assert output_a.read_bytes() == output_b.read_bytes()
    with pytest.raises(FileExistsError, match="stale analysis"):
        analyzer.main(
            [
                "--aggregate",
                str(aggregate),
                "--c0-board",
                str(board),
                "--output",
                str(output_a),
            ]
        )


def test_analyzer_rehashes_every_raw_result(tmp_path: Path) -> None:
    aggregate, board, raw = _fixture(tmp_path / "input")
    raw["miner00"].write_bytes(raw["miner00"].read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="raw result hash differs"):
        analyzer.analyze(aggregate, board)


def test_analyzer_rejects_non_reduced_or_misaligned_rows(tmp_path: Path) -> None:
    aggregate, board, _ = _fixture(tmp_path / "input", seed_mode="validator-exact-5")
    with pytest.raises(ValueError, match="raw result contract mismatch"):
        analyzer.analyze(aggregate, board)


def test_analyzer_rejects_different_order_even_with_rebound_hashes(
    tmp_path: Path,
) -> None:
    aggregate_path, board, raw = _fixture(tmp_path / "input")
    candidate_path = raw["miner00"]
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["scored_rows"][0]["image"], candidate["scored_rows"][1]["image"] = (
        candidate["scored_rows"][1]["image"],
        candidate["scored_rows"][0]["image"],
    )
    rebound_sha = _write_json(candidate_path, candidate)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    next(row for row in aggregate["results"] if row["id"] == "miner00")[
        "result_file_sha256"
    ] = rebound_sha
    body = {key: value for key, value in aggregate.items() if key != "aggregate_sha256"}
    aggregate["aggregate_sha256"] = runner._canonical_sha256(body)
    _write_json(aggregate_path, aggregate)
    with pytest.raises(ValueError, match="orders differ"):
        analyzer.analyze(aggregate_path, board)


def test_pre_score_classification_precedence() -> None:
    positive_ci = {"lower": 0.3, "upper": 0.8}
    advantage_ci = {"lower": 0.01, "upper": 0.1}
    assert analyzer._classify(
        rho=0.7,
        rho_ci=positive_ci,
        leader_floor=0.1,
        leader_floor_ci=advantage_ci,
        leader_ours=0.1,
        leader_ours_ci=advantage_ci,
        deltas_vs_zero=[0.0] * 14,
    ) == "TRANSFER-VALID"
    assert analyzer._classify(
        rho=0.4,
        rho_ci={"lower": -0.1, "upper": 0.7},
        leader_floor=0.1,
        leader_floor_ci={"lower": -0.1},
        leader_ours=0.1,
        leader_ours_ci={"lower": -0.1},
        deltas_vs_zero=[0.02] * 14,
    ) == "TRANSFER-PARTIAL"
    assert analyzer._classify(
        rho=-0.7,
        rho_ci={"lower": -0.9, "upper": -0.2},
        leader_floor=-0.1,
        leader_floor_ci={"lower": -0.2},
        leader_ours=-0.1,
        leader_ours_ci={"lower": -0.2},
        deltas_vs_zero=[0.02] * 14,
    ) == "TRANSFER-INVERTED"
    assert analyzer._classify(
        rho=None,
        rho_ci={"lower": None, "upper": None},
        leader_floor=0.0,
        leader_floor_ci={"lower": 0.0},
        leader_ours=0.0,
        leader_ours_ci={"lower": 0.0},
        deltas_vs_zero=[0.0049] * 14,
    ) == "NON-TRANSFER"
    assert analyzer._classify(
        rho=0.1,
        rho_ci={"lower": -0.2, "upper": 0.4},
        leader_floor=0.1,
        leader_floor_ci={"lower": -0.1},
        leader_ours=-0.1,
        leader_ours_ci={"lower": -0.2},
        deltas_vs_zero=[0.02] * 14,
    ) == "TRANSFER-NULL"
