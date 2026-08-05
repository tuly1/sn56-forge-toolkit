#!/usr/bin/env python3
"""Analyze the single-fixture Lane-C reduced-2 transfer diagnostic.

The analyzer is deterministic, dependency-free, and create-only.  It reopens
every raw scorer result, checks its aggregate hash binding and the frozen C0
field, then computes the predeclared transfer statistics.  Its output is
diagnostic evidence only; it is never release or deployment authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable

try:
    from . import lane_c_transfer_score as score_runner
except ImportError:  # pragma: no cover - direct script execution.
    import lane_c_transfer_score as score_runner  # type: ignore[no-redef]


_SHA256 = re.compile(r"[0-9a-f]{64}")
_EXPECTED_ARTIFACTS = 14
_EXPECTED_ROWS = 24
_BOOTSTRAP_RESAMPLES = 2_000
_BOOTSTRAP_SEED = 2_026_080_501
_PERMUTATIONS = 10_000
_PERMUTATION_SEED = 2_026_080_502
_CI = (0.025, 0.975)
_OURS_DEFAULT = "5HLA2QWY"


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--c0-board", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _finite(value: Any, label: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(f"{label} is invalid")
    return number


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        raise ValueError("mean requires at least one value")
    return math.fsum(rows) / len(rows)


def _ranks(values: list[float]) -> list[float]:
    """Return average one-based ranks, including exact-tie handling."""

    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have equal length >= 2")
    left_ranks = _ranks(left)
    right_ranks = _ranks(right)
    left_mean = _mean(left_ranks)
    right_mean = _mean(right_ranks)
    covariance = math.fsum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks, strict=True)
    )
    left_ss = math.fsum((value - left_mean) ** 2 for value in left_ranks)
    right_ss = math.fsum((value - right_mean) ** 2 for value in right_ranks)
    denominator = math.sqrt(left_ss * right_ss)
    if denominator == 0:
        return None
    return covariance / denominator


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _ci(values: list[float]) -> dict[str, Any]:
    return {
        "confidence": 0.95,
        "method": "image-paired percentile bootstrap",
        "lower": _percentile(values, _CI[0]),
        "upper": _percentile(values, _CI[1]),
        "valid_resamples": len(values),
    }


def _advantage(reference: float, comparison: float) -> float:
    """Relative loss reduction; positive means the reference is better."""

    if comparison <= 0:
        raise ValueError("comparison loss must be positive")
    return (comparison - reference) / comparison


def _board(path: Path) -> tuple[dict[str, float], dict[str, str], str]:
    value, file_sha = score_runner._read_json(path, "frozen C0 board")
    if not isinstance(value, dict) or value.get("schema") != (
        "sn56.week6.lane-c.c0.final-board.v1"
    ):
        raise ValueError("frozen C0 board schema mismatch")
    rows = value.get("scored_board")
    if not isinstance(rows, list) or len(rows) != _EXPECTED_ARTIFACTS:
        raise ValueError("frozen C0 board must contain exactly fourteen scores")
    losses: dict[str, float] = {}
    ranks: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("C0 board row is not an object")
        hotkey = row.get("hotkey8")
        rank = row.get("rank")
        if (
            not isinstance(hotkey, str)
            or not score_runner._SAFE_ID.fullmatch(hotkey)
            or hotkey in losses
            or isinstance(rank, bool)
            or not isinstance(rank, int)
        ):
            raise ValueError("C0 board identity is invalid")
        losses[hotkey] = _finite(row.get("test_loss"), f"{hotkey} test_loss")
        ranks.add(rank)
    if ranks != set(range(1, _EXPECTED_ARTIFACTS + 1)):
        raise ValueError("C0 board ranks are incomplete")

    frozen = value.get("frozen_reference")
    if not isinstance(frozen, dict):
        raise ValueError("C0 frozen reference is missing")
    roles: dict[str, str] = {}
    for role in ("leader", "floor", "ours", "robust"):
        row = frozen.get(role)
        if not isinstance(row, dict) or row.get("hotkey8") not in losses:
            raise ValueError(f"C0 {role} reference is invalid")
        hotkey = row["hotkey8"]
        if not math.isclose(
            _finite(row.get("test_loss"), f"C0 {role} test_loss"),
            losses[hotkey],
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(f"C0 {role} reference differs from scored board")
        roles[role] = hotkey
    if roles["ours"] != _OURS_DEFAULT:
        raise ValueError("C0 ours reference is not the project hotkey")
    return losses, roles, file_sha


def _load_results(
    aggregate_path: Path,
    *,
    expected_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], str, list[str]]:
    aggregate, aggregate_file_sha = score_runner._read_json(
        aggregate_path, "Lane-C aggregate"
    )
    if not isinstance(aggregate, dict):
        raise ValueError("Lane-C aggregate must be an object")
    if (
        aggregate.get("schema") != 1
        or aggregate.get("kind") != "sn56-lane-c-transfer-reduced-2-results"
        or aggregate.get("status") != "complete"
        or aggregate.get("candidate_count") != len(expected_ids)
        or aggregate.get("image_count") != _EXPECTED_ROWS
        or aggregate.get("generations") != 2
        or aggregate.get("seed_mode") != "reduced-2"
        or aggregate.get("expected_prompt_count_per_candidate")
        != _EXPECTED_ROWS * 2 * 2
    ):
        raise ValueError("Lane-C aggregate contract mismatch")
    semantic_sha = aggregate.get("aggregate_sha256")
    semantic_body = {
        key: value for key, value in aggregate.items() if key != "aggregate_sha256"
    }
    if (
        not isinstance(semantic_sha, str)
        or not _SHA256.fullmatch(semantic_sha)
        or score_runner._canonical_sha256(semantic_body) != semantic_sha
    ):
        raise ValueError("Lane-C aggregate semantic SHA-256 mismatch")
    dataset_sha = aggregate.get("expected_dataset_sha256")
    if not isinstance(dataset_sha, str) or not _SHA256.fullmatch(dataset_sha):
        raise ValueError("Lane-C aggregate lacks its expected dataset identity")
    summary_rows = aggregate.get("results")
    if not isinstance(summary_rows, list) or len(summary_rows) != len(expected_ids):
        raise ValueError("Lane-C aggregate result count mismatch")
    summaries: dict[str, dict[str, Any]] = {}
    raw_results: dict[str, dict[str, Any]] = {}
    common_order: list[str] | None = None
    for summary in summary_rows:
        if not isinstance(summary, dict):
            raise ValueError("Lane-C aggregate summary is not an object")
        candidate_id = summary.get("id")
        result_path_raw = summary.get("result_path")
        expected_result_sha = summary.get("result_file_sha256")
        if (
            not isinstance(candidate_id, str)
            or candidate_id in summaries
            or candidate_id not in expected_ids
            or not isinstance(result_path_raw, str)
            or not Path(result_path_raw).is_absolute()
            or not isinstance(expected_result_sha, str)
            or not _SHA256.fullmatch(expected_result_sha)
        ):
            raise ValueError("Lane-C aggregate summary identity mismatch")
        raw, observed_result_sha = score_runner._read_json(
            Path(result_path_raw), f"{candidate_id} raw result"
        )
        if observed_result_sha != expected_result_sha:
            raise RuntimeError(f"{candidate_id} raw result hash differs from aggregate")
        if not isinstance(raw, dict):
            raise ValueError(f"{candidate_id} raw result is not an object")
        if (
            raw.get("schema") != 2
            or raw.get("evaluator") != "god_krea2_img2img_exact"
            or raw.get("model_type") != "krea2"
            or raw.get("seed_mode") != "reduced-2"
            or raw.get("generations") != 2
            or raw.get("validator_default_generations") != 5
            or raw.get("seeds") != [2746317213, 1181241943]
            or raw.get("dataset_sha256") != dataset_sha
        ):
            raise ValueError(f"{candidate_id} raw result contract mismatch")
        if raw.get("candidate_sha256") != summary.get("candidate_sha256"):
            raise ValueError(f"{candidate_id} candidate hash differs from aggregate")
        scored_rows = raw.get("scored_rows")
        if not isinstance(scored_rows, list) or len(scored_rows) != _EXPECTED_ROWS:
            raise ValueError(f"{candidate_id} must contain 24 scored rows")
        order: list[str] = []
        blank: list[float] = []
        text: list[float] = []
        for index, row in enumerate(scored_rows):
            if not isinstance(row, dict) or row.get("index") != index:
                raise ValueError(f"{candidate_id} scored-row index mismatch")
            image = row.get("image")
            if not isinstance(image, str) or not image or Path(image).name != image:
                raise ValueError(f"{candidate_id} scored-row image is invalid")
            order.append(image)
            blank.append(
                _finite(row.get("blank_prompt_loss"), f"{candidate_id} blank loss")
            )
            text.append(
                _finite(row.get("text_guided_loss"), f"{candidate_id} text loss")
            )
        if len(set(order)) != _EXPECTED_ROWS:
            raise ValueError(f"{candidate_id} scored-row order has duplicates")
        if common_order is None:
            common_order = order
        elif order != common_order:
            raise ValueError("candidate scored-row orders differ")
        blank_mean = _mean(blank)
        text_mean = _mean(text)
        weighted = 0.25 * text_mean + 0.75 * blank_mean
        for key, observed, expected in (
            ("blank_mean", raw.get("blank_mean"), blank_mean),
            ("text_mean", raw.get("text_mean"), text_mean),
            ("weighted_loss", raw.get("weighted_loss"), weighted),
            ("aggregate weighted_loss", summary.get("weighted_loss"), weighted),
        ):
            if not math.isclose(
                _finite(observed, f"{candidate_id} {key}"),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError(f"{candidate_id} {key} was not reproduced")
        if not math.isclose(
            _finite(raw.get("text_weight"), f"{candidate_id} text weight"),
            0.25,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(f"{candidate_id} text weighting is not 25/75")
        summaries[candidate_id] = summary
        raw_results[candidate_id] = {
            "blank": blank,
            "text": text,
            "blank_mean": blank_mean,
            "weighted_25_75_mean": weighted,
            "candidate_sha256": summary.get("candidate_sha256"),
            "result_file_sha256": observed_result_sha,
        }
    if set(summaries) != expected_ids or common_order is None:
        raise ValueError("Lane-C aggregate IDs differ from board plus zero")
    return raw_results, aggregate, aggregate_file_sha, common_order


def _bootstrap(
    *,
    artifact_ids: list[str],
    board_losses: dict[str, float],
    scores: dict[str, dict[str, Any]],
    roles: dict[str, str],
) -> dict[str, Any]:
    rng = random.Random(_BOOTSTRAP_SEED)
    board = [board_losses[candidate] for candidate in artifact_ids]
    rho_values: list[float] = []
    advantages: dict[str, list[float]] = {
        "leader_vs_floor": [],
        "leader_vs_ours": [],
        "robust_vs_ours": [],
    }
    for _ in range(_BOOTSTRAP_RESAMPLES):
        indices = [rng.randrange(_EXPECTED_ROWS) for _ in range(_EXPECTED_ROWS)]
        means = {
            candidate: _mean(scores[candidate]["blank"][index] for index in indices)
            for candidate in scores
        }
        rho = _spearman(board, [means[candidate] for candidate in artifact_ids])
        if rho is not None:
            rho_values.append(rho)
        advantages["leader_vs_floor"].append(
            _advantage(means[roles["leader"]], means[roles["floor"]])
        )
        advantages["leader_vs_ours"].append(
            _advantage(means[roles["leader"]], means[roles["ours"]])
        )
        advantages["robust_vs_ours"].append(
            _advantage(means[roles["robust"]], means[roles["ours"]])
        )
    return {
        "resamples": _BOOTSTRAP_RESAMPLES,
        "seed": _BOOTSTRAP_SEED,
        "unit": "ordered image row, paired across all arms",
        "spearman": _ci(rho_values),
        "advantages": {name: _ci(values) for name, values in advantages.items()},
    }


def _permutation_test(board: list[float], observed_scores: list[float]) -> dict[str, Any]:
    observed = _spearman(board, observed_scores)
    if observed is None:
        return {
            "permutations": _PERMUTATIONS,
            "seed": _PERMUTATION_SEED,
            "alternative": "two-sided",
            "extreme_or_equal": None,
            "p_value": None,
        }
    rng = random.Random(_PERMUTATION_SEED)
    labels = list(board)
    extreme = 0
    threshold = abs(observed) - 1e-15
    for _ in range(_PERMUTATIONS):
        rng.shuffle(labels)
        permuted = _spearman(labels, observed_scores)
        if permuted is not None and abs(permuted) >= threshold:
            extreme += 1
    return {
        "permutations": _PERMUTATIONS,
        "seed": _PERMUTATION_SEED,
        "alternative": "two-sided",
        "extreme_or_equal": extreme,
        "p_value": (extreme + 1) / (_PERMUTATIONS + 1),
    }


def _classify(
    *,
    rho: float | None,
    rho_ci: dict[str, Any],
    leader_floor: float,
    leader_floor_ci: dict[str, Any],
    leader_ours: float,
    leader_ours_ci: dict[str, Any],
    deltas_vs_zero: list[float],
) -> str:
    rho_lower = rho_ci.get("lower")
    rho_upper = rho_ci.get("upper")
    leader_floor_lower = leader_floor_ci.get("lower")
    leader_ours_lower = leader_ours_ci.get("lower")
    if (
        rho is not None
        and rho >= 0.60
        and isinstance(rho_lower, (int, float))
        and rho_lower > 0.20
        and isinstance(leader_floor_lower, (int, float))
        and leader_floor_lower > 0
        and isinstance(leader_ours_lower, (int, float))
        and leader_ours_lower > 0
    ):
        return "TRANSFER-VALID"
    if (
        rho is not None
        and rho >= 0.30
        and leader_floor > 0
        and leader_ours > 0
    ):
        return "TRANSFER-PARTIAL"
    if (
        rho is not None
        and rho <= -0.30
        and isinstance(rho_upper, (int, float))
        and rho_upper < 0
    ):
        return "TRANSFER-INVERTED"
    if all(abs(value) < 0.005 for value in deltas_vs_zero):
        return "NON-TRANSFER"
    return "TRANSFER-NULL"


def analyze(aggregate_path: Path, board_path: Path) -> dict[str, Any]:
    board_losses, roles, board_file_sha = _board(board_path)
    artifact_ids = sorted(board_losses, key=lambda candidate: (
        board_losses[candidate], candidate
    ))
    expected_ids = set(artifact_ids) | {"zero"}
    scores, aggregate, aggregate_file_sha, ordered_images = _load_results(
        aggregate_path, expected_ids=expected_ids
    )
    blank_scores = [scores[candidate]["blank_mean"] for candidate in artifact_ids]
    board_vector = [board_losses[candidate] for candidate in artifact_ids]
    rho = _spearman(board_vector, blank_scores)
    bootstrap = _bootstrap(
        artifact_ids=artifact_ids,
        board_losses=board_losses,
        scores=scores,
        roles=roles,
    )
    permutation = _permutation_test(board_vector, blank_scores)
    point_advantages = {
        "leader_vs_floor": _advantage(
            scores[roles["leader"]]["blank_mean"],
            scores[roles["floor"]]["blank_mean"],
        ),
        "leader_vs_ours": _advantage(
            scores[roles["leader"]]["blank_mean"],
            scores[roles["ours"]]["blank_mean"],
        ),
        "robust_vs_ours": _advantage(
            scores[roles["robust"]]["blank_mean"],
            scores[roles["ours"]]["blank_mean"],
        ),
    }
    zero_blank = scores["zero"]["blank_mean"]
    arm_rows = []
    deltas = []
    for candidate in artifact_ids:
        delta = _advantage(scores[candidate]["blank_mean"], zero_blank)
        deltas.append(delta)
        arm_rows.append(
            {
                "id": candidate,
                "board_test_loss": board_losses[candidate],
                "blank_mean": scores[candidate]["blank_mean"],
                "weighted_25_75_mean": scores[candidate]["weighted_25_75_mean"],
                "delta_vs_zero_blank": delta,
                "candidate_sha256": scores[candidate]["candidate_sha256"],
                "result_file_sha256": scores[candidate]["result_file_sha256"],
            }
        )
    classification = _classify(
        rho=rho,
        rho_ci=bootstrap["spearman"],
        leader_floor=point_advantages["leader_vs_floor"],
        leader_floor_ci=bootstrap["advantages"]["leader_vs_floor"],
        leader_ours=point_advantages["leader_vs_ours"],
        leader_ours_ci=bootstrap["advantages"]["leader_vs_ours"],
        deltas_vs_zero=deltas,
    )
    output = {
        "schema": 1,
        "kind": "sn56-lane-c-single-d1-transfer-diagnostic",
        "classification": classification,
        "inputs": {
            "aggregate": str(aggregate_path),
            "aggregate_file_sha256": aggregate_file_sha,
            "aggregate_semantic_sha256": aggregate.get("aggregate_sha256"),
            "c0_board": str(board_path),
            "c0_board_file_sha256": board_file_sha,
            "dataset_sha256": aggregate["expected_dataset_sha256"],
        },
        "field": {
            "artifact_count": _EXPECTED_ARTIFACTS,
            "artifact_ids": artifact_ids,
            "zero_id": "zero",
            "roles": roles,
        },
        "scoring": {
            "fixture": "D1",
            "image_rows": _EXPECTED_ROWS,
            "ordered_images": ordered_images,
            "ordered_images_sha256": score_runner._canonical_sha256(ordered_images),
            "generations": 2,
            "seed_mode": "reduced-2",
            "primary_metric": "blank_prompt_mean",
            "secondary_metric": "0.25*text_guided_mean + 0.75*blank_prompt_mean",
            "direction": "lower_is_better",
        },
        "primary_spearman": {
            "rho": rho,
            "artifact_count": _EXPECTED_ARTIFACTS,
            "bootstrap_ci": bootstrap["spearman"],
            "permutation_test": permutation,
        },
        "paired_advantages": {
            name: {
                "point": point_advantages[name],
                "bootstrap_ci": bootstrap["advantages"][name],
                "definition": "(comparison_blank-reference_blank)/comparison_blank",
            }
            for name in (
                "leader_vs_floor",
                "leader_vs_ours",
                "robust_vs_ours",
            )
        },
        "zero": {
            "blank_mean": scores["zero"]["blank_mean"],
            "weighted_25_75_mean": scores["zero"]["weighted_25_75_mean"],
            "candidate_sha256": scores["zero"]["candidate_sha256"],
            "result_file_sha256": scores["zero"]["result_file_sha256"],
        },
        "arms": arm_rows,
        "resampling": {
            "bootstrap": {
                "resamples": _BOOTSTRAP_RESAMPLES,
                "seed": _BOOTSTRAP_SEED,
            },
            "artifact_label_permutation": {
                "permutations": _PERMUTATIONS,
                "seed": _PERMUTATION_SEED,
            },
        },
        "pre_score_rule": {
            "TRANSFER-VALID": (
                "blank rho >= 0.60; bootstrap lower bound > 0.20; "
                "leader-vs-floor and leader-vs-ours bootstrap lower bounds > 0"
            ),
            "TRANSFER-PARTIAL": (
                "blank rho >= 0.30 and leader-vs-floor plus leader-vs-ours "
                "point advantages are positive"
            ),
            "TRANSFER-INVERTED": "blank rho <= -0.30 and bootstrap upper bound < 0",
            "NON-TRANSFER": (
                "every artifact's absolute blank delta versus zero is < 0.5%"
            ),
            "otherwise": "TRANSFER-NULL",
            "precedence": [
                "TRANSFER-VALID",
                "TRANSFER-PARTIAL",
                "TRANSFER-INVERTED",
                "NON-TRANSFER",
                "TRANSFER-NULL",
            ],
        },
        "claim_limits": {
            "reduced_2_is_diagnostic": True,
            "single_d1_fixture_is_diagnostic": True,
            "release_authority": False,
            "deployment_authority": False,
        },
    }
    output["analysis_sha256"] = score_runner._canonical_sha256(output)
    return output


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    aggregate_path = Path(args.aggregate).expanduser().resolve(strict=True)
    board_path = Path(args.c0_board).expanduser().resolve(strict=True)
    output_path = Path(args.output).expanduser().absolute()
    if output_path.exists():
        raise FileExistsError(f"refusing stale analysis output: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"analysis output parent is missing: {output_path.parent}")
    result = analyze(aggregate_path, board_path)
    score_runner._publish_json(output_path, result)
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
