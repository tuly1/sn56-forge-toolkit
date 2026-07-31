"""Focused tests for the non-authorizing recovery-to-finalist bridge."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest


_CALIBRATION = Path(__file__).parents[1] / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))

import krea_provenance  # noqa: E402
import krea_recovery_evidence as recovery  # noqa: E402
import krea_waiver_finalist_freeze as waiver_freeze  # noqa: E402


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sha1(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8")).hexdigest()


def _canonical(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(krea_provenance.canonical_bytes(value) + b"\n")
    return krea_provenance.file_sha256(path)


def _actor(identity: str, role: str) -> dict:
    return {
        "actor_class": "agent",
        "actor_id": identity,
        "display_name": f"Test {identity} (agent)",
        "role": role,
        "review_instance_id": f"{identity}-instance",
        "identity_assurance": (
            "self-declared-agent-identity-not-human-or-cryptographic-authentication"
        ),
    }


def _task_rows(root: Path, *, complete: bool) -> list[dict[str, str]]:
    rows = []
    counts = {"K0": 8, "K1": 8, "K2": 8, "K3": 7, "K4": 7, "K5": 7}
    for fixture in ("D1", "D2"):
        for family, count in counts.items():
            for ordinal in range(1, count + 1):
                step = ordinal * 10
                final = ordinal == count
                kind = "final" if final else "step"
                task_id = f"{fixture.lower()}-{family.lower()}-{kind}{step}"
                candidate = root / "candidates" / f"{task_id}.safetensors"
                rows.append(
                    {
                        "task_id": task_id,
                        "fixture": fixture,
                        "cell": f"{fixture}-{family}",
                        "label": f"{kind}-{step}",
                        "coverage_tier": (
                            "SPARSE_PRIMARY" if ordinal % 2 else "EXHAUSTIVE_BACKFILL"
                        ),
                        "expected_candidate": str(candidate),
                        "candidate_sha256": "-",
                        "state": "COMPLETE" if complete else "QUEUED",
                    }
                )
        zero = root / "candidates" / "zero.safetensors"
        rows.append(
            {
                "task_id": f"{fixture.lower()}-zero-baseline",
                "fixture": fixture,
                "cell": "ALL",
                "label": "zero-baseline",
                "coverage_tier": "INDEPENDENT_ZERO",
                "expected_candidate": str(zero),
                "candidate_sha256": "-",
                "state": "COMPLETE" if complete else "QUEUED",
            }
        )
    assert len(rows) == 92
    return rows


def _write_ledger(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(recovery._LEDGER_HEADER)]
    lines.extend("\t".join(row[key] for key in recovery._LEDGER_HEADER) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _result(
    *,
    fixture: str,
    candidate: Path,
    candidate_sha: str,
    weighted_loss: float,
    comfy_log_sha256: str,
    comfy_log_bytes: int,
) -> dict:
    count = recovery.FIXTURE_ROWS[fixture]
    text = [weighted_loss for _ in range(count)]
    blank = [weighted_loss for _ in range(count)]
    scored = [
        {
            "index": index,
            "image": f"image-{index:03d}.png",
            "image_bytes": 10,
            "image_format": "PNG",
            "image_height": 8,
            "image_mode": "RGB",
            "image_sha256": _sha(f"{fixture}-image-{index}"),
            "image_width": 8,
            "prompt": f"prompt-{index:03d}.txt",
            "prompt_bytes": 10,
            "prompt_sha256": _sha(f"{fixture}-prompt-{index}"),
            "text_guided_loss": weighted_loss,
            "blank_prompt_loss": weighted_loss,
        }
        for index in range(count)
    ]
    return {
        "schema": 2,
        "evaluator": "god_krea2_img2img_exact",
        "candidate": candidate.name,
        "candidate_sha256": candidate_sha,
        "candidate_bytes": candidate.stat().st_size,
        "staged_candidate_sha256": candidate_sha,
        "comfy_lora_name": f"candidate-{candidate_sha}.safetensors",
        "model_type": "krea2",
        "dataset": f"/campaign/fixtures/{fixture}/evaluation",
        "dataset_sha256": _sha(f"dataset-{fixture}"),
        "image_count": count,
        "scored_rows": scored,
        "base_name": "krea2.safetensors",
        "asset_sha256": {"base": _sha("base")},
        "asset_bytes": {"base": 1},
        "steps": 20,
        "cfg": 12,
        "denoise": 0.8,
        "generations": 5,
        "master_seed": 42,
        "seeds": [1, 2, 3, 4, 5],
        "text_guided_losses": text,
        "blank_prompt_losses": blank,
        "text_mean": weighted_loss,
        "blank_mean": weighted_loss,
        "text_weight": 0.25,
        "weighted_loss": weighted_loss,
        "direction": "min",
        "elapsed_s": 1.0,
        "source": {
            "god": {
                "commit": _sha1("god-commit"),
                "tree": _sha1("god-tree"),
                "tracked_worktree_clean": True,
                "nonignored_worktree_clean": True,
            },
            "comfyui": {
                "commit": _sha1("comfy-commit"),
                "tree": _sha1("comfy-tree"),
                "tracked_worktree_clean": True,
                "nonignored_worktree_clean": True,
            },
            "tooling_nodes": {
                "commit": _sha1("tooling-commit"),
                "tree": _sha1("tooling-tree"),
                "tracked_worktree_clean": True,
                "nonignored_worktree_clean": True,
            },
            "expected_commits": {
                "god": _sha1("god-commit"),
                "comfyui": _sha1("comfy-commit"),
                "tooling_nodes": _sha1("tooling-commit"),
            },
            "god_import_bindings": {
                "core": {
                    "module": "core",
                    "path": "core/__init__.py",
                    "sha256": _sha("core-module"),
                }
            },
            "workflow_path": "validator/evaluation/workflow.json",
            "workflow_sha256": _sha("workflow"),
            "calibration_shim_sha256": _sha("shim"),
            "comfy_main_sha256": _sha("comfy-main"),
        },
        "runtime": {
            "fresh_comfy_process": True,
            "loopback": "127.0.0.1",
            "database": "memory",
            "api_nodes_disabled": True,
            "isolated_input_output_temp_user": True,
            "offline_environment": True,
            "custom_node_allowlist": ["comfyui-tooling-nodes"],
            "comfy_log_sha256": comfy_log_sha256,
            "comfy_log_bytes": comfy_log_bytes,
            "comfy_history": {"prompt_count": count * 5},
        },
    }


def _score_artifact(
    root: Path,
    receipt_dir: Path,
    row: dict[str, str],
    *,
    weighted_loss: float,
) -> None:
    candidate = Path(row["expected_candidate"])
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if row["task_id"].endswith("zero-baseline"):
        candidate.write_bytes(b"zero-lora")
    else:
        candidate.write_bytes(f"candidate:{row['task_id']}".encode())
    candidate_sha = krea_provenance.file_sha256(candidate)
    row["candidate_sha256"] = candidate_sha
    output = root / "scores" / row["fixture"] / row["task_id"]
    output.mkdir(parents=True)
    (output / "comfy.log").write_bytes(b"prompt execution complete\n")
    result = _result(
        fixture=row["fixture"],
        candidate=candidate,
        candidate_sha=candidate_sha,
        weighted_loss=weighted_loss,
        comfy_log_sha256=krea_provenance.file_sha256(output / "comfy.log"),
        comfy_log_bytes=(output / "comfy.log").stat().st_size,
    )
    (output / "exact-score.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "run-status.env").write_text(
        "\n".join(
            (
                "started_utc=2026-07-31T00:00:00Z",
                "started_unix_ns=100",
                f"fixture={row['fixture']}",
                f"candidate_sha256={candidate_sha}",
                "ended_utc=2026-07-31T00:01:00Z",
                "ended_unix_ns=200",
                "returncode=0",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "inputs.sha256").write_text(
        f"{candidate_sha}  {candidate}\n{_sha('lock')}  /runtime/lock.txt\n",
        encoding="utf-8",
    )
    (output / "evaluator.stderr").write_bytes(b"")
    (output / "evaluator.stdout").write_bytes(b"ok\n")
    (output / "gpu-telemetry.csv").write_bytes(b"time,memory\n0,1\n")
    (output / "resource-usage.txt").write_bytes(b"elapsed=1\n")
    evidence_rows = []
    for name in sorted(recovery._EVIDENCE_FILES):
        evidence_rows.append(
            f"{krea_provenance.file_sha256(output / name)}  /score/{row['task_id']}/{name}"
        )
    (output / "evidence.sha256").write_text(
        "\n".join(evidence_rows) + "\n", encoding="utf-8"
    )
    receipt_dir.mkdir(parents=True, exist_ok=True)
    (receipt_dir / f"{row['task_id']}.env").write_text(
        "\n".join(
            (
                f"task_id={row['task_id']}",
                "state=COMPLETE",
                f"fixture={row['fixture']}",
                f"candidate={candidate}",
                f"candidate_sha256={candidate_sha}",
                f"output_dir={output}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _loss(family: str, *, close_tie: bool) -> float:
    if close_tie and family in {"K1", "K2", "K3"}:
        return {"K1": 0.0500, "K2": 0.0505, "K3": 0.0508}[family]
    return {
        "K0": 0.090,
        "K1": 0.050,
        "K2": 0.065,
        "K3": 0.075,
        "K4": 0.085,
        "K5": 0.095,
        "ZERO": 0.100,
    }[family]


def _complete_index(root: Path, *, close_tie: bool = False) -> tuple[Path, dict]:
    rows = _task_rows(root, complete=True)
    receipts = root / "receipts"
    for row in rows:
        family = recovery._parse_task(row["task_id"])["family"]
        _score_artifact(
            root,
            receipts,
            row,
            weighted_loss=_loss(family, close_tie=close_tie),
        )
    ledger = root / "coverage-ledger.tsv"
    _write_ledger(ledger, rows)
    output = root / "krea-recovery-index.json"
    value = recovery.build_index(
        coverage_ledger=ledger,
        receipt_dir=receipts,
        output=output,
        indexed_at_utc="2026-07-31T01:00:00Z",
    )
    return output, value


def _waiver(path: Path, *, index_path: Path, index: dict) -> dict:
    body = {
        "schema": 1,
        "kind": waiver_freeze.WAIVER_KIND,
        "waiver_id": "week5-krea-recovery-waiver-test",
        "approved_at_utc": "2026-07-31T01:01:00Z",
        "accountable_owner_identity": "Atulya Shetty",
        "owner_identity_assurance": waiver_freeze.OWNER_IDENTITY_ASSURANCE,
        "recovery_index_sha256": index["index_sha256"],
        "recovery_index_file_sha256": krea_provenance.file_sha256(index_path),
        "scope": (
            "use_validated_recovery_scores_for_non_authorizing_D1_D2_finalist_freeze"
        ),
        "claims": dict(recovery.FALSE_CLAIMS),
        "maximum_noncontrol_finalists": 3,
        "independent_agent_review_required": True,
    }
    value = {**body, "waiver_sha256": krea_provenance.canonical_sha256(body)}
    _canonical(path, value)
    return value


def test_index_records_failures_and_only_rc0_exact_scores_are_eligible(tmp_path: Path):
    rows = _task_rows(tmp_path, complete=False)
    selected = next(row for row in rows if row["task_id"] == "d1-k1-step10")
    selected["state"] = "COMPLETE"
    _score_artifact(tmp_path, tmp_path / "receipts", selected, weighted_loss=0.05)
    failed = next(row for row in rows if row["task_id"] == "d1-k1-step20")
    failed_output = tmp_path / "scores" / "failed"
    failed_output.mkdir(parents=True)
    (failed_output / "comfy.log").write_text("failed log\n", encoding="utf-8")
    (tmp_path / "receipts" / f"{failed['task_id']}.env").write_text(
        "\n".join(
            (
                f"task_id={failed['task_id']}",
                "state=FAILED",
                "fixture=D1",
                f"candidate={failed['expected_candidate']}",
                f"candidate_sha256={_sha('not-run')}",
                f"output_dir={failed_output}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    ledger = tmp_path / "coverage-ledger.tsv"
    _write_ledger(ledger, rows)

    index = recovery.build_index(
        coverage_ledger=ledger,
        receipt_dir=tmp_path / "receipts",
        output=tmp_path / "index.json",
        indexed_at_utc="2026-07-31T01:00:00Z",
    )

    assert index["coverage"]["selection_eligible"] == 1
    assert index["coverage"]["selection_gate_ready"] is False
    assert index["claims"] == recovery.FALSE_CLAIMS
    failed_row = next(
        row for row in index["artifacts"] if row["task_id"] == failed["task_id"]
    )
    assert failed_row["selection_eligible"] is False
    assert failed_row["failures"] == ["coverage_or_receipt_incomplete:QUEUED:FAILED"]
    assert [row["name"] for row in failed_row["log_bindings"]] == ["comfy.log"]


def test_freeze_and_fresh_agent_review_are_create_only_and_non_authorizing(
    tmp_path: Path,
):
    index_path, index = _complete_index(tmp_path / "evidence")
    waiver_path = tmp_path / "waiver.json"
    _waiver(waiver_path, index_path=index_path, index=index)
    preparer_path = tmp_path / "preparer.json"
    reviewer_path = tmp_path / "reviewer.json"
    _canonical(
        preparer_path,
        _actor("fresh-recovery-preparer", waiver_freeze.PREPARER_ROLE),
    )
    _canonical(
        reviewer_path,
        _actor("fresh-recovery-reviewer", waiver_freeze.REVIEWER_ROLE),
    )
    freeze_path = tmp_path / "freeze.json"
    freeze = waiver_freeze.freeze_finalists(
        recovery_index_path=index_path,
        waiver_path=waiver_path,
        preparer_actor_path=preparer_path,
        output=freeze_path,
        frozen_at_utc="2026-07-31T01:02:00Z",
    )

    assert freeze["outcome"] == "finalists_frozen"
    assert freeze["finalist_family_ids"] == ["K1", "K2", "K0"]
    assert len([item for item in freeze["finalist_family_ids"] if item != "K0"]) <= 3
    assert freeze["claims"] == recovery.FALSE_CLAIMS
    assert freeze["authority"]["deployment_authorized"] is False
    with pytest.raises(FileExistsError):
        waiver_freeze.freeze_finalists(
            recovery_index_path=index_path,
            waiver_path=waiver_path,
            preparer_actor_path=preparer_path,
            output=freeze_path,
            frozen_at_utc="2026-07-31T01:02:00Z",
        )

    review_path = tmp_path / "review.json"
    review = waiver_freeze.review_finalist_freeze(
        recovery_index_path=index_path,
        waiver_path=waiver_path,
        freeze_path=freeze_path,
        reviewer_actor_path=reviewer_path,
        output=review_path,
        reviewed_at_utc="2026-07-31T01:03:00Z",
    )
    assert review["decision"] == "verified_exact_recomputation"
    assert review["reviewer_is_human"] is False
    assert review["deployment_claimed"] is False
    assert review["claims"] == recovery.FALSE_CLAIMS
    assert (
        waiver_freeze.validate_review(
            recovery_index_path=index_path,
            waiver_path=waiver_path,
            freeze_path=freeze_path,
            review_path=review_path,
        )
        == review
    )


def test_three_way_tie_recomputes_seed_b_and_never_freezes_seed_a_finalists(
    tmp_path: Path,
):
    index_path, index = _complete_index(tmp_path / "evidence", close_tie=True)
    waiver_path = tmp_path / "waiver.json"
    _waiver(waiver_path, index_path=index_path, index=index)
    actor_path = tmp_path / "preparer.json"
    _canonical(actor_path, _actor("fresh-tie-preparer", waiver_freeze.PREPARER_ROLE))
    freeze = waiver_freeze.freeze_finalists(
        recovery_index_path=index_path,
        waiver_path=waiver_path,
        preparer_actor_path=actor_path,
        output=tmp_path / "tie-freeze.json",
        frozen_at_utc="2026-07-31T01:02:00Z",
    )

    assert freeze["outcome"] == "seed_b_required"
    assert freeze["finalist_family_ids"] == []
    assert freeze["checkpoint_rules"] == {}
    assert freeze["seed_b_trigger"]["triggered"] is True
    assert (
        "three_or_more_noncontrols_inside_0.01_band"
        in freeze["seed_b_trigger"]["reasons"]
    )
    assert freeze["seed_b_trigger"]["waiver_cannot_substitute_for_seed_b"] is True


def test_material_cross_fixture_rank_reversal_recomputes_seed_b(tmp_path: Path):
    _, index = _complete_index(tmp_path / "evidence")
    synthetic = deepcopy(index)
    losses = {
        "D1": {"K1": 0.040, "K2": 0.080},
        "D2": {"K1": 0.080, "K2": 0.040},
    }
    for row in synthetic["artifacts"]:
        family = row["family_id"]
        if family in {"K1", "K2"}:
            row["validated_artifact"]["result"]["weighted_loss"] = losses[
                row["fixture_id"]
            ][family]

    derived = waiver_freeze._derive_selection(synthetic)

    assert derived["outcome"] == "seed_b_required"
    assert "material_D1_D2_rank_reversal" in derived["seed_b_trigger"]["reasons"]
    assert derived["finalist_family_ids"] == []


def test_index_revalidation_detects_post_index_result_drift(tmp_path: Path):
    index_path, index = _complete_index(tmp_path / "evidence")
    selected = next(row for row in index["artifacts"] if not row["zero_control"])
    result_path = Path(selected["validated_artifact"]["result"]["path"])
    result_path.write_bytes(result_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="drifted"):
        recovery.validate_index(index_path)


def test_review_rejects_preparer_identity_or_instance_reuse(tmp_path: Path):
    index_path, index = _complete_index(tmp_path / "evidence")
    waiver_path = tmp_path / "waiver.json"
    _waiver(waiver_path, index_path=index_path, index=index)
    preparer = _actor("same-agent", waiver_freeze.PREPARER_ROLE)
    preparer_path = tmp_path / "preparer.json"
    _canonical(preparer_path, preparer)
    freeze_path = tmp_path / "freeze.json"
    waiver_freeze.freeze_finalists(
        recovery_index_path=index_path,
        waiver_path=waiver_path,
        preparer_actor_path=preparer_path,
        output=freeze_path,
        frozen_at_utc="2026-07-31T01:02:00Z",
    )
    reused = deepcopy(preparer)
    reused["role"] = waiver_freeze.REVIEWER_ROLE
    reviewer_path = tmp_path / "reviewer.json"
    _canonical(reviewer_path, reused)

    with pytest.raises(ValueError, match="not independent"):
        waiver_freeze.review_finalist_freeze(
            recovery_index_path=index_path,
            waiver_path=waiver_path,
            freeze_path=freeze_path,
            reviewer_actor_path=reviewer_path,
            output=tmp_path / "review.json",
            reviewed_at_utc="2026-07-31T01:03:00Z",
        )
