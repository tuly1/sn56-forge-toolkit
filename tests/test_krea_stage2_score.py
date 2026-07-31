"""Adversarial tests for the fail-closed Stage-2 exact-score boundary."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from ops.calibration import krea_provenance
from ops.calibration import krea_stage2_score as score


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sha1(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8")).hexdigest()


def _binding(label: str, semantic_key: str) -> dict[str, str]:
    return {
        "file_sha256": _sha(f"{label}-file"),
        semantic_key: _sha(label),
    }


def _checkpoint_selection(
    planned_steps: int, *, numerator: int = 1, denominator: int = 1
) -> dict[str, object]:
    selected_step = score.krea_stage2_execution._selected_checkpoint_step(
        planned_steps=planned_steps,
        numerator=numerator,
        denominator=denominator,
    )
    return {
        "checkpoint_rule_sha256": _sha(
            f"checkpoint-rule-{planned_steps}-{numerator}-{denominator}"
        ),
        "target_fraction": {
            "numerator": numerator,
            "denominator": denominator,
        },
        "selected_step": selected_step,
        "denominator_steps": planned_steps,
        "mapping_rule": score.krea_stage2_execution._CHECKPOINT_MAPPING_RULE,
    }


def _rows() -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "image": f"image-{index}.png",
            "image_sha256": _sha(f"image-{index}"),
            "image_bytes": 100 + index,
            "image_width": 64,
            "image_height": 64,
            "image_format": "PNG",
            "image_mode": "RGB",
            "prompt": f"image-{index}.txt",
            "prompt_sha256": _sha(f"prompt-{index}"),
            "prompt_bytes": 20 + index,
        }
        for index in range(2)
    ]


def _fixture_manifest() -> dict[str, object]:
    rows = _rows()
    identity = {
        "evaluator_order": [row["image"] for row in rows],
        "rows": rows,
    }
    identity["sha256"] = krea_provenance.canonical_sha256(identity)
    return {
        "manifest_sha256": _sha("fixture-manifest"),
        "evaluation_dataset_identity": identity,
    }


def _source() -> dict[str, object]:
    repositories = {
        name: {
            "commit": _sha1(f"{name}-commit"),
            "tree": _sha1(f"{name}-tree"),
            "tracked_worktree_clean": True,
            "nonignored_worktree_clean": True,
        }
        for name in ("god", "comfyui", "tooling_nodes")
    }
    return {
        **repositories,
        "expected_commits": {
            name: repositories[name]["commit"] for name in repositories
        },
        "god_import_bindings": {
            "validator.core": {
                "module": "validator.core",
                "path": "validator/core.py",
                "sha256": _sha("validator-core"),
            }
        },
        "workflow_path": "validator/evaluation/workflow.json",
        "workflow_sha256": _sha("workflow"),
        "calibration_shim_sha256": _sha("shim"),
        "comfy_main_sha256": _sha("comfy-main"),
    }


def _runtime() -> dict[str, object]:
    return {
        "fresh_comfy_process": True,
        "loopback": "127.0.0.1",
        "port": 8188,
        "cache": "comfy_default_fresh_process",
        "database": "memory",
        "api_nodes_disabled": True,
        "isolated_input_output_temp_user": True,
        "offline_environment": True,
        "custom_node_allowlist": ["comfyui-tooling-nodes"],
        "startup_timeout_s": 30.0,
        "evaluation_timeout_s": 300.0,
        "shutdown_timeout_s": 10.0,
        "shutdown": {
            "returncode": -2,
            "stop_signal": "SIGINT",
            "forced": False,
        },
        "python": {"executable": "/venv/bin/python", "python": "3.11.9"},
        "driver_python": {"executable": "/usr/bin/python3", "python": "3.11.9"},
        "comfy_system_stats": {"system": {"os": "posix"}},
        "comfy_history": {
            "prompt_count": 8,
            "history_sha256": _sha("history"),
        },
        "comfy_log": "/score/evaluator.comfy.log",
        "comfy_log_sha256": _sha("comfy-log"),
        "comfy_log_bytes": 100,
    }


def _result(candidate: Path, dataset_path: Path) -> dict[str, object]:
    candidate_sha = krea_provenance.file_sha256(candidate)
    rows = _rows()
    text = [0.1, 0.2]
    blank = [0.3, 0.4]
    scored = [
        {
            **row,
            "text_guided_loss": text[index],
            "blank_prompt_loss": blank[index],
        }
        for index, row in enumerate(rows)
    ]
    text_mean = sum(text) / len(text)
    blank_mean = sum(blank) / len(blank)
    weight = 0.25
    return {
        "schema": 2,
        "evaluator": "god_krea2_img2img_exact",
        "candidate": candidate.name,
        "candidate_sha256": candidate_sha,
        "candidate_bytes": candidate.stat().st_size,
        "staged_candidate_sha256": candidate_sha,
        "comfy_lora_name": f"candidate-{candidate_sha}.safetensors",
        "model_type": "krea2",
        "dataset": str(dataset_path),
        "dataset_sha256": _fixture_manifest()["evaluation_dataset_identity"]["sha256"],
        "image_count": len(rows),
        "scored_rows": scored,
        "base_name": "krea2_raw_fp8_scaled.safetensors",
        "asset_sha256": {name: _sha(name) for name in score.ASSET_NAMES},
        "asset_bytes": {
            "diffusion_model": 1000,
            "text_encoder": 2000,
            "vae": 3000,
        },
        "steps": 28,
        "cfg": 1.0,
        "denoise": 0.85,
        "generations": 2,
        "master_seed": 42,
        "seeds": [42, 43],
        "text_guided_losses": text,
        "blank_prompt_losses": blank,
        "text_mean": text_mean,
        "blank_mean": blank_mean,
        "text_weight": weight,
        "weighted_loss": weight * text_mean + (1.0 - weight) * blank_mean,
        "direction": "min",
        "elapsed_s": 1.25,
        "source": _source(),
        "runtime": _runtime(),
    }


def _candidate_row(family: str, candidate: Path, ordinal: int) -> dict[str, object]:
    digest = krea_provenance.file_sha256(candidate)
    step = ordinal + 1
    return {
        "family_id": family,
        "training_candidate_id": f"training-{family}",
        "execution_plan_sha256": _sha(f"execution-plan-{family}"),
        "execution_approval_sha256": _sha(f"execution-approval-{family}"),
        "run_completion_sha256": _sha(f"run-completion-{family}"),
        "run_evidence_file_sha256": _sha(f"run-file-{family}"),
        "run_evidence_sha256": _sha(f"run-evidence-{family}"),
        "mechanics": deepcopy(score._RUN_MECHANICS),
        "candidate_id": f"candidate-{family}",
        "candidate_sha256": digest,
        "candidate_bytes": candidate.stat().st_size,
        "checkpoint_rule_sha256": _sha(f"checkpoint-rule-{family}"),
        "checkpoint_target_fraction": {
            "numerator": 1,
            "denominator": 10,
        },
        "checkpoint_mapping_rule": score.krea_stage2_execution._CHECKPOINT_MAPPING_RULE,
        "step": step,
        "fraction_numerator": step,
        "fraction_denominator": 10,
    }


def _plan_payload(
    *,
    dataset_path: Path,
    evaluator_result: dict[str, object],
    candidates: list[dict[str, object]],
    phase: str = "confirmation",
) -> dict[str, object]:
    fixture = _fixture_manifest()
    boundary = phase == "boundary"
    return {
        "schema": 1,
        "kind": score.PLAN_KIND,
        "phase": phase,
        "cell_id": "B-0p5-small" if boundary else "C1-A",
        "fixture_id": "B-0p5-small" if boundary else "C1",
        "seed_role": "A",
        "seed": 42565431,
        "hours": "0.5" if boundary else "0.75",
        "candidate_family_id": "K1",
        "public_reference_family_ids": ["K2", "K3", "K4"],
        "control_family_id": "K0",
        "candidates": candidates,
        "fixture_manifest": {
            "file_sha256": _sha("fixture-file"),
            "manifest_sha256": fixture["manifest_sha256"],
        },
        "evaluation_dataset_sha256": fixture["evaluation_dataset_identity"]["sha256"],
        "evaluation_dataset_path": str(dataset_path),
        "evaluation_row_count": len(fixture["evaluation_dataset_identity"]["rows"]),
        "evaluator_contract": score.evaluator_contract_from_result(evaluator_result),
        "waiver_finalist_freeze": _binding("freeze", "freeze_sha256"),
        "confirmation_materialization": _binding(
            "materialization", "materialization_sha256"
        ),
        "owner_ratification": _binding("ratification", "ratification_sha256"),
        "gpu_execution_authorization": _binding(
            "gpu-authorization", "gpu_execution_authorization_sha256"
        ),
        "production_identity": _binding(
            "production-identity", "production_identity_sha256"
        ),
        "production_image_id": f"sha256:{_sha('production-image')}",
        "created_at_utc": "2026-07-31T00:00:00Z",
        "fallback_allowed": False,
        "release_authorized": False,
        "production_mutation_authorized": False,
    }


def _reseal(value: dict, digest_key: str) -> dict:
    body = {key: item for key, item in value.items() if key != digest_key}
    return {**body, digest_key: krea_provenance.canonical_sha256(body)}


def _write_result(path: Path, result: dict[str, object]) -> None:
    path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")


@pytest.fixture(autouse=True)
def _accept_minimal_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(score.krea_fixture, "validate_manifest", lambda value: value)


def _confirmation_case(tmp_path: Path) -> tuple[
    dict[str, object],
    dict[str, Path],
    Path,
]:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    paths: dict[str, Path] = {}
    rows = []
    for ordinal, family in enumerate(("K0", "K1", "K2", "K3", "K4")):
        path = tmp_path / f"{family}.safetensors"
        path.write_bytes(f"candidate:{family}".encode("ascii"))
        paths[family] = path
        rows.append(_candidate_row(family, path, ordinal))
    exemplar = _result(paths["K1"], dataset)
    plan = score.seal_plan(
        _plan_payload(
            dataset_path=dataset,
            evaluator_result=exemplar,
            candidates=rows,
        )
    )
    return plan, paths, dataset


def _receipt(
    tmp_path: Path,
    *,
    plan: dict[str, object],
    family: str,
    candidate: Path,
    dataset: Path,
) -> dict[str, object]:
    result_path = tmp_path / f"result-{family}.json"
    _write_result(result_path, _result(candidate, dataset))
    return score.build_receipt(
        plan=plan,
        family_id=family,
        candidate_path=candidate,
        fixture_manifest=_fixture_manifest(),
        fixture_manifest_file_sha256=_sha("fixture-file"),
        result_path=result_path,
        status_file_sha256=_sha(f"status-{family}"),
        evidence_manifest_file_sha256=_sha(f"manifest-{family}"),
        completed_at_utc="2026-07-31T00:10:00Z",
    )


def test_exact_score_round_trip_recomputes_rows_assets_and_two_prompt_modes(
    tmp_path: Path,
) -> None:
    plan, paths, dataset = _confirmation_case(tmp_path)
    receipts = [
        _receipt(
            tmp_path,
            plan=plan,
            family=family,
            candidate=paths[family],
            dataset=dataset,
        )
        for family in ("K0", "K1", "K2", "K3", "K4")
    ]
    assert receipts[1]["result"]["prompt_count"] == 8
    assert receipts[1]["result"]["row_identity_sha256"] == (
        krea_provenance.canonical_sha256(_rows())
    )
    aggregate = score.build_aggregate(
        plan=plan,
        receipts=list(reversed(receipts)),
        emitted_at_utc="2026-07-31T00:20:00Z",
    )
    assert [row["family_id"] for row in aggregate["receipts"]] == [
        "K0",
        "K1",
        "K2",
        "K3",
        "K4",
    ]
    assert aggregate["all_candidates_scored"] is True
    assert aggregate["release_authorized"] is False
    assert aggregate["production_mutation_authorized"] is False
    score_files = {
        family: {
            "candidate_path": paths[family],
            "result_path": tmp_path / f"result-{family}.json",
        }
        for family in paths
    }
    assert (
        score.validate_aggregate_with_score_files(
            aggregate,
            plan=plan,
            fixture_manifest=_fixture_manifest(),
            fixture_manifest_file_sha256=_sha("fixture-file"),
            score_files_by_family=score_files,
        )
        == aggregate
    )


def test_strict_score_replay_rejects_a_resealed_summary_forgery(tmp_path: Path) -> None:
    plan, paths, dataset = _confirmation_case(tmp_path)
    receipt = _receipt(
        tmp_path,
        plan=plan,
        family="K1",
        candidate=paths["K1"],
        dataset=dataset,
    )
    forged = deepcopy(receipt)
    forged["result"]["weighted_loss"] = 0.01
    forged = _reseal(forged, "receipt_sha256")
    # The portable validator can only check bindings and ranges.  The decision
    # path must use the strict replay API below to re-open the bound bytes.
    assert score.validate_receipt(forged, plan=plan) == forged
    with pytest.raises(ValueError, match="recomputed exact-score bytes"):
        score.validate_receipt_with_score_files(
            forged,
            plan=plan,
            candidate_path=paths["K1"],
            fixture_manifest=_fixture_manifest(),
            fixture_manifest_file_sha256=_sha("fixture-file"),
            result_path=tmp_path / "result-K1.json",
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda result: result.__setitem__("dataset", "/score/other-dataset"),
            "candidate/evaluator identity",
        ),
        (
            lambda result: result["scored_rows"][0].__setitem__("image_width", 65),
            "rows differ",
        ),
        (
            lambda result: result["scored_rows"][0].__setitem__("extra", True),
            "keys differ",
        ),
        (
            lambda result: result["text_guided_losses"].__setitem__(0, 0.9),
            "row and loss arrays",
        ),
        (
            lambda result: result.__setitem__("weighted_loss", 0.1),
            "aggregate losses",
        ),
        (
            lambda result: result["blank_prompt_losses"].__setitem__(0, 1.1),
            r"outside \[0,1\]",
        ),
        (
            lambda result: result["asset_sha256"].__setitem__("vae", _sha("other")),
            "evaluator contract",
        ),
        (
            lambda result: result["source"]["expected_commits"].__setitem__(
                "god", _sha1("other")
            ),
            "expected commits",
        ),
        (
            lambda result: result["runtime"]["comfy_history"].__setitem__(
                "prompt_count", 4
            ),
            "prompt count",
        ),
    ],
)
def test_result_tampering_is_rejected(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    plan, paths, dataset = _confirmation_case(tmp_path)
    result = _result(paths["K1"], dataset)
    mutation(result)
    path = tmp_path / "tampered-result.json"
    _write_result(path, result)
    with pytest.raises(ValueError, match=match):
        score.build_receipt(
            plan=plan,
            family_id="K1",
            candidate_path=paths["K1"],
            fixture_manifest=_fixture_manifest(),
            fixture_manifest_file_sha256=_sha("fixture-file"),
            result_path=path,
            status_file_sha256=_sha("status"),
            evidence_manifest_file_sha256=_sha("manifest"),
            completed_at_utc="2026-07-31T00:10:00Z",
        )


def test_evaluator_contract_requires_structured_assets_and_runtime_source(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    candidate = tmp_path / "candidate.safetensors"
    candidate.write_bytes(b"candidate")
    result = _result(candidate, dataset)
    contract = score.evaluator_contract_from_result(result)
    assert set(contract["asset_sha256"]) == set(score.ASSET_NAMES)
    assert "runtime_source_sha256" in contract

    bad = deepcopy(contract)
    bad["asset_sha256"] = _sha("scalar")
    bad = _reseal(bad, "contract_sha256")
    with pytest.raises(ValueError, match="object"):
        score._evaluator_contract(bad)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "unordered"])
def test_plan_candidate_coverage_is_exhaustive_and_exact(
    tmp_path: Path, mutation: str
) -> None:
    plan, paths, _dataset = _confirmation_case(tmp_path)
    tampered = deepcopy(plan)
    if mutation == "missing":
        tampered["candidates"].pop()
    elif mutation == "extra":
        extra_path = tmp_path / "K5.safetensors"
        extra_path.write_bytes(b"candidate:K5")
        tampered["candidates"].append(_candidate_row("K5", extra_path, 7))
    elif mutation == "duplicate":
        tampered["candidates"][1]["candidate_sha256"] = tampered["candidates"][0][
            "candidate_sha256"
        ]
    else:
        tampered["candidates"][0], tampered["candidates"][1] = (
            tampered["candidates"][1],
            tampered["candidates"][0],
        )
    tampered = _reseal(tampered, "plan_sha256")
    with pytest.raises(ValueError):
        score.validate_plan(tampered)

    assert paths


def test_boundary_plan_rejects_an_extra_comparator(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    candidate = tmp_path / "K1.safetensors"
    candidate.write_bytes(b"candidate:K1")
    result = _result(candidate, dataset)
    rows = [_candidate_row("K1", candidate, 0)]
    plan = score.seal_plan(
        _plan_payload(
            dataset_path=dataset,
            evaluator_result=result,
            candidates=rows,
            phase="boundary",
        )
    )
    extra = deepcopy(plan)
    extra_path = tmp_path / "K0.safetensors"
    extra_path.write_bytes(b"candidate:K0")
    extra["candidates"].insert(0, _candidate_row("K0", extra_path, 1))
    extra = _reseal(extra, "plan_sha256")
    with pytest.raises(ValueError, match="exhaust"):
        score.validate_plan(extra)


def test_candidate_checkpoint_target_must_be_positive_and_reduced(
    tmp_path: Path,
) -> None:
    plan, _paths, _dataset = _confirmation_case(tmp_path)
    for target in (
        {"numerator": 0, "denominator": 1},
        {"numerator": 2, "denominator": 4},
    ):
        tampered = deepcopy(plan)
        tampered["candidates"][0]["checkpoint_target_fraction"] = target
        tampered = _reseal(tampered, "plan_sha256")
        with pytest.raises(ValueError, match="positive reduced fraction"):
            score.validate_plan(tampered)


def test_fallback_and_release_claims_fail_even_when_resealed(tmp_path: Path) -> None:
    plan, paths, dataset = _confirmation_case(tmp_path)
    bad_plan = deepcopy(plan)
    bad_plan["candidates"][0]["mechanics"]["fallback_used"] = True
    bad_plan = _reseal(bad_plan, "plan_sha256")
    with pytest.raises(ValueError, match="mechanics"):
        score.validate_plan(bad_plan)

    release_plan = deepcopy(plan)
    release_plan["release_authorized"] = True
    release_plan = _reseal(release_plan, "plan_sha256")
    with pytest.raises(ValueError, match="overclaims"):
        score.validate_plan(release_plan)

    receipt = _receipt(
        tmp_path,
        plan=plan,
        family="K1",
        candidate=paths["K1"],
        dataset=dataset,
    )
    bad_receipt = deepcopy(receipt)
    bad_receipt["fallback_used"] = True
    bad_receipt = _reseal(bad_receipt, "receipt_sha256")
    with pytest.raises(ValueError, match="failed or overclaims"):
        score.validate_receipt(bad_receipt, plan=plan)


def test_candidate_result_and_dataset_symlinks_are_rejected(tmp_path: Path) -> None:
    plan, paths, dataset = _confirmation_case(tmp_path)
    result_path = tmp_path / "result.json"
    _write_result(result_path, _result(paths["K1"], dataset))
    result_link = tmp_path / "result-link.json"
    result_link.symlink_to(result_path)
    candidate_link = tmp_path / "candidate-link.safetensors"
    candidate_link.symlink_to(paths["K1"])
    kwargs = {
        "plan": plan,
        "family_id": "K1",
        "fixture_manifest": _fixture_manifest(),
        "fixture_manifest_file_sha256": _sha("fixture-file"),
        "status_file_sha256": _sha("status"),
        "evidence_manifest_file_sha256": _sha("manifest"),
        "completed_at_utc": "2026-07-31T00:10:00Z",
    }
    with pytest.raises(ValueError, match="symlink"):
        score.build_receipt(
            **kwargs,
            candidate_path=candidate_link,
            result_path=result_path,
        )
    with pytest.raises(ValueError, match="symlink"):
        score.build_receipt(
            **kwargs,
            candidate_path=paths["K1"],
            result_path=result_link,
        )

    dataset_link = tmp_path / "dataset-link"
    dataset_link.symlink_to(dataset, target_is_directory=True)
    payload = _plan_payload(
        dataset_path=dataset_link,
        evaluator_result=_result(paths["K1"], dataset),
        candidates=plan["candidates"],
    )
    with pytest.raises(ValueError, match="symlink"):
        score.seal_plan(payload)


def _run_evidence(
    *, plan_payload: dict[str, object], candidate: Path
) -> dict[str, object]:
    candidate_sha = krea_provenance.file_sha256(candidate)
    return {
        "phase": plan_payload["phase"],
        "cell_id": plan_payload["cell_id"],
        "fixture_id": plan_payload["fixture_id"],
        "seed_role": plan_payload["seed_role"],
        "seed": plan_payload["seed"],
        "hours": plan_payload["hours"],
        "training_candidate_id": "training-K1",
        "execution_plan": {
            "file_sha256": _sha("execution-plan-file"),
            "plan_sha256": _sha("execution-plan"),
        },
        "execution_approval": {
            "file_sha256": _sha("execution-approval-file"),
            "approval_sha256": _sha("execution-approval"),
        },
        "run_completion": {
            "file_sha256": _sha("run-completion-file"),
            "completion_sha256": _sha("run-completion"),
        },
        "fixture_manifest": plan_payload["fixture_manifest"],
        "waiver_finalist_freeze": plan_payload["waiver_finalist_freeze"],
        "confirmation_materialization": plan_payload["confirmation_materialization"],
        "owner_ratification": plan_payload["owner_ratification"],
        "gpu_execution_authorization": plan_payload["gpu_execution_authorization"],
        "production_identity": plan_payload["production_identity"],
        "production_image_id": plan_payload["production_image_id"],
        "candidate_artifacts": [
            {
                "path": "checkpoints/last.safetensors",
                "bytes": candidate.stat().st_size,
                "sha256": candidate_sha,
            }
        ],
        "mechanics": deepcopy(score._RUN_MECHANICS),
        "natural_completion": True,
        "fallback_used": False,
        "evidence_sha256": _sha("run-evidence"),
    }


def _private_receipts(family: str) -> dict[str, dict[str, str]]:
    return {
        "config_control": {
            "file_sha256": _sha(f"config-control-file-{family}"),
            "receipt_sha256": _sha(f"config-control-receipt-{family}"),
            "config_sha256": _sha(f"effective-config-{family}"),
        },
        "training_terminal": {
            "file_sha256": _sha(f"training-terminal-file-{family}"),
            "receipt_sha256": _sha(f"training-terminal-receipt-{family}"),
        },
        "checkpoint_selection": {
            "file_sha256": _sha(f"checkpoint-selection-file-{family}"),
            "receipt_sha256": _sha(f"checkpoint-selection-receipt-{family}"),
        },
    }


def _run_completion(family: str) -> dict[str, object]:
    receipts = _private_receipts(family)
    return {
        "config_control_receipt": receipts["config_control"],
        "training_terminal_receipt": receipts["training_terminal"],
        "checkpoint_selection_receipt": receipts["checkpoint_selection"],
    }


def _mock_private_receipts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        score.krea_stage2_execution,
        "validate_private_run_receipts",
        lambda plan: _private_receipts(plan["calibration_profile"]),
    )


def test_run_mechanics_requires_terminal_step_completion() -> None:
    assert score._run_mechanics(deepcopy(score._RUN_MECHANICS)) == (
        score._RUN_MECHANICS
    )
    incomplete = deepcopy(score._RUN_MECHANICS)
    incomplete["planned_steps_completed"] = False
    with pytest.raises(ValueError, match="mechanics"):
        score._run_mechanics(incomplete)


def test_candidate_builder_and_strict_plan_replay_all_run_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    candidate = tmp_path / "last.safetensors"
    candidate.write_bytes(b"validated-last")
    exemplar = _result(candidate, dataset)
    initial = _plan_payload(
        dataset_path=dataset,
        evaluator_result=exemplar,
        candidates=[],
        phase="boundary",
    )
    evidence = _run_evidence(plan_payload=initial, candidate=candidate)
    evidence_path = tmp_path / "run-evidence.json"
    evidence_path.write_bytes(krea_provenance.canonical_bytes(evidence) + b"\n")
    calls = []

    def validate(value, *, plan, approval, completion):
        calls.append((plan, approval, completion))
        return value

    monkeypatch.setattr(
        score.krea_stage2_training_evidence, "validate_run_evidence", validate
    )
    _mock_private_receipts(monkeypatch)
    control = {
        "run_evidence_path": evidence_path,
        "execution_plan": {
            "training_candidate_id": "training-K1",
            "calibration_profile": "K1",
            "planned_steps": 72,
            "checkpoint_selection": {
                **_checkpoint_selection(72, numerator=1, denominator=2),
                "checkpoint_rule_sha256": _sha("checkpoint-rule-K1"),
            },
            "candidate_universe": [{"candidate_id": "training-K1", "family_id": "K1"}],
        },
        "execution_approval": {"control": "approval"},
        "run_completion": _run_completion("K1"),
        "candidate_path": candidate,
    }
    row = score.build_candidate_row(
        family_id="K1",
        candidate_id="candidate-K1",
        **control,
    )
    assert row["mechanics"] == score._RUN_MECHANICS
    assert row["mechanics"]["planned_steps_completed"] is True
    assert row["candidate_sha256"] == krea_provenance.file_sha256(candidate)
    assert row["checkpoint_rule_sha256"] == _sha("checkpoint-rule-K1")
    assert row["checkpoint_target_fraction"] == {"numerator": 1, "denominator": 2}
    assert (
        row["checkpoint_mapping_rule"]
        == score.krea_stage2_execution._CHECKPOINT_MAPPING_RULE
    )
    assert (row["step"], row["fraction_numerator"], row["fraction_denominator"]) == (
        36,
        36,
        72,
    )
    plan = score.seal_plan({**initial, "candidates": [row]})
    assert (
        score.validate_plan_with_run_controls(plan, controls_by_family={"K1": control})
        == plan
    )
    assert len(calls) == 2

    forged = deepcopy(plan)
    forged["candidates"][0].update(
        step=27, fraction_numerator=27, fraction_denominator=72
    )
    forged = _reseal(forged, "plan_sha256")
    assert score.validate_plan(forged) == forged
    with pytest.raises(ValueError, match="differs from the score plan"):
        score.validate_plan_with_run_controls(
            forged, controls_by_family={"K1": control}
        )

    forged_binding = deepcopy(plan)
    forged_binding["candidates"][0]["checkpoint_rule_sha256"] = _sha(
        "different-checkpoint-rule"
    )
    forged_binding = _reseal(forged_binding, "plan_sha256")
    assert score.validate_plan(forged_binding) == forged_binding
    with pytest.raises(ValueError, match="differs from the score plan"):
        score.validate_plan_with_run_controls(
            forged_binding, controls_by_family={"K1": control}
        )

    forged_target = deepcopy(plan)
    forged_target["candidates"][0]["checkpoint_target_fraction"] = {
        "numerator": 1,
        "denominator": 4,
    }
    forged_target = _reseal(forged_target, "plan_sha256")
    assert score.validate_plan(forged_target) == forged_target
    with pytest.raises(ValueError, match="differs from the score plan"):
        score.validate_plan_with_run_controls(
            forged_target, controls_by_family={"K1": control}
        )

    drifted_control = deepcopy(control)
    drifted_control["run_completion"]["training_terminal_receipt"]["receipt_sha256"] = (
        _sha("different-terminal-receipt")
    )
    with pytest.raises(ValueError, match="live terminal/config/selection receipts"):
        score.build_candidate_row(
            family_id="K1",
            candidate_id="candidate-K1",
            **drifted_control,
        )

    drifted_selection = deepcopy(control)
    drifted_selection["run_completion"]["checkpoint_selection_receipt"][
        "receipt_sha256"
    ] = _sha("different-selection-receipt")
    with pytest.raises(ValueError, match="live terminal/config/selection receipts"):
        score.build_candidate_row(
            family_id="K1",
            candidate_id="candidate-K1",
            **drifted_selection,
        )

    extended_selection = deepcopy(control)
    extended_selection["run_completion"]["checkpoint_selection_receipt"][
        "selected_step"
    ] = 36
    with pytest.raises(ValueError, match="keys differ"):
        score.build_candidate_row(
            family_id="K1",
            candidate_id="candidate-K1",
            **extended_selection,
        )

    aliased_candidate = tmp_path / "selected-step-36.safetensors"
    aliased_candidate.write_bytes(candidate.read_bytes())
    aliased_control = {**control, "candidate_path": aliased_candidate}
    with pytest.raises(ValueError, match="promoted last.safetensors"):
        score.build_candidate_row(
            family_id="K1",
            candidate_id="candidate-K1",
            **aliased_control,
        )

    receipts_without_selection = _private_receipts("K1")
    receipts_without_selection.pop("checkpoint_selection")
    monkeypatch.setattr(
        score.krea_stage2_execution,
        "validate_private_run_receipts",
        lambda _plan: receipts_without_selection,
    )
    completion_without_selection = deepcopy(control)
    completion_without_selection["run_completion"].pop("checkpoint_selection_receipt")
    with pytest.raises(ValueError, match="live terminal/config/selection receipts"):
        score.build_candidate_row(
            family_id="K1",
            candidate_id="candidate-K1",
            **completion_without_selection,
        )

    with pytest.raises(ValueError, match="exhaust"):
        score.validate_plan_with_run_controls(
            plan, controls_by_family={"K1": control, "K2": control}
        )


def test_strict_replay_rejects_reference_control_borrowed_from_another_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    exemplar = tmp_path / "exemplar.safetensors"
    exemplar.write_bytes(b"exemplar")
    initial = _plan_payload(
        dataset_path=dataset,
        evaluator_result=_result(exemplar, dataset),
        candidates=[],
    )
    monkeypatch.setattr(
        score.krea_stage2_training_evidence,
        "validate_run_evidence",
        lambda value, **_kwargs: value,
    )
    _mock_private_receipts(monkeypatch)
    rows = []
    controls = {}
    evidences = {}
    for ordinal, family in enumerate(("K0", "K1", "K2", "K3", "K4")):
        candidate = tmp_path / family / "last.safetensors"
        candidate.parent.mkdir()
        candidate.write_bytes(f"candidate:{family}".encode("ascii"))
        evidence = _run_evidence(plan_payload=initial, candidate=candidate)
        evidence["training_candidate_id"] = f"training-{family}"
        evidence["execution_plan"]["plan_sha256"] = _sha(f"plan-{family}")
        evidence["execution_approval"]["approval_sha256"] = _sha(f"approval-{family}")
        evidence["run_completion"]["completion_sha256"] = _sha(f"completion-{family}")
        evidence["evidence_sha256"] = _sha(f"evidence-{family}")
        evidence_path = tmp_path / f"evidence-{family}.json"
        evidence_path.write_bytes(krea_provenance.canonical_bytes(evidence) + b"\n")
        control = {
            "run_evidence_path": evidence_path,
            "execution_plan": {
                "training_candidate_id": f"training-{family}",
                "calibration_profile": family,
                "planned_steps": ordinal + 1,
                "checkpoint_selection": _checkpoint_selection(ordinal + 1),
                "candidate_universe": [
                    {
                        "candidate_id": f"training-{family}",
                        "family_id": family,
                    }
                ],
            },
            "execution_approval": {"family": family},
            "run_completion": _run_completion(family),
            "candidate_path": candidate,
        }
        rows.append(
            score.build_candidate_row(
                family_id=family,
                candidate_id=f"candidate-{family}",
                **control,
            )
        )
        controls[family] = control
        evidences[family] = evidence
    plan = score.seal_plan({**initial, "candidates": rows})
    assert (
        score.validate_plan_with_run_controls(plan, controls_by_family=controls) == plan
    )

    borrowed = deepcopy(evidences["K2"])
    borrowed["cell_id"] = "C2-A"
    borrowed["fixture_id"] = "C2"
    borrowed["evidence_sha256"] = _sha("borrowed-K2")
    borrowed_path = tmp_path / "borrowed-K2.json"
    borrowed_path.write_bytes(krea_provenance.canonical_bytes(borrowed) + b"\n")
    borrowed_control = {
        **controls["K2"],
        "run_evidence_path": borrowed_path,
    }
    borrowed_row = score.build_candidate_row(
        family_id="K2",
        candidate_id="candidate-K2",
        **borrowed_control,
    )
    tampered = deepcopy(plan)
    tampered["candidates"][2] = borrowed_row
    tampered = _reseal(tampered, "plan_sha256")
    assert score.validate_plan(tampered) == tampered
    tampered_controls = {**controls, "K2": borrowed_control}
    with pytest.raises(ValueError, match="K2.*score-plan cell"):
        score.validate_plan_with_run_controls(
            tampered, controls_by_family=tampered_controls
        )


def test_candidate_builder_rejects_fallback_or_nonfinal_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    candidate = tmp_path / "candidate.safetensors"
    candidate.write_bytes(b"candidate")
    initial = _plan_payload(
        dataset_path=dataset,
        evaluator_result=_result(candidate, dataset),
        candidates=[],
        phase="boundary",
    )
    evidence = _run_evidence(plan_payload=initial, candidate=candidate)
    evidence["mechanics"]["fallback_used"] = True
    evidence["fallback_used"] = True
    evidence_path = tmp_path / "run-evidence.json"
    evidence_path.write_bytes(krea_provenance.canonical_bytes(evidence) + b"\n")
    monkeypatch.setattr(
        score.krea_stage2_training_evidence,
        "validate_run_evidence",
        lambda value, **_kwargs: value,
    )
    with pytest.raises(ValueError, match="mechanics"):
        score.build_candidate_row(
            family_id="K1",
            candidate_id="candidate-K1",
            run_evidence_path=evidence_path,
            execution_plan={},
            execution_approval={},
            run_completion={},
            candidate_path=candidate,
        )


@pytest.mark.parametrize(
    ("phase", "family", "selected_family", "profile", "match"),
    [
        (
            "confirmation",
            "K2",
            "K1",
            "K1",
            "score family differs",
        ),
        (
            "confirmation",
            "K1",
            "K1",
            "K2",
            "calibration profile",
        ),
        (
            "boundary",
            "K1",
            "K1",
            None,
            "calibration profile",
        ),
    ],
)
def test_candidate_builder_rejects_family_or_calibration_profile_relabeling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    family: str,
    selected_family: str,
    profile: str | None,
    match: str,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    candidate = tmp_path / "candidate.safetensors"
    candidate.write_bytes(b"candidate")
    initial = _plan_payload(
        dataset_path=dataset,
        evaluator_result=_result(candidate, dataset),
        candidates=[],
        phase=phase,
    )
    evidence = _run_evidence(plan_payload=initial, candidate=candidate)
    evidence["training_candidate_id"] = "selected-candidate"
    evidence_path = tmp_path / "run-evidence.json"
    evidence_path.write_bytes(krea_provenance.canonical_bytes(evidence) + b"\n")
    monkeypatch.setattr(
        score.krea_stage2_training_evidence,
        "validate_run_evidence",
        lambda value, **_kwargs: value,
    )
    with pytest.raises(ValueError, match=match):
        score.build_candidate_row(
            family_id=family,
            candidate_id=f"candidate-{family}",
            run_evidence_path=evidence_path,
            execution_plan={
                "training_candidate_id": "selected-candidate",
                "calibration_profile": profile,
                "planned_steps": 1,
                "candidate_universe": [
                    {
                        "candidate_id": "selected-candidate",
                        "family_id": selected_family,
                    }
                ],
            },
            execution_approval={},
            run_completion={},
            candidate_path=candidate,
        )
