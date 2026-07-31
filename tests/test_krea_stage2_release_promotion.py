from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from ops.calibration import krea_provenance
from ops.calibration import krea_stage2_production_identity as production
from ops.calibration import krea_stage2_release_promotion as promotion


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


_BASE = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=2)


def _time(seconds: int) -> str:
    return (_BASE + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _identity(*, proposed: bool) -> dict:
    return production.build(
        forge={
            "commit_sha1": ("3" if proposed else "1") * 40,
            "tree_sha1": ("4" if proposed else "2") * 40,
            "worktree_state": "clean-including-untracked",
        },
        container_image={
            "image_id": "sha256:" + (("6" if proposed else "5") * 64),
            "repo_digest": "registry.example/forge@sha256:"
            + (("8" if proposed else "7") * 64),
        },
        dockerfile={
            "path": production.DOCKERFILE_PATH,
            "sha256": _sha("dockerfile"),
            "bytes": 100,
        },
        runtime_inputs=[
            {"path": path, "sha256": _sha(path), "bytes": index + 1}
            for index, path in enumerate(production.RUNTIME_INPUT_PATHS)
        ],
        base_model={
            "model_id": production.KREA_MODEL_ID,
            "revision": "a" * 40,
            "training_identity_sha256": _sha("assets"),
            "asset_attestation_sha256": _sha("attestation"),
            "text_encoder_id": production.KREA_TEXT_ENCODER_ID,
            "text_encoder_revision": "b" * 40,
        },
        runtime_contract={
            "runtime_identity_sha256": _sha("runtime"),
            "venv_tree_manifest_sha256": _sha("venv"),
            "trainer_identity_sha256": _sha("trainer"),
            "measurement_tool_sha256": _sha("measurement"),
            "jit_enabled": True,
        },
        captured_at_utc=_time(20 if proposed else 0),
    )


def _checkpoint_bindings(steps: int) -> tuple[dict, dict]:
    save_every = (steps + 7) // 8
    candidates = list(range(save_every, steps, save_every)) + [steps]
    candidates = sorted(set(candidates))
    selected = min(
        candidates,
        key=lambda step: (abs(step * 8 - steps * 7), step),
    )
    runtime = {
        "schema": 1,
        "mapping_rule": promotion.krea_stage2_execution._CHECKPOINT_MAPPING_RULE,
        "target_fraction": {"numerator": 7, "denominator": 8},
        "planned_steps": steps,
        "selected_step": selected,
        "candidate_steps": candidates,
    }
    plan = {
        "checkpoint_rule_sha256": _sha("K1-checkpoint-rule"),
        "target_fraction": {"numerator": 7, "denominator": 8},
        "selected_step": selected,
        "denominator_steps": steps,
        "mapping_rule": promotion.krea_stage2_execution._CHECKPOINT_MAPPING_RULE,
    }
    return runtime, plan


def _explicit_config(cell: str, *, seed: int, steps: int) -> dict:
    runtime_selection, _plan_selection = _checkpoint_bindings(steps)
    return {
        "job": "extension",
        "config": {
            "name": f"stage2-{cell.lower()}-k1",
            "process": [
                {
                    "type": "diffusion_trainer",
                    "training_seed": seed,
                    "network": {"linear": 32, "linear_alpha": 32},
                    "train": {"steps": steps, "lr": 0.0002},
                    "save": {"save_every": (steps + 7) // 8},
                }
            ],
        },
        "meta": {
            "name": "krea2_lora",
            "version": "1.0",
            "forge_krea_calibration_profile": {
                "calibration_only": True,
                "profile_id": "K1",
                "release_selected": False,
            },
            "forge_krea_checkpoint_selection": runtime_selection,
        },
    }


def _release_config(explicit: dict) -> dict:
    value = deepcopy(explicit)
    value["config"]["process"][0].pop("training_seed")
    value["meta"].pop("forge_krea_calibration_profile")
    return value


def _canonical_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(krea_provenance.canonical_bytes(value) + b"\n")


def _yaml_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    confirmed = _identity(proposed=False)
    proposed = _identity(proposed=True)
    confirmed_file_sha = promotion._canonical_file_sha(confirmed)
    proposed_file_sha = promotion._canonical_file_sha(proposed)
    decision = {
        "outcome": "PASS",
        "confirmation_passed": True,
        "candidate_family_id": "K1",
        "decided_at_utc": _time(10),
        "gates": {"quality": True, "boundary": True},
        "authority": {
            "production_identity_sha256": confirmed["production_identity_sha256"],
            "production_image_id": confirmed["container_image"]["image_id"],
        },
        "decision_sha256": _sha("decision"),
    }
    decision_file_sha = promotion._canonical_file_sha(decision)
    authority = {
        "production_identity": confirmed,
        "production_identity_file_sha256": confirmed_file_sha,
    }
    probe_script_sha = _sha("probe-script")
    plans: dict[str, dict] = {}
    aggregates: dict[str, dict] = {}
    cell_controls: dict[str, dict] = {}
    private_by_cell: dict[str, dict] = {}
    probe_controls: dict[str, dict] = {}
    for index, cell in enumerate(promotion.BOUNDARY_CELLS):
        steps = 400 + index
        runtime_selection, plan_selection = _checkpoint_bindings(steps)
        seed = 42565431
        fixture_path = tmp_path / cell / "fixture.json"
        fixture = {
            "manifest_sha256": _sha(f"manifest-{cell}"),
            "experimental_role": cell,
            "training_rows": [{"row": row} for row in range(12 + index)],
        }
        _canonical_write(fixture_path, fixture)
        explicit_path = tmp_path / cell / "private" / "effective-config.yaml"
        explicit = _explicit_config(cell, seed=seed, steps=steps)
        _yaml_write(explicit_path, explicit)
        explicit_sha = hashlib.sha256(explicit_path.read_bytes()).hexdigest()
        control_binding = {
            "file_sha256": _sha(f"control-file-{cell}"),
            "receipt_sha256": _sha(f"control-{cell}"),
            "config_sha256": explicit_sha,
        }
        terminal_binding = {
            "file_sha256": _sha(f"terminal-file-{cell}"),
            "receipt_sha256": _sha(f"terminal-{cell}"),
        }
        selection_binding = {
            "file_sha256": _sha(f"selection-file-{cell}"),
            "receipt_sha256": _sha(f"selection-{cell}"),
        }
        execution_plan = {
            "phase": "boundary",
            "cell_id": cell,
            "fixture_id": cell,
            "calibration_profile": "K1",
            "training_candidate_id": "K1-final691",
            "seed": seed,
            "planned_steps": steps,
            "checkpoint_selection": plan_selection,
            "task_id": f"stage2-{cell.lower()}",
            "model": "krea/Krea-2-Raw",
            "model_type": "krea2",
            "expected_repo_name": f"stage2-{cell.lower()}-k1",
            "trigger_word": "SN56",
            "hours": ("0.5" if "0p5" in cell else "0.75" if "0p75" in cell else "1.0"),
            "fixture_manifest": {
                "path": str(fixture_path),
                "file_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                "manifest_sha256": fixture["manifest_sha256"],
            },
            "plan_sha256": _sha(f"execution-plan-{cell}"),
        }
        approval = {"approval_sha256": _sha(f"approval-{cell}")}
        candidate_path = tmp_path / cell / "last.safetensors"
        candidate_path.write_bytes(cell.encode("ascii"))
        candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        candidate_bytes = candidate_path.stat().st_size
        selected_name = (
            f"{execution_plan['expected_repo_name']}.safetensors"
            if plan_selection["selected_step"] == steps
            else (
                f"{execution_plan['expected_repo_name']}_"
                f"{plan_selection['selected_step']:09d}.safetensors"
            )
        )
        completion = {
            "completion_sha256": _sha(f"completion-{cell}"),
            "mechanics": dict(promotion._CLEAN_MECHANICS),
            "config_control_receipt": control_binding,
            "training_terminal_receipt": terminal_binding,
            "checkpoint_selection_receipt": dict(selection_binding),
            "artifact_manifest": [
                {
                    "path": f"checkpoints/{selected_name}",
                    "bytes": candidate_bytes,
                    "sha256": candidate_sha,
                },
                {
                    "path": "checkpoints/last.safetensors",
                    "bytes": candidate_bytes,
                    "sha256": candidate_sha,
                },
                {
                    "path": "evidence/forge_checkpoint_selection.json",
                    "bytes": 100,
                    "sha256": selection_binding["file_sha256"],
                },
            ],
        }
        run_evidence_path = tmp_path / cell / "run-evidence.json"
        run_evidence = {"evidence_sha256": _sha(f"run-evidence-{cell}")}
        _canonical_write(run_evidence_path, run_evidence)
        run_control = {
            "run_evidence_path": str(run_evidence_path),
            "execution_plan": execution_plan,
            "execution_approval": approval,
            "run_completion": completion,
            "candidate_path": str(candidate_path),
        }
        plans[cell] = {
            "plan_sha256": _sha(f"score-plan-{cell}"),
            "candidates": [
                {
                    "family_id": "K1",
                    "training_candidate_id": "K1-final691",
                    "execution_plan_sha256": execution_plan["plan_sha256"],
                    "execution_approval_sha256": approval["approval_sha256"],
                    "run_completion_sha256": completion["completion_sha256"],
                    "run_evidence_file_sha256": hashlib.sha256(
                        run_evidence_path.read_bytes()
                    ).hexdigest(),
                    "run_evidence_sha256": run_evidence["evidence_sha256"],
                    "candidate_sha256": candidate_sha,
                    "candidate_bytes": candidate_bytes,
                    "checkpoint_rule_sha256": plan_selection["checkpoint_rule_sha256"],
                    "checkpoint_target_fraction": plan_selection["target_fraction"],
                    "checkpoint_mapping_rule": plan_selection["mapping_rule"],
                    "step": plan_selection["selected_step"],
                    "fraction_numerator": plan_selection["selected_step"],
                    "fraction_denominator": steps,
                    "mechanics": dict(promotion._CLEAN_MECHANICS),
                }
            ],
        }
        aggregates[cell] = {}
        cell_controls[cell] = {"run_controls_by_family": {"K1": run_control}}
        private_by_cell[cell] = {
            "evidence_root": str(explicit_path.parent),
            "effective_config_path": str(explicit_path),
            "config_control": control_binding,
            "training_terminal": terminal_binding,
            "checkpoint_selection": dict(selection_binding),
        }
        release_path = tmp_path / cell / "release-config.yaml"
        _yaml_write(release_path, _release_config(explicit))
        release_raw = release_path.read_bytes()
        probe_payload = {
            "schema": 1,
            "kind": promotion.PROBE_KIND,
            "cell_id": cell,
            "production_identity_sha256": proposed["production_identity_sha256"],
            "production_image_id": proposed["container_image"]["image_id"],
            "probe_script_sha256": probe_script_sha,
            "calibration_environment": {
                name: False for name in promotion.CALIBRATION_ENVIRONMENT
            },
            "image_spec": {
                "task_id": execution_plan["task_id"],
                "model": execution_plan["model"],
                "model_type": execution_plan["model_type"],
                "expected_repo_name": execution_plan["expected_repo_name"],
                "trigger_word": execution_plan["trigger_word"],
                "hours": execution_plan["hours"],
                "training_row_count": len(fixture["training_rows"]),
            },
            "config_file": {
                "name": release_path.name,
                "bytes": len(release_raw),
                "sha256": hashlib.sha256(release_raw).hexdigest(),
            },
            "rendered_at_utc": _time(30 + index),
            "release_authorized": False,
            "production_mutation_authorized": False,
            "deployment_authorized": False,
        }
        probe_receipt = promotion.seal_probe_receipt(probe_payload)
        probe_path = tmp_path / cell / "probe-receipt.json"
        _canonical_write(probe_path, probe_receipt)
        probe_controls[cell] = {
            "receipt_path": str(probe_path),
            "config_path": str(release_path),
        }

    monkeypatch.setattr(
        promotion.krea_stage2_decision,
        "validate_decision",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        promotion.krea_stage2_execution,
        "validate_plan",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        promotion.krea_stage2_execution,
        "validate_approval",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        promotion.krea_stage2_execution,
        "validate_completion",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        promotion.krea_stage2_execution,
        "validate_private_run_receipts",
        lambda plan: private_by_cell[plan["cell_id"]],
    )
    monkeypatch.setattr(
        promotion.krea_stage2_training_evidence,
        "validate_run_evidence",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        promotion.krea_fixture,
        "validate_manifest",
        lambda value: dict(value),
    )
    return {
        "decision_record": decision,
        "decision_file_sha256": decision_file_sha,
        "plans": plans,
        "aggregates": aggregates,
        "cell_controls": cell_controls,
        "authority_controls": authority,
        "proposed_release_identity": proposed,
        "proposed_release_identity_file_sha256": proposed_file_sha,
        "probe_script_sha256": probe_script_sha,
        "probe_controls_by_cell": probe_controls,
        "created_at_utc": _time(50),
    }


def _refresh_probe_config_binding(case: dict, cell: str) -> None:
    release_path = Path(case["probe_controls_by_cell"][cell]["config_path"])
    probe_path = Path(case["probe_controls_by_cell"][cell]["receipt_path"])
    probe = json.loads(probe_path.read_bytes())
    raw = release_path.read_bytes()
    probe["config_file"] = {
        "name": release_path.name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    body = {key: value for key, value in probe.items() if key != "receipt_sha256"}
    probe["receipt_sha256"] = krea_provenance.canonical_sha256(body)
    probe_path.write_bytes(krea_provenance.canonical_bytes(probe) + b"\n")


def test_release_promotion_round_trip_binds_all_six_env_unset_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    proof = promotion.build_proof(**case)

    assert proof["selected_family_id"] == "K1"
    assert len(proof["boundary_cells"]) == 6
    assert all(proof["gates"].values())
    assert proof["release_review_required"] is True
    assert proof["release_authorized"] is False
    assert proof["deployment_authorized"] is False
    assert promotion.krea_calibration_profiles.STAGE2_TARGET_NUMERATOR_ENV in (
        promotion.CALIBRATION_ENVIRONMENT
    )
    assert promotion.krea_calibration_profiles.STAGE2_TARGET_DENOMINATOR_ENV in (
        promotion.CALIBRATION_ENVIRONMENT
    )
    assert (
        promotion.validate_proof(
            proof, **{k: v for k, v in case.items() if k != "created_at_utc"}
        )
        == proof
    )


@pytest.mark.parametrize("field", ["steps", "save_every"])
def test_release_promotion_rejects_depth_or_save_policy_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    case = _case(tmp_path, monkeypatch)
    cell = promotion.BOUNDARY_CELLS[0]
    release_path = Path(case["probe_controls_by_cell"][cell]["config_path"])
    config = yaml.safe_load(release_path.read_bytes())
    section = "train" if field == "steps" else "save"
    config["config"]["process"][0][section][field] += 1
    _yaml_write(release_path, config)
    # Keep the probe internally consistent so the decisive failure is the
    # selected-profile versus env-unset full-config comparison.
    _refresh_probe_config_binding(case, cell)

    with pytest.raises(ValueError, match="env-unset release config"):
        promotion.build_proof(**case)


def test_release_promotion_rejects_calibration_environment_or_missing_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    cell = promotion.BOUNDARY_CELLS[0]
    probe_path = Path(case["probe_controls_by_cell"][cell]["receipt_path"])
    probe = json.loads(probe_path.read_bytes())
    probe["calibration_environment"][promotion.CALIBRATION_ENVIRONMENT[0]] = True
    body = {key: value for key, value in probe.items() if key != "receipt_sha256"}
    probe["receipt_sha256"] = krea_provenance.canonical_sha256(body)
    probe_path.write_bytes(krea_provenance.canonical_bytes(probe) + b"\n")
    with pytest.raises(ValueError, match="env-unset surface"):
        promotion.build_proof(**case)

    case = _case(tmp_path / "missing", monkeypatch)
    case["probe_controls_by_cell"].pop(promotion.BOUNDARY_CELLS[-1])
    with pytest.raises(ValueError, match="exact boundary matrix"):
        promotion.build_proof(**case)


def test_release_promotion_rejects_env_unset_final_checkpoint_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    cell = promotion.BOUNDARY_CELLS[0]
    release_path = Path(case["probe_controls_by_cell"][cell]["config_path"])
    config = yaml.safe_load(release_path.read_bytes())
    process = config["config"]["process"][0]
    policy = config["meta"]["forge_krea_checkpoint_selection"]
    # Keep every recipe/depth/save byte unchanged; only replace the frozen
    # early-checkpoint policy with an exact-final policy.
    policy["target_fraction"] = {"numerator": 1, "denominator": 1}
    policy["selected_step"] = process["train"]["steps"]
    _yaml_write(release_path, config)
    _refresh_probe_config_binding(case, cell)

    with pytest.raises(ValueError, match="checkpoint selection differs from its plan"):
        promotion.build_proof(**case)


def test_release_promotion_rejects_private_selection_or_score_policy_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    cell = promotion.BOUNDARY_CELLS[0]
    run_control = case["cell_controls"][cell]["run_controls_by_family"]["K1"]
    run_control["run_completion"]["checkpoint_selection_receipt"]["receipt_sha256"] = (
        _sha("tampered-completion-selection")
    )
    with pytest.raises(ValueError, match="private receipts drifted"):
        promotion.build_proof(**case)

    case = _case(tmp_path / "score", monkeypatch)
    cell = promotion.BOUNDARY_CELLS[0]
    candidate = case["plans"][cell]["candidates"][0]
    candidate["step"] += 1
    candidate["fraction_numerator"] = candidate["step"]
    with pytest.raises(ValueError, match="score checkpoint policy differs"):
        promotion.build_proof(**case)


def test_release_promotion_rejects_selected_source_last_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    cell = promotion.BOUNDARY_CELLS[0]
    completion = case["cell_controls"][cell]["run_controls_by_family"]["K1"][
        "run_completion"
    ]
    promoted = next(
        row
        for row in completion["artifact_manifest"]
        if row["path"] == "checkpoints/last.safetensors"
    )
    promoted["sha256"] = _sha("substituted-full-final")
    with pytest.raises(ValueError, match="frozen checkpoint promotion"):
        promotion.build_proof(**case)


def test_release_promotion_proof_cannot_grant_release_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    proof = promotion.build_proof(**case)
    proof["release_authorized"] = True
    body = {key: value for key, value in proof.items() if key != "proof_sha256"}
    proof["proof_sha256"] = krea_provenance.canonical_sha256(body)

    with pytest.raises(ValueError, match="identity or authority"):
        promotion.validate_proof(
            proof, **{k: v for k, v in case.items() if k != "created_at_utc"}
        )
