"""Operational score-plan builder tests, including a producer/consumer chain."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


_ROOT = Path(__file__).parents[1]
_CALIBRATION = _ROOT / "ops" / "calibration"
sys.path.insert(0, str(_CALIBRATION))
sys.path.insert(0, str(Path(__file__).parent))

import batch_evaluate_krea as batch  # noqa: E402
import evaluate_krea_local  # noqa: E402
import krea_decision  # noqa: E402
import krea_fixture_admission  # noqa: E402
import krea_provenance  # noqa: E402
import krea_score_plan as score_plan  # noqa: E402
import test_krea_training_evidence_cli as stage3_test  # noqa: E402
from test_krea_v2_batch_contract import ProducerHarness  # noqa: E402


def test_exact_scorer_environments_do_not_inherit_operator_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "PYTORCH_CUDA_ALLOC_CONF",
        "CUDA_HOME",
        "LD_LIBRARY_PATH",
        "HTTPS_PROXY",
        "OMP_NUM_THREADS",
        "NCCL_DEBUG",
        "CUBLAS_WORKSPACE_CONFIG",
    ):
        monkeypatch.setenv(name, "operator-value")
    outer_root = tmp_path / "outer"
    outer_root.mkdir()
    outer = batch._minimal_evaluator_environment(
        driver_python=sys.executable, isolated_root=outer_root
    )
    inner_root = tmp_path / "inner"
    inner_root.mkdir()
    inner = evaluate_krea_local._comfy_child_environment(
        comfy_python=Path(sys.executable), isolation_root=inner_root
    )
    for environment in (outer, inner):
        for name in (
            "PYTORCH_CUDA_ALLOC_CONF",
            "CUDA_HOME",
            "LD_LIBRARY_PATH",
            "HTTPS_PROXY",
            "OMP_NUM_THREADS",
            "NCCL_DEBUG",
            "CUBLAS_WORKSPACE_CONFIG",
        ):
            assert name not in environment
        assert environment["HF_HUB_OFFLINE"] == "1"
        assert environment["HF_HOME"].startswith(str(tmp_path))

    inspection = evaluate_krea_local._inspection_environment()
    for name in (
        "PYTHONPATH",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "GIT_CONFIG_GLOBAL",
    ):
        if name == "GIT_CONFIG_GLOBAL":
            assert inspection[name] == "/dev/null"
        else:
            assert name not in inspection
    assert inspection["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert inspection["GIT_CONFIG_VALUE_0"] == "false"


def test_exact_scorer_binds_approved_order_across_filesystem_enumeration(
    tmp_path: Path,
) -> None:
    # APFS and ext4 can enumerate the same immutable file set differently.
    # The approved fixture order must win on both the evaluator side and the
    # batch command side; native order remains only a set-integrity check.
    approved = ["fontana.jpg", "no-print.jpg"]
    native_ext4 = ["no-print.jpg", "fontana.jpg"]
    enumerate_images = evaluate_krea_local._ordered_image_enumerator(
        lambda _root, _extensions: list(native_ext4),
        approved,
    )
    assert enumerate_images(str(tmp_path), (".jpg",)) == approved

    original = lambda _root, _extensions: list(native_ext4)
    diffusion = SimpleNamespace(list_supported_images=original)

    def eval_loop(dataset, params, *, generations):
        assert dataset == str(tmp_path)
        assert params == "params"
        assert generations == 5
        return {
            "observed_order": diffusion.list_supported_images(dataset, (".jpg",))
        }

    diffusion.eval_loop = eval_loop
    raw, scored_order = evaluate_krea_local._run_exact_eval(
        diffusion,
        dataset=tmp_path,
        params="params",
        generations=5,
        list_supported_images=enumerate_images,
    )
    assert raw["observed_order"] == approved
    assert scored_order == approved
    assert diffusion.list_supported_images is original

    evaluator = {
        "driver_python": sys.executable,
        "comfy_root": str(tmp_path / "comfy"),
        "comfy_python": sys.executable,
        "god_root": str(tmp_path / "god"),
        "expected_god_commit": "a" * 40,
        "_expected_dataset_identity": {"evaluator_order": approved},
    }
    command = batch._evaluator_command(
        evaluator_script=_CALIBRATION / "evaluate_krea_local.py",
        dataset=tmp_path / "dataset",
        candidate={"path": tmp_path / "candidate.safetensors"},
        result_path=tmp_path / "result.json",
        evaluator=evaluator,
    )
    observed = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--expected-image"
    ]
    assert observed == approved

    mismatched = evaluate_krea_local._ordered_image_enumerator(
        lambda _root, _extensions: ["other.jpg"],
        approved,
    )
    with pytest.raises(RuntimeError, match="image set differs"):
        mismatched(str(tmp_path), (".jpg",))


def _canonical_file(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(krea_provenance.canonical_bytes(value) + b"\n")
    return krea_provenance.file_sha256(path)


def _agent_admission_result(envelope: dict, fixture: dict) -> dict:
    manifests = {
        role: (
            fixture["manifest_sha256"]
            if role == fixture["experimental_role"]
            else role.lower() * 32
        )
        for role in ("D1", "D2", "C1", "C2", "C3", "C4")
    }
    return {
        "envelope": envelope,
        "fixtures": {fixture["experimental_role"]: fixture},
        "blinded_acceptance": {
            "fixture_manifest_sha256s": manifests,
            "assertions": {
                "all_six_cross_fixture_review_preexists_discovery_execution": True,
                "agent_review_is_not_human_review": True,
            },
            "decision": "accepted_for_d1_d2_discovery_admission",
            "c1c4_revealed": False,
        },
    }


def test_agent_admission_envelope_is_the_score_plan_cross_fixture_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = {
        "experimental_role": "D1",
        "manifest_sha256": "d" * 64,
    }
    envelope = {
        "schema": 1,
        "kind": "forge-krea-fixture-admission-envelope",
        "envelope_sha256": "e" * 64,
    }
    envelope_path = tmp_path / "admission-envelope.json"
    _canonical_file(envelope_path, envelope)
    resolved = _agent_admission_result(envelope, fixture)
    monkeypatch.setattr(
        krea_fixture_admission, "validate_envelope", lambda _path: resolved
    )
    assert batch._validate_cross_fixture_review_surface(
        envelope,
        fixture=fixture,
        source_path=envelope_path,
    ) == envelope
    resolved["blinded_acceptance"]["assertions"][
        "agent_review_is_not_human_review"
    ] = False
    with pytest.raises(ValueError, match="does not bind this discovery fixture"):
        batch._validate_cross_fixture_review_surface(
            envelope, fixture=fixture, source_path=envelope_path
        )


def test_historical_fixture_admission_validator_is_explicitly_dispatched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = {
        "experimental_role": "D1",
        "manifest_sha256": "d" * 64,
    }
    envelope = {
        "schema": 1,
        "kind": "forge-krea-fixture-admission-envelope",
        "envelope_sha256": "e" * 64,
    }
    envelope_path = tmp_path / "historical-admission-envelope.json"
    _canonical_file(envelope_path, envelope)
    resolved = _agent_admission_result(envelope, fixture)
    historical = SimpleNamespace(validate_envelope=lambda _path: resolved)
    monkeypatch.setattr(
        krea_fixture_admission,
        "validate_envelope",
        lambda _path: pytest.fail("current admission validator was used"),
    )
    assert batch._validate_cross_fixture_review_surface(
        envelope,
        fixture=fixture,
        source_path=envelope_path,
        fixture_admission_validator=historical,
    ) == envelope


def test_d1_score_plan_builder_binds_agent_admission_not_fictional_human_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_cross_validator = batch._validate_cross_fixture_review_surface
    harness = BuilderHarness(tmp_path, monkeypatch)
    monkeypatch.setattr(
        batch, "_validate_cross_fixture_review_surface", real_cross_validator
    )
    envelope = {
        "schema": 1,
        "kind": "forge-krea-fixture-admission-envelope",
        "envelope_sha256": "e" * 64,
    }
    harness.base.fixture["experimental_role"] = "D1"
    harness.base.fixture_path.write_bytes(
        krea_provenance.canonical_bytes(harness.base.fixture) + b"\n"
    )
    harness.base.cross_review_path.write_bytes(
        krea_provenance.canonical_bytes(envelope) + b"\n"
    )
    resolved = _agent_admission_result(envelope, harness.base.fixture)
    monkeypatch.setattr(
        krea_fixture_admission, "validate_envelope", lambda _path: resolved
    )
    _campaign, draft = score_plan.build_documents(**harness.kwargs())
    assert draft["cross_fixture_review"] == {
        "path": str(harness.base.cross_review_path),
        "sha256": krea_provenance.file_sha256(harness.base.cross_review_path),
    }


def _schema3_approval_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    binding_path = tmp_path / "candidate-binding.json"
    binding_document = {"mode": "local_run_candidate"}
    binding_sha = _canonical_file(binding_path, binding_document)
    evaluator = {
        "expected_god_commit": "a" * 40,
        "expected_comfy_commit": "b" * 40,
        "expected_tooling_commit": "c" * 40,
        "expected_evaluator_script_sha256": "1" * 64,
        "expected_dataset_identity_module_sha256": "2" * 64,
        "expected_eval_defaults": {"steps": 25},
        "expected_runtime_identity": {
            "comfy_python_identity_sha256": "3" * 64,
            "driver_python_identity_sha256": "4" * 64,
        },
        "expected_assets": {"asset": {"sha256": "5" * 64}},
        "cache_provenance_sha256": "6" * 64,
        "containment": {"term_grace_s": 20.0},
        "startup_timeout_s": 300.0,
        "evaluation_timeout_s": 3600.0,
        "shutdown_timeout_s": 20.0,
    }
    plan = {
        "schema": 2,
        "kind": "forge-krea-exact-score-plan",
        "fixture_manifest": {"sha256": "7" * 64},
        "fixture_approval": {"sha256": "8" * 64},
        "campaign_manifest": {"sha256": "9" * 64},
        "decision_context": {"phase": "discovery"},
        "evaluator": evaluator,
        "candidates": [
            {
                "id": "K1-step-50",
                "candidate_binding": {
                    "path": str(binding_path),
                    "sha256": binding_sha,
                },
            }
        ],
    }
    actor = batch.krea_delegated_review_contract.actor("exact_score_plan_reviewer")
    authorization = {
        "authorization_sha256": "d" * 64,
        "accountable_owner_identity": "Atulya Shetty",
        "fixture_admission_envelope": {"owner_ratification_sha256": "e" * 64},
        "authorized_actions": ["offline_exact_scoring"],
    }
    authorization_binding = {
        "path": str(tmp_path / "authorization.json"),
        "file_sha256": "f" * 64,
        "authorization_sha256": authorization["authorization_sha256"],
    }
    readiness = {
        "schema": 1,
        "kind": "forge-krea-stage1-exact-scorer-readiness",
        "ready": True,
        "readiness_sha256": "0" * 64,
    }
    monkeypatch.setattr(batch, "_validate_evaluator", lambda value: dict(value))
    monkeypatch.setattr(batch, "_validate_stage1_exact_scorer", lambda value: value)
    monkeypatch.setattr(
        batch, "_stage1_exact_scorer_readiness", lambda value: readiness
    )
    monkeypatch.setattr(
        batch.krea_discovery_authorization,
        "load_binding",
        lambda value: (
            Path(authorization_binding["path"]),
            authorization,
            authorization_binding["file_sha256"],
        ),
    )
    return plan, evaluator, actor, authorization, authorization_binding, readiness


def _historical_campaign_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    modules: dict,
) -> dict[str, str]:
    identity = {
        "schema": 1,
        "kind": "forge-krea-historical-training-evidence-validator",
        "root": "/admitted/forge-daf9a252",
    }
    monkeypatch.setattr(
        batch.krea_historical_training_evidence,
        "validate_identity",
        lambda value: value,
    )
    monkeypatch.setattr(
        batch.krea_historical_training_evidence,
        "load_modules",
        lambda value: modules if value == identity else pytest.fail("wrong identity"),
    )
    payload = {
        "schema": 2,
        "kind": "forge-krea-exact-score-campaign",
        "fixture_manifest_sha256": "1" * 64,
        "discovery_plan_sha256": "2" * 64,
        "runs": [
            {
                "arm_id": "K1",
                "execution_plan_sha256": "3" * 64,
                "run_completion_sha256": "4" * 64,
                "candidates": [
                    {
                        "candidate_id": "K1-step-1",
                        "sha256": "5" * 64,
                        "bytes": 1,
                        "step": 1,
                        "fraction": {"numerator": 1, "denominator": 1},
                    }
                ],
            }
        ],
        "zero_control_manifest_sha256": "6" * 64,
        "decision_contract": batch._DISCOVERY_DECISION_BINDING,
        "confirmation_contract": batch._CONFIRMATION_DECISION_BINDING,
        "historical_training_evidence_validator": identity,
    }
    campaign = batch.seal_campaign_manifest(payload)
    path = tmp_path / "historical-campaign.json"
    digest = _canonical_file(path, campaign)
    return {"path": str(path), "sha256": digest}


def test_schema3_score_approval_uses_historical_authority_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, evaluator, actor, authorization, binding, readiness = _schema3_approval_case(
        tmp_path, monkeypatch
    )
    calls = []
    historical_authorization = SimpleNamespace(
        load_binding=lambda value: (
            calls.append(("authorization", value))
            or (Path(binding["path"]), authorization, binding["file_sha256"])
        )
    )
    historical_delegated = SimpleNamespace(
        validate_actor=lambda role, value: (
            calls.append(("actor", role)) or value
        ),
        binding=lambda: {"historical_contract": "daf9a252"},
    )
    modules = {
        "discovery_authorization": historical_authorization,
        "delegated_review_contract": historical_delegated,
    }
    plan["campaign_manifest"] = _historical_campaign_binding(
        tmp_path, monkeypatch, modules
    )
    approval = batch.build_agent_sealed_plan_approval(
        plan,
        technical_reviewer_actor=actor,
        discovery_execution_authorization=binding,
    )
    assert approval["delegated_review_contract"] == {
        "historical_contract": "daf9a252"
    }
    candidate = {
        "id": "K1-step-50",
        "candidate_binding": {
            "mode": "local_run_candidate",
            "binding_manifest_sha256": plan["candidates"][0][
                "candidate_binding"
            ]["sha256"],
        },
    }
    result = batch._validate_v2_approval(
        approval,
        krea_provenance.canonical_bytes(approval) + b"\n",
        plan=plan,
        candidates=[candidate],
        evaluator=evaluator,
        common_authorization_sha256=authorization["authorization_sha256"],
    )
    assert result["accountable_owner_identity"] == "Atulya Shetty"
    assert [row[0] for row in calls] == [
        "authorization",
        "actor",
        "authorization",
        "actor",
    ]
    assert approval["scorer_readiness"] == readiness


def test_schema3_agent_score_approval_builds_and_revalidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, evaluator, actor, authorization, binding, readiness = _schema3_approval_case(
        tmp_path, monkeypatch
    )
    approval = batch.build_agent_sealed_plan_approval(
        plan,
        technical_reviewer_actor=actor,
        discovery_execution_authorization=binding,
    )
    assert approval["schema"] == 3
    assert approval["technical_reviewer_actor"] == actor
    assert approval["accountable_owner_identity"] == "Atulya Shetty"
    assert approval["scorer_readiness"] == readiness
    candidate = {
        "id": "K1-step-50",
        "candidate_binding": {
            "mode": "local_run_candidate",
            "binding_manifest_sha256": plan["candidates"][0]["candidate_binding"][
                "sha256"
            ],
        },
    }
    result = batch._validate_v2_approval(
        approval,
        krea_provenance.canonical_bytes(approval) + b"\n",
        plan=plan,
        candidates=[candidate],
        evaluator=evaluator,
        common_authorization_sha256=authorization["authorization_sha256"],
    )
    assert result == {
        "technical_reviewer_actor": actor,
        "accountable_owner_identity": "Atulya Shetty",
        "decision": "approved",
        "agent_review_is_not_human_review": True,
    }


@pytest.mark.parametrize(
    "mutation",
    ("candidate", "runtime", "assets", "timeout", "readiness", "owner", "contract"),
)
def test_schema3_agent_score_approval_rejects_tampered_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    plan, evaluator, actor, authorization, binding, readiness = _schema3_approval_case(
        tmp_path, monkeypatch
    )
    approval = batch.build_agent_sealed_plan_approval(
        plan,
        technical_reviewer_actor=actor,
        discovery_execution_authorization=binding,
    )
    candidate = {
        "id": "K1-step-50",
        "candidate_binding": {
            "mode": "local_run_candidate",
            "binding_manifest_sha256": plan["candidates"][0]["candidate_binding"][
                "sha256"
            ],
        },
    }
    if mutation == "candidate":
        candidate["id"] = "K1-step-evil"
    elif mutation == "runtime":
        evaluator["expected_runtime_identity"] = {
            "comfy_python_identity_sha256": "a" * 64,
            "driver_python_identity_sha256": "b" * 64,
        }
    elif mutation == "assets":
        evaluator["expected_assets"] = {"asset": {"sha256": "a" * 64}}
    elif mutation == "timeout":
        evaluator["startup_timeout_s"] = 301.0
    elif mutation == "readiness":
        approval["scorer_readiness"] = {**readiness, "ready": False}
    elif mutation == "owner":
        approval["accountable_owner_identity"] = "Another Owner"
    else:
        approval["delegated_review_contract"] = {
            **approval["delegated_review_contract"],
            "contract_sha256": "a" * 64,
        }
    with pytest.raises(ValueError):
        batch._validate_v2_approval(
            approval,
            krea_provenance.canonical_bytes(approval) + b"\n",
            plan=plan,
            candidates=[candidate],
            evaluator=evaluator,
            common_authorization_sha256=authorization["authorization_sha256"],
        )


def test_schema3_score_approval_rejects_legacy_wrong_actor_action_and_missing_deps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, evaluator, actor, authorization, binding, _readiness = _schema3_approval_case(
        tmp_path, monkeypatch
    )
    candidate = {
        "id": "K1-step-50",
        "candidate_binding": {
            "mode": "local_run_candidate",
            "binding_manifest_sha256": plan["candidates"][0]["candidate_binding"][
                "sha256"
            ],
        },
    }
    legacy = batch.build_sealed_plan_approval(plan, reviewer_identity="Legacy Human")
    with pytest.raises(ValueError, match="requires the delegated schema-3 approval"):
        batch._validate_v2_approval(
            legacy,
            krea_provenance.canonical_bytes(legacy) + b"\n",
            plan=plan,
            candidates=[candidate],
            evaluator=evaluator,
            common_authorization_sha256=authorization["authorization_sha256"],
        )

    bad_actor = {**actor, "actor_id": "not-owner-ratified"}
    with pytest.raises(ValueError, match="owner-ratified delegated actor"):
        batch.build_agent_sealed_plan_approval(
            plan,
            technical_reviewer_actor=bad_actor,
            discovery_execution_authorization=binding,
        )
    authorization["authorized_actions"] = []
    with pytest.raises(ValueError, match="does not permit exact scoring"):
        batch.build_agent_sealed_plan_approval(
            plan,
            technical_reviewer_actor=actor,
            discovery_execution_authorization=binding,
        )

    draft_path = tmp_path / "draft.json"
    _canonical_file(draft_path, plan)
    monkeypatch.setattr(score_plan, "validate_draft", lambda value: value)
    monkeypatch.setattr(
        score_plan.batch,
        "build_agent_sealed_plan_approval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("exact scorer dependency lock is missing")
        ),
    )
    authorization_path = tmp_path / "authorization.json"
    _canonical_file(authorization_path, {"authorization_sha256": "d" * 64})
    approval_output = tmp_path / "approval.json"
    plan_output = tmp_path / "plan.json"
    with pytest.raises(ValueError, match="dependency lock is missing"):
        score_plan.approve_draft(
            draft_path=draft_path,
            reviewer_identity=None,
            approval_output=approval_output,
            plan_output=plan_output,
            technical_reviewer_actor=actor,
            discovery_authorization_path=authorization_path,
        )
    assert not approval_output.exists()
    assert not plan_output.exists()


def _readiness_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    contract = batch.krea_execution_surface_policy.POLICY[
        "stage1_exact_scorer_contract"
    ]
    comfy = tmp_path / "ComfyUI"
    god = tmp_path / "G.O.D"
    tooling = comfy / "custom_nodes" / "comfyui-tooling-nodes"
    for path in (comfy, god, tooling, comfy / "models" / "loras"):
        path.mkdir(parents=True, exist_ok=True)
    (comfy / "models" / "loras" / "put_loras_here").write_bytes(b"")
    requirements = {
        "comfyui": comfy / "requirements.txt",
        "tooling_nodes": tooling / "requirements.txt",
        "god_validator": god / "ops/docker/requirements/validator.txt",
    }
    for path in requirements.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("bound requirement\n", encoding="utf-8")
    expected_assets = {}
    for name, row in contract["assets"].items():
        path = comfy / row["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("utf-8"))
        expected_assets[name] = {
            "canonical_path": str(path),
            "sha256": row["sha256"],
            "bytes": path.stat().st_size,
        }
    python_path = Path(sys.executable).resolve()
    python_environment = {
        "executable": str(python_path),
        "prefix": str(python_path.parent.parent),
        "base_prefix": str(python_path.parent.parent),
        "python": contract["runtime_materialization"]["python_version"],
        "distribution_count": contract["runtime_materialization"]["distribution_count"],
        "distributions_sha256": contract["runtime_materialization"][
            "distributions_sha256"
        ],
        "normalized_distributions_sha256": contract["runtime_materialization"][
            "exact_lock"
        ]["normalized_name_version_sha256"],
        "requested_executable": str(python_path),
        "venv_root": str(python_path.parent.parent),
        "environment_kind": "venv",
        "identity_marker": str(tmp_path / "pyvenv.cfg"),
        "identity_marker_sha256": "1" * 64,
    }
    driver_environment = {
        key: python_environment[key]
        for key in (
            "executable",
            "prefix",
            "base_prefix",
            "python",
            "distribution_count",
            "distributions_sha256",
            "normalized_distributions_sha256",
        )
    }
    evaluator = {
        "comfy_root": str(comfy),
        "comfy_python": str(python_path),
        "driver_python": str(python_path),
        "god_root": str(god),
        "expected_god_commit": contract["god_commit"],
        "expected_comfy_commit": contract["comfy_commit"],
        "expected_tooling_commit": contract["tooling_commit"],
        "expected_evaluator_script_sha256": contract["evaluator_script_sha256"],
        "expected_dataset_identity_module_sha256": contract[
            "dataset_identity_module_sha256"
        ],
        "expected_eval_defaults": contract["eval_defaults"],
        "expected_runtime_identity": {
            "comfy_python_identity_sha256": krea_provenance.canonical_sha256(
                python_environment
            ),
            "driver_python_identity_sha256": krea_provenance.canonical_sha256(
                driver_environment
            ),
        },
        "expected_assets": expected_assets,
        "containment": {
            "term_grace_s": contract["timeouts_s"]["containment_term_grace"]
        },
        "startup_timeout_s": contract["timeouts_s"]["startup"],
        "evaluation_timeout_s": contract["timeouts_s"]["evaluation"],
        "shutdown_timeout_s": contract["timeouts_s"]["shutdown"],
        "scorer_extension_policy": deepcopy(
            batch.krea_scorer_extension_policy.POLICY
        ),
        "scorer_timeout_profile": "D1",
    }
    evaluator["evaluation_timeout_s"] = 5400.0
    snapshots = {
        "god": {
            "commit": contract["god_commit"],
            "tree": contract["source_trees"]["god"],
        },
        "comfyui": {
            "commit": contract["comfy_commit"],
            "tree": contract["source_trees"]["comfyui"],
        },
        "tooling_nodes": {
            "commit": contract["tooling_commit"],
            "tree": contract["source_trees"]["tooling_nodes"],
        },
    }

    def git_snapshot(path, *, expected_commit):
        if path == god:
            return dict(snapshots["god"])
        if path == comfy:
            return dict(snapshots["comfyui"])
        return dict(snapshots["tooling_nodes"])

    live_hashes = {
        **{
            str(path): contract["runtime_materialization"]["requirements_sha256"][name]
            for name, path in requirements.items()
        },
        **{row["canonical_path"]: row["sha256"] for row in expected_assets.values()},
    }
    real_sha256 = batch._sha256

    def fake_sha256(path):
        return live_hashes.get(str(path), real_sha256(path))

    monkeypatch.setattr(batch, "_validate_stage1_exact_scorer", lambda value: value)
    monkeypatch.setattr(evaluate_krea_local, "_git_snapshot", git_snapshot)
    monkeypatch.setattr(
        evaluate_krea_local, "_python_environment", lambda _path: python_environment
    )
    monkeypatch.setattr(batch, "_sha256", fake_sha256)
    def fake_runtime_probe(argv, **_kwargs):
        key = (
            "cuda_runtime_probe"
            if "Conv3d" in " ".join(str(item) for item in argv)
            else "critical_distributions"
        )
        return json.dumps(contract["runtime_materialization"][key])

    monkeypatch.setattr(batch.subprocess, "check_output", fake_runtime_probe)
    return evaluator, contract, snapshots, live_hashes, python_environment


def test_stage1_exact_scorer_readiness_recomputes_complete_bound_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator, contract, _snapshots, _hashes, _python = _readiness_case(
        tmp_path, monkeypatch
    )
    result = batch._stage1_exact_scorer_readiness(evaluator)
    assert result["ready"] is True
    assert (
        result["dependency_lock_sha256"]
        == contract["runtime_materialization"]["exact_lock"]["sha256"]
    )
    assert result["source_snapshots"]["god"]["tree"] == contract["source_trees"]["god"]


def test_stage1_d1_timeout_covers_measured_runtime_and_rejects_old_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = batch._validate_stage1_exact_scorer
    evaluator, contract, _snapshots, _hashes, _python = _readiness_case(
        tmp_path, monkeypatch
    )
    for name, row in evaluator["expected_assets"].items():
        row["bytes"] = contract["assets"][name]["bytes"]
    assert contract["timeouts_s"]["evaluation"] == 3600.0
    assert (
        batch.krea_scorer_extension_policy.POLICY[
            "changes"
        ]["evaluation_timeout_profiles"]["D1"]["evaluation_timeout_s"]
        == 5400.0
    )
    assert validator(evaluator)["timeouts_s"]["evaluation"] == 3600.0
    assert batch._timeout_policy(evaluator)["total_candidate_timeout_s"] == 5780.0

    evaluator["evaluation_timeout_s"] = 3600.0
    with pytest.raises(
        ValueError, match="effective timeouts differ from its extension"
    ):
        validator(evaluator)

    evaluator["evaluation_timeout_s"] = 5400.0
    evaluator["scorer_extension_policy"] = {
        **evaluator["scorer_extension_policy"],
        "policy_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="extension policy drifted"):
        validator(evaluator)


def test_stage1_d2_timeout_is_shape_bound_and_has_explicit_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = batch._validate_stage1_exact_scorer
    evaluator, contract, _snapshots, _hashes, _python = _readiness_case(
        tmp_path, monkeypatch
    )
    for name, row in evaluator["expected_assets"].items():
        row["bytes"] = contract["assets"][name]["bytes"]
    evaluator["scorer_timeout_profile"] = "D2"
    evaluator["evaluation_timeout_s"] = 9000.0
    fixture = {
        "experimental_role": "D2",
        "evaluation_rows": [{"row_id": f"row-{index:03d}"} for index in range(40)],
    }
    profile = batch._validate_scorer_fixture_timeout(evaluator, fixture)
    assert profile["prompt_count"] == 400
    assert profile["evaluation_timeout_s"] == 9000.0
    assert profile["evaluation_timeout_s"] - profile["measured_runtime_s"] == 1500.0
    assert validator(evaluator)["timeouts_s"] == contract["timeouts_s"]
    assert batch._timeout_policy(evaluator)["total_candidate_timeout_s"] == 9380.0

    fixture["evaluation_rows"] = fixture["evaluation_rows"][:24]
    with pytest.raises(ValueError, match="differs from fixture shape"):
        batch._validate_scorer_fixture_timeout(evaluator, fixture)

    fixture["experimental_role"] = "D1"
    with pytest.raises(ValueError, match="differs from fixture role"):
        batch._validate_scorer_fixture_timeout(evaluator, fixture)


@pytest.mark.parametrize(
    "drift", ("source", "requirements", "asset", "runtime", "torch", "cudnn")
)
def test_stage1_exact_scorer_readiness_fails_closed_on_dependency_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    evaluator, contract, snapshots, live_hashes, python_environment = _readiness_case(
        tmp_path, monkeypatch
    )
    if drift == "source":
        snapshots["god"]["tree"] = "f" * 40
    elif drift == "requirements":
        requirement = Path(evaluator["comfy_root"]) / "requirements.txt"
        live_hashes[str(requirement)] = "f" * 64
    elif drift == "asset":
        asset = next(iter(evaluator["expected_assets"].values()))
        live_hashes[asset["canonical_path"]] = "f" * 64
    elif drift == "runtime":
        python_environment["normalized_distributions_sha256"] = "f" * 64
    elif drift == "torch":
        monkeypatch.setattr(
            batch.subprocess,
            "check_output",
            lambda *_args, **_kwargs: json.dumps(
                {
                    **contract["runtime_materialization"]["critical_distributions"],
                    "torch": "0",
                }
            ),
        )
    else:
        def mismatched_cudnn(argv, **_kwargs):
            if "Conv3d" in " ".join(str(item) for item in argv):
                return json.dumps(
                    {
                        **contract["runtime_materialization"]["cuda_runtime_probe"],
                        "cudnn_version": 92000,
                    }
                )
            return json.dumps(
                contract["runtime_materialization"]["critical_distributions"]
            )

        monkeypatch.setattr(batch.subprocess, "check_output", mismatched_cudnn)
    with pytest.raises(ValueError):
        batch._stage1_exact_scorer_readiness(evaluator)


def test_stage1_policy_binds_live_evaluator_and_exact_krea_lock() -> None:
    contract = batch.krea_execution_surface_policy.POLICY[
        "stage1_exact_scorer_contract"
    ]
    assert contract["evaluator_script_sha256"] == krea_provenance.file_sha256(
        _CALIBRATION / "evaluate_krea_local.py"
    )
    lock = (
        _CALIBRATION
        / contract["runtime_materialization"]["exact_lock"]["relative_path"]
    )
    assert (
        krea_provenance.file_sha256(lock)
        == contract["runtime_materialization"]["exact_lock"]["sha256"]
    )
    assert len(lock.read_text(encoding="utf-8").splitlines()) == 229


def test_stage1_policy_binds_split_scorer_support_module_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        batch._validate_scorer_support_modules()
        == batch._SCORER_SUPPORT_MODULE_SHA256
    )
    monkeypatch.setitem(
        batch._SCORER_SUPPORT_MODULE_SHA256,
        "krea_scorer_extension_policy.py",
        "0" * 64,
    )
    with pytest.raises(ValueError, match="support module bytes drifted"):
        batch._validate_scorer_support_modules()


def test_stage1_evaluator_config_builder_emits_complete_live_containment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = batch.krea_execution_surface_policy.POLICY[
        "stage1_exact_scorer_contract"
    ]
    comfy = tmp_path / "ComfyUI"
    god = tmp_path / "G.O.D"
    comfy.mkdir()
    god.mkdir()
    systemd_run = tmp_path / "systemd-run"
    systemctl = tmp_path / "systemctl"
    for binary in (systemd_run, systemctl):
        binary.write_bytes(b"test systemd binary\n")
        binary.chmod(0o700)
    python_path = Path(sys.executable).resolve()
    python_environment = {
        "executable": str(python_path),
        "prefix": str(python_path.parent.parent),
        "base_prefix": str(python_path.parent.parent),
        "python": contract["runtime_materialization"]["python_version"],
        "distribution_count": contract["runtime_materialization"]["distribution_count"],
        "distributions_sha256": contract["runtime_materialization"][
            "distributions_sha256"
        ],
        "normalized_distributions_sha256": contract["runtime_materialization"][
            "exact_lock"
        ]["normalized_name_version_sha256"],
        "requested_executable": str(python_path),
        "venv_root": str(python_path.parent.parent),
        "environment_kind": "venv",
        "identity_marker": str(tmp_path / "pyvenv.cfg"),
        "identity_marker_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        evaluate_krea_local, "_python_environment", lambda _path: python_environment
    )
    result = score_plan.build_stage1_evaluator_config(
        comfy_root=comfy,
        god_root=god,
        python_path=python_path,
        cache_provenance_sha256="b" * 64,
        fixture_role="D1",
        systemd_run_path=systemd_run,
        systemctl_path=systemctl,
    )
    assert result["base_name"] == "krea2_raw_fp8_scaled.safetensors"
    assert result["containment"]["unit_type"] == "transient_service"
    assert result["containment"]["network_policy"] == {
        "private_network": True,
        "restrict_address_families": ["AF_UNIX", "AF_INET", "AF_INET6"],
        "loopback_allowed": True,
        "outbound_network_blocked": True,
    }
    assert result["containment"]["systemd_run_sha256"] == (
        krea_provenance.file_sha256(systemd_run)
    )


def test_stage1_asset_stager_is_pinned_create_only_and_token_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    lora_root = comfy / "models" / "loras"
    lora_root.mkdir(parents=True)
    (lora_root / "put_loras_here").write_bytes(b"")
    sources = tmp_path / "sources"
    sources.mkdir()
    policy = deepcopy(score_plan.krea_execution_surface_policy.POLICY)
    observed = []
    for name, filename in score_plan._KREA_ASSET_SOURCE_PATHS.items():
        content = f"asset:{name}".encode()
        source = sources / Path(filename).name
        source.write_bytes(content)
        row = policy["stage1_exact_scorer_contract"]["assets"][name]
        row["sha256"] = score_plan.hashlib.sha256(content).hexdigest()
        row["bytes"] = len(content)

    def download(*, repo_id, filename, revision, token):
        observed.append((repo_id, filename, revision, token))
        return str(sources / Path(filename).name)

    monkeypatch.setattr(score_plan.krea_execution_surface_policy, "POLICY", policy)
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download)
    )
    receipt_path = tmp_path / "asset-receipt.json"
    receipt = score_plan.stage_stage1_evaluator_assets(
        comfy_root=comfy, token="secret-token", receipt_output=receipt_path
    )
    assert len(observed) == 3
    assert all(row[0] == "Comfy-Org/Krea-2" for row in observed)
    assert all(row[2] == score_plan._KREA_ASSET_REVISION for row in observed)
    assert b"secret-token" not in receipt_path.read_bytes()
    assert receipt["credential_recorded"] is False
    with pytest.raises(FileExistsError):
        score_plan.stage_stage1_evaluator_assets(
            comfy_root=comfy, token="secret-token", receipt_output=receipt_path
        )


@pytest.mark.parametrize(
    ("placeholder_contents", "foreign_name", "error"),
    (
        (b"not-empty", None, "placeholder must be one zero-byte regular file"),
        (b"", "foreign.safetensors", "must be empty"),
        (b"", "foreign-zero-byte", "must be empty"),
    ),
)
def test_stage1_asset_stager_rejects_nonplaceholder_or_real_lora(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    placeholder_contents: bytes,
    foreign_name: str | None,
    error: str,
) -> None:
    comfy = tmp_path / "ComfyUI"
    lora_root = comfy / "models" / "loras"
    lora_root.mkdir(parents=True)
    (lora_root / "put_loras_here").write_bytes(placeholder_contents)
    if foreign_name is not None:
        (lora_root / foreign_name).write_bytes(
            b"foreign-model" if foreign_name.endswith(".safetensors") else b""
        )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            hf_hub_download=lambda **_kwargs: pytest.fail(
                "asset download started before LoRA staging validation"
            )
        ),
    )
    with pytest.raises(ValueError, match=error):
        score_plan.stage_stage1_evaluator_assets(
            comfy_root=comfy,
            token="secret-token",
            receipt_output=tmp_path / "receipt.json",
        )


class BuilderHarness:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch):
        producer_root = root / "producer"
        producer_root.mkdir()
        self.base = ProducerHarness(producer_root, monkeypatch)
        base = self.base
        monkeypatch.setattr(
            score_plan.krea_fixture, "validate_approval", lambda value, **_kwargs: value
        )
        monkeypatch.setattr(
            score_plan.batch,
            "_validate_cross_fixture_review_surface",
            lambda value, **_kwargs: value,
        )
        monkeypatch.setattr(
            score_plan.krea_training_evidence,
            "validate_zero_control",
            lambda value, **_kwargs: value,
        )
        # This harness predates the operational Stage-3 producer and uses a
        # deliberately abbreviated bundle to isolate score-plan projection.
        # Dedicated tests below exercise the real strong validator boundary.
        monkeypatch.setattr(
            score_plan.krea_training_evidence,
            "validate_run_evidence",
            lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
        )
        monkeypatch.setattr(
            score_plan.batch, "_validate_evaluator", lambda value: dict(value)
        )

        discovery_plan = root / "discovery-plan.json"
        discovery_plan_sha = _canonical_file(discovery_plan, {"status": "frozen"})
        base.discovery_plan_sha = discovery_plan_sha
        execution_plan = root / "execution-plan.json"
        execution_plan_sha = _canonical_file(
            execution_plan,
            {
                "plan_sha256": base.execution_sha,
                "discovery_plan": {
                    "path": str(discovery_plan),
                    "sha256": discovery_plan_sha,
                },
            },
        )
        execution_approval = root / "execution-approval.json"
        execution_approval_sha = _canonical_file(
            execution_approval, {"decision": "approved"}
        )
        run_record = root / "run-record.json"
        run_record_sha = _canonical_file(run_record, {"status": "complete"})
        training_log = root / "training.log"
        training_log.write_bytes(b"complete\n")
        training_log_sha = krea_provenance.file_sha256(training_log)
        completion = root / "run-completion.json"
        completion_value = {
            "schema": 3,
            "kind": "forge-krea-training-completion",
            "arm_id": base.arm,
            "execution_plan_sha256": base.execution_sha,
            "natural_completion": True,
            "in_task_proxy_selection": {"enabled": False, "reserve_s": 0},
            "candidates": [
                {
                    "candidate_id": base.sealed_candidate["candidate_id"],
                    "sha256": base.local_sha,
                    "bytes": base.local_path.stat().st_size,
                    "step": 50,
                    "fraction_numerator": 50,
                    "fraction_denominator": 100,
                    "aliases": ["last.safetensors"],
                    "safetensors": {"test": "identity"},
                }
            ],
        }
        completion_sha = _canonical_file(completion, completion_value)
        local_binding_path = base.candidates[0]["provenance_path"]
        local_binding = {
            "schema": 2,
            "kind": "forge-krea-local-candidate-binding",
            "mode": "local_run_candidate",
            "arm_id": base.arm,
            "candidate_id": base.sealed_candidate["candidate_id"],
            "candidate": {
                "path": str(base.local_path),
                "sha256": base.local_sha,
                "bytes": base.local_path.stat().st_size,
                "step": 50,
                "fraction_numerator": 50,
                "fraction_denominator": 100,
                "aliases": ["last.safetensors"],
                "safetensors": {"test": "identity"},
            },
            "execution_plan": {
                "path": str(execution_plan),
                "sha256": execution_plan_sha,
            },
            "execution_approval": {
                "path": str(execution_approval),
                "sha256": execution_approval_sha,
            },
            "run_completion": {"path": str(completion), "sha256": completion_sha},
            "run_record": {"path": str(run_record), "sha256": run_record_sha},
            "training_log": {
                "path": str(training_log),
                "sha256": training_log_sha,
            },
            "evaluation_dataset_sha256": base.dataset_sha,
        }
        local_binding_sha = _canonical_file(local_binding_path, local_binding)
        base.candidates[0]["provenance_file_sha256"] = local_binding_sha
        base.candidates[0]["candidate_binding"][
            "binding_manifest_sha256"
        ] = local_binding_sha
        base.candidates[0]["candidate_binding"][
            "run_completion_sha256"
        ] = completion_sha

        self.bundle_path = root / "bundle.json"
        bundle_body = {
            "schema": 2,
            "kind": "forge-krea-run-evidence-bundle",
            "arm_id": base.arm,
            "execution_plan_sha256": base.execution_sha,
            "discovery_profile_index_sha256": "1" * 64,
            "discovery_execution_authorization_sha256": "2" * 64,
            "host_bootstrap_receipt_sha256": "3" * 64,
            "execution_surface_policy_sha256": (
                score_plan.krea_execution_surface_policy.POLICY["policy_sha256"]
            ),
            "execution_surface": "staged_host_venv",
            "execution_scope": "discovery_only",
            "run_completion": {"path": str(completion), "sha256": completion_sha},
            "candidate_bindings": [
                {
                    "candidate_id": base.sealed_candidate["candidate_id"],
                    "candidate_sha256": base.local_sha,
                    "binding": {
                        "path": str(local_binding_path),
                        "sha256": local_binding_sha,
                    },
                }
            ],
        }
        self.bundle = {
            **bundle_body,
            "bundle_sha256": krea_provenance.canonical_sha256(bundle_body),
        }
        _canonical_file(self.bundle_path, self.bundle)

        self.zero_manifest_path = root / "zero-manifest.json"
        zero_body = {
            "schema": 2,
            "kind": "forge-krea-zero-lora-control",
            "mode": "zero_lora_control",
            "artifact": {
                "path": str(base.zero_path),
                "sha256": base.zero_sha,
                "bytes": base.zero_path.stat().st_size,
            },
            "evaluation_dataset_sha256": base.dataset_sha,
        }
        self.zero_manifest = {
            **zero_body,
            "manifest_sha256": base.zero_manifest_sha,
        }
        self.zero_manifest_file_sha = _canonical_file(
            self.zero_manifest_path, self.zero_manifest
        )
        base.candidates[1]["provenance_path"] = self.zero_manifest_path
        base.candidates[1]["provenance_file_sha256"] = self.zero_manifest_file_sha
        base.candidates[1]["candidate_binding"][
            "binding_manifest_sha256"
        ] = self.zero_manifest_file_sha

        self.evaluator_config_path = root / "evaluator.json"
        evaluator = {
            key: deepcopy(value)
            for key, value in base.evaluator.items()
            if not key.startswith("_")
        }
        evaluator["cache_provenance_sha256"] = score_plan.hashlib.sha256(
            b"cache"
        ).hexdigest()
        _canonical_file(self.evaluator_config_path, evaluator)
        self.output_dir = root / "built"

    def kwargs(self, *, phase: str = "boundary") -> dict:
        return {
            "bundle_paths": [self.bundle_path],
            "zero_manifest_path": self.zero_manifest_path,
            "dataset_path": self.base.dataset,
            "fixture_manifest_path": self.base.fixture_path,
            "fixture_approval_path": self.base.fixture_approval_path,
            "cross_fixture_review_path": self.base.cross_review_path,
            "evaluator_config_path": self.evaluator_config_path,
            "phase": phase,
            "campaign_output_path": self.output_dir / "campaign.json",
            "frozen_discovery_decision": (
                self.base.discovery_path if phase == "boundary" else None
            ),
            "candidate_family": self.base.arm if phase == "boundary" else None,
        }

    def build(self) -> tuple[dict, dict, Path, Path]:
        campaign, draft = score_plan.build_documents(**self.kwargs())
        campaign_path, draft_path = score_plan.publish_build(
            output_dir=self.output_dir, campaign=campaign, draft=draft
        )
        return campaign, draft, campaign_path, draft_path


def test_daf9_training_bundle_routes_through_historical_validator_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = BuilderHarness(tmp_path, monkeypatch)
    identity = {
        "schema": 1,
        "kind": "forge-krea-historical-training-evidence-validator",
        "root": "/admitted/forge-daf9a252",
        "commit_sha1": "daf9a2528f4079ed06180c7e6d712a684a4170f0",
        "tree_sha1": "953b58bdca842294ef7dfa1e54a16db52e5b74a2",
        "execution_surface_policy_sha256": (
            "98b59fd90dbf4ea213c860f873bc472cadc66714c7b9118672de2474f020f5f3"
        ),
        "module_sha256": {},
    }
    calls = []

    def validate(path, supplied_identity):
        calls.append((Path(path), supplied_identity))
        return json.loads(Path(path).read_text(encoding="utf-8"))

    monkeypatch.setattr(
        score_plan.krea_historical_training_evidence,
        "validate_run_evidence",
        validate,
    )
    admitted = json.loads(harness.bundle_path.read_text(encoding="utf-8"))
    admitted["execution_surface_policy_sha256"] = identity[
        "execution_surface_policy_sha256"
    ]
    admitted_body = {
        key: value for key, value in admitted.items() if key != "bundle_sha256"
    }
    admitted["bundle_sha256"] = krea_provenance.canonical_sha256(admitted_body)
    _canonical_file(harness.bundle_path, admitted)
    run, rows, *_ = score_plan._bundle_candidates(
        harness.bundle_path,
        historical_validator_identity=identity,
    )
    assert run["arm_id"] == harness.base.arm
    assert rows
    assert calls == [(harness.bundle_path, identity)]

    incompatible = json.loads(harness.bundle_path.read_text(encoding="utf-8"))
    incompatible["execution_surface_policy_sha256"] = (
        "8a45666ea1555600de79f657055c5d540a9a87e8a6807640059011f3ee540b3f"
    )
    body = {
        key: value for key, value in incompatible.items() if key != "bundle_sha256"
    }
    incompatible["bundle_sha256"] = krea_provenance.canonical_sha256(body)
    _canonical_file(harness.bundle_path, incompatible)
    with pytest.raises(ValueError, match="invalid run-evidence bundle"):
        score_plan._bundle_candidates(
            harness.bundle_path,
            historical_validator_identity=identity,
        )


def test_build_is_unapproved_and_dry_run_performs_no_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    harness = BuilderHarness(tmp_path, monkeypatch)
    dry_target = tmp_path / "dry-target"
    args = [
        "dry-run",
        "--bundle",
        str(harness.bundle_path),
        "--zero-manifest",
        str(harness.zero_manifest_path),
        "--dataset",
        str(harness.base.dataset),
        "--fixture-manifest",
        str(harness.base.fixture_path),
        "--fixture-approval",
        str(harness.base.fixture_approval_path),
        "--cross-fixture-review",
        str(harness.base.cross_review_path),
        "--evaluator-config",
        str(harness.evaluator_config_path),
        "--phase",
        "boundary",
        "--output-dir",
        str(dry_target),
        "--frozen-discovery-decision",
        str(harness.base.discovery_path),
        "--candidate-family",
        harness.base.arm,
    ]
    assert score_plan.main(args) == 0
    assert json.loads(capsys.readouterr().out)["writes_performed"] is False
    assert not dry_target.exists()

    _campaign, draft, _campaign_path, draft_path = harness.build()
    assert "sealed_plan_approval" not in draft
    assert set(path.name for path in harness.output_dir.iterdir()) == {
        "campaign.json",
        "score-plan.draft.json",
    }
    assert score_plan.main(["validate", "--draft", str(draft_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "human_approval_present": False,
        "status": "draft_valid",
    }


def test_approval_is_a_separate_command_and_final_plan_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    harness = BuilderHarness(tmp_path, monkeypatch)
    _campaign, _draft, _campaign_path, draft_path = harness.build()
    approval_path = tmp_path / "human-approval.json"
    plan_path = tmp_path / "score-plan.json"
    assert (
        score_plan.main(
            [
                "approve",
                "--draft",
                str(draft_path),
                "--reviewer-identity",
                "Morgan Auditor",
                "--approval-output",
                str(approval_path),
                "--plan-output",
                str(plan_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "approved_plan_published"
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert approval["reviewer_identity"] == "Morgan Auditor"
    assert plan["sealed_plan_approval"] == {
        "path": str(approval_path),
        "sha256": krea_provenance.file_sha256(approval_path),
    }
    assert score_plan.main(["validate", "--plan", str(plan_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "approved_plan_valid"


def test_fake_stage3_builder_to_batch_to_decision_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = BuilderHarness(tmp_path, monkeypatch)
    campaign, _draft, campaign_path, draft_path = harness.build()
    base = harness.base
    approval_path = tmp_path / "round-trip-approval.json"
    _approval, executable = score_plan.approve_draft(
        draft_path=draft_path,
        reviewer_identity="Round Trip Reviewer",
        approval_output=approval_path,
        plan_output=tmp_path / "round-trip-plan.json",
    )
    base.score_approval_path = approval_path
    base.score_approval_sha = krea_provenance.file_sha256(approval_path)
    base.evaluator["_sealed_plan_approval_path"] = str(approval_path)
    base.evaluator["_sealed_plan_approval_sha256"] = base.score_approval_sha
    base.evaluator["_sealed_plan_approval"] = {
        "decision": "approved",
        "reviewer_identity": "Round Trip Reviewer",
    }
    base.evaluator["_plan_payload_sha256"] = batch._plan_payload_sha256(executable)
    base.evaluator["_campaign_manifest_path"] = str(campaign_path)
    base.evaluator["_campaign_manifest_file_sha256"] = krea_provenance.file_sha256(
        campaign_path
    )
    base.evaluator["_campaign_manifest_sha256"] = campaign["manifest_sha256"]
    base.evaluator["_decision_context"] = batch._validate_score_decision_context(
        executable["decision_context"], campaign=campaign
    )
    base.campaign = campaign
    base.campaign_path = campaign_path
    base.campaign_file_sha = krea_provenance.file_sha256(campaign_path)
    # The mocked batch validator returns ``base.candidates`` directly; keep its
    # synthetic zero-control id aligned with the executable score plan that the
    # real validator would return.
    base.candidates[1]["id"] = "K0-zero-control"

    aggregate_path = tmp_path / "builder-aggregate.json"
    aggregate = batch.run_batch(
        executable,
        results_dir=tmp_path / "builder-results",
        output=aggregate_path,
    )
    normalized, _ = krea_decision._aggregate(aggregate_path)
    assert normalized["campaign"]["runs"] == campaign["runs"]
    base.patch_match_context(monkeypatch)
    observed, _bindings = krea_decision._match_aggregates(
        policy=base.boundary_policy(aggregate), aggregate_paths=[aggregate_path]
    )
    assert (
        observed["B-0p5-small"]["zero"]["zero_control_manifest_sha256"]
        == base.zero_manifest_sha
    )


def test_builder_rejects_truncation_and_phase_ambiguity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = BuilderHarness(tmp_path, monkeypatch)
    truncated = deepcopy(harness.bundle)
    truncated["candidate_bindings"] = []
    body = {key: value for key, value in truncated.items() if key != "bundle_sha256"}
    truncated["bundle_sha256"] = krea_provenance.canonical_sha256(body)
    truncated_path = tmp_path / "truncated-bundle.json"
    _canonical_file(truncated_path, truncated)
    kwargs = harness.kwargs()
    kwargs["bundle_paths"] = [truncated_path]
    with pytest.raises(ValueError, match="coverage is incomplete"):
        score_plan.build_documents(**kwargs)

    with pytest.raises(ValueError, match="boundary-ambiguous"):
        score_plan.build_documents(**harness.kwargs(phase="discovery"))


def test_publish_rejects_a_draft_bound_to_another_campaign_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = BuilderHarness(tmp_path, monkeypatch)
    campaign, draft = score_plan.build_documents(**harness.kwargs())
    draft["campaign_manifest"]["path"] = str(tmp_path / "elsewhere.json")
    with pytest.raises(ValueError, match="does not name the campaign"):
        score_plan.publish_build(
            output_dir=harness.output_dir,
            campaign=campaign,
            draft=draft,
        )
    assert not harness.output_dir.exists()


def test_draft_validation_rejects_a_tampered_fixture_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = BuilderHarness(tmp_path, monkeypatch)
    _campaign, _draft, _campaign_path, draft_path = harness.build()
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    draft["fixture_manifest"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fixture manifest file binding"):
        score_plan.validate_draft(draft)


def test_consumer_rejects_fully_rehashed_arbitrary_candidate_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A digest-consistent JSON chain cannot replace safetensors validation."""

    approved_run = stage3_test.approved_run.__wrapped__(tmp_path, monkeypatch)

    # Give the synthetic Stage-3 fixture the discovery binding consumed by the
    # score-plan projection.  Its real plan validator is already replaced by
    # the fixture's strict equality stub, so mutate the shared plan before emit.
    discovery_path = approved_run["root"] / "discovery-plan.json"
    discovery_sha = _canonical_file(discovery_path, {"status": "frozen"})
    approved_run["plan"]["discovery_plan"] = {
        "path": str(discovery_path),
        "sha256": discovery_sha,
    }
    plan_file_sha = _canonical_file(approved_run["plan_path"], approved_run["plan"])
    condition = json.loads(approved_run["condition_path"].read_text())
    condition["execution_plan_file_sha256"] = plan_file_sha
    _canonical_file(approved_run["condition_path"], condition)
    monkeypatch.setattr(
        score_plan.krea_training_evidence.krea_execution_plan,
        "validate_approval",
        lambda value, **_kwargs: value,
    )
    bundle_path = stage3_test._emit(approved_run)
    bundle = json.loads(bundle_path.read_text())

    target_ref = bundle["candidate_bindings"][0]
    target_binding_path = Path(target_ref["binding"]["path"])
    target_binding = json.loads(target_binding_path.read_text())
    target_artifact = Path(target_binding["candidate"]["path"])
    target_artifact.chmod(0o600)
    target_artifact.write_bytes(b"attacker-chosen arbitrary candidate bytes")
    forged_sha = krea_provenance.file_sha256(target_artifact)
    forged_bytes = target_artifact.stat().st_size
    forged_layout = {"attacker_assertion": "not-a-safetensors-identity"}
    target_id = target_ref["candidate_id"]
    target_step = target_binding["candidate"]["step"]
    forged_id = f"step-{target_step}-{forged_sha[:12]}"
    forged_artifact = target_artifact.with_name(f"{forged_id}.safetensors")
    target_artifact.parent.chmod(0o700)
    target_artifact.rename(forged_artifact)
    target_artifact.parent.chmod(0o500)

    # Rehash the run record, completion, every candidate binding, and bundle.
    # This models the exact weakness of a consumer that checks only JSON hashes.
    run_record_path = Path(target_binding["run_record"]["path"])
    run_record = json.loads(run_record_path.read_text())
    aliases = target_binding["candidate"]["aliases"]
    for alias in aliases:
        run_record["current_scope_candidates"][alias["name"]] = forged_sha
        run_record["artifacts"]["candidate_sha256"][alias["name"]] = forged_sha
    run_record_sha = _canonical_file(run_record_path, run_record)

    completion_path = Path(bundle["run_completion"]["path"])
    completion = json.loads(completion_path.read_text())
    completion["run_record_sha256"] = run_record_sha
    completion_target = next(
        row for row in completion["candidates"] if row["candidate_id"] == target_id
    )
    completion_target["candidate_id"] = forged_id
    completion_target["sha256"] = forged_sha
    completion_target["bytes"] = forged_bytes
    completion_target["safetensors"] = forged_layout
    completion_sha = _canonical_file(completion_path, completion)

    for row in bundle["candidate_bindings"]:
        binding_path = Path(row["binding"]["path"])
        binding = json.loads(binding_path.read_text())
        binding["run_record"]["sha256"] = run_record_sha
        binding["run_completion"]["sha256"] = completion_sha
        if binding["candidate_id"] == target_id:
            binding["candidate_id"] = forged_id
            binding["candidate"]["path"] = str(forged_artifact)
            binding["candidate"]["sha256"] = forged_sha
            binding["candidate"]["bytes"] = forged_bytes
            binding["candidate"]["safetensors"] = forged_layout
            row["candidate_id"] = forged_id
            row["candidate_sha256"] = forged_sha
        row["binding"]["sha256"] = _canonical_file(binding_path, binding)
    bundle["run_completion"]["sha256"] = completion_sha
    bundle_body = {
        key: value for key, value in bundle.items() if key != "bundle_sha256"
    }
    bundle["bundle_sha256"] = krea_provenance.canonical_sha256(bundle_body)
    _canonical_file(bundle_path, bundle)

    strong_validator = score_plan.krea_training_evidence.validate_run_evidence
    monkeypatch.setattr(
        score_plan.krea_training_evidence,
        "validate_run_evidence",
        lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )
    # The previous manual projection accepts the fully rehashed forgery.
    projected, *_rest = score_plan._bundle_candidates(bundle_path)
    assert projected["candidates"][0]["sha256"] == forged_sha

    monkeypatch.setattr(
        score_plan.krea_training_evidence,
        "validate_run_evidence",
        strong_validator,
    )
    with pytest.raises(ValueError, match="safetensors"):
        score_plan._bundle_candidates(bundle_path)
