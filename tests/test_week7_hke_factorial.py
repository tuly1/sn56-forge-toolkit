"""CPU-only fail-closed tests for the Week-7 Krea factor screen."""

from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml

from forge import adaptive_timing, krea_runtime


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ops" / "experiments" / "week7" / "run_hke_factorial.py"
SPEC = importlib.util.spec_from_file_location("run_hke_factorial", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
H = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = H
SPEC.loader.exec_module(H)


@pytest.fixture
def base_config():
    return yaml.safe_load(H.INCUMBENT_TEMPLATE_PATH.read_text(encoding="utf-8"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _profile(
    loss: str,
    seconds_per_step: float,
    *,
    dataset_size: int,
    accelerator: str = "NVIDIA H100 PCIe|81559-MiB",
):
    completed = 100
    first = 20
    value = adaptive_timing.ThroughputProfile(
        bundle_id=krea_runtime.INCUMBENT_BUNDLE,
        bundle_sha256=krea_runtime.bundle_contract_sha256(
            krea_runtime.INCUMBENT_BUNDLE
        ),
        model_type="krea2",
        measured_dataset_size=dataset_size,
        dataset_regime=adaptive_timing.dataset_regime(dataset_size),
        seconds_per_step=seconds_per_step,
        startup_seconds=0.0,
        completed_steps=completed,
        training_elapsed_seconds=completed * seconds_per_step,
        first_checkpoint_step=first,
        first_checkpoint_elapsed_seconds=first * seconds_per_step,
        source_run_id=f"week7-{loss}-probe:{'a' * 32}",
        source_record_sha256=_sha(f"record-{loss}-{dataset_size}-{seconds_per_step}"),
        runtime_commit=krea_runtime.PINNED_BASE_COMMIT,
        measured_at_utc="2026-08-10T20:00:00Z",
        accelerator_identity=accelerator,
        profile_sha256="0" * 64,
    )
    document = H._profile_document(value)
    document.pop("profile_sha256")
    return replace(
        value,
        profile_sha256=adaptive_timing.canonical_sha256(document),
    )


def _bound_profiles(
    base_config,
    *,
    count: int,
    hours: float = 0.75,
    mae_rate: float = 0.8,
    mse_rate: float = 0.9,
    accelerator: str = "NVIDIA H100 PCIe|81559-MiB",
):
    current = H.materialize_current_law_configs(
        base_config, num_images=count, hours_to_complete=hours
    )
    result = {}
    for loss, arm, rate in (("mae", "A", mae_rate), ("mse", "B", mse_rate)):
        result[loss] = H.bind_timing_profile(
            _profile(
                loss,
                rate,
                dataset_size=count,
                accelerator=accelerator,
            ),
            loss=loss,
            measured_dataset_size=count,
            measured_config_sha256=H.canonical_sha256(current[arm]),
        )
    return result


def _replay_evidence():
    return {
        "status": "PASS",
        "verified_rows": 98,
        "candidate_semantic_sha256": "1" * 64,
        "dedup_semantic_sha256": "4" * 64,
    }


def _discovery_row_identity(family: str):
    return H.canonical_sha256(
        [
            {
                "row_id": f"{family}-row-{index:02d}",
                "row_sha256": _sha(
                    f"public-discovery-row-record:{family}:{index}"
                ),
            }
            for index in range(
                H.EXPECTED_FIXTURE_COUNTS[family]["discovery"]
            )
        ]
    )


def _admission(family: str):
    body = {
        "schema": 1,
        "kind": "sn56-week7-hke-fixture-admission",
        "status": "PASS",
        "family": family,
        "counts": copy.deepcopy(H.EXPECTED_FIXTURE_COUNTS[family]),
        "governance": {
            "operator_attested_named_human_review": True,
            "agent_review_is_not_human_review": True,
            "admission_authorized": True,
            "gpu_execution_authorized": False,
            "owner_ratification_required_for_gpu": True,
        },
        "candidate_semantic_sha256": "1" * 64,
        "human_review_sha256": "2" * 64,
        "replay_evidence_sha256": H.canonical_sha256(_replay_evidence()),
        "dedup_evidence_sha256": "4" * 64,
        "ownership_record_sha256": "5" * 64,
        "confirmation_commitment_sha256": "5" * 64,
        "generator_source_sha256": "7" * 64,
        "admission_authority_source_sha256": H.sha256_file(
            H.ADMISSION_AUTHORITY_PATH
        ),
        "discovery_row_identity_sha256": _discovery_row_identity(family),
        "generator_commit": "8" * 40,
        "generator_tree": "9" * 40,
        "claim_limit": (
            "admitted for this procedural instrument only; field and "
            "opponent-relative transfer remain unproven"
        ),
    }
    return {**body, "admission_sha256": H.canonical_sha256(body)}


def _admission_set():
    receipts = {
        family: _admission(family) for family in H.EXPECTED_FIXTURE_COUNTS
    }
    replay = _replay_evidence()
    body = {
        "schema": 1,
        "kind": "sn56-week7-hke-fixture-admission-set",
        "status": "PASS",
        "candidate_semantic_sha256": "1" * 64,
        "human_review_sha256": "2" * 64,
        "replay_evidence": replay,
        "replay_evidence_sha256": H.canonical_sha256(replay),
        "dedup_evidence_sha256": "4" * 64,
        "ownership_record_sha256": "5" * 64,
        "generator_revision": {
            "repository": "https://github.com/tuly1/sn56-forge-toolkit.git",
            "commit": "8" * 40,
            "tree": "9" * 40,
            "renderer_source_sha256": "7" * 64,
            "admission_authority_source_sha256": H.sha256_file(
                H.ADMISSION_AUTHORITY_PATH
            ),
            "pinned_remote_refs": ["refs/heads/codex/week7-hke-cpu-prelaunch"],
        },
        "family_admission_sha256": {
            family: receipt["admission_sha256"]
            for family, receipt in receipts.items()
        },
        "receipts": receipts,
        "authorization": {
            "fixture_admission_authorized": True,
            "gpu_execution_authorized": False,
            "deployment_authorized": False,
        },
    }
    return {**body, "admission_set_sha256": H.canonical_sha256(body)}


def _rehash_admission_set(value):
    """Rehash a deliberately forged root so tests reach semantic checks."""

    result = copy.deepcopy(value)
    for family, receipt in result["receipts"].items():
        body = dict(receipt)
        body.pop("admission_sha256", None)
        receipt["admission_sha256"] = H.canonical_sha256(body)
        result["family_admission_sha256"][family] = receipt[
            "admission_sha256"
        ]
    body = dict(result)
    body.pop("admission_set_sha256", None)
    result["admission_set_sha256"] = H.canonical_sha256(body)
    return result


def _evaluator():
    return {
        "harness_sha256": _sha("exact-evaluator-harness"),
        "god_commit": "a" * 40,
        "comfy_commit": "b" * 40,
        "defaults_sha256": _sha("exact-evaluator-defaults"),
    }


def _plan(base_config):
    profiles = {
        family: _bound_profiles(
            base_config,
            count=counts["discovery"],
        )
        for family, counts in H.EXPECTED_FIXTURE_COUNTS.items()
    }
    return H.build_prelaunch_plan(
        base_config,
        admission_set=_admission_set(),
        profiles_by_family=profiles,
        evaluator_identity=_evaluator(),
    )


def _rows(family: str, value: float, n: int):
    return [
        {
            "row_id": f"{family}-row-{index:02d}",
            "row_sha256": _sha(
                f"public-discovery-row-record:{family}:{index}"
            ),
            "prompted_loss": value,
            "blank_loss": value,
        }
        for index in range(n)
    ]


def _cell(plan, family: str, arm: str, value: float, n: int):
    planned = plan["cells"][family][arm]["planned_steps"]
    artifact_sha = _sha(f"artifact-{family}-{arm}")
    rows = _rows(family, value, n)
    evidence_class = "content_bound_operator_attested_not_independent_proof"
    config_sha = plan["cells"][family][arm]["config_sha256"]
    training = {
        "schema": 1,
        "kind": "sn56-week7-hke-training-receipt",
        "status": "OPERATOR_ATTESTED_PASS",
        "evidence_class": evidence_class,
        "plan_sha256": plan["plan_sha256"],
        "config_sha256": config_sha,
        "training_seed": H.TRAINING_SEED_A,
        "planned_steps": planned,
        "sha256": artifact_sha,
        "bytes": 1024,
        "checkpoint_step": planned,
        "completed_steps": planned,
        "source_record_sha256": _sha(f"train-source-{family}-{arm}"),
    }
    training["training_receipt_sha256"] = H.canonical_sha256(training)
    attachment = {
        "schema": 1,
        "kind": "sn56-week7-hke-attachment-receipt",
        "status": "OPERATOR_ATTESTED_PASS",
        "evidence_class": evidence_class,
        "plan_sha256": plan["plan_sha256"],
        "config_sha256": config_sha,
        "artifact_sha256": artifact_sha,
        "loaded_key_count": 512,
        "unloaded_key_count": 0,
        "log_sha256": _sha(f"attach-{family}-{arm}"),
    }
    attachment["attachment_receipt_sha256"] = H.canonical_sha256(attachment)
    score = {
        "schema": 1,
        "kind": "sn56-week7-hke-exact-score-receipt",
        "status": "OPERATOR_ATTESTED_PASS",
        "evidence_class": evidence_class,
        "plan_sha256": plan["plan_sha256"],
        "config_sha256": config_sha,
        "artifact_sha256": artifact_sha,
        "evaluator_sha256": plan["evaluator_sha256"],
        "fixture_candidate_semantic_sha256": plan["fixtures"][family][
            "candidate_semantic_sha256"
        ],
        "rows": rows,
        "rows_sha256": H.canonical_sha256(rows),
    }
    score["score_receipt_sha256"] = H.canonical_sha256(score)
    return {
        "training_receipt": training,
        "attachment_receipt": attachment,
        "score_receipt": score,
    }


def _evidence(plan, *, logo_regression: bool = False):
    families = {}
    for family in ("product", "logo_ui"):
        values = (
            {"A": 1.0, "B": 1.02, "C": 1.03, "D": 1.05}
            if family == "logo_ui" and logo_regression
            else {"A": 1.0, "B": 0.9, "C": 0.8, "D": 0.7}
        )
        arms = {
            arm: _cell(
                plan,
                family,
                arm,
                value,
                H.EXPECTED_FIXTURE_COUNTS[family]["discovery"],
            )
            for arm, value in values.items()
        }
        identity = [
            {"row_id": row["row_id"], "row_sha256": row["row_sha256"]}
            for row in arms["A"]["score_receipt"]["rows"]
        ]
        families[family] = {
            "phase": "discovery",
            "fixture_admission_sha256": plan["fixtures"][family][
                "admission_sha256"
            ],
            "fixture_candidate_semantic_sha256": plan["fixtures"][family][
                "candidate_semantic_sha256"
            ],
            "discovery_row_identity_sha256": H.canonical_sha256(identity),
            "arms": arms,
        }
    social = {
        "A": _cell(
            plan,
            "social",
            "A",
            1.0,
            H.EXPECTED_FIXTURE_COUNTS["social"]["discovery"],
        ),
        "D": _cell(
            plan,
            "social",
            "D",
            0.95,
            H.EXPECTED_FIXTURE_COUNTS["social"]["discovery"],
        ),
    }
    identity = [
        {"row_id": row["row_id"], "row_sha256": row["row_sha256"]}
        for row in social["A"]["score_receipt"]["rows"]
    ]
    families["social"] = {
        "phase": "discovery",
        "fixture_admission_sha256": plan["fixtures"]["social"][
            "admission_sha256"
        ],
        "fixture_candidate_semantic_sha256": plan["fixtures"]["social"][
            "candidate_semantic_sha256"
        ],
        "discovery_row_identity_sha256": H.canonical_sha256(identity),
        "arms": social,
    }
    body = {
        "schema": 1,
        "kind": "sn56-week7-hke-score-evidence",
        "evidence_class": (
            "content_bound_operator_attested_not_independent_proof"
        ),
        "plan_sha256": plan["plan_sha256"],
        "evaluator_sha256": plan["evaluator_sha256"],
        "training_seed": H.TRAINING_SEED_A,
        "families": families,
    }
    return {**body, "evidence_sha256": H.canonical_sha256(body)}


def _rehash_score(cell):
    body = dict(cell["score_receipt"])
    body.pop("score_receipt_sha256", None)
    cell["score_receipt"]["score_receipt_sha256"] = H.canonical_sha256(body)


def _rehash_training(cell):
    body = dict(cell["training_receipt"])
    body.pop("training_receipt_sha256", None)
    cell["training_receipt"]["training_receipt_sha256"] = H.canonical_sha256(
        body
    )


def _rehash_attachment(cell):
    body = dict(cell["attachment_receipt"])
    body.pop("attachment_receipt_sha256", None)
    cell["attachment_receipt"]["attachment_receipt_sha256"] = (
        H.canonical_sha256(body)
    )


def _rehash_evidence(evidence):
    body = dict(evidence)
    body.pop("evidence_sha256", None)
    evidence["evidence_sha256"] = H.canonical_sha256(body)


def _rehash_plan(plan):
    body = dict(plan)
    body.pop("plan_sha256", None)
    plan["plan_sha256"] = H.canonical_sha256(body)


def test_contract_binds_exact_source_seed_and_stays_calibration_only():
    contract = H.experiment_contract()
    assert contract["incumbent_source"] == {
        "path": "forge/templates/base_diffusion_krea2.yaml",
        "file_sha256": H.INCUMBENT_TEMPLATE_FILE_SHA256,
    }
    assert contract["training_seed"] == {"role": "Seed-A", "value": 42565431}
    assert contract["runtime"]["owned_runtime_enabled"] is False
    assert contract["checkpoint_policy"]["decision_checkpoint"] == "terminal_only"
    assert contract["checkpoint_policy"]["live_promotion_enabled"] is False
    assert contract["routing"]["semantic_router_enabled"] is False
    assert contract["clock_fill"]["proof_of_measurement"] is False
    assert contract["clock_fill"]["source"].startswith("operator_attested_")
    assert "excludes seed, rerun, and hardware variance" in contract["decision"][
        "uncertainty_scope"
    ]


def test_prelaunch_rejects_semantically_or_byte_drifted_incumbent(base_config):
    drifted = copy.deepcopy(base_config)
    drifted["config"]["process"][0]["train"]["lr"] = 7e-5
    with pytest.raises(H.HKEContractError, match="exact reviewed incumbent"):
        H.build_prelaunch_plan(
            drifted,
            admission_set=_admission_set(),
            profiles_by_family={
                family: _bound_profiles(
                    drifted, count=counts["discovery"]
                )
                for family, counts in H.EXPECTED_FIXTURE_COUNTS.items()
            },
            evaluator_identity=_evaluator(),
        )


def test_unequal_measured_rates_produce_one_conservative_clock_plan(base_config):
    arms = H.materialize_arms(
        base_config,
        num_images=28,
        hours_to_complete=0.75,
        profiles=_bound_profiles(
            base_config,
            count=28,
            mae_rate=0.8,
            mse_rate=1.1,
        ),
    )
    assert arms["A"]["config"]["process"][0]["training_seed"] == H.TRAINING_SEED_A
    assert arms["B"]["config"]["process"][0]["training_seed"] == H.TRAINING_SEED_A
    assert H._train_node(arms["C"])["steps"] == H._train_node(arms["D"])["steps"]
    assert H.changed_pointers(arms["A"], arms["B"]) == {
        "/config/process/0/train/loss_type"
    }
    assert H.changed_pointers(arms["C"], arms["D"]) == {
        "/config/process/0/train/loss_type"
    }


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda bound: replace(bound, binding_sha256="f" * 64), "outer binding"),
        (
            lambda bound: replace(
                bound,
                profile=replace(bound.profile, profile_sha256="f" * 64),
            ),
            "internal digest",
        ),
        (
            lambda bound: replace(bound, measured_config_sha256="f" * 64),
            "config binding",
        ),
        (
            lambda bound: replace(bound, accelerator_identity="other-gpu"),
            "accelerator binding",
        ),
    ],
)
def test_forged_or_unbound_timing_profile_aborts(base_config, mutation, match):
    profiles = _bound_profiles(base_config, count=28)
    profiles["mae"] = mutation(profiles["mae"])
    with pytest.raises(H.HKEContractError, match=match):
        H.materialize_arms(
            base_config,
            num_images=28,
            hours_to_complete=0.75,
            profiles=profiles,
        )


def test_non_h100_timing_profile_aborts(base_config):
    profiles = _bound_profiles(
        base_config,
        count=28,
        accelerator="NVIDIA RTX 4090|24564-MiB",
    )
    with pytest.raises(H.HKEContractError, match="H100 80GB"):
        H.materialize_arms(
            base_config,
            num_images=28,
            hours_to_complete=0.75,
            profiles=profiles,
        )


def test_prelaunch_rejects_profiles_from_multiple_h100_identities(base_config):
    profiles = {
        family: _bound_profiles(
            base_config,
            count=counts["discovery"],
            accelerator=(
                "NVIDIA H100 SXM|81559-MiB"
                if family == "social"
                else "NVIDIA H100 PCIe|81559-MiB"
            ),
        )
        for family, counts in H.EXPECTED_FIXTURE_COUNTS.items()
    }
    with pytest.raises(H.HKEContractError, match="one H100 80GB"):
        H.build_prelaunch_plan(
            base_config,
            admission_set=_admission_set(),
            profiles_by_family=profiles,
            evaluator_identity=_evaluator(),
        )


def test_prelaunch_plan_binds_configs_fixtures_evaluator_and_seed(base_config):
    plan = _plan(base_config)
    assert plan["source"]["file_sha256"] == H.INCUMBENT_TEMPLATE_FILE_SHA256
    assert plan["training_seed"] == {"role": "Seed-A", "value": H.TRAINING_SEED_A}
    assert plan["evaluator_sha256"] == H.canonical_sha256(_evaluator())
    assert plan["admission_set_sha256"] == plan["admission_set"][
        "admission_set_sha256"
    ]
    for family in H.EXPECTED_FIXTURE_COUNTS:
        for loss in ("mae", "mse"):
            assert set(plan["timing_profiles"][family][loss]) == {
                "profile",
                "binding",
            }
    assert H._validate_plan(plan)["plan_sha256"] == plan["plan_sha256"]
    for family, cells in plan["cells"].items():
        for arm, cell in cells.items():
            assert cell["config_sha256"] == H.canonical_sha256(cell["config"])
            assert cell["training_seed"] == H.TRAINING_SEED_A
            if H.ARMS[arm]["planner"] == "measured_clock_fill":
                assert set(cell["timing_support"]["profile_sha256"]) == {
                    "mae",
                    "mse",
                }
                assert set(cell["timing_support"]["binding_sha256"]) == {
                    "mae",
                    "mse",
                }


@pytest.mark.parametrize("forge", ("fixture", "profile", "clock", "social_mse"))
def test_serialized_plan_is_semantically_reconstructed(base_config, forge):
    plan = _plan(base_config)
    if forge == "fixture":
        plan["fixtures"]["product"]["admission_sha256"] = "f" * 64
    elif forge == "profile":
        plan["timing_profiles"]["product"]["mae"]["profile"][
            "seconds_per_step"
        ] = 0.5
    elif forge == "clock":
        cell = plan["cells"]["product"]["C"]
        cell["config"]["config"]["process"][0]["train"]["steps"] -= 1
        cell["planned_steps"] -= 1
        cell["config_sha256"] = H.canonical_sha256(cell["config"])
    else:
        binding = plan["timing_profiles"]["social"]["mse"]["binding"]
        binding["measured_config_sha256"] = "f" * 64
        body = dict(binding)
        body.pop("binding_sha256")
        binding["binding_sha256"] = H.canonical_sha256(body)
    _rehash_plan(plan)
    with pytest.raises(H.HKEContractError):
        H._validate_plan(plan)


@pytest.mark.parametrize("forge", ("root", "governance", "authority"))
def test_prelaunch_rejects_unbound_or_forged_admission_set(base_config, forge):
    admission_set = _admission_set()
    if forge == "root":
        admission_set["receipts"]["product"]["counts"]["discovery"] += 1
    elif forge == "governance":
        admission_set["receipts"]["product"]["governance"][
            "owner_ratification_required_for_gpu"
        ] = False
        admission_set = _rehash_admission_set(admission_set)
    else:
        forged = "f" * 64
        admission_set["generator_revision"][
            "admission_authority_source_sha256"
        ] = forged
        for receipt in admission_set["receipts"].values():
            receipt["admission_authority_source_sha256"] = forged
        admission_set = _rehash_admission_set(admission_set)
    profiles = {
        family: _bound_profiles(
            base_config,
            count=counts["discovery"],
        )
        for family, counts in H.EXPECTED_FIXTURE_COUNTS.items()
    }
    with pytest.raises(H.HKEContractError):
        H.build_prelaunch_plan(
            base_config,
            admission_set=admission_set,
            profiles_by_family=profiles,
            evaluator_identity=_evaluator(),
        )


def test_bound_evidence_produces_predeclared_go(base_config):
    plan = _plan(base_config)
    decision = H.analyze(plan, _evidence(plan))
    assert decision["factors"]["loss"]["decision"] == "GO"
    assert decision["factors"]["planner"]["decision"] == "GO"
    assert decision["plan_sha256"] == plan["plan_sha256"]
    assert decision["evaluator_sha256"] == plan["evaluator_sha256"]
    assert decision["social_reliability"][
        "operator_attested_terminal_and_attached"
    ] is True
    assert decision["status"].startswith("OPERATOR_ATTESTED_")


def test_direction_or_one_percent_regression_holds_factor(base_config):
    plan = _plan(base_config)
    decision = H.analyze(plan, _evidence(plan, logo_regression=True))
    assert decision["factors"]["loss"]["decision"] == "HOLD"
    assert decision["factors"]["planner"]["decision"] == "HOLD"


@pytest.mark.parametrize(
    "forge",
    (
        "plan",
        "config",
        "fixture",
        "evaluator",
        "seed",
        "artifact",
        "rows",
    ),
)
def test_forged_analysis_identity_aborts(base_config, forge):
    plan = _plan(base_config)
    evidence = _evidence(plan)
    if forge == "plan":
        evidence["plan_sha256"] = "f" * 64
    elif forge == "config":
        evidence["families"]["product"]["arms"]["A"]["training_receipt"][
            "config_sha256"
        ] = "f" * 64
    elif forge == "fixture":
        evidence["families"]["product"][
            "fixture_candidate_semantic_sha256"
        ] = "f" * 64
    elif forge == "evaluator":
        evidence["families"]["product"]["arms"]["A"]["score_receipt"][
            "evaluator_sha256"
        ] = "f" * 64
    elif forge == "seed":
        evidence["families"]["product"]["arms"]["A"]["training_receipt"][
            "training_seed"
        ] += 1
    elif forge == "artifact":
        evidence["families"]["product"]["arms"]["A"]["score_receipt"][
            "artifact_sha256"
        ] = "f" * 64
    else:
        evidence["families"]["product"]["arms"]["A"]["score_receipt"][
            "rows"
        ][0]["blank_loss"] += 0.1
    with pytest.raises(H.HKEContractError):
        H.analyze(plan, evidence)


def test_nonterminal_and_unloaded_artifacts_abort_without_free_boolean(base_config):
    plan = _plan(base_config)
    evidence = _evidence(plan)
    cell = evidence["families"]["product"]["arms"]["D"]
    cell["training_receipt"]["checkpoint_step"] -= 1
    _rehash_training(cell)
    _rehash_evidence(evidence)
    with pytest.raises(H.HKEContractError, match="planned terminal"):
        H.analyze(plan, evidence)

    evidence = _evidence(plan)
    cell = evidence["families"]["product"]["arms"]["D"]
    cell["attachment_receipt"][
        "unloaded_key_count"
    ] = 1
    _rehash_attachment(cell)
    _rehash_evidence(evidence)
    with pytest.raises(H.HKEContractError, match="attachment"):
        H.analyze(plan, evidence)


@pytest.mark.parametrize("receipt", ("training", "attachment", "score"))
def test_self_hashed_receipts_still_require_plan_and_artifact_bindings(
    base_config, receipt
):
    plan = _plan(base_config)
    evidence = _evidence(plan)
    cell = evidence["families"]["product"]["arms"]["A"]
    if receipt == "training":
        cell["training_receipt"]["config_sha256"] = "f" * 64
        _rehash_training(cell)
    elif receipt == "attachment":
        cell["attachment_receipt"]["artifact_sha256"] = "f" * 64
        _rehash_attachment(cell)
    else:
        cell["score_receipt"]["artifact_sha256"] = "f" * 64
        _rehash_score(cell)
    _rehash_evidence(evidence)
    with pytest.raises(H.HKEContractError, match="provenance|bound"):
        H.analyze(plan, evidence)


def test_analysis_requires_exact_discovery_row_set_and_rejects_free_flags(
    base_config,
):
    plan = _plan(base_config)
    evidence = _evidence(plan)
    rows = evidence["families"]["product"]["arms"]["A"]["score_receipt"][
        "rows"
    ]
    rows.pop()
    evidence["families"]["product"]["arms"]["A"]["score_receipt"][
        "rows_sha256"
    ] = H.canonical_sha256(rows)
    _rehash_score(evidence["families"]["product"]["arms"]["A"])
    _rehash_evidence(evidence)
    with pytest.raises(H.HKEContractError, match="exact 28 discovery rows"):
        H.analyze(plan, evidence)

    evidence = _evidence(plan)
    evidence["families"]["product"]["arms"]["A"]["terminal_valid"] = True
    _rehash_evidence(evidence)
    with pytest.raises(H.HKEContractError, match="shape mismatch"):
        H.analyze(plan, evidence)


def test_confirmation_phase_or_confirmation_count_is_rejected(base_config):
    plan = _plan(base_config)
    evidence = _evidence(plan)
    evidence["families"]["product"]["phase"] = "confirmation"
    _rehash_evidence(evidence)
    with pytest.raises(H.HKEContractError, match="fixture identity mismatch"):
        H.analyze(plan, evidence)

    evidence = _evidence(plan)
    confirmation_count = H.EXPECTED_FIXTURE_COUNTS["product"]["confirmation"]
    for cell in evidence["families"]["product"]["arms"].values():
        rows = cell["score_receipt"]["rows"][:confirmation_count]
        cell["score_receipt"]["rows"] = rows
        cell["score_receipt"]["rows_sha256"] = H.canonical_sha256(rows)
        _rehash_score(cell)
    _rehash_evidence(evidence)
    with pytest.raises(H.HKEContractError, match="exact 28 discovery rows"):
        H.analyze(plan, evidence)
