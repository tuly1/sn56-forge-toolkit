"""Adversarial CPU tests for the FutureBound-first Krea factorial."""

from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
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
REAL_HEAD_BLOB_SHA256 = H._head_blob_sha256


@pytest.fixture
def base_config():
    return yaml.safe_load(H.INCUMBENT_TEMPLATE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def committed_sources_match_test_checkout(monkeypatch):
    monkeypatch.setattr(
        H,
        "_head_blob_sha256",
        lambda relative: H.sha256_file(ROOT / relative),
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_head_blob_hash_ignores_hostile_replace_ref(tmp_path, monkeypatch):
    def git(*arguments, input_bytes=None):
        completed = subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=tmp_path,
            input=input_bytes,
            check=True,
            capture_output=True,
        )
        return completed.stdout

    git("init", "-q")
    git("config", "user.name", "SN56 Test")
    git("config", "user.email", "sn56@example.invalid")
    original = b"reviewed source bytes\n"
    hostile = b"hostile replacement bytes\n"
    (tmp_path / "payload.txt").write_bytes(original)
    git("add", "payload.txt")
    git("commit", "-q", "-m", "fixture")
    original_blob = git("rev-parse", "HEAD:payload.txt").decode().strip()
    hostile_blob = (
        git("hash-object", "-w", "--stdin", input_bytes=hostile).decode().strip()
    )
    git("replace", original_blob, hostile_blob)
    try:
        assert git("show", "HEAD:payload.txt") == hostile
        monkeypatch.setattr(H, "REPO_ROOT", tmp_path)
        assert (
            REAL_HEAD_BLOB_SHA256("payload.txt") == hashlib.sha256(original).hexdigest()
        )
    finally:
        git("replace", "-d", original_blob)
    assert git("for-each-ref", "--format=%(refname)", "refs/replace") == b""


def _observation(
    *,
    name: str = "NVIDIA H100 PCIe",
    uuid: str = "GPU-12345678-abcd-1234-abcd-123456789abc",
    memory: int = 81_559,
):
    raw = f"{name}, {uuid}, {memory} MiB\n"

    def fixed_capture(command, **kwargs):
        assert command == list(H.FIXED_NVIDIA_SMI_COMMAND)
        assert kwargs["cwd"] == "/"
        assert kwargs["env"] == {
            "PATH": "/usr/bin:/bin",
            "HOME": "/nonexistent-sn56-h100-capture",
            "LANG": "C",
            "LC_ALL": "C",
        }
        return subprocess.CompletedProcess(command, 0, stdout=raw, stderr="")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(H.subprocess, "run", fixed_capture)
        patch.setattr(H, "_utc_now", lambda: "2026-08-11T20:00:00Z")
        return H.capture_h100_observation()


def _accelerator_label(observation):
    device = observation["device"]
    return f"{device['name']}|{device['memory_total_mib']}-MiB|{device['uuid']}"


def _profile(loss: str, count: int, observation, rate: float = 0.9):
    completed = 100
    first = 20
    value = adaptive_timing.ThroughputProfile(
        bundle_id=krea_runtime.WEEK7_FACTORIAL_MULTIRES_BUNDLE,
        bundle_sha256=krea_runtime.bundle_contract_sha256(
            krea_runtime.WEEK7_FACTORIAL_MULTIRES_BUNDLE
        ),
        model_type="krea2",
        measured_dataset_size=count,
        dataset_regime=adaptive_timing.dataset_regime(count),
        seconds_per_step=rate,
        startup_seconds=0.0,
        completed_steps=completed,
        training_elapsed_seconds=completed * rate,
        first_checkpoint_step=first,
        first_checkpoint_elapsed_seconds=first * rate,
        source_run_id=f"week7-{loss}-probe:{'a' * 32}",
        source_record_sha256=_sha(f"record-{loss}-{count}-{rate}"),
        runtime_commit=krea_runtime.OWNED_RUNTIME_COMMIT,
        measured_at_utc="2026-08-11T20:00:00Z",
        accelerator_identity=_accelerator_label(observation),
        profile_sha256="0" * 64,
    )
    document = H._profile_document(value)
    document.pop("profile_sha256")
    return replace(value, profile_sha256=adaptive_timing.canonical_sha256(document))


def _bound_profiles(base_config, count: int, observation=None):
    observation = observation or _observation()
    current = H.materialize_current_law_configs(
        base_config, num_images=count, hours_to_complete=0.75
    )
    return {
        loss: H.bind_timing_profile(
            _profile(loss, count, observation, 0.8 if loss == "mae" else 0.9),
            loss=loss,
            measured_dataset_size=count,
            measured_config_sha256=H.canonical_sha256(
                current["C" if loss == "mae" else "D"]
            ),
            accelerator_observation=observation,
        )
        for loss in ("mae", "mse")
    }


def _inventory(family: str, pack: str, count: int):
    files = []
    for index in range(count):
        stem = f"{family}-{pack}-{index:02d}"
        files.extend(
            [
                {
                    "path": f"{stem}.png",
                    "bytes": 101 + index,
                    "sha256": _sha(f"image:{stem}"),
                },
                {
                    "path": f"{stem}.txt",
                    "bytes": 21 + index,
                    "sha256": _sha(f"caption:{stem}"),
                },
            ]
        )
    files.sort(key=lambda item: item["path"])
    body = {"files": files, "file_count": len(files)}
    return {**body, "semantic_sha256": H.fixture_semantic_sha256(body)}


def _row_identity(family: str, pack: str, count: int):
    return [
        {
            "row_id": f"{family}-{pack}-row-{index:02d}",
            "row_sha256": _sha(f"row:{family}:{pack}:{index}"),
        }
        for index in range(count)
    ]


def _revealed_rows(family: str, pack: str):
    rows = []
    spec = H.EXPECTED_PACKS[family][pack]
    for split_role, label, count in (
        ("training", "train", spec["train"]),
        ("evaluation", "eval", spec["eval"]),
    ):
        for index in range(count):
            stem = f"{family}-confirm-{label}-{pack}-{index:02d}"
            parameters = {
                "pack": pack,
                "split_role": split_role,
                "phase_domain": "confirmation",
                "ordinal": index,
            }
            group_identity = {
                "family": family,
                "pack": pack,
                "split_role": split_role,
                "ordinal": index,
            }
            body = {
                "row_id": f"{family}-{pack}-{split_role}-{index:02d}",
                "fixture_id": f"fixture-{family}",
                "family": family,
                "phase": "confirmation",
                "pack": pack,
                "split_role": split_role,
                "ordinal": index,
                "relative_image_path": f"private/{stem}.png",
                "relative_caption_path": f"private/{stem}.txt",
                "parameters": parameters,
                "parameters_sha256": H.renderer_semantic_sha256(parameters),
                "seed_commitment_sha256": _sha(f"seed:{stem}"),
                "image_sha256": _sha(f"image:{stem}"),
                "image_bytes": 101 + index,
                "decoded_pixels_sha256": _sha(f"pixels:{stem}"),
                "caption_sha256": _sha(f"caption:{stem}"),
                "caption_bytes": 21 + index,
                "normalized_caption_sha256": _sha(f"normalized:{stem}"),
                "width": 1024,
                "height": 1024,
                "format": "PNG",
                "mode": "RGB",
                "visible_glyph_transcript": f"VISIBLE {index}",
                "group_identity": group_identity,
                "group_identity_sha256": H.renderer_semantic_sha256(group_identity),
                "rights_declaration": {
                    "classification": "first-party-procedural-candidate"
                },
            }
            rows.append({**body, "row_record_sha256": H.renderer_semantic_sha256(body)})
    return rows


def _generator_revision():
    commit = subprocess.check_output(
        ["/usr/bin/git", "rev-parse", "HEAD^{commit}"], cwd=ROOT, text=True
    ).strip()
    tree = subprocess.check_output(
        ["/usr/bin/git", "rev-parse", f"{commit}^{{tree}}"], cwd=ROOT, text=True
    ).strip()
    branch = subprocess.check_output(
        ["/usr/bin/git", "symbolic-ref", "--quiet", "HEAD"], cwd=ROOT, text=True
    ).strip()

    def blob_sha(path):
        payload = subprocess.check_output(
            [
                "/usr/bin/git",
                "--no-replace-objects",
                "cat-file",
                "blob",
                f"{commit}:{path}",
            ],
            cwd=ROOT,
        )
        return hashlib.sha256(payload).hexdigest()

    renderer_path = "ops/experiments/week7/hke_procedural_renderer.py"
    admission_path = "ops/experiments/week7/hke_fixture_admission.py"
    contract_path = "ops/experiments/week7/hke_fixture_contract.json"
    return {
        "repository": "https://github.com/tuly1/sn56-forge-toolkit.git",
        "commit": commit,
        "tree": tree,
        "renderer_source_path": renderer_path,
        "renderer_source_sha256": blob_sha(renderer_path),
        "admission_authority_path": admission_path,
        "admission_authority_source_sha256": blob_sha(admission_path),
        "contract_path": contract_path,
        "contract_source_sha256": blob_sha(contract_path),
        "pinned_remote_refs": [branch],
    }


def test_claimed_generator_revision_resolves_literal_tree_and_blobs():
    revision = _generator_revision()
    paths = (
        revision["renderer_source_path"],
        revision["admission_authority_path"],
        revision["contract_path"],
    )
    literal = H._literal_revision_identity(revision["commit"], paths)
    assert literal["commit"] == revision["commit"]
    assert literal["tree"] == revision["tree"]
    assert literal["blob_sha256"][paths[0]] == revision["renderer_source_sha256"]
    with pytest.raises(H.HKEContractError, match="unavailable"):
        H._literal_revision_identity("f" * 40, paths)


def _admission(family: str):
    revision = _generator_revision()
    packs = {}
    for pack, spec in H.EXPECTED_PACKS[family].items():
        if spec["phase"] == "discovery":
            training = _inventory(f"{family}-train", pack, spec["train"])
            evaluation = _inventory(f"{family}-eval", pack, spec["eval"])
            packs[pack] = {
                "phase": "discovery",
                "row_count": spec["count"],
                "training_row_count": spec["train"],
                "evaluation_row_count": spec["eval"],
                "training_row_identity_sha256": H.fixture_semantic_sha256(
                    _row_identity(f"{family}-train", pack, spec["train"])
                ),
                "evaluation_row_identity_sha256": H.fixture_semantic_sha256(
                    _row_identity(f"{family}-eval", pack, spec["eval"])
                ),
                "training_inventory": training,
                "training_inventory_sha256": training["semantic_sha256"],
                "evaluation_inventory": evaluation,
                "evaluation_inventory_sha256": evaluation["semantic_sha256"],
            }
        else:
            revealed_rows = _revealed_rows(family, pack)
            packs[pack] = {
                "phase": "confirmation",
                "row_count": spec["count"],
                "training_row_count": spec["train"],
                "evaluation_row_count": spec["eval"],
                "semantic_commitment_sha256": H.renderer_semantic_sha256(revealed_rows),
            }
    body = {
        "schema": 3,
        "kind": "sn56-week7-hke-fixture-admission",
        "status": "PASS",
        "family": family,
        "counts": copy.deepcopy(H.EXPECTED_FIXTURE_COUNTS[family]),
        "packs": packs,
        "candidate_semantic_sha256": "1" * 64,
        "human_review_sha256": "2" * 64,
        "replay_evidence_sha256": "3" * 64,
        "dedup_evidence_sha256": "4" * 64,
        "ownership_record_sha256": "5" * 64,
        "generator_repository": revision["repository"],
        "generator_commit": revision["commit"],
        "generator_tree": revision["tree"],
        "generator_source_path": revision["renderer_source_path"],
        "generator_source_sha256": revision["renderer_source_sha256"],
        "admission_authority_path": revision["admission_authority_path"],
        "admission_authority_source_sha256": H.sha256_file(H.ADMISSION_AUTHORITY_PATH),
        "contract_path": "ops/experiments/week7/hke_fixture_contract.json",
        "contract_source_sha256": H.sha256_file(
            ROOT / "ops/experiments/week7/hke_fixture_contract.json"
        ),
        "discovery_key_commitment_sha256": "a" * 64,
        "confirmation_key_commitment_sha256": "b" * 64,
        "confirmation_commitment_sha256": _sha(f"confirm-all:{family}"),
        "discovery_all_row_identity_sha256": _sha(f"discover-all:{family}"),
        "generator_revision_sha256": H.fixture_semantic_sha256(revision),
        "governance": {
            "operator_attested_named_human_review": True,
            "agent_review_is_not_human_review": True,
            "admission_authorized": True,
            "gpu_execution_authorized": False,
            "owner_ratification_required_for_gpu": True,
        },
        "claim_limit": "fixture-only; field transfer remains unproven",
    }
    return {**body, "admission_sha256": H.fixture_semantic_sha256(body)}


def _admission_set():
    revision = _generator_revision()
    receipts = {family: _admission(family) for family in H.EXPECTED_PACKS}
    replay = {
        "status": "PASS",
        "verified_rows": sum(
            sum(counts.values()) for counts in H.EXPECTED_FIXTURE_COUNTS.values()
        ),
        "candidate_semantic_sha256": "1" * 64,
        "dedup_semantic_sha256": "4" * 64,
        "discovery_key_commitment_sha256": "a" * 64,
        "confirmation_key_commitment_sha256": "b" * 64,
        "contract_path": "ops/experiments/week7/hke_fixture_contract.json",
        "contract_source_sha256": H.sha256_file(
            ROOT / "ops/experiments/week7/hke_fixture_contract.json"
        ),
    }
    replay_sha = H.fixture_semantic_sha256(replay)
    for receipt in receipts.values():
        receipt["replay_evidence_sha256"] = replay_sha
        body = dict(receipt)
        body.pop("admission_sha256")
        receipt["admission_sha256"] = H.fixture_semantic_sha256(body)
    body = {
        "schema": 3,
        "kind": "sn56-week7-hke-fixture-admission-set",
        "status": "PASS",
        "candidate_semantic_sha256": "1" * 64,
        "human_review_sha256": "2" * 64,
        "replay_evidence": replay,
        "replay_evidence_sha256": replay_sha,
        "dedup_evidence_sha256": "4" * 64,
        "ownership_record_sha256": "5" * 64,
        "generator_revision": revision,
        "generator_revision_sha256": H.fixture_semantic_sha256(revision),
        "family_admission_sha256": {
            family: receipt["admission_sha256"] for family, receipt in receipts.items()
        },
        "receipts": receipts,
        "authorization": {
            "fixture_admission_authorized": True,
            "gpu_execution_authorized": False,
            "deployment_authorized": False,
        },
    }
    return {**body, "admission_set_sha256": H.fixture_semantic_sha256(body)}


def _evaluator():
    return {
        "harness_sha256": _sha("evaluator-harness"),
        "god_commit": "a" * 40,
        "comfy_commit": "b" * 40,
        "defaults_sha256": _sha("evaluator-defaults"),
    }


def _execution():
    common = {
        "code_tree": "c" * 40,
        "container_digest": "sha256:" + "e" * 64,
    }
    return {
        "incumbent": {
            **common,
            "runtime_tree": krea_runtime.PINNED_BASE_TREE,
        },
        "owned": {
            **common,
            "runtime_tree": krea_runtime.OWNED_RUNTIME_TREE,
        },
    }


def _ratification(admission_set):
    body = {
        "schema": 3,
        "kind": "sn56-week7-hke-owner-ratification",
        "status": "SEALED_OPERATOR_ATTESTED_OWNER_RATIFICATION",
        "admission_set_sha256": admission_set["admission_set_sha256"],
        "human_review_sha256": admission_set["human_review_sha256"],
        "reviewer_identity": "Named Fixture Reviewer",
        "generator_revision": copy.deepcopy(admission_set["generator_revision"]),
        "generator_revision_sha256": H.fixture_semantic_sha256(
            admission_set["generator_revision"]
        ),
        "owner_identity": "Atulya Shetty",
        "ratified_at_utc": "2026-08-11T21:00:00Z",
        "decision": "RATIFY_FOR_PLAN_CONSUMPTION",
        "governance": {
            "operator_attested_not_cryptographically_authenticated": True,
            "agent_cannot_complete_owner_fields": True,
            "gpu_execution_authorized": False,
            "deployment_authorized": False,
            "later_plan_must_consume_exact_ratification": True,
        },
    }
    return {**body, "owner_ratification_sha256": H.fixture_semantic_sha256(body)}


def _plan(base_config):
    admission = _admission_set()
    return H.build_prelaunch_plan(
        base_config,
        admission_set=admission,
        owner_ratification=_ratification(admission),
        profiles_by_family={"social": {"D1": _bound_profiles(base_config, 10)}},
        evaluator_identity=_evaluator(),
        execution_identities=_execution(),
    )


def _rehash(receipt, digest_field):
    body = dict(receipt)
    body.pop(digest_field, None)
    receipt[digest_field] = H.canonical_sha256(body)


def _comparison(composite=0.04, prompted=None, blank=None):
    prompted = composite if prompted is None else prompted
    blank = composite if blank is None else blank
    incumbent = {
        key: {"prompted_loss": 1.0, "blank_loss": 1.0} for key in ("a", "b", "c", "d")
    }
    candidate = {
        key: {"prompted_loss": 1.0 - prompted, "blank_loss": 1.0 - blank}
        for key in incumbent
    }
    return H.compare_heldout_rows(incumbent, candidate)


def _passing_comparisons():
    return {
        "D1": _comparison(),
        "D2": {"Seed-A": _comparison(), "Seed-B": _comparison()},
        "C1": _comparison(),
        "product": _comparison(composite=0.0, prompted=0.0, blank=0.0),
        "logo_ui": _comparison(composite=0.0, prompted=0.0, blank=0.0),
    }


def _curve_for_cell(plan, cell, fixture, label: str, value: float):
    identity = cell["cell_identity"]
    common = H._receipt_cell_binding(identity)
    source_body = {
        "schema": 1,
        "kind": "sn56-week7-hke-training-source",
        "run_id": f"run:{label}",
        "cell_identity": copy.deepcopy(identity),
        "training_inventory": copy.deepcopy(fixture["training_inventory"]),
        "argv_sha256": _sha(f"argv:{label}"),
    }
    source = {
        **source_body,
        "source_record_sha256": H.canonical_sha256(source_body),
    }
    terminal_artifact = _sha(f"artifact:{label}:{cell['planned_steps']}")
    training = {
        **common,
        "schema": H.SCHEMA,
        "kind": "sn56-week7-hke-training-receipt",
        "status": "OPERATOR_ATTESTED_PASS",
        "evidence_class": "content_bound_operator_attested_not_independent_proof",
        "plan_sha256": plan["plan_sha256"],
        "planned_steps": cell["planned_steps"],
        "completed_steps": cell["planned_steps"],
        "terminal_artifact_sha256": terminal_artifact,
        "terminal_artifact_bytes": 1024,
        "terminal_checkpoint_step": cell["planned_steps"],
        "source_record": source,
        "source_record_sha256": source["source_record_sha256"],
    }
    _rehash(training, "training_receipt_sha256")
    checkpoints = []
    for step in cell["required_checkpoint_steps"]:
        artifact_sha = _sha(f"artifact:{label}:{step}")
        checkpoint = {
            **common,
            "schema": H.SCHEMA,
            "kind": "sn56-week7-hke-checkpoint-receipt",
            "status": "OPERATOR_ATTESTED_PASS",
            "plan_sha256": plan["plan_sha256"],
            "checkpoint_step": step,
            "artifact_sha256": artifact_sha,
            "artifact_bytes": 1024,
            "loadable": True,
            "source_record_sha256": source["source_record_sha256"],
            "training_receipt_sha256": training["training_receipt_sha256"],
        }
        _rehash(checkpoint, "checkpoint_receipt_sha256")
        attachment = {
            **common,
            "schema": H.SCHEMA,
            "kind": "sn56-week7-hke-attachment-receipt",
            "status": "OPERATOR_ATTESTED_PASS",
            "evidence_class": "content_bound_operator_attested_not_independent_proof",
            "plan_sha256": plan["plan_sha256"],
            "checkpoint_receipt_sha256": checkpoint["checkpoint_receipt_sha256"],
            "checkpoint_step": step,
            "artifact_sha256": artifact_sha,
            "loaded_key_count": 512,
            "unloaded_key_count": 0,
            "log_sha256": _sha(f"attach:{label}:{step}"),
        }
        _rehash(attachment, "attachment_receipt_sha256")
        identity_rows = fixture.get("evaluation_row_identity") or _row_identity(
            f"{fixture['family']}-eval",
            fixture["pack"],
            fixture["evaluation_row_count"],
        )
        rows = [
            {**row, "prompted_loss": value, "blank_loss": value}
            for row in identity_rows
        ]
        score = {
            **common,
            "schema": H.SCHEMA,
            "kind": "sn56-week7-hke-exact-score-receipt",
            "status": "OPERATOR_ATTESTED_PASS",
            "evidence_class": "content_bound_operator_attested_not_independent_proof",
            "plan_sha256": plan["plan_sha256"],
            "checkpoint_receipt_sha256": checkpoint["checkpoint_receipt_sha256"],
            "attachment_receipt_sha256": attachment["attachment_receipt_sha256"],
            "checkpoint_step": step,
            "artifact_sha256": artifact_sha,
            "evaluator_sha256": plan["evaluator_sha256"],
            "rows": rows,
            "rows_sha256": H.canonical_sha256(rows),
        }
        _rehash(score, "score_receipt_sha256")
        entry_body = {
            "checkpoint_receipt": checkpoint,
            "attachment_receipt": attachment,
            "score_receipt": score,
        }
        checkpoints.append(
            {**entry_body, "entry_sha256": H.canonical_sha256(entry_body)}
        )
    body = {
        "schema": 3,
        "kind": "sn56-week7-hke-score-curve",
        "status": "OPERATOR_ATTESTED_PASS",
        "evidence_class": "content_bound_operator_attested_not_independent_proof",
        "plan_sha256": plan["plan_sha256"],
        "cell_sha256": cell["cell_sha256"],
        "training_receipt": training,
        "checkpoints": checkpoints,
    }
    return {**body, "curve_sha256": H.canonical_sha256(body)}


def _curve(plan, stage: str, name: str, value: float):
    return _curve_for_cell(
        plan,
        plan["cells"][stage][name],
        plan["fixture"],
        f"{stage}:{name}",
        value,
    )


def _bridge_evidence(plan, incumbent=1.0, owned=1.001):
    body = {
        "schema": 3,
        "kind": "sn56-week7-hke-runtime-bridge-evidence",
        "plan_sha256": plan["plan_sha256"],
        "curves": {
            "incumbent": _curve(plan, "bridge", "incumbent", incumbent),
            "owned": _curve(plan, "bridge", "owned", owned),
        },
    }
    return {**body, "evidence_sha256": H.canonical_sha256(body)}


def _zero_lora(plan, value=1.05):
    fixture = plan["fixture"]
    rows = [
        {**row, "prompted_loss": value, "blank_loss": value}
        for row in _row_identity(
            f"{fixture['family']}-eval",
            fixture["pack"],
            fixture["evaluation_row_count"],
        )
    ]
    body = {
        "schema": H.SCHEMA,
        "kind": "sn56-week7-hke-zero-lora-score",
        "status": "OPERATOR_ATTESTED_PASS",
        "plan_sha256": plan["plan_sha256"],
        "evaluator_sha256": plan["evaluator_sha256"],
        "evaluation_row_identity_sha256": fixture["evaluation_row_identity_sha256"],
        "rows": rows,
        "rows_sha256": H.canonical_sha256(rows),
    }
    return {**body, "zero_receipt_sha256": H.canonical_sha256(body)}


def _d1_evidence(plan, values=None):
    values = values or {"R0": 1.0, "A": 0.98, "B": 0.97, "C": 0.96, "D": 0.95}
    body = {
        "schema": H.SCHEMA,
        "kind": "sn56-week7-hke-d1-core-evidence",
        "status": "OPERATOR_ATTESTED_PASS",
        "plan_sha256": plan["plan_sha256"],
        "curves": {
            arm: _curve(plan, "d1_core", arm, values[arm])
            for arm in ("R0", "A", "B", "C", "D")
        },
        "zero_lora": _zero_lora(plan),
    }
    return {**body, "evidence_sha256": H.canonical_sha256(body)}


def _d2_evidence(plan, incumbent=1.0, candidate=0.94):
    curves = {}
    for seed_label in ("Seed-A", "Seed-B"):
        curves[f"incumbent-{seed_label}"] = _curve_for_cell(
            plan,
            plan["cells"][f"incumbent-{seed_label}"],
            plan["fixture"],
            f"d2:incumbent:{seed_label}",
            incumbent,
        )
        curves[f"candidate-{seed_label}"] = _curve_for_cell(
            plan,
            plan["cells"][f"candidate-{seed_label}"],
            plan["fixture"],
            f"d2:candidate:{seed_label}",
            candidate,
        )
    body = {
        "schema": H.SCHEMA,
        "kind": "sn56-week7-hke-d2-score-evidence",
        "status": "OPERATOR_ATTESTED_PASS",
        "plan_sha256": plan["plan_sha256"],
        "curves": curves,
    }
    return {**body, "evidence_sha256": H.canonical_sha256(body)}


def _confirmation_reveal(
    freeze, family: str, pack: str, *, admission_set, authority=None
):
    spec = H.EXPECTED_PACKS[family][pack]
    training = _inventory(f"{family}-confirm-train", pack, spec["train"])
    evaluation = _inventory(f"{family}-confirm-eval", pack, spec["eval"])
    authority = authority or freeze
    is_c2 = pack == "C2"
    receipt = admission_set["receipts"][family]
    rows = _revealed_rows(family, pack)
    training_rows = [row for row in rows if row["split_role"] == "training"]
    evaluation_rows = [row for row in rows if row["split_role"] == "evaluation"]
    body = {
        "schema": H.SCHEMA,
        "kind": "sn56-week7-hke-private-confirmation-reveal",
        "status": (
            "PRIVATE_C2_REVEALED_AFTER_BORDERLINE_C1_TRIGGER"
            if is_c2
            else "PRIVATE_C1_REVEALED_AFTER_D2_CONFIRMATION_FREEZE"
        ),
        "privacy": "custodian_private_not_public_admission",
        "family": family,
        "pack": pack,
        "candidate_semantic_sha256": admission_set["candidate_semantic_sha256"],
        "admission_set_sha256": admission_set["admission_set_sha256"],
        "family_admission_sha256": receipt["admission_sha256"],
        "confirmation_commitment_sha256": freeze["confirmation_commitments"][family][
            pack
        ],
        "confirmation_authority_kind": (
            "sn56-week7-hke-c2-reveal-authorization"
            if is_c2
            else "sn56-week7-hke-confirmation-candidate-freeze"
        ),
        "confirmation_authority_sha256": (
            authority["c2_authority_sha256"]
            if is_c2
            else freeze["confirmation_freeze_sha256"]
        ),
        "revealed_rows": rows,
        "training_row_identity_sha256": H.fixture_semantic_sha256(
            [
                {"row_id": row["row_id"], "row_sha256": row["row_record_sha256"]}
                for row in training_rows
            ]
        ),
        "evaluation_row_identity_sha256": H.fixture_semantic_sha256(
            [
                {"row_id": row["row_id"], "row_sha256": row["row_record_sha256"]}
                for row in evaluation_rows
            ]
        ),
        "training_inventory": training,
        "training_inventory_sha256": training["semantic_sha256"],
        "evaluation_inventory": evaluation,
        "evaluation_inventory_sha256": evaluation["semantic_sha256"],
        "authorization": {
            "confirmation_revealed": True,
            "gpu_execution_authorized": False,
            "deployment_authorized": False,
            "candidate_selection_locked": True,
        },
    }
    return {**body, "confirmation_reveal_sha256": H.canonical_sha256(body)}


def _confirmation_evidence(plan, values=None):
    values = values or {
        "social": {"incumbent": 1.0, "candidate": 0.95},
        "product": {"incumbent": 1.0, "candidate": 0.995},
        "logo_ui": {"incumbent": 1.0, "candidate": 0.995},
    }
    curves = {
        family: {
            role: _curve_for_cell(
                plan,
                plan["cells"][family][role],
                plan["fixtures"][family],
                f"confirmation:{family}:{role}",
                values[family][role],
            )
            for role in ("incumbent", "candidate")
        }
        for family in ("social", "product", "logo_ui")
    }
    body = {
        "schema": H.SCHEMA,
        "kind": "sn56-week7-hke-confirmation-evidence",
        "status": "OPERATOR_ATTESTED_PASS",
        "plan_sha256": plan["plan_sha256"],
        "curves": curves,
    }
    return {**body, "evidence_sha256": H.canonical_sha256(body)}


def _c2_evidence(plan, incumbent=1.0, candidate=0.96):
    body = {
        "schema": H.SCHEMA,
        "kind": "sn56-week7-hke-c2-evidence",
        "status": "OPERATOR_ATTESTED_PASS",
        "plan_sha256": plan["plan_sha256"],
        "curves": {
            role: _curve_for_cell(
                plan,
                plan["cells"][role],
                plan["fixture"],
                f"c2:{role}",
                incumbent if role == "incumbent" else candidate,
            )
            for role in ("incumbent", "candidate")
        },
    }
    return {**body, "evidence_sha256": H.canonical_sha256(body)}


def _staged_confirmation(base_config, *, c1_candidate=0.95):
    prelaunch = _plan(base_config)
    bridge = _bridge_evidence(prelaunch)
    d1 = _d1_evidence(prelaunch)
    e_plan = H.build_optional_e_plan(prelaunch, d1, bridge_evidence=bridge)
    e_curve = _curve_for_cell(
        e_plan,
        e_plan["cell"],
        e_plan["fixture"],
        "optional-e",
        0.94,
    )
    frozen = H.freeze_d1_candidate(
        prelaunch,
        d1,
        bridge_evidence=bridge,
        frozen_at_utc="2026-08-12T01:00:00Z",
        optional_e_plan=e_plan,
        optional_e_curve=e_curve,
    )
    d2_plan = H.build_d2_replication_plan(
        prelaunch,
        frozen,
        d1_evidence=d1,
        bridge_evidence=bridge,
        optional_e_plan=e_plan,
        optional_e_curve=e_curve,
    )
    d2_evidence = _d2_evidence(d2_plan)
    confirmation_freeze = H.freeze_confirmation_candidate(
        d2_plan,
        d2_evidence,
        frozen_at_utc="2026-08-12T02:00:00Z",
    )
    reveals = {
        family: _confirmation_reveal(
            confirmation_freeze,
            family,
            "C1",
            admission_set=prelaunch["admission_set"],
        )
        for family in ("social", "product", "logo_ui")
    }
    confirmation_plan = H.build_confirmation_plan(
        d2_plan, confirmation_freeze, c1_reveals=reveals
    )
    confirmation_evidence = _confirmation_evidence(
        confirmation_plan,
        values={
            "social": {"incumbent": 1.0, "candidate": c1_candidate},
            "product": {"incumbent": 1.0, "candidate": 0.995},
            "logo_ui": {"incumbent": 1.0, "candidate": 0.995},
        },
    )
    return {
        "prelaunch": prelaunch,
        "bridge": bridge,
        "d1": d1,
        "e_plan": e_plan,
        "e_curve": e_curve,
        "frozen": frozen,
        "d2_plan": d2_plan,
        "d2_evidence": d2_evidence,
        "confirmation_freeze": confirmation_freeze,
        "confirmation_plan": confirmation_plan,
        "confirmation_evidence": confirmation_evidence,
    }


def test_futurebound_factorial_is_depth_matched_and_isolates_two_factors(base_config):
    profiles = _bound_profiles(base_config, 10)
    arms = H.materialize_arms(
        base_config,
        num_images=10,
        hours_to_complete=0.75,
        profiles=profiles,
    )
    assert {H._train_node(config)["steps"] for config in arms.values()} == {1200}
    assert H.changed_pointers(arms["A"], arms["B"]) == {
        "/config/process/0/train/loss_type"
    }
    assert H.changed_pointers(arms["A"], arms["C"]) == {
        "/config/process/0/train/multires_noise_iterations",
        "/config/process/0/train/multires_noise_discount",
    }
    assert H._train_node(arms["C"])["multires_noise_iterations"] == 6
    assert H._train_node(arms["C"])["multires_noise_discount"] == pytest.approx(0.3)


def test_plan_binds_every_pack_cell_to_physical_inputs(base_config):
    plan = _plan(base_config)
    assert H._validate_plan(plan) == plan
    assert set(plan["cells"]) == {"bridge", "d1_core"}
    assert set(plan["cells"]["d1_core"]) == {"R0", "A", "B", "C", "D"}
    assert plan["cells"]["d1_core"]["R0"]["required_checkpoint_steps"][-1] == 1166
    for arm in H.ARMS:
        cell = plan["cells"]["d1_core"][arm]
        identity = H._validate_cell_identity(cell["cell_identity"], "test")
        assert (
            identity["evaluation_inventory_sha256"]
            == plan["fixture"]["evaluation_inventory_sha256"]
        )
        assert identity["runtime_commit"] == krea_runtime.OWNED_RUNTIME_COMMIT
        assert 1166 not in cell["required_checkpoint_steps"]
        assert cell["required_checkpoint_steps"][-1] == 1200
    assert plan["authorization"]["d1_factorial_launch_authorized"] is False
    assert plan["authorization"]["gpu_execution_authorized"] is False
    assert plan["authorization"]["bridge_launch_authorized"] is False
    assert plan["authorization"]["separate_owner_gpu_order_required"] is True
    assert "C1" not in json.dumps(plan["fixture"])


def test_all_prior_plan_and_admission_schemas_are_invalidated():
    with pytest.raises(H.HKEContractError, match="superseded"):
        H._validate_admission_set({"schema": 1})
    with pytest.raises(H.HKEContractError, match="superseded"):
        H._validate_plan({"schema": 1})


def test_complete_bound_evidence_produces_canonical_decision(base_config):
    decision = H._evaluate_futurebound_comparisons(_passing_comparisons())
    assert decision["decision"] == "GO"
    assert json.loads(H.canonical_bytes(decision)) == decision
    assert "authorization" not in decision


def test_complete_staged_chain_is_the_only_public_go_path(base_config):
    staged = _staged_confirmation(base_config)
    assert staged["d2_plan"]["authorization"]["gpu_execution_authorized"] is False
    assert staged["d2_plan"]["authorization"]["d2_execution_authorized"] is False
    assert (
        staged["confirmation_plan"]["authorization"]["gpu_execution_authorized"]
        is False
    )
    assert (
        staged["confirmation_plan"]["authorization"][
            "c1_and_guardrail_execution_authorized"
        ]
        is False
    )
    decision = H.evaluate_futurebound_gates(
        staged["confirmation_plan"], staged["confirmation_evidence"]
    )
    assert decision["decision"] == "GO"
    assert (
        decision["confirmation_freeze_sha256"]
        == staged["confirmation_freeze"]["confirmation_freeze_sha256"]
    )
    assert decision["c2"] is None
    assert decision["authorization"]["deployment_authorized"] is False
    assert json.loads(H.canonical_bytes(decision)) == decision


def test_rehashed_outer_confirmation_cannot_hide_tampered_discovery_freeze(
    base_config,
):
    staged = _staged_confirmation(base_config)
    forged = copy.deepcopy(staged["confirmation_plan"])
    forged["confirmation_freeze"]["source_d2_plan_sha256"] = "0" * 64
    forged["confirmation_freeze"]["d2_evidence_sha256"] = "0" * 64
    _rehash(forged["confirmation_freeze"], "confirmation_freeze_sha256")
    forged["confirmation_freeze_sha256"] = forged["confirmation_freeze"][
        "confirmation_freeze_sha256"
    ]
    _rehash(forged, "plan_sha256")
    forged_evidence = _confirmation_evidence(forged)
    with pytest.raises(H.HKEContractError, match="confirmation freeze"):
        H.evaluate_futurebound_gates(forged, forged_evidence)


def test_rehashed_confirmation_plan_rejects_executed_config_not_matching_source_chain(
    base_config,
):
    staged = _staged_confirmation(base_config)
    forged = copy.deepcopy(staged["confirmation_plan"])
    train = H._train_node(forged["cells"]["social"]["candidate"]["config"])
    train["loss_type"] = "mse" if train["loss_type"] == "mae" else "mae"
    _rehash(forged, "plan_sha256")
    forged_evidence = _confirmation_evidence(forged)

    with pytest.raises(H.HKEContractError, match="does not reproduce"):
        H.evaluate_futurebound_gates(forged, forged_evidence)


def test_rehashed_d2_plan_cannot_hide_tampered_d1_source_chain(base_config):
    staged = _staged_confirmation(base_config)
    forged = copy.deepcopy(staged["d2_plan"])
    forged["frozen_candidate"]["d1_evidence_sha256"] = "0" * 64
    _rehash(forged["frozen_candidate"], "frozen_candidate_sha256")
    forged["frozen_candidate_sha256"] = forged["frozen_candidate"][
        "frozen_candidate_sha256"
    ]
    _rehash(forged, "plan_sha256")
    with pytest.raises(H.HKEContractError, match="does not reproduce"):
        H._validate_d2_plan(forged)


def test_rehashed_confirmation_reveal_cannot_transplant_rows_or_admission(
    base_config,
):
    staged = _staged_confirmation(base_config)
    forged_reveals = copy.deepcopy(staged["confirmation_plan"]["reveals"])
    forged_reveals["social"]["candidate_semantic_sha256"] = "f" * 64
    forged_reveals["social"]["revealed_rows"][0]["row_id"] = "forged-row"
    _rehash(forged_reveals["social"], "confirmation_reveal_sha256")
    with pytest.raises(H.HKEContractError, match="confirmation reveal"):
        H.build_confirmation_plan(
            staged["d2_plan"],
            staged["confirmation_freeze"],
            c1_reveals=forged_reveals,
        )


def test_optional_e_is_required_when_predeclared_gate_is_eligible(base_config):
    plan = _plan(base_config)
    bridge = _bridge_evidence(plan)
    d1 = _d1_evidence(plan)
    e_plan = H.build_optional_e_plan(plan, d1, bridge_evidence=bridge)
    assert e_plan["clock_fill_steps"] > H.FACTORIAL_STEPS
    assert e_plan["authorization"]["gpu_execution_authorized"] is False
    assert e_plan["authorization"]["optional_e_execution_authorized"] is False
    with pytest.raises(H.HKEContractError, match="optional E evidence is required"):
        H.freeze_d1_candidate(
            plan,
            d1,
            bridge_evidence=bridge,
            frozen_at_utc="2026-08-12T01:00:00Z",
        )

    forged = copy.deepcopy(e_plan)
    forged["clock_fill_steps"] += 1
    _rehash(forged, "plan_sha256")
    with pytest.raises(H.HKEContractError, match="does not reproduce"):
        H._validate_optional_e_plan(forged, prelaunch=plan)


def test_rehashed_optional_e_rejects_injected_executed_config(base_config):
    plan = _plan(base_config)
    bridge = _bridge_evidence(plan)
    d1 = _d1_evidence(plan)
    e_plan = H.build_optional_e_plan(plan, d1, bridge_evidence=bridge)
    forged = copy.deepcopy(e_plan)
    train = H._train_node(forged["cell"]["config"])
    train["loss_type"] = "mse" if train["loss_type"] == "mae" else "mae"
    _rehash(forged, "plan_sha256")
    forged_curve = _curve_for_cell(
        forged,
        forged["cell"],
        forged["fixture"],
        "forged-optional-e",
        0.94,
    )

    with pytest.raises(H.HKEContractError, match="does not reproduce"):
        H.freeze_d1_candidate(
            plan,
            d1,
            bridge_evidence=bridge,
            frozen_at_utc="2026-08-12T01:00:00Z",
            optional_e_plan=forged,
            optional_e_curve=forged_curve,
        )


def test_terminal_factorial_effects_are_predeclared_and_bound_to_freeze(base_config):
    plan = _plan(base_config)
    bridge = _bridge_evidence(plan)
    d1 = _d1_evidence(plan)
    e_plan = H.build_optional_e_plan(plan, d1, bridge_evidence=bridge)
    e_curve = _curve_for_cell(e_plan, e_plan["cell"], e_plan["fixture"], "e", 0.94)
    frozen = H.freeze_d1_candidate(
        plan,
        d1,
        bridge_evidence=bridge,
        frozen_at_utc="2026-08-12T01:00:00Z",
        optional_e_plan=e_plan,
        optional_e_curve=e_curve,
    )
    effects = frozen["terminal_factorial_effects"]
    assert effects["checkpoint_step"] == 1200
    assert set(effects["cell_composite_losses"]) == {"A", "B", "C", "D"}
    assert effects["mse_minus_mae"] == pytest.approx(-0.01)
    assert effects["multires_on_minus_off"] == pytest.approx(-0.02)


def test_bridge_pass_is_mechanical_only_and_never_gpu_authority(base_config):
    plan = _plan(base_config)
    receipt = H.analyze_runtime_bridge(plan, _bridge_evidence(plan))
    assert receipt["status"] == "PASS"
    assert receipt["d1_factorial_mechanically_unblocked"] is True
    assert receipt["separate_owner_gpu_order_required"] is True
    assert receipt["gpu_execution_authorized"] is False
    assert receipt["d1_factorial_launch_authorized"] is False


def test_borderline_c1_requires_and_accepts_predeclared_c2(base_config):
    staged = _staged_confirmation(base_config, c1_candidate=0.965)
    authority = H.build_c2_reveal_authority(
        staged["confirmation_plan"], staged["confirmation_evidence"]
    )
    assert authority["c2_reveal_authorized"] is True
    with pytest.raises(H.HKEContractError, match="requires the predeclared C2"):
        H.evaluate_futurebound_gates(
            staged["confirmation_plan"], staged["confirmation_evidence"]
        )
    reveal = _confirmation_reveal(
        staged["confirmation_freeze"],
        "social",
        "C2",
        admission_set=staged["prelaunch"]["admission_set"],
        authority=authority,
    )
    c2_plan = H.build_c2_plan(
        staged["d2_plan"],
        staged["confirmation_plan"],
        staged["confirmation_evidence"],
        c2_authority=authority,
        c2_reveal=reveal,
    )
    assert c2_plan["authorization"]["gpu_execution_authorized"] is False
    assert c2_plan["authorization"]["c2_execution_authorized"] is False
    c2_evidence = _c2_evidence(c2_plan)
    decision = H.evaluate_futurebound_gates(
        staged["confirmation_plan"],
        staged["confirmation_evidence"],
        c2_plan_value=c2_plan,
        c2_evidence=c2_evidence,
    )
    assert decision["decision"] == "GO"
    assert decision["c2"]["plan_sha256"] == c2_plan["plan_sha256"]


@pytest.mark.parametrize(
    "bad_timestamp",
    [
        "2026-02-30T01:00:00Z",
        "2026-08-12T25:00:00Z",
        "2026-08-12T01:00:00+00:00",
    ],
)
def test_candidate_freeze_rejects_noncanonical_or_impossible_timestamp(
    base_config, bad_timestamp
):
    plan = _plan(base_config)
    bridge = _bridge_evidence(plan)
    d1 = _d1_evidence(plan)
    with pytest.raises(
        H.HKEContractError, match="candidate freeze timestamp is invalid"
    ):
        H.freeze_d1_candidate(
            plan,
            d1,
            bridge_evidence=bridge,
            frozen_at_utc=bad_timestamp,
        )


def test_product_receipts_cannot_be_transplanted_into_logo_guardrail(base_config):
    staged = _staged_confirmation(base_config)
    plan = staged["confirmation_plan"]
    evidence = copy.deepcopy(staged["confirmation_evidence"])
    transplanted = copy.deepcopy(evidence["curves"]["product"]["candidate"])
    transplanted["cell_sha256"] = plan["cells"]["logo_ui"]["candidate"]["cell_sha256"]
    _rehash(transplanted, "curve_sha256")
    evidence["curves"]["logo_ui"]["candidate"] = transplanted
    _rehash(evidence, "evidence_sha256")
    with pytest.raises(H.HKEContractError, match="receipt cell binding mismatch"):
        H.evaluate_futurebound_gates(plan, evidence)


def test_cross_family_receipt_transplant_fails_even_after_rehash(base_config):
    admission = _admission_set()
    admission["receipts"]["logo_ui"]["generator_tree"] = "f" * 40
    body = dict(admission["receipts"]["logo_ui"])
    body.pop("admission_sha256")
    admission["receipts"]["logo_ui"]["admission_sha256"] = H.fixture_semantic_sha256(
        body
    )
    admission["family_admission_sha256"]["logo_ui"] = admission["receipts"]["logo_ui"][
        "admission_sha256"
    ]
    body = dict(admission)
    body.pop("admission_set_sha256")
    admission["admission_set_sha256"] = H.fixture_semantic_sha256(body)
    with pytest.raises(H.HKEContractError, match="disagrees"):
        H._validate_admission_set(admission)


def test_free_source_hash_or_foreign_inventory_cannot_be_rehashed_into_evidence(
    base_config,
):
    inventory = _inventory("bad", "D1", 2)
    inventory["files"][1]["path"] = "different.txt"
    inventory["files"].sort(key=lambda item: item["path"])
    body = dict(inventory)
    body.pop("semantic_sha256")
    inventory["semantic_sha256"] = H.fixture_semantic_sha256(body)
    with pytest.raises(H.HKEContractError, match="exactly one image and caption"):
        H._validate_inventory_body(inventory, "bad", expected_pairs=2)


@pytest.mark.parametrize("bad", [-1, True, "0.5"])
def test_invalid_score_classes_fail_before_statistics(base_config, bad):
    with pytest.raises(H.HKEContractError):
        H._composite({"prompted_loss": 0.5, "blank_loss": bad})


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_scores_are_rejected_before_receipt_serialization(bad):
    with pytest.raises(H.HKEContractError, match="finite"):
        H._composite({"prompted_loss": 0.5, "blank_loss": bad})


def test_zero_factor_baseline_is_a_contract_error(base_config):
    with pytest.raises(H.HKEContractError, match="baseline"):
        H.compare_heldout_rows(
            {
                "a": {"prompted_loss": 0.0, "blank_loss": 0.0},
                "b": {"prompted_loss": 0.0, "blank_loss": 0.0},
            },
            {
                "a": {"prompted_loss": 0.0, "blank_loss": 0.0},
                "b": {"prompted_loss": 0.0, "blank_loss": 0.0},
            },
        )


@pytest.mark.parametrize(
    "name,memory",
    [
        ("NOT-A-GPU h100 emulator", 80_000),
        ("prefix NVIDIA H100 PCIe", 81_559),
        ("NVIDIA H100 PCIe suffix", 81_559),
        ("NVIDIA H100 PCIe|extra", 81_559),
        ("NVIDIA H100 PCIe", 77_999),
    ],
)
def test_fake_h100_names_and_memory_are_rejected(name, memory):
    with pytest.raises(H.HKEContractError):
        H.h100_observation(
            name=name,
            uuid="GPU-12345678-abcd-1234-abcd-123456789abc",
            memory_total_mib=memory,
        )


def test_fixed_capture_is_the_only_non_synthetic_h100_constructor():
    captured = _observation()
    assert captured["evidence_origin"] == H.CAPTURED_H100_EVIDENCE_ORIGIN
    synthetic = H.h100_observation(
        name="NVIDIA H100 PCIe",
        uuid="GPU-12345678-abcd-1234-abcd-123456789abc",
        memory_total_mib=81_559,
    )
    assert synthetic["evidence_origin"] == "synthetic"
    with pytest.raises(TypeError):
        H.h100_observation(
            name="NVIDIA H100 PCIe",
            uuid="GPU-12345678-abcd-1234-abcd-123456789abc",
            memory_total_mib=81_559,
            evidence_origin=H.CAPTURED_H100_EVIDENCE_ORIGIN,
        )


def test_h100_observation_rejects_impossible_calendar_timestamp():
    with pytest.raises(H.HKEContractError, match="timestamp"):
        H.h100_observation(
            name="NVIDIA H100 PCIe",
            uuid="GPU-12345678-abcd-1234-abcd-123456789abc",
            memory_total_mib=81_559,
            captured_at_utc="2026-19-39T29:59:59Z",
        )


def test_h100_raw_record_and_uuid_are_bound(base_config):
    observation = _observation()
    forged = copy.deepcopy(observation)
    forged["device"]["uuid"] = "GPU-ffffffff-abcd-1234-abcd-123456789abc"
    body = dict(forged)
    body.pop("observation_sha256")
    forged["observation_sha256"] = H.canonical_sha256(body)
    with pytest.raises(H.HKEContractError, match="raw output"):
        H._validate_h100_observation(forged, "forged")
    with pytest.raises(H.HKEContractError, match="label"):
        profiles = _bound_profiles(base_config, 10, observation)
        H.bind_timing_profile(
            replace(profiles["mae"].profile, accelerator_identity="forged"),
            loss="mae",
            measured_dataset_size=10,
            measured_config_sha256=profiles["mae"].measured_config_sha256,
            accelerator_observation=observation,
        )


def test_timing_binding_uses_the_profile_runtime_not_incumbent_constants(base_config):
    binding = _bound_profiles(base_config, 10)["mae"]
    document = H._bound_profile_document(binding)["binding"]
    assert document["bundle_id"] == krea_runtime.WEEK7_FACTORIAL_MULTIRES_BUNDLE
    assert document["bundle_sha256"] == binding.profile.bundle_sha256
    assert document["runtime_commit"] == krea_runtime.OWNED_RUNTIME_COMMIT

    mismatched = replace(
        binding.profile,
        runtime_commit=krea_runtime.PINNED_BASE_COMMIT,
    )
    with pytest.raises(H.HKEContractError, match="bundle/runtime provenance"):
        H.bind_timing_profile(
            mismatched,
            loss="mae",
            measured_dataset_size=10,
            measured_config_sha256=binding.measured_config_sha256,
            accelerator_observation=binding.accelerator_observation,
        )


def test_training_directory_inventory_rejects_extra_symlink_and_directory(tmp_path):
    (tmp_path / "a.png").write_bytes(b"image")
    (tmp_path / "a.txt").write_text("caption", encoding="utf-8")
    valid = H.inventory_training_directory(tmp_path)
    assert valid["file_count"] == 2

    (tmp_path / "extra.bin").write_bytes(b"extra")
    with pytest.raises(H.HKEContractError, match="unexpected"):
        H.inventory_training_directory(tmp_path)
    (tmp_path / "extra.bin").unlink()

    (tmp_path / "nested").mkdir()
    with pytest.raises(H.HKEContractError, match="indirect"):
        H.inventory_training_directory(tmp_path)
    (tmp_path / "nested").rmdir()

    os.symlink(tmp_path / "a.png", tmp_path / "alias.png")
    with pytest.raises(H.HKEContractError, match="symlink"):
        H.inventory_training_directory(tmp_path)


def test_training_inventory_rejects_same_name_inode_swap_during_hash(
    tmp_path, monkeypatch
):
    image = tmp_path / "a.png"
    image.write_bytes(b"original-image")
    (tmp_path / "a.txt").write_text("caption", encoding="utf-8")
    real_read = H.os.read
    swapped = False

    def swap_after_read(descriptor, count):
        nonlocal swapped
        block = real_read(descriptor, count)
        if block and not swapped:
            replacement = tmp_path / "replacement.tmp"
            replacement.write_bytes(b"replacement-img")
            os.replace(replacement, image)
            swapped = True
        return block

    monkeypatch.setattr(H.os, "read", swap_after_read)
    with pytest.raises(H.HKEContractError, match="changed during inventory"):
        H.inventory_training_directory(tmp_path)


def test_terminal_checkpoint_must_be_from_same_cell(base_config):
    plan = _plan(base_config)
    curve = _curve(plan, "d1_core", "D", 0.9)
    curve["cell_sha256"] = plan["cells"]["d1_core"]["C"]["cell_sha256"]
    body = dict(curve)
    body.pop("curve_sha256")
    curve["curve_sha256"] = H.canonical_sha256(body)
    with pytest.raises(H.HKEContractError, match="authority binding"):
        H.validate_score_curve(
            curve,
            plan_sha256=plan["plan_sha256"],
            plan_cell=plan["cells"]["d1_core"]["D"],
            fixture=plan["fixture"],
            evaluator_sha256=plan["evaluator_sha256"],
        )


def test_foreign_physical_receipt_chain_cannot_be_relabelled_to_target_cell(
    base_config,
):
    plan = _plan(base_config)
    transplanted = _curve(plan, "d1_core", "C", 0.9)
    transplanted["cell_sha256"] = plan["cells"]["d1_core"]["D"]["cell_sha256"]
    _rehash(transplanted, "curve_sha256")
    with pytest.raises(H.HKEContractError, match="receipt cell binding mismatch"):
        H.validate_score_curve(
            transplanted,
            plan_sha256=plan["plan_sha256"],
            plan_cell=plan["cells"]["d1_core"]["D"],
            fixture=plan["fixture"],
            evaluator_sha256=plan["evaluator_sha256"],
        )


def test_every_predeclared_checkpoint_must_have_an_exact_score(base_config):
    plan = _plan(base_config)
    curve = _curve(plan, "d1_core", "C", 0.9)
    curve["checkpoints"].pop(0)
    body = dict(curve)
    body.pop("curve_sha256")
    curve["curve_sha256"] = H.canonical_sha256(body)
    with pytest.raises(H.HKEContractError, match="natural checkpoint schedule"):
        H.validate_score_curve(
            curve,
            plan_sha256=plan["plan_sha256"],
            plan_cell=plan["cells"]["d1_core"]["C"],
            fixture=plan["fixture"],
            evaluator_sha256=plan["evaluator_sha256"],
        )


def test_missing_or_failed_runtime_bridge_blocks_d1_freeze(base_config):
    plan = _plan(base_config)
    with pytest.raises(H.HKEContractError, match="runtime bridge evidence"):
        H.freeze_d1_candidate(
            plan,
            {},
            bridge_evidence={},
            frozen_at_utc="2026-08-11T22:00:00Z",
        )
    failed = _bridge_evidence(plan, incumbent=1.0, owned=1.02)
    assert H.analyze_runtime_bridge(plan, failed)["status"] == "HOLD"


def test_sub_three_percent_cannot_go():
    comparisons = _passing_comparisons()
    comparisons["D1"] = _comparison(composite=0.029)
    assert H._evaluate_futurebound_comparisons(comparisons)["decision"] == "HOLD"


def test_two_percent_prompted_regression_cannot_be_hidden_by_blank_gain():
    comparisons = _passing_comparisons()
    comparisons["D1"] = _comparison(composite=0.04, prompted=-0.02, blank=0.06)
    decision = H._evaluate_futurebound_comparisons(comparisons)
    assert decision["decision"] == "HOLD"
    assert decision["gates"]["prompted_and_blank_each_regress_less_than_1pct"] is False


def test_seed_b_missing_or_disagreeing_blocks_go():
    missing = _passing_comparisons()
    missing["D2"].pop("Seed-B")
    with pytest.raises(H.HKEContractError, match="both D2 finalist seeds"):
        H._evaluate_futurebound_comparisons(missing)
    disagreement = _passing_comparisons()
    disagreement["D2"]["Seed-B"] = _comparison(composite=-0.01)
    decision = H._evaluate_futurebound_comparisons(disagreement)
    assert decision["decision"] == "HOLD"
    assert decision["gates"]["both_finalist_seeds_agree"] is False


def test_prelaunch_contains_only_c1_commitment_not_confirmation_identity(base_config):
    plan = _plan(base_config)
    serialized = json.dumps(plan, sort_keys=True)
    assert plan["sealed_confirmation_commitments"]["social"][
        "C1"
    ] == H.renderer_semantic_sha256(_revealed_rows("social", "C1"))
    assert "confirmation_row_identity" not in serialized


def test_synthetic_h100_observation_cannot_make_launchable_plan(base_config):
    admission = _admission_set()
    profiles = _bound_profiles(
        base_config,
        10,
        H.h100_observation(
            name="NVIDIA H100 PCIe",
            uuid="GPU-12345678-abcd-1234-abcd-123456789abc",
            memory_total_mib=81_559,
        ),
    )
    with pytest.raises(H.HKEContractError, match="fixed command capture"):
        H.build_prelaunch_plan(
            base_config,
            admission_set=admission,
            owner_ratification=_ratification(admission),
            profiles_by_family={"social": {"D1": profiles}},
            evaluator_identity=_evaluator(),
            execution_identities=_execution(),
        )


def test_ten_second_profile_fails_fixed_depth_feasibility(base_config):
    admission = _admission_set()
    observation = _observation()
    profiles = _bound_profiles(base_config, 10, observation)
    profiles = {
        loss: H.bind_timing_profile(
            _profile(loss, 10, observation, rate=10.0),
            loss=loss,
            measured_dataset_size=10,
            measured_config_sha256=profiles[loss].measured_config_sha256,
            accelerator_observation=observation,
        )
        for loss in ("mae", "mse")
    }
    with pytest.raises(H.HKEContractError, match="cannot fit"):
        H.build_prelaunch_plan(
            base_config,
            admission_set=admission,
            owner_ratification=_ratification(admission),
            profiles_by_family={"social": {"D1": profiles}},
            evaluator_identity=_evaluator(),
            execution_identities=_execution(),
        )
