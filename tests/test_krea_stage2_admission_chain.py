"""Focused contracts for the atomic Stage-2 admission-chain adapter."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from ops.calibration import krea_confirmation_admission as admission
from ops.calibration import krea_density_seedb_freeze as density_freeze
from ops.calibration import krea_stage2_admission_chain as chain
from ops.calibration import krea_stage2_production_identity as production
from ops.calibration import krea_stage2_legacy_confirmation as legacy


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _time(base: datetime, seconds: int) -> str:
    return (base + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, value: dict) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = chain.canonical_bytes(value) + b"\n"
    path.write_bytes(payload)
    semantic = value[next(key for key in value if key.endswith("_sha256"))]
    return hashlib.sha256(payload).hexdigest(), semantic


def _identity(captured_at: str) -> dict:
    return production.build(
        forge={
            "commit_sha1": "a" * 40,
            "tree_sha1": "b" * 40,
            "worktree_state": "clean-including-untracked",
        },
        container_image={
            "image_id": "sha256:" + "c" * 64,
            "repo_digest": "registry.example/forge@sha256:" + "d" * 64,
        },
        dockerfile={
            "path": production.DOCKERFILE_PATH,
            "sha256": _sha("dockerfile"),
            "bytes": 10,
        },
        runtime_inputs=[
            {"path": path, "sha256": _sha(path), "bytes": index + 1}
            for index, path in enumerate(production.RUNTIME_INPUT_PATHS)
        ],
        base_model={
            "model_id": production.KREA_MODEL_ID,
            "revision": "e" * 40,
            "training_identity_sha256": _sha("training"),
            "asset_attestation_sha256": _sha("assets"),
            "text_encoder_id": production.KREA_TEXT_ENCODER_ID,
            "text_encoder_revision": "f" * 40,
        },
        runtime_contract={
            "runtime_identity_sha256": _sha("runtime"),
            "venv_tree_manifest_sha256": _sha("venv"),
            "trainer_identity_sha256": _sha("trainer"),
            "measurement_tool_sha256": _sha("measure"),
            "jit_enabled": True,
        },
        captured_at_utc=captured_at,
    )


def _prior_owner(ratified_at: str) -> dict:
    body = {
        "schema": 1,
        "kind": chain.PRIOR_OWNER_KIND,
        "owner_identity": chain.OWNER_IDENTITY,
        "owner_identity_assurance": (
            "interactive-owner-self-attestation-not-cryptographic-or-legal-signature"
        ),
        "ratified_at_utc": ratified_at,
        "portable_ratification_draft": {
            "file_sha256": _sha("draft-file"),
            "draft_sha256": _sha("draft"),
        },
        "governance_amendment": {
            "file_sha256": _sha("amendment-file"),
            "amendment_sha256": _sha("amendment"),
        },
        "decision_bindings": {"fixture": _sha("fixture")},
        "acknowledgements": {
            (
                "owner_authorizes_mechanical_gpu_approval_after_envelope_and_"
                "host_plan_validation"
            ): True,
            "stage2_requires_separate_commit_and_fresh_owner_ratification": True,
            "owner_accepts_accountability_for_using_bound_agent_evidence": True,
            "ratification_is_not_a_cryptographic_or_legal_signature": True,
        },
        "decision": "ratified_for_fixture_admission_input",
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": "test",
    }
    return {**body, "ratification_sha256": chain.canonical_sha256(body)}


def _freeze() -> dict:
    all_rules = {family: {"rule": family} for family in ("K0", "K1", "K2", "K3", "K4", "K5")}
    body = {
        "schema": density_freeze.SCHEMA,
        "kind": density_freeze.FREEZE_KIND,
        "outcome": "finalists_frozen",
        "blockers": [],
        "claims": density_freeze.FALSE_CLAIMS,
        "authority": density_freeze.AUTHORITY,
        "finalist_family_ids": ["K0", "K1", "K5"],
        "all_family_checkpoint_rules": all_rules,
        "checkpoint_rules": {family: all_rules[family] for family in ("K0", "K1", "K5")},
    }
    return {**body, "freeze_sha256": chain.canonical_sha256(body)}


def _deviation(recorded_at: str) -> dict:
    body = {
        "schema": 1,
        "kind": chain.DEVIATION_KIND,
        "recorded_at_utc": recorded_at,
        "occurrence_window_utc": {
            "after_exclusive": "2026-08-01T18:19:01Z",
            "before_exclusive": "2026-08-01T18:34:30Z",
            "precision": "bounded-window-exact-command-time-unavailable",
        },
        "actor": {"actor_class": "agent", "actor_id": "root/metagraph_recovery"},
        "frozen_selection_binding": {
            "public_commit_sha1": "f8d71ac1d0fcbab9dccf7f5a5a5f904f9f90b237",
            "public_binding_file_sha256": _sha("public-binding-file"),
            "public_binding_sha256": _sha("public-binding"),
        },
        "operation": {
            "host": "bittensor-ops",
            "sealed_root": "/opt/sn56-reviewer-sealed",
            "command_classes": [
                "find-maxdepth-metadata-listing",
                "find-filename-pattern-listing",
            ],
            "root_metadata_enumerated": True,
        },
        "observations": {
            "paths_sizes_modes_owners_mtimes_observed": True,
            "file_body_bytes_read": False,
            "caption_text_read": False,
            "image_pixels_read": False,
            "files_copied": False,
            "sealed_root_mutated": False,
        },
        "impact": {
            "strict_pre_materialization_barrier_deviation": True,
            "occurred_after_finalist_freeze": True,
            "finalist_selection_contaminated": False,
            "freeze_rerun_required": False,
        },
        "corrective_actions": [
            "access-stopped-and-disclosed-immediately",
            "no-further-sealed-root-access-before-authorized-materialization",
            "bind-this-record-into-stage2-admission-chain-receipt",
        ],
        "claim_limit": "test disclosure",
    }
    return {**body, "deviation_sha256": chain.canonical_sha256(body)}


def _inventory(
    confirmation_source: Path,
    boundary_source: Path,
    sealed: Path,
    controls: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    roles = {}
    commitments = {}
    boundary_rows = []
    for role in chain._ROLES:
        manifest_body = {"fixture_role": role, "task_id": role.lower()}
        manifest = {
            "schema": 99,
            "experimental_role": role,
            **manifest_body,
            "manifest_sha256": chain.canonical_sha256(manifest_body),
        }
        source = (
            confirmation_source if role in chain._CONFIRMATION_ROLES else boundary_source
        )
        role_root = source / role
        role_root.mkdir(parents=True, exist_ok=True)
        if role in chain._CONFIRMATION_ROLES:
            from PIL import Image

            shape = legacy.amendment.SHAPE_CONTRACT[role]
            listed = []
            for holdout, count in (
                (False, shape["training_pairs"]),
                (True, shape["evaluation_rows"]),
            ):
                prefix = "holdout/" if holdout else ""
                for index in range(1, count + 1):
                    image_rel = f"{prefix}image-{index:03d}.jpg"
                    text_rel = f"{prefix}image-{index:03d}.txt"
                    image_path = role_root / image_rel
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (8 + index, 9 + index), (index % 255, 1, 2)).save(
                        image_path, format="JPEG"
                    )
                    (role_root / text_rel).write_text(
                        f"natural caption {role} {index}\n", encoding="utf-8"
                    )
                    for relative in (image_rel, text_rel):
                        raw = (role_root / relative).read_bytes()
                        listed.append((hashlib.sha256(raw).hexdigest(), relative))
            manifest_path = role_root / f"MANIFEST-{role}.sha256"
            manifest_path.write_text(
                "".join(f"{digest}  {relative}\n" for digest, relative in listed),
                encoding="ascii",
            )
            archive_path = role_root / f"{role.lower()}_tourn.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
                for _, relative in listed:
                    if "/" not in relative:
                        archive.write(role_root / relative, relative)
            (role_root / "LICENSES.txt").write_text(
                f"test licence {role}\n", encoding="utf-8"
            )
            manifest_raw = manifest_path.read_bytes()
            patched = dict(legacy.amendment.MANIFEST_FILE_SHA256S)
            patched[role] = hashlib.sha256(manifest_raw).hexdigest()
            monkeypatch.setattr(legacy.amendment, "MANIFEST_FILE_SHA256S", patched)
            manifest_semantic = legacy.PRIOR_SEMANTIC_SHA256S[role]
        else:
            manifest_path = role_root / "fixture-manifest.json"
            archive_path = role_root / "training.zip"
            manifest_raw = chain.canonical_bytes(manifest) + b"\n"
            archive_raw = f"archive-{role}\n".encode()
            manifest_path.write_bytes(manifest_raw)
            archive_path.write_bytes(archive_raw)
            manifest_semantic = manifest["manifest_sha256"]
        roles[role] = {
            "root_prefix": role,
            "manifest_relative_path": manifest_path.relative_to(source).as_posix(),
            "manifest_sha256": manifest_semantic,
            "archive_relative_path": archive_path.relative_to(source).as_posix(),
        }
        if role in chain._CONFIRMATION_ROLES:
            commitments[role] = hashlib.sha256(manifest_raw).hexdigest()
        else:
            boundary_rows.append(
                {
                    "role": role,
                    "source_role": chain.boundary_derivation._ROLE_SOURCE[role],
                    "manifest_file_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                    "manifest_sha256": manifest["manifest_sha256"],
                    "training_archive_sha256": hashlib.sha256(archive_raw).hexdigest(),
                    "training_dataset_sha256": _sha(f"{role}-train"),
                    "evaluation_dataset_sha256": _sha(f"{role}-eval"),
                }
            )
    (confirmation_source / "COMMITMENT.txt").write_text(
        "excluded old-seal support\n", encoding="utf-8"
    )
    body = {
        "schema": 1,
        "kind": chain.LAYOUT_KIND,
        "sealed_root_locator_sha256": admission.sealed_root_locator_sha256(sealed),
        "roles": roles,
        "supporting_file_roles": {},
    }
    layout = {**body, "layout_sha256": chain.canonical_sha256(body)}
    layout_path = controls / "layout.json"
    _write(layout_path, layout)
    derivation_body = {
        "schema": chain.boundary_derivation.SCHEMA,
        "kind": chain.BOUNDARY_DERIVATION_KIND,
        "created_at_utc": "2026-08-01T18:35:00Z",
        "public_freeze_binding": {
            "path": chain.boundary_derivation.FREEZE_BINDING_PATH,
            "file_sha256": chain.boundary_derivation.FREEZE_BINDING_FILE_SHA256,
            "binding_sha256": chain.boundary_derivation.FREEZE_BINDING_SHA256,
            "commit_sha1": chain.boundary_derivation.FREEZE_BINDING_COMMIT,
        },
        "roles": sorted(boundary_rows, key=lambda row: row["role"]),
        "mechanics_only": True,
        "source_bytes_changed": False,
        "source_governance_reused_as_boundary_authority": False,
        "fresh_stage2_owner_ratification_required": True,
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": "mechanics-only test derivation",
    }
    derivation = {
        **derivation_body,
        "derivation_set_sha256": chain.canonical_sha256(derivation_body),
    }
    derivation_path = controls / "boundary-derivation.json"
    _write(derivation_path, derivation)
    real_validate = admission.krea_fixture.validate_manifest
    monkeypatch.setattr(
        admission.krea_fixture,
        "validate_manifest",
        lambda value: value if value.get("schema") == 99 else real_validate(value),
    )
    return chain.capture_inventory(
        confirmation_source_root=confirmation_source,
        boundary_source_root=boundary_source,
        stage2_root=sealed,
        layout_path=layout_path,
        boundary_derivation_path=derivation_path,
        public_commitment_sha256s=commitments,
        output=controls / "inventory.json",
        captured_at_utc="2026-08-01T18:40:00Z",
        remote_verifier=lambda: None,
    )


def _binding(path: Path, record: dict, semantic_key: str) -> dict:
    return {
        "path": str(path),
        "file_sha256": chain.canonical_file_sha256(record),
        semantic_key: record[semantic_key],
    }


def _case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    base = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=300)
    sealed = tmp_path / "sealed"
    confirmation_source = tmp_path / "old-seal"
    boundary_source = tmp_path / "boundary-source"
    identity = _identity(_time(base, 10))
    freeze = _freeze()
    prior = _prior_owner(_time(base, 0))
    inventory = _inventory(
        confirmation_source,
        boundary_source,
        sealed,
        tmp_path / "capture",
        monkeypatch,
    )
    deviation = _deviation(_time(base, 20))
    records = {
        "production_identity": (identity, "production_identity_sha256"),
        "waiver_finalist_freeze": (freeze, "freeze_sha256"),
        "prior_owner_ratification": (prior, "ratification_sha256"),
        "sealed_inventory": (inventory, "inventory_sha256"),
        "sealed_metadata_deviation": (deviation, "deviation_sha256"),
    }
    bindings = {}
    for name, (record, semantic) in records.items():
        path = tmp_path / "inputs" / f"{name}.json"
        _write(path, record)
        bindings[name] = _binding(path, record, semantic)
    body = {
        "schema": 1,
        "kind": chain.SPEC_KIND,
        **bindings,
        "sealed_root": str(sealed),
        "timestamps": {
            "request_prepared_at_utc": _time(base, 30),
            "owner_ratified_at_utc": _time(base, 40),
            "reveal_authorized_at_utc": _time(base, 50),
            "materialized_at_utc": _time(base, 60),
            "gpu_authorized_at_utc": _time(base, 70),
        },
    }
    spec = {**body, "spec_sha256": chain.canonical_sha256(body)}
    spec_path = tmp_path / "spec.json"
    _write(spec_path, spec)
    return spec_path, sealed


def test_admit_publishes_atomic_create_only_exactly_replayable_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, sealed = _case(tmp_path, monkeypatch)
    output = tmp_path / "authority"
    original = admission._read_sealed_file
    reads: list[str] = []

    def tracked(root: Path, relative: str) -> bytes:
        reads.append(relative)
        return original(root, relative)

    monkeypatch.setattr(admission, "_read_sealed_file", tracked)
    receipt = chain.admit(spec_path=spec, output_dir=output)

    assert receipt["authority"]["gpu_execution_authorized"] is True
    assert receipt["authority"]["release_authorized"] is False
    assert not (sealed / "COMMITMENT.txt").exists()
    assert receipt["post_freeze_custodian_inventory_capture"] == {
        "actor": chain.delegated.actor("confirmation_materialization_reviewer"),
        "sealed_fixture_bytes_read_and_copied": True,
        "fixture_content_emitted": False,
        "selection_was_frozen_before_capture": True,
        "confirmation_source_transfer": chain._CONFIRMATION_TRANSFER_BINDING,
    }
    inventory = json.loads(
        Path(json.loads(spec.read_bytes())["sealed_inventory"]["path"]).read_bytes()
    )
    assert len(reads) == len(inventory["files"])
    assert chain.replay(output) == receipt
    inventory = json.loads((output / "input-sealed-inventory.json").read_bytes())
    for relative in (
        "request.json",
        "ratification.json",
        "reveal.json",
        "materialized/materialization.json",
        "gpu-execution-authorization.json",
    ):
        record = json.loads((output / relative).read_bytes())
        assert record["sealed_inventory_sha256"] == inventory["inventory_sha256"]
        assert record["sealed_inventory_file_sha256"] == chain.canonical_file_sha256(
            inventory
        )
    with pytest.raises(FileExistsError):
        chain.admit(spec_path=spec, output_dir=output)


def test_invalid_prior_owner_binding_fails_before_any_sealed_interaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path, _ = _case(tmp_path, monkeypatch)
    spec = json.loads(spec_path.read_bytes())
    prior_path = Path(spec["prior_owner_ratification"]["path"])
    prior = json.loads(prior_path.read_bytes())
    prior["acknowledgements"][
        "owner_authorizes_mechanical_gpu_approval_after_envelope_and_host_plan_validation"
    ] = False
    prior_body = {key: value for key, value in prior.items() if key != "ratification_sha256"}
    prior["ratification_sha256"] = chain.canonical_sha256(prior_body)
    _write(prior_path, prior)
    spec["prior_owner_ratification"] = _binding(
        prior_path, prior, "ratification_sha256"
    )
    body = {key: value for key, value in spec.items() if key != "spec_sha256"}
    spec["spec_sha256"] = chain.canonical_sha256(body)
    _write(spec_path, spec)
    monkeypatch.setattr(
        admission,
        "_resolve_sealed_root",
        lambda _path: pytest.fail("invalid public evidence touched sealed root"),
    )
    with pytest.raises(ValueError, match="required Stage-2 acknowledgements"):
        chain.admit(spec_path=spec_path, output_dir=tmp_path / "authority")


def test_inventory_requires_every_boundary_role_without_aliasing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path, _ = _case(tmp_path, monkeypatch)
    spec = json.loads(spec_path.read_bytes())
    inventory = json.loads(Path(spec["sealed_inventory"]["path"]).read_bytes())
    inventory["files"] = [
        row for row in inventory["files"] if row["role"] != "B-1-large"
    ]
    body = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    inventory["inventory_sha256"] = chain.canonical_sha256(body)
    with pytest.raises(ValueError, match="cover every"):
        chain.validate_inventory(inventory)


def test_capture_rejects_pre_freeze_time_without_touching_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        admission,
        "_resolve_sealed_root",
        lambda _path: pytest.fail("pre-freeze capture touched sealed root"),
    )
    with pytest.raises(ValueError, match="predates pushed freeze"):
        chain.capture_inventory(
            confirmation_source_root=tmp_path / "must-not-resolve-c",
            boundary_source_root=tmp_path / "must-not-resolve-b",
            stage2_root=tmp_path / "must-not-create",
            layout_path=tmp_path / "must-not-read-layout.json",
            boundary_derivation_path=tmp_path / "must-not-read-derivation.json",
            public_commitment_sha256s={
                role: _sha(role) for role in chain._CONFIRMATION_ROLES
            },
            output=tmp_path / "should-not-exist.json",
            captured_at_utc="2026-08-01T18:18:00Z",
            remote_verifier=lambda: None,
        )


def _git(repo: Path, *args: str, input_text: str | None = None) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def test_remote_freeze_accepts_descendant_head_and_rejects_non_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Stage2 Test")
    _git(repo, "config", "user.email", "stage2@example.invalid")
    (repo / "binding.txt").write_text("freeze\n", encoding="utf-8")
    _git(repo, "add", "binding.txt")
    _git(repo, "commit", "-m", "freeze")
    freeze_commit = _git(repo, "rev-parse", "HEAD")
    (repo / "descendant.txt").write_text("descendant\n", encoding="utf-8")
    _git(repo, "add", "descendant.txt")
    _git(repo, "commit", "-m", "descendant")
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    _git(repo, "remote", "add", "test", str(remote))
    ref = "refs/heads/freeze-chain"
    _git(repo, "push", "test", f"HEAD:{ref}")
    monkeypatch.setattr(chain, "_PUBLIC_FREEZE_REPOSITORY", str(remote))
    monkeypatch.setattr(chain, "_PUBLIC_FREEZE_REF", ref)
    monkeypatch.setattr(chain, "_PUBLIC_FREEZE_COMMIT", freeze_commit)
    chain._verify_remote_freeze(repository_root=repo)

    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    unrelated = _git(repo, "commit-tree", tree, input_text="unrelated\n")
    _git(repo, "push", "--force", "test", f"{unrelated}:{ref}")
    with pytest.raises(ValueError, match="not a locally proven ancestor"):
        chain._verify_remote_freeze(repository_root=repo)


def test_full_tree_capture_and_post_capture_extra_file_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, sealed = _case(tmp_path, monkeypatch)
    output = tmp_path / "authority"
    (sealed / "uncommitted-after-capture.txt").write_text("late\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file set differs"):
        chain.admit(spec_path=spec, output_dir=output)
    assert not output.exists()


def test_replay_rejects_materialized_fixture_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _ = _case(tmp_path, monkeypatch)
    output = tmp_path / "authority"
    chain.admit(spec_path=spec, output_dir=output)
    archive = next((output / "materialized").rglob("*.zip"))
    archive.write_bytes(b"corrupt\n")
    with pytest.raises(ValueError, match="materialized fixture bytes differ"):
        chain.replay(output)


def test_replay_rejects_uncommitted_materialized_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _ = _case(tmp_path, monkeypatch)
    output = tmp_path / "authority"
    chain.admit(spec_path=spec, output_dir=output)
    (output / "materialized" / "late.txt").write_text("late\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file set differs"):
        chain.replay(output)


def test_checked_in_metadata_deviation_is_canonical_and_valid() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "ops/calibration/week5/krea-stage2-sealed-metadata-deviation-20260801.json"
    )
    record, _ = chain._load_canonical(path, "checked-in deviation")
    assert chain.validate_deviation(record)["impact"]["finalist_selection_contaminated"] is False


def test_legacy_wrapper_rejects_an_invented_postfreeze_trigger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, sealed = _case(tmp_path, monkeypatch)
    path = sealed / "C1" / legacy.WRAPPER_NAME
    wrapper = json.loads(path.read_bytes())
    wrapper["trigger_token"] = "invented-after-freeze"
    body = {key: value for key, value in wrapper.items() if key != "wrapper_sha256"}
    wrapper["wrapper_sha256"] = chain.canonical_sha256(body)
    with pytest.raises(ValueError, match="wrapper contract drifted"):
        legacy.validate_wrapper(wrapper)


def test_policy_and_ratification_truthfully_disclose_post_freeze_custodian_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _ = _case(tmp_path, monkeypatch)
    output = tmp_path / "authority"
    chain.admit(spec_path=spec, output_dir=output)
    ratification = json.loads((output / "ratification.json").read_bytes())
    rules = chain.admission.krea_stage2_execution_surface_policy.POLICY[
        "sealed_fixture_rules"
    ]
    assert "only_materialization_may_read_sealed_root" not in rules
    assert rules["post_freeze_custodian_may_hash_and_copy_into_fresh_root"] is True
    assert "sealed_fixture_content_not_read" not in ratification["acknowledgements"]
    assert (
        ratification["acknowledgements"][
            "post_freeze_custodian_hash_copy_and_inventory_reviewed"
        ]
        is True
    )
