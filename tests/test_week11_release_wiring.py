from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "scripts" / "sn56-week11-release-contract.py"
REPOINT_PATH = ROOT / "scripts" / "sn56-week11-repoint.sh"
ROLLBACK_PATH = ROOT / "scripts" / "sn56-week11-rollback.sh"
MANIFEST_PATH = ROOT / "release" / "week11-release-manifest.json"
READINESS_PATH = ROOT / "release" / "week11-release-readiness.json"
CANDIDATE_DOCKER_POLICY_PATH = ROOT / "release" / "week11-candidate-docker-policy.json"
MANIFEST_SIGNERS_PATH = ROOT / "release" / "week11-release-allowed-signers"
READINESS_SIGNERS_PATH = ROOT / "release" / "week11-readiness-allowed-signers"
SEALED_CANDIDATE = ROOT.parent / "week11-ideogram-content-release-codex"

ROLLBACK = "59e0698c952edaf1bf34a117ecad41bce87517cf"
ROLLBACK_TREE = "613a1cc2d750731df007cc9b2b49e461d0ae368f"
ROLLBACK_RECORDS = "3ea4b92e0f2a90537527a3809b6615a97122eb69b497282f4ecacc267bf382d4"
ROLLBACK_REF = "refs/heads/week10-trainer-candidate"
BREAK_GLASS = "75a0a20c2deda82cfa727e082e60a95bea5befb3"
TARGET_REF = "refs/heads/week11-ideogram-content-product"
TARGET = "fe9749c027df511b7566b474e6f8524f86b01f83"
TARGET_TREE = "c7e79fb326e3bc3ef7b573d2d0144e82e0e24a25"
TARGET_RECORDS = "c5362ef488af54c7729cba91279b0287b382eee05c5739e802bd894ef72ee582"
TARGET_SURFACE = "d15197e9bac0369efc71c35c3f7ded8e0e1bf47397f594cd5a2d4b877f397e05"
TEXT_PIN = "8f11684e30a556b305dec9dd8eec9794bdae8cde"
HARD_ABORT_EPOCH = 1787761800
IMAGE_ID = "sha256:987fa5c8964d04bb8aa2ad0fbc485aecafe63b1f85dae1da65cc960e917c8cd8"
CANARY_RECEIPT = "4020d7a9ef90992b97c637b794d86e143bddaf61a95a47fc16ea55aace97e6de"
IMAGE_HISTORY = "3044dca1bafa2c68e5dc99dc2b526b75af9f5a1a19a6f8e8b38c2f996d2a9209"
RUNTIME_INVENTORY = "9c4c15130508c547c67d891f559ca1a513cd62bd5a4b695eb25ceafccd0b850b"
PRODUCT_PROJECTION = "ce226348ca641932de4a0f27d59a69ad8a124825068a4f1e9e9d134166eba4cd"


def run(
    args: list[str | Path],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def git(repo: Path, *args: str) -> str:
    proc = run(["git", "-C", repo, *args])
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def contract():
    spec = importlib.util.spec_from_file_location("sn56_week11_release_contract", CONTRACT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_release(contract, tmp_path: Path) -> dict[str, Path | str | dict]:
    candidate = tmp_path / "candidate"
    proc = run(
        ["git", "clone", "--quiet", "--no-hardlinks", SEALED_CANDIDATE, candidate]
    )
    assert proc.returncode == 0, proc.stderr
    git(candidate, "checkout", "--quiet", TARGET_REF.removeprefix("refs/heads/"))
    git(candidate, "update-ref", ROLLBACK_REF, ROLLBACK)
    target = git(candidate, "rev-parse", "HEAD")
    target_tree = git(candidate, "rev-parse", "HEAD^{tree}")
    entries, _, surface_digest = contract.name_status(candidate, ROLLBACK, target)
    assert target == TARGET
    assert target_tree == TARGET_TREE
    assert contract.tree_records(candidate, target)[1] == TARGET_RECORDS
    assert surface_digest == TARGET_SURFACE

    generated_path = tmp_path / "selected-hold-manifest.json"
    generated = contract.regenerate_manifest(
        MANIFEST_PATH,
        candidate,
        generated_path,
        CANDIDATE_DOCKER_POLICY_PATH,
    )

    remote = tmp_path / "remote.git"
    proc = run(["git", "clone", "--quiet", "--bare", candidate, remote])
    assert proc.returncode == 0, proc.stderr
    git(remote, "update-ref", TARGET_REF, target)
    git(remote, "update-ref", ROLLBACK_REF, ROLLBACK)
    return {
        "candidate": candidate,
        "remote": remote,
        "target": target,
        "target_tree": target_tree,
        "entries": entries,
        "surface_digest": surface_digest,
        "policy": CANDIDATE_DOCKER_POLICY_PATH,
        "generated": generated,
        "generated_path": generated_path,
    }


def _ready_release(tmp_path: Path, release: dict) -> tuple[Path, Path]:
    manifest = copy.deepcopy(release["generated"])
    manifest["release_state"] = "ready"
    manifest_path = write_json(tmp_path / "ready-manifest.json", manifest)
    policy_path = Path(release["policy"])
    readiness = {
        "schema_version": 1,
        "readiness_state": "ready",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "docker_policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "target": copy.deepcopy(manifest["target"]),
        "rollback": copy.deepcopy(manifest["rollback"]),
        "allowed_changes": {
            "base_commit": manifest["allowed_changes"]["base_commit"],
            "name_status_sha256": manifest["allowed_changes"]["name_status_sha256"],
            "count": len(manifest["allowed_changes"]["entries"]),
        },
        "dockerfiles": copy.deepcopy(manifest["dockerfiles"]),
    }
    readiness_path = write_json(tmp_path / "ready-readiness.json", readiness)
    return manifest_path, readiness_path


def training_repo_source(image_pin: str) -> str:
    return f'''from enum import Enum

class TournamentType(Enum):
    IMAGE = "image"
    TEXT = "text"

class TrainingRepoResponse:
    def __init__(self, github_repo, commit_hash):
        self.github_repo = github_repo
        self.commit_hash = commit_hash

_REPOS = {{
    TournamentType.IMAGE: TrainingRepoResponse(
        github_repo="https://github.com/tuly1/sn56-forge-toolkit.git",
        commit_hash="{image_pin}",
    ),
    TournamentType.TEXT: TrainingRepoResponse(
        github_repo="https://github.com/rayonlabs/G.O.D.git",
        commit_hash="{TEXT_PIN}",
    ),
}}
'''


def prepare_mock_host(root: Path, image_pin: str) -> Path:
    root.mkdir()
    endpoint = root / "training_repo.py"
    endpoint.write_text(training_repo_source(image_pin), encoding="utf-8")
    pyc = root / "__pycache__" / "training_repo.cpython-312.pyc"
    pyc.parent.mkdir()
    pyc.write_bytes(b"mock-pyc\0" + image_pin.encode() + b"\0" + TEXT_PIN.encode())
    return endpoint


def repoint_command(
    release: dict,
    manifest_path: Path,
    readiness_path: Path,
    mockroot: Path,
    *extra: str,
) -> list[str | Path]:
    return [
        "bash",
        REPOINT_PATH,
        "--manifest",
        manifest_path,
        "--docker-policy",
        release["policy"],
        "--readiness-receipt",
        readiness_path,
        "--mock",
        mockroot,
        "--repository-url",
        release["remote"],
        "--reviewed-worktree",
        release["candidate"],
        *extra,
    ]


def test_primary_rollback_is_exact_current_production(contract):
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    contract.validate_schema(manifest)
    assert manifest["rollback"] == {
        "commit": ROLLBACK,
        "tree": ROLLBACK_TREE,
        "tree_records_sha256": ROLLBACK_RECORDS,
        "ref": ROLLBACK_REF,
    }
    assert git(ROOT, "rev-parse", f"{ROLLBACK}^{{tree}}") == ROLLBACK_TREE
    _, actual_records = contract.tree_records(ROOT, ROLLBACK)
    assert actual_records == ROLLBACK_RECORDS


def test_sealed_candidate_has_exact_local_ref_tree_and_changed_surface(contract):
    assert SEALED_CANDIDATE.is_dir()
    assert git(SEALED_CANDIDATE, "status", "--porcelain=v1", "--untracked-files=all") == ""
    assert git(SEALED_CANDIDATE, "rev-parse", "HEAD") == TARGET
    assert git(SEALED_CANDIDATE, "rev-parse", "HEAD^{tree}") == TARGET_TREE
    assert git(SEALED_CANDIDATE, "rev-parse", TARGET_REF) == TARGET
    _, records = contract.tree_records(SEALED_CANDIDATE, TARGET)
    assert records == TARGET_RECORDS
    entries, canonical, surface = contract.name_status(
        SEALED_CANDIDATE, ROLLBACK, TARGET
    )
    assert entries == [
        {"status": "M", "path": "forge/config.py"},
        {"status": "A", "path": "forge/ideogram_content_policy.py"},
        {"status": "M", "path": "forge/tasks/aitoolkit.py"},
        {"status": "A", "path": "tests/test_ideogram_content_policy.py"},
    ]
    assert canonical == (
        b"M\tforge/config.py\n"
        b"A\tforge/ideogram_content_policy.py\n"
        b"M\tforge/tasks/aitoolkit.py\n"
        b"A\ttests/test_ideogram_content_policy.py\n"
    )
    assert surface == TARGET_SURFACE


def test_candidate_docker_policy_is_canonical_exact_and_rejects_drift(
    contract, tmp_path: Path
):
    policy, raw = contract.load_docker_policy(CANDIDATE_DOCKER_POLICY_PATH)
    assert CANDIDATE_DOCKER_POLICY_PATH == contract.DEFAULT_DOCKER_POLICY
    repoint = REPOINT_PATH.read_text(encoding="utf-8")
    assert "week11-candidate-docker-policy.json" in repoint
    assert "week10-candidate-docker-policy.json" not in repoint
    assert hashlib.sha256(raw).hexdigest() == (
        contract.EXPECTED_CANDIDATE_DOCKER_POLICY_SHA256
    )
    assert raw == contract.canonical_json_bytes(policy)
    assert set(policy) == {
        "schema_version",
        "policy_state",
        "certification_source",
        "dockerfiles",
        "release_evidence",
    }
    assert policy["schema_version"] == 2
    assert policy["policy_state"] == "reviewed"
    assert policy["certification_source"] == {"commit": TARGET, "tree": TARGET_TREE}
    assert policy["dockerfiles"] == json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )["dockerfiles"]
    evidence = policy["release_evidence"]
    assert set(evidence) == {
        "rollback_commit",
        "allowed_changes_name_status_sha256",
        "image_id",
        "base_images",
        "offline_canary",
    }
    assert evidence["rollback_commit"] == ROLLBACK
    assert evidence["allowed_changes_name_status_sha256"] == TARGET_SURFACE
    assert evidence["image_id"] == IMAGE_ID
    assert evidence["base_images"] == [
        {
            "path": "ops/docker/standalone-image-toolkit-trainer.dockerfile",
            "image": "diagonalge/ai-toolkit:latest",
            "digest": "sha256:c24f8bb95bf1dc8da7cd6158a763f2c9782783ad7648dc4047c5757ef3447db8",
        },
        {
            "path": "ops/docker/standalone-image-trainer.dockerfile",
            "image": "diagonalge/ai-toolkit:latest",
            "digest": "sha256:c24f8bb95bf1dc8da7cd6158a763f2c9782783ad7648dc4047c5757ef3447db8",
        },
        {
            "path": "ops/docker/standalone-image-trainer.dockerfile",
            "image": "diagonalge/kohya_latest:latest",
            "digest": "sha256:d34dd5750e1018455e111f63c03bb2a4e16204607e00ba5af870dd7c71beb84e",
        },
    ]
    assert evidence["offline_canary"] == {
        "result": "PASS",
        "receipt_sha256": CANARY_RECEIPT,
        "history_sha256": IMAGE_HISTORY,
        "runtime_inventory_sha256": RUNTIME_INVENTORY,
        "product_projection_sha256": PRODUCT_PROJECTION,
    }

    substitutions = []
    wrong_image = copy.deepcopy(policy)
    wrong_image["release_evidence"]["image_id"] = "sha256:" + "0" * 64
    substitutions.append(wrong_image)
    wrong_canary = copy.deepcopy(policy)
    wrong_canary["release_evidence"]["offline_canary"]["receipt_sha256"] = "0" * 64
    substitutions.append(wrong_canary)
    wrong_base = copy.deepcopy(policy)
    wrong_base["release_evidence"]["base_images"][0]["digest"] = "sha256:" + "0" * 64
    substitutions.append(wrong_base)
    extra_field = copy.deepcopy(policy)
    extra_field["release_evidence"]["offline_canary"]["unreviewed"] = True
    substitutions.append(extra_field)
    for index, substituted in enumerate(substitutions):
        substituted_path = write_json(tmp_path / f"substituted-{index}.json", substituted)
        with pytest.raises(contract.ContractError, match="exact canonical candidate policy"):
            contract.load_docker_policy(substituted_path)

    noncanonical_path = tmp_path / "noncanonical.json"
    noncanonical_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(contract.ContractError, match="not canonical JSON"):
        contract.load_docker_policy(noncanonical_path)


def test_break_glass_commit_is_rejected_as_primary_rollback(contract):
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["rollback"] = {
        "commit": BREAK_GLASS,
        "tree": "bdf44638853e4cd96f0bd0e420d17bcd519c1d0b",
        "tree_records_sha256": "b158fae4fcf155cf754ce8056df28ae0eae247dd7300f132f29de92fd47006ad",
        "ref": "refs/heads/claude/week8-mse-revert",
    }
    manifest["allowed_changes"]["base_commit"] = BREAK_GLASS
    with pytest.raises(contract.ContractError, match="exact production rollback"):
        contract.validate_schema(manifest)


def test_checked_in_artifacts_are_canonical_unselected_hold(contract):
    manifest, manifest_raw = contract.load_manifest(MANIFEST_PATH)
    readiness, readiness_raw = contract._load_json_object(
        READINESS_PATH, "Week-11 readiness"
    )
    contract.validate_schema(manifest)

    assert manifest["release_state"] == "hold"
    assert manifest["target"] == contract.UNSELECTED_TARGET
    assert manifest["allowed_changes"]["entries"] == []
    assert manifest["candidate_source_evidence"] == contract.CANDIDATE_SOURCE_EVIDENCE
    assert readiness["readiness_state"] == "hold"
    assert readiness["target"] == contract.UNSELECTED_TARGET
    assert readiness["rollback"] == manifest["rollback"]
    assert readiness["manifest_sha256"] == hashlib.sha256(manifest_raw).hexdigest()
    assert readiness["docker_policy_sha256"] == hashlib.sha256(
        CANDIDATE_DOCKER_POLICY_PATH.read_bytes()
    ).hexdigest()
    assert readiness_raw == contract.canonical_json_bytes(readiness)

    tampered_evidence = copy.deepcopy(manifest)
    tampered_evidence["candidate_source_evidence"]["target"]["commit"] = BREAK_GLASS
    with pytest.raises(contract.ContractError, match="exact sealed Week-11 source contract"):
        contract.validate_schema(tampered_evidence)

    with pytest.raises(contract.ContractError, match="final target is unselected"):
        contract.validate_contract(MANIFEST_PATH)

    proc = run(["bash", REPOINT_PATH, "--contract-only"], cwd=ROOT)
    assert proc.returncode == 4
    assert "final target is unselected" in proc.stderr


def test_fresh_signing_domains_are_empty_and_live_ready_fails_closed(
    contract, tmp_path: Path
):
    assert contract.SIGNING_PRINCIPAL == "sn56-week11-release"
    assert contract.SIGNING_NAMESPACE == "sn56-week11-final-manifest"
    assert contract.READINESS_SIGNING_PRINCIPAL == "sn56-week11-readiness"
    assert contract.READINESS_SIGNING_NAMESPACE == "sn56-week11-final-readiness"
    for authority in (MANIFEST_SIGNERS_PATH, READINESS_SIGNERS_PATH):
        configured = [
            line
            for line in authority.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert configured == []
    assert not list((ROOT / "release").glob("week11-*.sig"))

    release = _candidate_release(contract, tmp_path)
    ready_path, _ = _ready_release(tmp_path, release)
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    with pytest.raises(contract.ContractError, match="cannot read final-manifest signature authority"):
        contract.verify_manifest_signature(
            ready_path,
            ready,
            ready_path.read_bytes(),
            mock=False,
        )


def test_create_only_preparation_binds_target_ref_and_exact_diff(
    contract, tmp_path: Path
):
    release = _candidate_release(contract, tmp_path)
    generated = release["generated"]
    assert generated["release_state"] == "hold"
    assert generated["candidate_source_evidence"] == contract.CANDIDATE_SOURCE_EVIDENCE
    assert generated["target"] == {
        "commit": TARGET,
        "tree": TARGET_TREE,
        "tree_records_sha256": TARGET_RECORDS,
        "ref": TARGET_REF,
    }
    assert generated["allowed_changes"] == {
        "base_commit": ROLLBACK,
        "name_status_sha256": TARGET_SURFACE,
        "entries": release["entries"],
    }
    assert {row["path"] for row in release["entries"]} == {
        "forge/config.py",
        "forge/ideogram_content_policy.py",
        "forge/tasks/aitoolkit.py",
        "tests/test_ideogram_content_policy.py",
    }
    assert not {
        "ops/docker/standalone-image-toolkit-trainer.dockerfile",
        "ops/docker/standalone-image-trainer.dockerfile",
    } & {row["path"] for row in release["entries"]}

    with pytest.raises(contract.ContractError, match="output already exists"):
        contract.regenerate_manifest(
            MANIFEST_PATH,
            Path(release["candidate"]),
            Path(release["generated_path"]),
            Path(release["policy"]),
        )

    git(Path(release["candidate"]), "checkout", "--quiet", "--detach", TARGET)
    git(Path(release["candidate"]), "update-ref", TARGET_REF, ROLLBACK)
    with pytest.raises(contract.ContractError, match="candidate target ref resolves"):
        contract.regenerate_manifest(
            MANIFEST_PATH,
            Path(release["candidate"]),
            tmp_path / "must-not-exist.json",
            Path(release["policy"]),
        )
    assert not (tmp_path / "must-not-exist.json").exists()
    git(Path(release["candidate"]), "update-ref", TARGET_REF, TARGET)

    receipt = contract.validate_contract(
        Path(release["generated_path"]),
        docker_policy_path=Path(release["policy"]),
        mock=True,
        repository_url=str(release["remote"]),
        reviewed_worktree=str(release["candidate"]),
    )
    assert receipt["target"]["commit"] == TARGET
    assert receipt["allowed_changes"]["name_status_sha256"] == TARGET_SURFACE

    wrong_tree = copy.deepcopy(generated)
    wrong_tree["target"]["tree"] = "0" * 40
    wrong_tree_path = write_json(tmp_path / "wrong-tree.json", wrong_tree)
    with pytest.raises(contract.ContractError, match="candidate_source_evidence"):
        contract.validate_contract(
            wrong_tree_path,
            docker_policy_path=Path(release["policy"]),
            mock=True,
            repository_url=str(release["remote"]),
            reviewed_worktree=str(release["candidate"]),
        )

    wrong_diff = copy.deepcopy(generated)
    wrong_diff["allowed_changes"]["name_status_sha256"] = "0" * 64
    with pytest.raises(contract.ContractError, match="do not hash"):
        contract.validate_schema(wrong_diff)

    git(Path(release["remote"]), "update-ref", TARGET_REF, ROLLBACK)
    with pytest.raises(contract.ContractError, match="anonymous remote refs differ"):
        contract.validate_contract(
            Path(release["generated_path"]),
            docker_policy_path=Path(release["policy"]),
            mock=True,
            repository_url=str(release["remote"]),
            reviewed_worktree=str(release["candidate"]),
        )


def test_forward_dry_run_refuses_at_cutoff_without_touching_host(
    contract, tmp_path: Path
):
    release = _candidate_release(contract, tmp_path)
    manifest_path, readiness_path = _ready_release(tmp_path, release)
    mockroot = tmp_path / "host-must-not-exist"
    env = os.environ.copy()
    env["SN56_FAKE_NOW_EPOCH"] = str(HARD_ABORT_EPOCH)
    proc = run(
        repoint_command(
            release,
            manifest_path,
            readiness_path,
            mockroot,
            "--dry-run",
        ),
        cwd=ROOT,
        env=env,
    )
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "current UTC time is past the hard abort 2026-08-26 16:30:00 UTC" in proc.stdout
    assert "18:00 UTC science deadline" in proc.stdout
    assert not mockroot.exists()


def test_forward_mutation_requires_exact_ready_readiness(contract, tmp_path: Path):
    release = _candidate_release(contract, tmp_path)
    manifest_path, _ = _ready_release(tmp_path, release)
    mockroot = tmp_path / "host"
    endpoint = prepare_mock_host(mockroot, ROLLBACK)
    before = endpoint.read_bytes()
    env = os.environ.copy()
    env["SN56_FAKE_NOW_EPOCH"] = str(HARD_ABORT_EPOCH - 3600)
    proc = run(
        repoint_command(
            release,
            manifest_path,
            READINESS_PATH,
            mockroot,
        ),
        cwd=ROOT,
        env=env,
    )
    assert proc.returncode == 5
    assert "independent reviewed readiness receipt did not bind" in proc.stderr
    assert endpoint.read_bytes() == before
    assert not (mockroot / "backups").exists()


def test_exact_rollback_wrapper_remains_available_after_forward_cutoff(
    contract, tmp_path: Path
):
    release = _candidate_release(contract, tmp_path)
    manifest_path, readiness_path = _ready_release(tmp_path, release)
    mockroot = tmp_path / "host"
    endpoint = prepare_mock_host(mockroot, str(release["target"]))
    env = os.environ.copy()
    env["SN56_FAKE_NOW_EPOCH"] = str(HARD_ABORT_EPOCH + 3600)
    proc = run(
        [
            "bash",
            ROLLBACK_PATH,
            "--manifest",
            manifest_path,
            "--docker-policy",
            release["policy"],
            "--readiness-receipt",
            readiness_path,
            "--mock",
            mockroot,
            "--repository-url",
            release["remote"],
            "--reviewed-worktree",
            release["candidate"],
        ],
        cwd=ROOT,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.splitlines()[0] == (
        f"SN56 WEEK-11 ROLLBACK -> {ROLLBACK} (exact current production)"
    )
    assert "forward repoint is barred, but ROLLBACK is always permitted" in proc.stdout
    assert ROLLBACK in endpoint.read_text(encoding="utf-8")
    assert str(release["target"]) not in endpoint.read_text(encoding="utf-8")


def test_one_line_wrapper_is_executable_and_has_only_primary_rollback():
    wrapper = ROLLBACK_PATH.read_text(encoding="utf-8")
    assert os.access(ROLLBACK_PATH, os.X_OK)
    assert f'EXPECTED_ROLLBACK="{ROLLBACK}"' in wrapper
    assert BREAK_GLASS not in wrapper
    assert 'exec "$REPOINT" --manifest "$MANIFEST" --rollback "${FORWARD_ARGS[@]}"' in wrapper
    assert REPOINT_PATH.name in wrapper
