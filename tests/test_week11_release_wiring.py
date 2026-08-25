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
SEALED_CANDIDATE = ROOT.parent / "week11-flux-release-codex"

ROLLBACK = "fe9749c027df511b7566b474e6f8524f86b01f83"
ROLLBACK_TREE = "c7e79fb326e3bc3ef7b573d2d0144e82e0e24a25"
ROLLBACK_RECORDS = "c5362ef488af54c7729cba91279b0287b382eee05c5739e802bd894ef72ee582"
ROLLBACK_REF = "refs/heads/week11-ideogram-content-product"
BREAK_GLASS = "59e0698c952edaf1bf34a117ecad41bce87517cf"
TARGET_REF = "refs/heads/codex/week11-flux-release"
TARGET = "3f7737c4045c3877e3c9f100a0341fe2566b1db6"
TARGET_TREE = "42001148b7b8342d9837dcb7e2b532b23892aeee"
TARGET_RECORDS = "cdb54ba614daa58d075d9049ee48a00ebed487f12fcd05e2eda5da929d987f44"
TARGET_SURFACE = "416c825bb8ccb840b2039a3052bffe946b4edc37fed22a5a1bd14824ac07e4f7"
TEXT_PIN = "8f11684e30a556b305dec9dd8eec9794bdae8cde"
HARD_ABORT_EPOCH = 1787761800
PENDING_POLICY_SHA256 = "272651e0725df451c1dba3c4371a8304ce6f3398d1625dc3336e57de7d9fbc60"


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


def mock_reviewed_policy(contract, tmp_path: Path, monkeypatch) -> Path:
    """Promote only a temp policy so CPU tests can exercise the ready mechanics."""

    policy, _ = contract._core._load_json_object(
        CANDIDATE_DOCKER_POLICY_PATH, "Week-11 Docker policy"
    )
    policy["policy_state"] = "reviewed"
    policy["release_evidence"]["production_entrypoint_canary"] = {
        "result": "PASS",
        "image_id": "sha256:" + "1" * 64,
        "receipt_sha256": "2" * 64,
        "runtime_identity_sha256": "3" * 64,
        "first_optimizer_step_sha256": "4" * 64,
        "entrypoint": ["dumb-init", "--", "python3", "-m", "forge.cli"],
        "mock_only": True,
    }
    policy_path = write_json(tmp_path / "mock-reviewed-docker-policy.json", policy)
    monkeypatch.setattr(
        contract,
        "EXPECTED_CANDIDATE_DOCKER_POLICY_SHA256",
        hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    )
    return policy_path


@pytest.fixture(scope="session")
def contract():
    spec = importlib.util.spec_from_file_location("sn56_week11_release_contract", CONTRACT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_release(
    contract, tmp_path: Path, monkeypatch
) -> dict[str, Path | str | dict]:
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

    policy_path = mock_reviewed_policy(contract, tmp_path, monkeypatch)
    generated_path = tmp_path / "selected-hold-manifest.json"
    generated = contract.regenerate_manifest(
        MANIFEST_PATH,
        candidate,
        generated_path,
        policy_path,
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
        "policy": policy_path,
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


def test_ideogram_content_surface_remains_byte_exact_live_production():
    expected = {
        "forge/config.py": "1f1ec28ff1d8cd64f1f06081b669688760dd246ef1498422f9bcf5cbfdec93c7",
        "forge/ideogram_content_policy.py": "1b5e9d2990564d2fe128f6f33958bbda589148e4ea672c37d0e9adc2a48659fe",
        "forge/tasks/aitoolkit.py": "a88718c1b0515b2fe07f1065a00b0efcfeede7b76de2de6f3120a56a483125f9",
        "tests/test_ideogram_content_policy.py": "cd6d158bfda588674436cc00e697d34e1fdfe62689c0d1f2cde9006d328e6f67",
    }
    for relative_path, expected_sha256 in expected.items():
        current = (ROOT / relative_path).read_bytes()
        live = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{ROLLBACK}:{relative_path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout
        assert current == live
        assert hashlib.sha256(current).hexdigest() == expected_sha256


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
        {"status": "M", "path": "forge/flux_kohya_config.py"},
        {"status": "M", "path": "forge/tasks/flux_kohya.py"},
        {
            "status": "A",
            "path": "tests/data/week11_flux_kohya/241cda6c-seed-1.toml",
        },
        {
            "status": "A",
            "path": "tests/data/week11_flux_kohya/db5fefc5-seed-1.toml",
        },
        {"status": "A", "path": "tests/test_week11_flux_release.py"},
    ]
    assert canonical == (
        b"M\tforge/flux_kohya_config.py\n"
        b"M\tforge/tasks/flux_kohya.py\n"
        b"A\ttests/data/week11_flux_kohya/241cda6c-seed-1.toml\n"
        b"A\ttests/data/week11_flux_kohya/db5fefc5-seed-1.toml\n"
        b"A\ttests/test_week11_flux_release.py\n"
    )
    assert surface == TARGET_SURFACE


def test_candidate_docker_policy_is_canonical_pending_and_rejects_drift(
    contract, tmp_path: Path
):
    policy, raw = contract._core._load_json_object(
        CANDIDATE_DOCKER_POLICY_PATH, "Week-11 Docker policy"
    )
    assert CANDIDATE_DOCKER_POLICY_PATH == contract.DEFAULT_DOCKER_POLICY
    repoint = REPOINT_PATH.read_text(encoding="utf-8")
    assert "week11-candidate-docker-policy.json" in repoint
    assert "week10-candidate-docker-policy.json" not in repoint
    assert hashlib.sha256(raw).hexdigest() == PENDING_POLICY_SHA256
    assert PENDING_POLICY_SHA256 == contract.EXPECTED_CANDIDATE_DOCKER_POLICY_SHA256
    assert raw == contract.canonical_json_bytes(policy)
    with pytest.raises(contract.ContractError, match="HOLD pending"):
        contract.load_docker_policy(CANDIDATE_DOCKER_POLICY_PATH)

    assert set(policy) == {
        "schema_version",
        "policy_state",
        "certification_source",
        "dockerfiles",
        "release_evidence",
    }
    assert policy["schema_version"] == 2
    assert policy["policy_state"] == "hold-pending-exact-h100-canary"
    assert policy["certification_source"] == {"commit": TARGET, "tree": TARGET_TREE}
    assert policy["dockerfiles"] == json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )["dockerfiles"]
    evidence = policy["release_evidence"]
    assert set(evidence) == {
        "rollback_commit",
        "secondary_rollback_commit",
        "allowed_changes_name_status_sha256",
        "science_decision_sha256",
        "execution_identity_correction_sha256",
        "base_images",
        "production_entrypoint_canary",
    }
    assert evidence["rollback_commit"] == ROLLBACK
    assert evidence["secondary_rollback_commit"] == BREAK_GLASS
    assert evidence["allowed_changes_name_status_sha256"] == TARGET_SURFACE
    assert evidence["science_decision_sha256"] == (
        "3c192338b0e31ec09d42160a0353588f0c883975990cd7633e351b5494693366"
    )
    assert evidence["execution_identity_correction_sha256"] == (
        "6ecd8a1c9a746c4c5e760fad7b440c2299aa58a443d67c2fa95717126eccc39b"
    )
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
    assert evidence["production_entrypoint_canary"] == {
        "result": "PENDING",
        "required_mode": "offline-one-h100-controlled-stop-after-first-optimizer-step",
        "required_entrypoint": ["dumb-init", "--", "python3", "-m", "forge.cli"],
    }

    substitutions = []
    wrong_canary = copy.deepcopy(policy)
    wrong_canary["release_evidence"]["production_entrypoint_canary"]["result"] = "PASS"
    substitutions.append(wrong_canary)
    wrong_base = copy.deepcopy(policy)
    wrong_base["release_evidence"]["base_images"][0]["digest"] = "sha256:" + "0" * 64
    substitutions.append(wrong_base)
    extra_field = copy.deepcopy(policy)
    extra_field["release_evidence"]["production_entrypoint_canary"]["unreviewed"] = True
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
        "tree": "613a1cc2d750731df007cc9b2b49e461d0ae368f",
        "tree_records_sha256": "3ea4b92e0f2a90537527a3809b6615a97122eb69b497282f4ecacc267bf382d4",
        "ref": "refs/heads/codex/week10-release-recovery",
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
    contract, tmp_path: Path, monkeypatch
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

    release = _candidate_release(contract, tmp_path, monkeypatch)
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
    contract, tmp_path: Path, monkeypatch
):
    release = _candidate_release(contract, tmp_path, monkeypatch)
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
        "forge/flux_kohya_config.py",
        "forge/tasks/flux_kohya.py",
        "tests/data/week11_flux_kohya/241cda6c-seed-1.toml",
        "tests/data/week11_flux_kohya/db5fefc5-seed-1.toml",
        "tests/test_week11_flux_release.py",
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
    contract, tmp_path: Path, monkeypatch
):
    release = _candidate_release(contract, tmp_path, monkeypatch)
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
    assert "insufficient verification and rollback margin" in proc.stdout
    assert not mockroot.exists()


def test_forward_dry_run_passes_before_cutoff_without_mutating_host(
    contract, tmp_path: Path, monkeypatch
):
    release = _candidate_release(contract, tmp_path, monkeypatch)
    manifest_path, readiness_path = _ready_release(tmp_path, release)
    mockroot = tmp_path / "host"
    endpoint = prepare_mock_host(mockroot, ROLLBACK)
    before = endpoint.read_bytes()
    env = os.environ.copy()
    env["SN56_FAKE_NOW_EPOCH"] = str(HARD_ABORT_EPOCH - 3600)
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
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DRY RUN COMPLETE -- nothing was modified." in proc.stdout
    assert TARGET in proc.stdout
    assert endpoint.read_bytes() == before
    assert not (mockroot / "backups").exists()


def test_forward_mutation_requires_exact_ready_readiness(
    contract, tmp_path: Path, monkeypatch
):
    release = _candidate_release(contract, tmp_path, monkeypatch)
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
    contract, tmp_path: Path, monkeypatch
):
    release = _candidate_release(contract, tmp_path, monkeypatch)
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
