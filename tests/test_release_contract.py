from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "scripts" / "sn56-release-contract.py"
REPOINT_PATH = ROOT / "scripts" / "sn56-week6-repoint.sh"
MANIFEST_PATH = ROOT / "release" / "week9-release-manifest.json"
DOCKER_POLICY_PATH = ROOT / "release" / "week9-docker-policy.json"
READINESS_PATH = ROOT / "release" / "week9-release-readiness.json"

MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
DOCKER_POLICY = json.loads(DOCKER_POLICY_PATH.read_text(encoding="utf-8"))
TARGET = MANIFEST["target"]["commit"]
TARGET_TREE = MANIFEST["target"]["tree"]
DOCKER_CERTIFICATION_COMMIT = DOCKER_POLICY["certification_source"]["commit"]
CED = "ced58e2e3db68f9ca094b4959de7e2f4a812c0ac"
ROLLBACK = "75a0a20c2deda82cfa727e082e60a95bea5befb3"
TEXT_PIN = "8f11684e30a556b305dec9dd8eec9794bdae8cde"


def run(args: list[str | Path], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
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


@pytest.fixture(scope="session")
def contract():
    spec = importlib.util.spec_from_file_location("sn56_release_contract", CONTRACT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def seed_remote(tmp_path_factory: pytest.TempPathFactory, manifest: dict) -> Path:
    remote = tmp_path_factory.mktemp("release-remote-seed") / "remote.git"
    proc = run(["git", "clone", "--quiet", "--bare", ROOT, remote])
    assert proc.returncode == 0, proc.stderr
    git(remote, "update-ref", manifest["target"]["ref"], TARGET)
    git(remote, "update-ref", manifest["rollback"]["ref"], ROLLBACK)
    return remote


@pytest.fixture
def isolated_release(tmp_path: Path, seed_remote: Path, manifest: dict) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    proc = run(["git", "clone", "--quiet", "--bare", seed_remote, remote])
    assert proc.returncode == 0, proc.stderr
    git(remote, "update-ref", manifest["target"]["ref"], TARGET)
    git(remote, "update-ref", manifest["rollback"]["ref"], ROLLBACK)

    reviewed = tmp_path / "reviewed"
    proc = run(["git", "clone", "--quiet", remote, reviewed])
    assert proc.returncode == 0, proc.stderr
    git(reviewed, "checkout", "--quiet", "--detach", TARGET)
    assert git(reviewed, "status", "--porcelain=v1", "--untracked-files=all") == ""
    return remote, reviewed


def write_manifest(tmp_path: Path, data: dict, name: str = "manifest.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def validate(contract, manifest_path: Path, remote: Path, reviewed: Path):
    return contract.validate_contract(
        manifest_path,
        mock=True,
        repository_url=str(remote),
        reviewed_worktree=str(reviewed),
    )


def training_repo_source(
    image_pin: str,
    image_repo: str = "https://github.com/tuly1/sn56-forge-toolkit.git",
) -> str:
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
        github_repo="{image_repo}",
        commit_hash="{image_pin}",
    ),
    TournamentType.TEXT: TrainingRepoResponse(
        github_repo="https://github.com/rayonlabs/G.O.D.git",
        commit_hash="{TEXT_PIN}",
    ),
}}
'''


def ready_receipt_for_manifest(manifest_path: Path, data: dict) -> dict:
    policy_raw = DOCKER_POLICY_PATH.read_bytes()
    return {
        "schema_version": 1,
        "readiness_state": "ready",
        "manifest_sha256": __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest(),
        "docker_policy_sha256": __import__("hashlib").sha256(policy_raw).hexdigest(),
        "target": copy.deepcopy(data["target"]),
        "rollback": copy.deepcopy(data["rollback"]),
        "allowed_changes": {
            "base_commit": data["allowed_changes"]["base_commit"],
            "name_status_sha256": data["allowed_changes"]["name_status_sha256"],
            "count": len(data["allowed_changes"]["entries"]),
        },
        "dockerfiles": copy.deepcopy(data["dockerfiles"]),
    }


def write_ready_release(tmp_path: Path, manifest: dict) -> tuple[Path, Path]:
    changed = copy.deepcopy(manifest)
    changed["release_state"] = "ready"
    ready_manifest = write_manifest(tmp_path, changed, "ready-manifest.json")
    readiness = write_manifest(
        tmp_path,
        ready_receipt_for_manifest(ready_manifest, changed),
        "reviewed-readiness.json",
    )
    return ready_manifest, readiness


def run_repoint(
    manifest_path: Path,
    remote: Path,
    reviewed: Path,
    mockroot: Path,
    *extra: str,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["SN56_FAKE_NOW_EPOCH"] = "1787000000"
    if env_overrides:
        env.update(env_overrides)
    return run(
        repoint_command(manifest_path, remote, reviewed, mockroot, *extra),
        cwd=ROOT,
        env=env,
    )


def repoint_command(
    manifest_path: Path,
    remote: Path,
    reviewed: Path,
    mockroot: Path,
    *extra: str,
) -> list[str]:
    return [
        "bash",
        str(REPOINT_PATH),
        "--manifest",
        str(manifest_path),
        "--mock",
        str(mockroot),
        "--repository-url",
        str(remote),
        "--reviewed-worktree",
        str(reviewed),
        *extra,
    ]


def test_checked_in_safe_baseline_hold_contract_passes_with_exact_local_refs(
    contract, manifest: dict, isolated_release: tuple[Path, Path]
):
    remote, reviewed = isolated_release
    receipt = validate(contract, MANIFEST_PATH, remote, reviewed)

    assert receipt["release_state"] == "hold"
    assert receipt["target"]["commit"] == TARGET
    assert receipt["target"]["tree"] == TARGET_TREE
    assert receipt["rollback"]["commit"] == ROLLBACK
    assert receipt["allowed_changes"] == {
        "base_commit": ROLLBACK,
        "name_status_sha256": "424e85fb4ce70ade6d5590ac2c3b9b3ec7c290fc904afd87d5a3ee44ffe16ade",
        "count": 38,
    }
    assert receipt["verified_anonymous_remote"]["target_ref"] == manifest["target"]["ref"]
    assert receipt["docker_policy"]["sha256"] == (
        "476ae3c34458ac547607e98c587d7f631bbc60263db3e83f75401b9454dab129"
    )
    assert receipt["verified_local"]["docker_certification_source"] == (
        receipt["verified_anonymous_remote"]["docker_certification_source"]
    )
    assert (
        receipt["verified_local"]["docker_certification_source"]["commit"]
        == DOCKER_CERTIFICATION_COMMIT
    )


def test_positional_ced_target_is_not_an_interface():
    proc = run(["bash", REPOINT_PATH, CED, "--contract-only"], cwd=ROOT)
    assert proc.returncode == 2
    assert "positional target SHAs are forbidden" in proc.stderr


def test_mutating_manifest_target_to_ced_fails_exact_head_binding(
    contract, manifest: dict, isolated_release: tuple[Path, Path], tmp_path: Path
):
    remote, reviewed = isolated_release
    changed = copy.deepcopy(manifest)
    changed["target"]["commit"] = CED
    path = write_manifest(tmp_path, changed)

    with pytest.raises(contract.ContractError, match="HEAD is .* expected"):
        validate(contract, path, remote, reviewed)


@pytest.mark.parametrize(
    ("flag", "tracked_path"),
    [
        ("--assume-unchanged", "ops/docker/standalone-image-toolkit-trainer.dockerfile"),
        ("--skip-worktree", "forge/config.py"),
    ],
    ids=["assume-unchanged-docker", "skip-worktree-science"],
)
def test_index_flags_cannot_hide_arbitrary_tracked_mutations(
    contract,
    isolated_release: tuple[Path, Path],
    flag: str,
    tracked_path: str,
):
    remote, reviewed = isolated_release
    git(reviewed, "update-index", flag, tracked_path)
    with (reviewed / tracked_path).open("ab") as handle:
        handle.write(b"\n# adversarial local mutation\n")
    assert git(reviewed, "status", "--porcelain=v1", "--untracked-files=all") == ""

    with pytest.raises(contract.ContractError, match="forbidden assume-unchanged/skip-worktree"):
        validate(contract, MANIFEST_PATH, remote, reviewed)


def test_stale_target_ref_fails_even_when_commit_object_exists(
    contract, manifest: dict, isolated_release: tuple[Path, Path]
):
    remote, reviewed = isolated_release
    git(remote, "update-ref", manifest["target"]["ref"], CED)

    with pytest.raises(contract.ContractError, match="anonymous remote refs differ"):
        validate(contract, MANIFEST_PATH, remote, reviewed)


def test_dirty_reviewed_worktree_fails_on_untracked_bytes(
    contract, isolated_release: tuple[Path, Path]
):
    remote, reviewed = isolated_release
    (reviewed / "unreviewed.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(contract.ContractError, match="reviewed worktree is dirty"):
        validate(contract, MANIFEST_PATH, remote, reviewed)


def test_wrong_rollback_rejected_even_if_manifest_fields_are_coordinated(
    contract, manifest: dict
):
    changed = copy.deepcopy(manifest)
    changed["rollback"] = {
        "commit": CED,
        "tree": "5a5c6cf0ef6a650f630b015d45a4d8b18c805e8d",
        "tree_records_sha256": "c9869a6ddce294f4501b23d7a3f05cc6743402bac228fd755779babb51b0fb7b",
        "ref": manifest["rollback"]["ref"],
    }
    changed["allowed_changes"]["base_commit"] = CED

    with pytest.raises(contract.ContractError, match="exact production rollback"):
        contract.validate_schema(changed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("service", "wrong-miner.service"),
        ("service_user", "root"),
        ("service_working_directory", "/home/miner"),
        (
            "service_exec_start",
            "/home/miner/.venv/bin/uvicorn miner.asgi:app --host 127.0.0.1 --port 7999",
        ),
        ("service_asgi_module", "/home/miner/god/miner/wrong_asgi.py"),
        ("endpoint_route", "/training_repo/ImageTask"),
    ],
)
def test_wrong_production_process_service_or_route_is_not_manifest_overridable(
    contract, manifest: dict, field: str, value: str
):
    changed = copy.deepcopy(manifest)
    changed["production"][field] = value

    with pytest.raises(contract.ContractError, match="production literals differ"):
        contract.validate_schema(changed)


def test_hold_manifest_refuses_forward_mutation_before_host_access(
    isolated_release: tuple[Path, Path], tmp_path: Path
):
    remote, reviewed = isolated_release
    mockroot = tmp_path / "host-does-not-exist"
    proc = run_repoint(MANIFEST_PATH, remote, reviewed, mockroot)

    assert proc.returncode == 5
    assert "release_state is 'hold', not 'ready'; mutation forbidden" in proc.stderr
    assert not mockroot.exists()


def test_manifest_only_hold_to_ready_edit_cannot_unlock_mutation(
    manifest: dict, isolated_release: tuple[Path, Path], tmp_path: Path
):
    remote, reviewed = isolated_release
    changed = copy.deepcopy(manifest)
    changed["release_state"] = "ready"
    ready_manifest = write_manifest(tmp_path, changed, "ready-manifest.json")
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    before = training_repo_source(ROLLBACK)
    endpoint.write_text(before, encoding="utf-8")

    proc = run_repoint(ready_manifest, remote, reviewed, mockroot)

    assert proc.returncode == 5
    assert "readiness receipt manifest_sha256 does not bind" in proc.stderr
    assert "nothing touched" in proc.stderr
    assert endpoint.read_text(encoding="utf-8") == before


def test_exact_independent_ready_receipt_unlocks_mock_only_mutation(
    manifest: dict, isolated_release: tuple[Path, Path], tmp_path: Path
):
    remote, reviewed = isolated_release
    changed = copy.deepcopy(manifest)
    changed["release_state"] = "ready"
    ready_manifest = write_manifest(tmp_path, changed, "ready-manifest.json")
    readiness = write_manifest(
        tmp_path,
        ready_receipt_for_manifest(ready_manifest, changed),
        "reviewed-readiness.json",
    )
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    endpoint.write_text(training_repo_source(ROLLBACK), encoding="utf-8")

    proc = run_repoint(
        ready_manifest,
        remote,
        reviewed,
        mockroot,
        "--readiness-receipt",
        str(readiness),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "READINESS PASS" in proc.stdout
    assert TARGET in endpoint.read_text(encoding="utf-8")


@pytest.mark.parametrize("drift", ["target-ref", "dirty-worktree"])
def test_post_confirmation_contract_revalidation_rejects_late_source_drift(
    manifest: dict,
    isolated_release: tuple[Path, Path],
    tmp_path: Path,
    drift: str,
):
    remote, reviewed = isolated_release
    ready_manifest, readiness = write_ready_release(tmp_path, manifest)
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    before = training_repo_source(ROLLBACK)
    endpoint.write_text(before, encoding="utf-8")
    gate = tmp_path / f"continue-after-initial-{drift}"
    ready_marker = Path(f"{gate}.ready")
    env = os.environ.copy()
    env.update(
        {
            "SN56_FAKE_NOW_EPOCH": "1787000000",
            "SN56_MOCK_AFTER_INITIAL_CONTRACT_GATE": str(gate),
        }
    )
    command = repoint_command(
        ready_manifest,
        remote,
        reviewed,
        mockroot,
        "--readiness-receipt",
        str(readiness),
    )
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 30
        while not ready_marker.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if not ready_marker.exists():
            stdout, stderr = proc.communicate(timeout=5)
            pytest.fail(f"repoint never reached initial-contract gate:\n{stdout}\n{stderr}")

        if drift == "target-ref":
            git(remote, "update-ref", manifest["target"]["ref"], CED)
        else:
            (reviewed / "late-unreviewed.txt").write_text("dirty\n", encoding="utf-8")
        gate.write_text("continue\n", encoding="utf-8")
        stdout, stderr = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    assert proc.returncode == 5, stdout + stderr
    assert "post-confirmation full release contract failed" in stdout
    assert "REPOINT COMPLETE" not in stdout
    assert endpoint.read_text(encoding="utf-8") == before
    assert not (mockroot / "backups").exists()


@pytest.mark.parametrize(
    ("clock_overrides", "backup_expected", "phase"),
    [
        ({"SN56_FAKE_PREBACKUP_EPOCH": "1787574600"}, False, "pre-backup"),
        (
            {
                "SN56_FAKE_PREBACKUP_EPOCH": "1787574599",
                "SN56_FAKE_PREAPPLY_EPOCH": "1787574600",
            },
            True,
            "pre-apply",
        ),
    ],
)
def test_forward_cutoff_is_resampled_at_each_mutation_boundary(
    manifest: dict,
    isolated_release: tuple[Path, Path],
    tmp_path: Path,
    clock_overrides: dict[str, str],
    backup_expected: bool,
    phase: str,
):
    remote, reviewed = isolated_release
    ready_manifest, readiness = write_ready_release(tmp_path, manifest)
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    before = training_repo_source(ROLLBACK)
    endpoint.write_text(before, encoding="utf-8")

    proc = run_repoint(
        ready_manifest,
        remote,
        reviewed,
        mockroot,
        "--readiness-receipt",
        str(readiness),
        env_overrides=clock_overrides,
    )

    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert f"at {phase}" in proc.stdout
    assert endpoint.read_text(encoding="utf-8") == before
    backups = list((mockroot / "backups").glob("*")) if (mockroot / "backups").exists() else []
    assert bool(backups) is backup_expected


def test_term_after_apply_runs_verified_armed_rollback(
    manifest: dict, isolated_release: tuple[Path, Path], tmp_path: Path
):
    remote, reviewed = isolated_release
    ready_manifest, readiness = write_ready_release(tmp_path, manifest)
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    before = training_repo_source(ROLLBACK)
    endpoint.write_text(before, encoding="utf-8")

    proc = run_repoint(
        ready_manifest,
        remote,
        reviewed,
        mockroot,
        "--readiness-receipt",
        str(readiness),
        env_overrides={"SN56_MOCK_SIGNAL_AT": "after-apply"},
    )

    assert proc.returncode == 143, proc.stdout + proc.stderr
    assert "TERM received with forward rollback armed" in proc.stdout
    assert "SIGNAL RECOVERY COMPLETE" in proc.stdout
    assert "probe 422 -- endpoint healthy on the ORIGINAL pin" in proc.stdout
    assert endpoint.read_text(encoding="utf-8") == before
    assert not list(mockroot.glob("evidence-*.json"))


@pytest.mark.parametrize(
    ("gate_env", "expected_error"),
    [
        ("SN56_MOCK_AFTER_BACKUP_GATE", "live-file preimage sha256"),
        ("SN56_MOCK_AFTER_EDIT_GATE", "post-edit live sha256"),
    ],
    ids=["before-editor", "after-editor-return"],
)
def test_same_length_live_file_race_is_rejected_and_rolled_back(
    manifest: dict,
    isolated_release: tuple[Path, Path],
    tmp_path: Path,
    gate_env: str,
    expected_error: str,
):
    remote, reviewed = isolated_release
    ready_manifest, readiness = write_ready_release(tmp_path, manifest)
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    before = training_repo_source(ROLLBACK).encode("utf-8")
    endpoint.write_bytes(before)
    gate = tmp_path / f"continue-{gate_env.lower()}"
    ready_marker = Path(f"{gate}.ready")
    env = os.environ.copy()
    env.update(
        {
            "SN56_FAKE_NOW_EPOCH": "1787000000",
            gate_env: str(gate),
        }
    )
    command = repoint_command(
        ready_manifest,
        remote,
        reviewed,
        mockroot,
        "--readiness-receipt",
        str(readiness),
    )
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 30
        while not ready_marker.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if not ready_marker.exists():
            stdout, stderr = proc.communicate(timeout=5)
            pytest.fail(f"repoint never reached after-backup gate:\n{stdout}\n{stderr}")

        at_gate = endpoint.read_bytes()
        raced = at_gate.replace(b"rayonlabs", b"adversary", 1)
        assert raced != at_gate
        assert len(raced) == len(before)
        endpoint.write_bytes(raced)
        gate.write_text("continue\n", encoding="utf-8")
        stdout, stderr = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    assert proc.returncode == 6, stdout + stderr
    assert expected_error in stdout
    assert "AUTO-ROLLBACK" in stdout
    assert "REPOINT COMPLETE" not in stdout
    assert endpoint.read_bytes() == before
    backups = list((mockroot / "backups").glob("training_repo.py.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == before


def test_final_anonymous_ref_reproof_rolls_back_if_target_ref_moves(
    manifest: dict, isolated_release: tuple[Path, Path], tmp_path: Path
):
    remote, reviewed = isolated_release
    ready_manifest, readiness = write_ready_release(tmp_path, manifest)
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    before = training_repo_source(ROLLBACK)
    endpoint.write_text(before, encoding="utf-8")
    gate = tmp_path / "continue-before-final-contract"
    ready_marker = Path(f"{gate}.ready")
    env = os.environ.copy()
    env.update(
        {
            "SN56_FAKE_NOW_EPOCH": "1787000000",
            "SN56_MOCK_BEFORE_FINAL_CONTRACT_GATE": str(gate),
        }
    )
    command = repoint_command(
        ready_manifest,
        remote,
        reviewed,
        mockroot,
        "--readiness-receipt",
        str(readiness),
    )
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 60
        while not ready_marker.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if not ready_marker.exists():
            stdout, stderr = proc.communicate(timeout=5)
            pytest.fail(f"repoint never reached final-contract gate:\n{stdout}\n{stderr}")

        assert TARGET in endpoint.read_text(encoding="utf-8")
        git(remote, "update-ref", manifest["target"]["ref"], CED)
        gate.write_text("continue\n", encoding="utf-8")
        stdout, stderr = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    assert proc.returncode == 6, stdout + stderr
    assert "final full release contract/ref reproof failed" in stdout
    assert "AUTO-ROLLBACK" in stdout
    assert "REPOINT COMPLETE" not in stdout
    assert endpoint.read_text(encoding="utf-8") == before


def test_manual_rollback_remains_available_offline_with_dirty_target_worktree(
    manifest: dict, isolated_release: tuple[Path, Path], tmp_path: Path
):
    _, reviewed = isolated_release
    (reviewed / "post-release-operator-note.txt").write_text(
        "dirty after release\n", encoding="utf-8"
    )
    offline_remote = tmp_path / "github-is-offline.git"
    ready_manifest, readiness = write_ready_release(tmp_path, manifest)
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    endpoint.write_text(training_repo_source(TARGET), encoding="utf-8")

    proc = run_repoint(
        ready_manifest,
        offline_remote,
        reviewed,
        mockroot,
        "--readiness-receipt",
        str(readiness),
        "--rollback",
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ROLLBACK CONTRACT PASS" in proc.stdout
    assert "emergency rollback identity is offline" in proc.stdout
    assert ROLLBACK in endpoint.read_text(encoding="utf-8")
    assert TARGET not in endpoint.read_text(encoding="utf-8")


def test_manual_rollback_rejects_unknown_served_pin_even_with_ready_identity(
    manifest: dict, isolated_release: tuple[Path, Path], tmp_path: Path
):
    _, reviewed = isolated_release
    ready_manifest, readiness = write_ready_release(tmp_path, manifest)
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    before = training_repo_source(CED)
    endpoint.write_text(before, encoding="utf-8")

    proc = run_repoint(
        ready_manifest,
        tmp_path / "offline.git",
        reviewed,
        mockroot,
        "--readiness-receipt",
        str(readiness),
        "--rollback",
    )

    assert proc.returncode == 5
    assert f"rollback preflight expected served release {TARGET}" in proc.stdout
    assert endpoint.read_text(encoding="utf-8") == before


def test_hold_manifest_dry_run_succeeds_from_exact_rollback_pin(
    isolated_release: tuple[Path, Path], tmp_path: Path
):
    remote, reviewed = isolated_release
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    before = training_repo_source(ROLLBACK)
    endpoint.write_text(before, encoding="utf-8")

    proc = run_repoint(MANIFEST_PATH, remote, reviewed, mockroot, "--dry-run")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DRY RUN COMPLETE -- nothing was modified" in proc.stdout
    assert "NON-SHIPPABLE" in proc.stdout
    assert endpoint.read_text(encoding="utf-8") == before


def test_forward_dry_run_rejects_wrong_served_pin(
    isolated_release: tuple[Path, Path], tmp_path: Path
):
    remote, reviewed = isolated_release
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    before = training_repo_source(CED)
    endpoint.write_text(before, encoding="utf-8")

    proc = run_repoint(MANIFEST_PATH, remote, reviewed, mockroot, "--dry-run")

    assert proc.returncode == 5
    assert "forward preflight requires the exact rollback pin" in proc.stdout
    assert endpoint.read_text(encoding="utf-8") == before


def test_forward_dry_run_rejects_wrong_live_image_repository_before_write(
    isolated_release: tuple[Path, Path], tmp_path: Path
):
    remote, reviewed = isolated_release
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    before = training_repo_source(
        ROLLBACK, "https://github.com/example/adversarial-repository.git"
    )
    endpoint.write_text(before, encoding="utf-8")

    proc = run_repoint(MANIFEST_PATH, remote, reviewed, mockroot, "--dry-run")

    assert proc.returncode == 5
    assert "IMAGE github_repo is" in proc.stdout
    assert "expected reviewed repository" in proc.stdout
    assert endpoint.read_text(encoding="utf-8") == before


def test_prepare_readiness_refuses_hold_manifest(tmp_path: Path):
    output = tmp_path / "must-not-exist.json"
    proc = run(
        [
            sys.executable,
            CONTRACT_PATH,
            "--prepare-readiness",
            "--manifest",
            MANIFEST_PATH,
            "--output",
            output,
        ],
        cwd=ROOT,
    )

    assert proc.returncode == 4
    assert "requires an exact canonical manifest with release_state ready" in proc.stderr
    assert not output.exists()


def test_prepare_readiness_derives_exact_hold_bindings_for_independent_review(
    contract,
    manifest: dict,
    isolated_release: tuple[Path, Path],
    tmp_path: Path,
):
    remote, reviewed = isolated_release
    ready_manifest, _ = write_ready_release(tmp_path, manifest)
    prepared_path = tmp_path / "prepared-readiness.json"
    proc = run(
        [
            sys.executable,
            CONTRACT_PATH,
            "--prepare-readiness",
            "--manifest",
            ready_manifest,
            "--output",
            prepared_path,
        ],
        cwd=ROOT,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PREPARED HOLD readiness receipt" in proc.stdout
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    assert prepared["readiness_state"] == "hold"
    assert prepared["manifest_sha256"] == __import__("hashlib").sha256(
        ready_manifest.read_bytes()
    ).hexdigest()
    assert prepared["docker_policy_sha256"] == __import__("hashlib").sha256(
        DOCKER_POLICY_PATH.read_bytes()
    ).hexdigest()
    assert prepared_path.read_bytes() == contract.canonical_json_bytes(prepared)

    prepared["readiness_state"] = "ready"
    reviewed_readiness = write_manifest(
        tmp_path, prepared, "independently-reviewed-readiness.json"
    )
    contract_receipt = write_manifest(
        tmp_path,
        validate(contract, ready_manifest, remote, reviewed),
        "validated-contract-receipt.json",
    )
    proof = contract.validate_readiness_receipt(
        reviewed_readiness, contract_receipt
    )
    assert proof["readiness_state"] == "ready"


def test_regeneration_from_clean_candidate_recomputes_and_forces_hold(
    contract, isolated_release: tuple[Path, Path], tmp_path: Path
):
    _, reviewed = isolated_release
    output = tmp_path / "successor-manifest.json"
    generated = contract.regenerate_manifest(MANIFEST_PATH, reviewed, output)

    assert output.exists()
    assert generated["release_state"] == "hold"
    assert generated["target"]["commit"] == TARGET
    assert generated["allowed_changes"]["base_commit"] == ROLLBACK
    assert generated["allowed_changes"]["name_status_sha256"] == (
        "424e85fb4ce70ade6d5590ac2c3b9b3ec7c290fc904afd87d5a3ee44ffe16ade"
    )
    assert generated["source"]["reviewed_worktree"] == str(reviewed.resolve())


def test_regeneration_refuses_dirty_candidate(
    contract, isolated_release: tuple[Path, Path], tmp_path: Path
):
    _, reviewed = isolated_release
    (reviewed / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(contract.ContractError, match="candidate worktree is dirty"):
        contract.regenerate_manifest(MANIFEST_PATH, reviewed, tmp_path / "out.json")


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_regeneration_rejects_hidden_index_flags_on_arbitrary_tracked_paths(
    contract,
    isolated_release: tuple[Path, Path],
    tmp_path: Path,
    flag: str,
):
    _, reviewed = isolated_release
    tracked_path = "forge/config.py"
    git(reviewed, "update-index", flag, tracked_path)
    with (reviewed / tracked_path).open("ab") as handle:
        handle.write(b"\n# hidden candidate mutation\n")
    assert git(reviewed, "status", "--porcelain=v1", "--untracked-files=all") == ""
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(
        contract.ContractError,
        match="candidate worktree has a forbidden assume-unchanged/skip-worktree",
    ):
        contract.regenerate_manifest(MANIFEST_PATH, reviewed, output)
    assert not output.exists()


def test_regeneration_cannot_derive_or_bless_changed_docker_hashes(
    contract, isolated_release: tuple[Path, Path], tmp_path: Path
):
    _, reviewed = isolated_release
    git(reviewed, "config", "user.name", "Release Contract Test")
    git(reviewed, "config", "user.email", "release-contract@example.invalid")
    dockerfile = reviewed / "ops/docker/standalone-image-toolkit-trainer.dockerfile"
    with dockerfile.open("ab") as handle:
        handle.write(b"\n# unreviewed successor Docker mutation\n")
    git(reviewed, "add", "ops/docker/standalone-image-toolkit-trainer.dockerfile")
    git(reviewed, "commit", "--quiet", "-m", "adversarial Docker mutation")
    policy = json.loads(DOCKER_POLICY_PATH.read_text(encoding="utf-8"))
    policy["dockerfiles"][0]["sha256"] = __import__("hashlib").sha256(
        dockerfile.read_bytes()
    ).hexdigest()
    coordinated_policy = write_manifest(tmp_path, policy, "coordinated-policy.json")

    with pytest.raises(contract.ContractError, match="hashes are immutable"):
        contract.regenerate_manifest(
            MANIFEST_PATH,
            reviewed,
            tmp_path / "must-not-exist.json",
            coordinated_policy,
        )
    assert not (tmp_path / "must-not-exist.json").exists()


def test_regeneration_accepts_flux_only_successor_with_fixed_docker_policy_and_stays_hold(
    contract, isolated_release: tuple[Path, Path], tmp_path: Path
):
    _, reviewed = isolated_release
    git(reviewed, "config", "user.name", "Release Contract Test")
    git(reviewed, "config", "user.email", "release-contract@example.invalid")
    relative = "release-flux-successor.txt"
    (reviewed / relative).write_text("flux-only successor\n", encoding="utf-8")
    git(reviewed, "add", relative)
    git(reviewed, "commit", "--quiet", "-m", "flux-only successor")
    successor = git(reviewed, "rev-parse", "HEAD")
    output = tmp_path / "successor.json"

    generated = contract.regenerate_manifest(MANIFEST_PATH, reviewed, output)

    assert generated["release_state"] == "hold"
    assert generated["target"]["commit"] == successor
    assert generated["dockerfiles"] == json.loads(
        DOCKER_POLICY_PATH.read_text(encoding="utf-8")
    )["dockerfiles"]


def test_mock_overrides_are_rejected_without_explicit_mock(contract, isolated_release):
    remote, reviewed = isolated_release
    with pytest.raises(contract.ContractError, match="require explicit --mock"):
        contract.validate_contract(
            MANIFEST_PATH,
            repository_url=str(remote),
            reviewed_worktree=str(reviewed),
        )


def test_anonymous_git_proof_drops_inherited_git_config_and_credentials(
    contract,
    isolated_release: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    remote, reviewed = isolated_release
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "url.file:///definitely-missing.insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(remote))
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-enter-anonymous-git")
    monkeypatch.setenv("SSH_AUTH_SOCK", str(tmp_path / "must-not-enter.sock"))

    anonymous_env = contract._git_env(anonymous=True, home=tmp_path / "isolated-home")
    assert "GIT_CONFIG_COUNT" not in anonymous_env
    assert "GIT_CONFIG_KEY_0" not in anonymous_env
    assert "GITHUB_TOKEN" not in anonymous_env
    assert "SSH_AUTH_SOCK" not in anonymous_env

    receipt = validate(contract, MANIFEST_PATH, remote, reviewed)
    assert receipt["target"]["commit"] == TARGET


def test_receipt_write_is_exclusive(isolated_release: tuple[Path, Path], tmp_path: Path):
    remote, reviewed = isolated_release
    receipt = tmp_path / "receipt.json"
    base = [
        sys.executable,
        CONTRACT_PATH,
        "--manifest",
        MANIFEST_PATH,
        "--contract-only",
        "--mock",
        "--repository-url",
        remote,
        "--reviewed-worktree",
        reviewed,
        "--receipt",
        receipt,
    ]
    first = run(base, cwd=ROOT)
    second = run(base, cwd=ROOT)

    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 4
    assert "receipt already exists; refusing to overwrite" in second.stderr
