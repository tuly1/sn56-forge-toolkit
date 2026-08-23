from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "scripts" / "sn56-release-contract.py"
REPOINT_PATH = ROOT / "scripts" / "sn56-week6-repoint.sh"
ROLLBACK_PATH = ROOT / "scripts" / "sn56-week6-rollback.sh"
RUNBOOK_PATH = ROOT / "SUNDAY-RELEASE-RUNBOOK.md"
MANIFEST_PATH = ROOT / "release" / "week9-release-manifest.json"
SELECTED_MANIFEST_PATH = ROOT / "tests" / "data" / "week9-release-selected-hold.json"
DOCKER_POLICY_PATH = ROOT / "release" / "week9-docker-policy.json"
READINESS_PATH = ROOT / "release" / "week9-release-readiness.json"
CANDIDATE_MANIFEST_PATH = ROOT / "release" / "week10-release-manifest.json"
CANDIDATE_DOCKER_POLICY_PATH = ROOT / "release" / "week10-candidate-docker-policy.json"
CANDIDATE_READINESS_PATH = ROOT / "release" / "week10-release-readiness.json"

MANIFEST = json.loads(SELECTED_MANIFEST_PATH.read_text(encoding="utf-8"))
DOCKER_POLICY = json.loads(DOCKER_POLICY_PATH.read_text(encoding="utf-8"))
TARGET = MANIFEST["target"]["commit"]
TARGET_TREE = MANIFEST["target"]["tree"]
DOCKER_CERTIFICATION_COMMIT = DOCKER_POLICY["certification_source"]["commit"]
UNRELATED = "bd852dc0986b661983b70a8e2d225b6da0be971e"
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


def public_key_material(private_key: Path) -> str:
    fields = private_key.with_suffix(".pub").read_text(encoding="utf-8").split()
    assert len(fields) >= 2
    return " ".join(fields[:2])


@pytest.fixture(scope="session")
def contract():
    spec = importlib.util.spec_from_file_location("sn56_release_contract", CONTRACT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def manifest() -> dict:
    return json.loads(SELECTED_MANIFEST_PATH.read_text(encoding="utf-8"))


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
    *,
    text_first: bool = False,
) -> str:
    image_entry = f'''    TournamentType.IMAGE: TrainingRepoResponse(
        github_repo="{image_repo}",
        commit_hash="{image_pin}",
    ),
'''
    text_entry = f'''    TournamentType.TEXT: TrainingRepoResponse(
        github_repo="https://github.com/rayonlabs/G.O.D.git",
        commit_hash="{TEXT_PIN}",
    ),
'''
    entries = text_entry + image_entry if text_first else image_entry + text_entry
    return f'''from enum import Enum

class TournamentType(Enum):
    IMAGE = "image"
    TEXT = "text"

class TrainingRepoResponse:
    def __init__(self, github_repo, commit_hash):
        self.github_repo = github_repo
        self.commit_hash = commit_hash

_REPOS = {{
{entries}
}}
'''


def fiber_auth_body(*, validator_count: int = 2, signature_count: int = 1) -> str:
    detail = (
        [
            {"type": "missing", "loc": ["header", "validator-hotkey"]}
            for _ in range(validator_count)
        ]
        + [
            {"type": "missing", "loc": ["header", "signature"]}
            for _ in range(signature_count)
        ]
        + [
            {"type": "missing", "loc": ["header", "miner-hotkey"]},
            {"type": "missing", "loc": ["header", "nonce"]},
        ]
    )
    return json.dumps({"detail": detail})


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
    endpoint = mockroot / "training_repo.py"
    pyc = mockroot / "__pycache__" / "training_repo.cpython-312.pyc"
    if endpoint.exists() and not pyc.exists():
        pins = __import__("re").findall(r"[0-9a-f]{40}", endpoint.read_text(encoding="utf-8"))
        pyc.parent.mkdir(parents=True, exist_ok=True)
        pyc.write_bytes(b"mock-pyc\0" + b"\0".join(pin.encode("ascii") for pin in pins))
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
    receipt = validate(contract, SELECTED_MANIFEST_PATH, remote, reviewed)

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


def test_checked_in_manifest_is_an_explicit_unselected_hold(contract):
    placeholder = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    contract.validate_schema(placeholder)
    assert placeholder["release_state"] == "hold"
    assert placeholder["target"] == contract.UNSELECTED_TARGET
    with pytest.raises(contract.ContractError, match="final target is unselected"):
        contract.validate_contract(MANIFEST_PATH)


def test_checked_in_candidate_policy_and_ready_receipt_bind_exact_target(contract):
    candidate, candidate_raw = contract.load_manifest(CANDIDATE_MANIFEST_PATH)
    contract.validate_schema(candidate)
    policy = contract.verify_docker_policy(candidate, CANDIDATE_DOCKER_POLICY_PATH)
    readiness, readiness_raw = contract._load_json_object(
        CANDIDATE_READINESS_PATH, "candidate readiness receipt"
    )

    assert candidate["release_state"] == "ready"
    assert candidate["target"]["commit"] == "59e0698c952edaf1bf34a117ecad41bce87517cf"
    assert candidate["target"]["ref"] == "refs/heads/week10-trainer-candidate"
    assert policy["schema_version"] == 2
    assert policy["certification_source"] == {
        "commit": candidate["target"]["commit"],
        "tree": candidate["target"]["tree"],
    }
    assert policy["release_evidence"]["image_digest"] == (
        "sha256:fc319058b098e569fe177c0f21f8c65beafed45e391211061fd0c61c1362cf0a"
    )
    assert readiness["readiness_state"] == "ready"
    assert readiness["manifest_sha256"] == __import__("hashlib").sha256(
        candidate_raw
    ).hexdigest()
    assert readiness["docker_policy_sha256"] == policy["sha256"]
    assert readiness["target"] == candidate["target"]
    assert readiness_raw == contract.canonical_json_bytes(readiness)


def test_unselected_template_identity_is_not_tied_to_one_target_ref(contract):
    placeholder = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    placeholder["target"]["ref"] = "refs/heads/week10-trainer-candidate"
    contract.validate_schema(placeholder)
    assert contract.target_is_unselected(placeholder)


def test_unselected_target_cannot_be_flipped_ready(contract):
    placeholder = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    placeholder["release_state"] = "ready"
    with pytest.raises(contract.ContractError, match="unselected final target"):
        contract.validate_schema(placeholder)


@pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="ssh-keygen unavailable")
def test_ready_manifest_requires_exact_trusted_detached_signature(
    contract, manifest: dict, tmp_path: Path
):
    ready = copy.deepcopy(manifest)
    ready["release_state"] = "ready"
    signed = write_manifest(tmp_path, ready, "final.json")
    key = tmp_path / "release-key"
    proc = run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", key])
    assert proc.returncode == 0, proc.stderr
    allowed = tmp_path / "allowed-signers"
    allowed.write_text(
        f"{contract.SIGNING_PRINCIPAL} {public_key_material(key)}\n",
        encoding="utf-8",
    )
    proc = run(
        [
            "ssh-keygen",
            "-Y",
            "sign",
            "-f",
            key,
            "-n",
            contract.SIGNING_NAMESPACE,
            signed,
        ]
    )
    assert proc.returncode == 0, proc.stderr

    receipt = contract.verify_manifest_signature(
        signed,
        ready,
        signed.read_bytes(),
        mock=False,
        allowed_signers_path=allowed,
    )
    assert receipt["state"] == "verified"
    assert receipt["principal"] == contract.SIGNING_PRINCIPAL

    tampered = copy.deepcopy(ready)
    tampered["target"]["commit"] = UNRELATED
    signed.write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(contract.ContractError, match="signature verification failed"):
        contract.verify_manifest_signature(
            signed,
            tampered,
            signed.read_bytes(),
            mock=False,
            allowed_signers_path=allowed,
        )


@pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="ssh-keygen unavailable")
def test_ready_readiness_requires_distinct_trusted_detached_signature(
    contract,
    manifest: dict,
    isolated_release: tuple[Path, Path],
    tmp_path: Path,
):
    remote, reviewed = isolated_release
    ready_manifest, readiness = write_ready_release(tmp_path, manifest)
    manifest_key = tmp_path / "manifest-key"
    readiness_key = tmp_path / "readiness-key"
    for key in (manifest_key, readiness_key):
        proc = run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", key])
        assert proc.returncode == 0, proc.stderr
    manifest_allowed = tmp_path / "manifest-allowed-signers"
    manifest_allowed.write_text(
        f"{contract.SIGNING_PRINCIPAL} {public_key_material(manifest_key)}\n",
        encoding="utf-8",
    )
    proc = run(
        [
            "ssh-keygen",
            "-Y",
            "sign",
            "-f",
            manifest_key,
            "-n",
            contract.SIGNING_NAMESPACE,
            ready_manifest,
        ]
    )
    assert proc.returncode == 0, proc.stderr
    contract_data = validate(contract, ready_manifest, remote, reviewed)
    contract_data["manifest_signature"] = contract.verify_manifest_signature(
        ready_manifest,
        json.loads(ready_manifest.read_text(encoding="utf-8")),
        ready_manifest.read_bytes(),
        mock=False,
        allowed_signers_path=manifest_allowed,
    )
    contract_receipt = write_manifest(
        tmp_path,
        contract_data,
        "validated-contract.json",
    )

    with pytest.raises(contract.ContractError, match="cannot read final-readiness signature authority"):
        contract.validate_readiness_receipt(readiness, contract_receipt)

    allowed = tmp_path / "readiness-allowed-signers"
    allowed.write_text(
        f"{contract.READINESS_SIGNING_PRINCIPAL} {public_key_material(readiness_key)}\n",
        encoding="utf-8",
    )

    wrong_key_readiness = tmp_path / "wrong-key-readiness.json"
    shutil.copyfile(readiness, wrong_key_readiness)
    proc = run(
        [
            "ssh-keygen",
            "-Y",
            "sign",
            "-f",
            manifest_key,
            "-n",
            contract.READINESS_SIGNING_NAMESPACE,
            wrong_key_readiness,
        ]
    )
    assert proc.returncode == 0, proc.stderr
    with pytest.raises(contract.ContractError, match="signature verification failed"):
        contract.validate_readiness_receipt(
            wrong_key_readiness,
            contract_receipt,
            allowed_signers_path=allowed,
        )

    wrong_namespace_readiness = tmp_path / "wrong-namespace-readiness.json"
    shutil.copyfile(readiness, wrong_namespace_readiness)
    proc = run(
        [
            "ssh-keygen",
            "-Y",
            "sign",
            "-f",
            readiness_key,
            "-n",
            contract.SIGNING_NAMESPACE,
            wrong_namespace_readiness,
        ]
    )
    assert proc.returncode == 0, proc.stderr
    with pytest.raises(contract.ContractError, match="signature verification failed"):
        contract.validate_readiness_receipt(
            wrong_namespace_readiness,
            contract_receipt,
            allowed_signers_path=allowed,
        )

    same_key_readiness = tmp_path / "same-key-readiness.json"
    shutil.copyfile(readiness, same_key_readiness)
    same_key_allowed = tmp_path / "same-key-readiness-allowed-signers"
    same_key_allowed.write_text(
        f"{contract.READINESS_SIGNING_PRINCIPAL} {public_key_material(manifest_key)}\n",
        encoding="utf-8",
    )
    proc = run(
        [
            "ssh-keygen",
            "-Y",
            "sign",
            "-f",
            manifest_key,
            "-n",
            contract.READINESS_SIGNING_NAMESPACE,
            same_key_readiness,
        ]
    )
    assert proc.returncode == 0, proc.stderr
    with pytest.raises(contract.ContractError, match="must differ from the final-manifest"):
        contract.validate_readiness_receipt(
            same_key_readiness,
            contract_receipt,
            allowed_signers_path=same_key_allowed,
        )

    proc = run(
        [
            "ssh-keygen",
            "-Y",
            "sign",
            "-f",
            readiness_key,
            "-n",
            contract.READINESS_SIGNING_NAMESPACE,
            readiness,
        ]
    )
    assert proc.returncode == 0, proc.stderr
    proof = contract.validate_readiness_receipt(
        readiness,
        contract_receipt,
        allowed_signers_path=allowed,
    )
    assert proof["signature"]["state"] == "verified"
    assert proof["signature"]["principal"] == contract.READINESS_SIGNING_PRINCIPAL
    assert proof["signature"]["namespace"] == contract.READINESS_SIGNING_NAMESPACE


def test_positional_target_is_not_an_interface():
    proc = run(["bash", REPOINT_PATH, UNRELATED, "--contract-only"], cwd=ROOT)
    assert proc.returncode == 2
    assert "positional target SHAs are forbidden" in proc.stderr


def test_one_line_rollback_and_current_runbook_bind_exact_prestate():
    wrapper = ROLLBACK_PATH.read_text(encoding="utf-8")
    runbook = RUNBOOK_PATH.read_text(encoding="utf-8")
    assert os.access(ROLLBACK_PATH, os.X_OK)
    assert f'EXPECTED_ROLLBACK="{ROLLBACK}"' in wrapper
    assert "084ea914c6c5cbac4fa26a2138bd7195ebd71488" not in wrapper
    assert 'exec "$REPOINT" --manifest "$MANIFEST" --rollback "${FORWARD_ARGS[@]}"' in wrapper
    assert ROLLBACK in runbook
    assert "ced58e2" not in runbook
    assert "SN56_MANIFEST=release/week10-release-manifest.json" in runbook
    assert "SN56_POLICY=release/week10-candidate-docker-policy.json" in runbook
    assert "SN56_READINESS=release/week10-release-readiness.json" in runbook
    assert '''bash scripts/sn56-week6-repoint.sh \\
  --manifest "$SN56_MANIFEST" \\
  --docker-policy "$SN56_POLICY" \\
  --readiness-receipt "$SN56_READINESS" \\
  --contract-only''' in runbook
    assert '''bash scripts/sn56-week6-repoint.sh \\
  --manifest "$SN56_MANIFEST" \\
  --docker-policy "$SN56_POLICY" \\
  --readiness-receipt "$SN56_READINESS" \\
  --dry-run''' in runbook
    assert "sn56-release-readiness" in runbook
    assert "sn56-final-readiness" in runbook
    assert "47261" in runbook and "preserve-forever" in runbook
    assert "optional) delete" not in runbook.lower()
    assert "orphan volume" not in runbook.lower()


def test_mutating_manifest_target_fails_exact_head_binding(
    contract, manifest: dict, isolated_release: tuple[Path, Path], tmp_path: Path
):
    remote, reviewed = isolated_release
    changed = copy.deepcopy(manifest)
    changed["target"]["commit"] = UNRELATED
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
        validate(contract, SELECTED_MANIFEST_PATH, remote, reviewed)


def test_stale_target_ref_fails_even_when_commit_object_exists(
    contract, manifest: dict, isolated_release: tuple[Path, Path]
):
    remote, reviewed = isolated_release
    git(remote, "update-ref", manifest["target"]["ref"], UNRELATED)

    with pytest.raises(contract.ContractError, match="anonymous remote refs differ"):
        validate(contract, SELECTED_MANIFEST_PATH, remote, reviewed)


def test_dirty_reviewed_worktree_fails_on_untracked_bytes(
    contract, isolated_release: tuple[Path, Path]
):
    remote, reviewed = isolated_release
    (reviewed / "unreviewed.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(contract.ContractError, match="reviewed worktree is dirty"):
        validate(contract, SELECTED_MANIFEST_PATH, remote, reviewed)


def test_wrong_rollback_rejected_even_if_manifest_fields_are_coordinated(
    contract, manifest: dict
):
    changed = copy.deepcopy(manifest)
    changed["rollback"] = {
        "commit": UNRELATED,
        "tree": "5a5c6cf0ef6a650f630b015d45a4d8b18c805e8d",
        "tree_records_sha256": "c9869a6ddce294f4501b23d7a3f05cc6743402bac228fd755779babb51b0fb7b",
        "ref": manifest["rollback"]["ref"],
    }
    changed["allowed_changes"]["base_commit"] = UNRELATED

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
    proc = run_repoint(SELECTED_MANIFEST_PATH, remote, reviewed, mockroot)

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
    rollback_line = next(
        line for line in proc.stdout.splitlines() if "ONE-LINE ROLLBACK:" in line
    )
    assert f"--manifest {ready_manifest}" in rollback_line
    assert "--docker-policy" in rollback_line
    assert "week9-docker-policy.json" in rollback_line
    assert f"--readiness-receipt {readiness}" in rollback_line
    assert rollback_line.endswith("--rollback")
    evidence_paths = list(mockroot.glob("evidence-*.json"))
    assert len(evidence_paths) == 1
    rollback_command = json.loads(evidence_paths[0].read_text(encoding="utf-8"))[
        "rollback_command"
    ]
    assert f"--manifest {ready_manifest}" in rollback_command
    assert "--docker-policy" in rollback_command
    assert f"--readiness-receipt {readiness}" in rollback_command
    assert rollback_command.endswith("--rollback")


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
            git(remote, "update-ref", manifest["target"]["ref"], UNRELATED)
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
    assert "endpoint reaches exact Fiber auth stage on the ORIGINAL pin" in proc.stdout
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
        git(remote, "update-ref", manifest["target"]["ref"], UNRELATED)
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


def test_manual_rollback_is_a_proven_noop_at_exact_source_and_pyc_prestate(
    manifest: dict, isolated_release: tuple[Path, Path], tmp_path: Path
):
    _, reviewed = isolated_release
    ready_manifest, readiness = write_ready_release(tmp_path, manifest)
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    before = training_repo_source(ROLLBACK)
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

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (
        "active service, exact source contract, running bytecode, and Fiber route already serve the rollback pin"
        in proc.stdout
    )
    assert (
        "AST proves exact IMAGE/TEXT mapping, reviewed IMAGE repository, and pins"
        in proc.stdout
    )
    assert endpoint.read_text(encoding="utf-8") == before
    assert not list((mockroot / "backups").glob("*.bak"))


def test_manual_rollback_noop_rejects_environment_repo_entry(
    manifest: dict, isolated_release: tuple[Path, Path], tmp_path: Path
):
    _, reviewed = isolated_release
    ready_manifest, readiness = write_ready_release(tmp_path, manifest)
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    before = training_repo_source(ROLLBACK)
    before = before.replace(
        '    TEXT = "text"\n',
        '    TEXT = "text"\n    ENVIRONMENT = "environment"\n',
    ).replace(
        "\n}\n",
        '\n    TournamentType.ENVIRONMENT: TrainingRepoResponse(\n'
        '        github_repo="https://example.invalid/environment.git",\n'
        f'        commit_hash="{UNRELATED}",\n'
        "    ),\n}\n",
        1,
    )
    endpoint = mockroot / "training_repo.py"
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
    assert "ENVIRONMENT must remain absent" in proc.stdout
    assert endpoint.read_text(encoding="utf-8") == before
    assert not list((mockroot / "backups").glob("*.bak"))


def test_manual_rollback_noop_rejects_wrong_repo(
    manifest: dict,
    isolated_release: tuple[Path, Path],
    tmp_path: Path,
):
    _, reviewed = isolated_release
    ready_manifest, readiness = write_ready_release(tmp_path, manifest)
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    before = training_repo_source(
        ROLLBACK, "https://example.invalid/wrong-image.git"
    )
    endpoint = mockroot / "training_repo.py"
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
    assert "IMAGE github_repo is" in proc.stdout
    assert endpoint.read_text(encoding="utf-8") == before
    assert not list((mockroot / "backups").glob("*.bak"))


def test_manual_rollback_rejects_unknown_served_pin_even_with_ready_identity(
    manifest: dict, isolated_release: tuple[Path, Path], tmp_path: Path
):
    _, reviewed = isolated_release
    ready_manifest, readiness = write_ready_release(tmp_path, manifest)
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    before = training_repo_source(UNRELATED)
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

    proc = run_repoint(SELECTED_MANIFEST_PATH, remote, reviewed, mockroot, "--dry-run")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DRY RUN COMPLETE -- nothing was modified" in proc.stdout
    assert "NON-SHIPPABLE" in proc.stdout
    assert endpoint.read_text(encoding="utf-8") == before


def test_forward_dry_run_accepts_only_observed_validator_hotkey_duplicate(
    isolated_release: tuple[Path, Path], tmp_path: Path
):
    remote, reviewed = isolated_release
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    before = training_repo_source(ROLLBACK)
    endpoint.write_text(before, encoding="utf-8")
    (mockroot / "fiber-auth-body.json").write_text(
        fiber_auth_body(),
        encoding="utf-8",
    )

    proc = run_repoint(SELECTED_MANIFEST_PATH, remote, reviewed, mockroot, "--dry-run")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "exact Fiber auth-stage route" in proc.stdout
    assert "DRY RUN COMPLETE -- nothing was modified" in proc.stdout
    assert endpoint.read_text(encoding="utf-8") == before


def test_forward_dry_run_accepts_live_text_first_repo_order_without_git_suffix(
    isolated_release: tuple[Path, Path], tmp_path: Path
):
    remote, reviewed = isolated_release
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    before = training_repo_source(
        ROLLBACK,
        "https://github.com/tuly1/sn56-forge-toolkit",
        text_first=True,
    )
    endpoint.write_text(before, encoding="utf-8")
    (mockroot / "fiber-auth-body.json").write_text(
        fiber_auth_body(),
        encoding="utf-8",
    )

    proc = run_repoint(SELECTED_MANIFEST_PATH, remote, reviewed, mockroot, "--dry-run")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "AST proves exact IMAGE/TEXT mapping" in proc.stdout
    assert "DRY RUN COMPLETE -- nothing was modified" in proc.stdout
    assert endpoint.read_text(encoding="utf-8") == before


@pytest.mark.parametrize(
    ("validator_count", "signature_count"),
    [(1, 1), (2, 2), (3, 1)],
    ids=["obsolete-four", "wrong-header-duplicate", "too-many-validator-duplicates"],
)
def test_forward_dry_run_rejects_every_other_auth_header_multiset(
    isolated_release: tuple[Path, Path],
    tmp_path: Path,
    validator_count: int,
    signature_count: int,
):
    remote, reviewed = isolated_release
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    before = training_repo_source(ROLLBACK)
    endpoint.write_text(before, encoding="utf-8")
    (mockroot / "fiber-auth-body.json").write_text(
        fiber_auth_body(
            validator_count=validator_count,
            signature_count=signature_count,
        ),
        encoding="utf-8",
    )

    proc = run_repoint(SELECTED_MANIFEST_PATH, remote, reviewed, mockroot, "--dry-run")

    assert proc.returncode == 5
    assert "exact Fiber auth stage" in proc.stdout
    assert endpoint.read_text(encoding="utf-8") == before


def test_forward_dry_run_rejects_wrong_served_pin(
    isolated_release: tuple[Path, Path], tmp_path: Path
):
    remote, reviewed = isolated_release
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    before = training_repo_source(UNRELATED)
    endpoint.write_text(before, encoding="utf-8")

    proc = run_repoint(SELECTED_MANIFEST_PATH, remote, reviewed, mockroot, "--dry-run")

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

    proc = run_repoint(SELECTED_MANIFEST_PATH, remote, reviewed, mockroot, "--dry-run")

    assert proc.returncode == 5
    assert "IMAGE github_repo is" in proc.stdout
    assert "expected reviewed repository" in proc.stdout
    assert endpoint.read_text(encoding="utf-8") == before


@pytest.mark.parametrize(
    "service_identity",
    [
        "root\n/home/miner/god\n/home/miner/.venv/bin/uvicorn miner.asgi:app --host 0.0.0.0 --port 7999 --env-file /home/miner/god/.1.env --log-level info\n",
        "miner\n/tmp/wrong-god\n/home/miner/.venv/bin/uvicorn miner.asgi:app --host 0.0.0.0 --port 7999 --env-file /home/miner/god/.1.env --log-level info\n",
        "miner\n/home/miner/god\n/usr/bin/python /tmp/wrong-service.py\n",
    ],
    ids=["wrong-user", "wrong-working-directory", "wrong-exec-start"],
)
def test_forward_dry_run_rejects_wrong_live_service_identity(
    isolated_release: tuple[Path, Path], tmp_path: Path, service_identity: str
):
    remote, reviewed = isolated_release
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    before = training_repo_source(ROLLBACK)
    endpoint.write_text(before, encoding="utf-8")
    (mockroot / "service-contract.out").write_text(service_identity, encoding="utf-8")

    proc = run_repoint(SELECTED_MANIFEST_PATH, remote, reviewed, mockroot, "--dry-run")

    assert proc.returncode == 5
    assert "exact service user/cwd/ExecStart" not in proc.stdout
    assert endpoint.read_text(encoding="utf-8") == before


def test_forward_dry_run_rejects_stale_running_pyc(
    isolated_release: tuple[Path, Path], tmp_path: Path
):
    remote, reviewed = isolated_release
    mockroot = tmp_path / "host"
    endpoint = mockroot / "training_repo.py"
    endpoint.parent.mkdir(parents=True)
    before = training_repo_source(ROLLBACK)
    endpoint.write_text(before, encoding="utf-8")
    pyc = mockroot / "__pycache__" / "training_repo.cpython-312.pyc"
    pyc.parent.mkdir()
    pyc.write_bytes(b"stale-pyc-without-reviewed-pins")

    proc = run_repoint(SELECTED_MANIFEST_PATH, remote, reviewed, mockroot, "--dry-run")

    assert proc.returncode == 5
    assert "running bytecode prestate differs" in proc.stdout
    assert endpoint.read_text(encoding="utf-8") == before


def test_forward_dry_run_rejects_wrong_fiber_auth_stage_body(
    isolated_release: tuple[Path, Path], tmp_path: Path
):
    remote, reviewed = isolated_release
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    endpoint = mockroot / "training_repo.py"
    before = training_repo_source(ROLLBACK)
    endpoint.write_text(before, encoding="utf-8")
    (mockroot / "fiber-auth-body.json").write_text(
        json.dumps({"detail": [{"type": "enum", "loc": ["path", "task_type"]}]}),
        encoding="utf-8",
    )

    proc = run_repoint(SELECTED_MANIFEST_PATH, remote, reviewed, mockroot, "--dry-run")

    assert proc.returncode == 5
    assert "exact Fiber auth stage" in proc.stdout
    assert endpoint.read_text(encoding="utf-8") == before


def test_forward_dry_run_rejects_environment_repo_entry(
    isolated_release: tuple[Path, Path], tmp_path: Path
):
    remote, reviewed = isolated_release
    mockroot = tmp_path / "host"
    mockroot.mkdir()
    before = training_repo_source(ROLLBACK)
    before = before.replace(
        '    TEXT = "text"\n',
        '    TEXT = "text"\n    ENVIRONMENT = "environment"\n',
    ).replace(
        "\n}\n",
        '\n    TournamentType.ENVIRONMENT: TrainingRepoResponse(\n'
        '        github_repo="https://example.invalid/environment.git",\n'
        f'        commit_hash="{UNRELATED}",\n'
        "    ),\n}\n",
        1,
    )
    endpoint = mockroot / "training_repo.py"
    endpoint.write_text(before, encoding="utf-8")

    proc = run_repoint(SELECTED_MANIFEST_PATH, remote, reviewed, mockroot, "--dry-run")

    assert proc.returncode == 5
    assert "ENVIRONMENT must remain absent" in proc.stdout
    assert endpoint.read_text(encoding="utf-8") == before


def test_prepare_readiness_refuses_hold_manifest(tmp_path: Path):
    output = tmp_path / "must-not-exist.json"
    proc = run(
        [
            sys.executable,
            CONTRACT_PATH,
            "--prepare-readiness",
            "--manifest",
            SELECTED_MANIFEST_PATH,
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
        reviewed_readiness, contract_receipt, mock=True
    )
    assert proof["readiness_state"] == "ready"
    assert proof["signature"] == {"state": "mock-bypassed", "live_usable": False}


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
            SELECTED_MANIFEST_PATH,
            reviewed,
            tmp_path / "must-not-exist.json",
            coordinated_policy,
        )
    assert not (tmp_path / "must-not-exist.json").exists()


def test_versioned_policy_is_target_specific_and_regenerates_hold(
    contract, isolated_release: tuple[Path, Path], manifest: dict, tmp_path: Path
):
    _, reviewed = isolated_release
    policy = {
        "schema_version": 2,
        "policy_state": "reviewed",
        "certification_source": copy.deepcopy(manifest["target"]),
        "dockerfiles": copy.deepcopy(manifest["dockerfiles"]),
        "release_evidence": {
            "rollback_commit": manifest["rollback"]["commit"],
            "allowed_changes_name_status_sha256": manifest["allowed_changes"][
                "name_status_sha256"
            ],
            "context_manifest_sha256": "1" * 64,
            "image_digest": "sha256:" + "2" * 64,
            "base_images": [
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
            ],
            "build": {
                "empty_store_log_sha256": "3" * 64,
                "empty_store_seconds": 1680.005,
                "empty_store_result": "timeout-after-verified-step10",
                "continuation_log_sha256": "4" * 64,
                "continuation_seconds": 18.81,
                "combined_seconds": 1698.815,
                "portfolio_classification": "infrastructure-retry-completed",
                "repaired_rebuild_log_sha256": "5" * 64,
                "repaired_rebuild_seconds": 1208.145,
                "repaired_rebuild_result": "pass",
            },
            "parity": {
                "authority": "four-runtime-roots-plus-bounded-os-toolchain",
                "old_rootfs_manifest_sha256": "6" * 64,
                "new_rootfs_manifest_sha256": "7" * 64,
                "config_sha256": "8" * 64,
                "runtime_provenance_sha256": "9" * 64,
                "pyc_payload_manifest_sha256": "a" * 64,
                "git_logical_receipt_sha256": "b" * 64,
                "entrypoint_smoke_sha256": "c" * 64,
            },
        },
    }
    policy["certification_source"] = {
        "commit": manifest["target"]["commit"],
        "tree": manifest["target"]["tree"],
    }
    policy_path = write_manifest(tmp_path, policy, "versioned-policy.json")
    output = tmp_path / "versioned-candidate.json"

    generated = contract.regenerate_manifest(
        MANIFEST_PATH, reviewed, output, policy_path
    )

    assert generated["release_state"] == "hold"
    assert generated["target"]["commit"] == manifest["target"]["commit"]
    assert contract.verify_docker_policy(generated, policy_path)["schema_version"] == 2

    wrong = copy.deepcopy(policy)
    wrong["release_evidence"]["allowed_changes_name_status_sha256"] = "d" * 64
    wrong_path = write_manifest(tmp_path, wrong, "wrong-surface-policy.json")
    with pytest.raises(contract.ContractError, match="changed-surface binding"):
        contract.verify_docker_policy(generated, wrong_path)


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
            SELECTED_MANIFEST_PATH,
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

    receipt = validate(contract, SELECTED_MANIFEST_PATH, remote, reviewed)
    assert receipt["target"]["commit"] == TARGET


def test_receipt_write_is_exclusive(isolated_release: tuple[Path, Path], tmp_path: Path):
    remote, reviewed = isolated_release
    receipt = tmp_path / "receipt.json"
    base = [
        sys.executable,
        CONTRACT_PATH,
        "--manifest",
        SELECTED_MANIFEST_PATH,
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
