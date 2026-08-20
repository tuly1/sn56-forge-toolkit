#!/usr/bin/env python3
"""Fail-closed source contract for an SN56 IMAGE-pin release.

The manifest is the reviewed target authority.  This program deliberately has
no target-SHA override: changing the candidate means regenerating and reviewing
a new HOLD manifest.  Live validation re-proves both pinned refs through a
fresh, credential-free anonymous clone.  Local URL/ref/worktree overrides are
test hooks and are accepted only together with ``--mock``.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable
from urllib.parse import urlparse


SCHEMA_VERSION = 1
DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "release" / "week9-release-manifest.json"
DEFAULT_DOCKER_POLICY = Path(__file__).resolve().parent.parent / "release" / "week9-docker-policy.json"
DEFAULT_READINESS_RECEIPT = Path(__file__).resolve().parent.parent / "release" / "week9-release-readiness.json"

EXPECTED_REPOSITORY_URL = "https://github.com/tuly1/sn56-forge-toolkit.git"
EXPECTED_ROLLBACK = {
    "commit": "75a0a20c2deda82cfa727e082e60a95bea5befb3",
    "tree": "bdf44638853e4cd96f0bd0e420d17bcd519c1d0b",
    "tree_records_sha256": "b158fae4fcf155cf754ce8056df28ae0eae247dd7300f132f29de92fd47006ad",
    "ref": "refs/heads/claude/week8-mse-revert",
}
EXPECTED_DOCKER_PATHS = (
    "ops/docker/standalone-image-toolkit-trainer.dockerfile",
    "ops/docker/standalone-image-trainer.dockerfile",
)
EXPECTED_DOCKER_CERTIFICATION_SOURCE = {
    "commit": "bd852dc0986b661983b70a8e2d225b6da0be971e",
    "tree": "49124005aba5c9fa810814bbcec0c7664726cedf",
}
EXPECTED_DOCKERFILES = [
    {
        "path": EXPECTED_DOCKER_PATHS[0],
        "sha256": "3b98fb1cf2b8ec92218bf30848a0f16c8a3ec48c9034bb5984309d67100663a4",
    },
    {
        "path": EXPECTED_DOCKER_PATHS[1],
        "sha256": "1b009e67e1eb6f87463cac6f9db986f55dccc7fbc70d86c0719d90739332e0d8",
    },
]
EXPECTED_PRODUCTION = {
    "ssh_host": "hetzner",
    "service": "gradients-miner.service",
    "service_user": "miner",
    "service_working_directory": "/home/miner/god",
    "service_exec_start": "/home/miner/.venv/bin/uvicorn miner.asgi:app --host 0.0.0.0 --port 7999 --env-file /home/miner/god/.1.env --log-level info",
    "service_asgi_module": "/home/miner/god/miner/asgi.py",
    "endpoint_host": "65.108.77.230",
    "endpoint_port": 7999,
    "endpoint_route": "/training_repo/image",
    "endpoint_source": "/home/miner/god/miner/endpoints/training_repo.py",
    "endpoint_pyc": "/home/miner/god/miner/endpoints/__pycache__/training_repo.cpython-312.pyc",
    "text_pin": "8f11684e30a556b305dec9dd8eec9794bdae8cde",
}

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FULL_REF = re.compile(r"^refs/heads/[A-Za-z0-9][A-Za-z0-9._/-]*$")
CHANGE_STATUSES = frozenset({"A", "D", "M", "T"})


class ContractError(RuntimeError):
    """A release identity or policy check failed."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def canonical_json_bytes(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, indent=2) + "\n").encode("utf-8")


def _load_json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc
    try:
        data = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ContractError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ContractError(f"{label} root must be an object")
    canonical = canonical_json_bytes(data)
    if raw != canonical:
        raise ContractError(
            f"{label} bytes are not canonical JSON (indent=2, UTF-8, one trailing newline)"
        )
    return data, raw


def load_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    return _load_json_object(path, "manifest")


def _expect_keys(value: Any, keys: Iterable[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContractError(f"{label} keys differ (missing={missing}, extra={extra})")
    return value


def _require_hex(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ContractError(f"{label} must be lowercase {40 if pattern is HEX40 else 64}-hex")
    return value


def _require_ref(value: Any, label: str) -> str:
    if not isinstance(value, str) or not FULL_REF.fullmatch(value) or ".." in value or "//" in value:
        raise ContractError(f"{label} must be an unambiguous full refs/heads/... ref")
    return value


def _require_repo_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise ContractError(f"{label} must be a non-empty relative repository path")
    path = Path(value)
    if any(part in {"", ".", ".."} for part in path.parts) or "\n" in value or "\t" in value:
        raise ContractError(f"{label} is not a canonical relative repository path")
    return value


def canonical_name_status(entries: list[dict[str, str]]) -> bytes:
    return "".join(f"{entry['status']}\t{entry['path']}\n" for entry in entries).encode("utf-8")


def validate_schema(data: dict[str, Any]) -> None:
    _expect_keys(
        data,
        {
            "schema_version",
            "release_state",
            "target",
            "rollback",
            "source",
            "dockerfiles",
            "allowed_changes",
            "production",
        },
        "manifest",
    )
    if data["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"schema_version must be {SCHEMA_VERSION}")
    if data["release_state"] not in {"hold", "ready"}:
        raise ContractError("release_state must be exactly 'hold' or 'ready'")

    for label in ("target", "rollback"):
        item = _expect_keys(
            data[label], {"commit", "tree", "tree_records_sha256", "ref"}, label
        )
        _require_hex(item["commit"], HEX40, f"{label}.commit")
        _require_hex(item["tree"], HEX40, f"{label}.tree")
        _require_hex(item["tree_records_sha256"], HEX64, f"{label}.tree_records_sha256")
        _require_ref(item["ref"], f"{label}.ref")

    if data["target"]["commit"] == data["rollback"]["commit"]:
        raise ContractError("target.commit must differ from rollback.commit")
    if data["target"]["ref"] == data["rollback"]["ref"]:
        raise ContractError("target.ref must differ from rollback.ref")
    if data["rollback"] != EXPECTED_ROLLBACK:
        raise ContractError("rollback identity/ref differs from the exact production rollback contract")

    source = _expect_keys(data["source"], {"repository_url", "reviewed_worktree"}, "source")
    if source["repository_url"] != EXPECTED_REPOSITORY_URL:
        raise ContractError(f"source.repository_url must be {EXPECTED_REPOSITORY_URL}")
    if not isinstance(source["reviewed_worktree"], str) or not Path(source["reviewed_worktree"]).is_absolute():
        raise ContractError("source.reviewed_worktree must be an absolute path")

    dockerfiles = data["dockerfiles"]
    if not isinstance(dockerfiles, list) or len(dockerfiles) != len(EXPECTED_DOCKER_PATHS):
        raise ContractError("dockerfiles must contain exactly the two certified Dockerfiles")
    seen_docker: list[str] = []
    for index, row in enumerate(dockerfiles):
        row = _expect_keys(row, {"path", "sha256"}, f"dockerfiles[{index}]")
        seen_docker.append(_require_repo_path(row["path"], f"dockerfiles[{index}].path"))
        _require_hex(row["sha256"], HEX64, f"dockerfiles[{index}].sha256")
    if tuple(seen_docker) != EXPECTED_DOCKER_PATHS:
        raise ContractError(f"dockerfiles paths/order must be exactly {EXPECTED_DOCKER_PATHS}")

    allowed = _expect_keys(
        data["allowed_changes"], {"base_commit", "name_status_sha256", "entries"}, "allowed_changes"
    )
    _require_hex(allowed["base_commit"], HEX40, "allowed_changes.base_commit")
    _require_hex(allowed["name_status_sha256"], HEX64, "allowed_changes.name_status_sha256")
    if allowed["base_commit"] != data["rollback"]["commit"]:
        raise ContractError("allowed_changes.base_commit must equal rollback.commit")
    if not isinstance(allowed["entries"], list) or not allowed["entries"]:
        raise ContractError("allowed_changes.entries must be a non-empty list")
    paths: list[str] = []
    for index, entry in enumerate(allowed["entries"]):
        entry = _expect_keys(entry, {"status", "path"}, f"allowed_changes.entries[{index}]")
        if entry["status"] not in CHANGE_STATUSES:
            raise ContractError(f"unsupported name-status code at allowed_changes.entries[{index}]")
        paths.append(_require_repo_path(entry["path"], f"allowed_changes.entries[{index}].path"))
    if len(paths) != len(set(paths)):
        raise ContractError("allowed_changes.entries contains duplicate paths")
    embedded_digest = hashlib.sha256(canonical_name_status(allowed["entries"])).hexdigest()
    if embedded_digest != allowed["name_status_sha256"]:
        raise ContractError(
            "allowed_changes entries do not hash to allowed_changes.name_status_sha256 "
            f"({embedded_digest} != {allowed['name_status_sha256']})"
        )
    missing_docker = sorted(set(EXPECTED_DOCKER_PATHS) - set(paths))
    if missing_docker:
        raise ContractError(f"allowed production delta omits Dockerfiles: {missing_docker}")

    production = _expect_keys(data["production"], EXPECTED_PRODUCTION, "production")
    if production != EXPECTED_PRODUCTION:
        diffs = {
            key: (production.get(key), expected)
            for key, expected in EXPECTED_PRODUCTION.items()
            if production.get(key) != expected
        }
        raise ContractError(f"production literals differ from fixed endpoint contract: {diffs}")


def load_docker_policy(path: Path) -> tuple[dict[str, Any], bytes]:
    policy, raw = _load_json_object(path.resolve(), "Docker policy")
    _expect_keys(
        policy,
        {"schema_version", "policy_state", "certification_source", "dockerfiles"},
        "Docker policy",
    )
    if policy["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"Docker policy schema_version must be {SCHEMA_VERSION}")
    if policy["policy_state"] != "reviewed":
        raise ContractError("Docker policy policy_state must be exactly 'reviewed'")
    certification_source = _expect_keys(
        policy["certification_source"],
        {"commit", "tree"},
        "Docker policy certification_source",
    )
    _require_hex(
        certification_source["commit"],
        HEX40,
        "Docker policy certification_source.commit",
    )
    _require_hex(
        certification_source["tree"],
        HEX40,
        "Docker policy certification_source.tree",
    )
    dockerfiles = policy["dockerfiles"]
    if not isinstance(dockerfiles, list) or len(dockerfiles) != len(EXPECTED_DOCKER_PATHS):
        raise ContractError("Docker policy must contain exactly the two certified Dockerfiles")
    seen: list[str] = []
    for index, row in enumerate(dockerfiles):
        row = _expect_keys(row, {"path", "sha256"}, f"Docker policy dockerfiles[{index}]")
        seen.append(_require_repo_path(row["path"], f"Docker policy dockerfiles[{index}].path"))
        _require_hex(row["sha256"], HEX64, f"Docker policy dockerfiles[{index}].sha256")
    if tuple(seen) != EXPECTED_DOCKER_PATHS:
        raise ContractError(f"Docker policy paths/order must be exactly {EXPECTED_DOCKER_PATHS}")
    if policy["certification_source"] != EXPECTED_DOCKER_CERTIFICATION_SOURCE:
        raise ContractError(
            "Docker policy must stay anchored to the exact bd852dc audited fixture"
        )
    if policy["dockerfiles"] != EXPECTED_DOCKERFILES:
        raise ContractError(
            "Docker policy hashes are immutable for this release; Docker-byte changes are forbidden"
        )
    return policy, raw


def verify_docker_policy(
    manifest: dict[str, Any], docker_policy_path: Path
) -> dict[str, Any]:
    policy, raw = load_docker_policy(docker_policy_path)
    if policy["dockerfiles"] != manifest["dockerfiles"]:
        raise ContractError(
            "manifest Docker hashes differ from the independent reviewed Docker policy"
        )
    return {
        "path": str(docker_policy_path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "policy_state": policy["policy_state"],
        "certification_source": copy.deepcopy(policy["certification_source"]),
        "dockerfiles": copy.deepcopy(policy["dockerfiles"]),
    }


def _validate_readiness_against_contract(
    readiness_path: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    readiness, readiness_raw = _load_json_object(readiness_path.resolve(), "readiness receipt")
    _expect_keys(
        readiness,
        {
            "schema_version",
            "readiness_state",
            "manifest_sha256",
            "docker_policy_sha256",
            "target",
            "rollback",
            "allowed_changes",
            "dockerfiles",
        },
        "readiness receipt",
    )
    if readiness["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"readiness receipt schema_version must be {SCHEMA_VERSION}")
    if readiness["readiness_state"] not in {"hold", "ready"}:
        raise ContractError("readiness_state must be exactly 'hold' or 'ready'")
    _require_hex(readiness["manifest_sha256"], HEX64, "readiness manifest_sha256")
    _require_hex(readiness["docker_policy_sha256"], HEX64, "readiness docker_policy_sha256")

    required_contract_keys = {
        "manifest_sha256",
        "release_state",
        "target",
        "rollback",
        "allowed_changes",
        "dockerfiles",
        "docker_policy",
    }
    missing = sorted(required_contract_keys - set(contract))
    if missing:
        raise ContractError(f"validated contract receipt is missing readiness fields: {missing}")
    expected = {
        "manifest_sha256": contract["manifest_sha256"],
        "docker_policy_sha256": contract["docker_policy"]["sha256"],
        "target": contract["target"],
        "rollback": contract["rollback"],
        "allowed_changes": contract["allowed_changes"],
        "dockerfiles": contract["dockerfiles"],
    }
    for key, value in expected.items():
        if readiness[key] != value:
            raise ContractError(
                f"readiness receipt {key} does not bind the exact validated contract"
            )
    if contract["release_state"] != "ready":
        raise ContractError("validated manifest release_state is not ready")
    if readiness["readiness_state"] != "ready":
        raise ContractError("independent readiness receipt state is not ready")
    return {
        "path": str(readiness_path.resolve()),
        "sha256": hashlib.sha256(readiness_raw).hexdigest(),
        "readiness_state": "ready",
        **expected,
    }


def validate_readiness_receipt(
    readiness_path: Path, validated_contract_receipt_path: Path
) -> dict[str, Any]:
    contract, _ = _load_json_object(
        validated_contract_receipt_path.resolve(), "validated contract receipt"
    )
    return _validate_readiness_against_contract(readiness_path, contract)


def _git_env(*, anonymous: bool = False, home: Path | None = None) -> dict[str, str]:
    if anonymous:
        if home is None:
            raise AssertionError("anonymous Git calls require an isolated HOME")
        # Start from an allowlist, not os.environ.copy(). In particular this
        # excludes GIT_CONFIG_COUNT/GIT_CONFIG_KEY_*, credential/token variables,
        # SSH_AUTH_SOCK, XDG config, proxy credentials, and Git object/index
        # redirections. The claim here is genuinely zero credentials in scope.
        env = {
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": str(home),
            "LC_ALL": "C",
            "LANG": "C",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": shutil.which("false") or "/usr/bin/false",
            "GIT_SSH_COMMAND": shutil.which("false") or "/usr/bin/false",
        }
        for optional in ("TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_CA_BUNDLE"):
            if optional in os.environ:
                env[optional] = os.environ[optional]
        return env

    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    anonymous: bool = False,
    home: Path | None = None,
) -> bytes:
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            env=_git_env(anonymous=anonymous, home=home),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise ContractError(f"could not execute {args[0]!r}: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise ContractError(f"command failed ({proc.returncode}): {' '.join(args[:4])}: {detail}")
    return proc.stdout


def _git(repo: Path, *args: str) -> bytes:
    return _run(["git", "-C", str(repo), *args])


def _git_text(repo: Path, *args: str) -> str:
    return _git(repo, *args).decode("utf-8", "strict").strip()


def tree_records(repo: Path, commit: str) -> tuple[bytes, str]:
    output = _git(repo, "ls-tree", "-r", "--full-tree", commit)
    lines = output.splitlines()
    try:
        lines.sort(key=lambda line: line.split(b"\t", 1)[1])
    except IndexError as exc:
        raise ContractError(f"malformed git ls-tree record at {commit}") from exc
    canonical = b"\n".join(lines) + (b"\n" if lines else b"")
    return canonical, hashlib.sha256(canonical).hexdigest()


def name_status(repo: Path, base: str, target: str) -> tuple[list[dict[str, str]], bytes, str]:
    output = _git(
        repo,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--name-status",
        "--no-renames",
        base,
        target,
        "--",
    )
    entries: list[dict[str, str]] = []
    for raw_line in output.splitlines():
        try:
            status_raw, path_raw = raw_line.split(b"\t", 1)
            status = status_raw.decode("ascii", "strict")
            path = path_raw.decode("utf-8", "strict")
        except (ValueError, UnicodeError) as exc:
            raise ContractError(f"malformed git diff --name-status record: {raw_line!r}") from exc
        if status not in CHANGE_STATUSES:
            raise ContractError(f"unsupported git name-status {status!r} for {path!r}")
        _require_repo_path(path, "git diff path")
        entries.append({"status": status, "path": path})
    canonical = canonical_name_status(entries)
    if output != canonical:
        raise ContractError("git name-status output is not canonical")
    return entries, canonical, hashlib.sha256(canonical).hexdigest()


def object_file(repo: Path, commit: str, path: str) -> bytes:
    spec = f"{commit}:{path}"
    object_type = _git_text(repo, "cat-file", "-t", spec)
    if object_type != "blob":
        raise ContractError(f"{spec} is {object_type!r}, expected blob")
    return _git(repo, "cat-file", "blob", spec)


def ensure_repo_root(path: Path) -> Path:
    resolved = path.resolve()
    top = Path(_git_text(resolved, "rev-parse", "--show-toplevel")).resolve()
    if top != resolved:
        raise ContractError(f"reviewed_worktree must name the repository root ({resolved} != {top})")
    return resolved


def verify_repo_objects(
    repo: Path,
    manifest: dict[str, Any],
    *,
    check_worktree: bool,
    docker_policy: dict[str, Any],
) -> dict[str, Any]:
    target = manifest["target"]
    rollback = manifest["rollback"]

    for label, item in (("target", target), ("rollback", rollback)):
        object_type = _git_text(repo, "cat-file", "-t", item["commit"])
        if object_type != "commit":
            raise ContractError(f"{label}.commit is {object_type!r}, expected commit")
        actual_tree = _git_text(repo, "rev-parse", f"{item['commit']}^{{tree}}")
        if actual_tree != item["tree"]:
            raise ContractError(f"{label}.tree mismatch ({actual_tree} != {item['tree']})")
        _, records_digest = tree_records(repo, item["commit"])
        if records_digest != item["tree_records_sha256"]:
            raise ContractError(
                f"{label}.tree_records_sha256 mismatch "
                f"({records_digest} != {item['tree_records_sha256']})"
            )

    actual_entries, _, actual_surface_digest = name_status(
        repo, manifest["allowed_changes"]["base_commit"], target["commit"]
    )
    if actual_entries != manifest["allowed_changes"]["entries"]:
        raise ContractError("rollback-to-target changed surface differs from allowed_changes.entries")
    if actual_surface_digest != manifest["allowed_changes"]["name_status_sha256"]:
        raise ContractError(
            "rollback-to-target name-status digest differs "
            f"({actual_surface_digest} != {manifest['allowed_changes']['name_status_sha256']})"
        )

    verified_docker: list[dict[str, str]] = []
    for row in manifest["dockerfiles"]:
        raw = object_file(repo, target["commit"], row["path"])
        actual = hashlib.sha256(raw).hexdigest()
        if actual != row["sha256"]:
            raise ContractError(
                f"certified Docker blob hash mismatch for {row['path']} ({actual} != {row['sha256']})"
            )
        if check_worktree:
            worktree_path = repo / row["path"]
            try:
                worktree_actual = hashlib.sha256(worktree_path.read_bytes()).hexdigest()
            except OSError as exc:
                raise ContractError(f"cannot read working Dockerfile {worktree_path}: {exc}") from exc
            if worktree_actual != row["sha256"]:
                raise ContractError(
                    f"working Dockerfile hash mismatch for {row['path']} "
                    f"({worktree_actual} != {row['sha256']})"
                )
        verified_docker.append({"path": row["path"], "sha256": actual})

    # The immutable hashes are not trusted merely because they appear in a
    # policy file. Re-prove them from the audited bd852dc certification source
    # in every repository used by the contract, then require the candidate
    # target blobs above to be byte-identical to that independently proven base.
    certification_source = docker_policy["certification_source"]
    source_commit = certification_source["commit"]
    source_type = _git_text(repo, "cat-file", "-t", source_commit)
    if source_type != "commit":
        raise ContractError(
            f"Docker certification source is {source_type!r}, expected commit"
        )
    source_tree = _git_text(repo, "rev-parse", f"{source_commit}^{{tree}}")
    if source_tree != certification_source["tree"]:
        raise ContractError(
            "Docker certification-source tree mismatch "
            f"({source_tree} != {certification_source['tree']})"
        )
    certified_docker: list[dict[str, str]] = []
    for row in docker_policy["dockerfiles"]:
        certified = hashlib.sha256(
            object_file(repo, source_commit, row["path"])
        ).hexdigest()
        if certified != row["sha256"]:
            raise ContractError(
                f"Docker certification-source blob mismatch for {row['path']} "
                f"({certified} != {row['sha256']})"
            )
        certified_docker.append({"path": row["path"], "sha256": certified})

    return {
        "target_tree": target["tree"],
        "target_tree_records_sha256": target["tree_records_sha256"],
        "rollback_tree": rollback["tree"],
        "rollback_tree_records_sha256": rollback["tree_records_sha256"],
        "allowed_changes_name_status_sha256": actual_surface_digest,
        "allowed_changes_count": len(actual_entries),
        "dockerfiles": verified_docker,
        "docker_certification_source": {
            "commit": source_commit,
            "tree": source_tree,
            "dockerfiles": certified_docker,
        },
    }


def verify_clean_worktree(repo: Path, label: str) -> None:
    # Porcelain status deliberately trusts assume-unchanged and skip-worktree
    # bits. Those flags can therefore hide arbitrary tracked byte changes from
    # a nominally clean checkout. A release oracle or manifest generator may not carry
    # either optimization, even when the currently hidden file happens to be
    # unchanged.
    flagged: list[tuple[str, str]] = []
    for raw in _git(repo, "ls-files", "-v", "-z").split(b"\0"):
        if not raw:
            continue
        if len(raw) < 3 or raw[1:2] != b" ":
            raise ContractError(f"malformed git ls-files -v record: {raw!r}")
        try:
            tag = chr(raw[0])
            tracked_path = raw[2:].decode("utf-8", "strict")
        except UnicodeError as exc:
            raise ContractError(f"non-UTF-8 tracked path in git ls-files -v: {raw!r}") from exc
        if tag == "S" or tag.islower():
            flagged.append((tag, tracked_path))
    if flagged:
        tag, tracked_path = flagged[0]
        raise ContractError(
            f"{label} worktree has a forbidden assume-unchanged/skip-worktree "
            f"index flag ({tag}) on {tracked_path}; clean-tree proof is incomplete"
        )
    status = _git_text(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        first = status.splitlines()[0]
        raise ContractError(f"{label} worktree is dirty (first entry: {first})")


def verify_local_worktree(
    path: Path, manifest: dict[str, Any], docker_policy: dict[str, Any]
) -> dict[str, Any]:
    repo = ensure_repo_root(path)
    verify_clean_worktree(repo, "reviewed")
    head = _git_text(repo, "rev-parse", "HEAD")
    if head != manifest["target"]["commit"]:
        raise ContractError(f"reviewed worktree HEAD is {head}, expected {manifest['target']['commit']}")
    verified = verify_repo_objects(
        repo, manifest, check_worktree=True, docker_policy=docker_policy
    )
    verified["reviewed_worktree"] = str(repo)
    verified["head"] = head
    return verified


def _is_local_url(value: str) -> bool:
    parsed = urlparse(value)
    if parsed.scheme == "file":
        return True
    return not parsed.scheme and Path(value).is_absolute()


def effective_inputs(
    manifest: dict[str, Any],
    *,
    mock: bool,
    repository_url: str | None,
    reviewed_worktree: str | None,
    target_ref: str | None,
    rollback_ref: str | None,
) -> dict[str, str]:
    overrides = {
        "repository_url": repository_url,
        "reviewed_worktree": reviewed_worktree,
        "target_ref": target_ref,
        "rollback_ref": rollback_ref,
    }
    supplied = [key for key, value in overrides.items() if value is not None]
    if supplied and not mock:
        raise ContractError(f"local contract overrides require explicit --mock: {supplied}")
    if mock:
        if repository_url is None or reviewed_worktree is None:
            raise ContractError("--mock requires explicit --repository-url and --reviewed-worktree")
        if not _is_local_url(repository_url):
            raise ContractError("--mock --repository-url must be an absolute local path or file:// URL")

    effective = {
        "repository_url": repository_url or manifest["source"]["repository_url"],
        "reviewed_worktree": reviewed_worktree or manifest["source"]["reviewed_worktree"],
        "target_ref": target_ref or manifest["target"]["ref"],
        "rollback_ref": rollback_ref or manifest["rollback"]["ref"],
    }
    _require_ref(effective["target_ref"], "effective target ref")
    _require_ref(effective["rollback_ref"], "effective rollback ref")
    if effective["target_ref"] == effective["rollback_ref"]:
        raise ContractError("effective target and rollback refs must differ")
    if not Path(effective["reviewed_worktree"]).is_absolute():
        raise ContractError("effective reviewed worktree must be an absolute path")
    return effective


def verify_anonymous_remote(
    manifest: dict[str, Any],
    effective: dict[str, str],
    docker_policy: dict[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sn56-release-contract-") as tmp:
        root = Path(tmp)
        home = root / "home"
        home.mkdir()
        clone = root / "clone"
        refs_output = _run(
            [
                "git",
                "-c",
                "credential.helper=",
                "ls-remote",
                "--refs",
                effective["repository_url"],
                effective["target_ref"],
                effective["rollback_ref"],
            ],
            anonymous=True,
            home=home,
        ).decode("utf-8", "strict")
        found: dict[str, str] = {}
        for line in refs_output.splitlines():
            try:
                sha, ref = line.split("\t", 1)
            except ValueError as exc:
                raise ContractError(f"malformed git ls-remote output: {line!r}") from exc
            if ref in found:
                raise ContractError(f"anonymous remote returned duplicate ref {ref}")
            found[ref] = sha
        expected_refs = {
            effective["target_ref"]: manifest["target"]["commit"],
            effective["rollback_ref"]: manifest["rollback"]["commit"],
        }
        if found != expected_refs:
            raise ContractError(f"anonymous remote refs differ (found={found}, expected={expected_refs})")

        _run(
            [
                "git",
                "-c",
                "credential.helper=",
                "clone",
                "--quiet",
                "--no-checkout",
                effective["repository_url"],
                str(clone),
            ],
            anonymous=True,
            home=home,
        )
        for effective_ref, expected_commit in expected_refs.items():
            short_name = effective_ref.removeprefix("refs/heads/")
            tracking_ref = f"refs/remotes/origin/{short_name}"
            cloned_ref = _git_text(clone, "rev-parse", "--verify", tracking_ref)
            if cloned_ref != expected_commit:
                raise ContractError(
                    f"fresh anonymous clone stored stale {effective_ref} "
                    f"({cloned_ref} != {expected_commit})"
                )
        verified = verify_repo_objects(
            clone, manifest, check_worktree=False, docker_policy=docker_policy
        )
        verified["repository_url"] = effective["repository_url"]
        verified["target_ref"] = effective["target_ref"]
        verified["rollback_ref"] = effective["rollback_ref"]
        return verified


def validate_contract(
    manifest_path: Path,
    *,
    docker_policy_path: Path = DEFAULT_DOCKER_POLICY,
    mock: bool = False,
    repository_url: str | None = None,
    reviewed_worktree: str | None = None,
    target_ref: str | None = None,
    rollback_ref: str | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest, raw = load_manifest(manifest_path)
    validate_schema(manifest)
    docker_policy = verify_docker_policy(manifest, docker_policy_path)
    effective = effective_inputs(
        manifest,
        mock=mock,
        repository_url=repository_url,
        reviewed_worktree=reviewed_worktree,
        target_ref=target_ref,
        rollback_ref=rollback_ref,
    )
    local = verify_local_worktree(
        Path(effective["reviewed_worktree"]), manifest, docker_policy
    )
    remote = verify_anonymous_remote(manifest, effective, docker_policy)

    return {
        "schema_version": SCHEMA_VERSION,
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "release_state": manifest["release_state"],
        "mock": mock,
        "target": copy.deepcopy(manifest["target"]),
        "rollback": copy.deepcopy(manifest["rollback"]),
        "source": copy.deepcopy(manifest["source"]),
        "production": copy.deepcopy(manifest["production"]),
        "allowed_changes": {
            "base_commit": manifest["allowed_changes"]["base_commit"],
            "name_status_sha256": manifest["allowed_changes"]["name_status_sha256"],
            "count": len(manifest["allowed_changes"]["entries"]),
        },
        "dockerfiles": copy.deepcopy(manifest["dockerfiles"]),
        "docker_policy": docker_policy,
        "effective": effective,
        "verified_local": local,
        "verified_anonymous_remote": remote,
    }


def validate_rollback_contract(
    manifest_path: Path,
    readiness_path: Path,
    *,
    docker_policy_path: Path = DEFAULT_DOCKER_POLICY,
    mock: bool = False,
) -> dict[str, Any]:
    """Validate reviewed rollback identity without forward network/worktree gates.

    Emergency rollback must remain available if the public ref is unavailable or
    the successor worktree has become dirty after release. Its authority is the
    canonical READY manifest plus the independently reviewed readiness receipt;
    the repoint script separately requires the host to serve that exact target
    pin before it can restore the hardcoded rollback pin.
    """

    manifest_path = manifest_path.resolve()
    manifest, raw = load_manifest(manifest_path)
    validate_schema(manifest)
    docker_policy = verify_docker_policy(manifest, docker_policy_path)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "validation_scope": "emergency-rollback",
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "release_state": manifest["release_state"],
        "mock": mock,
        "target": copy.deepcopy(manifest["target"]),
        "rollback": copy.deepcopy(manifest["rollback"]),
        "source": copy.deepcopy(manifest["source"]),
        "production": copy.deepcopy(manifest["production"]),
        "allowed_changes": {
            "base_commit": manifest["allowed_changes"]["base_commit"],
            "name_status_sha256": manifest["allowed_changes"]["name_status_sha256"],
            "count": len(manifest["allowed_changes"]["entries"]),
        },
        "dockerfiles": copy.deepcopy(manifest["dockerfiles"]),
        "docker_policy": docker_policy,
    }
    readiness = _validate_readiness_against_contract(readiness_path, receipt)
    receipt["readiness"] = readiness
    return receipt


def prepare_readiness_receipt(
    manifest_path: Path,
    output_path: Path,
    *,
    docker_policy_path: Path = DEFAULT_DOCKER_POLICY,
) -> dict[str, Any]:
    """Derive a reviewable HOLD receipt from one exact canonical READY manifest."""

    manifest_path = manifest_path.resolve()
    manifest, manifest_raw = load_manifest(manifest_path)
    validate_schema(manifest)
    if manifest["release_state"] != "ready":
        raise ContractError(
            "readiness preparation requires an exact canonical manifest with release_state ready"
        )
    docker_policy = verify_docker_policy(manifest, docker_policy_path)
    readiness = {
        "schema_version": SCHEMA_VERSION,
        "readiness_state": "hold",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "docker_policy_sha256": docker_policy["sha256"],
        "target": copy.deepcopy(manifest["target"]),
        "rollback": copy.deepcopy(manifest["rollback"]),
        "allowed_changes": {
            "base_commit": manifest["allowed_changes"]["base_commit"],
            "name_status_sha256": manifest["allowed_changes"]["name_status_sha256"],
            "count": len(manifest["allowed_changes"]["entries"]),
        },
        "dockerfiles": copy.deepcopy(manifest["dockerfiles"]),
    }
    output_path = output_path.resolve()
    if output_path.exists():
        raise ContractError(
            f"readiness output already exists; refusing to overwrite: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("xb") as handle:
            handle.write(canonical_json_bytes(readiness))
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ContractError(
            f"could not write prepared readiness receipt {output_path}: {exc}"
        ) from exc
    return readiness


def regenerate_manifest(
    template_path: Path,
    candidate_worktree: Path,
    output_path: Path,
    docker_policy_path: Path = DEFAULT_DOCKER_POLICY,
) -> dict[str, Any]:
    template, _ = load_manifest(template_path.resolve())
    validate_schema(template)
    candidate = ensure_repo_root(candidate_worktree)
    verify_clean_worktree(candidate, "candidate")
    target_commit = _git_text(candidate, "rev-parse", "HEAD")
    if target_commit == template["rollback"]["commit"]:
        raise ContractError("candidate HEAD equals rollback.commit; refusing a no-op release manifest")
    target_tree = _git_text(candidate, "rev-parse", "HEAD^{tree}")
    _, target_records_digest = tree_records(candidate, target_commit)
    docker_policy, _ = load_docker_policy(docker_policy_path)

    # Re-prove that the template's immutable rollback object is available and exact.
    for key in ("tree", "tree_records_sha256"):
        if key == "tree":
            actual = _git_text(candidate, "rev-parse", f"{template['rollback']['commit']}^{{tree}}")
        else:
            _, actual = tree_records(candidate, template["rollback"]["commit"])
        if actual != template["rollback"][key]:
            raise ContractError(f"candidate object store has wrong rollback {key}: {actual}")

    entries, _, surface_digest = name_status(candidate, template["rollback"]["commit"], target_commit)
    for reviewed in docker_policy["dockerfiles"]:
        path = reviewed["path"]
        object_bytes = object_file(candidate, target_commit, path)
        object_digest = hashlib.sha256(object_bytes).hexdigest()
        working_digest = hashlib.sha256((candidate / path).read_bytes()).hexdigest()
        if working_digest != object_digest:
            raise ContractError(f"candidate working Dockerfile differs from HEAD: {path}")
        if object_digest != reviewed["sha256"]:
            raise ContractError(
                f"candidate Docker hash was not explicitly reviewed by policy for {path} "
                f"({object_digest} != {reviewed['sha256']})"
            )

    generated = copy.deepcopy(template)
    generated["release_state"] = "hold"
    generated["target"] = {
        "commit": target_commit,
        "tree": target_tree,
        "tree_records_sha256": target_records_digest,
        "ref": template["target"]["ref"],
    }
    generated["source"]["reviewed_worktree"] = str(candidate)
    generated["dockerfiles"] = copy.deepcopy(docker_policy["dockerfiles"])
    generated["allowed_changes"] = {
        "base_commit": template["rollback"]["commit"],
        "name_status_sha256": surface_digest,
        "entries": entries,
    }
    validate_schema(generated)
    verify_docker_policy(generated, docker_policy_path)

    output_path = output_path.resolve()
    if output_path.exists():
        raise ContractError(f"output already exists; refusing to overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(generated, indent=2) + "\n").encode("utf-8")
    try:
        with output_path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ContractError(f"could not write regenerated manifest {output_path}: {exc}") from exc
    return generated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--docker-policy", type=Path, default=DEFAULT_DOCKER_POLICY)
    parser.add_argument("--contract-only", action="store_true", help="validate and exit (the default action)")
    parser.add_argument("--receipt", type=Path, help="write a JSON validation receipt; no receipt is written by default")
    parser.add_argument("--mock", action="store_true", help="enable local-only test overrides")
    parser.add_argument("--repository-url", help="mock-only local anonymous remote")
    parser.add_argument("--reviewed-worktree", help="mock-only exact candidate checkout")
    parser.add_argument("--target-ref", help="mock-only target full ref")
    parser.add_argument("--rollback-ref", help="mock-only rollback full ref")
    parser.add_argument("--regenerate-from", type=Path, metavar="MANIFEST")
    parser.add_argument("--prepare-readiness", action="store_true")
    parser.add_argument("--candidate-worktree", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--readiness-only", action="store_true")
    parser.add_argument("--rollback-contract-only", action="store_true")
    parser.add_argument("--readiness-receipt", type=Path)
    parser.add_argument("--validated-contract-receipt", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.readiness_only:
            if args.readiness_receipt is None or args.validated_contract_receipt is None:
                raise ContractError(
                    "--readiness-only requires --readiness-receipt and --validated-contract-receipt"
                )
            if (
                args.rollback_contract_only
                or args.prepare_readiness
                or args.regenerate_from
                or args.candidate_worktree
                or args.output
                or args.receipt
            ):
                raise ContractError("readiness-only cannot be combined with validation/regeneration output")
            readiness = validate_readiness_receipt(
                args.readiness_receipt, args.validated_contract_receipt
            )
            print("READINESS PASS")
            print(f"  receipt: {readiness['path']}")
            print(f"  sha256:  {readiness['sha256']}")
            print("  state:   ready (interactive operator authorization is still required)")
            return 0
        if args.rollback_contract_only:
            if args.readiness_receipt is None:
                raise ContractError(
                    "--rollback-contract-only requires --readiness-receipt"
                )
            if args.validated_contract_receipt is not None:
                raise ContractError(
                    "--validated-contract-receipt is only valid with --readiness-only"
                )
            if (
                args.prepare_readiness
                or args.regenerate_from
                or args.candidate_worktree
                or args.output
                or args.repository_url
                or args.reviewed_worktree
                or args.target_ref
                or args.rollback_ref
            ):
                raise ContractError(
                    "forward Git/worktree and regeneration options are forbidden in rollback-only validation"
                )
            receipt = validate_rollback_contract(
                args.manifest,
                args.readiness_receipt,
                docker_policy_path=args.docker_policy,
                mock=args.mock,
            )
            if args.receipt:
                destination = args.receipt.resolve()
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with destination.open("x", encoding="utf-8") as handle:
                        json.dump(receipt, handle, indent=2)
                        handle.write("\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                except FileExistsError as exc:
                    raise ContractError(
                        f"receipt already exists; refusing to overwrite: {destination}"
                    ) from exc
                print(f"  receipt: {destination}")
            print("ROLLBACK CONTRACT PASS")
            print(f"  released target: {receipt['target']['commit']}")
            print(f"  rollback target: {receipt['rollback']['commit']}")
            print("  scope: offline emergency rollback identity (host pin gate remains)")
            return 0
        if args.prepare_readiness:
            if args.output is None:
                raise ContractError("--prepare-readiness requires --output")
            if (
                args.contract_only
                or args.receipt
                or args.mock
                or args.repository_url
                or args.reviewed_worktree
                or args.target_ref
                or args.rollback_ref
                or args.regenerate_from
                or args.candidate_worktree
                or args.readiness_receipt
                or args.validated_contract_receipt
            ):
                raise ContractError(
                    "validation, mock, regeneration, and readiness-input options cannot be combined with --prepare-readiness"
                )
            readiness = prepare_readiness_receipt(
                args.manifest,
                args.output,
                docker_policy_path=args.docker_policy,
            )
            print(f"PREPARED HOLD readiness receipt: {args.output.resolve()}")
            print(f"  manifest sha256: {readiness['manifest_sha256']}")
            print(f"  target:          {readiness['target']['commit']}")
            print("  state:           hold (NON-SHIPPABLE; independent review required)")
            return 0
        if args.readiness_receipt is not None or args.validated_contract_receipt is not None:
            raise ContractError(
                "--readiness-receipt/--validated-contract-receipt require --readiness-only"
            )
        if args.regenerate_from is not None:
            if args.candidate_worktree is None or args.output is None:
                raise ContractError("--regenerate-from requires --candidate-worktree and --output")
            if args.receipt or args.mock or args.repository_url or args.reviewed_worktree or args.target_ref or args.rollback_ref:
                raise ContractError("validation/mock options cannot be combined with --regenerate-from")
            generated = regenerate_manifest(
                args.regenerate_from,
                args.candidate_worktree,
                args.output,
                args.docker_policy,
            )
            print(f"REGENERATED HOLD manifest: {args.output.resolve()}")
            print(f"  target: {generated['target']['commit']}")
            print(f"  tree:   {generated['target']['tree']}")
            print("  state:  hold (NON-SHIPPABLE; review before changing to ready)")
            return 0
        if args.candidate_worktree is not None or args.output is not None:
            raise ContractError("--candidate-worktree/--output require --regenerate-from")

        receipt = validate_contract(
            args.manifest,
            docker_policy_path=args.docker_policy,
            mock=args.mock,
            repository_url=args.repository_url,
            reviewed_worktree=args.reviewed_worktree,
            target_ref=args.target_ref,
            rollback_ref=args.rollback_ref,
        )
        if args.receipt:
            destination = args.receipt.resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with destination.open("x", encoding="utf-8") as handle:
                    json.dump(receipt, handle, indent=2)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError as exc:
                raise ContractError(f"receipt already exists; refusing to overwrite: {destination}") from exc
            print(f"  receipt: {destination}")
        print("CONTRACT PASS")
        print(f"  target:   {receipt['target']['commit']}")
        print(f"  rollback: {receipt['rollback']['commit']}")
        print(f"  surface:  {receipt['allowed_changes']['name_status_sha256']} ({receipt['allowed_changes']['count']} paths)")
        if receipt["release_state"] == "hold":
            print("  state:    hold — NON-SHIPPABLE; mutation is forbidden")
        else:
            print("  state:    ready — validation is not deployment authorization")
        return 0
    except ContractError as exc:
        print(f"CONTRACT FAIL: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
