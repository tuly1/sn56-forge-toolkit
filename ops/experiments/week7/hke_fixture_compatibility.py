#!/usr/bin/env python3
"""Bridge immutable HKE fixture authority to a new execution tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forge import krea_runtime  # noqa: E402
from forge.file_evidence import read_regular_bytes  # noqa: E402
from ops.experiments.week7 import hke_fixture_admission as admission  # noqa: E402

SCHEMA = 3
KIND = "sn56-week7-hke-fixture-execution-compatibility"
FIXTURE_PATHS = (
    "ops/experiments/week7/hke_procedural_renderer.py",
    "ops/experiments/week7/hke_fixture_admission.py",
    "ops/experiments/week7/hke_fixture_contract.json",
)
EXECUTION_PATHS = (
    "ops/experiments/week7/run_hke_factorial.py",
    "ops/experiments/week7/hke_fixture_compatibility.py",
    "forge/krea_runtime.py",
    "forge/tasks/aitoolkit.py",
)
BOUND_PATHS = (*FIXTURE_PATHS, *EXECUTION_PATHS)
MAX_RECORD_BYTES = 4 * 1024 * 1024
SOURCE_RECORDS = {
    "admission_set": {
        "bytes": 47_726,
        "file_sha256": "0a0e60c911f964a0033c9fc655ee6d68dac5dc0ab6d7718505c6cdfe47c271b4",
        "admission_set_sha256": "44ee883468eabb657e2b6e2a0b20133eee7cd7ae7b70a44de89947e13c94071d",
    },
    "sealed_human_review": {
        "bytes": 178_223,
        "file_sha256": "991a06d6ac3e046bef1f096f9c1c698bed6618785ad54cb34216e56e1a8a2aa2",
        "review_sha256": "2c0e7754f0a552660f3924242c90e55856922aa749b43cfb5cc790027806b0e2",
    },
    "owner_ratification": {
        "bytes": 1_961,
        "file_sha256": "bfb3265277a56f9a52134f26f137ad220a028d5ad1c912d39b4487370c8e2e6d",
        "owner_ratification_sha256": "dcc613a1f56366ed9a271d5919a4bbcfd1eeed77054b4e742ec42f3bcd17724e",
    },
}


class CompatibilityError(RuntimeError):
    """The old fixture authority cannot safely govern this execution."""


def canonical_bytes(value: Any) -> bytes:
    return admission.canonical_bytes(value)


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any, length: int, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(
        rf"[0-9a-f]{{{length}}}", value
    ) is None:
        raise CompatibilityError(f"{label} is not a {length * 4}-bit digest")
    return value


def _git_env() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent-sn56-fixture-compatibility",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }


def _remote_refs(repository: str, commit: str) -> list[str]:
    remote = subprocess.run(
        [
            "/usr/bin/git",
            "--no-replace-objects",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "ls-remote",
            "--heads",
            repository,
        ],
        cwd="/",
        env=_git_env(),
        check=False,
        capture_output=True,
        text=True,
    )
    if remote.returncode:
        raise CompatibilityError("cannot verify the execution Forge remote")
    refs = sorted(
        line.split("\t", 1)[1]
        for line in remote.stdout.splitlines()
        if "\t" in line and line.split("\t", 1)[0].lower() == commit
    )
    if not refs:
        raise CompatibilityError("execution Forge commit is not on a remote branch")
    return refs


def _utc(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(
        r"20\d\d-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\dZ", value
    ) is None:
        raise CompatibilityError("compatibility timestamp is invalid")
    try:
        admission._utc(value)
    except admission.AdmissionError as exc:
        raise CompatibilityError("compatibility timestamp is invalid") from exc
    return value


def _revision(revision: str, paths: Sequence[str]) -> dict[str, Any]:
    if any(
        not path
        or Path(path).is_absolute()
        or "\\" in path
        or ":" in path
        or any(part in {"", ".", ".."} for part in Path(path).parts)
        for path in paths
    ):
        raise CompatibilityError("authority path is not repository-relative")
    git = [
        "/usr/bin/git",
        "--no-replace-objects",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
    ]

    def run(*args: str) -> bytes:
        result = subprocess.run(
            [*git, *args],
            cwd=REPO_ROOT,
            env=_git_env(),
            check=False,
            capture_output=True,
        )
        if result.returncode:
            raise CompatibilityError(f"Git revision is unavailable: {revision}")
        return result.stdout

    commit = run("rev-parse", "--verify", f"{revision}^{{commit}}").decode().strip()
    tree = run("rev-parse", "--verify", f"{commit}^{{tree}}").decode().strip()
    _digest(commit, 40, "Forge commit")
    _digest(tree, 40, "Forge tree")
    blobs: dict[str, str] = {}
    for path in paths:
        blob = run("rev-parse", "--verify", f"{commit}:{path}").decode().strip()
        _digest(blob, 40, f"{path} blob")
        if run("cat-file", "-t", blob) != b"blob\n":
            raise CompatibilityError(f"authority object is not a blob: {path}")
        blobs[path] = _sha256(run("cat-file", "blob", blob))
    return {"commit": commit, "tree": tree, "blob_sha256": blobs}


def _current_execution(repository: str) -> dict[str, Any]:
    literal = _revision("HEAD", BOUND_PATHS)
    for path, expected in literal["blob_sha256"].items():
        if _sha256(admission.renderer._read_regular(REPO_ROOT / path, path)) != expected:
            raise CompatibilityError(f"ambient authority differs from HEAD: {path}")
    git = [
        "/usr/bin/git",
        "--no-replace-objects",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
    ]
    status = subprocess.run(
        [*git, "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT,
        env=_git_env(),
        check=False,
        capture_output=True,
    )
    if status.returncode or status.stdout:
        raise CompatibilityError("new Forge revision is not clean and remote-verifiable")
    refs = _remote_refs(repository, literal["commit"])
    return {**literal, "repository": repository, "pinned_remote_refs": refs}


def _runtimes() -> dict[str, dict[str, str]]:
    return {
        "incumbent": {
            "repository": krea_runtime.INCUMBENT_RUNTIME_REPOSITORY,
            "commit": krea_runtime.PINNED_BASE_COMMIT,
            "tree": krea_runtime.PINNED_BASE_TREE,
        },
        "owned": {
            "repository": krea_runtime.OWNED_RUNTIME_REPOSITORY,
            "commit": krea_runtime.OWNED_RUNTIME_COMMIT,
            "tree": krea_runtime.OWNED_RUNTIME_TREE,
        },
    }


def _source_revision(admission_set: Mapping[str, Any]) -> dict[str, Any]:
    revision = dict(admission_set["generator_revision"])
    declared = {
        revision["renderer_source_path"]: revision["renderer_source_sha256"],
        revision["admission_authority_path"]: revision[
            "admission_authority_source_sha256"
        ],
        revision["factor_authority_path"]: revision[
            "factor_authority_source_sha256"
        ],
        revision["contract_path"]: revision["contract_source_sha256"],
    }
    literal = _revision(revision["commit"], tuple(declared))
    if literal != {
        "commit": revision["commit"],
        "tree": revision["tree"],
        "blob_sha256": declared,
    }:
        raise CompatibilityError("admitted Forge revision does not resolve exactly")
    return revision


def _fixture_blobs(revision: Mapping[str, Any]) -> dict[str, str]:
    result = {
        revision["renderer_source_path"]: revision["renderer_source_sha256"],
        revision["admission_authority_path"]: revision[
            "admission_authority_source_sha256"
        ],
        revision["contract_path"]: revision["contract_source_sha256"],
    }
    if set(result) != set(FIXTURE_PATHS):
        raise CompatibilityError("fixture-authority path set changed")
    return result


def _validate_source_records(
    records: Any,
    admission_set: Mapping[str, Any],
    owner_ratification: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(records, Mapping) or records != SOURCE_RECORDS:
        raise CompatibilityError("source-file records are malformed")
    expected = {
        "admission_set": (
            canonical_bytes(admission_set),
            "admission_set_sha256",
            admission_set["admission_set_sha256"],
        ),
        "owner_ratification": (
            canonical_bytes(owner_ratification),
            "owner_ratification_sha256",
            owner_ratification["owner_ratification_sha256"],
        ),
    }
    for name, (raw, field, semantic) in expected.items():
        if records[name] != {
            "bytes": len(raw),
            "file_sha256": _sha256(raw),
            field: semantic,
        }:
            raise CompatibilityError(f"{name} exact raw-file binding mismatch")
    if records["sealed_human_review"]["review_sha256"] != admission_set[
        "human_review_sha256"
    ]:
        raise CompatibilityError("sealed review exact raw-file commitment is invalid")
    return {name: dict(record) for name, record in records.items()}


def _records_from_raw(
    raw: Mapping[str, bytes],
    admission_set: Mapping[str, Any],
    sealed_review: Mapping[str, Any],
    owner_ratification: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    values = {
        "admission_set": (
            admission_set,
            "admission_set_sha256",
            admission_set["admission_set_sha256"],
        ),
        "sealed_human_review": (
            sealed_review,
            "review_sha256",
            sealed_review["review_sha256"],
        ),
        "owner_ratification": (
            owner_ratification,
            "owner_ratification_sha256",
            owner_ratification["owner_ratification_sha256"],
        ),
    }
    if not isinstance(raw, Mapping) or set(raw) != set(values):
        raise CompatibilityError("exact source-file bytes are incomplete")
    records: dict[str, dict[str, Any]] = {}
    for name, (value, field, semantic) in values.items():
        payload = raw[name]
        if not isinstance(payload, bytes) or payload != canonical_bytes(value):
            raise CompatibilityError(f"{name} bytes do not encode the bound record")
        records[name] = {
            "bytes": len(payload),
            "file_sha256": _sha256(payload),
            field: semantic,
        }
    if records != SOURCE_RECORDS:
        raise CompatibilityError("source files are not the sealed admitted records")
    return records


def _body(
    admission_set: Mapping[str, Any],
    owner_ratification: Mapping[str, Any],
    source_records: Mapping[str, Mapping[str, Any]],
    target: Mapping[str, Any],
    created_at_utc: str,
) -> dict[str, Any]:
    source = _source_revision(admission_set)
    source_blobs = _fixture_blobs(source)
    if set(target) != {
        "repository",
        "commit",
        "tree",
        "pinned_remote_refs",
        "blob_sha256",
    } or set(target.get("blob_sha256", {})) != set(BOUND_PATHS):
        raise CompatibilityError("execution revision record is malformed")
    _digest(target["commit"], 40, "execution Forge commit")
    _digest(target["tree"], 40, "execution Forge tree")
    refs = target["pinned_remote_refs"]
    if (
        target["repository"] != source["repository"]
        or not isinstance(refs, list)
        or not refs
        or refs != sorted(set(refs))
        or any(
            not isinstance(ref, str)
            or re.fullmatch(r"refs/heads/[A-Za-z0-9._/-]+", ref) is None
            for ref in refs
        )
    ):
        raise CompatibilityError("execution repository or remote refs are invalid")
    literal = _revision(target["commit"], BOUND_PATHS)
    if literal != {
        "commit": target["commit"],
        "tree": target["tree"],
        "blob_sha256": dict(target["blob_sha256"]),
    }:
        raise CompatibilityError("execution Forge revision does not resolve exactly")
    _utc(created_at_utc)
    target_blobs = {path: target["blob_sha256"][path] for path in FIXTURE_PATHS}
    if target_blobs != source_blobs:
        raise CompatibilityError("fixture authority changed; re-admission is required")
    return {
        "schema": SCHEMA,
        "kind": KIND,
        "status": "PASS_FIXTURE_AUTHORITY_COMPATIBLE_GPU_CLOSED",
        "source_records": _validate_source_records(
            source_records, admission_set, owner_ratification
        ),
        "candidate_semantic_sha256": admission_set["candidate_semantic_sha256"],
        "source_admission_revision": {
            "repository": source["repository"],
            "commit": source["commit"],
            "tree": source["tree"],
        },
        "fixture_authority": {
            "paths": list(FIXTURE_PATHS),
            "source_blob_sha256": source_blobs,
            "execution_blob_sha256": target_blobs,
            "all_exact_git_blobs_identical": True,
        },
        "execution_authority": {
            "forge_repository": target["repository"],
            "forge_commit": target["commit"],
            "forge_tree": target["tree"],
            "pinned_remote_refs": list(target["pinned_remote_refs"]),
            "entrypoint_blob_sha256": {
                path: target["blob_sha256"][path] for path in EXECUTION_PATHS
            },
            "runtime": _runtimes(),
        },
        "created_at_utc": created_at_utc,
        "authorization": {
            "fixture_regeneration_required": False,
            "owner_reratification_required": False,
            "independent_exact_sha_audit_required": True,
            "gpu_execution_authorized": False,
            "deployment_authorized": False,
        },
    }


def build_receipt(
    *,
    admission_set: Mapping[str, Any],
    sealed_review: Mapping[str, Any],
    owner_ratification: Mapping[str, Any],
    source_raw_files: Mapping[str, bytes],
    created_at_utc: str,
    execution_probe: Callable[[str], Mapping[str, Any]] = _current_execution,
) -> dict[str, Any]:
    try:
        checked = admission._validate_admission_set_envelope(admission_set)
        admission._validate_private_review_for_ratification(sealed_review, checked)
        admission.validate_owner_ratification(
            owner_ratification,
            admission_set=admission_set,
            sealed_review=sealed_review,
        )
    except admission.AdmissionError as exc:
        raise CompatibilityError(str(exc)) from exc
    _utc(created_at_utc)
    source = _source_revision(admission_set)
    source_records = _records_from_raw(
        source_raw_files, admission_set, sealed_review, owner_ratification
    )
    body = _body(
        admission_set,
        owner_ratification,
        source_records,
        execution_probe(source["repository"]),
        str(created_at_utc),
    )
    return {**body, "compatibility_receipt_sha256": semantic_sha256(body)}


def validate_receipt(
    value: Mapping[str, Any],
    *,
    admission_set: Mapping[str, Any],
    owner_ratification: Mapping[str, Any],
    execution_identities: Mapping[str, Mapping[str, str]] | None = None,
    require_current_execution: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CompatibilityError("compatibility receipt is not an object")
    body = dict(value)
    declared = body.pop("compatibility_receipt_sha256", None)
    if _digest(declared, 64, "compatibility receipt") != semantic_sha256(body):
        raise CompatibilityError("compatibility receipt digest mismatch")
    try:
        admission._validate_admission_set_envelope(admission_set)
    except admission.AdmissionError as exc:
        raise CompatibilityError(str(exc)) from exc
    rat_body = dict(owner_ratification)
    rat_sha = rat_body.pop("owner_ratification_sha256", None)
    if (
        _digest(rat_sha, 64, "owner ratification")
        != admission.semantic_sha256(rat_body)
        or owner_ratification.get("admission_set_sha256")
        != admission_set["admission_set_sha256"]
        or owner_ratification.get("human_review_sha256")
        != admission_set["human_review_sha256"]
    ):
        raise CompatibilityError("owner ratification no longer binds admission")
    execution = body.get("execution_authority")
    if not isinstance(execution, Mapping):
        raise CompatibilityError("execution authority is malformed")
    target = {
        "repository": execution.get("forge_repository"),
        "commit": execution.get("forge_commit"),
        "tree": execution.get("forge_tree"),
        "pinned_remote_refs": execution.get("pinned_remote_refs"),
        "blob_sha256": {
            **dict(body.get("fixture_authority", {}).get("execution_blob_sha256", {})),
            **dict(execution.get("entrypoint_blob_sha256", {})),
        },
    }
    literal = _revision(str(target["commit"]), BOUND_PATHS)
    if target["commit"] != literal["commit"] or target["tree"] != literal["tree"]:
        raise CompatibilityError("execution revision does not resolve exactly")
    target["blob_sha256"] = literal["blob_sha256"]
    expected = _body(
        admission_set,
        owner_ratification,
        body.get("source_records", {}),
        target,
        str(body.get("created_at_utc", "")),
    )
    if body != expected:
        raise CompatibilityError("compatibility receipt does not reproduce")
    live_refs = _remote_refs(execution["forge_repository"], execution["forge_commit"])
    if not set(execution["pinned_remote_refs"]).issubset(live_refs):
        raise CompatibilityError("execution Forge pinned refs no longer resolve")
    if require_current_execution:
        if _revision("HEAD", BOUND_PATHS) != literal or any(
            _sha256(admission.renderer._read_regular(REPO_ROOT / path, path))
            != digest
            for path, digest in literal["blob_sha256"].items()
        ):
            raise CompatibilityError("receipt does not target exact current Forge bytes")
    if execution_identities is not None:
        if not isinstance(execution_identities, Mapping) or set(
            execution_identities
        ) != {"incumbent", "owned"}:
            raise CompatibilityError("execution identities are malformed")
        for name, runtime in _runtimes().items():
            identity = execution_identities[name]
            if (
                identity.get("code_tree") != execution["forge_tree"]
                or identity.get("runtime_tree") != runtime["tree"]
            ):
                raise CompatibilityError(f"{name} execution identity is incompatible")
    return dict(value)


def _read_exact_json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    admission.renderer._require_no_symlink_components(path, label)
    metadata = path.lstat()
    if metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise CompatibilityError(f"{label} must be 0600 and single-link")
    raw = read_regular_bytes(
        str(path), label=label, maximum_size=MAX_RECORD_BYTES
    )
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompatibilityError(f"{label} is not JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise CompatibilityError(f"{label} is not exact canonical JSON")
    return raw, value


def create_from_files(
    admission_path: Path,
    review_path: Path,
    ratification_path: Path,
    output_path: Path,
    created_at_utc: str,
    *,
    execution_probe: Callable[[str], Mapping[str, Any]] = _current_execution,
) -> dict[str, Any]:
    paths = (Path(admission_path), Path(review_path), Path(ratification_path))
    labels = ("admission set", "sealed human review", "owner ratification")
    opened = [
        _read_exact_json(path, label)
        for path, label in zip(paths, labels, strict=True)
    ]
    raw = [item[0] for item in opened]
    admission_set, review, ratification = [item[1] for item in opened]
    receipt = build_receipt(
        admission_set=admission_set,
        sealed_review=review,
        owner_ratification=ratification,
        source_raw_files={
            "admission_set": raw[0],
            "sealed_human_review": raw[1],
            "owner_ratification": raw[2],
        },
        created_at_utc=created_at_utc,
        execution_probe=execution_probe,
    )
    admission.renderer._write_exclusive(Path(output_path), canonical_bytes(receipt))
    return receipt


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission-set", type=Path, required=True)
    parser.add_argument("--sealed-review", type=Path, required=True)
    parser.add_argument("--owner-ratification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--created-at-utc", required=True)
    args = parser.parse_args(argv)
    create_from_files(
        args.admission_set,
        args.sealed_review,
        args.owner_ratification,
        args.output,
        args.created_at_utc,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
