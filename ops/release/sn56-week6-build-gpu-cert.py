#!/usr/bin/python3
"""Policy-neutral Week-6 Docker build and GPU certificate delegate.

The delegate consumes one exact pushed Git commit, materializes that commit from
Git objects into a private directory, and certifies both validator-routed image
trainer Dockerfiles.  It contains no campaign-policy or model-quality decision.

``production`` performs the physical H100 observations and can emit ``PASS``.
``cpu-integration`` executes the complete CPU/build surface but replaces only
the host/container GPU observations with an internal marker; that mode can emit
only ``DRY_RUN_PASS`` and therefore cannot be mistaken for release authority.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, BinaryIO, Callable, Iterable, Mapping, Sequence


RESULT_SCHEMA = "sn56.week6.build-gpu-cert.v2"
PRODUCTION_MODE = "production"
CPU_INTEGRATION_MODE = "cpu-integration"
PASS_STATE = "PASS"
DRY_RUN_PASS_STATE = "DRY_RUN_PASS"
CPU_ACCELERATOR_IDENTITY = "INTEGRATION-STUB-NO-GPU-CLAIM|0-MiB"

SHA1_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
NAMESPACE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
IMAGE_TAG_RE = re.compile(
    r"[a-z0-9][a-z0-9._/-]*:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}"
)
FROM_DIGEST_RE = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")

GIT_CONFIG_OVERRIDES = (
    ("core.fsmonitor", "false"),
    ("core.hooksPath", "/dev/null"),
    ("core.untrackedCache", "false"),
    ("core.ignoreStat", "false"),
    ("core.trustctime", "true"),
    ("core.checkStat", "default"),
    ("core.attributesFile", "/dev/null"),
    ("core.excludesFile", "/dev/null"),
)

FIXED_ENV = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "HOME": "/nonexistent",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_CONFIG_COUNT": str(len(GIT_CONFIG_OVERRIDES)),
}
for _git_config_index, (_git_config_key, _git_config_value) in enumerate(
    GIT_CONFIG_OVERRIDES
):
    FIXED_ENV[f"GIT_CONFIG_KEY_{_git_config_index}"] = _git_config_key
    FIXED_ENV[f"GIT_CONFIG_VALUE_{_git_config_index}"] = _git_config_value

ABSOLUTE_TOOLS: dict[str, str] = {
    "containerd": "/usr/bin/containerd",
    "docker": "/usr/bin/docker",
    "git": "/usr/bin/git",
    "nvidia_smi": "/usr/bin/nvidia-smi",
    "systemctl": "/usr/bin/systemctl",
}

LOCK_REL = "ops/docker/image-runtime-lock.txt"
CONSTRAINTS_REL = "ops/docker/image-runtime-phase1-constraints.txt"
VERIFIER_REL = "ops/docker/verify_image_runtime.py"

ROOT_START_MIN = 20 * 1024**3
WORK_START_MIN = 500 * 1024**3
EVIDENCE_START_MIN = 5 * 1024**3
ROOT_PRESSURE_FLOOR = 16 * 1024**3
WORK_PRESSURE_FLOOR = 450 * 1024**3
EVIDENCE_PRESSURE_FLOOR = 2 * 1024**3


@dataclass(frozen=True)
class Subject:
    name: str
    dockerfile: str


SUBJECTS = (
    Subject(
        name="toolkit",
        dockerfile="ops/docker/standalone-image-toolkit-trainer.dockerfile",
    ),
    Subject(
        name="legacy",
        dockerfile="ops/docker/standalone-image-trainer.dockerfile",
    ),
)


class DelegateError(RuntimeError):
    """A build/GPU certificate precondition or gate failed."""


def require(condition: bool, message: str) -> None:
    """Raise an explicit runtime exception; never rely on Python assertions."""

    if not condition:
        raise DelegateError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(payload: bytes) -> str:
    prefix = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(prefix + payload).hexdigest()  # noqa: S324 - Git identity


def fixed_command_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    result = dict(FIXED_ENV)
    if extra:
        for name, value in extra.items():
            require(
                re.fullmatch(r"[A-Z][A-Z0-9_]*", name) is not None,
                f"invalid fixed environment key: {name}",
            )
            require("\x00" not in value, f"fixed environment value contains NUL: {name}")
            result[name] = value
    return result


def validate_absolute_tool_paths(tools: Mapping[str, str], mode: str) -> None:
    required = {"containerd", "docker", "git", "systemctl"}
    if mode == PRODUCTION_MODE:
        required.add("nvidia_smi")
    for name in sorted(required):
        path = tools.get(name)
        require(path is not None and os.path.isabs(path), f"{name} tool is not absolute")
        require(
            os.path.isfile(path) and os.access(path, os.X_OK),
            f"required absolute tool is unavailable: {name}={path}",
        )


def run_capture(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    stdin: BinaryIO | None = None,
    timeout: int = 300,
    extra_env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    require(bool(argv) and os.path.isabs(argv[0]), "subprocess executable is not absolute")
    completed = subprocess.run(
        list(argv),
        cwd=str(cwd) if cwd is not None else None,
        env=fixed_command_env(extra_env),
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise DelegateError(
            f"command failed rc={completed.returncode} executable={argv[0]} stderr={stderr}"
        )
    return completed


def safe_absolute_directory(path: str, label: str, *, must_exist: bool = True) -> Path:
    require(isinstance(path, str) and os.path.isabs(path), f"{label} must be absolute")
    require(os.path.normpath(path) == path, f"{label} contains lexical indirection")
    candidate = Path(path)
    if must_exist:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise DelegateError(f"{label} is unavailable") from exc
        require(resolved == candidate, f"{label} is symlinked or indirect")
        require(candidate.is_dir(), f"{label} is not a directory")
    return candidate


def open_directory_chain(path: Path, *, create: bool, mode: int = 0o750) -> int:
    """Open an absolute directory without following a symlink component."""

    require(path.is_absolute(), "directory chain must be absolute")
    require(path == Path(os.path.normpath(str(path))), "directory chain is indirect")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            require(component not in {"", ".", ".."}, "invalid directory component")
            if create:
                try:
                    os.mkdir(component, mode=mode, dir_fd=descriptor)
                except FileExistsError:
                    pass
            try:
                child = os.open(component, flags | nofollow, dir_fd=descriptor)
            except OSError as exc:
                raise DelegateError(
                    f"directory chain contains an absent, non-directory, or symlink component: {path}"
                ) from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _rename_noreplace(
    source_name: str,
    destination_name: str,
    directory_fd: int,
) -> None:
    """Atomically publish one directory without replacing an existing name."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            directory_fd,
            os.fsencode(source_name),
            directory_fd,
            os.fsencode(destination_name),
            1,  # RENAME_NOREPLACE
        )
        if result == 0:
            return
        error = ctypes.get_errno()
        if error != errno.ENOSYS:
            raise OSError(error, os.strerror(error), destination_name)

    # Portable fallback for development hosts. Production Linux has renameat2.
    try:
        os.stat(destination_name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), destination_name)
    os.rename(
        source_name,
        destination_name,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
    )


class AtomicEvidence:
    """Build a complete evidence tree privately and publish it exactly once."""

    def __init__(self, base: Path, namespace: str):
        require(NAMESPACE_RE.fullmatch(namespace) is not None, "invalid evidence namespace")
        self.base = base
        self.namespace = namespace
        self.base_fd = open_directory_chain(base, create=True)
        try:
            os.stat(namespace, dir_fd=self.base_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            os.close(self.base_fd)
            raise DelegateError(f"evidence namespace already exists: {namespace}")
        self.stage_name = f".{namespace}.staging.{os.getpid()}.{os.urandom(8).hex()}"
        os.mkdir(self.stage_name, mode=0o700, dir_fd=self.base_fd)
        self.stage_path = base / self.stage_name
        self.published_path: Path | None = None

    def path(self, relative: str) -> Path:
        rel = PurePosixPath(relative)
        require(
            not rel.is_absolute() and ".." not in rel.parts and str(rel) not in {"", "."},
            "invalid evidence relative path",
        )
        result = self.stage_path.joinpath(*rel.parts)
        result.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
        return result

    def write_bytes(self, relative: str, payload: bytes, mode: int = 0o440) -> Path:
        destination = self.path(relative)
        temporary = destination.with_name(f".{destination.name}.tmp.{os.urandom(6).hex()}")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            mode,
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                require(written > 0, "evidence write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, destination)
        return destination

    def write_json(self, relative: str, value: Any) -> Path:
        return self.write_bytes(relative, canonical_json(value))

    def _manifest_payload(self) -> bytes:
        rows: list[str] = []
        for candidate in sorted(self.stage_path.rglob("*")):
            relative = candidate.relative_to(self.stage_path).as_posix()
            if relative == "MANIFEST.sha256" or candidate.is_dir():
                continue
            require(candidate.is_file() and not candidate.is_symlink(), "invalid evidence entry")
            rows.append(f"{sha256_file(candidate)}  {relative}\n")
        return "".join(rows).encode("ascii")

    def publish(self) -> Path:
        require(self.published_path is None, "evidence was already published")
        self.write_bytes("MANIFEST.sha256", self._manifest_payload())
        for candidate in sorted(self.stage_path.rglob("*"), reverse=True):
            if candidate.is_dir():
                candidate.chmod(0o550)
            else:
                candidate.chmod(0o440)
        self.stage_path.chmod(0o550)
        stage_fd = os.open(
            self.stage_path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)
        try:
            _rename_noreplace(self.stage_name, self.namespace, self.base_fd)
        except OSError as exc:
            raise DelegateError(f"atomic evidence publish failed: {exc}") from exc
        os.fsync(self.base_fd)
        self.published_path = self.base / self.namespace
        return self.published_path

    def close(self) -> None:
        if self.base_fd >= 0:
            os.close(self.base_fd)
            self.base_fd = -1

    def discard_unpublished(self) -> None:
        if self.published_path is None and self.stage_path.exists():
            shutil.rmtree(self.stage_path)


def git_command(tools: Mapping[str, str], repository: Path, *arguments: str) -> list[str]:
    command = [
        tools["git"],
        "--no-replace-objects",
    ]
    for key, value in GIT_CONFIG_OVERRIDES:
        command.extend(("-c", f"{key}={value}"))
    command.extend(("-c", f"safe.directory={repository}", "-C", str(repository)))
    command.extend(arguments)
    return command


@dataclass(frozen=True)
class SourceIdentity:
    commit: str
    tree: str
    forge_tree: str


def verify_source_repository(
    repository: Path,
    expected: SourceIdentity,
    tools: Mapping[str, str],
) -> None:
    git = lambda *args: run_capture(git_command(tools, repository, *args)).stdout.decode(
        "utf-8", errors="strict"
    ).strip()
    require(git("rev-parse", "HEAD") == expected.commit, "source HEAD differs")
    require(git("rev-parse", "HEAD^{tree}") == expected.tree, "source tree differs")
    require(git("rev-parse", "HEAD:forge") == expected.forge_tree, "Forge tree differs")
    top = Path(git("rev-parse", "--show-toplevel")).resolve(strict=True)
    require(top == repository, "source checkout is not the exact repository root")
    status = git("status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching")
    require(not status, "source checkout has changed, untracked, or ignored surfaces")


def parse_ls_tree(payload: bytes) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for raw in payload.split(b"\0"):
        if not raw:
            continue
        try:
            metadata, path_bytes = raw.split(b"\t", 1)
            mode_bytes, type_bytes, object_bytes = metadata.split(b" ", 2)
            path = path_bytes.decode("utf-8")
            mode = mode_bytes.decode("ascii")
            object_id = object_bytes.decode("ascii")
            object_type = type_bytes.decode("ascii")
        except (ValueError, UnicodeError) as exc:
            raise DelegateError("git ls-tree output is malformed") from exc
        require(object_type == "blob", f"unsupported committed object: {path} ({object_type})")
        require(mode in {"100644", "100755"}, f"unsupported committed mode: {path} ({mode})")
        require(path not in result, f"duplicate committed path: {path}")
        result[path] = (mode, object_id)
    require(bool(result), "release tree contains no files")
    return result


def _validate_tar_member(member: tarfile.TarInfo) -> PurePosixPath:
    path = PurePosixPath(member.name)
    require(
        not path.is_absolute() and ".." not in path.parts and str(path) not in {"", "."},
        f"unsafe archive member path: {member.name}",
    )
    require(member.isdir() or member.isreg(), f"unsupported archive member: {member.name}")
    return path


def extract_regular_archive(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, mode="r:") as bundle:
        members = bundle.getmembers()
        validated = [(member, _validate_tar_member(member)) for member in members]
        for member, relative in sorted(validated, key=lambda row: len(row[1].parts)):
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(mode=0o755, parents=True, exist_ok=True)
                continue
            target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            require(not target.exists() and not target.is_symlink(), "archive path collision")
            source = bundle.extractfile(member)
            require(source is not None, f"archive file is unreadable: {member.name}")
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o755 if member.mode & 0o111 else 0o644,
            )
            consumed = 0
            try:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    consumed += len(block)
                    view = memoryview(block)
                    while view:
                        written = os.write(descriptor, view)
                        require(written > 0, "archive extraction made no progress")
                        view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
                source.close()
            require(consumed == member.size, f"archive member size differs: {member.name}")


@dataclass(frozen=True)
class MaterializedSource:
    root: Path
    archive_sha256: str
    file_manifest_sha256: str
    production_manifest_sha256: str
    file_hashes: Mapping[str, str]


def materialize_exact_archive(
    repository: Path,
    expected: SourceIdentity,
    work_directory: Path,
    tools: Mapping[str, str],
) -> MaterializedSource:
    archive = work_directory / "release.tar"
    materialized = work_directory / "source"
    materialized.mkdir(mode=0o700)
    with archive.open("wb") as output:
        completed = subprocess.run(
            git_command(
                tools,
                repository,
                "archive",
                "--format=tar",
                expected.commit,
            ),
            env=fixed_command_env(),
            stdout=output,
            stderr=subprocess.PIPE,
            timeout=300,
            check=False,
        )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise DelegateError(f"git archive failed: {stderr}")
    require(archive.stat().st_size > 0, "git archive is empty")

    with archive.open("rb") as input_stream:
        embedded = run_capture(
            [tools["git"], "--no-replace-objects", "get-tar-commit-id"],
            cwd=Path("/"),
            stdin=input_stream,
        ).stdout.decode("ascii", errors="strict").strip()
    require(embedded == expected.commit, "git archive commit identity differs")
    extract_regular_archive(archive, materialized)

    tree_output = run_capture(
        git_command(
            tools,
            repository,
            "ls-tree",
            "-rz",
            "--full-tree",
            expected.commit,
        )
    ).stdout
    committed = parse_ls_tree(tree_output)
    actual_paths: set[str] = set()
    file_hashes: dict[str, str] = {}
    manifest_rows: list[str] = []
    production_rows: list[str] = []
    for candidate in sorted(materialized.rglob("*")):
        if candidate.is_dir():
            continue
        require(candidate.is_file() and not candidate.is_symlink(), "materialized tree is irregular")
        relative = candidate.relative_to(materialized).as_posix()
        actual_paths.add(relative)
        require(relative in committed, f"archive contains uncommitted path: {relative}")
        mode, object_id = committed[relative]
        payload = candidate.read_bytes()
        require(git_blob_sha1(payload) == object_id, f"archive blob differs: {relative}")
        executable = bool(candidate.stat().st_mode & 0o111)
        require(executable == (mode == "100755"), f"archive mode differs: {relative}")
        digest = sha256_bytes(payload)
        file_hashes[relative] = digest
        manifest_rows.append(f"{digest} {mode} {relative}\n")
        if relative.startswith("forge/") or relative.startswith("ops/docker/"):
            production_rows.append(f"{digest} {mode} {relative}\n")
    require(actual_paths == set(committed), "materialized file set differs from commit")
    require(bool(production_rows), "production source manifest is empty")
    return MaterializedSource(
        root=materialized,
        archive_sha256=sha256_file(archive),
        file_manifest_sha256=sha256_bytes("".join(manifest_rows).encode("utf-8")),
        production_manifest_sha256=sha256_bytes(
            "".join(production_rows).encode("utf-8")
        ),
        file_hashes=file_hashes,
    )


def validate_dockerfile_digest_pins(path: Path) -> list[str]:
    bases: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped.upper().startswith("FROM "):
            continue
        fields = stripped.split()
        require(len(fields) in {2, 4}, f"unsupported FROM syntax: {stripped}")
        require(len(fields) == 2 or fields[2].upper() == "AS", f"unsupported FROM alias")
        image = fields[1]
        require(FROM_DIGEST_RE.fullmatch(image) is not None, f"base image is not digest pinned: {image}")
        bases.append(image)
    require(bool(bases), f"Dockerfile has no FROM instruction: {path}")
    return bases


def source_forge_manifest(source: Path) -> dict[str, str]:
    root = source / "forge"
    require(root.is_dir() and not root.is_symlink(), "materialized Forge directory is absent")
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(source).as_posix()
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        require(path.is_file() and not path.is_symlink(), f"irregular Forge path: {relative}")
        result[relative] = sha256_file(path)
    require(bool(result), "source Forge manifest is empty")
    return result


def docker_build_command(
    tools: Mapping[str, str], subject: Subject, image_tag: str
) -> list[str]:
    return [
        tools["docker"],
        "build",
        "--no-cache",
        "--progress=plain",
        "--file",
        subject.dockerfile,
        "--tag",
        image_tag,
        ".",
    ]


def docker_run_command(
    tools: Mapping[str, str], image_tag: str, arguments: Sequence[str]
) -> list[str]:
    return [tools["docker"], "run", "--rm", "--pull", "never", "--network", "none", *arguments, image_tag]


def run_logged(
    argv: Sequence[str],
    log_path: Path,
    *,
    cwd: Path,
    timeout: int,
    pressure_paths: Mapping[str, tuple[Path, int]] | None = None,
    pressure_log: Path | None = None,
) -> None:
    require(bool(argv) and os.path.isabs(argv[0]), "logged executable is not absolute")
    started = time.monotonic()
    pressure_stream = pressure_log.open("w", encoding="utf-8") if pressure_log else None
    if pressure_stream is not None:
        pressure_stream.write("utc\tfilesystem\tavailable_bytes\tfloor_bytes\n")
        pressure_stream.flush()
    try:
        log = log_path.open("wb")
    except Exception:
        if pressure_stream is not None:
            pressure_stream.close()
        raise
    with log:
        process = subprocess.Popen(
            list(argv),
            cwd=str(cwd),
            env=fixed_command_env({"DOCKER_BUILDKIT": "1"}),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        next_pressure_check = 0.0
        try:
            while process.poll() is None:
                now = time.monotonic()
                if now - started > timeout:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=30)
                    raise DelegateError(f"command timed out: {argv[0]}")
                if pressure_paths and now >= next_pressure_check:
                    for name, (path, floor) in pressure_paths.items():
                        available = shutil.disk_usage(path).free
                        if pressure_stream is not None:
                            pressure_stream.write(
                                f"{utc_now()}\t{name}\t{available}\t{floor}\n"
                            )
                            pressure_stream.flush()
                        if available < floor:
                            raise DelegateError(
                                f"filesystem pressure floor crossed: {name} "
                                f"available={available} floor={floor}"
                            )
                    next_pressure_check = now + 10.0
                time.sleep(1)
        except Exception:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
            raise
        finally:
            if pressure_stream is not None:
                pressure_stream.close()
    require(process.returncode == 0, f"logged command failed rc={process.returncode}: {argv[0]}")


def parse_containerd_root(payload: str) -> str:
    matches = re.findall(r'^root = [\'\"]([^\'\"]+)[\'\"]$', payload, flags=re.MULTILINE)
    require(len(matches) == 1, "containerd root could not be parsed uniquely")
    return matches[0]


def docker_ps_ids(tools: Mapping[str, str]) -> list[str]:
    output = run_capture([tools["docker"], "ps", "--quiet"]).stdout.decode(
        "utf-8", errors="strict"
    )
    return [row for row in output.splitlines() if row]


def verify_host_contract(args: argparse.Namespace, tools: Mapping[str, str]) -> dict[str, Any]:
    require(sys.platform.startswith("linux"), "release delegate requires Linux")
    require(os.geteuid() == 0, "release delegate must run as root")
    validate_absolute_tool_paths(tools, args.mode)
    require(not docker_ps_ids(tools), "running containers exist before certification")
    for service in ("docker", "containerd"):
        state = run_capture([tools["systemctl"], "is-active", service]).stdout.decode(
            "ascii", errors="strict"
        ).strip()
        require(state == "active", f"service is not active: {service}")
    docker_root = run_capture(
        [tools["docker"], "info", "--format", "{{.DockerRootDir}}"]
    ).stdout.decode("utf-8", errors="strict").strip()
    require(docker_root == args.expected_docker_root, "Docker root differs")
    containerd_dump = run_capture([tools["containerd"], "config", "dump"]).stdout.decode(
        "utf-8", errors="strict"
    )
    containerd_root = parse_containerd_root(containerd_dump)
    require(containerd_root == args.expected_containerd_root, "containerd root differs")
    root_device = os.stat("/").st_dev
    work_device = os.stat(args.work_base).st_dev
    evidence_device = os.stat(args.evidence_base).st_dev
    require(work_device != root_device, "work filesystem is not independent from root")
    require(
        evidence_device not in {root_device, work_device},
        "evidence filesystem is not independent from root and work storage",
    )
    capacities = {
        "root": shutil.disk_usage("/").free,
        "work": shutil.disk_usage(args.work_base).free,
        "evidence": shutil.disk_usage(args.evidence_base).free,
    }
    minimums = {
        "root": ROOT_START_MIN,
        "work": WORK_START_MIN,
        "evidence": EVIDENCE_START_MIN,
    }
    for name, minimum in minimums.items():
        require(
            capacities[name] >= minimum,
            f"{name} filesystem lacks start capacity: "
            f"available={capacities[name]} minimum={minimum}",
        )
    return {
        "available_bytes": capacities,
        "containerd_root": containerd_root,
        "docker_root": docker_root,
        "filesystem_devices": {
            "evidence": evidence_device,
            "root": root_device,
            "work": work_device,
        },
        "minimum_start_bytes": minimums,
        "platform": platform.platform(),
    }


def image_id(tools: Mapping[str, str], image_tag: str) -> str:
    value = run_capture(
        [tools["docker"], "image", "inspect", "--format", "{{.Id}}", image_tag]
    ).stdout.decode("ascii", errors="strict").strip()
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None, "invalid image ID")
    return value


def image_exists(tools: Mapping[str, str], image_tag: str) -> bool:
    completed = subprocess.run(
        [tools["docker"], "image", "inspect", image_tag],
        env=fixed_command_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    return completed.returncode == 0


def run_image_python(
    tools: Mapping[str, str], image_tag: str, program: str, *program_args: str
) -> subprocess.CompletedProcess[bytes]:
    argv = docker_run_command(
        tools,
        image_tag,
        ["--entrypoint", "python3"],
    )
    argv.extend(["-c", program, *program_args])
    return run_capture(argv, timeout=600)


IMAGE_INPUT_PROGRAM = r"""
import hashlib
from pathlib import Path
import sys
expected = dict(item.split("=", 1) for item in sys.argv[1:])
paths = {
    "lock": Path("/opt/sn56/image-runtime-lock.txt"),
    "constraints": Path("/opt/sn56/image-runtime-phase1-constraints.txt"),
    "verifier": Path("/opt/sn56/verify-image-runtime.py"),
}
for name, path in paths.items():
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"image input is absent or irregular: {name}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected[name]:
        raise RuntimeError(f"image input hash differs: {name}")
print("SN56_IMAGE_INPUT_BINDING=PASS")
"""


IMAGE_FORGE_MANIFEST_PROGRAM = r"""
import hashlib
import json
from pathlib import Path
root = Path("/app")
forge = root / "forge"
if not forge.is_dir() or forge.is_symlink():
    raise RuntimeError("image Forge directory is absent")
result = {}
for path in sorted(forge.rglob("*")):
    if path.is_dir():
        continue
    if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
        continue
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"irregular image Forge path: {path}")
    result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
if not result:
    raise RuntimeError("image Forge manifest is empty")
print(json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True))
"""


IMAGE_CPU_IMPORT_PROGRAM = r"""
import json
import torch
import forge
print(json.dumps({
    "forge": forge.__file__,
    "state": "PASS",
    "torch": torch.__version__,
}, allow_nan=False, separators=(",", ":"), sort_keys=True))
"""


GPU_IMPORT_PROGRAM = r"""
import json
import torch
import forge
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
name = torch.cuda.get_device_name(0)
if "H100" not in name:
    raise RuntimeError(f"GPU is not H100: {name}")
print(json.dumps({
    "cuda": torch.version.cuda,
    "device": name,
    "forge": forge.__file__,
    "state": "PASS",
    "torch": torch.__version__,
}, allow_nan=False, separators=(",", ":"), sort_keys=True))
"""


def observe_image_cpu_imports(
    tools: Mapping[str, str], image_tag: str
) -> dict[str, Any]:
    """Import the shipped runtime without crossing the physical GPU boundary."""

    completed = run_image_python(tools, image_tag, IMAGE_CPU_IMPORT_PROGRAM)
    try:
        value = json.loads(completed.stdout)
    except Exception as exc:
        raise DelegateError("container CPU import observation is not valid JSON") from exc
    require(
        isinstance(value, dict) and value.get("state") == "PASS",
        "container CPU imports did not pass",
    )
    require(bool(str(value.get("torch", "")).strip()), "container torch version is absent")
    forge_path = str(value.get("forge", ""))
    require(
        forge_path.startswith("/app/forge/"),
        f"container Forge imported from an unexpected path: {forge_path}",
    )
    return value


def certify_cpu_image_surface(
    tools: Mapping[str, str],
    source: MaterializedSource,
    subject: Subject,
    image_tag: str,
    evidence: AtomicEvidence,
) -> dict[str, Any]:
    hashes = {
        "lock": source.file_hashes[LOCK_REL],
        "constraints": source.file_hashes[CONSTRAINTS_REL],
        "verifier": source.file_hashes[VERIFIER_REL],
    }
    binding = run_image_python(
        tools,
        image_tag,
        IMAGE_INPUT_PROGRAM,
        *(f"{name}={digest}" for name, digest in sorted(hashes.items())),
    )
    require(
        binding.stdout.decode("utf-8", errors="strict").strip()
        == "SN56_IMAGE_INPUT_BINDING=PASS",
        f"{subject.name} image input binding did not pass",
    )
    evidence.write_bytes(f"subjects/{subject.name}/image-input-binding.stdout", binding.stdout)

    import_observation = observe_image_cpu_imports(tools, image_tag)
    evidence.write_json(
        f"subjects/{subject.name}/cpu-import-observation.json",
        import_observation,
    )

    verifier = run_capture(
        docker_run_command(
            tools,
            image_tag,
            [
                "--entrypoint",
                "python3",
            ],
        )
        + [
            "/opt/sn56/verify-image-runtime.py",
            "--lock",
            "/opt/sn56/image-runtime-lock.txt",
            "--constraints",
            "/opt/sn56/image-runtime-phase1-constraints.txt",
        ],
        timeout=600,
    )
    require(
        "SN56_IMAGE_RUNTIME_INVENTORY=PASS"
        in verifier.stdout.decode("utf-8", errors="strict").splitlines(),
        f"{subject.name} offline runtime verifier did not pass",
    )
    evidence.write_bytes(f"subjects/{subject.name}/offline-verifier.stdout", verifier.stdout)
    evidence.write_bytes(f"subjects/{subject.name}/offline-verifier.stderr", verifier.stderr)

    help_result = run_capture(
        docker_run_command(tools, image_tag, []) + ["--help"], timeout=300
    )
    evidence.write_bytes(f"subjects/{subject.name}/cli-help.stdout", help_result.stdout)
    evidence.write_bytes(f"subjects/{subject.name}/cli-help.stderr", help_result.stderr)

    manifest_result = run_image_python(
        tools, image_tag, IMAGE_FORGE_MANIFEST_PROGRAM
    )
    try:
        image_manifest = json.loads(manifest_result.stdout)
    except Exception as exc:
        raise DelegateError(f"{subject.name} image Forge manifest is invalid") from exc
    require(
        image_manifest == source_forge_manifest(source.root),
        f"{subject.name} image Forge files differ from exact archive",
    )
    evidence.write_bytes(
        f"subjects/{subject.name}/image-forge-manifest.json",
        canonical_json(image_manifest),
    )
    return {
        "cli_help": "PASS",
        "forge_byte_manifest": "PASS",
        "offline_runtime_inventory": "PASS",
        "python_imports": "PASS",
        "runtime_inputs": "PASS",
    }


def observe_host_gpu(
    mode: str,
    tools: Mapping[str, str],
) -> dict[str, Any]:
    if mode == CPU_INTEGRATION_MODE:
        return {
            "accelerator_identity": CPU_ACCELERATOR_IDENTITY,
            "claim": "none",
            "reason": "physical-gpu-boundary-stubbed-for-cpu-integration",
            "state": "STUBBED",
        }
    output = run_capture(
        [
            tools["nvidia_smi"],
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    ).stdout.decode("utf-8", errors="strict")
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    require(len(rows) == 1, "nvidia-smi must return exactly one GPU row")
    try:
        fields = next(csv.reader([rows[0]], skipinitialspace=True))
    except Exception as exc:
        raise DelegateError("nvidia-smi GPU row is malformed") from exc
    require(len(fields) == 4, "nvidia-smi GPU row has an unexpected shape")
    name, uuid, driver, memory = (field.strip() for field in fields)
    require("H100" in name, "GPU 0 is not an H100")
    require(re.fullmatch(r"GPU-[0-9A-Fa-f-]+", uuid) is not None, "GPU UUID is invalid")
    require(re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", driver) is not None, "GPU driver is invalid")
    require(memory.isdigit() and int(memory) > 0, "GPU memory is invalid")
    accelerator_identity = f"{name}|{int(memory)}-MiB"
    require(len(accelerator_identity) <= 256, "GPU identity is too long")
    return {
        "accelerator_identity": accelerator_identity,
        "gpu_0": rows[0],
        "state": "PASS",
    }


def observe_image_gpu(
    mode: str,
    tools: Mapping[str, str],
    image_tag: str,
) -> dict[str, Any]:
    if mode == CPU_INTEGRATION_MODE:
        return {
            "claim": "none",
            "reason": "physical-container-gpu-boundary-stubbed-for-cpu-integration",
            "state": "STUBBED",
        }
    argv = [
        tools["docker"],
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--gpus",
        "device=0",
        "--entrypoint",
        "python3",
        image_tag,
        "-c",
        GPU_IMPORT_PROGRAM,
    ]
    completed = run_capture(argv, timeout=600)
    try:
        value = json.loads(completed.stdout)
    except Exception as exc:
        raise DelegateError("container GPU observation is not valid JSON") from exc
    require(isinstance(value, dict) and value.get("state") == "PASS", "container GPU import failed")
    require("H100" in str(value.get("device", "")), "container GPU is not H100")
    return value


def result_state_for_mode(mode: str) -> str:
    if mode == PRODUCTION_MODE:
        return PASS_STATE
    if mode == CPU_INTEGRATION_MODE:
        return DRY_RUN_PASS_STATE
    raise DelegateError(f"unknown release delegate mode: {mode}")


def result_env_payload(result: Mapping[str, str]) -> bytes:
    required_order = (
        "schema",
        "state",
        "mode",
        "certificate_scope",
        "source_commit",
        "source_tree",
        "forge_tree",
        "source_archive_sha256",
        "source_manifest_sha256",
        "production_manifest_sha256",
        "toolkit_dockerfile_sha256",
        "legacy_dockerfile_sha256",
        "toolkit_image_tag",
        "toolkit_image_id",
        "legacy_image_tag",
        "legacy_image_id",
        "gpu_boundary",
        "accelerator_identity",
        "completed_at_utc",
    )
    require(set(result) == set(required_order), "delegated result fields differ")
    rows: list[str] = []
    for field in required_order:
        value = result[field]
        require(value and "\n" not in value and "\r" not in value, f"invalid result value: {field}")
        rows.append(f"{field}={value}\n")
    return "".join(rows).encode("utf-8")


def validate_args(args: argparse.Namespace) -> SourceIdentity:
    require(args.mode in {PRODUCTION_MODE, CPU_INTEGRATION_MODE}, "invalid delegate mode")
    require(SHA1_RE.fullmatch(args.release_commit) is not None, "invalid release commit")
    require(SHA1_RE.fullmatch(args.release_tree) is not None, "invalid release tree")
    require(SHA1_RE.fullmatch(args.forge_tree) is not None, "invalid Forge tree")
    require(NAMESPACE_RE.fullmatch(args.evidence_namespace) is not None, "invalid evidence namespace")
    require(args.certificate_scope == "toolkit-krea-only", "invalid certificate scope")
    require(IMAGE_TAG_RE.fullmatch(args.toolkit_image_tag) is not None, "invalid toolkit image tag")
    require(IMAGE_TAG_RE.fullmatch(args.legacy_image_tag) is not None, "invalid legacy image tag")
    require(args.toolkit_image_tag != args.legacy_image_tag, "image tags must be distinct")
    for name in ("expected_docker_root", "expected_containerd_root"):
        value = getattr(args, name)
        require(os.path.isabs(value) and os.path.normpath(value) == value, f"invalid {name}")
    return SourceIdentity(args.release_commit, args.release_tree, args.forge_tree)


def build_and_certify(args: argparse.Namespace, tools: Mapping[str, str]) -> tuple[Path, str]:
    identity = validate_args(args)
    source_checkout = safe_absolute_directory(args.source_checkout, "source checkout")
    work_base = safe_absolute_directory(args.work_base, "work base")
    evidence_base = Path(args.evidence_base)
    require(evidence_base.is_absolute(), "evidence base must be absolute")
    verify_source_repository(source_checkout, identity, tools)

    evidence = AtomicEvidence(evidence_base, args.evidence_namespace)
    working_directory = Path(tempfile.mkdtemp(prefix="sn56-week6-build-cert-", dir=work_base))
    os.chmod(working_directory, 0o700)
    try:
        evidence.write_json(
            "authority-inputs.json",
            {
                "certificate_scope": args.certificate_scope,
                "forge_tree": identity.forge_tree,
                "mode": args.mode,
                "schema": 1,
                "source_commit": identity.commit,
                "source_tree": identity.tree,
                "state": "BOUND",
                "subjects": [
                    {"dockerfile": item.dockerfile, "name": item.name}
                    for item in SUBJECTS
                ],
            },
        )
        host = verify_host_contract(args, tools)
        evidence.write_json("host-contract.json", host)
        source = materialize_exact_archive(
            source_checkout, identity, working_directory, tools
        )
        evidence.write_json(
            "source-materialization.json",
            {
                "archive_sha256": source.archive_sha256,
                "file_manifest_sha256": source.file_manifest_sha256,
                "forge_tree": identity.forge_tree,
                "production_manifest_sha256": source.production_manifest_sha256,
                "source_commit": identity.commit,
                "source_tree": identity.tree,
                "state": "PASS",
            },
        )
        for required_path in (LOCK_REL, CONSTRAINTS_REL, VERIFIER_REL):
            require(required_path in source.file_hashes, f"release input is absent: {required_path}")
        source_manifest = source_forge_manifest(source.root)
        evidence.write_bytes("source-forge-manifest.json", canonical_json(source_manifest))

        host_gpu = observe_host_gpu(args.mode, tools)
        evidence.write_json("physical-gpu-observation.json", host_gpu)
        subject_results: dict[str, Any] = {}
        tags = {
            "toolkit": args.toolkit_image_tag,
            "legacy": args.legacy_image_tag,
        }
        for subject in SUBJECTS:
            dockerfile = source.root / subject.dockerfile
            require(dockerfile.is_file() and not dockerfile.is_symlink(), "Dockerfile is absent")
            bases = validate_dockerfile_digest_pins(dockerfile)
            tag = tags[subject.name]
            require(not image_exists(tools, tag), f"refusing to overwrite image tag: {tag}")
            build_log = evidence.path(f"subjects/{subject.name}/docker-build.log")
            run_logged(
                docker_build_command(tools, subject, tag),
                build_log,
                cwd=source.root,
                timeout=args.build_timeout_seconds,
                pressure_paths={
                    "root": (Path("/"), ROOT_PRESSURE_FLOOR),
                    "work": (work_base, WORK_PRESSURE_FLOOR),
                    "evidence": (evidence.stage_path, EVIDENCE_PRESSURE_FLOOR),
                },
                pressure_log=evidence.path(
                    f"subjects/{subject.name}/build-pressure.tsv"
                ),
            )
            built_id = image_id(tools, tag)
            inspect = run_capture([tools["docker"], "image", "inspect", tag])
            evidence.write_bytes(f"subjects/{subject.name}/image-inspect.json", inspect.stdout)
            cpu_gates = certify_cpu_image_surface(
                tools, source, subject, tag, evidence
            )
            gpu_gate = observe_image_gpu(args.mode, tools, tag)
            evidence.write_json(f"subjects/{subject.name}/gpu-observation.json", gpu_gate)
            require(not docker_ps_ids(tools), f"{subject.name} gates left a running container")
            subject_results[subject.name] = {
                "base_images": bases,
                "cpu_gates": cpu_gates,
                "dockerfile": subject.dockerfile,
                "dockerfile_sha256": source.file_hashes[subject.dockerfile],
                "gpu_gate": gpu_gate,
                "image_id": built_id,
                "image_tag": tag,
                "state": result_state_for_mode(args.mode),
            }
        evidence.write_json("subject-results.json", subject_results)

        state = result_state_for_mode(args.mode)
        gpu_boundary = "REAL_H100" if args.mode == PRODUCTION_MODE else "STUBBED_NO_CLAIM"
        accelerator_identity = str(host_gpu.get("accelerator_identity", ""))
        require(accelerator_identity, "host GPU observation lacks accelerator identity")
        result = {
            "schema": RESULT_SCHEMA,
            "state": state,
            "mode": args.mode,
            "certificate_scope": args.certificate_scope,
            "source_commit": identity.commit,
            "source_tree": identity.tree,
            "forge_tree": identity.forge_tree,
            "source_archive_sha256": source.archive_sha256,
            "source_manifest_sha256": source.file_manifest_sha256,
            "production_manifest_sha256": source.production_manifest_sha256,
            "toolkit_dockerfile_sha256": source.file_hashes[SUBJECTS[0].dockerfile],
            "legacy_dockerfile_sha256": source.file_hashes[SUBJECTS[1].dockerfile],
            "toolkit_image_tag": subject_results["toolkit"]["image_tag"],
            "toolkit_image_id": subject_results["toolkit"]["image_id"],
            "legacy_image_tag": subject_results["legacy"]["image_tag"],
            "legacy_image_id": subject_results["legacy"]["image_id"],
            "gpu_boundary": gpu_boundary,
            "accelerator_identity": accelerator_identity,
            "completed_at_utc": utc_now(),
        }
        evidence.write_bytes("result.env", result_env_payload(result))
        published = evidence.publish()
        return published, state
    except Exception as exc:
        try:
            evidence.write_bytes(
                "failure.txt",
                (f"{type(exc).__name__}: {exc}\n").encode("utf-8", errors="replace"),
            )
            failure = {
                "certificate_scope": args.certificate_scope,
                "failed_at_utc": utc_now(),
                "forge_tree": identity.forge_tree,
                "mode": args.mode,
                "schema": RESULT_SCHEMA,
                "source_commit": identity.commit,
                "source_tree": identity.tree,
                "state": "FAIL",
            }
            evidence.write_json("failure.json", failure)
            evidence.publish()
        except Exception:
            evidence.discard_unpublished()
        if isinstance(exc, DelegateError):
            raise
        raise DelegateError(f"unexpected delegate failure: {exc}") from exc
    finally:
        evidence.close()
        shutil.rmtree(working_directory, ignore_errors=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--mode", required=True, choices=(PRODUCTION_MODE, CPU_INTEGRATION_MODE))
    result.add_argument("--source-checkout", required=True)
    result.add_argument("--release-commit", required=True)
    result.add_argument("--release-tree", required=True)
    result.add_argument("--forge-tree", required=True)
    result.add_argument("--certificate-scope", required=True)
    result.add_argument("--evidence-base", required=True)
    result.add_argument("--evidence-namespace", required=True)
    result.add_argument("--work-base", required=True)
    result.add_argument("--toolkit-image-tag", required=True)
    result.add_argument("--legacy-image-tag", required=True)
    result.add_argument("--expected-docker-root", required=True)
    result.add_argument("--expected-containerd-root", required=True)
    result.add_argument("--build-timeout-seconds", type=int, default=10_800)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    sys.dont_write_bytecode = True
    args = parser().parse_args(argv)
    try:
        published, state = build_and_certify(args, ABSOLUTE_TOOLS)
    except DelegateError as exc:
        print(f"SN56_WEEK6_BUILD_GPU_CERT=FAIL reason={exc}", file=sys.stderr)
        return 1
    print(
        f"SN56_WEEK6_BUILD_GPU_CERT={state} evidence={published}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
