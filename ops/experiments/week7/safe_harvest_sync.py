#!/usr/bin/env python3
"""Create-only, read-only synchronizer for the SN56 tournament watcher archive.

This program deliberately does *not* use rsync, scp, tar, or ``Path.resolve``.
Those conveniences either copy more than was selected or can follow a link in
the source tree.  Instead it inventories regular files without following
links, reads each selected file through ``O_NOFOLLOW``, and publishes only
hash-verified observations and the CAS objects they reference.

Security / evidence invariants
------------------------------

* The source is read-only.  Local reads use file descriptors opened with
  ``O_NOFOLLOW``.  SSH mode executes a fixed Python reader that only supports
  ``list`` and ``read`` and never opens a remote path for writing.
* The destination is create-only.  Existing objects are accepted only after
  byte/hash verification; no file is replaced or truncated.
* ``latest/``, partials, locks, and all other mutable watcher machinery are
  ignored.  Immutable ``observations/**/*.json`` and reference-reachable CAS
  objects are the only primary evidence copied.
* HF-derived paths and bodies are screened before publication.  Anything
  matching hidden/holdout/test/quarantine/eval-derived terminology is omitted
  as a policy exclusion.  Its path and bytes are not repeated in the ledger.
* Mutable events and SQLite state are captured under a UTC timestamp, never at
  a stable name.  They receive the same content screen and may be excluded.
* Every CAS filename is its SHA-256.  Both new and pre-existing destination
  objects are rehashed before the run can be COMPLETE.

The filter is intentionally conservative.  A false positive loses public
intel; a false negative risks copying material the P0 directive forbids.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import unicodedata
from typing import Any, Iterable, Iterator, Mapping, Protocol
from urllib.parse import unquote
import uuid


SCHEMA = "sn56.week7.safe-harvest-sync"
SCHEMA_VERSION = 2
SHA256_RE = re.compile(r"[0-9a-f]{64}")
TOURNAMENT_RE = re.compile(r"tourn_[a-z0-9]+_[0-9]{8}")
TASK_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# Paths are a strict namespace boundary. A standalone ``test`` component is
# excluded there. Bodies are more precise: public scorer telemetry legitimately
# uses ``test_loss`` and must remain harvestable, while any name suggesting
# actual evaluation rows/data remains prohibited.
FORBIDDEN_PATH_RE = re.compile(
    r"(?<![a-z0-9])(?:hidden|holdout|test(?![-_\s]*loss(?:$|[^a-z0-9]))|"
    r"quarantine|eval(?:[-_\s]*derived))(?![a-z0-9])",
    re.IGNORECASE,
)
FORBIDDEN_BODY_RE = re.compile(
    r"(?<![a-z0-9])(?:"
    r"hidden|holdout|quarantine|"
    r"test[-_\s]*(?:data|rows?|set)|"
    r"eval[-_\s]*(?:data|derived)|evaluation[-_\s]*data"
    r")(?![a-z0-9])",
    re.IGNORECASE,
)
OBSERVATION_RE = re.compile(r"^observations/.+\.json$")
MUTABLE_FILES = (
    "events.jsonl",
    ".state/state.sqlite3",
    ".state/state.sqlite3-wal",
    ".state/state.sqlite3-shm",
)
CHUNK = 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024 * 1024


class SyncError(RuntimeError):
    """A fail-closed synchronizer error."""


class IntegrityError(SyncError):
    """Source or destination bytes did not satisfy their identity contract."""


class UnsafePathError(SyncError):
    """A path was absolute, traversing, linked, or otherwise unsafe."""


class PolicyExcluded(SyncError):
    """Public evidence matched the owner-declared exclusion vocabulary."""


@dataclasses.dataclass(frozen=True)
class SourceEntry:
    path: str
    kind: str
    size: int
    mtime_ns: int


class Source(Protocol):
    label: str

    def inventory(self) -> list[SourceEntry]: ...

    def iter_bytes(self, relative: str) -> Iterator[bytes]: ...


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def utc_token(value: dt.datetime) -> str:
    value = value.astimezone(dt.timezone.utc)
    return value.strftime("%Y%m%dT%H%M%S.%fZ")


def utc_iso(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def checked_relative(value: str) -> PurePosixPath:
    if "\x00" in value or "\\" in value:
        raise UnsafePathError("path contains a forbidden character")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafePathError("path is not a normalized relative POSIX path")
    return path


def _check_root_components(root: Path) -> Path:
    """Require every already-existing source-root component to be a real directory."""
    absolute = root.absolute()
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        info = cursor.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise UnsafePathError("source root contains a symlink or non-directory component")
    return absolute


def _open_regular_beneath(root: Path, relative: str) -> int:
    parts = checked_relative(relative).parts
    current = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=current,
            )
            os.close(current)
            current = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=current)
    finally:
        os.close(current)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        raise UnsafePathError("selected source is not a regular file")
    return fd


class LocalSource:
    """A local/mounted watcher archive, opened strictly read-only."""

    def __init__(self, root: Path, label: str | None = None):
        self.root = _check_root_components(root)
        self.label = label or str(self.root)

    def inventory(self) -> list[SourceEntry]:
        result: list[SourceEntry] = []

        def descend(directory_fd: int, prefix: PurePosixPath | None = None) -> None:
            # scandir/openat stay anchored to the already-open directory.  A
            # concurrent rename or link swap therefore cannot redirect this
            # traversal outside the selected archive.
            with os.scandir(directory_fd) as rows:
                for row in sorted(rows, key=lambda item: item.name):
                    relative = PurePosixPath(row.name) if prefix is None else prefix / row.name
                    info = row.stat(follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        result.append(SourceEntry(relative.as_posix(), "symlink", 0, info.st_mtime_ns))
                    elif stat.S_ISDIR(info.st_mode):
                        child = os.open(
                            row.name,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                            dir_fd=directory_fd,
                        )
                        try:
                            descend(child, relative)
                        finally:
                            os.close(child)
                    elif stat.S_ISREG(info.st_mode):
                        result.append(
                            SourceEntry(relative.as_posix(), "file", info.st_size, info.st_mtime_ns)
                        )
                    else:
                        result.append(SourceEntry(relative.as_posix(), "other", 0, info.st_mtime_ns))

        root_fd = os.open(
            self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        try:
            descend(root_fd)
        finally:
            os.close(root_fd)
        return result

    def iter_bytes(self, relative: str) -> Iterator[bytes]:
        fd = _open_regular_beneath(self.root, relative)
        try:
            while True:
                body = os.read(fd, CHUNK)
                if not body:
                    return
                yield body
        finally:
            os.close(fd)


_REMOTE_READER = r'''
import json, os, stat, sys

def checked_rel(value):
    if not value or value.startswith('/') or '\\' in value or '\x00' in value:
        raise ValueError('bad relative path')
    parts = value.split('/')
    if any(part in ('', '.', '..') for part in parts):
        raise ValueError('bad relative path')
    return parts

def open_root(root):
    if not root.startswith('/'):
        raise ValueError('remote root must be absolute')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in [p for p in root.split('/') if p]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                            dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except Exception:
        os.close(fd)
        raise

def open_beneath(root_fd, relative):
    fd = os.dup(root_fd)
    try:
        parts = checked_rel(relative)
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                            dir_fd=fd)
            os.close(fd)
            fd = child
        out = os.open(parts[-1], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=fd)
    finally:
        os.close(fd)
    if not stat.S_ISREG(os.fstat(out).st_mode):
        os.close(out)
        raise ValueError('not regular')
    return out

def inventory(dir_fd, prefix=''):
    with os.scandir(dir_fd) as rows:
        for row in sorted(rows, key=lambda item: item.name):
            info = row.stat(follow_symlinks=False)
            rel = row.name if not prefix else prefix + '/' + row.name
            if stat.S_ISLNK(info.st_mode):
                print(json.dumps({'path': rel, 'kind': 'symlink', 'size': 0,
                                  'mtime_ns': info.st_mtime_ns}), flush=True)
            elif stat.S_ISDIR(info.st_mode):
                child = os.open(row.name, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                                dir_fd=dir_fd)
                try:
                    inventory(child, rel)
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                print(json.dumps({'path': rel, 'kind': 'file', 'size': info.st_size,
                                  'mtime_ns': info.st_mtime_ns}), flush=True)
            else:
                print(json.dumps({'path': rel, 'kind': 'other', 'size': 0,
                                  'mtime_ns': info.st_mtime_ns}), flush=True)

root_fd = open_root(sys.argv[2])
try:
    if sys.argv[1] == 'list':
        inventory(root_fd)
    elif sys.argv[1] == 'read':
        fd = open_beneath(root_fd, sys.argv[3])
        try:
            while True:
                block = os.read(fd, 1024 * 1024)
                if not block:
                    break
                sys.stdout.buffer.write(block)
        finally:
            os.close(fd)
    else:
        raise ValueError('unknown operation')
finally:
    os.close(root_fd)
'''


class SSHSource:
    """Read-only SSH transport backed by the fixed descriptor-safe helper above."""

    def __init__(self, host: str, root: str, *, ssh_bin: str = "/usr/bin/ssh"):
        if not root.startswith("/"):
            raise UnsafePathError("remote root must be absolute")
        self.host = host
        self.root = root
        self.ssh_bin = ssh_bin
        self.label = f"ssh://{host}{root}"
        encoded = base64.b64encode(_REMOTE_READER.encode("utf-8")).decode("ascii")
        self._prefix = (
            "/usr/bin/python3 -I -c "
            + shlex.quote(f"import base64;exec(base64.b64decode({encoded!r}))")
        )

    def _command(self, operation: str, relative: str | None = None) -> list[str]:
        remote = " ".join(
            shlex.quote(value)
            for value in (self._prefix, operation, self.root, *(tuple() if relative is None else (relative,)))
        )
        # _prefix is itself a command fragment and must not be quoted as one token.
        remote = self._prefix + " " + " ".join(
            shlex.quote(value)
            for value in (operation, self.root, *(tuple() if relative is None else (relative,)))
        )
        return [
            self.ssh_bin,
            "-T",
            "-oBatchMode=yes",
            "-oClearAllForwardings=yes",
            "-oRequestTTY=no",
            self.host,
            remote,
        ]

    def inventory(self) -> list[SourceEntry]:
        run = subprocess.run(self._command("list"), capture_output=True, check=False)
        if run.returncode != 0:
            raise SyncError(f"remote inventory failed with exit {run.returncode}")
        rows: list[SourceEntry] = []
        for line in run.stdout.splitlines():
            try:
                item = json.loads(line)
                rows.append(
                    SourceEntry(
                        path=checked_relative(item["path"]).as_posix(),
                        kind=str(item["kind"]),
                        size=int(item["size"]),
                        mtime_ns=int(item["mtime_ns"]),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise IntegrityError("remote inventory returned a malformed row") from exc
        return sorted(rows, key=lambda row: row.path)

    def iter_bytes(self, relative: str) -> Iterator[bytes]:
        checked_relative(relative)
        process = subprocess.Popen(self._command("read", relative), stdout=subprocess.PIPE)
        assert process.stdout is not None
        try:
            while True:
                body = process.stdout.read(CHUNK)
                if not body:
                    break
                yield body
        finally:
            process.stdout.close()
        if process.wait() != 0:
            raise SyncError("remote read failed")


def normalized_sensitive_text(value: str) -> str:
    # Decode URL escaping twice: HF paths are sometimes embedded inside a URL
    # which is itself embedded in JSON.
    return unicodedata.normalize("NFKC", unquote(unquote(value))).lower()


def contains_forbidden_path(value: str) -> bool:
    return FORBIDDEN_PATH_RE.search(normalized_sensitive_text(value)) is not None


def contains_forbidden_body_text(value: str) -> bool:
    return FORBIDDEN_BODY_RE.search(normalized_sensitive_text(value)) is not None


def contains_forbidden_body(value: bytes) -> bool:
    text = value.decode("utf-8", errors="ignore")
    if contains_forbidden_body_text(text):
        return True
    try:
        decoded = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False

    def visit(item: Any) -> bool:
        if isinstance(item, str):
            return contains_forbidden_body_text(item)
        if isinstance(item, Mapping):
            return any(visit(key) or visit(child) for key, child in item.items())
        if isinstance(item, list):
            return any(visit(child) for child in item)
        return False

    return visit(decoded)


def is_hf_observation(relative: str, record: Mapping[str, Any]) -> bool:
    source = str(record.get("source", "")).lower()
    key = str(record.get("key", "")).lower()
    url = normalized_sensitive_text(str(record.get("request_url", "")))
    return (
        relative.lower().startswith("observations/hf-")
        or source.startswith("hf-")
        or key.startswith("hf-")
        or "huggingface.co/" in url
        or "hf.co/" in url
    )


def _ensure_destination_directory(path: Path) -> None:
    absolute = path.absolute()
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            cursor.mkdir(mode=0o700)
            info = cursor.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise UnsafePathError("destination path contains a symlink or non-directory")


@dataclasses.dataclass
class Staged:
    path: Path
    digest: str
    size: int

    def discard(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


class Publisher:
    def __init__(self, root: Path):
        self.root = root.absolute()
        _ensure_destination_directory(self.root)
        self.partial = self.root / ".partial"
        _ensure_destination_directory(self.partial)

    def stage(
        self,
        source: Source,
        entry: SourceEntry,
        *,
        expected_sha256: str | None = None,
        screen_body: bool = False,
    ) -> Staged:
        name = self.partial / f"{uuid.uuid4().hex}.part"
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        digest = hashlib.sha256()
        size = 0
        tail = b""
        excluded = False
        try:
            with os.fdopen(fd, "wb", buffering=0) as handle:
                for block in source.iter_bytes(entry.path):
                    size += len(block)
                    if size > entry.size:
                        raise IntegrityError("source grew after inventory")
                    digest.update(block)
                    if screen_body:
                        window = tail + block
                        if contains_forbidden_body(window):
                            excluded = True
                            raise PolicyExcluded("body matched the exclusion vocabulary")
                        tail = window[-256:]
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
            if size != entry.size:
                raise IntegrityError("source size changed after inventory")
            actual = digest.hexdigest()
            if expected_sha256 is not None and actual != expected_sha256:
                raise IntegrityError("CAS content does not match its SHA-256 identity")
            return Staged(name, actual, size)
        except Exception:
            try:
                name.unlink()
            except FileNotFoundError:
                pass
            raise
        finally:
            # Quiet a linter's false-positive about the intentionally local flag;
            # the exception type, not this variable, carries the policy result.
            del excluded

    def stage_bytes(self, body: bytes) -> Staged:
        name = self.partial / f"{uuid.uuid4().hex}.part"
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "wb", buffering=0) as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            return Staged(name, sha256_bytes(body), len(body))
        except Exception:
            try:
                name.unlink()
            except FileNotFoundError:
                pass
            raise

    def publish(self, staged: Staged, relative: str) -> tuple[str, bool]:
        rel = checked_relative(relative)
        destination = self.root.joinpath(*rel.parts)
        _ensure_destination_directory(destination.parent)
        try:
            os.link(staged.path, destination, follow_symlinks=False)
            created = True
        except FileExistsError:
            fd = _open_regular_beneath(self.root, relative)
            digest = hashlib.sha256()
            size = 0
            try:
                while True:
                    block = os.read(fd, CHUNK)
                    if not block:
                        break
                    size += len(block)
                    digest.update(block)
            finally:
                os.close(fd)
            if size != staged.size or digest.hexdigest() != staged.digest:
                raise IntegrityError("create-only destination already exists with different bytes")
            created = False
        return relative, created


def _read_metadata(source: Source, entry: SourceEntry) -> bytes:
    if entry.size > MAX_METADATA_BYTES:
        raise IntegrityError("metadata exceeds the 64 MiB safety ceiling")
    body = b"".join(source.iter_bytes(entry.path))
    if len(body) != entry.size:
        raise IntegrityError("metadata size changed after inventory")
    return body


def _read_inventory_prefix(source: Source, entry: SourceEntry) -> bytes:
    """Read exactly the bytes present when a live append-only file was inventoried.

    ``events.jsonl`` can legitimately grow between inventory and read.  Its
    inventory-time prefix is a stable point-in-time snapshot; bytes appended
    later belong to the next sync.  A shrink still fails closed.
    """
    body = bytearray()
    for block in source.iter_bytes(entry.path):
        remaining = entry.size - len(body)
        if remaining > 0:
            body.extend(block[:remaining])
    if len(body) != entry.size:
        raise IntegrityError("append-only source shrank after inventory")
    return bytes(body)


def _object_relative(digest: str) -> str:
    if SHA256_RE.fullmatch(digest) is None:
        raise IntegrityError("observation contains an invalid SHA-256")
    return f"objects/sha256/{digest[:2]}/{digest}"


def _nested_hf_objects(value: Any) -> list[tuple[str, str | None]]:
    """Return raw HF file CAS references and their nearest declared path."""
    found: list[tuple[str, str | None]] = []

    def visit(item: Any, inherited_path: str | None = None) -> None:
        if isinstance(item, Mapping):
            path = next(
                (
                    str(item[key])
                    for key in ("path", "rfilename", "filename")
                    if isinstance(item.get(key), str)
                ),
                inherited_path,
            )
            digest = item.get("object_sha256")
            if isinstance(digest, str):
                found.append((digest, path))
            for child in item.values():
                visit(child, path)
        elif isinstance(item, list):
            for child in item:
                visit(child, inherited_path)

    visit(value)
    return found


@dataclasses.dataclass(frozen=True)
class TournamentScope:
    tournament_id: str
    task_ids: frozenset[str]
    source_observation: str
    source_content_sha256: str


def _source_cas_bytes(
    source: Source, by_path: Mapping[str, SourceEntry], digest: str
) -> bytes:
    relative = _object_relative(digest)
    entry = by_path.get(relative)
    if entry is None or entry.kind != "file":
        raise IntegrityError("observation references a missing/non-regular CAS object")
    body = _read_metadata(source, entry)
    if sha256_bytes(body) != digest:
        raise IntegrityError("CAS content does not match its SHA-256 identity")
    return body


def _task_ids(value: Any) -> frozenset[str]:
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            task_id = item.get("task_id")
            if isinstance(task_id, str) and TASK_RE.fullmatch(task_id):
                found.add(task_id)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return frozenset(found)


def derive_tournament_scope(
    source: Source,
    entries: Iterable[SourceEntry],
    tournament_id: str,
) -> TournamentScope:
    """Bind a scope to the latest exact tournament observation in the archive."""
    if TOURNAMENT_RE.fullmatch(tournament_id) is None:
        raise SyncError("--tournament-id is not a canonical tournament ID")
    by_path = {entry.path: entry for entry in entries}
    prefix = f"observations/gradients-tournament/{tournament_id}/"
    candidates = sorted(
        entry.path
        for entry in entries
        if entry.kind == "file" and entry.path.startswith(prefix) and entry.path.endswith(".json")
    )
    if not candidates:
        raise SyncError("exact gradients-tournament observation is absent")
    latest = candidates[-1]
    snapshot = _read_metadata(source, by_path[latest])
    try:
        record = json.loads(snapshot)
    except json.JSONDecodeError as exc:
        raise IntegrityError("tournament observation snapshot is not JSON") from exc
    if (
        not isinstance(record, dict)
        or record.get("source") != "gradients-tournament"
        or record.get("key") != tournament_id
    ):
        raise IntegrityError("tournament observation identity does not match its path")
    digest = record.get("content_sha256")
    if not isinstance(digest, str) or record.get("object") != _object_relative(digest):
        raise IntegrityError("tournament observation has an invalid CAS binding")
    body = _source_cas_bytes(source, by_path, digest)
    if contains_forbidden_body(body):
        raise PolicyExcluded("tournament observation matched prohibited dataset-row terminology")
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise IntegrityError("tournament observation body is not JSON") from exc
    if not isinstance(value, dict) or value.get("tournament_id") != tournament_id:
        raise IntegrityError("tournament body does not assert the selected tournament ID")
    tasks = _task_ids(value)
    if not tasks:
        raise IntegrityError("selected tournament observation contains no task IDs")
    return TournamentScope(tournament_id, tasks, latest, digest)


def observation_in_scope(relative: str, scope: TournamentScope) -> bool:
    """Select only exact tournament/task/fixture/HF observation namespaces."""
    parts = checked_relative(relative).parts
    if len(parts) < 4 or parts[0] != "observations":
        return False
    source = parts[1]
    key_head = parts[2]
    if source in {"gradients-tournament", "acceptance"}:
        return key_head == scope.tournament_id
    if source == "gradients-task":
        return key_head in scope.task_ids
    if source == "fixtures":
        return key_head in scope.task_ids
    if source.startswith("hf-") and source != "hf-listing":
        # HF repo IDs are path components ``author/repo`` and the repo name is
        # tournament-<exact tournament>-<exact task>-<hotkey>. Requiring both
        # IDs prevents a date-wide listing from smuggling text/environment
        # artifacts into an image-tournament root.
        marker = f"tournament-{scope.tournament_id}-"
        return marker in relative and any(f"-{task_id}-" in relative for task_id in scope.task_ids)
    return False


def _read_destination_file(root: Path, relative: str) -> bytes:
    fd = _open_regular_beneath(root, relative)
    try:
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, CHUNK)
            if not block:
                return b"".join(chunks)
            chunks.append(block)
    finally:
        os.close(fd)


def bind_root_identity(destination: Path, source_label: str, tournament_id: str | None) -> str:
    """Create or verify the immutable source+tournament binding for a root."""
    identity = {
        "schema": "sn56.week7.harvest-root-identity",
        "schema_version": 1,
        "source": source_label,
        "tournament_id": tournament_id,
    }
    body = canonical_json(identity)
    digest = sha256_bytes(body)
    _ensure_destination_directory(destination)
    path = destination / "ROOT-IDENTITY.json"
    try:
        existing = _read_destination_file(destination, "ROOT-IDENTITY.json")
    except FileNotFoundError:
        with os.scandir(destination) as rows:
            unexpected = [row.name for row in rows if row.name != ".partial"]
        if unexpected:
            raise IntegrityError("non-empty destination has no root identity")
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        try:
            os.write(fd, body)
            os.fsync(fd)
        finally:
            os.close(fd)
    else:
        if existing != body:
            raise IntegrityError("destination root identity does not match source/tournament")
    return digest


def _filtered_events(body: bytes, scope: TournamentScope) -> tuple[bytes, int, int]:
    selected: list[bytes] = []
    out_of_scope = 0
    forbidden = 0
    needles = (scope.tournament_id, *sorted(scope.task_ids))
    for raw in body.splitlines():
        if not raw.strip():
            continue
        if contains_forbidden_body(raw):
            forbidden += 1
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            out_of_scope += 1
            continue
        canonical = json.dumps(value, ensure_ascii=True, sort_keys=True)
        if any(needle in canonical for needle in needles):
            selected.append(raw + b"\n")
        else:
            out_of_scope += 1
    return b"".join(selected), out_of_scope, forbidden


def sync_archive(
    source: Source,
    destination: Path,
    *,
    observed_at: dt.datetime,
    tournament_id: str | None = None,
) -> dict[str, Any]:
    """Synchronize one immutable observation frontier and one mutable snapshot."""
    stamp = utc_token(observed_at)
    entries = source.inventory()
    by_path = {entry.path: entry for entry in entries}
    if len(by_path) != len(entries):
        raise IntegrityError("source inventory contains duplicate paths")
    scope = derive_tournament_scope(source, entries, tournament_id) if tournament_id else None
    root_identity_sha256 = bind_root_identity(destination, source.label, tournament_id)
    publisher = Publisher(destination)

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "observed_at": utc_iso(observed_at),
        "source": source.label,
        "destination": str(destination.absolute()),
        "root_identity_sha256": root_identity_sha256,
        "scope": (
            {
                "tournament_id": scope.tournament_id,
                "task_ids": sorted(scope.task_ids),
                "derived_from_observation": scope.source_observation,
                "derived_from_content_sha256": scope.source_content_sha256,
            }
            if scope
            else None
        ),
        "policy": {
            "copy": "immutable observations plus reference-reachable SHA-256 CAS only",
            "mutable": "UTC-stamped snapshots only",
            "symlinks": "never followed or copied",
            "path_exclusion_terms": ["hidden", "holdout", "test", "quarantine", "eval-derived"],
            "body_exclusion_terms": [
                "hidden",
                "holdout",
                "quarantine",
                "test_data/test_rows/test_set",
                "eval_data/evaluation_data/eval-derived",
            ],
            "public_test_loss_telemetry": "allowed",
        },
        "inventory": {
            "files": sum(row.kind == "file" for row in entries),
            "symlinks": sum(row.kind == "symlink" for row in entries),
            "other": sum(row.kind == "other" for row in entries),
            "bytes": sum(row.size for row in entries if row.kind == "file"),
        },
        "observations": {"eligible": 0, "published": 0, "preexisting": 0},
        "cas": {"referenced": 0, "published": 0, "preexisting": 0, "bytes": 0},
        "mutable_snapshots": {"eligible": 0, "published": 0, "preexisting": 0},
        "excluded": {
            "out_of_scope": 0,
            "forbidden_path": 0,
            "forbidden_body": 0,
            "symlink": 0,
            "other": 0,
        },
        "errors": [],
        "files": [],
    }
    result["excluded"]["symlink"] = result["inventory"]["symlinks"]
    result["excluded"]["other"] = result["inventory"]["other"]
    published_cas: set[str] = set()

    def publish_cas(staged: Staged, digest: str) -> None:
        if digest in published_cas:
            staged.discard()
            return
        relative = _object_relative(digest)
        _, created = publisher.publish(staged, relative)
        result["cas"]["referenced"] += 1
        result["cas"]["published" if created else "preexisting"] += 1
        result["cas"]["bytes"] += staged.size
        result["files"].append(
            {"path": relative, "sha256": staged.digest, "bytes": staged.size, "created": created}
        )
        published_cas.add(digest)
        staged.discard()

    for entry in entries:
        if entry.kind != "file" or OBSERVATION_RE.fullmatch(entry.path) is None:
            continue
        if scope is not None and not observation_in_scope(entry.path, scope):
            result["excluded"]["out_of_scope"] += 1
            continue
        if contains_forbidden_path(entry.path):
            result["excluded"]["forbidden_path"] += 1
            continue
        group: list[Staged] = []
        try:
            snapshot_body = _read_metadata(source, entry)
            try:
                record = json.loads(snapshot_body)
            except json.JSONDecodeError as exc:
                raise IntegrityError("observation snapshot is not JSON") from exc
            if not isinstance(record, dict):
                raise IntegrityError("observation snapshot is not an object")
            hf = is_hf_observation(entry.path, record)
            if contains_forbidden_body(snapshot_body):
                raise PolicyExcluded("HF observation metadata matched exclusion vocabulary")
            digest = record.get("content_sha256")
            if not isinstance(digest, str):
                raise IntegrityError("observation has no content_sha256")
            object_relative = _object_relative(digest)
            if record.get("object") != object_relative:
                raise IntegrityError("observation object path does not match content_sha256")
            object_entry = by_path.get(object_relative)
            if object_entry is None or object_entry.kind != "file":
                raise IntegrityError("observation references a missing/non-regular CAS object")

            content_staged = publisher.stage(
                source, object_entry, expected_sha256=digest, screen_body=True
            )
            group.append(content_staged)
            content_value: Any = None
            if content_staged.size <= MAX_METADATA_BYTES:
                content_bytes = content_staged.path.read_bytes()
                try:
                    content_value = json.loads(content_bytes)
                except json.JSONDecodeError:
                    content_value = None
            for nested_digest, declared_path in _nested_hf_objects(content_value) if hf else []:
                if declared_path is None or contains_forbidden_path(declared_path):
                    raise PolicyExcluded("HF file path matched exclusion vocabulary or was absent")
                nested_relative = _object_relative(nested_digest)
                nested_entry = by_path.get(nested_relative)
                if nested_entry is None or nested_entry.kind != "file":
                    raise IntegrityError("HF manifest references a missing/non-regular CAS object")
                group.append(
                    publisher.stage(
                        source,
                        nested_entry,
                        expected_sha256=nested_digest,
                        screen_body=True,
                    )
                )

            snapshot_staged = publisher.stage_bytes(snapshot_body)
            group.append(snapshot_staged)
            result["observations"]["eligible"] += 1
            publish_cas(content_staged, digest)
            for (nested_digest, _), staged in zip(
                _nested_hf_objects(content_value) if hf else [], group[1:-1]
            ):
                publish_cas(staged, nested_digest)
            _, created = publisher.publish(snapshot_staged, entry.path)
            result["observations"]["published" if created else "preexisting"] += 1
            result["files"].append(
                {
                    "path": entry.path,
                    "sha256": snapshot_staged.digest,
                    "bytes": snapshot_staged.size,
                    "created": created,
                }
            )
            snapshot_staged.discard()
        except PolicyExcluded:
            result["excluded"]["forbidden_body"] += 1
        except (OSError, SyncError) as exc:
            result["errors"].append({"class": type(exc).__name__, "message": str(exc)})
        finally:
            for staged in group:
                staged.discard()

    if scope is None:
        for relative in MUTABLE_FILES:
            entry = by_path.get(relative)
            if entry is None:
                continue
            if entry.kind != "file":
                result["excluded"]["symlink" if entry.kind == "symlink" else "other"] += 1
                continue
            result["mutable_snapshots"]["eligible"] += 1
            staged: Staged | None = None
            try:
                staged = publisher.stage(source, entry, screen_body=True)
                snapshot_path = f"snapshots/{stamp}/{relative}"
                _, created = publisher.publish(staged, snapshot_path)
                result["mutable_snapshots"]["published" if created else "preexisting"] += 1
                result["files"].append(
                    {
                        "path": snapshot_path,
                        "sha256": staged.digest,
                        "bytes": staged.size,
                        "created": created,
                        "source_mtime_ns": entry.mtime_ns,
                    }
                )
            except PolicyExcluded:
                result["excluded"]["forbidden_body"] += 1
            except (OSError, SyncError) as exc:
                result["errors"].append({"class": type(exc).__name__, "message": str(exc)})
            finally:
                if staged is not None:
                    staged.discard()
    else:
        # A raw events log and SQLite DB are date-wide mutable surfaces.  In an
        # exact-tournament root, archive only matching event rows and a derived
        # state frontier; never copy the cross-tournament database bytes.
        events_entry = by_path.get("events.jsonl")
        if events_entry is not None and events_entry.kind == "file":
            result["mutable_snapshots"]["eligible"] += 1
            try:
                raw_events = _read_inventory_prefix(source, events_entry)
                event_bytes, filtered_rows, forbidden_rows = _filtered_events(raw_events, scope)
                result["excluded"]["out_of_scope"] += filtered_rows
                result["excluded"]["forbidden_body"] += forbidden_rows
                staged = publisher.stage_bytes(event_bytes)
                try:
                    snapshot_path = f"snapshots/{stamp}/events.filtered.jsonl"
                    _, created = publisher.publish(staged, snapshot_path)
                    result["mutable_snapshots"]["published" if created else "preexisting"] += 1
                    result["files"].append(
                        {
                            "path": snapshot_path,
                            "sha256": staged.digest,
                            "bytes": staged.size,
                            "created": created,
                            "source_mtime_ns": events_entry.mtime_ns,
                        }
                    )
                finally:
                    staged.discard()
            except (OSError, SyncError) as exc:
                result["errors"].append({"class": type(exc).__name__, "message": str(exc)})
        state = canonical_json(
            {
                "schema": "sn56.week7.filtered-state-frontier",
                "schema_version": 1,
                "observed_at": utc_iso(observed_at),
                "source": source.label,
                "tournament_id": scope.tournament_id,
                "task_ids": sorted(scope.task_ids),
                "tournament_observation": scope.source_observation,
                "tournament_content_sha256": scope.source_content_sha256,
                "raw_sqlite_copied": False,
                "reason": "raw watcher state is date-wide; exact-tournament roots use a derived frontier",
            }
        )
        staged = publisher.stage_bytes(state)
        try:
            snapshot_path = f"snapshots/{stamp}/state-selection.json"
            _, created = publisher.publish(staged, snapshot_path)
            result["mutable_snapshots"]["eligible"] += 1
            result["mutable_snapshots"]["published" if created else "preexisting"] += 1
            result["files"].append(
                {
                    "path": snapshot_path,
                    "sha256": staged.digest,
                    "bytes": staged.size,
                    "created": created,
                }
            )
        finally:
            staged.discard()

    # Unreferenced CAS is intentionally *not* copied.  This count proves the
    # completeness boundary without naming potentially forbidden objects.
    all_cas = {
        row.path
        for row in entries
        if row.kind == "file" and row.path.startswith("objects/sha256/")
    }
    result["cas"]["unreferenced_or_policy_excluded"] = len(all_cas - {
        item["path"] for item in result["files"] if item["path"].startswith("objects/sha256/")
    })
    result["files"].sort(key=lambda row: row["path"])
    result["status"] = "COMPLETE" if not result["errors"] else "PARTIAL"
    result["complete"] = not result["errors"]

    ledger_body = canonical_json(result)
    ledger = publisher.stage_bytes(ledger_body)
    ledger_path = f"ledgers/{stamp}.json"
    try:
        publisher.publish(ledger, ledger_path)
    finally:
        ledger.discard()
    checksum_body = f"{sha256_bytes(ledger_body)}  {stamp}.json\n".encode("ascii")
    checksum = publisher.stage_bytes(checksum_body)
    try:
        publisher.publish(checksum, f"ledgers/{stamp}.sha256")
    finally:
        checksum.discard()
    result["ledger_path"] = ledger_path
    result["ledger_sha256"] = sha256_bytes(ledger_body)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-root", type=Path)
    source.add_argument("--ssh-host")
    parser.add_argument("--remote-root", default="/opt/sn56-watcher/archive-20260810")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--tournament-id",
        help="exact tournament ID; derive and enforce its task/HF allowlist",
    )
    parser.add_argument(
        "--observed-at",
        help="UTC ISO-8601 observation time; default is the current time",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.source_root is not None:
        source: Source = LocalSource(args.source_root)
    else:
        source = SSHSource(args.ssh_host, args.remote_root)
    if args.observed_at:
        observed_at = dt.datetime.fromisoformat(args.observed_at.replace("Z", "+00:00"))
        if observed_at.tzinfo is None:
            raise SystemExit("--observed-at must include a timezone")
    else:
        observed_at = utc_now()
    result = sync_archive(
        source,
        args.destination,
        observed_at=observed_at,
        tournament_id=args.tournament_id,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
