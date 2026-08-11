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
* Request-query names are source-allowlisted. Accepted pagination/Xet values
  are discarded and only fixed, value-free redaction metadata is published;
  unexpected or unscoped queries fail closed.
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
import unicodedata
from typing import Any, Iterable, Iterator, Mapping, Protocol
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit
import uuid


SCHEMA = "sn56.week7.safe-harvest-sync"
SCHEMA_VERSION = 3
SHA256_RE = re.compile(r"[0-9a-f]{64}")
TOURNAMENT_RE = re.compile(r"tourn_[a-z0-9]+_[0-9]{8}")
TASK_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")
HF_REPOSITORY_OWNER_RE = re.compile(r"[1-9A-HJ-NP-Za-km-z]{8}")
HF_ORGANIZATION = "gradients-io-tournaments"
# Paths are a strict namespace boundary. A standalone ``test`` component is
# excluded there. Bodies are more precise: public scorer telemetry legitimately
# uses ``test_loss`` and must remain harvestable, while any name suggesting
# actual evaluation rows/data remains prohibited.
FORBIDDEN_PATH_RE = re.compile(
    r"(?<![a-z0-9])(?:"
    r"hidden|hold[-_\s]*outs?|quarantines?|"
    r"test[-_\s]*loss[-_\s]*(?:data|rows?|set|dataset|archive)|"
    r"test(?![-_\s]*loss(?:$|[^a-z0-9]))(?:s|ing)?"
    r"(?:[-_\s]*(?:data|rows?|set))?|"
    r"eval(?:uation)?(?:[-_\s]*(?:data|derived|rows?|set))?"
    r")(?![a-z0-9])",
    re.IGNORECASE,
)
FORBIDDEN_BODY_RE = re.compile(
    r"(?<![a-z0-9])(?:"
    r"hidden|hold[-_\s]*outs?|quarantines?|"
    r"test[-_\s]*loss[-_\s]*(?:data|rows?|set|datasets?|archives?)|"
    r"test[-_\s]*(?:data|rows?|set|datasets?|archives?|images?|assets?|prompts?)|"
    r"eval(?:uation)?[-_\s]*(?:data|derived|rows?|set|datasets?|archives?|images?|assets?|prompts?)"
    r")(?![a-z0-9])",
    re.IGNORECASE,
)
# Treat numeric and version suffixes as part of a prohibited dataset namespace.
# The main expressions above deliberately permit public ``test_loss`` telemetry;
# this independent expression closes identifiers such as ``test1``,
# ``holdout_v2`` and ``evaluation-version-3`` without broadening that exception
# to ordinary prose.
FORBIDDEN_VERSIONED_NAME_RE = re.compile(
    r"(?<![a-z0-9])(?:hidden|hold[-_\s]*outs?|quarantines?|"
    r"test|eval(?:uation)?)"
    r"(?:[-_\s]*(?:v(?:er(?:sion)?)?[-_\s]*)?)?[0-9]+(?![a-z0-9])",
    re.IGNORECASE,
)
OBSERVATION_RE = re.compile(r"^observations/.+\.json$")
OBSERVATION_FILE_RE = re.compile(
    r"([0-9]{8}T[0-9]{6}\.[0-9]{6}Z)-([0-9a-f]{12}|unchanged)\.json"
)
MUTABLE_FILES = (
    "events.jsonl",
    ".state/state.sqlite3",
    ".state/state.sqlite3-wal",
    ".state/state.sqlite3-shm",
)
ALLOWED_HF_OBSERVATION_SOURCES = frozenset(
    {"hf-model", "hf-revision-manifest", "hf-tree", "hf-file"}
)
PUBLIC_PROVENANCE_SOURCES = frozenset(
    {
        "acceptance",
        "gradients-task",
        "gradients-tournament",
        *ALLOWED_HF_OBSERVATION_SOURCES,
    }
)
CHUNK = 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024 * 1024

# Watcher wrappers normally arrive query-free because the watcher already
# redacts them.  These allowlists are defense in depth for externally produced
# wrappers: only query names needed for HF pagination or signed Xet transport
# may cross the validator, and their values are never published.
HF_TREE_QUERY_KEYS = frozenset({"cursor", "expand", "recursive"})
HF_XET_QUERY_KEYS = frozenset(
    {
        "expires",
        "key-pair-id",
        "policy",
        "response-content-disposition",
        "response-content-type",
        "signature",
        "x-amz-algorithm",
        "x-amz-content-sha256",
        "x-amz-credential",
        "x-amz-date",
        "x-amz-expires",
        "x-amz-security-token",
        "x-amz-signature",
        "x-amz-signedheaders",
        "x-id",
        "x-xet-cas-uid",
    }
)
HF_XET_SIGNATURE_KEYS = frozenset({"signature", "x-amz-signature"})
MAX_QUERY_BYTES = 16 * 1024
MAX_QUERY_FIELDS = 32
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_HF_XET_PATH_RE = re.compile(
    r"/xet-bridge-[a-z0-9-]+(?:/[A-Za-z0-9._~-]+)+",
    flags=re.ASCII,
)


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
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
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
        # A scoped multiplexing socket makes one exact harvest scale with the
        # number of immutable observations rather than paying a fresh SSH
        # handshake for every hash-verified CAS read.  The random, short /tmp
        # name is local-only, never changes the remote source, and expires
        # shortly after the collector stops.
        # macOS exposes a long per-user ``tempfile.gettempdir()`` path which,
        # once OpenSSH expands ``%C``, can exceed its 104-byte Unix-socket
        # ceiling.  ``/tmp`` is the stable short local namespace on both the
        # operator Mac and Linux; the random name plus OpenSSH's mode-0600
        # socket creation keeps this collector scoped to the current user.
        self.control_path = f"/tmp/sn56-w7-{uuid.uuid4().hex[:12]}-%C"
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
            "-oControlMaster=auto",
            "-oControlPersist=30",
            f"-oControlPath={self.control_path}",
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
    # Decode to a fixed point.  HF paths can be embedded in a URL which is
    # embedded in JSON, and an attacker can add arbitrarily many ``%25``
    # layers.  Two decoding passes therefore are not a security boundary.
    normalized = unicodedata.normalize("NFKC", value).lower()
    for _ in range(len(normalized) + 1):
        decoded = unicodedata.normalize("NFKC", unquote(normalized)).lower()
        if decoded == normalized:
            # Dataset-bearing names are identifiers, not prose.  Remove
            # default-ignorable/control characters that can split a token and
            # canonicalize punctuation, path separators, symbols and spacing
            # to one delimiter.  This makes ``test/data``, ``test.data``,
            # ``hold.out`` and ``te\u200bst_data`` equivalent to their ordinary
            # underscore forms while retaining the explicit ``test_loss``
            # telemetry exception in the matchers below.
            result: list[str] = []
            previous_separator = False
            for character in normalized:
                category = unicodedata.category(character)
                if category.startswith("C") or category in {"Mn", "Me"}:
                    continue
                if character.isdecimal():
                    # The prohibited-version matcher is intentionally ASCII so
                    # its grammar stays reviewable.  Canonicalize every Unicode
                    # decimal digit first; otherwise identifiers such as
                    # ``test\u0661`` bypass the ``[0-9]+`` suffix boundary.
                    result.append(str(unicodedata.decimal(character)))
                    previous_separator = False
                elif character.isalnum():
                    result.append(character)
                    previous_separator = False
                elif not previous_separator:
                    result.append("_")
                    previous_separator = True
            return "".join(result)
        normalized = decoded
    raise IntegrityError("sensitive text did not reach a decoding fixed point")


def _require_canonical_percent_encoding(value: str, *, label: str) -> None:
    """Reject malformed escapes and raw query ``+`` before URL decoding.

    ``urllib`` deliberately accepts malformed percent escapes and interprets a
    raw plus in a query as a space.  Neither behavior is suitable for an
    evidence boundary: after redaction, the accepted source spelling could no
    longer be reconstructed unambiguously.  Percent-encoded ``%2B`` remains a
    normal value byte and is discarded with every other credential value.
    """

    for index, character in enumerate(value):
        if character != "%":
            continue
        if (
            index + 2 >= len(value)
            or value[index + 1] not in _HEX_DIGITS
            or value[index + 2] not in _HEX_DIGITS
        ):
            raise IntegrityError(f"{label} contains malformed percent encoding")
    if label == "observation request query" and "+" in value:
        raise IntegrityError("observation request query contains ambiguous raw plus")


def _decode_path_to_fixed_point(value: str) -> str:
    """Decode a URL path completely so nested delimiters cannot hide syntax."""

    current = value
    for _ in range(8):
        _require_canonical_percent_encoding(
            current, label="observation request path"
        )
        decoded = unquote(current)
        if decoded == current:
            return decoded
        current = decoded
    raise IntegrityError("observation request path did not reach a decoding fixed point")


def _is_public_xet_path(value: str) -> bool:
    """Accept only object-path grammar; credentials belong in redacted query data."""

    folded = value.casefold()
    credential_keys = HF_XET_QUERY_KEYS | HF_XET_SIGNATURE_KEYS
    if any(f"{name}=" in folded for name in credential_keys):
        return False
    return _HF_XET_PATH_RE.fullmatch(value) is not None


def contains_forbidden_path(value: str) -> bool:
    normalized = normalized_sensitive_text(value)
    return (
        FORBIDDEN_PATH_RE.search(normalized) is not None
        or FORBIDDEN_VERSIONED_NAME_RE.search(normalized) is not None
    )


def contains_forbidden_body_text(value: str) -> bool:
    normalized = normalized_sensitive_text(value)
    return (
        FORBIDDEN_BODY_RE.search(normalized) is not None
        or FORBIDDEN_VERSIONED_NAME_RE.search(normalized) is not None
    )


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


def validate_observation_identity(
    relative: str, record: Mapping[str, Any]
) -> tuple[str, str]:
    """Bind a watcher wrapper's asserted source/key to its filesystem path."""
    observed = checked_relative(relative)
    if type(record.get("schema")) is not int or record.get("schema") != 1:
        raise IntegrityError("observation wrapper schema is not supported")
    parts = observed.parts
    if len(parts) < 4 or parts[0] != "observations":
        raise IntegrityError("observation path is outside the watcher namespace")
    source = record.get("source")
    key = record.get("key")
    if not isinstance(source, str) or source != parts[1]:
        raise IntegrityError("observation source does not match its path")
    if not isinstance(key, str):
        raise IntegrityError("observation key is absent")
    canonical_key = checked_relative(key).as_posix()
    if canonical_key != key:
        raise IntegrityError("observation key is not canonical")
    expected_parent = PurePosixPath("observations") / source / canonical_key
    if observed.parent != expected_parent:
        raise IntegrityError("observation key does not match its path")
    filename_match = OBSERVATION_FILE_RE.fullmatch(observed.name)
    if filename_match is None:
        raise IntegrityError("observation filename is not append-only watcher form")
    digest = record.get("content_sha256")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise IntegrityError("observation has no valid content SHA-256")
    suffix = filename_match.group(2)
    if suffix != "unchanged" and suffix != digest[:12]:
        raise IntegrityError("observation filename does not bind its content digest")
    observation_timestamp(relative, record)
    return source, canonical_key


def _redacted_request_query(
    record: Mapping[str, Any],
    parsed: Any,
    *,
    query_kind: str,
) -> dict[str, Any] | None:
    """Validate one source-specific query and return value-free metadata.

    Query values can contain temporary credentials.  They are accepted only
    for the two public transports that require them and are never copied into
    the destination wrapper, not even as hashes.
    """

    if query_kind == "hf-tree":
        allowed = HF_TREE_QUERY_KEYS
        required_signature = False
    elif query_kind == "hf-xet":
        allowed = HF_XET_QUERY_KEYS
        required_signature = True
    elif query_kind == "none":
        allowed = frozenset()
        required_signature = False
    else:  # pragma: no cover - internal closed enum
        raise IntegrityError("observation query contract is unknown")

    existing = record.get("request_query")
    if parsed.query:
        if existing is not None:
            raise IntegrityError("observation carries raw and redacted query metadata")
        if len(parsed.query.encode("utf-8")) > MAX_QUERY_BYTES:
            raise IntegrityError("observation request query exceeds its safety ceiling")
        _require_canonical_percent_encoding(
            parsed.query, label="observation request query"
        )
        try:
            pairs = parse_qsl(
                parsed.query,
                keep_blank_values=True,
                strict_parsing=True,
                max_num_fields=MAX_QUERY_FIELDS,
            )
        except ValueError as exc:
            raise IntegrityError("observation request query is malformed") from exc
        keys = [name.casefold() for name, _value in pairs]
        if (
            not pairs
            or any(not name or name not in allowed for name in keys)
            or len(keys) != len(set(keys))
            or any(not value or len(value.encode("utf-8")) > MAX_QUERY_BYTES for _, value in pairs)
        ):
            raise IntegrityError("observation request query violates its source allowlist")
        values = {name.casefold(): value for name, value in pairs}
        if query_kind == "hf-tree" and any(
            values[name].casefold() != "true"
            for name in ("expand", "recursive")
            if name in values
        ):
            raise IntegrityError("HF tree query flags are not canonical")
        if required_signature and not HF_XET_SIGNATURE_KEYS.intersection(keys):
            raise IntegrityError("signed Xet query has no signature field")
        return {
            "redacted": True,
            "keys": sorted(keys),
            "pair_count": len(keys),
        }

    if existing is None:
        return None
    if not isinstance(existing, Mapping) or set(existing) != {
        "redacted",
        "keys",
        "pair_count",
    }:
        raise IntegrityError("redacted request query metadata is malformed")
    keys_value = existing.get("keys")
    if (
        existing.get("redacted") is not True
        or not isinstance(keys_value, list)
        or not keys_value
        or not all(isinstance(name, str) and name == name.casefold() for name in keys_value)
        or keys_value != sorted(set(keys_value))
        or any(name not in allowed for name in keys_value)
        or type(existing.get("pair_count")) is not int
        or existing["pair_count"] != len(keys_value)
        or (required_signature and not HF_XET_SIGNATURE_KEYS.intersection(keys_value))
    ):
        raise IntegrityError("redacted request query metadata violates its source allowlist")
    return {
        "redacted": True,
        "keys": list(keys_value),
        "pair_count": existing["pair_count"],
    }


def validate_public_request_provenance(
    source: str, key: str, record: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind a watcher wrapper to its public endpoint and redact URL queries."""
    if type(record.get("status")) is not int or record.get("status") != 200:
        raise IntegrityError("observation does not bind a successful HTTP status")
    request_url = record.get("request_url")
    if not isinstance(request_url, str):
        raise IntegrityError("observation has no request URL provenance")
    if source == "acceptance":
        if request_url != f"local://acceptance/{key}":
            raise IntegrityError("acceptance observation URL provenance is invalid")
        if "request_query" in record:
            raise IntegrityError("acceptance observation has request query metadata")
        return dict(record)
    try:
        parsed = urlsplit(request_url)
        port = parsed.port
    except ValueError as exc:
        raise IntegrityError("observation request URL is malformed") from exc
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.fragment
    ):
        raise IntegrityError("observation request URL authority is invalid")
    if "?" in request_url and not parsed.query:
        raise IntegrityError("observation request query is malformed")
    path = _decode_path_to_fixed_point(parsed.path)
    # A percent-encoded question mark/hash in a Xet path can carry credential
    # syntax while ``urlsplit`` still reports an empty query/fragment.  Reject
    # decoded URL delimiters rather than later publishing their encoded bytes.
    if "?" in path or "#" in path:
        raise IntegrityError("observation request path contains decoded URL delimiters")
    expected_host: str
    expected_path: str
    if source == "gradients-tournament":
        expected_host = "api.gradients.io"
        expected_path = f"/tournament/{key}/details"
    elif source == "gradients-task":
        expected_host = "api.gradients.io"
        expected_path = f"/auditing/tasks/{key}"
    elif source == "hf-model":
        expected_host = "huggingface.co"
        expected_path = f"/api/models/{key}"
    elif source == "hf-revision-manifest":
        expected_host = "huggingface.co"
        parts = key.rsplit("/", 1)
        if len(parts) != 2:
            raise IntegrityError("HF revision request key is malformed")
        repo, revision = parts
        expected_path = f"/{repo}/tree/{revision}"
    elif source == "hf-tree":
        expected_host = "huggingface.co"
        parts = key.rsplit("/", 2)
        if len(parts) != 3:
            raise IntegrityError("HF tree request key is malformed")
        repo, revision, _page = parts
        expected_path = f"/api/models/{repo}/tree/{revision}"
    elif source == "hf-file":
        expected_host = "huggingface.co"
        parts = key.split("/")
        if len(parts) < 4:
            raise IntegrityError("HF file request key is malformed")
        repo = "/".join(parts[:2])
        revision = parts[2]
        file_path = "/".join(parts[3:])
        expected_path = f"/api/resolve-cache/models/{repo}/{revision}/{file_path}"
    else:
        raise IntegrityError("observation source has no public endpoint contract")
    xet_host = bool(
        source == "hf-file"
        and isinstance(parsed.hostname, str)
        and parsed.hostname.endswith(".cdn.hf.co")
    )
    if xet_host and path.startswith("/xet-bridge-") and not _is_public_xet_path(path):
        raise IntegrityError("observation Xet path violates its public object grammar")
    xet_transport = bool(xet_host and _is_public_xet_path(path))
    if xet_transport:
        # Hugging Face's public resolver records the final Xet CDN URL for
        # some small files.  The redirect URL cannot restate repo/revision/path;
        # the terminal manifest/tree/hf-file/CAS chain supplies that binding.
        query_kind = "hf-xet"
    elif parsed.hostname != expected_host or path != expected_path:
        raise IntegrityError("observation request URL contradicts its source/key")
    else:
        query_kind = "hf-tree" if source == "hf-tree" else "none"

    query_metadata = _redacted_request_query(
        record,
        parsed,
        query_kind=query_kind,
    )
    sanitized = dict(record)
    sanitized["request_url"] = urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, "", "")
    )
    if query_metadata is None:
        sanitized.pop("request_query", None)
    else:
        sanitized["request_query"] = query_metadata
    return sanitized


def reject_uncontracted_request_query(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep diagnostic/unscoped wrappers from archiving opaque URL queries."""

    request_url = record.get("request_url")
    if not isinstance(request_url, str):
        raise IntegrityError("observation has no request URL provenance")
    try:
        parsed = urlsplit(request_url)
    except ValueError as exc:
        raise IntegrityError("observation request URL is malformed") from exc
    if "?" in request_url or parsed.query or "request_query" in record:
        raise IntegrityError("uncontracted observation request query is prohibited")
    return dict(record)


def observation_timestamp(relative: str, record: Mapping[str, Any]) -> str:
    """Return filename-derived UTC time after binding the wrapper timestamp."""
    name = checked_relative(relative).name
    match = OBSERVATION_FILE_RE.fullmatch(name)
    if match is None:
        raise IntegrityError("observation filename is not append-only watcher form")
    try:
        filename_time = dt.datetime.strptime(
            match.group(1), "%Y%m%dT%H%M%S.%fZ"
        ).replace(tzinfo=dt.timezone.utc)
        raw_observed = record.get("observed_at")
        if not isinstance(raw_observed, str):
            raise ValueError("missing observed_at")
        wrapper_time = dt.datetime.fromisoformat(raw_observed.replace("Z", "+00:00"))
        if wrapper_time.tzinfo is None:
            raise ValueError("naive observed_at")
        wrapper_time = wrapper_time.astimezone(dt.timezone.utc)
    except (TypeError, ValueError) as exc:
        raise IntegrityError("observation timestamp identity is invalid") from exc
    if abs((filename_time - wrapper_time).total_seconds()) > 2.0:
        raise IntegrityError("observation wrapper time does not match its filename")
    return utc_iso(filename_time)


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
        fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        digest = hashlib.sha256()
        size = 0
        tail = b""
        excluded = False
        try:
            with os.fdopen(fd, "w+b", buffering=0) as handle:
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
                if screen_body:
                    if size > MAX_METADATA_BYTES:
                        raise PolicyExcluded("screened body exceeds the metadata safety ceiling")
                    handle.seek(0)
                    if contains_forbidden_body(handle.read()):
                        raise PolicyExcluded("body matched the exclusion vocabulary")
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


def _completed_r1_task_ids(value: Any) -> frozenset[str]:
    """Return the unique canonical task set from exactly one completed Round 1."""
    rounds = value.get("rounds") if isinstance(value, Mapping) else None
    if not isinstance(rounds, list):
        return frozenset()
    round_one = [
        row
        for row in rounds
        if isinstance(row, Mapping)
        and type(row.get("round_number")) is int
        and row.get("round_number") == 1
    ]
    if len(round_one) != 1 or str(round_one[0].get("status", "")).casefold() != "completed":
        return frozenset()
    tasks = round_one[0].get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return frozenset()
    found: list[str] = []
    for task in tasks:
        task_id = task.get("task_id") if isinstance(task, Mapping) else None
        if not isinstance(task_id, str) or TASK_RE.fullmatch(task_id) is None:
            return frozenset()
        found.append(task_id)
    if len(found) != len(set(found)):
        return frozenset()
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
        if entry.kind == "file"
        and entry.path.startswith(prefix)
        and entry.path.endswith(".json")
        and len(checked_relative(entry.path).parts) == 4
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
    validate_observation_identity(latest, record)
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
    if (
        not isinstance(value, dict)
        or value.get("tournament_id") != tournament_id
        or value.get("tournament_type") != "image"
    ):
        raise IntegrityError(
            "tournament body does not assert the selected image tournament identity"
        )
    tasks = _completed_r1_task_ids(value)
    if not tasks:
        raise IntegrityError(
            "selected tournament observation has no unique completed Round-1 task set"
        )
    return TournamentScope(tournament_id, tasks, latest, digest)


def observation_in_scope(relative: str, scope: TournamentScope) -> bool:
    """Select only exact tournament/task/public-submission namespaces.

    The watcher's ``fixtures`` namespace is sourced from ``image_text_pairs``.
    That is the full public pool and can include rows withheld from the miner's
    optimizer-visible ``training_data.zip`` partition.  The Week-7 owner
    boundary permits inventorying the public training archive only, so fixture
    observations are intentionally excluded without opening their bodies.
    """
    parts = checked_relative(relative).parts
    if len(parts) < 4 or parts[0] != "observations":
        return False
    source = parts[1]
    key_head = parts[2]
    if source in {"gradients-tournament", "acceptance"}:
        return len(parts) == 4 and key_head == scope.tournament_id
    if source == "gradients-task":
        return len(parts) == 4 and key_head in scope.task_ids
    if source == "fixtures":
        return False
    if source in ALLOWED_HF_OBSERVATION_SOURCES:
        # HF repo IDs are path components ``author/repo`` and the repo name is
        # tournament-<exact tournament>-<exact task>-<hotkey>. Requiring both
        # IDs prevents a date-wide listing from smuggling text/environment
        # artifacts into an image-tournament root.
        if len(parts) < 5 or parts[2] != HF_ORGANIZATION:
            return False
        repo_name = parts[3]
        marker = f"tournament-{scope.tournament_id}-"
        if not repo_name.startswith(marker):
            return False
        tail = repo_name[len(marker) :]
        if len(tail) <= 37 or tail[36] != "-":
            return False
        task_id, owner = tail[:36], tail[37:]
        if task_id not in scope.task_ids or HF_REPOSITORY_OWNER_RE.fullmatch(owner) is None:
            return False
        if source == "hf-model":
            return len(parts) == 5
        if len(parts) < 6 or REVISION_RE.fullmatch(parts[4]) is None:
            return False
        if source == "hf-revision-manifest":
            return len(parts) == 6
        if source == "hf-tree":
            return len(parts) == 7 and re.fullmatch(r"page-[0-9]+", parts[5]) is not None
        return source == "hf-file" and len(parts) >= 7
    return False


def validate_scoped_observation_body(
    source: str, key: str, value: Any, scope: TournamentScope
) -> None:
    """Cross-bind source-specific public response identity before publication."""
    if source == "gradients-tournament":
        if (
            not isinstance(value, Mapping)
            or value.get("tournament_id") != key
            or value.get("tournament_type") != "image"
        ):
            raise IntegrityError("tournament body contradicts its scoped key")
        return
    if source == "gradients-task":
        if not isinstance(value, Mapping) or value.get("task_id") != key:
            raise IntegrityError("task body contradicts its scoped key")
        return
    if source == "acceptance":
        if not isinstance(value, Mapping) or value.get("tournament_id") != key:
            raise IntegrityError("acceptance body contradicts its scoped key")
        return
    parts = key.split("/")
    repo = "/".join(parts[:2])
    if source == "hf-model":
        if (
            not isinstance(value, Mapping)
            or (value.get("id") or value.get("modelId")) != repo
            or (value.get("id") is not None and value.get("id") != repo)
            or (value.get("modelId") is not None and value.get("modelId") != repo)
        ):
            raise IntegrityError("HF model body contradicts its scoped key")
        return
    revision = parts[2]
    if source == "hf-revision-manifest":
        if (
            not isinstance(value, Mapping)
            or value.get("repo_id") != repo
            or value.get("revision") != revision
        ):
            raise IntegrityError("HF revision body contradicts its scoped key")
        return
    if source == "hf-tree":
        if not isinstance(value, list):
            raise IntegrityError("HF tree body is not a public tree list")
        return
    if source == "hf-file":
        path = "/".join(parts[3:])
        if (
            not isinstance(value, Mapping)
            or value.get("repo_id") != repo
            or value.get("revision") != revision
            or value.get("path") != path
        ):
            raise IntegrityError("HF file body contradicts its scoped key")
        return
    raise IntegrityError("unknown scoped observation source")


def is_full_pool_fixture_observation(relative: str) -> bool:
    """Identify watcher observations sourced from full image_text_pairs pools."""
    parts = checked_relative(relative).parts
    return len(parts) >= 2 and parts[0] == "observations" and parts[1] == "fixtures"


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
        canonical = json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True)
        tournaments = set(TOURNAMENT_RE.findall(canonical))
        tasks = set(TASK_RE.findall(canonical))
        if any(item != scope.tournament_id for item in tournaments) or any(
            item not in scope.task_ids for item in tasks
        ):
            out_of_scope += 1
            continue
        if scope.tournament_id in tournaments or bool(tasks & scope.task_ids):
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
            "request_queries": (
                "source-allowlisted; values discarded; value-free metadata only"
            ),
            "watcher_image_text_pair_fixtures": "excluded-unopened",
            "public_training_archives": "separate exact-task inventory only",
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
        # This exclusion is unconditional.  Unscoped/diagnostic invocations
        # must have the same privacy boundary as exact-tournament P0 runs.
        # Apply it before reading the wrapper, so neither fixture metadata nor
        # its referenced full-pool CAS bytes cross the boundary.
        if is_full_pool_fixture_observation(entry.path):
            result["excluded"]["out_of_scope"] += 1
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
            wrapper_source, wrapper_key = validate_observation_identity(entry.path, record)
            if wrapper_source in PUBLIC_PROVENANCE_SOURCES:
                record = validate_public_request_provenance(
                    wrapper_source, wrapper_key, record
                )
            else:
                record = reject_uncontracted_request_query(record)
            # Never publish the source wrapper verbatim after URL validation:
            # legitimate pagination/Xet query values are transport credentials
            # and survive only as fixed, value-free metadata.
            snapshot_body = canonical_json(record)
            if wrapper_source == "fixtures":
                raise PolicyExcluded("full-pool fixture observation is prohibited")
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
            content_bytes = record.get("content_bytes")
            if (
                type(content_bytes) is not int
                or content_bytes < 0
                or content_bytes != content_staged.size
            ):
                raise IntegrityError("observation content_bytes does not match its CAS object")
            content_value: Any = None
            if content_staged.size <= MAX_METADATA_BYTES:
                content_body = content_staged.path.read_bytes()
                try:
                    content_value = json.loads(content_body)
                except json.JSONDecodeError:
                    content_value = None
            if scope is not None:
                validate_scoped_observation_body(
                    wrapper_source, wrapper_key, content_value, scope
                )
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
