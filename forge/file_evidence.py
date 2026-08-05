"""Descriptor-bound reads for evidence files.

Evidence must be read from the object that was opened, not from a path checked
before a second path lookup.  Linux uses ``openat2(RESOLVE_NO_SYMLINKS)`` to
reject symlinks in every path component.  The portable fallback can protect
only the final component with ``O_NOFOLLOW``; descriptor identity, file type,
size drift, and mutation checks still apply to the object actually opened.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import errno
import os
import stat
import sys
from typing import Iterator


class RegularFileError(RuntimeError):
    """A purported evidence path is not one stable regular file."""


_AT_FDCWD = -100
_OPENAT2_SYSCALL = 437
_RESOLVE_NO_SYMLINKS = 0x04
_OPENAT2_MACHINES = frozenset(
    {"aarch64", "arm64", "riscv64", "x86_64", "amd64"}
)


class _OpenHow(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint64),
        ("mode", ctypes.c_uint64),
        ("resolve", ctypes.c_uint64),
    ]


def _openat2_no_symlinks(path: str, flags: int) -> int:
    """Open ``path`` while rejecting symlinks in every component on Linux."""

    encoded_path = os.fsencode(path)
    if b"\x00" in encoded_path:
        raise ValueError("evidence path contains an embedded NUL")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    how = _OpenHow(
        flags=flags,
        mode=0,
        resolve=_RESOLVE_NO_SYMLINKS,
    )
    result = libc.syscall(
        ctypes.c_long(_OPENAT2_SYSCALL),
        ctypes.c_int(_AT_FDCWD),
        ctypes.c_char_p(encoded_path),
        ctypes.byref(how),
        ctypes.c_size_t(ctypes.sizeof(how)),
    )
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), path)
    return int(result)


def _open_evidence_path(path: str, flags: int) -> int:
    """Prefer a full-path guard; fall back only when the API is unavailable."""

    path = os.fspath(path)
    machine = os.uname().machine.lower() if hasattr(os, "uname") else ""
    if sys.platform.startswith("linux") and machine in _OPENAT2_MACHINES:
        try:
            return _openat2_no_symlinks(path, flags)
        except OSError as exc:
            if exc.errno != errno.ENOSYS:
                raise
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise OSError(
            errno.ENOTSUP,
            "final-component symlink protection is unavailable",
            path,
        )
    return os.open(path, flags | nofollow)


def stat_identity(value: os.stat_result) -> dict[str, int]:
    """Return the filesystem fields used by the run-scope identity contract."""

    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


@contextmanager
def open_regular_file(
    path: str,
    *,
    label: str,
    minimum_size: int = 1,
    maximum_size: int | None = None,
) -> Iterator[tuple[int, os.stat_result]]:
    """Open one stable file with full-path Linux or final-component protection."""

    fd: int | None = None
    try:
        # O_NONBLOCK makes special files fail at the subsequent fstat/type gate
        # instead of allowing a FIFO/device open to stall release validation.
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        fd = _open_evidence_path(path, flags)
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise RegularFileError(f"{label} must be a regular file")
        if opened.st_size < minimum_size:
            raise RegularFileError(f"{label} is empty or truncated")
        if maximum_size is not None and opened.st_size > maximum_size:
            raise RegularFileError(f"{label} is too large")
        yield fd, opened
        closed_over = os.fstat(fd)
        if stat_identity(closed_over) != stat_identity(opened):
            raise RegularFileError(f"{label} changed while it was read")
    except RegularFileError:
        raise
    except Exception as exc:
        raise RegularFileError(f"{label} unavailable: {path}") from exc
    finally:
        if fd is not None:
            os.close(fd)


def read_regular_bytes(
    path: str,
    *,
    label: str,
    maximum_size: int,
) -> bytes:
    """Read the exact bytes of one bounded descriptor-verified regular file."""

    with open_regular_file(
        path,
        label=label,
        minimum_size=1,
        maximum_size=maximum_size,
    ) as (fd, opened):
        remaining = int(opened.st_size)
        chunks: list[bytes] = []
        while remaining:
            block = os.read(fd, min(1024 * 1024, remaining))
            if not block:
                raise RegularFileError(f"{label} was truncated while it was read")
            chunks.append(block)
            remaining -= len(block)
        if os.read(fd, 1):
            raise RegularFileError(f"{label} grew while it was read")
        return b"".join(chunks)
