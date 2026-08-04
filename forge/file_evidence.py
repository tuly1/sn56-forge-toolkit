"""Descriptor-bound reads for evidence files.

Evidence must be read from the object that was opened, not from a path checked
before a second path lookup.  These helpers reject symlinks, non-regular files,
size drift, and mutation while a descriptor is being consumed.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import stat
from typing import Iterator


class RegularFileError(RuntimeError):
    """A purported evidence path is not one stable regular file."""


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
    """Open one stable regular file without following its final path component."""

    fd: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
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
