"""Cheap safetensors integrity check for checkpoint promotion.

ai-toolkit writes checkpoints non-atomically, so a deadline SIGKILL landing
mid-save leaves a truncated file. Whatever we promote to last.safetensors is
the ONLY file the evaluator loads — promoting a truncated one zero-scores a
task that has an intact older checkpoint sitting right next to it.

Validation: bounded 8-byte little-endian header length, JSON object, known
dtypes, non-negative shapes, exact shape/span byte counts, contiguous
non-overlapping tensor ranges, and complete payload coverage. No tensor data is
read, so this remains O(header) even for multi-GB files.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import os
import struct

from forge.file_evidence import open_regular_file, stat_identity

_MAX_HEADER_BYTES = 100_000_000
_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "C64": 8,
}


@dataclass(frozen=True)
class TrainingArtifactEvidence:
    """Descriptor-bound identity for one loadable ai-toolkit checkpoint."""

    path: str
    size_bytes: int
    sha256: str
    checkpoint_step: int
    file_identity: dict[str, int]


def _read_exact(fd: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        block = os.read(fd, remaining)
        if not block:
            raise ValueError("safetensors file is truncated")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def _validated_header(fd: int, size: int) -> dict:
    if size < 10:
        raise ValueError("safetensors file is too small")
    os.lseek(fd, 0, os.SEEK_SET)
    (header_len,) = struct.unpack("<Q", _read_exact(fd, 8))
    if header_len <= 0 or header_len > min(_MAX_HEADER_BYTES, size - 8):
        raise ValueError("safetensors header length is invalid")
    header = json.loads(_read_exact(fd, header_len))
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not an object")
    payload_size = size - 8 - header_len
    tensors = 0
    ranges: list[tuple[int, int]] = []
    for name, info in header.items():
        if name == "__metadata__":
            if not isinstance(info, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in info.items()
            ):
                raise ValueError("safetensors metadata is invalid")
            continue
        if not isinstance(info, dict):
            raise ValueError("safetensors tensor entry is invalid")
        offs = info.get("data_offsets")
        if (
            not isinstance(offs, list)
            or len(offs) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in offs
            )
            or not 0 <= offs[0] <= offs[1] <= payload_size
        ):
            raise ValueError("safetensors offsets are invalid")
        dtype = info.get("dtype")
        shape = info.get("shape")
        if (
            dtype not in _DTYPE_BYTES
            or not isinstance(shape, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in shape
            )
        ):
            raise ValueError("safetensors tensor schema is invalid")
        if offs[1] - offs[0] != math.prod(shape) * _DTYPE_BYTES[dtype]:
            raise ValueError("safetensors tensor span is invalid")
        ranges.append((offs[0], offs[1]))
        tensors += 1
    if tensors == 0:
        raise ValueError("safetensors file has no tensors")
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError("safetensors payload is not contiguous")
        cursor = end
    if cursor != payload_size:
        raise ValueError("safetensors payload coverage is incomplete")
    return header


def _training_step(header: dict) -> int:
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise ValueError("training artifact has no metadata")
    raw = metadata.get("training_info")
    if not isinstance(raw, str):
        raise ValueError("training artifact has no training_info metadata")
    info = json.loads(raw)
    step = info.get("step") if isinstance(info, dict) else None
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("training artifact checkpoint step is invalid")
    return step


def inspect_training_artifact(path: str) -> TrainingArtifactEvidence:
    """Validate, hash, and read the training step from one opened descriptor."""

    absolute = os.path.abspath(path)
    with open_regular_file(
        absolute,
        label="training artifact",
        minimum_size=10,
    ) as (fd, opened):
        header = _validated_header(fd, int(opened.st_size))
        checkpoint_step = _training_step(header)
        os.lseek(fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        remaining = int(opened.st_size)
        while remaining:
            block = os.read(fd, min(1024 * 1024, remaining))
            if not block:
                raise ValueError("training artifact was truncated while hashing")
            digest.update(block)
            remaining -= len(block)
        if os.read(fd, 1):
            raise ValueError("training artifact grew while hashing")
        return TrainingArtifactEvidence(
            path=absolute,
            size_bytes=int(opened.st_size),
            sha256=digest.hexdigest(),
            checkpoint_step=checkpoint_step,
            file_identity=stat_identity(opened),
        )


def valid_safetensors(path: str) -> bool:
    try:
        with open_regular_file(
            path,
            label="safetensors artifact",
            minimum_size=10,
        ) as (fd, opened):
            _validated_header(fd, int(opened.st_size))
        return True
    except Exception:
        return False
