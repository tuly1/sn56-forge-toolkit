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
import json
import math
import os
import struct

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


def valid_safetensors(path: str) -> bool:
    try:
        size = os.path.getsize(path)
        if size < 10:
            return False
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            if header_len <= 0 or header_len > min(_MAX_HEADER_BYTES, size - 8):
                return False
            header = json.loads(f.read(header_len))
        if not isinstance(header, dict):
            return False
        payload_size = size - 8 - header_len
        tensors = 0
        ranges: list[tuple[int, int]] = []
        for name, info in header.items():
            if name == "__metadata__":
                if not isinstance(info, dict) or any(
                    not isinstance(key, str) or not isinstance(value, str)
                    for key, value in info.items()
                ):
                    return False
                continue
            if not isinstance(info, dict):
                return False
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
                return False
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
                return False
            if offs[1] - offs[0] != math.prod(shape) * _DTYPE_BYTES[dtype]:
                return False
            ranges.append((offs[0], offs[1]))
            tensors += 1
        if tensors == 0:
            return False
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                return False
            cursor = end
        return cursor == payload_size
    except Exception:
        return False
