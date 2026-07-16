"""Cheap safetensors integrity check for checkpoint promotion.

ai-toolkit writes checkpoints non-atomically, so a deadline SIGKILL landing
mid-save leaves a truncated file. Whatever we promote to last.safetensors is
the ONLY file the evaluator loads — promoting a truncated one zero-scores a
task that has an intact older checkpoint sitting right next to it.

Validation: 8-byte little-endian header length, JSON header parse, then the
actual file size must cover every tensor's data_offsets end. No tensor data is
read, so this is O(header) — microseconds even for multi-GB files.
"""
import json
import os
import struct


def valid_safetensors(path: str) -> bool:
    try:
        size = os.path.getsize(path)
        if size < 8:
            return False
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            if header_len <= 0 or 8 + header_len > size:
                return False
            header = json.loads(f.read(header_len))
        data_end = 0
        for name, info in header.items():
            if name == "__metadata__":
                continue
            offs = info.get("data_offsets")
            if not offs or len(offs) != 2:
                return False
            data_end = max(data_end, offs[1])
        return size >= 8 + header_len + data_end
    except Exception:
        return False
