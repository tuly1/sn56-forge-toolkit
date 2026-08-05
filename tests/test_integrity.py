from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import struct

import pytest

from forge.tasks import integrity


def _write_training_artifact(path: Path, *, step: int, value: float) -> bytes:
    header = json.dumps(
        {
            "__metadata__": {
                "training_info": json.dumps({"step": step, "epoch": 1})
            },
            "weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            },
        }
    ).encode("utf-8")
    payload = struct.pack("<Q", len(header)) + header + struct.pack("<f", value)
    path.write_bytes(payload)
    return payload


def test_descriptor_inspection_uses_opened_bytes_after_path_swap(tmp_path):
    artifact_path = tmp_path / "last.safetensors"
    original = _write_training_artifact(artifact_path, step=120, value=1.0)
    replacement = tmp_path / "replacement.safetensors"
    replacement_bytes = _write_training_artifact(
        replacement, step=999, value=2.0
    )

    fd = os.open(artifact_path, os.O_RDONLY)
    try:
        os.replace(replacement, artifact_path)
        evidence = integrity.inspect_training_artifact_fd(
            fd,
            path_label=str(artifact_path),
        )
    finally:
        os.close(fd)

    assert artifact_path.read_bytes() == replacement_bytes
    assert evidence.path == str(artifact_path)
    assert evidence.size_bytes == len(original)
    assert evidence.sha256 == hashlib.sha256(original).hexdigest()
    assert evidence.checkpoint_step == 120


def test_descriptor_inspection_rejects_nonregular_and_empty_files(tmp_path):
    read_fd, write_fd = os.pipe()
    try:
        with pytest.raises(ValueError, match="regular file"):
            integrity.inspect_training_artifact_fd(
                read_fd,
                path_label="pipe",
            )
    finally:
        os.close(read_fd)
        os.close(write_fd)

    empty = tmp_path / "empty.safetensors"
    empty.touch()
    fd = os.open(empty, os.O_RDONLY)
    try:
        with pytest.raises(ValueError, match="empty or truncated"):
            integrity.inspect_training_artifact_fd(
                fd,
                path_label=str(empty),
            )
    finally:
        os.close(fd)


def test_descriptor_inspection_rejects_identity_change(tmp_path, monkeypatch):
    artifact = tmp_path / "last.safetensors"
    _write_training_artifact(artifact, step=120, value=1.0)
    fd = os.open(artifact, os.O_RDONLY)
    real_fstat = os.fstat
    opened = real_fstat(fd)
    calls = 0

    def changing_fstat(candidate_fd: int):
        nonlocal calls
        calls += 1
        observed = real_fstat(candidate_fd)
        if calls < 2:
            return observed
        return SimpleNamespace(
            st_mode=observed.st_mode,
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_size=observed.st_size,
            st_mtime_ns=observed.st_mtime_ns + 1,
            st_ctime_ns=observed.st_ctime_ns,
        )

    monkeypatch.setattr(integrity.os, "fstat", changing_fstat)
    try:
        with pytest.raises(ValueError, match="changed while it was inspected"):
            integrity.inspect_training_artifact_fd(
                fd,
                path_label=str(artifact),
            )
    finally:
        monkeypatch.setattr(integrity.os, "fstat", real_fstat)
        os.close(fd)

    assert opened.st_size > 10


def test_path_api_preserves_absolute_path_and_descriptor_evidence(tmp_path):
    artifact = tmp_path / "last.safetensors"
    payload = _write_training_artifact(artifact, step=321, value=1.0)

    evidence = integrity.inspect_training_artifact(str(artifact))

    assert evidence.path == str(artifact.resolve())
    assert evidence.size_bytes == len(payload)
    assert evidence.sha256 == hashlib.sha256(payload).hexdigest()
    assert evidence.checkpoint_step == 321
    assert evidence.file_identity == integrity.stat_identity(artifact.stat())
