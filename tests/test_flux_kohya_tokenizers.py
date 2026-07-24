from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from forge import flux_kohya_tokenizers


class _InputIds:
    def numel(self) -> int:
        return 1


class _Tokenizer:
    def __init__(self, payload: bytes):
        self.payload = payload

    def save_pretrained(self, destination: str) -> None:
        os.makedirs(destination)
        Path(destination, "tokenizer.bin").write_bytes(self.payload)

    def __call__(self, _text: str, *, return_tensors: str):
        assert return_tensors == "pt"
        return {"input_ids": _InputIds()}


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_tokenizer_stage_recovers_cleanly_after_partial_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    clip_payload = b"clip-tokenizer"
    t5_payload = b"t5-tokenizer"
    specs = (
        {
            "repo": "test/clip",
            "revision": "clip-revision",
            "directory": "clip",
            "class": "CLIPTokenizer",
            "files": {"tokenizer.bin": _digest(clip_payload)},
        },
        {
            "repo": "test/t5",
            "revision": "t5-revision",
            "directory": "t5",
            "class": "T5TokenizerFast",
            "files": {"tokenizer.bin": _digest(t5_payload)},
        },
    )
    t5_downloads = 0

    class FakeClipTokenizer:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs):
            if kwargs.get("local_files_only"):
                return _Tokenizer(clip_payload)
            assert source == "test/clip"
            return _Tokenizer(clip_payload)

    class FakeT5TokenizerFast:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs):
            nonlocal t5_downloads
            if kwargs.get("local_files_only"):
                return _Tokenizer(t5_payload)
            assert source == "test/t5"
            t5_downloads += 1
            if t5_downloads == 1:
                raise ConnectionError("simulated TLS truncation")
            return _Tokenizer(t5_payload)

    monkeypatch.setattr(flux_kohya_tokenizers, "TOKENIZERS", specs)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            CLIPTokenizer=FakeClipTokenizer,
            T5TokenizerFast=FakeT5TokenizerFast,
        ),
    )

    root = tmp_path / "tokenizers"
    with pytest.raises(ConnectionError, match="TLS truncation"):
        flux_kohya_tokenizers.stage(str(root))

    assert not root.exists()
    assert not Path(f"{root}.staging").exists()

    result = flux_kohya_tokenizers.stage(str(root))

    assert result == {
        "result": "PASS",
        "tokenizers_verified": 2,
        "files_verified": 2,
        "root": str(root),
    }
    assert (root / "clip" / "tokenizer.bin").read_bytes() == clip_payload
    assert (root / "t5" / "tokenizer.bin").read_bytes() == t5_payload


def test_tokenizer_stage_rejects_an_unsafe_staging_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "tokenizers"
    staging = Path(f"{root}.staging")
    staging.symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(CLIPTokenizer=object, T5TokenizerFast=object),
    )

    with pytest.raises(RuntimeError, match="unsafe tokenizer staging path"):
        flux_kohya_tokenizers.stage(str(root))

    assert staging.is_symlink()
