"""Build-time staging and byte verification for offline FLUX tokenizers."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from typing import Any


TOKENIZER_ROOT = "/app/flux/tokenizers"
TOKENIZERS: tuple[dict[str, Any], ...] = (
    {
        "repo": "openai/clip-vit-large-patch14",
        "revision": "32bd64288804d66eefd0ccbe215aa642df71cc41",
        "directory": "openai_clip-vit-large-patch14",
        "class": "CLIPTokenizer",
        "files": {
            "merges.txt": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
            "special_tokens_map.json": "2cdb3b8331a60c92fc1e55a13e9fd61fd2293c5a51275fdcccd62b780052530e",
            "tokenizer_config.json": "6bdcee9ccce2a16ca2b4c0c5ed00b42c50ea225f4472a8c4c1e963a2902c2881",
            "vocab.json": "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349",
        },
    },
    {
        "repo": "google/t5-v1_1-xxl",
        "revision": "3db67ab1af984cf10548a73467f0e5bca2aaaeb2",
        "directory": "google_t5-v1_1-xxl",
        "class": "T5TokenizerFast",
        "files": {
            "special_tokens_map.json": "7a1985a994c41886db38c719d2a3d2f40606663cc19d7c5d6a85d349320e06d2",
            "spiece.model": "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86",
            "tokenizer.json": "f5dfec163765e18e270537fe896c49f5fad74db1525641d9b255a3008b999596",
            "tokenizer_config.json": "1a3d2db64215ed77854dd4208aac5f8361c1b5471cabd19c0ef1472d1a895eb0",
        },
    },
)


def stage(root: str = TOKENIZER_ROOT) -> dict[str, object]:
    """Download pinned public tokenizer sources and publish them atomically."""
    from transformers import CLIPTokenizer, T5TokenizerFast

    classes = {
        "CLIPTokenizer": CLIPTokenizer,
        "T5TokenizerFast": T5TokenizerFast,
    }
    if os.path.lexists(root):
        raise FileExistsError(root)

    staging_root = f"{root}.staging"
    _remove_staging_root(staging_root)
    os.makedirs(staging_root)
    try:
        for spec in TOKENIZERS:
            destination = os.path.join(staging_root, spec["directory"])
            tokenizer = classes[spec["class"]].from_pretrained(
                spec["repo"],
                revision=spec["revision"],
                local_files_only=False,
            )
            tokenizer.save_pretrained(destination)
        result = verify(staging_root)
        os.rename(staging_root, root)
    except BaseException:
        _remove_staging_root(staging_root)
        raise

    result["root"] = root
    return result


def _remove_staging_root(path: str) -> None:
    """Remove only a prior regular staging directory; reject unsafe entries."""
    if not os.path.lexists(path):
        return
    if os.path.islink(path) or not os.path.isdir(path):
        raise RuntimeError(f"unsafe tokenizer staging path: {path}")
    shutil.rmtree(path)


def verify(root: str = TOKENIZER_ROOT) -> dict[str, object]:
    """Prove exact files, bytes, and local-only loadability."""
    from transformers import CLIPTokenizer, T5TokenizerFast

    classes = {
        "CLIPTokenizer": CLIPTokenizer,
        "T5TokenizerFast": T5TokenizerFast,
    }
    verified_files = 0
    for spec in TOKENIZERS:
        directory = os.path.join(root, spec["directory"])
        if not os.path.isdir(directory) or os.path.islink(directory):
            raise RuntimeError(f"unsafe or missing tokenizer directory: {directory}")
        observed = {
            name
            for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name))
            and not os.path.islink(os.path.join(directory, name))
        }
        expected = set(spec["files"])
        if observed != expected:
            raise RuntimeError(
                f"tokenizer file set mismatch for {spec['repo']}: "
                f"expected {sorted(expected)}, got {sorted(observed)}"
            )
        for name, wanted in spec["files"].items():
            path = os.path.join(directory, name)
            actual = _sha256(path)
            if actual != wanted:
                raise RuntimeError(
                    f"tokenizer hash mismatch for {spec['repo']}/{name}: {actual}"
                )
            verified_files += 1
        tokenizer = classes[spec["class"]].from_pretrained(
            directory,
            local_files_only=True,
        )
        encoded = tokenizer("offline tokenizer contract", return_tensors="pt")
        if not encoded.get("input_ids").numel():
            raise RuntimeError(f"tokenizer produced no input ids: {spec['repo']}")
    return {
        "result": "PASS",
        "tokenizers_verified": len(TOKENIZERS),
        "files_verified": verified_files,
        "root": root,
    }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stage", "verify"))
    args = parser.parse_args(argv)
    print(stage() if args.action == "stage" else verify())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
