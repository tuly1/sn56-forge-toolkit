from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SYNC_MODULE = ROOT / "ops" / "experiments" / "week7" / "safe_harvest_sync.py"
sync_spec = importlib.util.spec_from_file_location("safe_harvest_sync", SYNC_MODULE)
sync = importlib.util.module_from_spec(sync_spec)
assert sync_spec.loader is not None
sys.modules[sync_spec.name] = sync
sync_spec.loader.exec_module(sync)

HEADER_MODULE = ROOT / "ops" / "experiments" / "week7" / "public_safetensors_headers.py"
header_spec = importlib.util.spec_from_file_location("public_safetensors_headers", HEADER_MODULE)
headers = importlib.util.module_from_spec(header_spec)
assert header_spec.loader is not None
sys.modules[header_spec.name] = headers
header_spec.loader.exec_module(headers)

TOURNAMENT = "tourn_aaaaaaaaaaaaaaaa_20260810"
OTHER_TOURNAMENT = "tourn_bbbbbbbbbbbbbbbb_20260810"
TASK = "11111111-1111-4111-8111-111111111111"
REVISION = "1" * 40
LFS_SHA = "2" * 64
NOW = dt.datetime(2026, 8, 10, 15, 0, tzinfo=dt.timezone.utc)


def sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def write(root: Path, relative: str, body: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def add_cas(root: Path, body: bytes) -> str:
    digest = sha(body)
    write(root, f"objects/sha256/{digest[:2]}/{digest}", body)
    return digest


def source_root(tmp_path: Path, *, include_other: bool = False, path: str = "checkpoints/last.safetensors") -> Path:
    root = tmp_path / "source"
    root.mkdir()
    write(
        root,
        "ROOT-IDENTITY.json",
        sync.canonical_json(
            {
                "schema": "sn56.week7.harvest-root-identity",
                "schema_version": 1,
                "source": "fixture",
                "tournament_id": TOURNAMENT,
            }
        ),
    )

    def add_tree(tournament: str, revision: str, task: str) -> None:
        repo = f"tournament-{tournament}-{task}-5Example"
        tree = sync.canonical_json(
            [
                {
                    "type": "file",
                    "path": path,
                    "size": 4096,
                    "lfs": {"oid": LFS_SHA, "size": 4096},
                }
            ]
        )
        tree_sha = add_cas(root, tree)
        wrapper = sync.canonical_json(
            {
                "schema": 1,
                "source": "hf-tree",
                "key": f"gradients-io-tournaments/{repo}/{revision}/page-0001",
                "request_url": f"https://huggingface.co/api/models/x/{repo}/tree/{revision}",
                "status": 200,
                "content_sha256": tree_sha,
                "content_bytes": len(tree),
                "object": f"objects/sha256/{tree_sha[:2]}/{tree_sha}",
            }
        )
        write(
            root,
            f"observations/hf-tree/gradients-io-tournaments/{repo}/{revision}/page-0001/obs.json",
            wrapper,
        )

    add_tree(TOURNAMENT, REVISION, TASK)
    if include_other:
        add_tree(OTHER_TOURNAMENT, "3" * 40, "22222222-2222-4222-8222-222222222222")
    return root


class FakeResponse:
    def __init__(self, status: int, response_headers: dict[str, str], body: bytes):
        self.status = status
        self.headers = response_headers
        self.body = body
        self.read_calls = 0
        self.closed = False

    def read(self, amount: int = -1) -> bytes:
        self.read_calls += 1
        return self.body if amount < 0 else self.body[:amount]

    def close(self) -> None:
        self.closed = True


def test_http_200_is_rejected_without_reading_the_weight_body():
    response = FakeResponse(200, {"Content-Length": "4096"}, b"x" * 4096)

    with pytest.raises(sync.IntegrityError, match="not 206"):
        headers.fetch_exact_range(
            "https://example.invalid/model.safetensors",
            0,
            7,
            4096,
            timeout=1,
            opener=lambda request, timeout: response,
        )

    assert response.read_calls == 0
    assert response.closed is True


def test_malformed_content_range_is_rejected():
    response = FakeResponse(
        206,
        {"Content-Range": "bytes 0-8/4096", "Content-Length": "8"},
        b"12345678",
    )
    with pytest.raises(sync.IntegrityError, match="Content-Range"):
        headers.fetch_exact_range(
            "https://example.invalid/model.safetensors",
            0,
            7,
            4096,
            timeout=1,
            opener=lambda request, timeout: response,
        )
    assert response.read_calls == 0


def fake_header_opener(parsed: dict, total: int = 4096):
    header = json.dumps(parsed, separators=(",", ":")).encode()
    full_header = struct.pack("<Q", len(header)) + header

    def open_request(request, timeout):
        wanted = request.headers["Range"]
        start, end = (int(value) for value in wanted.removeprefix("bytes=").split("-"))
        body = full_header[start : end + 1]
        return FakeResponse(
            206,
            {
                "Content-Range": f"bytes {start}-{end}/{total}",
                "Content-Length": str(len(body)),
            },
            body,
        )

    return open_request, full_header


def test_valid_header_is_parsed_with_no_tensor_body_request():
    parsed = {
        "__metadata__": {"training_info": '{"step": 1200}'},
        "layer.weight": {"dtype": "F16", "shape": [2, 3], "data_offsets": [0, 12]},
    }
    opener, full_header = fake_header_opener(parsed)

    raw, value = headers.fetch_safetensors_header(
        "https://example.invalid/model.safetensors", 4096, opener=opener
    )

    assert raw == full_header
    assert value == parsed


def test_harvest_is_exact_tournament_scoped_and_create_only(tmp_path):
    source = source_root(tmp_path, include_other=True)
    output = tmp_path / "headers"
    parsed = {
        "__metadata__": {"training_info": '{"step": 1200}'},
        "layer.weight": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
    }
    opener, _ = fake_header_opener(parsed)

    first = headers.harvest(
        source, output, TOURNAMENT, observed_at=NOW, opener=opener
    )
    second = headers.harvest(
        source, output, TOURNAMENT, observed_at=NOW, opener=opener
    )

    assert first == second
    assert first["candidate_count"] == 1
    inventory = json.loads((output / first["inventory"]).read_text())
    assert len(inventory["associations"]) == 1
    association = inventory["associations"][0]
    assert association["revision"] == REVISION
    assert association["task_id"] == TASK
    assert OTHER_TOURNAMENT not in json.dumps(inventory)
    record = json.loads((output / association["record"]).read_text())
    assert record["metadata"]["training_info"] == '{"step": 1200}'
    assert record["tensor_count"] == 1


def test_preexisting_different_record_aborts_without_overwrite(tmp_path):
    source = source_root(tmp_path)
    output = tmp_path / "headers"
    output.mkdir()
    write(
        output,
        "ROOT-IDENTITY.json",
        headers.canonical_json(
            {
                "schema": "sn56.week7.public-safetensors-header-root",
                "schema_version": 1,
                "input_root": str(source.absolute()),
                "tournament_id": TOURNAMENT,
            }
        ),
    )
    record_path = output / f"records/{LFS_SHA[:2]}/{LFS_SHA}.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text("corrupt\n")
    before = record_path.read_bytes()
    opener, _ = fake_header_opener(
        {"layer.weight": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}}
    )

    with pytest.raises(sync.IntegrityError, match="different bytes"):
        headers.harvest(source, output, TOURNAMENT, observed_at=NOW, opener=opener)

    assert record_path.read_bytes() == before


def test_prohibited_checkpoint_path_is_never_requested(tmp_path):
    source = source_root(tmp_path, path="hidden/test_rows.safetensors")
    output = tmp_path / "headers"
    calls = 0

    def must_not_open(request, timeout):
        nonlocal calls
        calls += 1
        raise AssertionError("network must not be touched")

    result = headers.harvest(
        source, output, TOURNAMENT, observed_at=NOW, opener=must_not_open
    )
    assert result["candidate_count"] == 0
    assert calls == 0
