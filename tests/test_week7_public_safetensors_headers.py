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


def tree_entry(path: str, *, oid: str = LFS_SHA, size: int = 4096) -> dict:
    return {
        "type": "file",
        "path": path,
        "size": size,
        "lfs": {"oid": oid, "size": size},
    }


def add_tree(
    root: Path,
    *,
    tournament: str = TOURNAMENT,
    revision: str = REVISION,
    task: str = TASK,
    page: str = "page-0001",
    entries: list[dict] | None = None,
    observed_at: str = "2026-08-10T15:00:00Z",
) -> tuple[str, str]:
    repo = f"tournament-{tournament}-{task}-5ExampLe"
    tree = sync.canonical_json(
        entries
        if entries is not None
        else [tree_entry("checkpoints/last.safetensors")]
    )
    tree_sha = add_cas(root, tree)
    key = f"gradients-io-tournaments/{repo}/{revision}/{page}"
    wrapper = sync.canonical_json(
        {
            "schema": 1,
            "source": "hf-tree",
            "key": key,
            "request_url": (
                "https://huggingface.co/api/models/"
                f"gradients-io-tournaments/{repo}/tree/{revision}"
            ),
            "status": 200,
            "observed_at": observed_at,
            "content_sha256": tree_sha,
            "content_bytes": len(tree),
            "object": f"objects/sha256/{tree_sha[:2]}/{tree_sha}",
        }
    )
    parsed = dt.datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    stamp = parsed.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    relative = f"observations/hf-tree/{key}/{stamp}-{tree_sha[:12]}.json"
    write(root, relative, wrapper)
    return repo, relative


def source_root(
    tmp_path: Path,
    *,
    include_other: bool = False,
    path: str = "checkpoints/last.safetensors",
    entries: list[dict] | None = None,
) -> Path:
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

    add_tree(
        root,
        entries=entries if entries is not None else [tree_entry(path)],
    )
    if include_other:
        add_tree(
            root,
            tournament=OTHER_TOURNAMENT,
            revision="3" * 40,
            task="22222222-2222-4222-8222-222222222222",
        )
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


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/file",
        "https://127.0.0.1/file",
        "https://user:pass@huggingface.co/file",
        "https://huggingface.co:444/file",
        "https://attacker.example/file",
    ],
)
def test_hf_transport_rejects_non_hf_redirect_authorities(url):
    with pytest.raises(sync.IntegrityError, match="Hugging Face HTTPS boundary"):
        headers._validate_hf_transport_url(url)


def test_hf_transport_accepts_official_resolver_and_cdn_hosts():
    headers._validate_hf_transport_url("https://huggingface.co/org/repo/resolve/rev/file")
    headers._validate_hf_transport_url("https://us.aws.cdn.hf.co/xet-bridge-us/object")


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
        source, output, TOURNAMENT, task_ids={TASK}, observed_at=NOW, opener=opener
    )
    second = headers.harvest(
        source, output, TOURNAMENT, task_ids={TASK}, observed_at=NOW, opener=opener
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
                "task_ids": [TASK],
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
        headers.harvest(
            source, output, TOURNAMENT, task_ids={TASK}, observed_at=NOW, opener=opener
        )

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
        source, output, TOURNAMENT, task_ids={TASK}, observed_at=NOW, opener=must_not_open
    )
    assert result["candidate_count"] == 0
    assert calls == 0


def test_same_tournament_task_outside_exact_allowlist_is_never_requested(tmp_path):
    source = source_root(tmp_path)
    other_task = "22222222-2222-4222-8222-222222222222"
    add_tree(source, task=other_task, revision="3" * 40)
    output = tmp_path / "headers"
    parsed = {
        "layer.weight": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}
    }
    base_opener, _ = fake_header_opener(parsed)
    calls: list[str] = []

    def opener(request, timeout):
        calls.append(request.full_url)
        return base_opener(request, timeout)

    result = headers.harvest(
        source,
        output,
        TOURNAMENT,
        task_ids={TASK},
        observed_at=NOW,
        opener=opener,
    )

    assert result["candidate_count"] == 1
    assert len(calls) == 2  # prefix and JSON header for the one allowed object
    assert other_task not in json.dumps(result)


@pytest.mark.parametrize(
    "path",
    [
        "../escape.safetensors",
        "checkpoints//escape.safetensors",
        "checkpoints/%2e%2e/escape.safetensors",
        "checkpoints/%252e%252e/escape.safetensors",
        "checkpoints/%5c..%5cescape.safetensors",
    ],
)
def test_noncanonical_or_encoded_traversal_path_fails_before_network(tmp_path, path):
    source = source_root(tmp_path, path=path)
    calls = 0

    def must_not_open(request, timeout):
        nonlocal calls
        calls += 1
        raise AssertionError("network must not be touched")

    with pytest.raises(sync.IntegrityError, match="safetensors tree path"):
        headers.harvest(
            source,
            tmp_path / "headers",
            TOURNAMENT,
            task_ids={TASK},
            observed_at=NOW,
            opener=must_not_open,
        )
    assert calls == 0


def test_tree_key_requires_exact_public_organization_before_network(tmp_path):
    source = source_root(tmp_path)
    wrapper_path = next(
        (source / "observations" / "hf-tree" / "gradients-io-tournaments").rglob(
            "*.json"
        )
    )
    wrapper = json.loads(wrapper_path.read_bytes())
    wrapper["key"] = wrapper["key"].replace(
        "gradients-io-tournaments/", "attacker-gradients-io-tournaments/", 1
    )
    wrapper_path.write_bytes(sync.canonical_json(wrapper))
    calls = 0

    def must_not_open(request, timeout):
        nonlocal calls
        calls += 1
        raise AssertionError("network must not be touched")

    with pytest.raises(sync.IntegrityError, match="public tournament organization"):
        headers.harvest(
            source,
            tmp_path / "headers",
            TOURNAMENT,
            task_ids={TASK},
            observed_at=NOW,
            opener=must_not_open,
        )
    assert calls == 0


def test_tree_key_must_match_observation_location_before_network(tmp_path):
    source = source_root(tmp_path)
    wrapper_path = next(
        (source / "observations" / "hf-tree" / "gradients-io-tournaments").rglob(
            "*.json"
        )
    )
    wrapper = json.loads(wrapper_path.read_bytes())
    wrapper["key"] = wrapper["key"].replace("page-0001", "page-0002")
    wrapper_path.write_bytes(sync.canonical_json(wrapper))
    calls = 0

    def must_not_open(request, timeout):
        nonlocal calls
        calls += 1
        raise AssertionError("network must not be touched")

    with pytest.raises(sync.IntegrityError, match="does not match its observation path"):
        headers.harvest(
            source,
            tmp_path / "headers",
            TOURNAMENT,
            task_ids={TASK},
            observed_at=NOW,
            opener=must_not_open,
        )
    assert calls == 0


def test_tree_observation_filename_must_use_append_only_watcher_form(tmp_path):
    source = source_root(tmp_path)
    wrapper_path = next(
        (source / "observations" / "hf-tree" / "gradients-io-tournaments").rglob(
            "*.json"
        )
    )
    wrapper_path.rename(wrapper_path.with_name("obs.json"))

    with pytest.raises(sync.IntegrityError, match="append-only watcher form"):
        headers.collect_candidates(source, TOURNAMENT, {TASK})


def test_tree_observation_filename_binds_content_digest(tmp_path):
    source = source_root(tmp_path)
    wrapper_path = next(
        (source / "observations" / "hf-tree" / "gradients-io-tournaments").rglob(
            "*.json"
        )
    )
    stamp = wrapper_path.name.rsplit("-", 1)[0]
    wrapper_path.rename(wrapper_path.with_name(f"{stamp}-{'f' * 12}.json"))

    with pytest.raises(sync.IntegrityError, match="bind its content digest"):
        headers.collect_candidates(source, TOURNAMENT, {TASK})


def test_identical_watcher_tree_retry_with_unchanged_suffix_is_valid(tmp_path):
    source = source_root(tmp_path)
    wrapper_path = next(
        (source / "observations" / "hf-tree" / "gradients-io-tournaments").rglob(
            "*.json"
        )
    )
    stamp = wrapper_path.name.rsplit("-", 1)[0]
    wrapper_path.rename(wrapper_path.with_name(f"{stamp}-unchanged.json"))

    candidates = headers.collect_candidates(source, TOURNAMENT, {TASK})

    assert len(candidates) == 1
    assert candidates[0]["task_id"] == TASK


def test_repeated_agreeing_tree_evidence_is_aggregated_deterministically(tmp_path):
    source = source_root(tmp_path)
    _, later_path = add_tree(
        source,
        page="page-0002",
        observed_at="2026-08-10T15:01:00Z",
    )

    candidates = headers.collect_candidates(source, TOURNAMENT, {TASK})

    assert len(candidates) == 1
    candidate = candidates[0]
    assert [row["tree_observation"] for row in candidate["tree_observations"]] == [
        row["tree_observation"]
        for row in sorted(candidate["tree_observations"], key=headers._provenance_sort_key)
    ]
    assert len(candidate["tree_observations"]) == 2
    assert candidate["tree_observation"] == later_path


@pytest.mark.parametrize(
    ("oid", "size"),
    [
        ("3" * 64, 4096),
        (LFS_SHA, 8192),
    ],
)
def test_repeated_tree_path_with_conflicting_lfs_identity_fails(tmp_path, oid, size):
    source = source_root(tmp_path)
    add_tree(
        source,
        page="page-0002",
        entries=[tree_entry("checkpoints/last.safetensors", oid=oid, size=size)],
    )

    with pytest.raises(sync.IntegrityError, match="conflicting LFS size or OID"):
        headers.collect_candidates(source, TOURNAMENT, {TASK})


def test_candidate_is_revalidated_against_referenced_tree_before_network(
    tmp_path, monkeypatch
):
    source = source_root(tmp_path)
    candidate = headers.collect_candidates(source, TOURNAMENT, {TASK})[0]
    forged = {**candidate, "lfs_sha256": "3" * 64}
    monkeypatch.setattr(
        headers, "collect_candidates", lambda root, tournament, task_ids: [forged]
    )
    calls = 0

    def must_not_open(request, timeout):
        nonlocal calls
        calls += 1
        raise AssertionError("network must not be touched")

    with pytest.raises(sync.IntegrityError, match="referenced tree entry"):
        headers.harvest(
            source,
            tmp_path / "headers",
            TOURNAMENT,
            task_ids={TASK},
            observed_at=NOW,
            opener=must_not_open,
        )
    assert calls == 0


def test_every_safetensors_tree_entry_is_included_without_an_eligible_plan(tmp_path):
    second_oid = "3" * 64
    source = source_root(
        tmp_path,
        entries=[
            tree_entry("checkpoints/last.safetensors"),
            tree_entry("exports/unplanned.safetensors", oid=second_oid),
        ],
    )
    parsed = {
        "layer.weight": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}
    }
    opener, _ = fake_header_opener(parsed)

    result = headers.harvest(
        source,
        tmp_path / "headers",
        TOURNAMENT,
        task_ids={TASK},
        observed_at=NOW,
        opener=opener,
    )

    inventory = json.loads((tmp_path / "headers" / result["inventory"]).read_bytes())
    assert result["candidate_count"] == 2
    assert {row["path"] for row in inventory["associations"]} == {
        "checkpoints/last.safetensors",
        "exports/unplanned.safetensors",
    }
