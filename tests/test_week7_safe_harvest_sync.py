from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

import pytest


MODULE = (
    Path(__file__).resolve().parents[1]
    / "ops"
    / "experiments"
    / "week7"
    / "safe_harvest_sync.py"
)
spec = importlib.util.spec_from_file_location("week7_safe_harvest_sync", MODULE)
sync = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = sync
spec.loader.exec_module(sync)

NOW = dt.datetime(2026, 8, 10, 13, 1, 2, 345678, tzinfo=dt.timezone.utc)
TOURNAMENT_A = "tourn_aaaaaaaaaaaaaaaa_20260810"
TOURNAMENT_B = "tourn_bbbbbbbbbbbbbbbb_20260810"
TASK_A = "11111111-1111-4111-8111-111111111111"
TASK_B = "22222222-2222-4222-8222-222222222222"


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_regular(root: Path, relative: str, body: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def add_object(root: Path, body: bytes, *, claimed: str | None = None) -> str:
    digest = claimed or sha(body)
    write_regular(root, f"objects/sha256/{digest[:2]}/{digest}", body)
    return digest


def add_observation(
    root: Path,
    *,
    source: str,
    key: str,
    body: bytes,
    stamp: str = "20260810T130000.000000Z",
) -> tuple[str, str]:
    digest = add_object(root, body)
    record = {
        "schema": 1,
        "observed_at": "2026-08-10T13:00:00Z",
        "source": source,
        "key": key,
        "request_url": f"https://example.invalid/{key}",
        "status": 200,
        "content_sha256": digest,
        "content_bytes": len(body),
        "object": f"objects/sha256/{digest[:2]}/{digest}",
    }
    relative = f"observations/{source}/{key}/{stamp}-{digest[:12]}.json"
    write_regular(root, relative, sync.canonical_json(record))
    return relative, digest


def tree_fingerprint(root: Path) -> dict[str, tuple[int, int, int, str]]:
    result = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        rel = path.relative_to(root).as_posix()
        digest = sha(path.read_bytes()) if stat.S_ISREG(info.st_mode) else ""
        result[rel] = (info.st_mode, info.st_size, info.st_mtime_ns, digest)
    return result


def basic_source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    add_observation(
        root,
        source="gradients-task",
        key="task-1",
        body=b'{"status":"active","model_type":"krea2"}\n',
    )
    write_regular(root, "events.jsonl", b'{"event":"preflight_complete"}\n')
    write_regular(root, ".state/state.sqlite3", b"SQLite format 3 safe public state")
    return root


def add_tournament(root: Path, tournament_id: str, task_id: str, **task_fields) -> str:
    body = sync.canonical_json(
        {
            "tournament_id": tournament_id,
            "tournament_type": "image",
            "status": "completed",
            "rounds": [
                {
                    "round_number": 1,
                    "status": "completed",
                    "tasks": [
                        {
                            "task_id": task_id,
                            "task_type": "ImageTask",
                            **task_fields,
                        }
                    ],
                }
            ],
        }
    )
    relative, _ = add_observation(
        root, source="gradients-tournament", key=tournament_id, body=body
    )
    return relative


def test_create_only_sync_copies_reachable_cas_and_timestamped_mutable_files(tmp_path):
    source = basic_source(tmp_path)
    orphan = add_object(source, b"unreferenced")
    before = tree_fingerprint(source)
    destination = tmp_path / "evidence"

    result = sync.sync_archive(sync.LocalSource(source), destination, observed_at=NOW)

    assert result["status"] == "COMPLETE"
    assert result["observations"] == {"eligible": 1, "published": 1, "preexisting": 0}
    assert result["cas"]["published"] == 1
    assert not (destination / f"objects/sha256/{orphan[:2]}/{orphan}").exists()
    stamp = sync.utc_token(NOW)
    assert (destination / f"snapshots/{stamp}/events.jsonl").is_file()
    assert (destination / f"snapshots/{stamp}/.state/state.sqlite3").is_file()
    ledger = json.loads((destination / f"ledgers/{stamp}.json").read_text())
    assert ledger["complete"] is True
    assert sha((destination / f"ledgers/{stamp}.json").read_bytes()) == result["ledger_sha256"]
    assert tree_fingerprint(source) == before


@pytest.mark.parametrize(
    "forbidden",
    ["hidden", "HOLDOUT", "test_data", "quarantine", "eval-derived", "eval%5Fderived"],
)
def test_hf_path_filter_is_case_and_encoding_aware(tmp_path, forbidden):
    source = tmp_path / "source"
    source.mkdir()
    relative, digest = add_observation(
        source,
        source="hf-file",
        key=f"org/repo/rev/{forbidden}/config.yaml",
        body=b"learning_rate: 0.0002\n",
    )
    destination = tmp_path / "evidence"

    result = sync.sync_archive(sync.LocalSource(source), destination, observed_at=NOW)

    assert result["complete"] is True
    assert result["excluded"]["forbidden_path"] == 1
    assert not (destination / relative).exists()
    assert not (destination / f"objects/sha256/{digest[:2]}/{digest}").exists()
    ledger = (destination / f"ledgers/{sync.utc_token(NOW)}.json").read_text().lower()
    # The policy vocabulary itself is recorded, but the excluded path is not.
    assert "org/repo/rev" not in ledger


def test_hf_body_and_nested_raw_body_are_screened_before_publication(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    raw = b"ordinary config\n# hidden evaluation row\n"
    raw_digest = add_object(source, raw)
    manifest = sync.canonical_json(
        {"path": "config.yaml", "object_sha256": raw_digest, "size": len(raw)}
    )
    relative, manifest_digest = add_observation(
        source,
        source="hf-file",
        key="org/repo/rev/config.yaml",
        body=manifest,
    )
    destination = tmp_path / "evidence"

    result = sync.sync_archive(sync.LocalSource(source), destination, observed_at=NOW)

    assert result["complete"] is True
    assert result["excluded"]["forbidden_body"] == 1
    assert not (destination / relative).exists()
    assert not (destination / f"objects/sha256/{manifest_digest[:2]}/{manifest_digest}").exists()
    assert not (destination / f"objects/sha256/{raw_digest[:2]}/{raw_digest}").exists()


def test_allowed_hf_manifest_copies_manifest_and_raw_object(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    raw = b"learning_rate: 0.0002\n"
    raw_digest = add_object(source, raw)
    manifest = sync.canonical_json(
        {"path": "config.yaml", "object_sha256": raw_digest, "size": len(raw)}
    )
    relative, manifest_digest = add_observation(
        source,
        source="hf-file",
        key="org/repo/rev/config.yaml",
        body=manifest,
    )

    destination = tmp_path / "evidence"
    result = sync.sync_archive(sync.LocalSource(source), destination, observed_at=NOW)

    assert result["complete"] is True
    assert (destination / relative).is_file()
    assert (destination / f"objects/sha256/{manifest_digest[:2]}/{manifest_digest}").is_file()
    assert (destination / f"objects/sha256/{raw_digest[:2]}/{raw_digest}").read_bytes() == raw
    assert result["cas"]["published"] == 2


def test_source_symlink_is_never_followed(tmp_path):
    source = basic_source(tmp_path)
    outside = tmp_path / "outside-secret"
    outside.write_text("secret")
    linked = source / "observations" / "hf-file" / "org" / "repo" / "linked.json"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(outside)
    destination = tmp_path / "evidence"

    result = sync.sync_archive(sync.LocalSource(source), destination, observed_at=NOW)

    assert result["complete"] is True
    assert result["excluded"]["symlink"] >= 1
    assert not (destination / linked.relative_to(source)).exists()
    assert b"secret" not in b"".join(
        path.read_bytes() for path in destination.rglob("*") if path.is_file()
    )


def test_destination_symlink_component_aborts(tmp_path):
    source = basic_source(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "evidence"
    destination.mkdir()
    (destination / "objects").symlink_to(outside)

    with pytest.raises(sync.IntegrityError, match="root identity"):
        sync.sync_archive(sync.LocalSource(source), destination, observed_at=NOW)
    assert not list(outside.iterdir())


def test_cas_filename_mismatch_is_partial_and_never_published(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    claimed = "a" * 64
    add_object(source, b"wrong bytes", claimed=claimed)
    record = {
        "source": "gradients-task",
        "key": "task-1",
        "content_sha256": claimed,
        "object": f"objects/sha256/{claimed[:2]}/{claimed}",
    }
    write_regular(source, "observations/gradients-task/task-1/x.json", sync.canonical_json(record))
    destination = tmp_path / "evidence"

    result = sync.sync_archive(sync.LocalSource(source), destination, observed_at=NOW)

    assert result["status"] == "PARTIAL"
    assert result["errors"][0]["class"] == "IntegrityError"
    assert not (destination / f"objects/sha256/{claimed[:2]}/{claimed}").exists()


def test_repeat_is_create_only_and_reverifies_preexisting_cas(tmp_path):
    source = basic_source(tmp_path)
    destination = tmp_path / "evidence"
    first = sync.sync_archive(sync.LocalSource(source), destination, observed_at=NOW)
    cas_path = next(
        destination / row["path"] for row in first["files"] if row["path"].startswith("objects/")
    )
    original = cas_path.read_bytes()
    later = NOW + dt.timedelta(seconds=1)

    second = sync.sync_archive(sync.LocalSource(source), destination, observed_at=later)

    assert second["complete"] is True
    assert second["cas"]["preexisting"] == 1
    assert cas_path.read_bytes() == original
    assert (destination / f"ledgers/{sync.utc_token(NOW)}.json").is_file()
    assert (destination / f"ledgers/{sync.utc_token(later)}.json").is_file()


def test_repeat_with_corrupt_preexisting_cas_fails_closed(tmp_path):
    source = basic_source(tmp_path)
    destination = tmp_path / "evidence"
    first = sync.sync_archive(sync.LocalSource(source), destination, observed_at=NOW)
    cas_path = next(
        destination / row["path"] for row in first["files"] if row["path"].startswith("objects/")
    )
    # Simulate post-sync disk corruption; the next run must not trust the name.
    cas_path.write_bytes(b"corrupt")

    later = NOW + dt.timedelta(seconds=1)
    result = sync.sync_archive(sync.LocalSource(source), destination, observed_at=later)

    assert result["status"] == "PARTIAL"
    assert any("different bytes" in row["message"] for row in result["errors"])


def test_mutable_snapshots_are_content_screened(tmp_path):
    source = basic_source(tmp_path)
    (source / "events.jsonl").write_text('{"event":"test_rows_seen"}\n')
    destination = tmp_path / "evidence"

    result = sync.sync_archive(sync.LocalSource(source), destination, observed_at=NOW)

    assert result["complete"] is True
    assert result["excluded"]["forbidden_body"] == 1
    assert not (destination / f"snapshots/{sync.utc_token(NOW)}/events.jsonl").exists()


def test_exact_tournament_scope_derives_tasks_and_excludes_cross_tournament(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    tournament_a_path = add_tournament(
        source,
        TOURNAMENT_A,
        TASK_A,
        participant_scores=[{"hotkey": "5Alice", "test_loss": 0.0123}],
    )
    tournament_b_path = add_tournament(source, TOURNAMENT_B, TASK_B)
    task_a_path, _ = add_observation(
        source, source="gradients-task", key=TASK_A, body=b'{"status":"success"}\n'
    )
    task_b_path, _ = add_observation(
        source, source="gradients-task", key=TASK_B, body=b'{"status":"success"}\n'
    )
    fixture_a_path, _ = add_observation(
        source,
        source="fixtures",
        key=f"{TASK_A}/0000/image",
        body=sync.canonical_json({"task_id": TASK_A, "object_sha256": "f" * 64}),
    )
    add_observation(
        source,
        source="gradients-scores",
        key="date-wide-board",
        body=sync.canonical_json({"tournaments": [TOURNAMENT_A, TOURNAMENT_B]}),
    )
    hf_key_a = (
        "gradients-io-tournaments/"
        f"tournament-{TOURNAMENT_A}-{TASK_A}-5Alice/rev/config.yaml"
    )
    hf_a_path, _ = add_observation(
        source,
        source="hf-file",
        key=hf_key_a,
        body=sync.canonical_json({"path": "config.yaml", "test_loss": 0.0123}),
    )
    hf_key_b = (
        "gradients-io-tournaments/"
        f"tournament-{TOURNAMENT_B}-{TASK_B}-5Bob/rev/config.yaml"
    )
    hf_b_path, _ = add_observation(
        source,
        source="hf-file",
        key=hf_key_b,
        body=sync.canonical_json({"path": "config.yaml"}),
    )
    write_regular(
        source,
        "events.jsonl",
        (json.dumps({"event": "task_seen", "task_id": TASK_A}, separators=(",", ":")) + "\n").encode()
        + (json.dumps({"event": "task_seen", "task_id": TASK_B}, separators=(",", ":")) + "\n").encode(),
    )
    write_regular(source, ".state/state.sqlite3", b"date-wide mutable state")

    destination = tmp_path / "evidence"
    result = sync.sync_archive(
        sync.LocalSource(source),
        destination,
        observed_at=NOW,
        tournament_id=TOURNAMENT_A,
    )

    assert result["complete"] is True
    assert result["scope"]["task_ids"] == [TASK_A]
    for allowed in (tournament_a_path, task_a_path, fixture_a_path, hf_a_path):
        assert (destination / allowed).is_file()
    for excluded in (tournament_b_path, task_b_path, hf_b_path):
        assert not (destination / excluded).exists()
    assert result["excluded"]["out_of_scope"] >= 4
    stamp = sync.utc_token(NOW)
    filtered_events = (destination / f"snapshots/{stamp}/events.filtered.jsonl").read_text()
    assert TASK_A in filtered_events
    assert TASK_B not in filtered_events
    state = json.loads((destination / f"snapshots/{stamp}/state-selection.json").read_text())
    assert state["task_ids"] == [TASK_A]
    assert state["raw_sqlite_copied"] is False
    assert not (destination / f"snapshots/{stamp}/.state/state.sqlite3").exists()


def test_public_test_loss_is_allowed_in_body_filter():
    assert sync.contains_forbidden_body(b'{"test_loss":0.05098}') is False
    assert sync.contains_forbidden_body(b'Ranked first by test_loss') is False
    assert sync.contains_forbidden_path("repo/revision/test_loss.json") is False
    assert sync.contains_forbidden_path("repo/revision/test_data.json") is True


@pytest.mark.parametrize(
    "field",
    [
        "test_data",
        "test-rows",
        "test set",
        "eval_data",
        "evaluation-data",
        "eval-derived",
        "hidden",
        "holdout",
        "quarantine",
    ],
)
def test_dataset_bearing_body_names_are_excluded(field):
    assert sync.contains_forbidden_body(f'{{"{field}":"rows"}}'.encode()) is True


def test_root_identity_mismatch_aborts_before_writing(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    add_tournament(source, TOURNAMENT_A, TASK_A)
    add_tournament(source, TOURNAMENT_B, TASK_B)
    destination = tmp_path / "evidence"
    first = sync.sync_archive(
        sync.LocalSource(source),
        destination,
        observed_at=NOW,
        tournament_id=TOURNAMENT_A,
    )
    assert first["complete"] is True
    before = tree_fingerprint(destination)

    with pytest.raises(sync.IntegrityError, match="root identity"):
        sync.sync_archive(
            sync.LocalSource(source),
            destination,
            observed_at=NOW + dt.timedelta(seconds=1),
            tournament_id=TOURNAMENT_B,
        )

    assert tree_fingerprint(destination) == before


def test_exact_scope_snapshots_inventory_time_events_prefix_when_source_grows(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    add_tournament(source_root, TOURNAMENT_A, TASK_A)
    original = (
        json.dumps({"event": "task_seen", "task_id": TASK_A}, separators=(",", ":"))
        + "\n"
    ).encode()
    appended = (
        json.dumps({"event": "later", "task_id": TASK_A}, separators=(",", ":"))
        + "\n"
    ).encode()
    write_regular(source_root, "events.jsonl", original)

    class GrowingAfterInventory:
        label = "growing-test-source"

        def __init__(self, root: Path):
            self.delegate = sync.LocalSource(root)

        def inventory(self):
            rows = self.delegate.inventory()
            with (source_root / "events.jsonl").open("ab") as handle:
                handle.write(appended)
            return rows

        def iter_bytes(self, relative: str):
            yield from self.delegate.iter_bytes(relative)

    destination = tmp_path / "evidence"
    result = sync.sync_archive(
        GrowingAfterInventory(source_root),
        destination,
        observed_at=NOW,
        tournament_id=TOURNAMENT_A,
    )

    assert result["complete"] is True
    snapshot = destination / f"snapshots/{sync.utc_token(NOW)}/events.filtered.jsonl"
    assert snapshot.read_bytes() == original
    assert appended not in snapshot.read_bytes()
