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


def public_request_url(source: str, key: str) -> str:
    if source == "gradients-tournament":
        return f"https://api.gradients.io/tournament/{key}/details"
    if source == "gradients-task":
        return f"https://api.gradients.io/auditing/tasks/{key}"
    if source == "acceptance":
        return f"local://acceptance/{key}"
    if source == "hf-model":
        return f"https://huggingface.co/api/models/{key}"
    if source == "hf-revision-manifest":
        repo, revision = key.rsplit("/", 1)
        return f"https://huggingface.co/{repo}/tree/{revision}"
    if source == "hf-tree":
        repo, revision, _page = key.rsplit("/", 2)
        return f"https://huggingface.co/api/models/{repo}/tree/{revision}"
    if source == "hf-file":
        parts = key.split("/")
        return (
            "https://huggingface.co/api/resolve-cache/models/"
            + "/".join(parts[:2])
            + "/"
            + parts[2]
            + "/"
            + "/".join(parts[3:])
        )
    return f"https://example.invalid/{key}"


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
        "request_url": public_request_url(source, key),
        "status": 200,
        "content_sha256": digest,
        "content_bytes": len(body),
        "object": f"objects/sha256/{digest[:2]}/{digest}",
    }
    relative = f"observations/{source}/{key}/{stamp}-{digest[:12]}.json"
    write_regular(root, relative, sync.canonical_json(record))
    return relative, digest


def rewrite_observation(root: Path, relative: str, mutator) -> dict:
    path = root / relative
    record = json.loads(path.read_bytes())
    mutator(record)
    path.write_bytes(sync.canonical_json(record))
    return record


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


def test_tournament_scope_ignores_task_ids_outside_round_task_membership(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    body = sync.canonical_json(
        {
            "tournament_id": TOURNAMENT_A,
            "tournament_type": "image",
            "rounds": [
                {
                    "round_number": 1,
                    "status": "completed",
                    "tasks": [{"task_id": TASK_A}],
                    "untrusted_metadata": {"task_id": TASK_B},
                }
            ],
        }
    )
    add_observation(
        source,
        source="gradients-tournament",
        key=TOURNAMENT_A,
        body=body,
    )

    scope = sync.derive_tournament_scope(
        sync.LocalSource(source),
        sync.LocalSource(source).inventory(),
        TOURNAMENT_A,
    )

    assert scope.task_ids == frozenset({TASK_A})


def test_tournament_scope_is_completed_round_one_only(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    body = sync.canonical_json(
        {
            "tournament_id": TOURNAMENT_A,
            "tournament_type": "image",
            "rounds": [
                {
                    "round_number": 1,
                    "status": "completed",
                    "tasks": [{"task_id": TASK_A}],
                },
                {
                    "round_number": 2,
                    "status": "active",
                    "tasks": [{"task_id": TASK_B}],
                },
            ],
        }
    )
    add_observation(
        source, source="gradients-tournament", key=TOURNAMENT_A, body=body
    )

    scope = sync.derive_tournament_scope(
        sync.LocalSource(source),
        sync.LocalSource(source).inventory(),
        TOURNAMENT_A,
    )

    assert scope.task_ids == frozenset({TASK_A})


def test_tournament_scope_refuses_active_round_one(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    body = sync.canonical_json(
        {
            "tournament_id": TOURNAMENT_A,
            "tournament_type": "image",
            "rounds": [
                {
                    "round_number": 1,
                    "status": "active",
                    "tasks": [{"task_id": TASK_A}],
                }
            ],
        }
    )
    add_observation(
        source, source="gradients-tournament", key=TOURNAMENT_A, body=body
    )

    with pytest.raises(sync.IntegrityError, match="completed Round-1"):
        sync.derive_tournament_scope(
            sync.LocalSource(source),
            sync.LocalSource(source).inventory(),
            TOURNAMENT_A,
        )


def test_tournament_scope_requires_exact_image_tournament_type(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    add_tournament(source, TOURNAMENT_A, TASK_A)
    wrapper_path = next((source / "observations/gradients-tournament").rglob("*.json"))
    wrapper = json.loads(wrapper_path.read_bytes())
    cas_path = source / wrapper["object"]
    body = json.loads(cas_path.read_bytes())
    body["tournament_type"] = "text"
    replacement = sync.canonical_json(body)
    replacement_digest = sha(replacement)
    replacement_path = source / f"objects/sha256/{replacement_digest[:2]}/{replacement_digest}"
    write_regular(source, replacement_path.relative_to(source).as_posix(), replacement)
    wrapper["content_sha256"] = replacement_digest
    wrapper["content_bytes"] = len(replacement)
    wrapper["object"] = replacement_path.relative_to(source).as_posix()
    replacement_wrapper = sync.canonical_json(wrapper)
    wrapper_path.unlink()
    write_regular(
        source,
        wrapper_path.with_name(
            f"20260810T130000.000000Z-{replacement_digest[:12]}.json"
        ).relative_to(source).as_posix(),
        replacement_wrapper,
    )

    with pytest.raises(sync.IntegrityError, match="selected image tournament"):
        sync.derive_tournament_scope(
            sync.LocalSource(source),
            sync.LocalSource(source).inventory(),
            TOURNAMENT_A,
        )


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
    [
        "hidden",
        "HOLDOUT",
        "hold_out",
        "hold-out",
        "test_data",
        "testing",
        "quarantine",
        "eval-derived",
        "eval_data",
        "evaluation_data",
        "eval%5Fderived",
        "h%252569dden",
        "te%252573t_data",
        "hold%25256fut",
    ],
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


def test_full_json_screen_catches_escaped_forbidden_key_beyond_first_chunk(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    body = b'{"pad":"' + (b"a" * (2 * 1024 * 1024)) + b'","test\\u005fdata":"x"}\n'
    relative, digest = add_observation(
        source,
        source="hf-file",
        key="org/repo/rev/config.json",
        body=body,
    )

    destination = tmp_path / "evidence"
    result = sync.sync_archive(sync.LocalSource(source), destination, observed_at=NOW)

    assert result["complete"] is True
    assert result["excluded"]["forbidden_body"] == 1
    assert not (destination / relative).exists()
    assert not (destination / f"objects/sha256/{digest[:2]}/{digest}").exists()


def test_wrapper_source_key_path_smuggling_aborts_before_cas_publication(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    add_tournament(source, TOURNAMENT_A, TASK_A)
    body = b'{"full_pool_row":"must not cross"}\n'
    digest = add_object(source, body)
    record = sync.canonical_json(
        {
            "schema": 1,
            "source": "fixtures",
            "key": f"{TASK_A}/0000",
            "content_sha256": digest,
            "content_bytes": len(body),
            "object": f"objects/sha256/{digest[:2]}/{digest}",
        }
    )
    forged = (
        f"observations/gradients-task/{TASK_A}/"
        f"20260810T000000.000000Z-{digest[:12]}.json"
    )
    write_regular(source, forged, record)

    destination = tmp_path / "evidence"
    result = sync.sync_archive(
        sync.LocalSource(source),
        destination,
        observed_at=NOW,
        tournament_id=TOURNAMENT_A,
    )

    assert result["complete"] is False
    assert result["status"] == "PARTIAL"
    assert any("source does not match" in row["message"] for row in result["errors"])
    assert not (destination / forged).exists()
    assert not (destination / f"objects/sha256/{digest[:2]}/{digest}").exists()


def test_unknown_hf_source_is_out_of_scope_even_with_exact_tournament_and_task(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    add_tournament(source, TOURNAMENT_A, TASK_A)
    body = b'{"public":"metadata"}\n'
    key = (
        "gradients-io-tournaments/"
        f"tournament-{TOURNAMENT_A}-{TASK_A}-5A1iceAB/rev/page-0001"
    )
    relative, digest = add_observation(
        source, source="hf-secret", key=key, body=body
    )

    destination = tmp_path / "evidence"
    result = sync.sync_archive(
        sync.LocalSource(source),
        destination,
        observed_at=NOW,
        tournament_id=TOURNAMENT_A,
    )

    assert result["status"] == "COMPLETE"
    assert result["excluded"]["out_of_scope"] >= 1
    assert not (destination / relative).exists()
    assert not (destination / f"objects/sha256/{digest[:2]}/{digest}").exists()


def test_hf_scope_cannot_smuggle_ids_in_revision_or_file_components(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    add_tournament(source, TOURNAMENT_A, TASK_A)
    body = sync.canonical_json(
        {
            "repo_id": "evil/unrelated",
            "revision": "1" * 40,
            "path": f"notes/tournament-{TOURNAMENT_A}-{TASK_A}-bait/config.yaml",
        }
    )
    key = (
        "evil/unrelated/"
        + "1" * 40
        + f"/notes/tournament-{TOURNAMENT_A}-{TASK_A}-bait/config.yaml"
    )
    relative, digest = add_observation(
        source, source="hf-file", key=key, body=body
    )

    destination = tmp_path / "evidence"
    result = sync.sync_archive(
        sync.LocalSource(source),
        destination,
        observed_at=NOW,
        tournament_id=TOURNAMENT_A,
    )

    assert result["status"] == "COMPLETE"
    assert result["excluded"]["out_of_scope"] >= 1
    assert not (destination / relative).exists()
    assert not (destination / f"objects/sha256/{digest[:2]}/{digest}").exists()


@pytest.mark.parametrize("source_name,key", [
    ("gradients-task", f"{TASK_A}/extra"),
    ("gradients-tournament", f"{TOURNAMENT_A}/extra"),
    ("acceptance", f"{TOURNAMENT_A}/extra"),
])
def test_single_component_public_keys_cannot_gain_nested_suffixes(
    tmp_path, source_name, key
):
    source = tmp_path / "source"
    source.mkdir()
    add_tournament(source, TOURNAMENT_A, TASK_A)
    relative, digest = add_observation(
        source, source=source_name, key=key, body=b'{"public":"metadata"}\n'
    )

    destination = tmp_path / "evidence"
    result = sync.sync_archive(
        sync.LocalSource(source),
        destination,
        observed_at=NOW,
        tournament_id=TOURNAMENT_A,
    )

    assert result["status"] == "COMPLETE"
    assert not (destination / relative).exists()
    assert not (destination / f"objects/sha256/{digest[:2]}/{digest}").exists()


def test_acceptance_body_must_match_exact_tournament_key(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    add_tournament(source, TOURNAMENT_A, TASK_A)
    relative, digest = add_observation(
        source,
        source="acceptance",
        key=TOURNAMENT_A,
        body=sync.canonical_json({"tournament_id": TOURNAMENT_B}),
    )

    destination = tmp_path / "evidence"
    result = sync.sync_archive(
        sync.LocalSource(source),
        destination,
        observed_at=NOW,
        tournament_id=TOURNAMENT_A,
    )

    assert result["status"] == "PARTIAL"
    assert any("acceptance body contradicts" in row["message"] for row in result["errors"])
    assert not (destination / relative).exists()
    assert not (destination / f"objects/sha256/{digest[:2]}/{digest}").exists()


@pytest.mark.parametrize("claimed", [None, True, -1, 999999])
def test_wrapper_content_length_must_exactly_bind_staged_cas(tmp_path, claimed):
    source = tmp_path / "source"
    source.mkdir()
    relative, digest = add_observation(
        source,
        source="gradients-task",
        key=TASK_A,
        body=b'{"status":"completed"}\n',
    )
    wrapper_path = source / relative
    wrapper = json.loads(wrapper_path.read_bytes())
    wrapper["content_bytes"] = claimed
    wrapper_path.write_bytes(sync.canonical_json(wrapper))

    destination = tmp_path / "evidence"
    result = sync.sync_archive(sync.LocalSource(source), destination, observed_at=NOW)

    assert result["status"] == "PARTIAL"
    assert any("content_bytes" in row["message"] for row in result["errors"])
    assert not (destination / relative).exists()
    assert not (destination / f"objects/sha256/{digest[:2]}/{digest}").exists()


def test_wrapper_observed_at_cannot_reorder_append_only_filename(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    relative, digest = add_observation(
        source,
        source="gradients-task",
        key=TASK_A,
        body=b'{"task_id":"11111111-1111-4111-8111-111111111111"}\n',
    )
    wrapper_path = source / relative
    wrapper = json.loads(wrapper_path.read_bytes())
    wrapper["observed_at"] = "2999-01-01T00:00:00Z"
    wrapper_path.write_bytes(sync.canonical_json(wrapper))

    destination = tmp_path / "evidence"
    result = sync.sync_archive(sync.LocalSource(source), destination, observed_at=NOW)

    assert result["status"] == "PARTIAL"
    assert any("wrapper time" in row["message"] for row in result["errors"])
    assert not (destination / relative).exists()
    assert not (destination / f"objects/sha256/{digest[:2]}/{digest}").exists()


@pytest.mark.parametrize("schema", [None, True, 0, 2, "1"])
def test_observation_wrapper_requires_exact_schema_one(tmp_path, schema):
    source = tmp_path / "source"
    source.mkdir()
    relative, digest = add_observation(
        source,
        source="gradients-task",
        key=TASK_A,
        body=b'{"task_id":"11111111-1111-4111-8111-111111111111"}\n',
    )
    wrapper_path = source / relative
    wrapper = json.loads(wrapper_path.read_bytes())
    wrapper["schema"] = schema
    wrapper_path.write_bytes(sync.canonical_json(wrapper))

    result = sync.sync_archive(
        sync.LocalSource(source), tmp_path / "evidence", observed_at=NOW
    )

    assert result["status"] == "PARTIAL"
    assert any("wrapper schema" in row["message"] for row in result["errors"])
    assert not (tmp_path / "evidence" / relative).exists()
    assert not (
        tmp_path / "evidence" / f"objects/sha256/{digest[:2]}/{digest}"
    ).exists()


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


def test_ssh_reader_reuses_a_scoped_local_control_socket():
    source = sync.SSHSource("example.invalid", "/archive")

    command = source._command("list")

    assert "-oControlMaster=auto" in command
    assert "-oControlPersist=30" in command
    control = next(item for item in command if item.startswith("-oControlPath="))
    assert control.startswith("-oControlPath=/tmp/")
    assert control.endswith("-%C")
    assert "sn56-w7-" in control
    # OpenSSH's Unix-domain socket path limit is 104 bytes on macOS.  The
    # longest expansion is the fixed path plus the 40-hex-character %C hash.
    assert len(control.removeprefix("-oControlPath=").replace("%C", "0" * 40)) < 104
    assert command[-2] == "example.invalid"
    assert command[-1].endswith(" list /archive")


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
        source,
        source="gradients-task",
        key=TASK_A,
        body=sync.canonical_json({"task_id": TASK_A, "status": "success"}),
    )
    task_b_path, _ = add_observation(
        source,
        source="gradients-task",
        key=TASK_B,
        body=sync.canonical_json({"task_id": TASK_B, "status": "success"}),
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
    repo_a = f"gradients-io-tournaments/tournament-{TOURNAMENT_A}-{TASK_A}-5A1iceAB"
    revision_a = "1" * 40
    hf_key_a = (
        f"{repo_a}/{revision_a}/config.yaml"
    )
    hf_a_path, _ = add_observation(
        source,
        source="hf-file",
        key=hf_key_a,
        body=sync.canonical_json(
            {
                "repo_id": repo_a,
                "revision": revision_a,
                "path": "config.yaml",
                "test_loss": 0.0123,
            }
        ),
    )
    hf_key_b = (
        "gradients-io-tournaments/"
        f"tournament-{TOURNAMENT_B}-{TASK_B}-5Bob/{'2' * 40}/config.yaml"
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
        + (json.dumps({"event": "task_seen", "task_id": TASK_B}, separators=(",", ":")) + "\n").encode()
        + (
            json.dumps(
                {"event": "mixed", "selected_task": TASK_A, "foreign_task": TASK_B},
                separators=(",", ":"),
            )
            + "\n"
        ).encode(),
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
    for allowed in (tournament_a_path, task_a_path, hf_a_path):
        assert (destination / allowed).is_file()
    for excluded in (tournament_b_path, task_b_path, fixture_a_path, hf_b_path):
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


def test_watcher_full_pool_fixture_namespace_is_never_in_exact_scope():
    scope = sync.TournamentScope(
        tournament_id=TOURNAMENT_A,
        task_ids=frozenset({TASK_A}),
        source_observation="observations/gradients-tournament/source.json",
        source_content_sha256="a" * 64,
    )

    assert sync.observation_in_scope(
        f"observations/fixtures/{TASK_A}/0000/image/observation.json", scope
    ) is False


def test_unscoped_sync_excludes_full_pool_fixture_before_reading_its_cas(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    fixture_body = b"FULL_IMAGE_TEXT_PAIR_BYTES"
    relative, digest = add_observation(
        source,
        source="fixtures",
        key=f"{TASK_A}/0000",
        body=fixture_body,
    )
    destination = tmp_path / "evidence"

    result = sync.sync_archive(sync.LocalSource(source), destination, observed_at=NOW)

    assert result["status"] == "COMPLETE"
    assert not (destination / relative).exists()
    assert not (destination / f"objects/sha256/{digest[:2]}/{digest}").exists()
    assert result["excluded"]["out_of_scope"] == 1


def test_public_test_loss_is_allowed_in_body_filter():
    assert sync.contains_forbidden_body(b'{"test_loss":0.05098}') is False
    assert sync.contains_forbidden_body(b'Ranked first by test_loss') is False
    assert sync.contains_forbidden_path("repo/revision/test_loss.json") is False
    assert sync.contains_forbidden_path("repo/revision/test_data.json") is True


@pytest.mark.parametrize(
    "field",
    ["test_loss_data", "test_loss_rows", "test_loss/set", "test.loss.data"],
)
def test_test_loss_exception_cannot_prefix_a_dataset_surface(field):
    assert sync.contains_forbidden_body(json.dumps({field: "rows"}).encode()) is True
    assert sync.contains_forbidden_path(f"repo/{field}/artifact.json") is True


@pytest.mark.parametrize(
    "field",
    [
        "test_data",
        "test_dataset",
        "test_datasets",
        "test_archive",
        "test_archives",
        "test_images",
        "test_assets",
        "test_prompts",
        "test-rows",
        "test set",
        "eval_data",
        "evaluation_dataset",
        "evaluation_datasets",
        "evaluation_archive",
        "evaluation_archives",
        "evaluation_images",
        "evaluation_assets",
        "eval_prompts",
        "evaluation-data",
        "eval-derived",
        "hidden",
        "holdout",
        "holdouts",
        "quarantine",
    ],
)
def test_dataset_bearing_body_names_are_excluded(field):
    assert sync.contains_forbidden_body(f'{{"{field}":"rows"}}'.encode()) is True


@pytest.mark.parametrize(
    "field",
    ["test/data", "test.data", "te\u200bst_data", "hold.out", "eval/rows"],
)
def test_dataset_bearing_names_cannot_hide_behind_punctuation_or_format_chars(field):
    body = json.dumps({field: "rows"}, ensure_ascii=False).encode()
    assert sync.contains_forbidden_body(body) is True
    assert sync.contains_forbidden_path(f"repo/{field}/artifact.json") is True


def test_canonical_json_rejects_nonfinite_numbers():
    with pytest.raises(ValueError, match="JSON compliant"):
        sync.canonical_json({"score": float("nan")})


def test_plural_holdout_and_quarantine_paths_are_excluded():
    assert sync.contains_forbidden_path("repo/holdouts/rows.json") is True
    assert sync.contains_forbidden_path("repo/quarantines/rows.json") is True


@pytest.mark.parametrize(
    "field",
    [
        "test1",
        "test01",
        "testv1",
        "test_version_2",
        "holdout1",
        "holdout_v2",
        "hidden1",
        "hidden-version-3",
        "eval1",
        "evaluationv2",
        "quarantine1",
        "quarantine_version_4",
        "test%31",
        "test%2531",
        "test１",
        "test١",
        "holdout_v۲",
    ],
)
def test_numeric_and_version_suffixes_remain_prohibited_after_normalization(field):
    assert sync.contains_forbidden_path(f"repo/{field}/artifact.json") is True
    assert sync.contains_forbidden_body(
        json.dumps({field: "rows"}, ensure_ascii=False).encode()
    ) is True


@pytest.mark.parametrize(
    "field",
    [
        "test_loss",
        "checkpoints/1000",
        "contest1",
        "latest1",
        "attestation1",
        "hiddenlayer1",
        "evaluationmetrics1",
    ],
)
def test_numeric_suffix_filter_does_not_capture_unrelated_public_names(field):
    assert sync.contains_forbidden_path(f"repo/{field}/artifact.json") is False
    assert sync.contains_forbidden_body(
        json.dumps({field: "public"}).encode()
    ) is False


@pytest.mark.parametrize(
    "field",
    ["test1", "holdout1", "hidden1", "evaluationv2", "quarantine1"],
)
def test_watcher_sync_excludes_numeric_suffix_names_before_publication(tmp_path, field):
    source = tmp_path / "source"
    source.mkdir()
    relative, digest = add_observation(
        source,
        source="hf-file",
        key=f"org/repo/{'1' * 40}/{field}/config.yaml",
        body=b"learning_rate: 0.0002\n",
    )

    destination = tmp_path / "evidence"
    result = sync.sync_archive(sync.LocalSource(source), destination, observed_at=NOW)

    assert result["status"] == "COMPLETE"
    assert result["excluded"]["forbidden_path"] == 1
    assert not (destination / relative).exists()
    assert not (destination / f"objects/sha256/{digest[:2]}/{digest}").exists()


def test_public_request_provenance_is_source_and_key_bound():
    record = {
        "status": 200,
        "request_url": f"https://api.gradients.io/auditing/tasks/{TASK_A}",
    }
    sync.validate_public_request_provenance("gradients-task", TASK_A, record)
    with pytest.raises(sync.IntegrityError, match="contradicts"):
        sync.validate_public_request_provenance(
            "gradients-task",
            TASK_A,
            {"status": 200, "request_url": "https://attacker.example/task"},
        )
    with pytest.raises(sync.IntegrityError, match="successful HTTP status"):
        sync.validate_public_request_provenance(
            "gradients-task",
            TASK_A,
            {"status": True, "request_url": record["request_url"]},
        )


def test_hf_tree_query_is_source_allowlisted_and_value_free_after_sync(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    key = f"org/repo/{'1' * 40}/page-0002"
    relative, _ = add_observation(
        source,
        source="hf-tree",
        key=key,
        body=b"[]\n",
    )
    secret = "opaque-pagination-secret"
    rewrite_observation(
        source,
        relative,
        lambda value: value.__setitem__(
            "request_url",
            f"https://huggingface.co/api/models/org/repo/tree/{'1' * 40}"
            f"?cursor={secret}&expand=true&recursive=true",
        ),
    )

    destination = tmp_path / "evidence"
    result = sync.sync_archive(sync.LocalSource(source), destination, observed_at=NOW)

    assert result["status"] == "COMPLETE"
    published = (destination / relative).read_bytes()
    wrapper = json.loads(published)
    assert wrapper["request_url"] == (
        f"https://huggingface.co/api/models/org/repo/tree/{'1' * 40}"
    )
    assert wrapper["request_query"] == {
        "redacted": True,
        "keys": ["cursor", "expand", "recursive"],
        "pair_count": 3,
    }
    assert secret.encode() not in published


def test_signed_xet_query_is_source_allowlisted_and_value_free_after_sync(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    key = f"org/repo/{'1' * 40}/checkpoints/last.safetensors"
    relative, _ = add_observation(
        source,
        source="hf-file",
        key=key,
        body=b'{"public":"metadata"}\n',
    )
    secret = "signed-secret-value"
    rewrite_observation(
        source,
        relative,
        lambda value: value.__setitem__(
            "request_url",
            "https://us-east-1.aws.cdn.hf.co/xet-bridge-us/object"
            f"?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature={secret}",
        ),
    )

    destination = tmp_path / "evidence"
    result = sync.sync_archive(sync.LocalSource(source), destination, observed_at=NOW)

    assert result["status"] == "COMPLETE"
    published = (destination / relative).read_bytes()
    wrapper = json.loads(published)
    assert wrapper["request_url"] == (
        "https://us-east-1.aws.cdn.hf.co/xet-bridge-us/object"
    )
    assert wrapper["request_query"] == {
        "redacted": True,
        "keys": ["x-amz-algorithm", "x-amz-signature"],
        "pair_count": 2,
    }
    assert secret.encode() not in published


@pytest.mark.parametrize(
    ("source", "key", "url"),
    [
        (
            "gradients-task",
            TASK_A,
            f"https://api.gradients.io/auditing/tasks/{TASK_A}?token=secret",
        ),
        (
            "hf-tree",
            f"org/repo/{'1' * 40}/page-0001",
            f"https://huggingface.co/api/models/org/repo/tree/{'1' * 40}?token=secret",
        ),
        (
            "hf-tree",
            f"org/repo/{'1' * 40}/page-0001",
            f"https://huggingface.co/api/models/org/repo/tree/{'1' * 40}?cursor=a&cursor=b",
        ),
        (
            "hf-tree",
            f"org/repo/{'1' * 40}/page-0001",
            f"https://huggingface.co/api/models/org/repo/tree/{'1' * 40}?cursor=",
        ),
        (
            "hf-file",
            f"org/repo/{'1' * 40}/checkpoints/last.safetensors",
            "https://us.aws.cdn.hf.co/xet-bridge-us/object?X-Amz-Expires=300",
        ),
    ],
)
def test_public_query_contract_rejects_unexpected_duplicate_empty_and_unsigned_xet(
    source, key, url
):
    with pytest.raises(sync.IntegrityError, match="query"):
        sync.validate_public_request_provenance(
            source,
            key,
            {"status": 200, "request_url": url},
        )


def test_raw_and_redacted_query_metadata_cannot_coexist():
    key = f"org/repo/{'1' * 40}/page-0001"
    with pytest.raises(sync.IntegrityError, match="raw and redacted"):
        sync.validate_public_request_provenance(
            "hf-tree",
            key,
            {
                "status": 200,
                "request_url": (
                    f"https://huggingface.co/api/models/org/repo/tree/{'1' * 40}"
                    "?cursor=secret"
                ),
                "request_query": {
                    "redacted": True,
                    "keys": ["cursor"],
                    "pair_count": 1,
                },
            },
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://us.aws.cdn.hf.co/xet-bridge-us/object%3FX-Amz-Signature%3Dsecret",
        "https://us.aws.cdn.hf.co/xet-bridge-us/object%23X-Amz-Signature%3Dsecret",
        "https://us.aws.cdn.hf.co/xet-bridge-us/object%253FX-Amz-Signature%253Dsecret",
        "https://us.aws.cdn.hf.co/xet-bridge-us/object%2523X-Amz-Signature%253Dsecret",
        "https://us.aws.cdn.hf.co/xet-bridge-us/object%",
        "https://us.aws.cdn.hf.co/xet-bridge-us/object%2",
        "https://us.aws.cdn.hf.co/xet-bridge-us/object%GG",
        "https://us.aws.cdn.hf.co/xet-bridge-us/object?X-Amz-Signature=raw+plus",
        "https://us.aws.cdn.hf.co/xet-bridge-us/object?X-Amz-Signature=%GG",
        (
            "https://us.aws.cdn.hf.co/xet-bridge-us/object/"
            "X-Amz-Credential=SECRET?X-Amz-Signature=sig"
        ),
        (
            "https://us.aws.cdn.hf.co/xet-bridge-us/object"
            "%2FX-Amz-Credential%3DSECRET?X-Amz-Signature=sig"
        ),
        (
            "https://us.aws.cdn.hf.co/xet-bridge-us/object"
            ";X-Amz-Credential=SECRET?X-Amz-Signature=sig"
        ),
    ],
)
def test_xet_encoded_delimiters_malformed_escapes_and_raw_plus_fail_closed(url):
    key = f"org/repo/{'1' * 40}/checkpoints/last.safetensors"
    with pytest.raises(sync.IntegrityError, match="path|query|percent|plus"):
        sync.validate_public_request_provenance(
            "hf-file", key, {"status": 200, "request_url": url}
        )


def test_percent_encoded_plus_is_unambiguous_and_value_free():
    key = f"org/repo/{'1' * 40}/checkpoints/last.safetensors"
    sanitized = sync.validate_public_request_provenance(
        "hf-file",
        key,
        {
            "status": 200,
            "request_url": (
                "https://us.aws.cdn.hf.co/xet-bridge-us/object"
                "?X-Amz-Signature=signed%2Bvalue"
            ),
        },
    )
    assert sanitized["request_url"].endswith("/xet-bridge-us/object")
    assert sanitized["request_query"] == {
        "redacted": True,
        "keys": ["x-amz-signature"],
        "pair_count": 1,
    }
    assert "signed" not in json.dumps(sanitized)


def test_unscoped_observation_query_fails_closed_in_sync(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    relative, digest = add_observation(
        source,
        source="diagnostic",
        key="public/row",
        body=b'{"public":"metadata"}\n',
    )
    rewrite_observation(
        source,
        relative,
        lambda value: value.__setitem__(
            "request_url", "https://example.invalid/public/row?token=secret"
        ),
    )

    destination = tmp_path / "evidence"
    result = sync.sync_archive(sync.LocalSource(source), destination, observed_at=NOW)

    assert result["status"] == "PARTIAL"
    assert any("uncontracted" in row["message"] for row in result["errors"])
    assert not (destination / relative).exists()
    assert not (destination / f"objects/sha256/{digest[:2]}/{digest}").exists()


def test_hf_file_public_xet_redirect_authority_is_allowed_but_not_arbitrary_cdn():
    repo = (
        "gradients-io-tournaments/"
        f"tournament-{TOURNAMENT_A}-{TASK_A}-5A1iceAB"
    )
    key = f"{repo}/{'1' * 40}/checkpoints/loss_log.db"
    sync.validate_public_request_provenance(
        "hf-file",
        key,
        {
            "status": 200,
            "request_url": "https://us.aws.cdn.hf.co/xet-bridge-us/id/object",
        },
    )
    with pytest.raises(sync.IntegrityError, match="contradicts"):
        sync.validate_public_request_provenance(
            "hf-file",
            key,
            {
                "status": 200,
                "request_url": "https://attacker.cdn.hf.co/not-xet/object",
            },
        )


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
