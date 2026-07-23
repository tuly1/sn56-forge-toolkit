"""Post-selection public-upload and private-recorder contract tests."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
import time

import pytest

from forge import telemetry
from forge import cli
from forge.clock import Deadline
from forge.data.schema import ImageSpec
from forge.tasks import checkpoints, dispatch, publication


@pytest.fixture(autouse=True)
def _isolated_telemetry(monkeypatch, tmp_path):
    monkeypatch.setattr(telemetry, "_PRIVATE_ROOT", str(tmp_path / "private"))
    monkeypatch.setattr(telemetry, "_RUN_NONCE", "publication-test-run")
    monkeypatch.setattr(telemetry, "_BOUND_RUN_KEYS", {})
    monkeypatch.setattr(telemetry, "_LATEST_PRIVATE_RECORDS", {})
    monkeypatch.setattr(telemetry, "_t0", time.monotonic())
    monkeypatch.setattr(
        telemetry,
        "_data",
        {
            "schema": 1,
            "meta": {},
            "env": {},
            "events": [],
            "train_curve": [],
            "eval_curve": [],
            "samples": {},
        },
    )
    checkpoints._ACTIVE_RUNS.clear()


def _write_st(path, tag=""):
    header = json.dumps(
        {
            "__metadata__": {"tag": tag},
            "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        }
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.0))
    return path.read_bytes()


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_hf_token() -> str:
    return "hf_" + "x" * 32


def _write_loss_db(path, state, losses):
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE steps (step INTEGER PRIMARY KEY, wall_time REAL NOT NULL);
            CREATE TABLE metric_keys (
                key TEXT PRIMARY KEY, first_seen_step INTEGER, last_seen_step INTEGER
            );
            CREATE TABLE metrics (
                step INTEGER NOT NULL, key TEXT NOT NULL, value_real REAL,
                value_text TEXT, PRIMARY KEY (step, key)
            );
            """
        )
        conn.execute(
            "INSERT INTO metric_keys VALUES ('loss/loss', 1, ?)", (len(losses),)
        )
        for step, loss in enumerate(losses, 1):
            conn.execute(
                "INSERT INTO steps VALUES (?, ?)",
                (step, state["started_unix"] + step / 1000.0),
            )
            conn.execute(
                "INSERT INTO metrics VALUES (?, 'loss/loss', ?, NULL)",
                (step, loss),
            )


def test_telemetry_public_projection_is_strict_and_hash_bound(tmp_path):
    root = tmp_path / "task" / "repo"
    root.mkdir(parents=True)
    telemetry.init(
        task_id="private-task",
        model_type="krea2",
        lr=0.0002,
        steps=367,
        operator_path="/Users/operator/private/config.yaml",
    )
    telemetry._data["env"].update({"gpu_name": "secret-machine", "token": "hf_secret"})
    telemetry.train_point(1, 0.123, 0.0002)
    telemetry.sample("gpu_peak_mb", 35000)
    telemetry.event(
        "handler_failed",
        error=f"RuntimeError: /root/private {_fake_hf_token()}",
        selected_step=294,
    )

    telemetry.write_into(str(root))

    private_path = telemetry.private_record_path(str(root))
    assert os.path.isfile(private_path)
    assert os.path.commonpath((private_path, str(root))) != str(root)
    private_bytes = open(private_path, "rb").read()
    public_bytes = (root / "forge_run.json").read_bytes()
    public = json.loads(public_bytes)

    assert set(public) == {"schema", "kind", "private_record_sha256", "events"}
    assert public["schema"] == 2
    assert public["kind"] == "forge-public-run-recorder"
    assert public["private_record_sha256"] == hashlib.sha256(private_bytes).hexdigest()
    assert public["events"] == [
        {
            "failure_class": "RuntimeError",
            "name": "handler_failed",
            "t": public["events"][0]["t"],
        }
    ]
    assert set(public["events"][0]) == {"t", "name", "failure_class"}
    for forbidden in (
        b"private-task",
        b"krea2",
        b"0.0002",
        b"367",
        b"secret-machine",
        b"selected_step",
        b"/root/",
        b"/Users/",
        b"hf_",
        b"train_curve",
    ):
        assert forbidden not in public_bytes
    assert not (root / "forge_run.full.json").exists()


def test_early_public_write_replaces_legacy_full_recorder(tmp_path):
    root = tmp_path / "task" / "repo"
    root.mkdir(parents=True)
    (root / "forge_run.json").write_text(
        json.dumps({"meta": {"lr": 0.0002, "token": _fake_hf_token()}}),
        encoding="utf-8",
    )
    telemetry.event("checkpoint_scope_started", private_value="do-not-publish")

    telemetry.write_into(str(root))

    public = json.loads((root / "forge_run.json").read_text())
    assert set(public) == {"schema", "kind", "private_record_sha256", "events"}
    assert "private_value" not in (root / "forge_run.json").read_text()
    assert "hf_" not in (root / "forge_run.json").read_text()


def test_cli_replaces_stale_public_recorder_directory_before_handler(
    tmp_path, monkeypatch
):
    root = tmp_path / "task" / "repo"
    stale = root / "forge_run.json"
    stale.mkdir(parents=True)
    (stale / "forge_run.full.json").write_text(
        json.dumps({"meta": {"token": _fake_hf_token()}}),
        encoding="utf-8",
    )
    spec = ImageSpec.build(
        task_id="task",
        model="krea/Krea-2-Raw",
        model_type="krea2",
        expected_repo_name="repo",
        trigger_word=None,
        dataset_zip=None,
    )
    monkeypatch.setattr(type(spec), "save_root", property(lambda self: str(root)))
    monkeypatch.setattr(type(spec), "output_dir", property(lambda self: str(root)))
    observed = {}

    def fake_handler(current_spec, _deadline):
        recorder = root / "forge_run.json"
        observed["is_file"] = recorder.is_file()
        observed["record"] = json.loads(recorder.read_text())
        scope = checkpoints.ensure_run(
            current_spec.save_root, current_spec.expected_repo_name
        )
        _write_st(root / "repo.safetensors", "final")
        assert checkpoints.finalize(str(root), "repo", scope) is not None

    monkeypatch.setattr(dispatch, "for_model_type", lambda _model_type: fake_handler)

    cli._run(
        spec,
        Deadline.from_hours(
            1.0, started_monotonic=time.monotonic(), export_reserve_s=0.0
        ),
    )

    assert observed["is_file"] is True
    assert set(observed["record"]) == {
        "schema",
        "kind",
        "private_record_sha256",
        "events",
    }
    assert not list(root.rglob("forge_run.full*.json"))


def test_private_bundle_symlink_cannot_redirect_full_record_into_upload(tmp_path):
    root = tmp_path / "task" / "repo"
    leak = root / "nested-leak"
    leak.mkdir(parents=True)
    telemetry.bind_private_bundle(str(root), "attempt-symlink-proof")
    bundle = telemetry.private_bundle_dir(str(root))
    os.makedirs(os.path.dirname(bundle), exist_ok=True)
    os.symlink(leak, bundle)
    telemetry.init(lr=0.0002, steps=367)

    telemetry.write_into(str(root))

    assert not list(root.rglob("forge_run.full*.json"))
    public = json.loads((root / "forge_run.json").read_text())
    assert set(public) == {"schema", "kind", "private_record_sha256", "events"}


def test_publication_runs_after_loss_selection_and_preserves_model_bytes(tmp_path):
    root = tmp_path / "task" / "repo"
    root.mkdir(parents=True)
    scope = checkpoints.begin_run(str(root), "repo")
    _write_st(root / "repo_000000045.safetensors", "45")
    early = _write_st(root / "repo_000000090.safetensors", "90")
    _write_st(root / "repo_000000135.safetensors", "135")
    _write_st(root / "repo.safetensors", "final")
    losses = [0.16] * 45 + [0.10] * 45 + [0.18] * 45 + [0.60] * 45
    _write_loss_db(root / "loss_log.db", scope, losses)
    (root / "loss_log.db-wal").write_bytes(b"")
    (root / "loss_log.db-shm").write_bytes(b"")
    (root / "config.yaml").write_text("private_recipe: true\n", encoding="utf-8")
    (root / "learnable_snr.json").write_text("{}", encoding="utf-8")
    record = checkpoints.finalize(str(root), "repo", scope)
    assert record["source"] == "training_loss_divergence"
    assert (root / "last.safetensors").read_bytes() == early
    before = _sha256(root / "last.safetensors")
    telemetry.event("checkpoint_finalized", source=record["source"], selected_step=90)
    telemetry.event("run_complete")

    result = publication.finalize_public_bundle(str(root))

    assert result["complete"] is True
    assert result["selection_attested"] is True
    assert result["artifact_sha256"] == before
    assert _sha256(root / "last.safetensors") == before
    for name in publication._PRIVATE_SIDECARS:
        assert not (root / name).exists()
        assert not (root / f"{name}.tmp").exists()
    private_dir = telemetry.private_bundle_dir(str(root))
    for name in (
        ".forge_checkpoint_scope.json",
        "forge_checkpoint_selection.json",
        "loss_log.db",
        "loss_log.db-wal",
        "loss_log.db-shm",
        "config.yaml",
        "learnable_snr.json",
    ):
        assert os.path.isfile(os.path.join(private_dir, name))
    public = json.loads((root / "forge_run.json").read_text())
    private_path = telemetry.private_record_path(str(root))
    assert public["private_record_sha256"] == _sha256_bytes(open(private_path, "rb").read())
    assert all(set(event).issubset({"t", "name", "failure_class"}) for event in public["events"])


def test_archive_failure_removes_private_sidecar_and_reports_incomplete(
    tmp_path, monkeypatch
):
    root = tmp_path / "task" / "repo"
    root.mkdir(parents=True)
    scope = checkpoints.begin_run(str(root), "repo")
    final = _write_st(root / "repo.safetensors", "final")
    record = checkpoints.finalize(str(root), "repo", scope)
    assert record is not None
    (root / "config.yaml").write_text("secret: recipe", encoding="utf-8")
    (root / "loss_log.db").write_bytes(b"private-db")
    real_copy = publication._copy_open_fd

    def fail_config_archive(source_fd, destination):
        if os.path.basename(destination) == "config.yaml":
            for detached_name in (
                "config.yaml",
                "loss_log.db",
                ".forge_checkpoint_scope.json",
                "forge_checkpoint_selection.json",
            ):
                assert not (root / detached_name).exists()
            raise OSError("forced archive failure")
        return real_copy(source_fd, destination)

    monkeypatch.setattr(publication, "_copy_open_fd", fail_config_archive)
    result = publication.finalize_public_bundle(str(root))

    assert result["complete"] is False
    assert any(error.startswith("archive_failed:config.yaml") for error in result["errors"])
    assert not (root / "config.yaml").exists()
    assert (root / "last.safetensors").read_bytes() == final
    public = json.loads((root / "forge_run.json").read_text())
    assert set(public) == {"schema", "kind", "private_record_sha256", "events"}


def test_terminal_scrub_is_safe_but_incomplete_without_selection(tmp_path):
    root = tmp_path / "task" / "repo"
    root.mkdir(parents=True)
    (root / "config.yaml").write_text("private_recipe: true", encoding="utf-8")
    telemetry.event("fallback_no_current_or_prior_checkpoint")

    result = publication.finalize_public_bundle(str(root))

    assert result["complete"] is False
    assert "selection_attestation_failed" in result["errors"]
    assert not (root / "config.yaml").exists()
    assert (root / "forge_run.json").is_file()


def test_stale_selection_cannot_attest_a_new_attempt(tmp_path):
    root = tmp_path / "task" / "repo"
    root.mkdir(parents=True)
    old_scope = checkpoints.begin_run(str(root), "repo")
    _write_st(root / "repo.safetensors", "old-final")
    assert checkpoints.finalize(str(root), "repo", old_scope) is not None
    # A retry starts and produces no current checkpoint/selection.  The old
    # selection JSON remains byte-identical but is now part of the pre-run scope.
    checkpoints.begin_run(str(root), "repo")

    result = publication.finalize_public_bundle(str(root))

    assert result["complete"] is False
    assert result["selection_attested"] is False
    assert "selection_attestation_failed" in result["errors"]


def test_stale_public_recorder_directory_is_removed_recursively(tmp_path):
    root = tmp_path / "task" / "repo"
    root.mkdir(parents=True)
    scope = checkpoints.begin_run(str(root), "repo")
    _write_st(root / "repo.safetensors", "final")
    assert checkpoints.finalize(str(root), "repo", scope) is not None
    stale = root / "forge_run.json"
    stale.mkdir()
    (stale / "forge_run.full.json").write_text(
        '{"meta":{"lr":0.0002}}', encoding="utf-8"
    )

    result = publication.finalize_public_bundle(str(root))

    assert result["complete"] is True
    assert (root / "forge_run.json").is_file()
    assert not (root / "forge_run.json" / "forge_run.full.json").exists()


def test_root_digest_named_private_recorder_is_archived(tmp_path):
    root = tmp_path / "task" / "repo"
    root.mkdir(parents=True)
    scope = checkpoints.begin_run(str(root), "repo")
    _write_st(root / "repo.safetensors", "final")
    assert checkpoints.finalize(str(root), "repo", scope) is not None
    leaked_name = f"forge_run.full.{'a' * 64}.json"
    (root / leaked_name).write_text(
        json.dumps({"meta": {"lr": 0.0002, "token": _fake_hf_token()}}),
        encoding="utf-8",
    )

    result = publication.finalize_public_bundle(str(root))

    assert result["complete"] is True
    assert not (root / leaked_name).exists()
    assert os.path.isfile(
        os.path.join(telemetry.private_bundle_dir(str(root)), leaked_name)
    )


def test_nested_digest_named_private_recorder_directory_is_removed(tmp_path):
    root = tmp_path / "task" / "repo"
    root.mkdir(parents=True)
    scope = checkpoints.begin_run(str(root), "repo")
    _write_st(root / "repo.safetensors", "final")
    assert checkpoints.finalize(str(root), "repo", scope) is not None
    leaked = root / "nested" / f"forge_run.full.{'a' * 64}.json"
    leaked.mkdir(parents=True)
    (leaked / "secret.txt").write_text(
        f"token={_fake_hf_token()}", encoding="utf-8"
    )

    result = publication.finalize_public_bundle(str(root))

    assert result["complete"] is True
    assert not leaked.exists()
    assert not list(root.rglob("forge_run.full*.json"))


def test_nested_legacy_public_recorder_temp_slots_are_removed(tmp_path):
    root = tmp_path / "task" / "repo"
    root.mkdir(parents=True)
    scope = checkpoints.begin_run(str(root), "repo")
    _write_st(root / "repo.safetensors", "final")
    assert checkpoints.finalize(str(root), "repo", scope) is not None
    leaked_file = root / "file-slot" / "forge_run.json.tmp"
    leaked_file.parent.mkdir()
    leaked_file.write_text(
        json.dumps({"meta": {"token": _fake_hf_token()}}),
        encoding="utf-8",
    )
    leaked_dir = root / "dir-slot" / "forge_run.json.tmp"
    leaked_dir.mkdir(parents=True)
    (leaked_dir / "secret.bin").write_bytes(b"private recorder")

    result = publication.finalize_public_bundle(str(root))

    assert result["complete"] is True
    assert not leaked_file.exists()
    assert not leaked_dir.exists()


def test_public_inventory_scrubs_unknown_text_binary_and_symlink(tmp_path):
    root = tmp_path / "task" / "repo"
    root.mkdir(parents=True)
    scope = checkpoints.begin_run(str(root), "repo")
    _write_st(root / "repo.safetensors", "final")
    assert checkpoints.finalize(str(root), "repo", scope) is not None
    nested = root / "unexpected"
    nested.mkdir()
    notes = nested / "notes.txt"
    notes.write_text(
        f"token={_fake_hf_token()}", encoding="utf-8"
    )
    opaque = nested / "debug.dat"
    opaque.write_bytes(b"private recipe bytes")
    fake_model = nested / "secret.safetensors"
    fake_model.write_bytes(f"token={_fake_hf_token()}".encode())
    _write_st(nested / "candidate.safetensors", "nested-valid")
    _write_st(root / "rogue.safetensors", "rogue-valid")
    target = tmp_path / "outside.safetensors"
    _write_st(target, "outside")
    symlink = nested / "linked.safetensors"
    symlink.symlink_to(target)

    result = publication.finalize_public_bundle(str(root))

    assert result["complete"] is True
    assert not notes.exists()
    assert not opaque.exists()
    assert not fake_model.exists()
    assert not (nested / "candidate.safetensors").exists()
    assert not (root / "rogue.safetensors").exists()
    assert not symlink.exists()
    assert target.exists()
    assert sorted(path.name for path in root.rglob("*.safetensors")) == [
        "last.safetensors",
        "repo.safetensors",
    ]
    assert not any(
        child.is_dir() and any(child.rglob("*.safetensors"))
        for child in root.iterdir()
    )


def test_attempt_binding_isolates_private_bundles_for_same_upload_root(tmp_path):
    root = tmp_path / "task" / "repo"
    root.mkdir(parents=True)
    telemetry.bind_private_bundle(str(root), "attempt-00000001")
    first = telemetry.private_bundle_dir(str(root))
    telemetry.bind_private_bundle(str(root), "attempt-00000002")
    second = telemetry.private_bundle_dir(str(root))

    assert first != second
    assert os.path.dirname(first) == telemetry._PRIVATE_ROOT
    assert os.path.dirname(second) == telemetry._PRIVATE_ROOT
    assert os.path.commonpath((first, str(root))) != str(root)
    assert os.path.commonpath((second, str(root))) != str(root)


def test_retry_start_does_not_overwrite_prior_immutable_private_record(tmp_path):
    root = tmp_path / "task" / "repo"
    root.mkdir(parents=True)
    telemetry.start_run(task_id="attempt-one")
    telemetry.bind_private_bundle(str(root), "attempt-00000001")
    telemetry.event("run_complete")
    telemetry.write_into(str(root))
    first_public = json.loads((root / "forge_run.json").read_text())
    first_path = telemetry.private_record_path_for_digest(
        str(root), first_public["private_record_sha256"]
    )
    first_bytes = open(first_path, "rb").read()

    telemetry.start_run(task_id="attempt-two")
    telemetry.write_public_snapshot(str(root))

    assert open(first_path, "rb").read() == first_bytes
    telemetry.bind_private_bundle(str(root), "attempt-00000002")
    telemetry.event("run_complete")
    telemetry.write_into(str(root))
    second_public = json.loads((root / "forge_run.json").read_text())
    assert second_public["private_record_sha256"] != first_public[
        "private_record_sha256"
    ]


def test_final_public_write_failure_is_detected_without_touching_model(
    tmp_path, monkeypatch
):
    root = tmp_path / "task" / "repo"
    root.mkdir(parents=True)
    scope = checkpoints.begin_run(str(root), "repo")
    model = _write_st(root / "repo.safetensors", "final")
    assert checkpoints.finalize(str(root), "repo", scope) is not None
    real_write_public = telemetry.write_public
    calls = 0

    def fail_final_write(path, digest):
        nonlocal calls
        calls += 1
        if calls >= 2:
            return False
        return real_write_public(path, digest)

    monkeypatch.setattr(telemetry, "write_public", fail_final_write)
    result = publication.finalize_public_bundle(str(root))

    assert result["complete"] is False
    assert "final_public_recorder_write_failed" in result["errors"]
    assert (root / "last.safetensors").read_bytes() == model
    public = json.loads((root / "forge_run.json").read_text())
    bound_path = telemetry.private_record_path_for_digest(
        str(root), public["private_record_sha256"]
    )
    assert _sha256_bytes(open(bound_path, "rb").read()) == public[
        "private_record_sha256"
    ]


@pytest.mark.parametrize("mode", ("success", "current_fallback", "prior_fallback"))
def test_cli_terminal_paths_publish_exactly_once(tmp_path, monkeypatch, mode):
    root = tmp_path / "task" / "repo"
    root.mkdir(parents=True)
    spec = ImageSpec.build(
        task_id="task",
        model="krea/Krea-2-Raw",
        model_type="krea2",
        expected_repo_name="repo",
        trigger_word=None,
        dataset_zip=None,
    )
    monkeypatch.setattr(type(spec), "save_root", property(lambda self: str(root)))
    monkeypatch.setattr(type(spec), "output_dir", property(lambda self: str(root)))
    if mode == "prior_fallback":
        prior = _write_st(root / "last.safetensors", "prior")
    else:
        prior = None

    def fake_handler(current_spec, _deadline):
        scope = checkpoints.ensure_run(
            current_spec.save_root, current_spec.expected_repo_name
        )
        if mode == "success":
            _write_st(root / "repo.safetensors", "final")
            assert checkpoints.finalize(str(root), "repo", scope) is not None
            return
        if mode == "current_fallback":
            _write_st(root / "repo_000000010.safetensors", "partial")
        raise RuntimeError("forced handler failure")

    monkeypatch.setattr(dispatch, "for_model_type", lambda _model_type: fake_handler)
    calls = 0
    real_finalize = publication.finalize_public_bundle

    def counted_finalize(path):
        nonlocal calls
        calls += 1
        return real_finalize(path)

    monkeypatch.setattr(publication, "finalize_public_bundle", counted_finalize)

    cli._run(
        spec,
        Deadline.from_hours(
            1.0, started_monotonic=time.monotonic(), export_reserve_s=0.0
        ),
    )

    assert calls == 1
    assert (root / "last.safetensors").is_file()
    if prior is not None:
        assert (root / "last.safetensors").read_bytes() == prior
    assert not (root / "forge_checkpoint_selection.json").exists()
    assert not (root / ".forge_checkpoint_scope.json").exists()
    public = json.loads((root / "forge_run.json").read_text())
    assert set(public) == {"schema", "kind", "private_record_sha256", "events"}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
