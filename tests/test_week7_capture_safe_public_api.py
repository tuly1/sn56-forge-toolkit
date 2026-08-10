from __future__ import annotations

import datetime as dt
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SYNC_MODULE = ROOT / "ops" / "experiments" / "week7" / "safe_harvest_sync.py"
sync_spec = importlib.util.spec_from_file_location("safe_harvest_sync", SYNC_MODULE)
sync = importlib.util.module_from_spec(sync_spec)
assert sync_spec.loader is not None
sys.modules[sync_spec.name] = sync
sync_spec.loader.exec_module(sync)

MODULE = ROOT / "ops" / "experiments" / "week7" / "capture_safe_public_api.py"
spec = importlib.util.spec_from_file_location("capture_safe_public_api", MODULE)
capture = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)

TOURNAMENT = "tourn_aaaaaaaaaaaaaaaa_20260810"
TASK = "11111111-1111-4111-8111-111111111111"
NOW = dt.datetime(2026, 8, 10, 15, 15, tzinfo=dt.timezone.utc)


class FakeResponse:
    def __init__(self, value, status=200):
        self.status = status
        self.headers = {"Content-Type": "application/json"}
        self.body = json.dumps(value).encode()
        self.closed = False

    def read(self, amount=-1):
        return self.body if amount < 0 else self.body[:amount]

    def close(self):
        self.closed = True


def fixtures():
    tournament = {
        "tournament_id": TOURNAMENT,
        "tournament_type": "image",
        "status": "active",
        "base_winner_hotkey": "boss",
        "winner_hotkey": None,
        "participants": [
            {"hotkey": "ours", "eliminated_in_round_id": None, "final_position": None}
        ],
        "rounds": [
            {
                "round_id": f"{TOURNAMENT}_round_001",
                "round_number": 1,
                "round_type": "group",
                "status": "active",
                "is_final_round": False,
                "participants": ["ours"],
                "tasks": [
                    {
                        "task_id": TASK,
                        "task_type": "ImageTask",
                        "winner": None,
                        "participant_scores": [],
                        "test_data": "must-not-persist",
                    }
                ],
            }
        ],
        "secret_field": "must-not-persist",
    }
    task = {
        "task_id": TASK,
        "task_type": "ImageTask",
        "model_id": "krea/Krea-2-Raw",
        "model_type": "krea2",
        "hours_to_complete": 0.75,
        "created_at": "2026-08-10T13:00:00Z",
        "updated_at": "2026-08-10T15:00:00Z",
        "termination_at": "2026-08-10T13:45:00Z",
        "status": "training",
        "image_text_pairs": [
            {"image_url": "public-train-a", "text": "caption a"},
            {"image_url": "public-train-b", "text": "caption b"},
        ],
        "training_data": None,
        "test_data": "forbidden-url-must-not-persist",
        "hidden_rows": [{"secret": "forbidden-row-must-not-persist"}],
        "hotkey_details": [
            {
                "hotkey": "ours",
                "repo": "org/repo",
                "submission_id": "sub-1",
                "test_loss": 0.05,
                "synth_loss": 0.05,
                "quality_score": 1.0,
                "rank": 1,
                "score_reason": None,
                "private_note": "must-not-persist",
            }
        ],
    }
    return tournament, task


def test_capture_persists_only_allowlisted_fields_and_source_hashes(tmp_path):
    tournament, task = fixtures()
    requested = []

    def opener(request, timeout):
        requested.append(request.full_url)
        return FakeResponse(tournament if request.full_url.endswith("/details") else task)

    result = capture.capture(
        tmp_path / "evidence",
        TOURNAMENT,
        observed_at=NOW,
        opener=opener,
    )
    body = (tmp_path / "evidence" / result["snapshot"]).read_bytes()
    value = json.loads(body)

    assert requested == [
        f"https://api.gradients.io/tournament/{TOURNAMENT}/details",
        f"https://api.gradients.io/auditing/tasks/{TASK}",
    ]
    assert value["tasks"][0]["public_training_archive_present"] is False
    assert value["tasks"][0]["participants"][0]["derived_state"] == "scored"
    for forbidden in (
        b"must-not-persist",
        b"forbidden-url",
        b"forbidden-row",
        b"public-train-a",
        b"caption a",
        b"test_data",
        b"hidden_rows",
    ):
        assert forbidden not in body
    assert len(value["source_responses"]) == 2


def test_cross_tournament_identity_aborts(tmp_path):
    tournament, _ = fixtures()
    tournament["tournament_id"] = "tourn_bbbbbbbbbbbbbbbb_20260810"
    with pytest.raises(sync.IntegrityError, match="identity mismatch"):
        capture.sanitize_tournament(tournament, TOURNAMENT)


def test_non_image_tournament_aborts():
    tournament, _ = fixtures()
    tournament["tournament_type"] = "text"
    with pytest.raises(sync.IntegrityError, match="not an image"):
        capture.sanitize_tournament(tournament, TOURNAMENT)


def test_valid_plus_malformed_round_task_aborts_instead_of_erasing_membership():
    tournament, _ = fixtures()
    tournament["rounds"][0]["tasks"].append({"task_type": "ImageTask"})
    with pytest.raises(sync.IntegrityError, match="canonical task ID"):
        capture.sanitize_tournament(tournament, TOURNAMENT)


def test_duplicate_round_participants_abort():
    tournament, _ = fixtures()
    tournament["rounds"][0]["participants"] = ["ours", "ours"]
    with pytest.raises(sync.IntegrityError, match="duplicate participants"):
        capture.sanitize_tournament(tournament, TOURNAMENT)


def test_round_and_task_detail_type_conflict_aborts(tmp_path):
    tournament, task = fixtures()
    task["task_type"] = "TextTask"

    def opener(request, timeout):
        return FakeResponse(tournament if request.full_url.endswith("/details") else task)

    with pytest.raises(sync.IntegrityError, match="not an image task"):
        capture.capture(
            tmp_path / "evidence", TOURNAMENT, observed_at=NOW, opener=opener
        )


def test_nonfinite_score_aborts_before_publication(tmp_path):
    tournament, task = fixtures()
    task["hotkey_details"][0]["test_loss"] = math.nan

    def opener(request, timeout):
        return FakeResponse(tournament if request.full_url.endswith("/details") else task)

    with pytest.raises(sync.IntegrityError, match="not JSON"):
        capture.capture(
            tmp_path / "evidence", TOURNAMENT, observed_at=NOW, opener=opener
        )
    assert not (tmp_path / "evidence").exists()


@pytest.mark.parametrize(
    ("field", "bad"),
    [("test_loss", "0.05"), ("synth_loss", []), ("rank", True), ("repo", {})],
)
def test_malformed_score_field_types_abort(field, bad):
    _, task = fixtures()
    task["hotkey_details"][0][field] = bad
    with pytest.raises(sync.IntegrityError, match=f"invalid {field}"):
        capture.sanitize_task(task, TASK)


@pytest.mark.parametrize(
    "payload",
    [
        "test_data=https://secret.invalid/rows.json",
        "evaluation_assets.zip",
        "holdouts/hidden-row.json",
    ],
)
def test_free_text_score_fields_cannot_smuggle_prohibited_material(payload):
    _, task = fixtures()
    task["hotkey_details"][0]["score_reason"] = payload
    with pytest.raises(sync.IntegrityError, match="prohibited material"):
        capture.sanitize_task(task, TASK)


def test_task_identity_mismatch_aborts():
    _, task = fixtures()
    task["task_id"] = "22222222-2222-4222-8222-222222222222"
    with pytest.raises(sync.IntegrityError, match="identity mismatch"):
        capture.sanitize_task(task, TASK)


def test_pending_and_submitted_states_are_distinct():
    pending = {
        "hotkey": "a",
        "repo": None,
        "submission_id": None,
        "test_loss": None,
        "synth_loss": None,
        "quality_score": None,
        "rank": None,
        "score_reason": None,
    }
    submitted = {**pending, "repo": "org/repo"}
    assert capture._score_row(pending)["derived_state"] == "pending"
    assert capture._score_row(submitted)["derived_state"] == "submitted"


def test_zero_placeholders_do_not_count_as_scores():
    row = {
        "hotkey": "a",
        "repo": None,
        "submission_id": None,
        "test_loss": 0.0,
        "synth_loss": 0.0,
        "quality_score": 0.0,
        "rank": 6,
        "score_reason": None,
    }
    assert capture._score_row(row)["derived_state"] == "pending"


def test_existing_snapshot_with_different_bytes_is_not_overwritten(tmp_path):
    tournament, task = fixtures()

    def opener(request, timeout):
        return FakeResponse(tournament if request.full_url.endswith("/details") else task)

    output = tmp_path / "evidence"
    result = capture.capture(output, TOURNAMENT, observed_at=NOW, opener=opener)
    snapshot = output / result["snapshot"]
    snapshot.write_text("corrupt\n")
    before = snapshot.read_bytes()
    with pytest.raises(sync.IntegrityError, match="different bytes"):
        capture.capture(output, TOURNAMENT, observed_at=NOW, opener=opener)
    assert snapshot.read_bytes() == before


def test_default_api_transport_rejects_redirects_before_followup():
    handler = capture._RejectRedirects()
    with pytest.raises(sync.IntegrityError, match="attempted a redirect"):
        handler.redirect_request(
            None, None, 302, "Found", {}, "https://attacker.example/redirected"
        )
