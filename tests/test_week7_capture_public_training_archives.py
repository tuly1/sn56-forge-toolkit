from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import sys
import warnings
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SYNC_MODULE = ROOT / "ops" / "experiments" / "week7" / "safe_harvest_sync.py"
sync_spec = importlib.util.spec_from_file_location("safe_harvest_sync", SYNC_MODULE)
sync = importlib.util.module_from_spec(sync_spec)
assert sync_spec.loader is not None
sys.modules[sync_spec.name] = sync
sync_spec.loader.exec_module(sync)

MODULE = ROOT / "ops" / "experiments" / "week7" / "capture_public_training_archives.py"
spec = importlib.util.spec_from_file_location("capture_public_training_archives", MODULE)
capture = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)

TOURNAMENT = "tourn_aaaaaaaaaaaaaaaa_20260810"
TASK = "11111111-1111-4111-8111-111111111111"
NOW = dt.datetime(2026, 8, 10, 18, 30, tzinfo=dt.timezone.utc)
TASK_ENDPOINT = f"https://api.gradients.io/auditing/tasks/{TASK}"
TRAIN_URL = (
    "https://s3.eu-central-003.backblazeb2.com/training.zip?X-Amz-Signature=secret"
)
TEST_URL = "https://forbidden.example/test.zip"
POOL_URLS = [f"https://forbidden.example/full-pool/{index}.png" for index in range(10)]


def sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def snapshot_value(*, schema: str = capture.SAFE_API_SCHEMA, round_status: str = "completed"):
    return {
        "schema": schema,
        "schema_version": capture.SAFE_API_SCHEMA_VERSION,
        "observed_at": "2026-08-10T18:00:00Z",
        "tournament": {
            "tournament_id": TOURNAMENT,
            "tournament_type": "image",
            "status": "active",
            "rounds": [
                {
                    "round_id": f"{TOURNAMENT}_round_001",
                    "round_number": 1,
                    "status": round_status,
                    "tasks": [{"task_id": TASK, "task_type": "ImageTask"}],
                }
            ],
        },
        "tasks": [
            {
                "task_id": TASK,
                "task_type": "ImageTask",
                "status": "success",
                "public_training_archive_present": True,
            }
        ],
        "source_responses": [],
    }


def write_snapshot(tmp_path: Path, value=None) -> Path:
    path = tmp_path / "20260810T180000.000000Z.json"
    body = sync.canonical_json(value if value is not None else snapshot_value())
    path.write_bytes(body)
    path.with_suffix(".sha256").write_text(f"{sha(body)}  {path.name}\n", encoding="ascii")
    return path


def test_snapshot_schema_bool_duplicate_detail_and_type_mismatch_fail_closed(tmp_path):
    value = snapshot_value()
    value["schema_version"] = True
    path = write_snapshot(tmp_path, value)
    with pytest.raises(sync.IntegrityError, match="schema/version"):
        capture._load_checked_snapshot(path, TOURNAMENT)

    duplicate_root = tmp_path / "duplicate"
    duplicate_root.mkdir()
    value = snapshot_value()
    value["tasks"].append(dict(value["tasks"][0]))
    path = write_snapshot(duplicate_root, value)
    with pytest.raises(sync.IntegrityError, match="duplicate task details"):
        capture._load_checked_snapshot(path, TOURNAMENT)

    mismatch_root = tmp_path / "type-mismatch"
    mismatch_root.mkdir()
    value = snapshot_value()
    value["tasks"][0]["task_type"] = "TextTask"
    path = write_snapshot(mismatch_root, value)
    with pytest.raises(sync.IntegrityError, match="not an image task"):
        capture._load_checked_snapshot(path, TOURNAMENT)


def training_zip(*, images: int = 9, duplicate_last: bool = False) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(images):
            image_index = 0 if duplicate_last and index == images - 1 else index
            archive.writestr(f"dataset/{index:03d}.png", f"image-{image_index}".encode())
            archive.writestr(f"dataset/{index:03d}.txt", f"caption {index}")
    return output.getvalue()


def task_response(*, training_data=TRAIN_URL, status="success"):
    return {
        "task_id": TASK,
        "status": status,
        "training_data": training_data,
        "image_text_pairs": [
            {"image_url": url, "text": f"full pool caption {index}"}
            for index, url in enumerate(POOL_URLS)
        ],
        "test_data": TEST_URL,
        "hidden_rows": [{"url": "https://forbidden.example/hidden.png"}],
        "quarantine": "https://forbidden.example/quarantine.zip",
    }


class FakeResponse:
    def __init__(self, body: bytes, *, status: int = 200, headers=None):
        self.status = status
        self.headers = headers or {}
        self._body = body
        self._offset = 0
        self.closed = False

    def read(self, amount=-1):
        if amount < 0:
            amount = len(self._body) - self._offset
        start = self._offset
        self._offset = min(len(self._body), self._offset + amount)
        return self._body[start : self._offset]

    def close(self):
        self.closed = True


def make_opener(task, archive: bytes | None, requested: list[str]):
    task_body = json.dumps(task).encode()

    def opener(request, timeout):
        requested.append(request.full_url)
        if request.full_url == TASK_ENDPOINT:
            return FakeResponse(task_body, headers={"Content-Type": "application/json"})
        if archive is not None and request.full_url == TRAIN_URL:
            return FakeResponse(
                archive,
                headers={
                    "Content-Type": "application/zip",
                    "Content-Length": str(len(archive)),
                },
            )
        raise AssertionError(f"collector requested a prohibited/unexpected URL: {request.full_url}")

    return opener


def read_inventory(output: Path, result: dict) -> dict:
    return json.loads((output / result["inventory"]).read_text())


def read_receipt(output: Path, inventory: dict) -> tuple[dict, bytes]:
    path = output / inventory["tasks"][0]["receipt"]
    body = path.read_bytes()
    return json.loads(body), body


def test_training_zip_not_full_pool_is_opened_and_nine_means_nine(tmp_path):
    snapshot = write_snapshot(tmp_path)
    requested: list[str] = []
    archive = training_zip(images=9)
    output = tmp_path / "archive-evidence"

    result = capture.capture(
        snapshot,
        output,
        TOURNAMENT,
        observed_at=NOW,
        opener=make_opener(task_response(), archive, requested),
    )

    assert result["status"] == "COMPLETE"
    assert requested == [TASK_ENDPOINT, TRAIN_URL]
    assert TEST_URL not in requested
    assert not set(POOL_URLS) & set(requested)
    inventory = read_inventory(output, result)
    receipt, receipt_body = read_receipt(output, inventory)
    assert receipt["inventory"]["image_member_count"] == 9
    assert receipt["inventory"]["caption_member_count"] == 9
    assert receipt["inventory"]["paired_stem_count"] == 9
    assert receipt["inventory"]["post_exact_byte_dedup_image_count"] == 9
    assert len(receipt["inventory"]["members"]) == 18
    assert all(row["sha256"] for row in receipt["inventory"]["members"])
    assert receipt["source_training_archive"]["url"] == (
        "https://s3.eu-central-003.backblazeb2.com/training.zip"
    )
    assert b"password" not in receipt_body
    assert b"X-Amz-Signature" not in receipt_body
    assert b"secret" not in receipt_body
    assert TEST_URL.encode() not in receipt_body
    assert not any(url.encode() in receipt_body for url in POOL_URLS)
    archive_identity = receipt["archive"]
    archive_path = output / archive_identity["object"]
    assert archive_path.read_bytes() == archive
    assert sha(archive) == archive_identity["sha256"]
    assert receipt["rights"] == {
        "public_access": "observed",
        "license_and_third_party_rights": "unverified",
        "allowed_use": "research-analysis-only",
        "fixture_admission": "not admitted",
    }


def test_exact_byte_image_dedup_is_reported_without_changing_raw_count(tmp_path):
    snapshot = write_snapshot(tmp_path)
    requested: list[str] = []
    archive = training_zip(images=9, duplicate_last=True)
    output = tmp_path / "evidence"
    result = capture.capture(
        snapshot,
        output,
        TOURNAMENT,
        observed_at=NOW,
        opener=make_opener(task_response(), archive, requested),
    )
    receipt, _ = read_receipt(output, read_inventory(output, result))
    assert receipt["inventory"]["image_member_count"] == 9
    assert receipt["inventory"]["post_exact_byte_dedup_image_count"] == 8


def test_selection_function_never_accesses_other_dataset_fields():
    class Guarded(dict):
        def get(self, key, default=None):
            if key in {"test_data", "image_text_pairs", "hidden_rows", "quarantine"}:
                raise AssertionError(f"prohibited key accessed: {key}")
            return super().get(key, default)

    value = Guarded(task_response())
    assert capture._select_training_url(value, TASK) == TRAIN_URL


def test_missing_training_archive_is_explicit_partial(tmp_path):
    snapshot = write_snapshot(tmp_path)
    requested: list[str] = []
    output = tmp_path / "evidence"
    result = capture.capture(
        snapshot,
        output,
        TOURNAMENT,
        observed_at=NOW,
        opener=make_opener(task_response(training_data=None), None, requested),
    )
    assert result["status"] == "PARTIAL"
    assert requested == [TASK_ENDPOINT]
    inventory = read_inventory(output, result)
    assert inventory["partial_task_count"] == 1
    receipt, _ = read_receipt(output, inventory)
    assert receipt["status"] == "PARTIAL"
    assert receipt["archive"] is None
    assert receipt["inventory"] is None
    assert receipt["reason"] == "public training_data URL absent"


def malicious_zip(kind: str) -> bytes:
    output = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(output, "w") as archive:
            if kind == "traversal":
                archive.writestr("../escape.png", b"image")
            elif kind == "symlink":
                info = zipfile.ZipInfo("dataset/link.png")
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, b"target")
            elif kind == "duplicate":
                archive.writestr("dataset/000.png", b"first")
                archive.writestr("dataset/000.png", b"second")
            elif kind == "prohibited":
                archive.writestr("test/000.png", b"image")
            elif kind == "hold-out":
                archive.writestr("hold_out/000.png", b"image")
            elif kind == "eval-data":
                archive.writestr("evaluation_data/000.png", b"image")
            elif kind == "testing":
                archive.writestr("testing/000.png", b"image")
            else:  # pragma: no cover - test helper guard
                raise AssertionError(kind)
    return output.getvalue()


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("traversal", "unsafe member path"),
        ("symlink", "symlink"),
        ("duplicate", "duplicate/case-colliding"),
        ("prohibited", "unsafe or prohibited"),
        ("hold-out", "unsafe or prohibited"),
        ("eval-data", "unsafe or prohibited"),
        ("testing", "unsafe or prohibited"),
    ],
)
def test_unsafe_archives_fail_closed_before_publication(tmp_path, kind, message):
    snapshot = write_snapshot(tmp_path)
    requested: list[str] = []
    output = tmp_path / f"evidence-{kind}"
    archive = malicious_zip(kind)
    with pytest.raises(sync.IntegrityError, match=message):
        capture.capture(
            snapshot,
            output,
            TOURNAMENT,
            observed_at=NOW,
            opener=make_opener(task_response(), archive, requested),
        )
    assert requested == [TASK_ENDPOINT, TRAIN_URL]
    assert not list(output.glob("inventories/*.json"))
    assert not list(output.glob("objects/sha256/*/*"))


def test_nonterminal_task_aborts_without_following_training_or_test_url(tmp_path):
    snapshot = write_snapshot(tmp_path)
    requested: list[str] = []
    with pytest.raises(sync.IntegrityError, match="not terminal"):
        capture.capture(
            snapshot,
            tmp_path / "evidence",
            TOURNAMENT,
            observed_at=NOW,
            opener=make_opener(task_response(status="training"), None, requested),
        )
    assert requested == [TASK_ENDPOINT]


def test_create_only_receipt_cannot_be_replaced(tmp_path):
    snapshot = write_snapshot(tmp_path)
    archive = training_zip(images=2)
    output = tmp_path / "evidence"
    first = capture.capture(
        snapshot,
        output,
        TOURNAMENT,
        observed_at=NOW,
        opener=make_opener(task_response(), archive, []),
    )
    inventory = read_inventory(output, first)
    receipt_path = output / inventory["tasks"][0]["receipt"]
    receipt_path.write_bytes(b"corrupt\n")
    before = receipt_path.read_bytes()

    with pytest.raises(sync.IntegrityError, match="different bytes"):
        capture.capture(
            snapshot,
            output,
            TOURNAMENT,
            observed_at=NOW,
            opener=make_opener(task_response(), archive, []),
        )
    assert receipt_path.read_bytes() == before


def test_checksum_mismatch_fails_before_network(tmp_path):
    snapshot = write_snapshot(tmp_path)
    snapshot.with_suffix(".sha256").write_text(
        f"{'0' * 64}  {snapshot.name}\n", encoding="ascii"
    )
    requested: list[str] = []
    with pytest.raises(sync.IntegrityError, match="checksum mismatch"):
        capture.capture(
            snapshot,
            tmp_path / "evidence",
            TOURNAMENT,
            observed_at=NOW,
            opener=make_opener(task_response(), training_zip(), requested),
        )
    assert requested == []


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (snapshot_value(schema="wrong.schema"), "schema/version"),
        (snapshot_value(round_status="active"), "not uniquely completed"),
    ],
)
def test_schema_and_terminal_round_gates_fail_before_network(tmp_path, value, message):
    snapshot = write_snapshot(tmp_path, value)
    requested: list[str] = []
    with pytest.raises(sync.IntegrityError, match=message):
        capture.capture(
            snapshot,
            tmp_path / "evidence",
            TOURNAMENT,
            observed_at=NOW,
            opener=make_opener(task_response(), training_zip(), requested),
        )
    assert requested == []


def test_inventory_sidecar_binds_exact_inventory(tmp_path):
    snapshot = write_snapshot(tmp_path)
    archive = training_zip(images=1)
    output = tmp_path / "evidence"
    result = capture.capture(
        snapshot,
        output,
        TOURNAMENT,
        observed_at=NOW,
        opener=make_opener(task_response(), archive, []),
    )
    inventory_body = (output / result["inventory"]).read_bytes()
    sidecar = (output / result["checksum"]).read_text(encoding="ascii")
    assert sidecar == f"{sha(inventory_body)}  {Path(result['inventory']).name}\n"
    assert result["inventory_sha256"] == sha(inventory_body)


@pytest.mark.parametrize(
    "url",
    [
        "https://s3.eu-central-003.backblazeb2.com/hidden.zip",
        "https://s3.eu-central-003.backblazeb2.com/test_data.zip",
        "https://s3.eu-central-003.backblazeb2.com/hold-out/archive.zip",
        "https://s3.eu-central-003.backblazeb2.com/evaluation_data/archive.zip",
        "https://s3.eu-central-003.backblazeb2.com/quarantine/archive.zip",
        "https://s3.eu-central-003.backblazeb2.com/%74est_data.zip",
        "https://s3.eu-central-003.backblazeb2.com/te%252573t_data.zip",
    ],
)
def test_prohibited_training_source_path_fails_before_archive_request(tmp_path, url):
    snapshot = write_snapshot(tmp_path)
    requested: list[str] = []
    task = task_response(training_data=url)

    with pytest.raises(sync.IntegrityError, match="prohibited dataset surface"):
        capture.capture(
            snapshot,
            tmp_path / "evidence",
            TOURNAMENT,
            observed_at=NOW,
            opener=make_opener(task, None, requested),
        )

    assert requested == [TASK_ENDPOINT]


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/private.zip",
        "https://169.254.169.254/latest/meta-data.zip",
        "https://[::1]/private.zip",
        "https://user:pass@s3.eu-central-003.backblazeb2.com/training.zip",
        "https://s3.eu-central-003.backblazeb2.com:444/training.zip",
        "https://attacker.example/training.zip",
    ],
)
def test_training_source_authority_is_fail_closed_before_archive_request(tmp_path, url):
    snapshot = write_snapshot(tmp_path)
    requested: list[str] = []

    with pytest.raises(sync.IntegrityError):
        capture.capture(
            snapshot,
            tmp_path / "evidence",
            TOURNAMENT,
            observed_at=NOW,
            opener=make_opener(task_response(training_data=url), None, requested),
        )

    assert requested == [TASK_ENDPOINT]


def test_redirect_handler_rejects_before_followup_request():
    handler = capture._RejectRedirects()
    with pytest.raises(sync.IntegrityError, match="attempted a redirect"):
        handler.redirect_request(None, None, 302, "Found", {}, "https://bad.example/test.zip")
