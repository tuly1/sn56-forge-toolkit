#!/usr/bin/env python3
"""Capture an allowlisted public tournament/task snapshot, create-only.

Only the exact tournament-details endpoint and its exact task endpoints are
requested.  Full API responses are held in memory long enough to hash and
parse, but are never persisted.  The published snapshot contains tournament
state, safe task metadata, and public participant scoring/submission fields.
Dataset URLs, training rows, test rows, and every other unlisted field are
discarded.  Merely-present training archives are recorded as booleans; this
tool never follows them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Protocol
from urllib.request import HTTPRedirectHandler, Request, build_opener

try:
    from safe_harvest_sync import (
        IntegrityError,
        Publisher,
        SyncError,
        canonical_json,
        contains_forbidden_body_text,
        sha256_bytes,
        utc_iso,
        utc_now,
        utc_token,
    )
except ImportError:  # pragma: no cover
    from .safe_harvest_sync import (
        IntegrityError,
        Publisher,
        SyncError,
        canonical_json,
        contains_forbidden_body_text,
        sha256_bytes,
        utc_iso,
        utc_now,
        utc_token,
    )


SCHEMA = "sn56.week7.safe-public-api-snapshot"
SCHEMA_VERSION = 1
API_ROOT = "https://api.gradients.io"
TOURNAMENT_RE = re.compile(r"tourn_[a-z0-9]+_[0-9]{8}")
TASK_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


class Response(Protocol):
    status: int
    headers: Mapping[str, str]

    def read(self, amount: int = -1) -> bytes: ...
    def close(self) -> None: ...


OpenRequest = Callable[[Request, float], Response]


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        del req, fp, code, msg, headers, newurl
        raise IntegrityError("public API request attempted a redirect")


_NO_REDIRECT_OPENER = build_opener(_RejectRedirects())


def _default_open(request: Request, timeout: float) -> Response:
    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)  # type: ignore[return-value]


def fetch_json(url: str, *, timeout: float, opener: OpenRequest = _default_open) -> tuple[Any, str, int]:
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "sn56-week7-safe-api/1"},
        method="GET",
    )
    response = opener(request, timeout)
    try:
        if response.status != 200:
            raise IntegrityError(f"public API returned HTTP {response.status}")
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise IntegrityError("public API response exceeds the safety ceiling")
    finally:
        response.close()
    try:
        value = json.loads(
            body,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise IntegrityError("public API response is not JSON") from exc
    return value, sha256_bytes(body), len(body)


def _scalar(value: Any, key: str) -> Any:
    item = value.get(key) if isinstance(value, dict) else None
    if isinstance(item, float) and not math.isfinite(item):
        raise IntegrityError(f"public API field {key!r} contains a non-finite number")
    if isinstance(item, str) and contains_forbidden_body_text(item):
        raise IntegrityError(f"public API field {key!r} contains prohibited material")
    return item if item is None or isinstance(item, (str, int, float, bool)) else None


def _score_rows(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise IntegrityError(f"{label} is not a list")
    rows = [_score_row(row) for row in value]
    hotkeys = [row["hotkey"] for row in rows]
    if len(hotkeys) != len(set(hotkeys)):
        raise IntegrityError(f"{label} contains duplicate hotkeys")
    return rows


def _score_row(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise IntegrityError("participant score row is not an object")
    result = {
        key: _scalar(value, key)
        for key in (
            "hotkey",
            "repo",
            "submission_id",
            "test_loss",
            "synth_loss",
            "quality_score",
            "rank",
            "score_reason",
        )
    }
    hotkey = result["hotkey"]
    if not isinstance(hotkey, str) or not hotkey:
        raise IntegrityError("participant score row has no hotkey")
    for field in ("repo", "submission_id", "score_reason"):
        raw_field = value.get(field)
        if raw_field is not None and not isinstance(raw_field, str):
            raise IntegrityError(f"participant score row has invalid {field}")
    for field in ("test_loss", "synth_loss", "quality_score"):
        metric = value.get(field)
        if metric is not None and (
            not isinstance(metric, (int, float))
            or isinstance(metric, bool)
            or not math.isfinite(metric)
        ):
            raise IntegrityError(f"participant score row has invalid {field}")
    if value.get("rank") is not None and (
        type(value.get("rank")) is not int or value["rank"] < 0
    ):
        raise IntegrityError("participant score row has invalid rank")
    submitted = result["repo"] is not None or result["submission_id"] is not None
    scored = any(
        isinstance(result[key], (int, float)) and not isinstance(result[key], bool) and result[key] != 0
        for key in ("test_loss", "synth_loss", "quality_score")
    ) or (isinstance(result["score_reason"], str) and bool(result["score_reason"].strip()))
    result["derived_state"] = "scored" if scored else "submitted" if submitted else "pending"
    return result


def _task_ids(tournament: Mapping[str, Any]) -> list[str]:
    found: set[str] = set()
    rounds = tournament.get("rounds")
    if not isinstance(rounds, list):
        raise IntegrityError("tournament has no rounds list")
    for round_value in rounds:
        if not isinstance(round_value, dict):
            raise IntegrityError("tournament round is not an object")
        tasks = round_value.get("tasks")
        if not isinstance(tasks, list):
            raise IntegrityError("tournament round has no tasks list")
        for task in tasks:
            if not isinstance(task, dict):
                raise IntegrityError("tournament task row is not an object")
            task_id = task.get("task_id")
            if not isinstance(task_id, str) or TASK_RE.fullmatch(task_id) is None:
                raise IntegrityError("tournament task row has no canonical task ID")
            if task_id in found:
                raise IntegrityError("tournament contains a duplicate task ID")
            found.add(task_id)
    if not found:
        raise IntegrityError("tournament contains no canonical task IDs")
    return sorted(found)


def sanitize_tournament(value: Any, tournament_id: str) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(value, dict) or value.get("tournament_id") != tournament_id:
        raise IntegrityError("tournament response identity mismatch")
    if value.get("tournament_type") != "image":
        raise IntegrityError("selected tournament is not an image tournament")
    task_ids = _task_ids(value)
    safe_rounds: list[dict[str, Any]] = []
    for round_value in value.get("rounds", []):
        # `_task_ids` has already established the complete shape.  Never
        # silently filter malformed rows here: the raw response is not kept,
        # so filtering would make missing tournament membership unauditable.
        if not isinstance(round_value, dict):  # defense in depth if validation changes
            raise IntegrityError("tournament round is not an object")
        tasks: list[dict[str, Any]] = []
        for task in round_value.get("tasks", []):
            if not isinstance(task, dict):  # defense in depth if validation changes
                raise IntegrityError("tournament task row is not an object")
            task_type = _scalar(task, "task_type")
            if task_type != "ImageTask":
                raise IntegrityError("image tournament contains a non-image task type")
            scores = task.get("participant_scores")
            tasks.append(
                {
                    "task_id": task["task_id"],
                    "task_type": task_type,
                    "winner": _scalar(task, "winner"),
                    "participant_scores": _score_rows(
                        scores, "tournament task participant_scores"
                    ),
                }
            )
        round_participants = round_value.get("participants")
        if not isinstance(round_participants, list) or not all(
            isinstance(row, str)
            and row
            and not contains_forbidden_body_text(row)
            for row in round_participants
        ):
            raise IntegrityError("tournament round participants are malformed")
        if len(round_participants) != len(set(round_participants)):
            raise IntegrityError("tournament round contains duplicate participants")
        safe_rounds.append(
            {
                "round_id": _scalar(round_value, "round_id"),
                "round_number": _scalar(round_value, "round_number"),
                "round_type": _scalar(round_value, "round_type"),
                "status": _scalar(round_value, "status"),
                "is_final_round": _scalar(round_value, "is_final_round"),
                "participants": list(round_participants),
                "tasks": tasks,
            }
        )
    participants: list[dict[str, Any]] = []
    raw_participants = value.get("participants")
    if not isinstance(raw_participants, list):
        raise IntegrityError("tournament participants are not a list")
    for row in raw_participants:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("hotkey"), str)
            or not row["hotkey"]
            or contains_forbidden_body_text(row["hotkey"])
        ):
            raise IntegrityError("tournament participant row is malformed")
        participants.append(
            {
                "hotkey": row["hotkey"],
                "eliminated_in_round_id": _scalar(row, "eliminated_in_round_id"),
                "final_position": _scalar(row, "final_position"),
            }
        )
    participant_hotkeys = [row["hotkey"] for row in participants]
    if len(participant_hotkeys) != len(set(participant_hotkeys)):
        raise IntegrityError("tournament contains duplicate participants")
    return (
        {
            "tournament_id": tournament_id,
            "tournament_type": "image",
            "status": _scalar(value, "status"),
            "base_winner_hotkey": _scalar(value, "base_winner_hotkey"),
            "winner_hotkey": _scalar(value, "winner_hotkey"),
            "participants": participants,
            "rounds": safe_rounds,
        },
        task_ids,
    )


def sanitize_task(value: Any, expected_task_id: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("task_id") != expected_task_id:
        raise IntegrityError("task response identity mismatch")
    rows = value.get("hotkey_details")
    if not isinstance(rows, list):
        raise IntegrityError("task response has no hotkey_details list")
    if value.get("task_type") != "ImageTask":
        raise IntegrityError("task detail is not an image task")
    return {
        key: _scalar(value, key)
        for key in (
            "task_id",
            "task_type",
            "model_id",
            "model_type",
            "hours_to_complete",
            "created_at",
            "updated_at",
            "termination_at",
            "status",
        )
    } | {
        "public_training_archive_present": value.get("training_data") is not None,
        "participants": _score_rows(rows, "task detail hotkey_details"),
    }


def capture(
    output_root: Path,
    tournament_id: str,
    *,
    observed_at: dt.datetime,
    timeout: float = 30.0,
    opener: OpenRequest = _default_open,
) -> dict[str, Any]:
    if TOURNAMENT_RE.fullmatch(tournament_id) is None:
        raise SyncError("invalid tournament ID")
    tournament_url = f"{API_ROOT}/tournament/{tournament_id}/details"
    tournament_raw, tournament_sha, tournament_bytes = fetch_json(
        tournament_url, timeout=timeout, opener=opener
    )
    tournament, task_ids = sanitize_tournament(tournament_raw, tournament_id)
    round_task_types = {
        task["task_id"]: task["task_type"]
        for round_value in tournament["rounds"]
        for task in round_value["tasks"]
    }
    tasks: list[dict[str, Any]] = []
    sources = [
        {
            "url": tournament_url,
            "response_sha256": tournament_sha,
            "response_bytes": tournament_bytes,
        }
    ]
    for task_id in task_ids:
        url = f"{API_ROOT}/auditing/tasks/{task_id}"
        raw, digest, size = fetch_json(url, timeout=timeout, opener=opener)
        safe_task = sanitize_task(raw, task_id)
        if safe_task["task_type"] != round_task_types[task_id]:
            raise IntegrityError("round/task-detail task type mismatch")
        tasks.append(safe_task)
        sources.append({"url": url, "response_sha256": digest, "response_bytes": size})

    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "observed_at": utc_iso(observed_at),
        "tournament": tournament,
        "tasks": tasks,
        "source_responses": sources,
        "exclusion_contract": {
            "persisted": "explicit tournament/task/submission/score allowlist only",
            "not_persisted": "all dataset URLs, row bodies, and unlisted API fields",
            "followed_urls": "only exact tournament details and exact task endpoints",
        },
    }
    body = canonical_json(result)
    publisher = Publisher(output_root)
    stamp = utc_token(observed_at)
    staged = publisher.stage_bytes(body)
    relative = f"snapshots/{stamp}.json"
    publisher.publish(staged, relative)
    staged.discard()
    checksum = f"{sha256_bytes(body)}  {stamp}.json\n".encode("ascii")
    checksum_stage = publisher.stage_bytes(checksum)
    publisher.publish(checksum_stage, f"snapshots/{stamp}.sha256")
    checksum_stage.discard()
    return {
        "status": "COMPLETE",
        "snapshot": relative,
        "snapshot_sha256": sha256_bytes(body),
        "task_count": len(tasks),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tournament-id", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        result = capture(
            args.output_root,
            args.tournament_id,
            observed_at=utc_now(),
            timeout=args.timeout,
        )
    except (OSError, ValueError, SyncError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
