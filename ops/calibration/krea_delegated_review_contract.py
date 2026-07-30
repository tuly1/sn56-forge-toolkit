"""Fail-closed loader for the owner-ratified Stage-1 agent contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import krea_fixture
    from . import krea_provenance
except ImportError:  # pragma: no cover - direct script execution.
    import krea_fixture  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]


CONTRACT_PATH = (
    Path(__file__).resolve().parent
    / "week5"
    / "krea-stage1-delegated-agent-review-contract.json"
)
CONTRACT_FILE_SHA256 = (
    "bc09239ad9f407fd1da9daf0a09838c93b765fb2fd869d3df433ad34adaf5d45"
)
CONTRACT_SHA256 = "519df7266ca8e22feb93373f5ddbba8be0455ec41a1da83a0f142a87ef57eebf"


def load() -> dict[str, Any]:
    path = CONTRACT_PATH
    if path.is_symlink() or not path.is_file():
        raise ValueError("Stage-1 delegated-review contract is missing or a symlink")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Stage-1 delegated-review contract is not JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Stage-1 delegated-review contract must be an object")
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    if (
        raw != krea_provenance.canonical_bytes(value) + b"\n"
        or hashlib.sha256(raw).hexdigest() != CONTRACT_FILE_SHA256
        or value.get("schema") != 1
        or value.get("kind")
        != "forge-krea-owner-ratified-stage1-delegated-agent-review-contract"
        or value.get("contract_sha256") != CONTRACT_SHA256
        or krea_provenance.canonical_sha256(body) != CONTRACT_SHA256
        or value.get("accountable_owner_identity") != "Atulya Shetty"
    ):
        raise ValueError("Stage-1 delegated-review contract drifted")
    actors = value.get("actors")
    outputs = value.get("allowed_outputs")
    if not isinstance(actors, dict) or not isinstance(outputs, dict):
        raise ValueError("Stage-1 delegated-review actors/outputs are invalid")
    if set(actors) != set(outputs):
        raise ValueError("Stage-1 delegated-review actor/output roles differ")
    normalized = {
        name: krea_fixture._agent_actor(actor, f"delegated actor {name}")
        for name, actor in actors.items()
    }
    if len({actor["actor_id"] for actor in normalized.values()}) != len(normalized):
        raise ValueError("delegated actor ids are not pairwise distinct")
    if len({actor["review_instance_id"] for actor in normalized.values()}) != len(
        normalized
    ):
        raise ValueError("delegated review instances are not pairwise distinct")
    for name, actor in normalized.items():
        output = outputs[name]
        if not isinstance(output, dict) or output.get("role") != actor["role"]:
            raise ValueError("delegated actor role differs from allowed output")
    return value


def binding() -> dict[str, Any]:
    contract = load()
    return {
        "path": "ops/calibration/week5/"
        "krea-stage1-delegated-agent-review-contract.json",
        "file_sha256": CONTRACT_FILE_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "contract": contract,
    }


def actor(name: str) -> dict[str, Any]:
    contract = load()
    actors = contract["actors"]
    if name not in actors:
        raise ValueError(f"unknown delegated actor: {name}")
    return dict(actors[name])


def validate_actor(name: str, value: Any) -> dict[str, Any]:
    observed = krea_fixture._agent_actor(value, f"delegated actor {name}")
    expected = actor(name)
    if observed != expected:
        raise ValueError(f"{name} differs from owner-ratified delegated actor")
    return observed


def validate_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or dict(value) != binding():
        raise ValueError("delegated-review contract binding drifted")
    return dict(value)


def reject_delegated_actor_reuse(value: Any, *, label: str) -> dict[str, Any]:
    observed = krea_fixture._agent_actor(value, label)
    contract = load()
    actors = contract["actors"].values()
    if observed["actor_id"] in {actor["actor_id"] for actor in actors} or observed[
        "review_instance_id"
    ] in {actor["review_instance_id"] for actor in actors}:
        raise ValueError(f"{label} reuses a delegated actor identity/review instance")
    return observed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True, choices=sorted(load()["actors"]))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = Path(os.path.abspath(os.path.expanduser(args.output)))
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = krea_provenance.canonical_bytes(actor(args.actor)) + b"\n"
    with output.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(hashlib.sha256(payload).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
