"""Fail-closed loader for the owner-ratified Stage-2 agent contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

try:
    from . import krea_fixture
    from . import krea_provenance
except ImportError:  # pragma: no cover - direct script execution.
    import krea_fixture  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]


CONTRACT_PATH = (
    Path(__file__).resolve().parent
    / "week5"
    / "krea-stage2-delegated-agent-review-contract.json"
)
CONTRACT_FILE_SHA256 = (
    "7d9503c60b0646b4adf9080c1ef154c919c1ca0acf88c1697a785fa2f22dbc66"
)
CONTRACT_SHA256 = "db4f4eed14ec51a1768309759ab62bf64ee98529d90a0bdb8c48f757b97f1949"


def load() -> dict[str, Any]:
    path = CONTRACT_PATH
    if path.is_symlink() or not path.is_file():
        raise ValueError("Stage-2 delegated-review contract is missing or a symlink")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Stage-2 delegated-review contract is not JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Stage-2 delegated-review contract must be an object")
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    if (
        raw != krea_provenance.canonical_bytes(value) + b"\n"
        or hashlib.sha256(raw).hexdigest() != CONTRACT_FILE_SHA256
        or value.get("schema") != 1
        or value.get("kind")
        != "forge-krea-owner-ratified-stage2-delegated-agent-review-contract"
        or value.get("contract_sha256") != CONTRACT_SHA256
        or krea_provenance.canonical_sha256(body) != CONTRACT_SHA256
        or value.get("accountable_owner_identity") != "Atulya Shetty"
    ):
        raise ValueError("Stage-2 delegated-review contract drifted")
    actors = value.get("actors")
    outputs = value.get("allowed_outputs")
    if not isinstance(actors, dict) or not isinstance(outputs, dict):
        raise ValueError("Stage-2 delegated-review actors/outputs are invalid")
    if set(actors) != set(outputs):
        raise ValueError("Stage-2 delegated-review actor/output roles differ")
    normalized = {
        name: krea_fixture._agent_actor(actor_value, f"delegated actor {name}")
        for name, actor_value in actors.items()
    }
    if len({item["actor_id"] for item in normalized.values()}) != len(normalized):
        raise ValueError("Stage-2 delegated actor ids are not pairwise distinct")
    if len({item["review_instance_id"] for item in normalized.values()}) != len(
        normalized
    ):
        raise ValueError("Stage-2 delegated review instances are not distinct")
    for name, item in normalized.items():
        output = outputs[name]
        if not isinstance(output, dict) or output.get("role") != item["role"]:
            raise ValueError("Stage-2 actor role differs from its allowed output")
    return value


def binding() -> dict[str, Any]:
    contract = load()
    return {
        "path": "ops/calibration/week5/"
        "krea-stage2-delegated-agent-review-contract.json",
        "file_sha256": CONTRACT_FILE_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "contract": contract,
    }


def actor(name: str) -> dict[str, Any]:
    actors = load()["actors"]
    if name not in actors:
        raise ValueError(f"unknown Stage-2 delegated actor: {name}")
    return dict(actors[name])


def validate_actor(name: str, value: Any) -> dict[str, Any]:
    observed = krea_fixture._agent_actor(value, f"delegated actor {name}")
    if observed != actor(name):
        raise ValueError(f"{name} differs from owner-ratified delegated actor")
    return observed


def validate_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or dict(value) != binding():
        raise ValueError("Stage-2 delegated-review contract binding drifted")
    return dict(value)


def publish_actor(name: str, output: str | Path) -> dict[str, Any]:
    """Publish one canonical actor record without replacing existing evidence."""

    value = actor(name)
    path = Path(os.path.abspath(os.path.expanduser(output)))
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(
                f"delegated actor output has a symlink component: {current}"
            )
        current = current.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(
                f"delegated actor output has a symlink component: {current}"
            )
        current = current.parent
    payload = krea_provenance.canonical_bytes(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": str(path),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "actor": value,
    }
