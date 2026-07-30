#!/usr/bin/env python3
"""Assemble and run fc70 Week-5 Krea cell plans without source mutation.

This tool is staged under the durable campaign controls, never under the Forge
source mount.  It imports the frozen fc70 modules, derives every dynamic budget
and schedule from their sealed controls, validates/seals every plan and approval
with those modules, and emits a hash-bound sequential queue.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


# Runtime source is the exact fc70 tree plus the native, hash-proven bridge that
# lets it consume the immutable admission sealed by its 588 ancestor.  Later
# external controller commits must not broaden or move this runtime pin.
FC70_COMMIT = "f6ce1ad044ff2aa920f2c63074dedd9c32035922"
_KIND = "forge-krea-fc70-cell-assembly-spec"
_QUEUE_KIND = "forge-krea-fc70-sequential-cell-queue"
_CELLS = tuple(f"{fixture}-K{arm}" for fixture in ("D1", "D2") for arm in range(6))
_INITIAL_CELLS = ("D1-K1",) + tuple(
    cell for cell in _CELLS if cell not in {"D1-K1", "D2-K4"}
)
_AXES = {
    "K0": [],
    "K1": ["planned_steps", "save_cadence"],
    "K2": ["planned_steps", "save_cadence"],
    "K3": ["dropout", "ema", "planned_steps", "save_cadence"],
    "K4": ["planned_steps", "save_cadence"],
    "K5": ["learning_rate"],
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TRACKED_DISCOVERY_PLAN = Path(
    "ops/calibration/week5/krea-discovery-plan.json"
)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _semantic_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_file(value: str | Path, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(value)))
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    return path


def _safe_directory(value: str | Path, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(value)))
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a directory: {path}")
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    return path


def _load(path: str | Path, label: str) -> tuple[Path, dict[str, Any], str]:
    path = _safe_file(path, label)
    raw = path.read_bytes()
    try:
        value = _object(json.loads(raw), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if raw != _canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline")
    return path, value, hashlib.sha256(raw).hexdigest()


def _load_tracked_discovery_plan(
    path: str | Path,
    *,
    forge_root: Path,
    expected_file_sha256: str,
) -> tuple[Path, dict[str, Any], str]:
    """Load the one tracked source JSON whose repository bytes are pretty-printed."""

    path = _safe_file(path, "tracked discovery plan")
    expected_path = _safe_file(
        forge_root / _TRACKED_DISCOVERY_PLAN,
        "tracked discovery plan",
    )
    if path != expected_path:
        raise ValueError("discovery plan is not the tracked Forge plan")
    raw = path.read_bytes()
    try:
        value = _object(json.loads(raw), "tracked discovery plan")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("tracked discovery plan is not JSON") from exc
    file_sha = hashlib.sha256(raw).hexdigest()
    if file_sha != _sha(
        expected_file_sha256, "indexed discovery-plan file SHA-256"
    ):
        raise ValueError("tracked discovery plan differs from the profile index")
    return path, value, file_sha


def _publish(path: Path, value: dict[str, Any]) -> None:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"output has a symlink ancestor: {current}")
        current = current.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _modules(forge_root: Path) -> dict[str, Any]:
    forge_root = _safe_directory(forge_root, "fc70 Forge root")
    observed = subprocess.run(
        ["git", "-C", str(forge_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    if observed != FC70_COMMIT:
        raise ValueError(f"Forge root is not frozen fc70: {observed}")
    if str(forge_root) not in sys.path:
        sys.path.insert(0, str(forge_root))
    from ops.calibration import krea_accelerated_discovery
    from ops.calibration import krea_budget
    from ops.calibration import krea_execution_plan
    from ops.calibration import krea_provenance
    from ops.calibration import krea_runtime_binding

    return {
        "accelerated": krea_accelerated_discovery,
        "budget": krea_budget,
        "execution": krea_execution_plan,
        "provenance": krea_provenance,
        "runtime": krea_runtime_binding,
    }


def _validate_spec(value: dict[str, Any]) -> dict[str, Any]:
    _exact(
        value,
        {
            "schema",
            "kind",
            "task_id_prefix",
            "expected_repo_prefix",
            "timing_evidence",
            "base_model",
            "fixtures",
            "arms",
            "spec_sha256",
        },
        "cell assembly spec",
    )
    body = {key: item for key, item in value.items() if key != "spec_sha256"}
    if (
        value["schema"] != 1
        or value["kind"] != _KIND
        or value["spec_sha256"] != _semantic_sha(body)
    ):
        raise ValueError("cell assembly spec identity is invalid")
    for key in ("task_id_prefix", "expected_repo_prefix"):
        if not isinstance(value[key], str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value[key]
        ):
            raise ValueError(f"assembly spec {key} is invalid")
    fixtures = _object(value["fixtures"], "assembly fixtures")
    arms = _object(value["arms"], "assembly arms")
    if set(fixtures) != {"D1", "D2"} or set(arms) != {f"K{i}" for i in range(6)}:
        raise ValueError("assembly spec must bind exactly D1/D2 and K0..K5")
    for fixture_id, fixture in fixtures.items():
        _exact(
            _object(fixture, f"assembly fixture {fixture_id}"),
            {"training_archive", "evaluation_dataset"},
            f"assembly fixture {fixture_id}",
        )
    for arm_id, arm in arms.items():
        _exact(
            _object(arm, f"assembly arm {arm_id}"),
            {"arm_basis", "execution_recipe"},
            f"assembly arm {arm_id}",
        )
    return value


def seal_spec(payload: dict[str, Any]) -> dict[str, Any]:
    if "spec_sha256" in payload:
        raise ValueError("unsealed assembly-spec payload contains spec_sha256")
    spec = {**payload, "spec_sha256": _semantic_sha(payload)}
    return _validate_spec(spec)


def _binding(path: Path) -> dict[str, str]:
    path = _safe_file(path, "bound control")
    return {"path": str(path), "sha256": _file_sha(path)}


def _set_recipe_value(recipe: dict[str, Any], name: str, value: int) -> None:
    fields = _object(recipe.get("fields"), "execution recipe fields")
    row = _object(fields.get(name), f"execution recipe {name}")
    if "effective_value" not in row:
        raise ValueError(f"execution recipe {name} lacks effective_value")
    row["effective_value"] = value
    classification = row.get("classification")
    if classification in {"known", "adapted"}:
        row["classification"] = (
            "known" if row.get("source_value") == value else "adapted"
        )


def _cell_payload(
    *,
    cell_id: str,
    spec: dict[str, Any],
    controls: dict[str, Any],
    profile_index: dict[str, Any],
    profile_index_path: Path,
    profile_index_file_sha: str,
    throughput_profile: dict[str, Any],
    throughput_profile_path: Path,
    throughput_profile_file_sha: str,
    host_manifest_path: Path,
    host_manifest_file_sha: str,
    forge_root: Path,
    modules: dict[str, Any],
) -> dict[str, Any]:
    fixture_id, arm_id = cell_id.split("-", 1)
    cell = controls["cell"]
    if cell["cell_id"] != cell_id:
        raise ValueError("derived controls bind another cell")
    discovery_binding = _object(
        profile_index["discovery_plan"], "indexed discovery plan"
    )
    _exact(
        discovery_binding,
        {"path", "file_sha256"},
        "indexed discovery plan",
    )
    discovery_path, discovery, discovery_file_sha = _load_tracked_discovery_plan(
        discovery_binding["path"],
        forge_root=forge_root,
        expected_file_sha256=discovery_binding["file_sha256"],
    )
    fixture_slot = profile_index["fixtures"][fixture_id]
    fixture_spec = spec["fixtures"][fixture_id]
    recipe = copy.deepcopy(spec["arms"][arm_id]["execution_recipe"])
    _set_recipe_value(recipe, "planned_steps", controls["recipe_overrides"]["planned_steps"])
    _set_recipe_value(recipe, "save_cadence", controls["recipe_overrides"]["save_cadence"])
    profile = modules["budget"].load_throughput_profile(throughput_profile)
    runner_path = forge_root / "ops/calibration/run_krea_ladder.py"
    seed = discovery["training_seed_a"]
    return {
        "schema": 3,
        "kind": "forge-krea-pretraining-execution-plan",
        "arm_id": arm_id,
        "task_id": f"{spec['task_id_prefix']}-{cell_id}",
        "expected_repo_name": f"{spec['expected_repo_prefix']}-{cell_id}",
        "discovery_plan": {"path": str(discovery_path), "sha256": discovery_file_sha},
        "discovery_fixture_id": fixture_id,
        "seed_role": "A",
        "fixture_manifest": {
            "path": fixture_slot["manifest"]["path"],
            "sha256": fixture_slot["manifest"]["file_sha256"],
        },
        "fixture_approval": {
            "path": fixture_slot["approval"]["path"],
            "sha256": fixture_slot["approval"]["file_sha256"],
        },
        "training_archive": copy.deepcopy(fixture_spec["training_archive"]),
        "evaluation_dataset": copy.deepcopy(fixture_spec["evaluation_dataset"]),
        "arm_basis": copy.deepcopy(spec["arms"][arm_id]["arm_basis"]),
        "execution_recipe": recipe,
        "throughput_profile": {
            "path": str(throughput_profile_path),
            "sha256": throughput_profile_file_sha,
        },
        "timing_evidence": copy.deepcopy(spec["timing_evidence"]),
        "host_execution_manifest": {
            "path": str(host_manifest_path),
            "sha256": host_manifest_file_sha,
        },
        "budget_plan": copy.deepcopy(controls["budget_plan"]),
        "budget_plan_sha256": controls["budget_plan_sha256"],
        "schedule": copy.deepcopy(controls["schedule"]),
        "base_model": copy.deepcopy(spec["base_model"]),
        "seed": seed,
        "runtime_identity_sha256": profile.runtime_identity_sha256,
        "execution_envelope_sha256": profile.execution_envelope.execution_envelope_sha256,
        "throughput_equivalence_class": cell["throughput_equivalence_class"],
        "predeclared_recipe_axes": list(_AXES[arm_id]),
        "in_task_proxy_selection": {"enabled": False, "reserve_s": 0},
        "runner_sha256": _file_sha(runner_path),
        "gpu_execution_authorized": False,
        "discovery_profile_index": {
            "path": str(profile_index_path),
            "file_sha256": profile_index_file_sha,
            "index_sha256": profile_index["index_sha256"],
        },
        "discovery_execution_authorization": copy.deepcopy(
            profile_index["discovery_execution_authorization"]
        ),
    }


def assemble(
    *,
    forge_root: Path,
    spec_path: Path,
    campaign_path: Path,
    profile_index_path: Path,
    throughput_profile_path: Path,
    host_manifest_path: Path,
    admission_envelope_path: Path,
    technical_actor_path: Path,
    output_dir: Path,
    campaign_root: Path,
    approved_at_utc: str,
    cells: Sequence[str],
    queue_output: Path,
) -> dict[str, Any]:
    modules = _modules(forge_root)
    spec_path, spec, spec_file_sha = _load(spec_path, "cell assembly spec")
    _validate_spec(spec)
    campaign_path, _campaign, _campaign_file_sha = _load(
        campaign_path, "accelerated campaign"
    )
    profile_index_path, profile_index, profile_index_file_sha = _load(
        profile_index_path, "accelerated profile index"
    )
    modules["runtime"].validate_profile_index(profile_index)
    throughput_profile_path, throughput_profile, throughput_profile_file_sha = _load(
        throughput_profile_path, "D1/A throughput profile"
    )
    host_manifest_path, _host_manifest, host_manifest_file_sha = _load(
        host_manifest_path, "fresh host manifest"
    )
    admission_envelope_path = _safe_file(
        admission_envelope_path, "fixture admission envelope"
    )
    _, technical_actor, _ = _load(technical_actor_path, "execution reviewer actor")
    output_dir = Path(os.path.abspath(os.path.expanduser(output_dir)))
    campaign_root = Path(os.path.abspath(os.path.expanduser(campaign_root)))
    if not output_dir.is_dir() or not campaign_root.is_dir():
        raise ValueError("output and campaign roots must already exist")
    if len(cells) != len(set(cells)) or any(cell not in _CELLS for cell in cells):
        raise ValueError("requested cells are invalid or duplicated")
    if "D2-K4" in cells and profile_index.get("k4_correction") is None:
        raise ValueError("D2-K4 requires the corrected post-D1-K4 profile index")
    try:
        datetime.strptime(approved_at_utc, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError("approved-at-utc must be whole-second UTC") from exc
    queue_rows = []
    for cell_id in cells:
        fixture_id, arm_id = cell_id.split("-", 1)
        controls = modules["accelerated"].derive_cell_controls(
            campaign_path=campaign_path,
            profile_index_path=profile_index_path,
            throughput_profile_path=throughput_profile_path,
            fixture_id=fixture_id,
            arm_id=arm_id,
        )
        controls_path = output_dir / f"{cell_id}.controls.fc70e616.json"
        payload_path = output_dir / f"{cell_id}.plan.payload.fc70e616.json"
        plan_path = output_dir / f"{cell_id}.plan.fc70e616.json"
        approval_path = output_dir / f"{cell_id}.approval.fc70e616.json"
        _publish(controls_path, controls)
        payload = _cell_payload(
            cell_id=cell_id,
            spec=spec,
            controls=controls,
            profile_index=profile_index,
            profile_index_path=profile_index_path,
            profile_index_file_sha=profile_index_file_sha,
            throughput_profile=throughput_profile,
            throughput_profile_path=throughput_profile_path,
            throughput_profile_file_sha=throughput_profile_file_sha,
            host_manifest_path=host_manifest_path,
            host_manifest_file_sha=host_manifest_file_sha,
            forge_root=forge_root,
            modules=modules,
        )
        _publish(payload_path, payload)
        plan = modules["execution"].seal_plan(payload)
        modules["runtime"].validate_plan_against_profile_index(
            plan, profile_index=profile_index
        )
        _publish(plan_path, plan)
        approval = modules["execution"].build_approval(
            plan,
            reviewer_identity=None,
            approved_at_utc=approved_at_utc,
            admission_envelope_path=admission_envelope_path,
            approval_output_path=approval_path,
            technical_reviewer_actor=technical_actor,
        )
        _publish(approval_path, approval)
        run_dir = campaign_root / cell_id
        argv = [
            "/app/venv/bin/python",
            "-I",
            "-c",
            (
                "import runpy,sys;sys.path.insert(0,'/app/forge');"
                "runpy.run_module('ops.calibration.run_krea_ladder',run_name='__main__')"
            ),
            "--execution-plan",
            str(plan_path),
            "--execution-approval",
            str(approval_path),
            "--campaign-dir",
            str(run_dir),
        ]
        queue_rows.append(
            {
                "cell_id": cell_id,
                "controls": {"path": str(controls_path), "sha256": _file_sha(controls_path)},
                "payload": {"path": str(payload_path), "sha256": _file_sha(payload_path)},
                "plan": {"path": str(plan_path), "sha256": _file_sha(plan_path)},
                "plan_sha256": plan["plan_sha256"],
                "approval": {"path": str(approval_path), "sha256": _file_sha(approval_path)},
                "approval_sha256": approval["approval_sha256"],
                "campaign_dir": str(run_dir),
                "argv": argv,
            }
        )
    tool_path = _safe_file(Path(__file__), "cell queue tool")
    body = {
        "schema": 1,
        "kind": _QUEUE_KIND,
        "forge_commit": FC70_COMMIT,
        "tool": {"path": str(tool_path), "sha256": _file_sha(tool_path)},
        "assembly_spec": {
            "path": str(spec_path),
            "file_sha256": spec_file_sha,
            "spec_sha256": spec["spec_sha256"],
        },
        "campaign": {"path": str(campaign_path), "sha256": _file_sha(campaign_path)},
        "profile_index": {
            "path": str(profile_index_path),
            "sha256": profile_index_file_sha,
            "index_sha256": profile_index["index_sha256"],
        },
        "cells": queue_rows,
        "sequential_fail_closed": True,
    }
    queue = {**body, "queue_sha256": _semantic_sha(body)}
    _publish(queue_output, queue)
    return queue


def validate_queue(path: Path) -> dict[str, Any]:
    _, queue, _ = _load(path, "cell queue")
    _exact(
        queue,
        {
            "schema",
            "kind",
            "forge_commit",
            "tool",
            "assembly_spec",
            "campaign",
            "profile_index",
            "cells",
            "sequential_fail_closed",
            "queue_sha256",
        },
        "cell queue",
    )
    body = {key: item for key, item in queue.items() if key != "queue_sha256"}
    if (
        queue["schema"] != 1
        or queue["kind"] != _QUEUE_KIND
        or queue["forge_commit"] != FC70_COMMIT
        or queue["sequential_fail_closed"] is not True
        or queue["queue_sha256"] != _semantic_sha(body)
    ):
        raise ValueError("cell queue identity is invalid")
    tool = _object(queue["tool"], "queue tool")
    tool_path = _safe_file(tool["path"], "queue tool")
    if not tool_path.is_relative_to(Path("/campaign/controls")):
        raise ValueError("queue tool is outside durable campaign controls")
    if _file_sha(tool_path) != _sha(
        tool["sha256"], "queue tool SHA-256"
    ):
        raise ValueError("queue tool drifted")
    seen = []
    for row in queue["cells"]:
        row = _object(row, "queue cell")
        cell_id = row["cell_id"]
        if cell_id not in _CELLS or cell_id in seen:
            raise ValueError("queue cell order/identity is invalid")
        seen.append(cell_id)
        for label in ("controls", "payload", "plan", "approval"):
            binding = _object(row[label], f"queue {cell_id} {label}")
            bound_path = _safe_file(binding["path"], f"queue {label}")
            if not bound_path.is_relative_to(Path("/campaign/controls")):
                raise ValueError(f"queue {cell_id} {label} escaped controls")
            if _file_sha(bound_path) != _sha(
                binding["sha256"], f"queue {label} SHA-256"
            ):
                raise ValueError(f"queue {cell_id} {label} drifted")
        campaign_dir = Path(row["campaign_dir"])
        expected_prefix = [
            "/app/venv/bin/python",
            "-I",
            "-c",
            (
                "import runpy,sys;sys.path.insert(0,'/app/forge');"
                "runpy.run_module('ops.calibration.run_krea_ladder',run_name='__main__')"
            ),
            "--execution-plan",
            row["plan"]["path"],
            "--execution-approval",
            row["approval"]["path"],
            "--campaign-dir",
            row["campaign_dir"],
        ]
        if (
            not campaign_dir.is_relative_to(Path("/campaign/runs"))
            or row["argv"] != expected_prefix
        ):
            raise ValueError("queue campaign directory differs from exact argv")
    return queue


def run_queue(path: Path) -> None:
    queue = validate_queue(path)
    _modules(Path("/app/forge"))
    for row in queue["cells"]:
        campaign_dir = Path(row["campaign_dir"])
        campaign_dir.mkdir(parents=True, exist_ok=False)
        subprocess.run(row["argv"], check=True)


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal-spec")
    seal.add_argument("--payload", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    assemble_parser = commands.add_parser("assemble")
    assemble_parser.add_argument("--forge-root", type=Path, required=True)
    assemble_parser.add_argument("--assembly-spec", type=Path, required=True)
    assemble_parser.add_argument("--campaign", type=Path, required=True)
    assemble_parser.add_argument("--profile-index", type=Path, required=True)
    assemble_parser.add_argument("--throughput-profile", type=Path, required=True)
    assemble_parser.add_argument("--host-manifest", type=Path, required=True)
    assemble_parser.add_argument("--admission-envelope", type=Path, required=True)
    assemble_parser.add_argument("--technical-reviewer-actor", type=Path, required=True)
    assemble_parser.add_argument("--output-dir", type=Path, required=True)
    assemble_parser.add_argument("--campaign-root", type=Path, required=True)
    assemble_parser.add_argument("--approved-at-utc", required=True)
    assemble_parser.add_argument("--cell", action="append", choices=_CELLS)
    assemble_parser.add_argument("--queue-output", type=Path, required=True)
    validate = commands.add_parser("validate-queue")
    validate.add_argument("--queue", type=Path, required=True)
    run = commands.add_parser("run-queue")
    run.add_argument("--queue", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    if args.command == "seal-spec":
        _, payload, _ = _load(args.payload, "cell assembly-spec payload")
        spec = seal_spec(payload)
        _publish(args.output, spec)
        print(
            _canonical_bytes(
                {"status": "PASS", "spec_sha256": spec["spec_sha256"]}
            ).decode()
        )
        return 0
    if args.command == "assemble":
        cells = tuple(args.cell) if args.cell else _INITIAL_CELLS
        queue = assemble(
            forge_root=args.forge_root,
            spec_path=args.assembly_spec,
            campaign_path=args.campaign,
            profile_index_path=args.profile_index,
            throughput_profile_path=args.throughput_profile,
            host_manifest_path=args.host_manifest,
            admission_envelope_path=args.admission_envelope,
            technical_actor_path=args.technical_reviewer_actor,
            output_dir=args.output_dir,
            campaign_root=args.campaign_root,
            approved_at_utc=args.approved_at_utc,
            cells=cells,
            queue_output=args.queue_output,
        )
        print(_canonical_bytes({"status": "PASS", "queue_sha256": queue["queue_sha256"]}).decode())
        return 0
    if args.command == "validate-queue":
        queue = validate_queue(args.queue)
        print(_canonical_bytes({"status": "PASS", "queue_sha256": queue["queue_sha256"]}).decode())
        return 0
    run_queue(args.queue)
    print(_canonical_bytes({"status": "PASS", "action": "run-queue"}).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
