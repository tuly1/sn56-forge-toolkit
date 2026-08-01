#!/usr/bin/env python3
"""Bounded fail-closed dispatcher for the Week-5 Stage-2 endgame.

The supervisor only coordinates already-sealed training claims and exact-score
claims.  It never changes plans, retries a failed cell, waives a gate, selects
a release, or extends its deadline.  One score worker is preferred as soon as
a grouped score plan becomes ready; remaining GPUs continue the fixed training
queue, so confirmation scoring overlaps later confirmation and boundary runs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, MutableMapping, Sequence

try:
    from . import krea_provenance
    from . import krea_stage2_endgame_matrix
    from . import krea_stage2_endgame_orchestrator as training
    from . import krea_stage2_endgame_scoring as scoring
    from . import krea_stage2_execution
except ImportError:  # pragma: no cover - direct CLI execution.
    import krea_provenance  # type: ignore[no-redef]
    import krea_stage2_endgame_matrix  # type: ignore[no-redef]
    import krea_stage2_endgame_orchestrator as training  # type: ignore[no-redef]
    import krea_stage2_endgame_scoring as scoring  # type: ignore[no-redef]
    import krea_stage2_execution  # type: ignore[no-redef]


SCHEMA = 1
CONFIG_KIND = "forge-krea-stage2-endgame-supervisor-config"
LAUNCH_KIND = "forge-krea-stage2-endgame-worker-launch"
EXIT_KIND = "forge-krea-stage2-endgame-worker-exit"
GPU_IDS = training.GPU_IDS


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(
            f"{label} keys differ: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )


def _load(path: str | Path, label: str) -> dict[str, Any]:
    return krea_stage2_endgame_matrix._load_canonical(path, label)


def _publish(path: str | Path, value: Mapping[str, Any], label: str) -> dict[str, Any]:
    return krea_stage2_endgame_matrix._publish_or_replay(path, value, label)


def _absolute(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a path string")
    path = Path(value)
    if not path.is_absolute() or str(path) != os.path.abspath(str(path)):
        raise ValueError(f"{label} must be absolute and normalized")
    return str(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _utc_datetime(value: str, label: str) -> datetime:
    resolved = krea_stage2_execution._utc(value, label)
    return datetime.strptime(resolved, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )


def build_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal one operator-supplied, absolute-path launch configuration."""

    raw = _object(payload, "supervisor config payload")
    keys = {
        "matrix",
        "plan_set",
        "authority_bundle",
        "training_claims_root",
        "score_config",
        "score_output_root",
        "score_claims_root",
        "gpu_lock_root",
        "worker_state_root",
        "training_gate",
        "score_gate",
        "deadline_utc",
        "scheduler_instance_id",
        "poll_interval_seconds",
    }
    _exact(raw, keys, "supervisor config payload")
    body = {"schema": SCHEMA, "kind": CONFIG_KIND, **dict(raw)}
    record = {
        **body,
        "config_sha256": krea_provenance.canonical_sha256(body),
    }
    return validate_config(record)


def validate_config(value: Any) -> dict[str, Any]:
    config = _object(value, "supervisor config")
    keys = {
        "schema",
        "kind",
        "matrix",
        "plan_set",
        "authority_bundle",
        "training_claims_root",
        "score_config",
        "score_output_root",
        "score_claims_root",
        "gpu_lock_root",
        "worker_state_root",
        "training_gate",
        "score_gate",
        "deadline_utc",
        "scheduler_instance_id",
        "poll_interval_seconds",
        "config_sha256",
    }
    _exact(config, keys, "supervisor config")
    body = {key: item for key, item in config.items() if key != "config_sha256"}
    if (
        config["schema"] != SCHEMA
        or config["kind"] != CONFIG_KIND
        or config["config_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("supervisor config identity differs")
    for field in (
        "matrix",
        "plan_set",
        "authority_bundle",
        "training_claims_root",
        "score_config",
        "score_output_root",
        "score_claims_root",
        "gpu_lock_root",
        "worker_state_root",
        "training_gate",
        "score_gate",
    ):
        _absolute(config[field], f"supervisor {field}")
    for field in ("matrix", "plan_set", "authority_bundle", "score_config"):
        path = Path(config[field])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"supervisor {field} is not a live regular file")
    training._safe_id(config["scheduler_instance_id"], "scheduler instance id")
    _utc_datetime(config["deadline_utc"], "supervisor deadline")
    interval = config["poll_interval_seconds"]
    if (
        not isinstance(interval, int)
        or isinstance(interval, bool)
        or not 1 <= interval <= 60
    ):
        raise ValueError("supervisor poll interval must be an integer from 1 to 60")
    return dict(config)


def _worker_id(kind: str, key: str) -> str:
    return training._safe_id(f"{kind}-{key}", "worker id")


def _worker_spec(
    *, kind: str, key: str, gpu_device: int, claim_path: Path, result_path: str
) -> dict[str, Any]:
    return {
        "worker_id": _worker_id(kind, key),
        "worker_kind": kind,
        "key": key,
        "gpu_device": gpu_device,
        "claim_path": str(claim_path),
        "result_path": result_path,
    }


def _outstanding_specs(
    *,
    config: Mapping[str, Any],
    matrix: Mapping[str, Any],
    plan_set: Mapping[str, Any],
    score_queue: Mapping[str, Any],
    running_worker_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    running_worker_ids = set() if running_worker_ids is None else running_worker_ids
    specs: list[dict[str, Any]] = []
    rows = {row["row_key"]: row for row in plan_set["rows"]}
    training_root = Path(config["training_claims_root"])
    if training_root.exists():
        for path in sorted(training_root.glob("*.json")):
            claim = training._load(path, "supervisor training claim")
            claim = training._validate_claim(claim, plan_set=plan_set, matrix=matrix)
            row = rows[claim["row_key"]]
            worker_id = _worker_id("training", claim["row_key"])
            state = Path(config["worker_state_root"]) / worker_id
            needs_reconcile = worker_id in running_worker_ids or (
                os.path.lexists(state / "launch.json")
                and not os.path.lexists(state / "exit.json")
            )
            if not os.path.lexists(row["receipt_path"]) or needs_reconcile:
                specs.append(
                    _worker_spec(
                        kind="training",
                        key=claim["row_key"],
                        gpu_device=claim["gpu_device"],
                        claim_path=path,
                        result_path=row["receipt_path"],
                    )
                )
    groups = {row["group_key"]: row for row in score_queue["groups"]}
    score_root = Path(config["score_claims_root"])
    if score_root.exists():
        for path in sorted(score_root.glob("*.json")):
            claim = scoring._validate_claim(
                scoring._load(path, "supervisor score claim"),
                score_queue=score_queue,
            )
            group = groups[claim["group_key"]]
            worker_id = _worker_id("score", claim["group_key"])
            state = Path(config["worker_state_root"]) / worker_id
            needs_reconcile = worker_id in running_worker_ids or (
                os.path.lexists(state / "launch.json")
                and not os.path.lexists(state / "exit.json")
            )
            if not os.path.lexists(group["aggregate_path"]) or needs_reconcile:
                specs.append(
                    _worker_spec(
                        kind="score",
                        key=claim["group_key"],
                        gpu_device=claim["gpu_device"],
                        claim_path=path,
                        result_path=group["aggregate_path"],
                    )
                )
    collisions: dict[int, list[str]] = {}
    for spec in specs:
        collisions.setdefault(spec["gpu_device"], []).append(spec["worker_id"])
    bad = {gpu: keys for gpu, keys in collisions.items() if len(keys) > 1}
    if bad:
        raise ValueError(f"outstanding worker claims collide by GPU: {bad}")
    return specs


def _command(config: Mapping[str, Any], spec: Mapping[str, Any]) -> list[str]:
    if spec["worker_kind"] == "training":
        script = Path(training.__file__).resolve(strict=True)
        return [
            sys.executable,
            str(script),
            "run-claim",
            "--claim",
            spec["claim_path"],
            "--plan-set",
            config["plan_set"],
            "--matrix",
            config["matrix"],
            "--authority-bundle",
            config["authority_bundle"],
            "--gpu-lock-root",
            config["gpu_lock_root"],
        ]
    script = Path(scoring.__file__).resolve(strict=True)
    return [
        sys.executable,
        str(script),
        "run-claim",
        "--config",
        config["score_config"],
        "--score-queue",
        str(Path(config["score_output_root"]) / "score-queue.json"),
        "--claim",
        spec["claim_path"],
        "--gpu-lock-root",
        config["gpu_lock_root"],
    ]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _launch_worker(
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    running: MutableMapping[str, subprocess.Popen],
) -> None:
    state = Path(config["worker_state_root"]) / spec["worker_id"]
    state.mkdir(parents=True, exist_ok=True)
    launch_path = state / "launch.json"
    if os.path.lexists(launch_path):
        raise ValueError(f"worker {spec['worker_id']} already has a launch record")
    command = _command(config, spec)
    stdout_path = state / "stdout.log"
    stderr_path = state / "stderr.log"
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            start_new_session=True,
        )
    body = {
        "schema": SCHEMA,
        "kind": LAUNCH_KIND,
        **dict(spec),
        "command": command,
        "command_sha256": krea_provenance.canonical_sha256(command),
        "pid": process.pid,
        "launched_at_utc": _utc_now(),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "retry_allowed": False,
        "waiver_used": False,
    }
    launch = {**body, "launch_sha256": krea_provenance.canonical_sha256(body)}
    krea_stage2_endgame_matrix._publish_new(launch_path, launch)
    running[spec["worker_id"]] = process


def _record_exit(
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    returncode: int,
    recovered_after_restart: bool,
) -> None:
    state = Path(config["worker_state_root"]) / spec["worker_id"]
    body = {
        "schema": SCHEMA,
        "kind": EXIT_KIND,
        "worker_id": spec["worker_id"],
        "returncode": returncode,
        "result_path": spec["result_path"],
        "result_present": os.path.lexists(spec["result_path"]),
        "recovered_after_restart": recovered_after_restart,
        "exited_at_utc": _utc_now(),
        "retry_allowed": False,
        "waiver_used": False,
    }
    record = {**body, "exit_sha256": krea_provenance.canonical_sha256(body)}
    krea_stage2_endgame_matrix._publish_new(state / "exit.json", record)


def _reconcile(
    config: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    running: MutableMapping[str, subprocess.Popen],
) -> tuple[set[int], set[str]]:
    active_gpus: set[int] = set()
    active_score_ids: set[str] = set()
    for spec in specs:
        state = Path(config["worker_state_root"]) / spec["worker_id"]
        launch_path = state / "launch.json"
        exit_path = state / "exit.json"
        process = running.get(spec["worker_id"])
        if process is not None:
            code = process.poll()
            if code is None:
                active_gpus.add(spec["gpu_device"])
                if spec["worker_kind"] == "score":
                    active_score_ids.add(spec["worker_id"])
                continue
            running.pop(spec["worker_id"], None)
            _record_exit(
                config,
                spec,
                returncode=code,
                recovered_after_restart=False,
            )
            if code != 0 or not os.path.lexists(spec["result_path"]):
                raise RuntimeError(
                    f"worker {spec['worker_id']} failed rc={code}; "
                    f"logs={state}"
                )
            continue
        if os.path.lexists(exit_path):
            if not os.path.lexists(spec["result_path"]):
                raise RuntimeError(
                    f"worker {spec['worker_id']} has an exit record without result"
                )
            continue
        if os.path.lexists(launch_path):
            launch = _load(launch_path, "worker launch")
            if launch.get("worker_id") != spec["worker_id"]:
                raise ValueError("worker launch identity differs")
            pid = launch.get("pid")
            if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
                raise ValueError("worker launch PID is invalid")
            if _pid_alive(pid):
                active_gpus.add(spec["gpu_device"])
                if spec["worker_kind"] == "score":
                    active_score_ids.add(spec["worker_id"])
                continue
            if not os.path.lexists(spec["result_path"]):
                raise RuntimeError(
                    f"worker {spec['worker_id']} vanished without result; logs={state}"
                )
            _record_exit(
                config,
                spec,
                returncode=0,
                recovered_after_restart=True,
            )
            continue
        _launch_worker(config, spec, running)
        active_gpus.add(spec["gpu_device"])
        if spec["worker_kind"] == "score":
            active_score_ids.add(spec["worker_id"])
    return active_gpus, active_score_ids


def _completion_counts(
    plan_set: Mapping[str, Any], score_queue: Mapping[str, Any]
) -> tuple[int, int]:
    trained = sum(os.path.lexists(row["receipt_path"]) for row in plan_set["rows"])
    scored = sum(
        os.path.lexists(row["aggregate_path"]) for row in score_queue["groups"]
    )
    return trained, scored


def _gate_time(score_queue: Mapping[str, Any], now_utc: str) -> str:
    latest = max(
        _utc_datetime(
            _load(row["aggregate_path"], "score aggregate time")["emitted_at_utc"],
            "score aggregate time",
        )
        for row in score_queue["groups"]
    )
    current = _utc_datetime(now_utc, "supervisor current time")
    return max(current, latest + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _seal_completion_gates(
    *,
    config: Mapping[str, Any],
    matrix: Mapping[str, Any],
    plan_set: Mapping[str, Any],
    authority: Mapping[str, Any],
    score_queue: Mapping[str, Any],
    now_utc: str,
) -> dict[str, Any]:
    training_time = now_utc
    training_path = Path(config["training_gate"])
    if os.path.lexists(training_path):
        training_time = _load(training_path, "existing training gate")[
            "completed_at_utc"
        ]
    training_gate = training.seal_exact60_gate(
        plan_set=plan_set,
        matrix=matrix,
        authority_bundle=authority,
        output=training_path,
        completed_at_utc=training_time,
    )
    score_path = Path(config["score_gate"])
    score_time = _gate_time(score_queue, now_utc)
    if os.path.lexists(score_path):
        score_time = _load(score_path, "existing score gate")["completed_at_utc"]
    score_gate = scoring.seal_score_gate(
        score_queue=score_queue,
        output=score_path,
        completed_at_utc=score_time,
    )
    return {
        "state": "complete",
        "training_gate_sha256": training_gate["gate_sha256"],
        "score_gate_sha256": score_gate["gate_sha256"],
    }


def dispatch_once(
    config: Mapping[str, Any],
    *,
    running: MutableMapping[str, subprocess.Popen] | None = None,
    now_utc: str | None = None,
) -> dict[str, Any]:
    """Advance the queue once; safe to call repeatedly or after a restart."""

    supplied = validate_config(config)
    running = {} if running is None else running
    now = _utc_now() if now_utc is None else krea_stage2_execution._utc(
        now_utc, "supervisor current time"
    )
    matrix = krea_stage2_endgame_matrix.validate_matrix(
        _load(supplied["matrix"], "supervisor matrix")
    )
    plan_set = training.validate_plan_set(
        _load(supplied["plan_set"], "supervisor plan set"), matrix=matrix
    )
    authority = training._validate_authority_bundle(
        _load(supplied["authority_bundle"], "supervisor authority bundle")
    )
    score_config = scoring._validate_config(
        _load(supplied["score_config"], "supervisor score config")
    )
    materialized = scoring.materialize_ready_score_plans(
        score_config, output_root=supplied["score_output_root"]
    )
    score_queue = scoring._validate_queue(materialized["queue"])
    if (
        score_queue["matrix_sha256"] != matrix["matrix_sha256"]
        or score_queue["training_plan_set_sha256"] != plan_set["plan_set_sha256"]
    ):
        raise ValueError("supervisor score queue differs from training authority")

    specs = _outstanding_specs(
        config=supplied,
        matrix=matrix,
        plan_set=plan_set,
        score_queue=score_queue,
        running_worker_ids=set(running),
    )
    active_gpus, active_scores = _reconcile(supplied, specs, running)
    trained, scored = _completion_counts(plan_set, score_queue)
    if trained == training.EXPECTED_ROWS and scored == scoring.GROUP_COUNT:
        return {
            **_seal_completion_gates(
                config=supplied,
                matrix=matrix,
                plan_set=plan_set,
                authority=authority,
                score_queue=score_queue,
                now_utc=now,
            ),
            "training_completed": trained,
            "score_groups_completed": scored,
            "active_workers": len(running),
        }
    if _utc_datetime(now, "supervisor current time") >= _utc_datetime(
        supplied["deadline_utc"], "supervisor deadline"
    ):
        raise RuntimeError(
            f"endgame deadline reached incomplete: training={trained}/60, "
            f"scores={scored}/16"
        )

    free = [gpu for gpu in GPU_IDS if gpu not in active_gpus]
    ready_scores = [
        row
        for row in score_queue["groups"]
        if os.path.lexists(row["group_path"])
        and not os.path.lexists(row["aggregate_path"])
    ]
    new_claims = 0
    if ready_scores and not active_scores and free:
        claims = scoring.claim_ready_groups(
            score_queue=score_queue,
            claims_root=supplied["score_claims_root"],
            claimed_at_utc=now,
            scheduler_instance_id=supplied["scheduler_instance_id"],
            gpu_devices=[free[0]],
        )
        new_claims += len(claims)
        active_gpus.update(claim["gpu_device"] for claim in claims)
        free = [gpu for gpu in GPU_IDS if gpu not in active_gpus]

    claims = training.claim_next(
        plan_set=plan_set,
        matrix=matrix,
        claims_root=supplied["training_claims_root"],
        claimed_at_utc=now,
        scheduler_instance_id=supplied["scheduler_instance_id"],
        gpu_devices=free,
    )
    new_claims += len(claims)
    active_gpus.update(claim["gpu_device"] for claim in claims)
    free = [gpu for gpu in GPU_IDS if gpu not in active_gpus]
    if not claims and ready_scores and free:
        claims = scoring.claim_ready_groups(
            score_queue=score_queue,
            claims_root=supplied["score_claims_root"],
            claimed_at_utc=now,
            scheduler_instance_id=supplied["scheduler_instance_id"],
            gpu_devices=free,
        )
        new_claims += len(claims)

    specs = _outstanding_specs(
        config=supplied,
        matrix=matrix,
        plan_set=plan_set,
        score_queue=score_queue,
        running_worker_ids=set(running),
    )
    active_gpus, _active_scores = _reconcile(supplied, specs, running)
    trained, scored = _completion_counts(plan_set, score_queue)
    return {
        "state": "running",
        "training_completed": trained,
        "score_groups_completed": scored,
        "active_gpus": sorted(active_gpus),
        "active_workers": len(specs),
        "new_claims": new_claims,
        "deadline_utc": supplied["deadline_utc"],
        "release_authorized": False,
        "production_mutation_authorized": False,
    }


def run(config: Mapping[str, Any]) -> dict[str, Any]:
    supplied = validate_config(config)
    running: dict[str, subprocess.Popen] = {}
    stopping = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    previous_int = signal.signal(signal.SIGINT, request_stop)
    try:
        while not stopping:
            status = dispatch_once(supplied, running=running)
            print(json.dumps(status, sort_keys=True, separators=(",", ":")), flush=True)
            if status["state"] == "complete":
                return status
            time.sleep(supplied["poll_interval_seconds"])
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
    raise RuntimeError("supervisor stopped before endgame gates closed")


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    configure = sub.add_parser("configure")
    configure.add_argument("--payload", required=True, type=Path)
    configure.add_argument("--output", required=True, type=Path)
    once = sub.add_parser("once")
    once.add_argument("--config", required=True, type=Path)
    execute = sub.add_parser("run")
    execute.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    try:
        if args.command == "configure":
            result = build_config(_load(args.payload, "supervisor config payload"))
            krea_stage2_endgame_matrix._publish_new(args.output, result)
        elif args.command == "once":
            result = dispatch_once(_load(args.config, "supervisor config"))
        else:
            result = run(_load(args.config, "supervisor config"))
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
