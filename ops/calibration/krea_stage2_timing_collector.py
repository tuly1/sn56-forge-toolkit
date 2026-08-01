#!/usr/bin/env python3
"""Run and seal exact production-image Stage-2 timing receipts.

This is a host-side collector.  It never enters the production image and it
does not infer timing from trainer logs.  Linux inotify timestamps the durable
config, terminal, checkpoint, and selection writes; the Docker process clock
bounds startup and upload-ready finalization.  Three measurement receipts and
one held-out receipt are mandatory before a throughput bundle can be emitted.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import stat
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence

try:
    from . import krea_provenance
    from . import krea_stage2_timing as timing
except ImportError:  # pragma: no cover - direct execution on the GPU host.
    import krea_provenance  # type: ignore[no-redef]
    import krea_stage2_timing as timing  # type: ignore[no-redef]


SCHEMA = 1
ARTIFACT_MANIFEST_KIND = "forge-krea-stage2-timing-run-artifact-manifest"
_IN_CLOSE_WRITE = 0x00000008
_IN_CREATE = 0x00000100
_IN_MOVED_TO = 0x00000080
_EVENT = struct.Struct("iIII")


def _utc_now(*, ceiling: bool = False) -> str:
    now = datetime.now(timezone.utc)
    if ceiling and now.microsecond:
        now = datetime.fromtimestamp(int(now.timestamp()) + 1, timezone.utc)
    return now.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_file_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(krea_provenance.canonical_bytes(value) + b"\n").hexdigest()


def _load_canonical(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a real file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if (
        not isinstance(value, dict)
        or raw != krea_provenance.canonical_bytes(value) + b"\n"
    ):
        raise ValueError(f"{label} is not canonical JSON plus newline")
    return value


def _publish(path: Path, value: Mapping[str, Any], *, mode: int = 0o400) -> None:
    payload = krea_provenance.canonical_bytes(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def _real_directory(path: Path, label: str, *, writable: bool) -> Path:
    resolved = Path(os.path.abspath(os.fspath(path)))
    if (
        resolved.is_symlink()
        or not resolved.is_dir()
        or os.path.realpath(resolved) != str(resolved)
    ):
        raise ValueError(f"{label} must be an existing real directory")
    if writable and not os.access(resolved, os.W_OK | os.X_OK):
        raise ValueError(f"{label} is not writable")
    return resolved


class _Inotify:
    def __init__(self, directory: Path, callback: Callable[[str, int], None]):
        if sys.platform != "linux":
            raise RuntimeError("Stage-2 timing collection requires Linux inotify")
        self.callback = callback
        self.errors: list[BaseException] = []
        self.stop = threading.Event()
        libc = ctypes.CDLL(None, use_errno=True)
        init = libc.inotify_init1
        init.argtypes = [ctypes.c_int]
        init.restype = ctypes.c_int
        self.fd = int(init(os.O_NONBLOCK | os.O_CLOEXEC))
        if self.fd < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        add = libc.inotify_add_watch
        add.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add.restype = ctypes.c_int
        watch = int(
            add(
                self.fd,
                os.fsencode(directory),
                _IN_CREATE | _IN_CLOSE_WRITE | _IN_MOVED_TO,
            )
        )
        if watch < 0:
            error = ctypes.get_errno()
            os.close(self.fd)
            raise OSError(error, os.strerror(error))
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _loop(self) -> None:
        try:
            while True:
                readable, _, _ = select.select([self.fd], [], [], 0.1)
                if not readable:
                    if self.stop.is_set():
                        break
                    continue
                data = os.read(self.fd, 64 * 1024)
                offset = 0
                while offset < len(data):
                    _watch, mask, _cookie, length = _EVENT.unpack_from(data, offset)
                    offset += _EVENT.size
                    raw_name = data[offset : offset + length]
                    offset += length
                    name = raw_name.split(b"\0", 1)[0].decode("utf-8", "strict")
                    self.callback(name, mask)
        except BaseException as exc:  # fail closed in close().
            self.errors.append(exc)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=3)
        os.close(self.fd)
        if self.thread.is_alive():
            raise RuntimeError("Stage-2 timing inotify observer did not stop")
        if self.errors:
            raise RuntimeError(
                "Stage-2 timing inotify observer failed"
            ) from self.errors[0]


class _EventStream:
    def __init__(self, receipt_ordinal: int):
        self.receipt_ordinal = receipt_ordinal
        self.events: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.states: dict[str, str] = {}
        self.checkpoint_tokens: dict[str, str] = {}
        self.closed_saves = 0

    def emit(self, token: str, metric: str, state: str, units: int) -> None:
        with self.lock:
            prior = self.states.get(token)
            if (state == "begin" and prior is not None) or (
                state == "end" and prior != "begin"
            ):
                raise RuntimeError(f"timing span lifecycle differs: {token}")
            self.states[token] = state
            self.events.append(
                timing.seal_event(
                    sequence=len(self.events),
                    span_token=token,
                    metric=metric,
                    state=state,
                    counter_value=0 if state == "begin" else units,
                    received_monotonic_ns=time.monotonic_ns(),
                )
            )

    def evidence(self, name: str, mask: int) -> None:
        if not mask & (_IN_CLOSE_WRITE | _IN_MOVED_TO):
            return
        if name == "config-control.json":
            self.emit(f"startup-r{self.receipt_ordinal}", "startup", "end", 1)
            self.emit(
                f"updates-r{self.receipt_ordinal}", "optimizer_update", "begin", 34
            )
        elif name == "training-terminal.json":
            self.emit(f"updates-r{self.receipt_ordinal}", "optimizer_update", "end", 34)
            self.emit(f"finalize-r{self.receipt_ordinal}", "finalization", "begin", 1)
        elif name == "forge_checkpoint_selection.json":
            self.emit(f"finalize-r{self.receipt_ordinal}", "finalization", "end", 1)
            self.emit(f"upload-r{self.receipt_ordinal}", "upload", "begin", 1)

    def checkpoint(self, name: str, mask: int) -> None:
        if not name.endswith(".safetensors") or name == "last.safetensors":
            return
        with self.lock:
            token = self.checkpoint_tokens.get(name)
            if mask & _IN_CREATE and token is None and self.closed_saves < 3:
                token = f"save-r{self.receipt_ordinal}-{self.closed_saves + len(self.checkpoint_tokens):02d}"
                self.checkpoint_tokens[name] = token
            elif mask & (_IN_CLOSE_WRITE | _IN_MOVED_TO):
                token = self.checkpoint_tokens.pop(name, None)
            else:
                token = None
        if token is None:
            return
        if mask & _IN_CREATE:
            self.emit(token, "checkpoint_save", "begin", 1)
        elif mask & (_IN_CLOSE_WRITE | _IN_MOVED_TO):
            self.emit(token, "checkpoint_save", "end", 1)
            with self.lock:
                self.closed_saves += 1

    def finish(self) -> list[dict[str, Any]]:
        self.emit(f"upload-r{self.receipt_ordinal}", "upload", "end", 1)
        with self.lock:
            if self.closed_saves != 3 or self.checkpoint_tokens:
                raise RuntimeError(
                    "exactly three complete checkpoint writes were not observed"
                )
            if any(state != "end" for state in self.states.values()):
                raise RuntimeError("timing event stream contains an open span")
            return list(self.events)


def _validate_semantic_receipt(path: Path, label: str) -> dict[str, Any]:
    value = _load_canonical(path, label)
    semantic = value.get("receipt_sha256")
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if semantic != krea_provenance.canonical_sha256(body):
        raise ValueError(f"{label} semantic digest differs")
    return value


def _validate_run_outputs(
    *,
    plan: Mapping[str, Any],
    probe: Mapping[str, Any],
    receipt_ordinal: int,
    checkpoint_root: Path,
    evidence_root: Path,
) -> None:
    fields = probe["command_fields"]
    schedule = probe["receipt_schedule"][receipt_ordinal]
    control = _validate_semantic_receipt(
        evidence_root / "config-control.json", "config-control receipt"
    )
    terminal = _validate_semantic_receipt(
        evidence_root / "training-terminal.json", "training-terminal receipt"
    )
    common = {
        "timing_plan_sha256": plan["plan_sha256"],
        "probe_contract_sha256": probe["probe_contract_sha256"],
        "profile_id": fields["profile_id"],
        "training_seed": schedule["seed"],
        "planned_steps": fields["bootstrap_steps"],
        "throughput_profile_sha256": None,
        "release_authorized": False,
    }
    for key, expected in common.items():
        if control.get(key) != expected or terminal.get(key) != expected:
            raise ValueError(f"timing bootstrap receipt {key} differs")
    if (
        control.get("kind")
        != "forge-krea-stage2-timing-bootstrap-config-control-receipt"
        or terminal.get("kind") != "forge-krea-stage2-timing-bootstrap-terminal-receipt"
        or control.get("mode") != "preprofile_timing_bootstrap"
        or terminal.get("mode") != "preprofile_timing_bootstrap"
        or control.get("production_mutation_authorized") is not False
        or terminal.get("production_mutation_authorized") is not False
        or terminal.get("last_step") != fields["bootstrap_steps"]
        or terminal.get("trainer_returncode") != 0
        or terminal.get("stopped_by_deadline") is not False
        or terminal.get("planned_steps_completed") is not True
        or terminal.get("natural_completion") is not True
    ):
        raise ValueError("timing bootstrap did not naturally complete")
    selection = _load_canonical(
        evidence_root / "forge_checkpoint_selection.json",
        "checkpoint-selection receipt",
    )
    selected = selection.get("selected_file")
    if (
        selection.get("status") != "selected_current_run"
        or selection.get("source") != "frozen_checkpoint_fraction"
        or selection.get("output_file") != "last.safetensors"
        or not isinstance(selected, str)
        or os.path.basename(selected) != selected
    ):
        raise ValueError("timing bootstrap checkpoint selection differs")
    last = checkpoint_root / "last.safetensors"
    source = checkpoint_root / selected
    for candidate in (last, source):
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.stat().st_size <= 8
        ):
            raise ValueError("timing bootstrap selected checkpoint is invalid")
    if _file_sha(last) != _file_sha(source) or _file_sha(last) != selection.get(
        "sha256"
    ):
        raise ValueError("timing bootstrap promoted checkpoint bytes differ")
    with last.open("rb") as handle:
        header_bytes = int.from_bytes(handle.read(8), "little")
        header = handle.read(header_bytes)
    try:
        parsed_header = json.loads(header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("timing bootstrap safetensors header is invalid") from exc
    if not isinstance(parsed_header, dict) or not parsed_header:
        raise ValueError("timing bootstrap safetensors header is empty")
    recorder = _load_canonical(checkpoint_root / "forge_run.json", "public recorder")
    names = [
        row.get("name") for row in recorder.get("events", []) if isinstance(row, dict)
    ]
    if (
        recorder.get("kind") != "forge-public-run-recorder"
        or "run_complete" not in names
        or "public_bundle_ready" not in names
        or any(
            isinstance(row, dict)
            and (
                row.get("failure_class") is not None
                or "fail" in str(row.get("name", ""))
                or "fallback" in str(row.get("name", ""))
            )
            for row in recorder.get("events", [])
        )
    ):
        raise ValueError("timing bootstrap public recorder is not clean/upload-ready")


def _artifact_manifest(
    *,
    plan_sha256: str,
    receipt_ordinal: int,
    roots: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    rows = []
    for prefix, root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("run artifact tree contains a symlink")
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            rows.append(
                {
                    "path": f"{prefix}/{relative}",
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha(path),
                }
            )
    if not rows:
        raise ValueError("run artifact manifest is empty")
    body = {
        "schema": SCHEMA,
        "kind": ARTIFACT_MANIFEST_KIND,
        "timing_plan_sha256": plan_sha256,
        "receipt_ordinal": receipt_ordinal,
        "files": rows,
    }
    return {**body, "artifact_manifest_sha256": krea_provenance.canonical_sha256(body)}


def _stage_archive(archive: Path, *, probe: Mapping[str, Any]) -> Path:
    if archive.is_symlink() or not archive.is_file():
        raise ValueError("training archive must be a real file")
    expected = probe["training_archive"]
    if (
        archive.stat().st_size != expected["bytes"]
        or _file_sha(archive) != expected["sha256"]
    ):
        raise ValueError("training archive differs from probe contract")
    datasets = next(row for row in probe["mounts"] if row["purpose"] == "dataset_cache")
    root = _real_directory(
        Path(datasets["source_root"]), "dataset cache", writable=True
    )
    target = root / (probe["command_fields"]["task_id"] + "_tourn.zip")
    if os.path.lexists(target):
        if (
            target.is_symlink()
            or target.stat().st_size != expected["bytes"]
            or _file_sha(target) != expected["sha256"]
        ):
            raise FileExistsError("dataset cache alias exists with different bytes")
        return target
    temporary = root / (
        "." + target.name + "." + probe["probe_contract_sha256"][:16] + ".tmp"
    )
    if os.path.lexists(temporary):
        raise FileExistsError(temporary)
    with archive.open("rb") as source, temporary.open("xb") as destination:
        shutil.copyfileobj(source, destination, 1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    os.chmod(temporary, 0o444)
    if _file_sha(temporary) != expected["sha256"]:
        raise ValueError("staged training archive digest differs")
    os.link(temporary, target)
    temporary.unlink()
    return target


def _run_one(
    *,
    plan: Mapping[str, Any],
    probe: Mapping[str, Any],
    receipt_ordinal: int,
    collection_root: Path,
    docker_path: Path,
) -> dict[str, Any]:
    writable = timing.receipt_mount_sources(probe, receipt_ordinal=receipt_ordinal)
    checkpoint_source = Path(writable["checkpoint_source"])
    evidence_source = Path(writable["evidence_source"])
    if os.path.lexists(checkpoint_source) or os.path.lexists(evidence_source):
        raise FileExistsError("timing receipt writable namespace already exists")
    checkpoint_source.mkdir(mode=0o700)
    evidence_source.mkdir(mode=0o700)
    fields = probe["command_fields"]
    checkpoint_root = (
        checkpoint_source / fields["task_id"] / fields["expected_repo_name"]
    )
    checkpoint_root.mkdir(parents=True, mode=0o700)
    evidence_root = evidence_source / plan["plan_sha256"]
    evidence_root.mkdir(mode=0o700)
    run_root = collection_root / f"run-{receipt_ordinal:03d}"
    run_root.mkdir(mode=0o700)
    stdout_path = run_root / "container.stdout"
    stderr_path = run_root / "container.stderr"
    stream = _EventStream(receipt_ordinal)
    checkpoint_watch = _Inotify(checkpoint_root, stream.checkpoint)
    evidence_watch = _Inotify(evidence_root, stream.evidence)
    checkpoint_watch.start()
    evidence_watch.start()
    command_argv = timing.render_probe_command(
        probe, timing_plan_sha256=plan["plan_sha256"], receipt_ordinal=receipt_ordinal
    )
    if Path(command_argv[0]) != docker_path:
        raise ValueError("Docker executable path differs from probe contract")
    started_unix_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    stream.emit(f"startup-r{receipt_ordinal}", "startup", "begin", 1)
    timed_out = False
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(command_argv, stdout=stdout, stderr=stderr)
            try:
                returncode = process.wait(timeout=float(plan["hard_budget_s"]))
            except subprocess.TimeoutExpired:
                timed_out = True
                subprocess.run(
                    [str(docker_path), "rm", "-f", command_argv[4]],
                    stdout=stderr,
                    stderr=stderr,
                    check=False,
                    timeout=60,
                )
                returncode = process.wait(timeout=60)
    except BaseException:
        checkpoint_watch.close()
        evidence_watch.close()
        raise
    checkpoint_watch.close()
    evidence_watch.close()
    if timed_out:
        raise RuntimeError("timing bootstrap exceeded its hard budget")
    events = stream.finish()
    ended_monotonic_ns = time.monotonic_ns()
    ended_unix_ns = time.time_ns()
    if returncode != 0:
        raise RuntimeError(f"timing bootstrap Docker command failed: {returncode}")
    _validate_run_outputs(
        plan=plan,
        probe=probe,
        receipt_ordinal=receipt_ordinal,
        checkpoint_root=checkpoint_root,
        evidence_root=evidence_root,
    )
    artifact = _artifact_manifest(
        plan_sha256=plan["plan_sha256"],
        receipt_ordinal=receipt_ordinal,
        roots=(
            ("checkpoints", checkpoint_source),
            ("evidence", evidence_source),
            ("collector", run_root),
        ),
    )
    artifact_path = run_root / "artifact-manifest.json"
    _publish(artifact_path, artifact)
    schedule = probe["receipt_schedule"][receipt_ordinal]
    run_receipt = timing.seal_run_receipt(
        measurement_role=schedule["measurement_role"],
        artifact_manifest_file_sha256=_file_sha(artifact_path),
        artifact_manifest_sha256=artifact["artifact_manifest_sha256"],
        recorded_unix_ns=time.time_ns(),
    )
    raw = timing.seal_raw_receipt(
        plan,
        probe_contract=probe,
        receipt_ordinal=receipt_ordinal,
        command={
            "argv": command_argv,
            "executable_id": probe["executable_id"],
            "executable_path": probe["executable_path"],
            "executable_sha256": probe["executable_sha256"],
            "returncode": returncode,
            "started_unix_ns": started_unix_ns,
            "ended_unix_ns": ended_unix_ns,
            "started_monotonic_ns": started_monotonic_ns,
            "ended_monotonic_ns": ended_monotonic_ns,
            "production_image_id": plan["production_image_id"],
            "network_mode": "none",
            "runtime": "nvidia",
        },
        events=events,
        run_receipt=run_receipt,
    )
    raw_path = run_root / "raw-receipt.json"
    _publish(raw_path, raw)
    return {
        "record": raw,
        "file_sha256": _file_sha(raw_path),
        "receipt_sha256": raw["receipt_sha256"],
    }


def collect_bundle(
    *,
    plan: Mapping[str, Any],
    controls: Mapping[str, Any],
    training_archive: Path,
    collection_root: Path,
    bundle_root: Path,
    framework_stop_boundary_s: float,
    framework_stop_boundary_source_sha256: str,
) -> dict[str, Any]:
    resolved = timing.validate_plan_with_controls(plan, controls=controls)
    probe = timing.validate_probe_contract(controls["probe_contract"], plan=resolved)
    collector_path = Path(__file__).resolve(strict=True)
    if _file_sha(collector_path) != probe["collector_executable_sha256"]:
        raise ValueError("running collector bytes differ from probe contract")
    docker_path = Path(probe["executable_path"])
    if (
        docker_path.is_symlink()
        or not docker_path.is_file()
        or _file_sha(docker_path) != probe["executable_sha256"]
    ):
        raise ValueError("live Docker executable differs from probe contract")
    for row in probe["mounts"]:
        _real_directory(
            Path(row["source_root"]),
            f"{row['purpose']} mount",
            writable=not row["read_only"],
        )
    if os.path.lexists(collection_root) or os.path.lexists(bundle_root):
        raise FileExistsError("timing collection/bundle roots must be create-only")
    collection_root.mkdir(parents=True, mode=0o700)
    _stage_archive(training_archive, probe=probe)
    collector = timing.seal_collector_identity(
        created_at_utc=_utc_now(),
        collector_executable_sha256=probe["collector_executable_sha256"],
        measurement_tool_sha256=probe["measurement_tool_sha256"],
    )
    collector_path_out = collection_root / "collector-identity.json"
    _publish(collector_path_out, collector)
    receipts = [
        _run_one(
            plan=resolved,
            probe=probe,
            receipt_ordinal=ordinal,
            collection_root=collection_root,
            docker_path=docker_path,
        )
        for ordinal in range(len(probe["receipt_schedule"]))
    ]
    manifest = timing.seal_receipt_manifest(
        created_at_utc=_utc_now(ceiling=True),
        collector_identity=collector,
        collector_identity_file_sha256=_file_sha(collector_path_out),
        receipt_bindings=receipts,
    )
    manifest_path = collection_root / "receipt-manifest.json"
    _publish(manifest_path, manifest)
    bundle = timing.produce_bundle(
        plan=resolved,
        controls=controls,
        receipt_manifest=manifest,
        expected_receipt_manifest_file_sha256=_file_sha(manifest_path),
        expected_receipt_manifest_sha256=manifest["receipt_manifest_sha256"],
        receipt_bindings=receipts,
        framework_stop_boundary_s=framework_stop_boundary_s,
        framework_stop_boundary_source_sha256=framework_stop_boundary_source_sha256,
        output_root=bundle_root,
    )
    binding = timing.bundle_binding(bundle_root)
    timing.replay_bundle(
        bundle_root,
        expected_bundle_file_sha256=binding["bundle_file_sha256"],
        expected_bundle_sha256=binding["bundle_sha256"],
        controls=controls,
        receipt_manifest=manifest,
        expected_receipt_manifest_file_sha256=_file_sha(manifest_path),
        expected_receipt_manifest_sha256=manifest["receipt_manifest_sha256"],
        receipt_bindings=receipts,
    )
    os.chmod(collection_root, 0o500)
    return bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--training-archive", required=True, type=Path)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--framework-stop-boundary-s", required=True, type=float)
    parser.add_argument("--framework-stop-boundary-source-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bundle = collect_bundle(
        plan=_load_canonical(args.plan, "timing plan"),
        controls=_load_canonical(args.controls, "timing controls"),
        training_archive=args.training_archive,
        collection_root=args.collection_root,
        bundle_root=args.bundle_root,
        framework_stop_boundary_s=args.framework_stop_boundary_s,
        framework_stop_boundary_source_sha256=args.framework_stop_boundary_source_sha256,
    )
    print(
        json.dumps(
            {"bundle_sha256": bundle["bundle_sha256"], "status": "PASS"}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
