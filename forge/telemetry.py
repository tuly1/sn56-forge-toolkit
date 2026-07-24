"""Private flight recorder plus a minimal, hash-bound public projection.

The validator uploads the exact checkpoint output directory.  Detailed recipe,
environment, curve, and selection telemetry is useful for local calibration but
also discloses the trainer's strategy when written there.  Full telemetry is
therefore checkpointed only in a nonce-scoped, container-local ``/tmp``
directory which is neither uploaded nor backed by G.O.D's shared checkpoint
volume.  The upload root receives a strict allowlist projection containing
event classes, relative timings, failure classes, and the SHA-256 of the exact
private record bytes.

Design rules: never raise (a diagnostic must not cost a run), never put the full
record below the upload root, construct the public schema by allowlist rather
than subtraction, and keep both writes atomic and durable on a best-effort basis.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit
import uuid

_FILENAME = "forge_run.json"
_PRIVATE_FILENAME = "forge_run.full.json"
_PRIVATE_RECORD_NAME_RE = re.compile(
    r"forge_run\.full(?:\.[0-9a-f]{64})?\.json"
)
_PRIVATE_ROOT = "/tmp/forge-private"
_RUN_NONCE = uuid.uuid4().hex
_BOUND_RUN_KEYS: dict[str, str] = {}
_LATEST_PRIVATE_RECORDS: dict[str, str] = {}
_MAX_EVENTS = 200
_MAX_CURVE_POINTS = 300
_MAX_SAMPLES = 120
_SENSITIVE_KEYS = {
    "authorization",
    "cookie",
    "credential",
    "credentials",
    "password",
    "passwd",
    "secret",
    "session",
    "signature",
    "token",
}
_SENSITIVE_COLLAPSED_KEYS = {
    "apikey",
    "accesskey",
    "authkey",
    "privatekey",
    "secretkey",
    "sessionid",
    "sessionkey",
    "sessiontoken",
}
_BEARER_RE = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]+")
_AUTH_HEADER_RE = re.compile(
    r"(?i)\b((?:proxy-)?authorization\s*:\s*)"
    r"(?:(?:basic|bearer|digest)\s+)?[^\s,;]+"
)
_COOKIE_HEADER_RE = re.compile(r"(?im)^(\s*(?:set-)?cookie\s*:)[^\r\n]*")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b("
    r"api[_-]?key|access[_-]?key|auth[_-]?key|cookie|"
    r"aws[_-]?secret[_-]?access[_-]?key|client[_-]?secret|credentials?|"
    r"password|passwd|private[_-]?key|secret(?:[_-]?key)?|"
    r"(?:access|refresh|id)[_-]?token|"
    r"session(?:[_-]?(?:id|key|token))?|signature|token"
    r")\s*([:=])\s*([^\s,;&]+)"
)
_KNOWN_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"github_pat_[A-Za-z0-9_]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"hf_[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}"
    r")(?![A-Za-z0-9])"
)
_URL_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s'\"<>]+")

_t0 = time.monotonic()
_data: dict[str, Any] = {
    "schema": 1,
    "meta": {},
    "env": {},
    "events": [],
    "train_curve": [],  # (rel_s, step, loss, lr)
    "eval_curve": [],  # (rel_s, step, eval_loss)
    "samples": {},  # name -> [(rel_s, value)]
}


def start_run(**meta: Any) -> None:
    """Reset process-global telemetry for one CLI invocation, then initialize it."""
    global _data, _t0, _RUN_NONCE, _BOUND_RUN_KEYS, _LATEST_PRIVATE_RECORDS
    try:
        _t0 = time.monotonic()
        _RUN_NONCE = uuid.uuid4().hex
        _BOUND_RUN_KEYS = {}
        _LATEST_PRIVATE_RECORDS = {}
        _data = {
            "schema": 1,
            "meta": {},
            "env": {},
            "events": [],
            "train_curve": [],
            "eval_curve": [],
            "samples": {},
        }
        init(**meta)
    except Exception:
        pass


def _rel() -> float:
    return round(time.monotonic() - _t0, 1)


def init(**meta: Any) -> None:
    try:
        _data["meta"].update(
            {k: _sanitize_value(v, key=k) for k, v in meta.items() if v is not None}
        )
        _data["meta"].setdefault("started_unix", int(time.time()))
    except Exception:
        pass


def set_meta(**kv: Any) -> None:
    init(**kv)


def event(name: str, **kv: Any) -> None:
    try:
        if len(_data["events"]) >= _MAX_EVENTS:
            return
        safe = {k: _sanitize_value(v, key=k) for k, v in kv.items()}
        _data["events"].append({"t": _rel(), "name": name, **safe})
    except Exception:
        pass


def sample(name: str, value: float) -> None:
    try:
        series = _data["samples"].setdefault(name, [])
        if len(series) < _MAX_SAMPLES:
            series.append((_rel(), round(float(value), 6)))
    except Exception:
        pass


def train_point(step: int, loss: float | None, lr: float | None) -> None:
    try:
        curve = _data["train_curve"]
        curve.append((_rel(), int(step), loss, lr))
        if len(curve) > _MAX_CURVE_POINTS:
            # Thin by dropping every other point from the older half.
            half = len(curve) // 2
            _data["train_curve"] = curve[:half:2] + curve[half:]
    except Exception:
        pass


def eval_point(step: int, eval_loss: float) -> None:
    try:
        _data["eval_curve"].append((_rel(), int(step), round(float(eval_loss), 6)))
    except Exception:
        pass


def collect_env() -> None:
    """Versions + hardware; called from handlers once heavy imports exist. Every
    import is optional (the kohya base image's exact package set varies), so a
    missing library just means a thinner env record, never a crash.
    """
    env: dict[str, Any] = {}
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except Exception:
        pass
    for lib in ("diffusers", "transformers", "peft", "accelerate"):
        try:
            env[lib] = __import__(lib).__version__
        except Exception:
            pass
    try:
        _data["env"].update(env)
    except Exception:
        pass


def note_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            _data["env"]["peak_cuda_alloc_gb"] = round(
                torch.cuda.max_memory_allocated() / 1e9, 2
            )
    except Exception:
        pass


def write_into(output_dir: str) -> None:
    """Checkpoint the private record and refresh its safe public projection.

    This compatibility entry point is used by callbacks during training.  It is
    intentionally safe to call before the terminal post-selection scrub: the
    only file it writes below ``output_dir`` is the strict public projection.
    """
    try:
        if not os.path.isdir(output_dir):
            return
        digest = write_private(output_dir) or private_record_sha256()
        if digest:
            write_public(output_dir, digest)
    except Exception:
        pass


def write_private(output_dir: str) -> str | None:
    """Atomically persist canonical full telemetry outside the upload root.

    Returns the SHA-256 of the exact bytes written, or ``None`` on any failure.
    The returned digest is suitable for the public recorder's binding field.
    """
    try:
        if not os.path.isdir(output_dir):
            return None
        _data["meta"]["last_write_rel_s"] = _rel()
        payload = _canonical_private_bytes()
        private_dir = _secure_private_bundle_dir(output_dir)
        if private_dir is None:
            return None
        digest = hashlib.sha256(payload).hexdigest()
        path = os.path.join(private_dir, f"forge_run.full.{digest}.json")
        if not os.path.isfile(path):
            _atomic_bytes(path, payload)
        elif _sha256_file(path) != digest:
            return None
        _LATEST_PRIVATE_RECORDS[os.path.abspath(output_dir)] = path
        return digest
    except Exception:
        return None


def write_public(output_dir: str, private_sha256: str) -> bool:
    """Atomically write the strict public recorder into the upload root."""
    try:
        if not os.path.isdir(output_dir) or not re.fullmatch(
            r"[0-9a-f]{64}", str(private_sha256)
        ):
            return False
        payload = json.dumps(
            public_record(private_sha256),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        _atomic_bytes(os.path.join(output_dir, _FILENAME), payload)
        return True
    except Exception:
        return False


def bind_private_bundle(output_dir: str, attempt_nonce: str) -> None:
    """Bind future private writes for one upload root to its checkpoint attempt."""
    try:
        if not isinstance(attempt_nonce, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]{8,128}", attempt_nonce
        ):
            return
        _BOUND_RUN_KEYS[os.path.abspath(output_dir)] = attempt_nonce
    except Exception:
        pass


def write_public_snapshot(output_dir: str) -> bool:
    """Replace stale public telemetry without touching any prior private record."""
    try:
        digest = private_record_sha256()
        return bool(digest and write_public(output_dir, digest))
    except Exception:
        return False


def prepare_public_recorder(output_dir: str) -> bool:
    """Remove stale/full recorder slots, then install a safe initial projection."""
    try:
        if not os.path.isdir(output_dir):
            return False
        names = {_FILENAME, _FILENAME + ".tmp", _PRIVATE_FILENAME}
        names.update(
            name
            for name in os.listdir(output_dir)
            if _PRIVATE_RECORD_NAME_RE.fullmatch(name)
        )
        for name in names:
            path = os.path.join(output_dir, name)
            if not os.path.lexists(path):
                continue
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.unlink(path)
        _fsync_directory(output_dir)
        return write_public_snapshot(output_dir)
    except Exception:
        return False


def private_record_sha256() -> str | None:
    """Hash the current in-memory private record without persisting it."""
    try:
        return hashlib.sha256(_canonical_private_bytes()).hexdigest()
    except Exception:
        return None


def public_record(private_sha256: str) -> dict[str, Any]:
    """Build the public recorder exclusively from explicitly allowed fields."""
    events: list[dict[str, Any]] = []
    for raw in _data.get("events", []):
        if not isinstance(raw, dict):
            continue
        name = _public_event_name(raw.get("name"))
        timing = raw.get("t")
        if name is None or isinstance(timing, bool) or not isinstance(
            timing, (int, float)
        ) or not math.isfinite(float(timing)):
            continue
        item: dict[str, Any] = {"t": round(float(timing), 1), "name": name}
        failure_class = _public_failure_class(raw)
        if failure_class is not None:
            item["failure_class"] = failure_class
        events.append(item)
    return {
        "schema": 2,
        "kind": "forge-public-run-recorder",
        "private_record_sha256": str(private_sha256),
        "events": events,
    }


def private_bundle_dir(output_dir: str) -> str:
    """Return this process-run's container-local ephemeral private bundle."""
    absolute = os.path.abspath(output_dir)
    run_key = _BOUND_RUN_KEYS.get(absolute, _RUN_NONCE)
    suffix = hashlib.sha256(
        f"{absolute}\0{run_key}".encode("utf-8")
    ).hexdigest()[:24]
    return os.path.join(_PRIVATE_ROOT, suffix)


def private_record_path(output_dir: str) -> str:
    """Return the latest immutable private record path for this run."""
    path = _LATEST_PRIVATE_RECORDS.get(os.path.abspath(output_dir))
    if path:
        return path
    return os.path.join(private_bundle_dir(output_dir), _PRIVATE_FILENAME)


def private_record_path_for_digest(output_dir: str, digest: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
        raise ValueError("private record digest must be lowercase SHA-256")
    return os.path.join(
        private_bundle_dir(output_dir), f"forge_run.full.{digest}.json"
    )


def ensure_private_bundle_dir(output_dir: str) -> str | None:
    """Return a securely validated ephemeral private directory, if available."""
    return _secure_private_bundle_dir(output_dir)


def _canonical_private_bytes() -> bytes:
    return json.dumps(
        _data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _public_event_name(value: Any) -> str | None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", value):
        return None
    return value


def _public_failure_class(event_value: dict[str, Any]) -> str | None:
    """Extract only an exception *type*, never a message or reason string."""
    raw = event_value.get("error")
    if not isinstance(raw, str):
        return None
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_.]{0,95})(?::|\()", raw)
    return match.group(1) if match else None


def _atomic_bytes(path: str, payload: bytes) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = os.path.join(
        parent, f".{os.path.basename(path)}.{uuid.uuid4().hex}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(tmp, flags, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_directory(parent)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _is_descendant(path: str, directory: str) -> bool:
    try:
        return os.path.commonpath(
            (os.path.abspath(path), os.path.abspath(directory))
        ) == os.path.abspath(directory)
    except (OSError, ValueError):
        return True


def _secure_private_bundle_dir(output_dir: str) -> str | None:
    """Create/validate the private directory without following symlink leaves."""
    try:
        root = os.path.abspath(_PRIVATE_ROOT)
        if os.path.lexists(root) and os.path.islink(root):
            return None
        os.makedirs(root, mode=0o700, exist_ok=True)
        os.chmod(root, 0o700)
        bundle = os.path.abspath(private_bundle_dir(output_dir))
        if os.path.commonpath((bundle, root)) != root:
            return None
        if os.path.lexists(bundle):
            if os.path.islink(bundle) or not os.path.isdir(bundle):
                return None
        else:
            os.mkdir(bundle, mode=0o700)
        os.chmod(bundle, 0o700)
        real_bundle = os.path.realpath(bundle)
        real_root = os.path.realpath(root)
        real_output = os.path.realpath(output_dir)
        if (
            os.path.commonpath((real_bundle, real_root)) != real_root
            or _is_descendant(real_bundle, real_output)
        ):
            return None
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(bundle, flags)
        try:
            stat_fd = os.fstat(fd)
            stat_path = os.lstat(bundle)
            if (stat_fd.st_dev, stat_fd.st_ino) != (
                stat_path.st_dev,
                stat_path.st_ino,
            ):
                return None
        finally:
            os.close(fd)
        return bundle
    except Exception:
        return None


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: str) -> None:
    """Best-effort durability for the atomic telemetry rename."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sanitize_value(value: Any, *, key: str = "") -> Any:
    """Redact credentials before public run telemetry is persisted."""
    key_parts = [
        part for part in re.split(r"[^a-z0-9]+", key.lower()) if part
    ]
    collapsed_key = "".join(key_parts)
    if any(part in _SENSITIVE_KEYS for part in key_parts) or any(
        marker in collapsed_key for marker in _SENSITIVE_COLLAPSED_KEYS
    ):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): _sanitize_value(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v, key=key) for v in value]
    if not isinstance(value, str):
        return value

    text = _COOKIE_HEADER_RE.sub(r"\1<redacted>", value)
    text = _AUTH_HEADER_RE.sub(r"\1<redacted>", text)
    text = _BEARER_RE.sub(r"\1<redacted>", text)

    def _strip_url(match: re.Match[str]) -> str:
        raw = match.group(0)
        suffix = ""
        while raw and raw[-1] in ".,);]":
            suffix = raw[-1] + suffix
            raw = raw[:-1]
        try:
            parts = urlsplit(raw)
            netloc = parts.netloc
            if parts.username is not None or parts.password is not None:
                host = parts.hostname or ""
                if ":" in host and not host.startswith("["):
                    host = f"[{host}]"
                netloc = host
                if parts.port is not None:
                    netloc += f":{parts.port}"
            clean = urlunsplit((parts.scheme, netloc, parts.path, "", ""))
            return clean + suffix
        except Exception:
            return "<redacted-url>" + suffix

    text = _URL_RE.sub(_strip_url, text)
    text = _KNOWN_TOKEN_RE.sub("<redacted-token>", text)
    return _SECRET_ASSIGNMENT_RE.sub(r"\1\2<redacted>", text)


def make_trainer_callback(output_dir: str):
    """A TrainerCallback that records curves and periodically persists the log.

    Imported lazily so this module stays usable without transformers.
    """
    from transformers import TrainerCallback

    class TelemetryCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            logs = logs or {}
            if "loss" in logs:
                train_point(state.global_step, logs.get("loss"), logs.get("learning_rate"))
            return control

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: ANN001
            loss = (metrics or {}).get("eval_loss")
            if loss is not None:
                eval_point(state.global_step, loss)
            write_into(output_dir)
            return control

        def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001
            event("train_end", steps=int(state.global_step))
            note_peak_memory()
            write_into(output_dir)
            return control

    return TelemetryCallback()
