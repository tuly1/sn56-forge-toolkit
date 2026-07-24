"""Terminal, post-selection construction of the validator's public upload.

Pinned G.O.D uploads ``/app/checkpoints/{task_id}/{repo}`` recursively and does
not filter Forge configuration, SQLite, scope, selection, holdout, or telemetry
sidecars.  This module runs only after the handler/fallback has finished its
selection attempt.  It removes exact private sidecars from the shared checkpoint
volume before copying their already-unlinked bytes into container-local ``/tmp``
for same-process calibration, preserves every model artifact byte-for-byte, and
writes only the strict public flight-recorder projection into the upload root.

Every operation is best effort and this module never raises into the trainer's
never-forfeit exit path.  A failed archive is reported in private telemetry and
the returned audit rather than being mistaken for a successful scrub.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from typing import Any
import uuid

from forge import telemetry
from forge.tasks import checkpoints
from forge.tasks.integrity import valid_safetensors

_PUBLIC_RECORDER = "forge_run.json"
_PRIVATE_SIDECARS = (
    "config.yaml",
    "config.toml",
    "loss_log.db",
    "loss_log.db-wal",
    "loss_log.db-shm",
    "forge_holdout_scores.json",
    "forge_checkpoint_selection.json",
    ".forge_checkpoint_scope.json",
    # Not excluded by the pinned uploader (unlike optimizer.pt).
    "learnable_snr.json",
    # Defense-in-depth for a recorder produced by an older build.
    "forge_run.full.json",
)
_LAST_FILE = "last.safetensors"
_UPLOADER_IGNORED_BASENAMES = frozenset(
    {
        "trainer_state.json",
        "training_args.bin",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
        "rng_state.pth",
    }
)


def finalize_public_bundle(save_root: str) -> dict[str, Any]:
    """Archive private sidecars and publish a hash-bound minimal recorder.

    The selection record and promoted ``last.safetensors`` are attested before
    any selection input is moved.  Scrubbing still proceeds after a failed
    selection attempt because the terminal caller will perform no further
    selection; the failed attestation is explicit in the returned result.
    """
    result: dict[str, Any] = {
        "complete": False,
        "selection_attested": False,
        "artifact_sha256": None,
        "archived": [],
        "removed": [],
        "errors": [],
    }
    try:
        root = os.path.abspath(save_root)
        if not os.path.isdir(root):
            result["errors"].append("upload_root_missing")
            return result
        private_dir = telemetry.ensure_private_bundle_dir(root)
        if private_dir is None:
            result["errors"].append("private_bundle_unavailable")

        public_models = _public_model_allowlist(root)
        attestation = _attest_selection(root)
        result.update(attestation)
        if not result["selection_attested"]:
            result["errors"].append("selection_attestation_failed")

        # The current public projection is safe, but remove it before the final
        # rewrite so an interrupted terminal update cannot be confused with a
        # completed scrub.  Full telemetry has never been written here by this
        # build; the private full record is already in ephemeral /tmp storage.
        _remove_public_slot(os.path.join(root, _PUBLIC_RECORDER), result)
        _remove_public_slot(os.path.join(root, _PUBLIC_RECORDER + ".tmp"), result)

        # Detach every private sidecar from the recursive upload first.  Copying
        # their open descriptors to /tmp happens only after all directory entries
        # are gone, minimizing the residual SIGKILL window to a short unlink loop
        # rather than the duration of several SQLite/config copies.
        detached: list[tuple[str, int]] = []
        for name in _PRIVATE_SIDECARS:
            for candidate in (name, name + ".tmp"):
                fd = _detach_exact(root, candidate, result)
                if fd is not None:
                    detached.append((candidate, fd))
        for name in sorted(os.listdir(root)):
            if not _is_private_recorder_name(name):
                continue
            fd = _detach_exact(root, name, result)
            if fd is not None:
                detached.append((name, fd))
        _scrub_nested_forbidden(root, result)
        _scrub_unexpected_public_entries(root, public_models, result)
        for name, fd in detached:
            _archive_detached(private_dir, name, fd, result)

        remaining = _forbidden_present(root)
        if remaining:
            result["errors"].append(
                "forbidden_sidecars_remain:" + ",".join(sorted(remaining))
            )
        _fsync_directory(root)

        telemetry.event(
            "public_bundle_scrubbed" if not result["errors"] else "public_bundle_scrub_incomplete",
            archived=sorted(result["archived"]),
            errors=list(result["errors"]),
            artifact_sha256=result.get("artifact_sha256"),
        )
        digest = telemetry.write_private(root)
        if not digest:
            result["errors"].append("private_recorder_write_failed")
        elif not telemetry.write_public(root, digest):
            result["errors"].append("public_recorder_write_failed")

        audit_errors = _audit_public_root(
            root,
            expected_private_sha256=digest,
            allowed_model_paths=public_models,
        )
        result["errors"].extend(audit_errors)
        result["errors"] = sorted(set(result["errors"]))
        result["complete"] = not result["errors"]

        # Bind the final audit outcome into the private bytes, then rewrite the
        # public hash once.  This avoids a circular hash while ensuring the
        # published digest commits to the terminal success/failure event.
        telemetry.event(
            "public_bundle_ready" if result["complete"] else "public_bundle_failed",
            errors=list(result["errors"]),
        )
        final_digest = telemetry.write_private(root)
        if final_digest:
            if telemetry.write_public(root, final_digest):
                result["private_record_sha256"] = final_digest
            else:
                result["errors"] = sorted(
                    set(result["errors"] + ["final_public_recorder_write_failed"])
                )
                result["complete"] = False
        else:
            result["errors"] = sorted(
                set(result["errors"] + ["final_private_recorder_write_failed"])
            )
            result["complete"] = False
        if final_digest:
            final_audit = _audit_public_root(
                root,
                expected_private_sha256=final_digest,
                allowed_model_paths=public_models,
            )
            if final_audit:
                result["errors"] = sorted(set(result["errors"] + final_audit))
                result["complete"] = False
        return result
    except BaseException as exc:  # diagnostics/scrub must not cost the model
        result["errors"].append(f"terminal_exception:{type(exc).__name__}")
        try:
            telemetry.event(
                "public_bundle_failed", error=f"{type(exc).__name__}: {exc}"
            )
            digest = telemetry.write_private(save_root)
            if digest:
                telemetry.write_public(save_root, digest)
                result["private_record_sha256"] = digest
        except BaseException:
            pass
        result["errors"] = sorted(set(result["errors"]))
        return result


def private_artifact_path(save_root: str, name: str) -> str:
    """Resolve one exact archived sidecar for calibration tooling/tests."""
    if not name or os.path.basename(name) != name or name in (".", ".."):
        raise ValueError("private artifact name must be one safe path component")
    return os.path.join(telemetry.private_bundle_dir(save_root), name)


def _attest_selection(root: str) -> dict[str, Any]:
    out = {"selection_attested": False, "artifact_sha256": None}
    try:
        state = checkpoints.load_run(root)
        selection = checkpoints.current_selection_record(root, state)
        if (
            selection is None
            or state is None
            or state.get("quarantine_complete") is not True
            or state.get("repo") in (None, "")
        ):
            return out
        declared = str(selection.get("sha256") or "").lower()
        last = os.path.join(root, _LAST_FILE)
        if (
            selection.get("schema") != 1
            or selection.get("status")
            not in {"selected_current_run", "preserved_previous_run"}
            or selection.get("output_file") != _LAST_FILE
            or not re.fullmatch(r"[0-9a-f]{64}", declared)
            or not valid_safetensors(last)
        ):
            return out
        actual = _sha256(last)
        out["artifact_sha256"] = actual
        out["selection_attested"] = actual == declared
        return out
    except Exception:
        return out


def _public_model_allowlist(root: str) -> frozenset[str]:
    """Name only flat, current-attempt candidates plus the promoted artifact."""
    allowed = {_LAST_FILE}
    try:
        absolute_root = os.path.abspath(root)
        state = checkpoints.load_run(absolute_root)
        for path in checkpoints.current_loras(absolute_root, state):
            absolute_path = os.path.abspath(path)
            if os.path.dirname(absolute_path) == absolute_root:
                allowed.add(os.path.basename(absolute_path))
    except Exception:
        pass
    return frozenset(allowed)


def _detach_exact(
    root: str,
    name: str,
    result: dict[str, Any],
) -> int | None:
    source = os.path.join(root, name)
    if not os.path.lexists(source):
        return None
    fd: int | None = None
    try:
        mode = os.lstat(source).st_mode
        if stat.S_ISLNK(mode):
            os.unlink(source)
            result["removed"].append(name)
            _fsync_directory(root)
            return None
        if stat.S_ISDIR(mode):
            # Generated sidecars are files.  A directory at one of these exact
            # private names is stale/malicious; remove the exact tree without
            # copying attacker-controlled nested content into private evidence.
            shutil.rmtree(source)
            result["removed"].append(name)
            _fsync_directory(root)
            return None
        if not stat.S_ISREG(mode):
            os.unlink(source)
            result["removed"].append(name)
            _fsync_directory(root)
            return None

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(source, flags)
        # Confidentiality first: remove the public directory entry and durably
        # commit that removal before copying from the still-open descriptor.
        os.unlink(source)
        result["removed"].append(name)
        _fsync_directory(root)
        detached_fd = fd
        fd = None
        return detached_fd
    except Exception as exc:
        result["errors"].append(f"detach_failed:{name}:{type(exc).__name__}")
        # If opening failed before the unlink, still prefer exact-name removal
        # over retaining a private sidecar in the validator's recursive upload.
        try:
            if os.path.lexists(source):
                if os.path.isdir(source) and not os.path.islink(source):
                    shutil.rmtree(source)
                else:
                    os.unlink(source)
                result["removed"].append(name)
                _fsync_directory(root)
        except Exception as remove_exc:
            result["errors"].append(
                f"remove_failed:{name}:{type(remove_exc).__name__}"
            )
        return None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _archive_detached(
    private_dir: str | None,
    name: str,
    source_fd: int,
    result: dict[str, Any],
) -> None:
    try:
        if private_dir is None:
            return
        destination = os.path.join(private_dir, name)
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        _copy_open_fd(source_fd, destination)
        try:
            result["removed"].remove(name)
        except ValueError:
            pass
        result["archived"].append(name)
    except Exception as exc:
        result["errors"].append(f"archive_failed:{name}:{type(exc).__name__}")
    finally:
        try:
            os.close(source_fd)
        except OSError:
            pass


def _remove_public_slot(path: str, result: dict[str, Any]) -> None:
    if not os.path.lexists(path):
        return
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)
        result["removed"].append(os.path.basename(path))
        _fsync_directory(os.path.dirname(path))
    except Exception as exc:
        result["errors"].append(
            f"remove_failed:{os.path.basename(path)}:{type(exc).__name__}"
        )


def _forbidden_present(root: str) -> list[str]:
    names: list[str] = []
    forbidden = _forbidden_basenames()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in (*dirnames, *filenames):
            public_slot = (
                name == _PUBLIC_RECORDER + ".tmp"
                or (
                    name == _PUBLIC_RECORDER
                    and os.path.abspath(dirpath) != os.path.abspath(root)
                )
            )
            if (
                name in forbidden
                or _is_private_recorder_name(name)
                or public_slot
            ):
                names.append(os.path.relpath(os.path.join(dirpath, name), root))
    return sorted(set(names))


def _scrub_nested_forbidden(root: str, result: dict[str, Any]) -> None:
    """Remove forbidden exact basenames below nested uploader-visible paths."""
    forbidden = _forbidden_basenames()
    for dirpath, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        if os.path.abspath(dirpath) == os.path.abspath(root):
            continue
        for name in filenames:
            if (
                name not in forbidden
                and name not in {_PUBLIC_RECORDER, _PUBLIC_RECORDER + ".tmp"}
                and not _is_private_recorder_name(name)
            ):
                continue
            path = os.path.join(dirpath, name)
            try:
                os.unlink(path)
                result["removed"].append(os.path.relpath(path, root))
            except Exception as exc:
                result["errors"].append(
                    f"nested_remove_failed:{os.path.relpath(path, root)}:"
                    f"{type(exc).__name__}"
                )
        for name in dirnames:
            if (
                name not in forbidden
                and name not in {_PUBLIC_RECORDER, _PUBLIC_RECORDER + ".tmp"}
                and not _is_private_recorder_name(name)
            ):
                continue
            path = os.path.join(dirpath, name)
            try:
                if os.path.islink(path):
                    os.unlink(path)
                else:
                    shutil.rmtree(path)
                result["removed"].append(os.path.relpath(path, root))
            except Exception as exc:
                result["errors"].append(
                    f"nested_remove_failed:{os.path.relpath(path, root)}:"
                    f"{type(exc).__name__}"
                )
    _fsync_directory(root)


def _scrub_unexpected_public_entries(
    root: str,
    allowed_model_paths: frozenset[str],
    result: dict[str, Any],
) -> None:
    """Reduce the recursive upload to its evaluator-safe public allowlist.

    The pinned diffusion evaluator consumes flat safetensors and prefers the
    promoted root ``last.safetensors``.  The pinned uploader can switch to a
    child directory when it finds nested safetensors, so only root-level model
    paths proven eligible by the active checkpoint scope are retained.
    """
    absolute_root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(
        absolute_root, topdown=False, followlinks=False
    ):
        for name in filenames:
            path = os.path.join(dirpath, name)
            relative = os.path.relpath(path, absolute_root)
            try:
                mode = os.lstat(path).st_mode
                allowed = (
                    relative == _PUBLIC_RECORDER
                    or (
                        name in _UPLOADER_IGNORED_BASENAMES
                        and stat.S_ISREG(mode)
                    )
                    or (
                        relative in allowed_model_paths
                        and stat.S_ISREG(mode)
                        and valid_safetensors(path)
                    )
                )
                if allowed and not stat.S_ISLNK(mode):
                    continue
                os.unlink(path)
                result["removed"].append(relative)
            except Exception as exc:
                result["errors"].append(
                    f"unexpected_remove_failed:{relative}:{type(exc).__name__}"
                )
        for name in dirnames:
            path = os.path.join(dirpath, name)
            relative = os.path.relpath(path, absolute_root)
            try:
                mode = os.lstat(path).st_mode
                if stat.S_ISLNK(mode):
                    os.unlink(path)
                    result["removed"].append(relative)
                    continue
                if stat.S_ISDIR(mode):
                    try:
                        os.rmdir(path)
                        result["removed"].append(relative)
                    except OSError:
                        # A non-empty directory is retained only as a container
                        # for recursively allowlisted safetensors/ignored state.
                        pass
                    continue
                os.unlink(path)
                result["removed"].append(relative)
            except FileNotFoundError:
                pass
            except Exception as exc:
                result["errors"].append(
                    f"unexpected_remove_failed:{relative}:{type(exc).__name__}"
                )
    _fsync_directory(absolute_root)


def _forbidden_basenames() -> set[str]:
    return {
        candidate
        for name in _PRIVATE_SIDECARS
        for candidate in (name, name + ".tmp")
    }


def _is_private_recorder_name(name: str) -> bool:
    return bool(
        name == "forge_run.full.json"
        or re.fullmatch(r"forge_run\.full\.[0-9a-f]{64}\.json", name)
    )


def _copy_open_fd(source_fd: int, destination: str) -> None:
    parent = os.path.dirname(destination)
    os.makedirs(parent, mode=0o700, exist_ok=True)
    temp = os.path.join(
        parent, f".{os.path.basename(destination)}.{uuid.uuid4().hex}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    target_fd: int | None = None
    try:
        target_fd = os.open(temp, flags, 0o600)
        os.lseek(source_fd, 0, os.SEEK_SET)
        while True:
            block = os.read(source_fd, 1024 * 1024)
            if not block:
                break
            view = memoryview(block)
            while view:
                written = os.write(target_fd, view)
                view = view[written:]
        os.fsync(target_fd)
        os.close(target_fd)
        target_fd = None
        os.replace(temp, destination)
        _fsync_directory(parent)
    except BaseException:
        if target_fd is not None:
            try:
                os.close(target_fd)
            except OSError:
                pass
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


def _audit_public_root(
    root: str,
    *,
    expected_private_sha256: str | None,
    allowed_model_paths: frozenset[str],
) -> list[str]:
    errors: list[str] = []
    remaining = _forbidden_present(root)
    if remaining:
        errors.append("forbidden_sidecars_remain:" + ",".join(sorted(remaining)))
    recorder = os.path.join(root, _PUBLIC_RECORDER)
    try:
        with open(recorder, "rb") as fh:
            raw = fh.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("public recorder exceeds 1 MiB")
        data = json.loads(raw)
        if (
            set(data) != {"schema", "kind", "private_record_sha256", "events"}
            or data.get("schema") != 2
            or data.get("kind") != "forge-public-run-recorder"
            or data.get("private_record_sha256") != expected_private_sha256
            or not isinstance(data.get("events"), list)
        ):
            raise ValueError("public recorder schema mismatch")
        for event in data["events"]:
            if not isinstance(event, dict) or not set(event).issubset(
                {"t", "name", "failure_class"}
            ) or not {"t", "name"}.issubset(event):
                raise ValueError("public event schema mismatch")
    except Exception as exc:
        errors.append(f"public_recorder_invalid:{type(exc).__name__}")

    try:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            for name in dirnames:
                path = os.path.join(dirpath, name)
                if os.path.islink(path):
                    errors.append(
                        "unexpected_public_entry:" + os.path.relpath(path, root)
                    )
            for name in filenames:
                path = os.path.join(dirpath, name)
                relative = os.path.relpath(path, root)
                mode = os.lstat(path).st_mode
                allowed = (
                    relative == _PUBLIC_RECORDER
                    or (
                        name in _UPLOADER_IGNORED_BASENAMES
                        and stat.S_ISREG(mode)
                    )
                    or (
                        relative in allowed_model_paths
                        and stat.S_ISREG(mode)
                        and valid_safetensors(path)
                    )
                )
                if not allowed or stat.S_ISLNK(mode):
                    errors.append(f"unexpected_public_entry:{relative}")
    except Exception as exc:
        errors.append(f"public_content_scan_failed:{type(exc).__name__}")
    return errors


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(fd)
        except OSError:
            # Publication is already best effort and must never discard a valid
            # model merely because a mounted filesystem rejects directory fsync.
            pass
    finally:
        os.close(fd)
