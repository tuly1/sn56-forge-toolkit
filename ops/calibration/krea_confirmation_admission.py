"""Stage-2 confirmation admission with a sealed-root read barrier.

Public governance is validated first.  Ratification and reveal authorization
may resolve the opaque sealed-root locator only after that validation; neither
operation reads a file below it.  The post-freeze inventory materialization
and final content materialization are the only operations that call the sealed
file reader.  The former emits byte identities only, never fixture payloads.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

try:
    from . import krea_fixture
    from . import krea_stage2_boundary_derivation
    from . import krea_stage2_delegated_review_contract
    from . import krea_stage2_execution_surface_policy
    from . import krea_stage2_legacy_confirmation
    from . import krea_stage2_production_identity
except ImportError:  # pragma: no cover - direct script execution.
    import krea_fixture  # type: ignore[no-redef]
    import krea_stage2_boundary_derivation  # type: ignore[no-redef]
    import krea_stage2_delegated_review_contract  # type: ignore[no-redef]
    import krea_stage2_execution_surface_policy  # type: ignore[no-redef]
    import krea_stage2_legacy_confirmation  # type: ignore[no-redef]
    import krea_stage2_production_identity  # type: ignore[no-redef]


REQUEST_KIND = "forge-krea-stage2-confirmation-admission-request"
RATIFICATION_KIND = "forge-krea-stage2-owner-ratification"
REVEAL_KIND = "forge-krea-stage2-confirmation-reveal-authorization"
MATERIALIZATION_KIND = "forge-krea-stage2-confirmation-materialization"
GPU_AUTHORIZATION_KIND = "forge-krea-stage2-gpu-execution-authorization"
POSTFREEZE_INVENTORY_KIND = "forge-krea-stage2-postfreeze-sealed-inventory"
SCHEMA = 1
OWNER_IDENTITY = "Atulya Shetty"
_CONFIRMATION_ROLES = ("C1", "C2", "C3", "C4")
_BOUNDARY_ROLES = krea_stage2_execution_surface_policy.BOUNDARY_ROLES
_ALL_ROLES = _CONFIRMATION_ROLES + _BOUNDARY_ROLES
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def canonical_file_sha256(value: Any) -> str:
    """Digest the canonical create-only representation of a record."""

    return hashlib.sha256(canonical_bytes(value) + b"\n").hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = keys - set(value)
    extra = set(value) - keys
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value
    ):
        raise ValueError(f"{label} must be canonical whole-second UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{label} is not real UTC") from exc
    if parsed < datetime(2020, 1, 1, tzinfo=timezone.utc) or parsed > datetime.now(
        timezone.utc
    ) + timedelta(seconds=60):
        raise ValueError(f"{label} is outside accepted evidence time bounds")
    return value


def _utc_value(value: Any, label: str) -> datetime:
    return datetime.strptime(_utc(value, label), "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a normalized relative path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or str(pure) != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    return value


def _reject_symlink_ancestors(
    path: Path, label: str, *, include_leaf: bool = True
) -> None:
    current = path if include_leaf else path.parent
    while True:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink component: {current}")
        if current == current.parent:
            return
        current = current.parent


def _role_digests(value: Any, *, roles: Sequence[str], label: str) -> dict[str, str]:
    mapping = _object(value, label)
    if set(mapping) != set(roles):
        raise ValueError(f"{label} must cover exactly {','.join(roles)}")
    return {role: _digest(mapping[role], f"{label}.{role}") for role in roles}


def _sealed_files(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("sealed_files must be a non-empty manifest")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        row = _object(raw, f"sealed_files[{index}]")
        _exact(row, {"role", "relative_path", "sha256", "bytes"}, "sealed file")
        role = row["role"]
        if role not in _ALL_ROLES:
            raise ValueError("sealed file has an unapproved fixture role")
        size = row["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("sealed file size must be a positive integer")
        rows.append(
            {
                "role": role,
                "relative_path": _relative_path(
                    row["relative_path"], "sealed file relative_path"
                ),
                "sha256": _digest(row["sha256"], "sealed file SHA-256"),
                "bytes": size,
            }
        )
    if any(row["relative_path"] == "materialization.json" for row in rows):
        raise ValueError("sealed_files uses the reserved materialization record path")
    order = [(row["role"], row["relative_path"]) for row in rows]
    if order != sorted(order) or len(order) != len(set(order)):
        raise ValueError("sealed_files must be sorted and unique by role/path")
    if {row["role"] for row in rows} != set(_ALL_ROLES):
        raise ValueError("sealed_files must cover every C1-C4 and boundary role")
    if len({row["relative_path"] for row in rows}) != len(rows):
        raise ValueError("sealed file relative paths must be globally unique")
    relative_parts = [PurePosixPath(row["relative_path"]).parts for row in rows]
    if any(
        left != right and len(left) < len(right) and right[: len(left)] == left
        for left in relative_parts
        for right in relative_parts
    ):
        raise ValueError("sealed file paths contain a file/directory collision")
    return rows


def sealed_root_locator_sha256(value: str | Path) -> str:
    """Commit to a locator string without resolving or reading that location."""

    raw = str(value)
    if not raw or raw != raw.strip():
        raise ValueError("sealed root locator must be non-empty canonical text")
    path = Path(os.path.abspath(os.path.expanduser(raw)))
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()


def _resolve_sealed_root(value: str | Path) -> Path:
    """Resolve a sealed directory without enumerating or opening its content."""

    raw = str(value)
    if not raw or raw != raw.strip():
        raise ValueError("sealed root must be non-empty canonical text")
    candidate = Path(os.path.abspath(os.path.expanduser(raw)))
    current = candidate
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"sealed root has a symlink component: {current}")
        current = current.parent
    resolved = candidate.resolve(strict=True)
    if resolved != candidate or not resolved.is_dir():
        raise ValueError("sealed root must be a real normalized directory")
    return resolved


def _check_root_locator(root: Path, expected: str) -> None:
    if sealed_root_locator_sha256(root) != expected:
        raise ValueError("sealed root differs from its ratified locator commitment")


def build_request(
    *,
    production_identity: Mapping[str, Any],
    production_identity_file_sha256: str,
    waiver_freeze_sha256: str,
    waiver_freeze_file_sha256: str,
    public_commitment_sha256s: Mapping[str, str],
    boundary_fixture_manifest_sha256s: Mapping[str, str],
    sealed_inventory_sha256: str,
    sealed_inventory_file_sha256: str,
    sealed_root_locator_sha256: str,
    sealed_files: Sequence[Mapping[str, Any]],
    prepared_at_utc: str,
) -> dict[str, Any]:
    identity = krea_stage2_production_identity.validate(dict(production_identity))
    expected_identity_file_sha256 = hashlib.sha256(
        krea_stage2_production_identity.canonical_bytes(identity) + b"\n"
    ).hexdigest()
    if production_identity_file_sha256 != expected_identity_file_sha256:
        raise ValueError("production identity file SHA-256 does not bind its record")
    contract = krea_stage2_delegated_review_contract.binding()
    policy = krea_stage2_execution_surface_policy.validate(
        krea_stage2_execution_surface_policy.POLICY
    )
    commitments = _role_digests(
        dict(public_commitment_sha256s),
        roles=_CONFIRMATION_ROLES,
        label="public commitment hashes",
    )
    boundary = _role_digests(
        dict(boundary_fixture_manifest_sha256s),
        roles=_BOUNDARY_ROLES,
        label="boundary fixture manifest hashes",
    )
    files = _sealed_files([dict(row) for row in sealed_files])
    prepared = _utc(prepared_at_utc, "request preparation time")
    if _utc_value(prepared, "request preparation time") <= _utc_value(
        identity["captured_at_utc"], "production identity capture time"
    ):
        raise ValueError("admission request must follow production identity capture")
    body = {
        "schema": SCHEMA,
        "kind": REQUEST_KIND,
        "prepared_at_utc": prepared,
        "policy_sha256": policy["policy_sha256"],
        "delegated_review_contract_sha256": contract["contract_sha256"],
        "delegated_review_contract_file_sha256": contract["file_sha256"],
        "production_identity_sha256": identity["production_identity_sha256"],
        "production_identity_file_sha256": _digest(
            production_identity_file_sha256, "production identity file SHA-256"
        ),
        "image_id": identity["container_image"]["image_id"],
        "waiver_freeze_sha256": _digest(
            waiver_freeze_sha256, "waiver freeze semantic SHA-256"
        ),
        "waiver_freeze_file_sha256": _digest(
            waiver_freeze_file_sha256, "waiver freeze file SHA-256"
        ),
        "public_commitment_sha256s": commitments,
        "boundary_fixture_manifest_sha256s": boundary,
        "boundary_matrix_sha256": canonical_sha256(boundary),
        "sealed_inventory_sha256": _digest(
            sealed_inventory_sha256, "sealed inventory semantic SHA-256"
        ),
        "sealed_inventory_file_sha256": _digest(
            sealed_inventory_file_sha256, "sealed inventory file SHA-256"
        ),
        "sealed_root_locator_sha256": _digest(
            sealed_root_locator_sha256, "sealed root locator SHA-256"
        ),
        "sealed_files": files,
        "sealed_file_set_sha256": canonical_sha256(files),
        "admission_authorized": False,
        "gpu_execution_authorized": False,
    }
    request = {**body, "request_sha256": canonical_sha256(body)}
    return validate_request(
        request,
        production_identity=identity,
        production_identity_file_sha256=production_identity_file_sha256,
    )


def validate_request(
    value: Any,
    *,
    production_identity: Mapping[str, Any],
    production_identity_file_sha256: str,
) -> dict[str, Any]:
    request = _object(value, "Stage-2 admission request")
    _exact(
        request,
        {
            "schema",
            "kind",
            "prepared_at_utc",
            "policy_sha256",
            "delegated_review_contract_sha256",
            "delegated_review_contract_file_sha256",
            "production_identity_sha256",
            "production_identity_file_sha256",
            "image_id",
            "waiver_freeze_sha256",
            "waiver_freeze_file_sha256",
            "public_commitment_sha256s",
            "boundary_fixture_manifest_sha256s",
            "boundary_matrix_sha256",
            "sealed_inventory_sha256",
            "sealed_inventory_file_sha256",
            "sealed_root_locator_sha256",
            "sealed_files",
            "sealed_file_set_sha256",
            "admission_authorized",
            "gpu_execution_authorized",
            "request_sha256",
        },
        "Stage-2 admission request",
    )
    identity = krea_stage2_production_identity.validate(dict(production_identity))
    expected_identity_file_sha256 = hashlib.sha256(
        krea_stage2_production_identity.canonical_bytes(identity) + b"\n"
    ).hexdigest()
    if production_identity_file_sha256 != expected_identity_file_sha256:
        raise ValueError("production identity file SHA-256 does not bind its record")
    policy = krea_stage2_execution_surface_policy.validate(
        krea_stage2_execution_surface_policy.POLICY
    )
    contract = krea_stage2_delegated_review_contract.binding()
    commitments = _role_digests(
        request["public_commitment_sha256s"],
        roles=_CONFIRMATION_ROLES,
        label="public commitment hashes",
    )
    boundary = _role_digests(
        request["boundary_fixture_manifest_sha256s"],
        roles=_BOUNDARY_ROLES,
        label="boundary fixture manifest hashes",
    )
    files = _sealed_files(request["sealed_files"])
    for key in (
        "production_identity_file_sha256",
        "waiver_freeze_sha256",
        "waiver_freeze_file_sha256",
        "boundary_matrix_sha256",
        "sealed_inventory_sha256",
        "sealed_inventory_file_sha256",
        "sealed_root_locator_sha256",
        "sealed_file_set_sha256",
        "request_sha256",
    ):
        _digest(request[key], key)
    if not isinstance(request["image_id"], str) or not _IMAGE_ID.fullmatch(
        request["image_id"]
    ):
        raise ValueError("request image_id is not immutable")
    body = {key: item for key, item in request.items() if key != "request_sha256"}
    if (
        request["schema"] != SCHEMA
        or request["kind"] != REQUEST_KIND
        or request["policy_sha256"] != policy["policy_sha256"]
        or request["delegated_review_contract_sha256"] != contract["contract_sha256"]
        or request["delegated_review_contract_file_sha256"] != contract["file_sha256"]
        or request["production_identity_sha256"]
        != identity["production_identity_sha256"]
        or request["production_identity_file_sha256"]
        != _digest(production_identity_file_sha256, "production identity file")
        or request["image_id"] != identity["container_image"]["image_id"]
        or request["public_commitment_sha256s"] != commitments
        or request["boundary_fixture_manifest_sha256s"] != boundary
        or request["boundary_matrix_sha256"] != canonical_sha256(boundary)
        or request["sealed_files"] != files
        or request["sealed_file_set_sha256"] != canonical_sha256(files)
        or request["admission_authorized"] is not False
        or request["gpu_execution_authorized"] is not False
        or request["request_sha256"] != canonical_sha256(body)
    ):
        raise ValueError("Stage-2 admission request drifted")
    if _utc_value(request["prepared_at_utc"], "request preparation time") <= _utc_value(
        identity["captured_at_utc"], "production identity capture time"
    ):
        raise ValueError("admission request must follow production identity capture")
    return request


def _ratification_acknowledgements() -> dict[str, bool]:
    return {
        "fresh_stage2_ratification": True,
        "production_identity_reviewed": True,
        "public_commitments_and_boundary_matrix_reviewed": True,
        "waiver_freeze_reviewed": True,
        "post_freeze_custodian_hash_copy_and_inventory_reviewed": True,
        "finalist_selection_actors_did_not_read_fixture_content_before_freeze": True,
        "ratification_operation_did_not_read_fixture_content": True,
        "gpu_execution_requires_separate_authorization": True,
    }


def build_ratification(
    request: Mapping[str, Any],
    *,
    production_identity: Mapping[str, Any],
    production_identity_file_sha256: str,
    owner_identity: str,
    ratified_at_utc: str,
) -> dict[str, Any]:
    request_value = validate_request(
        dict(request),
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
    )
    if owner_identity != OWNER_IDENTITY:
        raise ValueError("Stage-2 requires fresh ratification by the named owner")
    ratified = _utc(ratified_at_utc, "ratification time")
    if _utc_value(ratified, "ratification time") <= _utc_value(
        request_value["prepared_at_utc"], "request preparation time"
    ):
        raise ValueError("Stage-2 ratification must follow request preparation")
    body = {
        "schema": SCHEMA,
        "kind": RATIFICATION_KIND,
        "ratified_at_utc": ratified,
        "accountable_owner_identity": owner_identity,
        "owner_identity_assurance": (
            "interactive-owner-self-attestation-not-cryptographic-or-legal-signature"
        ),
        "request_sha256": request_value["request_sha256"],
        "policy_sha256": request_value["policy_sha256"],
        "delegated_review_contract_sha256": request_value[
            "delegated_review_contract_sha256"
        ],
        "delegated_review_contract_file_sha256": request_value[
            "delegated_review_contract_file_sha256"
        ],
        "production_identity_sha256": request_value["production_identity_sha256"],
        "production_identity_file_sha256": request_value[
            "production_identity_file_sha256"
        ],
        "image_id": request_value["image_id"],
        "waiver_freeze_sha256": request_value["waiver_freeze_sha256"],
        "waiver_freeze_file_sha256": request_value["waiver_freeze_file_sha256"],
        "public_commitment_sha256s": request_value["public_commitment_sha256s"],
        "boundary_matrix_sha256": request_value["boundary_matrix_sha256"],
        "sealed_inventory_sha256": request_value["sealed_inventory_sha256"],
        "sealed_inventory_file_sha256": request_value[
            "sealed_inventory_file_sha256"
        ],
        "sealed_root_locator_sha256": request_value["sealed_root_locator_sha256"],
        "acknowledgements": _ratification_acknowledgements(),
        "admission_authorized": False,
        "gpu_execution_authorized": False,
    }
    return {**body, "ratification_sha256": canonical_sha256(body)}


def validate_ratification(
    value: Any,
    *,
    request: Mapping[str, Any],
    production_identity: Mapping[str, Any],
    production_identity_file_sha256: str,
) -> dict[str, Any]:
    ratification = _object(value, "Stage-2 owner ratification")
    expected = build_ratification(
        request,
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
        owner_identity=ratification.get("accountable_owner_identity"),
        ratified_at_utc=ratification.get("ratified_at_utc"),
    )
    if ratification != expected:
        raise ValueError("Stage-2 owner ratification drifted")
    return ratification


def ratify(
    request: Mapping[str, Any],
    *,
    production_identity: Mapping[str, Any],
    production_identity_file_sha256: str,
    sealed_root: str | Path,
    owner_identity: str,
    ratified_at_utc: str,
) -> dict[str, Any]:
    """Validate all public inputs before resolving (but never reading) the root."""

    ratification = build_ratification(
        request,
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
        owner_identity=owner_identity,
        ratified_at_utc=ratified_at_utc,
    )
    validate_ratification(
        ratification,
        request=request,
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
    )
    # Deliberately last: no invalid public record can probe the sealed root.
    root = _resolve_sealed_root(sealed_root)
    _check_root_locator(root, ratification["sealed_root_locator_sha256"])
    return ratification


def build_reveal_authorization(
    request: Mapping[str, Any],
    ratification: Mapping[str, Any],
    *,
    ratification_file_sha256: str,
    production_identity: Mapping[str, Any],
    production_identity_file_sha256: str,
    actor: Mapping[str, Any],
    revealed_at_utc: str,
) -> dict[str, Any]:
    request_value = validate_request(
        dict(request),
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
    )
    ratification_value = validate_ratification(
        dict(ratification),
        request=request_value,
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
    )
    if ratification_file_sha256 != canonical_file_sha256(ratification_value):
        raise ValueError("ratification file SHA-256 does not bind its record")
    reviewer = krea_stage2_delegated_review_contract.validate_actor(
        "confirmation_reveal_reviewer", dict(actor)
    )
    revealed = _utc(revealed_at_utc, "reveal authorization time")
    if _utc_value(revealed, "reveal authorization time") <= _utc_value(
        ratification_value["ratified_at_utc"], "ratification time"
    ):
        raise ValueError("reveal authorization must follow owner ratification")
    body = {
        "schema": SCHEMA,
        "kind": REVEAL_KIND,
        "revealed_at_utc": revealed,
        "actor": reviewer,
        "request_sha256": request_value["request_sha256"],
        "ratification_sha256": ratification_value["ratification_sha256"],
        "ratification_file_sha256": _digest(
            ratification_file_sha256, "ratification file SHA-256"
        ),
        "policy_sha256": request_value["policy_sha256"],
        "delegated_review_contract_sha256": request_value[
            "delegated_review_contract_sha256"
        ],
        "delegated_review_contract_file_sha256": request_value[
            "delegated_review_contract_file_sha256"
        ],
        "production_identity_sha256": request_value["production_identity_sha256"],
        "production_identity_file_sha256": request_value[
            "production_identity_file_sha256"
        ],
        "image_id": request_value["image_id"],
        "waiver_freeze_sha256": request_value["waiver_freeze_sha256"],
        "waiver_freeze_file_sha256": request_value["waiver_freeze_file_sha256"],
        "public_commitment_sha256s": request_value["public_commitment_sha256s"],
        "boundary_matrix_sha256": request_value["boundary_matrix_sha256"],
        "sealed_inventory_sha256": request_value["sealed_inventory_sha256"],
        "sealed_inventory_file_sha256": request_value[
            "sealed_inventory_file_sha256"
        ],
        "sealed_root_locator_sha256": request_value["sealed_root_locator_sha256"],
        "sealed_content_read": False,
        "reveal_authorized": True,
        "admission_authorized": False,
        "gpu_execution_authorized": False,
    }
    return {**body, "reveal_sha256": canonical_sha256(body)}


def validate_reveal_authorization(
    value: Any,
    *,
    request: Mapping[str, Any],
    ratification: Mapping[str, Any],
    ratification_file_sha256: str,
    production_identity: Mapping[str, Any],
    production_identity_file_sha256: str,
) -> dict[str, Any]:
    reveal = _object(value, "Stage-2 reveal authorization")
    expected = build_reveal_authorization(
        request,
        ratification,
        ratification_file_sha256=ratification_file_sha256,
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
        actor=reveal.get("actor"),
        revealed_at_utc=reveal.get("revealed_at_utc"),
    )
    if reveal != expected:
        raise ValueError("Stage-2 reveal authorization drifted")
    return reveal


def authorize_reveal(
    request: Mapping[str, Any],
    ratification: Mapping[str, Any],
    *,
    ratification_file_sha256: str,
    production_identity: Mapping[str, Any],
    production_identity_file_sha256: str,
    sealed_root: str | Path,
    actor: Mapping[str, Any],
    revealed_at_utc: str,
) -> dict[str, Any]:
    """Validate request/ratification/actor before resolving the sealed root."""

    reveal = build_reveal_authorization(
        request,
        ratification,
        ratification_file_sha256=ratification_file_sha256,
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
        actor=actor,
        revealed_at_utc=revealed_at_utc,
    )
    validate_reveal_authorization(
        reveal,
        request=request,
        ratification=ratification,
        ratification_file_sha256=ratification_file_sha256,
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
    )
    # Deliberately last, and still no content read.
    root = _resolve_sealed_root(sealed_root)
    _check_root_locator(root, reveal["sealed_root_locator_sha256"])
    return reveal


def _read_sealed_file(root: Path, relative: str) -> bytes:
    """Sealed-content primitive used only by materialization operations."""

    path = root.joinpath(*PurePosixPath(relative).parts)
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError(f"sealed file has a symlink component: {relative}")
        current = current.parent
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"sealed file must be regular and non-symlink: {relative}")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(after):
        raise RuntimeError(f"sealed file changed while read: {relative}")
    return raw


def _postfreeze_inventory_body(
    *,
    public_commitment_sha256s: Mapping[str, str],
    confirmation_wrapper_file_sha256s: Mapping[str, str],
    boundary_fixture_manifest_sha256s: Mapping[str, str],
    boundary_fixture_manifest_file_sha256s: Mapping[str, str],
    actor: Mapping[str, Any],
    captured_at_utc: str,
    sealed_root_locator_sha256_value: str,
    files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    commitments = _role_digests(
        dict(public_commitment_sha256s),
        roles=_CONFIRMATION_ROLES,
        label="public commitment hashes",
    )
    wrappers = _role_digests(
        dict(confirmation_wrapper_file_sha256s),
        roles=_CONFIRMATION_ROLES,
        label="confirmation wrapper file hashes",
    )
    boundary_semantic = _role_digests(
        dict(boundary_fixture_manifest_sha256s),
        roles=_BOUNDARY_ROLES,
        label="boundary fixture manifest hashes",
    )
    boundary_files = _role_digests(
        dict(boundary_fixture_manifest_file_sha256s),
        roles=_BOUNDARY_ROLES,
        label="boundary fixture manifest file hashes",
    )
    materializer = krea_stage2_delegated_review_contract.validate_actor(
        "confirmation_materialization_reviewer", dict(actor)
    )
    rows = _sealed_files([dict(row) for row in files])
    return {
        "schema": SCHEMA,
        "kind": POSTFREEZE_INVENTORY_KIND,
        "captured_at_utc": _utc(captured_at_utc, "post-freeze inventory capture time"),
        "public_freeze_binding": {
            "path": krea_stage2_boundary_derivation.FREEZE_BINDING_PATH,
            "file_sha256": (krea_stage2_boundary_derivation.FREEZE_BINDING_FILE_SHA256),
            "binding_sha256": (krea_stage2_boundary_derivation.FREEZE_BINDING_SHA256),
            "commit_sha1": krea_stage2_boundary_derivation.FREEZE_BINDING_COMMIT,
            "remote_reachable": True,
        },
        "actor": materializer,
        "sealed_root_locator_sha256": _digest(
            sealed_root_locator_sha256_value, "sealed root locator SHA-256"
        ),
        "public_commitment_sha256s": commitments,
        "confirmation_wrapper_file_sha256s": wrappers,
        "boundary_fixture_manifest_sha256s": boundary_semantic,
        "boundary_fixture_manifest_file_sha256s": boundary_files,
        "files": rows,
        "file_set_sha256": canonical_sha256(rows),
        "role_file_counts": {
            role: sum(row["role"] == role for row in rows) for role in _ALL_ROLES
        },
        "fixture_payload_bytes_emitted": False,
        "path_and_digest_metadata_only": True,
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": (
            "post-finalist-freeze-private-inventory-materialization-only;contains-"
            "paths-sizes-and-digests-but-no-image-caption-or-manifest-payloads;not-"
            "admission-GPU-release-deployment-or-competitiveness-authority"
        ),
    }


def validate_postfreeze_inventory(value: Any) -> dict[str, Any]:
    record = _object(value, "post-freeze sealed inventory")
    _exact(
        record,
        {
            "schema",
            "kind",
            "captured_at_utc",
            "public_freeze_binding",
            "actor",
            "sealed_root_locator_sha256",
            "public_commitment_sha256s",
            "confirmation_wrapper_file_sha256s",
            "boundary_fixture_manifest_sha256s",
            "boundary_fixture_manifest_file_sha256s",
            "files",
            "file_set_sha256",
            "role_file_counts",
            "fixture_payload_bytes_emitted",
            "path_and_digest_metadata_only",
            "admission_authorized",
            "gpu_execution_authorized",
            "claim_limit",
            "inventory_sha256",
        },
        "post-freeze sealed inventory",
    )
    body = {key: item for key, item in record.items() if key != "inventory_sha256"}
    expected = _postfreeze_inventory_body(
        public_commitment_sha256s=record["public_commitment_sha256s"],
        confirmation_wrapper_file_sha256s=record[
            "confirmation_wrapper_file_sha256s"
        ],
        boundary_fixture_manifest_sha256s=record["boundary_fixture_manifest_sha256s"],
        boundary_fixture_manifest_file_sha256s=record[
            "boundary_fixture_manifest_file_sha256s"
        ],
        actor=record["actor"],
        captured_at_utc=record["captured_at_utc"],
        sealed_root_locator_sha256_value=record["sealed_root_locator_sha256"],
        files=record["files"],
    )
    if body != expected or record["inventory_sha256"] != canonical_sha256(body):
        raise ValueError("post-freeze sealed inventory drifted")
    return record


def materialize_postfreeze_inventory(
    *,
    public_freeze_binding_path: str | Path,
    remote_reachable_commit_sha1: str,
    public_commitment_sha256s: Mapping[str, str],
    confirmation_wrapper_file_sha256s: Mapping[str, str],
    boundary_fixture_manifest_sha256s: Mapping[str, str],
    boundary_fixture_manifest_file_sha256s: Mapping[str, str],
    sealed_root: str | Path,
    output_path: str | Path,
    actor: Mapping[str, Any],
    captured_at_utc: str,
) -> dict[str, Any]:
    """Materialize the exact inventory after the pushed finalist freeze.

    Public/freeze/actor inputs fail before root resolution.  The output is the
    private path/size/digest inventory needed by ``build_request``; no fixture
    file payload is copied or emitted.
    """

    materializer = krea_stage2_delegated_review_contract.validate_actor(
        "confirmation_materialization_reviewer", dict(actor)
    )
    freeze_path = Path(public_freeze_binding_path)
    if freeze_path.is_symlink() or not freeze_path.is_file():
        raise ValueError("public finalist-freeze binding is not a regular file")
    freeze_raw = freeze_path.read_bytes()
    try:
        freeze = json.loads(freeze_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("public finalist-freeze binding is not JSON") from exc
    freeze_file_sha = hashlib.sha256(freeze_raw).hexdigest()
    if freeze_raw != canonical_bytes(freeze) + b"\n":
        raise ValueError("public finalist-freeze binding is not canonical JSON")
    krea_stage2_boundary_derivation.validate_public_freeze_binding(
        freeze, file_sha256=freeze_file_sha
    )
    if (
        remote_reachable_commit_sha1
        != krea_stage2_boundary_derivation.FREEZE_BINDING_COMMIT
    ):
        raise ValueError("pushed finalist-freeze commit was not observed remotely")
    if _utc_value(captured_at_utc, "post-freeze inventory capture time") <= _utc_value(
        freeze["binding_created_at_utc"], "public finalist-freeze binding time"
    ):
        raise ValueError("sealed inventory must follow the public finalist freeze")
    commitments = _role_digests(
        dict(public_commitment_sha256s),
        roles=_CONFIRMATION_ROLES,
        label="public commitment hashes",
    )
    wrappers = _role_digests(
        dict(confirmation_wrapper_file_sha256s),
        roles=_CONFIRMATION_ROLES,
        label="confirmation wrapper file hashes",
    )
    boundary_semantic = _role_digests(
        dict(boundary_fixture_manifest_sha256s),
        roles=_BOUNDARY_ROLES,
        label="boundary fixture manifest hashes",
    )
    boundary_files = _role_digests(
        dict(boundary_fixture_manifest_file_sha256s),
        roles=_BOUNDARY_ROLES,
        label="boundary fixture manifest file hashes",
    )

    # First sealed-root interaction.  This is the inventory phase of
    # materialization and reads bytes only to hash them.
    root = _resolve_sealed_root(sealed_root)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(
                f"sealed root contains a symlink: {path.relative_to(root)}"
            )
        mode = path.stat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(
                f"sealed root contains a special node: {path.relative_to(root)}"
            )
        relative = path.relative_to(root).as_posix()
        parts = PurePosixPath(relative).parts
        if not parts or parts[0] not in _ALL_ROLES:
            raise ValueError("sealed root has a file outside an exact fixture role")
        raw = _read_sealed_file(root, relative)
        rows.append(
            {
                "role": parts[0],
                "relative_path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        )
    rows.sort(key=lambda row: (row["role"], row["relative_path"]))
    rows = _sealed_files(rows)
    expected_manifest_files = {**wrappers, **boundary_files}
    for role, expected_file_sha in expected_manifest_files.items():
        matches = [
            row
            for row in rows
            if row["role"] == role and row["sha256"] == expected_file_sha
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{role} committed manifest is absent from sealed inventory"
            )
        manifest_raw = _read_sealed_file(root, matches[0]["relative_path"])
        try:
            manifest = json.loads(manifest_raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{role} committed manifest is not JSON") from exc
        if manifest_raw != canonical_bytes(manifest) + b"\n":
            raise ValueError(f"{role} committed manifest is not canonical JSON")
        if role in _CONFIRMATION_ROLES:
            wrapper = krea_stage2_legacy_confirmation.validate_wrapper_file(
                wrapper=manifest, role_root=root / role
            )
            if (
                wrapper["experimental_role"] != role
                or wrapper["published_checksum_manifest"]["file_sha256"]
                != commitments[role]
            ):
                raise ValueError(f"{role} legacy wrapper commitment drifted")
        else:
            manifest = krea_fixture.validate_manifest(manifest)
            if manifest["experimental_role"] != role:
                raise ValueError(f"{role} committed manifest role drifted")
            if manifest["manifest_sha256"] != boundary_semantic[role]:
                raise ValueError(f"{role} boundary semantic manifest binding drifted")

    body = _postfreeze_inventory_body(
        public_commitment_sha256s=commitments,
        confirmation_wrapper_file_sha256s=wrappers,
        boundary_fixture_manifest_sha256s=boundary_semantic,
        boundary_fixture_manifest_file_sha256s=boundary_files,
        actor=materializer,
        captured_at_utc=captured_at_utc,
        sealed_root_locator_sha256_value=sealed_root_locator_sha256(root),
        files=rows,
    )
    record = {**body, "inventory_sha256": canonical_sha256(body)}
    validate_postfreeze_inventory(record)
    output = Path(output_path)
    _reject_symlink_ancestors(output, "post-freeze inventory output")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(canonical_bytes(record) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def _validate_sealed_tree(root: Path, expected: Sequence[Mapping[str, Any]]) -> None:
    """Reject uncommitted files, symlinks, and special nodes during materialization."""

    observed: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(
                f"sealed root contains a symlink: {path.relative_to(root)}"
            )
        if path.is_dir():
            continue
        if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError(
                f"sealed root contains a special node: {path.relative_to(root)}"
            )
        observed.append(path.relative_to(root).as_posix())
    committed = sorted(row["relative_path"] for row in expected)
    if observed != committed:
        raise ValueError("sealed root file set differs from its public commitment")


def _materialization_body(
    *,
    request: Mapping[str, Any],
    request_file_sha256: str,
    ratification: Mapping[str, Any],
    ratification_file_sha256: str,
    reveal: Mapping[str, Any],
    reveal_file_sha256: str,
    actor: Mapping[str, Any],
    materialized_at_utc: str,
) -> dict[str, Any]:
    if request_file_sha256 != canonical_file_sha256(request):
        raise ValueError("request file SHA-256 does not bind its record")
    if ratification_file_sha256 != canonical_file_sha256(ratification):
        raise ValueError("ratification file SHA-256 does not bind its record")
    if reveal_file_sha256 != canonical_file_sha256(reveal):
        raise ValueError("reveal file SHA-256 does not bind its record")
    materialized = _utc(materialized_at_utc, "materialization time")
    if _utc_value(materialized, "materialization time") <= _utc_value(
        reveal["revealed_at_utc"], "reveal authorization time"
    ):
        raise ValueError("materialization must follow reveal authorization")
    return {
        "schema": SCHEMA,
        "kind": MATERIALIZATION_KIND,
        "materialized_at_utc": materialized,
        "actor": actor,
        "request_sha256": request["request_sha256"],
        "request_file_sha256": _digest(request_file_sha256, "request file SHA-256"),
        "ratification_sha256": ratification["ratification_sha256"],
        "ratification_file_sha256": _digest(
            ratification_file_sha256, "ratification file SHA-256"
        ),
        "reveal_sha256": reveal["reveal_sha256"],
        "reveal_file_sha256": _digest(reveal_file_sha256, "reveal file SHA-256"),
        "policy_sha256": request["policy_sha256"],
        "delegated_review_contract_sha256": request["delegated_review_contract_sha256"],
        "delegated_review_contract_file_sha256": request[
            "delegated_review_contract_file_sha256"
        ],
        "production_identity_sha256": request["production_identity_sha256"],
        "production_identity_file_sha256": request["production_identity_file_sha256"],
        "image_id": request["image_id"],
        "waiver_freeze_sha256": request["waiver_freeze_sha256"],
        "waiver_freeze_file_sha256": request["waiver_freeze_file_sha256"],
        "public_commitment_sha256s": request["public_commitment_sha256s"],
        "boundary_matrix_sha256": request["boundary_matrix_sha256"],
        "sealed_inventory_sha256": request["sealed_inventory_sha256"],
        "sealed_inventory_file_sha256": request["sealed_inventory_file_sha256"],
        "sealed_root_locator_sha256": request["sealed_root_locator_sha256"],
        "files": request["sealed_files"],
        "file_set_sha256": request["sealed_file_set_sha256"],
        "admission_authorized": True,
        "gpu_execution_authorized": False,
    }


def materialize(
    request: Mapping[str, Any],
    ratification: Mapping[str, Any],
    reveal: Mapping[str, Any],
    *,
    request_file_sha256: str,
    ratification_file_sha256: str,
    reveal_file_sha256: str,
    production_identity: Mapping[str, Any],
    production_identity_file_sha256: str,
    sealed_root: str | Path,
    output_dir: str | Path,
    actor: Mapping[str, Any],
    materialized_at_utc: str,
) -> dict[str, Any]:
    """Validate every public gate, then read and copy only committed files."""

    # This entire block is intentionally before root resolution or any output
    # mutation.  Tests can replace both access primitives to enforce ordering.
    request_value = validate_request(
        dict(request),
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
    )
    ratification_value = validate_ratification(
        dict(ratification),
        request=request_value,
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
    )
    reveal_value = validate_reveal_authorization(
        dict(reveal),
        request=request_value,
        ratification=ratification_value,
        ratification_file_sha256=ratification_file_sha256,
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
    )
    materializer = krea_stage2_delegated_review_contract.validate_actor(
        "confirmation_materialization_reviewer", dict(actor)
    )
    body = _materialization_body(
        request=request_value,
        request_file_sha256=request_file_sha256,
        ratification=ratification_value,
        ratification_file_sha256=ratification_file_sha256,
        reveal=reveal_value,
        reveal_file_sha256=reveal_file_sha256,
        actor=materializer,
        materialized_at_utc=materialized_at_utc,
    )

    # First sealed-root interaction.  Only reads below this point are allowed.
    root = _resolve_sealed_root(sealed_root)
    _check_root_locator(root, request_value["sealed_root_locator_sha256"])
    _validate_sealed_tree(root, request_value["sealed_files"])
    output = Path(os.path.abspath(os.path.expanduser(output_dir)))
    _reject_symlink_ancestors(output, "materialization output")
    if os.path.lexists(output):
        raise FileExistsError(f"materialization output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(output, "materialization output")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for row in request_value["sealed_files"]:
            raw = _read_sealed_file(root, row["relative_path"])
            if (
                len(raw) != row["bytes"]
                or hashlib.sha256(raw).hexdigest() != row["sha256"]
            ):
                raise ValueError(
                    f"sealed file differs from commitment: {row['relative_path']}"
                )
            target = temporary.joinpath(*PurePosixPath(row["relative_path"]).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        record = {**body, "materialization_sha256": canonical_sha256(body)}
        validate_materialization(
            record,
            request=request_value,
            request_file_sha256=request_file_sha256,
            ratification=ratification_value,
            ratification_file_sha256=ratification_file_sha256,
            reveal=reveal_value,
            reveal_file_sha256=reveal_file_sha256,
            production_identity=production_identity,
            production_identity_file_sha256=production_identity_file_sha256,
        )
        with (temporary / "materialization.json").open("xb") as handle:
            handle.write(canonical_bytes(record) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return record


def validate_materialization(
    value: Any,
    *,
    request: Mapping[str, Any],
    request_file_sha256: str,
    ratification: Mapping[str, Any],
    ratification_file_sha256: str,
    reveal: Mapping[str, Any],
    reveal_file_sha256: str,
    production_identity: Mapping[str, Any],
    production_identity_file_sha256: str,
) -> dict[str, Any]:
    """Validate a materialization record without touching materialized content."""

    request_value = validate_request(
        dict(request),
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
    )
    ratification_value = validate_ratification(
        dict(ratification),
        request=request_value,
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
    )
    reveal_value = validate_reveal_authorization(
        dict(reveal),
        request=request_value,
        ratification=ratification_value,
        ratification_file_sha256=ratification_file_sha256,
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
    )
    record = _object(value, "Stage-2 confirmation materialization")
    materializer = krea_stage2_delegated_review_contract.validate_actor(
        "confirmation_materialization_reviewer", record.get("actor")
    )
    expected_body = _materialization_body(
        request=request_value,
        request_file_sha256=request_file_sha256,
        ratification=ratification_value,
        ratification_file_sha256=ratification_file_sha256,
        reveal=reveal_value,
        reveal_file_sha256=reveal_file_sha256,
        actor=materializer,
        materialized_at_utc=record.get("materialized_at_utc"),
    )
    expected = {
        **expected_body,
        "materialization_sha256": canonical_sha256(expected_body),
    }
    if record != expected:
        raise ValueError("Stage-2 confirmation materialization drifted")
    return record


def _gpu_authorization_acknowledgements() -> dict[str, bool]:
    return {
        "exact_admission_chain_reviewed": True,
        "exact_production_identity_and_image_reviewed": True,
        "waiver_freeze_and_public_commitments_reviewed": True,
        "authorization_is_gpu_execution_only": True,
        "production_mutation_and_release_remain_forbidden": True,
    }


def build_gpu_execution_authorization(
    request: Mapping[str, Any],
    ratification: Mapping[str, Any],
    reveal: Mapping[str, Any],
    materialization: Mapping[str, Any],
    *,
    request_file_sha256: str,
    ratification_file_sha256: str,
    reveal_file_sha256: str,
    materialization_file_sha256: str,
    production_identity: Mapping[str, Any],
    production_identity_file_sha256: str,
    owner_identity: str,
    authorized_at_utc: str,
) -> dict[str, Any]:
    """Build the distinct named-owner GPU authority after materialization."""

    request_value = validate_request(
        dict(request),
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
    )
    ratification_value = validate_ratification(
        dict(ratification),
        request=request_value,
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
    )
    reveal_value = validate_reveal_authorization(
        dict(reveal),
        request=request_value,
        ratification=ratification_value,
        ratification_file_sha256=ratification_file_sha256,
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
    )
    materialization_value = validate_materialization(
        dict(materialization),
        request=request_value,
        request_file_sha256=request_file_sha256,
        ratification=ratification_value,
        ratification_file_sha256=ratification_file_sha256,
        reveal=reveal_value,
        reveal_file_sha256=reveal_file_sha256,
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
    )
    if materialization_file_sha256 != canonical_file_sha256(materialization_value):
        raise ValueError("materialization file SHA-256 does not bind its record")
    if owner_identity != OWNER_IDENTITY:
        raise ValueError("only the named owner may authorize Stage-2 GPU execution")
    authorized = _utc(authorized_at_utc, "GPU authorization time")
    if _utc_value(authorized, "GPU authorization time") <= _utc_value(
        materialization_value["materialized_at_utc"], "materialization time"
    ):
        raise ValueError("GPU authorization must follow materialization")
    body = {
        "schema": SCHEMA,
        "kind": GPU_AUTHORIZATION_KIND,
        "authorized_at_utc": authorized,
        "accountable_owner_identity": owner_identity,
        "owner_identity_assurance": (
            "interactive-owner-self-attestation-not-cryptographic-or-legal-signature"
        ),
        "request_sha256": request_value["request_sha256"],
        "request_file_sha256": _digest(request_file_sha256, "request file SHA-256"),
        "ratification_sha256": ratification_value["ratification_sha256"],
        "ratification_file_sha256": _digest(
            ratification_file_sha256, "ratification file SHA-256"
        ),
        "reveal_sha256": reveal_value["reveal_sha256"],
        "reveal_file_sha256": _digest(reveal_file_sha256, "reveal file SHA-256"),
        "materialization_sha256": materialization_value["materialization_sha256"],
        "materialization_file_sha256": _digest(
            materialization_file_sha256, "materialization file SHA-256"
        ),
        "policy_sha256": request_value["policy_sha256"],
        "delegated_review_contract_sha256": request_value[
            "delegated_review_contract_sha256"
        ],
        "delegated_review_contract_file_sha256": request_value[
            "delegated_review_contract_file_sha256"
        ],
        "production_identity_sha256": request_value["production_identity_sha256"],
        "production_identity_file_sha256": request_value[
            "production_identity_file_sha256"
        ],
        "image_id": request_value["image_id"],
        "waiver_freeze_sha256": request_value["waiver_freeze_sha256"],
        "waiver_freeze_file_sha256": request_value["waiver_freeze_file_sha256"],
        "public_commitment_sha256s": request_value["public_commitment_sha256s"],
        "boundary_matrix_sha256": request_value["boundary_matrix_sha256"],
        "sealed_inventory_sha256": request_value["sealed_inventory_sha256"],
        "sealed_inventory_file_sha256": request_value[
            "sealed_inventory_file_sha256"
        ],
        "acknowledgements": _gpu_authorization_acknowledgements(),
        "admission_authorized": True,
        "gpu_execution_authorized": True,
        "production_mutation_authorized": False,
        "release_authorized": False,
    }
    return {
        **body,
        "gpu_execution_authorization_sha256": canonical_sha256(body),
    }


def validate_gpu_execution_authorization(
    value: Any,
    *,
    request: Mapping[str, Any],
    ratification: Mapping[str, Any],
    reveal: Mapping[str, Any],
    materialization: Mapping[str, Any],
    request_file_sha256: str,
    ratification_file_sha256: str,
    reveal_file_sha256: str,
    materialization_file_sha256: str,
    production_identity: Mapping[str, Any],
    production_identity_file_sha256: str,
) -> dict[str, Any]:
    authorization = _object(value, "Stage-2 GPU execution authorization")
    expected = build_gpu_execution_authorization(
        request,
        ratification,
        reveal,
        materialization,
        request_file_sha256=request_file_sha256,
        ratification_file_sha256=ratification_file_sha256,
        reveal_file_sha256=reveal_file_sha256,
        materialization_file_sha256=materialization_file_sha256,
        production_identity=production_identity,
        production_identity_file_sha256=production_identity_file_sha256,
        owner_identity=authorization.get("accountable_owner_identity"),
        authorized_at_utc=authorization.get("authorized_at_utc"),
    )
    if authorization != expected:
        raise ValueError("Stage-2 GPU execution authorization drifted")
    return authorization


_SEMANTIC_KEYS = {
    REQUEST_KIND: "request_sha256",
    RATIFICATION_KIND: "ratification_sha256",
    REVEAL_KIND: "reveal_sha256",
    MATERIALIZATION_KIND: "materialization_sha256",
    GPU_AUTHORIZATION_KIND: "gpu_execution_authorization_sha256",
}


def _validate_standalone(value: Any) -> dict[str, Any]:
    record = _object(value, "Stage-2 admission record")
    kind = record.get("kind")
    semantic_key = _SEMANTIC_KEYS.get(kind)
    if semantic_key is None or record.get("schema") != SCHEMA:
        raise ValueError("unsupported Stage-2 admission record")
    _digest(record.get(semantic_key), semantic_key)
    body = {key: item for key, item in record.items() if key != semantic_key}
    if record[semantic_key] != canonical_sha256(body):
        raise ValueError("Stage-2 admission record digest mismatch")
    if kind == REQUEST_KIND:
        _exact(
            record,
            {
                "schema",
                "kind",
                "prepared_at_utc",
                "policy_sha256",
                "delegated_review_contract_sha256",
                "delegated_review_contract_file_sha256",
                "production_identity_sha256",
                "production_identity_file_sha256",
                "image_id",
                "waiver_freeze_sha256",
                "waiver_freeze_file_sha256",
                "public_commitment_sha256s",
                "boundary_fixture_manifest_sha256s",
                "boundary_matrix_sha256",
                "sealed_inventory_sha256",
                "sealed_inventory_file_sha256",
                "sealed_root_locator_sha256",
                "sealed_files",
                "sealed_file_set_sha256",
                "admission_authorized",
                "gpu_execution_authorized",
                "request_sha256",
            },
            "Stage-2 admission request",
        )
        policy = krea_stage2_execution_surface_policy.validate(
            krea_stage2_execution_surface_policy.POLICY
        )
        contract = krea_stage2_delegated_review_contract.binding()
        boundary = _role_digests(
            record["boundary_fixture_manifest_sha256s"],
            roles=_BOUNDARY_ROLES,
            label="boundary fixture manifest hashes",
        )
        files = _sealed_files(record["sealed_files"])
        if (
            record["policy_sha256"] != policy["policy_sha256"]
            or record["delegated_review_contract_sha256"] != contract["contract_sha256"]
            or record["delegated_review_contract_file_sha256"]
            != contract["file_sha256"]
            or record["boundary_matrix_sha256"] != canonical_sha256(boundary)
            or record["sealed_file_set_sha256"] != canonical_sha256(files)
            or record["admission_authorized"] is not False
            or record["gpu_execution_authorized"] is not False
        ):
            raise ValueError("standalone Stage-2 admission request drifted")
        _utc(record["prepared_at_utc"], "request preparation time")
        _role_digests(
            record["public_commitment_sha256s"],
            roles=_CONFIRMATION_ROLES,
            label="public commitment hashes",
        )
    elif kind == RATIFICATION_KIND:
        _exact(
            record,
            {
                "schema",
                "kind",
                "ratified_at_utc",
                "accountable_owner_identity",
                "owner_identity_assurance",
                "request_sha256",
                "policy_sha256",
                "delegated_review_contract_sha256",
                "delegated_review_contract_file_sha256",
                "production_identity_sha256",
                "production_identity_file_sha256",
                "image_id",
                "waiver_freeze_sha256",
                "waiver_freeze_file_sha256",
                "public_commitment_sha256s",
                "boundary_matrix_sha256",
                "sealed_inventory_sha256",
                "sealed_inventory_file_sha256",
                "sealed_root_locator_sha256",
                "acknowledgements",
                "admission_authorized",
                "gpu_execution_authorized",
                "ratification_sha256",
            },
            "Stage-2 owner ratification",
        )
        if (
            record["accountable_owner_identity"] != OWNER_IDENTITY
            or record["acknowledgements"] != _ratification_acknowledgements()
            or record["admission_authorized"] is not False
            or record["gpu_execution_authorized"] is not False
        ):
            raise ValueError("standalone Stage-2 ratification drifted")
        _utc(record["ratified_at_utc"], "ratification time")
        _role_digests(
            record["public_commitment_sha256s"],
            roles=_CONFIRMATION_ROLES,
            label="public commitment hashes",
        )
    elif kind == REVEAL_KIND:
        _exact(
            record,
            {
                "schema",
                "kind",
                "revealed_at_utc",
                "actor",
                "request_sha256",
                "ratification_sha256",
                "ratification_file_sha256",
                "policy_sha256",
                "delegated_review_contract_sha256",
                "delegated_review_contract_file_sha256",
                "production_identity_sha256",
                "production_identity_file_sha256",
                "image_id",
                "waiver_freeze_sha256",
                "waiver_freeze_file_sha256",
                "public_commitment_sha256s",
                "boundary_matrix_sha256",
                "sealed_inventory_sha256",
                "sealed_inventory_file_sha256",
                "sealed_root_locator_sha256",
                "sealed_content_read",
                "reveal_authorized",
                "admission_authorized",
                "gpu_execution_authorized",
                "reveal_sha256",
            },
            "Stage-2 reveal authorization",
        )
        krea_stage2_delegated_review_contract.validate_actor(
            "confirmation_reveal_reviewer", record["actor"]
        )
        if (
            record["sealed_content_read"] is not False
            or record["reveal_authorized"] is not True
            or record["admission_authorized"] is not False
            or record["gpu_execution_authorized"] is not False
        ):
            raise ValueError("standalone Stage-2 reveal authorization drifted")
        _utc(record["revealed_at_utc"], "reveal authorization time")
        _role_digests(
            record["public_commitment_sha256s"],
            roles=_CONFIRMATION_ROLES,
            label="public commitment hashes",
        )
    elif kind == MATERIALIZATION_KIND:
        _exact(
            record,
            {
                "schema",
                "kind",
                "materialized_at_utc",
                "actor",
                "request_sha256",
                "request_file_sha256",
                "ratification_sha256",
                "ratification_file_sha256",
                "reveal_sha256",
                "reveal_file_sha256",
                "policy_sha256",
                "delegated_review_contract_sha256",
                "delegated_review_contract_file_sha256",
                "production_identity_sha256",
                "production_identity_file_sha256",
                "image_id",
                "waiver_freeze_sha256",
                "waiver_freeze_file_sha256",
                "public_commitment_sha256s",
                "boundary_matrix_sha256",
                "sealed_inventory_sha256",
                "sealed_inventory_file_sha256",
                "sealed_root_locator_sha256",
                "files",
                "file_set_sha256",
                "admission_authorized",
                "gpu_execution_authorized",
                "materialization_sha256",
            },
            "Stage-2 materialization",
        )
        krea_stage2_delegated_review_contract.validate_actor(
            "confirmation_materialization_reviewer", record["actor"]
        )
        files = _sealed_files(record["files"])
        if (
            record["file_set_sha256"] != canonical_sha256(files)
            or record["admission_authorized"] is not True
            or record["gpu_execution_authorized"] is not False
        ):
            raise ValueError("standalone Stage-2 materialization drifted")
        _utc(record["materialized_at_utc"], "materialization time")
        _role_digests(
            record["public_commitment_sha256s"],
            roles=_CONFIRMATION_ROLES,
            label="public commitment hashes",
        )
    elif kind == GPU_AUTHORIZATION_KIND:
        _exact(
            record,
            {
                "schema",
                "kind",
                "authorized_at_utc",
                "accountable_owner_identity",
                "owner_identity_assurance",
                "request_sha256",
                "request_file_sha256",
                "ratification_sha256",
                "ratification_file_sha256",
                "reveal_sha256",
                "reveal_file_sha256",
                "materialization_sha256",
                "materialization_file_sha256",
                "policy_sha256",
                "delegated_review_contract_sha256",
                "delegated_review_contract_file_sha256",
                "production_identity_sha256",
                "production_identity_file_sha256",
                "image_id",
                "waiver_freeze_sha256",
                "waiver_freeze_file_sha256",
                "public_commitment_sha256s",
                "boundary_matrix_sha256",
                "sealed_inventory_sha256",
                "sealed_inventory_file_sha256",
                "acknowledgements",
                "admission_authorized",
                "gpu_execution_authorized",
                "production_mutation_authorized",
                "release_authorized",
                "gpu_execution_authorization_sha256",
            },
            "Stage-2 GPU execution authorization",
        )
        if (
            record["accountable_owner_identity"] != OWNER_IDENTITY
            or record["acknowledgements"] != _gpu_authorization_acknowledgements()
            or record["admission_authorized"] is not True
            or record["gpu_execution_authorized"] is not True
            or record["production_mutation_authorized"] is not False
            or record["release_authorized"] is not False
        ):
            raise ValueError("standalone Stage-2 GPU authorization drifted")
        _utc(record["authorized_at_utc"], "GPU authorization time")
        _role_digests(
            record["public_commitment_sha256s"],
            roles=_CONFIRMATION_ROLES,
            label="public commitment hashes",
        )
    for key, item in record.items():
        if key.endswith("sha256") and key not in {
            "public_commitment_sha256s",
            "boundary_fixture_manifest_sha256s",
        }:
            _digest(item, key)
    policy = krea_stage2_execution_surface_policy.validate(
        krea_stage2_execution_surface_policy.POLICY
    )
    contract = krea_stage2_delegated_review_contract.binding()
    if (
        record["policy_sha256"] != policy["policy_sha256"]
        or record["delegated_review_contract_sha256"] != contract["contract_sha256"]
        or record["delegated_review_contract_file_sha256"] != contract["file_sha256"]
    ):
        raise ValueError("Stage-2 admission record policy/contract binding drifted")
    if (
        kind in {RATIFICATION_KIND, GPU_AUTHORIZATION_KIND}
        and record["owner_identity_assurance"]
        != "interactive-owner-self-attestation-not-cryptographic-or-legal-signature"
    ):
        raise ValueError("Stage-2 owner identity assurance drifted")
    if not isinstance(record.get("image_id"), str) or not _IMAGE_ID.fullmatch(
        record["image_id"]
    ):
        raise ValueError("Stage-2 admission record image_id is not immutable")
    return record


def validate(value: Any) -> dict[str, Any]:
    """Validate a portable record; relationship validators remain stricter."""

    return _validate_standalone(value)


def publish(value: Any, output: str | Path) -> dict[str, Any]:
    """Create a canonical public governance record without overwriting."""

    record = validate(value)
    path = Path(os.path.abspath(os.path.expanduser(output)))
    _reject_symlink_ancestors(path, "admission record output")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(path, "admission record output")
    payload = canonical_bytes(record) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return binding(path)


def load(path: str | Path) -> dict[str, Any]:
    source = Path(os.path.abspath(os.path.expanduser(path)))
    _reject_symlink_ancestors(source, "Stage-2 admission record")
    if not source.is_file() or not stat.S_ISREG(source.stat().st_mode):
        raise ValueError("Stage-2 admission record must be regular and non-symlink")
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Stage-2 admission record is not JSON") from exc
    record = validate(value)
    if raw != canonical_bytes(record) + b"\n":
        raise ValueError("Stage-2 admission record is not canonical JSON")
    return record


def binding(path: str | Path) -> dict[str, Any]:
    source = Path(os.path.abspath(os.path.expanduser(path)))
    record = load(source)
    semantic_key = _SEMANTIC_KEYS[record["kind"]]
    return {
        "path": str(source),
        "kind": record["kind"],
        "file_sha256": canonical_file_sha256(record),
        semantic_key: record[semantic_key],
    }
