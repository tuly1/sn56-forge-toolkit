"""Truthful Stage-2 wrappers for the legacy C1-C4 checksum seals.

The independent reviewer preserved checksum manifests, archives, captions,
images, and licence records, but did not preserve the canonical fixture JSON
whose semantic digest appeared in the blinded review.  This module never
reconstructs or impersonates those missing bytes.  After the finalist freeze
it verifies the exact legacy seal, proves archive/member parity, and emits a
new canonical wrapper whose authority starts only at fresh Stage-2 admission.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence
import zipfile

try:
    from . import krea_c1c4_amendment as amendment
    from . import krea_provenance
    from . import krea_stage2_boundary_derivation as boundary
except ImportError:  # pragma: no cover - direct execution.
    import krea_c1c4_amendment as amendment  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_stage2_boundary_derivation as boundary  # type: ignore[no-redef]


SCHEMA = 1
KIND = "forge-krea-stage2-legacy-confirmation-wrapper"
ROLES = ("C1", "C2", "C3", "C4")
WRAPPER_NAME = "legacy-fixture-wrapper.json"
BLINDED_ACCEPTANCE = {
    "path": (
        "SN56-project/week5-krea-curation-20260729/"
        "blinded-acceptance-v5.json"
    ),
    "file_sha256": (
        "94cc99d9d9fb8907fb5eb31ad7933522d48315c3420ab8b2214bf18267719aad"
    ),
    "acceptance_sha256": (
        "8b185a1291fd52838a4846651d15852612d0760be6ff94f2c4fb3690d1956571"
    ),
}
PRIOR_SEMANTIC_SHA256S = {
    "C1": "29b0017e58a521692037d3309f275ae18a8f492df1a30b97365b129cea509039",
    "C2": "09137a94bbe01c435fe303ddc25c0f7867f838532e641ae7a59ead5e12c1a4af",
    "C3": "cc6c69c95a0a25a6c4b60145e14cd2caf92aac44e60a68b3132e09cfed8c7a6c",
    "C4": "fc3b46741e451ef0773ce622de4f72bf32dc3951ac04cb666e3ee91757adb35b",
}
SOURCE_TRANSFER = {
    "transport_tar_sha256": (
        "126b794eddf8ca3334cab3dadd6460df0d37043bb5aee9ea008edf4c64f6c304"
    ),
    "source_and_campaign_file_count": 278,
    "relative_path_and_content_merkle_sha256": (
        "9fe17500cf2de9085d5f4af8fe2b068d9082b3c03f6f356d27b5f63fbdb20526"
    ),
    "source_and_campaign_copy_match": True,
}
CLAIM_LIMIT = (
    "post-freeze-derived-wrapper-over-exact-legacy-checksum-seal;the-missing-"
    "original-fixture-manifest-is-not-reconstructed;the-prior-blinded-semantic-"
    "digest-is-review-evidence-only;C1-C4-use-natural-captions-and-null-trigger-"
    "which-differs-from-tokenized-D1-D2;fresh-stage2-ratification-required;not-"
    "release-deployment-or-competitiveness-authority"
)
_SHA = re.compile(r"[0-9a-f]{64}")
_CHECKSUM = re.compile(r"([0-9a-f]{64})  ([^\r\n]+)")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = keys - set(value)
    extra = set(value) - keys
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _utc(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("wrapper time must be canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError("wrapper time must be canonical UTC") from exc
    if parsed <= datetime(2026, 8, 1, 18, 19, 1, tzinfo=timezone.utc):
        raise ValueError("legacy wrapper must follow the pushed finalist freeze")
    if parsed > datetime.now(timezone.utc) + timedelta(seconds=60):
        raise ValueError("legacy wrapper time is in the future")
    return value


def _relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a normalized relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{label} must be a normalized relative path")
    return value


def _stable(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(after):
        raise RuntimeError(f"{label} changed while read")
    return raw


def _identity(path: Path, label: str) -> dict[str, Any]:
    raw = _stable(path, label)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _expected_dataset_paths(role: str) -> set[str]:
    shape = amendment.SHAPE_CONTRACT[role]
    return {
        *(f"image-{index:03d}.{suffix}" for index in range(1, shape["training_pairs"] + 1) for suffix in ("jpg", "txt")),
        *(f"holdout/image-{index:03d}.{suffix}" for index in range(1, shape["evaluation_rows"] + 1) for suffix in ("jpg", "txt")),
    }


def _parse_checksum(raw: bytes, *, role: str) -> list[dict[str, Any]]:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{role} checksum manifest is not ASCII") from exc
    pairs: list[tuple[str, str]] = []
    for line in lines:
        match = _CHECKSUM.fullmatch(line)
        if match is None:
            raise ValueError(f"{role} checksum manifest has a malformed line")
        digest, relative = match.groups()
        pairs.append((_relative(relative, f"{role} checksum path"), digest))
    if not pairs or len({path for path, _ in pairs}) != len(pairs):
        raise ValueError(f"{role} checksum manifest is empty or duplicated")
    if {path for path, _ in pairs} != _expected_dataset_paths(role):
        raise ValueError(f"{role} checksum manifest differs from amended shape")
    return [
        {"relative_path": path, "sha256": digest}
        for path, digest in sorted(pairs)
    ]


def _role_files(root: Path) -> list[str]:
    observed: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("legacy confirmation role contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("legacy confirmation role contains a special node")
        observed.append(path.relative_to(root).as_posix())
    return observed


def _archive_members(path: Path, expected: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if len(names) != len(set(names)) or any(
                item.is_dir()
                or item.filename != _relative(item.filename, "archive member")
                or (item.external_attr >> 16) & 0o170000 == stat.S_IFLNK
                for item in infos
            ):
                raise ValueError("training archive has unsafe or duplicate members")
            if set(names) != set(expected):
                raise ValueError("training archive member set differs from training bytes")
            for info in sorted(infos, key=lambda item: item.filename):
                raw = archive.read(info)
                identity = expected[info.filename]
                if len(raw) != identity["bytes"] or hashlib.sha256(raw).hexdigest() != identity["sha256"]:
                    raise ValueError("training archive member differs from sealed bytes")
                rows.append(
                    {
                        "relative_path": info.filename,
                        "sha256": identity["sha256"],
                        "bytes": identity["bytes"],
                    }
                )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("training archive is not a readable ZIP") from exc
    return rows


def _media_shapes(root: Path, entries: Sequence[Mapping[str, Any]], *, holdout: bool) -> list[dict[str, Any]]:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - production image carries Pillow.
        raise RuntimeError("Pillow is required to validate legacy C fixtures") from exc
    rows: list[dict[str, Any]] = []
    for item in entries:
        relative = str(item["relative_path"])
        is_holdout = relative.startswith("holdout/")
        if is_holdout != holdout or not relative.endswith(".jpg"):
            continue
        with Image.open(root / relative) as opened:
            media_type = opened.format
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = rgb.size
        if media_type != "JPEG" or width <= 0 or height <= 0:
            raise ValueError("legacy C image is not a valid JPEG")
        rows.append(
            {
                "relative_path": relative,
                "width": width,
                "height": height,
                "mode": "RGB",
                "media_type": media_type,
            }
        )
    return sorted(rows, key=lambda row: row["relative_path"])


def _dataset_identity(
    entries: Sequence[Mapping[str, Any]],
    shapes: Sequence[Mapping[str, Any]],
    *,
    holdout: bool,
) -> dict[str, Any]:
    prefix = "holdout/" if holdout else ""
    by_path = {str(row["relative_path"]): row for row in entries}
    shape_by_path = {str(row["relative_path"]): row for row in shapes}
    images = sorted(
        path.removeprefix(prefix)
        for path in by_path
        if path.startswith(prefix)
        and path.endswith(".jpg")
        and (path.startswith("holdout/")) is holdout
    )
    rows: list[dict[str, Any]] = []
    for index, image_name in enumerate(images):
        source_image = prefix + image_name
        prompt_name = image_name.removesuffix(".jpg") + ".txt"
        source_prompt = prefix + prompt_name
        image = by_path[source_image]
        prompt = by_path[source_prompt]
        shape = shape_by_path[source_image]
        rows.append(
            {
                "index": index,
                "image": image_name,
                "image_sha256": image["sha256"],
                "image_bytes": image["bytes"],
                "image_width": shape["width"],
                "image_height": shape["height"],
                "image_format": shape["media_type"],
                "image_mode": shape["mode"],
                "prompt": prompt_name,
                "prompt_sha256": prompt["sha256"],
                "prompt_bytes": prompt["bytes"],
            }
        )
    body = {"evaluator_order": images, "rows": rows}
    return {**body, "sha256": krea_provenance.canonical_sha256(body)}


def build_wrapper(*, role_root: str | Path, role: str, created_at_utc: str) -> dict[str, Any]:
    """Validate one exact legacy seal and create its canonical wrapper."""

    if role not in ROLES:
        raise ValueError("legacy confirmation role must be C1-C4")
    created = _utc(created_at_utc)
    root = Path(os.path.abspath(os.path.expanduser(str(role_root))))
    if root.is_symlink() or not root.is_dir():
        raise ValueError("legacy confirmation role root must be a real directory")
    wrapper_path = root / WRAPPER_NAME
    if os.path.lexists(wrapper_path):
        raise FileExistsError("legacy confirmation wrapper already exists")
    checksum_name = f"MANIFEST-{role}.sha256"
    archive_name = f"{role.lower()}_tourn.zip"
    source_paths = _role_files(root)
    expected_source = _expected_dataset_paths(role) | {
        checksum_name,
        archive_name,
        "LICENSES.txt",
    }
    if set(source_paths) != expected_source:
        raise ValueError(f"{role} legacy seal has an unlisted or missing file")
    checksum_raw = _stable(root / checksum_name, f"{role} checksum manifest")
    checksum_sha = hashlib.sha256(checksum_raw).hexdigest()
    if checksum_sha != amendment.MANIFEST_FILE_SHA256S[role]:
        raise ValueError(f"{role} published checksum commitment differs")
    checksum_rows = _parse_checksum(checksum_raw, role=role)
    entries: list[dict[str, Any]] = []
    for row in checksum_rows:
        identity = _identity(root / row["relative_path"], f"{role} sealed byte")
        if identity["sha256"] != row["sha256"]:
            raise ValueError(f"{role} sealed byte differs from checksum manifest")
        entries.append({"relative_path": row["relative_path"], **identity})
    train = {
        row["relative_path"]: row
        for row in entries
        if "/" not in row["relative_path"]
    }
    archive_identity = _identity(root / archive_name, f"{role} training archive")
    archive_members = _archive_members(root / archive_name, train)
    licences = _identity(root / "LICENSES.txt", f"{role} licence record")
    training_shapes = _media_shapes(root, entries, holdout=False)
    evaluation_shapes = _media_shapes(root, entries, holdout=True)
    training_identity = _dataset_identity(
        entries, training_shapes, holdout=False
    )
    evaluation_identity = _dataset_identity(
        entries, evaluation_shapes, holdout=True
    )
    body = {
        "schema": SCHEMA,
        "kind": KIND,
        "created_at_utc": created,
        "experimental_role": role,
        "trigger_token": None,
        "public_freeze_binding": {
            "path": boundary.FREEZE_BINDING_PATH,
            "file_sha256": boundary.FREEZE_BINDING_FILE_SHA256,
            "binding_sha256": boundary.FREEZE_BINDING_SHA256,
            "commit_sha1": boundary.FREEZE_BINDING_COMMIT,
        },
        "source_transfer": SOURCE_TRANSFER,
        "published_checksum_manifest": {
            "relative_path": f"{role}/{checksum_name}",
            "file_sha256": checksum_sha,
            "entries": [
                {**row, "relative_path": f"{role}/{row['relative_path']}"}
                for row in entries
            ],
            "entry_set_sha256": krea_provenance.canonical_sha256(entries),
        },
        "training_archive": {
            "relative_path": f"{role}/{archive_name}",
            **archive_identity,
            "members": archive_members,
            "member_set_sha256": krea_provenance.canonical_sha256(archive_members),
        },
        "licence_record": {
            "relative_path": f"{role}/LICENSES.txt",
            **licences,
        },
        "shape_amendment": {
            "path": amendment.AMENDMENT_PATH,
            "file_sha256": amendment.AMENDMENT_FILE_SHA256,
            "amendment_sha256": amendment.AMENDMENT_SHA256,
            "shape": amendment.SHAPE_CONTRACT[role],
        },
        "training_media_shapes": training_shapes,
        "evaluation_media_shapes": evaluation_shapes,
        "training_dataset_shape_sha256": krea_provenance.canonical_sha256(
            [
                {key: row[key] for key in ("width", "height", "mode", "media_type")}
                for row in training_shapes
            ]
        ),
        "evaluation_dataset_shape_sha256": krea_provenance.canonical_sha256(
            [
                {key: row[key] for key in ("width", "height", "mode", "media_type")}
                for row in evaluation_shapes
            ]
        ),
        "training_dataset_identity": training_identity,
        "evaluation_dataset_identity": evaluation_identity,
        "prior_blinded_review_evidence": {
            **BLINDED_ACCEPTANCE,
            "semantic_manifest_sha256": PRIOR_SEMANTIC_SHA256S[role],
            "authority_use": (
                "prior-review-evidence-only-not-a-reconstructed-manifest-or-"
                "current-byte-commitment"
            ),
            "original_fixture_manifest_bytes_preserved": False,
        },
        "natural_caption_bytes_unchanged": True,
        "scope_difference_from_tokenized_discovery": True,
        "fresh_stage2_owner_ratification_required": True,
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": CLAIM_LIMIT,
    }
    wrapper = {**body, "wrapper_sha256": krea_provenance.canonical_sha256(body)}
    validate_wrapper(wrapper)
    payload = krea_provenance.canonical_bytes(wrapper) + b"\n"
    with wrapper_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return wrapper


def _identity_row(value: Any, label: str) -> dict[str, Any]:
    row = _object(value, label)
    _exact(row, {"relative_path", "sha256", "bytes"}, label)
    size = row["bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{label}.bytes must be positive")
    return {
        "relative_path": _relative(row["relative_path"], f"{label}.relative_path"),
        "sha256": _digest(row["sha256"], f"{label}.sha256"),
        "bytes": size,
    }


def validate_wrapper(value: Any) -> dict[str, Any]:
    wrapper = _object(value, "legacy confirmation wrapper")
    required = {
        "schema", "kind", "created_at_utc", "experimental_role", "trigger_token",
        "public_freeze_binding", "source_transfer", "published_checksum_manifest",
        "training_archive", "licence_record", "shape_amendment",
        "prior_blinded_review_evidence", "natural_caption_bytes_unchanged",
        "training_media_shapes", "evaluation_media_shapes",
        "training_dataset_shape_sha256", "evaluation_dataset_shape_sha256",
        "training_dataset_identity", "evaluation_dataset_identity",
        "scope_difference_from_tokenized_discovery",
        "fresh_stage2_owner_ratification_required", "admission_authorized",
        "gpu_execution_authorized", "claim_limit", "wrapper_sha256",
    }
    _exact(wrapper, required, "legacy confirmation wrapper")
    role = wrapper["experimental_role"]
    if role not in ROLES:
        raise ValueError("legacy confirmation wrapper role must be C1-C4")
    _utc(wrapper["created_at_utc"])
    checksum = _object(wrapper["published_checksum_manifest"], "checksum binding")
    _exact(checksum, {"relative_path", "file_sha256", "entries", "entry_set_sha256"}, "checksum binding")
    entries = [_identity_row(row, "checksum entry") for row in checksum["entries"]]
    archive = _object(wrapper["training_archive"], "training archive")
    _exact(archive, {"relative_path", "sha256", "bytes", "members", "member_set_sha256"}, "training archive")
    archive_identity = _identity_row(
        {key: archive[key] for key in ("relative_path", "sha256", "bytes")},
        "training archive",
    )
    members = [_identity_row(row, "archive member") for row in archive["members"]]
    licence = _identity_row(wrapper["licence_record"], "licence record")
    evidence = _object(wrapper["prior_blinded_review_evidence"], "prior blinded evidence")
    _exact(evidence, {*BLINDED_ACCEPTANCE, "semantic_manifest_sha256", "authority_use", "original_fixture_manifest_bytes_preserved"}, "prior blinded evidence")
    training_shapes = wrapper["training_media_shapes"]
    evaluation_shapes = wrapper["evaluation_media_shapes"]
    training_identity = _object(
        wrapper["training_dataset_identity"], "training dataset identity"
    )
    evaluation_identity = _object(
        wrapper["evaluation_dataset_identity"], "evaluation dataset identity"
    )
    try:
        from . import krea_dataset_identity
    except ImportError:  # pragma: no cover - direct execution.
        import krea_dataset_identity  # type: ignore[no-redef]
    krea_dataset_identity.validate_identity(training_identity)
    krea_dataset_identity.validate_identity(evaluation_identity)
    shape_keys = {"relative_path", "width", "height", "mode", "media_type"}
    for label, rows in (("training", training_shapes), ("evaluation", evaluation_shapes)):
        if not isinstance(rows, list):
            raise ValueError(f"{label} media shapes must be a list")
        for row in rows:
            _exact(_object(row, f"{label} media shape"), shape_keys, f"{label} media shape")
            _relative(row["relative_path"], f"{label} media path")
            if (
                isinstance(row["width"], bool)
                or not isinstance(row["width"], int)
                or row["width"] <= 0
                or isinstance(row["height"], bool)
                or not isinstance(row["height"], int)
                or row["height"] <= 0
                or row["mode"] != "RGB"
                or row["media_type"] != "JPEG"
            ):
                raise ValueError(f"{label} media shape is invalid")
    if (
        wrapper["schema"] != SCHEMA
        or wrapper["kind"] != KIND
        or wrapper["trigger_token"] is not None
        or wrapper["public_freeze_binding"] != {
            "path": boundary.FREEZE_BINDING_PATH,
            "file_sha256": boundary.FREEZE_BINDING_FILE_SHA256,
            "binding_sha256": boundary.FREEZE_BINDING_SHA256,
            "commit_sha1": boundary.FREEZE_BINDING_COMMIT,
        }
        or wrapper["source_transfer"] != SOURCE_TRANSFER
        or checksum["relative_path"] != f"{role}/MANIFEST-{role}.sha256"
        or checksum["file_sha256"] != amendment.MANIFEST_FILE_SHA256S[role]
        or checksum["entry_set_sha256"] != krea_provenance.canonical_sha256(
            [{**row, "relative_path": row["relative_path"].removeprefix(f"{role}/")} for row in entries]
        )
        or {row["relative_path"] for row in entries}
        != {f"{role}/{path}" for path in _expected_dataset_paths(role)}
        or archive_identity["relative_path"] != f"{role}/{role.lower()}_tourn.zip"
        or archive["member_set_sha256"] != krea_provenance.canonical_sha256(members)
        or {row["relative_path"] for row in members}
        != {path for path in _expected_dataset_paths(role) if "/" not in path}
        or licence["relative_path"] != f"{role}/LICENSES.txt"
        or wrapper["shape_amendment"] != {
            "path": amendment.AMENDMENT_PATH,
            "file_sha256": amendment.AMENDMENT_FILE_SHA256,
            "amendment_sha256": amendment.AMENDMENT_SHA256,
            "shape": amendment.SHAPE_CONTRACT[role],
        }
        or len(training_shapes) != amendment.SHAPE_CONTRACT[role]["training_pairs"]
        or len(evaluation_shapes) != amendment.SHAPE_CONTRACT[role]["evaluation_rows"]
        or wrapper["training_dataset_shape_sha256"]
        != krea_provenance.canonical_sha256(
            [
                {key: row[key] for key in ("width", "height", "mode", "media_type")}
                for row in training_shapes
            ]
        )
        or wrapper["evaluation_dataset_shape_sha256"]
        != krea_provenance.canonical_sha256(
            [
                {key: row[key] for key in ("width", "height", "mode", "media_type")}
                for row in evaluation_shapes
            ]
        )
        or wrapper["training_dataset_identity"]
        != _dataset_identity(
            [
                {**row, "relative_path": row["relative_path"].removeprefix(f"{role}/")}
                for row in entries
            ],
            training_shapes,
            holdout=False,
        )
        or wrapper["evaluation_dataset_identity"]
        != _dataset_identity(
            [
                {**row, "relative_path": row["relative_path"].removeprefix(f"{role}/")}
                for row in entries
            ],
            evaluation_shapes,
            holdout=True,
        )
        or evidence != {
            **BLINDED_ACCEPTANCE,
            "semantic_manifest_sha256": PRIOR_SEMANTIC_SHA256S[role],
            "authority_use": "prior-review-evidence-only-not-a-reconstructed-manifest-or-current-byte-commitment",
            "original_fixture_manifest_bytes_preserved": False,
        }
        or wrapper["natural_caption_bytes_unchanged"] is not True
        or wrapper["scope_difference_from_tokenized_discovery"] is not True
        or wrapper["fresh_stage2_owner_ratification_required"] is not True
        or wrapper["admission_authorized"] is not False
        or wrapper["gpu_execution_authorized"] is not False
        or wrapper["claim_limit"] != CLAIM_LIMIT
    ):
        raise ValueError("legacy confirmation wrapper contract drifted")
    body = {key: item for key, item in wrapper.items() if key != "wrapper_sha256"}
    if wrapper["wrapper_sha256"] != krea_provenance.canonical_sha256(body):
        raise ValueError("legacy confirmation wrapper digest drifted")
    return wrapper


def validate_wrapper_file(*, wrapper: Any, role_root: str | Path) -> dict[str, Any]:
    """Replay wrapper claims against the exact fresh-root role bytes."""

    value = validate_wrapper(wrapper)
    role = value["experimental_role"]
    root = Path(os.path.abspath(os.path.expanduser(str(role_root))))
    observed = set(_role_files(root))
    expected = {
        WRAPPER_NAME,
        "LICENSES.txt",
        f"MANIFEST-{role}.sha256",
        f"{role.lower()}_tourn.zip",
        *(row["relative_path"].removeprefix(f"{role}/") for row in value["published_checksum_manifest"]["entries"]),
    }
    if observed != expected:
        raise ValueError("fresh legacy role file set differs from wrapper")
    checksum = _identity(root / f"MANIFEST-{role}.sha256", "checksum manifest")
    if checksum["sha256"] != value["published_checksum_manifest"]["file_sha256"]:
        raise ValueError("fresh checksum manifest differs from wrapper")
    for row in value["published_checksum_manifest"]["entries"]:
        relative = row["relative_path"].removeprefix(f"{role}/")
        if _identity(root / relative, "wrapped fixture byte") != {
            "sha256": row["sha256"], "bytes": row["bytes"]
        }:
            raise ValueError("fresh fixture byte differs from wrapper")
    archive = value["training_archive"]
    if _identity(root / f"{role.lower()}_tourn.zip", "training archive") != {
        "sha256": archive["sha256"], "bytes": archive["bytes"]
    }:
        raise ValueError("fresh training archive differs from wrapper")
    train = {
        row["relative_path"].removeprefix(f"{role}/"): {
            "sha256": row["sha256"], "bytes": row["bytes"]
        }
        for row in value["published_checksum_manifest"]["entries"]
        if "/holdout/" not in row["relative_path"]
    }
    if _archive_members(root / f"{role.lower()}_tourn.zip", train) != archive["members"]:
        raise ValueError("fresh training archive member parity drifted")
    if _identity(root / "LICENSES.txt", "licence record") != {
        "sha256": value["licence_record"]["sha256"],
        "bytes": value["licence_record"]["bytes"],
    }:
        raise ValueError("fresh licence record differs from wrapper")
    return value


def score_view(value: Any) -> dict[str, Any]:
    """Return the exact scorer-facing identity without impersonating a manifest."""

    wrapper = validate_wrapper(value)
    return {
        "schema": SCHEMA,
        "kind": KIND,
        "experimental_role": wrapper["experimental_role"],
        "trigger_token": None,
        "manifest_sha256": wrapper["wrapper_sha256"],
        "training_archive": {
            key: wrapper["training_archive"][key] for key in ("sha256", "bytes")
        },
        "training_dataset_identity": wrapper["training_dataset_identity"],
        "evaluation_dataset_identity": wrapper["evaluation_dataset_identity"],
        "training_dataset_shape_sha256": wrapper["training_dataset_shape_sha256"],
        "evaluation_dataset_shape_sha256": wrapper[
            "evaluation_dataset_shape_sha256"
        ],
        "legacy_wrapper_sha256": wrapper["wrapper_sha256"],
        "original_fixture_manifest_reconstructed": False,
    }
