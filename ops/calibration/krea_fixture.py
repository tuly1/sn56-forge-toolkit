#!/usr/bin/env python3
"""Pre-GPU Krea fixture curation and independent approval contracts.

The manifest is deliberately produced before any arm is trained.  It binds the
exact byte/order identity consumed by the G.O.D evaluator and a second,
filename-independent content identity used to reject leakage and duplicate
rows.  Paths are descriptive only; execution code must stage and re-hash the
approved bytes before use.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import stat
from typing import Any, Callable
import unicodedata
import zipfile

try:
    from . import krea_dataset_identity
    from . import krea_provenance
except ImportError:  # pragma: no cover - direct script execution.
    import krea_dataset_identity  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]


_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_DISCOVERY_ROLE_COUNTS = {
    "D1": ((18, 24), (24, 24)),
    "D2": ((36, 48), (40, 40)),
}
_CONFIRMATION_ROLE_COUNTS = {
    "C1": ((20, 20), (6, 6)),
    "C2": ((45, 45), (6, 6)),
    "C3": ((30, 30), (8, 8)),
    "C4": ((12, 12), (5, 5)),
}
# Stage-2's six boundary cells are deliberately enumerated instead of parsed as
# general experiment labels.  Keeping them out of ``_ROLE_COUNTS`` preserves
# the exact legacy D1/D2/C1-C4 namespace used by the Stage-1 cross-fixture
# contracts while allowing the ordinary byte/leakage validators to be reused.
_STAGE2_BOUNDARY_ROLE_COUNTS = {
    "B-0p5-small": ((18, 24), (24, 24)),
    "B-0p5-large": ((36, 48), (40, 40)),
    "B-0p75-small": ((18, 24), (24, 24)),
    "B-0p75-large": ((36, 48), (40, 40)),
    "B-1-small": ((18, 24), (24, 24)),
    "B-1-large": ((36, 48), (40, 40)),
}
_STAGE2_BOUNDARY_SOURCE_ROLES = {
    "B-0p5-small": "D1",
    "B-0p5-large": "D2",
    "B-0p75-small": "D1",
    "B-0p75-large": "D2",
    "B-1-small": "D1",
    "B-1-large": "D2",
}
_STAGE2_BOUNDARY_DERIVATION_MODE = (
    "byte-identical-admitted-discovery-fixture-mechanics-only-v1"
)
_STAGE2_BOUNDARY_DERIVATION_CLAIM_LIMIT = (
    "mechanics-only-byte-preserving-derivation-from-an-owner-admitted-D1-or-D2-"
    "fixture;source-governance-and-approval-remain-evidence-only-and-do-not-"
    "authorize-the-boundary-role;fresh-Stage-2-owner-ratification-and-GPU-"
    "authorization-remain-required;not-competitiveness-release-or-deployment-"
    "evidence"
)
_ROLE_COUNTS = {**_DISCOVERY_ROLE_COUNTS, **_CONFIRMATION_ROLE_COUNTS}
_CROSS_FIXTURE_ROLES = ("D1", "D2", "C1", "C2", "C3", "C4")
_BASE_GROUP_FIELDS = frozenset(
    {
        "source_id",
        "creator_id",
        "burst_id",
        "scene_id",
        "play_root_id",
        "human_similarity_cluster_id",
    }
)
_D2_GROUP_FIELDS = frozenset({"play_component_id", "accession_family_id"})
_BASE_GROUP_DISJOINT_FIELDS = frozenset(
    {"burst_id", "scene_id", "play_root_id", "human_similarity_cluster_id"}
)
_HUMAN_IDENTITY_ASSURANCE = (
    "named-human-string-self-assertion-not-cryptographic-authentication"
)
_AGENT_IDENTITY_ASSURANCE = (
    "self-declared-agent-identity-not-human-or-cryptographic-authentication"
)
_AGENT_GOVERNANCE_MODE = "sole-human-owner-ratifies-agent-review-v1"
_OWNER_IDENTITY_ASSURANCE = (
    "interactive-owner-self-attestation-not-cryptographic-or-legal-signature"
)
_AGENT_CROSS_REVIEW_KIND = "forge-krea-cross-fixture-agent-similarity-review"
_AGENT_CROSS_BINDING_KIND = "forge-krea-cross-fixture-agent-review-binding"
_SEALED_CUSTODIAN_ROLE = "sealed_confirmation_custodian"
_AGENT_CROSS_CLAIM_LIMIT = (
    "cross-fixture-nonoverlap-agent-technical-evidence-only-not-human-review-"
    "content-disclosure-quality-proof-admission-or-gpu-authorization"
)
_ROLE_LABELS = frozenset(
    {
        "reviewer",
        "human reviewer",
        "human owner",
        "owner",
        "engineer",
        "response engineer",
        "review engineer",
        "user",
        "operator",
        "dri",
    }
)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(value: dict[str, Any], keys: set[str], label: str) -> None:
    missing = keys - set(value)
    extra = set(value) - keys
    if missing or extra:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return " ".join(value.split())


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def named_human(value: Any, label: str) -> str:
    identity = _text(value, label)
    if value != identity or unicodedata.normalize("NFKC", identity) != identity:
        raise ValueError(f"{label} must use canonical Unicode and spacing")
    if identity.casefold() in _ROLE_LABELS:
        raise ValueError(f"{label} is a role label, not a named human")
    words = identity.split()
    if len(words) < 2 or any(not any(c.isalpha() for c in word) for word in words):
        raise ValueError(f"{label} must contain a named human identity")
    return identity


def _human_identity_key(value: Any, label: str) -> str:
    """Normalize a human label for independence checks, not authentication.

    The evidence format records a named-human assertion.  It is deliberately
    not described as a signature or proof of identity.  Unicode/case/spacing
    normalization prevents the same asserted reviewer bypassing an
    independence gate with a cosmetic spelling change.
    """

    return unicodedata.normalize("NFKC", named_human(value, label)).casefold()


def _parse_canonical_utc(value: Any, label: str) -> datetime:
    return datetime.strptime(canonical_utc(value, label), "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )


def _validate_role_counts(
    role: str, training_count: int, evaluation_count: int
) -> None:
    """Enforce discovery ranges and each published confirmation shape exactly."""

    counts = _ROLE_COUNTS.get(role)
    if counts is None:
        counts = _STAGE2_BOUNDARY_ROLE_COUNTS.get(role)
    if counts is None:
        raise ValueError(
            "experimental_role must be D1, D2, C1-C4, or an exact Stage-2 "
            "boundary role"
        )
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in (training_count, evaluation_count)
    ):
        raise ValueError("fixture counts must be non-negative integers")
    train_range, eval_range = counts
    if not train_range[0] <= training_count <= train_range[1]:
        raise ValueError(f"{role} training count is outside {train_range}")
    if not eval_range[0] <= evaluation_count <= eval_range[1]:
        raise ValueError(f"{role} evaluation count is outside {eval_range}")


def _group_fields(role: str) -> frozenset[str]:
    if role not in _ROLE_COUNTS and role not in _STAGE2_BOUNDARY_ROLE_COUNTS:
        raise ValueError(
            "experimental_role must be D1, D2, C1-C4, or an exact Stage-2 "
            "boundary role"
        )
    return _BASE_GROUP_FIELDS | (_D2_GROUP_FIELDS if role == "D2" else frozenset())


def _normalize_group_identity(value: Any, *, role: str, label: str) -> dict[str, str]:
    group = _object(value, label)
    fields = _group_fields(role)
    _exact(group, set(fields), label)
    return {key: _text(group[key], f"{label}.{key}") for key in sorted(fields)}


def _normalize_group_disjoint_fields(value: Any, *, role: str) -> list[str]:
    fields = _group_fields(role)
    required = _BASE_GROUP_DISJOINT_FIELDS | (
        _D2_GROUP_FIELDS if role == "D2" else frozenset()
    )
    if (
        not isinstance(value, list)
        or any(not isinstance(field, str) for field in value)
        or value != sorted(set(value))
        or not required.issubset(value)
        or any(field not in fields for field in value)
    ):
        raise ValueError("group_disjoint_fields does not satisfy the leakage policy")
    return list(value)


def canonical_utc(value: Any, label: str) -> str:
    """Require an unambiguous canonical UTC timestamp within sane bounds."""

    text = _text(value, label)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", text):
        raise ValueError(f"{label} must be canonical UTC (YYYY-MM-DDTHH:MM:SSZ)")
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{label} is not a real UTC timestamp") from exc
    now = datetime.now(timezone.utc)
    if parsed < datetime(2020, 1, 1, tzinfo=timezone.utc) or parsed > now + timedelta(
        seconds=60
    ):
        raise ValueError(f"{label} is outside the accepted evidence time bounds")
    return text


def _canonical_record(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = _safe_file(path, label)
    raw = path.read_bytes()
    try:
        value = _object(json.loads(raw), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid JSON") from exc
    if raw != krea_provenance.canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline")
    return value, hashlib.sha256(raw).hexdigest()


def _reviewed_pairs(row_ids: list[str]) -> list[list[str]]:
    ordered = sorted(row_ids)
    return [
        [left, right]
        for index, left in enumerate(ordered)
        for right in ordered[index + 1 :]
    ]


def _rights_summary(path: Path, *, owner: str, locator: str) -> dict[str, Any]:
    record, digest = _canonical_record(path, "source rights record")
    _exact(
        record,
        {
            "schema",
            "kind",
            "owner",
            "locator",
            "reviewer_identity",
            "reviewed_at_utc",
            "decision",
            "assertions",
        },
        "source rights record",
    )
    assertions = _object(record["assertions"], "source rights assertions")
    _exact(
        assertions,
        {
            "lawful_access",
            "calibration_use_allowed",
            "redistribution_reviewed",
            "sensitive_content_absent",
        },
        "source rights assertions",
    )
    reviewer = named_human(record["reviewer_identity"], "rights reviewer")
    reviewed_at = canonical_utc(record["reviewed_at_utc"], "rights reviewed_at_utc")
    if (
        record["schema"] != 1
        or record["kind"] != "forge-krea-source-rights-review"
        or record["owner"] != owner
        or record["locator"] != locator
        or record["decision"] != "approved_for_calibration"
        or any(value is not True for value in assertions.values())
    ):
        raise ValueError("source rights record does not approve this exact source")
    return {
        "record_sha256": digest,
        "reviewer_identity": reviewer,
        "reviewed_at_utc": reviewed_at,
        "decision": record["decision"],
        "assertions": assertions,
    }


def _caption_summary(
    path: Path,
    *,
    concept_id: str,
    trigger_token: str,
    training_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    record, digest = _canonical_record(path, "caption policy record")
    _exact(
        record,
        {
            "schema",
            "kind",
            "concept_id",
            "trigger_token",
            "reviewer_identity",
            "reviewed_at_utc",
            "decision",
            "training_row_ids",
            "evaluation_row_ids",
            "assertions",
        },
        "caption policy record",
    )
    assertions = _object(record["assertions"], "caption policy assertions")
    _exact(
        assertions,
        {
            "manual_review_complete",
            "captions_match_images",
            "trigger_usage_consistent",
            "evaluation_leakage_absent",
        },
        "caption policy assertions",
    )
    expected_train = sorted(row["row_id"] for row in training_rows)
    expected_eval = sorted(row["row_id"] for row in evaluation_rows)
    reviewer = named_human(record["reviewer_identity"], "caption reviewer")
    reviewed_at = canonical_utc(record["reviewed_at_utc"], "caption reviewed_at_utc")
    if (
        record["schema"] != 1
        or record["kind"] != "forge-krea-caption-review"
        or record["concept_id"] != concept_id
        or record["trigger_token"] != trigger_token
        or record["decision"] != "approved"
        or record["training_row_ids"] != expected_train
        or record["evaluation_row_ids"] != expected_eval
        or any(value is not True for value in assertions.values())
    ):
        raise ValueError("caption record does not approve every fixture row")
    return {
        "record_sha256": digest,
        "reviewer_identity": reviewer,
        "reviewed_at_utc": reviewed_at,
        "decision": record["decision"],
        "training_row_ids_sha256": krea_provenance.canonical_sha256(expected_train),
        "evaluation_row_ids_sha256": krea_provenance.canonical_sha256(expected_eval),
        "assertions": assertions,
    }


def _similarity_summary(
    path: Path,
    *,
    concept_id: str,
    role: str,
    reviewer_identity: str,
    training_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    record, digest = _canonical_record(path, "human similarity review")
    _exact(
        record,
        {
            "schema",
            "kind",
            "concept_id",
            "experimental_role",
            "reviewer_identity",
            "reviewed_at_utc",
            "decision",
            "reviewed_pairs",
            "flagged_pairs",
        },
        "human similarity review",
    )
    expected_pairs = _reviewed_pairs(
        [row["row_id"] for row in training_rows + evaluation_rows]
    )
    reviewer = named_human(record["reviewer_identity"], "similarity reviewer")
    reviewed_at = canonical_utc(record["reviewed_at_utc"], "similarity reviewed_at_utc")
    if (
        record["schema"] != 1
        or record["kind"] != "forge-krea-human-similarity-review"
        or record["concept_id"] != concept_id
        or record["experimental_role"] != role
        or reviewer != reviewer_identity
        or record["decision"] != "passed"
        or record["reviewed_pairs"] != expected_pairs
        or record["flagged_pairs"] != []
    ):
        raise ValueError("human similarity review is incomplete or did not pass")
    return {
        "reviewer_identity": reviewer,
        "reviewed_at_utc": reviewed_at,
        "record_sha256": digest,
        "method": "named-human-exhaustive-pair-review-plus-pinned-ahash",
        "reviewed_pair_count": len(expected_pairs),
        "reviewed_pairs_sha256": krea_provenance.canonical_sha256(expected_pairs),
        "decision": "passed",
        "passed": True,
    }


def _sha(path: Path) -> str:
    return krea_provenance.file_sha256(path)


def _safe_file(path: Path, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _safe_directory(path: Path, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real directory: {path}")
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    return path


def _decode_row(
    root: Path, evaluator_row: dict[str, Any], group_identity: dict[str, Any]
) -> dict[str, Any]:
    try:
        from PIL import Image, ImageOps, __version__ as pillow_version
    except ImportError as exc:  # pragma: no cover - image runtime has Pillow.
        raise RuntimeError("Pillow is required to curate Krea fixtures") from exc

    image = root / evaluator_row["image"]
    caption = root / evaluator_row["prompt"]
    caption_bytes = caption.read_bytes()
    try:
        caption_text = caption_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"caption is not UTF-8: {caption}") from exc
    normalized_caption = " ".join(
        unicodedata.normalize("NFC", caption_text).casefold().split()
    ).encode("utf-8")
    if not normalized_caption:
        raise ValueError(f"caption normalizes to empty: {caption}")
    with Image.open(image) as opened:
        rgb = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = rgb.size
        pixels = rgb.tobytes()
        # Pin the resampling algorithm so Pillow upgrades cannot silently
        # redefine the near-duplicate detector.
        reduced = rgb.convert("L").resize((8, 8), resample=Image.Resampling.BILINEAR)
        values = list(reduced.getdata())
    mean = sum(values) / len(values)
    bits = 0
    for value in values:
        bits = (bits << 1) | int(value >= mean)
    content = {
        "image_sha256": evaluator_row["image_sha256"],
        "decoded_pixels_sha256": hashlib.sha256(pixels).hexdigest(),
        "caption_sha256": evaluator_row["prompt_sha256"],
        "normalized_caption_sha256": hashlib.sha256(normalized_caption).hexdigest(),
        "width": width,
        "height": height,
        "mode": "RGB",
    }
    return {
        "row_id": "row-" + krea_provenance.canonical_sha256(content),
        "relative_image_path": evaluator_row["image"],
        "relative_caption_path": evaluator_row["prompt"],
        "content_sha256": krea_provenance.canonical_sha256(content),
        **content,
        "media_type": evaluator_row["image_format"],
        "perceptual_hash64": f"{bits:016x}",
        "decoder": {"library": "Pillow", "version": pillow_version},
        "group_identity": group_identity,
    }


def _rows(
    root: Path,
    *,
    role: str,
    list_supported_images: Callable[[str, tuple[str, ...]], list[str]],
    extensions: tuple[str, ...],
    row_groups: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluator_identity = krea_dataset_identity.capture_dataset(
        root,
        list_supported_images=list_supported_images,
        extensions=extensions,
    )
    image_names = set(evaluator_identity["evaluator_order"])
    if set(row_groups) != image_names:
        raise ValueError("row_groups must cover every image exactly")
    normalized_groups = {}
    for image_name, raw_group in row_groups.items():
        normalized_groups[image_name] = _normalize_group_identity(
            raw_group, role=role, label=f"row group {image_name}"
        )
    rows = [
        _decode_row(root, row, normalized_groups[row["image"]])
        for row in evaluator_identity["rows"]
    ]
    rows.sort(key=lambda row: row["row_id"])
    return evaluator_identity, rows


def _duplicates(
    training_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    *,
    threshold: int,
    group_disjoint_fields: tuple[str, ...],
) -> dict[str, Any]:
    combined = [("training", row) for row in training_rows] + [
        ("evaluation", row) for row in evaluation_rows
    ]
    exact_keys = (
        "row_id",
        "content_sha256",
        "image_sha256",
        "decoded_pixels_sha256",
        "normalized_caption_sha256",
    )
    exact_matches: list[dict[str, Any]] = []
    near_matches: list[dict[str, Any]] = []
    group_matches: list[dict[str, Any]] = []
    minimum = 64
    comparisons = 0
    for left_index, (left_split, left) in enumerate(combined):
        for right_split, right in combined[left_index + 1 :]:
            comparisons += 1
            shared = [key for key in exact_keys if left[key] == right[key]]
            if shared:
                exact_matches.append(
                    {
                        "left": left["row_id"],
                        "right": right["row_id"],
                        "left_split": left_split,
                        "right_split": right_split,
                        "identities": shared,
                    }
                )
            distance = (
                int(left["perceptual_hash64"], 16) ^ int(right["perceptual_hash64"], 16)
            ).bit_count()
            minimum = min(minimum, distance)
            if distance <= threshold:
                near_matches.append(
                    {
                        "left": left["row_id"],
                        "right": right["row_id"],
                        "left_split": left_split,
                        "right_split": right_split,
                        "hamming_distance": distance,
                    }
                )
            if left_split != right_split:
                for field in group_disjoint_fields:
                    if left["group_identity"][field] == right["group_identity"][field]:
                        group_matches.append(
                            {
                                "left": left["row_id"],
                                "right": right["row_id"],
                                "field": field,
                                "value": left["group_identity"][field],
                            }
                        )
    return {
        "comparisons": comparisons,
        "minimum_hamming_distance": minimum,
        "exact_matches": exact_matches,
        "near_matches": near_matches,
        "cross_split_group_matches": group_matches,
    }


def _archive_identity(
    archive: Path, *, training_identity: dict[str, Any]
) -> dict[str, Any]:
    """Read a ZIP without extraction and prove exact safe member equivalence."""

    if not zipfile.is_zipfile(archive):
        raise ValueError("training archive must be a valid ZIP")
    expected: dict[str, tuple[str, int]] = {}
    for row in training_identity["rows"]:
        expected[row["image"]] = (row["image_sha256"], row["image_bytes"])
        expected[row["prompt"]] = (row["prompt_sha256"], row["prompt_bytes"])
    members = []
    seen: set[str] = set()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            name = info.filename
            if info.is_dir():
                raise ValueError("training archive must not contain directory entries")
            mode = (info.external_attr >> 16) & 0o170000
            if (
                not isinstance(name, str)
                or name in {"", ".", ".."}
                or Path(name).name != name
                or "/" in name
                or "\\" in name
                or name in seen
                or info.flag_bits & 0x1
                or mode not in {0, stat.S_IFREG}
            ):
                raise ValueError(
                    f"training archive contains an unsafe member: {name!r}"
                )
            seen.add(name)
            if name not in expected:
                raise ValueError(
                    f"training archive contains an unexpected member: {name}"
                )
            digest = hashlib.sha256()
            total = 0
            with bundle.open(info, "r") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                    total += len(block)
            if (digest.hexdigest(), total) != expected[name]:
                raise ValueError(
                    f"training archive member differs from fixture: {name}"
                )
            members.append({"path": name, "sha256": digest.hexdigest(), "bytes": total})
    if seen != set(expected):
        raise ValueError("training archive is missing approved dataset members")
    members.sort(key=lambda row: row["path"])
    body = {"format": "zip", "members": members}
    return {**body, "sha256": krea_provenance.canonical_sha256(body)}


def build_manifest(
    metadata: dict[str, Any],
    *,
    training_dir: Path,
    evaluation_dir: Path,
    training_archive: Path,
    rights_record: Path,
    caption_policy: Path,
    similarity_review_record: Path,
    list_supported_images: Callable[[str, tuple[str, ...]], list[str]],
    extensions: tuple[str, ...],
) -> dict[str, Any]:
    """Build one deterministic, pre-training split manifest."""

    metadata = _object(metadata, "fixture metadata")
    _exact(
        metadata,
        {
            "concept_id",
            "experimental_role",
            "trigger_token",
            "source_owner",
            "source_locator",
            "source_retrieved_at_utc",
            "preparer_identity",
            "god_commit",
            "near_duplicate_hamming_threshold",
            "group_disjoint_fields",
            "training_row_groups",
            "evaluation_row_groups",
            "similarity_reviewer_identity",
        },
        "fixture metadata",
    )
    concept_id = _text(metadata["concept_id"], "concept_id")
    if not _SAFE_ID.fullmatch(concept_id):
        raise ValueError("concept_id must be one conservative identifier")
    role = _text(metadata["experimental_role"], "experimental_role")
    if role not in _ROLE_COUNTS and role not in _STAGE2_BOUNDARY_ROLE_COUNTS:
        raise ValueError(
            "experimental_role must be D1, D2, C1-C4, or an exact Stage-2 "
            "boundary role"
        )
    threshold = metadata["near_duplicate_hamming_threshold"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int)
        or not 0 <= threshold <= 64
    ):
        raise ValueError("near-duplicate threshold must be an integer in [0, 64]")
    god_commit = _text(metadata["god_commit"], "god_commit").lower()
    if not re.fullmatch(r"[0-9a-f]{40}", god_commit):
        raise ValueError("god_commit must be a full 40-digit commit")
    source_owner = _text(metadata["source_owner"], "source_owner")
    source_locator = _text(metadata["source_locator"], "source_locator")
    source_retrieved_at = canonical_utc(
        metadata["source_retrieved_at_utc"], "source_retrieved_at_utc"
    )
    trigger_token = _text(metadata["trigger_token"], "trigger_token")
    similarity_reviewer = named_human(
        metadata["similarity_reviewer_identity"],
        "similarity_reviewer_identity",
    )

    training_dir = _safe_directory(training_dir, "training directory")
    evaluation_dir = _safe_directory(evaluation_dir, "evaluation directory")
    if training_dir == evaluation_dir:
        raise ValueError("training and exact-evaluation directories must differ")
    training_archive = _safe_file(training_archive, "training archive")
    rights_record = _safe_file(rights_record, "rights record")
    caption_policy = _safe_file(caption_policy, "caption policy")
    similarity_review_record = _safe_file(
        similarity_review_record, "human similarity review"
    )
    group_disjoint_fields = _normalize_group_disjoint_fields(
        metadata["group_disjoint_fields"], role=role
    )
    training_identity, training_rows = _rows(
        training_dir,
        role=role,
        list_supported_images=list_supported_images,
        extensions=extensions,
        row_groups=_object(metadata["training_row_groups"], "training_row_groups"),
    )
    evaluation_identity, evaluation_rows = _rows(
        evaluation_dir,
        role=role,
        list_supported_images=list_supported_images,
        extensions=extensions,
        row_groups=_object(metadata["evaluation_row_groups"], "evaluation_row_groups"),
    )
    _validate_role_counts(role, len(training_rows), len(evaluation_rows))
    report = _duplicates(
        training_rows,
        evaluation_rows,
        threshold=threshold,
        group_disjoint_fields=tuple(group_disjoint_fields),
    )
    if (
        report["exact_matches"]
        or report["near_matches"]
        or report["cross_split_group_matches"]
    ):
        raise ValueError("fixture contains exact, perceptual, or grouped leakage")
    archive_identity = _archive_identity(
        training_archive, training_identity=training_identity
    )
    rights = _rights_summary(rights_record, owner=source_owner, locator=source_locator)
    captions = _caption_summary(
        caption_policy,
        concept_id=concept_id,
        trigger_token=trigger_token,
        training_rows=training_rows,
        evaluation_rows=evaluation_rows,
    )
    similarity = _similarity_summary(
        similarity_review_record,
        concept_id=concept_id,
        role=role,
        reviewer_identity=similarity_reviewer,
        training_rows=training_rows,
        evaluation_rows=evaluation_rows,
    )
    enumerator_source = inspect.getsourcefile(list_supported_images)
    if enumerator_source is None:
        raise ValueError("image enumerator source file is unavailable")
    enumerator_source_path = _safe_file(Path(enumerator_source), "enumerator source")
    tool_identity = {
        "fixture_module_sha256": _sha(Path(__file__).resolve(strict=True)),
        "dataset_identity_module_sha256": _sha(
            Path(krea_dataset_identity.__file__).resolve(strict=True)
        ),
        "god_commit": god_commit,
        "enumerator_module": getattr(list_supported_images, "__module__", None),
        "enumerator_qualname": getattr(list_supported_images, "__qualname__", None),
        "enumerator_source_sha256": _sha(enumerator_source_path),
        "enumerator_callable_sha256": hashlib.sha256(
            inspect.getsource(list_supported_images).encode("utf-8")
        ).hexdigest(),
        "extensions": list(extensions),
        "perceptual_hash": ("rgb-luma-average-hash-8x8-bilinear-after-exif-transpose"),
    }
    body = {
        "schema": 1,
        "kind": "forge-krea-curated-fixture",
        "concept_id": concept_id,
        "experimental_role": role,
        "trigger_token": trigger_token,
        "caption_policy": captions,
        "source_rights": {
            "owner": source_owner,
            "locator": source_locator,
            "retrieved_at_utc": source_retrieved_at,
            **rights,
        },
        "preparer_identity": named_human(
            metadata["preparer_identity"], "preparer_identity"
        ),
        "training_archive": {
            "sha256": _sha(training_archive),
            "bytes": training_archive.stat().st_size,
        },
        "training_archive_identity": archive_identity,
        "training_dataset_identity": training_identity,
        "evaluation_dataset_identity": evaluation_identity,
        "training_dataset_shape_sha256": krea_provenance.canonical_sha256(
            [
                {
                    "width": row["width"],
                    "height": row["height"],
                    "mode": row["mode"],
                    "media_type": row["media_type"],
                }
                for row in training_rows
            ]
        ),
        "evaluation_dataset_shape_sha256": krea_provenance.canonical_sha256(
            [
                {
                    "width": row["width"],
                    "height": row["height"],
                    "mode": row["mode"],
                    "media_type": row["media_type"],
                }
                for row in evaluation_rows
            ]
        ),
        "training_rows": training_rows,
        "evaluation_rows": evaluation_rows,
        "tool_identity": tool_identity,
        "near_duplicate_policy": {
            "maximum_hamming_distance": threshold,
            "report": report,
            "report_sha256": krea_provenance.canonical_sha256(report),
            "passed": True,
            "group_disjoint_fields": group_disjoint_fields,
            "human_similarity_review": {
                **similarity,
            },
        },
    }
    manifest = {**body, "manifest_sha256": krea_provenance.canonical_sha256(body)}
    validate_manifest(manifest)
    return manifest


def _validate_legacy_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest = _object(manifest, "fixture manifest")
    expected = {
        "schema",
        "kind",
        "concept_id",
        "experimental_role",
        "trigger_token",
        "caption_policy",
        "source_rights",
        "preparer_identity",
        "training_archive",
        "training_archive_identity",
        "training_dataset_identity",
        "evaluation_dataset_identity",
        "training_dataset_shape_sha256",
        "evaluation_dataset_shape_sha256",
        "training_rows",
        "evaluation_rows",
        "tool_identity",
        "near_duplicate_policy",
        "manifest_sha256",
    }
    _exact(manifest, expected, "fixture manifest")
    if manifest["schema"] != 1 or manifest["kind"] != "forge-krea-curated-fixture":
        raise ValueError("unsupported fixture manifest")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest["manifest_sha256"] != krea_provenance.canonical_sha256(body):
        raise ValueError("fixture manifest digest mismatch")
    role = manifest["experimental_role"]
    if role not in _ROLE_COUNTS and role not in _STAGE2_BOUNDARY_ROLE_COUNTS:
        raise ValueError("invalid fixture role")
    named_human(manifest["preparer_identity"], "preparer_identity")
    for identity_key in ("training_dataset_identity", "evaluation_dataset_identity"):
        krea_dataset_identity.validate_identity(
            _object(manifest[identity_key], identity_key)
        )
    for key in (
        "training_dataset_shape_sha256",
        "evaluation_dataset_shape_sha256",
    ):
        if not isinstance(manifest[key], str) or not _SHA256.fullmatch(manifest[key]):
            raise ValueError(f"{key} is invalid")
    train_rows = manifest["training_rows"]
    eval_rows = manifest["evaluation_rows"]
    if not isinstance(train_rows, list) or not isinstance(eval_rows, list):
        raise ValueError("fixture rows must be arrays")
    _validate_role_counts(role, len(train_rows), len(eval_rows))
    rich_keys = {
        "row_id",
        "relative_image_path",
        "relative_caption_path",
        "content_sha256",
        "image_sha256",
        "decoded_pixels_sha256",
        "caption_sha256",
        "normalized_caption_sha256",
        "width",
        "height",
        "mode",
        "media_type",
        "perceptual_hash64",
        "decoder",
        "group_identity",
    }
    for split, rows in (("training", train_rows), ("evaluation", eval_rows)):
        identity_rows = manifest[f"{split}_dataset_identity"]["rows"]
        identity_by_pair = {(row["image"], row["prompt"]): row for row in identity_rows}
        if rows != sorted(rows, key=lambda row: row.get("row_id", "")):
            raise ValueError(f"{split} fixture rows are not canonically sorted")
        if len(rows) != len(identity_rows):
            raise ValueError(f"{split} rich/evaluator row counts differ")
        for row in rows:
            if not isinstance(row, dict) or set(row) != rich_keys:
                raise ValueError(f"{split} fixture row schema mismatch")
            evaluator_row = identity_by_pair.get(
                (row["relative_image_path"], row["relative_caption_path"])
            )
            if evaluator_row is None:
                raise ValueError(
                    f"{split} fixture row is absent from evaluator identity"
                )
            content = {
                "image_sha256": row["image_sha256"],
                "decoded_pixels_sha256": row["decoded_pixels_sha256"],
                "caption_sha256": row["caption_sha256"],
                "normalized_caption_sha256": row["normalized_caption_sha256"],
                "width": row["width"],
                "height": row["height"],
                "mode": row["mode"],
            }
            if (
                row["row_id"] != "row-" + krea_provenance.canonical_sha256(content)
                or row["content_sha256"] != krea_provenance.canonical_sha256(content)
                or row["image_sha256"] != evaluator_row["image_sha256"]
                or row["caption_sha256"] != evaluator_row["prompt_sha256"]
                or isinstance(row["width"], bool)
                or not isinstance(row["width"], int)
                or row["width"] <= 0
                or isinstance(row["height"], bool)
                or not isinstance(row["height"], int)
                or row["height"] <= 0
                or (row["width"], row["height"])
                not in {
                    (evaluator_row["image_width"], evaluator_row["image_height"]),
                    (evaluator_row["image_height"], evaluator_row["image_width"]),
                }
                or row["media_type"] != evaluator_row["image_format"]
                or row["mode"] != "RGB"
                or not isinstance(row["perceptual_hash64"], str)
                or not re.fullmatch(r"[0-9a-f]{16}", row["perceptual_hash64"])
            ):
                raise ValueError(f"{split} fixture row identity mismatch")
            for digest_field in (
                "decoded_pixels_sha256",
                "normalized_caption_sha256",
            ):
                if not isinstance(row[digest_field], str) or not _SHA256.fullmatch(
                    row[digest_field]
                ):
                    raise ValueError(f"{split} fixture {digest_field} is invalid")
            decoder = _object(row["decoder"], f"{split} fixture decoder")
            _exact(decoder, {"library", "version"}, f"{split} fixture decoder")
            if (
                decoder["library"] != "Pillow"
                or not isinstance(decoder["version"], str)
                or not decoder["version"]
            ):
                raise ValueError(f"{split} fixture decoder identity is invalid")
            _normalize_group_identity(
                row["group_identity"],
                role=role,
                label=f"{split} fixture group",
            )
        shape = [
            {
                "width": row["width"],
                "height": row["height"],
                "mode": row["mode"],
                "media_type": row["media_type"],
            }
            for row in rows
        ]
        if manifest[
            f"{split}_dataset_shape_sha256"
        ] != krea_provenance.canonical_sha256(shape):
            raise ValueError(f"{split} dataset shape digest mismatch")
    near = _object(manifest["near_duplicate_policy"], "near_duplicate_policy")
    _exact(
        near,
        {
            "maximum_hamming_distance",
            "report",
            "report_sha256",
            "passed",
            "group_disjoint_fields",
            "human_similarity_review",
        },
        "near_duplicate_policy",
    )
    threshold = near["maximum_hamming_distance"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int)
        or not 0 <= threshold <= 64
    ):
        raise ValueError("fixture duplicate threshold is invalid")
    try:
        group_fields = _normalize_group_disjoint_fields(
            near["group_disjoint_fields"], role=role
        )
    except ValueError as exc:
        raise ValueError("fixture group-disjointness policy is invalid") from exc
    recomputed_report = _duplicates(
        train_rows,
        eval_rows,
        threshold=threshold,
        group_disjoint_fields=tuple(group_fields),
    )
    if (
        near["passed"] is not True
        or near["report"] != recomputed_report
        or near["report_sha256"] != krea_provenance.canonical_sha256(near["report"])
        or near["report"].get("exact_matches") != []
        or near["report"].get("near_matches") != []
        or near["report"].get("cross_split_group_matches") != []
    ):
        raise ValueError("fixture duplicate controls did not pass")
    human_review = _object(near["human_similarity_review"], "human_similarity_review")
    _exact(
        human_review,
        {
            "reviewer_identity",
            "reviewed_at_utc",
            "record_sha256",
            "method",
            "reviewed_pair_count",
            "reviewed_pairs_sha256",
            "decision",
            "passed",
        },
        "human_similarity_review",
    )
    expected_pair_count = (
        (len(train_rows) + len(eval_rows)) * (len(train_rows) + len(eval_rows) - 1) // 2
    )
    if (
        human_review["passed"] is not True
        or human_review["method"]
        != "named-human-exhaustive-pair-review-plus-pinned-ahash"
        or human_review["decision"] != "passed"
        or human_review["reviewed_pair_count"] != expected_pair_count
        or human_review["reviewed_pairs_sha256"]
        != krea_provenance.canonical_sha256(
            _reviewed_pairs([row["row_id"] for row in train_rows + eval_rows])
        )
        or not isinstance(human_review["record_sha256"], str)
        or not _SHA256.fullmatch(human_review["record_sha256"])
        or not isinstance(human_review["reviewed_pairs_sha256"], str)
        or not _SHA256.fullmatch(human_review["reviewed_pairs_sha256"])
    ):
        raise ValueError("human similarity review is invalid")
    similarity_reviewer = named_human(
        human_review["reviewer_identity"], "similarity reviewer"
    )
    similarity_reviewed_at = _parse_canonical_utc(
        human_review["reviewed_at_utc"], "similarity reviewed_at_utc"
    )
    captions = _object(manifest["caption_policy"], "caption_policy")
    _exact(
        captions,
        {
            "record_sha256",
            "reviewer_identity",
            "reviewed_at_utc",
            "decision",
            "training_row_ids_sha256",
            "evaluation_row_ids_sha256",
            "assertions",
        },
        "caption_policy",
    )
    caption_assertions = _object(captions["assertions"], "caption assertions")
    _exact(
        caption_assertions,
        {
            "manual_review_complete",
            "captions_match_images",
            "trigger_usage_consistent",
            "evaluation_leakage_absent",
        },
        "caption assertions",
    )
    if (
        captions["decision"] != "approved"
        or any(value is not True for value in caption_assertions.values())
        or captions["training_row_ids_sha256"]
        != krea_provenance.canonical_sha256(sorted(row["row_id"] for row in train_rows))
        or captions["evaluation_row_ids_sha256"]
        != krea_provenance.canonical_sha256(sorted(row["row_id"] for row in eval_rows))
    ):
        raise ValueError("caption policy does not cover the exact fixture rows")
    named_human(captions["reviewer_identity"], "caption reviewer")
    caption_reviewed_at = _parse_canonical_utc(
        captions["reviewed_at_utc"], "caption reviewed_at_utc"
    )
    for key in (
        "record_sha256",
        "training_row_ids_sha256",
        "evaluation_row_ids_sha256",
    ):
        if not isinstance(captions[key], str) or not _SHA256.fullmatch(captions[key]):
            raise ValueError(f"caption_policy.{key} is invalid")
    training_archive = _object(manifest["training_archive"], "training_archive")
    _exact(training_archive, {"sha256", "bytes"}, "training_archive")
    if (
        not isinstance(training_archive["sha256"], str)
        or not _SHA256.fullmatch(training_archive["sha256"])
        or isinstance(training_archive["bytes"], bool)
        or not isinstance(training_archive["bytes"], int)
        or training_archive["bytes"] <= 0
    ):
        raise ValueError("training_archive digest is invalid")
    source = _object(manifest["source_rights"], "source_rights")
    _exact(
        source,
        {
            "owner",
            "locator",
            "retrieved_at_utc",
            "record_sha256",
            "reviewer_identity",
            "reviewed_at_utc",
            "decision",
            "assertions",
        },
        "source_rights",
    )
    for key in ("owner", "locator"):
        _text(source[key], f"source_rights.{key}")
    source_retrieved_at = _parse_canonical_utc(
        source["retrieved_at_utc"], "source_rights.retrieved_at_utc"
    )
    source_reviewed_at = _parse_canonical_utc(
        source["reviewed_at_utc"], "source_rights.reviewed_at_utc"
    )
    named_human(source["reviewer_identity"], "rights reviewer")
    rights_assertions = _object(source["assertions"], "source rights assertions")
    _exact(
        rights_assertions,
        {
            "lawful_access",
            "calibration_use_allowed",
            "redistribution_reviewed",
            "sensitive_content_absent",
        },
        "source rights assertions",
    )
    if source["decision"] != "approved_for_calibration" or any(
        value is not True for value in rights_assertions.values()
    ):
        raise ValueError("source rights record did not approve calibration use")
    if not isinstance(source["record_sha256"], str) or not _SHA256.fullmatch(
        source["record_sha256"]
    ):
        raise ValueError("source rights digest is invalid")
    if any(
        reviewed_at < source_retrieved_at
        for reviewed_at in (
            source_reviewed_at,
            caption_reviewed_at,
            similarity_reviewed_at,
        )
    ):
        raise ValueError("fixture review evidence predates source retrieval")
    if _human_identity_key(
        manifest["preparer_identity"], "preparer_identity"
    ) == _human_identity_key(similarity_reviewer, "similarity reviewer"):
        raise ValueError("fixture similarity reviewer is not independent from preparer")
    _text(manifest["trigger_token"], "trigger_token")
    tool = _object(manifest["tool_identity"], "tool_identity")
    _exact(
        tool,
        {
            "fixture_module_sha256",
            "dataset_identity_module_sha256",
            "god_commit",
            "enumerator_module",
            "enumerator_qualname",
            "enumerator_source_sha256",
            "enumerator_callable_sha256",
            "extensions",
            "perceptual_hash",
        },
        "tool_identity",
    )
    for key in (
        "fixture_module_sha256",
        "dataset_identity_module_sha256",
        "enumerator_source_sha256",
        "enumerator_callable_sha256",
    ):
        if not isinstance(tool[key], str) or not _SHA256.fullmatch(tool[key]):
            raise ValueError(f"tool_identity.{key} is invalid")
    if not isinstance(tool["god_commit"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", tool["god_commit"]
    ):
        raise ValueError("tool_identity.god_commit is invalid")
    if (
        not isinstance(tool["extensions"], list)
        or not tool["extensions"]
        or any(not isinstance(item, str) or not item for item in tool["extensions"])
        or tool["perceptual_hash"]
        != "rgb-luma-average-hash-8x8-bilinear-after-exif-transpose"
    ):
        raise ValueError("tool identity enumerator/perceptual-hash contract is invalid")
    archive_identity = _object(
        manifest["training_archive_identity"], "training_archive_identity"
    )
    _exact(
        archive_identity, {"format", "members", "sha256"}, "training_archive_identity"
    )
    if archive_identity["format"] != "zip" or archive_identity[
        "sha256"
    ] != krea_provenance.canonical_sha256(
        {"format": archive_identity["format"], "members": archive_identity["members"]}
    ):
        raise ValueError("training archive identity is invalid")
    expected_members = sorted(
        [
            {
                "path": row["image"],
                "sha256": row["image_sha256"],
                "bytes": row["image_bytes"],
            }
            for row in manifest["training_dataset_identity"]["rows"]
        ]
        + [
            {
                "path": row["prompt"],
                "sha256": row["prompt_sha256"],
                "bytes": row["prompt_bytes"],
            }
            for row in manifest["training_dataset_identity"]["rows"]
        ],
        key=lambda row: row["path"],
    )
    if archive_identity["members"] != expected_members:
        raise ValueError("training archive members differ from training dataset")
    return manifest


def _agent_actor(value: Any, label: str) -> dict[str, str]:
    """Validate an explicitly non-human review actor.

    Agent identities are deliberately separate from :func:`named_human`.
    They establish stable record linkage and review-instance separation, not
    human identity, legal agency, or cryptographic authentication.
    """

    actor = _object(value, label)
    _exact(
        actor,
        {
            "actor_class",
            "actor_id",
            "display_name",
            "role",
            "review_instance_id",
            "identity_assurance",
        },
        label,
    )
    if (
        actor["actor_class"] != "agent"
        or not isinstance(actor["actor_id"], str)
        or not _SAFE_ID.fullmatch(actor["actor_id"])
        or not isinstance(actor["review_instance_id"], str)
        or not _SAFE_ID.fullmatch(actor["review_instance_id"])
        or actor["identity_assurance"] != _AGENT_IDENTITY_ASSURANCE
    ):
        raise ValueError(f"{label} is not an explicit agent identity")
    _text(actor["display_name"], f"{label}.display_name")
    _text(actor["role"], f"{label}.role")
    return dict(actor)


def _validate_agent_governance(manifest: dict[str, Any]) -> dict[str, Any]:
    governance = _object(manifest.get("governance"), "fixture governance")
    _exact(
        governance,
        {
            "mode",
            "policy_sha256",
            "governance_amendment",
            "owner_ratification",
            "source_package",
            "candidate_manifest_sha256",
            "surface_agent_review",
            "independent_agent_review",
            "preparer_actor",
            "accountable_owner_identity",
            "owner_identity_assurance",
            "agent_review_is_not_human_review",
            "independent_human_review_performed",
            "claim_limit",
        },
        "fixture governance",
    )
    if governance["mode"] != _AGENT_GOVERNANCE_MODE:
        raise ValueError("unsupported fixture governance mode")
    for key in (
        "policy_sha256",
        "candidate_manifest_sha256",
    ):
        if not isinstance(governance[key], str) or not _SHA256.fullmatch(
            governance[key]
        ):
            raise ValueError(f"fixture governance {key} is invalid")
    for key, semantic_key in (
        ("governance_amendment", "amendment_sha256"),
        ("owner_ratification", "ratification_sha256"),
        ("source_package", "package_sha256"),
    ):
        binding = _object(governance[key], f"fixture governance {key}")
        _exact(binding, {"file_sha256", semantic_key}, f"fixture governance {key}")
        for digest_key in ("file_sha256", semantic_key):
            if not isinstance(binding[digest_key], str) or not _SHA256.fullmatch(
                binding[digest_key]
            ):
                raise ValueError(f"fixture governance {key}.{digest_key} is invalid")
    for key in ("surface_agent_review", "independent_agent_review"):
        binding = _object(governance[key], f"fixture governance {key}")
        _exact(
            binding,
            {"file_sha256", "review_sha256", "actor"},
            f"fixture governance {key}",
        )
        for digest_key in ("file_sha256", "review_sha256"):
            if not isinstance(binding[digest_key], str) or not _SHA256.fullmatch(
                binding[digest_key]
            ):
                raise ValueError(f"fixture governance {key}.{digest_key} is invalid")
    preparer = _agent_actor(governance["preparer_actor"], "fixture preparer actor")
    surface_actor = _agent_actor(
        governance["surface_agent_review"]["actor"], "surface review actor"
    )
    independent_actor = _agent_actor(
        governance["independent_agent_review"]["actor"],
        "independent review actor",
    )
    if (
        preparer["review_instance_id"] == surface_actor["review_instance_id"]
        or preparer["review_instance_id"] == independent_actor["review_instance_id"]
        or surface_actor["review_instance_id"]
        == independent_actor["review_instance_id"]
    ):
        raise ValueError("agent review instances are not distinct")
    owner = named_human(
        governance["accountable_owner_identity"], "accountable_owner_identity"
    )
    if (
        governance["owner_identity_assurance"] != _OWNER_IDENTITY_ASSURANCE
        or governance["agent_review_is_not_human_review"] is not True
        or governance["independent_human_review_performed"] is not False
        or governance["claim_limit"]
        != (
            "owner-ratified-agent-evidence; Stage-1 is staged-host-venv "
            "discovery-only, not production/release/tournament evidence; "
            "Stage-2 requires a separate Forge commit and fresh named-owner "
            "ratification"
        )
    ):
        raise ValueError("fixture governance overstates its assurance")
    return {**governance, "accountable_owner_identity": owner}


def _schema2_legacy_projection(manifest: dict[str, Any]) -> dict[str, Any]:
    """Project schema 2 onto schema 1 solely for byte-integrity validation.

    The placeholder names exist only inside this in-memory projection.  They
    are never evidence and never written to an artifact.  This lets schema 2
    reuse the mature dataset/archive/leakage validator without laundering an
    agent identity through ``named_human()``.
    """

    projected = deepcopy(manifest)
    projected.pop("governance")
    projected["schema"] = 1
    projected["preparer_identity"] = "Governance Projection Preparer"
    projected["source_rights"]["reviewer_identity"] = "Governance Projection Rights"
    projected["caption_policy"]["reviewer_identity"] = "Governance Projection Caption"
    projected["near_duplicate_policy"]["human_similarity_review"][
        "reviewer_identity"
    ] = "Governance Projection Similarity"
    projected["near_duplicate_policy"]["human_similarity_review"][
        "method"
    ] = "named-human-exhaustive-pair-review-plus-pinned-ahash"
    # Schema 2 preserves threshold-8 machine flags that were explicitly
    # adjudicated by the bound agent surface review.  Schema 1 cannot express
    # adjudicated flags and requires the raw list to be empty, so this
    # in-memory compatibility projection uses the strict exact-pHash threshold
    # after schema-2 validation has already recomputed and checked the real
    # threshold-8 report.
    projected["near_duplicate_policy"]["maximum_hamming_distance"] = 0
    projected["near_duplicate_policy"]["report"] = _duplicates(
        projected["training_rows"],
        projected["evaluation_rows"],
        threshold=0,
        group_disjoint_fields=tuple(
            projected["near_duplicate_policy"]["group_disjoint_fields"]
        ),
    )
    projected["near_duplicate_policy"]["report_sha256"] = (
        krea_provenance.canonical_sha256(projected["near_duplicate_policy"]["report"])
    )
    body = {key: value for key, value in projected.items() if key != "manifest_sha256"}
    projected["manifest_sha256"] = krea_provenance.canonical_sha256(body)
    return projected


def _validate_agent_governed_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "schema",
        "kind",
        "concept_id",
        "experimental_role",
        "trigger_token",
        "caption_policy",
        "source_rights",
        "preparer_identity",
        "training_archive",
        "training_archive_identity",
        "training_dataset_identity",
        "evaluation_dataset_identity",
        "training_dataset_shape_sha256",
        "evaluation_dataset_shape_sha256",
        "training_rows",
        "evaluation_rows",
        "tool_identity",
        "near_duplicate_policy",
        "governance",
        "manifest_sha256",
    }
    _exact(manifest, expected, "agent-governed fixture manifest")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest["schema"] != 2
        or manifest["kind"] != "forge-krea-curated-fixture"
        or manifest["manifest_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("agent-governed fixture manifest digest mismatch")
    governance = _validate_agent_governance(manifest)
    surface_binding = governance["surface_agent_review"]
    independent_binding = governance["independent_agent_review"]
    surface_actor = _agent_actor(surface_binding["actor"], "surface review actor")
    independent_actor = _agent_actor(
        independent_binding["actor"], "independent review actor"
    )
    preparer_actor = _agent_actor(
        governance["preparer_actor"], "fixture preparer actor"
    )
    if (
        manifest["preparer_identity"] != preparer_actor["display_name"]
        or manifest["source_rights"]["reviewer_identity"]
        != surface_actor["display_name"]
        or manifest["caption_policy"]["reviewer_identity"]
        != surface_actor["display_name"]
        or manifest["near_duplicate_policy"]["human_similarity_review"][
            "reviewer_identity"
        ]
        != surface_actor["display_name"]
        or manifest["near_duplicate_policy"]["human_similarity_review"]["method"]
        != "owner-ratified-agent-review-plus-pinned-ahash"
        or independent_actor["actor_id"] == surface_actor["actor_id"]
    ):
        raise ValueError("fixture actors differ from the explicit governance record")
    near = _object(manifest["near_duplicate_policy"], "near_duplicate_policy")
    if near.get("maximum_hamming_distance") != 8:
        raise ValueError("agent-governed fixture must preserve threshold-8 screening")
    actual_report = _duplicates(
        manifest["training_rows"],
        manifest["evaluation_rows"],
        threshold=8,
        group_disjoint_fields=tuple(near.get("group_disjoint_fields", [])),
    )
    if (
        near.get("report") != actual_report
        or near.get("report_sha256") != krea_provenance.canonical_sha256(actual_report)
        or actual_report["exact_matches"]
        or actual_report["cross_split_group_matches"]
    ):
        raise ValueError("agent-governed fixture duplicate report is invalid")
    _validate_legacy_manifest(_schema2_legacy_projection(manifest))
    return manifest


def _validate_stage2_boundary_derived_manifest(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Validate a truthful, mechanics-only D1/D2 boundary derivation.

    The embedded source governance and approval are deliberately kept under
    ``source_*`` names.  They prove the source fixture was admitted; they are
    never represented as authorizing the new boundary role.  Reconstructing
    and validating the exact source schema-2 document makes every dataset,
    archive, caption, rights, and leakage byte immutable across derivation.
    """

    expected = {
        "schema",
        "kind",
        "concept_id",
        "experimental_role",
        "trigger_token",
        "caption_policy",
        "source_rights",
        "preparer_identity",
        "training_archive",
        "training_archive_identity",
        "training_dataset_identity",
        "evaluation_dataset_identity",
        "training_dataset_shape_sha256",
        "evaluation_dataset_shape_sha256",
        "training_rows",
        "evaluation_rows",
        "tool_identity",
        "near_duplicate_policy",
        "source_governance",
        "source_approval",
        "boundary_derivation",
        "manifest_sha256",
    }
    _exact(manifest, expected, "Stage-2 boundary-derived fixture manifest")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest["schema"] != 3
        or manifest["kind"] != "forge-krea-curated-fixture"
        or manifest["manifest_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("Stage-2 boundary-derived manifest digest mismatch")

    role = manifest["experimental_role"]
    source_role = _STAGE2_BOUNDARY_SOURCE_ROLES.get(role)
    if source_role is None:
        raise ValueError("schema-3 fixture must use an exact Stage-2 boundary role")
    _validate_role_counts(
        role, len(manifest["training_rows"]), len(manifest["evaluation_rows"])
    )

    derivation = _object(manifest["boundary_derivation"], "boundary derivation")
    _exact(
        derivation,
        {
            "mode",
            "source_role",
            "source_manifest_file_sha256",
            "source_manifest_sha256",
            "source_approval_file_sha256",
            "source_approval_sha256",
            "public_freeze_binding",
            "actor",
            "fixture_bytes_changed",
            "group_evidence_changed",
            "source_governance_is_evidence_only",
            "source_governance_authorizes_boundary",
            "source_approval_authorizes_boundary",
            "fresh_stage2_owner_ratification_required",
            "boundary_admission_authorized",
            "gpu_execution_authorized",
            "science_selection_input",
            "claim_limit",
        },
        "boundary derivation",
    )
    freeze = _object(
        derivation["public_freeze_binding"], "boundary public-freeze binding"
    )
    _exact(
        freeze,
        {"path", "file_sha256", "binding_sha256", "commit_sha1"},
        "boundary public-freeze binding",
    )
    _text(freeze["path"], "boundary public-freeze path")
    if not isinstance(freeze["commit_sha1"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", freeze["commit_sha1"]
    ):
        raise ValueError("boundary public-freeze commit is invalid")
    for key in (
        "source_manifest_file_sha256",
        "source_manifest_sha256",
        "source_approval_file_sha256",
        "source_approval_sha256",
    ):
        _digest(derivation[key], f"boundary derivation {key}")
    for key in ("file_sha256", "binding_sha256"):
        _digest(freeze[key], f"boundary public-freeze {key}")
    _agent_actor(derivation["actor"], "boundary derivation actor")
    if (
        derivation["mode"] != _STAGE2_BOUNDARY_DERIVATION_MODE
        or derivation["source_role"] != source_role
        or derivation["fixture_bytes_changed"] is not False
        or derivation["group_evidence_changed"] is not False
        or derivation["source_governance_is_evidence_only"] is not True
        or derivation["source_governance_authorizes_boundary"] is not False
        or derivation["source_approval_authorizes_boundary"] is not False
        or derivation["fresh_stage2_owner_ratification_required"] is not True
        or derivation["boundary_admission_authorized"] is not False
        or derivation["gpu_execution_authorized"] is not False
        or derivation["science_selection_input"] is not False
        or derivation["claim_limit"] != _STAGE2_BOUNDARY_DERIVATION_CLAIM_LIMIT
    ):
        raise ValueError("Stage-2 boundary derivation overstates its authority")

    # Recover the exact admitted source fixture.  In particular, a D2-derived
    # large boundary retains the stronger play/accession grouping and its
    # already-reviewed report; dropping those fields would fail this equality.
    source = deepcopy(manifest)
    source.pop("boundary_derivation")
    source["governance"] = source.pop("source_governance")
    source.pop("source_approval")
    source["schema"] = 2
    source["experimental_role"] = source_role
    source["manifest_sha256"] = derivation["source_manifest_sha256"]
    _validate_agent_governed_manifest(source)
    if (
        hashlib.sha256(krea_provenance.canonical_bytes(source) + b"\n").hexdigest()
        != derivation["source_manifest_file_sha256"]
    ):
        raise ValueError("boundary derivation source-manifest file binding drifted")
    approval = _validate_agent_governed_approval(
        manifest["source_approval"], fixture_manifest=source
    )
    if (
        approval["approval_sha256"] != derivation["source_approval_sha256"]
        or hashlib.sha256(krea_provenance.canonical_bytes(approval) + b"\n").hexdigest()
        != derivation["source_approval_file_sha256"]
    ):
        raise ValueError("boundary derivation source-approval binding drifted")
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Dispatch without weakening the legacy named-human contract."""

    manifest = _object(manifest, "fixture manifest")
    if manifest.get("schema") == 1:
        return _validate_legacy_manifest(manifest)
    if manifest.get("schema") == 2:
        return _validate_agent_governed_manifest(manifest)
    if manifest.get("schema") == 3:
        return _validate_stage2_boundary_derived_manifest(manifest)
    raise ValueError("unsupported fixture manifest")


def build_approval(
    manifest: dict[str, Any],
    *,
    reviewer_identity: str,
    approved_at_utc: str,
) -> dict[str, Any]:
    validate_manifest(manifest)
    reviewer = named_human(reviewer_identity, "reviewer_identity")
    if _human_identity_key(reviewer, "reviewer_identity") == _human_identity_key(
        manifest["preparer_identity"], "preparer_identity"
    ):
        raise ValueError("fixture reviewer must be independent from the preparer")
    approved_at = canonical_utc(approved_at_utc, "approved_at_utc")
    if _parse_canonical_utc(approved_at, "approved_at_utc") < _latest_fixture_evidence(
        [manifest]
    ):
        raise ValueError("fixture approval predates fixture preparation evidence")
    body = {
        "schema": 1,
        "kind": "forge-krea-fixture-approval",
        "fixture_manifest_sha256": manifest["manifest_sha256"],
        "reviewer_identity": reviewer,
        "approved_at_utc": approved_at,
        "decision": "approved",
        "rights_reviewed": True,
        "concept_role_and_counts_reviewed": True,
        "disjointness_reviewed": True,
        "claim_limit": "split-integrity-only-not-competitiveness",
    }
    return {**body, "approval_sha256": krea_provenance.canonical_sha256(body)}


def _validate_legacy_approval(
    approval: dict[str, Any], *, fixture_manifest: dict[str, Any]
) -> dict[str, Any]:
    _validate_legacy_manifest(fixture_manifest)
    approval = _object(approval, "fixture approval")
    _exact(
        approval,
        {
            "schema",
            "kind",
            "fixture_manifest_sha256",
            "reviewer_identity",
            "approved_at_utc",
            "decision",
            "rights_reviewed",
            "concept_role_and_counts_reviewed",
            "disjointness_reviewed",
            "claim_limit",
            "approval_sha256",
        },
        "fixture approval",
    )
    body = {key: value for key, value in approval.items() if key != "approval_sha256"}
    if (
        approval["schema"] != 1
        or approval["kind"] != "forge-krea-fixture-approval"
        or approval["fixture_manifest_sha256"] != fixture_manifest["manifest_sha256"]
        or approval["decision"] != "approved"
        or approval["rights_reviewed"] is not True
        or approval["concept_role_and_counts_reviewed"] is not True
        or approval["disjointness_reviewed"] is not True
        or approval["claim_limit"] != "split-integrity-only-not-competitiveness"
        or approval["approval_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("fixture approval is not a valid binding approval")
    named_human(approval["reviewer_identity"], "reviewer_identity")
    approved_at = _parse_canonical_utc(approval["approved_at_utc"], "approved_at_utc")
    if _human_identity_key(
        approval["reviewer_identity"], "reviewer_identity"
    ) == _human_identity_key(
        fixture_manifest["preparer_identity"], "preparer_identity"
    ):
        raise ValueError("fixture approval reviewer is not independent")
    if approved_at < _latest_fixture_evidence([fixture_manifest]):
        raise ValueError("fixture approval predates fixture preparation evidence")
    return approval


def build_agent_governed_approval(
    manifest: dict[str, Any],
    *,
    technical_reviewer_actor: dict[str, Any],
    accountable_owner_identity: str,
    approved_at_utc: str,
) -> dict[str, Any]:
    """Build an admission input; only the later envelope admits a fixture."""

    _validate_agent_governed_manifest(manifest)
    governance = manifest["governance"]
    actor = _agent_actor(technical_reviewer_actor, "technical reviewer actor")
    if actor != governance["independent_agent_review"]["actor"]:
        raise ValueError("fixture approval actor differs from governance")
    owner = named_human(accountable_owner_identity, "accountable_owner_identity")
    if owner != governance["accountable_owner_identity"]:
        raise ValueError("fixture approval owner differs from ratified governance")
    approved_at = canonical_utc(approved_at_utc, "approved_at_utc")
    body = {
        "schema": 2,
        "kind": "forge-krea-owner-ratified-fixture-approval",
        "fixture_manifest_sha256": manifest["manifest_sha256"],
        "governance_policy_sha256": governance["policy_sha256"],
        "owner_ratification_sha256": governance["owner_ratification"][
            "ratification_sha256"
        ],
        "accountable_owner_identity": owner,
        "owner_identity_assurance": _OWNER_IDENTITY_ASSURANCE,
        "technical_reviewer_actor": actor,
        "approved_at_utc": approved_at,
        "decision": "approved_as_discovery_admission_input",
        "assertions": {
            "package_and_fixture_bytes_rederived": True,
            "agent_review_evidence_bound": True,
            "agents_are_not_humans": True,
            "independent_human_review_performed": False,
            "owner_ratification_bound": True,
        },
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": (
            "fixture-integrity-and-governance-input-only-not-independent-human-"
            "review-competitiveness-or-gpu-authorization"
        ),
    }
    return {**body, "approval_sha256": krea_provenance.canonical_sha256(body)}


def _validate_agent_governed_approval(
    approval: dict[str, Any], *, fixture_manifest: dict[str, Any]
) -> dict[str, Any]:
    _validate_agent_governed_manifest(fixture_manifest)
    approval = _object(approval, "agent-governed fixture approval")
    _exact(
        approval,
        {
            "schema",
            "kind",
            "fixture_manifest_sha256",
            "governance_policy_sha256",
            "owner_ratification_sha256",
            "accountable_owner_identity",
            "owner_identity_assurance",
            "technical_reviewer_actor",
            "approved_at_utc",
            "decision",
            "assertions",
            "admission_authorized",
            "gpu_execution_authorized",
            "claim_limit",
            "approval_sha256",
        },
        "agent-governed fixture approval",
    )
    body = {key: value for key, value in approval.items() if key != "approval_sha256"}
    governance = fixture_manifest["governance"]
    if (
        approval["schema"] != 2
        or approval["kind"] != "forge-krea-owner-ratified-fixture-approval"
        or approval["fixture_manifest_sha256"] != fixture_manifest["manifest_sha256"]
        or approval["governance_policy_sha256"] != governance["policy_sha256"]
        or approval["owner_ratification_sha256"]
        != governance["owner_ratification"]["ratification_sha256"]
        or approval["accountable_owner_identity"]
        != governance["accountable_owner_identity"]
        or approval["owner_identity_assurance"] != _OWNER_IDENTITY_ASSURANCE
        or approval["decision"] != "approved_as_discovery_admission_input"
        or approval["assertions"]
        != {
            "package_and_fixture_bytes_rederived": True,
            "agent_review_evidence_bound": True,
            "agents_are_not_humans": True,
            "independent_human_review_performed": False,
            "owner_ratification_bound": True,
        }
        or approval["admission_authorized"] is not False
        or approval["gpu_execution_authorized"] is not False
        or approval["claim_limit"]
        != (
            "fixture-integrity-and-governance-input-only-not-independent-human-"
            "review-competitiveness-or-gpu-authorization"
        )
        or approval["approval_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("agent-governed fixture approval is invalid")
    actor = _agent_actor(
        approval["technical_reviewer_actor"], "technical reviewer actor"
    )
    if actor != governance["independent_agent_review"]["actor"]:
        raise ValueError("agent-governed approval reviewer differs from governance")
    named_human(approval["accountable_owner_identity"], "accountable_owner_identity")
    _parse_canonical_utc(approval["approved_at_utc"], "approved_at_utc")
    return approval


def validate_approval(
    approval: dict[str, Any], *, fixture_manifest: dict[str, Any]
) -> dict[str, Any]:
    approval = _object(approval, "fixture approval")
    if approval.get("schema") == 1:
        return _validate_legacy_approval(approval, fixture_manifest=fixture_manifest)
    if approval.get("schema") == 2:
        return _validate_agent_governed_approval(
            approval, fixture_manifest=fixture_manifest
        )
    raise ValueError("unsupported fixture approval")


def _cross_fixture_pairs(fixtures: list[dict[str, Any]]) -> list[list[str]]:
    rows_by_role = {
        fixture["experimental_role"]: sorted(
            row["row_id"]
            for row in fixture["training_rows"] + fixture["evaluation_rows"]
        )
        for fixture in fixtures
    }
    pairs: list[list[str]] = []
    roles = sorted(rows_by_role)
    for index, left_role in enumerate(roles):
        for right_role in roles[index + 1 :]:
            for left in rows_by_role[left_role]:
                for right in rows_by_role[right_role]:
                    pairs.append([f"{left_role}:{left}", f"{right_role}:{right}"])
    return pairs


def _fixture_evidence_times(fixture: dict[str, Any]) -> list[tuple[str, datetime]]:
    """Return every timestamp whose evidence must predate a later approval."""

    return [
        (
            "source_rights.retrieved_at_utc",
            _parse_canonical_utc(
                fixture["source_rights"]["retrieved_at_utc"],
                "source_rights.retrieved_at_utc",
            ),
        ),
        (
            "source_rights.reviewed_at_utc",
            _parse_canonical_utc(
                fixture["source_rights"]["reviewed_at_utc"],
                "source_rights.reviewed_at_utc",
            ),
        ),
        (
            "caption_policy.reviewed_at_utc",
            _parse_canonical_utc(
                fixture["caption_policy"]["reviewed_at_utc"],
                "caption_policy.reviewed_at_utc",
            ),
        ),
        (
            "near_duplicate_policy.human_similarity_review.reviewed_at_utc",
            _parse_canonical_utc(
                fixture["near_duplicate_policy"]["human_similarity_review"][
                    "reviewed_at_utc"
                ],
                "human_similarity_review.reviewed_at_utc",
            ),
        ),
    ]


def _latest_fixture_evidence(fixtures: list[dict[str, Any]]) -> datetime:
    evidence = [
        timestamp
        for fixture in fixtures
        for _, timestamp in _fixture_evidence_times(fixture)
    ]
    if not evidence:  # pragma: no cover - callers require fixtures.
        raise ValueError("fixture evidence timestamps are absent")
    return max(evidence)


def _cross_fixture_automated_nonoverlap(
    fixtures: list[dict[str, Any]],
) -> list[list[str]]:
    """Recompute the complete machine-verifiable all-six nonoverlap surface."""

    expected_roles = set(_CROSS_FIXTURE_ROLES)
    if len(fixtures) != len(expected_roles):
        raise ValueError("cross-fixture review requires exactly D1,D2,C1,C2,C3,C4")
    for fixture in fixtures:
        validate_manifest(fixture)
    roles = {fixture["experimental_role"] for fixture in fixtures}
    if roles != expected_roles:
        raise ValueError("cross-fixture review requires exactly D1,D2,C1,C2,C3,C4")

    seen_concepts: set[str] = set()
    seen_triggers: set[str] = set()
    seen: dict[str, tuple[str, str]] = {}
    all_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for fixture in fixtures:
        concept = fixture["concept_id"]
        trigger = fixture["trigger_token"]
        if concept in seen_concepts:
            raise ValueError(f"duplicate fixture concept: {concept}")
        if trigger in seen_triggers:
            raise ValueError("fixtures must have unique roles and trigger tokens")
        seen_concepts.add(concept)
        seen_triggers.add(trigger)
        for split in ("training_rows", "evaluation_rows"):
            for row in fixture[split]:
                for key in (
                    "content_sha256",
                    "image_sha256",
                    "decoded_pixels_sha256",
                    "normalized_caption_sha256",
                ):
                    token = f"{key}:{row[key]}"
                    if token in seen:
                        raise ValueError(
                            f"fixtures overlap by {key}: {seen[token]} vs "
                            f"{(concept, row['row_id'])}"
                        )
                    seen[token] = (concept, row["row_id"])
                all_rows.append((fixture, row))
    for index, (left_fixture, left) in enumerate(all_rows):
        for right_fixture, right in all_rows[index + 1 :]:
            if left_fixture["concept_id"] == right_fixture["concept_id"]:
                continue
            distance = (
                int(left["perceptual_hash64"], 16) ^ int(right["perceptual_hash64"], 16)
            ).bit_count()
            threshold = max(
                left_fixture["near_duplicate_policy"]["maximum_hamming_distance"],
                right_fixture["near_duplicate_policy"]["maximum_hamming_distance"],
            )
            if distance <= threshold:
                raise ValueError("fixtures overlap by perceptual near-duplicate")
            shared_groups = sorted(
                key
                for key in set(
                    left_fixture["near_duplicate_policy"]["group_disjoint_fields"]
                )
                | set(right_fixture["near_duplicate_policy"]["group_disjoint_fields"])
                if key in left["group_identity"] and key in right["group_identity"]
                if left["group_identity"][key] == right["group_identity"][key]
            )
            if shared_groups:
                raise ValueError(
                    f"fixtures overlap by grouped identity: {shared_groups}"
                )
    return _cross_fixture_pairs(fixtures)


def _agent_cross_assertions() -> dict[str, bool]:
    return {
        "all_six_fixture_manifests_validated": True,
        "automated_cross_fixture_nonoverlap_passed": True,
        "agent_review_is_not_human_review": True,
        "independent_human_review_performed": False,
        "d2_selector_key_accessed": False,
        "c1c4_content_or_paths_disclosed": False,
        "c1c4_remain_in_sealed_custody": True,
    }


def _agent_cross_visual_scope(
    visual_reviewed_pairs: list[list[str]], *, all_pairs: list[list[str]]
) -> dict[str, Any]:
    index = {tuple(pair): offset for offset, pair in enumerate(all_pairs)}
    normalized: list[list[str]] = []
    offsets: list[int] = []
    for pair in visual_reviewed_pairs:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(item, str) for item in pair)
            or tuple(pair) not in index
        ):
            raise ValueError("agent visual-review pair is outside the all-six universe")
        normalized.append(list(pair))
        offsets.append(index[tuple(pair)])
    if offsets != sorted(set(offsets)):
        raise ValueError(
            "agent visual-review pairs must be unique and canonically ordered"
        )
    if not normalized:
        method = "not-performed"
        coverage = "none"
    else:
        method = "agent-visual-review-not-human-review"
        coverage = "complete" if normalized == all_pairs else "targeted"
    return {
        "performed": bool(normalized),
        "method": method,
        "coverage": coverage,
        "reviewed_pair_count": len(normalized),
        "reviewed_pairs_sha256": krea_provenance.canonical_sha256(normalized),
    }


def _agent_cross_automated_scope(all_pairs: list[list[str]]) -> dict[str, Any]:
    return {
        "method": "exact-hash-group-identity-and-pinned-ahash",
        "coverage": "all-cross-role-pairs",
        "reviewed_pair_count": len(all_pairs),
        "reviewed_pairs_sha256": krea_provenance.canonical_sha256(all_pairs),
        "maximum_hamming_distance_policy": "max-of-paired-fixture-thresholds",
    }


def _validate_cross_agent_distinct(
    actor: dict[str, str], parent_actor: dict[str, str]
) -> None:
    if actor["role"] != _SEALED_CUSTODIAN_ROLE:
        raise ValueError("cross-fixture agent is not the sealed custodian")
    if (
        actor["actor_id"] == parent_actor["actor_id"]
        or actor["review_instance_id"] == parent_actor["review_instance_id"]
    ):
        raise ValueError("sealed custodian must be distinct from the parent reviewer")


def build_agent_cross_fixture_review(
    fixtures: list[dict[str, Any]],
    *,
    actor: dict[str, Any],
    parent_independent_review: dict[str, Any],
    owner_ratification_sha256: str,
    acceptance_request_sha256: str,
    reviewed_at_utc: str,
    visual_reviewed_pairs: list[list[str]],
) -> dict[str, Any]:
    """Build an all-six agent review inside sealed custody without a D2 key."""

    actor = _agent_actor(actor, "sealed custodian actor")
    parent = _object(parent_independent_review, "parent independent review")
    _exact(parent, {"review_sha256", "actor"}, "parent independent review")
    parent_actor = _agent_actor(parent["actor"], "parent independent review actor")
    parent = {
        "review_sha256": _digest(parent["review_sha256"], "parent review SHA-256"),
        "actor": parent_actor,
    }
    _validate_cross_agent_distinct(actor, parent_actor)
    owner_ratification_sha256 = _digest(
        owner_ratification_sha256, "owner ratification SHA-256"
    )
    acceptance_request_sha256 = _digest(
        acceptance_request_sha256, "acceptance request SHA-256"
    )
    all_pairs = _cross_fixture_automated_nonoverlap(fixtures)
    manifests = {
        fixture["experimental_role"]: fixture["manifest_sha256"]
        for fixture in sorted(fixtures, key=lambda value: value["experimental_role"])
    }
    for fixture in fixtures:
        if fixture.get("schema") != 2:
            continue
        governance = fixture["governance"]
        if (
            governance["owner_ratification"]["ratification_sha256"]
            != owner_ratification_sha256
            or governance["independent_agent_review"]["review_sha256"]
            != parent["review_sha256"]
            or governance["independent_agent_review"]["actor"] != parent_actor
        ):
            raise ValueError("agent fixture governance differs from the cross review")
        for bound_actor in (
            governance["preparer_actor"],
            governance["surface_agent_review"]["actor"],
            governance["independent_agent_review"]["actor"],
        ):
            bound_actor = _agent_actor(bound_actor, "fixture governance actor")
            if (
                actor["actor_id"] == bound_actor["actor_id"]
                or actor["review_instance_id"] == bound_actor["review_instance_id"]
            ):
                raise ValueError("sealed custodian is not distinct from fixture actors")
    visual_pairs = [list(pair) for pair in visual_reviewed_pairs]
    visual_scope = _agent_cross_visual_scope(visual_pairs, all_pairs=all_pairs)
    automated_scope = _agent_cross_automated_scope(all_pairs)
    reviewed_at_utc = canonical_utc(reviewed_at_utc, "agent cross review time")
    if _parse_canonical_utc(
        reviewed_at_utc, "agent cross review time"
    ) < _latest_fixture_evidence(fixtures):
        raise ValueError("agent cross-fixture review predates fixture evidence")
    body = {
        "schema": 2,
        "kind": _AGENT_CROSS_REVIEW_KIND,
        "actor": actor,
        "parent_independent_review": parent,
        "owner_ratification_sha256": owner_ratification_sha256,
        "acceptance_request_sha256": acceptance_request_sha256,
        "fixture_manifest_sha256s": manifests,
        "reviewed_at_utc": reviewed_at_utc,
        "review_scope": {
            "automated": automated_scope,
            "visual": visual_scope,
        },
        "reviewed_pairs": all_pairs,
        "reviewed_pair_count": len(all_pairs),
        "reviewed_pairs_sha256": krea_provenance.canonical_sha256(all_pairs),
        "visual_reviewed_pairs": visual_pairs,
        "flagged_pairs": [],
        "decision": "passed_agent_automated_cross_fixture_nonoverlap",
        "assertions": _agent_cross_assertions(),
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": _AGENT_CROSS_CLAIM_LIMIT,
    }
    review = {
        **body,
        "review_sha256": krea_provenance.canonical_sha256(body),
    }
    validate_agent_cross_fixture_review(review, fixtures=fixtures)
    return review


def validate_agent_cross_fixture_review(
    record: dict[str, Any], *, fixtures: list[dict[str, Any]]
) -> dict[str, Any]:
    record = _object(record, "cross-fixture agent review")
    _exact(
        record,
        {
            "schema",
            "kind",
            "actor",
            "parent_independent_review",
            "owner_ratification_sha256",
            "acceptance_request_sha256",
            "fixture_manifest_sha256s",
            "reviewed_at_utc",
            "review_scope",
            "reviewed_pairs",
            "reviewed_pair_count",
            "reviewed_pairs_sha256",
            "visual_reviewed_pairs",
            "flagged_pairs",
            "decision",
            "assertions",
            "admission_authorized",
            "gpu_execution_authorized",
            "claim_limit",
            "review_sha256",
        },
        "cross-fixture agent review",
    )
    body = {key: value for key, value in record.items() if key != "review_sha256"}
    actor = _agent_actor(record["actor"], "sealed custodian actor")
    parent = _object(record["parent_independent_review"], "parent independent review")
    _exact(parent, {"review_sha256", "actor"}, "parent independent review")
    parent_actor = _agent_actor(parent["actor"], "parent independent review actor")
    _digest(parent["review_sha256"], "parent review SHA-256")
    _digest(record["owner_ratification_sha256"], "owner ratification SHA-256")
    _digest(record["acceptance_request_sha256"], "acceptance request SHA-256")
    _validate_cross_agent_distinct(actor, parent_actor)
    all_pairs = _cross_fixture_automated_nonoverlap(fixtures)
    manifests = {
        fixture["experimental_role"]: fixture["manifest_sha256"]
        for fixture in sorted(fixtures, key=lambda value: value["experimental_role"])
    }
    visual_pairs = record["visual_reviewed_pairs"]
    if not isinstance(visual_pairs, list):
        raise ValueError("agent visual-review pairs must be a list")
    expected_scope = {
        "automated": _agent_cross_automated_scope(all_pairs),
        "visual": _agent_cross_visual_scope(visual_pairs, all_pairs=all_pairs),
    }
    reviewed_at = _parse_canonical_utc(
        canonical_utc(record["reviewed_at_utc"], "agent cross review time"),
        "agent cross review time",
    )
    if reviewed_at < _latest_fixture_evidence(fixtures):
        raise ValueError("agent cross-fixture review predates fixture evidence")
    if (
        record["schema"] != 2
        or record["kind"] != _AGENT_CROSS_REVIEW_KIND
        or record["fixture_manifest_sha256s"] != manifests
        or record["review_scope"] != expected_scope
        or record["reviewed_pairs"] != all_pairs
        or record["reviewed_pair_count"] != len(all_pairs)
        or record["reviewed_pairs_sha256"]
        != krea_provenance.canonical_sha256(all_pairs)
        or record["flagged_pairs"] != []
        or record["decision"] != "passed_agent_automated_cross_fixture_nonoverlap"
        or record["assertions"] != _agent_cross_assertions()
        or record["admission_authorized"] is not False
        or record["gpu_execution_authorized"] is not False
        or record["claim_limit"] != _AGENT_CROSS_CLAIM_LIMIT
        or record["review_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("cross-fixture agent review is invalid")
    return record


def _agent_cross_fixture_binding_body(
    record: dict[str, Any], *, review_file_sha256: str
) -> dict[str, Any]:
    return {
        "schema": 2,
        "kind": _AGENT_CROSS_BINDING_KIND,
        "review_file_sha256": review_file_sha256,
        "review_sha256": record["review_sha256"],
        "fixture_manifest_sha256s": record["fixture_manifest_sha256s"],
        "fixture_manifest_set_sha256": krea_provenance.canonical_sha256(
            record["fixture_manifest_sha256s"]
        ),
        "actor": record["actor"],
        "parent_independent_review": record["parent_independent_review"],
        "owner_ratification_sha256": record["owner_ratification_sha256"],
        "acceptance_request_sha256": record["acceptance_request_sha256"],
        "reviewed_at_utc": record["reviewed_at_utc"],
        "reviewed_pair_count": record["reviewed_pair_count"],
        "reviewed_pairs_sha256": record["reviewed_pairs_sha256"],
        "review_scope": record["review_scope"],
        "decision": record["decision"],
        "assertions": record["assertions"],
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": record["claim_limit"],
    }


def build_agent_cross_fixture_binding(
    review_record: dict[str, Any],
    *,
    fixtures: list[dict[str, Any]],
    review_file_sha256: str,
) -> dict[str, Any]:
    validate_agent_cross_fixture_review(review_record, fixtures=fixtures)
    review_file_sha256 = _digest(review_file_sha256, "agent review file SHA-256")
    body = _agent_cross_fixture_binding_body(
        review_record, review_file_sha256=review_file_sha256
    )
    binding = {
        **body,
        "binding_sha256": krea_provenance.canonical_sha256(body),
    }
    validate_agent_cross_fixture_binding(
        binding,
        fixtures=fixtures,
        review_record=review_record,
        review_file_sha256=review_file_sha256,
    )
    return binding


def validate_agent_cross_fixture_binding(
    binding: dict[str, Any],
    *,
    fixtures: list[dict[str, Any]],
    review_record: dict[str, Any],
    review_file_sha256: str,
) -> dict[str, Any]:
    validate_agent_cross_fixture_review(review_record, fixtures=fixtures)
    review_file_sha256 = _digest(review_file_sha256, "agent review file SHA-256")
    expected = _agent_cross_fixture_binding_body(
        review_record, review_file_sha256=review_file_sha256
    )
    binding = _object(binding, "cross-fixture agent review binding")
    _exact(
        binding,
        set(expected) | {"binding_sha256"},
        "cross-fixture agent review binding",
    )
    if {
        key: value for key, value in binding.items() if key != "binding_sha256"
    } != expected or binding["binding_sha256"] != krea_provenance.canonical_sha256(
        expected
    ):
        raise ValueError("cross-fixture agent binding does not bind the exact review")
    return binding


def validate_agent_cross_fixture_binding_digest_only(
    binding: dict[str, Any],
    *,
    fixture_manifest_sha256s: dict[str, str],
    parent_independent_review: dict[str, Any],
    owner_ratification_sha256: str,
    acceptance_request_sha256: str,
) -> dict[str, Any]:
    """Validate the non-secret binding exported by a sealed custodian."""

    binding = _object(binding, "cross-fixture agent review binding")
    expected_keys = {
        "schema",
        "kind",
        "review_file_sha256",
        "review_sha256",
        "fixture_manifest_sha256s",
        "fixture_manifest_set_sha256",
        "actor",
        "parent_independent_review",
        "owner_ratification_sha256",
        "acceptance_request_sha256",
        "reviewed_at_utc",
        "reviewed_pair_count",
        "reviewed_pairs_sha256",
        "review_scope",
        "decision",
        "assertions",
        "admission_authorized",
        "gpu_execution_authorized",
        "claim_limit",
    }
    _exact(binding, expected_keys | {"binding_sha256"}, "cross-fixture agent binding")
    body = {key: value for key, value in binding.items() if key != "binding_sha256"}
    actor = _agent_actor(binding["actor"], "sealed custodian actor")
    parent = _object(parent_independent_review, "parent independent review")
    _exact(parent, {"review_sha256", "actor"}, "parent independent review")
    parent_actor = _agent_actor(parent["actor"], "parent independent review actor")
    _digest(parent["review_sha256"], "parent review SHA-256")
    _validate_cross_agent_distinct(actor, parent_actor)
    if set(fixture_manifest_sha256s) != set(_CROSS_FIXTURE_ROLES):
        raise ValueError("agent cross binding does not cover all six fixtures")
    for role, digest in fixture_manifest_sha256s.items():
        _digest(digest, f"{role} manifest SHA-256")
    for key in (
        "review_file_sha256",
        "review_sha256",
        "fixture_manifest_set_sha256",
        "reviewed_pairs_sha256",
        "binding_sha256",
    ):
        _digest(binding[key], f"agent cross binding {key}")
    owner_ratification_sha256 = _digest(
        owner_ratification_sha256, "owner ratification SHA-256"
    )
    acceptance_request_sha256 = _digest(
        acceptance_request_sha256, "acceptance request SHA-256"
    )
    scope = _object(binding["review_scope"], "agent cross review scope")
    _exact(scope, {"automated", "visual"}, "agent cross review scope")
    automated = _object(scope["automated"], "automated cross review scope")
    visual = _object(scope["visual"], "visual cross review scope")
    _exact(
        automated,
        {
            "method",
            "coverage",
            "reviewed_pair_count",
            "reviewed_pairs_sha256",
            "maximum_hamming_distance_policy",
        },
        "automated cross review scope",
    )
    _exact(
        visual,
        {
            "performed",
            "method",
            "coverage",
            "reviewed_pair_count",
            "reviewed_pairs_sha256",
        },
        "visual cross review scope",
    )
    count = binding["reviewed_pair_count"]
    visual_count = visual["reviewed_pair_count"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or automated
        != {
            "method": "exact-hash-group-identity-and-pinned-ahash",
            "coverage": "all-cross-role-pairs",
            "reviewed_pair_count": count,
            "reviewed_pairs_sha256": binding["reviewed_pairs_sha256"],
            "maximum_hamming_distance_policy": "max-of-paired-fixture-thresholds",
        }
        or isinstance(visual_count, bool)
        or not isinstance(visual_count, int)
        or visual_count < 0
        or visual_count > count
    ):
        raise ValueError("cross-fixture agent review scope is invalid")
    _digest(visual["reviewed_pairs_sha256"], "visual reviewed pairs SHA-256")
    expected_visual = (
        {
            "performed": False,
            "method": "not-performed",
            "coverage": "none",
        }
        if visual_count == 0
        else {
            "performed": True,
            "method": "agent-visual-review-not-human-review",
            "coverage": "complete" if visual_count == count else "targeted",
        }
    )
    if any(visual[key] != value for key, value in expected_visual.items()):
        raise ValueError("cross-fixture visual scope overstates its coverage")
    if (
        binding["schema"] != 2
        or binding["kind"] != _AGENT_CROSS_BINDING_KIND
        or binding["fixture_manifest_sha256s"] != fixture_manifest_sha256s
        or binding["fixture_manifest_set_sha256"]
        != krea_provenance.canonical_sha256(fixture_manifest_sha256s)
        or binding["parent_independent_review"] != parent
        or binding["owner_ratification_sha256"] != owner_ratification_sha256
        or binding["acceptance_request_sha256"] != acceptance_request_sha256
        or binding["decision"] != "passed_agent_automated_cross_fixture_nonoverlap"
        or binding["assertions"] != _agent_cross_assertions()
        or binding["admission_authorized"] is not False
        or binding["gpu_execution_authorized"] is not False
        or binding["claim_limit"] != _AGENT_CROSS_CLAIM_LIMIT
        or binding["binding_sha256"] != krea_provenance.canonical_sha256(body)
    ):
        raise ValueError("cross-fixture agent review binding is invalid")
    canonical_utc(binding["reviewed_at_utc"], "agent cross binding time")
    return binding


def _cross_fixture_binding_body(
    record: dict[str, Any], *, review_file_sha256: str
) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "forge-krea-cross-fixture-review-binding",
        "review_file_sha256": review_file_sha256,
        "fixture_manifest_sha256s": record["fixture_manifest_sha256s"],
        "fixture_manifest_set_sha256": krea_provenance.canonical_sha256(
            record["fixture_manifest_sha256s"]
        ),
        "reviewer_identity": record["reviewer_identity"],
        "reviewer_identity_assurance": _HUMAN_IDENTITY_ASSURANCE,
        "reviewed_at_utc": record["reviewed_at_utc"],
        "reviewed_pair_count": len(record["reviewed_pairs"]),
        "reviewed_pairs_sha256": krea_provenance.canonical_sha256(
            record["reviewed_pairs"]
        ),
        "decision": record["decision"],
        "claim_limit": record["claim_limit"],
    }


def validate_cross_fixture_binding(
    binding: dict[str, Any],
    *,
    fixtures: list[dict[str, Any]],
    review_record: dict[str, Any],
    review_file_sha256: str,
) -> dict[str, Any]:
    """Validate the compact attestation a scoring consumer can bind directly."""

    binding = _object(binding, "cross-fixture review binding")
    _exact(
        binding,
        {
            "schema",
            "kind",
            "review_file_sha256",
            "fixture_manifest_sha256s",
            "fixture_manifest_set_sha256",
            "reviewer_identity",
            "reviewer_identity_assurance",
            "reviewed_at_utc",
            "reviewed_pair_count",
            "reviewed_pairs_sha256",
            "decision",
            "claim_limit",
            "binding_sha256",
        },
        "cross-fixture review binding",
    )
    review_file_sha256 = _text(review_file_sha256, "review_file_sha256")
    if not _SHA256.fullmatch(review_file_sha256):
        raise ValueError("review_file_sha256 is invalid")
    validate_cross_fixture_review(review_record, fixtures=fixtures)
    expected = _cross_fixture_binding_body(
        review_record, review_file_sha256=review_file_sha256
    )
    if {
        key: value for key, value in binding.items() if key != "binding_sha256"
    } != expected or binding["binding_sha256"] != krea_provenance.canonical_sha256(
        expected
    ):
        raise ValueError("cross-fixture review binding does not bind the exact review")
    return binding


def validate_cross_fixture_review(
    record: dict[str, Any], *, fixtures: list[dict[str, Any]]
) -> dict[str, Any]:
    """Require named-human review of every cross-fixture row pair."""

    record = _object(record, "cross-fixture human review")
    _exact(
        record,
        {
            "schema",
            "kind",
            "fixture_manifest_sha256s",
            "reviewer_identity",
            "reviewed_at_utc",
            "decision",
            "reviewed_pairs",
            "flagged_pairs",
            "claim_limit",
        },
        "cross-fixture human review",
    )
    for fixture in fixtures:
        validate_manifest(fixture)
    expected_roles = set(_CROSS_FIXTURE_ROLES)
    roles = {fixture["experimental_role"] for fixture in fixtures}
    if roles != expected_roles or len(fixtures) != len(expected_roles):
        raise ValueError("cross-fixture review requires exactly D1,D2,C1,C2,C3,C4")
    expected_manifests = {
        fixture["experimental_role"]: fixture["manifest_sha256"]
        for fixture in sorted(fixtures, key=lambda item: item["experimental_role"])
    }
    expected_pairs = _cross_fixture_pairs(fixtures)
    reviewer = named_human(record["reviewer_identity"], "cross-fixture reviewer")
    reviewer_key = _human_identity_key(reviewer, "cross-fixture reviewer")
    excluded_reviewers = {
        _human_identity_key(identity, label)
        for fixture in fixtures
        for identity, label in (
            (fixture["preparer_identity"], "fixture preparer"),
            (
                fixture["near_duplicate_policy"]["human_similarity_review"][
                    "reviewer_identity"
                ],
                "fixture similarity reviewer",
            ),
            (
                fixture["caption_policy"]["reviewer_identity"],
                "fixture caption reviewer",
            ),
            (
                fixture["source_rights"]["reviewer_identity"],
                "fixture rights reviewer",
            ),
        )
    }
    if reviewer_key in excluded_reviewers:
        raise ValueError("cross-fixture reviewer is not independent")
    reviewed_at = _parse_canonical_utc(
        record["reviewed_at_utc"], "cross-fixture reviewed_at_utc"
    )
    if reviewed_at < _latest_fixture_evidence(fixtures):
        raise ValueError("cross-fixture review predates fixture preparation evidence")
    if (
        record["schema"] != 1
        or record["kind"] != "forge-krea-cross-fixture-human-similarity-review"
        or record["fixture_manifest_sha256s"] != expected_manifests
        or record["decision"] != "passed"
        or record["reviewed_pairs"] != expected_pairs
        or record["flagged_pairs"] != []
        or record["claim_limit"] != "cross-fixture-nonoverlap-only"
    ):
        raise ValueError("cross-fixture human review is incomplete or did not pass")
    return record


def cross_fixture_disjoint(
    fixtures: list[dict[str, Any]], *, human_review_record: Path
) -> dict[str, Any]:
    """Reject automated and independently-reviewed overlap across all fixtures."""

    seen_concepts: set[str] = set()
    seen_roles: set[str] = set()
    seen_triggers: set[str] = set()
    seen: dict[str, tuple[str, str]] = {}
    all_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for fixture in fixtures:
        validate_manifest(fixture)
        concept = fixture["concept_id"]
        if concept in seen_concepts:
            raise ValueError(f"duplicate fixture concept: {concept}")
        seen_concepts.add(concept)
        role = fixture["experimental_role"]
        trigger = fixture["trigger_token"]
        if role in seen_roles or trigger in seen_triggers:
            raise ValueError("fixtures must have unique roles and trigger tokens")
        seen_roles.add(role)
        seen_triggers.add(trigger)
        for split in ("training_rows", "evaluation_rows"):
            for row in fixture[split]:
                for key in (
                    "content_sha256",
                    "image_sha256",
                    "decoded_pixels_sha256",
                    "normalized_caption_sha256",
                ):
                    token = f"{key}:{row[key]}"
                    if token in seen:
                        raise ValueError(
                            f"fixtures overlap by {key}: {seen[token]} vs "
                            f"{(concept, row['row_id'])}"
                        )
                    seen[token] = (concept, row["row_id"])
                all_rows.append((fixture, row))
    for index, (left_fixture, left) in enumerate(all_rows):
        for right_fixture, right in all_rows[index + 1 :]:
            if left_fixture["concept_id"] == right_fixture["concept_id"]:
                continue
            distance = (
                int(left["perceptual_hash64"], 16) ^ int(right["perceptual_hash64"], 16)
            ).bit_count()
            threshold = max(
                left_fixture["near_duplicate_policy"]["maximum_hamming_distance"],
                right_fixture["near_duplicate_policy"]["maximum_hamming_distance"],
            )
            if distance <= threshold:
                raise ValueError("fixtures overlap by perceptual near-duplicate")
            shared_groups = sorted(
                key
                for key in set(
                    left_fixture["near_duplicate_policy"]["group_disjoint_fields"]
                )
                | set(right_fixture["near_duplicate_policy"]["group_disjoint_fields"])
                if key in left["group_identity"] and key in right["group_identity"]
                if left["group_identity"][key] == right["group_identity"][key]
            )
            if shared_groups:
                raise ValueError(
                    f"fixtures overlap by grouped identity: {shared_groups}"
                )
    review, review_file_sha = _canonical_record(
        human_review_record, "cross-fixture human review"
    )
    validate_cross_fixture_review(review, fixtures=fixtures)
    body = _cross_fixture_binding_body(review, review_file_sha256=review_file_sha)
    binding = {**body, "binding_sha256": krea_provenance.canonical_sha256(body)}
    validate_cross_fixture_binding(
        binding,
        fixtures=fixtures,
        review_record=review,
        review_file_sha256=review_file_sha,
    )
    # The path is descriptive and deliberately excluded from the portable
    # binding digest.  Consumers bind the record bytes and all six fixture
    # self-digests, never a mutable path string.
    return {
        "path": str(_safe_file(human_review_record, "cross-fixture human review")),
        # Compatibility alias for early Day-0 callers; the named field below
        # is the one included in the portable binding digest.
        "sha256": review_file_sha,
        **binding,
    }
