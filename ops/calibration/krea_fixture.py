#!/usr/bin/env python3
"""Pre-GPU Krea fixture curation and independent approval contracts.

The manifest is deliberately produced before any arm is trained.  It binds the
exact byte/order identity consumed by the G.O.D evaluator and a second,
filename-independent content identity used to reject leakage and duplicate
rows.  Paths are descriptive only; execution code must stage and re-hash the
approved bytes before use.
"""

from __future__ import annotations

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
_ROLE_COUNTS = {
    "D1": ((18, 24), (24, 24)),
    "D2": ((36, 48), (40, 40)),
    "C1": ((18, 24), (24, 24)),
    "C2": ((18, 24), (24, 24)),
    "C3": ((36, 48), (40, 40)),
    "C4": ((36, 48), (40, 40)),
}
_SMALL_ROLES = frozenset({"D1", "C1", "C2"})
_LARGE_ROLES = frozenset({"D2", "C3", "C4"})
_CROSS_FIXTURE_ROLES = ("D1", "D2", "C1", "C2", "C3", "C4")
_HUMAN_IDENTITY_ASSURANCE = (
    "named-human-string-self-assertion-not-cryptographic-authentication"
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
    """Enforce the frozen small/large fixture classes without inventing counts."""

    if role not in _ROLE_COUNTS:
        raise ValueError("experimental_role must be D1, D2, or C1-C4")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in (training_count, evaluation_count)
    ):
        raise ValueError("fixture counts must be non-negative integers")
    train_range, eval_range = _ROLE_COUNTS[role]
    if not train_range[0] <= training_count <= train_range[1]:
        raise ValueError(f"{role} training count is outside {train_range}")
    if not eval_range[0] <= evaluation_count <= eval_range[1]:
        raise ValueError(f"{role} evaluation count is outside {eval_range}")
    if role in (_SMALL_ROLES - {"D1"}) and (train_range, eval_range) != (
        (18, 24),
        (24, 24),
    ):
        raise AssertionError("C1/C2 must remain frozen to the small fixture class")
    if role in (_LARGE_ROLES - {"D2"}) and (train_range, eval_range) != (
        (36, 48),
        (40, 40),
    ):
        raise AssertionError("C3/C4 must remain frozen to the large fixture class")


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
        from PIL import Image, __version__ as pillow_version
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
        rgb = opened.convert("RGB")
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
    group_keys = {
        "source_id",
        "creator_id",
        "burst_id",
        "scene_id",
        "play_root_id",
        "human_similarity_cluster_id",
    }
    normalized_groups = {}
    for image_name, raw_group in row_groups.items():
        group = _object(raw_group, f"row group {image_name}")
        _exact(group, group_keys, f"row group {image_name}")
        normalized_groups[image_name] = {
            key: _text(group[key], f"row group {image_name}.{key}")
            for key in sorted(group_keys)
        }
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
    if role not in _ROLE_COUNTS:
        raise ValueError("experimental_role must be D1, D2, or C1-C4")
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
    group_field_vocabulary = {
        "source_id",
        "creator_id",
        "burst_id",
        "scene_id",
        "play_root_id",
        "human_similarity_cluster_id",
    }
    group_disjoint_fields = metadata["group_disjoint_fields"]
    if (
        not isinstance(group_disjoint_fields, list)
        or group_disjoint_fields != sorted(set(group_disjoint_fields))
        or not {
            "burst_id",
            "scene_id",
            "play_root_id",
            "human_similarity_cluster_id",
        }.issubset(group_disjoint_fields)
        or any(field not in group_field_vocabulary for field in group_disjoint_fields)
    ):
        raise ValueError("group_disjoint_fields does not satisfy the leakage policy")
    training_identity, training_rows = _rows(
        training_dir,
        list_supported_images=list_supported_images,
        extensions=extensions,
        row_groups=_object(metadata["training_row_groups"], "training_row_groups"),
    )
    evaluation_identity, evaluation_rows = _rows(
        evaluation_dir,
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
        "perceptual_hash": "rgb-luma-average-hash-8x8-bilinear",
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


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
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
    if role not in _ROLE_COUNTS:
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
                or row["width"] != evaluator_row["image_width"]
                or row["height"] != evaluator_row["image_height"]
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
            group = _object(row["group_identity"], f"{split} fixture group")
            _exact(
                group,
                {
                    "source_id",
                    "creator_id",
                    "burst_id",
                    "scene_id",
                    "play_root_id",
                    "human_similarity_cluster_id",
                },
                f"{split} fixture group",
            )
            for key, value in group.items():
                _text(value, f"{split} fixture group.{key}")
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
    group_fields = near["group_disjoint_fields"]
    if (
        not isinstance(group_fields, list)
        or group_fields != sorted(set(group_fields))
        or not {
            "burst_id",
            "scene_id",
            "play_root_id",
            "human_similarity_cluster_id",
        }.issubset(group_fields)
    ):
        raise ValueError("fixture group-disjointness policy is invalid")
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
        or tool["perceptual_hash"] != "rgb-luma-average-hash-8x8-bilinear"
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


def validate_approval(
    approval: dict[str, Any], *, fixture_manifest: dict[str, Any]
) -> dict[str, Any]:
    validate_manifest(fixture_manifest)
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
