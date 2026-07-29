#!/usr/bin/env python3
"""Build deterministic, explicitly non-admitted D1/D2 fixture candidates.

This tool sits between the reviewed source split and :mod:`krea_fixture`'s
named-human-approved manifest.  It stages the exact selected bytes, creates
training/evaluation captions under one frozen policy, emits a per-file rights
ledger and an exhaustive selected-pair screen, and binds the result in a
candidate manifest.  It never manufactures a human countersign, fixture
approval, admission authorization, or GPU authorization.

The candidate package is deliberately immutable.  Later reviewers write
separate approval records that bind its digests; they do not edit the package
in place.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import unicodedata
import zipfile
from typing import Any

try:
    from . import krea_provenance
    from . import krea_review_split
except ImportError:  # pragma: no cover - direct script execution.
    import krea_provenance  # type: ignore[no-redef]
    import krea_review_split  # type: ignore[no-redef]


_KIND = "forge-krea-pre-admission-fixture-candidate"
_PACKAGE_KIND = "forge-krea-pre-admission-package"
_RIGHTS_KIND = "forge-krea-selected-per-file-rights-ledger"
_CAPTION_KIND = "forge-krea-caption-candidate-ledger"
_SIMILARITY_KIND = "forge-krea-selected-pair-similarity-evidence"
_REVIEW_REQUEST_KIND = "forge-krea-bundled-review-request"
_SCHEMA = 1
_TRIGGERS = {"D1": "snuoqr4aypo5", "D2": "stmqqfgnrzhq"}
_TOKENIZER_EVIDENCE_SHA256 = (
    "95a9f0736a316314d1223234449f6e9ab5fe1f5c2e389777c3ef984429cf7314"
)
_TOKENIZER_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
_TOKENIZER_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
_EXPECTED_COUNTS = {"D1": (18, 24), "D2": (36, 40)}
_BASE_GROUP_FIELDS = (
    "source_id",
    "creator_id",
    "burst_id",
    "scene_id",
    "play_root_id",
    "human_similarity_cluster_id",
)
_D2_GROUP_FIELDS = ("play_component_id", "accession_family_id")
_CONCEPTS = {
    "D1": "fontana-del-moro",
    "D2": "tsukioka-kogyo-nogaku-zue",
}
_CLAIM_LIMIT = (
    "selected-byte-caption-rights-and-similarity-candidate-only-"
    "named-human-countersign-independent-approval-admission-and-gpu-"
    "authorization-remain-required"
)
_PREPARER_ACTOR = "Codex (implementation agent; not a named-human reviewer)"
_PENDING_GATES = [
    "named_human_per_file_rights_countersign",
    "named_human_caption_and_trigger_countersign",
    (
        "named_human_exhaustive_similarity_countersign_including_explicit_"
        "D1_label_pair_binding"
    ),
    "independent_reviewer_fixture_approval",
    "final_forge-krea-curated-fixture_manifest",
    "all_six_fixture_cross_review",
]
_REQUESTED_COUNTERSIGNS = [
    "per-file-rights-and-obligations",
    "caption-image-match-and-trigger-contract",
    (
        "accept-exhaustive-machine-similarity-screen-and-visually-review-"
        "explicit-D1-targeted-source-pairs"
    ),
]
_REQUESTED_INDEPENDENT_REVIEW = [
    "source-split-and-count-bindings",
    "rights-caption-similarity-countersign-independence",
    "fixture-admission-after-final-manifest-rederivation",
]
_D1_FORBIDDEN_CAPTION_TERMS = (
    "fontana",
    "del moro",
    "piazza navona",
    "bernini",
    "wikimedia",
    "commons",
)
_D2_FORBIDDEN_CAPTION_TERMS = (
    "tsukioka",
    "kogyo",
    "kogyō",
    "nogaku zue",
    "pictures of no performances",
    "art institute of chicago",
)
_SIMILARITY_DESCRIPTIONS = [
    "mascaron-dolphins spout close-up vs standing Moor wide view",
    "full Moor statue frontal vs kneeling Triton conch close-up",
    "Triton frontal close-up vs rim mascaron close-up",
    "Triton close-up vs whole-fountain wide basin view",
    "rim mascaron close-up vs whole-fountain wide view",
    "Moor front three-quarter vs Moor full rear view",
    "different sides: orange-building backdrop vs pale palazzo, mascaron foreground",
    "Moor rear view vs Moor front three-quarter",
    "rim mascaron landscape vs Moor rear portrait",
    "mascaron detail vs ultra-wide piazza with church",
    "tight Moor rear portrait vs wide piazza landscape",
    "rear view with obelisk vs Palazzo-Pamphilj-facing frontal wide",
    "wide piazza rear view vs sunlit mascaron close-up",
    "whole-fountain wide vs mascaron close-up detail",
    "stacked Moor-plus-Triton frontal vs single Triton side close-up",
]

_MATERIALIZATION_KEYS = {
    "schema",
    "kind",
    "experimental_role",
    "concept_id",
    "source_harvest_sha256",
    "source_policy_sha256",
    "source_enrichment",
    "retrieval_authorization_sha256",
    "retrieval_authorization_file_sha256",
    "retrieval_owner_identity",
    "retrieved_at_utc",
    "download_policy",
    "rows",
    "admission_state",
    "claim_limit",
    "human_gates",
    "human_approvals",
    "fixture_manifest_created",
    "gpu_execution_authorized",
    "materialization_sha256",
}
_MATERIALIZATION_ROW_KEYS = {
    "source_id",
    "relative_path",
    "source_url",
    "bytes",
    "sha256",
    "sha1",
    "mime",
    "etag",
    "last_modified",
}


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


def _reject_governance_escalation(value: Any, label: str = "record") -> None:
    """Reject authorization claims anywhere inside a pre-admission package."""

    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"admission_authorized", "gpu_execution_authorized"}:
                if child is not False:
                    raise ValueError(f"{label}.{key} must remain false")
            _reject_governance_escalation(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_governance_escalation(child, f"{label}[{index}]")


def _text(value: Any, label: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{label} must be a string")
    if value != unicodedata.normalize("NFC", value):
        raise ValueError(f"{label} must use NFC Unicode")
    return value


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
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real directory: {path}")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    path = _safe_file(path, "bound file")
    before = path.stat()
    digest = _file_sha256(path)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"file changed while hashed: {path}")
    return {"bytes": after.st_size, "sha256": digest}


def _canonical_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = _safe_file(path, label)
    raw = path.read_bytes()
    try:
        value = _object(json.loads(raw), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid JSON") from exc
    if raw != krea_provenance.canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline")
    return value, hashlib.sha256(raw).hexdigest()


def _write_canonical(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = krea_provenance.canonical_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_bytes(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _copy_bound(source: Path, destination: Path, expected_sha256: str) -> None:
    source = _safe_file(source, "selected source image")
    if _file_sha256(source) != expected_sha256:
        raise ValueError(f"selected source bytes differ from review: {source}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                digest.update(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if digest.hexdigest() != expected_sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"source changed while copied: {source}")


def _normalize_spaces(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def _descriptor(role: str, row: dict[str, Any]) -> str:
    if role == "D1":
        note = _normalize_spaces(_text(row["review_notes"], "review_notes", empty=True))
        note = re.sub(
            r"(?:commons title mislabels as Neptune|mislabeled Neptune)",
            "",
            note,
            flags=re.IGNORECASE,
        )
        note = re.sub(r"\s*;\s*;\s*", "; ", note).strip(" ;.")
        if not note:
            note = "full fountain in a centered architectural composition"
        descriptor = f"a bronze-and-marble fountain, {note}"
        forbidden = _D1_FORBIDDEN_CAPTION_TERMS
    else:
        play = _text(row["group_identity"]["play_root_id"], "play_root_id")
        play = re.sub(r"[^a-z0-9]+", " ", play.casefold()).strip()
        if not play:
            raise ValueError("D2 play root normalizes to empty")
        descriptor = f"a Japanese woodblock print depicting the {play} theatrical scene"
        forbidden = _D2_FORBIDDEN_CAPTION_TERMS
    descriptor = _normalize_spaces(descriptor).strip(" .") + "."
    folded = descriptor.casefold()
    present = [term for term in forbidden if term in folded]
    if present:
        raise ValueError(
            f"caption contains forbidden source/proper-name terms: {present}"
        )
    return descriptor


def _caption_bytes(role: str, split: str, row: dict[str, Any]) -> bytes:
    descriptor = _descriptor(role, row)
    if split == "training":
        caption = descriptor
    elif split == "evaluation":
        caption = f"{_TRIGGERS[role]}, {descriptor}"
    else:  # pragma: no cover - internal caller controls split.
        raise AssertionError(split)
    folded = caption.casefold()
    trigger_count = len(
        re.findall(rf"(?<![a-z0-9]){_TRIGGERS[role]}(?![a-z0-9])", folded)
    )
    if (split == "training" and trigger_count != 0) or (
        split == "evaluation" and trigger_count != 1
    ):
        raise ValueError("caption trigger placement is invalid")
    return caption.encode("utf-8") + b"\n"


def _rights_obligations(row: dict[str, Any]) -> dict[str, Any]:
    decision = row["rights_decision"]
    if decision == "approve_cc_by_obligations_recorded":
        if (
            not row["license_url"]
            or "creativecommons.org/licenses/by/" not in row["license_url"]
        ):
            raise ValueError("CC BY row lacks a CC BY license URL")
        return {
            "attribution_required": True,
            "share_alike_required": False,
            "preserve_creator_title_source_and_license": True,
            "indicate_modifications_if_distributed": True,
            "third_party_rights_not_warranted": True,
        }
    if decision == "approve_pd_or_cc0":
        return {
            "attribution_required": False,
            "share_alike_required": False,
            "preserve_provenance_as_project_policy": True,
            "third_party_rights_not_warranted": True,
        }
    raise ValueError(f"selected row has an unapproved rights decision: {decision}")


def _row_by_id(review: dict[str, Any], role: str) -> dict[str, dict[str, Any]]:
    result = {row["source_id"]: row for row in review["records"][role]}
    if len(result) != len(review["records"][role]):
        raise ValueError(f"{role} review repeats a source ID")
    return result


def _group_projection(role: str, row: dict[str, Any]) -> dict[str, str]:
    fields = _BASE_GROUP_FIELDS + (_D2_GROUP_FIELDS if role == "D2" else ())
    group = _object(row["group_identity"], f"{role} source group identity")
    projected = {field: _text(group.get(field), field) for field in fields}
    if role == "D1" and set(projected) != set(_BASE_GROUP_FIELDS):
        raise AssertionError("D1 group projection unexpectedly includes D2-only fields")
    return projected


def _materialized_by_id(
    root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    manifest_path = root / "materialization.json"
    manifest, file_sha = _canonical_json(manifest_path, "materialization manifest")
    _exact(manifest, _MATERIALIZATION_KEYS, "materialization manifest")
    body = {
        key: value for key, value in manifest.items() if key != "materialization_sha256"
    }
    if (
        manifest.get("kind") != "forge-krea-source-candidate-materialization"
        or manifest.get("schema") != 2
        or manifest.get("materialization_sha256")
        != krea_provenance.canonical_sha256(body)
        or manifest.get("admission_state") != "candidate_unreviewed"
        or manifest.get("gpu_execution_authorized") is not False
    ):
        raise ValueError("materialization boundary is invalid")
    raw_rows = manifest["rows"]
    if not isinstance(raw_rows, list) or raw_rows != sorted(
        raw_rows, key=lambda row: row.get("source_id", "")
    ):
        raise ValueError("materialization rows are not a sorted array")
    rows: dict[str, dict[str, Any]] = {}
    for raw_row in raw_rows:
        row = _object(raw_row, "materialization row")
        _exact(row, _MATERIALIZATION_ROW_KEYS, "materialization row")
        source_id = _text(row["source_id"], "materialization source_id")
        relative = _text(row["relative_path"], "materialization relative_path")
        parsed = PurePosixPath(relative)
        if (
            parsed.is_absolute()
            or len(parsed.parts) != 2
            or parsed.parts[0] != "images"
            or parsed.name != relative.split("/", 1)[1]
            or Path(parsed.name).stem != source_id
            or Path(parsed.name).suffix.lower()
            not in {".jpg", ".jpeg", ".png", ".webp"}
        ):
            raise ValueError("materialization row path is not canonical")
        if source_id in rows:
            raise ValueError("materialization repeats a source ID")
        if (
            isinstance(row["bytes"], bool)
            or not isinstance(row["bytes"], int)
            or row["bytes"] <= 0
            or not isinstance(row["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        ):
            raise ValueError("materialization row byte identity is invalid")
        rows[source_id] = row
    return manifest, rows, file_sha


def _validate_split(
    split: dict[str, Any], *, role: str, review: dict[str, Any]
) -> tuple[list[str], list[str]]:
    training = split.get("training_source_ids")
    evaluation = split.get("evaluation_source_ids")
    expected_train, expected_eval = _EXPECTED_COUNTS[role]
    if (
        split.get("schema") != 1
        or split.get("kind") != "forge-krea-source-split-plan"
        or split.get("experimental_role") != role
        or split.get("source_review_sha256") != review["review_sha256"]
        or split.get("split_sha256")
        != krea_provenance.canonical_sha256(
            {key: value for key, value in split.items() if key != "split_sha256"}
        )
        or not isinstance(training, list)
        or not isinstance(evaluation, list)
        or len(training) != expected_train
        or len(evaluation) != expected_eval
        or training != sorted(set(training))
        or evaluation != sorted(set(evaluation))
        or set(training) & set(evaluation)
        or split.get("admission_authorized") is not False
        or split.get("gpu_execution_authorized") is not False
    ):
        raise ValueError(f"{role} split boundary is invalid")
    known = _row_by_id(review, role)
    if any(source_id not in known for source_id in training + evaluation):
        raise ValueError(f"{role} split selects an unknown source")
    if any(
        known[source_id].get("disposition") != "CANDIDATE_ONLY_NOT_ADMITTED"
        for source_id in training + evaluation
    ):
        raise ValueError(f"{role} split selects a rejected or escalated source")
    return training, evaluation


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def _similarity_evidence(
    role: str,
    *,
    rows: dict[str, dict[str, Any]],
    selected_ids: list[str],
    review: dict[str, Any],
    split_file_sha256: str,
    screen_file_sha256: str,
) -> dict[str, Any]:
    prior = {
        _pair_key(item["left_source_id"], item["right_source_id"]): item
        for item in review["queued_pair_reviews"][role]
    }
    pairs: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    counts = {"prior_reviewed": 0, "machine_clear": 0, "new_flags": 0}
    for left_id, right_id in itertools.combinations(sorted(selected_ids), 2):
        left = rows[left_id]
        right = rows[right_id]
        distance = (
            int(left["perceptual_hash64"], 16) ^ int(right["perceptual_hash64"], 16)
        ).bit_count()
        shared = sorted(
            field
            for field in ("accession_family_id", "burst_id")
            if left["group_identity"].get(field) == right["group_identity"].get(field)
        )
        key = (left_id, right_id)
        if key in prior:
            queue = prior[key]
            if queue["relationship_decision"] != "distinct":
                raise ValueError("selected prior-reviewed pair was not distinct")
            disposition = "prior_reviewed_distinct"
            evidence = {"queue_pair_id": queue["pair_id"]}
            counts["prior_reviewed"] += 1
        elif distance > 8 and not shared:
            disposition = "machine_clear"
            evidence = {
                "rule": "perceptual_hamming_gt_8_and_no_accession_or_burst_collision"
            }
            counts["machine_clear"] += 1
        else:
            disposition = "targeted_visual_adjudication_pending_binding"
            evidence = {
                "screen_prose_verdict": "distinct",
                "screen_pair_mapping": (
                    "inferred_from-lexicographically-sorted-new-flag-order-and-"
                    "matching-description-not-explicitly-declared-by-source-record"
                ),
                "named_human_countersign_required": True,
            }
            counts["new_flags"] += 1
            flagged.append(
                {
                    "left_source_id": left_id,
                    "right_source_id": right_id,
                    "hamming_distance": distance,
                    "shared_metadata_fields": shared,
                }
            )
        pairs.append(
            {
                "left_source_id": left_id,
                "right_source_id": right_id,
                "hamming_distance": distance,
                "shared_metadata_fields": shared,
                "disposition": disposition,
                "evidence": evidence,
            }
        )
    expected_pairs = len(selected_ids) * (len(selected_ids) - 1) // 2
    if len(pairs) != expected_pairs or sum(counts.values()) != expected_pairs:
        raise AssertionError("pair accounting is incomplete")
    if role == "D1":
        expected_counts = {"prior_reviewed": 3, "machine_clear": 843, "new_flags": 15}
        if counts != expected_counts:
            raise ValueError(f"D1 similarity screen changed: {counts}")
        if len(_SIMILARITY_DESCRIPTIONS) != len(flagged):
            raise AssertionError("D1 prose verdict count changed")
        inferred = [
            {
                "screen_label": f"D1-selrow-{index:02d}",
                **item,
                "screen_description": _SIMILARITY_DESCRIPTIONS[index - 1],
                "screen_verdict": "distinct",
                "binding_state": "pending_named_human_countersign",
            }
            for index, item in enumerate(flagged, 1)
        ]
    else:
        expected_counts = {"prior_reviewed": 43, "machine_clear": 2807, "new_flags": 0}
        if counts != expected_counts:
            raise ValueError(f"D2 similarity screen changed: {counts}")
        inferred = []
    body = {
        "schema": _SCHEMA,
        "kind": _SIMILARITY_KIND,
        "experimental_role": role,
        "concept_id": _CONCEPTS[role],
        "split_file_sha256": split_file_sha256,
        "source_review_sha256": review["review_sha256"],
        "screen_markdown_file_sha256": screen_file_sha256,
        "method": {
            "pair_order": "lexicographically-sorted-source-ids-unordered-combinations",
            "prior_queue_precedence": True,
            "machine_clear_rule": (
                "hamming-distance-greater-than-8-and-no-equal-accession-family-"
                "or-burst-id"
            ),
            "perceptual_hash": (
                "rgb-luma-average-hash-8x8-bilinear-after-exif-transpose"
            ),
        },
        "selected_source_ids": sorted(selected_ids),
        "pair_counts": {**counts, "total": expected_pairs},
        "pairs": pairs,
        "inferred_screen_label_bindings": inferred,
        "screen_author": "Claude Fable 5 (agent; owner-authorized)",
        "review_state": "pending_named_human_countersign",
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": _CLAIM_LIMIT,
    }
    return {
        **body,
        "similarity_evidence_sha256": krea_provenance.canonical_sha256(body),
    }


def _stage_role(
    role: str,
    *,
    output_root: Path,
    review: dict[str, Any],
    review_file_sha256: str,
    split: dict[str, Any],
    split_file_sha256: str,
    materialization_root: Path,
    materialization: dict[str, Any],
    materialization_rows: dict[str, dict[str, Any]],
    materialization_file_sha256: str,
    tokenizer_evidence: dict[str, Any],
    tokenizer_file_sha256: str,
    screen_file_sha256: str,
    prepared_at_utc: str,
    d2_commitment_binding: dict[str, str] | None,
) -> dict[str, Any]:
    training_ids, evaluation_ids = _validate_split(split, role=role, review=review)
    review_rows = _row_by_id(review, role)
    selected_ids = training_ids + evaluation_ids
    if materialization.get("experimental_role") != role:
        raise ValueError(f"{role} materialization role mismatch")
    if (
        materialization.get("materialization_sha256")
        != review["source_evidence"][role]["materialization_sha256"]
    ):
        raise ValueError(f"{role} materialization is not bound by the review")
    role_root = output_root / role
    rights_rows: list[dict[str, Any]] = []
    caption_rows: list[dict[str, Any]] = []
    fixture_rows: list[dict[str, Any]] = []
    normalized_captions: set[str] = set()
    for split_name, source_ids in (
        ("training", training_ids),
        ("evaluation", evaluation_ids),
    ):
        for source_id in source_ids:
            reviewed = review_rows[source_id]
            materialized = materialization_rows.get(source_id)
            if materialized is None:
                raise ValueError(
                    f"selected source is absent from materialization: {source_id}"
                )
            if (
                materialized["sha256"] != reviewed["byte_sha256"]
                or materialized["bytes"] != reviewed["byte_count"]
            ):
                raise ValueError(f"review/materialization bytes disagree: {source_id}")
            source_image = materialization_root / materialized["relative_path"]
            suffix = Path(materialized["relative_path"]).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                raise ValueError(
                    f"selected source has unsupported extension: {source_id}"
                )
            image_name = f"{source_id}{suffix}"
            caption_name = f"{source_id}.txt"
            image_path = role_root / split_name / image_name
            caption_path = role_root / split_name / caption_name
            _copy_bound(source_image, image_path, reviewed["byte_sha256"])
            caption = _caption_bytes(role, split_name, reviewed)
            normalized = _normalize_spaces(caption.decode("utf-8").casefold())
            if normalized in normalized_captions:
                raise ValueError(
                    f"candidate caption duplicates another selected row: {source_id}"
                )
            normalized_captions.add(normalized)
            _write_bytes(caption_path, caption)
            caption_sha = hashlib.sha256(caption).hexdigest()
            rights = {
                "source_id": source_id,
                "split": split_name,
                "byte_sha256": reviewed["byte_sha256"],
                "bytes": reviewed["byte_count"],
                "source_page_url": reviewed["source_page_url"],
                "provider_title": reviewed["provider_title"],
                "creator_or_artist": reviewed["creator_or_artist"],
                "license_name": reviewed["license_name"],
                "license_url": reviewed["license_url"],
                "rights_decision": reviewed["rights_decision"],
                "attribution_or_pd_record": reviewed["attribution_or_pd_record"],
                "obligations": _rights_obligations(reviewed),
            }
            rights_rows.append(rights)
            caption_rows.append(
                {
                    "source_id": source_id,
                    "split": split_name,
                    "relative_caption_path": f"{split_name}/{caption_name}",
                    "caption_sha256": caption_sha,
                    "caption_bytes": len(caption),
                    "normalized_caption_sha256": hashlib.sha256(
                        normalized.encode("utf-8")
                    ).hexdigest(),
                    "trigger_occurrences": 0 if split_name == "training" else 1,
                    "caption": caption.decode("utf-8").rstrip("\n"),
                }
            )
            fixture_rows.append(
                {
                    "source_id": source_id,
                    "split": split_name,
                    "relative_image_path": f"{split_name}/{image_name}",
                    "relative_caption_path": f"{split_name}/{caption_name}",
                    "image_sha256": reviewed["byte_sha256"],
                    "image_bytes": reviewed["byte_count"],
                    "decoded_rgb_sha256": reviewed["decoded_rgb_sha256"],
                    "caption_sha256": caption_sha,
                    "normalized_caption_sha256": hashlib.sha256(
                        normalized.encode("utf-8")
                    ).hexdigest(),
                    "width": reviewed["width"],
                    "height": reviewed["height"],
                    "perceptual_hash64": reviewed["perceptual_hash64"],
                    "group_identity": _group_projection(role, reviewed),
                }
            )
    rights_rows.sort(key=lambda row: row["source_id"])
    caption_rows.sort(key=lambda row: row["source_id"])
    fixture_rows.sort(key=lambda row: row["source_id"])
    rights_body = {
        "schema": _SCHEMA,
        "kind": _RIGHTS_KIND,
        "experimental_role": role,
        "concept_id": _CONCEPTS[role],
        # The named operator owns this curation record, not the third-party
        # works.  Per-row creator/license fields carry the actual rights basis.
        "curation_owner_identity": materialization["retrieval_owner_identity"],
        "source_locator": (
            "https://commons.wikimedia.org/wiki/Category:Fontana_del_Moro_(Rome)"
            if role == "D1"
            else "https://www.artic.edu/open-access"
        ),
        "source_review_sha256": review["review_sha256"],
        "split_file_sha256": split_file_sha256,
        "retrieval_authorization_sha256": materialization[
            "retrieval_authorization_sha256"
        ],
        "rows": rights_rows,
        "counts": {
            "selected": len(rights_rows),
            "cc_by": sum(
                row["rights_decision"] == "approve_cc_by_obligations_recorded"
                for row in rights_rows
            ),
            "pd_or_cc0": sum(
                row["rights_decision"] == "approve_pd_or_cc0" for row in rights_rows
            ),
        },
        "review_state": "pending_named_human_countersign",
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": _CLAIM_LIMIT,
    }
    rights = {
        **rights_body,
        "rights_ledger_sha256": krea_provenance.canonical_sha256(rights_body),
    }
    rights_path = role_root / "rights-ledger.candidate.json"
    _write_canonical(rights_path, rights)
    caption_body = {
        "schema": _SCHEMA,
        "kind": _CAPTION_KIND,
        "experimental_role": role,
        "concept_id": _CONCEPTS[role],
        "trigger_token": _TRIGGERS[role],
        "trigger_evidence_sha256": tokenizer_evidence["evidence_sha256"],
        "caption_policy": {
            "training": (
                "observable-descriptor-only-trigger-injected-by-ai-toolkit-config"
            ),
            "evaluation": "same-observable-descriptor-prefixed-by-trigger-exactly-once",
            "proper_name_exclusions": list(
                _D1_FORBIDDEN_CAPTION_TERMS
                if role == "D1"
                else _D2_FORBIDDEN_CAPTION_TERMS
            ),
            "crop_policy": "no-crop",
            "normalization": "NFC-then-collapse-unicode-whitespace",
        },
        "rows": caption_rows,
        "review_state": "pending_named_human_countersign",
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": _CLAIM_LIMIT,
    }
    captions = {
        **caption_body,
        "caption_ledger_sha256": krea_provenance.canonical_sha256(caption_body),
    }
    caption_path = role_root / "caption-ledger.candidate.json"
    _write_canonical(caption_path, captions)
    similarity = _similarity_evidence(
        role,
        rows=review_rows,
        selected_ids=selected_ids,
        review=review,
        split_file_sha256=split_file_sha256,
        screen_file_sha256=screen_file_sha256,
    )
    similarity_path = role_root / "similarity-evidence.candidate.json"
    _write_canonical(similarity_path, similarity)
    review_surfaces = _review_surfaces(
        role_root,
        role=role,
        fixture_rows=fixture_rows,
        caption_rows=caption_rows,
        rights_rows=rights_rows,
        similarity=similarity,
    )
    archive_path = role_root / "training.zip"
    _deterministic_zip(role_root / "training", archive_path)
    archive_identity = _archive_identity(archive_path)
    bindings = {
        "source_review": {
            "relative_path": "evidence/D1D2.executable-review.json",
            "file_sha256": review_file_sha256,
            "semantic_sha256": review["review_sha256"],
        },
        "source_split": {
            "relative_path": f"evidence/{role}.source-split.json",
            "file_sha256": split_file_sha256,
            "semantic_sha256": split["split_sha256"],
        },
        "source_materialization": {
            "relative_path": f"evidence/{role}.materialization.json",
            "file_sha256": materialization_file_sha256,
            "semantic_sha256": materialization["materialization_sha256"],
        },
        "trigger_evidence": {
            "relative_path": "evidence/tokenizer-trigger-evidence.json",
            "file_sha256": tokenizer_file_sha256,
            "semantic_sha256": tokenizer_evidence["evidence_sha256"],
        },
        "similarity_screen_markdown": {
            "relative_path": "evidence/selected-row-similarity-screen.md",
            "file_sha256": screen_file_sha256,
        },
        "rights_ledger": {
            "relative_path": f"{role}/rights-ledger.candidate.json",
            "file_sha256": _file_sha256(rights_path),
            "semantic_sha256": rights["rights_ledger_sha256"],
        },
        "caption_ledger": {
            "relative_path": f"{role}/caption-ledger.candidate.json",
            "file_sha256": _file_sha256(caption_path),
            "semantic_sha256": captions["caption_ledger_sha256"],
        },
        "similarity_evidence": {
            "relative_path": f"{role}/similarity-evidence.candidate.json",
            "file_sha256": _file_sha256(similarity_path),
            "semantic_sha256": similarity["similarity_evidence_sha256"],
        },
    }
    if role == "D2":
        if d2_commitment_binding is None:
            raise ValueError("D2 commitment binding is required")
        bindings["d2_key_commitment"] = {
            "relative_path": "evidence/D2.key-commitment.json",
            **d2_commitment_binding,
        }
    elif d2_commitment_binding is not None:
        raise ValueError("D2 commitment binding cannot be attached to D1")
    body = {
        "schema": _SCHEMA,
        "kind": _KIND,
        "experimental_role": role,
        "concept_id": _CONCEPTS[role],
        "trigger_token": _TRIGGERS[role],
        "prepared_at_utc": prepared_at_utc,
        "preparer_actor": _PREPARER_ACTOR,
        "tool_identity": _tool_identity(),
        "bindings": bindings,
        "review_surfaces": review_surfaces,
        "training_archive": {
            "relative_path": "training.zip",
            **_file_identity(archive_path),
            "identity": archive_identity,
        },
        "training_source_ids": training_ids,
        "evaluation_source_ids": evaluation_ids,
        "rows": fixture_rows,
        "pending_gates": _PENDING_GATES,
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": _CLAIM_LIMIT,
    }
    candidate = {
        **body,
        "candidate_manifest_sha256": krea_provenance.canonical_sha256(body),
    }
    manifest_path = role_root / "fixture-manifest.candidate.json"
    _write_canonical(manifest_path, candidate)
    return candidate


def _deterministic_zip(directory: Path, output: Path) -> None:
    files = sorted(path for path in directory.iterdir() if path.is_file())
    if not files or any(path.is_symlink() for path in files):
        raise ValueError("training directory is empty or unsafe")
    expected = {
        path.stem
        for path in files
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    }
    captions = {path.stem for path in files if path.suffix.lower() == ".txt"}
    if expected != captions or len(files) != 2 * len(expected):
        raise ValueError(
            "training directory does not contain exact image/caption pairs"
        )
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            info.flag_bits = 0
            archive.writestr(info, path.read_bytes())


def _archive_identity(path: Path) -> dict[str, Any]:
    members = []
    seen: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if [info.filename for info in infos] != sorted(info.filename for info in infos):
            raise ValueError("training archive member order is not canonical")
        for info in infos:
            name = info.filename
            mode = (info.external_attr >> 16) & 0o177777
            if (
                info.is_dir()
                or not name
                or name in seen
                or Path(name).name != name
                or "\\" in name
                or any(ord(character) < 32 for character in name)
                or name in {".", ".."}
                or info.flag_bits & 0x1
                or info.date_time != (1980, 1, 1, 0, 0, 0)
                or info.compress_type != zipfile.ZIP_STORED
                or info.create_system != 3
                or mode != (stat.S_IFREG | 0o600)
            ):
                raise ValueError("training archive contains an unsafe member")
            seen.add(name)
            payload = archive.read(info)
            members.append(
                {
                    "path": name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    body = {"format": "zip-stored-fixed-1980-epoch", "members": members}
    return {**body, "identity_sha256": krea_provenance.canonical_sha256(body)}


def _review_surfaces(
    role_root: Path,
    *,
    role: str,
    fixture_rows: list[dict[str, Any]],
    caption_rows: list[dict[str, Any]],
    rights_rows: list[dict[str, Any]],
    similarity: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    captions = {row["source_id"]: row for row in caption_rows}
    rights = {row["source_id"]: row for row in rights_rows}
    fixtures = {row["source_id"]: row for row in fixture_rows}
    cards = []
    for source_id in sorted(fixtures):
        row = fixtures[source_id]
        right = rights[source_id]
        caption = captions[source_id]
        cards.append(
            "<article>"
            f"<h2>{html.escape(source_id)} · {html.escape(row['split'])}</h2>"
            f"<img loading='lazy' src='{html.escape(row['relative_image_path'])}' "
            f"alt='{html.escape(source_id)}'>"
            f"<p><strong>Caption:</strong> {html.escape(caption['caption'])}</p>"
            "<p><strong>Creator:</strong> "
            f"{html.escape(right['creator_or_artist'])}</p>"
            f"<p><strong>License:</strong> {html.escape(right['license_name'])}</p>"
            f"<p><a href='{html.escape(right['source_page_url'])}'>Source page</a></p>"
            "</article>"
        )
    style = (
        "body{font-family:system-ui;margin:1rem}main{display:grid;"
        "grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}"
        "article{border:1px solid #bbb;padding:.75rem}img{max-width:100%;"
        "max-height:420px;object-fit:contain;background:#eee}code{word-break:break-all}"
    )
    index_payload = (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{role} fixture review index</title><style>{style}</style>"
        f"<h1>{role} fixture candidate — visual/caption/rights review</h1>"
        "<p>This is a review surface, not an approval. Source IDs are provenance "
        "labels and do not appear in training captions.</p>"
        f"<main>{''.join(cards)}</main>"
    ).encode("utf-8")
    index_path = role_root / "review-index.html"
    _write_bytes(index_path, index_payload)
    pair_cards = []
    for binding in similarity["inferred_screen_label_bindings"]:
        left = fixtures[binding["left_source_id"]]
        right = fixtures[binding["right_source_id"]]
        pair_cards.append(
            "<article>"
            f"<h2>{html.escape(binding['screen_label'])} · binding pending</h2>"
            f"<p>{html.escape(binding['screen_description'])}</p>"
            "<div class='pair'>"
            f"<figure><img src='{html.escape(left['relative_image_path'])}' "
            f"alt='{html.escape(left['source_id'])}'><figcaption>"
            f"{html.escape(left['source_id'])}</figcaption></figure>"
            f"<figure><img src='{html.escape(right['relative_image_path'])}' "
            f"alt='{html.escape(right['source_id'])}'><figcaption>"
            f"{html.escape(right['source_id'])}</figcaption></figure>"
            "</div></article>"
        )
    pair_style = style + ".pair{display:grid;grid-template-columns:1fr 1fr;gap:1rem}"
    pair_payload = (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{role} targeted pair review</title><style>{pair_style}</style>"
        f"<h1>{role} targeted similarity pairs</h1>"
        + (
            "<p>No new targeted visual pairs; the exhaustive machine screen "
            "found zero unresolved risky pairs.</p>"
            if not pair_cards
            else "<p>Confirm each explicit source-ID pair as distinct or reject it.</p>"
        )
        + f"<main>{''.join(pair_cards)}</main>"
    ).encode("utf-8")
    pair_path = role_root / "targeted-pair-review.html"
    _write_bytes(pair_path, pair_payload)
    return {
        "review_index": {
            "relative_path": f"{role}/review-index.html",
            **_file_identity(index_path),
        },
        "targeted_pair_review": {
            "relative_path": f"{role}/targeted-pair-review.html",
            **_file_identity(pair_path),
        },
    }


def _package_files(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"package contains a symlink: {path}")
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"package contains a non-regular entry: {path}")
        if path != root / "package-manifest.json":
            rows.append(
                {"path": path.relative_to(root).as_posix(), **_file_identity(path)}
            )
    return sorted(rows, key=lambda row: row["path"])


def _validate_exact_topology(
    package_files: list[dict[str, Any]], expected_file_paths: set[str]
) -> None:
    observed_file_paths = {row["path"] for row in package_files}
    if observed_file_paths != expected_file_paths:
        raise ValueError(
            "package topology mismatch: "
            f"missing={sorted(expected_file_paths - observed_file_paths)}, "
            f"extra={sorted(observed_file_paths - expected_file_paths)}"
        )


def _tool_identity() -> dict[str, str]:
    return {
        "fixture_package_source_sha256": _file_sha256(
            Path(__file__).resolve(strict=True)
        ),
        "provenance_source_sha256": _file_sha256(
            Path(krea_provenance.__file__).resolve(strict=True)
        ),
        "review_split_source_sha256": _file_sha256(
            Path(krea_review_split.__file__).resolve(strict=True)
        ),
    }


def build_package(
    *,
    review_path: Path,
    d1_split_path: Path,
    d2_split_path: Path,
    d2_commitment_path: Path,
    d2_secret_path: Path,
    d1_materialization_root: Path,
    d2_materialization_root: Path,
    tokenizer_evidence_path: Path,
    similarity_screen_path: Path,
    prepared_at_utc: str,
    output_root: Path,
) -> dict[str, Any]:
    review, review_file_sha = _canonical_json(review_path, "executable review")
    _reject_governance_escalation(review, "executable review")
    krea_review_split.validate_review(review)
    d1_split, d1_split_file_sha = _canonical_json(d1_split_path, "D1 split")
    d2_split, d2_split_file_sha = _canonical_json(d2_split_path, "D2 split")
    _reject_governance_escalation(d1_split, "D1 split")
    _reject_governance_escalation(d2_split, "D2 split")
    _validate_split(d1_split, role="D1", review=review)
    _validate_split(d2_split, role="D2", review=review)
    krea_review_split.validate_d1_split(d1_split, review)
    d2_commitment, d2_commitment_file_sha = _canonical_json(
        d2_commitment_path, "D2 key commitment"
    )
    _reject_governance_escalation(d2_commitment, "D2 key commitment")
    d2_secret = krea_review_split._secret_file(d2_secret_path)
    krea_review_split.validate_d2_split(
        d2_split,
        review,
        d2_commitment,
        secret=d2_secret,
    )
    d2_commitment_binding = {
        "file_sha256": d2_commitment_file_sha,
        "semantic_sha256": d2_commitment["commitment_sha256"],
    }
    tokenizer, tokenizer_file_sha = _canonical_json(
        tokenizer_evidence_path, "trigger evidence"
    )
    _reject_governance_escalation(tokenizer, "trigger evidence")
    if (
        tokenizer.get("kind") != "forge-krea-tokenizer-trigger-evidence"
        or tokenizer.get("schema") != 1
        or tokenizer.get("evidence_sha256")
        != krea_provenance.canonical_sha256(
            {key: value for key, value in tokenizer.items() if key != "evidence_sha256"}
        )
        or tokenizer.get("evidence_sha256") != _TOKENIZER_EVIDENCE_SHA256
        or tokenizer.get("model_id") != _TOKENIZER_MODEL
        or tokenizer.get("revision") != _TOKENIZER_REVISION
        or tokenizer.get("admission_authorized") is not False
        or tokenizer.get("gpu_execution_authorized") is not False
    ):
        raise ValueError("trigger evidence boundary is invalid")
    for role, trigger in _TRIGGERS.items():
        entry = tokenizer.get("roles", {}).get(role)
        if (
            not isinstance(entry, dict)
            or entry.get("trigger") != trigger
            or entry.get("unknown_token_present") is not False
            or entry.get("absent_from_full_curation_tree") is not True
            or entry.get("human_approved") is not False
            or isinstance(entry.get("token_count"), bool)
            or not isinstance(entry.get("token_count"), int)
            or entry["token_count"] <= 0
            or not isinstance(entry.get("token_ids"), list)
            or len(entry["token_ids"]) != entry["token_count"]
        ):
            raise ValueError(f"{role} trigger evidence differs from the frozen token")
    similarity_screen_path = _safe_file(similarity_screen_path, "similarity screen")
    screen_file_sha = _file_sha256(similarity_screen_path)
    screen_text = similarity_screen_path.read_text(encoding="utf-8")
    for required in (
        "D1-selrow-15",
        "15/15 distinct",
        _TRIGGERS["D1"],
        _TRIGGERS["D2"],
    ):
        if required not in screen_text:
            raise ValueError(
                f"similarity screen is missing required evidence: {required}"
            )
    d1_root = _safe_directory(d1_materialization_root, "D1 materialization")
    d2_root = _safe_directory(d2_materialization_root, "D2 materialization")
    d1_materialization, d1_rows, d1_materialization_file_sha = _materialized_by_id(
        d1_root
    )
    d2_materialization, d2_rows, d2_materialization_file_sha = _materialized_by_id(
        d2_root
    )
    _reject_governance_escalation(d1_materialization, "D1 materialization")
    _reject_governance_escalation(d2_materialization, "D2 materialization")
    # This is an evidence timestamp, not permission.  Reuse the strict parser
    # already governing the source split records.
    krea_review_split._canonical_utc(prepared_at_utc)
    output_root = Path(os.path.abspath(os.path.expanduser(output_root)))
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite package: {output_root}")
    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{output_root.name}.partial-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"stale package temporary exists: {temporary}")
    temporary.mkdir(mode=0o700)
    try:
        evidence_sources = {
            "D1D2.executable-review.json": (review_path, review_file_sha),
            "D1.source-split.json": (d1_split_path, d1_split_file_sha),
            "D2.source-split.json": (d2_split_path, d2_split_file_sha),
            "D2.key-commitment.json": (
                d2_commitment_path,
                d2_commitment_file_sha,
            ),
            "D1.materialization.json": (
                d1_root / "materialization.json",
                d1_materialization_file_sha,
            ),
            "D2.materialization.json": (
                d2_root / "materialization.json",
                d2_materialization_file_sha,
            ),
            "tokenizer-trigger-evidence.json": (
                tokenizer_evidence_path,
                tokenizer_file_sha,
            ),
            "selected-row-similarity-screen.md": (
                similarity_screen_path,
                screen_file_sha,
            ),
        }
        for name, (source, digest) in sorted(evidence_sources.items()):
            _copy_bound(source, temporary / "evidence" / name, digest)
        candidates = {}
        role_inputs = (
            (
                "D1",
                d1_split,
                d1_split_file_sha,
                d1_root,
                d1_materialization,
                d1_rows,
                d1_materialization_file_sha,
            ),
            (
                "D2",
                d2_split,
                d2_split_file_sha,
                d2_root,
                d2_materialization,
                d2_rows,
                d2_materialization_file_sha,
            ),
        )
        for (
            role,
            split,
            split_sha,
            materialization_root,
            materialization,
            rows,
            materialization_sha,
        ) in role_inputs:
            candidates[role] = _stage_role(
                role,
                output_root=temporary,
                review=review,
                review_file_sha256=review_file_sha,
                split=split,
                split_file_sha256=split_sha,
                materialization_root=materialization_root,
                materialization=materialization,
                materialization_rows=rows,
                materialization_file_sha256=materialization_sha,
                tokenizer_evidence=tokenizer,
                tokenizer_file_sha256=tokenizer_file_sha,
                screen_file_sha256=screen_file_sha,
                prepared_at_utc=prepared_at_utc,
                d2_commitment_binding=(d2_commitment_binding if role == "D2" else None),
            )
        review_request_body = {
            "schema": _SCHEMA,
            "kind": _REVIEW_REQUEST_KIND,
            "prepared_at_utc": prepared_at_utc,
            "tool_identity": _tool_identity(),
            "candidate_manifest_sha256s": {
                role: candidate["candidate_manifest_sha256"]
                for role, candidate in sorted(candidates.items())
            },
            "requested_named_human_countersigns": _REQUESTED_COUNTERSIGNS,
            "requested_independent_review": _REQUESTED_INDEPENDENT_REVIEW,
            "reviewer_must_write_separate_hash_binding_records": True,
            "candidate_files_must_not_be_edited": True,
            "admission_authorized": False,
            "gpu_execution_authorized": False,
            "claim_limit": _CLAIM_LIMIT,
        }
        review_request = {
            **review_request_body,
            "review_request_sha256": krea_provenance.canonical_sha256(
                review_request_body
            ),
        }
        _write_canonical(temporary / "bundled-review.request.json", review_request)
        files = _package_files(temporary)
        body = {
            "schema": _SCHEMA,
            "kind": _PACKAGE_KIND,
            "prepared_at_utc": prepared_at_utc,
            "tool_identity": _tool_identity(),
            "candidate_manifest_sha256s": {
                role: candidate["candidate_manifest_sha256"]
                for role, candidate in sorted(candidates.items())
            },
            "review_request_sha256": review_request["review_request_sha256"],
            "files": files,
            "file_set_sha256": krea_provenance.canonical_sha256(files),
            "admission_authorized": False,
            "gpu_execution_authorized": False,
            "claim_limit": _CLAIM_LIMIT,
        }
        package = {**body, "package_sha256": krea_provenance.canonical_sha256(body)}
        _write_canonical(temporary / "package-manifest.json", package)
        validate_package(temporary)
        os.rename(temporary, output_root)
        return package
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_package(root: Path) -> dict[str, Any]:
    root = _safe_directory(root, "fixture candidate package")
    package, _ = _canonical_json(root / "package-manifest.json", "package manifest")
    package_keys = {
        "schema",
        "kind",
        "prepared_at_utc",
        "tool_identity",
        "candidate_manifest_sha256s",
        "review_request_sha256",
        "files",
        "file_set_sha256",
        "admission_authorized",
        "gpu_execution_authorized",
        "claim_limit",
        "package_sha256",
    }
    _exact(package, package_keys, "package manifest")
    _reject_governance_escalation(package, "package manifest")
    body = {key: value for key, value in package.items() if key != "package_sha256"}
    live_files = _package_files(root)
    files = package["files"]
    if not isinstance(files, list):
        raise ValueError("package files must be an array")
    for row in files:
        _exact(
            _object(row, "package file"),
            {"path", "bytes", "sha256"},
            "package file",
        )
    candidate_sha_map = _object(
        package["candidate_manifest_sha256s"], "candidate manifest digest map"
    )
    _exact(candidate_sha_map, set(_EXPECTED_COUNTS), "candidate manifest digest map")
    top_level_checks = {
        "schema": package["schema"] == _SCHEMA,
        "kind": package["kind"] == _PACKAGE_KIND,
        "package digest": package["package_sha256"]
        == krea_provenance.canonical_sha256(body),
        "tool identity": package["tool_identity"] == _tool_identity(),
        "file inventory": package["files"] == live_files,
        "file order/uniqueness": [row["path"] for row in files]
        == sorted({row["path"] for row in files}),
        "file-set digest": package["file_set_sha256"]
        == krea_provenance.canonical_sha256(files),
        "claim limit": package["claim_limit"] == _CLAIM_LIMIT,
    }
    failed_top_level = [name for name, passed in top_level_checks.items() if not passed]
    if failed_top_level:
        raise ValueError(
            "fixture candidate package is invalid or changed: "
            + ", ".join(failed_top_level)
        )
    prepared_at = krea_review_split._canonical_utc(package["prepared_at_utc"])

    evidence_paths = {
        "review": "evidence/D1D2.executable-review.json",
        "D1_split": "evidence/D1.source-split.json",
        "D2_split": "evidence/D2.source-split.json",
        "D2_commitment": "evidence/D2.key-commitment.json",
        "D1_materialization": "evidence/D1.materialization.json",
        "D2_materialization": "evidence/D2.materialization.json",
        "tokenizer": "evidence/tokenizer-trigger-evidence.json",
        "screen": "evidence/selected-row-similarity-screen.md",
    }
    review, review_file_sha = _canonical_json(root / evidence_paths["review"], "review")
    _reject_governance_escalation(review, "review")
    krea_review_split.validate_review(review)
    splits: dict[str, dict[str, Any]] = {}
    split_file_shas: dict[str, str] = {}
    for role in _EXPECTED_COUNTS:
        split, file_sha = _canonical_json(
            root / evidence_paths[f"{role}_split"], f"{role} split"
        )
        _reject_governance_escalation(split, f"{role} split")
        _validate_split(split, role=role, review=review)
        splits[role] = split
        split_file_shas[role] = file_sha
    krea_review_split.validate_d1_split(splits["D1"], review)
    commitment, commitment_file_sha = _canonical_json(
        root / evidence_paths["D2_commitment"], "D2 commitment"
    )
    _reject_governance_escalation(commitment, "D2 commitment")
    krea_review_split.validate_d2_commitment(commitment, review)
    if splits["D2"].get("d2_key_commitment_sha256") != commitment["commitment_sha256"]:
        raise ValueError("D2 split is not bound to the copied commitment")

    materializations: dict[str, dict[str, Any]] = {}
    materialization_rows: dict[str, dict[str, dict[str, Any]]] = {}
    materialization_file_shas: dict[str, str] = {}
    for role in _EXPECTED_COUNTS:
        materialization, file_sha = _canonical_json(
            root / evidence_paths[f"{role}_materialization"],
            f"{role} materialization",
        )
        _reject_governance_escalation(materialization, f"{role} materialization")
        _exact(materialization, _MATERIALIZATION_KEYS, f"{role} materialization")
        materialization_body = {
            key: value
            for key, value in materialization.items()
            if key != "materialization_sha256"
        }
        raw_rows = materialization["rows"]
        if (
            materialization["schema"] != 2
            or materialization["kind"] != "forge-krea-source-candidate-materialization"
            or materialization["experimental_role"] != role
            or materialization["concept_id"] != _CONCEPTS[role]
            or materialization["materialization_sha256"]
            != krea_provenance.canonical_sha256(materialization_body)
            or materialization["materialization_sha256"]
            != review["source_evidence"][role]["materialization_sha256"]
            or materialization["admission_state"] != "candidate_unreviewed"
            or materialization["human_approvals"] != []
            or materialization["fixture_manifest_created"] is not False
            or not isinstance(raw_rows, list)
            or raw_rows != sorted(raw_rows, key=lambda row: row.get("source_id", ""))
        ):
            raise ValueError(f"{role} copied materialization is invalid")
        seen_materialized: set[str] = set()
        for raw_row in raw_rows:
            row = _object(raw_row, f"{role} materialization row")
            _exact(row, _MATERIALIZATION_ROW_KEYS, f"{role} materialization row")
            source_id = row["source_id"]
            relative = PurePosixPath(row["relative_path"])
            if (
                source_id in seen_materialized
                or relative.is_absolute()
                or len(relative.parts) != 2
                or relative.parts[0] != "images"
                or Path(relative.name).stem != source_id
                or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                or isinstance(row["bytes"], bool)
                or not isinstance(row["bytes"], int)
                or row["bytes"] <= 0
            ):
                raise ValueError(f"{role} materialization row is invalid")
            seen_materialized.add(source_id)
        materializations[role] = materialization
        materialization_rows[role] = {row["source_id"]: row for row in raw_rows}
        materialization_file_shas[role] = file_sha

    tokenizer, tokenizer_file_sha = _canonical_json(
        root / evidence_paths["tokenizer"], "tokenizer trigger evidence"
    )
    _reject_governance_escalation(tokenizer, "tokenizer trigger evidence")
    if (
        tokenizer.get("evidence_sha256") != _TOKENIZER_EVIDENCE_SHA256
        or tokenizer.get("model_id") != _TOKENIZER_MODEL
        or tokenizer.get("revision") != _TOKENIZER_REVISION
        or tokenizer.get("evidence_sha256")
        != krea_provenance.canonical_sha256(
            {key: value for key, value in tokenizer.items() if key != "evidence_sha256"}
        )
        or any(
            tokenizer.get("roles", {}).get(role, {}).get("trigger") != trigger
            or tokenizer["roles"][role].get("human_approved") is not False
            for role, trigger in _TRIGGERS.items()
        )
    ):
        raise ValueError("copied tokenizer evidence is invalid")
    screen_path = _safe_file(root / evidence_paths["screen"], "similarity screen")
    screen_file_sha = _file_sha256(screen_path)
    screen = screen_path.read_text(encoding="utf-8")
    if any(
        required not in screen
        for required in (
            "D1-selrow-15",
            "15/15 distinct",
            _TRIGGERS["D1"],
            _TRIGGERS["D2"],
        )
    ):
        raise ValueError("copied similarity screen is incomplete")

    review_request, _ = _canonical_json(
        root / "bundled-review.request.json", "bundled review request"
    )
    review_request_keys = {
        "schema",
        "kind",
        "prepared_at_utc",
        "tool_identity",
        "candidate_manifest_sha256s",
        "requested_named_human_countersigns",
        "requested_independent_review",
        "reviewer_must_write_separate_hash_binding_records",
        "candidate_files_must_not_be_edited",
        "admission_authorized",
        "gpu_execution_authorized",
        "claim_limit",
        "review_request_sha256",
    }
    _exact(review_request, review_request_keys, "bundled review request")
    _reject_governance_escalation(review_request, "bundled review request")
    review_request_body = {
        key: value
        for key, value in review_request.items()
        if key != "review_request_sha256"
    }
    if (
        review_request["schema"] != _SCHEMA
        or review_request["kind"] != _REVIEW_REQUEST_KIND
        or review_request["review_request_sha256"]
        != krea_provenance.canonical_sha256(review_request_body)
        or review_request["tool_identity"] != _tool_identity()
        or review_request["prepared_at_utc"] != prepared_at
        or review_request["candidate_manifest_sha256s"]
        != package["candidate_manifest_sha256s"]
        or review_request["requested_named_human_countersigns"]
        != _REQUESTED_COUNTERSIGNS
        or review_request["requested_independent_review"]
        != _REQUESTED_INDEPENDENT_REVIEW
        or review_request["candidate_files_must_not_be_edited"] is not True
        or review_request["reviewer_must_write_separate_hash_binding_records"]
        is not True
        or review_request["claim_limit"] != _CLAIM_LIMIT
        or package["review_request_sha256"] != review_request["review_request_sha256"]
    ):
        raise ValueError("bundled review request is invalid")

    expected_file_paths = {"bundled-review.request.json", *evidence_paths.values()}
    candidate_digests: dict[str, str] = {}
    for role in sorted(_EXPECTED_COUNTS):
        manifest_relative = f"{role}/fixture-manifest.candidate.json"
        candidate, _ = _canonical_json(
            root / manifest_relative, f"{role} candidate manifest"
        )
        candidate_keys = {
            "schema",
            "kind",
            "experimental_role",
            "concept_id",
            "trigger_token",
            "prepared_at_utc",
            "preparer_actor",
            "tool_identity",
            "bindings",
            "review_surfaces",
            "training_archive",
            "training_source_ids",
            "evaluation_source_ids",
            "rows",
            "pending_gates",
            "admission_authorized",
            "gpu_execution_authorized",
            "claim_limit",
            "candidate_manifest_sha256",
        }
        _exact(candidate, candidate_keys, f"{role} candidate manifest")
        _reject_governance_escalation(candidate, f"{role} candidate manifest")
        candidate_body = {
            key: value
            for key, value in candidate.items()
            if key != "candidate_manifest_sha256"
        }
        expected_train, expected_eval = _EXPECTED_COUNTS[role]
        training_ids = candidate["training_source_ids"]
        evaluation_ids = candidate["evaluation_source_ids"]
        if (
            candidate["schema"] != _SCHEMA
            or candidate["kind"] != _KIND
            or candidate["experimental_role"] != role
            or candidate["concept_id"] != _CONCEPTS[role]
            or candidate["trigger_token"] != _TRIGGERS[role]
            or candidate["prepared_at_utc"] != prepared_at
            or candidate["preparer_actor"] != _PREPARER_ACTOR
            or candidate["tool_identity"] != _tool_identity()
            or candidate["pending_gates"] != _PENDING_GATES
            or candidate["claim_limit"] != _CLAIM_LIMIT
            or candidate["candidate_manifest_sha256"]
            != krea_provenance.canonical_sha256(candidate_body)
            or len(training_ids) != expected_train
            or len(evaluation_ids) != expected_eval
            or training_ids != splits[role]["training_source_ids"]
            or evaluation_ids != splits[role]["evaluation_source_ids"]
            or training_ids != sorted(set(training_ids))
            or evaluation_ids != sorted(set(evaluation_ids))
            or set(training_ids) & set(evaluation_ids)
        ):
            raise ValueError(f"{role} candidate manifest is invalid")
        candidate_digests[role] = candidate["candidate_manifest_sha256"]
        expected_by_id = {
            source_id: split_name
            for split_name, source_ids in (
                ("training", training_ids),
                ("evaluation", evaluation_ids),
            )
            for source_id in source_ids
        }
        row_keys = {
            "source_id",
            "split",
            "relative_image_path",
            "relative_caption_path",
            "image_sha256",
            "image_bytes",
            "decoded_rgb_sha256",
            "caption_sha256",
            "normalized_caption_sha256",
            "width",
            "height",
            "perceptual_hash64",
            "group_identity",
        }
        rows = candidate["rows"]
        if (
            not isinstance(rows, list)
            or rows != sorted(rows, key=lambda row: row.get("source_id", ""))
            or {row.get("source_id") for row in rows} != set(expected_by_id)
            or len(rows) != len(expected_by_id)
        ):
            raise ValueError(f"{role} candidate rows do not exactly cover the split")
        by_id: dict[str, dict[str, Any]] = {}
        review_by_id = _row_by_id(review, role)
        normalized_captions: set[str] = set()
        for raw_row in rows:
            row = _object(raw_row, f"{role} candidate row")
            _exact(row, row_keys, f"{role} candidate row")
            source_id = row["source_id"]
            split_name = expected_by_id[source_id]
            expected_group_fields = set(_BASE_GROUP_FIELDS) | (
                set(_D2_GROUP_FIELDS) if role == "D2" else set()
            )
            group = _object(row["group_identity"], f"{role} group identity")
            image_relative = row["relative_image_path"]
            caption_relative = row["relative_caption_path"]
            reviewed = review_by_id[source_id]
            materialized = materialization_rows[role].get(source_id)
            expected_group = _group_projection(role, reviewed)
            if (
                row["split"] != split_name
                or set(group) != expected_group_fields
                or any(
                    not isinstance(value, str) or not value for value in group.values()
                )
                or Path(image_relative).parent.as_posix() != split_name
                or Path(image_relative).stem != source_id
                or Path(image_relative).suffix.lower()
                not in {".jpg", ".jpeg", ".png", ".webp"}
                or Path(caption_relative).parent.as_posix() != split_name
                or Path(caption_relative).name != f"{source_id}.txt"
                or not re.fullmatch(r"[0-9a-f]{64}", row["image_sha256"])
                or not re.fullmatch(r"[0-9a-f]{64}", row["decoded_rgb_sha256"])
                or not re.fullmatch(r"[0-9a-f]{64}", row["caption_sha256"])
                or not re.fullmatch(r"[0-9a-f]{64}", row["normalized_caption_sha256"])
                or not re.fullmatch(r"[0-9a-f]{16}", row["perceptual_hash64"])
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in (row["image_bytes"], row["width"], row["height"])
                )
                or materialized is None
                or Path(image_relative).suffix.lower()
                != Path(materialized["relative_path"]).suffix.lower()
                or row["image_sha256"] != reviewed["byte_sha256"]
                or row["image_sha256"] != materialized["sha256"]
                or row["image_bytes"] != reviewed["byte_count"]
                or row["image_bytes"] != materialized["bytes"]
                or row["decoded_rgb_sha256"] != reviewed["decoded_rgb_sha256"]
                or row["width"] != reviewed["width"]
                or row["height"] != reviewed["height"]
                or row["perceptual_hash64"] != reviewed["perceptual_hash64"]
                or group != expected_group
            ):
                raise ValueError(f"{role} candidate row schema/value is invalid")
            image_path = _safe_file(root / role / image_relative, "staged image")
            caption_path = _safe_file(root / role / caption_relative, "staged caption")
            caption_payload = caption_path.read_bytes()
            try:
                caption_text = caption_payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{role} caption is not UTF-8") from exc
            normalized = _normalize_spaces(caption_text.casefold())
            trigger_occurrences = len(
                re.findall(
                    rf"(?<![a-z0-9]){_TRIGGERS[role]}(?![a-z0-9])",
                    caption_text.casefold(),
                )
            )
            if (
                _file_identity(image_path)
                != {"bytes": row["image_bytes"], "sha256": row["image_sha256"]}
                or hashlib.sha256(caption_payload).hexdigest() != row["caption_sha256"]
                or hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                != row["normalized_caption_sha256"]
                or not normalized
                or normalized in normalized_captions
                or (split_name == "training" and trigger_occurrences != 0)
                or (split_name == "evaluation" and trigger_occurrences != 1)
            ):
                raise ValueError(
                    f"{role} staged row identity or trigger contract failed"
                )
            normalized_captions.add(normalized)
            by_id[source_id] = row
            expected_file_paths.update(
                {f"{role}/{image_relative}", f"{role}/{caption_relative}"}
            )

        bindings = _object(candidate["bindings"], f"{role} candidate bindings")
        expected_binding_keys = {
            "source_review",
            "source_split",
            "source_materialization",
            "trigger_evidence",
            "similarity_screen_markdown",
            "rights_ledger",
            "caption_ledger",
            "similarity_evidence",
        } | ({"d2_key_commitment"} if role == "D2" else set())
        _exact(bindings, expected_binding_keys, f"{role} candidate bindings")
        expected_bindings = {
            "source_review": {
                "relative_path": evidence_paths["review"],
                "file_sha256": review_file_sha,
                "semantic_sha256": review["review_sha256"],
            },
            "source_split": {
                "relative_path": evidence_paths[f"{role}_split"],
                "file_sha256": split_file_shas[role],
                "semantic_sha256": splits[role]["split_sha256"],
            },
            "source_materialization": {
                "relative_path": evidence_paths[f"{role}_materialization"],
                "file_sha256": materialization_file_shas[role],
                "semantic_sha256": materializations[role]["materialization_sha256"],
            },
            "trigger_evidence": {
                "relative_path": evidence_paths["tokenizer"],
                "file_sha256": tokenizer_file_sha,
                "semantic_sha256": tokenizer["evidence_sha256"],
            },
            "similarity_screen_markdown": {
                "relative_path": evidence_paths["screen"],
                "file_sha256": screen_file_sha,
            },
        }
        if role == "D2":
            expected_bindings["d2_key_commitment"] = {
                "relative_path": evidence_paths["D2_commitment"],
                "file_sha256": commitment_file_sha,
                "semantic_sha256": commitment["commitment_sha256"],
            }
        for name, expected in expected_bindings.items():
            if bindings[name] != expected:
                raise ValueError(f"{role} {name} evidence binding is invalid")

        bound_records = {
            "rights_ledger": (
                f"{role}/rights-ledger.candidate.json",
                "rights_ledger_sha256",
                _RIGHTS_KIND,
            ),
            "caption_ledger": (
                f"{role}/caption-ledger.candidate.json",
                "caption_ledger_sha256",
                _CAPTION_KIND,
            ),
            "similarity_evidence": (
                f"{role}/similarity-evidence.candidate.json",
                "similarity_evidence_sha256",
                _SIMILARITY_KIND,
            ),
        }
        records: dict[str, dict[str, Any]] = {}
        for binding_name, (relative, digest_key, kind) in bound_records.items():
            record, file_sha = _canonical_json(root / relative, binding_name)
            _reject_governance_escalation(record, f"{role} {binding_name}")
            record_body = {
                key: value for key, value in record.items() if key != digest_key
            }
            expected_binding = {
                "relative_path": relative,
                "file_sha256": file_sha,
                "semantic_sha256": record[digest_key],
            }
            if (
                record.get("schema") != _SCHEMA
                or record.get("kind") != kind
                or record.get("experimental_role") != role
                or record.get("concept_id") != _CONCEPTS[role]
                or record.get(digest_key)
                != krea_provenance.canonical_sha256(record_body)
                or bindings[binding_name] != expected_binding
                or record.get("review_state") != "pending_named_human_countersign"
                or record.get("claim_limit") != _CLAIM_LIMIT
            ):
                raise ValueError(f"{role} {binding_name} is invalid or unbound")
            records[binding_name] = record
            expected_file_paths.add(relative)

        rights = records["rights_ledger"]
        rights_keys = {
            "schema",
            "kind",
            "experimental_role",
            "concept_id",
            "curation_owner_identity",
            "source_locator",
            "source_review_sha256",
            "split_file_sha256",
            "retrieval_authorization_sha256",
            "rows",
            "counts",
            "review_state",
            "admission_authorized",
            "gpu_execution_authorized",
            "claim_limit",
            "rights_ledger_sha256",
        }
        _exact(rights, rights_keys, f"{role} rights ledger")
        rights_row_keys = {
            "source_id",
            "split",
            "byte_sha256",
            "bytes",
            "source_page_url",
            "provider_title",
            "creator_or_artist",
            "license_name",
            "license_url",
            "rights_decision",
            "attribution_or_pd_record",
            "obligations",
        }
        rights_rows = rights["rows"]
        expected_source_locator = (
            "https://commons.wikimedia.org/wiki/Category:Fontana_del_Moro_(Rome)"
            if role == "D1"
            else "https://www.artic.edu/open-access"
        )
        if (
            rights["source_review_sha256"] != review["review_sha256"]
            or rights["split_file_sha256"] != split_file_shas[role]
            or rights["retrieval_authorization_sha256"]
            != materializations[role]["retrieval_authorization_sha256"]
            or rights["curation_owner_identity"]
            != materializations[role]["retrieval_owner_identity"]
            or rights["source_locator"] != expected_source_locator
            or not isinstance(rights_rows, list)
            or rights_rows
            != sorted(rights_rows, key=lambda row: row.get("source_id", ""))
            or len(rights_rows) != len(by_id)
        ):
            raise ValueError(f"{role} rights ledger coverage/binding is invalid")
        right_counts = {"selected": 0, "cc_by": 0, "pd_or_cc0": 0}
        for raw_right in rights_rows:
            right = _object(raw_right, f"{role} rights row")
            _exact(right, rights_row_keys, f"{role} rights row")
            source_id = right["source_id"]
            candidate_row = by_id.get(source_id)
            reviewed = review_by_id.get(source_id)
            expected_right = (
                {
                    "source_id": source_id,
                    "split": candidate_row["split"],
                    "byte_sha256": reviewed["byte_sha256"],
                    "bytes": reviewed["byte_count"],
                    "source_page_url": reviewed["source_page_url"],
                    "provider_title": reviewed["provider_title"],
                    "creator_or_artist": reviewed["creator_or_artist"],
                    "license_name": reviewed["license_name"],
                    "license_url": reviewed["license_url"],
                    "rights_decision": reviewed["rights_decision"],
                    "attribution_or_pd_record": reviewed["attribution_or_pd_record"],
                    "obligations": _rights_obligations(reviewed),
                }
                if candidate_row is not None and reviewed is not None
                else None
            )
            if (
                candidate_row is None
                or reviewed is None
                or right != expected_right
                or not right["source_page_url"].startswith("https://")
                or not right["creator_or_artist"]
                or not right["license_name"]
                or not right["attribution_or_pd_record"]
            ):
                raise ValueError(f"{role} rights row is false or incomplete")
            right_counts["selected"] += 1
            right_counts[
                (
                    "cc_by"
                    if right["rights_decision"] == "approve_cc_by_obligations_recorded"
                    else "pd_or_cc0"
                )
            ] += 1
        if rights["counts"] != right_counts:
            raise ValueError(f"{role} rights counts are not recomputed counts")

        captions = records["caption_ledger"]
        caption_keys = {
            "schema",
            "kind",
            "experimental_role",
            "concept_id",
            "trigger_token",
            "trigger_evidence_sha256",
            "caption_policy",
            "rows",
            "review_state",
            "admission_authorized",
            "gpu_execution_authorized",
            "claim_limit",
            "caption_ledger_sha256",
        }
        _exact(captions, caption_keys, f"{role} caption ledger")
        expected_caption_policy = {
            "training": (
                "observable-descriptor-only-trigger-injected-by-ai-toolkit-config"
            ),
            "evaluation": (
                "same-observable-descriptor-prefixed-by-trigger-exactly-once"
            ),
            "proper_name_exclusions": list(
                _D1_FORBIDDEN_CAPTION_TERMS
                if role == "D1"
                else _D2_FORBIDDEN_CAPTION_TERMS
            ),
            "crop_policy": "no-crop",
            "normalization": "NFC-then-collapse-unicode-whitespace",
        }
        caption_row_keys = {
            "source_id",
            "split",
            "relative_caption_path",
            "caption_sha256",
            "caption_bytes",
            "normalized_caption_sha256",
            "trigger_occurrences",
            "caption",
        }
        caption_rows = captions["rows"]
        if (
            captions["trigger_token"] != _TRIGGERS[role]
            or captions["trigger_evidence_sha256"] != tokenizer["evidence_sha256"]
            or captions["caption_policy"] != expected_caption_policy
            or not isinstance(caption_rows, list)
            or caption_rows
            != sorted(caption_rows, key=lambda row: row.get("source_id", ""))
            or len(caption_rows) != len(by_id)
        ):
            raise ValueError(f"{role} caption ledger binding/policy is invalid")
        for raw_caption in caption_rows:
            caption = _object(raw_caption, f"{role} caption row")
            _exact(caption, caption_row_keys, f"{role} caption row")
            source_id = caption["source_id"]
            candidate_row = by_id.get(source_id)
            if candidate_row is None:
                raise ValueError(f"{role} caption row selects an unknown source")
            reviewed = review_by_id[source_id]
            expected_relative = candidate_row["relative_caption_path"]
            payload = _safe_file(
                root / role / expected_relative, "staged caption"
            ).read_bytes()
            expected_payload = _caption_bytes(role, candidate_row["split"], reviewed)
            expected_text = payload.decode("utf-8")
            normalized = _normalize_spaces(expected_text.casefold())
            trigger_occurrences = len(
                re.findall(
                    rf"(?<![a-z0-9]){_TRIGGERS[role]}(?![a-z0-9])",
                    expected_text.casefold(),
                )
            )
            if (
                caption["split"] != candidate_row["split"]
                or caption["relative_caption_path"]
                != f"{candidate_row['split']}/{source_id}.txt"
                or caption["relative_caption_path"] != expected_relative
                or caption["caption_sha256"] != candidate_row["caption_sha256"]
                or caption["caption_sha256"] != hashlib.sha256(payload).hexdigest()
                or caption["caption_bytes"] != len(payload)
                or caption["normalized_caption_sha256"]
                != candidate_row["normalized_caption_sha256"]
                or caption["normalized_caption_sha256"]
                != hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                or caption["trigger_occurrences"] != trigger_occurrences
                or caption["caption"] != expected_text.rstrip("\n")
                or payload != caption["caption"].encode("utf-8") + b"\n"
                or payload != expected_payload
            ):
                raise ValueError(f"{role} caption row differs from staged bytes")

        similarity = records["similarity_evidence"]
        similarity_keys = {
            "schema",
            "kind",
            "experimental_role",
            "concept_id",
            "split_file_sha256",
            "source_review_sha256",
            "screen_markdown_file_sha256",
            "method",
            "selected_source_ids",
            "pair_counts",
            "pairs",
            "inferred_screen_label_bindings",
            "screen_author",
            "review_state",
            "admission_authorized",
            "gpu_execution_authorized",
            "claim_limit",
            "similarity_evidence_sha256",
        }
        _exact(similarity, similarity_keys, f"{role} similarity evidence")
        expected_method = {
            "pair_order": (
                "lexicographically-sorted-source-ids-unordered-combinations"
            ),
            "prior_queue_precedence": True,
            "machine_clear_rule": (
                "hamming-distance-greater-than-8-and-no-equal-accession-family-"
                "or-burst-id"
            ),
            "perceptual_hash": (
                "rgb-luma-average-hash-8x8-bilinear-after-exif-transpose"
            ),
        }
        selected_ids = sorted(by_id)
        prior = {
            _pair_key(item["left_source_id"], item["right_source_id"]): item
            for item in review["queued_pair_reviews"][role]
        }
        expected_pairs: list[dict[str, Any]] = []
        flagged: list[dict[str, Any]] = []
        counts = {"prior_reviewed": 0, "machine_clear": 0, "new_flags": 0}
        for left_id, right_id in itertools.combinations(selected_ids, 2):
            left = by_id[left_id]
            right = by_id[right_id]
            distance = (
                int(left["perceptual_hash64"], 16) ^ int(right["perceptual_hash64"], 16)
            ).bit_count()
            shared = sorted(
                field
                for field in ("accession_family_id", "burst_id")
                if field in left["group_identity"]
                and field in right["group_identity"]
                and left["group_identity"][field] == right["group_identity"][field]
            )
            key = (left_id, right_id)
            if key in prior:
                queue = prior[key]
                if queue["relationship_decision"] != "distinct":
                    raise ValueError(f"{role} selected queued pair is not distinct")
                disposition = "prior_reviewed_distinct"
                evidence = {"queue_pair_id": queue["pair_id"]}
                counts["prior_reviewed"] += 1
            elif distance > 8 and not shared:
                disposition = "machine_clear"
                evidence = {
                    "rule": (
                        "perceptual_hamming_gt_8_and_no_accession_or_burst_collision"
                    )
                }
                counts["machine_clear"] += 1
            else:
                disposition = "targeted_visual_adjudication_pending_binding"
                evidence = {
                    "screen_prose_verdict": "distinct",
                    "screen_pair_mapping": (
                        "inferred_from-lexicographically-sorted-new-flag-order-and-"
                        "matching-description-not-explicitly-declared-by-source-record"
                    ),
                    "named_human_countersign_required": True,
                }
                counts["new_flags"] += 1
                flagged.append(
                    {
                        "left_source_id": left_id,
                        "right_source_id": right_id,
                        "hamming_distance": distance,
                        "shared_metadata_fields": shared,
                    }
                )
            expected_pairs.append(
                {
                    "left_source_id": left_id,
                    "right_source_id": right_id,
                    "hamming_distance": distance,
                    "shared_metadata_fields": shared,
                    "disposition": disposition,
                    "evidence": evidence,
                }
            )
        expected_counts = {**counts, "total": len(expected_pairs)}
        expected_inferred = (
            [
                {
                    "screen_label": f"D1-selrow-{index:02d}",
                    **item,
                    "screen_description": _SIMILARITY_DESCRIPTIONS[index - 1],
                    "screen_verdict": "distinct",
                    "binding_state": "pending_named_human_countersign",
                }
                for index, item in enumerate(flagged, 1)
            ]
            if role == "D1"
            else []
        )
        if (
            similarity["split_file_sha256"] != split_file_shas[role]
            or similarity["source_review_sha256"] != review["review_sha256"]
            or similarity["screen_markdown_file_sha256"] != screen_file_sha
            or similarity["method"] != expected_method
            or similarity["selected_source_ids"] != selected_ids
            or similarity["pair_counts"] != expected_counts
            or similarity["pairs"] != expected_pairs
            or similarity["inferred_screen_label_bindings"] != expected_inferred
            or similarity["screen_author"] != "Claude Fable 5 (agent; owner-authorized)"
        ):
            raise ValueError(f"{role} similarity evidence is not rederived")

        surfaces = _object(candidate["review_surfaces"], f"{role} review surfaces")
        expected_surface_paths = {
            "review_index": f"{role}/review-index.html",
            "targeted_pair_review": f"{role}/targeted-pair-review.html",
        }
        if set(surfaces) != set(expected_surface_paths):
            raise ValueError(f"{role} review surface schema is invalid")
        for name, relative in expected_surface_paths.items():
            surface = _object(surfaces[name], f"{role} {name}")
            _exact(surface, {"relative_path", "bytes", "sha256"}, f"{role} {name}")
            if surface != {
                "relative_path": relative,
                **_file_identity(root / relative),
            }:
                raise ValueError(f"{role} {name} changed or is unbound")
            expected_file_paths.add(relative)

        archive_record = _object(candidate["training_archive"], "training archive")
        _exact(
            archive_record,
            {"relative_path", "bytes", "sha256", "identity"},
            "training archive",
        )
        if archive_record["relative_path"] != "training.zip":
            raise ValueError(f"{role} training archive path is not literal")
        archive_relative = f"{role}/training.zip"
        archive = _safe_file(root / archive_relative, "training archive")
        archive_identity = _archive_identity(archive)
        if (
            _file_identity(archive)
            != {key: archive_record[key] for key in ("bytes", "sha256")}
            or archive_identity != archive_record["identity"]
        ):
            raise ValueError(f"{role} training archive changed")
        archive_members = {
            row["path"]: (row["bytes"], row["sha256"])
            for row in archive_identity["members"]
        }
        expected_members = {}
        for row in rows:
            if row["split"] != "training":
                continue
            for key, bytes_key, sha_key in (
                ("relative_image_path", "image_bytes", "image_sha256"),
                ("relative_caption_path", None, "caption_sha256"),
            ):
                member = Path(row[key]).name
                staged = _safe_file(root / role / row[key], "training member")
                expected_members[member] = (
                    staged.stat().st_size if bytes_key is None else row[bytes_key],
                    row[sha_key],
                )
        if archive_members != expected_members:
            raise ValueError(f"{role} archive members differ from staged training rows")
        expected_file_paths.update({manifest_relative, archive_relative})

    if candidate_digests != package["candidate_manifest_sha256s"]:
        raise ValueError("candidate manifest digest map is not exact")
    if candidate_digests != review_request["candidate_manifest_sha256s"]:
        raise ValueError("review request candidate digest map is not exact")
    _validate_exact_topology(package["files"], expected_file_paths)
    return package


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--review", required=True, type=Path)
    build.add_argument("--d1-split", required=True, type=Path)
    build.add_argument("--d2-split", required=True, type=Path)
    build.add_argument("--d2-commitment", required=True, type=Path)
    build.add_argument("--d2-secret", required=True, type=Path)
    build.add_argument("--d1-materialization", required=True, type=Path)
    build.add_argument("--d2-materialization", required=True, type=Path)
    build.add_argument("--tokenizer-evidence", required=True, type=Path)
    build.add_argument("--similarity-screen", required=True, type=Path)
    build.add_argument("--prepared-at-utc", required=True)
    build.add_argument("--output", required=True, type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("--package", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    if args.command == "build":
        result = build_package(
            review_path=args.review,
            d1_split_path=args.d1_split,
            d2_split_path=args.d2_split,
            d2_commitment_path=args.d2_commitment,
            d2_secret_path=args.d2_secret,
            d1_materialization_root=args.d1_materialization,
            d2_materialization_root=args.d2_materialization,
            tokenizer_evidence_path=args.tokenizer_evidence,
            similarity_screen_path=args.similarity_screen,
            prepared_at_utc=args.prepared_at_utc,
            output_root=args.output,
        )
    else:
        result = validate_package(args.package)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess smoke.
    raise SystemExit(main())
