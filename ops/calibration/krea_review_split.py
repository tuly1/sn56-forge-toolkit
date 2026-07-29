#!/usr/bin/env python3
"""Turn the signed D1/D2 workbook into deterministic, non-admitted split plans.

The workbook is a human-review surface.  This module distrusts its formulas,
recomputes every disposition, binds the reviewed source bytes to the sealed
inspection records, and emits canonical JSON for deterministic selection.
Selection never creates captions, fixture approval, or GPU authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import itertools
import json
import os
import re
import secrets
import stat
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree

try:
    from . import krea_provenance
except ImportError:  # pragma: no cover - direct script execution.
    import krea_provenance  # type: ignore[no-redef]


_SHA256 = re.compile(r"[0-9a-f]{64}")
_CELL = re.compile(r"([A-Z]+)([1-9][0-9]*)")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_BASE_POLICY = Path(__file__).with_name("week5") / "krea-curation-selection-policy.json"
_AMENDMENT = (
    Path(__file__).with_name("week5") / "krea-curation-selection-amendment.json"
)
_REVIEW_KIND = "forge-krea-d1d2-executable-review"
_SPLIT_KIND = "forge-krea-source-split-plan"
_COMMITMENT_KIND = "forge-krea-d2-split-key-commitment"
_EXPECTED_SHEETS = {
    "Instructions",
    "D1 Review",
    "D2 Review",
    "D1 Pairs",
    "D2 Pairs",
    "Evidence & Licenses",
    "Correction Ledger",
}
_REVIEW_HEADERS = [
    "source_id",
    "local_image_path",
    "source_page_url",
    "provider_title",
    "license_name",
    "license_url",
    "creator_or_artist",
    "attribution_or_credit_source",
    "native_dimensions",
    "decoded_dimensions",
    "bytes",
    "byte_sha256",
    "decoded_rgb_sha256",
    "perceptual_hash64",
    "provider_object_id",
    "creator_id_hint",
    "burst_id_hint",
    "play_root",
    "play_component",
    "accession_family",
    "automatic_cluster_id",
    "queued_pair_count",
    "machine_triage_hint",
    "rights_decision",
    "attribution_or_PD_record",
    "visual_decision",
    "concept_suitability",
    "quality_grade",
    "scene_or_impression_id",
    "human_similarity_cluster_id",
    "factual_caption",
    "reviewer_name",
    "review_date_utc",
    "reviewer_notes",
    "review_complete",
    "provisional_disposition",
]
_PAIR_HEADERS = [
    "pair_id",
    "left_source_id",
    "right_source_id",
    "hamming_distance",
    "same_play_component",
    "left_auto_cluster",
    "right_auto_cluster",
    "left_image_path",
    "right_image_path",
    "relationship_decision",
    "canonical_group_id",
    "reviewer_name",
    "review_notes",
    "review_complete",
]
_RIGHTS = {
    "approve_pd_or_cc0",
    "approve_cc_by_obligations_recorded",
    "reject_rights",
    "needs_escalation",
}
_VISUAL = {"approve", "reject", "needs_review"}
_CONCEPT = {"in_scope", "out_of_scope", "ambiguous"}
_QUALITY = {"A", "B", "C", "reject"}
_RELATIONSHIPS = {
    "same_impression_or_frame",
    "same_scene_or_burst",
    "near_duplicate",
    "distinct",
    "unclear",
}
_PAIR_MATCH = _RELATIONSHIPS - {"distinct", "unclear"}
_OWNER_EXCLUSIONS = {
    "commons-374687",
    "commons-384687",
    "commons-66539312",
    "aic-155333",
}
_CLAIM_LIMIT = (
    "review-and-source-split-only-captions-exhaustive-selected-pair-review-"
    "independent-approval-fixture-admission-and-gpu-authorization-remain-required"
)


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _text(value: Any, label: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(
            f"{label} must be a{' possibly empty' if empty else ' non-empty'} string"
        )
    if value != unicodedata.normalize("NFC", value):
        raise ValueError(f"{label} must use NFC Unicode")
    return value


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


def _atomic_create(path: Path, value: dict[str, Any]) -> None:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial-{os.getpid()}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(krea_provenance.canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _policy() -> tuple[dict[str, Any], str, dict[str, Any], str]:
    policy, policy_file_sha = _canonical_json(_BASE_POLICY, "selection policy")
    policy_sha = policy_file_sha
    amendment, amendment_file_sha = _canonical_json(_AMENDMENT, "selection amendment")
    amendment_sha = amendment_file_sha
    if (
        policy.get("kind") != "forge-krea-curation-selection-policy"
        or policy.get("schema") != 1
        or policy.get("admission_authorized") is not False
        or amendment.get("kind") != "forge-krea-curation-selection-amendment"
        or amendment.get("schema") != 1
        or amendment.get("base_policy_sha256") != policy_sha
        or amendment.get("gpu_execution_authorized") is not False
    ):
        raise ValueError("selection policy/amendment boundary is invalid")
    return policy, policy_sha, amendment, amendment_sha


def _column_number(letters: str) -> int:
    result = 0
    for letter in letters:
        result = result * 26 + ord(letter) - 64
    return result


def _zip_member(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in name
    ):
        raise ValueError(f"unsafe XLSX member: {name!r}")
    return name


def _xml(raw: bytes, label: str) -> ElementTree.Element:
    folded = raw.upper()
    if b"<!DOCTYPE" in folded or b"<!ENTITY" in folded:
        raise ValueError(f"{label} contains a forbidden XML declaration")
    try:
        return ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise ValueError(f"{label} is malformed XML") from exc


def _xlsx_sheets(path: Path) -> dict[str, list[dict[str, str]]]:
    path = _safe_file(path, "review workbook")
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("review workbook exceeds the 64 MiB cap")
    with zipfile.ZipFile(path) as archive:
        names: set[str] = set()
        total = 0
        for info in archive.infolist():
            name = _zip_member(info.filename)
            if name in names or info.flag_bits & 0x1:
                raise ValueError("XLSX has duplicate or encrypted members")
            names.add(name)
            total += info.file_size
            if (
                info.file_size > 1024 * 1024
                and info.compress_size > 0
                and info.file_size / info.compress_size > 1000
            ):
                raise ValueError("XLSX contains an implausibly compressed member")
            if total > 256 * 1024 * 1024:
                raise ValueError("XLSX expanded content exceeds the 256 MiB cap")
        required = {"xl/workbook.xml", "xl/_rels/workbook.xml.rels"}
        if not required.issubset(names):
            raise ValueError("XLSX is missing workbook metadata")
        if any(name.casefold().endswith("vbaproject.bin") for name in names):
            raise ValueError("macro-enabled workbooks are forbidden")
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            shared_root = _xml(
                archive.read("xl/sharedStrings.xml"), "XLSX shared strings"
            )
            for item in shared_root.findall("m:si", _NS):
                shared.append(
                    "".join(node.text or "" for node in item.findall(".//m:t", _NS))
                )
        workbook = _xml(archive.read("xl/workbook.xml"), "XLSX workbook")
        relationships = _xml(
            archive.read("xl/_rels/workbook.xml.rels"),
            "XLSX workbook relationships",
        )
        if any(
            item.attrib.get("TargetMode", "Internal") != "Internal"
            for item in relationships.findall(f"{{{_REL_NS}}}Relationship")
        ):
            raise ValueError("XLSX external relationships are forbidden")
        targets = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in relationships.findall(f"{{{_REL_NS}}}Relationship")
        }
        result: dict[str, list[dict[str, str]]] = {}
        for sheet in workbook.findall("m:sheets/m:sheet", _NS):
            name = sheet.attrib.get("name")
            relation = sheet.attrib.get(f"{{{_NS['r']}}}id")
            if (
                not name
                or relation not in targets
                or name in result
                or sheet.attrib.get("state", "visible") != "visible"
            ):
                raise ValueError("XLSX sheet metadata is malformed")
            target = targets[relation].lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            target = _zip_member(target)
            if target not in names or not target.startswith("xl/worksheets/"):
                raise ValueError(f"XLSX sheet target is unsafe: {target}")
            root = _xml(archive.read(target), f"XLSX sheet {name}")
            rows: list[dict[str, str]] = []
            previous = 0
            for row in root.findall(".//m:sheetData/m:row", _NS):
                number = int(row.attrib["r"])
                if number <= previous:
                    raise ValueError(f"{name} rows are not strictly ordered")
                previous = number
                values: dict[str, str] = {"__row__": str(number)}
                formula_columns: list[str] = []
                for cell in row.findall("m:c", _NS):
                    match = _CELL.fullmatch(cell.attrib.get("r", ""))
                    if match is None or int(match.group(2)) != number:
                        raise ValueError(f"{name} has an invalid cell reference")
                    column = match.group(1)
                    if column in values:
                        raise ValueError(f"{name} has a duplicate cell")
                    if cell.find("m:f", _NS) is not None:
                        formula_columns.append(column)
                    cell_type = cell.attrib.get("t")
                    if cell_type not in {None, "inlineStr", "s", "str", "n", "b"}:
                        raise ValueError(f"{name} has an unsupported cell type")
                    if cell_type == "inlineStr":
                        value = "".join(
                            node.text or "" for node in cell.findall(".//m:t", _NS)
                        )
                    else:
                        raw = cell.find("m:v", _NS)
                        value = "" if raw is None else raw.text or ""
                        if cell_type == "s":
                            try:
                                value = shared[int(value)]
                            except (ValueError, IndexError) as exc:
                                raise ValueError(
                                    "XLSX shared string index is invalid"
                                ) from exc
                    values[column] = unicodedata.normalize("NFC", value)
                values["__formulas__"] = ",".join(sorted(formula_columns))
                rows.append(values)
            result[name] = rows
    if set(result) != _EXPECTED_SHEETS:
        raise ValueError(
            f"workbook sheets mismatch: expected={sorted(_EXPECTED_SHEETS)}, "
            f"actual={sorted(result)}"
        )
    return result


def _letters(count: int) -> list[str]:
    values = []
    for number in range(1, count + 1):
        current = number
        letters = ""
        while current:
            current, remainder = divmod(current - 1, 26)
            letters = chr(65 + remainder) + letters
        values.append(letters)
    return values


def _table(
    rows: list[dict[str, str]],
    *,
    header_row: int,
    headers: list[str],
    expected_count: int,
    label: str,
    formula_columns: set[str],
) -> list[dict[str, str]]:
    by_number = {int(row["__row__"]): row for row in rows}
    header = by_number.get(header_row)
    columns = _letters(len(headers))
    if header is None or [header.get(column, "") for column in columns] != headers:
        raise ValueError(f"{label} header is not exact")
    values = []
    for number in range(header_row + 1, header_row + 1 + expected_count):
        row = by_number.get(number)
        if row is None:
            raise ValueError(f"{label} is missing row {number}")
        mapped = {
            header_name: row.get(column, "")
            for column, header_name in zip(columns, headers)
        }
        observed_formulas = set(filter(None, row.get("__formulas__", "").split(",")))
        if observed_formulas != formula_columns:
            raise ValueError(
                f"{label} row {number} formulas mismatch: "
                f"expected={sorted(formula_columns)}, actual={sorted(observed_formulas)}"
            )
        if not mapped[headers[0]]:
            raise ValueError(f"{label} row {number} has no identity")
        values.append(mapped)
    later = [
        row
        for number, row in by_number.items()
        if number >= header_row + 1 + expected_count
        and any(
            value
            for key, value in row.items()
            if key not in {"__row__", "__formulas__"}
        )
    ]
    if later:
        raise ValueError(f"{label} has unexpected rows after its sealed range")
    return values


def _parse_dimensions(value: str, label: str) -> tuple[int, int]:
    match = re.fullmatch(r"([1-9][0-9]*)×([1-9][0-9]*)", value)
    if match is None:
        raise ValueError(f"{label} must use WIDTH×HEIGHT")
    return int(match.group(1)), int(match.group(2))


def _inspection(path: Path, role: str) -> tuple[dict[str, Any], str]:
    record, file_sha = _canonical_json(path, f"{role} inspection")
    body = {key: value for key, value in record.items() if key != "inspection_sha256"}
    if (
        record.get("schema") != 2
        or record.get("kind") != "forge-krea-source-candidate-inspection"
        or record.get("experimental_role") != role
        or record.get("inspection_sha256") != krea_provenance.canonical_sha256(body)
        or record.get("admission_state") != "candidate_unreviewed"
        or record.get("gpu_execution_authorized") is not False
    ):
        raise ValueError(f"{role} inspection boundary is invalid")
    return record, file_sha


def _disposition(row: dict[str, str]) -> str:
    rights = row["rights_decision"]
    visual = row["visual_decision"]
    concept = row["concept_suitability"]
    quality = row["quality_grade"]
    if (
        rights == "reject_rights"
        or visual == "reject"
        or concept == "out_of_scope"
        or quality == "reject"
    ):
        return "EXCLUDE"
    if (
        rights == "needs_escalation"
        or visual == "needs_review"
        or concept == "ambiguous"
    ):
        return "ESCALATE"
    return "CANDIDATE_ONLY_NOT_ADMITTED"


def _normalize_caption(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _normalize_group_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    plain = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", plain.casefold()).split())


def _group_value(value: str, field: str, source_id: str) -> str:
    return value if value else f"not-applicable:{field}:{source_id}"


def _review_rows(
    workbook_rows: list[dict[str, str]], inspection: dict[str, Any], role: str
) -> list[dict[str, Any]]:
    inspected = {row["source_id"]: row for row in inspection["rows"]}
    queue_counts: Counter[str] = Counter()
    for pair in inspection["machine_qc"]["human_review_queue"]:
        queue_counts[pair["left"]] += 1
        queue_counts[pair["right"]] += 1
    if len(inspected) != len(workbook_rows):
        raise ValueError(f"{role} workbook/inspection row counts differ")
    output = []
    seen: set[str] = set()
    for raw in workbook_rows:
        source_id = raw["source_id"]
        if source_id in seen or source_id not in inspected:
            raise ValueError(f"{role} has duplicate or unknown source_id: {source_id}")
        seen.add(source_id)
        machine = inspected[source_id]
        width, height = _parse_dimensions(
            raw["decoded_dimensions"], f"{source_id}.decoded_dimensions"
        )
        expected = {
            "provider_title": machine["provider_title"],
            "byte_sha256": machine["byte_sha256"],
            "decoded_rgb_sha256": machine["decoded_rgb_sha256"],
            "perceptual_hash64": machine["perceptual_hash64"],
        }
        if any(raw[key] != value for key, value in expected.items()) or (
            width,
            height,
        ) != (machine["width"], machine["height"]):
            raise ValueError(f"{role} sealed machine fields drifted for {source_id}")
        try:
            byte_count = int(raw["bytes"])
            queued = int(raw["queued_pair_count"])
        except ValueError as exc:
            raise ValueError(f"{source_id} numeric fields are invalid") from exc
        image = _safe_file(Path(raw["local_image_path"]), f"{source_id} image")
        if (
            image.name != Path(machine["relative_path"]).name
            or image.stat().st_size != byte_count
            or _file_sha256(image) != raw["byte_sha256"]
            or queued != queue_counts[source_id]
        ):
            raise ValueError(f"{role} source-byte binding failed for {source_id}")
        if raw["rights_decision"] not in _RIGHTS:
            raise ValueError(f"{source_id} rights decision is invalid")
        if raw["visual_decision"] not in _VISUAL:
            raise ValueError(f"{source_id} visual decision is invalid")
        if raw["concept_suitability"] not in _CONCEPT:
            raise ValueError(f"{source_id} concept decision is invalid")
        if raw["quality_grade"] not in _QUALITY:
            raise ValueError(f"{source_id} quality grade is invalid")
        for key in (
            "attribution_or_PD_record",
            "scene_or_impression_id",
            "human_similarity_cluster_id",
            "factual_caption",
            "reviewer_name",
            "review_date_utc",
        ):
            _text(raw[key], f"{source_id}.{key}")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw["review_date_utc"]):
            raise ValueError(f"{source_id} review date is not canonical UTC date")
        disposition = _disposition(raw)
        if disposition == "ESCALATE" or raw["provisional_disposition"] != disposition:
            raise ValueError(f"{source_id} has an unresolved or incorrect disposition")
        if raw["review_complete"] != "FIELDS_COMPLETE":
            raise ValueError(f"{source_id} review is incomplete")
        license_folded = raw["license_name"].casefold()
        if "sharealike" in license_folded or "by-sa" in license_folded:
            raise ValueError(f"{source_id} ShareAlike material is forbidden")
        if raw["rights_decision"] == "approve_cc_by_obligations_recorded" and (
            raw["source_page_url"] not in raw["attribution_or_PD_record"]
            or not raw["license_url"]
            or raw["license_url"] not in raw["attribution_or_PD_record"]
        ):
            raise ValueError(f"{source_id} CC BY attribution is not self-contained")
        hints = machine["automatic_group_hints"]
        sealed_hints = {
            "creator_id_hint": hints.get("creator_id") or "",
            "burst_id_hint": hints.get("burst_id") or "",
            "play_component": hints.get("play_component") or "",
            "accession_family": hints.get("accession_family") or "",
        }
        if any(raw[key] != value for key, value in sealed_hints.items()) or (
            _normalize_group_text(raw["play_root"]) != (hints.get("play_root") or "")
        ):
            raise ValueError(f"{role} automatic group hints drifted for {source_id}")
        if role == "D1" and (not raw["creator_id_hint"] or not raw["burst_id_hint"]):
            raise ValueError(f"{source_id} lacks D1 creator/burst identity")
        if role == "D2" and (
            not raw["play_root"]
            or not raw["play_component"]
            or not raw["accession_family"]
        ):
            raise ValueError(f"{source_id} lacks D2 protected identity")
        output.append(
            {
                "source_id": source_id,
                "relative_materialized_path": machine["relative_path"],
                "source_page_url": raw["source_page_url"],
                "provider_title": raw["provider_title"],
                "license_name": raw["license_name"],
                "license_url": raw["license_url"],
                "creator_or_artist": raw["creator_or_artist"],
                "attribution_or_pd_record": raw["attribution_or_PD_record"],
                "byte_count": byte_count,
                "byte_sha256": raw["byte_sha256"],
                "decoded_rgb_sha256": raw["decoded_rgb_sha256"],
                "perceptual_hash64": raw["perceptual_hash64"],
                "width": width,
                "height": height,
                "group_identity": {
                    "source_id": source_id,
                    "creator_id": _group_value(
                        raw["creator_id_hint"], "creator", source_id
                    ),
                    "burst_id": _group_value(raw["burst_id_hint"], "burst", source_id),
                    "scene_id": raw["scene_or_impression_id"],
                    "play_root_id": _group_value(
                        hints.get("play_root") or "", "play-root", source_id
                    ),
                    "play_component_id": _group_value(
                        raw["play_component"], "play-component", source_id
                    ),
                    "accession_family_id": _group_value(
                        raw["accession_family"], "accession-family", source_id
                    ),
                    "human_similarity_cluster_id": raw["human_similarity_cluster_id"],
                },
                "quality_grade": raw["quality_grade"],
                "factual_caption": raw["factual_caption"],
                "normalized_factual_caption_sha256": hashlib.sha256(
                    _normalize_caption(raw["factual_caption"]).encode("utf-8")
                ).hexdigest(),
                "reviewer_identity": raw["reviewer_name"],
                "review_date_utc": raw["review_date_utc"],
                "review_notes": raw["reviewer_notes"],
                "rights_decision": raw["rights_decision"],
                "visual_decision": raw["visual_decision"],
                "concept_suitability": raw["concept_suitability"],
                "disposition": disposition,
            }
        )
    if set(inspected) != seen:
        raise ValueError(f"{role} workbook does not cover the inspection exactly")
    output.sort(key=lambda row: row["source_id"])
    return output


def _pair_rows(
    workbook_pairs: list[dict[str, str]],
    inspection: dict[str, Any],
    role: str,
    review_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected = inspection["machine_qc"]["human_review_queue"]
    if len(expected) != len(workbook_pairs):
        raise ValueError(f"{role} pair queue count differs")
    rows = {row["source_id"]: row for row in review_rows}
    output = []
    for index, (raw, machine) in enumerate(zip(workbook_pairs, expected), start=1):
        expected_id = f"{role}-pair-{index:04d}"
        if (
            raw["pair_id"] != expected_id
            or raw["left_source_id"] != machine["left"]
            or raw["right_source_id"] != machine["right"]
        ):
            raise ValueError(f"{role} pair queue ordering drifted at {expected_id}")
        try:
            distance = int(raw["hamming_distance"])
            same_play = bool(int(raw["same_play_component"]))
        except ValueError as exc:
            raise ValueError(f"{expected_id} machine fields are invalid") from exc
        if (
            distance != machine["hamming_distance"]
            or same_play != machine["same_play_component"]
            or raw["relationship_decision"] not in _RELATIONSHIPS
            or raw["relationship_decision"] == "unclear"
            or raw["review_complete"] != "PAIR_REVIEW_COMPLETE"
            or not raw["reviewer_name"]
            or not raw["review_notes"]
        ):
            raise ValueError(f"{expected_id} is incomplete or invalid")
        decision = raw["relationship_decision"]
        left = rows[raw["left_source_id"]]
        right = rows[raw["right_source_id"]]
        if decision in _PAIR_MATCH:
            if (
                not raw["canonical_group_id"]
                or left["group_identity"]["scene_id"]
                != right["group_identity"]["scene_id"]
                or left["group_identity"]["human_similarity_cluster_id"]
                != right["group_identity"]["human_similarity_cluster_id"]
            ):
                raise ValueError(f"{expected_id} match is not bound to row groups")
        else:
            expected_group = "distinct:" + "|".join(
                sorted((raw["left_source_id"], raw["right_source_id"]))
            )
            if raw["canonical_group_id"] != expected_group:
                raise ValueError(
                    f"{expected_id} distinct decision lacks its canonical group"
                )
        output.append(
            {
                "pair_id": expected_id,
                "left_source_id": raw["left_source_id"],
                "right_source_id": raw["right_source_id"],
                "hamming_distance": distance,
                "same_play_component": same_play,
                "relationship_decision": decision,
                "canonical_group_id": raw["canonical_group_id"],
                "reviewer_identity": raw["reviewer_name"],
                "review_notes": raw["review_notes"],
            }
        )
    return output


def _owner_summary(path: Path) -> tuple[str, str]:
    path = _safe_file(path, "owner summary")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("owner summary must be UTF-8") from exc
    required = [
        "- [x] I ratify the four exclusions listed above.",
        "- [x] I accept the review record with corrections applied.",
        "Signed (owner): **Atulya Shetty**",
        *sorted(_OWNER_EXCLUSIONS),
    ]
    if any(item not in text for item in required):
        raise ValueError("owner summary does not contain the exact ratification")
    return hashlib.sha256(raw).hexdigest(), "Atulya Shetty"


def _json_file(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = _safe_file(path, label)
    raw = path.read_bytes()
    try:
        value = _object(json.loads(raw), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid JSON") from exc
    return value, hashlib.sha256(raw).hexdigest()


def _external_review_bindings(
    *,
    correction_ledger: Path,
    corrected_audit: Path,
    workbook_sha256: str,
    owner_summary_sha256: str,
    d1_inspection: dict[str, Any],
    d2_inspection: dict[str, Any],
) -> dict[str, str]:
    ledger, ledger_file_sha = _json_file(
        correction_ledger, "external correction ledger"
    )
    if (
        ledger.get("schema") != 1
        or ledger.get("kind") != "forge-krea-signed-review-mechanical-correction-ledger"
        or ledger.get("corrected_workbook_sha256") != workbook_sha256
        or ledger.get("owner_signoff_summary_sha256") != owner_summary_sha256
        or ledger.get("correction_count") != 66
        or ledger.get("disposition_changes") != 4
        or not isinstance(ledger.get("corrections"), list)
        or len(ledger["corrections"]) != 66
        or "no-fixture-admission-or-gpu-authorization"
        not in ledger.get("claim_limit", "")
    ):
        raise ValueError("external correction ledger binding is invalid")
    audit, audit_file_sha = _json_file(corrected_audit, "corrected workbook audit")
    bindings = _object(audit.get("sourceBindings"), "corrected audit bindings")
    if (
        audit.get("schema") != 1
        or audit.get("goForSplitGeneration") is not True
        or bindings.get("workbookSha256") != workbook_sha256
        or bindings.get("ownerSummarySha256") != owner_summary_sha256
        or bindings.get("d1MaterializationSha256")
        != d1_inspection["materialization_sha256"]
        or bindings.get("d2MaterializationSha256")
        != d2_inspection["materialization_sha256"]
        or bindings.get("d1InspectionSha256") != d1_inspection["inspection_sha256"]
        or bindings.get("d2InspectionSha256") != d2_inspection["inspection_sha256"]
    ):
        raise ValueError("corrected workbook audit is not green for these inputs")
    return {
        "external_correction_ledger_file_sha256": ledger_file_sha,
        "corrected_audit_file_sha256": audit_file_sha,
    }


def _correction_ledger(
    rows: list[dict[str, str]], owner_summary_sha256: str
) -> dict[str, Any]:
    cells = {
        f"{column}{row['__row__']}": value
        for row in rows
        for column, value in row.items()
        if column != "__row__"
    }
    try:
        count = int(cells["B5"])
        resolved = int(cells["B6"])
    except (KeyError, ValueError) as exc:
        raise ValueError("correction ledger metadata is invalid") from exc
    if (
        not _SHA256.fullmatch(cells.get("B3", ""))
        or cells.get("B4") != owner_summary_sha256
        or count != 66
        or resolved != 4
        or cells.get("B7") != "None"
    ):
        raise ValueError("correction ledger does not bind the owner-approved repair")
    entries = [row for row in rows if int(row["__row__"]) >= 10 and row.get("A", "")]
    if len(entries) != count:
        raise ValueError("correction ledger entry count differs")
    return {
        "original_workbook_sha256": cells["B3"],
        "owner_summary_sha256": cells["B4"],
        "correction_count": count,
        "ratified_exclusion_count": resolved,
        "admission_or_gpu_authorization": False,
    }


def export_review(
    *,
    workbook: Path,
    owner_summary: Path,
    d1_inspection: Path,
    d2_inspection: Path,
    correction_ledger: Path,
    corrected_audit: Path,
) -> dict[str, Any]:
    """Validate the signed surface and return canonical executable evidence."""

    policy, policy_sha, amendment, amendment_sha = _policy()
    del policy, amendment
    workbook = _safe_file(workbook, "review workbook")
    workbook_sha = _file_sha256(workbook)
    owner_sha, owner_identity = _owner_summary(owner_summary)
    sheets = _xlsx_sheets(workbook)
    ledger = _correction_ledger(sheets["Correction Ledger"], owner_sha)
    d1_inspection_record, d1_file_sha = _inspection(d1_inspection, "D1")
    d2_inspection_record, d2_file_sha = _inspection(d2_inspection, "D2")
    external_bindings = _external_review_bindings(
        correction_ledger=correction_ledger,
        corrected_audit=corrected_audit,
        workbook_sha256=workbook_sha,
        owner_summary_sha256=owner_sha,
        d1_inspection=d1_inspection_record,
        d2_inspection=d2_inspection_record,
    )
    if any(
        record.get("selection_policy_sha256") != policy_sha
        for record in (d1_inspection_record, d2_inspection_record)
    ):
        raise ValueError("inspection does not bind the frozen selection policy")
    d1_raw = _table(
        sheets["D1 Review"],
        header_row=5,
        headers=_REVIEW_HEADERS,
        expected_count=84,
        label="D1 review",
        formula_columns={"AI", "AJ"},
    )
    d2_raw = _table(
        sheets["D2 Review"],
        header_row=5,
        headers=_REVIEW_HEADERS,
        expected_count=222,
        label="D2 review",
        formula_columns={"AI", "AJ"},
    )
    d1_pair_raw = _table(
        sheets["D1 Pairs"],
        header_row=4,
        headers=_PAIR_HEADERS,
        expected_count=20,
        label="D1 pair review",
        formula_columns={"N"},
    )
    d2_pair_raw = _table(
        sheets["D2 Pairs"],
        header_row=4,
        headers=_PAIR_HEADERS,
        expected_count=396,
        label="D2 pair review",
        formula_columns={"N"},
    )
    d1_rows = _review_rows(d1_raw, d1_inspection_record, "D1")
    d2_rows = _review_rows(d2_raw, d2_inspection_record, "D2")
    d1_pairs = _pair_rows(d1_pair_raw, d1_inspection_record, "D1", d1_rows)
    d2_pairs = _pair_rows(d2_pair_raw, d2_inspection_record, "D2", d2_rows)
    counts = {
        "D1": Counter(row["disposition"] for row in d1_rows),
        "D2": Counter(row["disposition"] for row in d2_rows),
    }
    if counts["D1"] != {"CANDIDATE_ONLY_NOT_ADMITTED": 68, "EXCLUDE": 16} or counts[
        "D2"
    ] != {"CANDIDATE_ONLY_NOT_ADMITTED": 221, "EXCLUDE": 1}:
        raise ValueError("review outcomes differ from the owner-ratified record")
    excluded = {
        row["source_id"] for row in d1_rows + d2_rows if row["disposition"] == "EXCLUDE"
    }
    if not _OWNER_EXCLUSIONS.issubset(excluded):
        raise ValueError("owner-ratified exclusions are not all excluded")
    body = {
        "schema": 1,
        "kind": _REVIEW_KIND,
        "workbook": {
            "sha256": workbook_sha,
            "correction_ledger": ledger,
            **external_bindings,
        },
        "owner_signoff": {
            "owner_identity": owner_identity,
            "summary_sha256": owner_sha,
            "decision": "accepted_with_corrections",
            "ratified_exclusions": sorted(_OWNER_EXCLUSIONS),
        },
        "selection_policy_sha256": policy_sha,
        "selection_amendment_sha256": amendment_sha,
        "tool_identity": {
            "algorithm": "xlsx-review-export-v1",
            "source_sha256": _file_sha256(Path(__file__).resolve(strict=True)),
        },
        "source_evidence": {
            "D1": {
                "inspection_file_sha256": d1_file_sha,
                "inspection_sha256": d1_inspection_record["inspection_sha256"],
                "materialization_sha256": d1_inspection_record[
                    "materialization_sha256"
                ],
            },
            "D2": {
                "inspection_file_sha256": d2_file_sha,
                "inspection_sha256": d2_inspection_record["inspection_sha256"],
                "materialization_sha256": d2_inspection_record[
                    "materialization_sha256"
                ],
            },
        },
        "records": {"D1": d1_rows, "D2": d2_rows},
        "queued_pair_reviews": {"D1": d1_pairs, "D2": d2_pairs},
        "counts": {
            role: {
                "reviewed": len(rows),
                "candidates": counts[role]["CANDIDATE_ONLY_NOT_ADMITTED"],
                "excluded": counts[role]["EXCLUDE"],
                "queued_pairs_reviewed": len(d1_pairs if role == "D1" else d2_pairs),
            }
            for role, rows in (("D1", d1_rows), ("D2", d2_rows))
        },
        "selection_state": "review_validated_split_pending",
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": _CLAIM_LIMIT,
    }
    result = {**body, "review_sha256": krea_provenance.canonical_sha256(body)}
    validate_review(result)
    return result


def validate_review(value: dict[str, Any]) -> dict[str, Any]:
    value = _object(value, "executable review")
    expected_keys = {
        "schema",
        "kind",
        "workbook",
        "owner_signoff",
        "selection_policy_sha256",
        "selection_amendment_sha256",
        "tool_identity",
        "source_evidence",
        "records",
        "queued_pair_reviews",
        "counts",
        "selection_state",
        "admission_authorized",
        "gpu_execution_authorized",
        "claim_limit",
        "review_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError("executable review keys mismatch")
    body = {key: item for key, item in value.items() if key != "review_sha256"}
    _, policy_sha, amendment, amendment_sha = _policy()
    binding = _object(
        amendment.get("executable_review_binding"), "executable review binding"
    )
    if set(binding) != {"payload_fields", "payload_sha256"}:
        raise ValueError("executable review binding keys mismatch")
    payload_fields = binding["payload_fields"]
    if (
        not isinstance(payload_fields, list)
        or payload_fields != sorted(set(payload_fields))
        or any(
            field not in expected_keys - {"review_sha256"} for field in payload_fields
        )
        or not _SHA256.fullmatch(str(binding["payload_sha256"]))
    ):
        raise ValueError("executable review payload binding is invalid")
    payload = {field: value[field] for field in payload_fields}
    if (
        value.get("schema") != 1
        or value.get("kind") != _REVIEW_KIND
        or value.get("review_sha256") != krea_provenance.canonical_sha256(body)
        or krea_provenance.canonical_sha256(payload) != binding["payload_sha256"]
        or value.get("selection_policy_sha256") != policy_sha
        or value.get("selection_amendment_sha256") != amendment_sha
        or value.get("tool_identity")
        != {
            "algorithm": "xlsx-review-export-v1",
            "source_sha256": _file_sha256(Path(__file__).resolve(strict=True)),
        }
        or value.get("admission_authorized") is not False
        or value.get("gpu_execution_authorized") is not False
        or value.get("selection_state") != "review_validated_split_pending"
        or value.get("claim_limit") != _CLAIM_LIMIT
    ):
        raise ValueError("executable review boundary or digest is invalid")
    records = _object(value.get("records"), "review records")
    if set(records) != {"D1", "D2"}:
        raise ValueError("review must contain D1 and D2 records")
    expected = {"D1": (84, 68), "D2": (222, 221)}
    for role, (total, candidates) in expected.items():
        rows = records[role]
        if (
            not isinstance(rows, list)
            or len(rows) != total
            or rows != sorted(rows, key=lambda row: row["source_id"])
            or len({row["source_id"] for row in rows}) != total
            or sum(row["disposition"] == "CANDIDATE_ONLY_NOT_ADMITTED" for row in rows)
            != candidates
        ):
            raise ValueError(f"{role} executable records are invalid")
    return value


def _read_review(path: Path) -> tuple[dict[str, Any], str]:
    review, file_sha = _canonical_json(path, "executable review")
    validate_review(review)
    return review, file_sha


def _orientation(row: dict[str, Any]) -> str:
    if row["width"] > row["height"]:
        return "landscape"
    if row["height"] > row["width"]:
        return "portrait"
    return "square"


def _rank_integer(policy_sha: str, role: str, source_id: str) -> int:
    payload = f"{policy_sha}\0D1\0{role}\0{source_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _d1_options(
    rows: list[dict[str, Any]],
    *,
    policy_sha: str,
    duplicate_bits: dict[str, int],
) -> list[dict[str, Any]]:
    options = [
        {
            "role": "unused",
            "ids": (),
            "count": 0,
            "landscape": 0,
            "portrait": 0,
            "quality_a": 0,
            "quality_b": 0,
            "tie": 0,
            "caption_mask": 0,
        }
    ]
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for count in range(1, min(3, len(rows)) + 1):
        for chosen in itertools.combinations(rows, count):
            captions = [row["normalized_factual_caption_sha256"] for row in chosen]
            if len(captions) != len(set(captions)):
                continue
            if any(
                len({row["group_identity"][field] for row in chosen}) != len(chosen)
                for field in ("scene_id", "human_similarity_cluster_id")
            ):
                continue
            mask = 0
            for caption in captions:
                if caption in duplicate_bits:
                    mask |= 1 << duplicate_bits[caption]
            for role in ("training", "evaluation"):
                value = {
                    "role": role,
                    "ids": tuple(sorted(row["source_id"] for row in chosen)),
                    "count": count,
                    "landscape": sum(
                        _orientation(row) == "landscape" for row in chosen
                    ),
                    "portrait": sum(_orientation(row) == "portrait" for row in chosen),
                    "quality_a": sum(row["quality_grade"] == "A" for row in chosen),
                    "quality_b": sum(row["quality_grade"] == "B" for row in chosen),
                    "tie": sum(
                        _rank_integer(policy_sha, role, row["source_id"])
                        for row in chosen
                    ),
                    "caption_mask": mask,
                }
                signature = (
                    role,
                    count,
                    value["landscape"],
                    value["portrait"],
                    mask,
                )
                incumbent = best.get(signature)
                candidate_key = (
                    -value["quality_a"],
                    -value["quality_b"],
                    value["tie"],
                    value["ids"],
                )
                incumbent_key = (
                    (
                        -incumbent["quality_a"],
                        -incumbent["quality_b"],
                        incumbent["tie"],
                        incumbent["ids"],
                    )
                    if incumbent is not None
                    else None
                )
                if incumbent_key is None or candidate_key < incumbent_key:
                    best[signature] = value
    options.extend(best.values())
    return options


def _d1_dynamic_program(
    by_creator: dict[str, list[dict[str, Any]]],
    *,
    policy_sha: str,
    duplicate_bits: dict[str, int],
    training_target: int,
    evaluation_target: int,
) -> tuple[
    tuple[Any, ...], tuple[int, ...], tuple[Any, ...], tuple[str, ...], tuple[str, ...]
]:
    """Return the globally optimal D1 allocation for fixed role counts."""

    # State contains train/eval counts, orientation counts, global caption
    # conflicts, and the maximum contribution seen so far.  Maximum must be a
    # state dimension: a later creator can raise two different prefix maxima to
    # the same final value, so pruning the higher-quality prefix merely because
    # its *current* maximum is larger would not preserve global optimality.
    initial = (0, 0, 0, 0, 0, 0, 0, 0)
    states: dict[
        tuple[int, ...],
        tuple[int, int, int, int, tuple[str, ...], tuple[str, ...]],
    ] = {initial: (0, 0, 0, 0, (), ())}
    for creator in sorted(by_creator):
        options = _d1_options(
            sorted(by_creator[creator], key=lambda row: row["source_id"]),
            policy_sha=policy_sha,
            duplicate_bits=duplicate_bits,
        )
        next_states: dict[
            tuple[int, ...],
            tuple[int, int, int, int, tuple[str, ...], tuple[str, ...]],
        ] = {}
        for state, value in states.items():
            tn, en, tl, tp, el, ep, used_mask, maximum = state
            distinct, quality_a, quality_b, tie, training, evaluation = value
            for option in options:
                if option["caption_mask"] & used_mask:
                    continue
                new_tn = tn + (option["count"] if option["role"] == "training" else 0)
                new_en = en + (option["count"] if option["role"] == "evaluation" else 0)
                if new_tn > training_target or new_en > evaluation_target:
                    continue
                selected = option["count"] > 0
                new_state = (
                    new_tn,
                    new_en,
                    tl + (option["landscape"] if option["role"] == "training" else 0),
                    tp + (option["portrait"] if option["role"] == "training" else 0),
                    el + (option["landscape"] if option["role"] == "evaluation" else 0),
                    ep + (option["portrait"] if option["role"] == "evaluation" else 0),
                    used_mask | option["caption_mask"],
                    max(maximum, option["count"]),
                )
                new_value = (
                    distinct + int(selected),
                    quality_a + option["quality_a"],
                    quality_b + option["quality_b"],
                    tie + option["tie"],
                    tuple(
                        sorted(
                            training
                            + (option["ids"] if option["role"] == "training" else ())
                        )
                    ),
                    tuple(
                        sorted(
                            evaluation
                            + (option["ids"] if option["role"] == "evaluation" else ())
                        )
                    ),
                )
                incumbent = next_states.get(new_state)
                candidate_key = (
                    -new_value[0],
                    -new_value[1],
                    -new_value[2],
                    new_value[3],
                    new_value[4],
                    new_value[5],
                )
                incumbent_key = (
                    (
                        -incumbent[0],
                        -incumbent[1],
                        -incumbent[2],
                        incumbent[3],
                        incumbent[4],
                        incumbent[5],
                    )
                    if incumbent is not None
                    else None
                )
                if incumbent_key is None or candidate_key < incumbent_key:
                    next_states[new_state] = new_value
        states = next_states
    finals = []
    for state, value in states.items():
        tn, en, tl, tp, el, ep, _, maximum = state
        if (tn, en) != (training_target, evaluation_target):
            continue
        ts = tn - tl - tp
        es = en - el - ep
        distance = (
            abs(evaluation_target * tl - training_target * el)
            + abs(evaluation_target * tp - training_target * ep)
            + abs(evaluation_target * ts - training_target * es)
        )
        distinct, quality_a, quality_b, tie, training, evaluation = value
        finals.append(
            (
                (
                    -distinct,
                    maximum,
                    distance,
                    -quality_a,
                    -quality_b,
                    tie,
                    training,
                    evaluation,
                ),
                state,
                value,
                training,
                evaluation,
            )
        )
    if not finals:
        raise ValueError(
            f"D1 exact {training_target}/{evaluation_target} split is infeasible"
        )
    return min(finals, key=lambda item: item[0])


def select_d1(review: dict[str, Any]) -> dict[str, Any]:
    """Solve the frozen D1 objective exactly over the owner-approved pool."""

    review = validate_review(review)
    _, policy_sha, _, amendment_sha = _policy()
    candidates = [
        row
        for row in review["records"]["D1"]
        if row["disposition"] == "CANDIDATE_ONLY_NOT_ADMITTED"
    ]
    if len(candidates) - 42 < 8:
        raise ValueError("D1 has fewer than eight accepted unused reserves")
    by_creator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_creator[row["group_identity"]["creator_id"]].append(row)
    protected = ("burst_id", "scene_id", "human_similarity_cluster_id")
    for field in protected:
        owners: dict[str, set[str]] = defaultdict(set)
        for row in candidates:
            owners[row["group_identity"][field]].add(
                row["group_identity"]["creator_id"]
            )
        crossing = {key: value for key, value in owners.items() if len(value) > 1}
        if crossing:
            raise ValueError(
                f"D1 {field} crosses creators and needs one allocation unit"
            )
    caption_counts = Counter(
        row["normalized_factual_caption_sha256"] for row in candidates
    )
    duplicate_bits = {
        caption: index
        for index, caption in enumerate(
            sorted(key for key, count in caption_counts.items() if count > 1)
        )
    }
    objective, state, value, training, evaluation = _d1_dynamic_program(
        by_creator,
        policy_sha=policy_sha,
        duplicate_bits=duplicate_bits,
        training_target=18,
        evaluation_target=24,
    )
    selected = set(training) | set(evaluation)
    reserve = sorted(
        row["source_id"] for row in candidates if row["source_id"] not in selected
    )
    by_id = {row["source_id"]: row for row in candidates}
    for field in ("creator_id", "burst_id", "scene_id", "human_similarity_cluster_id"):
        train_groups = {by_id[item]["group_identity"][field] for item in training}
        eval_groups = {by_id[item]["group_identity"][field] for item in evaluation}
        if train_groups & eval_groups:
            raise AssertionError(f"D1 solver leaked {field}")
    _, _, tl, tp, el, ep, _, maximum = state
    distinct, quality_a, quality_b, _, _, _ = value
    body = {
        "schema": 1,
        "kind": _SPLIT_KIND,
        "experimental_role": "D1",
        "source_review_sha256": review["review_sha256"],
        "selection_policy_sha256": policy_sha,
        "selection_amendment_sha256": amendment_sha,
        "selection_tool_source_sha256": _file_sha256(
            Path(__file__).resolve(strict=True)
        ),
        "selection_algorithm": "exact_dynamic_programming_v2",
        "training_source_ids": list(training),
        "evaluation_source_ids": list(evaluation),
        "unused_accepted_reserve_source_ids": reserve,
        "objective": {
            "distinct_selected_creators": distinct,
            "maximum_selected_rows_per_creator": maximum,
            "orientation_distance": objective[2],
            "selected_A_count": quality_a,
            "selected_B_count": quality_b,
            "additive_sha256_tiebreak_integer": str(objective[5]),
            "training_orientation_counts": {
                "landscape": tl,
                "portrait": tp,
                "square": 18 - tl - tp,
            },
            "evaluation_orientation_counts": {
                "landscape": el,
                "portrait": ep,
                "square": 24 - el - ep,
            },
        },
        "pending_gates": [
            "tokenizer_checked_rare_trigger_and_training_captions",
            "exhaustive_selected_row_similarity_review_861_pairs",
            "selected_per_file_rights_and_attribution_record",
            "fixture_manifest_and_archive",
            "response_engineer_countersign",
            "independent_reviewer_approval",
            "all_six_fixture_cross_review",
        ],
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": _CLAIM_LIMIT,
    }
    result = {**body, "split_sha256": krea_provenance.canonical_sha256(body)}
    validate_d1_split(result, review, rederive=False)
    return result


def validate_d1_split(
    value: dict[str, Any], review: dict[str, Any], *, rederive: bool = True
) -> dict[str, Any]:
    value = _object(value, "D1 split")
    body = {key: item for key, item in value.items() if key != "split_sha256"}
    if (
        value.get("schema") != 1
        or value.get("kind") != _SPLIT_KIND
        or value.get("experimental_role") != "D1"
        or value.get("split_sha256") != krea_provenance.canonical_sha256(body)
        or value.get("source_review_sha256") != review.get("review_sha256")
        or value.get("selection_tool_source_sha256")
        != _file_sha256(Path(__file__).resolve(strict=True))
        or len(value.get("training_source_ids", [])) != 18
        or len(value.get("evaluation_source_ids", [])) != 24
        or len(value.get("unused_accepted_reserve_source_ids", [])) < 8
        or value.get("admission_authorized") is not False
        or value.get("gpu_execution_authorized") is not False
    ):
        raise ValueError("D1 split boundary or digest is invalid")
    if rederive and value != select_d1(review):
        raise ValueError("D1 split does not equal the reference selector output")
    return value


def _named_human(value: str) -> str:
    value = " ".join(_text(value, "reviewer identity").split())
    if len(value.split()) < 2 or not all(
        any(character.isalpha() for character in word) for word in value.split()
    ):
        raise ValueError("reviewer identity must name a human")
    return value


def _canonical_utc(value: str) -> str:
    if not isinstance(value, str) or not _UTC.fullmatch(value):
        raise ValueError("committed_at_utc must be canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError("committed_at_utc is not a real timestamp") from exc
    if parsed > datetime.now(timezone.utc):
        raise ValueError("committed_at_utc cannot be in the future")
    return value


def build_d2_commitment(
    review: dict[str, Any],
    *,
    reviewer_identity: str,
    committed_at_utc: str,
    secret: bytes,
) -> dict[str, Any]:
    review = validate_review(review)
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise ValueError("D2 split secret must contain at least 32 bytes")
    _, policy_sha, _, amendment_sha = _policy()
    body = {
        "schema": 1,
        "kind": _COMMITMENT_KIND,
        "reviewer_identity": _named_human(reviewer_identity),
        "committed_at_utc": _canonical_utc(committed_at_utc),
        "secret_sha256": hashlib.sha256(secret).hexdigest(),
        "selection_policy_sha256": policy_sha,
        "selection_amendment_sha256": amendment_sha,
        "selection_tool_source_sha256": _file_sha256(
            Path(__file__).resolve(strict=True)
        ),
        "executable_review_sha256": review["review_sha256"],
        "decision": "commit_private_d2_split_key_before_selection",
        "secret_disclosed": False,
        "admission_authorized": False,
        "gpu_execution_authorized": False,
    }
    return {**body, "commitment_sha256": krea_provenance.canonical_sha256(body)}


def validate_d2_commitment(
    value: dict[str, Any], review: dict[str, Any], *, secret: bytes | None = None
) -> dict[str, Any]:
    value = _object(value, "D2 key commitment")
    body = {key: item for key, item in value.items() if key != "commitment_sha256"}
    _, policy_sha, _, amendment_sha = _policy()
    if (
        value.get("schema") != 1
        or value.get("kind") != _COMMITMENT_KIND
        or value.get("commitment_sha256") != krea_provenance.canonical_sha256(body)
        or value.get("selection_policy_sha256") != policy_sha
        or value.get("selection_amendment_sha256") != amendment_sha
        or value.get("selection_tool_source_sha256")
        != _file_sha256(Path(__file__).resolve(strict=True))
        or value.get("executable_review_sha256") != review.get("review_sha256")
        or value.get("secret_disclosed") is not False
        or value.get("admission_authorized") is not False
        or value.get("gpu_execution_authorized") is not False
    ):
        raise ValueError("D2 key commitment boundary or digest is invalid")
    _named_human(value.get("reviewer_identity"))
    _canonical_utc(value.get("committed_at_utc"))
    if secret is not None and (
        len(secret) < 32 or hashlib.sha256(secret).hexdigest() != value["secret_sha256"]
    ):
        raise ValueError("D2 split secret does not open the commitment")
    return value


def _hmac_rank(secret: bytes, policy_sha: str, domain: str, source_id: str) -> bytes:
    message = f"{policy_sha}\0{domain}\0{source_id}".encode()
    return hmac.new(secret, message, hashlib.sha256).digest()


def select_d2(
    review: dict[str, Any], commitment: dict[str, Any], *, secret: bytes
) -> dict[str, Any]:
    review = validate_review(review)
    commitment = validate_d2_commitment(commitment, review, secret=secret)
    _, policy_sha, _, amendment_sha = _policy()
    candidates = [
        row
        for row in review["records"]["D2"]
        if row["disposition"] == "CANDIDATE_ONLY_NOT_ADMITTED"
    ]
    protected = (
        "play_component_id",
        "play_root_id",
        "accession_family_id",
        "scene_id",
        "human_similarity_cluster_id",
    )
    ranked = sorted(
        candidates,
        key=lambda row: (
            _hmac_rank(secret, policy_sha, "D2-row", row["source_id"]),
            row["source_id"],
        ),
    )
    used: dict[str, set[str]] = {field: set() for field in protected}
    selected: list[dict[str, Any]] = []
    for row in ranked:
        groups = row["group_identity"]
        if any(groups[field] in used[field] for field in protected):
            continue
        selected.append(row)
        for field in protected:
            used[field].add(groups[field])
        if len(selected) == 76:
            break
    if len(selected) != 76:
        raise ValueError("D2 protected-group selection cannot fill 76 rows")
    role_ranked = sorted(
        selected,
        key=lambda row: (
            _hmac_rank(secret, policy_sha, "D2-role", row["source_id"]),
            row["source_id"],
        ),
    )
    training = sorted(row["source_id"] for row in role_ranked[:36])
    evaluation = sorted(row["source_id"] for row in role_ranked[36:])
    selected_ids = set(training) | set(evaluation)
    reserve = sorted(
        row["source_id"] for row in candidates if row["source_id"] not in selected_ids
    )
    body = {
        "schema": 1,
        "kind": _SPLIT_KIND,
        "experimental_role": "D2",
        "source_review_sha256": review["review_sha256"],
        "selection_policy_sha256": policy_sha,
        "selection_amendment_sha256": amendment_sha,
        "selection_tool_source_sha256": _file_sha256(
            Path(__file__).resolve(strict=True)
        ),
        "d2_key_commitment_sha256": commitment["commitment_sha256"],
        "selection_algorithm": "committed_hmac_protected_identity_v1",
        "training_source_ids": training,
        "evaluation_source_ids": evaluation,
        "unused_accepted_reserve_source_ids": reserve,
        "pending_gates": [
            "tokenizer_checked_rare_trigger_and_training_captions",
            "exhaustive_selected_row_similarity_review_2850_pairs",
            "selected_per_file_rights_record",
            "fixture_manifest_and_archive",
            "response_engineer_countersign",
            "independent_reviewer_approval_and_key_reveal",
            "all_six_fixture_cross_review",
        ],
        "admission_authorized": False,
        "gpu_execution_authorized": False,
        "claim_limit": _CLAIM_LIMIT,
    }
    result = {**body, "split_sha256": krea_provenance.canonical_sha256(body)}
    validate_d2_split(result, review, commitment, secret=secret, rederive=False)
    return result


def validate_d2_split(
    value: dict[str, Any],
    review: dict[str, Any],
    commitment: dict[str, Any],
    *,
    secret: bytes,
    rederive: bool = True,
) -> dict[str, Any]:
    value = _object(value, "D2 split")
    body = {key: item for key, item in value.items() if key != "split_sha256"}
    validate_d2_commitment(commitment, review, secret=secret)
    if (
        value.get("schema") != 1
        or value.get("kind") != _SPLIT_KIND
        or value.get("experimental_role") != "D2"
        or value.get("split_sha256") != krea_provenance.canonical_sha256(body)
        or value.get("source_review_sha256") != review.get("review_sha256")
        or value.get("selection_tool_source_sha256")
        != _file_sha256(Path(__file__).resolve(strict=True))
        or value.get("d2_key_commitment_sha256") != commitment.get("commitment_sha256")
        or len(value.get("training_source_ids", [])) != 36
        or len(value.get("evaluation_source_ids", [])) != 40
        or value.get("admission_authorized") is not False
        or value.get("gpu_execution_authorized") is not False
    ):
        raise ValueError("D2 split boundary or digest is invalid")
    if rederive and value != select_d2(review, commitment, secret=secret):
        raise ValueError("D2 split does not equal the reference selector output")
    return value


def _secret_file(path: Path) -> bytes:
    path = _safe_file(path, "D2 secret")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ValueError("D2 secret must be owner-only (mode 0600 or stricter)")
    secret = path.read_bytes()
    if len(secret) < 32:
        raise ValueError("D2 secret must contain at least 32 bytes")
    return secret


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-review")
    export.add_argument("--workbook", required=True, type=Path)
    export.add_argument("--owner-summary", required=True, type=Path)
    export.add_argument("--d1-inspection", required=True, type=Path)
    export.add_argument("--d2-inspection", required=True, type=Path)
    export.add_argument("--correction-ledger", required=True, type=Path)
    export.add_argument("--corrected-audit", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    validate = commands.add_parser("validate-review")
    validate.add_argument("--review", required=True, type=Path)
    d1 = commands.add_parser("select-d1")
    d1.add_argument("--review", required=True, type=Path)
    d1.add_argument("--output", required=True, type=Path)
    validate_d1 = commands.add_parser("validate-d1")
    validate_d1.add_argument("--review", required=True, type=Path)
    validate_d1.add_argument("--split", required=True, type=Path)
    prepare = commands.add_parser("prepare-d2-key")
    prepare.add_argument("--review", required=True, type=Path)
    prepare.add_argument("--reviewer-identity", required=True)
    prepare.add_argument("--committed-at-utc", required=True)
    prepare.add_argument("--secret-output", required=True, type=Path)
    prepare.add_argument("--commitment-output", required=True, type=Path)
    d2 = commands.add_parser("select-d2")
    d2.add_argument("--review", required=True, type=Path)
    d2.add_argument("--commitment", required=True, type=Path)
    d2.add_argument("--secret", required=True, type=Path)
    d2.add_argument("--output", required=True, type=Path)
    validate_d2 = commands.add_parser("validate-d2")
    validate_d2.add_argument("--review", required=True, type=Path)
    validate_d2.add_argument("--commitment", required=True, type=Path)
    validate_d2.add_argument("--secret", required=True, type=Path)
    validate_d2.add_argument("--split", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    if args.command == "export-review":
        result = export_review(
            workbook=args.workbook,
            owner_summary=args.owner_summary,
            d1_inspection=args.d1_inspection,
            d2_inspection=args.d2_inspection,
            correction_ledger=args.correction_ledger,
            corrected_audit=args.corrected_audit,
        )
        _atomic_create(args.output, result)
    elif args.command == "validate-review":
        result, _ = _read_review(args.review)
    elif args.command == "select-d1":
        review, _ = _read_review(args.review)
        result = select_d1(review)
        _atomic_create(args.output, result)
    elif args.command == "validate-d1":
        review, _ = _read_review(args.review)
        result, _ = _canonical_json(args.split, "D1 split")
        validate_d1_split(result, review)
    elif args.command == "prepare-d2-key":
        review, _ = _read_review(args.review)
        secret_path = Path(os.path.abspath(os.path.expanduser(args.secret_output)))
        if secret_path.exists() or secret_path.is_symlink():
            raise FileExistsError(f"refusing to overwrite: {secret_path}")
        secret_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        secret = secrets.token_bytes(32)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(secret)
                handle.flush()
                os.fsync(handle.fileno())
            result = build_d2_commitment(
                review,
                reviewer_identity=args.reviewer_identity,
                committed_at_utc=args.committed_at_utc,
                secret=secret,
            )
            _atomic_create(args.commitment_output, result)
        except BaseException:
            secret_path.unlink(missing_ok=True)
            raise
    elif args.command == "select-d2":
        review, _ = _read_review(args.review)
        commitment, _ = _canonical_json(args.commitment, "D2 key commitment")
        secret = _secret_file(args.secret)
        result = select_d2(review, commitment, secret=secret)
        _atomic_create(args.output, result)
    elif args.command == "validate-d2":
        review, _ = _read_review(args.review)
        commitment, _ = _canonical_json(args.commitment, "D2 key commitment")
        secret = _secret_file(args.secret)
        result, _ = _canonical_json(args.split, "D2 split")
        validate_d2_split(result, review, commitment, secret=secret)
    else:  # pragma: no cover - argparse enforces this.
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess smoke.
    raise SystemExit(main())
