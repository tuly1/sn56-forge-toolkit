#!/usr/bin/env python3
"""Human-review and admission boundary for Week-7 HKE fixtures.

The procedural renderer can create and machine-check candidates.  It cannot
approve rights or impersonate a human reviewer.  This module keeps that
boundary explicit:

1. a custodian generates a private review template bound to all 98 exact rows;
2. an operator attests that a named human reviewed and filled every check;
3. ``seal-review`` validates that filled draft and publishes a create-only
   private record; and
4. ``admit`` replays the renderer, validates the sealed review, and emits three
   public family receipts.  Those receipts still say GPU execution is false.

The record does not cryptographically authenticate that person, so owner
ratification remains mandatory before any GPU authorization. No command trains,
scores, rents hardware, modifies production, or reveals confirmation membership
in the public admission receipts.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
RENDERER_PATH = SCRIPT_PATH.with_name("hke_procedural_renderer.py")
ADMISSION_SOURCE_PATH = "ops/experiments/week7/hke_fixture_admission.py"
_SPEC = importlib.util.spec_from_file_location("week7_hke_renderer_for_admission", RENDERER_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("procedural renderer cannot be imported")
renderer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = renderer
_SPEC.loader.exec_module(renderer)


SCHEMA = 1
REVIEW_KIND = "sn56-week7-hke-human-fixture-review"
ADMISSION_KIND = "sn56-week7-hke-fixture-admission"
ROOT_ADMISSION_KIND = "sn56-week7-hke-fixture-admission-set"
REQUIRED_CHECKS = {
    "visual_semantics_accepted",
    "caption_semantics_accepted",
    "first_party_rights_and_training_evaluation_use_accepted",
    "no_third_party_brand_mark_person_character_or_artist_reference",
    "no_external_asset_font_provider_reference_or_tournament_content",
    "cross_family_and_cross_phase_similarity_reviewed",
}
DISALLOWED_REVIEWER_LABELS = {
    "agent",
    "assistant",
    "codex",
    "human",
    "human owner",
    "human reviewer",
    "owner",
    "reviewer",
    "team",
}


class AdmissionError(RuntimeError):
    """The human/replay/admission chain is incomplete or misbound."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            separators=(",", ": "),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    raw = renderer._read_regular(Path(path), label)
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"{label} is not JSON") from exc
    if not isinstance(value, dict):
        raise AdmissionError(f"{label} is not an object")
    return value


def _write_new(path: Path, value: Any) -> str:
    path = Path(path)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise AdmissionError(f"output parent is unavailable: {path.parent}")
    payload = canonical_bytes(value)
    try:
        renderer._write_exclusive(path, payload)
    except FileExistsError as exc:
        raise AdmissionError(f"refusing to overwrite {path}") from exc
    return hashlib.sha256(payload).hexdigest()


def _private_record_path(
    path: Path, *, public_root: Path, custodian_root: Path, label: str
) -> Path:
    """Require human-review records to live outside public/candidate trees."""

    path = renderer._absolute(Path(path))
    public_root = renderer._absolute(Path(public_root))
    custodian_root = renderer._absolute(Path(custodian_root))
    if (
        renderer._paths_overlap(path, public_root)
        or renderer._paths_overlap(path, custodian_root)
        or renderer._paths_overlap(path, REPO_ROOT)
    ):
        raise AdmissionError(
            f"{label} must be outside public, candidate-custodian, and repository trees"
        )
    renderer._require_no_symlink_components(path.parent, f"{label} parent")
    if not path.parent.is_dir():
        raise AdmissionError(f"{label} parent is unavailable")
    return path


def _rows(verified: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for phase, manifest_name in (
        ("discovery", "discovery_manifest"),
        ("confirmation", "confirmation_manifest"),
    ):
        manifest = verified[manifest_name]
        for fixture_id, family in sorted(manifest["families"].items()):
            for row in family["rows"]:
                result.append(
                    {
                        "fixture_id": fixture_id,
                        "family": row["family"],
                        "phase": phase,
                        "row_id": row["row_id"],
                        "image_sha256": row["image_sha256"],
                        "caption_sha256": row["caption_sha256"],
                        "decoded_pixels_sha256": row["decoded_pixels_sha256"],
                        "parameters_sha256": row["parameters_sha256"],
                        "group_identity_sha256": row["group_identity_sha256"],
                        "row_record_sha256": row["row_record_sha256"],
                    }
                )
    return result


def build_review_template(public_root: Path, custodian_root: Path) -> dict[str, Any]:
    """Build a private, exact-row checklist; this is not a review."""

    verified = renderer.verify_candidate(Path(public_root), Path(custodian_root))
    candidate = verified["candidate_manifest"]
    body = {
        "schema": SCHEMA,
        "kind": REVIEW_KIND,
        "status": "PENDING_NAMED_HUMAN_REVIEW",
        "candidate_semantic_sha256": candidate["semantic_sha256"],
        "rights_record": candidate["rights_record"],
        "confirmation_manifest_file_sha256": candidate["confirmation"][
            "custodian_manifest_sha256"
        ],
        "reviewer_identity": "",
        "reviewed_at_utc": "",
        "rows": [
            {
                **row,
                "checks": {key: False for key in sorted(REQUIRED_CHECKS)},
                "notes": "",
            }
            for row in _rows(verified)
        ],
        "decision": "PENDING",
        "governance": {
            "agent_review_is_not_human_review": True,
            "review_must_cover_all_98_rows": True,
            "confirmation_record_is_private": True,
        },
    }
    return body


def _named_human(value: Any) -> str:
    text = " ".join(str(value or "").split())
    lowered_words = {word.casefold() for word in re.findall(r"[A-Za-z]+", text)}
    role_words = {"agent", "assistant", "codex", "human", "owner", "reviewer", "team"}
    if text.casefold() in DISALLOWED_REVIEWER_LABELS or lowered_words & role_words:
        raise AdmissionError("reviewer_identity is a role label, not a named human")
    words = re.findall(r"[A-Za-z][A-Za-z'.-]+", text)
    if len(words) < 2:
        raise AdmissionError("reviewer_identity must name a human")
    return text


def _utc(value: Any) -> str:
    text = str(value or "")
    if not text.endswith("Z"):
        raise AdmissionError("reviewed_at_utc must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise AdmissionError("reviewed_at_utc is invalid") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise AdmissionError("reviewed_at_utc is not UTC")
    return text


def _expected_review_rows(verified: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _rows(verified)


def validate_review_body(
    value: Mapping[str, Any], verified: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a filled human draft before it is sealed."""

    candidate = verified["candidate_manifest"]
    if (
        value.get("schema") != SCHEMA
        or value.get("kind") != REVIEW_KIND
        or value.get("candidate_semantic_sha256") != candidate["semantic_sha256"]
        or value.get("rights_record") != candidate["rights_record"]
        or value.get("confirmation_manifest_file_sha256")
        != candidate["confirmation"]["custodian_manifest_sha256"]
        or value.get("decision") != "PASS"
    ):
        raise AdmissionError("human review identity, binding, or decision is invalid")
    reviewer = _named_human(value.get("reviewer_identity"))
    reviewed_at = _utc(value.get("reviewed_at_utc"))
    rows = value.get("rows")
    expected = _expected_review_rows(verified)
    if not isinstance(rows, list) or len(rows) != len(expected) or len(rows) != 98:
        raise AdmissionError("human review must cover exactly all 98 candidate rows")
    for index, (actual, identity) in enumerate(zip(rows, expected, strict=True)):
        if not isinstance(actual, dict):
            raise AdmissionError(f"human review row {index} is not an object")
        projected = {key: actual.get(key) for key in identity}
        if projected != identity:
            raise AdmissionError(f"human review row {index} identity mismatch")
        checks = actual.get("checks")
        if not isinstance(checks, dict) or set(checks) != REQUIRED_CHECKS:
            raise AdmissionError(f"human review row {index} checks are incomplete")
        if any(value is not True for value in checks.values()):
            raise AdmissionError(f"human review row {index} has an unapproved check")
        if not isinstance(actual.get("notes", ""), str):
            raise AdmissionError(f"human review row {index} notes are invalid")
    governance = value.get("governance")
    if governance != {
        "agent_review_is_not_human_review": True,
        "review_must_cover_all_98_rows": True,
        "confirmation_record_is_private": True,
    }:
        raise AdmissionError("human review governance was changed")
    return {
        **dict(value),
        "status": "SEALED_OPERATOR_ATTESTED_NAMED_HUMAN_PASS",
        "reviewer_identity": reviewer,
        "reviewed_at_utc": reviewed_at,
    }


def seal_review(
    *, public_root: Path, custodian_root: Path, draft: Mapping[str, Any]
) -> dict[str, Any]:
    verified = renderer.verify_candidate(Path(public_root), Path(custodian_root))
    body = validate_review_body(draft, verified)
    body.pop("review_sha256", None)
    return {**body, "review_sha256": semantic_sha256(body)}


def validate_sealed_review(
    value: Mapping[str, Any], verified: Mapping[str, Any]
) -> dict[str, Any]:
    if value.get("status") != "SEALED_OPERATOR_ATTESTED_NAMED_HUMAN_PASS":
        raise AdmissionError("sealed human review has not passed")
    declared = value.get("review_sha256")
    body = dict(value)
    body.pop("review_sha256", None)
    if declared != semantic_sha256(body):
        raise AdmissionError("sealed human review digest mismatch")
    return validate_review_body(body, verified)


def _family_counts(candidate: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for value in candidate["families"].values():
        result[value["family"]] = {
            "discovery": value["discovery"]["row_count"],
            "confirmation": value["confirmation"]["row_count"],
        }
    return result


def _current_generator_identity() -> dict[str, Any]:
    """Bind admission to the committed and pushed renderer bytes it executes."""

    git = Path("/usr/bin/git")
    if not git.is_file():
        raise AdmissionError("fixed /usr/bin/git is unavailable")
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent-sn56-admission",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
    }

    def run(arguments: Sequence[str], *, binary: bool = False):
        completed = subprocess.run(
            [
                str(git),
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.hooksPath=/dev/null",
                *arguments,
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=not binary,
        )
        if completed.returncode != 0:
            raise AdmissionError(
                f"generator Git identity command failed: {' '.join(arguments)}"
            )
        return completed.stdout

    commit = str(run(("rev-parse", "--verify", "HEAD^{commit}"))).strip().lower()
    tree = str(run(("rev-parse", "--verify", "HEAD^{tree}"))).strip().lower()
    renderer_blob = run(
        ("cat-file", "blob", f"HEAD:{renderer.GENERATOR_SOURCE_PATH}"),
        binary=True,
    )
    if not isinstance(renderer_blob, bytes):  # pragma: no cover
        raise AdmissionError("committed renderer bytes are unavailable")
    source_sha = hashlib.sha256(renderer_blob).hexdigest()
    if source_sha != hashlib.sha256(RENDERER_PATH.read_bytes()).hexdigest():
        raise AdmissionError("executed renderer bytes differ from the committed blob")
    authority_blob = run(
        ("cat-file", "blob", f"HEAD:{ADMISSION_SOURCE_PATH}"),
        binary=True,
    )
    if not isinstance(authority_blob, bytes):  # pragma: no cover
        raise AdmissionError("committed admission-authority bytes are unavailable")
    authority_source_sha = hashlib.sha256(authority_blob).hexdigest()
    if authority_source_sha != hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest():
        raise AdmissionError("executed admission authority differs from committed blob")

    remote = subprocess.run(
        [
            str(git),
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "ls-remote",
            "--heads",
            renderer.GENERATOR_REPOSITORY,
        ],
        cwd="/",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if remote.returncode != 0:
        raise AdmissionError("cannot prove generator commit on the pinned remote")
    remote_refs = sorted(
        line.split("\t", 1)[1]
        for line in remote.stdout.splitlines()
        if "\t" in line and line.split("\t", 1)[0].lower() == commit
    )
    if not remote_refs:
        raise AdmissionError("generator commit is not present on the pinned remote")
    return {
        "repository": renderer.GENERATOR_REPOSITORY,
        "commit": commit,
        "tree": tree,
        "source_path": renderer.GENERATOR_SOURCE_PATH,
        "renderer_source_sha256": source_sha,
        "admission_authority_source_sha256": authority_source_sha,
        "pinned_remote_refs": remote_refs,
    }


def build_admissions(
    *,
    public_root: Path,
    custodian_root: Path,
    discovery_key: bytes,
    confirmation_key: bytes,
    sealed_review: Mapping[str, Any],
    generator_identity_probe: Callable[[], Mapping[str, Any]] = _current_generator_identity,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    verified = renderer.verify_candidate(Path(public_root), Path(custodian_root))
    replay = renderer.verify_replay(
        public_output=Path(public_root),
        custodian_output=Path(custodian_root),
        discovery_key=discovery_key,
        confirmation_key=confirmation_key,
    )
    review = validate_sealed_review(sealed_review, verified)
    candidate = verified["candidate_manifest"]
    live_generator = dict(generator_identity_probe())
    candidate_generator = candidate["generator"]
    for key in ("repository", "commit", "tree", "source_path", "renderer_source_sha256"):
        if live_generator.get(key) != candidate_generator.get(key):
            raise AdmissionError(f"candidate generator {key} is not the executed pushed revision")
    if not live_generator.get("pinned_remote_refs"):
        raise AdmissionError("candidate generator has no pinned-remote evidence")
    authority_source_sha = live_generator.get("admission_authority_source_sha256")
    if not isinstance(authority_source_sha, str) or len(authority_source_sha) != 64:
        raise AdmissionError("admission authority source identity is absent")
    dedup = verified["dedup_evidence"]
    if any(
        dedup[key] != 0
        for key in (
            "exact_image_duplicate_count",
            "decoded_pixel_duplicate_count",
            "normalized_caption_duplicate_count",
            "group_identity_duplicate_count",
            "perceptual_near_duplicate_count",
        )
    ):
        raise AdmissionError("fixture duplicate screen is not clean")
    replay_evidence = {
        "status": replay["status"],
        "verified_rows": replay["verified_rows"],
        "candidate_semantic_sha256": replay["candidate_semantic_sha256"],
        "dedup_semantic_sha256": replay["dedup_semantic_sha256"],
    }
    replay_sha = semantic_sha256(replay_evidence)
    counts = _family_counts(candidate)
    receipts: dict[str, dict[str, Any]] = {}
    by_family = {
        value["family"]: value for value in candidate["families"].values()
    }
    for family in ("social", "product", "logo_ui"):
        discovery_identity = [
            {
                "row_id": row["row_id"],
                "row_sha256": row["row_record_sha256"],
            }
            for row in by_family[family]["discovery"]["rows"]
        ]
        body = {
            "schema": SCHEMA,
            "kind": ADMISSION_KIND,
            "status": "PASS",
            "family": family,
            "counts": counts[family],
            "candidate_semantic_sha256": candidate["semantic_sha256"],
            "human_review_sha256": sealed_review["review_sha256"],
            "replay_evidence_sha256": replay_sha,
            "dedup_evidence_sha256": dedup["semantic_sha256"],
            "ownership_record_sha256": candidate["rights_record"][
                "semantic_sha256"
            ],
            "generator_commit": live_generator["commit"],
            "generator_tree": live_generator["tree"],
            "generator_source_sha256": live_generator["renderer_source_sha256"],
            "admission_authority_source_sha256": authority_source_sha,
            "confirmation_commitment_sha256": by_family[family]["confirmation"][
                "semantic_commitment_sha256"
            ],
            "discovery_row_identity_sha256": semantic_sha256(
                discovery_identity
            ),
            "governance": {
                "operator_attested_named_human_review": True,
                "agent_review_is_not_human_review": True,
                "admission_authorized": True,
                "gpu_execution_authorized": False,
                "owner_ratification_required_for_gpu": True,
            },
            "claim_limit": (
                "admitted for this procedural instrument only; field and "
                "opponent-relative transfer remain unproven"
            ),
        }
        receipts[family] = {**body, "admission_sha256": semantic_sha256(body)}
    root_body = {
        "schema": SCHEMA,
        "kind": ROOT_ADMISSION_KIND,
        "status": "PASS",
        "candidate_semantic_sha256": candidate["semantic_sha256"],
        "human_review_sha256": sealed_review["review_sha256"],
        "replay_evidence": replay_evidence,
        "replay_evidence_sha256": replay_sha,
        "dedup_evidence_sha256": dedup["semantic_sha256"],
        "ownership_record_sha256": candidate["rights_record"]["semantic_sha256"],
        "generator_revision": {
            "repository": live_generator["repository"],
            "commit": live_generator["commit"],
            "tree": live_generator["tree"],
            "renderer_source_sha256": live_generator["renderer_source_sha256"],
            "admission_authority_source_sha256": authority_source_sha,
            "pinned_remote_refs": live_generator["pinned_remote_refs"],
        },
        "family_admission_sha256": {
            family: receipt["admission_sha256"] for family, receipt in receipts.items()
        },
        "receipts": receipts,
        "authorization": {
            "fixture_admission_authorized": True,
            "gpu_execution_authorized": False,
            "deployment_authorized": False,
        },
    }
    root = {**root_body, "admission_set_sha256": semantic_sha256(root_body)}
    # Keep the named reviewer and private row identities out of all public receipts.
    public_payload = canonical_bytes(root) + b"".join(
        canonical_bytes(receipt) for receipt in receipts.values()
    )
    reviewer_token = json.dumps(
        review["reviewer_identity"], ensure_ascii=True
    )[1:-1].encode("ascii")
    if reviewer_token in public_payload:
        raise AdmissionError("public admission leaks the private reviewer identity")
    for family in verified["confirmation_manifest"]["families"].values():
        for row in family["rows"]:
            for key in (
                "row_id",
                "relative_image_path",
                "relative_caption_path",
                "image_sha256",
                "caption_sha256",
                "decoded_pixels_sha256",
                "parameters_sha256",
                "group_identity_sha256",
                "row_record_sha256",
            ):
                if str(row[key]).encode("ascii") in public_payload:
                    raise AdmissionError("public admission leaks confirmation membership")
    return root, receipts


def _read_key(path: Path, label: str) -> bytes:
    return renderer._read_key_file(Path(path), label)


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("review-template")
    seal = sub.add_parser("seal-review")
    admit = sub.add_parser("admit")
    for item in (template, seal, admit):
        item.add_argument("--public-root", type=Path, required=True)
        item.add_argument("--custodian-root", type=Path, required=True)
    template.add_argument("--output", type=Path, required=True)
    seal.add_argument("--draft", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    admit.add_argument("--sealed-review", type=Path, required=True)
    admit.add_argument("--discovery-key-file", type=Path, required=True)
    admit.add_argument("--confirmation-key-file", type=Path, required=True)
    admit.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    if args.command == "review-template":
        output = _private_record_path(
            args.output,
            public_root=args.public_root,
            custodian_root=args.custodian_root,
            label="human review template",
        )
        value = build_review_template(args.public_root, args.custodian_root)
        _write_new(output, value)
        return 0
    if args.command == "seal-review":
        draft_path = _private_record_path(
            args.draft,
            public_root=args.public_root,
            custodian_root=args.custodian_root,
            label="human review draft",
        )
        output = _private_record_path(
            args.output,
            public_root=args.public_root,
            custodian_root=args.custodian_root,
            label="sealed human review",
        )
        draft = _load_json(draft_path, "human review draft")
        value = seal_review(
            public_root=args.public_root,
            custodian_root=args.custodian_root,
            draft=draft,
        )
        _write_new(output, value)
        return 0
    sealed_path = _private_record_path(
        args.sealed_review,
        public_root=args.public_root,
        custodian_root=args.custodian_root,
        label="sealed human review",
    )
    sealed = _load_json(sealed_path, "sealed human review")
    root, receipts = build_admissions(
        public_root=args.public_root,
        custodian_root=args.custodian_root,
        discovery_key=_read_key(args.discovery_key_file, "discovery key"),
        confirmation_key=_read_key(args.confirmation_key_file, "confirmation key"),
        sealed_review=sealed,
    )
    if renderer._paths_overlap(args.output_root, args.public_root) or renderer._paths_overlap(
        args.output_root, args.custodian_root
    ):
        raise AdmissionError("admission output must be disjoint from candidate trees")
    output_root = renderer._ensure_new_root(args.output_root, "admission output")
    _write_new(output_root / "ADMISSION-SET.json", root)
    for family, receipt in receipts.items():
        _write_new(output_root / f"{family}.json", receipt)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
