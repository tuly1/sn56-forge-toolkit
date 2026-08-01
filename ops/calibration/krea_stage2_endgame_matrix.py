#!/usr/bin/env python3
"""Strict Week-5 Krea confirmation/boundary matrix and row launcher.

This adapter closes one narrow operational gap: the Stage-2 primitives can
validate and run one cell, but they do not derive the exact post-freeze matrix
or bind a row to its fixed GPU.  It does not reveal fixtures, create governance
authority, score candidates, select a release, or provide a waiver path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence

try:
    from . import krea_fixture
    from . import krea_density_seedb_freeze
    from . import krea_provenance
    from . import krea_stage2_execution
    from . import krea_stage2_production_identity
    from . import krea_stage2_training_evidence
except ImportError:  # pragma: no cover - direct CLI execution.
    import krea_fixture  # type: ignore[no-redef]
    import krea_density_seedb_freeze  # type: ignore[no-redef]
    import krea_provenance  # type: ignore[no-redef]
    import krea_stage2_execution  # type: ignore[no-redef]
    import krea_stage2_production_identity  # type: ignore[no-redef]
    import krea_stage2_training_evidence  # type: ignore[no-redef]


SCHEMA = 1
MATRIX_KIND = "forge-krea-stage2-endgame-matrix"
RECEIPT_KIND = "forge-krea-stage2-endgame-row-receipt"
SCORE_HOOK_KIND = "forge-krea-stage2-score-stream-hook"

# The fresh no-cache build/probe gate sealed this exact c000 subject.  The
# adapter intentionally refuses another commit, tree, or image identity.
SOURCE_COMMIT = "c0001556715ef2004ba70d6b5dc2fda55d26860b"
SOURCE_TREE = "bffc6414fc4f17a1708cfa6724a499a00235837d"
PRODUCTION_IMAGE_ID = (
    "sha256:f2df4df111192025c5977df600e8e91cbfd019b95858266a4a18b6d940892212"
)

FAMILY_ORDER = ("K0", "K1", "K2", "K3", "K4", "K5")
REFERENCE_FAMILIES = ("K0", "K2", "K3", "K4")
CONFIRMATION_FIXTURES = ("C1", "C2", "C3", "C4")
SEED_ROLES = ("A", "B")
BOUNDARY_CELLS = (
    "B-0p5-small",
    "B-0p5-large",
    "B-0p75-small",
    "B-0p75-large",
    "B-1-small",
    "B-1-large",
)
FIXTURE_GPU = {"C1": 0, "C2": 1, "C3": 2, "C4": 3}
BOUNDARY_GPU = {
    "B-0p5-small": 0,
    "B-0p5-large": 1,
    "B-0p75-small": 2,
    "B-0p75-large": 3,
    "B-1-small": 0,
    "B-1-large": 1,
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ROW_KEY = re.compile(r"(?:confirmation|boundary)-[A-Za-z0-9_.-]+")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(
            f"{label} keys differ: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_file_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(krea_provenance.canonical_bytes(value) + b"\n").hexdigest()


def _safe_path(path: str | Path, label: str, *, must_exist: bool) -> Path:
    value = Path(os.path.abspath(os.path.expanduser(str(path))))
    current = value
    while True:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink component")
        if current == current.parent:
            break
        current = current.parent
    if must_exist:
        try:
            mode = value.stat().st_mode
        except OSError as exc:
            raise ValueError(f"{label} is unavailable") from exc
        if not stat.S_ISREG(mode):
            raise ValueError(f"{label} must be a regular file")
    return value


def _load_canonical(path: str | Path, label: str) -> dict[str, Any]:
    source = _safe_path(path, label, must_exist=True)
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    value = _object(value, label)
    if raw != krea_provenance.canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} is not canonical JSON plus one newline")
    return value


def _publish_new(path: str | Path, value: Mapping[str, Any]) -> None:
    target = _safe_path(path, "create-only output", must_exist=False)
    if os.path.lexists(target):
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = krea_provenance.canonical_bytes(value) + b"\n"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _publish_or_replay(
    path: str | Path, value: Mapping[str, Any], label: str
) -> dict[str, Any]:
    """Publish once, or require an interrupted prior publication to match."""

    target = _safe_path(path, label, must_exist=os.path.lexists(path))
    if os.path.lexists(target):
        observed = _load_canonical(target, label)
        if observed != value:
            raise ValueError(f"{label} differs from the deterministic replay")
        return observed
    _publish_new(target, value)
    return dict(value)


def _validate_production_identity(value: Any) -> dict[str, Any]:
    identity = krea_stage2_production_identity.validate(value)
    forge = _object(identity["forge"], "production identity forge")
    image = _object(identity["container_image"], "production identity image")
    if (
        forge["commit_sha1"] != SOURCE_COMMIT
        or forge["tree_sha1"] != SOURCE_TREE
        or forge["worktree_state"] != "clean-including-untracked"
        or image["image_id"] != PRODUCTION_IMAGE_ID
    ):
        raise ValueError("production identity is not the fresh c000 image binding")
    return identity


def _active_variants(freeze: Mapping[str, Any]) -> list[str]:
    """Derive policies only from D1/D2 winners; reserve fields cannot expand it."""

    variants: list[str] = []
    for key in ("D1_winner_family_id", "D2_winner_family_id"):
        family = freeze.get(key)
        if family not in FAMILY_ORDER or family == "K0":
            raise ValueError(f"{key} is not an executable non-control family")
        if family not in variants:
            variants.append(family)
    return variants


def _family_role(family: str, active: Sequence[str]) -> str:
    if family in active:
        return "candidate"
    if family == "K0":
        return "control"
    return "public_reference"


def _row_key(phase: str, cell: str, family: str) -> str:
    value = f"{phase}-{cell}-{family}"
    if _ROW_KEY.fullmatch(value) is None:
        raise ValueError("matrix row key is unsafe")
    return value


def _matrix_body(
    *,
    freeze: Mapping[str, Any],
    freeze_file_sha256: str,
    production_identity: Mapping[str, Any],
    production_identity_file_sha256: str,
    created_at_utc: str,
) -> dict[str, Any]:
    active = _active_variants(freeze)
    family_set = set(REFERENCE_FAMILIES) | set(active)
    universe = [family for family in FAMILY_ORDER if family in family_set]
    initial = set(REFERENCE_FAMILIES) | {active[0]}
    confirmation_wave_for_family = {
        family: (
            f"confirmation-policy-1-{active[0]}"
            if family in initial
            else f"confirmation-policy-{active.index(family) + 1}-{family}"
        )
        for family in universe
    }
    rows: list[dict[str, Any]] = []
    for fixture in CONFIRMATION_FIXTURES:
        for seed_role in SEED_ROLES:
            cell = f"{fixture}-{seed_role}"
            for family in universe:
                rows.append(
                    {
                        "row_key": _row_key("confirmation", cell, family),
                        "phase": "confirmation",
                        "cell_id": cell,
                        "fixture_id": fixture,
                        "seed_role": seed_role,
                        "family_id": family,
                        "family_role": _family_role(family, active),
                        "candidate_policy_family_ids": (
                            list(active) if family in REFERENCE_FAMILIES else [family]
                        ),
                        "gpu_device": FIXTURE_GPU[fixture],
                        "wave_id": confirmation_wave_for_family[family],
                    }
                )
    for policy_index, family in enumerate(active, start=1):
        for cell in BOUNDARY_CELLS:
            rows.append(
                {
                    "row_key": _row_key("boundary", cell, family),
                    "phase": "boundary",
                    "cell_id": cell,
                    "fixture_id": cell,
                    "seed_role": "A",
                    "family_id": family,
                    "family_role": "candidate",
                    "candidate_policy_family_ids": [family],
                    "gpu_device": BOUNDARY_GPU[cell],
                    "wave_id": f"boundary-policy-{policy_index}-{family}",
                }
            )
    wave_ids: list[str] = []
    for row in rows:
        if row["wave_id"] not in wave_ids:
            wave_ids.append(row["wave_id"])
    rows = [row for wave_id in wave_ids for row in rows if row["wave_id"] == wave_id]
    waves = [
        {
            "wave_id": wave_id,
            "row_keys": [row["row_key"] for row in rows if row["wave_id"] == wave_id],
        }
        for wave_id in wave_ids
    ]
    created = krea_stage2_execution._utc(created_at_utc, "matrix creation time")
    return {
        "schema": SCHEMA,
        "kind": MATRIX_KIND,
        "created_at_utc": created,
        "freeze": {
            "file_sha256": _sha(freeze_file_sha256, "freeze file SHA-256"),
            "freeze_sha256": _sha(freeze["freeze_sha256"], "freeze semantic SHA-256"),
        },
        "production_identity": {
            "file_sha256": _sha(
                production_identity_file_sha256,
                "production identity file SHA-256",
            ),
            "production_identity_sha256": _sha(
                production_identity["production_identity_sha256"],
                "production identity semantic SHA-256",
            ),
        },
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "production_image_id": PRODUCTION_IMAGE_ID,
        "active_variant_family_ids": active,
        "family_execution_universe": universe,
        "confirmation_training_count": 8 * len(universe),
        "boundary_training_count": 6 * len(active),
        "training_count": len(rows),
        "fixture_gpu_mapping": dict(FIXTURE_GPU),
        "boundary_gpu_mapping": dict(BOUNDARY_GPU),
        "waves": waves,
        "rows": rows,
        "strict_admission_per_row": True,
        "waiver_path_available": False,
        "release_authorized": False,
        "production_mutation_authorized": False,
    }


def build_matrix(
    *,
    freeze_path: str | Path,
    production_identity_path: str | Path,
    created_at_utc: str,
) -> dict[str, Any]:
    """Replay the frozen science and seal the exact executable row universe."""

    freeze_file = _safe_path(freeze_path, "finalist freeze", must_exist=True)
    freeze = krea_density_seedb_freeze.validate_freeze(freeze_file)
    identity_file = _safe_path(
        production_identity_path, "production identity", must_exist=True
    )
    identity = _validate_production_identity(
        krea_stage2_production_identity.load(identity_file)
    )
    body = _matrix_body(
        freeze=freeze,
        freeze_file_sha256=_canonical_file_sha(freeze),
        production_identity=identity,
        production_identity_file_sha256=_canonical_file_sha(identity),
        created_at_utc=created_at_utc,
    )
    return validate_matrix(
        {**body, "matrix_sha256": krea_provenance.canonical_sha256(body)},
        freeze=freeze,
        production_identity=identity,
    )


def validate_matrix(
    value: Any,
    *,
    freeze: Mapping[str, Any] | None = None,
    production_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    matrix = _object(value, "Stage-2 endgame matrix")
    keys = {
        "schema",
        "kind",
        "created_at_utc",
        "freeze",
        "production_identity",
        "source_commit",
        "source_tree",
        "production_image_id",
        "active_variant_family_ids",
        "family_execution_universe",
        "confirmation_training_count",
        "boundary_training_count",
        "training_count",
        "fixture_gpu_mapping",
        "boundary_gpu_mapping",
        "waves",
        "rows",
        "strict_admission_per_row",
        "waiver_path_available",
        "release_authorized",
        "production_mutation_authorized",
        "matrix_sha256",
    }
    _exact(matrix, keys, "Stage-2 endgame matrix")
    body = {key: item for key, item in matrix.items() if key != "matrix_sha256"}
    if (
        matrix["schema"] != SCHEMA
        or matrix["kind"] != MATRIX_KIND
        or matrix["matrix_sha256"] != krea_provenance.canonical_sha256(body)
        or matrix["source_commit"] != SOURCE_COMMIT
        or matrix["source_tree"] != SOURCE_TREE
        or matrix["production_image_id"] != PRODUCTION_IMAGE_ID
        or matrix["strict_admission_per_row"] is not True
        or matrix["waiver_path_available"] is not False
        or matrix["release_authorized"] is not False
        or matrix["production_mutation_authorized"] is not False
    ):
        raise ValueError("Stage-2 endgame matrix identity or authority drifted")
    krea_stage2_execution._utc(matrix["created_at_utc"], "matrix creation time")
    freeze_binding = _object(matrix["freeze"], "matrix freeze binding")
    identity_binding = _object(
        matrix["production_identity"], "matrix production identity binding"
    )
    _exact(freeze_binding, {"file_sha256", "freeze_sha256"}, "freeze binding")
    _exact(
        identity_binding,
        {"file_sha256", "production_identity_sha256"},
        "production identity binding",
    )
    for item in freeze_binding.values():
        _sha(item, "freeze binding digest")
    for item in identity_binding.values():
        _sha(item, "production identity binding digest")
    if freeze is not None and production_identity is not None:
        identity = _validate_production_identity(production_identity)
        expected = _matrix_body(
            freeze=freeze,
            freeze_file_sha256=_canonical_file_sha(freeze),
            production_identity=identity,
            production_identity_file_sha256=_canonical_file_sha(identity),
            created_at_utc=matrix["created_at_utc"],
        )
        if body != expected:
            raise ValueError("Stage-2 endgame matrix does not recompute")
    rows = matrix["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Stage-2 endgame matrix rows are empty")
    row_keys = [row.get("row_key") for row in rows if isinstance(row, dict)]
    if len(row_keys) != len(rows) or len(row_keys) != len(set(row_keys)):
        raise ValueError("Stage-2 endgame matrix row keys are invalid or duplicated")
    confirmation = [row for row in rows if row.get("phase") == "confirmation"]
    boundary = [row for row in rows if row.get("phase") == "boundary"]
    if (
        len(confirmation) != matrix["confirmation_training_count"]
        or len(boundary) != matrix["boundary_training_count"]
        or len(rows) != matrix["training_count"]
    ):
        raise ValueError("Stage-2 endgame matrix counts differ from its rows")
    wave_rows = []
    if not isinstance(matrix["waves"], list):
        raise ValueError("Stage-2 endgame waves must be an array")
    for wave in matrix["waves"]:
        wave = _object(wave, "Stage-2 wave")
        _exact(wave, {"wave_id", "row_keys"}, "Stage-2 wave")
        if not isinstance(wave["row_keys"], list) or not wave["row_keys"]:
            raise ValueError("Stage-2 wave is empty")
        wave_rows.extend(wave["row_keys"])
    if wave_rows != row_keys:
        raise ValueError("Stage-2 waves do not preserve the exact row order")
    return dict(matrix)


def publish_matrix(value: Any, output: str | Path) -> dict[str, Any]:
    matrix = validate_matrix(value)
    _publish_new(output, matrix)
    return matrix


def _matrix_row(matrix: Mapping[str, Any], row_key: str) -> dict[str, Any]:
    matches = [row for row in matrix["rows"] if row["row_key"] == row_key]
    if len(matches) != 1:
        raise ValueError("row key is absent or duplicated in the matrix")
    return matches[0]


def validate_row_controls(
    *,
    matrix: Mapping[str, Any],
    row_key: str,
    plan: Mapping[str, Any],
    approval: Mapping[str, Any],
    authority_bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    controls = _object(authority_bundle, "Stage-2 authority bundle")
    identity = _validate_production_identity(controls.get("production_identity"))
    freeze = _object(
        controls.get("waiver_finalist_freeze"), "authority finalist freeze"
    )
    resolved_matrix = validate_matrix(
        matrix, freeze=freeze, production_identity=identity
    )
    row = _matrix_row(resolved_matrix, row_key)
    if (
        controls.get("production_identity_file_sha256")
        != resolved_matrix["production_identity"]["file_sha256"]
        or identity["production_identity_sha256"]
        != resolved_matrix["production_identity"]["production_identity_sha256"]
    ):
        raise ValueError("row authority uses a different production identity")
    resolved_plan = krea_stage2_execution.validate_plan_with_authority(
        plan, authority_controls=controls
    )
    resolved_approval = krea_stage2_execution.validate_approval(
        approval, plan=resolved_plan
    )
    selected = [
        candidate
        for candidate in resolved_plan["candidate_universe"]
        if candidate["candidate_id"] == resolved_plan["training_candidate_id"]
    ]
    expected = {
        "phase": row["phase"],
        "cell_id": row["cell_id"],
        "fixture_id": row["fixture_id"],
        "seed_role": row["seed_role"],
        "family_role": row["family_role"],
        "production_image_id": PRODUCTION_IMAGE_ID,
    }
    if (
        any(resolved_plan[key] != value for key, value in expected.items())
        or len(selected) != 1
        or selected[0]["family_id"] != row["family_id"]
        or resolved_plan["calibration_profile"] != row["family_id"]
        or resolved_plan["waiver_finalist_freeze"] != resolved_matrix["freeze"]
        or resolved_plan["production_identity"]
        != resolved_matrix["production_identity"]
    ):
        raise ValueError("execution plan differs from its exact matrix row")
    return resolved_plan, resolved_approval, identity


def _score_hook(
    *, matrix: Mapping[str, Any], row: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    body = {
        "schema": SCHEMA,
        "kind": SCORE_HOOK_KIND,
        "matrix_sha256": matrix["matrix_sha256"],
        "row_key": row["row_key"],
        "phase": row["phase"],
        "score_required": row["phase"] == "confirmation",
        "run_evidence_sha256": evidence["evidence_sha256"],
        "candidate_artifacts": evidence["candidate_artifacts"],
        "state": "ready-for-exact-score",
        "release_authorized": False,
        "production_mutation_authorized": False,
    }
    return {**body, "hook_sha256": krea_provenance.canonical_sha256(body)}


def validate_score_hook(
    value: Any,
    *,
    matrix: Mapping[str, Any],
    row: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    hook = _object(value, "Stage-2 score-stream hook")
    expected = _score_hook(matrix=matrix, row=row, evidence=evidence)
    if hook != expected:
        raise ValueError("Stage-2 score-stream hook does not replay exactly")
    return dict(hook)


def _receipt(
    *,
    matrix: Mapping[str, Any],
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    approval: Mapping[str, Any],
    completion: Mapping[str, Any],
    evidence: Mapping[str, Any],
    score_hook: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": SCHEMA,
        "kind": RECEIPT_KIND,
        "matrix_sha256": matrix["matrix_sha256"],
        "row_key": row["row_key"],
        "gpu_device": row["gpu_device"],
        "execution_plan_sha256": plan["plan_sha256"],
        "execution_approval_sha256": approval["approval_sha256"],
        "completion_sha256": completion["completion_sha256"],
        "run_evidence_sha256": evidence["evidence_sha256"],
        "score_hook_sha256": score_hook["hook_sha256"],
        "production_image_id": PRODUCTION_IMAGE_ID,
        "strict_authority_replayed": True,
        "waiver_used": False,
        "release_authorized": False,
        "production_mutation_authorized": False,
    }
    return {**body, "receipt_sha256": krea_provenance.canonical_sha256(body)}


def validate_receipt(
    value: Any,
    *,
    matrix: Mapping[str, Any],
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    approval: Mapping[str, Any],
    completion: Mapping[str, Any],
    evidence: Mapping[str, Any],
    score_hook: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _object(value, "Stage-2 row receipt")
    expected = _receipt(
        matrix=matrix,
        row=row,
        plan=plan,
        approval=approval,
        completion=completion,
        evidence=evidence,
        score_hook=score_hook,
    )
    if receipt != expected:
        raise ValueError("Stage-2 row receipt does not replay exactly")
    return dict(receipt)


def _replay_live_run(
    *,
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    approval: Mapping[str, Any],
    completion: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Replay private receipts and rehash every artifact from its live path."""

    if completion["gpu_device"] != row["gpu_device"]:
        raise ValueError("completion GPU differs from the matrix row")
    private = krea_stage2_execution.validate_private_run_receipts(plan)
    if (
        private["config_control"] != completion["config_control_receipt"]
        or private["training_terminal"] != completion["training_terminal_receipt"]
        or private["checkpoint_selection"] != completion["checkpoint_selection_receipt"]
    ):
        raise ValueError("private run receipts differ from the completion")
    fixture_binding = _object(plan["fixture_manifest"], "fixture binding")
    fixture_path = _safe_path(
        fixture_binding["path"], "bound fixture manifest", must_exist=True
    )
    fixture = krea_fixture.validate_manifest(
        _load_canonical(fixture_path, "bound fixture manifest")
    )
    if (
        _canonical_file_sha(fixture) != fixture_binding["file_sha256"]
        or fixture["manifest_sha256"] != fixture_binding["manifest_sha256"]
    ):
        raise ValueError("live fixture manifest differs from the execution plan")
    return krea_stage2_training_evidence.build_run_evidence(
        plan=dict(plan),
        plan_file_sha256=_canonical_file_sha(plan),
        approval=dict(approval),
        approval_file_sha256=_canonical_file_sha(approval),
        completion=dict(completion),
        completion_file_sha256=_canonical_file_sha(completion),
        run_output_root=Path(output_dir),
        fixture_manifest={
            "file_sha256": fixture_binding["file_sha256"],
            "manifest_sha256": fixture_binding["manifest_sha256"],
        },
        emitted_at_utc=completion["ended_at_utc"],
    )


def run_row(
    *,
    matrix: Mapping[str, Any],
    row_key: str,
    plan: Mapping[str, Any],
    approval: Mapping[str, Any],
    authority_bundle: Mapping[str, Any],
    output_dir: str | Path,
    completion_path: str | Path,
    run_evidence_path: str | Path,
    score_hook_path: str | Path,
    receipt_path: str | Path,
    run_cell=krea_stage2_execution.run_cell,
) -> tuple[dict[str, Any], bool]:
    """Run one fixed-GPU row, or strictly replay its receipt before skipping."""

    resolved_matrix = validate_matrix(matrix)
    row = _matrix_row(resolved_matrix, row_key)
    resolved_plan, resolved_approval, _identity = validate_row_controls(
        matrix=resolved_matrix,
        row_key=row_key,
        plan=plan,
        approval=approval,
        authority_bundle=authority_bundle,
    )
    receipt_target = _safe_path(receipt_path, "row receipt", must_exist=False)
    completion_target = _safe_path(
        completion_path, "row completion", must_exist=os.path.lexists(completion_path)
    )
    evidence_target = _safe_path(
        run_evidence_path, "row run evidence", must_exist=False
    )
    hook_target = _safe_path(score_hook_path, "row score hook", must_exist=False)
    if os.path.lexists(receipt_target) or os.path.lexists(completion_target):
        completion = _load_canonical(completion_target, "existing row completion")
        completion = krea_stage2_execution.validate_completion(
            completion, plan=resolved_plan, approval=resolved_approval
        )
        evidence = _replay_live_run(
            row=row,
            plan=resolved_plan,
            approval=resolved_approval,
            completion=completion,
            output_dir=output_dir,
        )
        evidence = _publish_or_replay(evidence_target, evidence, "row run evidence")
        hook = _score_hook(matrix=resolved_matrix, row=row, evidence=evidence)
        hook = _publish_or_replay(hook_target, hook, "row score hook")
        expected_receipt = _receipt(
            matrix=resolved_matrix,
            row=row,
            plan=resolved_plan,
            approval=resolved_approval,
            completion=completion,
            evidence=evidence,
            score_hook=hook,
        )
        receipt = _publish_or_replay(
            receipt_target, expected_receipt, "existing row receipt"
        )
        return (
            validate_receipt(
                receipt,
                matrix=resolved_matrix,
                row=row,
                plan=resolved_plan,
                approval=resolved_approval,
                completion=completion,
                evidence=evidence,
                score_hook=hook,
            ),
            True,
        )
    completion = run_cell(
        plan=resolved_plan,
        approval=resolved_approval,
        authority_controls=dict(authority_bundle),
        output_dir=Path(output_dir),
        completion_path=completion_target,
        gpu_device=row["gpu_device"],
    )
    completion = krea_stage2_execution.validate_completion(
        completion, plan=resolved_plan, approval=resolved_approval
    )
    evidence = _replay_live_run(
        row=row,
        plan=resolved_plan,
        approval=resolved_approval,
        completion=completion,
        output_dir=output_dir,
    )
    _publish_new(evidence_target, evidence)
    hook = _score_hook(matrix=resolved_matrix, row=row, evidence=evidence)
    _publish_new(hook_target, hook)
    receipt = _receipt(
        matrix=resolved_matrix,
        row=row,
        plan=resolved_plan,
        approval=resolved_approval,
        completion=completion,
        evidence=evidence,
        score_hook=hook,
    )
    _publish_new(receipt_target, receipt)
    return receipt, False


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--freeze", required=True, type=Path)
    build.add_argument("--production-identity", required=True, type=Path)
    build.add_argument("--created-at-utc", required=True)
    build.add_argument("--output", required=True, type=Path)
    validate = sub.add_parser("validate")
    validate.add_argument("--matrix", required=True, type=Path)
    validate.add_argument("--freeze", required=True, type=Path)
    validate.add_argument("--production-identity", required=True, type=Path)
    run = sub.add_parser("run-row")
    run.add_argument("--matrix", required=True, type=Path)
    run.add_argument("--row-key", required=True)
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--approval", required=True, type=Path)
    run.add_argument("--authority-bundle", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--completion", required=True, type=Path)
    run.add_argument("--run-evidence", required=True, type=Path)
    run.add_argument("--score-hook", required=True, type=Path)
    run.add_argument("--receipt", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    try:
        if args.command == "build":
            matrix = build_matrix(
                freeze_path=args.freeze,
                production_identity_path=args.production_identity,
                created_at_utc=args.created_at_utc,
            )
            publish_matrix(matrix, args.output)
            result: Any = matrix
        elif args.command == "validate":
            freeze = krea_density_seedb_freeze.validate_freeze(args.freeze)
            identity = _validate_production_identity(
                krea_stage2_production_identity.load(args.production_identity)
            )
            result = validate_matrix(
                _load_canonical(args.matrix, "Stage-2 matrix"),
                freeze=freeze,
                production_identity=identity,
            )
        else:
            result, replayed = run_row(
                matrix=_load_canonical(args.matrix, "Stage-2 matrix"),
                row_key=args.row_key,
                plan=_load_canonical(args.plan, "Stage-2 execution plan"),
                approval=_load_canonical(args.approval, "Stage-2 execution approval"),
                authority_bundle=_load_canonical(
                    args.authority_bundle, "Stage-2 authority bundle"
                ),
                output_dir=args.output_dir,
                completion_path=args.completion,
                run_evidence_path=args.run_evidence,
                score_hook_path=args.score_hook,
                receipt_path=args.receipt,
            )
            result = {"receipt": result, "replayed_existing": replayed}
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
