#!/usr/bin/env python3
"""Build the exact Week-5 fc70 cell assembly payload on the GPU host.

The accepted arm inputs are staged under ``/campaign/controls``.  The timing
paths below are the completed D1/A timing namespace and are deliberately fixed:
the helper does not search for "latest" evidence and cannot silently bind a
different probe.  It hashes every binding from the observed host bytes and
publishes the unsealed assembly payload create-only.  ``krea_fc70_cell_queue``
then seals that payload and assembles the per-cell plans.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


CONTROLS = Path("/campaign/controls")
ADMISSION_ENVELOPE = CONTROLS / "admission.58822b4/admission-envelope.json"
ARM_INPUTS = CONTROLS / "fc70-arm-inputs.json"
PROBE_PLAN = CONTROLS / "timing-probe-D1-A-58822b4-r2.json"
MARGIN_POLICY = CONTROLS / "timing-margin.58822b4.json"
TIMING_ROOT = Path("/campaign/evidence/timing/D1-A-58822b4-r2")
RAW_TIMING = TIMING_ROOT / "raw-timing.json"
END_TO_END_VALIDATION = TIMING_ROOT / "heldout-validation.json"
MEASUREMENT_CAPTURES = tuple(TIMING_ROOT / f"timing-{name}.json" for name in "abc")
HELDOUT_CAPTURES = (TIMING_ROOT / "heldout-e2e.json",)
HELDOUT_RUN_RECORDS = (
    Path("/campaign/krea-timing-D1-A-58822b4-r2/conditions")
    / "wk5-d1-k1-a-timing-70aac7899e94.json",
)
DEFAULT_OUTPUT = CONTROLS / "admitted-fixtures.assembly-spec.payload.json"

FIXTURE_PATHS = {
    "D1": {
        "training_archive": "/campaign/controls/admission/fixture-package-v2/D1/training.zip",
        "evaluation_dataset": "/campaign/controls/admission/fixture-package-v2/D1/evaluation",
    },
    "D2": {
        "training_archive": "/campaign/controls/admission/fixture-package-v2/D2/training.zip",
        "evaluation_dataset": "/campaign/controls/admission/fixture-package-v2/D2/evaluation",
    },
}

EXPECTED_BASE_MODEL = {
    "model_id": "krea/Krea-2-Raw",
    "revision": "b2e772263cfa934848fde713159d1553e086778c",
    "training_identity_sha256": (
        "ad934e4126cc01d408b6cdb980ca098ce95a7a0c2ef078b1286efeb7f91c666f"
    ),
    "evaluation_assets": {
        "diffusion_model": {
            "canonical_path": (
                "/workspace/krea-stage1/src/ComfyUI/models/diffusion_models/"
                "krea2_raw_fp8_scaled.safetensors"
            ),
            "sha256": (
                "48cd5d6c100297968349b41a8e77c6591d1dac18a215807f5f25f59e5c54cd61"
            ),
            "bytes": 13141730784,
        },
        "text_encoder": {
            "canonical_path": (
                "/workspace/krea-stage1/src/ComfyUI/models/text_encoders/"
                "qwen3vl_4b_fp8_scaled.safetensors"
            ),
            "sha256": (
                "54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094"
            ),
            "bytes": 5242467968,
        },
        "vae": {
            "canonical_path": (
                "/workspace/krea-stage1/src/ComfyUI/models/vae/"
                "qwen_image_vae.safetensors"
            ),
            "sha256": (
                "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"
            ),
            "bytes": 253806246,
        },
    },
}

_SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def semantic_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_file(value: str | Path, label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(value)))
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    return path


def load_canonical(value: str | Path, label: str) -> tuple[Path, dict[str, Any], str]:
    path = safe_file(value, label)
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    if raw != canonical_bytes(document) + b"\n":
        raise ValueError(f"{label} must be canonical JSON plus one newline: {path}")
    return path, document, hashlib.sha256(raw).hexdigest()


def binding(path: Path, label: str) -> dict[str, str]:
    observed, _, digest = load_canonical(path, label)
    return {"path": str(observed), "sha256": digest}


def validate_arm_inputs(path: Path) -> dict[str, Any]:
    path, document, _ = load_canonical(path, "fc70 arm inputs")
    expected = {"schema", "kind", "source", "arms", "staged_files", "manifest_sha256"}
    if set(document) != expected:
        raise ValueError("fc70 arm-input keys mismatch")
    body = {key: value for key, value in document.items() if key != "manifest_sha256"}
    if (
        document["schema"] != 1
        or document["kind"] != "forge-krea-fc70-arm-inputs"
        or document["source"] != "accepted-week5-artifacts-only"
        or document["manifest_sha256"] != semantic_sha(body)
    ):
        raise ValueError("fc70 arm-input identity is invalid")
    arms = document["arms"]
    if not isinstance(arms, dict) or set(arms) != {f"K{i}" for i in range(6)}:
        raise ValueError("fc70 arm inputs must bind exactly K0..K5")
    for arm_id, arm in arms.items():
        if not isinstance(arm, dict) or set(arm) != {"arm_basis", "execution_recipe"}:
            raise ValueError(f"fc70 arm {arm_id} is malformed")
    staged = document["staged_files"]
    if not isinstance(staged, list) or not staged:
        raise ValueError("fc70 arm inputs have no staged-file ledger")
    seen: set[str] = set()
    for index, row in enumerate(staged):
        if not isinstance(row, dict) or set(row) != {"relative_path", "sha256", "bytes"}:
            raise ValueError(f"staged_files[{index}] is malformed")
        relative = row["relative_path"]
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in seen
        ):
            raise ValueError(f"staged_files[{index}] path is unsafe or duplicated")
        seen.add(relative)
        staged_path = safe_file(path.parent / relative, f"staged_files[{index}]")
        if (
            not isinstance(row["sha256"], str)
            or not _SHA256.fullmatch(row["sha256"])
            or file_sha(staged_path) != row["sha256"]
            or not isinstance(row["bytes"], int)
            or isinstance(row["bytes"], bool)
            or staged_path.stat().st_size != row["bytes"]
        ):
            raise ValueError(f"staged_files[{index}] identity mismatch")
    return arms


def admitted_fixtures(admission_envelope: Path) -> dict[str, Any]:
    """Bind paths to the identities in the already sealed admission artifacts."""

    envelope_path, envelope, _ = load_canonical(
        admission_envelope, "fixture admission envelope"
    )
    envelope_body = {
        key: value for key, value in envelope.items() if key != "envelope_sha256"
    }
    slots = envelope.get("discovery_fixtures")
    if (
        envelope.get("schema") != 1
        or envelope.get("kind") != "forge-krea-fixture-admission-envelope"
        or envelope.get("admission_authorized") is not True
        or envelope.get("gpu_execution_authorized") is not False
        or envelope.get("envelope_sha256") != semantic_sha(envelope_body)
        or not isinstance(slots, dict)
        or set(slots) != {"D1", "D2"}
    ):
        raise ValueError("fixture admission envelope is invalid")
    fixtures: dict[str, Any] = {}
    for role in ("D1", "D2"):
        slot = slots[role]
        binding_row = slot.get("manifest") if isinstance(slot, dict) else None
        if not isinstance(binding_row, dict):
            raise ValueError(f"{role} admission manifest binding is invalid")
        relative = binding_row.get("relative_path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError(f"{role} admission manifest path is unsafe")
        manifest_path, manifest, manifest_file_sha = load_canonical(
            envelope_path.parent / relative, f"{role} admitted fixture manifest"
        )
        manifest_body = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        archive = manifest.get("training_archive")
        evaluation = manifest.get("evaluation_dataset_identity")
        if (
            binding_row.get("file_sha256") != manifest_file_sha
            or binding_row.get("manifest_sha256") != manifest.get("manifest_sha256")
            or manifest.get("manifest_sha256") != semantic_sha(manifest_body)
            or manifest.get("schema") != 2
            or manifest.get("kind") != "forge-krea-curated-fixture"
            or manifest.get("experimental_role") != role
            or not isinstance(archive, dict)
            or not _SHA256.fullmatch(str(archive.get("sha256", "")))
            or not isinstance(evaluation, dict)
            or not _SHA256.fullmatch(str(evaluation.get("sha256", "")))
        ):
            raise ValueError(f"{role} admitted fixture identity is invalid")
        fixtures[role] = {
            "training_archive": {
                "path": FIXTURE_PATHS[role]["training_archive"],
                "sha256": archive["sha256"],
            },
            "evaluation_dataset": {
                "path": FIXTURE_PATHS[role]["evaluation_dataset"],
                "sha256": evaluation["sha256"],
            },
        }
    return fixtures


def build_payload(
    *,
    admission_envelope: Path = ADMISSION_ENVELOPE,
    arm_inputs: Path = ARM_INPUTS,
    probe_plan: Path = PROBE_PLAN,
    margin_policy: Path = MARGIN_POLICY,
    raw_timing: Path = RAW_TIMING,
    end_to_end_validation: Path = END_TO_END_VALIDATION,
    measurement_captures: tuple[Path, ...] = MEASUREMENT_CAPTURES,
    heldout_captures: tuple[Path, ...] = HELDOUT_CAPTURES,
    heldout_run_records: tuple[Path, ...] = HELDOUT_RUN_RECORDS,
) -> dict[str, Any]:
    arms = validate_arm_inputs(arm_inputs)
    probe_path, probe, probe_sha = load_canonical(probe_plan, "D1/A probe plan")
    base_model = probe.get("base_model")
    if base_model != EXPECTED_BASE_MODEL:
        raise ValueError("D1/A probe plan base_model differs from the admitted identity")
    if not measurement_captures or not heldout_captures or not heldout_run_records:
        raise ValueError("timing evidence arrays must be non-empty")
    timing_evidence = {
        "raw_sample_manifest": binding(raw_timing, "raw timing sample manifest"),
        "margin_policy": binding(margin_policy, "timing margin policy"),
        "end_to_end_validation": binding(
            end_to_end_validation, "end-to-end timing validation"
        ),
        "probe_contract": {"path": str(probe_path), "sha256": probe_sha},
        "measurement_captures": [
            binding(path, f"measurement capture {index}")
            for index, path in enumerate(measurement_captures)
        ],
        "heldout_captures": [
            binding(path, f"heldout capture {index}")
            for index, path in enumerate(heldout_captures)
        ],
        "heldout_run_records": [
            binding(path, f"heldout run record {index}")
            for index, path in enumerate(heldout_run_records)
        ],
    }
    return {
        "schema": 1,
        "kind": "forge-krea-fc70-cell-assembly-spec",
        "task_id_prefix": "week5-krea",
        "expected_repo_prefix": "week5-krea",
        "timing_evidence": timing_evidence,
        "base_model": base_model,
        "fixtures": admitted_fixtures(admission_envelope),
        "arms": arms,
    }


def publish_create_only(path: Path, value: Mapping[str, Any]) -> tuple[Path, str]:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"output has a symlink ancestor: {current}")
        current = current.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return path, hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output, digest = publish_create_only(args.output, build_payload())
    print(
        json.dumps(
            {"output": str(output), "file_sha256": digest},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
