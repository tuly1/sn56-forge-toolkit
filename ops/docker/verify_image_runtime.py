#!/usr/bin/env python3
"""Verify the certified pip version/VCS metadata inventory and constraints."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable, Sequence


GOLDEN_INVENTORY_SHA256 = (
    "9c4c15130508c547c67d891f559ca1a513cd62bd5a4b695eb25ceafccd0b850b"
)
PHASE1_CONSTRAINTS_SHA256 = (
    "864ed2d3c45f86464b189e3f1685e0578eae2af9ecf49e6bb63cadf3a85986ac"
)
PHASE1_EXCLUDED_NAMES = frozenset(
    {"easy-dwpose", "torch", "torchaudio", "torchcodec", "torchvision", "triton"}
)
ALLOWED_PIP_CHECK_LINES = (
    "easy-dwpose 1.0.3 has requirement huggingface_hub<1.0,>=0.26, "
    "but you have huggingface-hub 1.10.1.",
)


class VerificationError(RuntimeError):
    """The serialized package metadata differs from its certified inventory."""


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _run(command: Sequence[str], *, runner: Runner) -> subprocess.CompletedProcess[str]:
    return runner(
        list(command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _read_hashed_file(path: Path, expected_hash: str, label: str) -> str:
    contents = path.read_bytes()
    actual_hash = hashlib.sha256(contents).hexdigest()
    if actual_hash != expected_hash:
        raise VerificationError(
            f"{label} digest mismatch: expected={expected_hash} actual={actual_hash}"
        )
    return contents.decode("utf-8")


def distribution_name(requirement_line: str) -> str:
    """Return the normalized name from this lock's == or direct-VCS syntax."""

    if " @ " in requirement_line:
        name = requirement_line.split(" @ ", 1)[0]
    else:
        name = requirement_line.split("==", 1)[0]
    return name.strip().lower().replace("_", "-")


def phase1_excludes(requirement_line: str) -> bool:
    name = distribution_name(requirement_line)
    return (
        name in PHASE1_EXCLUDED_NAMES
        or name.startswith("nvidia-")
        or name.startswith("cuda-")
    )


def derive_phase1_constraints(inventory: str) -> str:
    """Derive constraints while leaving the CUDA/Torch resolution island open."""

    return "".join(
        line
        for line in inventory.splitlines(keepends=True)
        if line.strip() and not phase1_excludes(line)
    )


def verify_metadata_files(lock_path: Path, constraints_path: Path) -> dict[str, object]:
    inventory = _read_hashed_file(
        lock_path, GOLDEN_INVENTORY_SHA256, "golden inventory"
    )
    constraints = _read_hashed_file(
        constraints_path, PHASE1_CONSTRAINTS_SHA256, "phase-1 constraints"
    )
    derived_constraints = derive_phase1_constraints(inventory)
    if constraints != derived_constraints:
        diff = "".join(
            difflib.unified_diff(
                derived_constraints.splitlines(keepends=True),
                constraints.splitlines(keepends=True),
                fromfile="derived-phase1-constraints",
                tofile="checked-in-phase1-constraints",
            )
        )
        raise VerificationError(f"phase-1 constraint derivation mismatch:\n{diff}")
    return {
        "inventory": inventory,
        "constraints": constraints,
        "inventory_sha256": GOLDEN_INVENTORY_SHA256,
        "constraints_sha256": PHASE1_CONSTRAINTS_SHA256,
    }


def verify_runtime(
    lock_path: Path,
    constraints_path: Path,
    *,
    runner: Runner = subprocess.run,
    python_executable: str = sys.executable,
) -> dict[str, object]:
    """Verify the pip metadata inventory and sole intentional metadata conflict."""

    files = verify_metadata_files(lock_path, constraints_path)
    expected_inventory = str(files["inventory"])
    freeze = _run(
        [python_executable, "-m", "pip", "freeze", "--all"], runner=runner
    )
    if freeze.returncode != 0:
        raise VerificationError(
            f"pip freeze failed with exit {freeze.returncode}: {freeze.stderr.strip()}"
        )
    if freeze.stderr.strip():
        raise VerificationError(f"pip freeze emitted stderr: {freeze.stderr.strip()}")
    if freeze.stdout != expected_inventory:
        diff = "".join(
            difflib.unified_diff(
                expected_inventory.splitlines(keepends=True),
                freeze.stdout.splitlines(keepends=True),
                fromfile="golden-metadata-inventory",
                tofile="runtime-metadata-inventory",
            )
        )
        raise VerificationError(f"runtime metadata inventory mismatch:\n{diff}")

    pip_check = _run(
        [python_executable, "-m", "pip", "check"], runner=runner
    )
    observed_conflicts = tuple(
        line.strip() for line in pip_check.stdout.splitlines() if line.strip()
    )
    if pip_check.returncode != 1:
        raise VerificationError(
            "pip check exit mismatch: "
            f"expected=1 actual={pip_check.returncode} stderr={pip_check.stderr.strip()!r}"
        )
    if pip_check.stderr.strip():
        raise VerificationError(f"pip check emitted stderr: {pip_check.stderr.strip()}")
    if observed_conflicts != ALLOWED_PIP_CHECK_LINES:
        raise VerificationError(
            "pip check conflict mismatch: "
            f"expected={ALLOWED_PIP_CHECK_LINES!r} actual={observed_conflicts!r}"
        )

    return {
        "result": "PASS",
        "verification_scope": "pip-freeze-version-vcs-metadata",
        "inventory_sha256": GOLDEN_INVENTORY_SHA256,
        "inventory_entry_count": len(expected_inventory.splitlines()),
        "phase1_constraints_sha256": PHASE1_CONSTRAINTS_SHA256,
        "phase1_constraint_count": len(str(files["constraints"]).splitlines()),
        "allowed_pip_check_conflicts": list(ALLOWED_PIP_CHECK_LINES),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--files-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.files_only:
            files = verify_metadata_files(args.lock, args.constraints)
            result = {
                "result": "PASS",
                "verification_scope": "checked-in-metadata-files",
                "inventory_sha256": files["inventory_sha256"],
                "phase1_constraints_sha256": files["constraints_sha256"],
            }
        else:
            result = verify_runtime(args.lock, args.constraints)
    except (OSError, UnicodeError, VerificationError) as error:
        print(f"SN56_IMAGE_RUNTIME_INVENTORY=FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    print("SN56_IMAGE_RUNTIME_INVENTORY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
