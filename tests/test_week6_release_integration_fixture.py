from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = (
    ROOT / "tests" / "integration" / "sn56-week6-generate-release-fixture.py"
)
VALIDATOR = ROOT / "ops" / "release" / "sn56-week6-validate-timing-provenance.py"


def _validator():
    spec = importlib.util.spec_from_file_location("fixture_test_validator", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _materialized_working_tree(tmp_path: Path) -> Path:
    root = tmp_path / "materialized"
    root.mkdir(mode=0o700)
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        source = ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_mode = source.stat().st_mode
        destination.chmod(0o755 if source_mode & stat.S_IXUSR else 0o644)
    return root


def _identity() -> tuple[str, str]:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit, tree = completed.stdout.splitlines()
    return commit, tree


def _run_generator(
    materialized: Path,
    manifest: str,
    output: Path,
) -> subprocess.CompletedProcess[str]:
    commit, tree = _identity()
    return subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--materialized-repository",
            str(materialized),
            "--materialized-manifest-sha256",
            manifest,
            "--release-commit",
            commit,
            "--release-tree",
            tree,
            "--output-dir",
            str(output),
        ],
        env={
            "PYTHONDONTWRITEBYTECODE": "1",
            "SN56_EVIDENCE_ORIGIN": "real",
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_generator_produces_schema_current_package_that_real_validator_accepts(
    tmp_path,
):
    authority = _validator()
    materialized = _materialized_working_tree(tmp_path)
    manifest = authority.materialized_tree_manifest_sha256(str(materialized))
    output = tmp_path / "fixture"

    completed = _run_generator(materialized, manifest, output)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.rstrip().endswith("SN56_WEEK6_INTEGRATION_FIXTURE=PASS")
    expected = {
        "effective-runtime.json",
        "fixture-validation-receipt.json",
        "friday-gate.jsonl",
        "terminal-artifact.safetensors",
        "timing-profile.json",
    }
    assert {item.name for item in output.iterdir()} == expected
    assert authority.materialized_tree_manifest_sha256(str(materialized)) == manifest
    receipt = authority.load_json_file(
        str(output / "fixture-validation-receipt.json"),
        label="integration fixture receipt",
        expected_sha256=hashlib.sha256(
            (output / "fixture-validation-receipt.json").read_bytes()
        ).hexdigest(),
        maximum_bytes=256 * 1024,
    )[1]
    assert receipt["state"] == "PASS"
    assert receipt["schema"] == 3
    assert receipt["origin"] == "synthetic"
    assert receipt["evidence_class"] == "operator-attested"
    assert receipt["forge"]["repository"] == str(materialized)
    assert receipt["forge"]["materialized_manifest_sha256"] == manifest
    generator_source = GENERATOR.read_text(encoding="utf-8")
    assert '"origin": "synthetic"' in generator_source
    assert "--origin" not in generator_source


def test_generator_rejects_wrong_manifest_before_publishing_fixture(tmp_path):
    materialized = _materialized_working_tree(tmp_path)
    output = tmp_path / "fixture"

    completed = _run_generator(materialized, "0" * 64, output)

    assert completed.returncode != 0
    assert "materialized tree manifest differs" in completed.stderr
    assert not output.exists()
