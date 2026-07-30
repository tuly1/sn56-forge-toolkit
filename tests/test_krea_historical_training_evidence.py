from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT / "ops" / "calibration"))

import krea_historical_training_evidence as historical  # noqa: E402


def test_exact_daf9_validator_graph_loads_in_isolated_namespace(tmp_path: Path) -> None:
    checkout = tmp_path / "forge-daf9a252"
    subprocess.run(
        ["/usr/bin/git", "clone", "--quiet", "--no-local", str(_ROOT), str(checkout)],
        check=True,
        timeout=60,
    )
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(checkout),
            "checkout",
            "--quiet",
            "daf9a2528f4079ed06180c7e6d712a684a4170f0",
        ],
        check=True,
        timeout=30,
    )
    identity = historical.capture_identity(checkout)
    modules = historical.load_modules(identity)
    assert identity["tree_sha1"] == "953b58bdca842294ef7dfa1e54a16db52e5b74a2"
    assert (
        modules["execution_surface_policy"].POLICY["policy_sha256"]
        == "98b59fd90dbf4ea213c860f873bc472cadc66714c7b9118672de2474f020f5f3"
    )
    assert Path(modules["training_evidence"].__file__).is_relative_to(checkout)
    for name in (
        "batch_evaluate",
        "execution_plan",
        "discovery_authorization",
        "delegated_review_contract",
        "fixture",
        "fixture_admission",
    ):
        assert Path(modules[name].__file__).is_relative_to(checkout)

    (checkout / "untracked-downgrade.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact clean daf9a252 worktree"):
        historical.capture_identity(checkout)
