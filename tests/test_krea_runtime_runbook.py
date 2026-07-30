"""Keep copy/paste JSON examples aligned with exact runtime schemas."""

from __future__ import annotations

import json
from pathlib import Path
import re


RUNBOOK = (
    Path(__file__).parents[1]
    / "ops"
    / "calibration"
    / "week5"
    / "KREA-ADDITIVE-RUNTIME-BINDING.md"
)
TIMING_RUNBOOK = RUNBOOK.with_name("KREA-FIRST-GPU-TIMING-RUNBOOK.md")
SCORER_RUNBOOK = RUNBOOK.with_name("KREA-STAGE1-EXACT-SCORER-RUNBOOK.md")


def _json_blocks() -> list[dict]:
    text = RUNBOOK.read_text(encoding="utf-8")
    return [
        json.loads(block)
        for block in re.findall(r"```json\n(.*?)\n```", text, flags=re.DOTALL)
    ]


def test_layout_example_names_the_only_stage1_execution_surface():
    layout = _json_blocks()[0]

    assert layout["runtime"]["execution_surface"] == "staged_host_venv"
    assert set(layout["runtime"]["stage1_runtime_receipt"]) == {
        "path",
        "file_sha256",
        "receipt_sha256",
    }
    assert (
        layout["runtime"]["runtime_cache_policy"]["cross_capture_or_plan_reuse"]
        is False
    )


def test_runbook_orders_materializer_before_layout_prepare_and_has_no_manual_venv():
    text = RUNBOOK.read_text(encoding="utf-8")

    assert text.index(" materialize \\") < text.index(" prepare-layout \\")
    assert "python3 -m venv --copies /srv/forge-venv" not in text
    assert "timing_plan_file_sha256_plus_capture_id" in text


def test_profile_index_example_binds_external_authorization():
    profile_payload = _json_blocks()[1]

    assert set(profile_payload) == {
        "discovery_plan",
        "discovery_execution_authorization",
        "fixtures",
        "profiles",
    }


def test_timing_runbook_is_agent_bound_current_and_mount_contained():
    text = TIMING_RUNBOOK.read_text(encoding="utf-8")
    assert text.index("seal-margin \\") < text.index("seal-probe \\")
    assert '--technical-actor "$MARGIN_ACTOR"' in text
    assert '--discovery-authorization "$DISCOVERY_AUTH"' in text
    assert "MARGIN_APPROVED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)" in text
    assert "PROBE_APPROVED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)" in text
    assert "2026-07-28T00:00:00Z" not in text
    assert '--reviewer "NAMED HUMAN"' not in text
    assert "below `/campaign/controls`" in text
    assert "below\n`/app/checkpoints`" in text


def test_exact_scorer_runbook_is_literal_and_matches_owner_contract():
    text = SCORER_RUNBOOK.read_text(encoding="utf-8")
    for token in (
        "5473a9da95cc729cac65ae0309b1044224a40eb1e8961b77cd0e39eab846bb08",
        "uv venv --python 3.10.20",
        "--index-strategy unsafe-best-match",
        'uv pip check --python "$PY"',
        "stage-stage1-assets",
        "unset HF_TOKEN",
        "build-stage1-evaluator",
        "krea2_raw_fp8_scaled.safetensors",
        '--technical-actor "$SCORE_ACTOR"',
        '--discovery-authorization "$AUTH"',
        "/campaign/controls/admission/fixture-package-v2/D1/evaluation",
        "/campaign/controls/admission/fixtures/D1/fixture-manifest.json",
        "/campaign/controls/admission/admission-envelope.json",
        "POLICY_APPROVED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "DISCOVERY_DECIDED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    ):
        assert token in text
    assert "-eval.zip" not in text
    assert "/campaign/controls/admission/cross-fixture-review.json" not in text
    assert '"$PY" -m pip check' not in text
    assert "fresh `exact_score_plan_reviewer`" not in text
    assert "2026-07-29T00:00:00Z" not in text

    install = text[text.index('uv pip install --python "$PY"') :]
    install = install[: install.index('uv pip check --python "$PY"')]
    assert "--no-deps" in install
    assert "--index-strategy unsafe-best-match" in install
    assert "--extra-index-url https://download.pytorch.org/whl/cu128" in install
    assert "registry distribution pinned with exact `==`" in text
    assert "sole VCS distribution pinned to a full commit" in text
    assert "normalized name/version set plus the lock file identity" in text
    assert "wheel-byte verification" in text

    lock = SCORER_RUNBOOK.with_name("krea-stage1-exact-scorer-lock.txt")
    lines = lock.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 229
    vcs = [line for line in lines if " @ git+" in line]
    registry = [line for line in lines if line not in vcs]
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^=\s]+", line) for line in registry)
    assert len(vcs) == 1
    assert re.fullmatch(r"[A-Za-z0-9_.-]+ @ git\+https://[^@]+@[0-9a-f]{40}", vcs[0])
