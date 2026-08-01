"""Adversarial tests for the superseding density + Seed-B freeze."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import pytest

from ops.calibration import krea_density_gate as density
from ops.calibration import krea_density_seedb_freeze as freeze
from ops.calibration import krea_provenance


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical(path: Path, value: object) -> None:
    path.write_bytes(krea_provenance.canonical_bytes(value) + b"\n")


def _decision(*, variant: int = 0) -> dict[str, Any]:
    rows = []
    for fixture in freeze.FIXTURES:
        rows.append(
            {
                "task_id": f"{fixture.lower()}-zero-baseline",
                "fixture": fixture,
                "family": None,
                "seed_role": "A",
                "step": 0,
                "image_exposures": 0,
                "weighted_loss": 1.0,
                "candidate_binding": {
                    "file_sha256": _sha(f"{fixture}-zero")
                },
            }
        )
        for family_index, family in enumerate(freeze.FAMILIES):
            final_step = density._GEOMETRY[(fixture, family)][1]
            for row_index, step in enumerate((final_step // 2, final_step)):
                # Both points are deliberately inside the within-curve band;
                # the earlier point must remain the A-only selected checkpoint.
                loss = DecimalText("0.90") + DecimalText(str(family_index)) / 1000
                loss += DecimalText(str(row_index + variant)) / 10000
                rows.append(
                    {
                        "task_id": f"{fixture.lower()}-{family.lower()}-{step}",
                        "fixture": fixture,
                        "family": family,
                        "seed_role": "A",
                        "step": step,
                        "image_exposures": step,
                        "weighted_loss": float(loss),
                        "candidate_binding": {
                            "file_sha256": _sha(f"{fixture}-{family}-{step}")
                        },
                    }
                )
    return {"candidate_rows_for_krea_decision": rows}


def DecimalText(value: str):
    from decimal import Decimal

    return Decimal(value)


def _seedb_rows(
    anchor_decision: dict[str, Any], *, invert: bool = False
) -> dict[tuple[str, str | None], dict[str, Any]]:
    analyses, public = freeze._analysis(anchor_decision)
    anchors = freeze._source_anchor(analyses, public)
    rows = {}
    for fixture in freeze.FIXTURES:
        rows[(fixture, None)] = {"weighted_loss_decimal": DecimalText("1")}
        for index, family in enumerate(freeze.FAMILIES):
            value = (
                DecimalText("0.50") + DecimalText(str(index)) / 100
                if invert
                else DecimalText("0.90") + DecimalText(str(index)) / 1000
            )
            rows[(fixture, family)] = {
                "weighted_loss_decimal": value,
                "task_id": f"seedb-{fixture}-{family}",
                "source": anchors[(fixture, family)]["task_id"],
            }
    return rows


def test_seed_b_never_influences_checkpoint_mapping_or_tie_depth() -> None:
    anchor = _decision()
    chosen = _decision(variant=1)
    first = freeze._derive(
        anchor_decision=anchor,
        chosen_decision=chosen,
        seedb_rows=_seedb_rows(anchor, invert=False),
    )
    second = freeze._derive(
        anchor_decision=anchor,
        chosen_decision=chosen,
        seedb_rows=_seedb_rows(anchor, invert=True),
    )

    assert first["all_family_checkpoint_rules"] == second[
        "all_family_checkpoint_rules"
    ]
    for rule in first["all_family_checkpoint_rules"].values():
        assert {row["seed_role"] for row in rule["actual_mappings"]} == {"A"}
        assert all(not row["candidate_id"].startswith("seedb-") for row in rule["actual_mappings"])
    assert first["selection_algorithm"]["checkpoint_curves_and_rules_seed_roles"] == [
        "A"
    ]


def test_missing_or_partial_seed_b_uses_seed_a_anchor_uniformly(
    tmp_path: Path,
) -> None:
    anchor = _decision()
    result = freeze._derive(
        anchor_decision=anchor,
        chosen_decision=_decision(variant=1),
        seedb_rows=None,
    )
    assert result["seed_a_seed_b_agreement"]["full_B14_pooled"] is False
    assert result["seed_a_seed_b_agreement"]["partial_B_mixed"] is False
    for family in freeze.FAMILIES:
        for fixture in freeze.FIXTURES:
            row = result["family_relative_improvements"][family][fixture]
            assert row["B_anchor_relative_improvement"] is None
            assert row["primary_relative_improvement"] == row[
                "A_anchor_relative_improvement"
            ]

    partial_path = tmp_path / "partial-seedb.json"
    partial = {
        "schema": freeze.SEEDB_SCHEMA,
        "kind": freeze.SEEDB_KIND,
        "complete": False,
        "row_count": 13,
        "rows": [{"weighted_loss": -999999}],
    }
    _canonical(partial_path, partial)
    rows, binding = freeze._validate_seedb_full(
        partial_path,
        anchors={},
        cutoff=datetime(2026, 8, 1, 18, tzinfo=timezone.utc),
    )
    assert rows is None
    assert binding["state"] == "partial_at_cutoff"
    assert binding["pooling_eligible"] is False


def _rule(exposure: int) -> dict[str, Any]:
    return {"actual_mappings": [{"image_exposures": exposure}]}


def test_tie_break_is_depth_then_global_spread_then_preference() -> None:
    concept = {
        "K1": {"D1": DecimalText("0.200"), "D2": DecimalText("0.200")},
        "K2": {"D1": DecimalText("0.200"), "D2": DecimalText("0.200")},
        "K3": {"D1": DecimalText("0.200"), "D2": DecimalText("0.200")},
        "K4": {"D1": DecimalText("0.200"), "D2": DecimalText("0.200")},
        "K5": {"D1": DecimalText("0.200"), "D2": DecimalText("0.200")},
    }
    rules = {family: _rule(100) for family in concept}
    rules["K3"] = _rule(101)
    invocations: list[dict[str, Any]] = []
    chosen = freeze._pick(
        list(concept),
        primary={family: DecimalText("0.2") for family in concept},
        concept=concept,
        rules=rules,
        label="depth",
        invocations=invocations,
    )
    assert chosen == "K3"

    rules = {family: _rule(100) for family in concept}
    concept["K2"] = {"D1": DecimalText("0.205"), "D2": DecimalText("0.195")}
    concept["K4"] = {"D1": DecimalText("0.2"), "D2": DecimalText("0.2")}
    chosen = freeze._pick(
        ["K2", "K4"],
        primary={"K2": DecimalText("0.2"), "K4": DecimalText("0.2")},
        concept=concept,
        rules=rules,
        label="spread",
        invocations=invocations,
    )
    assert chosen == "K4"

    concept["K2"] = concept["K4"]
    chosen = freeze._pick(
        ["K4", "K2"],
        primary={"K2": DecimalText("0.2"), "K4": DecimalText("0.2")},
        concept=concept,
        rules=rules,
        label="preference",
        invocations=invocations,
    )
    assert chosen == "K2"
    assert [row["selection"] for row in invocations] == [
        "depth",
        "spread",
        "preference",
    ]


def test_cutoff_completeness_rejects_late_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = "d1-k1-final691"
    index = {
        "index_sha256": _sha("index"),
        "coverage_ledger": {"file_sha256": _sha("ledger")},
        "artifacts": [
            {
                "task_id": task_id,
                "selection_eligible": True,
                "validated_artifact": {
                    "status": {
                        "returncode": 0,
                        "ended_utc": "2026-08-01T18:00:01Z",
                    }
                },
            }
        ],
    }
    monkeypatch.setattr(
        freeze.krea_recovery_evidence,
        "load_index",
        lambda _path: (deepcopy(index), _sha("index-file")),
    )
    bundle = {
        "label": "plan0",
        "decision": {
            "final_recovery_index": {
                "path": "/tmp/final-index.json",
                "file_sha256": _sha("index-file"),
                "index_sha256": _sha("index"),
                "coverage_ledger_file_sha256": _sha("ledger"),
            },
            "candidate_rows_for_krea_decision": [{"task_id": task_id}],
        },
        "plan": {"rows": [{"task_id": task_id, "selected": True}]},
    }
    cutoff = datetime(2026, 8, 1, 18, tzinfo=timezone.utc)
    assert freeze._cutoff_complete(bundle, cutoff) is False
    index["artifacts"][0]["validated_artifact"]["status"]["ended_utc"] = (
        "2026-08-01T18:00:00Z"
    )
    assert freeze._cutoff_complete(bundle, cutoff) is True


def _mock_build_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict[str, Any], dict[str, bool]]:
    a59_path = tmp_path / "a59.json"
    a59_path.write_text("a59\n")
    index_sha = _sha("a59-semantic")
    ledger_sha = _sha("ledger")
    file_sha = _sha("a59-file")
    index = {
        "index_sha256": index_sha,
        "coverage_ledger": {"file_sha256": ledger_sha},
        "coverage": {"selection_eligible": 59},
        "artifacts": [
            {"selection_eligible": position < 59} for position in range(92)
        ],
    }
    monkeypatch.setattr(
        freeze.krea_recovery_evidence,
        "load_index",
        lambda _path: (deepcopy(index), file_sha),
    )
    recovery = {
        "path": str(a59_path),
        "file_sha256": file_sha,
        "index_sha256": index_sha,
        "coverage_ledger_file_sha256": ledger_sha,
    }

    def bundle(label: str, count: int, additional: int) -> dict[str, Any]:
        return {
            "label": label,
            "additional_target_count": additional,
            "selected_count": count,
            "plan": {"recovery_index": deepcopy(recovery)},
            "sidecar": {},
            "decision": {"label": label},
            "plan_binding": {"path": f"/{label}-plan"},
            "sidecar_binding": {"path": f"/{label}-sidecar"},
            "decision_binding": {"path": f"/{label}-decision"},
        }

    bundles = {
        "plan11": bundle("plan11", 70, 11),
        "plan9": bundle("plan9", 68, 9),
        "plan0": bundle("plan0", 59, 0),
    }
    monkeypatch.setattr(
        freeze,
        "_load_density_triplet",
        lambda label, **_kwargs: deepcopy(bundles[label]),
    )
    monkeypatch.setattr(
        freeze,
        "_validate_failure",
        lambda *_args, **_kwargs: (
            {"failure_class": freeze.FAILURE_CLASS},
            {"path": "/failure", "failure_semantic_sha256": _sha("failure")},
        ),
    )
    states = {"plan11": False, "plan9": True, "plan0": True}
    monkeypatch.setattr(
        freeze, "_cutoff_complete", lambda item, _cutoff: states[item["label"]]
    )
    monkeypatch.setattr(
        freeze,
        "_analysis",
        lambda _decision: ({}, {}),
    )
    monkeypatch.setattr(freeze, "_source_anchor", lambda *_args: {})
    monkeypatch.setattr(
        freeze,
        "_validate_seedb_full",
        lambda *_args, **_kwargs: (
            None,
            {"state": "absent_at_cutoff", "pooling_eligible": False},
        ),
    )
    monkeypatch.setattr(
        freeze,
        "_derive",
        lambda **_kwargs: {
            "finalist_family_ids": ["K2", "K0"],
            "checkpoint_rules": {},
            "all_family_checkpoint_rules": {},
        },
    )
    kwargs = {
        "plan0_path": tmp_path / "p0",
        "sidecar0_path": tmp_path / "s0",
        "decision0_path": tmp_path / "d0",
        "plan9_path": tmp_path / "p9",
        "sidecar9_path": tmp_path / "s9",
        "decision9_path": tmp_path / "d9",
        "plan11_path": tmp_path / "p11",
        "sidecar11_path": tmp_path / "s11",
        "decision11_path": tmp_path / "d11",
        "a59_index_path": a59_path,
        "failure_path": tmp_path / "failure.env",
        "seedb_results_path": None,
        "frozen_at_utc": "2026-08-01T18:00:01Z",
    }
    return {"kwargs": kwargs, "bundles": bundles, "index": index}, states


def test_build_chooses_highest_complete_plan_and_rejects_mixed_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values, states = _mock_build_inputs(monkeypatch, tmp_path)
    record = freeze._build(**values["kwargs"])
    assert record["chosen_density_plan"]["label"] == "plan9"
    states["plan11"] = True
    record = freeze._build(**values["kwargs"])
    assert record["chosen_density_plan"]["label"] == "plan11"

    values["bundles"]["plan9"]["plan"]["recovery_index"]["index_sha256"] = _sha(
        "other"
    )
    with pytest.raises(freeze.DensitySeedBFreezeError, match="one immutable A59"):
        freeze._build(**values["kwargs"])


def test_create_only_c_path_and_tampered_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(freeze.DensitySeedBFreezeError, match="sealed C evidence"):
        freeze._safe_path(tmp_path / "C1" / "freeze.json", "output", must_exist=False)

    source = tmp_path / "source.txt"
    source.write_text("bound")
    binding = freeze._binding(source, "source")
    bad = {**binding, "file_sha256": _sha("tampered")}
    with pytest.raises(freeze.DensitySeedBFreezeError, match="drifted"):
        freeze._validate_binding(bad, "source")

    empty = tmp_path / "empty.stderr"
    empty.write_bytes(b"")
    assert freeze._validate_binding(
        freeze._binding(empty, "empty stderr"), "empty stderr"
    )["bytes"] == 0
    with pytest.raises(freeze.DensitySeedBFreezeError, match="byte count is invalid"):
        freeze._validate_binding(
            {**freeze._binding(empty, "empty stderr"), "bytes": -1},
            "empty stderr",
        )

    body = {
        "schema": freeze.SCHEMA,
        "kind": freeze.FREEZE_KIND,
        "freeze_sha256": _sha("freeze"),
    }
    monkeypatch.setattr(freeze, "_build", lambda **_kwargs: body)
    output = tmp_path / "freeze.json"
    freeze.freeze_finalists(output=output)
    assert output.exists()
    with pytest.raises(FileExistsError):
        freeze.freeze_finalists(output=output)
