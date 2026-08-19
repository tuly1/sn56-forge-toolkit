#!/usr/bin/env python3
"""Regenerate the week9-rc golden ai-toolkit flux config for the fallback
byte-identity test (tests/test_week9_flux_family.py).

The golden pins the EXACT config bytes the week9-rc (c70611f) ai-toolkit flux
path emits for a fixed task shape, with the run-specific filesystem root
replaced by the literal token ``FORGE-GOLDEN-ROOT``.  The byte-identity test replays the
same shape through the week9-flux-family kohya-route FAILURE path and asserts
the fallback wrote these bytes (after substituting its tmp root back in).

Regeneration (only needed if week9-rc itself moves — the golden is then no
longer a week9-rc attestation and this provenance block must be updated):

    PYTHONPATH=/path/to/checkout-of-week9-rc \
        python3 tests/data/week9_flux_family/regen_golden.py \
        tests/data/week9_flux_family/golden_week9rc_flux_aitoolkit_config.yaml

Provenance of the checked-in golden:
- generated 2026-08-19 with PYTHONPATH at forge-toolkit week9-rc
  c70611f929dea311c1646e6631fa559e8509ea2c
  (worktree workspaces/worktrees/week9-rc-trial, read-only import);
- inputs mirror the byte-identity test exactly: flux snapshot task, 12 train
  pairs, hours_to_complete 0.7 (a 0.75h task after cli/export accounting),
  selection/evalgrid/geometry env gates all unset (dormant);
- expected emission at week9-rc: steps 724 (STEP_TABLE law at n=12) and
  save_every 145 — the same numbers the beta Phase D "OURS" arm emission
  showed live (evidence/week9-gpu-campaign-20260819/beta/REPORT.md §5).
"""

from __future__ import annotations

import os
import sys
import tempfile

TOKEN = "FORGE-GOLDEN-ROOT"


def main(out_path: str) -> None:
    for gate in (
        "FORGE_HOLDOUT_SELECTION_TYPES",
        "FORGE_EVALGRID_SNAP_TYPES",
        "FORGE_EVAL_GEOMETRY_TYPES",
        "FORGE_TEMPLATES_DIR",
    ):
        os.environ.pop(gate, None)

    from forge.config import build_config, write_config
    from forge.data.schema import ImageSpec

    spec = ImageSpec.build(
        task_id="flux-task",
        model="org/snapshot-flux",
        model_type="flux",
        expected_repo_name="flux-output",
        trigger_word="TOK",
        dataset_zip=None,
    )
    # The config embeds these three path values; pin them to the token the
    # test substitutes.  (Class-level property override, same technique the
    # test suite uses.)
    ImageSpec.training_folder = property(
        lambda self: f"{TOKEN}/checkpoints/flux-task"
    )
    ImageSpec.dataset_images_dir = property(lambda self: f"{TOKEN}/dataset/images")
    ImageSpec.cached_model_dir = property(lambda self: f"{TOKEN}/cache/model")

    cfg = build_config(spec, num_images=12, hours_to_complete=0.7)
    with tempfile.TemporaryDirectory() as tmp:
        scratch = os.path.join(tmp, "config.yaml")
        write_config(cfg, scratch)
        with open(scratch, "rb") as fh:
            data = fh.read()
    with open(out_path, "wb") as fh:
        fh.write(data)
    print(f"wrote {out_path} ({len(data)} bytes)")


if __name__ == "__main__":
    main(sys.argv[1])
