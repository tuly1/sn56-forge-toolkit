#!/usr/bin/env python3
"""Compose sealed per-arm Krea score batches for one final decision batch.

Each input remains an independently approved, exhaustive schema-2 score batch.
The additive aggregate is published only when those batches exactly exhaust a
separately sealed full campaign and share the same fixture, zero control,
evaluator, scorer runtime, provenance, and approval authority.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import krea_decision
except ImportError:  # pragma: no cover - direct script execution.
    import krea_decision  # type: ignore[no-redef]


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--member", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    result = krea_decision.assemble_additive_score_aggregates(
        campaign_path=args.campaign,
        aggregate_paths=args.member,
        output=args.output,
    )
    print(result["aggregate_sha256"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
