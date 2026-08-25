#!/usr/bin/env bash
# One-line, manifest-bound Week-11 emergency rollback to exact current
# Ideogram-CONTENT production. The older 59e0698c production commit is
# intentionally not wired here: it is secondary recovery only, never the
# immediate rollback target.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOINT="$SCRIPT_DIR/sn56-week11-repoint.sh"
MANIFEST="$SCRIPT_DIR/../release/week11-release-manifest.json"
EXPECTED_ROLLBACK="fe9749c027df511b7566b474e6f8524f86b01f83"
declare -a FORWARD_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --manifest)
      [ $# -ge 2 ] || { echo "FATAL: --manifest requires a path" >&2; exit 2; }
      MANIFEST="$2"; shift 2 ;;
    *)
      FORWARD_ARGS+=("$1"); shift ;;
  esac
done

if [ ! -x "$REPOINT" ]; then
  echo "FATAL: repoint implementation is missing or not executable: $REPOINT" >&2
  exit 2
fi

ACTUAL_ROLLBACK="$(python3 - "$MANIFEST" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        print(json.load(handle)["rollback"]["commit"])
except Exception as exc:
    print(f"FATAL: cannot read rollback identity: {exc}", file=sys.stderr)
    raise SystemExit(2)
PY
)"

if [ "$ACTUAL_ROLLBACK" != "$EXPECTED_ROLLBACK" ]; then
  echo "FATAL: manifest rollback is $ACTUAL_ROLLBACK, expected $EXPECTED_ROLLBACK" >&2
  exit 2
fi

printf 'SN56 WEEK-11 ROLLBACK -> %s (exact current production)\n' "$EXPECTED_ROLLBACK"
exec "$REPOINT" --manifest "$MANIFEST" --rollback "${FORWARD_ARGS[@]}"
