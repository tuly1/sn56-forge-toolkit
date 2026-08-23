#!/usr/bin/env bash
# One-line, manifest-bound emergency rollback to the exact production prestate.
# This wrapper never accepts a target SHA. The signed final manifest remains the
# authority for the released target; its rollback identity is fixed here only as
# a defensive assertion before delegating to the verified repoint path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOINT="$SCRIPT_DIR/sn56-week6-repoint.sh"
MANIFEST="$SCRIPT_DIR/../release/week9-release-manifest.json"
EXPECTED_ROLLBACK="75a0a20c2deda82cfa727e082e60a95bea5befb3"
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

printf 'SN56 ROLLBACK -> %s (exact production prestate)\n' "$EXPECTED_ROLLBACK"
exec "$REPOINT" --manifest "$MANIFEST" --rollback "${FORWARD_ARGS[@]}"
