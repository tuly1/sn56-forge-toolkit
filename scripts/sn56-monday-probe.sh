#!/usr/bin/env bash
# Manifest-bound Monday pre-entry probe runner. Read-only: this wrapper only
# writes local evidence and delegates checks to sn56-preentry-probe-v2.sh.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$SCRIPT_DIR/../release/week9-release-manifest.json"
DOCKER_POLICY="$SCRIPT_DIR/../release/week9-docker-policy.json"
READINESS_RECEIPT="$SCRIPT_DIR/../release/week9-release-readiness.json"
PROBE="$SCRIPT_DIR/sn56-preentry-probe-v2.sh"
OUTDIR="${SN56_PROBE_OUTDIR:-/Users/atulyashetty/Test/SN56-project/evidence/monday-probe-20260824}"
ATTEMPTS="${SN56_PROBE_ATTEMPTS:-3}"
RETRY_DELAY="${SN56_PROBE_RETRY_DELAY:-20}"
declare -a PROBE_ARGS=()

usage() {
  cat <<'EOF'
Usage: sn56-monday-probe.sh [--manifest PATH]
       sn56-monday-probe.sh [--docker-policy PATH] [--readiness-receipt PATH]
       sn56-monday-probe.sh --manifest PATH --mode mock --fixtures DIR [--now ISO]

The target commit and production literals come only from the release manifest.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --manifest)
      [ $# -ge 2 ] || { echo "--manifest requires PATH" >&2; exit 2; }
      MANIFEST="$2"; shift 2 ;;
    --docker-policy)
      [ $# -ge 2 ] || { echo "$1 requires PATH" >&2; exit 2; }
      DOCKER_POLICY="$2"; shift 2 ;;
    --readiness-receipt)
      [ $# -ge 2 ] || { echo "$1 requires PATH" >&2; exit 2; }
      READINESS_RECEIPT="$2"; shift 2 ;;
    --mode|--fixtures|--now)
      [ $# -ge 2 ] || { echo "$1 requires a value" >&2; exit 2; }
      PROBE_ARGS+=("$1" "$2"); shift 2 ;;
    --skip-chain)
      PROBE_ARGS+=("$1"); shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

RUNDIR="$(mktemp -d "${TMPDIR:-/tmp}/sn56-monday-probe.XXXXXX")"
trap 'rm -rf "$RUNDIR"' EXIT
MANIFEST_SOURCE="$MANIFEST"
DOCKER_POLICY_SOURCE="$DOCKER_POLICY"
READINESS_SOURCE="$READINESS_RECEIPT"
MANIFEST_SNAPSHOT="$RUNDIR/release-manifest.json"
DOCKER_POLICY_SNAPSHOT="$RUNDIR/docker-policy.json"
READINESS_SNAPSHOT="$RUNDIR/release-readiness.json"
SNAPSHOT_SHA256S="$(python3 - \
  "$MANIFEST_SOURCE" "$MANIFEST_SNAPSHOT" \
  "$DOCKER_POLICY_SOURCE" "$DOCKER_POLICY_SNAPSHOT" \
  "$READINESS_SOURCE" "$READINESS_SNAPSHOT" <<'PY'
import hashlib
import os
import pathlib
import sys

try:
    pairs = list(zip(map(pathlib.Path, sys.argv[1::2]), map(pathlib.Path, sys.argv[2::2])))
    hashes = []
    for source, destination in pairs:
        if source.is_symlink():
            raise ValueError(f"reviewed release artifact must not be a symlink: {source}")
        raw = source.read_bytes()
        with destination.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        hashes.append(hashlib.sha256(raw).hexdigest())
    for source, destination in (pairs[0], pairs[2]):
        signature_source = pathlib.Path(f"{source}.sig")
        if signature_source.exists() or signature_source.is_symlink():
            if signature_source.is_symlink():
                raise ValueError(f"reviewed detached signature must not be a symlink: {signature_source}")
            signature_raw = signature_source.read_bytes()
            signature_destination = pathlib.Path(f"{destination}.sig")
            with signature_destination.open("xb") as handle:
                handle.write(signature_raw)
                handle.flush()
                os.fsync(handle.fileno())
except Exception as exc:
    print(f"release artifact snapshot error: {exc}", file=sys.stderr)
    raise SystemExit(2)
print(" ".join(hashes))
PY
)" || exit 2
read -r MANIFEST_SHA256 DOCKER_POLICY_SHA256 READINESS_SHA256 <<< "$SNAPSHOT_SHA256S"

# Every retry consumes this one private three-artifact snapshot. The banner,
# probe checks, and JSON therefore cannot consume a different manifest, policy,
# or readiness receipt if a source path is replaced during the wrapper run.
TARGET_COMMIT="$(python3 - "$MANIFEST_SNAPSHOT" <<'PY'
import json
import re
import sys
from pathlib import Path

try:
    doc = json.loads(Path(sys.argv[1]).read_text())
except Exception as exc:
    print(f"manifest error: cannot read {sys.argv[1]}: {exc}", file=sys.stderr)
    raise SystemExit(2)
if not isinstance(doc, dict) or doc.get("schema_version") != 1:
    print("manifest error: schema_version must equal 1", file=sys.stderr)
    raise SystemExit(2)
if doc.get("release_state") not in {"hold", "ready"}:
    print("manifest error: release_state must equal 'hold' or 'ready'", file=sys.stderr)
    raise SystemExit(2)
target = doc.get("target", {}).get("commit") if isinstance(doc.get("target"), dict) else None
if not isinstance(target, str) or re.fullmatch(r"[0-9a-f]{40}", target) is None:
    print("manifest error: target.commit must be a lowercase full 40-hex SHA", file=sys.stderr)
    raise SystemExit(2)
print(target)
PY
)" || exit 2

case "$ATTEMPTS" in 1|2|3) ;; *) echo "SN56_PROBE_ATTEMPTS must be 1, 2, or 3" >&2; exit 2 ;; esac
[[ "$RETRY_DELAY" =~ ^[0-9]+$ ]] || { echo "SN56_PROBE_RETRY_DELAY must be a non-negative integer" >&2; exit 2; }
[ -x "$PROBE" ] || [ -f "$PROBE" ] || { echo "probe not found: $PROBE" >&2; exit 2; }

mkdir -p "$OUTDIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
JSON="$OUTDIR/probe-$STAMP.json"
LOG="$OUTDIR/probe-$STAMP.log"

echo "SN56 MONDAY PROBE — $STAMP (manifest target ${TARGET_COMMIT:0:12})" | tee "$LOG"
echo "manifest source: $MANIFEST_SOURCE (sha256 $MANIFEST_SHA256; private snapshot for all attempts)" | tee -a "$LOG"
echo "Docker policy source: $DOCKER_POLICY_SOURCE (sha256 $DOCKER_POLICY_SHA256; private snapshot for all attempts)" | tee -a "$LOG"
echo "readiness source: $READINESS_SOURCE (sha256 $READINESS_SHA256; private snapshot for all attempts)" | tee -a "$LOG"

rc=1
attempt=1
while [ "$attempt" -le "$ATTEMPTS" ]; do
  echo "--- attempt $attempt/$ATTEMPTS ---" | tee -a "$LOG"
  if [ "${#PROBE_ARGS[@]}" -eq 0 ]; then
    # Bash 3.2 raises an unbound-variable error for an empty array expansion
    # under `set -u`; live mode legitimately has no optional probe arguments.
    bash "$PROBE" \
      --manifest "$MANIFEST_SNAPSHOT" \
      --docker-policy "$DOCKER_POLICY_SNAPSHOT" \
      --readiness-receipt "$READINESS_SNAPSHOT" \
      --json "$JSON" >>"$LOG" 2>&1
  else
    bash "$PROBE" \
      --manifest "$MANIFEST_SNAPSHOT" \
      --docker-policy "$DOCKER_POLICY_SNAPSHOT" \
      --readiness-receipt "$READINESS_SNAPSHOT" \
      --json "$JSON" "${PROBE_ARGS[@]}" >>"$LOG" 2>&1
  fi
  rc=$?
  [ "$rc" -eq 0 ] && break
  # Retry only the known-flaky chain lookup. Manifest, route, pin, host, and
  # watcher failures are deterministic stop signals, not reasons to burn time.
  RETRYABLE="$(python3 -c '
import json,sys
try: checks=json.load(open(sys.argv[1]))["checks"]
except Exception: print("no"); raise SystemExit
failed={row.get("id") for row in checks if row.get("state")=="FAIL"}
print("yes" if failed=={"chain.registered"} else "no")
' "$JSON" 2>/dev/null || echo no)"
  if [ "$RETRYABLE" != "yes" ]; then
    echo "attempt $attempt has a non-retryable failure; stopping retries" | tee -a "$LOG"
    break
  fi
  if [ "$attempt" -lt "$ATTEMPTS" ]; then
    echo "attempt $attempt exited $rc; retrying in ${RETRY_DELAY}s" | tee -a "$LOG"
    [ "$RETRY_DELAY" -eq 0 ] || sleep "$RETRY_DELAY"
  fi
  attempt=$((attempt + 1))
done

echo "======================================================" | tee -a "$LOG"
if [ "$rc" -eq 0 ]; then
  WARNS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("warns", "?"))' "$JSON" 2>/dev/null || echo '?')"
  RESULT_MODE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("mode", "?"))' "$JSON" 2>/dev/null || echo '?')"
  echo "RESULT: GREEN ($RESULT_MODE) — 0 failures, $WARNS warning(s); target ${TARGET_COMMIT:0:12}. ($JSON)" | tee -a "$LOG"
else
  echo "RESULT: FAIL (exit $rc) — REVIEW BEFORE 12:30 UTC. Failing checks:" | tee -a "$LOG"
  grep -iE "FAIL" "$LOG" | grep -v "exited\|REVIEW" | tail -20 | tee -a "$LOG"
  echo "Full JSON: $JSON" | tee -a "$LOG"
fi

if [ "${SN56_DISABLE_NOTIFICATION:-0}" != "1" ]; then
  osascript -e "display notification \"$( [ "$rc" -eq 0 ] && echo 'GREEN' || echo 'FAIL - review now' )\" with title \"SN56 Monday probe\" sound name \"Glass\"" 2>/dev/null || true
fi

exit "$rc"
