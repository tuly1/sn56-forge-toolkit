#!/usr/bin/env bash
# =============================================================================
# sn56-week6-repoint.sh
#
# Switch the IMAGE training-repo pin served by the SN56 miner endpoint, safely.
#
#   Target file : /home/miner/god/miner/endpoints/training_repo.py   (hetzner)
#   The pin     : a Python string literal, _REPOS[TournamentType.IMAGE].commit_hash
#                 It is TEXT IN SOURCE. It is NOT an environment variable and it
#                 does NOT appear in /home/miner/god/.1.env (verified).
#   Untouched   : _REPOS[TournamentType.TEXT].commit_hash (the text-tournament
#                 pin, 8f11684e30a556b305dec9dd8eec9794bdae8cde), both
#                 github_repo URLs, the deliberate absence of an ENVIRONMENT
#                 entry (that absence is what yields 404 and avoids the 0.25 TAO
#                 entry fee), and factory_router().
#
# The edit is located by AST, not by regex: the editor walks to the IMAGE entry
# of the _REPOS dict and rewrites exactly that one 40-char literal by byte
# offset. It is length-preserving, so the file size does not change and exactly
# one line differs.
#
# Verification after restart proves the RUNNING interpreter carries the new pin:
# the regenerated __pycache__/training_repo.cpython-312.pyc is grepped for the
# literal. (Verified on the live host: each sha appears exactly once in both the
# source and the compiled .pyc.)
#
# The target is read only from a reviewed release manifest. There is
# deliberately no target-SHA argument or environment override.
#
# USAGE
#   ./sn56-week6-repoint.sh --manifest MANIFEST --contract-only
#   ./sn56-week6-repoint.sh --manifest MANIFEST --dry-run
#   ./sn56-week6-repoint.sh --manifest MANIFEST
#   ./sn56-week6-repoint.sh --manifest MANIFEST --rollback
#
# EXIT CODES
#   0 success                     4 reachability proof failed
#   2 usage error                 5 preflight failed (host state not as expected)
#   3 hard-abort window passed    6 verification failed -> AUTO-ROLLED BACK
#                                 7 verification failed -> ROLLBACK ALSO FAILED
# =============================================================================
set -uo pipefail

# ------------------------------ constants -----------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTRACT="$SCRIPT_DIR/sn56-release-contract.py"
DEFAULT_MANIFEST="$SCRIPT_DIR/../release/week9-release-manifest.json"
DEFAULT_DOCKER_POLICY="$SCRIPT_DIR/../release/week9-docker-policy.json"
DEFAULT_READINESS_RECEIPT="$SCRIPT_DIR/../release/week9-release-readiness.json"
R_BACKUP_DIR="/home/miner/sn56-endpoint-backups"

# Hard abort: Monday 2026-08-24 12:30:00 UTC. Precomputed to avoid BSD/GNU
# `date -d` divergence between macOS control box and Linux host.
HARD_ABORT_EPOCH=1787574600
HARD_ABORT_HUMAN="2026-08-24 12:30:00 UTC"

EVIDENCE_DIR="${SN56_EVIDENCE_DIR:-/Users/atulyashetty/Test/SN56-project}"

# ------------------------------ arg parsing ---------------------------------
MANIFEST="$DEFAULT_MANIFEST"
DOCKER_POLICY="$DEFAULT_DOCKER_POLICY"
READINESS_RECEIPT="$DEFAULT_READINESS_RECEIPT"
MODE="repoint"
DRY_RUN=0
CONTRACT_ONLY=0
MOCK=0
MOCKROOT=""
MOCK_REPOSITORY_URL=""
MOCK_REVIEWED_WORKTREE=""
MOCK_TARGET_REF=""
MOCK_ROLLBACK_REF=""
RECEIPT_OUT=""
ASSUME_YES=0
SYNC_TIMEOUT=420
SKIP_EXT_PROBE=0

usage() { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --rollback)      MODE="rollback" ;;
    --dry-run)       DRY_RUN=1 ;;
    --contract-only) CONTRACT_ONLY=1 ;;
    --manifest)      MANIFEST="${2:-}"; shift ;;
    --docker-policy) DOCKER_POLICY="${2:-}"; shift ;;
    --receipt)       RECEIPT_OUT="${2:-}"; shift ;;
    --mock)          MOCK=1; MOCKROOT="${2:-}"; shift ;;
    --repository-url) MOCK_REPOSITORY_URL="${2:-}"; shift ;;
    --reviewed-worktree) MOCK_REVIEWED_WORKTREE="${2:-}"; shift ;;
    --target-ref)    MOCK_TARGET_REF="${2:-}"; shift ;;
    --rollback-ref)  MOCK_ROLLBACK_REF="${2:-}"; shift ;;
    --readiness-receipt) READINESS_RECEIPT="${2:-}"; shift ;;
    --yes|-y)        ASSUME_YES=1 ;;
    --sync-timeout)  SYNC_TIMEOUT="${2:-}"; shift ;;
    --no-ext-probe)  SKIP_EXT_PROBE=1 ;;
    -h|--help)       usage ;;
    -*)              echo "unknown option: $1" >&2; exit 2 ;;
    *)               echo "FATAL: positional target SHAs are forbidden; use --manifest" >&2; exit 2 ;;
  esac
  shift
done

[ -n "$MANIFEST" ] || { echo "FATAL: --manifest requires a path" >&2; exit 2; }
[ -f "$CONTRACT" ] || { echo "FATAL: contract validator missing: $CONTRACT" >&2; exit 2; }
if [ "$MOCK" = "1" ] && [ -z "$MOCKROOT" ]; then
  echo "FATAL: --mock requires a directory argument" >&2; exit 2
fi
if [ "$MOCK" = "1" ] && { [ -z "$MOCK_REPOSITORY_URL" ] || [ -z "$MOCK_REVIEWED_WORKTREE" ]; }; then
  echo "FATAL: --mock requires --repository-url and --reviewed-worktree" >&2; exit 2
fi
if [ "$MOCK" != "1" ] && { [ -n "$MOCK_REPOSITORY_URL" ] || [ -n "$MOCK_REVIEWED_WORKTREE" ] || [ -n "$MOCK_TARGET_REF" ] || [ -n "$MOCK_ROLLBACK_REF" ]; }; then
  echo "FATAL: repository/ref/worktree overrides require explicit --mock" >&2; exit 2
fi
if [ "$ASSUME_YES" = "1" ] && [ "$MOCK" != "1" ]; then
  echo "FATAL: --yes is a mock-only test hook; live mutation always requires interactive YES" >&2; exit 2
fi
[ -f "$DOCKER_POLICY" ] || { echo "FATAL: Docker policy missing: $DOCKER_POLICY" >&2; exit 2; }
[ -f "$READINESS_RECEIPT" ] || { echo "FATAL: readiness receipt missing: $READINESS_RECEIPT" >&2; exit 2; }

printf -v ROLLBACK_COMMAND '%q ' \
  "$0" \
  --manifest "$MANIFEST" \
  --docker-policy "$DOCKER_POLICY" \
  --readiness-receipt "$READINESS_RECEIPT" \
  --rollback
ROLLBACK_COMMAND="${ROLLBACK_COMMAND% }"
ROLLBACK_COMMAND_JSON="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$ROLLBACK_COMMAND")" || exit 2

# ------------------------------ plumbing ------------------------------------
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
NOW_EPOCH="$(date -u +%s)"
RUNDIR="$(mktemp -d "${TMPDIR:-/tmp}/sn56-repoint.XXXXXX")"
trap 'rm -rf "$RUNDIR"' EXIT
CONTRACT_RECHECK_TIMEOUT=90

snapshot_reviewed_file() {
  local source="$1" destination="$2"
  python3 - "$source" "$destination" <<'PY'
import os, pathlib, sys
source, destination = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
if source.is_symlink():
    raise SystemExit(f"reviewed release artifact must not be a symlink: {source}")
raw = source.read_bytes()
with destination.open("xb") as handle:
    handle.write(raw)
    handle.flush()
    os.fsync(handle.fileno())
signature_source = pathlib.Path(f"{source}.sig")
if signature_source.exists() or signature_source.is_symlink():
    if signature_source.is_symlink():
        raise SystemExit(f"reviewed detached signature must not be a symlink: {signature_source}")
    signature_destination = pathlib.Path(f"{destination}.sig")
    signature_raw = signature_source.read_bytes()
    with signature_destination.open("xb") as handle:
        handle.write(signature_raw)
        handle.flush()
        os.fsync(handle.fileno())
PY
}

run_bounded_contract() {
  local timeout_seconds="$1"
  shift
  python3 - "$timeout_seconds" "$CONTRACT" "$@" <<'PY'
import os, signal, subprocess, sys
timeout_seconds, contract, *arguments = sys.argv[1:]
process = subprocess.Popen(
    [sys.executable, contract, *arguments],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    start_new_session=True,
)
try:
    stdout, stderr = process.communicate(timeout=int(timeout_seconds))
except subprocess.TimeoutExpired:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        stdout, stderr = process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    sys.stdout.buffer.write(stdout)
    sys.stderr.buffer.write(stderr)
    print(f"CONTRACT FAIL: bounded revalidation exceeded {timeout_seconds}s", file=sys.stderr)
    raise SystemExit(124)
sys.stdout.buffer.write(stdout)
sys.stderr.buffer.write(stderr)
raise SystemExit(process.returncode)
PY
}

validate_forward_receipt_bounded() {
  local destination="$1"
  local args=(--manifest "$MANIFEST" --docker-policy "$DOCKER_POLICY" --contract-only --receipt "$destination")
  if [ "$MOCK" = "1" ]; then
    args+=(--mock --repository-url "$MOCK_REPOSITORY_URL" --reviewed-worktree "$MOCK_REVIEWED_WORKTREE")
    [ -n "$MOCK_TARGET_REF" ] && args+=(--target-ref "$MOCK_TARGET_REF")
    [ -n "$MOCK_ROLLBACK_REF" ] && args+=(--rollback-ref "$MOCK_ROLLBACK_REF")
  fi
  run_bounded_contract "$CONTRACT_RECHECK_TIMEOUT" "${args[@]}"
}

compare_contract_receipts() {
  python3 - "$1" "$2" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    first = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    second = json.load(handle)
for receipt in (first, second):
    receipt.pop("verified_at_utc", None)
if first != second:
    changed = sorted(key for key in set(first) | set(second) if first.get(key) != second.get(key))
    print(f"contract receipt identity changed across validation boundary: {changed}", file=sys.stderr)
    raise SystemExit(1)
PY
}

validate_ready_snapshot() {
  local args=(
    --readiness-only
    --readiness-receipt "$1"
    --validated-contract-receipt "$2"
  )
  [ "$MOCK" = "1" ] && args+=(--mock)
  python3 "$CONTRACT" "${args[@]}"
}

CONTRACT_RECEIPT="$RUNDIR/contract-receipt.json"
if [ "$MODE" = "rollback" ]; then
  CONTRACT_ARGS=(
    --manifest "$MANIFEST"
    --docker-policy "$DOCKER_POLICY"
    --rollback-contract-only
    --readiness-receipt "$READINESS_RECEIPT"
    --receipt "$CONTRACT_RECEIPT"
  )
  [ "$MOCK" = "1" ] && CONTRACT_ARGS+=(--mock)
else
  CONTRACT_ARGS=(--manifest "$MANIFEST" --docker-policy "$DOCKER_POLICY" --contract-only --receipt "$CONTRACT_RECEIPT")
  if [ "$MOCK" = "1" ]; then
    CONTRACT_ARGS+=(--mock --repository-url "$MOCK_REPOSITORY_URL" --reviewed-worktree "$MOCK_REVIEWED_WORKTREE")
    [ -n "$MOCK_TARGET_REF" ] && CONTRACT_ARGS+=(--target-ref "$MOCK_TARGET_REF")
    [ -n "$MOCK_ROLLBACK_REF" ] && CONTRACT_ARGS+=(--rollback-ref "$MOCK_ROLLBACK_REF")
  fi
fi
if ! python3 "$CONTRACT" "${CONTRACT_ARGS[@]}"; then
  echo "FATAL: release contract validation failed; nothing touched" >&2
  exit 4
fi
if [ -n "$RECEIPT_OUT" ]; then
  if ! python3 - "$CONTRACT_RECEIPT" "$RECEIPT_OUT" <<'PY'
import os, pathlib, sys
source, destination = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).resolve()
destination.parent.mkdir(parents=True, exist_ok=True)
try:
    with destination.open('xb') as out:
        out.write(source.read_bytes())
        out.flush()
        os.fsync(out.fileno())
except FileExistsError:
    print(f'FATAL: receipt already exists; refusing to overwrite: {destination}', file=sys.stderr)
    raise SystemExit(1)
PY
  then
    exit 5
  fi
fi

# Re-sample the clock at each mutation boundary. Test clocks exist only behind
# explicit --mock and are phase-specific so a before/after-cutoff transition is
# deterministic without weakening the live wall-clock check.
sample_phase_epoch() {
  local phase="$1" fake=""
  if [ "$MOCK" = "1" ]; then
    case "$phase" in
      pre-backup) fake="${SN56_FAKE_PREBACKUP_EPOCH:-}" ;;
      pre-apply)  fake="${SN56_FAKE_PREAPPLY_EPOCH:-}" ;;
    esac
  fi
  if [ -n "$fake" ]; then
    case "$fake" in *[!0-9]*) echo "invalid mock epoch for $phase" >&2; return 1 ;; esac
    printf '%s\n' "$fake"
  else
    date -u +%s
  fi
}

guard_forward_mutation_window() {
  local phase="$1" phase_now
  [ "$MODE" = "repoint" ] || return 0
  phase_now="$(sample_phase_epoch "$phase")" || {
    bad "could not sample a valid UTC epoch at $phase"; exit 5;
  }
  if [ "$phase_now" -ge "$HARD_ABORT_EPOCH" ]; then
    bad "UTC epoch $phase_now reached/passed the hard abort $HARD_ABORT_HUMAN at $phase."
    bad "REFUSING before mutation; the endpoint source is untouched."
    exit 3
  fi
  ok "$phase cutoff re-check passed with $((HARD_ABORT_EPOCH - phase_now))s remaining"
}

# Values become executable configuration only from the private receipt emitted
# from the validator's in-memory, already-verified manifest. Re-reading the
# manifest here would create a validate/swap/use race.
eval "$(python3 - "$CONTRACT_RECEIPT" <<'PY'
import json, shlex, sys
d=json.load(open(sys.argv[1], encoding='utf-8'))
values={
    'RELEASE_STATE': d['release_state'],
    'RELEASE_SHA': d['target']['commit'],
    'RELEASE_TREE': d['target']['tree'],
    'RELEASE_REF': d['target']['ref'],
    'TARGET_DIGEST': d['target']['tree_records_sha256'],
    'ROLLBACK_SHA': d['rollback']['commit'],
    'ROLLBACK_REF': d['rollback']['ref'],
    'ROLLBACK_DIGEST': d['rollback']['tree_records_sha256'],
    'REPO_URL': d['source']['repository_url'],
    'SSH_HOST': d['production']['ssh_host'],
    'SERVICE': d['production']['service'],
    'SERVICE_USER': d['production']['service_user'],
    'SERVICE_WORKING_DIRECTORY': d['production']['service_working_directory'],
    'SERVICE_EXEC_START': d['production']['service_exec_start'],
    'EXT_IP': d['production']['endpoint_host'],
    'PORT': str(d['production']['endpoint_port']),
    'ENDPOINT_ROUTE': d['production']['endpoint_route'],
    'R_FILE': d['production']['endpoint_source'],
    'R_PYC': d['production']['endpoint_pyc'],
    'TEXT_PIN_EXPECTED': d['production']['text_pin'],
    'ALLOWED_CHANGES_DIGEST': d['allowed_changes']['name_status_sha256'],
    'MANIFEST_SHA256': d['manifest_sha256'],
    'DOCKER_0_PATH': d['dockerfiles'][0]['path'],
    'DOCKER_0_SHA256': d['dockerfiles'][0]['sha256'],
    'DOCKER_1_PATH': d['dockerfiles'][1]['path'],
    'DOCKER_1_SHA256': d['dockerfiles'][1]['sha256'],
    'DOCKER_POLICY_SHA256': d['docker_policy']['sha256'],
    'MANIFEST_SIGNATURE_STATE': d['manifest_signature']['state'],
    'MANIFEST_SIGNATURE_SHA256': d['manifest_signature'].get('signature_sha256', 'mock-or-hold'),
}
for key,value in values.items():
    print(f'{key}={shlex.quote(value)}')
PY
)"

TARGET_SHA="$RELEASE_SHA"
[ "$MODE" = "rollback" ] && TARGET_SHA="$ROLLBACK_SHA"

if [ "$CONTRACT_ONLY" = "1" ]; then
  printf 'CONTRACT-ONLY COMPLETE — manifest state=%s; nothing was modified.\n' "$RELEASE_STATE"
  exit 0
fi
if [ "$DRY_RUN" != "1" ] && [ "$MODE" = "repoint" ]; then
  if [ "$RELEASE_STATE" != "ready" ]; then
    echo "FATAL: manifest release_state is '$RELEASE_STATE', not 'ready'; mutation forbidden" >&2
    echo "Validation success is not release authorization. Nothing was touched." >&2
    exit 5
  fi
  INITIAL_READINESS_SNAPSHOT="$RUNDIR/initial-readiness.json"
  if ! snapshot_reviewed_file "$READINESS_RECEIPT" "$INITIAL_READINESS_SNAPSHOT" \
      || ! validate_ready_snapshot "$INITIAL_READINESS_SNAPSHOT" "$CONTRACT_RECEIPT"; then
    echo "FATAL: independent reviewed readiness receipt did not bind this exact manifest; nothing touched" >&2
    exit 5
  fi
fi

C_R=$'\033[31m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_B=$'\033[1m'; C_0=$'\033[0m'
[ -t 1 ] || { C_R=""; C_G=""; C_Y=""; C_B=""; C_0=""; }

log()  { printf '%s\n' "$*"; }
step() { printf '\n%s==> %s%s\n' "$C_B" "$*" "$C_0"; }
ok()   { printf '  %sPASS%s  %s\n' "$C_G" "$C_0" "$*"; }
warn() { printf '  %sWARN%s  %s\n' "$C_Y" "$C_0" "$*"; }
bad()  { printf '  %sFAIL%s  %s\n' "$C_R" "$C_0" "$*"; }

FAILURES=0
note_fail() { bad "$*"; FAILURES=$((FAILURES+1)); }
# wall-clock in real runs; compressed in --mock so the suite is fast
nap() { if [ "${MOCK:-0}" = "1" ]; then sleep 0.02; else sleep "$1"; fi; }
mock_pause_gate() {
  local gate="$1" label="$2" gate_i=0
  : > "$gate.ready" || return 1
  log "  [mock] paused $label; waiting for gate $gate"
  while [ ! -f "$gate" ] && [ "$gate_i" -lt 500 ]; do
    sleep 0.02
    gate_i=$((gate_i+1))
  done
  [ -f "$gate" ]
}

if [ "$MOCK" = "1" ]; then
  M_FILE="$MOCKROOT/training_repo.py"
  M_PYC="$MOCKROOT/__pycache__/training_repo.cpython-312.pyc"
  M_BACKUP_DIR="$MOCKROOT/backups"
  M_STATE="$MOCKROOT/state.env"
  M_JOURNAL="$MOCKROOT/journal.log"
  MOCK_FAIL="${MOCK_FAIL:-none}"
  SYNC_TIMEOUT=20
  # Clock override, honoured ONLY in --mock. Lets the hard-abort guard be tested
  # deterministically. It is unreachable on any path that can touch the host.
  if [ -n "${SN56_FAKE_NOW_EPOCH:-}" ]; then
    NOW_EPOCH="$SN56_FAKE_NOW_EPOCH"
    echo "  [mock] clock overridden to epoch $NOW_EPOCH"
  fi
fi

# ------------------------- host operation layer -----------------------------
# Every operation that touches the miner host goes through one of these.
# In --mock they run against a local fixture; nothing else in the script knows.

h_file_sha()   { if [ "$MOCK" = 1 ]; then shasum -a 256 "$M_FILE" | cut -d' ' -f1
                 else ssh -o BatchMode=yes "$SSH_HOST" "sha256sum '$R_FILE' | cut -d' ' -f1"; fi; }
h_file_size()  { if [ "$MOCK" = 1 ]; then wc -c < "$M_FILE" | tr -d ' '
                 else ssh -o BatchMode=yes "$SSH_HOST" "stat -c %s '$R_FILE'"; fi; }
h_file_meta()  { if [ "$MOCK" = 1 ]; then stat -f '%Su:%Sg %Lp' "$M_FILE"
                 else ssh -o BatchMode=yes "$SSH_HOST" "stat -c '%U:%G %a' '$R_FILE'"; fi; }
h_cat_file()   { if [ "$MOCK" = 1 ]; then cat "$M_FILE"
                 else ssh -o BatchMode=yes "$SSH_HOST" "cat '$R_FILE'"; fi; }

# occurrences (not lines) of a literal in the source
h_count_src()  { local s="$1"
                 if [ "$MOCK" = 1 ]; then grep -o -F "$s" "$M_FILE" 2>/dev/null | wc -l | tr -d ' '
                 else ssh -o BatchMode=yes "$SSH_HOST" "grep -o -F '$s' '$R_FILE' 2>/dev/null | wc -l | tr -d ' '"; fi; }
# occurrences of a literal inside the COMPILED bytecode actually loaded
h_count_pyc()  { local s="$1"
                 if [ "$MOCK" = 1 ]; then [ -f "$M_PYC" ] || { echo NOPYC; return; }
                   grep -a -o -F "$s" "$M_PYC" 2>/dev/null | wc -l | tr -d ' '
                 else ssh -o BatchMode=yes "$SSH_HOST" "test -f '$R_PYC' || { echo NOPYC; exit 0; }; grep -a -o -F '$s' '$R_PYC' 2>/dev/null | wc -l | tr -d ' '"; fi; }

h_backup() {  # -> prints "<backup_path> <sha256>"
  local dest="$1"
  if [ "$MOCK" = 1 ]; then
    mkdir -p "$M_BACKUP_DIR"; cp -p "$M_FILE" "$M_BACKUP_DIR/$dest"
    printf '%s %s\n' "$M_BACKUP_DIR/$dest" "$(shasum -a 256 "$M_BACKUP_DIR/$dest" | cut -d' ' -f1)"
  else
    ssh -o BatchMode=yes "$SSH_HOST" "set -e
      mkdir -p '$R_BACKUP_DIR' && chmod 0700 '$R_BACKUP_DIR'
      cp -p --preserve=all '$R_FILE' '$R_BACKUP_DIR/$dest'
      printf '%s %s\n' '$R_BACKUP_DIR/$dest' \"\$(sha256sum '$R_BACKUP_DIR/$dest' | cut -d' ' -f1)\""
  fi
}

h_restore() {  # restore a backup byte-for-byte, preserving owner/mode
  local src="$1"
  if [ "$MOCK" = 1 ]; then cp -p "$src" "$M_FILE"
  else ssh -o BatchMode=yes "$SSH_HOST" "set -e
      cp -p --preserve=all '$src' '$R_FILE.restore-tmp'
      chown --reference='$src' '$R_FILE.restore-tmp' 2>/dev/null || true
      mv -f '$R_FILE.restore-tmp' '$R_FILE'
      sync"; fi
}

h_pyc_aside() {  # move the stale bytecode into the backup dir (derived artifact)
  local dest="$1"
  if [ "$MOCK" = 1 ]; then
    [ -f "$M_PYC" ] && { mkdir -p "$M_BACKUP_DIR"; mv -f "$M_PYC" "$M_BACKUP_DIR/$dest"; echo moved; } || echo absent
  else
    ssh -o BatchMode=yes "$SSH_HOST" "mkdir -p '$R_BACKUP_DIR'; if [ -f '$R_PYC' ]; then mv -f '$R_PYC' '$R_BACKUP_DIR/$dest' && echo moved; else echo absent; fi"
  fi
}

h_edit() {  # AST-precise pin swap. args: OLD NEW EXPECTED_PREIMAGE_SHA [--check]
  local old="$1" new="$2" expected_preimage="$3" chk="${4:-}"
  if [ "$MOCK" = 1 ]; then python3 -B "$RUNDIR/edit_pin.py" "$M_FILE" "$old" "$new" "$REPO_URL" "$TEXT_PIN_EXPECTED" "$expected_preimage" $chk
  else ssh -o BatchMode=yes "$SSH_HOST" "python3 -B - '$R_FILE' '$old' '$new' '$REPO_URL' '$TEXT_PIN_EXPECTED' '$expected_preimage' $chk" < "$RUNDIR/edit_pin.py"; fi
}

h_mainpid()   { if [ "$MOCK" = 1 ]; then . "$M_STATE"; echo "$MAINPID"
                else ssh -o BatchMode=yes "$SSH_HOST" "systemctl show $SERVICE -p MainPID --value"; fi; }
h_active()    { if [ "$MOCK" = 1 ]; then . "$M_STATE"; echo "$ACTIVE"
                else ssh -o BatchMode=yes "$SSH_HOST" "systemctl is-active $SERVICE"; fi; }
h_listener()  { # pid bound to PORT, or empty
                if [ "$MOCK" = 1 ]; then . "$M_STATE"; [ "$LISTENER" = "1" ] && echo "$MAINPID" || echo ""
                else ssh -o BatchMode=yes "$SSH_HOST" "ss -tlnp 2>/dev/null | awk '/:$PORT /{print}' | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1"; fi; }
h_restart()   { if [ "$MOCK" = 1 ]; then mock_restart
                else ssh -o BatchMode=yes "$SSH_HOST" "systemctl restart $SERVICE"; fi; }
h_service_identity() {
  if [ "$MOCK" = 1 ]; then
    if [ -f "$MOCKROOT/service-contract.out" ]; then
      cat "$MOCKROOT/service-contract.out"
    else
      printf '%s\n%s\n%s\n' "$SERVICE_USER" "$SERVICE_WORKING_DIRECTORY" "$SERVICE_EXEC_START"
    fi
  else
    ssh -o BatchMode=yes "$SSH_HOST" "printf '%s\\n%s\\n%s\\n' \"\$(systemctl show '$SERVICE' -p User --value)\" \"\$(systemctl show '$SERVICE' -p WorkingDirectory --value)\" \"\$(systemctl show '$SERVICE' -p ExecStart --value)\""
  fi
}
h_probe_code() { local host="$1" path="$2"
                 if [ "$MOCK" = 1 ]; then mock_probe "$path"
                 else ssh -o BatchMode=yes "$SSH_HOST" "curl -s -o /dev/null -w '%{http_code}' --max-time 10 'http://$host:$PORT$path'"; fi; }
h_probe_exchange() { local host="$1" path="$2" code
  if [ "$MOCK" = 1 ]; then
    code="$(mock_probe "$path")"
    printf '%s\n' "$code"
    if [ -f "$MOCKROOT/fiber-auth-body.json" ]; then
      cat "$MOCKROOT/fiber-auth-body.json"
    else
      printf '%s\n' '{"detail":[{"type":"missing","loc":["header","validator-hotkey"]},{"type":"missing","loc":["header","validator-hotkey"]},{"type":"missing","loc":["header","signature"]},{"type":"missing","loc":["header","miner-hotkey"]},{"type":"missing","loc":["header","nonce"]}]}'
    fi
  else
    ssh -o BatchMode=yes "$SSH_HOST" "body=\$(mktemp); code=\$(curl -sS -o \"\$body\" -w '%{http_code}' --max-time 10 'http://$host:$PORT$path' 2>/dev/null || printf 000); printf '%s\\n' \"\$code\"; cat \"\$body\"; rm -f \"\$body\""
  fi
}
h_probe_local_exchange() { local path="$1" body code
  if [ "$MOCK" = 1 ]; then
    h_probe_exchange "$EXT_IP" "$path"
  else
    body="$(mktemp "${TMPDIR:-/tmp}/sn56-repoint-body.XXXXXX")" || return 1
    code="$(curl -sS -o "$body" -w '%{http_code}' --max-time 12 "http://$EXT_IP:$PORT$path" 2>/dev/null || printf 000)"
    printf '%s\n' "$code"
    cat "$body"
    rm -f "$body"
  fi
}
h_auth_probe() { h_probe_exchange "$1" "$2" | python3 "$RUNDIR/verify_fiber_auth.py"; }
h_auth_probe_local() { h_probe_local_exchange "$1" | python3 "$RUNDIR/verify_fiber_auth.py"; }
h_sync_count() { # completed live metagraph syncs attributed to PID since EPOCH
  local pid="$1" since="$2"
  if [ "$MOCK" = 1 ]; then grep -F "uvicorn[$pid]" "$M_JOURNAL" 2>/dev/null | grep -c "Successfully synced" | tr -d ' '
  else ssh -o BatchMode=yes "$SSH_HOST" "journalctl -u $SERVICE --since '@$since' -o short-iso --no-pager 2>/dev/null | grep -F 'uvicorn[$pid]' | grep -c 'Successfully synced' | tr -d ' '"; fi
}
h_recent_log() {
  if [ "$MOCK" = 1 ]; then tail -12 "$M_JOURNAL" 2>/dev/null
  else ssh -o BatchMode=yes "$SSH_HOST" "journalctl -u $SERVICE -n 12 --no-pager -o short-iso"; fi
}

# ---------------------------- mock host model -------------------------------
# Fault model. A plain fault name (e.g. MOCK_FAIL=active) means "the change
# broke it": the fault engages on the FIRST restart only, so the rollback
# restart recovers -- which is exactly the situation auto-rollback exists for.
# A "_hard" suffix (MOCK_FAIL=active_hard) means the host is broken independent
# of our change, so the rollback restart fails too and we must escalate.
mock_engaged() {  # $1 = fault name, $2 = restart ordinal ; pure, reads no state
  case "$MOCK_FAIL" in
    "$1")        [ "${2:-0}" -le 1 ] ;;
    "${1}_hard") true ;;
    *)           false ;;
  esac
}
mock_restart() {
  local oldpid n newpid listener active
  . "$M_STATE"
  oldpid="$MAINPID"
  n=$(( ${RESTARTS:-0} + 1 ))
  newpid=$(( oldpid + 7 )); listener=1; active=active
  mock_engaged active   "$n" && { active=failed; listener=0; }
  mock_engaged listener "$n" && listener=0
  mock_engaged pid      "$n" && newpid="$oldpid"
  printf 'MAINPID=%s\nACTIVE=%s\nLISTENER=%s\nSTART_EPOCH=%s\nRESTARTS=%s\n' \
    "$newpid" "$active" "$listener" "$(date -u +%s)" "$n" > "$M_STATE"
  [ "$active" = active ] || return 0
  # simulate CPython recompiling the module from the (new) source on import
  mkdir -p "$(dirname "$M_PYC")"
  if ! mock_engaged pyc "$n"; then
    { printf '\xcb\x0d\x0d\x0a\x00\x00\x00\x00'; grep -o -E '[0-9a-f]{40}' "$M_FILE" | tr '\n' '\0'; } > "$M_PYC"
  fi
  printf '%s bittensor-ops uvicorn[%s]: INFO:     Started server process [%s]\n' "$(date -u +%Y-%m-%dT%H:%M:%S%z)" "$newpid" "$newpid" >> "$M_JOURNAL"
  if ! mock_engaged sync "$n"; then
    printf '%s bittensor-ops uvicorn[%s]: INFO | metagraph:sync_nodes:72 - Successfully synced 256 nodes!\n' "$(date -u +%Y-%m-%dT%H:%M:%S%z)" "$newpid" >> "$M_JOURNAL"
  fi
}
mock_probe() {
  . "$M_STATE"
  [ "$ACTIVE" = "active" ] || { echo 000; return; }
  # the probe fault only exists after our restart; the baseline probe is healthy
  if [ "${RESTARTS:-0}" -ge 1 ] && mock_engaged probe "${RESTARTS:-0}"; then echo 500; return; fi
  case "$1" in
    /openapi.json) echo 200 ;;
    *)
      if [ -f "$MOCKROOT/fiber-auth-code" ]; then
        sed -n '1p' "$MOCKROOT/fiber-auth-code"
      else
        echo 422
      fi
      ;;
  esac
}

cat > "$RUNDIR/verify_fiber_auth.py" <<'PYEOF'
import collections
import json
import sys

raw = sys.stdin.buffer.read()
code_raw, separator, body_raw = raw.partition(b"\n")
code = code_raw.decode("ascii", "replace").strip()
if not separator or code != "422":
    print(f"BAD: HTTP {code or 'unreadable'}, expected 422")
    raise SystemExit(1)
try:
    payload = json.loads(body_raw.decode("utf-8"))
except Exception as exc:
    print(f"BAD: malformed Fiber validation JSON: {exc}")
    raise SystemExit(1)
detail = payload.get("detail") if isinstance(payload, dict) else None
expected = {"validator-hotkey", "signature", "miner-hotkey", "nonce"}
if not isinstance(detail, list):
    print("BAD: Fiber validation detail is not an array")
    raise SystemExit(1)
counts = collections.Counter()
for item in detail:
    if not isinstance(item, dict) or item.get("type") != "missing":
        print(f"BAD: non-missing Fiber validation error: {item!r}")
        raise SystemExit(1)
    loc = item.get("loc")
    lowered = [str(part).lower() for part in loc] if isinstance(loc, list) else []
    if len(lowered) != 2 or lowered[0] != "header" or lowered[1] not in expected:
        print(f"BAD: unexpected Fiber validation location: {loc!r}")
        raise SystemExit(1)
    counts[lowered[1]] += 1
expected_once = collections.Counter({name: 1 for name in expected})
observed_live = expected_once.copy()
observed_live["validator-hotkey"] = 2
if counts != observed_live:
    print(
        "BAD: auth-header multiset "
        f"{dict(sorted(counts.items()))!r}, expected observed-live "
        f"{dict(sorted(observed_live.items()))!r}"
    )
    raise SystemExit(1)
print(f"OK: exact Fiber auth-stage response {dict(sorted(counts.items()))}")
PYEOF

# ============================================================================
# THE AST EDITOR (written out at runtime; piped to the host, never installed)
# ============================================================================
cat > "$RUNDIR/edit_pin.py" <<'PYEOF'
import ast, hashlib, json, os, sys

def die(msg, **kw):
    kw.update({"ok": False, "error": msg}); print(json.dumps(kw)); sys.exit(9)

if len(sys.argv) < 7:
    die("usage: edit_pin.py PATH OLD NEW EXPECTED_IMAGE_REPO EXPECTED_TEXT_PIN EXPECTED_PREIMAGE_SHA [--check]")
path, old_sha, new_sha, expected_image_repo, expected_text_pin, expected_preimage_sha = sys.argv[1:7]
check_only = "--check" in sys.argv[7:]
HEX = set("0123456789abcdef")
for s in (old_sha, new_sha):
    if len(s) != 40 or not set(s) <= HEX: die("sha not 40 lowercase hex: %r" % s)
if len(expected_preimage_sha) != 64 or not set(expected_preimage_sha) <= HEX:
    die("expected preimage sha256 is not 64 lowercase hex: %r" % expected_preimage_sha)
if old_sha == new_sha: die("old and new sha are identical; nothing to do")

with open(path, "rb") as fh: raw = fh.read()
actual_preimage_sha = hashlib.sha256(raw).hexdigest()
if actual_preimage_sha != expected_preimage_sha:
    die("live-file preimage sha256 is %s, expected sampled %s -- refusing" %
        (actual_preimage_sha, expected_preimage_sha),
        actual_preimage_sha256=actual_preimage_sha,
        expected_preimage_sha256=expected_preimage_sha)
try: src = raw.decode("utf-8")
except UnicodeDecodeError as e: die("file is not utf-8: %s" % e)
try: tree = ast.parse(src, filename=path)
except SyntaxError as e: die("file does not parse before edit: %s" % e)

repos = None
for node in tree.body:
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "_REPOS": repos = node.value
if not isinstance(repos, ast.Dict): die("_REPOS dict literal not found")

repo_names = []
for key in repos.keys:
    if not (
        isinstance(key, ast.Attribute)
        and isinstance(key.value, ast.Name)
        and key.value.id == "TournamentType"
    ):
        die("_REPOS contains a non-literal TournamentType key")
    repo_names.append(key.attr)
if len(repo_names) != 2 or set(repo_names) != {"IMAGE", "TEXT"}:
    die("_REPOS keys are %r, expected exactly IMAGE and TEXT in either order; ENVIRONMENT must remain absent" % repo_names)

def entry(name):
    for k, v in zip(repos.keys, repos.values):
        if (isinstance(k, ast.Attribute) and k.attr == name
                and isinstance(k.value, ast.Name) and k.value.id == "TournamentType"):
            return v
    return None

def fields(call, label):
    if not isinstance(call, ast.Call): die("%s entry is not a call" % label)
    fn = call.func
    nm = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
    if nm != "TrainingRepoResponse": die("%s entry is not TrainingRepoResponse (got %r)" % (label, nm))
    out = {}
    for kw in call.keywords:
        if kw.arg in ("commit_hash", "github_repo"):
            if not (isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)):
                die("%s.%s is not a plain string literal" % (label, kw.arg))
            out[kw.arg] = kw.value
    if "commit_hash" not in out: die("%s has no commit_hash keyword" % label)
    return out

img_e, txt_e = entry("IMAGE"), entry("TEXT")
if img_e is None: die("no TournamentType.IMAGE entry in _REPOS")
if txt_e is None: die("no TournamentType.TEXT entry in _REPOS")
img, txt = fields(img_e, "IMAGE"), fields(txt_e, "TEXT")

node = img["commit_hash"]
if node.value != old_sha:
    die("IMAGE commit_hash is %r, expected %r -- refusing" % (node.value, old_sha),
        actual_image_pin=node.value)

text_pin = txt["commit_hash"].value
img_repo = img["github_repo"].value if "github_repo" in img else None
txt_repo = txt["github_repo"].value if "github_repo" in txt else None
def canonical_git_repo_url(value):
    return value[:-4] if isinstance(value, str) and value.endswith(".git") else value
if canonical_git_repo_url(img_repo) != canonical_git_repo_url(expected_image_repo):
    die("IMAGE github_repo is %r, expected reviewed repository %r -- refusing" %
        (img_repo, expected_image_repo), actual_image_repo=img_repo,
        expected_image_repo=expected_image_repo)
if text_pin != expected_text_pin:
    die("TEXT commit_hash is %r, expected reviewed pin %r -- refusing" %
        (text_pin, expected_text_pin), actual_text_pin=text_pin,
        expected_text_pin=expected_text_pin)
if text_pin == old_sha: die("TEXT pin equals the IMAGE pin being replaced -- ambiguous, refusing")

# ---- rewrite exactly one literal, by byte offset --------------------------
lines = raw.splitlines(keepends=True)
if node.lineno != node.end_lineno: die("commit_hash literal spans multiple lines")
li = node.lineno - 1
line = lines[li]
span = line[node.col_offset:node.end_col_offset]
if span not in (b'"' + old_sha.encode() + b'"', b"'" + old_sha.encode() + b"'"):
    die("literal span mismatch at line %d: %r" % (node.lineno, span))
q = span[0:1]
lines[li] = line[:node.col_offset] + q + new_sha.encode() + q + line[node.end_col_offset:]
new_raw = b"".join(lines)

# ---- invariants -----------------------------------------------------------
if len(new_raw) != len(raw): die("edit changed file length (%d -> %d)" % (len(raw), len(new_raw)))
a, b = raw.splitlines(), new_raw.splitlines()
if len(a) != len(b): die("edit changed line count")
diffs = [i + 1 for i in range(len(a)) if a[i] != b[i]]
if diffs != [node.lineno]: die("edit touched lines %r, expected only [%d]" % (diffs, node.lineno))
if new_raw.count(old_sha.encode()) != 0: die("old sha still present after edit")
if new_raw.count(new_sha.encode()) != 1: die("new sha appears %d times, expected 1" % new_raw.count(new_sha.encode()))
if new_raw.count(text_pin.encode()) != 1: die("text pin count changed")

try: t2 = ast.parse(new_raw.decode("utf-8"), filename=path)
except SyntaxError as e: die("edited file does not parse: %s" % e)
r2 = None
for n2 in t2.body:
    if isinstance(n2, ast.Assign):
        for t in n2.targets:
            if isinstance(t, ast.Name) and t.id == "_REPOS": r2 = n2.value
def entry2(name):
    for k, v in zip(r2.keys, r2.values):
        if isinstance(k, ast.Attribute) and k.attr == name: return v
    return None
def f2(call):
    return {kw.arg: kw.value.value for kw in call.keywords
            if kw.arg in ("commit_hash", "github_repo") and isinstance(kw.value, ast.Constant)}
i2, x2 = f2(entry2("IMAGE")), f2(entry2("TEXT"))
if i2.get("commit_hash") != new_sha: die("post-parse: IMAGE pin is %r" % i2.get("commit_hash"))
if x2.get("commit_hash") != text_pin: die("post-parse: TEXT pin changed!")
if i2.get("github_repo") != img_repo: die("post-parse: IMAGE repo url changed!")
if x2.get("github_repo") != txt_repo: die("post-parse: TEXT repo url changed!")
repo_names2 = [
    key.attr
    for key in r2.keys
    if isinstance(key, ast.Attribute)
    and isinstance(key.value, ast.Name)
    and key.value.id == "TournamentType"
]
if repo_names2 != repo_names: die("post-parse: _REPOS key order changed or ENVIRONMENT appeared")

result = {
    "ok": True, "check_only": check_only, "line": node.lineno,
    "sha256_before": hashlib.sha256(raw).hexdigest(),
    "sha256_after": hashlib.sha256(new_raw).hexdigest(),
    "bytes": len(new_raw),
    "image_pin_before": old_sha, "image_pin_after": new_sha,
    "text_pin_unchanged": text_pin,
    "image_repo": img_repo, "text_repo": txt_repo,
}
if check_only:
    print(json.dumps(result)); sys.exit(0)

st = os.stat(path)
d = os.path.dirname(path) or "."
tmp = os.path.join(d, ".training_repo.py.repoint-tmp")   # not a .py: unimportable
with open(tmp, "wb") as fh:
    fh.write(new_raw); fh.flush(); os.fsync(fh.fileno())
os.chmod(tmp, st.st_mode & 0o7777)
try: os.chown(tmp, st.st_uid, st.st_gid)
except (PermissionError, AttributeError): pass
os.replace(tmp, path)
try:
    dfd = os.open(d, os.O_RDONLY); os.fsync(dfd); os.close(dfd)
except OSError: pass
with open(path, "rb") as fh: back = fh.read()
if hashlib.sha256(back).hexdigest() != result["sha256_after"]:
    die("readback mismatch after write")
st2 = os.stat(path)
result["owner_after"] = "%d:%d" % (st2.st_uid, st2.st_gid)
result["mode_after"] = oct(st2.st_mode & 0o7777)
print(json.dumps(result))
PYEOF

if [ "$MODE" = "repoint" ] && [ "$MOCK" = "1" ] \
    && [ -n "${SN56_MOCK_AFTER_INITIAL_CONTRACT_GATE:-}" ]; then
  mock_pause_gate "$SN56_MOCK_AFTER_INITIAL_CONTRACT_GATE" "after initial contract validation" \
    || { bad "mock after-initial-contract gate timed out"; exit 5; }
fi

# ============================================================================
# BANNER
# ============================================================================
log "${C_B}SN56 manifest-bound IMAGE-pin repoint${C_0}"
log "  mode          : $MODE$([ $DRY_RUN = 1 ] && echo ' (DRY RUN - read only)')$([ $MOCK = 1 ] && echo " (MOCK -> $MOCKROOT)")"
log "  manifest      : $MANIFEST"
log "  release state : $RELEASE_STATE"
log "  target sha    : $TARGET_SHA"
log "  rollback sha  : $ROLLBACK_SHA"
log "  host / file   : $([ $MOCK = 1 ] && echo "$M_FILE" || echo "$SSH_HOST:$R_FILE")"
log "  now (UTC)     : $(date -u +'%Y-%m-%d %H:%M:%S')"
log "  hard abort    : $HARD_ABORT_HUMAN"

# ============================================================================
# GUARD 0 - hard abort window
# ============================================================================
step "GUARD 0  hard-abort window"
if [ "$NOW_EPOCH" -ge "$HARD_ABORT_EPOCH" ]; then
  if [ "$MODE" = "rollback" ]; then
    warn "past $HARD_ABORT_HUMAN -- forward repoint is barred, but ROLLBACK is always permitted."
    warn "rollback restores the known-good resting state; refusing it would be the unsafe choice."
  else
    bad "current UTC time is past the hard abort $HARD_ABORT_HUMAN."
    bad "a forward repoint this close to (or after) the 13:00 snapshot is unrecoverable."
    bad "REFUSING. If the miner is on the new pin and you want out, run:  $ROLLBACK_COMMAND"
    exit 3
  fi
else
  ok "$(( (HARD_ABORT_EPOCH - NOW_EPOCH) / 3600 ))h $(( ((HARD_ABORT_EPOCH - NOW_EPOCH) % 3600) / 60 ))m of runway before the hard abort"
fi

# ============================================================================
# GATE 1 - COMPLETE MANIFEST CONTRACT
# ============================================================================
step "GATE 1  complete release contract (already re-proved, fail closed)"
if [ "$MODE" = "rollback" ]; then
  ok "canonical READY manifest is bound by the independent readiness receipt"
  ok "emergency rollback identity is offline; forward ref/worktree gates are intentionally not prerequisites"
else
  ok "reviewed local HEAD/tree is exact and clean"
  ok "anonymous target and rollback refs resolve exactly"
fi
ok "target tree-record digest $TARGET_DIGEST"
ok "rollback tree-record digest $ROLLBACK_DIGEST"
ok "rollback-to-target surface digest $ALLOWED_CHANGES_DIGEST"
if [ "$MANIFEST_SIGNATURE_STATE" = "verified" ]; then
  ok "detached final-manifest signature $MANIFEST_SIGNATURE_SHA256 is verified"
else
  warn "manifest signature state is $MANIFEST_SIGNATURE_STATE (live mutation cannot use this state)"
fi
if [ "$MODE" = "rollback" ]; then
  ok "target Docker identities are bound to the immutable bd852dc policy"
else
  ok "both target Dockerfile byte hashes match the reviewed manifest and certification source"
fi
[ "$RELEASE_STATE" = "ready" ] \
  && ok "manifest state is ready (operator authorization is still separate)" \
  || warn "manifest state is $RELEASE_STATE -- NON-SHIPPABLE; only contract/dry-run is allowed"

# ============================================================================
# PREFLIGHT - the host must be exactly where we think it is
# ============================================================================
step "PREFLIGHT  live host state"
if [ "$MOCK" = 1 ]; then
  [ -f "$M_FILE" ] || { bad "mock fixture missing: $M_FILE"; exit 5; }
  [ -f "$M_STATE" ] || printf 'MAINPID=100000\nACTIVE=active\nLISTENER=1\nSTART_EPOCH=%s\nRESTARTS=0\n' "$NOW_EPOCH" > "$M_STATE"
  touch "$M_JOURNAL"
else
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$SSH_HOST" true 2>/dev/null || { bad "cannot ssh to $SSH_HOST"; exit 5; }
  ok "ssh to $SSH_HOST"
fi

SERVICE_IDENTITY="$(h_service_identity)"
SERVICE_VERDICT="$(SN56_SERVICE_IDENTITY="$SERVICE_IDENTITY" python3 - \
  "$SERVICE_USER" "$SERVICE_WORKING_DIRECTORY" "$SERVICE_EXEC_START" <<'PY'
import os
import re
import sys

expected_user, expected_working_directory, expected_exec = sys.argv[1:]
lines = os.environ.get("SN56_SERVICE_IDENTITY", "").splitlines()
if len(lines) != 3:
    print(f"BAD: service identity returned {len(lines)} fields, expected 3")
    raise SystemExit
actual_user, actual_working_directory, exec_show = lines
match = re.search(r"(?:^|;\s*)argv\[\]=(.*?)\s*;", exec_show)
actual_exec = match.group(1).strip() if match else exec_show
problems = []
if actual_user != expected_user:
    problems.append(f"User={actual_user!r}, expected {expected_user!r}")
if actual_working_directory != expected_working_directory:
    problems.append(
        f"WorkingDirectory={actual_working_directory!r}, expected {expected_working_directory!r}"
    )
if actual_exec != expected_exec:
    problems.append(f"ExecStart argv={actual_exec!r}, expected {expected_exec!r}")
print("BAD: " + "; ".join(problems) if problems else "OK: exact service user/cwd/ExecStart")
PY
)"
case "$SERVICE_VERDICT" in
  OK:*) ok "${SERVICE_VERDICT#OK: }" ;;
  *) bad "$SERVICE_VERDICT"; exit 5 ;;
esac

PRE_SHA="$(h_file_sha)"; PRE_SIZE="$(h_file_size)"; PRE_META="$(h_file_meta)"
log "  file sha256   : $PRE_SHA"
log "  file size     : $PRE_SIZE bytes   owner/mode: $PRE_META"

CUR_IMAGE="$(h_cat_file | python3 -c '
import ast,sys
t=ast.parse(sys.stdin.read())
for n in t.body:
    if isinstance(n,ast.Assign) and any(getattr(x,"id","")=="_REPOS" for x in n.targets):
        for k,v in zip(n.value.keys,n.value.values):
            if getattr(k,"attr","")=="IMAGE":
                for kw in v.keywords:
                    if kw.arg=="commit_hash": print(kw.value.value)
' 2>/dev/null)"
CUR_TEXT="$(h_count_src "$TEXT_PIN_EXPECTED")"
[ -n "$CUR_IMAGE" ] || { bad "could not read the current IMAGE pin from the live file"; exit 5; }
log "  current IMAGE : $CUR_IMAGE"

if [ "$CUR_TEXT" != "1" ]; then bad "expected the TEXT pin $TEXT_PIN_EXPECTED exactly once, found $CUR_TEXT"; exit 5; fi
ok "TEXT tournament pin present exactly once (will not be touched)"

ROLLBACK_NOOP=0
if [ "$MODE" = "repoint" ]; then
  if [ "$CUR_IMAGE" != "$ROLLBACK_SHA" ]; then
    bad "forward preflight requires the exact rollback pin $ROLLBACK_SHA"
    bad "served IMAGE pin is $CUR_IMAGE -- refusing a stale/wrong starting state"
    exit 5
  fi
else
  if [ "$CUR_IMAGE" = "$ROLLBACK_SHA" ]; then
    ROLLBACK_NOOP=1
  elif [ "$CUR_IMAGE" != "$RELEASE_SHA" ]; then
    bad "rollback preflight expected served release $RELEASE_SHA"
    bad "served IMAGE pin is $CUR_IMAGE -- refusing an unknown starting state"
    exit 5
  fi
fi

PYC_CURRENT="$(h_count_pyc "$CUR_IMAGE")"
PYC_TEXT="$(h_count_pyc "$TEXT_PIN_EXPECTED")"
PYC_OTHER_SHA="$RELEASE_SHA"
[ "$CUR_IMAGE" = "$RELEASE_SHA" ] && PYC_OTHER_SHA="$ROLLBACK_SHA"
PYC_OTHER="$(h_count_pyc "$PYC_OTHER_SHA")"
if [ "$PYC_CURRENT" != "1" ] || [ "$PYC_TEXT" != "1" ] || [ "$PYC_OTHER" != "0" ]; then
  bad "running bytecode prestate differs: current $CUR_IMAGE x$PYC_CURRENT, other $PYC_OTHER_SHA x$PYC_OTHER, TEXT x$PYC_TEXT"
  exit 5
fi
ok "running bytecode matches source prestate: IMAGE x1, alternate pin x0, TEXT x1"
if [ "$ROLLBACK_NOOP" = "1" ]; then
  ok "IMAGE source and bytecode are already the exact rollback $ROLLBACK_SHA"
else
  ok "IMAGE pin will change: $CUR_IMAGE  ->  $TARGET_SHA"
fi

ACT0="$(h_active)"; PID0="$(h_mainpid)"; LSN0="$(h_listener)"
log "  service       : $ACT0  MainPID=$PID0  listener_pid=${LSN0:-none}"
[ "$ACT0" = "active" ] || { bad "$SERVICE is not active before we start ($ACT0) -- fix that first"; exit 5; }
[ -n "$LSN0" ] || { bad "nothing is listening on :$PORT before we start"; exit 5; }
P0="$(h_auth_probe 127.0.0.1 "$ENDPOINT_ROUTE")" || {
  bad "baseline route did not reach the exact Fiber auth stage: $P0"; exit 5;
}
ok "baseline: active, listening, exact Fiber auth-stage route"

step "PREFLIGHT  read-only source semantic/AST contract"
CHECK_TARGET_SHA="$TARGET_SHA"
[ "$ROLLBACK_NOOP" = "1" ] && CHECK_TARGET_SHA="$RELEASE_SHA"
CHECK_JSON="$(h_edit "$CUR_IMAGE" "$CHECK_TARGET_SHA" "$PRE_SHA" --check)" || { bad "source semantic/AST pre-check refused:"; log "      $CHECK_JSON"; exit 5; }
log "  $CHECK_JSON"
printf '%s' "$CHECK_JSON" | grep -q '"ok": *true' || { bad "edit pre-check did not report ok"; exit 5; }
ok "AST proves exact IMAGE/TEXT mapping, reviewed IMAGE repository, and pins with ENVIRONMENT absent"
if [ "$ROLLBACK_NOOP" = "1" ]; then
  log ""; log "${C_G}NO-OP: active service, exact source contract, running bytecode, and Fiber route already serve the rollback pin.${C_0}"
  exit 0
fi

if [ "$DRY_RUN" = "1" ]; then
  log ""; log "${C_G}DRY RUN COMPLETE -- nothing was modified.${C_0}"
  log "Re-run without --dry-run to apply."
  exit 0
fi

# ============================================================================
# CONFIRM
# ============================================================================
if [ "$ASSUME_YES" != "1" ] && [ "$MOCK" != "1" ]; then
  PROMPT_NOW_EPOCH="$(date -u +%s)" || { bad "could not sample UTC before confirmation"; exit 5; }
  PROMPT_TIMEOUT=$((HARD_ABORT_EPOCH - PROMPT_NOW_EPOCH))
  if [ "$PROMPT_TIMEOUT" -le 0 ]; then
    bad "hard abort $HARD_ABORT_HUMAN arrived before confirmation; refusing"
    exit 3
  fi
  log ""
  log "${C_Y}About to MUTATE PRODUCTION:${C_0}"
  log "  $SSH_HOST:$R_FILE   IMAGE pin  $CUR_IMAGE -> $TARGET_SHA"
  log "  then: systemctl restart $SERVICE   (endpoint down ~1-2s)"
  log "  auto-rollback is armed; hard abort is $HARD_ABORT_HUMAN"
  printf '\nType exactly YES to proceed (prompt expires at hard abort): '
  if ! read -r -t "$PROMPT_TIMEOUT" ANS < /dev/tty; then
    bad "confirmation timed out or input closed; nothing changed"
    exit 3
  fi
  [ "$ANS" = "YES" ] || { log "aborted by operator; nothing changed."; exit 2; }
fi

if [ "$MODE" = "repoint" ]; then
  step "GATE 2  post-confirmation contract and READY receipt revalidation"
  PREMUTATION_CONTRACT_RECEIPT="$RUNDIR/pre-mutation-contract.json"
  PREMUTATION_READINESS_SNAPSHOT="$RUNDIR/pre-mutation-readiness.json"
  if ! validate_forward_receipt_bounded "$PREMUTATION_CONTRACT_RECEIPT"; then
    bad "post-confirmation full release contract failed; refusing before backup/apply"
    exit 5
  fi
  if ! compare_contract_receipts "$CONTRACT_RECEIPT" "$PREMUTATION_CONTRACT_RECEIPT"; then
    bad "post-confirmation contract identity differs from the private initial snapshot"
    exit 5
  fi
  if ! snapshot_reviewed_file "$READINESS_RECEIPT" "$PREMUTATION_READINESS_SNAPSHOT" \
      || ! validate_ready_snapshot "$PREMUTATION_READINESS_SNAPSHOT" "$PREMUTATION_CONTRACT_RECEIPT"; then
    bad "post-confirmation READY receipt validation failed; refusing before backup/apply"
    exit 5
  fi
  ok "full contract and exact READY authorization re-proved after confirmation"
fi

# ============================================================================
# BACKUP
# ============================================================================
guard_forward_mutation_window "pre-backup"
step "BACKUP"
BK_NAME="training_repo.py.$STAMP.pre-$TARGET_SHA.bak"
BK_LINE="$(h_backup "$BK_NAME")" || { bad "backup failed"; exit 5; }
BK_PATH="${BK_LINE%% *}"; BK_SHA="${BK_LINE##* }"
[ "$BK_SHA" = "$PRE_SHA" ] || { bad "backup sha256 $BK_SHA != live file sha256 $PRE_SHA"; exit 5; }
ok "backup $BK_PATH"
ok "backup sha256 == live file sha256 ($BK_SHA)"
if [ "$MOCK" = "1" ] && [ -n "${SN56_MOCK_AFTER_BACKUP_GATE:-}" ]; then
  MOCK_GATE="$SN56_MOCK_AFTER_BACKUP_GATE"
  if ! mock_pause_gate "$MOCK_GATE" "after backup"; then
    bad "mock after-backup gate timed out; restoring the untouched preimage"
    h_restore "$BK_PATH" >/dev/null 2>&1 || true
    exit 5
  fi
fi

# ============================================================================
# ROLLBACK MACHINERY (armed from here on)
# ============================================================================
ROLLED_BACK=0
ROLLBACK_ARMED=0
ROLLBACK_IN_PROGRESS=0
SIGNAL_HANDLER_ACTIVE=0
do_rollback() {
  local why="$1"
  if [ "$ROLLBACK_IN_PROGRESS" = "1" ]; then
    bad "rollback is already in progress; refusing recursive recovery"
    return 1
  fi
  ROLLBACK_IN_PROGRESS=1
  step "${C_R}AUTO-ROLLBACK${C_0}  reason: $why"
  h_restore "$BK_PATH" || { bad "restore FAILED -- MANUAL INTERVENTION REQUIRED"; return 1; }
  local s; s="$(h_file_sha)"
  [ "$s" = "$PRE_SHA" ] || { bad "restored file sha256 $s != original $PRE_SHA -- MANUAL INTERVENTION REQUIRED"; return 1; }
  ok "file restored byte-for-byte ($s)"
  h_pyc_aside "training_repo.cpython-312.pyc.$STAMP.rollback" >/dev/null 2>&1 || true
  h_restart || { bad "restart during rollback FAILED"; return 1; }
  local i=0 act pid
  while [ $i -lt 60 ]; do
    act="$(h_active)"; pid="$(h_mainpid)"
    [ "$act" = "active" ] && [ -n "$pid" ] && [ "$pid" != "0" ] && break
    i=$((i+1)); nap 1
  done
  [ "$act" = "active" ] || { bad "service not active after rollback restart ($act)"; return 1; }
  ok "service active again, MainPID=$pid"
  local n o p
  n="$(h_count_src "$TARGET_SHA")"; o="$(h_count_src "$CUR_IMAGE")"
  [ "$n" = "0" ] && [ "$o" = "1" ] || { bad "post-rollback source pins wrong (target x$n, original x$o)"; return 1; }
  ok "source serves the ORIGINAL pin $CUR_IMAGE again"
  local pc pc_target pc_text
  pc="$(h_count_pyc "$CUR_IMAGE")"
  pc_target="$(h_count_pyc "$TARGET_SHA")"
  pc_text="$(h_count_pyc "$TEXT_PIN_EXPECTED")"
  [ "$pc" = "1" ] && [ "$pc_target" = "0" ] && [ "$pc_text" = "1" ] || {
    bad "post-rollback running bytecode differs (original x$pc, target x$pc_target, TEXT x$pc_text)"; return 1;
  }
  ok "compiled bytecode carries only the original IMAGE pin and unchanged TEXT pin"
  p="$(h_auth_probe 127.0.0.1 "$ENDPOINT_ROUTE")" || {
    bad "post-rollback route did not reach the exact Fiber auth stage: $p"; return 1;
  }
  ok "endpoint reaches exact Fiber auth stage on the ORIGINAL pin"
  ROLLED_BACK=1
  ROLLBACK_ARMED=0
  ROLLBACK_IN_PROGRESS=0
  return 0
}

handle_armed_signal() {
  local signal_name="$1" exit_code="$2"
  # Ignore further termination signals while attempting recovery. This avoids
  # recursive traps corrupting a restore/restart already in flight.
  trap '' HUP INT TERM
  if [ "$SIGNAL_HANDLER_ACTIVE" = "1" ] || [ "$ROLLBACK_IN_PROGRESS" = "1" ]; then
    bad "$signal_name arrived while rollback was already in progress"
    bad "automatic recovery cannot safely recurse; manual intervention is required"
    exit 7
  fi
  SIGNAL_HANDLER_ACTIVE=1
  if [ "$MODE" = "repoint" ] && [ "$ROLLBACK_ARMED" = "1" ]; then
    warn "$signal_name received with forward rollback armed; restoring the backup"
    if do_rollback "received $signal_name"; then
      log ""
      log "${C_Y}SIGNAL RECOVERY COMPLETE.${C_0} Original pin $CUR_IMAGE is verified healthy."
      exit "$exit_code"
    fi
    bad "signal-triggered rollback FAILED -- MANUAL INTERVENTION REQUIRED"
    exit 7
  fi
  exit "$exit_code"
}
die_rollback() {
  local why="$1"
  if do_rollback "$why"; then
    log ""; log "${C_Y}ROLLED BACK.${C_0} The miner is serving $CUR_IMAGE again. Nothing is lost."
    log "Investigate, then re-run. Hard abort remains $HARD_ABORT_HUMAN."
    exit 6
  else
    log ""; log "${C_R}ROLLBACK FAILED -- THE MINER MAY BE IN A BAD STATE.${C_0}"
    log "Manual recovery:"
    log "  ssh $SSH_HOST 'cp -p $BK_PATH $R_FILE && systemctl restart $SERVICE'"
    log "  ssh $SSH_HOST 'systemctl status $SERVICE; curl -s -o /dev/null -w \"%{http_code}\\n\" http://127.0.0.1:$PORT$ENDPOINT_ROUTE'"
    exit 7
  fi
}

if [ "$MODE" = "repoint" ]; then
  ROLLBACK_ARMED=1
  trap 'handle_armed_signal HUP 129' HUP
  trap 'handle_armed_signal INT 130' INT
  trap 'handle_armed_signal TERM 143' TERM
  ok "forward auto-rollback armed for HUP/INT/TERM"
fi

# ============================================================================
# APPLY
# ============================================================================
guard_forward_mutation_window "pre-apply"
step "APPLY  minimal edit (IMAGE pin only)"
EDIT_JSON="$(h_edit "$CUR_IMAGE" "$TARGET_SHA" "$PRE_SHA")" || die_rollback "editor refused / failed: $EDIT_JSON"
log "  $EDIT_JSON"
printf '%s' "$EDIT_JSON" | grep -q '"ok": *true' || die_rollback "editor did not report ok"
EXPECTED_POST_SHA="$(printf '%s' "$EDIT_JSON" | python3 -c '
import json, sys
value = json.load(sys.stdin).get("sha256_after")
if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
    raise SystemExit(1)
print(value)
')" || die_rollback "editor result omitted a valid expected sha256_after"
if [ "$MOCK" = "1" ] && [ "${SN56_MOCK_SIGNAL_AT:-}" = "after-apply" ]; then
  log "  [mock] injecting TERM after APPLY"
  kill -TERM "$$"
fi
if [ "$MOCK" = "1" ] && [ -n "${SN56_MOCK_AFTER_EDIT_GATE:-}" ]; then
  MOCK_GATE="$SN56_MOCK_AFTER_EDIT_GATE"
  mock_pause_gate "$MOCK_GATE" "after editor return" \
    || die_rollback "mock after-editor gate timed out"
fi
POST_SHA="$(h_file_sha)"; POST_SIZE="$(h_file_size)"; POST_META="$(h_file_meta)"
[ "$POST_SHA" = "$EXPECTED_POST_SHA" ] \
  || die_rollback "post-edit live sha256 $POST_SHA != editor-proven $EXPECTED_POST_SHA"
[ "$POST_SIZE" = "$PRE_SIZE" ] || die_rollback "file size changed ($PRE_SIZE -> $POST_SIZE)"
[ "$POST_META" = "$PRE_META" ] || warn "owner/mode changed: '$PRE_META' -> '$POST_META'"
ok "file rewritten: $PRE_SHA -> $POST_SHA (size unchanged, $POST_SIZE bytes; owner/mode $POST_META)"

step "APPLY  static verification before restart"
N1="$(h_count_src "$TARGET_SHA")";        [ "$N1" = "1" ] || die_rollback "new pin appears $N1 times in source, expected 1"
N2="$(h_count_src "$CUR_IMAGE")";         [ "$N2" = "0" ] || die_rollback "OLD pin still appears $N2 times in source"
N3="$(h_count_src "$TEXT_PIN_EXPECTED")"; [ "$N3" = "1" ] || die_rollback "TEXT pin count is $N3, expected 1"
ok "source: new pin x1, old pin x0, TEXT pin x1 (intact)"

step "APPLY  set aside stale bytecode"
PYC_RES="$(h_pyc_aside "training_repo.cpython-312.pyc.$STAMP.stale")"
ok "stale .pyc: $PYC_RES (derived artifact; CPython regenerates it on import)"

# ============================================================================
# RESTART
# ============================================================================
step "RESTART  $SERVICE"
RESTART_EPOCH="$(date -u +%s)"
h_restart || die_rollback "systemctl restart failed"
NEWPID=""; ACT=""
i=0
while [ $i -lt 90 ]; do
  ACT="$(h_active)"; NEWPID="$(h_mainpid)"
  if [ "$ACT" = "active" ] && [ -n "$NEWPID" ] && [ "$NEWPID" != "0" ] && [ "$NEWPID" != "$PID0" ]; then break; fi
  i=$((i+1)); nap 1
done
[ "$ACT" = "active" ] || die_rollback "service did not become active (state=$ACT after ${i}s)"
ok "ActiveState=active after ${i}s"
[ -n "$NEWPID" ] && [ "$NEWPID" != "0" ] || die_rollback "no MainPID after restart"
[ "$NEWPID" != "$PID0" ] || die_rollback "MainPID did not change ($PID0) -- the process was NOT replaced, it still serves the old pin"
ok "MainPID $PID0 -> $NEWPID (process genuinely replaced)"

LSN=""
i=0
while [ $i -lt 60 ]; do LSN="$(h_listener)"; [ -n "$LSN" ] && break; i=$((i+1)); nap 1; done
[ -n "$LSN" ] || die_rollback "nothing listening on :$PORT after ${i}s"
[ "$LSN" = "$NEWPID" ] || warn "listener pid $LSN != MainPID $NEWPID (uvicorn worker layout) -- continuing"
ok "listening on :$PORT (pid $LSN)"

# ============================================================================
# VERIFY
# ============================================================================
step "VERIFY  1/5  exact Fiber auth-stage routes"
V="$(h_auth_probe 127.0.0.1 "$ENDPOINT_ROUTE")" || die_rollback "loopback $ENDPOINT_ROUTE failed exact Fiber auth-stage validation: $V"
ok "127.0.0.1 $ENDPOINT_ROUTE -> exact Fiber auth stage"
V="$(h_auth_probe 127.0.0.1 /training_repo/text)" || die_rollback "loopback /training_repo/text failed exact Fiber auth-stage validation: $V"
ok "127.0.0.1 /training_repo/text -> exact Fiber auth stage"
V="$(h_probe_code 127.0.0.1 /openapi.json)"; [ "$V" = "200" ] || die_rollback "/openapi.json returned $V, expected 200 (route table not built)"
ok "127.0.0.1 /openapi.json -> 200 (router registered)"
if [ "$MOCK" != "1" ] && [ "$SKIP_EXT_PROBE" != "1" ]; then
  V="$(h_auth_probe "$EXT_IP" "$ENDPOINT_ROUTE")" || die_rollback "external $EXT_IP route failed exact Fiber auth-stage validation: $V"
  ok "$EXT_IP:$PORT $ENDPOINT_ROUTE -> exact Fiber auth stage (validator-facing path)"
  V="$(h_auth_probe_local "$ENDPOINT_ROUTE")" || die_rollback "off-host route failed exact Fiber auth-stage validation: $V"
  ok "off-host probe from this workstation -> exact Fiber auth stage"
fi

step "VERIFY  2/5  new pin appears exactly once, old pin absent (source)"
N1="$(h_count_src "$TARGET_SHA")"; [ "$N1" = "1" ] || die_rollback "new pin count in source is $N1, expected 1"
N2="$(h_count_src "$CUR_IMAGE")";  [ "$N2" = "0" ] || die_rollback "old pin still in source (count $N2)"
ok "source: $TARGET_SHA x1, $CUR_IMAGE x0"

step "VERIFY  3/5  the RUNNING interpreter's bytecode carries the new pin"
PC_NEW="$(h_count_pyc "$TARGET_SHA")"; PC_OLD="$(h_count_pyc "$CUR_IMAGE")"
if [ "$PC_NEW" = "NOPYC" ]; then
  die_rollback "__pycache__ was not regenerated at all -- cannot prove the module was re-imported"
fi
[ "$PC_OLD" = "0" ] || die_rollback "compiled bytecode STILL contains the old pin (count $PC_OLD) -- the module was not re-imported"
[ "$PC_NEW" = "1" ] || die_rollback "compiled bytecode contains the new pin $PC_NEW times, expected 1"
PC_TXT="$(h_count_pyc "$TEXT_PIN_EXPECTED")"; [ "$PC_TXT" = "1" ] || die_rollback "TEXT pin count in bytecode is $PC_TXT, expected 1"
ok "bytecode: $TARGET_SHA x1, $CUR_IMAGE x0, TEXT pin x1"
ok "the loaded module provably carries the new pin"

step "VERIFY  4/5  TEXT tournament pin untouched"
N3="$(h_count_src "$TEXT_PIN_EXPECTED")"; [ "$N3" = "1" ] || die_rollback "TEXT pin count is $N3"
ok "$TEXT_PIN_EXPECTED still present exactly once"

step "VERIFY  5/5  one LIVE metagraph sync completes on PID $NEWPID"
log "  note: on a cold start the miner reports 'in sync' from its saved node cache"
log "        and only performs the first LIVE chain sync ~5 minutes later."
log "        waiting up to ${SYNC_TIMEOUT}s ..."
SYNC_OK=0; i=0
while [ $i -lt "$SYNC_TIMEOUT" ]; do
  C="$(h_sync_count "$NEWPID" "$RESTART_EPOCH")"
  case "$C" in ''|*[!0-9]*) C=0 ;; esac
  if [ "$C" -ge 1 ]; then SYNC_OK=1; break; fi
  if [ $((i % 30)) -eq 0 ] && [ $i -gt 0 ]; then log "        ... ${i}s elapsed, no live sync yet"; fi
  i=$((i+5)); nap 5
done
[ "$SYNC_OK" = "1" ] || die_rollback "no live metagraph sync on PID $NEWPID within ${SYNC_TIMEOUT}s"
ok "live metagraph sync completed on the new PID (after ~${i}s)"

if [ "$MODE" = "repoint" ]; then
  if [ "$MOCK" = "1" ] && [ -n "${SN56_MOCK_BEFORE_FINAL_CONTRACT_GATE:-}" ]; then
    mock_pause_gate "$SN56_MOCK_BEFORE_FINAL_CONTRACT_GATE" "before final contract reproof" \
      || die_rollback "mock before-final-contract gate timed out"
  fi
  step "FINAL CONTRACT  anonymous ref, local tree, and READY authorization"
  FINAL_CONTRACT_RECEIPT="$RUNDIR/final-contract.json"
  FINAL_READINESS_SNAPSHOT="$RUNDIR/final-readiness.json"
  validate_forward_receipt_bounded "$FINAL_CONTRACT_RECEIPT" \
    || die_rollback "final full release contract/ref reproof failed"
  compare_contract_receipts "$PREMUTATION_CONTRACT_RECEIPT" "$FINAL_CONTRACT_RECEIPT" \
    || die_rollback "final contract identity differs from the pre-mutation snapshot"
  snapshot_reviewed_file "$READINESS_RECEIPT" "$FINAL_READINESS_SNAPSHOT" \
    || die_rollback "could not snapshot the final READY receipt"
  validate_ready_snapshot "$FINAL_READINESS_SNAPSHOT" "$FINAL_CONTRACT_RECEIPT" \
    || die_rollback "final READY receipt validation failed"
  ok "anonymous target ref and complete release authorization remain exact"
fi

FINAL_SHA="$(h_file_sha)"
[ "$FINAL_SHA" = "$EXPECTED_POST_SHA" ] \
  || die_rollback "final live sha256 $FINAL_SHA != editor-proven $EXPECTED_POST_SHA before evidence"
ok "final source bytes still equal the editor-proven sha256 ($FINAL_SHA)"

# All source/bytecode/service/route/live-sync checks have completed. Until this
# point every forward termination signal retains the verified rollback path.
if [ "$MODE" = "repoint" ]; then
  ROLLBACK_ARMED=0
  trap - HUP INT TERM
  ok "forward auto-rollback disarmed only after complete verification"
fi

# ============================================================================
# DONE
# ============================================================================
EV="$EVIDENCE_DIR/SN56-WEEK6-REPOINT-EVIDENCE-$STAMP.json"
[ "$MOCK" = "1" ] && EV="$MOCKROOT/evidence-$STAMP.json"
cat > "$EV" <<JSON
{
  "stamp_utc": "$STAMP",
  "mode": "$MODE",
  "mock": $([ "$MOCK" = 1 ] && echo true || echo false),
  "host": "$SSH_HOST",
  "file": "$([ "$MOCK" = 1 ] && echo "$M_FILE" || echo "$R_FILE")",
  "image_pin_before": "$CUR_IMAGE",
  "image_pin_after": "$TARGET_SHA",
  "text_pin_unchanged": "$TEXT_PIN_EXPECTED",
  "file_sha256_before": "$PRE_SHA",
  "file_sha256_after": "$FINAL_SHA",
  "file_bytes": $PRE_SIZE,
  "backup_path": "$BK_PATH",
  "backup_sha256": "$BK_SHA",
  "release_contract": {
    "manifest": "$MANIFEST",
    "manifest_sha256": "$MANIFEST_SHA256",
    "release_state": "$RELEASE_STATE",
    "manifest_signature_state": "$MANIFEST_SIGNATURE_STATE",
    "manifest_signature_sha256": "$MANIFEST_SIGNATURE_SHA256",
    "repo": "$REPO_URL",
    "target_commit": "$RELEASE_SHA",
    "target_tree": "$RELEASE_TREE",
    "target_ref": "$RELEASE_REF",
    "target_tree_digest": "$TARGET_DIGEST",
    "rollback_commit": "$ROLLBACK_SHA",
    "rollback_ref": "$ROLLBACK_REF",
    "rollback_tree_digest": "$ROLLBACK_DIGEST",
    "allowed_changes_name_status_sha256": "$ALLOWED_CHANGES_DIGEST",
    "docker_policy_sha256": "$DOCKER_POLICY_SHA256",
    "dockerfiles": [
      {"path": "$DOCKER_0_PATH", "sha256": "$DOCKER_0_SHA256"},
      {"path": "$DOCKER_1_PATH", "sha256": "$DOCKER_1_SHA256"}
    ]
  },
  "service": { "unit": "$SERVICE", "user": "$SERVICE_USER", "working_directory": "$SERVICE_WORKING_DIRECTORY", "exec_start": "$SERVICE_EXEC_START", "pid_before": "$PID0", "pid_after": "$NEWPID", "listener_pid": "$LSN" },
  "verified": ["fiber_auth_stage_body","openapi_200","no_environment_repo_entry","new_pin_x1_source","old_pin_absent_source",
               "new_pin_x1_bytecode","old_pin_absent_bytecode","text_pin_intact","live_metagraph_sync"],
  "rollback_command": $ROLLBACK_COMMAND_JSON
}
JSON

log ""
log "${C_G}${C_B}REPOINT COMPLETE${C_0}"
log "  IMAGE pin now : $TARGET_SHA"
log "  TEXT pin      : $TEXT_PIN_EXPECTED (untouched)"
log "  file sha256   : $PRE_SHA -> $FINAL_SHA"
log "  backup        : $BK_PATH"
log "  evidence      : $EV"
log ""
log "  ${C_B}ONE-LINE ROLLBACK:${C_0}  $ROLLBACK_COMMAND"
log ""
h_recent_log | sed 's/^/  | /'
exit 0
