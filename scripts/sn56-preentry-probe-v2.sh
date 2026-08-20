#!/usr/bin/env bash
# =============================================================================
# SN56 pre-entry readiness probe, v2 — "make Monday boring".
#
# Supersedes sn56-preentry-probe.sh. That script probed the SUPPLY CHAIN (can
# the validator clone and build our repo?). This one additionally probes our
# OWN OPERATIONAL STATE (can we actually enter, and will we be watching?),
# which is where every near-miss has actually come from.
#
# READ-ONLY. It never restarts a service, never edits the endpoint, never
# changes the served pin, and never moves funds. Run it and read the lines.
#
# Usage:
#   sn56-preentry-probe-v2.sh --manifest release/week9-release-manifest.json
#   sn56-preentry-probe-v2.sh --manifest PATH --json out.json
#   sn56-preentry-probe-v2.sh --manifest PATH --mode mock --fixtures DIR
#   sn56-preentry-probe-v2.sh --skip-chain        # skip the slow substrate query
#   sn56-preentry-probe-v2.sh --list              # list check ids and exit
#
# Output: exactly one `PASS <id> <detail>` or `FAIL <id> <detail>` line per
# check. Exit code = number of FAILs (0 = green). WARN lines are advisory and
# do not affect the exit code.
#
# Every external interaction goes through _http_body / _http_exchange / _host so
# that `--mode mock` can replay the whole probe from a fixture directory with
# no network and no host access. See tests/test_release_probe.py.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTRACT="$SCRIPT_DIR/sn56-release-contract.py"
DOCKER_POLICY="$SCRIPT_DIR/../release/week9-docker-policy.json"
READINESS_RECEIPT="$SCRIPT_DIR/../release/week9-release-readiness.json"

# --- subject under test ------------------------------------------------------
# The release manifest is the only source of the target SHA and production
# literals. The parser below rejects a manifest that widens or redirects the
# reviewed operational surface.
MANIFEST="$SCRIPT_DIR/../release/week9-release-manifest.json"
IMAGE_COMMIT=""
ROLLBACK_COMMIT=""
ENDPOINT_PATH=""
ENDPOINT_PYC=""
ENDPOINT_HOST=""
ENDPOINT_PORT=""
ENDPOINT_ROUTE=""
SSH_ALIAS=""
MINER_UNIT=""
MINER_USER=""
MINER_WORKING_DIRECTORY=""
MINER_EXEC_START=""
MINER_ASGI_MODULE=""
TEXT_PIN_EXPECTED=""
SOURCE_REPOSITORY_URL=""

COLDKEY="5Esa4zXnnqdiWZTG8T8wbY8xhBJ7f35NH6Yykmw8WFdvZ5o3"
HOTKEY="5HLA2QWYy3GpNwCiBCwMMhGw5hpPosSeZ41my8B3nuFDQHNF"
NETUID="56"

GOD_CHECKOUT="/home/miner/god"
BASELINE_ENV="${SN56_BASELINE_ENV:-$SCRIPT_DIR/sn56-upstream-baseline.env}"

# --- thresholds --------------------------------------------------------------
METAGRAPH_MAX_AGE_S=600      # miner syncs every 300s; 600s = at most one missed cycle
DISK_MIN_FREE_GIB=120        # watcher reserve is 100 GiB; keep headroom above it
MEM_MIN_AVAIL_MIB=8192
WATCHER_START_LEAD_MAX_H=6   # watcher must arm at most this long before the start
WATCHER_START_LEAD_MIN_M=15  # ...and at least this long before it
WATCHER_MIN_TAIL_H=24        # hard-stop must be at least this far past the start
WATCHER_ENTRY_MAX_LAG_H=3    # entry-capture must land within this of the start

# --- mode --------------------------------------------------------------------
MODE="live"
FIXTURES=""
JSON_OUT=""
SKIP_CHAIN=0
NOW_OVERRIDE=""

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --manifest)  MANIFEST="${2:-}"; shift 2 ;;
    --mode)      MODE="${2:-}"; shift 2 ;;
    --fixtures)  FIXTURES="${2:-}"; shift 2 ;;
    --json)      JSON_OUT="${2:-}"; shift 2 ;;
    --skip-chain) SKIP_CHAIN=1; shift ;;
    --now)       NOW_OVERRIDE="${2:-}"; shift 2 ;;   # test hook: ISO8601 Z
    --list)      grep -oE '^ *check [a-z]+\.[a-z_]+' "${BASH_SOURCE[0]}" | awk '{print $2}' | sort -u; exit 0 ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$MODE" in
  live|mock) ;;
  *) echo "--mode must be 'live' or 'mock'" >&2; exit 2 ;;
esac

if [ "$MODE" = "mock" ] && [ -z "$FIXTURES" ]; then
  echo "--mode mock requires --fixtures DIR" >&2; exit 2
fi
if [ "$MODE" = "live" ] && [ -n "$FIXTURES" ]; then
  echo "--fixtures is a mock-only test hook" >&2; exit 2
fi
if [ "$MODE" = "live" ] && [ -n "$NOW_OVERRIDE" ]; then
  echo "--now is a mock-only test hook; live probes use the wall clock" >&2; exit 2
fi
if [ "$MODE" = "live" ] && [ "$SKIP_CHAIN" = "1" ]; then
  echo "--skip-chain is a mock-only test hook; live probes must prove registration" >&2; exit 2
fi
if [ "$MODE" = "live" ] && [ "${SN56_BASELINE_ENV+x}" = "x" ]; then
  echo "SN56_BASELINE_ENV is a mock-only test hook; live probes use the tracked reviewed baseline" >&2
  exit 2
fi

MANIFEST_VALUES="$(python3 - "$CONTRACT" "$MANIFEST" "$DOCKER_POLICY" \
  "$READINESS_RECEIPT" "$MODE" <<'PY'
import copy
import hashlib
import importlib.util
import sys
from pathlib import Path

contract_path, manifest_path, policy_path, readiness_path, mode = sys.argv[1:]

def die(message):
    print(f"release artifact error: {message}", file=sys.stderr)
    raise SystemExit(2)

try:
    spec = importlib.util.spec_from_file_location("sn56_release_contract", contract_path)
    if spec is None or spec.loader is None:
        die(f"cannot load contract validator {contract_path}")
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    manifest, raw = contract.load_manifest(Path(manifest_path))
    contract.validate_schema(manifest)
    docker_policy = contract.verify_docker_policy(manifest, Path(policy_path))
    manifest_sha = hashlib.sha256(raw).hexdigest()
    readiness_sha = "mock-not-required"
    if mode == "live":
        if manifest["release_state"] != "ready":
            die("live probe requires release_state=ready and its independent readiness receipt")
        binding = {
            "manifest_sha256": manifest_sha,
            "release_state": manifest["release_state"],
            "target": copy.deepcopy(manifest["target"]),
            "rollback": copy.deepcopy(manifest["rollback"]),
            "allowed_changes": {
                "base_commit": manifest["allowed_changes"]["base_commit"],
                "name_status_sha256": manifest["allowed_changes"]["name_status_sha256"],
                "count": len(manifest["allowed_changes"]["entries"]),
            },
            "dockerfiles": copy.deepcopy(manifest["dockerfiles"]),
            "docker_policy": docker_policy,
        }
        readiness = contract._validate_readiness_against_contract(
            Path(readiness_path), binding
        )
        readiness_sha = readiness["sha256"]
except SystemExit:
    raise
except Exception as exc:
    die(str(exc))

production = manifest["production"]
values = [
    manifest["release_state"], manifest["target"]["commit"],
    manifest["rollback"]["commit"], production["ssh_host"],
    production["service"], production["service_user"],
    production["service_working_directory"], production["service_exec_start"],
    production["service_asgi_module"], production["endpoint_host"],
    str(production["endpoint_port"]), production["endpoint_route"],
    production["endpoint_source"], production["endpoint_pyc"],
    production["text_pin"], manifest["source"]["repository_url"],
    manifest_sha, docker_policy["sha256"], readiness_sha,
]
print("\t".join(values))
PY
)" || exit 2

IFS=$'\t' read -r RELEASE_STATE IMAGE_COMMIT ROLLBACK_COMMIT SSH_ALIAS \
  MINER_UNIT MINER_USER MINER_WORKING_DIRECTORY MINER_EXEC_START \
  MINER_ASGI_MODULE ENDPOINT_HOST ENDPOINT_PORT ENDPOINT_ROUTE \
  ENDPOINT_PATH ENDPOINT_PYC TEXT_PIN_EXPECTED SOURCE_REPOSITORY_URL \
  MANIFEST_SHA256 DOCKER_POLICY_SHA256 READINESS_SHA256 <<< "$MANIFEST_VALUES"

# =============================================================================
# I/O indirection — the only places that touch the outside world.
# =============================================================================

# _http_body <key> <url> [curl args...]  -> body on stdout; non-zero on failure
_mock_url_matches() {
  local key="$1" url="$2" expected
  [ -f "$FIXTURES/$key.url" ] || return 0
  expected="$(sed -n '1p' "$FIXTURES/$key.url")"
  [ "$url" = "$expected" ] || {
    echo "mock URL mismatch for $key: got '$url', expected '$expected'" >&2
    return 1
  }
}

_http_body() {
  local key="$1" url="$2"; shift 2
  if [ "$MODE" = "mock" ]; then
    _mock_url_matches "$key" "$url" || return 1
    [ -f "$FIXTURES/$key.body" ] || return 1
    cat "$FIXTURES/$key.body"; return 0
  fi
  curl -sS -m 25 "$@" "$url" 2>/dev/null
}

# _http_exchange <key> <url> <body-file> [curl args...] writes the body and
# prints the status from ONE request. Splitting these allowed a restart between
# samples to synthesize a route verdict no single serving process produced.
_http_exchange() {
  local key="$1" url="$2" body_file="$3" code; shift 3
  if [ "$MODE" = "mock" ]; then
    _mock_url_matches "$key" "$url" || { echo "000"; return 0; }
    if [ -f "$FIXTURES/$key.body" ]; then
      cp "$FIXTURES/$key.body" "$body_file"
    else
      : > "$body_file"
    fi
    if [ -f "$FIXTURES/$key.code" ]; then cat "$FIXTURES/$key.code"; else echo "000"; fi
    return 0
  fi
  code="$(curl -sS -m 25 -o "$body_file" -w "%{http_code}" "$@" "$url" 2>/dev/null)" || {
    : > "$body_file"
    echo "000"
    return 0
  }
  printf '%s\n' "$code"
}

# _host <key> <remote command> -> stdout of the remote command
_host() {
  local key="$1" cmd="$2"
  if [ "$MODE" = "mock" ]; then
    [ -f "$FIXTURES/$key.out" ] || return 1
    cat "$FIXTURES/$key.out"; return 0
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$SSH_ALIAS" "$cmd" 2>/dev/null
}

_now_epoch() {
  if [ -n "$NOW_OVERRIDE" ]; then
    python3 -c "import datetime,sys;print(int(datetime.datetime.strptime(sys.argv[1],'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()))" "$NOW_OVERRIDE"
  else
    date -u +%s
  fi
}

# =============================================================================
# result bookkeeping
# =============================================================================
FAILS=0
WARNS=0
declare -a RESULT_IDS=() RESULT_STATES=() RESULT_DETAILS=()

_emit() { # state id detail
  local state="$1" id="$2" detail="$3"
  printf '%-4s %-26s %s\n' "$state" "$id" "$detail"
  RESULT_IDS+=("$id"); RESULT_STATES+=("$state"); RESULT_DETAILS+=("$detail")
  case "$state" in
    FAIL) FAILS=$((FAILS+1)) ;;
    WARN) WARNS=$((WARNS+1)) ;;
  esac
}
pass() { _emit PASS "$1" "$2"; }
fail() { _emit FAIL "$1" "$2"; }
warn() { _emit WARN "$1" "$2"; }

# `check <id>` markers exist so --list can enumerate ids without executing.
check() { :; }

NOW_EPOCH="$(_now_epoch)"

# =============================================================================
# SECTION 1 — tournament schedule, derived from pinned upstream source
# =============================================================================
echo "== schedule (authority: upstream validator source, never local docs) =="

SCHED_DAY=""; SCHED_HOUR=""; NEXT_START_EPOCH=""; NEXT_START_ISO=""
UPSTREAM_CONSTANTS="$(_http_body upstream_constants \
  "https://api.github.com/repos/gradients-ai/G.O.D/contents/validator/tournament/constants.py?ref=main" \
  -H "Accept: application/vnd.github.raw")"

# _const <text> <NAME> -> integer value of `NAME = <int>` (portable; no GNU sed).
_const() {
  printf '%s' "$1" | python3 -c "
import re,sys
m=re.search(r'^'+re.escape(sys.argv[1])+r'\s*=\s*(\d+)', sys.stdin.read(), re.M)
print(m.group(1) if m else '')
" "$2" 2>/dev/null
}

check schedule.authority
if [ -n "${UPSTREAM_CONSTANTS:-}" ]; then
  SCHED_DAY="$(_const "$UPSTREAM_CONSTANTS" TOURNAMENT_SCHEDULE_IMAGE_DAY_OF_WEEK)"
  SCHED_HOUR="$(_const "$UPSTREAM_CONSTANTS" TOURNAMENT_SCHEDULE_IMAGE_HOUR)"
fi

if [ -n "${SCHED_DAY:-}" ] && [ -n "${SCHED_HOUR:-}" ]; then
  read -r NEXT_START_EPOCH NEXT_START_ISO CUR_START_EPOCH CUR_START_ISO <<<"$(python3 - "$SCHED_DAY" "$SCHED_HOUR" "$NOW_EPOCH" <<'PY'
import sys, datetime
day, hour, now_epoch = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
now = datetime.datetime.fromtimestamp(now_epoch, datetime.timezone.utc)
# Validator gate (should_start_new_tournament_after_interval): a tournament may
# start only while now.weekday()==day and now.hour==hour. Window = [hh:00, hh:59:59].
delta = (day - now.weekday()) % 7
cand = (now + datetime.timedelta(days=delta)).replace(hour=hour, minute=0, second=0, microsecond=0)
nxt = cand + datetime.timedelta(days=7) if cand <= now else cand
cur = nxt - datetime.timedelta(days=7)   # most recent start at/before now
print(int(nxt.timestamp()), nxt.strftime('%Y-%m-%dT%H:%M:%SZ'),
      int(cur.timestamp()), cur.strftime('%Y-%m-%dT%H:%M:%SZ'))
PY
)"
  DOW="$(python3 -c "import sys;print(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][int(sys.argv[1])])" "$SCHED_DAY")"
  HOURS_OUT="$(python3 -c "import sys;print(f'{(int(sys.argv[1])-int(sys.argv[2]))/3600:.1f}')" "$NEXT_START_EPOCH" "$NOW_EPOCH")"
  pass schedule.authority "image tourney = $DOW ${SCHED_HOUR}:00 UTC (window ${SCHED_HOUR}:00-${SCHED_HOUR}:59) -> next $NEXT_START_ISO, in ${HOURS_OUT}h [upstream validator/tournament/constants.py]"
else
  fail schedule.authority "could not read TOURNAMENT_SCHEDULE_IMAGE_{DAY_OF_WEEK,HOUR} from upstream constants.py"
fi

# The miner's local G.O.D checkout is NOT a schedule authority. Week 5 lost time
# to exactly this: the local checkout said 15:00, the real start was 13:00.
check schedule.local_drift
LOCAL_SCHED="$(_host local_sched "grep -hE '^TOURNAMENT_SCHEDULE_IMAGE_(DAY_OF_WEEK|HOUR)' $GOD_CHECKOUT/validator/tournament/constants.py | tr -d ' '")"
if [ -z "${LOCAL_SCHED:-}" ]; then
  fail schedule.local_drift "could not read schedule constants from local checkout $GOD_CHECKOUT"
else
  LOCAL_HOUR="$(_const "$LOCAL_SCHED" TOURNAMENT_SCHEDULE_IMAGE_HOUR)"
  LOCAL_DAY="$(_const "$LOCAL_SCHED" TOURNAMENT_SCHEDULE_IMAGE_DAY_OF_WEEK)"
  if [ "${LOCAL_HOUR:-x}" = "${SCHED_HOUR:-y}" ] && [ "${LOCAL_DAY:-x}" = "${SCHED_DAY:-y}" ]; then
    pass schedule.local_drift "local miner checkout agrees with upstream (day=$LOCAL_DAY hour=$LOCAL_HOUR)"
  else
    # The local miner checkout does not control validator scheduling and this
    # probe has already derived NEXT_START from live upstream source. Keep the
    # discrepancy visible, but do not turn a deliberately frozen production
    # checkout into a false no-go or invite a tournament-eve production edit.
    warn schedule.local_drift "local miner checkout says day=${LOCAL_DAY:-?} hour=${LOCAL_HOUR:-?} but upstream says day=${SCHED_DAY:-?} hour=${SCHED_HOUR:-?} — upstream remains the enforced schedule authority"
  fi
fi

# A new tournament only starts if the previous one is no longer ACTIVE/PENDING.
check tournament.prev_completed
PREV_JSON="$(_http_body latest_details "https://api.gradients.io/tournament/latest/details")"
PREV_STATE="$(printf '%s' "${PREV_JSON:-}" | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(1)
img=d.get('image') or {}
print(f\"{img.get('status','?')} {img.get('tournament_id','?')}\")
" 2>/dev/null)"
if [ -z "${PREV_STATE:-}" ]; then
  fail tournament.prev_completed "could not read image tournament state from /tournament/latest/details"
else
  set -- $PREV_STATE; PSTATUS="$1"; PID_="$2"
  case "$PSTATUS" in
    completed) pass tournament.prev_completed "previous image tournament $PID_ = completed; scheduler is free to start the next one" ;;
    active|pending) fail tournament.prev_completed "previous image tournament $PID_ = $PSTATUS — a new tournament will NOT be created while this holds" ;;
    *) warn tournament.prev_completed "previous image tournament $PID_ = $PSTATUS (unrecognised)" ;;
  esac
fi

# =============================================================================
# SECTION 2 — entry eligibility: registration + funds
# =============================================================================
echo
echo "== entry eligibility =="

check chain.registered
if [ "$SKIP_CHAIN" = "1" ]; then
  warn chain.registered "skipped (--skip-chain)"
else
  UID_OUT="$(_host chain_uid "timeout 150 sudo -u miner /home/miner/.venv/bin/python -c \"
from fiber.chain import interface
sub = interface.get_substrate(subtensor_network='finney')
r = sub.query('SubtensorModule','Uids',[$NETUID,'$HOTKEY'])
print('UID=%s' % (r.value if r is not None else 'NONE'))
\" 2>/dev/null | grep '^UID='")"
  UID_VAL="$(printf '%s' "${UID_OUT:-}" | sed -n 's/^UID=\(.*\)$/\1/p' | head -1)"
  if [ -n "${UID_VAL:-}" ] && [ "$UID_VAL" != "NONE" ] && [ "$UID_VAL" != "None" ]; then
    pass chain.registered "hotkey ${HOTKEY:0:12}... registered on netuid $NETUID at uid $UID_VAL"
  else
    fail chain.registered "hotkey ${HOTKEY:0:12}... has NO uid on netuid $NETUID (got '${UID_VAL:-unreadable}') — cannot enter"
  fi
fi

# Required fee is read LIVE from the validator, not from any local constant.
check entry.fee_required
FEES_JSON="$(_http_body fees "https://api.gradients.io/tournament/fees")"
REQ_RAO="$(printf '%s' "${FEES_JSON:-}" | python3 -c "
import json,sys
try: print(int(json.load(sys.stdin)['image_tournament_fee_rao']))
except Exception: pass
" 2>/dev/null)"
if [ -n "${REQ_RAO:-}" ]; then
  REQ_TAO="$(python3 -c "print(f'{int(__import__(\"sys\").argv[1])/1e9:.4f}')" "$REQ_RAO")"
  pass entry.fee_required "image entry fee = $REQ_RAO rao ($REQ_TAO TAO) [live /tournament/fees]"
else
  fail entry.fee_required "could not read image_tournament_fee_rao from /tournament/fees"
fi

check entry.balance
BAL_JSON="$(_http_body balance "https://api.gradients.io/tournament/balance/$COLDKEY")"
BAL_RAO="$(printf '%s' "${BAL_JSON:-}" | python3 -c "
import json,sys
try: print(int(json.load(sys.stdin)['balance_rao']))
except Exception: pass
" 2>/dev/null)"
if [ -z "${BAL_RAO:-}" ]; then
  fail entry.balance "could not read balance_rao for coldkey ${COLDKEY:0:12}..."
elif [ -z "${REQ_RAO:-}" ]; then
  fail entry.balance "balance is $BAL_RAO rao but the required fee is unknown — cannot assert eligibility"
elif [ "$BAL_RAO" -ge "$REQ_RAO" ]; then
  BAL_TAO="$(python3 -c "import sys;print(f'{int(sys.argv[1])/1e9:.4f}')" "$BAL_RAO")"
  pass entry.balance "balance $BAL_RAO rao ($BAL_TAO TAO) >= required $REQ_RAO rao"
else
  SHORT=$((REQ_RAO - BAL_RAO))
  SHORT_TAO="$(python3 -c "import sys;print(f'{int(sys.argv[1])/1e9:.4f}')" "$SHORT")"
  BAL_TAO="$(python3 -c "import sys;print(f'{int(sys.argv[1])/1e9:.4f}')" "$BAL_RAO")"
  # A zero/short balance is expected only just AFTER this cycle's entry fee has
  # been deducted.  A timestamp from last week's deduction must never make the
  # pre-start probe green for the next tournament: that was a deterministic DNF
  # path.  Accept the deduction exception only inside the bounded current-cycle
  # entry-capture window; before the scheduled start the full fee is required.
  BAL_UPDATED="$(printf '%s' "${BAL_JSON:-}" | python3 -c "
import json,sys
try: print(json.load(sys.stdin).get('updated_at') or '')
except Exception: pass
" 2>/dev/null)"
  ENTERED_THIS_CYCLE="$(python3 - "$BAL_UPDATED" "${CUR_START_ISO:-}" \
    "$NOW_EPOCH" "$WATCHER_ENTRY_MAX_LAG_H" <<'PYIN'
import sys, datetime
upd, start = (sys.argv[1] or '').strip(), (sys.argv[2] or '').strip()
now = datetime.datetime.fromtimestamp(int(sys.argv[3]), datetime.timezone.utc)
entry_lag = datetime.timedelta(hours=float(sys.argv[4]))
def parse(x):
    if not x: return None
    x = x.replace('Z', '+00:00')
    try:
        d = datetime.datetime.fromisoformat(x)
        return d if d.tzinfo else d.replace(tzinfo=datetime.timezone.utc)
    except Exception: return None
u, s = parse(upd), parse(start)
in_current_entry_window = bool(s and s <= now <= s + entry_lag)
ledger_matches_window = bool(u and s and s <= u <= now)
print("yes" if in_current_entry_window and ledger_matches_window else "no")
PYIN
)"
  if [ "$ENTERED_THIS_CYCLE" = "yes" ]; then
    pass entry.balance "balance $BAL_RAO rao ($BAL_TAO TAO) — fee already DEDUCTED for this cycle's entry (ledger updated $BAL_UPDATED, at/after this window's start). Fund $REQ_RAO rao before the NEXT start ${NEXT_START_ISO:-?}"
  else
    fail entry.balance "balance $BAL_RAO rao ($BAL_TAO TAO) < required $REQ_RAO rao — OWNER MUST SEND $SHORT rao ($SHORT_TAO TAO) to coldkey $COLDKEY before ${NEXT_START_ISO:-the start}"
  fi
fi

# =============================================================================
# SECTION 3 — the served release
# =============================================================================
echo
echo "== served release =="

check endpoint.reachable
# /health does not exist on this app; the signal that the app is alive AND
# routing is that the exact validator route answers with a 422 whose JSON errors
# have reached missing authentication headers. A 422 caused by path-enum
# rejection is a false positive; accepting status alone caused that exact defect
# in week 6. HTTP 400 is also not the reviewed route contract.
EP_URL="http://$ENDPOINT_HOST:$ENDPOINT_PORT$ENDPOINT_ROUTE"
EP_BODY_FILE="$(mktemp -t sn56-endpoint-body)"
EP_CODE="$(_http_exchange endpoint "$EP_URL" "$EP_BODY_FILE")"
EP_BODY="$(cat "$EP_BODY_FILE")"
rm -f "$EP_BODY_FILE"
EP_BODY_VERDICT="$(SN56_ENDPOINT_BODY="$EP_BODY" python3 - <<'PY'
import json
import os

try:
    payload = json.loads(os.environ.get("SN56_ENDPOINT_BODY", ""))
except Exception as exc:
    print(f"BAD\tmalformed JSON validation body: {exc}")
    raise SystemExit
detail = payload.get("detail") if isinstance(payload, dict) else None
if not isinstance(detail, list) or not detail:
    print("BAD\tvalidation body has no non-empty detail array")
    raise SystemExit

auth_header_errors = 0
auth_headers_seen = set()
fiber_v27_headers = {"validator-hotkey", "signature", "miner-hotkey", "nonce"}
for item in detail:
    if not isinstance(item, dict):
        print("BAD\tvalidation detail contains a non-object entry")
        raise SystemExit
    loc = item.get("loc")
    kind = item.get("type")
    kind_lower = kind.lower() if isinstance(kind, str) else ""
    if isinstance(loc, list):
        lowered = [str(value).lower() for value in loc]
    else:
        lowered = []
    if "enum" in kind_lower or ("path" in lowered and "task_type" in lowered):
        print("BAD\trequest was rejected at path/enum validation before authentication")
        raise SystemExit
    header_name = lowered[1] if len(lowered) == 2 and lowered[0] == "header" else ""
    if header_name not in fiber_v27_headers or kind != "missing":
        print(f"BAD\tunexpected validation error outside missing auth headers: {loc!r} / {kind!r}")
        raise SystemExit
    auth_header_errors += 1
    auth_headers_seen.add(header_name)

if auth_headers_seen != fiber_v27_headers or auth_header_errors != len(fiber_v27_headers):
    missing = sorted(fiber_v27_headers - auth_headers_seen)
    extra = sorted(auth_headers_seen - fiber_v27_headers)
    print(
        "BAD\texact Fiber v2.7 missing-header contract differs "
        f"(missing={missing}, extra={extra}, errors={auth_header_errors})"
    )
else:
    names = ",".join(sorted(auth_headers_seen))
    print(f"OK\t{auth_header_errors} missing auth-header error(s) [{names}], no path/enum error")
PY
)"
EP_BODY_STATE="${EP_BODY_VERDICT%%$'\t'*}"
EP_BODY_DETAIL="${EP_BODY_VERDICT#*$'\t'}"
if [ "$EP_CODE" = "422" ] && [ "$EP_BODY_STATE" = "OK" ]; then
  pass endpoint.reachable "$EP_URL -> 422 ($EP_BODY_DETAIL)"
elif [ "$EP_CODE" = "000" ]; then
  fail endpoint.reachable "http://$ENDPOINT_HOST:$ENDPOINT_PORT is UNREACHABLE from off-host — validators cannot pull our repo"
else
  fail endpoint.reachable "$EP_URL -> ${EP_CODE:-unreadable}; ${EP_BODY_DETAIL:-validation body unreadable}"
fi

check endpoint.pin
# Parse the semantic IMAGE/TEXT mapping from source, prove the current bytecode
# carries the same exact target, bind the loaded systemd unit and MainPID to the
# reviewed cwd/argv/import paths, and perform a loopback route exchange while
# that same MainPID remains stable. This is deliberately stronger than merely
# grepping source, which overclaimed what the running service was serving.
PIN_OUT="$(_host endpoint_pin "python3 -c '
import ast,json,os,re,shlex,subprocess,sys,urllib.error,urllib.request
from pathlib import Path
source,pyc_path,target,rollback,text_pin,source_repo,service,service_user,working_dir,expected_exec,asgi_module,port,route=sys.argv[1:]
def command(*args):
    return subprocess.run(args,text=True,stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL,check=False).stdout.strip()
pid_before=command(\"systemctl\",\"show\",service,\"-p\",\"MainPID\",\"--value\")
active_before=command(\"systemctl\",\"is-active\",service)
if not pid_before.isdigit() or int(pid_before)<=0:
    raise SystemExit(2)
actual_user=command(\"systemctl\",\"show\",service,\"-p\",\"User\",\"--value\")
actual_working_dir=command(\"systemctl\",\"show\",service,\"-p\",\"WorkingDirectory\",\"--value\")
exec_show=command(\"systemctl\",\"show\",service,\"-p\",\"ExecStart\",\"--value\")
exec_match=re.search(r\"(?:^|;\\s*)argv\\[\\]=(.*?)\\s*;\",exec_show)
actual_exec=exec_match.group(1).strip() if exec_match else None
proc_root=Path(f\"/proc/{pid_before}\")
process_cwd=os.readlink(proc_root/\"cwd\")
process_cmdline=[part.decode(\"utf-8\",\"surrogateescape\") for part in
                 proc_root.joinpath(\"cmdline\").read_bytes().split(b\"\\0\") if part]
process_pythonpath=None
for item in proc_root.joinpath(\"environ\").read_bytes().split(b\"\\0\"):
    if item.startswith(b\"PYTHONPATH=\"):
        process_pythonpath=item.split(b\"=\",1)[1].decode(\"utf-8\",\"surrogateescape\")
        break
expected_argv=shlex.split(expected_exec)
python_bin=str(Path(expected_argv[0]).with_name(\"python\"))
reviewed_python_realpath=os.path.realpath(python_bin)
process_exe=os.readlink(proc_root/\"exe\")
process_exe_realpath=os.path.realpath(process_exe)
resolve_script=\"import importlib.util,json,sys;print(json.dumps([importlib.util.find_spec(name).origin for name in sys.argv[1:]]))\"
resolve_run=subprocess.run(
    [python_bin,\"-c\",resolve_script,\"miner.asgi\",\"miner.endpoints.training_repo\"],
    cwd=working_dir,env={\"PATH\":str(Path(python_bin).parent)+\":/usr/bin:/bin\",\"ENV\":\"DEV\"},
    text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,check=False,
)
try:
    resolved_modules=json.loads(resolve_run.stdout) if resolve_run.returncode==0 else []
except Exception:
    resolved_modules=[]
raw=Path(source).read_bytes()
tree=ast.parse(raw.decode(\"utf-8\"),filename=source)
repos=[]
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==\"_REPOS\" for t in node.targets):
        repos.append(node.value)
if len(repos)!=1 or not isinstance(repos[0],ast.Dict):
    raise SystemExit(3)
entries={}
for key,value in zip(repos[0].keys,repos[0].values):
    name=key.attr if isinstance(key,ast.Attribute) else None
    if name not in (\"IMAGE\",\"TEXT\"):
        continue
    if name in entries or not isinstance(value,ast.Call):
        raise SystemExit(4)
    fields={}
    for kw in value.keywords:
        if kw.arg in (\"commit_hash\",\"github_repo\") and isinstance(kw.value,ast.Constant) and isinstance(kw.value.value,str):
            fields[kw.arg]=kw.value.value
    entries[name]=fields
if set(entries)!={\"IMAGE\",\"TEXT\"} or \"commit_hash\" not in entries[\"IMAGE\"] or \"commit_hash\" not in entries[\"TEXT\"]:
    raise SystemExit(5)
pyc=Path(pyc_path).read_bytes()
enc=lambda value:value.encode(\"ascii\")
source_mtime=Path(source).stat().st_mtime
pyc_mtime=Path(pyc_path).stat().st_mtime
proc_stat=Path(f\"/proc/{pid_before}/stat\").read_text()
proc_fields=proc_stat[proc_stat.rfind(\")\")+2:].split()
start_ticks=int(proc_fields[19])
boot_epoch=next(int(line.split()[1]) for line in Path(\"/proc/stat\").read_text().splitlines() if line.startswith(\"btime \"))
process_start=boot_epoch+start_ticks/os.sysconf(\"SC_CLK_TCK\")
listener_text=command(\"ss\",\"-ltnp\",f\"sport = :{port}\")
listener_pids=sorted({int(value) for value in re.findall(r\"pid=(\\d+)\",listener_text)})
def descends_from(pid,ancestor):
    seen=set()
    while pid>0 and pid not in seen:
        if pid==ancestor:
            return True
        seen.add(pid)
        try:
            value=Path(f\"/proc/{pid}/stat\").read_text()
            fields=value[value.rfind(\")\")+2:].split()
            pid=int(fields[1])
        except Exception:
            return False
    return False
listener_bound=any(descends_from(pid,int(pid_before)) for pid in listener_pids)
loopback_url=f\"http://127.0.0.1:{port}{route}\"
loopback_code=None
loopback_body=\"\"
loopback_error=None
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open(urllib.request.Request(loopback_url,method=\"GET\"),timeout=10) as response:
        loopback_code=response.status
        loopback_body=response.read().decode(\"utf-8\",\"replace\")
except urllib.error.HTTPError as exc:
    loopback_code=exc.code
    loopback_body=exc.read().decode(\"utf-8\",\"replace\")
except Exception as exc:
    loopback_error=f\"{type(exc).__name__}: {exc}\"
pid_after=command(\"systemctl\",\"show\",service,\"-p\",\"MainPID\",\"--value\")
active_after=command(\"systemctl\",\"is-active\",service)
print(json.dumps({
    \"source_image_pin\":entries[\"IMAGE\"][\"commit_hash\"],
    \"source_image_repo\":entries[\"IMAGE\"].get(\"github_repo\"),
    \"source_text_pin\":entries[\"TEXT\"][\"commit_hash\"],
    \"source_target_count\":raw.count(enc(target)),
    \"source_rollback_count\":raw.count(enc(rollback)),
    \"source_text_count\":raw.count(enc(text_pin)),
    \"pyc_target_count\":pyc.count(enc(target)),
    \"pyc_rollback_count\":pyc.count(enc(rollback)),
    \"pyc_text_count\":pyc.count(enc(text_pin)),
    \"pid_before\":pid_before,
    \"pid_after\":pid_after,
    \"service_state_before\":active_before,
    \"service_state_after\":active_after,
    \"service_user\":actual_user,
    \"service_working_directory\":actual_working_dir,
    \"service_exec_argv\":actual_exec,
    \"process_cwd\":process_cwd,
    \"process_cmdline\":process_cmdline,
    \"reviewed_python\":python_bin,
    \"reviewed_python_realpath\":reviewed_python_realpath,
    \"process_exe\":process_exe,
    \"process_exe_realpath\":process_exe_realpath,
    \"process_pythonpath\":process_pythonpath,
    \"resolved_asgi_module\":resolved_modules[0] if len(resolved_modules)==2 else None,
    \"resolved_training_repo_module\":resolved_modules[1] if len(resolved_modules)==2 else None,
    \"listener_pids\":listener_pids,
    \"listener_bound\":listener_bound,
    \"process_start_epoch\":process_start,
    \"source_mtime_epoch\":source_mtime,
    \"pyc_mtime_epoch\":pyc_mtime,
    \"loopback_url\":loopback_url,
    \"loopback_code\":loopback_code,
    \"loopback_body\":loopback_body,
    \"loopback_error\":loopback_error,
}))
' '$ENDPOINT_PATH' '$ENDPOINT_PYC' '$IMAGE_COMMIT' '$ROLLBACK_COMMIT' '$TEXT_PIN_EXPECTED' '$SOURCE_REPOSITORY_URL' '$MINER_UNIT' '$MINER_USER' '$MINER_WORKING_DIRECTORY' '$MINER_EXEC_START' '$MINER_ASGI_MODULE' '$ENDPOINT_PORT' '$ENDPOINT_ROUTE'")"
PIN_VERDICT="$(SN56_PIN_JSON="$PIN_OUT" python3 - "$IMAGE_COMMIT" "$TEXT_PIN_EXPECTED" \
  "$SOURCE_REPOSITORY_URL" "$MINER_USER" "$MINER_WORKING_DIRECTORY" \
  "$MINER_EXEC_START" "$MINER_ASGI_MODULE" "$ENDPOINT_PATH" \
  "$ENDPOINT_PORT" "$ENDPOINT_ROUTE" <<'PY'
import json
import os
import shlex
import sys

(
    target, text_pin, source_repo, service_user, working_dir, expected_exec,
    asgi_module, training_repo_module, endpoint_port, endpoint_route,
) = sys.argv[1:]
try:
    value = json.loads(os.environ.get("SN56_PIN_JSON", ""))
except Exception as exc:
    print(f"BAD\tendpoint source/pyc evidence unreadable: {exc}")
    raise SystemExit
wanted = {
    "source_image_pin": target,
    "source_text_pin": text_pin,
    "source_target_count": 1,
    "source_rollback_count": 0,
    "source_text_count": 1,
    "pyc_target_count": 1,
    "pyc_rollback_count": 0,
    "pyc_text_count": 1,
    "service_user": service_user,
    "service_working_directory": working_dir,
    "service_exec_argv": expected_exec,
    "process_cwd": working_dir,
    "resolved_asgi_module": asgi_module,
    "resolved_training_repo_module": training_repo_module,
    "loopback_url": f"http://127.0.0.1:{endpoint_port}{endpoint_route}",
    "loopback_code": 422,
}
wrong = [f"{key}={value.get(key)!r} (want {expected!r})"
         for key, expected in wanted.items() if value.get(key) != expected]
expected_argv = shlex.split(expected_exec)
expected_python = os.path.join(os.path.dirname(expected_argv[0]), "python")
process_cmdline = value.get("process_cmdline")
if not isinstance(process_cmdline, list) or not all(
    isinstance(item, str) for item in process_cmdline
):
    wrong.append(f"process_cmdline={process_cmdline!r} is unreadable")
else:
    reviewed_python_realpath = value.get("reviewed_python_realpath")
    allowed_interpreters = {expected_python}
    if isinstance(reviewed_python_realpath, str) and reviewed_python_realpath:
        allowed_interpreters.add(reviewed_python_realpath)
    if (
        len(process_cmdline) != len(expected_argv) + 1
        or process_cmdline[0] not in allowed_interpreters
        or process_cmdline[1:] != expected_argv
    ):
        wrong.append(
            f"process_cmdline={process_cmdline!r} is not the reviewed venv "
            f"interpreter followed by exact argv {expected_argv!r}"
        )
if value.get("reviewed_python") != expected_python:
    wrong.append(
        f"reviewed_python={value.get('reviewed_python')!r} (want {expected_python!r})"
    )
if (
    not isinstance(value.get("reviewed_python_realpath"), str)
    or not value.get("reviewed_python_realpath")
    or value.get("process_exe_realpath") != value.get("reviewed_python_realpath")
):
    wrong.append(
        f"process_exe={value.get('process_exe')!r} resolves to "
        f"{value.get('process_exe_realpath')!r}, not reviewed interpreter "
        f"{value.get('reviewed_python_realpath')!r}"
    )
if value.get("process_pythonpath") not in (None, ""):
    wrong.append(
        f"process_pythonpath={value.get('process_pythonpath')!r} can redirect imports"
    )
pid_before = value.get("pid_before")
pid_after = value.get("pid_after")
if (not isinstance(pid_before, str) or not pid_before.isdigit()
        or int(pid_before) <= 0 or pid_after != pid_before):
    wrong.append(f"pid_before/after={pid_before!r}/{pid_after!r} (want one stable nonzero MainPID)")
for key in ("service_state_before", "service_state_after"):
    if value.get(key) != "active":
        wrong.append(f"{key}={value.get(key)!r} (want 'active')")
if value.get("listener_bound") is not True:
    wrong.append(f"listener_pids={value.get('listener_pids')!r} are not MainPID/descendants")
process_start = value.get("process_start_epoch")
source_mtime = value.get("source_mtime_epoch")
if not isinstance(process_start, (int, float)) or not isinstance(source_mtime, (int, float)):
    wrong.append("process/source timestamps are unreadable")
elif process_start + 1 < source_mtime:
    wrong.append(
        f"MainPID started at {process_start}, before source mtime {source_mtime}; "
        "the active process may still carry the old mapping"
    )
if not isinstance(value.get("pyc_mtime_epoch"), (int, float)):
    wrong.append("pyc mtime is unreadable")

def exact_fiber_v27_route(body):
    try:
        payload = json.loads(body)
    except Exception as exc:
        return f"loopback response is malformed JSON: {exc}"
    detail = payload.get("detail") if isinstance(payload, dict) else None
    expected_headers = {"validator-hotkey", "signature", "miner-hotkey", "nonce"}
    if not isinstance(detail, list) or len(detail) != len(expected_headers):
        return f"loopback detail has {len(detail) if isinstance(detail, list) else 'no'} errors (want 4)"
    seen = set()
    for item in detail:
        if not isinstance(item, dict) or item.get("type") != "missing":
            return f"loopback detail contains a non-missing error: {item!r}"
        loc = item.get("loc")
        lowered = [str(part).lower() for part in loc] if isinstance(loc, list) else []
        if len(lowered) != 2 or lowered[0] != "header" or lowered[1] not in expected_headers:
            return f"loopback detail has unexpected location: {loc!r}"
        seen.add(lowered[1])
    if seen != expected_headers:
        return f"loopback auth headers={sorted(seen)!r} (want {sorted(expected_headers)!r})"
    return None

loopback_problem = exact_fiber_v27_route(value.get("loopback_body", ""))
if loopback_problem:
    wrong.append(loopback_problem)
if value.get("loopback_error") not in (None, ""):
    wrong.append(f"loopback_error={value.get('loopback_error')!r}")
def normalized_repo(url):
    if not isinstance(url, str):
        return url
    value = url.rstrip("/")
    return value[:-4] if value.endswith(".git") else value
if normalized_repo(value.get("source_image_repo")) != normalized_repo(source_repo):
    wrong.append(
        f"source_image_repo={value.get('source_image_repo')!r} "
        f"(want manifest repository {source_repo!r})"
    )
if wrong:
    print("BAD\t" + "; ".join(wrong))
else:
    print(
        f"OK\tstable active MainPID {pid_before} owns the listener, matches the "
        "reviewed unit/cwd/argv/import paths, and produced the exact loopback "
        "Fiber v2.7 auth-stage response; source/current pyc carry target x1, "
        "rollback x0, TEXT x1"
    )
PY
)"
PIN_STATE="${PIN_VERDICT%%$'\t'*}"
PIN_DETAIL="${PIN_VERDICT#*$'\t'}"
if [ "$PIN_STATE" = "OK" ]; then
  pass endpoint.pin "${IMAGE_COMMIT:0:12}: $PIN_DETAIL"
else
  fail endpoint.pin "target ${IMAGE_COMMIT}: ${PIN_DETAIL:-source/pyc evidence unreadable}"
fi

check miner.active
MINER_STATE="$(_host miner_state "printf '%s %s' \"\$(systemctl is-active $MINER_UNIT 2>/dev/null)\" \"\$(systemctl show $MINER_UNIT --property=NRestarts --value 2>/dev/null)\"")"
set -- ${MINER_STATE:-}
MSTATE="${1:-}"; MRESTARTS="${2:-}"
if [ "$MSTATE" = "active" ]; then
  pass miner.active "$MINER_UNIT active (NRestarts=${MRESTARTS:-?})"
else
  fail miner.active "$MINER_UNIT is '${MSTATE:-unreadable}' — expected active"
fi

check miner.metagraph_fresh
# Age is measured against the CURRENT MainPID only. A sync that succeeded under
# a previous PID says nothing about the process serving us now, and fiber's
# sync loop swallows exceptions and keeps a stale node set, so a dead sync
# thread is otherwise silent.
MG_AGE="$(_host metagraph "pid=\$(systemctl show $MINER_UNIT -p MainPID --value); journalctl _PID=\"\$pid\" --since '25 minutes ago' --no-pager -o json" | python3 -c '
import json,sys,time
def msg(v): return bytes(v).decode("utf-8","replace") if isinstance(v,list) else str(v)
ts=[int(x["__REALTIME_TIMESTAMP"]) for line in sys.stdin if line.strip()
    for x in [json.loads(line)] if "Successfully synced" in msg(x.get("MESSAGE",""))]
print(int(time.time()-max(ts)/1e6) if ts else -1)
' 2>/dev/null)"
if [[ "${MG_AGE:-}" =~ ^[0-9]+$ ]] && [ "$MG_AGE" -lt "$METAGRAPH_MAX_AGE_S" ]; then
  pass miner.metagraph_fresh "last successful metagraph sync ${MG_AGE}s ago under the current MainPID (< ${METAGRAPH_MAX_AGE_S}s)"
else
  fail miner.metagraph_fresh "no metagraph sync younger than ${METAGRAPH_MAX_AGE_S}s under the current MainPID (age=${MG_AGE:-unreadable}) — sync thread may be wedged"
fi

# =============================================================================
# SECTION 4 — upstream drift
# =============================================================================
echo
echo "== upstream validator drift =="

check upstream.baseline
UPSTREAM_REVIEWED_SHA=""
# shellcheck disable=SC1090
[ -f "$BASELINE_ENV" ] && . "$BASELINE_ENV"
UPSTREAM_HEAD="$(_http_body upstream_head "https://api.github.com/repos/gradients-ai/G.O.D/commits/main" | python3 -c "
import json,sys
try: print(json.load(sys.stdin)['sha'])
except Exception: pass
" 2>/dev/null)"
if [ -z "${UPSTREAM_REVIEWED_SHA:-}" ]; then
  fail upstream.baseline "no UPSTREAM_REVIEWED_SHA recorded in $BASELINE_ENV"
elif [ -z "${UPSTREAM_HEAD:-}" ]; then
  fail upstream.baseline "could not read upstream main SHA for gradients-ai/G.O.D"
elif [ "$UPSTREAM_HEAD" = "$UPSTREAM_REVIEWED_SHA" ]; then
  pass upstream.baseline "gradients-ai/G.O.D main = ${UPSTREAM_HEAD:0:12} matches reviewed baseline (reviewed ${UPSTREAM_REVIEWED_AT:-?})"
else
  fail upstream.baseline "gradients-ai/G.O.D main = ${UPSTREAM_HEAD:0:12} but reviewed baseline is ${UPSTREAM_REVIEWED_SHA:0:12} — re-review the diff, then bump $BASELINE_ENV"
fi

# =============================================================================
# SECTION 5 — host headroom
# =============================================================================
echo
echo "== host headroom =="

check host.disk
DISK_FREE_GIB="$(_host disk "df -BG --output=avail / | tail -1 | tr -dc '0-9'")"
if [[ "${DISK_FREE_GIB:-}" =~ ^[0-9]+$ ]] && [ "$DISK_FREE_GIB" -ge "$DISK_MIN_FREE_GIB" ]; then
  pass host.disk "${DISK_FREE_GIB}GiB free on / (>= ${DISK_MIN_FREE_GIB}GiB)"
else
  fail host.disk "free space on / = ${DISK_FREE_GIB:-unreadable}GiB (want >= ${DISK_MIN_FREE_GIB}GiB)"
fi

check host.memory
MEM_AVAIL_MIB="$(_host mem "free -m | awk '/^Mem:/{print \$7}'")"
if [[ "${MEM_AVAIL_MIB:-}" =~ ^[0-9]+$ ]] && [ "$MEM_AVAIL_MIB" -ge "$MEM_MIN_AVAIL_MIB" ]; then
  pass host.memory "${MEM_AVAIL_MIB}MiB available (>= ${MEM_MIN_AVAIL_MIB}MiB)"
else
  fail host.memory "available memory = ${MEM_AVAIL_MIB:-unreadable}MiB (want >= ${MEM_MIN_AVAIL_MIB}MiB)"
fi

# =============================================================================
# SECTION 6 — watcher armed for the NEXT tournament
# =============================================================================
echo
echo "== watcher =="

check watcher.window
WATCHER_UNITS="$(_host watcher_units "for u in \$(systemctl list-unit-files --no-legend --no-pager 'sn56*watcher*.service' 2>/dev/null | awk '{print \$1}'); do printf '%s\t%s\t%s\n' \"\$u\" \"\$(systemctl is-active \$u 2>/dev/null)\" \"\$(systemctl show \$u -p ExecStart --value 2>/dev/null | tr '\n' ' ')\"; done")"

if [ -z "${WATCHER_UNITS:-}" ]; then
  fail watcher.window "no sn56 watcher units found on the host"
elif [ -z "${NEXT_START_EPOCH:-}" ]; then
  fail watcher.window "cannot validate the capture window without an authoritative tournament start time"
else
  # NOTE: the heredoc below is python3's *script* on stdin, so the unit table is
  # handed over in an environment variable rather than piped.
  # Evaluate against the NEXT start first; if that fails, retry against the
  # CURRENT (in-flight) cycle. A watcher armed for the tournament running RIGHT
  # NOW is correct, not a failure -- the old code only ever checked NEXT, so
  # every mid-tournament run reported a false FAIL. Added 2026-08-17.
  WATCHER_TARGET_EPOCH="$NEXT_START_EPOCH"
  WATCHER_TARGET_LABEL="$NEXT_START_ISO"
  WATCHER_EVAL_PY="$(mktemp -t sn56watchereval)"
  cat >"$WATCHER_EVAL_PY" <<'PY'
import sys, re, os, datetime
start = int(sys.argv[1]); lead_max_h=float(sys.argv[2]); lead_min_m=float(sys.argv[3])
tail_min_h=float(sys.argv[4]); entry_max_h=float(sys.argv[5])
start_dt = datetime.datetime.fromtimestamp(start, datetime.timezone.utc)

def iso(s):
    try: return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    except Exception: return None

def arg(text, flag):
    m = re.search(re.escape(flag) + r"[= ]([^\s\"']+)", text)
    return m.group(1) if m else None

valid = []; active_bad = []; inactive = []
for line in os.environ.get("SN56_WATCHER_UNITS", "").splitlines():
    if not line.strip(): continue
    parts = line.split("\t")
    if len(parts) < 3: continue
    unit, state, execstart = parts[0], parts[1], parts[2]
    if state != "active":
        inactive.append(f"{unit}=inactive"); continue
    sa = iso(arg(execstart, "--start-at") or "")
    hs = iso(arg(execstart, "--hard-stop-at") or "")
    ec = iso(arg(execstart, "--entry-capture-at") or "")
    td = arg(execstart, "--tournament-date")
    problems = []
    if sa is None: problems.append("no --start-at")
    else:
        lead = (start_dt - sa).total_seconds()/3600.0
        if lead < lead_min_m/60.0: problems.append(f"--start-at {sa:%Y-%m-%dT%H:%M:%SZ} is only {lead*60:.0f}min before the start")
        elif lead > lead_max_h: problems.append(f"--start-at {sa:%Y-%m-%dT%H:%M:%SZ} is {lead:.1f}h before the start (>{lead_max_h}h)")
    if hs is None: problems.append("no --hard-stop-at")
    elif (hs - start_dt).total_seconds()/3600.0 < tail_min_h:
        problems.append(f"--hard-stop-at {hs:%Y-%m-%dT%H:%M:%SZ} is under {tail_min_h}h past the start")
    if ec is None: problems.append("no --entry-capture-at")
    else:
        lag = (ec - start_dt).total_seconds()/3600.0
        if lag < 0 or lag > entry_max_h:
            problems.append(f"--entry-capture-at {ec:%Y-%m-%dT%H:%M:%SZ} is {lag:+.1f}h from the start (want 0..{entry_max_h}h)")
    if td != start_dt.strftime("%Y%m%d"):
        problems.append(f"--tournament-date {td} != {start_dt:%Y%m%d}")
    if not problems:
        valid.append(unit)
    else:
        active_bad.append(f"{unit}: " + "; ".join(problems))

if len(valid) == 1 and not active_bad:
    print("OK\t" + valid[0])
else:
    notes = []
    if len(valid) > 1:
        notes.append("multiple valid active watchers: " + ", ".join(valid))
    elif valid:
        notes.append("valid active watcher: " + valid[0])
    notes.extend(active_bad)
    if not valid and not active_bad:
        notes.extend(inactive)
    print("BAD\t" + " | ".join(notes) if notes else "BAD\tno active watcher unit")
PY
  _watcher_eval() {  # $1 = target epoch
    SN56_WATCHER_UNITS="$WATCHER_UNITS" python3 "$WATCHER_EVAL_PY" "$1" \
      "$WATCHER_START_LEAD_MAX_H" "$WATCHER_START_LEAD_MIN_M" \
      "$WATCHER_MIN_TAIL_H" "$WATCHER_ENTRY_MAX_LAG_H" 2>/dev/null
  }
  WATCHER_VERDICT="$(_watcher_eval "$WATCHER_TARGET_EPOCH")"
  WV_STATE="${WATCHER_VERDICT%%$'\t'*}"; WV_DETAIL="${WATCHER_VERDICT#*$'\t'}"
  WATCHER_SCOPE="next"
  # Only accept the CURRENT cycle on tournament DAY (< 24h since its start).
  # On tournament day, "armed for today" is correct and must not read as FAIL.
  # Past that the cycle's capture is done and the duty is to arm for NEXT, so we
  # deliberately fall through to the FAIL -- being pointed at a finished
  # tournament with days of lead time is a real defect (selftest scenario 4).
  CUR_AGE_H=999
  if [ -n "${CUR_START_EPOCH:-}" ]; then
    CUR_AGE_H="$(python3 -c "import sys;print(int((int(sys.argv[1])-int(sys.argv[2]))//3600))" "$NOW_EPOCH" "$CUR_START_EPOCH" 2>/dev/null || echo 999)"
  fi
  if [ "$WV_STATE" != "OK" ] && [ -n "${CUR_START_EPOCH:-}" ] && [ "${CUR_AGE_H:-999}" -lt 24 ] 2>/dev/null; then
    WATCHER_TARGET_EPOCH="$CUR_START_EPOCH"
    WATCHER_TARGET_LABEL="$CUR_START_ISO"
    WV2="$(_watcher_eval "$CUR_START_EPOCH")"
    if [ -n "$WV2" ] && [ "${WV2%%$'\t'*}" = "OK" ]; then
      WV_STATE="OK"; WV_DETAIL="${WV2#*$'\t'}"; WATCHER_SCOPE="current"
    fi
  fi
  rm -f "$WATCHER_EVAL_PY" 2>/dev/null || true
  if [ "$WV_STATE" = "OK" ]; then
    if [ "$WATCHER_SCOPE" = "current" ]; then
      HRS_TO_NEXT="$(python3 -c "import sys;print(f'{(int(sys.argv[1])-int(sys.argv[2]))/3600:.0f}')" "$NEXT_START_EPOCH" "$NOW_EPOCH" 2>/dev/null)"
      if [ "${HRS_TO_NEXT:-999}" -le 48 ] 2>/dev/null; then
        warn watcher.window "$WV_DETAIL is armed for the IN-FLIGHT cycle $CUR_START_ISO, but the next start $NEXT_START_ISO is only ${HRS_TO_NEXT}h away -- RE-ARM for it"
      else
        pass watcher.window "$WV_DETAIL is active and armed for the in-flight cycle $CUR_START_ISO (re-arm for $NEXT_START_ISO before it starts)"
      fi
    else
      pass watcher.window "$WV_DETAIL is active and armed for $NEXT_START_ISO"
    fi
  else
    fail watcher.window "no active watcher armed for $NEXT_START_ISO — $WV_DETAIL"
  fi
fi

# =============================================================================
# summary
# =============================================================================
echo
if [ -n "$JSON_OUT" ]; then
  {
    printf '{\n  "generated_at_utc": "%s",\n' "$(python3 -c "import datetime,sys;print(datetime.datetime.fromtimestamp(int(sys.argv[1]),datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))" "$NOW_EPOCH")"
    python3 -c '
import json,sys
labels=("manifest","release_state","manifest_sha256","docker_policy_sha256","readiness_receipt_sha256","target_commit")
for label,value in zip(labels,sys.argv[1:]):
    print("  " + json.dumps(label) + ": " + json.dumps(value) + ",")
' "$MANIFEST" "$RELEASE_STATE" "$MANIFEST_SHA256" "$DOCKER_POLICY_SHA256" "$READINESS_SHA256" "$IMAGE_COMMIT"
    printf '  "mode": "%s",\n  "next_tournament_start_utc": "%s",\n  "fails": %d,\n  "warns": %d,\n  "checks": [\n' \
      "$MODE" "${NEXT_START_ISO:-unknown}" "$FAILS" "$WARNS"
    for i in "${!RESULT_IDS[@]}"; do
      sep=","; [ "$i" -eq $(( ${#RESULT_IDS[@]} - 1 )) ] && sep=""
      python3 -c "
import json,sys
print('    ' + json.dumps({'id':sys.argv[1],'state':sys.argv[2],'detail':sys.argv[3]}) + sys.argv[4])
" "${RESULT_IDS[$i]}" "${RESULT_STATES[$i]}" "${RESULT_DETAILS[$i]}" "$sep"
    done
    printf '  ]\n}\n'
  } > "$JSON_OUT"
  echo "wrote $JSON_OUT"
fi

if [ "$FAILS" -eq 0 ]; then
  echo "ALL GREEN (${WARNS} warn) — next image tournament ${NEXT_START_ISO:-unknown}. Touch nothing."
  exit 0
fi
echo "$FAILS FAILURE(S), $WARNS warn — resolve before ${NEXT_START_ISO:-the next start}."
[ "$FAILS" -gt 125 ] && exit 125
exit "$FAILS"
