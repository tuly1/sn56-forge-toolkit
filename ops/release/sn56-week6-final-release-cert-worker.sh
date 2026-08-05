#!/bin/bash
set -Eeuo pipefail
umask 077

# Archived Week-6 authority worker.  The public launcher executes this exact
# committed copy only after it has independently bound and materialized the
# selected release commit.

readonly FIXED_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH=${FIXED_PATH}
unset BASH_ENV CDPATH ENV GIT_CONFIG GIT_DIR GIT_WORK_TREE \
  PYTHONHOME PYTHONPATH PYTHONSTARTUP
readonly CERT_SCOPE=toolkit-krea-only
# Updated mechanically after both archived authority programs are final.
readonly VALIDATOR_SHA256=d1e148479b9a4bf8bfd48348f71100145f39c5233a9a90ae4731c40cef571ed0
readonly DELEGATED_SHA256=222df447fdc8820702a7837bb9842f4df128a97348f5d20a16b3cb194e953010

fail() {
  /usr/bin/printf 'SN56_WEEK6_FINAL_RELEASE_CERT=FAIL reason=%s\n' "$1" >&2
  exit "${2-1}"
}

require_env() {
  local name=$1
  [[ -n ${!name-} ]] || fail "required worker environment variable is unset: ${name}" 64
  [[ ${!name} != *$'\n'* && ${!name} != *$'\r'* ]] || \
    fail "worker environment variable contains a line break: ${name}" 64
}

for required_name in \
  SN56_RELEASE_CERT_MODE \
  SN56_RELEASE_COMMIT \
  SN56_RELEASE_TREE \
  SN56_RELEASE_FORGE_TREE \
  SN56_RELEASE_SOURCE_CHECKOUT \
  SN56_RELEASE_EXPECTED_ORIGIN_URL \
  SN56_RELEASE_REMOTE_REF \
  SN56_RELEASE_EVIDENCE_NAMESPACE \
  SN56_RELEASE_DELEGATE_EVIDENCE_BASE \
  SN56_RELEASE_ENVELOPE_BASE \
  SN56_RELEASE_WORK_BASE \
  SN56_RELEASE_PRIVATE_WORKSPACE \
  SN56_RELEASE_MATERIALIZED_SOURCE \
  SN56_RELEASE_MATERIALIZED_MANIFEST_SHA256 \
  SN56_RELEASE_ARCHIVE_SHA256 \
  SN56_RELEASE_TOOLKIT_IMAGE_TAG \
  SN56_RELEASE_LEGACY_IMAGE_TAG \
  SN56_RELEASE_EXPECTED_DOCKER_ROOT \
  SN56_RELEASE_EXPECTED_CONTAINERD_ROOT \
  SN56_RELEASE_TIMING_PROFILE \
  SN56_RELEASE_TIMING_PROFILE_SHA256 \
  SN56_RELEASE_TIMING_SOURCE_RECORD \
  SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256 \
  SN56_RELEASE_TIMING_TERMINAL_ARTIFACT \
  SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256 \
  SN56_RELEASE_FRIDAY_GATE_LOG \
  SN56_RELEASE_FRIDAY_GATE_LOG_SHA256 \
  SN56_RELEASE_TIMING_SOURCE_RUN_ID \
  SN56_RELEASE_H100_GATE_SESSION_ID \
  SN56_RELEASE_H100_RENTAL_STARTED_AT_UTC \
  SN56_RELEASE_H100_RENTAL_ENDED_AT_UTC \
  SN56_RELEASE_TIMING_BUNDLE_ID \
  SN56_RELEASE_TIMING_BUNDLE_SHA256 \
  SN56_RELEASE_TIMING_MODEL_TYPE \
  SN56_RELEASE_TIMING_CURRENT_DATASET_SIZE \
  SN56_RELEASE_TIMING_DATASET_REGIME \
  SN56_RELEASE_TIMING_ACCELERATOR_IDENTITY
do
  require_env "${required_name}"
done

[[ $# -eq 0 ]] || fail 'the archived release worker accepts no arguments' 64
[[ ${SN56_RELEASE_CERT_MODE} == production || \
   ${SN56_RELEASE_CERT_MODE} == cpu-integration ]] || \
  fail 'invalid archived release worker mode' 64
for identity in \
  "${SN56_RELEASE_COMMIT}" "${SN56_RELEASE_TREE}" "${SN56_RELEASE_FORGE_TREE}"
do
  [[ ${identity} =~ ^[0-9a-f]{40}$ ]] || fail 'invalid archived release Git identity' 64
done
for digest in \
  "${SN56_RELEASE_MATERIALIZED_MANIFEST_SHA256}" \
  "${SN56_RELEASE_ARCHIVE_SHA256}" \
  "${SN56_RELEASE_TIMING_PROFILE_SHA256}" \
  "${SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256}" \
  "${SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256}" \
  "${SN56_RELEASE_FRIDAY_GATE_LOG_SHA256}" \
  "${SN56_RELEASE_TIMING_BUNDLE_SHA256}"
do
  [[ ${digest} =~ ^[0-9a-f]{64}$ ]] || fail 'invalid archived release SHA-256 value' 64
done
[[ ${SN56_RELEASE_EVIDENCE_NAMESPACE} =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || \
  fail 'invalid archived release evidence namespace' 64

readonly RELEASE_MODE=${SN56_RELEASE_CERT_MODE}
readonly RELEASE_COMMIT=${SN56_RELEASE_COMMIT}
readonly RELEASE_TREE=${SN56_RELEASE_TREE}
readonly FORGE_TREE=${SN56_RELEASE_FORGE_TREE}
readonly NAMESPACE=${SN56_RELEASE_EVIDENCE_NAMESPACE}
readonly SOURCE=${SN56_RELEASE_MATERIALIZED_SOURCE}
readonly SOURCE_MANIFEST_SHA256=${SN56_RELEASE_MATERIALIZED_MANIFEST_SHA256}
readonly SOURCE_ARCHIVE_SHA256=${SN56_RELEASE_ARCHIVE_SHA256}
readonly PRIVATE_WORKSPACE=${SN56_RELEASE_PRIVATE_WORKSPACE}

# A directly invoked worker must not be able to turn an arbitrary path into the
# recursive cleanup target or its authority source.
/usr/bin/env -i \
  PATH="${FIXED_PATH}" HOME=/nonexistent LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /usr/bin/python3 - \
  "${SN56_RELEASE_WORK_BASE}" "${PRIVATE_WORKSPACE}" "${SOURCE}" <<'PY'
import os
from pathlib import Path
import re
import stat
import sys

def direct_directory(raw: str, label: str) -> Path:
    if not raw.startswith(os.sep) or os.path.normpath(raw) != raw:
        raise SystemExit(f"{label} is not a normalized absolute path")
    path = Path(raw)
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError as exc:
        raise SystemExit(f"{label} is unavailable: {exc}")
    if resolved != path or not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise SystemExit(f"{label} is indirect or not a directory")
    return path

if len(sys.argv) != 4:
    raise SystemExit("archived worker path preflight argument error")
base = direct_directory(sys.argv[1], "release work base")
workspace = direct_directory(sys.argv[2], "private release workspace")
source = direct_directory(sys.argv[3], "materialized release source")
if workspace.parent != base or re.fullmatch(r"sn56-week6-release\.[A-Za-z0-9]{8}", workspace.name) is None:
    raise SystemExit("private release workspace is outside the fixed work base")
if source != workspace / "source":
    raise SystemExit("materialized source is outside the private release workspace")
PY

cleanup() {
  /bin/chmod -R u+w "${PRIVATE_WORKSPACE}" 2>/dev/null || :
  /bin/rm -rf -- "${PRIVATE_WORKSPACE}"
}
trap cleanup EXIT HUP INT TERM

readonly AUTHORITY_STAGE=${PRIVATE_WORKSPACE}/authority
readonly TIMING_STAGE=${PRIVATE_WORKSPACE}/timing-evidence
/bin/mkdir -m 0700 "${AUTHORITY_STAGE}" "${TIMING_STAGE}" || \
  fail 'private worker stages could not be created'

readonly VALIDATOR_SOURCE=${SOURCE}/ops/release/sn56-week6-validate-timing-provenance.py
readonly DELEGATED_SOURCE=${SOURCE}/ops/release/sn56-week6-build-gpu-cert.py
readonly PINNED_VALIDATOR=${AUTHORITY_STAGE}/validator.py
readonly PINNED_DELEGATED=${AUTHORITY_STAGE}/build-gpu-cert.py

# Open each archived authority program once, hash that descriptor, and execute
# only its private copy.  This avoids a hash-then-reopen substitution window.
/usr/bin/env -i \
  PATH="${FIXED_PATH}" HOME=/nonexistent LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /usr/bin/python3 - \
  "${VALIDATOR_SOURCE}" "${VALIDATOR_SHA256}" "${PINNED_VALIDATOR}" \
  "${DELEGATED_SOURCE}" "${DELEGATED_SHA256}" "${PINNED_DELEGATED}" <<'PY'
import hashlib
import os
from pathlib import Path
import stat
import sys

def stage(source: str, expected: str, destination: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise SystemExit(f"authority program could not be opened: {source}: {exc}")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise SystemExit(f"authority program is not a nonempty regular file: {source}")
        digest = hashlib.sha256()
        chunks = []
        consumed = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
            consumed += len(block)
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
        )
        if (
            consumed != before.st_size
            or identity(before) != identity(after)
            or digest.hexdigest() != expected
        ):
            raise SystemExit(f"authority program identity/hash mismatch: {source}")
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    target = Path(destination)
    output = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o500,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(output, view)
            if written <= 0:
                raise SystemExit("authority program staging write failed")
            view = view[written:]
        os.fsync(output)
    finally:
        os.close(output)

values = sys.argv[1:]
if len(values) != 6:
    raise SystemExit("authority bootstrap argument error")
stage(*values[0:3])
stage(*values[3:6])
PY

validator() {
  /usr/bin/env -i \
    PATH="${FIXED_PATH}" \
    HOME=/nonexistent \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    GIT_CONFIG_NOSYSTEM=1 \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_NO_REPLACE_OBJECTS=1 \
    GIT_TERMINAL_PROMPT=0 \
    /usr/bin/python3 "${PINNED_VALIDATOR}" "$@"
}

stage_evidence() {
  local source=$1
  local expected=$2
  local destination=$3
  local label=$4
  local maximum=${5-}
  local arguments=(
    --stage-source "${source}"
    --stage-destination "${destination}"
    --stage-sha256 "${expected}"
    --stage-label "${label}"
  )
  if [[ -n ${maximum} ]]; then
    arguments+=(--stage-maximum-bytes "${maximum}")
  fi
  validator "${arguments[@]}" >"${AUTHORITY_STAGE}/last-stage.stdout"
}

hash_regular_file() {
  /usr/bin/env -i \
    PATH="${FIXED_PATH}" HOME=/nonexistent LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    /usr/bin/python3 - "$1" <<'PY'
import hashlib
import os
import stat
import sys

flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(sys.argv[1], flags)
try:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
        raise SystemExit("hash input is not a nonempty regular file")
    digest = hashlib.sha256()
    consumed = 0
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        consumed += len(block)
        digest.update(block)
    after = os.fstat(descriptor)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
    )
    if consumed != before.st_size or identity(before) != identity(after):
        raise SystemExit("hash input changed while read")
    print(digest.hexdigest())
finally:
    os.close(descriptor)
PY
}

readonly STAGED_PROFILE=${TIMING_STAGE}/timing-profile.json
readonly STAGED_SOURCE_RECORD=${TIMING_STAGE}/timing-source-record.json
readonly STAGED_ARTIFACT=${TIMING_STAGE}/terminal-artifact.safetensors
readonly STAGED_GATE_LOG=${TIMING_STAGE}/friday-h100-gate-log.jsonl
readonly RECEIPT=${TIMING_STAGE}/timing-provenance-receipt.json
readonly POLICY_RECEIPT=${TIMING_STAGE}/reviewed-release-timing-policy.json

stage_evidence "${SN56_RELEASE_TIMING_PROFILE}" \
  "${SN56_RELEASE_TIMING_PROFILE_SHA256}" "${STAGED_PROFILE}" \
  'timing profile' 65536
stage_evidence "${SN56_RELEASE_TIMING_SOURCE_RECORD}" \
  "${SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256}" "${STAGED_SOURCE_RECORD}" \
  'timing source record' 1048576
stage_evidence "${SN56_RELEASE_TIMING_TERMINAL_ARTIFACT}" \
  "${SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256}" "${STAGED_ARTIFACT}" \
  'terminal artifact'
stage_evidence "${SN56_RELEASE_FRIDAY_GATE_LOG}" \
  "${SN56_RELEASE_FRIDAY_GATE_LOG_SHA256}" "${STAGED_GATE_LOG}" \
  'Friday H100 gate log' 16777216

validator \
  --profile "${STAGED_PROFILE}" \
  --profile-file-sha256 "${SN56_RELEASE_TIMING_PROFILE_SHA256}" \
  --raw-record "${STAGED_SOURCE_RECORD}" \
  --raw-record-file-sha256 "${SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256}" \
  --terminal-artifact "${STAGED_ARTIFACT}" \
  --archived-terminal-artifact "${STAGED_ARTIFACT}" \
  --terminal-artifact-file-sha256 "${SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256}" \
  --gate-log "${STAGED_GATE_LOG}" \
  --gate-log-file-sha256 "${SN56_RELEASE_FRIDAY_GATE_LOG_SHA256}" \
  --source-run-id "${SN56_RELEASE_TIMING_SOURCE_RUN_ID}" \
  --gate-session-id "${SN56_RELEASE_H100_GATE_SESSION_ID}" \
  --rental-started-at-utc "${SN56_RELEASE_H100_RENTAL_STARTED_AT_UTC}" \
  --rental-ended-at-utc "${SN56_RELEASE_H100_RENTAL_ENDED_AT_UTC}" \
  --forge-repository "${SOURCE}" \
  --forge-materialized-manifest-sha256 "${SOURCE_MANIFEST_SHA256}" \
  --forge-commit "${RELEASE_COMMIT}" \
  --release-tree "${RELEASE_TREE}" \
  --certificate-scope "${CERT_SCOPE}" \
  --bundle-id "${SN56_RELEASE_TIMING_BUNDLE_ID}" \
  --bundle-sha256 "${SN56_RELEASE_TIMING_BUNDLE_SHA256}" \
  --model-type "${SN56_RELEASE_TIMING_MODEL_TYPE}" \
  --current-dataset-size "${SN56_RELEASE_TIMING_CURRENT_DATASET_SIZE}" \
  --dataset-regime "${SN56_RELEASE_TIMING_DATASET_REGIME}" \
  --accelerator-identity "${SN56_RELEASE_TIMING_ACCELERATOR_IDENTITY}" \
  --receipt "${RECEIPT}" \
  >"${AUTHORITY_STAGE}/timing-validation.stdout"

validator \
  --assert-receipt "${RECEIPT}" \
  --forge-repository "${SOURCE}" \
  --forge-materialized-manifest-sha256 "${SOURCE_MANIFEST_SHA256}" \
  --forge-commit "${RELEASE_COMMIT}" \
  --release-tree "${RELEASE_TREE}" \
  --certificate-scope "${CERT_SCOPE}" \
  --profile-file-sha256 "${SN56_RELEASE_TIMING_PROFILE_SHA256}" \
  --raw-record-file-sha256 "${SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256}" \
  --terminal-artifact-file-sha256 "${SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256}" \
  --gate-log-file-sha256 "${SN56_RELEASE_FRIDAY_GATE_LOG_SHA256}" \
  >"${AUTHORITY_STAGE}/timing-receipt-assertion.stdout"

validator \
  --assert-release-policy \
  --forge-repository "${SOURCE}" \
  --forge-materialized-manifest-sha256 "${SOURCE_MANIFEST_SHA256}" \
  --forge-commit "${RELEASE_COMMIT}" \
  --release-tree "${RELEASE_TREE}" \
  --receipt "${POLICY_RECEIPT}" \
  >"${AUTHORITY_STAGE}/timing-policy-assertion.stdout"

readonly DELEGATED_EVIDENCE=${SN56_RELEASE_DELEGATE_EVIDENCE_BASE}/${NAMESPACE}
readonly DELEGATED_STDOUT=${AUTHORITY_STAGE}/delegate.stdout
readonly DELEGATED_STDERR=${AUTHORITY_STAGE}/delegate.stderr
if ! /usr/bin/env -i \
  PATH="${FIXED_PATH}" \
  HOME=/nonexistent \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  GIT_CONFIG_NOSYSTEM=1 \
  GIT_CONFIG_GLOBAL=/dev/null \
  GIT_NO_REPLACE_OBJECTS=1 \
  GIT_TERMINAL_PROMPT=0 \
  /usr/bin/python3 "${PINNED_DELEGATED}" \
    --mode "${RELEASE_MODE}" \
    --source-checkout "${SN56_RELEASE_SOURCE_CHECKOUT}" \
    --release-commit "${RELEASE_COMMIT}" \
    --release-tree "${RELEASE_TREE}" \
    --forge-tree "${FORGE_TREE}" \
    --certificate-scope "${CERT_SCOPE}" \
    --evidence-base "${SN56_RELEASE_DELEGATE_EVIDENCE_BASE}" \
    --evidence-namespace "${NAMESPACE}" \
    --work-base "${SN56_RELEASE_WORK_BASE}" \
    --toolkit-image-tag "${SN56_RELEASE_TOOLKIT_IMAGE_TAG}" \
    --legacy-image-tag "${SN56_RELEASE_LEGACY_IMAGE_TAG}" \
    --expected-docker-root "${SN56_RELEASE_EXPECTED_DOCKER_ROOT}" \
    --expected-containerd-root "${SN56_RELEASE_EXPECTED_CONTAINERD_ROOT}" \
    >"${DELEGATED_STDOUT}" 2>"${DELEGATED_STDERR}"
then
  /bin/cat "${DELEGATED_STDERR}" >&2 || :
  fail 'generic Week-6 build/GPU delegate failed'
fi

if [[ ${RELEASE_MODE} == production ]]; then
  readonly RESULT_STATE=PASS
else
  readonly RESULT_STATE=DRY_RUN_PASS
fi

/usr/bin/env -i \
  PATH="${FIXED_PATH}" HOME=/nonexistent LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /usr/bin/python3 - \
  "${DELEGATED_STDOUT}" "${RESULT_STATE}" "${DELEGATED_EVIDENCE}" <<'PY'
from pathlib import Path
import sys

actual = Path(sys.argv[1]).read_bytes()
expected = f"SN56_WEEK6_BUILD_GPU_CERT={sys.argv[2]} evidence={sys.argv[3]}\n".encode("utf-8")
if actual != expected:
    raise SystemExit("generic delegate stdout differs from its exact result contract")
PY

readonly DELEGATED_RESULT=${DELEGATED_EVIDENCE}/result.env
readonly DELEGATED_MANIFEST=${DELEGATED_EVIDENCE}/MANIFEST.sha256
[[ -d ${DELEGATED_EVIDENCE} && ! -L ${DELEGATED_EVIDENCE} ]] || \
  fail 'delegated evidence directory is absent or indirect'
[[ -f ${DELEGATED_RESULT} && ! -L ${DELEGATED_RESULT} ]] || \
  fail 'delegated result.env is absent or indirect'
[[ -f ${DELEGATED_MANIFEST} && ! -L ${DELEGATED_MANIFEST} ]] || \
  fail 'delegated evidence manifest is absent or indirect'

validator \
  --assert-result-env "${DELEGATED_RESULT}" \
  --forge-commit "${RELEASE_COMMIT}" \
  --release-tree "${RELEASE_TREE}" \
  --forge-tree "${FORGE_TREE}" \
  --certificate-scope "${CERT_SCOPE}" \
  --delegated-mode "${RELEASE_MODE}" \
  --source-archive-sha256 "${SOURCE_ARCHIVE_SHA256}" \
  --source-manifest-sha256 "${SOURCE_MANIFEST_SHA256}" \
  >"${AUTHORITY_STAGE}/delegated-result-assertion.stdout"

# Verify every delegated evidence entry without trusting sha256sum path parsing.
delegated_manifest_sha256=$(
  /usr/bin/env -i \
    PATH="${FIXED_PATH}" HOME=/nonexistent LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    /usr/bin/python3 - "${DELEGATED_EVIDENCE}" <<'PY'
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys

root = Path(sys.argv[1])
if not root.is_absolute() or root.resolve(strict=True) != root or root.is_symlink():
    raise SystemExit("delegated evidence path is indirect")

def read_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit(f"delegated evidence entry is irregular: {path}")
        digest_input = []
        consumed = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            consumed += len(block)
            digest_input.append(block)
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
        )
        if consumed != before.st_size or identity(before) != identity(after):
            raise SystemExit(f"delegated evidence entry changed: {path}")
        return b"".join(digest_input)
    finally:
        os.close(descriptor)

manifest_path = root / "MANIFEST.sha256"
manifest = read_regular(manifest_path)
if not manifest or len(manifest) > 16 * 1024 * 1024:
    raise SystemExit("delegated manifest is empty or oversized")
try:
    text = manifest.decode("ascii")
except UnicodeError as exc:
    raise SystemExit("delegated manifest is not ASCII") from exc
declared: dict[str, str] = {}
for line_number, row in enumerate(text.splitlines(keepends=True), 1):
    match = re.fullmatch(r"([0-9a-f]{64})  (.+)\n", row)
    if match is None:
        raise SystemExit(f"delegated manifest row is malformed: {line_number}")
    relative = PurePosixPath(match.group(2))
    name = relative.as_posix()
    if (
        relative.is_absolute()
        or name in {"", ".", "MANIFEST.sha256"}
        or name != match.group(2)
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise SystemExit(f"delegated manifest path is unsafe: {line_number}")
    if name in declared:
        raise SystemExit(f"delegated manifest path is duplicated: {name}")
    declared[name] = match.group(1)

actual = set()
for candidate in sorted(root.rglob("*")):
    relative = candidate.relative_to(root).as_posix()
    if candidate.is_symlink():
        raise SystemExit(f"delegated evidence contains a symlink: {relative}")
    if candidate.is_dir():
        continue
    if not candidate.is_file():
        raise SystemExit(f"delegated evidence contains an irregular entry: {relative}")
    if relative == "MANIFEST.sha256":
        continue
    actual.add(relative)
    expected = declared.get(relative)
    if expected is None or hashlib.sha256(read_regular(candidate)).hexdigest() != expected:
        raise SystemExit(f"delegated evidence hash differs: {relative}")
if actual != set(declared):
    raise SystemExit("delegated manifest file set differs")
print(hashlib.sha256(manifest).hexdigest())
PY
) || fail 'delegated evidence manifest verification failed'
[[ ${delegated_manifest_sha256} =~ ^[0-9a-f]{64}$ ]] || \
  fail 'delegated evidence manifest hash is malformed'

receipt_sha256=$(hash_regular_file "${RECEIPT}") || fail 'timing receipt could not be hashed'
policy_sha256=$(hash_regular_file "${POLICY_RECEIPT}") || fail 'policy receipt could not be hashed'
delegated_result_sha256=$(hash_regular_file "${DELEGATED_RESULT}") || \
  fail 'delegated result could not be hashed'

prepare_output=$(
  validator \
    --prepare-envelope-base "${SN56_RELEASE_ENVELOPE_BASE}" \
    --prepare-envelope-namespace "${NAMESPACE}"
) || fail 'outer evidence envelope could not be prepared'
case ${prepare_output} in
  SN56_ENVELOPE_STAGE=/*) ENVELOPE_STAGE=${prepare_output#SN56_ENVELOPE_STAGE=} ;;
  *) fail 'outer envelope preparation returned an invalid path' ;;
esac
readonly ENVELOPE_STAGE

stage_evidence "${STAGED_PROFILE}" "${SN56_RELEASE_TIMING_PROFILE_SHA256}" \
  "${ENVELOPE_STAGE}/timing-profile.json" 'envelope timing profile' 65536
stage_evidence "${STAGED_SOURCE_RECORD}" "${SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256}" \
  "${ENVELOPE_STAGE}/timing-source-record.json" 'envelope timing source record' 1048576
stage_evidence "${STAGED_ARTIFACT}" "${SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256}" \
  "${ENVELOPE_STAGE}/terminal-artifact.safetensors" 'envelope terminal artifact'
stage_evidence "${STAGED_GATE_LOG}" "${SN56_RELEASE_FRIDAY_GATE_LOG_SHA256}" \
  "${ENVELOPE_STAGE}/friday-h100-gate-log.jsonl" 'envelope Friday gate log' 16777216
stage_evidence "${RECEIPT}" "${receipt_sha256}" \
  "${ENVELOPE_STAGE}/timing-provenance-receipt.json" 'envelope timing receipt' 262144
stage_evidence "${POLICY_RECEIPT}" "${policy_sha256}" \
  "${ENVELOPE_STAGE}/reviewed-release-timing-policy.json" 'envelope policy receipt' 65536
stage_evidence "${DELEGATED_RESULT}" "${delegated_result_sha256}" \
  "${ENVELOPE_STAGE}/delegated-result.env" 'envelope delegated result' 65536
stage_evidence "${DELEGATED_MANIFEST}" "${delegated_manifest_sha256}" \
  "${ENVELOPE_STAGE}/delegated-MANIFEST.sha256" 'envelope delegated manifest' 16777216

# Create fixed, data-only authority records without shell evaluation or append
# semantics.  The state is derived from the selected mode, not candidate data.
/usr/bin/env -i \
  PATH="${FIXED_PATH}" HOME=/nonexistent LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /usr/bin/python3 - \
  "${ENVELOPE_STAGE}" "${RESULT_STATE}" "${RELEASE_MODE}" "${CERT_SCOPE}" \
  "${RELEASE_COMMIT}" "${RELEASE_TREE}" "${FORGE_TREE}" \
  "${SOURCE_ARCHIVE_SHA256}" "${SOURCE_MANIFEST_SHA256}" \
  "${SN56_RELEASE_EXPECTED_ORIGIN_URL}" "${SN56_RELEASE_REMOTE_REF}" \
  "${SN56_RELEASE_TIMING_PROFILE_SHA256}" \
  "${SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256}" \
  "${SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256}" \
  "${SN56_RELEASE_FRIDAY_GATE_LOG_SHA256}" \
  "${receipt_sha256}" "${policy_sha256}" "${DELEGATED_EVIDENCE}" \
  "${delegated_result_sha256}" "${delegated_manifest_sha256}" <<'PY'
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

if len(sys.argv) != 21:
    raise SystemExit("outer result construction argument error")
stage = Path(sys.argv[1])
fields = {
    "schema": "sn56.week6.final-release-cert-envelope.v3",
    "state": sys.argv[2],
    "mode": sys.argv[3],
    "certificate_scope": sys.argv[4],
    "source_commit": sys.argv[5],
    "source_tree": sys.argv[6],
    "forge_tree": sys.argv[7],
    "source_archive_sha256": sys.argv[8],
    "source_manifest_sha256": sys.argv[9],
    "expected_origin_url": sys.argv[10],
    "remote_ref": sys.argv[11],
    "timing_evidence_scope": "lab-only",
    "production_timing_input": "reviewed-conservative-constant",
    "profile_file_sha256": sys.argv[12],
    "raw_record_file_sha256": sys.argv[13],
    "terminal_artifact_file_sha256": sys.argv[14],
    "gate_log_file_sha256": sys.argv[15],
    "timing_receipt_file_sha256": sys.argv[16],
    "release_policy_receipt_file_sha256": sys.argv[17],
    "delegated_evidence": sys.argv[18],
    "delegated_result_file_sha256": sys.argv[19],
    "delegated_manifest_sha256": sys.argv[20],
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
for name, value in fields.items():
    if not value or "\n" in value or "\r" in value or "\x00" in value:
        raise SystemExit(f"outer result field is invalid: {name}")
payload = "".join(f"{name}={value}\n" for name, value in fields.items()).encode("utf-8")
target = stage / "result.env"
descriptor = os.open(
    target,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
    0o440,
)
try:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise SystemExit("outer result write made no progress")
        view = view[written:]
    os.fsync(descriptor)
finally:
    os.close(descriptor)

check_payload = (
    f"state={sys.argv[2]}\n"
    f"delegated_evidence={sys.argv[18]}\n"
    f"delegated_result_file_sha256={sys.argv[19]}\n"
    f"delegated_manifest_sha256={sys.argv[20]}\n"
).encode("utf-8")
check_target = stage / "delegated-manifest-check.txt"
descriptor = os.open(
    check_target,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
    0o440,
)
try:
    os.write(descriptor, check_payload)
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY

# Generate the outer manifest privately, then make the complete tree immutable
# before asking the validator's no-replace publisher to fsync and rename it.
outer_manifest_sha256=$(
  /usr/bin/env -i \
    PATH="${FIXED_PATH}" HOME=/nonexistent LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    /usr/bin/python3 - "${ENVELOPE_STAGE}" <<'PY'
import hashlib
import os
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1])
if not root.is_absolute() or root.resolve(strict=True) != root or root.is_symlink():
    raise SystemExit("outer envelope stage is indirect")
rows = []
for candidate in sorted(root.rglob("*")):
    relative = candidate.relative_to(root).as_posix()
    if candidate.is_symlink():
        raise SystemExit(f"outer envelope contains a symlink: {relative}")
    if candidate.is_dir():
        continue
    if not candidate.is_file() or relative == "MANIFEST.sha256":
        raise SystemExit(f"outer envelope contains an invalid entry: {relative}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit(f"outer envelope entry is irregular: {relative}")
        digest = hashlib.sha256()
        consumed = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            consumed += len(block)
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
        )
        if consumed != before.st_size or identity(before) != identity(after):
            raise SystemExit(f"outer envelope entry changed: {relative}")
    finally:
        os.close(descriptor)
    rows.append(f"{digest.hexdigest()}  {relative}\n")
payload = "".join(rows).encode("ascii")
if not payload:
    raise SystemExit("outer envelope manifest would be empty")
target = root / "MANIFEST.sha256"
descriptor = os.open(
    target,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
    0o440,
)
try:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise SystemExit("outer manifest write made no progress")
        view = view[written:]
    os.fsync(descriptor)
finally:
    os.close(descriptor)
for candidate in sorted(root.rglob("*"), reverse=True):
    os.chmod(candidate, 0o550 if candidate.is_dir() else 0o440)
os.chmod(root, 0o550)
print(hashlib.sha256(payload).hexdigest())
PY
) || fail 'outer evidence manifest generation failed'
[[ ${outer_manifest_sha256} =~ ^[0-9a-f]{64}$ ]] || \
  fail 'outer evidence manifest hash is malformed'

publish_output=$(
  validator \
    --publish-envelope-base "${SN56_RELEASE_ENVELOPE_BASE}" \
    --publish-envelope-namespace "${NAMESPACE}" \
    --publish-envelope-stage "${ENVELOPE_STAGE}"
) || fail 'outer evidence envelope could not be published atomically'
readonly ENVELOPE=${SN56_RELEASE_ENVELOPE_BASE}/${NAMESPACE}
[[ ${publish_output} == "SN56_ENVELOPE_PUBLISHED=${ENVELOPE}" ]] || \
  fail 'outer envelope publisher returned an unexpected destination'

validator \
  --assert-file "${ENVELOPE}/MANIFEST.sha256" \
  --assert-file-sha256 "${outer_manifest_sha256}" \
  --stage-label 'published outer manifest' \
  >"${AUTHORITY_STAGE}/published-manifest-assertion.stdout"
validator \
  --assert-receipt "${ENVELOPE}/timing-provenance-receipt.json" \
  --forge-repository "${SOURCE}" \
  --forge-materialized-manifest-sha256 "${SOURCE_MANIFEST_SHA256}" \
  --forge-commit "${RELEASE_COMMIT}" \
  --release-tree "${RELEASE_TREE}" \
  --certificate-scope "${CERT_SCOPE}" \
  --profile-file-sha256 "${SN56_RELEASE_TIMING_PROFILE_SHA256}" \
  --raw-record-file-sha256 "${SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256}" \
  --terminal-artifact-file-sha256 "${SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256}" \
  --gate-log-file-sha256 "${SN56_RELEASE_FRIDAY_GATE_LOG_SHA256}" \
  >"${AUTHORITY_STAGE}/published-receipt-assertion.stdout"
validator \
  --assert-result-env "${ENVELOPE}/delegated-result.env" \
  --forge-commit "${RELEASE_COMMIT}" \
  --release-tree "${RELEASE_TREE}" \
  --forge-tree "${FORGE_TREE}" \
  --certificate-scope "${CERT_SCOPE}" \
  --delegated-mode "${RELEASE_MODE}" \
  --source-archive-sha256 "${SOURCE_ARCHIVE_SHA256}" \
  --source-manifest-sha256 "${SOURCE_MANIFEST_SHA256}" \
  >"${AUTHORITY_STAGE}/published-delegate-assertion.stdout"

/usr/bin/printf 'SN56_WEEK6_FINAL_RELEASE_CERT=%s evidence=%s\n' \
  "${RESULT_STATE}" "${ENVELOPE}"
