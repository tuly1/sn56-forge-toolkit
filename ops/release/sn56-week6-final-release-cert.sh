#!/usr/bin/env bash
set -Eeuo pipefail
umask 027

# Canonical Week-6 release authority. The H100 timing package is lab evidence;
# the tournament runtime consumes only the reviewed constant in forge.recipe.
# A single SN56_RELEASE_COMMIT binds this wrapper, the validator, the delegated
# build certificate, the release tree, and the production timing-policy diff.

require_env() {
  local name=$1
  [[ -n ${!name-} ]] || {
    printf 'required environment variable is unset: %s\n' "${name}" >&2
    exit 64
  }
}

for required_name in \
  SN56_RELEASE_COMMIT \
  SN56_RELEASE_SOURCE_CHECKOUT \
  SN56_RELEASE_EVIDENCE_NAMESPACE \
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

readonly CERT_SCOPE=toolkit-krea-only
readonly REPO=${SN56_RELEASE_SOURCE_CHECKOUT}
readonly RELEASE_COMMIT=${SN56_RELEASE_COMMIT}
[[ ${RELEASE_COMMIT} =~ ^[0-9a-f]{40}$ ]] || {
  printf 'SN56_RELEASE_COMMIT is not a full lowercase commit id\n' >&2
  exit 64
}
[[ ${SN56_RELEASE_EVIDENCE_NAMESPACE} =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || {
  printf 'invalid release evidence namespace\n' >&2
  exit 64
}

script_parent=$(dirname -- "${BASH_SOURCE[0]}")
script_dir=$(cd -- "${script_parent}" && pwd -P)
readonly SCRIPT_DIR=${script_dir}
readonly VALIDATOR_SOURCE=${SCRIPT_DIR}/sn56-week6-validate-timing-provenance.py
readonly DELEGATED_SOURCE=${SCRIPT_DIR}/sn56-week5-final-release-cert.sh
# Patched only after the two authority files are final; tests recompute both.
readonly VALIDATOR_SHA256=e9ffa2779a446a7b5e1cb684d5858f3ecf19d3070715a6f1445356c8acf5be74
readonly DELEGATED_SHA256=42e5e2530e831732a44d13aa60124f4ae69b9afd249dced74c21124950d2c562

temporary_directory=$(mktemp -d)
cleanup() {
  rm -rf -- "${temporary_directory}"
}
trap cleanup EXIT
chmod 0700 "${temporary_directory}"
readonly PINNED_VALIDATOR=${temporary_directory}/validator.py
readonly PINNED_DELEGATED=${temporary_directory}/delegated-cert.sh

# Bootstrap trust without a hash-then-reopen gap. Python opens each source once,
# hashes and copies that descriptor's bytes into our private directory, and
# refuses zero-byte/symlink/nonregular inputs before anything can execute.
python3 - \
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
    fd = os.open(source, flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size <= 0:
            raise SystemExit(f"authority program is not a nonempty regular file: {source}")
        digest = hashlib.sha256()
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
        after = os.fstat(fd)
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns,
        )
        if identity(opened) != identity(after) or digest.hexdigest() != expected:
            raise SystemExit(f"authority program identity/hash mismatch: {source}")
        payload = b"".join(chunks)
    finally:
        os.close(fd)
    target = Path(destination)
    out = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o500)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(out, view)
            if written <= 0:
                raise SystemExit("authority program staging write failed")
            view = view[written:]
        os.fsync(out)
    finally:
        os.close(out)

values = sys.argv[1:]
if len(values) != 6:
    raise SystemExit("authority bootstrap argument error")
stage(*values[0:3])
stage(*values[3:6])
PY

git_safe=(git -c safe.directory="${REPO}" -C "${REPO}")
release_head=$("${git_safe[@]}" rev-parse HEAD)
[[ ${release_head} == "${RELEASE_COMMIT}" ]] || {
  printf 'release checkout HEAD differs from SN56_RELEASE_COMMIT\n' >&2
  exit 1
}
release_tree=$("${git_safe[@]}" rev-parse 'HEAD^{tree}')
forge_tree=$("${git_safe[@]}" rev-parse 'HEAD:forge')
release_status=$("${git_safe[@]}" status --porcelain=v1 --untracked-files=all)
[[ -z ${release_status} ]] || {
  printf 'release checkout is not clean\n' >&2
  exit 1
}
readonly RELEASE_TREE=${release_tree}
readonly FORGE_TREE=${forge_tree}
export SN56_RELEASE_TREE=${RELEASE_TREE}
export SN56_RELEASE_FORGE_TREE=${FORGE_TREE}
export SN56_RELEASE_CERT_SCOPE=${CERT_SCOPE}

readonly STAGE=${temporary_directory}/lab-evidence
mkdir -m 0700 "${STAGE}"
readonly STAGED_PROFILE=${STAGE}/timing-profile.json
readonly STAGED_SOURCE_RECORD=${STAGE}/timing-source-record.json
readonly STAGED_ARTIFACT=${STAGE}/terminal-artifact.safetensors
readonly STAGED_GATE_LOG=${STAGE}/friday-h100-gate-log.jsonl
readonly RECEIPT=${STAGE}/timing-provenance-receipt.json
readonly POLICY_RECEIPT=${STAGE}/reviewed-release-timing-policy.json

stage_evidence() {
  local source=$1
  local expected=$2
  local destination=$3
  local label=$4
  local maximum=${5-}
  local arguments=(
    "${PINNED_VALIDATOR}"
    --stage-source "${source}"
    --stage-destination "${destination}"
    --stage-sha256 "${expected}"
    --stage-label "${label}"
  )
  if [[ -n ${maximum} ]]; then
    arguments+=(--stage-maximum-bytes "${maximum}")
  fi
  python3 "${arguments[@]}"
}

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

python3 "${PINNED_VALIDATOR}" \
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
  --forge-repository "${REPO}" \
  --forge-commit "${RELEASE_COMMIT}" \
  --release-tree "${RELEASE_TREE}" \
  --certificate-scope "${CERT_SCOPE}" \
  --bundle-id "${SN56_RELEASE_TIMING_BUNDLE_ID}" \
  --bundle-sha256 "${SN56_RELEASE_TIMING_BUNDLE_SHA256}" \
  --model-type "${SN56_RELEASE_TIMING_MODEL_TYPE}" \
  --current-dataset-size "${SN56_RELEASE_TIMING_CURRENT_DATASET_SIZE}" \
  --dataset-regime "${SN56_RELEASE_TIMING_DATASET_REGIME}" \
  --accelerator-identity "${SN56_RELEASE_TIMING_ACCELERATOR_IDENTITY}" \
  --receipt "${RECEIPT}"

python3 "${PINNED_VALIDATOR}" \
  --assert-receipt "${RECEIPT}" \
  --forge-commit "${RELEASE_COMMIT}" \
  --release-tree "${RELEASE_TREE}" \
  --certificate-scope "${CERT_SCOPE}" \
  --profile-file-sha256 "${SN56_RELEASE_TIMING_PROFILE_SHA256}" \
  --raw-record-file-sha256 "${SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256}" \
  --terminal-artifact-file-sha256 "${SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256}" \
  --gate-log-file-sha256 "${SN56_RELEASE_FRIDAY_GATE_LOG_SHA256}"

python3 "${PINNED_VALIDATOR}" \
  --assert-release-policy \
  --forge-repository "${REPO}" \
  --forge-commit "${RELEASE_COMMIT}" \
  --release-tree "${RELEASE_TREE}" \
  --receipt "${POLICY_RECEIPT}"

# The delegated build/GPU certificate receives the same release identity and a
# fixed hash-pinned script. It cannot be replaced with an environment override.
/bin/bash "${PINNED_DELEGATED}"

readonly DELEGATED_EVIDENCE=/mnt/sn56-evidence/final-release-cert/${SN56_RELEASE_EVIDENCE_NAMESPACE}
readonly DELEGATED_RESULT=${DELEGATED_EVIDENCE}/result.env
[[ -d ${DELEGATED_EVIDENCE} && -f ${DELEGATED_EVIDENCE}/MANIFEST.sha256 ]] || {
  printf 'delegated certificate evidence is incomplete: %s\n' "${DELEGATED_EVIDENCE}" >&2
  exit 1
}
python3 "${PINNED_VALIDATOR}" \
  --assert-result-env "${DELEGATED_RESULT}" \
  --forge-commit "${RELEASE_COMMIT}" \
  --release-tree "${RELEASE_TREE}" \
  --certificate-scope "${CERT_SCOPE}"
(
  cd "${DELEGATED_EVIDENCE}"
  sha256sum -c MANIFEST.sha256
) >"${temporary_directory}/delegated-manifest-check.txt"

readonly ENVELOPE_BASE=/mnt/sn56-evidence/week6-final-release-cert
readonly ENVELOPE=${ENVELOPE_BASE}/${SN56_RELEASE_EVIDENCE_NAMESPACE}
[[ ! -e ${ENVELOPE} && ! -L ${ENVELOPE} ]] || {
  printf 'Week-6 evidence envelope already exists: %s\n' "${ENVELOPE}" >&2
  exit 1
}
install -d -o root -g root -m 0750 "${ENVELOPE_BASE}"
install -d -o root -g root -m 0750 "${ENVELOPE}"

stage_evidence "${STAGED_PROFILE}" "${SN56_RELEASE_TIMING_PROFILE_SHA256}" \
  "${ENVELOPE}/timing-profile.json" 'envelope timing profile' 65536
stage_evidence "${STAGED_SOURCE_RECORD}" "${SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256}" \
  "${ENVELOPE}/timing-source-record.json" 'envelope timing source record' 1048576
stage_evidence "${STAGED_ARTIFACT}" "${SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256}" \
  "${ENVELOPE}/terminal-artifact.safetensors" 'envelope terminal artifact'
stage_evidence "${STAGED_GATE_LOG}" "${SN56_RELEASE_FRIDAY_GATE_LOG_SHA256}" \
  "${ENVELOPE}/friday-h100-gate-log.jsonl" 'envelope gate log' 16777216

receipt_hash_line=$(sha256sum "${RECEIPT}")
receipt_sha256=${receipt_hash_line%% *}
policy_hash_line=$(sha256sum "${POLICY_RECEIPT}")
policy_receipt_sha256=${policy_hash_line%% *}
stage_evidence "${RECEIPT}" "${receipt_sha256}" \
  "${ENVELOPE}/timing-provenance-receipt.json" 'envelope timing receipt' 262144
stage_evidence "${POLICY_RECEIPT}" "${policy_receipt_sha256}" \
  "${ENVELOPE}/reviewed-release-timing-policy.json" 'envelope release policy' 65536
install -o root -g root -m 0440 \
  "${temporary_directory}/delegated-manifest-check.txt" \
  "${ENVELOPE}/delegated-manifest-check.txt"

python3 "${PINNED_VALIDATOR}" \
  --assert-receipt "${ENVELOPE}/timing-provenance-receipt.json" \
  --forge-commit "${RELEASE_COMMIT}" \
  --release-tree "${RELEASE_TREE}" \
  --certificate-scope "${CERT_SCOPE}" \
  --profile-file-sha256 "${SN56_RELEASE_TIMING_PROFILE_SHA256}" \
  --raw-record-file-sha256 "${SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256}" \
  --terminal-artifact-file-sha256 "${SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256}" \
  --gate-log-file-sha256 "${SN56_RELEASE_FRIDAY_GATE_LOG_SHA256}"

delegated_manifest_hash_line=$(sha256sum "${DELEGATED_EVIDENCE}/MANIFEST.sha256")
delegated_manifest_sha256=${delegated_manifest_hash_line%% *}
completed_at_utc=$(date -u +%FT%TZ)
{
  printf 'schema=sn56.week6.final-release-cert-envelope.v2\n'
  printf 'state=PASS\n'
  printf 'certificate_scope=%s\n' "${CERT_SCOPE}"
  printf 'source_commit=%s\n' "${RELEASE_COMMIT}"
  printf 'source_tree=%s\n' "${RELEASE_TREE}"
  printf 'forge_tree=%s\n' "${FORGE_TREE}"
  printf 'timing_evidence_scope=lab-only\n'
  printf 'production_timing_input=reviewed-conservative-constant\n'
  printf 'profile_file_sha256=%s\n' "${SN56_RELEASE_TIMING_PROFILE_SHA256}"
  printf 'raw_record_file_sha256=%s\n' "${SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256}"
  printf 'terminal_artifact_file_sha256=%s\n' "${SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256}"
  printf 'gate_log_file_sha256=%s\n' "${SN56_RELEASE_FRIDAY_GATE_LOG_SHA256}"
  printf 'timing_receipt_file_sha256=%s\n' "${receipt_sha256}"
  printf 'release_policy_receipt_file_sha256=%s\n' "${policy_receipt_sha256}"
  printf 'delegated_evidence=%s\n' "${DELEGATED_EVIDENCE}"
  printf 'delegated_manifest_sha256=%s\n' "${delegated_manifest_sha256}"
  printf 'completed_at_utc=%s\n' "${completed_at_utc}"
} >"${ENVELOPE}/result.env"

(
  cd "${ENVELOPE}"
  find . -type f ! -name MANIFEST.sha256 -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 -r sha256sum >MANIFEST.sha256
)
chmod -R a-w "${ENVELOPE}"
sync
manifest_hash_line=$(sha256sum "${ENVELOPE}/MANIFEST.sha256")
manifest_sha256=${manifest_hash_line%% *}
printf 'SN56_WEEK6_FINAL_RELEASE_CERT_PASS evidence=%s manifest=%s\n' \
  "${ENVELOPE}" "${manifest_sha256}"
