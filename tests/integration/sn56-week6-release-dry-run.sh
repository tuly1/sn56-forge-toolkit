#!/bin/bash
set -Eeuo pipefail
umask 027

# Exact-head, clean-clone integration harness for the canonical outer wrapper.
#
# This harness intentionally calls the real Week-6 wrapper. It does not invoke
# the build/GPU delegate directly and does not replace Docker, Git, the timing
# validator, receipt parsing, image inspection, or envelope publication. The
# wrapper's `cpu-integration` mode may stub only the two physical GPU boundaries
# (host nvidia-smi observation and container `--gpus` import) and must finish as
# DRY_RUN_PASS. A production PASS is an integration-test failure.
#
# Expected outer-wrapper contract (wired by the release-authority patch):
#   * SN56_RELEASE_CERT_MODE=cpu-integration
#   * the existing SN56_RELEASE_* timing inputs below
#   * a fixed expected origin URL and independently resolved remote ref
#   * fixed delegate build inputs (two image tags, roots, work/evidence bases)
#   * final stdout:
#       SN56_WEEK6_FINAL_RELEASE_CERT=DRY_RUN_PASS evidence=<absolute-path>
#
# The caller supplies a real schema-current lab fixture package. Because the
# result has no production authority, its accelerator/timestamps may be a
# purpose-built internally consistent integration fixture. Run this only on an
# amd64 Linux host with rootful Docker, both configured storage roots, and a
# dedicated evidence filesystem. The transcript path must be on durable evidence
# storage outside the cloned repository.

usage() {
  /usr/bin/printf '%s\n' \
    'usage: sn56-week6-release-dry-run.sh' \
    '  --repo-url HTTPS_URL --remote-ref REF --commit SHA1' \
    '  --evidence-namespace NAME --transcript ABS --work-base ABS' \
    '  --delegate-evidence-base ABS --envelope-base ABS' \
    '  --timing-profile ABS --timing-source-record ABS' \
    '  --timing-terminal-artifact ABS --friday-gate-log ABS' \
    '  --timing-source-run-id ID --h100-gate-session-id ID' \
    '  --h100-rental-started-at-utc UTC --h100-rental-ended-at-utc UTC' \
    '  --timing-bundle-id ID --timing-bundle-sha256 SHA256' \
    '  --timing-model-type TYPE --timing-current-dataset-size N' \
    '  --timing-dataset-regime REGIME' \
    '  --toolkit-image-tag TAG --legacy-image-tag TAG' \
    '  --expected-docker-root ABS --expected-containerd-root ABS'
  exit 64
}

git_isolated() {
  /usr/bin/env -i \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HOME=/nonexistent LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null \
    GIT_NO_REPLACE_OBJECTS=1 GIT_TERMINAL_PROMPT=0 \
    GIT_CONFIG_COUNT=8 \
    GIT_CONFIG_KEY_0=core.fsmonitor GIT_CONFIG_VALUE_0=false \
    GIT_CONFIG_KEY_1=core.hooksPath GIT_CONFIG_VALUE_1=/dev/null \
    GIT_CONFIG_KEY_2=core.untrackedCache GIT_CONFIG_VALUE_2=false \
    GIT_CONFIG_KEY_3=core.ignoreStat GIT_CONFIG_VALUE_3=false \
    GIT_CONFIG_KEY_4=core.trustctime GIT_CONFIG_VALUE_4=true \
    GIT_CONFIG_KEY_5=core.checkStat GIT_CONFIG_VALUE_5=default \
    GIT_CONFIG_KEY_6=core.attributesFile GIT_CONFIG_VALUE_6=/dev/null \
    GIT_CONFIG_KEY_7=core.excludesFile GIT_CONFIG_VALUE_7=/dev/null \
    /usr/bin/git --no-replace-objects \
      -c core.fsmonitor=false \
      -c core.hooksPath=/dev/null \
      -c core.untrackedCache=false \
      -c core.ignoreStat=false \
      -c core.trustctime=true \
      -c core.checkStat=default \
      -c core.attributesFile=/dev/null \
      -c core.excludesFile=/dev/null "$@"
}

repo_url=''
remote_ref=''
commit=''
evidence_namespace=''
transcript=''
work_base=''
delegate_evidence_base=''
envelope_base=''
timing_profile=''
timing_source_record=''
timing_terminal_artifact=''
friday_gate_log=''
timing_source_run_id=''
h100_gate_session_id=''
h100_rental_started_at_utc=''
h100_rental_ended_at_utc=''
timing_bundle_id=''
timing_bundle_sha256=''
timing_model_type=''
timing_current_dataset_size=''
timing_dataset_regime=''
toolkit_image_tag=''
legacy_image_tag=''
expected_docker_root=''
expected_containerd_root=''

while (( $# )); do
  (( $# >= 2 )) || usage
  case $1 in
    --repo-url) repo_url=$2 ;;
    --remote-ref) remote_ref=$2 ;;
    --commit) commit=$2 ;;
    --evidence-namespace) evidence_namespace=$2 ;;
    --transcript) transcript=$2 ;;
    --work-base) work_base=$2 ;;
    --delegate-evidence-base) delegate_evidence_base=$2 ;;
    --envelope-base) envelope_base=$2 ;;
    --timing-profile) timing_profile=$2 ;;
    --timing-source-record) timing_source_record=$2 ;;
    --timing-terminal-artifact) timing_terminal_artifact=$2 ;;
    --friday-gate-log) friday_gate_log=$2 ;;
    --timing-source-run-id) timing_source_run_id=$2 ;;
    --h100-gate-session-id) h100_gate_session_id=$2 ;;
    --h100-rental-started-at-utc) h100_rental_started_at_utc=$2 ;;
    --h100-rental-ended-at-utc) h100_rental_ended_at_utc=$2 ;;
    --timing-bundle-id) timing_bundle_id=$2 ;;
    --timing-bundle-sha256) timing_bundle_sha256=$2 ;;
    --timing-model-type) timing_model_type=$2 ;;
    --timing-current-dataset-size) timing_current_dataset_size=$2 ;;
    --timing-dataset-regime) timing_dataset_regime=$2 ;;
    --toolkit-image-tag) toolkit_image_tag=$2 ;;
    --legacy-image-tag) legacy_image_tag=$2 ;;
    --expected-docker-root) expected_docker_root=$2 ;;
    --expected-containerd-root) expected_containerd_root=$2 ;;
    *) usage ;;
  esac
  shift 2
done

for required_value in \
  "${repo_url}" "${remote_ref}" "${commit}" "${evidence_namespace}" \
  "${transcript}" "${work_base}" "${delegate_evidence_base}" \
  "${envelope_base}" "${timing_profile}" \
  "${timing_source_record}" "${timing_terminal_artifact}" \
  "${friday_gate_log}" "${timing_source_run_id}" \
  "${h100_gate_session_id}" "${h100_rental_started_at_utc}" \
  "${h100_rental_ended_at_utc}" "${timing_bundle_id}" \
  "${timing_bundle_sha256}" "${timing_model_type}" \
  "${timing_current_dataset_size}" "${timing_dataset_regime}" \
  "${toolkit_image_tag}" \
  "${legacy_image_tag}" "${expected_docker_root}" \
  "${expected_containerd_root}"
do
  [[ -n ${required_value} ]] || usage
done

[[ ${repo_url} =~ ^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git$ ]] || usage
[[ ${remote_ref} =~ ^refs/heads/[A-Za-z0-9._/-]+$ ]] || usage
git_isolated -C / check-ref-format "${remote_ref}" || usage
[[ ${commit} =~ ^[0-9a-f]{40}$ ]] || usage
[[ ${timing_bundle_sha256} =~ ^[0-9a-f]{64}$ ]] || usage
[[ ${evidence_namespace} =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || usage
for absolute_path in \
  "${transcript}" "${work_base}" "${delegate_evidence_base}" \
  "${envelope_base}" \
  "${timing_profile}" "${timing_source_record}" \
  "${timing_terminal_artifact}" "${friday_gate_log}" \
  "${expected_docker_root}" "${expected_containerd_root}"
do
  [[ ${absolute_path} == /* && ${absolute_path} != *'/../'* ]] || usage
done
for input_file in \
  "${timing_profile}" "${timing_source_record}" \
  "${timing_terminal_artifact}" "${friday_gate_log}"
do
  [[ -f ${input_file} && ! -L ${input_file} ]] || {
    /usr/bin/printf 'integration input is absent or indirect: %s\n' "${input_file}" >&2
    exit 1
  }
  [[ $(/usr/bin/realpath -e "${input_file}") == "${input_file}" ]] || {
    /usr/bin/printf 'integration input path is indirect: %s\n' "${input_file}" >&2
    exit 1
  }
done
[[ -d ${work_base} && ! -L ${work_base} ]] || {
  /usr/bin/printf 'work base is absent or indirect: %s\n' "${work_base}" >&2
  exit 1
}
[[ -d ${delegate_evidence_base} && ! -L ${delegate_evidence_base} ]] || {
  /usr/bin/printf 'delegate evidence base is absent or indirect: %s\n' "${delegate_evidence_base}" >&2
  exit 1
}
[[ -d ${envelope_base} && ! -L ${envelope_base} ]] || {
  /usr/bin/printf 'outer envelope base is absent or indirect: %s\n' "${envelope_base}" >&2
  exit 1
}
transcript_parent=$(/usr/bin/dirname -- "${transcript}")
[[ -d ${transcript_parent} && ! -L ${transcript_parent} && ! -e ${transcript} ]] || {
  /usr/bin/printf 'transcript destination is unavailable, indirect, or occupied\n' >&2
  exit 1
}

temporary_directory=$(/usr/bin/mktemp -d "${work_base}/sn56-week6-wrapper-integration.XXXXXXXX")
cleanup() {
  /bin/rm -rf -- "${temporary_directory}"
}
trap cleanup EXIT
/bin/chmod 0700 "${temporary_directory}"
clone=${temporary_directory}/exact-head-clone

git_isolated -C / clone --no-checkout -- "${repo_url}" "${clone}"
git_isolated -C "${clone}" checkout --detach "${commit}"

# Exact hostile repository-local configuration from the independent audit.
# The wrapper must inspect/archive the repository without executing the
# fsmonitor command or applying its URL rewrite.
git_isolated -C "${clone}" config --local core.fsmonitor 'sleep 2 #'
git_isolated -C "${clone}" update-index --fsmonitor
git_isolated -C "${clone}" config --local \
  url.https://attacker.invalid/spoof.git.insteadOf "${repo_url}"

actual_head=$(git_isolated -C "${clone}" rev-parse HEAD)
[[ ${actual_head} == "${commit}" ]]
source_tree=$(git_isolated -C "${clone}" rev-parse 'HEAD^{tree}')
forge_tree=$(git_isolated -C "${clone}" rev-parse 'HEAD:forge')
status=$(git_isolated -C "${clone}" status \
  --porcelain=v1 --untracked-files=all --ignored=matching)
[[ -z ${status} ]] || {
  /usr/bin/printf 'exact-head integration clone is not clean:\n%s\n' "${status}" >&2
  exit 1
}

# Plant the independent audit's exact hostile-attributes surface only after the
# harness has done its own convenience clean check. The release wrapper must
# neither execute this filter nor let it spoof its source-byte verification.
hostile_filter=${clone}/.git/sn56-hostile-clean-filter
hostile_marker=${temporary_directory}/hostile-filter-executed
/usr/bin/printf '%s\n' \
  '#!/bin/sh' \
  '/usr/bin/touch -- "$1"' \
  '/bin/cat' >"${hostile_filter}"
/bin/chmod 0700 "${hostile_filter}"
/bin/mkdir -p "${clone}/.git/info"
/usr/bin/printf '%s\n' 'forge/recipe.py filter=sn56-hostile' \
  >"${clone}/.git/info/attributes"
git_isolated -C "${clone}" config --local filter.sn56-hostile.clean \
  "${hostile_filter} ${hostile_marker}"
/usr/bin/touch "${clone}/forge/recipe.py"

hash_file() {
  local path=$1
  local line
  line=$(/usr/bin/sha256sum -- "${path}")
  /usr/bin/printf '%s\n' "${line%% *}"
}

wrapper=${clone}/ops/release/sn56-week6-final-release-cert.sh
[[ -f ${wrapper} && ! -L ${wrapper} ]] || {
  /usr/bin/printf 'canonical outer wrapper is absent from exact clone\n' >&2
  exit 1
}

set +e
/usr/bin/env -i \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  HOME=/nonexistent \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  GIT_NO_REPLACE_OBJECTS=1 \
  GIT_TERMINAL_PROMPT=0 \
  SN56_RELEASE_CERT_MODE=cpu-integration \
  SN56_RELEASE_COMMIT="${commit}" \
  SN56_RELEASE_SOURCE_CHECKOUT="${clone}" \
  SN56_RELEASE_EXPECTED_ORIGIN_URL="${repo_url}" \
  SN56_RELEASE_REMOTE_REF="${remote_ref}" \
  SN56_RELEASE_EVIDENCE_NAMESPACE="${evidence_namespace}" \
  SN56_RELEASE_DELEGATE_EVIDENCE_BASE="${delegate_evidence_base}" \
  SN56_RELEASE_ENVELOPE_BASE="${envelope_base}" \
  SN56_RELEASE_WORK_BASE="${work_base}" \
  SN56_RELEASE_TOOLKIT_IMAGE_TAG="${toolkit_image_tag}" \
  SN56_RELEASE_LEGACY_IMAGE_TAG="${legacy_image_tag}" \
  SN56_RELEASE_EXPECTED_DOCKER_ROOT="${expected_docker_root}" \
  SN56_RELEASE_EXPECTED_CONTAINERD_ROOT="${expected_containerd_root}" \
  SN56_RELEASE_TIMING_PROFILE="${timing_profile}" \
  SN56_RELEASE_TIMING_PROFILE_SHA256="$(hash_file "${timing_profile}")" \
  SN56_RELEASE_TIMING_SOURCE_RECORD="${timing_source_record}" \
  SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256="$(hash_file "${timing_source_record}")" \
  SN56_RELEASE_TIMING_TERMINAL_ARTIFACT="${timing_terminal_artifact}" \
  SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256="$(hash_file "${timing_terminal_artifact}")" \
  SN56_RELEASE_FRIDAY_GATE_LOG="${friday_gate_log}" \
  SN56_RELEASE_FRIDAY_GATE_LOG_SHA256="$(hash_file "${friday_gate_log}")" \
  SN56_RELEASE_TIMING_SOURCE_RUN_ID="${timing_source_run_id}" \
  SN56_RELEASE_H100_GATE_SESSION_ID="${h100_gate_session_id}" \
  SN56_RELEASE_H100_RENTAL_STARTED_AT_UTC="${h100_rental_started_at_utc}" \
  SN56_RELEASE_H100_RENTAL_ENDED_AT_UTC="${h100_rental_ended_at_utc}" \
  SN56_RELEASE_TIMING_BUNDLE_ID="${timing_bundle_id}" \
  SN56_RELEASE_TIMING_BUNDLE_SHA256="${timing_bundle_sha256}" \
  SN56_RELEASE_TIMING_MODEL_TYPE="${timing_model_type}" \
  SN56_RELEASE_TIMING_CURRENT_DATASET_SIZE="${timing_current_dataset_size}" \
  SN56_RELEASE_TIMING_DATASET_REGIME="${timing_dataset_regime}" \
  /bin/sh "${wrapper}" 2>&1 | /usr/bin/tee "${transcript}"
pipeline_status=("${PIPESTATUS[@]}")
wrapper_rc=${pipeline_status[0]}
tee_rc=${pipeline_status[1]}
set -e
(( tee_rc == 0 )) || {
  /usr/bin/printf 'transcript write failed rc=%s\n' "${tee_rc}" >&2
  exit "${tee_rc}"
}
(( wrapper_rc == 0 )) || {
  /usr/bin/printf 'outer wrapper dry-run failed rc=%s\n' "${wrapper_rc}" >&2
  exit "${wrapper_rc}"
}

/usr/bin/grep -Eq \
  '^SN56_WEEK6_FINAL_RELEASE_CERT=DRY_RUN_PASS evidence=/[^[:space:]]+$' \
  "${transcript}" || {
    /usr/bin/printf 'outer wrapper did not emit the exact DRY_RUN_PASS marker\n' >&2
    exit 1
  }
if /usr/bin/grep -Eq \
  '^SN56_WEEK6_FINAL_RELEASE_CERT=PASS([[:space:]]|$)' \
  "${transcript}"; then
  /usr/bin/printf 'CPU integration illegally emitted production PASS\n' >&2
  exit 1
fi
[[ ! -e ${hostile_marker} ]] || {
  /usr/bin/printf 'release wrapper executed repository-local clean filter\n' >&2
  exit 1
}

/usr/bin/printf \
  'SN56_WEEK6_CLEAN_CLONE_WRAPPER_INTEGRATION=DRY_RUN_PASS commit=%s tree=%s forge_tree=%s transcript=%s\n' \
  "${commit}" "${source_tree}" "${forge_tree}" "${transcript}"
