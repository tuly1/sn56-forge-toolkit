#!/usr/bin/env bash
set -Eeuo pipefail
umask 027

# Week-5 final toolkit/Krea image release certificate.
#
# This draft deliberately has no campaign defaults. Every decision-bound identity
# is supplied explicitly, and the run fails before touching Docker if any binding
# is absent or malformed. The script never migrates/restarts Docker or containerd.
# Run it only after the campaign queue is empty and the final commit is pushed.

require_env() {
  local name=$1
  local value=${!name-}
  [[ -n ${value} ]] || {
    printf 'required environment variable is unset: %s\n' "${name}" >&2
    exit 64
  }
}

for required_name in \
  SN56_RELEASE_COMMIT \
  SN56_RELEASE_TREE \
  SN56_RELEASE_FORGE_TREE \
  SN56_RELEASE_PRODUCTION_TRUST_SHA256 \
  SN56_RELEASE_DOCKERFILE_SHA256 \
  SN56_RELEASE_LOCK_SHA256 \
  SN56_RELEASE_CONSTRAINTS_SHA256 \
  SN56_RELEASE_VERIFIER_SHA256 \
  SN56_RELEASE_SOURCE_CHECKOUT \
  SN56_RELEASE_EXPECTED_ORIGIN_URL \
  SN56_RELEASE_REMOTE_REF \
  SN56_RELEASE_EVIDENCE_NAMESPACE \
  SN56_RELEASE_IMAGE_TAG \
  SN56_RELEASE_EXPECTED_CONTAINERD_ROOT \
  SN56_RELEASE_POLICY \
  SN56_RELEASE_CERT_SCOPE \
  SN56_RELEASE_K5_DECISION_RECORD \
  SN56_RELEASE_K5_DECISION_RECORD_SHA256 \
  SN56_RELEASE_FORMAL_DECISION_SEMANTIC_SHA256 \
  SN56_RELEASE_KREA_POLICY_SHA256 \
  SN56_RELEASE_PRODUCTION_ACTIVATION_RECORD \
  SN56_RELEASE_PRODUCTION_ACTIVATION_RECORD_SHA256 \
  SN56_RELEASE_PRODUCTION_ACTIVATION_SHA256 \
  SN56_RELEASE_RELEASE_RECORD \
  SN56_RELEASE_RELEASE_RECORD_SHA256 \
  SN56_RELEASE_LEGACY_DOCKERFILE_SHA256 \
  SN56_RELEASE_LEGACY_NO_REGRESSION_RECORD \
  SN56_RELEASE_LEGACY_NO_REGRESSION_RECORD_SHA256 \
  SN56_RELEASE_KREA_PROBE \
  SN56_RELEASE_KREA_PROBE_SHA256 \
  SN56_RELEASE_IDEOGRAM_MODE
do
  require_env "${required_name}"
done

[[ ${SN56_RELEASE_IDEOGRAM_MODE} == unchanged-deferred ]] || {
  printf 'this K5-global certificate requires Ideogram unchanged-deferred\n' >&2
  exit 64
}

readonly EXPECTED_COMMIT=${SN56_RELEASE_COMMIT}
readonly EXPECTED_TREE=${SN56_RELEASE_TREE}
readonly EXPECTED_FORGE_TREE=${SN56_RELEASE_FORGE_TREE}
readonly EXPECTED_PRODUCTION_TRUST=${SN56_RELEASE_PRODUCTION_TRUST_SHA256}
readonly EXPECTED_DOCKERFILE=${SN56_RELEASE_DOCKERFILE_SHA256}
readonly EXPECTED_LOCK=${SN56_RELEASE_LOCK_SHA256}
readonly EXPECTED_CONSTRAINTS=${SN56_RELEASE_CONSTRAINTS_SHA256}
readonly EXPECTED_VERIFIER=${SN56_RELEASE_VERIFIER_SHA256}
readonly REPO=${SN56_RELEASE_SOURCE_CHECKOUT}
readonly EXPECTED_ORIGIN_URL=${SN56_RELEASE_EXPECTED_ORIGIN_URL}
readonly REMOTE_REF=${SN56_RELEASE_REMOTE_REF}
readonly EVIDENCE_NAMESPACE=${SN56_RELEASE_EVIDENCE_NAMESPACE}
readonly IMAGE=${SN56_RELEASE_IMAGE_TAG}
readonly EXPECTED_CONTAINERD_ROOT=${SN56_RELEASE_EXPECTED_CONTAINERD_ROOT}
readonly RELEASE_POLICY=${SN56_RELEASE_POLICY}
readonly CERT_SCOPE=${SN56_RELEASE_CERT_SCOPE}
readonly K5_DECISION_SOURCE=${SN56_RELEASE_K5_DECISION_RECORD}
readonly EXPECTED_K5_DECISION=${SN56_RELEASE_K5_DECISION_RECORD_SHA256}
readonly EXPECTED_FORMAL_DECISION_SEMANTIC=${SN56_RELEASE_FORMAL_DECISION_SEMANTIC_SHA256}
readonly EXPECTED_KREA_POLICY=${SN56_RELEASE_KREA_POLICY_SHA256}
readonly ACTIVATION_SOURCE=${SN56_RELEASE_PRODUCTION_ACTIVATION_RECORD}
readonly EXPECTED_ACTIVATION_FILE=${SN56_RELEASE_PRODUCTION_ACTIVATION_RECORD_SHA256}
readonly EXPECTED_ACTIVATION_SEMANTIC=${SN56_RELEASE_PRODUCTION_ACTIVATION_SHA256}
readonly RELEASE_RECORD_SOURCE=${SN56_RELEASE_RELEASE_RECORD}
readonly EXPECTED_RELEASE_RECORD=${SN56_RELEASE_RELEASE_RECORD_SHA256}
readonly EXPECTED_LEGACY_DOCKERFILE=${SN56_RELEASE_LEGACY_DOCKERFILE_SHA256}
readonly LEGACY_NO_REGRESSION_SOURCE=${SN56_RELEASE_LEGACY_NO_REGRESSION_RECORD}
readonly EXPECTED_LEGACY_NO_REGRESSION=${SN56_RELEASE_LEGACY_NO_REGRESSION_RECORD_SHA256}
readonly KREA_PROBE_SOURCE=${SN56_RELEASE_KREA_PROBE}
readonly EXPECTED_KREA_PROBE=${SN56_RELEASE_KREA_PROBE_SHA256}
readonly IDEOGRAM_MODE=${SN56_RELEASE_IDEOGRAM_MODE}

readonly DOCKERFILE_REL=ops/docker/standalone-image-toolkit-trainer.dockerfile
readonly LEGACY_DOCKERFILE_REL=ops/docker/standalone-image-trainer.dockerfile
readonly LOCK_REL=ops/docker/image-runtime-lock.txt
readonly CONSTRAINTS_REL=ops/docker/image-runtime-phase1-constraints.txt
readonly VERIFIER_REL=ops/docker/verify_image_runtime.py
readonly KREA_POLICY_REL=forge/policies/krea_week5_production_predeclaration.json
readonly KREA_POLICY_MODULE_REL=forge/krea_release_policy.py
readonly EVIDENCE_MOUNT=/mnt/sn56-evidence
readonly EVIDENCE_BASE=${EVIDENCE_MOUNT}/final-release-cert
readonly EVIDENCE=${EVIDENCE_BASE}/${EVIDENCE_NAMESPACE}
readonly DOCKER_ROOT=/ephemeral/docker
readonly ROOT_START_MIN=21474836480
readonly EPHEMERAL_START_MIN=536870912000
readonly EVIDENCE_START_MIN=5368709120
readonly ROOT_PRESSURE_FLOOR=17179869184
readonly EPHEMERAL_PRESSURE_FLOOR=483183820800
readonly EVIDENCE_PRESSURE_FLOOR=2147483648

for sha1_value in \
  "${EXPECTED_COMMIT}" \
  "${EXPECTED_TREE}" \
  "${EXPECTED_FORGE_TREE}"
do
  [[ ${sha1_value} =~ ^[0-9a-f]{40}$ ]] || {
    printf 'invalid 40-character git identity: %s\n' "${sha1_value}" >&2
    exit 64
  }
done

for sha256_value in \
  "${EXPECTED_PRODUCTION_TRUST}" \
  "${EXPECTED_DOCKERFILE}" \
  "${EXPECTED_LOCK}" \
  "${EXPECTED_CONSTRAINTS}" \
  "${EXPECTED_VERIFIER}" \
  "${EXPECTED_K5_DECISION}" \
  "${EXPECTED_FORMAL_DECISION_SEMANTIC}" \
  "${EXPECTED_KREA_POLICY}" \
  "${EXPECTED_ACTIVATION_FILE}" \
  "${EXPECTED_ACTIVATION_SEMANTIC}" \
  "${EXPECTED_RELEASE_RECORD}" \
  "${EXPECTED_LEGACY_DOCKERFILE}" \
  "${EXPECTED_LEGACY_NO_REGRESSION}" \
  "${EXPECTED_KREA_PROBE}"
do
  [[ ${sha256_value} =~ ^[0-9a-f]{64}$ ]] || {
    printf 'invalid 64-character sha256 identity: %s\n' "${sha256_value}" >&2
    exit 64
  }
done

[[ ${EUID} -eq 0 ]] || { printf 'must run as root\n' >&2; exit 1; }
[[ ${RELEASE_POLICY} == K5-global ]] || {
  printf 'this certificate is bound only to the predeclared K5-global release branch\n' >&2
  exit 64
}
[[ ${CERT_SCOPE} == toolkit-krea-only ]] || {
  printf 'this certificate scope must be toolkit-krea-only\n' >&2
  exit 64
}
[[ ${EXPECTED_ORIGIN_URL} =~ ^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git$ ]] || {
  printf 'expected origin must be a credential-free canonical GitHub HTTPS URL ending in .git\n' >&2
  exit 64
}
[[ ${REPO} == /ephemeral/* && ${REPO} != *'/../'* ]] || {
  printf 'source checkout must be an absolute /ephemeral path\n' >&2
  exit 64
}
[[ -d ${REPO} && ! -L ${REPO} && $(realpath -e "${REPO}") == "${REPO}" ]] || {
  printf 'source checkout is absent, indirect, or symlinked: %s\n' "${REPO}" >&2
  exit 1
}
[[ ${REMOTE_REF} =~ ^refs/heads/[A-Za-z0-9._/-]+$ && ${REMOTE_REF} != *'..'* ]] || {
  printf 'remote ref must be a full safe refs/heads/* name\n' >&2
  exit 64
}
[[ ${EVIDENCE_NAMESPACE} =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || {
  printf 'invalid evidence namespace\n' >&2
  exit 64
}
[[ ${IMAGE} =~ ^[a-z0-9][a-z0-9._/-]*:[A-Za-z0-9_.-]+$ ]] || {
  printf 'invalid local image tag: %s\n' "${IMAGE}" >&2
  exit 64
}
[[ ${EXPECTED_CONTAINERD_ROOT} == /ephemeral/* && ${EXPECTED_CONTAINERD_ROOT} != *'/../'* ]] || {
  printf 'expected containerd root must be an absolute /ephemeral path\n' >&2
  exit 64
}
for immutable_input in \
  "${K5_DECISION_SOURCE}" \
  "${ACTIVATION_SOURCE}" \
  "${RELEASE_RECORD_SOURCE}" \
  "${LEGACY_NO_REGRESSION_SOURCE}" \
  "${KREA_PROBE_SOURCE}"
do
  [[ ${immutable_input} == /* && -f ${immutable_input} && ! -L ${immutable_input} ]] || {
    printf 'bound input must be an absolute regular non-symlink file: %s\n' "${immutable_input}" >&2
    exit 64
  }
  [[ $(realpath -e "${immutable_input}") == "${immutable_input}" ]] || {
    printf 'bound input path contains an indirect component: %s\n' "${immutable_input}" >&2
    exit 64
  }
done

[[ -d ${EVIDENCE_MOUNT} && ! -L ${EVIDENCE_MOUNT} ]] || {
  printf 'persistent evidence mount is absent or symlinked: %s\n' "${EVIDENCE_MOUNT}" >&2
  exit 1
}
[[ $(findmnt -n -o TARGET -T "${EVIDENCE_MOUNT}") == "${EVIDENCE_MOUNT}" ]] || {
  printf '%s is not a dedicated mount target\n' "${EVIDENCE_MOUNT}" >&2
  exit 1
}
evidence_device=$(findmnt -n -o MAJ:MIN -T "${EVIDENCE_MOUNT}")
root_device=$(findmnt -n -o MAJ:MIN -T /)
ephemeral_device=$(findmnt -n -o MAJ:MIN -T /ephemeral)
[[ -n ${evidence_device} && ${evidence_device} != "${root_device}" && ${evidence_device} != "${ephemeral_device}" ]] || {
  printf 'evidence mount is not independent from root and ephemeral storage\n' >&2
  exit 1
}
[[ ! -e ${EVIDENCE} && ! -L ${EVIDENCE} ]] || {
  printf 'evidence namespace already exists: %s\n' "${EVIDENCE}" >&2
  exit 1
}
[[ ! -L ${EVIDENCE_BASE} ]] || {
  printf 'evidence base must not be a symlink: %s\n' "${EVIDENCE_BASE}" >&2
  exit 1
}

install -d -o root -g root -m 0750 "${EVIDENCE_BASE}"
install -d -o root -g root -m 0750 "${EVIDENCE}"

build_pid=''
guard_pid=''
probe_pid=''
probe_pgid=''

write_manifest() {
  (
    cd "${EVIDENCE}"
    find . -type f ! -name MANIFEST.sha256 -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 -r sha256sum >MANIFEST.sha256
  )
}

seal_failure() {
  local rc=${1:-1}
  local reason=${2:-unexpected-error}
  trap - ERR INT TERM
  stop_and_reap() {
    local pid=${1-}
    local scope=${2:-pid}
    local target
    [[ -n ${pid} ]] || return 0
    target=${pid}
    [[ ${scope} == group ]] && target=-${pid}
    if kill -0 -- "${target}" 2>/dev/null; then
      kill -TERM -- "${target}" 2>/dev/null || true
      for _ in $(seq 1 30); do
        if ! kill -0 -- "${target}" 2>/dev/null; then
          break
        fi
        sleep 1
      done
      if kill -0 -- "${target}" 2>/dev/null; then
        kill -KILL -- "${target}" 2>/dev/null || true
        for _ in $(seq 1 30); do
          kill -0 -- "${target}" 2>/dev/null || break
          sleep 1
        done
      fi
    fi
    wait "${pid}" 2>/dev/null || true
  }
  # Stop every process with an open evidence writer before hashing the archive.
  stop_and_reap "${probe_pgid}" group
  stop_and_reap "${probe_pid}"
  # The Krea wrapper uses this deterministic name.  Remove a surviving Docker
  # daemon workload before sealing; the wrapper asserted that it was absent at
  # launch, so any match belongs to this certificate attempt.
  if [[ -n ${EXPECTED_COMMIT-} ]]; then
    docker rm -f "sn56-final-krea-${EXPECTED_COMMIT:0:12}" >/dev/null 2>&1 || true
  fi
  stop_and_reap "${guard_pid}"
  stop_and_reap "${build_pid}"
  {
    printf 'schema=sn56.week5.final-release-cert.v2\n'
    printf 'state=FAIL\n'
    printf 'returncode=%s\n' "${rc}"
    printf 'reason=%s\n' "${reason}"
    printf 'source_commit=%s\n' "${EXPECTED_COMMIT}"
    printf 'certificate_scope=%s\n' "${CERT_SCOPE}"
    printf 'release_policy=%s\n' "${RELEASE_POLICY}"
    printf 'formal_decision_semantic_sha256=%s\n' "${EXPECTED_FORMAL_DECISION_SEMANTIC}"
    printf 'production_activation_sha256=%s\n' "${EXPECTED_ACTIVATION_SEMANTIC}"
    printf 'failed_at_utc=%s\n' "$(date -u +%FT%TZ)"
  } >"${EVIDENCE}/result.env"
  df -hT >"${EVIDENCE}/df-after.txt" 2>&1 || true
  docker info >"${EVIDENCE}/docker-info-after.txt" 2>&1 || true
  write_manifest || true
  chmod -R a-w "${EVIDENCE}" || true
  sync || true
  exit "${rc}"
}

on_error() {
  local rc=$?
  local line=${BASH_LINENO[0]:-unknown}
  seal_failure "${rc}" "error-at-line-${line}"
}

trap on_error ERR
trap 'seal_failure 130 interrupted' INT TERM

check_exact_hash() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "${path}" | awk '{print $1}')
  [[ ${actual} == "${expected}" ]] || {
    printf 'hash mismatch: %s expected=%s actual=%s\n' "${path}" "${expected}" "${actual}" >&2
    return 1
  }
}

date -u +%FT%TZ >"${EVIDENCE}/started-utc.txt"
{
  printf 'schema=sn56.week5.final-release-cert.inputs.v2\n'
  printf 'source_checkout=%s\n' "${REPO}"
  printf 'source_commit=%s\n' "${EXPECTED_COMMIT}"
  printf 'source_tree=%s\n' "${EXPECTED_TREE}"
  printf 'forge_tree=%s\n' "${EXPECTED_FORGE_TREE}"
  printf 'production_trust_sha256=%s\n' "${EXPECTED_PRODUCTION_TRUST}"
  printf 'dockerfile_sha256=%s\n' "${EXPECTED_DOCKERFILE}"
  printf 'lock_sha256=%s\n' "${EXPECTED_LOCK}"
  printf 'constraints_sha256=%s\n' "${EXPECTED_CONSTRAINTS}"
  printf 'verifier_sha256=%s\n' "${EXPECTED_VERIFIER}"
  printf 'origin_url=%s\n' "${EXPECTED_ORIGIN_URL}"
  printf 'remote_ref=%s\n' "${REMOTE_REF}"
  printf 'image_tag=%s\n' "${IMAGE}"
  printf 'expected_containerd_root=%s\n' "${EXPECTED_CONTAINERD_ROOT}"
  printf 'release_policy=%s\n' "${RELEASE_POLICY}"
  printf 'certificate_scope=%s\n' "${CERT_SCOPE}"
  printf 'k5_decision_record_sha256=%s\n' "${EXPECTED_K5_DECISION}"
  printf 'formal_decision_semantic_sha256=%s\n' "${EXPECTED_FORMAL_DECISION_SEMANTIC}"
  printf 'krea_policy_sha256=%s\n' "${EXPECTED_KREA_POLICY}"
  printf 'production_activation_record_sha256=%s\n' "${EXPECTED_ACTIVATION_FILE}"
  printf 'production_activation_sha256=%s\n' "${EXPECTED_ACTIVATION_SEMANTIC}"
  printf 'release_record_sha256=%s\n' "${EXPECTED_RELEASE_RECORD}"
  printf 'legacy_dockerfile_sha256=%s\n' "${EXPECTED_LEGACY_DOCKERFILE}"
  printf 'legacy_no_regression_record_sha256=%s\n' "${EXPECTED_LEGACY_NO_REGRESSION}"
  printf 'krea_probe_sha256=%s\n' "${EXPECTED_KREA_PROBE}"
  printf 'ideogram_mode=%s\n' "${IDEOGRAM_MODE}"
  printf 'ideogram_probe_sha256=none\n'
} >"${EVIDENCE}/bound-inputs.env"

# Freeze the formal K5 result and exact functional-probe wrappers before any
# long-running work. Supplying a path is not enough: every input is hash-bound.
install -d -o root -g root -m 0750 "${EVIDENCE}/decision-inputs"
install -d -o root -g root -m 0750 "${EVIDENCE}/probe-inputs"
install -o root -g root -m 0440 "${K5_DECISION_SOURCE}" "${EVIDENCE}/decision-inputs/k5-formal-result"
install -o root -g root -m 0440 "${ACTIVATION_SOURCE}" "${EVIDENCE}/decision-inputs/production-activation.json"
install -o root -g root -m 0440 "${RELEASE_RECORD_SOURCE}" "${EVIDENCE}/decision-inputs/release-record.json"
install -o root -g root -m 0440 "${LEGACY_NO_REGRESSION_SOURCE}" "${EVIDENCE}/decision-inputs/legacy-no-regression.json"
install -o root -g root -m 0550 "${KREA_PROBE_SOURCE}" "${EVIDENCE}/probe-inputs/krea.sh"
check_exact_hash "${EVIDENCE}/decision-inputs/k5-formal-result" "${EXPECTED_K5_DECISION}"
check_exact_hash "${EVIDENCE}/decision-inputs/production-activation.json" "${EXPECTED_ACTIVATION_FILE}"
check_exact_hash "${EVIDENCE}/decision-inputs/release-record.json" "${EXPECTED_RELEASE_RECORD}"
check_exact_hash "${EVIDENCE}/decision-inputs/legacy-no-regression.json" "${EXPECTED_LEGACY_NO_REGRESSION}"
check_exact_hash "${EVIDENCE}/probe-inputs/krea.sh" "${EXPECTED_KREA_PROBE}"

git_safe=(git -c "safe.directory=${REPO}" -C "${REPO}")
[[ $("${git_safe[@]}" rev-parse HEAD) == "${EXPECTED_COMMIT}" ]]
[[ $("${git_safe[@]}" rev-parse 'HEAD^{tree}') == "${EXPECTED_TREE}" ]]
[[ $("${git_safe[@]}" rev-parse HEAD:forge) == "${EXPECTED_FORGE_TREE}" ]]
[[ -z $("${git_safe[@]}" status --porcelain=v1 --untracked-files=all) ]] || {
  "${git_safe[@]}" status --porcelain=v1 --untracked-files=all >&2
  printf 'release source checkout is not exactly clean\n' >&2
  false
}
actual_origin_url=$("${git_safe[@]}" remote get-url origin)
[[ ${actual_origin_url} == "${EXPECTED_ORIGIN_URL}" ]] || {
  printf 'origin URL differs: expected=%s actual=%s\n' \
    "${EXPECTED_ORIGIN_URL}" "${actual_origin_url:-absent}" >&2
  false
}
remote_commit=$("${git_safe[@]}" ls-remote --exit-code origin "${REMOTE_REF}" | awk 'NR == 1 {print $1}')
[[ ${remote_commit} == "${EXPECTED_COMMIT}" ]] || {
  printf 'origin ref does not resolve to the bound release commit: ref=%s actual=%s\n' \
    "${REMOTE_REF}" "${remote_commit:-absent}" >&2
  false
}

check_exact_hash "${REPO}/${DOCKERFILE_REL}" "${EXPECTED_DOCKERFILE}"
check_exact_hash "${REPO}/${LEGACY_DOCKERFILE_REL}" "${EXPECTED_LEGACY_DOCKERFILE}"
check_exact_hash "${REPO}/${LOCK_REL}" "${EXPECTED_LOCK}"
check_exact_hash "${REPO}/${CONSTRAINTS_REL}" "${EXPECTED_CONSTRAINTS}"
check_exact_hash "${REPO}/${VERIFIER_REL}" "${EXPECTED_VERIFIER}"
actual_production_trust=$("${git_safe[@]}" ls-tree -r --full-tree HEAD forge ops/docker | sha256sum | awk '{print $1}')
[[ ${actual_production_trust} == "${EXPECTED_PRODUCTION_TRUST}" ]] || {
  printf 'production trust mismatch: expected=%s actual=%s\n' \
    "${EXPECTED_PRODUCTION_TRUST}" "${actual_production_trust}" >&2
  false
}

# Validate authority semantics, not merely caller-supplied file hashes.  These
# inputs use one canonical JSON encoding; duplicate keys, NaN, whitespace drift,
# unknown fields, and self-digest drift all fail before Docker is touched.
python3 - \
  "${EVIDENCE}/decision-inputs/k5-formal-result" \
  "${EVIDENCE}/decision-inputs/production-activation.json" \
  "${EVIDENCE}/decision-inputs/release-record.json" \
  "${EVIDENCE}/decision-inputs/legacy-no-regression.json" \
  "${REPO}/${KREA_POLICY_REL}" \
  "${REPO}/${KREA_POLICY_MODULE_REL}" \
  "${EXPECTED_FORMAL_DECISION_SEMANTIC}" \
  "${EXPECTED_KREA_POLICY}" \
  "${EXPECTED_ACTIVATION_SEMANTIC}" \
  "${EXPECTED_RELEASE_RECORD}" \
  "${EXPECTED_COMMIT}" \
  "${EXPECTED_TREE}" \
  "${EXPECTED_FORGE_TREE}" \
  "${EXPECTED_PRODUCTION_TRUST}" \
  "${EXPECTED_LEGACY_DOCKERFILE}" \
  >"${EVIDENCE}/decision-inputs/validated-bindings.json" <<'PY'
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

(
    decision_path,
    activation_path,
    release_path,
    legacy_path,
    policy_path,
    policy_module_path,
    expected_decision_sha,
    expected_policy_sha,
    expected_activation_sha,
    expected_release_file_sha,
    expected_commit,
    expected_tree,
    expected_forge_tree,
    expected_production_trust,
    expected_legacy_dockerfile,
) = sys.argv[1:]

SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load_canonical(path, label):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    assert isinstance(value, dict), f"{label} is not an object"
    assert raw == canonical(value) + b"\n", f"{label} is not canonical JSON"
    return value


def exact(value, keys, label):
    assert set(value) == set(keys), f"{label} keys differ"


decision = load_canonical(decision_path, "formal decision")
exact(
    decision,
    {
        "schema", "kind", "phase", "decided_at_utc", "matrix_sha256",
        "training_plan_set_sha256", "score_queue_sha256", "training_gate",
        "score_gate", "authority", "frozen_dataset_regime_policy",
        "policy_results", "aggregate_bindings", "outcome", "blockers",
        "overall_confirmation_passed", "surprise_review_required",
        "production_dataset_count_router_predeclared",
        "production_routing_authority", "release_family_selected",
        "release_review_required", "production_mutation_authorized",
        "release_authorized", "deployment_authorized", "win_guaranteed",
        "decision_sha256",
    },
    "formal decision",
)
body = {key: value for key, value in decision.items() if key != "decision_sha256"}
assert decision["schema"] == 1
assert decision["kind"] == "forge-krea-stage2-endgame-two-policy-decision"
assert decision["phase"] == "confirmation"
assert decision["decision_sha256"] == expected_decision_sha == canonical_sha(body)
assert set(decision["policy_results"]) == {"K1", "K5"}
k1 = decision["policy_results"]["K1"]
k5 = decision["policy_results"]["K5"]
assert k1["candidate_family_id"] == "K1"
assert k1["outcome"] == "FAIL" and k1["confirmation_passed"] is False
assert isinstance(k1["blockers"], list) and k1["blockers"]
assert k5["candidate_family_id"] == "K5"
assert k5["outcome"] == "PASS" and k5["confirmation_passed"] is True
assert k5["blockers"] == []
assert decision["outcome"] == "FAIL"
assert decision["overall_confirmation_passed"] is False
assert decision["surprise_review_required"] is True
assert decision["production_dataset_count_router_predeclared"] is False
assert decision["production_routing_authority"] is False
assert decision["release_family_selected"] is None
assert decision["release_review_required"] is True
for field in (
    "production_mutation_authorized",
    "release_authorized",
    "deployment_authorized",
    "win_guaranteed",
):
    assert decision[field] is False

policy = load_canonical(policy_path, "Krea policy")
policy_body = {key: value for key, value in policy.items() if key != "policy_sha256"}
assert policy["policy_sha256"] == expected_policy_sha == canonical_sha(policy_body)
assert policy["kind"] == "forge-krea-week5-production-router-predeclaration"
assert policy["activation_contract"]["conditional_release"]["K1_FAIL_K5_PASS"] == "K5_global_at_1/2"

release = load_canonical(release_path, "release record")
exact(
    release,
    {
        "schema", "kind", "release_policy", "formal_endgame_decision_sha256",
        "krea_policy_sha256", "policy_outcomes", "production_mutation_authorized",
        "release_authorized", "deployment_authorized",
    },
    "release record",
)
assert release == {
    "schema": 1,
    "kind": "sn56.week5.krea-k5-global-release-record",
    "release_policy": "K5-global",
    "formal_endgame_decision_sha256": expected_decision_sha,
    "krea_policy_sha256": expected_policy_sha,
    "policy_outcomes": {"K1": "FAIL", "K5": "PASS"},
    "production_mutation_authorized": True,
    "release_authorized": True,
    "deployment_authorized": False,
}
assert hashlib.sha256(Path(release_path).read_bytes()).hexdigest() == expected_release_file_sha

activation = load_canonical(activation_path, "production activation")
exact(
    activation,
    {
        "schema", "kind", "policy_sha256", "formal_endgame_decision_sha256",
        "boundary_plan_sha256s", "release_record_sha256",
        "overall_confirmation_passed", "policy_outcomes",
        "production_mutation_authorized", "release_authorized",
        "deployment_authorized", "activation_sha256",
    },
    "production activation",
)
activation_body = {
    key: value for key, value in activation.items() if key != "activation_sha256"
}
assert activation["schema"] == 1
assert activation["kind"] == "forge-krea-week5-production-router-activation"
assert activation["policy_sha256"] == expected_policy_sha
assert activation["formal_endgame_decision_sha256"] == expected_decision_sha
assert activation["release_record_sha256"] == expected_release_file_sha
assert activation["overall_confirmation_passed"] is False
assert activation["policy_outcomes"] == {"K1": "FAIL", "K5": "PASS"}
assert activation["production_mutation_authorized"] is True
assert activation["release_authorized"] is True
assert activation["deployment_authorized"] is False
assert activation["activation_sha256"] == expected_activation_sha == canonical_sha(activation_body)
required_cells = {
    "B-0p5-small", "B-0p5-large", "B-0p75-small", "B-0p75-large",
    "B-1-small", "B-1-large",
}
assert set(activation["boundary_plan_sha256s"]) == {"K1", "K5"}
for family in ("K1", "K5"):
    rows = activation["boundary_plan_sha256s"][family]
    assert set(rows) == required_cells
    assert all(isinstance(value, str) and SHA256.fullmatch(value) for value in rows.values())

# The reviewed external activation must be the literal record committed into
# the release source; source identity alone does not prove that cross-binding.
tree = ast.parse(Path(policy_module_path).read_text(encoding="utf-8"))
literal = None
found = 0
for node in tree.body:
    target = None
    value = None
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        target, value = node.target.id, node.value
    elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        target, value = node.targets[0].id, node.value
    if target == "PRODUCTION_ACTIVATION":
        found += 1
        literal = ast.literal_eval(value)
assert found == 1 and literal == activation

legacy = load_canonical(legacy_path, "legacy no-regression record")
exact(
    legacy,
    {
        "schema", "kind", "state", "certificate_scope", "source_commit",
        "source_tree", "forge_tree", "production_trust_sha256",
        "legacy_dockerfile_sha256", "legacy_dockerfile_rebuilt",
        "non_krea_noop_verified", "prior_legacy_certificate_manifest_sha256",
        "non_krea_noop_evidence_manifest_sha256", "claim_limit",
    },
    "legacy no-regression record",
)
assert legacy["schema"] == 1
assert legacy["kind"] == "sn56.week5.legacy-flux-no-regression-record"
assert legacy["state"] == "PASS"
assert legacy["certificate_scope"] == "toolkit-krea-only"
assert legacy["source_commit"] == expected_commit
assert legacy["source_tree"] == expected_tree
assert legacy["forge_tree"] == expected_forge_tree
assert legacy["production_trust_sha256"] == expected_production_trust
assert legacy["legacy_dockerfile_sha256"] == expected_legacy_dockerfile
assert legacy["legacy_dockerfile_rebuilt"] is False
assert legacy["non_krea_noop_verified"] is True
assert legacy["claim_limit"] == "legacy FLUX image not rebuilt; this record binds prior certification plus final-tree non-Krea no-op evidence"
for field in (
    "prior_legacy_certificate_manifest_sha256",
    "non_krea_noop_evidence_manifest_sha256",
):
    assert isinstance(legacy[field], str) and SHA256.fullmatch(legacy[field])

summary = {
    "activation_sha256": expected_activation_sha,
    "certificate_scope": "toolkit-krea-only",
    "formal_decision_sha256": expected_decision_sha,
    "krea_policy_sha256": expected_policy_sha,
    "legacy_no_regression": "PASS",
    "policy_outcomes": {"K1": "FAIL", "K5": "PASS"},
    "release_policy": "K5-global",
    "state": "PASS",
}
print(canonical(summary).decode("ascii"))
PY

"${git_safe[@]}" rev-parse HEAD 'HEAD^{tree}' >"${EVIDENCE}/git-identities.txt"
"${git_safe[@]}" status --porcelain=v1 --untracked-files=all >"${EVIDENCE}/git-status-before.txt"
printf '%s\n' "${actual_origin_url}" >"${EVIDENCE}/origin-url.txt"
"${git_safe[@]}" ls-remote --exit-code origin "${REMOTE_REF}" >"${EVIDENCE}/remote-ref.txt"
"${git_safe[@]}" ls-tree -r --full-tree HEAD forge >"${EVIDENCE}/forge-ls-tree.txt"
"${git_safe[@]}" ls-tree -r --full-tree HEAD forge ops/docker >"${EVIDENCE}/production-trust-ls-tree.txt"
sha256sum \
  "${REPO}/${DOCKERFILE_REL}" \
  "${REPO}/${LEGACY_DOCKERFILE_REL}" \
  "${REPO}/${LOCK_REL}" \
  "${REPO}/${CONSTRAINTS_REL}" \
  "${REPO}/${VERIFIER_REL}" \
  >"${EVIDENCE}/source-inputs.sha256"

# Build an exact, content-based Forge manifest from committed files only.
(
  cd "${REPO}"
  while IFS= read -r -d '' forge_path; do
    [[ -f ${forge_path} && ! -L ${forge_path} ]] || {
      printf 'unsupported non-regular committed Forge path: %s\n' "${forge_path}" >&2
      exit 1
    }
    sha256sum -- "${forge_path}"
  done < <(git -c "safe.directory=${REPO}" ls-tree -r --name-only -z HEAD forge | LC_ALL=C sort -z)
) >"${EVIDENCE}/source-forge-files.sha256"
[[ -s ${EVIDENCE}/source-forge-files.sha256 ]]

containerd config dump >"${EVIDENCE}/containerd-config-effective.txt"
actual_containerd_root=$(sed -n -E "s/^root = ['\"]([^'\"]+)['\"]$/\\1/p" \
  "${EVIDENCE}/containerd-config-effective.txt" | head -n 1)
[[ ${actual_containerd_root} == "${EXPECTED_CONTAINERD_ROOT}" ]] || {
  printf 'containerd root changed; no migration is permitted here: expected=%s actual=%s\n' \
    "${EXPECTED_CONTAINERD_ROOT}" "${actual_containerd_root:-unparsed}" >&2
  false
}
[[ $(docker info --format '{{.DockerRootDir}}') == "${DOCKER_ROOT}" ]] || {
  printf 'Docker root must remain %s\n' "${DOCKER_ROOT}" >&2
  false
}
[[ $(systemctl is-active docker) == active ]]
[[ $(systemctl is-active containerd) == active ]]

docker ps --no-trunc >"${EVIDENCE}/docker-ps-before.txt"
[[ -z $(docker ps -q) ]] || {
  docker ps --no-trunc >&2
  printf 'all running campaign containers must be stopped before release build\n' >&2
  false
}
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  printf 'refusing to overwrite an existing release image tag: %s\n' "${IMAGE}" >&2
  false
fi

root_available=$(df --output=avail -B1 / | tail -n 1)
ephemeral_available=$(df --output=avail -B1 /ephemeral | tail -n 1)
evidence_available=$(df --output=avail -B1 "${EVIDENCE_MOUNT}" | tail -n 1)
(( root_available >= ROOT_START_MIN ))
(( ephemeral_available >= EPHEMERAL_START_MIN ))
(( evidence_available >= EVIDENCE_START_MIN ))

df -hT >"${EVIDENCE}/df-before.txt"
findmnt >"${EVIDENCE}/findmnt-before.txt"
docker info >"${EVIDENCE}/docker-info-before.txt"
docker version >"${EVIDENCE}/docker-version.txt"
containerd --version >"${EVIDENCE}/containerd-version.txt"
nvidia-smi -q >"${EVIDENCE}/nvidia-smi-before.txt"

# Fresh production build. This script deliberately does not restart or migrate
# Docker/containerd; the exact roots above are immutable preconditions.
(
  cd "${REPO}"
  exec env -i \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HOME=/nonexistent \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DOCKER_BUILDKIT=1 \
    docker build --no-cache --progress=plain \
    -f "${DOCKERFILE_REL}" \
    -t "${IMAGE}" .
) >"${EVIDENCE}/docker-build.log" 2>&1 &
build_pid=$!
printf '%s\n' "${build_pid}" >"${EVIDENCE}/docker-build.pid"
printf 'utc\troot_available_bytes\tephemeral_available_bytes\tevidence_available_bytes\tbuild_pid_alive\n' \
  >"${EVIDENCE}/pressure-guard.tsv"

(
  # A guard failure is reported to the parent through its return code.  It must
  # never run the parent's sealing trap concurrently with the parent.
  trap - ERR INT TERM
  build_alive() {
    [[ -r /proc/${build_pid}/stat ]] \
      && [[ $(awk '{print $3}' "/proc/${build_pid}/stat") != Z ]]
  }
  while build_alive; do
    root_available=$(df --output=avail -B1 / | tail -n 1)
    ephemeral_available=$(df --output=avail -B1 /ephemeral | tail -n 1)
    evidence_available=$(df --output=avail -B1 "${EVIDENCE_MOUNT}" | tail -n 1)
    printf '%s\t%s\t%s\t%s\ttrue\n' \
      "$(date -u +%FT%TZ)" \
      "${root_available}" \
      "${ephemeral_available}" \
      "${evidence_available}" \
      >>"${EVIDENCE}/pressure-guard.tsv"
    if (( root_available < ROOT_PRESSURE_FLOOR \
      || ephemeral_available < EPHEMERAL_PRESSURE_FLOOR \
      || evidence_available < EVIDENCE_PRESSURE_FLOOR )); then
      {
        printf 'triggered_at_utc=%s\n' "$(date -u +%FT%TZ)"
        printf 'root_available_bytes=%s\n' "${root_available}"
        printf 'root_floor_bytes=%s\n' "${ROOT_PRESSURE_FLOOR}"
        printf 'ephemeral_available_bytes=%s\n' "${ephemeral_available}"
        printf 'ephemeral_floor_bytes=%s\n' "${EPHEMERAL_PRESSURE_FLOOR}"
        printf 'evidence_available_bytes=%s\n' "${evidence_available}"
        printf 'evidence_floor_bytes=%s\n' "${EVIDENCE_PRESSURE_FLOOR}"
      } >"${EVIDENCE}/pressure-guard-trigger.env"
      kill -TERM "${build_pid}" 2>/dev/null || true
      for _ in $(seq 1 30); do
        build_alive || exit 42
        sleep 1
      done
      kill -KILL "${build_pid}" 2>/dev/null || true
      exit 42
    fi
    sleep 10
  done
) &
guard_pid=$!
printf '%s\n' "${guard_pid}" >"${EVIDENCE}/pressure-guard.pid"

set +e
wait "${build_pid}"
build_rc=$?
wait "${guard_pid}"
guard_rc=$?
set -e
printf '%s\n' "${build_rc}" >"${EVIDENCE}/docker-build.returncode"
printf '%s\n' "${guard_rc}" >"${EVIDENCE}/pressure-guard.returncode"
build_pid=''
guard_pid=''
if (( build_rc != 0 || guard_rc != 0 )); then
  seal_failure "$(( build_rc != 0 ? build_rc : guard_rc ))" build-or-pressure-guard-failed
fi

image_id=$(docker image inspect --format '{{.Id}}' "${IMAGE}")
[[ -n ${image_id} && ${image_id} == sha256:* ]]
printf '%s\n' "${image_id}" >"${EVIDENCE}/image-id.txt"
docker image inspect "${IMAGE}" >"${EVIDENCE}/image-inspect.json"
docker history --no-trunc "${IMAGE}" >"${EVIDENCE}/image-history.txt"
docker ps --no-trunc >"${EVIDENCE}/docker-ps-after-build.txt"
[[ -z $(docker ps -q) ]] || {
  docker ps --no-trunc >&2
  printf 'fresh build left an unexpected running container\n' >&2
  false
}

docker run --rm --network none --entrypoint sha256sum "${IMAGE}" \
  /opt/sn56/image-runtime-lock.txt \
  /opt/sn56/image-runtime-phase1-constraints.txt \
  /opt/sn56/verify-image-runtime.py \
  >"${EVIDENCE}/image-runtime-inputs.sha256" \
  2>"${EVIDENCE}/image-runtime-inputs.stderr"
grep -Fxq "${EXPECTED_LOCK}  /opt/sn56/image-runtime-lock.txt" \
  "${EVIDENCE}/image-runtime-inputs.sha256"
grep -Fxq "${EXPECTED_CONSTRAINTS}  /opt/sn56/image-runtime-phase1-constraints.txt" \
  "${EVIDENCE}/image-runtime-inputs.sha256"
grep -Fxq "${EXPECTED_VERIFIER}  /opt/sn56/verify-image-runtime.py" \
  "${EVIDENCE}/image-runtime-inputs.sha256"

docker run --rm --network none --entrypoint python3 "${IMAGE}" \
  /opt/sn56/verify-image-runtime.py \
  --lock /opt/sn56/image-runtime-lock.txt \
  --constraints /opt/sn56/image-runtime-phase1-constraints.txt \
  >"${EVIDENCE}/offline-runtime-verifier.stdout" \
  2>"${EVIDENCE}/offline-runtime-verifier.stderr"
grep -Fxq 'SN56_IMAGE_RUNTIME_INVENTORY=PASS' "${EVIDENCE}/offline-runtime-verifier.stdout"

docker run --rm --network none --gpus 'device=0' --entrypoint python3 "${IMAGE}" -c \
  'import json, torch, forge; assert torch.cuda.is_available(); name=torch.cuda.get_device_name(0); assert "H100" in name; print(json.dumps({"result":"PASS","torch":torch.__version__,"cuda":torch.version.cuda,"device":name,"forge":forge.__file__}, sort_keys=True))' \
  >"${EVIDENCE}/offline-gpu-import.stdout" \
  2>"${EVIDENCE}/offline-gpu-import.stderr"
grep -Fq '"result": "PASS"' "${EVIDENCE}/offline-gpu-import.stdout"

docker run --rm --network none "${IMAGE}" --help \
  >"${EVIDENCE}/offline-cli-help.stdout" \
  2>"${EVIDENCE}/offline-cli-help.stderr"

# Prove that the Forge files embedded in /app are exactly the committed files,
# byte for byte. Build-generated Python caches are excluded from both semantics
# and the comparison; every other regular file must match the committed set.
docker run --rm --network none --entrypoint /bin/bash "${IMAGE}" -ceu '
  cd /app
  export LC_ALL=C
  find forge -type f ! -path "*/__pycache__/*" ! -name "*.pyc" ! -name "*.pyo" -print0 \
    | sort -z \
    | xargs -0 -r sha256sum
' >"${EVIDENCE}/image-forge-files.sha256" \
  2>"${EVIDENCE}/image-forge-files.stderr"
cmp "${EVIDENCE}/source-forge-files.sha256" "${EVIDENCE}/image-forge-files.sha256"

run_functional_probe() {
  local probe_name=$1
  local probe_script=$2
  local probe_evidence=${EVIDENCE}/functional-probes/${probe_name}
  local probe_rc
  local probe_child
  install -d -o root -g root -m 0750 "${probe_evidence}"
  # Empty environment: no HF, GitHub, cloud, ssh-agent, or shell-init secret is
  # inherited into a wrapper or its archived stdout/stderr.
  /usr/bin/setsid --fork --wait /usr/bin/env -i \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HOME=/nonexistent \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    SN56_CERT_PROBE_NAME="${probe_name}" \
    SN56_CERT_IMAGE_TAG="${IMAGE}" \
    SN56_CERT_IMAGE_ID="${image_id}" \
    SN56_CERT_SOURCE_CHECKOUT="${REPO}" \
    SN56_CERT_SOURCE_COMMIT="${EXPECTED_COMMIT}" \
    SN56_CERT_SOURCE_TREE="${EXPECTED_TREE}" \
    SN56_CERT_PRODUCTION_TRUST_SHA256="${EXPECTED_PRODUCTION_TRUST}" \
    SN56_CERT_EVIDENCE_DIR="${probe_evidence}" \
    /usr/bin/bash "${probe_script}" \
    >"${probe_evidence}/wrapper.stdout" \
    2>"${probe_evidence}/wrapper.stderr" &
  probe_pid=$!
  probe_pgid=''
  for _ in $(seq 1 50); do
    probe_child=$(ps -o pid= --ppid "${probe_pid}" | awk 'NR == 1 {gsub(/[[:space:]]/, ""); print}')
    if [[ -n ${probe_child} ]]; then
      probe_pgid=$(ps -o pgid= -p "${probe_child}" | awk 'NR == 1 {gsub(/[[:space:]]/, ""); print}')
      [[ -n ${probe_pgid} ]] && break
    fi
    kill -0 "${probe_pid}" 2>/dev/null || break
    sleep 0.1
  done
  set +e
  wait "${probe_pid}"
  probe_rc=$?
  set -e
  printf '%s\n' "${probe_rc}" >"${probe_evidence}/wrapper.returncode"
  probe_pid=''
  probe_pgid=''
  (( probe_rc == 0 )) || return "${probe_rc}"
  [[ -z $(docker ps -q) ]] || {
    docker ps --no-trunc >&2
    printf '%s probe left a running container\n' "${probe_name}" >&2
    return 1
  }
}

# Decision-time substitutions: each hash-bound wrapper performs the final chosen
# functional check and must write its own detailed evidence under
# SN56_CERT_EVIDENCE_DIR. No command string is eval'd by this harness. Krea is
# mandatory for K5-global.  This certificate fixes Ideogram to the separately
# checked unchanged/deferred state and makes no Ideogram quality claim.
run_functional_probe krea "${EVIDENCE}/probe-inputs/krea.sh"
python3 - \
  "${EVIDENCE}/functional-probes/krea/probe-result.json" \
  "${EVIDENCE}/functional-probes/krea/config-contract.json" \
  "${EVIDENCE}/functional-probes/krea/last.safetensors" \
  "${EVIDENCE}/functional-probes/krea/exact-final.safetensors" \
  "${image_id}" \
  "${EXPECTED_COMMIT}" \
  "${EXPECTED_TREE}" \
  "${EXPECTED_PRODUCTION_TRUST}" \
  "${EXPECTED_FORMAL_DECISION_SEMANTIC}" \
  "${EXPECTED_KREA_POLICY}" \
  "${EXPECTED_ACTIVATION_SEMANTIC}" \
  "${EXPECTED_RELEASE_RECORD}" \
  >"${EVIDENCE}/functional-probes/krea/certificate-validation.json" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

(
    result_path,
    config_contract_path,
    artifact_path,
    exact_final_path,
    expected_image,
    expected_commit,
    expected_tree,
    expected_trust,
    expected_decision,
    expected_policy,
    expected_activation,
    expected_release,
) = sys.argv[1:]


def canonical(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


result_file = Path(result_path)
contract_file = Path(config_contract_path)
artifact_file = Path(artifact_path)
exact_final_file = Path(exact_final_path)
for path in (result_file, contract_file, artifact_file, exact_final_file):
    assert path.is_file() and not path.is_symlink()

raw = result_file.read_bytes()
result = json.loads(raw)
assert raw == canonical(result) + b"\n"
assert set(result) == {
    "activation_mode", "activation_sha256", "artifact_bytes",
    "artifact_sha256", "dataset_sha256", "family",
    "formal_endgame_decision_sha256", "image_id", "kind",
    "learning_rate", "pair_count", "planned_steps", "policy_id",
    "policy_outcomes", "policy_sha256", "private_ledger_proof",
    "production_trust_sha256", "promotion_proof", "release_record_sha256",
    "routing_rows", "schema", "scrub_proof", "selected_step",
    "source_commit", "source_tree", "state",
    "target_fraction",
}
assert result["schema"] == 1
assert result["kind"] == "sn56-week5-krea-release-probe"
assert result["state"] == "PASS"
assert result["image_id"] == expected_image
assert result["source_commit"] == expected_commit
assert result["source_tree"] == expected_tree
assert result["production_trust_sha256"] == expected_trust
assert result["formal_endgame_decision_sha256"] == expected_decision
assert result["policy_id"] == "week5-krea-two-regime-v1"
assert result["policy_sha256"] == expected_policy
assert result["activation_sha256"] == expected_activation
assert result["release_record_sha256"] == expected_release
assert result["policy_outcomes"] == {"K1": "FAIL", "K5": "PASS"}
assert result["activation_mode"] == "K5_global"
assert result["family"] == "K5"
assert result["learning_rate"] == 0.0002
assert result["pair_count"] == 45
assert result["planned_steps"] == 213
assert result["selected_step"] == 108
assert result["target_fraction"] == {"numerator": 1, "denominator": 2}

expected_rows = {
    "small": {
        "activation_mode": "K5_global",
        "family": "K5",
        "learning_rate": 0.0002,
        "pair_count": 20,
        "planned_steps": 209,
        "selected_step": 108,
        "target_fraction": {"numerator": 1, "denominator": 2},
    },
    "large": {
        "activation_mode": "K5_global",
        "family": "K5",
        "learning_rate": 0.0002,
        "pair_count": 45,
        "planned_steps": 213,
        "selected_step": 108,
        "target_fraction": {"numerator": 1, "denominator": 2},
    },
}
assert result["routing_rows"] == expected_rows

contract = json.loads(contract_file.read_bytes())
assert contract["state"] == "PASS"
assert contract["activation_sha256"] == expected_activation
assert contract["formal_endgame_decision_sha256"] == expected_decision
assert contract["release_record_sha256"] == expected_release
assert contract["policy_outcomes"] == {"K1": "FAIL", "K5": "PASS"}
assert contract["rows"] == expected_rows == result["routing_rows"]

artifact_sha = hashlib.sha256(artifact_file.read_bytes()).hexdigest()
assert artifact_file.stat().st_size == result["artifact_bytes"] > 0
assert artifact_sha == result["artifact_sha256"]
proof = result["promotion_proof"]
assert set(proof) == {
    "exact_final_semantic_step", "exact_final_sha256", "last_sha256",
    "source_inodes_distinct", "selected_checkpoint_sha256",
    "selected_checkpoint_step",
}
assert proof["selected_checkpoint_step"] == result["selected_step"] == 108
assert proof["last_sha256"] == proof["selected_checkpoint_sha256"] == artifact_sha
assert proof["exact_final_semantic_step"] == result["planned_steps"] == 213
assert proof["source_inodes_distinct"] is True
exact_final_sha = hashlib.sha256(exact_final_file.read_bytes()).hexdigest()
assert proof["exact_final_sha256"] == exact_final_sha != artifact_sha
assert re.fullmatch(r"[0-9a-f]{64}", result["dataset_sha256"])

private_proof = result["private_ledger_proof"]
assert set(private_proof) == {
    "cuda_available", "gpu_name", "private_record_sha256",
    "selection_record_sha256", "state", "toolkit_returncode",
    "toolkit_stopped_by_deadline",
}
assert private_proof["state"] == "PASS"
assert private_proof["cuda_available"] is True
assert "H100" in private_proof["gpu_name"]
assert private_proof["toolkit_returncode"] == 0
assert private_proof["toolkit_stopped_by_deadline"] is False
assert re.fullmatch(r"[0-9a-f]{64}", private_proof["private_record_sha256"])
assert re.fullmatch(r"[0-9a-f]{64}", private_proof["selection_record_sha256"])

scrub_proof = result["scrub_proof"]
assert set(scrub_proof) == {
    "inventory_entry_count", "inventory_sha256", "state"
}
assert scrub_proof["state"] == "PASS"
assert scrub_proof["inventory_entry_count"] >= 4
assert re.fullmatch(r"[0-9a-f]{64}", scrub_proof["inventory_sha256"])

summary = {
    "activation_mode": "K5_global",
    "artifact_sha256": artifact_sha,
    "large_route": "PASS",
    "natural_completion": "PASS",
    "private_ledger": "PASS",
    "promotion": "PASS",
    "public_scrub": "PASS",
    "small_route": "PASS",
    "state": "PASS",
}
print(canonical(summary).decode("ascii"))
PY
install -d -o root -g root -m 0750 "${EVIDENCE}/functional-probes"
docker run --rm --pull never --network none --read-only \
    --security-opt=no-new-privileges --cap-drop=ALL \
    --entrypoint python3 "${IMAGE}" -c '
import importlib
import json

try:
    policy = importlib.import_module("forge.ideogram_release_policy")
except ModuleNotFoundError as exc:
    if exc.name != "forge.ideogram_release_policy":
        raise
    module_state = "module_absent"
else:
    assert getattr(policy, "PRODUCTION_ACTIVATION", object()) is None
    module_state = "activation_none"
print(json.dumps({
    "claim": "none",
    "kind": "sn56.week5.ideogram-unchanged-deferred-check",
    "module_state": module_state,
    "schema": 1,
    "state": "PASS",
}, allow_nan=False, separators=(",", ":"), sort_keys=True))
' >"${EVIDENCE}/functional-probes/ideogram-dormant-check.json" \
    2>"${EVIDENCE}/functional-probes/ideogram-dormant-check.stderr"
python3 - "${EVIDENCE}/functional-probes/ideogram-dormant-check.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
raw = path.read_bytes()
value = json.loads(raw)
canonical = json.dumps(
    value,
    allow_nan=False,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
).encode("ascii") + b"\n"
assert raw == canonical
assert value["schema"] == 1
assert value["kind"] == "sn56.week5.ideogram-unchanged-deferred-check"
assert value["state"] == "PASS"
assert value["claim"] == "none"
assert value["module_state"] in {"module_absent", "activation_none"}
PY
{
  printf 'schema=sn56.week5.ideogram-disposition.v2\n'
  printf 'state=NOT_RUN\n'
  printf 'reason=unchanged-deferred-by-release-decision\n'
  printf 'claim=none\n'
} >"${EVIDENCE}/functional-probes/ideogram-disposition.env"

[[ $("${git_safe[@]}" rev-parse HEAD) == "${EXPECTED_COMMIT}" ]]
[[ $("${git_safe[@]}" rev-parse 'HEAD^{tree}') == "${EXPECTED_TREE}" ]]
[[ -z $("${git_safe[@]}" status --porcelain=v1 --untracked-files=all) ]]
[[ $(docker image inspect --format '{{.Id}}' "${IMAGE}") == "${image_id}" ]]
[[ -z $(docker ps -q) ]]
actual_origin_after=$("${git_safe[@]}" remote get-url origin)
[[ ${actual_origin_after} == "${EXPECTED_ORIGIN_URL}" ]]
remote_commit_after=$("${git_safe[@]}" ls-remote --exit-code origin "${REMOTE_REF}" | awk 'NR == 1 {print $1}')
[[ ${remote_commit_after} == "${EXPECTED_COMMIT}" ]] || {
  printf 'origin ref moved during certification: ref=%s actual=%s\n' \
    "${REMOTE_REF}" "${remote_commit_after:-absent}" >&2
  false
}
printf '%s\n' "${actual_origin_after}" >"${EVIDENCE}/origin-url-after.txt"
"${git_safe[@]}" ls-remote --exit-code origin "${REMOTE_REF}" >"${EVIDENCE}/remote-ref-after.txt"
"${git_safe[@]}" status --porcelain=v1 --untracked-files=all >"${EVIDENCE}/git-status-after.txt"

df -hT >"${EVIDENCE}/df-after.txt"
docker info >"${EVIDENCE}/docker-info-after.txt"
docker ps --no-trunc >"${EVIDENCE}/docker-ps-after.txt"
date -u +%FT%TZ >"${EVIDENCE}/completed-utc.txt"
{
  printf 'schema=sn56.week5.final-release-cert.v2\n'
  printf 'state=PASS\n'
  printf 'certificate_scope=%s\n' "${CERT_SCOPE}"
  printf 'source_commit=%s\n' "${EXPECTED_COMMIT}"
  printf 'source_tree=%s\n' "${EXPECTED_TREE}"
  printf 'forge_tree=%s\n' "${EXPECTED_FORGE_TREE}"
  printf 'production_trust_sha256=%s\n' "${EXPECTED_PRODUCTION_TRUST}"
  printf 'remote_ref=%s\n' "${REMOTE_REF}"
  printf 'origin_url=%s\n' "${EXPECTED_ORIGIN_URL}"
  printf 'image_tag=%s\n' "${IMAGE}"
  printf 'image_id=%s\n' "${image_id}"
  printf 'containerd_root=%s\n' "${actual_containerd_root}"
  printf 'docker_root=%s\n' "${DOCKER_ROOT}"
  printf 'release_policy=%s\n' "${RELEASE_POLICY}"
  printf 'k5_decision_record_sha256=%s\n' "${EXPECTED_K5_DECISION}"
  printf 'formal_decision_semantic_sha256=%s\n' "${EXPECTED_FORMAL_DECISION_SEMANTIC}"
  printf 'krea_policy_sha256=%s\n' "${EXPECTED_KREA_POLICY}"
  printf 'production_activation_record_sha256=%s\n' "${EXPECTED_ACTIVATION_FILE}"
  printf 'production_activation_sha256=%s\n' "${EXPECTED_ACTIVATION_SEMANTIC}"
  printf 'release_record_sha256=%s\n' "${EXPECTED_RELEASE_RECORD}"
  printf 'legacy_dockerfile_sha256=%s\n' "${EXPECTED_LEGACY_DOCKERFILE}"
  printf 'legacy_no_regression_record_sha256=%s\n' "${EXPECTED_LEGACY_NO_REGRESSION}"
  printf 'krea_probe_sha256=%s\n' "${EXPECTED_KREA_PROBE}"
  printf 'ideogram_mode=%s\n' "${IDEOGRAM_MODE}"
  printf 'ideogram_probe_sha256=none\n'
  printf 'completed_at_utc=%s\n' "$(cat "${EVIDENCE}/completed-utc.txt")"
} >"${EVIDENCE}/result.env"

write_manifest
chmod -R a-w "${EVIDENCE}"
sync
trap - ERR INT TERM
final_manifest_hash_line=$(sha256sum "${EVIDENCE}/MANIFEST.sha256")
final_manifest_sha256=${final_manifest_hash_line%% *}
printf 'FINAL_RELEASE_CERT_PASS image=%s evidence=%s manifest=%s\n' \
  "${image_id}" "${EVIDENCE}" \
  "${final_manifest_sha256}"
