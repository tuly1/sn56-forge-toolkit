#!/bin/sh
set -eu
umask 077

# Public Week-6 release launcher.  This file performs only the bootstrap trust
# transition: bind one clean checkout to one independently resolved remote ref,
# materialize that commit without worktree bytes, prove this launcher's bytes
# came from that commit, and exec the archived authority worker in a fixed env.

PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH
unset BASH_ENV CDPATH ENV GIT_CONFIG GIT_CONFIG_GLOBAL GIT_DIR GIT_WORK_TREE \
  PYTHONHOME PYTHONPATH PYTHONSTARTUP
IFS=$(/usr/bin/printf ' \t\n_')
IFS=${IFS%_}

fail() {
  /usr/bin/printf 'SN56_WEEK6_FINAL_RELEASE_CERT=FAIL reason=%s\n' "$1" >&2
  exit "${2-1}"
}

require_env() {
  [ -n "$2" ] || fail "required environment variable is unset: $1" 64
  case $2 in
    *'
'*|*''*) fail "required environment variable contains a line break: $1" 64 ;;
  esac
}

[ "$#" -eq 0 ] || fail 'the release launcher accepts no arguments' 64
case $0 in
  /*) launcher_path=$0 ;;
  *) fail 'release launcher must be invoked by its absolute path' 64 ;;
esac

require_env SN56_RELEASE_CERT_MODE "${SN56_RELEASE_CERT_MODE-}"
require_env SN56_RELEASE_COMMIT "${SN56_RELEASE_COMMIT-}"
require_env SN56_RELEASE_SOURCE_CHECKOUT "${SN56_RELEASE_SOURCE_CHECKOUT-}"
require_env SN56_RELEASE_EXPECTED_ORIGIN_URL "${SN56_RELEASE_EXPECTED_ORIGIN_URL-}"
require_env SN56_RELEASE_REMOTE_REF "${SN56_RELEASE_REMOTE_REF-}"
require_env SN56_RELEASE_EVIDENCE_NAMESPACE "${SN56_RELEASE_EVIDENCE_NAMESPACE-}"
require_env SN56_RELEASE_DELEGATE_EVIDENCE_BASE "${SN56_RELEASE_DELEGATE_EVIDENCE_BASE-}"
require_env SN56_RELEASE_ENVELOPE_BASE "${SN56_RELEASE_ENVELOPE_BASE-}"
require_env SN56_RELEASE_WORK_BASE "${SN56_RELEASE_WORK_BASE-}"
require_env SN56_RELEASE_TOOLKIT_IMAGE_TAG "${SN56_RELEASE_TOOLKIT_IMAGE_TAG-}"
require_env SN56_RELEASE_LEGACY_IMAGE_TAG "${SN56_RELEASE_LEGACY_IMAGE_TAG-}"
require_env SN56_RELEASE_EXPECTED_DOCKER_ROOT "${SN56_RELEASE_EXPECTED_DOCKER_ROOT-}"
require_env SN56_RELEASE_EXPECTED_CONTAINERD_ROOT "${SN56_RELEASE_EXPECTED_CONTAINERD_ROOT-}"
require_env SN56_RELEASE_TIMING_PROFILE "${SN56_RELEASE_TIMING_PROFILE-}"
require_env SN56_RELEASE_TIMING_PROFILE_SHA256 "${SN56_RELEASE_TIMING_PROFILE_SHA256-}"
require_env SN56_RELEASE_TIMING_SOURCE_RECORD "${SN56_RELEASE_TIMING_SOURCE_RECORD-}"
require_env SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256 "${SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256-}"
require_env SN56_RELEASE_TIMING_TERMINAL_ARTIFACT "${SN56_RELEASE_TIMING_TERMINAL_ARTIFACT-}"
require_env SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256 "${SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256-}"
require_env SN56_RELEASE_FRIDAY_GATE_LOG "${SN56_RELEASE_FRIDAY_GATE_LOG-}"
require_env SN56_RELEASE_FRIDAY_GATE_LOG_SHA256 "${SN56_RELEASE_FRIDAY_GATE_LOG_SHA256-}"
require_env SN56_RELEASE_TIMING_SOURCE_RUN_ID "${SN56_RELEASE_TIMING_SOURCE_RUN_ID-}"
require_env SN56_RELEASE_H100_GATE_SESSION_ID "${SN56_RELEASE_H100_GATE_SESSION_ID-}"
require_env SN56_RELEASE_H100_RENTAL_STARTED_AT_UTC "${SN56_RELEASE_H100_RENTAL_STARTED_AT_UTC-}"
require_env SN56_RELEASE_H100_RENTAL_ENDED_AT_UTC "${SN56_RELEASE_H100_RENTAL_ENDED_AT_UTC-}"
require_env SN56_RELEASE_TIMING_BUNDLE_ID "${SN56_RELEASE_TIMING_BUNDLE_ID-}"
require_env SN56_RELEASE_TIMING_BUNDLE_SHA256 "${SN56_RELEASE_TIMING_BUNDLE_SHA256-}"
require_env SN56_RELEASE_TIMING_MODEL_TYPE "${SN56_RELEASE_TIMING_MODEL_TYPE-}"
require_env SN56_RELEASE_TIMING_CURRENT_DATASET_SIZE "${SN56_RELEASE_TIMING_CURRENT_DATASET_SIZE-}"
require_env SN56_RELEASE_TIMING_DATASET_REGIME "${SN56_RELEASE_TIMING_DATASET_REGIME-}"

# Run the launcher itself, not merely its children, under one explicit
# allowlist.  The clean sentinel is deliberately not accepted as release input;
# it is created only by this one-time re-exec.
if [ "${SN56_RELEASE_BOOTSTRAP_CLEAN-}" != 1 ]; then
  exec /usr/bin/env -i \
    PATH="${PATH}" \
    HOME=/nonexistent \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    GIT_CONFIG_NOSYSTEM=1 \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_CONFIG_COUNT=8 \
    GIT_CONFIG_KEY_0=core.fsmonitor GIT_CONFIG_VALUE_0=false \
    GIT_CONFIG_KEY_1=core.hooksPath GIT_CONFIG_VALUE_1=/dev/null \
    GIT_CONFIG_KEY_2=core.untrackedCache GIT_CONFIG_VALUE_2=false \
    GIT_CONFIG_KEY_3=core.ignoreStat GIT_CONFIG_VALUE_3=false \
    GIT_CONFIG_KEY_4=core.trustctime GIT_CONFIG_VALUE_4=true \
    GIT_CONFIG_KEY_5=core.checkStat GIT_CONFIG_VALUE_5=default \
    GIT_CONFIG_KEY_6=core.attributesFile GIT_CONFIG_VALUE_6=/dev/null \
    GIT_CONFIG_KEY_7=core.excludesFile GIT_CONFIG_VALUE_7=/dev/null \
    GIT_NO_REPLACE_OBJECTS=1 \
    GIT_TERMINAL_PROMPT=0 \
    SN56_RELEASE_BOOTSTRAP_CLEAN=1 \
    SN56_RELEASE_CERT_MODE="${SN56_RELEASE_CERT_MODE}" \
    SN56_RELEASE_COMMIT="${SN56_RELEASE_COMMIT}" \
    SN56_RELEASE_SOURCE_CHECKOUT="${SN56_RELEASE_SOURCE_CHECKOUT}" \
    SN56_RELEASE_EXPECTED_ORIGIN_URL="${SN56_RELEASE_EXPECTED_ORIGIN_URL}" \
    SN56_RELEASE_REMOTE_REF="${SN56_RELEASE_REMOTE_REF}" \
    SN56_RELEASE_EVIDENCE_NAMESPACE="${SN56_RELEASE_EVIDENCE_NAMESPACE}" \
    SN56_RELEASE_DELEGATE_EVIDENCE_BASE="${SN56_RELEASE_DELEGATE_EVIDENCE_BASE}" \
    SN56_RELEASE_ENVELOPE_BASE="${SN56_RELEASE_ENVELOPE_BASE}" \
    SN56_RELEASE_WORK_BASE="${SN56_RELEASE_WORK_BASE}" \
    SN56_RELEASE_TOOLKIT_IMAGE_TAG="${SN56_RELEASE_TOOLKIT_IMAGE_TAG}" \
    SN56_RELEASE_LEGACY_IMAGE_TAG="${SN56_RELEASE_LEGACY_IMAGE_TAG}" \
    SN56_RELEASE_EXPECTED_DOCKER_ROOT="${SN56_RELEASE_EXPECTED_DOCKER_ROOT}" \
    SN56_RELEASE_EXPECTED_CONTAINERD_ROOT="${SN56_RELEASE_EXPECTED_CONTAINERD_ROOT}" \
    SN56_RELEASE_TIMING_PROFILE="${SN56_RELEASE_TIMING_PROFILE}" \
    SN56_RELEASE_TIMING_PROFILE_SHA256="${SN56_RELEASE_TIMING_PROFILE_SHA256}" \
    SN56_RELEASE_TIMING_SOURCE_RECORD="${SN56_RELEASE_TIMING_SOURCE_RECORD}" \
    SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256="${SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256}" \
    SN56_RELEASE_TIMING_TERMINAL_ARTIFACT="${SN56_RELEASE_TIMING_TERMINAL_ARTIFACT}" \
    SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256="${SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256}" \
    SN56_RELEASE_FRIDAY_GATE_LOG="${SN56_RELEASE_FRIDAY_GATE_LOG}" \
    SN56_RELEASE_FRIDAY_GATE_LOG_SHA256="${SN56_RELEASE_FRIDAY_GATE_LOG_SHA256}" \
    SN56_RELEASE_TIMING_SOURCE_RUN_ID="${SN56_RELEASE_TIMING_SOURCE_RUN_ID}" \
    SN56_RELEASE_H100_GATE_SESSION_ID="${SN56_RELEASE_H100_GATE_SESSION_ID}" \
    SN56_RELEASE_H100_RENTAL_STARTED_AT_UTC="${SN56_RELEASE_H100_RENTAL_STARTED_AT_UTC}" \
    SN56_RELEASE_H100_RENTAL_ENDED_AT_UTC="${SN56_RELEASE_H100_RENTAL_ENDED_AT_UTC}" \
    SN56_RELEASE_TIMING_BUNDLE_ID="${SN56_RELEASE_TIMING_BUNDLE_ID}" \
    SN56_RELEASE_TIMING_BUNDLE_SHA256="${SN56_RELEASE_TIMING_BUNDLE_SHA256}" \
    SN56_RELEASE_TIMING_MODEL_TYPE="${SN56_RELEASE_TIMING_MODEL_TYPE}" \
    SN56_RELEASE_TIMING_CURRENT_DATASET_SIZE="${SN56_RELEASE_TIMING_CURRENT_DATASET_SIZE}" \
    SN56_RELEASE_TIMING_DATASET_REGIME="${SN56_RELEASE_TIMING_DATASET_REGIME}" \
    /bin/sh "${launcher_path}"
fi

case ${SN56_RELEASE_CERT_MODE} in
  production|cpu-integration) ;;
  *) fail 'SN56_RELEASE_CERT_MODE must be production or cpu-integration' 64 ;;
esac
case ${SN56_RELEASE_COMMIT} in
  *[!0-9a-f]*|'') fail 'SN56_RELEASE_COMMIT is not a full lowercase commit id' 64 ;;
esac
[ "${#SN56_RELEASE_COMMIT}" -eq 40 ] || \
  fail 'SN56_RELEASE_COMMIT is not a full lowercase commit id' 64
case ${SN56_RELEASE_EVIDENCE_NAMESPACE} in
  *[!A-Za-z0-9._-]*|'') fail 'invalid release evidence namespace' 64 ;;
esac
[ "${#SN56_RELEASE_EVIDENCE_NAMESPACE}" -le 128 ] || \
  fail 'invalid release evidence namespace' 64
case ${SN56_RELEASE_EVIDENCE_NAMESPACE} in
  [A-Za-z0-9]*) ;;
  *) fail 'invalid release evidence namespace' 64 ;;
esac
case ${SN56_RELEASE_REMOTE_REF} in
  refs/heads/*) ;;
  *) fail 'SN56_RELEASE_REMOTE_REF must be a full refs/heads ref' 64 ;;
esac
case ${SN56_RELEASE_EXPECTED_ORIGIN_URL} in
  https://*|ssh://*|git@*:*) ;;
  *) fail 'SN56_RELEASE_EXPECTED_ORIGIN_URL is not an allowed remote URL' 64 ;;
esac

for digest in \
  "${SN56_RELEASE_TIMING_PROFILE_SHA256}" \
  "${SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256}" \
  "${SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256}" \
  "${SN56_RELEASE_FRIDAY_GATE_LOG_SHA256}" \
  "${SN56_RELEASE_TIMING_BUNDLE_SHA256}"
do
  case ${digest} in
    *[!0-9a-f]*|'') fail 'a required SHA-256 value is malformed' 64 ;;
  esac
  [ "${#digest}" -eq 64 ] || fail 'a required SHA-256 value is malformed' 64
done

fixed_python() {
  /usr/bin/env -i \
    PATH="${PATH}" \
    HOME=/nonexistent \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    /usr/bin/python3 "$@"
}

git_checkout() {
  /usr/bin/env -i \
    PATH="${PATH}" \
    HOME=/nonexistent \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    GIT_CONFIG_NOSYSTEM=1 \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_CONFIG_COUNT=8 \
    GIT_CONFIG_KEY_0=core.fsmonitor GIT_CONFIG_VALUE_0=false \
    GIT_CONFIG_KEY_1=core.hooksPath GIT_CONFIG_VALUE_1=/dev/null \
    GIT_CONFIG_KEY_2=core.untrackedCache GIT_CONFIG_VALUE_2=false \
    GIT_CONFIG_KEY_3=core.ignoreStat GIT_CONFIG_VALUE_3=false \
    GIT_CONFIG_KEY_4=core.trustctime GIT_CONFIG_VALUE_4=true \
    GIT_CONFIG_KEY_5=core.checkStat GIT_CONFIG_VALUE_5=default \
    GIT_CONFIG_KEY_6=core.attributesFile GIT_CONFIG_VALUE_6=/dev/null \
    GIT_CONFIG_KEY_7=core.excludesFile GIT_CONFIG_VALUE_7=/dev/null \
    GIT_NO_REPLACE_OBJECTS=1 \
    GIT_TERMINAL_PROMPT=0 \
    /usr/bin/git --no-replace-objects \
      -c "safe.directory=${SN56_RELEASE_SOURCE_CHECKOUT}" \
      -c core.fsmonitor=false \
      -c core.hooksPath=/dev/null \
      -c core.untrackedCache=false \
      -c core.ignoreStat=false \
      -c core.trustctime=true \
      -c core.checkStat=default \
      -c core.attributesFile=/dev/null \
      -c core.excludesFile=/dev/null \
      -C "${SN56_RELEASE_SOURCE_CHECKOUT}" "$@"
}

git_remote() {
  /usr/bin/env -i \
    PATH="${PATH}" \
    HOME=/nonexistent \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    GIT_CONFIG_NOSYSTEM=1 \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_CONFIG_COUNT=8 \
    GIT_CONFIG_KEY_0=core.fsmonitor GIT_CONFIG_VALUE_0=false \
    GIT_CONFIG_KEY_1=core.hooksPath GIT_CONFIG_VALUE_1=/dev/null \
    GIT_CONFIG_KEY_2=core.untrackedCache GIT_CONFIG_VALUE_2=false \
    GIT_CONFIG_KEY_3=core.ignoreStat GIT_CONFIG_VALUE_3=false \
    GIT_CONFIG_KEY_4=core.trustctime GIT_CONFIG_VALUE_4=true \
    GIT_CONFIG_KEY_5=core.checkStat GIT_CONFIG_VALUE_5=default \
    GIT_CONFIG_KEY_6=core.attributesFile GIT_CONFIG_VALUE_6=/dev/null \
    GIT_CONFIG_KEY_7=core.excludesFile GIT_CONFIG_VALUE_7=/dev/null \
    GIT_NO_REPLACE_OBJECTS=1 \
    GIT_TERMINAL_PROMPT=0 \
    /usr/bin/git --no-replace-objects \
      -c core.fsmonitor=false \
      -c core.hooksPath=/dev/null \
      -c core.untrackedCache=false \
      -c core.ignoreStat=false \
      -c core.trustctime=true \
      -c core.checkStat=default \
      -c core.attributesFile=/dev/null \
      -c core.excludesFile=/dev/null -C / "$@"
}

# Validate all authority directory chains before a temporary directory is
# created.  Evidence bases may be absent only at their final component.
fixed_python - \
  "${launcher_path}" \
  "${SN56_RELEASE_SOURCE_CHECKOUT}" \
  "${SN56_RELEASE_WORK_BASE}" \
  "${SN56_RELEASE_DELEGATE_EVIDENCE_BASE}" \
  "${SN56_RELEASE_ENVELOPE_BASE}" <<'PY'
import os
from pathlib import Path
import stat
import sys

def fail(message: str) -> None:
    raise SystemExit(message)

def lexical(path: str, label: str) -> Path:
    if not path.startswith(os.sep) or os.path.normpath(path) != path:
        fail(f"{label} must be a normalized absolute path")
    return Path(path)

def direct_existing(path: str, label: str, *, directory: bool) -> None:
    candidate = lexical(path, label)
    try:
        resolved = candidate.resolve(strict=True)
        metadata = candidate.lstat()
    except OSError as exc:
        fail(f"{label} is unavailable: {exc}")
    if resolved != candidate or stat.S_ISLNK(metadata.st_mode):
        fail(f"{label} contains a symlink or lexical indirection")
    if directory and not stat.S_ISDIR(metadata.st_mode):
        fail(f"{label} is not a directory")
    if not directory and not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} is not a regular file")

def creatable_leaf(path: str, label: str) -> None:
    candidate = lexical(path, label)
    if candidate.exists() or candidate.is_symlink():
        direct_existing(path, label, directory=True)
    else:
        direct_existing(str(candidate.parent), f"{label} parent", directory=True)

if len(sys.argv) != 6:
    fail("launcher directory preflight argument error")
direct_existing(sys.argv[1], "executed launcher", directory=False)
direct_existing(sys.argv[2], "release checkout", directory=True)
direct_existing(sys.argv[3], "release work base", directory=True)
creatable_leaf(sys.argv[4], "delegate evidence base")
creatable_leaf(sys.argv[5], "outer envelope base")
PY

git_remote check-ref-format "${SN56_RELEASE_REMOTE_REF}" >/dev/null || \
  fail 'SN56_RELEASE_REMOTE_REF is not a valid Git ref' 64

repository_root=$(git_checkout rev-parse --show-toplevel) || \
  fail 'release checkout is not a Git worktree'
[ "${repository_root}" = "${SN56_RELEASE_SOURCE_CHECKOUT}" ] || \
  fail 'release checkout is not the exact repository root'
release_head=$(git_checkout rev-parse --verify 'HEAD^{commit}') || \
  fail 'release checkout HEAD could not be resolved'
[ "${release_head}" = "${SN56_RELEASE_COMMIT}" ] || \
  fail 'release checkout HEAD differs from SN56_RELEASE_COMMIT'
release_tree=$(git_checkout rev-parse --verify 'HEAD^{tree}') || \
  fail 'release tree could not be resolved'
forge_tree=$(git_checkout rev-parse --verify 'HEAD:forge') || \
  fail 'Forge tree could not be resolved'
origin_url=$(git_checkout config --local --no-includes --get-all remote.origin.url) || \
  fail 'release checkout has no origin remote'
[ "${origin_url}" = "${SN56_RELEASE_EXPECTED_ORIGIN_URL}" ] || \
  fail 'release checkout origin differs from SN56_RELEASE_EXPECTED_ORIGIN_URL'

private_workspace=$(
  /usr/bin/env -i PATH="${PATH}" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    /usr/bin/mktemp -d \
      "${SN56_RELEASE_WORK_BASE}/sn56-week6-release.XXXXXXXX"
) || fail 'private release workspace could not be created'
/bin/chmod 0700 "${private_workspace}" || \
  fail 'private release workspace permissions could not be set'
cleanup() {
  /bin/chmod -R u+w "${private_workspace}" 2>/dev/null || :
  /bin/rm -rf -- "${private_workspace}"
}
trap cleanup EXIT HUP INT TERM

remote_rows=${private_workspace}/remote-ref
git_remote ls-remote --exit-code -- \
  "${SN56_RELEASE_EXPECTED_ORIGIN_URL}" "${SN56_RELEASE_REMOTE_REF}" \
  >"${remote_rows}" || fail 'release remote ref could not be resolved independently'
fixed_python - \
  "${remote_rows}" "${SN56_RELEASE_REMOTE_REF}" "${SN56_RELEASE_COMMIT}" <<'PY'
import re
from pathlib import Path
import sys

payload = Path(sys.argv[1]).read_bytes()
expected = f"{sys.argv[3]}\t{sys.argv[2]}\n".encode("utf-8")
if re.fullmatch(rb"[0-9a-f]{40}\trefs/heads/[^\x00-\x20\x7f]+\n", payload) is None:
    raise SystemExit("remote ref did not resolve to one canonical row")
if payload != expected:
    raise SystemExit("remote ref differs from SN56_RELEASE_COMMIT")
PY

archive=${private_workspace}/release.tar
tree_index=${private_workspace}/release-tree.z
materialized_source=${private_workspace}/source
/bin/mkdir -m 0700 "${materialized_source}" || \
  fail 'materialized source directory could not be created'
git_checkout archive --format=tar "${SN56_RELEASE_COMMIT}" >"${archive}" || \
  fail 'exact release archive could not be produced'
[ -s "${archive}" ] || fail 'exact release archive is empty'
archive_commit=$(
  git_remote get-tar-commit-id <"${archive}"
) || fail 'release archive has no embedded commit identity'
[ "${archive_commit}" = "${SN56_RELEASE_COMMIT}" ] || \
  fail 'release archive embedded commit differs'
git_checkout ls-tree -rz --full-tree "${SN56_RELEASE_COMMIT}" \
  >"${tree_index}" || fail 'release tree index could not be produced'

# Extract without tar(1).  Only canonical regular files/directories are
# accepted; every file is checked against its committed Git blob and mode.
materialization_result=${private_workspace}/materialization.env
fixed_python - \
  "${archive}" "${tree_index}" "${materialized_source}" \
  "${launcher_path}" "${SN56_RELEASE_SOURCE_CHECKOUT}" \
  >"${materialization_result}" <<'PY'
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile

def fail(message: str) -> None:
    raise SystemExit(message)

def safe_relative(raw: str, label: str) -> PurePosixPath:
    if any(character in raw for character in ("\x00", "\n", "\r")):
        fail(f"{label} contains a control character")
    value = PurePosixPath(raw)
    if (
        value.is_absolute()
        or str(value) in {"", "."}
        or any(part in {"", ".", ".."} for part in value.parts)
        or value.as_posix() != raw.rstrip("/")
    ):
        fail(f"{label} is unsafe: {raw!r}")
    return value

def file_bytes_nofollow(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        fail(f"{label} could not be opened: {exc}")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            fail(f"{label} is not a nonempty regular file")
        chunks = []
        consumed = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            consumed += len(block)
        after = os.fstat(descriptor)
        identity = lambda item: (
            item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns
        )
        if consumed != before.st_size or identity(before) != identity(after):
            fail(f"{label} changed while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)

if len(sys.argv) != 6:
    fail("release extraction argument error")
archive = Path(sys.argv[1])
index = Path(sys.argv[2])
destination = Path(sys.argv[3])
launcher = Path(sys.argv[4])
repository = Path(sys.argv[5])

committed: dict[str, tuple[str, str]] = {}
for row in index.read_bytes().split(b"\0"):
    if not row:
        continue
    try:
        metadata, raw_path = row.split(b"\t", 1)
        mode_bytes, type_bytes, object_bytes = metadata.split(b" ", 2)
        path_text = raw_path.decode("utf-8", errors="strict")
        mode = mode_bytes.decode("ascii")
        object_type = type_bytes.decode("ascii")
        object_id = object_bytes.decode("ascii")
    except (UnicodeError, ValueError) as exc:
        fail(f"Git tree index is malformed: {exc}")
    relative = safe_relative(path_text, "committed path")
    if object_type != "blob" or mode not in {"100644", "100755"}:
        fail(f"unsupported committed entry: {path_text}")
    if len(object_id) != 40 or any(ch not in "0123456789abcdef" for ch in object_id):
        fail(f"committed object id is malformed: {path_text}")
    if relative.as_posix() in committed:
        fail(f"duplicate committed path: {path_text}")
    committed[relative.as_posix()] = (mode, object_id)
if not committed:
    fail("release commit contains no regular files")

# Never ask Git to classify the worktree: .git/info/attributes plus a
# filter.<name>.clean driver can both falsify that answer and execute an
# attacker command. Compare bytes and executable modes directly with the
# committed blobs instead. The later archive check independently proves the
# exact bytes that are handed to the worker.
worktree: dict[str, tuple[bool, str]] = {}
for current, directory_names, file_names in os.walk(
    repository,
    topdown=True,
    followlinks=False,
):
    current_path = Path(current)
    if current_path == repository:
        directory_names[:] = [name for name in directory_names if name != ".git"]
        file_names = [name for name in file_names if name != ".git"]
    directory_names.sort()
    file_names.sort()
    for name in directory_names:
        candidate = current_path / name
        try:
            metadata = candidate.lstat()
        except OSError as exc:
            fail(f"release worktree directory could not be inspected: {exc}")
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            fail(f"release worktree has an indirect directory: {candidate}")
    for name in file_names:
        candidate = current_path / name
        relative = candidate.relative_to(repository).as_posix()
        if relative in worktree:
            fail(f"release worktree duplicates a path: {relative}")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(candidate, flags)
        except OSError as exc:
            fail(f"release worktree file could not be opened safely: {exc}")
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                fail(f"release worktree entry is not regular: {relative}")
            blob = hashlib.sha1()  # noqa: S324 - Git SHA-1 object identity
            blob.update(f"blob {before.st_size}\0".encode("ascii"))
            consumed = 0
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                consumed += len(block)
                blob.update(block)
            after = os.fstat(descriptor)
            identity = lambda item: (
                item.st_dev,
                item.st_ino,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )
            if consumed != before.st_size or identity(before) != identity(after):
                fail(f"release worktree file changed while read: {relative}")
        finally:
            os.close(descriptor)
        worktree[relative] = (
            bool(before.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)),
            blob.hexdigest(),
        )

if set(worktree) != set(committed):
    fail("release checkout is not clean")
for name, (mode, object_id) in committed.items():
    executable, blob_id = worktree[name]
    if blob_id != object_id or executable != (mode == "100755"):
        fail("release checkout is not clean")

seen_members: set[str] = set()
extracted: dict[str, tuple[str, str]] = {}
with tarfile.open(archive, mode="r:") as bundle:
    members = bundle.getmembers()
    validated = []
    for member in members:
        relative = safe_relative(member.name, "archive member")
        name = relative.as_posix()
        if name in seen_members:
            fail(f"duplicate archive member: {name}")
        seen_members.add(name)
        if not (member.isdir() or member.isreg()):
            fail(f"archive member is not a regular file/directory: {name}")
        validated.append((member, relative))
    for member, relative in sorted(
        validated, key=lambda item: (len(item[1].parts), item[1].as_posix())
    ):
        target = destination.joinpath(*relative.parts)
        if member.isdir():
            if target.exists():
                if not target.is_dir() or target.is_symlink():
                    fail(f"archive directory collides with a file: {relative}")
            else:
                target.mkdir(mode=0o700)
            continue
        name = relative.as_posix()
        if name not in committed:
            fail(f"archive contains a path absent from the commit: {name}")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            fail(f"archive file path collides: {name}")
        source = bundle.extractfile(member)
        if source is None:
            fail(f"archive file is unreadable: {name}")
        mode, object_id = committed[name]
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags, 0o700 if mode == "100755" else 0o600)
        digest = hashlib.sha256()
        git_digest = hashlib.sha1()  # noqa: S324 - Git SHA-1 object identity
        git_digest.update(f"blob {member.size}\0".encode("ascii"))
        consumed = 0
        try:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                consumed += len(block)
                digest.update(block)
                git_digest.update(block)
                view = memoryview(block)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        fail(f"archive extraction made no progress: {name}")
                    view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
            source.close()
        if consumed != member.size or git_digest.hexdigest() != object_id:
            fail(f"archive bytes differ from committed blob: {name}")
        os.chmod(target, 0o700 if mode == "100755" else 0o600)
        extracted[name] = (mode, digest.hexdigest())

if set(extracted) != set(committed):
    missing = sorted(set(committed) - set(extracted))
    extra = sorted(set(extracted) - set(committed))
    fail(f"materialized file set differs: missing={missing[:3]} extra={extra[:3]}")

rows = [
    f"{digest} {mode} {name}\n"
    for name, (mode, digest) in sorted(extracted.items())
]
manifest_sha256 = hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()
committed_launcher = destination / "ops/release/sn56-week6-final-release-cert.sh"
if file_bytes_nofollow(launcher, "executed launcher") != file_bytes_nofollow(
    committed_launcher, "committed launcher"
):
    fail("executed launcher bytes differ from selected committed launcher")

archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
print(f"SN56_RELEASE_MATERIALIZED_MANIFEST_SHA256={manifest_sha256}")
print(f"SN56_RELEASE_ARCHIVE_SHA256={archive_sha256}")
PY

materialized_manifest=''
archive_sha256=''
materialization_rows=0
while IFS='=' read -r name value
do
  materialization_rows=$((materialization_rows + 1))
  case ${name} in
    SN56_RELEASE_MATERIALIZED_MANIFEST_SHA256)
      [ -z "${materialized_manifest}" ] || fail 'duplicate materialized manifest result'
      materialized_manifest=${value}
      ;;
    SN56_RELEASE_ARCHIVE_SHA256)
      [ -z "${archive_sha256}" ] || fail 'duplicate archive hash result'
      archive_sha256=${value}
      ;;
    *) fail 'unexpected materialization result' ;;
  esac
done <"${materialization_result}"
[ "${materialization_rows}" -eq 2 ] || fail 'materialization result is incomplete'
for digest in "${materialized_manifest}" "${archive_sha256}"
do
  case ${digest} in *[!0-9a-f]*|'') fail 'materialization hash is malformed' ;; esac
  [ "${#digest}" -eq 64 ] || fail 'materialization hash is malformed'
done

worker=${materialized_source}/ops/release/sn56-week6-final-release-cert-worker.sh
[ -f "${worker}" ] && [ ! -L "${worker}" ] && [ -x "${worker}" ] || \
  fail 'archived Week-6 release worker is absent or not executable'

exec /usr/bin/env -i \
  PATH="${PATH}" \
  HOME=/nonexistent \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  GIT_CONFIG_NOSYSTEM=1 \
  GIT_CONFIG_GLOBAL=/dev/null \
  GIT_CONFIG_COUNT=8 \
  GIT_CONFIG_KEY_0=core.fsmonitor GIT_CONFIG_VALUE_0=false \
  GIT_CONFIG_KEY_1=core.hooksPath GIT_CONFIG_VALUE_1=/dev/null \
  GIT_CONFIG_KEY_2=core.untrackedCache GIT_CONFIG_VALUE_2=false \
  GIT_CONFIG_KEY_3=core.ignoreStat GIT_CONFIG_VALUE_3=false \
  GIT_CONFIG_KEY_4=core.trustctime GIT_CONFIG_VALUE_4=true \
  GIT_CONFIG_KEY_5=core.checkStat GIT_CONFIG_VALUE_5=default \
  GIT_CONFIG_KEY_6=core.attributesFile GIT_CONFIG_VALUE_6=/dev/null \
  GIT_CONFIG_KEY_7=core.excludesFile GIT_CONFIG_VALUE_7=/dev/null \
  GIT_NO_REPLACE_OBJECTS=1 \
  GIT_TERMINAL_PROMPT=0 \
  SN56_RELEASE_CERT_MODE="${SN56_RELEASE_CERT_MODE}" \
  SN56_RELEASE_COMMIT="${SN56_RELEASE_COMMIT}" \
  SN56_RELEASE_TREE="${release_tree}" \
  SN56_RELEASE_FORGE_TREE="${forge_tree}" \
  SN56_RELEASE_SOURCE_CHECKOUT="${SN56_RELEASE_SOURCE_CHECKOUT}" \
  SN56_RELEASE_EXPECTED_ORIGIN_URL="${SN56_RELEASE_EXPECTED_ORIGIN_URL}" \
  SN56_RELEASE_REMOTE_REF="${SN56_RELEASE_REMOTE_REF}" \
  SN56_RELEASE_EVIDENCE_NAMESPACE="${SN56_RELEASE_EVIDENCE_NAMESPACE}" \
  SN56_RELEASE_DELEGATE_EVIDENCE_BASE="${SN56_RELEASE_DELEGATE_EVIDENCE_BASE}" \
  SN56_RELEASE_ENVELOPE_BASE="${SN56_RELEASE_ENVELOPE_BASE}" \
  SN56_RELEASE_WORK_BASE="${SN56_RELEASE_WORK_BASE}" \
  SN56_RELEASE_PRIVATE_WORKSPACE="${private_workspace}" \
  SN56_RELEASE_MATERIALIZED_SOURCE="${materialized_source}" \
  SN56_RELEASE_MATERIALIZED_MANIFEST_SHA256="${materialized_manifest}" \
  SN56_RELEASE_ARCHIVE_SHA256="${archive_sha256}" \
  SN56_RELEASE_TOOLKIT_IMAGE_TAG="${SN56_RELEASE_TOOLKIT_IMAGE_TAG}" \
  SN56_RELEASE_LEGACY_IMAGE_TAG="${SN56_RELEASE_LEGACY_IMAGE_TAG}" \
  SN56_RELEASE_EXPECTED_DOCKER_ROOT="${SN56_RELEASE_EXPECTED_DOCKER_ROOT}" \
  SN56_RELEASE_EXPECTED_CONTAINERD_ROOT="${SN56_RELEASE_EXPECTED_CONTAINERD_ROOT}" \
  SN56_RELEASE_TIMING_PROFILE="${SN56_RELEASE_TIMING_PROFILE}" \
  SN56_RELEASE_TIMING_PROFILE_SHA256="${SN56_RELEASE_TIMING_PROFILE_SHA256}" \
  SN56_RELEASE_TIMING_SOURCE_RECORD="${SN56_RELEASE_TIMING_SOURCE_RECORD}" \
  SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256="${SN56_RELEASE_TIMING_SOURCE_RECORD_SHA256}" \
  SN56_RELEASE_TIMING_TERMINAL_ARTIFACT="${SN56_RELEASE_TIMING_TERMINAL_ARTIFACT}" \
  SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256="${SN56_RELEASE_TIMING_TERMINAL_ARTIFACT_SHA256}" \
  SN56_RELEASE_FRIDAY_GATE_LOG="${SN56_RELEASE_FRIDAY_GATE_LOG}" \
  SN56_RELEASE_FRIDAY_GATE_LOG_SHA256="${SN56_RELEASE_FRIDAY_GATE_LOG_SHA256}" \
  SN56_RELEASE_TIMING_SOURCE_RUN_ID="${SN56_RELEASE_TIMING_SOURCE_RUN_ID}" \
  SN56_RELEASE_H100_GATE_SESSION_ID="${SN56_RELEASE_H100_GATE_SESSION_ID}" \
  SN56_RELEASE_H100_RENTAL_STARTED_AT_UTC="${SN56_RELEASE_H100_RENTAL_STARTED_AT_UTC}" \
  SN56_RELEASE_H100_RENTAL_ENDED_AT_UTC="${SN56_RELEASE_H100_RENTAL_ENDED_AT_UTC}" \
  SN56_RELEASE_TIMING_BUNDLE_ID="${SN56_RELEASE_TIMING_BUNDLE_ID}" \
  SN56_RELEASE_TIMING_BUNDLE_SHA256="${SN56_RELEASE_TIMING_BUNDLE_SHA256}" \
  SN56_RELEASE_TIMING_MODEL_TYPE="${SN56_RELEASE_TIMING_MODEL_TYPE}" \
  SN56_RELEASE_TIMING_CURRENT_DATASET_SIZE="${SN56_RELEASE_TIMING_CURRENT_DATASET_SIZE}" \
  SN56_RELEASE_TIMING_DATASET_REGIME="${SN56_RELEASE_TIMING_DATASET_REGIME}" \
  /bin/bash "${worker}"
