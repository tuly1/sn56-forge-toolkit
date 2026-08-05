# SN56 release authority

This directory is the only source of truth for the Week-6 release certificate
chain. The scripts formerly stored in `SN56-project/scripts/` are compatibility
launchers only.

## Authority contract

- `SN56_RELEASE_COMMIT` is the single release identity. The public launcher
  requires `SN56_RELEASE_EXPECTED_ORIGIN_URL` and a full
  `SN56_RELEASE_REMOTE_REF`, resolves that ref independently with `git
  ls-remote`, and requires the unique remote result, clean checkout `HEAD`,
  release tree, and Forge tree to agree.
- The `/bin/sh` launcher immediately re-execs itself through `/usr/bin/env -i`.
  It archives the selected commit with `/usr/bin/git --no-replace-objects`,
  extracts only canonical regular files/directories into a private `0700`
  workspace, verifies every file's Git blob and executable mode, and proves the
  executing launcher bytes equal the committed launcher. It then execs only the
  archived Bash worker with an explicit environment.
- The archived worker hash-pins and descriptor-stages both the timing validator
  and the generic Week-6 build/GPU delegate. The delegate must reproduce the
  launcher's exact source archive and full materialized-source manifest hashes;
  a matching commit/tree alone is insufficient. There is no executable path
  override, old Week-5 delegation, or validate-only release bypass.
- Every lab-evidence source is opened with nonblocking, no-follow protection,
  hashed while its descriptor bytes are copied, and forwarded by the validated
  content hash. The timing profile and raw record are handed to Forge as those
  exact captured bytes, while the terminal artifact is validated and hashed
  through one descriptor. Forge never reopens those paths. A persistent later
  path replacement fails the final binding check; an A/B/A replacement cannot
  change the evidence Forge consumed.
- The timing receipt must exist and parse as schema 3, kind
  `sn56.week6.operator-attested-timing-provenance.v3`, `state=PASS`, with a
  required machine-enforced `origin=real|synthetic` and the
  exact materialized source path/manifest, release commit, tree, scope, and
  evidence hashes. The delegated `result.env` is parsed as data and must bind
  the same commit, trees, scope, mode, archive, source manifest, and mode-specific
  result state. Production rejects synthetic receipts; CPU integration accepts
  only synthetic receipts. The receipt accelerator identity must exactly match
  the delegate's live H100 observation (or the non-GPU integration sentinel).
- The final Week-6 envelope is schema v4 and is prepared and published with an
  atomic no-replace rename under `SN56_RELEASE_ENVELOPE_BASE`. Production emits
  `PASS`; `cpu-integration` emits only `DRY_RUN_PASS`. Receipt/envelope v1 and
  envelopes v2/v3, timing receipts v1/v2, timing-profile schema 3, and
  raw-runtime schema 4 are
  intentionally invalid.

## Lab/production boundary

Host-bound throughput profiles are operator-attested lab evidence. They do not
leave the lab as tournament inputs. The production trainer does not discover a
profile, raw record, probe flag, accelerator identity, artifact path, or inode
from its environment. It consumes the reviewed `2.2` seconds/step Krea constant
in `forge.recipe.KREA_RELEASE_TIMING_POLICY`; the release certificate binds that
human-readable policy from the exact release diff.

Profiles remain useful for lab attribution. Profile schema 4 is explicitly
`lab-only`, raw runtime schema 5 retains the strict
`bootstrap -> first_checkpoint -> terminal` lifecycle, and the Friday gate log
must show a duration-consistent training window inside the rental interval.

## Scripts

- `../calibration/run_krea_timing_lab.py`: lab-only supervisor for the real
  `bootstrap -> first_checkpoint -> terminal -> profile` lifecycle. It performs
  a live `nvidia-smi` query, requires a new atomically created checkpoint root,
  captures and stages the YAML bytes before any mutation, passes those exact
  bytes through a private real `.yaml` file accepted by the pinned loader,
  executes only a verified archive of the selected runtime commit, emits the
  sealed v3 Friday gate event with `origin=real`, writes evidence atomically
  outside the upload
  tree, and is never imported by a production path.
- `sn56-week6-final-release-cert.sh`: minimal public trust/bootstrap launcher.
- `sn56-week6-final-release-cert-worker.sh`: archived strict Bash timing and
  outer-envelope authority.
- `sn56-week6-validate-timing-provenance.py`: descriptor staging, lab package
  validation, exact receipt/result parsing, materialized-source binding,
  reviewed-constant validation, atomic envelope helpers, and self-tests.
- `sn56-week6-build-gpu-cert.py`: policy-neutral exact-archive no-cache build
  and GPU delegate. Its `cpu-integration` mode stubs only physical host/container
  GPU observations and cannot emit production `PASS`.
- `../../tests/integration/sn56-week6-release-dry-run.sh`: exact-head clean-clone
  wrapper proof. Its companion fixture generator produces explicitly
  non-authoritative `origin=synthetic` schema-current timing input; the wrapper
  still executes all
  CPU/build, receipt, image, and publication gates for real.

CPU tests do not certify a Docker image, GPU execution, evaluator attachment,
or deployment. Those remain release gates.
