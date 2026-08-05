# SN56 release authority

This directory is the only source of truth for the Week-6 release certificate
chain. The scripts formerly stored in `SN56-project/scripts/` are compatibility
launchers only.

## Authority contract

- `SN56_RELEASE_COMMIT` is the single release identity. The wrapper derives the
  release tree and Forge subtree from that exact clean checkout and supplies the
  same identity to the timing validator and delegated build/GPU certificate.
- The wrapper hash-pins both executable authority files, copies the exact
  descriptor-opened bytes into a private directory, and executes only those
  copies. There is no delegated-script path override or validate-only release
  bypass.
- Every lab-evidence source is opened with nonblocking, no-follow protection,
  hashed while its descriptor bytes are copied, and forwarded by the validated
  content hash. The timing profile and raw record are handed to Forge as those
  exact captured bytes, while the terminal artifact is validated and hashed
  through one descriptor. Forge never reopens those paths. A persistent later
  path replacement fails the final binding check; an A/B/A replacement cannot
  change the evidence Forge consumed.
- The timing receipt must exist and parse as schema 2, kind
  `sn56.week6.operator-attested-timing-provenance.v2`, `state=PASS`, with the
  exact release commit, tree, scope, and evidence hashes. The delegated
  `result.env` is parsed as data and must bind the same commit, tree, scope, and
  PASS state.
- The final Week-6 envelope is schema v2. Receipt/envelope v1, timing-profile
  schema 3, and raw-runtime schema 4 are intentionally invalid.

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
  bytes to the trainer through an inherited descriptor, re-verifies the runtime
  at launch, emits the sealed v2 Friday gate event, writes evidence atomically
  outside the upload tree, and is never imported by a production path.
- `sn56-week6-final-release-cert.sh`: strict outer authority and envelope.
- `sn56-week6-validate-timing-provenance.py`: descriptor staging, lab package
  validation, receipt/result parsing, reviewed-constant validation, and
  self-tests.
- `sn56-week5-final-release-cert.sh`: hash-pinned delegated no-cache build/GPU
  certificate.

CPU tests do not certify a Docker image, GPU execution, evaluator attachment,
or deployment. Those remain release gates.
