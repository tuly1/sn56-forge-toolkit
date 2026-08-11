# Week-7 public tournament harvest

`safe_harvest_sync.py` is the off-production boundary between the always-on
Hetzner watcher and the local P0 evidence namespace. It is intentionally a
reference-driven synchronizer rather than a mirror:

- it reads the watcher over a fixed, read-only SSH helper;
- it never follows or copies symlinks;
- it copies immutable observation records and only the SHA-256 CAS objects
  reachable from allowed observations;
- it excludes the watcher's `image_text_pairs` fixture namespace unopened,
  because that public pool may contain rows withheld from the optimizer-visible
  `training_data.zip` partition;
- it excludes HF paths and bodies containing hidden, holdout, test,
  quarantine, or eval-derived terminology before publishing them locally;
- it snapshots mutable events and state under a unique UTC name; and
- it writes an immutable completeness ledger and companion SHA-256.

The destination is create-only. Re-running into the same root re-verifies and
reuses matching immutable objects, adds a new timestamped mutable snapshot and
ledger, and never changes prior evidence.

Example (run from a credential-free shell):

```bash
python ops/experiments/week7/safe_harvest_sync.py \
  --ssh-host hetzner \
  --remote-root /opt/sn56-watcher/archive-20260810 \
  --tournament-id tourn_EXACT_ID_20260810 \
  --destination /Users/atulyashetty/Test/SN56-project/evidence/week7-tournament-harvest-20260810/raw-watcher
```

After R1 closes, `--tournament-id` is the required narrow mode. The tool reads
the newest exact `gradients-tournament/<id>` observation, verifies its CAS and
asserted ID, derives the task allowlist from that body, and admits only:

- that tournament's detail/acceptance observations;
- its exact public task observations; and
- HF observations whose repository name contains both that exact tournament
  ID and one of the derived exact task IDs.

Global score pointers, latest-details records, listings, and other tournaments
from the same date are excluded. Filtered mode snapshots matching event rows
and writes a derived state frontier instead of copying the date-wide SQLite DB.
`ROOT-IDENTITY.json` permanently binds a destination to the source archive and
tournament ID; reusing it for another source or tournament aborts before sync.

The path and body filters differ deliberately. A standalone `test` path and
dataset-bearing body names (including singular/plural test/evaluation datasets,
archives, images, assets, prompts, hidden, holdout, and quarantine aliases) are
excluded. Public `test_loss` score telemetry remains allowed. Exact-tournament
syncs also bind every admitted wrapper to its source-specific public URL and a
successful status; public Hugging Face Xet redirects are admitted only under the
HF-owned CDN path and remain independently bound by tree/manifest/file/CAS
identity. Request queries are source-allowlisted: only bounded tree-pagination
parameters and the required signed Xet transport parameters are accepted.
Unexpected, duplicate, malformed, or unscoped query parameters abort the sync;
accepted URLs are published without query values and retain only value-free
redaction metadata (normalized key names and count).

The source watcher, miner endpoint, registration, and production repositories
are never mutated. A `PARTIAL` ledger means at least one allowed observation or
CAS failed integrity/read validation. Policy exclusions are counted but do not
make the run partial; excluded paths are deliberately not repeated in the
ledger.

## Public safetensors headers

`public_safetensors_headers.py` adds the checkpoint-metadata layer without
downloading weight bodies. It consumes only the exact-scoped local snapshot
above, verifies every tree observation and CAS binding, and derives repository,
task, immutable revision, path, LFS SHA-256, and size from those bytes. It then
requests exactly two ranges from each public `.safetensors` object: the 8-byte
length prefix and its bounded JSON header.

The response must be HTTP 206 with exact `Content-Range` and `Content-Length`
headers. HTTP 200 is rejected before reading a byte, preventing a redirect or
server behavior change from turning metadata collection into a multi-gigabyte
weight download. Parsed metadata and tensor name/dtype/shape/offset summaries
are bound to the LFS identity. The terminal verifier independently reparses the
raw header, rejects duplicate keys, and requires non-overlapping contiguous
tensor spans to cover the entire LFS data buffer. Raw header bytes, per-object
records, and each timestamped association inventory are published create-only.

```bash
python ops/experiments/week7/public_safetensors_headers.py \
  --input-root /Users/atulyashetty/Test/SN56-project/evidence/week7-tournament-harvest-20260810/raw-watcher \
  --output-root /Users/atulyashetty/Test/SN56-project/evidence/week7-tournament-harvest-20260810/public-safetensors-headers \
  --tournament-id tourn_EXACT_ID_20260810 \
  --task-id FIRST_EXACT_R1_TASK_UUID \
  --task-id SECOND_EXACT_R1_TASK_UUID
```

The script never reads mutable Hugging Face branches. A path or header matching
the owner-prohibited hidden/holdout/test-data/quarantine vocabulary is never
published; a prohibited checkpoint path is not requested at all. The exact,
non-empty task allowlist is mandatory at the network boundary, persisted in
both the root identity and inventory, and must equal the completed Round-1 API
task set when the terminal package is built.

## Safe public API snapshots

`capture_safe_public_api.py` polls only the exact tournament-details endpoint
and the exact task IDs asserted by that tournament. It retains the complete
source-response SHA-256/size but publishes only an explicit allowlist:
tournament/round state, task model and timing metadata, public participant
hotkeys, repository/submission IDs, scores, ranks, and score reasons. Dataset
URLs, training-row bodies, test-row bodies, and all other fields are discarded;
no dataset URL is followed. Task/participant membership, ImageTask identity,
metric types, finite values, and exact source endpoint hash/size provenance are
fail-closed. Free-text fields are screened before publication. Zero-valued score
placeholders remain `pending` rather than being mislabeled as completed scores.

```bash
python ops/experiments/week7/capture_safe_public_api.py \
  --output-root /Users/atulyashetty/Test/SN56-project/evidence/week7-tournament-harvest-20260810/public-api-safe \
  --tournament-id tourn_EXACT_ID_20260810
```

Each run creates a UTC-named snapshot and checksum. A repeated timestamp with
different bytes aborts rather than overwriting evidence.

## Round-1 P0 package

`build_p0_package.py` is the terminal CPU-only reconciliation step. It refuses
to publish until the exact image tournament reports Round 1 as `completed`.
It then verifies, rather than trusts, the safe API sidecar, the newest
checksum-bound watcher ledger and every ledger-bound observation/CAS object,
and the selected safetensors-header inventory. A newer PARTIAL watcher attempt
cannot be hidden behind an older COMPLETE ledger. Ledger/API/wrapper timestamps
are filename-bound. Repository captures are reconciled against immutable tree
counts, file observations, CAS byte counts, and exact checkpoint/config
identity. Repository prefixes and their embedded task/owner identities are
validated here, at terminal P0 packaging, against the exact tournament, task,
and participant records. The resulting package joins public scores and
submission status to immutable Hugging Face revisions, configs, checkpoint
identities, and bounded metadata.

The training-archive inventory is deliberately not an admission mechanism. It
uses only the exact task's public `training_data.zip`, never the watcher's full
`image_text_pairs` pool, and labels licensing and third-party rights
`unverified` and fixture admission `not admitted`. It never opens a prohibited
hidden, holdout, test, quarantine, or evaluation-derived path, and it never
reads a raw checkpoint body.

Create that inventory only after the safe API snapshot proves Round 1 is
complete:

```bash
python ops/experiments/week7/capture_public_training_archives.py \
  --api-snapshot /absolute/path/to/final-safe-api.json \
  --api-checksum /absolute/path/to/final-safe-api.sha256 \
  --output-root /absolute/path/to/public-training-archives \
  --tournament-id tourn_EXACT_ID_20260810
```

This dedicated collector receives each exact task's public API envelope, then
selects only its `training_data` URL; other dataset fields are neither
persisted nor dereferenced. It inventories ZIP members without extraction, reports
the raw image count and exact-byte-deduplicated image count, and preserves the
archive in a create-only SHA-256 CAS. It never selects or follows `test_data`
or `image_text_pairs` fields.

```bash
python ops/experiments/week7/build_p0_package.py \
  --api-snapshot /absolute/path/to/final-safe-api.json \
  --watcher-root /absolute/path/to/raw-watcher \
  --header-root /absolute/path/to/public-safetensors-headers \
  --header-inventory /absolute/path/to/final-header-inventory.json \
  --training-archive-root /absolute/path/to/public-training-archives \
  --training-archive-inventory /absolute/path/to/final-training-inventory.json \
  --output-root /absolute/path/to/p0-package \
  --tournament-id tourn_EXACT_ID_20260810
```

Outputs are UTC-named, checksum-bound, and create-only. `PARTIAL` is an honest
terminal result when a public surface is missing or lagging; it must not be
silently promoted to COMPLETE by inference.

## Rights-clean HKE fixture candidates

`hke_procedural_renderer.py` is the CPU-only fixture source for the next Krea
screen. It renders 98 first-party rows from integer geometry and a code-owned
bitmap alphabet: social 10 discovery + 8 confirmation, product 28 + 10, and
logo/UI 32 + 10. It has no network, font, reference-image, model-output, or
tournament-content input. Discovery and confirmation use distinct key domains
and must be written to disjoint create-only roots. The public candidate exposes
only confirmation counts and commitments; exact confirmation membership stays
in the custodian tree. Confirmation row IDs, file names, caption variation
tokens, visible variation tokens, and group identities are all derived from the
private confirmation key, so public source plus ordinal counts cannot recreate
them.

```bash
python ops/experiments/week7/hke_procedural_renderer.py build \
  --public-output /absolute/create-only/public-candidate \
  --custodian-output /absolute/create-only/private-confirmation \
  --discovery-key-file /absolute/private/discovery-key.bin \
  --confirmation-key-file /absolute/private/confirmation-key.bin \
  --generator-commit EXACT_PUSHED_40_HEX_COMMIT \
  --generator-tree EXACT_40_HEX_TREE \
  --author-record 'SN56 first-party procedural renderer' \
  --rights-owner 'NAMED RIGHTS OWNER' \
  --license-or-use-grant 'EXACT OWNER-APPROVED USE OR LICENSE RECORD'

python ops/experiments/week7/hke_procedural_renderer.py verify \
  --public-output /absolute/create-only/public-candidate \
  --custodian-output /absolute/create-only/private-confirmation \
  --discovery-key-file /absolute/private/discovery-key.bin \
  --confirmation-key-file /absolute/private/confirmation-key.bin
```

The result is deliberately `candidate_unreviewed`. Machine replay, rights
declarations, and a zero-match exact/pixel/caption/perceptual screen are not
human review.

`hke_fixture_admission.py` enforces that missing authority. A custodian first
creates the private 98-row review template. A named human fills every bound
row check and chooses PASS; `seal-review` validates and seals that exact draft;
then `admit` reruns candidate and keyed replay verification before emitting one
receipt per family. Public receipts omit reviewer identity and confirmation row
identities. The receipt honestly labels the named-human assertion as
operator-attested; it does not cryptographically authenticate a person. Even a
PASS authorizes fixture use only: owner ratification is still required and GPU
and deployment remain false. Review drafts and seals must live outside the
public candidate, candidate-custodian, and repository trees.

```bash
python ops/experiments/week7/hke_fixture_admission.py review-template \
  --public-root /absolute/public-candidate \
  --custodian-root /absolute/private-confirmation \
  --output /absolute/private/review-draft.json

# A named human reviews and edits review-draft.json outside the repository.

python ops/experiments/week7/hke_fixture_admission.py seal-review \
  --public-root /absolute/public-candidate \
  --custodian-root /absolute/private-confirmation \
  --draft /absolute/private/review-draft.json \
  --output /absolute/private/review-sealed.json

python ops/experiments/week7/hke_fixture_admission.py admit \
  --public-root /absolute/public-candidate \
  --custodian-root /absolute/private-confirmation \
  --sealed-review /absolute/private/review-sealed.json \
  --discovery-key-file /absolute/private/discovery-key.bin \
  --confirmation-key-file /absolute/private/confirmation-key.bin \
  --output-root /absolute/create-only/admission-receipts
```

## Calibration-only HKE factor screen

`run_hke_factorial.py` freezes the first controlled screen without adding a
router or checkpoint promoter:

- A: MAE + current dataset-size law;
- B: MSE + current dataset-size law;
- C: MAE + measured clock-fill; and
- D: MSE + measured clock-fill.

The CPU contract is frozen before rental. Clock-fill is materialized only after
the mechanical H100 gate produces internally validated, bundle-bound,
operator-attested profiles for the incumbent runtime and matching dataset
regime. These records establish internal consistency, not independent proof of
measurement, and every profile must name the same H100 80 GB identity. There
is no hard-coded HKE seconds/step fallback. Product and logo/UI use A–D; the
10-row social envelope uses A/D for operational reliability only. Decisions
use terminal checkpoints and paired exact-score rows. A factor reaches
discovery GO only with the same improving direction on both primary families,
a 95% paired interval clearing zero on at least one, and no 1% regression. GO
is not ship authority.

The serialized plan embeds the complete admission-set and complete timing
profile/binding documents. Analysis reruns their semantic validators and
recomputes every A–D config, including the unlaunched social MSE timing source;
a plan digest alone is never treated as evidence. Training, attachment, and
exact-score inputs are complete self-hashed receipt bodies bound to the plan,
config, artifact, fixture, and evaluator. They are still explicitly
operator-attested and do not independently prove that a human or GPU performed
the declared work. The decision record preserves that limitation and carries
no ship authority.

```bash
python ops/experiments/week7/run_hke_factorial.py contract
python ops/experiments/week7/run_hke_factorial.py analyze \
  --plan timing-bound-plan.json \
  --evidence discovery-score-evidence.json
```
