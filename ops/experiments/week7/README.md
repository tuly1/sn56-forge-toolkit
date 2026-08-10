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
identity.

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
metric types, finite values, repository prefixes, and exact source endpoint
hash/size provenance are fail-closed. Free-text fields are screened before
publication. Zero-valued score placeholders remain `pending` rather than being
mislabeled as completed scores.

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
identity. The resulting package joins public scores and submission status to
immutable Hugging Face revisions, configs, checkpoint identities, and bounded
metadata.

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
