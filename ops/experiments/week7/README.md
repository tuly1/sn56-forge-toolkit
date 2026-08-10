# Week-7 public tournament harvest

`safe_harvest_sync.py` is the off-production boundary between the always-on
Hetzner watcher and the local P0 evidence namespace. It is intentionally a
reference-driven synchronizer rather than a mirror:

- it reads the watcher over a fixed, read-only SSH helper;
- it never follows or copies symlinks;
- it copies immutable observation records and only the SHA-256 CAS objects
  reachable from allowed observations;
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
- its exact task and fixture observations; and
- HF observations whose repository name contains both that exact tournament
  ID and one of the derived exact task IDs.

Global score pointers, latest-details records, listings, and other tournaments
from the same date are excluded. Filtered mode snapshots matching event rows
and writes a derived state frontier instead of copying the date-wide SQLite DB.
`ROOT-IDENTITY.json` permanently binds a destination to the source archive and
tournament ID; reusing it for another source or tournament aborts before sync.

The path and body filters differ deliberately. A standalone `test` path and
dataset-bearing body names (`test_data`, `test_rows`, `test_set`, `eval_data`,
`evaluation_data`, hidden, holdout, quarantine, and eval-derived) are excluded.
Public `test_loss` score telemetry remains allowed.

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
are bound to the LFS identity. Raw header bytes, per-object records, and each
timestamped association inventory are published create-only.

```bash
python ops/experiments/week7/public_safetensors_headers.py \
  --input-root /Users/atulyashetty/Test/SN56-project/evidence/week7-tournament-harvest-20260810/raw-watcher \
  --output-root /Users/atulyashetty/Test/SN56-project/evidence/week7-tournament-harvest-20260810/public-safetensors-headers \
  --tournament-id tourn_EXACT_ID_20260810
```

The script never reads mutable Hugging Face branches. A path or header matching
the owner-prohibited hidden/holdout/test-data/quarantine vocabulary is never
published; a prohibited checkpoint path is not requested at all.

## Safe public API snapshots

`capture_safe_public_api.py` polls only the exact tournament-details endpoint
and the exact task IDs asserted by that tournament. It retains the complete
source-response SHA-256/size but publishes only an explicit allowlist:
tournament/round state, task model and timing metadata, public participant
hotkeys, repository/submission IDs, scores, ranks, and score reasons. Dataset
URLs, training-row bodies, test-row bodies, and all other fields are discarded;
no dataset URL is followed. Zero-valued score placeholders remain `pending`
rather than being mislabeled as completed scores.

```bash
python ops/experiments/week7/capture_safe_public_api.py \
  --output-root /Users/atulyashetty/Test/SN56-project/evidence/week7-tournament-harvest-20260810/public-api-safe \
  --tournament-id tourn_EXACT_ID_20260810
```

Each run creates a UTC-named snapshot and checksum. A repeated timestamp with
different bytes aborts rather than overwriting evidence.
