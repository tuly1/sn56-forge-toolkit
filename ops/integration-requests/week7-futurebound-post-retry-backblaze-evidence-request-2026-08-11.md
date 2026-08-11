# Week-7 FutureBound post-retry Backblaze evidence request

Status: **REQUEST ONLY / NO ROOT-CAUSE CLAIM**  
Date: 2026-08-11  
Scope: validator-side metadata for the original attempt and its successful
retry; no hidden/test rows, credentials, object bodies, or private dataset
content are requested.

## Public anchor

- Tournament: `tourn_2695bfced55edc57_20260810`
- Task: `b3854428-2172-4175-ba76-f55c87c2b1fd`
- Miner hotkey: `5HLA2QWYy3GpNwCiBCwMMhGw5hpPosSeZ41my8B3nuFDQHNF`
- Retry submission: `c9bffbf6-83ff-4b52-b868-97fc7b51d9b2`
- Public repository:
  `gradients-io-tournaments/tournament-tourn_2695bfced55edc57_20260810-b3854428-2172-4175-ba76-f55c87c2b1fd-5HLA2QWY`
- Final public retry result: test loss `0.0658568063599321`, rank `10 of 12`
- Safe public snapshot:
  `/Users/atulyashetty/Test/SN56-project/evidence/week7-post-retry-20260811/public-api-safe/snapshots/20260811T142602.496953Z.json`
- Snapshot SHA-256:
  `a77f3c5da54b1b81c891a3a27ff62a7aa1464a03fbcd2ac20898989e29d0d9f5`

The earlier no-submission observation was a pre-retry snapshot. The successful
retry proves that a scored artifact eventually reached the public repository;
it does not establish why the original attempt failed or which storage event
caused the retry. That original-attempt cause remains **INCONCLUSIVE** on the
public evidence.

## Requested validator-side records

Please provide a redacted, hash-bound event ledger for both the original
attempt and the retry, containing only operational metadata:

1. Internal attempt, worker, job, upload-transaction, and retry-lineage IDs,
   with assignment/start/end/status timestamps and the transition reason.
2. Backblaze bucket/account identifier in redacted or stable pseudonymous
   form, exact object key, object version ID, byte size, content SHA-256 (or
   native immutable checksum), upload start/completion timestamp, and the
   response status for every relevant PUT/copy/finalize request.
3. Any overwrite, tombstone, lifecycle, retention, replication, version-hide,
   or deletion event for those object keys, including event timestamps and
   version IDs.
4. The exact object version selected by the repository-publication worker,
   plus the publication transaction and final submission ID it produced.
5. A causal link, if one exists in the logs, between the original attempt's
   terminal state, the retry assignment, and the successful retry object.

Do not provide secrets, signed URLs, authorization headers, bucket
credentials, hidden/test dataset identifiers, evaluator rows, prompts, or
object bodies. If a requested field cannot be shared, preserve its equality
relationships with a stable redacted identifier and state that it was
withheld.

## Required disposition

The response should distinguish three independently checkable facts:

- whether the original attempt produced and finalized any object version;
- why that version did or did not become the public submission; and
- whether the retry used newly trained bytes, a recovered prior version, or a
  validator-side replay/copy.

Absent that metadata, retain the original-attempt root cause as
**INCONCLUSIVE**. The scored retry remains valid model-quality evidence and
must not be relabeled as a no-submission outcome.
