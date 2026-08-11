# SN56 FutureBound incident evidence request

Date: 2026-08-10 UTC
Incident status: **UNRESOLVED — ROOT CAUSE NOT ESTABLISHED**
Purpose: provide Gradients with precise identifiers and request the private execution records needed to distinguish assignment, infrastructure, clone/build, trainer, watchdog, output-discovery, and upload failures.

## Scope and interpretation guardrail

This packet is an evidence request, not an incident diagnosis. Public records establish that our FutureBound task produced no public submission or repository. They do **not** establish whether the failure was caused by hardware, assignment/retry handling, source acquisition, image build, container execution, watchdog enforcement, artifact discovery, or upload. No one should attribute a root cause until the corresponding validator-side records are reconciled.

## Incident identifiers

| Field | Value |
|---|---|
| Tournament | `tourn_2695bfced55edc57_20260810` |
| Round | `tourn_2695bfced55edc57_20260810_round_001` |
| Task | `b3854428-2172-4175-ba76-f55c87c2b1fd` |
| Public task URL | `https://api.gradients.io/auditing/tasks/b3854428-2172-4175-ba76-f55c87c2b1fd` |
| Task label / family | `#FutureBound`; social/design |
| Task type | `ImageTask`; `krea2`; model `krea/Krea-2-Raw` |
| Dataset / budget | 10 image-caption pairs; `0.75` hours |
| Our hotkey | `5HLA2QWYy3GpNwCiBCwMMhGw5hpPosSeZ41my8B3nuFDQHNF` |
| Served image commit | `4152f4cd4463df5bc083f534423240e9514c2750` |
| Task created | `2026-08-10T13:08:04.044944Z` |
| Task termination timestamp | `2026-08-10T19:09:35.523166Z` |
| Task updated timestamp | `2026-08-10T19:53:49.537571Z` |
| Final local public observation | `2026-08-10T21:49:15.931497Z` |

## Verified facts

The following are observations, not causal conclusions:

1. The Hetzner endpoint log recorded a successful authenticated validator request at approximately `2026-08-10T13:02Z`, before the task was created.
2. The miner PID remained stable during the relevant window: no service restart was observed, and the service continued reporting successful 256-node metagraph syncs.
3. In the final public task record, our row remained `submission_id=null`, `repo=null`, and `score_reason=null`; it carried no test-loss result.
4. The public Hugging Face harvest found neither a task-linked repository nor an unlinked/retry repository for our hotkey on this task.
5. The same served commit produced scored submissions on the other two Round-1 Krea tasks:
   - `797f3e7b-d7fb-4746-9e90-440d1cf4715d`: submission `4a73586d-c58b-4def-81be-1a48326aa6cc`, repository suffix `...-797f3e7b-d7fb-4746-9e90-440d1cf4715d-5HLA2QWY`;
   - `3bd1ebed-bc40-40ed-a313-227db37653fe`: submission `6b522b9d-6c0a-4fe9-9329-031b04489be4`, repository suffix `...-3bd1ebed-bc40-40ed-a313-227db37653fe-5HLA2QWY`.
6. Multiple tournament jobs were retried following the announced GPU incident, but the public surfaces do not show whether our task was assigned, retried, reassigned, or excluded from that recovery.

These facts make an endpoint-wide outage less likely and make an execution-or-upload failure within the validator-controlled path plausible. They do not distinguish among the possible failure stages.

## Unknowns that public evidence cannot resolve

- Whether an execution attempt was created for our hotkey, and how many attempts or retries existed.
- Which worker and physical GPU received each attempt, including whether either was affected by the announced incident.
- Whether the validator resolved and cloned the intended repository and exact commit.
- Whether the selected Dockerfile built successfully and which image digest executed.
- Whether a container started; if so, how it exited and what it logged.
- Whether a deadline watchdog, OOM handler, worker loss, or manual cancellation terminated the process.
- Whether a valid artifact existed in the expected output directory when execution stopped.
- Whether artifact discovery rejected an otherwise present file.
- Whether repository creation or upload was attempted, retried, rejected, or never reached.

## Evidence requested from Gradients

Please return the following records for the hotkey/task pair above. Redaction of credentials and unrelated tenant data is expected; preserve event timestamps, attempt/request IDs, statuses, exit reasons, paths, sizes, and hashes.

### 1. Assignment and retry history

- Every scheduler/job/attempt ID for this hotkey and task, including parent-child or supersession links.
- Queue, assignment, start, retry, reassignment, cancellation, completion, and terminal-state timestamps.
- The reason recorded for each retry or terminal transition.
- Whether each attempt was included in the announced GPU-incident recovery and, if not, why not.
- Worker/node ID and physical GPU identity for every attempt: GPU model, GPU UUID, driver/runtime version, and any health/retirement event during the attempt.

### 2. Repository resolution, clone, and build

- The training-repository response captured from our endpoint: repository URL, requested/resolved commit SHA, request timestamp, and validator request/correlation ID.
- Clone/fetch/checkout commands or structured equivalents, start/end timestamps, exit status, and complete relevant logs.
- The Dockerfile path selected for `krea2`, build invocation, builder/build ID, cache disposition, base-image digest, resulting image digest, start/end timestamps, exit status, and build logs.
- Any registry, network, disk, or daemon error associated with clone or build.

### 3. Container and trainer execution

- Container/pod/job ID, executed image digest, worker/GPU binding, create/start/stop timestamps, and the exact trainer command with secrets redacted.
- Container exit code and signal; Docker/containerd/Kubernetes termination reason; OOM and GPU Xid/ECC/reset records; host eviction or worker-loss record.
- Complete stdout/stderr and trainer logs for each attempt, including any platform wrapper logs emitted before the trainer started.
- Resource samples sufficient to identify startup failure, OOM, hang, host loss, or deadline termination.

### 4. Watchdog and deadline handling

- Calculated hard and soft deadlines, startup/export reserves, and the watchdog state transitions for each attempt.
- Any signal sent, its timestamp, configured grace period, final kill timestamp, and recorded reason.
- Whether the platform observed a trainer-requested graceful stop or an external forced termination.

### 5. Output-directory state

- A recursive listing of the expected output directory immediately before upload/discovery and at terminal cleanup, including relative path, file type, size, modification time, and SHA-256 where available.
- In particular: `last.safetensors`, numbered checkpoints, temporary/partial files, configuration, success markers, and trainer/recorder output.
- The artifact-discovery decision: expected path or glob, candidate files found, validation performed, and exact reason for accepting or rejecting each candidate.

### 6. Upload and submission response

- Whether uploader/repository creation was invoked; invocation timestamp and uploader job/request ID.
- Intended Hugging Face repository name and resolved revision, if any.
- Each upload/repository API attempt: HTTP/transport status, safe response/error body, retry number/backoff, and terminal result.
- The validator-database submission-creation request and result, including any transaction rollback or reconciliation job.
- If upload was never reached, the preceding state and recorded reason that prevented it.

## Requested response shape

A single chronological attempt table plus attached logs is sufficient. Suggested columns:

`attempt_id | parent_attempt_id | worker_id | gpu_uuid | assigned_at | clone_result | build_result | container_started_at | exit_code/signal | watchdog_reason | output_files | uploader_result | terminal_at`

Please also state which evidence class supports any proposed root cause. If the available records do not discriminate among multiple stages, the correct result is **INCONCLUSIVE**, with the missing records named.

## Local public evidence anchors

- Week-7 P0 handoff: `/Users/atulyashetty/Test/SN56-project/evidence/week7-tournament-harvest-20260810/SN56-WEEK7-P0-HARVEST-HANDOFF-2026-08-10.md`
- Final safe API snapshot: `/Users/atulyashetty/Test/SN56-project/evidence/week7-tournament-harvest-20260810/public-api-safe/snapshots/20260810T214915.931497Z.json`
- Snapshot SHA-256: `7d48b530fe8f384899275259107ebcaf7570c669457f20980b4085c27e37c2db`
- Final watcher sync ledger: `/Users/atulyashetty/Test/SN56-project/evidence/week7-tournament-harvest-20260810/raw-watcher/ledgers/20260810T215621.328448Z.json`
- Ledger SHA-256: `c491e9642f4eedd420f9701ecf3e9dcf4aad202c36cad1b2f2d81a4dc9c6f312`
- Reconciled P0 package: `/Users/atulyashetty/Test/SN56-project/evidence/week7-tournament-harvest-20260810/p0-package/packages/20260810T220727.036459Z.json`
- Package SHA-256: `ae381a950a3dda32490d3102e09d20e371789f0cf6b9003972eba954d98ebfec`

## Claim discipline after receipt

Do not report a FutureBound root cause merely because one stage lacks a public artifact. A root-cause statement requires a validator-side record that identifies the first failed stage and reconciles later stages that consequently did not run. Until then, the incident remains: **no public submission, cause unresolved**.
