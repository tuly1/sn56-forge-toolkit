# Krea first-GPU timing bootstrap (fail-closed)

This is the only supported order. The bootstrap probe exists precisely because
a final arm plan cannot be built until timing is measured. Production, the
Hetzner endpoint, and the released trainer pins are outside this workflow.

## Immutable inputs prepared before GPU use

- approved D1 or D2 fixture manifest, training archive, and fixture approval;
- parsed Week-5 discovery freeze with the chosen representative arm/class;
- immutable Krea base/runtime/container/host identities;
- a `forge-krea-bootstrap-timing-probe-plan` whose `gpu_execution_authorized`
  remains `false` and whose `command_argv` is the exact runner command below;
- a separate named-human `forge-krea-bootstrap-timing-probe-approval`;
- a timing-margin policy frozen before observing any measurements.

The probe plan is sealed with
`krea_execution_plan.seal_timing_probe_plan`; the approval is created with
`krea_execution_plan.build_timing_probe_approval`. Neither accepts or refers to
a throughput profile or a natural-completion certificate.

The `capture` command itself must run as root on a real Linux host whose PID 1
is systemd. It starts the exact sealed runner inside a uniquely named transient
`forge-krea-timing-*.scope` with `KillMode=control-group`, a manager-side
`RuntimeMaxSec`, recursive TERM-to-KILL cleanup, and mandatory unit collection.
This cgroup contains descendants even if they call `setsid()`. A Docker-only
shell without access to the host systemd manager fails closed before launching
the workload; stage the production runtime paths on the qualifying rootful VM
instead of weakening containment.

Use absolute paths. In the examples, replace `/campaign/...` once, then put the
exact child argv (everything after `--`) byte-for-byte in `command_argv` before
sealing the probe plan.

```bash
PY=/app/venv/bin/python
TOOL=/app/forge/ops/calibration/krea_timing_probe.py
RUNNER=/app/forge/ops/calibration/run_krea_ladder.py
PLAN=/campaign/controls/timing-probe-plan.json
APPROVAL=/campaign/controls/timing-probe-approval.json
CAMPAIGN=/campaign/krea-timing
CHECKPOINT_FS=/app/checkpoints
```

After preparing `/campaign/controls/timing-probe-payload.json` (the exact plan
body without `probe_contract_sha256`), seal it and create the distinct human
approval:

```bash
"$PY" "$TOOL" seal-probe \
  --payload /campaign/controls/timing-probe-payload.json \
  --output "$PLAN"

"$PY" "$TOOL" approve-probe \
  --probe-plan "$PLAN" \
  --reviewer "NAMED HUMAN" \
  --approved-at-utc 2026-07-28T00:00:00Z \
  --output "$APPROVAL"
```

Validate the acyclic pre-run controls before any GPU allocation:

```bash
"$PY" "$TOOL" validate-probe \
  --probe-plan "$PLAN" \
  --probe-approval "$APPROVAL"
```

Freeze a conservative policy *before* capture. These numeric values are an
example only; the named reviewer must choose and approve them in advance:

```bash
"$PY" "$TOOL" seal-margin \
  --reviewer "NAMED HUMAN" \
  --approved-at-utc 2026-07-28T00:00:00Z \
  --multiplier 1.25 \
  --startup-additive-s 5 \
  --optimizer-update-additive-s 0.05 \
  --checkpoint-save-additive-s 2 \
  --finalization-additive-s 10 \
  --upload-additive-s 10 \
  --output /campaign/controls/timing-margin.json
```

## Three measured captures

Run the exact same sealed workload three times. The outer producer passes only
the capture ID/role through its private environment; the runner derives an
isolated task/repo namespace from that ID. It observes actual safetensors
`CREATE -> CLOSE_WRITE` spans through Linux inotify. Atomic-rename-only or fewer
than eight visible saves fail closed; the runner never substitutes a disk-copy
microbenchmark or guessed duration.

```bash
for ID in timing-a timing-b timing-c; do
  "$PY" "$TOOL" capture \
    --probe-plan "$PLAN" \
    --probe-approval "$APPROVAL" \
    --checkpoint-path "$CHECKPOINT_FS" \
    --output "/campaign/evidence/${ID}.json" \
    --capture-id "$ID" \
    --measurement-role timing_measurement \
    --timeout-s 2700 \
    -- "$PY" "$RUNNER" \
      --timing-probe-plan "$PLAN" \
      --timing-probe-approval "$APPROVAL" \
      --campaign-dir "$CAMPAIGN"
done
```

The producer, not the child, timestamps marker receipt using
`time.monotonic_ns()`. The optimizer block records real update units; the raw
manifest therefore requires 100 observed updates, not 100 duplicated rows.

```bash
"$PY" "$TOOL" assemble-raw \
  --capture /campaign/evidence/timing-a.json \
  --capture /campaign/evidence/timing-b.json \
  --capture /campaign/evidence/timing-c.json \
  --output /campaign/evidence/raw-timing.json

"$PY" "$TOOL" verify --kind raw \
  --path /campaign/evidence/raw-timing.json
```

## Separate held-out end-to-end capture

Use a fresh capture ID and role. The command is still the same sealed child
argv. Read the runner's JSON stdout for the exact `condition_record` path.

```bash
"$PY" "$TOOL" capture \
  --probe-plan "$PLAN" \
  --probe-approval "$APPROVAL" \
  --checkpoint-path "$CHECKPOINT_FS" \
  --output /campaign/evidence/heldout-e2e.json \
  --capture-id heldout-e2e \
  --measurement-role held_out_end_to_end \
  --timeout-s 2700 \
  -- "$PY" "$RUNNER" \
    --timing-probe-plan "$PLAN" \
    --timing-probe-approval "$APPROVAL" \
    --campaign-dir "$CAMPAIGN"

"$PY" "$TOOL" build-e2e \
  --capture /campaign/evidence/heldout-e2e.json \
  --run-record /campaign/krea-timing/conditions/EXACT-REPORTED-NAME.json \
  --output /campaign/evidence/heldout-validation.json
```

`build-e2e` reparses telemetry and requires a zero return code, no deadline
stop, exact planned terminal step, one `run_complete`, no failure/fallback, and
outer runtime within the sealed hard budget.

## Build, verify, and consume the profile

```bash
"$PY" "$TOOL" build-profile \
  --raw /campaign/evidence/raw-timing.json \
  --margin /campaign/controls/timing-margin.json \
  --e2e /campaign/evidence/heldout-validation.json \
  --framework-stop-boundary-s 225 \
  --framework-boundary-source-sha256 FULL_SHA256_OF_FROZEN_BOUNDARY_SOURCE \
  --output /campaign/evidence/throughput-profile.json

"$PY" "$TOOL" verify --kind margin \
  --path /campaign/controls/timing-margin.json
"$PY" "$TOOL" verify --kind e2e \
  --path /campaign/evidence/heldout-validation.json
"$PY" "$TOOL" verify --kind profile \
  --path /campaign/evidence/throughput-profile.json
```

Only after those commands pass may an arm execution plan be sealed. Its
`timing_evidence` must bind the raw manifest, margin policy, held-out validation,
probe contract, all three measurement capture records, and the held-out capture
plus its run record. `validate_plan` reloads the producer records, rebuilds both
summaries, and then recomputes the profile; it does not trust supplied counts,
upper bounds, or hashes alone. The final plan receives its own named-human
pre-run approval.

Natural completion of that final arm is certified only after the run with
`build_postrun_certificate`. That certificate is never an input to
`build_approval`, so the authorization graph is acyclic.

Training seed is bound in the probe/final plan, capture manifest, run record,
and campaign evidence. It is deliberately excluded from the reusable compute
runtime identity so an otherwise identical seed-B repeat can reuse a valid
class profile without pretending the seeds are the same experiment.
