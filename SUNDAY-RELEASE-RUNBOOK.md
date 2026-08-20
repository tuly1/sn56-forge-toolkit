# SN56 Week 9 Sunday release

This runbook does not authorize a merge, push, deploy, production repoint,
service change, endpoint change, registration, provider action, or TAO payment.
Each mutating action needs its own explicit authorization. A green contract,
readiness receipt, dry-run, or probe is evidence, not authorization.

Run every repo-relative command below from the isolated release-wiring
worktree. At the start of each new shell, enter it once:

```bash
cd /Users/atulyashetty/Test/SN56-project/workspaces/worktrees/week9-release-wiring-codex
```

## Checked-in state: HOLD fallback target, not a release

The three reviewed release artifacts have separate jobs:

- `release/week9-release-manifest.json` is the exact target, rollback, changed
  surface, source, ref, and production contract. Its checked-in HOLD target is
  the owner-approved T-24 fallback science target
  `40b831a0a0d36cbed2f8e49905ba548768680be4`, tree
  `40f3a78a99f598d205edf76543ac792ccb7d28b1`, tree-record SHA-256
  `14a2be335169db4fd954f60f323a7a21d460d53e4bc1f0224106279e880efc0c`.
  It is a direct child of certified RC
  `bd852dc0986b661983b70a8e2d225b6da0be971e`. The direct-child containment
  delta changes only `forge/tasks/flux_kohya.py` and its three test files; it
  adds no Qwen, Krea, Ideogram, or Z science change. HOLD is not authorization.
- `release/week9-docker-policy.json` is the immutable Docker-byte policy. It is
  anchored to the audited `bd852dc` tree
  `49124005aba5c9fa810814bbcec0c7664726cedf` and the exact bytes below:

  ```text
  ops/docker/standalone-image-toolkit-trainer.dockerfile  3b98fb1cf2b8ec92218bf30848a0f16c8a3ec48c9034bb5984309d67100663a4
  ops/docker/standalone-image-trainer.dockerfile          1b009e67e1eb6f87463cac6f9db986f55dccc7fbc70d86c0719d90739332e0d8
  ```

  Its reviewed file SHA-256 is
  `476ae3c34458ac547607e98c587d7f631bbc60263db3e83f75401b9454dab129`.
  Do not edit, regenerate, or reformat this policy for the `40b831a` fallback
  or a later science candidate. `bd852dc` remains the Docker certification
  source. A candidate with either Dockerfile byte changed is ineligible for
  this release contract and needs a new, separately scoped certification.
- `release/week9-release-readiness.json` is a separate HOLD review receipt. It
  binds the exact raw manifest SHA, raw Docker-policy SHA, target, rollback,
  allowed-change digest/count, and both Docker identities. It must be derived
  again from the eventual canonical READY manifest and independently reviewed
  before its own state can become exactly `ready`.

Changing only the manifest's `release_state` never authorizes a release: the
checked-in readiness receipt remains HOLD and/or fails the exact raw-manifest
binding. Changing only the readiness state also cannot bless a different
manifest or policy. Live mutation and the live probe require both exact READY
artifacts; the CPU mock tests deliberately exercise the checked-in HOLD fixture.

The immutable rollback stays
`75a0a20c2deda82cfa727e082e60a95bea5befb3`. The fixed live surface stays
`hetzner`, `gradients-miner.service`, unit user `miner`, working directory
`/home/miner/god`, and exact `ExecStart`:

```text
/home/miner/.venv/bin/uvicorn miner.asgi:app --host 0.0.0.0 --port 7999 --env-file /home/miner/god/.1.env --log-level info
```

The ASGI module stays `/home/miner/god/miner/asgi.py`; the listener stays
`65.108.77.230:7999`; the exact route stays `/training_repo/image`; source stays
`/home/miner/god/miner/endpoints/training_repo.py`, bytecode
`/home/miner/god/miner/endpoints/__pycache__/training_repo.cpython-312.pyc`,
and TEXT pin `8f11684e30a556b305dec9dd8eec9794bdae8cde`. None is a command-line override.

## T-24 target decision and preparation

The lane cutoff is **2026-08-23 13:00:00 UTC**, exactly T-24 relative to the
planned tournament start. If every proposed science lane is fully certified by
that cutoff, the owner may select one exact clean candidate for the normal
review flow. If not, freeze the Qwen/Krea/Ideogram/Z lanes and use the checked-in
`40b831a0a0d36cbed2f8e49905ba548768680be4` fallback. The fallback still needs
its remaining containment smokes, independent READY receipt, and separate
merge/push/deploy/repoint authorizations; the T-24 choice authorizes none of
those actions.

If integration needs a merge, stop for separate merge authorization first.
Merge authorization does not authorize a push, deploy, or repoint. Preparation
below is local only and keeps every generated artifact non-shippable until an
independent review decision.

For any later fully certified science candidate, supply its exact clean
worktree as data to `--regenerate-from`. Do not edit a target SHA into the
validator, repoint, or probe scripts, and do not make a candidate depend on a
manifest that names the candidate itself. The release manifest is a separate
control artifact; the generator reads candidate `HEAD` and emits a new HOLD
manifest. This is the only preparation interface: no script SHA edits and no
commit/manifest self-reference. The checked-in HOLD manifest already encodes
the `40b831a` fallback; use regeneration only when supplying a different,
fully certified candidate. The output paths must not exist; both generators
are create-only.

```bash
set -euo pipefail
SN56_MANIFEST=release/week9-release-manifest.json
SN56_POLICY=release/week9-docker-policy.json
SN56_READINESS=release/week9-release-readiness.json
SN56_CANDIDATE=/absolute/path/to/exact-clean-fully-certified-candidate
SN56_PREP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sn56-week9-prepare.XXXXXX")"
SN56_HOLD_MANIFEST="$SN56_PREP_DIR/week9-release-manifest.hold.json"
SN56_HOLD_READINESS="$SN56_PREP_DIR/week9-release-readiness.hold.json"

test -z "$(git -C "$SN56_CANDIDATE" status --porcelain=v1 --untracked-files=all)"
test "$(shasum -a 256 "$SN56_POLICY" | awk '{print $1}')" = \
  476ae3c34458ac547607e98c587d7f631bbc60263db3e83f75401b9454dab129
python3 scripts/sn56-release-contract.py \
  --regenerate-from "$SN56_MANIFEST" \
  --candidate-worktree "$SN56_CANDIDATE" \
  --docker-policy "$SN56_POLICY" \
  --output "$SN56_HOLD_MANIFEST"
```

The generator recomputes the exact target commit/tree/tree-record digest and
rollback-to-target name/status entries and digest. It requires the clean
candidate's two Dockerfiles to match the immutable policy, preserves the exact
rollback/ref and production literals, and always emits HOLD.

Independently inspect every generated field and every allowed-change entry.
Install it at `release/week9-release-manifest.json` through the normal reviewed
patch/commit workflow. Only after that review, change exactly
`release_state: "hold"` to `"ready"`, preserving the manifest's indent and
trailing newline, and review that final canonical diff. Any candidate or
manifest-byte change after this point returns preparation to HOLD.

From that exact canonical READY manifest, derive a new HOLD readiness receipt:

```bash
python3 scripts/sn56-release-contract.py \
  --prepare-readiness \
  --manifest "$SN56_MANIFEST" \
  --docker-policy "$SN56_POLICY" \
  --output "$SN56_HOLD_READINESS"
```

A different reviewer must compare the receipt to the canonical manifest and
immutable policy: exact manifest/policy SHA-256 values, target, rollback,
allowed-change base/digest/count, and Docker list. Install it at
`release/week9-release-readiness.json`; only after that independent decision,
change exactly `readiness_state: "hold"` to `"ready"` and review the final
diff. Do not edit any binding field. Commit the reviewed release-wiring files
locally and require a clean worktree. That local commit still authorizes no
push, merge, deploy, or production action.

## Pending real-container GPU containment evidence

The `40b831a` containment delta still needs three real-container GPU smokes.
They are pending and require separate authorization; the CPU suite does not
stand in for them:

1. Run the as-shipped happy Kohya path successfully in the real release
   container on a GPU.
2. Inject a failure after `Popen` while a live escaped descendant exists, and
   prove the leader and descendant are reaped before any ai-toolkit fallback.
3. Inject unverified shutdown and prove execution stops without starting
   ai-toolkit and without leaving a surviving trainer process.

Until all three artifacts are reviewed green, keep the fallback HOLD. A failed
smoke is a stop signal, not permission to weaken containment or switch science
lanes after T-24.

## CPU-only verification

Run the focused release suite after every release-artifact or tooling change
(current result: **84 passed**), then the full suite (current result:
**804 passed, 1 skipped**). These tests use the HOLD fixture and local mocks;
they do not need a GPU, provider, endpoint, service, or public network.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" \
  /Users/atulyashetty/Test/SN56-project/workspaces/repos/forge-toolkit/.venv/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_release_contract.py tests/test_release_probe.py
python3 -c 'p="scripts/sn56-release-contract.py"; compile(open(p, encoding="utf-8").read(), p, "exec")'
bash -n scripts/sn56-week6-repoint.sh \
  scripts/sn56-monday-probe.sh \
  scripts/sn56-preentry-probe-v2.sh
git diff --check
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" \
  /Users/atulyashetty/Test/SN56-project/workspaces/repos/forge-toolkit/.venv/bin/python \
  -m pytest -q -p no:cacheprovider
```

Do not substitute a mock pass for the Sunday live gates. Conversely, do not
run a live probe against the checked-in HOLD fixture; it correctly refuses
before network or host I/O.

## Exact contract and readiness validation

After an authorized push has made the exact target full ref publicly readable,
create a private, non-existing receipt path and run all four read-only gates:

```bash
SN56_VALIDATE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sn56-week9-validate.XXXXXX")"
SN56_CONTRACT_RECEIPT="$SN56_VALIDATE_DIR/full-contract.json"
python3 scripts/sn56-release-contract.py \
  --manifest "$SN56_MANIFEST" \
  --docker-policy "$SN56_POLICY" \
  --contract-only \
  --receipt "$SN56_CONTRACT_RECEIPT"
python3 scripts/sn56-release-contract.py \
  --readiness-only \
  --readiness-receipt "$SN56_READINESS" \
  --validated-contract-receipt "$SN56_CONTRACT_RECEIPT"
bash scripts/sn56-week6-repoint.sh \
  --manifest "$SN56_MANIFEST" --contract-only
bash scripts/sn56-week6-repoint.sh \
  --manifest "$SN56_MANIFEST" --dry-run
```

The full contract re-proves a clean exact local HEAD/tree, the immutable Docker
bytes, the exact target and rollback full refs from a fresh anonymous
zero-credential clone, the exact allowed-change surface, and fixed production
literals. Readiness must print `READINESS PASS`. The repoint dry-run additionally
reads the live host and must see the exact rollback IMAGE pin, unchanged TEXT
pin, active service/listener, the exact route, and a one-line AST edit. It does
not write production. During an authorized apply, the editor hashes the exact
bytes it reads against the sampled preimage; the immediate and final live-file
hashes must equal its one-line output or verified rollback runs. A HOLD
contract may validate and dry-run with a
`NON-SHIPPABLE` banner, but it can never pass READY validation or mutate.

## Deadline, balance, and installed-state gates

- At **2026-08-23 13:00:00 UTC (T-24)**, lock the target decision. Uncertified
  science lanes freeze; the fallback path is exact `40b831a`. It remains
  subject to its pending containment smokes and the complete READY/authorization
  flow.
- Forward repoint has a hard abort at **2026-08-24 12:30:00 UTC**. The script
  checks the clock initially and again immediately before backup and before
  apply. At or after the cutoff it refuses before a forward edit. Do not use
  the cutoff as a target; if recovery time is gone, remain on rollback.
- The planned IMAGE tournament start is **2026-08-24 13:00 UTC**. The live
  probe derives the schedule from live upstream validator source and separately
  requires upstream `main` to equal the tracked reviewed baseline. If either
  gate differs, stop and review; do not waive it with this runbook.
- `entry.balance` must PASS. This runbook authorizes neither a TAO transfer nor
  a registration action.
- The loaded `com.sn56.monday-probe` LaunchAgent still points to the untracked
  external wrapper
  `/Users/atulyashetty/Test/SN56-project/scripts/sn56-monday-probe.sh`, which is
  stale at `ced58e2e3db68f9ca094b4959de7e2f4a812c0ac`. This isolated commit does
  not install, repair, verify, or re-arm that LaunchAgent. Do not rely on it.
  Installing it needs separate deploy authorization; until then, run the
  tracked probe manually at every checkpoint.

## What the live probe proves

Live mode accepts only the exact canonical READY manifest, immutable Docker
policy, and independently bound READY receipt. It uses the tracked
`scripts/sn56-upstream-baseline.env` pin
`f7caab6cb2786f4210036b0675123d6ba9633e8f`; upstream drift is a failure, and
changing that baseline requires a new validator diff review.

The off-host check separately performs an exact GET of
`http://65.108.77.230:7999/training_repo/image`. Success is HTTP 422 with the
Fiber v2.7 missing-header contract for exactly `validator-hotkey`, `signature`,
`miner-hotkey`, and `nonce`. An enum/path 422, partial/extra header errors,
malformed body, HTTP 400, or a different route fails.

The host-side served-pin proof requires the systemd unit to report exact user
`miner`, working directory `/home/miner/god`, and the manifest's exact
`ExecStart`. The active process's `/proc` executable must resolve to the
reviewed virtualenv Python, and its argv must be that interpreter followed by
the exact `ExecStart` argv; its cwd must agree. Its environment must have no
non-empty `PYTHONPATH` that can redirect imports. With the reviewed virtualenv
Python and working directory, import resolution for
`miner.asgi` must land on `/home/miner/god/miner/asgi.py`, and resolution for
`miner.endpoints.training_repo` must land on the manifest's endpoint source.

It also requires the configured IMAGE repository and target SHA exactly once
in source and current bytecode, rollback SHA absent, TEXT SHA unchanged, one
active nonzero `MainPID`, and the port listener owned by that process or its
descendant. The process must have started after the target source was installed.
The same `MainPID` must remain stable before and after a host-loopback GET of the
exact route, whose body must independently satisfy the exact Fiber 422 contract.
The public off-host exchange remains a separate check.

This is strong process, configuration, import-resolution, listener, route, and
pin evidence. It is not an authenticated validator payload: no validator
signing credential is authorized for this work, and the probe deliberately
stops at the missing-authentication-header stage. Do not describe it as proof
of a signed validator request.

## Bare Sunday sequence

Begin only after the T-24 target is locked, with a clean local release-wiring
commit containing its reviewed canonical READY manifest and READY readiness
receipt. If the target is `40b831a`, its three real-container GPU containment
smokes must already be reviewed green. The selected target must already be
integrated and certified; there is no hidden Sunday merge or lane-switch step.

```bash
set -euo pipefail
SN56_MANIFEST=release/week9-release-manifest.json
SN56_POLICY=release/week9-docker-policy.json
SN56_READINESS=release/week9-release-readiness.json
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["release_state"])' "$SN56_MANIFEST")" = ready
test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["readiness_state"])' "$SN56_READINESS")" = ready
test "$(shasum -a 256 "$SN56_POLICY" | awk '{print $1}')" = \
  476ae3c34458ac547607e98c587d7f631bbc60263db3e83f75401b9454dab129
SN56_TARGET_SHA="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["target"]["commit"])' "$SN56_MANIFEST")"
SN56_TARGET_REF="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["target"]["ref"])' "$SN56_MANIFEST")"
SN56_REPO_URL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["source"]["repository_url"])' "$SN56_MANIFEST")"
SN56_REVIEWED_WORKTREE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["source"]["reviewed_worktree"])' "$SN56_MANIFEST")"
test -z "$(git -C "$SN56_REVIEWED_WORKTREE" status --porcelain=v1 --untracked-files=all)"
test "$(git -C "$SN56_REVIEWED_WORKTREE" rev-parse HEAD)" = "$SN56_TARGET_SHA"
```

1. **Push authorization stop.** Read-only inspection is allowed. A differing
   destination is a hard stop. Only after separate push authorization, push
   the exact object to the exact full ref, without force:

   ```bash
   SN56_REMOTE_BEFORE="$(git ls-remote "$SN56_REPO_URL" "$SN56_TARGET_REF" | awk 'NR==1 {print $1}')"
   test -z "$SN56_REMOTE_BEFORE" || test "$SN56_REMOTE_BEFORE" = "$SN56_TARGET_SHA"
   git -C "$SN56_REVIEWED_WORKTREE" push --porcelain "$SN56_REPO_URL" \
     "$SN56_TARGET_SHA:$SN56_TARGET_REF"
   test "$(git ls-remote "$SN56_REPO_URL" "$SN56_TARGET_REF" | awk 'NR==1 {print $1}')" = \
     "$SN56_TARGET_SHA"
   ```

   Publication of the release-wiring commit is also a separate push decision.

2. Run the four commands in **Exact contract and readiness validation**. All
   must pass. The dry-run must still see production on the exact rollback pin.

3. **Deploy authorization stop.** This pin-only release has no implicit image,
   host, provider, service, endpoint, or LaunchAgent deployment. If one is
   proposed, obtain separate deploy authorization and verify it before moving
   on. Merge or push authorization does not authorize deploy.

4. **Production-repoint authorization stop.** Only with separate explicit
   repoint authorization and before the hard abort, run the interactive
   mutation. Live `--yes` is forbidden; type exact `YES` only after rechecking
   the printed manifest, target, rollback, host, route, and remaining runway:

   ```bash
   bash scripts/sn56-week6-repoint.sh --manifest "$SN56_MANIFEST"
   ```

5. Immediately run the tracked live probe. Require `RESULT: GREEN (live)`, zero
   failures, and review every warning; `entry.balance`, `endpoint.reachable`,
   and `endpoint.pin` must each PASS:

   ```bash
   bash scripts/sn56-monday-probe.sh --manifest "$SN56_MANIFEST"
   ```

   Unless a separately authorized LaunchAgent installation has been verified,
   repeat this exact manual command at 06:00 and 07:15 CDT (11:00 and 12:15
   UTC) on 2026-08-24. Do not substitute the stale loaded wrapper.

On any mismatch, stop. Do not improvise a SHA, ref, Docker hash, readiness
receipt, upstream baseline, route, service, or served pin.

## Recovery and offline rollback contract

After the forward script announces that rollback is armed, verification
failures and handled `HUP`, `INT`, or `TERM` signals restore the byte-for-byte
backup, restart the service, and verify the original source/bytecode/route.
The traps remain armed through complete forward verification. This cannot
guarantee recovery from `SIGKILL`, host power loss, control-machine loss, or a
failed restore; retain the printed backup path and be ready for manual recovery.

The identity-only emergency rollback proof is deliberately offline with
respect to the public Git ref and reviewed successor worktree. It still
requires the preserved canonical READY manifest, immutable policy, and exact
READY readiness receipt:

```bash
SN56_ROLLBACK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sn56-week9-rollback.XXXXXX")"
python3 scripts/sn56-release-contract.py \
  --manifest release/week9-release-manifest.json \
  --docker-policy release/week9-docker-policy.json \
  --rollback-contract-only \
  --readiness-receipt release/week9-release-readiness.json \
  --receipt "$SN56_ROLLBACK_DIR/rollback-contract.json"
```

That command proves identity only and does not touch the host. The operator
rollback command runs the same offline contract internally, remains available
after the hard abort or when the public ref is unavailable/worktree is dirty,
and then requires the host to serve either the exact released target or the
exact rollback no-op state. An unknown served pin fails closed.

```bash
bash scripts/sn56-week6-repoint.sh \
  --manifest release/week9-release-manifest.json --rollback
```

Rollback is a separate explicit operator decision, not pre-authorized here.
The checked-in HOLD fallback target cannot run live rollback; production
already rests on its exact rollback pin.
