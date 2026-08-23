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

## Week 10 exact candidate overlay — build READY, release artifacts HOLD

The historical Week-9 manifest, policy, readiness receipt, and their hashes
remain unchanged. They are not the Week-10 candidate authority. The reviewed
Week-10 artifact set is versioned separately:

```text
release/week10-release-manifest.json
release/week10-candidate-docker-policy.json
release/week10-release-readiness.json
release/sn56-readiness-allowed-signers
```

The target is exact commit
`59e0698c952edaf1bf34a117ecad41bce87517cf`, tree
`613a1cc2d750731df007cc9b2b49e461d0ae368f`, and future durable ref
`refs/heads/week10-trainer-candidate`. The rollback remains exact
`75a0a20c2deda82cfa727e082e60a95bea5befb3`. The rollback-to-target
name/status digest is
`c123df8f8c2543e46b5e5e11d289de962e9ea9f27c9cbe5bdce861543f8492d9`
across the 15 listed paths. The exact Docker SHA-256 values are:

```text
ops/docker/standalone-image-toolkit-trainer.dockerfile  3b98fb1cf2b8ec92218bf30848a0f16c8a3ec48c9034bb5984309d67100663a4
ops/docker/standalone-image-trainer.dockerfile          746fbb5084e2bd09fb3cfa01aef0070835571752cc762c9124e222a7d3a48cce
```

The schema-2 Docker policy is the candidate build/parity receipt. It binds the
Git-index-normalized context manifest
`4f753bb90bd5b4cfb3dcb7c4c4a7b562422a9c0b0b36504aefae105a078127e4`
and completed image
`sha256:fc319058b098e569fe177c0f21f8c65beafed45e391211061fd0c61c1362cf0a`.
It also re-proves the exact pinned base digests `c24f8bb9...3447db8`
(ai-toolkit) and `d34dd575...1beb84e` (Kohya) from the target Dockerfile bytes.
The first empty-store phase reached and verified Step 10 before its strict
1680-second wrapper stopped at final metadata (`1680.005s`); the exact cached
continuation completed that metadata in `18.810s`, for a portfolio-classified
upper bound of `1698.815s` under the validator's 1800-second wall. After the
archive-context mode defect was identified, a fresh context normalized every
tracked file from the Git index and every directory to non-writable canonical
modes. The cache-preserving repaired build passed in `1208.145s` and produced
the image above.

Runtime equivalence authority is deliberately bounded, not whole-rootfs:
exact path/type/mode/uid/gid/symlink/file SHA across `/app/ai-toolkit`,
`/app/forge`, `/opt/sn56/ai-toolkit-python`, and `/opt/sn56`; exact Config
`Entrypoint`, `WorkingDir`, and ordered Env; exact asset hashes; exact dpkg
inventory and pinned toolchain probes. The only raw manifest differences are
narrowly classified `.pyc` headers and Git storage housekeeping; compiled
payload and logical Git receipts are exact. The real entrypoint smoke passed.

These build/runtime gates make the candidate build READY. The checked-in
Week-10 release manifest and readiness receipt remain exactly `hold`, so they
are NON-SHIPPABLE. Do not flip them merely because the build passed. The target
ref is still unpublished, and neither final authority has a reviewed detached
signature. Push, manifest signing, independent readiness review/signing,
deploy, and production repoint each remain separate actions.

Use the three paths together; never pair the Week-10 manifest with the
historical Week-9 policy or receipt:

```bash
SN56_MANIFEST=release/week10-release-manifest.json
SN56_POLICY=release/week10-candidate-docker-policy.json
SN56_READINESS=release/week10-release-readiness.json
SN56_VALIDATE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sn56-week10-contract.XXXXXX")"
SN56_CONTRACT_RECEIPT="$SN56_VALIDATE_DIR/contract.json"

python3 scripts/sn56-release-contract.py \
  --manifest "$SN56_MANIFEST" \
  --docker-policy "$SN56_POLICY" \
  --contract-only \
  --receipt "$SN56_CONTRACT_RECEIPT"
bash scripts/sn56-week6-repoint.sh \
  --manifest "$SN56_MANIFEST" \
  --docker-policy "$SN56_POLICY" \
  --readiness-receipt "$SN56_READINESS" \
  --contract-only
```

Both commands currently fail closed because the selected manifest is HOLD and
the ref is unpublished. Do not use the historical Week-9 preparation section
below for this candidate. After separately authorized publication, review the
one-field manifest transition from `hold` to `ready`, configure exactly one
reviewed manifest key in `release/week9-release-allowed-signers`, and sign the
exact Week-10 manifest bytes with the mechanism's fixed manifest authority:

```bash
SN56_RELEASE_SIGNING_KEY=/absolute/path/to/reviewed-manifest-signing-key
ssh-keygen -Y sign \
  -f "$SN56_RELEASE_SIGNING_KEY" \
  -n sn56-week9-final-manifest \
  "$SN56_MANIFEST"
ssh-keygen -Y verify \
  -f release/week9-release-allowed-signers \
  -I sn56-week9-release \
  -n sn56-week9-final-manifest \
  -s "$SN56_MANIFEST.sig" < "$SN56_MANIFEST"
```

The manifest transition invalidates the checked-in HOLD readiness binding.
Derive a new create-only HOLD receipt from the exact signed READY manifest and
the exact candidate policy; independently review it, install it at
`$SN56_READINESS`, and review its sole state transition to `ready`:

```bash
SN56_READY_PREP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sn56-week10-ready.XXXXXX")"
SN56_REVIEW_READINESS="$SN56_READY_PREP_DIR/week10-readiness.hold.json"
python3 scripts/sn56-release-contract.py \
  --prepare-readiness \
  --manifest "$SN56_MANIFEST" \
  --docker-policy "$SN56_POLICY" \
  --output "$SN56_REVIEW_READINESS"
```

Configure exactly one option-free entry in
`release/sn56-readiness-allowed-signers` for a key whose public key differs
from the manifest authority. The validator enforces that distinction. Sign and
verify the exact reviewed READY receipt:

```bash
SN56_READINESS_SIGNING_KEY=/absolute/path/to/independent-readiness-signing-key
ssh-keygen -Y sign \
  -f "$SN56_READINESS_SIGNING_KEY" \
  -n sn56-final-readiness \
  "$SN56_READINESS"
ssh-keygen -Y verify \
  -f release/sn56-readiness-allowed-signers \
  -I sn56-release-readiness \
  -n sn56-final-readiness \
  -s "$SN56_READINESS.sig" < "$SN56_READINESS"
```

After those separately reviewed steps, the same explicit paths are used for
dry-run, probe, repoint, and rollback:

```bash
bash scripts/sn56-week6-repoint.sh \
  --manifest "$SN56_MANIFEST" --docker-policy "$SN56_POLICY" \
  --readiness-receipt "$SN56_READINESS" --dry-run
bash scripts/sn56-monday-probe.sh \
  --manifest "$SN56_MANIFEST" --docker-policy "$SN56_POLICY" \
  --readiness-receipt "$SN56_READINESS"
bash scripts/sn56-week6-rollback.sh \
  --manifest "$SN56_MANIFEST" --docker-policy "$SN56_POLICY" \
  --readiness-receipt "$SN56_READINESS"
```

## Historical Week-9 checked-in state: unselected HOLD, not a release

The three reviewed release artifacts have separate jobs:

- `release/week9-release-manifest.json` is the final-target template. Its
  all-zero commit/tree fields, empty changed surface, and explicit
  `/REQUIRED/FINAL/...` worktree are sentinels, not Git identities. Contract,
  probe, rollback, and mutation modes all fail closed while those sentinels
  remain. No science winner or fallback candidate is guessed in tooling.
- `release/week9-docker-policy.json` is the immutable Docker-byte policy. It is
  anchored to the audited `bd852dc` tree
  `49124005aba5c9fa810814bbcec0c7664726cedf` and the exact bytes below:

  ```text
  ops/docker/standalone-image-toolkit-trainer.dockerfile  3b98fb1cf2b8ec92218bf30848a0f16c8a3ec48c9034bb5984309d67100663a4
  ops/docker/standalone-image-trainer.dockerfile          1b009e67e1eb6f87463cac6f9db986f55dccc7fbc70d86c0719d90739332e0d8
  ```

  Its reviewed file SHA-256 is
  `476ae3c34458ac547607e98c587d7f631bbc60263db3e83f75401b9454dab129`.
  Do not edit, regenerate, or reformat this policy for a science candidate.
  `bd852dc` remains the Docker certification
  source. A candidate with either Dockerfile byte changed is ineligible for
  this release contract and needs a new, separately scoped certification.
- `release/week9-release-readiness.json` is a separate HOLD review receipt. It
  binds the exact raw manifest SHA, raw Docker-policy SHA, target, rollback,
  allowed-change digest/count, and both Docker identities. It must be derived
  again from the eventual canonical READY manifest and independently reviewed
  before its own state can become exactly `ready`.
- `release/week9-release-allowed-signers` is the fixed trust root for the
  detached `week9-release-manifest.json.sig`. It intentionally contains no key
  in the checked-in HOLD state. Add exactly one independently reviewed OpenSSH
  allowed-signers entry for principal `sn56-week9-release`; live validation
  requires a good signature in namespace `sn56-week9-final-manifest`.
- `release/sn56-readiness-allowed-signers` is the separate, candidate-agnostic
  trust root for the exact READY receipt. It also contains no key while HOLD.
  Live validation requires principal `sn56-release-readiness`, namespace
  `sn56-final-readiness`, and the detached `$SN56_READINESS.sig` bytes.

Changing only the manifest's `release_state` never authorizes a release: its
detached signature is missing/invalid, and the
checked-in readiness receipt remains HOLD and/or fails the exact raw-manifest
binding. Changing only the readiness state also cannot bless a different
manifest or policy because an exact independently trusted signature is required.
Live mutation and the live probe require both exact signed READY artifacts; the
CPU mock tests deliberately exercise the checked-in HOLD fixture.

The required live prestate and immutable rollback both stay
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
and TEXT pin `8f11684e30a556b305dec9dd8eec9794bdae8cde`. The endpoint mapping must contain
exactly IMAGE and TEXT; ENVIRONMENT remains absent. None is a command-line
override.

Release tooling never deletes provider resources or persistent volumes. In
particular, Hyperstack volume `47261` is preserve-forever state and is outside
every release, rollback, cleanup, and cost-saving command in this runbook.

## Historical Week-9 fallback preparation — not the Week-10 Sunday path

This retained block documents how the old all-zero Week-9 template could be
regenerated for a separately chosen fallback. It is not the preparation path
for the selected Week-10 candidate above. Do not run it during the Week-10
Sunday sequence. If the Week-10 authority cannot become exact signed READY,
leave production on exact prestate `75a0a20...`.

If integration needs a merge, stop for separate merge authorization first.
Merge authorization does not authorize a push, deploy, or repoint. Preparation
below is local only and keeps every generated artifact non-shippable until an
independent review decision.

For the final fully certified science candidate, supply its exact clean
worktree as data to `--regenerate-from`. Do not edit a target SHA into the
validator, repoint, or probe scripts, and do not make a candidate depend on a
manifest that names the candidate itself. The release manifest is a separate
control artifact; the generator reads candidate `HEAD` and emits a new HOLD
manifest. This is the only preparation interface: no script SHA edits and no
commit/manifest self-reference. The checked-in manifest is only the unselected
template. The output paths must not exist; both generators are create-only.

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

The generator replaces every sentinel and recomputes the exact target
commit/tree/tree-record digest and
rollback-to-target name/status entries and digest. It requires the clean
candidate's two Dockerfiles to match the immutable policy, preserves the exact
rollback/ref and production literals, and always emits HOLD.

Independently inspect every generated field and every allowed-change entry.
Install it at `release/week9-release-manifest.json` through the normal reviewed
patch/commit workflow. Only after that review, change exactly
`release_state: "hold"` to `"ready"`, preserving the manifest's indent and
trailing newline, and review that final canonical diff. Configure exactly one
reviewed signer in `release/week9-release-allowed-signers`, then sign those exact
manifest bytes:

```bash
SN56_RELEASE_SIGNING_KEY=/absolute/path/to/reviewed-release-signing-key
ssh-keygen -Y sign \
  -f "$SN56_RELEASE_SIGNING_KEY" \
  -n sn56-week9-final-manifest \
  "$SN56_MANIFEST"
test -s "$SN56_MANIFEST.sig"
ssh-keygen -Y verify \
  -f release/week9-release-allowed-signers \
  -I sn56-week9-release \
  -n sn56-week9-final-manifest \
  -s "$SN56_MANIFEST.sig" < "$SN56_MANIFEST"
```

The private key is never checked in, copied to production, or named in the
manifest. Any candidate or manifest-byte change after signing invalidates the
signature and returns preparation to HOLD.

From that exact signed canonical READY manifest, derive a new HOLD readiness
receipt:

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
diff. Do not edit any binding field. Configure exactly one separately reviewed
key in `release/sn56-readiness-allowed-signers`, then sign the exact READY
receipt bytes under the distinct readiness principal and namespace:

```bash
SN56_READINESS_SIGNING_KEY=/absolute/path/to/reviewed-readiness-signing-key
ssh-keygen -Y sign \
  -f "$SN56_READINESS_SIGNING_KEY" \
  -n sn56-final-readiness \
  "$SN56_READINESS"
test -s "$SN56_READINESS.sig"
ssh-keygen -Y verify \
  -f release/sn56-readiness-allowed-signers \
  -I sn56-release-readiness \
  -n sn56-final-readiness \
  -s "$SN56_READINESS.sig" < "$SN56_READINESS"
```

The readiness signer must be independent of the manifest-signing authority.
Its signature authenticates the exact policy hash and all receipt bindings.
Commit the reviewed release-wiring files locally and require a clean worktree.
That local commit still authorizes no push, merge, deploy, or production action.

## Candidate evidence remains external to tooling

The final target must already carry its own independently reviewed science,
runtime, and any required real-container evidence before manifest generation.
The release scripts verify identity and operational safety; they do not infer a
winner, waive a failed gate, or turn CPU tests into GPU evidence.

## CPU-only verification

Run the focused release suite after every release-artifact or tooling change.
Record the fresh result rather than copying a historical count. These tests use
an explicit selected HOLD fixture and local mocks;
they do not need a GPU, provider, endpoint, service, or public network.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" \
  /Users/atulyashetty/Test/SN56-project/workspaces/repos/forge-toolkit/.venv/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_release_contract.py tests/test_release_probe.py
python3 -c 'p="scripts/sn56-release-contract.py"; compile(open(p, encoding="utf-8").read(), p, "exec")'
bash -n scripts/sn56-week6-repoint.sh \
  scripts/sn56-week6-rollback.sh \
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
  --manifest "$SN56_MANIFEST" \
  --docker-policy "$SN56_POLICY" \
  --readiness-receipt "$SN56_READINESS" \
  --contract-only
bash scripts/sn56-week6-repoint.sh \
  --manifest "$SN56_MANIFEST" \
  --docker-policy "$SN56_POLICY" \
  --readiness-receipt "$SN56_READINESS" \
  --dry-run
```

The full contract first verifies the exact manifest's detached signature, then
re-proves a clean exact local HEAD/tree, the immutable Docker
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

- At **2026-08-23 13:00:00 UTC (T-24)**, lock the target decision. If no exact
  candidate has complete independent support, keep production unchanged at
  `75a0a20...`; there is no implicit fallback release.
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
- Any loaded `com.sn56.monday-probe` LaunchAgent points to an external wrapper
  `/Users/atulyashetty/Test/SN56-project/scripts/sn56-monday-probe.sh`, which is
  not established by this isolated commit as matching the signed final
  manifest. This work does not install, repair, verify, or re-arm that
  LaunchAgent. Do not rely on it.
  Installing it needs separate deploy authorization; until then, run the
  tracked probe manually at every checkpoint.

## What the live probe proves

Live mode accepts only the exact signed canonical READY manifest, immutable
Docker policy, and independently bound READY receipt. It re-proves the exact
target commit/tree/ref, repository, clean reviewed worktree, rollback, changed
surface, and Docker bytes before endpoint I/O. It uses the tracked
`scripts/sn56-upstream-baseline.env` pin
`f7caab6cb2786f4210036b0675123d6ba9633e8f`; upstream drift is a failure, and
changing that baseline requires a new validator diff review.

The off-host check separately performs an exact GET of
`http://65.108.77.230:7999/training_repo/image`. Success is HTTP 422 with the
observed Fiber v2.7 missing-header multiset: `validator-hotkey` exactly twice,
and `signature`, `miner-hotkey`, and `nonce` exactly once each. Every row must
have type `missing` and a two-part `header` location. The obsolete four-row
variant, an enum/path 422, any other partial/extra header error, malformed body,
HTTP 400, or a different route fails.

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
commit containing its reviewed signed canonical READY manifest and READY
readiness receipt. The selected target must already be integrated and
certified; there is no hidden Sunday merge or lane-switch step.

```bash
set -euo pipefail
SN56_MANIFEST=release/week10-release-manifest.json
SN56_POLICY=release/week10-candidate-docker-policy.json
SN56_READINESS=release/week10-release-readiness.json
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["release_state"])' "$SN56_MANIFEST")" = ready
test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["readiness_state"])' "$SN56_READINESS")" = ready
test -s "$SN56_MANIFEST.sig"
test -s "$SN56_READINESS.sig"
test "$(shasum -a 256 "$SN56_POLICY" | awk '{print $1}')" = \
  4aa745e5cfb07cfa74d33e85d6f9cefd98de7de21149e5699e91be7fa5c01805
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
   bash scripts/sn56-week6-repoint.sh \
     --manifest "$SN56_MANIFEST" --docker-policy "$SN56_POLICY" \
     --readiness-receipt "$SN56_READINESS"
   ```

5. Immediately run the tracked live probe. Require `RESULT: GREEN (live)`, zero
   failures, and review every warning; `entry.balance`, `endpoint.reachable`,
   and `endpoint.pin` must each PASS:

   ```bash
   bash scripts/sn56-monday-probe.sh \
     --manifest "$SN56_MANIFEST" --docker-policy "$SN56_POLICY" \
     --readiness-receipt "$SN56_READINESS"
   ```

   Unless a separately authorized LaunchAgent installation has been verified,
   repeat this exact manual command at 06:00 and 07:15 CDT (11:00 and 12:15
   UTC) on 2026-08-24. Do not substitute an unverified external wrapper.

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
SN56_ROLLBACK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sn56-week10-rollback.XXXXXX")"
python3 scripts/sn56-release-contract.py \
  --manifest "$SN56_MANIFEST" \
  --docker-policy "$SN56_POLICY" \
  --rollback-contract-only \
  --readiness-receipt "$SN56_READINESS" \
  --receipt "$SN56_ROLLBACK_DIR/rollback-contract.json"
```

That command proves identity only and does not touch the host. The operator
rollback command runs the same offline contract internally, remains available
after the hard abort or when the public ref is unavailable/worktree is dirty,
and then requires the host to serve either the exact released target or the
exact rollback no-op state. An unknown served pin fails closed.

```bash
# Preferred one-line form:
bash scripts/sn56-week6-rollback.sh \
  --manifest "$SN56_MANIFEST" --docker-policy "$SN56_POLICY" \
  --readiness-receipt "$SN56_READINESS"

# Equivalent explicit form (use one form, never both):
# bash scripts/sn56-week6-repoint.sh \
#   --manifest "$SN56_MANIFEST" --docker-policy "$SN56_POLICY" \
#   --readiness-receipt "$SN56_READINESS" --rollback
```

The wrapper is the preferred one-line form and asserts exact rollback
`75a0a20c2deda82cfa727e082e60a95bea5befb3` before delegating. Rollback is a
separate explicit operator decision, not pre-authorized here. The checked-in
Week-10 HOLD cannot run live rollback; production already rests on its exact
rollback pin.
