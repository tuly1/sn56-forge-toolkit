# SN56 Week-11 Flux Kohya release — HOLD

State: `HOLD / UNSELECTED`. The causal Flux grid is `PROMOTE`, the source
candidate is sealed, and CPU release wiring is fail-closed. No H100 canary PASS,
signature, readiness authorization, push, repoint, deployment, or production
mutation is claimed by these checked-in bytes.

## Sealed source candidate

- Target commit `3f7737c4045c3877e3c9f100a0341fe2566b1db6`
- Target tree `42001148b7b8342d9837dcb7e2b532b23892aeee`
- Target tree-records SHA-256
  `cdb54ba614daa58d075d9049ee48a00ebed487f12fcd05e2eda5da929d987f44`
- Target ref `refs/heads/codex/week11-flux-release`
- Immediate rollback/current production
  `fe9749c027df511b7566b474e6f8524f86b01f83`
- Exact ordered `fe...` to target surface: two modified Flux implementation
  files plus two frozen TOML goldens and one focused test file; canonical
  name-status SHA-256
  `416c825bb8ccb840b2039a3052bffe946b4edc37fed22a5a1bd14824ac07e4f7`.

The four Ideogram CONTENT files remain byte-identical to live `fe...`:

- `forge/config.py`: `1f1ec28f...`
- `forge/ideogram_content_policy.py`: `1b5e9d29...`
- `forge/tasks/aitoolkit.py`: `a88718c1...`
- `tests/test_ideogram_content_policy.py`: `cd6d158b...`

## Causal and execution identities

- Flux decision SHA-256
  `3c192338b0e31ec09d42160a0353588f0c883975990cd7633e351b5494693366`
- Corrective execution-identity receipt SHA-256
  `6ecd8a1c9a746c4c5e760fad7b440c2299aa58a443d67c2fa95717126eccc39b`
- Actual scientific cells used immutable Kohya base digest
  `sha256:d34dd5750e1018455e111f63c03bb2a4e16204607e00ba5af870dd7c71beb84e`.

## Candidate image gate

`release/week11-candidate-docker-policy.json` is intentionally
`hold-pending-exact-h100-canary`. It binds the candidate, rollback chain,
unchanged Dockerfiles/base images, science verdict, and corrected execution
identity, but contains no invented image or canary receipt.

The remaining gate is one offline H100 controlled-stop canary through the
candidate image's default production entrypoint, exactly
`["dumb-init","--","python3","-m","forge.cli"]`. It must prove the production
entrypoint emits the frozen 94-step seed-1 Kohya TOML, starts
`/app/sd-scripts/flux_train_network.py`, reaches one finite optimizer step on
the H100, never enters ai-toolkit fallback, stops the owned container cleanly,
and syncs the exact image/runtime/config/process/GPU receipts off-host.

Until those facts replace the pending policy and its fixed digest, the Week-11
contract rejects candidate preparation and forward dry-run. This is deliberate.

## Recovery identities

- Immediate rollback wired into the manifest and one-line wrapper: live
  `fe9749c027df511b7566b474e6f8524f86b01f83`, tree
  `c7e79fb326e3bc3ef7b573d2d0144e82e0e24a25`, ref
  `refs/heads/week11-ideogram-content-product`.
- Secondary break-glass only:
  `59e0698c952edaf1bf34a117ecad41bce87517cf`. It is documented and tested as
  rejected from the primary rollback field and wrapper.

## Post-canary release sequence

After the exact H100 receipt is sealed, update only the candidate image policy,
its fixed SHA-256 in `scripts/sn56-week11-release-contract.py`, and the HOLD
readiness binding; then rerun the focused contract suite and full repository
tests. The create-only target selection command is:

```bash
python3 scripts/sn56-week11-release-contract.py \
  --regenerate-from release/week11-release-manifest.json \
  --candidate-worktree /Users/atulyashetty/Test/SN56-project/workspaces/worktrees/week11-flux-release-codex \
  --docker-policy release/week11-candidate-docker-policy.json \
  --output /ABSOLUTE/NEW/week11-flux-final-manifest.hold.json
```

Then set the exact generated manifest to READY, sign it in the
`sn56-week11-final-manifest` namespace, produce/sign a distinct READY readiness
receipt in `sn56-week11-final-readiness`, push the exact target ref, and run:

```bash
bash scripts/sn56-week11-repoint.sh \
  --manifest /ABSOLUTE/week11-flux-final-manifest.ready.json \
  --docker-policy release/week11-candidate-docker-policy.json \
  --readiness-receipt /ABSOLUTE/week11-flux-final-readiness.ready.json \
  --dry-run
```

Only a clean dry-run plus fresh live source/bytecode/service/route/metagraph
preflight may precede the same command without `--dry-run`. The immediate
rollback command remains:

```bash
bash scripts/sn56-week11-rollback.sh \
  --manifest /ABSOLUTE/week11-flux-final-manifest.ready.json \
  --docker-policy release/week11-candidate-docker-policy.json \
  --readiness-receipt /ABSOLUTE/week11-flux-final-readiness.ready.json
```

Forward repoint is barred at `2026-08-26 16:30:00 UTC` (epoch `1787761800`).
Rollback remains available after that cutoff.
