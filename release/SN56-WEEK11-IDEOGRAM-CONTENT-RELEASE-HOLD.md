# SN56 Week-11 Ideogram CONTENT release — HOLD

State: `HOLD / UNSELECTED`. The exact candidate image policy and offline canary
are now PASS, but no signature, readiness authorization, push, probe, or
deployment is present in this wiring change. The source identity is recorded
below and in the manifest's immutable `candidate_source_evidence`, and is
exercised by preparation/contract mocks. The HOLD readiness receipt binds those
exact manifest and candidate-policy bytes through `manifest_sha256` and
`docker_policy_sha256`. Passing the image gate does not select the checked-in
target or authorize release.

## Sealed source candidate (not release authorization)

- Commit `fe9749c027df511b7566b474e6f8524f86b01f83`
- Tree `c7e79fb326e3bc3ef7b573d2d0144e82e0e24a25`
- Tree-records SHA-256
  `c5362ef488af54c7729cba91279b0287b382eee05c5739e802bd894ef72ee582`
- Ref `refs/heads/week11-ideogram-content-product`
- Exact ordered changed surface from `59e...`: `M forge/config.py`,
  `A forge/ideogram_content_policy.py`, `M forge/tasks/aitoolkit.py`, and
  `A tests/test_ideogram_content_policy.py`; canonical name-status SHA-256
  `d15197e9bac0369efc71c35c3f7ded8e0e1bf47397f594cd5a2d4b877f397e05`.

## Candidate image policy (PASS, not release authorization)

The canonical schema-2 policy is
`release/week11-candidate-docker-policy.json`, SHA-256
`a11ddb1731856ff643b1b1e72439584b4cd753117400d31449b70a0c78b375f2`.
The Week-11 adapter accepts only those exact canonical bytes and rejects any
field addition, substitution, serialization drift, or evidence drift. It binds:

- Image ID
  `sha256:987fa5c8964d04bb8aa2ad0fbc485aecafe63b1f85dae1da65cc960e917c8cd8`.
- Offline canary result `PASS`, receipt SHA-256
  `4020d7a9ef90992b97c637b794d86e143bddaf61a95a47fc16ea55aace97e6de`.
- Complete image-history SHA-256
  `3044dca1bafa2c68e5dc99dc2b526b75af9f5a1a19a6f8e8b38c2f996d2a9209`.
- Runtime-inventory SHA-256
  `9c4c15130508c547c67d891f559ca1a513cd62bd5a4b695eb25ceafccd0b850b`.
- Product-projection SHA-256
  `ce226348ca641932de4a0f27d59a69ad8a124825068a4f1e9e9d134166eba4cd`.
- The two unchanged Dockerfile hashes and three unchanged pinned base-image
  bindings from current production.

No Week-10 build, parity, or context claim is inherited: those receipts describe
a different image and are not causal evidence for this candidate.

## Bound recovery identities

- Immediate rollback (the only identity wired into the manifest and one-line
  rollback wrapper): commit `59e0698c952edaf1bf34a117ecad41bce87517cf`,
  tree `613a1cc2d750731df007cc9b2b49e461d0ae368f`, ref
  `refs/heads/week10-trainer-candidate`.
- Secondary break-glass only: `75a0a20c2deda82cfa727e082e60a95bea5befb3`.
  It is deliberately absent from the primary rollback contract and wrapper. It
  is not the first rollback from a Week-11 release.

## Forward cutoff

Forward repoint is barred at `2026-08-26 16:30:00 UTC` (epoch `1787761800`),
90 minutes before the Wednesday `18:00 UTC` science deadline. The margin is
reserved for verification and restoration to exact production. Emergency
rollback remains available after the cutoff.

## Create-only final-manifest preparation

The reviewed target ref is
`refs/heads/week11-ideogram-content-product`. The image gate is complete; create
a new selected HOLD manifest without overwriting this template:

```bash
python3 scripts/sn56-week11-release-contract.py \
  --regenerate-from release/week11-release-manifest.json \
  --candidate-worktree /ABSOLUTE/CLEAN/WEEK11/CANDIDATE/WORKTREE \
  --docker-policy release/week11-candidate-docker-policy.json \
  --output /ABSOLUTE/NEW/week11-final-manifest.hold.json
```

Preparation derives the candidate HEAD commit, tree, tree-record digest, and
`59e...`-to-target name-status entries/digest from Git; verifies the clean HEAD
is the exact target ref; and writes with create-only semantics. There is no
target-SHA input. The result remains non-shippable HOLD until independent review
changes the exact bytes to READY, provisions the two distinct Week-11 signer
authorities, and produces detached signatures plus an exact READY readiness
receipt through the existing reviewed mechanics.

The exact Week-11 policy is the default for preparation, contract, dry-run,
repoint, and rollback. The checked-in target and readiness state nevertheless
remain zero/unselected and HOLD until the independent signing gates complete.
