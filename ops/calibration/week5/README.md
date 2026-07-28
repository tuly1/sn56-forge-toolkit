# Week 5 Krea evidence workspace

This directory contains pre-GPU, experiment-only controls for the Week 5 Krea
program. Production code under `forge/` is intentionally unchanged. Neither
production Dockerfile copies `ops/calibration/`, so these files cannot alter a
tournament image unless a later reviewed release explicitly ports a winning
policy.

The order is fail-closed:

1. Preserve and hash public field artifacts. The first seal passed with 12
   repositories, 24 revisions, 80 unique LFS objects, and manifest-set SHA-256
   `62a0476de06ea729591899beecc4d365e96af4bb0cc79a6714bd783f22a4ea84`.
2. Immutable K2-K4 public-arm provenance manifests are generated and
   byte-reproduced, but remain explicitly `unreviewed`; K5 has a hash-bound
   internal evidence record.
3. Name every accountable human DRI and obtain an externally anchored reviewer
   approval for the sealed execution plan.
4. Curate and seal D1/D2 matched-but-disjoint train/evaluation splits under
   `KREA-FIXTURE-CURATION-CONTRACT.md`. A C1-C4 pre-finalist commitment was
   published externally; it remains sealed and requires named-human acceptance
   before it can satisfy the independent-review gate. The later public-shape
   correction is bound by `krea-c1c4-shape-contract-amendment.json`: C1-C4 use
   their exact published concept classes and 20/6, 45/6, 30/8, and 12/5
   train/evaluation counts. It is an amendment to the plan contract, never a
   reseal of the unchanged fixture commitment.
5. Measure conservative actual GPU/runtime upper bounds, validate the resulting
   plan in one held-out end-to-end timing run, and bind the profile by SHA-256.
6. Resolve a budget-fill plan, actual candidate fractions, and image exposures.
7. Run D1/D2 discovery exactly as frozen in `krea-discovery-plan.json`.
8. Freeze finalists and a deterministic checkpoint rule before the independent
   reviewer reveals C1-C4.
9. Update `evidence-ledger.json` only from linked evidence. A missing field is a
   red gate, not permission to use a guessed default.

`krea-discovery-plan.json` is currently `draft_blocked_pre_gpu`, not an
execution freeze. Final Round-1 evidence changed the screen from five to six
arms: K2 is the actual rank-2 LR-8.6e-5 family, K3 is the rank-3 MAE/guidance-3
family, K4 is the rank-5 rank-64 Automagic family, and K5 retains the internal
LR-2e-4 challenger. This correction must be independently reviewed before GPU
execution.

`evidence_ledger.py` enforces exact structural consistency and refuses null
readiness claims. Readiness requires explicit result-PASS attestations and
their digests; an independent human still authenticates the signer and the
scientific meaning of those records. Its narrower `--require-named-dris` check
rejects role labels or unnamed human owners; it is not a GPU authorization gate.
`krea_budget.py` never falls back to guessed timing: it requires bound
geometry, conservative component bounds, a predeclared margin policy, and a
held-out end-to-end validation record.

The CPU tree is currently covered by 436 passing tests plus Black, pyflakes,
compilation, and whitespace checks. A literal Hetzner systemd smoke verifies
the exact-score containment primitive. Those results do not replace the
required schema-2 H100 host capture and literal first-GPU bootstrap smoke.

Offline exact-evaluator output is calibration evidence only. It must never be
written to the production `forge_holdout_scores.json` proxy contract.
