# Week-7 HKE CPU prelaunch gates

Status: **SCHEMA-3 CPU REMEDIATION ONLY / GPU NOT AUTHORIZED**
Production status: **untouched**
Authority: **no merge, rental, launch, admission, release, or deployment**

> **Legacy artifacts are invalid.** Every earlier HKE candidate, review,
> admission, experiment plan, timing record, training receipt, attachment
> receipt, score receipt, and decision receipt uses a superseded schema. None
> may be rehashed, wrapped, migrated, or reused. Regenerate all schema-3
> material from the exact reviewed source SHA in fresh create-only roots.

This record defines CPU prelaunch contracts. It is not a quality claim, GPU
request, fixture admission, FutureBound root-cause finding, or deployment
authorization.

## Evidence and dependency boundary

The FutureBound no-submission observation was a pre-retry snapshot, not the
final result. The safe post-retry API snapshot records a scored submission at
`0.0658568064`, rank `10 of 12`. The preserved evidence boundary is:

- correction record:
  `/Users/atulyashetty/Test/SN56-project/evidence/week7-post-retry-20260811/PR16-CORRECTION-RECORD-2026-08-11.md`;
- safe API snapshot:
  `/Users/atulyashetty/Test/SN56-project/evidence/week7-post-retry-20260811/public-api-safe/snapshots/20260811T142602.496953Z.json`;
- snapshot SHA-256:
  `a77f3c5da54b1b81c891a3a27ff62a7aa1464a03fbcd2ac20898989e29d0d9f5`.

The correction record's earlier PASS claim for PR #16 head
`ed99cee0d6c9cab1b68f43c742e2735fee99a0d0` was superseded by a later
independent HOLD: prohibited-path aliases, explicit `request_query: null`, and
invalid UTF-8 query encodings remained accepted. PR #16 must receive a new
exact-SHA independent PASS after those defects are fixed. PR #17 must then be
rebased onto that exact cleared base, pushed, read back, and independently
audited before any fresh candidate is generated.

## Schema-3 fixture gate

Only the following fresh first-party packs are in scope:

| Family | Pack | Role | Training rows | Held-out evaluation rows |
|---|---|---|---:|---:|
| social / FutureBound | D1 | discovery | 10 | 8 |
| social / FutureBound | D2 | discovery replication | 10 | 8 |
| social / FutureBound | C1 | sealed confirmation | 10 | 8 |
| social / FutureBound | C2 | sealed borderline reserve | 10 | 8 |
| product | D1 | discovery guardrail preparation | 10 | 8 |
| product | C1 | sealed confirmation guardrail | 10 | 8 |
| logo/UI | D1 | discovery guardrail preparation | 10 | 8 |
| logo/UI | C1 | sealed confirmation guardrail | 10 | 8 |

Every training/evaluation split is disjoint, and every pack is independent.
Discovery and confirmation require distinct private phase-key files, each held
at mode `0600`. Within each phase, row seeds are derived from that one phase
key using domain-separated inputs that bind phase, fixture, pack, split role,
and ordinal; there are no separate training/evaluation key files. Effective
row seeds must be unequal across both axes and never reused across phases,
packs, split roles, or rows. Discovery and confirmation outputs must be
separate create-only trees.

Admission remains closed until the fresh candidate has exact ownership/use
records, deterministic byte replay, complete source and dependency bindings,
cross-family/cross-phase/cross-pack/train-evaluation deduplication, sealed
confirmation commitments, named-human review of every image-caption row, and
explicit owner ratification. Public tournament archives, opponent material,
hidden/test data, and validator evaluation rows are not fixture inputs.

## Schema-3 experiment gate

The primary D1 screen is an exact 1,200-step loss (MAE versus MSE) ×
multires-noise factorial:

| Cell | Loss | Multires noise | Terminal depth | Runtime |
|---|---|---|---:|---|
| R0 | exact retry recipe | exact retry setting | 1,166 | incumbent |
| A | MAE | off | 1,200 | owned |
| B | MSE | off | 1,200 | owned |
| C | MAE | 6 iterations, 0.3 discount | 1,200 | owned |
| D | MSE | 6 iterations, 0.3 discount | 1,200 | owned |

The factorial depth is fixed at 1,200; timing observations may establish
feasibility but cannot redefine it. R0 must terminate at exactly 1,166. Score
natural periodic and terminal checkpoints without inventing a 1,166 checkpoint
for A–D.

A no-multires incumbent-versus-owned bridge must clear its predeclared
equivalence tolerance before A–D. In the prelaunch plan, the owned runtime is
allowed only for the owned side of that bridge and A–D. R0 stays on the
incumbent runtime. No runtime swap, non-Krea change, or production mutation is
authorized.

## Staged freeze and confirmation gate

The plan must enforce this order:

1. Freeze fixture identities, evaluator and preprocessing, inference seeds,
   prompted/blank weighting, runtime identities, and decision rules.
2. Clear the runtime bridge; then run and exact-score social D1 R0 and A–D only
   against D1 held-out evaluation rows.
3. Freeze one D1 recipe and checkpoint target; compare it with the incumbent on
   social D2 under both predeclared finalist seeds.
4. After D2, freeze the candidate digest, recipe, checkpoint target, evaluator,
   inference settings, seed policy, and decision rule before any C1 identity
   or byte is revealed.
5. Run social C1 once without confirmation-driven reselection. Product C1 and
   logo/UI C1 are guardrails, not co-equal discovery targets.
6. Keep social C2 sealed unless a separate, predeclared borderline-C1 trigger
   fires. C2 cannot tune a threshold, replace routine C1 confirmation, or
   rescue a failed result.

Within those stages, listed cell sequences are an operator procedure only.
The schema binds the declared list and individual cell receipts but does not
machine-prove actual chronology, predecessor completion, non-overlap, or
counterbalancing. Do not report counterbalancing as established without an
independent chronological run record.

Training rows may never enter evaluation inventories. Confirmation results may
not select a recipe, checkpoint, threshold, or router.

## Evidence and authority gate

Timing profiles and execution, training, attachment, score, freeze, and
decision records must be complete, self-hashed, and bound to the exact plan,
config, runtime, accelerator observation, artifact, fixture split, and
evaluator. Their evidence class is `operator_attested`. These records can prove
internal consistency and declared provenance; they are not independent proof
that a person reviewed a row or that a GPU, attachment, or scoring event
occurred.

Later-stage validation must reproduce the complete D1, optional-E, D2,
confirmation, and conditional-C2 source chain from embedded source plans and
evidence. A digest-valid outer envelope is insufficient. The executed config
must be the config rebuilt from the frozen source chain, revealed confirmation
rows must reproduce their admission-bound commitments and physical
inventories, and the terminal D1 freeze must retain the predeclared 2x2
factorial effects. Any source/evidence transplant or post-hoc config mutation
is a hard failure.

CPU contract generation does not authorize GPU rental or execution. Even a
future all-green experiment would produce evidence for a reviewed candidate,
not merge or ship authority. GPU use, fixture admission, checkpoint promotion,
semantic routing, release, deployment, endpoint repointing, and production
changes each remain closed pending separate explicit owner authorization and
their own review gates.
