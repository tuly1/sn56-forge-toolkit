# Krea Day-0 throughput audit

**Status:** profile not yet eligible for budget planning  
**Production impact:** none  
**Release base:** `c654c4b24376f7aa9e12dcb82f5e73dcddee3bdb`

## What the surviving evidence establishes

Two same-envelope Week-4 H100 runs each requested 367 optimizer updates with
four periodic saves plus an exact final:

| Run | Training pairs | Toolkit wall time | Updates | Aggregate seconds/update |
|---|---:|---:|---:|---:|
| LR 1e-4, seed 42565431 | 32 | 280.1 s | 367 | 0.7632 |
| LR 2e-4, seed 42565431 | 32 | 285.0 s | 367 | 0.7766 |

The Jul-27 tournament run used a different runtime/hardware envelope and took
roughly 835 seconds end to end for an inferred 267-update plan. The public
record does not isolate update, save, startup, and finalization costs. Dividing
that aggregate by 267 would yield roughly 3.1 seconds/update and would be an
invalid throughput estimate. The approximately fourfold cross-environment gap
also proves that the H100 aggregate cannot be copied into production.

The prior condition records expose another confound: after the 280–285 second
toolkit phase, the in-task held-out proxy spent roughly 477–479 seconds scoring
five candidates. That proxy is shadow-only and its 900-second reserve cannot be
treated as free training time without a separate production decision.

## Fail-closed conclusion

No current record contains the conservative component bounds required by the Week-5 budget
contract. Therefore `throughput_profile_sha256` remains null and no K1–K5
budget-fill depth is frozen. This is not a reason to reuse the max-400 policy;
it is the first required GPU measurement.

## First GPU measurement protocol

Use the exact frozen Docker subject, Krea assets, and one D1-shaped dataset.
Instrument monotonic timestamps around:

1. container/process start to dataset-ready;
2. toolkit launch to first completed optimizer update;
3. each steady-state optimizer update after warm-up;
4. every checkpoint save, including fsync completion;
5. exact-final export, validation, copy, and directory fsync;
6. upload-ready packaging and a local object-store upload simulation.

Collect at least three independent startup observations, 100 steady-state
updates, and eight save observations. Predeclare a margin policy, derive
conservative upper bounds from observed maxima plus that margin, and bind the
profile to dataset/base/code/runtime/GPU, micro-batch, accumulation, replica,
resolution-policy, and precision-policy hashes. The profile is not usable until
the resulting plan completes a separate held-out end-to-end timing validation.
It is rejected if any sample class or binding is absent, if its source or margin
manifest hash does not match, if candidate-save I/O exceeds 10% of the
post-reserve window, or if optimizer time plus save I/O consumes less than 90%
of that window without an explicit evidence-backed early-optimum waiver.

Discovery exact scoring is offline after training and therefore has a zero
in-envelope reserve. A future live selector may consume a positive reserve only
when its scorer identity and measured upper bound are bound into a new profile;
the caller cannot silently reserve time for an unvalidated proxy.

The first measured profile is a host-calibration input for D1/D2, not a universal
validator constant. A later production policy must either measure the live
runtime before committing its scheduler or be certified on the validator's
actual GPU envelope with an explicit portability bound.

## Source evidence

- Week-4 condition records:
  `week4-gpu-evidence-2026-07-22/krea-short-s42565431-r1/campaign/conditions/`
- Jul-27 timeline and claim limits:
  `SN56-WEEK4-COMPETITIVE-POSTMORTEM-AND-WEEK5-WIN-PLAN-2026-07-27.md`
- The deployed guessed constants and max-400 Krea cap:
  `forge/recipe.py` at release base `c654c4b`
