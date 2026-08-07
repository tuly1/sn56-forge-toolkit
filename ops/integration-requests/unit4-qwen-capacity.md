# Integration request — Unit 4 (qwen-image capacity) → Unit 2

**Date:** 2026-08-07 · **Target file:** `forge/templates/base_diffusion_qwen_image.yaml` (owned by Unit 2)
**Type:** COMMENT-ONLY. **No value changes requested.** The template parses to an identical
dictionary before and after.

---

## Verdict

**HOLD `linear: 32` / `linear_alpha: 32` / `lr: 0.0001`.** The "4× capacity deficit" flagged by
the blind audit does not survive the artifacts. Full reasoning, evidence and budget arithmetic:
`SN56-project/SN56-WEEK6-QWEN-CAPACITY-ANALYSIS-2026-08-07.md`.

Rank 128 **is affordable** — at our own measured rate it fits every real qwen shape with 40–70% of
the optimizer window unused — so this is not a clock veto. It is rejected because the evidence
says it buys nothing:

- Across **7 qwen tasks in 2 tournaments**, rank 128 took **5 of 7** first places; rank > 128 took
  2, both by one operator on one day against opponents deadline-killed at 46–54% of plan.
- The only task where a high-rank and a low-rank entrant **both completed** (Jul-27 `5722b124`)
  went to **rank 128 at 300 steps over rank 152 at 1197 steps, by 8.4%**.
- One of the two high-rank wins (`ff643470`, +0.82%) is **inside the 2.145% same-recipe noise
  floor** measured on `7421f056`, where two entrants ran identical configs to the same depth.
- With `alpha == rank`, ‖ΔW‖ ∝ √rank·lr·steps — which is *why* the field's law is lr ∝ 1/√rank.
  The coupled change is therefore **magnitude-neutral by construction**; the uncoupled one is a
  2× learning-rate change in disguise.
- On `E = lr·√rank·steps`, field winners span 0.339–1.191 and **our plans give 0.473 / 0.541 /
  0.579 — inside the band**, 1.40× above the shallowest observed qwen winner.

## Requested change (comments only)

Insert above the existing `network:` block in `forge/templates/base_diffusion_qwen_image.yaml`.
Verbatim, ready to apply:

```yaml
    # RANK 32 IS HELD DELIBERATELY.  The Aug-3 qwen field ran 128/141/149 and a
    # blind audit read that as a 4x capacity deficit.  Investigated 2026-08-07
    # against the qwen record of BOTH retrievable tournaments (7 tasks, 13
    # entrants, Jul-27 recovered live from the public API + HF because its
    # DATASETS expired but its ARTIFACTS did not).  Four reasons to hold:
    #  1. RANK DOES NOT SEPARATE WINNERS.  Rank 128 took 5 of the 7 first
    #     places; rank >128 took 2, both `5FBmn1ax` on Aug-3, both against
    #     opponents DEADLINE-KILLED at 600/1300 and 700/1300.  The only task
    #     where a high-rank and a low-rank entrant BOTH completed their plan is
    #     Jul-27 `5722b124`: rank 128 at 300 steps beat rank 152 at 1197 by
    #     8.4%.  Rank here is an inherited ai-toolkit per-architecture default
    #     (published verbatim by 4 different hotkeys), not a tuned edge --
    #     `5FBmn1ax` himself runs rank 32 on krea2, ideogram4 AND z-image.
    #  2. ONE OF THOSE TWO WINS IS NOISE.  On `7421f056` two entrants published
    #     configs identical after removing hotkey-bearing paths and both were
    #     killed at step 850; their losses differ by 2.145%.  The rank-149 win
    #     on `ff643470` was by 0.82%.
    #  3. THE COUPLED CHANGE IS MAGNITUDE-NEUTRAL.  alpha == rank here and in
    #     every field config, so the LoRA scale is 1.0 and ||dW|| ~
    #     sqrt(rank)*lr*steps -- which is exactly what the field's recovered
    #     lr = 1.0877e-3/sqrt(rank) law exists to hold invariant.  Rank 128 with
    #     the coupled lr buys only extra DIRECTIONS, and 75% of the score is a
    #     prompt-free reconstruction MSE at denoise 0.93 (constants.py:25-32,
    #     diffusion.py:203-209/311-314, tasks.py:280-298) -- a first-moment,
    #     prior-location objective that cannot use them.  Rank 32 already gives
    #     294,912,000 trainable params against N=28-50, i.e. 6-10M per image.
    #  4. WE ARE NOT BEHIND.  On E = lr*sqrt(rank)*steps the field's qwen
    #     winners span 0.339..1.191; our three planned shapes give
    #     0.473/0.541/0.579 -- inside the band, 1.40x above the shallowest
    #     winner.  The nominal 4x rank gap is at most 2x in magnitude.
    # COST, FOR THE RECORD: rank 128 would FIT (our own measured 2.903 s/step
    # at rank 32 -- 2978.1 s for 1026 completed steps,
    # evidence/hyperstack-qwen-forge-operational-20260724 -- leaves room for
    # 1425-1748 steps against plans of 836-1023).  Its real price is +6.1% step
    # time, a 4x artifact (0.59 -> 2.36 GB, needing ~13 MB/s of the 180 s export
    # reserve vs 3.3 MB/s today) and ~9 GB more periodic-save I/O on the one
    # type with ~91 s of modelled slack.  Affordable, but unbought.
    # See SN56-WEEK6-QWEN-CAPACITY-ANALYSIS-2026-08-07.md and
    # tests/test_qwen_capacity.py, which FAILS if rank is raised without
    # re-pricing SEC_PER_IT["qwen-image"].
```

## What Unit 2 must NOT do without re-reading the analysis

`tests/test_qwen_capacity.py` (added in this worktree, 13 assertions, all passing) fails if:

- `linear`, `linear_alpha` or `lr` change in this template;
- `linear_alpha` is decoupled from `linear` (that voids the magnitude law the decision rests on);
- `do_cfg` is added to the qwen template (that voids every budget number in §5 of the analysis);
- rank is raised **without** raising `SEC_PER_IT["qwen-image"]` to at least
  `4.70·(3 + rank/512)/(3 + 32/512)` — 4.988 s/step at rank 128.

Both mutations were exercised: raising rank to 128 fails 2 tests with the required constant named
in the message; decoupling alpha fails 2 tests.

## Not requested, but noted for whoever owns the next qwen change

The evidence for `loss_type: mae` (set by `5FBmn1ax` on **both** Aug-3 qwen wins and all three of
his krea2 wins, by no rank-128-template entrant) is **better than the evidence for rank**, and it
is free on the clock. Same for `min_denoising_steps: 70`. Neither is proposed for Monday — both
are unvalidated on our pipeline — but if a qwen change is going to be spent, it should not be
spent on rank. See §8 of the analysis.

## Separate observation referred to the checkpoint-selection owner

Four Aug-3 entrants published a ladder with **no `checkpoints/last.safetensors`**. Since `.`
(0x2E) sorts before `_` (0x5F), a lexicographic "first match" glob would select
`last_000000050.safetensors` — the *shallowest* rung — where `analysis.json` assumes the deepest
(`shipped_src: "terminal-periodic(INFERRED)"`). This does not change the recommendation above and
I did not build on it, but it needs the validator source to settle. §7 of the analysis gives the
one datum that bears on it.
