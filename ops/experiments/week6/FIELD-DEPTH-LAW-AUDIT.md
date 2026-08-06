# FIELD DEPTH LAW AUDIT — every Aug-3 tournament artifact, all five model types

**Branch:** `claude/week6-real-fixture-experiment` (cut from production pin `084ea914`)
**Date:** 2026-08-06 · **Tournament:** `tourn_c54bb970b5d0aa91_20260803` (14 tasks, 5 rounds)
**Decides:** how deep forge trains every model type on Monday 2026-08-10 13:00 UTC.
**Compute:** CPU only. No GPU spend. No production host touched.

---

## 0. TL;DR — headline gap per type

`now/win` = what `forge/recipe.py` would have shipped ÷ what the task's rank-1 miner
actually shipped, averaged over that type's tasks.

| type | tasks | winners shipped (steps) | forge would ship | **now/win** | verdict |
|---|---|---|---|---|---|
| **krea2** | 4 | 2012, 1000, 2012, 2012 | 1172, 824, 1172, 1172 | **0.64×** | **UNDER-TRAINED. Biggest single gap. Change.** |
| **ideogram4** | 3 | 174, 341, 1300 | 107, 194, 181 | **0.44×** | **`now/win` IS THE WRONG METRIC HERE — see §6.2 REVISION. Change, but not toward these step counts.** |
| **z-image** | 2 | 1317, 1188 | 860, 860 | **0.69×** | **UNDER-TRAINED, purely by a bad clock. Change.** |
| **flux** | 2 | 870, 750 | 726, 726 | **0.90×** | Near-correct. Small clock fix only. |
| **qwen-image** | 3 | 949, 850, 1095 | 1027, 836, 1027 | **1.00×** | **Already calibrated. Do not touch the depth.** |

The single most important line: **for krea2, ideogram4 and z-image the binding
constraint is not a considered depth policy — it is `SEC_PER_IT`.** The field
completed runs at rates that put our per-step constants 1.4×–1.9× too
conservative, so our clock cap silently truncates the size law before the size
law is even consulted (Table 5, `binds` column).

**The Jul-16 premise "deep training never helped" is falsified by direct field
evidence** — see §3.1 (two independent miners published their own per-checkpoint
reconstruction curves, both monotonically improving out to the deepest
checkpoint they trained) and §3.2 (rank correlation on the only 14-way field).

---

## 1. Evidence base — what was harvested and how

All raw evidence is at
`/Users/atulyashetty/Test/SN56-project/evidence/week6-field-depth-audit-20260806/`
(scripts + cached payloads + checksums; re-running is idempotent).

| step | source | result |
|---|---|---|
| 1 | `https://api.gradients.io/auditing/tasks/<task_id>` × 14 | `audits/*.json`, `audit-index.json` |
| 2 | `https://huggingface.co/api/models?search=tournament-<tourn>-<task_id>` | `repo-index.json` — **40 repos**; the HF org contains *exactly* the audit's scored submissions, no extras |
| 3 | `/api/models/<repo>/tree/main?recursive=true` + range reads | `raw/<repo-slug>.json` — **305 safetensors headers**, 0 weight bytes downloaded |
| 4 | join | `artifacts.json` (40 rows), `checkpoints.csv` (305 rows), `analysis.json` |

Scripts: `fetch_audits.py` → `enumerate_repos.py` → `harvest_artifacts.py` →
`extract.py` → `analyse.py` → `recommend.py`. Full console output preserved in
`FULL-OUTPUT.txt`; sha256 of every derived table in `CHECKSUMS.txt`.

### 1.1 The field is much thinner than the task brief assumed — state this first

Only **Round 1 is an open field**. Rounds 2–5 are **head-to-head bracket
matches with exactly 2 submissions each**.

```
R1  41025fb5 krea2   →  15 scored submissions (14 with a published repo)
R2..R5 (13 tasks)    →   2 submissions each
TOTAL artifacts = 40
```

So there is exactly **one** task in the whole tournament with a usable score
distribution. Everything else is a pairwise comparison. Per-type artifact
counts: krea2 18, qwen-image 6, ideogram4 5, flux 4, z-image 4. Every
conclusion below is scoped to that reality, and §6 flags where it is too thin.

### 1.2 What the artifacts actually contain (better than expected)

Miners published their **entire checkpoint directory**, not just the adapter:

* **ai-toolkit** artifacts carry `__metadata__.training_info = {"step": N, "epoch": E}`
  on *every* checkpoint → the shipped step count is **OBSERVED**, not inferred.
* Most carry `checkpoints/config.yaml` — the complete resolved training config.
* Five carry **`.<arch>_checkpoint_evaluations.json`** — the miner's own
  reconstruction-probe scores per checkpoint. This is the champion cluster's
  selection machinery, published.
* **kohya** (flux) artifacts carry partial `ss_*` / `modelspec.*` metadata
  (`no_metadata=true` strips the training block but leaves `ss_network_dim`,
  `ss_network_alpha`, `ss_network_args`, `modelspec.date`).

### 1.3 Provenance rules used for "shipped depth"

The pinned G.O.D diffusion evaluator **prefers `checkpoint/last.safetensors`**
(source-contract check recorded in `SN56-GPU-CERTIFICATION-RESULTS.md`).
Accordingly:

| case | rule | label |
|---|---|---|
| `last.safetensors` present with `training_info` | that step | **OBSERVED** |
| `last.safetensors` absent (4 qwen repos) | terminal periodic checkpoint | **INFERRED** |
| metadata scrubbed (6 repos) | ladder ceiling + save cadence | **INFERRED** |
| kohya flux (`last-000040.safetensors` = **epoch** 40) | epoch × N × repeats(=1) | **INFERRED** |

The kohya→steps conversion is the weakest link. `1_<trigger>` is the G.O.D
dataset folder (num_repeats = 1, proven by the probe paths inside the flux eval
sidecars: `/dataset/images/<task>/img/1_lora style/0.png`), and the per-epoch
save cadence of 25.7 s (5D7iEJm5) implies ~1.7 s/step at batch 1 — physically
sane for FLUX rank-128 with T5 training on an H100. It is still an inference;
**flux epochs (40–58) are the OBSERVED quantity, flux steps are derived.**

---

## 2. OBSERVED — the full artifact × score table

`s/img` = shipped steps ÷ n_pairs (the normalising unit the brief asked for).
`OURS` = what `recipe.size_scaled_steps()` on the current pin returns for that
exact (type, N, hours).

```
type       fam       N    h hotkey8   rk test_loss   cfg shipped  s/img  OURS our s/img  ratio prov
flux       product  15 0.75 5FW2Eaae   1  0.007477  None     870   58.0   726      48.4   0.83 INFERRED(kohya epoch 58)
flux       product  15 0.75 5EACrayt   2  0.007768  None     600   40.0   726      48.4   1.21 INFERRED(kohya epoch 40, PROMOTED from 45)
flux       product  15 0.75 5D7iEJm5   1  0.012543  None     750   50.0   726      48.4   0.97 INFERRED(kohya epoch 50)
flux       product  15 0.75 5FNLSgh8   2  0.012625  None     600   40.0   726      48.4   1.21 INFERRED(kohya epoch 40)
ideogram4  product  14 0.75 5FBmn1ax   1  0.012652   174     174   12.4   107       7.6   0.61 OBSERVED
ideogram4  product  14 0.75 5GU4Xkd3   2  0.018480  None    1000   71.4   107       7.6   0.11 INFERRED
ideogram4  style    46 1.00 5FBmn1ax   1  0.080523   341     341    7.4   194       4.2   0.57 OBSERVED
ideogram4  style    46 1.00 5GpcTKW7   2  0.102399  None    None      -   194       4.2    nan UNKNOWN (1 file, no ladder)
ideogram4  style    40 1.00 5GU4Xkd3   1  0.035051  None    1300   32.5   181       4.5   0.14 INFERRED
ideogram4  style    40 1.00 5FBmn1ax   2  0.036588  1523    1523   38.1   181       4.5   0.12 OBSERVED
krea2      design   42 1.00 5FBmn1ax   1  0.026554  2012    2012   47.9  1172      27.9   0.58 OBSERVED
krea2      design   42 1.00 5GU4Xkd3   2  0.026585  None    2000   47.6  1172      27.9   0.59 INFERRED
krea2      design   21 0.75 5EACrayt   1  0.046978  1000    1000   47.6   824      39.2   0.82 OBSERVED
krea2      design   21 0.75 5FNLSgh8   2  0.047567  1000     800   38.1   824      39.2   1.03 OBSERVED (time-killed at 800)
krea2      design   21 0.75 5FW2Eaae   3  0.048727  None    2000   95.2   824      39.2   0.41 INFERRED
krea2      design   21 0.75 5FBmn1ax   4  0.048934  1432    1432   68.2   824      39.2   0.58 OBSERVED
krea2      design   21 0.75 5D2Qee4V   5  0.049950  2000    1750   83.3   824      39.2   0.47 OBSERVED (time-killed at 1750)
krea2      design   21 0.75 5GpcTKW7   6  0.050284  None    None      -   824      39.2    nan UNKNOWN
krea2      design   21 0.75 5FpdSckw   7  0.050457  None    2000   95.2   824      39.2   0.41 INFERRED
krea2      design   21 0.75 5D7iEJm5   8  0.050491  1278    1278   60.9   824      39.2   0.64 OBSERVED
krea2      design   21 0.75 5HLA2QWY   9  0.050981  None     823   39.2   824      39.2   1.00 OBSERVED  <== OURS
krea2      design   21 0.75 5HKEAZxF  10  0.051696  None    1400   66.7   824      39.2   0.59 INFERRED
krea2      design   21 0.75 5GKoYQm7  11  0.051993   972     972   46.3   824      39.2   0.85 OBSERVED
krea2      design   21 0.75 5Ca32LwM  12  0.052468  None    None      -   824      39.2    nan UNKNOWN
krea2      design   21 0.75 5HWPK9f6  13  0.052511  None     200    9.5   824      39.2   4.12 OBSERVED
krea2      design   21 0.75 5FjDsFGA  14  0.053039  1432    1432   68.2   824      39.2   0.58 OBSERVED
krea2      design   43 1.00 5FBmn1ax   1  0.055552  2012    2012   46.8  1172      27.3   0.58 OBSERVED
krea2      design   43 1.00 5D7iEJm5   2  0.058991  1778    1778   41.3  1172      27.3   0.66 OBSERVED
krea2      design   50 1.00 5FBmn1ax   1  0.048011  2012    2012   40.2  1172      23.4   0.58 OBSERVED
krea2      design   50 1.00 5GU4Xkd3   2  0.049463  None    2000   40.0  1172      23.4   0.59 INFERRED
qwen-image logo     31 1.50 5FBmn1ax   1  0.058812   949     949   30.6  1027      33.1   1.08 OBSERVED
qwen-image logo     31 1.50 5GU4Xkd3   2  0.075908  1300     600   19.4  1027      33.1   1.71 OBSERVED (time-killed at 600)
qwen-image design   28 1.25 5FW2Eaae   1  0.090832  1150     850   30.4   836      29.9   0.98 OBSERVED (time-killed at 850)
qwen-image design   28 1.25 5FpdSckw   2  0.092780  1150     850   30.4   836      29.9   0.98 OBSERVED (time-killed at 850)
qwen-image social   41 1.50 5FBmn1ax   1  0.115937  1095    1095   26.7  1027      25.0   0.94 OBSERVED
qwen-image social   41 1.50 5FW2Eaae   2  0.116884  1300     700   17.1  1027      25.0   1.47 OBSERVED (time-killed at 700)
z-image    social   48 1.00 5FBmn1ax   1  0.061716  1317    1317   27.4   860      17.9   0.65 OBSERVED
z-image    social   48 1.00 5GU4Xkd3   2  0.063187  1000    1000   20.8   860      17.9   0.86 OBSERVED
z-image    design   39 1.00 5EACrayt   1  0.094002  1188    1188   30.5   860      22.1   0.72 OBSERVED
z-image    design   39 1.00 5D2Qee4V   2  0.101417  2000    2000   51.3   860      22.1   0.43 OBSERVED
```

Our own artifact is byte-identified: `5HLA2QWY` published
`checkpoints/forge_run.json` (`kind: forge-public-run-recorder`), with
`toolkit_start` → `toolkit_end` = **1041.1 s for 823 steps ⇒ 1.265 s/step**
measured on the tournament host. Rank 9/15, eliminated by 0.97 % against
rank 8.

### 2.1 steps-per-image, rank-1 vs the rest

```
  krea2       RANK1: n=4 min=40.2 med=47.2 max=47.9   | LOSERS: n=14 min= 9.5 med=54.2 max=95.2
  ideogram4   RANK1: n=3 min= 7.4 med=12.4 max=32.5   | LOSERS: n= 2 min=38.1 med=54.8 max=71.4
  qwen-image  RANK1: n=3 min=26.7 med=30.4 max=30.6   | LOSERS: n= 3 min=17.1 med=19.4 max=30.4
  z-image     RANK1: n=2 min=27.4 med=28.9 max=30.5   | LOSERS: n= 2 min=20.8 med=36.1 max=51.3
  flux        RANK1: n=2 min=50.0 med=54.0 max=58.0   | LOSERS: n= 2 min=40.0 med=40.0 max=40.0
```

The winners' steps-per-image band is **tight per type and different between
types** — ~47 (krea2), ~30 (qwen, z-image), ~54 (flux), and ideogram4 alone is
wide (7–33). That is the depth law, in the right unit.

> **REVISION (ideogram4 only).** The ideogram4 row is not a band, it is scatter,
> and it is scatter around *one operator's* choices at `lr 4e-4`. Widen the
> sample to the Jul-20 R1 ideogram4 field (16 miners, task `3cfa1578`,
> SN56-WEEK3-POSTMORTEM §6a) and the range runs 85 → 1200 steps on N=9 with the
> 85-step entry placing 4/16 and the 1200-step entry winning the bracket. Across
> both tournaments ideogram4 depth is **flat and wide**, not a tight interior
> optimum. Do not calibrate our ideogram4 depth off a steps-per-image band.

---

## 3. Does depth actually cause score? Three independent tests

### 3.1 The field published its own depth curves (strongest evidence)

Three miners shipped `.<arch>_checkpoint_evaluations.json` — their own
reconstruction probe (2–4 training images, 3 generations, split into
`text_guided_loss` / `no_text_loss` / `combined_loss`, i.e. **the validator's own
0.25/0.75 decomposition**). These are direct depth→loss curves on real
tournament data:

**krea2, N=21, R1 winner `5EACrayt` (combined_loss, lower is better):**

```
  step  200 → 0.034302
  step  400 → 0.033499
  step  600 → 0.032801
  step  800 → 0.031887
  step 1000 → 0.031846   <- shipped (best, monotone to the end)
```

**z-image, N=39, winner `5EACrayt`:**

```
  step  250 → 0.046502
  step  500 → 0.044723
  step  750 → 0.044902
  step 1000 → 0.043153
  step 1188 → 0.042721   <- shipped (best, deepest)
```

**flux, N=15, `5FNLSgh8` (epochs) and `5EACrayt` (epochs):**

```
  ep  5 → 0.018868 / 0.017372
  ep 10 → 0.017663 / 0.015903
  ep 15 → 0.017614 / 0.015865
  ep 20 → 0.018179 / 0.016366     <- local bump
  ep 25 → 0.018322 / 0.016234
  ep 30 → 0.017806 / 0.016053
  ep 35 → 0.017203 / 0.015840
  ep 40 → 0.016992 / 0.015663     <- best in both, deepest evaluated
```

**Five independent curves, five minima at the deepest checkpoint evaluated.**
None of them turns over. The Jul-16 calibration's conclusion — "deep training
never helped", derived from an 8–128-step probe on 12 photos — is contradicted
by every field curve we can see. It measured the wrong regime.

### 3.2 Rank correlation on the only 14-way field (R1 krea2, N=21, 0.75 h)

```
      200 |                                                      o  0.052511 rk13 5HWPK9f6
      800 |     o  0.047567 rk 2 5FNLSgh8   (recipe outlier: TE-LoRA + krea2_eval_sigmas)
      823 |                                       #  0.050981 rk 9 5HLA2QWY  <== OURS
      972 |                                                 o  0.051993 rk11 5GKoYQm7 (automagic opt)
     1000 |o  0.046978 rk 1 5EACrayt        (recipe outlier: TE-LoRA + krea2_eval_sigmas)
     1278 |                                  o  0.050491 rk 8 5D7iEJm5
     1400 |                                              o  0.051696 rk10 5HKEAZxF
     1432 |                   o  0.048934 rk 4 5FBmn1ax  (loss=mae, dg=3)
     1432 |                                                            o  0.053039 rk14 5FjDsFGA
     1750 |                             o  0.049950 rk 5 5D2Qee4V
     2000 |                 o  0.048727 rk 3 5FW2Eaae
     2000 |                                  o  0.050457 rk 7 5FpdSckw
```

* all 12 with known depth: `spearman(steps, test_loss) = −0.200`
* **excluding the 3 TE-LoRA recipe outliers (the template family only, n = 9):
  `spearman = −0.605`** ⇒ within a fixed recipe, **deeper is better**.

The top-of-pack in the template family shipped 1432–2000 (68–95 steps/img). We
shipped 823 (39 steps/img) and finished 9th. The two artifacts that beat the
whole template family did so at 800–1000 steps but with an entirely different
recipe (§4.2) — depth is not the only axis, but among like-for-like recipes it
is monotone in our favour.

### 3.3 Head-to-head, all 14 tasks

```
krea2   3e0fdcde N=42: 2012 (47.9/img) beat 2000 (47.6/img) by  0.12%   DEEPER won
krea2   db9f7244 N=43: 2012 (46.8/img) beat 1778 (41.3/img) by  6.19%   DEEPER won
krea2   f6725c2b N=50: 2012 (40.2/img) beat 2000 (40.0/img) by  3.03%   DEEPER won
krea2   41025fb5 N=21: 1000 (47.6/img) beat 1432 (68.2/img) by 12.90%   SHALLOWER won (recipe-confounded)
ideo4   1365fa1c N=14:  174 (12.4/img) beat 1000 (71.4/img) by 46.06%   SHALLOWER won
ideo4   b72da8c6 N=40: 1300 (32.5/img) beat 1523 (38.1/img) by  4.39%   SHALLOWER won (both deep)
qwen    4782f46f N=31:  949 (30.6/img) beat  600 (19.4/img) by 29.07%   DEEPER won
qwen    ff643470 N=41: 1095 (26.7/img) beat  700 (17.1/img) by  0.82%   DEEPER won
qwen    7421f056 N=28:  850 (30.4/img) tied  850 (30.4/img)             depth-neutral
z-img   b2582457 N=48: 1317 (27.4/img) beat 1000 (20.8/img) by  2.38%   DEEPER won
z-img   b290d171 N=39: 1188 (30.5/img) beat 2000 (51.3/img) by  7.89%   SHALLOWER won
flux    241cda6c N=15:  870 (58.0/img) beat  600 (40.0/img) by  3.90%   DEEPER won
flux    db5fefc5 N=15:  750 (50.0/img) beat  600 (40.0/img) by  0.66%   DEEPER won
```

Pooled per-type Spearman of `steps_per_image` vs `loss / best_in_task`
(negative = deeper is better): flux −0.889, qwen −0.493, krea2 +0.096,
z-image +0.316, ideogram4 +0.894.

**Interpretation.** The winning depth is an *interior optimum*, not a monotone
direction: on both sides of the band you lose. z-image at 51 steps/img loses to
30; ideogram4 at 71 steps/img loses catastrophically to 12. The band per type is
what §2.1 shows. **Every one of our five current settings sits on or below the
shallow edge of its band, never inside it, except qwen-image.**

> **REVISION — the ideogram4 Spearman (+0.894) must not be read as a result.**
> It is computed over n=5 artifacts of which only **two** are usable pairs, and
> the pair that drives the sign (`1365fa1c`: 12.4/img beat 71.4/img by 46.1%) is
> **confounded**: the deep arm published no `config.yaml` and stripped its
> `__metadata__`, so its lr, EMA and network are all unknown. The other usable
> pair (`b72da8c6`) is deep-vs-deeper, and the *shallower* of the two deep arms
> won by 4.4%. A third "point" (`84be9fcd`) contributes no depth information at
> all — that opponent published two files, no metadata and no ladder. Net: one
> confounded pair, and it is the entire ideogram4 depth signal in this
> tournament.

---

## 4. What the top of the field actually configured

### 4.1 The operator table — one operator owns this tournament

```
  5FBmn1ax  entries=10 wins=8     <- the champion; enters every type
  5GU4Xkd3  entries= 6 wins=1
  5FW2Eaae  entries= 4 wins=2
  5EACrayt  entries= 3 wins=2
  5D7iEJm5  entries= 3 wins=1
  ... 10 more with 1-2 entries, 0 wins
  5HLA2QWY  entries= 1 wins=0     <== OURS (R1 elimination)
```

### 4.2 Two distinct top recipes exist, and they are not the same operator

| | `5FBmn1ax` (8 wins) | `5EACrayt` / `5FNLSgh8` / `5FW2Eaae` / `5FpdSckw` / `5D2Qee4V` cluster |
|---|---|---|
| depth policy | **time-fill** on krea2, **power law** on qwen/z-image, **unstable/exploratory** on ideogram4 (4.5× swing between two same-size style tasks; see §4.4 — this was previously called "family-routed", which §4.4 itself refutes) | power law, moderate |
| network | 32/32 (krea2, z-image, ideogram4); **141/141 and 149/149 on qwen** | 32/32, 40/40, 48/48; 128/128 on qwen |
| lr | template (1e-4 / 4e-4); **qwen `lr = 1.0877e-3 / sqrt(rank)`** | 1e-4 |
| loss | **`mae`** on krea2/qwen/z-image (template is `mse`) | `mse` |
| EMA | on (0.99 ideogram4, 0.995 qwen/z-image); **off on krea2** | **on, 0.995** |
| scheduler | template constant | **`cosine_by_group` with `min_lr_by_initial_lr`** |
| timestep | `linear`/`weighted` (template) | **`krea2_eval_sigmas`** on krea2 |
| noise | template | **`multires_noise_iterations: 6`, `discount: 0.3`** |
| guidance | `differential_guidance_scale: 3` (template 2) | **`differential_guidance_scale: 12.0`** |
| text encoder | off | **on, `text_encoder_lr: 2.5e-07`** (504 extra tensors) |
| selection | ship the terminal step | **reconstruction-probe ladder + promote to `last.safetensors`** |

Both clusters beat us. `5FBmn1ax` wins by depth + a size-scaled network; the
`5EACrayt` cluster wins R1 by an evaluator-aligned objective and checkpoint
selection at only 1000 steps.

### 4.3 Exact laws recovered from the champion's own numbers

Several of `5FBmn1ax`'s configs are exactly reproducible closed forms. These are
not fits to noise — they reproduce every published point to <0.5 %.

**krea2 — a pure clock-fill with no size term:**

```
  steps = (hours*3600 − 478) / 1.552
    h=1.00 → 2012  (published: 2012 on N=42, N=43, N=50 — identical across 8× the size range)
    h=0.75 → 1432  (published: 1432 on N=21)
```
Two equations, two unknowns, exact. Their krea2 policy is *"train until the
clock stops you"* at an assumed 1.552 s/step with 478 s of fixed overhead. Note
how close 478 s is to forge's own `STARTUP_S + EXPORT_RESERVE_S = 480`.

**qwen-image — a power law plus rank-coupled lr/decay:**

```
  steps = 834 * (N/24)^0.51        N=31 → 950 (published 949) ; N=41 → 1096 (published 1095)
  lr    = 1.0877e-3 / sqrt(rank)   rank 141 → 9.16e-5 ✓ ; rank 149 → 8.91e-5 ✓
  wd    = 1.1786e-3 / rank         rank 141 → 8.36e-6 ✓ ; rank 149 → 7.91e-6 ✓
  rank  ≈ 71.7 * N^0.197           N=31 → 141 ✓ ; N=41 → 149 ✓   (2 points; exponent fragile)
```

**z-image — two independent operators land on the same law:**

```
  5FBmn1ax  N=48 → 1317  ⇒ base = 1317*(24/48)^0.5 = 931
  5EACrayt  N=39 → 1188  ⇒ base = 1188*(24/39)^0.5 = 932
```
Different operators, different recipes, both rank-1, `base ≈ 931 @ n_ref 24,
p = 0.5`. The loser on that task ran the flat 2000-step template and lost 7.9 %.

**ideogram4 — the two champion wins, and why they are NOT calibratable:**

```
  N=14 → 174 (won by 46.06%) ; N=46 → 341 (won by 27.2%)
  ⇒ p = ln(341/174)/ln(46/14) = 0.5655 ,  base = 174/(14/24)^0.5655 = 237.5
```

> **REVISION — this two-point law was fitted, shipped in c424362, and is now
> WITHDRAWN.** Two independent reasons, either sufficient on its own:
>
> 1. **The 27.2% win at N=46 carries no depth information.** The opponent on
>    `84be9fcd` published `.gitattributes` + `last.safetensors` and nothing
>    else — no `config.yaml`, no `__metadata__`, no checkpoint ladder. Their
>    depth is unknown, so "341 beat them" is not evidence that 341 is a good
>    depth. Fitting `p` through that point fits noise.
> 2. **Step counts do not transfer across a 16× learning-rate gap.** Every
>    ideogram4 config in this field runs `lr: 0.0004` constant (OBSERVED,
>    `5FBmn1ax` ×3). `forge.ideogram_release_policy` runs *us* at `lr 2.5e-5`
>    cosine-decayed to `2.5e-6`. Under Adam the per-coordinate displacement per
>    step is ≈ lr, so the comparable quantity across two recipes is the lr
>    integral: `174 × 4e-4 = 0.0696` for the champion versus `0.00245` for us at
>    177 steps — **28.5× less parameter movement at "the same" depth** — and the
>    EMA export costs another 2.3× on top (65.6× total). Reproducing his step
>    count reproduces ~1.5% of his training.
>
> ideogram4 depth is therefore set from our own pipeline's measurable
> constraints, not from the field. See §6.2.

### 4.4 Per-family routing — NO (this section previously said "YES, but only on ideogram4")

Same operator, same 1.00 h budget, same model type:

```
  5FBmn1ax ideogram4 product N=14 → 174 steps (12.4/img)   rank 1
  5FBmn1ax ideogram4 style   N=46 → 341 steps ( 7.4/img)   rank 1
  5FBmn1ax ideogram4 style   N=40 →1523 steps (38.1/img)   rank 2   <- 4.5x deeper, LOST
  5FBmn1ax krea2     design  N=21/42/43/50 → clock-fill, size-independent
  5FBmn1ax qwen      logo N=31 / social N=41 → pure f(N), family-independent
  5FBmn1ax z-image   social N=48 → pure f(N)
```

**Verdict: krea2, qwen-image and z-image show NO family routing** — the same
formula runs for design/logo/social. **ideogram4 shows a 4.5× depth swing
between two style tasks of near-identical size**, which is *not* family routing
either — it looks like an unstable or exploratory chooser, and the deep arm
lost. There is no evidence in this tournament for a design/product/logo/social
depth router. Do not build one.

> **REVISION — a style router is FEASIBLE but still rejected, and it is worth
> writing down which of those two facts is load-bearing.**
>
> *Feasible.* The task family is never delivered to the miner container — the
> CLI takes only `--task-id --model --dataset-zip --model-type
> --expected-repo-name --hours-to-complete --trigger-word` (`forge/cli.py:32-42`)
> — so a literal `family` switch is impossible. But **`trigger_word is None`
> separates `style` from every other family 12/12** across the Aug-3 configs
> that published one: both style tasks (`84be9fcd`, `b72da8c6`) carry no trigger
> word, and all ten design/social/logo/product configs do (`AetherFlow UI`,
> `AetherCanvas UI`, `PixelPulse`, `PixelCraft UI`, `AxiomScreens`,
> `BrandEssence`, `AuraFlow`, `LuminaGlow Orb`, …). A style router is therefore
> implementable from data we actually receive.
>
> *Still rejected.* Not on feasibility — on the target. The two style tasks
> disagree with **each other** by 4.5× (341 won at N=46; ~1250 won at N=40), so
> there is no style depth to route *to*. Routing on a clean signal toward an
> unresolved value adds a one-task-per-branch free parameter to the
> highest-variance row in the table, which is half the round-1 draw. Rejected.
> If a future tournament produces a *second* style head-to-head with a shallow
> arm present, revisit — the signal is already there and costs nothing to wire.

### 4.5 Checkpoint selection is real and is present in the field

```
  flux  241cda6c rk2 5EACrayt: trained to epoch 45, ladder e5..e45, but
        last.safetensors is byte-timestamp-identical to epoch 40
        => PROMOTED_EARLIER, confirmed by .flux-best.safetensors
  krea2 41025fb5 rk2 5FNLSgh8: last.safetensors sshs_model_hash == step-800 checkpoint
  krea2 41025fb5 rk1 5EACrayt: .krea2_checkpoint_evaluations.json scored 200/400/600/800/last,
        selected last (1000) because the curve was still improving
  z-img b290d171 rk1 5EACrayt: .zimage_checkpoint_evaluations.json scored 250..1188, selected 1188
  flux  db5fefc5 rk2 5FNLSgh8: .flux_checkpoint_evaluations.json scored e5..e40, selected e40
```

Note carefully: **the selection machinery mostly chose the DEEPEST checkpoint.**
Selection is not a way to ship shallow. It is a way to ship deep *safely* — you
overshoot, you measure, and the measurement keeps telling you to ship the end.
This is an argument *for* raising depth, not against it.

### 4.6 Rank / alpha / LR / scheduler / EMA in the top quartile

14 artifacts took rank 1; **11 of them published a `config.yaml`** (the two flux
winners and `5GU4Xkd3` on `b72da8c6` scrubbed theirs). Counts below are over
those 11 unless stated.

* **network rank/alpha (from tensor shapes, so all 14 are covered):** 32/32 in
  **9 of 14** — every krea2, ideogram4 and z-image winner. The other 5 are
  structural, not stylistic: both flux winners run kohya `dim 128 / alpha 64`
  (so do both flux *losers* — it is that toolchain's norm), and the three qwen
  winners run 128/128, 141/141, 149/149. **No non-flux, non-qwen winner used
  anything but 32/32**, and the two losers who *raised* rank on those types
  (40/40 on krea2 R1 → rank 5, 48/48 on z-image → rank 2) both lost.
* **lr:** the template value in **9 of 11** (1e-4, or 4e-4 on ideogram4). The
  two exceptions are the champion's `lr = 1.0877e-3/sqrt(rank)` on qwen
  (9.16e-5, 8.91e-5). The only artifact in the whole tournament with a
  genuinely different optimizer (`automagic`, effective lr 9.27e-7) ranked
  11/14.
* **scheduler:** absent (template constant) in **10 of 11**. `cosine_by_group`
  with `min_lr_by_initial_lr: {1e-4: 1e-5}` appears only in the `5EACrayt`
  cluster — and that cluster took R1 rank 1 *and* rank 2.
* **EMA:** the sharpest signal available is negative — **all 5 artifacts that
  explicitly set `use_ema: false` lost their task** (krea2 ranks 5, 8, 11, 2;
  z-image rank 2). On ideogram4 the champion's two wins carry `ema_decay 0.99`
  and his one loss omits EMA entirely. On qwen-image **all six** artifacts run
  `0.995`, so EMA is not discriminative there. The champion's krea2 wins omit
  EMA. Read this as "do not disable EMA", not "EMA wins".
* **loss_type:** the champion runs `mae` on krea2/qwen/z-image (template is
  `mse`) and `mse` on ideogram4. 8 of his 10 entries won.

---

## 5. What binds in `forge/recipe.py` today — and it is not the depth law

```
task      type       fam       N    h |   law   cap   NOW |  law*  cap*   REC | winner now/win rec/win  binds
241cda6c  flux       product  15 0.75 |   870   726   726 |   870  1002   870 |    870    0.83    1.00  clock->size
db5fefc5  flux       product  15 0.75 |   870   726   726 |   870  1002   870 |    750    0.97    1.16  clock->size
1365fa1c  ideogram4  product  14 0.75 |   107   605   107 |   421   477   421 |    174    0.61     -    size->size
b72da8c6  ideogram4  style    40 1.00 |   181   860   181 |   589   674   589 |   1300    0.14     -    size->size
84be9fcd  ideogram4  style    46 1.00 |   194   860   194 |   616   674   616 |    341    0.57     -    size->size
41025fb5  krea2      design   21 0.75 |  1122   824   824 |  1432  1484  1432 |   1000    0.82    1.43  clock->size
3e0fdcde  krea2      design   42 1.00 |  1587  1172  1172 |  1825  2097  1825 |   2012    0.58    0.91  clock->size
db9f7244  krea2      design   43 1.00 |  1606  1172  1172 |  1840  2097  1840 |   2012    0.58    0.91  clock->size
f6725c2b  krea2      design   50 1.00 |  1732  1172  1172 |  1939  2097  1939 |   2012    0.58    0.96  clock->size
7421f056  qwen-image design   28 1.25 |  1080   836   836 |   909   836   836 |    850    0.98    0.98  clock->clock
4782f46f  qwen-image logo     31 1.50 |  1137  1027  1027 |   957  1023   957 |    949    1.08    1.01  clock->size
ff643470  qwen-image social   41 1.50 |  1307  1027  1027 |  1104  1023  1023 |   1095    0.94    0.93  clock->clock
b290d171  z-image    design   39 1.00 |  1402   860   860 |  1186  1573  1186 |   1188    0.72    1.00  clock->size
b2582457  z-image    social   48 1.00 |  1556   860   860 |  1315  1573  1315 |   1317    0.65    1.00  clock->size
```

`law` = the size power law before capping; `cap` = the `SEC_PER_IT` wall-time
cap; `NOW` = `min(law, cap)`. **In 11 of 14 tasks the clock cap is what
truncates us, not the depth policy.** On z-image the size law wanted 1402/1556
(right on the field) and the clock cut it to 860.

> **REVISION (post-review).** The `cap*`/`REC` columns above are the SHIPPED
> constants, not the ones this document first recommended. Two changed after an
> adversarial review:
>
> * **`SEC_PER_IT["krea2"]` 1.5 → 1.35.** At 1.5 the clock still truncated the
>   R1 shape (1432 → 1336) — i.e. the recommendation did not achieve its own
>   stated goal of making the size law the binding constraint. 1.35 is the round
>   value inside the threshold at which that stops happening (1.399 at 0.75 h);
>   1.30 is indistinguishable in output. §6.1.
> * **`SEC_PER_IT["qwen-image"]` 4.0 → 4.7 with `MARGIN_BY_TYPE["qwen-image"]
>   = 0.98`.** MARGIN 0.92 applied globally raised the qwen cap 1027 → 1122
>   without touching a rate constant that was already 14 % faster than the
>   field's own reproduced measurement, and the result would have been killed on
>   two of the three real qwen shapes. §5.1 and §6.4.

### 5.1 The clock is wrong — proven from completed field runs

For any miner whose shipped step count **reached its configured step count**,
the run completed inside the budget, so
`s/step ≤ (hours·3600 − 478) / shipped_steps` is a hard upper bound on the
achievable rate on tournament hardware.

```
  type        tightest field bound     forge SEC_PER_IT     over-conservative by
  krea2         1.55 s/step             2.2                  1.42x     (our own measured: 1.265)
  ideogram4     2.05 s/step             3.0                  1.46x
  z-image       1.56 s/step             3.0                  1.92x
  qwen-image    4.49 s/step             4.0                  0.89x
  flux          1.71..4.27 s/step*      2.5                  ~1.25x  (*kohya epoch cadence, batch-1 inference)
```

Combined with `MARGIN = 0.85` applied *on top of* a 480 s fixed reserve, forge
discards ~40 % of every krea2/z-image budget. This is the mechanical cause of
the entire krea2 and z-image gap.

> **CORRECTION (post-review).** The qwen line above originally read
> *"already tight — do not lower"*, and the whole table conflated two different
> kinds of evidence. **A completed run gives a BOUND, not a rate.** It says the
> miner was *no slower than* that; it says nothing about how much slack he had.
> A **killed** run is the one that yields an actual rate.
>
> Recomputed against the real terminate gate — training is stopped at
> `budget − EXPORT_RESERVE_S(180) − STOP_MARGIN_S(45)` and `STARTUP_S(300)` of
> what is left is model load, so the step window is `W(h) = h·3600 − 525`:
>
> ```
>   type        kind      s/step   artifact
>   krea2       MEASURED   1.265   OURS — 5HLA2QWY forge_run.json, 823 steps in
>                                  1041.1 s of toolkit_start..toolkit_end
>   krea2       BOUND      1.519   5FBmn1ax AND 5FjDsFGA each completed 1432
>                                  on the real R1 shape (2175/1432)
>   qwen-image  MEASURED   4.676   5FW2Eaae AND 5FpdSckw, identical configs,
>                                  both cfg 1150 on 7421f056 (h=1.25), both
>                                  killed with their last save at 850 (3975/850)
>   qwen-image  BOUND      4.452   5FBmn1ax completed 1095 in 1.5 h
>   z-image     BOUND      1.538   5D2Qee4V completed 2000 in 1.0 h
>   ideogram4   BOUND      2.019   5FBmn1ax completed 1523 in 1.0 h; DOUBLE it
>                                  to 4.038 for our do_cfg batch-2 step
>   flux        BOUND      2.500   rank-1 5FW2Eaae, 58 kohya epochs x N=15
> ```
>
> **`SEC_PER_IT["qwen-image"] = 4.0` was therefore not "tight" — it was 14 %
> FASTER than the field's own reproduced measurement**, and the only thing
> hiding that was `MARGIN = 0.85` discarding 15 % of the budget. Raising MARGIN
> to 0.92 globally removed the accidental compensation and left the type
> planning work the field has never shown fits.
>
> Two qwen artifacts imply much slower rates (5FW2Eaae 6.96 on ff643470;
> 5GU4Xkd3 8.13 on 4782f46f) and are **excluded**: neither is reproduced, and
> each is contradicted by the *same operator* running far faster on another qwen
> task in the same tournament, so they cannot be a stable property of the
> hardware. A constant that survived 8.13 s/step would plan ~400 qwen steps
> against a field that shipped 949–1095.
>
> Likewise the krea2 artifacts of 5EACrayt and 5FNLSgh8 are excluded from rate
> estimation: both carry 504 text-encoder tensors (TE-LoRA, which we never
> train), and 5FNLSgh8's kill implies 2.72 s/step — a config difference, not a
> hardware rate.

---

## 6. RECOMMENDED STEP_TABLE

```python
_N_REF = 24

STEP_TABLE = {
    #                base  n_ref    p    min    max
    "flux":      dict(base=1100, n_ref=24, p=0.50, min=500, max=2000),  # UNCHANGED
    "krea2":     dict(base=1500, n_ref=24, p=0.35, min=600, max=2200),  # was 1200/0.50/100/2000
    "ideogram4": dict(base= 500, n_ref=24, p=0.32, min=350, max= 620),  # was  140/0.50/ 48/ 400; the 240/0.57/120/1600 fit is WITHDRAWN — §6.2
    "z-image":   dict(base= 930, n_ref=24, p=0.50, min=350, max=1800),  # was 1100/0.50/400/2000
    "qwen-image":dict(base= 840, n_ref=24, p=0.51, min=300, max=1600),  # was 1000/0.50/400/3000
}

SEC_PER_IT = {"flux": 2.0,       # was 2.5
              "krea2": 1.35,     # was 2.2, and 1.5 in the first version of this
                                 # document.  1.5 still let the clock truncate
                                 # the R1 shape 1432 -> 1336; 1.35 does not, and
                                 # 1.30 is indistinguishable in output.  Our own
                                 # measured rate is 1.265 GROSS OF STARTUP.
              "ideogram4": 4.2,  # was 3.0. NOT the field bound: the 2.05 bound
                                 # was measured on configs WITHOUT do_cfg. Ours
                                 # sets do_cfg -> transformer at batch 2 every
                                 # step + a second text-encoder forward, ~2x.
                                 # (This line previously read 2.1 and disagreed
                                 # with the shipped recipe.py.)
              "z-image": 1.8,    # was 3.0  (field bound 1.56)
              "qwen-image": 4.7} # was 4.0, and 4.0 was NOT "tight": it is 14%
                                 # faster than the field's own reproduced
                                 # measurement of 4.676 s/step (two operators,
                                 # identical configs, both killed at 850/1150 on
                                 # 7421f056).  See the §5.1 correction.

MARGIN = 0.92            # was 0.85 — see 6.6.  DEFAULT ONLY.
MARGIN_BY_TYPE = {       # applying 0.92 globally was a regression: it raised the
    "flux": 0.92,        # qwen cap 1027 -> 1122 on the one type with no clock
    "krea2": 0.92,       # headroom, and at the field's own 4.676 s/step that
    "ideogram4": 0.92,   # plan is killed on 2 of the 3 real qwen shapes
    "z-image": 0.92,     # (7421f056: 909 planned -> stops at 850 -> ships 728;
    "qwen-image": 0.98,  #  ff643470: 1104 planned -> stops at 1042 -> ships 884).
}                        # qwen's SEC now carries no pad, so its margin carries
                         # the cushion instead: 1 - 45/budget ~= 0.99 would plan
                         # exactly to the terminate trigger, so 0.98 IS a cushion.
```

Replayed against the 14 real Aug-3 shapes this lands at:

| type | winners shipped | forge today | now/win | **recommended** | **rec/win** |
|---|---|---|---|---|---|
| krea2 | 2012, 1000, 2012, 2012 | 1172, 824, 1172, 1172 | 0.64× | 1825, 1432, 1840, 1939 | **1.05×** |
| ideogram4 | 174, 341, 1300 | 107, 194, 181 | 0.44× | **421, 616, 589** | **n/a — see below** |
| qwen-image | 949, 850, 1095 | 1027, 836, 1027 | 1.00× | 957, 836, 1023 | **0.98×** |
| z-image | 1317, 1188 | 860, 860 | 0.69× | 1315, 1186 | **1.00×** |
| flux | 870, 750 | 726, 726 | 0.90× | 870, 870 | 1.08× |

### 6.1 krea2 — `base 1200→1500, p 0.50→0.35, min 100→600, max 2000→2200, SEC 2.2→1.35`

*Reasoning.* The champion's krea2 policy has **no size term at all** — 2012
steps on N=42, N=43 *and* N=50, and 1432 on N=21 at the shorter budget. Their
depth is set entirely by the clock. Flattening `p` from 0.50 to 0.35 mirrors
that (weak size dependence) while keeping a floor for pathologically small
sets; the raised `base` and `max` exist purely so the *recalibrated clock*, not
the size law, becomes the binding constraint on 1.0 h tasks — which is exactly
the champion's behaviour.
`min` raised 100→600 because a 100-step krea2 is not a competitive artifact in
any observation we have (the one 200-step krea2 in the field ranked 13/14).

> **REVISION (post-review): `SEC_PER_IT` is 1.35, not 1.5, and the R1 depth is
> 1432, not 1336.** At 1.5 the clock cap on a 0.75 h budget is 1336, which
> truncates the law's 1431.5 → so the recommendation above did not actually
> achieve the goal it states one paragraph earlier. Replay of all three
> candidates over the four real krea2 shapes:
>
> ```
>   shape                     law   SEC 1.5   SEC 1.35   SEC 1.30
>   41025fb5 N=21 h=0.75     1432      1336       1432       1432
>   3e0fdcde N=42 h=1.00     1825      1825       1825       1825
>   db9f7244 N=43 h=1.00     1840      1840       1840       1840
>   f6725c2b N=50 h=1.00     1939      1888       1939       1939
> ```
>
> The law stops being truncated for any `SEC ≤ 1.399` (0.75 h) and `≤ 1.461`
> (1.0 h). **1.35 and 1.30 are identical in output**, so 1.30's extra 4 % of
> optimism buys nothing and only makes `projected_wall_s` less honest. 1.35 is a
> 6.7 % pad over our own measured 1.265 s/step — and that 1.265 is
> `toolkit_start → toolkit_end`, i.e. **gross of startup**, so charging
> `STARTUP_S = 300 s` on top of it is pure additional cushion (net of a 300 s
> startup the same artifact implies 0.90 s/step). At the field's tightest krea2
> bound (1.519 s/step) all four planned depths still complete.

*Residual uncertainty — the largest of any type.* The R1 winner shipped only
1000 steps and beat the 1432/1750/2000 pack. Depth was **not** the differentiator
there: they ran `timestep_type: krea2_eval_sigmas`, TE-LoRA, EMA 0.995,
multires noise, `cosine_by_group`, and `differential_guidance_scale: 12.0`.

> **CORRECTION (post-review).** This section previously claimed the change moves
> us "into the same band as ranks 3–5". **That claim is withdrawn — it is not
> supported by the ladder it cites.** Inside the 9-artifact R1 template family
> the observed depth→rank map is
>
> ```
>    823 → 9 (OURS)   972 → 11   1278 → 8   1400 → 10
>   1432 → 4 AND 14   1750 → 5   2000 → 3 AND 7
> ```
>
> and OLS on that family gives `loss = 0.053134 − 1.677e-6·steps` with a
> residual sd of `1.16e-3`. Interpolated onto the observed loss ladder:
>
> ```
>   steps    predicted loss    predicted rank    ±1 sd band
>     823        0.051755            11             9..14
>    1148        0.051210            10             6..12
>    1336        0.050894             9             5..12
>    1432        0.050734             9             5..11
>    2000        0.049781             5             3..9
> ```
>
> So **1336 and 1432 are statistically indistinguishable** (0.14 residual sd
> apart), and *neither* is predicted to reach ranks 3–5. What the change is
> actually worth: 823 → 1432 is 0.88 sd of predicted loss, and the R1 cut we
> missed was 0.42 sd — so the depth deficit was real and roughly twice the gap,
> but a single 1432 artifact took rank 4 while another took rank 14. **Depth is
> necessary, not sufficient. Treat the recipe/selection work as the thing that
> closes R1.**
>
> The case for 1432 over 1336 is therefore not the regression — it is that (a)
> the clock constant that produced 1336 is unmeasured pad over a first-party
> measurement, and (b) 1432 is a depth **two independent operators demonstrably
> completed on this exact (type, N, hours) triple**, whereas 1336 is a depth
> nobody ran.
>
> **Downside if the box is slower than the field's tightest bound.** A deadline
> stop DEGRADES depth rather than forfeiting: `_run_toolkit → _terminate →
> _finalize` promotes the newest valid periodic save. With `save_every = 287` a
> stop anywhere in (1148, 1432) ships 1148 — predicted rank 10, still the same
> band as the 1336 that `SEC 1.5` would have produced, and 1.4× the 823 we
> actually shipped. Verified by execution against the real finalizer at all
> three candidate rates (`test_krea2_overrun_degrades_depth_instead_of_forfeiting`).

### 6.2 ideogram4 — `base 140→500, p 0.50→0.32, min 48→350, max 400→620, SEC 3.0→4.2`

> **THIS SECTION IS A REVISION.** It previously recommended
> `base 240 / p 0.57 / min 120 / max 1600 / SEC 2.1`, a two-point fit to the
> champion's own step counts. That recommendation shipped in c424362 and is
> **withdrawn** for the reasons in §4.3 and below. What follows replaces it.

**Why the two-point fit was wrong.** Not because it was thin — because it was
measuring the wrong quantity.

1. One of its two points carries no depth information. On `84be9fcd` the
   opponent published two files (`.gitattributes`, `last.safetensors`), no
   `config.yaml`, no `__metadata__`, no ladder. "341 beat them by 27.2%" says
   nothing about 341.
2. Step counts are not comparable across recipes at different learning rates,
   and the gap here is 16×. Field ideogram4 configs: `lr 0.0004` constant.
   Ours (`forge.ideogram_release_policy`, OBSERVED active): `lr 2.5e-5` cosine
   → `eta_min 2.5e-6`. Under Adam, displacement/step ≈ lr, so:

   | | steps | Σ lr (Adam path length) | EMA-weighted Σ lr | vs champion |
   |---|---|---|---|---|
   | champion `1365fa1c` | 174 | 0.0696 | — (`0.99^174` = 0.17) | 1× |
   | us, old law | 177 | 0.00245 | 0.00106 | **65.6× less** |
   | us, new law | 421 | 0.00580 | 0.00417 | 16.7× less |
   | us, at the do_cfg clock ceiling | 477 | 0.00657 | 0.00498 | 14.0× less |

   **There is no reachable depth at which our ideogram4 recipe over-trains
   relative to the field.** The whole "shallow beat deep by 46%" concern is
   about a regime we are two orders of magnitude away from.

**What the evidence actually is** (all six Aug-3 ideogram4 artifacts
re-derived independently from their safetensors `__metadata__` headers, plus
the Jul-20 R1 field):

| task | family | N | h | rank 1 | rank 2 | what it proves |
|---|---|---|---|---|---|---|
| `1365fa1c` | product | 14 | 0.75 | **174** | >900 (+46.1%) | shallow ≫ deep — but rank 2 stripped its metadata and published no config, so recipe is confounded with depth |
| `84be9fcd` | style | 46 | 1.0 | **341** | unknown (+27.2%) | **nothing** |
| `b72da8c6` | style | 40 | 1.0 | **≥1200** | 1523 (+4.4%) | 1250 ≈ 1523; says nothing about 321 |
| Jul-20 `3cfa1578` (16 miners) | — | 9 | — | **1200** | … | 85 steps → 4/16; deep cluster 722–1000+; recorded finding: the metric "did NOT punish overtraining" |

Across two tournaments ideogram4 depth is **flat and wide** — 85 through 1250
are all competitive — with the deep end favoured in the larger (16-miner)
field. And there is **no size law to fit**: N=9→1200, N=14→174, N=40→1250,
N=46→341 is uncorrelated with N.

**So the row is set from the two constraints we can measure on our own
pipeline**, not from the field:

* **(a) The EMA floor.** `save()` always exports the EMA shadow; the shadow is
  seeded from the LoRA at init (B = 0, zero effect) and built with
  `use_num_updates=False`, so there is no warm-up and the decay is a flat 0.995
  from step 1 (`BaseSDTrainProcess.py:566-568,840-851`; `toolkit/ema.py`).
  `0.995^steps` of every artifact we upload is literally the untrained init:
  **41% at 177 steps, 12% at 421, 4.6% at 616.** Depth is the only lever on
  this that does not require re-signing the hash-bound release activation.
* **(b) The do_cfg clock ceiling.** `do_cfg: true` runs the transformer at
  batch 2 every step, so our rate is ~4.2 s/step, not the field's 2.05. The
  reachable ceiling is 674 steps at 1.0 h and 477 at 0.75 h. **The `b72da8c6`
  winning regime (~1250) is unreachable while do_cfg is on** — a do_cfg
  decision, not a depth decision.

`base 500 / p 0.32` puts the size law 10–15% under that ceiling on all three
real shapes:

```
  1365fa1c  N=14 h=0.75 -> 421  (cap 477)  wall 2248/2700 = 83%   0.995^T = 12.1%
  b72da8c6  N=40 h=1.00 -> 589  (cap 674)  wall 2954/3600 = 82%   0.995^T =  5.2%
  84be9fcd  N=46 h=1.00 -> 616  (cap 674)  wall 3067/3600 = 85%   0.995^T =  4.6%
```

Because the **law** binds and the clock does not, this row is invariant to
`MARGIN` (0.85→0.95) and to any `SEC_PER_IT` revision (2.1→4.2), and it absorbs
a 21% error in the INFERRED 4.2 s/step constant before anything truncates. A
truncation then degrades rather than forfeits: `forge/tasks/aitoolkit.py`
`_run_toolkit → _terminate → _finalize` promotes the newest periodic save, and
each shape budgets four of them.

`p 0.57→0.32` mirrors krea2's deliberately flat 0.35, because the field shows
no size signal. `min 350` binds only below N≈8, under the smallest ideogram4
dataset ever observed (N=9). **`max 1600→620` fixes an inert constant**: the
old law topped out at 365 at N=50, so 1600 could never bind within 4×, whereas
620 binds from N≥47 — inside the 9–50 size range the tournaments have actually
produced.

*Residual uncertainty — still the highest-variance row in the table.* We have
never run the activated ideogram4 recipe past ~200 steps in a tournament, so
421–616 is a 3–6× extrapolation of an unvalidated recipe on half the round-1
draw. The bet is that the two mechanical constraints above dominate a
confounded 46% head-to-head. **Pre-commit: if ideogram4 is the R1 draw and we
lose, the correct next experiment is `do_cfg` on/off at matched depth — which
buys back 2× the reachable depth — NOT another depth change.**

### 6.3 z-image — `base 1100→930, p unchanged 0.50, min 400→350, max 2000→1800, SEC 3.0→1.8`

*Reasoning.* The cleanest result in the audit. Two **different** operators
(`5FBmn1ax`, `5EACrayt`), different recipes, both rank 1, imply
`base = 931` and `932` at `n_ref 24, p = 0.5` — agreement to 0.1 %. The
recommended `base=930` reproduces both winners to within 2 steps (1315 vs 1317;
1186 vs 1188). The `base` is being *lowered* 1100→930, but the shipped depth
*rises* 860→1315 because the real problem was `SEC_PER_IT = 3.0` when a field
miner completed 2000 steps in the same 1.0 h (⇒ ≤ 1.56 s/step). The 2000-step
flat-template entrant lost by 7.9 %, so `max` is pulled to 1800 to keep us out
of that regime.

*Residual uncertainty.* Only 2 tasks, 4 artifacts — but the two-operator
agreement and the clear loss of the 2000-step arm make this the
highest-confidence row despite the small n. Unmeasured: our own z-image
throughput. `SEC_PER_IT = 1.8` is a 15 % pad over the field's 1.56 bound but we
have never run z-image on our host.

### 6.4 qwen-image — `base 1000→840, p 0.50→0.51, min 400→300, max 3000→1600, SEC 4.0→4.7, MARGIN 0.98`

*Reasoning.* **The depth is already right — `now/win = 1.00`.** This row is
changed only to make it right *for the right reason*. Today we land on the field
by accident: the size law wants 1137/1307 and `SEC_PER_IT = 4.0` happens to cut
it to 1027. If anyone fixes the clock without fixing the law, qwen-image
immediately over-trains by 25 %. The champion's exact recovered law is
`834 · (N/24)^0.51` (949 and 1104 vs published 949 and 1095), so encoding
`base=840, p=0.51` makes the size law the intended binding constraint. `max`
drops 3000→1600 because nothing in the field went past 1300 configured / 1095
shipped, and four of the six qwen artifacts were **time-killed** by
over-scheduling (1150→850, 1300→700, 1300→600).

> **REVISION (post-review): `SEC_PER_IT` does NOT stay at 4.0.** The claim that
> 4.0 "is the only per-type constant the field does not contradict" rested on a
> completed-run BOUND (4.49), and a bound is not a rate. The field also contains
> a *reproduced measurement*, and it is slower than 4.0: on `7421f056` (h=1.25)
> **5FW2Eaae and 5FpdSckw ran identical configs, both configured 1150, and both
> were killed with their last periodic save at 850** ⇒ `3975/850 = 4.676 s/step`.
> Two independent operators agreeing exactly is the strongest rate datum in the
> qwen set. **4.0 was 14 % optimistic, and `MARGIN = 0.85` was the only thing
> hiding it.**
>
> Raising MARGIN to 0.92 for every type at once removed that accidental
> compensation and left qwen — the one type whose clock actually binds — planning
> past what the field shows fits. Replayed at 4.676 s/step against the real
> terminate gate:
>
> ```
>   task       h     planned (M=0.92, SEC=4.0)   stops at   SHIPS
>   7421f056  1.25            909                   850      728   (save_every 182)
>   ff643470  1.50           1104                  1042      884   (save_every 221)
>   4782f46f  1.50            957                  1042      957   FINISH
> ```
>
> i.e. two of three real qwen shapes lose 13 % and 20 % of their depth to a
> deadline stop, purely from over-scheduling.
>
> **Fix: `SEC_PER_IT["qwen-image"] = 4.7` (the honest rate) with
> `MARGIN_BY_TYPE["qwen-image"] = 0.98` (the cushion).** That plans 836 / 957 /
> 1023, all of which complete at 4.676 with 14 / 85 / 19 steps to spare, and it
> makes `projected_wall_s` truthful — at 4.0 it claimed 676 s of slack on
> `7421f056` where the real figure is ~91 s. Keeping SEC at 4.0 and reverting the
> margin to 0.85 gives near-identical steps (836 / 957 / 1027) but leaves a rate
> constant that is knowingly wrong, which is the class of defect this whole
> recalibration exists to remove.
>
> A per-type margin of **0.98 is not "less safe than 0.92"** — with an honest
> rate the identity `budget·M − 480 == budget − 525` gives `M = 1 − 45/budget ≈
> 0.99` for "plan exactly to the terminate trigger", so 0.98 *is* the cushion.
> The four other rows keep 0.92 because their size law binds well below the
> clock, which makes their margin inert.

*Residual uncertainty.* Two-point law from a single operator; `p = 0.51` is
recovered from exactly two tasks. But it is corroborated by the fact that both
of that operator's qwen entries *completed* while all four opposing qwen
artifacts were deadline-killed — i.e. his law is calibrated to finish, ours
should be too. Not addressed here: the champion's `rank ≈ 141–149` on qwen
(vs our template 32) and `lr = C/sqrt(rank)`. That is a capacity change, not a
depth change, and belongs in a separate decision.

### 6.5 flux — `base/p/min/max UNCHANGED, SEC 2.5→2.0`

*Reasoning.* The flux size law is already correct: uncapped it returns
**870 steps at N=15**, which is *exactly* what the rank-1 miner shipped
(58 epochs × 15). The only defect is that `SEC_PER_IT = 2.5` cuts it to 726.
The field's kohya save cadence (25.7 s per epoch at N=15 for `5D7iEJm5`) implies
~1.7 s/step at batch 1; 2.0 restores the law as the binding constraint.

*Residual uncertainty — two separate ones.*
1. **The step count is inferred, not observed.** kohya writes epochs, not
   steps. The conversion assumes batch 1 and `num_repeats = 1`. The repeat
   factor is proven from the dataset path; the batch size is not. If any flux
   miner ran batch > 1 their true step count is lower and their steps/image is
   unchanged — which is why §2.1 reports steps/image, the batch-invariant
   quantity.
2. **`STEP_TABLE["flux"]` may not even apply.** `forge/tasks/flux_kohya.py`
   routes *standalone-checkpoint* FLUX bases (e.g. `dataautogpt3/FLUX-MonochromeManga`
   on task `db5fefc5`) to the kohya path, where depth is governed by
   `flux_kohya_config.MAX_TRAIN_STEPS = 250` at `train_batch_size 4 ×
   gradient_accumulation 2`, **not** by `STEP_TABLE`. Snapshot bases
   (`rayonlabs/FLUX.1-dev` on task `241cda6c`) go to ai-toolkit and do use
   `STEP_TABLE`. Both Aug-3 flux tasks were N=15/0.75 h and the field ran
   **batch 1**. Our kohya arm would ship roughly 75–80 optimiser steps at
   effective batch 8 (≈ 40–43 sample-passes per image — inside the field's
   40–58 band, but with ~8× fewer gradient updates; estimate, not measured).
   **This is an unresolved divergence, out of scope for `STEP_TABLE`, and
   flagged for a separate decision before Monday.**

### 6.6 `MARGIN 0.85 → 0.92`, but PER TYPE (adjacent, and load-bearing)

Not part of the requested table, but the numbers above do not materialise
without it. `size_scaled_steps` computes
`train_s = budget*MARGIN − STARTUP_S − EXPORT_RESERVE_S`, i.e. it applies a 15 %
haircut **on top of** a 480 s fixed reserve — double-counting. The champion's
recovered model is `(budget − 478)` with **no** multiplicative margin, and
478 ≈ our 480. At `MARGIN = 0.85` even a corrected `SEC_PER_IT` leaves krea2 at
1700 on a 1.0 h task. Sensitivity, krea2 @ 1.0 h, `SEC = 1.35`:

```
  MARGIN 0.85 → 1700    0.90 → 1844    0.92 → 1901    0.95 → 1988    0.98 → 2076
```

0.92 keeps ~290 s of jitter headroom beyond the 480 s reserve. **Note:**
over-scheduling is recoverable — `forge/tasks/checkpoints.py` promotes the
highest valid periodic save to `last.safetensors` on a kill — so the asymmetry
favours scheduling slightly deep over slightly shallow, *within* what the clock
can fund.

> **CORRECTION (post-review): applying 0.92 GLOBALLY was a regression, and this
> section is why it happened.** MARGIN was reasoned about on krea2, where it is
> inert once the size law binds, and then applied to every type. It landed on
> qwen-image — the one row whose clock is the active constraint and whose rate
> constant carried no pad — raising its cap 1027 → 1122 (+9.3 %) with no
> compensating change. At the field's own reproduced 4.676 s/step that plan is
> killed on two of the three real qwen shapes (§6.4).
>
> **A margin is a per-type dial because the headroom it spends is per-type.**
> The shipped policy is `MARGIN = 0.92` as the default with
> `MARGIN_BY_TYPE = {..., "qwen-image": 0.98}`, and the guard is no longer the
> margin value at all — it is an invariant asserted over every type × every real
> Aug-3 shape:
>
> > *the planned step count must still complete at the slowest rate that type's
> > own published artifacts support.*
>
> (`recipe.FIELD_DEMONSTRATED_DEPTH` + `test_every_shape_finishes_at_its_field_rate`,
> exact integer arithmetic so no float rounding can flip a verdict.)

### 6.7 Where the evidence is too thin to justify a change — stated explicitly

| type | n artifacts | n usable H2H | verdict |
|---|---|---|---|
| krea2 | 18 | 4 | **Sufficient.** Only type with a real distribution (R1, n=14) plus 3 brackets. Change with confidence in the direction; uncertain in the magnitude at 0.75 h. |
| z-image | 4 | 2 | **Sufficient despite n=4** — two independent operators agree on the law to 0.1 % and the over-deep arm demonstrably lost. |
| qwen-image | 6 | 3 | **Sufficient for the law, but it is a 2-point fit from one operator.** The depth recommendation is near-neutral (0.98× the winners). The *clock* is the risk here, not the law: qwen is the only type where the cap binds, it is UNMEASURED on our own host, and four of its six artifacts were deadline-killed. |
| ideogram4 | 5 | **1** (not 2 — `84be9fcd`'s opponent published no metadata) | **NOT USABLE FOR CALIBRATION AT ALL**, and the reason is stronger than thinness: the field runs `lr 4e-4` and we run `lr 2.5e-5`, so their step counts do not transfer (28.5× less Adam path length at matched depth). §6.2 sets this row from our own EMA floor and do_cfg clock ceiling instead. The one usable head-to-head is retained as a caution, not as a calibration point. |
| flux | 4 | 2 | **TOO THIN for the depth law** — and the depth law may not even be the governing code path (§6.5). Recommending only the clock change, which is separately supported. |

**Not evidence for anything:** per-family depth routing (§4.4) — three of five
types show none, and the one type that varies did so incoherently and lost the
deep arm. Do not build a family router from this tournament.

---

## 7. Things found in passing that are not depth, ranked by expected value

1. **`timestep_type: krea2_eval_sigmas`** — the R1 rank-1 and rank-2 artifacts
   both train on the evaluator's own sigma schedule. Neither of the two clusters
   that beat the whole 14-way template field lacked it.
2. **Reconstruction-probe checkpoint selection is shipped in the field**, with
   the miners' own probe scores published (§4.5). Every selected checkpoint was
   at or near the deepest evaluated. `SN56-WEEK5-POSTMORTEM` H7 is confirmed
   from a second tournament.
3. **`loss_type: mae`** on krea2/qwen/z-image — the 8-win operator's setting;
   the template is `mse`.
4. **Never set `use_ema: false`** — all five artifacts that did lost their task.
   EMA 0.995 is universal on qwen-image (winners and losers alike), so it is a
   floor, not an edge.
5. **Effective dataset multiplier is measurable and is NOT 3.0.** Solving
   `floor(step/E) == epoch` over the published (step, epoch) pairs gives, e.g.,
   `E = 54` for the 21-image R1 dataset — identical across six independent
   miners, i.e. **2.571 samples per source image**, not the 3.0 that a naive
   `resolution: [512, 768, 1024]` fork implies. qwen artifacts from one operator
   sit at 2.14, another at 2.64. This is an independent, cheap measurement of
   the geometry-mismatch surface and it is per-task computable from the harvest.
6. **Text-encoder LoRA:** the R1 rank-1/rank-2 artifacts carry 504 TE tensors
   (1016 total vs 512). Consistent with the established finding that these are
   inert at eval — they cost nothing and prove nothing, but note that the
   13th-ranked artifact also carried them. Do not read TE presence as the cause.

---

## 8. Reproduce

```bash
cd /Users/atulyashetty/Test/SN56-project/evidence/week6-field-depth-audit-20260806
python3 fetch_audits.py        # 14 auditing records
python3 enumerate_repos.py     # 40 HF repos
python3 harvest_artifacts.py   # 305 safetensors headers via range reads (cached; idempotent)
python3 extract.py             # artifacts.json + checkpoints.csv
python3 analyse.py             # Tables 1-4
python3 recommend.py           # Tables 5-7
```

Derived-table checksums are in `CHECKSUMS.txt`; the full console transcript is
`FULL-OUTPUT.txt`. Nothing under
`week6-tournament-dataset-harvest-20260806/` was mutated, and
`QUARANTINE-test-data/` was never opened.
