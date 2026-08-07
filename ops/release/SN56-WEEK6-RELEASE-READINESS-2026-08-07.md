# SN56 Week-6 release readiness — image tournament Monday 2026-08-10 13:00 UTC

**Prepared:** 2026-08-07 · **Role:** integrator (four units merged, adjudicated, verified)
**Integration branch:** `claude/week6-real-fixture-experiment`
**Merged HEAD:** `6f6730faf1a16e0ee182c4656af962d4c0242ce5`
**Cut from:** `550af2c` (itself cut from production pin `084ea914c6c5cbac4fa26a2138bd7195ebd71488`)
**Test state:** **465 passed, 0 failed** from a clean `git archive` of HEAD into a temp dir (base `550af2c` = 434 passed)
**Posture:** read-only against production throughout. Hetzner untouched, no service restarted, no pin moved, no funds touched, no GPU spend, nothing pushed to any remote.

---

## 0. The one thing to read if you read nothing else

**None of the work below reaches Monday unless a new image is built and the endpoint is repointed.** Production serves repo pin `084ea914`; this branch is 4 files of `forge/` different from it. `deployment_authorized` in the release policy is deliberately still `False`. If no repoint happens, we enter Monday with the **Aug-3 recipe**, and every depth number in §1's "after" column is hypothetical. See §5.

---

## 1. Final per-type policy table

Materialised through the **real emission path** (`forge.config.build_config`) on the **real harvested Aug-3 task shapes**
(`SN56-project/evidence/week6-tournament-dataset-harvest-20260806/tasks/`), on a clean archive of the merged HEAD.

"Before" = production pin `084ea914` (what is served today). "After" = merged HEAD.
"Field rank-1" = the winner's shipped depth from `evidence/week6-field-depth-audit-20260806/analysis.json`.

| Task | Type | R | N | h | Before | **After** | Change | Field rank-1 | After ÷ field |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| `41025fb5` | krea2 | 1 | 21 | 0.75 | 824 | **1432** | +74% | 1000 | 1.43× |
| `db9f7244` | krea2 | 3 | 43 | 1.0 | 1172 | **1840** | +57% | 2012 | 0.91× |
| `3e0fdcde` | krea2 | 5 | 42 | 1.0 | 1172 | **1825** | +56% | 2012 | 0.91× |
| `f6725c2b` | krea2 | 5 | 50 | 1.0 | 1172 | **1939** | +65% | 2012 | 0.96× |
| `1365fa1c` | ideogram4 | 5 | 14 | 0.75 | 107 | **421** | +293% | 174 | 2.42× |
| `84be9fcd` | ideogram4 | 2 | 46 | 1.0 | 194 | **616** | +218% | 341 | 1.81× |
| `b72da8c6` | ideogram4 | 5 | 40 | 1.0 | 181 | **589** | +225% | 1300 | 0.45× |
| `7421f056` | qwen-image | 2 | 28 | 1.25 | 836 | **836** | 0% | 850 | 0.98× |
| `ff643470` | qwen-image | 4 | 41 | 1.5 | 1027 | **1023** | −0.4% | 1095 | 0.93× |
| `4782f46f` | qwen-image | 5 | 31 | 1.5 | 1027 | **957** | −7% | 949 | 1.01× |
| `b290d171` | z-image | 2 | 39 | 1.0 | 860 | **1186** | +38% | 1188 | 1.00× |
| `b2582457` | z-image | 5 | 48 | 1.0 | 860 | **1315** | +53% | 1317 | 1.00× |
| `241cda6c` | flux (snapshot → ai-toolkit) | 3 | 15 | 0.75 | 726 | **870** | +20% | 870 views | 1.00× |
| `db5fefc5` | flux (standalone → **kohya**) | 2 | 15 | 0.75 | 94 steps / 705 views | **94 steps / 705 views** | 0% | 750 views | 0.94× |

**Ideogram4 is unchanged by this integration** — 421/589/616 were already set at `550af2c`. The merge deliberately did **not** move them (see §2, IR-1).

**Read the flux rows carefully.** The audit's flux "depth" column is `epochs × N × repeats` = **sample views**, not optimiser steps, and it is labelled `INFERRED` in the source data. `db5fefc5` routes to the kohya backend (single-file base), where our 94 optimiser steps at `train_batch_size 4 × gradient_accumulation_steps 2` are **705 image views** — i.e. 0.94× the winner, not 0.11×. This correction came out of the dropped flux unit and is the most valuable thing that unit produced; §3 explains why its *policy* was dropped anyway.

**Also emitted and confirmed on the merged tree** (real emission path, per type):

| Type | steps | `ema_config` | `caption_dropout_rate` | `cache_text_embeddings` |
|---|---:|---|---|---|
| ideogram4 | 421 | `use_ema: True, ema_decay: 0.99` ← **changed** | 0.05 | False |
| krea2 | 1432 | `use_ema: False` | 0.05 | False |
| qwen-image | 836 | absent | absent | True |
| z-image | 1186 | absent | absent | absent |
| flux | 870 | absent | 0.05 | absent |

---

## 2. Every change merged, with its evidence basis

| # | Change | Evidence (one line) |
|---|---|---|
| M1 | **ideogram4 `ema_decay` 0.995 → 0.99** (`forge/ideogram_release_policy.py`) | `save()` exports the EMA shadow unconditionally at ai-toolkit pin `99be3d96` (`BaseSDTrainProcess.py:491,495-497`), the warm-up ramp is unreachable (`use_num_updates` never plumbed through `setup_ema` at `:769-781`), and `lora_up` is zero-init (`lora_special.py:122`), so the export is the trained delta **attenuated**; under our cosine 2.5e-5→2.5e-6 that is f=0.720→**0.894** at 421 steps (×1.24), and ~×1.7 at every salvageable early checkpoint. |
| M2 | The decay is carried as a **named, individually hashed amendment** (`WEEK6_EMA_AMENDMENT`, `amendment_sha256`) rather than an in-place edit, and the activation record is re-signed and **scoped** to it | The recipe is now honestly "the I-J20-D2 port **plus one named amendment**", not the port; `covers_recipe_projection_exactly: False`. I verified on the merged tree that the shipped `PRODUCTION_ACTIVATION` still **validates** (`_validated_activation(...) is not None` → True) — a botched hash would have silently deactivated the policy and shipped the Week-4 template. |
| M3 | **0.99 is field-backed and was already our own template value** | Both Aug-3 rank-1 ideogram4 artifacts (`1365fa1c`, `84be9fcd`, hotkey `5FBmn1ax`) ran `use_ema: true, ema_decay: 0.99`; `forge/templates/base_diffusion_ideogram4.yaml:61` already carried 0.99 and the release policy was overwriting it with 0.995. |
| M4 | **No caption dropout added to z-image or qwen-image** (a deliberate NON-change; comments only in both templates) | 0/2 z-image and 0/3 qwen Aug-3 rank-1 configs ran *effective* dropout; the qwen R2 winner `7421f056` shipped `0.05` **with** `cache_text_embeddings: true`, i.e. the exact no-op (gate at `dataloader_mixins.py:387`). Both templates remain **semantically identical** to the production pin. |
| M5 | **qwen `cache_text_embeddings` stays on** | Un-inerting dropout costs `steps − 3N` extra Qwen2.5-VL-7B forwards against 46 s / 67 s of planner slack = ~61 / 74 ms each, on the tightest type in the table, **never timed on our own host**. |
| M6 | **Fourteen corrections to the week-6 record** (`forge/recipe.py`, two audit docs, test docstrings) | AST-verified comments/docstrings only — I re-parsed both `.py` files at `550af2c` and `48a2cd4`, stripped docstrings, and confirmed `ast.dump` identity. Notable: the retired "41% untrained at 177 steps" figure was literally `0.995**T` multiplying a **zero-initialised** matrix; the real quantity is the lag integral. |
| M7 | **IR-1 discharged: the ideogram4 depth tripwire** (`tests/test_week6_ideogram_depth.py`) | See below — this was the reviewer's single named blocking risk. |
| M8 | **Two merged-test defects repaired** (`tests/test_blank_prompt_training.py`) | See below. |
| M9 | **A pre-existing 1.7% test flake fixed** (`tests/test_publication.py`) | See below. |

### M7 — IR-1, handled as a discharge and not a disarm

`tests/test_week6_ideogram_depth.py` pinned `ema_decay == 0.995` and went red the moment M1 landed. It was written to force the depth law to be **re-derived rather than silently inherited**, and its reviewer's stated worst case was that someone would "fix" it by bumping one constant, defeating a tripwire placed on purpose. What I actually did:

- `EMA_DECAY` is now **genuinely read out of the policy**. The old line claimed to do that *in a comment* while hardcoding 0.995 — which is precisely how the two drifted apart.
- The expected value is pinned **separately** (0.99) **and bound to the hashed amendment record**, so the decay cannot move again unaudited.
- **The shipped depths do not move.** 421/589/616 stand. They are set by the size law and the do_cfg clock ceiling; the EMA term is a *floor* those depths clear, not the driver. Cutting ideogram4 depth in response to the new decay would have been the error, and a separate test pins the depths exactly.
- **The floor now uses the quantity that actually binds.** `EMA_DECAY ** steps <= 0.15` was a monotone proxy for a misnamed quantity, and at 0.99 it is **toothless**: it binds only at T ≥ 189 against a shallowest shipped depth of 421. Replaced with the attenuation model f = s_T/B_T, floor 0.85, which binds at **T ≥ 338** and **fails the retired two-point law's 177 steps at N=14 (f = 0.664)**.
- The model is anchored to the constant-lr closed form at the two figures derived independently three times this week (0.338689 @177, 0.584607 @421). I reproduced all of them to the digit before asserting on them.

**Mutation-verified, all RED:** reverting the policy decay; cutting the ideogram4 size law; weakening the attenuation model; editing only the pinned constant.

### M8 — two defects the blank-prompt reviewer found, both real

- `test_shipped_caption_dropout_is_never_inert` read only `train.cache_text_embeddings`. The gate at `dataloader_mixins.py:387` reads the **dataset's** flag and `BaseSDTrainProcess.py:148-151` only propagates train→dataset **when train is True** — it never clears a dataset-level flag. A dataset-level flag therefore made dropout inert while the test passed. Now the OR of both levels. **The reviewer's exact reproduction (dataset-level cache + dropout on z-image) now fails.**
- `test_qwen_budget_cannot_absorb_a_live_text_encoder` counted extra encoder forwards as `steps − N`. `resolution: [512, 768, 1024]` forks the dataset into **3 copies** (`config_modules.py:1050-1062`), so the cached setup pass covers 3N. The correction moves the per-forward budget **up** (~61/74 ms vs 56.7/68.1) — it works *against* the rejection, and the rejection still holds. The failure message no longer reads as authorisation to disable caching when the real cause is a planner recalibration.

### M9 — a pre-existing flake, not merge breakage

The first full clean-archive run failed `test_publication.py::test_telemetry_public_projection_is_strict_and_hash_bound`. I did **not** assume it was mine. On the **untouched parent `550af2c`** it fails **5 times in 300 runs (1.7%)**: the forbidden-token scan ran over the whole public projection including `private_record_sha256`, and the 3-digit token `b"367"` (the private `steps` value) lands inside that 64-char hex by chance. The digest is deliberately public and is separately verified against the private bytes two assertions earlier. The scan now masks the digest; schema, kind and events are still scanned in full. **0 failures in 400 runs after the fix**, and a mutation that leaks the private event payload into the public projection is still caught.

---

## 3. Every change DROPPED, and why

### D1 — the entire flux-kohya depth unit (`claude/week6-flux-kohya-depth`, `6883dec`) — **DROPPED**

Its reviewer said DO-NOT-SHIP. I did not take that on faith; I reproduced the decisive facts myself before dropping.

1. **The headline measurement is wrong.** The policy block states the throughput run was "H100 PCIe, **N=15**". The run's own private record says otherwise — `meta.pairs: 10`, `dedup {'kept': 10}`, `dataset_ready {'pairs': 10}`
   (`evidence/hyperstack-flux-kohya-operational-20260724/monochrome-pool-kohya-0p5h-69b77f0-r5/private-recorder/252c895f4d1572f97fa3e20b/forge_run.full.f090a2ee….json`).
2. **The unit's own function contradicts its own constant.** I re-derived the GPU trace from `nvidia-smi.csv` and reproduced the four checkpoint plateaus exactly (434.1–458.1 / 802.2–824.2 / 1177.3–1203.3 / 1551.5–1571.5 s) → 13.76/14.12/13.93 = **13.94 s/optimiser step**. At N=10, `views_per_optimizer_step(10, 4, 2)` returns **5.0** — I executed it. So the true rate is **13.94 / 5.0 = 2.79 s/view**. The unit divided by 8 and shipped **1.737**, then planned at **1.95**, which is **1.43× faster than our own measurement**.
3. **The plan does not fit.** Simulating the real `db5fefc5` shape (N=15, 0.75 h, window 2475 s, measured startup 96 s, measured save 26 s):

   | s/view | NEW (116 steps) | OLD (94 steps) | Δ views |
   |---:|---|---|---:|
   | 1.713 (field host) | 870 complete | 705 complete | +165 |
   | **2.788 (our host, measured)** | **750, KILLED@100** | **705 complete** | +45 |
   | 3.00 | 750, KILLED@100 | 705 complete | +45 |
   | **3.05** | **562, KILLED@75** | **705 complete** | **−143** |
   | 3.50 | 562, KILLED@75 | 562, KILLED@75 | 0 |

   The regression cliff opens **~9% above our own measured rate**. Best case is +6% depth on **one of fourteen** tasks, achieved by relying on the kill path instead of completing; the downside is −20% depth. That is a bad trade on tournament eve.
4. **The one-constant remediation buys nothing.** At the corrected rate + the unit's own 12% pad (3.14 s/view), the affordable depth is **91 steps — below the incumbent's 94.** On our hardware at 0.75 h there is no flux depth gap to close, so the unit's headline ("the real gap was 0.81×, and it is now closed exactly") is wrong in both directions.
5. **The incumbent is safe.** With the unit dropped, `db5fefc5` finishes at 87.5% of its window at our measured rate, still finishes at 3.10 s/view (96.4%), and even at 3.50 s/view salvages step 75. No forfeit at any rate tested.

**What survives the drop:** the finding that `FIELD-DEPTH-LAW-AUDIT` §2.1/§6.5 converts kohya epochs as `epochs × N × repeats`, which silently assumes batch 1. That conversion yields sample **views**, not optimiser steps. I verified it against the source data (`depth_prov: "INFERRED(kohya epoch 58 x N x1)"`) and recorded it in §1. **It is a documentation defect that is still unfixed** — see §4, X4.

### D2 — caption dropout for z-image and qwen-image — **not shipped** (this was the unit's own verdict, and it is correct)

The 75%-blank-prompt metric makes dropout look free, and it is exactly the kind of "obvious" edit that would have shipped un-measured. 1 of 9 ai-toolkit winner configs in the entire Aug-3 field ran an effective dropout. Not shipping was the right call, and the reasoning is now pinned by tests instead of living in a report.

### D3 — options (a), (c) and (d) on the EMA defect — **dead at the pin**

`use_ema: false` (rejected on field evidence), `use_num_updates` via config (**not plumbed through `setup_ema`; `EMAConfig` would drop the key without error** — a silently inert "fix", this project's signature failure mode), and "export true weights, keep EMA" (`:495` is gated only on `self.ema is not None`). Only `ema_decay` was actually reachable.

---

## 4. Known unfixed exposure going into Monday — ranked by expected cost

| # | Exposure | Why it costs, in one line |
|---|---|---|
| **X1** | **The merged work is not deployed.** Production serves `084ea914`; this is an integration branch and `deployment_authorized` is `False`. | If there is no image build + endpoint repoint, we run the **Aug-3 recipe** and every "after" number in §1 is fiction. **Highest expected cost by a wide margin.** |
| **X2** | **qwen-image is the tightest type in the table and is UNMEASURED on our own host** — 94.0% / 94.6% of budget projected, and 4 of 6 Aug-3 qwen artifacts were deadline-killed. | `SEC_PER_IT=4.7` was read off field runs at LoRA rank 128 with `do_cfg` (batch 2) while our config is rank 32 batch 1; if the true rate is worse we salvage ~60% of the plan on 3 of 14 tasks. |
| **X3** | **The ideogram4 EMA payoff is a MODEL, not a measurement.** f = s_T/B_T accounts for EMA's attenuation but has no term for its variance reduction. | We are amplifying an adapter by ×1.24 without ever having verified on a GPU that its *direction* is right; if it is not, the amendment is neutral-to-slightly-negative on 3 of 14 tasks. |
| **X4** | **`FIELD-DEPTH-LAW-AUDIT` §2.1/§6.5 still converts kohya epochs as `epochs × N × repeats`.** | Anyone reading that section will still conclude our flux path is "9× too shallow" and may repeat the dropped unit's mistake. The file is unowned by any surviving unit; I did not edit it. |
| **X5** | **z-image `SEC_PER_IT = 1.8` is the least-verified constant in the table** — never run on our host, padded 15% over a field *bound* (not a rate). | Both z-image shapes project at 67–74% of budget, so there is real slack; a bad constant costs depth, not a forfeit. |
| **X6** | **ideogram4 depth (421/589/616) is derived, not validated.** Our lr integral is ~12× below the field's, and the Jul-20 R1 ideogram4 **winner** configured 378 steps — below all three. | If the field is right about depth and we are wrong, we lose the 3 ideogram4 tasks on a decision no measurement supports. |
| **X7** | **`recipe.MARGIN` is read by `aitoolkit.py:164` and `holdout.py:98` and is currently inert** (holdout allow-list is `{krea2, ideogram4}`, both 0.92; env unset). | A future per-type margin change would take effect in one of those paths and not the other. Zero cost Monday; filed, not patched, deliberately. |
| **X8** | **`b72da8c6` ships 589 against a rank-1 of 1300 (0.45×)** — the largest single-task depth gap remaining. | Both Aug-3 arms on that task were deep; if depth is what won it, this is the one task where our law is most likely simply wrong. |

---

## 5. Exact remaining release steps, in order

Nothing below has been done. Steps 1 and 4–7 are **OWNER ACTIONS** and are called out as such.

| # | When | Step | Who |
|---|---|---|---|
| 1 | **Now** | **Decide whether to deploy at all.** If the answer is no, stop here — §1's "after" column does not apply and X1 is realised. This is a judgement call about shipping an un-GPU-validated recipe change on tournament eve, and it is not mine to make. | **OWNER** |
| 2 | On a deploy decision | Re-run the full suite from a clean archive of the release SHA and record the count. Current: **465 passed, 0 failed** at `6f6730f`. | engineer |
| 3 | On a deploy decision | Build the image from the release SHA and verify the ai-toolkit pin assertion in `ops/docker/standalone-image-trainer.dockerfile:82` still hard-fails on anything but `99be3d96a2468d3a5228a4eb05ba67e63c586b4e`. **Do not** substitute the ai-toolkit fork; it is not in the image and cannot be used Monday. | engineer |
| 4 | On a deploy decision | **Issue the release certificate** and flip `deployment_authorized`. It is `False` today by design; the re-signed `PRODUCTION_ACTIVATION` carries `release_authorized: True` but **explicitly not deployment**. | **OWNER** |
| 5 | After 4 | **Endpoint repoint** on Hetzner: served pin `084ea914` → the release SHA, then re-run `sn56-preentry-probe-v2.sh` and confirm `endpoint.pin` shows the new SHA exactly once and the prior pin is absent. | **OWNER** |
| 6 | **Sunday evening**, before Monday | **Watcher re-arm for Aug-10.** The installed `sn56-week6-watcher.service` is armed for `20260803` and will not capture Aug-10; it is also burning a large request budget on a stale retry queue (1,232 `task_fetch_failed` in 24 h), which risks rate-limiting during the live window. Retire it **first**, then install `sn56-watcher-20260810.service` — never both at once. Procedure is in `SN56-WEEK6-MONDAY-READINESS-2026-08-06.md` §4. | **OWNER** |
| 7 | **Mon 12:15 UTC** | **Pre-entry probe.** Run **both** `sn56-preentry-probe-v2.sh` and the legacy `sn56-preentry-probe.sh`. Finish any intervention by **12:30 UTC**, then freeze. Required greens: `entry.balance` (0.2 TAO credited — per the current brief this is now FUNDED; the Aug-6 probe still showed 0 rao, so **re-verify**), `watcher.window` armed for `2026-08-10T13:00:00Z`, `endpoint.pin`, `miner.metagraph_fresh`, `upstream.baseline` at `b026da04`. | **OWNER** |
| 8 | Mon 13:00–13:59 UTC | Tournament window. R1 is a single task drawn from {krea2, ideogram4} at the tightest budget (0.75 h). Both lanes verified to land inside it (§6). | — |

**Never schedule from the local G.O.D checkout.** It says the image tournament starts at 15:00 UTC. It starts at **13:00** (`validator/tournament/constants.py` at `b026da04`). This was last week's near-miss and it is still live; the probe now enforces it.

---

## 6. End-to-end verification of the merged result

Real emission path, real Aug-3 shapes, clean archive of `6f6730f`. Wall clock walked at each type's **field-observed slowest** rate against the real terminate gate (`budget − EXPORT_RESERVE_S − STOP_MARGIN_S`), charging the full `STARTUP_S = 300`.

| Task | Type | R | N | h | steps | save@ | 1st save | proj. wall | % budget | @ field-slowest | salvage | forfeit? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| `41025fb5` | krea2 | 1 | 21 | 0.75 | 1432 | 287 | 687 s | 2233 s | 82.7% | KILLED@1148 | 1148 | **no** |
| `db5fefc5` | flux (kohya) | 2 | 15 | 0.75 | 94 | 25 | 618 s | 2165 s | 87.5% | FINISH | 94 | **no** |
| `84be9fcd` | ideogram4 | 2 | 46 | 1.0 | 616 | 124 | 821 s | 2887 s | 80.2% | KILLED@496 | 496 | **no** |
| `7421f056` | qwen-image | 2 | 28 | 1.25 | 836 | 168 | 1090 s | 4229 s | 94.0% | KILLED@504 | 504 | **no** |
| `b290d171` | z-image | 2 | 39 | 1.0 | 1186 | 238 | 728 s | 2435 s | 67.6% | FINISH | 1186 | **no** |
| `241cda6c` | flux (ai-toolkit) | 3 | 15 | 0.75 | 870 | 175 | 650 s | 2040 s | 75.6% | FINISH | 870 | **no** |
| `db9f7244` | krea2 | 3 | 43 | 1.0 | 1840 | 369 | 798 s | 2784 s | 77.3% | FINISH | 1840 | **no** |
| `ff643470` | qwen-image | 4 | 41 | 1.5 | 1023 | 205 | 1264 s | 5108 s | 94.6% | KILLED@615 | 615 | **no** |
| `1365fa1c` | ideogram4 | 5 | 14 | 0.75 | 421 | 85 | 657 s | 2068 s | 76.6% | FINISH | 421 | **no** |
| `b72da8c6` | ideogram4 | 5 | 40 | 1.0 | 589 | 118 | 796 s | 2774 s | 77.0% | FINISH | 589 | **no** |
| `3e0fdcde` | krea2 | 5 | 42 | 1.0 | 1825 | 366 | 794 s | 2764 s | 76.8% | FINISH | 1825 | **no** |
| `f6725c2b` | krea2 | 5 | 50 | 1.0 | 1939 | 388 | 824 s | 2918 s | 81.0% | FINISH | 1939 | **no** |
| `4782f46f` | qwen-image | 5 | 31 | 1.5 | 957 | 192 | 1202 s | 4798 s | 88.9% | KILLED@576 | 576 | **no** |
| `b2582457` | z-image | 5 | 48 | 1.0 | 1315 | 264 | 775 s | 2667 s | 74.1% | FINISH | 1315 | **no** |

Field-slowest rates used (s/step, except flux in s/view): krea2 **1.519** (two operators each completing 1432 on the R1 shape), ideogram4 **4.992** (our own runtime-kill threshold; the field bound is 2.05 on non-`do_cfg` configs), z-image **1.560**, qwen-image **6.964** (the slowest reproduced qwen rate), flux **2.788 s/view** (our own corrected r5 measurement — deliberately harsher than the field's 1.713).

**NO FORFEIT PATH: CONFIRMED.** On all 14 shapes the first periodic save lands before the terminate gate at the stress rate, so a deadline kill always has a valid periodic checkpoint to promote. Five shapes are killed at their stress rate and all five salvage a substantial artifact (1148/1432, 496/616, 504/836, 615/1023, 576/957). The tightest first-save is `ff643470` at 1264 s against a 5175 s gate — 4.1× of margin.

---

## 7. Blunt honest assessment

### Are we materially more competitive than the Aug-3 entry?

**On depth, yes, and it is not close — but depth is a hypothesis, not a result.**

The Aug-3 entry was eliminated in R1. Against the R1 shape (`41025fb5`, krea2, N=21, 0.75 h) we now ship **1432 steps against 824** — and the rank-1 artifact on that task shipped 1000. We went from 0.82× the winner to 1.43×. Ideogram4 went from 107–194 steps to 421–616, i.e. from 0.3–0.6× the field to 1.8–2.4×. z-image now reproduces both rank-1 depths to within 2 steps. Those are real, and the Aug-3 depths were plainly, quantifiably too shallow.

**But be clear about what that claim rests on.** Not one of these depths has been validated on a GPU against the actual scoring metric. We calibrated to *what winners shipped*, which is a proxy for *what scores well* only if the field is roughly right and if depth transfers across recipes — and our recipe differs from the field's in the one dimension that most affects how much depth you need: our ideogram4 lr integral is **~12× below** theirs. A step count matched across a 16× lr gap matches nothing. The ideogram4 numbers in particular are an inference stacked on an inference.

**The EMA fix is real but modest.** ×1.24 on the exported delta at the R1 ideogram4 shape, ~×1.7 on early salvageable checkpoints. It is a genuine defect genuinely fixed, and it is *not* the fifth inert setting — I confirmed the shipped activation still validates and that the emitted config carries 0.99. It also moves us from ~11% to ~14% of the winner's exported parameter movement on that shape. That framing matters more than the multiplier: **we are still an order of magnitude behind on lr integral, and the amendment does not touch that.**

**The most competitive thing that happened this week may be the non-changes.** Four units produced work; one shipped a behaviour change of a single float, one shipped nothing but comments and tests, one shipped only documentation corrections, and one was dropped outright. Three of the four "obvious" improvements available this week — caption dropout for the 75%-blank metric, deeper flux, disabling the qwen text-embedding cache — were all rejected on evidence, and each would have been shipped un-measured by a less disciplined process. The flux unit in particular would have taken a task that currently **completes** at 87.5% of its window and made it a knife-edge that overruns at our own measured throughput.

### Where are we still likely to lose?

1. **We may not be running any of this.** X1 is not a technical risk, it is the whole bet. Nothing here matters without a build and a repoint.
2. **qwen-image, on the clock.** Three tasks at 89–95% of budget, on the only type we have never timed ourselves, where 4 of 6 field artifacts were deadline-killed. We will probably ship truncated artifacts on at least one qwen task. We won't forfeit — but a 60%-depth artifact against a completed one loses.
3. **ideogram4, on the lr integral.** If depth is not the binding constraint on that type and lr is, we have optimised the wrong variable by a factor of 12 and made three tasks marginally worse in exchange for wall clock. `b72da8c6` (589 vs a rank-1 of 1300) is where this shows up first.
4. **R1 is still a coin flip on a single task.** One task, 0.75 h, drawn from {krea2, ideogram4}. Our krea2 R1 lane is the strongest thing we have: 1.43× the winner's depth, projecting at 90.2% of the terminate gate at the modelled rate, and salvaging 1148 of 1432 steps even at the field's tightest observed krea2 rate. Our ideogram4 R1 lane is the *least* validated. **We are meaningfully better on the krea2 draw than on the ideogram4 draw, and we don't get to choose.**
5. **Everything is still calibration, not measurement.** Every rate constant in the table except krea2's is padded inference. The single highest-value thing available before Aug-17 is GPU time on our own host for qwen and z-image — not another recipe idea.

**Summary:** materially deeper, better instrumented, and honest about its own error bars in a way the Aug-3 entry was not. Still fundamentally an un-validated recipe entering a tournament, whose largest single risk is that it does not get deployed at all.

---

## 8. Provenance

- Merged HEAD `6f6730faf1a16e0ee182c4656af962d4c0242ce5` on `claude/week6-real-fixture-experiment`. Not pushed to any remote.
- Merged units: `claude/week6-correctness-sweep` (`48a2cd4`), `claude/week6-unitc-blank-prompt` (`b3b089a`), `claude/week6-unitA-ideogram-ema` (`5b124a0`). Dropped: `claude/week6-flux-kohya-depth` (`6883dec`) — branch retained, not merged.
- **No file was touched by two units.** Union of merged paths is disjoint; all three merges were clean with zero conflicts.
- Ideogram policy identity: `POLICY_ID = week6-ideogram-exact-final-ema-horizon-v1`, `POLICY_SHA256 = fcf9ad8a…68138f91`, `AMENDMENT_SHA256 = 88442e05…debfdb4`; shipped activation validates; `release_authorized: True`, `deployment_authorized: False`.
- **Uncommitted, unrelated:** `ops/experiments/week6/` carries pre-existing working-tree changes to the two-arm fixture experiment (arms JSON, `build_fixtures.py`, `run_two_arm.py`, the two krea2 YAMLs) from a prior session. They are **not part of any unit and are NOT committed**: they set the arms to 750 steps while `tests/test_week6_two_arm.py` still pins 1000, and with them applied that file goes **38 failed / 12 passed**. Committing them as-is would break the release. Preserved in `git stash@{0}` **and** left in the working tree. **Owner/branch author: these need reconciling with their tests before they land.**
