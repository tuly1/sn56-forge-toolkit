# Pipeline Materialization Audit — all five model types × the fourteen Aug-3 task shapes

**Task A2 (standing order 13: assumption inventory).** What our production pipeline at pin
`084ea914` *actually* generates and executes, per model type, on the real Aug-3 tournament
task shapes — determined by **running** the config code path and by reading the **pinned
ai-toolkit source**, not by reading our own comments.

Everything below is either **OBSERVED** (executed code / read source at the stated file:line)
or **INFERRED** (reasoned consequence, no measurement). Every claim is tagged.

---

## 0. Provenance and reproduction

| Item | Value |
|---|---|
| Branch | `claude/week6-real-fixture-experiment` @ `2fa4594d02305b23b6a3ed1145eb4a84c4ada52e` |
| Production pin audited | `084ea914c6c5cbac4fa26a2138bd7195ebd71488` — `git diff 084ea914 HEAD -- forge/` is **empty**, so the audited code IS production |
| ai-toolkit runtime pin | `99be3d96a2468d3a5228a4eb05ba67e63c586b4e` (from `ops/docker/standalone-image-toolkit-trainer.dockerfile`), extracted read-only from `/Users/atulyashetty/Test/sn56-ai-toolkit` |
| Evaluator geometry source | `/Users/atulyashetty/Test/SN56-project/week5-krea-curation-20260729/admission-envelope-v6/evaluator/image_io.py:16-40` |
| Task shapes source | `/Users/atulyashetty/Test/SN56-project/evidence/week6-tournament-dataset-harvest-20260806/tasks/<id>/task-meta.json` (read-only; `hours_to_complete`, `n_pairs`, per-image `width`/`height`) |
| Materialisation script | `/private/tmp/claude-501/-Users-atulyashetty-Test-ChildrenHospital/9ab20a3d-6342-4fc3-8fa8-f1cf73a850ec/scratchpad/materialize.py` (calls the real `forge.config.build_config`) |
| Geometry script | `.../scratchpad/geometry.py` |
| GPU used | none — CPU only, no network |

> **Line numbers for `forge/*` refer to commit `084ea914`, not to the working tree.** Another
> session is concurrently editing `forge/tasks/aitoolkit.py` on this branch (an opt-in trainer
> watchdog); nothing in this audit depends on that change, and every `forge/` claim below was
> checked against the committed pin.
>
> **`ops/experiments/` is matched by the bare `experiments/` pattern in `.gitignore`.** This
> file is therefore untracked and invisible to `git add .` — it needs `git add -f` (which is how
> the sibling files in this directory got in).

Audited source hashes (sha256):

```
7ef663da8c262037f8590130a7a71dc8d0c58b22cf52e5548f011c7b4502a9e6  forge/templates/base_diffusion_flux.yaml
86c56bbbe210317782796937f5a67c3f98cd5444c63bbff7caac0fc2d3955b5d  forge/templates/base_diffusion_ideogram4.yaml
2425632d493e6ee9a0387f92ba840553b019cbbd09182bbacc159a7469b40b93  forge/templates/base_diffusion_krea2.yaml
2b9a8590059ede7f1ac0308ff5662cf06599a5db2fa7563de35977e26d25479d  forge/templates/base_diffusion_qwen_image.yaml
f64a0954029ed3afcf656a9e2901bcf1fa5afec730a44109cd188327962c3269  forge/templates/base_diffusion_zimage.yaml
79a72c5d05ec7f5414fd549f68be7b2f04d48bdcde5abeca332316aabd83a34b  forge/recipe.py
1cb8090ae076e891041fdff84f0bf4c55587b6cc0d6a7df31fa691cea58f67da  forge/config.py
cfa89421a90476300accba2b1a33631def2eeb94a9e9abd390f0d426421ee0d5  forge/ideogram_release_policy.py
```

Assumptions baked into the materialisation (both OBSERVED from code):
* `forge.tasks.holdout.enabled_for()` reads `FORGE_HOLDOUT_SELECTION_TYPES`, which is unset in
  production → `holdout_pairs = 0`, `scoring_reserve_s = 0`, and `_recipe_hours()` reduces to
  `deadline.remaining_hard()/3600` (`forge/tasks/aitoolkit.py:62-103,151-169`). So
  `num_images == n_pairs` and `hours_to_complete` is the raw task value minus dataset-prep
  elapsed (treated as 0 below; real runs lose ~1-3 min, i.e. ~30-60 steps of budget cap).
* `spec.model_type` is the validator's `--model-type` string, matching `recipe.STEP_TABLE` keys.

---

## 1. What gets materialised — steps, cadence, wall clock, budget fit

`law` = the pure power law `clamp(base·(N/24)^p, min, max)`; `cap` = the wall-clock cap
`int((hours·3600·0.85 − 300 − 180) / SEC_PER_IT)`; `steps = min(law, cap)`
(`forge/recipe.py:61-81`). `save_every` from `forge/recipe.py:84-113`.
`epochs` counts passes over the **materialised** dataset, which is `3 × n_pairs` items
because of the resolution split (§3) — this is *not* the same as "times each image is seen",
which is `steps / n_pairs`.

`wall @policy` uses `recipe.SEC_PER_IT`; `unused` = `budget − wall − 180 s export reserve`.

### krea2 — `SEC_PER_IT = 2.2`, law `base 1200 / n_ref 24 / p 0.50 / min 100 / max 2000`

| Task | family | n | hrs | law | cap | **steps** | bound by | save_every | saves | epochs | lr | wall @policy | unused @policy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 `41025fb5` | design | 21 | 0.75 | 1122 | 824 | **824** | clock | 165 | 4+1 | 13.1 | 1e-4 | 35 min (78%) | 7 min |
| R3 `db9f7244` | design | 43 | 1.0 | 1606 | 1172 | **1172** | clock | 235 | 4+1 | 9.1 | 1e-4 | 48 min (80%) | 9 min |
| R5 `3e0fdcde` | design | 42 | 1.0 | 1587 | 1172 | **1172** | clock | 235 | 4+1 | 9.3 | 1e-4 | 48 min (80%) | 9 min |
| R5 `f6725c2b` | design | 50 | 1.0 | 1732 | 1172 | **1172** | clock | 235 | 4+1 | 7.8 | 1e-4 | 48 min (80%) | 9 min |

At the **OBSERVED** measured rate of **1.26 s/it** the same configs finish far early, and the
clock cap would not bind at all:

| hrs | cap @2.2 s/it (what we schedule) | cap @1.27 s/it (what the box can do) | actual wall @1.27 | unused @1.27 |
|---|---|---|---|---|
| 0.75 | 824 | **1429** (law 1122 would fit whole) | 22 min (50%) | **22 min** |
| 1.0 | 1172 | **2031** (law 1587-1732 would fit whole) | 30 min (50%) | **27 min** |

*(The two cells above were computed at 1.27 s/it and `MARGIN = 0.85`; the corrected rate is
1.259, which moves them to 1441 / 2049 — a 0.9 % difference that changes nothing. Retained as
the historical pre-fix picture.)*

> **REVISION (week-6 depth pass).** Closed, and with `SEC_PER_IT = 1.35` rather than the 1.5
> this document's sibling first recommended. At 1.5 the 0.75 h cap is 1336, which still
> truncates the new law's 1432 — so the fix would not have achieved its own stated goal on the
> R1 shape. At 1.35 the caps are **1484 / 2097** and the size law binds on all four krea2
> shapes: **1432 / 1825 / 1840 / 1939** (`save_every` 287 / 366 / 369 / 388, four periodic
> candidates each), against winners 1000 / 2012 / 2012 / 2012.
>
> One number above is worth restating precisely, because it is the only first-party
> measurement in this audit: **1.259 s/step is `toolkit_start → toolkit_end` ÷ 823, i.e. it
> already contains ai-toolkit's own startup.** The budget model charges `STARTUP_S = 300 s`
> separately, so net of that the same artifact implies **0.895 s/step** and 1.35 is a 51 % pad,
> not a 7 % one. The 7.2 % figure (1.35 / 1.259) is the conservative reading and is the one the
> tests assert.
>
> **CORRECTION (week-6 correctness sweep, 2026-08-06):** this used to read "1.265" and "0.90".
> The forge_run.json events are timestamps relative to `checkpoint_scope_started`
> (`toolkit_start t=4.7`, `toolkit_end t=1041.1`), so the elapsed window is **1036.4 s**, not
> 1041.1. Conservative direction, no constant moves. The stale 1041.1/823 is still baked into
> two assertions in `tests/test_week6_depth_geometry.py` — filed as an integration request.

**INFERRED:** every Aug-3 krea2 task left ~50% of its wall clock unused and ran 26-32% fewer
steps than our own step law asked for, purely because `SEC_PER_IT["krea2"] = 2.2` is wrong by
1.7×. The comment justifying 2.2 (`forge/recipe.py:50-52`, "krea2's `do_differential_guidance`
adds a second guidance forward per step → highest") describes a mechanism that **does not
execute** — see defect D3.

### ideogram4 — `SEC_PER_IT = 3.0`, law `base 140 / n_ref 24 / p 0.50 / min 48 / max 400`

| Task | family | n | hrs | law | cap | **steps** | bound by | save_every | saves | epochs | lr | wall @policy | unused @policy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R2 `84be9fcd` | style | 46 | 1.0 | 194 | 860 | **194** | law | 39 | 4+1 | 1.4 | **2.5e-5** | 15 min (24%) | **42 min** |
| R5 `1365fa1c` | product | 14 | 0.75 | 107 | 605 | **107** | law | 25 | 4+1 | 2.5 | **2.5e-5** | 10 min (23%) | **32 min** |
| R5 `b72da8c6` | style | 40 | 1.0 | 181 | 860 | **181** | law | 37 | 4+1 | 1.5 | **2.5e-5** | 14 min (23%) | **43 min** |

The lr is **not** the 1e-4 that `forge/config.py:164` sets. `forge.ideogram_release_policy.apply()`
matches and rewrites the config (OBSERVED — it fires; the materialised YAML carries the
`meta.forge_ideogram_production_policy` binding). It sets `lr/unet_lr = 2.5e-5`,
`text_encoder_lr = 1e-7`, `lr_scheduler: cosine (eta_min 2.5e-6)`, `ema_config {use_ema: true,
ema_decay: 0.995}`, `do_cfg: true`, `cfg_scale: 10.0`, `train_text_encoder: true`,
`cache_latents_to_disk: true`, `training_seed: 20260802`.

**Every ideogram4 task under-uses 75-77% of its budget.** These are the boss-round tasks.

> **REVISION (week-6 depth pass).** Closed. With `base 500 / p 0.32 / min 350 / max 620` and
> the do_cfg-corrected `SEC_PER_IT = 4.2`, the same three shapes now materialise
> **421 / 589 / 616** steps (`save_every` 85 / 118 / 124, four periodic candidates each) and
> use **83% / 82% / 85%** of their grants. The size law binds on all three — caps are
> 477 / 674 / 674 — so this is invariant to `MARGIN` and to any further `SEC_PER_IT` revision.
> FIELD-DEPTH-LAW-AUDIT §6.2.

### qwen-image — `SEC_PER_IT = 4.0`, law `base 1000 / p 0.50 / min 400 / max 3000`

| Task | family | n | hrs | law | cap | **steps** | bound by | save_every | saves | epochs | lr | wall @policy | unused @policy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R2 `7421f056` | design | 28 | 1.25 | 1080 | 836 | **836** | clock | 168 | 4+1 | 10.0 | 1e-4 | 61 min (81%) | 11 min |
| R4 `ff643470` | social | 41 | 1.5 | 1307 | 1027 | **1027** | clock | 206 | 4+1 | 8.3 | 1e-4 | 73 min (82%) | 14 min |
| R5 `4782f46f` | logo | 31 | 1.5 | 1137 | 1027 | **1027** | clock | 206 | 4+1 | 11.0 | 1e-4 | 73 min (82%) | 14 min |

> **CORRECTION (post-review): the `wall @policy` and `unused @policy` columns above are
> OPTIMISTIC for qwen, because `SEC_PER_IT["qwen-image"] = 4.0` is not the observed rate.**
> The field's reproduced measurement is **4.676 s/step** (5FW2Eaae and 5FpdSckw, identical
> configs, both configured 1150 on `7421f056` and both killed with their last save at 850;
> FIELD-DEPTH-LAW-AUDIT §5.1). Recomputed at that rate the Aug-3 pin's own qwen plans use
> **94 % / 94 % / 94 %** of their budgets, not 81-82 % — qwen was the one type that was
> already near the wall, and §4.5's "the failure mode across the board is under-use" does not
> hold for it.
>
> This is exactly what the global `MARGIN 0.85 → 0.92` walked into: it raised the qwen cap
> 1027 → 1122 and the resulting plans (909 / 957 / 1104) need 101 % of the budget on two of the
> three shapes; they are killed there, shipping their 728- and 884-step periodic saves,
> instead of the plan. The shipped policy is
> `SEC_PER_IT = 4.7` with `MARGIN_BY_TYPE["qwen-image"] = 0.98`, materialising
> **836 / 957 / 1023** — all of which complete at 4.676.

### z-image — `SEC_PER_IT = 3.0`, law `base 1100 / p 0.50 / min 400 / max 2000`

| Task | family | n | hrs | law | cap | **steps** | bound by | save_every | saves | epochs | lr | wall @policy | unused @policy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R2 `b290d171` | design | 39 | 1.0 | 1402 | 860 | **860** | clock | 173 | 4+1 | 7.4 | 1e-4 | 48 min (80%) | 9 min |
| R5 `b2582457` | social | 48 | 1.0 | 1556 | 860 | **860** | clock | 173 | 4+1 | 6.0 | 1e-4 | 48 min (80%) | 9 min |

### flux — `SEC_PER_IT = 2.5`, law `base 1100 / p 0.50 / min 500 / max 2000`

| Task | family | n | hrs | law | cap | **steps** | bound by | save_every | saves | epochs | lr | wall @policy | unused @policy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R2 `db5fefc5` | product | 15 | 0.75 | 870 | 726 | **726** | clock | 146 | 4+1 | 16.1 | 1e-4 | 35 min (78%) | 7 min |
| R3 `241cda6c` | product | 15 | 0.75 | 870 | 726 | **726** | clock | 146 | 4+1 | 16.1 | 1e-4 | 35 min (78%) | 7 min |

### Save semantics (OBSERVED, `jobs/process/BaseSDTrainProcess.py:2311,2332,2476-2489,2596-2601`)

* Loop is `for step in range(0, steps)`; `step_num = step`. A numbered save fires at the *start*
  of a step when `step_num % save_every == 0` and `step_num != 0` → the file named
  `_{K:09d}` holds the weights after exactly K optimizer steps.
* After the loop: `sample()` (if not disabled) **then** `save()` (unnumbered) — the exact final.
* Every shape above yields exactly **4 numbered candidates + 1 exact final**, as designed.
* `max_step_saves_to_keep` prunes only files matching `{job.name}_*`
  (`BaseSDTrainProcess.py:412`) — our promoted `last.safetensors` is never a pruning target.
  With 4 numbered saves, neither the value 100 (krea2/ideogram4) nor 4 (others) ever prunes.

---

## 2. Field-by-field: what the pinned runtime actually consumes

Legend: **LIVE** = read and changes behaviour · **NO-OP** = read but the value equals the
default so nothing changes · **INERT** = written, parsed, and then never reaches an executed
branch for this architecture · **DEAD-BY-GATE** = only consumed inside a branch guarded by a
flag we never set.

### 2.1 INERT / DEAD — the definitive list

| Field | Templates | Why it does nothing | Evidence |
|---|---|---|---|
| `train.do_differential_guidance: true` | krea2 | **The `if` is nested inside `if self.train_config.do_guidance_loss:`** (indent 12 under indent 8). `do_guidance_loss` defaults `False` and is set by no template. | `extensions_built_in/sd_trainer/SDTrainer.py:692` (guard, indent 8) vs `:734` (`do_differential_guidance`, indent 12) |
| `train.differential_guidance_scale: 2` | krea2 | same block | `SDTrainer.py:736` |
| `train.unet_lr` (2.5e-5) | ideogram4 (policy) | The LoRA branch calls `network.prepare_optimizer_params(text_encoder_lr=self.train_config.lr, unet_lr=self.train_config.lr, default_lr=self.train_config.lr)` — it passes **`train_config.lr`** for all three, never `unet_lr`. | `BaseSDTrainProcess.py:1836-1846` |
| `train.text_encoder_lr` (1e-7) | ideogram4 (policy) | same — the value is parsed into `TrainConfig` and then never referenced on the LoRA path. | `BaseSDTrainProcess.py:1836-1846` |
| `train.train_text_encoder: true` | ideogram4 (policy) | The TE LoRA target list is `TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention","CLIPMLP"]`. Ideogram4's TE is Qwen3-VL-8B — zero modules match, so **0 TE LoRA modules are created**. The TE is additionally `requires_grad_(False)` + `.eval()` before the network is built. It is *not* free, though — see D6. | `toolkit/lora_special.py:146,476-496`; `BaseSDTrainProcess.py:1710-1715`; `ideogram4.py:228-229` |
| `network.lokr_full_rank: true` | krea2, ideogram4 | Only applied `if self.lokr_full_rank and self.type.lower() == 'lokr'`. Our `network.type` is `lora`. | `toolkit/config_modules.py:203-208` |
| `network.lokr_factor: -1` | krea2, ideogram4 | same (lokr-only) | `config_modules.py:210` |
| `network.network_kwargs.ignore_if_contains: []` | krea2, ideogram4 | passed through as `ignore_if_contains=[]`; the constructor normalises `None → []`, so the empty list is byte-identical to the default. | `lora_special.py:188,223-225,360` |
| `save.save_format: diffusers` | **all five** | Only read inside the fine-tune / `merge_network_on_save` branch. Our LoRA branch writes `.safetensors` unconditionally. | `BaseSDTrainProcess.py:522-540` (LoRA branch, no `save_format`) vs `:642` (fine-tune branch) |
| `model.unconditional_lora_path` | ideogram4 | Loaded at model load, wrapped over **every** `nn.Linear` (`transformer_only=False`), then `is_active = False`. Its only consumer is the *inference* pipeline. With `disable_sampling: true` there is no inference. | `ideogram4.py:276-356,433-434`; sole use at `ideogram4/src/pipeline.py:381` |
| `train.bypass_guidance_embedding: false` | krea2, ideogram4 | Forwarded only if `get_noise_prediction` declares the kwarg. Krea2/Ideogram4 do not (`**kwargs` signature drops it), and only flux/flex2/kontext implement it. | `toolkit/models/base_model.py:923-931`; `ideogram4.py:475-481`; `krea2.py` |
| `train.switch_boundary_every: 1` | krea2, ideogram4 | Only read `if self.sd.is_multistage` (wan22 family). | `SDTrainer.py:2116` |
| `train.diff_output_preservation_multiplier`, `..._class: person` | krea2, ideogram4 | `diff_output_preservation` is `false`, so both are unreachable. | `SDTrainer.py:1661,2059-2079` |
| `datasets[0].mask_min_value: 0.1` | krea2, ideogram4 | Only applied when a mask tensor exists; `mask_path: null` and `alpha_mask: false`. | `dataloader_mixins.py:1415,1507` |
| `datasets[0].num_frames: 1`, `shrink_video_to_frames: true` | krea2, ideogram4 | video-only; `is_video = num_frames > 1 or auto_frame_count` → False. | `data_loader.py:396` |
| `datasets[0].controls: []` | krea2, ideogram4 | `is_generating_controls = len(controls) > 0` → False. | `data_loader.py:408` |
| `datasets[0].flip_x/flip_y: false`, `num_repeats: 1`, `network_weight: 1`, `is_reg: false` | various | all equal the parser defaults → NO-OP. | `config_modules.py:914-921,961`; `data_loader.py:447` |
| `model.compile: false`, `layer_offloading: false`, `layer_offloading_*_percent: 1` | krea2, ideogram4 | The two percents are only read inside `if layer_offloading` blocks. | `config_modules.py:682-696`; `z_image.py:185-197` (pattern) |
| `model.quantize_te: false` + `qtype_te: qfloat8` | krea2, ideogram4, flux | `qtype_te` is only used when quantising the TE. | `toolkit/util/quantize.py` call sites |
| `sqlite_db_path: ./aitk_db.db` | krea2, ideogram4 | `is_ui_trainer` requires **both** the DB file *and* `AITK_JOB_ID` in env. Forge never sets `AITK_JOB_ID`, so the whole UI-trainer path (stop watcher, `maybe_save`, step publishing) is disabled. | `DiffusionTrainer.py:19-32` |
| `save.push_to_hub: false` | krea2, ideogram4 | equals default; the push branch never runs (and must not). | `BaseSDTrainProcess.py:2606-2613` |
| `train.content_or_style: balanced` | krea2, ideogram4 | equals default. **But it is a LIVE knob we are not using** — `style` biases timestep sampling to late steps, `content` to early. 8 of 14 Aug-3 tasks are `design`, 2 are `style`. | `BaseSDTrainProcess.py:1225-1295` |
| `train.loss_type: mse` | krea2, ideogram4 | equals default. | `config_modules.py:502` |
| `train.gradient_accumulation: 1`, `batch_size: 1`, `unload_text_encoder: false`, `force_first_sample: false` | krea2, ideogram4 | all equal defaults. | `config_modules.py` |
| `network.conv: 16`, `conv_alpha: 16` | z-image | *INFERRED*: `conv_lora_dim` appends the SD-era UNet conv class names (`ResnetBlock2D`, `Downsample2D`, …) to the target list; `ZImageTransformer2DModel` contains none of them, so no conv LoRA modules are created. Not verified against the transformer's module tree (external to this repo). | `lora_special.py:193,500-502` |

### 2.2 LIVE but silently different from what the comment/name implies

| Field | Reality |
|---|---|
| `train.timestep_type: linear` (krea2, ideogram4) vs `weighted` (flux, z-image, qwen) | **The sampling schedule is identical**: `if timestep_type == 'linear' or timestep_type == 'weighted': timesteps = torch.linspace(1000, 1, n)` (`toolkit/samplers/custom_flowmatch_sampler.py:116-119`). The *only* difference is the **loss weight**: `weighted` multiplies the per-sample loss by `default_weighing_scheme[t]`, `linear` does not (`SDTrainer.py:830-850`, `custom_flowmatch_sampler.py:59-76`). Neither matches the evaluator's sampler, whose krea2 scheduler config declares `use_dynamic_shifting: True` + exponential time shift (`krea2.py:70-79`) — that branch is `timestep_type: shift`, which we never select. |
| `save.max_step_saves_to_keep: 100` vs `4` | Only prunes `{name}_*`; with 4 periodic saves neither value ever triggers. Cosmetic divergence. |
| `logging.use_ui_logger: true` (krea2, ideogram4) | Creates a SQLite `loss_log.db` **inside `save_root`**, i.e. the directory the validator uploads (`BaseSDTrainProcess.py:126`; `BaseTrainProcess.py:45`). Already mitigated: `forge/tasks/publication.py:35-37` explicitly removes `loss_log.db{,-wal,-shm}`. Not a defect — but it is load-bearing that publication keeps that list in sync. |
| `performance_log_every: 10` (krea2, ideogram4) | LIVE: prints and resets the timer table every 10 steps (`BaseSDTrainProcess.py:2560-2567`). Adds `N/M`-shaped lines to the log that `forge/tasks/aitoolkit.py:350` scans with `(\d+)\s*/\s*(\d+)`; the parser takes the **last** match, so it is currently benign, but it is an unguarded coupling. |
| `logging.log_every: 1` (krea2, ideogram4) vs absent (default 100) for the other three | LIVE: per-step tensorboard/UI commit for krea2/ideogram4 only. |
| `model.qtype: uint3\|/cache/hf_cache/qwen_image_torchao_uint3.safetensors` (qwen) | LIVE: `ModelConfig` splits on `|` into `qtype='uint3'` + `accuracy_recovery_adapter=<path>` (`config_modules.py:706-708`). |
| `model.assistant_lora_path` (z-image) | LIVE and load-bearing: merged into the transformer with weight +1.0, kept as a `-1.0` network that is activated **only during sampling** to subtract itself (`z_image.py:67-148`). Also forces `qtype qfloat8 → float8`. |
| `model.arch: zimage:turbo` (z-image) | LIVE: the `:turbo` tag is stripped, resolving to `ZImageModel.arch == "zimage"` (`config_modules.py:734-736`; `toolkit/util/get_model.py:44-51`). |
| `model.is_flux: true`, no `arch` (flux) | `ModelConfig` derives `arch='flux'`; **no registered `BaseModel` has `arch == "flux"`**, so `get_model_class` falls through to the legacy `StableDiffusion` class (`get_model.py:50-51`). That is what selects flux's bucket divisibility of 32 (§3). |
| `train.cache_text_embeddings: true` (qwen) | LIVE: caches per-caption embeddings to disk **and replaces the text encoder with a `FakeTextEncoder` whose `forward()` raises** (`SDTrainer.py:307-345`; `toolkit/unloader.py:43-76`). Real captions are still used (`batch.prompt_embeds`, `SDTrainer.py:1571-1578`). Side effect: it **hard-disables caption dropout** (`dataloader_mixins.py:386` — the dropout branch is `and not self.dataset_config.cache_text_embeddings`). |

---

## 3. Resolution / bucketing, per type — and where it diverges from the evaluator

**All five templates set `resolution: [512, 768, 1024]`. `preprocess_dataset_raw_config`
forks that into THREE independent dataset copies of the same folder** (one per resolution),
concatenated with `ConcatDataset` (`config_modules.py:1044-1062`;
`BaseSDTrainProcess.py:140-142`; `data_loader.py:669-690`). So a task with N pairs
materialises **3N samples**, and only the `resolution: 1024` third is anywhere near the size
the evaluator scores.

**`bucket_tolerance` in our config is dead.** `AiToolkitDataset.__init__` overwrites it
unconditionally with the architecture's own divisibility
(`data_loader.py:395`: `self.dataset_config.bucket_tolerance = sd.get_bucket_divisibility()`).
The value is therefore **not 64** — it is:

| type | divisibility | source |
|---|---|---|
| krea2 | **16** (`vae_scale_factor 8 × patch 2`) | `krea2.py:144-145,159-161` |
| ideogram4 | **16** (`8 × 2`) | `ideogram4.py:181-182,207-209` |
| z-image | **16** (`8 × 2`) | `z_image.py:64-65` |
| qwen-image | **32** (`16 × 2`) | `qwen_image.py:81-82` |
| flux | **32** (`2^(4-1) × 2 (is_flux) × 2`) | `stable_diffusion_model.py:283-291` (legacy class) |

Interpolation is **BICUBIC** for training (`dataloader_mixins.py:817`), followed by a centre
crop to the bucket. Bucketing caps **total pixels** at `resolution²`, preserving aspect ratio
(`toolkit/buckets.py:17-48`).

The evaluator instead scales the **long edge** to 1024 with **LANCZOS**, floors both dims to a
multiple of **16**, and centre-crops (`evaluator/image_io.py:16-40`).

### Exact per-task geometry (computed, OBSERVED)

`eval` = the evaluator's scored size. `bkt@R` = the ai-toolkit bucket for that resolution copy.
`px vs eval` = pixel-count ratio of the `@1024` copy to the evaluator's size.

| R | type | src dims | cnt | eval | bkt@512 | bkt@768 | bkt@1024 | exact match | px vs eval |
|---|---|---|---|---|---|---|---|---|---|
| 1 | krea2 | 1024x768 | 21 | 1024x768 | 576x448 | 896x656 | **1024x768** | @1024 | 1.00× |
| 2 | flux | 1195x896 | 15 | 1024x752 | 576x448 | 864x672 | 1152x896 | **none** | 1.34× |
| 2 | ideogram4 | 1408x768 | 45 | 1024x544 | 704x368 | 1024x576 | 1392x752 | **none** | 1.88× |
| 2 | ideogram4 | 768x1376 | 1 | 560x1024 | 384x672 | 576x1024 | 768x1360 | **none** | 1.82× |
| 2 | qwen-image | 1024x768 | 28 | 1024x768 | 576x448 | 864x672 | **1024x768** | @1024 | 1.00× |
| 2 | z-image | 1408x768 | 37 | 1024x544 | 704x368 | 1024x576 | 1392x752 | **none** | 1.88× |
| 2 | z-image | 768x1376 | 2 | 560x1024 | 384x672 | 576x1024 | 768x1360 | **none** | 1.82× |
| 3 | flux | 1195x896 | 15 | 1024x752 | 576x448 | 864x672 | 1152x896 | **none** | 1.34× |
| 3 | krea2 | 768x1376 | 18 | 560x1024 | 384x672 | 576x1024 | 768x1360 | **none** | 1.82× |
| 3 | krea2 | 1408x768 | 17 | 1024x544 | 704x368 | 1024x576 | 1392x752 | **none** | 1.88× |
| 3 | krea2 | 1376x768 | 8 | 1024x560 | 672x384 | 1024x576 | 1360x768 | **none** | 1.82× |
| 4 | qwen-image | 1024x768 | 41 | 1024x768 | 576x448 | 864x672 | **1024x768** | @1024 | 1.00× |
| 5 | ideogram4 | 1195x896 | 13 | 1024x752 | 576x448 | 896x656 | 1168x896 | **none** | 1.36× |
| 5 | ideogram4 | 1376x768 | 1 | 1024x560 | 672x384 | 1024x576 | 1360x768 | **none** | 1.82× |
| 5 | ideogram4 | 1024x768 | 40 | 1024x768 | 576x448 | 896x656 | **1024x768** | @1024 | 1.00× |
| 5 | krea2 | 1024x768 | 42 | 1024x768 | 576x448 | 896x656 | **1024x768** | @1024 | 1.00× |
| 5 | krea2 | 1024x768 | 50 | 1024x768 | 576x448 | 896x656 | **1024x768** | @1024 | 1.00× |
| 5 | qwen-image | 1408x768 | 31 | 1024x544 | 672x384 | 1024x576 | 1408x736 | **none** | 1.86× |
| 5 | z-image | 1024x768 | 48 | 1024x768 | 576x448 | 896x656 | **1024x768** | @1024 | 1.00× |

**Aggregate over all fourteen Aug-3 shapes: 1419 materialised training samples, of which
270 (19.0%) land at the exact geometry the evaluator scores.**

| type | on-geometry samples | share |
|---|---|---|
| flux | 0 / 90 | **0.0%** |
| ideogram4 | 40 / 300 | 13.3% |
| z-image | 48 / 261 | 18.4% |
| qwen-image | 69 / 300 | 23.0% |
| krea2 | 113 / 468 | 24.1% |

Two distinct failure modes:
1. **4:3 sources (1024×768).** The `@1024` copy is exactly right; the `@512` and `@768` copies
   are not. Two thirds of the training signal is at the wrong scale.
2. **Wide/odd sources (1408×768, 1376×768, 1195×896, 768×1376).** **No** resolution copy ever
   matches, because the evaluator clamps the *long edge* while ai-toolkit clamps *total pixels*:
   the `@1024` bucket keeps a 1392-1408 px long edge where the evaluator scores at 1024, a
   1.8-1.9× pixel-count difference. Six of the fourteen Aug-3 tasks are in this class,
   including **both** flux tasks (0% on-geometry) and the R2 ideogram4 boss-family style task.

Also OBSERVED: latent cache keys include `crop_width`/`crop_height`
(`dataloader_mixins.py:1606-1662`), so `cache_latents_to_disk: true` writes **3 separate latent
files per image** into `/dataset/images/_latent_cache/`, and the pre-training VAE encode pass
runs over 3N images. That cost is not modelled in `recipe.STARTUP_S = 300`.

---

## 4. Type-specific breakage

### 4.1 z-image and qwen-image **sample during training** (both templates omit `disable_sampling`)

`TrainConfig.disable_sampling` defaults to `False` (`config_modules.py:538`). Neither
`base_diffusion_zimage.yaml` nor `base_diffusion_qwen_image.yaml` sets it, and neither sets
`skip_first_sample` either. Consequences (all OBSERVED in the pinned source):

* `hook_before_train_loop` → `elif self.step_num <= 1 ... self.sample(self.step_num)` runs a
  **baseline sample at step 0** (`BaseSDTrainProcess.py:2255-2259`).
* In the loop, `is_sample_step = self.sample_config.sample_every and step_num % sample_every == 0`
  with `SampleConfig.sample_every` defaulting to **100** (`config_modules.py:82`), and the
  `if self.train_config.disable_sampling: is_sample_step = False` override never fires
  (`BaseSDTrainProcess.py:2333-2335`).
* After the loop, `if not self.train_config.disable_sampling: self.sample(self.step_num)` runs
  **before** the unnumbered final `self.save()` (`BaseSDTrainProcess.py:2596-2601`).

There is no `sample:` block in either template, so `sample_config.prompts == []` and zero images
are produced — but `sample()` still calls `self.sd.generate_images([])`, which
`save_device_state()`s, applies the `'generate'` device preset (moves VAE + **the whole
transformer** + TE onto the GPU), constructs a full inference pipeline
(`ZImagePipeline(...)` / `QwenImagePipeline(...)` then `.to(device)`), toggles the z-image
assistant LoRA, iterates zero items, and restores the device state
(`base_model.py:371-420,697-711`; `z_image.py:279-292`; `qwen_image.py:234-247`).

* z-image, 860 steps → **11 such events** (step 0, ×9 in-loop, terminal).
* qwen-image, 1027 steps → **13 such events**, each round-tripping a uint3-quantised 20B
  transformer under `low_vram: true`.

**INFERRED impact:** pure wasted wall clock proportional to model-move bandwidth, plus an
OOM/exception surface. The terminal one is the dangerous one: it runs **before** the exact-final
save, so if it throws, the unnumbered final checkpoint is never written and finalisation falls
back to a numbered candidate.

### 4.2 z-image and qwen-image train with **zero blank-prompt exposure**

`caption_dropout_rate: 0.05` is present in the flux, krea2 and ideogram4 templates and
**absent** from the z-image and qwen-image templates → `DatasetConfig.caption_dropout_rate`
defaults to `0.0` (`config_modules.py:919`).

The scoring metric is **0.75 × blank-prompt** + 0.25 × caption-guided reconstruction. Caption
dropout is the mechanism that puts the blank-prompt condition into the training distribution:
`get_caption` returns `''` **before** trigger injection, i.e. a genuinely empty prompt
(`dataloader_mixins.py:383-389`). z-image and qwen-image therefore never see the condition that
carries three quarters of the score.

Note the coupling for qwen: `train.cache_text_embeddings: true` propagates
`cache_text_embeddings` to every dataset copy (`BaseSDTrainProcess.py:149-151`), and the dropout
branch is explicitly `and not self.dataset_config.cache_text_embeddings` — so **adding
`caption_dropout_rate` to the qwen template alone would be silently ignored.**

### 4.3 ideogram4: the release policy is active and its levers are half-inert

The materialised ideogram4 config differs from the template on nine keys. Of those:

* `unet_lr` and `text_encoder_lr` are **INERT** (§2.1) — the effective lr is `train.lr = 2.5e-5`
  for the whole (only) LoRA param group.
* `train_text_encoder: true` creates **zero** trainable modules (§2.1) but still flips
  `grad_on_text_encoder = True` (`SDTrainer.py:1438-1440`), which (a) selects the
  gradient-enabled `encode_prompt` branch, (b) puts the 8B Qwen3-VL through
  `accelerator.prepare()` and into `modules_being_trained`
  (`BaseSDTrainProcess.py:732-739`), and (c) enables gradient checkpointing on it
  (`BaseSDTrainProcess.py:1689-1699`).
* `do_cfg: true` + `cfg_scale: 10.0` are **LIVE and expensive**: `predict_noise` concatenates
  `[uncond, cond]`, so **every training step runs the transformer at batch 2**, and the loss
  target becomes `uncond + 10.0·(cond − uncond)` with the uncond branch *not* detached
  (`base_model.py:888-950`; `SDTrainer.py:1249-1274`). A second full text-encoder forward per
  step is also added (`SDTrainer.py:1612-1625`).
* `ema_config {use_ema: true, ema_decay: 0.995}` is **LIVE and, at these step counts, dominant**:
  `ExponentialMovingAverage` is constructed with `use_num_updates=False`, so there is **no bias
  correction or warm-up**; the shadow is seeded from the LoRA at init (B = 0, i.e. zero effect),
  and `save()` always exports the EMA weights (`BaseSDTrainProcess.py:491,495-497` save;
  `:769-781` `setup_ema`; `:2031` call site; `toolkit/ema.py:43-72` `__init__`, `:100-152`
  `update`, `:336-341` `eval`). The export is therefore an **attenuated** copy of the trained
  delta:

  | task | steps | `0.995^n` (EMA memory horizon) | exported adapter as a share of the final iterate |
  |---|---|---|---|
  | R5 `1365fa1c` (14 pairs) | 107 | 58.5% | **~30%** (const-lr model 23%) |
  | R5 `b72da8c6` (40 pairs) | 181 | 40.5% | **~44%** (const-lr model 34%) |
  | R2 `84be9fcd` (46 pairs) | 194 | 37.9% | **~46%** (const-lr model 36%) |

  > **CORRECTION (week-6 correctness sweep, 2026-08-06).** The middle column used to be
  > labelled *"share of the export that is the untrained init"*, and the text above it said the
  > export retains `0.995ⁿ` of the *initial zero-effect* parameters. **Both are false.**
  > `lora_up` is zero-initialised (`toolkit/lora_special.py:122` at the pin), so the adapter's
  > effect at init is exactly zero and the EMA shadow has no untrained signal to retain.
  > `0.995ⁿ` is the EMA's **memory horizon**, nothing more. The quantity that actually matters
  > is the right-hand column: `(1−d)·Σ_t d^(T−t)·θ_t / θ_T` evaluated on the cumulative-lr path
  > with `θ₀ = 0`. **INFERRED** — it assumes Adam displacement/step ≈ lr and a stable update
  > direction, and it treats `B·A` attenuation as `B` attenuation (`A` is non-zero at init but
  > contributes no effect while `B = 0`). It is not a measurement of our adapter. The same model
  > reproduces the "EMA-weighted Σ lr" column in FIELD-DEPTH-LAW-AUDIT §6.2 exactly.

  **INFERRED:** the ideogram4 artefact we upload is roughly a 0.3-0.5-strength, lagged copy of
  the LoRA we actually trained.

  > **REVISION (week-6 depth pass) — CONFIRMED, with the mechanism verified line by line, and
  > PARTIALLY MITIGATED.** `ExponentialMovingAverage.__init__` defaults `use_num_updates=False`
  > and `setup_ema` does not override it, so `num_updates` stays `None` and the
  > `min(decay, (1+n)/(10+n))` warm-up in `update()` never runs — the decay is a flat 0.995 from
  > step 1. `shadow_params` is `[p.clone().detach() for p in parameters]` taken after
  > `setup_ema()` (`BaseSDTrainProcess.py:2031` — this citation previously read `:2101-2102`,
  > which does not contain the call at pin `99be3d96`), i.e. at LoRA init where B = 0. `save()`
  > calls `self.ema.eval()` → `store()` + `copy_to()` unconditionally, so **every** export,
  > periodic saves included, is the shadow. The attenuation is real; the "untrained init"
  > framing was not.
  >
  > It is **not** fixable for Monday: `ema_decay` lives in
  > `ideogram_release_policy._EXPECTED_RECIPE`, which feeds `POLICY_SHA256`, which
  > `PRODUCTION_ACTIVATION.policy_sha256` must equal. Changing the value invalidates the
  > owner-signed activation record, `_validated_activation` returns `None`, and the **entire**
  > production policy silently deactivates (reverting lr, TE, `do_cfg`, `cache_latents`). That
  > record is an owner-authority artefact and is not ours to re-sign.
  >
  > Mitigated instead through the one lever we own — depth. The new law
  > (`base 500 / p 0.32`) raises the exported share of the trained delta from **~44% → ~72%**
  > at N=14 (177 → 421 steps) and **~46% → ~83%** at N=46 (194 → 616 steps). *(These lines
  > previously read "untrained-init share 41% → 12%" and "17.9% → 4.6%", i.e. `0.995^T`, which
  > is the wrong quantity — see the correction above.)* FIELD-DEPTH-LAW-AUDIT §6.2.

Net: ideogram4 ran 107-194 steps at an effective 2.5e-5 cosine-decayed to 2.5e-6, exported a
~30-46%-strength EMA copy of that, and left 75% of the clock unused. For scale, the Jul-20 R1
ideogram4 field (16 miners, task `3cfa1578`, SN56-WEEK3-POSTMORTEM §6a) had a deep cluster at
722-1000+ steps; its **winner configured 378 steps** at `lr 2.5e-5` with EMA + cosine + TE, and
the 1200/1650-step `lr 4e-4` arm placed 8th
(`SN56-WEEK4-INDEPENDENT-REVIEW-2026-07-22.md` §2). *(This sentence previously said "a
1200-step bracket winner"; no cited source supports that — see the D2 correction.)* The
`max: 400` ceiling in `STEP_TABLE["ideogram4"]` meant **no ideogram4 dataset size could ever
reach the deep cluster's band**.

> **REVISION.** Superseded as of the week-6 depth pass: the law now ships **421 / 589 / 616**
> on these three shapes, using 82-85% of the grant. It still does not reach the Aug-3
> `b72da8c6` winning band (~1250) — not because of a ceiling, but because `do_cfg` halves our
> reachable depth (D6). It is, however, already above the Jul-20 R1 winner's 378. Note also that matching
> the field's step counts is not the goal for this type: they run `lr 4e-4` and we run
> `lr 2.5e-5`, so equal steps mean 28.5x less Adam path length.

### 4.4 The 14-pair shape is not degenerate

R5 `1365fa1c` (ideogram4, 14 pairs, 0.75 h): steps 107, `save_every` 25 (from
`max(25, 107//5+1)`), 4 numbered + 1 final, 42 materialised samples in 2 source aspect ratios ×
3 resolutions = 6 buckets, batch size 1 with single-item buckets padded by
`build_batch_indices` (`dataloader_mixins.py:196-207`). No degeneracy. The latent risk is
elsewhere: `size_scaled_steps` returns `max(1, min(scaled, budget_cap))` and the comment admits
"cap may push below `min`" (`recipe.py:76`) — on a slow/contended box that can silently emit a
sub-25-step run, for which `kill_safe_save_every` falls to the `s // 2` branch and only one
mid-run recovery point exists.

### 4.5 No type overruns its budget

At the policy `SEC_PER_IT`, every one of the fourteen shapes plans to finish at 23-82% of its
wall clock (§1). The failure mode across the board is **under-use**, not overrun.

> **CORRECTION (post-review).** That conclusion is only as good as `SEC_PER_IT` itself, and it
> was wrong for one type. Measured against the rate the field's own qwen artifacts imply
> (4.676 s/step, not the policy's 4.0), the Aug-3 qwen plans sit at 94 % of budget — and the
> post-recalibration MARGIN 0.92 pushed two of them past 100 %. "No type overruns" is now an
> invariant that is *asserted*, not observed: `recipe.FIELD_DEMONSTRATED_DEPTH` plus
> `test_every_shape_finishes_at_its_field_rate` check every type × every real Aug-3 shape
> against the slowest rate that type's own published artifacts support, in exact integer
> arithmetic.

---

## 5. Ranked defects — biggest expected score impact first

**D1 — Only 19% of training samples are at the geometry the evaluator scores; flux is at 0%.**
`resolution: [512, 768, 1024]` (inherited byte-identical from the upstream template) forks the
dataset into three copies, and the total-pixel bucketing rule diverges from the evaluator's
long-edge rule on every non-4:3 source. Wide sources (6/14 Aug-3 tasks) match at **no**
resolution — the `@1024` copy is 1.8-1.9× the scored pixel count. Affects all five types.
*Fix shape:* set `resolution` per task from the observed source aspect ratio so the single
materialised bucket equals `adjust_image_size(w,h)`; or drop to a single resolution entry and
choose it to hit the evaluator size. §3.

**D2 — ideogram4 is capped at ~200 steps and exports a heavily EMA-attenuated LoRA.**
`STEP_TABLE["ideogram4"] = base 140 / min 48 / max 400` comes from the discredited Jul-16
"deep training never helped" experiment; the Jul-20 R1 field's deep cluster ran 722-1000+.
On top of that the release policy's `ema_decay 0.995` with no bias correction means the export
is only a fraction of the delta we actually trained: at 107-194 steps the exported adapter is
≈30-46% of the final iterate (constant-lr model: 23-36%).

> **TWO CORRECTIONS (week-6 correctness sweep, 2026-08-06).**
> 1. This entry said "38-59% of every exported ideogram4 artefact is the untrained
>    initialisation". **That is false.** `lora_up` is zero-initialised
>    (`toolkit/lora_special.py:122` at pin `99be3d96`), so the adapter has *no* effect at init
>    and there is no untrained signal for the EMA shadow to carry. `0.995^n` is the EMA's
>    memory horizon, not an untrained share. The real defect is **attenuation of the trained
>    delta**, and it is worse than the old wording implied at shallow depth: ≈30% retained at
>    107 steps, ≈46% at 194, ≈72% at 421, ≈83% at 616 under our cosine 2.5e-5 -> 2.5e-6
>    schedule (23% / 36% / 58% / 69% on a constant-lr path). INFERRED — see §4.3 for the model.
> 2. "its bracket winner 1200" is unsupported. `SN56-WEEK3-POSTMORTEM` §6a does not say it,
>    and `SN56-WEEK4-INDEPENDENT-REVIEW-2026-07-22.md` §2 records the Jul-20 R1 ideogram4
>    **winner at 378 configured steps** (`lr 2.5e-5` + EMA + cosine + TE), with the
>    1200/1650-step `lr 4e-4` arm at 8th. FIELD-DEPTH-LAW-AUDIT §6.2.

Three of fourteen Aug-3 tasks — and R1 is drawn from `{krea2, ideogram4}`. §1, §4.3.

> **STATUS: half fixed.** The depth half is closed — the law now ships 421/589/616 (82-85% of
> the grant, up from 23-24%), which raises EMA retention from ~30-46% to ~72-83%. The EMA half is
> **open and deliberately not fixed for Monday**: `ema_decay` is inside the hash-bound
> `ideogram_release_policy` recipe projection, and changing it invalidates the owner-signed
> activation record and silently deactivates the whole policy. Recommended next, in order:
> (1) `do_cfg` on/off at matched depth — it is what caps our reachable depth at ~674;
> (2) `ema_decay 0.995 → 0.99` (the value both field ideogram4 wins used) or `use_ema: false`,
> behind a **re-signed** activation record. Neither is a same-week change.

**D3 — `SEC_PER_IT["krea2"] = 2.2` is 1.7× the measured 1.27, throwing away ~50% of the wall
clock on every krea2 task and cutting 26-32% of the requested steps.** The comment justifying
2.2 cites `do_differential_guidance` adding a second forward — that code is unreachable (D4).
At the measured rate the clock cap would not bind on any Aug-3 krea2 shape. krea2 is the single
most common type (4/14). §1. **CLOSED at `SEC_PER_IT = 1.35`** — the clock no longer binds on
any of the four real krea2 shapes.

**D3b — `SEC_PER_IT["qwen-image"] = 4.0` is 14% FASTER than the field's own reproduced
measurement (4.676 s/step), and was mislabelled "already tight — do not lower".** A completed
run gives an upper BOUND on a rate; only a killed run gives the rate. The bound (4.45) was read
as if it were the rate, and `MARGIN = 0.85` masked the error by discarding 15% of the budget.
Raising MARGIN to 0.92 for all five types removed the accidental compensation and left qwen
planning past what the field shows fits. **CLOSED at `SEC_PER_IT = 4.7` +
`MARGIN_BY_TYPE["qwen-image"] = 0.98`**, with the invariant asserted per type × per shape
rather than argued. FIELD-DEPTH-LAW-AUDIT §5.1, §6.4, §6.6.

**D4 — `do_differential_guidance` / `differential_guidance_scale` are dead: they are nested
inside `if self.train_config.do_guidance_loss:`, which we never set.** We believed krea2 was
training with differential guidance. It is not, and it never has been.
`SDTrainer.py:692` (guard) vs `:734`. §2.1.

**D5 — z-image and qwen-image have `caption_dropout_rate = 0`, so they never train on the blank
prompt that carries 0.75 of the score.** flux/krea2/ideogram4 have 0.05; the two newest
templates simply omit the key. For qwen the fix additionally requires disabling
`train.cache_text_embeddings`, which hard-disables dropout. 5 of 14 Aug-3 tasks. §4.2.

**D6 — ideogram4 pays ~2× compute per step for `do_cfg: true` and a second grad-enabled 8B TE
forward, while `unet_lr` / `text_encoder_lr` / `train_text_encoder` in the same policy block are
inert.** The policy's cost is real; a third of its stated levers are not. It also means the
`SEC_PER_IT["ideogram4"] = 3.0` cost model understates the true rate by ~2×. §4.3, §2.1.

**D7 — z-image and qwen-image run 11-13 zero-image "sampling" events per task (step 0, every
100 steps, and once more immediately before the exact-final save), because both templates omit
`disable_sampling`.** Each event constructs an inference pipeline and round-trips the whole
transformer across the PCIe bus. The terminal one precedes `save()`, so an exception there
costs us the exact final. §4.1.

**D8 — `save_format: diffusers` is written into all five templates and consumed by none.**
The LoRA save path ignores it entirely; only the fine-tune path reads it. Harmless today,
but it is a lie in five config files and in the ideogram policy's `_EXPECTED_RECIPE`. §2.1.

**D9 — `timestep_type` diverges between our types for no sampling reason.** `linear` and
`weighted` produce the *same* timesteps; they differ only in whether the per-timestep loss
weight is applied. Neither matches krea2's own declared inference schedule
(`use_dynamic_shifting` + exponential shift = `timestep_type: shift`). This is an untested,
free A/B lever. §2.2.

**D10 — ideogram4 loads a full-model-width rank-16 `unconditional_lora` at startup that can
never be used, because sampling is disabled.** It wraps every `nn.Linear` in the transformer
(`transformer_only=False`) and sits in VRAM for the whole run. §2.1.

**D11 — `network.lokr_full_rank`, `lokr_factor`, `network_kwargs.ignore_if_contains`,
`mask_min_value`, `controls`, `num_frames`, `shrink_video_to_frames`, `switch_boundary_every`,
`bypass_guidance_embedding`, `diff_output_preservation_{multiplier,class}`, `compile`,
`layer_offloading_*_percent`, `qtype_te`, `sqlite_db_path` are all written and never
consumed on our path.** Individually zero-impact; collectively they are why the templates read
as if they encode a strategy they do not. Full table in §2.1.

**D12 — `content_or_style` is a LIVE, on-metric knob pinned to `balanced` and absent from three
templates.** 8/14 Aug-3 tasks are `design` and 2 are `style`; the champion's playbook separates
style tasks explicitly. Untested. §2.1.

**D13 — `cache_latents_to_disk` writes 3× the latents (one per resolution copy) and the VAE
pre-encode pass over 3N images is not in `STARTUP_S = 300`.** Under-modelled startup makes the
budget cap slightly optimistic for flux / z-image / qwen / ideogram4-under-policy. §3.

**D15 — latent: two call sites still divide by the GLOBAL `recipe.MARGIN` after the per-type
margin change.** `forge/tasks/aitoolkit.py:164` (`_recipe_hours`, which inflates the scorer
reserve by `1/MARGIN` so the reserve survives the later `budget*MARGIN` haircut) and
`forge/tasks/holdout.py:98` (`budget_allows`, which projects the training window as
`hard_equivalent*MARGIN - reserve - boundary - STARTUP - EXPORT`) both read
`recipe.MARGIN` rather than `recipe.margin_for(model_type)`. **Inert today**: the holdout
producer is allow-listed to `{krea2, ideogram4}` (`holdout._IMPLEMENTED_TYPES`) and both sit at
the 0.92 default, so `margin_for(t) == MARGIN`; and `FORGE_HOLDOUT_SELECTION_TYPES` is unset in
production, so `_recipe_hours` reduces to `remaining_hard()/3600` regardless. **Silently wrong**
the moment a type whose margin differs from the default enters the holdout set — qwen-image is
at 0.98 — in which case `budget_allows` would under-project the training window by
`0.06 × hard_budget` (~216 s on a 1.0 h task) and `_recipe_hours` would over-reserve by
`(reserve + 45)·(1/0.92 − 1/0.98)` (~63 s at a 900 s reserve), both in the direction of
refusing or shortening a split that would in fact fit. `budget_allows` already has `model_type` in scope; the fix
is one call each. Filed as an integration request by the week-6 correctness sweep (neither file
is owned by that unit). Not triggered on Monday.

**D14 — latent risk: `size_scaled_steps` can return a sub-25-step run** when `budget_cap`
undercuts `min`; `kill_safe_save_every` then yields a single mid-run recovery point. Not
triggered by any Aug-3 shape (smallest cap was 605). §4.4.

---

## 6. Corrections to previously stated findings

> **Week-6 correctness sweep (2026-08-06)** applied four further corrections to this document,
> each marked in place: the EMA "untrained init" mechanism (D2, §4.3), the EMA source citations
> at pin `99be3d96` (§4.3), the Jul-20 R1 ideogram4 "1200-step bracket winner" provenance
> (D2, §4.3), and our own krea2 rate 1.265 → 1.259 s/step (§1). It also added **D15**. It
> changed no constant and no line of logic. The companion register is
> FIELD-DEPTH-LAW-AUDIT "Known limitations and unfixed exposure".


* The geometry note "divisibility 64" is **wrong**. `bucket_tolerance` is overwritten at dataset
  construction with `sd.get_bucket_divisibility()` — 16 for krea2/ideogram4/z-image, 32 for
  qwen-image and flux (`data_loader.py:395`). krea2/ideogram4/z-image therefore agree with the
  evaluator's multiple-of-16 rule; the divergence is entirely the **total-pixel vs long-edge**
  cap, plus BICUBIC-vs-LANCZOS. The headline "~19% of training samples land at the geometry the
  evaluator scores" is **confirmed exactly** (270 / 1419 = 19.0%).
* "Text-encoder LoRA is inert" is confirmed *at the trainer as well as the evaluator*: for
  ideogram4 the pinned `LoRASpecialNetwork` creates **zero** TE modules because its target list
  is CLIP-only. But `train_text_encoder: true` is not free — it changes the encode path and
  prepares an 8B model for training.
