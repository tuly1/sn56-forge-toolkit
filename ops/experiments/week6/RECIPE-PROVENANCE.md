# Week-6 krea2 head-to-head — recipe provenance

**Date:** 2026-08-06
**Branch:** `claude/week6-real-fixture-experiment`, created from the immutable
production release pin `084ea914c6c5cbac4fa26a2138bd7195ebd71488`.
**Fixture task:** `41025fb5-8473-40c6-a88d-20c0bb303edc` — krea2, dataset
`design_aetherflow_ui`, trigger `AetherFlow UI`, 21 pairs, 0.75 h budget.
Our miner `5HLA2QWY` placed 9/14, 0.97 % behind the top-8 cut. Leader: `5EACrayt`.

**Deliverables:** `leader_derived_krea2.yaml` (Arm A), `incumbent_krea2.yaml` (Arm B).

Everything in this document is labelled **OBSERVED** (I fetched it, ran it, or
read the exact source) or **INFERRED** (reasoning on top of observations). No
inference is stated as a measurement.

---

## 0. Headline: the leader's recipe was never secret

**OBSERVED.** The Aug-3 R1 leader publishes its complete training config. I
fetched it, hashed it, and it is reproduced field-by-field below.

| Item | Value |
|---|---|
| Repo | `gradients-io-tournaments/tournament-tourn_c54bb970b5d0aa91_20260803-41025fb5-8473-40c6-a88d-20c0bb303edc-5EACrayt` |
| Revision | `a28c6a0f64c06bf81e191515a1d80e04fc793b44` |
| Path | `checkpoints/config.yaml` |
| sha256 | `50fe6eec02281d0e8acf0ea7d3d3b15b3b320a1ad0b6b6d450e17930dbd5dc1c` |

**OBSERVED.** Its `checkpoints/last.safetensors` header (range request: first 8
bytes little-endian uint64 = 135384, then bytes 8..135391 = JSON) carries:

```
software      {"name": "ai-toolkit", "repo": "https://github.com/ostris/ai-toolkit", "version": "0.10.28"}
training_info {"step": 1000, "epoch": 18}
ss_tag_frequency {"1_AetherFlow UI": {"AetherFlow UI": 1}}
```
1016 tensors: 512 `diffusion_model.*` + 504 `lora_te.model.language_model.*`.

**INFERRED.** The `software` string claims stock ai-toolkit 0.10.28, but three
of the leader's settings (`timestep_type: krea2_eval_sigmas`,
`lr_scheduler: cosine_by_group`, `multires_noise_*`) do not exist in stock
ai-toolkit. The leader therefore ran a private fork that did not update the
metadata string. **We cannot observe their gating logic**, so we cannot prove
which of their settings were live *on their runtime*. We can only prove which
are live *on ours*.

---

## 1. What the whole field shipped (context)

**OBSERVED.** All 14 R1 competitors have public artifact repos; 7 published a
`config.yaml`. Fetched 2026-08-06, sha256 of each recorded.

| hotkey | steps | timestep_type | lr_sched | diff-guid scale | EMA | multires | TE | loss | revision |
|---|---|---|---|---|---|---|---|---|---|
| **5EACrayt** (R1 leader) | **1000** | **krea2_eval_sigmas** | **cosine_by_group** | **12.0** | **0.995 on** | **6 / 0.3** | **on** | (default mse) | `a28c6a0f` |
| 5FNLSgh8 | 1000 | krea2_eval_sigmas | cosine_by_group | 12.0 | 0.995 on | 6 / 0.3 | on | (default mse) | `1851de5a` |
| 5FBmn1ax | 1432 | linear | — | 3 | — | — | off | mae | `63f94211` |
| 5FjDsFGA | 1432 | linear | — | 2 | — | — | off | mse | `e6a8f46b` |
| 5D7iEJm5 | 1278 | linear | — | 2 | 0.99 on | — | off | mse | `144d0ba2` |
| 5GKoYQm7 | 972 | linear | — | 2 | 0.99 on | — | off | mse | `db33109c` |
| 5D2Qee4V | 2000 | linear | — | 2 | 0.99 on | — | off | mse | `65c4b015` |
| **5HLA2QWY (us)** | 824 planned / 823 in metadata | linear | — | 2 (inert) | off | — | off | mse | `b206d0ec` |

sha256: `5EACrayt` `50fe6eec…`, `5FNLSgh8` `b4224ee6…`, `5FBmn1ax` `eceb74ae…`,
`5FjDsFGA` `ba01d298…`, `5D7iEJm5` `7722d2f8…`, `5GKoYQm7` `f36b7625…`,
`5D2Qee4V` `56d23178…`. Our own repo publishes no config.yaml (scrubbed by the
publication scrub); its numbers come from artifact metadata + checkpoint names.

**OBSERVED.** `5FNLSgh8`'s config is byte-identical to the leader's after
normalizing its own hotkey out of the two path strings. Two of fourteen entrants
shipped this exact recipe — it is a distributed recipe, not a one-off.

**OBSERVED, universal across all 7:** `arch: krea2`, `noise_scheduler: flowmatch`,
`batch_size: 1`, `gradient_accumulation: 1`, `gradient_checkpointing: true`,
`dtype: bf16`, `save.dtype: bf16`, `save_format: diffusers`, `quantize: false`,
`low_vram: false`, `resolution: [512, 768, 1024]`, and `lr: 1e-4` (6 of 7;
`5GKoYQm7` used `automagic` at 9.267e-07). Held equal in both arms — none of
these is a differentiator.

---

## 2. Step budget: why 1000 for both arms

**OBSERVED — the leader's number.** `steps: 1000` in config; `training_info.step
== 1000` in the artifact metadata. Both, independently.

**OBSERVED — our number, recomputed from our own committed code** by importing
`forge.recipe` at pin `084ea914` with `N=21, hours=0.75, template_steps=2000`:

```
size-law target             1122      # 1200 * (21/24)**0.5, clamped to [100, 2000]
train_s                     1815.0    # 0.75*3600*0.85 - 300 (startup) - 180 (export reserve)
budget cap                   824      # int(1815.0 / 2.2)   <-- this wins
size_scaled_steps(...)       824
kill_safe_save_every(824)    165
```

**OBSERVED — that 165 is real.** Our published artifact's checkpoint ladder is
`…_000000165`, `…_000000330`, `…_000000495`, `…_000000660`, and its
`last.safetensors` metadata reads `training_info {"step": 823, "epoch": 15}`.
The policy computation above reproduces the artifact exactly.

**OBSERVED — the binding constraint was time, not depth.** The size law wanted
1122 steps; the wall-time model cut it to 824. Recomputing the same model for
1000 steps: `(1000*2.2 + 300 + 180) / 0.85 / 3600 = 0.876 h` > the 0.75 h grant.
**Under our own conservative 2.2 s/step constant our production policy can never
schedule 1000 steps for this task.** The leader did 1000 in that same grant.
At the ~1.27 s/step real throughput established earlier in Week 6, 1000 steps
needs `0.572 h` — comfortably inside 0.75 h.

**Decision: 1000 steps in both arms.** Justification:
1. It is the leader's exact OBSERVED value — sourced, not invented.
2. It is legal for our incumbent: inside the krea2 clamp `[100, 2000]`, and
   between our size-law target (1122) and our budget-capped value (824).
3. Holding it equal makes step count a controlled variable. If Arm A wins at an
   unequal budget we learn nothing about recipe shape.
4. It is achievable in the real 0.75 h envelope at measured throughput, so the
   experiment answers a question about a config we could actually ship.

**INFERRED, flagged for the GPU gate:** Arm A adds EMA updates, a 6-iteration
noise pyramid, and (unlike the incumbent) a *live* second differential-guidance
forward per step. Its s/step will be higher than Arm B's. Nobody has measured
how much. Measure both arms' s/step during the gate before drawing any
conclusion about the shippable budget.

---

## 3. Setting-by-setting provenance

Legend for **Runtime**: **INCUMBENT** = executes on
`ostris/ai-toolkit@99be3d96a2468d3a5228a4eb05ba67e63c586b4e`; **FORK-STRICT** =
executes only on `tuly1/sn56-ai-toolkit-mirror@71e133b4e73a716d1094f22355a46be07953b828`
with `sn56_strict_krea_fields: true`; **INERT** = parsed and discarded.

| Setting | Arm A (leader-derived) | Arm B (incumbent) | Source | Runtime |
|---|---|---|---|---|
| `steps` | 1000 | 1000 | leader config (A); ours pinned to match (B) | both |
| `timestep_type` | `krea2_eval_sigmas` | `linear` | leader config / our template | A: FORK-STRICT · B: both |
| `lr_scheduler` | `cosine_by_group` | absent → `constant` | leader config / our template | A: FORK-STRICT · B: both |
| `lr_scheduler_params.min_lr_by_initial_lr` | `{'0.0001': 1e-05}` | — | leader config, TE key removed (§4) | FORK-STRICT |
| `multires_noise_iterations` / `_discount` | 6 / 0.3 | absent → 0 (off) | leader config | A: FORK-STRICT |
| `ema_config.use_ema` / `ema_decay` | true / 0.995 | false / 0.99 | leader config / our template | both (EMA itself is stock) |
| `do_differential_guidance` | true | true | leader config / our template | A: FORK-STRICT · B: **INERT** (§5) |
| `differential_guidance_scale` | 12.0 | 2 | leader config / our template | A: FORK-STRICT · B: **INERT** (§5) |
| `noise_offset` | 0.0 explicit | absent → 0.0 | leader config | both (same effective value) |
| `lr` / `unet_lr` | 1e-4 / 1e-4 | 1e-4 | leader config / our template | both |
| `train_text_encoder` / `text_encoder_lr` | **false / omitted** | false | DEVIATION from leader (§4) | both |
| `optimizer` / `weight_decay` | adamw8bit / 1e-4 | adamw8bit / 1e-4 | leader + our template agree | both |
| `loss_type` | `mse` explicit | `mse` | leader omitted → default `mse` (`config_modules.py:514`) | both |
| `content_or_style` | `balanced` explicit | `balanced` | leader omitted → default `balanced` (`config_modules.py:363`) | both |
| `batch_size` / `grad_accum` / `grad_ckpt` / `dtype` | 1 / 1 / true / bf16 | same | leader + our template + all 7 field configs agree | both |
| `network.type/linear/linear_alpha` | lora / 32 / 32 | lora / 32 / 32 | leader + our template agree | both |
| `datasets.resolution` | `[512, 768, 1024]` | `[512, 768, 1024]` | leader + our template + all 7 agree | both |
| `caption_dropout_rate` | 0.05 | 0.05 | leader + our template agree | both |
| `cache_latents_to_disk` | true | false | leader config / our template | both |
| `cache_text_embeddings` | false | false | leader + our template agree | both |
| `save.save_every` | 200 | **201** | leader config (A); `kill_safe_save_every(1000, 200)` (B) | both |
| `save.max_step_saves_to_keep` | 6 | 100 | leader config / our template | both |
| `sn56_strict_krea_fields` | **true** | absent | **our own fork's opt-in switch — not a leader setting** | FORK only |
| `push_to_hub: false` | present | present | **our own choice** — lab runs must not publish | both |
| `skip_first_sample` / `force_first_sample` | true / false | true / false | **our own choice** in A (leader omitted both); template in B. Inert: `disable_sampling: true` | both |
| `disable_sampling` | true | true | leader + our template agree | both |
| `lokr_full_rank` / `lokr_factor` | absent | true / -1 | our template | B: **INERT** (LoKr-only under `type: lora`) |
| `diff_output_preservation*`, `switch_boundary_every` | **absent** | present (template values) | our template | B only — **rejected by strict schema** (§4) |

**Not sourced anywhere — declared as our own choices:** `push_to_hub: false`,
`skip_first_sample`/`force_first_sample` in Arm A, the `save_every: 201` value
in Arm B (computed by our own function, not copied), and the decision to write
ai-toolkit defaults explicitly in Arm A so the two files are textually
comparable. Nothing else in either file is invented.

---

## 4. Leader settings we deliberately do NOT execute

### 4.1 Text-encoder training — excluded by standing decision

**OBSERVED.** Leader ran `train_text_encoder: true`, `text_encoder_lr: 2.5e-07`,
and exported 504 `lora_te.model.language_model.…` tensors. Pinned ComfyUI
(`Comfy-Org/ComfyUI@091b70edda0c062fc9338a1d7e8e2f94f4c0ad0b`) does not map that
dotted prefix and G.O.D does not preprocess it, so those tensors never load at
evaluation. Excluded from both arms per the standing project decision. Neither
config trains a text encoder.

### 4.2 The forced consequence: one optimizer group, one LR floor

**OBSERVED, verified by executing the shipped validator** (`toolkit/scheduler.py`,
sha256 `ab38788258851cfb70a82af16bd1a85a0313ee9259135536af9b7e480c2d4602`):

```
unet-only optimizer @1e-4, floors {'0.0001': 1e-05}          -> (1000, [1e-05])            OK
unet-only optimizer @1e-4, leader's floors verbatim          -> ValueError:
                                                    "Unused min_lr_by_initial_lr entry: 2.5E-7"
leader's real 2-group case @[1e-4, 2.5e-7], floors verbatim  -> (1000, [1e-05, 0.0])       OK
floors without total_iters                                   -> ValueError: "Missing … total_iters"
```

Reason (OBSERVED, `toolkit/kohya_lora.py:1015`): the LoRA network appends a
text-encoder param group **only** when `text_encoder_loras` is non-empty. With
TE training off there is exactly one group, at `unet_lr`. Arm A therefore ships
`min_lr_by_initial_lr: {'0.0001': 1.0e-05}` and drops the leader's
`'0.00000025': 0.0`. Copying the leader's mapping verbatim would crash at
startup — this is a fail-closed check doing its job, not a silent difference.

### 4.3 `total_iters` must not appear in YAML

**OBSERVED.** `jobs/process/BaseSDTrainProcess.py:2183` injects
`lr_scheduler_params['total_iters'] = train_config.steps` at construction time,
and the strict validator rejects the key if it is present in the file:
`"Unknown cosine_by_group lr_scheduler_params field(s): total_iters"` (verified).
Arm A omits it; the scheduler's horizon is automatically `steps` = 1000.

### 4.4 `model_kwargs.checkpoint_filename: raw.safetensors` — not copied

**OBSERVED.** The leader set it. **OBSERVED.** Our Aug-3 production run trained
to completion without it (our published artifact carries
`training_info.step 823`). **Decision:** not copied. It is not required, and
hardcoding a filename that may not exist in our cache layout converts a lab run
into a base-model load failure. Recorded here so the omission is explicit rather
than accidental.

### 4.5 In-run checkpoint selection — OBSERVED in the field, OUT OF SCOPE here

This is the most consequential thing found while sourcing the recipe, and it is
**not** a training-config setting, so neither YAML implements it. Recording it
because it materially affects how the head-to-head should be read.

**OBSERVED.** Both `5EACrayt` and `5FNLSgh8` publish
`checkpoints/.krea2_checkpoint_evaluations.json` and `checkpoints/.krea2-best.safetensors`.
The JSON scores every periodic checkpoint plus the final one using 2 probe
images × 3 generations, reporting `text_guided_loss`, `no_text_loss` and
`combined_loss`.

**OBSERVED — the metric weighting is confirmed arithmetically.** For all 10
rows across both repos, `combined_loss == 0.25*text_guided_loss + 0.75*no_text_loss`
to full printed precision (e.g. `0.25*0.046631 + 0.75*0.030773 = 0.034737143`
vs reported `0.034737143`). This independently corroborates the
0.75 blank-prompt / 0.25 caption-guided weighting from an outside source.

**OBSERVED — selection is live, and it changes the uploaded artifact.** In
`5FNLSgh8`, `last.safetensors`, `.krea2-best.safetensors` and
`last_000000800.safetensors` share one git blob oid (`88e41406…`) and one LFS
oid (`dde2a904…`) — byte-identical. Its evaluations file scores step 800 best
(`0.032265`) versus the final step (`0.032689`). **The best checkpoint was
copied over `last.safetensors` before upload.** Consistent with this, that
repo's `last.safetensors` metadata reads `training_info.step 800`, not 1000.
In `5EACrayt` the best happened to *be* the final step, so `last.safetensors`
and `.krea2-best.safetensors` are also identical and the metadata reads 1000.

**Scope decision:** not implemented. Selection remains disabled per the standing
Lane-A boundary (the existing proxy matched exact evaluator ordering only 4/8
times). Both arms save a comparable candidate ladder (Arm A: 200/400/600/800 +
final; Arm B: 201/402/603/804 + final), so a *post-hoc* offline selection study
is possible from the artifacts without either arm doing selection in-run.

**INFERRED, not measured:** part of the leader's margin may come from selection
rather than from the recipe. This experiment cannot separate the two. Treat any
Arm-A win as "recipe + save ladder", not "recipe alone".

---

## 5. Silently-inert traps (this project has been burned by these before)

Each verified by reading the exact shipped source.

1. **`do_differential_guidance` is inert in our production runs.**
   **OBSERVED** at incumbent pin `99be3d96`: in
   `extensions_built_in/sd_trainer/SDTrainer.py` the
   `if self.train_config.do_differential_guidance:` branch is nested inside
   `if self.train_config.do_guidance_loss:`. `do_guidance_loss` is absent from
   our template and defaults to `False` (`toolkit/config_modules.py:580`).
   On the fork the predicate is
   `do_differential_guidance and (do_guidance_loss or strict_krea_fields)`
   (`toolkit/training_semantics.py:183`) — still inert without the strict flag.
   **Our shipped `differential_guidance_scale: 2` has never been applied to a
   single training step.** The five non-strict field configs carry the same flag
   and, on stock ai-toolkit, would be equally inert.
   → Arm A activates it (strict flag) at the leader's 12.0. Arm B keeps 2 and
   keeps it inert, because that is the control we actually shipped.

2. **`multires_noise_*` is inert outside strict Krea mode.**
   **OBSERVED**: `toolkit/training_semantics.py::resolve_multires_noise_config`
   returns `(0, 0.3)` — i.e. disabled — whenever `strict_krea_fields` is false,
   regardless of what the YAML says. Copying the leader's `6 / 0.3` onto a
   non-strict runtime would be a no-op.

3. **`lokr_full_rank` / `lokr_factor` in our template are LoKr-only** and do
   nothing under `network.type: lora`. Kept in Arm B for fidelity, flagged.

4. **`checkpoint_filename`** — see §4.4.

5. **The leader's 504 TE tensors** never load in pinned ComfyUI (§4.1).

**Both files are written so that no setting is present-but-discarded without an
inline comment saying so.** Arm A additionally sets
`sn56_strict_krea_fields: true`, which makes the fork reject unknown
score-critical train fields outright rather than defaulting them.

---

## 6. Runtime capability determination

Against the owned fork at `71e133b4` (working tree verified clean; files hashed):

| Leader setting | Our runtime supports it? | Evidence |
|---|---|---|
| Evaluator sigmas (`timestep_type: krea2_eval_sigmas`) | **YES**, fork-strict only | `toolkit/samplers/custom_flowmatch_sampler.py::get_krea2_eval_sigmas` derives 25 levels from ComfyUI's own code path (steps 25, denoise 0.8, simple scheduler, shift 1.15, grid 10 000); `toolkit/training_semantics.py::balanced_timestep_randint_bounds` widens the randint upper bound by 1 so all 25 levels get equal support. Allowlisted in `STRICT_KREA_TIMESTEP_TYPES`. |
| `cosine_by_group` | **YES**, fork-strict only | `toolkit/scheduler.py` lines 14–123; per-group cosine with per-initial-LR floors. Executed and verified (§4.2). |
| Multires noise | **YES**, fork-strict only | `resolve_multires_noise_config` + `pyramid_noise_like` (deterministic; separate Python-ratio and Torch-tensor RNG streams), wired at `BaseSDTrainProcess.py:1062-1066`. Bounds 0–64 iterations, discount 0–1. |
| EMA (`use_ema`, `ema_decay`) | **YES**, on both runtimes | Stock ai-toolkit feature (`toolkit/ema.py`, `BaseSDTrainProcess.setup_ema`). The strict flag only gates the *component-consistent recovery transaction* (`ema.py:358`), not EMA itself. `ema_config` also accepts `use_feedback` and `param_multiplier`; the leader used neither, so neither arm sets them. |
| Differential guidance, independently activated | **YES**, fork-strict only | `toolkit/training_semantics.py:168-185`, consumed at `SDTrainer.py:738-751`. |
| Optimizer LR groups (`unet_lr` / `text_encoder_lr`) | **YES**, fork-strict only | Fork keeps them separate; the incumbent parses then collapses to global `lr` (`toolkit/lora_contracts.py:118-131`). **Moot for these configs** — with TE training off there is only one group, so Arm A's `unet_lr` merely restates `lr`. |
| Leader's TE LoRA on Qwen3-VL | Runtime *can* train it; **we choose not to** | `toolkit/lora_contracts.py`, `extensions_built_in/diffusion_models/krea2/krea2.py:483`, plus the optional `sn56_krea_comfy_text_encoder_export` re-keying. Excluded per §4.1; neither config sets that flag. |
| In-run checkpoint selection | **NOT IMPLEMENTED** | No equivalent of `.krea2_checkpoint_evaluations.json` exists in our tree. Out of scope (§4.5). |
| `checkpoint_filename` model kwarg | Supported, **not used** | §4.4. |

**Cannot execute / not attempted — explicit list:**
- **In-run best-checkpoint selection and best→`last.safetensors` overwrite.**
  Not built; deliberately out of scope. This is the one leader behaviour we
  observe and cannot reproduce in either arm.
- **Text-encoder LoRA training.** Technically executable on the fork; excluded
  by standing decision, so the 504-tensor export and the `2.5e-07` LR group are
  absent from Arm A by design, not by limitation.
- **Anything requiring the leader's private runtime semantics.** Their fork is
  not public. Where their gating differs from ours, Arm A reproduces *their
  config* under *our verified semantics*, not their unobservable behaviour.

### Runtime/branch caveat — do not skip

Arm A **cannot be launched by the Forge on this branch.** This branch is cut
from `084ea914`, whose `forge/` has no experimental-bundle plumbing, no
`krea_runtime.py`, and no strict-Krea launcher; its `forge/config.py` builds
krea2 configs from `forge/templates/base_diffusion_krea2.yaml` only. Arm A is a
**direct ai-toolkit run** (`python run.py ops/experiments/week6/leader_derived_krea2.yaml`)
inside a container built from the owned fork at `71e133b4`. Arm B runs on either
runtime. Bringing Arm A under Forge control is the Lane-A PR #13 work and is not
part of this deliverable.

---

## 7. Verification actually performed (CPU, deterministic, reproducible)

**OBSERVED.** All of the following were executed on 2026-08-06:

1. **Arm A's train block passes the fork's strict schema.** Fed to
   `toolkit.sn56_config_validation.validate_strict_krea_train_fields`
   (sha256 `4d63735c9ab6621c4a3d4e0bf573546c73750601ac500cd35e386b2f93591271`) → PASS.
2. **Negative controls, all raised as expected:** adding
   `diff_output_preservation`, `diff_output_preservation_multiplier`,
   `diff_output_preservation_class`, or `switch_boundary_every` →
   `"Unknown score-critical Krea train field(s): …"`. Adding `total_iters` to
   `lr_scheduler_params` → `"Unknown cosine_by_group lr_scheduler_params field(s): total_iters"`.
   These are exactly the keys in our incumbent template, which is why Arm B
   cannot carry the strict flag.
3. **`cosine_by_group` group/floor binding** — four cases, §4.2.
4. **Our step law reproduces our own artifact** — §2 (824 steps → cadence 165 →
   matches the published checkpoint ladder and metadata).
5. **Field survey** — 14 repos enumerated, 7 configs fetched and hashed,
   safetensors metadata headers read via range request for 7 artifacts.
6. **Selection evidence** — blob/LFS oid identity and the
   `0.25*text + 0.75*no_text` arithmetic, §4.5.
7. **Arm B fidelity, mechanically diffed** against
   `forge/templates/base_diffusion_krea2.yaml` at pin `084ea914`. The complete
   set of differences is:

   | key | template | Arm B | why |
   |---|---|---|---|
   | `train.steps` | 2000 | 1000 | the one deliberate change (§2) |
   | `save.save_every` | 200 | 201 | `kill_safe_save_every(1000, 200)` — our own function |
   | `training_folder` | `/app/checkpoints` | task-scoped | env binding forge injects |
   | `trigger_word` | `null` | `AetherFlow UI` | env binding forge injects |
   | `datasets[0].folder_path` | `/dataset/images` | task-scoped | env binding forge injects |
   | `model.name_or_path` | `/cache/models` | cache dir | env binding forge injects |
   | `model.model_kwargs.text_encoder_path` | absent | Qwen3-VL-4B | `forge.config._KREA2_TE` injects |
   | `model.model_kwargs.vae_path` | absent | cache dir | forge injects `spec.cached_model_dir` |

   Nothing else differs. The control is the shipped template with a step count.
8. **Cross-arm audit**: all 15 held-equal train keys plus network rank/alpha,
   resolution list and caption dropout compared programmatically — zero
   unintended divergence. Exactly five settings differ (§8).

**Not performed — required before any conclusion:** every GPU gate. No CUDA
execution of evaluator sigmas, multires noise, differential guidance at 12.0, or
EMA; no s/step measurement for either arm; no exact-evaluator scoring. Nothing
here is a score claim.

### Source hashes

| File | sha256 |
|---|---|
| `sn56-forge-toolkit@084ea914 forge/templates/base_diffusion_krea2.yaml` | `2425632d493e6ee9a0387f92ba840553b019cbbd09182bbacc159a7469b40b93` |
| `sn56-forge-toolkit@084ea914 forge/recipe.py` | `79a72c5d05ec7f5414fd549f68be7b2f04d48bdcde5abeca332316aabd83a34b` |
| `sn56-ai-toolkit toolkit/sn56_config_validation.py` | `4d63735c9ab6621c4a3d4e0bf573546c73750601ac500cd35e386b2f93591271` |
| `sn56-ai-toolkit toolkit/scheduler.py` | `ab38788258851cfb70a82af16bd1a85a0313ee9259135536af9b7e480c2d4602` |
| `sn56-ai-toolkit toolkit/training_semantics.py` | `d2af97d4c0b17744303c8a3db5deb087ccf9d6c798aa0229ed46977e2b99c200` |
| `sn56-ai-toolkit toolkit/samplers/custom_flowmatch_sampler.py` | `435b30e8eb4b4daaa73cd56d17e0d39c44288a76e5fc974c9031b8ff1a95660e` |
| `sn56-ai-toolkit extensions_built_in/sd_trainer/SDTrainer.py` | `03e5146da2322320b32533c74dad22ea034358f9554c710a4f77da84a32ef2d8` |
| `sn56-ai-toolkit jobs/process/BaseSDTrainProcess.py` | `d83faaf1d32fd29155353db0a10d4c3355582e53ebd078582305f553b08629f4` |
| leader `checkpoints/config.yaml` @ `a28c6a0f` | `50fe6eec02281d0e8acf0ea7d3d3b15b3b320a1ad0b6b6d450e17930dbd5dc1c` |
| leader `checkpoints/.krea2_checkpoint_evaluations.json` @ `a28c6a0f` | `e0125a033c77388a903a61281f296e3d44b730d6a608d8852df07a3d7673a44e` |

Determinism note: neither YAML contains a seed field, because ai-toolkit's
`diffusion_trainer` process exposes none in this schema and inventing one would
violate the "do not invent settings" rule. Run-to-run variance is therefore
real and must be handled by the experiment design (repeat runs), not by the
config. Flag this to the gate owner.

---

## 8. What the head-to-head actually isolates

Held equal: steps (1000), rank/alpha (32/32), optimizer (adamw8bit, wd 1e-4),
LR (1e-4), batch/accum (1/1), dtype (bf16), resolution list, caption dropout,
`cache_text_embeddings` (false), sampling (disabled), TE training (off in both),
save-candidate count (4 periodic + final).

Varied (five things, all leader-sourced):
1. `timestep_type`: `krea2_eval_sigmas` vs `linear`
2. `lr_scheduler`: `cosine_by_group` (floor 1e-05) vs `constant`
3. `ema_config`: on @ 0.995 vs off
4. multires noise: 6 / 0.3 vs off
5. differential guidance: **live** @ 12.0 vs **inert** @ 2

Incidental, believed score-neutral: `cache_latents_to_disk` (true vs false — no
random augmentation is configured in either arm, so the cached latents equal
what the uncached path computes; **INFERRED from source, not measured**), and
`save_every` 200 vs 201. An operator wanting a stricter A/B may equalize
`cache_latents_to_disk` to `true` in both; that is the only setting I would
consider changing without weakening either arm's fidelity to its source.

**This is a five-variable bundle comparison, not an ablation.** A win tells you
the leader's bundle beats ours at equal steps; it does not tell you which of the
five did the work. Plan the ablation before spending GPU hours on the bundle, or
accept that a win produces a shippable config but no understanding.
