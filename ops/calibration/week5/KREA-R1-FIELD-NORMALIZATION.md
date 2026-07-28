# Krea2 Round-1 public-field normalization

## Scope and status

This is a read-only normalization of the 12 scored public submissions for task
`73013636-8a91-4533-8c0e-4ae449c5184d` at the captured official snapshot. The
task endpoint remained stale at `training`, but the tournament record closed
Round 1 as `completed`: 12 scored submissions among 18 assigned participants,
eight advancement positions, and our rank-11 entry explicitly eliminated.
Tournament and task snapshots are bound in the machine-readable ledger.

The machine-readable source of record is
[`krea-r1-field-ledger.json`](krea-r1-field-ledger.json). It carries the full
hotkeys, submission IDs, immutable Hugging Face revisions, artifact paths, LFS
SHA-256 values, safetensors-header metadata, and per-field
known/inferred/unknown provenance.

## Normalized field

| Rank | Miner | Loss | Planned -> submitted | Observable recipe | Submission/selection signal |
|---:|---|---:|---:|---|---|
| 1 | [5FpdSckw](https://huggingface.co/gradients-io-tournaments/tournament-tourn_0f7a1d3f6b5b66f9_20260727-73013636-8a91-4533-8c0e-4ae449c5184d-5FpdSckw/tree/031ca0f9ef6f22175209637967e483b6ae8ab0d4) | 0.04368309 | unknown (at least 1200) -> 1200 | Config scrubbed; rank approximately 32 only by artifact-size inference | `last` is byte-identical to step 1200 |
| 2 | [5C7yZ5wg](https://huggingface.co/gradients-io-tournaments/tournament-tourn_0f7a1d3f6b5b66f9_20260727-73013636-8a91-4533-8c0e-4ae449c5184d-5C7yZ5wg/tree/f4766189afc0f0ce46b52ac2991efc5f005ebbfd) | 0.04376920 | 1140 -> **960** | LR 8.6e-5; rank/alpha 32/32; AdamW8bit; MSE; guidance 2; dropout 0.1; EMA off | Holdout-selected; header proves step 960 despite selection JSON's sentinel step |
| 3 | [5EeLcV3L](https://huggingface.co/gradients-io-tournaments/tournament-tourn_0f7a1d3f6b5b66f9_20260727-73013636-8a91-4533-8c0e-4ae449c5184d-5EeLcV3L/tree/919e07cd4505cf64c13a9baef4402f2b42a6fb59) | 0.04414271 | 1432 -> 1200 | LR 1e-4; 32/32; AdamW8bit; **MAE; guidance 3**; dropout and EMA not public | No `last`; validator selects highest numbered step 1200 |
| 4 | [5GpcTKW7](https://huggingface.co/gradients-io-tournaments/tournament-tourn_0f7a1d3f6b5b66f9_20260727-73013636-8a91-4533-8c0e-4ae449c5184d-5GpcTKW7/tree/43a13b37d95f454f85b29c675dd59e5a9f7e6597) | 0.04431743 | 2000 -> 900 | LR 1e-4; 32/32; AdamW8bit; MSE; dropout 0.1; guidance not declared; EMA off | `last` is step 900; soup exists but was not scored |
| 5 | [5GKoYQm7](https://huggingface.co/gradients-io-tournaments/tournament-tourn_0f7a1d3f6b5b66f9_20260727-73013636-8a91-4533-8c0e-4ae449c5184d-5GKoYQm7/tree/71bf349eb44640289b00fc620640a1302cc3c485) | 0.04455086 | 1140 -> **840** | Automagic, base LR 8.6e-7 with min 1e-7/max 1e-3; **64/64**; MSE; guidance 2; dropout 0.3; EMA off | Holdout-selected; header proves step 840 despite selection JSON's sentinel step |
| 6 | [5FW2Eaae](https://huggingface.co/gradients-io-tournaments/tournament-tourn_0f7a1d3f6b5b66f9_20260727-73013636-8a91-4533-8c0e-4ae449c5184d-5FW2Eaae/tree/78007ee2af05ad19800c880d71db7c1f1231638b) | 0.04463835 | unknown (at least 1400) -> **1200** | Config scrubbed; rank approximately 32 only by artifact-size inference | Step 1400 exists, but `last` is byte-identical to step 1200: definite promotion, unknown metric |
| 7 | [5HKEAZxF](https://huggingface.co/gradients-io-tournaments/tournament-tourn_0f7a1d3f6b5b66f9_20260727-73013636-8a91-4533-8c0e-4ae449c5184d-5HKEAZxF/tree/7f517fce37c16167160020ddc30262c06135e12b) | 0.04476434 | 1333 -> 1200 | LR 1e-4; 32/32; AdamW8bit; MSE; guidance 2; dropout 0.05; EMA off | `last`, `last_premerge`, and step 1200 are byte-identical |
| 8 | [5Ca32LwM](https://huggingface.co/gradients-io-tournaments/tournament-tourn_0f7a1d3f6b5b66f9_20260727-73013636-8a91-4533-8c0e-4ae449c5184d-5Ca32LwM/tree/afb8add788622c7337087ccea84cc1131f1283a5) | 0.04511286 | 1100 -> 900 | LR 1e-4; 32/32; AdamW8bit; MSE; dropout 0.3; guidance not declared; EMA off | No `last`; validator selects highest numbered step 900 |
| 9 | [5FBmn1ax](https://huggingface.co/gradients-io-tournaments/tournament-tourn_0f7a1d3f6b5b66f9_20260727-73013636-8a91-4533-8c0e-4ae449c5184d-5FBmn1ax/tree/f62874af3010b82d13f13d5718b257ef3be1e97e) | 0.04546401 | 1432 -> 1100 | LR 1e-4; 32/32; AdamW8bit; MSE; guidance 2; dropout 0.05; EMA not public | No `last`; validator selects highest numbered step 1100 |
| 10 | [5D7iEJm5](https://huggingface.co/gradients-io-tournaments/tournament-tourn_0f7a1d3f6b5b66f9_20260727-73013636-8a91-4533-8c0e-4ae449c5184d-5D7iEJm5/tree/145508e8e73ec977e098d176a705331c7495d79a) | 0.04547033 | 2000 -> 1200 | LR 1e-4; 32/32; AdamW8bit; MSE; guidance 2; dropout 0.05; EMA off | No `last`; validator selects highest numbered step 1200 |
| 11 | [5HLA2QWY (ours)](https://huggingface.co/gradients-io-tournaments/tournament-tourn_0f7a1d3f6b5b66f9_20260727-73013636-8a91-4533-8c0e-4ae449c5184d-5HLA2QWY/tree/1a2da9ac72e63ff96b4c7a20ce640ef24e4234e4) | 0.04797804 | 267 -> **267** | LR 1e-4; 32/32; AdamW8bit; MSE; guidance 2; dropout 0.05; EMA off | Exact natural final; clean run; no proxy promotion |
| 12 | [5CV7cKML](https://huggingface.co/gradients-io-tournaments/tournament-tourn_0f7a1d3f6b5b66f9_20260727-73013636-8a91-4533-8c0e-4ae449c5184d-5CV7cKML/tree/a1bfc0aa605fceddd492fdc2f99ab4dc07d8fd81) | 0.04844503 | 296 -> **296** | LR 1e-4 cosine; 32/32; AdamW8bit; MSE; guidance 2; dropout 0.05; EMA on; grad accumulation 2 | Exact natural final |

## What the field does and does not prove

Every official top-10 artifact represents 840-1200 updates. The only two shallow
artifacts represent 267 and 296 updates, and they occupy ranks 11 and 12. Our
loss is 6.35% above the final rank-8 cutoff and 9.83% above rank 1. This is
strong same-task evidence that the shallow policy belongs at the front of the
discovery queue.

It is not a causal estimate of depth. The field changes several factors at once:
rank 2 uses LR 8.6e-5 and holdout selection; rank 3 uses MAE and guidance 3;
rank 5 uses rank 64, Automagic, dropout 0.3, and selection; rank 6 visibly
promotes step 1200 over a later step 1400. A controlled ladder must separate
depth from these recipe and selection effects.

Two config-scrubbed leaders remain genuinely unknown. Their approximately
rank-32 classification is an artifact-size inference, not permission to fill in
their LR, optimizer, loss, guidance, dropout, EMA, or alpha from the modal field
recipe.

## Checkpoint-selection facts

The public validator's checkpoint-discovery function at
[`f947279`](https://github.com/rayonlabs/G.O.D/blob/f9472791032ddf035ff01d3cbd8b870d48de62cd/validator/evaluation/evaluators/diffusion.py#L96-L134)
selects exact `last.safetensors` first. If `last` is absent, it falls back to the
highest numeric safetensors checkpoint, subject to its unnumbered-file branch.
Consequently:

- 5GpcTKW7's soup file was not evaluated; its step-900 `last.safetensors` was.
- 5FW2Eaae definitely submitted step 1200 even though step 1400 exists.
- The rank-2 and rank-5 final steps are 960 and 840, respectively, because the
  submitted safetensors headers say so. Their selection JSON uses a sentinel
  step of `1000000000`, which must not be mistaken for training depth.
- For repos without `last`, the table identifies the exact highest-numbered file
  the validator selected.

## Immediate planning consequence

The public families should be represented as separate controlled arms rather
than collapsed into a generic "deep" recipe:

1. A modal rank-32, LR-1e-4, MSE, guidance-2 budget-fill curve.
2. The current rank-2 family: LR 8.6e-5, rank 32, MSE, guidance 2, dropout 0.1,
   with candidate selection centered on the observed step-960 region.
3. The current rank-3 family: LR 1e-4, rank 32, MAE, guidance 3.
4. The current rank-5 family: rank 64, Automagic with its exact published
   bounds, MSE, guidance 2, dropout 0.3, and candidate selection.
5. The internal LR-2e-4 challenger, kept separate from all field-derived claims.

The existing discovery freeze currently describes the MAE/guidance-3 arm as an
"observable public rank-2 family." At this snapshot that label is wrong: it is
the rank-3 family; rank 2 is the LR-8.6e-5 selected-step-960 family. The freeze
should be corrected through its own review before GPU execution, without
silently changing either recipe.

## Evidence discipline

- Official scores and ranks come only from the
  [Gradients task API](https://api.gradients.io/auditing/tasks/73013636-8a91-4533-8c0e-4ae449c5184d).
- Config values come from immutable-revision URLs recorded in the JSON ledger.
- Submitted-step claims prefer safetensors `training_info.step`; exact LFS
  equality is used when metadata was scrubbed.
- Missing config keys remain unknown unless explicitly supported by another
  immutable source. Framework defaults are not silently treated as facts.
- No public artifact was modified or downloaded in full to produce this record;
  only repository manifests, configs, selection records, and bounded
  safetensors headers were read.
