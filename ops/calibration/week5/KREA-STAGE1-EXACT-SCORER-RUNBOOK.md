# Krea Stage-1 exact scorer: literal operator runbook

This is the only owner-ratifiable Stage-1 scoring surface. It is offline
discovery evidence, never a production selector and never deployment authority.
All paths below are fixed. A missing byte, unexpected package, dirty checkout,
extra Comfy model path, foreign LoRA, wrong timeout, or missing delegated-agent
authorization prevents approval and execution.

## 1. Recreate the attested Krea scorer runtime

Do this while network access is still allowed. Do not reuse the Jul-24 flux R6
freeze: the owner-bound Krea environment is the 229-distribution Jul-22 runtime.

```bash
set -Eeuo pipefail
ROOT=/workspace/krea-stage1
PY="$ROOT/venv/bin/python"
LOCK=/app/forge/ops/calibration/week5/krea-stage1-exact-scorer-lock.txt
mkdir -p "$ROOT/src"
test "$(sha256sum "$LOCK" | awk '{print $1}')" = 5473a9da95cc729cac65ae0309b1044224a40eb1e8961b77cd0e39eab846bb08
test "$(wc -l < "$LOCK" | tr -d ' ')" = 229

uv python install 3.10.20
uv venv --python 3.10.20 "$ROOT/venv"
uv pip install --python "$PY" --no-deps \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r "$LOCK"
# The lock intentionally contains CUDA-12 and CUDA-13 support distributions.
# Four wheel pairs own the same runtime paths.  Reinstall every torch-cu128
# namespace owner last so concurrent installer extraction order cannot choose
# any CUDA-13 payload nondeterministically.
uv pip install --python "$PY" --no-deps --reinstall \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  nvidia-cudnn-cu12==9.10.2.21 \
  nvidia-cusparselt-cu12==0.7.1 \
  nvidia-nccl-cu12==2.27.5 \
  nvidia-nvshmem-cu12==3.3.20
uv pip check --python "$PY"

"$PY" -I - <<'PY'
import torch

assert torch.__version__ == "2.9.1+cu128"
assert torch.version.cuda == "12.8"
assert torch.backends.cudnn.version() == 91002
x = torch.randn(1, 4, 4, 32, 32, device="cuda", dtype=torch.bfloat16)
layer = torch.nn.Conv3d(4, 8, 3, padding=1, device="cuda", dtype=torch.bfloat16)
result = layer(x)
torch.cuda.synchronize()
assert tuple(result.shape) == (1, 8, 4, 32, 32)
print("SN56_KREA_SCORER_CUDNN_CONV3D=PASS")
PY
```

`uv venv` deliberately omits `--seed`: the attested 229-distribution lock does
not contain pip. `uv pip` performs the installation from outside the venv.
The explicit `unsafe-best-match` name is uv terminology, not a relaxation of
the dependency contract: uv's default first-index policy can stop at a package
name present on the PyTorch index (for example `certifi`) even when that index
does not carry the owner-pinned version. This command is safe only in this
bounded form: `--no-deps`, every registry distribution pinned with exact `==`,
the sole VCS distribution pinned to a full commit, and the complete installed
normalized name/version set plus the lock file identity verified against the
owner-bound 229-line contract before any score plan can be approved. This does
not claim wheel-byte verification. Do not reuse this index strategy for an
unpinned requirements file.

Stage exact source trees and reject local mutation:

```bash
git clone https://github.com/rayonlabs/G.O.D.git "$ROOT/src/G.O.D"
git -C "$ROOT/src/G.O.D" checkout --detach b026da04b6179cf82945e8736590dd923114342b
git clone https://github.com/comfyanonymous/ComfyUI.git "$ROOT/src/ComfyUI"
git -C "$ROOT/src/ComfyUI" checkout --detach 091b70edda0c062fc9338a1d7e8e2f94f4c0ad0b
mkdir -p "$ROOT/src/ComfyUI/custom_nodes"
git clone https://github.com/Acly/comfyui-tooling-nodes.git \
  "$ROOT/src/ComfyUI/custom_nodes/comfyui-tooling-nodes"
git -C "$ROOT/src/ComfyUI/custom_nodes/comfyui-tooling-nodes" checkout --detach \
  5d3194f4d4158ab31df7a060e1e4c56fa03f320c

test "$(git -C "$ROOT/src/G.O.D" rev-parse HEAD^{tree})" = 60d5e579aed31b69bf07d0513aace1518c974c30
test "$(git -C "$ROOT/src/ComfyUI" rev-parse HEAD^{tree})" = 1936f65713a6a6d88066b0d6127931ec50c1a2c1
test "$(git -C "$ROOT/src/ComfyUI/custom_nodes/comfyui-tooling-nodes" rev-parse HEAD^{tree})" = c7f2378076420703e933bb7619f5f1d67eb1dbeb
test -z "$(git -C "$ROOT/src/G.O.D" status --porcelain=v1 --untracked-files=all)"
test -z "$(git -C "$ROOT/src/ComfyUI" status --porcelain=v1 --untracked-files=all)"
test -z "$(git -C "$ROOT/src/ComfyUI/custom_nodes/comfyui-tooling-nodes" status --porcelain=v1 --untracked-files=all)"
```

## 2. Stage exactly three evaluator assets

Download from `Comfy-Org/Krea-2` at immutable revision
`952f49d49653cb42e7d6cf7cbfad74738073ec7d` using the owner's authorized HF
credential only during staging, then unset and remove that credential before
scoring. Use the credential-safe producer (the token is read from the
environment and is never placed in argv, stdout, or the receipt):

```bash
SCORE_MODULE='import runpy,sys;sys.path.insert(0,"/app/forge");runpy.run_module("ops.calibration.krea_score_plan",run_name="__main__")'
ASSET_RECEIPT=/campaign/controls/krea-stage1-evaluator-assets.json
read -rsp 'Hugging Face token: ' HF_TOKEN
printf '\n'
export HF_TOKEN
"$PY" -I -c "$SCORE_MODULE" stage-stage1-assets \
  --comfy-root "$ROOT/src/ComfyUI" \
  --receipt "$ASSET_RECEIPT"
unset HF_TOKEN
```

The producer downloads only the three allowlisted files at the pinned revision,
copies them create-only, hashes during the copy, checks exact size and SHA-256,
sets mode `0400`, and publishes a token-free canonical receipt. The destination
is literal:

```text
/workspace/krea-stage1/src/ComfyUI/models/diffusion_models/krea2_raw_fp8_scaled.safetensors  13141730784  48cd5d6c100297968349b41a8e77c6591d1dac18a215807f5f25f59e5c54cd61
/workspace/krea-stage1/src/ComfyUI/models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors       5242467968  54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094
/workspace/krea-stage1/src/ComfyUI/models/vae/qwen_image_vae.safetensors                      253806246  a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f
```

There must be no `extra_model_paths.yaml` and
`models/loras/` must be empty except for ComfyUI's tracked, zero-byte regular
file `put_loras_here` (one link, never a symlink). `base_name` in the evaluator config is exactly
`krea2_raw_fp8_scaled.safetensors`. The scorer starts one loopback-only Comfy
process per candidate in a transient systemd service and supplies a constructed
offline environment; it does not inherit `HF_HOME`, proxy, allocator, loader,
thread, CUDA, or Python controls from the operator shell.

```bash
test ! -e "$ROOT/src/ComfyUI/extra_model_paths.yaml"
LORA_MARKER="$ROOT/src/ComfyUI/models/loras/put_loras_here"
test -f "$LORA_MARKER" && test ! -L "$LORA_MARKER"
test "$(stat -c %s "$LORA_MARKER")" = 0
test "$(stat -c %h "$LORA_MARKER")" = 1
test "$(find "$ROOT/src/ComfyUI/models/loras" -mindepth 1 -maxdepth 1 -printf '%f\n')" = put_loras_here
test "$(stat -c %s "$ROOT/src/ComfyUI/models/diffusion_models/krea2_raw_fp8_scaled.safetensors")" = 13141730784
test "$(sha256sum "$ROOT/src/ComfyUI/models/diffusion_models/krea2_raw_fp8_scaled.safetensors" | awk '{print $1}')" = 48cd5d6c100297968349b41a8e77c6591d1dac18a215807f5f25f59e5c54cd61
test "$(stat -c %s "$ROOT/src/ComfyUI/models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors")" = 5242467968
test "$(sha256sum "$ROOT/src/ComfyUI/models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors" | awk '{print $1}')" = 54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094
test "$(stat -c %s "$ROOT/src/ComfyUI/models/vae/qwen_image_vae.safetensors")" = 253806246
test "$(sha256sum "$ROOT/src/ComfyUI/models/vae/qwen_image_vae.safetensors" | awk '{print $1}')" = a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f
```

## 3. Build and approve the D1/A and D2/A exact-score plans

All JSON controls are canonical JSON plus one newline and are create-only. The
evaluator config binds the commits, trees, three files, Python 3.10.20, the raw
distribution identity `bbcd979cae4ca3cc3e8a35c16c3d1908512bec1b8b7e9a540582122e97648bed`,
and the immutable base timeouts `startup=300`, `evaluation=3600`,
`shutdown=20`, `term_grace=20`. The exact scorer-only extension selects a
fixture-shape-bound effective evaluation ceiling: D1 (24 rows, 240 prompt
comparisons) is 5400 seconds; D2 (40 rows, 400 prompt comparisons) is 9000
seconds. `driver_python` and `comfy_python` are both `$PY`.

```bash
BATCH_MODULE='import runpy,sys;sys.path.insert(0,"/app/forge");runpy.run_module("ops.calibration.batch_evaluate_krea",run_name="__main__")'
DECISION_MODULE='import runpy,sys;sys.path.insert(0,"/app/forge");runpy.run_module("ops.calibration.krea_decision",run_name="__main__")'
DELEGATE_MODULE='import runpy,sys;sys.path.insert(0,"/app/forge");runpy.run_module("ops.calibration.krea_delegated_review_contract",run_name="__main__")'
AUTH=/campaign/controls/discovery-execution-authorization.json
SCORE_ACTOR=/campaign/controls/exact-score-plan-technical-actor.json
TRAINING_VALIDATOR_ROOT=/app/forge-5469851
ASSET_RECEIPT_FILE_SHA256=$(sha256sum "$ASSET_RECEIPT" | awk '{print $1}')

test "$(git -C "$TRAINING_VALIDATOR_ROOT" rev-parse HEAD)" = 546985195687696cf10dff3e2c58f7f0d1dd12d5
test "$(git -C "$TRAINING_VALIDATOR_ROOT" rev-parse HEAD^{tree})" = 27e43dc171f01c50cdd68331890394e32298687d
test -z "$(git -C "$TRAINING_VALIDATOR_ROOT" status --porcelain=v1 --untracked-files=all)"

"$PY" -I -c "$DELEGATE_MODULE" \
  --actor exact_score_plan_reviewer --output "$SCORE_ACTOR"
"$PY" -I -c "$SCORE_MODULE" build-stage1-evaluator \
  --comfy-root "$ROOT/src/ComfyUI" \
  --god-root "$ROOT/src/G.O.D" \
  --python "$PY" \
  --cache-provenance-sha256 "$ASSET_RECEIPT_FILE_SHA256" \
  --fixture-role D1 \
  --output /campaign/controls/D1-stage1-exact-evaluator.json

"$PY" -I -c "$SCORE_MODULE" build \
  --bundle /campaign/evidence/D1-K0-A/bundle.json \
  --bundle /campaign/evidence/D1-K1-A/bundle.json \
  --bundle /campaign/evidence/D1-K2-A/bundle.json \
  --bundle /campaign/evidence/D1-K3-A/bundle.json \
  --bundle /campaign/evidence/D1-K4-A/bundle.json \
  --bundle /campaign/evidence/D1-K5-A/bundle.json \
  --zero-manifest /campaign/controls/D1-zero-control.json \
  --dataset /campaign/controls/admission/fixture-package-v2/D1/evaluation \
  --fixture-manifest /campaign/controls/admission/fixtures/D1/fixture-manifest.json \
  --fixture-approval /campaign/controls/admission/fixtures/D1/fixture-approval.json \
  --cross-fixture-review /campaign/controls/admission/admission-envelope.json \
  --evaluator-config /campaign/controls/D1-stage1-exact-evaluator.json \
  --historical-training-validator-root "$TRAINING_VALIDATOR_ROOT" \
  --phase discovery --output-dir /campaign/scoring/D1-A

"$PY" -I -c "$SCORE_MODULE" approve \
  --draft /campaign/scoring/D1-A/score-plan.draft.json \
  --technical-actor "$SCORE_ACTOR" \
  --discovery-authorization "$AUTH" \
  --approval-output /campaign/controls/D1-A-score-approval.json \
  --plan-output /campaign/controls/D1-A-score-plan.json

"$PY" -I -c "$BATCH_MODULE" \
  --plan /campaign/controls/D1-A-score-plan.json \
  --results-dir /campaign/scoring/D1-A/results \
  --output /campaign/scoring/D1-A/aggregate.json
```

For D2/A, first build a separate evaluator config with the same command plus
`--fixture-role D2` and output it as
`/campaign/controls/D2-stage1-exact-evaluator.json`; reusing the D1 config is a
hard failure. Then repeat the literal build/approve/run sequence with
`/campaign/controls/admission/fixture-package-v2/D2/evaluation`, the D2 admitted
manifest/approval, D2-K0-A through D2-K5-A bundles, the D2 evaluator config,
and the same exact historical-validator root. The one owner-ratified
`exact_score_plan_reviewer` reviews both plans independently in its single bound
campaign review instance; an invented second actor would fail validation.
Approval recomputes live
source, dependency-lock, requirements, asset, runtime, empty-LoRA, containment,
and timeout readiness before it publishes either approval or executable plan.
The batch recomputes readiness again immediately before execution.

## 4. Agent-bound discovery decision

Prepare the canonical schema-3 policy payload binding the owner ratification,
the discovery authorization, the precommitted C1-C4 agent seal, and exactly the
D1/A and D2/A aggregates. Use separate fresh delegated-agent instances for
`discovery_decision_policy_preparer` and
`discovery_decision_policy_reviewer`.

```bash
"$PY" -I -c "$DELEGATE_MODULE" \
  --actor discovery_decision_policy_preparer \
  --output /campaign/controls/discovery-policy-preparer.json
"$PY" -I -c "$DELEGATE_MODULE" \
  --actor discovery_decision_reviewer \
  --output /campaign/controls/discovery-policy-reviewer.json

"$PY" -I -c "$DECISION_MODULE" seal-discovery-policy \
  --payload /campaign/controls/discovery-policy-payload.json \
  --output /campaign/controls/discovery-policy.json

sleep 1
POLICY_APPROVED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
"$PY" -I -c "$DECISION_MODULE" approve-discovery-policy \
  --policy /campaign/controls/discovery-policy.json \
  --technical-actor /campaign/controls/discovery-policy-reviewer.json \
  --approved-at-utc "$POLICY_APPROVED_AT_UTC" \
  --output /campaign/controls/discovery-policy-approval.json

sleep 1
DISCOVERY_DECIDED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
"$PY" -I -c "$DECISION_MODULE" decide-discovery \
  --policy /campaign/controls/discovery-policy.json \
  --approval /campaign/controls/discovery-policy-approval.json \
  --aggregate /campaign/scoring/D1-A/aggregate.json \
  --aggregate /campaign/scoring/D2-A/aggregate.json \
  --decided-at-utc "$DISCOVERY_DECIDED_AT_UTC" \
  --output /campaign/decisions/discovery.json
```

Any legacy named-human approval presented after the discovery authorization
exists is an explicit failure, not a compatibility path. The decision record
selects finalists only under the precommitted policy; it cannot authorize C1-C4
reveal, confirmation execution, production release, or deployment.
