# Krea additive host and six-profile binding

This runbook records the additive runtime surface that must be covered by a
fresh schema-2 owner ratification and its derived discovery authorization. It
does **not** supplement an older ratification by implication. None of the
artifacts below independently authorizes GPU execution; the final schema-4
approval remains separate.

## Why the profile index is separate

The immutable discovery freeze must exist before the timing probes. The
profiles do not exist until those probes finish, so inserting profile hashes
back into that freeze would create a hash cycle and make the probe and final
plan bind different documents.

`krea_runtime_binding.py` instead creates a post-timing index with six cells:

```text
D1 × {A-rank32-adamw8bit-mse-guidance2,
      B-rank32-adamw8bit-mae-guidance3,
      C-rank64-automagic-mse-guidance2}
D2 × {A-rank32-adamw8bit-mse-guidance2,
      B-rank32-adamw8bit-mae-guidance3,
      C-rank64-automagic-mse-guidance2}
```

Each cell opens and validates its profile, then compares the execution
envelope's training-pair count and training-dataset-shape SHA-256 to the exact
approved fixture. A D1 profile cannot fill a D2 cell. The final-plan wrapper
calls the ratified full plan validator and additionally requires its exact
fixture/profile to occupy the corresponding index cell.

## Host layout

Prepare a canonical payload for `seal-layout-spec`. It has these top-level
keys:

```json
{
  "schema": 1,
  "kind": "forge-krea-host-bootstrap-spec",
  "sources": {
    "forge_repo": "/srv/sn56-forge",
    "ai_toolkit_repo": "/srv/ai-toolkit",
    "venv": "/srv/forge-venv",
    "checkpoints": "/ephemeral/checkpoints",
    "dataset": "/ephemeral/dataset",
    "cache": "/ephemeral/cache",
    "campaign": "/mnt/sn56-evidence/campaign",
    "evidence_root": "/mnt/sn56-evidence"
  },
  "source_identities": {
    "forge_commit": "FULL_40_CHARACTER_COMMIT",
    "ai_toolkit_commit": "FULL_40_CHARACTER_COMMIT"
  },
  "requirements": {
    "ubuntu_release": "22.04",
    "minimum_effective_cpu_capacity": 16,
    "minimum_effective_memory_bytes": 68719476736,
    "minimum_checkpoint_filesystem_bytes": 536870912000,
    "minimum_checkpoint_free_bytes": 375809638400,
    "minimum_evidence_filesystem_bytes": 214748364800,
    "minimum_evidence_free_bytes": 107374182400,
    "minimum_gpu_memory_mib": 78000,
    "maximum_gpu_memory_mib": 85000,
    "minimum_cuda_version": "12.8",
    "required_docker_runtime": "nvidia",
    "systemd_pid1_required": true,
    "unified_cgroup_v2_required": true,
    "rootful_docker_required": true,
    "separate_evidence_filesystem_required": true
  },
  "runtime": {
    "execution_surface": "staged_host_venv",
    "container_image_reference": "sha256:FULL_64_CHARACTER_LOCAL_IMAGE_ID",
    "container_image_sha256": "FULL_64_CHARACTER_DIGEST",
    "ai_toolkit_dir": "/app/ai-toolkit",
    "jit_enabled": true,
    "stage1_runtime_receipt": {
      "path": "/mnt/sn56-evidence/campaign/controls/stage1-runtime-receipt.json",
      "file_sha256": "FULL_64_CHARACTER_FILE_DIGEST",
      "receipt_sha256": "FULL_64_CHARACTER_SEMANTIC_DIGEST"
    },
    "runtime_cache_policy": {
      "root": "/cache/krea-runtime",
      "namespace_derivation": "timing_plan_file_sha256_plus_capture_id_or_execution_plan_file_sha256",
      "initial_state": "root-empty-before-bootstrap",
      "cross_capture_or_plan_reuse": false,
      "within_process_reuse": true
    }
  },
  "gpu_execution_authorized": false
}
```

The example byte thresholds are explicit policy inputs, not hidden defaults.
Review them against provider-reported binary capacities before sealing.
The 200 GiB evidence floor is binary (214,748,364,800 bytes); a provider's
nominal 200 GB decimal volume does not qualify. Do not create the venv by hand.
The tracked Stage-1 materializer is the sole supported producer. It reproduces
the Dockerfile dependency phase order, checks the frozen source commits and
inputs, runs the essential import and CUDA/Inductor probes, records every
command, removes its transient install cache, and publishes a complete-tree,
create-only receipt.

Use clean physical staging paths first; `/app/*` does not exist yet and is not
an input to materialization. The receipt binds these exact physical paths.

```bash
sudo install -d -m 0700 /mnt/sn56-evidence/campaign/controls
STAGE1_MODULE='import runpy,sys; sys.path.insert(0,"/srv/sn56-forge"); runpy.run_module("ops.calibration.krea_stage1_runtime",run_name="__main__")'

sudo /usr/bin/python3 -I -c "$STAGE1_MODULE" dry-run \
  --forge-repo /srv/sn56-forge \
  --ai-toolkit-repo /srv/ai-toolkit \
  --destination /srv/forge-venv \
  --receipt /mnt/sn56-evidence/campaign/controls/stage1-runtime-receipt.json

sudo /usr/bin/python3 -I -c "$STAGE1_MODULE" materialize \
  --forge-repo /srv/sn56-forge \
  --ai-toolkit-repo /srv/ai-toolkit \
  --destination /srv/forge-venv \
  --receipt /mnt/sn56-evidence/campaign/controls/stage1-runtime-receipt.json

sudo /usr/bin/python3 -I -c "$STAGE1_MODULE" validate-receipt \
  --receipt /mnt/sn56-evidence/campaign/controls/stage1-runtime-receipt.json
```

Only after validation, copy the receipt's literal `receipt_sha256` and the
file's `sha256sum` into the layout payload above, then seal and prepare it.
Bootstrap recaptures the receipt against the physical sources, proves the
complete venv tree, and only then mounts the same sources read-only at
`/app/forge`, `/app/ai-toolkit`, and `/app/venv`. Run preparation and later
commands in one root shell so root-owned mode-0700 controls remain accessible.

```bash
SYSTEM_MODULE='import runpy,sys; sys.path.insert(0,"/srv/sn56-forge"); runpy.run_module("ops.calibration.krea_host_bootstrap",run_name="__main__")'

/usr/bin/python3 -I -c "$SYSTEM_MODULE" seal-layout-spec \
  --payload /mnt/sn56-evidence/campaign/controls/layout.payload.json \
  --output /mnt/sn56-evidence/campaign/controls/layout.spec.json

sudo /usr/bin/python3 -I -c "$SYSTEM_MODULE" prepare-layout \
  --spec /mnt/sn56-evidence/campaign/controls/layout.spec.json \
  --output /campaign/controls/layout.receipt.json

sudo /usr/bin/python3 -I -c "$SYSTEM_MODULE" verify-layout \
  --receipt /campaign/controls/layout.receipt.json
```

The bootstrap mounts these fixed targets:

- read-only: `/app/forge`, `/app/ai-toolkit`, `/app/venv`, `/dataset`;
- read-write: `/app/checkpoints`, `/cache`, `/campaign`.

It fails before timing if the host is not root on Ubuntu 22.04 with systemd PID
1, unified cgroup v2, one non-MIG H100 in the 78,000–85,000 MiB range, CUDA
compatibility at least 12.8, rootful Docker with the NVIDIA runtime/toolkit, no
GPU compute process, at least 16 effective CPUs and 64 GiB effective RAM, at
least 500 GiB checkpoint storage, or a separately mounted evidence filesystem.
It also binds clean Forge/ai-toolkit Git commits and exact calibration-tool
hashes, including the Stage-1 materializer itself. The reference-image gate
runs a network-disabled, read-only in-container CUDA `torch.compile` smoke;
there is no fictional image-environment-variable assertion. Bootstrap performs
no install or download.

`/cache/krea-runtime` must be empty at bootstrap. The runner creates a mode-0700
cold namespace for each timing capture from the sealed plan-file hash plus the
validated capture ID (`timing-a`, `timing-b`, `timing-c`, `heldout-e2e`). Final
execution is keyed by its plan-file hash. Reusing a capture/plan namespace fails
closed; cache reuse is allowed only within that one runner process.

## Reviewed live preflight policy and host manifest

The policy payload contains every threshold accepted by
`krea_host_identity.build_manifest` except `storage_probe_tool_sha256`; the
additive builder computes that value from the ratified host-identity module.
No performance threshold is defaulted. The foreign-process allowance must be
zero.

```bash
RUNTIME_MODULE='import runpy,sys; sys.path.insert(0,"/app/forge"); runpy.run_module("ops.calibration.krea_runtime_binding",run_name="__main__")'

/app/venv/bin/python -I -c "$RUNTIME_MODULE" \
  seal-preflight-policy \
  --payload /campaign/controls/preflight-policy.payload.json \
  --output /campaign/controls/preflight-policy.json

/app/venv/bin/python -I -c "$RUNTIME_MODULE" \
  build-host-manifest \
  --checkpoint-path /app/checkpoints \
  --preflight-policy /campaign/controls/preflight-policy.json \
  --bootstrap-receipt /campaign/controls/layout.receipt.json \
  --output /campaign/controls/host-execution-manifest.json

/app/venv/bin/python -I -c "$RUNTIME_MODULE" \
  verify-host-live \
  --manifest /campaign/controls/host-execution-manifest.json \
  --checkpoint-path /app/checkpoints \
  --output /campaign/evidence/preflight-observation.json
```

The ratified `verify_live` remains the final idle-host and storage-throughput
gate. It checks load, effective memory, checkpoint free space, GPU utilization,
free GPU memory, foreign processes, bounded write/read throughput, and fsync
latency against the reviewed policy.

## Post-timing profile index and execution plan

Prepare one canonical profile-index payload:

```json
{
  "discovery_plan": "/app/forge/ops/calibration/week5/krea-discovery-plan.json",
  "discovery_execution_authorization": "/campaign/controls/discovery-execution-authorization.json",
  "fixtures": {
    "D1": {"manifest": "/campaign/controls/admission/fixtures/D1/fixture-manifest.json", "approval": "/campaign/controls/admission/fixtures/D1/fixture-approval.json"},
    "D2": {"manifest": "/campaign/controls/admission/fixtures/D2/fixture-manifest.json", "approval": "/campaign/controls/admission/fixtures/D2/fixture-approval.json"}
  },
  "profiles": {
    "D1": {"A-rank32-adamw8bit-mse-guidance2": "/campaign/evidence/D1-A.profile.json", "B-rank32-adamw8bit-mae-guidance3": "/campaign/evidence/D1-B.profile.json", "C-rank64-automagic-mse-guidance2": "/campaign/evidence/D1-C.profile.json"},
    "D2": {"A-rank32-adamw8bit-mse-guidance2": "/campaign/evidence/D2-A.profile.json", "B-rank32-adamw8bit-mae-guidance3": "/campaign/evidence/D2-B.profile.json", "C-rank64-automagic-mse-guidance2": "/campaign/evidence/D2-C.profile.json"}
  }
}
```

```bash
/app/venv/bin/python -I -c "$RUNTIME_MODULE" \
  seal-profile-index --payload /campaign/controls/profile-index.payload.json \
  --output /campaign/controls/profile-index.json

/app/venv/bin/python -I -c "$RUNTIME_MODULE" \
  seal-execution-plan --payload /campaign/controls/D1-K1.payload.json \
  --profile-index /campaign/controls/profile-index.json \
  --output /campaign/controls/D1-K1.plan.json
```

All outputs are canonical JSON, mode `0600`, create-only with `O_EXCL` and
`O_NOFOLLOW`, and fsynced. They preserve the ratified execution-plan and host
manifest schemas and never write the production trainer or endpoint.
