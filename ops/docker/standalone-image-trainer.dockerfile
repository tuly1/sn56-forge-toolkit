# Validator-routed FLUX trainer. G.O.D deliberately selects this legacy-named
# Dockerfile for model_type=flux; the other four image architectures use the
# toolkit-named Dockerfile. A standalone FLUX cache contains one transformer
# checkpoint, which ai-toolkit's directory loader cannot consume offline. The
# Kohya base supplies the matching FLUX loader plus immutable AE/CLIP/T5 assets.

FROM diagonalge/kohya_latest:latest@sha256:d34dd5750e1018455e111f63c03bb2a4e16204607e00ba5af870dd7c71beb84e

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    FORGE_FLUX_BACKEND=kohya \
    SD_SCRIPTS_DIR=/app/sd-scripts

# Fail the build if the pinned base ever stops matching the exact runtime and
# support assets certified for this path. These are public model weights baked
# into the base image; no credential or runtime download is involved.
RUN test -f /app/sd-scripts/flux_train_network.py && \
    printf '%s  %s\n' \
      afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38 /app/flux/ae.safetensors \
      660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd /app/flux/clip_l.safetensors \
      6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635 /app/flux/t5xxl_fp16.safetensors \
      | sha256sum --check --strict && \
    python3 -c "import accelerate, lion_pytorch, PIL, safetensors, toml, torch, yaml"

WORKDIR /app
COPY forge/ /app/forge/
RUN HF_HOME=/tmp/forge-flux-tokenizer-download \
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    python3 -m forge.flux_kohya_tokenizers stage && \
    python3 -c "import shutil; shutil.rmtree('/tmp/forge-flux-tokenizer-download')" && \
    python3 -m forge.flux_kohya_tokenizers verify && \
    python3 -m forge.verify_flux_kohya_runtime

ENTRYPOINT ["dumb-init", "--", "python3", "-m", "forge.cli"]
