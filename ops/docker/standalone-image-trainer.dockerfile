# Validator-routed FLUX trainer. G.O.D deliberately selects this legacy-named
# Dockerfile for model_type=flux. Its downloader emits one of two cache shapes:
# an exact-one-root-file standalone checkpoint, or a full snapshot directory.
# Keep both pinned runtime graphs in one image and select only from cache shape.

FROM diagonalge/ai-toolkit:latest@sha256:c24f8bb95bf1dc8da7cd6158a763f2c9782783ad7648dc4047c5757ef3447db8 AS aitoolkit-runtime

COPY ops/docker/image-runtime-lock.txt /opt/sn56/image-runtime-lock.txt
COPY ops/docker/image-runtime-phase1-constraints.txt /opt/sn56/image-runtime-phase1-constraints.txt
COPY ops/docker/verify_image_runtime.py /opt/sn56/verify-image-runtime.py
RUN python3 /opt/sn56/verify-image-runtime.py \
        --lock /opt/sn56/image-runtime-lock.txt \
        --constraints /opt/sn56/image-runtime-phase1-constraints.txt \
        --files-only

WORKDIR /app/ai-toolkit
# Reproduce the same pinned two-phase runtime used by the toolkit-named image.
RUN git fetch origin 99be3d96a2468d3a5228a4eb05ba67e63c586b4e && \
    git checkout 99be3d96a2468d3a5228a4eb05ba67e63c586b4e && \
    pip install --no-cache-dir \
        --constraint /opt/sn56/image-runtime-phase1-constraints.txt \
        --requirement requirements.txt && \
    pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
        --constraint /opt/sn56/image-runtime-phase1-constraints.txt \
        torchcodec==0.2.1 pyyaml Pillow numpy safetensors

RUN python3 -m pip install --no-cache-dir --no-deps \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        --requirement /opt/sn56/image-runtime-lock.txt && \
    python3 /opt/sn56/verify-image-runtime.py \
        --lock /opt/sn56/image-runtime-lock.txt \
        --constraints /opt/sn56/image-runtime-phase1-constraints.txt && \
    test "$(git rev-parse HEAD)" = 99be3d96a2468d3a5228a4eb05ba67e63c586b4e


FROM diagonalge/kohya_latest:latest@sha256:d34dd5750e1018455e111f63c03bb2a4e16204607e00ba5af870dd7c71beb84e

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    FORGE_FLUX_BACKEND=kohya \
    AI_TOOLKIT_DIR=/app/ai-toolkit \
    FORGE_TEMPLATES_DIR=/app/forge/templates \
    FORGE_KOHYA_PYTHONPATH=/home/.local/lib/python3.10/site-packages \
    FORGE_KOHYA_LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64 \
    FORGE_KOHYA_LD_PRELOAD=libtcmalloc.so \
    FORGE_KOHYA_PROTOBUF_IMPLEMENTATION=python \
    FORGE_KOHYA_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:/home//.local/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    SD_SCRIPTS_DIR=/app/sd-scripts \
    PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=/opt/sn56/ai-toolkit-python \
    LD_LIBRARY_PATH= \
    LD_PRELOAD= \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb

# The final process uses the frozen ai-toolkit package graph by default. The
# Kohya base's graph remains intact under /home/.local and is substituted only
# in the standalone child process; the two incompatible Torch stacks never
# share one interpreter.
COPY --from=aitoolkit-runtime /app/ai-toolkit/ /app/ai-toolkit/
COPY --from=aitoolkit-runtime /usr/local/lib/python3.10/dist-packages/ /opt/sn56/ai-toolkit-python/
COPY --from=aitoolkit-runtime /opt/sn56/image-runtime-lock.txt /opt/sn56/image-runtime-lock.txt
COPY --from=aitoolkit-runtime /opt/sn56/image-runtime-phase1-constraints.txt /opt/sn56/image-runtime-phase1-constraints.txt
COPY --from=aitoolkit-runtime /opt/sn56/verify-image-runtime.py /opt/sn56/verify-image-runtime.py

WORKDIR /app
COPY forge/ /app/forge/

# Prove the copied ai-toolkit graph still has the certified metadata and entry
# surface when loaded by the final image's ABI-compatible Python 3.10 runtime.
RUN test -f /app/ai-toolkit/run.py && \
    LD_PRELOAD= \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb \
    PYTHONPATH=/opt/sn56/ai-toolkit-python \
    python3 /opt/sn56/verify-image-runtime.py \
        --lock /opt/sn56/image-runtime-lock.txt \
        --constraints /opt/sn56/image-runtime-phase1-constraints.txt && \
    cd /app/ai-toolkit && \
    LD_PRELOAD= \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb \
    PYTHONPATH=/opt/sn56/ai-toolkit-python \
    python3 -c "import os, toolkit, torch; assert torch.__version__ == '2.6.0+cu124'; assert torch.version.cuda == '12.4'; assert os.path.realpath(torch.__file__).startswith('/opt/sn56/ai-toolkit-python/')"

# Separately prove the original Kohya source, dependency graph, support assets,
# config parser, and checkpoint naming contract. These public weights are baked
# into the pinned Kohya base; no credential or runtime download is involved.
RUN test -f /app/sd-scripts/flux_train_network.py && \
    printf '%s  %s\n' \
      afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38 /app/flux/ae.safetensors \
      660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd /app/flux/clip_l.safetensors \
      6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635 /app/flux/t5xxl_fp16.safetensors \
      | sha256sum --check --strict && \
    LD_PRELOAD=libtcmalloc.so \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64 \
    PYTHONPATH=/home/.local/lib/python3.10/site-packages \
    python3 -c "import os, accelerate, lion_pytorch, PIL, safetensors, toml, torch, yaml; assert torch.__version__ == '2.1.2+cu121'; assert torch.version.cuda == '12.1'; assert os.path.realpath(torch.__file__).startswith('/home/.local/lib/python3.10/site-packages/')"

RUN HF_HOME=/tmp/forge-flux-tokenizer-download \
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    LD_PRELOAD=libtcmalloc.so \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64 \
    PYTHONPATH=/home/.local/lib/python3.10/site-packages \
    python3 -m forge.flux_kohya_tokenizers stage && \
    python3 -c "import shutil; shutil.rmtree('/tmp/forge-flux-tokenizer-download')" && \
    LD_PRELOAD=libtcmalloc.so \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64 \
    PYTHONPATH=/home/.local/lib/python3.10/site-packages \
    python3 -m forge.flux_kohya_tokenizers verify && \
    LD_PRELOAD=libtcmalloc.so \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64 \
    PYTHONPATH=/home/.local/lib/python3.10/site-packages \
    python3 -m forge.verify_flux_kohya_runtime

ENTRYPOINT ["dumb-init", "--", "python3", "-m", "forge.cli"]
