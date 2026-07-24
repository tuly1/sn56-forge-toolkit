FROM diagonalge/ai-toolkit:latest@sha256:c24f8bb95bf1dc8da7cd6158a763f2c9782783ad7648dc4047c5757ef3447db8

ENV AI_TOOLKIT_DIR=/app/ai-toolkit
ENV FORGE_TEMPLATES_DIR=/app/forge/templates

# This is the exact 185-entry version/VCS metadata inventory observed in both
# independently built H100 subjects. It does not attest downloaded wheel bytes.
COPY ops/docker/image-runtime-lock.txt /opt/sn56/image-runtime-lock.txt
COPY ops/docker/image-runtime-phase1-constraints.txt /opt/sn56/image-runtime-phase1-constraints.txt
COPY ops/docker/verify_image_runtime.py /opt/sn56/verify-image-runtime.py
RUN python3 /opt/sn56/verify-image-runtime.py \
        --lock /opt/sn56/image-runtime-lock.txt \
        --constraints /opt/sn56/image-runtime-phase1-constraints.txt \
        --files-only

WORKDIR /app/ai-toolkit
# Phase 1 constrains the ordinary Python graph while leaving the explicitly
# installed CUDA/Torch island and easy_dwpose's known metadata conflict open.
RUN retry_network() { \
        attempt=1; \
        while :; do \
            "$@" && return 0; \
            status=$?; \
            if [ "$attempt" -ge 5 ]; then \
                echo "SN56_NETWORK_RETRY exhausted attempts=$attempt command=$1 status=$status" >&2; \
                return "$status"; \
            fi; \
            delay=$((attempt * 5)); \
            echo "SN56_NETWORK_RETRY retry=$((attempt + 1))/5 delay_seconds=$delay command=$1 status=$status" >&2; \
            sleep "$delay"; \
            attempt=$((attempt + 1)); \
        done; \
    }; \
    retry_network git fetch origin 99be3d96a2468d3a5228a4eb05ba67e63c586b4e && \
    git checkout 99be3d96a2468d3a5228a4eb05ba67e63c586b4e && \
    retry_network pip install --no-cache-dir \
        --constraint /opt/sn56/image-runtime-phase1-constraints.txt \
        --requirement requirements.txt && \
    retry_network pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124

# ai-toolkit currently pins torchcodec 0.9.1, whose compiled extension targets
# Torch 2.9.  G.O.D deliberately pins this image to Torch 2.6/cu124; the official
# TorchCodec compatibility matrix maps that ABI to the 0.2 series.  Repin after
# requirements.txt and Torch so even optional media imports remain loadable.
RUN retry_network() { \
        attempt=1; \
        while :; do \
            "$@" && return 0; \
            status=$?; \
            if [ "$attempt" -ge 5 ]; then \
                echo "SN56_NETWORK_RETRY exhausted attempts=$attempt command=$1 status=$status" >&2; \
                return "$status"; \
            fi; \
            delay=$((attempt * 5)); \
            echo "SN56_NETWORK_RETRY retry=$((attempt + 1))/5 delay_seconds=$delay command=$1 status=$status" >&2; \
            sleep "$delay"; \
            attempt=$((attempt + 1)); \
        done; \
    }; \
    retry_network pip install --no-cache-dir \
        --constraint /opt/sn56/image-runtime-phase1-constraints.txt \
        torchcodec==0.2.1 pyyaml Pillow numpy safetensors

# Phase 2 requests every certified version/source commit without dependency
# resolution. --no-deps preserves the intentional easy_dwpose/hub metadata
# mismatch. The verifier checks the serialized metadata inventory and requires
# that mismatch to be the only output from `pip check`.
RUN retry_network() { \
        attempt=1; \
        while :; do \
            "$@" && return 0; \
            status=$?; \
            if [ "$attempt" -ge 5 ]; then \
                echo "SN56_NETWORK_RETRY exhausted attempts=$attempt command=$1 status=$status" >&2; \
                return "$status"; \
            fi; \
            delay=$((attempt * 5)); \
            echo "SN56_NETWORK_RETRY retry=$((attempt + 1))/5 delay_seconds=$delay command=$1 status=$status" >&2; \
            sleep "$delay"; \
            attempt=$((attempt + 1)); \
        done; \
    }; \
    retry_network python3 -m pip install --no-cache-dir --no-deps \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        --requirement /opt/sn56/image-runtime-lock.txt && \
    python3 /opt/sn56/verify-image-runtime.py \
        --lock /opt/sn56/image-runtime-lock.txt \
        --constraints /opt/sn56/image-runtime-phase1-constraints.txt

WORKDIR /app
# Templates ship inside the package (forge/templates/*.yaml), so this one COPY
# brings the trainer AND its config templates. FORGE_TEMPLATES_DIR still points
# at /app/forge/templates as a belt-and-braces override.
COPY forge/ /app/forge/

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "python3", "-m", "forge.cli"]
