FROM diagonalge/ai-toolkit:latest@sha256:c24f8bb95bf1dc8da7cd6158a763f2c9782783ad7648dc4047c5757ef3447db8

ENV AI_TOOLKIT_DIR=/app/ai-toolkit
ENV FORGE_TEMPLATES_DIR=/app/forge/templates

WORKDIR /app/ai-toolkit
RUN git fetch origin 99be3d96a2468d3a5228a4eb05ba67e63c586b4e && \
    git checkout 99be3d96a2468d3a5228a4eb05ba67e63c586b4e && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124

# The base image and ai-toolkit source are immutable.  A fully resolved Python
# constraints lock is still required; do not mistake these direct installs for
# a transitive dependency lock.
# ai-toolkit currently pins torchcodec 0.9.1, whose compiled extension targets
# Torch 2.9.  G.O.D deliberately pins this image to Torch 2.6/cu124; the official
# TorchCodec compatibility matrix maps that ABI to the 0.2 series.  Repin after
# requirements.txt and Torch so even optional media imports remain loadable.
RUN pip install --no-cache-dir \
        torchcodec==0.2.1 pyyaml Pillow numpy safetensors

WORKDIR /app
# Templates ship inside the package (forge/templates/*.yaml), so this one COPY
# brings the trainer AND its config templates. FORGE_TEMPLATES_DIR still points
# at /app/forge/templates as a belt-and-braces override.
COPY forge/ /app/forge/

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "python3", "-m", "forge.cli"]
