FROM diagonalge/ai-toolkit:latest

ENV AI_TOOLKIT_DIR=/app/ai-toolkit
ENV FORGE_TEMPLATES_DIR=/app/forge/templates

WORKDIR /app/ai-toolkit
RUN git fetch origin 99be3d96a2468d3a5228a4eb05ba67e63c586b4e && \
    git checkout 99be3d96a2468d3a5228a4eb05ba67e63c586b4e && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124

# Our trainer + deps (pyyaml/pillow/numpy usually present in the base; pin anyway)
RUN pip install --no-cache-dir pyyaml Pillow numpy safetensors

WORKDIR /app
# Templates ship inside the package (forge/templates/*.yaml), so this one COPY
# brings the trainer AND its config templates. FORGE_TEMPLATES_DIR still points
# at /app/forge/templates as a belt-and-braces override.
COPY forge/ /app/forge/

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "python3", "-m", "forge.cli"]
