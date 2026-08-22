# Validator-routed FLUX trainer. G.O.D deliberately selects this legacy-named
# Dockerfile for model_type=flux. Its downloader emits one of two cache shapes:
# an exact-one-root-file standalone checkpoint, or a full snapshot directory.
# Keep both pinned runtime graphs in one image and select only from cache shape.

FROM diagonalge/ai-toolkit:latest@sha256:c24f8bb95bf1dc8da7cd6158a763f2c9782783ad7648dc4047c5757ef3447db8 AS aitoolkit-runtime

COPY ops/docker/image-runtime-lock.txt \
    ops/docker/image-runtime-phase1-constraints.txt \
    ops/docker/verify_image_runtime.py \
    /opt/sn56/

# Reproduce the same pinned two-phase runtime used by the toolkit-named image.
# WEEK-9 HAZARD-1 (evidence/week9-hazards-20260819/CHANGES.md + ADDENDUM).
# retry_network <per-attempt seconds> <total budget seconds> <max attempts>.
# Every network attempt is bounded by an outer `timeout` (a hung TCP stream
# becomes exit 124, which the retry loop can act on); a total per-command
# budget bounds the all-attempts-stall case; and git's own stall detector is
# exported for the git clones pip runs for the two `git+https` requirements,
# which `timeout` alone could only kill wholesale.
# Caps are sized against the VALIDATOR'S 1800 s build limit (upstream f7caab6c
# trainer/constants.py:41 DOCKER_BUILD_TIMEOUT_MINUTES = 30, no retry on a
# failed build) -- NOT against what a slow link would like: above that wall a
# slow-but-working build is a DNF too, so a large cap only spends the window.
# Classic Docker spent 170 s committing four separate Ubuntu pip/wheel COPY
# layers after the site-packages transplant.  After the phase-3 verifier passes,
# fold those exact absent destinations into that already-required tree so the
# final stage commits it once.  The final verifier checks the merged result.
# The empty-store 6f0d0d1 measurement also proved that classic intermediate
# commits, not the commands, consume the remaining wall.  Keep the ordered
# install/verify sequence and every cap in one layer so no partial runtime is
# snapshotted between phases.
RUN set -eu; \
    test ! -e /opt/sn56/verify-image-runtime.py; \
    mv /opt/sn56/verify_image_runtime.py /opt/sn56/verify-image-runtime.py; \
    python3 /opt/sn56/verify-image-runtime.py \
        --lock /opt/sn56/image-runtime-lock.txt \
        --constraints /opt/sn56/image-runtime-phase1-constraints.txt \
        --files-only; \
    cd /app/ai-toolkit; \
    export GIT_HTTP_LOW_SPEED_LIMIT=1000 GIT_HTTP_LOW_SPEED_TIME=120; \
    retry_network() { \
        attempt_timeout_s=$1; \
        attempt_budget_s=$2; \
        attempt_max=$3; \
        shift 3; \
        command -v timeout >/dev/null 2>&1 || { \
            echo "SN56_NETWORK_TIMEOUT unavailable=timeout command=$1" >&2; \
            return 127; \
        }; \
        budget_deadline=$(( $(date +%s) + attempt_budget_s )); \
        attempt=1; \
        while :; do \
            timeout -k 30 "$attempt_timeout_s" "$@" && return 0; \
            status=$?; \
            if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then \
                echo "SN56_NETWORK_TIMEOUT attempt=$attempt timeout_seconds=$attempt_timeout_s command=$1 status=$status" >&2; \
            fi; \
            if [ "$attempt" -ge "$attempt_max" ] || [ "$(date +%s)" -ge "$budget_deadline" ]; then \
                echo "SN56_NETWORK_RETRY exhausted attempts=$attempt command=$1 status=$status" >&2; \
                return "$status"; \
            fi; \
            delay=$((attempt * 5)); \
            echo "SN56_NETWORK_RETRY retry=$((attempt + 1))/$attempt_max delay_seconds=$delay command=$1 status=$status" >&2; \
            sleep "$delay"; \
            attempt=$((attempt + 1)); \
        done; \
    }; \
    retry_network 60 150 3 git fetch origin 99be3d96a2468d3a5228a4eb05ba67e63c586b4e && \
    git checkout 99be3d96a2468d3a5228a4eb05ba67e63c586b4e && \
    retry_network 600 660 2 pip install --no-cache-dir \
        --constraint /opt/sn56/image-runtime-phase1-constraints.txt \
        --requirement requirements.txt && \
    retry_network 90 200 3 pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    retry_network 90 200 3 pip install --no-cache-dir \
        --constraint /opt/sn56/image-runtime-phase1-constraints.txt \
        torchcodec==0.2.1 pyyaml Pillow numpy safetensors && \
    retry_network 180 220 2 python3 -m pip install --no-cache-dir --no-deps \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        --requirement /opt/sn56/image-runtime-lock.txt && \
    python3 /opt/sn56/verify-image-runtime.py \
        --lock /opt/sn56/image-runtime-lock.txt \
        --constraints /opt/sn56/image-runtime-phase1-constraints.txt && \
    site=/usr/local/lib/python3.10/dist-packages && \
    test ! -e "$site/pip" && \
    test ! -e "$site/pip-22.0.2.dist-info" && \
    test ! -e "$site/wheel" && \
    test ! -e "$site/wheel-0.37.1.egg-info" && \
    cp -a /usr/lib/python3/dist-packages/pip "$site/pip" && \
    cp -a /usr/lib/python3/dist-packages/pip-22.0.2.dist-info "$site/pip-22.0.2.dist-info" && \
    cp -a /usr/lib/python3/dist-packages/wheel "$site/wheel" && \
    cp -a /usr/lib/python3/dist-packages/wheel-0.37.1.egg-info "$site/wheel-0.37.1.egg-info" && \
    test "$(git rev-parse HEAD)" = 99be3d96a2468d3a5228a4eb05ba67e63c586b4e

FROM diagonalge/kohya_latest:latest@sha256:d34dd5750e1018455e111f63c03bb2a4e16204607e00ba5af870dd7c71beb84e

# The default ai-toolkit graph imports bitsandbytes, whose Triton backend JITs
# a tiny CUDA driver helper on first use.  The pinned Kohya base deliberately
# omits every C toolchain component, so merely transplanting the Python graph
# is insufficient for snapshot-directory FLUX caches.  Install the minimal
# pinned Debian compiler/header surface here; the standalone Kohya child still
# uses its isolated Python/Torch graph below.  HTTPS is required because the
# provider's HTTP path has returned hash-mismatched package bodies in practice.
# WEEK-9 HAZARD-1: this hand-rolled apt loop retries on FAILURE only, exactly
# like retry_network did.  Bound each network apt call so a wedged mirror
# connection becomes a retryable non-zero status instead of an endless build.
RUN set -eu; \
    command -v timeout >/dev/null 2>&1 || { \
      echo "SN56_NETWORK_TIMEOUT unavailable=timeout command=apt-toolchain" >&2; \
      exit 127; \
    }; \
    rm -f /etc/apt/sources.list.d/cuda-debian11-x86_64.list; \
    sed -i \
      -e 's#http://deb.debian.org#https://deb.debian.org#g' \
      -e 's#http://security.debian.org#https://security.debian.org#g' \
      /etc/apt/sources.list.d/debian.sources; \
    attempt=1; installed=0; \
    while [ "$attempt" -le 3 ]; do \
      if rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/partial/* && \
        apt-get clean && \
        timeout -k 30 60 apt-get -o Acquire::Retries=3 update && \
        DEBIAN_FRONTEND=noninteractive \
          timeout -k 30 150 apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
            gcc=4:12.2.0-3 \
            gcc-12=12.2.0-14+deb12u1 \
            gcc-12-base=12.2.0-14+deb12u1 \
            libc-bin=2.36-9+deb12u14 \
            libc6=2.36-9+deb12u14 \
            libc6-dev=2.36-9+deb12u14 \
            libgcc-s1=12.2.0-14+deb12u1 \
            libstdc++6=12.2.0-14+deb12u1; then \
        installed=1; \
        break; \
      else \
        status=$?; \
      fi; \
      if [ "$attempt" -ge 3 ]; then \
        echo "SN56_NETWORK_RETRY exhausted attempts=$attempt command=apt-toolchain status=$status" >&2; \
        exit "$status"; \
      fi; \
      delay=$((attempt * 5)); \
      echo "SN56_NETWORK_RETRY retry=$((attempt + 1))/3 delay_seconds=$delay command=apt-toolchain status=$status" >&2; \
      sleep "$delay"; \
      attempt=$((attempt + 1)); \
    done; \
    test "$installed" = 1; \
    command -v cc; command -v gcc; \
    test -f /usr/include/stdlib.h; \
    test -f /usr/local/include/python3.10/Python.h; \
    printf '#include <stdlib.h>\nint main(void) { return 0; }\n' >/tmp/sn56-cc-probe.c; \
    cc /tmp/sn56-cc-probe.c -o /tmp/sn56-cc-probe; \
    /tmp/sn56-cc-probe; \
    rm -f /tmp/sn56-cc-probe.c /tmp/sn56-cc-probe; \
    install -d -m 0755 /opt/sn56; \
    dpkg-query -W -f='${Package}=${Version}\n' \
      gcc gcc-12 gcc-12-base libc-bin libc6 libc6-dev libgcc-s1 libstdc++6 \
      >/opt/sn56/legacy-aitoolkit-toolchain-lock.txt; \
    dpkg-query -W -f='${binary:Package}=${Version}\n' | \
      LC_ALL=C sort >/opt/sn56/legacy-os-package-inventory.txt; \
    sha256sum /opt/sn56/legacy-os-package-inventory.txt \
      >/opt/sn56/legacy-os-package-inventory.sha256; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# WEEK-9: FORGE_HOLDOUT_SELECTION_TYPES removed (promotion blocked by gates;
# see evidence/week9-gpu-campaign-20260819/alpha). No shadow tax.
ENV PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
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
# Copy exactly the three reviewed lock/verifier sources directly from the
# build context; the merged verifier below restores the runtime basename.
# Empty-store 329eaed showed that placing the first context COPY after the two
# runtime transplants still cost 109.946 s (versus 111.895 s cross-stage): the
# 43.3 GB root position, not the source, was the bottleneck.  Keep both context
# copies on the 29.2 GB Kohya root before growing it with the frozen runtimes.
COPY ops/docker/image-runtime-lock.txt \
    ops/docker/image-runtime-phase1-constraints.txt \
    ops/docker/verify_image_runtime.py \
    /opt/sn56/
COPY forge/ /app/forge/
COPY --from=aitoolkit-runtime /app/ai-toolkit/ /app/ai-toolkit/
COPY --from=aitoolkit-runtime /usr/local/lib/python3.10/dist-packages/ /opt/sn56/ai-toolkit-python/

# The pinned Kohya base already declares /app as its working directory.  Do not
# add a redundant metadata-only layer (22 s on the measured classic builder).

# Prove both isolated runtime graphs and stage/verify the pinned tokenizers in
# one layer.  This preserves every command and timeout while avoiding one more
# full classic-Docker snapshot of the 43.3 GB final filesystem.
# WEEK-9 HAZARD-1 (evidence/week9-hazards-20260819/CHANGES.md + ADDENDUM).
# retry_network <per-attempt seconds> <total budget seconds> <max attempts>.
# Every network attempt is bounded by an outer `timeout` (a hung TCP stream
# becomes exit 124, which the retry loop can act on); a total per-command
# budget bounds the all-attempts-stall case; and git's own stall detector is
# exported for the git clones pip runs for the two `git+https` requirements,
# which `timeout` alone could only kill wholesale.
# Caps are sized against the VALIDATOR'S 1800 s build limit (upstream f7caab6c
# trainer/constants.py:41 DOCKER_BUILD_TIMEOUT_MINUTES = 30, no retry on a
# failed build) -- NOT against what a slow link would like: above that wall a
# slow-but-working build is a DNF too, so a large cap only spends the window.
RUN set -eu; \
    test ! -e /opt/sn56/verify-image-runtime.py; \
    mv /opt/sn56/verify_image_runtime.py /opt/sn56/verify-image-runtime.py; \
    test -f /app/ai-toolkit/run.py && \
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
    python3 -c "import os, toolkit, torch; assert torch.__version__ == '2.6.0+cu124'; assert torch.version.cuda == '12.4'; assert os.path.realpath(torch.__file__).startswith('/opt/sn56/ai-toolkit-python/')" && \
    cd /app && \
    test -f /app/sd-scripts/flux_train_network.py && \
    printf '%s  %s\n' \
      afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38 /app/flux/ae.safetensors \
      660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd /app/flux/clip_l.safetensors \
      6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635 /app/flux/t5xxl_fp16.safetensors \
      | sha256sum --check --strict && \
    LD_PRELOAD=libtcmalloc.so \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64 \
    PYTHONPATH=/home/.local/lib/python3.10/site-packages \
    python3 -c "import os, accelerate, lion_pytorch, PIL, safetensors, toml, torch, yaml; assert torch.__version__ == '2.1.2+cu121'; assert torch.version.cuda == '12.1'; assert os.path.realpath(torch.__file__).startswith('/home/.local/lib/python3.10/site-packages/')" && \
    export GIT_HTTP_LOW_SPEED_LIMIT=1000 GIT_HTTP_LOW_SPEED_TIME=120; \
    retry_network() { \
        attempt_timeout_s=$1; \
        attempt_budget_s=$2; \
        attempt_max=$3; \
        shift 3; \
        command -v timeout >/dev/null 2>&1 || { \
            echo "SN56_NETWORK_TIMEOUT unavailable=timeout command=$1" >&2; \
            return 127; \
        }; \
        budget_deadline=$(( $(date +%s) + attempt_budget_s )); \
        attempt=1; \
        while :; do \
            timeout -k 30 "$attempt_timeout_s" "$@" && return 0; \
            status=$?; \
            if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then \
                echo "SN56_NETWORK_TIMEOUT attempt=$attempt timeout_seconds=$attempt_timeout_s command=$1 status=$status" >&2; \
            fi; \
            if [ "$attempt" -ge "$attempt_max" ] || [ "$(date +%s)" -ge "$budget_deadline" ]; then \
                echo "SN56_NETWORK_RETRY exhausted attempts=$attempt command=$1 status=$status" >&2; \
                return "$status"; \
            fi; \
            delay=$((attempt * 5)); \
            echo "SN56_NETWORK_RETRY retry=$((attempt + 1))/$attempt_max delay_seconds=$delay command=$1 status=$status" >&2; \
            sleep "$delay"; \
            attempt=$((attempt + 1)); \
        done; \
    }; \
    retry_network 90 120 2 env \
        HF_HOME=/tmp/forge-flux-tokenizer-download \
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
