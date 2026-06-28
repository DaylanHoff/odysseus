FROM python:3.14-slim

# System deps. tmux is required by Cookbook for background downloads/serves.
# openssh-client is required for Cookbook remote server tests, setup, probes,
# downloads, and serves from Docker installs.
# git/cmake are required when Cookbook builds llama.cpp on first llama.cpp
# launch inside Docker.
# nodejs/npm provide npx for the optional built-in Browser MCP server.
# gosu lets the entrypoint drop privileges cleanly so signals still reach
# uvicorn directly (no extra shell layer like `su`/`sudo` would add).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    nodejs \
    npm \
    tmux \
    openssh-client \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Docker CLI (client only — daemon stays on the host via the
# /var/run/docker.sock mount). The Debian `docker.io` package ships
# dockerd but not the client binary on slim, so grab the static client
# tarball from download.docker.com instead.
ARG DOCKER_CLI_VERSION=27.5.1
RUN ARCH="$(dpkg --print-architecture)" \
    && case "$ARCH" in \
         amd64) DARCH=x86_64 ;; \
         arm64) DARCH=aarch64 ;; \
         *) echo "unsupported arch $ARCH"; exit 1 ;; \
       esac \
    && curl -fsSL "https://download.docker.com/linux/static/stable/${DARCH}/docker-${DOCKER_CLI_VERSION}.tgz" \
       -o /tmp/docker.tgz \
    && tar -xzf /tmp/docker.tgz -C /tmp \
    && install -m 0755 /tmp/docker/docker /usr/local/bin/docker \
    && rm -rf /tmp/docker /tmp/docker.tgz

WORKDIR /app

# Install Python deps first (layer cache). Optional extras (PyMuPDF AGPL, etc.)
# are opt-in so the default image stays MIT-core; see requirements-optional.txt.
ARG INSTALL_OPTIONAL=false
COPY requirements.txt requirements-optional.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_OPTIONAL" = "true" ]; then pip install --no-cache-dir -r requirements-optional.txt; fi

# Pre-install llama-cpp-python[server] (CPU-only build) so Cookbook's
# `llama-server` launch path works out of the box on first use, without a
# runtime source build. Python 3.14 has no prebuilt wheels on the cpu index,
# so this compiles llama.cpp from the sdist.
#
# CC=gcc/CXX=g++ MUST be set explicitly: without CXX=g++ the sdist's final
# link of libggml-base.so drops libstdc++, producing an .so with an undefined
# `__cxxabiv1::__class_type_info` vtable symbol that fails to load at import
# time. With CXX=g++ the .so correctly links libstdc++.so.6.
#
# GGML_NATIVE=OFF on amd64 is REQUIRED: with the default GGML_NATIVE=ON cmake
# probes the GHA build runner's CPU and enables every extension it finds.
# The github-hosted amd64 runner has AVX-512-VNNI, so the build emits `vpdpbusd`
# instructions. The deployment target (TrueNAS i7-14700) has AVX2 + AVX-VNNI
# but NO AVX-512, so those instructions trap as "Illegal instruction" at
# runtime and the model fails to load. Pinning to AVX2/FMA/F16C/AVX-VNNI
# (and leaving every GGML_AVX512_* off) produces a binary tuned to the target
# without using instructions it cannot execute. arm64 keeps native detection.
#
# Parallelism is capped low: the GHA build runner is a 2-core / 7GB instance,
# and GCC 14 hits an internal compiler error ("Bus error" in stl_vector.h)
# under -O3 with high -j on the ggml-cpu ops, which previously bricked builds
# here. CMAKE_BUILD_PARALLEL_LEVEL=2 keeps it within the runner's RAM and
# dodges the ICE.
ARG LLAMA_CPP_PYTHON=true
RUN if [ "$LLAMA_CPP_PYTHON" = "true" ]; then \
        if [ "$(dpkg --print-architecture)" = "amd64" ]; then \
            EXTRA_CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX_VNNI=ON -DGGML_SSE42=ON -DGGML_AVX512=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF -DGGML_AVX512_VBMI=OFF"; \
        else \
            EXTRA_CMAKE_ARGS=""; \
        fi ; \
        CC=gcc CXX=g++ \
        CMAKE_ARGS="$EXTRA_CMAKE_ARGS" \
        CMAKE_BUILD_PARALLEL_LEVEL=2 \
        MAKEFLAGS="-j2" \
        pip install --no-cache-dir \
            "llama-cpp-python[server]" \
            --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu ; \
    fi

# Copy app code
COPY . .

# Create data directory (mount a volume here for persistence)
RUN mkdir -p data logs services/cache/search

# Entrypoint that drops to PUID/PGID (default 1000:1000) and repairs
# ownership on the bind-mounted /app/data and /app/logs. Without this,
# the container runs as root and writes root-owned files into host
# bind mounts — any later non-root run (or a host user trying to
# update them) silently fails on EPERM, breaking skill extraction,
# prefs persistence, mail attachments, etc.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 7000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7000"]
