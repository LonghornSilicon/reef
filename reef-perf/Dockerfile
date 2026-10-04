# reef-perf development and CI image.
#
#   tools/docker.sh build      # build this image
#   tools/docker.sh shell      # interactive shell with the repo mounted
#   tools/docker.sh test       # run the full test suite
#
# Stages: `base` (system packages + uv), `deps` (yaml-cpp, Sparta and Spike
# built from the ext/ submodules) and the final image (reef_perf built and the
# Python environment synced). The `deps` stage is the slow one (~1 hour) and is
# cached until something under ext/ or tools/build_deps.sh changes.
FROM ubuntu:22.04 AS base

# The shell should be bash so sourcing will work.
SHELL ["/bin/bash", "-c"]

# `apt-get update` was failing with DNS resolution errors (e.g. "Temporary
# failure resolving 'archive.ubuntu.com'"). Root cause: apt downloads as the
# unprivileged `_apt` user, which lacked write access to
# /var/lib/apt/lists/partial in this base image; the resulting sandbox
# permission failure was misreported as a DNS error. Fix: disable apt's
# download sandboxing, reset the lists dir, and explicitly fail the build
# on any Err/W output, since `apt-get update` exits 0 even on partial failure.
RUN echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/99no-sandbox
RUN rm -rf /var/lib/apt/lists
RUN set -o pipefail && \
    apt-get update 2>&1 | \
    tee /tmp/apt-update.log && \
    if grep -E '^(Err:|W:)' /tmp/apt-update.log; then \
        exit 1; \
    fi

# Install packages.
#   boost, rapidjson, sqlite3, hdf5, zlib, lzma  Sparta
#   device-tree-compiler                         Spike
#   binutils-riscv64-linux-gnu                   assembles the RISC-V workloads
#   doxygen                                      C++ API documentation check
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        less \
        vim \
        tzdata \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        device-tree-compiler \
        libboost-all-dev \
        rapidjson-dev \
        libsqlite3-dev \
        libhdf5-dev \
        zlib1g-dev \
        liblzma-dev \
        binutils-riscv64-linux-gnu \
        doxygen \
    && rm -rf /var/lib/apt/lists/*

# Optional extra root certificate, for networks that intercept TLS
# (corporate proxies, antivirus "HTTPS scanning"). Provide it with
# `REEF_PERF_EXTRA_CA=/path/to/root.pem tools/docker.sh build`; without it
# this step does nothing. uv is told to use the system certificate store.
RUN --mount=type=secret,id=extra_ca \
    if [ -s /run/secrets/extra_ca ]; then \
        cp /run/secrets/extra_ca /usr/local/share/ca-certificates/extra_ca.crt && \
        update-ca-certificates; \
    fi
ENV UV_SYSTEM_CERTS=1

# Install uv. pipefail makes a failed download stop the build here, instead
# of `sh` silently running an empty script.
RUN set -o pipefail && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

FROM base AS deps

# Parallel compile jobs. Sparta needs ~1 GB of RAM per job.
ARG JOBS=4
WORKDIR /app/reef/reef-perf
COPY tools/build_deps.sh tools/build_deps.sh
# One layer per dependency, so bumping one submodule rebuilds only that one.
COPY ext/yaml-cpp ext/yaml-cpp
RUN bash tools/build_deps.sh /opt/reef-perf "${JOBS}" yaml-cpp
COPY ext/map ext/map
RUN bash tools/build_deps.sh /opt/reef-perf "${JOBS}" sparta
COPY ext/spike ext/spike
RUN bash tools/build_deps.sh /opt/reef-perf "${JOBS}" spike
ENV REEF_PERF_DEPS=/opt/reef-perf

FROM deps

ARG JOBS=4
WORKDIR /app/reef/reef-perf
# Keep the virtual environment outside the source tree, so mounting a
# checkout over /app/reef/reef-perf (tools/docker.sh shell) does not hide it.
ENV UV_PROJECT_ENVIRONMENT=/opt/reef-perf-venv

# Python environment first, so source edits do not invalidate it.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Install main project.
COPY . .
RUN uv sync --frozen
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j "${JOBS}"
ENV PATH="/app/reef/reef-perf/build/bin:/app/reef/reef-perf/tools:$PATH"
