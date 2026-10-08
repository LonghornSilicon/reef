FROM ubuntu:22.04

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
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        less \
        git \
        gh \
        gnupg \
        ca-certificates \
        openssh-client \
        curl \
        wget \
        build-essential \
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
        cmake \
        doxygen \
        graphviz \
        libssl-dev \
        vim \
        libglib2.0-dev \
        tzdata \
        docker.io \
        docker-buildx \
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

# Install uv.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install main project.
WORKDIR /app/reef 
COPY . .

### Install inference-engine. ###
WORKDIR /app/reef/inference-engine
RUN uv sync --frozen

### Install reef-perf. ### 
# Parallel compile jobs. Sparta needs ~1 GB of RAM per job.
WORKDIR /app/reef/reef-perf
ARG JOBS=4
RUN bash tools/build_deps.sh /opt/reef-perf "${JOBS}" yaml-cpp
RUN bash tools/build_deps.sh /opt/reef-perf "${JOBS}" sparta
RUN bash tools/build_deps.sh /opt/reef-perf "${JOBS}" spike
ENV REEF_PERF_DEPS=/opt/reef-perf
RUN uv sync --frozen
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j "${JOBS}"
ENV PATH="/app/reef/reef-perf/build/bin:/app/reef/reef-perf/tools:$PATH"

# Set starting directory. 
WORKDIR /app/reef
