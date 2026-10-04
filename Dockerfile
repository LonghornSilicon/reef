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
        cmake \
        doxygen \
        graphviz \
        libssl-dev \
        vim \
        libglib2.0-dev \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install uv.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install main project.
WORKDIR /app/reef 
COPY . .

# Install inference-engine.
WORKDIR /app/reef/inference-engine
RUN uv sync --frozen

# Set starting directory. 
WORKDIR /app/reef
