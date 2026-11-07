# RTL tool image. Build with docker/run.sh build.
FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git make build-essential xz-utils \
    && rm -rf /var/lib/apt/lists/*

# OSS CAD Suite: verilator, yosys, yosys-slang, iverilog. One dated tag pins all.
ARG OSS_CAD_SUITE_TAG=2026-10-12
RUN curl -fsSL --retry 3 \
      "https://github.com/YosysHQ/oss-cad-suite-build/releases/download/${OSS_CAD_SUITE_TAG}/oss-cad-suite-linux-x64-${OSS_CAD_SUITE_TAG//-/}.tgz" \
    | tar -xz -C /opt
# py3bin: verilator --binary runs python3, which ubuntu:22.04 does not ship.
ENV PATH="${PATH}:/opt/oss-cad-suite/bin:/opt/oss-cad-suite/py3bin"

ARG VERIBLE_TAG=v0.0-4296-g0f262651
RUN curl -fsSL --retry 3 \
      "https://github.com/chipsalliance/verible/releases/download/${VERIBLE_TAG}/verible-${VERIBLE_TAG}-linux-static-x86_64.tar.gz" \
    | tar -xz -C /opt && mv "/opt/verible-${VERIBLE_TAG}" /opt/verible
ENV PATH="${PATH}:/opt/verible/bin"

ARG UV_VERSION=0.10.10
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_LINK_MODE=copy

# The repo is bind-mounted and owned by the host user.
RUN git config --global --add safe.directory "*"

WORKDIR /app/reef/rtl
