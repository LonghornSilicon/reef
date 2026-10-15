# PD image. No EDA tools yet, the stages are stubs. No PDK in the image;
# docker/install_pdk.sh puts it in $PDK_ROOT.
FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git make shellcheck yamllint \
    && rm -rf /var/lib/apt/lists/*

ARG UV_VERSION=0.10.10
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_LINK_MODE=copy

# The repo is bind-mounted and owned by the host user.
RUN git config --global --add safe.directory "*"

ENV PDK_ROOT=/pdk
WORKDIR /app/reef/pd
