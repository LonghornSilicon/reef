#!/usr/bin/env bash
# Installs the system packages the build needs (Ubuntu 22.04 / WSL2). Run once.
# Everything else (Sparta, yaml-cpp, MPACT) comes from the git submodules in
# ext/ and is built by scripts/build.sh without sudo.
set -euo pipefail

sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  build-essential cmake ninja-build git curl python3 \
  libboost-all-dev rapidjson-dev libsqlite3-dev libhdf5-dev zlib1g-dev liblzma-dev \
  clang default-jre-headless \
  binutils-riscv64-linux-gnu
# Why these:
#   boost, rapidjson, sqlite3, hdf5, zlib, lzma  - Sparta
#   clang, default-jre-headless                  - MPACT (its .bazelrc selects clang;
#                                                  its ISA parser is generated with ANTLR/Java)
#   binutils-riscv64-linux-gnu                   - assembles the workloads (RVV 1.0 support)

if ! command -v bazelisk >/dev/null; then
  sudo curl -sSL -o /usr/local/bin/bazelisk \
    https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
  sudo chmod +x /usr/local/bin/bazelisk
fi

echo "System dependencies installed. Next: scripts/build.sh"
