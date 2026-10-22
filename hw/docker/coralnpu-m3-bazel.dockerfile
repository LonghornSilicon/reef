# Overlay on top of upstream CoralNPU M3-2026-04-27 dev image.
# Upstream utils/coralnpu.dockerfile installs Bazel from the bazel-apt repo with a
# signing key that has since been rotated (NO_PUBKEY 3D5919B448457EE0), and its
# RUN heredoc has no `set -e`, so the image builds "successfully" with no bazel.
# Install Bazelisk as `bazel`; it reads .bazelversion (7.4.1 at M3) from the repo.
#
# Build:  docker build -t coralnpu:m3 -f utils/coralnpu.dockerfile .   (in coralnpu/)
#         docker build -t coralnpu:m3-bazel -f docker/coralnpu-m3-bazel.dockerfile .
FROM coralnpu:m3
USER root
ARG BAZELISK_VERSION=v1.25.0
RUN curl -fsSL -o /usr/local/bin/bazel \
      https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64 \
 && chmod +x /usr/local/bin/bazel
# Keep bazelisk downloads inside the mounted, builder-owned bazel cache.
ENV BAZELISK_HOME=/home/builder/.cache/bazel/_bazelisk
USER builder
