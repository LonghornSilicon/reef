# Shared settings for the perf-model scripts. Sourced, not run.
#
#   PM     perf-model directory (this repo)
#   BUILD  where all build outputs go (never inside a synced folder)
#   DEPS   install prefix for yaml-cpp and Sparta built from ext/
#   JOBS   parallel compile jobs (keep at 4 on 8 GB machines)

PM="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="${JOBS:-4}"

# Build outside the source tree when the checkout lives on a Windows drive
# (/mnt/c/...): it is slow to build there, and OneDrive would try to sync
# hundreds of MB of objects.
if [[ "${PM}" == /mnt/* ]]; then
  BUILD="${BUILD:-${HOME}/.cache/coralnpu_perf}"
else
  BUILD="${BUILD:-${PM}/build}"
fi
DEPS="${BUILD}/deps"
MODEL="${BUILD}/model/coralnpu_perf"
DRIVER_DIR="${BUILD}/driver"

# WSL appends the Windows PATH, which can shadow Linux tools (e.g. a Windows
# bazel.exe). Put the Linux directories first.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}"
