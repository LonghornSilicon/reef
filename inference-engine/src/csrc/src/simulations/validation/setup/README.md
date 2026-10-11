# Machine-specific validation setup: mew3

`mew3-host-clang-includes.patch` records a local Coral host-toolchain adjustment
needed by mew3's existing compiler wrappers. It adds the wrappers' GCC 11 and
private dependency header roots to Bazel's builtin include directories.

This is **machine-specific setup, not a portable kernel requirement**. The
W8A8 kernel and reference checks do not depend on these absolute paths. Do not
apply this patch on another host unless it uses the same wrappers and paths;
configure that host's own compiler and builtin header roots instead.

The patch targets Coral's `toolchain/host_clang/BUILD`, not Reef's inference
engine. It is already applied in the validated Coral checkout. Applying it
again will fail; use `git apply --check` before applying it to a fresh matching
checkout. The simulations README records the command and successful revision.

Bazel's `--sandbox_add_mount_pair=/tmp/reef-w8a8-repo-contents` is likewise a
setup setting for the recorded `/tmp` cache layout. It keeps that cache visible
inside Bazel's hermetic temporary directory; it is not a guest kernel setting.
