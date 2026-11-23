# Patches for the pinned-RTL Spike

`ext/spike-pinned` is Spike at `93a10ae`, the commit CoralNPU `8a5e1b31`
pins in `rules/repo_defs.bzl` (`riscv_isa_sim`). The patches here are
CoralNPU's own, copied unchanged from `third_party/spike/` at `8a5e1b31`
(Apache License 2.0, as CoralNPU). `scripts/build_deps.sh spike-pinned`
applies them, in order, to a copy of the submodule; the submodule itself is
not modified.

They add `mpause`, CoralNPU's custom CSRs and its documented deviations,
and align Spike's Zvt v0.4 with the RTL (0009). CoralNPU runs this build in
lockstep with the RTL and MPACT in its Zvt co-simulation (`b4be3e79`).

Only `reef_perf_pinned` links it. `reef_perf`, and so `m3.yaml`,
`m3_vme.yaml` and the M3 golden, keep `ext/spike` at `fd72ee2d`, the commit
M3 pins (docs/deviations.md).
