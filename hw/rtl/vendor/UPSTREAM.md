# Vendored CoralNPU RTL

**Do not hand-edit anything under `rtl/vendor/coralnpu/`.** The files are read-only and
get replaced on every regen. Integration logic goes in our own SV (`reef_top`, SRAM wrappers).

| Field | Value |
|---|---|
| Upstream | https://github.com/google-coral/coralnpu |
| Release | `M3-2026-04-27` |
| Commit | `72d700ef724669163e1b6f88e8622676ef6b0a74` (2026-04-24) |
| Top | `RvvCoreMiniAxi` (scalar + RVV 1.0 VLEN 128 + FPU, AXI4 128-bit) |
| Bundle | release asset `RvvCoreMiniAxi.sv`, sha256 `74d3bc7e3295353fd879f70a513f7c3fde79912010a1f62f08d815fbcd6f4ff4` |
| Chisel flags (for `--build`) | `--enableFetchL0=False --fetchDataBits=128 --lsuDataBits=128 --enableRvv=True --enableFloat=True --useAxi` |
| Defines | `USE_GENERIC TB_SUPPORT VLEN_128 ZVE32F_ON` (+ `SYNTHESIS` for synth) |
| Filelists | `coralnpu/synth.f` (design, 129 entries), `coralnpu/sim.f` (+ 26 firtool verification layers) |

Regenerate: `scripts/regen_coralnpu.sh` (release asset) or `--build` (re-emit from Chisel).
Check: `scripts/lint_vendor.sh`. Details: `docs/baseline-m3.md`.
