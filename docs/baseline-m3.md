# Reef RTL baseline: CoralNPU M3-2026-04-27

**Status:** pinned 2026-09-27 (kickoff deck milestone 9/30: "upstream baseline pinned (Aaron)").
**Owner after handoff:** RTL generation owner (Samuel Qiao per deck v5).

## What is pinned

| | |
|---|---|
| Release | `M3-2026-04-27`, commit `72d700ef724669163e1b6f88e8622676ef6b0a74` |
| Top | `RvvCoreMiniAxi`: scalar core + RVV 1.0 (VLEN 128, vector FP32) + scalar FPU, AXI4 128-bit |
| Source of the SV | the release's own `RvvCoreMiniAxi.sv` asset, sha256 `74d3bc7e…6f4ff4` |
| Proof it matches Chisel | re-emitting from Chisel at `72d700e` gives a byte-identical file (`results/m3_verification.md`) |
| Regression | cocotb `core_mini_axi_sim_cocotb` 18/18 PASS, `rvv_core_mini_axi_sim_cocotb` 18/18 PASS |
| Lint (this repo) | `scripts/lint_vendor.sh`: sim set PASS, synth set PASS (Verilator 5.032, M3's hermetic build) |

Not in M3: the VME/Zvt matrix engine (upstream main only) and L1 I/D caches (Chisel exists, not
instantiated in this top). See `docs/rtl-map-v0.md`.

## How to use it

```bash
scripts/regen_coralnpu.sh   # download release asset, check sha256, split into rtl/vendor/coralnpu/
scripts/lint_vendor.sh      # lint sim + synth define sets
```

No Chisel, Docker, or Bazel is needed for this: we are SV-first. `regen_coralnpu.sh --build` re-emits
from Chisel in the `coralnpu:m3` container. That is only needed if a generator flag changes (TCM size,
`enableFloat`, ...), which is a leads decision, and the output must then get a new sha256 and a
full regression.

## The files

`firtool` ships the whole design as one 83,147-line bundle of 154 concatenated files. We split it
back (`scripts/split_generated_sv.py`; reassembly is byte-identical):

| | Count | Notes |
|---|---|---|
| Design sources | 119 | Chisel-generated `RvvCoreMiniAxi.sv` + hand-written SV (RVV backend, fpnew, T-Head dividers, `Sram.v`, `ClockGate.sv`, `RstSync.sv`) |
| Headers | 10 | Listed in the filelists **in bundle order**: fpnew uses macros from `registers.svh` without `include`-ing it |
| Verification layers | 26 | firtool assert/assume/cover `bind` files. In `sim.f` only |

**Filelists are one compilation unit.** Give tools `synth.f` as-is, in order, and set
`CORALNPU_RTL=<repo>/rtl/vendor/coralnpu`.

## Defines (important)

| Define | Sim | Synth | Why |
|---|---|---|---|
| `USE_GENERIC` | ✓ | ✓ | generic (non-vendor) cells in `Sram.v` / `ClockGate.sv` until we add our node's branch |
| `TB_SUPPORT` | ✓ | **✓** | **Not testbench-only in M3.** It adds `uop_pc` / `last_uop_valid` to the RVV ROB→retire struct. The scalar `RetirementBuffer` matches vector retirements on those fields, and it drives dispatch (`nSpace`, `empty`, `trapPending`). Without it `RvvCoreWrapper.sv` fails to elaborate (8 errors). Cost: a 32-bit PC carried per RVV uop (area, for Samuel's estimates) |
| `VLEN_128` | ✓ | ✓ | vector length |
| `ZVE32F_ON` | ✓ | ✓ | vector FP32; dropping it is an area decision for Arch, and it is an untested config |
| `SYNTHESIS` | | ✓ | removes sim-only init/assert code |

## For PD (10/14 drop)

- Filelist: `rtl/vendor/coralnpu/synth.f`, top `RvvCoreMiniAxi`, defines above.
- Memories: ITCM `SRAM_512x128` and DTCM `SRAM_2048x128`, both `Sram` (single port, sync read,
  128-bit, 16-bit byte mask). `Sram.v` already has `USE_TSMC12FFC` / `USE_GF22` macro branches; ours goes next to them.
- Clock gating: `ClockGate.sv` (latch-based under `USE_GENERIC`). Needs our node's ICG cell branch.
- Top ports: `results/RvvCoreMiniAxi_ports.csv` (397 ports). Functional pins are clock, reset,
  AXI slave (39) + master (39), `io_irq`, `io_timer_irq`, `io_software_irq`, `io_boot_addr[31:0]`, `io_te`,
  `io_halted`, `io_fault`, `io_wfi`, and `io_dm_*` (9). The other 300 (`io_debug_*`) are sim trace and stay unconnected.

## Handoff to the RTL generation owner

1. Run the two commands above on a clean clone; both should pass.
2. Any change to `rtl/vendor/` is a PR that updates the sha256 in `scripts/regen_coralnpu.sh`,
   `rtl/vendor/UPSTREAM.md`, and this file, with lint + both cocotb suites attached.
3. Area/timing estimates should use the synth define set above (with `TB_SUPPORT`).
