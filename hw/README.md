# hw: Reef RTL

Reef's hardware: Google's [CoralNPU](https://github.com/google-coral/coralnpu) core plus the sky130
test-chip shell, the sky130 TCM and clock-gate swaps, the PD Drop 0 handoff, and the scripts that
lint, simulate and synthesize it.

**Baseline: CoralNPU release `M3-2026-04-27` (commit `72d700e`), top `RvvCoreMiniAxi`.** SV-first:
we vendor the release's generated SystemVerilog (sha256-pinned) and never edit Chisel or the
generated files. sky130 parts replace two vendored files through the filelist. See
`docs/baseline-m3.md`.

All paths below are relative to this `hw/` directory. Every script finds it from its own location,
so they run from anywhere.

## Layout

```
rtl/
  vendor/coralnpu/      # pinned CoralNPU RTL, split into files + synth.f / sim.f (read-only)
  vendor/UPSTREAM.md    # what is pinned (release, sha256, defines)
  reef/                 # reef_top test-chip shell: SPI->AXI host bridge, AXI DECERR sink, reset sync
  reef/synth/           # synthesis-only stand-ins (Sram black box, DC edff workaround)
  sram/                 # TCM SRAMs for sky130: Sram_sram22.sv (5 SRAM22 macros, Drop 0) and
                        #   Sram_sky130.sv (20 OpenRAM macros, fallback); README.md has the details
  clock/                # clock gate: ClockGate_sky130.sv (sky130_fd_sc_hd__sdlclkp_4) + Verilator model
scripts/
  regen_coralnpu.sh          # vendor the pinned RTL (release asset, or --build from Chisel)
  split_generated_sv.py      # split firtool's single-file bundle into files + filelists
  lint_vendor.sh             # Verilator lint of the vendored core, sim + synth define sets
  lint_reef.sh               # Verilator lint of rtl/reef (-Wall), the TCM wrappers, reef_top + core
  gen_filelists.sh           # flat sky130 reef_top filelists for PD (configs/filelists/)
  fetch_sram22.sh            # SRAM22 macro .lib/LEF(/GDS) at the pinned commit, sha256-checked
  run_reef_smoke.sh          # reef_top smoke test: SPI load, CSR boot, RVV program, readback
  run_sram_sky130_tb.sh      # TCM wrapper unit test (VCS, or SIM=verilator)
  run_clock_gate_sky130_tb.sh # clock gate vs the PDK cell model (VCS)
  synth_yosys.sh             # Yosys + yosys-slang synthesis for area (open flow, also in CI)
  check_synth.py             # pass/fail check of a Yosys run (cell types, macro counts, area budget)
  area_report.py             # roll Yosys area up by block
  synth_dc.sh, sta_pt.sh     # Design Compiler / PrimeTime runs (ECE servers), e.g. sky130 ss at 20 ns
  verify_upstream_release.sh # full re-verification of a release tag (Docker + Bazel + cocotb)
  check_no_confidential.sh   # run before pushing: no PDK/NDA files, secrets or server paths
configs/
  filelists/reef_top_sky130{,_sim}.f  # PD Drop 0 filelists (generated; set REEF_ROOT, below)
  filelists/reef_top_sky130_openram{,_sim}.f  # same with the OpenRAM fallback TCM
  sdc/reef_top.sdc           # SDC for PD (sky130, 20 ns)
tests/                       # reef_smoke (program + Verilator testbench), sram_sky130, clock_gate_sky130
docs/                        # baseline-m3, reef-top, pd-drop0, dft-v0, area-feasibility, rtl-map-v0
results/                     # summaries of every verification and synthesis run quoted in docs/
```

## PD Drop 0

`docs/pd-drop0.md` lists what PD gets and how to read it. In short: top `reef_top`, filelist
`configs/filelists/reef_top_sky130.f` (set `REEF_ROOT` to this `hw/` directory), SDC
`configs/sdc/reef_top.sdc`, std cells `sky130_fd_sc_hd`, 5 x `sram22_512x128m4w8` TCM macros
(`scripts/fetch_sram22.sh` for the `.lib`/LEF/GDS), 2 x `sky130_fd_sc_hd__sdlclkp_4` clock gates.

## Quick start

```bash
scripts/lint_vendor.sh && scripts/lint_reef.sh                   # Verilator 5.032
SIM=verilator MODELS="sram22 cycle" scripts/run_sram_sky130_tb.sh
NODE=sky130 scripts/run_reef_smoke.sh     # Verilator >= 5.032, riscv-none-elf-gcc >= 14; ~10 min model build
OSS_CAD_SUITE=/path/to/oss-cad-suite scripts/synth_yosys.sh --lib sky130hd --top reef_top --node sky130
```

Use Verilator 5.032 (M3's version; `verify_upstream_release.sh` leaves it in the Bazel cache, and the
`verilator/verilator:v5.032` image has it). apt's Verilator reports false `BLKANDNBLK` errors in
the fpnew FPU. Yosys needs the slang plugin (OSS CAD Suite). The full design takes several GB to
elaborate.

Design Compiler (ECE servers; sky130 `.lib` -> `.db` with `lc_shell` `read_lib`/`write_lib`):

```bash
scripts/fetch_sram22.sh
scripts/synth_dc.sh --top reef_top --node sky130 --period 20 \
  --db sky130_fd_sc_hd__ss_100C_1v60.db --sram-db sram22_512x128m4w8_ss_100C_1v60.db
```

## CI

`.github/workflows/rtl.yml` runs on changes under `hw/`:

- **rtl-lint** (Verilator 5.032 container): `lint_vendor.sh`, `lint_reef.sh` (includes the
  filelist staleness check), the TCM wrapper unit test in Verilator, and `check_no_confidential.sh`.
- **rtl-synth** (OSS CAD Suite): Yosys synthesis of the Drop 0 build (`reef_top`, sky130, SRAM22),
  then `check_synth.py`: only `sky130_fd_sc_hd` cells plus the 5 TCM macros, 2 clock gates, and a
  standard-cell area budget. The area table goes to the job summary.

The smoke test (needs the RISC-V toolchain and a ~10 min model build), the VCS tests and the DC/PT
runs stay manual; their latest results are in `results/`.

## Open items

- SRAM22 `rstb` is tied high (no reset reaches the generated TCM modules); see `rtl/sram/README.md`.
- Reset/clock alignment with Lambda (`io_aresetn` async assert / sync deassert via `RstSync`).
- AXI interconnect hookup; CDC if the NPU clock differs; AXI ID width is 6 bits.
- `io_dm_*` tied off in `reef_top` (debug over AXI, CSR +0x800).

Baseline regression at M3: `core_mini_axi_sim_cocotb` 18/18 + `rvv_core_mini_axi_sim_cocotb` 18/18
PASS (`results/m3_verification.md`).
