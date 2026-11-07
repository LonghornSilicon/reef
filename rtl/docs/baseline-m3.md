# Baseline: CoralNPU M3-2026-04-27

The Reef core baseline is CoralNPU release `M3-2026-04-27`, top `RvvCoreMiniAxi`: scalar core,
RVV 1.0 (VLEN 128, vector FP32), scalar FPU, AXI4 128-bit. Target `coral_m3`.

| | |
|---|---|
| Release | [`M3-2026-04-27`](https://github.com/google-coral/coralnpu/releases/tag/M3-2026-04-27), commit `72d700ef724669163e1b6f88e8622676ef6b0a74` (2026-04-24) |
| Source | `src/reef/`, a plain copy of the tag (below) |
| Release asset | `RvvCoreMiniAxi.sv`, sha256 `74d3bc7e3295353fd879f70a513f7c3fde79912010a1f62f08d815fbcd6f4ff4` |
| Generated SV | `src/reef/gen/`, bundle sha256 in `gen/BUNDLE.sha256` |
| Chisel flags | `--enableFetchL0=False --fetchDataBits=128 --lsuDataBits=128 --enableRvv=True --enableFloat=True --useAxi` (Bazel target `//hdl/chisel/src/coralnpu:rvv_core_mini_axi_cc_library_emit_verilog`) |
| Defines | `USE_GENERIC TB_SUPPORT VLEN_128 ZVE32F_ON`, plus `SYNTHESIS` for synthesis |
| Core boundary | `RvvCoreMiniAxi_ports.csv`, 397 ports |

Not in M3: the VME/Zvt matrix engine (upstream main only) and L1 I/D caches (the Chisel exists,
this top does not instantiate them).

## What src/reef holds

`git archive` of the tag, no edits on copy:

- `hdl/chisel/src` (scalar core, buses, soc), `hdl/verilog` without `rvv/sve` (RVV backend,
  `Sram.v`, `ClockGate.sv`, `RstSync.sv`)
- `rules/`, `toolchain/`, `platforms/`, `third_party/`, `WORKSPACE`, `.bazelrc`,
  `.bazelversion`, `BUILD.bazel`
- `tests/cocotb`, `coralnpu_test_utils`, `doc/`, `LICENSE`
- 8 files from directories otherwise left out, because the emit build loads them
  (`bazel query 'buildfiles(deps(...))'` on the full tag): `lib/BUILD`, `utils/BUILD`,
  `utils/scm_info.py`, and what `rules/repos.bzl` points at from `WORKSPACE`: `fpga/BUILD`,
  `fpga/0001-Export-hw-ip_templates.patch`, `fpga/0002-Use-hermetic-verilator-in-fusesoc-build.patch`,
  `fpga/ip/ispyocto/BUILD`, `fpga/ip/ispyocto/ispyocto.BUILD`

Left out: the rest of `fpga/` and `utils/`, `sw/`, `hw_sim/`, `examples/`, `lib/` beyond its
BUILD, `hdl/verilog/rvv/sve`, `tests/uvm`, `tests/vcs_sim`, `tests/verilator_sim`,
`tests/systemc`.

## Reef changes on top of M3

`tools/verify_upstream_release.sh` lists them. Every one needs a line here.

| File | Change | Why |
|---|---|---|
| `hdl/chisel/src/coralnpu/float/FloatCore.scala:113` | FPU `PipeRegs: '{default: 'd3}` written as one `'{default: 'd3}` per op group | Verilator (5.032 to 5.053) puts 3 only in the last format slot, so the FP32 FMA builds with 0 pipeline stages in sim. slang and DC already read 3 everywhere; with this all three agree and the synthesized netlists don't change. Lands as `gen/FloatCoreWrapper.sv:38`. |

## The generated files

firtool ships the design as one bundle; `tools/split_generated_sv.py` splits it at firtool's own
file markers:

| | Count | |
|---|---|---|
| Design | 119 | 4 firtool output (`RvvCoreMiniAxi.sv`, `RvvCoreWrapper.sv`, `FloatCoreWrapper.sv`, `Aligner_119_4.sv`), 115 mirrors of upstream's hand-written SV |
| Headers | 10 | In the filelist in bundle order: fpnew uses macros from `registers.svh` without including it |
| Verification | 26 | firtool assert/assume/cover layers, in `gen/sim.f` only |

`filelists/coral_m3.f` is the design in compile order, one compilation unit. Give it to tools as is.

The core reports a Coral revision in its `kscm0`..`kscm4` CSRs, which the Chisel takes from
Bazel's workspace status. `regen_coralnpu.sh` pins it to the M3 commit so the SV never depends on
the reef commit. That is also why an unmodified copy reproduces the release asset exactly.

## Defines

| Define | Sim | Synth | Why |
|---|---|---|---|
| `USE_GENERIC` | yes | yes | generic branches of `Sram.v` and `ClockGate.sv` |
| `TB_SUPPORT` | yes | **yes** | Not testbench-only in M3: it adds `uop_pc` / `last_uop_valid` to the RVV ROB-to-retire struct, which the scalar `RetirementBuffer` matches on. Without it `RvvCoreWrapper.sv` does not elaborate. Costs a 32-bit PC per RVV uop. |
| `VLEN_128` | yes | yes | vector length |
| `ZVE32F_ON` | yes | yes | vector FP32. Dropping it is an Arch area decision and an untested config |
| `SYNTHESIS` | | yes | removes sim-only init and assert code |

## Checked

- An unmodified copy builds to the release asset byte for byte, also from an empty Bazel cache
  (`verify_upstream_release.sh --build`).
- `regen_coralnpu.sh --check` passes on this tree from an empty Bazel cache, and fails when a
  file in `gen/` is edited by hand.
- `make check`: Verilator `-Wall` lint of `coral_m3` with the defines above passes.
- At M3 (2026-09-23, on the tag's own Bazel tree): cocotb `core_mini_axi_sim_cocotb` 18/18 and
  `rvv_core_mini_axi_sim_cocotb` 18/18. Not rerun on `src/reef`.
