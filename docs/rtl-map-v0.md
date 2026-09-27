# Reef RTL Map v0 — Core & Compute

**Baseline:** CoralNPU `M3-2026-04-27` (commit `72d700e`)
**Top assumed:** `RvvCoreMiniAxi` (scalar + RVV + FPU, AXI). GEMM on M3 runs on RVV, so the scalar-only `CoreMiniAxi` is not enough.
**Status:** v0 draft, 2026-09-27, Aaron Chien. Every number below was measured from the M3 sources and the generated `RvvCoreMiniAxi.sv`, not taken from docs.

## Rules for this map

**SystemVerilog first.** Nobody writes or edits Chisel in v1. We freeze the SystemVerilog generated from M3 and work only in SV.

| Label | Meaning |
|---|---|
| **SV (Chisel)** | Source language: SystemVerilog generated from Chisel. We work on the frozen SV; Chisel is reference only. |
| **SV** | Source language: hand-written SystemVerilog upstream. Can be edited directly. |
| **Reuse** | Use the frozen SV as-is. |
| **Wrap** | Keep as-is and handle it in our own hand-written SV around it (tie-offs, adapters, `reef_top`). |
| **Modify (SV)** | Edit hand-written upstream SV directly. |
| **SystemVerilog (Chisel)** | The natural fix is in Chisel or a generator flag. Per the SV-first rule, we do it in SV instead: a wrapper, a new SV block, or a scripted patch on the generated SV. Chisel stays as reference. |
| **Not used / New** | Not in our build; anything needed is new SV RTL. |

Lines = `wc -l`, tests excluded. "Gen" = lines of that block's modules inside `RvvCoreMiniAxi.sv` (83,147 lines total, 201 modules).

## Summary

| # | Block | Source language | Lines (source → gen) | Action | Owner (proposed) |
|---|---|---|---|---|---|
| 1 | Scalar core (incl. scalar FPU) | SV (Chisel); FPU = SV (Chisel) wrapper + third-party SV | ~5.7k Chisel → ~17.0k gen core + ~15.7k gen FPU | **Reuse**; any ISA/datapath change = **SystemVerilog (Chisel)** | Krish; Aaron review |
| 2 | RVV | Frontend SV (Chisel); backend **SV** | 1.3k Chisel + 32.1k SV → ~36.6k gen | **Reuse**; GEMM-related changes = **Modify (SV)** in the backend | Krish (datapath), Alex (MAC/GEMM) |
| 3 | VME / Zvt | — | 0 (**not in M3**) | **Not used / New** | Alex; Aaron review |
| 4 | L1 I/D caches | Chisel only | 1.0k Chisel → **0 gen (not instantiated)** | **Not used / New** | Aarush (analysis) + Arch |
| 5 | TCM (ITCM/DTCM) | SV (Chisel) wrapper + **SV** `Sram.v` | 212 Chisel + 269 SV → 861 gen | **Modify (SV)** for macros; size change = **SystemVerilog (Chisel)** | Aarush + PD |
| 6 | Fabric | SV (Chisel) | 130 → 225 gen | **Reuse**; memory-map change = **SystemVerilog (Chisel)** | Peter |
| 7 | AXI (slave + master bridges) | SV (Chisel) | 588 → 1,156 gen | **Reuse** + **Wrap** at `reef_top` | Peter; shell owner TBD (was Pranav) |
| 8 | CSR (control/boot) | SV (Chisel) | 221 → 208 gen | **Reuse**; new control regs = **SystemVerilog (Chisel)** as a separate SV block | Peter |
| 9 | Debug | SV (Chisel) | 350 → 484 gen | **Reuse** + **Wrap** (tie off `io_dm_*` and `io_debug_*`) | Peter decides; Arya tests |
| 10 | Clock gate | **SV** | 75 SV (28-line Chisel BlackBox) → 56 gen | **Modify (SV)**: add our node's ICG cell | PD + Peter |
| 11 | Reset sync | **SV** | 63 SV (31-line Chisel BlackBox) → 45 gen | **Reuse**; DFT check with PD | Peter + PD |
| — | Assertions (sim only) | SV (Chisel) | → 2.4k gen (23 modules, 26 files) | **Exclude from synthesis filelist** | Regen owner TBD (was Samuel) + Arya |

## Block details

### 1. Scalar core
- **What:** `SCore`: fetch (`UncachedFetch`, `FetchControl`, `InstructionBuffer`), decode/dispatch (`DispatchV2`), ALU, BRU, MLU (multiply), DVU (divide), LSU (`LsuV2`), integer and float regfiles, `RetirementBuffer`, `FaultManager`, machine CSRs (`Csr`). The scalar FPU is `FloatCore`, which wraps third-party fpnew plus the T-Head `pa_fdsu` divider/sqrt.
- **Biggest modules:** `LsuV2` 3,779 · `DispatchV2` 2,466 · `SCore` 2,145 · `Regfile` 2,095 · `RetirementBuffer` 1,341 lines.
- **Gaps:**
  - The generated SV is machine-named firtool output in one 83k-line file. Hand edits are fragile and are lost on every regen. **Any change must be a scripted patch** (`rtl/patches/`, applied by `scripts/regen_coralnpu.sh`), never a manual edit.
  - The build has `--enableFloat=True`. Decide whether Reef needs scalar F. Dropping it saves about 15.7k lines of FPU but is a generator flag, so it's **SystemVerilog (Chisel)**. If we keep M3's SV as-is, the FPU stays in.
  - Krish's ISA-extension work is analysis only in v1. Any implementation would be **SystemVerilog (Chisel)**.
  - Krish should document LSU back-pressure and the retirement path, since those set the RVV/GEMM data rate.

### 2. RVV
- **What:** Chisel frontend (`RvvCoreWrapper`, `RvvCoreShim`, `RvvDecode`) feeding a **hand-written SV backend** under `hdl/verilog/rvv/`: 54 design files, 29,266 lines, plus 2,869 lines of `common/` + `inc/`. The backend covers decode, dispatch, ROB, retire, VRF, ALU, MUL/MAC, DIV, permute/reduce, FMA, and T-Head `ct_vfdsu` vector div/sqrt.
- **GEMM-relevant:** `rvv_backend_mac_unit` (983 gen lines), `rvv_backend_mulmac`, `rvv_backend_mul_unit_mul8`. The backend is already SV, so **GEMM changes can go straight into SV here**.
- **Gaps:**
  - **Define set:** the tested sim build uses `-DUSE_GENERIC -DTB_SUPPORT -DVLEN_128 -DZVE32F_ON` (`hdl/chisel/src/coralnpu/BUILD:654-658`). **Update 9/27: `TB_SUPPORT` must stay on for synthesis too.** It is not testbench-only: it adds `uop_pc`/`last_uop_valid` to the ROB→retire struct, which the scalar `RetirementBuffer` uses to retire vector ops, and without it `RvvCoreWrapper.sv` doesn't elaborate. The synth set (`+SYNTHESIS`) lints clean. See `docs/baseline-m3.md`.
  - `ZVE32F_ON` (vector float) is an area decision. `VLEN=128` is fixed. `ZVFBFWMA_ON` (bf16 widening MAC) is off.

### 3. VME / Zvt
- **Not in M3.** No Zvt/VME files exist at `72d700e`. Upstream main has it (`hdl/verilog/rvv/design/Zvt/`, 20 SV files, ~4.7k lines), but it isn't part of our baseline.
- **Action:** any dedicated GEMM engine is **new SV** attached to the RVV backend. Upstream Zvt can be read as a design reference only.
- **Gaps:**
  - The "v1 boundary" for Alex's prototype isn't defined yet.
  - We need the **RVV-only matmul baseline** first. M3 ships the int8/float kernels in `tests/cocotb/rvv/ml_ops/` and `tests/cocotb/rvv_ml_ops_cocotb_test.py`. Measure cycles/MAC and bandwidth, compare against Richard's workload models, and let that decide whether a GEMM unit is worth building.

### 4. L1 I/D caches
- **Not instantiated.** `L1ICache.scala` (277) and `L1DCache.scala` (726) exist, but neither `RvvCoreMiniAxi.sv` nor `CoreMiniAxi.sv` contains them (0 references). Fetch comes from ITCM through `UncachedFetch` (`--enableFetchL0=False`), and data goes to DTCM or out over AXI.
- **Correction:** earlier notes listed "8KB I-cache / 16KB D-cache" as an SRAM item. That's wrong for our build.
- **Action:** if Arch wants a cache, it's **new SV**. Aarush's cache analysis should start from "no cache today".

### 5. TCM (ITCM / DTCM)
- **What:** ITCM 8 KB = `TCM128` → `SRAM_512x128` → `Sram`. DTCM 32 KB = `SRAM_2048x128` → `Sram`. Single port, synchronous read, 128-bit data, 16-bit byte mask.
- **`hdl/verilog/Sram.v` (SV) already switches on macros:** `USE_TSMC12FFC`, `USE_GF22`, else behavioral.
- **Action:** **Modify (SV)** by adding a `USE_<our node>` branch that instantiates our compiled macros (512x128 and 2048x128 with byte mask). No Chisel needed.
- **Gaps:**
  - The node is still undecided (TSMC16 vs 65nm), and so are the macro names and ports.
  - Changing TCM depth needs the `--itcmSizeKBytes/--dtcmSizeKBytes` flags → **SystemVerilog (Chisel)**. M3 already has 512 KB and 1 MB targets in `BUILD` if we ever need them.
  - `Sram_1rw_256x256.v` and `Sram_1rwm_256x288.v` are unused in our build.

### 6. Fabric
- **What:** `FabricMux` / `FabricArbiter` route the core's data bus by address:
  - ITCM `0x0000_0000` (8 KB)
  - DTCM `0x0001_0000` (32 KB)
  - CSR `0x0003_0000` (4 KB)
  - everything else → AXI master
- **Action:** **Reuse**. Changing the memory map is **SystemVerilog (Chisel)**. Avoid it in v1 and remap in `reef_top` instead.

### 7. AXI
- **What:**
  - `AxiSlave`: external masters → TCM/CSR.
  - `DBus2AxiV2`, `IBus2Axi`: core → external master.
  - Ports: 32-bit address, **128-bit data**, **6-bit ID**. The master ID is always 0, so there's no reordering.
- **Action:** **Reuse** + **Wrap** in `reef_top`.
- **Gaps:**
  - Everything runs on one clock (`io_aclk`). Any CDC to the SoC fabric has to be our SV, outside the core.
  - We still have to match the SoC interconnect: width, AXI4 vs AXI-Lite, ID width.
  - The existing `rtl/coralnpu_wrapper.sv` was generated for scalar `CoreMiniAxi` at `0fc715e` (205 ports). `RvvCoreMiniAxi` has **397** port names, so the wrapper has to be regenerated.

### 8. CSR (control / boot)
- **What:** `CoreAxiCSR` at CSR base `0x0003_0000`: `RESET` +0x0, `PC_START` +0x4, `STATUS` +0x8, general regs from +0x100, debug window +0x800 to +0x814. Boot sequence: write `PC_START` → clear clock gate → wait → clear `RESET` → poll `STATUS`.
- **Also:** `io_boot_addr[31:0]` is a top-level pin, and `PC_START` loads from it on the first clock after reset. A ROM boot path can tie it to the ROM base, with no CSR write needed.
- **Action:** **Reuse**. If Reef needs extra control registers, add them as **a separate SV block** in `reef_top` → **SystemVerilog (Chisel)**. Don't patch `CoreAxiCSR`.

### 9. Debug
- **What:** `DebugModule` (from `scalar/Debug.scala`) can be reached **two ways**, arbitrated inside `CoreAxi.scala:71-73`:
  - the `io_dm_*` port
  - the CSR debug window over the AXI slave
- **Result:** **v1 does not need a JTAG DTM.** Tie off `io_dm_*` (`req_valid=0`, `rsp_ready=1`) and debug through AXI → CSR. Post-silicon debug then just needs some path to drive the AXI slave, such as a host or JTAG-to-AXI.
- **Also:** `io_debug_*` is **300 output ports** of sim trace. Leave them unconnected in `reef_top`. Full list: `results/RvvCoreMiniAxi_ports.csv`.
- **Decision for Peter:** confirm the AXI-only debug path is acceptable for bring-up.

### 10. Clock gate
- **What:** `hdl/verilog/ClockGate.sv` (SV) picks a cell per node: `USE_TSMC12FFC` (`CKLNQD10…`), `USE_GF22*`, `FPGA_XILINX`, else a behavioral latch. It gates the core from the CSR clock-gate bit and inside `RstSync`, and has a `te` input.
- **Action:** **Modify (SV)** by adding a `USE_<our node>` branch with our std-cell ICG. PD supplies the cell name.
- **Note:** T-Head's `gated_clk_cell` inside the FPU is a pass-through (`clk_out = clk_in`), so there's no hidden gating there.

### 11. Reset sync
- **What:** `hdl/verilog/RstSync.sv` (SV), instantiated in `CoreAxi.scala:49`. `io_aresetn` asserts asynchronously and deasserts through a 2-flop sync. The clock enable follows after 2 more cycles through `ClockGate`. Internal logic uses synchronous reset.
- **Action:** **Reuse**.
- **Gap:** there's no test-mode reset bypass. Check with PD whether DFT needs scan control of this reset, and drive `io_te` correctly (never tie it to 0 blindly).

### Assertions (not a block, but a synthesis blocker)
- `RvvCoreMiniAxi.sv` is **154 files concatenated** (`// ----- 8< ----- FILE "..."` markers). 26 of them are under `verification/` and add 23 `*_Verification_Assert` modules through `bind`.
- **Action:** the regen script splits the file on those markers and **leaves `verification/*` out of the synthesis filelist**. DV can keep them for sim.

## Top priorities from this map
1. ~~Switch the integration repo to `RvvCoreMiniAxi` at M3~~ **Done 9/27:** `rtl/vendor/coralnpu/` + `synth.f`/`sim.f`; the 397-port list is in `results/`, and the wrapper itself belongs to the shell owner.
2. ~~Lint + regression with the synth define set~~ **Done 9/27:** lint passes for both sets (`scripts/lint_vendor.sh`); synth keeps `TB_SUPPORT`.
3. **RVV matmul baseline numbers** → go/no-go on a GEMM unit (Alex).
4. **Node decision → `USE_<node>` branches** in `Sram.v` and `ClockGate.sv` (Aarush + PD).
5. **Confirm AXI-only debug** and tie off `io_dm_*` / `io_debug_*` (Peter).

## Open decisions
- `RvvCoreMiniAxi` vs `CoreMiniAxi`: this map assumes RVV.
- Keep scalar F (`enableFloat`) and vector F (`ZVE32F_ON`), or drop them for area.
- Who owns regen and the `reef_top` shell now that Samuel and Pranav aren't on the latest role table.
