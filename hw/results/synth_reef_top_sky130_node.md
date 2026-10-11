# Area estimate: reef_top from the PD Drop 0 filelist (sky130 TCMs + clock gate)

**Run:** 2026-10-04 on koopa, `scripts/synth_yosys.sh --lib sky130hd --top reef_top --node sky130`
(working tree on top of `8da6529`; the `--node` option is committed with this file), pinned M3
`72d700e`. Yosys 0.69+173 + yosys-slang (OSS CAD Suite 2026-10-01). 953 s, 7.3 GB peak RSS, exit 0.
Same flow and caveats as `synth_reef_top_sky130hd.md` (area-only mapping, tt, no timing).

What this run shows:

- `configs/filelists/reef_top_sky130.f` elaborates and synthesizes in Yosys with 0 errors and no
  unknown modules once both `.lib` files are read with `read_liberty -lib` (std cells +
  `sky130_sram_2kbyte_1rw1r_32x512_8_TT_1p8V_25C.lib`).
- The netlist has **2 `sky130_fd_sc_hd__sdlclkp_4`** (the two core clock gates) and **20
  `sky130_sram_2kbyte_1rw1r_32x512_8`** (ITCM 4, DTCM 16).
- Std cells: **7.250 mm^2** vs 7.240 mm^2 with `Sram` black boxes and the generic clock gate. The
  +0.011 mm^2 is the TCM wrapper logic (hold register, bank select, read mux: "TCM logic" 13,651 ->
  24,734 um^2). Macro area is not in the total: 20 x 0.285 = 5.69 mm^2 before halo.

## Area by block

Liberty: `sky130_fd_sc_hd__tt_025C_1v80.lib`, NAND2 = `sky130_fd_sc_hd__nand2_1` (3.754 um^2). Std cells only; TCM SRAM macros are excluded.

| Block | Cells | Area (um^2) | % | kGE (NAND2-eq) | Seq % |
|---|---:|---:|---:|---:|---:|
| RVV vector unit | 747,061 | 6,007,548 | 82.9 | 1,600.5 | 23 |
| Scalar FPU | 16,395 | 129,020 | 1.8 | 34.4 | 25 |
| Scalar core (no FPU) | 125,945 | 939,766 | 13.0 | 250.4 | 18 |
| TCM logic (macros excluded) | 2,495 | 24,734 | 0.3 | 6.6 | 21 |
| AXI / fabric | 6,055 | 84,141 | 1.2 | 22.4 | 67 |
| Debug module | 1,581 | 12,950 | 0.2 | 3.4 | 38 |
| Top-level CSR / clock / reset / misc | 1,361 | 16,302 | 0.2 | 4.3 | 53 |
| Reef shell (SPI bridge, DECERR sink, reset sync) | 2,333 | 36,030 | 0.5 | 9.6 | 72 |
| **Total** | **903,226** | **7,250,490** | 100 | **1,931.6** | 23 |

### Blackboxed macros (not in totals)

- none

### Macros (no area in the std-cell liberty, not in totals)

- `sky130_sram_2kbyte_1rw1r_32x512_8` x20

### Largest instances (depth <= 7, >= 1.0% of total)

| Instance path | Area (um^2) | % |
|---|---:|---:|
| `u_core` | 7,214,460 | 99.5 |
| `u_core.core` | 7,076,334 | 97.6 |
| `u_core.core.rvvCore` | 6,007,548 | 82.9 |
| `u_core.core.rvvCore.rvvCoreWrapper` | 6,007,007 | 82.8 |
| `u_core.core.rvvCore.rvvCoreWrapper.core` | 6,007,007 | 82.8 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend` | 5,983,539 | 82.5 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_alu` | 550,857 | 7.6 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_alu_rs` | 361,801 | 5.0 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_command_queue` | 77,838 | 1.1 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_dispatch` | 202,169 | 2.8 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_div` | 659,397 | 9.1 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_div_rs` | 90,148 | 1.2 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_fma` | 914,123 | 12.6 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_fma_rs` | 322,771 | 4.5 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_legal_command_queue` | 98,189 | 1.4 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_lsu_rs` | 90,960 | 1.3 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_mul_rs` | 265,726 | 3.7 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_mulmac` | 633,728 | 8.7 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_pmtrdt` | 250,614 | 3.5 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_pmtrdt_rs` | 154,931 | 2.1 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_rob` | 218,083 | 3.0 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_uop_queue` | 258,343 | 3.6 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_vrf` | 450,099 | 6.2 |
| `u_core.core.score` | 1,068,786 | 14.7 |
| `u_core.core.score.fRegfile` | 81,773 | 1.1 |
| `u_core.core.score.fetch` | 83,832 | 1.2 |
| `u_core.core.score.floatCore` | 129,020 | 1.8 |
| `u_core.core.score.floatCore.floatCoreWrapper` | 123,528 | 1.7 |
| `u_core.core.score.floatCore.floatCoreWrapper.core` | 123,528 | 1.7 |
| `u_core.core.score.lsu` | 256,725 | 3.5 |
| `u_core.core.score.regfile` | 167,454 | 2.3 |
| `u_core.core.score.retirement_buffer` | 132,401 | 1.8 |
| `u_core.core.score.retirement_buffer.instBuffer` | 83,374 | 1.1 |

## Design Compiler + PrimeTime: the full Drop 0 build at ss, 20 ns

**Run:** 2026-10-04 on koopa, `scripts/synth_dc.sh --top reef_top --node sky130 --period 20 --db
sky130_fd_sc_hd__ss_100C_1v60.db --sram-db sky130_sram_2kbyte_1rw1r_32x512_8_TT_1p8V_25C.db` at
`c51bdc5` (DC T-2022.03-SP2, 3947 s, exit 0; elaborate/link 0 errors), then `scripts/sta_pt.sh --top
reef_top` on the netlist at the same corner with the macro `.db` (PT T-2022.03-SP5, `test_en_i` = 0).
Constraints as in `synth_dc.sh` (5% uncertainty, IO 20%, wire-load model, ideal clock). The macro
`.db` is TT only.

| | |
|---|---|
| Std cells | 5.070 mm^2 (combinational 3.691, of which buf/inv 0.347; sequential 1.379) |
| Macros | 20 x `sky130_sram_2kbyte_1rw1r_32x512_8` = 5.691 mm^2 |
| **Total cell area** | **10.761 mm^2** (before utilization and halo) |
| ICG cells | 2 x `sky130_fd_sc_hd__sdlclkp_4`: `u_core/cg/u_cg`, `u_core/rst_sync/icg/u_cg` |
| Setup (DC and PT) | **met**, 0 violating endpoints in PT (recovery checks included); reg->reg slack +0.0002 ns, arrival 18.69 ns |
| Macro -> core (half cycle) | +0.007 ns: ITCM `g_bank[0].g_col[2].u_macro` -> `core/score/fetch/instructionBuffer/.../buffer_5_inst_reg[0]` |
| Core -> macro | +0.08 ns |
| ICG enable (GATE pins) | +17.26 ns (`csr/resetReg` -> `cg/u_cg`), +17.87 ns (`rst_sync` -> `icg/u_cg`) |
| Hold (pre-CTS) | worst -0.31 ns; for P&R |
| Max-transition violations | 582; for P&R buffering |

Same binding path as the core-only run with macros (`results/sram_sky130.md` section 2): the
half-cycle ITCM read into fetch, closed by DC with no margin against TT-only macro timing. The
sky130 clock gates add no timing risk.
