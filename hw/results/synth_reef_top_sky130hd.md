# Area estimate: whole reef_top (M3 core + Reef shell) on sky130hd

**Run:** 2026-10-01 on koopa, `scripts/synth_yosys.sh --lib sky130hd --top reef_top` at repo commit
`43f8275`, pinned M3 `72d700e`. 730 s, 7.3 GB peak RSS, exit 0. Same flow and caveats as
`synth_m3_sky130hd.md` (area-only mapping, std cells only, SRAMs blackboxed). This also shows the
full test-chip top elaborates and synthesizes in yosys-slang.

reef_top = 7.240 mm^2; the core alone = 7.199 mm^2; the Reef shell = 0.036 mm^2 (0.5%).

## Area by block

Liberty: `sky130_fd_sc_hd__tt_025C_1v80.lib`, NAND2 = `sky130_fd_sc_hd__nand2_1` (3.754 um^2). Std cells only; TCM SRAM macros are excluded.

| Block | Cells | Area (um^2) | % | kGE (NAND2-eq) | Seq % |
|---|---:|---:|---:|---:|---:|
| RVV vector unit | 747,167 | 6,007,791 | 83.0 | 1,600.5 | 23 |
| Scalar FPU | 16,395 | 129,020 | 1.8 | 34.4 | 25 |
| Scalar core (no FPU) | 125,945 | 939,766 | 13.0 | 250.4 | 18 |
| TCM logic (macros excluded) | 1,840 | 13,651 | 0.2 | 3.6 | 0 |
| AXI / fabric | 6,055 | 84,141 | 1.2 | 22.4 | 67 |
| Debug module | 1,581 | 12,950 | 0.2 | 3.4 | 38 |
| Top-level CSR / clock / reset / misc | 1,365 | 16,312 | 0.2 | 4.3 | 53 |
| Reef shell (SPI bridge, DECERR sink, reset sync) | 2,333 | 36,030 | 0.5 | 9.6 | 72 |
| **Total** | **902,681** | **7,239,660** | 100 | **1,928.7** | 23 |

### Blackboxed macros (not in totals)

- `Sram` x2

### Largest instances (depth <= 7, >= 1.0% of total)

| Instance path | Area (um^2) | % |
|---|---:|---:|
| `u_core` | 7,203,630 | 99.5 |
| `u_core.core` | 7,076,577 | 97.7 |
| `u_core.core.rvvCore` | 6,007,791 | 83.0 |
| `u_core.core.rvvCore.rvvCoreWrapper` | 6,007,250 | 83.0 |
| `u_core.core.rvvCore.rvvCoreWrapper.core` | 6,007,250 | 83.0 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend` | 5,983,781 | 82.7 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_alu` | 550,857 | 7.6 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_alu_rs` | 361,801 | 5.0 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_command_queue` | 77,838 | 1.1 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_dispatch` | 202,169 | 2.8 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_div` | 659,397 | 9.1 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_div_rs` | 90,148 | 1.2 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_fma` | 914,366 | 12.6 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_fma_rs` | 322,771 | 4.5 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_legal_command_queue` | 98,189 | 1.4 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_lsu_rs` | 90,960 | 1.3 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_mul_rs` | 265,726 | 3.7 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_mulmac` | 633,728 | 8.8 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_pmtrdt` | 250,614 | 3.5 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_pmtrdt_rs` | 154,931 | 2.1 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_rob` | 218,083 | 3.0 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_uop_queue` | 258,343 | 3.6 |
| `u_core.core.rvvCore.rvvCoreWrapper.core.backend.u_vrf` | 450,099 | 6.2 |
| `u_core.core.score` | 1,068,786 | 14.8 |
| `u_core.core.score.fRegfile` | 81,773 | 1.1 |
| `u_core.core.score.fetch` | 83,832 | 1.2 |
| `u_core.core.score.floatCore` | 129,020 | 1.8 |
| `u_core.core.score.floatCore.floatCoreWrapper` | 123,528 | 1.7 |
| `u_core.core.score.floatCore.floatCoreWrapper.core` | 123,528 | 1.7 |
| `u_core.core.score.lsu` | 256,725 | 3.5 |
| `u_core.core.score.regfile` | 167,454 | 2.3 |
| `u_core.core.score.retirement_buffer` | 132,401 | 1.8 |
| `u_core.core.score.retirement_buffer.instBuffer` | 83,374 | 1.2 |
