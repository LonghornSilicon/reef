# Area variant: M3 RvvCoreMiniAxi without vector FP32 (`ZVE32F_ON` off)

**Run:** 2026-10-01 on koopa, `scripts/synth_yosys.sh --lib <lib> -UZVE32F_ON --exclude-list
configs/synth_exclude_nozve32f.txt` at repo commit `18cec03`, pinned baseline M3 `72d700e`.
Defines: `USE_GENERIC TB_SUPPORT VLEN_128 SYNTHESIS`. sky130hd 542 s / 6.4 GB, NanGate45 518 s /
6.4 GB, both exit 0. Same caveats as `synth_m3_sky130hd.md` (area-only mapping, std cells only).

How the variant is made: only the `ZVE32F_ON` define changes; no RTL edits. The 18 files in the
exclude list (vector FMA, FP reduction, FP div/sqrt incl. T-Head `ct_vfdsu`) are not instantiated
without the define, but yosys-slang checks every module in the filelist, so they are dropped from
the list. The list comes from Verilator `--json-only` with and without the define.
**Not functionally verified:** upstream calls this an untested config (`docs/baseline-m3.md`).

| | default (vector FP32 on) | no vector FP32 | change |
|---|---:|---:|---:|
| sky130hd total (mm^2) | 7.199 | 5.178 | -28% |
| sky130hd RVV unit (mm^2) | 6.005 | 3.983 | -34% |
| NanGate45 total (kGE) | 1,797.7 | 1,290.8 | -28% |

## sky130hd, by block

Liberty: `sky130_fd_sc_hd__tt_025C_1v80.lib`, NAND2 = `sky130_fd_sc_hd__nand2_1` (3.754 um^2). Std cells only; TCM SRAM macros are excluded.

| Block | Cells | Area (um^2) | % | kGE (NAND2-eq) | Seq % |
|---|---:|---:|---:|---:|---:|
| RVV vector unit | 493,720 | 3,983,106 | 76.9 | 1,061.1 | 21 |
| Scalar FPU | 16,563 | 130,051 | 2.5 | 34.6 | 25 |
| Scalar core (no FPU) | 125,510 | 938,085 | 18.1 | 249.9 | 18 |
| TCM logic (macros excluded) | 1,840 | 13,651 | 0.3 | 3.6 | 0 |
| AXI / fabric | 6,035 | 84,004 | 1.6 | 22.4 | 67 |
| Debug module | 1,549 | 12,970 | 0.3 | 3.5 | 38 |
| Top-level CSR / clock / reset / misc | 1,366 | 16,323 | 0.3 | 4.3 | 53 |
| **Total** | **646,583** | **5,178,190** | 100 | **1,379.5** | 21 |

### Blackboxed macros (not in totals)

- `Sram` x2

### Largest instances (depth <= 7, >= 1.0% of total)

| Instance path | Area (um^2) | % |
|---|---:|---:|
| `core` | 5,051,242 | 97.5 |
| `core.rvvCore` | 3,983,106 | 76.9 |
| `core.rvvCore.rvvCoreWrapper` | 3,982,566 | 76.9 |
| `core.rvvCore.rvvCoreWrapper.core` | 3,982,566 | 76.9 |
| `core.rvvCore.rvvCoreWrapper.core.backend` | 3,959,352 | 76.5 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_alu` | 545,109 | 10.5 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_alu.u_alu_cmp_unit` | 294,813 | 5.7 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_alu_rs` | 361,880 | 7.0 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_command_queue` | 75,799 | 1.5 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_decode_de2` | 64,872 | 1.3 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_dispatch` | 201,706 | 3.9 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_div` | 136,816 | 2.6 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_div_rs` | 88,291 | 1.7 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_legal_command_queue` | 96,474 | 1.9 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_lsu_rs` | 90,963 | 1.8 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_mul_rs` | 265,513 | 5.1 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_mulmac` | 634,342 | 12.3 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_pmtrdt` | 167,790 | 3.2 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_pmtrdt_rs` | 154,164 | 3.0 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_rob` | 164,520 | 3.2 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_rob.u_uop_info_fifo` | 61,898 | 1.2 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_uop_queue` | 251,155 | 4.9 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_vrf` | 450,099 | 8.7 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_vrf.vrf_reg` | 148,623 | 2.9 |
| `core.score` | 1,068,136 | 20.6 |
| `core.score.fRegfile` | 81,626 | 1.6 |
| `core.score.fetch` | 83,599 | 1.6 |
| `core.score.fetch.instructionBuffer` | 69,186 | 1.3 |
| `core.score.fetch.instructionBuffer.circularBuffer` | 69,051 | 1.3 |
| `core.score.floatCore` | 130,051 | 2.5 |
| `core.score.floatCore.floatCoreWrapper` | 124,559 | 2.4 |
| `core.score.floatCore.floatCoreWrapper.core` | 124,559 | 2.4 |
| `core.score.lsu` | 256,700 | 5.0 |
| `core.score.regfile` | 167,358 | 3.2 |
| `core.score.retirement_buffer` | 132,568 | 2.6 |
| `core.score.retirement_buffer.instBuffer` | 83,858 | 1.6 |

## NanGate45, by block

Liberty: `NangateOpenCellLibrary_typical.lib`, NAND2 = `NAND2_X1` (0.798 um^2). Std cells only; TCM SRAM macros are excluded.

| Block | Cells | Area (um^2) | % | kGE (NAND2-eq) | Seq % |
|---|---:|---:|---:|---:|---:|
| RVV vector unit | 536,492 | 794,194 | 77.1 | 995.2 | 22 |
| Scalar FPU | 17,952 | 26,034 | 2.5 | 32.6 | 25 |
| Scalar core (no FPU) | 131,517 | 184,675 | 17.9 | 231.4 | 19 |
| TCM logic (macros excluded) | 2,316 | 2,322 | 0.2 | 2.9 | 0 |
| AXI / fabric | 7,206 | 16,979 | 1.6 | 21.3 | 55 |
| Debug module | 1,732 | 2,629 | 0.3 | 3.3 | 32 |
| Top-level CSR / clock / reset / misc | 1,511 | 3,196 | 0.3 | 4.0 | 48 |
| **Total** | **698,726** | **1,030,029** | 100 | **1,290.8** | 22 |

### Blackboxed macros (not in totals)

- `Sram` x2

### Largest instances (depth <= 7, >= 1.0% of total)

| Instance path | Area (um^2) | % |
|---|---:|---:|
| `core` | 1,004,903 | 97.6 |
| `core.rvvCore` | 794,194 | 77.1 |
| `core.rvvCore.rvvCoreWrapper` | 794,093 | 77.1 |
| `core.rvvCore.rvvCoreWrapper.core` | 794,093 | 77.1 |
| `core.rvvCore.rvvCoreWrapper.core.backend` | 789,159 | 76.6 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_alu` | 107,275 | 10.4 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_alu.u_alu_cmp_unit` | 58,004 | 5.6 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_alu_rs` | 72,125 | 7.0 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_command_queue` | 15,109 | 1.5 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_decode_de2` | 14,268 | 1.4 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_dispatch` | 39,759 | 3.9 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_div` | 27,798 | 2.7 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_div_rs` | 17,168 | 1.7 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_legal_command_queue` | 18,521 | 1.8 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_lsu_rs` | 17,717 | 1.7 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_mul_rs` | 52,935 | 5.1 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_mulmac` | 126,273 | 12.3 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_pmtrdt` | 33,693 | 3.3 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_pmtrdt_rs` | 29,969 | 2.9 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_rob` | 32,567 | 3.2 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_rob.u_uop_info_fifo` | 12,387 | 1.2 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_uop_queue` | 50,579 | 4.9 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_vrf` | 92,501 | 9.0 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_vrf.vrf_reg` | 29,417 | 2.9 |
| `core.score` | 210,709 | 20.5 |
| `core.score.fRegfile` | 16,622 | 1.6 |
| `core.score.fetch` | 15,788 | 1.5 |
| `core.score.fetch.instructionBuffer` | 12,913 | 1.3 |
| `core.score.fetch.instructionBuffer.circularBuffer` | 12,885 | 1.3 |
| `core.score.floatCore` | 26,034 | 2.5 |
| `core.score.floatCore.floatCoreWrapper` | 24,946 | 2.4 |
| `core.score.floatCore.floatCoreWrapper.core` | 24,946 | 2.4 |
| `core.score.lsu` | 51,328 | 5.0 |
| `core.score.regfile` | 33,430 | 3.2 |
| `core.score.retirement_buffer` | 24,659 | 2.4 |
| `core.score.retirement_buffer.instBuffer` | 15,327 | 1.5 |

