# Area estimate: M3 RvvCoreMiniAxi on sky130hd (quick Yosys flow)

**Run:** 2026-10-01 on koopa, `scripts/synth_yosys.sh --lib sky130hd` at repo commit `6bd058f` on the
pinned baseline (M3-2026-04-27, `72d700e`, bundle sha256 `74d3bc7e…6f4ff4`). Yosys 0.69+173 +
yosys-slang (OSS CAD Suite 2026-10-01). 671 s wall, 7.3 GB peak RSS, exit 0.
Defines: `USE_GENERIC TB_SUPPORT VLEN_128 ZVE32F_ON SYNTHESIS`. Liberty: sky130_fd_sc_hd tt 25C 1.8V.

**Caveats (same as `synth_m3_nangate45.md`):** area-only mapping (no clock target, no SDC, no
buffering for wire load), std cells only (TCM SRAMs blackboxed), register files as flops. This is a
**lower bound**: timing-driven synthesis and placement grow it. Compare with
`synth_m3_sky130hd_10ns.md` for a 10 ns target.

## Area by block

Liberty: `sky130_fd_sc_hd__tt_025C_1v80.lib`, NAND2 = `sky130_fd_sc_hd__nand2_1` (3.754 um^2). Std cells only; TCM SRAM macros are excluded.

| Block | Cells | Area (um^2) | % | kGE (NAND2-eq) | Seq % |
|---|---:|---:|---:|---:|---:|
| RVV vector unit | 746,667 | 6,004,984 | 83.4 | 1,599.8 | 23 |
| Scalar FPU | 16,582 | 129,965 | 1.8 | 34.6 | 25 |
| Scalar core (no FPU) | 125,292 | 936,840 | 13.0 | 249.6 | 18 |
| TCM logic (macros excluded) | 1,840 | 13,651 | 0.2 | 3.6 | 0 |
| AXI / fabric | 6,035 | 84,004 | 1.2 | 22.4 | 67 |
| Debug module | 1,566 | 12,916 | 0.2 | 3.4 | 38 |
| Top-level CSR / clock / reset / misc | 1,366 | 16,323 | 0.2 | 4.3 | 53 |
| **Total** | **899,348** | **7,198,683** | 100 | **1,917.8** | 23 |

### Blackboxed macros (not in totals)

- `Sram` x2

### Largest instances (depth <= 7, >= 1.0% of total)

| Instance path | Area (um^2) | % |
|---|---:|---:|
| `core` | 7,071,789 | 98.2 |
| `core.rvvCore` | 6,004,984 | 83.4 |
| `core.rvvCore.rvvCoreWrapper` | 6,004,444 | 83.4 |
| `core.rvvCore.rvvCoreWrapper.core` | 6,004,444 | 83.4 |
| `core.rvvCore.rvvCoreWrapper.core.backend` | 5,980,977 | 83.1 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_alu` | 549,277 | 7.6 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_alu.u_alu_cmp_unit` | 296,896 | 4.1 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_alu_rs` | 361,880 | 5.0 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_command_queue` | 77,761 | 1.1 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_dispatch` | 201,867 | 2.8 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_div` | 660,004 | 9.2 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_div_rs` | 90,175 | 1.3 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_fma` | 911,576 | 12.7 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_fma_rs` | 322,546 | 4.5 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_legal_command_queue` | 98,223 | 1.4 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_lsu_rs` | 90,963 | 1.3 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_mul_rs` | 265,513 | 3.7 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_mulmac` | 635,829 | 8.8 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_pmtrdt` | 250,259 | 3.5 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_pmtrdt_rs` | 154,885 | 2.2 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_rob` | 218,856 | 3.0 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_uop_queue` | 257,513 | 3.6 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_vrf` | 450,099 | 6.3 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_vrf.vrf_reg` | 148,623 | 2.1 |
| `core.score` | 1,066,804 | 14.8 |
| `core.score.fRegfile` | 81,626 | 1.1 |
| `core.score.fetch` | 83,599 | 1.2 |
| `core.score.floatCore` | 129,965 | 1.8 |
| `core.score.floatCore.floatCoreWrapper` | 124,473 | 1.7 |
| `core.score.floatCore.floatCoreWrapper.core` | 124,473 | 1.7 |
| `core.score.lsu` | 255,529 | 3.5 |
| `core.score.regfile` | 167,284 | 2.3 |
| `core.score.retirement_buffer` | 132,568 | 1.8 |
| `core.score.retirement_buffer.instBuffer` | 83,858 | 1.2 |
