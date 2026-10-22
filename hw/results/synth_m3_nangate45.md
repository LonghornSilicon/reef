# First area estimate: M3 RvvCoreMiniAxi (quick Yosys flow)

**Run:** 2026-10-01, `scripts/synth_yosys.sh --lib nangate45` at repo commit `068ee96` (scripts) on
the pinned baseline (M3-2026-04-27, `72d700e`, bundle sha256 `74d3bc7e…6f4ff4`).
Yosys 0.69+173 + yosys-slang (OSS CAD Suite 2026-10-01). 11 min wall, 7.3 GB peak RSS, exit 0.
Defines: `USE_GENERIC TB_SUPPORT VLEN_128 ZVE32F_ON SYNTHESIS`.

**Caveats (read before quoting these numbers):**
- NanGate45 open library, typical corner. Not our node (Reef is sky130, see `synth_m3_sky130hd.md`), so use the
  kGE column and the percentages to compare; absolute um^2 does not transfer.
- Generic `synth` + `abc` with no clock target and no SDC: no timing closure, no buffering for
  real wire loads. Expect real-node, timing-driven area to come out higher.
- Std cells only. The TCM SRAMs (`Sram` x2: ITCM 512x128 = 8 KB, DTCM 2048x128 = 32 KB) are
  blackboxed and are NOT in the totals. Add the memory compiler's macro area once PD has one.
- Includes the `TB_SUPPORT` cost (32-bit PC per RVV uop in the ROB->retire path), which M3 needs to
  elaborate. See `docs/baseline-m3.md`.
- Register files (scalar, FP, VRF) are synthesized as flops, not RF macros.

## Area by block

Liberty: `NangateOpenCellLibrary_typical.lib`, NAND2 = `NAND2_X1` (0.798 um^2). Std cells only; TCM SRAM macros are excluded.

| Block | Cells | Area (um^2) | % | kGE (NAND2-eq) | Seq % |
|---|---:|---:|---:|---:|---:|
| RVV vector unit | 793,337 | 1,198,750 | 83.6 | 1,502.2 | 25 |
| Scalar FPU | 17,928 | 26,001 | 1.8 | 32.6 | 25 |
| Scalar core (no FPU) | 131,403 | 184,696 | 12.9 | 231.4 | 19 |
| TCM logic (macros excluded) | 2,316 | 2,322 | 0.2 | 2.9 | 0 |
| AXI / fabric | 7,206 | 16,979 | 1.2 | 21.3 | 55 |
| Debug module | 1,720 | 2,613 | 0.2 | 3.3 | 33 |
| Top-level CSR / clock / reset / misc | 1,511 | 3,196 | 0.2 | 4.0 | 48 |
| **Total** | **955,421** | **1,434,557** | 100 | **1,797.7** | 24 |

### Blackboxed macros (not in totals)

- `Sram` x2

### Largest instances (depth <= 7, >= 1.0% of total)

| Instance path | Area (um^2) | % |
|---|---:|---:|
| `core` | 1,409,447 | 98.2 |
| `core.rvvCore` | 1,198,750 | 83.6 |
| `core.rvvCore.rvvCoreWrapper` | 1,198,648 | 83.6 |
| `core.rvvCore.rvvCoreWrapper.core` | 1,198,648 | 83.6 |
| `core.rvvCore.rvvCoreWrapper.core.backend` | 1,193,697 | 83.2 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_alu` | 107,901 | 7.5 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_alu.u_alu_cmp_unit` | 58,331 | 4.1 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_alu_rs` | 72,125 | 5.0 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_command_queue` | 15,459 | 1.1 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_decode_de2` | 15,020 | 1.0 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_dispatch` | 39,879 | 2.8 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_div` | 134,070 | 9.3 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_div_rs` | 17,591 | 1.2 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_fma` | 183,648 | 12.8 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_fma_rs` | 64,392 | 4.5 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_legal_command_queue` | 18,795 | 1.3 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_lsu_rs` | 17,717 | 1.2 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_mul_rs` | 52,935 | 3.7 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_mulmac` | 126,497 | 8.8 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_pmtrdt` | 50,075 | 3.5 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_pmtrdt_rs` | 30,128 | 2.1 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_rob` | 42,734 | 3.0 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_uop_queue` | 51,788 | 3.6 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_vrf` | 92,501 | 6.4 |
| `core.rvvCore.rvvCoreWrapper.core.backend.u_vrf.vrf_reg` | 29,417 | 2.1 |
| `core.score` | 210,697 | 14.7 |
| `core.score.fRegfile` | 16,622 | 1.2 |
| `core.score.fetch` | 15,788 | 1.1 |
| `core.score.floatCore` | 26,002 | 1.8 |
| `core.score.floatCore.floatCoreWrapper` | 24,913 | 1.7 |
| `core.score.floatCore.floatCoreWrapper.core` | 24,913 | 1.7 |
| `core.score.lsu` | 51,333 | 3.6 |
| `core.score.regfile` | 33,445 | 2.3 |
| `core.score.retirement_buffer` | 24,659 | 1.7 |
| `core.score.retirement_buffer.instBuffer` | 15,327 | 1.1 |
