# Timing-driven synthesis: M3 RvvCoreMiniAxi, Design Compiler, sky130hd, 10 ns

**Run:** 2026-10-01 on koopa, `scripts/synth_dc.sh --db sky130_fd_sc_hd__tt_025C_1v80.db --period 10
--cores 16` at repo commit `96de17f`, pinned baseline M3 `72d700e`. Design Compiler T-2022.03-SP2,
`compile_ultra -no_autoungroup`, 5530 s wall. Library: sky130_fd_sc_hd tt 25C 1.8 V `.lib`, compiled
to `.db` with Library Compiler (`read_lib`/`write_lib`, 0 errors).
Defines: `USE_GENERIC TB_SUPPORT VLEN_128 ZVE32F_ON SYNTHESIS`. Elaborate/link: 0 errors, no
unresolved references (the script stops otherwise).

Constraints: one clock on `io_aclk`, 10 ns, 0.5 ns uncertainty; IO delays 2 ns (20%); 0.01 pF load.
Synthesis-only substitutions: `Sram` -> black box (`rtl/reef/synth/Sram_blackbox.v`, dont_touch);
`edff` -> `rtl/reef/synth/edff_dc.sv` (equivalent; works around a DC VER-37 crash on M3's
`edff` with a struct type parameter in `rvv_backend_div_unit`).

## Result

| | |
|---|---|
| Std-cell area | **5.052 mm^2** (combinational 3.672, sequential 1.380, of which buf/inv 0.340) |
| Cells | 675,877 leaf cells, 55,624 sequential |
| Setup, all paths | **met**, WNS 0.00 ns, TNS 0 (DC optimizes up to the target, so slack 0 means "just met") |
| Setup, reg-to-reg (PrimeTime on the netlist) | met, slack 0.00 ns. Worst path inside the vector FMA's FP conversion pipeline (`u_fma/.../cvt`), arrival 9.41 ns |
| Hold | 19,102 violations, worst -0.16 ns: expected with ideal clocks before CTS; PD fixes hold |
| DRC | 203 max-transition, 1 max-cap violations |

**What this does and does not say:** at the typical corner, with a wire-load model and an ideal
clock, the M3 core synthesizes to about 5 mm^2 of sky130 standard cells at 10 ns with no margin.
Real wires, clock skew and the slow corner will eat into that, so 10 ns on sky130 should be treated
as optimistic until PD runs placement and STA.

## Area by block (DC hierarchy)

| Block | mm^2 | % |
|---|---:|---:|
| RVV (`core/rvvCore`) | 3.999 | 79.1 |
|   vector FMA `u_fma` + `u_fma_rs` | 0.891 | 17.6 |
|   MUL/MAC `u_mulmac` + `u_mul_rs` | 0.673 | 13.3 |
|   ALU `u_alu` + `u_alu_rs` | 0.540 | 10.7 |
|   divider `u_div` + `u_div_rs` | 0.395 | 7.8 |
|   VRF `u_vrf` | 0.363 | 7.2 |
|   permute/reduce `u_pmtrdt` + rs | 0.294 | 5.8 |
|   rest: uop queue 0.177, ROB 0.144, dispatch 0.130, command queues 0.114, LSU 0.085, RVV frontend/wrapper and small units | 0.843 | 16.7 |
| Scalar core (`core/score`, incl. FPU) | 0.947 | 18.8 |
|   LSU 0.247, regfile 0.168, retirement buffer 0.111, scalar FPU 0.100, fetch 0.073 | | |
| AXI slave | 0.042 | 0.8 |
| Debug module | 0.012 | 0.2 |
| **Total** | **5.052** | 100 |
