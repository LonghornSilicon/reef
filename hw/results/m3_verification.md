# M3-2026-04-27 baseline verification (2026-09-23)

Candidate: official CoralNPU release `M3-2026-04-27` = commit `72d700ef724669163e1b6f88e8622676ef6b0a74`
(2026-04-24, ~310 commits behind `main` 5310635 as of 9/23). Reproduce with
`scripts/verify_upstream_release.sh M3-2026-04-27`.

| Check | Result |
|---|---|
| Dev container from M3's own `utils/coralnpu.dockerfile` | **Fails**: Bazel apt key URL `bazel.build/bazel-release.pub.gpg` is dead. Fixed copy (URL only): `configs/coralnpu-m3-fixedkey.dockerfile` |
| Emit `CoreMiniAxi` + `RvvCoreMiniAxi` Verilog (Bazel 7.4.1) | OK |
| Generated `RvvCoreMiniAxi.sv` vs release asset | **Byte-identical** (sha256 `74d3bc7e…6f4ff4`) |
| cocotb `core_mini_axi_sim_cocotb` (Verilator) | **18/18 PASS** (`rvv_exceptions_test` is a 0-cycle no-op on the scalar core) |
| cocotb `rvv_core_mini_axi_sim_cocotb` (Verilator) | **18/18 PASS** |
| `CoreMiniAxi` top-level ports vs `0fc715e` | 205 vs 205, same names. Only change: `io_debug_float_writeData_{0,1}_bits_addr` is `[31:0]` in M3 vs `[4:0]` in 0fc715e (sim debug bundle) |

Note: M3's regression has 18 testcases; `0fc715e` has 26 (Google added tests after April).
Build note: WSL has ~7.6 GB RAM; the script caps Docker at 5 GB and Bazel at 4 jobs.

## cocotb summaries

### core_mini_axi_sim_cocotb
```
** core_mini_axi_sim.core_mini_axi_basic_write_read_memory PASS 1171148.75 36.65 31958.79 **
** core_mini_axi_sim.core_mini_axi_run_wfi_in_all_slots PASS 1168.75 0.03 35384.72 **
** core_mini_axi_sim.core_mini_axi_slow_bready PASS 1382.50 0.05 30541.26 **
** core_mini_axi_sim.core_mini_axi_write_read_memory_stress_test PASS 181707.50 9.27 19610.82 **
** core_mini_axi_sim.core_mini_axi_master_write_alignment PASS 995.00 0.03 32375.76 **
** core_mini_axi_sim.core_mini_axi_finish_txn_before_halt_test PASS 296.25 0.01 30031.72 **
** core_mini_axi_sim.core_mini_axi_riscv_tests PASS 73098.75 2.05 35697.16 **
** core_mini_axi_sim.core_mini_axi_riscv_dv PASS 33206.25 1.15 28968.95 **
** core_mini_axi_sim.core_mini_axi_csr_test PASS 415780.00 15.71 26463.78 **
** core_mini_axi_sim.core_mini_axi_exceptions_test PASS 2225.00 0.08 28248.19 **
** core_mini_axi_sim.rvv_exceptions_test PASS 0.00 0.00 0.00 **
** core_mini_axi_sim.core_mini_axi_coralnpu_isa_test PASS 1616.25 0.05 29920.57 **
** core_mini_axi_sim.core_mini_axi_rand_instr_test PASS 43390.00 1.35 32149.52 **
** core_mini_axi_sim.core_mini_axi_burst_types_test PASS 675745.00 26.64 25368.52 **
** core_mini_axi_sim.core_mini_axi_float_csr_test PASS 243.75 0.01 29517.31 **
** core_mini_axi_sim.unreachable_prefetch_fault PASS 2986.25 0.09 32024.85 **
** core_mini_axi_sim.core_mini_axi_frm_test PASS 2017.50 0.05 36870.69 **
** core_mini_axi_sim.core_mini_axi_backdoor_load_test PASS 13560.00 0.54 25070.78 **
** TESTS=18 PASS=18 FAIL=0 SKIP=0 2620567.52 93.77 27948.18 **
```

### rvv_core_mini_axi_sim_cocotb
```
** core_mini_axi_sim.core_mini_axi_basic_write_read_memory PASS 1171148.75 41.21 28418.58 **
** core_mini_axi_sim.core_mini_axi_run_wfi_in_all_slots PASS 1168.75 0.07 17897.05 **
** core_mini_axi_sim.core_mini_axi_slow_bready PASS 1382.50 0.05 27714.91 **
** core_mini_axi_sim.core_mini_axi_write_read_memory_stress_test PASS 181707.50 18.20 9981.63 **
** core_mini_axi_sim.core_mini_axi_master_write_alignment PASS 995.00 0.07 15119.29 **
** core_mini_axi_sim.core_mini_axi_finish_txn_before_halt_test PASS 296.25 0.02 14340.18 **
** core_mini_axi_sim.core_mini_axi_riscv_tests PASS 73098.75 4.60 15888.34 **
** core_mini_axi_sim.core_mini_axi_riscv_dv PASS 33206.25 2.14 15524.24 **
** core_mini_axi_sim.core_mini_axi_csr_test PASS 415780.00 34.18 12165.84 **
** core_mini_axi_sim.core_mini_axi_exceptions_test PASS 2225.00 0.14 15345.46 **
** core_mini_axi_sim.rvv_exceptions_test PASS 260.00 0.02 14892.31 **
** core_mini_axi_sim.core_mini_axi_coralnpu_isa_test PASS 1616.25 0.10 16607.12 **
** core_mini_axi_sim.core_mini_axi_rand_instr_test PASS 43602.50 2.42 17999.39 **
** core_mini_axi_sim.core_mini_axi_burst_types_test PASS 675745.00 28.88 23400.13 **
** core_mini_axi_sim.core_mini_axi_float_csr_test PASS 243.75 0.03 9016.49 **
** core_mini_axi_sim.unreachable_prefetch_fault PASS 2988.75 0.20 14658.34 **
** core_mini_axi_sim.core_mini_axi_frm_test PASS 2017.50 0.13 15991.04 **
** core_mini_axi_sim.core_mini_axi_backdoor_load_test PASS 13560.00 0.61 22397.94 **
** TESTS=18 PASS=18 FAIL=0 SKIP=0 2621042.52 133.07 19696.93 **
```
