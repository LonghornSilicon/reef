# reef_top smoke test results

**Run:** 2026-10-01 on koopa, `scripts/run_reef_smoke.sh` at repo commit `6f62f99`.
Verilator 5.052, M3 sim filelist (`rtl/vendor/coralnpu/sim.f`, incl. firtool verification layers)
+ `rtl/reef/reef.f`, upstream defines `USE_GENERIC TB_SUPPORT VLEN_128 ZVE32F_ON`.
Program built with riscv-none-elf-gcc 15.2, `-march=rv32imf_zve32f_zicsr_zifencei_zbb` (M3's
toolchain string). The testbench only drives `reef_top` pins.

| Run | f_clk/f_sck | Result |
|---|---:|---|
| `smoke` (RVV int add, RVV FP32 reduction, scalar int/FP, vlmax) | 16 | PASS, `mpause` halt, no fault |
| `smoke` | 32 | PASS |
| `smoke_buserr` (load from 0x8000_0000) | 16 | PASS: DECERR -> trap, mcause 5, `fault_o` = 1 |

What each run checks: CSR reset state over SPI; ITCM program load and full readback; DTCM 4-beat
write/readback; `PC_START` write at the unaligned CSR address and readback; boot via `RESET` CSR
(1 then 0); halt/fault pins and `STATUS`; 8 result words read from DTCM over SPI.

Not covered yet: interrupts (tied off), `test_en_i = 1`, back-to-back frames with no gap, SPI
ratios below 16, gate-level simulation.

## Log

```
program: 512 bytes, smoke_result @0x00010400
=== smoke, f_clk/f_sck=16, pass
program 512 bytes, result @0x00010400, f_clk/f_sck = 16, mode pass
[1] CSR RESET = 3
[2] ITCM load + readback: 32 beats, 0 mismatches
[3] DTCM 4-beat write/read: ok
[4] PC_START write/readback over SPI: ok
[6] halted=1 fault=0 (halt seen 615 cycles after the RESET=0 frame)
    magic      = 0x600dcafe (want 0x600dcafe)
    vadd_sum   = 0x00001fc0 (want 0x00001fc0)
    fsum_bits  = 0x44fc0000 (want 0x44fc0000)
    trap_cause = 0x00000000 (want 0x00000000)
    scalar_mul = 0xfff552e2 (want 0xfff552e2)
    scalar_fp  = 0x40600000 (want 0x40600000)
    vl_max     = 0x00000004 (want 0x00000004)
    main() body: 1655 core cycles (mcycle)
SMOKE PASS: 0 failure(s), 267845 cycles total
=== smoke, f_clk/f_sck=32, pass
program 512 bytes, result @0x00010400, f_clk/f_sck = 32, mode pass
[1] CSR RESET = 3
[2] ITCM load + readback: 32 beats, 0 mismatches
[3] DTCM 4-beat write/read: ok
[4] PC_START write/readback over SPI: ok
[6] halted=1 fault=0 (halt seen 0 cycles after the RESET=0 frame)
    magic      = 0x600dcafe (want 0x600dcafe)
    vadd_sum   = 0x00001fc0 (want 0x00001fc0)
    fsum_bits  = 0x44fc0000 (want 0x44fc0000)
    trap_cause = 0x00000000 (want 0x00000000)
    scalar_mul = 0xfff552e2 (want 0xfff552e2)
    scalar_fp  = 0x40600000 (want 0x40600000)
    vl_max     = 0x00000004 (want 0x00000004)
    main() body: 1655 core cycles (mcycle)
SMOKE PASS: 0 failure(s), 534430 cycles total
=== smoke_buserr, f_clk/f_sck=16, buserr
program 528 bytes, result @0x00010400, f_clk/f_sck = 16, mode buserr
[1] CSR RESET = 3
[2] ITCM load + readback: 33 beats, 0 mismatches
[3] DTCM 4-beat write/read: ok
[4] PC_START write/readback over SPI: ok
[6] halted=1 fault=1 (halt seen 622 cycles after the RESET=0 frame)
    magic      = 0xbad00bad (want 0xbad00bad)
    vadd_sum   = 0x00001fc0 (want 0x00001fc0)
    fsum_bits  = 0x44fc0000 (want 0x44fc0000)
    trap_cause = 0x00000005 (want 0x00000005)
    scalar_mul = 0xfff552e2 (want 0xfff552e2)
    scalar_fp  = 0x40600000 (want 0x40600000)
    vl_max     = 0x00000004 (want 0x00000004)
SMOKE PASS: 0 failure(s), 284300 cycles total
ALL SMOKE RUNS PASS

[exited with code 0]
```
