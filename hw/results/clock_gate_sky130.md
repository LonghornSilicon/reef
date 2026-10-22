# sky130 clock gate: function

**Wrapper:** `rtl/clock/ClockGate_sky130.sv` (one `sky130_fd_sc_hd__sdlclkp_4`, notes in
`rtl/clock/README.md`). Pinned baseline M3 `72d700e`. Cell models: `sky130A/libs.ref/sky130_fd_sc_hd/verilog/`
from the open_pdks build in `fossi-foundation/ciel-releases` (same release as
`results/sram_sky130.md`).

## Contract test

`SKY130_FD_SC_HD=<pdk>/sky130A/libs.ref/sky130_fd_sc_hd scripts/run_clock_gate_sky130_tb.sh` on
koopa, VCS T-2022.06. 20,000 cycles at 20 ns; `clk_o` compared with the reference every 0.1 ns
(4,000,000 samples).

| DUT | Errors | `clk_o` pulses (expected) | Result |
|---|---:|---|---|
| vendored generic `ClockGate.sv` (latch + AND) | 0 | 14,126 (14,126) | PASS |
| `ClockGate_sky130` + PDK model, `FUNCTIONAL` | 0 | 14,126 (14,126) | PASS |
| `ClockGate_sky130` + PDK model, timing variant | 0 | 14,126 (14,126) | PASS |
| `ClockGate_sky130` + our Verilator model | 0 | 14,126 (14,126) | PASS |
| mutant: cell model without the latch | >0 (first at 10.27 ns, enable/te rose while `clk_i` high) | | FAIL, as it should |

Coverage, same in every run (same seed): rising edges with the gate on 10,043, off 5,874, on by
`te` alone 4,083; enable/te changes while `clk_i` was high 26,656, low 26,404.

VCS rejects the PDK's UDP definitions in `primitives.v` under `` `default_nettype none `` (IND
errors on the UDP ports). The script compiles a copy with only those directives changed to `wire`
and saves the diff (23 lines).

## In reef_top (sky130 TCMs + clock gate)

| Check | Tool | Result |
|---|---|---|
| Lint, `ClockGate_sky130` + our cell model, `-Wall` | Verilator 5.032 (M3 hermetic), `scripts/lint_reef.sh` on WSL | PASS |
| Lint, reef_top + core from `configs/filelists/reef_top_sky130.f` (synth set) | same | PASS; without the ICG cell model it fails (`Can't resolve module reference: 'sky130_fd_sc_hd__sdlclkp_4'`) |
| Smoke test, `NODE=sky130 scripts/run_reef_smoke.sh` | Verilator 5.052, koopa | **3/3 PASS**, result words as in `results/reef_smoke.md`; `main()` 1,655 core cycles, total 267,845 / 534,430 / 284,300, identical to the generic build |

The smoke model contains both core clock gates as the cell model:
`u_core.cg.u_cg` and `u_core.rst_sync.icg.u_cg` (Verilator signals `...__DOT__u_cg__DOT__en_latch`).
