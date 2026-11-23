# Kernel correlation

This directory compares the inference engine's kernel benchmarks
(`inference-engine/coral/bench/`) on the M3 RTL and on the model. Both use
the M3 as built: 8 KB ITCM, 32 KB DTCM, `configs/machines/m3.yaml`, no
`--highmem`.

Each benchmark puts its kernel calls between `bench::roi_begin()` and
`bench::roi_end()`. These use s10 and s11, as the microbenchmarks do. Thus
both sides time the kernel and not the input fill.

| File | What |
| --- | --- |
| `elfs/` | The benchmarks that fit: `CORAL_MAP=m3 coral/tools/build.sh bench` |
| `run_rtl.sh` | Runs `elfs/` on the RTL simulator and writes `rtl_m3.csv` |
| `model_m3.csv` | ROI cycles and instructions from `reef_perf --elf <elf> -c configs/machines/m3.yaml` |
| `rtl_m3.csv` | ROI cycles and instructions on `rvv_core_mini_verification_axi_sim` (`-c opt`) |

Four benchmarks do not fit in 32 KB of DTCM, so they are not here:
`linear_f32` (over by 231,436 bytes), `linear_w8a8` (242,964), `lm_head`
(231,440) and `mlp_w8a8` (509,004). They use the 8M decode shapes. Their
weights alone are 256 KB or more.

## Results (M3 RTL at `72d700ef`)

ROI instruction counts match on every kernel, so both sides time the same
work. Error = (model - RTL) / RTL. Model as of 2026-11-16, with the M3
dispatch rules, the divider fixes, the unary-vector decode fix and the
whole-line fetch rule in
`m3.yaml` (`docs/pinned.md`, "M3 fixes").

| Kernel | Model ROI | RTL ROI | Error | ROI insts |
| --- | ---: | ---: | ---: | ---: |
| `activation_int8` | 12,208 | 12,562 | -2.8% | 4,030 |
| `attention` | 13,852 | 13,814 | +0.3% | 2,689 |
| `attention_int8` | 35,011 | 34,849 | +0.5% | 13,758 |
| `embedding_int8` | 1,749 | 1,749 | +0.0% | 1,126 |
| `layernorm` | 3,651 | 3,572 | +2.2% | 633 |
| `layernorm_int8` | 7,982 | 8,131 | -1.8% | 4,123 |
| `linear_int8` | 59,468 | 59,408 | +0.1% | 29,696 |
| `mlp_int8` | 123,479 | 123,324 | +0.1% | 61,180 |

Mean absolute error: 1.0% over 8 kernels (0.9% before the fixes).
`layernorm_int8` went from -0.6% to -3.2% with the unary-vector decode fix,
which had been cancelling M3's whole-line fetch rule; with that rule too it
is -1.8%.

Before this fix the model was at 3.6%: `linear_int8` +7.0%,
`mlp_int8` +6.7%, `attention_int8` +4.7%. These kernels have the most scalar
loads (one `lb` per inner-loop step, feeding `vwmacc.vx`). The model charged
every load consumer the 5-cycle `load_chain` latency. On M3, only a load or
store address, scalar store data and a `jalr` target read the register file
without write forwarding (`Decode.scala:347-361`, `regd`); every other reader
gets the value one cycle earlier (`comb`). The model now does the same:
`Dispatch` adds one cycle for those reads, and `lsu_scalar_latency_overhead`
goes from 3 to 2, so `load_chain` stays at 5 cycles and other load uses take 4.
(#42 later set `dtcm_latency` to 1, the M3 SRAM, and the overhead to 3; the
totals are the same.)

A registered read waits only once its writer has a ready cycle. A writer the
model has not scheduled yet (`vmv.x.s` until the vector unit retires it) is
`Inst::Never`, and adding the extra cycle to that would wrap to 0 and let the
reader through. `gemv_int8` improves from -7.4% to -3.8%: its `sw` now waits
for `vmv.x.s` plus the registered cycle.
