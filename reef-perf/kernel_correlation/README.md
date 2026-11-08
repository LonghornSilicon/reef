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
work. Error = (model - RTL) / RTL.

| Kernel | Model ROI | RTL ROI | Error | ROI insts |
| --- | ---: | ---: | ---: | ---: |
| `activation_int8` | 12,103 | 12,562 | -3.7% | 4,030 |
| `attention` | 13,800 | 13,814 | -0.1% | 2,689 |
| `attention_int8` | 34,858 | 34,849 | +0.0% | 13,758 |
| `embedding_int8` | 1,693 | 1,749 | -3.2% | 1,126 |
| `layernorm` | 3,632 | 3,572 | +1.7% | 633 |
| `layernorm_int8` | 7,987 | 8,131 | -1.8% | 4,123 |
| `linear_int8` | 59,446 | 59,408 | +0.1% | 29,696 |
| `mlp_int8` | 123,452 | 123,324 | +0.1% | 61,180 |

Mean absolute error: 1.3% over 8 kernels.

Before this fix the model was at 3.6%: `linear_int8` +7.0%,
`mlp_int8` +6.7%, `attention_int8` +4.7%. These kernels have the most scalar
loads (one `lb` per inner-loop step, feeding `vwmacc.vx`). The model charged
every load consumer the 5-cycle `load_chain` latency. On M3, only a load or
store address, scalar store data and a `jalr` target read the register file
without write forwarding (`Decode.scala:347-361`, `regd`); every other reader
gets the value one cycle earlier (`comb`). The model now does the same:
`Dispatch` adds one cycle for those reads, and `lsu_scalar_latency_overhead`
goes from 3 to 2, so `load_chain` stays at 5 cycles and other load uses take 4.

One microbenchmark gets worse: `gemv_int8` goes from -7.4% to -12.9%. Its
`sw` now waits one more cycle for `vmv.x.s`, and the model gets faster, not
slower. With 64 Rob entries the effect goes away, so the cause is in the Rob
and vector completion timing, not in memory. It is not fixed here.
