# Kernel correlation

The inference engine's kernel benchmarks (`inference-engine/coral/bench/`),
run on the M3 RTL and on the model with the M3 as built: 8 KB ITCM, 32 KB
DTCM, `configs/machines/m3.yaml`, no `--highmem`. Each benchmark brackets its
kernel calls with `bench::roi_begin()` and `bench::roi_end()` (s10 and s11,
as the microbenchmarks do), so both sides time the kernel without the input
fill.

| File | What |
| --- | --- |
| `elfs/` | The benchmarks that fit: `CORAL_MAP=m3 coral/tools/build.sh bench` |
| `run_rtl.sh` | Runs `elfs/` on the RTL simulator and writes `rtl_m3.csv` |
| `model_m3.csv` | ROI cycles and instructions from `reef_perf --elf <elf> -c configs/machines/m3.yaml` |
| `rtl_m3.csv` | ROI cycles and instructions on `rvv_core_mini_verification_axi_sim` (`-c opt`) |

Four benchmarks do not fit in 32 KB of DTCM and are not here: `linear_f32`
(over by 231,436 bytes), `linear_w8a8` (242,964), `lm_head` (231,440) and
`mlp_w8a8` (509,004). They use the 8M decode shapes, whose weights alone are
256 KB or more.

## Results (model at PR #40, M3 RTL at `72d700ef`)

ROI instruction counts match on every kernel, so both sides time the same
work. Error = (model - RTL) / RTL.

| Kernel | Model ROI | RTL ROI | Error | ROI insts |
| --- | ---: | ---: | ---: | ---: |
| `activation_int8` | 12,151 | 12,562 | -3.3% | 4,030 |
| `attention` | 14,088 | 13,814 | +2.0% | 2,689 |
| `attention_int8` | 36,494 | 34,849 | +4.7% | 13,758 |
| `embedding_int8` | 1,710 | 1,749 | -2.2% | 1,126 |
| `layernorm` | 3,636 | 3,572 | +1.8% | 633 |
| `layernorm_int8` | 8,036 | 8,131 | -1.2% | 4,123 |
| `linear_int8` | 63,537 | 59,408 | +7.0% | 29,696 |
| `mlp_int8` | 131,636 | 123,324 | +6.7% | 61,180 |

Mean absolute error: 3.6% over 8 kernels.
