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
| `model_m3.csv` | ROI cycles from `reef_perf --elf <elf> -c configs/machines/m3.yaml` |

Four benchmarks do not fit in 32 KB of DTCM and are not here: `linear_f32`
(over by 231,436 bytes), `linear_w8a8` (242,964), `lm_head` (231,440) and
`mlp_w8a8` (509,004). They use the 8M decode shapes, whose weights alone are
256 KB or more.
