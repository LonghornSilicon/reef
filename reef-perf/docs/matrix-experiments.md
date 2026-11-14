# Matrix-engine experiments for the design review

Experiments that test the matrix engine on the GPT-Neo kernels from
`inference-engine/coral` (PR #57), run with the sweep experiment
(`python/reef_perf/experiments/sweep/`). Predictions go in
`matrix-experiments-predictions.csv` and are committed before any run, as
for the RTL programs (`microbenchmarks/matrix/heldout_predictions.csv`).

## Why the experiments are shaped this way

At realistic flash speeds the kernels wait on the weight link, not on the
engine: on 2026-11-13, at 1 B/cycle, fc1 took 1.206M cycles for 8 tokens
and 1.203M for 1, and the array was busy 65,536 of those cycles. A matrix
knob turned in that state mostly measures the memory. So every experiment
either removes the other limits to expose the engine, or finds the memory
speed at which the engine starts to matter. Other modules' parameters
(memory, core loads, ROB) are changed only as tools for that.

Each experiment runs under two memory settings and reports both:

- **Realistic:** `xspi_str.yaml` (1 B/cycle, 68 cycles to first byte), with
  `xspi_dtr.yaml` (2 B/cycle) as a variant.
- **Exposed:** the link not modeled: `top.memory.params.ext_mem_bytes_per_cycle
  0` and `ext_mem_latency 0`, and the same for `ext_ram_*`. The DMA still
  pays its setup and beat costs. Hypothetical; every slide that uses it
  says so.

Baseline for everything else: `m3.yaml` + `m3_vme.yaml` + `coral_vme.yaml`
(64 PEs), ROB 8, M3's LSU. Change one thing at a time against it.

## What to read in every run

| Metric | summary.csv column | Meaning |
|---|---|---|
| ROI cycles | `roi_cycles` | The kernel's time between its two mcycle reads |
| Engine busy share | `matrix_array_busy_cycles` / `roi_cycles` | How much of the time the engine works; low means starved (busy cycles cover the whole run, the ROI almost all of it) |
| Array fill | `matrix_mac_utilization` | How full the array is while busy; low means the shapes are small for it |
| Dispatch stalls | artifact `dispatch` | What blocks the next instruction: matrix credits, loads, ROB |
| Sensitivity | ROI change per knob step | Near zero: the knob does not matter in this setting |

## Workloads

PR #57 kernel ELFs, built per token count and attention position
(`coral/README.md`; `sweeps/e2e_kernels.yaml` shows the layout):

- Linear: `qkv`, `attn_out`, `fc1`, `fc2` at `E2E_TOKENS` 1 and 8 (2 and 4
  for E2). Position does not change them.
- `attention` at positions 0 and 8 (8 tokens, prefill) and 16 and 79
  (1 token, decode).
- `lm_head` (1 token): the largest matrix-vector product, about 26 MB of
  weights.
- `layernorm` and `embed` only for whole-token totals; they do not use the
  engine.
- Control: `matrix_presets.yaml` microbenchmarks, the engine alone.

Composition (README of `inference-engine/coral`): decode token n =
embed + 8 x (2 layernorm + qkv + attention@(n-1) + attn_out + fc1 + fc2)
at 1 token + layernorm + lm_head.

## Experiments, in order

### E1. When does the engine start to matter

At what flash bandwidth does the engine, not memory, limit the linear
kernels?

- qkv, fc1, fc2, lm_head at 1 and 8 tokens.
- `ext_mem_bytes_per_cycle` 1, 2, 4, 8 and exposed; 64 and 128 PEs
  (`coral_128pe.yaml`). 4 and 8 B/cycle are beyond XSPI: they locate the
  crossover, they are not proposals.
- Look for the bandwidth where 64 and 128 PEs diverge, where busy share
  rises, and whether that differs between 1 and 8 tokens.

### E2. Array fill against token count

How well does the 16-row array use its rows at 1, 2, 4 and 8 tokens, and
what would a 16-token block be worth?

- fc1 and qkv at `E2E_TOKENS` 1, 2, 4, 8, exposed memory. Try
  `-DE2E_BLOCK_TOKENS=16`; build.sh fails if the DTCM does not fit it.
- Look for busy cycles and ROI per token, and array fill.

### E3. Array size and organization on real shapes

Which array suits Reef's matrix products?

- Presets `coral_vme` (64), `coral_128pe`, `os_systolic_16x16`,
  `ws_systolic_16x16`, `tree_8x8`; kernels at 1 and 8 tokens plus the
  `gemm_*` microbenchmarks; exposed memory.
- Look for the ranking and whether it flips between 1 and 8 tokens; array
  fill.
- Coral 64 and 128 have RTL behind them; the others only checked formulas.

### E4. Latency the engine can absorb

What does a deeper pipeline or a slower engine clock cost (the PD
trade-off for fmax)?

- `top.matrix.params.mtx_pe_pipeline_depth` 9 (RTL), 12, 16;
  `mtx_fabric_clock_divider` 1, 2; `mtx_acc_raw_cycles` 4 (RTL) and 0.
- Kernels at realistic and exposed memory; `matrix/mac_chain` as the worst
  case.
- Look for the slowdown in each setting: a budget such as "N more stages
  cost under X%".

### E5. Is the engine fed

With memory out of the way, is the engine limited by the core's load path
or by its own queues?

- Exposed memory. Core: `top.core.lsu.params.lsu_vector_load_reg_overhead`
  3 (M3) and 1; `top.core.rob.params.rob_entries` 8, 16. Matrix:
  `mtx_uop_buffer_entries` 4, 8, 16; `mtx_queue_entries`;
  `mtx_lsu_fifo_entries`.
- Look for which knob moves busy share most, and stalls moving from loads
  to matrix credits. A large effect from cheaper loads supports proposing
  a matrix memory port (not modeled yet).

### E6. Reading results out

How much of a linear kernel goes to reading tiles out (`vtmv`, one per
slice, each reconfiguring the mode) and requantizing?

- `mtx_move_latency` 0 (RTL), 1, 2; split the epilogue with `reef_trace`
  or the topdown experiment.
- Look for the epilogue's share of ROI, at 1 token especially.
- Follow-on, for the int16 requantization question in #rtl-design: an fc1
  variant that holds its whole output before requantizing, against today's
  per-tile requantization. Needs a kernel change.

### E7. Int4: native lanes or software unpack

Which int4 implementation gives more, at which memory speed?

- Native int4 lane (2x MACs): `mtx_macs_per_pe_int8` 4 and 8.
- Software unpack (half the weight bytes): double `ext_mem_bytes_per_cycle`.
  The unpack instructions are not in this proxy; say so.
- At 1 and 2 B/cycle; compare decode tokens/s at token 17.
- An estimate from int8 runs, as planned for the review.

### E8. Two arrays (optional)

The kernels rotate over four accumulators; would a second array overlapping
independent tiles help?

- `mtx_num_arrays` 1 and 2, exposed memory.

## Predictions

`matrix-experiments-predictions.csv` has one row per measurable quantity.
Fill `prediction` (or `low` and `high`), `reasoning` and `predicted_on`,
then commit before running the experiment. After the run, fill `measured`
and `verdict` (hit, miss). A miss is explained (a waveform-style dig into
the run, not a retune) before the result goes on a slide. Add rows freely;
don't edit a prediction after its run.

## Limits

- The kernels have never run on RTL and depend on the vector-load path,
  where the model still has known gaps (`matrix.md`). Comparisons between
  settings are stronger than absolute cycles.
- PR #57's firmware (83f7152) uses the single-memory layout; until it
  moves to flash plus PSRAM, attention's KV reads are provisional.
- No area or energy numbers yet: rankings are by cycles only.
- A dedicated matrix memory port and shared RVV dispatch need model work
  before they can be swept (`sweep-candidates.md`).
