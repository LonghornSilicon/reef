# The pinned-RTL config

`configs/machines/coral_pinned.yaml` models CoralNPU at `8a5e1b31`
(`VmeCoreMini`) as a whole: the Zvt matrix engine and everything else that
changed in the core since M3. No RTL has M3 with a matrix engine, so this is
the machine the matrix model can be checked on end to end.

It is an overlay on `m3.yaml`. It does not change M3 (`m3.yaml`,
`tests/golden/m3_results.json`) or M3 + Zvt (`m3_vme.yaml` and the matrix
tests), and it is not Reef's design.

```sh
reef_perf --elf x.elf -c configs/machines/m3.yaml -c configs/machines/coral_pinned.yaml
uv run python -m reef_perf.experiments.correlation.pinned   # score against the RTL
```

## Rules

- Every difference from M3 is one switch in the file: a value, or a
  parameter that turns new model code on. New parameters default to M3's
  behavior in the code, so removing a line turns that change off.
- Two golden files. `tests/golden/m3_results.json` never changes for this
  work. `tests/golden/pinned_results.json` pins this machine and is
  regenerated with `REEF_PERF_UPDATE_PINNED_GOLDEN=1` when a switch is added.
- The matrix programs that match the RTL today must still match on this
  config (`test_matches_rtl_on_pinned_config`).
- RTL numbers are in `rtl_pinned.csv` next to each `rtl_m3.csv`, and
  `microbenchmarks/matrix/rtl_vme.csv` for the matrix programs. A new program
  gets a model prediction committed before its RTL run.

## Switches that are on

| ID | Switch | M3 | Pinned | Source | Fixes |
| --- | --- | --- | --- | --- | --- |
| F2 | `redirect_penalty` | 2 | 3 | measured | `branch_fwd_taken`, `branch_bwd_not_taken`, `jalr_chain`: -14.3% to 0.0% |
| C1 | `rob_entries` | 8 | 16 | `Parameters.scala:141` | independent vector streams, `vec_add_m1` +50% to +19%, MAC chains |
| V1 | `vec_rob_entries` | 8 | 16 | `rvv_backend_define.svh` `ROB_DEPTH` | little on its own |
| V2 | `vec_completion_matches_pc` | true | false | `RetirementBuffer.scala:452`, `:532` | `vdiv_m8_indep` -37% to -14% |
| V3 | `vec_fp_latency` | 4 | 5 | `FALU_PIPEREGS` 4; measured | `vfadd_chain`, `vfmacc_chain`: -14.5% to -0.5% |
| V3b | `vec_fdiv_cycles_per_uop` | 8 | 9 | measured | `vfdiv_indep` -11.3% to -0.5% |
| X1 | `zvt` | false | true | `--enableVme=True` | Zvt programs run |
| X2 | `lsu_tile_cost` and `lsu_tile_*` | off | 1 + 0 per beat + 1 | measured (`tile_*_burst`) | tile loads and stores |

## Differences not modeled yet

Each needs model code behind a new switch. A single existing value was tried
for the first three and fixed one program while breaking others.

| ID | Difference | Evidence | Why not a value |
| --- | --- | --- | --- |
| F1 | Fetch: two fetches in flight through a reorder buffer (`UncachedFetch.scala`, `maxConcurrentTx = 2`) | `alu_indep` +71%, `branch_group` +43% | `fetch_interval 1` matches both, but makes `branch_tight`, `jal_fwd` and `branch_fwd_not_taken` 42% to 50% fast: a taken branch still costs a cycle |
| C2 | LsuV3 for scalar accesses | `load_stream` +33% | `lsu_scalar_occupancy_overhead 2` matches it, but makes `addr_forward` 25% fast |
| M1 | External bus | `ext_load_chain` -25%, `ext_load_stream` -18%, `ext_store_stream` -34% | `axi_latency 3` matches the load stream only; a store is 2 cycles slower than on M3, a chained load 1 |
| C2 | LsuV3 for vector accesses: 2 cycles per load + 1 per register, 3 + 1 for a store | `vec_memcpy` +174%, `vec_add_m4` +118%, `gemv_loads` +177%, every kernel +7% to +65% | M3's LSU pays its overhead per register |
| V4 | The new integer divider at LMUL 8 | `vdiv_m8_indep` -14% | not looked at yet |
| V5 | Independent vector streams run 7% to 10% fast | `vadd_indep`, `vmul_indep`, `vcpop_indep`, `vfadd_indep`, `vmixed_lmul` | fetch, the result arbiter or the registered completion token; to separate after F1 |
| C5 | A CSR waits for an empty retirement buffer | matrix ROIs end early (matrix.md, open case 2) | no such rule in Dispatch |
| C6 | `vtdiscard` dispatches alone, with the core idle | `vtdiscard_after_mac` 24.0 against 5.5 | no such rule in Dispatch |
| C7 | Scalar FP | `fp_chain` +41%, `fp_mix` +20% | `fp_chain` is the same on both RTLs, so part of this is M3's |

Gaps the model already has on M3, unchanged on this RTL: `vdiv_small_*`
(the divider's early finish), `alu_chain` -6%, `vec_pingpong` -6%.

## Score

Mean absolute ROI error against the pinned RTL, and programs within 5%.

| State | core (20) | frontend (5) | vector units (32) | kernels (8) | all (65) | within 5% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M3 model, no switch | 42.3% | 8.6% | 29.4% | 43.5% | 33.5% | 26 |
| Values only (the table above) | 41.0% | 0.1% | 19.1% | 38.2% | 26.7% | 33 |

Matrix programs (per-op cost, `matrix_diff`): 31 of 44 within 5% on this
config, the same 31 as with `m3_vme.yaml` + `REFERENCE_CORE_PARAMS`.
