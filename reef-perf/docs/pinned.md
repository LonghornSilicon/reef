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
| C2 | `lsu_v3` | false | true | waveforms (below) | vector memory: `vec_memcpy` +174% to -0.5%, `vec_add_m4` +118% to +0.1%; kernels 38% to 4.4% mean |
| C3 | `load_waw_advance` | 1 | 0 | waveform of `addr_forward` | keeps `addr_forward` at 4 cycles per load with LsuV3's 3-cycle loads |
| V6 | `vec_store_completes_on_last_register` | false | true | `Lsu.scala:1970-1971` | multi-register stores |
| V7 | `vec_memory_completion_latency` | 0 | 1 | `RetirementBuffer.scala:86`; measured | `lsu/vload_rob`, `lsu/vstore_rob` |
| M1 | `axi_latency`, `axi_write_latency` | 2, as a read | 4, 5 | waveforms of `ext_*` | `ext_load_chain` -25% to +0.2%, the two streams to -1.8% |
| X1 | `zvt` | false | true | `--enableVme=True` | Zvt programs run |
| X2 | `lsu_tile_cost` and `lsu_tile_*` | off | 1 + 0 per beat + 1 | measured (`tile_*_burst`) | tile loads and stores |

## Differences not modeled yet

Each needs model code behind a new switch.

| ID | Difference | Evidence | Why not a value |
| --- | --- | --- | --- |
| F1 | Fetch: two fetches in flight through a reorder buffer (`UncachedFetch.scala`, `maxConcurrentTx = 2`) | `alu_indep` +71%, `branch_group` +43% | `fetch_interval 1` matches both, but makes `branch_tight`, `jal_fwd` and `branch_fwd_not_taken` 42% to 50% fast: a taken branch still costs a cycle |
| V8 | `vmv.x.s` of a loaded register | `lsu/vload_use` +8% (1 cycle per link); `vec_pingpong` -6% | the vector-to-scalar path, not the LSU |
| C8 | Scalar FP divide and square root in kernels | `layernorm` +18% | with C7 |
| V4 | The new integer divider at LMUL 8 | `vdiv_m8_indep` -14% | not looked at yet |
| V5 | Independent vector streams run 7% to 10% fast | `vadd_indep`, `vmul_indep`, `vcpop_indep`, `vfadd_indep`, `vmixed_lmul` | fetch, the result arbiter or the registered completion token; to separate after F1 |
| C5 | A CSR waits for an empty retirement buffer | matrix ROIs end early (matrix.md, open case 2) | no such rule in Dispatch |
| C6 | `vtdiscard` dispatches alone, with the core idle | `vtdiscard_after_mac` 24.0 against 5.5 | no such rule in Dispatch |
| C7 | Scalar FP | `fp_chain` +41%, `fp_mix` +20% | `fp_chain` is the same on both RTLs, so part of this is M3's |

Gaps the model already has on M3, unchanged on this RTL: `vdiv_small_*`
(the divider's early finish), `alu_chain` -6%, `vec_pingpong` -6%.

## LsuV3

Read from waveforms of `microbenchmarks/lsu/` on the RTL (the LSU's
`io_req`, `io_dbus`, `io_rd`, `io_rvv2lsu`, `io_lsu2rvv` and
`io_storeComplete`). D is the cycle the core dispatches the instruction.

| Access | What the RTL does | Cost |
| --- | --- | --- |
| Scalar load | bus read in D+2 when the LSU is idle, value written 2 cycles after the read, next instruction's bus cycle after that | 3 cycles per load (M3: 4); load-to-use unchanged |
| Scalar store | bus write, complete the next cycle | 2 per store |
| Vector load, unit-stride or strided, unmasked | first bus read in D+2, without waiting for the vector backend; one 16-byte line per cycle; each register handed back 2 cycles after its last line | 2 + 1 per line (M3: about 4 per register) |
| Vector store; indexed or masked load | the backend hands over data, index or mask first; bus cycle the cycle after; register done 2 cycles after that | 3 + 1 per line |
| External memory | a read holds the bus 4 cycles, a write 5; then as above | 6 per scalar load or store |

The slot runs one instruction at a time in dispatch order, so a store
waiting for its data holds the LSU. `lsu_v3_setup_cycles`,
`lsu_v3_return_cycles`, `lsu_v3_scalar_store_cycles` and
`lsu_v3_handshake_cycles` hold the numbers above.

One value is fitted, not read: `lsu_v3_backend_lead` 0, the earliest the
vector backend takes a loaded register relative to when the load's uop could
issue. Two candidates (0 and 1) were recorded as predictions for four new
probe programs before their RTL run (`microbenchmarks/lsu/heldout_predictions.csv`).
`vload_dep_rob` measured 12.0 cycles; 0 predicted 12.0 and 1 predicted 11.0.

`tests/python/integration/test_pinned_lsu.py` holds 16 of the 20 LSU
programs within 2% of the RTL; the other four are listed there with the
reason (three are fetch-bound, one is V8).

Tile beats keep their own cost (`lsu_tile_*`) and do not go through the new
path.

## Score

Mean absolute ROI error against the pinned RTL, and programs within 5%.

| State | core (20) | frontend (5) | vector units (32) | kernels (8) | all (65) | within 5% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M3 model, no switch | 42.3% | 8.6% | 29.4% | 43.5% | 33.5% | 26 |
| Values only (the table above) | 41.0% | 0.1% | 19.1% | 38.2% | 26.7% | 33 |
| + LsuV3, external bus, completion (C2, C3, V6, V7, M1) | 9.9% | 0.1% | 19.1% | 4.4% | 13.0% | 47 |

Matrix programs (per-op cost, `matrix_diff`), within 5% of the RTL:

| State | of 44 | The GEMMs (`gemm_tile`, `gemm_multi`, `gemm_longk`) |
| --- | ---: | --- |
| `m3_vme.yaml` + `REFERENCE_CORE_PARAMS` (unchanged) | 31 | +67%, +55%, +67% |
| This config, values only | 31 | +67%, +56%, +67% |
| This config with LsuV3 | 35 | +10%, +4%, +12% |

The four that joined are `vec_load_burst`, `vec_load4_burst`,
`vec_store4_burst` and `gemm_multi`. None was used to build the LsuV3 path.
