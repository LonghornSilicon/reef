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
| F1 | `fetch_reorder_buffer` (with `fetch_max_in_flight` 2, `fetch_response_cycles` 2, `redirect_from_dispatch_cycles` 2) | false | true | `UncachedFetch.scala`: `FetchControl`, `Fetcher`, `FetchReorderBuffer` | `alu_indep` +71% to 0.0%; every frontend program within 0.1% |
| C1 | `rob_entries` | 8 | 16 | `Parameters.scala:141` | independent vector streams, `vec_add_m1` +50% to +19%, MAC chains |
| V1 | `vec_rob_entries` | 8 | 16 | `rvv_backend_define.svh` `ROB_DEPTH` | little on its own |
| V2 | `vec_completion_matches_pc` | true | false | `RetirementBuffer.scala:452`, `:532` | `vdiv_m8_indep` -37% to -14% |
| V2b | `vec_store_completion_matches_pc` (new) | false | true | `RetirementBuffer.scala:558` | at most 0.2% on the current programs |
| V3 | `vec_fp_latency` | 4 | 5 | `FALU_PIPEREGS` 4; measured | `vfadd_chain`, `vfmacc_chain`: -14.5% to -0.5% |
| V3b | `vec_fdiv_cycles_per_uop` | 8 | 9 | measured | `vfdiv_indep` -11.3% to -0.5% |
| C2 | `lsu_v3` | false | true | waveforms (below) | vector memory: `vec_memcpy` +174% to -0.5%, `vec_add_m4` +118% to +0.1%; kernels 38% to 4.4% mean |
| C4 | `branch_ends_group` | false (the M3 model omits the rule) | true | `Decode.scala:419-422`; waveform of `matrix/mac_rob` | `branch_group` -14.3% to 0.0%, `vec_pingpong` -6.0% to -0.1% |
| C5 | `slot0_only_rule`, `jump_ends_group`, `csr_waits_for_empty_rob`, `vtdiscard_interlock` (new, four switches) | false (the M3 model omits them) | true | `Decode.scala:196-205`, `:601-607`, `:707-708` | `vdiv_m8_indep` -14% to -0.1%; every chain's -0.5% to 0.0%; `vtdiscard_after_mac` 5.5 to 24.0 (RTL 24.0) |
| C3 | `load_waw_advance` | 1 | 0 | waveform of `addr_forward` | keeps `addr_forward` at 4 cycles per load with LsuV3's 3-cycle loads |
| V6 | `vec_store_completes_on_last_register` | false | true | `Lsu.scala:1970-1971` | multi-register stores |
| M1 | `axi_latency`, `axi_write_latency` | 2, as a read | 4, 5 | waveforms of `ext_*` | `ext_load_chain` -25% to +0.2%, the two streams to -1.8% |
| C2b | `lsu_v3_tile` | false | true | waveforms of `tile_load32_burst`, `tile_store32_burst`, `load_then_store` | replaces the fitted `lsu_tile_*` cost; `load_then_store` -7% to 0.0% |
| X4 | `mtx_lsu_holds_full_fifo_beat` (new, matrix module) | false (`m3_vme.yaml`, unchanged) | true | waveform of `tile_load_behind_mac`; `zvt.sv:188-193`, `multi_fifo.sv:44`, `:111-114` | needed by C2b |
| X3 | `mtx_complete_latency` | 1 (`m3_vme.yaml`, unchanged) | 0 | waveform of `matrix/mac_rob` | with C4, the `*_rob` matrix programs and `mac_queue` stay exact |
| X1 | `zvt` | false | true | `--enableVme=True` | Zvt programs run |

## Differences not modeled yet

Each needs model code behind a new switch.

| ID | Difference | Evidence | Why not a value |
| --- | --- | --- | --- |
| V8 | `vmv.x.s` of a loaded register | `lsu/vload_use` +8% (1 cycle per link); `vec_pingpong` -6% | the vector-to-scalar path, not the LSU |
| C8 | Scalar FP divide and square root in kernels | `layernorm` +18% | with C7 |
| V5 | Independent vector streams run 7% to 10% fast | `vadd_indep`, `vmul_indep`, `vcpop_indep`, `vfadd_indep`, `vmixed_lmul` | fetch, the result arbiter or the registered completion token; to separate after F1 |
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

`tests/python/integration/test_pinned_lsu.py` holds 19 of the 20 LSU
programs within 2% of the RTL; the other is V8.

Tile beats keep their own cost (`lsu_tile_*`) and do not go through the new
path.

## Fetch

The model follows the RTL's structure (`fetch_reorder_buffer`), not a fitted
rate:

- `FetchControl` issues a fetch every cycle, speculatively down the next
  16-byte line, while the reorder buffer has room (`fetch_max_in_flight` 2,
  `maxConcurrentTx`).
- A line is written to the instruction buffer `fetch_response_cycles` (2)
  after its fetch: the bus response is registered, then the reorder buffer's
  state. Lines are written in order, one per cycle, and only when the whole
  line fits (`sufficientBuffer`), so a full buffer stalls the reorder buffer
  and then the fetches.
- Predecode sees a predicted-taken branch or jump when its line is written
  and fetches the target in that cycle; the sequential fetches behind it are
  dropped. That is 2 cycles per taken jump (`branch_tight`, `jal_fwd`).
- A mispredicted branch or a `jalr` is resolved after the core dispatches it:
  its target is fetched `redirect_from_dispatch_cycles` (2) later. If the
  branch waits in the buffer, the redirect waits with it.

- While a predicted jump's line waits for buffer space, the reorder buffer
  fills behind it and `FetchControl` is left holding a request for the next
  sequential line. That request cannot be withdrawn (`io.fetchAddr` is
  `Irrevocable`, `ongoingFetch`), so when room appears it goes out, wasted,
  and the target is fetched a cycle later. Waveform of `matrix/mac_queue`:
  the loop's `bnez` line is written in cycle 104, a fetch of 0xa0 goes out in
  104 and the target, 0x50, in 105. This is the case the fitted values of
  the first attempt could not have produced.

Not modeled: wrong-path instructions entering the buffer after a
mispredict (they are flushed; only the time is charged), and fetch from
outside the ITCM.

## The branch rule

A conditional branch is the last instruction dispatched in its cycle
(`branch_ends_group`, C4; `Decode.scala:419-422`). The waveform of
`matrix/mac_rob` shows it: the loop's `bnez` is dispatched alone and the MAC
behind it a cycle later.

It first looked wrong: on, it made six matrix programs and three LSU probes
a cycle slow. Three other things were off by a cycle and had been hiding the
missing rule. Each was found in a waveform:

| What | RTL | Model before | Now |
| --- | --- | --- | --- |
| A scalar load's retire cycle (`lsu/lw_rob`) | D + 4, the cycle its value is written | D + 5 | D + 4 (LsuV3 path) |
| A MAC's retire cycle (`matrix/mac_rob`) | D + 20 | D + 21 (`mtx_complete_latency` 1) | D + 20 on this config (X3, `mtx_complete_latency` 0); `m3_vme.yaml` keeps 1 |
| The redirect behind a stalled jump (`matrix/mac_queue`) | target fetched a cycle late, after a wasted fetch | not modeled | modeled (Fetch, above) |

A fourth value, an extra completion cycle for vector loads and stores that
step 3 had fitted to `vload_rob` and `vstore_rob`, was the missing rule
under another name and is removed.

`mac_queue` was the one program that did not fit: 18.5 against 19.0 with the
rule on and the MAC retiring in D + 20. The stalled redirect is what it was
missing. With all of it on, the nine probe loops match exactly: `mac_rob`
22, `zero_rob` 12, `move_tv_rob` 12, `mac_rob_tm4` 20, `mac_queue` 19,
`vload_rob` 9, `vload_dep_rob` 12, `vstore_rob` 11, `lw_rob` 6. Without the
rule `mac_queue` is now the one that is off (19.5).

## Tile beats

Waveforms of `tile_load32_burst`, `tile_store32_burst` and `load_then_store`:

- a tile load is an unmasked unit-stride load: first bus read in D + 2, one
  16-byte beat per cycle, each beat pushed to the engine (`lsu2vme`) 2 cycles
  after its read;
- a tile store's beat is taken from the engine's FIFO (`vme2lsu`), written
  to the bus the next cycle, and the instruction is complete the cycle after
  its last write: 2 + 1 per beat, one less than a vector store, which also
  hands each register back to the RVV backend;
- a store waiting for its data holds the slot (`load_then_store`: the next
  load's bus read follows the store's completion).

`lsu_v3_tile` runs tile beats through the LsuV3 slot this way, in place of
the fitted `lsu_tile_*` cost, which this config no longer sets.

It needed one change in the matrix module, `mtx_lsu_holds_full_fifo_beat`
(on for this config only). The module used to delay the *start* of a beat
that would find the 2-entry `lsu2vme` FIFO full. That equals the RTL's stall
only while the LSU reports a beat's data in its bus cycle, as the fitted
cost did. The LsuV3 path reports the push, 2 cycles after the read, so the
module now passes the FIFO's room to the LSU (`ReceiverReady`), which reads
the beat at once and holds it, and its slot, until then. Waveform of
`tile_load_behind_mac`: the third beat is read in cycle 71, offered from 73
and taken in 82, and the next bus read is 83.

The room itself is one cycle later than the module assumed: `lsu2vme` is
built without `FULL_PUSH`, so a full FIFO shows room the cycle after a pop,
not in the same cycle. The old rule had the two errors cancel.

With both, every tile program matches: `tile_load32_burst` 6.0,
`tile_store32_burst` 6.0, `load_then_mac` 18, `mac_then_store` 18,
`tile_load_behind_mac` 32, and `load_then_store` 14 (13 before, a known
gap).

## Audit against the goal (2026-11-15)

The goal: each switch is a piece of the RTL's behavior, read from the source
or a waveform, and separate enough to build a machine between M3 and this
one.

| Switch | Basis | Standing |
| --- | --- | --- |
| F1 fetch | RTL structure | read, not fitted |
| C1, V1 buffer sizes | RTL parameters | read |
| C2 `lsu_v3`, with `lsu_v3_skip_handshake` separate | waveforms | read; the handshake skip can be turned off alone |
| `lsu_v3_backend_lead` | fitted | one value, chosen by a held-out prediction |
| C3 `load_waw_advance` | waveform | observed; the RTL lines behind it not located |
| C4 branch rule | RTL source and waveform | read |
| V2, V2b tag matching | RTL source | read: results by tag, stores by PC |
| V3 `vec_fp_latency` | RTL parameter and measurement | read |
| V3b `vec_fdiv_cycles_per_uop` | fitted | measured only; no RTL reason found yet |
| V6 store completion | RTL source | read |
| M1 external bus | waveform | observed bus hold times; the RTL stages behind them not located |
| C2b, X4 tile beats | waveforms and RTL source | read; the fitted `lsu_tile_*` cost is no longer used here |
| X3 tile retire cycle | waveform | read; set on this config only |

The dispatch rules that were missing (slot 0 only, nothing after a jump, a
CSR waiting for an empty retirement buffer, the `vtdiscard` interlock) are
C5. Two results came with them. The LMUL 8 divide programs were 14% fast
not because of the new divider but because their closing `csrr mcycle` did
not wait for the divide uops still in flight. And `vtdiscard_after_mac`
matches with the retirement-buffer half of the interlock alone; the RVV-idle
and LSU-idle halves are still not modeled. Still in neither model: the
vector register file's read-port limit.

## Score

Mean absolute ROI error against the pinned RTL, and programs within 5%.

| State | core (20) | frontend (5) | vector units (32) | kernels (8) | all (65) | within 5% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M3 model, no switch | 42.3% | 8.6% | 29.4% | 43.5% | 33.5% | 26 |
| Values only (the table above) | 41.0% | 0.1% | 19.1% | 38.2% | 26.7% | 33 |
| + LsuV3, external bus, completion (C2, C3, V6, V7, M1) | 9.9% | 0.1% | 19.1% | 4.4% | 13.0% | 47 |
| + fetch structure (F1); V7 removed; scalar load retire cycle | 4.8% | 0.1% | 19.4% | 4.6% | 11.6% | 48 |
| + stalled redirect, branch rule, MAC retire cycle (C4, X3) | 3.5% | 0.1% | 19.0% | 4.2% | 11.0% | 52 |
| + stores by PC, dispatch rules (V2b, C5) | 3.4% | 0.1% | 17.9% | 4.9% | 10.4% | 54 |

Matrix programs (per-op cost, `matrix_diff`), within 5% of the RTL:

| State | of 44 | The GEMMs (`gemm_tile`, `gemm_multi`, `gemm_longk`) |
| --- | ---: | --- |
| `m3_vme.yaml` + `REFERENCE_CORE_PARAMS` (unchanged) | 31 | +67%, +55%, +67% |
| This config, values only | 31 | +67%, +56%, +67% |
| This config with LsuV3 | 35 | +10%, +4%, +12% |
| This config with LsuV3 and fetch | 37 | +10%, +4%, +12% |
| This config with C4 and X3 as well | 39 | +7%, +3%, +8% |
| This config with the dispatch rules (C5) | 40 | +6%, +1%, +8% |
| This config with tile beats on LsuV3 (C2b, X4) | 41 | +6%, +1%, +8% |

`vec_load_burst`, `vec_load4_burst`, `vec_store4_burst` and `gemm_multi`
joined with LsuV3; `tile_load`, `tile_store`, `move_tv8_burst` and
`move_vt8_burst` (short loops, +7% to +27% before) with fetch and the branch
rule; `vtdiscard_after_mac` (-77% before) with C5. None was used to build
any of them. The three outside 5% are `gemm_tile` (+5.6%), `gemm_longk`
(+7.5%) and `move_vt_use` (+15%, V8).