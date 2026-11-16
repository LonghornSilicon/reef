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

## Functional model

The pinned config runs on its own simulator, `reef_perf_pinned`. It is the
same model built against Spike as CoralNPU `8a5e1b31` builds it: Spike
`93a10ae` (`ext/spike-pinned`) with CoralNPU's 9 patches
(`third_party/spike-pinned/`, copied unchanged). That is one of the two
functional simulators CoralNPU runs in lockstep with this RTL (with MPACT,
`b4be3e79`). `reef_perf` keeps M3's Spike (`fd72ee2d`), so `m3.yaml`,
`m3_vme.yaml` and the M3 golden cannot be affected.

`experiments/correlation/pinned.py`, the pinned golden, `test_pinned_lsu.py`
and the pinned matrix gate run `reef_perf_pinned` (`find_pinned_binary()`).
Where it is not built they fall back to `reef_perf` with a warning. To build
it in the current image: `bash scripts/build_deps.sh build/deps spike-pinned`
inside Docker, then reconfigure. The Dockerfile now installs it in the image
as well (`/opt/reef-perf/spike-pinned`).

Zvt instructions are still executed by reef-perf's own layer
(`frontend/Zvt.cpp`), ahead of Spike, as before. On all 267 runs of the
pinned config (suites, kernels, the LSU and coverage probes at two loop
counts, the 44 matrix programs at two loop counts) the two simulators give
identical instruction counts and cycles. Next is letting the pinned Spike
execute Zvt itself, which also computes tile contents.

## Out of scope: floating point

Reef will most likely have no scalar floating-point unit (decided
2026-11-16). Until that changes, the floating-point programs are run but not
scored, and no further FP model changes are made:

- core: `fp_chain`, `fp_mix`;
- kernels: `attention`, `layernorm` (their work is FP: 96 and 38 FP
  instructions; the other kernels have 2);
- coverage: `pinned/fcvt_chain`, `fdiv_chain`, `fdiv_indep`, `fsqrt_chain`,
  `fmul_chain`, `flw_fsw`, `fp_to_vector`.

`OUT_OF_SCOPE` in `experiments/correlation/pinned.py` and in
`tests/python/integration/test_pinned_lsu.py` holds the list. Two FP values
were set before the decision and are left in `coral_pinned.yaml` as
measured (C7 `fpu_latency` 2, C7b `fdiv_latency` 13); they change nothing
else. The vector FP unit programs (`vfadd_*`, `vfmacc_*`, `vfdiv_*`,
`vfredusum_*`) are still scored: they match, and they are the RVV backend's.

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
| C6 | `rvv_config_interlock` (new) | false | true | `RvvFrontEnd.sv:86-93`, `Decode.scala` `rvvConfigInterlock`; waveform of `lsu/vload_macc` | `linear_int8`, `mlp_int8` |
| C7 | `fpu_latency` | 3 | 2 | waveform of `fp_chain` | `fp_chain` +41% to 0.0%, `fp_mix` +20% to 0.0% |
| F3 | `decode_unary_vs1_is_opcode` (new) | false | true | RVV 1.0 encoding; `pinned/vwmacc_m2_chain` | a decode fix, see "A decode bug" |
| V4 | `vec_completion_token_latency` (new) | 0 | 1 | CoralNPU `a0974da8`; measured (`lsu/vload_rob`, `vload_dep_rob`) | independent streams -7.7% to within 1% |
| V5 | `vec_results_forward_from_rob` (new) | false | true | measured (`pinned/vwmacc_m2_chain` 5 to 4) | a slow older uop no longer delays readers of a younger one |
| X5 | `mtx_move_vector_latency` 1, `mtx_move_vector_floor` 2 (new, matrix module) | 0 (`m3_vme.yaml`, unchanged) | set | measured: `move_vt_use` and three probes | `move_vt_use` +15% to 0.0% |
| C8 | `div_release_cycles` (new) | 0 | 1 | `Dvu.scala` (`io.req.ready`, `count`) | `div_chain` -4.4% to 0.0% |
| V8 | `vec_div_data_dependent` (new) | false | true | `intdivider.sv` | `vdiv_small_chain` +200% and `vdiv_small_indep` +333% to 0.0%; `pinned/fetch_mispredict_rob` 28 to 17 (exact) |
| C9 | `vector_scalar_waw_delay` (new) | 0 | 1 | waveform of `vcpop_indep` | `vcpop_indep` -9.9% to 0.0% |
| V9 | `vec_to_scalar_writes_per_cycle` (new) | 0 (no limit) | not set | `RvvCore` `async_rd` is one port | moves nothing today; left off |
| X1 | `zvt` | false | true | `--enableVme=True` | Zvt programs run |

## Differences not modeled yet

Each needs model code behind a new switch.

| ID | Difference | Evidence | Why not a value |
| --- | --- | --- | --- |

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

## A decode bug

`vsext.vf2`, `vzext`, `vmsbf`/`vmsif`/`vmsof`, `viota`/`vid`, `vfsqrt`, `vfcvt`
and `vfclass` carry a sub-opcode in their vs1 field. The frontend read it as
a source register, so `vsext.vf2 v2, v1` (vs1 field 7) depended on v7. In the
quantized linear kernel v7 is part of the accumulator, which chained every
`vsext` behind the previous `vwmacc`. `pinned/vwmacc_m2_chain` isolates it:
8 cycles per iteration with the bug, 4 on the RTL.

This is in the M3 model too. `decode_unary_vs1_is_opcode` fixes it and is on
here only, so M3's numbers stay as they are; the M3 kernel results that
match today (`linear_int8` +0.1% on M3) do so with this bug in place, where
the slower LSU hides it.

## The vector-to-scalar path and the linear loop

Four things, found one at a time with probes whose predictions were recorded
first (`microbenchmarks/lsu/heldout_predictions.csv`):

- **V4.** The retirement buffer sees a vector instruction done a cycle after
  its last uop leaves the vector ROB.
- **`lsu_v3_backend_lead` 1.** The waveform of `lsu/vload_macc` shows the LSU
  never waits for the vector backend: a loaded register is handed over 2
  cycles after its bus read, always. It is in the vector ROB from the cycle
  its uop is dispatched. The earlier value of 0 had been standing in for V4.
- **V5.** A dependent uop reads a result once it is in the vector ROB, not
  only after it retires in order.
- **C6.** A vector or tile load or store is not dispatched in the cycle
  after any vector-engine instruction, because the RVV front end reports
  its config state invalid while it holds an instruction. In the linear
  kernel's loop this is one idle dispatch cycle per iteration.

With F3 they bring `linear_int8` and `mlp_int8` to within 1%, and all 26
differenced probes in `lsu/` and `pinned/` match the RTL within 2% with no
known gap.

X5 is the matrix half: a `vtmv.v.t` register is readable the cycle after its
own uop and none sooner than 2 cycles after the first (the chain through v4,
v5, v6, v7 takes 13, 13, 14, 15 cycles on the RTL). The floor of 2 is
observed, not explained; it may be the reader's uop waiting behind the
move's four uops in the shared RVV dispatch.

## Dividers

The vector integer divider (`intdivider.sv`, one instance per element lane)
takes, per element:

| Case | Cycles |
| --- | --- |
| dividend of b significant bits (after taking the magnitude) | 1 + ceil(b / 3) + 1: 13 for 31 bits, 3 for 1 bit |
| divisor 0, or signed overflow | 2 |
| the same dividend, divisor and signs as that instance's previous operation | 2: the result is reused |

A uop takes its slowest element. The functional side records each element's
operands before Spike runs the instruction (`Inst::DivOperands`), and
`vec_div_data_dependent` uses them. This was also the cause of
`pinned/fetch_mispredict_rob`: its loop repeats one `vdiv` with the same
operands, so every divide after the first takes 2 cycles, not 13, and the
branch behind it waits that much less. The fetch unit was right.

The scalar divider (`Dvu.scala`) iterates once per quotient bit as the M3
model already had, and its result is on time. But it takes the next divide
only the cycle after its result is taken: `io.req.ready` waits for `count`
to clear, which happens the cycle after `rd` fires. `div_release_cycles` 1
holds the divider that extra cycle. A first attempt, `div_base` 3, also
fixed `div_chain` but made `pinned/fetch_mispredict_wait` (a branch reading
the result) a cycle slow, which ruled out a later result. The M3 model has
the same gap (`div_chain` -4.5% on M3).

## Vector writes to x registers

`vcpop_indep` rewrites a0 to a7 with `vcpop.m` every iteration. On the RTL a
`vcpop` that overwrites an x register last written by another vector
instruction dispatches the cycle after that write (a0 and a1 written in
cycles 62 and 63, the next iteration's pair dispatched in 64). A scalar
instruction overwriting the result goes in the write cycle itself
(`vec_pingpong`: `addi t0` after `vmv.x.s t0`), so the rule is only for a
vector writer (`vector_scalar_waw_delay`). The RTL lines behind it are not
located; `rvvRdMark` and the scoreboard's registered read are the likely
place. The vector engine also hands scalar results over one per cycle
(`async_rd`); `vec_to_scalar_writes_per_cycle` models that limit, but no
program reaches it, so it is off.

## Coverage

`microbenchmarks/pinned/` holds 31 programs written for behavior no earlier
program exercised, each with the model's prediction recorded before its RTL
run (`heldout_predictions.csv` there; committed in `caba13e`, RTL numbers a
commit later). 25 of the 31 predictions were exact. The six that were not:

| Program | RTL | Predicted | What it shows |
| --- | ---: | ---: | --- |
| `fetch_mispredict_rob` | 17 | 28 | the vector divider reuses its last result for repeated operands; now modeled (V8), exact |
| `fcvt_chain` | 48 | 17 | out of scope (FP) |
| `fdiv_chain` | 144 | 96 | out of scope (FP) |
| `fdiv_indep`, `fsqrt_chain` | 104 | 96 | out of scope (FP) |
| `flw_fsw` | 57 | 49 | out of scope (FP) |

No known gaps remain among the in-scope programs. What the other 25
cover and confirm: a loop longer than the instruction buffer, jump targets
in the middle of a line, a taken jump in every line, a mispredict that waits
for its branch to dispatch; masked and indexed accesses, accesses that cross
a line, e16 and e32 widths, a short `vl`, scalar and vector accesses
interleaved; `fmul`, an FP result feeding a vector op; `vsext.vf4`, `vid`,
`viota`, `vfsqrt`, a masked chain, a reduction through `vmv.x.s`, more uops
than the vector ROB holds; MACs with vector adds, operand loads with MACs,
tile stores with vector loads.

Still not covered: the 4 large Coral kernels (no highmem verification build)
and the Zvt firmware kernels (they need a variant without the DMA).

## Audit against the goal (2026-11-15)

The goal: each switch is a piece of the RTL's behavior, read from the source
or a waveform, and separate enough to build a machine between M3 and this
one.

| Switch | Basis | Standing |
| --- | --- | --- |
| F1 fetch | RTL structure | read, not fitted |
| C1, V1 buffer sizes | RTL parameters | read |
| C2 `lsu_v3`, with `lsu_v3_skip_handshake` separate | waveforms | read; the handshake skip can be turned off alone |
| `lsu_v3_backend_lead` | waveform of `lsu/vload_macc` | read: the LSU never waits for the backend |
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
| + FPU, completion token, ROB forwarding, config interlock, decode fix (C6, C7, F3, V4, V5) | 0.2% | 0.1% | 17.0% | 1.4% | 8.6% | 61 |

Against the 3% target (set 2026-11-15), 59 of 65 are inside. Outside:
`vdiv_small_chain` and `vdiv_small_indep` (the divider's early finish, which
is all of the vector-unit mean), `vcpop_indep` -9.9%, `layernorm` +6.8%,
`div_chain` -4.4%, `attention` +3.4%.
| + dividers, vector x-register WAW; FP not scored (C8, C9, V8) | 0.0% | 0.1% | 0.0% | 0.2% | 0.0% | 61 of 61 within 3% |

Matrix programs: 44 of 44 within 5%, 42 within 3%. Outside 3%:
`vtmm_misaligned` (5.75 against 6.0; ROB-bound, bursty on the RTL, no
mechanism found yet) and `gemm_longk` (+3.6%).

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
| This config with steps 1 and 2 (C6, C7, F3, V4, V5, X5) | 44 | +2%, +1%, +4% |

`vec_load_burst`, `vec_load4_burst`, `vec_store4_burst` and `gemm_multi`
joined with LsuV3; `tile_load`, `tile_store`, `move_tv8_burst` and
`move_vt8_burst` (short loops, +7% to +27% before) with fetch and the branch
rule; `vtdiscard_after_mac` (-77% before) with C5. None was used to build
any of them. Against the 3% target, 42 of 44 are inside; outside are
`vtmm_misaligned` (-4.2%) and `gemm_longk` (+3.6%).