# Frontend correlation with the M3 RTL

Status: v0, for the 10/18 review. Owner: Frontend.

This report explains where reef-perf's frontend disagrees with the CoralNPU M3
RTL, which RTL behaviors the model is missing, and how much of the error each
one accounts for. It goes with the frontend v0 spec
([frontend-v0-spec.md](frontend-v0-spec.md)), which has the full RTL
description with citations.

## 1. Summary

- **Baseline:** the super-coarse model on `reef-perf-modules` (`5e08f31`) is
  37.6% off the RTL on average (the `correlation` experiment); the RTL and the
  model retire the same instructions in every loop (15/15), so the cycle
  errors compare like with like.
- **Three microbenchmarks are frontend-bound:** `alu_indep`, `branch_group` and
  `branch_tight`. Their errors are explained, to within 0.1%, by three M3
  fetch behaviors the model lacks:
  1. **Fetch every other cycle** (D7). Added in this PR (B04).
  2. **Static branch prediction:** a correctly predicted backward branch costs
     no extra fetch cycle. The model charges `redirect_penalty` on every taken
     branch instead. Next: B05.
  3. **16-byte aligned fetch blocks:** a loop that starts mid-block needs an
     extra fetch per iteration. Next: B06.
- **The other microbenchmarks are not frontend-bound** (in the model, fetch never
  runs dry on them). Their errors also include dispatch rules, the LSU, the
  divider, the FPU and the vector unit. So those will be second priority once
  correlation accuracy is achieved in frontend microbenchmarks

| Model | Mean absolute error |
| --- | --- |
| Baseline (`fetch_interval: 1`, `redirect_penalty: 1`) | 37.6% |
| B04, this PR (`fetch_interval: 2`, `redirect_penalty: 1`) | 36.1% |
| B04 + predicted-taken branches free (what B05 gives these loops) | 34.0% |
| + 16-byte alignment (B06), derived in section 4.3 | about 33% |

The mean moves little.
Expecting most of the error is in backend-bound
microbenchmarks. On the three frontend-bound microbenchmarks the error goes from
-41.7 / -30.0 / -0.1% to about 0 / 0 / -0.1%.

## 2. Method

- **RTL:** CoralNPU `M3-2026-04-27`, `rvv_core_mini_verification_axi_sim`,
  one run per microbenchmark. Loop cycles are `s11 - s10` from the two
  `csrr mcycle` markers (`microbenchmarks/rtl_m3.csv`, B01). The verification
  build's timing matches production to within 2 cycles (spot check of total
  cycles to `mpause` on `alu_chain`, `load_chain` and `gemv_int8`).
- **Model:** reef-perf on `reef-perf-modules` (`5e08f31`). The ROI counters
  (B02) report the cycles between the same two markers, so both sides measure
  only the loop.
- **What-if runs:** the same binary with `-p` overrides of
  `top.frontend.params.fetch_interval` and `redirect_penalty` (commands in
  section 8).
- **Limits:** the RTL reports total loop cycles only, not stall reasons, so
  every stall breakdown below is the model's. Each `csrr mcycle` drains the
  pipeline on M3 (deviation M2); with 1000+ iterations per loop this changes
  results by a few cycles.

## 3. Where the M3 frontend and reef-perf differ

Paths are relative to `hdl/chisel/src/coralnpu/` in the M3 checkout.

| Behavior | M3 RTL | reef-perf today | Effect on the model | Ticket |
| --- | --- | --- | --- | --- |
| Fetch rate | One 16-byte fetch in flight; no new request while response data is valid, so at most 4 instructions per 2 cycles (`scalar/UncachedFetch.scala:194-195`, D7) | Up to 4 instructions every cycle (`fetch_interval: 1`) | Too fast on straight-line code | **B04, this PR** |
| Taken-branch cost | Static prediction in predecode: `jal` and backward conditional branches predicted taken, target computed from the instruction bits, next fetch goes straight to it (`UncachedFetch.scala:87-100,133-134,177-185`). No extra cycle when correct | Flat `redirect_penalty` (1) on every taken branch or jump, no prediction | After B04: too slow on every loop | B05 |
| Mispredicts | The BRU redirects fetch one cycle after dispatch only when the prediction was wrong: forward taken, backward not taken, every `jalr` (`Bru.scala:142-143,202-212`). Penalty not yet measured (U1) | Same flat penalty, and no penalty for not-taken branches | Unknown; no microbenchmark has mispredicts yet | B05 |
| Return / indirect prediction | None: no BTB, no return-address stack (`jalr` never predicted) | None | Matches | - |
| Fetch alignment | Requests are 16-byte aligned; a fetch at a mid-block PC delivers only the rest of the block; a predicted-taken branch ends its group (`UncachedFetch.scala:56,107-131`) | Groups of 4 at any address | Too fast on loops that start mid-block | B06 |
| Instruction buffer | 8 entries inside the fetch unit; a new fetch needs `nValid + 4` free entries (`UncachedFetch.scala:192,234`); one cycle from enqueue to visible (`common/InstructionBuffer.scala:34`) | 8 entries in the frontend (`ibuf_entries`; `Fetch` owns the buffer); fetch only needs one free entry | Small; not isolated yet | v0 spec Q4 |
| Redirect path | BRU `taken` to fetch, flushing the buffer (`scalar/SCore.scala:120-122`, `UncachedFetch.scala:239`) | No core-to-frontend signal; fetch knows the outcome from Spike | Fine for timing today; needed for real mispredict modeling | Interface PR |
| Instruction memory | Fetch reads the ITCM through the core instruction bus; LSU loads from the ITCM (`.rodata`) take the bus first (`scalar/SCore.scala:484-497`); 1-cycle ITCM read | Fetch is free (no `IFETCH` request to the memory module) | None on current microbenchmarks (data is in the DTCM); real kernels with `.rodata` would stall fetch | Interface PR |
| Dispatch-group rules | A conditional branch or jump ends the group; slot-0-only ops; registered scoreboard for addresses; CSRs wait for an empty ROB (`scalar/Decode.scala:313-327,347-351,400-406,519`) | None of these | See section 4.4 (core) | B07-B10 |

## 4. Evidence

### 4.1 Frontend-bound microbenchmarks, three fetch configurations

Loop cycles; `ibuf %` is the share of cycles the model's dispatch had nothing
to dispatch (frontend starved).

| Workload | RTL | Baseline | err | B04 | err | ibuf % | B04 + free taken | err | ibuf % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `alu_indep` | 12003 | 7001 | -41.7% | 10999 | -8.4% | 45.4 | 10000 | -16.7% | 40.0 |
| `branch_group` | 10005 | 7001 | -30.0% | 10999 | +9.9% | 45.4 | 10000 | -0.0% | 40.0 |
| `branch_tight` | 10005 | 9998 | -0.1% | 14998 | +49.9% | 33.3 | 9999 | -0.1% | 0.0 |

- **Baseline:** fetch never runs dry (`ibuf_empty` = 1 cycle in every
  microbenchmark), so the model sustains up to 4 instructions per cycle where M3
  manages about 2.
- **B04:** fetch spacing makes these microbenchmarks fetch-bound, as on M3
  (`ibuf_empty` becomes the top stall). `alu_indep` moves from -41.7% to
  -8.4%.
- **But B04 alone over-charges taken branches.** The model now waits
  `fetch_interval + redirect_penalty` = 3 cycles after every taken branch.
  `branch_tight` is a 2-instruction loop ending in a backward branch: the
  model takes 3 cycles per iteration (14998), M3 takes 2 (10005). M3 predicts
  the branch taken and fetches the target on the normal 2-cycle rhythm.
- **Setting the taken-branch cost to 0** (what B05's static predictor gives a
  correctly predicted backward branch) brings `branch_tight` and
  `branch_group` to within 0.1% of the RTL. `alu_indep` stays at -16.7%:
  section 4.3.

### 4.2 Why `redirect_penalty: 0` is not the fix

Every loop in the current microbenchmarks ends in a backward branch, which M3
predicts correctly, so `redirect_penalty: 0` happens to fit all of them. It
would also make mispredicts and every `jalr` free. B05 keeps the two apart:
`taken_bubble` (cost of a correctly predicted taken branch, measured here as
0) and `mispredict_penalty` (to be measured with new microbenchmarks).

### 4.3 Fetch alignment explains the rest of `alu_indep`

Where each loop starts within its 16-byte fetch block (from `reef_trace`):

| Workload | Loop start | Slot in block | Loop length | Fetches per iteration with M3 alignment | Cycles per iteration (2 per fetch) | RTL cycles per iteration |
| --- | --- | --- | --- | --- | --- | --- |
| `alu_indep` | `0x1c` | 3 | 18 | 1 + 4 + 4 + 4 + 4 + 1 = **6** | 12 | 12.0 |
| `branch_group` | `0x20` | 0 | 18 | 4 + 4 + 4 + 4 + 2 = 5 | 10 | 10.0 |
| `branch_tight` | `0x20` | 0 | 2 | 1 | 2 | 2.0 |
| `alu_chain` | `0x1c` | 3 | 18 | 6 | 12 (not the limit: 16-deep dependency chain) | 18.0 |

Without alignment the model packs `alu_indep`'s 18 instructions into 5
fetches (10 cycles per iteration, 10000 total). M3 needs 6, because the loop
starts at the last slot of a block: 12 cycles per iteration, which is the
RTL's 12003 to within the marker overhead. With B04, free predicted-taken
branches and B06, all three frontend-bound microbenchmarks should land within 0.1%
of the RTL. (Derived from the code and the trace; confirmed once B06 is in.)

### 4.4 The other microbenchmarks are not limited by the frontend right now

In all three configurations, fetch never starves the other twelve microbenchmarks
(`ibuf_empty` at most 1.5% after removing the double-counted branch cost), and
their errors do not change with the fetch parameters (except `load_chain`,
whose B04-only change disappears once taken branches are free). Their top
stalls in the model are register hazards, LSU and vector:

| Workload | RTL | Model (B04 + free taken) | err | Model top stall | Likely cause | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| `alu_chain` | 18004 | 16000 | -11.1% | none (dependency chain) | Loop overhead not overlapping on M3: branch ends the group, registered-scoreboard rules (D2, D4) | Core (B07, B09) |
| `addr_forward` | 32005 | 16001 | -50.0% | WAW | Address operands use the registered scoreboard (D4) | Core (B09) |
| `div_chain` | 18003 | 25602 | +42.2% | RAW | Data-dependent divider (U10) | Core (B12) |
| `fp_chain` | 8505 | 12000 | +41.1% | RAW | FPU latency (U2) | Core (B11) |
| `load_chain`, `load_stream` | 10002, 8204 | 4000, 2072 | -60.0, -74.7% | ROB, none | LSU: one slot, 16-byte lines (D5, U3) | Core (B13) |
| `vec_add_m1`, `vec_add_m4`, `vec_memcpy`, `gemv_int8` | | | -37 to -70% | RAW | Vector units, vector memory through the LSU (U4-U8) | Vector (B14-B18) |
| `fp_mix`, `mul_chain` | | | -0.0, -0.1% | | Already close | |

## 5. What this PR changes

1. **B04:** `fetch_interval: 2` in `configs/machines/m3.yaml`, citing
   `UncachedFetch.scala:194-195`. Golden regenerated (timing changes on
   purpose); `test_sim` now compares fetch interval 1 and 2 explicitly.
2. **This report and the frontend v0 spec.**

Correlation after this PR: MAE 36.1%, 15/15 instruction counts match.
`branch_tight` and `branch_group` get worse with B04 alone (section 4.1); they
are fixed by B05, which is next.

## 6. Remaining gaps and owners

| Gap | Owner | Status |
| --- | --- | --- |
| Static prediction: `taken_bubble` = 0 (measured here), `mispredict_penalty` and `jalr` cost (need new microbenchmarks: `branch_fwd_taken`, `branch_bwd_not_taken`, `jalr_chain`) | Frontend | Next (B05) |
| 16-byte aligned fetch blocks (`fetch_align_0..3` microbenchmarks to confirm on the RTL) | Frontend | Next (B06) |
| The `nValid + 4` fetch rule (the buffer itself now lives in the frontend) | Frontend | Open |
| Core-to-frontend redirect signal | Frontend + core | Interface PR |
| Instruction fetch through the memory module (`IFETCH`), `.rodata` loads stalling fetch | Frontend + memory | Interface PR |
| Dispatch-group rules (branch ends group, slot 0, registered scoreboard, CSR drain) | Core | B07-B10 |
| RTL stall counters (the RTL gives no stall breakdown, so frontend starvation on M3 is inferred, not measured) | Frontend | Proposal: a buffer-empty counter in the M3 Chisel |

## 7. Next steps

1. B05: implement `predictor: static_btfn` with `taken_bubble: 0` and a
   measured `mispredict_penalty`; add the mispredict microbenchmarks and measure them
   on the RTL.
2. B06: `aligned_blocks: true`; add `fetch_align_0..3` and confirm the
   section 4.3 numbers on both sides.
3. Agree with the core owner where the group-ending rules live, then open the
   wrapper-interface PR (redirect, `IFETCH`).

## 8. Reproducing

Inside the container, from `reef-perf/`:

```sh
# Correlation table (uses configs/machines/m3.yaml)
uv run python -m reef_perf.experiments.correlation.main

# One microbenchmark under the three fetch configurations
for p in "1 1" "2 1" "2 0"; do
  set -- $p
  build/bin/reef_perf --elf build/microbenchmarks/branch_tight.elf \
    -c configs/machines/m3.yaml \
    -p top.frontend.params.fetch_interval $1 \
    -p top.frontend.params.redirect_penalty $2 | grep "ROI cycles"
done

# Where a loop starts within its fetch block
build/bin/reef_trace build/microbenchmarks/alu_indep.elf 60
```
