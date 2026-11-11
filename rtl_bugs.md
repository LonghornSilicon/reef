# CoralNPU M3 RTL bugs

Bugs found in the CoralNPU RTL that Reef starts from, release tag
`M3-2026-04-27` (`coralnpu@72d700ef`). Most turned up while correlating the
performance model (`reef-perf/`) against the RTL. Timing disagreements that
aren't bugs are in [reef-perf/docs/deviations.md](reef-perf/docs/deviations.md).

Each entry says where the bug is, how it was seen, what it breaks, whether
CoralNPU has fixed it upstream, and what Reef and reef-perf do about it. File
paths are relative to the `coralnpu` repository at M3 unless stated otherwise.

| ID | Bug | Upstream |
| --- | --- | --- |
| [RB1](#rb1-the-retirement-buffer-completes-vector-instructions-by-pc) | The retirement buffer completes vector instructions and vector stores by PC, so in a short loop a later iteration retires before its own work is done | Fixed for vector results on `main` (`302a42d3`, 2026-08-21); stores still matched by PC |

## RB1: The retirement buffer completes vector instructions by PC

**Where.** `hdl/chisel/src/coralnpu/RetirementBuffer.scala`.

- Lines 268-272: a vector write counts for a buffer entry holding a vector
  instruction when `x.bits.uop_pc === bufferEntry.addr`, the instruction's PC.
- Lines 284-286: the entry is ready when such a write has
  `last_uop_valid`.
- Line 326: a store entry is ready when
  `storeComplete.bits === bufferEntry.addr`.
- `scalar/Lsu.scala:1003-1006,1039-1041`: for a vector store, the LSU raises
  `storeComplete` with the store's PC (`slot.pc`) on every register's
  writeback, not only the last (`bits.last` is set on every vector store
  writeback, despite its name).

Each entry checks the incoming writes on its own, and nothing restricts a write
to the oldest matching entry. Retirement then drains every completed entry in
order (`countValid`, line 410).

**Why it's a bug.** The class comment (lines 42-50) defines an entry as
completed when "all side effects (register writes, store completions, or
faults) are committed to the architectural state". A PC doesn't identify an
instruction in flight: in any loop shorter than the 8-entry buffer, several
instances of the same vector instruction are in it at once. The first one to
finish marks all of them completed. Scalar entries avoid this because they
match on the destination register, and the scalar scoreboard never lets two
writers of the same register be in flight. Vector registers have no such
scoreboard.

**How it was seen.** RTL waveforms (`~/simbins/sim_O2 --trace`, M3 tag):

- **Vector results.** A loop of two independent `vdiv.vv` at LMUL=8
  (8 uops each) plus `addi`/`bnez`, 2 iterations: the second iteration's
  divides were marked done when the first iteration's finished. The buffer
  drained and the `csrr mcycle` that ends the measurement ran while 16 divide
  uops were still in the backend. The ROI measured 217 cycles; the divider
  alone needs 32 uops x 13 cycles.
- **Vector stores.** A loop of `vse32.v` at LMUL=8 (8 registers) plus
  `addi`/`bnez`: the first `storeComplete` of the first instance (cycle 59)
  retired three instances of the store at once (cycle 60). The LSU hadn't
  written any register of the second and third instances yet. The ROI measured
  17 cycles; the LSU kept storing until cycle 155.

The microbenchmark `reef-perf/microbenchmarks/vector_units/vdiv_m8_indep.S`
reproduces it at full length: 3384 cycles on the RTL, against at least 5200
cycles of divider work.

**What it breaks.**

- An instruction retires before its results exist, against the buffer's own
  definition of completed.
- Anything that waits for an empty retirement buffer to mean "all earlier work
  is done" is wrong after a short vector loop. CSRs wait for it (`csrr mcycle`
  among them), so `mcycle` measurements of short vector loops end early.
- The upstream fix's commit message adds "premature retirement and data
  corruption", and in-flight vector operations racing post-trap instructions
  when an exception flushes them. We haven't reproduced the corruption or the
  trap race ourselves.

**Upstream fix.** CoralNPU commit
`302a42d36c5964b5c33fc0c260726ae9b2450cf0` on `main`, dated 2026-08-21,
"Implement ROB tag metadata and RVV pipeline flush". It carries a `rob_tag`
through the RVV pipeline (RvvFrontEnd, decode, uop queue, dispatch, vector
ROB, `rd_rob2rt_o`). The retirement buffer then matches vector results on
`rob_tag === pIdx` instead of the PC (`RetirementBuffer.scala:452,532` on
`main`). It adds the test `tests/cocotb/rvv/rvv_small_loop_test.S`. We checked
this against `main` at `08eadd1c` (2026-10-14).

**Still open upstream: stores.** On `main` at `08eadd1c`, a store entry is
still ready when `storeComplete.bits === bufferEntry.addr`
(`RetirementBuffer.scala:558`). The rewritten LSU raises `storeComplete` only
once all of a store's cells are done (`scalar/Lsu.scala:1923-1924` on `main`),
which removes the early per-register signal. But a finished store can
probably still complete a newer in-flight instance of the same store. This is
from reading the source; it hasn't been simulated.

**What Reef should do.** Reef starts from M3, so it has this bug. Take the
upstream `rob_tag` fix, and match stores by tag too.

**What reef-perf does.** The model reproduces the bug by default, because the
M3 RTL is its correlation reference: `vec_completion_matches_pc: true` in
`reef-perf/configs/machines/m3.yaml` (implemented in
`Vector::getScalarComplete()`, `reef-perf/model/vector/Vector.cpp`). Set it to
`false` to model the fixed behavior, where each instance completes on its own
work:

```sh
reef_perf --elf x.elf -c configs/machines/m3.yaml \
    -p top.vector.params.vec_completion_matches_pc false
```

Deviation U8 in `reef-perf/docs/deviations.md` has the timing details.
