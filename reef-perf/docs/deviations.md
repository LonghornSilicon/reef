# RTL Deviation Register

Reef's baseline is the CoralNPU M3 release. This register lists the places
where our references disagree with the M3 RTL, or where the RTL's timing is
still unknown. It records each problem. It doesn't resolve any of them; that
work is tracked in tickets. When a ticket closes, update the entry's status and
link the ticket.

## Source of truth

- **The CoralNPU RTL at release `M3-2026-04-27` is the source of truth.** The
  repo docs shipped with that release, and the Spike build reef_perf uses as its
  functional model, are references that can be wrong.
- **The datasheet is out of scope for now.** A short note on it is in the appendix.

| Repo | Pinned commit | How it's pinned |
| --- | --- | --- |
| `coralnpu` | `72d700ef` (tag `M3-2026-04-27`) | Release tag; not a submodule (it is only the correlation reference) |
| `riscv-isa-sim` (Spike) | `fd72ee2d` (2025-12-17) | `reef-perf/ext/spike` submodule; the same commit `coralnpu/rules/deps.bzl` pins at M3 |

`coralnpu` main is 334 commits past M3. Correlation work must use the M3 tag.
The appendix lists what changes after M3, for when the pin moves.

### Reference configuration (M3)

Only two core families exist at M3: `CoreMini` and `RvvCoreMini`. There is no
VME/Zvt, and there's no BF16.

| Parameter | `Parameters.scala` default | M3 build targets |
| --- | --- | --- |
| `fetchDataBits` | 256 (`Parameters.scala:109`) | **128** (`hdl/chisel/src/coralnpu/BUILD:572-577`) |
| `lsuDataBits` | 256 (`Parameters.scala:119`) | **128** |
| `enableFetchL0` | true | **false**, so `UncachedFetch` is used (`scalar/SCore.scala:57`) |
| `enableFloat` | false | **true** |
| `retirementBufferSize` | 8 (`Parameters.scala:95`) | 8 |
| RVV backend | - | `DISPATCH3` (`hdl/verilog/rvv/inc/rvv_backend_config.svh:5`); vector ROB depth 8 (`rvv_backend_define.svh:35`) |
| SV defines | - | `VLEN_128`, `ZVE32F_ON`, `TB_SUPPORT`, `USE_GENERIC`, passed as Verilator `vopts` (`BUILD:653-658`) |
| Toolchain | - | `-march=rv32imf_zve32f_zicsr_zifencei_zbb` (`toolchain/cc_toolchain_config.bzl:163`) |
| Memories | 8 KB ITCM / 32 KB DTCM | Default, 512 KB/512 KB and 1 MB/1 MB (highmem) variants. DTCM is a single SRAM, not banked. |

## Entry format

- **ID**: the class prefix plus a number. The classes are D, U, M, F and A,
  described below.
- **Status**:
  - `code-confirmed`: seen in the RTL, but the cycle-level effect hasn't been
    measured.
  - `hypothesis`: inferred, not yet checked.
  - `measured`: closed by a benchmark. Link the result.
  - `won't-fix`: accepted as-is.
- **Order**:
  - `1st`: expected to move kernel cycle counts by more than ~5%.
  - `2nd`: a smaller effect.

All file paths are relative to `coralnpu/` at M3, unless stated otherwise.

## D: Repo docs vs RTL

The docs shipped with M3 describe something different from what the M3 RTL does.

| ID | Docs say | RTL does | Evidence | Status | Order |
| --- | --- | --- | --- | --- | --- |
| D1 | `doc/overview.md` describes the legacy Kelvin design: custom SIMD instead of RVV, 64 × 256-bit vector registers, an 8×8×32 accumulator, an rv32im frontend with no F, and reclaimed C-extension encoding. | The M3 RTL is RV32IMF + Zbb + RVV 1.0 (Zve32f), with VLEN=128, 32 vector registers and no accumulator array. The whole overview is stale. | `Parameters.scala`, `hdl/verilog/rvv/`, toolchain `-march` | code-confirmed | none directly, but it misleads anyone reading it |
| D2 | `doc/microarch/dispatch.md`: after a branch, only ALU/BRU ops may dispatch. | Any conditional branch **ends the dispatch group**. No later lane dispatches that cycle, so there's at most one conditional branch per cycle. | `scalar/Decode.scala:322-327` | code-confirmed | 1st |
| D3 | `dispatch.md` lists the slot-0-only instructions as CSR ops, `ebreak`, `ecall`, `mret`, `fence`, `fence.i` and `wfi`. | The list also includes **every F-extension instruction** (including `flw`/`fsw`), every RVV instruction that reads or writes an f-register (`.vf` forms, `vfmv.f.s`), and `mpause`, `flushat` and `flushall`. All of them dispatch alone from slot 0. | `scalar/Decode.scala:163-169` | code-confirmed | 1st |
| D4 | `dispatch.md`: all units read operands the cycle after dispatch. No forwarding restrictions are mentioned. | The base register of a load/store, the source of a `jalr`, and the data register of a scalar store are checked against the *registered* scoreboard (`regd`), not the forwarded one (`comb`). A value produced the cycle before can't be used as an address. | `scalar/Decode.scala:347-356` | code-confirmed | 1st |
| D5 | `dispatch.md`: "Memory today is limited to dispatching one instruction per cycle." | Several LSU ops can dispatch per cycle, up to the free space in a 4-entry queue. Execution is serialised through **one 16-byte slot**. | `scalar/Decode.scala:459-463`, `scalar/Lsu.scala:835,880` | code-confirmed | 1st |
| D6 | `doc/overview.md`: backward branches predicted taken, forward not taken; **one penalty cycle** on a mismatch. | Predecode also predicts `jal` as taken, and never predicts `jalr` (it always redirects from the BRU). The size of the penalty is unmeasured (see U1). | `scalar/UncachedFetch.scala:87-93` | hypothesis | 1st |
| D7 | `doc/microarch/microarch.md`: dispatch of up to 4 instructions per cycle from the instruction buffer. No fetch-bandwidth limit is mentioned. | The fetcher issues **at most one 128-bit fetch every other cycle**: new fetches are blocked while response data is valid. That caps sustained fetch at ~2 instructions per cycle, so scalar IPC is capped at ~2 despite 4-wide dispatch. Only one fetch is in flight. | `scalar/UncachedFetch.scala:44-71,194-195` | code-confirmed | 1st |
| D8 | `doc/microarch/microarch.md`, `mlu.md`: MLU latency is 2. | The MLU is an arbiter, then compute, then writeback, with a 1-entry queue between each stage. Effective latency is 2 or 3 cycles. Only one MLU exists. | `scalar/Mlu.scala:67-101` | hypothesis | 2nd |

The M3 `doc/microarch/lsu.md` ("one slot") matches the RTL. It has no timing
detail, which is tracked in U3 and U6.

## U: RTL timing unknowns

Reading the code isn't enough to pin these down; each needs a measurement.

| ID | Unknown | Where it lives | Status | Order |
| --- | --- | --- | --- | --- |
| U1 | Cost of a predicted-taken branch; mispredict penalty; `jalr` redirect penalty | `scalar/UncachedFetch.scala`, `scalar/Bru.scala` | hypothesis | 1st |
| U2 | FloatCore latency and throughput per op. It's fpnew with `PipeRegs=3` on ADDMUL/NONCOMP/CONV and a merged, iterative div/sqrt. | `float/FloatCore.scala:112-125` | hypothesis | 2nd |
| U3 | LSU scalar timing: load-to-use latency, the cost of a line-crossing access, when a store counts as complete, external-bus latency | `scalar/Lsu.scala:824-1000` | hypothesis | 1st |
| U4 | Scalar-to-RVV crossing: dispatch-to-execute latency, `vsetvl*` cost, RVV-to-scalar writeback latency (`vmv.x.s`, reductions, `vfmv.f.s`) | `rvv/RvvCore.scala`, `hdl/verilog/rvv/design/RvvFrontEnd.sv` | hypothesis | 1st |
| U5 | Vector FU latency and throughput per op class × SEW × LMUL; how instructions split into micro-ops (LMUL > 1, widening, narrowing) | `rvv_backend_decode_unit_*.sv`, FU modules | hypothesis | 1st |
| U6 | Vector load/store cost per register. Each vector register passes through the single 16-byte slot as vector-update → one bus transaction per distinct 16-byte line → writeback. The cost per addressing mode (unit-stride aligned/misaligned, strided, indexed, segment) is unknown. | `scalar/Lsu.scala:362-632,880-1000` | hypothesis | 1st |
| U7 | The vector backend is built with `NUM_LSU=2`, but the M3 scalar LSU only services `rvv2lsu` port 0. Does this limit vector memory throughput, or only one of the two backend LSU paths? | `rvv_backend_define.svh` (`NUM_LSU`), `scalar/Lsu.scala:891` | hypothesis | 1st |
| U8 | How the 8-entry scalar retirement buffer and the 8-entry vector ROB stall dispatch behind long-latency ops | `Parameters.scala:95`, `RetirementBuffer.scala:328`, `rvv_backend_define.svh:35`, `scalar/Decode.scala:517` | hypothesis | 1st |
| U9 | Stalls from VRF read-port conflicts (6 ports, 3 micro-ops/cycle, 3-source ops) | `rvv_backend_dispatch*.sv` | hypothesis | 2nd |
| U10 | DVU latency formula (radix-2 with a leading-zero early skip) | `scalar/Dvu.scala:90-136` | hypothesis | 2nd |

## M: Measurement methodology

These are problems with how we measure the RTL, not with the RTL itself.

| ID | Issue | Evidence | Status | Order |
| --- | --- | --- | --- | --- |
| M1 | Production builds run the retirement buffer in `mini` mode, so `--instr_trace` prints no instruction bits. At M3, the retirement-ready condition (`dataReady`) is **the same in both modes**; the extra verification state only feeds the debug output. Verification-build traces should therefore have production timing. Confirm this with one measurement before relying on it. | `RetirementBuffer.scala:284-328`, `scalar/SCore.scala:64` | code-confirmed | 2nd |
| M2 | CSR instructions, including `csrr mcycle`, wait until the retirement buffer is empty and dispatch alone from slot 0. So `mcycle` brackets drain the pipeline. They measure latency correctly, but they distort short throughput measurements. | `scalar/Decode.scala:519`, `:167-169` | code-confirmed | 1st |
| M3 | External memory timing comes from the Verilator testbench's AXI/TLM model, not a real memory system. External-memory results correlate against that model only. | `tests/verilator_sim/` | hypothesis | 1st for LLM decode |
| M4 | M3 has **no per-instruction cycle benchmark**. `tests/cocotb/isa_cycle_bench.cc` was added after M3. It's software only, so it could be backported, but it must build with the M3 toolchain. | `tests/cocotb/` at M3 | code-confirmed | 1st |
| M5 | The M3 workload set is thin: `tests/cocotb/rvv/ml_ops` has only `rvv_matmul` (int8) and `rvv_float_matmul`. The Gemma kernels were added after M3, and some of them use BF16, which isn't in the M3 toolchain. | `tests/cocotb/rvv/ml_ops/` at M3 | code-confirmed | 1st |
| M6 | Default ITCM/DTCM (8 KB/32 KB) is too small for most kernels, so correlation runs need highmem builds. Check whether highmem changes any timing. | `hdl/chisel/src/coralnpu/BUILD` | hypothesis | 2nd |
| M7 | The CoralNPU M3 RTL is not part of the reef-perf build: correlation runs use a separate `coralnpu` checkout at tag `M3-2026-04-27`. Make sure it is at that tag, not main. | - | code-confirmed | setup |

## F: Spike (functional model) vs RTL

reef_perf uses pristine Spike at `fd72ee2d` with ISA
`rv32imf_zicsr_zifencei_zbb_zve32f_zvl128b` (`spike_driver.hpp`). CoralNPU's own
M3 regression runs that Spike commit with five patches
(`coralnpu/third_party/spike/*.patch`); reef_perf applies none of them.

| ID | Gap | Evidence | Status | Order |
| --- | --- | --- | --- | --- |
| F1 | Spike does not implement `mpause` (`0x08000073`), which ends programs on Reef. The driver stops when it reaches `mpause`, without executing it. CoralNPU's patch 0001 instead makes Spike call `exit(0)`, which would kill the simulator process. | `reef-perf/src/csrc/src/spike_driver.cpp` (`kMpauseEncoding`); `coralnpu/third_party/spike/0001-Add-mpause.patch` | code-confirmed | none (handled) |
| F2 | Spike lacks CoralNPU's custom CSRs (`0xFC0`-`0xFD4` KISA/KSCM, `0x7C0`-`0x7C7` MCONTEXT, `0x7E0`/`0x7E1`) and reports `mvendorid` = 0. Code that touches them traps in reef_perf, and the driver reports the trap. The current workloads don't use them. | `coralnpu/third_party/spike/0002-*.patch`, `0004-*.patch` | code-confirmed | 2nd |
| F3 | Vector policy choices: Spike's tail/mask-agnostic fill values, fault-only-first and `vstart` behaviour may differ from the RTL. These change data values, rarely the instruction stream, so they rarely affect timing. | - | hypothesis | 2nd |
| F4 | Spike decodes the Kelvin-era `flushat`/`flushall` ops that the M3 decoder still recognises as illegal instructions (they trap). | `scalar/Decode.scala:163` | hypothesis | 2nd |
| F5 | CSR set, `misa` and trap behaviour haven't been compared against the M3 RTL. B01's check (same instruction count between the ROI markers on Spike and the RTL) is the first test of this. | - | hypothesis | 2nd |

Spike at the pin runs everything the M3 toolchain emits (RV32IMF + Zbb + Zve32f,
including vector FP), so FP kernels such as `rvv_float_matmul` can be modelled.

## A: Intentional model abstractions

These are deliberate simplifications in the coarse model. Each needs an error
budget once measurements exist.

| ID | Abstraction | Expected worst case |
| --- | --- | --- |
| A1 | LSU modelled as whole 16B-line transactions, with a fixed per-register overhead for vector ops, instead of byte-level slot state | Indexed, strided and misaligned accesses |
| A2 | RVV micro-op expansion driven by a table (LMUL/SEW rules), not a model of the decode logic | Mask, permute and reduction ops |
| A3 | No VRF read-port conflict modelling at first | Dense 3-source MAC code |
| A4 | Wrong-path fetch modelled as a timed bubble only | Should be close to exact, since the core is in-order |

## Appendix: changes after M3 (for when the pin moves)

These were seen at `coralnpu@0efcec66` (main, 2026-09). They don't apply to M3,
but each one invalidates entries above when the pin is bumped:

- **Fetch:** 2 fetches in flight, via a fetch reorder buffer. This removes most of D7.
- **LSU:** rewritten as LSUv3, with `LsuSuperSlot`, up to 8×VLENB byte cells and
  row coalescing. This replaces the whole U3/U6 behaviour.
- **Retirement buffers:** scalar retirement buffer grows to 16, and the vector ROB
  to 16.
- **Verification vs production timing:** verification builds retire vector and
  tile instructions under different conditions from production. M1 becomes
  1st order again.
- **VME/Zvt matrix engine:** added (`hdl/verilog/rvv/design/Zvt/`), along with the
  `VmeCoreMini` configs.
- **BF16 extensions:** Zfbfmin, Zvfbfmin and Zvfbfwma added to the RTL and the
  toolchain. Spike supports them; add them to the driver's ISA string.
- **Zvt in Spike:** upstream Spike implements the Zvt extensions (`EXT_ZVT*`),
  but after the `fd72ee2d` pin. Modelling the VME needs a Spike bump.
- **DTCM:** becomes 8 banks, split by address range (`BankedDtcm.scala`).
- **Docs:** `overview.md` is rewritten for RVV and VME, and adds new claims (Zba,
  4 × 32b DTCM banks, 4 branches/cycle) that also disagree with the RTL.
- **Test assets:** `isa_cycle_bench` and the Gemma kernel suite are added.

## Appendix: datasheet (out of scope, not tracked)

The datasheet's matrix-unit description (8×8×32 accumulator, "4× 8-bit multiplies
reduced into 32-bit accumulators") and its 256-bit datapath match the **legacy
Kelvin text in the M3 `overview.md`**. They don't match the M3 RTL. Treat the
datasheet as documentation of the older design.
