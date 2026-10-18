# memory/interconnect miscorrelation with RTL

## goal
in order to have an accurate (microbenchmarks within 5%, kernels within 10%) performance model that we can run experiments/ideas on, the measurements it records must be accurate to the actual values from the RTL. we perform a comparison from the memory/interconnect portion of the performance model located in `reef-perf` and compare it to the CoralNPU's M3 RTL, since Reef starts from it.

in this context, correlation means running the same program on both the performance model and the RTL and comparing the difference (if any) in cycle counts. when we refer to error, it's measured as

$$
\frac{cycles_{model} - cycles_{RTL}}{cycles_{RTL}}
$$

an accurate performance model will allow us to get accurate answers to any questions we have that need changes to architecture.

this document details how far off the performance model is for memory/interconnect heavy code, the reason why, and which RTL features the model is missing.

## method

we evaluate two main types of workloads that are memory intensive:

### scalar microbenchmarks
- `load_chain`: a measurement of load-to-use latency of instructions by walking a 64-node circular list with each node on its own 16-byte line for 2000 loads
- `load_stream`: a measurement of load-store-unit throughput (how often a new instruction load can start)
- `addr_forward`: measurement of LSU throughput limitations, runs 8 `addi` → `lw` pairs, and the loads all read the same address for 8000 loads. 

### vector/kernel workloads (memory mixed with vector operations)
- `vec_memcpy`: measuring pure vector memory bandwidth by moving 128 bytes (8 vector registers/8 lines)
- `vec_add_m4`: measuring mixed memory and vector compute by performing c = a + b on 1,024 int32 values at LMUL=4 (64 bytes per op)
- `gemv_int8`: mixed memory, vector, and vector-scalar compute by performing an int8 matrix-vector product (64x64)

each workload's ELF runs unchanged on the M3 RTL (Verilator rvv_core_mini_verification_axi_sim at tag M3-2026-04-27) and on the model (reef_perf with configs/m3.yaml). on the RTL we take s11 − s10, the two mcycle reads that bracket each workload's loop. the model only reports whole-program cycles, so we run each workload at 1× and 2× its loop count and take the difference, which cancels the setup code and leaves just the loop.


## results

the following table compares cycles in each workload's measured loop on the RTL and the model. every memory-heavy workload is under-predicted by the model (model is too fast/optimistic)

| Workload | RTL cycles | Model cycles | Error |
|---|--:|--:|--:|
| `load_stream` | 8,204 | 2,072 | −75% |
| `vec_memcpy` | 18,467 | 5,490 | −70% |
| `load_chain` | 10,002 | 4,001 | −60% |
| `vec_add_m4` | 61,544 | 29,520 | −52% |
| `addr_forward` | 32,005 | 16,000 | −50% |
| `gemv_int8` | 13,932 | 8,744 | −37% |

total cycles depend on how many operations a workload runs, so the following table divides by the operation count. the scalar rows give the cost of a single load, which can be compared directly with the LSU's latency and initiation interval.

| Workload | Unit | Count | RTL cycles / unit | Model cycles / unit |
|---|---|--:|--:|--:|
| `load_chain` | dependent `lw` | 2,000 | 5.0 | 2.0 |
| `load_stream` | independent `lw` | 2,048 | 4.0 | 1.0 |
| `addr_forward` | `addi` → `lw` pair | 8,000 | 4.0 | 2.0 |
| `vec_memcpy` | vector load or store (8 registers) | 640 | 28.9 | 8.6 |
| `vec_add_m4` | loop iteration (2 loads, 1 store, 1 add) | 1,280 | 48.1 | 23.1 |
| `gemv_int8` | matrix row | 256 | 54.4 | 34.2 |

## observations/interpretations

the main thing that stands out is how the RTL cycles/unit and the Model cycles/unit for both independent and dependent `lw`s differ by 3 cycles in both memory load scenarios. this most likely indicates that the model is missing some sort of occupancy cost within the LSU slot.

when viewing the model's depiction of the LSU per cycle, we see that
- cycle 1: the dispatch sends the `lw` (risc-v load word instruction)
- cycle 2: it arrives at the LSU. the DTCM is booked for `lines × dtcm_cycles_per_line = 1 × 1` which is a single cycle, and the slot is occupied for that one cycle
- cycle 3: the slot is free, and the next load can start

within the RTL, though, there is more granularity
- cycle 1: dispatch sends the `lw` and it's written into `opQueue`
- cycle 2: the slot sees the `lw` and checks if the slot is idle. it is, so `lw` is loaded into the slot
- cycle 3: as the `lw` is in the slot, the DTCM read request fires
- cycle 4: the loaded data comes back, and the slot is still held during this time
- cycle 5: the data is written into the register file, and then the slot becomes idle
- cycle 6: as the slot is idle, the next `lw` which is in the queue is dequeued into the slot
- cycle 7: the DTCM request fires for the next `lw`.

in the model, the slot is held only for the DTCM request, so the next load can start immediately one cycle later. in the RTL, the slot also stays held while the data comes back, during writeback, and for one idle cycle before the next load is dequeued. those 3 cycles are missing from the model, which shows why the load costs 4 cycles on the RTL but 1 in the model.

similarly in `load_chain`, the 3 extra cycles are caused by the same loading into the slot and the writeback described in the RTL which isn't in the model.

## rtl features missing from the model

status: **measured** = backed by the results above; **code-read** = seen in the M3 RTL but not measured; **untested** = no workload exercises it.

| # | m3 RTL | model today | status |
|---|---|---|---|
| 1 | the LSU slot stays held while the data comes back, during writeback, and for one idle cycle before the next load is dequeued (`Lsu.scala:962,515,1056`) | the slot is held only for the DTCM transfer (`lsu.cpp`, `mem_timing.cpp`) | measured |
| 2 | cost is per instruction: loads to the same 16-byte line each take a full slot turn | cost is per distinct line | measured |
| 3 | the DTCM is a 1-cycle SRAM and never stalls the core (`ready` tied high, `CoreAxi.scala:210`) | the whole memory cost is charged to the DTCM port (`dtcm_latency: 2`) | code-read |
| 4 | vector loads/stores pass through the same slot one 16-byte register at a time; only `rvv2lsu` port 0 is serviced (`Lsu.scala:891-892`) | charged per line, with no per-register cost | code-read (`vec_memcpy`'s ~3.6 cycles/register is consistent) |
| 5 | scalar stores complete without a writeback step (`Lsu.scala:1000`), so they're likely cheaper than loads | stores cost the same as loads | untested |
| 6 | instruction fetch reads the ITCM through its own arbiter (`CoreAxi.scala:160-185`) | fetch never goes through the memory module | code-read |
| 7 | the DTCM arbiter is shared with the AXI slave and debug ports (`CoreAxi.scala:201,216,223`) | no contention between requesters | untested |
| 8 | external memory goes over AXI via `DBus2Axi`, timed by the testbench's bus model | `axi_latency` and `axi_max_outstanding` are assumptions (`configs/m3.yaml`) | untested |

## appendix

non-memory instructions of interest during workloads

| Workload | RTL cycles | RTL retired insts | Model cycles | Error |
|---|--:|--:|--:|--:|
| `addr_forward` | 32,005 | 18,000 | 16,000 | −50% |
| `alu_chain` | 18,004 | 18,000 | 16,000 | −11% |
| `alu_indep` | 12,003 | 18,000 | 7,000 | −42% |
| `branch_group` | 10,005 | 18,000 | 7,000 | −30% |
| `branch_tight` | 10,005 | 10,000 | 10,000 | 0% |
| `div_chain` | 18,003 | 1,800 | 25,600 | +42% |
| `fp_chain` | 8,505 | 5,000 | 12,000 | +41% |
| `fp_mix` | 12,003 | 18,000 | 12,000 | 0% |
| `gemv_int8` | 13,932 | 3,376 | 8,744 | −37% |
| `load_chain` | 10,002 | 6,000 | 4,001 | −60% |
| `load_stream` | 8,204 | 2,856 | 2,072 | −75% |
| `mul_chain` | 8,005 | 5,000 | 8,000 | 0% |
| `vec_add_m1` | timed out | — | 56,400 | — |
| `vec_add_m4` | 61,544 | 14,260 | 29,520 | −52% |
| `vec_memcpy` | 18,467 | 2,310 | 5,490 | −70% |
