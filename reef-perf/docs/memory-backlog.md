# Memory backlog

Ideas and blockers for the memory module that are not built yet.

## Experiments

### Cache the next prompt during a run

The host can write the next prompt into the DTCM while the current
inference runs. Then the core does not wait for the host after each run, and
the compute units stay busy. The cost is DTCM space for the waiting prompt.
This is a sweep: measure the time saved against the DTCM used.

### LSU occupancy

The LSU starts one memory instruction every 4 cycles. A load-heavy kernel is
bound by that rate. Measure what 2 or 3 cycles would gain. The LSU is in the
core module (Yuvesh), so propose it there.

## To finish

### Vector access to external memory

ext_load_chain, ext_load_stream and ext_store_stream use scalar loads and
stores. A kernel that reads weights from external memory would use vector
loads, which take the vector LSU path with its own per-register costs. That
path is not measured against the RTL yet. Add an ext_vec_load
microbenchmark, run it on the RTL with scripts/run_rtl.sh and compare.

## Blocked

### HyperRAM

Not modeled. The model needs the HyperRAM part's latency, bandwidth and
interface, and those depend on which HyperRAM the SoC uses. Until then, the
DMA's HyperRAM side is the placeholder `dma_setup_cycles` (deviations.md A5).
