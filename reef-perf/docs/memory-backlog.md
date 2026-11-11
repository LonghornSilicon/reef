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

### The external memory link's remaining numbers

The SoC plan replaced HyperRAM with an XSPI flash link. The model has the
link (`ExtMemModel`, deviations.md A7), and `configs/memory/xspi_dtr.yaml`
and `xspi_str.yaml` set its rate and first-byte latency from the spec. These
are still unknown, so they are not modeled or are placeholders:

- N, the number of lines that carry the command. It sets the first-byte
  latency: 24 core cycles for N = 8, 108 for N = 1.
- The controller's own latency and any pipeline stages on the route to the
  DTCM.
- The controller FIFO depth and what a full FIFO does (`dma_queue_depth`).
- The minimum chip-select high time between transfers.
- How the KV cache is written. The spec's link is read only so far.
- What `dma_setup_cycles` (20) stands for now that the link has its own
  latency. It may count the same delay twice.
