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

### The external memories' remaining numbers

The SoC plan is two memories on separate links: XSPI flash for the weights
(read only) and a PSRAM for the KV cache (read and written). The model has
both (`ExtMemModel`, deviations.md A7). `configs/memory/xspi_str.yaml`,
`xspi_dtr.yaml` and `psram_kv.yaml` hold pessimistic numbers to sweep. These
are still open:

- The flash's first-byte latency. The config uses 68, the top of the RTL
  team's 26 to 68 range; the spec's formula gives 24 with the command on 8
  lines.
- Whether the flash sustains its peak rate.
- The PSRAM part. Its latencies, rate and burst limit are typical values.
- The controller's own latency and any pipeline stages on the route to the
  DTCM.
- The controller FIFO depth and what a full FIFO does (`dma_queue_depth`).
- Whether the DMA has one channel or one per memory. The model has one, so
  DMA transfers to the two memories do not overlap.
- What `dma_setup_cycles` (20) stands for now that each link has its own
  latency. It may count the same delay twice.
- The firmware's memory map. It keeps the KV cache after the weights in one
  region (at `0x24A73800`, with 74.5 MB of model before it), not at
  `0x24000000` after 64 MB of flash. Runs set `ext_ram_base` and
  `ext_ram_mb` to the firmware's range until the two agree.
