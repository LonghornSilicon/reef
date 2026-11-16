# Memory backlog

Ideas and blockers for the memory module that are not built yet.

## Experiments

### Cache the next prompt during a run

The host can write the next prompt into the DTCM while the current
inference runs. Then the core does not wait for the host after each run, and
the compute units stay busy. The cost is DTCM space for the waiting prompt.
This is a sweep: measure the time saved against the DTCM used.

### Use a chunk while it arrives

`dma_stream_loads` makes a core load from the destination of an unfinished
DMA transfer wait for its own bytes, not for the whole transfer
(`Dma::getArrival`). With it, a firmware loop can start on a chunk before
the chunk is all in. Hardware for this would let the LSU see how far a
transfer is.

Measured on a token 200 decode step with the wait removed from `linear()`
of the end-to-end firmware: +2.8% at 50 MB/s of flash, +4.4% at 87.5 MB/s,
+0.4% at 100 MB/s and +0.2% above. The gain is from transfers that run back
to back; the link rate still sets the time below 100 MB/s, and the core
sets it above. The arrival time does not include a beat that a core access
pushed out.

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
`xspi_dtr.yaml` and `psram_kv.yaml` hold the RTL team's datasheet numbers
at a 25 MHz link clock: 18 cycles to the flash's first byte (it was 68, a
pessimistic guess), and 28 read / 16 write cycles with 93-byte bursts for a
QSPI PSRAM. These are still open:

- Whether the flash sustains its peak rate.
- Whether the PSRAM stays QSPI (0.25 B per cycle) or becomes octal (1 B per
  cycle).
- The controller's own latency and any pipeline stages on the route to the
  DTCM.
- The controller FIFO depth and what a full FIFO does (`dma_queue_depth`).
- Whether the DMA has one channel or one per memory. The model has one, so
  DMA transfers to the two memories do not overlap.
- What `dma_setup_cycles` (20) stands for now that each link has its own
  latency. It may count the same delay twice.
- The firmware's memory map. It keeps the KV cache after the weights in one
  region (at `0x24A73800`, with 74.5 MB of model before it), not at
  `0x24000000` after 64 MB of flash. `configs/memory/psram_kv_e2e.yaml`
  sets `ext_ram_base` and `ext_ram_mb` to the firmware's range until the
  two agree (e2e-short-runs.md).
