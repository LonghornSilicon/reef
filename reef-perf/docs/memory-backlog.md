# Memory backlog

Ideas and blockers for the memory module that are not built yet.

## Experiments

### Cache the next prompt during a run

The host can write the next prompt into the DTCM while the current
inference runs. Then the core does not wait for the host after each run, and
the compute units stay busy. The cost is DTCM space for the waiting prompt.
This is a sweep: measure the time saved against the DTCM used.

## Blocked

### HyperRAM

Not modeled. The model needs the HyperRAM part's latency, bandwidth and
interface, and those depend on which HyperRAM the SoC uses. Until then, the
DMA's HyperRAM side is the placeholder `dma_setup_cycles` (deviations.md A5).
