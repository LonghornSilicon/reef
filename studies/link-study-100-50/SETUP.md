# Study setup (frozen)

- Links: flash 100 MB/s (ext_mem_bytes_per_cycle 2, 18 cycles to first byte),
  PSRAM 50 MB/s (ext_ram_bytes_per_cycle 1, 28 read / 16 write latency, 93-byte bursts).
- Core: M3 (m3.yaml, m3_vme.yaml). Matrix: Coral mapping, 32 PEs (8 x 4). DTCM 32 KB.
- Simulator: /tmp/snap3/reef_perf, built from reef 06bcf53 (vikshar/dma-in-flight,
  dma_stream_loads off) plus two scratch matrix counters (num_macs, num_busy_cycles).
  Timing equals yuvesh-kernel-integration af11909 with these settings.
- Firmware: inference-engine e2e-npu-firmware at 83f7152 (Joshua, PR #57).
- Model: TinyStories-28M, hidden 512, 8 layers, vocab 50,257, about 52M parameters, int8.
- Workloads: decode of token 200 (-DE2E_START_POS=199 -DE2E_NEW_TOKENS=1), batch 1;
  time to first token on the default 32-token prompt (-DE2E_PROMPT=32 -DE2E_NEW_TOKENS=1).
  Batch 2, 4, 8 after real multi-story decode exists.
- Runner: study/run.sh; one run per metric; results in study/runs.
- Stopping rules: fixed candidate list; at most 3 attempts per candidate; a change is kept
  only if it gains more than 2% on some metric; stop tuning a kernel under 5% of the token
  at every batch or when flash is at least 95% busy; stop a path when no candidate gains
  more than 2%.
- The 2% bar is not noise (the simulator is deterministic); it matches the model's checked
  accuracy against RTL (LSU within 1%, matrix engine within 2%) and keeps the firmware readable.
- Retry rule: a rejected change goes back on the candidate list and is retried once the
  bottleneck it targets becomes the top one at some batch size.
- Queue rule: the user approves a queue of steps; a queue pauses if the bottleneck moves to
  something the queue does not target, a step misses its prediction widely, or a step needs
  new hardware or a judgment call.
