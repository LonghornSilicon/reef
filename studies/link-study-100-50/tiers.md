## The hill climb in one table

52M model (TinyStories-28M), int8, decode of token 200, flash 100 MB/s, PSRAM 50 MB/s. Tokens/s, all stories. Each kept step adds to the one before. **Tier 1** uses only real hardware: the M3 core with two upstream Coral core changes, and the basic one-channel DMA that the firmware already assumes. **Tier 2** adds assumed hardware (not in the RTL): per-transfer DMA waits, a second DMA channel, and weight and KV streams into the matrix engine (about 12.5 KB of FIFOs). Experiment A (low-rank weights, a firmware and weight-format change, accuracy not checked) is shown on top of each tier.

| Tier | Step | Change | What it is | Batch 1 | Batch 8 |
| --- | --- | --- | --- | ---: | ---: |
| 1 | 0 | Baseline: M3 core, Joshua's firmware | Starting point | 1.43 | n/a |
| 1 | 3.1 | Real multi-story decode | Firmware: one weight pass serves up to 8 stories | 1.43 | 6.53 |
| 1 | 3.2 | 16-entry reorder buffer | Upstream Coral core | 1.52 | 7.53 |
| 1 | 3.3 | Newer load/store unit | Upstream Coral core | 1.56 | 7.86 |
| 1 | + A | Low-rank weights on tier 1 | Firmware and weight format | **2.71** | **10.71** |
| 2 | 3.9 | Per-transfer DMA wait | Assumed DMA feature: start the next weight transfer before waiting | 1.60 | 7.99 |
| 2 | 3.14 | Epilogue one row per token | Firmware | 1.66 | 8.08 |
| 2 | 3.16 | Attention during the qkv weights | Assumed: second DMA channel; firmware reorders qkv by head pair | 1.70 | 8.49 |
| 2 | 3.17 | Weight stream into the matrix engine | Assumed: FIFO in the engine, MACs read weights from it | 1.76 | 8.77 |
| 2 | 3.18 | 4 KB stream transfers | Assumed: 8.3 KB weight FIFO | 1.80 | 8.88 |
| 2 | 3.20 | KV stream into the matrix engine | Assumed: 4.2 KB KV FIFO | 1.82 | 9.19 |
| 2 | S1-S3 | Cross-size firmware steps S1, S2, S3 | Firmware: head-ready fix, layer norm per token, next weights early (with the code-space pass) | 1.87 | 9.24 |
| 2 | + A | Low-rank weights on tier 2 | Firmware and weight format | **3.33** | **12.35** |
| | | Ceiling, int8 weights (from the link times) | | 1.92 | 9.46 |
| | | Ceiling with A's weights (half the flash bytes) | | 3.66 | 13.14 |

The tier-2 firmware steps (3.14, S1 to S3) were measured on tier 2 only; some would also help tier 1, so tier 1 is a conservative figure. The ceiling is for int8 weights; A and int4 lower the bytes and so raise their own ceiling.

![Decode speed step by step, by tier](tiers_fig.png)

