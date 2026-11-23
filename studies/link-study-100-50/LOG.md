# Study log: Reef at flash 100 MB/s, PSRAM 50 MB/s (TinyStories-28M, int8)

Setup: SETUP.md. Metrics: metrics.py. Runs: runs/.

## Phase 1: baseline (M3 core, 32-PE Coral array, Joshua's firmware 83f7152)

| Metric | Value | Share of ceiling |
| --- | ---: | ---: |
| Decode of token 200, batch 1 | 1.427 tokens/s (35.05 M cycles) | 74% |
| Time to first token, 32-token prompt | 2.010 s (100.5 M cycles) | 64% |
| Matrix utilization, decode / prefill | 1.17% / 6.51% | |

Batch 2, 4, 8: not runnable until multi-story decode exists (step 3.1).

Bottleneck: the core, not the links. Evidence:
- Same programs with unlimited links (runs/s0_*_unl): decode 34.47 M cycles against 26.09 M of flash
  time (core 1.32 x slower than the flash); time to first token 100.4 M against 64.8 M (1.55 x).
- Decode: LSU busy 79%; dispatch stalled on a full ROB in 61% of slots (vector 32.7%, matrix 28.4%),
  waiting on operands 20.8%; retiring 9.8%; front end 0.7%.
- Flash busy 74.5% (decode), 64.4% (prefill).

## Phase 2: ceilings (from the measured link busy cycles; today's transfer sizes)

Flash 26.09 M cycles per pass (51.3 MB of weights); PSRAM 2.42 M cycles per story at token 200.

| Decode, token 200 | Links overlapped | Links take turns |
| --- | --- | --- |
| Batch 1 | 1.92 tokens/s, 1.58% util | 1.75, 1.44% |
| Batch 2 | 3.83 total (1.92 each), 3.15% | 3.23 (1.62 each), 2.66% |
| Batch 4 | 7.67 total (1.92 each), 6.30% | 5.59 (1.40 each), 4.60% |
| Batch 8 | 15.33 total (1.92 each), 12.61% | 8.80 (1.10 each), 7.24% |

Time to first token, 32-token prompt: 1.295 s with links overlapped, 1.329 s taking turns
(flash 64.77 M cycles: layer weights read 4 times, lm_head once). Matrix util at the ceiling 10.1%.
"Overlapped" needs the flash and PSRAM to transfer at the same time, which the firmware does not do today.

## Phase 3: matrix path (one change per step)

| Step | Change | Kind | Source | Decode tok/s (b1 / b2 / b4 / b8) | TTFT | Share of ceiling | Bottleneck after, and evidence | Timing basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Baseline | | 83f7152 | 1.427 / - / - / - | 2.010 s | 74% / 64% | Core (see above) | RTL-fitted M3 core |
| 3.1 | Real multi-story decode: a pass carries up to 8 stories, each with its own KV cache, position, attention call and token pick (`E2E_STORIES`); KV cache reserved for 256 positions per story | Firmware | New (study), step1_multistory.patch | 1.428 / 2.58 / 4.33 / 6.53 total (1.43 / 1.29 / 1.08 / 0.82 each) | 2.009 s | b1 75%, b2 67%, b4 56%, b8 43%; TTFT 64% | Core (b1) and links taking turns (b8); see notes | Same as baseline |

Step 3.1 notes:
- Batch 1 equals the baseline within 0.1% (35.01 M against 35.05 M cycles); TTFT within 0.04%.
- MACs per pass are exactly batch x one story (421.0 M at batch 8). Flash bytes per pass are unchanged
  (26.09 M cycles at every batch); PSRAM cycles are exactly batch x one story (19.35 M at batch 8).
- The earlier batch-8 estimate (an 8-token prefill pass plus 8 x one story's attention) gave 6.53 tokens/s and
  5.37%; the real run gives the same, so the estimate method was sound for this firmware.
- Batch size stays capped at 8: the DTCM layout is sized for 8 token columns (bss 30,832 of 32,768 B).
- Matrix utilization: b1 1.17%, b2 2.12%, b4 3.56%, b8 5.37%.
- Bottleneck evidence at batch 8 (runs/s1_b8): flash busy 42.6% and PSRAM 31.6% of the run, never at the same
  time (74% together); the rest is core time between transfers. Attention is 35.0% of the pass, lm_head 32.4%.
  Dispatch slots: full ROB 60% (vector 31%, matrix 29%), waiting on operands 16%.
- Bottleneck evidence at batch 1: unchanged from the baseline; the core is 1.32 x slower than the flash
  (runs/s0_dec200_unl), full ROB 61% of slots.
| 3.2 | 16-entry ROB (M3 has 8) | Upstream Coral hardware | Upstream retirement buffer, 16 entries (Parameters.scala:141 at 8a5e1b31); `rob_entries 16` | 1.52 / 2.80 / 4.82 / 7.53 total (1.52 / 1.40 / 1.20 / 0.94 each) | 1.670 s | b1 79%, b2 73%, b4 63%, b8 49%; TTFT 78% | LSU queue full; see notes | ROB size from RTL source; effect on MAC timing RTL-checked in the matrix correlation (docs/matrix.md) |

Step 3.2 notes:
- Gains: decode +6.6% (b1) to +15.3% (b8); time to first token -16.9%. Matrix utilization b1 1.25%, b8 6.19%.
- Motivation (step 3.1 runs): full ROB in 60% of dispatch slots at every batch size.
- After (runs/s2_*): full-ROB stalls fall to 10-17% (matrix). The largest stall is now a full LSU queue,
  40% of slots at b1 and 26% at b8, with the LSU busy 78-84%; waiting on operands 17-21%.
- Batch 8: flash 49.1% and PSRAM 36.4% busy, still taking turns.
| 3.3 | Newer LSU: per-instruction instead of per-register overhead (`lsu_vector_load_reg_overhead 0`, `lsu_vector_store_reg_overhead 0`, `lsu_vector_switch_cycles 2`) | Upstream Coral hardware | Upstream LsuV3 (commits 15425970, 0b2924d8); costs fitted to the upstream RTL's vec_load_burst (3.0 cycles) and vec_load4_burst (6.0) in docs/lsu-tile-beats.md | 1.56 / 2.88 / 4.99 / 7.86 total (1.56 / 1.44 / 1.25 / 0.98 each) | 1.598 s | b1 81%, b2 75%, b4 65%, b8 51%; TTFT 81% | Load waiting on an older write to the same register; see notes | Fitted to RTL measurements for loads; vector stores 1 cycle fast (3 against 4 for one register); the newer LSU's faster scalar loads are not modeled |

Step 3.3 notes:
- Predicted +5 to 10% at batch 1; got +2.5% (b1) to +4.3% (b8), and -4.3% on time to first token.
- Motivation (runs/s2_*): full LSU queue 26-40% of dispatch slots, LSU busy 78-84%.
- After (runs/s3_*): LSU busy falls to 55-59%; full LSU queue 8-11%. The largest stall is now a memory
  instruction waiting on an older write to the same register (memory.waw_hazard), 33-34% of slots at every
  batch size; waiting on operands 18-22%.
- Queue paused here: the new top stall is not targeted by the queued steps.
| 3.5 | One pointer per weight row in the MAC loop; running chunk counters instead of a divide and remainder per chunk | Firmware | Earlier hill climb (kernel_hillclimb rounds 2-3) | 1.57 / 2.90 / 5.02 / 7.90 total | 1.588 s | +0.6% everywhere | **Rejected** (under 2%) | Same as 3.3 |

Step 3.5 notes:
- Motivation (runs/s3_*): waiting on operands 18-22% of slots. Predicted +1 to 3%.
- Result (runs/s4_*, patch s4.patch): operand waits fall to 4-7%, but the time moves to a full LSU queue
  (19-40%) and loads waiting on an older write to the same register (34-36%). Speed +0.6%. Reverted.
| 3.5b | Alternate two register sets (v16-v22, v24-v30) for the weight loads | Firmware | New (study) | 1.56 / 2.88 / 4.99 / 7.86 total | 1.598 s | 0.0% (identical cycles) | **Rejected**: the hypothesis was wrong; see notes | Same as 3.3 |

Step 3.5b notes:
- Motivation (runs/s3_*): memory.waw_hazard 33-34% of slots, read as vector loads waiting on the previous load
  into the same register. Predicted +5 to 15%. Result: identical cycle counts (runs/s5_*); the program ran (same
  instruction count, new ELF checked by md5). Reverted.
- Why: Dispatch tracks write-after-write only for scalar registers and for vector registers written by matrix
  instructions (model/core/Dispatch.cpp, isTracked); vector loads are tracked by the vector module. The stalled
  writes are scalar. The weight loop's only scalar load is dma_wait() (a read of the DMA WAIT register that
  returns when the transfer is done), so the stall is the core waiting for the flash.
- Confirmed by the same step-3.3 programs with unlimited links (runs/s3_b*_unl): the stall disappears.
- Where the decode stands after 3.3:
  - Batch 1: link-bound. The core alone needs 21.9 M cycles per token, against 26.1 M of flash time
    (core/flash 0.84). Flash busy 81%; the rest is the flash idle while attention waits on the PSRAM and
    between transfers.
  - Batch 8: core-bound and links taking turns. The core alone needs 34.1 M cycles per pass (core/flash 1.31);
    with unlimited links its top stalls are waiting on operands 28%, full ROB behind matrix 23%, full LSU
    queue 13%. Flash busy 51% and PSRAM 38%, never together. Attention is 40% of the pass.
| 3.6 | Two-pass decode attention (scores for all blocks first, then P V accumulated in the tile and read out once per head) | Firmware | Earlier hill climb, round 1 | 1.56 / 2.88 / 4.98 / 7.84 total | 1.598 s | 0.0% (b1) to -0.2% (b8) | **Rejected**; back on the retry list | Same as 3.3 |

Step 3.6 notes:
- Motivation (runs/s3_b8): attention is 40% of the batch-8 pass. Predicted about 0 at b1 and +3 to 8% at b8.
- Result (runs/s6_*): attention takes the same time (2.54 M cycles per story at b1 before, 2.55 M after). At
  50 MB/s attention waits on the PSRAM: its transfers alone are 2.42 M cycles per story, while with unlimited
  links the same attention takes 1.43 M (runs/s3_b1_unl). Retry if the PSRAM stops being attention's limit
  (for example once flash and PSRAM transfers overlap, step 3.8).
| 3.4 | Segment-load weight layout: one 4-field segment load per MAC instead of 4 loads (vtype LMUL 2); weights repacked offline, byte-interleaved | Firmware and flash layout | Earlier hill climb, round 3 (segment-load part only) | 1.57 / 2.90 / 5.02 / 7.91 total | 1.587 s | +0.3% (b1) to +0.7% (b8) | **Rejected**; retry list | Segment-load timing not RTL-measured (deviations.md U6); the model charges it like a 4-register load |

Step 3.4 notes:
- Predicted about 0 at b1 and +5 to 10% at b8; got +0.3 to +0.7%. Wide miss; queue paused.
- Result (runs/s7_*): LSU busy falls from 55-59% to 44-45%, but the time moves to waiting on the flash
  (memory.waw_hazard, the dma_wait() load) at 45-47% of slots.
- Correction to the batch-8 reading after 3.3: batch 8 is limited by the two links taking turns, not by the
  core. The flash (26.1 M cycles per pass) and the PSRAM (8 x 2.42 M = 19.3 M) never transfer at the same
  time, so together they need 45.4 M cycles, which is 89% of the 50.9 M-cycle pass after 3.3 and 90% after
  3.4. The core alone needs 34.1 M (runs/s3_b8_unl), less than that. Same at b1: 28.5 M of serialized link
  time against a 32.0 M-cycle token (89%).
- So with one DMA and the firmware as it is, every step is within about 11% of the "links take turns" limit.
  The remaining large gain needs the flash and PSRAM to transfer at the same time (step 3.8).
| 3.7 | Start the next linear's first weight chunk during the current linear's last chunk and write-out (q→k→v, attn_out→fc1 across ln2, fc1→fc2, fc2→next layer's q or lm_head across ln1/ln_f; not across attention, whose softmax state shares the weight buffers) | Firmware | New (study), s8.patch (includes step 3.1) | 1.56 / 2.89 / 5.00 / 7.87 total | 1.604 s | -0.2% (b1) to +0.2% (b2, b4); TTFT +0.4% slower | **Rejected** after 1 of 3 attempts; the idea's bound is under 1% | Same as 3.3 |

Step 3.7 notes:
- Motivation (runs/s3_b1): the flash idles between kernels and during layer norm. Measured idle in the linears
  beyond their link time (including setup and first-byte latency): 2.34 M cycles per token (7.3%), but most of
  it is per transfer (about 40 to 180 cycles each, the core reacting to dma_wait() and starting the next
  transfer), not at kernel boundaries. Layer norm: 0.57 M cycles per token (1.8%) with the flash idle.
- Prediction: about +0.3% at every batch size (about 40 boundaries per pass of about 1,000 cycles each).
- Result (runs/s8_*): -0.2% to +0.2%; flash busy cycles unchanged. Each linear's tail gains, but layer norm's
  parameter transfer now queues behind the early chunk, and the extra issue code costs as much as it saves.
- Why no further attempts: with one DMA whose WAIT register waits for every started transfer, the firmware can
  keep at most one transfer ahead, so the per-transfer gaps cannot be hidden. The remaining boundary gaps are
  about 0.3%. Filling layer norm's idle flash would need about 32 chunks (74 KB) per layer norm; reusing the
  roughly 16 KB of the shared DTCM region that is free during layer norm moves 6 chunks, about 0.34% per token.
  Together under 1%, below the 2% bar.
- What would change this: a per-transfer or per-channel completion status (so more than one transfer can be in
  flight), or the two-channel DMA of step 3.8.
| 3.9 | Start weight chunk i+1 before waiting for chunk i, with a new DMA register that waits for every transfer except the newest | **Assumed hardware** (model change) and firmware | New (study). Not in the RTL: the DMA register block is an assumption (npu.hpp), and this adds a register at offset 0x14. Simulator: snap4 (sim.patch); firmware: s9.patch | 1.60 / 2.95 / 5.10 / 7.99 total (1.60 / 1.48 / 1.27 / 1.00 each) | 1.558 s | b1 84%, b2 77%, b4 67%, b8 52%; TTFT 83% | **Kept** (+2.8% b1). Flash idle during each output group's epilogue; see notes | Same as 3.3; the new register's timing is the model's, with no RTL behind it |

Step 3.9 notes:
- Assumption, stated on purpose: the DMA gets a second wait register (0x14) that returns when every started
  transfer except the newest is done, and it accepts a GO while a transfer is in flight (dma_queue_depth 0, as
  in every run so far). The real DMA's completion status is an open question for the RTL.
- Model change (sim.patch, on top of origin/vikshar/dma-channels 50b5bca): Dma::getFinishExceptNewest() and
  three new read offsets (0x14 older, 0x18 flash channel, 0x1c RAM channel). Checked first: the new binary runs
  the step-3.3 programs to the same cycle at every batch size and TTFT (runs/v4_*), and the 3.8 binary (snap5)
  runs the 3.9 program to the same cycle (runs/v5_b1).
- Motivation (step 3.7 notes, runs/s3_b1): 2.34 M cycles per token (7.3%) of flash idle inside the linears,
  mostly per transfer. Predicted: up to +7.9% at b1 if all of it went. Got +2.8% (b1), +2.5%, +2.1%, +1.7% (b8),
  TTFT -2.5%. Wide miss.
- Why the miss (runs/s9_b1, d9_nosetup, d9_nolat): two causes remain, and the change could not touch either.
  - DMA setup (20 cycles) and flash first-byte latency (18) are paid on every transfer, after the previous one
    finishes. Set to 0 one at a time: -1.7% and -1.4% of the token. lm_head's whole excess over its link time
    is about 40 cycles per chunk, which is this.
  - With two weight buffers, chunk i+2 cannot start until the epilogue of an output group has finished with
    chunk i's buffer (it holds the bias). The flash then stops for the epilogue: about 1.9 K cycles per group
    for kFeatureMajor/kGelu/kValueRows and 3 to 4.7 K for kResidual, about 1.5 M cycles per token at b1 (about
    5%). Estimate from the per-kernel excess over link time (fc1 18%, attn_out 22%, qkv 15%, fc2 11%, lm_head
    3.9%) after taking out 37 cycles per chunk; not yet tested by a run.
- Flash idle that is left at b1 (31.18 M cycles, flash busy 83.7%): attention 2.54 M (8%, PSRAM-bound),
  layer norm 0.57 M (1.8%), epilogues about 1.5 M, per-transfer setup and latency about 0.9 M.
| 3.8 | Second DMA channel, one per link; attn_out's first weight chunk comes in during attention; attention waits only on the PSRAM channel; the last KV write-backs stay in flight through attn_out. DTCM: weight buffer 0 gets its own space, buffer 1 shares with softmax, best arrays move into the shared region (net -896 B) | **Assumed hardware** (`dma_channels 2`) and firmware | origin/vikshar/dma-channels (50b5bca), no Reef source for a second channel; per-link waits are new (0x18, 0x1c). Simulator snap5 (sim5.patch); firmware fw38/, s38.patch | 1.60 / 2.96 / 5.10 / 8.00 total | 1.558 s | +0.0% (b1) to +0.2% (b2, b4) | **Rejected** after 1 of 3 attempts; bound under 1% with 32 KB | Two-channel timing is the model's; no RTL |

Step 3.8 notes:
- Assumption, stated on purpose: a second DMA channel with its own completion status, one channel per link, as
  modeled on origin/vikshar/dma-channels. Not in M3 or in any Reef document.
- Motivation (Phase 2): at b8 the links taking turns cost 45.4 M against 26.1 M cycles if overlapped.
- Bound before the run: an overlap needs somewhere to put the data. Attention needs q from the qkv pass before
  it can use the KV it reads, so KV reads cannot start early without holding a layer's K and V (up to about 200 KB per
  story); weights read during attention need a buffer for them (only one 2.3 KB chunk fits). So the gain is
  at most one chunk per layer plus the last write-backs: under 1% at every batch size.
- Result (runs/s38_*): +0.0% to +0.2%. Attention is 7.6 K cycles per token faster at b1 (the write-backs now
  overlap attn_out). attn_out is not faster: linear() already starts chunk 0 before its act_sum loop, which
  hides it, so the early chunk saves nothing.
- Why no more attempts: the bound holds for any layout in 32 KB of DTCM. What would change it is more on-chip
  buffering (out of scope) or a different attention order, which the qkv dependency rules out.

Queue approved after 3.8 (in this order): 3.14 (r2), 3.10, 3.11, 3.12 (r5), 3.13 (lanes), 3.15 (DMA setup overlap).
r2, r5 and lanes come from the earlier hill climb (util/kernel_hillclimb.md) and were missing from the first list.
| 3.14 | Epilogue by tile row: one tile row per token (16 features, strided store into the feature-major layout) instead of one tile column per feature | Firmware | Earlier hill climb round 2, step r2 (kernel_hillclimb.patch); s10.patch | 1.66 / 3.05 / 5.22 / 8.08 total (1.66 / 1.53 / 1.31 / 1.01 each) | 1.525 s | b1 87%, b2 80%, b4 68%, b8 53%; TTFT 85% | **Kept** (+3.5% b1) | Same as 3.3; strided loads and stores are charged as the model's LSU does, not RTL-checked for this pattern |

Step 3.14 notes:
- Motivation (step 3.9 notes): the flash stops during each output group's epilogue, about 5% of a token at b1.
  The epilogue ran 16 iterations (one per feature) at every batch size; now it runs `tokens` iterations.
- Predicted +3 to 5% at b1, +1 to 3% at b8. Got +3.5% (b1), +3.4%, +2.4%, +1.1% (b8); TTFT -2.1%.
- Per kernel at b1 (runs/s9_b1 to s10_b1): qkv -6.1%, attn_out -11.7%, fc1 -11.1%, fc2 -2.7%; lm_head (argmax
  epilogue, unchanged) and attention unchanged. Excess over link time is now 5 to 8% in each layer kernel,
  mostly the 37 cycles of setup and first-byte latency per chunk.
| 3.10 | Write back only the new V row of the newest KV block (32 B instead of 512 B per head); K still goes back whole, because K is [dim][position] and the new position's bytes are not contiguous | Firmware | New (study); s310.patch (on 3.9) | 1.61 / 2.97 / 5.13 / 8.08 total | 1.558 s | +0.2% (b1) to +1.1% (b8) | **Rejected** (under 2%) | Same as 3.3 |

Step 3.10 notes:
- Motivation (runs/s9_b8): attention, which waits on the PSRAM, is 41% of the b8 pass.
- Bound before the run: 480 B saved per head, 30.7 KB per story per token, about 1.7% of the 1.84 MB of PSRAM
  traffic. Predicted under 1% at b8. Got +1.1% (attention 20.30 M to 19.74 M cycles per pass, -2.8%).
- Writing back only the new K bytes would need a K layout with positions contiguous, which the score MAC does
  not use; not tried.
| 3.11 | At an output group's last chunk, copy the 256 B bias to the stack after the MAC loop and start chunk i+2 before the epilogue | Firmware (uses the 3.9 register) | New (study); s311.patch (on 3.14) | 1.66 / 3.06 / 5.26 / 8.12 total | 1.514 s | +0.0% (b1) to +0.7% (b4); TTFT -0.7% | **Rejected** (under 2%) | Same as 3.3 |

Step 3.11 notes:
- Motivation (step 3.9 notes): flash idle during epilogues with both weight buffers full. First predicted +3 to
  5% at b1; after 3.14 shortened the epilogue, revised to +0.5 to 1.5% before the run. Got +0.0% (b1).
- Why: after 3.14 the b1 epilogue is one iteration, shorter than one chunk transfer, so the flash no longer
  waits for it. At b4 and b8 (4 and 8 iterations) a little is left: +0.7% and +0.5%.
| 3.12 | Layer norm one token at a time, 16 features per vector (strided loads and stores), gamma and beta as vectors | Firmware | Earlier hill climb round 2, step r5; s312.patch (on 3.14) | 1.69 / 3.10 / 5.27 / 8.11 total | 1.512 s | +1.8% (b1) to +0.3% (b8); TTFT -0.8% | **Rejected** (under 2%); retry list | Same as 3.3 |

Step 3.12 notes:
- Motivation (step 3.9 notes): layer norm is 0.57 M cycles per token at b1 (1.8%) with the flash idle.
  Predicted about +1.6% at b1, less at b8. Got +1.8% (b1), +1.4%, +1.0%, +0.3% (b8).
- Layer norm per token: 574 K to 52 K cycles at b1; 564 K to 401 K at b8 (8 tokens of 32 iterations each,
  against 512 iterations before).
- Close to the bar at b1. Retry if layer norm becomes a larger share (for example if the weight kernels get
  faster).
| 3.13 | Activation (A) rows load only `tokens` lanes (msettn around the 4 A loads), so no A row crosses a 16-byte DTCM line | Firmware | Earlier hill climb round 3, "lanes" (there inside the segment-load loop; here on the current loop); s313.patch (on 3.14) | 1.66 / 3.05 / 5.21 / 8.07 total | 1.526 s | -0.1% to 0.0% | **Rejected** | Same as 3.3; msettn cost not RTL-checked |

Step 3.13 notes:
- Motivation: in round 3 it gave -4% with unlimited links. Predicted about 0 here: the weight kernels wait on
  the flash, not the LSU.
- Result (runs/s313_*): LSU busy 61% to 59% (b1), 56% to 55% (b8); speed unchanged. Confirms the weight
  kernels are flash-bound after 3.14.
| 3.15 | DMA does a queued transfer's setup (20 cycles) while the transfer before it is still moving data | **Assumed hardware** (model change, `dma_setup_overlap true`) | New (study). The 20-cycle setup itself is an assumption with no Reef source (m3.yaml). Simulator snap6 (sim6.patch); same programs as 3.14 | 1.69 / 3.10 / 5.29 / 8.18 total | 1.502 s | +1.7% (b1) to +1.2% (b8); TTFT -1.5% | **Rejected** (under 2%) | The option's timing is the model's; no RTL |

Step 3.15 notes:
- Motivation (runs/d9_nosetup): setup at 0 gives -1.7% of the token at b1 on 3.9. Predicted +1 to 1.7% at b1.
  Got +1.7%, +1.5%, +1.4%, +1.2%.
- Check: snap6 with the option off runs the 3.14 programs to the same cycle (runs/v6_b1).
| (data) | 3.10 + 3.11 + 3.12 + 3.15 together; not a step | Firmware + assumed hardware | fwc/, sc.patch; snap6 with `dma_setup_overlap true` | 1.72 / 3.17 / 5.42 / 8.34 total | 1.479 s | +3.8% (b1, b2, b4), +3.2% (b8); TTFT -3.0% | Data only: each part is under 2% alone | As the parts |

Notes on the combined run and on stopping the matrix path:
- The parts target separate idle sources, so their gains add: 1.8 + 1.7 + 0.2 + 0.0 at b1 against 3.8 measured.
- Share of the "links take turns" ceiling: 98.5% (b1), 98.2%, 96.9%, 94.7% (b8). (3.10 cuts the PSRAM bytes a
  little, so the ceiling for this firmware is slightly higher than the one computed from the baseline.)
- Stop rule: after the approved queue no single step gains more than 2%. Not attempted, with their bounds:
  dropping lm_head's bias (0.4% of flash bytes); 4 KB weight chunks (half of the 18-cycle first-byte latency per
  chunk, about 0.9%, and needs 4.6 KB more DTCM); retries of 3.4, 3.5, 3.6, 3.7 (their bottlenecks, the LSU and
  attention's core time, are not the limit: 3.13 and the d9 runs show the weight kernels wait on the flash and
  attention on the PSRAM).
- What is left at b1 (runs/sc_b1, 29.02 M cycles): flash busy 89.9%. Attention 2.45 M (8.4%) with only the
  PSRAM working, which is the "take turns" structure itself; the rest is the 18-cycle first-byte latency per
  chunk and the epilogue tails.

Queue approved after 3.15: 3.16 (attention during the qkv weight stream), then 3.17 (weights streamed into the
matrix engine). Correction to the 3.8 notes: the "about 200 KB" bound holds for fetching weights early during
attention (what 3.8 did), not for overlap in general. Attention needs only its own head's q, which is ready
after one output group of the Q linear, and the new k and v only for the last KV block.
| 3.16 | Decode: Q, K and V by output group (Q0 K0 V0 Q1 ...), so each head pair has q, k, v after 48 of 384 chunks; attention runs as 2-block steps between weight chunks (paced by a credit counter, KV reads of the next step always in flight); a 176 B decode softmax state outside the weight-buffer union; attention output written over q (ln1's output is still being read); each wait is for one numbered transfer | **Assumed hardware** (`dma_channels 2`, wait-for-transfer register 0x20) and firmware | New (study). Simulator snap7 (sim7.patch); firmware fw316/, s316.patch (on 3.14) | 1.70 / 3.20 / 5.53 / 8.49 total (1.70 / 1.60 / 1.38 / 1.06 each) | 1.523 s | b1 89%, b2 84%, b4 72%, b8 55%; TTFT 85% | **Kept** (+2.4% b1, +5.0% b8). At b1 the flash idles during attention steps (one weight chunk in flight); see notes | Two-channel and per-transfer wait timing are the model's; no RTL |

Step 3.16 notes:
- Checks: snap7 runs the 3.14 programs and the 3.8 programs (2 channels) to the same cycle (runs/v7_b1,
  v7_38b8). MAC count unchanged. ITCM text at most 7,996 of 8,192 B (the prefill build); DTCM +264 B (stack
  1,404 B left).
- Predicted +5 to 8% at b1 and +6 to 9% at b8. Got +2.4%, +4.9%, +5.9%, +5.0%. Below the range at both ends.
- One channel, same program (runs/d316_1ch): 30.13 M cycles at b1, the same as 3.14. The gain is the overlap.
- b1 (runs/s316_b1): qkv plus attention 5.28 M cycles per token (was 5.96 M): a 4.97 M window with 7/8 of
  attention inside it, then 0.31 M for the last head pair. The window is longer than the core-limited estimate
  (4.0 M) because an attention step takes about 1.6 K core cycles and only one weight chunk (about 1.1 K cycles
  of flash) is in flight, so the flash stops during each step. More chunks in flight would need DTCM that is not
  free (2.3 KB per buffer) or the weight stream of 3.17.
- b8 (runs/s316_b8): the window (19.29 M) is PSRAM-bound and hides all of qkv's flash time (3.17 M). The rest
  is the last head pair's attention, 2.54 M, which nothing can overlap: attn_out needs every head.
- Mark positions are kept, so "qkv" in the kernel tables now includes 7/8 of attention.
| 3.17 | Weight rows go from the flash into a FIFO in the matrix engine (DMA destination in a stream window); MACs with B = v25 take the next 64 B from it, so the MAC loop loads only the 4 activation rows per step; each group's 256 B bias goes to the DTCM; two chunks in flight | **Assumed hardware** (model change, `mtx_stream_b true`) and firmware; weights repacked offline so each MAC's 64 B are contiguous | New (study). Simulator snap8 (sim8.patch: common/StreamFifo.h, Dma.cpp stream window, Matrix.cpp stream MAC); firmware fw317/, s317.patch (on 3.16), built with `-DE2E_STREAM_B` | 1.76 / 3.29 / 5.66 / 8.77 total (1.76 / 1.65 / 1.42 / 1.10 each) | 1.456 s | 92% / 91% / 92% / 93% of the "attention overlaps qkv only" ceiling | **Kept** (+3.6% b1, +3.4% b8, TTFT -4.4%) | Stream FIFO and stream MAC timing are the model's; no RTL. No FIFO backpressure modeled: the firmware keeps two chunks in flight, and the run measures the most the FIFO held |

Step 3.17 notes:
- Checks: snap8 with the stream off runs the 3.16 program to the same cycle (runs/v8_b1). MAC count unchanged.
  Every 64 B unit pushed was popped, with no underflow (a MAC never ran before its transfer was started).
- FIFO size the run needed: at most 65 units of 64 B (4.2 KB) held at once, at every batch size. The stream
  frees the DTCM weight buffers (4.6 KB), which this step does not reuse.
- First attempt failed: B = v31 is not an aligned int8 operand (v0, v1, v8, v9, v16, v17, v24, v25), so the
  model decoded those MACs as no-ops (MAC count changed, 0 pops). Fixed by using v25. Counted as attempt 1 of 3.
- Predicted +6 to 12% at b1 and +3 to 6% at b8. Got +3.6% (b1), +2.7%, +2.4%, +3.4% (b8); TTFT -4.4%.
- After (runs/s317_*): LSU busy 21% at b1 (was 65%). The top stall is now dispatch waiting on a full matrix
  queue (73% of slots at b1, 44% at b8): MACs wait for stream bytes, so the core waits on the flash through the
  matrix engine.
- What is left at b1 (28.41 M cycles), as time over each kernel's link time: qkv window 3.0%, layer norm 2.0%,
  lm_head 1.8%, the last head pair's attention 1.1%, fc2 1.0%, fc1 0.7%. Most of the per-kernel excess is the
  37 cycles of DMA setup and first-byte latency per transfer, plus one extra small transfer per group (bias).
- b8 (45.60 M cycles): the qkv window is PSRAM-bound (34.5% of the pass over qkv's flash time, attention inside
  it), and the last head pair's attention is 5.6%.
- Data point, not a step (asked: is every PE used?): no step changes the array mapping; all runs use the Coral
  mapping (coral_vme.yaml, as the RTL), where half the MACs of a busy array cycle are useful. The array is busy
  2.9% of the pass at b1 and 14.4% at b8 (runs/s317_*). With the generic mapping (`mtx_mapping generic`,
  runs/d317_pe_*): b1 -0.02% (mapping use 12.5%, array busy 4x longer), b8 +0.03% (mapping use 82%, array busy
  6.58 M to 4.03 M cycles). The array is not on the critical path at these link speeds.

Queue approved after 3.17: retries of 3.12 and 3.15 (their bottlenecks are now among the top ones), 3.18
(larger stream transfers), 3.19 (every-PE array mapping, asked for), and a retry of 3.7 (start the next
linear's weights early), which the stream makes possible. All on top of 3.17, each alone.
| 3.15 (retry 1) | DMA setup overlapped with the previous transfer, on 3.17 | **Assumed hardware** (`dma_setup_overlap true`) | As 3.15; runs/sr15_* | 1.79 / 3.35 / 5.75 / 8.90 total | 1.430 s | +1.9% (b1) to +1.4% (b8); TTFT -1.8% | **Rejected** again (under 2%) | As 3.15 |
| 3.19 | Every-PE array mapping (`mtx_mapping generic`) instead of the Coral rule | **Assumed hardware** (array control, not firmware) | Asked for; the earlier exploration's idealization; runs/d317_pe_* | 1.76 / 3.29 / 5.66 / 8.78 total | 1.456 s | -0.1% to +0.0% everywhere, TTFT included | **Rejected**: the array is not on the critical path | Mapping timing is the model's generic rule |

Step 3.19 notes:
- Not a firmware change: the Coral array's control decides which PEs a MAC uses (zvt_pe_array.sv:216-217,
  "partial-tile rule"); firmware can only give it fuller tiles (more tokens per MAC, which is batching).
- With the Coral rule half the MACs of a busy array cycle are useful at every batch size. Generic mapping:
  12.5% at b1 (one token row of 8), 82% at b8. Either way the array is busy 3 to 14% of the pass (runs/s317_*),
  and decode and prefill wait on the flash and the PSRAM, so the mapping changes nothing at these link speeds.
| 3.12 (retry 1) | Layer norm per token, on 3.17 | Firmware | As 3.12; fwr12/, sr12.patch | 1.79 / 3.34 / 5.72 / 8.80 total | 1.445 s | +1.8% (b1) to +0.3% (b8); TTFT -0.8% | **Rejected** again (under 2%) | As 3.12 |
| 3.7 (retry 1) | With the stream: the next linear's first two chunks start during this linear's last two (Qg to Kg to Vg to Qg+1, V7 to attn_out, fc1 to fc2; never across a layer norm, whose parameter transfer would queue behind them) | Firmware (needs the 3.17 stream) | As 3.7; fwr7/, sr7.patch | 1.77 / 3.31 / 5.68 / 8.80 total | 1.463 s | +0.6% (b1) to +0.3% (b8); TTFT +0.5% | **Rejected** (under 2%). ITCM full: 8,192 of 8,192 B in the prefill build | As 3.17 |
| 3.18 | Stream transfers of 4 KB (64 rows) instead of 2 KB, two in flight; the DTCM buffers do not change | **Assumed hardware** (stream FIFO size) and firmware | New (study); fw318/ built with `-DE2E_STREAM_ROWS=64`, s318a.patch | 1.80 / 3.35 / 5.75 / 8.88 total (1.80 / 1.68 / 1.44 / 1.11 each) | 1.425 s | 94% of the "attention overlaps qkv only" ceiling at every batch size | **Kept** (+2.4% b1, TTFT -2.2%). Needs a stream FIFO of 8.3 KB (measured) | As 3.17 |

Step 3.18 notes:
- Motivation (step 3.17 notes): DMA setup and first-byte latency (37 cycles) on every 2 KB transfer, about 3.5%
  of the flash time. Twice the transfer size halves it.
- Attempt 1, 4 KB (runs/s318a_*): +2.4% (b1), +1.8%, +1.6%, +1.3% (b8); TTFT -2.2%. FIFO held at most 129 units
  of 64 B (8.3 KB), against 4.2 KB with 2 KB transfers.
- Attempt 2, 8 KB (runs/s318b_*): +2.8% (b1), +2.7%, +1.7%, +1.0% (b8); TTFT -3.1% against 3.17. Against
  attempt 1 that is +0.4% (b1) and -1.0% TTFT, under 2%, so attempt 1 is kept. FIFO 16.4 KB.
- At b8, larger transfers mean fewer chunks between attention steps, so more of attention moves after qkv
  ("attention" mark 6% to 11% of the pass with 4 KB, 28% with 8 KB). The pacing works in chunks; it is not
  retuned here.
| (data 2) | 3.12 + 3.15 on 3.18 together; not a step | Firmware + assumed hardware | fwc2/, sc2.patch; snap8 with `dma_setup_overlap true` | 1.85 / 3.44 / 5.87 / 9.00 total | 1.400 s | +2.9% (b1) to +1.3% (b8); 95 to 97% of the "attention overlaps qkv only" ceiling; flash busy 96.0% at b1 | Data only: each part is under 2% alone | As the parts |

Stop rule for the matrix path: after this queue no single step gains more than 2%, and with the data-2 row the
flash is 96% busy at b1 (the stop rule's 95%). The kept steps end at 3.18: 1.80 / 3.35 / 5.75 / 8.88 tokens/s,
TTFT 1.425 s, 94% of the dependency-aware ceiling at every batch size.
| 3.20 | KV cache streamed into the matrix engine: each decode attention step's blocks (except the tail block, which takes the new token) go from the PSRAM into a second stream FIFO in one transfer, two steps in flight; the attention MACs take K and V with B = v24; the tail block comes to the DTCM as before | **Assumed hardware** (second stream FIFO) and firmware; KV cache layout assumed: each (head, block) stored as K then V, V by 16-dim halves | Asked for (raised in the 3.17 discussion). Simulator snap9 (sim9.patch); firmware fw320/, s320.patch (on 3.18), built with `-DE2E_STREAM_KV` | 1.82 / 3.43 / 5.91 / 9.19 total (1.82 / 1.72 / 1.48 / 1.15 each) | 1.426 s | 95% (b1) to 97% (b8) of the "attention overlaps qkv only" ceiling | **Kept** (+3.4% b8) | Stream timing is the model's; no RTL. Prefill attention is unchanged (it would need the same KV layout; its timing does not depend on it) |

Step 3.20 notes:
- Bound before the run (runs/s318a_*): at b1 the qkv window is 2.4% over its flash time (attention's core work
  inside it); at b8 the PSRAM is about 91% busy during attention, held back by the one-step-ahead KV double
  buffer. Predicted +1 to 2.4% (b1) and +1 to 4% (b8). Got +1.2%, +2.3%, +2.8%, +3.4%; TTFT unchanged.
- Checks: snap9 runs the 3.18 program to the same cycle (runs/v9_b1). MAC count unchanged. KV FIFO held at most
  65 units of 64 B (4.2 KB), weight FIFO 8.3 KB as in 3.18; no underflows.
- PSRAM busy cycles fall slightly (19.35 M to 19.18 M per pass at b8): one larger transfer per step instead of two.
- ITCM: 8,184 of 8,192 B in the prefill build; 5.5 to 5.7 KB in the decode builds.

## Phase 6: experiments on the kept matrix design (3.20)

Each experiment changes the workload, so each has its own ceilings, computed the same way from its own runs'
link busy cycles. Config notes: the KV cache address moves with the flash layout, so each build gets its own
`ext_ram_base` (kvbase.py applies npu.hpp's layout formulas; it gives 0x24A73800 for the int8 28M build, the
address in psram_kv_e2e.yaml).

### 6.1 int4 weights (28M model)

| Build | Decode tok/s (b1 / b2 / b4 / b8) | TTFT | Share of its dependency-aware ceiling | Notes |
| --- | --- | --- | --- | --- |
| int8, step 3.20 | 1.82 / 3.43 / 5.91 / 9.19 | 1.426 s | 95 / 95 / 96 / 97% | |
| int4 weights, `-DE2E_W4` | 3.23 / 5.67 / 8.89 / 12.41 (+77% / +65% / +50% / +35%) | 0.802 s (-44%) | 90 / 93 / 94 / 95% | Ceiling 3.58 / 6.12 / 9.46 / 13.01 |

- **Assumed hardware:** int4 weights in the flash, unpacked to int8 in the weight stream at no cost, so the
  matrix engine still does int8 MACs (snap10, sim10.patch: a third stream window where each 32 B in becomes one
  64 B unit of the weight FIFO). Biases stay int32; the embedding table, layer norm parameters and the KV cache
  stay int8. Accuracy is not checked (no functional model of the values).
- Checks: snap10 runs the 3.20 program to the same cycle (runs/v10_b1); MAC count unchanged; weight FIFO at most
  131 units (8.4 KB), KV FIFO 4.2 KB; ITCM 8,184 B (prefill build).
- Flash per pass 13.16 M cycles (was 26.1 M); PSRAM per story unchanged (2.40 M). Predicted b1 3.2 to 3.6 and
  b8 11 to 12.5 tokens/s; got 3.23 and 12.41.
- What limits it now at b1 (runs/w4_b1, 15.46 M cycles): flash 85% busy. One story's PSRAM time (2.40 M) is
  more than the qkv window's flash time (1.60 M), so part of attention cannot overlap; layer norm is 3.7% of
  the pass (step 3.12, rejected at int8 with +1.8%, would likely clear the 2% bar here); lm_head 0.27 M over its
  link time. At b8: PSRAM 59.5% busy, flash 40.8%; the PSRAM is the larger link now.

### 6.2 Small model (hidden 256, 6 layers, 4k vocabulary, 5.8M parameters), int8, and 6.3 small model with int4

Same firmware as 3.20 (fw_w4/ tree), built with `-DE2E_HIDDEN=256 -DE2E_LAYERS=6 -DE2E_VOCAB=4096` (and
`-DE2E_W4` for 6.3); `ext_ram_base` 0x20713200 (int8) and 0x20453200 (int4) from kvbase.py. Tables: phase6.md.
- Max batch: 15 stories per pass (`E2E_BLOCK_TOKENS=15`): 30.6 KB of data, 1.9 KB of stack. 16 fails the
  linker's 1 KB stack check. The MAC tile allows 16 rows, so DTCM is the limit.
- int8: 12.53 tokens/s at b1 (82% of its ceiling), 39.36 at b8 (94%), 45.13 at b15 (95%); TTFT 0.268 s. Flash per pass
  2.95 M cycles, PSRAM per story 0.90 M: from b4 up it is PSRAM-bound (PSRAM 55 to 81% busy), so 15 stories give
  1.15 x the throughput of 8.
- int4: 19.12 at b1 (+53%, 81% of its ceiling), 44.63 at b8 (+13%, 94%), 47.85 at b15 (+6%, 94%); TTFT 0.170 s. The
  PSRAM limits it more (86% busy at b15).
- FIFO sizes the same as 28M: weights 8.3 to 8.4 KB, KV 4.2 KB. Matrix utilization stays under 5%.
- Note: metrics.py now takes the layer count from the number of mcycle reads (46 per pass for 6 layers).

### 6.4 PSRAM burst length (asked for)

Config only (`ext_ram_max_burst_bytes`, today 93 B: after 93 B a new command and the latency again), on the 28M
int8 (3.20) and int4 builds, runs/pb*. PSRAM cycles per story: 2.40 M (93 B), 2.13 M (186 B), 1.99 M (372 B),
1.87 M (no limit). Decode gain at b1 / b8: int8 +0.2 / +5.2% (186 B), +0.3 / +8.1% (372 B), +0.3 / +10.8% (no
limit); int4 +1.5 / +7.2%, +2.3 / +11.3%, +2.7 / +15.1%. The gain follows how PSRAM-bound the run is: small at b1,
large at b8 and with int4. Which burst length a real part allows (its chip-select limit at the chosen clock and
temperature range) is not checked.
- Correction: shares of the ceiling for the small model first given in chat as 93% (b1) and 98% (b15) were wrong:
  the qkv flash time left out the number of column groups (4 x too small), which made the ceiling too low. The
  values above come from phase6.py with the corrected formula. The small model is further from its ceiling at b1
  because fixed costs (layer norm, kernel starts, epilogues) are a larger share of a 7 x shorter pass.

Report change (asked for): the "links overlapped" ceiling, max(flash, batch x PSRAM), is removed from REPORT.md
and the figure, because no hardware or firmware in scope can reach it: attention is the only PSRAM work, and only
the qkv weights can stream while it runs (attn_out, fc1, fc2 need its output; lm_head's weights come after the
last layer). The report's one ceiling is now the dependency-aware one, and every "share of ceiling" in the
results table is against it (before, the step table used the overlapped ceiling, so batch 8 read about 60%).
"Links take turns" stays as a reference, labeled as the limit of the original one-DMA design. The overlapped
numbers above (Phase 2, early step rows) are kept here as history.

Found after Phase 6 (asked: is the small model far from its ceiling?): a firmware bug from step 3.16 that affects
only the small model. attn_unlock(g) marks 2 more heads ready per 64-column qkv group, which is right for 28M (head
size 32) but not for the small model (head size 16: 4 heads per group). So in the small-model runs heads 8 to 15
only started after qkv finished: "attention" after qkv is 13.6% of the b1 pass (runs/sm_b1) and 34% at b4. The 28M
runs are not affected. The small-model results in Phase 6 are therefore below what the design can do.
Small-model b1 breakdown (runs/sm_b1, 3.99 M cycles): flash 74% busy; attention after qkv 13.6%; layer norm 6.1%
(9.3% at int4); qkv window over its flash time 4.9%; the weight kernels after attention within 0.2 to 1.1% of
their link time.
Also: "How the ceilings are calculated" added to REPORT.md.

### 6.5 Hill climb across both model sizes (asked for: fixes that work for 52M and the small model)

Same rules as Phase 3. Each step is one change, written to work for both sizes, measured on all four workloads
(52M int8, 52M int4, small int8, small int4; b1 to b8, TTFT, and the small model's max batch), and kept if it
gains more than 2% on any. Scripts: build4.sh, run4.sh, jobs4.sh, cmp4.py. Base: step 3.20 (fw_w4/).

| Step | Change | 52M int8 | 52M int4 | Small int8 | Small int4 | Result |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | Bug fix: attn_unlock marks the heads of each finished qkv output group ready, 64 / head size of them (2 at 52M, 4 in the small model); it marked 2 for both | 0.0% | 0.0% | +4.4% b1, +3.9% b2, +1.7% b4 | +3.8% b1, +1.5% b2 | **Kept** |
| S2 | Layer norm per token (step 3.12), plus ITCM room for it: the one-token branches of the old attention() (dead since 3.16, decode goes through step()) removed, and attn_begin sets its two positions field by field (a struct copy pulled in memcpy) | +1.7% b1, -0.5% TTFT | +3.1% b1, +2.7% b2 | +5.8% b1, +4.4% b2, +2.4% b4, -0.9% b15 | +8.6% b1, +6.0% b2, +3.0% b4, -1.0% b15 | **Kept** |
| S3 | Next linear's weights early (step 3.7 retry) | | | | | **Not run**: needs about 270 B of ITCM; the prefill builds have 8 B (52M) and 108 B (small) free |

Notes:
- S1 predicted about +6% at small b1, got +4.4%. 52M runs give the same cycle counts, as expected (2 heads per
  group there).
- S2 predicted about +5% at small b1, got +5.8% (int8) and +8.6% (int4). At max batch it loses about 1%: the
  per-token layer norm does tokens x (hidden / 16) iterations against hidden iterations before, so at 15 tokens
  the old form is slightly faster. A batch-dependent choice would need both forms in the ITCM.
- ITCM after S2: 8,184 of 8,192 B in the 52M prefill build, 8,084 in the small one. The first S2 build overflowed
  by 128 B (52M) and 28 B (small); merging lm_head and lm_head_rows made the 52M prefill build larger, not
  smaller (only the one-token form is linked there), so it was undone.
- After S2, share of each workload's ceiling: 52M int8 96 to 97%; 52M int4 93 to 96%; small int8 90 to 95%;
  small int4 91 to 94%. Small int8 b1 (runs/S2si8_b1): flash 82% busy; attention after qkv 5.6% of the pass;
  layer norm 0.6%.
- Run hiccups: Docker stopped during the S1 copies (restarted, the runs were kept), and xargs -L joined lines
  ending in a space, so the 52M int8 S1 jobs were rerun alone; S1i4_b8 was lost in the restart and rerun.
| S4 | With the weight stream, the DTCM weight buffers hold only each group's 256 B bias (the union with the softmax state shrinks by 2,176 B), so the small model fits 16 stories per pass (28.8 KB of data) | +0.2% b1, -0.4% TTFT | +0.4% b1, -0.8% TTFT | +0.3% b1; b16 45.12, +0.9% over b15 | +1.0% b1, -1.3% TTFT; b16 47.53, +0.4% over b15 | **Rejected** (under 2%) |

- S4 predicted b1 to b8 unchanged and b16 about 1 to 2% over b15; got +0.4 to 0.9%: the small model is
  PSRAM-bound there (PSRAM 81 to 85% busy). 52M still cannot reach batch 9 (about 3.2 KB more per story).
- Final state after 6.5: S2 on fws2/ (s320 design plus S1 and S2). Phase 6 tables and figure in REPORT.md now
  show S2; the burst sweep stays on step 3.20.

### 6.6 ITCM space pass (asked for)

The build already used -Os, -ffunction-sections and --gc-sections. Tried on the 52M prefill build (8,184 B):
- `-msave-restore` (shared prologue and epilogue code; needs the rv32 libgcc linked, E2E_LDEXTRA in the
  firmware build script): 7,744 B (-440 B).
- dma() as one function in linear.cc instead of inlined at every call site: 7,796 B (-388 B).
- Both: 7,208 B (-976 B). Link-time optimization: not available in this toolchain (no lto1).
- Merging lm_head and lm_head_rows (tried in S2): larger, undone.
Measured as step Z1 on all four workloads (fwsz/, runs/Z1*): -0.1% to +0.6% everywhere, TTFT within 1%. Not
a speed gain, so not "kept" by the 2% rule; adopted as the base for the experiments below because it costs
nothing measurable and frees about 1 KB of ITCM (52M prefill build 7,208 of 8,192 B; small 7,324 B). With it,
S3 (about 270 B) would now fit; not run yet.

### 6.7 Experiment A: low-rank weights computed as (x U) V (asked for; separate from the main results)

Each weight matrix W [in x out] is stored as U [in x r] and V [r x out] and the firmware runs two linears.
r is the multiple of 64 closest to half the bytes: 52M q/k/v/attn_out 128, fc1/fc2 192, lm_head 256 (flash
bytes 49.7% of int8); small 64 / 128 / 128 (58%). x U goes to a temporary in the DTCM (one slot each for q, k,
v so 3.16's group order stays; shares space with the prefill softmax state, so the DTCM does not grow). Base:
Z1. Firmware fwA/ (`-DE2E_LOWRANK`), runs/A*. KV cache kept at each model's usual address (`-DE2E_KV_BASE`).
- 52M: 3.33 / 5.74 / 8.93 / 12.35 tokens/s (+80% / +65% / +49% / +34% over int8), TTFT 0.765 s (was 1.420).
  Flash per pass 12.87 M cycles (int8 25.89 M); MACs 0.51 x. Same speed as int4 weights (within 1.3%): the
  second kernel call per linear costs almost nothing.
- Small: 18.99 / 27.79 / 36.28 / 42.76, b15 46.21 (+37% to +3%); TTFT 0.194 s (was 0.262). Flash 1.72 M (58% of
  int8, the rank rounding), so 2 to 12% below int4.
- Accuracy is not known: the simulator does not compute values. It needs the offline check on the real weights
  (stopped on request). Rank r of the whole matrix, not of each tile.

### 6.8 Experiment B: rebuild each 16x16 weight tile on the matrix engine (Aarush's idea; separate from A)

Each 16x16 int8 tile is stored as rank-4 factors U (16x4) and V (4x16), 128 B instead of 256 B (flash bytes
half of int8). Per chunk of 64 rows x 32 columns (8 tiles): U by DMA into the DTCM, V through the weight stream;
for each tile, zero mt8, one MAC (A = U, B = V from the stream), then 16 reads of mt8's rows (vtmv), narrowed to
int8 and stored into a 2 KB DTCM chunk in the normal layout; then the normal MACs from that chunk into mt0 and
mt4 (32 columns per pass, because mt8 is busy rebuilding). Base: Z1. Firmware fwB/ (`-DE2E_EXPAND`), runs/B*.
- 52M: 0.52 / 1.01 / 1.96 / 3.61 tokens/s (-72% / -71% / -67% / -61% against int8). Small: 4.30 / 8.08 / 14.39 /
  22.77, b15 30.32 (-69% to -32%); TTFT 0.839 s (was 0.262).
- Why (runs/Bi8_b1, 97.0 M cycles): flash busy 14%, matrix array 2% (with 4.9 x the MACs), LSU 43%, ALU 60%;
  dispatch waits on a full vector queue 45% of slots. The core spends the pass reading rebuilt tiles out of
  the accumulator: 16 vtmv per tile (each with its own mtype set-up and a 64 B move), then narrowing and a store,
  about 100 instructions per tile, about 200 K tiles per token at 52M.
- One implementation choice: rebuilt rows go through a DTCM chunk. Keeping them in vector registers would save
  the stores and loads but not the 16 accumulator reads per tile, which are most of the cost.
- What would make B worthwhile: a hardware path that feeds a rebuilt tile from the accumulator back in as the
  weight operand without the vector registers. Not in Coral's matrix engine; not modeled.
- Weight FIFO at most 14 units of 64 B; no underflows. Accuracy not checked (as for A).

| S3 (run after the ITCM pass, on Z1) | Next linear's first chunks start during this linear's last two (Qg to Kg to Vg to Qg+1, V7 to attn_out, fc1 to fc2; never across a layer norm) | +1.0% b1 | +1.5% b1 | +2.6% b1 | +0.4% b1, TTFT +1.3% slower | **Kept** (+2.6% small int8 b1) |

- S3 fits after Z1: 52M prefill build 7,496 B (Z1 7,208 B); fws3/, runs/S3*. Predicted a few % at small b1, got
  +2.6%; small int4 TTFT 1.3% slower.
- B's 52M TTFT (runs/Bi8_ttft32): 4.968 s against 1.420 s for int8.
- Final state after 6.5 to 6.8: step 3.20 + S1 + S2 + Z1 + S3 (fws3/). Phase 6 tables and figure show it.

Open item recorded (asked for): return to experiment A. REPORT.md "Open items" lists it first, with the accuracy
check to do next (offline, real weights, per matrix and per tile) and A with int4 factors after that.

| S5 | Decode attention with the KV stream: each step covers twice the blocks (one longer transfer) and three steps are in flight (was two); tail blocks alternate between two DTCM buffers | -0.4% b1, +0.5% b8 | +1.8% b1 | +2.5% b1, +0.7% b15 | +2.7% b1, +2.1% b2, +0.4% b15 | **Kept by the 2% rule; costs a 3 x larger KV FIFO** |

- Predicted a few % at large batch (PSRAM gaps); got +2.5 to 2.7% at small b1 and under 1% at large batch. The
  diagnosis was wrong: at small int4 b15 the PSRAM is about 98% busy while attention runs (runs/S3si4_b15:
  qkv window plus attention 13.8 M cycles against 13.48 M of PSRAM); the 7% gap there is layer norm (2.4%, per
  token is slower at 15 stories) and the weight kernels' overhead. The b1 gain is attention's reads stalling less
  inside the qkv window with three steps in flight.
- Hardware cost: the KV FIFO held at most 192 units of 64 B (12.3 KB) at 52M b8 and 162 (10.4 KB) at small b15,
  against 65 (4.2 KB) before S5. The 2% rule does not weigh this; flagged for the area budget.
- fws5/, runs/S5*. ITCM: 52M prefill build 7,660 B.

S5 dropped (asked: the larger KV FIFO is not wanted). The 2.5 to 2.7% at small-model b1 is not worth a 12.3 KB
KV FIFO against 4.2 KB. Final state back to 3.20 + S1 + S2 + Z1 + S3 (fws3/). Report tables and figure use S3.

### 6.9 Experiment A on the original one-DMA design (asked for; separate in the report)

fwA1/ = step 3.1 firmware (fw_s1_snapshot) with `-DE2E_LOWRANK`, run like step 3.3 (core args, one channel).
x U goes into the KV staging buffer (q, k, v, attn_out, lm_head), the layer norm output (fc2), or for fc1 the
staging buffer then a 1.5 KB copy into the layer norm output, because the step-3.1 DTCM has no spare room.
- First run invalid: the KV address pin did not apply in this older npu.hpp (the replace was unchecked), so the
  KV cache sat in flash address space and its traffic went over the flash link (PSRAM 0 at b1 to b4). Fixed
  and rerun; those results discarded.
- Result (runs/A1i8_*): 2.71 / 4.73 / 7.54 / 10.71 tokens/s against 1.56 / 2.88 / 4.99 / 7.86 (+74% / +64% /
  +51% / +36%); TTFT 0.923 s against 1.598 s. Flash 26.09 M to 12.95 M cycles per pass; PSRAM unchanged; MACs
  0.51 x. Predicted b1 2.6 to 2.8 and b8 about +35%.

### 6.10 Experiment A and A + int4 on the final design (asked for: rerun A exactly, and A with int4)

fwA3/ = fws3 (final design) + the low-rank change (fwA's diff against fwsz). S3's next-linear hint is off in
low-rank builds (it assumes the dense layout; with A the next call is a factor of another shape), so these runs
lack S3's prefetch (worth 1 to 3% at b1 without A). KV cache pinned at each model's usual address. runs/A3*.
- 52M int8: 1.87 / 3.50 / 5.99 / 9.24 to 3.35 / 5.72 / 8.90 / 12.33 (+79% / +63% / +48% / +33%); TTFT 1.414 to
  0.759 s. Flash 25.9 M to 12.9 M cycles per pass.
- 52M int4 (A + int4): 3.38 / 5.85 / 9.06 / 12.50 to 5.49 / 8.57 / 12.00 / 14.84 (+63% / +47% / +32% / +19%);
  TTFT 0.790 to 0.485 s. Flash 13.2 M to 6.6 M. Against plain int8: 2.9 x at b1, 1.6 x at b8 (estimated before
  the run: about 3 x and 1.7 x).
- Small int8: 14.23 to 18.90 (+33%) at b1, +8% at b8, +3% at b15; TTFT 0.263 to 0.195 s.
- Small int4: 21.78 to 26.29 (+21%) at b1, +2% at b8, 0% at b15; TTFT 0.167 to 0.148 s.
- Not 4 x: flash bytes drop to a quarter, but PSRAM time per story and per-pass fixed costs do not. Weight FIFO
  at most 134 units (8.6 KB). Accuracy not checked; A + int4 stacks two lossy changes.
