# End-to-end short runs

A full inference (16-token prefill, then 80 generated tokens) takes 90 to
105 minutes to simulate. Four short runs that take one to two minutes each
predict its cycle count to 0.012%. Use the short runs for sweeps.

The firmware is the end-to-end executable of PR #57
(`inference-engine/coral/kernels`, branch `e2e-npu-firmware`, at 83f7152).
It is not on this branch. No simulator code changed for these runs.

## Build the executables

Build on the firmware branch, from `inference-engine/`. Its
`coral/README.md` has the toolchain setup and the flags.

| Run | Flags for `coral/tools/build.sh` |
| --- | --- |
| 16-token prefill | `-DE2E_PROMPT=16 -DE2E_NEW_TOKENS=1` |
| Decode of token 17 | `-DE2E_START_POS=16 -DE2E_NEW_TOKENS=1` |
| Decode of token 18 | `-DE2E_START_POS=17 -DE2E_NEW_TOKENS=1` |
| Decode of token 80 | `-DE2E_START_POS=79 -DE2E_NEW_TOKENS=1` |
| Prefill + 80 tokens | `-DE2E_PROMPT=16 -DE2E_NEW_TOKENS=80` |

Give each build its own `E2E_OUT` directory. Each one writes `e2e.elf`.

The last build does not compile as merged: the mailbox holds 64 tokens and
the run needs 96. For that build only, `kMailboxTokens` in
`coral/kernels/main.cc` was set to 128. This change is not on a branch.

## Run them

```bash
reef_perf --elf e2e.elf \
    -c configs/machines/m3.yaml -c configs/machines/m3_vme.yaml \
    -c configs/matrix/coral_vme.yaml \
    -c configs/memory/xspi_str.yaml -c configs/memory/psram_kv.yaml \
    -c configs/memory/psram_kv_e2e.yaml \
    -p top.memory.params.host_boot true \
    -p top.memory.params.host_boot_data true \
    --ext-mem-mb 160 --json out.json
```

- `xspi_str.yaml`: weight flash at 1 B per cycle, 18 cycles to the first
  byte.
- `psram_kv.yaml`: KV cache RAM at 0.25 B per cycle, 28 cycles to the first
  byte of a read, 16 of a write, 93-byte bursts.
- `psram_kv_e2e.yaml`: moves the RAM's address range to the firmware's KV
  cache at 0x24A73800. Without it the KV cache uses the flash link.
- The host boot and the prompt write are before the region of interest.

The link numbers are the RTL team's datasheet values (deviations.md A7).
The DMA has 20 setup cycles, 16 B per beat, one channel and no queue limit.

The firmware reads `mcycle` after each kernel. `mcycle_reads` in the JSON
file holds those cycles in order, so one run gives the time of each kernel.
The region of interest is the first read to the last. A decode step has 60
reads: `embed`, then seven kernels for each of the 8 layers (`layernorm`,
`qkv`, `attention`, `attn_out`, `layernorm`, `fc1`, `fc2`), then the final
`layernorm` and `lm_head`.

## Results

With the configs as they are now, the short runs give:

| Run | Cycles | Time |
| --- | ---: | ---: |
| 16-token prefill | 87,278,823 | 1.75 s |
| Decode of token 17 | 57,400,021 | 1.15 s |
| Decode of token 18 | 57,400,021 | 1.15 s |
| Decode of token 80 | 59,114,357 | 1.18 s |

The rest of this file, with the long run, used the earlier link numbers:
68 cycles to the flash's first byte, and 30 read cycles and 100-byte
bursts for the RAM. The short runs are about 2% slower there. The method
does not depend on these numbers.

At b8365e1. Time is at a 50 MHz core clock.

| Run | Cycles | Instructions | Time |
| --- | ---: | ---: | ---: |
| 16-token prefill | 89,080,757 | 20,779,327 | 1.78 s |
| Decode of token 17 | 58,615,735 | 13,021,590 | 1.17 s |
| Decode of token 18 | 58,615,735 | 13,021,590 | 1.17 s |
| Decode of token 80 | 60,331,607 | 13,227,286 | 1.21 s |
| Prefill + 80 tokens | 4,808,394,329 | 1,060,219,788 | 96.2 s |

Cycles for each kernel in one decode step, summed over the 8 layers:

| Kernel | Token 17 | Token 80 |
| --- | ---: | ---: |
| `embed` | 2,203 | 2,203 |
| `layernorm` (17 runs) | 634,001 | 634,001 |
| `qkv` | 7,143,792 | 7,143,792 |
| `attention` | 1,727,604 | 3,443,476 |
| `attn_out` | 2,476,064 | 2,476,064 |
| `fc1` | 9,658,856 | 9,658,856 |
| `fc2` | 9,316,616 | 9,316,616 |
| `lm_head` | 27,656,599 | 27,656,599 |

Only `attention` depends on the position. It reads the KV cache in blocks
of 16 positions, so the cost is a staircase: tokens 17 and 18 cost the
same, and each new block adds about 572,000 cycles to a step.

## Predict the long run

The long run is the prefill and then 79 decode steps (the prefill produces
the first new token). The steps fall in five KV cache blocks:

| Tokens | Steps | Cycles for each step in the long run |
| --- | ---: | ---: |
| 17 to 32 | 16 | 58,609,684 |
| 33 to 48 | 16 | 59,179,316 |
| 49 to 64 | 16 | 59,751,196 |
| 65 to 80 | 16 | 60,325,556 |
| 81 to 95 | 15 | 60,897,436 |

The estimate uses only the short runs. The step between blocks is
(token 80 - token 17) / 3 = 571,957 cycles, because token 80 is three
blocks after token 17.

| Estimate | Cycles | Error |
| --- | ---: | ---: |
| Prefill + steps for each block | 4,808,949,166 | +0.012% |
| Prefill + steps on a straight line through tokens 17 and 80 | 4,803,638,134 | -0.099% |
| Prefill + 79 x token 17 | 4,719,723,822 | -1.84% |

A step in the long run costs 6,051 cycles less than the same step as its
own executable, which has start-up code before the first kernel.

## Limits

- This compares the simulator with itself. It does not show how close
  either run is to the chip.
- Timing only: the DMA does not copy data, the matrix engine does not
  compute results, and no weights are loaded. The outputs are not checked.
- A longer decode needs one more short run in a later block, or the
  staircase from the table above.
