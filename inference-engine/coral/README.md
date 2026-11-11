# coral

GPT-Neo TinyStories-28M on the NPU (M3 core, Zvt matrix unit, DMA). Weights
and the KV cache stream from XSPI flash into the 32 KB DTCM; code fits the
8 KB ITCM. Integer only: the build targets `rv32im…zve32x` (no F), so a float
instruction fails to compile.

ELFs are not committed. Build them with `coral/tools/build.sh`, run from
`inference-engine/`. It uses Coral's GCC from `$CORAL_TOOLCHAIN` or `PATH`,
or else Docker with the volume that `coral/tools/setup_toolchain.sh` fills.
The ELFs land in `$E2E_OUT` (default `build/e2e/`), and compiler flags pass
straight through:

```
coral/tools/build.sh                                  # defaults: 8-token block at position 0, layer 0
E2E_OUT=build/e2e/dec59 coral/tools/build.sh -DE2E_TOKENS=1 -DE2E_START_POS=58
```

Give every variant its own `E2E_OUT` so builds can run in parallel.

| Kernel | File | ELF | Cycles (8-token block, XSPI 2 B/cycle) |
|---|---|---|---:|
| Embedding | `embed.cc` | `embed.elf` | 11,212 |
| LayerNorm | `layernorm.cc` | `layernorm.elf` | 36,560 |
| Q, K, V projections | `linear.cc` | `qkv.elf` | 539,998 |
| Attention (positions 0–7) | `attention.cc` | `attention.elf` | 115,936 |
| Out-projection + residual | `linear.cc` | `attn_out.elf` | 189,545 |
| fc1 + GeLU | `linear.cc` | `fc1.elf` | 730,963 |
| fc2 + residual | `linear.cc` | `fc2.elf` | 666,057 |
| `lm_head` + argmax (1 token) | `lm_head.cc` | `lm_head.elf` | 14,443,774 |
| Whole model (16-token prompt + 1 decode step) | `main.cc` | `e2e.elf` | 87,366,903 |

Each kernel ELF (`standalone.cc`) runs its kernel once on one block of
tokens whose residual is already in the DTCM. `common.hpp` holds the DTCM
layout. `npu.hpp` holds the model shape, the flash map, and the DMA and Zvt
wrappers.

## Knobs

| Define | ELFs | Default | Meaning |
|---|---|---|---|
| `E2E_TOKENS` | kernel ELFs | 8 | Tokens in the block, 1–8. Use 1 for a decode step. |
| `E2E_START_POS` | all | 0 | Position of the block's first token. Attention reads this many cached positions. In `e2e.elf`, a nonzero value skips prefill and decodes from this position. |
| `E2E_LAYER` | kernel ELFs | 0 | Which layer's weights to use. Even layers use global attention; odd layers use local attention over a 256-token window. |
| `E2E_PROMPT`, `E2E_NEW_TOKENS` | `e2e.elf` | 32, 8 | Prompt length and number of generated tokens. |

## Estimating a workload from kernel runs

Each run gives its cycles in the ROI between the two mcycle reads. One layer is
ln1 + qkv + attention + attn_out + ln2 + fc1 + fc2, where ln2 costs the same as
`layernorm`. Every token you generate also costs ln_f (the same cost as
`layernorm`) plus `lm_head`.

- **Only attention depends on position.** The linear kernels and `layernorm`
  depend only on `E2E_TOKENS`, and `embed` barely changes. So run the
  position-independent kernels once per token count, and run `attention` for
  each position you care about.
- **Layers repeat.** All 8 layers have the same shape. Attention is the same in
  every layer below position 256. From position 256 on, run `attention` with
  `E2E_LAYER=0` (global) and `E2E_LAYER=1` (local), then weight each by 4.
- **Prefill of P tokens** runs as ⌈P/8⌉ blocks at positions 0, 8, 16, … (the
  last block can be shorter). Add `embed` once per block, and ln_f + `lm_head`
  once.
- **Decoding token n** (1-based) means `E2E_TOKENS=1 E2E_START_POS=n-1`.

Recipes for the examples from review:

1. Prefill of 8 tokens, all layers. Use the defaults:
   embed + 8 × (2·layernorm + qkv + attention + attn_out + fc1 + fc2) + layernorm + lm_head.
2. Prefill of 16 tokens, one layer. The defaults give block 1. Rebuild
   `attention.elf` with `-DE2E_START_POS=8` for block 2:
   2 × (2·layernorm + qkv + attn_out + fc1 + fc2) + attention@0 + attention@8.
3. Decode of token 17, one layer. Build with `-DE2E_TOKENS=1 -DE2E_START_POS=16`:
   2·layernorm + qkv + attention + attn_out + fc1 + fc2.
4. Decode of token 59, all layers. Build with `-DE2E_TOKENS=1 -DE2E_START_POS=58`:
   embed + 8 × (one layer from 3) + layernorm + lm_head.

To check an estimate, build `e2e.elf` with
`-DE2E_START_POS=58 -DE2E_NEW_TOKENS=1`. It runs the whole decode step and
reads mcycle after every kernel.

## Limits

- **Sequence length (prefill + decode): at most 2048 positions**, the size of
  the position-embedding table. The kernel ELFs reject a larger
  `E2E_START_POS + E2E_TOKENS` at compile time.
- **Flash use:** about 74.5 MB for the model, plus 8 KB of KV cache per
  position. Space is reserved for `E2E_MAX_CONTEXT` = 2064 positions (16 MB),
  so the total is about 91 MB. That fits `--ext-mem-mb 160`. A real external
  memory smaller than this would lower the limit.
- **Prefill block: 8 tokens**, fixed by the DTCM. Longer prompts run as more
  blocks.

## Memory

- ITCM at `0x0`: code.
- DTCM at `0x10000`: the mailbox comes first (`prompt_length`, `new_tokens`,
  `status`, `tokens[64]`). The host writes the prompt there and reads the
  generated tokens back.
- XSPI flash at `0x2000_0000`: the model, laid out by `npu.hpp`.
- DMA registers at `0x4000_0000`.

## Run

Run reef-perf from the `yuvesh-kernel-integration` branch:

```
reef_perf --elf build/e2e/e2e.elf \
  -c configs/machines/m3.yaml -c configs/machines/m3_vme.yaml \
  -c configs/matrix/coral_vme.yaml -p top.memory.params.host_boot true \
  -p top.memory.params.ext_mem_latency 1 \
  -p top.memory.params.ext_mem_bytes_per_cycle 2 --ext-mem-mb 160
```

reef-perf reports cycle counts only. It moves no data, so the outputs are
meaningless.
