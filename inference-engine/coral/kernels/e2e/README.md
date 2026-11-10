# e2e

GPT-Neo TinyStories-28M on the NPU (M3 core, Zvt matrix unit, DMA). Weights
and the KV cache stream from XSPI flash into the 32 KB DTCM; code fits the
8 KB ITCM. `coral/tools/build_e2e.sh` builds every ELF into `coral/bin/`.

| Kernel | File | ELF | Cycles (8-token block, XSPI 2 B/cycle) |
|---|---|---|---:|
| Embedding | `embed.cc` | `embed.elf` | 11,212 |
| LayerNorm | `layernorm.cc` | `layernorm.elf` | 42,695 |
| Q, K, V projections | `linear.cc` | `qkv.elf` | 539,998 |
| Attention (positions 0–7) | `attention.cc` | `attention.elf` | 115,936 |
| Out-projection + residual | `linear.cc` | `attn_out.elf` | 189,545 |
| fc1 + GeLU | `linear.cc` | `fc1.elf` | 730,963 |
| fc2 + residual | `linear.cc` | `fc2.elf` | 666,057 |
| `lm_head` + argmax (1 token) | `lm_head.cc` | `lm_head.elf` | 14,443,774 |
| Whole model (16-token prompt + 1 decode step) | `main.cc` | `e2e.elf` | 87,366,903 |

Each kernel ELF (`standalone.cc`) runs its kernel once, layer 0, on one block
of 8 tokens whose residual is already in the DTCM. `common.hpp` holds the
DTCM layout; `npu.hpp` the model shape, flash map, DMA and Zvt wrappers.

**Memory.** ITCM `0x0`: code. DTCM `0x10000`: mailbox first (`prompt_length`,
`new_tokens`, `status`, `tokens[64]`); the host writes the prompt there and
reads the generated tokens back. XSPI flash `0x2000_0000`: the model, laid
out by `npu.hpp`. DMA registers: `0x4000_0000`.

**Run** (reef-perf from `yuvesh-kernel-integration`):

```
reef_perf --elf coral/bin/e2e.elf \
  -c configs/machines/m3.yaml -c configs/machines/m3_vme.yaml \
  -c configs/matrix/coral_vme.yaml -p top.memory.params.host_boot true \
  -p top.memory.params.ext_mem_latency 1 \
  -p top.memory.params.ext_mem_bytes_per_cycle 2 --ext-mem-mb 160
```

Cycle counts only: reef-perf moves no data, so outputs are meaningless.
