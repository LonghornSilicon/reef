# e2e: GPT-Neo end to end on the Reef NPU

`e2e.cc` runs the whole TinyStories model on the NPU: prefill, then greedy
decode. The core is the CoralNPU M3, with the Zvt matrix unit and a DMA.
Every weight and the KV cache stream from XSPI flash into the 32 KB DTCM by
DMA, double buffered. The code fits the 8 KB ITCM (5.7 KB for 28M).

| File | What |
|---|---|
| `npu.hpp` | Model shape, flash map, DMA driver, Zvt wrappers |
| `e2e.cc` | Kernels (linear, attention, LayerNorm, embedding, `lm_head`) and `main` |
| `../../runtime/link_e2e.ld` | M3 as built: 8 KB ITCM, 32 KB DTCM |
| `../../tools/build_e2e.sh` | Build script |

Prebuilt TinyStories-28M ELFs are in `coral/bin/e2e/`.

## Memory setup

The program runs on the M3 as built:

- **ITCM** (`0x0000_0000`, 8 KB) holds the code. The host loads the ELF's
  `.text` over the SPI host link, then releases the core from reset.
- **DTCM** (`0x0001_0000`, 32 KB) starts with the mailbox: `prompt_length`,
  `new_tokens`, `status`, then `tokens[64]`. The host writes the prompt
  token ids there before releasing the core. The firmware writes the
  generated tokens after the prompt, sets `status` to 1 and halts (`mpause`).
  The rest of the DTCM is working buffers and the stack; the host does not
  touch it.
- **XSPI flash** (`0x2000_0000`) holds the model. The host writes it once.
  For TinyStories-28M:

| Region | Address | Size | Format |
|---|---|---:|---|
| Token embeddings | `0x2000_0000` | 24.5 MB | int8 `[50257][512]` |
| Position embeddings | `0x2188_A200` | 1 MB | int8 `[2048][512]` |
| Layers × 8 | `0x2198_A200` | 3,166,208 B each | see below |
| Final LayerNorm | `0x231B_2200` | 1 KB | gamma, beta: int8 `[512]` each |
| `lm_head` | `0x231B_2600` | 24.8 MB | linear format, 50,304 columns (pad rows: −inf bias) |
| K cache | `0x24A7_3800` | 8.1 MB | written by the NPU |
| V cache | `0x2528_3800` | 8.1 MB | written by the NPU |

  A layer is LayerNorm parameters (`ln1` gamma, beta, `ln2` gamma, beta:
  int8 `[512]` each) at offset 0, then Q, K, V, out-projection, fc1 and
  fc2 at offsets 2,048, 266,240, 530,432, 794,624, 1,058,816 and
  2,115,584. Each linear `[K x N]` is stored as N/64 column groups: K rows
  of 64 uint8 weights (zero point 128), then 64 int32 biases.
  `npu.hpp` computes all of these from the model shape.

- **DMA registers** (`0x4000_0000`): SRC, DST, LEN, GO and WAIT, as
  reef-perf's `memory/Dma.h` models them. The register layout is an
  assumption; there is no DMA RTL yet.

## Building

```
coral/tools/build_e2e.sh -DE2E_HIDDEN=512 -DE2E_LAYERS=8 -DE2E_BLOCK_TOKENS=8 \
    -DE2E_PROMPT=16 -DE2E_NEW_TOKENS=2
```

`E2E_HIDDEN`, `E2E_LAYERS` and `E2E_BLOCK_TOKENS` set the model (28M:
512, 8, 8). `E2E_PROMPT` and `E2E_NEW_TOKENS` set the mailbox's default
request. `E2E_START_POS=N` skips prefill and decodes one token at position
N, as if the KV cache already held N positions; the decode-at-context ELFs
use it.

## Running on reef-perf

Needs the reef-perf from `yuvesh-kernel-integration` (matrix unit, DMA,
external memory link, host boot). `main`'s reef-perf has none of these.

```
reef_perf --elf coral/bin/e2e/e2e_ts28m.elf \
  -c configs/machines/m3.yaml -c configs/machines/m3_vme.yaml \
  -c configs/matrix/coral_vme.yaml \
  -p top.memory.params.host_boot true \
  -p top.memory.params.ext_mem_latency 1 \
  -p top.memory.params.ext_mem_bytes_per_cycle 2 \
  --ext-mem-mb 160 --json out.json
```

That is XSPI at 2 B/cycle (16 bit per clock at the 50 MHz core clock) with
1 cycle (6 ns) latency. The firmware reads `mcycle` after every kernel; the
JSON's `mcycle_reads` lists them (7 per layer, 1 per embedding, 2 per
`lm_head` step).

**Cycle counts only.** reef-perf's DMA moves no data and its matrix unit
computes no tile values, so the flash contents do not matter and the
generated tokens are meaningless. No loop bound depends on data, so the
timing is that of real weights.
