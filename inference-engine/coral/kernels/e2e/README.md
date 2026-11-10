# e2e

GPT-Neo (TinyStories) end to end on the NPU: M3 core, Zvt matrix unit, DMA.
Weights and the KV cache stream from XSPI flash into the 32 KB DTCM; code
fits the 8 KB ITCM. Prebuilt TinyStories-28M ELFs are in `coral/bin/`.

**Memory.** ITCM `0x0`: code. DTCM `0x10000`: mailbox first
(`prompt_length`, `new_tokens`, `status`, `tokens[64]`), the host writes the
prompt there and reads the generated tokens back. XSPI flash `0x2000_0000`:
the model, laid out by `npu.hpp`. DMA registers: `0x4000_0000`.

**Build.** `coral/tools/build_e2e.sh -DE2E_HIDDEN=512 -DE2E_LAYERS=8 -DE2E_BLOCK_TOKENS=8`

**Run** (reef-perf from `yuvesh-kernel-integration`):

```
reef_perf --elf coral/bin/e2e_ts28m.elf \
  -c configs/machines/m3.yaml -c configs/machines/m3_vme.yaml \
  -c configs/matrix/coral_vme.yaml -p top.memory.params.host_boot true \
  -p top.memory.params.ext_mem_latency 1 \
  -p top.memory.params.ext_mem_bytes_per_cycle 2 --ext-mem-mb 160
```

Cycle counts only: reef-perf moves no data, so outputs are meaningless.
