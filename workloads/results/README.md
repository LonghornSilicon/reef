# Results

Each directory is written by one script under `experiments/`, run from
`workloads/` as `uv run python -m experiments.<name> [family ...]`. CSVs are
tracked; PNGs are regenerated.

| Directory | Script | What it measures |
|---|---|---|
| `operator_macs/` | `operator_macs` | MACs per operator, batch 1, 224² images or 128 tokens |
| `prefill_decode/` | `prefill_decode` | MACs, bytes and arithmetic intensity per operator over batch × length × phase |
| `roofline/` | `roofline` | Latency per operator under a peak-MAC/s × bandwidth × SRAM grid (CSV), operator rooflines at three named points (PNG) |
| `quantization/` | `quantization` | WikiText-2 perplexity / ImageNet top-1, weight bytes and quantization ops per scheme |
| `kv_context/` | `kv_context` | Context length that fits a memory budget; decode tokens/s from weight + KV bytes per step |

## Conventions

- MACs: one per scalar multiply, add, compare or nonlinearity; a d-wide dot
  product costs d. Reshapes, copies and masked fills are free.
- Bytes: bf16 (2 B) per element unless a scheme says otherwise. Element counts
  come from meta-device shapes, so the tracer never allocates.
- Traffic is what each operator module reads and writes: input, parameters
  and buffers, output. Module hooks cannot see residual adds, the SwiGLU gate
  multiply, GPT-2's `wte + wpe` add or EfficientNet's squeeze-excite multiply,
  so those are not counted. Also not counted, because they are artifacts of
  this implementation rather than of the workload: the im2col columns,
  `pad2d` and `transpose().reshape()` copies, the fp32 promotion of attention
  scores and the `torch.cat` growth of the KV cache.
- `bytes` counts attention as implemented (scores written, read by a separate
  Softmax, probabilities read back); `fused_bytes` counts only Q, K, V and
  the output, as a flash-style kernel would.
- Decode at length L is the L-th token over L − 1 cached tokens, so prefill
  and decode rows at the same L cover the same positions.
- Quantization is fake: quantize then dequantize, compute in fp32. Weights
  are symmetric per output channel (int8) or per 128-wide group (int4);
  activations and KV are symmetric per token, or per pixel over channels
  for feature maps. `lm_head` stays fp32 because its weight is tied to the
  embedding. Storage counts one fp16 scale per channel, group or vector.
- `quantization` needs network for the checkpoints and WikiText-2, and an
  ImageFolder-style ImageNet validation set at `--imagenet PATH` for the
  CNN rows; `--limit N` scores N stride windows or N evenly spaced images,
  and the `samples` column records how many were used.
