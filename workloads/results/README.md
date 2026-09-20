# Results

Each directory is written by one script under `experiments/`, run from
`workloads/` as `uv run python -m experiments.<name> [family ...]`. CSVs are
tracked; PNGs are regenerated.

| Directory | Script | What it measures |
|---|---|---|
| `operator_macs/` | `operator_macs` | MACs per operator, batch 1, 224² images or 128 tokens |

## Conventions

- MACs: one per scalar multiply, add, compare or nonlinearity; a d-wide dot
  product costs d. Reshapes, copies and masked fills are free.
- Element counts come from meta-device shapes, so the tracer never allocates.
- Module hooks cannot see residual adds, the SwiGLU gate multiply, GPT-2's
  `wte + wpe` add or EfficientNet's squeeze-excite multiply, so those are not
  counted. Also not counted, because they are artifacts of this
  implementation rather than of the workload: the im2col columns, `pad2d` and
  `transpose().reshape()` copies, the fp32 promotion of attention scores and
  the `torch.cat` growth of the KV cache.
