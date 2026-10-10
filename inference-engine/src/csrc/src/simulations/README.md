# W8A8 MLP functional simulator harness

This is a scalar correctness baseline, not an optimized RVV/VME kernel or a
latency claim. It does not depend on Joshua's simulator branch or change the
existing host MLP implementation. The fixture uses synthetic weights, not a
trained checkpoint: it verifies operator math, not model/story accuracy.

## Files

- `kernels/mlp_w8a8.h/.cc`: signed INT8 dot products, INT32 accumulation,
  dequantization, bias, tanh-GELU, dynamic activation requantization.
- `integration_tests/mlp_main.cc`: standalone harness checking both layers and
  intermediate values; uses static buffers, no allocation or C++ containers.
- `fixtures/mlp_fixture.h`: generated deterministic 3-token, 8→16→8 fixture;
  includes negative values, an all-zero input token, and all-zero weight rows.
- `BUILD.bazel`: Coral build rule to use after copying this folder into a
  separate Coral checkout. It is not part of reef's host CMake build.
- `inference-engine/src/inference_engine/experiments/w8a8_mlp/main.py`: PyTorch reference
  generator (under `inference-engine/src/inference_engine/experiments/`).

## Numerical contract

Inputs to each Linear and weights use a symmetric INT8 grid [-127,127], with
round-to-nearest, ties-to-even. Weight scales are per output channel; activation
scales are per token over the entire input vector. All-zero vectors use scale 1
and integer zeros (equivalent zero values to story_quality's tiny-scale rule).
Inputs must be finite and nonzero scales representable in FP32; subnormal-scale
underflow is not supported. Bias and nonlinear operations stay FP32.

Weights are row-major [output,input]. After an exact integer dot product:
`y = (float(accumulator) * input_scale) * weight_scale + bias`.
The first Linear is followed by GPT-Neo `gelu_new` (tanh approximation), then
per-token INT8 requantization, then the second Linear. Final output stays FP32.
There is no residual, normalization, or full transformer execution here.

This uses the quality experiment's quantization granularity and activation
formula, but integer accumulation/dequantization has a different floating-point
rounding order from FP32 matmul on fake-quantized operands. The generator checks
those linear outputs agree within atol=2e-6, rtol=2e-5. The harness checks integer
accumulators and quantized inputs exactly, and FP32 intermediates/outputs with
those tolerances. Near rounding boundaries, a different target tanhf may change
an activation code; investigate such failures rather than loosening integer
checks. Reduction widths must be below 131072 to prevent INT32 overflow even
with -128 operands. Callers own correctly sized, non-overlapping buffers.

## Local checks

From the reef repository root:

```sh
uv run --project inference-engine pytest \
  inference-engine/tests/python/test_w8a8_simulation.py
```

To regenerate the fixture using the workloads PyTorch environment:

```sh
uv run --project workloads python \
  inference-engine/src/inference_engine/experiments/w8a8_mlp/main.py \
  --output inference-engine/src/csrc/src/simulations/fixtures/mlp_fixture.h
```

The generator also accepts `--tokens`, `--width`, and `--hidden`, with seed 42.
The 28M architecture has width 512 and hidden width 2048. Generating that shape
creates about 2 MiB of INT8 weights plus reference data, far exceeding the default
32 KiB DTCM. Do not stage a full-size fixture with the small-fixture build rule:
external-memory placement, linker map, scratch space and simulator memory setup
must be addressed first. The current milestone is a small correctness harness.

## Remote build and run

On the remote machine, pull this reef branch. Set these paths to your actual
checkouts (absolute paths), then stage a copy of this folder into Coral:

```sh
REEF_CHECKOUT=/absolute/path/to/reef
CORAL_CHECKOUT=/absolute/path/to/coralnpu
mkdir -p "$CORAL_CHECKOUT/examples/reef_mlp"
cp -R "$REEF_CHECKOUT/inference-engine/src/csrc/src/simulations/." \
  "$CORAL_CHECKOUT/examples/reef_mlp/"
cd "$CORAL_CHECKOUT"
bazel build //examples/reef_mlp:mlp_w8a8_test.elf
bazel cquery //examples/reef_mlp:mlp_w8a8_test.elf --output=files
```

Use the ELF path printed by cquery, converted to an absolute path, for:

```sh
bazel run //sim:coralnpu_v2_sim -- --i /absolute/path/to/mlp_w8a8_test.elf
```

In the interactive simulator, enter `run`, then inspect:

```text
mem get mlp_status d32
mem get mlp_failures u32
```

Success is status 0 and failures 0; status -1 means the harness has not completed.
`mlp_output` is a named array for inspecting output values. The target harness
uses no printf and does not assume simulator process exit equals test success.
Record simulator version, ELF build options, status, failures and any instruction
statistics. Functional results are not cycle-accurate hardware timing.

The build rule was prepared against the upstream `coralnpu_v2_binary` macro:
https://github.com/google-coral/coralnpu/blob/main/rules/coralnpu_v2.bzl
Simulator commands follow:
https://developers.google.com/coral/guides/software/simulator
The remote checkout must provide that rule and C math library support (`tanhf`,
`floorf`, `fmaxf`, linked with `-lm`). Startup/linker setup comes from Coral's rule.
Cross-compilation and simulator execution have not been verified locally;
validate them on the configured remote computer before claiming Coral success.
