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

## Verified functional results — 2026-10-11

**PASS on the MPACT functional simulator.** The existing Coral float-add example
passed first, then the original synthetic 3-token 8→16→8 W8A8 MLP completed with
`mlp_status=0`, `mlp_failures=0`, and `_ret=0`. Its loaded status was -1 before
execution. A fresh instruction-trace run repeated all checks and also passed.
The kernel, fixture, exact integer checks and FP32 tolerances remain unchanged
from harness commit `a199d14`; the executed ELF has the same SHA256 as the
original ELF inspected on October 10. No optimized kernel, trained-model test,
full-size run, precision sweep or hardware performance result is claimed by
this validation.

| Check | Result |
| --- | --- |
| Simulator/validator build | PASS after the two environment fixes below |
| Existing float-add | PASS: 8 exact FP32 sums; `_ret=0`; normal `mpause` halt |
| W8A8 MLP | PASS: `mlp_status=0`, `mlp_failures=0`, `_ret=0`; normal `mpause` halt |
| Exact INT32 accumulators and quantized codes | PASS via original target harness |
| FP32 intermediates/output and scales | PASS at `atol=2e-6`, `rtol=2e-5` |
| Ties-to-even, all-zero vector, invalid-width checks | PASS via original target harness |
| Fresh traced MLP execution | PASS; same outputs and status words |
| Kernel/fixture/harness equal `a199d14` and staged Coral copy | PASS |

## Checkout, simulator API and build settings

- Reef branch: `sam/w8a8-mlp-simulator`; original harness commit:
  `a199d14e8406c7bc67db36991173e49d38fae751`.
- Coral revision: `dab25ad632e35d6399b6d1aa1f05c30e700f06eb`.
- Coral MPACT dependency: `7670ee4c3df4cbc0ea06220c39a19b5d14338ed6`,
  with Coral's existing patches from `rules/repo_defs.bzl`.
- Bazel **9.1.0**, as pinned by `.bazelversion`; the README's 8.6.0 is stale.
  Its quick start uses Verilator, not the functional simulator used here.
- There is no root `sim` package. `//sw/coralnpu_sim:coralnpu_v2_sim` produces
  a Python extension through `mpact_binary`, not a CLI accepting `--i`.
  The validator uses `CoralNPUSimulatorPy.LoadProgram`, `Step`, `ReadRegister`,
  `ReadMemory`, and `WriteMemory`. Existing examples of this API are in
  `sw/coralnpu_sim/coralnpu_v2_sim_test.py` and
  `coralnpu_test_utils/sim_backends/mpact_npusim_test_fixture.py`.
- Guest build: `-std=c++17 -O2 -ffp-contract=off`, `-lm`, 2048-byte stack,
  default CRT, no semihosting, 8 KiB ITCM, 32 KiB DTCM. The toolchain declares
  `ilp32` and `rv32imf_zve32f_zicsr_zifencei_zbb_zfbfmin_zvfbfmin_zvfbfwma`.
- Existing machine configuration: host Clang 15 wrappers selecting GCC 11,
  8 jobs, CPU resource limit 8, memory resource limit 32768 MiB and
  `CORALNPU_MAKE_JOBS=8`. `.bazelrc.user` and the guest BUILD rule are captured
  in `validation/logs/inspection-20261010.log`.

The validator explicitly maps ITCM `[0,0x2000)` and DTCM `[0x10000,0x18000)`;
Coral's Python helper defaults to high-memory DTCM at `0x100000`, which does
not match this ELF. It sets MISA to `0x40801120` as in the helper, disables
semihosting, and enables exit-on-EBREAK. It rejects ELF load segments outside
this small map. This is not a runner for full-size model weights.

## Build-environment diagnosis and corrections

The initial restricted session could not fetch `pybind11_bazel` 2.13.6. Running
the build in a normal SSH terminal resolved the missing dependency and completed
analysis. The next failure was `PyWriteBuildData` reporting
`execvp(external/rules_python+/python/private/build_data_writer.sh): No such file or directory`.

The script exists and is executable (`0775`). Its shebang is plain LF
`#!/bin/sh`, its syntax passes `/bin/sh -n`, and `/bin/sh` resolves to
`/bin/dash`. `/bin/sh` executes inside the failing sandbox, confirming the
interpreter and dynamic loader are accessible. The execroot symlink chain
ultimately points into `/tmp/reef-w8a8-repo-contents`.

`--verbose_failures --sandbox_debug` confirmed that Bazel replaces `/tmp` with
hermetic temporary storage, mounting back `/tmp/reef-w8a8-bazel` but omitting
the sibling repo-contents cache. The script's symlink target is therefore
hidden inside the sandbox. The correction is:

```text
--sandbox_add_mount_pair=/tmp/reef-w8a8-repo-contents
```

A direct sandbox probe then executed the real script with exit 0 and wrote its
build-data output. The full build also passed PyWriteBuildData with this flag.
Linux sandboxing and hermetic temporary storage remain enabled.

The retry exposed a second environment issue: the existing Clang wrappers
select GCC 11 through `setup/gcc11`, but Coral's explicit host Clang toolchain
did not recognize that header path as a builtin include root. The local fix
adds these two known header roots to `toolchain/host_clang/BUILD`:

```text
/home/samlin/coralnpu-dissection/coralnpu/setup/gcc11/include
/nfs/samlin/coral-build/host-deps/usr/include
```

The reviewable patch is `validation/setup/mew3-host-clang-includes.patch`. It is
machine-specific setup for mew3's pre-existing wrappers, **not a portable
kernel requirement**. See `validation/setup/README.md` for its scope. The current Coral checkout already has it. For an unpatched checkout
with the same setup, apply it once:

```bash
cd /home/samlin/coralnpu-dissection/coralnpu
git apply --check ../reef-mlp-simulator/inference-engine/src/csrc/src/simulations/validation/setup/mew3-host-clang-includes.patch
git apply ../reef-mlp-simulator/inference-engine/src/csrc/src/simulations/validation/setup/mew3-host-clang-includes.patch
```

Neither environment fix changes the target numerical code or reference checks.
The pre-existing Coral change to `coralnpu_test_utils/core_mini_axi_interface.py`
was retained.

## Working commands

The dependencies are now cached. These commands reproduce the successful build
and functional runs with the local toolchain patch above. `--repository_disable_download`
keeps these reruns offline; omit it in a normal SSH terminal if the caches have
been removed or new dependencies must be fetched.

```bash
set -euo pipefail
REEF_CHECKOUT=/home/samlin/coralnpu-dissection/reef-mlp-simulator
CORAL_CHECKOUT=/home/samlin/coralnpu-dissection/coralnpu
SIM_SOURCE="$REEF_CHECKOUT/inference-engine/src/csrc/src/simulations"
REEF_SIM_LOGS="$SIM_SOURCE/validation/logs"
mkdir -p "$CORAL_CHECKOUT/examples/reef_mlp" "$REEF_SIM_LOGS"
cp -R "$SIM_SOURCE/." "$CORAL_CHECKOUT/examples/reef_mlp/"
cd "$CORAL_CHECKOUT"

REEF_BAZEL_FLAGS=(
  --experimental_collect_system_network_usage=false
  --repository_cache=/nfs/samlin/coral-build/repository-cache
  --repo_contents_cache=/tmp/reef-w8a8-repo-contents
  --disk_cache=/tmp/reef-w8a8-disk-cache
  --sandbox_add_mount_pair=/tmp/reef-w8a8-repo-contents
  --repository_disable_download
)
bazel --output_user_root=/tmp/reef-w8a8-bazel --batch build \
  //examples/reef_mlp:validate_functional "${REEF_BAZEL_FLAGS[@]}" \
  2>&1 | tee "$REEF_SIM_LOGS/functional-build-rerun.log"

timeout 120 bazel --output_user_root=/tmp/reef-w8a8-bazel --batch run \
  //examples/reef_mlp:validate_functional "${REEF_BAZEL_FLAGS[@]}" -- \
  --smoke-only 2>&1 | tee "$REEF_SIM_LOGS/functional-smoke-rerun.log"
# Proceed only after float-add passes; this run rechecks it before the MLP.
timeout 120 bazel --output_user_root=/tmp/reef-w8a8-bazel --batch run \
  //examples/reef_mlp:validate_functional "${REEF_BAZEL_FLAGS[@]}" \
  2>&1 | tee "$REEF_SIM_LOGS/functional-mlp-rerun.log"

# After correctness passes, optionally recapture the executed instruction mix.
timeout 120 bazel --output_user_root=/tmp/reef-w8a8-bazel --batch run \
  //examples/reef_mlp:validate_functional "${REEF_BAZEL_FLAGS[@]}" -- \
  --trace-prefix "$REEF_SIM_LOGS/mlp-instructions-rerun" \
  2>&1 | tee "$REEF_SIM_LOGS/functional-trace-rerun.log"
```

The validator checks input writes and eight exact FP32 float-add outputs first.
It loads a fresh MLP simulator, verifies loaded `mlp_status=-1`, and requires a
normal halt, `mlp_status=0`, `mlp_failures=0`, and `_ret=0`. It prints all 24 FP32
outputs. Process exit alone is not the success criterion. Execution has a
1,000,000-instruction step budget plus the command's 120-second timeout.

## ELF and startup evidence

```text
ELF: bazel-out/k8-fastbuild-ST-dd8dc713f32d/bin/examples/reef_mlp/mlp_w8a8_test.elf
SHA256: 5c43b988880f8da1c7e94b40bf3597ef6314c83db9f49ba4b0e35a8e976cfbf5
entry: 0x0
mlp_status:   0x10000, size 4, loaded -1; completed 0
mlp_failures: 0x10060, size 4; completed 0
mlp_output:   0x1008c, size 96
_ret:         0x10054; completed 0
halted PC:    0x150, after terminal mpause at 0x14c
SP:           0x18000
```

Loadable text/CRT/rodata occupies `[0,0x16c0)` (5824 bytes); `.data` is
96 bytes, `.bss` is 744 bytes, and the stack is `[0x17800,0x18000)`.
The intervening unused DTCM is a linker heap section; the harness does not
allocate. Startup initializes SP/GP, clears BSS, installs the EBREAK trap
handler, enables FS/VS, and sets `_ret=0x0badd00d` before main. It saves main's
return value before the success `mpause` or failure `ebreak`.

## Executed instruction mix

The pinned MPACT RISC-V core maintains total and per-opcode executed-instruction
counters and connects `minstret`/`instret` CSRs. Its `RiscVTop::Step` increments
the instruction and opcode counters once for each successfully executed
instruction, including the halt instruction. The Python binding does not
expose the per-opcode counters or a complete trace callback. Its PC, memory
reads and `Step(1)` APIs are sufficient to capture a dynamic trace without
changing the kernel, fixture, CRT or simulator implementation.

`--trace-prefix` records the PC and instruction word before each successful
`Step(1)`, classifies the encoding, and stops immediately after executing
`mpause`/`ebreak`. This matters because another `Step` call would resume the
halted simulator. Unsupported/compressed encodings fail classification instead
of being silently counted. The resulting CSV has 20,069 executed rows, matching
the ordinary run's returned step total; category counts sum to the same total.
The traced run passed the full original correctness checks again.

| Whole-program category | Executed instructions |
| --- | ---: |
| Loads, including scalar FP32 and vector loads | 3,886 |
| Stores, including scalar FP32 and vector stores | 1,083 |
| Scalar integer arithmetic | 7,030 |
| Scalar address arithmetic (`lui`/`auipc`) | 1,151 |
| Scalar FP arithmetic/conversions/comparisons | 3,956 |
| Branches | 2,552 |
| Jumps/calls/returns | 400 |
| System/CSR/halt instructions | 9 |
| Vector arithmetic/configuration | 2 |
| **Total, disjoint rows above** | **20,069** |

Scalar arithmetic including address construction totals **12,137**. All vector
instructions total **6**: two `vsetvli`, two `vle8.v`, and two `vse8.v`; the four
vector memory instructions are already included in the loads/stores rows.
Trace PCs and disassembly attribute all six to a compiler-generated 32-byte
copy of the ties-to-even `ties[]` fixture in `main` at `0xa30`, `0xa34` and
`0xa40`. This does not establish an RVV-optimized MLP kernel.

**Scope is the entire MLP ELF**, from entry through terminal `mpause` inclusive:
CRT/startup, input quantization, MLP and math-library calls, exact/reference
checks, edge-case tests and finalization. The separate float-add run is excluded.
These are not kernel-only counts. No hardware cycle, latency, utilization,
throughput, memory traffic in bytes or vector-lane occupancy is inferred.

## Saved logs and review

All evidence is under `validation/logs/`. Packaged logs have trailing whitespace
removed, and the trace CSV uses LF line endings; recorded values are unchanged:

- `functional-build-host-includes-20261011.log`: successful build with exact flags.
- `functional-smoke-20261011.log`: float-add PASS, outputs and `_ret`.
- `functional-mlp-20261011.log`: MLP PASS, loaded/completed status and all outputs.
- `functional-trace-20261011.log`: fresh traced run, all checks PASS.
- `mlp-instructions-20261011.csv/.json`: dynamic trace and categorized counts.
- `vector-instruction-attribution-20261011.log`: executed vector PCs and disassembly.
- `sandbox-build-debug-20261011.log`, `sandbox-path-diagnosis-20261011.log`:
  script/symlink/interpreter checks and original-versus-corrected sandbox evidence.
- `functional-build-cache-mount-20261011.log`: PyWriteBuildData fixed, next host
  include-root failure exposed.
- `ssh-functional-build-20261011.log`: initial SSH build after dependency fetch.
- `inspection-20261010.log`: Coral revision, ELF layout/symbols/SHA256 and
  machine build settings; unrelated Coral diff content omitted during packaging.
- `package-checks-20261011.log`: final packaging checks and small-fixture rerun.
- `review-20261011.log`: source consistency, trace accounting, final results and review.

The reviewed changes are the functional runner/Bazel target, optional dynamic
trace capture, documentation/logs, and the machine-specific host-toolchain
patch. The numerical implementation and original target checks are unchanged.
No correctness blocker remains for this small fixture; real-model validation
and hardware performance remain future work. The package includes only harness
source, documentation, machine-specific setup evidence and useful text logs;
caches, binaries and temporary files are excluded.

## Next milestone after the small harness passes

1. Proposed next milestone (not implemented): validate one real
   TinyStories-Instruct-28M MLP block using trained weights and representative
   pre-MLP activations captured from real token sequences. Record checkpoint,
   layer, tokenizer/input, quantization scales, reference version and artifact
   hashes. Check both INT32 accumulations, requantized codes, and FP32 stages
   against the same W8A8 reference contract before making model-quality claims.
2. Plan placement for 512→2048→512 before compiling a full-size fixture. The two
   signed INT8 matrices total `512*2048 + 2048*512 = 2,097,152` bytes (2 MiB),
   excluding scales, biases, reference data and scratch. Keep weights in
   EXTMEM/DDR using explicit linker sections, matching simulator memory regions,
   and a loader whose address/permissions are verified. Budget stack and runtime
   scratch separately; 3-token hidden FP32 and INT32 buffers are each 24 KiB,
   hidden INT8 is 6 KiB, and the other buffers add further pressure. Consider
   token/channel tiling, buffer reuse, and keeping reference data on the host.
   A high-memory TCM configuration alone does not establish default-device fit.
3. After real-weight scalar correctness and memory-placement checks pass,
   implement RVV kernels and compare every stage to the scalar baseline. Then
   measure performance on an appropriate RTL simulator or hardware, including
   external-memory traffic and stalls, with dimensions, tool settings and timing
   methodology recorded. Functional simulator counts are not cycle-accurate
   performance measurements. Do not run a simulator precision sweep (Richard's
   requested constraint).
