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

## Real-model host milestone — 2026-10-11

Implementation is confined to Reef. The scalar kernel, synthetic fixture,
reference generator and target harness are unchanged. Coral was inspected only;
its existing modifications and simulator setup were left untouched. This stage
runs on the host, with no full-size Coral execution, optimization or precision
sweep. The existing small functional result above remains the target baseline.

### Pinned capture and export

The model and tokenizer are `roneneldan/TinyStories-Instruct-28M`, both pinned to
revision `96c7ddf6012aaec1b227a1a663f09de016de0621`. Capture the fourth block,
`transformer.h.3.mlp`, in CPU FP32 eval/inference mode, eager attention, one
thread, deterministic algorithms. Use the existing `summary_only` prompt from
`workloads.experiments.story_quality.main.PROMPTS`, without added special tokens.
A forward pre-hook captures actual post-`ln_2` inputs from a single prompt
prefill. Select positions `0`, `N//2`, and `N-1`; record the complete input IDs,
attention mask, selected positions/tokens and prompt hash. These are three
representative samples from one prompt, not a calibration corpus or a model
quality benchmark.

Checkpoint tensors are row-major `[output,input]`: `c_fc.weight` is
`[2048,512]`, `c_proj.weight` is `[512,2048]`; both FP32 biases are retained.
`export_real.py` reuses the existing per-row quantizer: signed symmetric
`[-127,127]`, ties-to-even, weight scales per output row, activation scales per
token, all-zero scale 1. Original FP32 operands, INT8 weights, scales, biases,
three activation inputs and all expected stages are saved in an external NPZ.
An immutable manifest records each array's shape/type/byte count/value hash,
archive hash, checkpoint file hashes, tokenizer identity, software versions,
source hashes and workloads lock hash. Generated arrays, libraries, caches,
environments and build files must stay outside Git.

### Comparison and validation criteria

`validation/host_bridge.cc` calls the unchanged kernel entry points; a separate
first-Linear call preserves the pre-GELU diagnostic buffer. Build with C++17,
`-O2 -ffp-contract=off`, warnings as errors, and the host math library. The
comparison first checks artifact integrity and recomputes the PyTorch oracle
from original operands, validating shapes before entering C++. Operand hashes
must remain unchanged after execution.

Require exact INT8 input/hidden codes and exact INT32 first/second accumulators.
The oracle computes integer products in INT64 and checks the INT32 range.
Require finite input/hidden scales, pre-GELU, GELU and final FP32 values within
`abs(actual-reference) <= 2e-6 + 2e-5*abs(reference)`. The dequantization order
remains `(float(accumulator)*input_scale)*weight_scale+bias`; GELU retains the
existing PyTorch expression. Differences from host `tanhf` or power-expression
rounding are investigated, especially near activation half-code boundaries;
integer checks and tolerances are never relaxed automatically. FAIL reports
preserve the first mismatch and requantization diagnostics.

Independently check the exported unquantized reference against the actual
checkpoint MLP output using the same tolerances. Report W8A8 versus original
FP32 errors separately (max absolute, mean absolute, RMSE, relative L2) for
pre-GELU, hidden and output. These errors describe quantization, not kernel
correctness. Fake-quantized FP32 matmul uses a different reduction order and is
reported separately with activation-code differences; it is not the exact
integer oracle. The original synthetic generator's checks remain intact.

### Files and reproducible commands

- `inference-engine/experiments/w8a8_mlp/export_real.py`: pinned capture/export.
- `inference-engine/experiments/w8a8_mlp/real_support.py`: artifact isolation,
  native build provenance and static placement checks (both under `src/`).
- `validation/compare_real_host.py`, `host_bridge.cc`: stage comparisons and ABI.
- `validation/test_real_numerics.py`: Torch-dependent comparison/integrity tests.
- `tests/python/test_real_mlp_support.py`: dependency-light placement tests.
- `validation/real_block_memory.json`: checked proposed buffer map.
- `validation/Dockerfile.host`, `run_real_host_container.sh`: isolated locked run.

From the Reef root in a normal SSH terminal:

```bash
bash inference-engine/src/csrc/src/simulations/validation/run_real_host_container.sh
```

The runner prints `/tmp/reef-real-host.XXXXXX`, runs as the caller's UID/GID,
mounts Reef read-only, mounts CMake build output separately and uses external
Python environments. It installs Python 3.12, Doxygen, GoogleTest through the
existing CMake pin, uv 0.10.10 and locked lint/workloads dependencies. It runs
all inference-engine CI checks, the new numerical tests, synthetic export/host
comparison, then real capture/host comparison. Coral is not mounted. Logs,
exit status, fixtures and reports remain in the printed directory, writable
by the caller. Keep NPZ/checkpoint files there; retain only manifests and
concise reports in this checkout after review.

Equivalent host commands with the locked workloads environment already set up:

```bash
export PYTHONPATH="$PWD/inference-engine/src:$PWD/workloads/src"
uv run --project workloads --locked --no-default-groups --group dev python -m \
  inference_engine.experiments.w8a8_mlp.export_real \
  --cache /tmp/reef-real-cache --output /tmp/reef-real-fixture
uv run --project workloads --locked --no-default-groups --group dev python \
  inference-engine/src/csrc/src/simulations/validation/compare_real_host.py \
  --fixture /tmp/reef-real-fixture --report /tmp/reef-real-results/report.json
```

Use a new fixture directory for each export. Add `--synthetic-check` to export
and `--allow-synthetic` to comparison only for the separate small smoke path.
The committed fixture is checked by the original host harness test, not replaced
by this newly exported smoke fixture.

### External-memory review

The inspected Coral revision is
`dab25ad632e35d6399b6d1aa1f05c30e700f06eb`. Definitions are
`rules/linker.bzl`, `toolchain/coralnpu_tcm.ld.tpl` and
`toolchain/crt/coralnpu_start.S`. The default map is 8 KiB ITCM at 0, 32 KiB
DTCM at `0x10000`, and 4 MiB EXTMEM at `0x20000000`. A nondefault TCM size
changes the DTCM origin; use explicit default regions for a future guest run.
The two INT8 matrices alone consume **2,097,152 bytes (2 MiB)**.

| Proposed placement | Bytes | Address span |
| --- | ---: | --- |
| Two INT8 matrices in EXTMEM | 2,097,152 | `0x20000000`–`0x20200000` |
| FP32 weight scales and biases | 20,480 | `0x20200000`–`0x20205000` |
| First INT32 accumulators | 24,576 | `0x20205000`–`0x2020b000` |
| FP32 hidden scratch | 24,576 | `0x2020b000`–`0x20211000` |
| INT8 hidden scratch | 6,144 | `0x20211000`–`0x20212800` |
| Preserved FP32 pre-GELU diagnostics | 24,576 | `0x20212800`–`0x20218800` |
| DTCM inputs, scales, second accumulators, output | 19,992 + 8 padding | `0x10000`–`0x14e20` |
| DTCM runtime/global reserve | 10,720 | `0x14e20`–`0x17800` |
| DTCM stack proposal | 2,048 | `0x17800`–`0x18000` |

Spans are end-exclusive. EXTMEM usage is **2,197,504 bytes**, leaving
1,996,800 bytes. DTCM reserves the entire 32 KiB including alignment gaps.
The JSON checks every shape, byte count, alignment, overlap and region bound.
Keep reference tensors on the host. Put constant weights/scales/biases in
`.extdata` and scratch in `.extbss`; the CRT clears normal BSS only, so explicitly
initialize external scratch. These are proposed offsets, not a verified linker
allocation. A future guest ELF must verify section placement, loader mappings,
ITCM code fit, globals fitting the runtime reserve and stack sufficiency.
No DDR allocation is needed for this proposal, and no timing/utilization claim
follows from these sizes or from functional instruction counts.

### Verified host results — 2026-10-11

**PASS** in `/tmp/reef-real-host.Ey53TI`, container exit status 0. The locked
run uses Python 3.12.15, PyTorch 2.14.0+cu126, Transformers 5.17.0, NumPy 2.5.3
and GCC 12.2.0, CPU only. All 12 inference-engine pytest cases (including
Doxygen and the original synthetic harness), all 6 numerical validation tests,
Ruff, C++ formatting/build/clang-tidy and bridge lint passed. The synthetic
export/host comparison also passed. The 17 existing C++ tests previously passed
in the native build; the container runner builds these targets but does not
separately invoke CTest.

The fixed prompt has 22 tokens. Captured positions `0,11,21` decode to
`"Summary"`, `" helps"`, and `"\n\n"`, respectively. All **15,360 integer
values** match exactly: 1,536 input codes, 6,144 first accumulators, 6,144 hidden
codes, and 1,536 second accumulators. All operands remain unchanged.

| Stage against the PyTorch W8A8 oracle | Maximum absolute FP32 difference |
| --- | ---: |
| Input scales | 0 |
| First Linear / pre-GELU | 0 |
| Tanh-GELU hidden | 1.1920928955078125e-7 |
| Hidden scales | 0 |
| Final output | 0 |

Every value passes the unchanged `atol=2e-6, rtol=2e-5` checks. The independent
original-FP32 expression equals the actual checkpoint MLP output exactly.
The fake-quantized FP32 diagnostic has maximum differences of
`4.76837158203125e-7` before GELU and `1.430511474609375e-6` at output, with
zero hidden-code differences; its reduction order remains distinct from the
integer oracle.

Quantization error against the original unquantized FP32 block is separate:

| Stage | Max absolute | RMSE | Relative L2 |
| --- | ---: | ---: | ---: |
| Pre-GELU | 0.0886997581 | 0.0234239040 | 2.47218695% |
| Hidden | 0.0727348924 | 0.00853564112 | 4.38648210% |
| Output | 0.0391699374 | 0.0109413178 | 3.75246786% |

These describe this block on three captured inputs; they are not story-quality
or full-model accuracy claims.

### Checkpoint loading audit

The container loader reports only `UNEXPECTED` groups for
`transformer.h.{0..7}.attn.attention.bias` and `.masked_bias`. A separate
read-only checkpoint audit verified **108 independent learned parameter
tensors**, with no missing keys, shape/type mismatches or nonfinite values.
The only 16 extra keys are eight boolean causal/local masks of shape
`[1,1,2048,2048]`, and eight obsolete scalar FP32 masking sentinels equal to
`-1e9`. Every mask exactly matches the configured global/local attention rule.
In the installed Transformers source, the mask is now an `nn.Buffer` with
`persistent=False`; the old `masked_bias` buffer is absent.

The checkpoint omits the separate `lm_head.weight` state-dict alias. It is tied
to `transformer.wte.weight`, which is present: this is not an independently
missing learned parameter. The installed Transformers class explicitly records
that tie. The audit used Reef's GPT-Neo architecture on the meta device to check
all independent names/shapes, without executing a model. All four exported
original layer-3 MLP weights/biases equal checkpoint tensors exactly. The audit
used local PyTorch 2.13.0+cu130 for tensor inspection; numerical validation used
the locked container versions above.

### Saved evidence and review

Concise JSON reports and manifests are under `validation/reports/`:

- `real-host-20261011.json`: all stage checks, quantization errors, native build
  command/source hashes and static memory check.
- `real-provenance-20261011.json`: pinned model/tokenizer, complete token IDs,
  checkpoint/source/lock hashes and every array's shape/type/value hash.
- `checkpoint-audit-20261011.json`: complete unexpected-key inventory, mask
  checks, learned-parameter coverage and tied-alias explanation.
- `synthetic-container-20261011.json`: synthetic path on locked dependencies.
- `host-review-20261011.json`: reviewed results and evidence hashes.

The useful container log is `validation/logs/real-host-container-20261011.log`;
ANSI styling, progress-line separators and trailing whitespace are normalized.
The read-only audit recipe is retained verbatim as
`validation/logs/checkpoint-audit-script-20261011.txt` (its hash is in the audit
report). Earlier preliminary synthetic reports and focused-check logs remain
clearly labeled with their Python/PyTorch versions. All generated arrays,
checkpoint files, native libraries, environments and caches remain outside Git
in the printed evidence directory. Their hashes and the executed source/lock/map
hashes were independently checked against current Reef files.

The memory-plan checker passes every buffer size/type/alignment, region bound
and overlap check: EXTMEM uses **2,197,504 bytes** and leaves **1,996,800 bytes**.
DTCM uses 32,760 buffer/reserve bytes plus **8 bytes of alignment gaps**, spanning
the full 32 KiB; those gaps are not extra contiguous runtime space. ITCM remains
unallocated by this static data plan. Guest ELF code fit, loader mappings,
global/runtime usage, explicit external scratch initialization and stack
sufficiency are future guest checks.

The reviewed change adds only the real-host exporter/comparison/bridge, tests,
container runner, static placement proposal, documentation and text/JSON
evidence. The scalar kernel, original fixture, synthetic generator, target
harness and numerical checks are unchanged. Coral remains untouched. No host
correctness blocker remains for this pinned block and these three inputs.
Nothing has been committed or pushed. Stop here: full-size Coral execution,
optimization, precision sweeps and hardware timing/utilization claims remain
outside this milestone.
