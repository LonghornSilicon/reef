# coral: GPT-Neo on the Coral NPU core

Everything here is cross-compiled for the CoralNPU M3 core (RV32IMF, Zbb,
Zve32f, VLEN 128) and runs on `reef_perf` or the chip. The host C++ engine in
`../src/csrc/include/inference_engine/` is the source of truth. Every kernel
reproduces one engine function **bit for bit**, and the tests fail if a
single bit differs.

```
coral/
  include/reef_kernels/
    kernels.hpp      kernel API; each kernel names the engine function it matches
    model_blob.hpp   byte layout of a packed model (host writes, firmware reads)
    host_if.hpp      mailbox the host uses to drive the firmware
  kernels/           the library: one function per operator and mode, no main, no globals
  runtime/           crt0.S (start-up, written once), link.ld (M3 highmem map), libc.cc
  firmware/          gpt_neo.cc stitches the kernels into the model; main.cc serves the mailbox
  tests/             golden tests: run a kernel on the core, compare with the engine's output
  bench/             reef-perf benchmarks: 8M decode shapes; *_int8 at the original 16x64 shapes
  host/              gen_golden (writes the tests' expected outputs), pack_model (packs weights)
  sim/               coral_sim: Spike without the timing model, for CI
  tools/             setup_toolchain.sh, setup_sim.sh, build.sh, run.sh, test.sh
```

The benchmarks, the tests and the firmware all link the same `kernels/` code.
The kernel that gets benchmarked is the kernel that gets tested, and that's
the kernel the demo runs.

## Precision modes

Each mode is a set of engine functions with kernels that match them:

| Mode | Engine | Kernels | Firmware |
|---|---|---|---|
| fp32 | float operators, `math.hpp` | `*_f32` | yes |
| W8A8 | `quant.hpp`: int8 projections, float elsewhere | `linear_w8a8*` + `*_f32` | yes |
| int8 | `int8.hpp`: integer only, shifts and lookup tables | `*_int8` | not yet |

The int8 kernels are the original standalone ones (linear, GELU and softmax
tables, attention, LayerNorm, embedding), now library functions checked
against `int8.hpp`. Its `gelu_table(0.05, 0.05)` and `exp_table()` reproduce
the tables they hard-coded. The firmware can't run an int8 model yet: that
needs a scale for every activation tensor, picked by calibrating against
the fp32 model, plus a rescaling residual add.

## How kernels are kept equal to the engine

1. The engine defines the arithmetic exactly. Float steps happen in a fixed
   order with no fused multiply-adds (`-ffp-contract=off` on both sides).
   `exp` and `tanh` come from `math.hpp` rather than the C library. W8A8
   rounding follows `quant.hpp`.
2. `host/gen_golden.cpp` calls the engine on seeded inputs and writes
   `build/coral/golden/<test>.hpp`. The expected outputs are hex-float
   literals, so they're exact.
3. Each `tests/test_*.cc` runs the kernels on those inputs on the core and
   compares the outputs bit for bit. A mismatch executes `ebreak`, so
   reef_perf reports a trap and exits non-zero.
4. `tests/test_gpt_neo.cc` checks the firmware's whole forward pass on a
   small random model, in fp32 and W8A8:
   - the logits of every generated token,
   - the generated tokens when the request goes through the host mailbox.

Changing a kernel's arithmetic without changing the engine fails these tests.
To change the arithmetic itself, change the engine first: its gtests and the
Hugging Face comparison judge whether the change is acceptable. Then update
the kernel until `test.sh` passes again.

Integer sums may run in any order. Float sums must run in the engine's
order. Lanes that each accumulate over the inner dimension in order are fine,
and so is the ordered reduction `vfredosum`. An unordered `vfredusum` or a
fused `vfmacc` is not.

## Running it

One-time setup. You need Docker, plus the reef-perf image that
`tools/docker.sh build` builds at the repo root:

```sh
coral/tools/setup_toolchain.sh     # Coral's GCC into the coral-toolchain Docker volume
```

Then, from `inference-engine/`:

```sh
coral/tools/test.sh                          # golden headers, cross-compile, run every test
coral/tools/build.sh bench                   # build/coral/bench/*.elf
coral/tools/run.sh build/coral/bench/linear_w8a8.elf   # reef_perf's cycle summary
```

`build.sh` uses `riscv64-unknown-elf-g++` from `$CORAL_TOOLCHAIN/bin` or
`PATH` if it finds one. Otherwise it runs inside a `linux/amd64` container,
because the toolchain is x86-64 Linux. `run.sh` uses `$CORAL_SIM` if set,
else `reef_perf` from `$REEF_PERF` or `PATH`, else the reef-perf image.

## CI

`.github/workflows/coral-kernels.yml` runs `test.sh` on every push and pull
request, natively on an x86-64 runner:

- `setup_toolchain.sh DIR` installs only GCC, stripped, from the pinned
  archive.
- `setup_sim.sh PREFIX` builds Spike (reef-perf's `build_deps.sh`) and
  `sim/coral_sim.cpp`. That's reef-perf's `SpikeDriver` without the Sparta
  timing model: same ISA, memory map and `mpause` handling, so pass/fail
  matches reef_perf, but it builds in minutes rather than an hour.

Both are cached, so a warm run mostly compiles and runs the tests. You can
reproduce CI on any x86-64 Linux box:

```sh
coral/tools/setup_sim.sh ~/coral-sim          # needs the reef-perf/ext/spike submodule
coral/tools/setup_toolchain.sh ~/coral-gcc
CORAL_TOOLCHAIN=~/coral-gcc CORAL_SIM=~/coral-sim/bin/coral_sim coral/tools/test.sh
```

## Adding or changing a kernel

1. Make sure the engine function exists and is tested under `tests/cpp/`.
2. Declare the kernel in `kernels.hpp` and name the engine function it
   matches. Implement it in `kernels/`.
3. Add cases to `host/gen_golden.cpp`. Use odd sizes, so vector strips end
   partway through, and use edge values. Then check them in a
   `tests/test_*.cc`.
4. Add a benchmark under `bench/` if reef-perf should measure the kernel.
5. Run `coral/tools/test.sh` and `bash tools/check_cpp.sh` (formatting).

Before relying on a new test, mutate the kernel once (for example, fuse a
multiply-add) and confirm that the test fails.

## Host and firmware

The host packs a checkpoint with `pack_model --model DIR [--precision
fp32|w8a8] OUT`, writes the blob where the core can read it, fills the
mailbox at `0x100000`, and releases the core. The firmware feeds the prompt,
appends greedy tokens, sets `status`, and halts with `mpause` (see
`host_if.hpp`). One request is one run, and every kernel call happens inside
`firmware::GptNeo::step`.

## Known limits

- Packed TinyStories models are too big for the simulator. The 28M model is
  133 MB in W8A8 and 208 MB in fp32, mostly the float token embedding that
  doubles as `lm_head`. reef_perf only offers the 1 MB highmem map, so the
  firmware test uses a small random model. Running a real checkpoint needs an
  external-memory region in reef_perf and on the chip, with the weights
  streamed into DTCM.
- CI checks correctness only. Cycle counts still need reef_perf (the
  Docker image), so benchmarks are not tracked in CI.
- `bench/` programs count the input fill as well as the kernel. Use
  reef-perf's `reef_trace` to isolate the kernel.
