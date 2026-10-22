# Coral NPU kernels

Int8 transformer kernels for Coral NPU M3, the base Reef uses
(rv32imf_zve32f: scalar core plus the RVV vector unit). Each .cc file
implements one inference-engine operator
(`src/csrc/include/inference_engine/operator/`).

    linear.cc      linear: matmul on the vector unit, linear()
    attention.cc   attention: block-flash attention
    activation.cc  activation: GeLU, LUT softmax
    mlp.cc         mlp: c_fc -> GeLU -> c_proj, streamed over the inner width
    embedding.cc   embedding: token lookup, position add
    kernels.h      every kernel's signature and contract (shapes, scales, limits)
    helpers.h      quantization and RVV instruction wrappers
    main.cc        runs one kernel on M3 or reef_perf (see below)

M3 has no matrix engine (VME). Newer Coral has one; `-DCORAL_VME` turns
it on for matmuls with tile-friendly shapes. Never use it on M3.

## Passing values

Every kernel is a plain `extern "C"` function that takes pointers, so the
caller decides where the data lives. main.cc sets the standard for running
one on a simulator: every input, output and scale is a **named global in
`.data`**. This works on both simulators:

- **reef_perf (Spike):** there's no host to write memory, so main() fills
  the inputs from a fixed seed. Every run sees the same data.
- **M3 RTL testbench:** it writes the globals by symbol name (`x`, `wt`,
  `acc_scale`, ...) and sets `fill_inputs = 0` before the core starts, the
  same way Coral's own tests load `lhs_input`. Outputs are read back the
  same way.

`cycles` holds mcycle around the kernel calls only, not the one-time table
builds. The globals for each kernel are in main.cc under its
`KERNEL_<NAME>` block.

## Build

Pick one kernel per ELF with `-DKERNEL_LINEAR`, `_ATTENTION`, `_MLP`,
`_GELU`, `_SOFTMAX` or `_EMBEDDING`. Shapes default to 16 rows of a
64-wide head; set `-DROWS=1` for decode, and `-DD=` and `-DCOLS=` for
other widths.

**Without Bazel**, use M3's GCC (`toolchain_kelvin_v2-2025-09-11`, the
archive coralnpu's M3 WORKSPACE downloads) and reef-perf's linker script:

    riscv32-unknown-elf-g++ -march=rv32imf_zve32f_zicsr_zifencei_zbb \
        -mabi=ilp32 -std=c++17 -O3 -fno-exceptions -fno-rtti \
        -ffunction-sections -fdata-sections -Wl,--gc-sections \
        -nostartfiles -T reef-perf/src/reef_perf/experiments/run/workloads/link.ld \
        -DREEF_START -DKERNEL_LINEAR -I. \
        main.cc linear.cc attention.cc activation.cc mlp.cc embedding.cc \
        -lm -o linear.elf
    reef_perf --elf linear.elf -c configs/m3.yaml

**In coralnpu at the M3 tag**, Coral's own startup code replaces
`-DREEF_START`:

    coralnpu_v2_binary(
        name = "linear",
        srcs = ["main.cc", "linear.cc", "attention.cc", "activation.cc",
                "mlp.cc", "embedding.cc"],
        hdrs = ["kernels.h", "helpers.h"],
        copts = ["-DKERNEL_LINEAR"],
    )

Both builds use M3's default memory map (8 KB ITCM at 0x0, 32 KB DTCM at
0x10000), and the program ends in `mpause`. So the same ELF runs on
reef_perf and on the M3 RTL.

To check the logic on a host machine, `-DRVV_EMULATE -DVME_EMULATE`
replaces the vector instructions with C:

    clang++ -std=c++17 -DRVV_EMULATE -DVME_EMULATE -DKERNEL_LINEAR -I. \
        main.cc linear.cc attention.cc activation.cc mlp.cc embedding.cc

## Calling

Weights are uint8 with a zero point and stored transposed, `[in x out]`.
Activations are int8 with zero point 0. Two operators take more than one
call:

- **Attention**, per head: call `vme_attention_block` once per K/V block,
  then `attention_flash_finish`. The `vme_` name is historical; on M3 it
  runs on the vector unit.
- **MLP**: call `mlp_block` once per inner-width chunk, then `mlp_finish`.

Keep the running state in memory between those calls. kernels.h gives
each function's buffer sizes and limits.
