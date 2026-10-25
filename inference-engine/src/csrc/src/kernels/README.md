# Coral M3 kernels for reef-perf

One standalone int8 kernel per engine operator (`src/csrc/include/inference_engine/operator/`).
Each file has its own `_start`, fills its inputs from a fixed seed, runs the kernel once and
stops with `mpause`, so it runs directly on reef_perf. Lookup tables (GELU, softmax exp) are
precomputed on the host and stored as `const` arrays.

    linear.cc      y = x . W^T + bias
    activation.cc  GELU table lookup, softmax via an exp table
    attention.cc   one causal head: q . k^T, softmax, p . v
    mlp.cc         linear -> GELU -> linear
    embedding.cc   token row + position row
    layernorm.cc   integer mean, variance and inverse std per row

## Build and run

Compile with Coral M3's GCC (`toolchain_kelvin_v2`) and reef-perf's linker script:

    ./compile.sh linear            # -> linear.elf

Then, from `reef-perf/` in its Docker shell:

    build/bin/reef_perf --elf ../inference-engine/src/csrc/src/kernels/linear.elf -c configs/m3.yaml

Add `-DCHECK` (`./compile.sh linear -DCHECK`) to compare against a scalar reference at the end; a mismatch
traps reef_perf (`trap at ... (ebreak)`). reef_perf's count covers the whole program, input
fill included; `build/bin/reef_trace` shows which instructions belong to the kernel.
