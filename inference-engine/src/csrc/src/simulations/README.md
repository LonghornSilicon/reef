# Coral NPU kernels

Int8 transformer kernels for the Coral NPU (RISC-V rv32imf_zve32x with the
VME matrix engine), with host unit tests and tests on Coral's RTL simulator.

## Layout

    kernels/      the kernels (vme_* run on the matrix engine)
    includes/     helpers: VME and RVV instruction wrappers, quantization
    unit_tests/   host C++ tests, one per kernel family
    sim/          bench_main.cc (the ELF harness) and the Bazel BUILD
    tests/        sim tests, one per ELF, plus the shared harness bench.py
    bin/          the bench ELFs
    analysis/     performance model, report and kernel notes (not needed
                  to build or test)

## Commands

Run from this folder. Docker is needed for anything that touches the
simulator; the first run builds the coralnpu image and RTL model, which
takes hours on Apple Silicon.

    python run_unit_tests.py              all host unit tests (clang++ only)
    python run_unit_tests.py test_flash   one host unit test
    python build_elfs.py                  build every ELF into bin/
    python build_elfs.py bench_matmul     build one ELF
    python run_tests.py                   run every sim test on bin/
    python run_tests.py vme_attention     run one sim test

Sim tests and the ELF each one loads:

    matmul          bin/bench_matmul.elf         VME and RVV matmul tiles
    attention       bin/bench_attention.elf      scalar serial/flash attention
    vme_attention   bin/bench_vme_attention.elf  attention on the VME
    pointwise       bin/bench_pointwise.elf      softmax, GeLU, LayerNorm, residual
    ubench          bin/bench_ubench.elf         load latency, host write speed

Each sim test checks every kernel call against NumPy, prints its cycle
count, and fails if any result is out of tolerance. Logs and per-call
records go to out/<test>/. The tests only need the ELFs in bin/, so
prebuilt ELFs can be dropped in there and run without building.

## Running the ELFs on another simulator

Each ELF runs one kernel call per run and halts. Before a run, write the
inputs into the `arena` symbol (20 KB in DTCM) and the call into
`bench_args` (word 0 is the op, then its arguments; pointers are byte
offsets into the arena, floats are their IEEE-754 bits). After the halt,
`bench_cycles` holds the kernel's cycle count and the outputs are in the
arena. Op numbers and argument order are in sim/bench_main.cc, and the
tests in tests/ show the exact call sequence for each workflow, e.g. one
attention head is vme_attention_block once per K/V block, then the finish
step, with the running state left in the arena between runs.

## Performance model

    python analysis/collect_results.py    out/ -> analysis/results/results.json
    python analysis/roofline.py           roofline and end-to-end estimate
    python analysis/roofline.py --model analysis/config/gpt_neo_28m.json

These need numpy and matplotlib (`uv run --group sim python ...`). The
write-up is analysis/REPORT.md.
