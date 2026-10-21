# Coral NPU kernels

Int8 transformer kernels for the Coral NPU (RISC-V rv32imf_zve32x with the
VME matrix engine), with host unit tests and tests on Coral's RTL simulator.
Only the fastest version of each op is kept; superseded kernels (scalar
serial/flash attention, `expf` softmax) are in git history before the PR
#16 trim.

## Files

    kernels.h          every kernel's contract
    helpers.h          quantization, RVV and VME instruction wrappers
    matmul.cc          VME and RVV matmul, and the matmul() dispatcher
    attention.cc       block-flash attention on the VME, LUT exp table
    pointwise.cc       GeLU, LayerNorm, residual add
    test_kernels.cc    host unit tests (RVV and VME emulated in C)
    bench_main.cc      the sim ELF harness; BUILD builds it in coralnpu
    bench.py           sim test harness and NumPy references
    test_attention.py  sim test: matmul tiles, k sweep, attention streams
    test_pointwise.py  sim test: GeLU, LayerNorm, residual add
    sim.py             runs all of the above
    analysis/          results log, roofline model and its configs

## Commands

Run from this folder. Docker is needed for anything that touches the
simulator; the first run builds the coralnpu image and RTL model, which
takes hours on Apple Silicon.

    python sim.py unit                  host unit tests (clang++ only)
    python sim.py build                 build both ELFs into bin/
    python sim.py test                  run both sim tests on bin/
    python sim.py test attention        run one sim test

Each sim test checks every kernel call against NumPy, prints its cycle
count, and fails if any result is out of tolerance. Logs and per-call
records go to out/<test>/. To record a run and update the estimate:

    uv run --group sim python analysis/collect_results.py --log "what changed"
    uv run --group sim python analysis/roofline.py

See analysis/RESULTS.md for what each metric means and the log so far.

## Running the ELFs on another simulator

Each ELF runs one kernel call per run and halts. Before a run, write the
inputs into the `arena` symbol (20 KB in DTCM) and the call into
`bench_args` (word 0 is the op, then its arguments; pointers are byte
offsets into the arena, floats are their IEEE-754 bits). After the halt,
`bench_cycles` holds the kernel's cycle count and the outputs are in the
arena. Op numbers and argument order are in bench_main.cc, and the sim
tests show the exact call sequence. For example, one attention head is
vme_attention_block once per K/V block, then attention_flash_finish, with
the running state left in the arena between runs.
