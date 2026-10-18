# Coral NPU Simulation Kernels

Hand-written kernels for the Coral NPU, checked on the Coral simulator. Each
kernel uses the vector unit (RVV) or the matrix engine (VME), and the tests
compare it with a scalar version for correctness and cycle count.

The code builds freestanding for `riscv32-unknown-elf` with
`-march=rv32imf_zve32x -mabi=ilp32f`. It has no C++ standard library, so only
the C headers (`<stddef.h>`, `<stdint.h>`) are available.

## Run everything

From `inference-engine/`, one command builds every kernel, runs it on the
host and on Coral's VME RTL model, checks it and writes the results:

```sh
uv run --group sim python src/csrc/src/simulations/run_all.py
uv run --group sim python src/csrc/src/simulations/analysis/roofline.py
```

- Host: every unit test (`clang++ -DVME_EMULATE -DRVV_EMULATE`) and the
  embedding timing.
- Sim: Docker image `coralnpu-reef` (built from coralnpu's
  `utils/coralnpu.dockerfile`) and volume `reef-coralnpu` (a checkout of
  coralnpu at the commit pinned in `config/hardware.json`, plus the Bazel
  cache). `~/Projects/coralnpu` (override with `CORALNPU=`) is only read.
  The benches run with cocotb on `VmeCoreMiniAxi` under Verilator, with
  coralnpu's behavioral SRAM model. **The first run builds the RTL model
  and takes hours** under amd64 emulation on Apple Silicon; later runs
  reuse the cache.
- `--skip-sim` reuses `results/sim_raw/` from the last sim run.
- Outputs: `results/results.{json,csv}`, `analysis/roofline.json`,
  `analysis/roofline_{prefill,decode}.png`. The write-up is
  `analysis/REPORT.md`.

## Layout

```
simulations/
├── config/
│   ├── model.json       # GPT-2 small dims, dtypes, quant scales, tiles
│   ├── hardware.json    # hardware facts: value, units, source, confirmed
│   └── gen_params.py    # writes includes/params.h from the two JSON files
├── includes/
│   ├── params.h         # GENERATED: namespace params, kCamelCase constants
│   ├── quant_helpers.h  # namespace quant: requantization + newlib math decls
│   ├── rvv_helpers.h    # namespace rvv: RVV wrappers (+ -DRVV_EMULATE model)
│   └── vme_helpers.h    # namespace vme: VME (Zvt) wrappers (+ -DVME_EMULATE)
├── kernels/
│   ├── matmul.h/.cc            # the one matmul entry point (VME or RVV)
│   ├── matmul_rvv.h/.cc        # matmul on the vector unit
│   ├── matmul_vme.h/.cc        # matmul on the matrix engine
│   ├── online_softmax.h/.cc    # int32 scores -> uint8 probabilities
│   ├── attention_serial.h/.cc  # unfused attention head, the reference
│   ├── attention_flash.h/.cc   # fused attention head (online softmax)
│   ├── attention_flash_chunk.h/.cc # flash with KV streamed in blocks
│   ├── attention_vme.h/.cc     # block flash, both matmuls on the VME
│   ├── gelu.h/.cc              # 256-entry lookup table
│   ├── layernorm.h/.cc
│   └── residual.h/.cc
├── unit_tests/          # one host test per kernel family
├── host/embedding_bench.cc  # host-side token + position embedding, timed
├── sim/                 # coralnpu Bazel package: bench ELFs + cocotb bench
├── run_all.py           # the one command (see above)
├── analysis/            # roofline.py, REPORT.md, plots
├── results/             # results.json/.csv, raw sim records
└── integration_tests/   # test_main.cc (stub)
```

## Kernels

All matrices are row-major. The entry points are `extern "C"` and sit outside
any namespace, so tests and the simulator harness can call them by plain name.
The helpers they are built on live in the `rvv` and `vme` namespaces.

### `matmul_rvv` (vector unit)

```c
void matmul_rvv(size_t n, size_t k, size_t m, const uint8_t* a,
                const uint8_t* b, uint32_t* c);
```

`C[n x m] (uint32) = A[n x k] (uint8) * B[k x m] (uint8)`.

- Works for any `n`, `k`, `m`. It walks across C in strips of `vl` columns
  (at most 16 with VLEN = 128). For each row it accumulates one row of B at a
  time with a widening multiply-add.
- `k` must stay below 66,051 or the uint32 sums can overflow
  (255 × 255 × k < 2³²).

### `matmul_vme_u8` and `matmul_vme_s8u8` (matrix engine)

```c
void matmul_vme_u8(size_t n, size_t k, size_t m, const uint8_t* a,
                   const uint8_t* b, int32_t* c);   // uint8 A (vtmmu.tvv)
void matmul_vme_s8u8(size_t n, size_t k, size_t m, const int8_t* a,
                     const uint8_t* b, int32_t* c); // int8 A (vtmms.tvv)
```

`C[n x m] (int32) = A[n x k] (8-bit) * B[k x m] (uint8)`. B is always uint8,
because VME has no int8 × int8 matmul.

- Requires `n % 16 == 0`, `m % 16 == 0` and `k % 4 == 0`, which GPT-2 sizes
  meet.
- Each 16×16 block of C is held in matrix tile 0 while the kernel walks along
  `k`. Each step consumes 4 slices of k (16 × 16 × 4 = 1,024 multiply-adds),
  then the finished block is copied to C one row at a time.

### `attention_serial` and `attention_flash` (scalar)

```c
void attention_serial(size_t nq, size_t nk, size_t d, size_t query_start,
                      size_t window, const int8_t* q, const uint8_t* kt,
                      int32_t k_zero, const uint8_t* v, int32_t v_zero,
                      float score_scale, float v_scale, float out_scale,
                      int8_t* out);
void attention_flash(...);  // identical signature
```

One attention head: `out[nq x d] (int8) = softmax(q . kt) * v`, row-major.
K arrives transposed (`kt[d x nk]`) so it matches the uint8 B operand the
VME matmul wants; `k_zero`/`v_zero` are the uint8 zero points and the three
scales carry the quantization. We run GPT-2, so `score_scale` includes its
1/sqrt(d_head) (`params::kScoreScale`) and `window >= nk` gives the full
causal prefix; a smaller `window` still works as a sliding window. Causal and
sliding-window masking happen through loop bounds — query row `i` sees keys
`[max(0, query_start + i + 1 - window), query_start + i]` — so there are no
mask buffers or sentinel scores. `window >= nk` disables the local limit;
`window` must be at least 1.

- `attention_serial` materializes one int32 score row at a time, turns it
  into uint8 probabilities with `online_softmax`, and does P·V in int32
  with a `v_zero * sum(p)` lift (the probabilities sum to ~255, not
  exactly 255).
- `attention_flash` fuses everything into one pass per query row with the
  online-softmax recurrence (running max, denominator and weighted sum), so
  it never materializes a score row and its per-row state is O(d).
- Shape limits (documented, not checked, like the matmuls): `d <= 128`,
  `nk <= 1024`, `query_start + nq <= nk`.
- Internals use scalar float registers (`rv32imf`); only int8/uint8 data
  touches memory. `expf` comes from newlib and is declared directly in
  `quant_helpers.h`, since the freestanding build has no `<math.h>`.
- The scalar kernels stay as the reference; `attention_vme_block` (below)
  is the matrix-engine version and is what the end-to-end estimate uses.

### `matmul` (dispatcher)

```c
void matmul(size_t n, size_t k, size_t m, const uint8_t* a, bool signed_a,
            const uint8_t* b, int32_t* c);
```

The one entry point attention and the MLP call. VME when `n % 16`, `m % 16`
and `k % 4` are all 0, RVV otherwise (decode's `n = 1`). It does not pad to
VME's shape: that needs a 16 × k scratch copy in DTCM and wastes 15 of 16
rows. `matmul_rvv_s8u8` is the signed RVV fallback for the score matmul.

### `online_softmax`

```c
void online_softmax(const int32_t* scores, size_t lo, size_t hi,
                    float score_scale, uint8_t* probs);
```

One int32 score row (the q·kt accumulator after the `k_zero * q_sum` fixup),
visible slice `[lo, hi]`, to uint8 probabilities with scale 1/255 and zero
point 0, the A operand `matmul_vme_u8` takes for P·V. Pass 1 keeps the
running max and normalizer, pass 2 normalizes.

### `attention_vme_block`

```c
void attention_vme_block(size_t nq, size_t d, size_t query_start,
                         size_t window, size_t kv_start, size_t kv_len,
                         const int8_t* q, const uint8_t* kt, int32_t k_zero,
                         const uint8_t* v, float score_scale,
                         int32_t* scratch, float* m, float* l, float* acc);
```

Block flash attention with both matmuls through `matmul()`: S = q·kt
(int8 × uint8, `matmul_vme_s8u8`), then per row the `k_zero` fixup, causal
mask, running-max update and uint8 probabilities, then P·V (uint8 × uint8,
`matmul_vme_u8`) folded into the same `(m, l, acc)` state as
`attention_flash_chunk`, finished by `attention_flash_finish`. Prefill's
16-query blocks run on the matrix engine; decode's single query falls back
to RVV inside `matmul()`, which measured faster than a padded VME tile.
`scratch` holds the int32 scores / P·V and the uint8 P. Matches
`attention_serial` within one int8 step.

### `gelu`, `layernorm`, `residual_add`

- `gelu_table` builds a 256-entry int8 table of GPT-2's tanh GeLU from the
  input and output scales; `gelu` is then a pure lookup, exact for int8.
- `layernorm`: exact int32 row sums, float mean/variance/1/sqrt, eps 1e-5,
  int8 gamma and beta with their own scales.
- `residual_add`: dequantize both inputs, add, requantize.

### `attention_flash_chunk` and `attention_flash_finish`

One head's K and V at n_ctx = 1024 are 64 KB each, twice the 32 KB DTCM.
The chunked variant takes one KV block at absolute offset `kv_start` plus
the per-row running state (`m`, `l`, `acc`) in and out, so the host
streams blocks through DTCM; `attention_flash_finish` normalizes and
requantizes. Zero `l` and `acc` before the first block. It matches
`attention_serial` within one int8 step. `attention_flash` is unchanged.

## Helpers

### `quant_helpers.h`

The newlib `expf`, `tanhf` and `sqrtf` declarations and `quant::to_int8`
(round to nearest, ties away from zero, clamped), shared by the kernels.

### `rvv_helpers.h`

Type aliases (`rvv::vu8`, `rvv::vu16`, `rvv::vu32`, and the signed
`vi16`/`vi32`) and small always-inline wrappers (`set_vl`, `zeros`,
`load_widen`, `mac`, `store` and their `_s` variants), so the element
widths and register-group sizes are chosen in one place. Build with
`-DRVV_EMULATE` for a plain C model of each wrapper (16 lanes), like
`-DVME_EMULATE` below.

### `vme_helpers.h`

The toolchain doesn't know the VME instructions yet, so each wrapper emits
the raw instruction encoding. The encodings follow Coral's own tests
(`google-coral/coralnpu`: `tests/cocotb/vme_test/`).

- **Setup:** `enable_matrix_state()`, `config_tile_moves()`,
  `config_int8_matmul()`, `set_tk()`.
- **Tile operations:** `zero_tile<TILE>()`, `matmul_step<TILE, SIGNED_A>()`
  and `store_tile_row<TILE>()`.
- **Fixed registers:** VME instructions read fixed vector registers (A in
  v8–v14, B in v16–v22, row moves write v4–v7). That's why each load and its
  matmul share one asm block.
- **vtype/vl:** the config wrappers overwrite `vtype` and `vl`. Don't mix them
  with RVV intrinsics in the same function without calling `vsetvl` again.

#### Emulation mode

Build with `-DVME_EMULATE` to replace every VME wrapper with a plain C model.
The VME kernel then runs on any machine, which tests the tiling and indexing
logic but not the instruction encodings. For example, on macOS:

```sh
clang++ -std=c++17 -DVME_EMULATE -Ikernels -Iincludes \
    kernels/matmul_vme.cc your_test.cc -o vme_test
```

## Editor and lint setup

- **clangd:** use Homebrew's (`/opt/homebrew/opt/llvm/bin/clangd`). Apple's
  `/usr/bin/clangd` doesn't ship `riscv_vector.h`, so it reports the RVV types
  as unknown. clangd reads `compile_flags.txt`, and the `-I` paths there are
  relative to this folder.
- **clang-tidy:** `.clang-tidy` inherits the inference-engine rules with two
  exceptions:
  - `modernize-deprecated-headers` is off, because `<cstdint>` and friends
    don't exist without a C++ standard library.
  - `__riscv_*` names are ignored by the reserved-identifier and naming
    checks. Clang declares the RVV intrinsics where they're first used, so the
    checks would otherwise blame the calling file for those names.
- To lint by hand from `inference-engine/` (run both modes, because the VME
  emulation code is only checked with `-DVME_EMULATE`):

  ```sh
  S=src/csrc/src/simulations
  uv run clang-format --dry-run --Werror $S/kernels/* $S/includes/*
  uv run clang-tidy --quiet -p $S $S/kernels/* $S/includes/*
  uv run clang-tidy --quiet -p $S --extra-arg=-DVME_EMULATE $S/kernels/* $S/includes/*
  ```

  `tools/lint.sh` and `tools/check_cpp.sh` do not cover this folder yet.

## Verification status

Sim = coralnpu `VmeCoreMiniAxi` RTL under Verilator (cocotb), pinned commit
in `config/hardware.json`. Numbers live in `results/results.json`.

| What | Status |
| --- | --- |
| Lint (clang-tidy, clang-format, ruff) | Clean, in real and emulation modes (the only tidy finding left is `main` linkage under `-ffreestanding`, as before) |
| Host unit tests (`-DVME_EMULATE -DRVV_EMULATE`) | All pass: matmul (every variant, incl. non-multiple-of-16 shapes), softmax, GeLU, LayerNorm, residual, serial/flash/chunked attention |
| VME encodings and `tk = 4` int8 steps | **Verified on the sim**: matmul_vme_u8/s8u8 bit-exact vs NumPy; coralnpu's own `vme_matmul_int8_test` passes on the same build |
| `matmul_rvv`, `matmul_rvv_s8u8`, `matmul()` dispatch | **Verified on the sim**: bit-exact vs NumPy (decode, prefill and 7×13×23 shapes) |
| softmax, GeLU, LayerNorm, residual | **Verified on the sim**: within 0.5 int8 LSB of float references |
| Attention (serial, flash, chunked flash) | **Verified on the sim**: serial within 0.9 LSB of a float reference; flash and chunked flash (16- and 64-key streamed blocks) within 1 LSB of serial |
| Attention on the VME (`attention_vme_block`) | **Verified on the sim**: within 1 LSB of serial at 64 keys, 2 LSB at 256 (uint8 probabilities rounded per block); 1.35 LSB vs float at 256 keys, where the serial algorithm itself is 1.44. About 337k cycles per 16-query × 64-key block, versus ~2.2M for scalar |
| `mstatus.MS` enable (bit 29) | Works on the sim: the VME benches call `enable_matrix_state()` and run |

## Known issues and open questions

1. **The VME kernel doesn't check its shape rules.** If `n` or `m` isn't a
   multiple of 16, or `k` isn't a multiple of 4, it silently reads and writes
   past the arrays. `matmul()` only routes valid shapes to it.
2. **The RVV types must stay in step.** `rvv::set_vl` uses 8-bit elements in
   one register (e8m1), and the accumulator uses 32-bit elements in four
   registers (e32m4). Both give the same `vl`. If you change one, change the
   other.
3. **uint8 probabilities lose mass on long rows.** On a flat 1024-key row
   the average probability is 0.25 LSB, so many round to 0 (row sum 246 of
   255 in `test_softmax`). Attention lifts `v_zero` with the actual sum, but
   long-context accuracy should be checked on real activations.
4. **Quantization scales are placeholders** (`config/model.json`) until
   calibrated on real GPT-2 activations.
5. **Only the behavioral SRAM model.** coralnpu's Verilator flow has no
   `sram22` macro option.
6. **`test_main.cc` is still a stub.** A real integration test (one GPT-2
   block chained from these kernels) is the next step.
