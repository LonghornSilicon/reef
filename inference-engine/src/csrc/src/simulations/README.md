# Coral NPU Simulation Kernels

Hand-written kernels for the Coral NPU, checked on the Coral simulator. Each
kernel uses the vector unit (RVV) or the matrix engine (VME), and the tests
compare it with a scalar version for correctness and cycle count.

The code builds freestanding for `riscv32-unknown-elf` with
`-march=rv32imf_zve32x -mabi=ilp32f`. It has no C++ standard library, so only
the C headers (`<stddef.h>`, `<stdint.h>`) are available.

## Layout

```
simulations/
├── .clang-tidy          # repo lint rules, plus the exceptions below
├── compile_flags.txt    # target flags and include paths for clangd/clang-tidy
├── includes/
│   ├── rvv_helpers.h    # namespace rvv: wrappers over RVV intrinsics
│   └── vme_helpers.h    # namespace vme: wrappers over VME (Zvt) instructions
├── kernels/
│   ├── matmul_rvv.h/.cc # matmul on the vector unit
│   └── matmul_vme.h/.cc # matmul on the matrix engine
├── unit_tests/          # one test per kernel (stubs for now)
└── integration_tests/   # test_main.cc (stub for now)
```

`kernels/` and `includes/` are on the include path, so any file includes a
header by its bare name: `#include "matmul_rvv.h"`, `#include "vme_helpers.h"`.

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

## Helpers

### `rvv_helpers.h`

Type aliases (`rvv::vu8`, `rvv::vu16`, `rvv::vu32`) and small always-inline
wrappers (`set_vl`, `zeros`, `load_widen`, `mac`, `store`), so the element
widths and register-group sizes are chosen in one place.

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

| What | Status |
| --- | --- |
| Lint (clang-tidy, clang-format, clangd) | Clean, in real and emulation modes |
| RISC-V compile (`-O2 -Wall -Wextra -Wpedantic`) | Both kernels build with no warnings; the VME asm assembles |
| VME tiling logic | Matches a scalar reference under `-DVME_EMULATE` for 16×4×16, 32×8×48 and 64×768×32, signed and unsigned |
| `matmul_rvv` results | **Not run yet.** No RISC-V emulator was available, so the first real run will be on the Coral sim |
| VME encodings on hardware | **Not checked.** Only Coral's RTL model can run them |

## Known issues and open questions

1. **No simulator build.** Nothing cross-compiles this folder or runs it on
   the Coral sim yet (see the TODO in `inference-engine/CMakeLists.txt`).
   `compile_flags.txt` only configures the editor and lint, so the future build
   (CMake toolchain file or Coral's Bazel rules) must pass the same flags,
   including `-std=c++17`, `-ffreestanding`, `-nostdinc++`, `-Ikernels` and
   `-Iincludes`.
2. **The VME kernel doesn't check its shape rules.** If `n` or `m` isn't a
   multiple of 16, or `k` isn't a multiple of 4, it silently reads and writes
   past the arrays. A check needs a way to report failure that works without a
   C library, so the harness has to decide how to handle it.
3. **Nothing calls `vme::enable_matrix_state()`.** Matrix instructions trap as
   illegal (mcause 2) until matrix state is on, and Coral's startup code
   doesn't turn it on. The test harness should call it once before any VME
   kernel.
4. **The output types differ.** `matmul_rvv` writes `uint32_t` and the VME
   kernels write `int32_t`, so `test_matmul` needs a cast to compare them.
5. **The RVV types must stay in step.** `rvv::set_vl` uses 8-bit elements in
   one register (e8m1), and the accumulator uses 32-bit elements in four
   registers (e32m4). Both give the same `vl`. If you change one, change the
   other, or `vl` will mean different things in the two places.
6. **Unverified VME details.** The instruction encodings and the
   `mstatus.MS` bit (bit 29 in `enable_matrix_state`) have only been checked to
   assemble. Emulation mode can't test them.
7. **The tests are stubs.** `unit_tests/` and `integration_tests/` hold only
   comments. Planned coverage:
   - `test_matmul`: RVV and VME against a scalar reference, plus cycle counts.
     Include an `m > 16` case for `matmul_rvv`, since that's where its
     column-strip loop matters.
   - `test_softmax`: online softmax accuracy and cycles.
   - `test_flash`: flash attention against serial attention.
8. **`test_main.cc` is a test runner rather than an integration test.** Each
   ELF usually runs as its own program on the simulator. Either build one
   binary per test, or turn this file into a real integration test, such as an
   attention block built from the matmul, softmax and flash kernels.
