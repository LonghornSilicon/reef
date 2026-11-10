# Prebuilt ELFs

Built with Coral's GCC (`toolchain_coralnpu_v2`) from this commit.

| Folder | Built by | Memory map |
|---|---|---|
| `tests/`, `bench/`, `firmware/` | `coral/tools/build.sh` (tests need `build/coral/golden`, see `test.sh`) | highmem: 1 MB ITCM, 1 MB DTCM (`runtime/link.ld`); run with `reef_perf --highmem` |
| `e2e/` | `coral/tools/build_e2e.sh` | M3 as built: 8 KB ITCM, 32 KB DTCM (`runtime/link_e2e.ld`) |

`e2e/` holds TinyStories-28M: `e2e_ts28m.elf` (16-token prompt, 2 new
tokens) and `e2e_ts28m_decode_ctx<N>.elf` (one decode step at context N).
How to run them: `coral/kernels/e2e/README.md`.
