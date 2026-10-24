# SRAM22 sky130 macro model

| Field | Value |
|---|---|
| Upstream | https://github.com/ucb-substrate/sram22_sky130_macros (generator: https://github.com/ucb-substrate/sram22) |
| Commit | `de0a9a7cdabf941c807a5ab35864f80d97fccfea` |
| Macro | `sram22_512x128m4w8`: 512 words x 128 bits, column mux 4, 8-bit write mask |
| License | BSD-3-Clause (`LICENSE.sram22`) |
| Silicon | listed by upstream as taped out and measured working at 1.8 V, 25 MHz |

Only the behavioral Verilog model is in the repo (`sram22_512x128m4w8.v`, unmodified,
sha256 `f6ad924d2ce42c71617b02ac2931cd3dabbb8db09a7fe27acf2807cac28002db`).
Lint uses it in place of the macro. PD takes the Liberty (ss/tt/ff), LEF and GDS from the
upstream repo at the same commit.
