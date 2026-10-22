# reef_top Drop 0 build with SRAM22 TCMs (sky130)

The PD Drop 0 build (`configs/filelists/reef_top_sky130.f`) after the TCM swap from 20 OpenRAM macros
to 5 SRAM22 `sram22_512x128m4w8` (`rtl/sram/Sram_sram22.sv`). The OpenRAM build is in
`synth_reef_top_sky130_node.md`.

## Verification (2026-10-10)

| Check | Tool / where | Result |
|---|---|---|
| `scripts/lint_vendor.sh`, `scripts/lint_reef.sh` (both TCM wrappers `-Wall`, both sky130 reef_top lists, filelist staleness) | Verilator 5.032: WSL (M3 Bazel cache) and the `verilator/verilator:v5.032` image | PASS |
| TCM unit test `scripts/run_sram_sky130_tb.sh`, models sram22 / openram / cycle | VCS T-2022.06, koopa, `858a991` | 3/3 PASS, each 7,553 (ITCM) + 6,017 (DTCM) checks, 0 errors |
| Same test, `SIM=verilator MODELS="sram22 cycle"` | Verilator 5.032 image | PASS, same check counts |
| reef_top smoke test, `NODE=sky130 scripts/run_reef_smoke.sh` | Verilator 5.052 + gcc 15.2, koopa, `858a991` | 3/3 PASS. Result words and cycle counts identical to `reef_smoke.md` (267,845 / 534,430 / 284,300 cycles; `main()` 1,655 cycles). Model built from `Sram_sram22.sv` + SRAM22's model (checked in `Vreef_top__ver.d`) |
| CI dry run of `.github/workflows/rtl.yml` (team repo layout, `hw/`) | GitHub Actions, private staging copy, commit `9029d08` | rtl-lint PASS, rtl-synth PASS |
| Re-run of the three checks above on a fresh clone of `8dcefb3` (the exported tree; RTL, tests and filelists equal to `858a991` except a filelist comment line) | daisy: VCS T-2022.06 (system gcc 8); Verilator 5.052 (gcc-toolset-14); Yosys from OSS CAD Suite 2026-10-01 | TCM tb VCS sram22 / openram / cycle and Verilator sram22 / cycle: ALL PASS, same check counts. Smoke 3/3 PASS, same cycle counts. Yosys 687 s, `check_synth.py` PASS: 7.242 mm^2, 5 x `sram22_512x128m4w8`, 2 x `sdlclkp_4` |

## Yosys (area only)

**Run:** koopa, `scripts/synth_yosys.sh --lib sky130hd --top reef_top --node sky130` at `b58f97c`
(Yosys + yosys-slang, OSS CAD Suite 2026-10-01; std cells `sky130_fd_sc_hd__tt_025C_1v80`, macro
`.lib` SRAM22 tt). 734 s, 7.4 GB peak RSS, exit 0. The CI run on GitHub gave the same netlist
statistics (654 s, 6.9 GB).

`scripts/check_synth.py --expect sram22_512x128m4w8=5 --expect sky130_fd_sc_hd__sdlclkp_4=2 --max-area-mm2 8.0`:

| | |
|---|---|
| Std cells | 902,705 instances, **7.242 mm^2** (1.684 mm^2 sequential) |
| `sram22_512x128m4w8` | 5 |
| `sky130_fd_sc_hd__sdlclkp_4` | 2 |
| Other cell types (black boxes, unmapped cells) | none |
| Result | PASS |

| Block | Area (um^2) | % |
|---|---:|---:|
| RVV vector unit | 6,007,558 | 82.9 |
| Scalar core (no FPU) | 939,766 | 13.0 |
| Scalar FPU | 129,020 | 1.8 |
| AXI / fabric | 84,141 | 1.2 |
| Reef shell (SPI bridge, DECERR sink, reset sync) | 36,030 | 0.5 |
| TCM logic (macros excluded) | 16,671 | 0.2 |
| Top-level CSR / clock / reset / misc | 16,302 | 0.2 |
| Debug module | 12,950 | 0.2 |
| **Total** | **7,242,437** | 100 |

"TCM logic" is 16,671 um^2 against 24,734 with the OpenRAM wrapper: the SRAM22 wrapper has no
128-bit hold register.

## Design Compiler + PrimeTime at ss, 20 ns

**Run:** koopa, `scripts/synth_dc.sh --top reef_top --node sky130 --period 20 --db
sky130_fd_sc_hd__ss_100C_1v60.db --sram-db sram22_512x128m4w8_ss_100C_1v60.db` at `858a991`
(DC T-2022.03-SP2, 4958 s, exit 0; 0 errors; `REEF: 5 SRAM macros`, `REEF: 2 ICG cells`), then
`scripts/sta_pt.sh --top reef_top --period 20 --extra-db sram22_512x128m4w8_ss_100C_1v60.db` on the
netlist (PT T-2022.03-SP5, `test_en_i` = 0). Same constraints as the OpenRAM run (5% uncertainty, IO
20%, wire-load model, ideal clock). The macro `.db` comes from SRAM22's ss .lib via `lc_shell`
(0 errors; warnings are missing `internal_power` groups on data pins).

| | SRAM22 build | OpenRAM build (10/4) |
|---|---|---|
| Std cells | **5.063 mm^2** (combinational 3.689, of which buf/inv 0.346; sequential 1.374) | 5.070 mm^2 |
| Macros | 5 x `sram22_512x128m4w8` = **2.821 mm^2** | 20 x OpenRAM = 5.691 mm^2 |
| **Total cell area** | **7.884 mm^2** (before utilization and halo) | 10.761 mm^2 |
| Macro timing | SRAM22 ss 100C 1.60 V | OpenRAM TT only |
| Setup (PT) | **met**, 0 violating endpoints (recovery included); worst slack +0.000006 ns | met, +0.0002 ns |
| Macro -> core | +0.029 ns: ITCM `g_bank[0].u_macro` -> `core/score/fetch/ctrl/pc_bits_reg[30]` (full cycle; SRAM22 launches on the rising edge) | +0.007 ns (half cycle, falling edge) |
| Core -> macro | +0.019 ns: `score/lsu/readFired_bits_lineAddr_reg[18]` -> ITCM `din[100]` | +0.08 ns |
| Hold (PT, ideal clock, pre-CTS) | worst -0.67 ns, 20,713 reg->reg endpoints (for P&R to fix) | - |

DC sizes every path to about zero slack, so the slack values say "met", not "margin". With SRAM22
the TCM paths are timed against the macro's own slow-corner model, which the OpenRAM run could not do.
Not done: P&R, CTS, hold, gate-level simulation or LEC of this netlist.
