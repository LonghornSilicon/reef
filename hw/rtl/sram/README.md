# TCM SRAMs

M3's core has two memories behind the `Sram` module (`rtl/vendor/coralnpu/Sram.v`): ITCM 512x128
(8 KB) and DTCM 2048x128 (32 KB). Both are single port, with a 1-cycle synchronous read and a
16-bit byte write mask. Every other memory-like structure in the core (vector register file,
scalar register file, FIFOs, retirement buffer) stays flip-flops: they need several ports and
same-cycle reads, and they are too small for a macro to save area.

## sky130 (Drop 0): `Sram_sram22.sv` (SRAM22, 5 macros)

Built from SRAM22's `sram22_512x128m4w8` (512 x 128 bit, column mux 4, 8-bit write mask), a
pre-built macro from [ucb-substrate/sram22_sky130_macros](https://github.com/ucb-substrate/sram22_sky130_macros)
(BSD-3-Clause, pinned in `sram22/UPSTREAM.md`). Upstream lists this macro as taped out and working
in silicon at 1.8 V / 25 MHz. Alan Schwartz's sky130 one-pager (9/30) picked it over OpenRAM: 5 macros
instead of 20, and routed with timing closed in his flow.

| TCM | Macros | Arrangement | Macro area |
|---|---:|---|---:|
| ITCM 512x128 | 1 | 1 macro | 0.56 mm^2 |
| DTCM 2048x128 | 4 | 4 banks (`addr[10:9]`), 4:1 read mux | 2.26 mm^2 |
| **Total** | **5** | | **2.82 mm^2** (564,231 um^2 each, before halo) |

- The macro has the same interface as M3's `Sram`: active-high `ce`/`we`, `wmask[i]` enables byte
  `i`, inputs captured on the rising edge, `dout` launched from the rising edge and held by the
  macro's output latch until the next read. So the wrapper is only a bank decode and a read mux on
  the bank read last; there is no hold register.
- As with the TSMC/GF macros in the vendored `Sram.v`, a write to the address last read does not
  change `rdata`.
- `.lib` corners: ss 100C 1.60 V, tt 25C 1.80 V, ff -40C 1.95 V (SRAM22 characterization). ss:
  clk -> dout about 7.2 ns, minimum period 5.79 ns.

**rstb.** The macro's active-low `rstb` resets its input flops and self-timed control latches; it does
not clear the array or `dout` (SRAM22 docs). The generated core instantiates `Sram` with a clock
only, so the wrapper ties `rstb` high. UCB's STAC test chip drove `rstb` from chip reset, so a
tied-high `rstb` is not the configuration measured in silicon. Open item: confirm with SRAM22's
authors / Alan, or route the core reset to the macros (needs a reset port on the generated TCM
modules, i.e. a patch to vendored SV).

**dout drive.** Alan's one-pager: SRAM22 `dout` is weak; keep the load/store and fetch logic next to
the macros, or buffer at each macro (a register would add a cycle and break the 1-cycle contract).

### Files

- `Sram_sram22.sv`: the wrapper (`Sram` module, same parameters and ports as the vendored one).
- `sram22/sram22_512x128m4w8.v`: SRAM22's behavioral model, unmodified (works in Verilator and VCS).
- `sram22.f` / `sram22_sim.f`: synthesis / simulation filelists.
- `.lib` and LEF (and GDS with `--gds`): `scripts/fetch_sram22.sh` (pinned commit, sha256-checked) into
  `build/sram22/`. Not committed.

## Filelist swap

The vendored `Sram.v` is not edited. sky130 builds drop it from the filelist and add a wrapper:

```bash
grep -v '/Sram.v$' $CORALNPU_RTL/synth.f > synth_sky130.f      # core without the generic Sram
# synthesis:  -f synth_sky130.f -f $SRAM_RTL/sram22.f  + link the sram22_512x128m4w8 .lib/.db
# Verilator:  -f sim_sky130.f   -f $SRAM_RTL/sram22_sim.f   (sim.f filtered the same way)
```

`configs/filelists/reef_top_sky130{,_sim}.f` (PD Drop 0), `scripts/synth_dc.sh --sram-db <macro .db>`,
`NODE=sky130 scripts/run_reef_smoke.sh`, `scripts/synth_yosys.sh --node sky130` and
`scripts/lint_reef.sh` already do this.

### Tests (both wrappers)

`scripts/run_sram_sky130_tb.sh`: `tests/sram_sky130/tb_sram_sky130.sv` on both TCM shapes, for
`sram22` (Sram_sram22 on SRAM22's model), `openram` (Sram_sky130 on OpenRAM's model) and `cycle`
(Sram_sky130 on our cycle model). VCS by default; `SIM=verilator MODELS="sram22 cycle"` runs in
Verilator 5.032 (CI). It checks read data, hold across idle cycles and writes, byte masks and
`rvalid` every cycle 1 ns before the rising edge.

## Fallback: `Sram_sky130.sv` (OpenRAM, 20 macros)

Built from the open PDK's OpenRAM macro `sky130_sram_2kbyte_1rw1r_32x512_8` (512 x 32, 8-bit
write mask, 683.1 x 416.5 um = 0.285 mm^2). It uses the pre-built macro from
`sky130A/libs.ref/sky130_sram_macros`; we do not run the OpenRAM compiler ourselves.

| TCM | Macros | Arrangement | Macro area |
|---|---:|---|---:|
| ITCM 512x128 | 4 | 4 side by side (32 bits each) | 1.14 mm^2 |
| DTCM 2048x128 | 16 | 4 banks (`addr[10:9]`) x 4 wide, 4:1 read mux | 4.55 mm^2 |
| **Total** | **20** | | **5.69 mm^2** (before halo) |

- `wmask[4c+3:4c]` drives macro column `c` (bytes map one to one). Only the RW port is used;
  the read-only port has `csb1` tied high and `clk1` on the same clock.
- **Read data launch on the falling edge** (`.lib` `timing_type : falling_edge`) and are only
  valid until just after the next rising edge. The wrapper therefore holds the last read word in a
  128-bit register and muxes: `rdata = rd_q ? macro_dout : hold_q`. The core sees the same
  1-cycle read latency and held `rdata` as with upstream's macros. The path from the macro to the
  first core flop has **half a clock period**.
- A write to the address last read does not change `rdata` (upstream's generic behavioral model
  does; the TSMC/GF macro branches in `Sram.v` don't).
- Synthesis area for the wrapper logic is small (hold register, bank-select flops, read mux).

Builds select it with `TCM=openram` (`run_reef_smoke.sh`, `synth_yosys.sh`), an OpenRAM `.db` in
`synth_dc.sh --sram-db`, or `configs/filelists/reef_top_sky130_openram{,_sim}.f`.

### Models

- `sim/sky130_sram_2kbyte_1rw1r_32x512_8_cycle.v`: our 2-state cycle model of the macro for
  Verilator. OpenRAM's own model uses `#` delays inside its clock blocks and races without
  `--timing`. Ours drives a poison value (`0xDEAD0BAD`) whenever the real macro's output would be X,
  so a design that relies on held macro output fails.
- OpenRAM's model (`sky130A/libs.ref/sky130_sram_macros/verilog/`) is used by the VCS unit test.

### Caveats (open)

- **The ITCM -> instruction fetch path uses the whole half cycle** (DC sizes it to 0 slack at ss/20 ns
  and tt/10 ns; `results/sram_sky130.md`). Any extra macro clock-to-output delay comes out of it.
- The macro `.lib` is **TT 1.8 V 25 C only**, generated by OpenRAM's characterizer, not measured.
  Slow-corner runs pair ss standard cells with the TT macro, so their SRAM paths are optimistic.
- No MBIST: memory test is functional over SPI (`docs/dft-v0.md`). In scan mode the macros are
  not bypassed.
- PD: 20 macros need halo, power hookup (`vccd1`/`vssd1`) and placement. The LEF and GDS are in
  the same PDK directory.
