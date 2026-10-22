// sky130 TCM (synthesis). Use instead of ${CORALNPU_RTL}/Sram.v, e.g.
//   grep -v '/Sram.v$' $CORALNPU_RTL/synth.f > synth_sky130.f
//   ... -f synth_sky130.f -f $SRAM_RTL/sky130.f
// and link the macro's .lib/.db (sky130_sram_2kbyte_1rw1r_32x512_8). See rtl/sram/README.md.
${SRAM_RTL}/Sram_sky130.sv
