/* Reef smoke test: M3 RvvCoreMiniAxi memory map (ITCM 8 KB @0, DTCM 32 KB @0x10000). */
MEMORY {
  ITCM (rx) : ORIGIN = 0x00000000, LENGTH = 8K
  DTCM (rw) : ORIGIN = 0x00010000, LENGTH = 32K
}
ENTRY(_start)
SECTIONS {
  .text : ALIGN(16) { *(.start) *(.text) *(.text.*) *(.rodata) *(.rodata.*) *(.srodata*) . = ALIGN(16); } > ITCM
  /* Only the ITCM image is loaded over SPI, so initialized data is not allowed. */
  .data : { *(.data) *(.data.*) *(.sdata) *(.sdata.*) } > DTCM
  ASSERT(SIZEOF(.data) == 0, "smoke test must not use initialized data")
  .bss (NOLOAD) : ALIGN(16) { __bss_start = .; *(.sbss*) *(.bss*) *(COMMON) . = ALIGN(16); __bss_end = .; } > DTCM
  __stack_top = ORIGIN(DTCM) + LENGTH(DTCM);
  __global_pointer$ = __bss_start + 0x800;
}
