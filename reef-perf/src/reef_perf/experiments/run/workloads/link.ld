/* Linker script for reef-perf workloads.
 * Matches Reef's M3 baseline *default* memory map (the CoralNPU M3 default),
 * so the same ELF runs on reef_perf (Spike) and on the M3 RTL simulator
 * (core_mini / rvv_core_mini):
 *   ITCM: 0x00000 - 0x01FFF (8 KB)   code
 *   DTCM: 0x10000 - 0x17FFF (32 KB)  data, bss, stack
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY {
  ITCM (rx) : ORIGIN = 0x00000000, LENGTH = 8K
  DTCM (rw) : ORIGIN = 0x00010000, LENGTH = 32K
}

SECTIONS {
  .text : {
    *(.text.start)
    *(.text*)
  } > ITCM

  .rodata : { *(.rodata*) } > DTCM
  .data   : { *(.data*) }   > DTCM
  .bss    : { *(.bss*) *(COMMON) } > DTCM

  _stack_top = ORIGIN(DTCM) + LENGTH(DTCM);
}
