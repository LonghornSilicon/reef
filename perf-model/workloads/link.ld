/* Linker script for perf-model workloads.
 * Matches the CoralNPU M3 *default* memory map, so the same ELF runs on the
 * perf model (MPACT) and on the M3 RTL simulator (core_mini / rvv_core_mini):
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
