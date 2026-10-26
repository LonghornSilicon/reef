/* Memory map for Coral programs: the CoralNPU M3 highmem build, the map
 * reef_perf uses with --highmem.
 *   ITCM: 0x000000 - 0x0FFFFF (1 MB)  code
 *   DTCM: 0x100000 - 0x1FFFFF (1 MB)  host mailbox, data, bss, stack
 * The mailbox (include/reef_kernels/host_if.hpp) comes first in DTCM so the
 * host can find it at a fixed address.
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY {
  ITCM (rx) : ORIGIN = 0x00000000, LENGTH = 1M
  DTCM (rw) : ORIGIN = 0x00100000, LENGTH = 1M
}

SECTIONS {
  .text : {
    KEEP(*(.text.start))
    *(.text*)
  } > ITCM

  .mailbox : { KEEP(*(.mailbox)) } > DTCM
  .rodata  : { *(.rodata*) *(.srodata*) } > DTCM
  .data    : { *(.data*) *(.sdata*) } > DTCM
  .bss : {
    . = ALIGN(4);
    __bss_start = .;
    *(.bss*) *(.sbss*) *(COMMON)
    . = ALIGN(4);
    __bss_end = .;
  } > DTCM

  _stack_top = ORIGIN(DTCM) + LENGTH(DTCM);
}
