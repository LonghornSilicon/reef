/* Memory map for Coral programs on the CoralNPU M3 as built. This is the
 * default reef_perf map (no --highmem):
 *   ITCM: 0x00000 - 0x01FFF (8 KB)   code
 *   DTCM: 0x10000 - 0x17FFF (32 KB)  host mailbox, data, bss, stack
 * link.ld is the 1 MB / 1 MB highmem map. build.sh uses this map when
 * CORAL_MAP=m3.
 * The mailbox (include/reef_kernels/host_if.hpp) is first in DTCM, so the
 * host finds it at a fixed address.
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY {
  ITCM (rx) : ORIGIN = 0x00000000, LENGTH = 8K
  DTCM (rw) : ORIGIN = 0x00010000, LENGTH = 32K
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
