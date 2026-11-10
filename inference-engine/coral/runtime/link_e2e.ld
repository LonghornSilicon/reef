/* The M3 as built: 8 KB ITCM (code only), 32 KB DTCM (mailbox, data, bss,
 * stack). Weights and the KV cache live in XSPI flash at 0x20000000
 * (npu.hpp); they are not part of this image. */
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
  ASSERT(_stack_top - __bss_end >= 1024, "DTCM: less than 1 KB left for the stack")
}
