// Reef smoke test startup: stack, gp, FS/VS on, zero .bss, call main.
// End like M3's crt (toolchain/crt/coralnpu_start.S): mpause = clean halt (io_halted only),
// ebreak = failure (io_halted + io_fault). A trap records mcause and ends with ebreak.
  .section .start, "ax"
  .global _start
_start:
  la   sp, __stack_top
.option push
.option norelax
  la   gp, __global_pointer$
.option pop
  la   t0, trap
  csrw mtvec, t0
  li   t0, 0x6600          // mstatus.FS and .VS = dirty
  csrs mstatus, t0
  la   a0, __bss_start
  la   a1, __bss_end
1:bgeu a0, a1, 2f
  sw   zero, 0(a0)
  addi a0, a0, 4
  j    1b
2:call main
  .word 0x08000073         // mpause
3:j    3b

  .balign 4
trap:
  la   t0, smoke_result
  csrr t1, mcause
  sw   t1, 12(t0)          // smoke_result.trap_cause
  li   t1, 0xBAD00BAD
  sw   t1, 0(t0)           // smoke_result.magic
  ebreak
