# Shared startup/exit macros for perf-model workloads.

# START: entry point. Sets up the stack and enables the FP and vector units
# (mstatus.FS = Initial, mstatus.VS = Initial).
.macro START
  .section .text.start, "ax"
  .globl _start
_start:
  la    sp, _stack_top
  li    t0, (1 << 13) | (1 << 9)
  csrs  mstatus, t0
.endm

# END: stop the program. mpause halts both MPACT and the CoralNPU RTL testbench.
.macro END
  .word 0x08000073
.endm

# Region-of-interest markers. They read mcycle into s10/s11 so the RTL cycle
# count of the loop can be measured (RTL: s11 - s10). The perf model currently
# reports whole-program cycles; see the ROI ticket.
.macro ROI_BEGIN
  csrr  s10, mcycle
.endm
.macro ROI_END
  csrr  s11, mcycle
.endm
