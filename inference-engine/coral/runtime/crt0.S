// Start-up for every Coral program: benchmarks, tests and firmware.
// There is no C runtime. Set the stack, turn on the FPU and vector unit,
// round to nearest even (the host reference's mode), clear .bss, run main,
// then stop with mpause, which halts the M3 core and ends reef_perf.

    .section .text.start, "ax"
    .globl _start
_start:
    la    sp, _stack_top
    li    t0, (1 << 13) | (1 << 9)  // mstatus.FS and mstatus.VS = Initial
    csrs  mstatus, t0
    csrwi frm, 0                    // round to nearest, ties to even
    la    a0, __bss_start
    la    a1, __bss_end
1:  bgeu  a0, a1, 2f
    sw    zero, 0(a0)
    addi  a0, a0, 4
    j     1b
2:  call  main
    .word 0x08000073                // mpause
