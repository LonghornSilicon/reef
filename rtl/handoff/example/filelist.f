// Shape only. A real export lists rtl/<path> per line, compile order, one
// compilation unit. Grouped the way the arch team's v0 spec reads the M3 core.
// +incdir+rtl/<vector backend headers>
// packages and headers
// front-end: pc update, address decode (tcm or axi), itcm port, instruction
//   buffer (8 instr), partial decode and static branch predict, decode, dispatch
// back-end: scoreboard, alu x4, mlu, dvu, lsu (4-entry queue + 1 slot), fpu,
//   bru x4, integer and fp regfiles, retirement buffer (8), csr
// vector: command queue, vector scoreboard, vector rob, vrf (32 x 128b),
//   functional units: alu, mul/mac, div, fma, permute/reduce
// memory and interconnect: itcm, dtcm, tcm arbiters (core > axi slave > debug),
//   axi slave bridge, axi master bridges (ebus id 0, fetch id 1), control regs
// top: RvvCoreMiniAxi
