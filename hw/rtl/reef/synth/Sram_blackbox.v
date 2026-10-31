// Black-box stand-in for the vendored Sram.v in synthesis area/timing runs (DC flow).
// Same ports as rtl/vendor/coralnpu/Sram.v (M3); no body. The real ITCM/DTCM become SRAM
// macros in PD, so their area is not part of the std-cell numbers. synth_dc.sh keeps the
// instances (dont_touch) so the logic around them is not optimized away.
module Sram #(
    parameter NUM_ENTRIES = 128,
    parameter GLOBAL_BASE_ADDR = 0
) (
    input                            clock,
    input                            enable,
    input                            write,
    input  [$clog2(NUM_ENTRIES)-1:0] addr,
    input  [                  127:0] wdata,
    input  [                   15:0] wmask,
    output [                  127:0] rdata,
    output                           rvalid
);
endmodule
