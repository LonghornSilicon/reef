// D4 check: how many pipeline registers does each tool give fpnew for FP32?
//
// CoralNPU's FloatCore (float/FloatCore.scala:112-119) builds fpnew with
//   PipeRegs: '{default: 'd3}
// PipeRegs is fmt_unsigned_t [0:3], and fmt_unsigned_t is a PACKED
// logic [0:4][31:0] (one 160-bit value per op group, one 32-bit slice per format).
// Intended: 3 for every format. Verilator 5.032/5.052 instead put the integer 3
// into each 160-bit row, so only slice [4] (FP16ALT) gets 3 and FP32 [0] gets 0.
//
//   A = what FloatCore emits             B = the same, written row by row
//
// Simulators: top_sim prints the values. Elaboration-only tools: top_elab
// prints them as $info. Format index: 0=FP32 1=FP64 2=FP16 3=FP8 4=FP16ALT.
//
// Run in Verilator (prints "A: 0 0 0 0 3" in our runs; intended is "3 3 3 3 3"):
//   $ verilator --binary -Wno-fatal -Wno-WIDTH --top-module top_sim \
//     <path>/fpnew_pkg.sv pipe_regs_check.sv && ./obj_dir/Vtop_sim
// (Comments here never start with the tool's name: it reads those as directives.)
// slang (prints "A ... 3 3 3 3 3" as notes):
//   python3 -m pip install pyslang   # in a venv
//   python3 -c "import sys;from pyslang import ast,syntax,DiagnosticEngine as D;c=ast.Compilation();[c.addSyntaxTree(syntax.SyntaxTree.fromFile(f)) for f in sys.argv[1:]];print(D.reportAll(c.sourceManager,c.getAllDiagnostics()))" <path>/fpnew_pkg.sv pipe_regs_check.sv
// In Reef (PR #15 branch rtl/hw-drop0) the package is hw/rtl/vendor/coralnpu/fpnew_pkg.sv.

package d4_check;
  localparam fpnew_pkg::fpu_implementation_t A = '{
    PipeRegs:   '{default: 'd3},
    UnitTypes:  '{'{default: fpnew_pkg::PARALLEL}, '{default: fpnew_pkg::MERGED},
                  '{default: fpnew_pkg::PARALLEL}, '{default: fpnew_pkg::MERGED}},
    PipeConfig: fpnew_pkg::DISTRIBUTED
  };
  localparam fpnew_pkg::fpu_implementation_t B = '{
    PipeRegs:   '{'{default: 'd3}, '{default: 'd3}, '{default: 'd3}, '{default: 'd3}},
    UnitTypes:  '{'{default: fpnew_pkg::PARALLEL}, '{default: fpnew_pkg::MERGED},
                  '{default: fpnew_pkg::PARALLEL}, '{default: fpnew_pkg::MERGED}},
    PipeConfig: fpnew_pkg::DISTRIBUTED
  };
endpackage

module top_sim;
  import d4_check::*;
  initial begin
    for (int g = 0; g < 4; g++)
      $display("opgrp %0d  A: %0d %0d %0d %0d %0d   B: %0d %0d %0d %0d %0d", g,
        A.PipeRegs[g][0], A.PipeRegs[g][1], A.PipeRegs[g][2], A.PipeRegs[g][3], A.PipeRegs[g][4],
        B.PipeRegs[g][0], B.PipeRegs[g][1], B.PipeRegs[g][2], B.PipeRegs[g][3], B.PipeRegs[g][4]);
    $finish;
  end
endmodule

module top_elab;
  import d4_check::*;
  $info("A (as emitted) opgrp0: %0d %0d %0d %0d %0d",
        A.PipeRegs[0][0], A.PipeRegs[0][1], A.PipeRegs[0][2], A.PipeRegs[0][3], A.PipeRegs[0][4]);
  $info("B (row by row) opgrp0: %0d %0d %0d %0d %0d",
        B.PipeRegs[0][0], B.PipeRegs[0][1], B.PipeRegs[0][2], B.PipeRegs[0][3], B.PipeRegs[0][4]);
endmodule
