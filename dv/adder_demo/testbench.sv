// UVM testbench for adder.sv  -- paste into the TESTBENCH pane (design.sv goes in the DESIGN pane)
`include "uvm_macros.svh"
import uvm_pkg::*;

// ---------- 1. INTERFACE: the bundle of wires between testbench and adder ----------
interface adder_if(input bit clk);
  logic [7:0] a, b, sum;
  logic       cin, cout;
  logic       valid;       // 1 while the driver is presenting a real input
endinterface

// ---------- 2. ITEM: one "transaction" = one set of inputs (+ the answer we saw) ----------
class adder_item extends uvm_sequence_item;
  rand bit [7:0] a, b;
  rand bit       cin;
  bit [7:0]      sum;      // filled by the monitor (not random)
  bit            cout;

  // make corner values (0 and 255) show up often
  constraint corners_c {
    a dist {8'h00 := 5, 8'hFF := 5, [8'h01:8'hFE] :/ 90};
    b dist {8'h00 := 5, 8'hFF := 5, [8'h01:8'hFE] :/ 90};
  }

  `uvm_object_utils(adder_item)
  function new(string name = "adder_item"); super.new(name); endfunction

  function string convert2string();
    return $sformatf("a=%0d b=%0d cin=%0d -> sum=%0d cout=%0d", a, b, cin, sum, cout);
  endfunction
endclass

// ---------- 3. SEQUENCE: the "recipe" that creates items ----------
class adder_seq extends uvm_sequence #(adder_item);
  `uvm_object_utils(adder_seq)
  function new(string name = "adder_seq"); super.new(name); endfunction

  task body();
    repeat (50) begin
      adder_item it = adder_item::type_id::create("it");
      start_item(it);
      if (!it.randomize()) `uvm_error("SEQ", "randomize failed")
      finish_item(it);
    end
  endtask
endclass

// ---------- 4. DRIVER: turns an item into wire wiggles ----------
class adder_driver extends uvm_driver #(adder_item);
  `uvm_component_utils(adder_driver)
  virtual adder_if vif;
  function new(string name, uvm_component parent); super.new(name, parent); endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual adder_if)::get(this, "", "vif", vif))
      `uvm_fatal("DRV", "no virtual interface")
  endfunction

  task run_phase(uvm_phase phase);
    vif.valid = 0;
    forever begin
      seq_item_port.get_next_item(req);
      @(posedge vif.clk);
      vif.a = req.a;  vif.b = req.b;  vif.cin = req.cin;  vif.valid = 1;
      @(posedge vif.clk);
      vif.valid = 0;
      seq_item_port.item_done();
    end
  endtask
endclass

// ---------- 5. MONITOR: watches the wires and reports what it saw ----------
class adder_monitor extends uvm_monitor;
  `uvm_component_utils(adder_monitor)
  virtual adder_if vif;
  uvm_analysis_port #(adder_item) ap;
  function new(string name, uvm_component parent); super.new(name, parent); endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    ap = new("ap", this);
    if (!uvm_config_db#(virtual adder_if)::get(this, "", "vif", vif))
      `uvm_fatal("MON", "no virtual interface")
  endfunction

  task run_phase(uvm_phase phase);
    forever begin
      @(negedge vif.clk);              // inputs have settled, adder output is ready
      if (vif.valid) begin
        adder_item it = adder_item::type_id::create("seen");
        it.a = vif.a;  it.b = vif.b;  it.cin = vif.cin;
        it.sum = vif.sum;  it.cout = vif.cout;
        ap.write(it);
      end
    end
  endtask
endclass

// ---------- 6. SCOREBOARD: the checker. Computes the right answer, compares ----------
class adder_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(adder_scoreboard)
  uvm_analysis_imp #(adder_item, adder_scoreboard) imp;
  int checked, errors;
  function new(string name, uvm_component parent); super.new(name, parent); endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    imp = new("imp", this);
  endfunction

  function void write(adder_item it);
    bit [8:0] expected = it.a + it.b + it.cin;     // 9 bits so the carry is not lost
    checked++;
    if ({it.cout, it.sum} !== expected) begin
      errors++;
      `uvm_error("SCB", $sformatf("MISMATCH %s  expected=%0d", it.convert2string(), expected))
    end else
      `uvm_info("SCB", $sformatf("ok  %s", it.convert2string()), UVM_MEDIUM)
  endfunction

  function void report_phase(uvm_phase phase);
    `uvm_info("SCB", $sformatf("checked=%0d errors=%0d  --> %s", checked, errors,
                               (errors == 0 && checked > 0) ? "PASS" : "FAIL"), UVM_NONE)
  endfunction
endclass

// ---------- 6b. SEQUENCER: our own, empty on purpose ----------
// The stock uvm_sequencer already does the job. Extending it gives us our own named class
// to add things to later (shared settings, arbitration rules, ...).
class adder_sequencer extends uvm_sequencer #(adder_item);
  `uvm_component_utils(adder_sequencer)
  function new(string name, uvm_component parent); super.new(name, parent); endfunction
endclass

// ---------- 7. AGENT: bundles sequencer + driver + monitor ----------
class adder_agent extends uvm_agent;
  `uvm_component_utils(adder_agent)
  adder_sequencer sqr;
  adder_driver   drv;
  adder_monitor  mon;
  function new(string name, uvm_component parent); super.new(name, parent); endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    sqr = adder_sequencer::type_id::create("sqr", this);
    drv = adder_driver ::type_id::create("drv", this);
    mon = adder_monitor::type_id::create("mon", this);
  endfunction

  function void connect_phase(uvm_phase phase);
    drv.seq_item_port.connect(sqr.seq_item_export);
  endfunction
endclass

// ---------- 8. ENV: agent + scoreboard, wired together ----------
class adder_env extends uvm_env;
  `uvm_component_utils(adder_env)
  adder_agent      agent;
  adder_scoreboard scb;
  function new(string name, uvm_component parent); super.new(name, parent); endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    agent = adder_agent     ::type_id::create("agent", this);
    scb   = adder_scoreboard::type_id::create("scb", this);
  endfunction

  function void connect_phase(uvm_phase phase);
    agent.mon.ap.connect(scb.imp);
  endfunction
endclass

// ---------- 9. TEST: builds the env and runs the sequence ----------
class adder_test extends uvm_test;
  `uvm_component_utils(adder_test)
  adder_env env;
  function new(string name, uvm_component parent); super.new(name, parent); endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    env = adder_env::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    adder_seq seq = adder_seq::type_id::create("seq");
    phase.raise_objection(this);        // "don't end the simulation yet"
    seq.start(env.agent.sqr);
    repeat (3) @(posedge env.agent.drv.vif.clk);   // let the last item reach the scoreboard
    phase.drop_objection(this);
  endtask
endclass

// ---------- 10. TOP: clock, interface, the adder itself, start UVM ----------
module top;
  bit clk;
  always #5 clk = ~clk;

  adder_if aif(clk);
  adder #(.ADD_NUM(1), .ADD_WIDTH(8)) dut (
    .a(aif.a), .b(aif.b), .cin(aif.cin), .sum(aif.sum), .cout(aif.cout));

  initial begin
    uvm_config_db#(virtual adder_if)::set(null, "*", "vif", aif);
    run_test("adder_test");
  end
endmodule