# UVM Adder Demo

A small, self-contained example of design verification with UVM.

- `design.sv`: parameterized adder (`ADD_NUM` lanes of `ADD_WIDTH` bits, with carry in/out)
- `testbench.sv`: UVM testbench that drives the adder as 1 lane x 8 bits

## Testbench structure

`adder_if` (interface) -> `adder_item` (randomized transaction, biased toward 0/255 corner values)
-> `adder_seq` -> `adder_sequencer` / `adder_driver` / `adder_monitor` (inside `adder_agent`)
-> `adder_scoreboard` (checks `{cout, sum} == a + b + cin`) -> `adder_env` -> `adder_test`

## Running

Easiest: [EDA Playground](https://www.edaplayground.com/). Paste `design.sv` into the Design pane
and `testbench.sv` into the Testbench pane, select UVM 1.2 and a SystemVerilog simulator, then run.
Any scoreboard mismatch is reported as a `UVM_ERROR`.
