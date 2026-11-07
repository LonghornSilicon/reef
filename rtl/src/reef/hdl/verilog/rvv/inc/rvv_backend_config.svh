`ifndef RVV_CONFIG_SVH
`define RVV_CONFIG_SVH

// config for multi-dispatch
`define DISPATCH3
//`define DISPATCH2

// FP ISA 
//`define ZVE32F_ON
//`define ZVFBFWMA_ON

// LSU interaction
// Disable until scalar side supports NOHANDSHAKE
// `define UNMK_USCS_LOAD_NOHANDSHAKE

// ARBITER
`define ARBITER_ON

`endif // RVV_CONFIG_SVH
