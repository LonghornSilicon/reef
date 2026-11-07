`ifndef HDL_VERILOG_RVV_DESIGN_RVV_SVH
`define HDL_VERILOG_RVV_DESIGN_RVV_SVH

`ifndef HDL_VERILOG_RVV_DESIGN_RVV_DEFINE_SVH
`ifndef HDL_VERILOG_RVV_DESIGN_RVV_DEFINE_SVH
`include "rvv_backend_define.svh"
`endif
`endif  // not defined HDL_VERILOG_RVV_DESIGN_RVV_DEFINE_SVH

//
// IF stage, RVS send instruction package to Command Queue
//

// Enum type for SEW. See Table 2 in:
// https://github.com/riscv/riscv-v-spec/blob/master/v-spec.adoc#341-vector-selected-element-width-vsew20
typedef enum logic [2:0] {
  SEW8=0,
  SEW16=1,
  SEW32=2,
  SEW64=3
} RVVSEW;

// Enum type for LMUL. See:
// https://github.com/riscv/riscv-v-spec/blob/master/v-spec.adoc#vector-instruction-formats
typedef enum logic [2:0] {
  LMUL1=0,
  LMUL2=1,
  LMUL4=2,
  LMUL8=3,
  LMULRESERVED=4,
  LMUL1_8=5, // 1/8
  LMUL1_4=6, // 1/4
  LMUL1_2=7  // 1/2
} RVVLMUL;

// Enum type for vtype.vxrm: rounding mode
typedef enum logic [1:0] {
  RNU = 0,
  RNE = 1,
  RDN = 2,
  ROD = 3
} RVVXRM;

// Floating-point Rounding Mode
typedef enum logic [2:0] {
  FRNE=0,
  FRTZ=1,
  FRDN=2,
  FRUP=3,
  FRMM=4,
  FR_UNREACH_5=5,
  FR_UNREACH_6=6,
  FRDYN=7
} RVFRM;

// Floating-point Exception
typedef struct packed {
  logic nv;   
  logic dz; 
  logic of;   
  logic uf;   
  logic nx;   
} RVFEXP_t;

// The architectural configuration state of the RVV core.
typedef struct packed {
  logic                         vill; // This configuration is illegal
  logic [`VL_WIDTH-1:0]         vl;       // Max 128, need one extra bit
  logic [`VSTART_WIDTH-1:0]     vstart;
  logic [`VTYPE_VMA_WIDTH-1:0]  ma;        // 0:inactive element undisturbed, 1:inactive element agnostic
  logic [`VTYPE_VTA_WIDTH-1:0]  ta;        // 0:tail undisturbed, 1:tail agnostic
  RVVXRM                        xrm;       
  logic [`VCSR_VXSAT_WIDTH-1:0] xsat;   // rvv dont need this bit, but output this to rvs
`ifdef ZVE32F_ON
  RVFRM                         frm;
`endif
  RVVSEW                        sew;
  RVVLMUL                       lmul;
  RVVLMUL                       lmul_orig;
} RVVConfigState;

// Enum to encode the major opcode of the instruction. See "Section 5. Vector
// Instruction Formats" of the RVV 1.0 spec.
typedef enum logic [1:0] {
  LOAD=0,
  STORE=1,
  RVV=2
} RVVOpCode;

// A decoded instruction forwarded to the RVVCore from the scalar core.
typedef struct packed {
  logic [`PC_WIDTH-1:0] pc;
  RVVOpCode             opcode;   // effectively bits [6:0] from instruction
  logic [24:0]          bits;     // bits [31:7] from instruction
} RVVInstruction;

// An command internal to the RVVCore. The immediate value of this command has
// been read from the scalar register file if necessary. It also contains
// additional data to track configuration register state (ie: SEW, LMUL, etc).
typedef struct packed {
`ifdef TB_SUPPORT
  logic [`PC_WIDTH-1:0] inst_pc;
`endif
  RVVOpCode             opcode;
  logic [24:0]          bits;
  logic [31:0]          rs1;
  RVVConfigState        arch_state;
} RVVCmd;

//
// DE stage, Command Queue to Uops Queue
//
// Effective MUL enum
typedef enum logic [3:0] {
  EMUL1=0,
  EMUL2=1,
  EMUL4=2,
  EMUL8=3,
  EMUL3,
  EMUL5,
  EMUL6,
  EMUL7,
  EMUL_NONE     // it means this is not supported 
} EMUL_e;

// Effective Element Width
typedef enum logic [2:0] {
  EEW_NONE,    // it means this is not supported 
  EEW1,
  EEW8, 
  EEW16,
  EEW32,
  EEW64
} EEW_e;

// the legal RVVCmd after decoding
typedef struct packed {
  RVVCmd                          cmd;
  EEW_e                           eew_vs1;
  EEW_e                           eew_vs2;
  EEW_e                           eew_vd;
  EEW_e                           eew_max;
  EMUL_e                          emul_vs1;
  EMUL_e                          emul_vs2;
  EMUL_e                          emul_vd;
  EMUL_e                          emul_max;
  logic   [`UOP_INDEX_WIDTH-1:0]  uop_vstart;         
  logic   [`UOP_INDEX_WIDTH-1:0]  uop_index_max;         
  logic   [`VL_WIDTH-1:0]         evl;
  logic                           force_vma_agnostic;
  logic                           force_vta_agnostic;
} LCMD_t;

// execution unit
typedef enum logic [3:0] {
  ALU,
  MUL,
  MAC,
  PMT,
  RDT,
  CMP,
  DIV,
  LSU,
`ifdef ZVE32F_ON
  FMA,
  FCVT,
  FRDT,
  FNCMP,
  FCMP,
  FDIV,
  FTBL,
`endif
  MISC
} EXE_UNIT_e;

// funct3
  parameter  OPIVV=3'b000;      // vs2,      vs1, vd.
  parameter  OPFVV=3'b001;      // vs2,      vs1, vd/rd. float, not support
  parameter  OPMVV=3'b010;      // vs2,      vs1, vd/rd.
  parameter  OPIVI=3'b011;      // vs2, imm[4:0], vd.
  parameter  OPIVX=3'b100;      // vs2,      rs1, vd.
  parameter  OPFVF=3'b101;      // vs2,      rs1, vd. float, not support
  parameter  OPMVX=3'b110;      // vs2,      rs1, vd/rd.
  parameter  OPCFG=3'b111;      // vset* instructions        

// funct6
  // OPI* instructions
  parameter VADD            =   6'b000_000;
  parameter VSUB            =   6'b000_010;
  parameter VRSUB           =   6'b000_011;
  parameter VMINU           =   6'b000_100;
  parameter VMIN            =   6'b000_101;
  parameter VMAXU           =   6'b000_110;
  parameter VMAX            =   6'b000_111;
  parameter VAND            =   6'b001_001;
  parameter VOR             =   6'b001_010;
  parameter VXOR            =   6'b001_011;
  parameter VRGATHER        =   6'b001_100;
  parameter VSLIDEUP_RGATHEREI16    =   6'b001_110;
  parameter VSLIDEDOWN      =   6'b001_111;
  parameter VADC            =   6'b010_000;
  parameter VMADC           =   6'b010_001;
  parameter VSBC            =   6'b010_010;
  parameter VMSBC           =   6'b010_011;
  parameter VMERGE_VMV      =   6'b010_111;     // it could be vmerge or vmv, based on vm field
  parameter VMSEQ           =   6'b011_000;
  parameter VMSNE           =   6'b011_001;
  parameter VMSLTU          =   6'b011_010;
  parameter VMSLT           =   6'b011_011;
  parameter VMSLEU          =   6'b011_100;
  parameter VMSLE           =   6'b011_101;
  parameter VMSGTU          =   6'b011_110;
  parameter VMSGT           =   6'b011_111;
  parameter VSADDU          =   6'b100_000;
  parameter VSADD           =   6'b100_001;
  parameter VSSUBU          =   6'b100_010;
  parameter VSSUB           =   6'b100_011;
  parameter VSLL            =   6'b100_101;
  parameter VSMUL_VMVNRR    =   6'b100_111;     // it could be vsmul or vmv<nr>r, based on vm field
  parameter VSRL            =   6'b101_000;
  parameter VSRA            =   6'b101_001;
  parameter VSSRL           =   6'b101_010;
  parameter VSSRA           =   6'b101_011;
  parameter VNSRL           =   6'b101_100;
  parameter VNSRA           =   6'b101_101;
  parameter VNCLIPU         =   6'b101_110;
  parameter VNCLIP          =   6'b101_111;
  parameter VWREDSUMU       =   6'b110_000;
  parameter VWREDSUM        =   6'b110_001;   

  // OPM* instructions
  parameter VREDSUM         =   6'b000_000;
  parameter VREDAND         =   6'b000_001;
  parameter VREDOR          =   6'b000_010;
  parameter VREDXOR         =   6'b000_011;
  parameter VREDMINU        =   6'b000_100;
  parameter VREDMIN         =   6'b000_101;
  parameter VREDMAXU        =   6'b000_110;
  parameter VREDMAX         =   6'b000_111;
  parameter VAADDU          =   6'b001_000;
  parameter VAADD           =   6'b001_001;
  parameter VASUBU          =   6'b001_010;
  parameter VASUB           =   6'b001_011;
  parameter VSLIDE1UP       =   6'b001_110;
  parameter VSLIDE1DOWN     =   6'b001_111;
  parameter VWRXUNARY0      =   6'b010_000;     
  parameter VXUNARY0        =   6'b010_010;     
  parameter VMUNARY0        =   6'b010_100;     
  parameter VCOMPRESS       =   6'b010_111;
  parameter VMANDN          =   6'b011_000;
  parameter VMAND           =   6'b011_001;
  parameter VMOR            =   6'b011_010;
  parameter VMXOR           =   6'b011_011;
  parameter VMORN           =   6'b011_100;
  parameter VMNAND          =   6'b011_101;
  parameter VMNOR           =   6'b011_110;
  parameter VMXNOR          =   6'b011_111;
  parameter VDIVU           =   6'b100_000;
  parameter VDIV            =   6'b100_001;
  parameter VREMU           =   6'b100_010;
  parameter VREM            =   6'b100_011;
  parameter VMULHU          =   6'b100_100;
  parameter VMUL            =   6'b100_101;
  parameter VMULHSU         =   6'b100_110;
  parameter VMULH           =   6'b100_111;
  parameter VMADD           =   6'b101_001;
  parameter VNMSUB          =   6'b101_011;
  parameter VMACC           =   6'b101_101;
  parameter VNMSAC          =   6'b101_111;
  parameter VWADDU          =   6'b110_000;
  parameter VWADD           =   6'b110_001;
  parameter VWSUBU          =   6'b110_010;
  parameter VWSUB           =   6'b110_011;
  parameter VWADDU_W        =   6'b110_100;
  parameter VWADD_W         =   6'b110_101;
  parameter VWSUBU_W        =   6'b110_110;
  parameter VWSUB_W         =   6'b110_111;
  parameter VWMULU          =   6'b111_000;
  parameter VWMULSU         =   6'b111_010;
  parameter VWMUL           =   6'b111_011;
  parameter VWMACCU         =   6'b111_100;
  parameter VWMACC          =   6'b111_101;
  parameter VWMACCUS        =   6'b111_110;
  parameter VWMACCSU        =   6'b111_111;  

// vwxunary0 and vrxunary0, the uop could be vcpop.m, vfirst.m and vmv. They can be distinguished by vs1 field(inst_encoding[19:15]).
  parameter VMV_X_S         =   5'b00000;
  parameter VCPOP           =   5'b10000;
  parameter VFIRST          =   5'b10001;
  parameter VMV_S_X         =   5'b00000;  // vs2 field

// vxunary0, the uop could be vzext.vf2, vzext.vf4, vsext.vf2, vsext.vf4. They can be distinguished by vs1 field(inst_encoding[19:15]).
  parameter VZEXT_VF4       =   5'b00100;
  parameter VSEXT_VF4       =   5'b00101;
  parameter VZEXT_VF2       =   5'b00110;
  parameter VSEXT_VF2       =   5'b00111;

// vmunary0, the uop could be vmsbf, vmsof, vmsif, viota, vid. They can be distinguished by vs1 field(inst_encoding[19:15]).
  parameter VMSBF           =   5'b00001;
  parameter VMSOF           =   5'b00010;
  parameter VMSIF           =   5'b00011;
  parameter VIOTA           =   5'b10000;
  parameter VID             =   5'b10001;

`ifdef ZVE32F_ON
  // OPF* instructions
  parameter VFADD           =   6'b000_000;
  parameter VFSUB           =   6'b000_010;
  parameter VFRSUB          =   6'b100_111;
  parameter VFMUL           =   6'b100_100;
  parameter VFDIV           =   6'b100_000;
  parameter VFRDIV          =   6'b100_001;
  parameter VFMACC          =   6'b101_100;
  parameter VFNMACC         =   6'b101_101;
  parameter VFMSAC          =   6'b101_110;
  parameter VFNMSAC         =   6'b101_111;
  parameter VFMADD          =   6'b101_000;
  parameter VFNMADD         =   6'b101_001;
  parameter VFMSUB          =   6'b101_010;
  parameter VFNMSUB         =   6'b101_011;
  parameter VFUNARY1        =   6'b010_011;
  parameter VFMIN           =   6'b000_100;
  parameter VFMAX           =   6'b000_110;
  parameter VFSGNJ          =   6'b001_000;
  parameter VFSGNJN         =   6'b001_001;
  parameter VFSGNJX         =   6'b001_010;
  parameter VMFEQ           =   6'b011_000;
  parameter VMFNE           =   6'b011_100;
  parameter VMFLT           =   6'b011_011;
  parameter VMFLE           =   6'b011_001;
  parameter VMFGT           =   6'b011_101;
  parameter VMFGE           =   6'b011_111;
  parameter VFMERGE_VFMV    =   6'b010_111;     // it could be vfmerge or vfmv, based on vm field
  parameter VFUNARY0        =   6'b010_010;
  parameter VFREDOSUM       =   6'b000_011;
  parameter VFREDUSUM       =   6'b000_001;
  parameter VFREDMAX        =   6'b000_111;
  parameter VFREDMIN        =   6'b000_101;
  parameter VWRFUNARY0      =   6'b010_000;
  parameter VFSLIDE1UP      =   6'b001_110;
  parameter VFSLIDE1DOWN    =   6'b001_111;

  // vfunary0. They can be distinguished by vs1 field(inst_encoding[19:15]).
  parameter VFCVT_XUFV      =   5'b00000;
  parameter VFCVT_XFV       =   5'b00001;
  parameter VFCVT_RTZXUFV   =   5'b00110;
  parameter VFCVT_RTZXFV    =   5'b00111;
  parameter VFCVT_FXUV      =   5'b00010;
  parameter VFCVT_FXV       =   5'b00011;
  
  // vfunary1. They can be distinguished by vs1 field(inst_encoding[19:15]).
  parameter VFSQRT          =   5'b00000;
  parameter VFRSQRT7        =   5'b00100;
  parameter VFREC7          =   5'b00101;
  parameter VFCLASS         =   5'b10000;
  
  // vwfunary0 and vrfunary0
  parameter VFMV_F_S        =   5'b00000;
  parameter VFMV_S_F        =   5'b00000;  // vs2 field

  `ifdef ZVFBFWMA_ON
  // OPF* instructions
  parameter VFWMACCBF16     =   6'b111_011; 

  // vfunary0. They can be distinguished by vs1 field(inst_encoding[19:15]).
  parameter VFNCVTBF16      =   5'b11101;
  parameter VFWCVTBF16      =   5'b01101;
  `endif
`endif

// when EXE_UNIT_e is LSU, it identifys what LSU instruction, unit-stride load or indexed store or ..? based on inst_encoding[31:26]
typedef enum logic [1:0] {
  US,         // Unit-Stride
  IU,         // Indexed Unordered
  CS,         // Constant Stride
  IO          // Indexed Ordered
} LSU_MOP_e;

// It identifys what unit-stride instruction when LSU_MOP_e=US, based on inst_encoding[24:20]
typedef enum logic [1:0] {
  US_US,         // Unit-Stride load/store
  US_WR,         // Whole Register load/store
  US_MK,         // MasK load/store, EEW=8(inst_encoding[14:12]=3'b000)
  US_FF          // Faul-only-First load
} LSU_UMOP_e;

// parameter for lsu decoding
  parameter  UNIT_STRIDE       = 3'b000;
  parameter  UNORDERED_INDEX   = 3'b001;
  parameter  CONSTANT_STRIDE   = 3'b010;
  parameter  ORDERED_INDEX     = 3'b011;

  parameter  US_REGULAR        = 5'b00000;
  parameter  US_WHOLE_REGISTER = 5'b01000;
  parameter  US_MASK           = 5'b01011;
  parameter  US_FAULT_FIRST    = 5'b10000;

// It identifys what inst_encoding[11:7] is used for when LSU instruction, based on inst_encoding[5]
typedef enum logic [0:0] {
  IS_LOAD,       
  IS_STORE       
} LSU_IS_STORE_e;

// segment load/store 
typedef enum logic [0:0] {
  IS_SEGMENT,
  NONE      
} LSU_IS_SEG_e;

// combine those signals to LSU_TYPE
typedef struct packed {
  LSU_MOP_e         lsu_mop;
  LSU_UMOP_e        lsu_umop;
  LSU_IS_STORE_e    lsu_is_store;
  LSU_IS_SEG_e      lsu_is_seg;
} LSU_TYPE_t;

// function opcode
typedef union packed {
  logic   [`FUNCT6_WIDTH-1:0]         ari_funct6;
  LSU_TYPE_t                          lsu_funct6;
} FUNCT6_u;

// uop classification used for dispatch rule
typedef enum logic [2:0] {
  XXX=0,
  XXV=1,
  XVX=2,
  XVV=3,
  VXX=4,
  VXV=5,
  VVX=6,
  VVV=7
} UOP_CLASS_e;

// Number of REG
  parameter NREG1 = 3'b000;  
  parameter NREG2 = 3'b001;
  parameter NREG4 = 3'b011;
  parameter NREG8 = 3'b111;

// Number of FIELD
  parameter NF1 = 3'b000;  
  parameter NF2 = 3'b001;
  parameter NF3 = 3'b010;
  parameter NF4 = 3'b011;
  parameter NF5 = 3'b100;
  parameter NF6 = 3'b101;
  parameter NF7 = 3'b110;
  parameter NF8 = 3'b111;

// Destination data struct
`ifdef ZVE32F_ON
typedef enum logic [1:0] {
`else
typedef enum logic [0:0] {
`endif
  VRF,
  XRF
`ifdef ZVE32F_ON
  ,FRF
`endif
} W_DATA_TYPE_e;

// the uop struct stored in Uops Queue
typedef struct packed {
`ifdef TB_SUPPORT
  logic   [`PC_WIDTH-1:0]             uop_pc;
`endif
  logic   [`FUNCT3_WIDTH-1:0]         uop_funct3;
  FUNCT6_u                            uop_funct6;
  EXE_UNIT_e                          uop_exe_unit; 
  UOP_CLASS_e                         uop_class;   
  RVVConfigState                      vector_csr;  
  logic   [`VL_WIDTH-1:0]             vs_evl;             
  logic                               ignore_vma;
  logic                               ignore_vta;
  logic                               force_vma_agnostic; 
  logic                               force_vta_agnostic; 
  logic                               vm;                 
  logic                               v0_valid;           
  logic   [`REGFILE_INDEX_WIDTH-1:0]  dst_index;
  EEW_e                               vd_eew;  
  logic                               vd_valid;
  logic                               vs3_valid;          
  logic                               xd_valid; 
`ifdef ZVE32F_ON
  logic                               fd_valid; 
`endif
  logic   [`REGFILE_INDEX_WIDTH-1:0]  vs1;              
  EEW_e                               vs1_eew;            
  logic                               vs1_valid;
  logic   [`REGFILE_INDEX_WIDTH-1:0]  vs2_index; 	        
  EEW_e                               vs2_eew;
  logic                               vs2_valid;
  logic   [`XLEN-1:0] 	              rs1_data;           
  logic        	                      rs1_data_valid;                                
  logic   [`UOP_INDEX_WIDTH-1:0]      uop_index;          
  logic                               first_uop_valid;    
  logic                               last_uop_valid;     
  logic   [$clog2(`EMUL_MAX)-1:0]     seg_field_index;    
  logic                               pshrob_valid;       
  logic                               pshlsu_valid;
} UOP_QUEUE_t;    

// specify whether the current byte belongs to 'prestart' or 'body-inactive' or 'body-active' or 'tail'
typedef enum logic [1:0] {
  NOT_CHANGE    = 2'b00,      // the byte is not changed, which may belong to 'prestart' or superfluous element in widening/narrowing uop
  TAIL          = 2'b01,      // tail byte
  BODY_INACTIVE = 2'b10,      // body-inactive byte
  BODY_ACTIVE   = 2'b11       // body-active byte
} BYTE_TYPE_e;

// trap handle
typedef enum logic [1:0] {
  TRAP_LSU,           
  TRAP_LSU_FF       
} TRAP_INFO_e;

// the max number of byte in a vector register is VLENB
typedef BYTE_TYPE_e [`VLENB-1:0]      BYTE_TYPE_t;

typedef struct packed {
`ifdef TB_SUPPORT
  logic   [`PC_WIDTH-1:0]             uop_pc;
`endif
  logic   [`ROB_DEPTH_WIDTH-1:0]      rob_entry;
  FUNCT6_u                            uop_funct6;  
  logic   [`FUNCT3_WIDTH-1:0]         uop_funct3;
  logic                               is_cmp;
  logic   [`VSTART_WIDTH-1:0]         vstart;
  logic   [`VL_WIDTH-1:0]             vl;       
  logic                               vm;               
  RVVXRM                              vxrm;       
  logic   [`VLEN-1:0]                 v0_data;
  logic                               v0_data_valid;
  logic   [`VLEN-1:0]                 vd_data;
  logic                               vd_data_valid;
  EEW_e                               vd_eew;  
  logic   [`REGFILE_INDEX_WIDTH-1:0]  vs1;              
  logic   [`VLEN-1:0]                 vs1_data;           
  logic                               vs1_data_valid; 
  logic        	                      rs1_data_valid;                                   
  logic   [`VLEN-1:0]                 vs2_data;	        
  logic                               vs2_data_valid;  
  EEW_e                               vs2_eew;
  logic                               first_uop_valid;     
  logic                               last_uop_valid;
  logic   [`UOP_INDEX_WIDTH_ALU-1:0]  uop_index;      
} ALU_RS_t;    

// DIV reservation station struct
typedef struct packed {
`ifdef TB_SUPPORT
  logic   [`PC_WIDTH-1:0]             uop_pc;
`endif
  logic   [`ROB_DEPTH_WIDTH-1:0]      rob_entry;
  FUNCT6_u                            uop_funct6;
  logic   [`FUNCT3_WIDTH-1:0]         uop_funct3;
  logic                               is_div;
  logic   [`VLEN-1:0]                 vs1_data;           
  logic                               vs1_data_valid; 
  logic        	                      rs1_data_valid;      
  logic   [`VLEN-1:0]                 vs2_data;	        
  logic                               vs2_data_valid;  
  EEW_e                               vs2_eew;
`ifdef ZVE32F_ON
  RVFRM                               frm;
  logic   [`REGFILE_INDEX_WIDTH-1:0]  vs1;              
`endif
} DIV_RS_t; 

// DIV reservation station struct
typedef struct packed {
`ifdef TB_SUPPORT
  logic   [`PC_WIDTH-1:0]             uop_pc;
`endif
  logic   [`ROB_DEPTH_WIDTH-1:0]      rob_entry;
  FUNCT6_u                            uop_funct6;
  logic   [`FUNCT3_WIDTH-1:0]         uop_funct3;
  logic   [`VLEN-1:0]                 vs1_data;           
  logic                               vs1_data_valid; 
  logic        	                      rs1_data_valid;      
  logic   [`VLEN-1:0]                 vs2_data;	        
  logic                               vs2_data_valid;  
  EEW_e                               vs2_eew;
`ifdef ZVE32F_ON
  RVFRM                               frm;
  logic   [`REGFILE_INDEX_WIDTH-1:0]  vs1;              
`endif
} DIV_SUB_t;

// MUL and MAC reservation station struct
typedef struct packed {   
`ifdef TB_SUPPORT
  logic   [`PC_WIDTH-1:0]             uop_pc;
`endif
  logic   [`ROB_DEPTH_WIDTH-1:0]      rob_entry;
  FUNCT6_u                            uop_funct6;
  logic   [`FUNCT3_WIDTH-1:0]         uop_funct3;
  RVVXRM                              vxrm;       
  logic   [`VLEN-1:0]                 vs1_data;           
  logic                               vs1_data_valid; 
  logic          	                    rs1_data_valid;   
  logic   [`VLEN-1:0]                 vs2_data;	        
  logic                               vs2_data_valid; 
  EEW_e                               vs2_eew; 
  logic   [`VLEN-1:0]                 vs3_data;	
  logic                               vs3_data_valid; 
  logic                               uop_index;
} MUL_RS_t;    

// PMT and RDT reservation station struct
typedef struct packed {   
`ifdef TB_SUPPORT
  logic   [`PC_WIDTH-1:0]             uop_pc;
`endif
  logic   [`ROB_DEPTH_WIDTH-1:0]      rob_entry;
  EXE_UNIT_e                          uop_exe_unit; 
  FUNCT6_u                            uop_funct6;
  logic   [`FUNCT3_WIDTH-1:0]         uop_funct3;
  logic   [`VL_WIDTH-1:0]             vl;       
  logic                               vm;
  logic   [`VL_WIDTH-1:0]             vlmax;       
  logic   [`VLEN-1:0]                 v0_data;
  EEW_e                               vs1_eew;
  logic   [`VLEN-1:0]                 vs1_data;          
  logic                               vs1_data_valid; 
  EEW_e                               vs2_eew;
  logic   [`VLEN-1:0]                 vs2_data;	        
  BYTE_TYPE_t                         vs2_type;
  logic   [`REGFILE_INDEX_WIDTH-1:0]  vs2_index;
  EEW_e                               vd_eew;  
  logic   [`REGFILE_INDEX_WIDTH-1:0]  dst_index;
  logic   [`XLEN-1:0] 	              rs1_data;         
  logic                               first_uop_valid;     
  logic                               last_uop_valid;     
  logic   [`UOP_INDEX_WIDTH_ALU-1:0]  uop_index;      
`ifdef ZVE32F_ON
  RVFRM                               frm;
`endif
} PMT_RDT_RS_t;    

// LSU reservation station struct
typedef struct packed {   
`ifdef TB_SUPPORT
  logic   [`PC_WIDTH-1:0]             uop_pc;
`endif

  logic                               vidx_valid; 
  logic   [`REGFILE_INDEX_WIDTH-1:0]  vidx_addr;
  logic   [`VLEN-1:0]                 vidx_data;            // vs2        
  logic                               vregfile_read_valid; 
  logic   [`REGFILE_INDEX_WIDTH-1:0]  vregfile_read_addr;
  logic   [`VLEN-1:0]                 vregfile_read_data;   // vs3       
  logic                               v0_valid;
  logic   [`VLENB-1:0]                v0_data;              // byte strobe signal for mask load/store. 
                                                            // v0[i]=1 means vd/vs3[8*i +: 8] data is valid. 
} UOP_RVV2LSU_t;    

 // FMA reservation station struct
typedef struct packed {   
`ifdef TB_SUPPORT
  logic   [`PC_WIDTH-1:0]             uop_pc;
`endif
  logic   [`ROB_DEPTH_WIDTH-1:0]      rob_entry;
  FUNCT6_u                            uop_funct6;
  logic   [`FUNCT3_WIDTH-1:0]         uop_funct3;
  EXE_UNIT_e                          uop_exe_unit; 
  logic   [`VSTART_WIDTH-1:0]         vstart;
  logic   [`VL_WIDTH-1:0]             vl;       
  logic                               vm;               
  RVFRM                               frm;
  logic   [`VLENW*`EMUL_MAX-1:0]      v0_data;
  logic                               v0_data_valid;
  logic   [`REGFILE_INDEX_WIDTH-1:0]  vs1;              
  logic   [`VLEN-1:0]                 vs1_data;           
  logic                               vs1_data_valid; 
  logic   [`VLEN-1:0]                 vs2_data;	        
  logic                               vs2_data_valid; 
  EEW_e                               vs2_eew; 
  logic   [`VLEN-1:0]                 vs3_data;	
  logic                               vs3_data_valid; 
  logic   [`XLEN-1:0] 	              rs1_data;          
  logic          	                    rs1_data_valid;   
  logic                               last_uop_valid;
  logic   [`UOP_INDEX_WIDTH_ALU-1:0]  uop_index;      
} FMA_RS_t;  

//
// EX stage, 
//
// send PU's result to ROB
typedef struct packed {
`ifdef TB_SUPPORT
  logic     [`PC_WIDTH-1:0]           uop_pc;
`endif
  logic     [`ROB_DEPTH_WIDTH-1:0]    rob_entry;
  logic     [`VLEN-1:0]               w_data;             // when w_type=XRF, w_data[`XLEN-1:0] will store the scalar result
  logic                               w_valid;
  logic     [`VLENB-1:0]              vsaturate;
`ifdef ZVE32F_ON
  RVFEXP_t  [`VLENB-1:0]              fpexp;
`endif
} PU2ROB_t;  

// lsu uop info to remap rob_entry for UOP_LSU2RVV_t
typedef struct packed {   
`ifdef TB_SUPPORT
  logic   [`PC_WIDTH-1:0]             uop_pc;
`endif
  logic                               valid;
  logic   [`ROB_DEPTH_WIDTH-1:0]      rob_entry;
  LSU_IS_STORE_e                      lsu_class; 
  logic	[`REGFILE_INDEX_WIDTH-1:0] 	  vregfile_write_addr;  
} LSU_MAP_INFO_t; 

// LSU feedback to RVV
typedef struct packed {   
`ifdef TB_SUPPORT
  // To trace wave
  logic   [`PC_WIDTH-1:0]             uop_pc;
  logic   [`UOP_INDEX_WIDTH-1:0]      uop_index;         
`endif
  // For load data
  logic                               vregfile_write_valid;
  logic	[`REGFILE_INDEX_WIDTH-1:0] 	  vregfile_write_addr;  
  logic	[`VLEN-1:0] 			          	vregfile_write_data;  	// vd   
  // Store done signal to help ROB retire the store uop
  logic                               lsu_vstore_last;
} UOP_LSU2RVV_t;  

typedef struct packed {   
  UOP_LSU2RVV_t                       uop_lsu2rvv;
  logic                               trap_valid;
} UOP_LSU_t;  

typedef struct packed {
`ifdef TB_SUPPORT
  logic   [`PC_WIDTH-1:0]             uop_pc;
`endif
  logic                               w_valid;            // write valid
  logic [`VLEN-1:0]                   w_data;             // write data; w_data[`XLEN-1:0] is scalar result if write type is XRF
  logic [`VLENB-1:0]                  vsaturate;
`ifdef ZVE32F_ON
  RVFEXP_t  [`VLENB-1:0]              fpexp;
`endif
} RES_ROB_t;

// send uop to ROB
typedef struct packed {
`ifdef TB_SUPPORT
  logic   [`PC_WIDTH-1:0]             uop_pc;
`endif
  logic   [`REGFILE_INDEX_WIDTH-1:0]  w_index;            //wr addr
  W_DATA_TYPE_e                       w_type;             //write type: 0 for VRF, 1 for XRF
  BYTE_TYPE_t                         byte_type;          //wr Byte mask
  RVVConfigState                      vector_csr;         //Receive Vstart, vlen,... And need to update vcsr when trap
  logic                               last_uop_valid;
} DP2ROB_t;

// send ROB info to DP
typedef struct packed {
`ifdef TB_SUPPORT
  logic   [`PC_WIDTH-1:0]             uop_pc;
`endif
  logic                               valid;              //entry valid
  logic                               w_valid;            //vd valid
  logic   [`REGFILE_INDEX_WIDTH-1:0]  w_index;            //vd addr
  W_DATA_TYPE_e                       w_type;             //write type: 0 for VRF, 1 for XRF
  logic   [`VLEN-1:0]                 w_data;             //when w_type=XRF, w_data[`XLEN-1:0] will store the scalar result
  BYTE_TYPE_t                         byte_type;          //wr Byte mask
  RVVConfigState                      vector_csr;         //Receive Vstart, vlen,... And need to update vcsr when trap
} ROB2DP_t;

typedef struct packed {
`ifdef TB_SUPPORT
  logic   [`PC_WIDTH-1:0]             uop_pc;
  logic                               last_uop_valid;
`endif
  logic                               w_valid;            //entry valid
  logic   [`REGFILE_INDEX_WIDTH-1:0]  w_index;            //wr addr
  logic   [`VLEN-1:0]                 w_data;             //when w_type=XRF, w_data[`XLEN-1:0] will store the scalar result
  W_DATA_TYPE_e                       w_type;             //to VRF or XRF
  BYTE_TYPE_t                         vd_type;            //wr Byte mask
  logic                               trap_flag;          //whether this entry in a trap
  RVVConfigState                      vector_csr;         //Receive Vstart, vlen,... And need to update vcsr when trap
  logic   [`VLENB-1:0]                vxsaturate;         //Update saturation bit
`ifdef ZVE32F_ON
  RVFEXP_t  [`VLENB-1:0]              fpexp;
`endif
} ROB2RT_t;  

// the rob struct stored in ROB
typedef struct packed {
`ifdef TB_SUPPORT
  logic   [`PC_WIDTH-1:0]             uop_pc;
`endif
  logic                               valid;              // entry valid
  DP2ROB_t                            uop_info;           // Uop information
  RES_ROB_t                           uop_res;            // Uop result
  logic                               uop_done;           // Uop is finished.
  logic                               trap;
} ROB_t;

//
// Retire stage
//
// write back to XRF/FRF
typedef struct packed {
`ifdef TB_SUPPORT
  logic   [`PC_WIDTH-1:0]             uop_pc;
`endif
  logic   [`REGFILE_INDEX_WIDTH-1:0]  rt_index; 
  logic   [`XLEN-1:0]                 rt_data; 
}RT2RVS_t;

// write back to VRF
typedef struct packed {
`ifdef TB_SUPPORT
  logic   [`PC_WIDTH-1:0]             uop_pc;
`endif
  logic   [`REGFILE_INDEX_WIDTH-1:0]  rt_index; 
  logic   [`VLEN-1:0]                 rt_data;
  logic   [`VLENB-1:0]                rt_strobe; 
}RT2VRF_t;

`endif  // HDL_VERILOG_RVV_DESIGN_RVV_SVH
