
`ifndef HDL_VERILOG_RVV_DESIGN_RVV_SVH
`include "rvv_backend.svh"
`endif
`ifndef RVV_ASSERT__SVH
`include "rvv_backend_sva.svh"
`endif

module rvv_backend_decode_unit_lsu_de2
(
  lcmd_valid,
  lcmd,
  uop_index_remain,
  uop_valid,
  uop
);
//
// interface signals
//
  input   logic                               lcmd_valid;
  input   LCMD_t                              lcmd;

  input   logic       [`UOP_INDEX_WIDTH-1:0]  uop_index_remain;
  output  logic       [`NUM_DE_UOP-1:0]       uop_valid;
  output  UOP_QUEUE_t [`NUM_DE_UOP-1:0]       uop;

//
// internal signals
//
  logic   [`FUNCT6_WIDTH-1:0]                         inst_funct6;      // inst original encoding[31:26]
  logic   [`NFIELD_WIDTH-1:0]                         inst_nf;          // inst original encoding[31:29]
  logic   [`VM_WIDTH-1:0]                             inst_vm;          // inst original encoding[25] 
  logic   [`REGFILE_INDEX_WIDTH-1:0]                  inst_vs2;         // inst original encoding[24:20]
  logic   [`UMOP_WIDTH-1:0]                           inst_umop;        // inst original encoding[24:20]
  logic   [`FUNCT3_WIDTH-1:0]                         inst_funct3;      // inst original encoding[14:12]
  logic   [`REGFILE_INDEX_WIDTH-1:0]                  inst_vd;          // inst original encoding[11:7]
  RVVOpCode                                           inst_opcode;      // inst original encoding[6:0]
  RVVConfigState                                      vector_csr_lsu;
  logic   [`VSTART_WIDTH-1:0]                         csr_vstart;
  logic   [`UOP_INDEX_WIDTH-1:0]                      uop_index_max;         
  EMUL_e                                              emul_vd;          
  EMUL_e                                              emul_vs2;          
  EMUL_e                                              emul_max; 
  EEW_e                                               eew_vd;          
  EEW_e                                               eew_vs2;          
  EEW_e                                               eew_max;          

  logic                                               valid_lsu;
  logic                                               valid_lsu_opcode;
  logic                                               valid_lsu_mop;
  logic   [`UOP_INDEX_WIDTH-1:0]                      uop_index_base;         
  logic   [`NUM_DE_UOP-1:0][`UOP_INDEX_WIDTH:0]       uop_index_current;   
  logic   [`NUM_DE_UOP-1:0]                           first_uop_valid;    
  logic   [`NUM_DE_UOP-1:0]                           last_uop_valid;     
  UOP_CLASS_e                                         uop_class;   
  RVVConfigState  [`NUM_DE_UOP-1:0]                   vector_csr;  
  logic   [`NUM_DE_UOP-1:0][`REGFILE_INDEX_WIDTH-1:0] vd_index;           
  logic   [`NUM_DE_UOP-1:0][$clog2(`EMUL_MAX)-1:0]    vd_offset;
  logic                                               vd_valid;
  logic                                               vs3_valid;          
  logic   [`NUM_DE_UOP-1:0][`REGFILE_INDEX_WIDTH-1:0] vs2_index; 	        
  logic   [`NUM_DE_UOP-1:0][$clog2(`EMUL_MAX)-1:0]    vs2_offset;
  logic   [`NUM_DE_UOP-1:0]                           vs2_valid;
  logic   [`NUM_DE_UOP-1:0][`UOP_INDEX_WIDTH-1:0]     uop_index;          
  logic   [`NUM_DE_UOP-1:0][$clog2(`EMUL_MAX)-1:0]    seg_field_index;
  logic   [`NUM_DE_UOP-1:0]                           pshrob_valid;  
  logic                                               pshlsu_valid;
  FUNCT6_u                                            funct6_lsu;  
  genvar                                              j;

//
// decode
//
  assign inst_funct6    = lcmd_valid ? lcmd.cmd.bits[24:19] : 'b0;
  assign inst_nf        = lcmd_valid ? lcmd.cmd.bits[24:22] : 'b0;
  assign inst_vm        = lcmd_valid ? lcmd.cmd.bits[18] : 'b0;
  assign inst_vs2       = lcmd_valid ? lcmd.cmd.bits[17:13] : 'b0;
  assign inst_umop      = lcmd_valid ? lcmd.cmd.bits[17:13] : 'b0;
  assign inst_funct3    = lcmd_valid ? lcmd.cmd.bits[7:5] : 'b0;
  assign inst_vd        = lcmd_valid ? lcmd.cmd.bits[4:0] : 'b0;
  assign inst_opcode    = lcmd_valid ? lcmd.cmd.opcode : LOAD;
  assign vector_csr_lsu = lcmd_valid ? lcmd.cmd.arch_state : RVVConfigState'('0);
  assign csr_vstart     = lcmd_valid ? lcmd.cmd.arch_state.vstart : 'b0;
  assign uop_index_max  = lcmd_valid ? lcmd.uop_index_max : 'b0;
  assign emul_vd        = lcmd_valid ? lcmd.emul_vd : EMUL_NONE; 
  assign emul_vs2       = lcmd_valid ? lcmd.emul_vs2 : EMUL_NONE;
  assign emul_max       = lcmd_valid ? lcmd.emul_max : EMUL_NONE;
  assign eew_vd         = lcmd_valid ? lcmd.eew_vd : EEW_NONE; 
  assign eew_vs2        = lcmd_valid ? lcmd.eew_vs2 : EEW_NONE;
  assign eew_max        = lcmd_valid ? lcmd.eew_max : EEW_NONE;

  // valid signal
  assign valid_lsu = valid_lsu_opcode&valid_lsu_mop&lcmd_valid;

  // identify load or store
  always_comb begin
    funct6_lsu.lsu_funct6.lsu_is_store = IS_LOAD;
    valid_lsu_opcode                   = 'b0;

    case(inst_opcode)
      LOAD: begin
        funct6_lsu.lsu_funct6.lsu_is_store = IS_LOAD;
        valid_lsu_opcode                   = 1'b1;
      end
      STORE: begin
        funct6_lsu.lsu_funct6.lsu_is_store = IS_STORE;
        valid_lsu_opcode                   = 1'b1;
      end
    endcase

  // lsu_mop distinguishes unit-stride, constant-stride, unordered index, ordered index
  // lsu_umop identifies what unit-stride instruction belong to when lsu_mop=US
    // initial 
    funct6_lsu.lsu_funct6.lsu_mop    = US;
    funct6_lsu.lsu_funct6.lsu_umop   = US_US;
    funct6_lsu.lsu_funct6.lsu_is_seg = NONE;
    valid_lsu_mop                    = 'b0;
    
    case(inst_funct6[2:0])
      UNIT_STRIDE: begin
        case(inst_umop)
          US_REGULAR: begin          
            funct6_lsu.lsu_funct6.lsu_mop    = US;
            funct6_lsu.lsu_funct6.lsu_umop   = US_US;
            valid_lsu_mop                    = 1'b1;
            funct6_lsu.lsu_funct6.lsu_is_seg = (inst_nf!=NF1) ? IS_SEGMENT : NONE;
          end
          US_WHOLE_REGISTER: begin
            funct6_lsu.lsu_funct6.lsu_mop    = US;
            funct6_lsu.lsu_funct6.lsu_umop   = US_WR;
            valid_lsu_mop                    = 1'b1;
          end
          US_MASK: begin
            funct6_lsu.lsu_funct6.lsu_mop    = US;
            funct6_lsu.lsu_funct6.lsu_umop   = US_MK;
            valid_lsu_mop                    = 1'b1;
          end
          US_FAULT_FIRST: begin
            funct6_lsu.lsu_funct6.lsu_mop    = US;
            funct6_lsu.lsu_funct6.lsu_umop   = US_FF;
            valid_lsu_mop                    = 1'b1;
            funct6_lsu.lsu_funct6.lsu_is_seg = (inst_nf!=NF1) ? IS_SEGMENT : NONE;
          end
        endcase
      end
      UNORDERED_INDEX: begin
        funct6_lsu.lsu_funct6.lsu_mop    = IU;
        valid_lsu_mop                    = 1'b1;
        funct6_lsu.lsu_funct6.lsu_is_seg = (inst_nf!=NF1) ? IS_SEGMENT : NONE;
      end
      CONSTANT_STRIDE: begin
        funct6_lsu.lsu_funct6.lsu_mop    = CS;
        valid_lsu_mop                    = 1'b1;
        funct6_lsu.lsu_funct6.lsu_is_seg = (inst_nf!=NF1) ? IS_SEGMENT : NONE;
      end
      ORDERED_INDEX: begin
        funct6_lsu.lsu_funct6.lsu_mop    = IO;
        valid_lsu_mop                    = 1'b1;
        funct6_lsu.lsu_funct6.lsu_is_seg = (inst_nf!=NF1) ? IS_SEGMENT : NONE;
      end
    endcase
  end

//
// split instruction to uops
//
  // uop_index_remain as the base uop_index
  assign uop_index_base = uop_index_remain;

  // calculate the uop_index used in decoding uops 
  generate
    for(j=0;j<`NUM_DE_UOP;j++) begin: GET_UOP_INDEX
      assign uop_index_current[j] = {1'b0, uop_index_base} + j[`UOP_INDEX_WIDTH:0];
    end
  endgenerate

  // generate uop valid
  always_comb begin        
    for(int i=0;i<`NUM_DE_UOP;i++) begin: GET_UOP_VALID
      uop_valid[i] = lcmd_valid&(uop_index_current[i]<={1'b0,uop_index_max});
    end
  end

  // update last_uop valid
  always_comb begin
    for(int i=0;i<`NUM_DE_UOP;i++) begin: GET_UOP_LAST
      first_uop_valid[i] = uop_index_current[i][`UOP_INDEX_WIDTH-1:0] == 'b0;
      last_uop_valid[i] = uop_index_current[i][`UOP_INDEX_WIDTH-1:0] == uop_index_max;
    end
  end

  // update uop class
  always_comb begin
    // initial 
    uop_class = XXX;
    
    case(inst_opcode) 
      LOAD:begin
        case(inst_funct6[2:0])
          UNIT_STRIDE,
          CONSTANT_STRIDE: begin
            uop_class = XXX;
          end
          UNORDERED_INDEX,
          ORDERED_INDEX: begin
            uop_class = XVX;
          end
        endcase
      end

      STORE: begin
        case(inst_funct6[2:0])
          UNIT_STRIDE,
          CONSTANT_STRIDE: begin
            uop_class = VXX;
          end
          UNORDERED_INDEX,
          ORDERED_INDEX: begin
            uop_class = VVX;
          end
        endcase
      end
    endcase
  end
  
  // update vector_csr and vstart
  always_comb begin
    for(int i=0;i<`NUM_DE_UOP;i++) begin: GET_UOP_VCSR
      // initial
      vector_csr[i] = vector_csr_lsu;

      // update vstart of every uop
      if(funct6_lsu.lsu_funct6.lsu_is_seg!=IS_SEGMENT) begin
        case({eew_vd,eew_vs2})
          // index load with eew_vd<eew_vs2
          {EEW8 ,EEW16}: begin
            vector_csr[i].vstart  = (`VSTART_WIDTH)'({uop_index[i][`UOP_INDEX_WIDTH-1:1],($clog2(`VLENB)'('b0))}) < csr_vstart ? 
                                      csr_vstart : 
                                      (`VSTART_WIDTH)'({uop_index[i][`UOP_INDEX_WIDTH-1:1],($clog2(`VLENB)'('b0))});
          end
          {EEW16,EEW32}: begin
            vector_csr[i].vstart  = (`VSTART_WIDTH)'({uop_index[i][`UOP_INDEX_WIDTH-1:1],($clog2(`VLEN/`HWORD_WIDTH)'('b0))}) < csr_vstart ? 
                                      csr_vstart : 
                                      (`VSTART_WIDTH)'({uop_index[i][`UOP_INDEX_WIDTH-1:1],($clog2(`VLEN/`HWORD_WIDTH)'('b0))});
          end
          {EEW8 ,EEW32}: begin
            vector_csr[i].vstart  = (`VSTART_WIDTH)'({uop_index[i][`UOP_INDEX_WIDTH-1:2],($clog2(`VLENB)'('b0))}) < csr_vstart ? 
                                      csr_vstart : 
                                      (`VSTART_WIDTH)'({uop_index[i][`UOP_INDEX_WIDTH-1:2],($clog2(`VLENB)'('b0))});
          end
          // other situations
          default: begin
            case(eew_max)
              EEW8: begin
                vector_csr[i].vstart  = (`VSTART_WIDTH)'({uop_index[i][`UOP_INDEX_WIDTH-1:0],($clog2(`VLENB)'('b0))}) < csr_vstart ? 
                                          csr_vstart : 
                                          (`VSTART_WIDTH)'({uop_index[i][`UOP_INDEX_WIDTH-1:0],($clog2(`VLENB)'('b0))});
              end
              EEW16: begin
                vector_csr[i].vstart  = (`VSTART_WIDTH)'({uop_index[i][`UOP_INDEX_WIDTH-1:0],($clog2(`VLEN/`HWORD_WIDTH)'('b0))}) < csr_vstart ? 
                                          csr_vstart : 
                                          (`VSTART_WIDTH)'({uop_index[i][`UOP_INDEX_WIDTH-1:0],($clog2(`VLEN/`HWORD_WIDTH)'('b0))});
              end
              EEW32: begin
                vector_csr[i].vstart  = (`VSTART_WIDTH)'({uop_index[i][`UOP_INDEX_WIDTH-1:0],($clog2(`VLEN/`WORD_WIDTH)'('b0))}) < csr_vstart ? 
                                          csr_vstart : 
                                          (`VSTART_WIDTH)'({uop_index[i][`UOP_INDEX_WIDTH-1:0],($clog2(`VLEN/`WORD_WIDTH)'('b0))});
              end
            endcase
          end
        endcase
      end
    end
  end

  // update vd_offset 
  always_comb begin
    for(int i=0;i<`NUM_DE_UOP;i++) begin: GET_VD_OFFSET
      // initial
      vd_offset[i] = 'b0;

      case(inst_funct6[2:0])
        UNIT_STRIDE: begin
          case(inst_umop)
            US_REGULAR,          
            US_FAULT_FIRST: begin
              case({inst_nf,emul_vd})
                {NF2,EMUL4}: begin 
                  case(uop_index_current[i][`UOP_INDEX_WIDTH-1:0])
                    5'd1   : vd_offset[i] = 3'd4;
                    5'd2   : vd_offset[i] = 3'd1;
                    5'd3   : vd_offset[i] = 3'd5;
                    5'd4   : vd_offset[i] = 3'd2;
                    5'd5   : vd_offset[i] = 3'd6;
                    5'd6   : vd_offset[i] = 3'd3;
                    default: vd_offset[i] = uop_index_current[i][2:0];
                  endcase   
                end
                {NF2,EMUL2}: begin
                  case(uop_index_current[i][`UOP_INDEX_WIDTH-1:0])
                    5'd1   : vd_offset[i] = 3'd2;
                    5'd2   : vd_offset[i] = 3'd1;
                    default: vd_offset[i] = uop_index_current[i][2:0];
                  endcase   
                end
                {NF3,EMUL2}: begin
                  case(uop_index_current[i][`UOP_INDEX_WIDTH-1:0])
                    5'd1   : vd_offset[i] = 3'd2;
                    5'd2   : vd_offset[i] = 3'd4;
                    5'd3   : vd_offset[i] = 3'd1;
                    5'd4   : vd_offset[i] = 3'd3;
                    default: vd_offset[i] = uop_index_current[i][2:0];
                  endcase   
                end
                {NF4,EMUL2}: begin
                  case(uop_index_current[i][`UOP_INDEX_WIDTH-1:0])
                    5'd1   : vd_offset[i] = 3'd2;
                    5'd2   : vd_offset[i] = 3'd4;
                    5'd3   : vd_offset[i] = 3'd6;
                    5'd4   : vd_offset[i] = 3'd1;
                    5'd5   : vd_offset[i] = 3'd3;
                    5'd6   : vd_offset[i] = 3'd5;
                    default: vd_offset[i] = uop_index_current[i][2:0];
                  endcase
                end
                default: 
                  vd_offset[i] = uop_index_current[i][2:0];
              endcase
            end
            US_WHOLE_REGISTER: begin
              vd_offset[i] = uop_index_current[i][2:0];
            end
            US_MASK: begin
              vd_offset[i] = 'b0;
            end
          endcase
        end

        CONSTANT_STRIDE: begin
          case({inst_nf,emul_vd})
            {NF2,EMUL4}: begin 
              case(uop_index_current[i][`UOP_INDEX_WIDTH-1:0])
                5'd1   : vd_offset[i] = 3'd4;
                5'd2   : vd_offset[i] = 3'd1;
                5'd3   : vd_offset[i] = 3'd5;
                5'd4   : vd_offset[i] = 3'd2;
                5'd5   : vd_offset[i] = 3'd6;
                5'd6   : vd_offset[i] = 3'd3;
                default: vd_offset[i] = uop_index_current[i][2:0];
              endcase   
            end
            {NF2,EMUL2}: begin
              case(uop_index_current[i][`UOP_INDEX_WIDTH-1:0])
                5'd1   : vd_offset[i] = 3'd2;
                5'd2   : vd_offset[i] = 3'd1;
                default: vd_offset[i] = uop_index_current[i][2:0];
              endcase   
            end
            {NF3,EMUL2}: begin
              case(uop_index_current[i][`UOP_INDEX_WIDTH-1:0])
                5'd1   : vd_offset[i] = 3'd2;
                5'd2   : vd_offset[i] = 3'd4;
                5'd3   : vd_offset[i] = 3'd1;
                5'd4   : vd_offset[i] = 3'd3;
                default: vd_offset[i] = uop_index_current[i][2:0];
              endcase   
            end
            {NF4,EMUL2}: begin
              case(uop_index_current[i][`UOP_INDEX_WIDTH-1:0])
                5'd1   : vd_offset[i] = 3'd2;
                5'd2   : vd_offset[i] = 3'd4;
                5'd3   : vd_offset[i] = 3'd6;
                5'd4   : vd_offset[i] = 3'd1;
                5'd5   : vd_offset[i] = 3'd3;
                5'd6   : vd_offset[i] = 3'd5;
                default: vd_offset[i] = uop_index_current[i][2:0];
              endcase
            end
            default: vd_offset[i] = uop_index_current[i][2:0];
          endcase
        end
        
        UNORDERED_INDEX,
        ORDERED_INDEX: begin
          case({eew_vs2,eew_vd})
            // EEW_vs2:EEW_vd=1:1
            {EEW8,EEW8},
            {EEW16,EEW16},
            {EEW32,EEW32},            
            // 1:2
            {EEW8,EEW16},
            {EEW16,EEW32},
            // 1:4
            {EEW8,EEW32}: begin            
              case({inst_nf,emul_vd})
                {NF2,EMUL4}: begin 
                  case(uop_index_current[i][`UOP_INDEX_WIDTH-1:0])
                    5'd1   : vd_offset[i] = 3'd4;
                    5'd2   : vd_offset[i] = 3'd1;
                    5'd3   : vd_offset[i] = 3'd5;
                    5'd4   : vd_offset[i] = 3'd2;
                    5'd5   : vd_offset[i] = 3'd6;
                    5'd6   : vd_offset[i] = 3'd3;
                    default: vd_offset[i] = uop_index_current[i][2:0];
                  endcase   
                end
                {NF2,EMUL2}: begin
                  case(uop_index_current[i][`UOP_INDEX_WIDTH-1:0])
                    5'd1   : vd_offset[i] = 3'd2;
                    5'd2   : vd_offset[i] = 3'd1;
                    default: vd_offset[i] = uop_index_current[i][2:0];
                  endcase   
                end
                {NF3,EMUL2}: begin
                  case(uop_index_current[i][`UOP_INDEX_WIDTH-1:0])
                    5'd1   : vd_offset[i] = 3'd2;
                    5'd2   : vd_offset[i] = 3'd4;
                    5'd3   : vd_offset[i] = 3'd1;
                    5'd4   : vd_offset[i] = 3'd3;
                    default: vd_offset[i] = uop_index_current[i][2:0];
                  endcase   
                end
                {NF4,EMUL2}: begin
                  case(uop_index_current[i][`UOP_INDEX_WIDTH-1:0])
                    5'd1   : vd_offset[i] = 3'd2;
                    5'd2   : vd_offset[i] = 3'd4;
                    5'd3   : vd_offset[i] = 3'd6;
                    5'd4   : vd_offset[i] = 3'd1;
                    5'd5   : vd_offset[i] = 3'd3;
                    5'd6   : vd_offset[i] = 3'd5;
                    default: vd_offset[i] = uop_index_current[i][2:0];
                  endcase
                end
                default: vd_offset[i] = uop_index_current[i][2:0];
              endcase
            end
            // 2:1
            {EEW16,EEW8},
            {EEW32,EEW16},
            // 4:1
            {EEW32,EEW8}: begin            
              case({emul_vs2,emul_vd})
                {EMUL1,EMUL1}: 
                  vd_offset[i] = uop_index_current[i][2:0];
                {EMUL2,EMUL1}:
                  vd_offset[i] = uop_index_current[i][3:1];
                {EMUL4,EMUL2}: begin
                  case(inst_nf)
                    NF2: begin
                      vd_offset[i] = {1'b0, uop_index_current[i][1], uop_index_current[i][2]};
                    end
                    NF3: begin
                      case(uop_index_current[i][`UOP_INDEX_WIDTH-1:1])
                        4'd1   : vd_offset[i] = 3'd2;
                        4'd2   : vd_offset[i] = 3'd4;
                        4'd3   : vd_offset[i] = 3'd1;
                        4'd4   : vd_offset[i] = 3'd3;
                        default: vd_offset[i] = uop_index_current[i][3:1];
                      endcase   
                    end
                    NF4: begin
                      vd_offset[i] = {uop_index_current[i][2:1], uop_index_current[i][3]};
                    end
                    default: vd_offset[i] = {1'b0, uop_index_current[i][2:1]};
                  endcase
                end
                {EMUL8,EMUL4}: begin 
                  if (inst_nf==NF2)
                    vd_offset[i] = {uop_index_current[i][1], uop_index_current[i][3:2]};
                  else
                    vd_offset[i] = {1'b0, uop_index_current[i][2:1]};
                end
                {EMUL4,EMUL1}: 
                  vd_offset[i] = uop_index_current[i][4:2];
                {EMUL8,EMUL2}: begin 
                  case(inst_nf)
                    NF2: begin
                      vd_offset[i] = {1'b0, uop_index_current[i][2], uop_index_current[i][3]};
                    end
                    NF3: begin
                      case(uop_index_current[i][`UOP_INDEX_WIDTH-1:2])
                        3'd1   : vd_offset[i] = 3'd2;
                        3'd2   : vd_offset[i] = 3'd4;
                        3'd3   : vd_offset[i] = 3'd1;
                        3'd4   : vd_offset[i] = 3'd3;
                        default: vd_offset[i] = uop_index_current[i][4:2];
                      endcase   
                    end
                    NF4: begin
                      vd_offset[i] = {uop_index_current[i][3:2], uop_index_current[i][4]};
                    end
                    default: vd_offset[i] = uop_index_current[i][4:2];
                  endcase
                end
              endcase
            end
          endcase
        end
      endcase
    end
  end

  // update vd_index and eew 
  always_comb begin
    for(int i=0;i<`NUM_DE_UOP;i++) begin: GET_VD
      vd_index[i] = inst_vd + {2'b0, vd_offset[i]};
    end
  end

  // update vd_valid and vs3_valid
  // some uop need vd as the vs3 vector operand
  always_comb begin
    // initial
    vs3_valid = 'b0;
    vd_valid  = 'b0;

    if(inst_opcode==STORE)
      vs3_valid = 1'b1;
    else
      vd_valid  = 1'b1;
  end

  // update vs2 offset and valid  
  always_comb begin
    // initial
    vs2_offset = 'b0; 
    vs2_valid  = 'b0; 
    
    for(int i=0;i<`NUM_DE_UOP;i++) begin: GET_VS2_OFFSET
      case(inst_funct6[2:0])
        UNORDERED_INDEX,
        ORDERED_INDEX: begin
          case({eew_vs2,eew_vd})
            // EEW_vs2:EEW_vd=1:1
            {EEW8,EEW8},
            {EEW16,EEW16},
            {EEW32,EEW32}: begin
              case(emul_vs2)
                EMUL2: begin
                  case(inst_nf)
                    NF2:     vs2_offset[i] = {2'b0, uop_index_current[i][1]};
                    NF3:     vs2_offset[i] = (uop_index_current[i]>='d3) ? 3'd1 : 3'b0;
                    NF4:     vs2_offset[i] = {2'b0, uop_index_current[i][2]};
                    default: vs2_offset[i] = {2'b0, uop_index_current[i][0]};
                  endcase
                  vs2_valid[i]  = 1'b1; 
                end
                EMUL4: begin
                  vs2_offset[i] = (inst_nf==NF2) ? {1'b0, uop_index_current[i][2:1]} : {1'b0, uop_index_current[i][1:0]};
                  vs2_valid[i]  = 1'b1; 
                end
                EMUL8: begin
                  vs2_offset[i] = uop_index_current[i][2:0];
                  vs2_valid[i]  = 1'b1; 
                end
                default: begin //EMUL1
                  vs2_offset[i] = 'b0;
                  vs2_valid[i]  = 1'b1; 
                end
              endcase
            end
            // 2:1
            {EEW16,EEW8},
            {EEW32,EEW16}: begin
              case(emul_vs2)
                EMUL2: begin
                  vs2_offset[i] = {2'b0, uop_index_current[i][0]};
                  vs2_valid[i]  = 1'b1; 
                end
                EMUL4: begin
                  case(inst_nf)
                    NF2:     vs2_offset[i] = {1'b0, uop_index_current[i][2], uop_index_current[i][0]};
                    NF3:     vs2_offset[i] = {1'b0, uop_index_current[i][3:1] >= 3'd3, uop_index_current[i][0]};
                    NF4:     vs2_offset[i] = {1'b0, uop_index_current[i][3], uop_index_current[i][0]};
                    default: vs2_offset[i] = uop_index_current[i][2:0]; // NF1
                  endcase
                  vs2_valid[i]  = 1'b1; 
                end
                EMUL8: begin
                  vs2_offset[i] = (inst_nf==NF2) ? {uop_index_current[i][3:2], uop_index_current[i][0]} : uop_index_current[i][2:0];
                  vs2_valid[i]  = 1'b1; 
                end
                default: begin //EMUL1
                  vs2_offset[i] = 'b0;
                  vs2_valid[i]  = 1'b1; 
                end
              endcase
            end
            // 4:1
            {EEW32,EEW8}: begin    
              case(emul_vs2)
                EMUL2: begin
                  vs2_offset[i] = {2'b0, uop_index_current[i][0]};
                  vs2_valid[i]  = 1'b1; 
                end
                EMUL4: begin
                  vs2_offset[i] = {1'b0, uop_index_current[i][1:0]};
                  vs2_valid[i]  = 1'b1; 
                end
                EMUL8: begin
                  case(inst_nf)
                    NF2:     vs2_offset[i] = {uop_index_current[i][3], uop_index_current[i][1:0]};
                    NF3:     vs2_offset[i] = {uop_index_current[i][4:2] >= 3'd3, uop_index_current[i][1:0]};
                    NF4:     vs2_offset[i] = {uop_index_current[i][4], uop_index_current[i][1:0]};
                    default: vs2_offset[i] = uop_index_current[i][2:0];  //NF1
                  endcase
                  vs2_valid[i]  = 1'b1; 
                end
                default: begin //EMUL1
                  vs2_offset[i] = 'b0;
                  vs2_valid[i]  = 1'b1; 
                end
              endcase
            end
            // 1:2
            {EEW8,EEW16},
            {EEW16,EEW32}: begin
              case(emul_vs2)
                EMUL1: begin
                  vs2_offset[i] = 'b0;
                  vs2_valid[i]  = 1'b1; 
                end
                EMUL2: begin
                  vs2_offset[i] = (inst_nf==NF2) ? {2'b0, uop_index_current[i][2]} : {2'b0, uop_index_current[i][1]};
                  vs2_valid[i]  = 1'b1; 
                end
                EMUL4: begin
                  vs2_offset[i] = {1'b0, uop_index_current[i][2:1]};
                  vs2_valid[i]  = 1'b1; 
                end
              endcase
            end
            // 1:4
            {EEW8,EEW32}: begin     
              case(emul_vs2)
                EMUL1: begin
                  vs2_offset[i] = 'b0;
                  vs2_valid[i]  = 1'b1; 
                end
                EMUL2: begin
                  vs2_offset[i] = {2'b0, uop_index_current[i][2]};
                  vs2_valid[i]  = 1'b1; 
                end
              endcase
            end
          endcase
        end
      endcase
    end
  end

  // update vs2 index and eew 
  always_comb begin
    for(int i=0;i<`NUM_DE_UOP;i++) begin: GET_VS2
      vs2_index[i] = inst_vs2 + {2'b0, vs2_offset[i]}; 
    end
  end

  // update uop index
  always_comb begin
    for(int i=0;i<`NUM_DE_UOP;i++) begin: ASSIGN_UOP_INDEX
      uop_index[i] = uop_index_current[i][`UOP_INDEX_WIDTH-1:0];
    end
  end


  // seg_field_index indicates the uop index for the same field in the segment.
  always_comb begin
    for(int i=0;i<`NUM_DE_UOP;i++) begin: GET_SEG_INDEX
      // default
      // unit-strid, constant-stride, and index with EEW_vs2<=EEW_vd
      if(inst_nf==NF2)
        seg_field_index[i] = {1'b0, uop_index_current[i][2:1]};
      else if(inst_nf==NF3)
        seg_field_index[i] = (uop_index_current[i]>=6'd3) ? 'd1 : 'b0;  
      else if(inst_nf==NF4)
        seg_field_index[i] = {2'b0,uop_index_current[i][2]};
      else
        seg_field_index[i] = 'b0;

      // EEW_vs2>EEW_vd for index load/store
      case(inst_funct6[2:0])
        UNORDERED_INDEX,
        ORDERED_INDEX: begin
          case({eew_vs2,eew_vd})
            // 2:1
            {EEW16,EEW8},
            {EEW32,EEW16}: begin
              case(emul_vs2)
                EMUL2: seg_field_index[i] = {2'b0, uop_index_current[i][0]};
                EMUL4: begin
                  case(inst_nf)
                    NF2:     seg_field_index[i] = {1'b0, uop_index_current[i][2], uop_index_current[i][0]};
                    NF3:     seg_field_index[i] = {1'b0, uop_index_current[i]>='d6, uop_index_current[i][0]};
                    NF4:     seg_field_index[i] = {1'b0, uop_index_current[i][3], uop_index_current[i][0]};
                    default: seg_field_index[i] = 'b0;
                  endcase
                end
                EMUL8: seg_field_index[i] = (inst_nf==NF2) ? {uop_index_current[i][3:2], uop_index_current[i][0]} : uop_index_current[i][2:0];
              endcase
            end
            // 4:1
            {EEW32,EEW8}: begin   
              case(emul_vs2)
                EMUL2: seg_field_index[i] = {2'b0, uop_index_current[i][0]};
                EMUL4: seg_field_index[i] = {1'b0, uop_index_current[i][1:0]};
                EMUL8: begin
                  case(inst_nf)
                    NF2:     seg_field_index[i] = {uop_index_current[i][3], uop_index_current[i][1:0]};
                    NF3:     seg_field_index[i] = {uop_index_current[i]>='d12, uop_index_current[i][1:0]};
                    NF4:     seg_field_index[i] = {uop_index_current[i][4], uop_index_current[i][1:0]};
                    default: seg_field_index[i] = uop_index_current[i][2:0];
                  endcase
                end
              endcase
            end
          endcase
        end
      endcase
    end
  end

  // pshrob_valid decide on whether this uop is pushed into ROB.
  always_comb begin
    for(int i=0;i<`NUM_DE_UOP;i++) begin: PSHROB_VLD
      // EEW_vs2>EEW_vd for index load/store
      case({eew_vs2,eew_vd})
        // 2:1
        {EEW16,EEW8},
        {EEW32,EEW16}: begin
          case(emul_vs2)
            EMUL2,
            EMUL4,
            EMUL8:   pshrob_valid[i] = uop_index_current[i][0];
            default: pshrob_valid[i] = 'b1;
          endcase
        end
        // 4:1
        {EEW32,EEW8}: begin   
          case(emul_vs2)
            EMUL2:   pshrob_valid[i] = uop_index_current[i][0];
            EMUL4,
            EMUL8:   pshrob_valid[i] = uop_index_current[i][1:0]==2'b11;
            default: pshrob_valid[i] = 'b1;
          endcase
        end
        default: pshrob_valid[i] = 'b1;
      endcase
    end
  end

  // pshlsu_valid decide on whether this uop is pushed into LSU RS.
`ifdef UNMK_USCS_LOAD_NOHANDSHAKE
  assign pshlsu_valid = !( inst_vm & 
                          (funct6_lsu.lsu_funct6.lsu_is_store==IS_LOAD) &
                          ((funct6_lsu.lsu_funct6.lsu_mop==US)||(funct6_lsu.lsu_funct6.lsu_mop==CS))
                         );
`else
  assign pshlsu_valid = 1'b1;
`endif

  // assign result to output
  generate
    for(j=0;j<`NUM_DE_UOP;j++) begin: ASSIGN_RES
    `ifdef TB_SUPPORT
      assign uop[j].uop_pc                = lcmd.cmd.inst_pc;
    `endif  
      assign uop[j].uop_funct3            = inst_funct3;
      assign uop[j].uop_funct6            = funct6_lsu;
      assign uop[j].uop_exe_unit          = LSU; 
      assign uop[j].uop_class             = uop_class;   
      assign uop[j].vector_csr            = vector_csr[j];  
      assign uop[j].vs_evl                = lcmd.evl;
      assign uop[j].ignore_vma            = 'b0;
      assign uop[j].ignore_vta            = 'b0;
      assign uop[j].force_vma_agnostic    = lcmd.force_vma_agnostic;
      assign uop[j].force_vta_agnostic    = lcmd.force_vta_agnostic;
      assign uop[j].vm                    = inst_vm;
      assign uop[j].v0_valid              = 'b1;          
      assign uop[j].dst_index             = vd_index[j];          
      assign uop[j].vd_eew                = lcmd.eew_vd;  
      assign uop[j].vd_valid              = vd_valid;
      assign uop[j].vs3_valid             = vs3_valid;
      assign uop[j].xd_valid              = 'b0; 
    `ifdef ZVE32F_ON
      assign uop[j].fd_valid              = 'b0; 
    `endif
      assign uop[j].vs1                   = 'b0;              
      assign uop[j].vs1_eew               = EEW_NONE;           
      assign uop[j].vs1_valid             = 'b0;
      assign uop[j].vs2_index 	          = vs2_index[j]; 	       
      assign uop[j].vs2_eew               = lcmd.eew_vs2;
      assign uop[j].vs2_valid             = vs2_valid[j];
      assign uop[j].rs1_data              = 'b0;           
      assign uop[j].rs1_data_valid        = 'b0;    
      assign uop[j].uop_index             = uop_index[j];         
      assign uop[j].first_uop_valid       = first_uop_valid[j];   
      assign uop[j].last_uop_valid        = last_uop_valid[j];    
      assign uop[j].seg_field_index       = seg_field_index[j];   
      assign uop[j].pshrob_valid          = pshrob_valid[j];   
      assign uop[j].pshlsu_valid          = pshlsu_valid;
    end
  endgenerate

endmodule
