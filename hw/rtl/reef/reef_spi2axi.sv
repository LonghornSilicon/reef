// reef_spi2axi: SPI slave -> AXI4 master, so a host can reach the core's AXI slave
// (ITCM, DTCM, CSR) through 4 pins.
//
// Wire protocol = upstream CoralNPU Spi2TLULV2 (M3 72d700e, hdl/chisel/src/bus/Spi2TLULV2.scala),
// so upstream's host tools (sw/utils/nexus_loader, FtdiSpiMaster) work unchanged:
//   SPI mode 0, MSB first, one frame per CS low.
//   Header (7 bytes): op, addr[31:24], addr[23:16], addr[15:8], addr[7:0], len[15:8], len[7:0]
//     op 0x02 = write, 0x01 = read; len = number of 16-byte beats - 1.
//   Write: (len+1)*16 data bytes follow; byte k of a beat is data[8k+7:8k].
//   Read:  for each beat MISO sends a 0xFE sync byte, then the 16 data bytes (byte 0 first).
//          MISO is 0 while idle; the host searches the bit stream for 0xFE.
//   Beat i goes to addr + 16*i, AXI single beat, size 16 B, strobe 0xFFFF << addr[3:0].
//
// Difference from upstream: one clock domain. The SPI pins are oversampled in clk instead of
// clocking logic with SCK, so there is no CDC FIFO and no SCK clock tree.
// Requirement: f_clk >= 16 * f_sck.
module reef_spi2axi #(
  parameter int IdW = 6
) (
  input  logic           clk_i,
  input  logic           rst_ni,

  input  logic           spi_sck_i,
  input  logic           spi_csn_i,
  input  logic           spi_mosi_i,
  output logic           spi_miso_o,

  output logic           aw_valid_o,
  input  logic           aw_ready_i,
  output logic [31:0]    aw_addr_o,
  output logic [IdW-1:0] aw_id_o,
  output logic [7:0]     aw_len_o,
  output logic [2:0]     aw_size_o,
  output logic [1:0]     aw_burst_o,
  output logic           w_valid_o,
  input  logic           w_ready_i,
  output logic [127:0]   w_data_o,
  output logic [15:0]    w_strb_o,
  output logic           w_last_o,
  input  logic           b_valid_i,
  output logic           b_ready_o,
  input  logic [1:0]     b_resp_i,
  output logic           ar_valid_o,
  input  logic           ar_ready_i,
  output logic [31:0]    ar_addr_o,
  output logic [IdW-1:0] ar_id_o,
  output logic [7:0]     ar_len_o,
  output logic [2:0]     ar_size_o,
  output logic [1:0]     ar_burst_o,
  input  logic           r_valid_i,
  output logic           r_ready_o,
  input  logic [127:0]   r_data_i,
  input  logic [1:0]     r_resp_i,

  // Sticky: an AXI error response, or a write beat dropped because the queue was full.
  // Cleared at the start of the next frame.
  output logic           err_o
);

  localparam logic [7:0] OpRead  = 8'h01;
  localparam logic [7:0] OpWrite = 8'h02;
  localparam logic [7:0] SyncTok = 8'hFE;

  // ---------------------------------------------------------------- pin sync
  logic [2:0] sck_q;
  logic [1:0] csn_q, mosi_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      sck_q  <= '0;
      csn_q  <= 2'b11;
      mosi_q <= '0;
    end else begin
      sck_q  <= {sck_q[1:0], spi_sck_i};
      csn_q  <= {csn_q[0], spi_csn_i};
      mosi_q <= {mosi_q[0], spi_mosi_i};
    end
  end
  wire sck_rise = sck_q[1] & ~sck_q[2];
  wire cs_act   = ~csn_q[1];
  wire mosi     = mosi_q[1];

  // A frame starts when CS goes low.
  logic cs_act_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) cs_act_q <= 1'b0;
    else         cs_act_q <= cs_act;
  end
  wire frame_start = cs_act & ~cs_act_q;

  // ---------------------------------------------------------------- MOSI bytes
  logic [2:0] rx_cnt;
  logic [6:0] rx_sr;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rx_cnt <= '0;
      rx_sr  <= '0;
    end else if (!cs_act) begin
      rx_cnt <= '0;
    end else if (sck_rise) begin
      rx_sr  <= {rx_sr[5:0], mosi};
      rx_cnt <= rx_cnt + 3'd1;
    end
  end
  wire       rx_valid = cs_act & sck_rise & (rx_cnt == 3'd7);
  wire [7:0] rx_byte  = {rx_sr, mosi};

  // ---------------------------------------------------------------- frame parser
  typedef enum logic [3:0] {
    PsOp, PsA3, PsA2, PsA1, PsA0, PsL1, PsL0, PsWData, PsEnd
  } ps_e;
  ps_e         ps;
  logic [7:0]  op;
  logic [31:0] addr;
  logic [7:0]  len_hi;
  logic [31:0] wr_addr;
  logic [16:0] wr_beats;
  logic [3:0]  wr_idx;
  logic [119:0] wr_buf;

  // Write beat queue (2 entries) between the parser and the AXI write engine.
  logic         wq_push, wq_pop;
  logic [159:0] wq_din, wq_dout;
  logic         wq_full, wq_empty;

  logic        rd_start;
  logic [31:0] rd_start_addr;
  logic [16:0] rd_start_beats;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      ps       <= PsOp;
      op       <= '0;
      addr     <= '0;
      len_hi   <= '0;
      wr_addr  <= '0;
      wr_beats <= '0;
      wr_idx   <= '0;
      wr_buf   <= '0;
    end else if (!cs_act) begin
      ps <= PsOp;
    end else if (rx_valid) begin
      unique case (ps)
        PsOp: begin op <= rx_byte; ps <= PsA3; end
        PsA3: begin addr[31:24] <= rx_byte; ps <= PsA2; end
        PsA2: begin addr[23:16] <= rx_byte; ps <= PsA1; end
        PsA1: begin addr[15:8]  <= rx_byte; ps <= PsA0; end
        PsA0: begin addr[7:0]   <= rx_byte; ps <= PsL1; end
        PsL1: begin len_hi      <= rx_byte; ps <= PsL0; end
        PsL0: begin
          wr_addr  <= addr;
          wr_beats <= {1'b0, len_hi, rx_byte} + 17'd1;
          wr_idx   <= '0;
          ps       <= (op == OpWrite) ? PsWData : PsEnd;
        end
        PsWData: begin
          wr_buf[wr_idx*8 +: 8] <= rx_byte;
          wr_idx <= wr_idx + 4'd1;
          if (wr_idx == 4'd15) begin
            wr_addr  <= wr_addr + 32'd16;
            wr_beats <= wr_beats - 17'd1;
            if (wr_beats == 17'd1) ps <= PsEnd;
          end
        end
        default: ;  // PsEnd: ignore bytes until CS goes high
      endcase
    end
  end

  assign wq_push = rx_valid & (ps == PsWData) & (wr_idx == 4'd15);
  assign wq_din  = {wr_addr, rx_byte, wr_buf};

  assign rd_start       = rx_valid & (ps == PsL0) & (op == OpRead);
  assign rd_start_addr  = addr;
  assign rd_start_beats = {1'b0, len_hi, rx_byte} + 17'd1;

  // 2-entry FIFO
  logic [159:0] wq_mem [2];
  logic         wq_wp, wq_rp;
  logic [1:0]   wq_cnt;
  assign wq_full  = (wq_cnt == 2'd2);
  assign wq_empty = (wq_cnt == 2'd0);
  assign wq_dout  = wq_mem[wq_rp];
  wire   wq_do_push = wq_push & ~wq_full;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wq_wp  <= 1'b0;
      wq_rp  <= 1'b0;
      wq_cnt <= '0;
    end else begin
      if (wq_do_push) begin
        wq_mem[wq_wp] <= wq_din;
        wq_wp <= ~wq_wp;
      end
      if (wq_pop) wq_rp <= ~wq_rp;
      wq_cnt <= wq_cnt + {1'b0, wq_do_push} - {1'b0, wq_pop};
    end
  end

  // ---------------------------------------------------------------- AXI write engine
  // AW and W are offered together; the beat is popped when B returns.
  logic aw_done, w_done, wr_busy;
  wire  [31:0] wq_addr = wq_dout[159:128];
  assign wr_busy    = ~wq_empty;
  assign aw_valid_o = wr_busy & ~aw_done;
  assign aw_addr_o  = wq_addr;
  assign aw_id_o    = '0;
  assign aw_len_o   = 8'd0;
  assign aw_size_o  = 3'd4;
  assign aw_burst_o = 2'b01;
  assign w_valid_o  = wr_busy & ~w_done;
  assign w_data_o   = wq_dout[127:0];
  assign w_strb_o   = 16'hFFFF << wq_addr[3:0];
  assign w_last_o   = 1'b1;
  assign b_ready_o  = wr_busy & aw_done & w_done;
  assign wq_pop     = b_valid_i & b_ready_o;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      aw_done <= 1'b0;
      w_done  <= 1'b0;
    end else if (wq_pop) begin
      aw_done <= 1'b0;
      w_done  <= 1'b0;
    end else begin
      if (aw_valid_o & aw_ready_i) aw_done <= 1'b1;
      if (w_valid_o & w_ready_i)   w_done  <= 1'b1;
    end
  end

  // ---------------------------------------------------------------- AXI read engine
  // One beat in flight. A beat is requested only when the TX queue has room for it, so R is
  // always accepted. If CS goes high mid-read, no new AR is issued and an in-flight R is dropped.
  logic        rd_active, ar_pend, r_pend;
  logic [31:0] rd_addr;
  logic [16:0] rd_beats;
  logic        tq_push;
  logic [1:0]  tq_cnt;
  logic        tq_has_room;

  assign ar_valid_o = ar_pend;
  assign ar_addr_o  = rd_addr;
  assign ar_id_o    = '0;
  assign ar_len_o   = 8'd0;
  assign ar_size_o  = 3'd4;
  assign ar_burst_o = 2'b01;
  assign r_ready_o  = r_pend;
  assign tq_push    = r_valid_i & r_pend & rd_active;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rd_active <= 1'b0;
      ar_pend   <= 1'b0;
      r_pend    <= 1'b0;
      rd_addr   <= '0;
      rd_beats  <= '0;
    end else begin
      if (rd_start) begin
        rd_active <= 1'b1;
        rd_addr   <= rd_start_addr;
        rd_beats  <= rd_start_beats;
      end else if (!cs_act) begin
        rd_active <= 1'b0;
      end
      if (ar_pend && ar_ready_i) begin
        ar_pend  <= 1'b0;
        r_pend   <= 1'b1;
        rd_addr  <= rd_addr + 32'd16;
        rd_beats <= rd_beats - 17'd1;
      end else if (!ar_pend && !r_pend && rd_active && cs_act && !rd_start &&
                   rd_beats != 17'd0 && tq_has_room) begin
        ar_pend <= 1'b1;
      end
      if (r_valid_i && r_pend) r_pend <= 1'b0;
    end
  end

  // ---------------------------------------------------------------- TX queue + MISO
  logic [127:0] tq_mem [2];
  logic         tq_wp, tq_rp;
  logic         tq_pop;
  assign tq_has_room = (tq_cnt == 2'd0);  // with one beat in flight this keeps the queue <= 1
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      tq_wp  <= 1'b0;
      tq_rp  <= 1'b0;
      tq_cnt <= '0;
    end else if (!cs_act) begin
      tq_wp  <= 1'b0;
      tq_rp  <= 1'b0;
      tq_cnt <= '0;
    end else begin
      if (tq_push) begin
        tq_mem[tq_wp] <= r_data_i;
        tq_wp <= ~tq_wp;
      end
      if (tq_pop) tq_rp <= ~tq_rp;
      tq_cnt <= tq_cnt + {1'b0, tq_push} - {1'b0, tq_pop};
    end
  end

  // tx_idx 0 = sync byte, 1..16 = data bytes 0..15.
  logic [4:0] tx_idx;
  logic [2:0] tx_cnt;
  logic [6:0] tx_sr;
  wire        tx_have = (tq_cnt != 2'd0);
  wire [127:0] tq_head = tq_mem[tq_rp];
  wire [7:0]  tx_byte = (tx_idx == 5'd0) ? SyncTok : tq_head[(tx_idx - 5'd1)*8 +: 8];
  wire        tx_load = cs_act & sck_rise & (tx_cnt == 3'd0) & tx_have;
  assign tq_pop = tx_load & (tx_idx == 5'd16);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      tx_idx     <= '0;
      tx_cnt     <= '0;
      tx_sr      <= '0;
      spi_miso_o <= 1'b0;
    end else if (!cs_act) begin
      tx_idx     <= '0;
      tx_cnt     <= '0;
      spi_miso_o <= 1'b0;
    end else if (sck_rise) begin
      if (tx_cnt == 3'd0) begin
        if (tx_have) begin
          spi_miso_o <= tx_byte[7];
          tx_sr      <= tx_byte[6:0];
          tx_cnt     <= 3'd7;
          tx_idx     <= (tx_idx == 5'd16) ? 5'd0 : tx_idx + 5'd1;
        end else begin
          spi_miso_o <= 1'b0;
        end
      end else begin
        spi_miso_o <= tx_sr[6];
        tx_sr      <= {tx_sr[5:0], 1'b0};
        tx_cnt     <= tx_cnt - 3'd1;
      end
    end
  end

  // ---------------------------------------------------------------- error flag
  // resp[1] set = SLVERR/DECERR; resp[0] alone (EXOKAY) is not an error.
  /* verilator lint_off UNUSEDSIGNAL */
  wire unused_resp = b_resp_i[0] ^ r_resp_i[0];
  /* verilator lint_on UNUSEDSIGNAL */
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)                           err_o <= 1'b0;
    else if (frame_start)                  err_o <= 1'b0;
    else if (wq_push & wq_full)            err_o <= 1'b1;
    else if (wq_pop & b_resp_i[1])         err_o <= 1'b1;
    else if (r_valid_i & r_pend & r_resp_i[1]) err_o <= 1'b1;
  end

endmodule
