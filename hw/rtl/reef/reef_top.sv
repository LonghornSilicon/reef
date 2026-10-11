// reef_top: Reef v1 test-chip top around the pinned CoralNPU M3 RvvCoreMiniAxi.
//
// Pins (10):
//   clk_i, rst_ni             core clock; active-low async reset (the core syncs it internally)
//   spi_sck_i, spi_csn_i,     host access to ITCM/DTCM/CSR through reef_spi2axi
//   spi_mosi_i, spi_miso_o    (upstream Spi2TLULV2 protocol, f_clk >= 16 * f_sck)
//   test_en_i                 DFT test enable -> core io_te, clock gates, reset bypass
//   halted_o, fault_o, wfi_o  core status
//
// Tie-offs and decisions (see docs/reef-top.md):
//   - io_boot_addr = 0 (ITCM base). The host loads ITCM, writes PC_START, releases RESET.
//   - io_irq / io_timer_irq / io_software_irq = 0 (no interrupt sources on the v1 chip).
//   - io_dm_* tied off: the debug module is reachable over AXI at CSR +0x800.
//   - io_debug_* (300 sim-trace outputs) left unconnected.
//   - The core's AXI master goes to reef_axi_err_slave (no off-chip memory): DECERR.
module reef_top (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic spi_sck_i,
  input  logic spi_csn_i,
  input  logic spi_mosi_i,
  output logic spi_miso_o,
  input  logic test_en_i,
  output logic halted_o,
  output logic fault_o,
  output logic wfi_o
);

  logic rst_n_sync;
  reef_rst_sync u_rst_sync (
    .clk_i     (clk_i),
    .rst_ni    (rst_ni),
    .test_en_i (test_en_i),
    .rst_no    (rst_n_sync)
  );

  // Core AXI ports
  logic s_write_addr_ready;
  logic s_write_addr_valid;
  logic [31:0] s_write_addr_bits_addr;
  logic [2:0] s_write_addr_bits_prot;
  logic [5:0] s_write_addr_bits_id;
  logic [7:0] s_write_addr_bits_len;
  logic [2:0] s_write_addr_bits_size;
  logic [1:0] s_write_addr_bits_burst;
  logic s_write_addr_bits_lock;
  logic [3:0] s_write_addr_bits_cache;
  logic [3:0] s_write_addr_bits_qos;
  logic [3:0] s_write_addr_bits_region;
  logic s_write_data_ready;
  logic s_write_data_valid;
  logic [127:0] s_write_data_bits_data;
  logic s_write_data_bits_last;
  logic [15:0] s_write_data_bits_strb;
  logic s_write_resp_ready;
  logic s_write_resp_valid;
  logic [5:0] s_write_resp_bits_id;
  logic [1:0] s_write_resp_bits_resp;
  logic s_read_addr_ready;
  logic s_read_addr_valid;
  logic [31:0] s_read_addr_bits_addr;
  logic [2:0] s_read_addr_bits_prot;
  logic [5:0] s_read_addr_bits_id;
  logic [7:0] s_read_addr_bits_len;
  logic [2:0] s_read_addr_bits_size;
  logic [1:0] s_read_addr_bits_burst;
  logic s_read_addr_bits_lock;
  logic [3:0] s_read_addr_bits_cache;
  logic [3:0] s_read_addr_bits_qos;
  logic [3:0] s_read_addr_bits_region;
  logic s_read_data_ready;
  logic s_read_data_valid;
  logic [127:0] s_read_data_bits_data;
  logic [5:0] s_read_data_bits_id;
  logic [1:0] s_read_data_bits_resp;
  logic s_read_data_bits_last;
  logic m_write_addr_ready;
  logic m_write_addr_valid;
  logic [31:0] m_write_addr_bits_addr;
  logic [2:0] m_write_addr_bits_prot;
  logic [5:0] m_write_addr_bits_id;
  logic [7:0] m_write_addr_bits_len;
  logic [2:0] m_write_addr_bits_size;
  logic [1:0] m_write_addr_bits_burst;
  logic m_write_addr_bits_lock;
  logic [3:0] m_write_addr_bits_cache;
  logic [3:0] m_write_addr_bits_qos;
  logic [3:0] m_write_addr_bits_region;
  logic m_write_data_ready;
  logic m_write_data_valid;
  logic [127:0] m_write_data_bits_data;
  logic m_write_data_bits_last;
  logic [15:0] m_write_data_bits_strb;
  logic m_write_resp_ready;
  logic m_write_resp_valid;
  logic [5:0] m_write_resp_bits_id;
  logic [1:0] m_write_resp_bits_resp;
  logic m_read_addr_ready;
  logic m_read_addr_valid;
  logic [31:0] m_read_addr_bits_addr;
  logic [2:0] m_read_addr_bits_prot;
  logic [5:0] m_read_addr_bits_id;
  logic [7:0] m_read_addr_bits_len;
  logic [2:0] m_read_addr_bits_size;
  logic [1:0] m_read_addr_bits_burst;
  logic m_read_addr_bits_lock;
  logic [3:0] m_read_addr_bits_cache;
  logic [3:0] m_read_addr_bits_qos;
  logic [3:0] m_read_addr_bits_region;
  logic m_read_data_ready;
  logic m_read_data_valid;
  logic [127:0] m_read_data_bits_data;
  logic [5:0] m_read_data_bits_id;
  logic [1:0] m_read_data_bits_resp;
  logic m_read_data_bits_last;

  // Host -> core AXI slave
  logic spi_err;
  reef_spi2axi #(.IdW(6)) u_spi2axi (
    .clk_i      (clk_i),
    .rst_ni     (rst_n_sync),
    .spi_sck_i  (spi_sck_i),
    .spi_csn_i  (spi_csn_i),
    .spi_mosi_i (spi_mosi_i),
    .spi_miso_o (spi_miso_o),
    .aw_valid_o (s_write_addr_valid),
    .aw_ready_i (s_write_addr_ready),
    .aw_addr_o  (s_write_addr_bits_addr),
    .aw_id_o    (s_write_addr_bits_id),
    .aw_len_o   (s_write_addr_bits_len),
    .aw_size_o  (s_write_addr_bits_size),
    .aw_burst_o (s_write_addr_bits_burst),
    .w_valid_o  (s_write_data_valid),
    .w_ready_i  (s_write_data_ready),
    .w_data_o   (s_write_data_bits_data),
    .w_strb_o   (s_write_data_bits_strb),
    .w_last_o   (s_write_data_bits_last),
    .b_valid_i  (s_write_resp_valid),
    .b_ready_o  (s_write_resp_ready),
    .b_resp_i   (s_write_resp_bits_resp),
    .ar_valid_o (s_read_addr_valid),
    .ar_ready_i (s_read_addr_ready),
    .ar_addr_o  (s_read_addr_bits_addr),
    .ar_id_o    (s_read_addr_bits_id),
    .ar_len_o   (s_read_addr_bits_len),
    .ar_size_o  (s_read_addr_bits_size),
    .ar_burst_o (s_read_addr_bits_burst),
    .r_valid_i  (s_read_data_valid),
    .r_ready_o  (s_read_data_ready),
    .r_data_i   (s_read_data_bits_data),
    .r_resp_i   (s_read_data_bits_resp),
    .err_o      (spi_err)
  );
  assign s_write_addr_bits_prot   = 3'b000;
  assign s_write_addr_bits_lock   = 1'b0;
  assign s_write_addr_bits_cache  = 4'b0000;
  assign s_write_addr_bits_qos    = 4'b0000;
  assign s_write_addr_bits_region = 4'b0000;
  assign s_read_addr_bits_prot    = 3'b000;
  assign s_read_addr_bits_lock    = 1'b0;
  assign s_read_addr_bits_cache   = 4'b0000;
  assign s_read_addr_bits_qos     = 4'b0000;
  assign s_read_addr_bits_region  = 4'b0000;

  // Core AXI master -> error responder
  reef_axi_err_slave #(.IdW(6)) u_err_slave (
    .clk_i      (clk_i),
    .rst_ni     (rst_n_sync),
    .aw_valid_i (m_write_addr_valid),
    .aw_ready_o (m_write_addr_ready),
    .aw_id_i    (m_write_addr_bits_id),
    .w_valid_i  (m_write_data_valid),
    .w_ready_o  (m_write_data_ready),
    .w_last_i   (m_write_data_bits_last),
    .b_valid_o  (m_write_resp_valid),
    .b_ready_i  (m_write_resp_ready),
    .b_id_o     (m_write_resp_bits_id),
    .b_resp_o   (m_write_resp_bits_resp),
    .ar_valid_i (m_read_addr_valid),
    .ar_ready_o (m_read_addr_ready),
    .ar_id_i    (m_read_addr_bits_id),
    .ar_len_i   (m_read_addr_bits_len),
    .r_valid_o  (m_read_data_valid),
    .r_ready_i  (m_read_data_ready),
    .r_id_o     (m_read_data_bits_id),
    .r_data_o   (m_read_data_bits_data),
    .r_resp_o   (m_read_data_bits_resp),
    .r_last_o   (m_read_data_bits_last)
  );

  RvvCoreMiniAxi u_core (
    .io_aclk                                 (clk_i),
    .io_aresetn                              (rst_ni),
    .io_axi_slave_write_addr_ready           (s_write_addr_ready),
    .io_axi_slave_write_addr_valid           (s_write_addr_valid),
    .io_axi_slave_write_addr_bits_addr       (s_write_addr_bits_addr),
    .io_axi_slave_write_addr_bits_prot       (s_write_addr_bits_prot),
    .io_axi_slave_write_addr_bits_id         (s_write_addr_bits_id),
    .io_axi_slave_write_addr_bits_len        (s_write_addr_bits_len),
    .io_axi_slave_write_addr_bits_size       (s_write_addr_bits_size),
    .io_axi_slave_write_addr_bits_burst      (s_write_addr_bits_burst),
    .io_axi_slave_write_addr_bits_lock       (s_write_addr_bits_lock),
    .io_axi_slave_write_addr_bits_cache      (s_write_addr_bits_cache),
    .io_axi_slave_write_addr_bits_qos        (s_write_addr_bits_qos),
    .io_axi_slave_write_addr_bits_region     (s_write_addr_bits_region),
    .io_axi_slave_write_data_ready           (s_write_data_ready),
    .io_axi_slave_write_data_valid           (s_write_data_valid),
    .io_axi_slave_write_data_bits_data       (s_write_data_bits_data),
    .io_axi_slave_write_data_bits_last       (s_write_data_bits_last),
    .io_axi_slave_write_data_bits_strb       (s_write_data_bits_strb),
    .io_axi_slave_write_resp_ready           (s_write_resp_ready),
    .io_axi_slave_write_resp_valid           (s_write_resp_valid),
    .io_axi_slave_write_resp_bits_id         (s_write_resp_bits_id),
    .io_axi_slave_write_resp_bits_resp       (s_write_resp_bits_resp),
    .io_axi_slave_read_addr_ready            (s_read_addr_ready),
    .io_axi_slave_read_addr_valid            (s_read_addr_valid),
    .io_axi_slave_read_addr_bits_addr        (s_read_addr_bits_addr),
    .io_axi_slave_read_addr_bits_prot        (s_read_addr_bits_prot),
    .io_axi_slave_read_addr_bits_id          (s_read_addr_bits_id),
    .io_axi_slave_read_addr_bits_len         (s_read_addr_bits_len),
    .io_axi_slave_read_addr_bits_size        (s_read_addr_bits_size),
    .io_axi_slave_read_addr_bits_burst       (s_read_addr_bits_burst),
    .io_axi_slave_read_addr_bits_lock        (s_read_addr_bits_lock),
    .io_axi_slave_read_addr_bits_cache       (s_read_addr_bits_cache),
    .io_axi_slave_read_addr_bits_qos         (s_read_addr_bits_qos),
    .io_axi_slave_read_addr_bits_region      (s_read_addr_bits_region),
    .io_axi_slave_read_data_ready            (s_read_data_ready),
    .io_axi_slave_read_data_valid            (s_read_data_valid),
    .io_axi_slave_read_data_bits_data        (s_read_data_bits_data),
    .io_axi_slave_read_data_bits_id          (s_read_data_bits_id),
    .io_axi_slave_read_data_bits_resp        (s_read_data_bits_resp),
    .io_axi_slave_read_data_bits_last        (s_read_data_bits_last),
    .io_axi_master_write_addr_ready          (m_write_addr_ready),
    .io_axi_master_write_addr_valid          (m_write_addr_valid),
    .io_axi_master_write_addr_bits_addr      (m_write_addr_bits_addr),
    .io_axi_master_write_addr_bits_prot      (m_write_addr_bits_prot),
    .io_axi_master_write_addr_bits_id        (m_write_addr_bits_id),
    .io_axi_master_write_addr_bits_len       (m_write_addr_bits_len),
    .io_axi_master_write_addr_bits_size      (m_write_addr_bits_size),
    .io_axi_master_write_addr_bits_burst     (m_write_addr_bits_burst),
    .io_axi_master_write_addr_bits_lock      (m_write_addr_bits_lock),
    .io_axi_master_write_addr_bits_cache     (m_write_addr_bits_cache),
    .io_axi_master_write_addr_bits_qos       (m_write_addr_bits_qos),
    .io_axi_master_write_addr_bits_region    (m_write_addr_bits_region),
    .io_axi_master_write_data_ready          (m_write_data_ready),
    .io_axi_master_write_data_valid          (m_write_data_valid),
    .io_axi_master_write_data_bits_data      (m_write_data_bits_data),
    .io_axi_master_write_data_bits_last      (m_write_data_bits_last),
    .io_axi_master_write_data_bits_strb      (m_write_data_bits_strb),
    .io_axi_master_write_resp_ready          (m_write_resp_ready),
    .io_axi_master_write_resp_valid          (m_write_resp_valid),
    .io_axi_master_write_resp_bits_id        (m_write_resp_bits_id),
    .io_axi_master_write_resp_bits_resp      (m_write_resp_bits_resp),
    .io_axi_master_read_addr_ready           (m_read_addr_ready),
    .io_axi_master_read_addr_valid           (m_read_addr_valid),
    .io_axi_master_read_addr_bits_addr       (m_read_addr_bits_addr),
    .io_axi_master_read_addr_bits_prot       (m_read_addr_bits_prot),
    .io_axi_master_read_addr_bits_id         (m_read_addr_bits_id),
    .io_axi_master_read_addr_bits_len        (m_read_addr_bits_len),
    .io_axi_master_read_addr_bits_size       (m_read_addr_bits_size),
    .io_axi_master_read_addr_bits_burst      (m_read_addr_bits_burst),
    .io_axi_master_read_addr_bits_lock       (m_read_addr_bits_lock),
    .io_axi_master_read_addr_bits_cache      (m_read_addr_bits_cache),
    .io_axi_master_read_addr_bits_qos        (m_read_addr_bits_qos),
    .io_axi_master_read_addr_bits_region     (m_read_addr_bits_region),
    .io_axi_master_read_data_ready           (m_read_data_ready),
    .io_axi_master_read_data_valid           (m_read_data_valid),
    .io_axi_master_read_data_bits_data       (m_read_data_bits_data),
    .io_axi_master_read_data_bits_id         (m_read_data_bits_id),
    .io_axi_master_read_data_bits_resp       (m_read_data_bits_resp),
    .io_axi_master_read_data_bits_last       (m_read_data_bits_last),
    .io_halted                               (halted_o),
    .io_fault                                (fault_o),
    .io_wfi                                  (wfi_o),
    .io_irq                                  (1'b0),
    .io_boot_addr                            (32'h0000_0000),
    .io_timer_irq                            (1'b0),
    .io_software_irq                         (1'b0),
    .io_dm_req_ready                         (),
    .io_dm_req_valid                         (1'b0),
    .io_dm_req_bits_address                  ('0),
    .io_dm_req_bits_data                     ('0),
    .io_dm_req_bits_op                       ('0),
    .io_dm_rsp_ready                         (1'b1),
    .io_dm_rsp_valid                         (),
    .io_dm_rsp_bits_data                     (),
    .io_dm_rsp_bits_op                       (),
    .io_te                                   (test_en_i)
  );

endmodule
