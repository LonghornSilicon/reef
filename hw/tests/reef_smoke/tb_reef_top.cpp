// Reef smoke test: drives reef_top only through its pins, like the bring-up host will.
//
//   1. SPI read of the CSR line (reset state)
//   2. SPI load of the program into ITCM, read back and compare
//   3. SPI write/read of a DTCM pattern
//   4. PC_START write over SPI (unaligned CSR address), read back
//   5. release clock gate, then reset, through the RESET CSR
//   6. run until halted_o, check fault_o
//   7. SPI read of the result struct in DTCM
//
// The SPI host follows upstream nexus_loader (M3 sw/utils/nexus_loader/spi_master.cc):
// 7-byte header, 16 flush bytes after a write, bit-level search for the 0xFE sync on reads.
//
// Usage: Vreef_top <program.bin> <result_addr_hex> [sck_ratio] [pass|buserr]
//   buserr: the program reads an off-chip address; expect DECERR -> trap (mcause 5) -> fault_o.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

#include "Vreef_top.h"
#include "verilated.h"

namespace {

constexpr uint32_t kCsrBase = 0x30000;
constexpr uint32_t kDtcmTestAddr = 0x14000;

int g_fail = 0;
#define CHECK(cond, ...)                         \
  do {                                           \
    if (!(cond)) {                               \
      std::printf("FAIL: " __VA_ARGS__);         \
      std::printf("\n");                         \
      g_fail++;                                  \
    }                                            \
  } while (0)

class Tb {
 public:
  explicit Tb(int ratio) : ratio_(ratio), top_(new Vreef_top) {
    top_->clk_i = 0;
    top_->rst_ni = 0;
    top_->spi_sck_i = 0;
    top_->spi_csn_i = 1;
    top_->spi_mosi_i = 0;
    top_->test_en_i = 0;
    top_->eval();
  }
  ~Tb() { top_->final(); }

  void Tick() {
    top_->clk_i = 1;
    top_->eval();
    top_->clk_i = 0;
    top_->eval();
    cycles_++;
  }
  void Ticks(int n) {
    for (int i = 0; i < n; i++) Tick();
  }
  void Reset() {
    top_->rst_ni = 0;
    Ticks(10);
    top_->rst_ni = 1;
    Ticks(20);
  }
  uint64_t cycles() const { return cycles_; }
  Vreef_top* top() { return top_.get(); }

  // One SPI byte, mode 0, MSB first. MOSI changes while SCK is low; MISO is sampled at the
  // end of the high phase (the host's falling edge).
  uint8_t Xfer(uint8_t out, std::vector<int>* bits = nullptr) {
    uint8_t in = 0;
    for (int b = 7; b >= 0; b--) {
      top_->spi_mosi_i = (out >> b) & 1;
      Ticks(ratio_ / 2);
      top_->spi_sck_i = 1;
      Ticks(ratio_ / 2);
      int bit = top_->spi_miso_o & 1;
      in = (in << 1) | bit;
      if (bits) bits->push_back(bit);
      top_->spi_sck_i = 0;
    }
    return in;
  }
  void CsLow() {
    top_->spi_csn_i = 0;
    Ticks(ratio_);
  }
  void CsHigh() {
    Ticks(ratio_);
    top_->spi_csn_i = 1;
    Ticks(4 * ratio_);
  }
  void Header(uint8_t op, uint32_t addr, uint16_t beats) {
    uint16_t len = beats - 1;
    const uint8_t h[7] = {op,
                          uint8_t(addr >> 24),
                          uint8_t(addr >> 16),
                          uint8_t(addr >> 8),
                          uint8_t(addr),
                          uint8_t(len >> 8),
                          uint8_t(len)};
    for (uint8_t x : h) Xfer(x);
  }

  void WriteLines(uint32_t addr, const uint8_t* data, uint16_t beats) {
    CsLow();
    Header(0x02, addr, beats);
    for (int i = 0; i < beats * 16; i++) Xfer(data[i]);
    for (int i = 0; i < 16; i++) Xfer(0);  // drain, as nexus_loader does
    CsHigh();
  }

  bool ReadLines(uint32_t addr, uint16_t beats, uint8_t* out) {
    std::vector<int> bits;
    CsLow();
    Header(0x01, addr, beats);
    const int window = beats * 17 + 64;
    for (int i = 0; i < window; i++) Xfer(0, &bits);
    CsHigh();
    size_t pos = 0;
    int beat = 0;
    uint32_t sync = 0;
    while (beat < beats && pos < bits.size()) {
      sync = (sync << 1) | bits[pos++];
      if ((sync & 0xFF) == 0xFE) {
        for (int i = 0; i < 16; i++) {
          uint8_t v = 0;
          for (int k = 0; k < 8 && pos < bits.size(); k++) v = (v << 1) | bits[pos++];
          out[beat * 16 + i] = v;
        }
        beat++;
        sync = 0;
      }
    }
    return beat == beats;
  }

  // 32-bit register access. The word sits in its natural lane of the 16-byte line,
  // and the AXI address is the register's own address.
  void Write32(uint32_t addr, uint32_t v) {
    uint8_t line[16] = {};
    std::memcpy(&line[addr & 0xF], &v, 4);  // only lanes at/after addr are strobed
    WriteLines(addr, line, 1);
  }
  uint32_t Read32(uint32_t addr) {
    uint8_t line[16] = {};
    bool ok = ReadLines(addr & ~0xFu, 1, line);
    CHECK(ok, "no sync token reading 0x%08x", addr);
    uint32_t v;
    std::memcpy(&v, &line[addr & 0xF], 4);
    return v;
  }

 private:
  int ratio_;
  std::unique_ptr<Vreef_top> top_;
  uint64_t cycles_ = 0;
};

std::vector<uint8_t> LoadBin(const char* path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<uint8_t> v((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  v.resize((v.size() + 15) & ~size_t(15), 0);
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s program.bin result_addr_hex [sck_ratio]\n", argv[0]);
    return 2;
  }
  const std::vector<uint8_t> prog = LoadBin(argv[1]);
  const uint32_t result_addr = std::strtoul(argv[2], nullptr, 16);
  const int ratio = argc > 3 ? std::atoi(argv[3]) : 16;
  const bool buserr = argc > 4 && std::strcmp(argv[4], "buserr") == 0;
  std::printf("program %zu bytes, result @0x%08x, f_clk/f_sck = %d, mode %s\n", prog.size(),
              result_addr, ratio, buserr ? "buserr" : "pass");

  Tb tb(ratio);
  tb.Reset();

  // 1. CSR reset state: RESET = 3 (reset + clock gate), STATUS = 0.
  uint32_t rst = tb.Read32(kCsrBase + 0x0);
  CHECK(rst == 3, "RESET after reset = 0x%x, want 3", rst);
  CHECK(!tb.top()->halted_o && !tb.top()->fault_o, "halted/fault set while in reset");
  std::printf("[1] CSR RESET = %u\n", rst);

  // 2. Program -> ITCM, read back.
  const int kChunkBeats = 32;
  const int beats = int(prog.size() / 16);
  for (int b = 0; b < beats; b += kChunkBeats) {
    int n = std::min(kChunkBeats, beats - b);
    tb.WriteLines(uint32_t(b * 16), &prog[b * 16], uint16_t(n));
  }
  std::vector<uint8_t> back(prog.size());
  for (int b = 0; b < beats; b += kChunkBeats) {
    int n = std::min(kChunkBeats, beats - b);
    CHECK(tb.ReadLines(uint32_t(b * 16), uint16_t(n), &back[b * 16]), "ITCM readback sync");
  }
  int mism = 0;
  for (size_t i = 0; i < prog.size(); i++) mism += back[i] != prog[i];
  CHECK(mism == 0, "ITCM readback: %d bytes differ", mism);
  std::printf("[2] ITCM load + readback: %d beats, %d mismatches\n", beats, mism);

  // 3. DTCM pattern, 4 beats in one frame.
  uint8_t pat[64], pat_back[64] = {};
  for (int i = 0; i < 64; i++) pat[i] = uint8_t(0xA5 ^ (i * 37));
  tb.WriteLines(kDtcmTestAddr, pat, 4);
  CHECK(tb.ReadLines(kDtcmTestAddr, 4, pat_back), "DTCM readback sync");
  CHECK(std::memcmp(pat, pat_back, 64) == 0, "DTCM readback mismatch");
  std::printf("[3] DTCM 4-beat write/read: %s\n",
              std::memcmp(pat, pat_back, 64) == 0 ? "ok" : "MISMATCH");

  // 4. PC_START is at CSR +0x4: write a non-zero value, read it back, then the real one.
  tb.Write32(kCsrBase + 0x4, 0x40);
  uint32_t pc = tb.Read32(kCsrBase + 0x4);
  CHECK(pc == 0x40, "PC_START readback 0x%x, want 0x40", pc);
  tb.Write32(kCsrBase + 0x4, 0x0);
  pc = tb.Read32(kCsrBase + 0x4);
  CHECK(pc == 0x0, "PC_START readback 0x%x, want 0", pc);
  CHECK(tb.Read32(kCsrBase + 0x0) == 3, "RESET changed by PC_START write");
  std::printf("[4] PC_START write/readback over SPI: ok\n");

  // 5. Release clock gate, then reset (upstream execute_from()).
  tb.Write32(kCsrBase + 0x0, 1);
  tb.Write32(kCsrBase + 0x0, 0);
  const uint64_t start = tb.cycles();

  // 6. Run.
  const uint64_t kTimeout = 2000000;
  while (!tb.top()->halted_o && tb.cycles() - start < kTimeout) tb.Tick();
  const uint64_t run = tb.cycles() - start;
  CHECK(tb.top()->halted_o, "core did not halt within %llu cycles", (unsigned long long)kTimeout);
  CHECK(tb.top()->fault_o == buserr, "fault_o = %d, want %d", tb.top()->fault_o, buserr);
  std::printf("[6] halted=%d fault=%d (halt seen %llu cycles after the RESET=0 frame)\n",
              tb.top()->halted_o, tb.top()->fault_o, (unsigned long long)run);
  uint32_t status = tb.Read32(kCsrBase + 0x8);
  const uint32_t want_status = buserr ? 3 : 1;
  CHECK(status == want_status, "STATUS = 0x%x, want 0x%x", status, want_status);

  // 7. Results.
  uint8_t r[32];
  CHECK(tb.ReadLines(result_addr, 2, r), "result readback sync");
  uint32_t w[8];
  std::memcpy(w, r, 32);
  const char* names[] = {"magic", "vadd_sum", "fsum_bits", "trap_cause",
                         "scalar_mul", "scalar_fp", "vl_max", "cycles"};
  const uint32_t want[] = {buserr ? 0xBAD00BADu : 0x600DCAFEu, 8128, 0x44FC0000 /* 2016.0f */,
                           buserr ? 5u : 0u, uint32_t(1234 * -567), 0x40600000 /* 3.5f */, 4, 0};
  for (int i = 0; i < 7; i++) {
    std::printf("    %-10s = 0x%08x (want 0x%08x)%s\n", names[i], w[i], want[i],
                w[i] == want[i] ? "" : "  <-- MISMATCH");
    CHECK(w[i] == want[i], "%s", names[i]);
  }

  if (!buserr) {
    CHECK(w[7] > 0 && w[7] < 100000, "cycles = %u", w[7]);
    std::printf("    main() body: %u core cycles (mcycle)\n", w[7]);
  }
  std::printf("%s: %d failure(s), %llu cycles total\n", g_fail ? "SMOKE FAIL" : "SMOKE PASS",
              g_fail, (unsigned long long)tb.cycles());
  return g_fail ? 1 : 0;
}
