// Generates the sample RTL extract and simulation runs under fixtures/.
// Deterministic: the same seed always writes the same files. Run with `npm run fixtures`.
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";

const root = path.join(path.dirname(new URL(import.meta.url).pathname), "..", "fixtures");
const write = (rel, data) => {
  const file = path.join(root, rel);
  mkdirSync(path.dirname(file), { recursive: true });
  writeFileSync(file, typeof data === "string" ? data : JSON.stringify(data, null, 1) + "\n");
};

// ---------------------------------------------------------------- RTL extract
const SHA = "3f9c2a1";
const bus = (prefix, dirIn = "in", dirOut = "out") => [
  { name: `io_${prefix}_req_valid`, dir: dirOut, width: 1 },
  { name: `io_${prefix}_req_ready`, dir: dirIn, width: 1 },
  { name: `io_${prefix}_req_bits_addr`, dir: dirOut, width: 32 },
  { name: `io_${prefix}_rsp_valid`, dir: dirIn, width: 1 },
  { name: `io_${prefix}_rsp_bits_data`, dir: dirIn, width: 128 },
];
const clk = [{ name: "clock", dir: "in", width: 1 }, { name: "reset", dir: "in", width: 1 }];
let line = 1;
const mod = (sv, scala, doc, ports) => {
  const svLine = line; line += 40 + ports.length * 3;
  return {
    sv: { file: `gen/${sv}.sv`, line: svLine % 37 + 1 },
    src: { file: `hdl/chisel/src/coral/${scala}.scala`, line: (svLine * 7) % 180 + 12 },
    doc,
    ports: [...clk, ...ports].map((p, i) => ({ ...p, src: { file: `hdl/chisel/src/coral/${scala}.scala`, line: (svLine * 7) % 180 + 20 + i } })),
  };
};
const extract = {
  schema: "atlas.rtl-extract/1",
  commit: SHA,
  tools: { firtool: "1.86.0", slang: "8.1", yosys: "0.48" },
  top: "CoralCore",
  modules: {
    CoralCore: mod("CoralCore", "CoralCore", "Top of the Coral core: TCMs, compute cluster, AXI master and slave.", [...bus("dma", "out", "in"), ...bus("axi")]),
    CoreAxiSlave: mod("CoreAxiSlave", "axi/CoreAxiSlave", "AXI4 slave port into the core.", bus("fabric")),
    AxiToFabric: mod("AxiToFabric", "axi/AxiToFabric", "Converts inbound AXI bursts into fabric beats.", [...bus("axi", "out", "in"), ...bus("fabric")]),
    FabricMux: mod("FabricMux", "fabric/FabricMux", "Routes inbound fabric requests to DTCM or ITCM by address.", [...bus("ingress", "out", "in"), ...bus("dtcm"), ...bus("itcm")]),
    ComputeCluster: mod("ComputeCluster", "ComputeCluster", "Scalar core, vector engine and systolic array.", [...bus("dbus"), ...bus("ibus"), ...bus("ebus")]),
    SCore: mod("SCore", "scalar/SCore", "In-order scalar core. Dispatches vector commands through a Decoupled port.", [
      ...bus("fetch"),
      { name: "io_vcmd_valid", dir: "out", width: 1 },
      { name: "io_vcmd_ready", dir: "in", width: 1 },
      { name: "io_vcmd_bits", dir: "out", width: 64 },
      { name: "io_array_valid", dir: "out", width: 1 },
    ]),
    VectorEngine: mod("VectorEngine", "vector/VectorEngine", "Vector engine: command queue, VALU, VMAC and the vector register file.", [
      { name: "io_cmd_valid", dir: "in", width: 1 },
      { name: "io_cmd_ready", dir: "out", width: 1 },
      { name: "io_cmd_bits", dir: "in", width: 32 },
      { name: "io_dbg_sel", dir: "in", width: 4 },
    ]),
    SystolicArray: mod("SystolicArray", "systolic/SystolicArray", "16×16 weight-stationary array (in progress).", [{ name: "io_command_valid", dir: "in", width: 1 }]),
    CoreMasterArbiter: mod("CoreMasterArbiter", "axi/CoreMasterArbiter", "Arbitrates uncached requests onto the AXI master.", [...bus("core", "out", "in"), ...bus("axi")]),
    CoreAxiMaster: mod("CoreAxiMaster", "axi/CoreAxiMaster", "AXI4 master out of the core.", bus("core", "out", "in")),
    IBusMux: mod("IBusMux", "mem/IBusMux", "Selects the instruction fetch target.", [...bus("core", "out", "in"), ...bus("itcm")]),
    DtcmArbiter: mod("DtcmArbiter", "mem/DtcmArbiter", "Arbitrates core and fabric access to the DTCM banks.", [...bus("fabric", "out", "in"), ...bus("core", "out", "in"), ...bus("memory")]),
    Dtcm: mod("Dtcm", "mem/Dtcm", "Four-bank data TCM.", bus("dbus", "out", "in")),
    ItcmArbiter: mod("ItcmArbiter", "mem/ItcmArbiter", "Arbitrates fetch and fabric access to the ITCM.", [...bus("fabric", "out", "in"), ...bus("core", "out", "in"), ...bus("memory")]),
    Itcm: mod("Itcm", "mem/Itcm", "Instruction TCM.", bus("ibus", "out", "in")),
    VectorCommandQueue: mod("VectorCommandQueue", "vector/VectorCommandQueue", "Four-entry Chisel Queue in front of vector issue.", [
      { name: "io_enq_valid", dir: "in", width: 1 }, { name: "io_enq_ready", dir: "out", width: 1 }, { name: "io_enq_bits", dir: "in", width: 64 },
      { name: "io_deq_valid", dir: "out", width: 1 }, { name: "io_deq_ready", dir: "in", width: 1 }, { name: "io_count", dir: "out", width: 3 },
    ]),
    VAlu: mod("VAlu", "vector/VAlu", "Vector integer ALU, two-cycle latency.", []),
    VRegFile: mod("VRegFile", "vector/VRegFile", "Vector register file with two read ports and one write port.", []),
    Fetch: mod("Fetch", "scalar/Fetch", "Instruction fetch.", bus("decode")),
    Decode: mod("Decode", "scalar/Decode", "Instruction decode.", bus("fetch", "out", "in")),
  },
};
rmSync(path.join(root, "rtl"), { recursive: true, force: true });
write(`rtl/${SHA}/extract.json`, extract);

// ---------------------------------------------------------------- runs
function prng(seed) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 2 ** 32; };
}
const OPS = ["vadd.vv", "vmul.vv", "vsub.vv", "vand.vv", "vmacc.vv", "vsll.vi", "vmax.vv", "vredsum.vs"];

/** Cycle model of fetch → decode → VCQ(4) → issue → VRF → VALU(2) → WB. `dropAt` injects the no-drop bug. */
function simulate({ seed, cycles, commands, dropAt }) {
  const rnd = prng(seed);
  const sig = {};
  const set = (name, width, c, v) => {
    const t = (sig[name] ??= { width, changes: [] });
    const last = t.changes.at(-1);
    if (!last || last[1] !== v) t.changes.push([c, v]);
  };
  const events = [];
  const ev = (cycle, block, type, details) => events.push({ cycle, block, type, details });
  const tx = [];
  const queue = [];
  let decode = null, nextFetch = 12, issued = 0, issueBusyUntil = -1, dropped = false, stalled = false;
  // VRF read ports are taken by a scalar-to-vector move burst in these windows, which backs the queue up.
  const busy = [[180, 236], [610, 690], [1040, 1072], [1490, 1560]];
  const vrfBusy = (c) => busy.some(([a, b]) => c >= a && c < b);

  for (let c = 0; c < cycles; c++) {
    // Fetch a new vector command into decode.
    if (!decode && issued < commands && c >= nextFetch) {
      const n = issued++;
      const op = OPS[Math.floor(rnd() * OPS.length)];
      const vd = 1 + Math.floor(rnd() * 30), vs = 1 + Math.floor(rnd() * 30);
      const t = { id: `i${n}`, label: `${op} v${vd}, v${vs}, v${(vs + 3) % 32}`, pc: `0x${(0x8000 + n * 4 * 3).toString(16)}`, stages: { fetch: [c, c] }, bits: (0x1000_0000 + n * 0x10_0001 + vd * 0x100 + vs) };
      tx.push(t);
      decode = { t, ready: c + 1 };
      t.stages.decode = [c + 1, c + 1];
      nextFetch = c + 2 + Math.floor(rnd() * (rnd() < 0.25 ? 14 : 4));
      ev(c, "scalar-fetch", "fetch", `${t.pc} ${t.label}`);
    }
    const enqValid = !!decode && c >= decode.ready;
    const enqReady = queue.length < 4;
    const deqValid = queue.length > 0;
    const deqReady = !vrfBusy(c) && c > issueBusyUntil;

    let validOut = enqValid ? 1 : 0;
    // The bug: decode drops the command it is offering when the queue has been full for 6 cycles.
    if (dropAt !== undefined && !dropped && enqValid && !enqReady && c >= dropAt) {
      validOut = 0;
      dropped = true;
      decode.t.dropped = true;
      decode.t.stages.decode[1] = c;
      ev(c, "scalar-decode", "drop", `${decode.t.label} lost: vcmd.valid fell while vcmd.ready was low`);
      decode = null;
    }
    set("score.io_vcmd_valid", 1, c, validOut);
    set("score.io_vcmd_ready", 1, c, enqReady ? 1 : 0);
    set("score.io_vcmd_bits", 64, c, decode ? decode.t.bits : (sig["score.io_vcmd_bits"]?.changes.at(-1)?.[1] ?? 0));
    set("vcq.io_deq_valid", 1, c, deqValid ? 1 : 0);
    set("vcq.io_deq_ready", 1, c, deqReady ? 1 : 0);
    set("vcq.io_count", 3, c, queue.length);
    set("vrf.io_rport_busy", 1, c, vrfBusy(c) ? 1 : 0);

    if (deqValid && deqReady) {
      const t = queue.shift();
      t.stages.vcq[1] = c;
      t.stages.issue = [c + 1, c + 1];
      t.stages.vrf = [c + 2, c + 2];
      t.stages.valu = [c + 3, c + 4];
      t.stages.wb = [c + 5, c + 5];
      issueBusyUntil = c;
      ev(c, "vector-command-queue", "deq", `${t.id} → issue (${queue.length} left)`);
      ev(c + 1, "vector-engine", "issue", t.label);
      ev(c + 5, "vector-register-file", "retire", `${t.id} ${t.label.split(" ")[0]} wrote back`);
    }
    if (validOut && enqReady && decode) {
      const t = decode.t;
      t.stages.decode[1] = c;
      t.stages.vcq = [c + 1, c + 1];
      queue.push(t);
      ev(c, "vector-command-queue", "enq", `${t.id} ${t.label} (bits 0x${t.bits.toString(16)})`);
      if (queue.length === 4) ev(c, "vector-command-queue", "full", "4/4 entries; decode will stall");
      decode = null;
    } else if (decode && enqValid && !enqReady) {
      decode.t.stages.decode[1] = c;
      if (!stalled) ev(c, "scalar-decode", "stall", `${decode.t.id} waiting: VCQ full`);
      stalled = true;
    }
    if (!(decode && enqValid && !enqReady)) stalled = false;
    for (const t of queue) t.stages.vcq[1] = Math.max(t.stages.vcq[1], c);
  }
  const retired = tx.filter((t) => t.stages.wb && t.stages.wb[1] < cycles).length;
  const expected = tx.length;
  if (retired !== expected) ev(cycles - 1, "testbench", "error", `scoreboard: expected ${expected} retired commands, saw ${retired}`);
  else ev(cycles - 1, "testbench", "pass", `scoreboard: ${retired} commands retired in order`);
  events.sort((a, b) => a.cycle - b.cycle);
  return { sig, events, tx: tx.map(({ bits, ...t }) => t), retired, expected };
}

rmSync(path.join(root, "runs"), { recursive: true, force: true });
const BR = "atlas/vector-dispatch-64b";
const full = ["run.json", "events.jsonl", "signals.json", "transactions.json"];
const runs = [
  { id: "212", seed: 7, cycles: 1800, commands: 180, dropAt: 640, test: "test_vector_dispatch_burst", startedAt: "2026-10-26T09:42:00Z", durationS: 214 },
  { id: "211", seed: 7, cycles: 1800, commands: 180, test: "test_vector_dispatch_burst", startedAt: "2026-10-25T17:03:00Z", durationS: 209, commit: "b71e0c4" },
];
for (const r of runs) {
  const s = simulate(r);
  const failed = s.retired !== s.expected;
  write(`runs/${r.id}/run.json`, {
    id: r.id, branch: BR, commit: r.commit ?? SHA, test: r.test, simulator: "verilator 5.030 · cocotb 1.9.2", seed: r.seed,
    cycles: r.cycles, dirty: false, status: failed ? "failed" : "passed", startedAt: r.startedAt, durationS: r.durationS, files: full,
    summary: failed ? `${s.expected - s.retired} vector command dropped under back-pressure` : `${s.retired} vector commands retired`,
  });
  write(`runs/${r.id}/events.jsonl`, s.events.map((e) => JSON.stringify(e)).join("\n") + "\n");
  write(`runs/${r.id}/signals.json`, { schema: "atlas.signals/1", cycles: r.cycles, signals: s.sig });
  write(`runs/${r.id}/transactions.json`, { schema: "atlas.transactions/1", process: "vector-dispatch", transactions: s.tx });
}
// 206: the upload stopped after the log.
const partial = simulate({ seed: 3, cycles: 1800, commands: 160 });
write("runs/206/run.json", { id: "206", branch: BR, commit: "a04d9e2", test: "test_vector_dispatch_random", simulator: "verilator 5.030 · cocotb 1.9.2", seed: 3, cycles: 1800, dirty: true, status: "incomplete", startedAt: "2026-10-24T11:20:00Z", durationS: 0, files: ["run.json", "events.jsonl"], summary: "Upload interrupted after events.jsonl" });
write("runs/206/events.jsonl", partial.events.slice(0, 220).map((e) => JSON.stringify(e)).join("\n") + "\n");
// Runs with metadata only (older, logs expired).
const meta = (id, extra) => write(`runs/${id}/run.json`, { id, simulator: "verilator 5.030 · cocotb 1.9.2", dirty: false, durationS: 180, files: ["run.json"], ...extra });
meta("210", { branch: BR, commit: "b71e0c4", test: "test_vcq_full", seed: 11, cycles: 900, status: "passed", startedAt: "2026-10-25T14:10:00Z", summary: "Logs expired after 14 days" });
meta("209", { branch: BR, commit: "b71e0c4", test: "test_tcm_arbiter", seed: 2, cycles: 4200, status: "passed", startedAt: "2026-10-25T10:55:00Z", summary: "Logs expired after 14 days" });
meta("213", { branch: BR, commit: SHA, test: "test_vector_dispatch_random", seed: 19, cycles: 0, status: "running", startedAt: "2026-10-26T10:05:00Z", durationS: 0, summary: "Running on the lab server" });
meta("208", { branch: "atlas/axi-burst-fix", commit: "9e1d07b", test: "test_axi_burst_wrap", seed: 5, cycles: 2600, status: "failed", startedAt: "2026-10-26T08:12:00Z", summary: "AXI WRAP burst crossed a 4 KiB boundary" });
console.log("fixtures written to", root);
