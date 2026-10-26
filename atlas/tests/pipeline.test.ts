import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { branchInfo, loadSpec } from "../lib/spec/load";
import { loadExtract } from "../lib/rtl/store";
import { compare } from "../lib/rtl/compare";
import { checkRun } from "../lib/sim/check";
import { parseExpr, evaluate, RuleSyntaxError } from "../lib/spec/rules";
import { rtlName } from "../lib/rtl/names";
import type { SignalsFile } from "../lib/runs/types";

const sample = branchInfo("atlas/vector-dispatch-64b");

describe("spec", () => {
  it("migrates main with every block planned", () => {
    const spec = loadSpec(branchInfo("main"));
    expect(Object.keys(spec.blocks).length).toBeGreaterThan(50);
    expect(Object.values(spec.blocks).every((b) => b.status === "planned")).toBe(true);
    expect(spec.connections.every((c) => c.status === "planned")).toBe(true);
  });
  it("applies the overlay: ports follow their block, wires take the lower status", () => {
    const spec = loadSpec(sample);
    expect(spec.blocks["vector-engine"].status).toBe("modified");
    expect(spec.ports["vector-engine"][0]).toMatchObject({ status: "modified", interface: "vcmd-64", role: "consumer" });
    expect(spec.connections.find((c) => c.id === "cluster-scalar-vector")?.status).toBe("implemented");
    expect(spec.connections.find((c) => c.id === "core-debug-dtcm")?.status).toBe("planned");
  });
});

describe("rules", () => {
  it("parses and evaluates", () => {
    const e = parseExpr("valid && !ready || $past(valid) == 1");
    const v: Record<string, number[]> = { valid: [1, 0], ready: [1, 1] };
    expect(evaluate(e, 0, (n, c) => v[n]?.[c] ?? 0)).toBe(0);
    expect(evaluate(e, 1, (n, c) => v[n]?.[c] ?? 0)).toBe(1);
  });
  it("rejects bad syntax", () => {
    expect(() => parseExpr("valid &&")).toThrow(RuleSyntaxError);
    expect(() => parseExpr("valid # ready")).toThrow(RuleSyntaxError);
  });
});

describe("compare", () => {
  const spec = loadSpec(sample);
  const { extract } = loadExtract(sample)!;
  const report = compare(spec, extract);

  it("names decoupled and flat signals", () => {
    expect(rtlName("decoupled", "cmd", "bits")).toBe("io_cmd_bits");
    expect(rtlName("decoupled", "cmd", "op")).toBe("io_cmd_bits_op");
    expect(rtlName("flat", "axi", "awvalid")).toBe("axi_awvalid");
  });
  it("finds the 32-bit command payload as a warning on a modified block", () => {
    const port = report.blocks["vector-engine"].ports.command;
    expect(port.result).toBe("different");
    expect(port.warnOnly).toBe(true);
    expect(port.diffs).toEqual([{ signal: "bits", rtlName: "io_cmd_bits", kind: "width", expected: 64, actual: 32 }]);
    expect(report.blocks["vector-engine"].extraRtlPorts.map((p) => p.name)).toEqual(["io_dbg_sel"]);
    expect(report.connections["cluster-scalar-vector"]).toMatchObject({ result: "different", warnOnly: true });
  });
  it("does not call RTL signals of a not-yet-checked port extra", () => {
    expect(report.blocks["scalar-core"].ports.array.result).toBe("unchecked");
    expect(report.blocks["scalar-core"].extraRtlPorts).toEqual([]);
  });
  it("fails on the missing DTCM debug port and nothing else", () => {
    expect(report.blocks["dtcm-arbiter"].ports.debug.result).toBe("missing");
    expect(report.blocks["scalar-core"].ports.vector.result).toBe("match");
    expect(report.blocks["core-debug"].result).toBe("unchecked");
    expect(report.failing).toBe(1);
  });
  it("is deterministic", () => {
    expect(JSON.stringify(compare(spec, extract))).toBe(JSON.stringify(report));
  });
});

describe("check", () => {
  const spec = loadSpec(sample);
  const load = (id: string) => JSON.parse(readFileSync(`fixtures/runs/${id}/signals.json`, "utf8")) as SignalsFile;
  it("finds the dropped command in run 212", () => {
    const r = checkRun(spec, load("212"));
    const errors = r.violations.filter((v) => v.severity === "error");
    expect(errors.map((v) => v.rule)).toContain("no-drop");
    expect(errors.find((v) => v.rule === "no-drop")?.cycle).toBe(640);
    expect(r.untraced).toEqual(["vector-engine.command"]);
  });
  it("passes run 211", () => {
    expect(checkRun(spec, load("211")).violations.filter((v) => v.severity === "error")).toEqual([]);
  });
});
