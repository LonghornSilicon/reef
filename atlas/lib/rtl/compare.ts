import { RTL_CHECKED, type SpecModel } from "../spec/types";
import { expectedSignals, IGNORED_RTL_PORTS, presencePrefixes } from "./names";
import type { BlockReport, CompareReport, PortReport, Result, RtlExtract, RtlPort } from "./types";

const worst = (results: Result[]): Result =>
  results.includes("missing") ? "missing" : results.includes("different") ? "different" : results.length && results.every((r) => r === "unchecked") ? "unchecked" : "match";

/**
 * Compare a v2 spec with an RTL extract. Pure and deterministic: the same two
 * documents always produce the same report (plan.md principle 2).
 */
export function compare(spec: SpecModel, extract: RtlExtract, staleBy = 0): CompareReport {
  const blocks: Record<string, BlockReport> = {};

  for (const block of Object.values(spec.blocks)) {
    const specPorts = spec.ports[block.id] ?? [];
    const checked = RTL_CHECKED.has(block.status);
    const mod = block.rtl ? extract.modules[block.rtl.module] : undefined;
    const ports: Record<string, PortReport> = {};
    const consumed = new Set<string>();

    for (const port of specPorts) {
      const base = { status: port.status, warnOnly: port.status === "modified" };
      if (!RTL_CHECKED.has(port.status) || !checked) {
        // Not checked yet, but RTL signals that already exist for it are not "extra".
        if (mod) {
          const prefixes = presencePrefixes(port);
          for (const p of mod.ports) if (prefixes.some((pre) => p.name.startsWith(pre))) consumed.add(p.name);
        }
        ports[port.id] = { ...base, result: "unchecked", expected: [], found: [], diffs: [] };
        continue;
      }
      if (!mod) {
        ports[port.id] = { ...base, result: "missing", expected: [], found: [], diffs: [{ signal: "*", rtlName: block.rtl?.module ?? "?", kind: "absent" }] };
        continue;
      }
      const iface = port.interface ? spec.interfaces[port.interface] : undefined;
      const expected = expectedSignals(port, iface, iface ? spec.protocols[iface.protocol] : undefined);
      if (expected) {
        const found: RtlPort[] = [];
        const diffs: PortReport["diffs"] = [];
        for (const e of expected) {
          const actual = mod.ports.find((p) => p.name === e.rtlName);
          if (!actual) { diffs.push({ signal: e.signal, rtlName: e.rtlName, kind: "absent" }); continue; }
          consumed.add(actual.name);
          found.push(actual);
          if (e.width !== undefined && actual.width !== e.width) diffs.push({ signal: e.signal, rtlName: e.rtlName, kind: "width", expected: e.width, actual: actual.width });
          if (e.dir && actual.dir !== e.dir) diffs.push({ signal: e.signal, rtlName: e.rtlName, kind: "dir", expected: e.dir, actual: actual.dir });
        }
        const result: Result = !found.length ? "missing" : diffs.length ? "different" : "match";
        ports[port.id] = { ...base, result, expected, found, diffs };
      } else {
        const prefixes = presencePrefixes(port);
        const found = mod.ports.filter((p) => prefixes.some((pre) => p.name === pre || p.name.startsWith(pre.endsWith("_") ? pre : `${pre}_`)));
        found.forEach((p) => consumed.add(p.name));
        ports[port.id] = { ...base, result: found.length ? "match" : "missing", expected: [], found, diffs: found.length ? [] : [{ signal: "*", rtlName: `${prefixes[0]}*`, kind: "absent" }] };
      }
    }

    const extraRtlPorts = mod ? mod.ports.filter((p) => !consumed.has(p.name) && !IGNORED_RTL_PORTS.has(p.name)) : [];
    let result: Result;
    if (!checked) result = "unchecked";
    else if (!mod) result = "missing";
    else result = worst(Object.values(ports).map((p) => p.result).filter((r) => r !== "unchecked")) ;
    if (checked && mod && Object.values(ports).every((p) => p.result === "unchecked")) result = "match";
    blocks[block.id] = { status: block.status, module: block.rtl?.module ?? null, result, sv: mod?.sv, src: mod?.src, doc: mod?.doc, ports, extraRtlPorts };
  }

  const connections: CompareReport["connections"] = {};
  for (const c of spec.connections) {
    if (!RTL_CHECKED.has(c.status)) { connections[c.id] = { result: "unchecked", status: c.status, mode: "endpoints-only", warnOnly: false }; continue; }
    const a = blocks[c.from]?.ports[c.fromPort];
    const b = blocks[c.to]?.ports[c.toPort];
    const results = [a?.result ?? "missing", b?.result ?? "missing"].map((r) => (r === "unchecked" ? "missing" : r));
    // A difference that comes only from a `modified` endpoint is a warning on the wire too.
    const warnOnly = c.status === "modified" || [a, b].some((p) => p?.warnOnly && p.result !== "match");
    connections[c.id] = { result: worst(results), status: c.status, mode: "endpoints-only", warnOnly };
  }

  const summary: Record<Result, number> = { match: 0, different: 0, missing: 0, unchecked: 0 };
  let failing = 0;
  const tally = (r: Result, gating: boolean) => { summary[r] += 1; if (gating && (r === "different" || r === "missing")) failing += 1; };
  for (const b of Object.values(blocks)) {
    for (const p of Object.values(b.ports)) tally(p.result, !p.warnOnly);
    if (!Object.keys(b.ports).length) tally(b.result, b.status !== "modified");
  }
  for (const c of Object.values(connections)) tally(c.result, !c.warnOnly);

  return {
    schema: "atlas.compare/1",
    spec: { branch: spec.source.branch, commit: spec.source.commit },
    rtl: { commit: extract.commit, staleBy, tools: extract.tools },
    blocks, connections, summary, failing,
  };
}
