import { RTL_CHECKED, type SpecModel, type TimingRule } from "../spec/types";
import { evaluate, identifiers, parseExpr, type Expr } from "../spec/rules";
import { expectedSignals } from "../rtl/names";
import type { SignalsFile } from "../runs/types";
import { valueAt } from "./signals";

export type Violation = {
  rule: string;
  protocol: string;
  port: string;
  severity: TimingRule["severity"];
  /** Cycle the rule failed on. */
  cycle: number;
  /** Cycle the triggering condition started. */
  from: number;
  message: string;
  /** Trace keys the rule reads, for the waveform view. */
  signals: string[];
};

export type CheckResult = {
  violations: Violation[];
  /** Ports with timing rules whose signals are not in signals.json. */
  untraced: string[];
  checked: Array<{ port: string; rules: number }>;
};

/**
 * Evaluate every protocol timing rule over a run's signals.json (plan.md §M4).
 * A port is checked when its interface's protocol has rules and its block names an RTL instance.
 * Deterministic: same spec + same signals → same violations.
 */
export function checkRun(spec: SpecModel, signals: SignalsFile): CheckResult {
  const violations: Violation[] = [];
  const untraced: string[] = [];
  const checked: CheckResult["checked"] = [];

  for (const list of Object.values(spec.ports)) {
    for (const port of list) {
      const iface = port.interface ? spec.interfaces[port.interface] : undefined;
      const protocol = iface ? spec.protocols[iface.protocol] : undefined;
      const block = spec.blocks[port.blockId];
      if (!iface || !protocol?.timingRules?.length || !RTL_CHECKED.has(port.status)) continue;
      const portKey = `${port.blockId}.${port.id}`;
      const instance = block.rtl?.instance;
      const expected = expectedSignals(port, iface, protocol) ?? [];
      const keyOf = Object.fromEntries(expected.map((e) => [e.signal, `${instance}.${e.rtlName}`]));
      if (!instance || !expected.every((e) => signals.signals[keyOf[e.signal]])) { untraced.push(portKey); continue; }
      const read = (name: string, cycle: number) => (cycle < 0 ? 0 : valueAt(signals.signals[keyOf[name]], cycle));
      checked.push({ port: portKey, rules: protocol.timingRules.length });

      for (const rule of protocol.timingRules) {
        const when = parseExpr(rule.when);
        const then: Expr | null = rule.then ? parseExpr(rule.then) : null;
        const used = [...new Set([...identifiers(when), ...(then ? identifiers(then) : []), ...(rule.signals ?? [])])];
        const push = (cycle: number, from: number, message: string) =>
          violations.push({ rule: rule.id, protocol: protocol.id, port: portKey, severity: rule.severity, cycle, from, message, signals: used.map((s) => keyOf[s]).filter(Boolean) });

        for (let c = 0; c < signals.cycles; c++) {
          if (!evaluate(when, c, read)) continue;
          if (rule.kind === "implies" && then && !evaluate(then, c, read)) push(c, c, `${rule.then} was low while ${rule.when}`);
          if (rule.kind === "next" && then && c + 1 < signals.cycles && !evaluate(then, c + 1, read)) push(c + 1, c, `${rule.then} fell at ${c + 1} after ${rule.when} at ${c}`);
          if (rule.kind === "stable_while" && c + 1 < signals.cycles) {
            for (const s of rule.signals ?? []) {
              if (read(s, c + 1) !== read(s, c)) push(c + 1, c, `${s} changed while ${rule.when}`);
            }
          }
          if (rule.kind === "within" && then) {
            // Report once per episode: only from the cycle the condition started.
            if (c > 0 && evaluate(when, c - 1, read) && !evaluate(then, c - 1, read)) continue;
            const limit = rule.cycles ?? 1;
            let ok = false;
            for (let k = c; k <= c + limit && k < signals.cycles; k++) if (evaluate(then, k, read)) { ok = true; break; }
            if (!ok && c + limit < signals.cycles) push(c + limit, c, `${rule.then} did not rise within ${limit} cycles of ${rule.when}`);
          }
        }
      }
    }
  }
  violations.sort((a, b) => a.cycle - b.cycle || a.rule.localeCompare(b.rule));
  return { violations, untraced, checked };
}
