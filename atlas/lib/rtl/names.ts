import type { SpecInterface, SpecPort, SpecProtocol } from "../spec/types";
import type { ExpectedSignal } from "./types";

/**
 * Spec port → the RTL signal names it should produce (plan.md §M1.5).
 *   decoupled: io_<prefix>_valid, io_<prefix>_ready, io_<prefix>_bits[_<sig>]
 *   flat:      <prefix>_<sig>
 * An explicit `rtl.map` entry always wins.
 */
export function rtlName(naming: SpecInterface["roleNaming"], prefix: string, signal: string): string {
  if (naming === "flat") return `${prefix}_${signal}`;
  if (signal === "valid" || signal === "ready" || signal === "bits") return `io_${prefix}_${signal}`;
  return `io_${prefix}_bits_${signal}`;
}

export function expectedSignals(port: SpecPort, iface: SpecInterface | undefined, protocol: SpecProtocol | undefined): ExpectedSignal[] | null {
  if (!iface || !protocol?.signals?.length) return null;
  const prefix = port.rtl?.prefix ?? port.id;
  const leader = protocol.roles[0];
  return protocol.signals.map((s) => {
    const width = typeof s.width === "number" ? s.width : iface.params[s.width.slice("param:".length)];
    // Direction is relative to the first role (e.g. producer): fwd signals leave it.
    const dir = port.role === undefined ? undefined : (s.dir === "fwd") === (port.role === leader) ? "out" : "in";
    return { signal: s.name, rtlName: port.rtl?.map?.[s.name] ?? rtlName(iface.roleNaming, prefix, s.name), width, dir };
  });
}

/** For ports without a signal-level interface: any RTL port under the prefix counts as presence. */
export function presencePrefixes(port: SpecPort): string[] {
  const prefix = port.rtl?.prefix ?? port.id;
  return [`io_${prefix}_`, `${prefix}_`, `io_${prefix}`];
}

export const IGNORED_RTL_PORTS = new Set(["clock", "reset", "clk", "rst", "rst_n"]);
