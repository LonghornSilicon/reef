import type { SpecModel } from "../spec/types";
import type { SignalsFile } from "../runs/types";
import { expectedSignals } from "../rtl/names";

export type SignalGroup = { title: string; keys: string[] };

/** Waveform groups: one per spec interface port with traced signals, then the rest by RTL instance. */
export function signalGroups(spec: SpecModel, signals: SignalsFile): SignalGroup[] {
  const used = new Set<string>();
  const groups: SignalGroup[] = [];
  for (const list of Object.values(spec.ports)) {
    for (const port of list) {
      const iface = port.interface ? spec.interfaces[port.interface] : undefined;
      const inst = spec.blocks[port.blockId]?.rtl?.instance;
      if (!iface || !inst) continue;
      const keys = (expectedSignals(port, iface, spec.protocols[iface.protocol]) ?? []).map((e) => `${inst}.${e.rtlName}`).filter((k) => signals.signals[k]);
      if (!keys.length) continue;
      keys.forEach((k) => used.add(k));
      groups.push({ title: `${port.blockId}.${port.id} · ${iface.id}`, keys });
    }
  }
  const byInst = new Map<string, string[]>();
  for (const k of Object.keys(signals.signals)) {
    if (used.has(k)) continue;
    const inst = k.split(".")[0];
    byInst.set(inst, [...(byInst.get(inst) ?? []), k]);
  }
  for (const [inst, keys] of byInst) {
    const block = Object.values(spec.blocks).find((b) => b.rtl?.instance === inst);
    groups.push({ title: block ? `${block.id} (${inst})` : inst, keys });
  }
  return groups;
}
