import type { Maturity } from "@/lib/model/schema";

/**
 * Lenses replace the incumbent viewer's five top-level views.
 *
 * They are additive state over one canvas rather than destinations, so
 * switching a lens never resets your focus or selection — which is the
 * specific failure PRODUCT.md says success depends on avoiding.
 */
export type Lens =
  | { kind: "structure" }
  | { kind: "protocol"; protocol: string }
  | { kind: "readiness" }
  | { kind: "sequence"; sequenceId: string; step: number };

export const MATURITY_ORDER: Maturity[] = ["proposal", "specified", "in-progress", "implemented", "verified"];

/** Maturity encodes a named engineering state, so it always ships with a legend. */
export const MATURITY_COLOR: Record<Maturity, string> = {
  proposal: "var(--amber)",
  specified: "var(--blue)",
  "in-progress": "var(--purple)",
  implemented: "var(--green)",
  verified: "var(--teal)",
};

export const MATURITY_LABEL: Record<Maturity, string> = {
  proposal: "Proposal",
  specified: "Specified",
  "in-progress": "In progress",
  implemented: "Implemented",
  verified: "Verified",
};

export function lensFromParams(params: URLSearchParams): Lens {
  const kind = params.get("lens");
  if (kind === "protocol") return { kind: "protocol", protocol: params.get("protocol") ?? "" };
  if (kind === "readiness") return { kind: "readiness" };
  if (kind === "sequence") {
    const step = Number.parseInt(params.get("step") ?? "0", 10);
    return { kind: "sequence", sequenceId: params.get("seq") ?? "", step: Number.isFinite(step) ? Math.max(0, step) : 0 };
  }
  return { kind: "structure" };
}

export function lensToParams(lens: Lens): URLSearchParams {
  const params = new URLSearchParams();
  if (lens.kind === "structure") return params;
  params.set("lens", lens.kind);
  if (lens.kind === "protocol") params.set("protocol", lens.protocol);
  if (lens.kind === "sequence") {
    params.set("seq", lens.sequenceId);
    params.set("step", String(lens.step));
  }
  return params;
}

export function lensHref(blockId: string, lens: Lens): string {
  const query = lensToParams(lens).toString();
  return `/architecture/${blockId}${query ? `?${query}` : ""}`;
}
