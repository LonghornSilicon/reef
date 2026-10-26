"use client";

import type { ReactNode } from "react";
import type { Status } from "@/lib/spec/types";
import type { Result } from "@/lib/rtl/types";

/** Segmented control. Cells are equal width so the indicator slides by its own width (transform only). */
export function Seg<T extends string>({ value, options, onChange, label }: { value: T; options: Array<{ value: T; label: ReactNode }>; onChange: (v: T) => void; label: string }) {
  const i = Math.max(0, options.findIndex((o) => o.value === value));
  return (
    <div className="seg" role="radiogroup" aria-label={label} style={{ ["--n" as string]: options.length }}>
      <span className="seg-ind" style={{ ["--i" as string]: i }} />
      {options.map((o) => (
        <button key={o.value} type="button" role="radio" aria-checked={o.value === value} className={o.value === value ? "on" : ""} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export const RESULT_LABEL: Record<Result, string> = { match: "match", different: "different", missing: "missing in RTL", unchecked: "not checked" };

export function ResultDot({ result, small }: { result: Result; small?: boolean }) {
  return <span className={`dot k-${result}${small ? " sm" : ""}`} title={RESULT_LABEL[result]} aria-label={RESULT_LABEL[result]} />;
}

export function StatusChip({ status }: { status: Status }) {
  return <span className={`chip status st-${status}`}>{status}</span>;
}

export function StatusDot({ status }: { status: Status }) {
  return <span className={`sdot st-${status}`} title={status} />;
}

export function Empty({ icon = "∅", title, children, action }: { icon?: string; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      <div className="empty-icon" aria-hidden>{icon}</div>
      <b>{title}</b>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

export function Note({ kind, children, action }: { kind: "err" | "warn" | "info"; children: ReactNode; action?: ReactNode }) {
  return (
    <div className={`note note-${kind}`} role={kind === "err" ? "alert" : "status"}>
      <div className="grow">{children}</div>
      {action}
    </div>
  );
}

export function SourceLink({ file, line, href }: { file: string; line?: number; href?: string | null }) {
  const text = `${file}${line ? `:${line}` : ""}`;
  return href ? <a className="src" href={href} target="_blank" rel="noreferrer">{text} ↗</a> : <span className="src muted" title="Sample data: no public source">{text}</span>;
}
