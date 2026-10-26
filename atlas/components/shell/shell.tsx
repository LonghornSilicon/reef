"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState, useTransition, type ReactNode } from "react";
import { hasFile, useSession } from "./session";
import { listKeys, usePopover } from "./popover";
import { when } from "@/lib/format";

const NAV = [
  { href: "/canvas", label: "Canvas" },
  { href: "/spec", label: "Spec" },
  { href: "/runs", label: "Runs" },
  { href: "/process", label: "Process" },
  { href: "/workspace", label: "Workspace" },
];

export function Shell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const router = useRouter();
  const { query, setCycle } = useSession();
  const [pending, start] = useTransition();
  const go = (href: string) => start(() => router.push(href));

  // ← / → step the shared cycle anywhere; Shift steps by 10.
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (e.metaKey || e.ctrlKey || e.altKey || t.closest("input, textarea, select, [contenteditable]")) return;
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        if (t.closest(".viewport")) return;
        e.preventDefault();
        const d = (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 10 : 1);
        setCycle((c) => c + d);
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [setCycle]);

  return (
    <>
      <div className={`progress${pending ? " on" : ""}`} aria-hidden />
      <header className="topbar float">
        <Link href={`/canvas${query({ run: null, cycle: null })}`} className="logo press" aria-label="Atlas home">
          <span className="logo-mark">A</span>Atlas
        </Link>
        <nav className="nav" aria-label="Views">
          {NAV.map((n) => (
            <Link key={n.href} href={`${n.href}${query()}`} className={path.startsWith(n.href) ? "on" : ""} aria-current={path.startsWith(n.href) ? "page" : undefined}>
              <span>{n.label}</span>
            </Link>
          ))}
        </nav>
        <BranchPicker onPick={(b) => go(`${path}${b === "main" ? "" : `?branch=${encodeURIComponent(b)}`}`)} />
        <span className="grow" />
        <CyclePill />
      </header>
      <main className="main" data-pending={pending}>{children}</main>
    </>
  );
}

function BranchPicker({ onPick }: { onPick: (name: string) => void }) {
  const { branch, branches } = useSession();
  const { open, setOpen, root } = usePopover();
  return (
    <div className="field-wrap" ref={root}>
      <button type="button" data-trigger className="btn branch-btn" aria-haspopup="listbox" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span className="muted">branch</span>
        <span className="mono ellipsis">{branch.name}</span>
        {branch.sample && <span className="tag sample">sample</span>}
        <span aria-hidden className="faint">▾</span>
      </button>
      <div className={`pop branch-pop${open ? " open" : ""}`} role="listbox" aria-label="Branches" onKeyDown={listKeys}>
        {branches.map((b) => (
          <button key={b.name} type="button" role="option" aria-selected={b.name === branch.name} tabIndex={open ? 0 : -1}
            className={`opt${b.name === branch.name ? " on" : ""}`}
            onClick={() => { setOpen(false); if (b.name !== branch.name) onPick(b.name); }}>
            <span className="grow">
              <span className="row"><span className="mono">{b.name}</span>{b.sample && <span className="tag sample">sample</span>}</span>
              <small>
                {b.commit} · {b.spec === "none" ? "no spec" : b.extract === "pending" ? "RTL extract pending" : b.extract ? `RTL extract ${b.extract}` : "spec only, no RTL"}
              </small>
              <small>{b.note}</small>
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

function CyclePill() {
  const { run, runs, cycle, setCycle, setRun } = useSession();
  const { open, setOpen, root } = usePopover();
  const [draft, setDraft] = useState<string | null>(null);
  const playable = hasFile(run, "signals.json") || hasFile(run, "events.jsonl");
  const max = Math.max(0, (run?.cycles ?? 1) - 1);
  const invalid = draft !== null && (!/^\d+$/.test(draft.trim()) || Number(draft) > max);

  const commit = () => {
    if (draft === null) return;
    if (!invalid) { setCycle(Number(draft)); setDraft(null); }
  };

  return (
    <div className="pill" ref={root}>
      <div className="field-wrap">
        <button type="button" data-trigger className="btn ghost sm" aria-haspopup="listbox" aria-expanded={open} onClick={() => setOpen(!open)} disabled={!runs.length}>
          {run ? <><span className="muted">run</span> <b className="mono">#{run.id}</b></> : <span className="muted">no runs</span>}
          <span aria-hidden className="faint">▾</span>
        </button>
        <div className={`pop right run-pop${open ? " open" : ""}`} role="listbox" aria-label="Runs on this branch" onKeyDown={listKeys}>
          {runs.map((r) => (
            <button key={r.id} type="button" role="option" aria-selected={r.id === run?.id} tabIndex={open ? 0 : -1}
              className={`opt${r.id === run?.id ? " on" : ""}`} onClick={() => { setRun(r.id); setOpen(false); }}>
              <span className="mono">#{r.id}</span>
              <span className="grow">
                <span className="row"><span className="ellipsis">{r.test}</span><span className={`rs rs-${r.status}`}>{r.status}</span></span>
                <small className="muted">{when(r.startedAt)} · {r.files.includes("signals.json") ? "waveform" : "no waveform"}</small>
              </span>
            </button>
          ))}
        </div>
      </div>
      <button type="button" className="btn ghost sm icon" aria-label="Previous cycle" disabled={!playable || cycle <= 0} onClick={() => setCycle((c) => c - 1)}>‹</button>
      <div className="field-wrap">
        <input aria-label="Cycle" className={`field${invalid ? " invalid" : ""}`} disabled={!playable} inputMode="numeric"
          value={draft ?? String(cycle)} onChange={(e) => setDraft(e.target.value)} onBlur={() => { if (!invalid) commit(); else setDraft(null); }}
          onKeyDown={(e) => { if (e.key === "Enter") commit(); if (e.key === "Escape") setDraft(null); }} />
        {invalid && <span className="field-err">Cycle must be 0–{max}</span>}
      </div>
      <button type="button" className="btn ghost sm icon" aria-label="Next cycle" disabled={!playable || cycle >= max} onClick={() => setCycle((c) => c + 1)}>›</button>
      <span className="muted mono" style={{ paddingRight: 8 }}>/ {run ? max : "—"}</span>
    </div>
  );
}
