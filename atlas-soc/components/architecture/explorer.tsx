"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Icon } from "@/components/icons";
import type { ViewSnapshot } from "@/lib/model/view";
import { DiagramCanvas } from "./diagram-canvas";
import { Inspector } from "./inspector";
import { LensControls, SequenceStepDetail } from "./lens-controls";
import { OutlineTree } from "./outline-tree";
import { lensFromParams, lensToParams, type Lens } from "./lens";

type Props = {
  snapshot: ViewSnapshot;
  initialSelectedId: string | null;
  initialLens: Lens;
};

function ancestorsOf(snapshot: ViewSnapshot, blockId: string | null): string[] {
  const chain: string[] = [];
  let cursor = blockId ? snapshot.blocks[blockId]?.parent ?? null : null;
  const guard = new Set<string>();
  while (cursor && !guard.has(cursor)) {
    guard.add(cursor);
    chain.push(cursor);
    cursor = snapshot.blocks[cursor]?.parent ?? null;
  }
  return chain;
}

/**
 * The architecture explorer.
 *
 * Selection and focus are deliberately separate pieces of state. Selecting a
 * block always succeeds and only moves the inspector; focus — which authored
 * canvas is drawn — changes only when you enter a composite or arrive at a
 * block the current canvas does not contain. That split is what removes the
 * dead "Down" control from 81 of the 97 blocks.
 *
 * State is mirrored into the URL with history.pushState rather than a router
 * navigation, so back/forward and sharing work without remounting the canvas
 * and losing pan/zoom.
 */
export function Explorer({ snapshot, initialSelectedId, initialLens }: Props) {
  const [selectedId, setSelectedId] = useState<string | null>(initialSelectedId);
  const [lens, setLens] = useState<Lens>(initialLens);
  const [focusId, setFocusId] = useState<string>(
    initialSelectedId ? snapshot.blocks[initialSelectedId]?.focusId ?? snapshot.product.defaultFocus : snapshot.product.defaultFocus,
  );
  // Two booleans because the control means different things per breakpoint:
  // on desktop the outline is a column that can be collapsed for canvas room,
  // on narrow screens it is a drawer that starts closed.
  const [outlineOpen, setOutlineOpen] = useState(false);
  const [outlineCollapsed, setOutlineCollapsed] = useState(false);
  const [expanded, setExpanded] = useState<Set<string>>(
    () => new Set([...snapshot.rootIds, ...ancestorsOf(snapshot, initialSelectedId)]),
  );

  const toggleExpanded = useCallback((blockId: string) => {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(blockId)) next.delete(blockId);
      else next.add(blockId);
      return next;
    });
  }, []);

  /** Open the outline down to a block so a selection is never hidden. */
  const reveal = useCallback(
    (blockId: string) => {
      setExpanded((current) => {
        const chain = ancestorsOf(snapshot, blockId);
        if (chain.every((id) => current.has(id))) return current;
        return new Set([...current, ...chain]);
      });
    },
    [snapshot],
  );

  const pushUrl = useCallback((blockId: string | null, nextLens: Lens) => {
    if (typeof window === "undefined") return;
    const query = lensToParams(nextLens).toString();
    const url = `/architecture${blockId ? `/${blockId}` : ""}${query ? `?${query}` : ""}`;
    if (url === window.location.pathname + window.location.search) return;
    window.history.pushState({ blockId, lens: nextLens }, "", url);
  }, []);

  const select = useCallback(
    (blockId: string) => {
      const block = snapshot.blocks[blockId];
      if (!block) return;
      setSelectedId(blockId);
      // Only move the canvas when the current one cannot show the block.
      setFocusId((current) => (snapshot.layouts[current]?.nodes[blockId] ? current : block.focusId));
      reveal(blockId);
      setOutlineOpen(false);
      pushUrl(blockId, lens);
    },
    [snapshot, lens, pushUrl, reveal],
  );

  const enter = useCallback(
    (blockId: string) => {
      if (!snapshot.layouts[blockId]) return;
      setFocusId(blockId);
      setSelectedId(blockId);
      reveal(blockId);
      pushUrl(blockId, lens);
    },
    [snapshot.layouts, lens, pushUrl, reveal],
  );

  const trail = useMemo(() => {
    const chain: string[] = [];
    let cursor: string | null = focusId;
    const guard = new Set<string>();
    while (cursor && !guard.has(cursor)) {
      guard.add(cursor);
      chain.unshift(cursor);
      cursor = snapshot.blocks[cursor]?.parent ?? null;
    }
    return chain;
  }, [focusId, snapshot.blocks]);

  const parentFocus = useMemo(() => {
    let cursor = snapshot.blocks[focusId]?.parent ?? null;
    const guard = new Set<string>();
    while (cursor && !guard.has(cursor)) {
      guard.add(cursor);
      if (snapshot.layouts[cursor]) return cursor;
      cursor = snapshot.blocks[cursor]?.parent ?? null;
    }
    return null;
  }, [focusId, snapshot]);

  const leave = useCallback(() => {
    if (!parentFocus) return;
    setFocusId(parentFocus);
    setSelectedId(parentFocus);
    reveal(parentFocus);
    pushUrl(parentFocus, lens);
  }, [parentFocus, lens, pushUrl, reveal]);

  const changeLens = useCallback(
    (next: Lens) => {
      setLens(next);
      pushUrl(selectedId, next);
    },
    [selectedId, pushUrl],
  );

  // Back/forward restores both selection and lens from the URL itself, which
  // keeps the address bar authoritative rather than a decoration.
  useEffect(() => {
    const onPopState = () => {
      const path = window.location.pathname.replace(/^\/architecture\/?/, "");
      const blockId = path ? decodeURIComponent(path) : null;
      const nextLens = lensFromParams(new URLSearchParams(window.location.search));
      setLens(nextLens);
      setSelectedId(blockId && snapshot.blocks[blockId] ? blockId : null);
      if (blockId && snapshot.blocks[blockId]) {
        setFocusId((current) => (snapshot.layouts[current]?.nodes[blockId] ? current : snapshot.blocks[blockId].focusId));
        reveal(blockId);
      }
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, [snapshot, reveal]);

  // Global shortcuts: "/" jumps to the outline filter, Alt+Up leaves the canvas.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const typing = target?.matches("input, textarea, select, [contenteditable='true']");
      if (event.key === "/" && !typing) {
        event.preventDefault();
        setOutlineOpen(true);
        requestAnimationFrame(() => {
          document.querySelector<HTMLInputElement>(".outline-search input")?.focus();
        });
        return;
      }
      if (event.key === "ArrowUp" && event.altKey) {
        event.preventDefault();
        leave();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [leave]);

  return (
    <div className={`explorer${outlineOpen ? " outline-open" : ""}${outlineCollapsed ? " outline-collapsed" : ""}`}>
      <div className="explorer-bar">
        <button
          className="outline-toggle"
          type="button"
          aria-expanded={!outlineCollapsed || outlineOpen}
          aria-label={outlineCollapsed ? "Show the outline" : "Hide the outline"}
          onClick={() => {
            setOutlineOpen((open) => !open);
            setOutlineCollapsed((collapsed) => !collapsed);
          }}
        >
          <Icon name="lens" aria-hidden="true" />
          <span>Outline</span>
        </button>

        <nav className="trail" aria-label="Canvas location">
          {trail.map((id, index) => {
            const block = snapshot.blocks[id];
            if (!block) return null;
            const isLast = index === trail.length - 1;
            const drawable = Boolean(snapshot.layouts[id]);
            return (
              <span key={id}>
                {index > 0 && <Icon name="chevron" aria-hidden="true" />}
                {isLast || !drawable ? (
                  <b aria-current={isLast ? "true" : undefined}>{block.name}</b>
                ) : (
                  <button type="button" onClick={() => enter(id)}>
                    {block.name}
                  </button>
                )}
              </span>
            );
          })}
        </nav>

        <LensControls snapshot={snapshot} lens={lens} onLens={changeLens} />
      </div>

      {outlineOpen && <button className="outline-scrim" aria-label="Close outline" onClick={() => setOutlineOpen(false)} />}

      <div className="explorer-body">
        <OutlineTree
          snapshot={snapshot}
          selectedId={selectedId}
          expanded={expanded}
          onToggle={toggleExpanded}
          onSelect={select}
        />

        <div className="explorer-stage">
          <SequenceStepDetail snapshot={snapshot} lens={lens} />
          <DiagramCanvas
            snapshot={snapshot}
            focusId={focusId}
            selectedId={selectedId}
            lens={lens}
            onSelect={select}
            onEnter={enter}
            onLeave={leave}
            canLeave={Boolean(parentFocus)}
          />
          <p className="stage-hint">
            Drag to pan · ⌘/Ctrl+scroll to zoom · double-click a composite to open it ·{" "}
            <Link href="/wiki/editing-guide">how layouts are authored</Link>
          </p>
        </div>

        <Inspector snapshot={snapshot} selectedId={selectedId} lens={lens} onSelect={select} />
      </div>
    </div>
  );
}
