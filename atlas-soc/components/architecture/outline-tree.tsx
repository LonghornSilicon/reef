"use client";

import { useMemo, useRef, useState } from "react";
import { Icon } from "@/components/icons";
import type { ViewSnapshot } from "@/lib/model/view";
import { MATURITY_COLOR, MATURITY_LABEL } from "./lens";

type Props = {
  snapshot: ViewSnapshot;
  selectedId: string | null;
  /** Owned by the explorer so revealing a selection happens in the same
   *  handler that changes it, rather than in a cascading effect. */
  expanded: ReadonlySet<string>;
  onToggle(blockId: string): void;
  onSelect(blockId: string): void;
};

type Row = { id: string; depth: number; hasChildren: boolean; expanded: boolean };

/**
 * The complete model as a tree.
 *
 * This exists because the incumbent viewer made the diagram the only way in:
 * 81 of 97 blocks are leaves and 38 sit under a second root, so most of the
 * model was reachable only by search. Every block is listed here.
 */
export function OutlineTree({ snapshot, selectedId, expanded, onToggle, onSelect }: Props) {
  const [query, setQuery] = useState("");
  const listRef = useRef<HTMLDivElement | null>(null);

  const matches = useMemo(() => {
    const term = query.trim().toLowerCase();
    if (!term) return null;
    const hits = new Set<string>();
    for (const block of Object.values(snapshot.blocks)) {
      const haystack = `${block.name} ${block.id} ${block.kind} ${block.summary}`.toLowerCase();
      if (!haystack.includes(term)) continue;
      hits.add(block.id);
      // Keep ancestors so a match never appears detached from its parent.
      let cursor = block.parent;
      const guard = new Set<string>();
      while (cursor && !guard.has(cursor)) {
        guard.add(cursor);
        hits.add(cursor);
        cursor = snapshot.blocks[cursor]?.parent ?? null;
      }
    }
    return hits;
  }, [query, snapshot.blocks]);

  const rows = useMemo(() => {
    const result: Row[] = [];
    const walk = (id: string, depth: number) => {
      const block = snapshot.blocks[id];
      if (!block) return;
      if (matches && !matches.has(id)) return;
      const childIds = matches ? block.childIds.filter((childId) => matches.has(childId)) : block.childIds;
      const isExpanded = matches ? true : expanded.has(id);
      result.push({ id, depth, hasChildren: childIds.length > 0, expanded: isExpanded });
      if (!isExpanded) return;
      for (const childId of childIds) walk(childId, depth + 1);
    };
    for (const rootId of snapshot.rootIds) walk(rootId, 0);
    return result;
  }, [snapshot, expanded, matches]);

  const focusRow = (id: string) => {
    requestAnimationFrame(() => {
      listRef.current?.querySelector<HTMLElement>(`[data-tree-row="${CSS.escape(id)}"]`)?.focus();
    });
  };

  const onKeyDown = (event: React.KeyboardEvent, row: Row, index: number) => {
    const move = (target: Row | undefined) => {
      if (!target) return;
      event.preventDefault();
      focusRow(target.id);
    };
    switch (event.key) {
      case "ArrowDown":
        move(rows[index + 1]);
        break;
      case "ArrowUp":
        move(rows[index - 1]);
        break;
      case "Home":
        move(rows[0]);
        break;
      case "End":
        move(rows[rows.length - 1]);
        break;
      case "ArrowRight":
        if (row.hasChildren && !row.expanded) {
          event.preventDefault();
          onToggle(row.id);
        } else move(rows[index + 1]);
        break;
      case "ArrowLeft":
        if (row.hasChildren && row.expanded) {
          event.preventDefault();
          onToggle(row.id);
        } else {
          const parent = snapshot.blocks[row.id]?.parent;
          if (parent) move(rows.find((candidate) => candidate.id === parent));
        }
        break;
      case "Enter":
      case " ":
        event.preventDefault();
        onSelect(row.id);
        break;
      default:
        break;
    }
  };

  return (
    <div className="outline-rail">
      <div className="outline-head">
        <label className="outline-search">
          <Icon name="search" aria-hidden="true" />
          <input
            type="search"
            value={query}
            placeholder="Filter blocks"
            aria-label="Filter architecture blocks"
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        <p className="outline-count">
          {matches ? `${rows.length} matching` : `${snapshot.counts.blocks} blocks`} · {snapshot.rootIds.length} roots
        </p>
      </div>

      <div className="outline-list" role="tree" aria-label="Architecture outline" ref={listRef}>
        {rows.length === 0 && <p className="outline-empty">No block matches that filter.</p>}
        {rows.map((row, index) => {
          const block = snapshot.blocks[row.id];
          const selected = row.id === selectedId;
          return (
            <div
              key={row.id}
              className={`outline-row${selected ? " selected" : ""}`}
              style={{ paddingLeft: 8 + row.depth * 13 }}
              role="treeitem"
              aria-level={row.depth + 1}
              aria-selected={selected}
              aria-expanded={row.hasChildren ? row.expanded : undefined}
              data-tree-row={row.id}
              tabIndex={selected || (!selectedId && index === 0) ? 0 : -1}
              onKeyDown={(event) => onKeyDown(event, row, index)}
              onClick={() => onSelect(row.id)}
            >
              {row.hasChildren ? (
                <button
                  className={`outline-twist${row.expanded ? " open" : ""}`}
                  type="button"
                  tabIndex={-1}
                  aria-label={`${row.expanded ? "Collapse" : "Expand"} ${block.name}`}
                  onClick={(event) => {
                    event.stopPropagation();
                    onToggle(row.id);
                  }}
                >
                  <Icon name="chevron" />
                </button>
              ) : (
                <span className="outline-twist placeholder" aria-hidden="true" />
              )}
              <span
                className="outline-maturity"
                style={{ background: MATURITY_COLOR[block.maturity] }}
                title={MATURITY_LABEL[block.maturity]}
                aria-hidden="true"
              />
              <span className="outline-name" title={block.name}>
                {block.name}
              </span>
              {row.hasChildren && <span className="outline-kids">{block.childIds.length}</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
